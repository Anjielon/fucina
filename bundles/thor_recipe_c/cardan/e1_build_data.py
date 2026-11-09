#!/usr/bin/env python3
"""e1_build_data.py — costruisce il set di addestramento "letter-KD" SENZA dati MMLU.

Fonti ammesse (revisioni fissate, sha di commit verificati il 3/10/2026):
  obbligatorie: SciQ, CommonsenseQA (ridotta a 4 opzioni: oro + 3 distrattori, seme fisso)
  opzionali   : QASC, MedMCQA, MathQA, LogiQA
VIETATE (sono dentro MMLU auxiliary_train): ARC, OpenBookQA, RACE, MCTest — non compaiono
qui e la decontaminazione contro TUTTI gli split di cais/mmlu (auxiliary_train compreso)
scarta comunque ogni sovrapposizione di 13-grammi.

Uscita: un jsonl con UNA riga per rotazione (4 per domanda), oro uniforme su A/B/C/D:
  {id, group_id, source, split, subject, rot, perm, gold_pos, options, gold, prompt}
  - options/gold = ordine CANONICO del contenuto; perm[p] = contenuto alla posizione p
  - split = "train" (dallo split train della fonte) oppure "heldout" (validation/test)

Esempio:
  python3 e1_build_data.py --out data/items.jsonl --sources sciq,csqa,qasc,logiqa \
      --cache ~/.cache/mmlu_free --max-per-source 20000
"""
from __future__ import annotations

import argparse
import ast
import json
import logging
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from common import (hf_file, ngram_hashes, norm_text, norm_words, read_parquet_rows,
                    render_prompt, rotate, stable_int, write_jsonl)

logger = logging.getLogger("e1_build")

# ── Revisioni fissate ────────────────────────────────────────────────────────
# (repo, sha di commit, {split: percorso parquet}) — sha letti da /api/datasets il 3/10/2026.
# MathQA e LogiQA sono dataset "a script": si usa il ramo refs/convert/parquet, fissato al suo sha.
SOURCES: dict[str, tuple[str, str, dict[str, str]]] = {
    "sciq": ("allenai/sciq", "2c94ad3e1aafab77146f384e23536f97a4849815", {
        "train": "data/train-00000-of-00001.parquet",
        "validation": "data/validation-00000-of-00001.parquet",
        "test": "data/test-00000-of-00001.parquet"}),
    "csqa": ("tau/commonsense_qa", "94630fe30dad47192a8546eb75f094926d47e155", {
        "train": "data/train-00000-of-00001.parquet",
        "validation": "data/validation-00000-of-00001.parquet"}),   # test senza risposte
    "qasc": ("allenai/qasc", "a34ba204eb9a33b919c10cc08f4f1c8dae5ec070", {
        "train": "data/train-00000-of-00001.parquet",
        "validation": "data/validation-00000-of-00001.parquet"}),   # test senza risposte
    "medmcqa": ("openlifescienceai/medmcqa", "91c6572c454088bf71b679ad90aa8dffcd0d5868", {
        "train": "data/train-00000-of-00001.parquet",
        "validation": "data/validation-00000-of-00001.parquet"}),   # test senza risposte
    "mathqa": ("allenai/math_qa", "fafb9f7ee5b9ec4da9499f9c4177a4c91389f2d6", {
        "train": "default/train/0000.parquet",
        "validation": "default/validation/0000.parquet",
        "test": "default/test/0000.parquet"}),
    "logiqa": ("lucasmccabe/logiqa", "fa9f9918fa81eca088805c1395d7f592f7755ae0", {
        "train": "default/train/0000.parquet",
        "validation": "default/validation/0000.parquet",
        "test": "default/test/0000.parquet"}),
}
MMLU_REPO = ("cais/mmlu", "c30699e8356da336a370243923dbaf21066bb9fe", {
    "auxiliary_train": "all/auxiliary_train-00000-of-00001.parquet",
    "dev": "all/dev-00000-of-00001.parquet",
    "validation": "all/validation-00000-of-00001.parquet",
    "test": "all/test-00000-of-00001.parquet"})

DEFAULT_SUBJECT = {"sciq": "science", "csqa": "common sense", "qasc": "science",
                   "medmcqa": "medicine", "mathqa": "mathematics", "logiqa": "logic"}

# Un "item" grezzo: (domanda, lista opzioni, indice oro, soggetto opzionale, testo extra per decontam)
RawItem = tuple[str, list[str], int, Optional[str], str]


# ── Parser per fonte ─────────────────────────────────────────────────────────
def _as_dict(x) -> dict:
    """Le colonne struct arrivano come dict; per prudenza accetta anche la stringa repr."""
    return x if isinstance(x, dict) else ast.literal_eval(str(x))


def parse_sciq(row: dict) -> Optional[RawItem]:
    opts = [row["correct_answer"], row["distractor1"], row["distractor2"], row["distractor3"]]
    if any(not str(o).strip() for o in opts):
        return None
    # il supporto NON entra nel prompt, ma entra nella decontaminazione
    return row["question"], [str(o) for o in opts], 0, None, row.get("support") or ""


def parse_csqa(row: dict) -> Optional[RawItem]:
    ch = _as_dict(row["choices"])
    labels, texts = list(ch["label"]), list(ch["text"])
    key = (row.get("answerKey") or "").strip()
    if key not in labels:
        return None
    return row["question"], texts, labels.index(key), None, ""


def parse_qasc(row: dict) -> Optional[RawItem]:
    ch = _as_dict(row["choices"])
    labels, texts = list(ch["label"]), list(ch["text"])
    key = (row.get("answerKey") or "").strip()
    if key not in labels:
        return None
    extra = " ".join(str(row.get(k) or "") for k in ("fact1", "fact2", "combinedfact"))
    return row["question"], texts, labels.index(key), None, extra


def parse_medmcqa(row: dict) -> Optional[RawItem]:
    try:
        cop = int(row["cop"])
    except (TypeError, ValueError):
        return None
    if cop not in (0, 1, 2, 3):
        return None
    opts = [row["opa"], row["opb"], row["opc"], row["opd"]]
    if any(o is None or not str(o).strip() for o in opts):
        return None
    subj = str(row.get("subject_name") or "medicine").lower()
    return row["question"], [str(o) for o in opts], cop, subj, str(row.get("exp") or "")


_MQ_OPT = re.compile(r"([a-e])\s*\)\s*(.*?)\s*(?=,\s*[a-e]\s*\)|$)")


def parse_mathqa(row: dict) -> Optional[RawItem]:
    found = _MQ_OPT.findall(str(row["options"]).strip())
    if len(found) != 5:
        return None
    letters = [f[0] for f in found]
    texts = [f[1].strip() for f in found]
    key = str(row["correct"]).strip().lower()
    if key not in letters:
        return None
    return row["Problem"], texts, letters.index(key), None, str(row.get("Rationale") or "")


def parse_logiqa(row: dict) -> Optional[RawItem]:
    opts = list(row["options"])
    g = int(row["correct_option"])
    if len(opts) != 4 or not 0 <= g < 4:
        return None
    q = str(row["context"]).strip() + "\n" + str(row["query"]).strip()
    return q, [str(o) for o in opts], g, None, ""


PARSERS: dict[str, Callable[[dict], Optional[RawItem]]] = {
    "sciq": parse_sciq, "csqa": parse_csqa, "qasc": parse_qasc,
    "medmcqa": parse_medmcqa, "mathqa": parse_mathqa, "logiqa": parse_logiqa}


# ── Riduzione a 4 opzioni e ordine canonico ──────────────────────────────────
def to_four(options: list[str], gold: int, key: str, seed: int) -> tuple[list[str], int]:
    """Tiene l'oro + 3 distrattori estratti a caso; poi mescola l'ordine canonico.

    Il generatore e' seminato da (seed, chiave dell'item): il risultato non dipende
    dall'ordine di lettura ne' dal processo. Distrattori identici all'oro (testo
    normalizzato) vengono esclusi prima dell'estrazione.
    """
    rng = random.Random(seed ^ stable_int(key))
    g_txt = norm_text(options[gold])
    distr = [i for i in range(len(options)) if i != gold and norm_text(options[i]) != g_txt]
    # distrattori duplicati fra loro: ne resta uno solo
    seen, uniq = set(), []
    for i in distr:
        t = norm_text(options[i])
        if t not in seen:
            seen.add(t)
            uniq.append(i)
    if len(uniq) < 3:
        raise ValueError("meno di 3 distrattori distinti")
    keep = [gold] + rng.sample(uniq, 3)
    rng.shuffle(keep)                       # l'oro non resta sempre in testa
    return [options[i] for i in keep], keep.index(gold)


# ── Decontaminazione ─────────────────────────────────────────────────────────
class MMLUIndex:
    """Insieme ordinato degli hash dei 13-grammi di TUTTO cais/mmlu + domande normalizzate."""

    def __init__(self, cache: Path, n: int = 13):
        self.n = n
        npy = cache / f"mmlu_{MMLU_REPO[1][:12]}_{n}gram.npy"
        qtxt = cache / f"mmlu_{MMLU_REPO[1][:12]}_questions.json"
        if npy.exists() and qtxt.exists():
            self.hashes = np.load(npy)
            self.questions = set(json.loads(qtxt.read_text()))
            logger.info("indice MMLU dalla cache: %d 13-grammi", len(self.hashes))
            return
        repo, rev, files = MMLU_REPO
        chunks, questions = [], set()
        for split, path in files.items():
            rows = read_parquet_rows(hf_file(repo, rev, path, cache))
            logger.info("MMLU %s: %d righe", split, len(rows))
            for r in rows:
                if isinstance(r.get("train"), dict):        # vecchio schema annidato
                    r = r["train"]
                text = str(r["question"]) + " " + " ".join(str(c) for c in r["choices"])
                chunks.append(ngram_hashes(norm_words(text), n))
                questions.add(norm_text(str(r["question"])))
        self.hashes = np.unique(np.concatenate(chunks))
        self.questions = questions
        cache.mkdir(parents=True, exist_ok=True)
        np.save(npy, self.hashes)
        qtxt.write_text(json.dumps(sorted(questions)))
        logger.info("indice MMLU costruito: %d 13-grammi unici", len(self.hashes))

    def contaminated(self, text: str, question: str) -> bool:
        """Vero se UN 13-gramma coincide, o se la domanda normalizzata e' identica a una MMLU
        (serve per le domande corte, sotto i 13 token, dove gli n-grammi non esistono)."""
        if norm_text(question) in self.questions:
            return True
        h = ngram_hashes(norm_words(text), self.n)
        if h.size == 0:
            return False
        pos = np.searchsorted(self.hashes, h)
        pos = np.minimum(pos, len(self.hashes) - 1)
        return bool(np.any(self.hashes[pos] == h))


# ── Costruzione ──────────────────────────────────────────────────────────────
def load_source(name: str, cache: Path, seed: int, max_items: int,
                max_heldout: int) -> list[dict]:
    repo, rev, files = SOURCES[name]
    parser = PARSERS[name]
    out = []
    for split, path in files.items():
        rows = read_parquet_rows(hf_file(repo, rev, path, cache))
        kept = 0
        for i, row in enumerate(rows):
            item = parser(row)
            if item is None:
                continue
            q, opts, g, subj, extra = item
            key = f"{name}/{split}/{i}"
            try:
                opts4, g4 = to_four(opts, g, key, seed) if len(opts) != 4 else \
                    _shuffle4(opts, g, key, seed)
            except ValueError:
                continue
            out.append({"key": key, "source": name,
                        "split": "train" if split == "train" else "heldout",
                        "subject": subj or DEFAULT_SUBJECT[name], "question": q,
                        "options": opts4, "gold": g4, "extra": extra})
            kept += 1
        logger.info("%s/%s: %d item validi su %d", name, split, kept, len(rows))
    # i tetti si applicano separatamente a train e heldout, con estrazione seminata
    rng = random.Random(seed ^ stable_int(name))
    capped = []
    for sp, cap in (("train", max_items), ("heldout", max_heldout)):
        part = [o for o in out if o["split"] == sp]
        if cap and len(part) > cap:
            rng.shuffle(part)
            part = part[:cap]
        capped.extend(part)
    return capped


def _shuffle4(opts: list[str], g: int, key: str, seed: int) -> tuple[list[str], int]:
    """Fonti gia' a 4 opzioni: controlla i duplicati e mescola l'ordine canonico."""
    if len({norm_text(o) for o in opts}) < 4:
        raise ValueError("opzioni duplicate")
    rng = random.Random(seed ^ stable_int(key))
    idx = list(range(4))
    rng.shuffle(idx)
    return [opts[i] for i in idx], idx.index(g)


def build_fewshot(pool: list[dict], item: dict, k: int, seed: int) -> list[tuple]:
    """k esempi dalla STESSA fonte (split train), mai l'item stesso; le lettere d'oro
    degli esempi ciclano A,B,C,D cosi' il few-shot non insegna una lettera preferita."""
    if k <= 0:
        return []
    rng = random.Random(seed ^ stable_int("fs/" + item["key"]))
    cands = [p for p in pool if p["key"] != item["key"]]
    shots = rng.sample(cands, min(k, len(cands)))
    out = []
    for j, s in enumerate(shots):
        target = j % 4
        r = (s["gold"] - target) % 4          # rotazione che porta l'oro sulla lettera target
        opts, gp, _ = rotate(s["options"], s["gold"], r)
        out.append((s["question"], opts, gp))
    return out


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sources", default="sciq,csqa",
                    help="sottoinsieme di: " + ",".join(SOURCES))
    ap.add_argument("--cache", default=str(Path.home() / ".cache" / "mmlu_free"))
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max-per-source", type=int, default=0,
                    help="tetto item TRAIN per fonte (0 = nessun limite)")
    ap.add_argument("--max-heldout-per-source", type=int, default=2000,
                    help="tetto item HELDOUT per fonte (0 = nessun limite)")
    ap.add_argument("--fewshot", type=int, default=0, help="esempi few-shot dalla stessa fonte")
    ap.add_argument("--ngram", type=int, default=13)
    ap.add_argument("--no-decontam", action="store_true", help="SOLO per debug")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cache = Path(args.cache).expanduser()
    names = [s.strip() for s in args.sources.split(",") if s.strip()]
    for n in names:
        if n not in SOURCES:
            ap.error(f"fonte sconosciuta o vietata: {n}")
    index = None if args.no_decontam else MMLUIndex(cache, args.ngram)

    items, dropped, dup = [], Counter(), Counter()
    seen = set()
    for name in names:
        for it in load_source(name, cache, args.seed, args.max_per_source,
                              args.max_heldout_per_source):
            sig = norm_text(it["question"]) + "||" + norm_text(it["options"][it["gold"]])
            if sig in seen:                        # duplicati interni (stessa domanda+oro)
                dup[name] += 1
                continue
            seen.add(sig)
            text = " ".join([it["question"], *it["options"], it["extra"]])
            if index is not None and index.contaminated(text, it["question"]):
                dropped[name] += 1
                continue
            items.append(it)

    pools = {}
    for it in items:
        if it["split"] == "train":
            pools.setdefault(it["source"], []).append(it)

    rows, gold_hist = [], Counter()
    for it in items:
        gid = it["key"]
        shots = build_fewshot(pools.get(it["source"], []), it, args.fewshot, args.seed)
        for r in range(4):
            opts, gp, perm = rotate(it["options"], it["gold"], r)
            gold_hist[gp] += 1
            rows.append({"id": f"{gid}#r{r}", "group_id": gid, "source": it["source"],
                         "split": it["split"], "subject": it["subject"], "rot": r,
                         "perm": perm, "gold_pos": gp, "options": it["options"],
                         "gold": it["gold"],
                         "prompt": render_prompt(it["subject"], it["question"], opts, shots)})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n = write_jsonl(args.out, rows)
    per_src = Counter((it["source"], it["split"]) for it in items)
    logger.info("scritte %d righe (%d domande x 4 rotazioni) in %s", n, len(items), args.out)
    logger.info("per fonte/split: %s", dict(per_src))
    logger.info("scartate per contaminazione MMLU: %s", dict(dropped))
    logger.info("duplicati interni scartati: %s", dict(dup))
    logger.info("istogramma posizione oro (deve essere uniforme): %s",
                {"ABCD"[k]: v for k, v in sorted(gold_hist.items())})
    return 0


if __name__ == "__main__":
    sys.exit(main())
