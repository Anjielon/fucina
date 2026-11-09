#!/usr/bin/env python3
"""Corpus di calibrazione v2 CORRETTO per contaminazione SEMANTICA — file NUOVI `*_v2_sem.pt` / `*_v2_senza_llama_sem.pt`.
25/9/2026 notte, dalla verifica avversaria del corpus v2 (script e prove in scratchpad/corpus_v2/verifica/).

## Perche'
Il costruttore v2 (`costruisci_corpus_calibrazione_v2.py`, NON modificato: lo si importa) garantisce 0 finestre ESATTE
di 32 token e 0 prompt normalizzati IDENTICI in comune col pannello e coi benchmark. Tre casi passano sotto quelle
due maglie (misurati sui file v2, prove in `verifica/g_cot_semantica`, `k_quasi_doppioni`, `l_verdetto_parole`):

1. **Domanda MMLU intera ma corta** (< 32 token): nella variante senza_llama la domanda cot «What is the remainder when
   $2^{87} +3$ is divided by $7$?» (MATH train, number_theory) e' la domanda MMLU test #4240, con la risposta di Thor.
   72 domande MMLU (>= 8 parole) stanno intere in GSM8K/MATH train, il serbatoio da cui pesca la cot.
2. **Stesso problema in un'altra forma MetaMathQA**: nel pannello math_code c'e' la forma SV di «Chris bought 8 movies
   on DVD for $12 each and 4 movies on Blu-ray for $x each...», e la calibrazione senza_llama ha l'originale AnsAug
   (con $18, cioe' la risposta). Le finestre esatte non lo vedono: le parole cambiano.
3. **Quasi-doppioni Aya fra i due lati della partizione per hash**: la partizione separa i prompt IDENTICI, non quelli
   che differiscono di una parola. Nel pannello it_chat «L'Australia e' il secondo Paese del mondo per estensione. Vero o
   falso?» e in calibrazione «... il sesto / il terzo ...» (con la risposta); «Chi ha dipinto La Gioconda?» due volte;
   Achemenide/Mongolo; Severus Piton; lo sbarco sulla luna. Entrambe le varianti.

## Regole aggiunte (strutturali, non caso per caso) — un esempio di calibrazione e' tolto se:
- R1 MMLU: un suo testo normalizzato contiene INTERA una domanda MMLU test di almeno 8 parole (qualsiasi dominio);
- R2 cot: il suo problema ORIGINALE (campo `original_question` di MetaMathQA@aa4f34d3, primi 60.000.001 byte come il
  generatore) coincide con quello di un esempio dell'held-out (0-8) o del pannello math_code (279-385), o con una domanda
  MMLU; per il codice (nessun originale) vale l'uguaglianza della domanda normalizzata;
- R3 quasi-doppioni: un suo turno utente ha, con un turno utente del pannello v2 (it_chat, it_pensiero, strumenti_it,
  strumenti_en), rapporto difflib >= 0.80 sul testo normalizzato, massimo dei due versi perche' difflib non e'
  simmetrico (prefiltro: Jaccard dei trigrammi di parole >= 0.40, almeno 5 parole). Conservativa: toglie anche i template ricorrenti di Aya («Fammi un esempio di curiosita' in questa
  categoria: ...») con risposte diverse — costo misurato nel manifesto, nessun esempio pulito da salvare a mano.
Le regole si innestano nel ciclo di pulizia del v2 (fino a 3 giri): gli esempi tolti sono rimpiazzati dai successivi e
ricontrollati. Tutto il resto (fonti, quote, template, seme, verifiche v2) e' IDENTICO al v2.

## Cosa NON tocca
Non scrive calibrazione_corpus*.pt esistenti, pannello_nuovi_domini*.pt (il meta del pannello v2 NON viene aggiornato:
`aggiorna_meta_pannello` e' spento qui), manifesti delle sorgenti, LEGGIMI.md. Scrive SOLO:
`calibrazione/calibrazione_corpus_v2_sem.{pt,json}`, `calibrazione/calibrazione_corpus_v2_senza_llama_sem.{pt,json}`,
`calibrazione/sorgenti/metamathqa_originali_cot.jsonl` (+ `.provenienza.json`), `calibrazione/LEGGIMI_v2_sem.md`.

    env CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=4 RAYON_NUM_THREADS=4 \\
        nice -n 19 @@ROOT:venv@@/bin/python3 costruisci_corpus_calibrazione_v2_sem.py
Uso nella forgia: come il v2 (`THOR_VALIDA_BATCH_SEQ=8 THOR_VALIDA_N_HELDOUT=1`, miscele U/V/W; M2 solo «tutte»).
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("RAYON_NUM_THREADS", "4")

import difflib  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import unicodedata  # noqa: E402
import urllib.request  # noqa: E402
from collections import defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import costruisci_corpus_calibrazione_v2 as V2  # noqa: E402  (v2, solo import)

C = V2.C
DIR_CAL, SORG = V2.DIR_CAL, V2.SORG
OUT_PT = {"tutte": DIR_CAL / "calibrazione_corpus_v2_sem.pt",
          "senza_llama": DIR_CAL / "calibrazione_corpus_v2_senza_llama_sem.pt"}
ORIGINALI = SORG / "metamathqa_originali_cot.jsonl"
MMQA_REV = "aa4f34d3d2d3231299b5b03d9b3e5a20da45aa18"
MMQA_URL = f"https://huggingface.co/datasets/meta-math/MetaMathQA/resolve/{MMQA_REV}/MetaMathQA-395K.json"
MMQA_BYTE = 60_000_000                      # Range 0-60000000 = 60.000.001 byte, come genera_cot_calibrazione.py
SOGLIA_RAPPORTO, SOGLIA_JACCARD, MIN_PAROLE_QUASI, MIN_PAROLE_MMLU = 0.80, 0.40, 5, 8
RX_UTENTE = re.compile(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", re.S)
REGISTRO: dict = {}


def parole(t: str) -> list:
    t = unicodedata.normalize("NFKC", t or "").lower()
    return re.sub(r"[^\w]+", " ", t).split()


def trigrammi(w: list) -> set:
    return {tuple(w[i:i + 3]) for i in range(max(1, len(w) - 2))}


def utenti(testo: str) -> list:
    return [u for u in RX_UTENTE.findall(testo) if not u.lstrip().startswith("<tool_response>")]


def domande_mmlu(p: Path) -> list:
    """Le SOLE domande (senza risposte) del file di `llama-perplexity --multiple-choice` (formato di leggi_mmlu v1)."""
    import struct
    d = p.read_bytes()
    n = struct.unpack_from("<I", d, 0)[0]
    out = []
    for o in struct.unpack_from(f"<{n}I", d, 4):
        ln = struct.unpack_from("<I", d, o)[0]
        out.append(d[o + 4:o + 4 + ln].decode("utf-8", errors="replace"))
    return out


# ── R2: problemi originali di MetaMathQA (estratto nuovo, con provenienza) ──
def originali_metamath() -> dict:
    """{sha256(domanda cot math): [original_question, ...]} per le domande di cot_math.jsonl."""
    prov = ORIGINALI.with_suffix(".provenienza.json")
    if not ORIGINALI.exists():
        V2.log(f"scarico MetaMathQA@{MMQA_REV[:8]} byte 0-{MMQA_BYTE} (come il generatore)")
        raw = urllib.request.urlopen(urllib.request.Request(MMQA_URL, headers={"Range": f"bytes=0-{MMQA_BYTE}"}),
                                     timeout=600).read()
        txt = raw.decode("utf-8", "ignore")
        orig = defaultdict(set)
        for blocco in txt.split('{"query"')[1:]:
            try:
                v = json.loads('{"query"' + blocco.rsplit("}", 1)[0] + "}")
            except json.JSONDecodeError:
                continue
            orig[v["query"]].add((v.get("original_question") or "", v.get("type") or ""))
        righe, trovate = [], 0
        for riga in (C.DIR_COT / "cot_math.jsonl").open(encoding="utf-8"):
            try:
                q = json.loads(riga)["q"]
            except (json.JSONDecodeError, KeyError):
                continue
            o = sorted(orig.get(q, ()))
            trovate += bool(o)
            righe.append({"sha256_q": hashlib.sha256(q.encode("utf-8")).hexdigest(),
                          "original_question": sorted({x for x, _ in o}), "tipi": sorted({t for _, t in o})})
        ORIGINALI.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in righe), encoding="utf-8")
        prov.write_text(json.dumps({"repo": "meta-math/MetaMathQA", "revisione": MMQA_REV, "file": "MetaMathQA-395K.json",
                                    "byte_letti": len(raw), "sha256_byte_letti": hashlib.sha256(raw).hexdigest(),
                                    "voci_nel_prefisso": sum(len(v) for v in orig.values()),
                                    "domande_cot_math": len(righe), "ritrovate": trovate,
                                    "sha256_scritto": C.sha256_file(ORIGINALI),
                                    "quando": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1), encoding="utf-8")
    p = json.loads(prov.read_text())
    if C.sha256_file(ORIGINALI) != p["sha256_scritto"]:
        raise RuntimeError(f"{ORIGINALI.name} non combacia con la sua provenienza")
    if p["ritrovate"] != p["domande_cot_math"]:
        raise RuntimeError(f"MetaMathQA: ritrovate {p['ritrovate']} domande su {p['domande_cot_math']}")
    return {r["sha256_q"]: r["original_question"] for r in C.jsonl(ORIGINALI)}


class Semantica:
    """Le tre regole, con i riferimenti (MMLU, held-out/pannello cot, turni utente del pannello) calcolati una volta."""

    def __init__(self, tk):
        self.tk = tk
        self.orig = originali_metamath()
        self.mmlu = {}
        for q in domande_mmlu(C.FILE_MMLU):
            w = parole(q)
            if len(w) >= MIN_PAROLE_MMLU:
                self.mmlu.setdefault(" ".join(w[:8]), set()).add(" ".join(w))
        self.mmlu_intere = {x for v in self.mmlu.values() for x in v}
        riservati = {i for r in C.COT_ESCLUSI for i in r}
        self.cot_riservati = set()
        for e in C.esempi_cot_tutti(tk):
            if e.extra["indice_mescolato"] in riservati:
                self.cot_riservati |= self.chiavi_cot(e)
        self.pan = None

    def chiavi_cot(self, e) -> set:
        """Problema originale (math) o domanda stessa, normalizzati."""
        k = {" ".join(parole(e.prompt))}
        if e.fonte == "cot_math":
            k |= {" ".join(parole(o)) for o in self.orig.get(hashlib.sha256(e.prompt.encode("utf-8")).hexdigest(), [])}
        return {x for x in k if x}

    def pannello(self, pannello: dict):
        if self.pan is None:
            turni, indice = [], defaultdict(set)
            for d in ("it_chat", "it_pensiero", "strumenti_it", "strumenti_en"):
                for u in utenti(self.tk.decodifica([x for s in pannello[d] for x in s])):
                    w = parole(u)
                    if len(w) < MIN_PAROLE_QUASI:
                        continue
                    t = trigrammi(w)
                    k = len(turni)
                    turni.append((d, u, " ".join(w), t))
                    for g in t:
                        indice[g].add(k)
            self.pan = (turni, indice)
        return self.pan

    def mmlu_in(self, testo: str) -> "str | None":
        w = parole(testo)
        s = " " + " ".join(w) + " "
        for i in range(len(w) - 7):
            for q in self.mmlu.get(" ".join(w[i:i + 8]), ()):
                if " " + q + " " in s:
                    return q
        return None

    def quasi_doppione(self, u: str, pannello: dict) -> "tuple | None":
        turni, indice = self.pannello(pannello)
        w = parole(u)
        if len(w) < MIN_PAROLE_QUASI:
            return None
        t, n = trigrammi(w), " ".join(w)
        cand = defaultdict(int)
        for g in t:
            for k in indice.get(g, ()):
                cand[k] += 1
        for k, inter in cand.items():
            d, up, np_, tp = turni[k]
            if inter / len(t | tp) >= SOGLIA_JACCARD:
                # difflib non e' simmetrico (euristica autojunk): si prende il massimo dei due versi
                r = max(difflib.SequenceMatcher(None, n, np_).ratio(), difflib.SequenceMatcher(None, np_, n).ratio())
                if r >= SOGLIA_RAPPORTO:
                    return d, up, round(r, 3)
        return None

    def controlla(self, corpus: dict, usati_dom: dict, pannello: dict) -> "tuple[dict, dict]":
        extra, dett = defaultdict(set), {"R1_mmlu": [], "R2_cot": [], "R3_quasi_doppioni": []}
        wq = self.mmlu_in(self.tk.decodifica([x for s in corpus["wiki"] for x in s]))
        if wq:
            raise RuntimeError(f"wiki contiene la domanda MMLU «{wq}»: il pool wiki non si puo' filtrare per esempio")
        for d, us in usati_dom.items():
            for e in us:
                for c in e.contenuti:
                    q = self.mmlu_in(c)
                    if q:
                        extra[d].add(e.id)
                        dett["R1_mmlu"].append([d, e.id, q[:160]])
                        break
        for e in usati_dom["cot"]:
            k = self.chiavi_cot(e) & (self.cot_riservati | self.mmlu_intere)
            if k:
                extra["cot"].add(e.id)
                dett["R2_cot"].append([e.id, e.prompt[:160], next(iter(k))[:160]])
        for d in ("it_chat", "it_pensiero", "strumenti"):
            for e in usati_dom.get(d, []):
                for u in utenti(e.testo):
                    q = self.quasi_doppione(u, pannello)
                    if q:
                        extra[d].add(e.id)
                        dett["R3_quasi_doppioni"].append([d, e.id, u[:160], q[0], q[1][:160], q[2]])
                        break
        return {k: v for k, v in extra.items() if v}, dett


SEM: "Semantica | None" = None
_VERIFICA_V2 = V2.verifica_v2


def verifica_sem(tk, variante, corpus, usati_dom, pannello, usati_pan, fonti_cal, fonti_pan):
    """La verifica v2 intera, poi le regole R1-R3: i loro esempi entrano nei `contaminati` del ciclo di pulizia."""
    global SEM
    ver, contaminati = _VERIFICA_V2(tk, variante, corpus, usati_dom, pannello, usati_pan, fonti_cal, fonti_pan)
    if SEM is None:
        SEM = Semantica(tk)
    extra, dett = SEM.controlla(corpus, usati_dom, pannello)
    for d, ids in extra.items():
        contaminati.setdefault(d, set()).update(ids)
    ver["semantica"] = {"regole": {"R1_mmlu_min_parole": MIN_PAROLE_MMLU, "R3_rapporto_min": SOGLIA_RAPPORTO,
                                   "R3_jaccard_trigrammi_min": SOGLIA_JACCARD, "R3_min_parole": MIN_PAROLE_QUASI},
                        "tolti_in_questo_giro": {d: len(v) for d, v in extra.items()}, "dettaglio": dett}
    if extra:
        ver["esito"] = f"⛔ semantica {sorted(extra)}"
    REGISTRO.setdefault(variante, []).append(ver["semantica"])
    V2.log(f"  semantica {variante}: tolti {({d: len(v) for d, v in extra.items()}) or 0}")
    return ver, contaminati


def leggimi(manifesti: dict) -> None:
    righe = ["# Corpus di calibrazione v2 CORRETTO per contaminazione semantica (`*_v2_sem`)", "",
             f"Scritto il {time.strftime('%Y-%m-%d %H:%M:%S')} da `thor_v2/costruisci_corpus_calibrazione_v2_sem.py` "
             "leggendo i manifesti (nessun numero trascritto a mano). Stesse fonti, quote, template e uso del v2 "
             "(`LEGGIMI.md`); in piu' le regole R1-R3 (docstring dello script). Obbligatorio come il v2: "
             "`THOR_VALIDA_BATCH_SEQ=8 THOR_VALIDA_N_HELDOUT=1`.", "",
             "| file | sha256 | esempi tolti dalle regole R1 / R2 / R3 (tutti i giri) | domini diversi dal v2 |", "|---|---|---|---|"]
    for v, m in manifesti.items():
        r = REGISTRO.get(v, [])
        conta = [sum(len(g["dettaglio"][k]) for g in r) for k in ("R1_mmlu", "R2_cot", "R3_quasi_doppioni")]
        diversi = [d for d, u in m.get("domini_identici_al_v2", {}).items() if not u]
        righe.append(f"| `calibrazione/{m['file']}` | `{m['sha256_file']}` | {conta[0]} / {conta[1]} / {conta[2]} | "
                     f"{', '.join(diversi) or 'nessuno'} |")
    righe += ["", "Dettaglio (esempio tolto, testo, riferimento) nel campo `correzione_semantica` di ogni manifesto `.json`.",
              "Il pannello v2 e i file v2 non sono stati toccati."]
    (DIR_CAL / "LEGGIMI_v2_sem.md").write_text("\n".join(righe) + "\n", encoding="utf-8")


def main() -> int:
    import torch
    V2.OUT_PT = OUT_PT
    V2.verifica_v2 = verifica_sem
    V2.aggiorna_meta_pannello = lambda meta, risultati: V2.log("meta del pannello v2 NON aggiornato (variante _sem)")
    for p in OUT_PT.values():
        if p.name in V2.PROTETTI:
            raise AssertionError(p)
    sys.argv = [sys.argv[0], "--variante", "entrambe"]
    rc = V2.main()
    manifesti = {}
    for v, p in OUT_PT.items():
        m = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
        nuovo = torch.load(p, weights_only=True)
        base = V2.DIR_CAL / p.name.replace("_sem.pt", ".pt")
        vecchio = torch.load(base, weights_only=True)
        m["domini_identici_al_v2"] = {d: bool(torch.equal(nuovo[d], vecchio[d])) for d in nuovo}
        m["derivato_da"] = {"file": base.name, "sha256": C.sha256_file(base)}
        m["correzione_semantica"] = {"script": "thor_v2/costruisci_corpus_calibrazione_v2_sem.py",
                                     "originali_metamathqa": {"file": str(ORIGINALI),
                                                              "sha256": C.sha256_file(ORIGINALI)},
                                     "giri": REGISTRO.get(v, [])}
        p.with_suffix(".json").write_text(json.dumps(m, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        manifesti[v] = m
        V2.log(f"{p.name}: domini identici al v2 {m['domini_identici_al_v2']}")
    leggimi(manifesti)
    return rc


if __name__ == "__main__":
    sys.exit(main())
