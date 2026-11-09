#!/usr/bin/env python3
"""Scarica i dati PUBBLICI del pannello held-out allargato (leva #1, prova A0/A1 di
`thor_v2/PIANO_LEVE_GAP_Q2K.md`). 24/9/2026.

## Perche' esiste

L'held-out della finestra oggi e' 1 batch da 9 esempi (math/code in formato chat):
le ricette Q1/Q2/Q4 (0,509 / 0,502 / 0,499x v20) non si distinguono, e nessuno
sa se il guadagno vale anche su TESTO GREZZO, cioe' sulla metrica del mandato
(PPL wikitext). Il pannello (`pannello_heldout.py`) ha bisogno di due corpora
pubblici e ripetibili (regola di progetto: SOLO test pubblici):

- **wikitext-2-raw-v1, split VALIDATION** (`Salesforce/wikitext`). ⛔ MAI lo
  split test: e' la PPL di verdetto (`wiki.test.raw`, 72x4096). Usare la
  validation per scegliere la ricetta e' una taratura lieve sul dominio: va
  dichiarata (PIANO sez. 1, leva #1).
- **FineWeb-Edu `sample-10BT`** (`HuggingFaceFW/fineweb-edu`): SOLO i primi
  2 row group (2000 documenti) del primo file parquet. Il file intero e' 2,15 GB
  (oltre il tetto di 2 GB della sessione): si leggono i row group via richieste
  HTTP a intervalli (HfFileSystem + pyarrow), mai il file intero.

⛔ Mai `calib_grande` (contiene il repository privato MOGAVIS).
⛔ Il venv `venv-catq` NON ha `datasets` e non si installa nulla (lo usa la
forgia in corso): bastano `huggingface_hub` + `pyarrow`, gia' presenti.

## Uscite (in `@@ROOT:models@@/gguf/tony-tern/pannello/`, tetto 1 GB)

- `wikitext-2-raw-v1.validation.raw` — le righe dello split concatenate (ogni
  riga porta gia' il suo "\\n"; le righe vuote, che il parquet HF scrive "",
  tornano " \\n"): lo stesso formato di `wiki.test.raw` che `llama-perplexity` legge.
- `fineweb-edu.sample-10BT.000_00000.rg0-1.jsonl` — un documento per riga:
  `text`, `id`, `url`, `dump`, `token_count` (quello dichiarato dalla sorgente,
  tokenizzatore GPT-2), `score`.
- `LEGGIMI.md` + `manifesto.json` — fonte, split, revisione (sha del commit HF),
  sha256 dei file sorgente e di quelli scritti, righe/documenti, byte e token
  (tokenizzatore di Thor, `@@ROOT:models@@/gguf/thor-src-hf`, senza token speciali).

## Uso (CPU leggera, zero GPU)

    env CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=2 RAYON_NUM_THREADS=2 \\
        TOKENIZERS_PARALLELISM=false nice -n 19 \\
        @@ROOT:venv@@/bin/python3 scarica_dati_pannello.py

Idempotente: se i file esistono e combaciano col manifesto, non riscarica.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("RAYON_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

DIR = Path(os.getenv("THOR_PANNELLO_DATI", "@@ROOT:models@@/gguf/tony-tern/pannello"))
SRC_TOK = Path(os.getenv("THOR_HF_DIR", "@@ROOT:models@@/gguf/thor-src-hf"))

WIKI_REPO = "Salesforce/wikitext"
WIKI_REV = "b08601e04326c79dfdd32d625aee71d232d685c3"      # sha del commit HF letto il 24/9 21:2x
WIKI_FILE = "wikitext-2-raw-v1/validation-00000-of-00001.parquet"
WIKI_OUT = "wikitext-2-raw-v1.validation.raw"

EDU_REPO = "HuggingFaceFW/fineweb-edu"
EDU_REV = "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9"       # sha del commit HF letto il 24/9 21:2x
EDU_FILE = "sample/10BT/000_00000.parquet"
EDU_ROW_GROUPS = (0, 1)                                     # 2 x 1000 documenti
EDU_OUT = "fineweb-edu.sample-10BT.000_00000.rg0-1.jsonl"
EDU_CAMPI = ("text", "id", "url", "dump", "token_count", "score")

TETTO_BYTE = 1 * 2**30


def log(*a) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for blocco in iter(lambda: f.read(1 << 20), b""):
            h.update(blocco)
    return h.hexdigest()


def _info_sorgente(fs, repo: str, rev: str, file: str) -> dict:
    """Metadati del file sorgente al commit fissato (dimensione e sha256 LFS)."""
    info = fs.info(f"datasets/{repo}@{rev}/{file}")
    lfs = info.get("lfs")
    return {"repo": repo, "revisione": rev, "file": file, "byte_sorgente": int(info["size"]),
            "sha256_sorgente": (getattr(lfs, "sha256", None) if lfs is not None else None)}


def scarica_wikitext(fs) -> dict:
    import pyarrow.parquet as pq
    meta = _info_sorgente(fs, WIKI_REPO, WIKI_REV, WIKI_FILE)
    with fs.open(f"datasets/{WIKI_REPO}@{WIKI_REV}/{WIKI_FILE}", "rb") as f:
        tab = pq.read_table(f, columns=["text"])
    righe = tab.column("text").to_pylist()
    # Il vecchio script di caricamento HF di wikitext scriveva "" per ogni riga
    # vuota del file .raw originale (dove la riga vuota e' " \n", verificato sulla
    # testa di wiki.test.raw: " \n = Robert Boulter = \n \n Robert ..."). Senza
    # questa mappa i paragrafi si incollano e il formato non e' piu' quello che
    # llama-perplexity legge.
    vuote = sum(1 for r in righe if not r)
    testo = "".join(r if r else " \n" for r in righe)
    out = DIR / WIKI_OUT
    out.write_text(testo, encoding="utf-8")
    log(f"wikitext validation: {len(righe)} righe, {len(testo)} caratteri → {out}")
    meta.update({"split": "validation", "config": "wikitext-2-raw-v1", "righe": len(righe),
                 "righe_vuote_ripristinate": vuote, "caratteri": len(testo), "uscita": out.name,
                 "formato": "righe concatenate; le righe vuote (\"\" nel parquet HF) tornano \" \\n\" come nel .raw originale"})
    return meta


def scarica_fineweb(fs) -> dict:
    import pyarrow.parquet as pq
    meta = _info_sorgente(fs, EDU_REPO, EDU_REV, EDU_FILE)
    out = DIR / EDU_OUT
    n_doc = 0
    with fs.open(f"datasets/{EDU_REPO}@{EDU_REV}/{EDU_FILE}", "rb", block_size=8 * 2**20) as f:
        pf = pq.ParquetFile(f)
        n_rg_totali = pf.metadata.num_row_groups
        with out.open("w", encoding="utf-8") as g:
            for rg in EDU_ROW_GROUPS:
                tab = pf.read_row_group(rg, columns=list(EDU_CAMPI))
                for riga in tab.to_pylist():
                    g.write(json.dumps({k: riga[k] for k in EDU_CAMPI}, ensure_ascii=False) + "\n")
                    n_doc += 1
    log(f"FineWeb-Edu: {n_doc} documenti (row group {list(EDU_ROW_GROUPS)} di {n_rg_totali}) → {out}")
    meta.update({"config": "sample-10BT", "split": "train (unico split del campione)",
                 "row_group": list(EDU_ROW_GROUPS), "row_group_totali_nel_file": n_rg_totali,
                 "documenti": n_doc, "uscita": out.name,
                 "nota": "solo i row group indicati, letti a intervalli HTTP: il file sorgente non e' mai scaricato per intero"})
    return meta


def conta_token(meta_wiki: dict, meta_edu: dict) -> None:
    """Token col tokenizzatore di Thor, SENZA token speciali (il pannello tokenizza
    cosi' il testo grezzo). Wikitext: il file intero in un colpo (come llama-perplexity
    tokenizza wiki.test.raw). FineWeb: somma per documento (separatori esclusi)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(SRC_TOK))
    testo = (DIR / meta_wiki["uscita"]).read_text(encoding="utf-8")
    meta_wiki["token_thor"] = len(tok(testo, add_special_tokens=False).input_ids)
    tot, dichiarati = 0, 0
    with (DIR / meta_edu["uscita"]).open(encoding="utf-8") as f:
        for riga in f:
            d = json.loads(riga)
            tot += len(tok(d["text"], add_special_tokens=False).input_ids)
            dichiarati += int(d.get("token_count") or 0)
    meta_edu["token_thor_somma_documenti"] = tot
    meta_edu["token_count_dichiarato_sorgente_somma"] = dichiarati
    log(f"token (tokenizzatore Thor): wikitext validation {meta_wiki['token_thor']} · "
        f"FineWeb-Edu {tot} (dichiarati dalla sorgente, GPT-2: {dichiarati})")


def scrivi_leggimi(manifesto: dict) -> None:
    w, e = manifesto["wikitext"], manifesto["fineweb_edu"]
    testo = f"""# Dati del pannello held-out allargato (Thor, leva #1)

Scaricati il {manifesto['quando']} da `thor_v2/scarica_dati_pannello.py`. Letti da
`thor_v2/pannello_heldout.py`. Nessun dato privato: solo HuggingFace pubblico.

## wikitext-2-raw-v1 — split VALIDATION (mai il test: e' la PPL di verdetto)

| campo | valore |
|---|---|
| sorgente | `{w['repo']}` · `{w['file']}` |
| revisione (commit HF) | `{w['revisione']}` |
| sha256 del parquet sorgente | `{w['sha256_sorgente']}` ({w['byte_sorgente']} byte) |
| righe | {w['righe']} (di cui {w['righe_vuote_ripristinate']} vuote) |
| file scritto | `{w['uscita']}` ({w['byte_scritti']} byte, sha256 `{w['sha256_scritto']}`) |
| formato | righe concatenate (ognuna ha gia' il suo `\\n`); le righe vuote, che il parquet HF scrive `""`, tornano `" \\n"` come nel `.raw` originale letto da `llama-perplexity` |
| token (tokenizzatore Thor, senza speciali) | **{w['token_thor']}** |

Avvertenza (PIANO sez. 1, leva #1): scegliere la ricetta sulla validation di
wikitext e' una taratura lieve sul dominio della metrica. Va dichiarata. La PPL
di verdetto resta sul solo split test.

## FineWeb-Edu — `sample-10BT`, primi {e['documenti']} documenti

| campo | valore |
|---|---|
| sorgente | `{e['repo']}` · `{e['file']}` |
| revisione (commit HF) | `{e['revisione']}` |
| sha256 del parquet sorgente | `{e['sha256_sorgente']}` ({e['byte_sorgente']} byte: NON scaricato per intero) |
| parte letta | row group {e['row_group']} su {e['row_group_totali_nel_file']} (1000 documenti ciascuno), via HTTP a intervalli |
| file scritto | `{e['uscita']}` ({e['byte_scritti']} byte, sha256 `{e['sha256_scritto']}`) |
| campi | {', '.join(EDU_CAMPI)} |
| token (tokenizzatore Thor, somma per documento) | **{e['token_thor_somma_documenti']}** |
| token dichiarati dalla sorgente (GPT-2) | {e['token_count_dichiarato_sorgente_somma']} |

Il pannello usa i documenti DALL'INIZIO del file, concatenati con `\\n\\n`, finche'
non riempie le sue batch; il resto e' riserva (es. miscela di domini, leva 9a):
chi lo usa per ADDESTRARE deve partire DOPO l'ultimo documento del pannello
(il numero e' scritto nel manifesto del pannello, `pannello_heldout.py`).
"""
    (DIR / "LEGGIMI.md").write_text(testo, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--forza", action="store_true", help="riscarica anche se i file ci sono")
    args = ap.parse_args()
    DIR.mkdir(parents=True, exist_ok=True)
    man_path = DIR / "manifesto.json"
    if man_path.exists() and not args.forza:
        man = json.loads(man_path.read_text())
        ok = all((DIR / man[k]["uscita"]).exists()
                 and sha256_file(DIR / man[k]["uscita"]) == man[k]["sha256_scritto"]
                 for k in ("wikitext", "fineweb_edu"))
        if ok:
            log(f"dati gia' presenti e integri ({man_path}): niente da fare")
            return 0
        log("manifesto presente ma file mancanti/alterati: riscarico")

    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    meta_wiki = scarica_wikitext(fs)
    meta_edu = scarica_fineweb(fs)
    for m in (meta_wiki, meta_edu):
        p = DIR / m["uscita"]
        m["byte_scritti"] = p.stat().st_size
        m["sha256_scritto"] = sha256_file(p)
    totale = sum(m["byte_scritti"] for m in (meta_wiki, meta_edu))
    if totale > TETTO_BYTE:
        raise SystemExit(f"⛔ {totale} byte scritti: oltre il tetto di {TETTO_BYTE} (1 GB)")
    conta_token(meta_wiki, meta_edu)
    manifesto = {"quando": time.strftime("%Y-%m-%d %H:%M:%S"), "wikitext": meta_wiki, "fineweb_edu": meta_edu,
                 "tokenizzatore": str(SRC_TOK), "byte_totali": totale}
    man_path.write_text(json.dumps(manifesto, ensure_ascii=False, indent=2), encoding="utf-8")
    scrivi_leggimi(manifesto)
    log(f"✅ fatto: {totale / 2**20:.1f} MiB in {DIR} · LEGGIMI.md + manifesto.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
