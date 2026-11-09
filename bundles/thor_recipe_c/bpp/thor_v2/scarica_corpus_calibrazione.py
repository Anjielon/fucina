#!/usr/bin/env python3
"""Scarica le FONTI PUBBLICHE del corpus di calibrazione multi-dominio di Thor
(italiano, chat, ragionamento, strumenti) e del pannello dei domini nuovi.
24/9/2026, notte.

## Perche' esiste

Il pannello held-out (`pannello_heldout.py`) ha mostrato che la forgia migliora
SOLO i domini presenti nella calibrazione: con la sola chat math/code la finestra
[0,1] faceva 0,53x gli esperti v20 su math/code ma 1,70x su Wikipedia e 1,43x su
FineWeb-Edu. La miscela M2 (cot 2, edu 3, wiki 3) scende sotto 1 ovunque, col
margine stretto. Thor sara' il cervello di MOGAVIS, assistente domestico ITALIANO
(chat senza ragionamento, strumenti, pianificazione): Angelo, 24/9: «Aggiungi
l'italiano! Ed espandi anche con altro se e' il caso». Questo script prende SOLO
le parti che servono delle fonti scelte dalla ricerca (row group o intervalli di
byte letti via HTTP, mai file interi quando non serve), revisione HF FISSATA,
sha256 dei file sorgente (LFS) e dei file scritti, licenza dichiarata.

⛔ Nessun dato privato di MOGAVIS (conversazioni, posta, documenti della famiglia).
⛔ Niente GPU. Il venv `venv-catq` non ha `datasets`: bastano `huggingface_hub`
   e `pyarrow`, gia' presenti (non si installa nulla: la forgia usa quel venv).
⛔ Disco: tetto 3 GB per tutto (sorgenti + corpus). /tmp e' RAM: mai file grandi li'.

## Uscite: `@@ROOT:models@@/gguf/tony-tern/calibrazione/sorgenti/`

Un `.jsonl` per estratto (solo le colonne utili) + `manifesto_sorgenti.json` con,
per ogni estratto: repo, revisione, file, byte e sha256 del file sorgente (LFS),
parte letta (row group / byte), righe lette e scritte, filtri, byte e sha256
scritti, licenza, chi ha generato il testo (persone o quale modello), uso
(calibrazione / pannello). Registra anche le fonti LOCALI gia' presenti (tracce
CoT di Thor, wikitext-2 train, FineWeb-Edu, MASSIVE 1.1) con la loro impronta; il
wikitext-2 TRAIN locale e' RICOSTRUITO dal parquet HF e confrontato byte per byte.

## Uso (CPU leggera)

    env CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=4 RAYON_NUM_THREADS=4 \\
        nice -n 19 @@ROOT:venv@@/bin/python3 scarica_corpus_calibrazione.py [--solo ID,ID] [--forza]

Idempotente: un estratto gia' scritto e integro (sha256 = manifesto) non si riscarica.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sys
import tarfile
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("RAYON_NUM_THREADS", "4")

DIR = Path(os.getenv("THOR_CALIB_SORGENTI", "@@ROOT:models@@/gguf/tony-tern/calibrazione/sorgenti"))
DIR_PANNELLO = Path(os.getenv("THOR_PANNELLO_DATI", "@@ROOT:models@@/gguf/tony-tern/pannello"))
DIR_COT = Path(os.getenv("THOR_COT_DIR", "@@ROOT:forge@@/thor/cot_calib"))
MANIFESTO = DIR / "manifesto_sorgenti.json"
TETTO_BYTE = 3 * 10**9              # tutto il corpus (sorgenti + .pt) sta sotto 3 GB
TETTO_ROW_GROUP_MIB = 64.0          # un row group piu' grosso di cosi' e' un errore di scelta

# ── revisioni HF FISSATE (sha del commit, risolte con HfApi il 24/9 notte) ──
REV = {
    "Salesforce/wikitext": "b08601e04326c79dfdd32d625aee71d232d685c3",
    "HuggingFaceFW/fineweb-2": "af9c13333eb981300149d5ca60a8e9d659b276b9",
    "wikimedia/wikipedia": "b04c8d1ceb2f5cd4588862100d08de323dccfbaa",
    "CohereLabs/aya_dataset": "f9ea04583f02a8f86404ff6c58bf75fe637df8a2",
    "OpenAssistant/oasst2": "179dd21fc55192153d94adb0e0ce8f69e222bf75",
    "clips/mfaq": "87a7bada8da4fe2a7b738c6d3e549153383198ad",
    "HuggingFaceTB/smoltalk2": "fc6cc2103c066455aade5d7fbb346039ae36ca5e",
    "toroe/ReasonXL-SFT": "321da71a81762dd28b3462e25cde4be3e7103840",
    "nvidia/Nemotron-Post-Training-Dataset-v1": "74e23eb6f830fef4a9e96a92f6f6262214cbb9a8",
    "AmazonScience/massive-agents": "b6156972182bdf34e68c5b5dfbfe6d30db82f104",
    "AmazonScience/massive": "ff6bd8e4b27c3543e4f8fe2108f32bb95a6f8740",
}

# ── licenze e provenienza del testo (dichiarate, dalle schede HF lette il 24/9) ──
LICENZA = {
    "Salesforce/wikitext": "CC-BY-SA-3.0 / GFDL (testo Wikipedia inglese)",
    "HuggingFaceFW/fineweb-edu": "ODC-By 1.0 (+ termini d'uso di Common Crawl)",
    "HuggingFaceFW/fineweb-2": "ODC-By 1.0 (+ termini d'uso di Common Crawl)",
    "wikimedia/wikipedia": "CC-BY-SA-3.0 / GFDL (testo Wikipedia italiana, dump 20231101)",
    "CohereLabs/aya_dataset": "Apache-2.0",
    "OpenAssistant/oasst2": "Apache-2.0",
    "clips/mfaq": "CC0-1.0 (dichiarata dalla scheda; testo di pagine FAQ da Common Crawl)",
    "HuggingFaceTB/smoltalk2": "Apache-2.0 (scheda smoltalk2: i dataset nuovi, compreso "
                               "smoltalk-multilingual8-Qwen3-32B; la parte multilingual-8languages "
                               "viene da SmolTalk, Apache-2.0)",
    "toroe/ReasonXL-SFT": "Apache-2.0 (origine Nemotron-Cascade CC-BY-4.0)",
    "nvidia/Nemotron-Post-Training-Dataset-v1": "CC-BY-4.0",
    "AmazonScience/massive-agents": "CC-BY-4.0",
    "AmazonScience/massive": "CC-BY-4.0",
    "locale:thor_cot": "tracce generate da Thor stesso (Ornith-1.5-35B-A3B, pesi pubblici) su domande "
                       "math/code: nessun termine di terze parti sulle uscite",
}
PROVENIENZA = {
    "wikitext": "testo umano (Wikipedia EN)",
    "fineweb-edu": "testo umano (web, filtrato da un classificatore)",
    "fineweb-2": "testo umano (web italiano, Common Crawl)",
    "wikipedia": "testo umano (Wikipedia IT)",
    "aya": "testo umano (annotatori Aya: 164 righe IT originali, 574 ri-annotate)",
    "oasst2": "testo umano (volontari OpenAssistant, prompt e risposte)",
    "mfaq": "testo umano (pagine FAQ di siti italiani)",
    "smoltalk2_nothink": "SINTETICO: prompt Magpie da Llama-3.1-405B-Instruct (licenza Llama 3.1: "
                         "attribuzione se si distribuisce un modello addestrato), tradotti e risposti "
                         "da un modello Qwen (versione non dichiarata; attribuzione «Built with Qwen»)",
    "smoltalk2_think": "SINTETICO: prompt SmolTalk tradotti, risposte e ragionamento di Qwen3-32B (Apache-2.0)",
    "reasonxl": "SINTETICO: da Nemotron-Cascade (risposte DeepSeek-R1, MIT), tradotto in italiano da "
                "Qwen3-32B; tenute SOLO le righe di origine HuggingFaceTB/smoltalk (le righe da "
                "ShareGPT_Vicuna_unfiltered sono escluse: prompt raccolti da conversazioni ChatGPT, zona "
                "grigia evitabile)",
    "nemotron_tool": "SINTETICO: Qwen3-235B-A22B (Apache-2.0), rilasciato da NVIDIA in CC-BY-4.0",
    "massive_agents": "frasi di persone (MASSIVE, localizzate in italiano da traduttori) + schemi di "
                      "funzione e chiamate attese del gruppo Amazon",
    "massive": "frasi di persone (MASSIVE 1.1, localizzazione it-IT fatta da persone)",
    "thor_cot": "generato da Thor (il modello stesso)",
}


def log(*a) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for blocco in iter(lambda: f.read(1 << 20), b""):
            h.update(blocco)
    return h.hexdigest()


# ── riconoscitore di lingua a parole funzionali (lo stesso della ricerca, verificato a campione) ──
_PAROLE = {
    "it": "il lo gli della delle degli dello che di è sono anche più perché questo questa nel nella alla "
          "non una ma come cosa".split(),
    "es": "el los las del que por para está también más pero como una muy esto cuando".split(),
    "pt": "não você são uma do da dos das também mais com isso para muito".split(),
    "fr": "le les des est pas vous une avec pour dans qui et sur au aux ce cette".split(),
    "de": "der die und nicht ist das mit sie ein eine auch für auf dem den".split(),
    "en": "the and is are of to in that with for this you it be".split(),
}
_SET = {k: set(v) for k, v in _PAROLE.items()}


def lingua(testo: str) -> str:
    """Lingua per conteggio di parole funzionali sulle prime 400 parole; '?' se incerta
    (vincitore < 3 parole o < 1,4x il secondo). Euristica: le righe '?' si scartano."""
    parole = re.findall(r"[a-zàèéìòóùâêôãõç]+", testo.lower())[:400]
    if not parole:
        return "?"
    punti = {k: sum(p in s for p in parole) for k, s in _SET.items()}
    migliore = max(punti, key=punti.get)
    ordinati = sorted(punti.values(), reverse=True)
    if ordinati[0] < 3 or ordinati[0] < 1.4 * max(ordinati[1], 1):
        return "?"
    return migliore


def _senza_pensiero(testo: str) -> str:
    return re.sub(r"(?s)<think>.*?</think>", "", testo or "")


# ─────────────────────────────── accesso HF ───────────────────────────────
_FS = None


def fs():
    global _FS
    if _FS is None:
        from huggingface_hub import HfFileSystem
        _FS = HfFileSystem()
    return _FS


def _url(repo: str, file: str) -> str:
    return f"datasets/{repo}@{REV[repo]}/{file}"


def info_sorgente(repo: str, file: str) -> dict:
    info = fs().info(_url(repo, file))
    lfs = info.get("lfs")
    sha = getattr(lfs, "sha256", None) if lfs is not None else None
    if sha is None and isinstance(lfs, dict):
        sha = lfs.get("sha256")
    return {"repo": repo, "revisione": REV[repo], "file": file, "byte_sorgente": int(info["size"]),
            "sha256_sorgente": sha, "blob_id_git": info.get("blob_id"),
            "licenza": LICENZA.get(repo)}


def leggi_row_group(repo: str, file: str, rgs, colonne=None) -> "tuple[list, dict]":
    """Legge SOLO i row group indicati (richieste HTTP a intervalli). Ritorna le righe
    e i dati del footer (row group totali, byte compressi delle colonne lette)."""
    import pyarrow.parquet as pq
    righe, compressi = [], 0
    with fs().open(_url(repo, file), "rb", block_size=4 * 2**20) as f:
        pf = pq.ParquetFile(f)
        md = pf.metadata
        nomi = [md.schema.column(j).path.split(".")[0] for j in range(md.num_columns)]
        for rg in rgs:
            g = md.row_group(rg)
            dim = sum(g.column(j).total_compressed_size for j in range(g.num_columns)
                      if colonne is None or nomi[j] in colonne)
            if dim / 2**20 > TETTO_ROW_GROUP_MIB:
                raise RuntimeError(f"{repo}/{file} row group {rg}: {dim / 2**20:.1f} MiB > {TETTO_ROW_GROUP_MIB}")
            compressi += dim
            for i, r in enumerate(pf.read_row_group(rg, columns=colonne).to_pylist()):
                r["_pos"] = f"rg{rg}:{i}"          # posizione nel file sorgente (provenienza per riga)
                righe.append(r)
        foot = {"row_group_totali_nel_file": md.num_row_groups, "righe_nel_file": md.num_rows}
    foot["byte_compressi_colonne_lette"] = compressi
    return righe, foot


def leggi_colonne_intere(repo: str, file: str, colonne) -> "tuple[list, dict]":
    """Solo le colonne indicate, su tutto il file (per filtrare file medi per lingua)."""
    import pyarrow.parquet as pq
    with fs().open(_url(repo, file), "rb", block_size=16 * 2**20) as f:
        pf = pq.ParquetFile(f)
        md = pf.metadata
        nomi = [md.schema.column(j).path.split(".")[0] for j in range(md.num_columns)]
        compressi = sum(md.row_group(i).column(j).total_compressed_size
                        for i in range(md.num_row_groups) for j in range(md.num_columns) if nomi[j] in colonne)
        tab = pf.read(columns=list(colonne))
    return tab.to_pylist(), {"row_group_totali_nel_file": md.num_row_groups, "righe_nel_file": md.num_rows,
                             "byte_compressi_colonne_lette": compressi}


def scrivi_jsonl(nome: str, righe: list) -> dict:
    out = DIR / nome
    tmp = out.with_suffix(out.suffix + ".parziale")
    with tmp.open("w", encoding="utf-8") as g:
        for r in righe:
            g.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, out)
    return {"uscita": out.name, "righe_scritte": len(righe), "byte_scritti": out.stat().st_size,
            "sha256_scritto": sha256_file(out)}


# ───────────────────────────────── estratti ─────────────────────────────────
def e_fineweb2_it(split: str, rgs, uso: str, soglia: float):
    repo, file = "HuggingFaceFW/fineweb-2", f"data/ita_Latn/{split}/000_00000.parquet"
    meta = info_sorgente(repo, file)
    righe, foot = leggi_row_group(repo, file, rgs, ["text", "id", "url", "language_score"])
    tenute = [r for r in righe if (r["language_score"] or 0) >= soglia]
    meta.update(foot)
    meta.update({"config": "ita_Latn", "split": split, "row_group": list(rgs), "righe_lette": len(righe),
                       "filtro": f"language_score >= {soglia}", "scartate_filtro": len(righe) - len(tenute),
                       "provenienza_testo": PROVENIENZA["fineweb-2"], "uso": uso})
    meta.update(scrivi_jsonl(f"fineweb2_ita_{split}_000_rg{'-'.join(map(str, rgs))}.jsonl", tenute))
    return meta


def e_wiki_it(rgs, uso: str):
    repo, file = "wikimedia/wikipedia", "20231101.it/train-00000-of-00010.parquet"
    meta = info_sorgente(repo, file)
    righe, foot = leggi_row_group(repo, file, rgs, ["id", "url", "title", "text"])
    meta.update(foot)
    meta.update({"config": "20231101.it", "split": "train (unico)", "row_group": list(rgs),
                       "righe_lette": len(righe), "provenienza_testo": PROVENIENZA["wikipedia"], "uso": uso})
    meta.update(scrivi_jsonl(f"wikipedia_20231101_it_f00_rg{'-'.join(map(str, rgs))}.jsonl", righe))
    return meta


def e_aya_it():
    repo, file = "CohereLabs/aya_dataset", "data/train-00000-of-00001.parquet"
    meta = info_sorgente(repo, file)
    import pyarrow.parquet as pq
    colonne = ["inputs", "targets", "language", "language_code", "annotation_type", "user_id"]
    with fs().open(_url(repo, file), "rb", block_size=8 * 2**20) as f:
        pf = pq.ParquetFile(f)
        md = pf.metadata
        lingue = pf.read(columns=["language"]).column("language").to_pylist()
        # quali row group contengono righe italiane
        inizio, rg_it = 0, []
        for i in range(md.num_row_groups):
            n = md.row_group(i).num_rows
            if any(x == "Italian" for x in lingue[inizio:inizio + n]):
                rg_it.append(i)
            inizio += n
        righe = []
        for rg in rg_it:
            righe += [r for r in pf.read_row_group(rg, columns=colonne).to_pylist() if r["language"] == "Italian"]
    meta.update({"row_group_totali_nel_file": md.num_row_groups, "righe_nel_file": md.num_rows,
                 "row_group_letti": rg_it, "filtro": "language == 'Italian'", "righe_lette": len(righe),
                 "provenienza_testo": PROVENIENZA["aya"],
                 "uso": "calibrazione + pannello (partizione per hash del prompt normalizzato: vedi costruttore)",
                 "nota": "nessuna riga italiana nello split test ufficiale (verificato dalla ricerca)"})
    meta.update(scrivi_jsonl("aya_dataset_it.jsonl", righe))
    return meta


def e_oasst2_it(split: str, file: str, uso: str):
    repo = "OpenAssistant/oasst2"
    meta = info_sorgente(repo, file)
    colonne = ["message_id", "parent_id", "message_tree_id", "text", "role", "lang", "rank", "deleted",
               "review_result"]
    righe, foot = leggi_colonne_intere(repo, file, colonne)
    it = [r for r in righe if r["lang"] == "it"]
    meta.update(foot)
    meta.update({"split": split, "filtro": "lang == 'it' (alberi interi: i messaggi del thread)",
                       "righe_lette": len(righe), "provenienza_testo": PROVENIENZA["oasst2"], "uso": uso})
    meta.update(scrivi_jsonl(f"oasst2_it_{split}.jsonl", it))
    return meta


def e_mfaq_it(nome_file: str, uso: str, max_byte: "int | None"):
    repo = "clips/mfaq"
    file = f"data/it/{nome_file}"
    meta = info_sorgente(repo, file)
    with fs().open(_url(repo, file), "rb", block_size=4 * 2**20) as f:
        dati = f.read(max_byte) if max_byte else f.read()
    completo = max_byte is None or len(dati) < max_byte
    linee = dati.split(b"\n")
    if not completo:
        linee = linee[:-1]          # l'ultima riga di un intervallo e' troncata: si scarta
    righe = [json.loads(x) for x in linee if x.strip()]
    meta.update({"parte_letta": ("file intero" if completo else f"primi {max_byte} byte, solo righe complete"),
                 "byte_letti": len(dati), "righe_lette": len(righe), "provenienza_testo": PROVENIENZA["mfaq"],
                 "uso": uso})
    meta.update(scrivi_jsonl(f"mfaq_it_{nome_file.split('.')[0]}{'' if completo else '_testa'}.jsonl", righe))
    return meta


def e_smoltalk2(file: str, rgs, chiave: str, uso: str):
    repo = "HuggingFaceTB/smoltalk2"
    meta = info_sorgente(repo, file)
    righe, foot = leggi_row_group(repo, file, rgs, ["messages", "chat_template_kwargs"])
    tenute, conta = [], {}
    for r in righe:
        ms = r["messages"]
        utente = " ".join(m["content"] for m in ms if m["role"] == "user")
        risposte = " ".join(_senza_pensiero(m["content"]) for m in ms if m["role"] == "assistant")
        lu, lr = lingua(utente), lingua(risposte)
        conta[f"{lu}/{lr}"] = conta.get(f"{lu}/{lr}", 0) + 1
        if lu == "it" and lr == "it":
            kw = r.get("chat_template_kwargs") or {}
            tenute.append({"_pos": r["_pos"], "messages": ms,
                           "custom_instructions": kw.get("custom_instructions") or "",
                           "enable_thinking": kw.get("enable_thinking")})
    meta.update(foot)
    meta.update({"split_hf": file.split("/")[-1].rsplit("-", 2)[0], "row_group": list(rgs),
                       "righe_lette": len(righe),
                       "filtro": "lingua(utente) == lingua(risposte senza <think>) == 'it' (euristica a parole "
                                 "funzionali; le '?' si scartano)",
                       "lingue_utente_risposta": dict(sorted(conta.items(), key=lambda x: -x[1])),
                       "provenienza_testo": PROVENIENZA[chiave], "uso": uso})
    meta.update(scrivi_jsonl(f"smoltalk2_{file.split('/')[-1].replace('.parquet', '')}_rg"
                             f"{'-'.join(map(str, rgs))}_it.jsonl", tenute))
    return meta


def e_reasonxl(file: str, rgs, uso: str, fonti_ammesse: tuple):
    repo = "toroe/ReasonXL-SFT"
    meta = info_sorgente(repo, file)
    righe, foot = leggi_row_group(repo, file, rgs, ["messages", "source", "dataset_name", "ds_uid", "language",
                                                    "row_index"])
    conta: dict = {}
    for r in righe:
        k = f"{r['dataset_name']} | {r['source']}"
        conta[k] = conta.get(k, 0) + 1
    tenute = [r for r in righe if str(r["dataset_name"]).endswith("-general") and r["source"] in fonti_ammesse]
    meta.update(foot)
    meta.update({"split": "italian", "row_group": list(rgs), "righe_lette": len(righe),
                       "filtro": f"dataset_name *-general e source in {list(fonti_ammesse)}",
                       "composizione_righe_lette": dict(sorted(conta.items(), key=lambda x: -x[1])),
                       "provenienza_testo": PROVENIENZA["reasonxl"], "uso": uso})
    meta.update(scrivi_jsonl(f"reasonxl_{file.split('/')[-1].replace('.parquet', '')}_rg"
                             f"{'-'.join(map(str, rgs))}_general.jsonl", tenute))
    return meta


def e_nemotron_tool(file: str, rgs, uso: str):
    repo = "nvidia/Nemotron-Post-Training-Dataset-v1"
    meta = info_sorgente(repo, file)
    righe, foot = leggi_row_group(repo, file, rgs, ["uuid", "license", "generator", "reasoning", "messages",
                                                    "metadata"])
    gen = {}
    for r in righe:
        gen[r["generator"]] = gen.get(r["generator"], 0) + 1
    meta.update(foot)
    meta.update({"split_hf": "tool", "row_group": list(rgs), "righe_lette": len(righe),
                       "generatori": gen, "provenienza_testo": PROVENIENZA["nemotron_tool"], "uso": uso})
    meta.update(scrivi_jsonl(f"nemotron_v1_{file.split('/')[-1].replace('.parquet', '')}_rg"
                             f"{'-'.join(map(str, rgs))}.jsonl", righe))
    return meta


def e_massive_agents():
    """it-IT di MASSIVE-Agents: 602 esempi del test di MASSIVE con gli schemi delle 55
    funzioni e le chiamate attese. File piccoli: interi, byte per byte."""
    repo = "AmazonScience/massive-agents"
    uscite = []
    for nome in ("input.jsonl", "golden.jsonl"):
        file = f"massive-full-converted-all-langs-with-id/it-IT/{nome}"
        meta = info_sorgente(repo, file)
        with fs().open(_url(repo, file), "rb", block_size=8 * 2**20) as f:
            dati = f.read()
        if len(dati) != meta["byte_sorgente"]:
            raise RuntimeError(f"{file}: letti {len(dati)} byte, attesi {meta['byte_sorgente']}")
        out = DIR / f"massive_agents_it-IT_{nome}"
        out.write_bytes(dati)
        sha = sha256_file(out)
        if meta["sha256_sorgente"] and sha != meta["sha256_sorgente"]:
            raise RuntimeError(f"{file}: sha256 {sha} != LFS {meta['sha256_sorgente']}")
        meta.update({"parte_letta": "file intero", "uscita": out.name, "byte_scritti": out.stat().st_size,
                     "sha256_scritto": sha, "righe_scritte": dati.count(b"\n"),
                     "provenienza_testo": PROVENIENZA["massive_agents"],
                     "uso": "pannello strumenti_it (602 esempi, tutti dello split test di MASSIVE) + schemi "
                            "delle funzioni per la conversione delle frasi train di MASSIVE 1.1"})
        uscite.append(meta)
    return uscite


def verifica_massive_11() -> dict:
    """MASSIVE 1.1 it-IT gia' estratto dal tar ufficiale (S3): si verifica l'impronta."""
    cart = DIR / "massive-1.1"
    prov = json.loads((cart / "provenienza.json").read_text())
    f = cart / "it-IT.jsonl"
    sha = sha256_file(f)
    if sha != prov["sha256_membro"]:
        raise RuntimeError(f"{f}: sha256 {sha} != provenienza {prov['sha256_membro']}")
    conta: dict = {}
    for riga in f.open(encoding="utf-8"):
        p = json.loads(riga)["partition"]
        conta[p] = conta.get(p, 0) + 1
    return {"repo": "AmazonScience/massive (loader HF) · tar ufficiale S3", "revisione": REV["AmazonScience/massive"],
            "url_tar": prov["url"], "byte_tar": prov["byte"], "sha256_tar": prov["sha256_tar"],
            "membro": prov["membro"], "uscita": f"massive-1.1/{f.name}", "byte_scritti": f.stat().st_size,
            "sha256_scritto": sha, "righe_per_partizione": conta, "licenza": LICENZA["AmazonScience/massive"],
            "provenienza_testo": PROVENIENZA["massive"],
            "uso": "calibrazione strumenti (SOLO partition == train; dev e test mai)"}


def verifica_wikitext_train() -> dict:
    """Il wikitext-2 TRAIN locale (in pannello/, scaricato il 24/9 22:3x senza manifesto)
    si RICOSTRUISCE dal parquet HF alla revisione fissata con la stessa regola del
    validation (righe vuote "" -> " \\n") e si confronta byte per byte."""
    import pyarrow.parquet as pq
    repo, file = "Salesforce/wikitext", "wikitext-2-raw-v1/train-00000-of-00001.parquet"
    meta = info_sorgente(repo, file)
    with fs().open(_url(repo, file), "rb") as f:
        righe = pq.read_table(f, columns=["text"]).column("text").to_pylist()
    testo = "".join(r if r else " \n" for r in righe)
    locale = DIR_PANNELLO / "wikitext-2-raw-v1.train.raw"
    ricostruito = hashlib.sha256(testo.encode("utf-8")).hexdigest()
    sha_loc = sha256_file(locale)
    meta.update({"split": "train", "config": "wikitext-2-raw-v1", "righe": len(righe),
                 "righe_vuote_ripristinate": sum(1 for r in righe if not r), "uscita_locale": str(locale),
                 "byte_locale": locale.stat().st_size, "sha256_locale": sha_loc,
                 "sha256_ricostruito_da_hf": ricostruito, "identico_a_hf": ricostruito == sha_loc,
                 "provenienza_testo": PROVENIENZA["wikitext"], "uso": "calibrazione (dominio wiki)"})
    if ricostruito != sha_loc:
        raise RuntimeError(f"{locale}: non combacia con la ricostruzione da HF ({sha_loc} != {ricostruito})")
    return meta


def registra_locali() -> dict:
    out = {}
    for tipo in ("math", "code"):
        p = DIR_COT / f"cot_{tipo}.jsonl"
        out[f"cot_{tipo}"] = {"percorso": str(p), "byte": p.stat().st_size, "sha256": sha256_file(p),
                              "licenza": LICENZA["locale:thor_cot"], "provenienza_testo": PROVENIENZA["thor_cot"],
                              "uso": "calibrazione (dominio cot), ESCLUSI gli esempi 0-8 (held-out storico) e "
                                     "279-385 (pannello) dell'ordine mescolato col seme 1234"}
    man_p = json.loads((DIR_PANNELLO / "manifesto.json").read_text())
    e = man_p["fineweb_edu"]
    p = DIR_PANNELLO / e["uscita"]
    sha = sha256_file(p)
    if sha != e["sha256_scritto"]:
        raise RuntimeError(f"{p}: sha256 {sha} != manifesto del pannello {e['sha256_scritto']}")
    out["fineweb_edu"] = {**{k: e[k] for k in ("repo", "revisione", "file", "byte_sorgente", "sha256_sorgente",
                                                "row_group", "documenti")},
                          "percorso": str(p), "sha256": sha, "licenza": LICENZA["HuggingFaceFW/fineweb-edu"],
                          "provenienza_testo": PROVENIENZA["fineweb-edu"],
                          "uso": "calibrazione (dominio edu) dal documento 173: 0-172 sono del pannello"}
    return out


# ─────────────────────────────── elenco estratti ───────────────────────────────
def estratti() -> dict:
    """id -> funzione che scarica e ritorna il meta (o una lista di meta)."""
    return {
        "fineweb2_it_train": lambda: e_fineweb2_it("train", (0, 1), "calibrazione (it_web)", 0.9),
        "fineweb2_it_test": lambda: e_fineweb2_it("test", (0,), "pannello (it_web): split test ufficiale", 0.9),
        "wiki_it_cal": lambda: e_wiki_it((0,), "calibrazione (it_wiki)"),
        "wiki_it_pan": lambda: e_wiki_it((1,), "pannello (it_wiki): row group diverso, id verificati disgiunti"),
        "aya_it": e_aya_it,
        "oasst2_it_train": lambda: e_oasst2_it("train", "data/train-00000-of-00001-88ba0162028a73fc.parquet",
                                               "calibrazione (it_chat)"),
        "oasst2_it_valid": lambda: e_oasst2_it("validation",
                                               "data/validation-00000-of-00001-1deeef95c3248fe0.parquet",
                                               "pannello (it_chat): split validation ufficiale"),
        "mfaq_it_train": lambda: e_mfaq_it("train.jsonl", "calibrazione (it_chat)", 4 * 2**20),
        "mfaq_it_valid": lambda: e_mfaq_it("valid.jsonl", "pannello (it_chat): split valid ufficiale", None),
        "smoltalk2_nothink_cal": lambda: e_smoltalk2(
            "SFT/smoltalk_multilingual_8languages_lang_5_no_think-00000-of-00002.parquet", (0, 1, 2, 3),
            "smoltalk2_nothink", "calibrazione (it_chat)"),
        "smoltalk2_nothink_pan": lambda: e_smoltalk2(
            "SFT/smoltalk_multilingual_8languages_lang_5_no_think-00001-of-00002.parquet", (0, 1, 2, 3),
            "smoltalk2_nothink", "pannello (it_chat)"),
        "smoltalk2_think_cal": lambda: e_smoltalk2(
            "SFT/smoltalk_multilingual8_Qwen3_32B_think-00000-of-00004.parquet", (0,),
            "smoltalk2_think", "calibrazione (it_pensiero)"),
        "smoltalk2_think_pan": lambda: e_smoltalk2(
            "SFT/smoltalk_multilingual8_Qwen3_32B_think-00003-of-00004.parquet", (0,),
            "smoltalk2_think", "pannello (it_pensiero)"),
        "reasonxl_cal": lambda: e_reasonxl("data/italian-00001-of-00058.parquet", (0,), "calibrazione (it_pensiero)",
                                           ("HuggingFaceTB/smoltalk",)),
        "reasonxl_pan": lambda: e_reasonxl("data/italian-00030-of-00058.parquet", (0,), "pannello (it_pensiero)",
                                           ("HuggingFaceTB/smoltalk",)),
        "nemotron_tool_cal": lambda: e_nemotron_tool("data/tool-00000-of-00013.parquet", (0,),
                                                     "calibrazione (strumenti, inglese)"),
        "nemotron_tool_pan": lambda: e_nemotron_tool("data/tool-00012-of-00013.parquet", (0,),
                                                     "pannello (strumenti_en): solo strumenti assenti in calibrazione"),
        "massive_agents_it": e_massive_agents,
        "massive_11_it": verifica_massive_11,
        "wikitext2_train": verifica_wikitext_train,
    }


def integro(meta) -> bool:
    for m in (meta if isinstance(meta, list) else [meta]):
        u = m.get("uscita")
        if not u:
            continue
        p = DIR / u
        if not p.exists() or sha256_file(p) != m.get("sha256_scritto"):
            return False
    return True


def byte_su_disco() -> int:
    return sum(p.stat().st_size for p in DIR.rglob("*") if p.is_file())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--solo", default="", help="sottoinsieme di id separati da virgola")
    ap.add_argument("--forza", action="store_true", help="riscarica anche gli estratti integri")
    args = ap.parse_args()
    DIR.mkdir(parents=True, exist_ok=True)
    man = json.loads(MANIFESTO.read_text()) if MANIFESTO.exists() else {"estratti": {}}
    tutti = estratti()
    scelti = [s for s in args.solo.split(",") if s] or list(tutti)
    for s in scelti:
        if s not in tutti:
            raise SystemExit(f"estratto sconosciuto: {s} (noti: {', '.join(tutti)})")
    t0 = time.time()
    for s in scelti:
        if not args.forza and s in man["estratti"] and integro(man["estratti"][s]):
            log(f"{s}: gia' presente e integro")
            continue
        t = time.time()
        log(f"{s}: scarico ...")
        meta = tutti[s]()
        man["estratti"][s] = meta
        for m in (meta if isinstance(meta, list) else [meta]):
            log(f"  {s}: {m.get('uscita') or m.get('uscita_locale')} · righe {m.get('righe_scritte', m.get('righe'))} · "
                f"{(m.get('byte_scritti') or m.get('byte_locale') or 0) / 2**20:.2f} MiB · "
                f"compressi letti {(m.get('byte_compressi_colonne_lette') or 0) / 2**20:.1f} MiB · {time.time() - t:.0f} s")
        occupati = byte_su_disco()
        if occupati > TETTO_BYTE:
            raise SystemExit(f"⛔ {occupati} byte in {DIR}: oltre il tetto di {TETTO_BYTE}")
        MANIFESTO.write_text(json.dumps(man, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    man["locali"] = registra_locali()
    man["quando"] = time.strftime("%Y-%m-%d %H:%M:%S")
    man["byte_su_disco"] = byte_su_disco()
    man["revisioni"] = REV
    MANIFESTO.write_text(json.dumps(man, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    log(f"✅ fatto in {time.time() - t0:.0f} s · {man['byte_su_disco'] / 2**20:.1f} MiB in {DIR} · {MANIFESTO.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
