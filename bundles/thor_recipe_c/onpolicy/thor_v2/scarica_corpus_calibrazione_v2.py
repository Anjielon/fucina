#!/usr/bin/env python3
"""Fonti v2 del corpus di calibrazione di Thor: licenze CORRETTE e origine di ogni fonte,
revisione fissata per le domande del dominio cot, fonti nuove per la variante SENZA
Llama/GPT. 25/9/2026, notte (difetto D4 della verifica indipendente del corpus v1).

## Perche' esiste

La verifica del corpus v1 ha trovato dichiarazioni di licenza sbagliate o incomplete
(`scarica_corpus_calibrazione.py`, dizionari LICENZA e PROVENIENZA):
  - la licenza Llama 3.1 era descritta come «attribuzione». Il testo vero (sezione
    1.b.i, copiato in `prove_licenze/`) chiede «Built with Llama» e, per un modello
    addestrato o MIGLIORATO con le uscite di Llama, il nome che inizia con «Llama»;
  - smoltalk2 con ragionamento e ReasonXL erano dichiarati senza l'origine Llama dei
    loro prompt (Smol-Magpie-Ultra = Magpie su Llama-3.1-405B-Instruct);
  - ReasonXL «di origine HuggingFaceTB/smoltalk»: SmolTalk v1 contiene anche
    OpenHermes-2.5 (GPT-4), MetaMathQA (GPT-3.5), SystemChats...: il sotto-insieme
    non e' verificabile;
  - dominio cot, «nessun termine di terze parti»: le risposte sono di Thor, ma le
    DOMANDE vengono da MetaMathQA (MIT; parte riscritta da GPT-3.5-Turbo) e da
    nvidia/OpenCodeInstruct (CC-BY-4.0), scaricate da `resolve/main` senza revisione.

Questo script NON tocca `manifesto_sorgenti.json` (v1, citato per sha256 dal corpus v1):
scrive `manifesto_sorgenti_v2.json`, che ricopia ogni estratto v1 con licenza, origine e
obblighi corretti, e aggiunge:
  - `cot_origine_domande`: le domande dei file cot ritrovate TESTO PER TESTO nelle fonti
    alla revisione fissata (MetaMathQA@aa4f34d3, primi 60.000.001 byte come il
    generatore; OpenCodeInstruct@8f3ba5ba, file data/train-00000-of-00050.parquet),
    con il tipo MetaMathQA di ogni domanda (AnsAug = domanda originale GSM8K/MATH);
    in piu' si RIPRODUCE il campionamento del generatore (seme 20260921) e si confronta
    con l'insieme delle domande nei file;
  - `reasonxl_pulito_cal`: righe di ReasonXL (file italian-00001, row group 0-2) le cui
    fonti non hanno Llama ne' GPT nella catena: matematica OpenMathReasoning (problemi
    AoPS rifiniti da Qwen2.5-32B-Instruct, soluzioni DeepSeek-R1) e generale
    allenai/qasc e flan_v2 (prompt umani, risposte DeepSeek-R1-0528); traduzione
    Qwen3-32B (Apache-2.0);
  - `massive_11_en`: MASSIVE 1.1 en-US dallo STESSO tar ufficiale (sha256 verificato),
    frasi umane per gli strumenti in inglese della variante senza Llama/GPT;
  - `prove_licenze/`: schede e testi di licenza usati come prova, con sha256.

Classi di origine (campo `origine.classe`), la stessa regola in costruzione:
  umana            testo scritto da persone (eventuali filtri automatici dichiarati a parte)
  sintetica_pulita generato da modelli dichiarati, nessuno Llama ne' GPT nella catena
  llama_gpt        prompt o risposte (anche in parte) da Llama o da GPT
  non_dichiarata   la fonte non dichiara chi ha scritto prompt o risposte: NON si puo'
                   certificare «senza Llama/GPT», quindi la variante pulita la esclude

## Uso (CPU leggera, rete)

    env CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=4 \\
        nice -n 19 @@ROOT:venv@@/bin/python3 scarica_corpus_calibrazione_v2.py [--solo ID,ID] [--forza]

Idempotente come la v1: un estratto integro (sha256 = manifesto v2) non si riscarica.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import re
import sys
import tarfile
import time
import unicodedata
import urllib.request
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "4")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scarica_corpus_calibrazione as S  # noqa: E402  (v1: helper di rete e di scrittura, NON modificata)

DIR = S.DIR
MAN_V1 = DIR / "manifesto_sorgenti.json"
MAN_V2 = DIR / "manifesto_sorgenti_v2.json"
PROVE = DIR / "prove_licenze"
DIR_COT = S.DIR_COT

# ── revisioni fissate in piu' (risolte con HfApi il 25/9: ultimo commit 2023-12-21 e 2025-04-28, cioe'
#    gia' «main» quando il generatore cot le lesse da resolve/main il 21/9/2026) ──
REV_NUOVE = {
    "meta-math/MetaMathQA": "aa4f34d3d2d3231299b5b03d9b3e5a20da45aa18",
    "nvidia/OpenCodeInstruct": "8f3ba5bafe4d6e8db46082cf7ae6741bc370604d",
}
S.REV.update(REV_NUOVE)
METAMATH_FILE = "MetaMathQA-395K.json"
METAMATH_PREFISSO = 60_000_001          # il generatore chiedeva Range bytes=0-60000000 (estremi inclusi)
OCI_FILE = "data/train-00000-of-00050.parquet"   # siblings[0] fra i parquet, come il generatore
SEME_GENERATORE_COT = 20260921

# ReasonXL: (dataset_name, source) senza Llama ne' GPT nella catena (schede Nemotron-Cascade-SFT
# Stage-1/Stage-2, OpenMathReasoning, ReasonXL: vedi ORIGINE["reasonxl_pulito"])
REASONXL_PULITE = {
    ("Nemotron-Cascade-SFT-Stage-1-math", "OpenMathReasoning"): "matematica",
    ("Nemotron-Cascade-SFT-Stage-2-math", "OpenMathReasoning"): "matematica",
    ("Nemotron-Cascade-SFT-Stage-1-general", "allenai/qasc"): "generale",
    ("Nemotron-Cascade-SFT-Stage-2-general", "allenai/qasc"): "generale",
    ("Nemotron-Cascade-SFT-Stage-1-general", "flan_v2"): "generale",
    ("Nemotron-Cascade-SFT-Stage-2-general", "flan_v2"): "generale",
}

# ── licenza, testo, origine e obblighi per FONTE LOGICA (letti dalle schede alla revisione fissata) ──
LLAMA31 = ("Licenza Llama 3.1 (sez. 1.b.i, testo in prove_licenze/llama3_1_LICENSE.txt): chi distribuisce un "
           "modello «addestrato, affinato o altrimenti migliorato» con uscite di Llama deve far iniziare il suo "
           "nome con «Llama», mostrare «Built with Llama» e allegare copia dell'accordo")
ORIGINE = {
    "wikitext": {"classe": "umana", "licenza": "CC-BY-SA-3.0 / GFDL", "testo": "Wikipedia EN, persone",
                 "obblighi": "attribuzione e condivisione allo stesso modo per il TESTO ridistribuito"},
    "fineweb_edu": {"classe": "umana", "licenza": "ODC-By 1.0 (+ termini d'uso di Common Crawl)",
                    "testo": "web inglese, persone",
                    "nota": "SELEZIONATO da fineweb-edu-classifier, addestrato su annotazioni di "
                            "Llama3-70B-Instruct (scheda FineWeb-Edu@87f09149, righe 505 e 585): legame "
                            "INDIRETTO, nessun testo Llama nel corpus. Lettura nostra, non parere legale: gli "
                            "obblighi della licenza Llama 3 (non 3.1) riguardano chi ha usato quelle uscite per "
                            "addestrare il classificatore (HuggingFace), non chi usa il testo selezionato; resta "
                            "segnalato"},
    "fineweb2_it": {"classe": "umana", "licenza": "ODC-By 1.0 (+ termini d'uso di Common Crawl)",
                    "testo": "web italiano, persone",
                    "nota": "filtri: GlotLID ed euristiche, nessun filtro di qualita' a modello (scheda FineWeb-2)"},
    "wikipedia_it": {"classe": "umana", "licenza": "CC-BY-SA-3.0 / GFDL (come dichiara la scheda HF)",
                     "testo": "Wikipedia IT, persone"},
    "aya": {"classe": "umana", "licenza": "Apache-2.0",
            "testo": "annotatori Aya (164 righe IT originali, 574 ri-annotate da persone)"},
    "oasst2": {"classe": "umana", "licenza": "Apache-2.0", "testo": "volontari OpenAssistant, prompt e risposte"},
    "mfaq": {"classe": "umana", "licenza": "CC0-1.0 (dichiarata dalla scheda; pagine FAQ da Common Crawl)",
             "testo": "pagine FAQ di siti italiani"},
    "massive": {"classe": "umana", "licenza": "CC-BY-4.0",
                "testo": "frasi di persone (MASSIVE 1.1: originali en-US e localizzazioni fatte da traduttori)"},
    "massive_agents": {"classe": "umana", "licenza": "CC-BY-4.0",
                       "testo": "frasi MASSIVE (test) + descrizioni di funzioni e argomenti ANNOTATE DA PERSONE "
                                "(paper MASSIVE-Agents, EMNLP Findings 2025, sez. 2.5: «We human annotate "
                                "descriptions for each of the functions»)"},
    "smoltalk2_nothink": {
        "classe": "llama_gpt", "licenza": "Apache-2.0 (scheda smoltalk2: i dataset nuovi; per SmolTalk rimanda "
                                          "all'originale, Apache-2.0)",
        "testo": "SINTETICO: prompt di Smol-Magpie-Ultra (Magpie su Llama-3.1-405B-Instruct) e Smol-Constraints "
                 "tradotti in 8 lingue da un modello Qwen (versione NON dichiarata) che genera anche le risposte "
                 "(scheda smoltalk2@fc6cc210, sezione SFT no_think)",
        "obblighi": LLAMA31 + "; per il Qwen non dichiarato: se fosse un modello a licenza Qwen (non Apache) "
                              "servirebbe anche «Built with Qwen» — non verificabile"},
    "smoltalk2_think": {
        "classe": "llama_gpt", "licenza": "Apache-2.0 (dataset nuovo smoltalk-multilingual8-Qwen3-32B)",
        "testo": "SINTETICO: prompt SmolTalk multilingual-8languages (le stesse traduzioni dei prompt "
                 "Smol-Magpie-Ultra/Smol-Constraints, origine Llama-3.1-405B-Instruct), risposte e ragionamento "
                 "di Qwen3-32B (Apache-2.0: nessun «Built with Qwen»)",
        "obblighi": LLAMA31},
    "reasonxl_smoltalk": {
        "classe": "llama_gpt",
        "licenza": "Apache-2.0 (ReasonXL) su Nemotron-Cascade-SFT (CC-BY-4.0)",
        "testo": "SINTETICO: prompt della fonte «HuggingFaceTB/smoltalk» di Nemotron-Cascade: SmolTalk v1 mescola "
                 "Smol-Magpie-Ultra (Llama-3.1-405B-Instruct), OpenHermes-2.5 (GPT-4), MetaMathQA (GPT-3.5), "
                 "SystemChats, everyday-conversations-llama3.1 ...; il sotto-insieme NON e' verificabile. "
                 "Ragionamento e risposte DeepSeek-R1 / R1-0528 (MIT), traduzione Qwen3-32B (Apache-2.0), filtro "
                 "con ellamind/propella-1-4b (Apache-2.0, solo annotazione)",
        "obblighi": LLAMA31 + " (se il prompt viene da Llama); termini OpenAI per le parti GPT"},
    "reasonxl_pulito": {
        "classe": "sintetica_pulita",
        "licenza": "Apache-2.0 (ReasonXL) su Nemotron-Cascade-SFT (CC-BY-4.0); OpenMathReasoning CC-BY-4.0; "
                   "allenai/qasc CC-BY-4.0; FLAN v2 (sottoinsieme Tulu v2) Apache-2.0",
        "testo": "SINTETICO: matematica = problemi AoPS (persone) rifiniti da Qwen2.5-32B-Instruct (Apache-2.0), "
                 "soluzioni DeepSeek-R1 (MIT) [scheda OpenMathReasoning]; generale = domande QASC (crowd) e FLAN v2 "
                 "(modelli su dataset umani); ragionamento e risposte DeepSeek-R1 (Stage-1) o DeepSeek-R1-0528 "
                 "(Stage-2) [schede Nemotron-Cascade-SFT]; traduzione in italiano Qwen3-32B (Apache-2.0) [scheda "
                 "ReasonXL]; filtro di qualita' ellamind/propella-1-4b (Apache-2.0, solo annotazione)",
        "obblighi": "attribuzione CC-BY-4.0 (NVIDIA Nemotron-Cascade, OpenMathReasoning, AllenAI QASC); MIT "
                    "(DeepSeek); Apache-2.0 (ReasonXL, Qwen3-32B, Qwen2.5-32B)"},
    "nemotron_tool": {
        "classe": "non_dichiarata", "licenza": "CC-BY-4.0",
        "testo": "SINTETICO: risposte Qwen3-235B-A22B (Apache-2.0; colonna generator = Qwen3-235B-A22B su tutte le "
                 "righe lette); PROMPT e schemi degli strumenti: la scheda dice solo «sourced from either public and "
                 "open corpus or synthetically generated» e non nomina il generatore",
        "obblighi": "attribuzione CC-BY-4.0 (NVIDIA)"},
    "cot_math": {
        "classe": "per_tipo_di_domanda",
        "licenza": "risposte e ragionamento: Thor (Ornith-1.5-35B-A3B, il modello stesso); domande: MetaMathQA "
                   "(MIT) @aa4f34d3",
        "testo": "domande per tipo MetaMathQA: AnsAug = domanda ORIGINALE di GSM8K/MATH train (persone, MIT); "
                 "Rephrased e SV = riscritte da GPT-3.5-Turbo (paper MetaMath 2309.12284, sez. 4.1: «GPT-3.5-Turbo "
                 "is used for rephrasing questions as well as generating answers in all four augmentations»); FOBAR "
                 "= domanda originale con un numero mascherato da x e frase fissa (sez. 3.3; chi abbia eseguito il "
                 "mascheramento non e' dichiarato)",
        "classi_per_tipo": {"AnsAug": "umana", "Rephrased": "llama_gpt", "SV": "llama_gpt", "FOBAR": "non_dichiarata"},
        "obblighi": "MIT (MetaMathQA); per le domande GPT: termini d'uso OpenAI"},
    "cot_code": {
        "classe": "non_dichiarata",
        "licenza": "risposte e ragionamento: Thor; domande: nvidia/OpenCodeInstruct (CC-BY-4.0) @8f3ba5ba",
        "testo": "domande generate con Genetic-Instruct da un «Instruct-LLM» che il rapporto (arXiv 2504.04030, "
                 "sez. 2.2) NON nomina; semi: domande TACO (persone) e istruzioni OSS-Instruct di "
                 "Qwen2.5-32B-Instruct (sez. 2.1); soluzioni originali Qwen2.5-Coder-32B-Instruct (non usate)",
        "obblighi": "attribuzione CC-BY-4.0 (NVIDIA)"},
}
# estratto (chiave del manifesto v1/v2) -> fonte logica
FONTE_DI = {
    "fineweb2_it_train": "fineweb2_it", "fineweb2_it_test": "fineweb2_it", "wiki_it_cal": "wikipedia_it",
    "wiki_it_pan": "wikipedia_it", "aya_it": "aya", "oasst2_it_train": "oasst2", "oasst2_it_valid": "oasst2",
    "mfaq_it_train": "mfaq", "mfaq_it_valid": "mfaq", "smoltalk2_nothink_cal": "smoltalk2_nothink",
    "smoltalk2_nothink_pan": "smoltalk2_nothink", "smoltalk2_think_cal": "smoltalk2_think",
    "smoltalk2_think_pan": "smoltalk2_think", "reasonxl_cal": "reasonxl_smoltalk", "reasonxl_pan": "reasonxl_smoltalk",
    "nemotron_tool_cal": "nemotron_tool", "nemotron_tool_pan": "nemotron_tool", "massive_agents_it": "massive_agents",
    "massive_11_it": "massive", "wikitext2_train": "wikitext", "massive_11_en": "massive",
    "reasonxl_pulito_cal": "reasonxl_pulito", "cot_origine_domande": "cot_math",
}
# prove documentali: (nome file in prove_licenze/, url o repo@rev, cosa prova)
PROVE_DOC = [
    ("llama3_1_LICENSE.txt", "https://raw.githubusercontent.com/meta-llama/llama-models/"
                             "1f0feb795a4130697ced243fb53051670d591653/models/llama3_1/LICENSE",
     "clausola 1.b.i: nome «Llama» e «Built with Llama»"),
    ("massive_agents_emnlp2025.pdf", "https://aclanthology.org/2025.findings-emnlp.1099.pdf",
     "sez. 2.5: descrizioni di funzioni e argomenti annotate da persone"),
    ("metamath_2309.12284v4.html", "https://arxiv.org/html/2309.12284v4",
     "sez. 4.1: GPT-3.5-Turbo per le riscritture; sez. 3.1-3.3: AnsAug, Rephrasing, SV, FOBAR"),
    ("opencodeinstruct_2504.04030v2.html", "https://arxiv.org/html/2504.04030v2",
     "sez. 2.1-2.3: semi TACO + Qwen2.5-32B-Instruct; Instruct-LLM di Genetic-Instruct non nominato"),
    ("genetic_instruct_2407.21077v3.html", "https://arxiv.org/html/2407.21077v3",
     "nel paper originale di Genetic-Instruct l'Instructor-LLM e' Mixtral-8x22B (non vale per OpenCodeInstruct)"),
]
PROVE_SCHEDE = [   # (repo, revisione, prova)
    ("HuggingFaceTB/smoltalk2", "fc6cc2103c066455aade5d7fbb346039ae36ca5e", "origine di multilingual-8languages"),
    ("HuggingFaceTB/smoltalk", "5feaf2fd3ffca7c237fc38d1861bc30365d48ffa", "composizione di SmolTalk v1"),
    ("toroe/ReasonXL-SFT", "321da71a81762dd28b3462e25cde4be3e7103840", "traduzione Qwen3-32B, fonti"),
    ("nvidia/Nemotron-Cascade-SFT-Stage-1", "e59a356b107737562f064ec252e6adacbc737899", "risposte DeepSeek-R1"),
    ("nvidia/Nemotron-Cascade-SFT-Stage-2", "54cfa23ba28c9d94aeae15142e73bde6f8d14d8b", "risposte DeepSeek-R1-0528"),
    ("nvidia/OpenMathReasoning", "d3d08664755704f422af97d43a7ff0ded4bd95df", "AoPS + Qwen2.5-32B + DeepSeek-R1"),
    ("nvidia/Nemotron-Post-Training-Dataset-v1", "74e23eb6f830fef4a9e96a92f6f6262214cbb9a8", "prompt non dichiarati"),
    ("nvidia/OpenCodeInstruct", "8f3ba5bafe4d6e8db46082cf7ae6741bc370604d", "licenza, generatore non in scheda"),
    ("meta-math/MetaMathQA", "aa4f34d3d2d3231299b5b03d9b3e5a20da45aa18", "licenza MIT, origine GSM8K/MATH"),
    ("HuggingFaceFW/fineweb-edu", "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9", "classificatore da Llama3-70B"),
    ("HuggingFaceFW/fineweb-2", "af9c13333eb981300149d5ca60a8e9d659b276b9", "nessun filtro a modello"),
]


def log(*a) -> None:
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def norm(t: str) -> str:
    t = unicodedata.normalize("NFKC", t or "").lower()
    return " ".join(re.sub(r"[^\w]+", " ", t).split())


# ─────────────────────────────── prove documentali ───────────────────────────────
def e_prove() -> dict:
    from huggingface_hub import hf_hub_download
    PROVE.mkdir(parents=True, exist_ok=True)
    out = {"documenti": [], "schede": []}
    for nome, url, cosa in PROVE_DOC:
        dati = urllib.request.urlopen(url, timeout=60).read()
        (PROVE / nome).write_bytes(dati)
        out["documenti"].append({"file": nome, "url": url, "byte": len(dati),
                                 "sha256": hashlib.sha256(dati).hexdigest(), "prova": cosa})
    for repo, rev, cosa in PROVE_SCHEDE:
        p = Path(hf_hub_download(repo, "README.md", repo_type="dataset", revision=rev))
        nome = f"{repo.replace('/', '__')}@{rev[:12]}.README.md"
        (PROVE / nome).write_bytes(p.read_bytes())
        out["schede"].append({"file": nome, "repo": repo, "revisione": rev, "sha256": S.sha256_file(PROVE / nome),
                              "prova": cosa})
    out["uscita"] = None
    return out


# ─────────────────────────────── cot: origine delle domande ───────────────────────────────
def _voci_metamath(raw: bytes) -> list:
    """Stesso taglio del generatore (thor/genera_cot_calibrazione.py:campioni_math)."""
    txt = raw.decode("utf-8", "ignore")
    voci = []
    for blocco in txt.split('{"query"')[1:]:
        try:
            voci.append(json.loads('{"query"' + blocco.rsplit('}', 1)[0] + '}'))
        except Exception:  # noqa: BLE001 — l'ultimo oggetto e' troncato per costruzione (come nel generatore)
            pass
    return voci


def e_cot_origine() -> dict:
    import pyarrow.parquet as pq
    repo_m, repo_c = "meta-math/MetaMathQA", "nvidia/OpenCodeInstruct"
    meta_m, meta_c = S.info_sorgente(repo_m, METAMATH_FILE), S.info_sorgente(repo_c, OCI_FILE)
    meta_m["licenza"], meta_c["licenza"] = "MIT", "CC-BY-4.0"
    with S.fs().open(S._url(repo_m, METAMATH_FILE), "rb", block_size=8 * 2**20) as f:
        raw = f.read(METAMATH_PREFISSO)
    voci = _voci_metamath(raw)
    with S.fs().open(S._url(repo_c, OCI_FILE), "rb", block_size=8 * 2**20) as f:
        pf = pq.ParquetFile(f)
        tab = pf.read(columns=["id", "input", "domain", "generation_algorithm"]).to_pylist()
    # riproduzione del campionamento del generatore (seme a livello di modulo, math poi code)
    rnd = random.Random(SEME_GENERATORE_COT)
    voci_mesc = list(voci)
    rnd.shuffle(voci_mesc)
    q_math_rip = [v["query"] for v in voci_mesc[:1024]]
    q_code = [r["input"] for r in tab if r["input"] is not None]
    rnd.shuffle(q_code)
    q_code_rip = [str(x) for x in q_code[:1024]]
    tipo_di: dict = {}
    for v in voci:
        tipo_di.setdefault(v["query"], set()).add(v.get("type", "?"))
    oci_di = {r["input"]: r for r in tab if r["input"] is not None}
    righe, conta = [], {"math": {}, "code": {}}
    esito = {}
    for tipo, rip in (("math", q_math_rip), ("code", q_code_rip)):
        qs = []
        for riga in (DIR_COT / f"cot_{tipo}.jsonl").open(encoding="utf-8"):
            d = json.loads(riga)
            qs.append(d["q"])
            if tipo == "math":
                t = sorted(tipo_di.get(d["q"], set()))
                trovato = bool(t)
                tag = t[0] if len(t) == 1 else ("|".join(t) if t else "ASSENTE")
                extra = {"tipo_metamath": tag}
            else:
                r = oci_di.get(d["q"])
                trovato = r is not None
                tag = "OpenCodeInstruct" if trovato else "ASSENTE"
                extra = {"oci_id": r["id"] if r else None, "domain": r["domain"] if r else None,
                         "generation_algorithm": r["generation_algorithm"] if r else None}
            conta[tipo][tag] = conta[tipo].get(tag, 0) + 1
            righe.append({"tipo_cot": tipo, "sha256_q": hashlib.sha256(d["q"].encode("utf-8")).hexdigest(),
                          "trovata": trovato, "errore_generazione": bool(d.get("errore")), **extra})
        esito[tipo] = {"domande_nel_file": len(qs), "ritrovate_testo_per_testo": sum(1 for q in qs if (
            q in tipo_di if tipo == "math" else q in oci_di)),
            "campionamento_riprodotto_uguale": sorted(qs) == sorted(rip),
            "stesso_ordine_del_campionamento": qs == rip}
    meta = {"repo": f"{repo_m} + {repo_c}", "revisione": f"{S.REV[repo_m]} + {S.REV[repo_c]}",
            "file": f"{METAMATH_FILE} (primi {METAMATH_PREFISSO} byte) + {OCI_FILE} (colonne id, input, domain, "
                    "generation_algorithm)",
            "sorgenti": [meta_m, meta_c], "sha256_prefisso_metamath": hashlib.sha256(raw).hexdigest(),
            "byte_prefisso_metamath": len(raw), "voci_metamath_nel_prefisso": len(voci), "righe_oci": len(tab),
            "seme_generatore": SEME_GENERATORE_COT, "verifica": esito, "conta_per_tipo": conta,
            "provenienza_testo": ORIGINE["cot_math"]["testo"] + " · " + ORIGINE["cot_code"]["testo"],
            "uso": "classificazione dell'origine delle domande del dominio cot (variante senza Llama/GPT)"}
    meta.update(S.scrivi_jsonl("cot_origine_domande.jsonl", righe))
    return meta


# ─────────────────────────────── ReasonXL pulito ───────────────────────────────
def e_reasonxl_pulito(rgs=(0, 1, 2)) -> dict:
    repo, file = "toroe/ReasonXL-SFT", "data/italian-00001-of-00058.parquet"
    meta = S.info_sorgente(repo, file)
    righe, foot = S.leggi_row_group(repo, file, rgs, ["messages", "source", "dataset_name", "ds_uid", "language",
                                                      "row_index"])
    conta: dict = {}
    tenute = []
    for r in righe:
        k = (str(r["dataset_name"]), str(r["source"]))
        if k in REASONXL_PULITE:
            r["ramo"] = REASONXL_PULITE[k]
            tenute.append(r)
            conta[" | ".join(k)] = conta.get(" | ".join(k), 0) + 1
    meta.update(foot)
    meta.update({"split": "italian", "row_group": list(rgs), "righe_lette": len(righe),
                 "filtro": "(dataset_name, source) in " + json.dumps([list(k) for k in REASONXL_PULITE]),
                 "tenute_per_fonte": conta, "licenza": ORIGINE["reasonxl_pulito"]["licenza"],
                 "provenienza_testo": ORIGINE["reasonxl_pulito"]["testo"],
                 "uso": "calibrazione it_pensiero della variante senza Llama/GPT (stesso file 1 della v1, righe "
                        "diverse; il pannello usa il file 30)"})
    meta.update(S.scrivi_jsonl(f"reasonxl_italian-00001-of-00058_rg{'-'.join(map(str, rgs))}_pulito.jsonl", tenute))
    return meta


# ─────────────────────────────── MASSIVE 1.1 en-US ───────────────────────────────
def e_massive_en() -> dict:
    cart = DIR / "massive-1.1"
    prov = json.loads((cart / "provenienza.json").read_text())
    dati = urllib.request.urlopen(prov["url"], timeout=300).read()
    sha_tar = hashlib.sha256(dati).hexdigest()
    if len(dati) != prov["byte"] or sha_tar != prov["sha256_tar"]:
        raise RuntimeError(f"tar MASSIVE: {len(dati)} B sha {sha_tar} != provenienza {prov['byte']} {prov['sha256_tar']}")
    membro = "1.1/data/en-US.jsonl"
    with tarfile.open(fileobj=io.BytesIO(dati), mode="r:gz") as t:
        contenuto = t.extractfile(membro).read()
    out = cart / "en-US.jsonl"
    out.write_bytes(contenuto)
    conta: dict = {}
    for riga in contenuto.decode("utf-8").splitlines():
        p = json.loads(riga)["partition"]
        conta[p] = conta.get(p, 0) + 1
    return {"repo": "AmazonScience/massive (loader HF) · tar ufficiale S3", "revisione": S.REV["AmazonScience/massive"],
            "url_tar": prov["url"], "byte_tar": len(dati), "sha256_tar": sha_tar, "membro": membro,
            "uscita": f"massive-1.1/{out.name}", "byte_scritti": out.stat().st_size, "sha256_scritto": S.sha256_file(out),
            "righe_per_partizione": conta, "licenza": ORIGINE["massive"]["licenza"],
            "provenienza_testo": "frasi di persone (MASSIVE 1.1 en-US, la lingua di partenza del corpus)",
            "uso": "calibrazione strumenti in inglese della variante senza Llama/GPT (SOLO partition == train, "
                   "scenari della calibrazione; dev e test mai)"}


def estratti() -> dict:
    return {"prove_licenze": e_prove, "cot_origine_domande": e_cot_origine,
            "reasonxl_pulito_cal": e_reasonxl_pulito, "massive_11_en": e_massive_en}


def _con_origine(chiave: str, meta):
    """Aggiunge a un meta v1 l'origine corretta (senza toccare impronte e numeri)."""
    f = FONTE_DI.get(chiave)
    for m in (meta if isinstance(meta, list) else [meta]):
        if f:
            o = ORIGINE[f]
            m["licenza"] = o["licenza"]
            m["provenienza_testo"] = o["testo"]
            m["origine"] = {"fonte_logica": f, **o}
    return meta


def integro(meta) -> bool:
    for m in (meta if isinstance(meta, list) else [meta]):
        u = m.get("uscita")
        if u and (not (DIR / u).exists() or S.sha256_file(DIR / u) != m.get("sha256_scritto")):
            return False
    if isinstance(meta, dict) and "documenti" in meta:
        return all((PROVE / d["file"]).exists() and S.sha256_file(PROVE / d["file"]) == d["sha256"]
                   for d in meta["documenti"] + meta["schede"])
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--solo", default="")
    ap.add_argument("--forza", action="store_true")
    a = ap.parse_args()
    v1 = json.loads(MAN_V1.read_text())
    man = json.loads(MAN_V2.read_text()) if MAN_V2.exists() else {"estratti": {}}
    # 1) estratti v1: stessi file e impronte, licenza/origine corrette
    for k, v in v1["estratti"].items():
        for m in (v if isinstance(v, list) else [v]):
            u = m.get("uscita")
            if u and S.sha256_file(DIR / u) != m["sha256_scritto"]:
                raise RuntimeError(f"estratto v1 {k}: {u} non combacia col manifesto v1")
        man["estratti"][k] = _con_origine(k, json.loads(json.dumps(v)))
    # 2) estratti nuovi
    tutti = estratti()
    scelti = [s for s in a.solo.split(",") if s] or list(tutti)
    for s in scelti:
        if not a.forza and s in man.get("nuovi", {}) and integro(man["nuovi"][s]):
            log(f"{s}: gia' presente e integro")
            continue
        t = time.time()
        log(f"{s}: scarico ...")
        meta = tutti[s]()
        man.setdefault("nuovi", {})[s] = _con_origine(s, meta)
        log(f"  {s}: {meta.get('uscita')} · righe {meta.get('righe_scritte')} · {time.time() - t:.0f} s")
        if s == "cot_origine_domande":
            log(f"  verifica cot: {json.dumps(meta['verifica'])} · per tipo {json.dumps(meta['conta_per_tipo'])}")
        if S.byte_su_disco() > S.TETTO_BYTE:
            raise SystemExit(f"⛔ {S.byte_su_disco()} byte in {DIR}: oltre il tetto")
        MAN_V2.write_text(json.dumps(man, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    for k in list(man.get("nuovi", {})):          # origine sempre dal dizionario ORIGINE di OGGI (anche se non riscaricato)
        man["nuovi"][k] = _con_origine(k, man["nuovi"][k])
    locali = json.loads(json.dumps(v1["locali"]))
    for k in ("cot_math", "cot_code"):
        o = ORIGINE[k]
        locali[k].update({"licenza": o["licenza"], "provenienza_testo": o["testo"], "origine": {"fonte_logica": k, **o},
                          "revisione_domande": S.REV["meta-math/MetaMathQA" if k == "cot_math"
                                                      else "nvidia/OpenCodeInstruct"]})
    locali["fineweb_edu"].update({"licenza": ORIGINE["fineweb_edu"]["licenza"],
                                  "origine": {"fonte_logica": "fineweb_edu", **ORIGINE["fineweb_edu"]}})
    man["locali"] = locali
    man["origini"] = ORIGINE
    man["revisioni"] = S.REV
    man["manifesto_v1"] = {"file": MAN_V1.name, "sha256": S.sha256_file(MAN_V1),
                           "nota": "non modificato: le dichiarazioni di licenza v1 sono corrette QUI (errata)"}
    man["quando"] = time.strftime("%Y-%m-%d %H:%M:%S")
    man["byte_su_disco"] = S.byte_su_disco()
    MAN_V2.write_text(json.dumps(man, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    log(f"✅ {MAN_V2} · {man['byte_su_disco'] / 2**20:.1f} MiB in {DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
