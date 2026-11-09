#!/usr/bin/env python3
"""HELD-OUT DI CONOSCENZA per la forgia Thor — MMLU VALIDATION, deduplicata contro il TEST (26/9/2026).

## Perche' esiste
Misure di Thor-T (35B-A3B tutto ternario, 8,69 GiB): MMLU base 75,55 > forgia B (150 passi, 32 lotti) 71,95 > forgia A
(300 passi) 69,10 > B-150 (150 lotti) 65,85, mentre la PPL wikitext MIGLIORA (9,43-9,49 contro 10,04 della v20).
L'held-out di oggi (1 lotto di ragionamenti cot in inglese, stessa famiglia della calibrazione) NON predice l'MMLU:
in B-150 era -14% rispetto alla B mentre l'MMLU perdeva 6 punti. La perdita della forgia vede solo le direzioni che la
calibrazione attiva; la conoscenza delle 57 materie vale zero e l'ottimizzatore la spende gratis. Serve un held-out
che guardi PROPRIO quelle direzioni: domande MMLU con la risposta giusta, nella forma esatta del banco.

## Cosa fa
1. Legge SOLO lo split `validation` di cais/mmlu (1531 domande, 57 materie) dai parquet in
   `@@ROOT:models@@/gguf/tony-tern/conoscenza/all/`. ⛔ Il TEST e' il banco di valutazione: qui si legge solo per
   DEDUPLICARE (mai un suo token entra nell'held-out). `dev` e `auxiliary_train` non si toccano.
2. Forma di ogni domanda: IDENTICA a `diagnosi_strati_mmlu.py fase_testi`, cioe' `f"{q} {risposta}\\n\\n"` con `q` il
   prompt del banco (`converti_multiple_choice.py carica_mmlu`: descrizione della materia + domanda + «A. .. D. ..» +
   «Answer:») e `risposta` la LETTERA giusta. La forma si PROVA, non si assume: il formattatore di qui rigenera
   tutte le 14042 domande del test e deve ridare byte per byte i file del banco (`mc_v2/mmlu_*.bin`, stesse
   etichette). Se una sola differisce lo script si ferma.
3. DEDUP contro il test (testo normalizzato come `parole()` del corpus v2_sem: NFKC, minuscole, non-parola -> spazio):
   - UGUALE: domanda normalizzata identica a una del test;
   - R3: rapporto difflib >= 0.80, massimo dei due versi (difflib non e' simmetrico, euristica autojunk), come la
     regola R3 di `costruisci_corpus_calibrazione_v2_sem.py`, contro TUTTE le 14042 domande del test, senza soglia
     minima di parole (conservativo: una domanda corta uguale e' contaminazione quanto una lunga);
   - R1 (in piu', solo nel verso prudente): la domanda contiene INTERA una domanda del test di almeno 8 parole, come
     la R1 del v2_sem. Il manifesto dice quante ne toglie SOLO lei.
   Prefiltri ESATTI, non scorciatoie: ratio() = 2M/T con M <= intersezione dei multinsiemi di caratteri <=
   min(la, lb) (sono quick_ratio e real_quick_ratio di difflib, limiti SUPERIORI in entrambi i versi). Una coppia
   si scarta senza ratio() solo se il limite superiore e' sotto 0.8 con margine (5u <= 2T-2): non puo' superare la
   soglia. Il R3 del v2_sem usava un prefiltro Jaccard che PUO' perdere coppie; qui no.
4. Mescola con seme fisso (`random.Random(seme).shuffle`, come `carica_batch_calibrazione`), tokenizza ogni domanda
   da sola col tokenizzatore di `@@ROOT:models@@/gguf/thor-src-hf` (`add_special_tokens=False`), concatena e impacchetta
   in sequenze da L_SEQ=1024 token (stessa costruzione di `valida_finestra_8_12.carica_batch_calibrazione`: la coda
   che non riempie una sequenza si scarta, il manifesto dice quanti token).
5. Scrive (in modo atomico, mai in /tmp) `heldout_conoscenza.pt` = tensore int64 (n_seq, L_SEQ) e
   `heldout_conoscenza.json` (sha256 dei token, domande usate/scartate con il dettaglio di ogni scarto, seme, prime 3
   domande in chiaro, sha256 delle sorgenti), poi RILEGGE il file e verifica tutto.

Solo CPU, nessun modello caricato (solo il tokenizzatore):
    env CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=4 RAYON_NUM_THREADS=4 \\
        nice -n 19 @@ROOT:venv@@/bin/python3 thor_v2/prepara_conoscenza.py
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("RAYON_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import difflib  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import random  # noqa: E402
import re  # noqa: E402
import struct  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import unicodedata  # noqa: E402
from collections import defaultdict  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

QUI = Path(__file__).resolve().parent
CARTELLA = Path(os.getenv("THOR_CONOSCENZA_DIR", "@@ROOT:models@@/gguf/tony-tern/conoscenza"))
PARQUET = CARTELLA / "all"
FILE_VAL = PARQUET / "validation-00000-of-00001.parquet"
FILE_TEST = PARQUET / "test-00000-of-00001.parquet"
HF = Path(os.getenv("THOR_CONOSCENZA_TOKENIZER", "@@ROOT:models@@/gguf/thor-src-hf"))
BANCO = QUI.parent / "mc_v2"                      # i file che llama-perplexity legge: mmlu_00000.bin ... mmlu_14000.bin
OUT_PT = CARTELLA / "heldout_conoscenza.pt"
OUT_JSON = CARTELLA / "heldout_conoscenza.json"
L_SEQ = int(os.getenv("THOR_VALIDA_L_SEQ", "1024"))
SEME = int(os.getenv("THOR_CONOSCENZA_SEED", "1234"))
N_PROC = int(os.getenv("THOR_CONOSCENZA_PROC", "4"))
SOGLIA_RAPPORTO, MIN_PAROLE_R1 = 0.80, 8          # come R3 e R1 di costruisci_corpus_calibrazione_v2_sem.py
LETTERE = ["A", "B", "C", "D"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for blocco in iter(lambda: f.read(1 << 20), b""):
            h.update(blocco)
    return h.hexdigest()


def parole(t: str) -> list:
    """Identica a `parole()` di costruisci_corpus_calibrazione_v2_sem.py (normalizzazione della R3)."""
    t = unicodedata.normalize("NFKC", t or "").lower()
    return re.sub(r"[^\w]+", " ", t).split()


def prompt_banco(r: dict) -> str:
    """Il prompt di una domanda nel banco MMLU: STESSA f-string di converti_multiple_choice.carica_mmlu (lm-eval
    tasks/mmlu/default, 0-shot). Verificata byte per byte contro i file del banco in `verifica_forma`."""
    c = list(r["choices"])
    materia = r["subject"].replace("_", " ")
    return (f"The following are multiple choice questions (with answers) about {materia}.\n\n"
            f"{r['question'].strip()}\nA. {c[0]}\nB. {c[1]}\nC. {c[2]}\nD. {c[3]}\nAnswer:")


def testo_con_risposta(r: dict) -> str:
    """«domanda + risposta corretta» nella forma di diagnosi_strati_mmlu.fase_testi: f"{q} {risposta}\\n\\n"."""
    return f"{prompt_banco(r)} {LETTERE[int(r['answer'])]}\n\n"


def leggi_banco(p: Path) -> list:
    """[(domanda, [risposte], [etichette]), ...] dal formato di llama-perplexity --multiple-choice."""
    b = p.read_bytes()
    n = struct.unpack_from("<I", b, 0)[0]

    def stringa(o):
        ln = struct.unpack_from("<I", b, o)[0]
        return b[o + 4:o + 4 + ln].decode("utf-8"), o + 4 + ln
    out = []
    for o in struct.unpack_from(f"<{n}I", b, 4):
        q, o = stringa(o)
        k = struct.unpack_from("<I", b, o)[0]
        o += 4
        ans = []
        for _ in range(k):
            a, o = stringa(o)
            ans.append(a)
        out.append((q, ans, list(struct.unpack_from(f"<{k}i", b, o))))
    return out


def verifica_forma(test: list) -> dict:
    """Il formattatore di qui DEVE rigenerare il banco: ogni file mc_v2/mmlu_NNNNN.bin copre le domande del test da
    NNNNN in poi (a blocchi da 2000), mmlu_test.bin le copre tutte. Stessa domanda, risposte A-D, stessa etichetta.
    Prova anche che il test di questi parquet e' LO STESSO del banco: la dedup si fa contro il test giusto."""
    esito = {}
    for p in sorted(BANCO.glob("mmlu_*.bin")):
        compiti = leggi_banco(p)
        inizio = 0 if p.stem == "mmlu_test" else int(p.stem.split("_")[1])
        if p.stem == "mmlu_test" and len(compiti) != len(test):
            raise RuntimeError(f"{p.name}: {len(compiti)} domande, il test ne ha {len(test)}")
        for k, (q, ans, lab) in enumerate(compiti):
            r = test[inizio + k]
            if q != prompt_banco(r) or ans != LETTERE or lab != [int(i == int(r["answer"])) for i in range(4)]:
                raise RuntimeError(f"{p.name} domanda {k} (test #{inizio + k}): la forma di qui NON e' quella del banco")
        esito[p.name] = {"domande": len(compiti), "da_test": inizio, "sha256": sha256_file(p)}
    if "mmlu_00000.bin" not in esito:
        raise RuntimeError(f"banco {BANCO}/mmlu_00000.bin assente: la forma non si puo' verificare")
    return esito


# ── dedup: stato condiviso coi processi figli (fork: copiato una volta, nessuna serializzazione per coppia) ──
_NV: list = []
_NT: list = []
_CV = _CT = _LV = _LT = None


def _confronta(i: int) -> tuple:
    """Per la domanda di validation i: (i, rapporto massimo, test di quel massimo, quante >= soglia, coppie
    valutate con ratio(), test UGUALE o -1). Prefiltri esatti: 5u <= 2T-2 => 2M/T < 0.8 per ogni M <= u."""
    a, la = _NV[i], int(_LV[i])
    T = la + _LT
    u_len = np.minimum(la, _LT)
    c = np.nonzero(5 * u_len >= 2 * T - 1)[0]
    if c.size:
        u_chr = np.minimum(_CT[c], _CV[i]).sum(1)
        c = c[5 * u_chr >= 2 * T[c] - 1]
    migliore, j_mig, sopra, uguale = 0.0, -1, 0, -1
    for j in c.tolist():
        b = _NT[j]
        if a == b:
            uguale = j if uguale < 0 else uguale
        # difflib non e' simmetrico (euristica autojunk): si prende il massimo dei due versi, come R3
        r = max(difflib.SequenceMatcher(None, a, b).ratio(), difflib.SequenceMatcher(None, b, a).ratio())
        if r >= SOGLIA_RAPPORTO:
            sopra += 1
        if r > migliore:
            migliore, j_mig = r, j
    return i, migliore, j_mig, sopra, int(c.size), uguale


def dedup_contro_test(val: list, test: list) -> tuple:
    """Ritorna (scarti {i: dettaglio}, statistiche). Regole: UGUALE, R3 (difflib >= 0.80), R1 (test intero >= 8
    parole contenuto nella domanda)."""
    global _NV, _NT, _CV, _CT, _LV, _LT
    _NV = [" ".join(parole(r["question"])) for r in val]
    _NT = [" ".join(parole(r["question"])) for r in test]
    vuote = [i for i, s in enumerate(_NV) if not s] + [-1 - j for j, s in enumerate(_NT) if not s]
    if vuote:
        raise RuntimeError(f"domande che normalizzate sono VUOTE (la R3 non le confronterebbe bene): {vuote[:10]}")
    alfabeto = {ch: k for k, ch in enumerate(sorted(set("".join(_NV)) | set("".join(_NT))))}

    def conta(s: str) -> np.ndarray:
        v = np.zeros(len(alfabeto), dtype=np.int32)
        np.add.at(v, [alfabeto[ch] for ch in s], 1)
        return v
    _CV = np.stack([conta(s) for s in _NV])
    _CT = np.stack([conta(s) for s in _NT])
    _LV = np.array([len(s) for s in _NV], dtype=np.int64)
    _LT = np.array([len(s) for s in _NT], dtype=np.int64)
    t0 = time.time()
    with mp.get_context("fork").Pool(N_PROC) as pool:
        risultati = sorted(pool.imap_unordered(_confronta, range(len(val)), chunksize=8))
    log(f"R3: {sum(r[4] for r in risultati)} coppie valutate con ratio() su {len(val) * len(test)} "
        f"({time.time() - t0:.0f} s, {N_PROC} processi)")
    # R1: indice degli 8-grammi iniziali delle domande del test di almeno 8 parole (come mmlu_in del v2_sem)
    indice = defaultdict(set)
    for j, s in enumerate(_NT):
        w = s.split()
        if len(w) >= MIN_PAROLE_R1:
            indice[" ".join(w[:MIN_PAROLE_R1])].add(j)

    def r1(s: str) -> int:
        w, piena = s.split(), " " + s + " "
        for k in range(len(w) - MIN_PAROLE_R1 + 1):
            for j in sorted(indice.get(" ".join(w[k:k + MIN_PAROLE_R1]), ())):
                if " " + _NT[j] + " " in piena:
                    return j
        return -1
    scarti = {}
    for i, migliore, j_mig, sopra, _, uguale in risultati:
        j_r1 = r1(_NV[i])
        regole = [nome for nome, vero in (("uguale", uguale >= 0), ("R3", migliore >= SOGLIA_RAPPORTO),
                                          ("R1", j_r1 >= 0)) if vero]
        if not regole:
            continue
        j = uguale if uguale >= 0 else (j_mig if migliore >= SOGLIA_RAPPORTO else j_r1)
        scarti[i] = {"validation": i, "materia": val[i]["subject"], "regole": regole,
                     "rapporto_max": round(migliore, 4), "test": j, "materia_test": test[j]["subject"],
                     "test_sopra_soglia": sopra, "stesse_scelte": list(val[i]["choices"]) == list(test[j]["choices"]),
                     "stessa_risposta": (list(val[i]["choices"]) == list(test[j]["choices"])
                                         and int(val[i]["answer"]) == int(test[j]["answer"])),
                     "domanda_validation": val[i]["question"][:300], "domanda_test": test[j]["question"][:300]}
    stat = {"coppie_totali": len(val) * len(test), "coppie_valutate_con_ratio": sum(r[4] for r in risultati),
            "uguali": sum("uguale" in s["regole"] for s in scarti.values()),
            "R3": sum("R3" in s["regole"] for s in scarti.values()),
            "R1": sum("R1" in s["regole"] for s in scarti.values()),
            "solo_R1": sum(s["regole"] == ["R1"] for s in scarti.values()),
            "R3_non_uguali": sum("R3" in s["regole"] and "uguale" not in s["regole"] for s in scarti.values()),
            "stesse_scelte": sum(s["stesse_scelte"] for s in scarti.values()),
            "stessa_risposta": sum(s["stessa_risposta"] for s in scarti.values()),
            "rapporto_max_fra_le_tenute": round(max((r[1] for r in risultati if r[0] not in scarti), default=0.0), 4)}
    return scarti, stat


def scrivi_atomico(p: Path, scrivi) -> None:
    tmp = p.with_name(p.name + ".tmp")
    scrivi(tmp)
    os.replace(tmp, p)


def main() -> None:
    import pyarrow.parquet as pq
    import torch
    from transformers import AutoTokenizer
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    val = pq.read_table(FILE_VAL).to_pylist()
    test = pq.read_table(FILE_TEST).to_pylist()
    log(f"validation {len(val)} domande, test {len(test)} (solo per la dedup)")
    if any(len(r["choices"]) != 4 for r in val + test):
        raise RuntimeError("domande con != 4 scelte: carica_mmlu le salterebbe, la forma cambierebbe")
    forma = verifica_forma(test)
    log(f"forma verificata byte per byte contro il banco: {', '.join(forma)}")

    scarti, stat = dedup_contro_test(val, test)
    log(f"dedup: scartate {len(scarti)} (uguali {stat['uguali']}, R3 {stat['R3']}, R1 {stat['R1']}, "
        f"solo R1 {stat['solo_R1']}); rapporto massimo fra le tenute {stat['rapporto_max_fra_le_tenute']}")
    tenute = [i for i in range(len(val)) if i not in scarti]
    random.Random(SEME).shuffle(tenute)

    tok = AutoTokenizer.from_pretrained(str(HF))
    flusso, confini, testi = [], [], []
    for i in tenute:
        testo = testo_con_risposta(val[i])
        flusso += tok(testo, add_special_tokens=False).input_ids
        confini.append(len(flusso))
        testi.append(testo)
    n_seq = len(flusso) // L_SEQ
    if n_seq < 1:
        raise RuntimeError(f"solo {len(flusso)} token: non basta una sequenza da {L_SEQ}")
    usati = n_seq * L_SEQ
    ids = torch.tensor(flusso[:usati], dtype=torch.int64).reshape(n_seq, L_SEQ)
    intere = sum(fine <= usati for fine in confini)
    sha_tok = hashlib.sha256(ids.numpy().astype("<i8").tobytes()).hexdigest()

    per_materia = defaultdict(lambda: {"validation": 0, "scartate": 0, "tenute": 0, "intere_nel_tensore": 0})
    for i, r in enumerate(val):
        per_materia[r["subject"]]["validation"] += 1
        per_materia[r["subject"]]["scartate" if i in scarti else "tenute"] += 1
    for k, i in enumerate(tenute):
        if confini[k] <= usati:
            per_materia[val[i]["subject"]]["intere_nel_tensore"] += 1
    try:
        commit = subprocess.run(["git", "-C", str(QUI), "rev-parse", "HEAD"], capture_output=True, text=True,
                                check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as e:
        commit = f"sconosciuto ({e})"

    scrivi_atomico(OUT_PT, lambda p: torch.save(ids, p))
    manifesto = {
        "creato": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "thor_v2/prepara_conoscenza.py", "commit_repo": commit,
        "scopo": "held-out di CONOSCENZA per la forgia Thor: domande MMLU validation con la risposta giusta, forma del banco",
        "file": str(OUT_PT), "forma_tensore": [n_seq, L_SEQ], "dtype": "int64",
        "sha256_token": sha_tok, "sha256_token_come": "sha256 dei byte int64 little-endian del tensore (n_seq, L_SEQ) in ordine C",
        "sha256_file_pt": sha256_file(OUT_PT),
        "seme": SEME, "L_SEQ": L_SEQ,
        "sorgente": {"dataset": "cais/mmlu, config all", "split": "validation", "file": str(FILE_VAL),
                     "sha256": sha256_file(FILE_VAL), "domande": len(val),
                     "materie": len({r["subject"] for r in val})},
        "test_solo_per_dedup": {"file": str(FILE_TEST), "sha256": sha256_file(FILE_TEST), "domande": len(test),
                                "nota": "nessun token del test entra nell'held-out: il test e' il banco di valutazione"},
        "forma_testo": 'f"{prompt_banco} {lettera_giusta}\\n\\n" come diagnosi_strati_mmlu.fase_testi; prompt_banco = '
                       "converti_multiple_choice.carica_mmlu (lm-eval, 0-shot), verificato byte per byte sui file del banco",
        "banco_verificato": forma,
        "tokenizzatore": {"cartella": str(HF), "sha256_tokenizer_json": sha256_file(HF / "tokenizer.json"),
                          "add_special_tokens": False, "come": "ogni domanda tokenizzata da sola, poi concatenate"},
        "domande": {"validation": len(val), "scartate_dedup": len(scarti), "usate": len(tenute),
                    "intere_nel_tensore": intere, "tagliate_in_coda": len(tenute) - intere},
        "token": {"totali": len(flusso), "nel_tensore": usati, "scartati_in_coda": len(flusso) - usati},
        "dedup": {"normalizzazione": "parole() del v2_sem: NFKC, minuscole, [^\\w]+ -> spazio, unite da uno spazio",
                  "regole": {"uguale": "domanda normalizzata identica a una del test",
                             "R3": f"difflib SequenceMatcher ratio >= {SOGLIA_RAPPORTO}, massimo dei due versi, "
                                   "contro tutte le domande del test, nessuna soglia di parole",
                             "R1": f"contiene intera una domanda del test di almeno {MIN_PAROLE_R1} parole"},
                  "prefiltri": "esatti: limite superiore di ratio() (quick_ratio / real_quick_ratio), scarto solo "
                               "se 5u <= 2T-2", **stat,
                  "dettaglio": [scarti[i] for i in sorted(scarti)]},
        "per_materia": dict(sorted(per_materia.items())),
        "prime_3_domande": [{"validation": tenute[k], "materia": val[tenute[k]]["subject"], "testo": testi[k]}
                            for k in range(min(3, len(tenute)))],
    }
    scrivi_atomico(OUT_JSON, lambda p: p.write_text(json.dumps(manifesto, ensure_ascii=False, indent=1),
                                                     encoding="utf-8"))
    log(f"scritti {OUT_PT} {tuple(ids.shape)} e {OUT_JSON}")
    verifica(tok, flusso, testi)


def verifica(tok, flusso: list, testi: list) -> None:
    """Rilettura indipendente dal disco: forma, dtype, sha256, contenuto, niente domande del test dentro."""
    import pyarrow.parquet as pq
    import torch
    m = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    ids = torch.load(OUT_PT, weights_only=True)
    assert isinstance(ids, torch.Tensor) and ids.dtype == torch.int64, (type(ids), getattr(ids, "dtype", None))
    assert list(ids.shape) == m["forma_tensore"] and ids.shape[1] == L_SEQ, ids.shape
    assert hashlib.sha256(ids.numpy().astype("<i8").tobytes()).hexdigest() == m["sha256_token"]
    assert sha256_file(OUT_PT) == m["sha256_file_pt"]
    assert ids.flatten().tolist() == flusso[:ids.numel()], "il tensore non e' il flusso tokenizzato"
    assert 0 <= int(ids.min()) and int(ids.max()) < len(tok), (int(ids.min()), int(ids.max()), len(tok))
    d = m["domande"]
    # il testo decodificato deve RIDARE le domande intere, nell'ordine mescolato (la coda tagliata puo' spezzare
    # un carattere multibyte, per questo si confrontano solo le domande intere)
    decodificato = tok.decode(ids.flatten().tolist())
    assert decodificato.startswith("".join(testi[:d["intere_nel_tensore"]])), "il tensore non ridà le domande"
    assert [x["testo"] for x in m["prime_3_domande"]] == testi[:3]
    assert d["validation"] == d["scartate_dedup"] + d["usate"] and d["usate"] == d["intere_nel_tensore"] + d["tagliate_in_coda"]
    # controllo indipendente dalla dedup: nessuna domanda del test (>= 8 parole) dentro il testo DECODIFICATO
    test = pq.read_table(FILE_TEST).to_pylist()
    piena = " " + " ".join(parole(decodificato)) + " "
    dentro = [j for j, r in enumerate(test) if len(parole(r["question"])) >= MIN_PAROLE_R1
              and " " + " ".join(parole(r["question"])) + " " in piena]
    assert not dentro, f"domande del test nel testo decodificato: {dentro[:10]}"
    log(f"VERIFICA OK: {ids.shape[0]} sequenze x {ids.shape[1]}, sha256 token {m['sha256_token']}, "
        f"0 domande del test (>= {MIN_PAROLE_R1} parole) nel testo decodificato")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit(f"uso: {sys.argv[0]} (nessun argomento; variabili THOR_CONOSCENZA_* per cambiare percorsi/seme)")
    main()
