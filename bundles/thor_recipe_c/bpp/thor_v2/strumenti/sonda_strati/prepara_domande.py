#!/usr/bin/env python3
"""SONDA LINEARE PER STRATO — fase 1 (CPU): le domande (28/9/2026, leva L5 di analisi_leve_conoscenza.md).

## Perche' esiste
Fonte: Cloe, arXiv 2608.28809, Tab. 7: una sonda lineare sugli stati di uno strato ritrova la risposta MMLU (43,76 % dal
maestro, 26,19 % ≈ caso dal ternario). Qui la stessa misura, strato per strato (tutti i 40), maestro Q8 contro Thor-T:
IN QUALE strato lo studente smette di «sapere» la risposta, per decidere se riforgiare anche 0-19 e dove.
Questo script prepara SOLO gli input (id dei prompt, etichette); gli stati li estrae estrai_stati.py (GPU), la sonda la
addestra sonda.py (CPU).

## Quali domande — MAI il banco
- MMLU `validation` (1531) e `dev` (285) di cais/mmlu (parquet in @@ROOT:models@@/gguf/tony-tern/conoscenza/all/).
  ⛔ Il `test` (14042, da cui vengono i 2000 compiti del banco mc_v2/mmlu_00000.bin) si legge SOLO per la dedup e per
  provare la forma del prompt: nessuna sua domanda entra.
- Dedup contro il test con le regole di prepara_conoscenza.py (UGUALE · R3 difflib >= 0,80 · R1), la STESSA funzione
  (importata, non copiata). Validation: ricalcolata e confrontata con gli scarti del manifesto heldout_conoscenza.json
  (devono coincidere). Dev: calcolata qui, poi dedup del dev anche contro la validation tenuta (una quasi-copia fra
  pieghe gonfierebbe la validazione incrociata).
- Forma del prompt = quella del banco: prepara_conoscenza.prompt_banco, rigenerata byte per byte sui file del banco
  (prepara_conoscenza.verifica_forma). Il prompt finisce con «Answer:»; lo stato letto dalla sonda e' quello dell'ultimo
  token del prompt, cioe' lo stato da cui il modello sceglie la lettera (llama-perplexity --multiple-choice tokenizza
  domanda + " " + lettera: la lettera e' UN token dopo il prompt, controllato qui su tutte le domande).
- Bandierina `vista_forgia`: le domande che cadono nelle prime SONDA_SEQ_VISTE (32 = 4 lotti x 8 della riforgia)
  sequenze di heldout_conoscenza.pt, cioe' quelle che la forgia MISURA (mai addestrate) per accettare le finestre.
  Selezione debolissima (poche decisioni si/no su ~270 domande in blocco), ma la sonda riporta anche il sottoinsieme
  senza di loro. Il flusso del held-out si ricostruisce e si CONTROLLA con lo sha256 dei token del manifesto.

## Quante domande servono (il conto)
Si misura, per strato, la differenza APPAIATA d'accuratezza della sonda maestro − studente sulle stesse domande.
Errore standard appaiato ≈ sqrt(q / N), q = frazione di domande discordi (una sonda giusta, l'altra no).
Lo scarto finale atteso e' ~7,4 punti (maestro 82,0 contro 74,65 sul banco); per localizzare dove comincia serve
vedere un terzo di quello scarto (~2,5 punti) a 2 sigma: SE <= 1,25 punti. Con q = 0,25 [S, sonde a meta' rete meno
concordi delle uscite] => N >= 0,25 / 0,0125^2 = 1600. Validation+dev dopo la dedup ne danno ~1600 (misurato dallo
script: vedi `conto` nel manifesto). La log-verosimiglianza della sonda (misura continua) ha varianza piu' bassa
dell'accuratezza: e' la misura principale di sonda.py, l'accuratezza la seconda.

## Tokenizzazione
Id della STESSA via di llama-perplexity: `e2e_dump_logit tokenizza` (common_tokenize, add_special=true,
parse_special=false; solo vocabolario), una domanda per chiamata, su TUTTE le domande. Controlli: vocabolario del
maestro == vocabolario dello studente (gguf-py: token, merges, pre, tipi); id llama.cpp == id del tokenizzatore HF su
tutte le domande (riportato, la verita' e' llama.cpp); lettere: HF su tutte, llama.cpp su una domanda per materia.

## Uscite ($SONDA_DIR/domande, default @@ROOT:depot@@/sonda_strati/domande)
  ids/q_NNNNN.i32   id del prompt (int32 grezzi), uno per domanda, nell'ordine di domande.json
  domande.json      manifesto: domande (k, split, indice, materia, risposta, n_token, vista_forgia, fnv1a64), id delle
                    lettere, dedup, forma verificata, conto, sha256 delle sorgenti. Scritto PER ULTIMO (atomico).

Uso (CPU confinata; niente GPU):
  XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus systemd-run --user --scope \\
    -p CPUQuota=200% nice -n 19 env OMP_NUM_THREADS=2 HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= \\
    python3 prepara_domande.py
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("THOR_CONOSCENZA_PROC", os.getenv("SONDA_PROC", "2"))

import hashlib  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

QUI = Path(__file__).resolve().parent
THOR_V2 = QUI.parents[1]
sys.path.insert(0, str(THOR_V2))
sys.path.insert(0, "@@ROOT:models@@/build/llamacpp-tq1-k1024/gguf-py")
import prepara_conoscenza as PC  # noqa: E402

SONDA_DIR = Path(os.getenv("SONDA_DIR", "@@ROOT:depot@@/sonda_strati"))
USCITA = Path(os.getenv("SONDA_DOMANDE_DIR", str(SONDA_DIR / "domande")))
BIN = Path(os.getenv("SONDA_DUMP_BIN", str(THOR_V2 / "e2e_dump_logit")))
VOCAB_GGUF = Path(os.getenv("SONDA_VOCAB_GGUF", "@@ROOT:models@@/gguf/tony-tern/Thor-T-EB3-passo_000020.gguf"))
MAESTRO_GGUF = Path(os.getenv("SONDA_MAESTRO_GGUF",
                              "@@ROOT:models@@/gguf/tony-tern/teacher-q8/Ornith-1.5-35B-A3B-Q8_0.gguf"))
HELDOUT_PT = PC.CARTELLA / "heldout_conoscenza.pt"
HELDOUT_JSON = PC.CARTELLA / "heldout_conoscenza.json"
FILE_DEV = PC.PARQUET / "dev-00000-of-00001.parquet"
SEQ_VISTE = int(os.getenv("SONDA_SEQ_VISTE", "32"))     # riforgia: --lotti-con 4 x --batch-seq 8
N_TOK_PROC = int(os.getenv("SONDA_TOK_PROC", "2"))
LETTERE = PC.LETTERE
CHIAVI_VOCAB = ["tokenizer.ggml.model", "tokenizer.ggml.pre", "tokenizer.ggml.tokens", "tokenizer.ggml.token_type",
                "tokenizer.ggml.merges", "tokenizer.ggml.add_bos_token", "tokenizer.ggml.bos_token_id",
                "tokenizer.ggml.eos_token_id"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fnv1a64(b: bytes) -> str:
    h = 1469598103934665603
    for x in b:
        h ^= x
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def impronta_vocabolario(gguf: Path) -> str:
    """sha256 delle chiavi del tokenizzatore lette con gguf-py (solo intestazione; i pesi restano su disco)."""
    from gguf import GGUFReader
    r = GGUFReader(str(gguf), "r")
    h = hashlib.sha256()
    for k in CHIAVI_VOCAB:
        f = r.fields.get(k)
        h.update(k.encode())
        if f is None:
            h.update(b"<assente>")
            continue
        for i in f.data:
            h.update(np.asarray(f.parts[i]).tobytes())
    return h.hexdigest()


def tokenizza_llama(testo: str, lavoro: Path) -> list:
    """id di common_tokenize(testo, add_special=true, parse_special=false), come llama-perplexity."""
    lavoro.mkdir(parents=True, exist_ok=True)
    try:
        (lavoro / "t.txt").write_bytes(testo.encode("utf-8"))
        r = subprocess.run([str(BIN), "tokenizza", str(VOCAB_GGUF), str(lavoro / "t.txt"), "2", str(lavoro / "o"), "0"],
                           capture_output=True, text=True, env={**os.environ, "HIP_VISIBLE_DEVICES": "",
                                                                "CUDA_VISIBLE_DEVICES": ""})
        if r.returncode != 0:
            raise RuntimeError(f"tokenizza rc={r.returncode}: {r.stderr[-400:]}")
        m = json.loads((lavoro / "o" / "tokenizza.json").read_text())
        if m.get("tolto_newline_finale"):
            raise RuntimeError("il prompt finiva con \\n: tokenizza lo toglie, il prompt non sarebbe quello del banco")
        return np.fromfile(lavoro / "o" / "token_tutti.i32", dtype=np.int32).tolist()
    finally:
        shutil.rmtree(lavoro, ignore_errors=True)


def ricostruisci_heldout(val: list, scarti_val: set, man: dict) -> dict:
    """Ricostruisce il flusso di heldout_conoscenza.pt come prepara_conoscenza.main e lo CONTROLLA (sha256 dei token
    del manifesto). Ritorna {indice validation: (inizio, fine)} in token del flusso."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(PC.HF / "tokenizer.json"))
    if PC.sha256_file(PC.HF / "tokenizer.json") != man["tokenizzatore"]["sha256_tokenizer_json"]:
        raise RuntimeError("tokenizer.json diverso da quello del held-out: il flusso non si puo' ricostruire")
    tenute = [i for i in range(len(val)) if i not in scarti_val]
    random.Random(int(man["seme"])).shuffle(tenute)
    flusso, span = [], {}
    for i in tenute:
        a = len(flusso)
        flusso += tok.encode(PC.testo_con_risposta(val[i]), add_special_tokens=False).ids
        span[i] = (a, len(flusso))
    n_seq, l_seq = man["forma_tensore"]
    usati = n_seq * l_seq
    sha = hashlib.sha256(np.asarray(flusso[:usati], dtype="<i8").tobytes()).hexdigest()
    if sha != man["sha256_token"]:
        raise RuntimeError(f"flusso ricostruito sha256 {sha} != manifesto {man['sha256_token']}: bandierine non affidabili")
    return {"span": span, "l_seq": l_seq, "n_seq": n_seq, "sha256_verificato": sha}


def main() -> None:
    import pyarrow.parquet as pq
    t0 = time.time()
    for p in (BIN, VOCAB_GGUF, MAESTRO_GGUF, HELDOUT_JSON, PC.FILE_VAL, PC.FILE_TEST, FILE_DEV):
        if not p.exists():
            raise FileNotFoundError(p)
    val = pq.read_table(PC.FILE_VAL).to_pylist()
    dev = pq.read_table(FILE_DEV).to_pylist()
    test = pq.read_table(PC.FILE_TEST).to_pylist()
    if any(len(r["choices"]) != 4 for r in val + dev + test):
        raise RuntimeError("domande con != 4 scelte")
    log(f"validation {len(val)}, dev {len(dev)}, test {len(test)} (test: solo dedup e forma)")

    forma = PC.verifica_forma(test)
    log(f"forma del prompt verificata byte per byte sul banco: {', '.join(forma)}")

    # ── vocabolari: maestro == studente (gli stessi id vanno a tutti e due) ──
    iv_s, iv_m = impronta_vocabolario(VOCAB_GGUF), impronta_vocabolario(MAESTRO_GGUF)
    if iv_s != iv_m:
        raise RuntimeError(f"vocabolario diverso fra {VOCAB_GGUF.name} e {MAESTRO_GGUF.name}")
    log(f"vocabolario studente == maestro ({iv_s[:16]}...)")

    # ── dedup ──
    man_h = json.loads(HELDOUT_JSON.read_text(encoding="utf-8"))
    if man_h["sorgente"]["sha256"] != PC.sha256_file(PC.FILE_VAL):
        raise RuntimeError("validation diversa da quella del held-out di conoscenza")
    if man_h["test_solo_per_dedup"]["sha256"] != PC.sha256_file(PC.FILE_TEST):
        raise RuntimeError("test diverso da quello del held-out di conoscenza")
    scarti_man = {d["validation"] for d in man_h["dedup"]["dettaglio"]}
    scarti_v, stat_v = PC.dedup_contro_test(val, test)
    if set(scarti_v) != scarti_man:
        raise RuntimeError(f"dedup validation ricalcolata ({len(scarti_v)}) != manifesto ({len(scarti_man)})")
    log(f"dedup validation ricalcolata == manifesto: {len(scarti_v)} scartate")
    scarti_d, stat_d = PC.dedup_contro_test(dev, test)
    log(f"dedup dev contro test: {len(scarti_d)} scartate")
    val_ten = [i for i in range(len(val)) if i not in scarti_v]
    dev_ten0 = [i for i in range(len(dev)) if i not in scarti_d]
    scarti_dv, stat_dv = PC.dedup_contro_test([dev[i] for i in dev_ten0], [val[i] for i in val_ten])
    dev_ten = [i for k, i in enumerate(dev_ten0) if k not in scarti_dv]
    log(f"dedup dev contro validation tenuta: {len(scarti_dv)} scartate; dev tenute {len(dev_ten)}")

    # ── domande della forgia (misurate, mai addestrate) ──
    ric = ricostruisci_heldout(val, scarti_v, man_h)
    confine = SEQ_VISTE * ric["l_seq"]
    viste = {i for i, (a, _) in ric["span"].items() if a < confine}
    log(f"held-out ricostruito e verificato (sha256 {ric['sha256_verificato'][:16]}...): {len(viste)} domande nelle "
        f"prime {SEQ_VISTE} sequenze")

    elenco = [("validation", i, val[i]) for i in val_ten] + [("dev", i, dev[i]) for i in dev_ten]
    # duplicati esatti (normalizzati) DENTRO l'insieme: si tiene il primo
    visti_testi, tenuti = set(), []
    for split, i, r in elenco:
        chiave = " ".join(PC.parole(r["question"])) + "||" + "|".join(" ".join(PC.parole(c)) for c in r["choices"])
        if chiave in visti_testi:
            continue
        visti_testi.add(chiave)
        tenuti.append((split, i, r))
    dup_interni = len(elenco) - len(tenuti)
    log(f"insieme: {len(tenuti)} domande ({dup_interni} duplicati esatti interni tolti)")

    # ── tokenizzazione (llama.cpp su tutte, HF come riscontro) ──
    from tokenizers import Tokenizer
    hf = Tokenizer.from_file(str(PC.HF / "tokenizer.json"))
    prompt = [PC.prompt_banco(r) for _, _, r in tenuti]
    lav = USCITA.parent / (USCITA.name + ".lavoro")
    shutil.rmtree(lav, ignore_errors=True)
    t_tok = time.time()
    with ThreadPoolExecutor(N_TOK_PROC) as ex:
        ids_ll = list(ex.map(lambda kp: tokenizza_llama(kp[1], lav / f"t{kp[0]:05d}"), enumerate(prompt)))
    log(f"tokenizzate {len(prompt)} domande con llama.cpp in {time.time() - t_tok:.0f} s")
    ids_hf = [hf.encode(p, add_special_tokens=False).ids for p in prompt]
    diverse_hf = [k for k in range(len(prompt)) if ids_hf[k] != ids_ll[k]]
    # lettere: la lettera e' UN token dopo il prompt, lo stesso per tutte le domande (HF su tutte)
    id_lettere = []
    for L in LETTERE:
        code = set()
        for k, p in enumerate(prompt):
            e = hf.encode(p + " " + L, add_special_tokens=False).ids
            if e[:-1] != ids_hf[k]:
                raise RuntimeError(f"domanda {k}: «prompt + ' {L}'» non estende il prompt di un solo token")
            code.add(e[-1])
        if len(code) != 1:
            raise RuntimeError(f"lettera {L}: id diversi fra le domande {sorted(code)[:5]}")
        id_lettere.append(code.pop())
    # lettere con llama.cpp: una domanda per materia
    campione, viste_m = [], set()
    for k, (_, _, r) in enumerate(tenuti):
        if r["subject"] not in viste_m:
            viste_m.add(r["subject"])
            campione.append(k)
    lavori = [(k, L) for k in campione for L in LETTERE]
    with ThreadPoolExecutor(N_TOK_PROC) as ex:
        estesi = list(ex.map(lambda kl: tokenizza_llama(prompt[kl[0]] + " " + kl[1], lav / f"l{kl[0]:05d}{kl[1]}"),
                             lavori))
    for (k, L), e in zip(lavori, estesi):
        if e != ids_ll[k] + [id_lettere[LETTERE.index(L)]]:
            raise RuntimeError(f"llama.cpp, domanda {k}, lettera {L}: {e[-3:]} non e' prompt + {id_lettere}")
    log(f"lettere {dict(zip(LETTERE, id_lettere))}: un token dopo il prompt (HF {len(prompt)} domande, llama.cpp "
        f"{len(campione)} materie x 4)")

    # ── scrittura ──
    tmp = USCITA.parent / (USCITA.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "ids").mkdir(parents=True)
    domande, lunghezze = [], []
    for k, ((split, i, r), ids) in enumerate(zip(tenuti, ids_ll)):
        a = np.asarray(ids, dtype=np.int32)
        (tmp / "ids" / f"q_{k:05d}.i32").write_bytes(a.tobytes())
        lunghezze.append(len(ids))
        domande.append({"k": k, "split": split, "indice": i, "materia": r["subject"], "risposta": int(r["answer"]),
                        "n_token": len(ids), "vista_forgia": bool(split == "validation" and i in viste),
                        "fnv1a64_ids": fnv1a64(a.tobytes()), "file": f"ids/q_{k:05d}.i32"})
    n = len(domande)
    L = np.asarray(lunghezze)
    etich = Counter(d["risposta"] for d in domande)
    q_ipotesi, se_obiettivo = 0.25, 0.0125
    try:
        commit = subprocess.run(["git", "-C", str(QUI), "rev-parse", "HEAD"], capture_output=True, text=True,
                                check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as e:
        commit = f"sconosciuto ({e})"
    man = {
        "creato": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "thor_v2/strumenti/sonda_strati/prepara_domande.py", "commit_repo": commit,
        "scopo": "sonda lineare per strato (leva L5): prompt MMLU validation+dev nella forma del banco, MAI il test",
        "n_domande": n, "per_split": dict(Counter(d["split"] for d in domande)),
        "vista_forgia": {"domande": sum(d["vista_forgia"] for d in domande), "seq_viste": SEQ_VISTE,
                         "come": "prime SEQ_VISTE sequenze di heldout_conoscenza.pt (carica_heldout_conoscenza prende "
                                 "i primi lotti); flusso ricostruito e verificato con lo sha256 del manifesto",
                         "sha256_token_heldout": ric["sha256_verificato"]},
        "etichette": {LETTERE[c]: etich[c] for c in range(4)},
        "maggioranza": round(max(etich.values()) / n, 4),
        "id_lettere": dict(zip(LETTERE, id_lettere)),
        "token_prompt": {"totale": int(L.sum()), "media": round(float(L.mean()), 1), "mediana": int(np.median(L)),
                         "p95": int(np.percentile(L, 95)), "max": int(L.max()), "min": int(L.min())},
        "tokenizzazione": {"via": "e2e_dump_logit tokenizza (common_tokenize add_special=true parse_special=false)",
                           "binario": str(BIN), "sha256_binario": PC.sha256_file(BIN),
                           "vocabolario_da": str(VOCAB_GGUF), "impronta_vocabolario": iv_s,
                           "vocabolario_maestro_uguale": True, "maestro": str(MAESTRO_GGUF),
                           "diverse_da_hf": len(diverse_hf), "diverse_da_hf_esempi": diverse_hf[:10]},
        "forma_prompt": "prepara_conoscenza.prompt_banco (lm-eval mmlu 0-shot), finisce con 'Answer:'",
        "banco_verificato": forma,
        "dedup": {"validation": {"scartate": len(scarti_v), "uguale_al_manifesto_heldout": True,
                                 **{k: v for k, v in stat_v.items()}},
                  "dev_contro_test": {"scartate": len(scarti_d), **stat_d,
                                      "dettaglio": [scarti_d[i] for i in sorted(scarti_d)]},
                  "dev_contro_validation": {"scartate": len(scarti_dv), **stat_dv},
                  "duplicati_esatti_interni": dup_interni},
        "conto": {"obiettivo": "SE della differenza APPAIATA d'accuratezza maestro-studente per strato <= 1,25 punti "
                               "(un terzo dello scarto finale atteso ~7,4 a 2 sigma)",
                  "formula": "SE ≈ sqrt(q/N), q = frazione di domande discordi",
                  "q_ipotesi_S": q_ipotesi, "N_richiesto": int(np.ceil(q_ipotesi / se_obiettivo ** 2)),
                  "N_disponibile": n, "SE_atteso_con_N": round(float(np.sqrt(q_ipotesi / n)) * 100, 2)},
        "sorgenti": {"validation": {"file": str(PC.FILE_VAL), "sha256": PC.sha256_file(PC.FILE_VAL)},
                     "dev": {"file": str(FILE_DEV), "sha256": PC.sha256_file(FILE_DEV)},
                     "test_solo_dedup_e_forma": {"file": str(PC.FILE_TEST), "sha256": PC.sha256_file(PC.FILE_TEST)},
                     "heldout_conoscenza": str(HELDOUT_JSON)},
        "secondi": round(time.time() - t0, 1),
        "domande": domande,
    }
    (tmp / "domande.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    # rilettura indipendente prima di rendere visibile
    for d in domande:
        a = np.fromfile(tmp / d["file"], dtype=np.int32)
        if fnv1a64(a.tobytes()) != d["fnv1a64_ids"] or a.size != d["n_token"]:
            raise RuntimeError(f"rilettura: {d['file']} diverso")
    if USCITA.exists():
        vecchia = USCITA.parent / (USCITA.name + f".vecchia-{int(time.time())}")
        USCITA.rename(vecchia)
        log(f"cartella precedente spostata in {vecchia}")
    tmp.rename(USCITA)
    shutil.rmtree(lav, ignore_errors=True)
    log(f"FATTO: {n} domande in {USCITA} · token medi {L.mean():.1f} (max {L.max()}) · maggioranza "
        f"{man['maggioranza']:.3f} · SE atteso {man['conto']['SE_atteso_con_N']} punti · HF diverse {len(diverse_hf)} · "
        f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
