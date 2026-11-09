#!/usr/bin/env python3
"""MMLU PER DOMANDA — dove perde B-150 rispetto a B, e contro base / v20 / Q2_K (26/9/2026).

## Perche' esiste
Serie MMLU sullo stesso file (mc_v2/mmlu_00000.bin, 2000 compiti): base (Thor-v20-densiv5, niente forgia globale) 75,55 ·
forgia B 71,95 · B-150 (stessa ricetta, 150 lotti invece di 32) 65,85 · v20 76,50 · Q2_K 78,15. Ma sull'errore degli strati
0-8 contro il teacher, misurato su «domanda + risposta giusta» di MMLU VALIDATION, B-150 e' ~10% MEGLIO di B in ogni finestra
(prova del 26/9 su 8 finestre). Prima di cambiare misuratore bisogna sapere COME B-150 perde i 6 punti: conoscenza sparsa su
tutte le materie, poche materie (dominio), o il meccanismo della scelta (una lettera).

## Cosa fa (sola lettura dei registri, sola CPU)
1. Per ogni registro di llama-perplexity --multiple-choice ricava la correttezza per domanda dall'accuratezza progressiva
   (riga «i<TAB>acc_i»): n_i = round(acc_i·i/100), c_i = n_i − n_{i−1}. Controlla: indici 1..2000 consecutivi, n_i intero
   entro 1e-4, c_i in {0,1}, somma = «Final result» e sigma stampati (formula di perplexity.cpp).
2. Prova che ogni registro viene dallo STESSO file: i registri non stampano il percorso, quindi (a) cerca nel copione che
   lo ha scritto la riga con «-f .../mc_v2/mmlu_00000.bin» e la redirezione verso quel registro, (b) misura la concordanza
   per domanda (kappa di Cohen) fra i modelli, con un controllo negativo su un file diverso (mmlu_02000).
3. Ordine: dimostra dal sorgente di llama.cpp usato (perplexity.cpp) che con --multiple-choice-tasks >= n_task i compiti si
   leggono in ordine e la riga i e' il compito i; poi rigenera ogni compito del file dal parquet del test di cais/mmlu
   (prepara_conoscenza.prompt_banco) e verifica byte per byte domanda, risposte ed etichetta → materia e lettera giusta.
4. Tabelle (a)-(e) e conclusione in thor_v2/risultati/mmlu_per_domanda.md; vettori per domanda in mmlu_per_domanda.json.

Statistica: intervalli di Wilson (95%) per le accuratezze; differenza appaiata con errore standard di McNemar
(sqrt(n01+n10−(n01−n10)²/n)/n); McNemar esatto (binomiale bilaterale sui discordanti) e chi² con correzione di continuita';
omogeneita' fra gruppi con chi² di Pearson e p per permutazione (seme fisso, N_PERM permutazioni); Mann-Whitney con
correzione per i pareggi; tendenza di Cochran-Armitage. Niente scipy: tutto implementato qui, verificabile.

Uso (confinato: sulla GPU gira una forgia):
    XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus \\
    systemd-run --user --scope -p CPUQuota=200% nice -n 19 env OMP_NUM_THREADS=2 \\
        @@ROOT:venv@@/bin/python3 @@ROOT:forge@@/thor_v2/strumenti/mmlu_per_domanda.py
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

QUI = Path(__file__).resolve().parent            # thor_v2/strumenti
THOR_V2 = QUI.parent
FUCINA = THOR_V2.parent
sys.path.insert(0, str(THOR_V2))
from prepara_conoscenza import FILE_TEST, LETTERE, leggi_banco, prompt_banco  # noqa: E402

BANCO = FUCINA / "mc_v2" / "mmlu_00000.bin"
SORGENTE_LLAMA = Path("@@ROOT:models@@/build/llamacpp-tq1-k1024/tools/perplexity/perplexity.cpp")
USCITA_MD = THOR_V2 / "risultati" / "mmlu_per_domanda.md"
USCITA_JSON = THOR_V2 / "risultati" / "mmlu_per_domanda.json"
N_COMPITI = 2000
SEME = 20260926
N_PERM = int(os.getenv("MMLU_PD_PERM", "100000"))
Z95 = 1.959963984540054
RIGA_F = "-f @@ROOT:forge@@/mc_v2/mmlu_00000.bin"

# (chiave, registro, copione che lo scrive, redirezione nel copione, descrizione)
MODELLI = [
    ("base", "stato/densi_v5/mmlu.log", "thor_v2/valuta_a3_densi.sh", '> "$D/mmlu.log"',
     "Thor-v20-densiv5, niente forgia globale"),
    ("B", "stato/v2_ThorT-scor150/mmlu.log", "thor_v2/catena_v2.sh", '> "$D/mmlu.log"',
     "Thor-T forgia scorrevole, 150 passi, 32 lotti (NOME=ThorT-scor150)"),
    ("B150", "stato/v2_ThorT-scor150-L150/mmlu.log", "thor_v2/catena_v2.sh", '> "$D/mmlu.log"',
     "stessa ricetta di B con 150 lotti (NOME=ThorT-scor150-L150)"),
    ("A", "stato/v2_ThorT-sep300/mmlu.log", "thor_v2/catena_v2.sh", '> "$D/mmlu.log"',
     "Thor-T forgia a finestre separate, 300 passi (NOME=ThorT-sep300)"),
    ("v20", "thor/mmlu_v20_downaggr3_00000.log", "thor/cancello_v20_downaggr3.sh", '> "$D/mmlu_${NOME}_00000.log"',
     "campione Thor-v20-downaggr3"),
    ("Q2_K", "thor/mmlu_rivale_q2k_00000.log", "thor/coda_gpu_rivale_q2k.sh", '> "$D/mmlu_${NOME}_00000.log"',
     "rivale Ornith-1.5-35B-A3B-Q2_K"),
    ("B[0-19]", "stato/bisezione_mmlu/mmlu_0-19.log", "thor_v2/bisezione_mmlu.sh", '> "$D/mmlu_$r.log"',
     "ibrido di diagnosi: base + esperti di B negli strati 0-19"),
    ("B[20-39]", "stato/bisezione_mmlu/mmlu_20-39.log", "thor_v2/bisezione_mmlu.sh", '> "$D/mmlu_$r.log"',
     "ibrido di diagnosi: base + esperti di B negli strati 20-39"),
]
# controllo negativo: stesso tipo di banco, file DIVERSO (compiti 2000-3999) → la concordanza deve crollare a ~0
CONTROLLO_NEGATIVO = ("Q2_K_L su mmlu_02000", "risultati_cloud/rival_q2k/mc_mmlu_02000.log")
RIFERIMENTI = ["base", "v20", "Q2_K"]          # modelli indipendenti da B e B150 per la «difficolta'» di una domanda

TOKEN_MEDI = [float("nan")]                    # riempito da verifica_token_lettera

RIGA = re.compile(r"^(\d+)\t(\d+\.\d+)\s*$")
FINALE = re.compile(r"Final result: (\d+\.\d+) \+/- (\d+\.\d+)")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for blocco in iter(lambda: f.read(1 << 20), b""):
            h.update(blocco)
    return h.hexdigest()


# ───────────────────────────── formattazione (virgola decimale) ─────────────────────────────
def fn(x: float, nd: int = 2) -> str:
    return "n/d" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}".replace(".", ",")


def fs(x: float, nd: int = 2) -> str:
    return "n/d" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:+.{nd}f}".replace(".", ",")


def fp(p: float) -> str:
    if p < 1e-4:
        return "< 0,0001"
    return f"{p:.4f}".replace(".", ",")


def tabella(intest: list, righe: list) -> str:
    out = ["| " + " | ".join(intest) + " |", "|" + "|".join("---" for _ in intest) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in righe]
    return "\n".join(out)


# ───────────────────────────── statistica (senza scipy) ─────────────────────────────
def wilson(k: int, n: int) -> tuple:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + Z95 * Z95 / n
    c = p + Z95 * Z95 / (2 * n)
    r = Z95 * math.sqrt(p * (1 - p) / n + Z95 * Z95 / (4 * n * n))
    return (c - r) / d, (c + r) / d


def binomiale_bilaterale(k: int, m: int) -> float:
    """p esatto bilaterale di k successi su m prove con p=1/2 (McNemar esatto / test del segno)."""
    if m == 0:
        return 1.0
    kk = min(k, m - k)
    return min(1.0, 2 * sum(math.comb(m, i) for i in range(kk + 1)) / 2 ** m)


def _gamma_q(a: float, x: float) -> float:
    """Gamma incompleta regolarizzata superiore Q(a, x) (serie per x < a+1, frazione continua altrimenti)."""
    if x <= 0:
        return 1.0
    lg = -x + a * math.log(x) - math.lgamma(a)
    if x < a + 1:
        ap, s = a, 1.0 / a
        d = s
        for _ in range(10000):
            ap += 1
            d *= x / ap
            s += d
            if abs(d) < abs(s) * 1e-15:
                break
        return max(0.0, 1.0 - s * math.exp(lg))
    tiny = 1e-300
    b = x + 1 - a
    c = 1 / tiny
    d = 1 / b
    h = d
    for i in range(1, 10000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-15:
            break
    return math.exp(lg) * h


def chi2_coda(x: float, gl: int) -> float:
    return _gamma_q(gl / 2, x / 2)


def mcnemar(a: np.ndarray, b: np.ndarray) -> dict:
    """a = modello di riferimento, b = modello confrontato (vettori 0/1 sulle stesse domande).
    n10 = a giusta e b sbagliata (perse da b), n01 = a sbagliata e b giusta (guadagnate da b)."""
    n = len(a)
    n11 = int(((a == 1) & (b == 1)).sum())
    n10 = int(((a == 1) & (b == 0)).sum())
    n01 = int(((a == 0) & (b == 1)).sum())
    n00 = int(((a == 0) & (b == 0)).sum())
    m = n10 + n01
    d = (n01 - n10) / n if n else float("nan")
    se = math.sqrt(max(0.0, m - (n01 - n10) ** 2 / n)) / n if n else float("nan")
    chi = (abs(n10 - n01) - 1) ** 2 / m if m else 0.0
    return {"n": n, "n11": n11, "n10": n10, "n01": n01, "n00": n00, "diff": d, "lo": d - Z95 * se, "hi": d + Z95 * se,
            "p_esatto": binomiale_bilaterale(n01, m), "chi2_cc": chi, "p_chi2": chi2_coda(chi, 1) if m else 1.0,
            "p_perdita_se_giusta": n10 / (n11 + n10) if (n11 + n10) else float("nan")}


def omogeneita(gruppi: np.ndarray, esito: np.ndarray, rng: np.random.Generator, n_perm: int = N_PERM) -> dict:
    """Il tasso di `esito` (0/1) e' lo stesso in tutti i gruppi? Chi² di Pearson 2xS (gl = S-1) e p per permutazione
    (si rimescolano gli esiti fra le unita', i gruppi restano fermi: test condizionato ai margini)."""
    etichette, g = np.unique(gruppi, return_inverse=True)
    s, m = len(etichette), len(g)
    y = esito.astype(np.float64)
    p = y.mean()
    if s < 2 or p in (0.0, 1.0):
        return {"chi2": 0.0, "gl": s - 1, "p_chi2": 1.0, "p_perm": 1.0, "gruppi": s, "unita": m}
    uno = np.zeros((m, s))
    uno[np.arange(m), g] = 1.0
    tot = uno.sum(0)

    def stat(ymat: np.ndarray) -> np.ndarray:
        k = ymat @ uno
        return ((k - tot * p) ** 2 / (tot * p * (1 - p))).sum(-1)

    oss = float(stat(y[None])[0])
    sopra, fatte = 0, 0
    while fatte < n_perm:
        r = min(2000, n_perm - fatte)
        ymat = rng.permuted(np.tile(y, (r, 1)), axis=1)
        sopra += int((stat(ymat) >= oss - 1e-9).sum())
        fatte += r
    return {"chi2": oss, "gl": s - 1, "p_chi2": chi2_coda(oss, s - 1), "p_perm": (sopra + 1) / (n_perm + 1),
            "gruppi": s, "unita": m}


def omogeneita_stratificata(gruppi: np.ndarray, esito: np.ndarray, strati: np.ndarray, rng: np.random.Generator,
                            n_perm: int = N_PERM) -> dict:
    """Come `omogeneita`, ma a parita' di STRATO (la lettera giusta): l'atteso di ogni unita' e' il tasso del suo strato,
    e gli esiti si rimescolano SOLO dentro lo strato. Statistica: somma sui gruppi di (O−E)²/V, V = Σ e(1−e).
    Risponde: tolto l'effetto della lettera, i gruppi (materie) differiscono ancora?"""
    et_g, g = np.unique(gruppi, return_inverse=True)
    et_s, s = np.unique(strati, return_inverse=True)
    y = esito.astype(np.float64)
    tasso = np.array([y[s == j].mean() for j in range(len(et_s))])
    e = tasso[s]
    uno = np.zeros((len(y), len(et_g)))
    uno[np.arange(len(y)), g] = 1.0
    E, V = e @ uno, (e * (1 - e)) @ uno
    ok = V > 0

    def stat(ymat: np.ndarray) -> np.ndarray:
        d = (ymat @ uno) - E
        return ((d[:, ok] ** 2) / V[ok]).sum(-1)

    O = y @ uno
    oss = float(stat(y[None])[0])
    per_strato = [np.nonzero(s == j)[0] for j in range(len(et_s))]
    sopra, fatte = 0, 0
    while fatte < n_perm:
        r = min(2000, n_perm - fatte)
        ymat = np.tile(y, (r, 1))
        for ii in per_strato:
            ymat[:, ii] = rng.permuted(ymat[:, ii], axis=1)
        sopra += int((stat(ymat) >= oss - 1e-9).sum())
        fatte += r
    return {"stat": oss, "gruppi": int(ok.sum()), "p_perm": (sopra + 1) / (n_perm + 1),
            "O": dict(zip(et_g.tolist(), O.tolist())), "E": dict(zip(et_g.tolist(), E.tolist()))}


def ranghi(x: np.ndarray) -> np.ndarray:
    ordine = np.argsort(x, kind="mergesort")
    xs = x[ordine]
    r = np.empty(len(x))
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        r[ordine[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return r


def mann_whitney(a: np.ndarray, b: np.ndarray) -> dict:
    """U di Mann-Whitney con correzione per i pareggi (approssimazione normale). prob = P(a > b) + P(a = b)/2."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n1, n2 = len(a), len(b)
    x = np.concatenate([a, b])
    r = ranghi(x)
    u1 = r[:n1].sum() - n1 * (n1 + 1) / 2
    n = n1 + n2
    _, t = np.unique(x, return_counts=True)
    var = n1 * n2 / 12 * ((n + 1) - (t ** 3 - t).sum() / (n * (n - 1)))
    z = (u1 - n1 * n2 / 2) / math.sqrt(var)
    return {"prob": u1 / (n1 * n2), "z": z, "p": math.erfc(abs(z) / math.sqrt(2))}


def armitage(k: np.ndarray, n: np.ndarray, s: np.ndarray) -> tuple:
    """Tendenza di Cochran-Armitage: z e p bilaterale per k eventi su n in gruppi ordinati con punteggi s."""
    k, n, s = (np.asarray(v, float) for v in (k, n, s))
    N, p = n.sum(), k.sum() / n.sum()
    t = (s * (k - n * p)).sum()
    v = p * (1 - p) * ((n * s * s).sum() - (n * s).sum() ** 2 / N)
    z = t / math.sqrt(v)
    return z, math.erfc(abs(z) / math.sqrt(2))


def kappa(a: np.ndarray, b: np.ndarray) -> float:
    po = float((a == b).mean())
    pa, pb = a.mean(), b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe)


# ───────────────────────────── dati ─────────────────────────────
def leggi_registro(p: Path) -> dict:
    testo = p.read_text(errors="replace")
    if f"there are {N_COMPITI} tasks in prompt" not in testo:
        raise RuntimeError(f"{p}: il registro non dice «there are {N_COMPITI} tasks in prompt»")
    if re.search(r"selecting \d+ random tasks", testo):
        raise RuntimeError(f"{p}: compiti estratti a caso (ordine perso)")
    idx, acc = [], []
    for r in testo.splitlines():
        m = RIGA.match(r)
        if m:
            idx.append(int(m.group(1)))
            acc.append(float(m.group(2)))
    if idx != list(range(1, N_COMPITI + 1)):
        raise RuntimeError(f"{p}: indici non 1..{N_COMPITI} consecutivi ({len(idx)} righe)")
    cum = [0]
    for i, a in zip(idx, acc):
        x = a * i / 100
        n = round(x)
        if abs(x - n) > 1e-4:
            raise RuntimeError(f"{p}: riga {i}: acc {a} non da' un numero intero di giuste ({x})")
        cum.append(n)
    c = np.diff(np.array(cum))
    if not set(c.tolist()) <= {0, 1}:
        raise RuntimeError(f"{p}: correttezze fuori da {{0,1}}: {sorted(set(c.tolist()))}")
    finali = FINALE.findall(testo)
    if not finali:
        raise RuntimeError(f"{p}: «Final result» assente")
    fin, sig = float(finali[-1][0]), float(finali[-1][1])
    k = int(c.sum())
    pp = k / N_COMPITI
    sig_calc = 100 * math.sqrt(pp * (1 - pp) / (N_COMPITI - 1))       # perplexity.cpp: sqrt(p(1-p)/(n_done-1))
    if abs(100 * pp - fin) > 5e-5 or abs(sig_calc - sig) > 1e-4:
        raise RuntimeError(f"{p}: somma {k} → {100 * pp:.4f} ± {sig_calc:.4f}, il registro dice {fin} ± {sig}")
    return {"c": c.astype(np.int8), "giuste": k, "finale": fin, "sigma": sig, "sha256": sha256_file(p)}


def prova_copione(copione: Path, redirezione: str) -> int:
    """Riga (1-based) del copione che lancia llama-perplexity su mmlu_00000.bin con TUTTI i compiti e scrive nel registro."""
    righe = copione.read_text().splitlines()
    for i, r in enumerate(righe):
        if RIGA_F in r and redirezione in r and "--multiple-choice-tasks 100000" in r and "--multiple-choice" in r:
            return i + 1
    raise RuntimeError(f"{copione}: nessuna riga con «{RIGA_F}» + «{redirezione}» + tutti i compiti")


def prova_ordine_sorgente() -> dict:
    """Dal sorgente usato dai copioni (B=@@ROOT:models@@/build/llamacpp-tq1-k1024/build/bin): i compiti si leggono in ordine
    quando --multiple-choice-tasks >= n_task, e la riga stampata e' «n_done<TAB>acc» nell'ordine del ciclo sui compiti."""
    s = SORGENTE_LLAMA.read_text().splitlines()
    cerca = {"tutti_in_ordine": "if (params.multiple_choice_tasks == 0 || params.multiple_choice_tasks >= (size_t)n_task) {",
             "stampa": 'LOG("%d\\t%.8lf\\n", n_done, 100.*n_correct/n_done);',
             "ciclo": "for (size_t i0 = 0; i0 < tasks.size(); i0++) {",
             "scelta_max_stretto": "if (cur_task.log_probs[s] > logprob_max_val) {",
             "logprob_primo_token": "float  log_prob  = std::log(first_probs[cur_task.seq_tokens[s][cur_task.common_prefix]]);",
             "domanda_spazio_risposta": 'task.seq_tokens.emplace_back(::common_tokenize(ctx, task.question + " " + answer, true));',
             "log_commentato": '//LOG("    <%s> : %g\\n", cur_task.mc1.answers[s].c_str(), log_prob/count);'}
    trovate = {}
    for k, v in cerca.items():
        n = [i + 1 for i, r in enumerate(s) if v in r]
        if len(n) != 1:
            raise RuntimeError(f"{SORGENTE_LLAMA}: «{v}» trovata {len(n)} volte")
        trovate[k] = n[0]
    return trovate


def carica_compiti() -> list:
    import pyarrow.parquet as pq
    compiti = leggi_banco(BANCO)
    if len(compiti) != N_COMPITI:
        raise RuntimeError(f"{BANCO}: {len(compiti)} compiti, attesi {N_COMPITI}")
    test = pq.read_table(FILE_TEST).to_pylist()
    out = []
    for k, (q, ans, lab) in enumerate(compiti):
        r = test[k]
        giusta = int(r["answer"])
        if q != prompt_banco(r) or ans != LETTERE or lab != [int(i == giusta) for i in range(4)]:
            raise RuntimeError(f"compito {k}: NON e' test[{k}] di cais/mmlu (domanda/risposte/etichetta diverse)")
        scelte = [str(x) for x in r["choices"]]
        lung = [len(x) for x in scelte]
        out.append({"prompt": q, "materia": r["subject"], "lettera": LETTERE[giusta], "car": len(q), "parole": len(q.split()),
                    "car_domanda": len(r["question"].strip()), "car_scelte": sum(lung),
                    "giusta_la_piu_lunga": int(lung[giusta] == max(lung) and lung.count(max(lung)) == 1)})
    return out


# ───────────────────────────── analisi ─────────────────────────────
def blocco_dati(reg: dict, prove: dict, ordine: dict, compiti: list, kap: dict, kneg: dict) -> str:
    righe = []
    for k, rl, cp, _, descr in MODELLI:
        r = reg[k]
        righe.append([k, descr, f"`{rl}`", f"`{cp}` riga {prove[k]}", fn(r["finale"], 2) + " ± " + fn(r["sigma"], 2),
                      f"{r['giuste']}/{N_COMPITI}", "ok"])
    materie = Counter(c["materia"] for c in compiti)
    s = ["## Dati e verifiche", "",
         f"- File dei compiti: `mc_v2/mmlu_00000.bin` (sha256 `{sha256_file(BANCO)[:16]}…`). Rigenerato compito per compito "
         f"dal test di cais/mmlu (`{FILE_TEST.name}`, sha256 `{sha256_file(FILE_TEST)[:16]}…`) con "
         "`prepara_conoscenza.prompt_banco`: domanda, risposte «A-D» ed etichetta uguali byte per byte per tutti i 2000. "
         "Il compito i del file e' `test[i]`.",
         f"- ⚠ Il file NON copre 57 materie: sono i primi 2000 del test, ordinato per materia → **{len(materie)} materie** "
         f"(l'ultima, electrical_engineering, parziale: {materie.get('electrical_engineering', 0)}/145). "
         "La posizione nel file e' quindi quasi tutta materia.",
         f"- Ordine registro ↔ file: `{SORGENTE_LLAMA}` (lo stesso albero del binario dei copioni). Con "
         f"`--multiple-choice-tasks 100000 ≥ 2000` i compiti si leggono in sequenza (riga {ordine['tutti_in_ordine']}), il "
         f"ciclo li valuta in ordine (riga {ordine['ciclo']}) e stampa «n_done<TAB>acc» per ciascuno (riga {ordine['stampa']}): "
         "la riga i del registro e' il compito i−1 del file. La risposta scelta e' l'argmax stretto del logprob "
         f"(riga {ordine['scelta_max_stretto']}): a pari merito vince A.",
         "- I registri NON stampano il percorso del file dei compiti. La prova che e' lo stesso file e': (1) la riga del "
         "copione che li scrive (colonna sotto), (2) la concordanza per domanda (kappa sotto), con un controllo negativo.",
         "- Correttezza per domanda: n_i = round(acc_i·i/100), c_i = n_i − n_{i−1}; verificato n_i intero (±1e-4), "
         "c_i ∈ {0,1}, somma e sigma uguali al «Final result» stampato.", "",
         tabella(["modello", "cos'e'", "registro", "copione che lo scrive", "Final result", "giuste ricostruite", "somma"],
                 righe), "",
         "Concordanza per domanda (kappa di Cohen; 0 = nessuna oltre il caso). Se un registro venisse da un altro file "
         "starebbe intorno a 0, come il controllo negativo.", ""]
    chiavi = [m[0] for m in MODELLI]
    righe_k = [[a] + [("—" if a == b else fn(kap[(a, b)], 3)) for b in chiavi] for a in chiavi]
    righe_k.append([f"*controllo: {CONTROLLO_NEGATIVO[0]}*"] + [fn(kneg[b], 3) for b in chiavi])
    s.append(tabella(["kappa"] + chiavi, righe_k))
    return "\n".join(s)


def blocco_2x2(mc: dict, nome_a: str, nome_b: str) -> str:
    righe = [[f"{nome_a} giusta", mc["n11"], mc["n10"], mc["n11"] + mc["n10"]],
             [f"{nome_a} sbagliata", mc["n01"], mc["n00"], mc["n01"] + mc["n00"]],
             ["totale", mc["n11"] + mc["n01"], mc["n10"] + mc["n00"], mc["n"]]]
    return tabella(["", f"{nome_b} giusta", f"{nome_b} sbagliata", "totale"], righe)


def per_lettera(reg: dict, compiti: list, rng) -> tuple:
    lett = np.array([c["lettera"] for c in compiti])
    chiavi = [m[0] for m in MODELLI]
    righe = []
    for L in LETTERE:
        sel = lett == L
        riga = [L, int(sel.sum())]
        for k in chiavi:
            c = reg[k]["c"][sel]
            riga.append(fn(100 * c.mean(), 1))
        righe.append(riga)
    om = {}
    riga_p = ["omogeneita' fra lettere (p perm.)", ""]
    for k in chiavi:
        o = omogeneita(lett, reg[k]["c"], rng)
        om[k] = o
        riga_p.append(fp(o["p_perm"]))
    righe.append(riga_p)
    t1 = tabella(["lettera giusta", "n"] + [f"{k} %" for k in chiavi], righe)

    b, x = reg["B"]["c"], reg["B150"]["c"]
    righe2, dati = [], {}
    for L in LETTERE:
        sel = lett == L
        mc = mcnemar(b[sel], x[sel])
        lo_b, hi_b = wilson(int(b[sel].sum()), int(sel.sum()))
        lo_x, hi_x = wilson(int(x[sel].sum()), int(sel.sum()))
        dati[L] = mc
        righe2.append([L, mc["n"], f"{fn(100 * b[sel].mean(), 1)} [{fn(100 * lo_b, 1)}–{fn(100 * hi_b, 1)}]",
                       f"{fn(100 * x[sel].mean(), 1)} [{fn(100 * lo_x, 1)}–{fn(100 * hi_x, 1)}]",
                       f"{fs(100 * mc['diff'], 1)} [{fs(100 * mc['lo'], 1)}; {fs(100 * mc['hi'], 1)}]",
                       mc["n10"], mc["n01"], fn(100 * mc["p_perdita_se_giusta"], 1), fp(mc["p_esatto"])])
    t2 = tabella(["lettera", "n", "B % [IC95]", "B150 % [IC95]", "Δ punti [IC95 appaiato]", "perse", "guadagnate",
                  "P(persa | B giusta) %", "p McNemar"], righe2)
    disc = (b != x)
    o_dir = omogeneita(lett[disc], (b[disc] == 1).astype(int), rng)
    giusteB = b == 1
    o_cond = omogeneita(lett[giusteB], (x[giusteB] == 0).astype(int), rng)
    return t1, t2, om, o_dir, o_cond, dati


def per_materia(reg: dict, compiti: list, rng, rif: str, conf: str) -> dict:
    mat = np.array([c["materia"] for c in compiti])
    a, x = reg[rif]["c"], reg[conf]["c"]
    righe = []
    for m in sorted(set(mat.tolist())):
        sel = mat == m
        mc = mcnemar(a[sel], x[sel])
        righe.append((m, mc, a[sel].mean(), x[sel].mean(), int(sel.sum()), int(np.nonzero(sel)[0][0])))
    righe.sort(key=lambda t: (t[1]["diff"], t[0]))
    disc = a != x
    o_dir = omogeneita(mat[disc], (a[disc] == 1).astype(int), rng)
    giuste = a == 1
    o_cond = omogeneita(mat[giuste], (x[giuste] == 0).astype(int), rng)
    # togliere le materie peggiori una alla volta: la perdita resta nelle altre?
    togli, rimaste = [], []
    for kk in range(0, 6):
        via = {r[0] for r in righe[:kk]}
        sel = ~np.isin(mat, list(via))
        mc = mcnemar(a[sel], x[sel])
        rimaste.append((kk, [r[0] for r in righe[:kk]], mc, int(sel.sum())))
    netto = sum(r[1]["n10"] - r[1]["n01"] for r in righe)
    return {"righe": righe, "o_dir": o_dir, "o_cond": o_cond, "rimaste": rimaste, "netto": netto,
            "in_calo": sum(r[1]["diff"] < 0 for r in righe), "in_crescita": sum(r[1]["diff"] > 0 for r in righe),
            "pari": sum(r[1]["diff"] == 0 for r in righe)}


def tabella_materie(pm: dict, rif: str, conf: str) -> str:
    righe = []
    for m, mc, pa, px, n, _ in pm["righe"]:
        righe.append([m, n, fn(100 * pa, 1), fn(100 * px, 1),
                      f"{fs(100 * mc['diff'], 1)} [{fs(100 * mc['lo'], 1)}; {fs(100 * mc['hi'], 1)}]",
                      mc["n10"], mc["n01"], mc["n10"] - mc["n01"], fn(100 * mc["p_perdita_se_giusta"], 1),
                      fp(mc["p_esatto"])])
    return tabella(["materia", "n", f"{rif} %", f"{conf} %", "Δ punti [IC95 appaiato]", "perse", "guadagnate", "netto",
                    f"P(persa | {rif} giusta) %", "p McNemar"], righe)


def gruppi_lunghezza_posizione(reg: dict, compiti: list, rng) -> dict:
    b, x = reg["B"]["c"], reg["B150"]["c"]
    car = np.array([c["car"] for c in compiti], float)
    parole = np.array([c["parole"] for c in compiti], float)
    pos = np.arange(N_COMPITI, dtype=float)
    mat = np.array([c["materia"] for c in compiti])
    lunga = np.array([c["giusta_la_piu_lunga"] for c in compiti])
    gruppi = {"perse (B sì, B150 no)": (b == 1) & (x == 0), "guadagnate (B no, B150 sì)": (b == 0) & (x == 1),
              "giuste per entrambi": (b == 1) & (x == 1), "sbagliate per entrambi": (b == 0) & (x == 0)}
    righe_g = []
    for nome, sel in gruppi.items():
        q = np.percentile(car[sel], [25, 50, 75])
        righe_g.append([nome, int(sel.sum()), fn(car[sel].mean(), 0), f"{fn(q[0], 0)} / {fn(q[1], 0)} / {fn(q[2], 0)}",
                        fn(parole[sel].mean(), 1), fn(pos[sel].mean() + 1, 0), fn(np.median(pos[sel]) + 1, 0)])
    perse, tenute, guad = gruppi["perse (B sì, B150 no)"], gruppi["giuste per entrambi"], gruppi["guadagnate (B no, B150 sì)"]
    mw_car_pt = mann_whitney(car[perse], car[tenute])
    mw_car_pg = mann_whitney(car[perse], car[guad])
    mw_pos_pt = mann_whitney(pos[perse], pos[tenute])

    # tasso di perdita fra le domande che B sa, per quartile di lunghezza (quartili su tutte le 2000)
    qs = np.percentile(car, [25, 50, 75])
    quart = np.searchsorted(qs, car, side="left")           # 0..3: Q1 <= qs[0] < Q2 <= qs[1] < Q3 <= qs[2] < Q4
    giusteB = b == 1
    righe_q, kq, nq = [], [], []
    for i in range(4):
        sel = giusteB & (quart == i)
        k, n = int((x[sel] == 0).sum()), int(sel.sum())
        lo, hi = wilson(k, n)
        sel_g = (b == 0) & (quart == i)
        mc = mcnemar(b[quart == i], x[quart == i])
        estremi = (f"≤ {fn(qs[0], 0)}" if i == 0 else f"> {fn(qs[2], 0)}" if i == 3 else
                   f"{fn(qs[i - 1], 0)}–{fn(qs[i], 0)}")
        righe_q.append([f"Q{i + 1} ({estremi} car.)", int((quart == i).sum()), n, k,
                        f"{fn(100 * k / n, 1)} [{fn(100 * lo, 1)}–{fn(100 * hi, 1)}]",
                        int((x[sel_g] == 1).sum()), f"{fs(100 * mc['diff'], 1)} [{fs(100 * mc['lo'], 1)}; {fs(100 * mc['hi'], 1)}]"])
        kq.append(k)
        nq.append(n)
    z_q, p_q = armitage(np.array(kq), np.array(nq), np.arange(4))

    # posizione nel file: blocchi da 250 (≈ materia) e, dentro ogni materia, prima meta' contro seconda meta'
    righe_p, kp, np_ = [], [], []
    for i in range(0, N_COMPITI, 250):
        blocco = (pos >= i) & (pos < i + 250)
        sel = giusteB & blocco
        k, n = int((x[sel] == 0).sum()), int(sel.sum())
        lo, hi = wilson(k, n)
        mc = mcnemar(b[blocco], x[blocco])
        materie_blocco = sorted(set(mat[blocco].tolist()), key=lambda m: np.nonzero(mat == m)[0][0])
        righe_p.append([f"{i + 1}–{i + 250}", ", ".join(materie_blocco), n, k,
                        f"{fn(100 * k / n, 1)} [{fn(100 * lo, 1)}–{fn(100 * hi, 1)}]",
                        f"{fs(100 * mc['diff'], 1)} [{fs(100 * mc['lo'], 1)}; {fs(100 * mc['hi'], 1)}]"])
        kp.append(k)
        np_.append(n)
    z_p, p_p = armitage(np.array(kp), np.array(np_), np.arange(len(kp)))
    meta = np.zeros(N_COMPITI, int)
    for m in set(mat.tolist()):
        ii = np.nonzero(mat == m)[0]
        meta[ii[len(ii) // 2:]] = 1
    righe_m = []
    for h, nome in ((0, "prima meta' di ogni materia"), (1, "seconda meta' di ogni materia")):
        sel = giusteB & (meta == h)
        k, n = int((x[sel] == 0).sum()), int(sel.sum())
        lo, hi = wilson(k, n)
        righe_m.append([nome, n, k, f"{fn(100 * k / n, 1)} [{fn(100 * lo, 1)}–{fn(100 * hi, 1)}]"])
    o_meta = omogeneita(meta[giusteB], (x[giusteB] == 0).astype(int), rng)

    righe_l = []
    for v, nome in ((1, "la giusta e' la scelta piu' lunga"), (0, "la giusta NON e' la piu' lunga")):
        sel = giusteB & (lunga == v)
        k, n = int((x[sel] == 0).sum()), int(sel.sum())
        lo, hi = wilson(k, n)
        righe_l.append([nome, n, k, f"{fn(100 * k / n, 1)} [{fn(100 * lo, 1)}–{fn(100 * hi, 1)}]"])
    o_lunga = omogeneita(lunga[giusteB], (x[giusteB] == 0).astype(int), rng)
    return {"righe_g": righe_g, "mw": (mw_car_pt, mw_car_pg, mw_pos_pt), "righe_q": righe_q, "trend_q": (z_q, p_q),
            "righe_p": righe_p, "trend_p": (z_p, p_p), "righe_m": righe_m, "o_meta": o_meta, "righe_l": righe_l,
            "o_lunga": o_lunga, "quartili": qs}


def lettere_per_coppie(reg: dict, compiti: list, coppie: list, rng) -> tuple:
    """Per ogni coppia riferimento→confrontato: Δ per lettera giusta (IC appaiato) e, a confronto, quanto la perdita
    dipende dalla LETTERA e quanto dalla MATERIA (stessa statistica: omogeneita' di P(persa | rif. giusta))."""
    lett = np.array([c["lettera"] for c in compiti])
    mat = np.array([c["materia"] for c in compiti])
    righe, dati = [], {}
    for a, b in coppie:
        ra, rb = reg[a]["c"], reg[b]["c"]
        riga = [f"{a} → {b}", fs(100 * mcnemar(ra, rb)["diff"], 2)]
        d = {}
        for L in LETTERE:
            sel = lett == L
            mc = mcnemar(ra[sel], rb[sel])
            d[L] = mc
            riga.append(f"{fs(100 * mc['diff'], 1)} [{fs(100 * mc['lo'], 1)}; {fs(100 * mc['hi'], 1)}]")
        giuste = ra == 1
        ol = omogeneita(lett[giuste], (rb[giuste] == 0).astype(int), rng)
        om = omogeneita(mat[giuste], (rb[giuste] == 0).astype(int), rng)
        oms = omogeneita_stratificata(mat[giuste], (rb[giuste] == 0).astype(int), lett[giuste], rng)
        riga += [f"{fn(ol['chi2'] / ol['gl'], 1)} ({fp(ol['p_perm'])})", f"{fn(om['chi2'] / om['gl'], 1)} ({fp(om['p_perm'])})",
                 fp(oms["p_perm"])]
        righe.append(riga)
        dati[(a, b)] = {"lettere": d, "o_lett": ol, "o_mat": om, "o_mat_strat": oms}
    t = tabella(["riferimento → confrontato", "Δ tot.", "Δ A [IC95]", "Δ B [IC95]", "Δ C [IC95]", "Δ D [IC95]",
                 "lettere: chi²/gl (p perm.)", "materie: chi²/gl (p perm.)", "materie a parita' di lettera: p perm."],
                righe)
    return t, dati


def materie_contro_lettere(reg: dict, compiti: list, rng, rif: str = "B", conf: str = "B150") -> tuple:
    """Quante perse/guadagnate avrebbe ogni materia se contasse SOLO la sua composizione di lettere giuste (tassi per
    lettera presi da tutte le 2000), contro quante ne ha davvero."""
    lett = np.array([c["lettera"] for c in compiti])
    mat = np.array([c["materia"] for c in compiti])
    a, x = reg[rif]["c"], reg[conf]["c"]
    p_persa = {L: ((a == 1) & (x == 0) & (lett == L)).sum() / ((a == 1) & (lett == L)).sum() for L in LETTERE}
    p_guad = {L: ((a == 0) & (x == 1) & (lett == L)).sum() / ((a == 0) & (lett == L)).sum() for L in LETTERE}
    righe = []
    for m in sorted(set(mat.tolist()), key=lambda m: np.nonzero(mat == m)[0][0]):
        sel = mat == m
        comp = [int((sel & (lett == L)).sum()) for L in LETTERE]
        perse = int((sel & (a == 1) & (x == 0)).sum())
        guad = int((sel & (a == 0) & (x == 1)).sum())
        att_p = sum(((sel & (a == 1) & (lett == L)).sum()) * p_persa[L] for L in LETTERE)
        att_g = sum(((sel & (a == 0) & (lett == L)).sum()) * p_guad[L] for L in LETTERE)
        righe.append((m, int(sel.sum()), comp, perse, att_p, guad, att_g))
    righe.sort(key=lambda r: -((r[3] - r[5]) - (r[4] - r[6])))
    t = tabella(["materia", "n", "giuste A/B/C/D", "perse oss.", "perse attese (lettere)", "guadagnate oss.",
                 "guadagnate attese", "netto oss.", "netto atteso", "scarto"],
                [[m, n, "/".join(str(c) for c in comp), p, fn(ap, 1), g, fn(ag, 1), p - g, fn(ap - ag, 1),
                  fs((p - g) - (ap - ag), 1)] for m, n, comp, p, ap, g, ag in righe])
    giuste = a == 1
    oms = omogeneita_stratificata(mat[giuste], (x[giuste] == 0).astype(int), lett[giuste], rng)
    disc = a != x
    oms_dir = omogeneita_stratificata(mat[disc], (a[disc] == 1).astype(int), lett[disc], rng)
    return t, oms, oms_dir, p_persa, p_guad


def verifica_token_lettera(compiti: list) -> str:
    """Col tokenizzatore della sorgente di Thor (stesso vocabolario del GGUF): «prompt + ' ' + X» per X in A-D deve
    differire in UN solo token finale, cioe' la risposta e' un token e il prefisso comune arriva a «Answer:»."""
    hf = Path(os.getenv("THOR_TOKENIZER", "@@ROOT:models@@/gguf/thor-src-hf"))
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(hf))
    except Exception as e:  # noqa: BLE001 — la verifica e' in piu': se manca il tokenizzatore lo si dice, non si finge
        return f"(Verifica del token-lettera NON fatta: tokenizzatore {hf} non caricabile: {type(e).__name__}.)"
    finali = Counter()
    lunghezze = []
    for c in compiti:
        seq = [tok(c["prompt"] + " " + L, add_special_tokens=False).input_ids for L in LETTERE]
        lunghezze.append(len(seq[LETTERE.index(c["lettera"])]))
        n = min(len(x) for x in seq)
        pref = 0
        while pref < n and all(x[pref] == seq[0][pref] for x in seq):
            pref += 1
        code = tuple(len(x) - pref for x in seq)
        if code != (1, 1, 1, 1):
            raise RuntimeError(f"risposta NON di un token: finali {code} in «{c['prompt'][-40:]}»")
        finali[tuple(tok.convert_ids_to_tokens([x[-1] for x in seq]))] += 1
    if len(finali) != 1:
        raise RuntimeError(f"token-lettera diversi fra i compiti: {finali}")
    ((t, k),) = finali.items()
    TOKEN_MEDI[0] = float(np.mean(lunghezze))
    return (f"Verificato col tokenizzatore `{hf}` su tutti i {k} compiti: ogni risposta e' UN token "
            f"({', '.join(repr(x) for x in t)}) dopo un prefisso comune; «domanda + risposta giusta» e' in media "
            f"{fn(TOKEN_MEDI[0], 1)} token.")


def lettera_per_difficolta(reg: dict, compiti: list, rif: str, conf: str) -> str:
    """P(persa | rif. giusta) per lettera giusta e per difficolta' (quanti fra RIFERIMENTI sanno la domanda). Una perdita di
    conoscenza colpirebbe tutte le lettere; uno spostamento di preferenza colpisce le lettere sfavorite anche sulle
    domande che tutti sanno."""
    lett = np.array([c["lettera"] for c in compiti])
    d = sum(reg[k]["c"].astype(int) for k in RIFERIMENTI)
    a, x = reg[rif]["c"], reg[conf]["c"]
    righe = []
    for v in range(len(RIFERIMENTI) + 1):
        riga = [f"{v} su {len(RIFERIMENTI)}"]
        for L in LETTERE:
            sel = (d == v) & (lett == L) & (a == 1)
            k, n = int((x[sel] == 0).sum()), int(sel.sum())
            riga.append(f"{k}/{n} = {fn(100 * k / n, 1)}%" if n else "—")
        righe.append(riga)
    return tabella([f"{rif} → {conf}: la sanno"] + [f"lettera {L}: perse/{rif} giuste" for L in LETTERE], righe)


def difficolta(reg: dict) -> tuple:
    """Quanti dei modelli di riferimento (indipendenti da B e B150) sanno la domanda: 0..3."""
    b, x = reg["B"]["c"], reg["B150"]["c"]
    d = sum(reg[k]["c"].astype(int) for k in RIFERIMENTI)
    righe, dati = [], {}
    for v in range(len(RIFERIMENTI) + 1):
        sel = d == v
        mc = mcnemar(b[sel], x[sel])
        dati[v] = mc
        righe.append([f"{v} su {len(RIFERIMENTI)}", mc["n"], fn(100 * b[sel].mean(), 1), fn(100 * x[sel].mean(), 1),
                      mc["n10"], mc["n01"], mc["n10"] - mc["n01"],
                      fn(100 * mc["p_perdita_se_giusta"], 1) if mc["n11"] + mc["n10"] else "n/d"])
    return righe, dati


def sovrapposizione(reg: dict, rif: str, chiavi: list) -> str:
    """Fra le domande che `rif` sa: quante ne perde ciascun modello e quante perdite hanno in comune due modelli,
    contro l'atteso se perdessero domande a caso (indipendenza)."""
    giuste = reg[rif]["c"] == 1
    n = int(giuste.sum())
    perse = {k: (reg[k]["c"] == 0) & giuste for k in chiavi}
    righe = []
    for i, a in enumerate(chiavi):
        for bb in chiavi[i + 1:]:
            ka, kb = int(perse[a].sum()), int(perse[bb].sum())
            both = int((perse[a] & perse[bb]).sum())
            att = ka * kb / n
            righe.append([f"{a} · {bb}", ka, kb, both, fn(att, 1), fn(both / att, 2) if att else "n/d",
                          fn(100 * both / min(ka, kb), 1) if min(ka, kb) else "n/d"])
    return tabella(["coppia", "perse dal 1°", "perse dal 2°", "perse da entrambi", "atteso a caso", "rapporto",
                    "% delle perse del minore"], righe) + f"\n\n(domande che {rif} sa: {n})"


# ───────────────────────────── principale ─────────────────────────────
def main() -> None:
    rng = np.random.default_rng(SEME)
    ordine = prova_ordine_sorgente()
    log(f"ordine nel sorgente verificato: {ordine}")
    compiti = carica_compiti()
    log(f"compiti: {len(compiti)} verificati contro il test di cais/mmlu")
    reg, prove = {}, {}
    for k, rl, cp, redir, _ in MODELLI:
        prove[k] = prova_copione(FUCINA / cp, redir)
        reg[k] = leggi_registro(FUCINA / rl)
        log(f"{k}: {reg[k]['giuste']}/{N_COMPITI} = {reg[k]['finale']} (copione {cp} riga {prove[k]})")
    neg = leggi_registro(FUCINA / CONTROLLO_NEGATIVO[1])
    chiavi = [m[0] for m in MODELLI]
    kap = {(a, b): kappa(reg[a]["c"], reg[b]["c"]) for a in chiavi for b in chiavi if a != b}
    kneg = {b: kappa(neg["c"], reg[b]["c"]) for b in chiavi}

    mc_bx = mcnemar(reg["B"]["c"], reg["B150"]["c"])
    t_lett, t_lett2, om_lett, o_lett_dir, o_lett_cond, lett_dati = per_lettera(reg, compiti, rng)
    pm = per_materia(reg, compiti, rng, "B", "B150")
    lp = gruppi_lunghezza_posizione(reg, compiti, rng)
    diff_righe, diff_dati = difficolta(reg)
    pm_base_x = per_materia(reg, compiti, rng, "base", "B150")
    pm_base_b = per_materia(reg, compiti, rng, "base", "B")
    coppie = [("B", "B150"), ("base", "B"), ("base", "B150"), ("base", "A"), ("v20", "B"), ("v20", "B150"),
              ("base", "v20"), ("Q2_K", "B150"), ("B", "A"), ("base", "B[0-19]"), ("base", "B[20-39]")]
    mcs = {c: mcnemar(reg[c[0]]["c"], reg[c[1]]["c"]) for c in coppie}
    coppie_lett = [("B", "B150"), ("base", "B"), ("base", "B150"), ("base", "A"), ("B", "A"), ("base", "B[0-19]"),
                   ("base", "B[20-39]"), ("v20", "B150"), ("base", "v20"), ("base", "Q2_K")]
    t_lett_coppie, lett_coppie = lettere_per_coppie(reg, compiti, coppie_lett, rng)
    t_mat_lett, oms_bx, oms_bx_dir, p_persa_L, p_guad_L = materie_contro_lettere(reg, compiti, rng)
    token_lettera = verifica_token_lettera(compiti)
    log(token_lettera)

    # ── documento ──
    md = [f"# MMLU per domanda: come perde B-150 (e le forge) — {time.strftime('%d/%m/%Y %H:%M')}", "",
          "Generato da `thor_v2/strumenti/mmlu_per_domanda.py` (sola lettura dei registri, sola CPU). "
          f"Permutazioni: {N_PERM} (seme {SEME}). IC = 95%. «Δ» = confrontato − riferimento, in punti percentuali. "
          "«perse» = il riferimento la sa e il confrontato no; «guadagnate» = il contrario.", "",
          blocco_dati(reg, prove, ordine, compiti, kap, kneg), ""]

    md += ["## (c) B contro B-150: matrice 2x2 e McNemar", "", blocco_2x2(mc_bx, "B", "B150"), "",
           f"- Δ = {fs(100 * mc_bx['diff'], 2)} punti [IC95 appaiato {fs(100 * mc_bx['lo'], 2)}; {fs(100 * mc_bx['hi'], 2)}]; "
           f"netto {mc_bx['n10'] - mc_bx['n01']} domande = {mc_bx['n10']} perse − {mc_bx['n01']} guadagnate "
           f"su {mc_bx['n10'] + mc_bx['n01']} discordanti.",
           f"- McNemar esatto p = {fp(mc_bx['p_esatto'])}; chi² con correzione = {fn(mc_bx['chi2_cc'], 2)} (p = {fp(mc_bx['p_chi2'])}).",
           f"- Delle {mc_bx['n11'] + mc_bx['n10']} domande che B sa, B-150 ne sbaglia {mc_bx['n10']} "
           f"({fn(100 * mc_bx['p_perdita_se_giusta'], 1)}%); delle {mc_bx['n01'] + mc_bx['n00']} che B sbaglia, B-150 ne "
           f"indovina {mc_bx['n01']} ({fn(100 * mc_bx['n01'] / (mc_bx['n01'] + mc_bx['n00']), 1)}%). "
           f"Il cambiamento NON e' solo un calo: {mc_bx['n10'] + mc_bx['n01']} domande su {N_COMPITI} "
           f"({fn(100 * (mc_bx['n10'] + mc_bx['n01']) / N_COMPITI, 1)}%) cambiano esito.", ""]

    md += ["## (a) Accuratezza per lettera giusta", "",
           "Una preferenza per una lettera alzerebbe l'accuratezza sulle domande con quella lettera giusta e abbasserebbe "
           "le altre. Le domande con lettere diverse sono domande diverse, quindi conta il confronto APPAIATO B→B-150 "
           "(seconda tabella), non il livello assoluto.", "", t_lett, "", t_lett2, "",
           f"- Le perse si distribuiscono fra le lettere come le guadagnate? (fra le {o_lett_dir['unita']} discordanti) "
           f"chi² = {fn(o_lett_dir['chi2'], 2)}, gl {o_lett_dir['gl']}, p = {fp(o_lett_dir['p_chi2'])} (perm. {fp(o_lett_dir['p_perm'])}).",
           f"- P(persa | B giusta) uguale per le quattro lettere? chi² = {fn(o_lett_cond['chi2'], 2)}, gl {o_lett_cond['gl']}, "
           f"p = {fp(o_lett_cond['p_chi2'])} (perm. {fp(o_lett_cond['p_perm'])}).",
           f"- Come si sceglie: perplexity.cpp tokenizza «domanda + ' ' + lettera» (riga {ordine['domanda_spazio_risposta']}), "
           f"il punteggio parte dal logprob del primo token dopo il prefisso comune (riga {ordine['logprob_primo_token']}) e "
           f"vince il massimo (riga {ordine['scelta_max_stretto']}). {token_lettera} Quindi la scelta dipende SOLO dai "
           "quattro logit delle lettere all'ultima posizione del prompt: uno spostamento di quei quattro logit, a parita' "
           "di tutto il resto, cambia la risposta.", "",
           "### La stessa lettura per tutte le coppie", "",
           "Δ per lettera giusta (IC95 appaiato) e, a confronto, quanto P(persa | rif. giusta) dipende dalla lettera e "
           "quanto dalla materia: chi²/gl (1 = nessuna differenza oltre il caso) con p per permutazione. L'ultima colonna: "
           "le materie differiscono ancora a PARITA' di lettera (permutazioni dentro la lettera)?", "",
           t_lett_coppie, "",
           f"### Lettera × difficolta' (difficolta' = quanti fra {', '.join(RIFERIMENTI)} sanno la domanda)", "",
           lettera_per_difficolta(reg, compiti, "B", "B150"), "", lettera_per_difficolta(reg, compiti, "base", "B150"), ""]

    md += ["## (b) Per materia: B contro B-150 (ordinate per perdita)", "",
           "IC appaiato di McNemar; p McNemar esatto per materia. Con 15 materie la soglia di Bonferroni e' 0,05/15 = 0,0033.",
           "", tabella_materie(pm, "B", "B150"), "",
           f"- Materie in calo: {pm['in_calo']} su {len(pm['righe'])} (in crescita {pm['in_crescita']}, pari {pm['pari']}); "
           f"test del segno p = {fp(binomiale_bilaterale(pm['in_calo'], pm['in_calo'] + pm['in_crescita']))}.",
           f"- Omogeneita' della DIREZIONE (perse contro guadagnate fra le discordanti, per materia): chi² = "
           f"{fn(pm['o_dir']['chi2'], 2)}, gl {pm['o_dir']['gl']}, p = {fp(pm['o_dir']['p_chi2'])} (perm. {fp(pm['o_dir']['p_perm'])}).",
           f"- Omogeneita' di P(persa | B giusta) fra materie: chi² = {fn(pm['o_cond']['chi2'], 2)}, gl {pm['o_cond']['gl']}, "
           f"p = {fp(pm['o_cond']['p_chi2'])} (perm. {fp(pm['o_cond']['p_perm'])}).", "",
           "Togliendo le materie peggiori una alla volta, la perdita resta nelle altre?", ""]
    righe_r = []
    for kk, via, mc, n in pm["rimaste"]:
        righe_r.append([kk, ", ".join(via) if via else "—", n,
                        f"{fs(100 * mc['diff'], 2)} [{fs(100 * mc['lo'], 2)}; {fs(100 * mc['hi'], 2)}]",
                        mc["n10"] - mc["n01"], fp(mc["p_esatto"])])
    md += [tabella(["tolte", "materie tolte", "domande rimaste", "Δ punti [IC95]", "netto", "p McNemar"], righe_r), ""]
    top3 = pm["righe"][:3]
    netto_top3 = sum(r[1]["n10"] - r[1]["n01"] for r in top3)
    n_top3 = sum(r[4] for r in top3)
    md += [f"- Le 3 materie peggiori portano {netto_top3} delle {pm['netto']} domande nette perse "
           f"({fn(100 * netto_top3 / pm['netto'], 1)}%) con il {fn(100 * n_top3 / N_COMPITI, 1)}% delle domande.", "",
           "### Materie a parita' di lettera", "",
           "Perse e guadagnate attese in ogni materia se contasse SOLO quali lettere sono giuste in quella materia (tassi "
           "per lettera di tutte le 2000: P(persa | B giusta) = " +
           " · ".join(f"{L} {fn(100 * p_persa_L[L], 1)}%" for L in LETTERE) + "; P(guadagnata | B sbagliata) = " +
           " · ".join(f"{L} {fn(100 * p_guad_L[L], 1)}%" for L in LETTERE) + "). Ordinate per scarto (osservato − atteso).",
           "", t_mat_lett, "",
           f"- Materie diverse a parita' di lettera, P(persa | B giusta): statistica {fn(oms_bx['stat'], 2)} su "
           f"{oms_bx['gruppi']} materie, p perm. = {fp(oms_bx['p_perm'])} (senza tener conto della lettera: "
           f"p perm. = {fp(pm['o_cond']['p_perm'])}).",
           f"- Direzione fra le discordanti a parita' di lettera: statistica {fn(oms_bx_dir['stat'], 2)}, "
           f"p perm. = {fp(oms_bx_dir['p_perm'])} (senza: {fp(pm['o_dir']['p_perm'])}).", ""]

    md += ["## (d) Lunghezza della domanda e posizione nel file", "",
           "Lunghezza = caratteri del prompt intero del banco (descrizione della materia + domanda + scelte + «Answer:»).", "",
           tabella(["gruppo", "n", "car. media", "car. Q1 / mediana / Q3", "parole media", "posizione media",
                    "posizione mediana"], lp["righe_g"]), "",
           f"- Lunghezza, perse contro giuste per entrambi: P(persa piu' lunga) = {fn(lp['mw'][0]['prob'], 3)}, "
           f"z = {fn(lp['mw'][0]['z'], 2)}, p = {fp(lp['mw'][0]['p'])} (Mann-Whitney).",
           f"- Lunghezza, perse contro guadagnate: P = {fn(lp['mw'][1]['prob'], 3)}, z = {fn(lp['mw'][1]['z'], 2)}, "
           f"p = {fp(lp['mw'][1]['p'])}.",
           f"- Posizione, perse contro giuste per entrambi: P = {fn(lp['mw'][2]['prob'], 3)}, z = {fn(lp['mw'][2]['z'], 2)}, "
           f"p = {fp(lp['mw'][2]['p'])}.", "",
           "Tasso di perdita fra le domande che B sa, per quartile di lunghezza (quartili sulle 2000):", "",
           tabella(["quartile", "domande", "B giuste", "perse da B150", "P(persa | B giusta) % [IC95]",
                    "guadagnate", "Δ punti nel quartile [IC95]"], lp["righe_q"]), "",
           f"- Tendenza con la lunghezza (Cochran-Armitage): z = {fn(lp['trend_q'][0], 2)}, p = {fp(lp['trend_q'][1])}.", "",
           "Per posizione nel file (blocchi da 250; il file e' ordinato per materia):", "",
           tabella(["compiti", "materie", "B giuste", "perse", "P(persa | B giusta) % [IC95]", "Δ punti nel blocco [IC95]"],
                   lp["righe_p"]), "",
           f"- Tendenza con la posizione: z = {fn(lp['trend_p'][0], 2)}, p = {fp(lp['trend_p'][1])}.", "",
           "Deriva lungo la corsa separata dalla materia: prima contro seconda meta' dentro ogni materia.", "",
           tabella(["meta'", "B giuste", "perse", "P(persa | B giusta) % [IC95]"], lp["righe_m"]), "",
           f"- Differenza: chi² = {fn(lp['o_meta']['chi2'], 2)}, p = {fp(lp['o_meta']['p_chi2'])} (perm. {fp(lp['o_meta']['p_perm'])}).", "",
           "Scorciatoia «la risposta piu' lunga»: se B-150 si appoggiasse a quella euristica perderebbe soprattutto le "
           "domande in cui la giusta NON e' la piu' lunga.", "",
           tabella(["caso", "B giuste", "perse", "P(persa | B giusta) % [IC95]"], lp["righe_l"]), "",
           f"- Differenza: chi² = {fn(lp['o_lunga']['chi2'], 2)}, p = {fp(lp['o_lunga']['p_chi2'])} (perm. {fp(lp['o_lunga']['p_perm'])}).", "",
           f"Difficolta' della domanda = quanti fra {', '.join(RIFERIMENTI)} la sanno (modelli indipendenti da B e B-150):", "",
           tabella(["la sanno", "n", "B %", "B150 %", "perse", "guadagnate", "netto", "P(persa | B giusta) %"], diff_righe),
           ""]

    md += ["## (e) Contro base e v20 (e gli altri registri sullo stesso file)", "",
           tabella(["riferimento → confrontato", "rif. %", "conf. %", "Δ punti [IC95 appaiato]", "perse", "guadagnate",
                    "netto", "P(persa | rif. giusta) %", "p McNemar"],
                   [[f"{a} → {b}", fn(reg[a]["finale"], 2), fn(reg[b]["finale"], 2),
                     f"{fs(100 * m['diff'], 2)} [{fs(100 * m['lo'], 2)}; {fs(100 * m['hi'], 2)}]", m["n10"], m["n01"],
                     m["n10"] - m["n01"], fn(100 * m["p_perdita_se_giusta"], 1), fp(m["p_esatto"])]
                    for (a, b), m in mcs.items()]), ""]
    mat = [c["materia"] for c in compiti]
    ordine_mat = [r[0] for r in pm["righe"]]
    righe_all = []
    for m in ordine_mat:
        sel = np.array([x == m for x in mat])
        righe_all.append([m, int(sel.sum())] + [fn(100 * reg[k]["c"][sel].mean(), 1) for k in chiavi])
    md += ["Accuratezza per materia di tutti i modelli (stesso ordine della tabella (b)):", "",
           tabella(["materia", "n"] + [f"{k} %" for k in chiavi], righe_all), "",
           "### base → B-150 per materia", "", tabella_materie(pm_base_x, "base", "B150"), "",
           f"- In calo {pm_base_x['in_calo']} su {len(pm_base_x['righe'])}; direzione omogenea fra materie: chi² = "
           f"{fn(pm_base_x['o_dir']['chi2'], 2)}, gl {pm_base_x['o_dir']['gl']}, p = {fp(pm_base_x['o_dir']['p_chi2'])} "
           f"(perm. {fp(pm_base_x['o_dir']['p_perm'])}); P(persa | base giusta) omogenea: chi² = "
           f"{fn(pm_base_x['o_cond']['chi2'], 2)}, p = {fp(pm_base_x['o_cond']['p_chi2'])} (perm. {fp(pm_base_x['o_cond']['p_perm'])}).", "",
           "### base → B per materia", "", tabella_materie(pm_base_b, "base", "B"), "",
           f"- In calo {pm_base_b['in_calo']} su {len(pm_base_b['righe'])}; direzione omogenea fra materie: chi² = "
           f"{fn(pm_base_b['o_dir']['chi2'], 2)}, gl {pm_base_b['o_dir']['gl']}, p = {fp(pm_base_b['o_dir']['p_chi2'])} "
           f"(perm. {fp(pm_base_b['o_dir']['p_perm'])}); P(persa | base giusta) omogenea: chi² = "
           f"{fn(pm_base_b['o_cond']['chi2'], 2)}, p = {fp(pm_base_b['o_cond']['p_chi2'])} (perm. {fp(pm_base_b['o_cond']['p_perm'])}).", "",
           "### Le forge perdono le STESSE domande?", "",
           "Fra le domande che la base sa: perdite in comune fra due modelli contro l'atteso se ognuno perdesse domande a "
           "caso (rapporto 1 = nessuna sovrapposizione oltre il caso).", "",
           sovrapposizione(reg, "base", ["B", "B150", "A", "B[0-19]", "B[20-39]", "v20", "Q2_K"]), ""]

    # ── conclusione: costruita dai numeri calcolati sopra ──
    md += conclusione(mc_bx, lett_dati, lett_coppie, pm, oms_bx, oms_bx_dir, lp, diff_dati, reg, compiti, ordine)
    USCITA_MD.parent.mkdir(parents=True, exist_ok=True)
    tmp = USCITA_MD.with_name(USCITA_MD.name + ".tmp")
    tmp.write_text("\n".join(md) + "\n")
    os.replace(tmp, USCITA_MD)

    dati = {"generato": time.strftime("%Y-%m-%dT%H:%M:%S"), "file_compiti": str(BANCO), "sha256_compiti": sha256_file(BANCO),
            "materia": mat, "lettera": [c["lettera"] for c in compiti], "car": [c["car"] for c in compiti],
            "modelli": {k: {"registro": rl, "copione": cp, "riga_copione": prove[k], "sha256_registro": reg[k]["sha256"],
                            "finale": reg[k]["finale"], "giuste": reg[k]["giuste"],
                            "corrette": "".join(str(int(v)) for v in reg[k]["c"])}
                        for k, rl, cp, _, _ in MODELLI}}
    tmp = USCITA_JSON.with_name(USCITA_JSON.name + ".tmp")
    tmp.write_text(json.dumps(dati, ensure_ascii=False, indent=1))
    os.replace(tmp, USCITA_JSON)
    log(f"scritti {USCITA_MD} e {USCITA_JSON}")


def _ic(m: dict, nd: int = 1) -> str:
    return f"{fs(100 * m['diff'], nd)} [{fs(100 * m['lo'], nd)}; {fs(100 * m['hi'], nd)}]"


def conclusione(mc_bx, lett_dati, lett_coppie, pm, oms_bx, oms_bx_dir, lp, diff_dati, reg, compiti, ordine) -> list:
    """La lettura, con i numeri calcolati sopra. Le frasi di verdetto dipendono dai test (soglie scritte qui), non da
    un giudizio a mano: rigirando lo script su registri nuovi la conclusione cambia con i dati."""
    lett = np.array([c["lettera"] for c in compiti])
    L = lett_dati                                                   # B → B150 per lettera
    lb = lett_coppie[("B", "B150")]
    ol, om = lb["o_lett"], lb["o_mat"]
    r3, r5 = pm["rimaste"][3], pm["rimaste"][5]
    n_AB = int(np.isin(lett, ["A", "B"]).sum())
    perse_AB = L["A"]["n10"] + L["B"]["n10"]
    guad_CD = L["C"]["n01"] + L["D"]["n01"]
    salite = [k for k in LETTERE if L[k]["lo"] > 0]
    lettera = ol["p_perm"] < 0.001 and bool(salite) and (ol["chi2"] / ol["gl"]) > 3 * (om["chi2"] / om["gl"])
    dominio = oms_bx_dir["p_perm"] < 0.05
    diffusa = r3[2]["hi"] < 0 and r3[2]["p_esatto"] < 0.05
    lc = lett_coppie
    b20, b0 = lc[("base", "B[20-39]")]["lettere"]["A"], lc[("base", "B[0-19]")]["lettere"]["A"]
    v20l = lc[("base", "v20")]["o_lett"]
    q2l = lc[("base", "Q2_K")]["o_lett"]
    d3 = (sum(reg[k]["c"].astype(int) for k in RIFERIMENTI) == len(RIFERIMENTI))
    b, x = reg["B"]["c"], reg["B150"]["c"]

    def conta(sel):
        return int(((b == 1) & (x == 0) & sel).sum()), int(((b == 1) & sel).sum())

    def tasso(sel):
        k, n = conta(sel)
        return f"{k}/{n} = {fn(100 * k / n, 1)}%"

    def verso(d):
        giu = [k for k in LETTERE if d[k]["hi"] < 0]
        su = [k for k in LETTERE if d[k]["lo"] > 0]
        return f"toglie a: {'/'.join(giu) or 'nessuna'}; da' a: {'/'.join(su) or 'nessuna'}"

    kab, nab = conta(d3 & np.isin(lett, ["A", "B"]))
    kcd, ncd = conta(d3 & np.isin(lett, ["C", "D"]))
    wab, wcd = wilson(kab, nab), wilson(kcd, ncd)
    mobile = max(pm["righe"], key=lambda r: (r[1]["n10"] + r[1]["n01"]) / r[4])
    forge_lettere = all(lc[("base", k)]["o_lett"]["p_perm"] < 0.001 for k in ("B", "B150", "A"))
    sede_20_39 = b20["hi"] < 0 and b0["lo"] <= 0 <= b0["hi"]

    if lettera and not dominio:
        verdetto = ("**Verdetto: la perdita di B-150 e' legata alla LETTERA (meccanismo della scelta). Tocca quasi tutte "
                    "le materie perche' ogni materia ha risposte A e B, ma a parita' di lettera giusta il verso delle domande "
                    "che cambiano non differisce fra materie: non e' dominio. Quanta parte sia anche conoscenza, "
                    "la sola correttezza non lo puo' dire.**")
    elif lettera and dominio:
        verdetto = ("**Verdetto: la perdita di B-150 e' in gran parte legata alla LETTERA, ma anche a parita' di lettera "
                    "alcune materie perdono piu' di altre (dominio).**")
    elif diffusa:
        verdetto = "**Verdetto: la perdita di B-150 e' distribuita sulle materie e non legata a una lettera.**"
    else:
        verdetto = "**Verdetto: la perdita di B-150 e' concentrata in poche materie.**"

    righe = ["## Conclusione", "", verdetto, "",
             f"1. **Lettera.** Δ B→B-150 per lettera giusta: A {_ic(L['A'])} · B {_ic(L['B'])} · C {_ic(L['C'])} · "
             f"D {_ic(L['D'])} punti. Delle {mc_bx['n10']} domande perse, {perse_AB} ({fn(100 * perse_AB / mc_bx['n10'], 0)}%) "
             f"hanno giusta A o B, che sono il {fn(100 * n_AB / N_COMPITI, 0)}% del file; delle {mc_bx['n01']} guadagnate, "
             f"{guad_CD} ({fn(100 * guad_CD / mc_bx['n01'], 0)}%) hanno giusta C o D. P(persa | B giusta) dipende dalla "
             f"lettera con chi²/gl = {fn(ol['chi2'] / ol['gl'], 1)} (p perm. {fp(ol['p_perm'])}) e dalla materia con "
             f"chi²/gl = {fn(om['chi2'] / om['gl'], 1)} (p perm. {fp(om['p_perm'])}). Una perdita di conoscenza, da sola (stessa preferenza), non fa "
             "SALIRE nessuna lettera; qui " +
             (", ".join(f"{k} sale di {_ic(L[k])}" for k in salite) if salite else "nessuna sale") +
             (". B-150 ha spostato la preferenza fra i quattro token « A» … « D» rispetto a B." if salite else "."),
             f"2. **Materia.** {pm['in_calo']} materie su {len(pm['righe'])} in calo e, tolte le 3 peggiori, restano "
             f"{_ic(r3[2], 2)} punti (p {fp(r3[2]['p_esatto'])}); tolte le 5 peggiori {_ic(r5[2], 2)}"
             + (": la perdita non sta in poche materie." if diffusa else ".") + " Ma il VERSO delle domande che cambiano (perse contro guadagnate) non differisce fra le "
             f"materie una volta tenuta ferma la lettera giusta: p perm. {fp(oms_bx_dir['p_perm'])} (senza tenerla ferma "
             f"{fp(pm['o_dir']['p_perm'])}). Resta diverso fra materie QUANTO si muove (P(persa | B giusta) a parita' di "
             f"lettera: p perm. {fp(oms_bx['p_perm'])}), cioe' il margine: la materia che cambia esito piu' spesso e' "
             f"{mobile[0]} ({mobile[1]['n10']} perse e {mobile[1]['n01']} guadagnate su {mobile[4]}, B "
             f"{fn(100 * mobile[2], 0)}%). Gli scarti osservato−atteso della tabella «Materie a parita' di lettera» vengono "
             "da li'.",
             f"3. **Non** lunghezza (tendenza sui quartili z = {fn(lp['trend_q'][0], 2)}, p {fp(lp['trend_q'][1])}), **non** "
             f"posizione nel file (dentro le materie p perm. {fp(lp['o_meta']['p_perm'])}), **non** l'euristica della "
             f"risposta piu' lunga (p perm. {fp(lp['o_lunga']['p_perm'])}).",
             ("4. **Tutte le forge globali spostano le lettere; v20 no.** " if forge_lettere and v20l["p_perm"] >= 0.05
              else "4. **Lettere rispetto alla base.** ") + "Δ per lettera rispetto alla base (A/B/C/D): B "
             + " / ".join(fs(100 * lc[("base", "B")]["lettere"][k]["diff"], 1) for k in LETTERE)
             + " · B-150 " + " / ".join(fs(100 * lc[("base", "B150")]["lettere"][k]["diff"], 1) for k in LETTERE)
             + " · A " + " / ".join(fs(100 * lc[("base", "A")]["lettere"][k]["diff"], 1) for k in LETTERE)
             + f". v20 contro base: chi²/gl lettere {fn(v20l['chi2'] / v20l['gl'], 1)} (p perm. {fp(v20l['p_perm'])}); "
             f"Q2_K contro base {fn(q2l['chi2'] / q2l['gl'], 1)} (p perm. {fp(q2l['p_perm'])}). Lettere con IC tutto sotto "
             f"/ sopra zero rispetto alla base — B {verso(lc[('base', 'B')]['lettere'])}; B-150 "
             f"{verso(lc[('base', 'B150')]['lettere'])}; A {verso(lc[('base', 'A')]['lettere'])}. B e B-150 distorcono la "
             "preferenza in modi DIVERSI, e il confronto B→B-150 del punto 1 e' in gran parte la differenza fra le due "
             "distorsioni.",
             f"5. **Sulle domande che tutti e 3 i riferimenti ({', '.join(RIFERIMENTI)}) sanno**, B-150 perde, fra quelle che "
             f"B sa: giusta A {tasso(d3 & (lett == 'A'))} · B {tasso(d3 & (lett == 'B'))} · C {tasso(d3 & (lett == 'C'))} · "
             f"D {tasso(d3 & (lett == 'D'))}; cioe' A+B {kab}/{nab} = {fn(100 * kab / nab, 1)}% "
             f"[{fn(100 * wab[0], 1)}–{fn(100 * wab[1], 1)}] contro C+D {kcd}/{ncd} = {fn(100 * kcd / ncd, 1)}% "
             f"[{fn(100 * wcd[0], 1)}–{fn(100 * wcd[1], 1)}]. "
             + ("Le domande facili si perdono quasi solo quando la giusta e' una lettera sfavorita: e' la firma di una "
                "preferenza che scavalca la conoscenza." if kab * ncd > 3 * kcd * nab else
                "Le domande facili si perdono anche con le lettere favorite: non e' solo preferenza."),
             "6. **Incertezza.** La correttezza 0/1 non separa preferenza e conoscenza: un modello che ha perso "
             "conoscenza ma preferisce C/D indovina comunque le domande C/D, e uno spostamento di preferenza puo' da solo "
             "produrre un calo netto (si perdono domande sapute e se ne indovinano altre solo per la lettera). Quanti dei "
             f"{fn(-100 * mc_bx['diff'], 1)} punti (IC95 {fn(-100 * mc_bx['hi'], 2)}…{fn(-100 * mc_bx['lo'], 2)}) siano "
             "conoscenza NON si puo' dire da questi dati.",
             f"7. **Col fatto nuovo (B-150 ~10% meglio di B sugli strati 0-8 su «domanda + risposta giusta»).** Non c'e' "
             f"contraddizione: quell'errore pesa la lettera come UN token su {fn(TOKEN_MEDI[0], 0)} in media e guarda gli "
             "strati 0-8, mentre la scelta dipende solo dai quattro logit delle lettere all'ultima posizione, dopo tutti i "
             "40 strati. " + ("Negli ibridi di diagnosi di B lo spostamento contro A sta negli esperti degli strati 20-39: "
                              if sede_20_39 else "Ibridi di diagnosi di B, lettera A: ")
             + f"base→B[20-39] A {_ic(b20)}, base→B[0-19] A {_ic(b0)} (ibridi di B, non di B-150: per B-150 la sede non e' "
             "misurata). Un held-out che deve predire l'MMLU deve guardare la distribuzione delle quattro lettere in "
             "uscita, non l'errore nascosto dei primi strati.",
             "8. **Prove che separano preferenza e conoscenza (proposte, non eseguite; GPU dopo la forgia).** (i) Rigirare "
             "l'MMLU di B e B-150 stampando i quattro logprob per domanda: la riga c'e' gia', commentata, in "
             f"perplexity.cpp riga {ordine['log_commentato']}; da li' la preferenza per lettera di ogni modello e "
             "l'accuratezza tolta la preferenza (PriDe: Zheng et al., «Large Language Models Are Not Robust Multiple "
             "Choice Selectors», ICLR 2024, arXiv 2309.03882). (ii) Stesso banco con le scelte ruotate (4 file costruiti "
             "dal parquet su CPU): la conoscenza resta, la preferenza cambia lettera. (iii) Per la forgia: sulla "
             "validation MMLU gia' deduplicata, KL fra le distribuzioni delle quattro lettere dopo «Answer:» di "
             "studente e teacher, sull'uscita del modello.", ""]
    return righe


if __name__ == "__main__":
    main()
