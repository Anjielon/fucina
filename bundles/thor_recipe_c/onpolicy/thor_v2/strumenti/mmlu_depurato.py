#!/usr/bin/env python3
"""MMLU DEPURATO — quanta parte della perdita B → B-150 e' preferenza per la lettera e quanta conoscenza (26/9/2026).

## Perche' esiste
Su mc_v2/mmlu_00000.bin (primi 2000 compiti del test cais/mmlu, 15 materie) B-150 perde 6,1 punti contro B, ma per
lettera giusta: A -19,0 · B -15,4 · C -1,6 · D +8,7 (risultati/mmlu_per_domanda.md). E' la firma di uno spostamento di
preferenza fra i token ' A'..' D'; coi soli 0/1 non si separa dalla conoscenza persa. La copia isolata di
llama-perplexity (@@ROOT:models@@/gguf/tony-tern/mmlu_logprob, patch perplexity_mclp.patch: SOLO stampe) ha scritto per ogni
compito i 4 log p(lettera | prompt) — righe «MCLP» — ridando identici Final result e accuratezze progressive della catena.

## Il metodo, letto dall'articolo (non ricordato) — Zheng et al., «Large Language Models Are Not Robust Multiple Choice
## Selectors», ICLR 2024, arXiv 2309.03882v4, §3 (PDF scaricato e letto il 26/9)
- Eq. 3: P_obs(d_i | q, x^I) = Z^-1 · P_prior(d_i | q) · P_debiased(o_{f_I(i)} | q, x): la distribuzione osservata sugli ID
  delle opzioni (A/B/C/D) = preferenza a priori per l'ID × credenza sul CONTENUTO dell'opzione.
- Eq. 7 (stima PriDe della preferenza): P_prior(d_i | q) = softmax( (1/|I|) Σ_{I∈I} log P_obs(d_i | q, x^I) ), con I =
  permutazioni CICLICHE dei contenuti delle opzioni: servono n = 4 passaggi del modello per compito, coi contenuti
  ruotati. L'algoritmo 1 lo fa su K compiti (es. 5 %) e media le preferenze in una «preferenza globale».
- Eq. 8 (depurazione): P_debiased(o_i | q, x) ∝ P_obs(d_i | q, x) / P~_prior(d_i).
⇒ La forma ESATTA di PriDe (Eq. 7) richiede le permutazioni: coi 4 logprob dell'ordine originale NON si calcola.
  L'Eq. 8 invece si applica coi 4 logprob, data una preferenza. Uso la variante dichiarata «a sottrazione della
  preferenza media»: la media di Eq. 7 fatta sui COMPITI (ordine originale) invece che sulle permutazioni dello stesso
  compito. Con l'Eq. 3 e una preferenza globale, log P_obs(d_i|q) = log P_prior(d_i) + log P_deb(contenuto in posizione
  i | q) − log Z_q: mediando sui compiti il termine del contenuto e' uguale per le 4 posizioni SE il contenuto e'
  scambiabile fra le posizioni. Le giuste non sono bilanciate (A 462 · B 481 · C 503 · D 554): la media semplice darebbe
  a D anche la conoscenza del modello; percio' la stima principale e' la media BILANCIATA per lettera giusta.
  Varianti: media semplice senza etichette (come PriDe, che non usa etichette); stima INCROCIATA a 10 pieghe (la
  preferenza di un compito si stima sugli altri 9/10, niente adattamento ai compiti depurati): globale, per materia, per
  terzile della massa che il modello mette sulle 4 lettere. L'articolo stesso tratta la preferenza come dipendente dal
  compito (P_prior(d_i | q)) e la globale come approssimazione: le varianti per materia/massa misurano quanto regge.
  Proprieta' esatta: la scelta depurata e' INVARIANTE a qualsiasi spostamento costante dei 4 logit-lettera (controllata
  numericamente).

## Cosa produce (sola lettura dei registri, sola CPU) → thor_v2/risultati/mmlu_depurato.md (+ .json)
(a) accuratezza grezza = Final result di ogni registro; (b) preferenza per lettera (e massa sulle 4 lettere);
(c) accuratezza depurata con IC appaiato (bootstrap sui compiti, preferenza RI-STIMATA in ogni ricampionamento, anche
nelle varianti incrociate); (d) B contro B-150 grezza e depurata, per lettera e per materia, e la scomposizione dei
6,1 punti (a preferenza nulla, alle due preferenze vere, Shapley); (e) margine della giusta (grezzo e depurato) e
vittorie a coppie (non dipendono dalla scala dei logit). Revisione avversaria: (a) e (c) rifatti con codice indipendente
(solo stdlib, parsing diverso, IC col jackknife), piu' le stime puntuali delle varianti incrociate rifatte da capo.

Uso (CPU confinata):
    XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus \\
    systemd-run --user --scope -p CPUQuota=800% nice -n 19 env OMP_NUM_THREADS=8 \\
        python3 @@ROOT:forge@@/thor_v2/strumenti/mmlu_depurato.py
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import struct
import time
from pathlib import Path

import numpy as np

QUI = Path(__file__).resolve().parent            # thor_v2/strumenti
THOR_V2 = QUI.parent
FUCINA = THOR_V2.parent
SB = Path("@@ROOT:models@@/gguf/tony-tern/mmlu_logprob")
MAPPA = THOR_V2 / "risultati" / "mmlu_per_domanda.json"
USCITA_MD = THOR_V2 / "risultati" / "mmlu_depurato.md"
USCITA_JSON = THOR_V2 / "risultati" / "mmlu_depurato.json"
SHA_BANCO = "af987cc3369c15f90df9f390def043fd9fa4787e075b3b2170ff697a9dabbc6a"
N = 2000
LETTERE = "ABCD"
SEME = 20260926
N_BOOT = int(os.getenv("MMLU_DEP_BOOT", "10000"))
BLOCCO = 250
K_PIEGHE = 10
Z95 = 1.959963984540054

# (chiave, GGUF, registro della copia isolata con le righe MCLP, registro della catena)
MODELLI = [
    ("base", "Thor-v20-densiv5.gguf", SB / "base.log", FUCINA / "stato/densi_v5/mmlu.log"),
    ("B", "Thor-v2-ThorT-scor150.gguf", SB / "B.log", FUCINA / "stato/v2_ThorT-scor150/mmlu.log"),
    ("B150", "Thor-v2-ThorT-scor150-L150.gguf", SB / "B150.log", FUCINA / "stato/v2_ThorT-scor150-L150/mmlu.log"),
    ("v20", "Thor-v20-downaggr3.gguf", SB / "v20.log", FUCINA / "thor/mmlu_v20_downaggr3_00000.log"),
]
CHIAVI = [m[0] for m in MODELLI]
NOME = {"base": "base", "B": "B", "B150": "B-150", "v20": "v20"}
COPPIE = [("B", "B150"), ("base", "B"), ("base", "B150"), ("v20", "B"), ("v20", "B150"), ("base", "v20")]
# (chiave, descrizione, celle della preferenza, stima incrociata?)
VARIANTI = [
    ("glob", "globale bilanciata (principale)", "una", False),
    ("lib", "globale senza etichette", "lib", False),
    ("glob_x", "globale bilanciata, incrociata 10 pieghe", "una", True),
    ("mat_x", "per materia, incrociata 10 pieghe", "materia", True),
    ("massa_x", "per terzile di massa sulle 4 lettere, incrociata 10 pieghe", "massa", True),
]
NOME_VAR = {v[0]: v[1] for v in VARIANTI}
CESTI = [-math.inf, -4, -2, -1, -0.5, 0, 0.5, 1, 2, 4, math.inf]   # margine in nat (log naturale)

RE_ACC = re.compile(r"^(\d+)\t(\d+\.\d+)$")
RE_FIN = re.compile(r"Final result: ([0-9.]+ \+/- [0-9.]+)")


# ─────────────────────────────────────────── formattazione (virgola decimale)
def fn(x: float, d: int = 2) -> str:
    return f"{x:.{d}f}".replace(".", ",").replace("-", "−")


def fs(x: float, d: int = 2) -> str:
    return (f"{x:+.{d}f}").replace(".", ",").replace("-", "−")


def ic(lo: float, hi: float, d: int = 2, segno: bool = True) -> str:
    f = fs if segno else fn
    return f"[{f(lo, d)}; {f(hi, d)}]"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blocco in iter(lambda: fh.read(1 << 20), b""):
            h.update(blocco)
    return h.hexdigest()


def pct(x: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(x, [2.5, 97.5])
    return float(lo), float(hi)


def pct100(x: np.ndarray) -> tuple[float, float]:
    lo, hi = pct(x)
    return 100 * lo, 100 * hi


def giudizio(lo: float, hi: float) -> str:
    return "IC tutto sotto zero" if hi < 0 else ("IC tutto sopra zero" if lo > 0 else "IC con dentro lo zero")


# ─────────────────────────────────────────── lettura (calcolo principale)
def leggi_registro(p: Path) -> dict:
    """Righe MCLP (indice, giusta, scelta, 4 logprob, 4 conteggi di token), righe acc, Final result."""
    lp = np.full((N, 4), np.nan, dtype=np.float32)
    giusta = np.full(N, -9, dtype=np.int64)
    scelta = np.full(N, -9, dtype=np.int64)
    acc: list[str] = []
    finale = None
    atteso = 0
    for riga in p.read_text(encoding="utf-8").splitlines():
        if riga.startswith("MCLP\t"):
            c = riga.split("\t")
            if len(c) != 13 or int(c[1]) != atteso or c[4] != "4":
                raise SystemExit(f"⛔ {p}: riga MCLP anomala: {riga!r}")
            if c[9:13] != ["1", "1", "1", "1"]:
                raise SystemExit(f"⛔ {p}: compito {atteso} con risposte di piu' token: {c[9:13]}")
            giusta[atteso], scelta[atteso] = int(c[2]), int(c[3])
            lp[atteso] = np.array([float(x) for x in c[5:9]], dtype=np.float32)
            atteso += 1
        elif RE_ACC.match(riga):
            acc.append(riga)
        elif (m := RE_FIN.search(riga)):
            finale = m.group(1)
    if atteso != N or not np.isfinite(lp).all() or (lp > 0).any():
        raise SystemExit(f"⛔ {p}: {atteso} righe MCLP o logprob non finiti/positivi")
    return {"lp32": lp, "giusta": giusta, "scelta": scelta, "acc": acc, "finale": finale}


def leggi_registro_catena(p: Path) -> dict:
    """Registro della catena: solo righe acc e Final result (non ha righe MCLP)."""
    acc, finale = [], None
    for riga in p.read_text(encoding="utf-8").splitlines():
        if RE_ACC.match(riga):
            acc.append(riga)
        elif (m := RE_FIN.search(riga)):
            finale = m.group(1)
    if len(acc) != N or finale is None:
        raise SystemExit(f"⛔ {p}: {len(acc)} righe acc, Final {finale}")
    return {"acc": acc, "finale": finale}


def finale_f32(n_giuste: int, n: int) -> str:
    """Final result come perplexity.cpp: float p = 1.f*n_correct/n_done; sigma = sqrt(p*(1-p)/(n_done-1))."""
    f = np.float32
    p = f(f(1.0) * f(n_giuste)) / f(n)
    s = np.sqrt(p * (f(1) - p) / f(n - 1))
    return "%.4f +/- %.4f" % (float(f(100) * p), float(f(100) * s))


def logsm(x: np.ndarray) -> np.ndarray:
    """log-softmax sull'ultimo asse."""
    m = x.max(axis=-1, keepdims=True)
    return x - (m + np.log(np.exp(x - m).sum(axis=-1, keepdims=True)))


def mean_nan(a: np.ndarray, asse: int) -> np.ndarray:
    """Media che ignora i NaN senza avvisi (gruppi vuoti in un ricampionamento)."""
    ok = ~np.isnan(a)
    return np.where(ok, a, 0.0).sum(axis=asse) / np.maximum(ok.sum(axis=asse), 1)


def terzili_per_rango(x: np.ndarray) -> np.ndarray:
    """Terzile 0/1/2 per RANGO (x crescente; pari merito in ordine d'indice): definizione rifatta uguale nella revisione."""
    ordine = np.argsort(x, kind="stable")
    t = np.empty(N, dtype=np.int64)
    t[ordine] = np.arange(N) * 3 // N
    return t


# ─────────────────────────────────────────── il nucleo: preferenze e depurazione con pesi W (b × N)
def valuta(D: np.ndarray, oro: np.ndarray) -> dict:
    """D: (b, N, 4) punteggi. Scelta = argmax (primo massimo), margine, log P4(giusta), vittorie a coppie."""
    b = D.shape[0]
    idx = np.broadcast_to(oro[None, :, None], (b, N, 1))
    scelta = D.argmax(axis=2)
    dg = np.take_along_axis(D, idx, axis=2)[:, :, 0]
    altre = D.copy()
    np.put_along_axis(altre, idx, -np.inf, axis=2)
    lp4 = np.take_along_axis(logsm(D), idx, axis=2)[:, :, 0]
    # la giusta, messa a -inf in «altre», e' sempre «battuta» da dg (finito): si toglie quella vittoria finta
    coppie = ((dg[:, :, None] > altre).sum(axis=2) - 1) / 3.0
    return {"scelta": scelta, "giusta": scelta == oro[None, :], "margine": dg - altre.max(axis=2), "lp4": lp4,
            "coppie": coppie}


def prior_celle(W: np.ndarray, S: np.ndarray, cella: np.ndarray, n_celle: int, oro: np.ndarray,
                piega: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Preferenza (log-softmax della media BILANCIATA per lettera giusta di log P4) per cella, pesata con W.
    piega None: stima sugli stessi compiti; altrimenti per ogni piega f si stima sui compiti delle ALTRE pieghe.
    Ritorna la preferenza per compito (b, N, 4) e l'ultima tabella per cella (b, n_celle, 4)."""
    b = W.shape[0]
    CG = np.eye(n_celle * 4)[cella * 4 + oro]                                  # (N, 4·celle)
    X = (CG[:, :, None] * S[:, None, :]).reshape(N, n_celle * 16)
    out = np.empty((b, N, 4))
    lp = None
    for f in ([None] if piega is None else range(K_PIEGHE)):
        fuori = np.ones(N, dtype=bool) if f is None else piega != f
        dentro = np.arange(N) if f is None else np.nonzero(piega == f)[0]
        Wf = W[:, fuori]
        somme = (Wf @ X[fuori]).reshape(b, n_celle, 4, 4)
        conti = (Wf @ CG[fuori]).reshape(b, n_celle, 4)
        medie = np.where(conti[..., None] > 0, somme / np.maximum(conti, 1)[..., None], np.nan)
        lp = logsm(mean_nan(medie, asse=2))                                     # (b, celle, 4)
        out[:, dentro, :] = lp[:, cella[dentro], :]
    return out, lp


def aggrega(r: dict, nome: str, ev: dict, W: np.ndarray, G: np.ndarray) -> None:
    for q in ("margine", "lp4", "coppie"):
        v = np.broadcast_to(ev[q], W.shape)
        r[f"{q}_{nome}"] = (W * v).sum(axis=1) / N
    r[f"margine_lett_{nome}"] = ((W * np.broadcast_to(ev["margine"], W.shape)) @ G) / (W @ G)


def statistiche(W: np.ndarray, dati: dict, ctx: dict) -> dict:
    """Tutte le quantita' con i pesi W (conteggi di un ricampionamento; W = 1 = stima puntuale)."""
    G, M, oro = ctx["G"], ctx["M"], ctx["oro"]
    cntG, cntM = W @ G, W @ M
    out: dict = {}
    prior_glob = {}
    for k in CHIAVI:
        S = dati[k]["S"]
        cr = dati[k]["giusta_grezza"].astype(np.float64)
        r: dict = {"acc_grezza": W @ cr / N, "lett_grezza": (W @ (G * cr[:, None])) / cntG,
                   "mat_grezza": (W @ (M * cr[:, None])) / cntM, "logmassa": W @ dati[k]["logmassa"] / N}
        aggrega(r, "grezza", valuta(S[None], oro), W, G)
        for v, _, celle, incrociata in VARIANTI:
            if celle == "lib":
                lp = logsm((W @ S) / W.sum(axis=1, keepdims=True))
                PR = lp[:, None, :]
                r["prior_lib"] = lp
            else:
                c, nc = {"una": (np.zeros(N, dtype=np.int64), 1), "materia": (ctx["mat_idx"], ctx["n_mat"]),
                         "massa": (dati[k]["terzile"], 3)}[celle]
                PR, tab = prior_celle(W, S, c, nc, oro, ctx["piega"] if incrociata else None)
                if v == "glob":
                    r["prior"] = prior_glob[k] = tab[:, 0, :]
            ev = valuta(S[None] - PR, oro)
            r[f"acc_{v}"] = (W * ev["giusta"]).sum(axis=1) / N
            r[f"lett_{v}"] = ((W * ev["giusta"]) @ G) / cntG
            aggrega(r, v, ev, W, G)
            if v == "glob":
                r["mat_glob"] = ((W * ev["giusta"]) @ M) / cntM
        r["rstd_grezza"] = r["lett_grezza"].std(axis=1)       # RStd dell'articolo: dev. std dei richiami per lettera
        r["rstd_glob"] = r["lett_glob"].std(axis=1)
        out[k] = r
    # scambi di preferenza (globale): la conoscenza di y con la preferenza di x
    for a, bb in COPPIE:
        for x, y in ((a, bb), (bb, a)):
            D = dati[y]["S"][None, :, :] - prior_glob[y][:, None, :] + prior_glob[x][:, None, :]
            out[f"scambio_{y}_con_pref_{x}"] = (W * valuta(D, oro)["giusta"]).sum(axis=1) / N
    return out


def concatena(pezzi: list[dict]) -> dict:
    uscita = {}
    for k, v in pezzi[0].items():
        uscita[k] = concatena([p[k] for p in pezzi]) if isinstance(v, dict) else np.concatenate([p[k] for p in pezzi])
    return uscita


# ─────────────────────────────────────────── revisione avversaria: secondo calcolo, codice indipendente (solo stdlib)
def revisione_indipendente() -> dict:
    """Rifa' (a) e (c) SENZA usare niente del calcolo principale: rilegge i file con un'espressione regolare diversa,
    ricava la correttezza grezza dalle righe CUMULATIVE «n<TAB>acc» (non dalle righe MCLP), ricalcola il Final result
    in float32 con struct, rifa' la depurazione con cicli in puro Python e gli IC col JACKKNIFE (non col bootstrap).
    Rifa' da capo anche le stime puntuali delle varianti (senza etichette, incrociate globale/materia/massa)."""
    def f32(x: float) -> float:
        return struct.unpack("<f", struct.pack("<f", x))[0]

    def primo_max(v: list[float]) -> int:
        best = 0
        for s in range(1, len(v)):
            if v[s] > v[best]:
                best = s
        return best

    def jk(valori: list[float]) -> float:
        n = len(valori)
        med = sum(valori) / n
        return math.sqrt((n - 1) / n * sum((t - med) ** 2 for t in valori))

    def depura_celle(Sn, oro, cella, K):
        """Preferenza bilanciata per cella stimata sulle altre pieghe (compito q nella piega q % K); scelte depurate."""
        scelte = [0] * len(Sn)
        for f in range(K):
            somme, conti = {}, {}
            for q in range(len(Sn)):
                if q % K == f:
                    continue
                chiave = (cella[q], oro[q])
                conti[chiave] = conti.get(chiave, 0) + 1
                s = somme.setdefault(chiave, [0.0, 0.0, 0.0, 0.0])
                for l in range(4):
                    s[l] += Sn[q][l]
            for q in range(f, len(Sn), K):
                gruppi = [g for g in range(4) if (cella[q], g) in conti]
                pr = [sum(somme[(cella[q], g)][l] / conti[(cella[q], g)] for g in gruppi) / len(gruppi)
                      for l in range(4)]
                scelte[q] = primo_max([Sn[q][l] - pr[l] for l in range(4)])
        return scelte

    ris: dict = {}
    for chiave, _, reg, reg_cat in MODELLI:
        testo = reg.read_text(encoding="utf-8")
        testo_cat = reg_cat.read_text(encoding="utf-8")
        acc = re.findall(r"(?m)^(\d+)\t(\d+\.\d{8})$", testo)
        acc_cat = re.findall(r"(?m)^(\d+)\t(\d+\.\d{8})$", testo_cat)
        assert [int(i) for i, _ in acc] == list(range(1, N + 1)), chiave
        c, prec = [], 0
        for i, a in acc:
            x = float(a) * int(i) / 100.0
            r = int(round(x))
            assert abs(x - r) < 1e-4 and r - prec in (0, 1), (chiave, i)
            c.append(r - prec)
            prec = r
        p = f32(f32(float(sum(c))) / f32(float(N)))
        s = f32(math.sqrt(f32(f32(p * f32(1.0 - p)) / f32(float(N - 1)))))
        finale = "%.4f +/- %.4f" % (f32(100.0 * p), f32(100.0 * s))
        fin_copia = re.findall(r"Final result: (\S+ \+/- \S+)", testo)
        fin_cat = re.findall(r"Final result: (\S+ \+/- \S+)", testo_cat)
        righe = re.findall(r"(?m)^MCLP\t(\d+)\t(\d)\t(\d)\t4\t(\S+)\t(\S+)\t(\S+)\t(\S+)\t1\t1\t1\t1$", testo)
        assert [int(r[0]) for r in righe] == list(range(N)), chiave
        oro = [int(r[1]) for r in righe]
        L = [[float(x) for x in r[3:7]] for r in righe]
        grezza = [1 if primo_max(v) == oro[q] else 0 for q, v in enumerate(L)]
        Sn, logm = [], []
        for v in L:
            mx = max(v)
            lse = mx + math.log(sum(math.exp(t - mx) for t in v))
            Sn.append([t - lse for t in v])
            logm.append(lse)
        somme = [[0.0] * 4 for _ in range(4)]
        conti = [0] * 4
        for q in range(N):
            conti[oro[q]] += 1
            for l in range(4):
                somme[oro[q]][l] += Sn[q][l]
        media = [sum(somme[g][l] / conti[g] for g in range(4)) / 4.0 for l in range(4)]
        mx = max(media)
        lse = mx + math.log(sum(math.exp(t - mx) for t in media))
        prior = [t - lse for t in media]
        scelte_dep = [primo_max([Sn[q][l] - media[l] for l in range(4)]) for q in range(N)]
        dep = [1 if scelte_dep[q] == oro[q] else 0 for q in range(N)]
        # varianti (stime puntuali, da capo)
        media_lib = [sum(Sn[q][l] for q in range(N)) / N for l in range(4)]
        scelte_lib = [primo_max([Sn[q][l] - media_lib[l] for l in range(4)]) for q in range(N)]
        ordine = sorted(range(N), key=lambda q: (logm[q], q))
        terz = [0] * N
        for pos, q in enumerate(ordine):
            terz[q] = pos * 3 // N
        mappa = json.loads(MAPPA.read_text(encoding="utf-8"))
        mat = mappa["materia"]
        varianti = {"lib": scelte_lib, "glob_x": depura_celle(Sn, oro, [0] * N, K_PIEGHE),
                    "mat_x": depura_celle(Sn, oro, mat, K_PIEGHE), "massa_x": depura_celle(Sn, oro, terz, K_PIEGHE)}
        # jackknife: tolto il compito j, la preferenza si ri-stima e si depurano gli altri N-1
        loo_dep, loo_grezza = [], []
        ng = sum(grezza)
        for j in range(N):
            gj = oro[j]
            med = []
            for l in range(4):
                t = 0.0
                for g in range(4):
                    t += (somme[g][l] - Sn[j][l]) / (conti[g] - 1) if g == gj else somme[g][l] / conti[g]
                med.append(t / 4.0)
            m0, m1, m2, m3 = med
            giuste = 0
            for q in range(N):
                if q == j:
                    continue
                v = Sn[q]
                a0, a1, a2, a3 = v[0] - m0, v[1] - m1, v[2] - m2, v[3] - m3
                best, bv = 0, a0
                if a1 > bv:
                    best, bv = 1, a1
                if a2 > bv:
                    best, bv = 2, a2
                if a3 > bv:
                    best = 3
                if best == oro[q]:
                    giuste += 1
            loo_dep.append(giuste / (N - 1))
            loo_grezza.append((ng - grezza[j]) / (N - 1))
        ris[chiave] = {
            "grezza_da_acc": c, "grezza_da_mclp": grezza, "finale_ricalcolato": finale,
            "finale_copia": fin_copia[-1] if fin_copia else None, "finale_catena": fin_cat[-1] if fin_cat else None,
            "acc_uguali_catena": acc == acc_cat, "oro": oro, "prior": prior, "scelte_dep": scelte_dep, "dep": dep,
            "acc_dep": sum(dep) / N, "acc_grezza": sum(c) / N, "loo_dep": loo_dep, "loo_grezza": loo_grezza,
            "varianti": varianti, "terzile": terz,
            "acc_varianti": {v: sum(1 for q in range(N) if sc[q] == oro[q]) / N for v, sc in varianti.items()},
        }
    stime: dict = {}
    for k in CHIAVI:
        r = ris[k]
        stime[f"acc_dep_{k}"] = (r["acc_dep"], jk(r["loo_dep"]))
        stime[f"dep_meno_grezza_{k}"] = (r["acc_dep"] - r["acc_grezza"],
                                         jk([d - g for d, g in zip(r["loo_dep"], r["loo_grezza"])]))
    for a, b in COPPIE:
        ra, rb = ris[a], ris[b]
        dg = [y - x for x, y in zip(ra["loo_grezza"], rb["loo_grezza"])]
        dd = [y - x for x, y in zip(ra["loo_dep"], rb["loo_dep"])]
        stime[f"dgrezza_{a}_{b}"] = (rb["acc_grezza"] - ra["acc_grezza"], jk(dg))
        stime[f"ddep_{a}_{b}"] = (rb["acc_dep"] - ra["acc_dep"], jk(dd))
        stime[f"pref_{a}_{b}"] = ((rb["acc_grezza"] - ra["acc_grezza"]) - (rb["acc_dep"] - ra["acc_dep"]),
                                  jk([x - y for x, y in zip(dg, dd)]))
        # IC di McNemar della differenza grezza (come mmlu_per_domanda.py), dai soli 0/1 delle righe cumulative
        n01 = sum(1 for x, y in zip(ra["grezza_da_acc"], rb["grezza_da_acc"]) if x == 1 and y == 0)
        n10 = sum(1 for x, y in zip(ra["grezza_da_acc"], rb["grezza_da_acc"]) if x == 0 and y == 1)
        stime[f"mcnemar_{a}_{b}"] = ((n10 - n01) / N, math.sqrt(n01 + n10 - (n01 - n10) ** 2 / N) / N)
    ris["_stime"] = stime
    return ris


# ─────────────────────────────────────────── principale
def main() -> None:
    t0 = time.time()
    mappa = json.loads(MAPPA.read_text(encoding="utf-8"))
    assert mappa["sha256_compiti"] == SHA_BANCO
    materia = mappa["materia"]
    materie = sorted(set(materia), key=materia.index)
    mat_idx = np.array([materie.index(m) for m in materia])
    oro = np.array([LETTERE.index(x) for x in mappa["lettera"]])
    sha_banco = sha256(FUCINA / "mc_v2" / "mmlu_00000.bin")

    # ── lettura e controlli di (a)
    dati: dict = {}
    controlli: dict = {}
    for k, gguf, reg, reg_cat in MODELLI:
        r = leggi_registro(reg)
        rc = leggi_registro_catena(reg_cat)
        lp = r["lp32"]
        scelta_np = lp.argmax(axis=1)                 # primo massimo, come il ciclo di perplexity.cpp
        giusta_grezza = scelta_np == r["giusta"]
        # correttezza dalle righe cumulative della CATENA (non della copia)
        n_i = [round(float(x.split("\t")[1]) * (i + 1) / 100) for i, x in enumerate(rc["acc"])]
        c_cat = np.diff(np.array([0] + n_i)) == 1
        cc = {
            "gguf": gguf, "registro": str(reg), "registro_catena": str(reg_cat),
            "sha256_registro": sha256(reg), "sha256_registro_catena": sha256(reg_cat),
            "scelta_uguale_mclp": int((scelta_np == r["scelta"]).sum()),
            "giusta_uguale_mappa": int((r["giusta"] == oro).sum()),
            "corretta_uguale_catena": int((giusta_grezza == c_cat).sum()),
            "acc_uguali_catena": r["acc"] == rc["acc"],
            "n_giuste": int(giusta_grezza.sum()),
            "finale_ricalcolato": finale_f32(int(giusta_grezza.sum()), N),
            "finale_copia": r["finale"], "finale_catena": rc["finale"],
        }
        cc["finale_ok"] = cc["finale_ricalcolato"] == cc["finale_copia"] == cc["finale_catena"]
        if not (cc["scelta_uguale_mclp"] == N and cc["giusta_uguale_mappa"] == N and cc["corretta_uguale_catena"] == N
                and cc["acc_uguali_catena"] and cc["finale_ok"]):
            raise SystemExit(f"⛔ controlli di (a) falliti per {k}: {cc}")
        controlli[k] = cc
        lp64 = lp.astype(np.float64)
        mx = lp64.max(axis=1)
        logmassa = mx + np.log(np.exp(lp64 - mx[:, None]).sum(axis=1))      # log P(una delle 4 lettere)
        dati[k] = {"lp": lp64, "S": logsm(lp64), "giusta_grezza": giusta_grezza, "logmassa": logmassa,
                   "terzile": terzili_per_rango(logmassa)}
    ctx = {"oro": oro, "G": np.eye(4)[oro], "M": np.eye(len(materie))[mat_idx], "mat_idx": mat_idx,
           "n_mat": len(materie), "piega": np.arange(N) % K_PIEGHE}
    print(f"[{time.time() - t0:.1f}s] letti e controllati 4 registri", flush=True)

    # ── stima puntuale
    P0 = statistiche(np.ones((1, N)), dati, ctx)
    P = {k: ({q: v[0] for q, v in P0[k].items()} if isinstance(P0[k], dict) else P0[k][0]) for k in P0}

    # ── invarianza esatta: uno spostamento costante dei 4 logit-lettera non cambia la scelta depurata
    inv = {}
    rng_inv = np.random.default_rng(SEME + 1)
    for k in CHIAVI:
        spost = rng_inv.normal(0, 1.5, size=4)
        d2 = dict(dati)
        d2[k] = dict(dati[k])
        d2[k]["S"] = logsm(dati[k]["lp"] + spost[None, :])
        d2[k]["giusta_grezza"] = d2[k]["S"].argmax(axis=1) == oro
        Q = statistiche(np.ones((1, N)), d2, ctx)[k]
        prima = (dati[k]["S"] - P[k]["prior"][None, :]).argmax(axis=1)
        dopo = (d2[k]["S"] - Q["prior"][0][None, :]).argmax(axis=1)
        errore = (Q["prior"][0] - Q["prior"][0].mean()) - (P[k]["prior"] - P[k]["prior"].mean()) - (spost - spost.mean())
        inv[k] = {"spostamento": spost.tolist(), "acc_grezza_spostata": float(Q["acc_grezza"][0]),
                  "scelte_uguali": int((prima == dopo).sum()), "errore_prior_max": float(np.abs(errore).max())}

    # ── bootstrap appaiato: stessi compiti ricampionati per tutti i modelli, preferenza ri-stimata ogni volta
    rng = np.random.default_rng(SEME)
    pezzi = []
    for i in range(N_BOOT // BLOCCO):
        W = rng.multinomial(N, np.full(N, 1.0 / N), size=BLOCCO).astype(np.float64)
        pezzi.append(statistiche(W, dati, ctx))
        if (i + 1) % 4 == 0:
            print(f"[{time.time() - t0:.1f}s] bootstrap {(i + 1) * BLOCCO}/{N_BOOT}", flush=True)
    BT = concatena(pezzi)

    # ── revisione avversaria (codice indipendente)
    print(f"[{time.time() - t0:.1f}s] revisione indipendente (puro Python, jackknife)…", flush=True)
    RV = revisione_indipendente()
    print(f"[{time.time() - t0:.1f}s] revisione fatta", flush=True)

    scrivi(P, BT, RV, dati, ctx, controlli, inv, materie, sha_banco, time.time() - t0)


# ─────────────────────────────────────────── scrittura del rapporto
def scrivi(P, BT, RV, dati, ctx, controlli, inv, materie, sha_banco, durata) -> None:
    oro, mat_idx = ctx["oro"], ctx["mat_idx"]
    L: list[str] = []
    w = L.append
    ora = time.strftime("%d/%m/%Y %H:%M")

    def dlt(a, b, q):          # stima puntuale e campioni bootstrap di (b − a) per la quantita' q
        return P[b][q] - P[a][q], BT[b][q] - BT[a][q]

    def cella_pt(x, campioni, d=2, scala=100.0):
        return f"{fs(scala * x, d)} {ic(*[scala * v for v in pct(campioni)], d)}"

    # scomposizione (globale) per ogni coppia
    scomp = {}
    for a, b in COPPIE:
        g, gb = dlt(a, b, "acc_grezza")
        dg, db = dlt(a, b, "acc_glob")
        k_rif = P[f"scambio_{b}_con_pref_{a}"] - P[a]["acc_grezza"]
        k_rif_b = BT[f"scambio_{b}_con_pref_{a}"] - BT[a]["acc_grezza"]
        k_conf = P[b]["acc_grezza"] - P[f"scambio_{a}_con_pref_{b}"]
        k_conf_b = BT[b]["acc_grezza"] - BT[f"scambio_{a}_con_pref_{b}"]
        sh, shb = (k_rif + k_conf) / 2, (k_rif_b + k_conf_b) / 2
        scomp[(a, b)] = {"grezza": (g, gb), "dep": (dg, db), "pref": (g - dg, gb - db), "k_rif": (k_rif, k_rif_b),
                         "k_conf": (k_conf, k_conf_b), "shapley_con": (sh, shb), "shapley_pref": (g - sh, gb - shb)}
        for v in ("lib", "glob_x", "mat_x", "massa_x"):
            x, xb = dlt(a, b, f"acc_{v}")
            scomp[(a, b)][v] = (x, xb)
            scomp[(a, b)][f"pref_{v}"] = (g - x, gb - xb)

    w(f"# MMLU depurato dalla preferenza per lettera — B contro B-150, e contro base / v20 — {ora}")
    w("")
    w("Generato da `thor_v2/strumenti/mmlu_depurato.py` (sola lettura dei registri, sola CPU confinata: CPUQuota 800 %, "
      f"nice 19). Bootstrap appaiato sui compiti: {N_BOOT} ricampionamenti (seme {SEME}), preferenza RI-STIMATA in "
      "ogni ricampionamento (anche nelle stime incrociate); IC 95 % = percentili 2,5–97,5. «Δ» = confrontato − "
      "riferimento, in punti percentuali. Logprob in nat. Durata del calcolo: " + fn(durata / 60, 1) + " min.")
    w("")
    # ── dati
    w("## Dati e verifiche di (a)")
    w("")
    w(f"- File dei compiti: `mc_v2/mmlu_00000.bin`, sha256 `{sha_banco[:16]}…` (= quello della mappa verificata "
      "`risultati/mmlu_per_domanda.json`, da cui vengono materia e lettera giusta per compito).")
    w("- Righe «MCLP» (copia isolata `@@ROOT:models@@/gguf/tony-tern/mmlu_logprob`, patch di sola stampa): per ogni compito "
      "4 logprob = log p(token-lettera | prompt) sulla softmax dell'intero vocabolario; 1 token per scelta in 2000/2000 "
      "compiti (controllato su ogni riga).")
    w("")
    w("| modello | GGUF | registro con MCLP (sha256) | registro della catena | scelta = argmax | giusta = mappa | "
      "0/1 = catena | righe acc = catena | Final ricalcolato (float32) | Final copia | Final catena |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for k in CHIAVI:
        c = controlli[k]
        w(f"| {NOME[k]} | `{c['gguf']}` | `{Path(c['registro']).name}` (`{c['sha256_registro'][:12]}…`) | "
          f"`{Path(c['registro_catena']).relative_to(FUCINA)}` | {c['scelta_uguale_mclp']}/{N} | "
          f"{c['giusta_uguale_mappa']}/{N} | {c['corretta_uguale_catena']}/{N} | "
          f"{'identiche' if c['acc_uguali_catena'] else 'DIVERSE'} | {c['finale_ricalcolato']} | {c['finale_copia']} | "
          f"{c['finale_catena']} |")
    w("")
    w("«scelta = argmax»: l'argmax dei 4 logprob (primo massimo, come il ciclo di perplexity.cpp) coincide con la scelta "
      "stampata; «0/1 = catena»: la correttezza per compito coincide con quella ricavata dalle righe cumulative del "
      "registro della CATENA; il Final result ricalcolato con la formula float32 di perplexity.cpp coincide, sigma "
      "compresa, con copia e catena. **(a) regge per i 4 modelli.**")
    w("")
    # ── metodo
    w("## Il metodo (letto dall'articolo, arXiv 2309.03882v4, §3) e cosa si puo' calcolare qui")
    w("")
    w("- **Eq. 3** — P_obs(d_i | q, x^I) = Z⁻¹ · P_prior(d_i | q) · P_debiased(o_{f_I(i)} | q, x): preferenza per l'ID "
      "dell'opzione × credenza sul contenuto.")
    w("- **Eq. 7** (stima PriDe) — P_prior(d_i | q) = softmax((1/|I|) Σ_I log P_obs(d_i | q, x^I)) con I = permutazioni "
      "**cicliche dei contenuti**: 4 passaggi del modello per compito coi contenuti ruotati; l'algoritmo 1 lo fa su K "
      "compiti (es. 5 %) e media le preferenze in una preferenza globale per gli altri.")
    w("- **Eq. 8** (depurazione) — P_debiased(o_i | q, x) ∝ P_obs(d_i | q, x) / P̃_prior(d_i).")
    w("")
    w("**La forma esatta di PriDe (Eq. 7) richiede le permutazioni: coi 4 logprob dell'ordine originale non si calcola.** "
      "Qui si applica l'Eq. 8 con la **variante dichiarata «a sottrazione della preferenza media»**: la media di Eq. 7 "
      "fatta sui compiti (ordine originale) invece che sulle rotazioni dello stesso compito. Con una preferenza "
      "globale, log P_obs(d_i|q) = log P_prior(d_i) + log P_deb(contenuto in posizione i|q) − log Z_q, e il termine del "
      "contenuto mediato sui compiti e' uguale per le 4 posizioni solo se i contenuti sono scambiabili fra le posizioni. "
      "Le giuste non sono bilanciate (A 462 · B 481 · C 503 · D 554): la **stima principale e' la media BILANCIATA per "
      "lettera giusta** di log P4 (logprob rinormalizzati sulle 4 lettere): ogni posizione e' la giusta con lo stesso "
      "peso. Punteggio depurato = log P4(d_i) − log P_prior(d_i); scelta = argmax.")
    w("")
    w("Varianti (tutte con IC bootstrap): " + " · ".join(f"**{v}** = {d}" for v, d, _, _ in VARIANTI) + ". "
      f"«Incrociata»: compito i nella piega i mod {K_PIEGHE}; la preferenza per i compiti di una piega si stima sulle "
      "altre 9 (nessun adattamento ai compiti che si depurano). «Massa» = log P(una delle 4 lettere | prompt) del "
      "modello stesso; terzili per rango. L'articolo tratta la preferenza come dipendente dal compito, P_prior(d_i | q), "
      "e la globale come approssimazione (§3.3: somiglianza fra domini «moderata»): le varianti per materia e per massa "
      "misurano quanto l'approssimazione regge per ciascun modello.")
    w("")
    w("Proprieta' esatta, controllata sui dati: la scelta depurata e' invariante a ogni spostamento costante dei 4 "
      "logit-lettera (sommato a tutti i compiti, la preferenza stimata si sposta dello stesso vettore). Spostamento "
      "casuale N(0; 1,5) per lettera:")
    w("")
    w("| modello | spostamento A/B/C/D (nat) | acc. grezza dopo lo spostamento | scelte depurate uguali | "
      "errore max sulla preferenza ritrovata |")
    w("|---|---|---|---|---|")
    for k in CHIAVI:
        v = inv[k]
        w(f"| {NOME[k]} | {' / '.join(fs(x, 2) for x in v['spostamento'])} | {fn(100 * v['acc_grezza_spostata'])} % | "
          f"{v['scelte_uguali']}/{N} | {v['errore_prior_max']:.1e} |")
    w("")
    # ── (b) preferenza
    w("## (b) Preferenza a priori per lettera, e massa sulle 4 lettere")
    w("")
    w("P_prior in % (stima principale: globale bilanciata), IC 95 % bootstrap; «offset» = log P_prior − media dei 4 "
      "(nat): quanto la lettera e' favorita a parita' di contenuto. RStd = deviazione standard (popolazione) dei 4 "
      "richiami per lettera giusta, la misura di distorsione dell'articolo (§2).")
    w("")
    w("| modello | P_prior A % | P_prior B % | P_prior C % | P_prior D % | offset A/B/C/D (nat) | senza etichette "
      "A/B/C/D % | RStd grezza | RStd depurata |")
    w("|---|---|---|---|---|---|---|---|---|")
    for k in CHIAVI:
        pr, prb = np.exp(P[k]["prior"]), np.exp(BT[k]["prior"])
        cel = [f"{fn(100 * pr[i], 1)} {ic(*pct100(prb[:, i]), 1, False)}" for i in range(4)]
        off = P[k]["prior"] - P[k]["prior"].mean()
        w(f"| {NOME[k]} | " + " | ".join(cel) + f" | {' / '.join(fs(x, 2) for x in off)} | "
          f"{' / '.join(fn(100 * x, 1) for x in np.exp(P[k]['prior_lib']))} | {fn(100 * P[k]['rstd_grezza'], 1)} | "
          f"{fn(100 * P[k]['rstd_glob'], 1)} |")
    w("")
    w("Differenza di preferenza fra modelli (offset del confrontato − offset del riferimento, nat, IC 95 % appaiato):")
    w("")
    w("| rif. → confr. | Δ offset A | Δ offset B | Δ offset C | Δ offset D |")
    w("|---|---|---|---|---|")
    for a, b in COPPIE:
        oa = BT[a]["prior"] - BT[a]["prior"].mean(axis=1, keepdims=True)
        ob = BT[b]["prior"] - BT[b]["prior"].mean(axis=1, keepdims=True)
        pa, pb = P[a]["prior"] - P[a]["prior"].mean(), P[b]["prior"] - P[b]["prior"].mean()
        w(f"| {NOME[a]} → {NOME[b]} | " + " | ".join(
            f"{fs(pb[i] - pa[i])} {ic(*pct(ob[:, i] - oa[:, i]))}" for i in range(4)) + " |")
    w("")
    w("**Massa sulle 4 lettere** = P(« A»)+P(« B»)+P(« C»)+P(« D») dopo «Answer:», sull'intero vocabolario. Quali "
      "token prendano il resto della probabilita' NON e' nei registri (servirebbe stampare i token piu' probabili).")
    w("")
    w("| modello | Q10 % | Q25 % | mediana % | Q75 % | Q90 % | log-massa media (nat) |")
    w("|---|---|---|---|---|---|---|")
    for k in CHIAVI:
        q = np.percentile(100 * np.exp(dati[k]["logmassa"]), [10, 25, 50, 75, 90])
        w(f"| {NOME[k]} | " + " | ".join(fn(x, 2) for x in q) + f" | {fs(P[k]['logmassa'], 3)} |")
    w("")
    w("| rif. → confr. | Δ log-massa media (nat) [IC95] | compiti con massa in calo % |")
    w("|---|---|---|")
    for a, b in COPPIE:
        x, xb = dlt(a, b, "logmassa")
        calo = float((dati[b]["logmassa"] < dati[a]["logmassa"]).mean())
        w(f"| {NOME[a]} → {NOME[b]} | {fs(x, 3)} {ic(*pct(xb), 3)} | {fn(100 * calo, 1)} |")
    w("")
    w("Preferenza (offset, nat) e accuratezza per terzile della PROPRIA massa (stima sugli stessi compiti del terzile):")
    w("")
    w("| modello | terzile di massa | massa % (intervallo) | offset A/B/C/D | grezza % | depurata col terzile % |")
    w("|---|---|---|---|---|---|")
    for k in CHIAVI:
        S, t = dati[k]["S"], dati[k]["terzile"]
        for j, nome in enumerate(("basso", "medio", "alto")):
            sel = t == j
            m = np.stack([S[sel & (oro == g)].mean(axis=0) for g in range(4)]).mean(axis=0)
            lp = logsm(m)
            mm = 100 * np.exp(dati[k]["logmassa"][sel])
            w(f"| {NOME[k]} | {nome} | {fn(mm.min(), 2)}–{fn(mm.max(), 2)} | "
              f"{' / '.join(fs(x, 2) for x in lp - lp.mean())} | {fn(100 * dati[k]['giusta_grezza'][sel].mean(), 1)} | "
              f"{fn(100 * ((S[sel] - lp).argmax(axis=1) == oro[sel]).mean(), 1)} |")
    w("")
    # ── (c) accuratezza depurata
    w("## (a)+(c) Accuratezza grezza e depurata")
    w("")
    w("| modello | grezza % | " + " | ".join(f"{v} %" for v, _, _, _ in VARIANTI) + " |")
    w("|---|---|" + "---|" * len(VARIANTI))
    for k in CHIAVI:
        w(f"| {NOME[k]} | {fn(100 * P[k]['acc_grezza'])} | " + " | ".join(
            f"{fn(100 * P[k][f'acc_{v}'])} {ic(*pct100(BT[k][f'acc_{v}']), 2, False)}" for v, _, _, _ in VARIANTI)
          + " |")
    w("")
    w("Depurata − grezza dello stesso modello (punti, IC 95 % appaiato):")
    w("")
    w("| modello | " + " | ".join(VARIANTI[i][0] for i in range(len(VARIANTI))) + " |")
    w("|---|" + "---|" * len(VARIANTI))
    for k in CHIAVI:
        w(f"| {NOME[k]} | " + " | ".join(
            cella_pt(P[k][f"acc_{v}"] - P[k]["acc_grezza"], BT[k][f"acc_{v}"] - BT[k]["acc_grezza"])
            for v, _, _, _ in VARIANTI) + " |")
    w("")
    # ── (d) scomposizione
    w("## (d) Confronti appaiati e scomposizione della perdita")
    w("")
    w("**Conoscenza** = Δ fra i due modelli con la preferenza tolta a entrambi (Δ depurata); **preferenza** = Δ grezza "
      "− Δ depurata. Con la preferenza globale anche gli scambi: «conoscenza a pref. del rif.» = (confrontato con la "
      "preferenza del riferimento) − riferimento; «a pref. del confr.» = confrontato − (riferimento con la preferenza "
      "del confrontato). Le due differiscono se la stessa preferenza costa diversamente ai due modelli (interazione); "
      "Shapley = la loro media, che divide l'interazione a meta'.")
    w("")
    w("| rif. → confr. | Δ grezza | conoscenza (glob) | preferenza (glob) | conoscenza a pref. del rif. | "
      "conoscenza a pref. del confr. | Shapley: conoscenza | Shapley: preferenza |")
    w("|---|---|---|---|---|---|---|---|")
    for a, b in COPPIE:
        s = scomp[(a, b)]
        w(f"| {NOME[a]} → {NOME[b]} | " + " | ".join(
            cella_pt(*s[q]) for q in ("grezza", "dep", "pref", "k_rif", "k_conf", "shapley_con", "shapley_pref"))
          + " |")
    w("")
    w("Conoscenza (Δ depurata) con le altre stime della preferenza:")
    w("")
    w("| rif. → confr. | Δ grezza | " + " | ".join(f"conoscenza ({v})" for v, _, _, _ in VARIANTI) + " |")
    w("|---|---|" + "---|" * len(VARIANTI))
    for a, b in COPPIE:
        s = scomp[(a, b)]
        w(f"| {NOME[a]} → {NOME[b]} | {cella_pt(*s['grezza'])} | " + " | ".join(
            cella_pt(*s["dep" if v == "glob" else v]) for v, _, _, _ in VARIANTI) + " |")
    w("")
    # B → B-150 per variante: lettere, margine, vittorie
    w("### B → B-150: cosa resta dopo ogni depurazione")
    w("")
    w("Se la differenza fosse solo una preferenza ben stimata, dopo la depurazione il Δ sarebbe lo stesso per le 4 "
      "lettere giuste (colonna «A − D» ≈ 0).")
    w("")
    w("| stima della preferenza | B % | B-150 % | Δ (conoscenza) | Δ lettera A | Δ lettera B | Δ lettera C | "
      "Δ lettera D | Δ A − Δ D | Δ margine medio (nat) | Δ vittorie a coppie (punti) |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    for v in ["grezza"] + [x[0] for x in VARIANTI]:
        acc = "acc_grezza" if v == "grezza" else f"acc_{v}"
        lett = f"lett_{v}"
        x, xb = dlt("B", "B150", acc)
        dl, dlb = dlt("B", "B150", lett)
        mg, mgb = dlt("B", "B150", f"margine_{v}")
        cp, cpb = dlt("B", "B150", f"coppie_{v}")
        w(f"| {'nessuna (grezza)' if v == 'grezza' else v} | {fn(100 * P['B'][acc])} | {fn(100 * P['B150'][acc])} | "
          f"{cella_pt(x, xb)} | " + " | ".join(cella_pt(dl[i], dlb[:, i], 1) for i in range(4))
          + f" | {cella_pt(dl[0] - dl[3], dlb[:, 0] - dlb[:, 3], 1)} | {cella_pt(mg, mgb, 3, 1.0)} | "
          f"{cella_pt(cp, cpb)} |")
    w("")
    w("### B → B-150 per lettera giusta, grezza e depurata (globale)")
    w("")
    w("| lettera | n | B grezza % | B-150 grezza % | Δ grezza [IC95] | B depurata % | B-150 depurata % | "
      "Δ depurata [IC95] |")
    w("|---|---|---|---|---|---|---|---|")
    for i, l in enumerate(LETTERE):
        g, gb = dlt("B", "B150", "lett_grezza")
        d, db = dlt("B", "B150", "lett_glob")
        w(f"| {l} | {int((oro == i).sum())} | {fn(100 * P['B']['lett_grezza'][i], 1)} | "
          f"{fn(100 * P['B150']['lett_grezza'][i], 1)} | {cella_pt(g[i], gb[:, i], 1)} | "
          f"{fn(100 * P['B']['lett_glob'][i], 1)} | {fn(100 * P['B150']['lett_glob'][i], 1)} | "
          f"{cella_pt(d[i], db[:, i], 1)} |")
    w("")
    w("Accuratezza per lettera giusta di tutti i modelli (grezza → depurata globale):")
    w("")
    w("| modello | A | B | C | D | RStd |")
    w("|---|---|---|---|---|---|")
    for k in CHIAVI:
        w(f"| {NOME[k]} | " + " | ".join(
            f"{fn(100 * P[k]['lett_grezza'][i], 1)} → {fn(100 * P[k]['lett_glob'][i], 1)}" for i in range(4))
          + f" | {fn(100 * P[k]['rstd_grezza'], 1)} → {fn(100 * P[k]['rstd_glob'], 1)} |")
    w("")
    w("### B → B-150 per materia, grezza e depurata (globale)")
    w("")
    w("Ordinate per Δ grezza. Con IC per materia larghi, la tabella serve a vedere se la depurazione cambia il quadro, "
      "non a giudicare una materia.")
    w("")
    w("| materia | n | B grezza % | B-150 grezza % | Δ grezza [IC95] | B dep. % | B-150 dep. % | Δ depurata [IC95] |")
    w("|---|---|---|---|---|---|---|---|")
    g, gb = dlt("B", "B150", "mat_grezza")
    d, db = dlt("B", "B150", "mat_glob")
    for j in sorted(range(len(materie)), key=lambda j: g[j]):
        w(f"| {materie[j]} | {int((mat_idx == j).sum())} | {fn(100 * P['B']['mat_grezza'][j], 1)} | "
          f"{fn(100 * P['B150']['mat_grezza'][j], 1)} | {cella_pt(g[j], gb[:, j], 1)} | "
          f"{fn(100 * P['B']['mat_glob'][j], 1)} | {fn(100 * P['B150']['mat_glob'][j], 1)} | "
          f"{cella_pt(d[j], db[:, j], 1)} |")
    w("")
    w(f"- Materie in calo: grezza {int((g < 0).sum())} su 15, depurata {int((d < 0).sum())} su 15.")
    w("")
    # ── (e) margini
    w("## (e) Margine della risposta giusta e vittorie a coppie")
    w("")
    w("Margine = log-punteggio della giusta − quello della migliore sbagliata (nat): > 0 ⇔ risposta giusta. Grezzo = "
      "sui logprob stampati; depurato = sui punteggi depurati (preferenza globale). ⚠ Il margine GREZZO contiene la "
      "preferenza (una lettera favorita alza il margine delle domande con quella giusta e abbassa le altre). Il margine "
      "dipende anche dalla SCALA dei logit (un modello meno sicuro ha margini piu' piccoli senza ordinare peggio) e "
      "da una preferenza residua non costante (rumore sui punteggi → margine medio piu' basso): per questo anche le "
      "vittorie a coppie (la giusta batte ciascuna delle 3 sbagliate), che non dipendono dalla scala. log P4 = "
      "log-probabilita' della giusta rinormalizzata sulle 4 lettere.")
    w("")
    w("| modello | margine medio grezzo | margine medio depurato [IC95] | mediana dep. | Q10 / Q25 / Q75 / Q90 dep. | "
      "vittorie a coppie grezze % | vittorie a coppie dep. % [IC95] | log P4 medio grezzo | log P4 medio dep. |")
    w("|---|---|---|---|---|---|---|---|---|")
    margini = {}
    for k in CHIAVI:
        S = dati[k]["S"]
        mg = valuta(S[None], oro)["margine"][0]
        md = valuta(S[None] - P[k]["prior"][None, None, :], oro)["margine"][0]
        margini[k] = (mg, md)
        q = np.percentile(md, [10, 25, 50, 75, 90])
        w(f"| {NOME[k]} | {fs(P[k]['margine_grezza'], 3)} | {fs(P[k]['margine_glob'], 3)} "
          f"{ic(*pct(BT[k]['margine_glob']), 3)} | {fs(q[2], 3)} | {fs(q[0], 2)} / {fs(q[1], 2)} / {fs(q[3], 2)} / "
          f"{fs(q[4], 2)} | {fn(100 * P[k]['coppie_grezza'], 2)} | {fn(100 * P[k]['coppie_glob'], 2)} "
          f"{ic(*pct100(BT[k]['coppie_glob']), 2, False)} | {fs(P[k]['lp4_grezza'], 3)} | {fs(P[k]['lp4_glob'], 3)} |")
    w("")
    w("Distribuzione del margine (% dei 2000 compiti per fascia, nat), grezzo e depurato:")
    w("")
    etich = []
    for lo, hi in zip(CESTI[:-1], CESTI[1:]):
        etich.append(f"≤ {fs(hi, 1)}" if lo == -math.inf else (f"> {fs(lo, 1)}" if hi == math.inf
                                                                 else f"({fs(lo, 1)}; {fs(hi, 1)}]"))
    w("| modello | " + " | ".join(etich) + " |")
    w("|---|" + "---|" * len(etich))
    for k in CHIAVI:
        for nome, m in (("grezzo", margini[k][0]), ("depurato", margini[k][1])):
            cont = [((m > lo) & (m <= hi)).sum() for lo, hi in zip(CESTI[:-1], CESTI[1:])]
            w(f"| {NOME[k]} {nome} | " + " | ".join(fn(100 * c / N, 1) for c in cont) + " |")
    w("")
    w("Confronti appaiati sul margine e sulle vittorie a coppie (confr. − rif., IC 95 %):")
    w("")
    w("| rif. → confr. | Δ margine grezzo | Δ margine dep. (glob) | compiti col margine dep. in calo % | "
      "Δ margine dep. (mat_x) | Δ margine dep. (massa_x) | Δ vittorie a coppie dep. (glob) | "
      "Δ vittorie a coppie (mat_x) | Δ vittorie a coppie (massa_x) | Δ log P4 dep. (glob) |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for a, b in COPPIE:
        calo = float((margini[b][1] < margini[a][1]).mean())
        w(f"| {NOME[a]} → {NOME[b]} | {cella_pt(*dlt(a, b, 'margine_grezza'), 3, 1.0)} | "
          f"{cella_pt(*dlt(a, b, 'margine_glob'), 3, 1.0)} | {fn(100 * calo, 1)} | "
          f"{cella_pt(*dlt(a, b, 'margine_mat_x'), 3, 1.0)} | {cella_pt(*dlt(a, b, 'margine_massa_x'), 3, 1.0)} | "
          f"{cella_pt(*dlt(a, b, 'coppie_glob'))} | {cella_pt(*dlt(a, b, 'coppie_mat_x'))} | "
          f"{cella_pt(*dlt(a, b, 'coppie_massa_x'))} | {cella_pt(*dlt(a, b, 'lp4_glob'), 3, 1.0)} |")
    w("")
    w("Margine medio per lettera giusta, B e B-150 (grezzo → depurato globale, nat):")
    w("")
    w("| lettera | B grezzo → dep. | B-150 grezzo → dep. | Δ grezzo [IC95] | Δ depurato [IC95] |")
    w("|---|---|---|---|---|")
    g, gb = dlt("B", "B150", "margine_lett_grezza")
    d, db = dlt("B", "B150", "margine_lett_glob")
    for i, l in enumerate(LETTERE):
        w(f"| {l} | {fs(P['B']['margine_lett_grezza'][i])} → {fs(P['B']['margine_lett_glob'][i])} | "
          f"{fs(P['B150']['margine_lett_grezza'][i])} → {fs(P['B150']['margine_lett_glob'][i])} | "
          f"{cella_pt(g[i], gb[:, i], 2, 1.0)} | {cella_pt(d[i], db[:, i], 2, 1.0)} |")
    w("")
    # ── revisione avversaria
    tutto_ok = revisione_md(w, P, BT, RV, dati, ctx, scomp)
    # ── limiti, lettura, conclusione
    lettura_md(w, P, BT, dati, scomp, RV)
    USCITA_MD.write_text("\n".join(L) + "\n", encoding="utf-8")

    js = {"generato": time.strftime("%Y-%m-%dT%H:%M:%S"), "n_boot": N_BOOT, "seme": SEME, "pieghe": K_PIEGHE,
          "articolo": "Zheng et al., Large Language Models Are Not Robust Multiple Choice Selectors, arXiv 2309.03882v4",
          "variante": "preferenza = log-softmax della media bilanciata per lettera giusta di log P4 (sui compiti, non "
                      "sulle permutazioni); punteggio depurato = log P4 - log P_prior (Eq. 8)",
          "controlli": controlli, "invarianza": inv,
          "modelli": {k: {q: (v.tolist() if isinstance(v, np.ndarray) else float(v)) for q, v in P[k].items()}
                      for k in CHIAVI},
          "scomposizione": {f"{a}->{b}": {q: {"stima": float(v[0]), "ic95": list(pct(v[1]))} for q, v in d.items()}
                            for (a, b), d in scomp.items()},
          "revisione_ok": tutto_ok}
    USCITA_JSON.write_text(json.dumps(js, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"scritti {USCITA_MD} e {USCITA_JSON}; revisione_ok={tutto_ok}", flush=True)


def revisione_md(w, P, BT, RV, dati, ctx, scomp) -> bool:
    oro = ctx["oro"]
    w("## Revisione avversaria: (a) e (c) rifatti con un secondo calcolo indipendente")
    w("")
    w("Funzione `revisione_indipendente()` dello script: solo libreria standard (math, struct, re), nessuna funzione "
      "del calcolo principale. Rilegge i registri con un'espressione regolare diversa; per (a) ricava i 0/1 dalle righe "
      "CUMULATIVE «n<TAB>acc» (non dalle MCLP) e ricalcola il Final result in float32 con `struct`; per (c) "
      "rinormalizza e depura con cicli in puro Python (sottraendo la media NON centrata: stessa scelta, altra "
      "aritmetica), rifa' da capo le varianti (terzili di massa per rango, pieghe, materie dalla mappa) e fa gli IC "
      "col **jackknife** (preferenza ri-stimata togliendo un compito alla volta) invece che col bootstrap.")
    w("")
    w("| modello | 0/1 da righe acc = da MCLP (indip.) = principale | Final indip. | = copia = catena | "
      "righe acc copia = catena | preferenza indip. − principale (max) | scelte depurate glob indip. = principale | "
      "varianti: scelte uguali lib / glob_x / mat_x / massa_x | acc. depurata indip. % |")
    w("|---|---|---|---|---|---|---|---|---|")
    tutto_ok = True
    for k in CHIAVI:
        r = RV[k]
        S = dati[k]["S"]
        s_princ = (S - P[k]["prior"][None, :]).argmax(axis=1)
        uguali01 = sum(1 for q in range(N) if r["grezza_da_acc"][q] == r["grezza_da_mclp"][q]
                       == int(dati[k]["giusta_grezza"][q]))
        fin_ok = r["finale_ricalcolato"] == r["finale_copia"] == r["finale_catena"]
        dprior = max(abs(x - y) for x, y in zip(r["prior"], P[k]["prior"]))
        uguali_dep = int(sum(1 for q in range(N) if r["scelte_dep"][q] == s_princ[q]))
        # scelte delle varianti nel calcolo principale (stima puntuale)
        W1 = np.ones((1, N))
        princ = {"lib": (S - logsm(S.mean(axis=0))[None, :]).argmax(axis=1)}
        for v, cella, nc in (("glob_x", np.zeros(N, dtype=np.int64), 1), ("mat_x", ctx["mat_idx"], ctx["n_mat"]),
                             ("massa_x", dati[k]["terzile"], 3)):
            PR, _ = prior_celle(W1, S, cella, nc, oro, ctx["piega"])
            princ[v] = (S - PR[0]).argmax(axis=1)
        terz_ok = list(dati[k]["terzile"]) == r["terzile"]
        uguali_var = {v: int((np.array(r["varianti"][v]) == princ[v]).sum()) for v in princ}
        acc_var_ok = all(abs(r["acc_varianti"][v] - P[k][f"acc_{v}"]) < 1e-12 for v in princ)
        ok = (uguali01 == N and fin_ok and r["acc_uguali_catena"] and uguali_dep == N and dprior < 1e-9
              and abs(r["acc_dep"] - P[k]["acc_glob"]) < 1e-12 and terz_ok and acc_var_ok
              and all(x == N for x in uguali_var.values()))
        tutto_ok &= ok
        w(f"| {NOME[k]} | {uguali01}/{N} | {r['finale_ricalcolato']} | {'sì' if fin_ok else 'NO'} | "
          f"{'sì' if r['acc_uguali_catena'] else 'NO'} | {dprior:.1e} | {uguali_dep}/{N} | "
          f"{' / '.join(str(uguali_var[v]) for v in ('lib', 'glob_x', 'mat_x', 'massa_x'))} "
          f"(terzili {'uguali' if terz_ok else 'DIVERSI'}) | {fn(100 * r['acc_dep'])} |")
    w("")
    w("IC 95 %: bootstrap percentile (principale) contro stima ± 1,96·se jackknife (indipendente); per la Δ grezza "
      "anche l'IC di McNemar dai soli 0/1. ⚠ La scelta depurata dipende dalla preferenza stimata attraverso un argmax "
      "(funzione a gradini): il jackknife e' consistente per statistiche lisce e qui tende a dare se un po' piu' grandi "
      "del bootstrap; dove le due conclusioni differiscono, vale la piu' prudente.")
    w("")
    w("| quantita' | principale: stima [IC95 bootstrap] | indipendente: stima [IC95 jackknife] | se bootstrap | "
      "se jackknife |")
    w("|---|---|---|---|---|")
    st = RV["_stime"]

    def riga_rev(nome, stima_p, campioni, chiave):
        s_i, se_i = st[chiave]
        w(f"| {nome} | {fs(100 * stima_p)} {ic(*pct100(campioni))} | {fs(100 * s_i)} "
          f"{ic(100 * (s_i - Z95 * se_i), 100 * (s_i + Z95 * se_i))} | {fn(100 * float(np.std(campioni)), 3)} | "
          f"{fn(100 * se_i, 3)} |")

    for k in CHIAVI:
        riga_rev(f"acc. depurata {NOME[k]} (livello)", P[k]["acc_glob"], BT[k]["acc_glob"], f"acc_dep_{k}")
        riga_rev(f"{NOME[k]}: depurata − grezza", P[k]["acc_glob"] - P[k]["acc_grezza"],
                 BT[k]["acc_glob"] - BT[k]["acc_grezza"], f"dep_meno_grezza_{k}")
    for a, b in COPPIE:
        s = scomp[(a, b)]
        riga_rev(f"{NOME[a]} → {NOME[b]}: Δ grezza", s["grezza"][0], s["grezza"][1], f"dgrezza_{a}_{b}")
        s_m, se_m = st[f"mcnemar_{a}_{b}"]
        w(f"| {NOME[a]} → {NOME[b]}: Δ grezza, IC di McNemar | | {fs(100 * s_m)} "
          f"{ic(100 * (s_m - Z95 * se_m), 100 * (s_m + Z95 * se_m))} | | {fn(100 * se_m, 3)} |")
        riga_rev(f"{NOME[a]} → {NOME[b]}: conoscenza (Δ depurata glob)", s["dep"][0], s["dep"][1], f"ddep_{a}_{b}")
        riga_rev(f"{NOME[a]} → {NOME[b]}: preferenza (glob)", s["pref"][0], s["pref"][1], f"pref_{a}_{b}")
    w("")
    w(f"**Esito della revisione: {'(a) e (c) coincidono nei due calcoli, scelta per scelta, anche nelle varianti.' if tutto_ok else 'DISCORDANZE: vedi tabelle.'}**")
    w("")
    return tutto_ok


def lettura_md(w, P, BT, dati, scomp, RV) -> None:
    """Limiti, lettura e conclusione: ogni numero viene dai calcoli sopra (26/9)."""
    oro = np.array([LETTERE.index(x) for x in json.loads(MAPPA.read_text(encoding="utf-8"))["lettera"]])
    s = scomp[("B", "B150")]
    VAR_ALT = ("lib", "glob_x", "mat_x", "massa_x")

    def v(t, q, d=2):
        x, xb = t[q]
        return f"{fs(100 * x, d)} {ic(*pct100(xb), d)}"

    def gz(t, q):
        return giudizio(*pct100(t[q][1]))

    def pt(k, q):
        return fn(100 * P[k][q])

    def forbice(t, pref=False):
        """Minimo e massimo, fra le 5 stime della preferenza, della conoscenza (o della parte di preferenza)."""
        chiavi = ["pref"] + [f"pref_{x}" for x in VAR_ALT] if pref else ["dep", *VAR_ALT]
        valori = [100 * t[c][0] for c in chiavi]
        return f"da {fs(min(valori))} a {fs(max(valori))}"

    def forbice_multi(*ts, pref=False):
        chiavi = ["pref"] + [f"pref_{x}" for x in VAR_ALT] if pref else ["dep", *VAR_ALT]
        valori = [100 * t[c][0] for t in ts for c in chiavi]
        return f"da {fs(min(valori))} a {fs(max(valori))}"

    def dd(a, b, q, scala=100.0, d=2):
        x, xb = P[b][q] - P[a][q], BT[b][q] - BT[a][q]
        return f"{fs(scala * x, d)} {ic(*[scala * y for y in pct(xb)], d)}", pct(xb)

    offB = P["B"]["prior"] - P["B"]["prior"].mean()
    off1 = P["B150"]["prior"] - P["B150"]["prior"].mean()
    massa = {k: 100 * float(np.median(np.exp(dati[k]["logmassa"]))) for k in CHIAVI}
    # costo di una preferenza rispetto a NESSUNA preferenza, sulla conoscenza di ciascun modello
    c150_150 = P["B150"]["acc_glob"] - P["B150"]["acc_grezza"]
    c150_B = P["B"]["acc_glob"] - P["scambio_B_con_pref_B150"]
    cB_B = P["B"]["acc_glob"] - P["B"]["acc_grezza"]
    cB_150 = P["B150"]["acc_glob"] - P["scambio_B150_con_pref_B"]
    guad = {k: {x: P[k][f"acc_{x}"] - P[k]["acc_glob_x"] for x in ("mat_x", "massa_x")} for k in CHIAVI}
    dl_g = 100 * (P["B150"]["lett_grezza"] - P["B"]["lett_grezza"])
    resto = {}
    for x in ("glob", *VAR_ALT):
        d = P["B150"][f"lett_{x}"] - P["B"][f"lett_{x}"]
        db = BT["B150"][f"lett_{x}"] - BT["B"][f"lett_{x}"]
        resto[x] = (100 * (d[0] - d[3]), pct100(db[:, 0] - db[:, 3]))
    marg = {}
    for k in ("B", "B150"):
        D = dati[k]["S"] - P[k]["prior"][None, :]
        md = valuta(D[None], oro)["margine"][0]
        marg[k] = (float(md.mean()), 100 * float((md > 4).mean()))
    cop = {x: dd("B", "B150", f"coppie_{x}") for x in ("glob", "mat_x", "massa_x")}

    def jk_ic(chiave):
        x, se = RV["_stime"][chiave]
        return 100 * (x - Z95 * se), 100 * (x + Z95 * se)

    def jk_str(chiave):
        lo, hi = jk_ic(chiave)
        return f"{fs(100 * RV['_stime'][chiave][0])} {ic(lo, hi)} ({giudizio(lo, hi)})"

    w("## Limiti (cosa questa depurazione non puo' dire)")
    w("")
    w("1. Non e' PriDe esatto: la preferenza e' stimata mediando sui compiti, non sulle rotazioni dei contenuti dello "
      "stesso compito. Se in MMLU certi contenuti stanno sistematicamente in certe posizioni (es. «tutte le "
      "precedenti» di solito in D, numeri in ordine crescente), quella regolarita' finisce nella «preferenza» e la "
      "depurazione la toglie. Una regolarita' delle DOMANDE aiuterebbe pero' tutti i modelli: la tabella (a)+(c) "
      "mostra chi guadagna da ciascuna variante.")
    w("2. Le varianti per materia e per massa sono modelli della preferenza piu' flessibili: la stima incrociata "
      "impedisce l'adattamento ai compiti depurati, ma un modello piu' flessibile puo' comunque assorbire nella "
      "«preferenza» una parte di segnale che dipende dal contenuto. Per questo si riportano tutte, non la piu' "
      "favorevole.")
    w("3. «Conoscenza» qui = scelta fatta sul contenuto una volta tolta la preferenza stimata. L'articolo (§2.4) "
      "separa la preferenza per il token-ID da quella per la POSIZIONE; senza rotazioni le due si confondono, e un "
      "danno che colpisce la lettura del contenuto in una certa posizione non e' una preferenza costante: nessuna delle "
      "5 stime lo toglie (vedi punto 3 della lettura).")
    w("4. Il margine e log P4 dipendono anche dalla scala dei logit; le vittorie a coppie no.")
    w("5. La massa sulle 4 lettere dice quanto il modello mette sui token « A»…« D» dopo «Answer:»; dove vada il resto "
      "della probabilita' non e' misurato (serve una stampa dei token piu' probabili, su GPU).")
    w("")
    w("## Lettura e conclusione")
    w("")
    w(f"**Risposta.** Dei {fn(-100 * s['grezza'][0])} punti persi da B a B-150 (grezza {v(s, 'grezza')}), con la "
      f"preferenza per lettera tolta a entrambi i modelli (Eq. 8, variante a sottrazione della preferenza media) la "
      f"**conoscenza** vale {v(s, 'dep')} ({gz(s, 'dep')}) e la **preferenza** {v(s, 'pref')} ({gz(s, 'pref')}). "
      f"Con le altre 4 stime della preferenza la conoscenza va {forbice(s)} e la preferenza {forbice(s, True)} punti. "
      f"Revisione indipendente (jackknife): conoscenza {jk_str('ddep_B_B150')}, preferenza {jk_str('pref_B_B150')}."
      + "".join(f" ⚠ Per la {nome} bootstrap e jackknife danno giudizi diversi ({gz(s, q)} contro {giudizio(*jk_ic(ch))}): "
                "vale il piu' prudente." for nome, q, ch in (("conoscenza", "dep", "ddep_B_B150"),
                                                            ("preferenza", "pref", "pref_B_B150"))
                if gz(s, q) != giudizio(*jk_ic(ch))))
    w("")
    w(f"**Interazione.** La stessa preferenza costa di piu' a chi ha il segnale di contenuto piu' debole: rispetto a "
      f"nessuna preferenza, quella di B-150 toglie {fn(100 * c150_150)} punti alla conoscenza di B-150 e "
      f"{fn(100 * c150_B)} a quella di B; quella di B toglie {fn(100 * cB_150)} a B-150 e {fn(100 * cB_B)} a B (un "
      f"valore negativo = la preferenza aiuta). Se la conoscenza si conta alle preferenze VERE dei due modelli invece "
      f"che a preferenza nulla, vale {v(s, 'k_rif')} (con la preferenza di B) o {v(s, 'k_conf')} (con quella di "
      f"B-150); la ripartizione di Shapley da' conoscenza {v(s, 'shapley_con')} e preferenza "
      f"{v(s, 'shapley_pref')}. Le due letture non si contraddicono: a preferenza nulla B-150 sceglie a "
      f"{fn(-100 * s['dep'][0])} punti da B; con la sua preferenza, piu' forte, su un segnale di contenuto piu' "
      f"debole, perde di piu'.")
    w("")
    w(f"1. **Preferenza di B-150 molto piu' forte di quella di B.** Offset A/B/C/D (nat): B-150 "
      f"{' / '.join(fs(x) for x in off1)}, B {' / '.join(fs(x) for x in offB)}; RStd grezza "
      f"{fn(100 * P['B150']['rstd_grezza'], 1)} contro {fn(100 * P['B']['rstd_grezza'], 1)}. B-150 mette sui 4 "
      f"token-lettera una probabilita' mediana del {fn(massa['B150'], 2)} % (B {fn(massa['B'], 2)} %, base "
      f"{fn(massa['base'], 2)} %, v20 {fn(massa['v20'], 2)} %): sceglie fra le lettere nella coda della distribuzione, "
      "e nei compiti dove la massa e' piu' bassa la preferenza pesa di piu' (tabella per terzile di massa).")
    w(f"2. **Tolta la preferenza globale** B-150 sale da {pt('B150', 'acc_grezza')} a {pt('B150', 'acc_glob')} %, "
      f"B passa da {pt('B', 'acc_grezza')} a {pt('B', 'acc_glob')} %, base da {pt('base', 'acc_grezza')} a "
      f"{pt('base', 'acc_glob')}, v20 da {pt('v20', 'acc_grezza')} a {pt('v20', 'acc_glob')}: la depurazione non gonfia "
      "meccanicamente l'accuratezza (sui modelli con preferenza debole cambia di decimi). La variante senza etichette e "
      "quella incrociata danno lo stesso quadro (tabella (a)+(c)).")
    w(f"3. **Residuo per lettera che nessuna stima toglie.** Δ lettera A − Δ lettera D (B → B-150): grezzo "
      f"{fs(dl_g[0] - dl_g[3], 1)} punti; dopo la depurazione "
      + "; ".join(f"{x} {fs(resto[x][0], 1)} {ic(*resto[x][1], 1)}" for x in ("glob", *VAR_ALT))
      + f". Il residuo (globale) e' il {fn(100 * resto['glob'][0] / (dl_g[0] - dl_g[3]), 0)} % del disegno grezzo "
      "e resta anche con la preferenza per materia o per massa: "
      "B-150 perde di piu' dove la giusta e' A anche a preferenza tolta. Con questi dati non si separa una preferenza "
      "che cambia da compito a compito da un danno che dipende dalla posizione del contenuto: serve PriDe esatto "
      "(rotazioni).")
    w(f"4. **La preferenza di B-150 non e' costante.** Stimata per materia o per terzile di massa (incrociata), B-150 "
      f"guadagna {fs(100 * guad['B150']['mat_x'])} e {fs(100 * guad['B150']['massa_x'])} punti oltre la globale "
      f"incrociata; B {fs(100 * guad['B']['mat_x'])} e {fs(100 * guad['B']['massa_x'])}, base "
      f"{fs(100 * guad['base']['mat_x'])} e {fs(100 * guad['base']['massa_x'])}, v20 {fs(100 * guad['v20']['mat_x'])} e "
      f"{fs(100 * guad['v20']['massa_x'])}. Una regolarita' delle domande aiuterebbe tutti; la flessibilita' serve solo "
      "a B-150.")
    w(f"5. **Conoscenza misurata senza la scelta finale (punto (e)).** Il segnale di contenuto di B-150 e' piu' "
      f"debole: margine depurato medio {fs(marg['B150'][0], 3)} nat contro {fs(marg['B'][0], 3)} di B "
      f"({fs(100 * (marg['B150'][0] / marg['B'][0] - 1), 0)} %; Δ {dd('B', 'B150', 'margine_glob', 1.0, 3)[0]}), "
      f"compiti con margine > 4 nat {fn(marg['B150'][1], 1)} % contro {fn(marg['B'][1], 1)} %. L'ORDINE fra i contenuti "
      f"cambia meno: vittorie a coppie depurate {cop['glob'][0]} punti (preferenza per materia {cop['mat_x'][0]}, "
      f"{giudizio(*[100 * y for y in cop['mat_x'][1]])}; per massa {cop['massa_x'][0]}). Cioe': B-150 distingue la "
      "giusta con meno sicurezza e quasi con lo stesso ordine; ed e' quella minore sicurezza che lascia agire la sua "
      "preferenza.")
    tb, tv = scomp[("base", "B")], scomp[("v20", "B")]
    t1, t2 = scomp[("base", "B150")], scomp[("v20", "B150")]
    tbv = scomp[("base", "v20")]
    w(f"6. **base e v20 contro le forge.** B perde conoscenza, non preferenza: base → B conoscenza {v(tb, 'dep')} "
      f"({forbice(tb)} con le altre stime), preferenza {v(tb, 'pref')} ({gz(tb, 'pref')}); v20 → B conoscenza "
      f"{v(tv, 'dep')} ({forbice(tv)}), preferenza {v(tv, 'pref')} ({gz(tv, 'pref')}). B-150 perde entrambe: base → "
      f"B-150 conoscenza {v(t1, 'dep')} ({forbice(t1)}), preferenza {v(t1, 'pref')} ({forbice(t1, True)}); v20 → "
      f"B-150 conoscenza {v(t2, 'dep')} ({forbice(t2)}), preferenza {v(t2, 'pref')} ({forbice(t2, True)}). Le vittorie a "
      f"coppie depurate confermano l'ordine: base → B {dd('base', 'B', 'coppie_glob')[0]}, base → B-150 "
      f"{dd('base', 'B150', 'coppie_glob')[0]} punti. base → v20: grezza {v(tbv, 'grezza')}, conoscenza "
      f"{v(tbv, 'dep')}. In sintesi: la distanza di B da base/v20 e' conoscenza ({forbice_multi(tb, tv)} punti "
      f"con le 5 stime), quella di B-150 e' conoscenza ({forbice_multi(t1, t2)}) piu' preferenza "
      f"({forbice_multi(t1, t2, pref=True)}); quello che B-150 aggiunge rispetto a B e' soprattutto preferenza.")
    w("7. **Per chiudere la domanda serve poca GPU:** (i) PriDe esatto (Eq. 7) su un campione — 4 rotazioni cicliche "
      "dei contenuti per compito, per esempio 100 compiti = 400 passaggi per modello — che misura la preferenza per "
      "compito e separa ID da posizione (punto 3); (ii) la stampa dei token piu' probabili dopo «Answer:» per B e "
      "B-150, per sapere dove va la probabilita' che B-150 toglie alle lettere.")
    w("")


if __name__ == "__main__":
    main()
