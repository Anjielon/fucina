#!/usr/bin/env python3
"""SONDA LINEARE PER STRATO — fase 3 (CPU): in quale strato lo studente perde la risposta MMLU (28/9/2026).

## La misura
Per ogni modello e ogni strato l (l_out-l, ultimo token del prompt «...Answer:»): una REGRESSIONE LOGISTICA a 4 classi
(A-D, penalita' L2) che legge la risposta giusta dallo stato. Validazione incrociata:
  - esterna: StratifiedKFold a K pieghe (default 5), RIPETUTA R volte (default 3) con semi diversi; le STESSE pieghe per
    tutti i modelli e tutti gli strati (confronti appaiati); ogni domanda riceve R previsioni fuori piega, mediate;
  - interna (nella sola parte d'addestramento): C scelto su una griglia (default 1e-5..10, mezze decadi) con 3 pieghe,
    criterio la log-verosimiglianza (piu' liscia dell'accuratezza), a parita' il C piu' piccolo; fra i candidati c'e'
    anche il PRIORE (frequenze delle classi, «C = 0»): la sonda deve batterlo per essere usata, cosi' a segnale nullo
    entrambi i modelli danno la stessa previsione e la Δ non misura il sovradattamento; poi riaddestrata sull'intera
    parte d'addestramento. Standardizzazione (media/varianza) stimata SOLO sulla parte d'addestramento.
Perche' logistica L2 con C interno: N (~1600) < d (2048), i dati sono linearmente separabili; senza penalita' la sonda
impara a memoria. Il C scelto dentro la piega non vede mai la piega di prova.

## Cosa si riporta (per strato, per modello)
  acc        accuratezza della sonda (argmax della previsione media), IC 95 % bootstrap stratificato per materia
  ll         log-loss della sonda (nat; misura CONTINUA, principale: l'accuratezza e' a soglia, Schaeffer 2304.15004)
  Δ appaiate studente − riferimento (maestro): Δll con z, p e IC bootstrap; Δacc con McNemar esatto; Holm sui 40 strati
  trasf      la sonda del MAESTRO applicata agli stati dello studente (stesse pieghe, stessa standardizzazione del
             maestro): quanto lo studente «parla ancora la lingua» del maestro in quello strato
  cka, err   CKA lineare maestro↔studente (Kornblith 2019; stessa definizione di CKA-QAD 2606.05682 Eq. 4), anche sulle
             dimensioni standardizzate (cka_std: le attivazioni massicce tengono il CKA grezzo vicino a 1), ed errore
             relativo mediano ||h_s − h_m|| / ||h_m|| sugli stessi token
  uscita     accuratezza e log-loss del MODELLO sulle 4 lettere (dai logit): l'ancora della sonda all'ultimo strato
Riassunto di localizzazione: primo strato con Δll significativo (Holm), strato da cui resta significativo fino in
fondo, frazione della perdita finale gia' accumulata al confine della riforgia (Δll(19)/Δll(39), IC bootstrap).
⚠️ Lettura: prima che il maestro renda la risposta leggibile, Δ ≈ 0 per costruzione (nessuno dei due la «sa» ancora):
un Δ nullo negli strati bassi NON assolve quegli strati. Per quelli contano trasf, cka ed err (deriva della
rappresentazione) — e la prova causale richiederebbe un innesto degli stati (non fatto qui).

## Controlli che fermano la corsa
Stessi id per domanda (fnv1a64) in tutti i modelli e in domande.json; stesse forme (N, L, d); stati finiti;
accuratezza d'uscita del riferimento >= SONDA_MIN_ACC_USCITA (default 0,5: con gli id delle lettere sbagliati sarebbe
~0,25); ogni previsione fuori piega presente; classi 0-3 in ogni piega.

Uso (CPU confinata):
  python3 sonda.py --modello maestro=@@ROOT:depot@@/sonda_strati/stati/maestro \\
                   --modello studente=@@ROOT:depot@@/sonda_strati/stati/EB3p20 [--jobs 2] [--uscita DIR]
Il primo --modello e' il riferimento. Uscite: sonda_strati.json, sonda_strati.md, sonda_strati.svg, oof.npz.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

SONDA_DIR = Path(os.getenv("SONDA_DIR", "@@ROOT:depot@@/sonda_strati"))
CS_DEFAULT = [10.0 ** e for e in np.arange(-5.0, 1.01, 0.5)]
MIN_ACC_USCITA = float(os.getenv("SONDA_MIN_ACC_USCITA", "0.5"))
LETTERE = "ABCD"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


# ─────────────────────────────── caricamento e controlli ───────────────────────────────
def carica(domande_dir: Path, modelli: list) -> dict:
    """modelli = [(nome, cartella_stati), ...]; il primo e' il riferimento."""
    man = json.loads((domande_dir / "domande.json").read_text(encoding="utf-8"))
    dom = man["domande"]
    y = np.asarray([d["risposta"] for d in dom], dtype=np.int64)
    fnv = [d["fnv1a64_ids"] for d in dom]
    sha_dom = sha256_file(domande_dir / "domande.json")
    stati, lettere, meta = {}, {}, {}
    forma = None
    for nome, cart in modelli:
        cart = Path(cart)
        m = json.loads((cart / "meta.json").read_text(encoding="utf-8"))
        if m["domande"]["fnv1a64_ids"] != fnv:
            raise RuntimeError(f"{nome}: domande diverse da {domande_dir}/domande.json (id fnv1a64)")
        if m["domande"]["sha256_domande_json"] != sha_dom:
            raise RuntimeError(f"{nome}: stati estratti da un domande.json diverso")
        S = np.load(cart / "stati.npy", mmap_mode="r")
        if forma is None:
            forma = S.shape
        if S.shape != forma or S.shape[0] != len(y):
            raise RuntimeError(f"{nome}: forma {S.shape}, attesa {forma} con N={len(y)}")
        lt = np.load(cart / "lettere.npy")
        if lt.shape != (len(y), 4) or not np.isfinite(lt).all():
            raise RuntimeError(f"{nome}: lettere.npy {lt.shape} o non finite")
        stati[nome], lettere[nome], meta[nome] = S, lt, m
    return {"y": y, "materie": np.asarray([d["materia"] for d in dom]),
            "vista": np.asarray([d["vista_forgia"] for d in dom]), "split": np.asarray([d["split"] for d in dom]),
            "stati": stati, "lettere": lettere, "meta": meta, "man": man, "sha_domande": sha_dom,
            "rif": modelli[0][0], "nomi": [n for n, _ in modelli]}


def fai_pieghe(y: np.ndarray, k: int, r: int, seme: int) -> list:
    """R ripetizioni di K pieghe stratificate per etichetta. Uguali per tutti i modelli e gli strati."""
    from sklearn.model_selection import StratifiedKFold
    return [list(StratifiedKFold(k, shuffle=True, random_state=seme + i).split(np.zeros(len(y)), y))
            for i in range(r)]


# ─────────────────────────────── la sonda ───────────────────────────────
class Priore:
    """La sonda «senza informazione»: le frequenze delle classi nella parte d'addestramento. E' un candidato della
    selezione interna (C = 0): una sonda deve BATTERE il priore sulla piega interna per essere usata. Senza questo, a
    segnale nullo la log-loss misura quanto ciascuna geometria fa sovradattare la logistica, non l'informazione
    (collaudo A2: falso positivo z 2,95 allo strato 11 con le etichette mescolate)."""
    classes_ = np.arange(4)

    def fit(self, X, y):
        self.p = np.bincount(y, minlength=4).astype(np.float64) / len(y)
        return self

    def predict_proba(self, X):
        return np.tile(self.p, (len(X), 1))


def addestra(Xtr: np.ndarray, ytr: np.ndarray, Cs: list, n_interne: int, seme: int, max_iter: int):
    """Standardizzazione + logistica L2; C scelto con CV interna (log-verosimiglianza). -> (scaler, clf, C, n_nonconv)"""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    nonconv = 0
    ll = np.zeros(len(Cs) + 1)               # [priore, C_1, ..., C_n]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning)
        for itr, iva in StratifiedKFold(n_interne, shuffle=True, random_state=seme).split(Xtr, ytr):
            sc = StandardScaler().fit(Xtr[itr])
            A, B = sc.transform(Xtr[itr]), sc.transform(Xtr[iva])
            righe = np.arange(len(iva))
            p = Priore().fit(A, ytr[itr]).predict_proba(B)
            ll[0] += -np.log(np.clip(p[righe, ytr[iva]], 1e-300, None)).sum()
            clf = LogisticRegression(C=Cs[0], max_iter=max_iter, warm_start=True)
            for i, C in enumerate(Cs):       # dal C piu' piccolo (piu' penalizzato): partenza calda lungo il cammino
                clf.set_params(C=C)
                clf.fit(A, ytr[itr])
                p = clf.predict_proba(B)
                ll[i + 1] += -np.log(np.clip(p[righe, ytr[iva]], 1e-300, None)).sum()
        scelta = int(np.argmin(ll))          # a parita' il primo: prima il priore, poi il C piu' penalizzato
        sc = StandardScaler().fit(Xtr)
        if scelta == 0:
            C, clf = 0.0, Priore().fit(Xtr, ytr)
        else:
            C = Cs[scelta - 1]
            clf = LogisticRegression(C=C, max_iter=max_iter).fit(sc.transform(Xtr), ytr)
        nonconv = sum(issubclass(x.category, ConvergenceWarning) for x in w)
    if list(clf.classes_) != [0, 1, 2, 3]:
        raise RuntimeError(f"classi della piega {list(clf.classes_)}")
    return sc, clf, C, nonconv


def zscore(X: np.ndarray) -> np.ndarray:
    sd = X.std(0)
    return (X - X.mean(0)) / np.where(sd > 0, sd, 1.0)


def cka_lineare(X: np.ndarray, Y: np.ndarray) -> float:
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    if X.shape[0] < X.shape[1]:
        K, L = X @ X.T, Y @ Y.T
        return float((K * L).sum() / (np.linalg.norm(K) * np.linalg.norm(L)))
    return float(np.linalg.norm(X.T @ Y) ** 2 / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)))


def valuta_strato(li: int, stati: dict, y: np.ndarray, pieghe: list, rif: str, Cs: list, n_interne: int,
                  seme: int, max_iter: int) -> dict:
    Xs = {n: np.asarray(S[:, li, :], dtype=np.float64) for n, S in stati.items()}
    for n, X in Xs.items():
        if not np.isfinite(X).all():
            raise RuntimeError(f"{n} strato {li}: stati non finiti")
    R, N = len(pieghe), len(y)
    P = {n: np.full((R, N, 4), np.nan) for n in Xs}
    T = {n: np.full((R, N, 4), np.nan) for n in Xs if n != rif}
    Cs_scelti = {n: [] for n in Xs}
    nonconv = 0
    for r, pieghe_r in enumerate(pieghe):
        for f, (tr, te) in enumerate(pieghe_r):
            for n, X in Xs.items():
                sc, clf, C, nc = addestra(X[tr], y[tr], Cs, n_interne, seme + 1000 * r + f, max_iter)
                nonconv += nc
                Cs_scelti[n].append(C)
                P[n][r, te] = clf.predict_proba(sc.transform(X[te]))
                if n == rif:
                    for m in T:
                        T[m][r, te] = clf.predict_proba(sc.transform(Xs[m][te]))
    for d in (P, T):
        for n, a in d.items():
            if np.isnan(a).any():
                raise RuntimeError(f"strato {li} {n}: previsioni fuori piega mancanti")
    out = {"strato": li, "P": {n: a.mean(0) for n, a in P.items()},
           "acc_rip": {n: [float((a[r].argmax(1) == y).mean()) for r in range(R)] for n, a in P.items()},
           "T": {n: a.mean(0) for n, a in T.items()}, "C": Cs_scelti, "C_griglia": list(Cs), "nonconv": nonconv, "cka": {}, "cka_std": {},
           "err": {}}
    Xr = Xs[rif]
    nr = np.linalg.norm(Xr, axis=1)
    for n, X in Xs.items():
        if n == rif:
            continue
        out["cka"][n] = cka_lineare(Xr, X)
        # con le attivazioni massicce (poche dimensioni enormi e quasi costanti) il CKA grezzo resta ~1 anche quando
        # il resto diverge: si riporta anche sulle dimensioni standardizzate (ognuna col suo modello)
        out["cka_std"][n] = cka_lineare(zscore(Xr), zscore(X))
        out["err"][n] = float(np.median(np.linalg.norm(X - Xr, axis=1) / np.maximum(nr, 1e-30)))
    return out


# ─────────────────────────────── statistica ───────────────────────────────
def per_domanda(P: np.ndarray, y: np.ndarray) -> tuple:
    giusta = (P.argmax(1) == y).astype(np.float64)
    ll = -np.log(np.clip(P[np.arange(len(y)), y], 1e-300, None))
    return giusta, ll


def indici_bootstrap(materie: np.ndarray, B: int, seme: int) -> np.ndarray:
    """B ricampionamenti stratificati per materia (la composizione per materia resta fissa)."""
    rng = np.random.default_rng(seme)
    gruppi = [np.flatnonzero(materie == m) for m in np.unique(materie)]
    out = np.empty((B, len(materie)), dtype=np.int64)
    for b in range(B):
        out[b] = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in gruppi])
    return out


def ic(v: np.ndarray, idx: np.ndarray) -> list:
    m = v[idx].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def mcnemar_esatto(a: np.ndarray, b: np.ndarray) -> tuple:
    """a, b = 0/1 per domanda. -> (n01 a giusta b sbagliata, n10, p bilaterale esatto)"""
    n01 = int(((a == 1) & (b == 0)).sum())
    n10 = int(((a == 0) & (b == 1)).sum())
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return n01, n10, min(1.0, 2 * p)


def holm(p: list) -> list:
    o = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    corrente = 0.0
    for rank, i in enumerate(o):
        corrente = max(corrente, min(1.0, (m - rank) * p[i]))
        adj[i] = corrente
    return adj.tolist()


def riassumi(dati: dict, strati: list, B: int, seme: int, confine: int, sottoinsieme: np.ndarray | None = None) -> dict:
    """Tutte le misure su un sottoinsieme di domande (maschera; None = tutte). Le sonde non si riaddestrano."""
    y_all, rif = dati["y"], dati["rif"]
    sel = np.ones(len(y_all), bool) if sottoinsieme is None else sottoinsieme
    y = y_all[sel]
    idx = indici_bootstrap(dati["materie"][sel], B, seme)
    L = len(strati)
    ris = {"n_domande": int(sel.sum()), "maggioranza": float(np.bincount(y, minlength=4).max() / len(y)),
           "modelli": {}, "confronti": {}}
    uscita = {}
    for n in dati["nomi"]:
        lt = dati["lettere"][n][sel].astype(np.float64)
        p4 = np.exp(lt - lt.max(1, keepdims=True))
        p4 /= p4.sum(1, keepdims=True)
        g, l_ = per_domanda(p4, y)
        uscita[n] = (g, l_)
        mod = {"uscita": {"acc": float(g.mean()), "acc_ic": ic(g, idx), "ll": float(l_.mean()), "ll_ic": ic(l_, idx)},
               "strati": []}
        for s in strati:
            g, l_ = per_domanda(s["P"][n][sel], y)
            mod["strati"].append({"strato": s["strato"], "acc": float(g.mean()), "acc_ic": ic(g, idx),
                                  "acc_rip": s["acc_rip"][n], "ll": float(l_.mean()), "ll_ic": ic(l_, idx),
                                  "C_mediano": float(np.median(s["C"][n])),
                                  "pieghe_al_priore": int(sum(c == 0.0 for c in s["C"][n])),
                                  "pieghe_al_bordo_alto": int(sum(c == max(s["C_griglia"]) for c in s["C"][n]))})
        ris["modelli"][n] = mod
    for n in dati["nomi"]:
        if n == rif:
            continue
        righe, p_ll, p_acc = [], [], []
        dll_boot = np.empty((L, idx.shape[0]))
        for j, s in enumerate(strati):
            gr, lr = per_domanda(s["P"][rif][sel], y)
            gs, ls = per_domanda(s["P"][n][sel], y)
            gt, lt_ = per_domanda(s["T"][n][sel], y)
            d = ls - lr
            se = float(d.std(ddof=1) / math.sqrt(len(d)))
            z = float(d.mean() / se) if se > 0 else 0.0
            p = math.erfc(abs(z) / math.sqrt(2))
            n01, n10, pm = mcnemar_esatto(gr, gs)
            dll_boot[j] = d[idx].mean(1)
            righe.append({"strato": s["strato"], "d_ll": float(d.mean()), "d_ll_se": se, "d_ll_z": z, "d_ll_p": p,
                          "d_ll_ic": [float(np.percentile(dll_boot[j], 2.5)), float(np.percentile(dll_boot[j], 97.5))],
                          "d_acc": float(gr.mean() - gs.mean()), "mcnemar_n01": n01, "mcnemar_n10": n10,
                          "mcnemar_p": pm, "acc_trasf": float(gt.mean()), "acc_trasf_ic": ic(gt, idx),
                          "ll_trasf": float(lt_.mean()), "deriva_ll": float((lt_ - lr).mean()),
                          "cka": s["cka"][n], "cka_std": s["cka_std"][n], "err_rel": s["err"][n]})
            p_ll.append(p)
            p_acc.append(pm)
        for r_, a, b in zip(righe, holm(p_ll), holm(p_acc)):
            r_["d_ll_p_holm"], r_["mcnemar_p_holm"] = a, b
        sig = [r_["d_ll_p_holm"] < 0.05 and r_["d_ll"] > 0 for r_ in righe]
        primo = next((righe[j]["strato"] for j in range(L) if sig[j]), None)
        sempre = None
        for j in range(L - 1, -1, -1):
            if not sig[j]:
                break
            sempre = righe[j]["strato"]
        pos = {s["strato"]: j for j, s in enumerate(strati)}
        frazione = None
        if confine in pos and righe[-1]["d_ll"] != 0:
            jc = pos[confine]
            fb = dll_boot[jc] / np.where(dll_boot[-1] == 0, np.nan, dll_boot[-1])
            frazione = {"strato": confine, "ultimo": righe[-1]["strato"],
                        "valore": righe[jc]["d_ll"] / righe[-1]["d_ll"],
                        "ic": [float(np.nanpercentile(fb, 2.5)), float(np.nanpercentile(fb, 97.5))]}
        go, lo = uscita[n]
        gor, lor = uscita[rif]
        ris["confronti"][n] = {
            "riferimento": rif, "strati": righe, "primo_strato_significativo": primo, "da_qui_sempre": sempre,
            "frazione_perdita_al_confine": frazione,
            "uscita": {"d_acc": float(gor.mean() - go.mean()), "d_ll": float((lo - lor).mean()),
                       "mcnemar": list(mcnemar_esatto(gor, go))}}
    return ris


# ─────────────────────────────── uscite ───────────────────────────────
def tabella_md(ris: dict, rif: str, titolo: str) -> str:
    righe = [f"### {titolo} — {ris['n_domande']} domande, maggioranza {ris['maggioranza'] * 100:.1f} %", ""]
    for n, c in ris["confronti"].items():
        mr, ms = ris["modelli"][rif], ris["modelli"][n]
        righe += [f"**{n}** contro **{rif}** · uscita del modello (4 lettere): {rif} "
                  f"{mr['uscita']['acc'] * 100:.2f} %, {n} {ms['uscita']['acc'] * 100:.2f} % · primo strato con Δll "
                  f"significativo (Holm 5 %): {c['primo_strato_significativo']} · significativo da qui in poi: "
                  f"{c['da_qui_sempre']}", ""]
        f = c["frazione_perdita_al_confine"]
        if f:
            righe += [f"Frazione della perdita finale (Δll) gia' accumulata allo strato {f['strato']}: "
                      f"{f['valore']:.2f} [IC 95 % {f['ic'][0]:.2f}; {f['ic'][1]:.2f}]", ""]
        righe += [f"| strato | acc {rif} | acc {n} | Δacc | acc trasf | ll {rif} | ll {n} | Δll [IC 95 %] | z | "
                  "p Holm | CKA | CKA std | err rel |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for a, b, r_ in zip(mr["strati"], ms["strati"], c["strati"]):
            righe.append(f"| {r_['strato']} | {a['acc'] * 100:.1f} | {b['acc'] * 100:.1f} | {r_['d_acc'] * 100:+.1f} | "
                         f"{r_['acc_trasf'] * 100:.1f} | {a['ll']:.4f} | {b['ll']:.4f} | {r_['d_ll']:+.4f} "
                         f"[{r_['d_ll_ic'][0]:+.4f}; {r_['d_ll_ic'][1]:+.4f}] | {r_['d_ll_z']:.1f} | "
                         f"{r_['d_ll_p_holm']:.2g} | {r_['cka']:.3f} | {r_['cka_std']:.3f} | {r_['err_rel']:.3f} |")
        righe.append("")
    return "\n".join(righe)


# palette di riferimento della skill dataviz (slot categorici 1-3, validati: scripts/validate_palette.js, chiaro)
COLORI = ["#2a78d6", "#eb6834", "#1baf7a"]
INCHIOSTRO, SECONDARIO, MUTO, GRIGLIA, SUPERFICIE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"


def grafico(ris: dict, rif: str, percorso: Path, confine: int) -> None:
    """SVG scritto a mano (nessuna dipendenza: il python di sistema ha sklearn ma non matplotlib). Due pannelli con lo
    stesso asse degli strati: accuratezza della sonda (IC 95 %) e Δ log-loss appaiata (IC 95 %, punti = Holm 5 %)."""
    nomi = list(ris["modelli"])
    colore = {n: COLORI[i % len(COLORI)] for i, n in enumerate(nomi)}
    W, H, sx, dx = 980, 760, 70, 150
    pannelli = [(50, 440), (500, 700)]                        # (alto, basso) in pixel
    strati = [s["strato"] for s in ris["modelli"][rif]["strati"]]
    x0, x1 = min(strati) - 0.5, max(strati) + 3.0
    X = lambda v: sx + (v - x0) / (x1 - x0) * (W - sx - dx)  # noqa: E731
    acc_tutte = [ris["maggioranza"]] + [v for n in nomi for s in ris["modelli"][n]["strati"] for v in s["acc_ic"]]
    acc_tutte += [ris["modelli"][n]["uscita"]["acc"] for n in nomi]
    for c in ris["confronti"].values():
        acc_tutte += [r["acc_trasf"] for r in c["strati"]]
    a_lo, a_hi = max(0.0, min(acc_tutte) * 100 - 3), min(100.0, max(acc_tutte) * 100 + 3)
    dl = [v for c in ris["confronti"].values() for r in c["strati"] for v in r["d_ll_ic"]] + [0.0]
    d_lo, d_hi = min(dl), max(dl)
    marg = 0.08 * (d_hi - d_lo or 1.0)
    d_lo, d_hi = d_lo - marg, d_hi + marg
    Y1 = lambda v: pannelli[0][1] - (v - a_lo) / (a_hi - a_lo) * (pannelli[0][1] - pannelli[0][0])  # noqa: E731
    Y2 = lambda v: pannelli[1][1] - (v - d_lo) / (d_hi - d_lo) * (pannelli[1][1] - pannelli[1][0])  # noqa: E731
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
         f'font-family="system-ui, sans-serif" font-size="12">',
         f'<rect width="{W}" height="{H}" fill="{SUPERFICIE}"/>',
         f'<text x="{sx}" y="26" font-size="15" fill="{INCHIOSTRO}">Sonda lineare per strato — MMLU validation+dev, '
         f'{ris["n_domande"]} domande, IC 95 %</text>']

    def tacche(lo, hi, quante):
        """Tacche «tonde» (1, 2, 5 x 10^k) dentro [lo, hi]."""
        grezzo = (hi - lo) / max(quante, 1)
        e = 10 ** math.floor(math.log10(grezzo))
        passo = next(m * e for m in (1, 2, 5, 10) if m * e >= grezzo)
        v = math.ceil(lo / passo) * passo
        out = []
        while v <= hi + 1e-12:
            out.append(round(v, 10))
            v += passo
        return out

    def griglia(Y, lo, hi, passi, fmt, alto, basso, etichetta):
        for v in tacche(lo, hi, passi):
            o.append(f'<line x1="{sx}" x2="{W - dx}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="{GRIGLIA}" '
                     f'stroke-width="1"/>')
            o.append(f'<text x="{sx - 6}" y="{Y(v) + 4:.1f}" text-anchor="end" fill="{MUTO}">{fmt(v)}</text>')
        o.append(f'<text x="16" y="{(alto + basso) / 2:.0f}" fill="{SECONDARIO}" transform="rotate(-90 16 '
                 f'{(alto + basso) / 2:.0f})" text-anchor="middle">{etichetta}</text>')
        xc = X(confine + 0.5)
        o.append(f'<line x1="{xc:.1f}" x2="{xc:.1f}" y1="{alto}" y2="{basso}" stroke="{MUTO}" stroke-dasharray="2 3"/>')
    griglia(Y1, a_lo, a_hi, 5, lambda v: f"{v:.0f}", *pannelli[0], "accuratezza della sonda (%)")
    griglia(Y2, d_lo, d_hi, 4, lambda v: f"{v:+.2g}" if v else "0", *pannelli[1], "Δ log-loss della sonda (nat)")
    o.append(f'<text x="{X(confine + 0.5) + 4:.1f}" y="{pannelli[0][0] + 12}" fill="{SECONDARIO}">confine riforgia'
             '</text>')
    for li in strati:
        if li % 5 == 0:
            o.append(f'<text x="{X(li):.1f}" y="{pannelli[1][1] + 18}" text-anchor="middle" fill="{MUTO}">{li}</text>')
    o.append(f'<text x="{X((x0 + max(strati)) / 2):.1f}" y="{pannelli[1][1] + 40}" text-anchor="middle" '
             f'fill="{SECONDARIO}">strato (uscita del blocco, l_out)</text>')
    o.append(f'<line x1="{sx}" x2="{X(max(strati) + 0.5):.1f}" y1="{Y1(ris["maggioranza"] * 100):.1f}" '
             f'y2="{Y1(ris["maggioranza"] * 100):.1f}" stroke="{MUTO}"/>')
    o.append(f'<text x="{sx + 4}" y="{Y1(ris["maggioranza"] * 100) - 5:.1f}" fill="{SECONDARIO}">classe piu\' '
             'frequente</text>')
    o.append(f'<line x1="{sx}" x2="{W - dx}" y1="{Y2(0):.1f}" y2="{Y2(0):.1f}" stroke="#c3c2b7"/>')

    def spezzata(xs, ys, Y, col, tratteggio=""):
        pts = " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(xs, ys))
        o.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2" stroke-linejoin="round" '
                 f'{tratteggio}/>')

    def banda(xs, lo, hi, Y, col):
        pts = " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(xs, lo))
        pts += " " + " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(xs[::-1], hi[::-1]))
        o.append(f'<polygon points="{pts}" fill="{col}" fill-opacity="0.15" stroke="none"/>')
    etichette_fine = []
    for n in nomi:
        st = ris["modelli"][n]["strati"]
        xs = [s["strato"] for s in st]
        banda(xs, [s["acc_ic"][0] * 100 for s in st], [s["acc_ic"][1] * 100 for s in st], Y1, colore[n])
        spezzata(xs, [s["acc"] * 100 for s in st], Y1, colore[n])
        etichette_fine.append((Y1(st[-1]["acc"] * 100), f"sonda {n}"))
        u = ris["modelli"][n]["uscita"]["acc"] * 100
        xu, yu = X(max(strati) + 2), Y1(u)
        o.append(f'<path d="M{xu:.1f},{yu - 6:.1f} L{xu + 6:.1f},{yu:.1f} L{xu:.1f},{yu + 6:.1f} L{xu - 6:.1f},'
                 f'{yu:.1f} Z" fill="{colore[n]}" stroke="{SUPERFICIE}" stroke-width="2"/>')
        o.append(f'<text x="{xu + 10:.1f}" y="{yu + 4:.1f}" fill="{INCHIOSTRO}">uscita {n} {u:.1f}</text>')
    for n, c in ris["confronti"].items():
        xs = [r["strato"] for r in c["strati"]]
        spezzata(xs, [r["acc_trasf"] * 100 for r in c["strati"]], Y1, colore[n], 'stroke-dasharray="6 4"')
        etichette_fine.append((Y1(c["strati"][-1]["acc_trasf"] * 100), f"sonda {rif} su {n}"))
        banda(xs, [r["d_ll_ic"][0] for r in c["strati"]], [r["d_ll_ic"][1] for r in c["strati"]], Y2, colore[n])
        spezzata(xs, [r["d_ll"] for r in c["strati"]], Y2, colore[n])
        for r in c["strati"]:
            if r["d_ll_p_holm"] < 0.05 and r["d_ll"] > 0:
                o.append(f'<circle cx="{X(r["strato"]):.1f}" cy="{Y2(r["d_ll"]):.1f}" r="4.5" fill="{colore[n]}" '
                         f'stroke="{SUPERFICIE}" stroke-width="2"/>')
        o.append(f'<text x="{X(max(xs)) + 8:.1f}" y="{Y2(c["strati"][-1]["d_ll"]) + 4:.1f}" fill="{INCHIOSTRO}">'
                 f'{n} − {rif}</text>')
    # etichette a fine linea senza sovrapposizioni (distanza minima 14 px)
    etichette_fine.sort()
    ult = -1e9
    for y, t in etichette_fine:
        y = max(y, ult + 14)
        ult = y
        o.append(f'<text x="{X(max(strati)) + 8:.1f}" y="{y + 4:.1f}" fill="{INCHIOSTRO}">{t}</text>')
    # legenda (sempre presente con >= 2 serie)
    lx, ly = sx, pannelli[0][1] + 22
    for n in nomi:
        o.append(f'<line x1="{lx}" x2="{lx + 22}" y1="{ly}" y2="{ly}" stroke="{colore[n]}" stroke-width="2"/>')
        o.append(f'<text x="{lx + 28}" y="{ly + 4}" fill="{SECONDARIO}">sonda {n}</text>')
        lx += 140
    for n in ris["confronti"]:
        o.append(f'<line x1="{lx}" x2="{lx + 22}" y1="{ly}" y2="{ly}" stroke="{colore[n]}" stroke-width="2" '
                 f'stroke-dasharray="6 4"/>')
        o.append(f'<text x="{lx + 28}" y="{ly + 4}" fill="{SECONDARIO}">sonda di {rif} sugli stati di {n}</text>')
        lx += 260
    o.append(f'<circle cx="{lx + 6}" cy="{ly}" r="4.5" fill="{MUTO}"/><text x="{lx + 16}" y="{ly + 4}" '
             f'fill="{SECONDARIO}">Δ significativo (Holm 5 %)</text>')
    o.append("</svg>")
    percorso.write_text("\n".join(o), encoding="utf-8")


def main(argv: list | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--domande", type=Path, default=SONDA_DIR / "domande")
    ap.add_argument("--modello", action="append", required=True, help="NOME=CARTELLA_STATI (il primo e' il riferimento)")
    ap.add_argument("--uscita", type=Path, default=SONDA_DIR / "risultati")
    ap.add_argument("--pieghe", type=int, default=5)
    ap.add_argument("--ripetizioni", type=int, default=3)
    ap.add_argument("--interne", type=int, default=3)
    ap.add_argument("--C", default=",".join(f"{c:.3g}" for c in CS_DEFAULT))
    ap.add_argument("--max-iter", type=int, default=3000)
    ap.add_argument("--strati", default="tutti", help="tutti | a-b | lista 0,5,39")
    ap.add_argument("--confine", type=int, default=19, help="ultimo strato NON riforgiato (riforgia 20-39)")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seme", type=int, default=20260928)
    ap.add_argument("--jobs", type=int, default=int(os.getenv("SONDA_JOBS", "2")))
    ap.add_argument("--senza-grafico", action="store_true")
    a = ap.parse_args(argv)
    t0 = time.time()
    modelli = [tuple(m.split("=", 1)) for m in a.modello]
    dati = carica(a.domande, modelli)
    y = dati["y"]
    N, L, D = next(iter(dati["stati"].values())).shape
    if a.strati == "tutti":
        strati = list(range(L))
    elif "-" in a.strati:
        i, j = map(int, a.strati.split("-"))
        strati = list(range(i, j + 1))
    else:
        strati = [int(s) for s in a.strati.split(",")]
    Cs = sorted(float(c) for c in a.C.split(","))
    pieghe = fai_pieghe(y, a.pieghe, a.ripetizioni, a.seme)
    log(f"{N} domande, {L} strati x {D}; modelli {dati['nomi']} (rif {dati['rif']}); {a.ripetizioni}x{a.pieghe} "
        f"pieghe, C interno su {len(Cs)} valori ({Cs[0]:.0e}..{Cs[-1]:.0e}), {a.jobs} processi")
    # accuratezza d'uscita del riferimento: se gli id delle lettere fossero sbagliati sarebbe ~caso
    lt = dati["lettere"][dati["rif"]]
    acc_rif = float((lt.argmax(1) == y).mean())
    if acc_rif < MIN_ACC_USCITA:
        raise RuntimeError(f"accuratezza d'uscita del riferimento {acc_rif:.3f} < {MIN_ACC_USCITA}: lettere o stati "
                           "sbagliati, la sonda non avrebbe senso")
    args = (y, pieghe, dati["rif"], Cs, a.interne, a.seme, a.max_iter)
    if a.jobs == 1:
        ris_strati = []
        for li in strati:
            ris_strati.append(valuta_strato(li, dati["stati"], *args))
            log(f"strato {li}: acc " + " · ".join(f"{n} {(ris_strati[-1]['P'][n].argmax(1) == y).mean() * 100:.1f}"
                                                   for n in dati["nomi"]))
    else:
        from joblib import Parallel, delayed
        cartelle = {n: str(Path(c)) for n, c in modelli}

        def lavoro(li):
            st = {n: np.load(Path(c) / "stati.npy", mmap_mode="r") for n, c in cartelle.items()}
            return valuta_strato(li, st, *args)
        ris_strati = Parallel(n_jobs=a.jobs, verbose=5)(delayed(lavoro)(li) for li in strati)
    ris_strati.sort(key=lambda s: s["strato"])
    nonconv = sum(s["nonconv"] for s in ris_strati)
    tutte = riassumi(dati, ris_strati, a.bootstrap, a.seme, a.confine)
    pulite = riassumi(dati, ris_strati, a.bootstrap, a.seme, a.confine, sottoinsieme=~dati["vista"])
    a.uscita.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.uscita / "oof.npz", y=y, strati=np.asarray([s["strato"] for s in ris_strati]),
                        **{f"P_{n}": np.stack([s["P"][n] for s in ris_strati]) for n in dati["nomi"]},
                        **{f"T_{n}": np.stack([s["T"][n] for s in ris_strati]) for n in dati["nomi"]
                           if n != dati["rif"]})
    esito = {"creato": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "script": "thor_v2/strumenti/sonda_strati/sonda.py", "argv": argv if argv is not None else None,
             "domande": {"cartella": str(a.domande), "sha256": dati["sha_domande"], "n": N},
             "modelli": {n: {"stati": c, "gguf": dati["meta"][n]["modello"]} for n, c in modelli},
             "riferimento": dati["rif"], "strati": strati, "forma": [N, L, D],
             "cv": {"pieghe": a.pieghe, "ripetizioni": a.ripetizioni, "interne": a.interne, "C": Cs,
                    "max_iter": a.max_iter, "seme": a.seme, "avvisi_non_convergenza": nonconv},
             "bootstrap": {"B": a.bootstrap, "stratificato_per": "materia"},
             "tutte": tutte, "senza_domande_viste_dalla_forgia": pulite, "secondi": round(time.time() - t0, 1)}
    (a.uscita / "sonda_strati.json").write_text(json.dumps(esito, ensure_ascii=False, indent=1))
    md = ["# Sonda lineare per strato", "",
          f"Creato {esito['creato']} · domande {N} (sha256 {dati['sha_domande'][:16]}...) · CV {a.ripetizioni}x"
          f"{a.pieghe}, C interno · avvisi di non convergenza {nonconv}", "",
          tabella_md(tutte, dati["rif"], "Tutte le domande"),
          tabella_md(pulite, dati["rif"], "Senza le domande misurate dalla forgia (vista_forgia); le sonde sono le "
                                          "stesse (addestrate su tutte), CKA ed err rel su tutte le domande")]
    (a.uscita / "sonda_strati.md").write_text("\n".join(md))
    if not a.senza_grafico:
        grafico(tutte, dati["rif"], a.uscita / "sonda_strati.svg", a.confine)
    for n, c in tutte["confronti"].items():
        f = c["frazione_perdita_al_confine"]
        log(f"{n}: primo strato significativo {c['primo_strato_significativo']}, da qui sempre {c['da_qui_sempre']}"
            + (f", frazione al {f['strato']} {f['valore']:.2f} [{f['ic'][0]:.2f}; {f['ic'][1]:.2f}]" if f else ""))
    log(f"FATTO in {time.time() - t0:.0f} s -> {a.uscita}")
    return esito


if __name__ == "__main__":
    main()
