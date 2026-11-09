#!/usr/bin/env python3
"""Revisione avversaria INDIPENDENTE di mmlu_per_domanda.py (26/9/2026).

Nato perche' l'analisi «la perdita MMLU di B-150 dipende dalla lettera giusta»
va rifatta con codice diverso prima di crederci. Non importa nulla dallo
script rivisto: legge da solo il file dei compiti, il parquet del test
cais/mmlu e i registri di llama-perplexity (sola lettura), e rifa' i conti
con metodi diversi dove possibile:

  * correttezza per domanda con DUE ricostruzioni diverse (cumulata arrotondata
    e differenza fra righe consecutive), che devono coincidere, piu' controllo
    di Final result e sigma con la formula float32 di perplexity.cpp;
  * mappa compito -> materia/lettera presa DAL FILE DEI COMPITI (testo del
    prompt ed etichette), confrontata col parquet per contenuto su tutti i
    2000 e stampata su 20 indici estratti a caso;
  * accuratezza per lettera, 2x2, McNemar esatto, IC di Wald appaiato E IC
    bootstrap appaiato;
  * eterogeneita' fra materie a parita' di lettera con un test di rapporto di
    verosimiglianza su regressione logistica (IRLS scritto qui), non con
    permutazioni;
  * sovrapposizione delle perdite con atteso STRATIFICATO per difficolta' e
    lettera (l'atteso «a caso» uniforme gonfia il rapporto).

Uso (CPU confinata, la GPU e' della forgia):
  XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus \
  systemd-run --user --scope -p CPUQuota=200% nice -n 19 env OMP_NUM_THREADS=2 \
    python3 thor_v2/strumenti/revisione_mmlu_per_domanda.py
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
import struct
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy import stats

RADICE = Path("@@ROOT:forge@@")
BIN = RADICE / "mc_v2/mmlu_00000.bin"
PARQUET = Path("@@ROOT:models@@/gguf/tony-tern/conoscenza/all/test-00000-of-00001.parquet")
USCITA = RADICE / "thor_v2/risultati/revisione_mmlu_per_domanda.json"
REGISTRI = {
    "base": "stato/densi_v5/mmlu.log",
    "B": "stato/v2_ThorT-scor150/mmlu.log",
    "B150": "stato/v2_ThorT-scor150-L150/mmlu.log",
    "A": "stato/v2_ThorT-sep300/mmlu.log",
    "v20": "thor/mmlu_v20_downaggr3_00000.log",
    "Q2_K": "thor/mmlu_rivale_q2k_00000.log",
    "B0_19": "stato/bisezione_mmlu/mmlu_0-19.log",
    "B20_39": "stato/bisezione_mmlu/mmlu_20-39.log",
}
LETTERE = "ABCD"
SEME = 26092026


# ── file dei compiti (formato di perplexity.cpp: deserialize_string / multiple_choice_answers)
def leggi_compiti(p: Path) -> list[dict]:
    b = p.read_bytes()
    n = struct.unpack_from("<I", b, 0)[0]
    pos = struct.unpack_from(f"<{n}I", b, 4)
    o = 4 + 4 * n

    def stringa(o: int) -> tuple[str, int]:
        k = struct.unpack_from("<I", b, o)[0]
        return b[o + 4:o + 4 + k].decode("utf-8"), o + 4 + k

    def risposte(o: int) -> tuple[list[str], tuple[int, ...], int]:
        k = struct.unpack_from("<I", b, o)[0]
        o += 4
        ris = []
        for _ in range(k):
            s, o = stringa(o)
            ris.append(s)
        lab = struct.unpack_from(f"<{k}i", b, o)
        return ris, lab, o + 4 * k

    compiti = []
    for i in range(n):
        assert o == pos[i], f"offset compito {i}: {o} != {pos[i]}"
        q, o = stringa(o)
        r1, l1, o = risposte(o)
        r2, l2, o = risposte(o)
        assert r1 == list(LETTERE) and sum(l1) == 1 and not r2, (i, r1, l1, r2)
        m = re.match(r"The following are multiple choice questions \(with answers\) about (.+?)\.\n\n", q)
        assert m, i
        compiti.append({"prompt": q, "materia": m.group(1).replace(" ", "_"), "lettera": LETTERE[l1.index(1)]})
    assert o == len(b), "byte avanzati in coda"
    return compiti


# ── registro: due ricostruzioni indipendenti della correttezza per domanda
def leggi_registro(p: Path) -> dict:
    righe = p.read_text(errors="replace").splitlines()
    acc, fin = [], None
    for r in righe:
        m = re.fullmatch(r"(\d+)\t(\d+\.\d{8})", r)
        if m:
            assert int(m.group(1)) == len(acc) + 1, (p, r)
            acc.append(float(m.group(2)))
        m2 = re.search(r"Final result: ([0-9.]+) \+/- ([0-9.]+)", r)
        if m2:
            fin = (float(m2.group(1)), float(m2.group(2)))
    assert "there are 2000 tasks in prompt" in p.read_text(errors="replace"), p
    assert len(acc) == 2000 and fin, (p, len(acc))
    a = np.array(acc)
    i = np.arange(1, 2001)
    # 1) cumulata: n_i = round(acc_i*i/100), residuo di stampa %.8lf <= 5e-9
    cum = np.rint(a * i / 100)
    res_cum = np.abs(100 * cum / i - a).max()
    c1 = np.diff(np.concatenate([[0], cum]))
    # 2) differenza fra righe consecutive, senza cumulare arrotondamenti
    prev = np.concatenate([[0.0], a[:-1] * (i[:-1])])
    d = (a * i - prev) / 100
    c2 = np.rint(d)
    res_diff = np.abs(d - c2).max()
    assert res_cum < 1e-8 and res_diff < 1e-4, (p, res_cum, res_diff)
    assert set(np.unique(c1)) <= {0, 1} and np.array_equal(c1, c2), p
    k = int(c1.sum())
    # Final result come lo calcola perplexity.cpp (float32)
    pf = np.float32(k) / np.float32(2000)
    sig = np.sqrt(pf * (np.float32(1) - pf) / np.float32(1999))
    ok = (f"{100 * float(pf):.4f}" == f"{fin[0]:.4f}") and (f"{100 * float(sig):.4f}" == f"{fin[1]:.4f}")
    return {"c": c1.astype(int), "giuste": k, "final": fin, "coerente": ok,
            "residuo_cumulata": float(res_cum), "residuo_differenza": float(res_diff)}


def wald_appaiato(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    n = len(x)
    b = int(((x == 1) & (y == 0)).sum())
    c = int(((x == 0) & (y == 1)).sum())
    dlt = (c - b) / n
    se = math.sqrt(max((b + c) - (c - b) ** 2 / n, 0)) / n
    return 100 * dlt, 100 * (dlt - 1.96 * se), 100 * (dlt + 1.96 * se)


def boot_appaiato(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, R: int = 20000) -> tuple[float, float]:
    n = len(x)
    idx = rng.integers(0, n, size=(R, n))
    dd = (y[idx] - x[idx]).mean(axis=1) * 100
    return float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))


def mcnemar(x: np.ndarray, y: np.ndarray) -> tuple[int, int, float]:
    b = int(((x == 1) & (y == 0)).sum())
    c = int(((x == 0) & (y == 1)).sum())
    p = stats.binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return b, c, float(p)


def chi2_omogeneita(tab: np.ndarray) -> tuple[float, int, float]:
    tab = tab[tab.sum(axis=1) > 0]
    chi2, p, gl, _ = stats.chi2_contingency(tab, correction=False)
    return float(chi2), int(gl), float(p)


# ── regressione logistica IRLS per il test di rapporto di verosimiglianza
def logit_ll(X: np.ndarray, y: np.ndarray, it: int = 100) -> float:
    beta = np.zeros(X.shape[1])
    for _ in range(it):
        eta = np.clip(X @ beta, -30, 30)
        mu = 1 / (1 + np.exp(-eta))
        W = mu * (1 - mu) + 1e-9
        H = X.T @ (X * W[:, None]) + 1e-8 * np.eye(X.shape[1])
        passo = np.linalg.solve(H, X.T @ (y - mu))
        beta += passo
        if np.abs(passo).max() < 1e-10:
            break
    eta = np.clip(X @ beta, -30, 30)
    mu = np.clip(1 / (1 + np.exp(-eta)), 1e-12, 1 - 1e-12)
    return float((y * np.log(mu) + (1 - y) * np.log(1 - mu)).sum())


def dummies(v: list[str], livelli: list[str]) -> np.ndarray:
    return np.array([[1.0 if x == l else 0.0 for l in livelli[1:]] for x in v]).reshape(len(v), len(livelli) - 1)


def lrt(y: np.ndarray, base_cols: list[np.ndarray], extra: np.ndarray) -> tuple[float, int, float]:
    uno = np.ones((len(y), 1))
    X0 = np.hstack([uno] + base_cols)
    X1 = np.hstack([X0, extra])
    l0, l1 = logit_ll(X0, y), logit_ll(X1, y)
    g = 2 * (l1 - l0)
    gl = extra.shape[1]
    return g, gl, float(stats.chi2.sf(g, gl))


def main() -> None:
    rng = np.random.default_rng(SEME)
    out: dict = {"seme": SEME}
    sha_bin = hashlib.sha256(BIN.read_bytes()).hexdigest()
    sha_pq = hashlib.sha256(PARQUET.read_bytes()).hexdigest()
    out["sha256_bin"], out["sha256_parquet"] = sha_bin, sha_pq
    compiti = leggi_compiti(BIN)
    materie_bin = [c["materia"] for c in compiti]
    lettere = [c["lettera"] for c in compiti]

    # ── parquet: mappa per CONTENUTO (non per posizione), poi controllo della posizione
    t = pq.read_table(PARQUET).to_pylist()
    chiave = {}
    for j, r in enumerate(t):
        k = (r["subject"], r["question"].strip(), tuple(x.strip() for x in r["choices"]))
        chiave.setdefault(k, []).append(j)
    corrisp, ambigui, mancanti = [], 0, 0
    for i, c in enumerate(compiti):
        corpo = c["prompt"].split("\n\n", 1)[1]
        assert corpo.endswith("\nAnswer:")
        corpo = corpo[: -len("\nAnswer:")]
        pezzi = corpo.rsplit("\nA. ", 1)
        q = pezzi[0]
        resto = "A. " + pezzi[1]
        m = re.fullmatch(r"A\. (.*)\nB\. (.*)\nC\. (.*)\nD\. (.*)", resto, re.S)
        assert m, i
        k = (c["materia"], q.strip(), tuple(x.strip() for x in m.groups()))
        js = chiave.get(k, [])
        if not js:
            mancanti += 1
            corrisp.append(None)
            continue
        if len(js) > 1:
            ambigui += 1
        # fra i candidati identici prendi quello con la stessa lettera, poi il piu' vicino all'indice
        js2 = [j for j in js if LETTERE[t[j]["answer"]] == c["lettera"]] or js
        corrisp.append(min(js2, key=lambda j: abs(j - i)))
    stessa_pos = sum(1 for i, j in enumerate(corrisp) if j == i)
    lettera_ok = sum(1 for i, j in enumerate(corrisp) if j is not None and LETTERE[t[j]["answer"]] == lettere[i])
    out["mappa"] = {"compiti": len(compiti), "trovati_nel_parquet": len(compiti) - mancanti, "mancanti": mancanti,
                    "con_duplicati_identici": ambigui, "stessa_posizione_i": stessa_pos,
                    "lettera_uguale_al_parquet": lettera_ok}
    campione = sorted(random.Random(SEME).sample(range(2000), 20))
    out["campione_20"] = [{"i": i, "bin_materia": materie_bin[i], "bin_lettera": lettere[i],
                           "pq_riga": corrisp[i], "pq_materia": t[i]["subject"],
                           "pq_lettera": LETTERE[t[i]["answer"]],
                           "domanda": t[i]["question"][:60]} for i in campione]
    conteggio_materie: dict[str, int] = {}
    for m_ in materie_bin:
        conteggio_materie[m_] = conteggio_materie.get(m_, 0) + 1
    out["materie"] = conteggio_materie
    out["lettere_giuste"] = {l: lettere.count(l) for l in LETTERE}

    # ── registri
    R = {}
    for nome, rel in REGISTRI.items():
        R[nome] = leggi_registro(RADICE / rel)
    out["registri"] = {n: {"giuste": r["giuste"], "final": r["final"], "coerente": r["coerente"],
                           "residuo_cumulata": r["residuo_cumulata"], "residuo_differenza": r["residuo_differenza"]}
                       for n, r in R.items()}
    C = {n: r["c"] for n, r in R.items()}
    L = np.array(lettere)
    M = np.array(materie_bin)

    # accuratezza per lettera
    out["acc_per_lettera"] = {n: {l: round(100 * C[n][L == l].mean(), 2) for l in LETTERE} for n in C}

    # 2x2 B vs B150
    x, y = C["B"], C["B150"]
    out["2x2"] = {"BB150_11": int(((x == 1) & (y == 1)).sum()), "B1_B150_0": int(((x == 1) & (y == 0)).sum()),
                  "B0_B150_1": int(((x == 0) & (y == 1)).sum()), "BB150_00": int(((x == 0) & (y == 0)).sum())}
    b, c, p = mcnemar(x, y)
    out["B_B150"] = {"wald": wald_appaiato(x, y), "boot": boot_appaiato(x, y, rng), "perse": b, "guadagnate": c,
                     "p_mcnemar": p}

    # per lettera, tutte le coppie
    coppie = [("B", "B150"), ("base", "B"), ("base", "B150"), ("base", "A"), ("base", "B0_19"),
              ("base", "B20_39"), ("base", "v20"), ("base", "Q2_K"), ("v20", "B150"), ("B", "A")]
    per_coppia = {}
    for r_, s_ in coppie:
        x, y = C[r_], C[s_]
        riga = {"delta_tot": round(100 * (y.mean() - x.mean()), 2)}
        tab_persa = []
        for l in LETTERE:
            mk = L == l
            d, lo, hi = wald_appaiato(x[mk], y[mk])
            blo, bhi = boot_appaiato(x[mk], y[mk], rng, 10000)
            bb, cc, pp = mcnemar(x[mk], y[mk])
            riga[l] = {"n": int(mk.sum()), "rif": round(100 * x[mk].mean(), 1), "conf": round(100 * y[mk].mean(), 1),
                       "delta": round(d, 2), "wald": [round(lo, 2), round(hi, 2)], "boot": [round(blo, 2), round(bhi, 2)],
                       "perse": bb, "guadagnate": cc, "p": pp}
            tab_persa.append([bb, int((x[mk] == 1).sum()) - bb])
        chi2, gl, pv = chi2_omogeneita(np.array(tab_persa))
        riga["lettere_chi2_gl"] = round(chi2 / gl, 2)
        riga["lettere_p"] = pv
        tab_m = []
        for m_ in sorted(set(materie_bin)):
            mk = M == m_
            bb = int(((x == 1) & (y == 0) & mk).sum())
            tab_m.append([bb, int(((x == 1) & mk).sum()) - bb])
        chi2m, glm, pvm = chi2_omogeneita(np.array(tab_m))
        riga["materie_chi2_gl"] = round(chi2m / glm, 2)
        riga["materie_p"] = pvm
        per_coppia[f"{r_}->{s_}"] = riga
    out["per_coppia"] = per_coppia

    # ── materie B -> B150, con e senza lettera (LRT logistico)
    x, y = C["B"], C["B150"]
    livelli_l = list(LETTERE)
    livelli_m = sorted(set(materie_bin))
    disc = x != y
    ydir = (x[disc] == 1).astype(float)  # 1 = persa, 0 = guadagnata
    Ld = [lettere[i] for i in np.where(disc)[0]]
    Md = [materie_bin[i] for i in np.where(disc)[0]]
    out["direzione_materia_senza_lettera"] = lrt(ydir, [], dummies(Md, livelli_m))
    out["direzione_materia_dato_lettera"] = lrt(ydir, [dummies(Ld, livelli_l)], dummies(Md, livelli_m))
    out["direzione_lettera_dato_materia"] = lrt(ydir, [dummies(Md, livelli_m)], dummies(Ld, livelli_l))
    sa = x == 1
    yp = (y[sa] == 0).astype(float)
    Ls = [lettere[i] for i in np.where(sa)[0]]
    Ms = [materie_bin[i] for i in np.where(sa)[0]]
    out["persa_materia_dato_lettera"] = lrt(yp, [dummies(Ls, livelli_l)], dummies(Ms, livelli_m))
    out["persa_lettera_dato_materia"] = lrt(yp, [dummies(Ms, livelli_m)], dummies(Ls, livelli_l))

    # netto per materia: osservato contro atteso da lettere e contro atteso da sola taglia
    tassi_p = {l: ((x == 1) & (y == 0) & (L == l)).sum() / ((x == 1) & (L == l)).sum() for l in LETTERE}
    tassi_g = {l: ((x == 0) & (y == 1) & (L == l)).sum() / ((x == 0) & (L == l)).sum() for l in LETTERE}
    pp_all = ((x == 1) & (y == 0)).sum() / (x == 1).sum()
    pg_all = ((x == 0) & (y == 1)).sum() / (x == 0).sum()
    oss, att_l, att_n = [], [], []
    per_mat = {}
    for m_ in livelli_m:
        mk = M == m_
        o_ = int(((x == 1) & (y == 0) & mk).sum() - ((x == 0) & (y == 1) & mk).sum())
        al = sum(((x == 1) & mk & (L == l)).sum() * tassi_p[l] - ((x == 0) & mk & (L == l)).sum() * tassi_g[l]
                 for l in LETTERE)
        an = ((x == 1) & mk).sum() * pp_all - ((x == 0) & mk).sum() * pg_all
        oss.append(o_); att_l.append(float(al)); att_n.append(float(an))
        d, lo, hi = wald_appaiato(x[mk], y[mk])
        bb, cc, pv = mcnemar(x[mk], y[mk])
        per_mat[m_] = {"n": int(mk.sum()), "B": round(100 * x[mk].mean(), 1), "B150": round(100 * y[mk].mean(), 1),
                       "delta": round(d, 2), "wald": [round(lo, 2), round(hi, 2)], "perse": bb, "guadagnate": cc,
                       "p": pv, "netto_atteso_lettere": round(float(al), 2), "netto_atteso_taglia": round(float(an), 2)}
    oss_a = np.array(oss, float)
    out["per_materia"] = per_mat
    out["netto_materia"] = {
        "r_oss_vs_atteso_lettere": float(np.corrcoef(oss_a, att_l)[0, 1]),
        "r_oss_vs_atteso_taglia": float(np.corrcoef(oss_a, att_n)[0, 1]),
        "ssr_lettere": float(((oss_a - np.array(att_l)) ** 2).sum()),
        "ssr_taglia": float(((oss_a - np.array(att_n)) ** 2).sum()),
    }
    in_calo = sum(1 for v in per_mat.values() if v["perse"] > v["guadagnate"])
    in_crescita = sum(1 for v in per_mat.values() if v["perse"] < v["guadagnate"])
    out["segno"] = {"calo": in_calo, "crescita": in_crescita,
                    "p": float(stats.binomtest(in_calo, in_calo + in_crescita, 0.5).pvalue)}
    peggiori = sorted(livelli_m, key=lambda m_: per_mat[m_]["delta"])
    togli = {}
    for k in (3, 5):
        mk = ~np.isin(M, peggiori[:k])
        d, lo, hi = wald_appaiato(x[mk], y[mk])
        togli[k] = {"tolte": peggiori[:k], "n": int(mk.sum()), "delta": round(d, 2), "wald": [round(lo, 2), round(hi, 2)],
                    "p": mcnemar(x[mk], y[mk])[2]}
    out["togli_peggiori"] = togli

    # ── domande che base, v20 e Q2_K sanno tutti e tre
    tutti = (C["base"] == 1) & (C["v20"] == 1) & (C["Q2_K"] == 1)
    facili = {}
    for gruppo in ("AB", "CD"):
        mk = tutti & np.isin(L, list(gruppo)) & (x == 1)
        k_ = int((mk & (y == 0)).sum()); n_ = int(mk.sum())
        ci = stats.binomtest(k_, n_).proportion_ci(method="wilson")
        facili[gruppo] = {"perse": k_, "B_giuste": n_, "pct": round(100 * k_ / n_, 2),
                          "wilson": [round(100 * ci.low, 2), round(100 * ci.high, 2)]}
    out["facili_tre_riferimenti"] = facili

    # ── sovrapposizione: atteso uniforme contro atteso stratificato (lettera x materia)
    sov = {}
    base_g = C["base"] == 1
    for a_, b_ in (("B", "B150"), ("v20", "Q2_K"), ("B", "A"), ("B150", "A")):
        pa = base_g & (C[a_] == 0)
        pb = base_g & (C[b_] == 0)
        oss_ = int((pa & pb).sum())
        att_u = pa.sum() * pb.sum() / base_g.sum()
        att_s = 0.0
        for l in LETTERE:
            for m_ in livelli_m:
                s = base_g & (L == l) & (M == m_)
                if s.sum():
                    att_s += (pa & s).sum() * (pb & s).sum() / s.sum()
        sov[f"{a_}.{b_}"] = {"comuni": oss_, "atteso_uniforme": round(float(att_u), 1),
                             "rapporto_uniforme": round(oss_ / att_u, 2),
                             "atteso_lettera_x_materia": round(float(att_s), 1),
                             "rapporto_strat": round(oss_ / att_s, 2)}
    out["sovrapposizione"] = sov

    # ── B: preferenza di B misurata a parita' di difficolta' (domande che base, v20, Q2_K sanno)
    facile = (C["base"] == 1) & (C["v20"] == 1) & (C["Q2_K"] == 1)
    out["acc_facili_per_lettera"] = {}
    for n in ("B", "B150", "A", "B0_19", "B20_39"):
        riga = {}
        for l in LETTERE:
            mk = facile & (L == l)
            k_ = int(C[n][mk].sum()); ci = stats.binomtest(k_, int(mk.sum())).proportion_ci(method="wilson")
            riga[l] = [round(100 * k_ / mk.sum(), 1), round(100 * ci.low, 1), round(100 * ci.high, 1), int(mk.sum())]
        out["acc_facili_per_lettera"][n] = riga
    mkD = facile & (L == "D")
    out["facili_D_B_B150_mcnemar"] = mcnemar(C["B"][mkD], C["B150"][mkD])
    # tassi di perdita C contro D fra le domande che B sa (tutte)
    x, y = C["B"], C["B150"]
    tabCD = [[int(((x == 1) & (y == 0) & (L == l)).sum()), int(((x == 1) & (y == 1) & (L == l)).sum())] for l in "CD"]
    out["perdita_C_vs_D_fisher"] = {"C": tabCD[0], "D": tabCD[1], "p": float(stats.fisher_exact(tabCD)[1])}
    # materia a parita' di lettera E di difficolta' (difficolta' = quanti fra base, v20, Q2_K la sanno)
    diff = [str(v) for v in (C["base"] + C["v20"] + C["Q2_K"]).astype(int)]
    sa = x == 1
    yp = (y[sa] == 0).astype(float)
    Ls = [lettere[i] for i in np.where(sa)[0]]
    Ms = [materie_bin[i] for i in np.where(sa)[0]]
    Ds = [diff[i] for i in np.where(sa)[0]]
    out["persa_materia_dato_lettera_e_difficolta"] = lrt(
        yp, [dummies(Ls, livelli_l), dummies(Ds, ["0", "1", "2", "3"])], dummies(Ms, livelli_m))
    out["quota_AB_per_materia"] = {m_: round(float(np.isin(L[M == m_], ["A", "B"]).mean()), 2) for m_ in livelli_m}

    # ── controllo: altri modelli sullo STESSO file, non forgiati globalmente (quantizzazioni sorelle)
    altri = sorted(p for p in (RADICE / "thor").glob("mmlu_*_00000.log")
                   if "rivale_q2k" not in p.name and "v20_downaggr3" not in p.name)
    altri.append(RADICE / "stato/q0_c4/q0_mmlu.log")
    ctrl = {}
    xb = C["base"]
    for p in altri:
        try:
            rr = leggi_registro(p)
        except AssertionError as e:
            ctrl[p.name] = {"scartato": str(e)[:80]}
            continue
        yy = rr["c"]
        po = float((xb == yy).mean())
        pe = float(xb.mean() * yy.mean() + (1 - xb.mean()) * (1 - yy.mean()))
        tab = []
        riga = {"acc": round(100 * yy.mean(), 2), "delta_tot": round(100 * (yy.mean() - xb.mean()), 2),
                "kappa_con_base": round((po - pe) / (1 - pe), 3), "coerente": rr["coerente"]}
        for l in LETTERE:
            mk = L == l
            bb = int(((xb == 1) & (yy == 0) & mk).sum())
            tab.append([bb, int(((xb == 1) & mk).sum()) - bb])
            riga[l] = round(100 * (yy[mk].mean() - xb[mk].mean()), 1)
        chi2, gl, pv = chi2_omogeneita(np.array(tab))
        riga["lettere_chi2_gl"] = round(chi2 / gl, 2)
        riga["lettere_p"] = pv
        ctrl[p.name] = riga
    out["controllo_altri_modelli"] = ctrl

    USCITA.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float))
    print(json.dumps(out, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    sys.exit(main())
