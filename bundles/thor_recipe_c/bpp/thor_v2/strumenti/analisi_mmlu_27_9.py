#!/usr/bin/env python3
"""Analisi MMLU domanda per domanda del 27/9/2026: dove Thor-T (B, EB2 p12) perde contro Q2_K e v20.

Sola lettura dei registri, sola CPU. Correttezza per domanda di Q2_K / B[0-19] / B[20-39] dalla mappa gia' verificata
(risultati/mmlu_per_domanda.json, chiave modelli.*.corrette); logprob dalle righe MCLP dei registri in
@@ROOT:models@@/gguf/tony-tern/mmlu_logprob (Corsair) e .../aragorn (Aragorn). I confronti con i checkpoint EB2 sono APPAIATI
sulla stessa macchina (Aragorn: |dlogprob| medio ~0,45 nat fra le due macchine, valuta_checkpoint_aragorn.sh).

Uscita: stampa JSON con tutti i numeri (il documento thor_v2/analisi_mmlu_27_9.md li riporta).
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np

FUCINA = Path("@@ROOT:forge@@")
MAPPA = FUCINA / "thor_v2/risultati/mmlu_per_domanda.json"
R = Path("@@ROOT:models@@/gguf/tony-tern/mmlu_logprob")
N = 2000
REG = {
    "teacher": R / "teacher.log", "base": R / "base.log", "v20": R / "v20.log", "B": R / "B.log",
    "B150": R / "B150.log", "p10pv_parz": R / "EB2_p10_pv.log",
    "B_ar": R / "aragorn/B_aragorn.log", "p2_ar": R / "aragorn/EB2-migliore_000002.log",
    "p12_ar": R / "aragorn/EB2-migliore_000012.log",
}
# categorie ufficiali MMLU (Hendrycks, categories.py) per le 15 materie presenti nei primi 2000 compiti
CAT = {"abstract_algebra": "STEM-mat", "college_mathematics": "STEM-mat", "college_physics": "STEM-fis",
       "conceptual_physics": "STEM-fis", "astronomy": "STEM-fis", "college_chemistry": "STEM-chim",
       "college_biology": "STEM-bio", "college_computer_science": "STEM-inf", "computer_security": "STEM-inf",
       "electrical_engineering": "STEM-ing", "anatomy": "salute", "clinical_knowledge": "salute",
       "college_medicine": "salute", "business_ethics": "altro", "econometrics": "sociali"}
# asse «calcolo/ragionamento» contro «conoscenza fattuale» (classificazione dichiarata qui, non ufficiale)
CALCOLO = {"abstract_algebra", "college_mathematics", "college_physics", "college_chemistry", "econometrics",
           "college_computer_science", "electrical_engineering"}


def leggi_mclp(p: Path) -> tuple[np.ndarray, np.ndarray]:
    lp = np.full((N, 4), np.nan)
    gi = np.full(N, -1)
    for riga in p.read_text(errors="replace").splitlines():
        if not riga.startswith("MCLP\t"):
            continue
        c = riga.split("\t")
        i = int(c[1])
        gi[i] = int(c[2])
        lp[i] = [float(x) for x in c[5:9]]
    return lp, gi


def logsm4(lp: np.ndarray) -> np.ndarray:
    m = lp.max(axis=1, keepdims=True)
    return lp - (m + np.log(np.exp(lp - m).sum(axis=1, keepdims=True)))


def margine(lp: np.ndarray, oro: np.ndarray) -> np.ndarray:
    """logprob della giusta meno il massimo delle altre tre (nat): > 0 giusta, < 0 sbagliata."""
    g = lp[np.arange(len(oro)), oro]
    alt = lp.copy()
    alt[np.arange(len(oro)), oro] = -np.inf
    return g - alt.max(axis=1)


def mcnemar_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    s = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * s)


def prior_glob(S4: np.ndarray, oro: np.ndarray) -> np.ndarray:
    """preferenza globale BILANCIATA per lettera giusta (come mmlu_depurato.py, variante glob)."""
    medie = np.stack([S4[oro == k].mean(axis=0) for k in range(4)]).mean(axis=0)
    return medie - medie.mean()


def main() -> int:
    mappa = json.loads(MAPPA.read_text())
    mat = np.array(mappa["materia"])
    oro = np.array(["ABCD".index(x) for x in mappa["lettera"]])
    materie = list(dict.fromkeys(mappa["materia"]))
    corr = {k: np.array([ch == "1" for ch in v["corrette"]]) for k, v in mappa["modelli"].items()}
    LP, OK = {}, {}
    for k, p in REG.items():
        lp, gi = leggi_mclp(p)
        pres = ~np.isnan(lp[:, 0])
        if pres.sum() == N:
            assert (gi == oro).all(), k
        LP[k] = lp
        sc = np.where(pres, np.nan_to_num(lp, nan=-1e9).argmax(axis=1), -1)
        OK[k] = (sc == oro) & pres
    OK["Q2_K"] = corr["Q2_K"]
    OK["B_0_19"] = corr["B[0-19]"]
    OK["B_20_39"] = corr["B[20-39]"]
    assert (OK["B"] == corr["B"]).all() and (OK["v20"] == corr["v20"]).all()
    out: dict = {"acc": {k: 100 * float(v.mean()) for k, v in OK.items() if k != "p10pv_parz"}}

    # (1) per materia
    righe = []
    for m in materie:
        s = mat == m
        r = {"materia": m, "n": int(s.sum()), "cat": CAT[m], "calcolo": m in CALCOLO}
        for k in ("teacher", "Q2_K", "v20", "base", "B", "B_ar", "p12_ar", "B_0_19", "B_20_39"):
            r[k] = 100 * float(OK[k][s].mean())
        for rif in ("Q2_K", "v20"):
            for mod in ("B", "p12_ar"):
                perse = int((OK[rif] & ~OK[mod] & s).sum())
                guad = int((~OK[rif] & OK[mod] & s).sum())
                r[f"{mod}_vs_{rif}"] = {"perse": perse, "guad": guad, "netto": perse - guad,
                                        "p": mcnemar_p(perse, guad)}
        righe.append(r)
    out["materie"] = righe
    tot = {}
    for rif in ("Q2_K", "v20", "teacher"):
        for mod in ("B", "p12_ar"):
            perse = int((OK[rif] & ~OK[mod]).sum())
            guad = int((~OK[rif] & OK[mod]).sum())
            tot[f"{mod}_vs_{rif}"] = {"perse": perse, "guad": guad, "netto": perse - guad, "p": mcnemar_p(perse, guad)}
    out["totali"] = tot
    # omogeneita' del netto per materia contro Q2_K (tasso di perdita netta) — chi2 a permutazione sulle discordanti
    rng = np.random.default_rng(20260927)
    for mod in ("B", "p12_ar"):
        disc = OK["Q2_K"] != OK[mod]
        persa = (OK["Q2_K"] & ~OK[mod])[disc]
        md = mat[disc]

        def chi(pp: np.ndarray) -> float:
            c = 0.0
            f = pp.mean()
            for m in materie:
                s = md == m
                n = s.sum()
                if n == 0:
                    continue
                o = pp[s].sum()
                e = n * f
                c += (o - e) ** 2 / (e * (1 - f)) if 0 < f < 1 else 0
            return c
        c0 = chi(persa)
        cnt = sum(chi(rng.permutation(persa)) >= c0 for _ in range(20000))
        out[f"omog_direzione_{mod}_vs_Q2_K"] = {"chi2": c0, "gl": len(materie) - 1, "p_perm": (cnt + 1) / 20001}
    # per gruppi
    grp = {}
    for nome, sel in (("STEM", np.array([CAT[m].startswith("STEM") for m in mat])),
                      ("salute+altro+sociali", np.array([not CAT[m].startswith("STEM") for m in mat])),
                      ("calcolo", np.array([m in CALCOLO for m in mat])),
                      ("conoscenza", np.array([m not in CALCOLO for m in mat]))):
        g = {"n": int(sel.sum())}
        for k in ("teacher", "Q2_K", "v20", "base", "B", "B_ar", "p12_ar", "B_0_19", "B_20_39"):
            g[k] = 100 * float(OK[k][sel].mean())
        for mod in ("B", "p12_ar"):
            perse = int((OK["Q2_K"] & ~OK[mod] & sel).sum())
            guad = int((~OK["Q2_K"] & OK[mod] & sel).sum())
            g[f"{mod}_vs_Q2_K_netto"] = perse - guad
        grp[nome] = g
    out["gruppi"] = grp

    # (2) margini: B sbaglia, Q2_K azzecca
    def blocco_margini(chiave: str, okm: np.ndarray, lp: np.ndarray) -> dict:
        mg = margine(lp, oro)
        S4 = logsm4(lp)
        pr = prior_glob(S4, oro)
        mgd = margine(S4 - pr, oro)
        rank = (lp > lp[np.arange(N), oro][:, None]).sum(axis=1) + 1
        sel = OK["Q2_K"] & ~okm
        sel2 = ~OK["Q2_K"] & ~okm
        d = {"n": int(sel.sum())}
        x = mg[sel]
        d["mediana"] = float(np.median(x))
        d["quartili"] = [float(np.percentile(x, 25)), float(np.percentile(x, 75))]
        for t in (0.25, 0.5, 1.0, 2.0):
            d[f"entro_{t}"] = int((x > -t).sum())
        d["rango_giusta"] = {str(r): int((rank[sel] == r).sum()) for r in (2, 3, 4)}
        d["scelta_lettera"] = {L: int((lp[sel].argmax(axis=1) == i).sum()) for i, L in enumerate("ABCD")}
        d["depurata_diventa_giusta"] = int((mgd[sel] > 0).sum())
        d["teacher_giusta"] = int(OK["teacher"][sel].sum())
        d["mediana_entrambi_sbagliati"] = float(np.median(mg[sel2]))
        d["n_entrambi_sbagliati"] = int(sel2.sum())
        # quante sbagliate in tutto, e quante vicine: carburante per arrivare a 1601 giuste (80,05)
        sb = ~okm
        d["sbagliate_tot"] = int(sb.sum())
        for t in (0.25, 0.5, 1.0):
            d[f"sbagliate_entro_{t}"] = int((mg[sb] > -t).sum())
        d["giuste_entro_0.25"] = int(((mg > 0) & (mg < 0.25)).sum())
        d["giuste_entro_0.5"] = int(((mg > 0) & (mg < 0.5)).sum())
        d["prior_glob_offset"] = [float(v) for v in pr]
        # per materia: mediana del margine sulle «B sbaglia, Q2_K sa»
        d["per_materia"] = {m: {"n": int((sel & (mat == m)).sum()),
                                "mediana": float(np.median(mg[sel & (mat == m)])) if (sel & (mat == m)).any() else None}
                            for m in materie}
        return d
    out["margini"] = {"B": blocco_margini("B", OK["B"], LP["B"]),
                      "p12_ar": blocco_margini("p12_ar", OK["p12_ar"], LP["p12_ar"]),
                      "B_ar": blocco_margini("B_ar", OK["B_ar"], LP["B_ar"]),
                      "v20": blocco_margini("v20", OK["v20"], LP["v20"])}
    # margine del teacher sulle stesse domande (quanto e' «facile» per il modello pieno)
    mgT = margine(LP["teacher"], oro)
    selB = OK["Q2_K"] & ~OK["B"]
    out["margini"]["teacher_su_B_sbaglia_Q2K_sa"] = {"mediana": float(np.median(mgT[selB])),
                                                     "giuste": int((mgT[selB] > 0).sum())}

    # (3) B_ar -> p12_ar (e p2)
    def cambi(a: str, b: str) -> dict:
        sa = LP[a].argmax(axis=1)
        sb = LP[b].argmax(axis=1)
        ch = sa != sb
        g = int((ch & ~OK[a] & OK[b]).sum())
        p = int((ch & OK[a] & ~OK[b]).sum())
        ss = int((ch & ~OK[a] & ~OK[b]).sum())
        mgA = LP[a].max(axis=1) - np.sort(LP[a], axis=1)[:, -2]   # distacco fra 1a e 2a scelta in a
        tA = LP["teacher"].argmax(axis=1)
        d = {"cambiate": int(ch.sum()), "sbagliata->giusta": g, "giusta->sbagliata": p, "sbagliata->sbagliata": ss,
             "netto": g - p, "p_segno": mcnemar_p(g, p),
             "distacco_top2_mediana_cambiate": float(np.median(mgA[ch])),
             "distacco_top2_mediana_tutte": float(np.median(mgA)),
             "cambiate_con_distacco_<0.25": int((ch & (mgA < 0.25)).sum()),
             "tutte_con_distacco_<0.25": int((mgA < 0.25).sum()),
             "cambiate_verso_teacher": int((ch & (sb == tA)).sum()),
             "cambiate_via_da_teacher": int((ch & (sa == tA)).sum()),
             "accordo_teacher_a": int((sa == tA).sum()), "accordo_teacher_b": int((sb == tA).sum()),
             "lettera_prima": {L: int((sa[ch] == i).sum()) for i, L in enumerate("ABCD")},
             "lettera_dopo": {L: int((sb[ch] == i).sum()) for i, L in enumerate("ABCD")}}
        # spostamento dei logprob a 4 (centrati) verso il teacher: correlazione per compito
        A4, B4, T4 = logsm4(LP[a]), logsm4(LP[b]), logsm4(LP["teacher"])
        dlt = (B4 - B4.mean(1, keepdims=True)) - (A4 - A4.mean(1, keepdims=True))
        dir_ = (T4 - T4.mean(1, keepdims=True)) - (A4 - A4.mean(1, keepdims=True))
        d["corr_delta_verso_teacher"] = float(np.corrcoef(dlt.ravel(), dir_.ravel())[0, 1])
        d["delta_lp4_giusta_media"] = float((B4 - A4)[np.arange(N), oro].mean())
        d["delta_margine_medio"] = float((margine(LP[b], oro) - margine(LP[a], oro)).mean())
        d["delta_lp4_per_lettera"] = [float(v) for v in (dlt.mean(0))]
        return d
    out["cambi"] = {"B_ar->p12_ar": cambi("B_ar", "p12_ar"), "B_ar->p2_ar": cambi("B_ar", "p2_ar"),
                    "p2_ar->p12_ar": cambi("p2_ar", "p12_ar"), "B->B_ar(macchina)": cambi("B", "B_ar")}
    s2 = LP["B_ar"].argmax(1) != LP["p2_ar"].argmax(1)
    s12 = LP["B_ar"].argmax(1) != LP["p12_ar"].argmax(1)
    out["cambi"]["sovrapposizione_p2_p12"] = {"entrambe": int((s2 & s12).sum()), "solo_p2": int((s2 & ~s12).sum()),
                                             "solo_p12": int((~s2 & s12).sum())}
    ch = s12
    out["cambi"]["p12_cambiate_per_materia"] = {
        m: {"n": int((ch & (mat == m)).sum()), "netto": int((ch & (mat == m) & OK["p12_ar"]).sum()
                                                            - (ch & (mat == m) & OK["B_ar"]).sum())} for m in materie}
    # p10 PV parziale (Corsair) contro B (Corsair) sugli stessi compiti
    pres = ~np.isnan(LP["p10pv_parz"][:, 0])
    out["p10pv_parziale"] = {"compiti": int(pres.sum()), "acc_p10": 100 * float(OK["p10pv_parz"][pres].mean()),
                             "acc_B_stessi": 100 * float(OK["B"][pres].mean()),
                             "acc_Q2K_stessi": 100 * float(OK["Q2_K"][pres].mean()),
                             "scelte_uguali_B": int((LP["p10pv_parz"][pres].argmax(1) == LP["B"][pres].argmax(1)).sum())}
    # (4) meta' di strati: il divario per materia sta negli esperti degli strati 20-39?
    out["bisezione"] = {m: {"base": 100 * float(OK["base"][mat == m].mean()),
                            "B_0_19": 100 * float(OK["B_0_19"][mat == m].mean()),
                            "B_20_39": 100 * float(OK["B_20_39"][mat == m].mean()),
                            "B": 100 * float(OK["B"][mat == m].mean())} for m in materie}
    json.dump(out, sys.stdout, indent=1, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
