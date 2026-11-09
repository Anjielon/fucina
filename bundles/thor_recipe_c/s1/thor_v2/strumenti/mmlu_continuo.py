#!/usr/bin/env python3
"""MMLU CONTINUO — misure continue del progresso sui registri MCLP gia' salvati (27/9/2026).

## Perche' esiste
Il passo 12 di EB2 fa 72,80 contro 72,50 di B (Aragorn, +0,30, nel rumore), mentre la KL held-out e' scesa del 10,8 %.
L'accuratezza e' una misura a SOGLIA (argmax sui 4 logprob): puo' restare piatta mentre il modello migliora —
Schaeffer, Miranda, Koyejo, «Are Emergent Abilities of Large Language Models a Mirage?», NeurIPS 2023, arXiv 2304.15004
(metriche lineari/continue come il Brier score danno cambiamenti lisci; accuratezza e multiple-choice grade danno salti
apparenti).
Schaeffer et al., arXiv 2406.04391, §4: la catena log p_Vocab(giusta) → p_Vocab → p_Choices (normalizzata sulle 4)
→ Brier → accuratezza PERDE correlazione col calcolo a ogni anello, perche' dall'anello p_Choices in poi entra la massa
sulle SBAGLIATE. Percio' qui si misurano tutti gli anelli, non solo quello normalizzato.

## Misure per compito (i = compito, g = lettera giusta, lp = i 4 log p(lettera | prompt) sul vocabolario intero)
- logp_scelte  = log softmax_4(lp)[g]                 (log p_Choices della giusta; media = −entropia incrociata sulle 4)
- p_scelte     = exp(logp_scelte)                     (probabilita' della giusta normalizzata sulle 4)
- brier        = Σ_k (p_k − 1[k=g])²                  (multiclasse sulle 4, 0 = perfetto, piu' BASSO e' meglio)
- margine      = lp[g] − max_{k≠g} lp[k]              (nat; > 0 giusta; dipende dalla temperatura dei logit)
- margine_T    = margine / temperatura propria (stimata incrociata come sotto): il margine a calibrazione uguale
- coppie       = frazione delle 3 sbagliate sotto la giusta (pari merito ½): discriminazione pura, invariante alla
                 temperatura
- logp_vocab   = lp[g]                                (log p_Vocab della giusta, anello piu' «prevedibile», 2406.04391)
- log_massa    = log Σ_k exp(lp[k])                    (massa sulle 4 lettere)
- logp_scelte_T= logp_scelte a temperatura propria, stimata INCROCIATA (pari → dispari e viceversa): separa la
                 calibrazione (scala dei logit) dalla discriminazione
- acc          = argmax (primo massimo) == g           (ricontrollata contro il Final result)

## Intervalli
Bootstrap sui compiti STRATIFICATO per materia (la composizione del banco e' fissa: 15 materie in blocchi), APPAIATO
(gli stessi indici per tutti i registri), 10000 ricampionamenti, percentili 2,5/97,5. Le differenze, i rapporti e le
frazioni di distanza sono ricalcolati dentro ogni ricampionamento; per le misure _T anche la TEMPERATURA e' ri-stimata
in ogni ricampionamento (bootstrap_T, stessi conteggi verificati), cosi' i loro IC non sono ottimisti.

## Frazione della distanza verso il maestro
Il maestro (Q8, 82,00) ha logprob solo sul Corsair; p2 e p12 solo su Aragorn. Frazione principale, APPAIATA per
macchina:
  f = [M(p12, Aragorn) − M(B, Aragorn)] / [M(maestro, Corsair) − M(B, Corsair)]
(numeratore e denominatore sono differenze fra registri della stessa macchina). Q2_K non ha logprob (solo 0/1 per
compito):
per Q2_K si da' la taratura fra modelli (accuratezza contro misura continua sui 5 registri Corsair completi).
Controllo interamente Corsair: p10 PV (registro parziale, primi 1214 compiti) contro B e maestro sugli stessi compiti.

Sola lettura, sola CPU. Uso (CPU confinata, la GPU del Corsair e' della corsa EB2):
    nice -n 19 env OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python3 thor_v2/strumenti/mmlu_continuo.py [--json USCITA]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mmlu_depurato as MD  # noqa: E402  (sola lettura: lettore dei registri, banco, mappa)

R = MD.SB
N = MD.N
SEME = 20260927
N_BOOT = 10000
BLOCCO = 500
REG = {  # chiave: (registro, macchina)
    "maestro": (R / "teacher.log", "Corsair"), "base": (R / "base.log", "Corsair"), "v20": (R / "v20.log", "Corsair"),
    "B150": (R / "B150.log", "Corsair"), "B_co": (R / "B.log", "Corsair"),
    "B_ar": (R / "aragorn/B_aragorn.log", "Aragorn"), "p2_ar": (R / "aragorn/EB2-migliore_000002.log", "Aragorn"),
    "p12_ar": (R / "aragorn/EB2-migliore_000012.log", "Aragorn"),
    "e16_ar": (R / "aragorn/EB2-migliore_ema_000016.log", "Aragorn"),   # 27/9 16:37: EMA del passo 16 (CAKLD+F32+LoRA dal 14)
}
PARZIALE = ("p10pv_co", R / "EB2_p10_pv.log")                     # Corsair, primi 1214 compiti
REGISTRO_EB2 = Path("@@ROOT:lavoro@@/e2e/allena/EB2/registro.log")    # held-out KL per passo (sola lettura)
METRICHE = ["acc", "logp_scelte", "logp_scelte_T", "p_scelte", "brier", "margine", "margine_T", "coppie", "logp_vocab",
            "log_massa"]
MEGLIO_BASSO = {"brier"}
# differenze (a − b): (nome, a, b)
DELTE = [("e16−B (Aragorn)", "e16_ar", "B_ar"), ("e16−p12 (Aragorn)", "e16_ar", "p12_ar"), ("p12−B (Aragorn)", "p12_ar", "B_ar"), ("p2−B (Aragorn)", "p2_ar", "B_ar"),
         ("p12−p2 (Aragorn)", "p12_ar", "p2_ar"),
         ("solo macchina: B Aragorn − B Corsair", "B_ar", "B_co"), ("maestro−B (Corsair)", "maestro", "B_co"),
         ("v20−B (Corsair)", "v20", "B_co"), ("base−B (Corsair)", "base", "B_co")]


def leggi_parziale(p: Path) -> tuple[np.ndarray, np.ndarray]:
    """Registro in corso: righe MCLP contigue da 0, stessi controlli di MD.leggi_registro."""
    lp, gi = [], []
    for riga in p.read_text(encoding="utf-8").splitlines():
        if riga.startswith("MCLP\t"):
            c = riga.split("\t")
            if len(c) != 13 or int(c[1]) != len(lp) or c[4] != "4" or c[9:13] != ["1", "1", "1", "1"]:
                raise SystemExit(f"⛔ {p}: riga MCLP anomala: {riga!r}")
            gi.append(int(c[2]))
            lp.append([float(x) for x in c[5:9]])
    lp_a = np.array(lp, dtype=np.float32).astype(np.float64)
    if not np.isfinite(lp_a).all() or (lp_a > 0).any():
        raise SystemExit(f"⛔ {p}: logprob non finiti/positivi")
    return lp_a, np.array(gi)


def nll_t(lp: np.ndarray, oro: np.ndarray, t: float) -> float:
    return -float(MD.logsm(lp / t)[np.arange(len(oro)), oro].mean())


def temperatura(lp: np.ndarray, oro: np.ndarray) -> float:
    """Temperatura che minimizza l'entropia incrociata sulle 4 (sezione aurea su log t in [log 0,1; log 10])."""
    a, b = np.log(0.1), np.log(10.0)
    phi = (np.sqrt(5) - 1) / 2
    for _ in range(80):
        c, d = b - phi * (b - a), a + phi * (b - a)
        if nll_t(lp, oro, np.exp(c)) < nll_t(lp, oro, np.exp(d)):
            b = d
        else:
            a = c
    return float(np.exp((a + b) / 2))


def metriche(lp: np.ndarray, oro: np.ndarray) -> tuple[dict, dict]:
    n = len(oro)
    ar = np.arange(n)
    S = MD.logsm(lp)
    P = np.exp(S)
    uno = np.zeros_like(P)
    uno[ar, oro] = 1.0
    g = lp[ar, oro]
    altre = lp.copy()
    altre[ar, oro] = -np.inf
    mx = lp.max(axis=1)
    pari, dispari = ar % 2 == 0, ar % 2 == 1
    t_p, t_d = temperatura(lp[pari], oro[pari]), temperatura(lp[dispari], oro[dispari])
    lsT = np.empty(n)
    lsT[dispari] = MD.logsm(lp[dispari] / t_p)[np.arange(dispari.sum()), oro[dispari]]
    lsT[pari] = MD.logsm(lp[pari] / t_d)[np.arange(pari.sum()), oro[pari]]
    marg = g - altre.max(axis=1)
    margT = np.where(pari, marg / t_d, marg / t_p)
    m = {"acc": (lp.argmax(axis=1) == oro).astype(np.float64), "logp_scelte": S[ar, oro], "logp_scelte_T": lsT,
         "p_scelte": P[ar, oro], "brier": ((P - uno) ** 2).sum(axis=1), "margine": marg, "margine_T": margT,
         "coppie": ((g[:, None] > lp).sum(axis=1) + 0.5 * ((g[:, None] == lp).sum(axis=1) - 1)) / 3.0,
         "logp_vocab": g, "log_massa": mx + np.log(np.exp(lp - mx[:, None]).sum(axis=1))}
    return m, {"t_pari": t_p, "t_dispari": t_d, "t_tutti": temperatura(lp, oro)}


def conti_bootstrap(rng: np.random.Generator, strati: list[np.ndarray], n: int, b: int) -> np.ndarray:
    """(b, n) conteggi di un bootstrap stratificato (ogni strato ricampionato dentro se stesso)."""
    C = np.zeros(b * n)
    righe = np.arange(b)[:, None] * n
    for idx in strati:
        pesc = idx[rng.integers(0, len(idx), size=(b, len(idx)))]
        C += np.bincount((righe + pesc).ravel(), minlength=b * n)
    return C.reshape(b, n)


def bootstrap_medie(X: np.ndarray, strati: list[np.ndarray], seme: int) -> np.ndarray:
    """X: (n, K) valori per compito → (N_BOOT, K) medie ricampionate (stessi indici per tutte le colonne)."""
    rng = np.random.default_rng(seme)
    n = X.shape[0]
    out = []
    for inizio in range(0, N_BOOT, BLOCCO):
        C = conti_bootstrap(rng, strati, n, min(BLOCCO, N_BOOT - inizio))
        out.append(C @ X / n)
    return np.concatenate(out)


GRIGLIA_LOG_T = np.linspace(np.log(0.25), np.log(8.0), 801)     # passo 0,0043 in log t


def _t_pesata(lp: np.ndarray, oro: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Temperatura ottima per OGNI riga di pesi W (b, n). L'entropia incrociata pesata e' lineare nei pesi:
    F_b(t) = −Σ_i W_bi · g_i(t), con g_i(t) = log softmax₄(lp_i / t)[g] uguale per tutti i ricampionamenti → si calcola
    g su una griglia di log t UNA volta, F = W @ G (un prodotto di matrici), minimo sulla griglia + raffinamento
    parabolico sui 3 punti vicini (F e' liscia; l'accordo con la sezione aurea e' verificato in bootstrap_T)."""
    ar = np.arange(len(oro))
    G = MD.logsm(lp[None] / np.exp(GRIGLIA_LOG_T)[:, None, None])[:, ar, oro].T          # (n, griglia)
    F = -(W @ G) / W.sum(axis=1, keepdims=True)                                          # (b, griglia)
    k = F.argmin(axis=1)
    if (k == 0).any() or (k == len(GRIGLIA_LOG_T) - 1).any():
        raise SystemExit("⛔ temperatura ottima sul bordo della griglia [0,25; 8]")
    r = np.arange(F.shape[0])
    y0, y1, y2 = F[r, k - 1], F[r, k], F[r, k + 1]
    den = y0 - 2 * y1 + y2
    spost = np.where(den > 0, 0.5 * (y0 - y2) / np.where(den > 0, den, 1.0), 0.0)
    h = GRIGLIA_LOG_T[1] - GRIGLIA_LOG_T[0]
    return np.exp(GRIGLIA_LOG_T[k] + np.clip(spost, -1, 1) * h)


def bootstrap_T(lps: list[np.ndarray], oro: np.ndarray, strati: list[np.ndarray], seme: int,
                controllo: np.ndarray) -> np.ndarray:
    """Medie di logp_scelte_T e margine_T con la temperatura RI-STIMATA in ogni ricampionamento (incrociata pari ↔
    dispari come la stima puntuale). Stessi conteggi di bootstrap_medie(…, seme): stesso generatore, stessi blocchi —
    verificato confrontando la media ricampionata di logp_scelte con la colonna `controllo` (N_BOOT, len(lps)).
    Ritorna (N_BOOT, len(lps), 2)."""
    rng = np.random.default_rng(seme)
    n = len(oro)
    ar = np.arange(n)
    pari = ar % 2 == 0
    for lp in lps:                    # la griglia deve ritrovare la sezione aurea sui pesi unitari (stima puntuale)
        for msk in (pari, ~pari):
            tg = _t_pesata(lp[msk], oro[msk], np.ones((1, int(msk.sum()))))[0]
            ta = temperatura(lp[msk], oro[msk])
            assert abs(np.log(tg / ta)) < 1e-4, f"griglia {tg} contro sezione aurea {ta}"
    out = []
    for inizio in range(0, N_BOOT, BLOCCO):
        C = conti_bootstrap(rng, strati, n, min(BLOCCO, N_BOOT - inizio))
        blocco = np.empty((C.shape[0], len(lps), 2))
        for mi, lp in enumerate(lps):
            S = MD.logsm(lp)
            assert np.allclose(C @ S[ar, oro] / n, controllo[inizio:inizio + C.shape[0], mi]), "conteggi diversi"
            g = lp[ar, oro]
            altre = lp.copy()
            altre[ar, oro] = -np.inf
            marg = g - altre.max(axis=1)
            t_p = _t_pesata(lp[pari], oro[pari], C[:, pari])            # stimata sui pari → usata sui dispari
            t_d = _t_pesata(lp[~pari], oro[~pari], C[:, ~pari])
            ls_d = MD.logsm(lp[None, ~pari] / t_p[:, None, None])[:, np.arange((~pari).sum()), oro[~pari]]
            ls_p = MD.logsm(lp[None, pari] / t_d[:, None, None])[:, np.arange(pari.sum()), oro[pari]]
            blocco[:, mi, 0] = ((C[:, ~pari] * ls_d).sum(axis=1) + (C[:, pari] * ls_p).sum(axis=1)) / n
            blocco[:, mi, 1] = ((C[:, ~pari] * marg[~pari] / t_p[:, None]).sum(axis=1)
                                + (C[:, pari] * marg[pari] / t_d[:, None]).sum(axis=1)) / n
        out.append(blocco)
    return np.concatenate(out)


def ic(x: np.ndarray) -> list[float]:
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def kl_heldout() -> dict:
    """held-out KL per passo dal registro della corsa (sola lettura; vuoto se il file non c'e')."""
    out = {}
    if REGISTRO_EB2.exists():
        for riga in REGISTRO_EB2.read_text(encoding="utf-8").splitlines():
            m = re.search(r"passo (\d+)(?::|\s).*?held-out KL ([0-9.]+)|held-out al passo (\d+): KL ([0-9.]+)", riga)
            if m:
                passo, kl = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
                out[int(passo)] = float(kl)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    mappa = json.loads(MD.MAPPA.read_text(encoding="utf-8"))
    assert mappa["sha256_compiti"] == MD.SHA_BANCO
    oro = np.array([MD.LETTERE.index(x) for x in mappa["lettera"]])
    materia = np.array(mappa["materia"])
    strati = [np.nonzero(materia == s)[0] for s in sorted(set(mappa["materia"]))]
    q2k = np.array([c == "1" for c in mappa["modelli"]["Q2_K"]["corrette"]])

    per, temp, finali, lps = {}, {}, {}, {}
    for k, (p, _) in REG.items():
        r = MD.leggi_registro(p)
        if not (r["giusta"] == oro).all():
            raise SystemExit(f"⛔ {p}: lettera giusta diversa dalla mappa")
        lps[k] = r["lp32"].astype(np.float64)
        per[k], temp[k] = metriche(lps[k], oro)
        finali[k] = r["finale"]
        acc_fin = float(r["finale"].split()[0])
        if abs(100 * per[k]["acc"].mean() - acc_fin) > 0.01:
            raise SystemExit(f"⛔ {p}: accuratezza ricalcolata {100 * per[k]['acc'].mean():.2f} ≠ Final {acc_fin}")

    chiavi = list(REG)
    K = len(METRICHE)
    X = np.stack([per[k][m] for k in chiavi for m in METRICHE], axis=1)          # (N, modelli·metriche)
    BT = bootstrap_medie(X, strati, SEME).reshape(N_BOOT, len(chiavi), K)
    jl, jT, jM = METRICHE.index("logp_scelte"), METRICHE.index("logp_scelte_T"), METRICHE.index("margine_T")
    BTT = bootstrap_T([lps[k] for k in chiavi], oro, strati, SEME, BT[:, :, jl])
    BT[:, :, jT], BT[:, :, jM] = BTT[:, :, 0], BTT[:, :, 1]
    PT = X.mean(axis=0).reshape(len(chiavi), K)
    ix = {k: i for i, k in enumerate(chiavi)}

    res = {"finali": finali, "temperature": temp, "n_boot": N_BOOT, "seme": SEME, "punti": {}, "delte": {},
           "frazioni": {}, "snr": {}, "sottoinsiemi": {}, "direzione": {}, "taratura_corsair": {}, "per_passo": {}}
    for k in chiavi:
        res["punti"][k] = {m: {"stima": float(PT[ix[k], j]), "ic": ic(BT[:, ix[k], j])} for j, m in enumerate(METRICHE)}
    for nome, x, y in DELTE:
        d = BT[:, ix[x]] - BT[:, ix[y]]
        dp = PT[ix[x]] - PT[ix[y]]
        res["delte"][nome] = {m: {"stima": float(dp[j]), "ic": ic(d[:, j]), "sd": float(d[:, j].std()),
                                  "z": float(dp[j] / d[:, j].std())} for j, m in enumerate(METRICHE)}
    # frazione della distanza B→maestro (appaiata per macchina) e variante incrociata (denominatore maestro − B Aragorn)
    den = BT[:, ix["maestro"]] - BT[:, ix["B_co"]]
    den_x = BT[:, ix["maestro"]] - BT[:, ix["B_ar"]]
    for x in ("p2_ar", "p12_ar"):
        num = BT[:, ix[x]] - BT[:, ix["B_ar"]]
        np_ = PT[ix[x]] - PT[ix["B_ar"]]
        res["frazioni"][x] = {m: {"stima": float(np_[j] / (PT[ix["maestro"], j] - PT[ix["B_co"], j])),
                                  "ic": ic(num[:, j] / den[:, j]),
                                  "incrociata": float(np_[j] / (PT[ix["maestro"], j] - PT[ix["B_ar"], j])),
                                  "ic_incrociata": ic(num[:, j] / den_x[:, j])} for j, m in enumerate(METRICHE)}
    # compiti necessari perche' la differenza p12 − B dia z = 2 a pari effetto (z ∝ √n)
    for j, m in enumerate(METRICHE):
        z = res["delte"]["p12−B (Aragorn)"][m]["z"]
        res["snr"][m] = {"z_p12_B": z, "compiti_per_z2": float(N * (2 / z) ** 2) if z != 0 else None}

    # sottoinsiemi: dove si muove logp_scelte (e il margine) fra B e p12 (Aragorn), e fra B e maestro (Corsair).
    # ⚠️ Gli insiemi scelti sul risultato di B(Ar) subiscono la regressione verso la media: la colonna «controllo»
    # e' lo stesso spostamento per B Corsair − B Aragorn (stesso modello, solo numerica diversa) sugli stessi compiti.
    ins = {"tutti": np.ones(N, bool), "giusta C": oro == 2, "giusta non C": oro != 2,
           "Q2_K sa": q2k, "Q2_K sbaglia": ~q2k,
           "B(Ar) giusta": per["B_ar"]["acc"] == 1, "B(Ar) sbagliata": per["B_ar"]["acc"] == 0,
           "Q2_K sa, B(Ar) sbaglia": q2k & (per["B_ar"]["acc"] == 0),
           "B(Ar) sbaglia entro 1 nat": (per["B_ar"]["margine"] < 0) & (per["B_ar"]["margine"] > -1)}
    # terzili della probabilita' (sulle 4) che il MAESTRO da' alla giusta: insiemi scelti sul maestro, non su B
    # (niente regressione verso la media per p12 − B; la colonna maestro − B invece la subisce, dichiarato)
    pm = per["maestro"]["p_scelte"]
    terz = MD.terzili_per_rango(pm)
    for t, nome in enumerate(("maestro incerto (terzile basso)", "maestro sicuro (terzile medio)",
                              "maestro sicurissimo (terzile alto)")):
        ins[nome] = terz == t
        res.setdefault("terzili_maestro", {})[nome] = {
            "p_maestro_media": float(pm[terz == t].mean()),
            **{f"acc_{k}": float(per[k]["acc"][terz == t].mean())
               for k in ("maestro", "B_co", "B_ar", "p2_ar", "p12_ar")}}
    for nome, msk in ins.items():
        idx = np.nonzero(msk)[0]
        st = [s[np.isin(s, idx)] for s in strati]
        st = [np.searchsorted(idx, s) for s in st if len(s)]
        cols = []
        for m in ("logp_scelte", "margine", "p_scelte"):
            cols += [per["p12_ar"][m][idx] - per["B_ar"][m][idx], per["maestro"][m][idx] - per["B_co"][m][idx],
                     per["B_co"][m][idx] - per["B_ar"][m][idx], per["p2_ar"][m][idx] - per["B_ar"][m][idx]]
        Y = np.stack(cols, axis=1)
        bt = bootstrap_medie(Y, st, SEME + 1)
        pt = Y.mean(axis=0)
        riga = {"n": int(len(idx))}
        for j, m in enumerate(("logp_scelte", "margine", "p_scelte")):
            c = 4 * j
            riga[m] = {"p12−B": float(pt[c]), "ic": ic(bt[:, c]), "maestro−B": float(pt[c + 1]),
                       "frazione": float(pt[c] / pt[c + 1]) if pt[c + 1] else None,
                       "controllo_macchina": float(pt[c + 2]), "ic_controllo": ic(bt[:, c + 2]),
                       "p2−B": float(pt[c + 3]),
                       "p12−B_netto_controllo": float(pt[c] - pt[c + 2]), "ic_netto": ic(bt[:, c] - bt[:, c + 2]),
                       "compiti_migliorati": float((Y[:, c] > 0).mean())}
        res["sottoinsiemi"][nome] = riga

    # direzione per compito: lo spostamento di logp_scelte p12 − B va dove il maestro sta sopra B?
    rng = np.random.default_rng(SEME + 2)
    for x in ("p2_ar", "p12_ar"):
        u = per[x]["logp_scelte"] - per["B_ar"]["logp_scelte"]
        v = per["maestro"]["logp_scelte"] - per["B_co"]["logp_scelte"]
        w = per["B_ar"]["logp_scelte"] - per["B_co"]["logp_scelte"]           # solo macchina, per paragone
        rb, rw = [], []
        for _ in range(2000):
            ii = np.concatenate([s[rng.integers(0, len(s), len(s))] for s in strati])
            rb.append(np.corrcoef(u[ii], v[ii])[0, 1])
            rw.append(np.corrcoef(w[ii], v[ii])[0, 1])
        res["direzione"][x] = {"r_logp_scelte": float(np.corrcoef(u, v)[0, 1]), "ic": ic(np.array(rb)),
                               "r_solo_macchina": float(np.corrcoef(w, v)[0, 1]), "ic_macchina": ic(np.array(rw))}

    # taratura fra modelli (Corsair, registri completi): accuratezza contro misura continua, retta ai minimi quadrati
    cors = ["maestro", "base", "v20", "B150", "B_co"]
    acc_c = np.array([PT[ix[k], METRICHE.index("acc")] for k in cors]) * 100
    for m in METRICHE[1:]:
        xm = np.array([PT[ix[k], METRICHE.index(m)] for k in cors])
        pend, inter = np.polyfit(xm, acc_c, 1)
        pred = pend * xm + inter
        r = np.corrcoef(xm, acc_c)[0, 1]
        livello = {str(t): float((t - inter) / pend) for t in (78.15, 80.05)}
        j = METRICHE.index(m)
        p12 = PT[ix["p12_ar"], j]
        b_ar = PT[ix["B_ar"], j]
        # passi al livello di 80,05 al ritmo p2 → p12 (10 passi); IC dal ritmo agli estremi del bootstrap
        ritmo = (PT[ix["p12_ar"], j] - PT[ix["p2_ar"], j]) / 10
        r_lo, r_hi = ic((BT[:, ix["p12_ar"], j] - BT[:, ix["p2_ar"], j]) / 10)
        manca = livello["80.05"] - p12
        passi = [manca / q if q != 0 and manca / q > 0 else None for q in (ritmo, r_lo, r_hi)]  # None = nullo/contrario
        estremi = [x for x in passi[1:] if x is not None]
        passi = [passi[0], min(estremi) if estremi else None,
                 max(estremi) if len(estremi) == 2 else None]                      # max None = infinito
        res["taratura_corsair"][m] = {"pendenza_punti_per_unita": float(pend), "r": float(r),
                                      "residui": {k: float(acc_c[i] - pred[i]) for i, k in enumerate(cors)},
                                      "livello_per_acc": livello,
                                      "acc_prevista": {k: float(pend * PT[ix[k], METRICHE.index(m)] + inter)
                                                       for k in ("B_ar", "p2_ar", "p12_ar")},
                                      "frazione_verso_78_15": float((p12 - b_ar) / (livello["78.15"] - b_ar)),
                                      "frazione_verso_80_05": float((p12 - b_ar) / (livello["80.05"] - b_ar)),
                                      "passi_a_80_05_ritmo_p2_p12": {"stima": passi[0], "min": passi[1],
                                                                      "max": passi[2]}}

    # ritmo per passo: misure continue (Aragorn) contro held-out KL (registro della corsa)
    kl = kl_heldout()
    res["kl_heldout"] = kl
    for m in METRICHE:
        j = METRICHE.index(m)
        d_passo = (PT[ix["p12_ar"], j] - PT[ix["p2_ar"], j]) / 10
        res["per_passo"][m] = {"delta_per_passo_p2_p12": float(d_passo),
                               "frazione_per_passo": float(d_passo / (PT[ix["maestro"], j] - PT[ix["B_co"], j]))}
    if 0 in kl and 12 in kl:
        res["per_passo"]["kl_heldout"] = {"frazione_p12_verso_0": (kl[0] - kl[12]) / kl[0],
                                         "frazione_p2_verso_0": (kl[0] - kl[2]) / kl[0] if 2 in kl else None,
                                         "frazione_per_passo_p2_p12":
                                             (kl[2] - kl[12]) / kl[0] / 10 if 2 in kl else None}

    # controllo tutto Corsair: p10 PV (parziale) contro B e maestro sugli stessi compiti
    lp10, gi10 = leggi_parziale(PARZIALE[1])
    n10 = len(gi10)
    if not (gi10 == oro[:n10]).all():
        raise SystemExit("⛔ p10: lettera giusta diversa dalla mappa")
    m10, t10 = metriche(lp10, oro[:n10])
    sub = {k: metriche(lps[k][:n10], oro[:n10])[0] for k in ("B_co", "maestro")}
    st10 = [s[s < n10] for s in strati if (s < n10).any()]
    Z = np.stack([v[m] for v in (m10, sub["B_co"], sub["maestro"]) for m in METRICHE], axis=1)
    bz = bootstrap_medie(Z, st10, SEME + 3).reshape(N_BOOT, 3, K)
    bzT = bootstrap_T([lp10, lps["B_co"][:n10], lps["maestro"][:n10]], oro[:n10], st10, SEME + 3, bz[:, :, jl])
    bz[:, :, jT], bz[:, :, jM] = bzT[:, :, 0], bzT[:, :, 1]
    pz = Z.mean(axis=0).reshape(3, K)
    res["p10pv_corsair"] = {"n": n10, "temperatura": t10, **{m: {
        "p10": float(pz[0, j]), "B": float(pz[1, j]), "maestro": float(pz[2, j]),
        "delta": float(pz[0, j] - pz[1, j]), "ic_delta": ic(bz[:, 0, j] - bz[:, 1, j]),
        "frazione": float((pz[0, j] - pz[1, j]) / (pz[2, j] - pz[1, j])),
        "ic_frazione": ic((bz[:, 0, j] - bz[:, 1, j]) / (bz[:, 2, j] - bz[:, 1, j]))} for j, m in enumerate(METRICHE)}}

    testo = json.dumps(res, indent=1, ensure_ascii=False)
    if a.json:
        Path(a.json).write_text(testo, encoding="utf-8")
    print(testo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
