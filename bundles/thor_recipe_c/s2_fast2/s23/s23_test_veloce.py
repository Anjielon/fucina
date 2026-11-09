#!/usr/bin/env python3
"""VELOCE 14/10 — prove CPU delle leve di velocita' (ricerche/C_S23_VELOCE_1014.md), in aggiunta alle 33 di s23_test.py:
  L1  fattorizzazioni riusate (cholesky_smorzata della base, Cholesky di DRQ per (unita', beta), blocchi di Babai di JARQ,
      L_unit di K1): stesse uscite AL BIT della chiamata senza cache
  L2  jarq_lotto con cache (esperti) == cardan_piega.jarq_lotto AL BIT, anche col ramo smorzato e con la cache condivisa
      fra due bersagli a (come nei candidati degli esperti)
  L3  fastkron_banda == fastkron_sd (stessi trit, fp64 e fp32); guardia di margine: con delta enorme K1 e' sempre
      ricalcolato esatto e risolvi_denso e' identico a quello senza banda
  L4  gptq_bil_lotto coi trit scritti a fine gruppo: coperto dalla prova 1 di s23_test.py (== solutori, al bit)
Stampa OK/KO per riga; codice d'uscita 1 se un KO."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("LEVE_DT", "float64")
import torch  # noqa: E402

QUI = Path(__file__).resolve().parent
for p in (str(QUI), str(QUI.parent / "lib"), str(QUI.parent / "s1b"), str(QUI.parent / "rif")):
    if p not in sys.path:
        sys.path.insert(0, p)
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
import s23_algebra as AL  # noqa: E402
import s23_risolvi as RS  # noqa: E402
import s1b_comune as K  # noqa: E402
import comune  # noqa: E402,F401
import solutori as SO  # noqa: E402
import cardan_piega as CP  # noqa: E402

F64 = torch.float64
g = torch.Generator().manual_seed(20261014)
esiti = []


def prova(nome, cond, info=""):
    esiti.append(bool(cond))
    print(f"{'OK' if cond else 'KO'} {nome} {info}", flush=True)


def spd(n, N=None):
    X = torch.randn(N or 3 * n, n, generator=g, dtype=F64) * torch.linspace(0.2, 3, n, dtype=F64)
    return X.T @ X / X.shape[0]


def norm(H):
    return H / torch.diagonal(H).mean()


# L3 fastkron_banda == fastkron_sd (trit), fp64 e fp32, fette diverse, banda forzata e soglia di default
ok, info = True, []
for (R, C, ng) in ((48, 96, 3), (64, 32, 1), (40, 256, 8), (256, 128, 4), (96, 384, 3)):
    W = torch.randn(R, C, generator=g, dtype=F64) * 0.02
    A, B = norm(spd(C)), norm(spd(R))
    S = AL.fp16(W.abs().reshape(R, ng, -1).mean(-1) * 1.3)
    for dt in (F64, torch.float32):
        a = (W.to(dt), A.to(dt), B.to(dt), S.to(dt))
        T0 = AL.fastkron_sd(*a)
        for fe, rap in ((64, 4.0), (8, 1e9), (3, 1e9), (16, 1.0)):
            T1 = AL.fastkron_banda(*a, fetta=fe, rapporto=rap)
            d = int((T0 != T1).sum())
            ok &= d == 0
            if d:
                info.append(f"{R}x{C} {dt} fetta {fe}: {d} trit diversi")
prova("L3 fastkron_banda == fastkron_sd (stessi trit; fp64 e fp32; fette 3/8/16/64)", ok, "; ".join(info))
# L1 L_unit date == calcolate dentro
LA, LB = AL.L_unit(A), AL.L_unit(B)
prova("L1 fastkron_sd con LA/LB date == senza (al bit)", torch.equal(AL.fastkron_sd(W, A, B, S),
                                                                       AL.fastkron_sd(W, A, B, S, LA=LA, LB=LB)))

# L1 costruisci_base con fattori condivisi fra tre bersagli == senza
R, C, ng = 32, 512, 2
H = norm(spd(C))
W = torch.randn(R, C, generator=g, dtype=F64) * 0.02
cN = 9000.0
fatt = RS.fattori_base(H)
ok = True
for a in (0.0, 0.5, 1.0):
    Wa = W * (1 + 0.3 * a)
    T1, S1, c1 = RS.costruisci_base(Wa, H, cN, ng)
    T2, S2, c2 = RS.costruisci_base(Wa, H, cN, ng, fatt)
    ok &= torch.equal(T1, T2) and torch.equal(S1, S2) and c1 == c2
prova("L1 costruisci_base con cholesky e blocchi di Babai riusati fra 3 bersagli == senza (al bit)", ok,
      f"blocchi in cache {len(fatt['chol'])}")
# L1 drq_combo con cache (unita', beta) == senza, su piu' combinazioni e due bersagli
Hraw, Hc = spd(C) * 3.7, spd(C) * 3.1
ok = True
cache = {}
T0, S0 = T1, S1
for a in (0.0, 1.0):
    for k in (RS.COMBO_FISSA, "norm:a0.1:b1", "norm:a0.3:b0.1", "grezza:a0.5:b0.1", "norm:a1:b0.1"):
        Wa = W * (1 + 0.3 * a)
        Ta, ca = RS.drq_combo(Wa, Hraw, Hc, T0, S0, k)
        Tb, cb = RS.drq_combo(Wa, Hraw, Hc, T0, S0, k, cache)
        ok &= torch.equal(Ta, Tb) and json.dumps(ca, default=str) == json.dumps(cb, default=str)
prova("L1 drq_combo con la Cholesky di DRQ riusata per (unita', beta) == senza (al bit, catena compresa)", ok,
      f"fattori in cache {sum(len(v) for v in cache.values())} per 10 chiamate")

# L2 jarq_lotto con cache == cardan_piega.jarq_lotto, due bersagli con la stessa H (cache condivisa) + ramo smorzato
JQ = AL.jarq_lotto_cache_da(CP)
Bb, R, C, ng = 3, 24, 256, 2
Hs = []
for b in range(Bb):
    Hb = norm(spd(C))
    if b == 1:                                   # blocco 0 indefinito -> chol_ex fallisce -> ramo smorzato di babai_lotto
        Hb[0, :] = 0.0
        Hb[:, 0] = 0.0
        Hb[0, 0] = -2e-4                         # + nu2 1e-4 resta < 0; lo smorzamento 1e-3·media lo rende definito
    Hs.append(Hb)
Hl = torch.stack(Hs)
ok, ramo = True, False
cache = {}
for a in (0.0, 0.5):
    Wl = torch.randn(Bb, R, C, generator=g, dtype=F64) * 0.02 * (1 + a)
    U = torch.stack([AL.cholesky_smorzata(norm(spd(C)))[1] for _ in range(Bb)])
    T, S = AL.gptq_bil_lotto(Wl, U, ng)
    Ta, Sa, ca = CP.jarq_lotto(Wl, Hl, T.unsqueeze(1), S.unsqueeze(1), giri=3, nu2=1e-4)
    Tb, Sb, cb = JQ(Wl, Hl, T.unsqueeze(1), S.unsqueeze(1), giri=3, nu2=1e-4, cache=cache)
    Tc, Sc_, cc = JQ(Wl, Hl, T.unsqueeze(1), S.unsqueeze(1), giri=3, nu2=1e-4)
    ok &= torch.equal(Ta, Tb) and torch.equal(Sa, Sb) and ca == cb and torch.equal(Ta, Tc) and torch.equal(Sa, Sc_)
Ha_ = Hl + 1e-4 * torch.eye(C, dtype=F64)
ramo = bool((CP.chol_ex(Ha_[:, :128, :128])[1] != 0).any())
prova("L2 jarq_lotto con cache dei blocchi di Babai == cardan_piega.jarq_lotto (al bit, cache condivisa fra 2 a)",
      ok and ramo, f"ramo smorzato attraversato {ramo}")

# L3 guardia di margine: delta enorme -> K1 sempre ricalcolato esatto -> identico a K1 senza banda
Cd, Rd = 256, 64
Wd = torch.randn(Rd, Cd, generator=g, dtype=F64) * 0.02


def acc_finto():
    out = {}
    for s_n, n in (("M", 900), ("Kv", 600), ("Kw", 500)):
        X = torch.randn(n, Cd, generator=g, dtype=F64) * torch.linspace(0.3, 2, Cd, dtype=F64)
        Xf_ = X + 0.05 * torch.randn(n, Cd, generator=g, dtype=F64)
        out[s_n] = (X.T @ X, Xf_.T @ X, n)
    return out


class KronFinto:
    ok = True


accA, accB = acc_finto(), acc_finto()
kr = KronFinto()
kr.Ac, kr.Bc, kr.Ak, kr.Bk = norm(spd(Cd)), norm(spd(Rd)), norm(spd(Cd)), norm(spd(Rd))
Hm_, _, _ = RS.mescola(accA, 1.0)
T0p, S0p, _ = SO.gptq_bilanciato(Wd, norm(Hm_), 1, 2)
base = {"alfe": [0.0, 0.5, 1.0], "w": 1.0, "damp_qep": 0.01, "drq": True, "k1": ("ffn_down_shexp",), "giri_k1": 2,
        "k2": False, "giri_k2": 2, "limite_c": 8.0, "peso_kw": 3.5}
r0 = RS.risolvi_denso("blk.0.ffn_down_shexp.weight", "ffn_down_shexp", 34, Wd, accA, accB, T0p[0], SO.fp16(S0p[0]),
                      {**base, "k1_banda": False}, kr)
r1 = RS.risolvi_denso("blk.0.ffn_down_shexp.weight", "ffn_down_shexp", 34, Wd, accA, accB, T0p[0], SO.fp16(S0p[0]),
                      {**base, "k1_banda": True, "k1_banda_delta": 1e9}, kr)
r2 = RS.risolvi_denso("blk.0.ffn_down_shexp.weight", "ffn_down_shexp", 34, Wd, accA, accB, T0p[0], SO.fp16(S0p[0]),
                      {**base, "k1_banda": True, "k1_banda_delta": -1e9}, kr)


def senza_s(reg):
    d = json.loads(json.dumps(reg, default=str))
    d.pop("secondi", None)
    d.pop("k1_banda", None)
    for v in d["candidati"].values():
        v.pop("s", None)
        v.pop("banda", None)
    return json.dumps(d, sort_keys=True)


kb = r1["registro"]["k1_banda"]
prova("L3 guardia di margine, delta enorme: K1 ricalcolato esatto e risolvi_denso identico al codice senza banda",
      kb["ricalcolato_esatto"] and torch.equal(r0["T"], r1["T"]) and torch.equal(r0["S"], r1["S"])
      and r0["scelto"] == r1["scelto"] and senza_s(r0["registro"]) == senza_s(r1["registro"]),
      f"scelto {r0['scelto']}; k1 banda {kb['JB_banda']:.6g} esatto {kb.get('JB_esatto', float('nan')):.6g}")
kb2 = r2["registro"]["k1_banda"]
prova("L3 guardia di margine, delta -inf: K1 a banda tenuto, stessa scelta e stessi trit (fp64: banda = esatto)",
      not kb2["ricalcolato_esatto"] and r2["scelto"] == r0["scelto"] and torch.equal(r0["T"], r2["T"])
      and abs(r2["registro"]["candidati"]["k1"]["JB"] - r0["registro"]["candidati"]["k1"]["JB"]) < 1e-9,
      f"k1 {r2['registro']['candidati']['k1']['JB']} vs {r0['registro']['candidati']['k1']['JB']}")

print(f"{sum(esiti)}/{len(esiti)} verdi")
sys.exit(0 if all(esiti) else 1)
