#!/usr/bin/env python3
"""cardan_piega_veloce.py — 22/10 (Angelo: piega di una tappa molto sotto i 45-50 min, IDENTICA AL BYTE al motore attuale).

Same fold as cardan_piega_lotti.py / cardan_piega.py (both imported, NOT modified). Every lever below removes work whose
result is ALREADY computed elsewhere with the same inputs and the same kernels, so the tensors are byte-identical by
construction (the GPU byte check on layers 3-4 against tappa_350 confirms it: prova_piega_veloce_gpu.sh).
Where the time goes and why each lever is safe: @@ROOT:forge@@/ricerche/PIEGA_VELOCE_1022.md.

Switches (env, all default ON except V5; set to 0 to fall back to the lotti behaviour for that lever):
  PIEGA_V_RIUSO=1      V1: jarq without recomputation. f0 (host objective) == e_dw (same W, Q0, Ha expression); the giro-1
                       scales == Ssc (same scale_lotto inputs); the per-giro objectives of `cat` are diagnostics that
                       cardan_piega never reads (dropped); f1 computed once; e_f = where(peggio, f0, f1) row by row
                       (row-local GEMM + row sum: same bits). 8 -> 3 big objective GEMMs, 4 -> 3 scale_lotto per batch.
  PIEGA_V_CONGIUNTI=1  V2: gate and up of the same experts in the same pass: they share H_e (per-expert Hessian, rotated):
                       one Gq read + FWHT rotation + upload + Ha per expert instead of two. Each family keeps its own
                       batch of 16 (same GEMM shapes as before). Falls back to per-family when the expert lists differ.
  PIEGA_V_CHOL=1       V3: the 16 Cholesky factors of the 128x128 diagonal blocks of Ha used by Babai depend only on Ha:
                       computed ONCE per batch (same function CP.chol_ex / CP.chol, same damping rule) and reused by the
                       3 giri and by gate+up (6x fewer CPU fp64 factorizations and GPU<->CPU copies).
  PIEGA_V_SYNC=1       V4: no host sync per Babai group (`int(acc.sum())`, used only for the unused `cambi` counter): the
                       acceptance update runs unconditionally (torch.where with an all-false mask and HE - 0 leave every
                       bit unchanged).
  PIEGA_V_H32=1        V6: fp64 -> working dtype cast of H, S, C D^T done in the CPU prep threads (IEEE round-to-nearest-
                       even on both sides: same bits) -> half the bytes uploaded, no fp64 staging on the device.
  PIEGA_V_CHOL16=cpu   V5 (OFF by default, empirical identity only): `gpu64` factors the 16x16 joint-scale systems of
                       scale_lotto (B*R = 8192 per call) on the GPU in fp64 with our own column Cholesky (plain torch ops,
                       NO rocSOLVER: feedback_rocsolver_gfx1151_inaffidabile) then rounds to fp32 like the CPU path.
                       Not bit-identical by construction (fp64 operation order differs from LAPACK at 1e-16): the GPU
                       byte check decides.
FORGIA_BABAI_TRITON=1 (V7, empirical identity only: Triton sums corr_l in another order): the column loop of Babai runs as
                       ONE Triton kernel per group (cardan_babai_triton_gpu.babai_colonne) inside the V3/V4 Babai, so the
                       ~1M small launches per family disappear together with the syncs and the CPU round trips.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cardan_piega as CP  # noqa: E402
import cardan_piega_lotti as PL  # noqa: E402  (installs the Triton Babai if FORGIA_BABAI_TRITON=1)


def _on(nome, default="1"):
    return os.environ.get(nome, default) == "1"


RIUSO = _on("PIEGA_V_RIUSO")
CONGIUNTI = _on("PIEGA_V_CONGIUNTI")
TRITON = os.environ.get("FORGIA_BABAI_TRITON") == "1"
CHOL_CACHE = _on("PIEGA_V_CHOL")
SENZA_SYNC = _on("PIEGA_V_SYNC")
H32 = _on("PIEGA_V_H32")
CHOL16 = os.environ.get("PIEGA_V_CHOL16", "cpu")


# ───────────────────────────── V5: own batched fp64 Cholesky on the GPU (no rocSOLVER) ─────────────────────────────
def chol_colonne64(M: torch.Tensor):
    """Column (left-looking) Cholesky of a batch of small SPD matrices in fp64 with plain tensor ops.
    Returns (L lower in M.dtype on M.device, info like torch.linalg.cholesky_ex: 0 = ok, k = order of the first
    non-positive (or NaN) pivot, same test as LAPACK dpotf2)."""
    A = M.detach().to(torch.float64)
    n = A.shape[-1]
    L = torch.zeros_like(A)
    info = torch.zeros(A.shape[:-2], dtype=torch.int32, device=A.device)
    for j in range(n):
        Lj = L[..., j, :j]
        d = A[..., j, j] - (Lj * Lj).sum(-1)
        cattivo = ~(d > 0)                                                  # d <= 0 or NaN
        info = torch.where((info == 0) & cattivo, torch.full_like(info, j + 1), info)
        ljj = torch.sqrt(torch.where(cattivo, torch.ones_like(d), d))
        L[..., j, j] = ljj
        if j + 1 < n:
            c = A[..., j + 1:, j] - (L[..., j + 1:, :j] * Lj.unsqueeze(-2)).sum(-1)
            L[..., j + 1:, j] = c / ljj.unsqueeze(-1)
    return L.to(M.dtype), info


def _chol16(M):
    if CHOL16 == "gpu64" and M.device.type != "cpu":
        return chol_colonne64(M)
    return CP.chol_ex(M)


def scale_lotto(W, H, T, ng: int, righe_lotto: int = 128, ridge: float = 1e-4) -> torch.Tensor:
    """CP.scale_lotto with the 16x16 factorization switchable (V5). Default (cpu) = CP.scale_lotto itself."""
    if CHOL16 != "gpu64":
        return CP.scale_lotto(W, H, T, ng, righe_lotto, ridge)
    B, K, R, C = T.shape
    G = C // ng
    dt = W.dtype
    Hg = H.reshape(B, ng, G, C)
    out = torch.zeros(B, K, R, ng, dtype=dt, device=W.device)
    for a in range(0, R, righe_lotto):
        b = min(R, a + righe_lotto)
        r_ = b - a
        Tt = T[:, :, a:b].to(dt).reshape(B, K, r_, ng, G)
        BH = torch.einsum("bkrgj,bgjc->bkrgc", Tt, Hg)
        Tf = T[:, :, a:b].to(dt).reshape(B, K, r_, ng, G)
        M = torch.einsum("bkrgqj,blrqj->brkglq", BH.reshape(B, K, r_, ng, ng, G), Tf)
        rhs = torch.einsum("bkrgc,brc->brkg", BH, W[:, a:b])
        n = K * ng
        M = M.reshape(B, r_, n, n)
        rhs = rhs.reshape(B, r_, n)
        dia = torch.diagonal(M, dim1=-2, dim2=-1)
        vuoti = dia <= 0
        M = M + torch.diag_embed(vuoti.to(dt))
        Lc, info = _chol16(M)
        sol = CP.risolvi_chol(rhs.unsqueeze(-1), Lc).squeeze(-1)
        cattivi = (info != 0) | ~torch.isfinite(sol).all(-1)
        if cattivi.any():
            M2 = M[cattivi] + ridge * torch.diag_embed(dia[cattivi].clamp_min(1e-12))
            sol[cattivi] = CP.risolvi(M2, rhs[cattivi].unsqueeze(-1)).squeeze(-1)
        sol = torch.where(vuoti, torch.zeros_like(sol), sol).reshape(B, r_, K, ng)
        out[:, :, a:b] = sol.permute(0, 2, 1, 3)
    return out


# ───────────────────────────── V3/V4: Babai with cached block factors, no per-group sync ─────────────────────────────
def fattori_blocchi(Ha: torch.Tensor, ng: int) -> list:
    """For each group g: (Hgg as used by Babai, possibly damped; Rch = upper factor). Same code path as
    CP.babai_lotto (CP.chol_ex, damping of the failures only, CP.chol)."""
    B, C, _ = Ha.shape
    G = C // ng
    dt = Ha.dtype
    out = []
    for g in range(ng):
        sl = slice(g * G, (g + 1) * G)
        Hgg = Ha[:, sl, sl]
        Lg, info = CP.chol_ex(Hgg)
        if bool((info != 0).any()):
            bad = info != 0
            eye = torch.eye(G, dtype=dt, device=Ha.device)
            Hgg = Hgg.clone()
            Hgg[bad] = Hgg[bad] + 1e-3 * torch.diagonal(Hgg[bad], dim1=-2, dim2=-1).mean(-1).view(-1, 1, 1) * eye
            Lg = CP.chol(Hgg)
        out.append((Hgg, Lg.transpose(-1, -2).contiguous()))
    return out


def babai_lotto(W, H, T, S, fattori=None):
    """CP.babai_lotto, line by line, with the block factors taken from `fattori` (V3) and no host sync (V4)."""
    if fattori is None:
        return CP.babai_lotto(W, H, T, S)
    B, K, R, C = T.shape
    ng = S.shape[-1]
    G = C // ng
    dt = W.dtype
    tab = CP._tab(K, W.device)
    T = T.clone()
    What = CP.ricostruisci_lotto(T, S)
    HE = torch.bmm(W - What, H)
    cambi = 0
    for g in range(ng):
        sl = slice(g * G, (g + 1) * G)
        Hgg, Rch = fattori[g]
        zcur = What[:, :, sl]
        y = zcur + CP.risolvi_chol(HE[:, :, sl].transpose(-1, -2), Rch.transpose(-1, -2)).transpose(-1, -2)
        A = torch.einsum("bkr,ak->bra", S[:, :, :, g], tab.to(dt))
        if TRITON and W.device.type != "cpu":
            import cardan_babai_triton_gpu as _BT
            z, idx = _BT.babai_colonne(y.contiguous(), Rch, A.contiguous())
            l_iter = ()
        else:
            z = torch.empty_like(y)
            idx = torch.empty(B, R, G, dtype=torch.long, device=W.device)
            l_iter = range(G - 1, -1, -1)
        for l in l_iter:
            if l < G - 1:
                corr = ((z[:, :, l + 1:] - y[:, :, l + 1:]) * Rch[:, l, l + 1:].unsqueeze(1)).sum(-1) \
                    / Rch[:, l, l].unsqueeze(1)
            else:
                corr = torch.zeros(B, R, dtype=dt, device=W.device)
            v = y[:, :, l] - corr
            j = (A - v.unsqueeze(-1)).abs().argmin(-1)
            idx[:, :, l] = j
            z[:, :, l] = A.gather(-1, j.unsqueeze(-1)).squeeze(-1)
        dz_new = z - y
        dz_old = zcur - y
        e_new = (torch.bmm(dz_new, Hgg) * dz_new).sum(-1)
        e_old = (torch.bmm(dz_old, Hgg) * dz_old).sum(-1)
        acc = (e_new < e_old - 1e-12 * e_old.abs()) & torch.isfinite(e_new)
        if not SENZA_SYNC:
            n = int(acc.sum())
            if not n:
                continue
            cambi += n
        tr = tab[idx]
        for k in range(K):
            T[:, k, :, sl] = torch.where(acc.unsqueeze(-1), tr[..., k], T[:, k, :, sl])
        dl = torch.where(acc.unsqueeze(-1), z - zcur, torch.zeros_like(z))
        What[:, :, sl] = torch.where(acc.unsqueeze(-1), z, zcur)
        HE -= torch.bmm(dl, H[:, sl, :])
    return T, cambi


# ───────────────────────────── one batch, one family (V1) ─────────────────────────────
def piega_lotto(T, S, Wd, H, Ha, fattori, giri: int = 3):
    """Everything piega_strato does on the device for one batch of one family. Returns (T2, S2, e_dw, e_sc, e_f).
    RIUSO off -> exactly the lotti sequence (CP.jarq_lotto)."""
    ng = S.shape[-1]
    Q0 = CP.ricostruisci_lotto(T, S)
    W = Q0 + Wd
    if not RIUSO:
        e_dw = CP.obiettivo_lotto(W, Q0, Ha).sum(-1)
        Ssc = CP._fp16(CP.scale_lotto(W, Ha, T, ng))
        e_sc = CP.obiettivo_lotto(W, CP.ricostruisci_lotto(T, Ssc), Ha).sum(-1)
        T2, S2, _ = CP.jarq_lotto(W, H, T, S, giri=giri, nu2=1e-4)
        e_f = CP.obiettivo_lotto(W, CP.ricostruisci_lotto(T2, S2), Ha).sum(-1)
        return T2, S2, e_dw, e_sc, e_f
    B, K, R, C = T.shape
    f0 = CP.obiettivo_lotto(W, Q0, Ha)                                     # == jarq f0 (same tensors, same kernel)
    e_dw = f0.sum(-1)
    Ssc = CP._fp16(scale_lotto(W, Ha, T, ng))                               # == jarq giro-1 scales
    e_sc = CP.obiettivo_lotto(W, CP.ricostruisci_lotto(T, Ssc), Ha).sum(-1)
    Tl, Sl = T.clone(), Ssc
    for it in range(giri):
        if it:
            Sl = CP._fp16(scale_lotto(W, Ha, Tl, ng))
        Tl, _ = babai_lotto(W, Ha, Tl, Sl, fattori)
    f1 = CP.obiettivo_lotto(W, CP.ricostruisci_lotto(Tl, Sl), Ha)
    peggio = f1 > f0
    m = peggio.view(B, 1, R, 1)
    T2 = torch.where(m, T, Tl)
    S2 = torch.where(m, S, Sl)
    e_f = torch.where(peggio, f0, f1).sum(-1)                               # rows back to host keep f0
    return T2, S2, e_dw, e_sc, e_f


# ───────────────────────────── CPU preparation ─────────────────────────────
def _prep_H(hs, CM, e, dt):
    H, fonte = hs.di(e)
    Hr = CM.ruota_H(H)
    return (Hr.to(dt) if H32 else Hr), fonte


def _prep_f(src, TensorKey, L, f, e, Cf, D, dt, src_b=None):
    tt = src.load(TensorKey(L, e, f))
    Wd = Cf[e] @ D.T
    if src_b is not None:
        tb = src_b.load(TensorKey(L, e, f))
        qb = CP.ricostruisci_lotto(tb.planes[None], tb.scales.double()[None])[0]
        qh = CP.ricostruisci_lotto(tt.planes[None], tt.scales.double()[None])[0]
        Wd = Wd + (qb - qh)
    S = tt.scales.double()
    return tt.planes, (S.to(dt) if H32 else S), (Wd.to(dt) if H32 else Wd)


class _Famiglia:
    def __init__(self, src, L, f, cardan_stato, esperti, gp, TensorKey):
        self.f = f
        self.e_ = src.entry(TensorKey(L, 0, f))
        e_ = self.e_
        self.tid = e_.fmt.type_id
        self.raw1 = np.array(e_.tensors[0].data).reshape(e_.n_experts, e_.out, -1).copy()
        self.raw2 = (np.array(e_.tensors[1].data).reshape(int(e_.tensors[1].shape[2]), e_.out, -1).copy()
                     if len(e_.tensors) > 1 else None)
        self.D = cardan_stato[f"D_{f}"].double()
        self.Cf = cardan_stato[f"C_{f}"].double()
        nE = e_.n_experts if esperti is None else min(esperti, e_.n_experts)
        self.per_K = {}
        for e in range(nE):
            if float(self.Cf[e].abs().max()) == 0.0:
                continue
            self.per_K.setdefault(e_.n_planes_of(e), []).append(e)
        self.tot = {"dW": 0.0, "fold": 0.0, "scale_sole": 0.0, "esperti": 0, "fonti": {}}
        self.gp = gp
        self.t0 = time.time()
        self.secondi = 0.0

    def lotti(self, lotto):
        return [(K, lista[i:i + lotto]) for K, lista in self.per_K.items() for i in range(0, len(lista), lotto)]

    def scrivi_lotto(self, K, ee, T2, S2, e_dw, e_sc, e_f):
        gp, e_ = self.gp, self.e_
        tot = self.tot
        tot["dW"] += float(e_dw.sum()); tot["fold"] += float(e_f.sum()); tot["scale_sole"] += float(e_sc.sum())
        tot["esperti"] += len(ee)
        T2c, S2c = T2.cpu(), S2.cpu().to(torch.float16)
        for jj, e in enumerate(ee):
            for k in range(K):
                b = gp.encode_rows(T2c[jj, k].numpy().astype(np.int8), S2c[jj, k].numpy(), self.tid)
                tp, sp = gp.decode_rows(b, self.tid, e_.in_)
                assert np.array_equal(tp, T2c[jj, k].numpy()) and np.array_equal(
                    sp.view(np.uint16), S2c[jj, k].numpy().view(np.uint16)), (self.f, e, k)
                if k == 0:
                    self.raw1[e] = b
                else:
                    self.raw2[e_.slot(1, e)] = b

    def pubblica(self, L, td, reg, motore):
        f = self.f
        for nome_, arr_ in ((f"blk.{L}.ffn_{f}_exps2.weight.npy", self.raw2), (f"blk.{L}.ffn_{f}_exps.weight.npy", self.raw1)):
            if arr_ is None:
                continue
            tmp_ = td / (nome_ + ".parziale.npy")
            np.save(tmp_, arr_); os.replace(tmp_, td / nome_)
        tot = self.tot
        cap = 1 - tot["fold"] / max(tot["dW"], 1e-300)
        cap_s = 1 - tot["scale_sole"] / max(tot["dW"], 1e-300)
        riga = {"strato": L, "famiglia": f, "esperti_piegati": tot["esperti"], "energia_correzione_H": tot["dW"],
                "residuo_dopo_piega_H": tot["fold"], "catturata": cap, "catturata_sole_scale": cap_s,
                "hessiane": tot["fonti"], "secondi": round(self.secondi or (time.time() - self.t0), 1), "motore": motore}
        reg.write(json.dumps(riga) + "\n"); reg.flush()
        CP.log(f"L{L} {f}: piegati {tot['esperti']} esperti, catturata {100 * cap:.1f} % (sole scale {100 * cap_s:.1f} %) "
               f"[{riga['secondi']} s, {motore}]")


def _motore():
    v = [n for n, on in (("V1", RIUSO), ("V2", CONGIUNTI), ("V3", CHOL_CACHE), ("V4", SENZA_SYNC), ("V6", H32),
                         ("V7", TRITON)) if on]
    if CHOL16 == "gpu64":
        v.append("V5")
    return "veloce " + "+".join(v)


def piega_strato(L, cardan_stato: dict, gguf: str, dev, dt, lotto: int, td: Path, reg, pool, esperti=None,
                 aperto=None, bersaglio=None):
    sys.path.insert(0, CP.RAFF)
    import comune as CM
    from rd_ternary.formato import TensorKey
    gp = CM.gp
    src, hs = aperto if aperto is not None else PL.apri(L, gguf)
    src_b = gp.GGUFPlaneSource(bersaglio, layers=[L], bases=["ffn_gate_exps", "ffn_up_exps"]) if bersaglio else None
    fam = [_Famiglia(src, L, f, cardan_stato, esperti, gp, TensorKey) for f in ("gate", "up")]
    congiunti = CONGIUNTI and fam[0].per_K == fam[1].per_K
    gruppi = [fam] if congiunti else [[x] for x in fam]                     # fallback: gate then up, as lotti
    motore = _motore() + ("" if congiunti else " (famiglie separate)")
    for grp in gruppi:
        t_grp = time.time()
        lotti = grp[0].lotti(lotto)
        fut = {}

        def lancia(j):
            if j < len(lotti) and j not in fut:
                ee = lotti[j][1]
                fut[j] = ([pool.submit(_prep_H, hs, CM, e, dt) for e in ee],
                          [[pool.submit(_prep_f, src, TensorKey, L, x.f, e, x.Cf, x.D, dt, src_b) for e in ee] for x in grp])

        for j in range(PL.ANTICIPO + 1):
            lancia(j)
        for j, (K, ee) in enumerate(lotti):
            fh, ff = fut.pop(j)
            prepH = [x.result() for x in fh]
            prepF = [[x.result() for x in lst] for lst in ff]
            lancia(j + PL.ANTICIPO + 1)
            H = torch.stack([h for h, _ in prepH]).to(dev, dt)
            Ha = H + 1e-4 * torch.eye(H.shape[-1], dtype=dt, device=dev)
            fattori = fattori_blocchi(Ha, prepF[0][0][1].shape[-1]) if CHOL_CACHE else None
            for x, pf in zip(grp, prepF):
                for _, fonte in prepH:
                    x.tot["fonti"][fonte] = x.tot["fonti"].get(fonte, 0) + 1
                T = torch.stack([p for p, _, _ in pf]).to(dev)
                S = torch.stack([s for _, s, _ in pf]).to(dev, dt)
                Wd = torch.stack([w for _, _, w in pf]).to(dev, dt)
                x.scrivi_lotto(K, ee, *piega_lotto(T, S, Wd, H, Ha, fattori))
        dtg = time.time() - t_grp
        for x in grp:
            x.secondi = dtg / len(grp)                                      # shared pass: time split evenly
            x.pubblica(L, td, reg, motore)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--strati", default="0-39")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lotto", type=int, default=16)
    ap.add_argument("--esperti", type=int, default=None)
    ap.add_argument("--bersaglio", default=None)
    ap.add_argument("--dtype", default=None, help="(tests) float32 on CPU to mimic the GPU dtype")
    ap.add_argument("--confronta", default=None, help="tensori/ of a reference run: byte check")
    a = ap.parse_args(argv)
    dev = torch.device(a.device)
    dt = getattr(torch, a.dtype) if a.dtype else (torch.float64 if dev.type == "cpu" else torch.float32)
    os.environ.setdefault("LEVE_DT", "float64" if dt == torch.float64 else "float32")
    d = torch.load(a.pt, map_location="cpu", weights_only=False)
    lo, _, hi = a.strati.partition("-")
    strati = [L for L in range(int(lo), int(hi or lo) + 1) if L in d["stato"] or str(L) in d["stato"]]
    out = Path(a.out)
    td = out / "tensori"
    td.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    CP.log(f"motore {_motore()} | algebra {os.environ.get('PIEGA_ALGEBRA', 'cpu')} | device {dev} {dt}")
    with open(out / "registro.jsonl", "a") as reg, ThreadPoolExecutor(PL.PREP_FILI) as pool:
        da_fare = [L for L in strati
                   if not all((td / f"blk.{L}.ffn_{f}_exps.weight.npy").exists() for f in ("gate", "up"))]
        if os.environ.get("PIEGA_ORDINE") == "inverso":
            da_fare = da_fare[::-1]
        prossimo = pool.submit(PL.apri, da_fare[0], a.gguf) if da_fare else None
        for i, L in enumerate(da_fare):
            aperto = prossimo.result()
            prossimo = pool.submit(PL.apri, da_fare[i + 1], a.gguf) if i + 1 < len(da_fare) else None
            if all((td / f"blk.{L}.ffn_{f}_exps.weight.npy").exists() for f in ("gate", "up")):
                continue
            stato = d["stato"].get(L, d["stato"].get(str(L)))
            piega_strato(L, stato, a.gguf, dev, dt, a.lotto, td, reg, pool, esperti=a.esperti, aperto=aperto,
                         bersaglio=a.bersaglio)
    CP.log(f"FINE piega (veloce) {out} in {time.time() - t0:.1f} s")
    if a.confronta:
        sys.exit(0 if PL.confronta(Path(a.confronta), td) else 1)


if __name__ == "__main__":
    main()
