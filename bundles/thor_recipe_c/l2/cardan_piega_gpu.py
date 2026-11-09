#!/usr/bin/env python3
"""
20/10 (Angelo: "versione che va sulla GPU"): copia di cardan_piega.py in cui la PREPARAZIONE delle Hessiane per esperto
(lettura Gq, simmetrizzazione, normalizzazione, rotazione Hadamard) gira sulla GPU in fp64 e la lettura dal disco del lotto
successivo e' anticipata in un thread. Operazioni elemento per elemento (FWHT: somme/differenze; normalizzazione: divisione per
uno scalare calcolato sulla CPU come nell'originale) -> H identica al bit. Fattorizzazioni: invariate (CPU fp64, rocSOLVER guasto).
cardan_piega.py — FOLD of the distilled CARDAN correction into the EXISTING ternary planes and fp16 scales of the
routed experts (gate/up), so that the GGUF keeps the host's tensor names, types, shapes and bytes: the engine runs it
unchanged, at the host's speed (rule: a lever that lowers the speed is rejected; the separate low-rank branch cost
-12.5 % tg on 8/10).

For each layer L, family f and expert e (stored = rotated basis of TQ1_0R_1024):
    target  W*_e = sum_k s_k T_k (host, ALL planes: main + exps2 of the hot experts) + C_e D^T   (distilled correction)
    fold    JARQ (arXiv 2609.38599 Alg. 1, as raffina/solutori.jarq): joint least-squares of every scale of the row
            (all planes, all groups, rounded to fp16) alternated with Babai nearest-plane proposals on the composite
            alphabet of the planes, accepted row by row only if they lower the error; rows that end worse than the
            host go back to the host (so the fold never moves a row away from the target)
    metric  H_e = per-expert input Hessian captured IN THE QUANTIZED STUDENT FLOW (pod-schur-esperti/Gq_LL, layers
            0-31), rotated: H' = R H R^T; layers without Gq use the layer's FFN-input Hessian H_ffn_LL (shared).
`jarq_lotto` is the batched (over experts) twin of solutori.jarq: same steps, same acceptance rule, same fp16
rounding (test_cardan_kd.py checks it against solutori.jarq in fp64).
Output: <out>/tensori/blk.L.ffn_{gate,up}_exps.weight.npy and ..._exps2.weight.npy (uint8, host shapes) for
raffina/patch_gguf.py, <out>/registro.jsonl with the captured fraction 1 - |W* - fold|_H^2 / |C_e D^T|_H^2.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

RAFF = "@@ROOT:models@@/build/leve/raffina"
GQ_DIR = "@@ROOT:models@@/gguf/tony-tern/pod-schur-esperti"
HFFN_DIR = "@@ROOT:models@@/gguf/tony-tern/hessiane-densi-grande"


# ───────────────────── small factorizations OFF rocSOLVER (9/10, fault at 15:31) ─────────────────────
# The first GPU fold died in layer 0, first batch, with "Memory access fault by GPU node-1 on address (nil)": the
# batched rocSOLVER factorizations on gfx1151 (same library whose Cholesky gave the G2PTQ NaNs, AUDIT/g2ptq/
# DIAGNOSI_NAN.md); sonda_linalg_gpu.py names the primitive. Cure: every FACTORIZATION (Cholesky, LU solve) runs on
# the CPU in fp64 and comes back in the working dtype; the triangular solves with the factor stay on the GPU
# (rocBLAS trsm, not rocSOLVER). Same mathematics as before, more precise (fp64 factor). PIEGA_ALGEBRA=gpu restores
# the old path (only to reproduce the fault).
ALGEBRA_CPU = os.environ.get("PIEGA_ALGEBRA", "cpu") != "gpu"


def _f64cpu(M):
    return M.detach().to("cpu", torch.float64)


def chol_ex(M):
    """torch.linalg.cholesky_ex, factored on CPU fp64. Returns (L lower in M.dtype on M.device, info on M.device)."""
    if not ALGEBRA_CPU or M.device.type == "cpu":
        return torch.linalg.cholesky_ex(M)
    L, info = torch.linalg.cholesky_ex(_f64cpu(M))
    return L.to(M.device, M.dtype), info.to(M.device)


def chol(M):
    """torch.linalg.cholesky (raises on a non-SPD matrix), factored on CPU fp64."""
    if not ALGEBRA_CPU or M.device.type == "cpu":
        return torch.linalg.cholesky(M)
    return torch.linalg.cholesky(_f64cpu(M)).to(M.device, M.dtype)


def risolvi_chol(B, L):
    """x with L L^T x = B (= torch.cholesky_solve(B, L)), by two triangular solves (rocBLAS trsm on the GPU)."""
    if not ALGEBRA_CPU or L.device.type == "cpu":
        return torch.cholesky_solve(B, L)
    y = torch.linalg.solve_triangular(L, B, upper=False)
    return torch.linalg.solve_triangular(L.transpose(-1, -2), y, upper=True)


def risolvi(M, B):
    """torch.linalg.solve, LU on CPU fp64."""
    if not ALGEBRA_CPU or M.device.type == "cpu":
        return torch.linalg.solve(M, B)
    return torch.linalg.solve(_f64cpu(M), _f64cpu(B)).to(M.device, M.dtype)


# 20/10 23:4x: PIEGA_ALGEBRA=gpu_sicura -> fattorizzazioni sulla GPU SENZA rocSOLVER (algebra_gpu.py: Cholesky a blocchi con
# matmul + solve_triangular, ricerche/ROCSOLVER_GFX1151_1020.md), in fp64 come il percorso CPU, risultato riportato al dtype.
if os.environ.get("PIEGA_ALGEBRA") == "gpu_sicura":
    sys.path.insert(0, "@@ROOT:models@@/gguf/tony-tern/OBIETTIVO_C/ROCSOLVER_RIPARA_1020")
    import algebra_gpu as _AG

    def chol_ex(M):
        M64 = M.double(); lead = M64.shape[:-2]; Mb = M64.reshape(-1, *M64.shape[-2:])
        Ls, Is = [], []
        for i in range(0, Mb.shape[0], 8192):      # stessi pezzi di risolvi_chol (lotti enormi: trsm batched fallisce)
            L, info = _AG.chol_ex(Mb[i:i + 8192]); Ls.append(L); Is.append(info)
        return torch.cat(Ls).reshape(*lead, *M64.shape[-2:]).to(M.dtype), torch.cat(Is).reshape(lead)

    def chol(M):
        # 21/10: la Cholesky a blocchi GPU puo' dare «non definita positiva» su matrici al limite che LAPACK fp64 CPU
        # (il percorso originale) fattorizza (tappa 100, L16, lotto 5, pivot 1): solo per quelle si ripiega sulla CPU fp64,
        # cioe' sul calcolo dell'originale. Le altre restano sulla GPU.
        M64 = M.double(); lead = M64.shape[:-2]; Mb = M64.reshape(-1, *M64.shape[-2:])
        L, info = chol_ex(Mb)
        bad = (info != 0).nonzero().flatten()
        if bad.numel():
            print(f"[chol] ripiego CPU fp64 su {bad.numel()} matrici su {Mb.shape[0]}", flush=True)
            L[bad] = torch.linalg.cholesky(Mb[bad].cpu()).to(L.device, L.dtype)
        return L.reshape(*lead, *M64.shape[-2:]).to(M.dtype)

    _TRSM_MAX = int(os.environ.get("PIEGA_TRSM_LOTTO", "8192"))   # 21/10: hipblasDtrsmBatched fallisce con lotti enormi

    def _risolvi_chol_pezzi(B, L):
        lead = torch.broadcast_shapes(B.shape[:-2], L.shape[:-2])
        Bb, Lb = B.expand(*lead, *B.shape[-2:]).reshape(-1, *B.shape[-2:]), L.expand(*lead, *L.shape[-2:]).reshape(-1, *L.shape[-2:])
        out = torch.empty_like(Bb)
        for i in range(0, Bb.shape[0], _TRSM_MAX):
            out[i:i + _TRSM_MAX] = _AG.risolvi_chol(Bb[i:i + _TRSM_MAX], Lb[i:i + _TRSM_MAX])
        return out.reshape(*lead, *B.shape[-2:])

    def risolvi_chol(B, L):
        # un passo di raffinamento iterativo in fp64 (r = B - L L^T x): errore della soluzione <= quello del solo LAPACK
        B64, L64 = B.double(), L.double()
        x = _risolvi_chol_pezzi(B64, L64)
        r = B64 - L64 @ (L64.transpose(-1, -2) @ x)
        return (x + _risolvi_chol_pezzi(r, L64)).to(B.dtype)

    def risolvi(M, B):
        return _AG.risolvi(M.double(), B.double()).to(M.dtype)


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def _fp16(x):
    return x.to(torch.float16).to(x.dtype)


def _tab(K, dev):
    import itertools
    return torch.tensor(list(itertools.product((-1, 0, 1), repeat=K)), dtype=torch.int8, device=dev)


def ricostruisci_lotto(T: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """T int8 [B,K,R,C], S [B,K,R,ng] -> W [B,R,C] in S.dtype."""
    B, K, R, C = T.shape
    ng = S.shape[-1]
    return (T.to(S.dtype).reshape(B, K, R, ng, C // ng) * S.unsqueeze(-1)).sum(1).reshape(B, R, C)


def obiettivo_lotto(W, What, H):
    E = W - What
    return (torch.bmm(E, H) * E).sum(-1)                                    # [B, R]


def scale_lotto(W, H, T, ng: int, righe_lotto: int = 128, ridge: float = 1e-4) -> torch.Tensor:
    """Batched solutori.scale_congiunte (no fixed planes, no tie): S [B,K,R,ng] unrounded."""
    B, K, R, C = T.shape
    G = C // ng
    dt = W.dtype
    Hg = H.reshape(B, ng, G, C)
    out = torch.zeros(B, K, R, ng, dtype=dt, device=W.device)
    for a in range(0, R, righe_lotto):
        b = min(R, a + righe_lotto)
        r_ = b - a
        Tt = T[:, :, a:b].to(dt).reshape(B, K, r_, ng, G)
        # BH[b,k,r,g,:] = T[b,k,r,g,:] @ H[b, g-rows, :]
        BH = torch.einsum("bkrgj,bgjc->bkrgc", Tt, Hg)
        Tf = T[:, :, a:b].to(dt).reshape(B, K, r_, ng, G)
        # M[b,r,k,g,l,q] = sum_{j in q} BH[b,k,r,g,(q,j)] T[b,l,r,q,j]
        M = torch.einsum("bkrgqj,blrqj->brkglq", BH.reshape(B, K, r_, ng, ng, G), Tf)
        rhs = torch.einsum("bkrgc,brc->brkg", BH, W[:, a:b])
        n = K * ng
        M = M.reshape(B, r_, n, n)
        rhs = rhs.reshape(B, r_, n)
        dia = torch.diagonal(M, dim1=-2, dim2=-1)
        vuoti = dia <= 0
        M = M + torch.diag_embed(vuoti.to(dt))
        Lc, info = chol_ex(M)
        sol = risolvi_chol(rhs.unsqueeze(-1), Lc).squeeze(-1)
        cattivi = (info != 0) | ~torch.isfinite(sol).all(-1)
        if cattivi.any():
            M2 = M[cattivi] + ridge * torch.diag_embed(dia[cattivi].clamp_min(1e-12))
            sol[cattivi] = risolvi(M2, rhs[cattivi].unsqueeze(-1)).squeeze(-1)
        sol = torch.where(vuoti, torch.zeros_like(sol), sol).reshape(B, r_, K, ng)
        out[:, :, a:b] = sol.permute(0, 2, 1, 3)
    return out


def babai_lotto(W, H, T, S):
    """Batched solutori.codici_babai (no fixed planes): one JARQ code pass. Returns (T new, n. of changed group-rows)."""
    B, K, R, C = T.shape
    ng = S.shape[-1]
    G = C // ng
    dt = W.dtype
    tab = _tab(K, W.device)
    T = T.clone()
    What = ricostruisci_lotto(T, S)
    HE = torch.bmm(W - What, H)
    cambi = 0
    ar = torch.arange(B, device=W.device)
    for g in range(ng):
        sl = slice(g * G, (g + 1) * G)
        Hgg = H[:, sl, sl]
        Lg, info = chol_ex(Hgg)
        if bool((info != 0).any()):                                          # (fp32 on GPU) damp only the failures
            bad = info != 0
            eye = torch.eye(G, dtype=dt, device=W.device)
            Hgg = Hgg.clone()
            Hgg[bad] = Hgg[bad] + 1e-3 * torch.diagonal(Hgg[bad], dim1=-2, dim2=-1).mean(-1).view(-1, 1, 1) * eye
            Lg = chol(Hgg)
        Rch = Lg.transpose(-1, -2).contiguous()                              # upper, R^T R = Hgg  [B,G,G]
        zcur = What[:, :, sl]
        y = zcur + risolvi_chol(HE[:, :, sl].transpose(-1, -2), Rch.transpose(-1, -2)).transpose(-1, -2)
        A = torch.einsum("bkr,ak->bra", S[:, :, :, g], tab.to(dt))           # [B, R, 3^K]
        z = torch.empty_like(y)
        idx = torch.empty(B, R, G, dtype=torch.long, device=W.device)
        for l in range(G - 1, -1, -1):
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
        acc = (e_new < e_old - 1e-12 * e_old.abs()) & torch.isfinite(e_new)  # [B, R]
        n = int(acc.sum())
        if n:
            cambi += n
            tr = tab[idx]                                                    # [B, R, G, K]
            for k in range(K):
                T[:, k, :, sl] = torch.where(acc.unsqueeze(-1), tr[..., k], T[:, k, :, sl])
            dl = torch.where(acc.unsqueeze(-1), z - zcur, torch.zeros_like(z))
            What[:, :, sl] = torch.where(acc.unsqueeze(-1), z, zcur)
            HE -= torch.bmm(dl, H[:, sl, :])
    return T, cambi


def jarq_lotto(W, H, T, S, giri: int = 3, nu2: float = 1e-4):
    """Batched solutori.jarq with ALL planes free. W [B,R,C], H [B,C,C], T int8 [B,K,R,C], S [B,K,R,ng] (fp16 values).
    Returns (T2, S2 fp16 values in W.dtype, chain)."""
    B, K, R, C = T.shape
    ng = S.shape[-1]
    dt = W.dtype
    Ha = H + nu2 * torch.eye(C, dtype=dt, device=H.device) if nu2 > 0 else H
    S = S.to(dt)
    f0 = obiettivo_lotto(W, ricostruisci_lotto(T, S), Ha)
    Tl, Sl = T.clone(), S.clone()
    cat = {"ospite": float(f0.sum())}
    for it in range(giri):
        Sl = _fp16(scale_lotto(W, Ha, Tl, ng))
        Tl, c = babai_lotto(W, Ha, Tl, Sl)
        cat[f"giro_{it + 1}"] = float(obiettivo_lotto(W, ricostruisci_lotto(Tl, Sl), Ha).sum())
        cat[f"cambi_{it + 1}"] = c
    f1 = obiettivo_lotto(W, ricostruisci_lotto(Tl, Sl), Ha)
    peggio = f1 > f0                                                         # [B, R]
    if peggio.any():
        m = peggio.view(B, 1, R, 1)
        Tl = torch.where(m, T, Tl)
        Sl = torch.where(m, S, Sl)
    cat["righe_tornate_ospite"] = int(peggio.sum())
    return Tl, Sl, cat


# ─────────────────────────────────────────── Hessians ───────────────────────────────────────────
def _normalizza(H: torch.Tensor) -> Optional[torch.Tensor]:
    H = 0.5 * (H + H.T)
    dg = torch.diagonal(H)
    if float(dg.max()) <= 0:
        return None
    morti = dg <= 0
    if morti.any():
        H[morti, morti] = dg[~morti].mean()
    return H / torch.diagonal(H).mean()


def _normalizza_gpu(H: torch.Tensor) -> Optional[torch.Tensor]:
    """= _normalizza, con H sulla GPU: le operazioni elemento per elemento restano sulla GPU, le RIDUZIONI (max, media della
    diagonale) si fanno sulla CPU sullo stesso vettore, come nell'originale -> risultato identico al bit."""
    H = 0.5 * (H + H.T)
    dg = torch.diagonal(H).cpu()
    if float(dg.max()) <= 0:
        return None
    morti = dg <= 0
    if morti.any():
        m = morti.to(H.device)
        H[m, m] = dg[~morti].mean().to(H.device)
        dg = torch.diagonal(H).cpu()
    return H / dg.mean().to(H.device)


def _ruota_H_gpu(CM, H: torch.Tensor) -> torch.Tensor:
    """= comune.ruota_H con il vettore dei segni sul device di H (l'originale lo crea sulla CPU)."""
    s = torch.from_numpy(CM.gp.sign1024().astype(np.float32)).to(H.device, H.dtype)
    def ruota(W):
        C = W.shape[-1]
        x = (W.reshape(*W.shape[:-1], C // 1024, 1024) * s).reshape(W.shape)
        return CM.fwht(x, 1024)
    Hr = ruota(H)
    return ruota(Hr.T.contiguous()).T.contiguous()


class Hessiane:
    def __init__(self, L: int, gq_dir: str = GQ_DIR, hffn_dir: str = HFFN_DIR):
        p = Path(gq_dir) / f"Gq_{L:02d}.npy"
        self.gq = np.load(p, mmap_mode="r") if p.exists() else None
        Hs = torch.from_numpy(np.load(Path(hffn_dir) / f"H_ffn_{L:02d}.npy").astype(np.float64))
        self.hs = _normalizza(Hs)
        self.fonte = "Gq per esperto (flusso quantizzato)" if self.gq is not None else "H_ffn condivisa"

    def grezza(self, e: int):
        """Solo la lettura dal disco (thread di anticipo): array float64 come nell'originale."""
        return np.asarray(self.gq[e], np.float64) if self.gq is not None else None

    def di_gpu(self, e: int, dev, grezza=None) -> tuple:
        if self.gq is not None:
            g = grezza if grezza is not None else self.grezza(e)
            H = _normalizza_gpu(torch.from_numpy(g).to(dev))
            if H is not None:
                return H, "Gq"
        return self.hs.to(dev), "H_ffn"

    def di(self, e: int) -> tuple:
        if self.gq is not None:
            H = _normalizza(torch.from_numpy(np.asarray(self.gq[e], np.float64)))
            if H is not None:
                return H, "Gq"
        return self.hs.clone(), "H_ffn"


# ─────────────────────────────────────────── main ───────────────────────────────────────────
def piega_strato(L, cardan_stato: dict, gguf: str, dev, dt, lotto: int, td: Path, reg, gq_dir=GQ_DIR,
                 hffn_dir=HFFN_DIR, esperti: Optional[int] = None):
    sys.path.insert(0, RAFF)
    import comune as CM
    from rd_ternary.formato import TensorKey
    gp = CM.gp
    src = gp.GGUFPlaneSource(gguf, layers=[L], bases=["ffn_gate_exps", "ffn_up_exps"])
    hs = Hessiane(L, gq_dir, hffn_dir)
    for f in ("gate", "up"):
        t0 = time.time()
        e_ = src.entry(TensorKey(L, 0, f))
        tid = e_.fmt.type_id
        raw1 = np.array(e_.tensors[0].data).reshape(e_.n_experts, e_.out, -1).copy()
        raw2 = (np.array(e_.tensors[1].data).reshape(int(e_.tensors[1].shape[2]), e_.out, -1).copy()
                if len(e_.tensors) > 1 else None)
        D = cardan_stato[f"D_{f}"].double()
        Cf = cardan_stato[f"C_{f}"].double()
        nE = e_.n_experts if esperti is None else min(esperti, e_.n_experts)
        per_K = {}
        for e in range(nE):
            if float(Cf[e].abs().max()) == 0.0:
                continue
            per_K.setdefault(e_.n_planes_of(e), []).append(e)
        tot = {"dW": 0.0, "fold": 0.0, "scale_sole": 0.0, "esperti": 0, "fonti": {}}
        for K, lista in per_K.items():
            from concurrent.futures import ThreadPoolExecutor
            pool = ThreadPoolExecutor(max_workers=2)
            lotti = [lista[i:i + lotto] for i in range(0, len(lista), lotto)]
            prossimo = pool.submit(lambda ee: [hs.grezza(e) for e in ee], lotti[0]) if lotti else None
            for li, ee in enumerate(lotti):
                grezze = prossimo.result()
                prossimo = pool.submit(lambda ee: [hs.grezza(e) for e in ee], lotti[li + 1]) if li + 1 < len(lotti) else None
                Ts, Ss, Hs, Ws, Q0s = [], [], [], [], []
                for e, gz in zip(ee, grezze):
                    tt = src.load(TensorKey(L, e, f))
                    H, fonte = hs.di_gpu(e, dev, gz)
                    tot["fonti"][fonte] = tot["fonti"].get(fonte, 0) + 1
                    Hr = _ruota_H_gpu(CM, H)
                    Ts.append(tt.planes)
                    Ss.append(tt.scales.double())
                    Hs.append(Hr)
                    Ws.append((Cf[e] @ D.T))
                T = torch.stack(Ts).to(dev)
                S = torch.stack(Ss).to(dev, dt)
                H = torch.stack(Hs).to(dev, dt)
                Q0 = ricostruisci_lotto(T, S)
                W = Q0 + torch.stack(Ws).to(dev, dt)
                Ha = H + 1e-4 * torch.eye(H.shape[-1], dtype=dt, device=dev)
                e_dw = obiettivo_lotto(W, Q0, Ha).sum(-1)
                Ssc = _fp16(scale_lotto(W, Ha, T, S.shape[-1]))
                e_sc = obiettivo_lotto(W, ricostruisci_lotto(T, Ssc), Ha).sum(-1)
                T2, S2, cat = jarq_lotto(W, H, T, S, giri=3, nu2=1e-4)
                e_f = obiettivo_lotto(W, ricostruisci_lotto(T2, S2), Ha).sum(-1)
                tot["dW"] += float(e_dw.sum()); tot["fold"] += float(e_f.sum()); tot["scale_sole"] += float(e_sc.sum())
                tot["esperti"] += len(ee)
                T2c, S2c = T2.cpu(), S2.cpu().to(torch.float16)
                for j, e in enumerate(ee):
                    for k in range(K):
                        b = gp.encode_rows(T2c[j, k].numpy().astype(np.int8), S2c[j, k].numpy(), tid)
                        tp, sp = gp.decode_rows(b, tid, e_.in_)                 # bit-exact round trip
                        assert np.array_equal(tp, T2c[j, k].numpy()) and np.array_equal(
                            sp.view(np.uint16), S2c[j, k].numpy().view(np.uint16)), (L, f, e, k)
                        if k == 0:
                            raw1[e] = b
                        else:
                            raw2[e_.slot(1, e)] = b
        np.save(td / f"blk.{L}.ffn_{f}_exps.weight.npy", raw1)
        if raw2 is not None:
            np.save(td / f"blk.{L}.ffn_{f}_exps2.weight.npy", raw2)
        cap = 1 - tot["fold"] / max(tot["dW"], 1e-300)
        cap_s = 1 - tot["scale_sole"] / max(tot["dW"], 1e-300)
        riga = {"strato": L, "famiglia": f, "esperti_piegati": tot["esperti"], "energia_correzione_H": tot["dW"],
                "residuo_dopo_piega_H": tot["fold"], "catturata": cap, "catturata_sole_scale": cap_s,
                "hessiane": tot["fonti"], "secondi": round(time.time() - t0, 1)}
        reg.write(json.dumps(riga) + "\n"); reg.flush()
        log(f"L{L} {f}: piegati {tot['esperti']} esperti, catturata {100 * cap:.1f} % (sole scale {100 * cap_s:.1f} %) "
            f"[{riga['secondi']} s]")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True, help="cardan_migliore.pt (cardan_kd.py)")
    ap.add_argument("--gguf", required=True, help="host GGUF (PV12P)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--strati", default="0-39")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lotto", type=int, default=16)
    ap.add_argument("--esperti", type=int, default=None, help="(probes) only the first N experts")
    a = ap.parse_args(argv)
    dev = torch.device(a.device)
    dt = torch.float64 if dev.type == "cpu" else torch.float32
    os.environ.setdefault("LEVE_DT", "float64" if dt == torch.float64 else "float32")
    d = torch.load(a.pt, map_location="cpu", weights_only=False)
    lo, _, hi = a.strati.partition("-")
    strati = [L for L in range(int(lo), int(hi or lo) + 1) if L in d["stato"] or str(L) in d["stato"]]
    out = Path(a.out)
    td = out / "tensori"
    td.mkdir(parents=True, exist_ok=True)
    with open(out / "registro.jsonl", "a") as reg:
        for L in strati:
            if all((td / f"blk.{L}.ffn_{f}_exps.weight.npy").exists() for f in ("gate", "up")):
                continue
            stato = d["stato"].get(L, d["stato"].get(str(L)))
            piega_strato(L, stato, a.gguf, dev, dt, a.lotto, td, reg, esperti=a.esperti)
    log("FINE piega", out)


if __name__ == "__main__":
    main()
