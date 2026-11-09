#!/usr/bin/env python3
"""Tests of e2e_kernel_qat on REAL tensors of the COMBO A GGUF (Thor-T-PV12P-tiedjarq), fp64 reference.

CPU (Triton interpreter, nice, few threads):   TRITON_INTERPRET=1 HIP_VISIBLE_DEVICES= python3 e2e_test_kernel_qat.py
GPU (inside the house lock, full shapes):       python3 e2e_test_kernel_qat.py --gpu --out FILE.json

K1  dd from partial sums == fp64 reference within the deterministic bound, every element, on a gate family
    (TQ1_0R_1024, GS 128) and a down family (TQ1_0, GS 256); uneven expert sizes, an empty expert, sizes not
    multiple of the tile; dY with a wide dynamic range. Also reports today's split route (hi/lo, _k_dd2) error.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

if "--gpu" not in sys.argv:
    os.environ.setdefault("TRITON_INTERPRET", "1")
    torch_threads = 4
else:
    torch_threads = 8

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import e2e_studente as S  # noqa: E402
import e2e_kernel_qat as K  # noqa: E402
import e2e_indietro_hilo as HL  # noqa: E402

GGUF = os.getenv("KQAT_GGUF", "@@ROOT:models@@/gguf/tony-tern/COMBO_A/Thor-T-PV12P-tiedjarq.gguf")


def tensori():
    from gguf import GGUFReader
    r = GGUFReader(GGUF)
    return {t.name: t for t in r.tensors}


def instradamento(conteggi, device):
    idx = torch.cat([torch.full((c,), e, dtype=torch.long) for e, c in enumerate(conteggi)])
    idx = idx[torch.randperm(idx.numel())]
    return S.Piano(idx.view(-1, 1).to(device), len(conteggi))


def dati(S_, C, R, device, seed):
    g = torch.Generator().manual_seed(seed)
    X = (torch.randn(S_, C, generator=g) * 0.7).to(torch.float16)
    # dY: per-token magnitudes spread over 2^-30 .. 2^0 (backward signals are tiny and uneven)
    mag = torch.exp2(-torch.randint(0, 31, (S_, 1), generator=g).float())
    dY = torch.randn(S_, R, generator=g) * mag
    return X.to(device), dY.to(device)


def vecchia_via(dY, X, fam, piano, mot):
    """Today's B3_ind2 route (e2e_indietro_hilo.indietro_famiglia, modo_v 0): dY split hi/lo per (expert, row)."""
    eid = HL.esperto_di_riga(piano, dY.shape[0])
    s_er, inv_er = HL.scala_pot2(HL.massimi_segmento(dY.abs(), eid, fam.E))
    hi, lo = HL.dividi_fp16(dY * s_er[eid])
    dd = mot.dd(lo, X, fam, piano, torch.float32)
    dd += mot.dd(hi, X, fam, piano, torch.float32)
    return dd * inv_er.unsqueeze(-1)


def prova_k1(tens, device, piena: bool) -> dict:
    out = {}
    mot = S.motore("triton2")
    casi = [("blk.5.ffn_gate_exps.weight", "gate"), ("blk.5.ffn_down_exps.weight", "down"),
            ("blk.20.ffn_up_exps.weight", "up"), ("blk.0.attn_qkv.weight", "attn_qkv")]
    for nome, fn in casi:
        t = tens[nome]
        if piena:
            fam = S.famiglia_da_tensore(t, fn, device=device)
            E = fam.E
            conteggi = (np.random.default_rng(1).multinomial(8192 * 8, [1 / E] * E)).tolist() if E > 1 else [8192]
        else:
            fam = S.famiglia_da_tensore(t, fn, device=device, esperti=slice(0, 6) if t.data.ndim == 3 else None,
                                        righe=slice(0, 96))
            conteggi = [37, 0, 64, 129, 5, 70][:fam.E] if fam.E > 1 else [150]
        piano = instradamento(conteggi, device)
        X, dY = dati(sum(conteggi), fam.C, fam.R, device, seed=len(out))
        t0 = time.time()
        nuovo = K.dd_p_triton(dY, X, fam, piano)
        if device != "cpu":
            torch.cuda.synchronize()
        tn = time.time() - t0
        sub = list(range(min(fam.E, 24))) if piena else None     # GPU: fp64 reference on 24 experts (fp64 is slow)
        rif = K.dd_riferimento64(dY, X, fam, piano, sub)
        lim = K.limite_dd(dY, X, fam, piano, sub)
        vec = vecchia_via(dY, X, fam, piano, mot)
        if sub is not None:
            nuovo, vec, rif, lim = nuovo[:len(sub)], vec[:len(sub)], rif[:len(sub)], lim[:len(sub)]
        err = (nuovo.double() - rif).abs()
        err_v = (vec.double() - rif).abs()
        scala = rif.abs().max().item() or 1.0
        out[nome] = {"E": fam.E, "R": fam.R, "C": fam.C, "GS": fam.gs, "token": sum(conteggi),
                     "dentro_limite": bool((err <= lim).all()), "max_err_su_limite": float((err / lim.clamp_min(1e-300)).max()),
                     "nuovo_max_rel": float(err.max() / scala), "vecchio_max_rel": float(err_v.max() / scala),
                     "nuovo_rms_rel": float(err.pow(2).mean().sqrt() / rif.pow(2).mean().sqrt()),
                     "vecchio_rms_rel": float(err_v.pow(2).mean().sqrt() / rif.pow(2).mean().sqrt()),
                     "s_kernel": tn}
        print(nome, json.dumps(out[nome]), flush=True)
        assert out[nome]["dentro_limite"], nome
    return out


def prova_k2(tens, device, piena: bool) -> dict:
    """K2: dX hi/lo in one kernel == today's two launches (dx(lo) + dx(hi)) * inv, BIT for bit."""
    out = {}
    blk = S.Blocchi()
    casi = [("blk.5.ffn_gate_exps.weight", "gate"), ("blk.5.ffn_down_exps.weight", "down"),
            ("blk.0.attn_qkv.weight", "attn_qkv")]
    for nome, fn in casi:
        t = tens[nome]
        if piena:
            fam = S.famiglia_da_tensore(t, fn, device=device)
            E = fam.E
            conteggi = (np.random.default_rng(2).multinomial(8192 * 8, [1 / E] * E)).tolist() if E > 1 else [8192]
        else:
            fam = S.famiglia_da_tensore(t, fn, device=device, esperti=slice(0, 4) if t.data.ndim == 3 else None,
                                        righe=slice(0, 64))
            conteggi = [37, 0, 64, 29][:fam.E] if fam.E > 1 else [70]
        piano = instradamento(conteggi, device)
        _, g32 = dati(sum(conteggi), fam.C, fam.R, device, seed=7)
        d = fam.d0.to(torch.float16)
        s, inv = HL.scala_pot2(g32.abs().amax(1, keepdim=True))
        hi, lo = HL.dividi_fp16(g32 * s)
        vec = S._dx2_triton(lo, fam, d, piano, torch.float32, blk)
        vec += S._dx2_triton(hi, fam, d, piano, torch.float32, blk)
        vec = vec.mul_(inv)
        nuovo = K.dx_hl_triton(hi, lo, inv, fam, d, piano, blk)
        diversi = int((nuovo != vec).sum())
        nuovo_s = K.dx_s_triton(g32, s, inv, fam, d, piano, blk, 3)
        diversi_s = int((nuovo_s != vec).sum())
        x1 = S._dx2_triton(hi, fam, d, piano, torch.float32, blk).mul_(inv)
        diversi_x1 = int((K.dx_s_triton(g32, s, inv, fam, d, piano, blk, 1) != x1).sum())
        out[nome] = {"elementi": vec.numel(), "diversi": diversi, "diversi_registri": diversi_s,
                     "diversi_x1": diversi_x1, "max_abs_diff": float((nuovo - vec).abs().max())}
        assert diversi_s == 0 and diversi_x1 == 0, nome
        print("K2", nome, out[nome], flush=True)
        assert diversi == 0, nome
    return out


def prova_k3(tens, device, piena: bool) -> dict:
    """K3: E2E_KQAT=caldi (2nd expert plane only on the experts that have it) == today, forward and dX, bit for bit
    (zeros included; at most the sign of a zero may differ, counted separately)."""
    out = {}
    blk = S.Blocchi()
    mot = S.motore("triton2")
    L = 5
    fam1 = {n: S.famiglia_da_tensore(tens[f"blk.{L}.{b}.weight"], n, device=device,
                                     esperti=None if piena else slice(0, 12))
            for n, b in S.FAMIGLIE.items()}
    f2 = S.famiglie_piani_esperti(tens, L, fam1, device=device, esperti=None if piena else slice(0, 12))
    for n in ("gate2", "down2"):
        fam = f2[n]
        E = fam.E
        conteggi = (np.random.default_rng(3).multinomial(8192 * 8, [1 / E] * E)).tolist() if piena else \
            [9, 0, 33, 17, 5, 40, 2, 11, 64, 1, 23, 7][:E]
        piano = instradamento(conteggi, device)
        X, g32 = dati(sum(conteggi), fam.C, fam.R, device, seed=11)
        d = fam.d0.to(torch.float16)
        r = {"esperti_caldi": int(fam.caldi.sum()), "esperti": E}
        for tipo in ("avanti", "dx"):
            res = []
            for voci in ("", "caldi"):
                K.imposta(voci)
                if tipo == "avanti":
                    res.append(S._avanti2_triton(X, [fam], [d], piano, torch.float32, blk)[0])
                else:
                    res.append(HL.dx_diviso(mot, g32, d, fam, piano, "fp16x2"))
            K.imposta("")
            a_, b_ = res
            r[tipo] = {"diversi": int((a_ != b_).sum()), "diversi_salvo_segno_zero": int(((a_ != b_) & ~((a_ == 0) & (b_ == 0))).sum())}
        out[n] = r
        print("K3", n, r, flush=True)
        assert r["avanti"]["diversi"] == 0 and r["dx"]["diversi"] == 0, n
    return out


def prova_k4(device, piena: bool) -> dict:
    """K4: _k_mmx (dense products with the split in registers) == today's sequences of _k_mm16 launches + torch
    elementwise (e2e_densi_piegati x2: forward, dX, G, and the x1 / piega1 variants), bit for bit."""
    import e2e_densi_piegati as DP
    g = torch.Generator().manual_seed(5)
    S_, R, C = (8192, 2048, 4096) if piena else (150, 96, 256)
    xg = (torch.randn(S_, C, generator=g) * 0.7).half().to(device)
    W = (torch.randn(R, C, generator=g) * 0.02).to(device)
    hi, lo, inv = DP.peso_x2(W)
    g32 = (torch.randn(S_, R, generator=g) * torch.exp2(-torch.randint(5, 30, (S_, 1), generator=g).float())).to(device)
    mm = DP.mm16_triton
    out = {}
    # forward
    rif = mm(xg, lo.T)
    rif += mm(xg, hi.T)
    rif.mul_(inv[None, :])
    out["avanti"] = int((K.mmx(xg, hi.T, lo.T, 3, ocscale=inv) != rif).sum())
    rif1 = mm(xg, hi.T).mul_(inv[None, :])
    out["avanti_piega1"] = int((K.mmx(xg, hi.T, lo.T, 1, ocscale=inv) != rif1).sum())
    # dX
    h = g32 * inv[None, :]
    s, inv_s = HL.scala_pot2(h.abs().amax(1, keepdim=True))
    hh, hl = HL.dividi_fp16(h * s)
    for nome, termini in (("dx", ((hl, hi), (hh, lo), (hh, hi))), ("dx_x1", ((hh, lo), (hh, hi))),
                          ("dx_piega1", ((hl, hi), (hh, hi))), ("dx_x1_piega1", ((hh, hi),))):
        r = None
        for a_, b_ in termini:
            r = mm(a_, b_) if r is None else r.add_(mm(a_, b_))
        r = r.mul_(inv_s)
        t = (4 if "x1" not in nome else 0) | (2 if "piega1" not in nome else 0) | 1
        s2, _ = HL.scala_pot2((g32.abs() * inv[None, :]).amax(1, keepdim=True))
        out[nome] = int((K.mmx(g32, hi, lo, t, rscale=s2, cscale=inv, oscale=inv_s) != r).sum())
    # G
    s, invg = HL.scala_pot2(g32.abs().amax(0))
    gh, gl = HL.dividi_fp16(g32 * s[None, :])
    rG = mm(gl.T, xg)
    rG += mm(gh.T, xg)
    rG.mul_(invg[:, None])
    out["G"] = int((K.mmx(g32.T, xg, None, 5, rscale=s, oscale=invg) != rG).sum())
    rG1 = mm(gh.T, xg).mul_(invg[:, None])
    out["G_x1"] = int((K.mmx(g32.T, xg, None, 1, rscale=s, oscale=invg) != rG1).sum())
    print("K4", out, flush=True)
    assert all(v == 0 for v in out.values()), out
    return out


def prova_k5(tens, device, piena: bool) -> dict:
    """K5 (fp16 tier, E2E_KQAT=esperti1): forward and dX with the 2nd plane folded in ONE fp16 weight against the
    fp64 product with the SAME folded weight W_eff = fp16(t1 d1 + t2 d2) (kernel == model, ~1e-7) and against the exact
    two-plane fp64 product (the size of the fold itself, ~2^-12); dd of plane 1 against fp64."""
    out = {}
    blk = S.Blocchi()
    mot = S.motore("triton2")
    L = 5
    fam1 = {n: S.famiglia_da_tensore(tens[f"blk.{L}.{b}.weight"], n, device=device,
                                     esperti=None if piena else slice(0, 12)) for n, b in S.FAMIGLIE.items()}
    f2 = S.famiglie_piani_esperti(tens, L, fam1, device=device, esperti=None if piena else slice(0, 12))
    for n in ("gate", "down"):
        f, fb = fam1[n], f2[n + "2"]
        E = f.E
        conteggi = (np.random.default_rng(4).multinomial(8192 * 8, [1 / E] * E)).tolist() if piena else \
            [9, 0, 33, 17, 5, 40, 2, 11, 64, 1, 23, 7][:E]
        piano = instradamento(conteggi, device)
        X, g32 = dati(sum(conteggi), f.C, f.R, device, seed=13)
        d1, d2 = f.d0.to(torch.float16), fb.d0.to(torch.float16)
        y = K.avanti_piani(X, [f], [fb], [d1], [d2], piano, blk)[0]
        dx = K.dx_piani(g32, f, fb, d1, d2, piano, blk)
        off = piano.offs.tolist()
        ry = torch.zeros_like(y, dtype=torch.float64)
        ry2 = torch.zeros_like(ry)
        rdx = torch.zeros(g32.shape[0], f.C, dtype=torch.float64, device=device)
        sub = range(min(E, 24)) if piena else range(E)        # GPU: fp64 reference on 24 experts
        fino = off[len(sub)]
        y, dx = y[:fino], dx[:fino]
        ry, ry2, rdx = ry[:fino], ry2[:fino], rdx[:fino]
        for e in sub:
            a_, b_ = off[e], off[e + 1]
            if a_ == b_:
                continue
            W1 = HL._W_esperto(f, e, d1).double()
            W2 = HL._W_esperto(fb, e, d2).double()
            Weff = (W1 + W2).to(torch.float16).double()
            ry[a_:b_] = X[a_:b_].double() @ Weff.T
            ry2[a_:b_] = X[a_:b_].double() @ (W1 + W2).T
            s_, inv_ = HL.scala_pot2(g32[a_:b_].abs().amax(1, keepdim=True))
            rdx[a_:b_] = ((g32[a_:b_] * s_).half().double() @ Weff) * inv_.double()
        sc = ry.abs().max()
        out[n] = {"avanti_vs_modello": float((y.double() - ry).abs().max() / sc),
                  "avanti_vs_due_piani_fp64": float((y.double() - ry2).abs().max() / sc),
                  "dx_vs_modello": float((dx.double() - rdx).abs().max() / rdx.abs().max())}
        print("K5", n, out[n], flush=True)
        assert out[n]["avanti_vs_modello"] < 1e-5 and out[n]["dx_vs_modello"] < 1e-5, out[n]
    return out


def prova_k6(device, piena: bool) -> dict:
    """K6: KL head of the exact tier (TestaX3K) and of the fp16 tier (TestaX1) against fp64: logits h W^T and p W."""
    g = torch.Generator().manual_seed(6)
    V, H, P = (248320, 2048, 32) if piena else (1000, 256, 40)
    W = (torch.randn(V, H, generator=g) * 0.02 * torch.exp2(-torch.randint(0, 6, (V, 1), generator=g).float()))
    h = torch.randn(P, H, generator=g)
    p = torch.randn(P, V, generator=g) * torch.exp2(-torch.randint(8, 30, (P, 1), generator=g).float())
    W, h, p = W.to(device), h.to(device), p.to(device)
    Vr = min(V, 32768)                                    # fp64 reference on the first Vr vocabulary rows (fp64 is slow)
    p[:, Vr:] = 0
    z64 = h.double() @ W[:Vr].double().T
    d64 = p[:, :Vr].double() @ W[:Vr].double()
    out = {}
    for nome, T_ in (("testa3", K.TestaX3K(W)), ("testa1", K.TestaX1(W))):
        z = T_.logit(h)[:, :Vr]
        d = T_.per(p)
        out[nome] = {"logit_max_rel": float((z.double() - z64).abs().max() / z64.abs().max()),
                     "dh_max_rel": float((d.double() - d64).abs().max() / d64.abs().max())}
    if piena:
        import e2e_bf16x3 as X3
        T_ = X3.TestaX3(W)
        out["oggi_bf16x3"] = {"logit_max_rel": float((T_.logit(h)[:, :Vr].double() - z64).abs().max() / z64.abs().max()),
                              "dh_max_rel": float((T_.per(p).double() - d64).abs().max() / d64.abs().max())}
    print("K6", out, flush=True)
    assert out["testa3"]["logit_max_rel"] < 2e-6 and out["testa3"]["dh_max_rel"] < 2e-6, out
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    torch.set_num_threads(torch_threads)
    torch.manual_seed(0)
    device = "cuda" if a.gpu else "cpu"
    tens = tensori()
    res = {"K1": prova_k1(tens, device, piena=a.gpu), "K2": prova_k2(tens, device, piena=a.gpu),
           "K3": prova_k3(tens, device, piena=a.gpu), "K4": prova_k4(device, piena=a.gpu),
           "K5": prova_k5(tens, device, piena=a.gpu),
           "K6": prova_k6(device, piena=a.gpu)}
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
    print("K1 K2 K3 K4 K5 K6 ok")


if __name__ == "__main__":
    main()
