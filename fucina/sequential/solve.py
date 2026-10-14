#!/usr/bin/env python3
"""S2+S3 — candidate construction and GUARD, for one dense tensor (S2) or for one batch of experts (S3).

Statistics (from the sequential pass, C stream with the upstream tensors ALREADY new); half A builds, half B guards:
  Hq = E[x_q x_q^T] (C stream), Mx = E[x_f x_q^T] (x_f = bf16 teacher stream on the same tokens).
DENSE (S2), candidates for each tensor:
  QEP_a   asymmetric QEP target (2504.09629): W_a = W + a (W Mx H^-1 - W), a in S23_ALPHAS (default {0.5, 1}: «a calibrated
          in {0.5, 1}» of the plan; here chosen PER TENSOR by the guard on half B — declared Thor variant, instead of two
          layers and block-wise KL4); H = H_manifest + w H_knowledge (K6, w = S23_W_KNOWLEDGE, default 1), with
          H_manifest = source M (calib_auto2, the one behind every H of C) and H_knowledge = Kv (wiki+edu) + Kw
          (WikiText-103), each normalized per token; same mixture for Mx. Solver = recipe of C: BIL K=1 (balanced GPTQ) +
          faithful JARQ (K=3, nu 0.6) + DRQ with CONTROL set = H of half B; DRQ (alpha, beta) CALIBRATED PER FAMILY on
          half B from the grid of 2610.11226 App. C (2026-10-13: see «DRQ per family» below).
  K1      (ssm_out, attn_output, attn_v, ffn_down_shexp) TWO-SIDED objective tr(B E A' E^T) with A' = 1/2 (A_KL + H) (AMIX,
          FRONTIER §3.2; YAQA 2505.22988 / FastKron 2608.06291): start from the best QEP_a, 2 rounds of FastKron codes at
          fixed scales <-> two-sided scales (CG) = maestro_kron2 (declared Thor variant); A_KL, B = Kronecker factors of
          the KL of C computed by S1 (S1 host, NOT recomputed on the sequential stream: declared approximation).
  K2      (ffn_down_shexp -> rows of ffn_up_shexp; attn_output -> rows of attn_v, GQA tie) column scale c folded
          upstream, ALTERNATED with the codes (SCALE §1.1: closed-form c, ARB-LLM_RC 2410.03129, H-weighted and
          asymmetric; then codes and scales redone with BIL K=1 + faithful JARQ on the target W_a diag(1/c) with
          H' = D H D), 2 rounds.
  GUARD:  pick the candidate with the lowest objective on half B (one-sided asymmetric; for the K1 families, if the
          factors exist, two-sided with the disjoint CONTROL factors of S1 and A'_B = 1/2 (A_ctl + H_B)) and accept it
          ONLY if it beats the start (plan §2 S2.5: «otherwise the host's stays»).
EXPERTS (S3, E1 of the plan / EXPERTS line §4), in batches on the device:
  H_e(k) = (n_e Gq_e/weight_e + k H_layer)/(n_e + k) (MoEQuant AGQ 2505.03804 + shrinkage k = S23_K, F1 of T2), cross
  term shrunk the same way; asymmetric QEP target a = S23_ALPHA_EXPERTS (default 0.5 = the paper's value);
  balanced GPTQ K=1 + batched JARQ (cardan_fold.jarq_batch, the one of the CARDAN fold, nu2 1e-4); down on the TRUE
  input of C's down, recaptured AFTER the new gate/up, with the cross term a_f (teacher) a_q^T. Guard per expert and
  per tensor on half B («never worse than the host»).

CORRECTED 2026-10-14 (after the S1b audit). Copy of the frozen S23 code (frozen SHA unchanged) with three substantive
differences:
  G1  GUARD INDEPENDENT of the optimized objective. Previously the guard measured the same ASYMMETRIC objective
      E||W x_f - Q x_q||^2 (QEP target a=1) on half B: with frozen or nearly frozen codes, its minimum is W attenuated by
      the teacher/student correlation factor rho (errors-in-variables), a population property -> it also holds on the
      disjoint half and always picks a=1 (11/11 dense «chosen qep1» in the 2026-10-14 pass). Now every choice
      (dense candidate, base for K1/K2, DRQ combination during calibration, alpha and per-expert acceptance) is judged
      on the SYMMETRIC output error towards the teacher on half B, ||(W - Q) x_q||^2 = tr((W-Q) H_B (W-Q)^T)
      (S23_GUARD=sim, default; S23_GUARD=kl_local: for the K1 families with factors, two-sided metric
      tr(B_ctl E A'_B E^T) with E = Q - W, i.e. local KL on the disjoint CONTROL factors), PLUS an output-energy check:
      r = ||Q x_q|| / ||W x_q|| (= sqrt(tr(Q H_B Q^T)/tr(W H_B W^T))) outside [S23_ENERGY_MIN, S23_ENERGY_MAX]
      (default [0.9, 1.1]) -> candidate rejected. The start is not judged on energy (it is the fallback).
  G2  candidates a in S23_ALPHAS (default {0, 0.5, 1}, dense) and S23_ALPHAS_EXPERTS (default {0, 0.5}, experts): faithful
      QEP (2504.09629: alpha is a hyperparameter < 1 and the weights are requantized) with a=0 = symmetric GPTQ; all
      judged by G1. K2: column target Clin = W((1-a) H_A + a Mx_A) with its own a (it was W Mx_A, i.e. always a = 1).
  G3  SHORT H (declared Thor variant, measurement B of the 2026-10-13 short-calibration study §4-5): in the K6 mixture
      the Kw source (WikiText-103, 256-token windows, fresh state) weighs its tokens S23_WEIGHT_KW times inside the
      knowledge part: H_K = (S_Kv + p S_Kw)/(n_Kv + p n_Kw). p = 1 = the previous mixture, bit-exact. Same mixture for
      Mx, for the guard's H_B and for the layer H towards which the experts are shrunk (Gq_e and down stay per token).
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
from fucina.sequential import algebra as AL  # noqa: E402
from fucina.sequential import common_s1b as K  # noqa: E402   (rotate/unrotate/rotate_H, formats: frozen copy of S1b)

F64 = torch.float64
FAM_K1_DEFAULT = ("ssm_out", "attn_output", "attn_v", "ffn_down_shexp")
PAIRS_K2 = {"ffn_down_shexp": "ffn_up_shexp", "attn_output": "attn_v"}
T_TQ1_0R_1024 = 45


def mem(X, rot):
    return K.rotate(X) if rot else X


def memH(H, rot):
    return K.rotate_H(H) if rot else H


def normalize(H):
    """as common.bersaglio_e_H: symmetric, dead diagonal entries -> 1, diagonal mean 1. Returns (Hn, dbar)."""
    H = 0.5 * (H + H.transpose(-1, -2))
    d = torch.diagonal(H, dim1=-2, dim2=-1)
    dead = d <= 0
    if bool(dead.any()):
        H = H.clone()
        if H.dim() == 2:
            H[dead, dead] = 1.0
        else:
            for b in range(H.shape[0]):
                m = dead[b]
                if m.any():
                    H[b, m, m] = 1.0
    dbar = torch.diagonal(H, dim1=-2, dim2=-1).clamp_min(1e-30).mean(-1)
    return H / (dbar.view(*dbar.shape, 1, 1) if H.dim() == 3 else dbar), dbar


def shuffle(acc: Dict[str, tuple], w: float, weight_kw: float = 1.0):
    """acc {source: (H_sum, Mx_sum, n)} -> (H, Mx, N) normalized per token, H = (H_M + w H_K)/(1+w) (K6).
    2026-10-14 G3 (short H): inside H_K the Kw tokens count weight_kw times (weight_kw = 1: bit-exact with before)."""
    nM = acc.get("M", (None, None, 0))[2]
    Hk = Mk = None
    nK = 0
    nKe = 0.0
    for s in ("Kv", "Kw"):
        if s in acc and acc[s][2]:
            c = weight_kw if s == "Kw" else 1.0
            hs = acc[s][0] if c == 1.0 else c * acc[s][0]
            ms = acc[s][1] if c == 1.0 else c * acc[s][1]
            Hk = hs if Hk is None else Hk + hs
            Mk = ms if Mk is None else Mk + ms
            nK += acc[s][2]
            nKe += c * acc[s][2]
    parts = []
    if nM:
        parts.append((1.0, acc["M"][0] / nM, acc["M"][1] / nM))
    if nK:
        den_k = nK if nKe == nK else nKe
        parts.append((w, Hk / den_k, Mk / den_k))
    assert parts, "no source"
    pt = sum(p[0] for p in parts)
    H = sum(p[0] * p[1] for p in parts) / pt
    Mx = sum(p[0] * p[2] for p in parts) / pt
    return H, Mx, nM + nK


def J_asim(Qm, Hm, Cm) -> float:
    """tr(Q H Q^T) - 2 tr(Q C^T) (= E||W x_f - Q x_q||^2 up to a constant), stored basis."""
    return float(((Qm @ Hm) * Qm).sum() - 2.0 * (Qm * Cm).sum())


# ───────────── 2026-10-14 G1: symmetric guard + energy (independent of the optimized objective) ─────────────
def opz_guard(opz: dict):
    """(energy_min, energy_max, mode). Default [0.9, 1.1], mode «sim» (see G1 in the module docstring)."""
    return float(opz.get("energy_min", 0.9)), float(opz.get("energy_max", 1.1)), opz.get("guard", "sim")


def error_sim(Qm, Wm, Hm) -> float:
    """tr((Q - W) H (Q - W)^T) = E||(W - Q) x_q||^2 (stored basis: Wm, Qm, Hm in the same basis)."""
    E = Qm - Wm
    return float(((E @ Hm) * E).sum())


def ratio_energy(Qm, Hm, den_w: float) -> float:
    """||Q x_q|| / ||W x_q|| = sqrt(tr(Q H Q^T) / tr(W H W^T))."""
    num = float(((Qm @ Hm) * Qm).sum())
    return math.sqrt(max(num, 0.0) / max(den_w, 1e-300))


def energy_ok(r: float, emin: float, emax: float, r0: Optional[float] = None) -> bool:
    """r in [emin, emax]; with r0 (START energy, S23_ENERGY_RELATIVE=1) it is also enough not to be farther from 1
    than the start (|ln r| <= |ln r0|): measured 2026-10-14, the expert down at a=0 sits at 0.899 with the start at
    0.884 and a lower symmetric error — a rejection by absolute threshold there would discard a real improvement."""
    if not math.isfinite(r) or r <= 0:
        return False
    if emin <= r <= emax:
        return True
    return r0 is not None and math.isfinite(r0) and r0 > 0 and abs(math.log(r)) <= abs(math.log(r0))


class Kron:
    """S1 factors (stored basis, diagonal mean 1) of one tensor: calibration and control A, B."""

    def __init__(self, cal: Optional[Path], ctl: Optional[Path], name: str, which_B: str, dev, dt):
        self.ok = False
        if cal is None or ctl is None:
            return
        f = {k: d / f"{t}_{name}.npy" for k, d, t in (("Ac", cal, "A"), ("Bc", cal, which_B), ("Ak", ctl, "A"),
                                                       ("Bk", ctl, which_B))}
        if not all(p.exists() for p in f.values()):
            return
        for k, p in f.items():
            M = torch.from_numpy(np.load(p).astype(np.float64)).to(dev, dt)
            M, _ = normalize(M)
            setattr(self, k, M)
        self.ok = True


# ───────────── DRQ: (alpha, beta) calibrated PER FAMILY (2026-10-13, fix from the DRQ/JARQ audit) ─────────────
# 2610.11226 App. C «Main parameters»: alpha in {0.1, 0.3, 0.5, 1}, beta in {0.005, 0.1, 0.25, 0.75, 1} (20 combinations),
# chosen on 64 VALIDATION samples (Pile) per model and base method. Here, declared:
#   (V1) the choice is PER FAMILY (not per model), on the first S23_DRQ_LAYERS_TAR (default 2) layers in which the
#        family appears, then FIXED for the later layers; criterion = asymmetric objective on HALF B (the validation the
#        pass already has, disjoint from search half A), summed over the calibrated layers of the relative gain against
#        JARQ without DRQ, at the best QEP_a target. The paper selects on model perplexity: here per-layer reconstruction
#        (same logic as the guard, much cheaper). Half B also serves as the DRQ control set (as before); the paper keeps
#        control and validation disjoint.
#   (V2) units: «raw» = the paper's grid on the RAW scale of H (C = (1-b) H + b I, faithful); «norm» = the same grid
#        with H (search and control) divided by m = mean of the diagonal of the search H, i.e. beta and alpha relative
#        to the input scale (Thor variant: the audit measured that on attn_output, ssm_out, ffn_down_shexp the inputs
#        are tiny, C ~ b I, S0/L0 ~ 1e4 and DRQ does nothing). The tolerances (L and controls <= +1 %) are ratios:
#        invariant. S23_DRQ_GRID = paper+norm (default, 40) | paper (20, faithful) | off (the previous pair,
#        (1, 0.005) raw, bit-exact).
GRID_ALPHA = (0.1, 0.3, 0.5, 1.0)
GRID_BETA = (0.005, 0.1, 0.25, 0.75, 1.0)
COMBO_FIXED = "raw:a1:b0.005"          # previous (alpha, beta) (QuaRot+GPTQ W2 of the paper), raw units
MODES_GRID = ("paper+norm", "paper", "off")


def combo_key(unit: str, a: float, b: float) -> str:
    return f"{unit}:a{a:g}:b{b:g}"


def combo_values(key: str):
    u, a, b = key.split(":")
    assert u in ("raw", "norm") and a[0] == "a" and b[0] == "b", key
    return u, float(a[1:]), float(b[1:])


def grid_drq(mode: str) -> list:
    """the combinations evaluated during calibration, in tie-break order (on a tie the earlier wins: COMBO_FIXED is first)."""
    if mode not in MODES_GRID:
        raise ValueError(f"S23_DRQ_GRID {mode!r} not in {MODES_GRID}")
    if mode == "off":
        return [COMBO_FIXED]
    unit = ("raw",) if mode == "paper" else ("raw", "norm")
    g = [combo_key(u, a, b) for u in unit for a in GRID_ALPHA[::-1] for b in GRID_BETA]
    g.remove(COMBO_FIXED)
    return [COMBO_FIXED] + g


def drq_combo(Wa_m, HA_m_raw, HB_m_raw, T, S, key: str = COMBO_FIXED, cache: Optional[dict] = None):
    """DRQ with the given combination. «raw»: identical to AL.drq_dev(..., alpha, beta). «norm»: search and control H
    divided by the mean of the diagonal of the search H (same factor: the comparisons stay consistent)."""
    u, a, b = combo_values(key)
    if u == "norm":
        m = float(torch.diagonal(HA_m_raw).mean())
        if not (math.isfinite(m) and m > 0):
            raise ValueError(f"DRQ norm: mean of the diagonal of H {m}")
        HA_m_raw, HB_m_raw = HA_m_raw / m, HB_m_raw / m
    # FAST 2026-10-14 (L1): cache {unit: {beta: L}} for THESE HA_m_raw (one per tensor): the DRQ Cholesky does not depend
    # on a, on W_a or on T -> one per (unit, beta) instead of one per call
    Lc = None if cache is None else cache.setdefault(u, {})
    T, cd = AL.drq_dev(Wa_m, HA_m_raw, [HB_m_raw], T, S, alpha=a, beta=b, L_cache=Lc)
    cd["combo"] = key
    return T, cd


def choose_combo(sums: Dict[str, float], order: list) -> str:
    """argmin of the sums (relative gains, more negative = better); tie-break = grid order."""
    valid = [k for k in order if k in sums and math.isfinite(sums[k])]
    if not valid:
        return COMBO_FIXED
    return min(valid, key=lambda k: (sums[k], order.index(k)))


def calibration_family(registers: list, fam: str, n_tar: int, mode: str) -> dict:
    """registers = [(layer, registry.json)] of the layers ALREADY CLOSED (before the current one). Counts the layers in
    which the family was CALIBRATED with the same mode; fewer than n_tar -> calibrate the current one too (previous
    sums carried forward), otherwise the combination is FIXED = argmin of the sums. Safe resume: the sums are recomputed
    from the closed registers, so a redone layer is not counted twice."""
    order = grid_drq(mode)
    sums: Dict[str, float] = {}
    calibrated = set()
    for L, reg in registers:
        for name, v in reg.get("dense_", {}).items():
            if name.split(".")[2] != fam:
                continue
            g = v.get("drq_grid") or {}
            if not g.get("calibrated") or g.get("mode") != mode:
                continue
            calibrated.add(int(L))
            for k, x in g["rel"].items():
                sums[k] = sums.get(k, 0.0) + float(x)
    if mode == "off":
        return {"calibrated": False, "combo": COMBO_FIXED, "mode": mode, "layers_calibrated": []}
    if len(calibrated) < n_tar:
        return {"calibrated": True, "grid": order, "sums_prec": sums, "mode": mode, "layers_calibrated": sorted(calibrated)}
    return {"calibrated": False, "combo": choose_combo(sums, order), "mode": mode, "layers_calibrated": sorted(calibrated),
            "sums": sums}


def factors_base(HnA_m) -> dict:
    """FAST 2026-10-14 (L1): what build_base factorizes that does NOT depend on the target W_a: U, pd of
    cholesky_damped(HnA_m) and the Cholesky factors of the JARQ Babai blocks (filled on first use). Same H -> same bits."""
    _, U, pd = AL.cholesky_damped(HnA_m)
    return {"U": U, "pd": pd, "chol": {}}


def build_base(Wa_m, HnA_m, cN, ng, fatt: Optional[dict] = None):
    """BIL K=1 -> faithful JARQ (the recipe of C without DRQ). Returns (T, S, cat). fatt = factors_base(HnA_m) (same
    HnA_m and same cN for all the calls that share it)."""
    if fatt is None:
        fatt = {**factors_base(HnA_m), "chol": None}
    U, pd = fatt["U"], fatt["pd"]
    T, S = AL.gptq_bil_batch(Wa_m.unsqueeze(0), U.unsqueeze(0), ng)
    T, S = T[0], S[0]
    T, S, cj = AL.jarq_faithful_dev(Wa_m, HnA_m, T, S, cN, chol_cache=fatt["chol"])
    return T, S, {"damp": pd, "jarq_finale": cj.get("finale_fp16")}


def build_os(Wa_m, HnA_m, HA_m_raw, HB_m_raw, cN, ng, drq: bool, combo: str = COMBO_FIXED,
                  fatt: Optional[dict] = None, cache_drq: Optional[dict] = None):
    """recipe of C on the given target: BIL K=1 -> faithful JARQ -> DRQ (control = half B). Returns (T, S, cat).
    combo = COMBO_FIXED: bit-exact with the previous version."""
    T, S, cat = build_base(Wa_m, HnA_m, cN, ng, fatt)
    if drq:
        T, cd = drq_combo(Wa_m, HA_m_raw, HB_m_raw, T, S, combo, cache_drq)
        cat["drq_trit_changed"] = cd["trit_changed"]
        cat["drq_lanczos_fallbacks"] = cd["fallbacks_eigh"]
        cat["drq_combo"] = combo
    return T, S, cat


def solve_dense(name: str, fam: str, tid: int, W: torch.Tensor, accA: dict, accB: dict, T0: torch.Tensor,
                  S0: torch.Tensor, opz: dict, kron: Optional[Kron] = None, P_gqa: Optional[torch.Tensor] = None,
                  upstream: Optional[tuple] = None):
    """W [R,C] bf16 teacher in GGUF order (dt, device). T0/S0 = start (stored basis). upstream = (T_m, S_m) of the
    already-chosen upstream tensor (K2 families only). Returns a dict with T, S, (S_upstream), registry."""
    t0 = time.time()
    dev, dt = W.device, W.dtype
    rot = tid == T_TQ1_0R_1024
    R, C = W.shape
    ng = S0.shape[-1]
    pkw = float(opz.get("weight_kw", 1.0))
    emin, emax, mode_g = opz_guard(opz)
    HA, MxA, NA = shuffle(accA, opz["w"], pkw)                  # fp64 (accumulators)
    HB, MxB, NB = shuffle(accB, opz["w"], pkw)
    W64 = W.to(HA.device, F64)
    if any(a != 0 for a in opz["alphas"]):
        WsA = (W64 @ MxA.to(F64) @ AL.inverse_damped(HA.to(F64), opz["damp_qep"])).to(dev, dt)   # QEP a = 1, in fp64
    else:
        WsA = W                                                  # only a = 0: W_a = W (symmetric GPTQ)
    HA, MxA, HB, MxB = (x.to(dev, dt) for x in (HA, MxA, HB, MxB))
    HnA, dA = normalize(HA)
    HnA_m = memH(HnA, rot)
    HA_m = memH(0.5 * (HA + HA.T), rot)
    HB_m = memH(0.5 * (HB + HB.T), rot)
    CB_m = mem(W @ MxB, rot)
    CA_m = mem(W @ MxA, rot)
    Wm = mem(W, rot)
    den_w = float(((W @ HB) * W).sum())                          # tr(W H_B W^T) = ||W x_q||^2 (half B)
    den = den_w
    rel_e = bool(opz.get("energy_relative", True))
    reg = {"type_": tid, "shape": [R, C], "token_A": NA, "token_B": NB, "weight_kw": pkw, "energy_relative": rel_e,
           "energy_interval": [emin, emax]}
    use_k1 = fam in opz["k1"] and kron is not None and kron.ok
    kl_local = use_k1 and mode_g == "kl_local"
    if use_k1:
        HnB, _ = normalize(HB)
        ApB = 0.5 * (kron.Ak + memH(HnB, rot))
        ApA = 0.5 * (kron.Ac + HnA_m)
        den_kl = max(abs(AL.J_two_sides(Wm, ApB, kron.Bk)), 1e-300)

    def guard(Qm):
        """2026-10-14 G1: SYMMETRIC output error towards the teacher on half B, relative to ||W x_q||^2 (or two-sided
        local KL on the control factors with E = Q - W, S23_GUARD=kl_local for the K1 families)."""
        if kl_local:
            return AL.J_two_sides(Qm - Wm, ApB, kron.Bk) / den_kl
        return error_sim(Qm, Wm, HB_m) / max(den_w, 1e-300)

    def diagnosis(Qm):
        d = {"sim_B": error_sim(Qm, Wm, HB_m) / max(den_w, 1e-300), "energy": ratio_energy(Qm, HB_m, den_w),
             "JB_asim": J_asim(Qm, HB_m, CB_m) / max(den_w, 1e-300)}
        if use_k1:
            d["kl_local"] = AL.J_two_sides(Qm - Wm, ApB, kron.Bk) / den_kl
        return d

    Q0 = AL.rebuild(T0.to(dev), S0.to(dev, dt))
    cand = {"start": {"T": T0.to(dev), "S": S0.to(dev, dt), "c": None, "JB": guard(Q0), "valid": True,
                         **diagnosis(Q0)}}
    e0 = cand["start"]["energy"] if rel_e else None
    # FAST 2026-10-14 (L1): factorizations independent of a, computed once per tensor (HnA_m, cN, HA_m fixed here)
    fatt = factors_base(HnA_m)
    cache_drq: dict = {}
    # 2026-10-13 DRQ per family: tar = None -> the previous pair (bit-exact); calibrated -> grid over every QEP_a,
    # relative gain per combination = (min_a JB_asim(a, combo) - min_a JB_asim(a, without DRQ)) / ||W||_HB,
    # choice = argmin of the sums (layers calibrated before + this one); not calibrated -> the family's fixed combination
    tar = opz.get("drq_calibration") if opz["drq"] else None
    combo = COMBO_FIXED if tar is None else tar.get("combo", COMBO_FIXED)
    if tar is not None and tar["calibrated"]:
        den0 = max(abs(float(((W @ HB) * W).sum())), 1e-300)
        tg = time.time()
        bases, jb_g = {}, {}
        for a in opz["alphas"]:
            Wa_m = mem(W + a * (WsA - W), rot)
            Tb, Sb, catb = build_base(Wa_m, HnA_m, NA * float(dA), ng, fatt)
            bases[a] = (Wa_m, Tb, Sb, catb)
            jb_g[("without", a)] = error_sim(AL.rebuild(Tb, Sb), Wm, HB_m)       # 2026-10-14 G1: symmetric
            for k in tar["grid"]:
                Tk, _ = drq_combo(Wa_m, HA_m, HB_m, Tb, Sb, k, cache_drq)
                jb_g[(k, a)] = error_sim(AL.rebuild(Tk, Sb), Wm, HB_m)
        j0 = min(jb_g[("without", a)] for a in opz["alphas"])
        rel = {k: (min(jb_g[(k, a)] for a in opz["alphas"]) - j0) / den0 for k in tar["grid"]}
        sums = {k: tar["sums_prec"].get(k, 0.0) + rel[k] for k in tar["grid"]}
        combo = choose_combo(sums, tar["grid"])
        reg["drq_grid"] = {"calibrated": True, "mode": tar["mode"], "metric": "symmetric_B", "rel": rel, "choice": combo,
                              "layers_calibrated_before": tar["layers_calibrated"], "sum_choice": sums[combo],
                              "rel_fixed": rel.get(COMBO_FIXED), "seconds": round(time.time() - tg, 1)}
    elif tar is not None:
        reg["drq_grid"] = {"calibrated": False, "mode": tar["mode"], "choice": combo,
                              "layers_calibrated": tar["layers_calibrated"]}
    os_best = None
    for a in opz["alphas"]:
        tc = time.time()
        Wa = W + a * (WsA - W)
        Wa_m = mem(Wa, rot)
        if tar is not None and tar["calibrated"]:
            _, Tb, Sb, cat = bases[a]
            cat = dict(cat)
            T, cd = drq_combo(Wa_m, HA_m, HB_m, Tb, Sb, combo, cache_drq)       # recomputed (deterministic): no 40 copies of T
            S = Sb
            cat.update({"drq_trit_changed": cd["trit_changed"], "drq_lanczos_fallbacks": cd["fallbacks_eigh"],
                        "drq_combo": combo})
        else:
            T, S, cat = build_os(Wa_m, HnA_m, HA_m, HB_m, NA * float(dA), ng, opz["drq"], combo, fatt, cache_drq)
        Q = AL.rebuild(T, S)
        dg_ = diagnosis(Q)
        cand[f"qep{a:g}"] = {"T": T, "S": S, "c": None, "JB": guard(Q), "a": a,
                             "valid": energy_ok(dg_["energy"], emin, emax, e0),
                             "JA_asim": J_asim(Q, HA_m, CA_m) / max(den_w, 1e-300), **dg_, "cat": cat,
                             "s": round(time.time() - tc, 1)}
    # 2026-10-14 G1: base for K1/K2 = best QEP_a on the symmetric guard among those with energy inside the interval
    # (if none is, the best on the guard: K1/K2 restart from there and are judged anyway)
    qeps = [k for k in cand if k.startswith("qep")]
    valid_q = [k for k in qeps if cand[k]["valid"]] or qeps
    os_best = min(valid_q, key=lambda k: (cand[k]["JB"], qeps.index(k)))
    base = cand[os_best]
    Wa_m = mem(W + base["a"] * (WsA - W), rot)
    k1_delta = float(opz.get("k1_band_delta", 0.0)) if opz.get("k1_band") else None

    def make_k1(fk):
        """K1 = rounds_k1 of (FastKron codes at fixed scales <-> two-sided scales) from the best QEP_a; fk = fastkron_sd
        (exact, the previous one) or fastkron_band (FAST L3)."""
        tc = time.time()
        T, S = base["T"], base["S"]
        LA_k1 = AL.L_unit(ApA.to(Wa_m.dtype), 0.01)          # FAST 2026-10-14 (L1): = the ones fastkron_sd recomputed at every round
        LB_k1 = AL.L_unit(kron.Bc.to(Wa_m.dtype), 0.01) if kron.Bc is not None else None
        for _ in range(opz["rounds_k1"]):
            T = fk(Wa_m, ApA, kron.Bc, S, LA=LA_k1, LB=LB_k1)
            S = AL.fp16(AL.scale_two_sides(Wa_m, T, S, ApA, kron.Bc))
        Q = AL.rebuild(T, S)
        dg_ = diagnosis(Q)
        return {"T": T, "S": S, "c": None, "JB": guard(Q), "a": base["a"], **dg_,
                "valid": energy_ok(dg_["energy"], emin, emax, e0),
                "trit_different_from_qep": float((T != base["T"]).double().mean()), "s": round(time.time() - tc, 1)}
    if use_k1:
        cand["k1"] = make_k1(AL.fastkron_sd if k1_delta is None else AL.fastkron_band)
    if fam in PAIRS_K2 and upstream is not None and opz["k2"]:
        tc = time.time()
        a = base["a"]
        Wa = W + a * (WsA - W)
        ClinA = W @ ((1.0 - a) * HA + a * MxA) if a != 1 else W @ MxA      # 2026-10-14 G2: target with its own a
        T, S = base["T"], base["S"]
        best = None
        history = []
        for _ in range(opz["rounds_k2"]):
            Qo = K.unrotate(AL.rebuild(T, S)) if rot else AL.rebuild(T, S)
            c = AL.step_columns(Qo, HA, ClinA, P_gqa if fam == "attn_output" else None)
            if not bool(torch.isfinite(c).all()) or float(c.abs().min()) <= 0:
                break
            Hc = HA * c.unsqueeze(0) * c.unsqueeze(1)
            Hcn, dc = normalize(Hc)
            Hc_m = memH(0.5 * (Hc + Hc.T), rot)
            Wc_m = mem(Wa / c.unsqueeze(0), rot)
            _, U, _ = AL.cholesky_damped(memH(Hcn, rot))
            T2, S2 = AL.gptq_bil_batch(Wc_m.unsqueeze(0), U.unsqueeze(0), ng)
            T2, S2, _ = AL.jarq_faithful_dev(Wc_m, memH(Hcn, rot), T2[0], S2[0], NA * float(dc))
            jA = J_asim(AL.rebuild(T2, S2), Hc_m, mem(ClinA * c.unsqueeze(0), rot))
            history.append(jA)
            if best is None or jA < best[0]:
                best = (jA, T2, S2, c.clone())
            T, S = T2, S2
        if best is not None:
            _, T, S, c = best
            Qeff_m = mem((K.unrotate(AL.rebuild(T, S)) if rot else AL.rebuild(T, S)) * c.unsqueeze(0), rot)
            Tm, Sm = upstream
            if fam == "attn_output":
                cm = (P_gqa.T @ c) / P_gqa.sum(0)
            else:
                cm = c
            Smn = AL.fp16(Sm.to(dt) * cm.unsqueeze(1))
            ok = bool((cm.abs() >= 1 / opz["limit_c"]).all() and (cm.abs() <= opz["limit_c"]).all()
                      and AL.scale_writable(Smn, Sm.to(dt)) and AL.scale_writable(S, S))
            # the effective column scale after the fold is the one the upstream fp16 scales realize: judge THAT one
            if ok:
                c_eff = Smn / Sm.to(dt).where(Sm.to(dt) != 0, torch.ones_like(Smn))
                c_eff = (c_eff * (Sm.to(dt) != 0)).sum(1) / (Sm.to(dt) != 0).sum(1).clamp_min(1)
                c_eff = P_gqa @ c_eff if fam == "attn_output" else c_eff
                Qeff_m = mem((K.unrotate(AL.rebuild(T, S)) if rot else AL.rebuild(T, S)) * c_eff.unsqueeze(0), rot)
                dg_ = diagnosis(Qeff_m)
                cand["k2"] = {"T": T, "S": S, "c": c, "S_upstream": Smn, "JB": guard(Qeff_m), "a": a, **dg_,
                              "valid": energy_ok(dg_["energy"], emin, emax, e0), "Qeff": Qeff_m,
                              "history_JA": history, "c_min": float(cm.min()), "c_max": float(cm.max()),
                              "s": round(time.time() - tc, 1)}
            else:
                reg["k2_rejected"] = {"c_min": float(cm.min()), "c_max": float(cm.max())}
    # FAST 2026-10-14 (L3): K1 computed in band form (re-associated arithmetic). If it can affect the choice — band K1
    # guard below (best valid rival, start included) + delta — it is RECOMPUTED with the exact fastkron_sd, which takes
    # its place: then candidates, choice and trits are those of the previous code. Otherwise K1 stays a loser with
    # margin > delta, which is > the measured band/exact gap (2026-10-14 speed-up study): the choice does not change and
    # the K1 trits are not written; the registry keeps the numbers of the band K1 (marked).
    if use_k1 and k1_delta is not None:
        rivals = [cand["start"]["JB"]] + [v["JB"] for k, v in cand.items()
                                             if k not in ("start", "k1") and v["valid"] and math.isfinite(v["JB"])]
        rival = min(rivals)
        jb_b = cand["k1"]["JB"]
        exact = not (math.isfinite(jb_b) and jb_b >= rival + k1_delta)
        reg["k1_band"] = {"delta": k1_delta, "JB_band": jb_b, "rival": rival, "margin": jb_b - rival,
                           "recomputed_exact": exact, "s_band": cand["k1"]["s"]}
        if exact:
            cand["k1"] = make_k1(AL.fastkron_sd)
            reg["k1_band"]["JB_exact"] = cand["k1"]["JB"]
        else:
            cand["k1"]["band"] = True
    # 2026-10-14 G1: the minimum of the symmetric guard among the candidates with energy in [emin, emax] wins; it is
    # accepted only if it beats the start (which stays the fallback, without energy check)
    order = list(cand)
    valid = [k for k in order if k != "start" and cand[k]["valid"] and math.isfinite(cand[k]["JB"])]
    chosen = min(valid, key=lambda k: (cand[k]["JB"], order.index(k))) if valid else "start"
    if not cand[chosen]["JB"] < cand["start"]["JB"]:
        chosen = "start"
    reg["rejected_energy"] = [k for k in order if k != "start" and not cand[k]["valid"]]
    reg["candidates"] = {k: {kk: (round(v, 6) if isinstance(v, float) else v) for kk, v in d.items()
                            if kk not in ("T", "S", "c", "S_upstream", "Qeff")} for k, d in cand.items()}
    for k, d in reg["candidates"].items():
        d["dJB_rel"] = cand[k]["JB"] - cand["start"]["JB"]            # the guard is already relative
    reg["metric_guard"] = ("kl_local_ctl" if kl_local else "symmetric_B") + "+energy"
    reg["chosen"] = chosen
    if opz.get("diag_sources", True):
        Qs = cand[chosen]["Qeff"] if chosen == "k2" else AL.rebuild(cand[chosen]["T"], cand[chosen]["S"])
        reg["sim_per_source"] = {}
        for s_, v_ in accB.items():
            if not v_[2] or Qs is None:
                continue
            Hs_ = memH(0.5 * (v_[0] + v_[0].T).to(dev, dt) / v_[2], rot)
            dws = max(float(((Wm @ Hs_) * Wm).sum()), 1e-300)
            reg["sim_per_source"][s_] = {"start": error_sim(Q0, Wm, Hs_) / dws, "chosen": error_sim(Qs, Wm, Hs_) / dws}
    reg["trit_changed"] = float((cand[chosen]["T"] != cand["start"]["T"]).double().mean())
    reg["seconds"] = round(time.time() - t0, 1)
    d = cand[chosen]
    return {"T": d["T"], "S": d["S"], "S_upstream": d.get("S_upstream"), "chosen": chosen, "registry": reg}


# ───────────────────────────── experts (S3) ─────────────────────────────
def rotate_H_batch(H):
    Hr = K.rotate(H)
    return K.rotate(Hr.transpose(-1, -2).contiguous()).transpose(-1, -2).contiguous()


def U_batch(Hn):
    return torch.stack([AL.cholesky_damped(Hn[b])[1] for b in range(Hn.shape[0])])


def inv_batch(H, damp):
    return torch.stack([AL.inverse_damped(H[b], damp) for b in range(H.shape[0])])


def build_experts_batch(Wm, Hm, ng, jarq_batch):
    """balanced GPTQ K=1 + batched JARQ. Wm [b,R,C], Hm [b,C,C] (diag mean 1). Returns T int8 [b,R,C], S [b,R,ng]."""
    U = U_batch(Hm)
    T, S = AL.gptq_bil_batch(Wm, U, ng)
    T2, S2, cat = jarq_batch(Wm, Hm, T.unsqueeze(1), S.unsqueeze(1), rounds=3, nu2=1e-4)
    return T2[:, 0], S2[:, 0], cat


def J_asim_batch(Qm, Hm, Cm):
    """per expert and per row: [b, R]."""
    return (torch.bmm(Qm, Hm) * Qm).sum(-1) - 2.0 * (Qm * Cm).sum(-1)


def J_sim_batch(Qm, Hm, Wm):
    """2026-10-14 G1: symmetric output error per expert and per row: [b, R] = diag((Q - W) H (Q - W)^T)."""
    E = Qm - Wm
    return (torch.bmm(E, Hm) * E).sum(-1)


def N_batch(Qm, Hm):
    """output energy per expert and per row: [b, R] = diag(Q H Q^T)."""
    return (torch.bmm(Qm, Hm) * Qm).sum(-1)
