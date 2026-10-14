#!/usr/bin/env python3
"""V step (PV-Tuning) on the ternary CODES of Thor-T, with no per-code state — prototype of the
PV-codes project design (2026-09-26).

New module. No existing module is modified: it imports READ-ONLY `student` (Plane, FamilyData, engines,
`_trit_from_byte`, `trit_expert`), `scale_codes` (LAYOUT) and `scale_fields` (torch decode from the bytes).
It supersedes the per-code fp32 latent (488 GiB) of the phase-2 codes design.

Idea (design §1). In the student's `_k_dd` kernel the tile `acc` (BR x GS, fp32, registers) holds
sum_s dY[s, r] X_rot[s, c] = dL/dW_rot[r, c] for ONE row of ONE group; today it is reduced into dL/dd and discarded.
Here the same tile, in the same kernel (`_k_dd_v`, constant MODE_V), also yields:
  MODE_V = 0   today's kernel, nothing else (bit-exact test against `_k_dd`);
  MODE_V = 1   PROPOSAL (micro-batches 1-2): per tile row, among the codes with a feasible ONE-level jump
               (Delta = -sign(acc * d0), t + Delta in {-1,0,1}, d0 != 0) the one with the largest |acc| -> `cand` (uint8,
               column within the group) and `prop` (fp32, signed acc); between the two micro-batches the larger |.| wins;
  MODE_V = 2   VERIFY (micro-batches 3-16, data DISJOINT from the proposal): v = acc[row, cand] -> S += v, Q += v^2.
  MODE_V = 3   CURVATURE (2026-09-27, fix for the "first order only" defect): the backward carries a RANDOM vector
               with covariance = Hessian of the loss w.r.t. the outputs (sampled Fisher = Gauss-Newton: for the KL
               on the logits g = sqrt(c) (p - e_y), y ~ p of the student); acc2 = sum_s dY^2 X^2 -> H += acc2[row, cand]
               = UNBIASED estimate of the Gauss-Newton diagonal on the candidate code. dL/dd = 0 in this mode
               (theta.grad is not touched: it is a measurement backward, not a training one).
No atomics (one program per (expert, row tile, group)), 17 bytes per GROUP (cand 1 + prop 4 +
S 4 + Q 4 + H 4), nothing per code. The dL/dd output is the same in modes 0-2 (bit-exact test in T1).

At the end of the step (after `opt.step`, the P step on the scales): for each group with a candidate, Student's t over
the 14 verification micro-batches (guard against noise), Benjamini-Hochberg q = 0.1 PER MATRIX; then the
SECOND-ORDER criterion (2026-09-27 revision: first order alone made the loss RISE with a teacher off the grid):
predicted gain of a jump of amplitude d = g_step |d| - H_step d^2 / 2, with g_step = (N/n_ver) S (projected
onto the proposal direction) and H_step = (N/n_curv) H; jump only if > 0 (this is rounding the local quadratic
minimum to the nearest level). Among the convenient ones, per expert, in gain order, as long as the
EFFECTIVE norm of the chosen jumps keeps ||Delta W_e|| <= rho ||W_e|| (sum d_g^2 of the chosen <= rho^2 sum_g d_g^2 nnz_g:
trust ratio of PV §4.2, guaranteed, no longer on average). ONE-level jumps; the scales (theta, Adam) are untouched.

Jumps are written to the BYTES (in place on `fam.blocks`, the same tensor the kernels read): q of the byte,
q += sum Delta 3^(4-n) (several jumps in the same byte add up: different digits, no carry), byte =
ceil(256 q / 243). Every jump goes into the `Registry` (9 bytes: L, f, e, r, c, t_old, t_new): cancellation
(trust region), replay (resume = GGUF + registries), export (`patch_codes` on a COPY of the GGUF).

Contents:
  1. BufferV / VState                  buffers per (layer, family), mode of the current micro-batch
  2. _k_dd_v / dd_v_triton             kernel (GPU; on CPU in the Triton interpreter)
  3. dd_v_torch                        reference WITH materialised dW per expert (CPU, tests)
  4. TorchEngineV / TritonEngineV      subclasses of the student engines: only `dd` changes; `install`
  5. sf_t / p_values / bh / tau / select
  6. position_t / skip_byte / apply_skips / cancel / replay / Registry
  7. patch_codes / verify_patch_codes (export on a COPY of the GGUF)
  8. TrustRegion                    probe: worse KL -> cancel and rho/2; calibration of rho over the first steps;
                                       2026-09-27 `adaptive`: true trust region (r = actual / predicted decrease)
  9. VStep                            the orchestrator (buffers + engines + selection + jumps + registry)
  1b. ConfigBop / BopState / select_bop   2026-09-27 "per-group V-Bop" (--pv-mode vbop, TECNICHE_E2E_PV.md §21):
                                       persistent state per group (candidate with hysteresis, m and v in EMA, H in EMA,
                                       age), jump if gain N (m|d| - H d^2/2) > 0 and z = m / sqrt(v / n_eff) passes BH,
                                       rho budget and probe unchanged, ECHO correction after the jump. Default (bop
                                       None) = the previous PV, bit-exact.
  1c. BufferV.HT / _k_ht / ht_triton / ht_torch   2026-09-27 newton-theta branch (--ottimizzatore-theta sophia):
                                       Gauss-Newton curvature of the SCALES per group, HT += sum_s dY^2 (t . X_g)^2, in
                                       the step's curvature micro-batches (VState(curv_theta=True); off = no buffer,
                                       bit-exact).
  5b. ConfigV.accept / z_min / z_gain_min   2026-09-27 --pv-accept gain (TECNICHE_E2E_PV.md §38): candidate
                                       accepted if the estimated second-order gain is > 0 (and z > z_min on the
                                       direction, t of the gain > z_gain_min), WITHOUT the two BH tests;
                                       ordering, rho budget, probe and adaptive rho unchanged. Default "bh" = bit-exact.
  1d. 2026-09-28 ternary dense families (student.FAMILIES_DENSE): the single-expert (E = 1) dense families live in
                                       `ExpertLayer.fam`, so `install` sees them like the experts (buffer, V engine,
                                       selection with rho budget over the whole matrix, jumps, cancellation); in the
                                       registry f = 3 + index in FAMILIES_DENSE, e = 0; `patch_codes` /
                                       `verify_patch_codes` read the 2D tensors as (1, R, NBR); `reload_codes`
                                       rebuilds the students with the dense families from the registries. Without
                                       dense families: everything bit-exact.
  8b. DoseConfig / counts_doses / VStep.select_e_trial_doses / VState.prepare_recenter /
      TrustRegion.outcome_doses   2026-09-27 v-interaction-b branch (TECNICHE_E2E_PV.md §32.5 actions 1-2): the
                                       chosen jumps are tried in DOSES (nested prefixes in gain order) on probe A, the
                                       best dose is confirmed on probe B; then (--pv-recenter) leave-one-out from the
                                       reached state with ONE backward of probe A (mode 2 on the jump columns) and
                                       cancellation of the harmful ones half at a time until B confirms. Partial-dose
                                       retention does NOT halve rho. New methods: `select_e_skip` and `outcome`
                                       unchanged (off = bit-exact; `apply_skips` = `rows_skips` + `skip_registry`).
  8c. PRIMITIVE_PM / midpoint_price / choose_primitive / VStep._midpoint_prices / DoseConfig.point_mean
      2026-09-29 midpoint branch (arXiv 2609.09867 Prop. 2 and §6(a); math note on the PV midpoint): in the dose probe
      the chosen jumps are grouped into primitives, the pm_max with the highest predicted gain are priced at their OWN
      MIDPOINT with the trapezoidal rule over the integer states (one probe backward per primitive + one at q0), and
      the doses are prefixes of the primitives with price < 0 (with objective A and constraint B: `choose_primitive`);
      confirmation on probe B, recenter and (optional) confirmation of the objective on disjoint data. Off (default):
      bit-exact.

⛔ CPU and memmap: `family_from_tensor` on a whole tensor (or with only an expert slice) returns bytes that are
a VIEW of the read-only GGUF memmap; writing to them would crash the process. On CPU jumps are applied only to
families passed through `family_writable` (a copy); on the GPU the bytes are already a copy.

Tests: `e2e_test_pv_codici.py` (mandatory T1-T4 + T5 gfx1151 compilation + T6 real student), CPU only.
"""
from __future__ import annotations

import hashlib
import io
import math
import os
import shutil
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
import torch

HERE = Path(__file__).resolve().parent

from fucina.student import student as S  # noqa: E402  (read-only)
from fucina.ternary import scale_codes as CS  # noqa: E402  (read-only)
from fucina.ternary import scale_fields as FIELDS  # noqa: E402  (read-only)

if S._TRITON:
    import triton
    import triton.language as tl
    _trit_from_byte = S._trit_from_byte          # the SAME jit function as the student kernel

FAMILIES_IDX = {"gate": 0, "up": 1, "down": 2}
# 2026-09-28 ternary dense families: the dense families (single expert, e = 0) in the registry, with indices AFTER the
# expert ones (the original three unchanged: registries already written read back identically). 3 + position in
# student.FAMILIES_DENSE.
FAMILIES_IDX.update({n: 3 + i for i, n in enumerate(S.FAMILIES_DENSE)})
# 2026-09-29 second plane of the dense families (<name>2): their OWN families in the registry, indices after the first
# planes (the original ones unchanged): 3 + len(FAMILIES_DENSE) + position in student.FAMILIES_DENSE2.
FAMILIES_IDX.update({n: 3 + len(S.FAMILIES_DENSE) + i for i, n in enumerate(S.FAMILIES_DENSE2)})
# 2026-09-30 third plane (<name>3): their OWN families, indices after the second planes (first and second unchanged:
# registries already written read back identically): 3 + 2 len(FAMILIES_DENSE) + position in student.FAMILIES_DENSE3
# (max 32 < 256, u1).
FAMILIES_IDX.update({n: 3 + 2 * len(S.FAMILIES_DENSE) + i for i, n in enumerate(S.FAMILIES_DENSE3)})
NAMES_IDX = {v: k for k, v in FAMILIES_IDX.items()}
P3 = (1, 3, 9, 27, 81)
BYTE_PER_GROUP = 17                         # cand uint8 + prop fp32 + S fp32 + Q fp32 + H fp32


# ═══════════════════════════════════ 1. buffer ═══════════════════════════════════
@dataclass
class BufferV:
    """V-step state of ONE family of ONE layer, per group (E, R, NG): nothing per code.
    cand: column of the candidate within the group (uint8: GS <= 256); prop: signed proposal acc (0 = no
    candidate); S, Q: sum and sum of squares of acc[row, cand] over the verification micro-batches; H: sum of
    acc2[row, cand] (sampled Gauss-Newton curvature) over the curvature micro-batches."""
    cand: torch.Tensor
    prop: torch.Tensor
    S: torch.Tensor
    Q: torch.Tensor
    H: torch.Tensor
    # 2026-09-27 newton-theta branch (--ottimizzatore-theta sophia): Gauss-Newton curvature of the SCALES per group,
    # without d^2: HT += sum_s dY_{s,r}^2 u_{s,r,g}^2, u = sum_{c in g} X_{s,c} t_{r,c}, over the curvature micro-batches
    # (mode 3) of the STEP (reset at every step, even inside an accumulation window). None = off (default: no change).
    HT: Optional[torch.Tensor] = None

    @staticmethod
    def create(fam: S.FamilyData, device=None, curv_theta: bool = False) -> "BufferV":
        dev = device if device is not None else fam.blocks.device
        assert fam.gs <= 256, "cand is uint8"
        shape = (fam.E, fam.R, fam.NG)
        return BufferV(torch.zeros(shape, dtype=torch.uint8, device=dev),
                       torch.zeros(shape, dtype=torch.float32, device=dev),
                       torch.zeros(shape, dtype=torch.float32, device=dev),
                       torch.zeros(shape, dtype=torch.float32, device=dev),
                       torch.zeros(shape, dtype=torch.float32, device=dev),
                       torch.zeros(shape, dtype=torch.float32, device=dev) if curv_theta else None)

    def reset(self) -> None:
        for t in (self.cand, self.prop, self.S, self.Q, self.H):
            t.zero_()

    def byte(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.cand, self.prop, self.S, self.Q, self.H, self.HT)
                   if t is not None)


def key(fam: S.FamilyData) -> tuple:
    return (int(fam.layer), fam.name)


@contextmanager
def timer(times: Optional[dict], name: str):
    """2026-10-02 tt5-phase-v branch (--pv-times-v): adds the block's seconds to `times[name]`, with a device
    synchronisation before and after (clock only: no number changes). `times` None: nothing (no synchronisation)."""
    if times is None:
        yield
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times[name] = times.get(name, 0.0) + time.perf_counter() - t0
        times["n_" + name] = times.get("n_" + name, 0) + 1


COUNTS_DD = {"skipped": 0}                    # 2026-10-02 --pv-dd-only-skips: dL/dd not computed (diagnostics, no effect)


def dd_redundant(mot, fam) -> bool:
    """2026-10-02 tt5-phase-v branch (--pv-dd-only-skips, VState.skip_dd_redundant): True if the dL/dd of `fam` in THIS
    backward is needed by nobody: we are in a V-phase gradient (`prepare_recenter` has set mode 2 and
    `families_recenter`) and the family has no prepared jumps (prop = 0 everywhere: S and Q would not change; the dL/dd
    would only go into the theta gradient, discarded by the caller). Outside those gradients, or with the option off:
    False (as always)."""
    st = getattr(mot, "state", None)
    if st is None or not getattr(st, "skip_dd_redundant", False) or int(getattr(st, "mode", 0) or 0) != 2:
        return False
    fr = getattr(st, "families_recenter", None)
    if fr is None or key(fam) in fr:
        return False
    COUNTS_DD["skipped"] += 1
    return True


# ═══════════════════════════════════ 1b. per-group V-Bop (2026-09-27, TECNICHE_E2E_PV.md §21) ═══════════════════════════════════
# PERSISTENT state across steps per group (only with `--pv-mode vbop`): candidate with hysteresis, momentum m and spread
# v of the gradient projected on the candidate (EMA with `gamma`), curvature H in EMA, age and proposal direction.
# Units: m = mean per MICRO-BATCH of acc[row, cand] (the S/n of the V step), v = variance across micro-batches
# (divisor n-1), H = curvature per curvature micro-batch (H/n_curv); the STEP gain is N (m|d| - H d^2/2).
# Persistent memory: 8 B/group (cand u8 + m fp16 + sqrt(v) fp16 + H fp16 + age u8 with bit 7 = direction), plus
# 12 B/expert of fp32 scales (m, sqrt(v), H in fp16 with a power of 2 per expert: no underflow). The step buffer
# (17 B/group) stays as before. Thor-T (~208 M groups): +1.7 GB; ODINO (~16.7 G groups): +133 GB.
BYTE_PER_GROUP_VBOP = 8
ETA_MAX = 127                                 # 7 bits; bit 7 of the age is the direction (1 = proposal with acc < 0)
_FIELDS_BOP = ("cand", "m", "r", "H", "eta", "sm", "sr", "sH")


@dataclass
class ConfigBop:
    gamma: float = 0.5             # EMA of m, v, H: x <- (1 - gamma) x + gamma x_step (gamma = 1: no memory)
    curvature_every: int = 1        # curvature measured every N V steps (1 = every step, as in PV); in between its EMA
    hysteresis: float = 2.0          # the candidate changes only if |new prop| > hysteresis |m| (or if no longer feasible)
    nu_max: int = 200              # maximum degrees of freedom of Student's t (beyond that, t ~ normal)

    def __post_init__(self):
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError(f"gamma V-Bop {self.gamma} outside (0, 1]")
        if int(self.curvature_every) < 1:
            raise ValueError(f"curvature_every {self.curvature_every} < 1")


def _encode16(x: torch.Tensor) -> tuple:
    """x (ne, ...) float64 -> (fp16, fp32 scale per expert): scale = 2^(ceil(log2 max|x|) - 14), max|x/scale| <= 2^14
    (divisions by powers of 2 are exact: same result everywhere, bit-exact resume)."""
    ne = x.shape[0]
    amax = x.abs().reshape(ne, -1).amax(1)
    esp = torch.where(amax > 0, torch.ceil(torch.log2(torch.where(amax > 0, amax, torch.ones_like(amax)))) - 14,
                      torch.zeros_like(amax)).clamp(-120, 120)
    s = torch.pow(torch.full_like(esp, 2.0), esp)
    shape = (ne,) + (1,) * (x.dim() - 1)
    return (x / s.reshape(shape)).clamp(-65504.0, 65504.0).to(torch.float16), s.to(torch.float32)


def _decode16(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    return q.to(torch.float64) * s.to(torch.float64).reshape((q.shape[0],) + (1,) * (q.dim() - 1))


@dataclass
class BopState:
    """V-Bop state of ONE family (E, R, NG) + per-expert scales (E,). age = 0: no candidate with statistics."""
    cand: torch.Tensor
    m: torch.Tensor
    r: torch.Tensor
    H: torch.Tensor
    eta: torch.Tensor
    sm: torch.Tensor
    sr: torch.Tensor
    sH: torch.Tensor

    @staticmethod
    def create(fam, device=None) -> "BopState":
        dev = device if device is not None else fam.blocks.device
        shape = (fam.E, fam.R, fam.NG)
        z16 = lambda: torch.zeros(shape, dtype=torch.float16, device=dev)  # noqa: E731
        u1 = lambda: torch.ones(fam.E, dtype=torch.float32, device=dev)  # noqa: E731
        return BopState(torch.zeros(shape, dtype=torch.uint8, device=dev), z16(), z16(), z16(),
                        torch.zeros(shape, dtype=torch.uint8, device=dev), u1(), u1(), u1())

    def eta_direction(self, e0: int = 0, e1: Optional[int] = None) -> tuple:
        """(age int64, direction +-1 float64) of experts [e0, e1): direction = sign of the adopted proposal acc."""
        et = self.eta[e0:e1].to(torch.int64)
        return et & ETA_MAX, torch.where((et & 128) != 0, -1.0, 1.0).to(torch.float64)

    def byte(self) -> int:
        return sum(getattr(self, n).numel() * getattr(self, n).element_size() for n in _FIELDS_BOP)


def _correction_eta(gamma: float, eta: torch.Tensor) -> tuple:
    """(c, sum of weights^2) of the EMA after `eta` updates from zero: c = 1 - (1-gamma)^eta (bias correction,
    as in Adam), sum w^2 of the normalised weights w_t = gamma (1-gamma)^(eta-t) / c. gamma = 1: (1, 1)."""
    a = eta.to(torch.float64)
    b = 1.0 - gamma
    c = 1.0 - torch.pow(torch.full_like(a, b), a)
    c = torch.where(eta > 0, c, torch.ones_like(c))
    den = 1.0 - b * b
    sw2 = gamma * gamma * (1.0 - torch.pow(torch.full_like(a, b * b), a)) / (den * c * c)
    return c, torch.where(eta > 0, sw2, torch.ones_like(sw2))


class VState:
    """Buffers of all families + the mode of the current micro-batch (read by the engine inside `dd`).
    The trainer sets the mode: `start_step()`, then `microbatch(k)` before each loss backward, then
    `curvature()` before each curvature-measurement backward, then `end_collection()` (mode 0: any other
    backward, e.g. the G1 gate, stays as today)."""

    def __init__(self, n_proposal: int = 2, steps_accumulator: int = 1, bop: Optional[ConfigBop] = None,
                 curv_theta: bool = False):
        # 2026-09-27 V-Bop (`bop`): persistent per-group state across steps (BopState); None = previous PV, bit-exact
        self.bop = bop
        # 2026-09-27 newton-theta: with `curv_theta` the buffers have HT (curvature of the scales, per STEP) and the
        # step's curvature micro-batches are counted (`n_curvature_step`); off = no extra buffer, same numbers
        self.curv_theta = bool(curv_theta)
        self.n_curvature_step = 0
        self.bop_: dict = {}
        self.steps_bop = 0
        if bop is not None and int(steps_accumulator) > 1:
            raise ValueError("V-Bop and accumulation window together: the memory across steps is already the V-Bop EMA")
        self.buffer_: dict = {}
        self.mode = 0
        self.n_proposal = n_proposal
        self.n_verify = 0
        self.n_propose = 0
        self.n_curvature = 0
        # 2026-09-27: accumulation window (pv_passi_accumulo). With 1 (default) every step resets as before, bit-exact.
        # With N > 1 the statistics (S, Q, H, counts) are summed over N steps with FROZEN codes: the proposal (cand,
        # prop) is made only in the first step of the window, the other steps use ALL micro-batches for verification
        # (more power for the tests). The trainer closes the window (`close_window`) after the decision. `d_window`:
        # fp16 scales of each group at the start of the window (staleness: see `select(d_ref=...)`).
        self.steps_accumulator = max(1, int(steps_accumulator))
        self.steps_in_window = 0
        self.d_window: dict = {}
        # 2026-09-27 layer-domain branch (--layers-trained): keys (layer, family) of the TRAINED families; None = all
        # (as always). The others have buffers and V-Bop state (bit-exact resume) but no proposal/verification (dL/dd
        # is not computed in frozen layers), no V-Bop update and no jump (`VStep.select`)
        self.active: Optional[set] = None
        # 2026-10-02 tt5-phase-v branch (--pv-dd-only-skips, FASE_V.md fix V1). Off (default) = as always, bit-exact.
        # On: in the backwards of the V-phase GRADIENTS (`prepare_recenter` -> mode 2: midpoint prices and recenter)
        # the dL/dd of families WITHOUT prepared jumps is not computed (`dd_redundant`): that dL/dd only goes into the
        # theta gradient, which gradient_recenter/gradiente_lettere discard and restore; S (read only on the prepared
        # jumps) and dX (which carries the gradient to the layers below) stay identical, as does the overflow check
        # (it watches dX and the fp16 inputs, not dL/dd). `families_recenter`: keys with at least one prepared jump
        # (None outside the V-phase gradients: the P-step verification in mode 2 computes everything as always).
        self.skip_dd_redundant = False
        self.families_recenter: Optional[set] = None

    def _active(self, k) -> bool:
        return self.active is None or k in self.active

    def prepare(self, families: Iterable[S.FamilyData]) -> None:
        for f in families:
            k = key(f)
            if k in self.buffer_:
                raise KeyError(f"family {k} twice")
            if getattr(f, "E", 1) > 256 or getattr(f, "R", 0) > 65536 or getattr(f, "C", 0) > 65536:
                # 2026-09-28: the registry writes e as u1, r and c as u2 (REG_DTYPE): out of range they would be
                # silently truncated (e.g. token_embd/output of variant "B", 248,320 rows)
                raise ValueError(f"family {k} ({f.E}x{f.R}x{f.C}) out of range of the jump registry")
            self.buffer_[k] = BufferV.create(f, curv_theta=self.curv_theta)
            if self.bop is not None:
                self.bop_[k] = BopState.create(f)

    def buffer(self, fam: S.FamilyData) -> BufferV:
        k = key(fam)
        if k not in self.buffer_:
            raise KeyError(f"no V buffer for {k}: `prepare` did not see it")
        return self.buffer_[k]

    def start_step(self) -> None:
        if self.steps_accumulator <= 1 or self.steps_in_window == 0:
            for b in self.buffer_.values():
                b.reset()
            self.mode, self.n_verify, self.n_propose, self.n_curvature = 0, 0, 0, 0
            self.steps_in_window = 0
            self.d_window = {}
        self.steps_in_window += 1
        self.mode = 0
        self.families_recenter = None
        self.n_curvature_step = 0                       # newton-theta: HT belongs to the STEP, even inside a window
        if self.curv_theta:
            for b in self.buffer_.values():
                b.HT.zero_()

    def close_window(self) -> None:
        """After the decision (jumps kept or cancelled): the next `start_step` opens a new window."""
        self.steps_in_window = 0

    def microbatch(self, k: int) -> int:
        """k = index of the micro-batch within the step (0-based). 0..n_proposal-1 -> proposal, then verification. In an
        accumulation window only the FIRST step proposes: the following steps verify with all micro-batches."""
        first = self.steps_in_window <= 1
        self.mode = 1 if (first and k < self.n_proposal) else 2
        if self.mode == 1:
            self.n_propose += 1
        else:
            if self.bop is not None and self.n_verify == 0:
                self._merge_candidates()                  # V-Bop: first verification of the step = the step's candidates
            self.n_verify += 1
        return self.mode

    # ─── V-Bop ───
    def _merge_candidates(self) -> None:
        """Between proposal and verification: the persistent candidate stays (hysteresis) unless |new prop| >
        hysteresis |m|; if it changes (or there was none) m, v, H and the age are reset and the direction is that of the
        new proposal. The step buffer (read by the kernel in modes 2 and 3) receives the step's candidate: persistent
        cand and prop = +-1 (its direction)."""
        ist = float(self.bop.hysteresis)
        for k, b in self.buffer_.items():
            if not self._active(k):                      # layer domain: V-Bop state of frozen layers kept bit-exact
                continue
            z = self.bop_[k]
            eta, direction = z.eta_direction()
            ha = eta >= 1
            c, _ = _correction_eta(self.bop.gamma, eta)
            m_hat = _decode16(z.m, z.sm) / c
            new = b.prop != 0
            subst = new & (~ha | (b.prop.to(torch.float64).abs() > ist * m_hat.abs()))
            keep = ha & ~subst
            z.cand.copy_(torch.where(subst, b.cand, z.cand))
            z.eta.copy_(torch.where(subst, torch.where(b.prop < 0, 128, 0).to(torch.uint8), z.eta))
            zero = torch.zeros((), dtype=torch.float16, device=z.m.device)
            for n in ("m", "r", "H"):
                getattr(z, n).copy_(torch.where(subst, zero, getattr(z, n)))
            b.cand.copy_(torch.where(keep, z.cand, b.cand))
            b.prop.copy_(torch.where(keep, direction.to(torch.float32), b.prop))

    def need_curvature(self) -> bool:
        """V-Bop with `curvature_every` N: the curvature is measured at V steps 0, N, 2N, ... (in between its EMA holds)."""
        return self.bop is None or self.steps_bop % int(self.bop.curvature_every) == 0

    def update_bop(self, piece: int = 32) -> None:
        """End of V-Bop collection: EMA of m (mean per micro-batch of acc on the candidate), v (variance across the
        verification micro-batches) and H (curvature per micro-batch, only if measured in this step; the first
        measurement enters in full) for the groups with a candidate in the step; age + 1 (cap ETA_MAX). New per-expert
        scales."""
        n, nc, gam = self.n_verify, self.n_curvature, float(self.bop.gamma)
        if n < 2:
            raise ValueError(f"V-Bop: at least 2 verification micro-batches are needed (there are {n})")
        for k, b in self.buffer_.items():
            if not self._active(k):                      # layer domain: V-Bop state of frozen layers kept bit-exact
                continue
            z = self.bop_[k]
            E = b.prop.shape[0]
            for e0 in range(0, E, piece):
                e1 = min(E, e0 + piece)
                ha = b.prop[e0:e1] != 0
                Sd, Qd = b.S[e0:e1].to(torch.float64), b.Q[e0:e1].to(torch.float64)
                xm = Sd / n
                s2 = ((Qd - Sd * Sd / n) / (n - 1)).clamp_min(0.0)
                m = _decode16(z.m[e0:e1], z.sm[e0:e1])
                v = _decode16(z.r[e0:e1], z.sr[e0:e1]) ** 2
                H = _decode16(z.H[e0:e1], z.sH[e0:e1])
                m = torch.where(ha, (1.0 - gam) * m + gam * xm, m)
                v = torch.where(ha, (1.0 - gam) * v + gam * s2, v)
                if nc > 0:
                    Ht = b.H[e0:e1].to(torch.float64) / nc
                    H = torch.where(ha, torch.where(H > 0, (1.0 - gam) * H + gam * Ht, Ht), H)
                eta = z.eta[e0:e1].to(torch.int64)
                eta_n = torch.where(ha, torch.clamp((eta & ETA_MAX) + 1, max=ETA_MAX) | (eta & 128), eta)
                z.eta[e0:e1] = eta_n.to(torch.uint8)
                for field_, sc, x in (("m", "sm", m), ("r", "sr", torch.sqrt(v)), ("H", "sH", H)):
                    q, s = _encode16(x)
                    getattr(z, field_)[e0:e1] = q
                    getattr(z, sc)[e0:e1] = s
        self.steps_bop += 1

    def byte_bop(self) -> int:
        return sum(z.byte() for z in self.bop_.values())

    def snapshot_bop(self) -> dict:
        """V-Bop state for resuming (tensors cloned to CPU)."""
        return {"gamma": float(self.bop.gamma), "curvature_every": int(self.bop.curvature_every),
                "steps_bop": self.steps_bop,
                "state": {f"{k[0]}.{k[1]}": {n: getattr(z, n).detach().cpu().clone() for n in _FIELDS_BOP}
                          for k, z in self.bop_.items()}}

    def restore_bop(self, ist: dict) -> None:
        names = {f"{k[0]}.{k[1]}": k for k in self.bop_}
        if set(ist["state"]) != set(names):
            raise ValueError("V-Bop state saved with different families")
        for kn, d in ist["state"].items():
            z = self.bop_[names[kn]]
            for n in _FIELDS_BOP:
                t = getattr(z, n)
                if d[n].shape != t.shape or d[n].dtype != t.dtype:
                    raise ValueError(f"V-Bop state {kn}.{n}: shape/type {tuple(d[n].shape)} {d[n].dtype}")
                t.copy_(d[n].to(t.device))
        self.steps_bop = int(ist["steps_bop"])

    def curvature(self) -> int:
        """Before a MEASUREMENT backward (random vector with covariance = Hessian of the outputs): mode 3."""
        if self.n_propose == 0:
            raise RuntimeError("curvature before the proposal: the candidates do not exist yet")
        self.mode = 3
        self.n_curvature += 1
        self.n_curvature_step += 1
        return self.mode

    @property
    def n_loss(self) -> int:
        """LOSS micro-batches of the step (proposal + verification): the step gradient sums all of them."""
        return self.n_propose + self.n_verify

    def end_collection(self) -> None:
        self.mode = 0
        self.families_recenter = None

    def prepare_recenter(self, skips: dict) -> None:
        """2026-09-27 v-interaction-b branch (--pv-recenter): the step buffers, AFTER the decision (nobody reads them
        again until the next `start_step`, which resets them), are reused to read the probe gradient in the reached
        state on the column of each applied jump: cand := column of the jump, prop := 1 (groups without a jump:
        prop = 0, ignored by the kernel), S = Q = 0, mode 2 (verification). `skips`: {key: (e, r, g, cand)}. After the
        backward S[e, r, g] = dL/dW_rot on the jumped code; the caller sets mode 0 again (`end_collection`). The step
        counts (n_verify, ...) are not touched; H and HT remain those of the step.
        2026-10-02 (`skip_dd_redundant`): `families_recenter` = keys with at least one jump (read by `dd_redundant`)."""
        for k, b in self.buffer_.items():
            for t in (b.cand, b.prop, b.S, b.Q):
                t.zero_()
            if k in skips:
                e, r, g, c = skips[k]
                if e.numel():
                    dev = b.cand.device
                    e, r, g = e.to(dev), r.to(dev), g.to(dev)
                    b.cand[e, r, g] = c.to(dev, torch.uint8)
                    b.prop[e, r, g] = 1.0
        self.families_recenter = {k for k, t in skips.items() if t[0].numel()}
        self.mode = 2

    def byte(self) -> int:
        return sum(b.byte() for b in self.buffer_.values())

    def snapshot(self) -> dict:
        """Window state for resuming (tensors on CPU): buffers, snapshot of the scales, counts."""
        return {"steps_accumulator": self.steps_accumulator, "steps_in_window": self.steps_in_window,
                "n_verify": self.n_verify, "n_propose": self.n_propose, "n_curvature": self.n_curvature,
                "buffer": {f"{k[0]}.{k[1]}": {n: getattr(b, n).detach().cpu() for n in ("cand", "prop", "S", "Q", "H")}
                           for k, b in self.buffer_.items()},
                "d_window": {f"{k[0]}.{k[1]}": v.detach().cpu() for k, v in self.d_window.items()}}

    def restore(self, ist: dict) -> None:
        """Inverse of `snapshot`: same window, same numbers bit-exact (the buffers return to their device)."""
        if int(ist["steps_accumulator"]) != self.steps_accumulator:
            raise ValueError(f"window saved with steps_accumulator {ist['steps_accumulator']}, now {self.steps_accumulator}")
        names = {f"{k[0]}.{k[1]}": k for k in self.buffer_}
        if set(ist["buffer"]) != set(names):
            raise ValueError("window saved with different families")
        for kn, d in ist["buffer"].items():
            b = self.buffer_[names[kn]]
            for n in ("cand", "prop", "S", "Q", "H"):
                getattr(b, n).copy_(d[n].to(getattr(b, n).device))
        self.d_window = {names[kn]: v.to(self.buffer_[names[kn]].prop.device) for kn, v in ist["d_window"].items()}
        self.steps_in_window = int(ist["steps_in_window"])
        self.n_verify, self.n_propose, self.n_curvature = (int(ist["n_verify"]), int(ist["n_propose"]),
                                                             int(ist["n_curvature"]))
        self.mode = 0


# ═══════════════════════════════════ 2. kernel ═══════════════════════════════════
if S._TRITON:
    @triton.jit
    def _k_dd_v(dy_ptr, sdy_s, sdy_r, x_ptr, sx_s, sx_c,
                pb, sb_e, NBR, dd_ptr, sdd_e, NG, offs_ptr, R, C,
                d0_ptr, sd0_e, cand_ptr, prop_ptr, s_ptr, q_ptr, h_ptr, sv_e,
                GS: tl.constexpr, QK: tl.constexpr, NB: tl.constexpr, ROT: tl.constexpr,
                BR: tl.constexpr, BS: tl.constexpr, MODE_V: tl.constexpr,
                dylo_ptr=None, inv_ptr=None, sinv_e=0, SPLIT: tl.constexpr = 0):
        """The student's `_k_dd` (identical lines up to the dL/dd store) + V-step epilogue on the same
        tile acc = dY^T X (BR x GS) in registers. MODE_V 0: dL/dd only. 1: proposal. 2: verification.
        3: curvature (acc = (dY^2)^T (X^2) in fp32, dL/dd = 0, H += acc[row, cand]).
        2026-09-27 (backward-bf16x3 branch, `e2e_backward_hilo` via "fp16x2"): SPLIT=1 -> dY = (hi + lo) * inv[e, r], hi
        in dy_ptr and lo in dylo_ptr (fp16, same strides), inv (E, R) fp32 powers of two: acc_lo and acc_hi in two
        accumulators, acc = (acc_lo + acc_hi) * inv BEFORE the epilogue (dL/dd and the V statistics see the full dY^T X).
        Not with MODE_V 3. SPLIT=0 (default): the previous kernel (constant branch eliminated at compile time)."""
        e = tl.program_id(0)
        pid_r = tl.program_id(1)
        g = tl.program_id(2)
        s_ini = tl.load(offs_ptr + e)
        s_end = tl.load(offs_ptr + e + 1)
        rr = pid_r * BR + tl.arange(0, BR)
        cc = g * GS + tl.arange(0, GS)
        mr = rr < R
        acc = tl.zeros((BR, GS), dtype=tl.float32)
        if SPLIT:
            acc_lo = tl.zeros((BR, GS), dtype=tl.float32)
        for s0 in range(s_ini, s_end, BS):
            rs = s0 + tl.arange(0, BS)
            ms = rs < s_end
            dyt = tl.load(dy_ptr + rs[None, :].to(tl.int64) * sdy_s + rr[:, None] * sdy_r,
                          mask=ms[None, :] & mr[:, None], other=0.0)
            x = tl.load(x_ptr + rs[:, None].to(tl.int64) * sx_s + cc[None, :] * sx_c,
                        mask=ms[:, None], other=0.0)
            if MODE_V == 3:
                d32 = dyt.to(tl.float32)
                x32 = x.to(tl.float32)
                acc += tl.dot(d32 * d32, x32 * x32)
            elif SPLIT:
                dyl = tl.load(dylo_ptr + rs[None, :].to(tl.int64) * sdy_s + rr[:, None] * sdy_r,
                              mask=ms[None, :] & mr[:, None], other=0.0)
                acc_lo += tl.dot(dyl, x)
                acc += tl.dot(dyt, x)
            else:
                acc += tl.dot(dyt, x)
        e64 = e.to(tl.int64)
        if SPLIT:
            inv = tl.load(inv_ptr + e64 * sinv_e + rr, mask=mr, other=1.0)
            acc = (acc_lo + acc) * inv[:, None]
        if MODE_V == 3:
            tl.store(dd_ptr + e64 * sdd_e + rr * NG + g, tl.zeros((BR,), dtype=tl.float32), mask=mr)
            ov3 = e64 * sv_e + rr * NG + g
            loc3 = tl.arange(0, GS)
            pr3 = tl.load(prop_ptr + ov3, mask=mr, other=0.0)
            ci3 = tl.load(cand_ptr + ov3, mask=mr, other=0).to(tl.int32)
            v3 = tl.sum(tl.where(loc3[None, :] == ci3[:, None], acc, 0.0), axis=1)
            v3 = tl.where(pr3 != 0.0, v3, 0.0)
            hv = tl.load(h_ptr + ov3, mask=mr, other=0.0)
            tl.store(h_ptr + ov3, hv + v3, mask=mr)
        else:
            t, m2 = _trit_from_byte(pb + e64 * sb_e, rr[:, None], cc[None, :], NBR, R, C, QK, NB, ROT)
            val = tl.sum(acc * t.to(tl.float32), axis=1)
            tl.store(dd_ptr + e64 * sdd_e + rr * NG + g, val, mask=mr)
            if MODE_V != 0:
                ov = e64 * sv_e + rr * NG + g
                loc = tl.arange(0, GS)
                if MODE_V == 1:
                    dv = tl.load(d0_ptr + e64 * sd0_e + rr * NG + g, mask=mr, other=0.0)
                    sd = tl.where(dv > 0.0, 1.0, tl.where(dv < 0.0, -1.0, 0.0))
                    pend = acc * sd[:, None]                 # dL/dt = acc * d: same sign as acc * sign(d)
                    dz = tl.where(pend > 0.0, -1, tl.where(pend < 0.0, 1, 0))
                    tn = t + dz
                    fatt = (dz != 0) & (tn >= -1) & (tn <= 1) & m2
                    punt = tl.where(fatt, tl.abs(acc), -1.0)
                    ci = tl.argmax(punt, axis=1)             # on ties, the lowest column (as torch.argmax)
                    best = tl.max(punt, axis=1)
                    pv = tl.sum(tl.where(loc[None, :] == ci[:, None], acc, 0.0), axis=1)   # exact: acc + zeros
                    pn = tl.where(best > 0.0, pv, 0.0)
                    po = tl.load(prop_ptr + ov, mask=mr, other=0.0)
                    co = tl.load(cand_ptr + ov, mask=mr, other=0)
                    rep = tl.abs(pn) > tl.abs(po)
                    tl.store(prop_ptr + ov, tl.where(rep, pn, po), mask=mr)
                    tl.store(cand_ptr + ov, tl.where(rep, ci.to(tl.uint8), co), mask=mr)
                else:
                    pr = tl.load(prop_ptr + ov, mask=mr, other=0.0)
                    ci = tl.load(cand_ptr + ov, mask=mr, other=0).to(tl.int32)
                    v = tl.sum(tl.where(loc[None, :] == ci[:, None], acc, 0.0), axis=1)
                    v = tl.where(pr != 0.0, v, 0.0)
                    sv = tl.load(s_ptr + ov, mask=mr, other=0.0)
                    qv = tl.load(q_ptr + ov, mask=mr, other=0.0)
                    tl.store(s_ptr + ov, sv + v, mask=mr)
                    tl.store(q_ptr + ov, qv + v * v, mask=mr)


if S._TRITON:
    @triton.jit
    def _k_ht(dy_ptr, sdy_s, sdy_r, x_ptr, sx_s, sx_c, pb, sb_e, NBR, offs_ptr, R, C, ht_ptr, sv_e, NG,
              GS: tl.constexpr, QK: tl.constexpr, NB: tl.constexpr, ROT: tl.constexpr,
              BR: tl.constexpr, BS: tl.constexpr, GSB: tl.constexpr):
        """2026-09-27 newton-theta branch: Gauss-Newton curvature of the SCALES per group (without d^2), as a SEPARATE
        kernel (inside `_k_dd_v` mode 3 it spilled 177-1061 VGPRs on gfx1151): HT[e, r, g] += sum_s dY_{s,r}^2 u_{s,r}^2,
        u_{s,r} = sum_{c in g} t_{r,c} X_{s,c} (the group's contribution to the output, at scale 1). Same grid as
        `_k_dd_v` (expert, row tile, group); X read already transposed; no atomics. GSB = columns per
        sub-block: GSB == GS (R1024, group 128) -> trits decoded ONCE before the loop; GSB < GS (TQ1_0,
        group 256) -> u in sub-blocks with the trits re-decoded at every token tile (with the full GS in
        registers the kernel spilled 46-60 VGPRs)."""
        e = tl.program_id(0)
        pid_r = tl.program_id(1)
        g = tl.program_id(2)
        s_ini = tl.load(offs_ptr + e)
        s_end = tl.load(offs_ptr + e + 1)
        rr = pid_r * BR + tl.arange(0, BR)
        mr = rr < R
        e64 = e.to(tl.int64)
        h = tl.zeros((BR,), dtype=tl.float32)
        if GSB == GS:
            cc = g * GS + tl.arange(0, GS)
            t, mt = _trit_from_byte(pb + e64 * sb_e, rr[:, None], cc[None, :], NBR, R, C, QK, NB, ROT)
            tf = tl.where(mt, t.to(tl.float32), 0.0)
            for s0 in range(s_ini, s_end, BS):
                rs = s0 + tl.arange(0, BS)
                ms = rs < s_end
                dyt = tl.load(dy_ptr + rs[None, :].to(tl.int64) * sdy_s + rr[:, None] * sdy_r,
                              mask=ms[None, :] & mr[:, None], other=0.0).to(tl.float32)
                xt = tl.load(x_ptr + rs[None, :].to(tl.int64) * sx_s + cc[:, None] * sx_c,
                             mask=ms[None, :], other=0.0).to(tl.float32)        # (GS, BS)
                u = tl.dot(tf, xt)                                               # (BR, BS)
                h += tl.sum(dyt * dyt * u * u, axis=1)
        else:
            for s0 in range(s_ini, s_end, BS):
                rs = s0 + tl.arange(0, BS)
                ms = rs < s_end
                dyt = tl.load(dy_ptr + rs[None, :].to(tl.int64) * sdy_s + rr[:, None] * sdy_r,
                              mask=ms[None, :] & mr[:, None], other=0.0).to(tl.float32)
                u = tl.zeros((BR, BS), dtype=tl.float32)
                for j in range(GS // GSB):
                    cc = g * GS + j * GSB + tl.arange(0, GSB)
                    t, mt = _trit_from_byte(pb + e64 * sb_e, rr[:, None], cc[None, :], NBR, R, C, QK, NB, ROT)
                    xt = tl.load(x_ptr + rs[None, :].to(tl.int64) * sx_s + cc[:, None] * sx_c,
                                 mask=ms[None, :], other=0.0).to(tl.float32)    # (GSB, BS)
                    u += tl.dot(tl.where(mt, t.to(tl.float32), 0.0), xt)
                h += tl.sum(dyt * dyt * u * u, axis=1)
        ov = e64 * sv_e + rr * NG + g
        hv = tl.load(ht_ptr + ov, mask=mr, other=0.0)
        tl.store(ht_ptr + ov, hv + h, mask=mr)


# Sizes of `_k_ht`: ahead-of-time compilation for gfx1151 (2026-09-27, test N5 of e2e_test_sophia.py): BR 16, BS 16,
# 8 warps, sub-blocks of 128 columns -> 0 VGPRs spilled with R1024 and TQ1_0, fp16 and fp32 inputs. Time must be measured
# on the GPU.
BLOCKS_HT = S.Blocks(BR_dd=16, BS_dd=16, num_warps=8)
GSB_HT = 128


def ht_triton(dY: torch.Tensor, X: torch.Tensor, fam: S.FamilyData, plane: S.Plane, HT: torch.Tensor,
              blk: Optional[S.Blocks] = None) -> None:
    """HT += curvature of the scales for the micro-batch (in place). dY and X as in `dd_v_triton`."""
    b = blk or BLOCKS_HT
    dY, X = dY.contiguous(), X.contiguous()
    grid = (fam.E, triton.cdiv(fam.R, b.BR_dd), fam.NG)
    _k_ht[grid](dY, dY.stride(0), dY.stride(1), X, X.stride(0), X.stride(1),
                   fam.blocks, fam.blocks.stride(0), fam.NBR, plane.offs, fam.R, fam.C, HT, fam.R * fam.NG, fam.NG,
                   fam.gs, fam.qk, fam.nb, int(fam.rotated), b.BR_dd, b.BS_dd, min(fam.gs, GSB_HT),
                   num_warps=b.num_warps)


def ht_torch(gy: torch.Tensor, x: torch.Tensor, fam: S.FamilyData, plane: S.Plane, HT: torch.Tensor) -> None:
    """Reference for `_k_ht`, expert by expert (u materialised (s, R, NG)); fp64 if the inputs are."""
    acc = torch.float64 if gy.dtype == torch.float64 else torch.float32
    o = plane.offs.tolist()
    for e in range(fam.E):
        if o[e + 1] > o[e]:
            g2 = gy[o[e]:o[e + 1]].to(acc) ** 2                                  # (s, R)
            xe = x[o[e]:o[e + 1]].to(acc).view(-1, fam.NG, fam.gs)               # (s, NG, gs)
            t = S.trit_expert(fam, e).to(acc).view(fam.R, fam.NG, fam.gs)
            u = torch.einsum("sgj,rgj->srg", xe, t)
            HT[e] += torch.einsum("sr,srg->rg", g2, u * u).to(HT.dtype)


def dd_v_triton(dY: torch.Tensor, X: torch.Tensor, fam: S.FamilyData, plane: S.Plane, blk: S.Blocks,
                buf: Optional[BufferV], mode: int, dy_lo: Optional[torch.Tensor] = None,
                inv: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Like `S._dd_triton` (same grid, same sizes) with the `_k_dd_v` kernel. mode 0: `buf` may be None.
    2026-09-27: `dy_lo` + `inv` (E, R) = split dY (fp16x2 of `e2e_backward_hilo`): true dY = (dY + dy_lo) * inv[e, r]."""
    dY, X = dY.contiguous(), X.contiguous()
    assert dY.dtype == X.dtype
    split = dy_lo is not None
    if split:
        if mode == 3 or inv is None:
            raise ValueError("dd_v_triton split: needs inv, and not in mode 3 (curvature)")
        dy_lo = dy_lo.contiguous()
        inv = inv.to(torch.float32).contiguous()
        if dy_lo.shape != dY.shape or dy_lo.dtype != dY.dtype or dy_lo.stride() != dY.stride() \
                or tuple(inv.shape) != (fam.E, fam.R):
            raise ValueError("dd_v_triton split: dy_lo/inv with wrong shape, type or strides")
    dd = torch.empty(fam.E, fam.R, fam.NG, dtype=torch.float32, device=dY.device)
    d0 = fam.d0.contiguous()
    if buf is None:
        assert mode == 0
        z8 = torch.zeros(1, dtype=torch.uint8, device=dY.device)
        z32 = torch.zeros(1, dtype=torch.float32, device=dY.device)
        cand, prop, s_, q_, h_, sv_e = z8, z32, z32, z32, z32, 0
    else:
        cand, prop, s_, q_, h_, sv_e = buf.cand, buf.prop, buf.S, buf.Q, buf.H, fam.R * fam.NG
    grid = (fam.E, triton.cdiv(fam.R, blk.BR_dd), fam.NG)
    _k_dd_v[grid](dY, dY.stride(0), dY.stride(1), X, X.stride(0), X.stride(1),
                     fam.blocks, fam.blocks.stride(0), fam.NBR, dd, dd.stride(0), fam.NG, plane.offs,
                     fam.R, fam.C, d0, d0.stride(0), cand, prop, s_, q_, h_, sv_e,
                     fam.gs, fam.qk, fam.nb, int(fam.rotated), blk.BR_dd, blk.BS_dd, int(mode),
                     **({"dylo_ptr": dy_lo, "inv_ptr": inv, "sinv_e": inv.stride(0), "SPLIT": 1} if split else {}),
                     num_warps=blk.num_warps)
    if mode == 3 and buf is not None and buf.HT is not None:     # newton-theta: curvature of the scales
        ht_triton(dY, X, fam, plane, buf.HT)
    return dd


# ═══════════════════════════════════ 3. torch reference (materialised dW) ═══════════════════════════════════
def proposal_from_dw(buf: BufferV, e: int, dW: torch.Tensor, q: torch.Tensor, d0e: torch.Tensor, gs: int) -> None:
    """Proposal from the EXPLICIT dW (R, C) of expert e (reference for mode 1): same rule as the kernel."""
    R, C = dW.shape
    A = dW.view(R, C // gs, gs)
    t = q.to(torch.int32).view(R, C // gs, gs)
    pend = A * torch.sign(d0e.to(A.dtype)).unsqueeze(-1)
    dz = -torch.sign(pend).to(torch.int32)
    tn = t + dz
    fatt = (dz != 0) & (tn >= -1) & (tn <= 1)
    punt = torch.where(fatt, A.abs(), torch.full_like(A, -1.0))
    best = punt.amax(-1)
    ci = punt.argmax(-1)                          # on ties the first index (as tl.argmax, tie_break_left)
    pv = A.gather(-1, ci.unsqueeze(-1)).squeeze(-1)
    pn = torch.where(best > 0, pv, torch.zeros_like(pv)).to(torch.float32)
    rep = pn.abs() > buf.prop[e].abs()
    buf.prop[e] = torch.where(rep, pn, buf.prop[e])
    buf.cand[e] = torch.where(rep, ci.to(torch.uint8), buf.cand[e])


def verify_from_dw(buf: BufferV, e: int, dW: torch.Tensor, gs: int) -> None:
    R, C = dW.shape
    A = dW.view(R, C // gs, gs)
    v = A.gather(-1, buf.cand[e].to(torch.int64).unsqueeze(-1)).squeeze(-1).to(torch.float32)
    v = torch.where(buf.prop[e] != 0, v, torch.zeros_like(v))
    buf.S[e] += v
    buf.Q[e] += v * v


def curvature_from_dw2(buf: BufferV, e: int, dW2: torch.Tensor, gs: int) -> None:
    """Mode 3 from the EXPLICIT (dY^2)^T (X^2) (R, C) of expert e: H += value on the candidate (where there is one)."""
    R, C = dW2.shape
    A = dW2.view(R, C // gs, gs)
    v = A.gather(-1, buf.cand[e].to(torch.int64).unsqueeze(-1)).squeeze(-1).to(torch.float32)
    buf.H[e] += torch.where(buf.prop[e] != 0, v, torch.zeros_like(v))


def dd_v_torch(gy, x, fam: S.FamilyData, plane: S.Plane, d_dtype, buf: Optional[BufferV], mode: int):
    """`S.TorchEngine.dd` (same operations, bit-exact same output) + proposal/verification from the explicit dW (R, C)
    of each expert: it is the "materialised" reference the kernel is tested against. Transient memory: ONE dW.
    Mode 3 (curvature): dL/dd = 0 and H from the explicit (dY^2)^T (X^2)."""
    acc = torch.float64 if gy.dtype == torch.float64 else torch.float32
    dd = torch.zeros(fam.E, fam.R, fam.NG, dtype=d_dtype, device=gy.device)
    o = plane.offs.tolist()
    if mode == 3:
        for e in range(fam.E):
            if o[e + 1] > o[e]:
                g2 = gy[o[e]:o[e + 1]].to(torch.float64 if acc == torch.float64 else torch.float32) ** 2
                x2 = x[o[e]:o[e + 1]].to(g2.dtype) ** 2
                curvature_from_dw2(buf, e, torch.matmul(g2.T, x2), fam.gs)
        if buf.HT is not None:                                   # newton-theta: curvature of the scales
            ht_torch(gy, x, fam, plane, buf.HT)
        return dd
    for e in range(fam.E):
        if o[e + 1] > o[e]:
            dW = torch.matmul(gy[o[e]:o[e + 1]].to(acc).T, x[o[e]:o[e + 1]].to(acc))       # (R, C)
            q = S.trit_expert(fam, e).to(acc)
            dd[e] = (q * dW).view(fam.R, fam.NG, fam.gs).sum(-1).to(d_dtype)
            if mode == 1:
                proposal_from_dw(buf, e, dW, q, fam.d0[e], fam.gs)
            elif mode == 2:
                verify_from_dw(buf, e, dW, fam.gs)
        # expert without tokens: acc = 0 -> null proposal (no change), verification v = 0 (S, Q unchanged)
    return dd


# ═══════════════════════════════════ 4. engines ═══════════════════════════════════
class TorchEngineV(S.TorchEngine):
    """Torch engine of the student; `dd` in V mode also collects proposal/verification (dW per expert)."""

    def __init__(self, state: VState):
        self.state = state
        self.name = "torch+V"

    def dd(self, gy, x, fam, plane, d_dtype):
        m = self.state.mode
        if m == 0:
            return super().dd(gy, x, fam, plane, d_dtype)
        return dd_v_torch(gy, x, fam, plane, d_dtype, self.state.buffer(fam), m)


# Sizes of the V kernel (T5, compilation for gfx1151): with the production Blocks (BR 64, 8 warps) mode 1 spills
# 76-530 VGPRs; with BR 32 and 8 warps: 0 for R1024 fp16/fp32 and TQ1_0 fp16, 4 (mode 1) and 1 (mode 2) for TQ1_0 fp32,
# where today's production `_k_dd` already spills 171. Timing is decided by V-M0 on the GPU.
BLOCKS_V = S.Blocks(BR_dd=32)

if S._TRITON:
    class TritonEngineV(S.TritonEngine):
        """Triton engine of the student (isinstance TritonEngine: the fused combination stays on); `dd` in V mode
        uses `_k_dd_v` with `blk_v` (default BLOCKS_V); mode 0 keeps the dL/dd of the base engine (`_k_dd`, or
        `_k_dd2` if the base is "triton2": v2=True, which also applies to forward and dX). V modes 1-3 ALWAYS use
        `_k_dd_v`, identical for both bases."""

        def __init__(self, state: VState, dense: bool = False, blk: Optional[S.Blocks] = None,
                     blk_v: Optional[S.Blocks] = None, v2: bool = False):
            super().__init__(dense=dense, blk=blk, v2=v2)
            self.state = state
            self.blk_v = blk_v or BLOCKS_V
            self.name = self.name + "+V"

        def dd(self, gy, x, fam, plane, d_dtype):
            m = self.state.mode
            if m == 0:
                return super().dd(gy, x, fam, plane, d_dtype)
            return dd_v_triton(gy, x, fam, plane, self.blk_v, self.state.buffer(fam), m).to(d_dtype)


def install(st, state: VState, engine: str = "auto", blk_v: Optional[S.Blocks] = None):
    """Puts the V engine in ALL layers of an `S.Student` (same object for all: a single `state`),
    prepares the buffers of its families and returns {(L, name): FamilyData}. Touches neither theta nor the bytes."""
    if engine == "auto":
        engine = "triton" if isinstance(st.mot, S.TritonEngine) else "torch"
    if engine == "torch":
        mv = TorchEngineV(state)
    else:
        base = st.mot
        mv = TritonEngineV(state, dense=getattr(base, "dense", False), blk=getattr(base, "blk", None), blk_v=blk_v,
                           v2=getattr(base, "v2", False))
    fams = {}
    for L in st.layers:
        es = st.experts(L)
        es.mot = mv
        for n, f in es.fam.items():
            fams[(L, n)] = f
    st.mot = mv
    state.prepare(fams.values())
    return fams


# ═══════════════════════════════════ 5. selection ═══════════════════════════════════
@dataclass
class ConfigV:
    q_fdr: float = 0.1             # Benjamini-Hochberg per matrix
    n_proposal: int = 2            # proposal micro-batches (the others verify)
    steps_accumulator: int = 1        # accumulation window of the V statistics (1 = decision at every step, as before)
    piece_experts: int = 32        # experts per piece in the selection (bounded transient memory)
    piece_decode: int = 8      # experts per piece when counting non-zeros
    threshold_staleness: float = 0.02 # accumulation window: a group whose scale moved by more than 2% since the start
                                   # of the window does not jump (its signal was collected with a different scale)
    q_gain: float = 0.1        # BH per matrix on the hypothesis "the jump does NOT reduce the loss" (second-order
                                   # gain <= 0): with the point value, a teacher off the grid (residual 0.4 d) made
                                   # the loss rise (winner's curse among thousands of codes)
    # 2026-09-27 --pv-accept (v-without-bh branch, TECNICHE_E2E_PV.md §38). "bh" (default) = the two BH tests above,
    # bit-exact. "gain" = every candidate with ESTIMATED second-order gain > 0 is accepted (|g| d > d^2 H / 2; V-Bop:
    # N (m^ |d| - H d^2 / 2) > 0) WITHOUT the BH tests (q_fdr and q_gain are not read); gain ordering, guaranteed rho
    # budget, probe with bit-exact cancellation and adaptive rho stay unchanged. `z_min` (only with "gain"):
    # additionally z > z_min, z = Student's t of the gradient in the proposal direction (the same z as the direction
    # test; with gain > 0 z is always > 0, so z > z_min == |z| > z_min). 0 = no threshold.
    # `z_gain_min` (only with "gain"): additionally t_g > z_gain_min, t_g = t of the GAIN (the one of the
    # q_gain test, `t_gain`; V-Bop: (m^ direction - H|d|/2) / se) = lower confidence bound of the gain > 0 at
    # z_gain_min standard errors, WITHOUT multiplicity correction. 0 = no threshold (t_g > 0 <=> gain > 0).
    accept: str = "bh"
    z_min: float = 0.0
    z_gain_min: float = 0.0

    def __post_init__(self):
        if self.accept not in ("bh", "gain"):
            raise ValueError(f"accept {self.accept!r}: 'bh' or 'gain'")
        for name in ("z_min", "z_gain_min"):
            v = getattr(self, name)
            if not v >= 0.0:
                raise ValueError(f"{name} {v} < 0")
            if self.accept == "bh" and v != 0.0:
                raise ValueError(f"{name} > 0 only with accept 'gain' (with 'bh' the BH tests set the threshold)")


def sf_t(t: torch.Tensor, nu: int) -> torch.Tensor:
    """P(T > t), Student's T with nu (integer >= 1) degrees: closed form of Abramowitz-Stegun 26.7.3 (odd nu)
    and 26.7.4 (even nu), in float64, on the tensor's device (no scipy). t = +-inf allowed."""
    t = t.to(torch.float64)
    a = t.abs()
    th = torch.atan(a / math.sqrt(nu))
    c, s = torch.cos(th), torch.sin(th)
    c2 = c * c
    if nu % 2 == 1:
        if nu == 1:
            A = 2.0 * th / math.pi
        else:
            term = c.clone()
            sum_ = c.clone()
            for k in range(1, (nu - 3) // 2 + 1):
                term = term * c2 * (2.0 * k) / (2.0 * k + 1.0)
                sum_ = sum_ + term
            A = (2.0 / math.pi) * (th + s * sum_)
    else:
        term = torch.ones_like(c)
        sum_ = torch.ones_like(c)
        for k in range(1, (nu - 2) // 2 + 1):
            term = term * c2 * (2.0 * k - 1.0) / (2.0 * k)
            sum_ = sum_ + term
        A = s * sum_
    p = (0.5 * (1.0 - A)).clamp(0.0, 1.0)
    return torch.where(t >= 0, p, 1.0 - p)


def p_values(prop: torch.Tensor, Ssum: torch.Tensor, Qsum: torch.Tensor, n: int) -> tuple:
    """ONE-SIDED Student's t in the proposal direction: z = S sign(prop) / sqrt(n s^2), s^2 = variance
    across the n verification micro-batches (n-1 degrees). Returns (p, z, S_dir) in float64."""
    if n < 2:
        raise ValueError(f"at least 2 verification micro-batches are needed (there are {n})")
    Sd = Ssum.to(torch.float64)
    Qd = Qsum.to(torch.float64)
    s_dir = Sd * torch.sign(prop.to(torch.float64))
    var = ((Qd - Sd * Sd / n) / (n - 1)).clamp_min(0.0)
    den = torch.sqrt(n * var)
    inf = torch.full_like(s_dir, math.inf)
    z = torch.where(den > 0, s_dir / den.clamp_min(1e-300),
                    torch.where(s_dir > 0, inf, torch.where(s_dir < 0, -inf, torch.zeros_like(s_dir))))
    return sf_t(z, n - 1), z, s_dir


def bh(p: torch.Tensor, tested: torch.Tensor, q: float) -> torch.Tensor:
    """Benjamini-Hochberg row by row (one row = one matrix). p (n, M) float64, `tested` (n, M) bool: the
    hypotheses (the groups with a candidate); m = tested per row. Returns the mask of rejections (findings)."""
    pp = torch.where(tested, p, torch.full_like(p, math.inf))
    ps, _ = torch.sort(pp, dim=1)
    M = pp.shape[1]
    k = torch.arange(1, M + 1, dtype=torch.float64, device=p.device)
    m = tested.sum(1).to(torch.float64).clamp_min(1.0)
    ok = ps <= k[None, :] * q / m[:, None]
    kmax = (ok.to(torch.float64) * k[None, :]).amax(1).to(torch.int64)
    cut = ps.gather(1, (kmax - 1).clamp_min(0).unsqueeze(1)).squeeze(1)
    cut = torch.where(kmax > 0, cut, torch.full_like(cut, -math.inf))
    return tested & (pp <= cut[:, None])


def nnz_groups(fam: S.FamilyData, e0: int, e1: int, piece: int = 8) -> torch.Tensor:
    """Non-zero trits per group (e1-e0, R, NG) int32, from the bytes (decode of scale_fields), in pieces of experts."""
    out = []
    for a in range(e0, e1, piece):
        b = min(e1, a + piece)
        t = FIELDS.trit_torch(fam.blocks[a:b].reshape(-1, fam.nb), fam.type_)
        out.append((t != 0).view(b - a, fam.R, fam.NG, fam.gs).sum(-1, dtype=torch.int32))
        del t
    return torch.cat(out)


def budget_experts(d: torch.Tensor, nnz: torch.Tensor, rho: float) -> torch.Tensor:
    """Trust-ratio budget per expert: rho^2 ||W_e||^2 = rho^2 sum_g d_g^2 nnz_g (float64, (n,)). The chosen jumps
    must have sum d_g^2 <= budget, i.e. ||Delta W_e|| / ||W_e|| <= rho (PV §4.2), guaranteed by construction."""
    ne = d.shape[0]
    d2 = d.to(torch.float64).reshape(ne, -1) ** 2
    return rho * rho * (d2 * nnz.to(torch.float64).reshape(ne, -1)).sum(1)


@dataclass
class Selection:
    """Chosen jumps for ONE family: indices (e, r, g), column within the group, direction, predicted gain (second
    order, in units of the step loss); and per expert: tested, BH findings, convenient (gain > 0 among the
    findings), budget (rho^2 ||W_e||^2), chosen, cost (sum d^2 of the chosen <= budget)."""
    e: torch.Tensor
    r: torch.Tensor
    g: torch.Tensor
    cand: torch.Tensor
    dir: torch.Tensor
    gain: torch.Tensor
    tested: torch.Tensor
    findings: torch.Tensor
    convenient: torch.Tensor
    budget: torch.Tensor
    chosen: torch.Tensor
    cost: torch.Tensor

    def __len__(self) -> int:
        return int(self.e.numel())

    def columns(self, gs: int) -> torch.Tensor:
        return self.g * gs + self.cand


def gain_second_order(S_: torch.Tensor, prop: torch.Tensor, H: torch.Tensor, d: torch.Tensor, n_loss: int,
                            n_verify: int, n_curvature: int, Q_: Optional[torch.Tensor] = None,
                            t_lcb: float = 0.0) -> torch.Tensor:
    """Predicted gain (decrease of the STEP loss) of a ONE-level jump on the candidate:
    g_step |d| - H_step d^2 / 2, g_step = (N/n_ver) S sign(prop) (the sum over the loss micro-batches, estimated
    from the verification ones only: unbiased estimate), H_step = (N/n_curv) H (sampled Gauss-Newton curvature, same
    tokens by construction of the loss). It is positive only if the local quadratic minimum lies beyond half a level:
    exactly rounding to the nearest level. With `Q_` and `t_lcb` > 0 the gradient is its LOWER confidence
    BOUND: S_dir - t_lcb sqrt(n s^2) (s^2 = variance across the verification micro-batches). float64."""
    if n_curvature < 1:
        raise ValueError("second-order criterion without curvature: curvature micro-batches (mode 3) are needed")
    N = float(n_loss)
    Sd = S_.to(torch.float64) * torch.sign(prop.to(torch.float64))
    if Q_ is not None and t_lcb > 0:
        n = float(n_verify)
        var = ((Q_.to(torch.float64) - S_.to(torch.float64) ** 2 / n) / (n - 1)).clamp_min(0.0)
        Sd = Sd - t_lcb * torch.sqrt(n * var)
    gp = (N / n_verify) * Sd
    Hp = (N / n_curvature) * H.to(torch.float64)
    ad = d.to(torch.float64).abs()
    return gp * ad - 0.5 * Hp * ad * ad


def t_gain(prop: torch.Tensor, S_: torch.Tensor, Q_: torch.Tensor, H: torch.Tensor, d: torch.Tensor,
               n_verify: int, n_curvature: int) -> torch.Tensor:
    """Student's t of the gain: t = (S_dir - threshold) / sqrt(n s^2), threshold = (n_ver / n_curv) H |d| / 2 (in units
    of the sum over the verification micro-batches); t > 0 <=> estimated second-order gain > 0. float64.
    (2026-09-27: extracted from `p_gain`, same operations; also used by --pv-accept gain with z_gain_min.)"""
    n = n_verify
    Sd = S_.to(torch.float64) * torch.sign(prop.to(torch.float64))
    threshold = 0.5 * (n / float(n_curvature)) * H.to(torch.float64) * d.to(torch.float64).abs()
    var = ((Q_.to(torch.float64) - S_.to(torch.float64) ** 2 / n) / (n - 1)).clamp_min(0.0)
    den = torch.sqrt(n * var)
    x = Sd - threshold
    inf = torch.full_like(x, math.inf)
    return torch.where(den > 0, x / den.clamp_min(1e-300), torch.where(x > 0, inf, -inf))


def p_gain(prop: torch.Tensor, S_: torch.Tensor, Q_: torch.Tensor, H: torch.Tensor, d: torch.Tensor,
               n_verify: int, n_curvature: int) -> torch.Tensor:
    """p-value (one-sided Student's t, n-1 degrees) of the null hypothesis "the jump does not reduce the step loss":
    gain > 0  <=>  S_dir > threshold = (n_ver / n_curv) H |d| / 2 (in units of the sum over the verification
    micro-batches). t = (S_dir - threshold) / sqrt(n s^2) (`t_gain`). Then BH per matrix: under the global null no jump
    with probability >= 1 - q; among the chosen jumps the fraction of useless ones is <= q on average."""
    return sf_t(t_gain(prop, S_, Q_, H, d, n_verify, n_curvature), n_verify - 1)


def select(buf: BufferV, fam: S.FamilyData, d_eff: torch.Tensor, n_verify: int, rho: float,
              cfg: ConfigV = ConfigV(), *, n_loss: int, n_curvature: int,
              d_ref: Optional[torch.Tensor] = None) -> Selection:
    """End-of-step selection for ONE family. `d_eff`: the CURRENT scale (E, R, NG), the one of the forward after the
    P step (exact fp16). Findings = BH on Student's t (guard against noise) in the proposal direction;
    convenient = findings with SECOND-ORDER gain > 0; then per expert in gain order as long as the
    sum of the chosen d^2 fits in the trust-ratio budget (prefix: no skipping over an interval). In pieces.
    2026-09-27 `cfg.accept` "gain": findings = tested with z > z_min in the proposal direction, convenient =
    findings with estimated gain > 0 and t of the gain > z_gain_min (no BH); the rest unchanged."""
    E, R, NG = fam.E, fam.R, fam.NG
    G = R * NG
    parts = {k: [] for k in ("e", "r", "g", "cand", "dir", "gain")}
    cont = {k: [] for k in ("tested", "findings", "convenient", "budget", "chosen", "cost")}
    for e0 in range(0, E, cfg.piece_experts):
        e1 = min(E, e0 + cfg.piece_experts)
        ne = e1 - e0
        pr = buf.prop[e0:e1]
        d = d_eff[e0:e1].to(torch.float64)
        tested = ((pr != 0) & (d != 0)).reshape(ne, G)
        if d_ref is not None:                    # accumulation window: exclude groups whose scale moved beyond the threshold
            dr = d_ref[e0:e1].to(torch.float64)
            frozen = (dr != 0) & ((d / torch.where(dr != 0, dr, torch.ones_like(dr)) - 1).abs() <= cfg.threshold_staleness)
            tested = tested & frozen.reshape(ne, G)
        p, z, s_dir = p_values(pr, buf.S[e0:e1], buf.Q[e0:e1], n_verify)
        if cfg.accept == "bh":
            scop = bh(p.reshape(ne, G), tested, cfg.q_fdr) & (s_dir.reshape(ne, G) > 0)
            gq = gain_second_order(buf.S[e0:e1], pr, buf.H[e0:e1], d, n_loss, n_verify, n_curvature)
            p_g = p_gain(pr, buf.S[e0:e1], buf.Q[e0:e1], buf.H[e0:e1], d, n_verify, n_curvature)
            conv = bh(p_g.reshape(ne, G), tested, cfg.q_gain) & (gq.reshape(ne, G) > 0)
        else:                                    # --pv-accept gain: estimated gain > 0, without BH
            scop = tested & (s_dir.reshape(ne, G) > 0) & (z.reshape(ne, G) > cfg.z_min)
            gq = gain_second_order(buf.S[e0:e1], pr, buf.H[e0:e1], d, n_loss, n_verify, n_curvature)
            conv = scop & (gq.reshape(ne, G) > 0)
            if cfg.z_gain_min > 0:
                t_g = t_gain(pr, buf.S[e0:e1], buf.Q[e0:e1], buf.H[e0:e1], d, n_verify, n_curvature)
                conv = conv & (t_g.reshape(ne, G) > cfg.z_gain_min)
        bud = budget_experts(d, nnz_groups(fam, e0, e1, cfg.piece_decode), rho)
        gu = torch.where(conv, gq.reshape(ne, G), torch.full((ne, G), -math.inf, dtype=torch.float64,
                                                             device=d.device))
        val, idx = torch.sort(gu, dim=1, descending=True, stable=True)
        cost = (d.reshape(ne, G) ** 2).gather(1, idx)
        fin = torch.isfinite(val)
        cum = torch.cumsum(torch.where(fin, cost, torch.zeros_like(cost)), dim=1)
        keep = fin & (cum <= bud[:, None])
        ee, jj = torch.nonzero(keep, as_tuple=True)
        gi = idx[ee, jj]
        e_abs, r_, g_ = ee + e0, gi // NG, gi % NG
        parts["e"].append(e_abs)
        parts["r"].append(r_)
        parts["g"].append(g_)
        parts["cand"].append(buf.cand[e_abs, r_, g_].to(torch.int64))
        sp = torch.sign(buf.prop[e_abs, r_, g_].to(torch.float64) * d_eff[e_abs, r_, g_].to(torch.float64))
        parts["dir"].append((-sp).to(torch.int8))
        parts["gain"].append(val[ee, jj])
        cont["tested"].append(tested.sum(1))
        cont["findings"].append(scop.sum(1))
        cont["convenient"].append(conv.sum(1))
        cont["budget"].append(bud)
        cont["chosen"].append(keep.sum(1))
        cont["cost"].append(torch.where(keep, cost, torch.zeros_like(cost)).sum(1))
    dev = buf.prop.device
    empty = {"e": torch.int64, "r": torch.int64, "g": torch.int64, "cand": torch.int64, "dir": torch.int8,
             "gain": torch.float64}
    out = {k: (torch.cat(v) if v else torch.zeros(0, dtype=empty[k], device=dev)) for k, v in parts.items()}
    return Selection(**out, **{k: torch.cat(v) for k, v in cont.items()})


def _sf_t_per_nu(t: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
    """sf_t with per-element degrees of freedom (nu int64 >= 1): one computation per distinct value."""
    p = torch.empty_like(t, dtype=torch.float64)
    for v in torch.unique(nu).tolist():
        mk = nu == v
        p[mk] = sf_t(t[mk], int(v))
    return p


def select_bop(z: BopState, buf: BufferV, fam: S.FamilyData, d_eff: torch.Tensor, n_verify: int, rho: float,
                  cfg: ConfigV, bop: ConfigBop, *, n_loss: int) -> Selection:
    """V-Bop selection for ONE family (after `VState.update_bop`). Per group with a candidate (age >= 1, H > 0):
    m^ = m / c, v^ = v / c (c = 1 - (1-gamma)^age), n_eff = n_verify / sum w^2, standard error sqrt(v^ / n_eff),
    z = m^ direction / se (Student's t with nu = round((n_verify - 1) / sum w^2) degrees, cap nu_max) -> BH (q_fdr);
    step gain N (m^ direction |d| - H d^2 / 2) > 0 with its test (m^ direction - H |d| / 2) / se -> BH (q_gain);
    then guaranteed rho budget per expert as in `select`. With gamma = 1 it is the PV test (same z, same degrees).
    2026-09-27 `cfg.accept` "gain": findings = tested with m^ direction > 0 and z > z_min, convenient = findings with
    gain N (m^ direction |d| - H d^2 / 2) > 0 and (m^ direction - H|d|/2) / se > z_gain_min (no BH); budget,
    ordering and the rest unchanged."""
    E, R, NG = fam.E, fam.R, fam.NG
    G = R * NG
    gam = float(bop.gamma)
    N = float(n_loss)
    parts = {k: [] for k in ("e", "r", "g", "cand", "dir", "gain")}
    cont = {k: [] for k in ("tested", "findings", "convenient", "budget", "chosen", "cost")}
    for e0 in range(0, E, cfg.piece_experts):
        e1 = min(E, e0 + cfg.piece_experts)
        ne = e1 - e0
        d = d_eff[e0:e1].to(torch.float64)
        eta, direction = z.eta_direction(e0, e1)
        c, sw2 = _correction_eta(gam, eta)
        m_dir = _decode16(z.m[e0:e1], z.sm[e0:e1]) / c * direction
        v_hat = _decode16(z.r[e0:e1], z.sr[e0:e1]) ** 2 / c
        Hh = _decode16(z.H[e0:e1], z.sH[e0:e1])
        tested = ((eta >= 1) & (Hh > 0) & (d != 0)).reshape(ne, G)
        se = torch.sqrt(v_hat * sw2 / n_verify)
        nu = torch.round((n_verify - 1) / sw2).clamp(1, bop.nu_max).to(torch.int64)
        ad = d.abs()

        def z_of(x):
            inf = torch.full_like(x, math.inf)
            return torch.where(se > 0, x / se.clamp_min(1e-300),
                               torch.where(x > 0, inf, torch.where(x < 0, -inf, torch.zeros_like(x))))
        if cfg.accept == "bh":
            p = _sf_t_per_nu(z_of(m_dir), nu)
            scop = bh(p.reshape(ne, G), tested, cfg.q_fdr) & (m_dir.reshape(ne, G) > 0)
            gq = N * (m_dir * ad - 0.5 * Hh * ad * ad)
            p_g = _sf_t_per_nu(z_of(m_dir - 0.5 * Hh * ad), nu)
            conv = scop & bh(p_g.reshape(ne, G), tested, cfg.q_gain) & (gq.reshape(ne, G) > 0)
        else:                                    # --pv-accept gain: estimated gain > 0, without BH
            scop = tested & (m_dir.reshape(ne, G) > 0) & (z_of(m_dir).reshape(ne, G) > cfg.z_min)
            gq = N * (m_dir * ad - 0.5 * Hh * ad * ad)
            conv = scop & (gq.reshape(ne, G) > 0)
            if cfg.z_gain_min > 0:
                conv = conv & (z_of(m_dir - 0.5 * Hh * ad).reshape(ne, G) > cfg.z_gain_min)
        bud = budget_experts(d, nnz_groups(fam, e0, e1, cfg.piece_decode), rho)
        gu = torch.where(conv, gq.reshape(ne, G), torch.full((ne, G), -math.inf, dtype=torch.float64,
                                                             device=d.device))
        val, idx = torch.sort(gu, dim=1, descending=True, stable=True)
        cost = (d.reshape(ne, G) ** 2).gather(1, idx)
        fin = torch.isfinite(val)
        cum = torch.cumsum(torch.where(fin, cost, torch.zeros_like(cost)), dim=1)
        keep = fin & (cum <= bud[:, None])
        ee, jj = torch.nonzero(keep, as_tuple=True)
        gi = idx[ee, jj]
        e_abs, r_, g_ = ee + e0, gi // NG, gi % NG
        parts["e"].append(e_abs)
        parts["r"].append(r_)
        parts["g"].append(g_)
        parts["cand"].append(z.cand[e_abs, r_, g_].to(torch.int64))
        sp = direction[ee, r_, g_] * torch.sign(d_eff[e_abs, r_, g_].to(torch.float64))
        parts["dir"].append((-sp).to(torch.int8))
        parts["gain"].append(val[ee, jj])
        cont["tested"].append(tested.sum(1))
        cont["findings"].append(scop.sum(1))
        cont["convenient"].append(conv.sum(1))
        cont["budget"].append(bud)
        cont["chosen"].append(keep.sum(1))
        cont["cost"].append(torch.where(keep, cost, torch.zeros_like(cost)).sum(1))
    dev = buf.prop.device
    empty = {"e": torch.int64, "r": torch.int64, "g": torch.int64, "cand": torch.int64, "dir": torch.int8,
             "gain": torch.float64}
    out = {k: (torch.cat(v) if v else torch.zeros(0, dtype=empty[k], device=dev)) for k, v in parts.items()}
    return Selection(**out, **{k: torch.cat(v) for k, v in cont.items()})


# ═══════════════════════════════════ 6. jumps on the bytes and registry ═══════════════════════════════════
def position_t(c: torch.Tensor, type_: str) -> tuple:
    """Column c of a row -> (block within the row, byte within the block, digit n). Inverse of `_trit_from_byte`
    (vectorised; `e2e_pv_misure.posizione` is the scalar version, T2 compares them over all columns)."""
    nb, qk, _, _ = CS.LAYOUT[type_]
    c = c.to(torch.int64)
    blk, p = c // qk, c % qk
    if type_ == "TQ1_0R_1024":
        qu, j = p // 256, p % 256
        qs0, qh0 = qu * 48, 192 + qu * 4
    else:
        j = p
        qs0, qh0 = torch.zeros_like(p), torch.full_like(p, 48)
    in1, in2 = j < 160, j < 240
    j2, j3 = (j - 160).clamp_min(0), (j - 240).clamp_min(0)
    byte = torch.where(in1, qs0 + j % 32, torch.where(in2, qs0 + 32 + j2 % 16, qh0 + j3 % 4))
    n = torch.where(in1, j // 32, torch.where(in2, j2 // 16, j3 // 4))
    return blk, byte, n


def _digit(v: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    """Digit n (0 = most significant) of byte v in {0,1,2}: arithmetic of dequantize_row_tq1_0 (without the -1)."""
    pw = torch.tensor(P3, dtype=torch.int32, device=v.device)[n]
    return (((v.to(torch.int32) * pw) & 255) * 3) >> 8


def _in_qh(ab: torch.Tensor, type_: str) -> torch.Tensor:
    """ab = byte within the row -> True if it lies in a qh region (4 digits + null digit)."""
    nb = CS.LAYOUT[type_][0]
    bb = ab % nb
    if type_ == "TQ1_0R_1024":
        return (bb >= 192) & (bb < 208)
    return (bb >= 48) & (bb < 52)


def _value_q(v: torch.Tensor, qh: torch.Tensor) -> torch.Tensor:
    """q of the byte = sum_n digit_n 3^(4-n) over the digits the decoder reads (5 in qs, 4 in qh)."""
    q = torch.zeros_like(v, dtype=torch.int32)
    for n in range(5):
        cn = _digit(v, torch.full_like(v, n, dtype=torch.int64))
        if n == 4:
            cn = torch.where(qh, torch.zeros_like(cn), cn)
        q = q + cn * (3 ** (4 - n))
    return q


def skip_byte(B: torch.Tensor, type_: str, e: torch.Tensor, r: torch.Tensor, c: torch.Tensor,
               t_expected: torch.Tensor, t_new: torch.Tensor) -> int:
    """Writes in place into the bytes B (E, R, NBR) uint8 the trits t_new at positions (e, r, c), checking that the
    CURRENT ones are t_expected (otherwise raises, nothing is written). A single digit change per position, of one
    level; several positions in the same byte add up on q. Returns the number of rewritten bytes."""
    if e.numel() == 0:
        return 0
    assert B.dtype == torch.uint8 and B.dim() == 3 and B.is_contiguous()
    E, R, NBR = B.shape
    nb, qk = CS.LAYOUT[type_][0], CS.LAYOUT[type_][1]
    C = NBR // nb * qk
    dev = B.device
    e, r, c = e.to(dev, torch.int64), r.to(dev, torch.int64), c.to(dev, torch.int64)
    ta, tn = t_expected.to(dev, torch.int32), t_new.to(dev, torch.int32)
    if bool(((e < 0) | (e >= E) | (r < 0) | (r >= R) | (c < 0) | (c >= C)).any()):
        raise IndexError("position outside the tensor")
    pos = (e * R + r) * C + c                                          # code position (uniqueness)
    if torch.unique(pos).numel() != pos.numel():
        raise ValueError("two jumps on the same code")
    dt = tn - ta
    if bool((dt.abs() != 1).any()) or bool((tn.abs() > 1).any()):
        raise ValueError("jump not of ONE level within {-1,0,1}")
    blk, byte, n = position_t(c, type_)
    ab = blk * nb + byte
    flat = (e * R + r) * NBR + ab
    Bf = B.view(-1)
    now = _digit(Bf[flat], n) - 1
    if not torch.equal(now, ta):
        raise RuntimeError(f"current trits differ from the expected ones at {int((now != ta).sum())} positions: registry "
                           "and bytes are not aligned")
    u, inv = torch.unique(flat, return_inverse=True)
    pw = torch.tensor([81, 27, 9, 3, 1], dtype=torch.int32, device=dev)[n]
    dq = torch.zeros(u.numel(), dtype=torch.int32, device=dev).index_add_(0, inv, dt * pw)
    vu = Bf[u].to(torch.int32)
    qh = _in_qh(u % NBR, type_)
    q = _value_q(vu, qh) + dq
    if bool(((q < 0) | (q > 242) | (qh & (q % 3 != 0)) | (qh & (q > 240))).any()):
        raise RuntimeError("q out of range after the jumps")
    Bf[u] = (((q * 256 + 242) // 243) & 255).to(torch.uint8)
    return int(u.numel())


REG_DTYPE = np.dtype([("L", "u1"), ("f", "u1"), ("e", "u1"), ("r", "<u2"), ("c", "<u2"),
                      ("t_old", "i1"), ("t_new", "i1")])
assert REG_DTYPE.itemsize == 9


@dataclass
class Registry:
    """Jumps in application order: 9 bytes per jump (L, f, e, r, c, t_old, t_new)."""
    rows: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=REG_DTYPE))

    def __len__(self) -> int:
        return int(self.rows.shape[0])

    @staticmethod
    def merge(regs: list) -> "Registry":
        """Concatenation for READING (counts, export check: first t_old / last t_new per position).
        ⛔ Not to be APPLIED: the same code may jump in different steps and `skip_byte` rejects duplicates within a
        single registry (2026-09-26 revision). Replay, cancellation and export take the LIST of registries, in order."""
        return Registry(np.concatenate([r.rows for r in regs]) if regs else np.zeros(0, dtype=REG_DTYPE))

    def inverse(self) -> "Registry":
        inv = self.rows[::-1].copy()
        inv["t_old"], inv["t_new"] = self.rows[::-1]["t_new"], self.rows[::-1]["t_old"]
        return Registry(inv)

    def keys(self) -> list:
        return sorted({(int(a), int(b)) for a, b in zip(self.rows["L"], self.rows["f"])})

    def di(self, L: int, f: int) -> np.ndarray:
        return self.rows[(self.rows["L"] == L) & (self.rows["f"] == f)]

    def save(self, p: Path) -> str:
        """ATOMIC write (2026-09-26 revision: a crash halfway left 256 B instead of the valid registry): temporary
        file alongside, fsync, rename, fsync of the folder. Returns the sha256 of the file."""
        buf = io.BytesIO()
        np.save(buf, self.rows, allow_pickle=False)
        return _write_atomic(Path(p), buf.getvalue())

    @staticmethod
    def save_list(p: Path, registers: list) -> str:
        """The LIST of registries (one per step) in one atomic .npz: concatenated rows + end of each registry."""
        end = np.cumsum([len(r) for r in registers], dtype=np.int64) if registers else np.zeros(0, dtype=np.int64)
        rows = np.concatenate([r.rows for r in registers]) if registers else np.zeros(0, dtype=REG_DTYPE)
        buf = io.BytesIO()
        np.savez(buf, rows=rows, end=end)
        return _write_atomic(Path(p), buf.getvalue())

    @staticmethod
    def load_list(p: Path, sha256: Optional[str] = None) -> list:
        p = Path(p)
        data = p.read_bytes()
        if sha256 is not None and hashlib.sha256(data).hexdigest() != sha256:
            raise RuntimeError(f"{p}: sha256 differs from the one recorded for the resume")
        with np.load(io.BytesIO(data), allow_pickle=False) as z:
            rows, end = z["rows"], z["end"]
        if rows.dtype != REG_DTYPE:
            raise TypeError(f"{p}: dtype {rows.dtype}")
        starts = np.concatenate([[0], end[:-1]]) if end.size else np.zeros(0, dtype=np.int64)
        return [Registry(rows[a:b].copy()) for a, b in zip(starts, end)]

    @staticmethod
    def load(p: Path, sha256: Optional[str] = None) -> "Registry":
        p = Path(p)
        if sha256 is not None and hashlib.sha256(p.read_bytes()).hexdigest() != sha256:
            raise RuntimeError(f"{p}: sha256 differs from the one recorded for the resume")
        r = np.load(p, allow_pickle=False)
        if r.dtype != REG_DTYPE:
            raise TypeError(f"{p}: dtype {r.dtype}")
        return Registry(r)


def _write_atomic(p: Path, data: bytes) -> str:
    tmp = p.with_name(p.name + ".partial")
    try:
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, p)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    fd = os.open(str(p.parent), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return hashlib.sha256(data).hexdigest()


def family_writable(fam: S.FamilyData) -> S.FamilyData:
    """On CPU: copy of the bytes (the bytes of `family_from_tensor` may be a view of the read-only memmap).
    Marks the family as writable: `apply_skips` rejects the others on CPU."""
    fam.blocks = fam.blocks.clone()
    fam._pv_writable = True
    return fam


def _check_writable(fam: S.FamilyData) -> None:
    if fam.blocks.device.type == "cpu" and not getattr(fam, "_pv_writable", False):
        raise PermissionError("bytes on CPU not passed through `family_writable`: they could be the read-only "
                              "GGUF memmap")


def skip_registry(fam: S.FamilyData, rows: np.ndarray) -> int:
    """Applies registry rows (already filtered for this family) to the bytes: t_old -> t_new."""
    _check_writable(fam)
    if rows.shape[0] == 0:
        return 0
    f = lambda k, dt: torch.from_numpy(rows[k].astype(dt))  # noqa: E731
    return skip_byte(fam.blocks, fam.type_, f("e", np.int64), f("r", np.int64), f("c", np.int64),
                      f("t_old", np.int32), f("t_new", np.int32))


def apply_skips(fam: S.FamilyData, sel: Selection, L: Optional[int] = None) -> Registry:
    """Applies the jumps of a Selection to the family's bytes; returns the Registry. t_old is read from the bytes."""
    _check_writable(fam)
    L = int(fam.layer) if L is None else int(L)
    if len(sel) == 0:
        return Registry()
    rows = rows_skips(fam, sel, L)
    skip_registry(fam, rows)
    return Registry(rows)


def rows_skips(fam: S.FamilyData, sel: Selection, L: Optional[int] = None) -> np.ndarray:
    """Registry rows (REG_DTYPE) of the jumps of a Selection, WITHOUT writing them: t_old read from the current bytes,
    t_new = t_old + dir (2026-09-27 v-interaction-b branch: the doses apply subsets of these rows). Same
    operations that `apply_skips` performed before writing (apply_skips = rows_skips + skip_registry)."""
    L = int(fam.layer) if L is None else int(L)
    if len(sel) == 0:
        return np.zeros(0, dtype=REG_DTYPE)
    c = sel.columns(fam.gs)
    blk, byte, n = position_t(c.to(fam.blocks.device), fam.type_)
    flat = (sel.e.to(fam.blocks.device) * fam.R + sel.r.to(fam.blocks.device)) * fam.NBR + blk * fam.nb + byte
    tv = (_digit(fam.blocks.view(-1)[flat], n) - 1).to(torch.int32)
    tn = tv + sel.dir.to(fam.blocks.device, torch.int32)
    if bool((tn.abs() > 1).any()):
        raise RuntimeError("selection with a jump outside {-1,0,1}: the proposal feasibility check has failed")
    rows = np.zeros(len(sel), dtype=REG_DTYPE)
    rows["L"], rows["f"] = L, FAMILIES_IDX[fam.name]
    rows["e"], rows["r"], rows["c"] = sel.e.cpu().numpy(), sel.r.cpu().numpy(), c.cpu().numpy()
    rows["t_old"], rows["t_new"] = tv.cpu().numpy(), tn.cpu().numpy()
    return rows


def cancel(families: dict, reg: Registry) -> None:
    """Cancels a registry (reverse order, checking that the bytes carry the t_new values)."""
    replay(families, [reg.inverse()])


def replay(families: dict, registers: list) -> None:
    """Applies the registries in order. families: {(L, name): FamilyData}. Resume = GGUF bytes + replay."""
    for reg in registers:
        for (L, fi) in reg.keys():
            skip_registry(families[(L, NAMES_IDX[fi])], reg.di(L, fi))


# ═══════════════════════════════════ 7. export ═══════════════════════════════════
def tensor_name(L: int, fi: int) -> str:
    return f"blk.{L}.{S.family_tensor_name(NAMES_IDX[fi])}.weight"


def _byte_3d(raw) -> torch.Tensor:
    """Bytes of a file tensor as (E, R, NBR): a 2D dense tensor (ternary dense families) is the family of ONE expert.
    View (no copy): the writes of `skip_byte` reach the memmap."""
    B = torch.from_numpy(raw)
    return B.unsqueeze(0) if B.dim() == 2 else B


def patch_codes(dst: Path, registers: list, name: Callable = tensor_name) -> dict:
    """Writes the registries into the code bytes of `dst` (a COPY of the GGUF: never the source), memmap r+, same
    arithmetic as `skip_byte`, ONE registry after another (repeated jumps of the same code across steps allowed).
    Returns {tensor: rewritten bytes}. Exporters use `export_codes` (.partial copy, verification, rename)."""
    from gguf import GGUFReader
    r = GGUFReader(str(dst), mode="r+")
    tens = {t.name: t for t in r.tensors}
    written: dict = {}
    for reg in registers:
        for (L, fi) in reg.keys():
            t = tens[name(L, fi)]
            raw = t.data                                          # r+ view (E, R, NBR) of the file ((R, NBR) if dense)
            B = _byte_3d(raw)
            rows = reg.di(L, fi)
            f = lambda k, dt: torch.from_numpy(rows[k].astype(dt))  # noqa: E731
            written[t.name] = written.get(t.name, 0) + skip_byte(
                B, t.tensor_type.name, f("e", np.int64), f("r", np.int64), f("c", np.int64),
                f("t_old", np.int32), f("t_new", np.int32))
    r.data.flush()
    del r, tens
    return written


def verify_patch_codes(src: Path, dst: Path, registers: list, name: Callable = tensor_name,
                          piece: int = 1 << 26) -> dict:
    """Export checks, re-reading the FILES (not the in-memory tensors):
    (1) same size; (2) comparison in pieces of `piece` bytes: every differing byte lies among the code bytes of the
    registry (never a d field, never another tensor, never the header); (3) in the touched blocks, trits of dst ==
    trits of src with the FINAL registry state (for each position: the first t_old must be that of src,
    the last t_new that of dst), all other trits identical; (4) d fields of the touched blocks identical and
    llama.cpp C code on the touched blocks of dst == trit x d (de-rotated for TQ1_0R_1024), bit-exact."""
    from gguf import GGUFReader
    from fucina.student import export as EX                                         # read-only: _deq_c (C code)
    from fucina.ternary.tq1_0r_1024_codec import S1024
    src, dst = Path(src), Path(dst)
    if src.stat().st_size != dst.stat().st_size:
        raise RuntimeError("sizes differ")
    rs, rd = GGUFReader(str(src)), GGUFReader(str(dst))
    ts, td = {t.name: t for t in rs.tensors}, {t.name: t for t in rd.tensors}
    reg = Registry.merge(registers)
    permissions, blocks_c, trit_changed = [], 0, 0
    for (L, fi) in reg.keys():
        rows = reg.di(L, fi)
        t, u = ts[name(L, fi)], td[name(L, fi)]
        type_ = t.tensor_type.name
        nb, qk, off, k = CS.LAYOUT[type_]
        shape = np.asarray(t.data).shape
        E, R, NBR = (1,) + tuple(shape) if len(shape) == 2 else shape     # 2026-09-28: 2D dense = one expert
        C, bpr = NBR // nb * qk, NBR // nb
        e_, r_, c_ = (rows[x].astype(np.int64) for x in ("e", "r", "c"))
        blk, byte, _ = (x.numpy() for x in position_t(torch.from_numpy(c_), type_))
        permissions.append(int(t.data_offset) + (e_ * R + r_) * NBR + blk * nb + byte)
        key_pos = (e_ * R + r_) * C + c_
        uk, i_first = np.unique(key_pos, return_index=True)
        _, i_rev = np.unique(key_pos[::-1], return_index=True)
        i_last = len(key_pos) - 1 - i_rev
        gb = (e_ * R + r_) * bpr + blk                                # block within the tensor
        bi = np.unique(gb)
        Bs = np.ascontiguousarray(np.asarray(t.data).reshape(-1, nb)[bi])
        Bd = np.ascontiguousarray(np.asarray(u.data).reshape(-1, nb)[bi])
        Ts = CS.trit_from_blocks(torch.from_numpy(Bs), type_)
        Td = CS.trit_from_blocks(torch.from_numpy(Bd), type_)
        row_b = torch.from_numpy(np.searchsorted(bi, gb[i_first]))
        col_b = torch.from_numpy(c_[i_first] % qk)
        if not torch.equal(Ts[row_b, col_b].to(torch.int64), torch.from_numpy(rows["t_old"][i_first]
                                                                                  .astype(np.int64))):
            raise RuntimeError(f"{name(L, fi)}: SOURCE trits != first t_old of the registry")
        expected = Ts.clone()
        expected[row_b, col_b] = torch.from_numpy(rows["t_new"][i_last].astype(np.int8))
        if not torch.equal(Td, expected):
            raise RuntimeError(f"{name(L, fi)}: trits of the touched blocks != source + registry "
                               f"({int((Td != expected).sum())} different)")
        if not np.array_equal(Bs[:, off:off + 2 * k], Bd[:, off:off + 2 * k]):
            raise RuntimeError(f"{name(L, fi)}: d fields changed")
        dd = torch.from_numpy(Bd[:, off:off + 2 * k].copy().view(np.float16).astype(np.float32))
        W = CS.weights_from_trit_scale(Td, dd, type_, torch.from_numpy(S1024) if type_ != "TQ1_0" else None).numpy()
        Wc = EX._deq_c(Bd, type_)
        if not np.array_equal(W, Wc):
            raise RuntimeError(f"{name(L, fi)}: llama.cpp C code != new trit x d (max {np.abs(W - Wc).max()})")
        blocks_c += int(bi.size)
        trit_changed += int((Ts != Td).sum())
    del rs, rd, ts, td
    permissions = np.unique(np.concatenate(permissions)) if permissions else np.zeros(0, dtype=np.int64)
    a = np.memmap(src, dtype=np.uint8, mode="r")
    b = np.memmap(dst, dtype=np.uint8, mode="r")
    different = 0
    for i0 in range(0, a.shape[0], piece):
        i1 = min(a.shape[0], i0 + piece)
        dv = np.flatnonzero(np.asarray(a[i0:i1]) != np.asarray(b[i0:i1])) + i0
        if dv.size:
            outside = dv[~np.isin(dv, permissions)]
            if outside.size:
                raise RuntimeError(f"{outside.size} differing bytes OUTSIDE the registry bytes (first {int(outside[0])})")
            different += int(dv.size)
    del a, b
    return {"byte_different": different, "byte_permissions": int(permissions.size), "blocks_code_c": blocks_c,
            "trit_changed": trit_changed}


def export_codes(scaled: Path, dst: Path, registers: list, name: Callable = tensor_name) -> dict:
    """Export of the CODES on top of a GGUF already exported with the checkpoint scales (`e2e_export`, which verifies
    the scales bit-exact and decodes the source trits, unchanged): copy to .partial, `patch_codes` registry by
    registry, `verify_patch_codes` against `scaled` (d fields identical, bytes differing only where the registries
    say, final trits = first t_old / last t_new, llama.cpp C code == trit x d), then rename. Never on the
    source; an error leaves only the .partial (deleted)."""
    scaled, dst = Path(scaled), Path(dst)
    if dst.resolve() == scaled.resolve():
        raise ValueError("export_codes: the destination is the source")
    tmp = dst.with_name(dst.name + ".partial")
    shutil.copyfile(scaled, tmp)
    try:
        written = patch_codes(tmp, registers, name)
        ver = verify_patch_codes(scaled, tmp, registers, name)
        with open(tmp, "rb+") as fh:
            os.fsync(fh.fileno())
        os.replace(tmp, dst)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return {"byte_written": written, **ver, "registers": len(registers), "skips": sum(len(r) for r in registers)}


def dense_of_registers(registers: list) -> tuple:
    """2026-09-28: ternary dense families that appear in the registries (f >= 3), canonical order."""
    fi = set()
    for r in registers:
        fi.update(int(x) for x in np.unique(r.rows["f"]) if int(x) >= 3)
    return S.families_dense([NAMES_IDX[i] for i in sorted(fi)])


def reload_codes(scaled: Path, dst: Path, registers: list, layers: list, token: int = 64,
                    dense_ternary: Optional[tuple] = None) -> dict:
    """RELOAD check of the final file (2026-09-26 revision: `reload` of e2e_export rebuilds from source +
    theta, without replay): for each layer, student from the `scaled` file + replay of the registries on its bytes vs
    student from the `dst` file: identical code bytes and layer output identical BIT-EXACT (CPU, torch engine).
    2026-09-28 `dense_ternary` (None = those of the registries, `dense_of_registers`): both students with the same
    single-expert dense families, so jumps on the dense families are replayed and compared like the expert ones."""
    dense_ = dense_of_registers(registers) if dense_ternary is None else S.families_dense(dense_ternary)
    kw = {"dense_ternary": dense_} if dense_ else {}
    outcomes = {}
    for L in layers:
        sa = S.Student(gguf=Path(scaled), layers=[L], recipe="B", engine_name="torch", device="cpu",
                        log=lambda *_: None, **kw)
        sb = S.Student(gguf=Path(dst), layers=[L], recipe="B", engine_name="torch", device="cpu",
                        log=lambda *_: None, **kw)
        fams = {}
        for n, f in sa.experts(L).fam.items():
            fams[(L, n)] = family_writable(f)
        regs_L = [Registry(r.rows[r.rows["L"] == L]) for r in registers]
        replay(fams, [r for r in regs_L if len(r)])
        byte_ok = all(torch.equal(sa.experts(L).fam[n].blocks, sb.experts(L).fam[n].blocks) for n in fams_n(sa, L))
        g = torch.Generator().manual_seed(20260927 + L)
        x = (torch.randn(1, token, sa.cfg.hidden_size, generator=g) * 0.5).to(sa.recipe.dtype_residual)
        with torch.no_grad():
            ya = sa.forward_(torch.zeros(1, token, dtype=torch.long), x0=x)["xL"]
            yb = sb.forward_(torch.zeros(1, token, dtype=torch.long), x0=x)["xL"]
        outcomes[L] = {"byte_codes_equal": byte_ok, "output_bit_exact": bool(torch.equal(ya, yb)),
                    "skips_in_layer": sum(len(r) for r in regs_L)}
        del sa, sb
        if not (outcomes[L]["byte_codes_equal"] and outcomes[L]["output_bit_exact"]):
            raise RuntimeError(f"reload_codes layer {L} FAILED: {outcomes[L]}")
    return outcomes


def fams_n(st, L: int) -> list:
    return list(st.experts(L).fam.keys())


# ═══════════════════════════════════ 8. trust region ═══════════════════════════════════
class TrustRegion:
    """Probe of design §3: the jumps are kept only if the loss on the probe (a never-seen micro-batch) does NOT
    get worse; otherwise they are cancelled from the registry and rho is halved. The first len(trial) steps try the
    rho values of `trial` (calibration) and then the one that lowered the loss the most is kept; if none lowered it,
    min(trial)/2.
    2026-09-27 `adaptive` (--pv-rho-adaptive, TECNICHE_E2E_PV.md §19): true trust region. r = actual decrease of the
    probe / predicted decrease (`gain_predicted`, same units as the probe); r > 0.75 -> rho x 2 (cap rho_max),
    r < 0.25 -> rho / 2 (floor rho_min), otherwise unchanged; without jumps (r undefined) unchanged. The jumps are
    still cancelled when the probe rises (r < 0 -> rho / 2). Off: identical to before."""

    def __init__(self, trial=(0.01, 0.03, 0.1), rho_min: float = 1e-4, adaptive: bool = False,
                 rho_max: float = 0.1):
        self.trial = list(trial)
        self.rho: Optional[float] = None
        self.rho_min = rho_min
        self.adaptive = bool(adaptive)
        self.rho_max = float(rho_max)
        self.history: list = []

    def rho_step(self) -> float:
        n = len(self.history)
        return self.trial[n] if n < len(self.trial) else float(self.rho)

    def _adapt(self, rho: float, r: Optional[float], keeps: bool) -> float:
        if r is None:
            return rho if keeps else max(self.rho_min, rho / 2)
        if r > 0.75:
            return max(rho, min(self.rho_max, 2 * rho))
        if r < 0.25:
            return max(self.rho_min, rho / 2)
        return rho

    def outcome(self, rho: float, loss_before: float, loss_after: float, n_skips: int,
              gain_predicted: Optional[float] = None) -> bool:
        keeps = loss_after <= loss_before
        entry = {"rho": rho, "before": loss_before, "after": loss_after, "skips": n_skips,
                "delta": loss_after - loss_before, "kept": keeps}
        r = None
        if self.adaptive:
            if n_skips > 0 and gain_predicted is not None and gain_predicted > 0:
                r = (loss_before - loss_after) / gain_predicted
            entry["r"] = r
        self.history.append(entry)
        n = len(self.history)
        if self.adaptive and n >= len(self.trial) and (len(self.trial) == 1 or n > len(self.trial)):
            self.rho = self._adapt(rho, r, keeps)
            entry["rho_after"] = self.rho
        elif n == len(self.trial):
            good = [s for s in self.history if s["kept"] and s["delta"] < 0]
            self.rho = min(good, key=lambda s: s["delta"])["rho"] if good else min(self.trial) / 2
        elif n > len(self.trial) and not keeps:
            self.rho = max(self.rho_min, self.rho / 2)
        return keeps

    def outcome_doses(self, rho: float, loss_before: float, loss_after: float, n_skips: int, keeps: bool,
                   partial: bool, gain_predicted: Optional[float] = None) -> bool:
        """2026-09-27 v-interaction-b branch (--pv-sonda-doses): the decision (`keeps`) is taken by the dose probe with
        confirmation, not by the before/after comparison here. Rho rules, as in `outcome` except:
          - `keeps` and PARTIAL dose (fewer jumps kept than chosen, by dose or recenter): rho does NOT decrease (the dose
            has already separated the interacting jumps; halving rho would starve V without fixing the cause, §32.4);
            with adaptive rho it can still increase (r > 0.75), i.e. rho = max(rho, rule(r));
          - `keeps` and full dose: previous rule (adaptive: r; non-adaptive: unchanged);
          - not `keeps` (no dose improves probe A, or the second probe does not confirm): rho / 2 as today.
        r = actual decrease / predicted decrease of the KEPT jumps (`gain_predicted` of the kept ones only), measured by
        the caller on the confirmation probe (not on the one that chose the dose: no winner's curse)."""
        entry = {"rho": rho, "before": loss_before, "after": loss_after, "skips": n_skips,
                "delta": loss_after - loss_before, "kept": bool(keeps), "dose_partial": bool(partial)}
        r = None
        if self.adaptive:
            if keeps and n_skips > 0 and gain_predicted is not None and gain_predicted > 0:
                r = (loss_before - loss_after) / gain_predicted
            entry["r"] = r
        self.history.append(entry)
        n = len(self.history)
        if self.adaptive and n >= len(self.trial) and (len(self.trial) == 1 or n > len(self.trial)):
            new = self._adapt(rho, r, keeps)
            self.rho = max(rho, new) if (keeps and partial) else new
            entry["rho_after"] = self.rho
        elif n == len(self.trial):
            good = [s for s in self.history if s["kept"] and s["delta"] < 0]
            self.rho = min(good, key=lambda s: s["delta"])["rho"] if good else min(self.trial) / 2
        elif n > len(self.trial) and not keeps:
            self.rho = max(self.rho_min, self.rho / 2)
        return bool(keeps)


# ═══════════════════════ 8b. dose probe and recenter (2026-09-27 v-interaction-b branch, TECNICHE_E2E_PV.md §32) ═══════════════════════
FRACTIONS_DOSES = (1 / 256, 1 / 64, 1 / 16, 1 / 4, 1.0)
# 2026-09-29 midpoint branch: granularity of the primitives. family = (layer, family); type_ = family across all layers;
# layer = all families of one layer; expert = (layer, family, expert). The cost is ONE probe backward per priced
# primitive (at most pm_max) and per functional.
PRIMITIVE_PM = ("family", "type_", "layer", "expert")


def midpoint_price(dw: torch.Tensor, g0: torch.Tensor, g1: torch.Tensor) -> torch.Tensor:
    """Price of each component of a move at the midpoint, over the INTEGER states (trapezoidal rule):
    v = 1/2 dw (g0 + g1), g0 = grad F in the start state, g1 = grad F in the state with the move applied. At constant
    curvature (quadratic F) grad F(q0 + d/2) = (grad F(q0) + grad F(q0 + d)) / 2 exactly, hence sum v = F(q0 + d) -
    F(q0) EXACTLY (arXiv 2609.09867 Prop. 2; trapezoidal error <= L_H |d|^3 / 12 in general). float64."""
    return 0.5 * dw.to(torch.float64) * (g0.to(torch.float64) + g1.to(torch.float64))


def choose_primitive(s_obj: torch.Tensor, s_vin: Optional[torch.Tensor] = None) -> list:
    """Choice of the primitives (indices, in dose ORDER) from the midpoint prices. One functional: those with s_obj < 0,
    most negative first. Two (objective A, constraint B, arXiv 2609.09867 §6(a): min sum s_A subject to sum s_B <= 0):
    first the "safe" ones (s_A < 0 and s_B <= 0) by increasing s_A, then the "expensive" ones (s_A < 0, s_B > 0) by
    increasing s_A / s_B (most objective per unit of constraint) as long as the sum of s_B stays <= 0 (fractional
    knapsack greedy: exact for the linear relaxation except for the last one). Non-finite prices = unpriced primitive:
    excluded."""
    so = s_obj.to(torch.float64)
    ok = torch.isfinite(so) & (so < 0)
    if s_vin is None:
        idx = torch.nonzero(ok).reshape(-1)
        return idx[torch.argsort(so[idx], stable=True)].tolist()
    sv = s_vin.to(torch.float64)
    ok = ok & torch.isfinite(sv)
    safe = torch.nonzero(ok & (sv <= 0)).reshape(-1)
    safe = safe[torch.argsort(so[safe], stable=True)].tolist()
    expensive = torch.nonzero(ok & (sv > 0)).reshape(-1)
    expensive = expensive[torch.argsort(so[expensive] / sv[expensive], stable=True)].tolist()
    choices, margin = list(safe), -float(sv[safe].sum()) if safe else 0.0
    for i in expensive:
        if float(sv[i]) <= margin:
            choices.append(i)
            margin -= float(sv[i])
    return choices


def sort_primitive(s_obj: torch.Tensor, s_vin: Optional[torch.Tensor] = None) -> list:
    """2026-09-29 pv-letters-doses branch (`DoseConfig.pm_greedy`): the same order as `choose_primitive` (safe by
    increasing s_A, then expensive by increasing s_A / s_B) but WITHOUT the cut on the predicted balance sum s_B <= 0:
    the constraint is checked by MEASURING the generic probe after each primitive (midpoint prices do not see the
    interaction BETWEEN primitives, only the one within). One functional: identical to `choose_primitive`."""
    if s_vin is None:
        return choose_primitive(s_obj)
    so, sv = s_obj.to(torch.float64), s_vin.to(torch.float64)
    ok = torch.isfinite(so) & (so < 0) & torch.isfinite(sv)
    safe = torch.nonzero(ok & (sv <= 0)).reshape(-1)
    expensive = torch.nonzero(ok & (sv > 0)).reshape(-1)
    return (safe[torch.argsort(so[safe], stable=True)].tolist()
            + expensive[torch.argsort(so[expensive] / sv[expensive], stable=True)].tolist())


@dataclass
class DoseConfig:
    """--pv-sonda-doses: fractions (increasing, in (0, 1]) of the chosen jumps, nested prefixes in gain order;
    --pv-recenter: V recentred after the dose, at most `recenter_rounds` halvings confirmed by probe B."""
    fractions: tuple = FRACTIONS_DOSES
    recenter: bool = False
    recenter_rounds: int = 3
    # 2026-09-29 midpoint branch (--pv-point-medio, arXiv 2609.09867 Prop. 2 and §6(a); math note on the PV midpoint).
    # Off (default) = the usual stream, bit-exact. On: the chosen jumps are grouped into PRIMITIVES (`pm_primitive`); the
    # `pm_max` primitives with the highest predicted gain are priced at their OWN MIDPOINT with the trapezoidal rule
    # over the integer states (s_i = 1/2 d_i^T (grad F(q0) + grad F(q0 + d_i)), exact at constant curvature: the half
    # trit does not exist in the format), one probe backward per primitive (+ one at q0); within each primitive, with
    # `pm_filter`, only the jumps with price < 0 at the primitive's centre are kept; the primitives with price < 0 (with
    # two functionals: objective A, constraint sum s_B <= 0) are the DOSES (prefixes of primitives), then confirmation
    # and recenter as always.
    point_mean: bool = False
    pm_primitive: str = "family"
    pm_max: int = 12
    pm_filter: bool = True
    # 2026-09-29 pv-letters-doses branch (--pv-pm-greedy, only with point_mean). Off (default) = bit-exact. On: the
    # doses are no longer prefixes on the `fractions` scale judged by the argmin of the GENERIC probe A, but a GREEDY trial
    # primitive by primitive in the order of `sort_primitive`: the primitive is applied, the decision OBJECTIVE
    # functional is measured (`measure_objective_doses`, the 4 letters on M1; without it, probe A) together with probe A;
    # it is kept if the objective decreases w.r.t. the current state AND probe A stays <= its start value (the
    # constraint sum s_B <= 0 of arXiv 2609.09867 §6(a), MEASURED instead of predicted), otherwise it is cancelled
    # bit-exact and the next one is tried. No new threshold: the comparisons are the same as the dose probe (<, <=).
    # Then confirmation B, recenter and confirmation of the objective on M2 as always.
    pm_greedy: bool = False
    # 2026-09-29 tt3-fixes branch (--pv-recenter-objective, only with recenter and pm_greedy). Off (default) = bit-exact.
    # Cause (TT3 V step 2): the greedy dose is decided by the LETTERS (M1) but the recenter removes the "harmful" jumps
    # for the GENERIC probe and accepts its rounds looking only at B; the final confirmation on the M2 letters then
    # failed (0.33433 -> 0.34973) and EVERYTHING was cancelled, including the dose the letters had chosen. On: each
    # recenter round is kept only if, besides B, the decision objective functional (`measure_objective_doses`, M1: the
    # same data as the choice, M2 stays untouched for confirmation) does NOT rise w.r.t. the value reached by the dose;
    # and the M2 confirmation is also measured BEFORE the recenter (registry only: it tells who made the letters rise,
    # the decision does not read it).
    recenter_objective: bool = False
    # 2026-10-02 tt5-phase-v branch (FASE_V.md). Execution, not decision: off (default) = as always, bit-exact; on = SAME
    # decisions, less work.
    # V2 `pm_order_layers` (--pv-pm-order-layers): the g1 gradients of the midpoint prices (one per primitive, each
    # at q0 + d_p, independent of the others) are computed in DECREASING layer order instead of predicted-gain order:
    # with reuse (--pv-riuso) the forward restarts from the lowest layer changed since the last measurement, i.e.
    # min(layer of the previous primitive, layer of this one); in decreasing order that minimum is always this one's
    # layer. Every g1 and every price stays the same number (same byte state, same batch), stored by index as before.
    pm_order_layers: bool = False
    # V3 `greedy_lazy` (--pv-greedy-lazy, only with pm_greedy): in the greedy trial a primitive is kept only if
    # (objective < current) AND (probe A <= start). If probe A is already above the start the primitive is rejected
    # WHATEVER the objective: its measurement (one forward of the letters on M1) is skipped. Registry: "objective" None,
    # "objective_skipped" True for that entry.
    greedy_lazy: bool = False

    def __post_init__(self):
        if self.greedy_lazy and not self.pm_greedy:
            raise ValueError("greedy_lazy only with pm_greedy")
        f = [float(x) for x in self.fractions]
        if not f or any(not 0.0 < x <= 1.0 for x in f) or any(b <= a for a, b in zip(f, f[1:])):
            raise ValueError(f"dose fractions {self.fractions}: must be increasing, in (0, 1]")
        if int(self.recenter_rounds) < 1:
            raise ValueError(f"recenter_rounds {self.recenter_rounds} < 1")
        if self.pm_primitive not in PRIMITIVE_PM:
            raise ValueError(f"pm_primitive {self.pm_primitive!r}: one of {PRIMITIVE_PM}")
        if int(self.pm_max) < 1:
            raise ValueError(f"pm_max {self.pm_max} < 1")
        if self.pm_greedy and not self.point_mean:
            raise ValueError("pm_greedy only with point_mean (the primitives come from the midpoint prices)")
        if self.recenter_objective and not (self.recenter and self.pm_greedy):
            raise ValueError("recenter_objective only with recenter and pm_greedy (needs the decision objective, M1)")


def counts_doses(n: int, fractions) -> list:
    """Jumps of each dose: max(1, ceil(f n)) (cap n), increasing, without duplicates (with few jumps the small doses
    coincide: a single measurement)."""
    out = []
    for f in fractions:
        c = min(n, max(1, math.ceil(round(float(f) * n, 9))))
        if not out or c > out[-1]:
            out.append(c)
    return out


def changes_routing(idx_a: torch.Tensor, idx_b: torch.Tensor) -> float:
    """Fraction of tokens whose top-k (as a set) differs between two routings of the same layer."""
    return float((torch.sort(idx_a, -1).values != torch.sort(idx_b, -1).values).any(-1).double().mean())


# ═══════════════════════════════════ 9. orchestrator ═══════════════════════════════════
class VStep:
    """Buffers + mode per micro-batch + selection + jumps + registry for a set of families
    {(L, name): FamilyData}. Usage per step:
        pv.start_step(); for k, m in enumerate(micro): pv.microbatch(k); <forward/backward of m>
        opt.step()                                                       # P step (scales)
        for m in curv: pv.curvature(); <forward, backward of a vector with cov. = Hessian of the outputs>
        pv.end_collection()
        reg, diag = pv.select_e_skip(current_scales, rho)            # V step
        kept = region.outcome(rho, kl_before, kl_after_on_probe, len(reg)); if not kept: pv.cancel(reg)"""

    def __init__(self, families: dict, cfg: ConfigV = ConfigV(), state: Optional[VState] = None,
                 bop: Optional[ConfigBop] = None):
        self.fam = families
        self.cfg = cfg
        if state is None:
            self.state = VState(cfg.n_proposal, cfg.steps_accumulator, bop=bop)
            self.state.prepare(families.values())
        else:
            if bop is not None and state.bop is None:
                raise ValueError("V-Bop VStep with a VState without V-Bop")
            self.state = state
        self.registers: list = []
        self._bop_before: Optional[tuple] = None      # (registry, {key: (e, r, g, values)}) for cancellation
        self.times: Optional[dict] = None             # 2026-10-02 --pv-times-v: seconds per sub-phase (None = no clock)

    def start_step(self) -> None:
        self.state.start_step()

    def microbatch(self, k: int) -> int:
        return self.state.microbatch(k)

    def curvature(self) -> int:
        return self.state.curvature()

    def end_collection(self) -> None:
        self.state.end_collection()

    def register_scale_window(self, scale: dict) -> None:
        """Scales at the START of the accumulation window (before the P step of its first step): staleness
        reference. With a window of 1 it is not needed (no filter: previous behaviour, bit-exact)."""
        if self.state.steps_accumulator > 1:
            self.state.d_window = {k: scale[k].detach().to(torch.float16).clone() for k in self.fam}

    def families_active(self) -> dict:
        """The families the V step decides on (2026-09-27 layer domain: only those of the trained layers, `VState.active`;
        None = all, as always). Cancellation and replay stay on ALL of them (`self.fam`: earlier registries)."""
        a = getattr(self.state, "active", None)
        return self.fam if a is None else {k: f for k, f in self.fam.items() if k in a}

    def select(self, scale: dict, rho: float) -> dict:
        fam = self.families_active()
        if self.state.bop is not None:
            return {k: select_bop(self.state.bop_[k], self.state.buffer(f), f, scale[k], self.state.n_verify,
                                     rho, self.cfg, self.state.bop, n_loss=self.state.n_loss)
                    for k, f in fam.items()}
        return {k: select(self.state.buffer(f), f, scale[k], self.state.n_verify, rho, self.cfg,
                             n_loss=self.state.n_loss, n_curvature=self.state.n_curvature,
                             d_ref=self.state.d_window.get(k) if self.state.steps_accumulator > 1 else None)
                for k, f in fam.items()}

    def select_e_skip(self, scale: dict, rho: float) -> tuple:
        """All or nothing: if the jumps of one family fail, those already written in the others are cancelled.
        V-Bop: first the update of the persistent state (EMA), after the jumps the ECHO correction."""
        if self.state.bop is not None:
            self.state.update_bop()
        sel = self.select(scale, rho)
        regs = []
        try:
            for k, s in sel.items():
                regs.append(apply_skips(self.fam[k], s))
        except Exception:
            cancel(self.fam, Registry.merge(regs))
            raise
        reg = Registry.merge(regs)
        self.registers.append(reg)
        diag = {f"{k[0]}.{k[1]}": {"tested": int(s.tested.sum()), "findings": int(s.findings.sum()),
                                   "convenient": int(s.convenient.sum()), "skips": len(s),
                                   "gain_predicted": float(s.gain.sum())} for k, s in sel.items()}
        if self.state.bop is not None:
            self._bop_before = (reg, {k: self._after_skip_bop(k, s, scale[k], r)
                                     for (k, s), r in zip(sel.items(), regs)})
            for k in sel:
                z = self.state.bop_[k]
                diag[f"{k[0]}.{k[1]}"]["with_candidate"] = int(((z.eta & ETA_MAX) >= 1).sum())
        return reg, diag

    def _after_skip_bop(self, k, sel: Selection, d_eff: torch.Tensor, reg: Registry) -> tuple:
        """ECHO correction on the jumped groups: m^ <- m^ - H |d| sign(m^) (the gradient after the jump, in the quadratic
        model); if the direction is no longer feasible from the new trit (edge of {-1,0,1}) the state is reset. Returns
        the previous values (bit-exact cancellation)."""
        z = self.state.bop_[k]
        e, r, g = sel.e, sel.r, sel.g
        before = {n: getattr(z, n)[e, r, g].clone() for n in ("m", "r", "H", "eta")}
        if len(sel) == 0:
            return e, r, g, before
        eta, _ = z.eta_direction()
        eta = eta[e, r, g]
        c, _ = _correction_eta(float(self.state.bop.gamma), eta)
        sm = z.sm[e].to(torch.float64)
        m_hat = z.m[e, r, g].to(torch.float64) * sm / c
        Hh = z.H[e, r, g].to(torch.float64) * z.sH[e].to(torch.float64)
        m_hat = m_hat - Hh * d_eff[e, r, g].to(torch.float64).abs() * torch.sign(m_hat)
        z.m[e, r, g] = (m_hat * c / sm).clamp(-65504.0, 65504.0).to(torch.float16)
        tn = torch.from_numpy(reg.rows["t_new"].astype(np.int64)).to(sel.dir.device)
        outside = (tn + sel.dir.to(torch.int64)).abs() > 1
        if bool(outside.any()):
            ef, rf, gf = e[outside], r[outside], g[outside]
            for n in ("m", "r", "H"):
                getattr(z, n)[ef, rf, gf] = 0
            z.eta[ef, rf, gf] = 0
        return e, r, g, before

    # ─── 2026-09-27 v-interaction-b branch: dose probe with confirmation (--pv-sonda-doses) and recentred V (--pv-recenter) ───
    def _curvature_skips(self, k, s: Selection) -> torch.Tensor:
        """Gauss-Newton curvature PER MICRO-BATCH on the code of each chosen jump (float64): PV = step H / n_curv
        (the curvature is measured on the candidate, which is the jump column); V-Bop = its EMA (already per micro-batch).
        Read BEFORE the ECHO correction (which may reset the state)."""
        if len(s) == 0:
            return torch.zeros(0, dtype=torch.float64, device=s.e.device)
        if self.state.bop is not None:
            z = self.state.bop_[k]
            return z.H[s.e, s.r, s.g].to(torch.float64) * z.sH[s.e].to(torch.float64)
        nc = self.state.n_curvature
        if nc < 1:
            raise ValueError("recenter without step curvature (curvature micro-batches, mode 3, are needed)")
        return self.state.buffer(self.fam[k]).H[s.e, s.r, s.g].to(torch.float64) / nc

    @staticmethod
    def _below(s: Selection, li: torch.Tensor) -> Selection:
        """The rows `li` (local indices) of a Selection; the per-expert counts stay those of the selection."""
        li = li.to(s.e.device)
        return Selection(s.e[li], s.r[li], s.g[li], s.cand[li], s.dir[li], s.gain[li], s.tested, s.findings,
                         s.convenient, s.budget, s.chosen, s.cost)

    def select_e_trial_doses(self, scale: dict, rho: float, measure_a: Callable, measure_b: Callable,
                               cfg: "DoseConfig", gradient_recenter: Optional[Callable] = None,
                               gradients_pm: Optional[dict] = None, factors_pm: Optional[dict] = None,
                               measure_objective: Optional[Callable] = None,
                               measure_objective_doses: Optional[Callable] = None) -> tuple:
        """V step with DOSE PROBE and CONFIRMATION (TECNICHE_E2E_PV.md §32.5 action 1) and, with `cfg.recenter`, RECENTRED
        V (action 2). Replaces `select_e_skip` + all-or-nothing probe.
          1. (V-Bop) EMA of the state; selection as always (same criteria, rho budget, per-expert ordering).
          2. All chosen jumps in GLOBAL order of predicted gain (decreasing, stable); doses = nested prefixes
             of max(1, ceil(f n)) jumps for f in `cfg.fractions` (duplicates removed). `measure_a()` before everything
             and after each dose (jumps are added in registry increments: no jump is re-applied).
          3. Best dose = argmin of probe A (on ties the smallest). If it does not go below the starting probe A,
             or if on probe B (`measure_b`, disjoint micro-batch: no winner's curse from choosing among 5 doses)
             it is not <= the starting B: EVERYTHING cancelled bit-exact, as today.
          4. Recentring (if `cfg.recenter` and `gradient_recenter`): in the reached state q_S, `gradient_recenter()`
             runs ONE probe-A backward with the VState in `prepare_recenter` (returns False if the gradient is not
             valid: the recenter is skipped); for each kept jump the leave-one-out (§32.4 (ii))
                 f(S) - f(S \\ i) ~ dw_i g1_i - 1/2 H_ii dw_i^2,   dw_i = (t_new - t_old) d_i,
             g1 = S of the buffer (dL/dW_rot on the code, in units of one loss micro-batch of the step), H_ii = curvature
             per micro-batch (`_curvature_skips`). > 0 = the jump, given the others, does harm. Harmful ones in order of
             harm; the most harmful HALF is cancelled, then probe B: if it does not rise the cancellation is kept and
             we move on to half of the remainder, otherwise it is restored and we stop (damped Jacobi: simultaneous
             cancellations interact too); at most `cfg.recenter_rounds` rounds.
          5. Registry of the KEPT jumps (gain order) in `self.registers`; V-Bop: ECHO correction ONLY on the kept ones
             (a jump tried and cancelled leaves the state as after the EMA, like today's cancellation).
        2026-09-29 `cfg.point_mean` (midpoint branch): between 1 and 2 the order and the doses come from the midpoint
        prices of the primitives (`_midpoint_prices`, `choose_primitive`); `gradients_pm` = {"B": backward of the generic
        probe, and optional "A": backward of the objective functional (KL of the 4 letters)}, each like
        `gradient_recenter` (the VState is in `prepare_recenter`; False = gradient not valid); `factors_pm` = {name:
        gradient unit -> per token} (registry only); `measure_objective` (optional, data DISJOINT from those of the
        prices): at the end the objective functional must have decreased, or everything is cancelled. Off: none of
        these paths, bit-exact.
        2026-09-29 `cfg.pm_greedy` (pv-letters-doses branch): steps 2-3 become `_doses_greedy` (primitive by primitive,
        decided by the objective `measure_objective_doses` with the measured probe-A constraint).
        Returns (kept registry or None, diag as in `select_e_skip`, outcome)."""
        if cfg.point_mean and not (gradients_pm and "B" in gradients_pm):
            raise ValueError("midpoint without the generic probe gradient (gradients_pm['B'])")
        times = self.times
        if self.state.bop is not None:
            self._bop_before = None
            with timer(times, "update_bop"):
                self.state.update_bop()
        with timer(times, "select"):
            sel = self.select(scale, rho)
            diag = {f"{k[0]}.{k[1]}": {"tested": int(s.tested.sum()), "findings": int(s.findings.sum()),
                                       "convenient": int(s.convenient.sum()), "skips": len(s),
                                       "gain_predicted": float(s.gain.sum())} for k, s in sel.items()}
        keys = list(sel.keys())
        rows_f, gua, Hs, dws, fid, loc = [], [], [], [], [], []
        _cr = timer(times, "prepare_skips")
        _cr.__enter__()
        for i, k in enumerate(keys):
            s = sel[k]
            rows_f.append(rows_skips(self.fam[k], s))
            gua.append(s.gain.detach().to("cpu", torch.float64))
            Hs.append(self._curvature_skips(k, s).to("cpu"))
            d = scale[k][s.e, s.r, s.g].to(torch.float64) if len(s) else torch.zeros(0, dtype=torch.float64)
            dws.append((s.dir.to(torch.float64).to(d.device) * d).to("cpu"))
            fid.append(torch.full((len(s),), i, dtype=torch.int64))
            loc.append(torch.arange(len(s), dtype=torch.int64))
        rows = np.concatenate(rows_f) if rows_f else np.zeros(0, dtype=REG_DTYPE)
        gain_t = torch.cat(gua) if gua else torch.zeros(0, dtype=torch.float64)
        H_t = torch.cat(Hs) if Hs else torch.zeros(0, dtype=torch.float64)
        dw_t = torch.cat(dws) if dws else torch.zeros(0, dtype=torch.float64)
        fid_t = torch.cat(fid) if fid else torch.zeros(0, dtype=torch.int64)
        loc_t = torch.cat(loc) if loc else torch.zeros(0, dtype=torch.int64)
        _cr.__exit__(None, None, None)
        n = int(rows.shape[0])
        outcome = {"skips": n, "probe_before_a": measure_a(), "probe_before_b": None, "doses": [], "dose": None,
                 "probe_after_a": None, "probe_b_dose": None, "probe_after_b": None, "confirmed": None,
                 "kept": False, "skips_kept": 0, "partial": False, "gain_predicted": float(gain_t.sum()),
                 "gain_predicted_kept": 0.0, "recenter": None}
        if n == 0:                                       # as today: empty registry kept, no extra measurement
            reg = Registry()
            self.registers.append(reg)
            if self.state.bop is not None:
                self._bop_before = (reg, {})
            outcome.update(kept=True, probe_after_a=outcome["probe_before_a"])
            self._diag_bop(diag)
            return reg, diag, outcome
        obj0 = None
        if cfg.point_mean:                              # 2026-09-29 midpoint branch: order and doses from midpoint prices
            order, counts, outcome["point_mean"] = self._midpoint_prices(
                keys, sel, rows, fid_t, loc_t, dw_t, gain_t, cfg, gradients_pm, factors_pm or {})
            if order.numel() == 0:                      # no convenient primitive: no jump, as with n == 0
                reg = Registry()
                self.registers.append(reg)
                if self.state.bop is not None:
                    self._bop_before = (reg, {})
                outcome.update(kept=True, probe_after_a=outcome["probe_before_a"])
                self._diag_bop(diag)
                return reg, diag, outcome
            if measure_objective is not None:
                obj0 = measure_objective()
        else:
            order = torch.argsort(gain_t, descending=True, stable=True)
            counts = counts_doses(n, cfg.fractions)
        rows_o = rows[order.numpy()]
        cum = torch.cumsum(gain_t[order], 0)
        outcome["probe_before_b"] = measure_b()
        if cfg.point_mean and cfg.pm_greedy:           # 2026-09-29 pv-letters-doses branch: greedy trial per primitive
            kept = self._doses_greedy(order, counts, rows_o, cum, n, measure_a, measure_b, measure_objective_doses, outcome,
                                       lazy=cfg.greedy_lazy)
            if kept is None:
                self._diag_bop(diag)
                return None, diag, outcome
        else:
            increments, a = [], 0
            try:
                for c in counts:
                    inc = Registry(rows_o[a:c].copy())
                    replay(self.fam, [inc])
                    increments.append(inc)
                    a = c
                    outcome["doses"].append({"fraction": c / n, "skips": c, "probe_a": measure_a(),
                                          "gain_predicted": float(cum[c - 1])})
            except BaseException:
                for inc in reversed(increments):
                    cancel(self.fam, inc)
                raise
            j = min(range(len(counts)), key=lambda i: outcome["doses"][i]["probe_a"])
            outcome["probe_after_a"] = outcome["doses"][j]["probe_a"]
            if outcome["doses"][j]["probe_a"] < outcome["probe_before_a"]:
                for inc in reversed(increments[j + 1:]):
                    cancel(self.fam, inc)
                outcome["probe_after_b"] = outcome["probe_b_dose"] = measure_b()
                outcome["confirmed"] = outcome["probe_after_b"] <= outcome["probe_before_b"]
            else:
                outcome["confirmed"] = False
            outcome["dose"] = {"fraction": counts[j] / n, "skips": counts[j], "probe_a": outcome["doses"][j]["probe_a"]}
            if not outcome["confirmed"]:
                for inc in reversed(increments[:j + 1] if outcome["probe_after_b"] is not None else increments):
                    cancel(self.fam, inc)
                self._diag_bop(diag)
                return None, diag, outcome
            kept = order[:counts[j]].clone()
        kw_obj = {}
        if cfg.recenter_objective and cfg.recenter and gradient_recenter is not None \
                and measure_objective_doses is not None and outcome.get("greedy"):
            kw_obj = {"measure_obj": measure_objective_doses, "objective": outcome["greedy"]["objective_after"]}
            if obj0 is not None:                         # registry only: the M2 letters at the dose, before the recenter
                outcome["point_mean"]["objective_dose"] = measure_objective()
        if cfg.recenter and gradient_recenter is not None:
            kept, outcome["recenter"], outcome["probe_after_b"] = self._recenter(
                keys, sel, rows, fid_t, loc_t, dw_t, H_t, kept, gradient_recenter, measure_b,
                outcome["probe_after_b"], int(cfg.recenter_rounds), **kw_obj)
        if kept.numel() == 0:                          # the recenter removed everything: like a cancellation
            outcome.update(kept=False, skips_kept=0)
            self._diag_bop(diag)
            return None, diag, outcome
        if obj0 is not None:                             # 2026-09-29 midpoint: confirmation of the OBJECTIVE on disjoint data
            obj1 = measure_objective()
            outcome["point_mean"]["objective"] = {"before": obj0, "after": obj1, "confirmed": bool(obj1 < obj0)}
            if not obj1 < obj0:
                cancel(self.fam, Registry(rows[kept.numpy()].copy()))
                outcome.update(kept=False, skips_kept=0, confirmed=False)
                self._diag_bop(diag)
                return None, diag, outcome
        reg = Registry(rows[kept.numpy()].copy())
        self.registers.append(reg)
        if self.state.bop is not None:
            before = {}
            for i, k in enumerate(keys):
                li = loc_t[kept][fid_t[kept] == i]
                sk = self._below(sel[k], li)
                before[k] = self._after_skip_bop(k, sk, scale[k], Registry(rows_f[i][li.numpy()].copy()))
            self._bop_before = (reg, before)
            self._diag_bop(diag)
        outcome.update(kept=True, skips_kept=int(kept.numel()), partial=int(kept.numel()) < n,
                     gain_predicted_kept=float(gain_t[kept].sum()))
        return reg, diag, outcome

    # ─── 2026-09-29 midpoint branch (--pv-point-medio): prices of the primitives at their own midpoint ───
    def _gradient_on_skips(self, keys, sel, fid_t, loc_t, idx: torch.Tensor, gradient: Callable):
        """dL/dW_rot on the column of each jump `idx` (global indices in the list of chosen jumps) in the CURRENT state
        of the bytes: `prepare_recenter` on the jumps `idx` only (cand := jump column, mode 2), ONE backward (`gradient`),
        read S. None if the gradient is not valid (overflow). float64 on CPU, in the order of `idx`."""
        skips, pos = {}, {}
        for i, k in enumerate(keys):
            m = fid_t[idx] == i
            if not bool(m.any()):
                continue
            s = sel[k]
            li = loc_t[idx][m].to(s.e.device)
            skips[k] = (s.e[li], s.r[li], s.g[li], s.cand[li])
            pos[k] = torch.nonzero(m).reshape(-1)
        self.state.prepare_recenter(skips)
        try:
            valid = gradient()
        finally:
            self.state.end_collection()
        if valid is False:
            return None
        g = torch.zeros(idx.numel(), dtype=torch.float64)
        for k, (e, r, gg, _) in skips.items():
            g[pos[k]] = self.state.buffer(self.fam[k]).S[e, r, gg].to("cpu", torch.float64)
        return g

    def _midpoint_prices(self, keys, sel, rows, fid_t, loc_t, dw_t, gain_t, cfg: "DoseConfig",
                            gradients: dict, factors: dict) -> tuple:
        """Step 1b of `select_e_trial_doses` with `cfg.point_mean` (arXiv 2609.09867 Prop. 2 and §6(a), math note on the
        PV midpoint). Returns (order of the jumps kept for the doses, cumulative dose counts in jumps, info).
          1. Primitives: jumps grouped by `cfg.pm_primitive`; the `cfg.pm_max` with the highest predicted gain (from the
             usual criterion) are priced, the others are discarded for this V step.
          2. For each functional (B = generic probe; A = objective, optional): g0 = gradient at q0 on the columns of
             ALL priced jumps (ONE backward: one jump per group, distinct columns); for each primitive p: jumps of p
             applied (replay), g1 = gradient at q0 + d_p on its columns (ONE backward), bit-exact cancellation.
          3. v_j = 1/2 dw_j (g0_j + g1_j) (`midpoint_price`): s_p = sum_{j in p} v_j = F(q0 + d_p) - F(q0) exact at
             constant curvature (including ALL the interaction within the primitive, which the diagonal criterion
             ignores); v_j is the price of j at the primitive's centre (Eq. 4: off by 1/2 x^T beta (1 - x) on the jumps
             within only).
          4. `cfg.pm_filter`: within each primitive only the jumps with v_obj < 0 remain (s_p recomputed on them, read at
             the centre); then `choose_primitive`; doses = prefixes of primitives (`counts_doses` on the NUMBER of
             primitives)."""
        n = int(rows.shape[0])
        fid, loc = fid_t, loc_t
        e_all = torch.cat([sel[k].e.to("cpu", torch.int64) for k in keys]) if keys else torch.zeros(0, dtype=torch.int64)
        type_ = cfg.pm_primitive
        if type_ == "family":
            key_p = [(int(keys[int(f)][0]), keys[int(f)][1]) for f in fid.tolist()]
        elif type_ == "type_":
            key_p = [keys[int(f)][1] for f in fid.tolist()]
        elif type_ == "layer":
            key_p = [int(keys[int(f)][0]) for f in fid.tolist()]
        else:
            key_p = [(int(keys[int(f)][0]), keys[int(f)][1], int(e)) for f, e in zip(fid.tolist(), e_all.tolist())]
        names_p, pid = {}, torch.empty(n, dtype=torch.int64)
        for j, c in enumerate(key_p):
            pid[j] = names_p.setdefault(c, len(names_p))
        keys_p = list(names_p)
        P = len(keys_p)
        g_prev = torch.zeros(P, dtype=torch.float64).index_add_(0, pid, gain_t.to(torch.float64))
        priced = torch.argsort(g_prev, descending=True, stable=True)[:int(cfg.pm_max)]
        info = {"primitive_kind": type_, "primitive": P, "priced": int(priced.numel()), "backward_": 0, "functional":
                sorted(gradients), "valid": True, "discarded_non_priced": 0}
        members = {int(p): torch.nonzero(pid == p).reshape(-1) for p in priced.tolist()}
        all_ = torch.cat([members[int(p)] for p in priced.tolist()])
        info["discarded_non_priced"] = n - int(all_.numel())
        empty = (torch.zeros(0, dtype=torch.int64), [], info)
        v = {name: torch.full((n,), math.nan, dtype=torch.float64) for name in gradients}
        current = {name: torch.full((n,), math.nan, dtype=torch.float64) for name in gradients}
        for name, grad in gradients.items():
            g0 = self._gradient_on_skips(keys, sel, fid, loc, all_, grad)
            info["backward_"] += 1
            if g0 is None:                               # gradient at q0 not valid: no price, no jump
                info["valid"] = False
                return empty
            g0_full = torch.zeros(n, dtype=torch.float64)
            g0_full[all_] = g0
            current[name][all_] = dw_t[all_] * g0
            round_p = priced.tolist()
            if cfg.pm_order_layers:                     # 2026-10-02 V2: decreasing layer (lowest of the primitive)
                round_p = sorted(round_p, key=lambda p: -int(rows["L"][members[int(p)].numpy()].min()))
            for p in round_p:
                idx = members[int(p)]
                reg_p = Registry(rows[idx.numpy()].copy())
                replay(self.fam, [reg_p])
                try:
                    g1 = self._gradient_on_skips(keys, sel, fid, loc, idx, grad)
                finally:
                    cancel(self.fam, reg_p)
                info["backward_"] += 1
                if g1 is not None:
                    v[name][idx] = midpoint_price(dw_t[idx], g0_full[idx], g1)
        obj = "A" if "A" in gradients else "B"
        vin = "B" if obj == "A" else None
        kept_p, s_obj, s_vin, rows_info = [], [], [], []
        for p in priced.tolist():
            idx = members[int(p)]
            vo = v[obj][idx]
            inside = idx[vo < 0] if cfg.pm_filter else idx
            if not bool(torch.isfinite(vo).all()):
                inside = idx[:0]                         # g1 not valid: primitive not priced
            inside = inside[torch.argsort(v[obj][inside], stable=True)]
            kept_p.append(inside)
            so = float(v[obj][inside].sum()) if inside.numel() else math.nan
            sv = float(v[vin][inside].sum()) if (vin and inside.numel()) else math.nan
            s_obj.append(so)
            s_vin.append(sv)
            f = lambda name: float(factors.get(name, 1.0))  # noqa: E731
            rows_info.append({"primitive_kind": str(keys_p[int(p)]), "skips": int(idx.numel()), "kept": int(inside.numel()),
                               "gain_predicted": float(g_prev[int(p)]),
                               **{f"s_{name}": float(v[name][idx].sum()) * f(name) for name in gradients},
                               **{f"corrente_{name}": float(current[name][idx].sum()) * f(name) for name in gradients},
                               **{f"s_{name}_filtrata": (float(v[name][inside].sum()) * f(name) if inside.numel()
                                                         else None) for name in gradients}})
        sc = choose_primitive(torch.tensor(s_obj, dtype=torch.float64),
                              torch.tensor(s_vin, dtype=torch.float64) if vin else None)
        if cfg.pm_greedy:                                # 2026-09-29 pv-letters-doses: the balance is measured, not predicted
            info["choices_predicted"] = [rows_info[i]["primitive_kind"] for i in sc]
            sc = sort_primitive(torch.tensor(s_obj, dtype=torch.float64),
                                  torch.tensor(s_vin, dtype=torch.float64) if vin else None)
        info["per_primitive"] = rows_info
        info["choices"] = [rows_info[i]["primitive_kind"] for i in sc]
        if not sc:
            return empty
        order = torch.cat([kept_p[i] for i in sc])
        cut = np.cumsum([int(kept_p[i].numel()) for i in sc]).tolist()
        counts = cut if cfg.pm_greedy else [cut[c - 1] for c in counts_doses(len(sc), cfg.fractions)]
        info["skips_chosen"] = int(order.numel())
        for name in gradients:
            info[f"previsto_{name}"] = float(v[name][order].sum()) * float(factors.get(name, 1.0))
            info[f"previsto_dosi_{name}"] = [float(v[name][order[:c]].sum()) * float(factors.get(name, 1.0))
                                             for c in counts]
        return order, counts, info

    def _doses_greedy(self, order, counts, rows_o, cum, n: int, measure_a: Callable, measure_b: Callable,
                     measure_obj: Optional[Callable], outcome: dict, lazy: bool = False) -> Optional[torch.Tensor]:
        """2026-09-29 pv-letters-doses branch (`DoseConfig.pm_greedy`), replacing steps 2-3 of `select_e_trial_doses`.
        `counts` = cumulative boundaries of the primitives in `order` (one primitive per segment). For each primitive:
        replay, measurement of probe A (constraint) and of the DECISION objective functional `measure_obj` (the 4 letters
        on M1; None = probe A itself, no separate constraint); retained if objective < current objective and probe A <=
        starting probe A, otherwise cancelled bit-exact. Then probe B (disjoint) <= starting B on the kept ones only, or
        everything cancelled. Returns the indices (in `order`) of the kept jumps, or None (no jump: bytes as at start)."""
        a0 = outcome["probe_before_a"]
        separate = measure_obj is not None
        o0 = measure_obj() if separate else a0
        cur_o = o0
        cur_a = a0
        kept, applied, pending, a = [], [], None, 0
        try:
            for c in counts:
                inc = Registry(rows_o[a:c].copy())
                replay(self.fam, [inc])
                pending = inc
                ma = measure_a()
                if separate and lazy and not bool(ma <= a0):     # 2026-10-02 V3: rejected anyway, objective not needed
                    mo, keeps = None, False
                else:
                    mo = measure_obj() if separate else ma
                    keeps = bool(mo < cur_o) and (not separate or bool(ma <= a0))
                outcome["doses"].append({"fraction": c / n, "skips": c - a, "probe_a": ma,
                                      "objective": mo if separate else None, "retention": keeps,
                                      "gain_predicted": float(cum[c - 1]) - (float(cum[a - 1]) if a else 0.0)})
                if separate and lazy and mo is None:
                    outcome["doses"][-1]["objective_skipped"] = True
                if keeps:
                    applied.append(inc)
                    kept.append(order[a:c])
                    cur_o, cur_a = mo, ma
                else:
                    cancel(self.fam, inc)
                pending = None
                a = c
        except BaseException:
            if pending is not None:
                cancel(self.fam, pending)
            for inc in reversed(applied):
                cancel(self.fam, inc)
            raise
        outcome["greedy"] = {"objective_before": o0 if separate else None, "objective_after": cur_o if separate else None,
                           "primitive": len(counts), "kept": len(applied)}
        outcome["probe_after_a"] = cur_a
        k = int(sum(int(t.numel()) for t in kept))
        outcome["dose"] = {"fraction": k / n, "skips": k, "probe_a": cur_a}
        if not applied:
            outcome["confirmed"] = False
            return None
        outcome["probe_after_b"] = outcome["probe_b_dose"] = measure_b()
        outcome["confirmed"] = outcome["probe_after_b"] <= outcome["probe_before_b"]
        if not outcome["confirmed"]:
            for inc in reversed(applied):
                cancel(self.fam, inc)
            return None
        return torch.cat(kept).clone()

    def _diag_bop(self, diag: dict) -> None:
        """V-Bop: groups with a candidate per family in the diag (as in `select_e_skip`, after the ECHO correction)."""
        if self.state.bop is None:
            return
        for k, z in self.state.bop_.items():
            if f"{k[0]}.{k[1]}" in diag:
                diag[f"{k[0]}.{k[1]}"]["with_candidate"] = int(((z.eta & ETA_MAX) >= 1).sum())

    def _recenter(self, keys, sel, rows, fid_t, loc_t, dw_t, H_t, kept, gradient, measure_b, probe_b: float,
                  rounds: int, measure_obj: Optional[Callable] = None, objective: Optional[float] = None) -> tuple:
        """Step 4 of `select_e_trial_doses`. Returns (kept, info, final probe B). 2026-09-29 tt3 fixes: with `measure_obj`
        (DoseConfig.recenter_objective) a round is kept only if the objective also does not rise above `objective`."""
        skips = {}
        for i, k in enumerate(keys):
            s = sel[k]
            li = loc_t[kept][fid_t[kept] == i].to(s.e.device)
            skips[k] = (s.e[li], s.r[li], s.g[li], s.cand[li])
        self.state.prepare_recenter(skips)
        try:
            valid = gradient()
        finally:
            self.state.end_collection()
        info = {"valid": bool(valid is not False), "harmful": 0, "rounds": [], "cancelled": 0,
                "kept": int(kept.numel())}
        if valid is False:
            return kept, info, probe_b
        g1 = torch.zeros(kept.numel(), dtype=torch.float64)
        for i, k in enumerate(keys):
            m = fid_t[kept] == i
            if not bool(m.any()):
                continue
            e, r, g, _ = skips[k]
            g1[m] = self.state.buffer(self.fam[k]).S[e, r, g].to("cpu", torch.float64)
        dw, H = dw_t[kept], H_t[kept]
        loo = dw * g1 - 0.5 * H * dw * dw
        info["loo_sum"] = float(loo.sum())
        info["loo_max"] = float(loo.max())
        dan = torch.nonzero(loo > 0).reshape(-1)
        dan = dan[torch.argsort(loo[dan], descending=True, stable=True)]
        info["harmful"] = int(dan.numel())
        info["loo_harmful"] = float(loo[dan].sum()) if dan.numel() else 0.0
        live = torch.ones(kept.numel(), dtype=torch.bool)
        remaining = dan
        for _ in range(max(0, rounds)):
            if remaining.numel() == 0:
                break
            h = (int(remaining.numel()) + 1) // 2
            removed = remaining[:h]
            reg_t = Registry(rows[kept[removed].numpy()].copy())
            cancel(self.fam, reg_t)
            new = measure_b()
            accept = new <= probe_b
            round_ = {"cancelled": h, "loo_predicted": float(loo[removed].sum()), "probe_b_before": probe_b,
                    "probe_b_after": new, "accepted": bool(accept)}
            if measure_obj is not None:                   # 2026-09-29 tt3 fixes: the round must not raise the objective
                o = measure_obj()
                round_.update(objective_before=objective, objective_after=o)
                accept = accept and o <= objective
                round_["accepted"] = bool(accept)
            info["rounds"].append(round_)
            if not accept:
                replay(self.fam, [reg_t])
                break
            live[removed] = False
            probe_b = new
            if measure_obj is not None:
                objective = o
            remaining = remaining[h:]
        kept = kept[live]
        info["cancelled"] = int((~live).sum())
        info["kept"] = int(kept.numel())
        return kept, info, probe_b

    def cancel(self, reg: Registry) -> None:
        cancel(self.fam, reg)
        for i in range(len(self.registers) - 1, -1, -1):
            if self.registers[i] is reg:
                del self.registers[i]
                break
        if self._bop_before is not None and self._bop_before[0] is reg:
            for k, (e, r, g, before) in self._bop_before[1].items():
                z = self.state.bop_[k]
                for n, v in before.items():
                    getattr(z, n)[e, r, g] = v
            self._bop_before = None


# ═══════════════════════════════════ 10. command line ═══════════════════════════════════
def main(argv=None) -> int:
    """`export`: final GGUF = GGUF with the checkpoint scales (from `e2e_export`) + codes from the checkpoint's PV
    registries (`migliore_*/registri_pv.npz`), verified on file and reloaded layer by layer."""
    import argparse
    import json
    ap = argparse.ArgumentParser(description=main.__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("export")
    p.add_argument("--scaled", required=True, help="GGUF already exported with the checkpoint scales")
    p.add_argument("--dst", required=True)
    p.add_argument("--registers", required=True, help="registri_pv.npz of the checkpoint (list, atomic)")
    p.add_argument("--sha256", default=None)
    p.add_argument("--layers-reload", default="0,3,39")
    a = ap.parse_args(argv)
    regs = Registry.load_list(Path(a.registers), a.sha256)
    rep = export_codes(Path(a.scaled), Path(a.dst), regs)
    rep["reload"] = reload_codes(Path(a.scaled), Path(a.dst), regs,
                                      [int(x) for x in a.layers_reload.split(",") if x != ""])
    print(json.dumps(rep, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
