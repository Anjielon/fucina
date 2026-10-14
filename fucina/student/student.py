#!/usr/bin/env python3
"""Thor-T student for the E2E phase that trains only the fp16 scales of the experts (project `e2e_progetto_scelto.md`).

2026-09-26. NEW file (the «student» module of the E2E stream). No existing file is modified; the following are imported
READ-ONLY: `load_layer_gguf` (verified GGUF->HF map, A_log corrected by audit B9), `dequant_gguf_torch`,
`e2e_codes_scale` / `e2e_fields_scale` (decoding from the bytes, layout), `rotation_gpu` (FWHT), and the in-house
codec (`S1024`).

What is here, and why (numbers from the adversarial review, §9 of the project):

1. **Student from the GGUF.** The non-experts are dequantized with the verified map (`load_layer_gguf.MAP`,
   `_apply`, `dequant_np`: the SAME path as `weights_layer_from_gguf`, skipping only the `per_expert` tensors) and
   loaded into a `Qwen3_5MoeDecoderLayer` (full attention / GDN as checked by B9). The experts stay as the GGUF
   BYTES (frozen codes) and are read only by the kernels: no second copy of the codes exists in memory. The only
   trained variables are theta (E, R, ng) fp32, d_eff = fp16_RNE(d0 * exp(theta)).
2. **Autograd function PER FAMILY** (`ExpertFamily`, A1). One call per (layer, family) over all the tokens, already
   sorted by expert. It saves only the input and the scales; in the backward pass each expert's weight is
   REBUILT from the bytes: dX with a grouped GEMM, dL/dd = group_sum trit (.) (dY^T X) accumulated in fp32.
   No (E, R, C) tensor is saved, no gradient as large as W is summed E times: backward is O(E), not O(E^2).
3. **FUSED trit decoding** (A3). Triton kernel: the trits are derived from the bytes INSIDE the product (forward and
   dX) and inside the dL/dd reduction, with the same integer arithmetic as `dequantize_row_tq1_0`
   (((b * 3^n) & 255) * 3) >> 8. The decoded weight never goes through global memory (engine "triton").
   Variant "triton_dense": one kernel decodes bytes -> fp16 weight and the product reads it (M0 picks the faster
   one). Engine "torch": the same math expert by expert, for the CPU and for tests.
   Engine "triton2" (2026-09-27): kernels _k_gemm2/_k_dx2/_k_dd2 (fixed scheme of 32-column pieces, scale per
   group, GROUP_M grouped ordering for L2, dL/dd with variable K and dY read contiguously, gate+up in the same
   sweep): forward and dX BIT-EXACT with "triton", dL/dd except for the order of the fp32 sums (~2e-7 relative).
   Tests: e2e_test_triton2.py; GPU gate: tools/cancello_kernel.sh.
4. **Recipes** (§2.6 F0b):
   - **B** («like the engine»): rotated activation R x = FWHT(S (.) x)/32 computed in fp32 and cast to fp16
     (like `rot_y_tq1_0r_1024`), fp16 weight = trit x d (EXACT), fp32 accumulation, fp32 output and residual;
     fp32 non-experts. With fp16 GEMMs in the backward pass the loss must be scaled by K (A2): the trainer does it.
   - **A** («like the forge»): de-rotated weight R^T(trit x d) in fp32 cast to bf16, bf16 activations,
     bf16 non-experts.
   - **C** (2026-09-26, speed candidate): experts EXACTLY as in B, non-experts and residual in bf16 as in A,
     fp32 router (F32 weight from the GGUF). In B the non-expert projections run in fp32, without the WMMA path
     of gfx1151. M0 (time) and F0b (fidelity against Vulkan) choose, as between A and B.
   - **B3** (2026-09-27, speed candidate that keeps the precision of B): as B, but the non-expert projections
     (>= 256 rows) in bf16x3 (`e2e_bf16x3`, Henry et al. 2019): 3 bf16 GEMMs with fp32 accumulation, weights split
     once. `B3_ind32` = B3 with the experts' backward in fp32 (like B_ind32). Gate: `tools/cancello_b3.sh`.
5. **Per-layer checkpoint** with the top-k routing SAVED in the first sweep and reused in the recomputation;
   a counter of changes (live top-k of the recomputation differing from the saved one), kept on the device: no
   synchronization in the loop (routing plan and tile tables without .item()).
6. **Context like the teacher**: mode «row» (default: the 4096 row is one causal sequence) or mode
   «segments» (`starts` (B, T) 0/1 from the cu_seqlens: each segment is a sequence of its own, positions from 0,
   GDN state reset), like `e2e_teacher --mode`. The trainer MUST use the teacher's mode (`CacheMaestro.mode`).
7. **No copies or synchronizations in the loop** (2026-09-26): S1024 signs cached per (dtype, device) (before:
   a host->GPU copy at every rotation); FWHT compiled with `torch.compile` (identical arithmetic: S11 bit-exact on
   CPU, `measure` re-checks it on the GPU and disables it if not bit-exact; E2E_ROTATE_COMPILED=0 disables it
   manually); fused combination of the experts (`CombineExperts`: routing weights, gathering and sum in one kernel,
   backward in two; S13 against the torch path; E2E_COMBINE_FUSED=0 disables it); `check` with ONE synchronization;
   `Observer`: fp16 overflow (risk 8) and fractions of the gradient that are zero/subnormal in fp16 (gate G1, S12).

API for the trainer: `Student(layers=range(40), recipe="B", engine_name="triton", device="cuda")`,
`parameters()` (theta), `gradient(ids, loss, starts=...)` or `forward_` / `exits` / `backward_`,
`check()` once per step (non-finite scales, fp16 overflow, routing changes),
`diagnostics_backward()` at step 0 (gate G1), `scale_per_export()` for
`e2e_codes_scale.patch_scale`, `load_embedding()` and `head()` (Q5_K from the GGUF, fp32).
2026-09-27 `train_f32=(families)` (`--allena-f32` of e2e_allena; empty = bit-exact identical): the F32 tensors of the
GGUF for those families (norms, router `ffn_gate_inp`, gate of the shared expert, GDN parameters, output_norm)
become `F32Tensor.master` (file values, fp32) with a gradient; `parameters_f32()`, `keys_f32()`,
`values_f32()` / `set_f32()` (resume), `f32_per_export()` (export), `f32_frozen()`; `check()` also raises
on non-finite masters (or ssm_a >= 0). The HF parameter is recomputed from the master at every call of the
layer: the forward pass and the backward recomputation see the same router (saved routing and changes unchanged).
Tests: `e2e_test_f32.py` (A23-A26).
2026-09-27 `lora=e2e_lora.ConfigLoRA(rank, alpha, families)` (`--lora-nonesperti` of e2e_allena; None/rank 0 =
bit-exact identical, no module touched): the non-expert nn.Linear of the requested families become
`e2e_lora.LinearLoRA` (y = x W^T + s (x A^T) B^T, B = 0 at step 0); `parameters_lora()`, `keys_lora()`,
`values_lora()` / `set_lora()` (resume), `lora_per_export()` (Delta W in file order), `lora_frozen()`.
Tests: e2e_test_lora.py.
2026-09-27 LoRA-GA (`e2e_lora.inizializza_ga`, `--lora-init ga` of e2e_allena): with A0, B0 in the adapters the
forward pass is y = x W^T + s [(x (A-A0)^T) B^T + (x A0^T) (B-B0)^T] (exactly zero at step 0); `values_lora()`
carries them, `set_lora()` restores them (missing = the usual LoRA). Tests: e2e_test_lora_ga.py.
2026-09-28 `dense_ternary=(families)` (`--densi-ternari` of e2e_allena; empty = bit-exact identical): the DENSE
non-expert tensors that are already ternary in the GGUF (attn_q/k/v/output, attn_qkv, attn_gate, ssm_out,
ffn_gate/up/down_shexp in TQ1_0R_1024 / TQ1_0) become SINGLE-expert families (E = 1) inside `ExpertLayer.fam`
after gate/up/down: theta, `keys()`, `parameters()`, `scale_per_export()`, engine V, the registers and the export
treat them like the experts. The HF Linear becomes `TernaryLinear` (ExpertFamily on a one-expert plane, file<->HF
permutations). Tests: e2e_test_densi_ternari.py.

Usage (CPU tests, Triton kernels in the interpreter): `e2e_test_studente.py`.
GPU measurement (M0: only when the GPU is free, `tools/gpu_occupata.sh` at 0):
    python3 e2e_student.py --measure --layers 0,1,2,3 --token 8192,16384,32768 --recipe B,C \
        --engines triton,triton_dense --combine fused,torch --sweep --profile
        # -> measure_m0.json in the e2e_student data directory
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F

from fucina.extdeps import ENV_GGUF_PY, add_path, data_path

HERE = Path(__file__).resolve().parent
add_path(ENV_GGUF_PY)   # gguf-py with the rotated ternary types (TQ1_0R_1024); see fucina.extdeps

from gguf import GGUFReader  # noqa: E402
from fucina.ternary.tq1_0r_1024_codec import S1024  # noqa: E402  (read-only)
from fucina.ternary import scale_codes as CS  # noqa: E402
from fucina.ternary import scale_fields as FIELDS  # noqa: E402
from fucina.kernels import folded_dense as _DP  # noqa: E402  (2026-10-01 folded dense; off by default)

GGUF_THOR_T = Path(os.getenv("E2E_GGUF", data_path("Thor-v2-ThorT-scor150.gguf")))
SRC_HF = Path(os.getenv("THOR_HF_DIR", data_path("thor-src-hf")))     # ONLY the configuration
FAMILIES = {"gate": "ffn_gate_exps", "up": "ffn_up_exps", "down": "ffn_down_exps"}
# 2026-09-28 --densi-ternari (dense-ternary branch): DENSE non-expert tensors that are already ternary in the GGUF
# (variant "A" of the all-ternary model: TQ1_0R_1024 with input 2048/4096, TQ1_0 for ffn_down_shexp) trained as a
# «SINGLE expert» (E = 1): same bytes, same theta (ExpertLayer.theta[family]), same ExpertFamily, same
# engine (and engine V), same registers and same export of the codes. family -> tensor name without "blk.N." and ".weight".
FAMILIES_DENSE = {
    "attn_q": "attn_q", "attn_k": "attn_k", "attn_v": "attn_v", "attn_output": "attn_output",
    "attn_qkv": "attn_qkv", "attn_gate": "attn_gate", "ssm_out": "ssm_out",
    "ffn_gate_shexp": "ffn_gate_shexp", "ffn_up_shexp": "ffn_up_shexp", "ffn_down_shexp": "ffn_down_shexp",
}
# 2026-10-11 E2E-C1 (GOAL C, host C0B): in C0B ssm_alpha / ssm_beta (in_proj_a / in_proj_b of the GDN,
# 2048 -> 32, perm_rows relation like attn_qkv) are also TQ1_0R_1024 (in TT2 they were F32 and trained with --allena-f32).
# With E2E_DENSE_SSM_AB=1 they become two extra dense families (single expert: theta, codes of step V, export), placed
# AFTER the usual ten: the registry indices of the first ten planes stay as always. Off (default) =
# original tree, bit-exact. Tests: e2e_test_ssm_ab.py (CPU).
_SSM_AB = os.environ.get("E2E_DENSE_SSM_AB") == "1"
if _SSM_AB:
    FAMILIES_DENSE.update({"ssm_alpha": "ssm_alpha", "ssm_beta": "ssm_beta"})
GROUPS_DENSE = {
    "attn": ("attn_q", "attn_k", "attn_v", "attn_output"),            # full attention (10 layers)
    "gdn": ("attn_qkv", "attn_gate", "ssm_out") + (("ssm_alpha", "ssm_beta") if _SSM_AB else ()),   # Gated DeltaNet (30)
    "shexp": ("ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp"),      # shared expert (40 layers)
    "all_": tuple(FAMILIES_DENSE),
}
# 2026-09-29 second ternary plane of the dense tensors (second-plane-e2e branch; engine: llamacpp-tq1-k1024 fork,
# second-plane-dense-all branch 8d2506e): tensors blk.N.<name>2.weight, same type and shape as the first plane, output
# W1 x + W2 x. They are not requested separately: with --densi-ternari, the second plane of a requested family is
# trained automatically IF it is in the GGUF (its own theta scales, codes of step V as a family of its own, export);
# without *2 in the GGUF nothing changes (bit-exact with 7d92c83). On the dequantized path (dense family NOT
# requested) the HF Linear receives W1 + W2 (fp32 sum of the two dequantizations).
SUFFIX_P2 = "2"
FAMILIES_DENSE2 = {f"{k}{SUFFIX_P2}": f"{v}{SUFFIX_P2}" for k, v in FAMILIES_DENSE.items()}
# 2026-09-30 THIRD plane of the dense tensors (tt5-third-plane branch from cfb718f; R2B3/R2B3m, engine third-plane-dense
# 0ed1faa): tensors blk.N.<name>3.weight, same type and shape as the first plane, output W1 x + W2 x + W3 x (sums in the
# engine's order). Same machinery as the second plane: a family of its own (theta, codes of step V, export), after ALL
# the *2 in the keys; on the dequantized path the HF Linear receives W1 + W2 + W3. Without *3 in the GGUF nothing
# changes (bit-exact with cfb718f).
SUFFIX_P3 = "3"
FAMILIES_DENSE3 = {f"{k}{SUFFIX_P3}": f"{v}{SUFFIX_P3}" for k, v in FAMILIES_DENSE.items()}
# 2026-10-09 (fix P1 of RILQT/COMBO_A_shexp/DIAGNOSI_DIVARIO.md) FOURTH plane of the dense tensors (R2B4 / PV12P /
# COMBO A; engine thor-e1-shexp qwen35moe.cpp: wq4/wqkv4/wqkv_gate4/ssm_out4/ffn_*_shexp4, `cur += mm(W4, x)` AFTER the
# third plane and BEFORE the _lra/_lrb residuals). Same machinery as the third: a family of its own after ALL the *3,
# output W1x+W2x+W3x+W4x; on the dequantized path the HF Linear receives ((W1 + W2) + W3) + W4. Without *4 in the
# GGUF nothing changes (bit-exact).
SUFFIX_P4 = "4"
FAMILIES_DENSE4 = {f"{k}{SUFFIX_P4}": f"{v}{SUFFIX_P4}" for k, v in FAMILIES_DENSE.items()}
# 2026-10-09 (fix P1) SECOND plane of the routed experts (PV12 / COMBO A; engine llama-graph.cpp build_moe_ffn
# `second_plane`): ffn_{gate,up,down}_exps2 (E2 slots) + F32 map ffn_X_exps2_map [n_expert] (slot of the expert in the
# 2nd plane, -1 = none). Output per assignment (token, expert e): mm(W_e, x) + [map[e] >= 0] mm(W2[map[e]], x), for
# gate, up and down separately (each with its own map), BEFORE silu*up and BEFORE the router weight. Without a map: a
# prefix of the first E2 experts (id < E2). ODINO_P2_COMP (compensa_freddi) does not apply with the map and is not
# used by the runs.
SUFFIX_E2 = "2"


def e_dense(n: str) -> bool:
    """Single-expert dense family (first, second, third or fourth plane)."""
    return n in FAMILIES_DENSE or n in FAMILIES_DENSE2 or n in FAMILIES_DENSE3 or n in FAMILIES_DENSE4


def first_plane(n: str) -> str:
    """"attn_q2" -> "attn_q"; a first-plane family stays itself."""
    if n in FAMILIES_DENSE2:
        return n[:-len(SUFFIX_P2)]
    if n in FAMILIES_DENSE4:                      # 2026-10-09 fourth plane
        return n[:-len(SUFFIX_P4)]
    return n[:-len(SUFFIX_P3)] if n in FAMILIES_DENSE3 else n


def families_dense(entries) -> tuple:
    """"a,b" or an iterable (families or groups attn/gdn/shexp/all_) -> canonical tuple (order of FAMILIES_DENSE).
    Empty = none (default: bit-exact identical student)."""
    if isinstance(entries, str):
        entries = entries.split(",")
    requested = set()
    for v in entries:
        v = v.strip()
        if not v:
            continue
        if v.endswith(".weight"):
            v = v[:-len(".weight")]
        if v in GROUPS_DENSE:
            requested.update(GROUPS_DENSE[v])
        elif v in FAMILIES_DENSE2 or v in FAMILIES_DENSE3 or v in FAMILIES_DENSE4:   # 2026-09-29 (third 09-30, fourth 10-09)
            requested.add(first_plane(v))
        elif v in FAMILIES_DENSE:
            requested.add(v)
        else:
            raise ValueError(f"unknown dense family {v!r}: {sorted(FAMILIES_DENSE) + sorted(GROUPS_DENSE)}")
    return tuple(f for f in FAMILIES_DENSE if f in requested)


def family_tensor_name(n: str) -> str:
    """Family (experts gate/up/down or dense) -> GGUF tensor name without "blk.N." and without ".weight"."""
    return FAMILIES[n] if n in FAMILIES else FAMILIES_DENSE[n] if n in FAMILIES_DENSE else \
        FAMILIES_DENSE2[n] if n in FAMILIES_DENSE2 else FAMILIES_DENSE3[n] if n in FAMILIES_DENSE3 else \
        FAMILIES_DENSE4[n]


def families_dense_from_names(names) -> tuple:
    """GGUF tensor names ("blk.12.attn_q.weight", ...) -> dense families present (canonical order). Used by whoever
    rebuilds the student from a snapshot or a registry (export, reload): the scales tell which dense tensors existed."""
    per_suffix = {f"{v}.weight": k for k, v in FAMILIES_DENSE.items()}
    per_suffix.update({f"{v}.weight": first_plane(k) for k, v in FAMILIES_DENSE2.items()})   # 09-29: *2 -> first plane
    per_suffix.update({f"{v}.weight": first_plane(k) for k, v in FAMILIES_DENSE3.items()})   # 09-30: *3 -> first plane
    per_suffix.update({f"{v}.weight": first_plane(k) for k, v in FAMILIES_DENSE4.items()})   # 10-09: *4 -> first plane
    found = set()
    for n in names:
        parts = str(n).split(".", 2)
        if len(parts) == 3 and parts[0] == "blk" and parts[2] in per_suffix:
            found.add(per_suffix[parts[2]])
    return tuple(f for f in FAMILIES_DENSE if f in found)

try:  # Triton is in the forge venv; without it, only the torch engine (CPU) remains
    import triton
    import triton.language as tl
    _TRITON = True
except Exception:  # pragma: no cover
    _TRITON = False

if _TRITON and os.getenv("TRITON_INTERPRET") == "1":
    # Only for CPU tests (Triton interpreter). 2026-09-26: with numpy 2.5 the Triton 3.6 interpreter fails on
    # `range(0, K, BK)` with a runtime K: its scalars are (1,) arrays and `int()` on a non-0-d array now raises
    # («only 0-dimensional arrays can be converted to Python scalars»). `__index__` is restored on the single
    # element; no effect outside the interpreter (the compiled kernel does not go through here).
    from triton.runtime import interpreter as _interp

    _orig_patch = _interp._patch_lang_tensor

    def _patch_lang_tensor_np2(tensor, scope):
        _orig_patch(tensor, scope)
        scope.set_attr(tensor, "__index__", lambda self: int(np.asarray(self.handle.data).reshape(-1)[0]))

    _interp._patch_lang_tensor = _patch_lang_tensor_np2


# ═══════════════════════════════════ recipes and formats ═══════════════════════════════════
@dataclass(frozen=True)
class Recipe:
    """dtype_residual: residual stream and non-experts. dtype_gemm: inputs of the expert products (the
    outputs are fp32 if dtype_gemm is fp16, otherwise dtype_gemm). rotate: True = the ACTIVATION is rotated and the
    weight stays trit x d in the rotated domain (recipe B); False = dense de-rotated weight (recipe A)."""
    name: str
    dtype_residual: torch.dtype
    dtype_gemm: torch.dtype
    rotate: bool
    dtype_backward: Optional[torch.dtype] = None     # None = dtype_gemm; fp32 = reference backward (tests)
    router_fp32: bool = False                        # router in fp32 (F32 weight from the GGUF) even with a bf16 residual
    non_experts_x3: bool = False                     # 09-27: non-expert projections in bf16x3 (e2e_bf16x3), recipe B3
    non_experts_k2: bool = False                     # 09-27: the same in fp32 with K split in two halves (noise arm, Bk2)
    # 2026-09-27 (backward-bf16x3 branch): backward of the EXPERTS with hi/lo split products using the fp16/bf16 kernels
    # of recipe B ("fp16x2" | "bf16x3", `e2e_backward_hilo`); "" = off. Applies only with dtype_backward None: the
    # fp32 reference recipe of G1 (dtype_backward fp32) goes back to the exact fp32 backward.
    backward_split: str = ""

    @property
    def dtype_output(self) -> torch.dtype:
        return torch.float32 if self.dtype_gemm == torch.float16 else self.dtype_gemm


RECIPES = {
    "A": Recipe("A", torch.bfloat16, torch.bfloat16, False),
    "B": Recipe("B", torch.float32, torch.float16, True),
    # tests only: the same algebra in float64 / float32, for tight comparisons with the reference
    "B64": Recipe("B64", torch.float64, torch.float64, True),
    "A64": Recipe("A64", torch.float64, torch.float64, False),
    "B32": Recipe("B32", torch.float32, torch.float32, True),
    "A32": Recipe("A32", torch.float32, torch.float32, False),
    # same forward as B, backward (dX and dL/dd) with fp32 inputs: the reference of G1 / A2 for the K scaling
    "B_ind32": Recipe("B_ind32", torch.float32, torch.float16, True, torch.float32),
    # 2026-09-26, SPEED candidate for M0/F0b (chosen by the measurement, like A and B): experts EXACTLY as in B
    # (fp16 rotated activation, fp16 trit x d, fp32 accumulation, fp16 backward with the K scaling), non-experts and
    # residual in bf16 as in A (the forge). Reason: in B the attention/GDN projections, the shared expert and the GDN
    # run in fp32, and gfx1151 has no WMMA path for fp32 (§2.6 rev.): that is ~35 M parameters per layer, i.e.
    # as many FLOPs as the active experts. The router stays fp32 (its weight in the GGUF is F32; bf16 would move
    # top-k choices that llama.cpp does not move).
    "C": Recipe("C", torch.bfloat16, torch.float16, True, None, True),
    # 2026-09-27, SPEED candidate that does not touch the precision of B: everything as in B (fp32 residual, exact fp16
    # experts, fp32 router), but the non-expert projections with >= 256 rows (attention q/k/v/o, GDN in_proj_qkv/in_proj_z/
    # out_proj, shared expert) in bf16x3 (`e2e_bf16x3`: 3 bf16 GEMMs with fp32 accumulation, ~1e-5 relative error
    # instead of ~1e-6 for the fp32 GEMM, test X1). B3_ind32 = B3 with the experts' backward in fp32, like B_ind32 for B.
    # Chosen by the gate `tools/cancello_b3.sh` (time, KL(B||B3), dL/dtheta against B_ind32).
    "B3": Recipe("B3", torch.float32, torch.float16, True, None, False, True),
    "B3_ind32": Recipe("B3_ind32", torch.float32, torch.float16, True, torch.float32, False, True),
    # 2026-09-27 afternoon, ONLY for the B3 gate ("noise" arm): B_ind32 with the same non-expert projections in
    # fp32 but the sum over K split in two halves (`e2e_bf16x3.K2Linear`): a different fp32 rounding, nothing else.
    # Measures how much KL and dL/dtheta of B move on their own over the 40 layers (top-k routing + recurrences).
    "Bk2_ind32": Recipe("Bk2_ind32", torch.float32, torch.float16, True, torch.float32, False, False, True),
    # 2026-09-27 (backward-bf16x3 branch), SPEED candidates for the backward pass: forward EXACTLY as in B (B_ind*) or
    # B3 (B3_ind*), experts' backward with split products (`e2e_backward_hilo`) instead of the fp32 products of
    # B_ind32 / B3_ind32: "fp16x2" = dY scaled by powers of two (per row for dX, per expert x row of W for dL/dd) and
    # split into two fp16, W and X already exact in fp16 (2 fp16 WMMA products); "bf16x3" = three bf16 products. Chosen
    # by the gate `tools/cancello_indietro.sh` (time, dL/dtheta against B_ind32 on the SAME forward).
    "B_ind2": Recipe("B_ind2", torch.float32, torch.float16, True, None, False, False, False, "fp16x2"),
    "B3_ind2": Recipe("B3_ind2", torch.float32, torch.float16, True, None, False, True, False, "fp16x2"),
    "B_ind3": Recipe("B_ind3", torch.float32, torch.float16, True, None, False, False, False, "bf16x3"),
    "B3_ind3": Recipe("B3_ind3", torch.float32, torch.float16, True, None, False, True, False, "bf16x3"),
}


@dataclass
class FamilyData:
    """One family of experts of ONE layer: bytes (E, R, bytes_per_row) uint8 exactly as they are in the GGUF
    (numpy order: expert, output row, blocks of the row), initial scales d0 (E, R, ng) fp32."""
    name: str
    type_: str               # "TQ1_0R_1024" | "TQ1_0"
    blocks: torch.Tensor   # uint8 (E, R, NBR)
    d0: torch.Tensor        # fp32 (E, R, NG)
    layer: int = -1        # layer index (from the GGUF tensor name), for diagnostics

    @property
    def E(self) -> int:
        return self.blocks.shape[0]

    @property
    def R(self) -> int:
        return self.blocks.shape[1]

    @property
    def NBR(self) -> int:
        return self.blocks.shape[2]

    @property
    def nb(self) -> int:
        return CS.LAYOUT[self.type_][0]

    @property
    def qk(self) -> int:
        return CS.LAYOUT[self.type_][1]

    @property
    def gs(self) -> int:
        return CS.LAYOUT[self.type_][1] // CS.LAYOUT[self.type_][3]

    @property
    def C(self) -> int:
        return (self.NBR // self.nb) * self.qk

    @property
    def NG(self) -> int:
        return self.C // self.gs

    @property
    def rotated(self) -> bool:
        return self.type_ == "TQ1_0R_1024"


def family_from_tensor(t, name: str, device="cpu", experts: Optional[slice] = None,
                        rows: Optional[slice] = None) -> FamilyData:
    """GGUF ReaderTensor of experts -> FamilyData. `experts`/`rows`: subset (tests only).
    2026-09-28 --densi-ternari: a 2D DENSE tensor (R, NBR) becomes the family of ONE expert (1, R, NBR), same bytes
    (a view of the memmap as for the experts); `experts` does not apply (there is only one expert)."""
    type_ = t.tensor_type.name
    raw = np.asarray(t.data)                                   # memmap (E, R, NBR)
    if raw.ndim == 2:
        raw = raw[None]
    elif experts is not None:
        raw = raw[experts]
    if rows is not None:
        raw = raw[:, rows]
    B = torch.from_numpy(np.ascontiguousarray(raw))
    nb = CS.LAYOUT[type_][0]
    d0 = CS.scale_from_blocks(B.reshape(-1, nb), type_).to(torch.float32).reshape(B.shape[0], B.shape[1], -1)
    parts = str(t.name).split(".")
    L = int(parts[1]) if len(parts) > 1 and parts[0] == "blk" and parts[1].isdigit() else -1
    return FamilyData(name, type_, B.to(device), d0.to(device), L)


# ═══════════════════════════════════ torch reference ═══════════════════════════════════
def trit_expert(fam: FamilyData, e: int) -> torch.Tensor:
    """Trits (R, C) int8 of expert e, from its own bytes only (device-agnostic decoder of e2e_fields_scale)."""
    b = fam.blocks[e].reshape(-1, fam.nb)
    return FIELDS.trit_torch(b, fam.type_).reshape(fam.R, fam.C)


_S_CACHE: dict = {}


def _S(dtype, device) -> torch.Tensor:
    """S1024 signs in the requested dtype/device, built ONCE per pair (2026-09-26: before, `from_numpy(S1024).to(device)`
    was redone at every rotation, i.e. a synchronous host->GPU copy from pageable memory inside every layer of the
    loop)."""
    key = (dtype, torch.device(device))
    s = _S_CACHE.get(key)
    if s is None:
        s = _S_CACHE[key] = torch.from_numpy(S1024).to(device=device, dtype=dtype)
    return s


# Compiled rotation (2026-09-26, speed lever A3/«no copies»): the eager FWHT materializes 10 tensors as large
# as the activation (one per stage) in the forward pass and as many in the backward pass; `torch.compile` fuses it.
# Identical arithmetic (only sums/differences and an exact *2^-5, no reassociation): S11 tests it BIT-EXACT on CPU
# (inductor), `measure` re-checks it on the GPU before timing. None = automatic (cuda only);
# E2E_ROTATE_COMPILED=0 disables it.
COMPILE_ROTATE: Optional[bool] = None if os.getenv("E2E_ROTATE_COMPILED", "1") == "1" else False
_ROT_COMPILED: dict = {}


def _function_rotation(name: str, device: torch.device):
    from fucina.ternary import gpu_rotation as RG  # read-only
    f = getattr(RG, name)
    compile_ = COMPILE_ROTATE if COMPILE_ROTATE is not None else device.type == "cuda"
    if not compile_:
        return f
    if name not in _ROT_COMPILED:
        _ROT_COMPILED[name] = torch.compile(f, dynamic=True)
    return _ROT_COMPILED[name]


def rotate(x: torch.Tensor) -> torch.Tensor:
    """R x = FWHT(S (.) x) / 32 in blocks of 1024 along the last axis (differentiable). Same arithmetic
    as `rotation_gpu._R_1024_torch` and the shader `rot_y_tq1_0r_1024` (S, then H, with 1/32 at the start)."""
    return _function_rotation("_R_1024_torch", x.device)(x, _S(x.dtype, x.device))


def de_rotate(w: torch.Tensor) -> torch.Tensor:
    """R^T w (rows of weights from the rotated domain to the real one): `rotation_gpu._RT_1024_torch`."""
    return _function_rotation("_RT_1024_torch", w.device)(w, _S(w.dtype, w.device))


def weight_expert_torch(fam: FamilyData, e: int, d: torch.Tensor, recipe: Recipe) -> torch.Tensor:
    """Weight (R, C) of expert e in the dtype of the products. Recipe with `rotate`: trit x d in the rotated domain
    (exact in fp16 because d is fp16). Recipe without: R^T(trit x d) in fp32 (or fp64), then dtype_gemm."""
    q = trit_expert(fam, e)
    calc = torch.float64 if recipe.dtype_gemm == torch.float64 else torch.float32
    W = (q.to(calc).view(fam.R, fam.NG, fam.gs) * d[e].to(calc).unsqueeze(-1)).view(fam.R, fam.C)
    if fam.rotated and not recipe.rotate:
        W = de_rotate(W)
    return W.to(recipe.dtype_gemm)


# ═══════════════════════════════════ routing plan ═══════════════════════════════════
class Plane:
    """(token, slot) assignments sorted by expert, all on the device and WITHOUT synchronizations:
    order (S,) = stable argsort of the experts, inverse (S,), offs (E+1,) from searchsorted.
    The Triton tile tables (one per height BS) are built on demand, also without sync:
    the grid is the upper bound S//BS + E and the extra tiles exit immediately (expert -1)."""

    def __init__(self, idx: torch.Tensor, E: int):
        self.T, self.k = idx.shape
        self.E = E
        flat = idx.reshape(-1)
        self.order = torch.argsort(flat, stable=True)
        self.inverse = torch.empty_like(self.order)
        self.inverse[self.order] = torch.arange(self.order.numel(), device=idx.device)
        sorted_ = flat[self.order]
        self.offs = torch.searchsorted(sorted_, torch.arange(E + 1, device=idx.device, dtype=sorted_.dtype))
        self.S = flat.numel()
        self._tables: dict = {}
        self._counts: Optional[list] = None

    def counts(self) -> list:
        """Only for the torch engine (CPU): one synchronization, once per plan."""
        if self._counts is None:
            o = self.offs.tolist()
            self._counts = [o[e + 1] - o[e] for e in range(self.E)]
        return self._counts

    def table(self, BS: int) -> tuple:
        if BS not in self._tables:
            cnt = self.offs[1:] - self.offs[:-1]
            nt = (cnt + BS - 1) // BS
            cum = torch.cumsum(nt, 0)
            start = cum - nt
            n_max = self.S // BS + self.E
            t = torch.arange(n_max, device=self.offs.device)
            e = torch.searchsorted(cum, t, right=True)
            valid = e < self.E
            ec = torch.clamp(e, max=self.E - 1)
            s = self.offs[ec] + (t - start[ec]) * BS
            self._tables[BS] = (torch.where(valid, e, -1).to(torch.int32).contiguous(),
                                 torch.where(valid, s, 0).to(torch.int32).contiguous(), n_max)
        return self._tables[BS]


# 2026-09-29 tt3-fixes branch (--pv-riuso, e2e_riuso_prove.RiusoProve): reuse inside step V, where theta, F32 and LoRA
# are FROZEN and only the bytes of the codes change. Off (default) = none of these paths, bit-exact identical.
#   plane_single  the single-expert Plane of the dense tensors (all tokens to expert 0) depends only on (tokens, device):
#                one per shape instead of one per call (250 x 2 dense x forward and recomputation);
#   piano_instr  the forward Plane of a MoE layer stays attached to its saved top-k (`last_idx`): the backward
#                recomputation with that same top-k reuses it instead of rebuilding it (same tables, same values);
#   scale        the scale fp16(d0 exp(theta)) of each family, outside the graph (forward without gradient), is
#                recomputed only if theta, d0, the sentinel or the dtype change (key: identity and version of the tensors).
REUSE = {"active": False, "planes_unique": {}, "scale": {}}


def reuse_active(on: bool) -> None:
    """Turns reuse on/off (and empties the caches: no value outlives whoever turned it on)."""
    REUSE["active"] = bool(on)
    REUSE["planes_unique"].clear()
    REUSE["scale"].clear()


def plane_single(T: int, device) -> "Plane":
    """Single-expert Plane for T tokens (reused if reuse is on)."""
    if not REUSE["active"]:
        return Plane(torch.zeros(T, 1, dtype=torch.long, device=device), 1)
    k = (int(T), str(device))
    p = REUSE["planes_unique"].get(k)
    if p is None:
        p = REUSE["planes_unique"][k] = Plane(torch.zeros(T, 1, dtype=torch.long, device=device), 1)
    return p


# ═══════════════════════════════════ kernel Triton ═══════════════════════════════════
if _TRITON:
    @triton.jit
    def _trit_from_byte(pb, r, c, NBR, R, C, QK: tl.constexpr, NB: tl.constexpr, ROT: tl.constexpr):
        """Trit (int32 in {-1,0,1}, 0 out of range) of weight (r, c) of ONE expert (`pb` points to its bytes).
        Layout of ggml-common.h (k1024): TQ1_0 qs[48] qh[4] d; TQ1_0R_1024 = 4 TQ1_0 quarters (qs[4][48],
        qh[4][4]) + d[8]. Within a quarter: bytes 0..31 -> elements n*32+m, 32..47 -> 160+n*16+m, qh -> 240+n*4+m."""
        blk = c // QK
        p = c % QK
        if ROT:
            q = p // 256
            j = p % 256
            qs0 = q * 48
            qh0 = 192 + q * 4
        else:
            j = p
            qs0 = 0
            qh0 = 48
        in1 = j < 160
        in2 = j < 240
        j2 = j - 160
        j3 = j - 240
        bi = tl.where(in1, qs0 + (j % 32), tl.where(in2, qs0 + 32 + (j2 % 16), qh0 + (j3 % 4)))
        n = tl.where(in1, j // 32, tl.where(in2, j2 // 16, j3 // 4))
        pw = tl.where(n == 0, 1, tl.where(n == 1, 3, tl.where(n == 2, 9, tl.where(n == 3, 27, 81))))
        m = (r < R) & (c < C)
        v = tl.load(pb + r.to(tl.int64) * NBR + blk * NB + bi, mask=m, other=0).to(tl.int32)
        t = ((((v * pw) & 255) * 3) >> 8) - 1
        return tl.where(m, t, 0), m

    @triton.jit
    def _k_gemm(a_ptr, sa_s, sa_k,
                pb, sb_e, NBR,
                d_ptr, sd_e, NG,
                w_ptr, sw_e, sw_r,
                o_ptr, so_s, so_n,
                te_ptr, ts_ptr, offs_ptr,
                R, C, K, N,
                GS: tl.constexpr, QK: tl.constexpr, NB: tl.constexpr, ROT: tl.constexpr,
                TRANSP: tl.constexpr, DENSE: tl.constexpr,
                BS: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        """Product grouped by expert. TRANSP=0: out[s, r] = sum_c A[s, c] W_e[r, c] (forward);
        TRANSP=1: out[s, c] = sum_r A[s, r] W_e[r, c] (backward dX). DENSE=0: W decoded from the bytes inside the
        tile (trit x d, exact); DENSE=1: W read from an (E, R, C) tensor. fp32 accumulation."""
        pid_t = tl.program_id(0)
        pid_n = tl.program_id(1)
        e = tl.load(te_ptr + pid_t)
        if e < 0:
            return
        s0 = tl.load(ts_ptr + pid_t)
        s_end = tl.load(offs_ptr + e + 1)
        e64 = e.to(tl.int64)
        rs = s0 + tl.arange(0, BS)
        ms = rs < s_end
        rn = pid_n * BN + tl.arange(0, BN)
        mn = rn < N
        acc = tl.zeros((BS, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            rk = k0 + tl.arange(0, BK)
            mk = rk < K
            a = tl.load(a_ptr + rs[:, None].to(tl.int64) * sa_s + rk[None, :] * sa_k,
                        mask=ms[:, None] & mk[None, :], other=0.0)
            if TRANSP:
                r2 = rk[:, None]
                c2 = rn[None, :]
            else:
                r2 = rn[None, :]
                c2 = rk[:, None]
            if DENSE:
                b = tl.load(w_ptr + e64 * sw_e + r2.to(tl.int64) * sw_r + c2, mask=(r2 < R) & (c2 < C), other=0.0)
            else:
                t, m2 = _trit_from_byte(pb + e64 * sb_e, r2, c2, NBR, R, C, QK, NB, ROT)
                dv = tl.load(d_ptr + e64 * sd_e + r2.to(tl.int64) * NG + c2 // GS, mask=m2, other=0.0)
                b = t.to(tl.float32) * dv
            acc += tl.dot(a, b.to(a.dtype))
        tl.store(o_ptr + rs[:, None].to(tl.int64) * so_s + rn[None, :] * so_n,
                 acc.to(o_ptr.dtype.element_ty), mask=ms[:, None] & mn[None, :])

    @triton.jit
    def _k_dd(dy_ptr, sdy_s, sdy_r, x_ptr, sx_s, sx_c,
              pb, sb_e, NBR, dd_ptr, sdd_e, NG, offs_ptr, R, C,
              GS: tl.constexpr, QK: tl.constexpr, NB: tl.constexpr, ROT: tl.constexpr,
              BR: tl.constexpr, BS: tl.constexpr):
        """dL/dd[e, r, g] = sum_{c in g} trit[e, r, c] * sum_{s in e} dY[s, r] X[s, c], fp32. The (BR x GS) block
        of dY^T X never leaves the registers: the trits are decoded from the bytes at the end."""
        e = tl.program_id(0)
        pid_r = tl.program_id(1)
        g = tl.program_id(2)
        s_ini = tl.load(offs_ptr + e)
        s_end = tl.load(offs_ptr + e + 1)
        rr = pid_r * BR + tl.arange(0, BR)
        cc = g * GS + tl.arange(0, GS)
        mr = rr < R
        acc = tl.zeros((BR, GS), dtype=tl.float32)
        for s0 in range(s_ini, s_end, BS):
            rs = s0 + tl.arange(0, BS)
            ms = rs < s_end
            dyt = tl.load(dy_ptr + rs[None, :].to(tl.int64) * sdy_s + rr[:, None] * sdy_r,
                          mask=ms[None, :] & mr[:, None], other=0.0)
            x = tl.load(x_ptr + rs[:, None].to(tl.int64) * sx_s + cc[None, :] * sx_c,
                        mask=ms[:, None], other=0.0)
            acc += tl.dot(dyt, x)
        e64 = e.to(tl.int64)
        t, m2 = _trit_from_byte(pb + e64 * sb_e, rr[:, None], cc[None, :], NBR, R, C, QK, NB, ROT)
        val = tl.sum(acc * t.to(tl.float32), axis=1)
        tl.store(dd_ptr + e64 * sdd_e + rr * NG + g, val, mask=mr)

    @triton.jit
    def _k_decode(pb, sb_e, NBR, d_ptr, sd_e, NG, w_ptr, sw_e, sw_r, R, C,
                      GS: tl.constexpr, QK: tl.constexpr, NB: tl.constexpr, ROT: tl.constexpr,
                      BR: tl.constexpr, BC: tl.constexpr):
        """bytes + scales -> W (E, R, C) = trit x d in the dtype of w_ptr, in a single pass (fused decode)."""
        e64 = tl.program_id(0).to(tl.int64)
        rr = tl.program_id(1) * BR + tl.arange(0, BR)
        cc = tl.program_id(2) * BC + tl.arange(0, BC)
        r2 = rr[:, None]
        c2 = cc[None, :]
        t, m2 = _trit_from_byte(pb + e64 * sb_e, r2, c2, NBR, R, C, QK, NB, ROT)
        dv = tl.load(d_ptr + e64 * sd_e + r2.to(tl.int64) * NG + c2 // GS, mask=m2, other=0.0)
        w = t.to(tl.float32) * dv
        tl.store(w_ptr + e64 * sw_e + r2.to(tl.int64) * sw_r + c2, w.to(w_ptr.dtype.element_ty), mask=m2)

    # ─────────────── engine «triton2» (2026-09-27): same math, decode in pieces ───────────────
    # A 256-column sub-block (one TQ1_0 block, or one quarter of TQ1_0R_1024) is split into 8 PIECES of 32
    # columns, following the FIXED scheme of ggml-common.h:
    #   pieces 0..4: columns 32p..32p+31 = bytes qs[0:32] (contiguous) with multiplier 3^p (one per piece);
    #   pieces 5, 6: columns 160..223     = bytes qs[32:48] (16 contiguous, each used twice), 3^0|3^1 and 3^2|3^3;
    #   piece 7   : columns 224..239 = qs[32:48] x 3^4, 240..255 = qh[m] x 3^n (column 240 + 4n + m).
    # In _k_dx2 and _k_dd2 (pieces along the program's columns) the 32 bytes of qs[0:32] are read ONCE for
    # 5 pieces (contiguous load), the multiplier is a constant or a fixed vector of 32 and the scale a per-row
    # vector loaded once per group (GS = 128 or 256 contains 4 or 8 whole pieces). In _k_gemm2 (pieces
    # along K) the 8 unrolled pieces spilled 150-600 registers on gfx1151: there the loop steps by 32 with
    # the piece scheme computed on 1D vectors of 32 (bytes and multiplier per column, per-row scale once
    # per piece): per weight there remain 1 load + 5 integer operations + conversion and product, without the
    # divisions, the chains of where and the per-element int64 addresses of `_trit_from_byte` (114-165 VGPR, 0 spill).
    # No byte -> 5 trit table: on the GPU an indexed load per weight costs more than 4 integer operations.
    # The weight value is the same as `_trit_from_byte` x d (t in {-1, 0, 1}: fp16(t d) = t fp16(d), including
    # the sign of zero) and the order of the products along K is that of `_k_gemm` with BK = 32.
    @triton.jit
    def _dec_trit(v, pw):
        """((((b * 3^n) & 255) * 3) >> 8) - 1 (dequantize_row_tq1_0), v int32, pw scalar or tensor."""
        return ((((v * pw) & 255) * 3) >> 8) - 1

    @triton.jit
    def _long_col(v, TR: tl.constexpr):
        """Vector over the 32 columns of the piece -> tile shape (TR=1: columns on axis 0)."""
        if TR:
            r = v[:, None]
        else:
            r = v[None, :]
        return r

    @triton.jit
    def _long_row(v, TR: tl.constexpr):
        """Vector over the weight rows -> tile shape (TR=1: rows on axis 1)."""
        if TR:
            r = v[None, :]
        else:
            r = v[:, None]
        return r

    @triton.jit
    def _t_piece(v0, v5, v7, pw5, pw6, pw7, P: tl.constexpr, TR: tl.constexpr):
        """int32 trit of piece P from already loaded bytes (v0: qs[0:32]; v5: qs[32 + i%16]; v7: piece 7)."""
        if P == 0:
            t = _dec_trit(v0, 1)
        elif P == 1:
            t = _dec_trit(v0, 3)
        elif P == 2:
            t = _dec_trit(v0, 9)
        elif P == 3:
            t = _dec_trit(v0, 27)
        elif P == 4:
            t = _dec_trit(v0, 81)
        elif P == 5:
            t = _dec_trit(v5, _long_col(pw5, TR))
        elif P == 6:
            t = _dec_trit(v5, _long_col(pw6, TR))
        else:
            t = _dec_trit(v7, _long_col(pw7, TR))
        return t

    @triton.jit
    def _w_piece(v0, v5, v7, dA, dB, pw5, pw6, pw7, P: tl.constexpr, GS: tl.constexpr, TR: tl.constexpr,
                 DT: tl.constexpr):
        """Weight trit x d of piece P in dtype DT. dA: scale of the group of pieces 0..3 (and 4..7 if GS = 256),
        dB: of the group of pieces 4..7 if GS = 128. Already in DT (exact conversion, see above)."""
        t = _t_piece(v0, v5, v7, pw5, pw6, pw7, P, TR)
        if GS == 128:
            if P >= 4:
                dv = dB
            else:
                dv = dA
        else:
            dv = dA
        return t.to(DT) * _long_row(dv, TR)

    @triton.jit
    def _schema_subblock(c0, QK: tl.constexpr, NB: tl.constexpr, ROT: tl.constexpr):
        """Offsets (bytes, within the row) of qs and qh of the 256-column sub-block starting at column c0."""
        b0 = (c0 // QK) * NB
        if ROT:
            q = (c0 % QK) // 256
            qs = b0 + q * 48
            qh = b0 + 192 + q * 4
        else:
            qs = b0
            qh = b0 + 48
        return qs, qh

    @triton.jit
    def _order_groups(pid, NT, NN, GROUP_M: tl.constexpr):
        """pid -> (token tile, output tile): GROUP_M consecutive token tiles (mostly of the SAME expert) sweep
        the output tiles together, so the bytes, scales and rows of A stay in L2. Only the execution order
        changes: each tile does the same computations."""
        per_g = GROUP_M * NN
        t_ini = (pid // per_g) * GROUP_M
        gm = tl.minimum(NT - t_ini, GROUP_M)
        loc = pid % per_g
        return t_ini + loc % gm, loc // gm

    @triton.jit
    def _k_gemm2(a_ptr, sa_s,
                 pb, sb_e, NBR, d_ptr, sd_e, NG, o_ptr, so_s,
                 pb2, sb2_e, d2_ptr, sd2_e, o2_ptr,
                 te_ptr, ts_ptr, offs_ptr, R, C, NT,
                 GS: tl.constexpr, QK: tl.constexpr, NB: tl.constexpr, ROT: tl.constexpr, DUE: tl.constexpr,
                 BS: tl.constexpr, BN: tl.constexpr, GROUP_M: tl.constexpr):
        """Forward: out[s, r] = sum_c A[s, c] W_e[r, c], like `_k_gemm` TRANSP=0 (one product per step of 32
        columns, in the same order as BK = 32). DUE=1: the same sweep for a second family of the same shape
        (gate and up) on the SAME tile of A: two accumulators, A read once. Addresses = scalar base
        (int64) + int32 offsets of the tile, loop-invariant: few vector registers."""
        DT = a_ptr.dtype.element_ty
        pid_t, pid_n = _order_groups(tl.program_id(0), NT, tl.cdiv(R, BN), GROUP_M)
        e = tl.load(te_ptr + pid_t)
        if e < 0:
            return
        s0 = tl.load(ts_ptr + pid_t)
        s_end = tl.load(offs_ptr + e + 1)
        e64 = e.to(tl.int64)
        ls = tl.arange(0, BS)
        ms = (s0 + ls) < s_end
        ln = tl.arange(0, BN)
        r0 = pid_n * BN
        mn = (r0 + ln) < R
        i = tl.arange(0, 32)
        pa = a_ptr + s0.to(tl.int64) * sa_s
        offa = (ls * sa_s)[:, None] + i[None, :]
        pbw = pb + e64 * sb_e + r0.to(tl.int64) * NBR
        pbw2 = pb2 + e64 * sb2_e + r0.to(tl.int64) * NBR
        pdd = d_ptr + e64 * sd_e + r0.to(tl.int64) * NG
        pdd2 = d2_ptr + e64 * sd2_e + r0.to(tl.int64) * NG
        offr = (ln * NBR)[None, :]
        offd = ln * NG
        m2 = mn[None, :]
        acc = tl.zeros((BS, BN), dtype=tl.float32)
        acc2 = tl.zeros((BS, BN), dtype=tl.float32)
        for k in range(0, C, 32):
            # scheme of piece (k % 256) // 32 over 32 columns: 1D vectors (negligible cost compared to the tile)
            qs, qh = _schema_subblock(k, QK, NB, ROT)
            j = (k % 256) + i
            n1 = j // 32
            n2 = (j - 160) // 16
            n3 = (j - 240) // 4
            n = tl.where(j < 160, n1, tl.where(j < 240, n2, n3))
            pw = tl.where(n == 0, 1, tl.where(n == 1, 3, tl.where(n == 2, 9, tl.where(n == 3, 27, 81))))
            ob = tl.where(j < 160, qs + i, tl.where(j < 240, qs + 32 + (i % 16), qh + (i % 4)))
            offb = offr + ob[:, None]
            dv = tl.load(pdd + k // GS + offd, mask=mn, other=0.0).to(DT)
            v = tl.load(pbw + offb, mask=m2, other=0).to(tl.int32)
            a = tl.load(pa + k + offa, mask=ms[:, None], other=0.0)
            acc += tl.dot(a, _dec_trit(v, pw[:, None]).to(DT) * dv[None, :])
            if DUE:
                ev = tl.load(pdd2 + k // GS + offd, mask=mn, other=0.0).to(DT)
                u = tl.load(pbw2 + offb, mask=m2, other=0).to(tl.int32)
                acc2 += tl.dot(a, _dec_trit(u, pw[:, None]).to(DT) * ev[None, :])
        offo = (ls * so_s)[:, None] + ln[None, :]
        mo = ms[:, None] & mn[None, :]
        po = s0.to(tl.int64) * so_s + r0
        tl.store(o_ptr + po + offo, acc.to(o_ptr.dtype.element_ty), mask=mo)
        if DUE:
            tl.store(o2_ptr + po + offo, acc2.to(o2_ptr.dtype.element_ty), mask=mo)

    @triton.jit
    def _k_dx2(a_ptr, sa_s, pb, sb_e, NBR, d_ptr, sd_e, NG, o_ptr, so_s,
               te_ptr, ts_ptr, offs_ptr, R, C, NT,
               GS: tl.constexpr, QK: tl.constexpr, NB: tl.constexpr, ROT: tl.constexpr,
               BS: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr):
        """Backward dX: out[s, c] = sum_r A[s, r] W_e[r, c], like `_k_gemm` TRANSP=1 (same order along r in
        steps of BK). One program = token tile x ONE 256-column sub-block (8 accumulators of 32):
        A is re-read C/256 times instead of C/BN; the byte and scale offsets are loop-invariant
        (only the scalar row base changes)."""
        DT = a_ptr.dtype.element_ty
        pid_t, pid_n = _order_groups(tl.program_id(0), NT, C // 256, GROUP_M)
        e = tl.load(te_ptr + pid_t)
        if e < 0:
            return
        s0 = tl.load(ts_ptr + pid_t)
        s_end = tl.load(offs_ptr + e + 1)
        e64 = e.to(tl.int64)
        ls = tl.arange(0, BS)
        ms = (s0 + ls) < s_end
        lk = tl.arange(0, BK)
        c0 = pid_n * 256
        qs, qh = _schema_subblock(c0, QK, NB, ROT)
        i = tl.arange(0, 32)
        offr = (lk * NBR)[:, None]
        off0 = offr + (qs + i)[None, :]
        off5 = offr + (qs + 32 + (i % 16))[None, :]
        off7 = offr + tl.where(i < 16, qs + 32 + i, qh + (i % 4))[None, :]
        offd = lk * NG
        offa = (ls * sa_s)[:, None] + lk[None, :]
        pw5 = tl.where(i < 16, 1, 3)
        pw6 = tl.where(i < 16, 9, 27)
        pw7 = tl.where(i < 16, 81, tl.where(i < 20, 1, tl.where(i < 24, 3, tl.where(i < 28, 9, 27))))
        g = c0 // GS
        pe = pb + e64 * sb_e
        pde = d_ptr + e64 * sd_e + g
        pa = a_ptr + s0.to(tl.int64) * sa_s
        acc0 = tl.zeros((BS, 32), dtype=tl.float32)
        acc1 = tl.zeros((BS, 32), dtype=tl.float32)
        acc2 = tl.zeros((BS, 32), dtype=tl.float32)
        acc3 = tl.zeros((BS, 32), dtype=tl.float32)
        acc4 = tl.zeros((BS, 32), dtype=tl.float32)
        acc5 = tl.zeros((BS, 32), dtype=tl.float32)
        acc6 = tl.zeros((BS, 32), dtype=tl.float32)
        acc7 = tl.zeros((BS, 32), dtype=tl.float32)
        for r0 in range(0, R, BK):
            mr = (r0 + lk) < R
            prow = pe + r0 * NBR
            m2 = mr[:, None]
            v0 = tl.load(prow + off0, mask=m2, other=0).to(tl.int32)
            v5 = tl.load(prow + off5, mask=m2, other=0).to(tl.int32)
            v7 = tl.load(prow + off7, mask=m2, other=0).to(tl.int32)
            pdr = pde + r0 * NG
            dA = tl.load(pdr + offd, mask=mr, other=0.0).to(DT)
            if GS == 128:
                dB = tl.load(pdr + 1 + offd, mask=mr, other=0.0).to(DT)
            else:
                dB = dA
            a = tl.load(pa + r0 + offa, mask=ms[:, None] & mr[None, :], other=0.0)
            acc0 += tl.dot(a, _w_piece(v0, v5, v7, dA, dB, pw5, pw6, pw7, 0, GS, 0, DT))
            acc1 += tl.dot(a, _w_piece(v0, v5, v7, dA, dB, pw5, pw6, pw7, 1, GS, 0, DT))
            acc2 += tl.dot(a, _w_piece(v0, v5, v7, dA, dB, pw5, pw6, pw7, 2, GS, 0, DT))
            acc3 += tl.dot(a, _w_piece(v0, v5, v7, dA, dB, pw5, pw6, pw7, 3, GS, 0, DT))
            acc4 += tl.dot(a, _w_piece(v0, v5, v7, dA, dB, pw5, pw6, pw7, 4, GS, 0, DT))
            acc5 += tl.dot(a, _w_piece(v0, v5, v7, dA, dB, pw5, pw6, pw7, 5, GS, 0, DT))
            acc6 += tl.dot(a, _w_piece(v0, v5, v7, dA, dB, pw5, pw6, pw7, 6, GS, 0, DT))
            acc7 += tl.dot(a, _w_piece(v0, v5, v7, dA, dB, pw5, pw6, pw7, 7, GS, 0, DT))
        OT = o_ptr.dtype.element_ty
        po = o_ptr + s0.to(tl.int64) * so_s + c0
        offo = (ls * so_s)[:, None] + i[None, :]
        mo = ms[:, None]
        tl.store(po + offo, acc0.to(OT), mask=mo)
        tl.store(po + 32 + offo, acc1.to(OT), mask=mo)
        tl.store(po + 64 + offo, acc2.to(OT), mask=mo)
        tl.store(po + 96 + offo, acc3.to(OT), mask=mo)
        tl.store(po + 128 + offo, acc4.to(OT), mask=mo)
        tl.store(po + 160 + offo, acc5.to(OT), mask=mo)
        tl.store(po + 192 + offo, acc6.to(OT), mask=mo)
        tl.store(po + 224 + offo, acc7.to(OT), mask=mo)

    @triton.jit
    def _k_dd2(dy_ptr, sdy_s, x_ptr, sx_s, pb, sb_e, NBR, dd_ptr, sdd_e, NG, offs_ptr, R, C,
               GS: tl.constexpr, QK: tl.constexpr, NB: tl.constexpr, ROT: tl.constexpr,
               BR: tl.constexpr, BS: tl.constexpr):
        """dL/dd as a GEMM with variable K (K = tokens of the expert): program = (expert, BR rows, ONE
        256-column sub-block); acc_p (BR x 32) = dY^T X for the 8 pieces, token tile BS; dY
        read in contiguous rows (BS x BR, then transposed in the registers) once per sub-block (C/256 times
        instead of NG); trits (pieces) and per-group sums ONLY in the epilogue. Compared to `_k_dd` only the
        order of the fp32 sums changes (token step and group reduction)."""
        e = tl.program_id(0)
        pid_r = tl.program_id(1)
        c0 = tl.program_id(2) * 256
        s_ini = tl.load(offs_ptr + e)
        s_end = tl.load(offs_ptr + e + 1)
        r0 = pid_r * BR
        lr = tl.arange(0, BR)
        mr = (r0 + lr) < R
        ls = tl.arange(0, BS)
        i = tl.arange(0, 32)
        offy = (ls * sdy_s)[:, None] + lr[None, :]
        offx = (ls * sx_s)[:, None] + i[None, :]
        pdy = dy_ptr + r0
        px = x_ptr + c0
        acc0 = tl.zeros((BR, 32), dtype=tl.float32)
        acc1 = tl.zeros((BR, 32), dtype=tl.float32)
        acc2 = tl.zeros((BR, 32), dtype=tl.float32)
        acc3 = tl.zeros((BR, 32), dtype=tl.float32)
        acc4 = tl.zeros((BR, 32), dtype=tl.float32)
        acc5 = tl.zeros((BR, 32), dtype=tl.float32)
        acc6 = tl.zeros((BR, 32), dtype=tl.float32)
        acc7 = tl.zeros((BR, 32), dtype=tl.float32)
        for s0 in range(s_ini, s_end, BS):
            ms = (s0 + ls) < s_end
            s64 = s_ini.to(tl.int64) + (s0 - s_ini)       # int64 in the interpreter too
            dy = tl.load(pdy + s64 * sdy_s + offy, mask=ms[:, None] & mr[None, :], other=0.0)
            dyt = tl.trans(dy)
            pxs = px + s64 * sx_s + offx
            mx = ms[:, None]
            acc0 += tl.dot(dyt, tl.load(pxs, mask=mx, other=0.0))
            acc1 += tl.dot(dyt, tl.load(pxs + 32, mask=mx, other=0.0))
            acc2 += tl.dot(dyt, tl.load(pxs + 64, mask=mx, other=0.0))
            acc3 += tl.dot(dyt, tl.load(pxs + 96, mask=mx, other=0.0))
            acc4 += tl.dot(dyt, tl.load(pxs + 128, mask=mx, other=0.0))
            acc5 += tl.dot(dyt, tl.load(pxs + 160, mask=mx, other=0.0))
            acc6 += tl.dot(dyt, tl.load(pxs + 192, mask=mx, other=0.0))
            acc7 += tl.dot(dyt, tl.load(pxs + 224, mask=mx, other=0.0))
        e64 = e.to(tl.int64)
        qs, qh = _schema_subblock(c0, QK, NB, ROT)
        pw5 = tl.where(i < 16, 1, 3)
        pw6 = tl.where(i < 16, 9, 27)
        pw7 = tl.where(i < 16, 81, tl.where(i < 20, 1, tl.where(i < 24, 3, tl.where(i < 28, 9, 27))))
        prow = pb + e64 * sb_e + r0.to(tl.int64) * NBR
        offr = (lr * NBR)[:, None]
        m2 = mr[:, None]
        v0 = tl.load(prow + offr + (qs + i)[None, :], mask=m2, other=0).to(tl.int32)
        v5 = tl.load(prow + offr + (qs + 32 + (i % 16))[None, :], mask=m2, other=0).to(tl.int32)
        v7 = tl.load(prow + offr + tl.where(i < 16, qs + 32 + i, qh + (i % 4))[None, :], mask=m2,
                     other=0).to(tl.int32)
        p0 = acc0 * _t_piece(v0, v5, v7, pw5, pw6, pw7, 0, 0).to(tl.float32)
        p0 += acc1 * _t_piece(v0, v5, v7, pw5, pw6, pw7, 1, 0).to(tl.float32)
        p0 += acc2 * _t_piece(v0, v5, v7, pw5, pw6, pw7, 2, 0).to(tl.float32)
        p0 += acc3 * _t_piece(v0, v5, v7, pw5, pw6, pw7, 3, 0).to(tl.float32)
        p1 = acc4 * _t_piece(v0, v5, v7, pw5, pw6, pw7, 4, 0).to(tl.float32)
        p1 += acc5 * _t_piece(v0, v5, v7, pw5, pw6, pw7, 5, 0).to(tl.float32)
        p1 += acc6 * _t_piece(v0, v5, v7, pw5, pw6, pw7, 6, 0).to(tl.float32)
        p1 += acc7 * _t_piece(v0, v5, v7, pw5, pw6, pw7, 7, 0).to(tl.float32)
        g = c0 // GS
        pdd = dd_ptr + e64 * sdd_e + r0.to(tl.int64) * NG + g + lr * NG
        if GS == 128:
            tl.store(pdd, tl.sum(p0, axis=1), mask=mr)
            tl.store(pdd + 1, tl.sum(p1, axis=1), mask=mr)
        else:
            tl.store(pdd, tl.sum(p0 + p1, axis=1), mask=mr)

    # 2026-09-26: FUSED combination of the experts (see CombineExperts). Summation order j = 0..K-1 like
    # `build_moe_ffn` in llama.cpp (moe_out = e0 + e1 + ...); deterministic, no atomics.
    @triton.jit
    def _k_combine(o_ptr, so_s, w_ptr, inv_ptr, y_ptr, sy_t, H, K: tl.constexpr, BH: tl.constexpr):
        """y[t, h] = sum_j w[t, j] * o[inverse[t*K + j], h], fp32."""
        t64 = tl.program_id(0).to(tl.int64)
        h = tl.program_id(1) * BH + tl.arange(0, BH)
        m = h < H
        acc = tl.zeros((BH,), dtype=tl.float32)
        for j in tl.static_range(K):
            r = tl.load(inv_ptr + t64 * K + j).to(tl.int64)
            wj = tl.load(w_ptr + t64 * K + j)
            acc += tl.load(o_ptr + r * so_s + h, mask=m, other=0.0) * wj
        tl.store(y_ptr + t64 * sy_t + h, acc, mask=m)

    @triton.jit
    def _k_combine_do(g_ptr, sg_t, w_ptr, ord_ptr, do_ptr, sdo_s, H, K: tl.constexpr, BH: tl.constexpr):
        """dL/do[s, h] = w[q] * g[q // K, h], q = order[s] (row s of the per-expert order)."""
        s64 = tl.program_id(0).to(tl.int64)
        h = tl.program_id(1) * BH + tl.arange(0, BH)
        m = h < H
        q = tl.load(ord_ptr + s64).to(tl.int64)
        wv = tl.load(w_ptr + q)
        g = tl.load(g_ptr + (q // K) * sg_t + h, mask=m, other=0.0)
        tl.store(do_ptr + s64 * sdo_s + h, g * wv, mask=m)

    @triton.jit
    def _k_combine_dw(g_ptr, sg_t, o_ptr, so_s, inv_ptr, dw_ptr, H, K: tl.constexpr, BH: tl.constexpr):
        """dL/dw[q] = sum_h g[q // K, h] * o[inverse[q], h], fp32."""
        q64 = tl.program_id(0).to(tl.int64)
        r = tl.load(inv_ptr + q64).to(tl.int64)
        t = q64 // K
        acc = tl.zeros((BH,), dtype=tl.float32)
        for h0 in range(0, H, BH):
            h = h0 + tl.arange(0, BH)
            m = h < H
            acc += tl.load(g_ptr + t * sg_t + h, mask=m, other=0.0) * tl.load(o_ptr + r * so_s + h, mask=m, other=0.0)
        tl.store(dw_ptr + q64, tl.sum(acc, axis=0))


@dataclass
class Blocks:
    """Triton tile sizes. Starting values chosen WITHOUT a GPU (2026-09-26): ahead-of-time compilation for
    gfx1151 (`triton.compile`, GPUTarget("hip", "gfx1151", 32)) and counting of v_wmma and of the registers
    spilled to memory. With num_warps 4 and BS=BN=64, BK=32 the fused product spilled 34 registers,
    and dd with GS=256 up to 139: these values give 0 spills on all variants. The time is measured by M0
    (`--measure --blocks '{...}'`)."""
    BS: int = 64
    BN: int = 64
    BK: int = 32
    BR_dd: int = 64
    BS_dd: int = 16
    BR_dec: int = 32
    BC_dec: int = 128
    num_warps: int = 8
    # engine «triton2» (2026-09-27): tiles of the kernels _k_gemm2 / _k_dx2 / _k_dd2. Values chosen with the same
    # ahead-of-time compilation for gfx1151 (e2e_test_triton2 K6): 0 spilled registers with fp16 inputs in all
    # three (forward also with gate+up together); the time is decided by tools/cancello_kernel.sh (--sweep).
    BS2: int = 64          # tokens per forward tile (also gate+up together)
    BN2: int = 64          # output rows per forward tile
    GROUP_M: int = 8       # token tiles per group (ordering for L2), forward and dX
    BS_dx2: int = 32       # tokens per dX tile (x 256 columns: 8 accumulators of 32)
    BK_dx2: int = 32       # step along R for dX (32 = same order of the sums as _k_gemm)
    BR_dd2: int = 64       # rows per dL/dd tile (x 256 columns)
    BS_dd2: int = 32       # tokens per dL/dd step
    num_warps2: int = 8
    stages_gemm2: int = 0   # num_stages of the forward (0 = Triton default; 1 = no prefetch, fewer registers)
    stages_dx2: int = 1     # dX and dL/dd: with prefetch (default) the 8 unrolled pieces spill registers
    stages_dd2: int = 1
    gate_up: bool = True   # gate and up in the same forward sweep (same input)


def decode_weights(fam: FamilyData, d: torch.Tensor, dtype: torch.dtype, blk: Blocks = Blocks()) -> torch.Tensor:
    """Fused kernel bytes + d -> W (E, R, C) = trit x d in the byte domain (rotated for TQ1_0R_1024)."""
    E, R, C = fam.E, fam.R, fam.C
    W = torch.empty(E, R, C, dtype=dtype, device=fam.blocks.device)
    d = d.contiguous()
    grid = (E, triton.cdiv(R, blk.BR_dec), triton.cdiv(C, blk.BC_dec))
    _k_decode[grid](fam.blocks, fam.blocks.stride(0), fam.NBR, d, d.stride(0), fam.NG,
                           W, W.stride(0), W.stride(1), R, C, fam.gs, fam.qk, fam.nb, int(fam.rotated),
                           blk.BR_dec, blk.BC_dec, num_warps=blk.num_warps)
    return W


def _gemm_triton(A: torch.Tensor, fam: FamilyData, d: Optional[torch.Tensor], W: Optional[torch.Tensor],
                 plane: Plane, transp: bool, out_dtype: torch.dtype, blk: Blocks) -> torch.Tensor:
    A = A.contiguous()
    K = A.shape[1]
    N = fam.C if transp else fam.R
    assert K == (fam.R if transp else fam.C), (K, fam.R, fam.C, transp)
    out = torch.empty(A.shape[0], N, dtype=out_dtype, device=A.device)
    te, ts, n_max = plane.table(blk.BS)
    dense = W is not None
    dd_ = d.contiguous() if d is not None else A      # placeholder, not read when DENSE
    Wd = W if dense else A
    grid = (n_max, triton.cdiv(N, blk.BN))
    _k_gemm[grid](A, A.stride(0), A.stride(1),
                     fam.blocks, fam.blocks.stride(0), fam.NBR,
                     dd_, dd_.stride(0) if not dense else 0, fam.NG,
                     Wd, Wd.stride(0) if dense else 0, Wd.stride(1) if dense else 0,
                     out, out.stride(0), out.stride(1),
                     te, ts, plane.offs,
                     fam.R, fam.C, K, N,
                     fam.gs, fam.qk, fam.nb, int(fam.rotated), int(transp), int(dense),
                     blk.BS, blk.BN, blk.BK, num_warps=blk.num_warps)
    return out


def _dd_triton(dY: torch.Tensor, X: torch.Tensor, fam: FamilyData, plane: Plane, blk: Blocks) -> torch.Tensor:
    dY, X = dY.contiguous(), X.contiguous()
    assert dY.dtype == X.dtype
    dd = torch.empty(fam.E, fam.R, fam.NG, dtype=torch.float32, device=dY.device)
    grid = (fam.E, triton.cdiv(fam.R, blk.BR_dd), fam.NG)
    _k_dd[grid](dY, dY.stride(0), dY.stride(1), X, X.stride(0), X.stride(1),
                   fam.blocks, fam.blocks.stride(0), fam.NBR, dd, dd.stride(0), fam.NG, plane.offs,
                   fam.R, fam.C, fam.gs, fam.qk, fam.nb, int(fam.rotated), blk.BR_dd, blk.BS_dd,
                   num_warps=blk.num_warps)
    return dd


def _stages(n: int) -> dict:
    return {"num_stages": n} if n > 0 else {}


def _check_v2(fam: FamilyData) -> None:
    assert fam.C % 256 == 0 and fam.gs in (128, 256) and fam.qk % 256 == 0, (fam.type_, fam.C, fam.gs)


def _forward2_triton(A: torch.Tensor, fams: list, ds: list, plane: Plane, out_dtype: torch.dtype,
                    blk: Blocks) -> list:
    """Forward of the triton2 engine for ONE family or for TWO of the same shape (gate and up, same A)."""
    A = A.contiguous()
    f = fams[0]
    _check_v2(f)
    assert A.shape[1] == f.C and len(fams) in (1, 2)
    for g in fams[1:]:
        assert (g.type_, g.E, g.R, g.NBR) == (f.type_, f.E, f.R, f.NBR)
    ds = [d.contiguous() for d in ds]
    from fucina.kernels import qat as _KQ               # 10-09 E2E_KQAT=warm: planes 2/3 only on the experts that have them
    blk = _KQ.blk_forward(blk)                  # 10-09 E2E_KQAT=tile: measured tiles (same values, bit-exact)
    te, ts, n_t, skip = _KQ.table(plane, blk.BS2, fams)
    new = torch.zeros if skip else torch.empty
    outs = [new(A.shape[0], f.R, dtype=out_dtype, device=A.device) for _ in fams]
    f2, d2, o2 = fams[-1], ds[-1], outs[-1]
    grid = (n_t * triton.cdiv(f.R, blk.BN2),)
    _k_gemm2[grid](A, A.stride(0),
                      f.blocks, f.blocks.stride(0), f.NBR, ds[0], ds[0].stride(0), f.NG, outs[0], outs[0].stride(0),
                      f2.blocks, f2.blocks.stride(0), d2, d2.stride(0), o2,
                      te, ts, plane.offs, f.R, f.C, n_t,
                      f.gs, f.qk, f.nb, int(f.rotated), int(len(fams) == 2),
                      blk.BS2, blk.BN2, blk.GROUP_M, num_warps=blk.num_warps2, **_stages(blk.stages_gemm2))
    return outs


def _dx2_triton(gy: torch.Tensor, fam: FamilyData, d: torch.Tensor, plane: Plane, out_dtype: torch.dtype,
                blk: Blocks) -> torch.Tensor:
    gy, d = gy.contiguous(), d.contiguous()
    _check_v2(fam)
    assert gy.shape[1] == fam.R
    from fucina.kernels import qat as _KQ               # 10-09 E2E_KQAT=warm (see _forward2_triton)
    te, ts, n_t, skip = _KQ.table(plane, blk.BS_dx2, [fam])
    out = (torch.zeros if skip else torch.empty)(gy.shape[0], fam.C, dtype=out_dtype, device=gy.device)
    _k_dx2[(n_t * (fam.C // 256),)](gy, gy.stride(0), fam.blocks, fam.blocks.stride(0), fam.NBR,
                                    d, d.stride(0), fam.NG, out, out.stride(0), te, ts, plane.offs,
                                    fam.R, fam.C, n_t, fam.gs, fam.qk, fam.nb, int(fam.rotated),
                                    blk.BS_dx2, blk.BK_dx2, blk.GROUP_M, num_warps=blk.num_warps2,
                                    **_stages(blk.stages_dx2))
    return out


def _dd2_triton(dY: torch.Tensor, X: torch.Tensor, fam: FamilyData, plane: Plane, blk: Blocks) -> torch.Tensor:
    dY, X = dY.contiguous(), X.contiguous()
    assert dY.dtype == X.dtype
    _check_v2(fam)
    dd = torch.empty(fam.E, fam.R, fam.NG, dtype=torch.float32, device=dY.device)
    grid = (fam.E, triton.cdiv(fam.R, blk.BR_dd2), fam.C // 256)
    _k_dd2[grid](dY, dY.stride(0), X, X.stride(0), fam.blocks, fam.blocks.stride(0), fam.NBR,
                    dd, dd.stride(0), fam.NG, plane.offs, fam.R, fam.C,
                    fam.gs, fam.qk, fam.nb, int(fam.rotated), blk.BR_dd2, blk.BS_dd2, num_warps=blk.num_warps2,
                    **_stages(blk.stages_dd2))
    return dd


# ═══════════════════════════════════ engines ═══════════════════════════════════
class TorchEngine:
    """Same math as the kernels, expert by expert (transient = ONE expert). CPU and tests."""
    name = "torch"

    def weights(self, fam, d, recipe):
        return None

    def forward_(self, x, d, fam, plane, recipe, weights=None):
        """With dtype_gemm fp16 the product is done in fp32 on the (exact) fp16 values: this is what an fp32
        accumulation over fp16 inputs does, except for the order of the sums."""
        out = torch.empty(x.shape[0], fam.R, dtype=recipe.dtype_output, device=x.device)
        o = plane.offs.tolist()
        for e in range(fam.E):
            if o[e + 1] > o[e]:
                W = weight_expert_torch(fam, e, d, recipe)
                xs = x[o[e]:o[e + 1]]
                if recipe.dtype_gemm == torch.float16:
                    out[o[e]:o[e + 1]] = torch.matmul(xs.float(), W.float().T)
                else:
                    out[o[e]:o[e + 1]] = torch.matmul(xs, W.T).to(out.dtype)
        return out

    def backward_x(self, gy, d, fam, plane, recipe, weights=None):
        dx = torch.empty(gy.shape[0], fam.C, dtype=gy.dtype, device=gy.device)
        o = plane.offs.tolist()
        for e in range(fam.E):
            if o[e + 1] > o[e]:
                W = weight_expert_torch(fam, e, d, recipe)
                if recipe.dtype_gemm == torch.float16:
                    dx[o[e]:o[e + 1]] = torch.matmul(gy[o[e]:o[e + 1]].float(), W.float()).to(dx.dtype)
                else:
                    dx[o[e]:o[e + 1]] = torch.matmul(gy[o[e]:o[e + 1]], W)
        return dx

    def dd(self, gy, x, fam, plane, d_dtype):
        acc = torch.float64 if gy.dtype == torch.float64 else torch.float32
        dd = torch.zeros(fam.E, fam.R, fam.NG, dtype=d_dtype, device=gy.device)
        o = plane.offs.tolist()
        for e in range(fam.E):
            if o[e + 1] > o[e]:
                dW = torch.matmul(gy[o[e]:o[e + 1]].to(acc).T, x[o[e]:o[e + 1]].to(acc))       # (R, C)
                q = trit_expert(fam, e).to(acc)
                dd[e] = (q * dW).view(fam.R, fam.NG, fam.gs).sum(-1).to(d_dtype)
        return dd


class TritonEngine:
    """Triton kernels. dense=False: decode inside the product (the weight never goes through global memory).
    dense=True: one kernel decodes bytes -> weight, the product reads it (and for recipe A the de-rotated weight
    is built dense anyway, bf16).
    v2=True (engine «triton2», 2026-09-27): kernels _k_gemm2 / _k_dx2 / _k_dd2 (decode in pieces, scale per group,
    grouped ordering for L2, dL/dd with variable K, gate+up in the same sweep). Forward and dX: the same values
    as «triton» BIT-EXACT (same products, same order along K); dL/dd: same value except for the order of the fp32
    sums. Where the dense weight is needed (recipe A on the rotated families) the «triton» path is kept."""

    def __init__(self, dense: bool = False, blk: Optional[Blocks] = None, v2: bool = False):
        if dense and v2:
            raise ValueError("triton2 only supports fused decoding")
        self.dense = dense
        self.v2 = v2
        self.blk = blk or Blocks()
        self.name = "triton_dense" if dense else ("triton2" if v2 else "triton")

    @property
    def gate_up_together(self) -> bool:
        return self.v2 and self.blk.gate_up

    def can_merge(self, fg, fu, recipe) -> bool:
        """Gate and up in the same sweep: triton2 engine, same shape, weight decoded inside the kernel."""
        return (self.gate_up_together and (fg.type_, fg.E, fg.R, fg.NBR) == (fu.type_, fu.E, fu.R, fu.NBR)
                and not (fg.rotated and not recipe.rotate))

    def weights(self, fam, d, recipe, piece: int = 32):
        """Dense weight (E, R, C) when needed: recipe A for the rotated families (R^T(trit x d) in fp32, then
        dtype_gemm, in pieces of `piece` experts: the fp32 transient is 1/8 of the family) or the dense engine;
        otherwise None (decode inside the product)."""
        if fam.rotated and not recipe.rotate:
            W = torch.empty(fam.E, fam.R, fam.C, dtype=recipe.dtype_gemm, device=fam.blocks.device)
            for e0 in range(0, fam.E, piece):
                e1 = min(fam.E, e0 + piece)
                below = FamilyData(fam.name, fam.type_, fam.blocks[e0:e1], fam.d0[e0:e1])
                W[e0:e1] = de_rotate(decode_weights(below, d[e0:e1], torch.float32, self.blk)).to(recipe.dtype_gemm)
            return W
        if self.dense:
            return decode_weights(fam, d, recipe.dtype_gemm, self.blk)
        return None

    def forward_(self, x, d, fam, plane, recipe, weights=None):
        W = self.weights(fam, d, recipe) if weights is None else weights
        if self.v2 and W is None:
            return _forward2_triton(x, [fam], [d], plane, recipe.dtype_output, self.blk)[0]
        return _gemm_triton(x, fam, d, W, plane, False, recipe.dtype_output, self.blk)

    def forward_two(self, x, ds, fams, plane, recipe):
        """Gate and up on the same x in one sweep (only if `can_merge`)."""
        return _forward2_triton(x, list(fams), list(ds), plane, recipe.dtype_output, self.blk)

    def backward_x(self, gy, d, fam, plane, recipe, weights=None):
        W = self.weights(fam, d, recipe) if weights is None else weights
        if self.v2 and W is None:
            return _dx2_triton(gy, fam, d, plane, gy.dtype, self.blk)
        return _gemm_triton(gy, fam, d, W, plane, True, gy.dtype, self.blk)

    def dd(self, gy, x, fam, plane, d_dtype):
        if self.v2:
            return _dd2_triton(gy, x, fam, plane, self.blk).to(d_dtype)
        return _dd_triton(gy, x, fam, plane, self.blk).to(d_dtype)


def engine(name: str, blk: Optional[Blocks] = None):
    if name == "torch":
        return TorchEngine()
    if name in ("triton", "triton_dense", "triton2"):
        if not _TRITON:
            raise RuntimeError("Triton not available")
        return TritonEngine(dense=(name == "triton_dense"), blk=blk, v2=(name == "triton2"))
    raise ValueError(name)


# ═══════════════════════════════════ autograd per family ═══════════════════════════════════
class Observer:
    """2026-09-26. Guards on the fp16 tensors of recipes B/C (risk 8 of the project: overflow at 65504; A2/G1: the
    gradient below 2^-24 becomes zero), WITHOUT synchronizations in the loop:
    - `overflow`: bool on the device, set if an fp16 input of an expert product (rotated activation,
      silu(g)*u, gradient scaled by K) or an fp16 dX is not finite. Read in `Student.check`.
    - `diagnostics = True` (only at step 0, for gate G1): for each (layer, family) it counts, on the
      output gradient in fp32 ALREADY scaled by K, the non-zero elements that go to zero in fp16 (< 2^-24)
      and those that become subnormal (< 2^-14), plus the maximum |g|. Tensors are collected; `summary()`
      reads them with a single synchronization."""

    def __init__(self, device):
        self.overflow = torch.zeros((), dtype=torch.bool, device=device)
        self.diagnostics = False
        self._rows: list = []

    def watch(self, t: torch.Tensor) -> None:
        if t.dtype == torch.float16:
            from fucina.kernels import qat as _KQ           # 10-09 E2E_KQAT=guard: a single sweep, same predicate
            if _KQ.active("guard"):
                _KQ.watch_fp16(self.overflow, t, self.__dict__.setdefault("_flag", {}))
            else:
                self.overflow |= ~torch.isfinite(t).all()

    def gradient(self, fam: "FamilyData", g32: torch.Tensor, g16: torch.Tensor) -> None:
        self.watch(g16)
        if self.diagnostics:
            a = g32.abs()
            nz = a > 0
            self._rows.append((fam.layer, fam.name, nz.sum(), (nz & (g16 == 0)).sum(),
                                (nz & (g16 != 0) & (g16.abs() < 2.0 ** -14)).sum(), a.max()))

    def summary(self) -> list:
        if not self._rows:
            return []
        numbers = torch.stack([torch.stack([r[2].double(), r[3].double(), r[4].double(), r[5].double()])
                              for r in self._rows]).tolist()
        out = [{"layer": r[0], "family": r[1], "non_zeros": int(n[0]),
                "a_zero_in_fp16": int(n[1]), "subnormals_in_fp16": int(n[2]), "max_abs": n[3],
                "fraction_a_zero": n[1] / max(n[0], 1), "fraction_subnormals": n[2] / max(n[0], 1)}
               for r, n in zip(self._rows, numbers)]
        self._rows = []
        return out


class ExpertFamily(torch.autograd.Function):
    """y = grouped product of x (tokens sorted by expert, (S, C)) with the weights of ALL the experts of
    a family, W_e = trit_e x d_e. Differentiable in x and in d (the scales, already rounded to fp16).

    Saved for the backward pass: x and d (and nothing else). Backward:
      dx      = grouped(gy, W_e) with W_e rebuilt from the bytes;
      dL/dd   = per-group sum of trit (.) (gy_e^T x_rot_e), fp32, without materializing dW;
    where x_rot = x for recipe B (x is already rotated) and for down; = R x for recipe A (real x).
    Cost O(E): each expert is touched once per sweep."""

    @staticmethod
    def forward(ctx, x, d, fam: FamilyData, plane: Plane, recipe: Recipe, mot, oss: Optional[Observer] = None):
        y = mot.forward_(x, d, fam, plane, recipe)
        ctx.save_for_backward(x, d)
        ctx.fam, ctx.plane, ctx.recipe, ctx.mot, ctx.oss = fam, plane, recipe, mot, oss
        return y

    @staticmethod
    def backward(ctx, gy):
        x, d = ctx.saved_tensors
        dx, dd = _backward_family(gy, x, d, ctx.fam, ctx.plane, ctx.recipe, ctx.mot, ctx.oss,
                                    ctx.needs_input_grad[0], ctx.needs_input_grad[1])
        return dx, dd, None, None, None, None, None


def _backward_family(gy, x, d, fam, plane, recipe, mot, oss, need_x: bool, need_d: bool):
    """Backward of ONE family (body of `ExpertFamily.backward`, shared with `GateUpFamilies`)."""
    if recipe.backward_split and recipe.dtype_backward is None:      # 09-27: split products (e2e_backward_hilo)
        from fucina.kernels import hilo_backward as HL
        return HL.backward_family(gy, x, d, fam, plane, recipe, mot, oss, need_x, need_d)
    di = recipe.dtype_backward or x.dtype          # dtype of the INPUTS of the backward products
    g_in = gy
    gy = gy.to(di).contiguous()
    if oss is not None:
        oss.gradient(fam, g_in, gy)
    if need_d and getattr(getattr(mot, "state", None), "skip_dd_redundant", False):
        from fucina.student import plane_codes as _PV                # 10-02 --pv-dd-only-skips: dL/dd that nobody reads
        if _PV.dd_redundant(mot, fam):
            need_d = False
    dx = None
    if need_x:
        weights = mot.weights(fam, d, recipe)
        dx = mot.backward_x(gy, d, fam, plane, recipe, weights).to(x.dtype)
        del weights
        if oss is not None:
            oss.watch(dx)
    dd = None
    if need_d:
        xr = rotate(x.to(torch.float64 if x.dtype == torch.float64 else torch.float32)) \
            if (fam.rotated and not recipe.rotate) else x
        dd = mot.dd(gy, xr.to(di).contiguous(), fam, plane, d.dtype)
    return dx, dd


class GateUpFamilies(torch.autograd.Function):
    """2026-09-27, triton2 engine: (g, u) = gate and up on the SAME x in one kernel sweep (A read once).
    Forward bit-exact with two `ExpertFamily`. Backward: the two families as before, first up then gate
    (the order of the autograd engine with two nodes), dx = dx_up + dx_gate in the dtype of x (the sum that autograd
    did when accumulating the gradient of x: commutative, same result)."""

    @staticmethod
    def forward(ctx, x, dg, du, fg: FamilyData, fu: FamilyData, plane: Plane, recipe: Recipe, mot,
                oss: Optional[Observer] = None):
        g, u = mot.forward_two(x, (dg, du), (fg, fu), plane, recipe)
        ctx.save_for_backward(x, dg, du)
        ctx.fg, ctx.fu, ctx.plane, ctx.recipe, ctx.mot, ctx.oss = fg, fu, plane, recipe, mot, oss
        return g, u

    @staticmethod
    def backward(ctx, gg, gu):
        x, dg, du = ctx.saved_tensors
        nx, ng, nu = ctx.needs_input_grad[0], ctx.needs_input_grad[1], ctx.needs_input_grad[2]
        dx_u, ddu = _backward_family(gu, x, du, ctx.fu, ctx.plane, ctx.recipe, ctx.mot, ctx.oss, nx, nu)
        dx_g, ddg = _backward_family(gg, x, dg, ctx.fg, ctx.plane, ctx.recipe, ctx.mot, ctx.oss, nx, ng)
        dx = dx_u + dx_g if nx else None
        return dx, ddg, ddu, None, None, None, None, None, None


class RoundFP16(torch.autograd.Function):
    """Forward: the fp16 RNE value (== `.half()`, numpy astype(float16), GGML_FP32_TO_FP16), cast back to the
    input dtype: it is the scale the GGUF will contain. Backward: identity (straight-through). The finiteness
    check does NOT synchronize: it accumulates into `sentinel` (bool tensor on the device); the trainer
    reads it once per step (`Student.check`). Copy of `e2e_prototipo.RoundFP16` apart from this."""

    @staticmethod
    def forward(ctx, s: torch.Tensor, sentinel: Optional[torch.Tensor]):
        r = s.to(torch.float16)
        if sentinel is None:
            if not torch.isfinite(r).all():
                raise FloatingPointError("scale outside the fp16 range (|d| > 65504 or NaN)")
        else:
            sentinel |= ~torch.isfinite(r).all()
        return r.to(s.dtype)

    @staticmethod
    def backward(ctx, g):
        return g, None


class _Collect(torch.autograd.Function):
    """x_ord = x[order // k] (x: (T, H)). Direct backward: the gradient returns to the original order through the
    INVERSE permutation (one gather) and the k copies of each token are summed in fp32 (like esperti_thor)."""

    @staticmethod
    def forward(ctx, x, order, inverse, k):
        ctx.save_for_backward(inverse)
        ctx.k, ctx.T = k, x.shape[0]
        return x[order // k]

    @staticmethod
    def backward(ctx, g):
        (inverse,) = ctx.saved_tensors
        acc = torch.float64 if g.dtype == torch.float64 else torch.float32
        dx = g[inverse].view(ctx.T, ctx.k, -1).to(acc).sum(1).to(g.dtype)
        return dx, None, None, None


class _Scatter(torch.autograd.Function):
    """y[t] = sum_j o[inverse[t*k + j]] (o: (S, H) sorted by expert), summed in fp32/fp64."""

    @staticmethod
    def forward(ctx, o, order, inverse, k):
        ctx.save_for_backward(order)
        ctx.k = k
        acc = torch.float64 if o.dtype == torch.float64 else torch.float32
        return o[inverse].view(-1, k, o.shape[1]).to(acc).sum(1)

    @staticmethod
    def backward(ctx, g):
        (order,) = ctx.saved_tensors
        k = ctx.k
        gs = g.unsqueeze(1).expand(-1, k, -1).reshape(-1, g.shape[1])       # row t*k+j
        return gs[order].contiguous(), None, None, None


# Fused combination (2026-09-26, speed lever measurable before/after in M0: `measure --combine fused,torch`).
# E2E_COMBINE_FUSED=0 goes back to the torch path (o*wo, gather, sum). Active only with the Triton engine and fp32 outputs.
COMBINE_FUSED = os.getenv("E2E_COMBINE_FUSED", "1") == "1"
BH_COMBINE = 512


class CombineExperts(torch.autograd.Function):
    """y[t] = sum_j w[t, j] * o[inverse[t*k + j]]: o (S, H) fp32 = UNWEIGHTED outputs of down, in per-expert
    order; w (T, k) fp32 = renormalized routing weights. Replaces `o * wo` + `_Scatter`, which at
    8192 tokens and H = 2048 move (S, H) fp32 = 512 MiB four times in the forward pass and as many in the backward
    (weighted copy, gather, sum; in the backward expansion, gather and product). Here: forward reads o and writes y;
    backward writes dL/do (one sweep) and reads o for dL/dw (one scalar product per assignment). Saves o, w and
    the permutations (the torch path saved the weighted o and wo: same memory)."""

    @staticmethod
    def forward(ctx, o, w, order, inverse, k):
        o = o.contiguous()
        w = w.contiguous()
        T, H = w.shape[0], o.shape[1]
        y = torch.empty(T, H, dtype=torch.float32, device=o.device)
        _k_combine[(T, triton.cdiv(H, BH_COMBINE))](o, o.stride(0), w, inverse, y, y.stride(0), H, k, BH_COMBINE,
                                                    num_warps=4)
        ctx.save_for_backward(o, w, order, inverse)
        ctx.k = k
        return y

    @staticmethod
    def backward(ctx, gy):
        o, w, order, inverse = ctx.saved_tensors
        k = ctx.k
        g = gy.to(torch.float32).contiguous()
        S_, H = o.shape
        do = dw = None
        if ctx.needs_input_grad[0]:
            do = torch.empty(S_, H, dtype=torch.float32, device=o.device)
            _k_combine_do[(S_, triton.cdiv(H, BH_COMBINE))](g, g.stride(0), w, order, do, do.stride(0), H, k,
                                                             BH_COMBINE, num_warps=4)
        if ctx.needs_input_grad[1]:
            dw = torch.empty(w.shape, dtype=torch.float32, device=o.device)
            _k_combine_dw[(S_,)](g, g.stride(0), o, o.stride(0), inverse, dw, H, k, BH_COMBINE, num_warps=4)
        return do, dw, None, None, None


# ═══════════════════════════════════ student MoE ═══════════════════════════════════
class MoEStudente(torch.nn.Module):
    """Replaces `Qwen3_5MoeSparseMoeBlock`: router (fp32 softmax, top-k, renormalization, like
    Qwen3.5-MoE and llama.cpp), shared expert with a sigmoid gate (HF modules with the GGUF weights), routed
    experts with `ExpertFamily`. `instr` (set by the caller) = saved top-k to reuse;
    `last_idx` = top-k used in the last call; `changes` = tensor (device) of the tokens whose live top-k
    differs from the saved one."""

    def __init__(self, router_w, shared, shared_gate, layer: "ExpertLayer", top_k: int):
        super().__init__()
        self.router_w = router_w          # (E, H), nn.Parameter without a gradient
        self.shared_expert = shared
        self.shared_expert_gate = shared_gate
        self.experts = layer
        self.top_k = top_k
        self.instr: Optional[torch.Tensor] = None
        self.last_idx: Optional[torch.Tensor] = None
        self.changes: Optional[torch.Tensor] = None
        self.router_w32: Optional[torch.Tensor] = None   # plain attribute (NOT a buffer: .to(bf16) does not touch it)

    def _plane2(self, name: str, idx: torch.Tensor, plane: "Plane", cache: dict):
        """2026-10-09 (fix P1): (Plane of the 2nd plane, perm_12, perm_21) for projection `name`, or None if absent.
        idx2 = map2[idx] (experts without a 2nd plane -> null slot E2). perm_12[q] = row in the first plane's order
        of assignment q of the second (x2 = x_ord[perm_12]); perm_21[p] = row of the second for row p of the first
        (y += y2[perm_21]). Same meaning as the engine's `second_plane` (mask = null slot, exact 0 output)."""
        st = self.experts
        m = st.map2.get(name)
        if m is None:
            return None
        k = id(m)
        if k not in cache:
            p2 = Plane(m[idx], st.fam2[name].E)
            cache[k] = (p2, plane.inverse[p2.order], p2.inverse[plane.order])
        return cache[k]

    def _second_plane(self, name: str, x_ord: torch.Tensor, idx, plane, cache, oss):
        """Contribution of the 2nd plane (0 if absent) in the first plane's order: mm(W2[map[e]], x) per assignment."""
        st = self.experts
        t = self._plane2(name, idx, plane, cache)
        if t is None:
            return None
        p2, perm_12, perm_21 = t
        y2 = ExpertFamily.apply(x_ord[perm_12].contiguous(), st.scale2(name), st.fam2[name], p2, st.recipe, st.mot,
                                   oss)
        return y2[perm_21]

    def forward(self, hidden_states):
        B, T, H = hidden_states.shape
        x = hidden_states.reshape(-1, H)
        st = self.experts
        recipe = st.recipe
        if self.router_w32 is not None and x.dtype not in (torch.float32, torch.float64):
            logit = F.linear(x.float(), self.router_w32)
        else:
            logit = F.linear(x, self.router_w)
        prob = torch.softmax(logit, dim=-1, dtype=torch.float32 if x.dtype != torch.float64 else torch.float64)
        with torch.no_grad():
            live = torch.topk(prob, self.top_k, dim=-1).indices
        if self.instr is None:
            idx = live
        else:
            idx = self.instr
            with torch.no_grad():
                different = (torch.sort(live, -1).values != torch.sort(idx, -1).values).any(-1).sum()
                self.changes = different if self.changes is None else self.changes + different
        self.last_idx = idx.detach()
        w = prob.gather(-1, idx)
        w = w / w.sum(-1, keepdim=True)
        shared = torch.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)

        plane = getattr(idx, "_plane_reuse", None) if REUSE["active"] else None   # 09-29 tt3 fixes (--pv-riuso)
        if plane is None:
            plane = Plane(idx, st.E)
            if REUSE["active"]:
                self.last_idx._plane_reuse = plane
        if recipe.rotate:
            xg = rotate(x.to(torch.float64 if recipe.dtype_gemm == torch.float64 else torch.float32)).to(recipe.dtype_gemm)
        else:
            xg = x.to(recipe.dtype_gemm)
        oss = st.oss
        if oss is not None:
            oss.watch(xg)
        x_ord = _Collect.apply(xg, plane.order, plane.inverse, self.top_k)
        dg, du, dd = st.scale("gate"), st.scale("up"), st.scale("down")
        if getattr(st.mot, "gate_up_together", False) and st.mot.can_merge(st.fam["gate"], st.fam["up"], recipe):
            g, u = GateUpFamilies.apply(x_ord, dg, du, st.fam["gate"], st.fam["up"], plane, recipe, st.mot, oss)
        else:
            g = ExpertFamily.apply(x_ord, dg, st.fam["gate"], plane, recipe, st.mot, oss)
            u = ExpertFamily.apply(x_ord, du, st.fam["up"], plane, recipe, st.mot, oss)
        c2: dict = {}
        if st.fam2:                         # 10-09 (fix P1) 2nd plane of the experts: up += p2(up), gate += p2(gate)
            u2 = self._second_plane("up", x_ord, idx, plane, c2, oss)
            if u2 is not None:
                u = u + u2.to(u.dtype)
            g2 = self._second_plane("gate", x_ord, idx, plane, c2, oss)
            if g2 is not None:
                g = g + g2.to(g.dtype)
        cardan = self._modules.get("cardan")   # 10-09 CARDAN-KD (cardan_kd.py): absent = bit-identical
        if cardan is not None:
            cg, cu = cardan(x_ord, plane)
            g = g + cg.to(g.dtype)
            u = u + cu.to(u.dtype)
        a = (F.silu(g) * u).to(recipe.dtype_gemm)
        if oss is not None:
            oss.watch(a)
        o = ExpertFamily.apply(a, dd, st.fam["down"], plane, recipe, st.mot, oss)
        if st.fam2:                         # 10-09: down += p2(down) BEFORE the router weight (like the engine)
            o2 = self._second_plane("down", a, idx, plane, c2, oss)
            if o2 is not None:
                o = o + o2.to(o.dtype)
        if COMBINE_FUSED and isinstance(st.mot, TritonEngine) and o.dtype == torch.float32 \
                and w.dtype == torch.float32:
            y = CombineExperts.apply(o, w, plane.order, plane.inverse, self.top_k)
        else:
            wo = w.reshape(-1)[plane.order].to(o.dtype if o.dtype != torch.bfloat16 else torch.float32)
            o = o.to(wo.dtype) * wo.unsqueeze(-1)
            y = _Scatter.apply(o, plane.order, plane.inverse, self.top_k)
        rexp = self._modules.get("rexp")       # 10-03 recovery experts (e2e_rexp): absent = bit-exact identical
        if rexp is not None:
            return (y.to(x.dtype) + shared + rexp(x)).reshape(B, T, H)
        return (y.to(x.dtype) + shared).reshape(B, T, H)


class ExpertLayer:
    """The three families of ONE layer: bytes + d0 + theta. theta fp32 (E, R, NG), the only trained parameters."""

    def __init__(self, fam: dict, recipe: Recipe, mot, sentinel: Optional[torch.Tensor] = None,
                 oss: Optional[Observer] = None):
        self.fam = fam
        self.recipe = recipe
        self.mot = mot
        self.E = fam["gate"].E
        self.theta = {n: torch.zeros_like(f.d0, dtype=torch.float32, requires_grad=True) for n, f in fam.items()}
        self.sentinel = sentinel
        self.oss = oss
        # 2026-10-09 (fix P1) second plane of the experts: {gate/up/down: FamilyData with E2 + 1 experts (the last one is
        # NULL: bytes 0, d0 0, it receives the assignments of experts without a 2nd plane)} and {gate/up/down: LongTensor
        # (E,) expert -> slot (E2 = null)}. OUTSIDE fam/theta: frozen (no parameter, no key, no export of the codes).
        self.fam2: dict = {}
        self.map2: dict = {}

    def scale2(self, name: str) -> torch.Tensor:
        """Scale of the experts' 2nd plane: fp16(d0) (= d0, already fp16 in the file), in the dtype of the products."""
        d = self.fam2[name].d0
        return d.to(torch.float64) if self.recipe.dtype_gemm == torch.float64 else d

    def scale(self, name: str) -> torch.Tensor:
        if REUSE["active"] and not torch.is_grad_enabled():   # 09-29 tt3 fixes (--pv-riuso): outside the graph, reused
            f, th = self.fam[name], self.theta[name]
            k = (id(self), name)
            signature = (id(f.d0), f.d0._version, id(th), th._version, id(self.sentinel), self.recipe.dtype_gemm)
            v = REUSE["scale"].get(k)
            if v is None or v[0] != signature:
                v = REUSE["scale"][k] = (signature, self._scale(name))
            return v[1]
        return self._scale(name)

    def _scale(self, name: str) -> torch.Tensor:
        f = self.fam[name]
        s = f.d0 * torch.exp(self.theta[name])
        r = RoundFP16.apply(s, self.sentinel)
        return r.to(torch.float64) if self.recipe.dtype_gemm == torch.float64 else r


# ═══════════════════════════════════ ternary dense tensors (2026-09-28, --densi-ternari) ═══════════════════════════════════
def _entry_dense_allowed(entry: dict, name: str) -> None:
    """File <-> HF relations that TernaryLinear supports: identity, permuted rows (heads of attn_gate/attn_qkv),
    permuted columns (ssm_out). No transposition, slice or transform (no dense tensor of Thor has one)."""
    if entry["rel"] not in ("identity", "perm_rows", "perm_columns") or entry.get("transposed") or entry.get("slice_") \
            or entry.get("f"):
        raise ValueError(f"{name}: relation {entry} not allowed for a ternary dense tensor")


class TernaryLinear(torch.nn.Module):
    """2026-09-28 --densi-ternari. Replaces the HF `nn.Linear` of ONE ternary dense tensor of the GGUF with the
    experts' machinery using a SINGLE expert: y_hf = [x_file W_file(d)^T] mapped back to HF order, with
      x_file = x_hf[:, perm] (perm_columns relation, ssm_out), otherwise x_hf;
      W_file = R^T (trit x d) for TQ1_0R_1024 (input rotated like the experts: recipe B = R x in fp32 -> fp16),
               trit x d for TQ1_0 (input in the dtype of the products, like the experts' down);
      y_hf = y_file[:, inv] (perm_rows relation, attn_gate/attn_qkv: W_hf = W_file[inv]).
    The product is `ExpertFamily` on a Plane with ONE expert (all tokens to expert 0): same kernels, same dL/dd
    (per-group sum of trit (.) dY^T X_rot), same engine as the layer (`es.mot`, read at every call: engine V
    installed by e2e_pv_codes applies here too), same scale d = fp16(d0 exp(theta)) (`es.scale`), same recipe
    (`es.recipe`, changeable via `imposta_ricetta`). The module has no parameters (theta lives in ExpertLayer)."""

    def __init__(self, es: "ExpertLayer", name: str, entry: dict, outside: int, inside: int,
                 name2: Optional[str] = None, name3: Optional[str] = None, name4: Optional[str] = None):
        super().__init__()
        _entry_dense_allowed(entry, name)
        self.es = es
        self.name = name
        # 2026-09-29 second plane (tensor <name>2 in the GGUF): y_file = x W1(d1)^T + x W2(d2)^T, same file <-> HF
        # relation (the engine writes it in the same order as the first plane), its own scales and codes. None = as always.
        self.name2 = name2
        self.name3 = name3          # 09-30 third plane: + x W3(d3)^T after the second (engine order); None = as always
        self.name4 = name4          # 10-09 fourth plane: + x W4(d4)^T after the third (engine order); None = as always
        self.in_features, self.out_features = int(inside), int(outside)
        px = iy = None
        if entry["rel"] in ("perm_rows", "perm_columns"):
            from fucina.ternary import gguf_layer_loader as CSG  # read-only
            perm = np.asarray(CSG.PERM[entry["perm_npz"]], dtype=np.int64)
            if entry["rel"] == "perm_columns":
                px = torch.from_numpy(perm.copy())                 # x_file[:, c] = x_hf[:, perm[c]]
            else:
                inv = np.empty_like(perm)
                inv[perm] = np.arange(len(perm))
                iy = torch.from_numpy(inv)                         # y_hf[:, j] = y_file[:, inv[j]]
        # non-persistent integer buffers: `.to(dtype)` of the layer does not touch them, `.to(device)` moves them
        self.register_buffer("perm_x", px, persistent=False)
        self.register_buffer("inv_y", iy, persistent=False)

    @staticmethod
    def _input(xf, fam, recipe):
        if fam.rotated and recipe.rotate:
            return rotate(xf.to(torch.float64 if recipe.dtype_gemm == torch.float64 else torch.float32)).to(recipe.dtype_gemm)
        return xf.to(recipe.dtype_gemm)

    def forward(self, x):
        es = self.es
        fam = es.fam[self.name]
        recipe = es.recipe
        shape = x.shape
        xf = x.reshape(-1, shape[-1])
        if self.perm_x is not None:
            xf = xf[:, self.perm_x]
        xg = self._input(xf, fam, recipe)
        if es.oss is not None:
            es.oss.watch(xg)
        if (self.name2 is not None or self.name4 is not None) and _DP.active():   # 10-01 folded dense: off = as always
            names = [self.name] + [n for n in (self.name2, self.name3, self.name4) if n is not None]
            fams = [es.fam[n] for n in names]
            if _DP.allowed(fams, recipe, es.mot, xg):
                from fucina.kernels import qat as _KQ          # 10-09 E2E_KQAT=cache_dense: theta key of the folded weight
                _KQ.KEY_THETA = tuple((id(es.theta[n]), es.theta[n]._version) for n in names) \
                    if _KQ.active("cache_dense") else None
                try:
                    y = _DP.apply(xg, [es.scale(n) for n in names], fams, recipe, es.mot, es.oss)
                finally:
                    _KQ.KEY_THETA = None
                if self.inv_y is not None:
                    y = y[:, self.inv_y]
                return y.to(x.dtype).reshape(*shape[:-1], fam.R)
            _DP.COUNTS["fallback"] += 1
        plane = plane_single(xg.shape[0], xg.device)            # 09-29 tt3 fixes: off = new Plane, as always
        y = ExpertFamily.apply(xg.contiguous(), es.scale(self.name), fam, plane, recipe, es.mot, es.oss)
        if self.name2 is not None:                 # 09-29 second plane: + x W2(d2)^T (same plan, same engine)
            fam2 = es.fam[self.name2]
            xg2 = xg if fam2.rotated == fam.rotated else self._input(xf, fam2, recipe)
            if xg2 is not xg and es.oss is not None:
                es.oss.watch(xg2)
            y = y + ExpertFamily.apply(xg2.contiguous(), es.scale(self.name2), fam2, plane, recipe, es.mot, es.oss)
        if self.name3 is not None:                 # 09-30 third plane: + x W3(d3)^T (same plan, same engine)
            fam3 = es.fam[self.name3]
            xg3 = xg if fam3.rotated == fam.rotated else self._input(xf, fam3, recipe)
            if xg3 is not xg and es.oss is not None:
                es.oss.watch(xg3)
            y = y + ExpertFamily.apply(xg3.contiguous(), es.scale(self.name3), fam3, plane, recipe, es.mot, es.oss)
        if self.name4 is not None:                 # 10-09 fourth plane: + x W4(d4)^T (same plan, same engine)
            fam4 = es.fam[self.name4]
            xg4 = xg if fam4.rotated == fam.rotated else self._input(xf, fam4, recipe)
            if xg4 is not xg and es.oss is not None:
                es.oss.watch(xg4)
            y = y + ExpertFamily.apply(xg4.contiguous(), es.scale(self.name4), fam4, plane, recipe, es.mot, es.oss)
        if self.inv_y is not None:
            y = y[:, self.inv_y]
        return y.to(x.dtype).reshape(*shape[:-1], fam.R)

    def extra_repr(self) -> str:
        p2 = f", second_plane={self.name2}" if self.name2 is not None else ""
        p2 += f", third_plane={self.name3}" if getattr(self, "name3", None) is not None else ""
        p2 += f", fourth_plane={self.name4}" if getattr(self, "name4", None) is not None else ""
        return f"in_features={self.in_features}, out_features={self.out_features}, ternary={self.name}{p2}"


def entries_dense(tens: dict, L: int, full: bool, dense_) -> dict:
    """{family: (GGUF name, map entry)} of the requested dense families PRESENT in layer L (a family of the other
    layer type, e.g. attn_q in a GDN layer, is skipped). Raises if the tensor is not ternary (TQ1_0R_1024 /
    TQ1_0) or not 2D: with --densi-ternari a k-quant must not be silently trained in place of a ternary tensor."""
    if not dense_:
        return {}
    from fucina.ternary import gguf_layer_loader as CSG  # read-only
    map_ = CSG.MAP["full" if full else "linear"]
    out = {}
    for n in dense_:
        gg = f"{FAMILIES_DENSE[n]}.weight"
        name = f"blk.{L}.{gg}"
        if gg not in map_ or name not in tens:
            continue
        t = tens[name]
        type_ = t.tensor_type.name
        if type_ not in CS.LAYOUT:
            raise ValueError(f"{name}: type {type_}, not ternary (--densi-ternari requires TQ1_0R_1024 or TQ1_0)")
        if np.asarray(t.data).ndim != 2:
            raise ValueError(f"{name}: tensor is not 2D {np.asarray(t.data).shape}")
        _entry_dense_allowed(map_[gg], name)
        out[n] = (name, map_[gg])
    # 2026-09-29 second plane: <name>2 next to each requested family present, AFTER all the first planes (the usual
    # keys and parameters stay in place). Same type and same shape as the first plane, otherwise it raises.
    for n in list(out):
        n2 = f"{n}{SUFFIX_P2}"
        name2 = f"blk.{L}.{FAMILIES_DENSE2[n2]}.weight"
        if name2 not in tens:
            continue
        t1, t2 = tens[out[n][0]], tens[name2]
        if t2.tensor_type.name != t1.tensor_type.name or np.asarray(t2.data).shape != np.asarray(t1.data).shape:
            raise ValueError(f"{name2}: type {t2.tensor_type.name} {np.asarray(t2.data).shape}, the first plane is "
                             f"{t1.tensor_type.name} {np.asarray(t1.data).shape}")
        out[n2] = (name2, out[n][1])
    # 2026-09-30 third plane: <name>3 next to each requested family present, AFTER all the *2 (keys and parameters of
    # the first and second planes stay in place). Same type and same shape as the first plane, otherwise it raises.
    for n in [m for m in out if m in FAMILIES_DENSE]:
        n3 = f"{n}{SUFFIX_P3}"
        name3 = f"blk.{L}.{FAMILIES_DENSE3[n3]}.weight"
        if name3 not in tens:
            continue
        t1, t3 = tens[out[n][0]], tens[name3]
        if t3.tensor_type.name != t1.tensor_type.name or np.asarray(t3.data).shape != np.asarray(t1.data).shape:
            raise ValueError(f"{name3}: type {t3.tensor_type.name} {np.asarray(t3.data).shape}, the first plane is "
                             f"{t1.tensor_type.name} {np.asarray(t1.data).shape}")
        out[n3] = (name3, out[n][1])
    # 2026-10-09 fourth plane: <name>4 next to each requested family present, AFTER all the *3. Same type and shape.
    for n in [m for m in out if m in FAMILIES_DENSE]:
        n4 = f"{n}{SUFFIX_P4}"
        name4 = f"blk.{L}.{FAMILIES_DENSE4[n4]}.weight"
        if name4 not in tens:
            continue
        t1, t4 = tens[out[n][0]], tens[name4]
        if t4.tensor_type.name != t1.tensor_type.name or np.asarray(t4.data).shape != np.asarray(t1.data).shape:
            raise ValueError(f"{name4}: type {t4.tensor_type.name} {np.asarray(t4.data).shape}, the first plane is "
                             f"{t1.tensor_type.name} {np.asarray(t1.data).shape}")
        out[n4] = (name4, out[n][1])
    return out


def name_p2(name_g: str) -> str:
    """"blk.3.attn_q.weight" -> "blk.3.attn_q2.weight" (and "attn_q.weight" -> "attn_q2.weight")."""
    return f"{name_g[:-len('.weight')]}{SUFFIX_P2}.weight"


def name_p3(name_g: str) -> str:
    """"blk.3.attn_q.weight" -> "blk.3.attn_q3.weight" (and "output.weight" -> "output3.weight")."""
    return f"{name_g[:-len('.weight')]}{SUFFIX_P3}.weight"


def name_p4(name_g: str) -> str:
    """"blk.3.attn_q.weight" -> "blk.3.attn_q4.weight" (2026-10-09 fourth plane; "output.weight" -> "output4.weight")."""
    return f"{name_g[:-len('.weight')]}{SUFFIX_P4}.weight"


def planes_of(tens: dict, name_g: str) -> list:
    """2026-09-30: [name_g] + the extra planes PRESENT in the GGUF, in the engine's summation order (<name>2, then
    <name>3). Applies to the dense tensors (dequantized path) and to the token_embd / output tables (engine
    second-plane-embd-output 0b81365: get_rows(te) + get_rows(te2) + get_rows(te3); mm(out) + mm(out2) + mm(out3))."""
    if not name_g.endswith(".weight"):
        return [name_g]
    # 10-09: + <name>4 (engine thor-e1-shexp: output4 `cur += mm(output4, h)` after output3; dense *4 after *3)
    return [name_g] + [p for p in (name_p2(name_g), name_p3(name_g), name_p4(name_g)) if p in tens]


def _parent_linear(layer, hf: str) -> tuple:
    """"linear_attn.in_proj_qkv.weight" -> (module that contains the Linear, name of the Linear)."""
    pref = hf.rpartition(".")[0]
    parent_n, _, lin = pref.rpartition(".")
    return (layer.get_submodule(parent_n) if parent_n else layer), lin


# ═══════════════════════════════════ trainable F32 tensors (2026-09-27, --allena-f32) ═══════════════════════════════════
# family -> GGUF tensor name without "blk.N." (canonical order: it is the order of the parameters in the Adam group)
FAMILIES_F32 = {
    "attn_norm": "attn_norm.weight", "post_attention_norm": "post_attention_norm.weight",
    "attn_q_norm": "attn_q_norm.weight", "attn_k_norm": "attn_k_norm.weight", "ssm_norm": "ssm_norm.weight",
    "ffn_gate_inp": "ffn_gate_inp.weight", "ffn_gate_inp_shexp": "ffn_gate_inp_shexp.weight",
    "ssm_a": "ssm_a", "ssm_alpha": "ssm_alpha.weight", "ssm_beta": "ssm_beta.weight",
    "ssm_conv1d": "ssm_conv1d.weight", "ssm_dt": "ssm_dt.bias",
}
F32_GLOBAL = {"output_norm": "output_norm.weight"}


def families_f32(entries) -> tuple:
    """"a,b" or an iterable -> canonical tuple (order of FAMILIES_F32, then output_norm). Empty = none (default:
    bit-exact identical behaviour). Also accepts the name with its suffix (ssm_dt.bias, attn_norm.weight)."""
    if isinstance(entries, str):
        entries = [v for v in entries.split(",")]
    requested = set()
    for v in entries:
        v = v.strip()
        if not v:
            continue
        for suf in (".weight", ".bias"):
            if v.endswith(suf):
                v = v[:-len(suf)]
        if v not in FAMILIES_F32 and v not in F32_GLOBAL:
            raise ValueError(f"unknown F32 family {v!r}: {sorted(FAMILIES_F32) + sorted(F32_GLOBAL)}")
        requested.add(v)
    return tuple(f for f in list(FAMILIES_F32) + list(F32_GLOBAL) if f in requested)


def family_of_tensor_f32(name: str) -> str:
    """2026-09-28: GGUF tensor name ("blk.12.ffn_gate_inp.weight", "output_norm.weight") -> F32 family
    ("ffn_gate_inp", "output_norm"); raises for a name that does not belong to a trainable F32 family."""
    for fam, suf in F32_GLOBAL.items():
        if name == suf:
            return fam
    parts = name.split(".", 2)
    if len(parts) == 3 and parts[0] == "blk" and parts[1].isdigit():
        for fam, suf in FAMILIES_F32.items():
            if parts[2] == suf:
                return fam
    raise ValueError(f"{name!r}: not the name of a trainable F32 tensor")


def _transform_torch(m: torch.Tensor, entry: Optional[dict], inv: Optional[torch.Tensor]) -> torch.Tensor:
    """From the GGUF value (numpy order of the file) to the HF parameter: the SAME algebra as `load_layer_gguf._apply`
    in torch (differentiable). Subtracting 1 and the permutations are exact: same bits as numpy (verified at
    construction); log(-v) (ssm_a) goes through `_NumpyValue` for the value and uses this only in the backward pass."""
    if entry is None:
        return m
    rel = entry["rel"]
    if rel == "identity":
        out = m
    elif rel == "transform":
        out = m - 1.0 if entry["f"] == "plus_one" else torch.log(-m)
    elif rel in ("perm_rows", "perm_1d"):
        out = m[inv]
        if entry.get("f") == "neg_exp":
            out = torch.log(-out)
    elif rel == "perm_columns":
        out = m[:, inv]
    else:
        raise ValueError(f"unknown relation: {rel}")
    if entry.get("transposed"):
        out = out.T
    if entry.get("slice_") == "squeeze":
        out = out[:, None, :]
    if entry["hf"].endswith("shared_expert_gate.weight") and out.ndim == 1:
        out = out[None, :]
    return out


class _NumpyValue(torch.autograd.Function):
    """Value = `load_layer_gguf._apply` (numpy, the loader of the exported GGUF: same bits on reload,
    even on GPU, where an fp32 log may differ from numpy by 1 ulp); gradient = that of `_transform_torch`."""

    @staticmethod
    def forward(ctx, m, value, entry, inv):
        ctx.save_for_backward(m)
        ctx.entry, ctx.inv = entry, inv
        return value.clone()

    @staticmethod
    def backward(ctx, g):
        (m,) = ctx.saved_tensors
        with torch.enable_grad():
            mm = m.detach().requires_grad_(True)
            (gm,) = torch.autograd.grad(_transform_torch(mm, ctx.entry, ctx.inv), mm, g)
        return gm, None, None, None


class F32Tensor:
    """ONE trained F32 tensor of the GGUF. `master` = the file values (numpy order of the GGUF, fp32; fp64 only with
    the 64-bit test recipes), a leaf with a gradient: it is what Adam updates and what the export writes IN PLACE
    of the old bytes. The HF parameter of the module is no longer an nn.Parameter: `install()` recomputes it from
    `master` (with a graph if gradients are on) before every call of the layer, so the graph-free forward and the
    backward recomputation see the SAME value (saved routing and change counter unchanged)."""

    def __init__(self, name: str, family: str, L: int, entry: Optional[dict], value_np: np.ndarray, device,
                 dtype_master: torch.dtype, module=None, attribute: Optional[str] = None,
                 dtype_module: Optional[torch.dtype] = None, rows: Optional[slice] = None):
        self.name, self.family, self.L, self.entry = name, family, L, entry
        self.shape_file = tuple(value_np.shape)
        self.rows = rows
        v = value_np if rows is None else value_np[rows]
        self.master = torch.from_numpy(np.array(v, dtype=np.float32, copy=True)).to(device=device,
                                                                                    dtype=dtype_master)
        self.master.requires_grad_(True)
        self.module, self.attribute, self.dtype_module = module, attribute, dtype_module
        self.inv = None
        if entry is not None and entry["rel"] in ("perm_rows", "perm_1d", "perm_columns"):
            from fucina.ternary import gguf_layer_loader as CSG  # read-only
            perm = CSG.PERM[entry["perm_npz"]]
            inv = np.empty_like(perm)
            inv[perm] = np.arange(len(perm))
            self.inv = torch.from_numpy(inv.astype(np.int64)).to(device)
        self._numpy = entry is not None and entry.get("f") == "neg_exp"
        self._cache = (None, None)

    def _value_numpy(self) -> torch.Tensor:
        ver = self.master._version
        if self._cache[0] != ver:
            from fucina.ternary import gguf_layer_loader as CSG  # read-only
            a = CSG._apply(self.entry, self.master.detach().cpu().numpy())
            self._cache = (ver, torch.from_numpy(np.array(a, copy=True)).to(self.master.device))
        return self._cache[1]

    def value_hf(self) -> torch.Tensor:
        if self._numpy:
            if torch.is_grad_enabled() and self.master.requires_grad:
                return _NumpyValue.apply(self.master, self._value_numpy(), self.entry, self.inv)
            return self._value_numpy()
        return _transform_torch(self.master, self.entry, self.inv)

    def install(self) -> None:
        v = self.value_hf()
        if self.attribute == "router_w":
            self.module.router_w = v.to(self.dtype_module)
            if self.module.router_w32 is not None:
                self.module.router_w32 = v.to(torch.float32)
        elif self.module is not None:
            setattr(self.module, self.attribute, v.to(self.dtype_module))

    def value_file(self) -> np.ndarray:
        """fp32 numpy with the shape of the tensor in the GGUF (for the export)."""
        if self.rows is not None:
            raise RuntimeError(f"{self.name}: trained on a subset of experts (tests), not exportable")
        return self.master.detach().to(torch.float32).cpu().numpy().reshape(self.shape_file)


def _module_e_attribute(layer, hf: str):
    """HF name relative to the layer -> (module, attribute) where the forward pass reads it."""
    if hf == "mlp.gate.weight":
        return layer.mlp, "router_w"                   # MoEStudente: the router is `router_w` (+ `router_w32`)
    pref, _, attr = hf.rpartition(".")
    return layer.get_submodule(pref), attr


# ═══════════════════════════════════ loading from the GGUF ═══════════════════════════════════
def config_text():
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(str(SRC_HF)).text_config
    cfg._attn_implementation = "sdpa"
    if any(v in os.getenv("E2E_KQAT", "") for v in ("flash1", "flash3", "attn1", "attn3")):   # 10-09 (off: sdpa)
        from fucina.kernels import flash_qat as _FQ
        _FQ.install(cfg)
    return cfg


def weights_non_experts(tens: dict, L: int, full: bool, skip=()) -> dict:
    """{HF name relative to the layer: fp32 tensor} for ALL the tensors of the layer except the routed
    experts. Same loop, same map and same functions as `load_layer_gguf.weights_layer_from_gguf`
    (the verified loader), skipping the `per_expert` entries. 2026-09-28 `skip`: GGUF names without "blk.N." NOT to
    dequantize (--densi-ternari: those Linears become `TernaryLinear` on the bytes); empty = as always."""
    from fucina.ternary import gguf_layer_loader as CSG  # read-only
    from fucina.ternary.gguf_dequant import dequant_np  # read-only
    map_ = CSG.MAP["full" if full else "linear"]
    out = {}
    for fam, entry in map_.items():
        name_g = f"blk.{L}.{fam}"
        if name_g not in tens or entry.get("per_expert") or fam in skip:
            continue
        W = dequant_np(tens[name_g])
        if name_g.endswith(".weight") and name_p2(name_g) in tens:      # 09-29 second plane: W1 + W2 (fp32)
            W = np.asarray(W, dtype=np.float32) + np.asarray(dequant_np(tens[name_p2(name_g)]), dtype=np.float32)
        if name_g.endswith(".weight") and name_p3(name_g) in tens:      # 09-30 third plane: (W1 + W2) + W3 (fp32)
            W = np.asarray(W, dtype=np.float32) + np.asarray(dequant_np(tens[name_p3(name_g)]), dtype=np.float32)
        if name_g.endswith(".weight") and name_p4(name_g) in tens:      # 10-09 fourth plane: ((W1+W2)+W3) + W4 (fp32)
            W = np.asarray(W, dtype=np.float32) + np.asarray(dequant_np(tens[name_p4(name_g)]), dtype=np.float32)
        # copy: for F32 tensors `_apply` may return a view of the memmap (read-only)
        out[entry["hf"]] = torch.from_numpy(np.array(CSG._apply(entry, W), dtype=np.float32, copy=True))
    return out


def map_second_plane(tens: dict, L: int, name_fam: str, E: int, E2: int) -> np.ndarray:
    """2026-10-09 (fix P1): (E,) int64 expert -> slot of the 2nd plane, -1 = none. With ffn_X_exps2_map (F32 [E]) it is
    read; without it, a prefix of the first E2 experts (engine semantics). Raises on non-integer or out-of-range values."""
    nm = f"blk.{L}.{name_fam}{SUFFIX_E2}_map"
    if nm not in tens:
        return np.where(np.arange(E) < E2, np.arange(E), -1).astype(np.int64)
    v = np.asarray(tens[nm].data, dtype=np.float64).reshape(-1)
    if v.size != E or not np.all(np.isfinite(v)) or not np.all(v == np.round(v)) or v.min() < -1 or v.max() > E2 - 1:
        raise ValueError(f"{nm}: invalid map (len {v.size}, expected {E}; range [{v.min()}, {v.max()}], E2 {E2})")
    m = v.astype(np.int64)
    used = m[m >= 0]
    if len(np.unique(used)) != len(used):
        raise ValueError(f"{nm}: two experts on the same slot of the 2nd plane")
    return m


def seconds_planes_experts(tens: dict, L: int, fam: dict, device, experts: Optional[slice] = None) -> dict:
    """2026-10-09 (fix P1): {gate/up/down: (FamilyData of the 2nd plane with E2 + 1 experts, LongTensor (E,) expert -> slot)}
    for the ffn_X_exps2 tensors PRESENT in layer L. The extra expert (index E2) is NULL (bytes 0, d0 0: weight exactly
    0) and receives the assignments of the experts without a 2nd plane: output 0, like `mask` in the engine. Same type
    as the first plane (same input rotation) and same rows/columns, otherwise it raises."""
    out = {}
    for n, name_fam in FAMILIES.items():
        name2 = f"blk.{L}.{name_fam}{SUFFIX_E2}.weight"
        nm = f"blk.{L}.{name_fam}{SUFFIX_E2}_map"
        if name2 not in tens:
            if nm in tens:
                raise ValueError(f"{nm} present without {name2}")
            continue
        f1 = fam[n]
        f2 = family_from_tensor(tens[name2], f"{n}{SUFFIX_E2}", "cpu")
        E2 = f2.E
        if f2.type_ != f1.type_ or (f2.R, f2.NBR) != (f1.R, f1.NBR):
            raise ValueError(f"{name2}: {f2.type_} ({f2.R}, {f2.NBR}), the first plane is {f1.type_} ({f1.R}, {f1.NBR})")
        E_file = int(np.asarray(tens[f"blk.{L}.{name_fam}.weight"].data).shape[0])
        m = map_second_plane(tens, L, name_fam, E_file, E2)
        if experts is not None:
            m = m[experts]
        m = np.where(m >= 0, m, E2)                                  # no 2nd plane -> null expert
        blocks = torch.cat([f2.blocks, torch.zeros_like(f2.blocks[:1])], 0)
        d0 = torch.cat([f2.d0, torch.zeros_like(f2.d0[:1])], 0)
        out[n] = (FamilyData(f2.name, f2.type_, blocks.to(device), d0.to(device), f2.layer),
                  torch.from_numpy(m).to(device))
    return out


def build_layer(cfg, tens: dict, L: int, recipe: Recipe, mot, device, sentinel=None,
                      experts: Optional[slice] = None, oss: Optional[Observer] = None, dense_=()):
    """Qwen3_5MoeDecoderLayer with the non-experts from the GGUF (no original bf16 weight: if one is missing it
    raises) and `mlp` replaced by `MoEStudente` on the experts' bytes.
    2026-09-28 `dense_` (--densi-ternari; empty = as always, bit-exact): the requested ternary dense families present
    in the layer are NOT dequantized: their bytes become single-expert families in `ExpertLayer.fam` (after
    gate/up/down, order of FAMILIES_DENSE: theta, keys and registers see them like the experts) and their Linear a
    `TernaryLinear`."""
    from accelerate import init_empty_weights
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeDecoderLayer
    with init_empty_weights():
        layer = Qwen3_5MoeDecoderLayer(cfg, layer_idx=L)
    full = cfg.layer_types[L] == "full_attention"
    dense = entries_dense(tens, L, full, dense_)
    if dense:
        sd = weights_non_experts(tens, L, full, skip={name.split(".", 2)[2] for name, _ in dense.values()})
    else:
        sd = weights_non_experts(tens, L, full)
    missing, unexpected = layer.load_state_dict(sd, strict=False, assign=True)
    hf_dense = {v["hf"] for _, v in dense.values()}
    missing = [k for k in missing if not k.startswith("mlp.experts.") and k not in hf_dense]
    if missing or unexpected:
        raise RuntimeError(f"layer {L}: GGUF weights missing {missing} / unexpected {unexpected}")
    mlp = layer.mlp
    fam = {n: family_from_tensor(tens[f"blk.{L}.{FAMILIES[n]}.weight"], n, device, experts=experts)
           for n in FAMILIES}
    for n, (name_g, _) in dense.items():
        fam[n] = family_from_tensor(tens[name_g], n, device)
    router = mlp.gate.weight if experts is None else torch.nn.Parameter(mlp.gate.weight[experts].clone())
    es = ExpertLayer(fam, recipe, mot, sentinel, oss)
    for n, (f2, m2) in seconds_planes_experts(tens, L, fam, device, experts).items():   # 10-09 fix P1
        es.fam2[n], es.map2[n] = f2, m2
    moe = MoEStudente(router, mlp.shared_expert, mlp.shared_expert_gate, es, cfg.num_experts_per_tok)
    router32 = router.detach().to(torch.float32).clone() if recipe.router_fp32 else None   # F32 from the GGUF, exact
    layer.mlp = moe
    for n, (name_g, entry) in dense.items():      # BEFORE .to: the Linear to replace still has its weight on meta
        if n in FAMILIES_DENSE2 or n in FAMILIES_DENSE3 or n in FAMILIES_DENSE4:   # they live in the first plane's TernaryLinear
            continue
        parent, name_lin = _parent_linear(layer, entry["hf"])
        lin = getattr(parent, name_lin)
        f = fam[n]
        if not isinstance(lin, torch.nn.Linear) or lin.bias is not None or (f.R, f.C) != (lin.out_features,
                                                                                         lin.in_features):
            raise RuntimeError(f"{name_g}: {entry['hf']} ({type(lin).__name__} {getattr(lin, 'out_features', '?')}x"
                               f"{getattr(lin, 'in_features', '?')}) is not the Linear of the tensor ({f.R}x{f.C})")
        n2 = f"{n}{SUFFIX_P2}"
        n3 = f"{n}{SUFFIX_P3}"
        n4 = f"{n}{SUFFIX_P4}"
        setattr(parent, name_lin, TernaryLinear(es, n, entry, lin.out_features, lin.in_features,
                                                 name2=n2 if n2 in dense else None,
                                                 name3=n3 if n3 in dense else None,
                                                 name4=n4 if n4 in dense else None))
    layer = layer.to(device=device, dtype=recipe.dtype_residual)
    if recipe.non_experts_x3:                       # 09-27 B3: AFTER .to (the bf16 hi/lo are not buffers: .to does not touch them)
        from fucina.kernels import bf16x3 as X3
        layer.linear_x3 = X3.replace_linear(layer)
    elif recipe.non_experts_k2:                     # 09-27 Bk2 (noise arm of the B3 gate)
        from fucina.kernels import bf16x3 as X3
        layer.linear_x3 = X3.replace_linear(layer, class_=X3.K2Linear)
    if router32 is not None:
        layer.mlp.router_w32 = router32.to(device)
    for p in layer.parameters():
        p.requires_grad_(False)
    return layer.eval()


class TracedTensors(dict):
    """2026-10-09 (fix P1): name -> ReaderTensor dictionary that RECORDS every read (`[]`, `get`). `in` does not count:
    whoever only asks whether a tensor exists does not use it. Used by the coverage gate (`check_coverage`)."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.read: set = set()

    def __getitem__(self, k):
        v = super().__getitem__(k)
        self.read.add(k)
        return v

    def get(self, k, default=None):
        if k in self:
            self.read.add(k)
        return super().get(k, default)


# 2026-10-09 (fix P1) EXPLICIT ALLOW-LIST of the coverage gate: GGUF tensors that the student's forward must NOT
# use because the engine does not use them in the measured graph (logits of the main decoder). Each entry: (rule,
# reason). Nothing else is extraneous: the _lra/_lrb residuals already present in the STARTING model are not in the
# list (the engine applies them, the student does not: it raises), nor are unknown extra planes (e.g. *5, exps3).
LIST_WHITE_COVERAGE = (
    ("blk.L.* with L >= n_layers of the main graph", "nextn/MTP layer (blk.40): outside the logits forward"),
    ("blk.L.* with L not loaded", "partial student (tests on 1-2 layers): coverage applies to the loaded layers"),
)


def check_coverage(tens: dict, read, layers, n_layers: int, used_after=()) -> list:
    """2026-10-09 (fix P1, fail-closed): every GGUF tensor must be READ by the construction of the student, or read
    later by a declared path (`used_after`: embedding, head, final norm), or be in LIST_WHITE_COVERAGE. Raises
    ValueError with the list of orphans: a silently ignored tensor means training a model different from the GGUF
    (DIAGNOSI_DIVARIO.md: 308 tensors, 3.06 GiB ignored). Returns the names set aside by the allow-list."""
    read = set(read) | set(used_after)
    layers = set(layers)
    orphans, white = [], []
    for n in tens:
        if n in read:
            continue
        m = re.match(r"blk\.(\d+)\.", n)
        if m and (int(m.group(1)) >= n_layers or int(m.group(1)) not in layers):
            white.append(n)
            continue
        orphans.append(n)
    if orphans:
        raise ValueError(f"student: {len(orphans)} GGUF tensors not represented in the forward (e.g. "
                         f"{sorted(orphans)[:8]}): the trained model would not be the GGUF. Allow-list: "
                         f"{[r for r, _ in LIST_WHITE_COVERAGE]}")
    return white


class Student:
    """Thor-T student from the GGUF. `layers`: the loaded indices (all for training; 1-2 for tests).
    Per-layer checkpoint: `forward_` (without a graph) saves the input of each layer and the top-k; `backward_`
    recomputes layer by layer with the SAVED routing and propagates the gradient, which accumulates in theta."""

    def __init__(self, gguf: Path = GGUF_THOR_T, layers=range(40), recipe: str = "B", engine_name: str = "triton",
                 device: str = "cuda", blk: Optional[Blocks] = None, experts: Optional[slice] = None,
                 log: Callable = print, train_f32=(), lora=None, dense_ternary=()):
        self.recipe = RECIPES[recipe]
        self.mot = engine(engine_name, blk)
        self.device = torch.device(device)
        self._log = log                                 # 10-01: log of the dequantization cache
        self.cfg = config_text()
        self.gguf = Path(gguf)
        self.sentinel = torch.zeros((), dtype=torch.bool, device=self.device)
        self.oss = Observer(self.device)
        r = GGUFReader(str(self.gguf))
        self._tens = TracedTensors({t.name: t for t in r.tensors})   # 10-09: recorded reads (coverage)
        self.layers = list(layers)
        self.modules = {}
        # 2026-09-28 --densi-ternari: ternary dense families trained as a single expert (empty = bit-exact identical)
        self.dense_ternary = families_dense(dense_ternary)
        kw_dense = {"dense_": self.dense_ternary} if self.dense_ternary else {}
        t0 = time.time()
        for L in self.layers:
            self.modules[L] = build_layer(self.cfg, self._tens, L, self.recipe, self.mot, self.device,
                                               self.sentinel, experts, self.oss, **kw_dense)
        log(f"student: {len(self.layers)} layers from {self.gguf.name}, recipe {self.recipe.name}, engine "
            f"{self.mot.name}, {time.time() - t0:.1f} s")
        self.dense_counts = {f: sum(1 for L in self.layers if f in self.experts(L).fam) for f in self.dense_ternary}
        # 2026-09-29 second plane: how many *2 per family (empty without *2 in the GGUF: new attribute, no effect)
        self.p2_counts = {f"{f}{SUFFIX_P2}": c for f in self.dense_ternary
                            if (c := sum(1 for L in self.layers if f"{f}{SUFFIX_P2}" in self.experts(L).fam))}
        if self.dense_ternary:
            n_g = sum(int(self.experts(L).fam[f].d0.numel()) for L in self.layers for f in self.dense_ternary
                      if f in self.experts(L).fam)
            log(f"student: ternary dense tensors (single expert) {self.dense_counts}, {n_g} scale groups")
            if self.p2_counts:
                n_g2 = sum(int(self.experts(L).fam[f].d0.numel()) for L in self.layers for f in self.p2_counts
                           if f in self.experts(L).fam)
                log(f"student: second plane of the dense tensors {self.p2_counts}, {n_g2} extra scale groups")
        # 2026-09-30 third plane: how many *3 per family (empty without *3 in the GGUF: no effect)
        self.p3_counts = {f"{f}{SUFFIX_P3}": c for f in self.dense_ternary
                            if (c := sum(1 for L in self.layers if f"{f}{SUFFIX_P3}" in self.experts(L).fam))}
        if self.p3_counts:
            n_g3 = sum(int(self.experts(L).fam[f].d0.numel()) for L in self.layers for f in self.p3_counts
                       if f in self.experts(L).fam)
            log(f"student: third plane of the dense tensors {self.p3_counts}, {n_g3} extra scale groups")
        # 2026-10-09 (fix P1) fourth plane of the dense tensors and second plane of the experts: counts in the log
        self.p4_counts = {f"{f}{SUFFIX_P4}": c for f in self.dense_ternary
                            if (c := sum(1 for L in self.layers if f"{f}{SUFFIX_P4}" in self.experts(L).fam))}
        if self.p4_counts:
            log(f"student: fourth plane of the dense tensors {self.p4_counts}")
        self.e2_counts = {n: c for n in FAMILIES if (c := sum(1 for L in self.layers if n in self.experts(L).fam2))}
        if self.e2_counts:
            n_e2 = {n: sum(int((self.experts(L).map2[n] < self.experts(L).fam2[n].E - 1).sum())
                           for L in self.layers if n in self.experts(L).fam2) for n in self.e2_counts}
            log(f"student: second plane of the experts, layers {self.e2_counts}, experts covered {n_e2}")
        self.norm_finale = torch.from_numpy(np.array(self._tens["output_norm.weight"].data, dtype=np.float32)) \
            .to(self.device)
        self.eps = float(self.cfg.rms_norm_eps)
        self._emb = None
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextRotaryEmbedding
        self.rotary = Qwen3_5MoeTextRotaryEmbedding(config=self.cfg).to(self.device)
        self.train_f32 = families_f32(train_f32)
        self.f32: dict = {}                       # GGUF name -> F32Tensor, in canonical order (layer, family)
        self._f32_layer: dict = {}
        self.f32_counts = {f: 0 for f in self.train_f32}
        if self.train_f32:
            self._prepare_f32(experts)
            log(f"student: F32 trained {self.f32_counts} ({sum(t.master.numel() for t in self.f32.values())} "
                f"values)")
        # 2026-09-27 --lora-nonesperti (e2e_lora.py): B A adapters on the non-expert k-quants. None or rank 0 = none
        # (no module replaced: bit-exact identical); with rank > 0, B = 0 and the forward stays as always at step 0.
        self.lora: dict = {}                      # GGUF name -> e2e_lora.AdattatoreLoRA, canonical order (layer, family)
        self.lora_cfg = None
        self.lora_counts: dict = {}
        if lora is not None and lora.rank > 0:
            from fucina.student import lora as LR
            LR.install(self, lora)
            log(f"student: LoRA rank {lora.rank} (scale {lora.scale:g}) on {self.lora_counts} "
                f"({sum(p.numel() for a in self.lora.values() for p in a.parameters())} values)")
        # 2026-09-27 layer-domain branch (--layers-trained, `freeze_layers`): None = all layers trained (as always)
        self.layers_trained: Optional[list] = None
        # 2026-10-09 (fix P1) coverage gate, fail-closed: embedding and head are read later (planes_of)
        after = planes_of(self._tens, "token_embd.weight") + planes_of(self._tens, "output.weight") + ["output_norm.weight"]
        white = check_coverage(self._tens, self._tens.read, self.layers, int(self.cfg.num_hidden_layers), after)
        log(f"student: GGUF coverage complete ({len(self._tens) - len(white)} tensors used, {len(white)} in "
            f"the allow-list)")

    # ── frozen layers (2026-09-27, layer-domain branch, --layers-trained)
    def freeze_layers(self, trained=None) -> None:
        """`trained` = indices of the trained layers; None, empty or ALL the loaded layers = no frozen layer
        (requires_grad of everything goes back to true: the usual student, bit-exact identical). In frozen layers theta,
        the F32 masters and LoRA A/B have requires_grad false: no gradient (grad stays None, AdamW and Sophia skip them:
        optimizer state bit-exact), no step V statistics (dL/dd is not computed: `_backward_family`
        need_d false) and the backward pass stops at the first trained layer (`backward_`: no dL/dx below it). The
        GLOBAL F32 tensors (output_norm, above the last layer) belong to no layer: they stay trained if requested."""
        s = sorted({int(L) for L in (trained or ())})
        outside = [L for L in s if L not in self.layers]
        if outside:
            raise ValueError(f"trained layers {outside} not loaded in the student (layers {self.layers})")
        self.layers_trained = s if s and set(s) != set(self.layers) else None
        on = set(self.layers if self.layers_trained is None else self.layers_trained)
        for L in self.layers:
            for t in self.experts(L).theta.values():
                t.requires_grad_(L in on)
            for tf in self._f32_layer.get(L, ()):
                tf.master.requires_grad_(L in on)
        for a in self.lora.values():
            for p in a.parameters():
                p.requires_grad_(a.L in on)

    def trained(self, L: int) -> bool:
        """Layer L is trained (--layers-trained; without freezing: all; L = -1, the global F32 tensors: always)."""
        return self.layers_trained is None or L < 0 or L in self.layers_trained

    # ── LoRA on the non-experts (e2e_lora.py)
    def keys_lora(self) -> list:
        return list(self.lora)

    def parameters_lora(self) -> list:
        """[A, B] per adapter, in the order of `keys_lora`: it is the order of the Adam state of the LoRA group."""
        return [p for a in self.lora.values() for p in a.parameters()]

    def values_lora(self) -> dict:
        """{GGUF name: {"A": ..., "B": ...}} (detached copies, CPU): resume and snapshot. 2026-09-27 LoRA-GA: also "A0",
        "B0" (frozen initial values) for adapters in GA form; without GA the dictionary is the same as before."""
        return {n: a.values() for n, a in self.lora.items()}

    @torch.no_grad()
    def set_lora(self, values: dict) -> None:
        """A, B (and LoRA-GA's A0, B0 if present: the DATA decide the form of the forward) of each adapter."""
        for n, v in values.items():
            self.lora[n].set_(v)

    def lora_per_export(self) -> dict:
        """{GGUF tensor name: Delta W = s B A fp32 in the shape and order of the FILE}."""
        return {n: a.delta_file() for n, a in self.lora.items()}

    @contextmanager
    def lora_frozen(self):
        from fucina.student import lora as LR
        with LR.frozen(self.lora.values()):
            yield

    def _prepare_f32(self, experts: Optional[slice]) -> None:
        """--allena-f32: for each loaded layer and requested family present in its layer type, the file value becomes
        `F32Tensor.master` and the module parameter is recomputed from it. At construction the recomputed value is
        verified BIT-EXACT against that of the usual loader (same path as `weights_non_experts`)."""
        from fucina.ternary import gguf_layer_loader as CSG  # read-only
        from fucina.ternary.gguf_dequant import dequant_np  # read-only
        dt_m = torch.float64 if self.recipe.dtype_residual == torch.float64 else torch.float32
        for L in self.layers:
            map_ = CSG.MAP["full" if self.cfg.layer_types[L] == "full_attention" else "linear"]
            layer = self.modules[L]
            for fam in self.train_f32:
                if fam in F32_GLOBAL or FAMILIES_F32[fam] not in map_:
                    continue
                name = f"blk.{L}.{FAMILIES_F32[fam]}"
                t = self._tens.get(name)
                if t is None:
                    continue
                if t.tensor_type.name != "F32":
                    raise ValueError(f"{name}: type {t.tensor_type.name}, not F32")
                entry = map_[FAMILIES_F32[fam]]
                mod, attr = _module_e_attribute(layer, entry["hf"])
                old = getattr(mod, attr)
                rows = experts if fam == "ffn_gate_inp" else None
                tf = F32Tensor(name, fam, L, entry, dequant_np(t), self.device, dt_m, mod, attr, old.dtype, rows)
                with torch.no_grad():
                    if tf._numpy:
                        a = CSG._apply(entry, tf.master.detach().to(torch.float32).cpu().numpy())
                        rif = torch.from_numpy(np.array(a, copy=True))
                    else:
                        rif = _transform_torch(tf.master.detach().to(torch.float32), entry, tf.inv)
                    if not torch.equal(rif.to(device=old.device, dtype=old.dtype), old.detach()):
                        raise RuntimeError(f"{name}: the value recomputed from the master is not the loader's")
                    if attr == "router_w" and mod.router_w32 is not None and not torch.equal(
                            rif.to(mod.router_w32.device), mod.router_w32):
                        raise RuntimeError(f"{name}: recomputed fp32 router differs")
                if attr in mod._parameters:
                    del mod._parameters[attr]              # from here on it is a plain attribute, recomputed from the master
                tf.install()
                self.f32[name] = tf
                self._f32_layer.setdefault(L, []).append(tf)
                self.f32_counts[fam] += 1
        if "output_norm" in self.train_f32:
            name = F32_GLOBAL["output_norm"]
            tf = F32Tensor(name, "output_norm", -1, None, np.asarray(self._tens[name].data), self.device, dt_m)
            if not torch.equal(tf.master.detach().to(torch.float32), self.norm_finale):
                raise RuntimeError(f"{name}: master differs from norm_finale")
            self.norm_finale = tf.master                   # exits() reads it: direct gradient
            self.f32[name] = tf
            self.f32_counts["output_norm"] += 1

    def keys_f32(self) -> list:
        return list(self.f32)

    def parameters_f32(self) -> list:
        return [t.master for t in self.f32.values()]

    def values_f32(self) -> dict:
        """{GGUF name: master (detached copy, CPU)}: resume and snapshot."""
        return {n: t.master.detach().cpu().clone() for n, t in self.f32.items()}

    @torch.no_grad()
    def set_f32(self, values: dict) -> None:
        """Master values: from a resume (master shape) or from an export (file shape; with a subset
        of experts the router is sliced as at construction)."""
        for n, v in values.items():
            tf = self.f32[n]
            v = torch.as_tensor(np.asarray(v) if isinstance(v, np.ndarray) else v)
            if tf.rows is not None and tuple(v.shape) == tf.shape_file:
                v = v[tf.rows]
            tf.master.copy_(v.reshape(tf.master.shape).to(device=tf.master.device, dtype=tf.master.dtype))

    def f32_per_export(self) -> dict:
        """{GGUF tensor name: fp32 numpy in the file shape}: the bytes that the export writes in place of the old ones."""
        return {n: t.value_file() for n, t in self.f32.items()}

    def values_f32_gguf(self, names=None) -> dict:
        """2026-09-28 --f32-restore: {name: value from the GGUF FILE} of the trainable F32 tensors (all, or `names`), read
        as at construction (`_prepare_f32`: dequant_np for the layer tensors, the raw data for output_norm), in the file
        shape: `set_f32` puts it back into the master exactly as the construction does (same bits)."""
        from fucina.ternary.gguf_dequant import dequant_np  # read-only
        out = {}
        for n in (self.f32 if names is None else names):
            family_of_tensor_f32(n)                  # raises for a name that is not of an F32 family
            t = self._tens.get(n)
            if t is None or t.tensor_type.name != "F32":
                raise KeyError(f"{n}: F32 tensor missing from the student's GGUF")
            out[n] = np.asarray(t.data) if n in F32_GLOBAL.values() else dequant_np(t)
        return out

    @contextmanager
    def f32_frozen(self):
        """For the duration of the block the masters collect no gradient (step V curvatures: dL/dd = 0 here too)."""
        before = [t.master.requires_grad for t in self.f32.values()]
        try:
            for t in self.f32.values():
                t.master.requires_grad_(False)
            yield
        finally:
            for t, r in zip(self.f32.values(), before):
                t.master.requires_grad_(r)

    # ── parameters
    def keys(self) -> list:
        """(layer, family) in parameter order: gate, up, down of each layer (as always) and, with --densi-ternari,
        right after them the dense families present in the layer (order of ExpertLayer.fam, i.e. of FAMILIES_DENSE)."""
        return [(L, n) for L in self.layers for n in self.experts(L).fam]

    def experts(self, L: int) -> ExpertLayer:
        return self.modules[L].mlp.experts

    def parameters(self) -> list:
        return [self.experts(L).theta[n] for (L, n) in self.keys()]

    @torch.no_grad()
    def scale_fp16(self) -> dict:
        """{(layer, family): fp16 scales (E, R, NG)} EXACTLY those of the current forward: the ones to export
        (order of the GGUF blocks: expert, row, block of the row, scale within the block)."""
        return {(L, n): self.experts(L).scale(n).to(torch.float16).clone() for (L, n) in self.keys()}

    def scale_per_export(self) -> dict:
        """{GGUF tensor name: fp16 numpy scales (n_blocks, scales_per_block)}: the shape that
        `e2e_codes_scale.patch_scale` expects (same order as the file's blocks, test S9)."""
        out = {}
        for (L, n), s in self.scale_fp16().items():
            fam = self.experts(L).fam[n]
            out[f"blk.{L}.{family_tensor_name(n)}.weight"] = s.cpu().numpy().reshape(-1, CS.LAYOUT[fam.type_][3])
        return out

    def check(self, tolerance_overflow: bool = False) -> dict:
        """ONE synchronization per step (2026-09-26: before, one per layer): non-finite scales (raises), fp16
        overflow in the experts' inputs/gradients (raises, unless `tolerance_overflow`: the trainer skips the
        step and halves K; the gradient of that step is NOT valid) and routing changes. Resets the
        overflow flag after reading it."""
        cs = [self.modules[L].mlp.changes for L in self.layers if self.modules[L].mlp.changes is not None]
        changes = torch.stack(cs).sum() if cs else torch.zeros((), device=self.device, dtype=torch.long)
        entries = [self.sentinel.long(), self.oss.overflow.long(), changes.to(self.sentinel.device).long()]
        if self.f32:                       # --allena-f32: finite values (and ssm_a < 0: A_log = log(-ssm_a)), same sync
            ko = [(~torch.isfinite(t.master)).any() | ((t.master >= 0).any() if t.family == "ssm_a" else False)
                  for t in self.f32.values()]
            entries.append(torch.stack(ko).any().to(self.sentinel.device).long())
        i_lora = None
        if self.lora:                      # --lora-nonesperti: finite A and B, same sync
            i_lora = len(entries)
            entries.append(torch.stack([~torch.isfinite(p).all() for p in self.parameters_lora()]).any()
                        .to(self.sentinel.device).long())
        read = torch.stack(entries).tolist()
        scale_ko, ovf, n_changes = read[:3]
        self.oss.overflow.zero_()
        if scale_ko:
            raise FloatingPointError("a scale went outside the fp16 range")
        if self.f32 and read[3]:
            bad = [n for n, t in self.f32.items() if not bool(torch.isfinite(t.master).all())
                       or (t.family == "ssm_a" and bool((t.master >= 0).any()))]
            raise FloatingPointError(f"non-finite trained F32 tensors (or ssm_a >= 0): {bad[:8]}")
        if i_lora is not None and read[i_lora]:
            bad = [n for n, a in self.lora.items() if not all(bool(torch.isfinite(p).all()) for p in a.parameters())]
            raise FloatingPointError(f"non-finite LoRA adapters: {bad[:8]}")
        if ovf and not tolerance_overflow:
            raise FloatingPointError("fp16 overflow in the experts' inputs or gradients (K too large?)")
        return {"changes_routing": int(n_changes), "overflow_fp16": bool(ovf)}

    def changes(self) -> int:
        cs = [self.modules[L].mlp.changes for L in self.layers if self.modules[L].mlp.changes is not None]
        return int(torch.stack(cs).sum()) if cs else 0

    def reset_changes(self) -> None:
        for L in self.layers:
            self.modules[L].mlp.changes = None

    def diagnostics_backward(self, on: bool = True) -> list:
        """Gate G1 (step 0): turns the Observer's collection on/off and returns the rows collected
        so far (fractions of the K-scaled gradient that are zero / subnormal in fp16, per layer and family)."""
        rows = self.oss.summary()
        self.oss.diagnostics = on
        return rows

    # ── input and output
    def embedding(self, ids: torch.Tensor) -> torch.Tensor:
        """Rows of token_embd (Q5_K) dequantized like llama.cpp's get_rows (fp32), then dtype_residual.
        With `load_embedding()` the whole table is on the device; otherwise only the rows present
        are dequantized (CPU, tests)."""
        if self._emb is not None:
            return self._emb[ids.to(self.device)].to(self.recipe.dtype_residual)
        from fucina.ternary.gguf_dequant import _dequant_raw
        unique = torch.unique(ids.cpu())
        planes = planes_of(self._tens, "token_embd.weight")
        t = self._tens[planes[0]]
        rows = torch.from_numpy(_dequant_raw(np.asarray(t.data)[unique.numpy()], t))
        for name in planes[1:]:                  # 09-30: + token_embd2/3 (engine get_rows in fp32, summed in order)
            t = self._tens[name]
            rows = rows.to(torch.float32) + torch.from_numpy(
                np.asarray(_dequant_raw(np.asarray(t.data)[unique.numpy()], t), dtype=np.float32))
        pos = torch.searchsorted(unique, ids.cpu())
        return rows[pos].to(device=self.device, dtype=self.recipe.dtype_residual)

    def load_embedding(self) -> None:
        from fucina.ternary.gguf_dequant import dequant_np
        self._emb = torch.from_numpy(self._sum_planes("token_embd.weight")).to(self.device)

    def _sum_planes(self, name_g: str) -> np.ndarray:
        """2026-09-30: fp32 dequantization of a table with its extra planes (<name>2, <name>3) summed in the engine's
        order. Without extra planes it is the usual `dequant_np` (same array)."""
        from fucina.ternary.gguf_dequant import dequant_np
        planes = planes_of(self._tens, name_g)

        def compute():
            W = dequant_np(self._tens[planes[0]])
            for p in planes[1:]:
                W = np.asarray(W, dtype=np.float32) + np.asarray(dequant_np(self._tens[p]), dtype=np.float32)
            return W
        cache = os.environ.get("E2E_CACHE_DEQUANT", "").strip()
        if cache:                               # 10-01 tt5-optimized: same array from the cache (e2e_cache_dequant)
            from fucina.student import dequant_cache as CD
            return CD.sum_planes(self._tens, name_g, planes, Path(cache), compute, log=getattr(self, "_log", print))
        return compute()

    def head(self, dtype=torch.float32) -> torch.Tensor:
        """Dequantized output.weight (Q5_K): the STUDENT's head (the teacher's is the bf16 one)."""
        return torch.from_numpy(self._sum_planes("output.weight")).to(device=self.device, dtype=dtype)   # 09-30: + output2/3

    def exits(self, x: torch.Tensor) -> torch.Tensor:
        """output_norm like llama.cpp: rmsnorm in fp32 times the GGUF weight (which already contains 1 + w)."""
        acc = torch.float64 if x.dtype == torch.float64 else torch.float32
        xf = x.to(acc)
        h = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps) * self.norm_finale.to(acc)
        return h.reshape(-1, x.shape[-1])

    # ── forward and backward with per-layer checkpoint
    def _context(self, B: int, T: int, dtype, starts: Optional[torch.Tensor] = None) -> dict:
        """Positional context. `starts` None = mode «row» (the row is ONE causal sequence, positions 0..T-1,
        like `e2e_teacher --mode row` and the forge). `starts` (B, T) 0/1 with 1 at the start of each segment of the
        cu_seqlens = mode «segments» (like `e2e_teacher --mode segments` / `e2e_dump_logit --segments`): each
        segment is a sequence of its own, positions from 0 and (GDN) memory reset. The trainer uses the SAME mode
        as the teacher (`CacheMaestro.mode`)."""
        from transformers.masking_utils import create_causal_mask

        def pos_e_mask(b: int, t: int):
            pos4 = torch.arange(t, device=self.device).view(1, 1, -1).expand(4, b, -1)
            text, pos = pos4[0], pos4[1:]
            fake = torch.zeros(b, t, self.cfg.hidden_size, device=self.device, dtype=dtype)
            pe = self.rotary(fake, pos)
            m = create_causal_mask(config=self.cfg, inputs_embeds=fake, attention_mask=None,
                                   past_key_values=None, position_ids=text)
            return pe, m, text

        if starts is None:
            pe, m, text = pos_e_mask(B, T)
            return {"mode": "row", "pe": pe, "mask": m, "text": text}
        starts = starts.cpu()
        if starts.shape != (B, T) or not bool((starts[:, 0] == 1).all()):
            raise ValueError(f"starts: shape {tuple(starts.shape)} (expected {(B, T)}), the first token must open a segment")
        segments, per_length = [], {}
        for b in range(B):
            cu = torch.nonzero(starts[b]).reshape(-1).tolist() + [T]
            for s0, s1 in zip(cu[:-1], cu[1:]):
                segments.append((b, s0, s1))
                if s1 - s0 not in per_length:
                    per_length[s1 - s0] = pos_e_mask(1, s1 - s0)
        return {"mode": "segments", "segments": segments, "per_length": per_length, "B": B, "T": T}

    def _mix_segments(self, module, type_: str, h, ctx):
        """The token mixer (full attention or GDN) segment by segment; the rest of the layer
        (norms, MoE, residuals) stays on all the tokens together."""
        rows = []
        for b in range(ctx["B"]):
            pieces = []
            for (bb, s0, s1) in ctx["segments"]:
                if bb != b:
                    continue
                hs = h[b:b + 1, s0:s1]
                if type_ == "linear_attention":
                    o = module.linear_attn(hidden_states=hs, cache_params=None, attention_mask=None)
                else:
                    pe, m, text = ctx["per_length"][s1 - s0]
                    o, _ = module.self_attn(hidden_states=hs, attention_mask=m, position_ids=text,
                                            past_key_values=None, position_embeddings=pe)
                pieces.append(o)
            rows.append(torch.cat(pieces, dim=1))
        return torch.cat(rows, dim=0)

    def _layer(self, L: int, x, ctx, instr=None):
        module = self.modules[L]
        for tf in self._f32_layer.get(L, ()):
            tf.install()                  # --allena-f32: parameters recomputed from the masters (with a graph in the backward)
        module.mlp.instr = instr
        type_ = self.cfg.layer_types[L]
        try:
            if ctx["mode"] == "row":
                o = module(x, position_embeddings=ctx["pe"],
                           attention_mask=ctx["mask"] if type_ == "full_attention" else None,
                           position_ids=ctx["text"])
                return o[0] if isinstance(o, tuple) else o
            # segments mode: SAME sequence as Qwen3_5MoeDecoderLayer.forward (transformers 5.15), with only the
            # mixer split (test S10: one segment per row == row mode, bit-exact)
            h = module.input_layernorm(x)
            x = x + self._mix_segments(module, type_, h, ctx)
            h = module.mlp(module.post_attention_layernorm(x))
            return x + (h[0] if isinstance(h, tuple) else h)
        finally:
            module.mlp.instr = None

    @torch.no_grad()
    def forward_(self, ids: torch.Tensor, x0: Optional[torch.Tensor] = None,
               starts: Optional[torch.Tensor] = None) -> dict:
        """ids (B, T). No graph. Returns {"xL": output of the last layer, "inputs": [...],
        "instr": [top-k per layer], "ctx": context}. `x0`: explicit input (tests). `starts`: see
        `_context` (segments mode)."""
        x = self.embedding(ids) if x0 is None else x0
        B, T = x.shape[:2]
        ctx = self._context(B, T, x.dtype, starts)
        inputs, instr = [], []
        for L in self.layers:
            inputs.append(x)
            x = self._layer(L, x, ctx)
            instr.append(self.modules[L].mlp.last_idx)
        return {"xL": x, "inputs": inputs, "instr": instr, "ctx": ctx}

    def backward_(self, state: dict, g: torch.Tensor) -> torch.Tensor:
        """g = dL/dxL. Layer-by-layer recomputation with the saved top-k; accumulates dL/dtheta. Returns dL/dx0.
        2026-09-27 --layers-trained (`freeze_layers`): `_backward_truncated` (stops at the first trained layer, returns None)."""
        if self.layers_trained is not None:
            return self._backward_truncated(state, g)
        for i in reversed(range(len(self.layers))):
            L = self.layers[i]
            xin = state["inputs"][i].detach().requires_grad_(True)
            with torch.enable_grad():
                y = self._layer(L, xin, state["ctx"], instr=state["instr"][i])
                y.backward(g)
            g = xin.grad
        return g

    def _backward_truncated(self, state: dict, g: torch.Tensor) -> None:
        """2026-09-27 --layers-trained: like `backward_` but from the last layer down to the FIRST trained layer (in
        forward order) and no further. The frozen layers above it propagate only dL/dx (their parameters require no
        gradient); in the first trained layer the input requires no gradient (no dL/dx: nobody below would use it);
        below it, no recomputation. The gradients of the trained layers are the SAME computations as the full backward
        (test D3 of e2e_test_dominio_strati: bit-exact). Returns None (dL/dx0 no longer exists)."""
        on = set(self.layers_trained)
        i0 = min(i for i, L in enumerate(self.layers) if L in on)
        for i in reversed(range(i0, len(self.layers))):
            L = self.layers[i]
            xin = state["inputs"][i].detach()
            if i > i0:
                xin.requires_grad_(True)
            with torch.enable_grad():
                y = self._layer(L, xin, state["ctx"], instr=state["instr"][i])
                if y.requires_grad:
                    y.backward(g)
            if i > i0:
                g = xin.grad
                if g is None:                      # no path towards the input (does not happen: the residual is there)
                    raise RuntimeError(f"layer {L}: dL/dx missing in the truncated backward")
        return None

    def gradient(self, ids: torch.Tensor, loss: Callable, x0: Optional[torch.Tensor] = None,
                  starts: Optional[torch.Tensor] = None) -> float:
        """Full step: forward without a graph, loss(h) on h = exits(xL) (scalar, ALREADY scaled by the
        caller: the K scaling of G1), backward with checkpoint. Returns the value of the loss."""
        state = self.forward_(ids, x0, starts)
        xL = state["xL"].detach().requires_grad_(True)
        with torch.enable_grad():
            val = loss(self.exits(xL))
            val.backward()
        self.backward_(state, xL.grad)
        return float(val.detach())

    def hidden(self, ids: torch.Tensor, x0: Optional[torch.Tensor] = None,
                 starts: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Output after output_norm, with the full graph (tests only: reference for the checkpoint)."""
        x = self.embedding(ids) if x0 is None else x0
        ctx = self._context(x.shape[0], x.shape[1], x.dtype, starts)
        for L in self.layers:
            x = self._layer(L, x, ctx)
        return self.exits(x)


# ═══════════════════════════════════ measure (M0, GPU) ═══════════════════════════════════
def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# Tile configurations with 0 spilled registers on ALL variants (ahead-of-time compilation for gfx1151,
# 2026-09-26, e2e_test_studente S8 for the first one): the sweep candidates of M0.
CANDIDATE_GEMM = [(64, 64, 32, 8), (64, 32, 32, 4), (128, 64, 32, 8), (128, 64, 16, 4), (128, 128, 16, 8),
                  (64, 128, 16, 8), (128, 32, 32, 4)]
CANDIDATE_DD = [(64, 16, 8), (32, 16, 8), (16, 32, 8), (16, 16, 4)]


def _ids_real(B: int, T: int) -> torch.Tensor:
    """Real rows of the E2E corpus (the routing, i.e. the load per expert, depends on the text)."""
    c = torch.load(os.getenv("E2E_CORPUS", data_path("calibrazione", "e2e", "e2e_corpus_4096_tutte.pt")), weights_only=True)
    rows = torch.cat([c["domini"][d] for d in c["nomi_domini"]])
    return rows[:B, :T].clone()


def _measure_step(st: "Student", ids: torch.Tensor, repetitions: int) -> dict:
    """Forward without a graph + recomputation and backward, layer by layer, with synchronizations ONLY here (measurement)."""
    per = {L: {"forward_": [], "backward_": []} for L in st.layers}
    tot = []
    for rip in range(repetitions + 1):
        _sync()
        t_in = time.time()
        with torch.no_grad():
            x = st.embedding(ids)
            ctx = st._context(x.shape[0], x.shape[1], x.dtype)
            inputs, instr = [], []
            for L in st.layers:
                inputs.append(x)
                _sync()
                t0 = time.time()
                x = st._layer(L, x, ctx)
                _sync()
                per[L]["forward_"].append(time.time() - t0)
                instr.append(st.modules[L].mlp.last_idx)
        xL = x.detach().requires_grad_(True)
        with torch.enable_grad():
            (st.exits(xL).float().pow(2).mean() * 1e3).backward()
        g = xL.grad
        for i in reversed(range(len(st.layers))):
            L = st.layers[i]
            _sync()
            t0 = time.time()
            xin = inputs[i].detach().requires_grad_(True)
            with torch.enable_grad():
                y = st._layer(L, xin, ctx, instr=instr[i])
                y.backward(g)
            g = xin.grad
            _sync()
            per[L]["backward_"].append(time.time() - t0)
        tot.append(time.time() - t_in)
        if rip == 0:                                              # the first one compiles the kernels: discarded
            for L in st.layers:
                per[L]["forward_"].clear()
                per[L]["backward_"].clear()
            tot.clear()
        for p_ in st.parameters():
            p_.grad = None
    return {"per_layer_s": {L: {k: min(v) for k, v in d.items()} for L, d in per.items()}, "step_s": min(tot)}


def _sweep_blocks(st: "Student", ids: torch.Tensor) -> dict:
    """Time of ONE forward + dX + dL/dd call per family (gate and down of layer st.layers[0]) with
    the real routing, for each candidate configuration."""
    L = st.layers[0]
    with torch.no_grad():
        x = st.embedding(ids)
        ctx = st._context(x.shape[0], x.shape[1], x.dtype)
        st._layer(L, x, ctx)
    plane = Plane(st.modules[L].mlp.last_idx, st.experts(L).E)
    out = {}
    for name in ("gate", "down"):
        fam = st.experts(L).fam[name]
        d = st.experts(L).scale(name).detach()
        xs = torch.randn(plane.S, fam.C, device=st.device, dtype=torch.float16)
        gy = torch.randn(plane.S, fam.R, device=st.device, dtype=torch.float16)
        rows = []
        for (BS, BN, BK, w) in CANDIDATE_GEMM:
            blk = Blocks(BS=BS, BN=BN, BK=BK, num_warps=w)
            try:
                for _ in range(2):
                    _gemm_triton(xs, fam, d, None, plane, False, torch.float32, blk)
                    _gemm_triton(gy, fam, d, None, plane, True, torch.float16, blk)
                _sync()
                t0 = time.time()
                for _ in range(5):
                    _gemm_triton(xs, fam, d, None, plane, False, torch.float32, blk)
                _sync()
                t1 = time.time()
                for _ in range(5):
                    _gemm_triton(gy, fam, d, None, plane, True, torch.float16, blk)
                _sync()
                rows.append({"BS": BS, "BN": BN, "BK": BK, "warps": w, "forward_ms": (t1 - t0) / 5 * 1e3,
                              "dx_ms": (time.time() - t1) / 5 * 1e3})
            except Exception as ex:  # a configuration that does not run is recorded, not hidden
                rows.append({"BS": BS, "BN": BN, "BK": BK, "warps": w, "error": repr(ex)[:200]})
        dd_rows = []
        for (BR, BSd, w) in CANDIDATE_DD:
            blk = Blocks(BR_dd=BR, BS_dd=BSd, num_warps=w)
            try:
                for _ in range(2):
                    _dd_triton(gy, xs, fam, plane, blk)
                _sync()
                t0 = time.time()
                for _ in range(5):
                    _dd_triton(gy, xs, fam, plane, blk)
                _sync()
                dd_rows.append({"BR": BR, "BS": BSd, "warps": w, "dd_ms": (time.time() - t0) / 5 * 1e3})
            except Exception as ex:
                dd_rows.append({"BR": BR, "BS": BSd, "warps": w, "error": repr(ex)[:200]})
        # comparison: torch dense product on the same work (there is no per-expert bmm: mm per expert)
        out[name] = {"gemm": rows, "dd": dd_rows, "assignments": plane.S}
    return out


def _profile(st: "Student", ids: torch.Tensor, n: int = 30) -> dict:
    """2026-09-26: one full step (forward without a graph + recomputation + backward on the loaded layers) under
    torch.profiler, with the kernels already compiled: the `n` names with the most SELF time on the device. Used to
    choose the next lever from the numbers (experts, non-experts, attention, elementwise). If the profiler does not run
    on ROCm the error is recorded, not a number."""
    try:
        from torch.profiler import ProfilerActivity, profile
        _sync()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            st.gradient(ids, lambda h: h.float().pow(2).mean() * 1e3)
            _sync()
        for p_ in st.parameters():
            p_.grad = None
        means = list(prof.key_averages())
        field_ = "self_device_time_total"
        if sum(float(getattr(e, field_, 0.0)) for e in means) <= 0:      # no device time: CPU
            field_ = "self_cpu_time_total"
        ev = sorted(means, key=lambda e: -float(getattr(e, field_, 0.0)))
        tot = sum(float(getattr(e, field_, 0.0)) for e in ev)
        return {"token": int(ids.numel()), "field_": field_, "ms_total": tot / 1e3,
                "top": [{"name": e.key[:140], "calls": e.count, "ms": float(getattr(e, field_, 0.0)) / 1e3,
                         "fraction": float(getattr(e, field_, 0.0)) / max(tot, 1e-9)} for e in ev[:n]]}
    except Exception as ex:  # pragma: no cover  (depends on the profiler of the ROCm build)
        return {"error": repr(ex)[:300]}


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double(), b.double()
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-300))


def _trial_combine(st: "Student", ids: torch.Tensor) -> dict:
    """2026-09-26: on the real device and with the chosen engine's products, layer st.layers[0] with the fused
    combination against the torch path: same input, same routing, same output gradient (x 256, like
    a K scaling). Output, dL/dx and dL/dtheta (relative error at the maximum). On the CPU (interpreter, S13) they were
    bit-exact identical; here the order of the GPU's fp32 sums is tolerated: output and dL/dx <= 1e-5,
    dL/dtheta <= 1e-4. If it does not match, the fused path is NOT used (COMBINE_FUSED=False) and the outcome says so."""
    global COMBINE_FUSED
    old = COMBINE_FUSED
    L = st.layers[0]
    with torch.no_grad():
        x = st.embedding(ids)
        ctx = st._context(x.shape[0], x.shape[1], x.dtype)
    ris, instr, G = {}, None, None
    try:
        for fused in (False, True):
            COMBINE_FUSED = fused
            xin = x.detach().clone().requires_grad_(True)
            with torch.enable_grad():
                y = st._layer(L, xin, ctx, instr=instr)
                if instr is None:
                    instr = st.modules[L].mlp.last_idx
                if G is None:
                    gen = torch.Generator(device=y.device).manual_seed(0)
                    G = torch.randn(y.shape, generator=gen, device=y.device, dtype=torch.float32)
                (y.float() * G).sum().mul(256.0).backward()
            ris[fused] = (y.detach(), xin.grad, [st.experts(L).theta[n].grad.clone() for n in FAMILIES])
            for n in FAMILIES:
                st.experts(L).theta[n].grad = None
    finally:
        COMBINE_FUSED = old
    st.reset_changes()
    e = {"output": _rel(ris[True][0], ris[False][0]), "dx": _rel(ris[True][1], ris[False][1]),
         "dtheta": max(_rel(a, b) for a, b in zip(ris[True][2], ris[False][2]))}
    ok = e["output"] <= 1e-5 and e["dx"] <= 1e-5 and e["dtheta"] <= 1e-4
    if not ok:
        COMBINE_FUSED = False
    return {**e, "ok": ok, "in_use": "fused" if ok and old else "torch"}


def _trial_rotation(device, T: int = 8192, repetitions: int = 5) -> dict:
    """2026-09-26: eager versus compiled rotation on the real device, forward and backward, fp32 (recipes B/C
    rotate in fp32). BIT-EXACT equality is required; if it is missing, the compiled path is NOT used
    (COMPILE_ROTATE=False) and the outcome says so. Times for 1 forward + 1 backward of (T, 2048)."""
    global COMPILE_ROTATE
    old = COMPILE_ROTATE
    gen = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(T, 2048, device=device, generator=gen)
    gy = torch.randn(T, 2048, device=device, generator=gen)
    ris = {}
    for mode, flag in (("eager", False), ("compiled", True)):
        COMPILE_ROTATE = flag
        for _ in range(2):                                            # the first one compiles
            xx = x.clone().requires_grad_(True)
            rotate(xx).backward(gy)
        _sync()
        t0 = time.time()
        for _ in range(repetitions):
            xx = x.clone().requires_grad_(True)
            y = rotate(xx)
            y.backward(gy)
        _sync()
        ris[mode] = {"ms": (time.time() - t0) / repetitions * 1e3, "y": y.detach(), "g": xx.grad}
    equal = torch.equal(ris["eager"]["y"], ris["compiled"]["y"]) and torch.equal(ris["eager"]["g"],
                                                                                  ris["compiled"]["g"])
    COMPILE_ROTATE = old if equal else False
    return {"T": T, "ms_eager": ris["eager"]["ms"], "ms_compiled": ris["compiled"]["ms"],
            "equal_bit_exact": bool(equal), "in_use": "compiled" if equal and old is not False else "eager"}


def measure(argv=None) -> dict:
    """Student M0 (GPU). For each requested micro-batch: ms/token of forward + recomputation + backward on the
    REAL loaded layers (default 0-3: three GDN and one full attention, real chain), per layer and per layer type;
    projection to 40 layers FROM THE MEASURED TIMES (30 GDN + 10 full; excluding embedding, head/KL and teacher, which
    belong to other modules); memory peak; fused decode alone; engine comparison; tile sweep.
    Refuses to run if the GPU is busy (tools/gpu_occupata.sh)."""
    import argparse
    import subprocess
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--layers", default="0,1,2,3")
    ap.add_argument("--token", default="8192", help="micro-batches to try, e.g. 8192,16384,32768")
    ap.add_argument("--l-seq", type=int, default=4096)
    ap.add_argument("--recipe", default="B,C", help="one or more comma-separated recipes (A, B, C, B_ind32, B3, B3_ind32)")
    ap.add_argument("--engines", default="triton,triton_dense")
    ap.add_argument("--repetitions", type=int, default=2)
    ap.add_argument("--blocks", default="", help="JSON of the Blocks fields")
    ap.add_argument("--sweep", action="store_true", help="sweep of the tile configurations")
    ap.add_argument("--profile", action="store_true", help="torch.profiler on one step, per recipe and engine")
    ap.add_argument("--combine", default="fused", help="fused, torch or both (before/after of the lever)")
    ap.add_argument("--output", default=str(Path(data_path("e2e_student")) / "measure_m0.json"))
    ap.add_argument("--device", default="cuda", help="cpu = dry run of the code (torch engine)")
    a = ap.parse_args(argv)
    gpu = a.device == "cuda"
    if gpu:
        rc = subprocess.run([str(HERE / "tools" / "gpu_occupata.sh")], capture_output=True, text=True)
        if rc.returncode != 0:
            raise SystemExit(f"GPU busy, not measuring:\n{rc.stdout}")
    blk = Blocks(**json.loads(a.blocks)) if a.blocks else Blocks()
    layers = [int(s) for s in a.layers.split(",")]
    outcome = {"data": time.strftime("%Y-%m-%d %H:%M:%S"), "gguf": str(GGUF_THOR_T), "layers": layers,
             "recipes": a.recipe, "blocks": blk.__dict__, "runs": []}
    if gpu:
        free, total = torch.cuda.mem_get_info()
        outcome["vram_all_start_GiB"] = {"free": free / 2 ** 30, "total": total / 2 ** 30}
        outcome["rotation"] = _trial_rotation(torch.device(a.device), T=min(8192, int(a.token.split(",")[0])))
        print(json.dumps({"rotation": outcome["rotation"]}), flush=True)
    global COMBINE_FUSED
    for ricn, name_mot, comb in [(r, m, c) for r in a.recipe.split(",") for m in a.engines.split(",")
                                 for c in a.combine.split(",")]:
        COMBINE_FUSED = comb == "fused"
        if comb == "fused" and outcome.get("combine_fused_discarded"):
            continue                                        # the on-device test rejected it: not measured
        alloc0 = torch.cuda.memory_allocated() if gpu else 0
        st = Student(layers=layers, recipe=ricn, engine_name=name_mot, device=a.device, blk=blk)
        resident_layer = (torch.cuda.memory_allocated() - alloc0) / len(layers) if gpu else 0
        theta_layer = sum(st.experts(L).theta[n].numel() for (L, n) in st.keys()) * 4 / len(layers)
        if gpu:
            st.load_embedding()
        fam = st.experts(layers[0]).fam["gate"]
        d = st.experts(layers[0]).scale("gate").detach()
        if gpu and "decode_fused_gate" not in outcome:
            for _ in range(2):
                decode_weights(fam, d, torch.float16, blk)
            _sync()
            t0 = time.time()
            for _ in range(5):
                decode_weights(fam, d, torch.float16, blk)
            _sync()
            t_dec = (time.time() - t0) / 5
            outcome.setdefault("decode_fused_gate", {"ms": t_dec * 1e3,
                                                      "GB_s_written": fam.E * fam.R * fam.C * 2 / t_dec / 1e9})
        if comb == "fused" and isinstance(st.mot, TritonEngine):
            n0 = int(a.token.split(",")[0])
            trial = _trial_combine(st, _ids_real(max(1, n0 // a.l_seq), min(n0, a.l_seq)))
            outcome.setdefault("trial_combine", {})[f"{ricn}_{name_mot}"] = trial
            print(json.dumps({"trial_combine": {f"{ricn}_{name_mot}": trial}}), flush=True)
            if not trial["ok"]:                             # no «fused» run that actually measures the torch path
                outcome["combine_fused_discarded"] = True
                COMBINE_FUSED = False
                del st
                if gpu:
                    torch.cuda.empty_cache()
                continue
        for n_tok in [int(v) for v in a.token.split(",")]:
            B = max(1, n_tok // a.l_seq)
            T = n_tok // B
            ids = _ids_real(B, T)
            if gpu:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated() if gpu else 0
            try:
                m = _measure_step(st, ids, a.repetitions)
            except torch.OutOfMemoryError as ex:
                outcome["runs"].append({"recipe": ricn, "engine": name_mot, "token": B * T,
                                       "error": "OOM " + str(ex)[:200]})
                continue
            types = {L: st.cfg.layer_types[L] for L in layers}
            per_type = {}
            for L, t in m["per_layer_s"].items():
                per_type.setdefault(types[L], []).append(t["forward_"] + t["backward_"])
            mean = {k: sum(v) / len(v) for k, v in per_type.items()}
            projection = None
            if {"linear_attention", "full_attention"} <= set(mean):
                n_type = {k: sum(1 for L in range(40) if st.cfg.layer_types[L] == k) for k in mean}
                projection = sum(mean[k] * n_type[k] for k in mean) / (B * T) * 1e3
            run = {"recipe": ricn, "engine": name_mot, "combine": comb, "token": B * T, "B": B, "T": T,
                     "ms_token_per_layer": {str(L): (t["forward_"] + t["backward_"]) / (B * T) * 1e3
                                             for L, t in m["per_layer_s"].items()},
                     "per_layer_s": {str(L): t for L, t in m["per_layer_s"].items()},
                     "ms_token_step_layers_loaded": m["step_s"] / (B * T) * 1e3,
                     "projection_40_layers_ms_token": projection,
                     "peak_GiB": (torch.cuda.max_memory_allocated() - base) / 2 ** 30 if gpu else None,
                     "changes_routing": st.changes()}
            if gpu:
                # 2026-09-26: memory projection of the STUDENT to 40 layers, by component (excluding head, KL and
                # teacher, which belong to other modules). Transient of one layer = step peak minus the
                # checkpoints and the theta gradients of the loaded layers (estimate by difference, declared).
                ck = B * T * st.cfg.hidden_size * torch.tensor([], dtype=st.recipe.dtype_residual).element_size()
                peak = torch.cuda.max_memory_allocated() - base
                trans = peak - len(layers) * (ck + theta_layer)
                comp = {"resident_per_layer_GiB": resident_layer / 2 ** 30,
                        "adam_grad_m_v_per_layer_GiB": 3 * theta_layer / 2 ** 30,
                        "checkpoint_per_layer_GiB": ck / 2 ** 30,
                        "transient_one_layer_GiB": trans / 2 ** 30,
                        "embedding_fp32_GiB": st._emb.numel() * 4 / 2 ** 30 if st._emb is not None else 0.0}
                comp["total_40_layers_GiB"] = (40 * (resident_layer + 3 * theta_layer + ck) + trans) / 2 ** 30 \
                    + comp["embedding_fp32_GiB"]
                run["projection_memory"] = comp
            outcome["runs"].append(run)
            print(json.dumps(run), flush=True)
        if a.sweep and name_mot == "triton" and "sweep" not in outcome:
            ids = _ids_real(max(1, int(a.token.split(",")[0]) // a.l_seq), a.l_seq)
            outcome["sweep"] = _sweep_blocks(st, ids)
        if a.profile:
            n0 = int(a.token.split(",")[0])
            outcome.setdefault("profile", {})[f"{ricn}_{name_mot}_{comb}"] = _profile(
                st, _ids_real(max(1, n0 // a.l_seq), min(n0, a.l_seq)))
        del st
        if gpu:
            torch.cuda.empty_cache()
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(outcome, indent=1))
    print(json.dumps(outcome, indent=1))
    return outcome


if __name__ == "__main__":
    if "--measure" in sys.argv:
        measure(sys.argv[1:])
    else:
        print(__doc__)
