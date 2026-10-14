#!/usr/bin/env python3
"""E2E on the scales (E2E design note, part 1): the `d` fields of the GGUF ternary
blocks, read and rewritten BIT-EXACTLY, and the trits decoded in torch from the same bytes.

Written on 2026-09-25. NEW file: it neither imports nor modifies any forge file that is
running (session constraint); it depends only on the gguf-py of the k1024 build.

Two formats, layouts verified against the source (not recalled from memory):

| type_          | bytes/block | weights/block | scales/block | d field        | scale group          |
|---------------|-------------|---------------|--------------|----------------|----------------------|
| TQ1_0R_1024   | 224         | 1024          | 8 (fp16)     | bytes 208..223 | 128 ROTATED weights  |
| TQ1_0         | 54          | 256           | 1 (fp16)     | bytes 52..53   | 256 weights          |

Sources: `fucina.ternary.tq1_0r_1024_codec` (layout docstring, identical to the C of
ggml-quants.c in the k1024 build) and `pack_tq1_0_standard.py` (qs 48 + qh 4 + d 2).

Block order of a GGUF experts tensor of shape [ne0=in, ne1=out, ne2=E]:
block = (e*out + r)*nb + b, with nb = in / weights_block. Scale g of block b of
row r of expert e goes to D[e, r, b*scale_block + g]: this is the (E, out, ng) shape used
by the window-validation weight/family extraction (verified by
`e2e_test_campi_scala.py` against `gguf_dequant.dequant_np_slice`).

The trit decode repeats the integer arithmetic of `dequantize_row_tq1_0`
((b*3^n) & 255) * 3 >> 8, so the codes in VRAM ARE the GGUF bytes: no second
copy of the codes, no possibility that the trained codes diverge from the
exported ones.
"""
from __future__ import annotations

import sys

import numpy as np

from fucina.extdeps import ENV_GGUF_PY, add_path  # noqa: E402
add_path(ENV_GGUF_PY)   # gguf-py with the rotated ternary types (TQ1_0R_1024); see fucina.extdeps

LAYOUT = {
    "TQ1_0R_1024": {"byte_block": 224, "weights_block": 1024, "scale_block": 8, "off_d": 208, "group": 128,
                    "quarters": 4},
    "TQ1_0": {"byte_block": 54, "weights_block": 256, "scale_block": 1, "off_d": 52, "group": 256,
              "quarters": 1},
}


def _lay(type_: str) -> dict:
    if type_ not in LAYOUT:
        raise ValueError(f"type_ {type_!r} is not ternary: expected {sorted(LAYOUT)}")
    return LAYOUT[type_]


def blocks(raw: np.ndarray, type_: str) -> np.ndarray:
    """Tensor bytes (any shape) -> view (N_blocks, byte_block) uint8."""
    L = _lay(type_)
    b = np.ascontiguousarray(raw).reshape(-1).view(np.uint8)
    if b.size % L["byte_block"]:
        raise ValueError(f"{b.size} bytes, not a multiple of {L['byte_block']} ({type_})")
    return b.reshape(-1, L["byte_block"])


def mask_codes(type_: str) -> np.ndarray:
    """(byte_block,) bool: True on the CODE bytes (qs+qh), False on the d bytes."""
    L = _lay(type_)
    m = np.ones(L["byte_block"], dtype=bool)
    m[L["off_d"]:L["off_d"] + 2 * L["scale_block"]] = False
    return m


def read_scale(raw: np.ndarray, type_: str, E: int | None = None, n_out: int | None = None) -> np.ndarray:
    """EXACT (bit for bit) fp16 scales of the tensor. With E and n_out: shape (E, n_out, ng)."""
    L = _lay(type_)
    B = blocks(raw, type_)
    d = B[:, L["off_d"]:L["off_d"] + 2 * L["scale_block"]].copy().view(np.float16).reshape(-1)
    if E is not None:
        d = d.reshape(E, n_out, -1)
    return d


def write_scale(raw: np.ndarray, type_: str, d_fp16: np.ndarray) -> np.ndarray:
    """Copy of the bytes with ONLY the d fields replaced by `d_fp16` (any shape with the
    same number of scales, already float16: the fp16 rounding is decided by the trainer,
    because it must be the same one seen in the forward). Codes untouched."""
    L = _lay(type_)
    d = np.ascontiguousarray(d_fp16)
    if d.dtype != np.float16:
        raise TypeError(f"d must be float16 (rounded by the training), not {d.dtype}")
    if not np.isfinite(d.astype(np.float32)).all():
        raise FloatingPointError("non-finite scale: the GGUF would store it, llama.cpp would produce NaN")
    new = blocks(raw, type_).copy()
    if d.size != new.shape[0] * L["scale_block"]:
        raise ValueError(f"{d.size} scales for {new.shape[0]} blocks x {L['scale_block']}")
    new[:, L["off_d"]:L["off_d"] + 2 * L["scale_block"]] = d.reshape(-1, L["scale_block"]).view(np.uint8) \
        .reshape(new.shape[0], 2 * L["scale_block"])
    return new.reshape(np.asarray(raw).shape) if np.asarray(raw).dtype == np.uint8 else new


def verify_patch(old: np.ndarray, new: np.ndarray, type_: str, d_fp16: np.ndarray) -> dict:
    """Export checks (not tautological: they re-read the written BYTES).
    1) code bytes bit-exact; 2) d fields = d_fp16 bit-exact; 3) finite scales."""
    Bv, Bn = blocks(old, type_), blocks(new, type_)
    if Bv.shape != Bn.shape:
        raise RuntimeError(f"block shape changed {Bv.shape} -> {Bn.shape}")
    m = mask_codes(type_)
    codes_equal = bool(np.array_equal(Bv[:, m], Bn[:, m]))
    d_read = read_scale(new, type_)
    d_equal = bool(np.array_equal(d_read.view(np.uint16), np.ascontiguousarray(d_fp16).reshape(-1).view(np.uint16)))
    finite = bool(np.isfinite(d_read.astype(np.float32)).all())
    outcome = {"codes_identical": codes_equal, "d_identical_bit_exact": d_equal, "d_finite": finite,
             "blocks": int(Bv.shape[0]),
             "scale_changed": int((read_scale(old, type_).view(np.uint16) != d_read.view(np.uint16)).sum())}
    if not (codes_equal and d_equal and finite):
        raise RuntimeError(f"export NOT bit-exact: {outcome}")
    return outcome


# ───────────────────────────── trits in torch ─────────────────────────────
def _digits_torch(b, n_digits: int):
    """(…, nb) uint8 -> (…, n_digits, nb) int8 in {-1,0,1}: digit n = ((b*3^n)&255)*3>>8, -1.
    Same arithmetic as dequantize_row_tq1_0 (8-bit truncation included)."""
    import torch
    b16 = b.to(torch.int16)                    # 255*81 = 20655 < 32767
    out = [((((b16 * (3 ** n)) & 255) * 3) >> 8) - 1 for n in range(n_digits)]
    return torch.stack(out, dim=-2).to(torch.int8)


def _fourth_torch(qs, qh):
    """qs (…,48), qh (…,4) uint8 -> (…,256) int8, natural order of the 256-weight block:
    bytes 0..31 -> elements n*32+m; bytes 32..47 -> 160+n*16+m; qh -> 240+n*4+m."""
    import torch
    lead = qs.shape[:-1]
    a = _digits_torch(qs[..., 0:32], 5).reshape(*lead, 160)
    b = _digits_torch(qs[..., 32:48], 5).reshape(*lead, 80)
    c = _digits_torch(qh, 4).reshape(*lead, 16)
    return torch.cat([a, b, c], dim=-1)


def trit_torch(B, type_: str):
    """B: torch uint8 tensor (N, byte_block) -> (N, weights_block) int8 in {-1,0,1}.
    For TQ1_0R_1024 the trits are in the ROTATED domain (like the scales)."""
    import torch
    L = _lay(type_)
    if L["quarters"] == 1:
        return _fourth_torch(B[:, 0:48], B[:, 48:52])
    parts = [_fourth_torch(B[:, q * 48:(q + 1) * 48], B[:, 192 + q * 4:192 + (q + 1) * 4]) for q in range(4)]
    return torch.cat(parts, dim=-1)


def scale_torch(B, type_: str):
    """B: torch uint8 (N, byte_block) -> (N, scale_block) float16, bit for bit."""
    import torch
    L = _lay(type_)
    s = B[:, L["off_d"]:L["off_d"] + 2 * L["scale_block"]].contiguous()
    return s.view(torch.float16).reshape(B.shape[0], L["scale_block"])
