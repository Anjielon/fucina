#!/usr/bin/env python3
"""UNIFIED dequantization of every tensor type of the Thor GGUFs → float32
in numpy/HF order (…, out_rows, in_feat). First building block of the sequential
SchurQuant forge (plan of 2026-09-22: the teacher pass over the ALREADY quantized
model requires running the quantized weights in torch, type by type).

Types present in Thor-v20-downaggr3 (measured 2026-09-22 04:35): F32 370 · Q4_K 111 ·
TQ1_0 82 · Q8_0 70 · TQ1_0R_1024 61 · IQ4_XS 29 · Q6_K 16 · Q4_0 8 · Q5_K 6.
gguf-py can dequantize all of them except TQ1_0R_1024 (our format: uses the in-house
codec, `dequantize_blocks`) and F16/F32 (already float: passthrough).

Shape convention (verified on the file): `t.shape` is in GGUF order
[ne0=in_feat, ne1=out_rows, (ne2=experts)]; `t.data` is in numpy order
(experts, out_rows, bytes_per_row). The output has shape data.shape[:-1]+(in_feat,).

Usage: `dequant_np(t)` on a `ReaderTensor`, or `dequant_np_slice(t, e)` for
ONE SINGLE expert of a 3D tensor (avoids 1 GiB of floats per tensor: host RAM
is the scarce resource on this kind of machine).
CPU test: `python3 -m fucina.ternary.gguf_dequant` — correlation with the real bf16 for one
tensor of each type, experts included (it also discovers the layout of the fused HF
gate_up_proj).
"""
from __future__ import annotations
import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
from fucina.extdeps import ENV_GGUF_PY, add_path, data_path  # noqa: E402
add_path(ENV_GGUF_PY)   # gguf-py with the rotated ternary types (TQ1_0R_1024); see fucina.extdeps
from gguf import quants as GQ  # noqa: E402

_PASSTHROUGH = {"F32", "F16"}


def in_feat_of(t) -> int:
    return int(t.shape[0])


# 2026-09-28 (ternary-dense branch): a large TQ1_0R_1024 (token_embd/output of variant "B", 248,320 x 2048) is
# dequantized in pieces of rows: `dequantize_blocks` creates 4-5 float32 temporaries as large as the output (~2 GiB
# each for the whole vocabulary). Block by block the arithmetic is the same (no sum across blocks): same bits.
PIECE_R1024 = 1 << 24                      # elements per piece (64 MiB float32); below this, the usual path


def _dequant_raw(raw: np.ndarray, t) -> np.ndarray:
    """raw: bytes (…, bytes_per_row) of the type of t → float32 (…, in_feat)."""
    name = t.tensor_type.name
    if name in _PASSTHROUGH:
        return np.asarray(raw, dtype=np.float32)
    if name == "TQ1_0R_1024":
        from fucina.ternary.tq1_0r_1024_codec import dequantize_blocks
        lead = raw.shape[:-1]
        C = in_feat_of(t)
        rows = int(np.prod(lead)) if lead else 1
        if rows * C <= PIECE_R1024:
            return dequantize_blocks(raw).reshape(*lead, C)
        flat = raw.reshape(rows, raw.shape[-1])
        out = np.empty((rows, C), dtype=np.float32)
        step = max(1, PIECE_R1024 // C)
        for a in range(0, rows, step):
            b = min(rows, a + step)
            out[a:b] = dequantize_blocks(flat[a:b]).reshape(b - a, C)
        return out.reshape(*lead, C)
    out = GQ.dequantize(np.ascontiguousarray(raw), t.tensor_type)
    return np.asarray(out, dtype=np.float32)


def dequant_np(t) -> np.ndarray:
    raw = np.asarray(t.data)
    out = _dequant_raw(raw, t)
    expected = tuple(raw.shape[:-1]) + (in_feat_of(t),) if t.tensor_type.name not in _PASSTHROUGH else tuple(raw.shape)
    assert out.shape == expected, f"{t.name}: shape {out.shape} ≠ expected {expected}"
    return out


def dequant_np_slice(t, e: int) -> np.ndarray:
    """A single expert e of a 3D tensor (…_exps): (out_rows, in_feat)."""
    raw = np.asarray(t.data)
    assert raw.ndim == 3, f"{t.name} is not 3D"
    return _dequant_raw(raw[e:e + 1], t)[0]


def dequant_torch(t, device="cpu", dtype=None):
    import torch
    x = torch.from_numpy(dequant_np(t))
    return x.to(device, dtype) if dtype is not None else x.to(device)


# ─────────────────────────────────────────────────────────────────────────────
def _test():
    import json
    PERM = dict(np.load(os.path.join(_HERE, "perm_gguf_da_hf.npz")))
    from gguf import GGUFReader
    from safetensors import safe_open
    HF = data_path("thor-src-hf")
    idx = json.load(open(f"{HF}/model.safetensors.index.json"))["weight_map"]
    r = GGUFReader(os.environ.get("THOR_GGUF", data_path("Thor-v20-downaggr3.gguf")))
    tens = {t.name: t for t in r.tensors}

    def hf(k):
        with safe_open(f"{HF}/{idx[k]}", "pt") as f:
            return f.get_tensor(k).float().numpy()

    def hf_slice_e(k, e):
        with safe_open(f"{HF}/{idx[k]}", "pt") as f:
            return f.get_slice(k)[e:e + 1][0].float().numpy()

    def corr(a, b):
        a = a.ravel(); b = b.ravel(); n = min(a.size, b.size)
        return float(np.corrcoef(a[:n], b[:n])[0, 1])

    P = "model.language_model.layers."
    # dense: (gguf name, HF key)
    dense_ = [
        ("blk.0.attn_gate.weight",      P + "0.linear_attn.in_proj_z.weight"),    # Q8_0
        ("blk.0.attn_qkv.weight",       P + "0.linear_attn.in_proj_qkv.weight"),  # Q6_K
        ("blk.0.ffn_gate_shexp.weight", P + "0.mlp.shared_expert.gate_proj.weight"),  # TQ1_0R_1024
        ("blk.11.attn_k.weight",        P + "11.self_attn.k_proj.weight"),        # Q4_K
        ("blk.3.attn_q.weight",         P + "3.self_attn.q_proj.weight"),
        ("blk.0.ffn_down_shexp.weight", P + "0.mlp.shared_expert.down_proj.weight"),
        ("blk.0.attn_norm.weight",      P + "0.input_layernorm.weight"),          # F32
    ]
    print(f"{'tensor':32s} {'type_':12s} {'shape':18s} {'corr':>7s}")
    ok = True
    for g, k in dense_:
        if g not in tens or k not in idx:
            print(f"{g:32s} (missing: gguf={g in tens} hf={k in idx})"); continue
        t = tens[g]; W = dequant_np(t); V = hf(k)
        # ⛔ the HF rows must be reordered into the converter's order (heads
        # of attn_gate/attn_qkv): bit-exact map in perm_gguf_da_hf.npz
        fam = g.split(".", 2)[2]
        if fam in PERM:
            V = V[PERM[fam]]
        c = corr(W, V)
        threshold = 0.55 if t.tensor_type.name.startswith("TQ1") else 0.98
        seg = "✅" if c >= threshold else "⛔"
        ok &= c >= threshold
        print(f"{seg} {g:30s} {t.tensor_type.name:12s} {str(W.shape):18s} {c:7.4f}  (bf16 {V.shape})")

    # experts: layout of the fused HF gate_up_proj, discovered by correlation
    print("\n── experts (one expert at a time, e=5) ──")
    e = 5
    gu = hf_slice_e(P + "0.mlp.experts.gate_up_proj", e)      # (?, ?)
    dn = hf_slice_e(P + "0.mlp.experts.down_proj", e)
    print(f"HF gate_up_proj[e]: {gu.shape} · down_proj[e]: {dn.shape}")
    Wg = dequant_np_slice(tens["blk.0.ffn_gate_exps.weight"], e)   # TQ1_0 (512, 2048)
    Wu = dequant_np_slice(tens["blk.0.ffn_up_exps.weight"], e)
    Wd = dequant_np_slice(tens["blk.0.ffn_down_exps.weight"], e)   # IQ4_XS (2048, 512)
    print(f"GGUF gate[e] {Wg.shape} up[e] {Wu.shape} down[e] {Wd.shape}")
    hypothesis = {}
    if gu.shape[0] == 2 * Wg.shape[0]:            # (1024, 2048): rows = [gate; up]
        hypothesis["gate=rows[:512]"] = corr(Wg, gu[:Wg.shape[0]]); hypothesis["up=rows[512:]"] = corr(Wu, gu[Wg.shape[0]:])
    if gu.shape[1] == 2 * Wg.shape[0]:            # (2048, 1024): columns = [gate | up], transposed
        hypothesis["gate=col[:512].T"] = corr(Wg, gu[:, :Wg.shape[0]].T); hypothesis["up=col[512:].T"] = corr(Wu, gu[:, Wg.shape[0]:].T)
    for k, v in hypothesis.items():
        print(f"  {k:20s} corr={v:7.4f}")
    cd = corr(Wd, dn) if dn.shape == Wd.shape else corr(Wd, dn.T)
    print(f"  down (IQ4_XS){'' if dn.shape == Wd.shape else ' vs HF.T'} corr={cd:7.4f}")
    ok &= cd >= 0.98 and max(hypothesis.values()) >= 0.55
    print("\n" + ("✅ dequant consistent with bf16 on all tested types" if ok else "⛔ AT LEAST ONE TYPE DOES NOT MATCH — do not use for the quantized pass"))
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if _test() else 1)
