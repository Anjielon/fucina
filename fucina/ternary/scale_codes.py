#!/usr/bin/env python3
"""E2E on the scales (E2E design note, part 2) — the three «fidelity» building blocks, CPU only.

2026-09-25. NEW file: it modifies nothing and is not imported by any existing file. It reads,
READ-ONLY, `fucina.ternary.gpu_rotation` (already verified torch FWHT) and the in-house codec
`fucina.ternary.tq1_0r_1024_codec` (op-by-op reference of llama.cpp).

Why it exists. The E2E phase trains ONLY the fp16 scales of the ternary experts, with frozen
codes. Three things must hold bit-exactly, otherwise we optimize a model different from the
one llama.cpp runs (lesson B9):

1. the student reads the trits from the SAME GGUF bytes that llama.cpp reads
   (`trit_from_blocks`: torch port of the `dequantize_row_tq1_0` decode);
2. the weight the student uses, at the initial scales, is identical to the reference
   dequantized weight (`weights_from_trit_scale`: s (x) trit per group, then unrotation for
   TQ1_0R_1024, differentiable in s);
3. the export rewrites ONLY the d fields of the blocks (2 bytes per block for TQ1_0, 16 for
   TQ1_0R_1024) in a COPY of the GGUF: every other byte stays identical
   (`patch_scale` + `verify_patch`).

Layout (ggml-common.h of the k1024 build, verified):
  TQ1_0        54 bytes / 256 weights: qs[0:48] qh[48:52] d[52:54]            (1 scale / 256)
  TQ1_0R_1024 224 bytes / 1024 weights: qs[0:192] (4 quarters x 48) qh[192:208] d[208:224]
               (8 fp16 scales, d[g] for elements g*128..g*128+127 in the ROTATED domain)

Usage:  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 19 python3 -m fucina.ternary.scale_codes --test
        (writes a toy GGUF of a few MB to $E2E_SCRATCH, default ./_e2e_scratch)
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
from fucina.extdeps import ENV_GGUF_PY, add_path, data_path  # noqa: E402
add_path(ENV_GGUF_PY)   # gguf-py with the rotated ternary types (TQ1_0R_1024); see fucina.extdeps

from gguf import GGUFReader  # noqa: E402

# type_ -> (bytes per block, weights per block, offset of d, scales per block)
LAYOUT = {"TQ1_0": (54, 256, 52, 1), "TQ1_0R_1024": (224, 1024, 208, 8)}
_POW3 = (1, 3, 9, 27, 81)


# ─────────────────────────── 1. trit decode (torch) ───────────────────────────
def _fourth(qs: torch.Tensor, qh: torch.Tensor) -> torch.Tensor:
    """(N,48),(N,4) uint8 -> (N,256) int8 in {-1,0,1}: SAME loops as dequantize_row_tq1_0
    (8-bit truncation included), natural element order."""
    out = torch.empty(qs.shape[0], 256, dtype=torch.int8)
    b = qs[:, 0:32].to(torch.int32)
    for n, p in enumerate(_POW3):
        out[:, n * 32:(n + 1) * 32] = ((((b * p) & 255) * 3) >> 8).to(torch.int8) - 1
    b = qs[:, 32:48].to(torch.int32)
    for n, p in enumerate(_POW3):
        out[:, 160 + n * 16:160 + (n + 1) * 16] = ((((b * p) & 255) * 3) >> 8).to(torch.int8) - 1
    b = qh.to(torch.int32)
    for n, p in enumerate(_POW3[:4]):
        out[:, 240 + n * 4:240 + (n + 1) * 4] = ((((b * p) & 255) * 3) >> 8).to(torch.int8) - 1
    return out


def trit_from_blocks(blocks: torch.Tensor, type_: str) -> torch.Tensor:
    """blocks (N, bytes_per_block) uint8 -> trits (N, weights_per_block) int8."""
    nb, qk, _, _ = LAYOUT[type_]
    assert blocks.dtype == torch.uint8 and blocks.shape[-1] == nb
    if type_ == "TQ1_0":
        return _fourth(blocks[:, 0:48], blocks[:, 48:52])
    return torch.cat([_fourth(blocks[:, q * 48:(q + 1) * 48], blocks[:, 192 + q * 4:192 + (q + 1) * 4])
                      for q in range(4)], dim=1)


def scale_from_blocks(blocks: torch.Tensor, type_: str) -> torch.Tensor:
    """blocks (N, bytes_per_block) uint8 -> scales (N, k) float16, the d bytes as they are."""
    nb, _, off, k = LAYOUT[type_]
    return blocks[:, off:off + 2 * k].contiguous().view(torch.float16).reshape(-1, k)


# ─────────────────────── 2. weight differentiable in the scales ───────────────────────
def weights_from_trit_scale(trit: torch.Tensor, scale: torch.Tensor, type_: str,
                       S: "torch.Tensor | None" = None) -> torch.Tensor:
    """trit (N, qk) int8 · scale (N, k) (fp32, with grad if trained) -> weights (N, qk) fp32.
    Same arithmetic as the reference: for TQ1_0R_1024 trit*d[g] in fp32, then FWHT-1024
    (with 1/32 up front) and sign S — `gpu_rotation._RT_1024_torch`, reused."""
    _, qk, _, k = LAYOUT[type_]
    W = (trit.to(torch.float32).reshape(trit.shape[0], k, qk // k) * scale.to(torch.float32).unsqueeze(-1))
    W = W.reshape(trit.shape[0], qk)
    if type_ == "TQ1_0R_1024":
        from fucina.ternary.gpu_rotation import _RT_1024_torch  # read-only
        W = _RT_1024_torch(W, S.to(torch.float32))
    return W


def scale_fp16_ste(s: torch.Tensor) -> torch.Tensor:
    """Forward: the scale rounded to fp16 (round-to-nearest-even, like the export and like
    GGML_FP32_TO_FP16). Backward: identity (straight-through). This way the trained student
    IS the exported model, at every step."""
    return s + (s.to(torch.float16).to(s.dtype) - s).detach()


# ─────────────────────────────── 3. bit-exact export ───────────────────────────────
def position_scale(t) -> dict:
    """Where the d fields of a ReaderTensor are: absolute offset in the file, number of
    blocks, layout. Blocks are in numpy order (experts, rows, blocks per row)."""
    type_ = t.tensor_type.name
    nb, qk, off, k = LAYOUT[type_]
    n_blocks = int(t.n_bytes) // nb
    assert n_blocks * nb == int(t.n_bytes)
    return {"type_": type_, "start": int(t.data_offset), "n_blocks": n_blocks, "byte_block": nb,
            "off_d": off, "k": k}


def patch_scale(dst: Path, new: dict) -> dict:
    """`new`: {tensor_name: float array (n_blocks, k) or (…, n_row_blocks*k)} — the scales
    in the tensor's block order. Writes fp16(RNE) ONLY into the d fields of `dst`, which must
    be a COPY already made (never the source file). Returns {name: number of scales changed}."""
    r = GGUFReader(str(dst))
    tens = {t.name: t for t in r.tensors}
    pos = {n: position_scale(tens[n]) for n in new}
    del r, tens
    mm = np.memmap(dst, dtype=np.uint8, mode="r+")
    changed = {}
    for name, values in new.items():
        p = pos[name]
        v16 = np.ascontiguousarray(np.asarray(values, dtype=np.float32).reshape(p["n_blocks"], p["k"])
                                   .astype(np.float16))
        if not np.isfinite(v16.astype(np.float32)).all():
            raise FloatingPointError(f"{name}: scale outside the fp16 range")
        view = mm[p["start"]:p["start"] + p["n_blocks"] * p["byte_block"]].reshape(p["n_blocks"], p["byte_block"])
        before = view[:, p["off_d"]:p["off_d"] + 2 * p["k"]].copy()
        view[:, p["off_d"]:p["off_d"] + 2 * p["k"]] = v16.view(np.uint8).reshape(p["n_blocks"], 2 * p["k"])
        changed[name] = int((before.view(np.uint16) != v16.view(np.uint16).reshape(before.shape[0], -1)).sum())
    mm.flush()
    del mm
    return changed


def verify_patch(src: Path, dst: Path, names: list, piece: int = 1 << 26) -> dict:
    """Compares the two files in pieces. All bytes OUTSIDE the d fields of the `names` tensors must
    be identical (raises otherwise). Returns the sha256 of the 'without d fields' stream (equal by
    construction if it passes) and of the codes alone (qs/qh) of the patched tensors."""
    if src.stat().st_size != dst.stat().st_size:
        raise RuntimeError(f"different sizes: {src.stat().st_size} versus {dst.stat().st_size}")
    r = GGUFReader(str(src))
    tens = {t.name: t for t in r.tensors}
    masks = []   # (start, end, byte_block, off_d, k)
    for n in names:
        p = position_scale(tens[n])
        masks.append((p["start"], p["start"] + p["n_blocks"] * p["byte_block"], p["byte_block"],
                         p["off_d"], p["k"]))
    del r, tens
    h_remainder, h_codes = hashlib.sha256(), hashlib.sha256()
    a = np.memmap(src, dtype=np.uint8, mode="r")
    b = np.memmap(dst, dtype=np.uint8, mode="r")
    n = a.shape[0]
    for i0 in range(0, n, piece):
        i1 = min(n, i0 + piece)
        x, y = np.asarray(a[i0:i1]), np.asarray(b[i0:i1])
        inside_d = np.zeros(i1 - i0, dtype=bool)
        codes = np.zeros(i1 - i0, dtype=bool)
        for (m0, m1, nb, off, k) in masks:
            j0, j1 = max(i0, m0), min(i1, m1)
            if j0 >= j1:
                continue
            rel = np.arange(j0, j1) - m0
            in_block = rel % nb
            e_d = (in_block >= off) & (in_block < off + 2 * k)
            inside_d[j0 - i0:j1 - i0] |= e_d
            codes[j0 - i0:j1 - i0] |= ~e_d
        different = (x != y) & ~inside_d
        if different.any():
            j = int(np.flatnonzero(different)[0]) + i0
            raise RuntimeError(f"byte {j} differs OUTSIDE the d fields: {int(a[j])} versus {int(b[j])}")
        h_remainder.update(x[~inside_d].tobytes())
        h_codes.update(x[codes].tobytes())
    return {"sha256_without_d": h_remainder.hexdigest(), "sha256_codes": h_codes.hexdigest(), "byte": n}


# ─────────────────────────────────── test CPU ───────────────────────────────────
def _test(scratch: Path) -> bool:
    from gguf.quants import TQ1_0 as GQ_TQ1_0
    from fucina.ternary.tq1_0r_1024_codec import dequantize_blocks as deq_r1024, S1024
    torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "2")))
    ok = True
    sources = {"TQ1_0": (data_path("Thor-v20-densiv5.gguf"), ["blk.0.ffn_gate_exps.weight",
                                                                          "blk.39.ffn_up_exps.weight"]),
             "TQ1_0R_1024": (data_path("Thor-T-ibrido-0-19-ThorT-veloce.gguf"),
                             ["blk.0.ffn_gate_exps.weight", "blk.0.ffn_down_exps.weight",
                              "blk.19.ffn_up_exps.weight"])}
    S = torch.from_numpy(S1024)
    samples = {}
    for type_expected, (path, names) in sources.items():
        r = GGUFReader(path)
        tens = {t.name: t for t in r.tensors}
        for name in names:
            t = tens[name]
            type_ = t.tensor_type.name
            nb, qk, _, k = LAYOUT[type_]
            raw = np.asarray(t.data)                       # memmap (E, rows, byte)
            for e in (0, 97, 255):
                blocks = np.ascontiguousarray(raw[e]).reshape(-1, nb)
                bt = torch.from_numpy(blocks)
                trit = trit_from_blocks(bt, type_)
                sc = scale_from_blocks(bt, type_)
                if type_ == "TQ1_0":
                    rif = GQ_TQ1_0.dequantize_blocks(blocks)                      # gguf-py
                    W = weights_from_trit_scale(trit, sc.float(), type_).numpy()
                else:
                    rif = deq_r1024(blocks)                                     # in-house codec
                    W = weights_from_trit_scale(trit, sc.float(), type_, S).numpy()
                equal = np.array_equal(W, rif)
                dmax = float(np.abs(W - rif).max())
                ok &= equal
                print(f"[T1/T2] {Path(path).name} {name} {type_} expert {e}: {blocks.shape[0]} blocks · "
                      f"torch weight == reference bit-exact: {equal} (max|diff| {dmax:.3e})")
            samples[type_] = (np.ascontiguousarray(raw[:2, :4]), t)
        del r, tens

    # T3: gradient with respect to the scales = per-group sum of q * dL/dW (EfficientQAT eq.: dw/ds = w_q)
    torch.manual_seed(0)
    blocks = torch.from_numpy(np.ascontiguousarray(samples["TQ1_0R_1024"][0]).reshape(-1, 224)[:4])
    trit = trit_from_blocks(blocks, "TQ1_0R_1024")
    s = scale_from_blocks(blocks, "TQ1_0R_1024").double().requires_grad_(True)
    x = torch.randn(1024, 7, dtype=torch.float64)
    Wd = (trit.double().reshape(4, 8, 128) * s.unsqueeze(-1)).reshape(4, 1024)
    from fucina.ternary.gpu_rotation import _RT_1024_torch
    y = _RT_1024_torch(Wd, S.double()) @ x
    L = (y ** 2).sum()
    L.backward()
    g = s.grad.clone()
    eps = 1e-6
    num = torch.zeros_like(g)
    with torch.no_grad():
        for i in range(4):
            for j in range(8):
                sp, sm = s.detach().clone(), s.detach().clone()
                sp[i, j] += eps
                sm[i, j] -= eps
                f = lambda ss: ((_RT_1024_torch((trit.double().reshape(4, 8, 128) * ss.unsqueeze(-1))
                                                .reshape(4, 1024), S.double()) @ x) ** 2).sum()
                num[i, j] = (f(sp) - f(sm)) / (2 * eps)
    rel = float(((g - num).abs().max() / num.abs().max()))
    ok &= rel < 1e-6
    print(f"[T3] autograd gradient versus finite differences (float64, 32 scales): relative error {rel:.2e}")

    # T5: recipe B (like the Vulkan rot_y shader): W_real @ x == W_rotated @ (R x), R x = FWHT(S (.) x)
    from fucina.ternary.gpu_rotation import _R_1024_torch
    r = GGUFReader(sources["TQ1_0R_1024"][0])
    t = {q.name: q for q in r.tensors}["blk.19.ffn_up_exps.weight"]
    blocks = torch.from_numpy(np.ascontiguousarray(np.asarray(t.data)[5]).reshape(-1, 224))
    trit, sc = trit_from_blocks(blocks, "TQ1_0R_1024"), scale_from_blocks(blocks, "TQ1_0R_1024").double()
    W_rot = (trit.double().reshape(-1, 8, 128) * sc.unsqueeze(-1)).reshape(512, 2048)
    W_real = _RT_1024_torch(W_rot, S.double())
    x = torch.randn(16, 2048, dtype=torch.float64)
    a, b = x @ W_real.T, _R_1024_torch(x, S.double()) @ W_rot.T
    rel = float((a - b).abs().max() / a.abs().max())
    ok &= rel < 1e-12
    print(f"[T5] expert 5 of blk.19.ffn_up: output with unrotated weight versus rotated activation (float64): "
          f"relative error {rel:.2e}")
    del r, t

    # T4: patch on a toy GGUF built from REAL blocks, then bit-exact verification
    from gguf import GGUFWriter, GGMLQuantizationType as T
    scratch.mkdir(parents=True, exist_ok=True)
    src, dst = scratch / "e2e_giocattolo.gguf", scratch / "e2e_giocattolo_patch.gguf"
    w = GGUFWriter(str(src), arch="qwen35moe")
    w.add_uint32("e2e.trial", 1)
    w.add_tensor("blk.0.ffn_gate_exps.weight", samples["TQ1_0R_1024"][0], raw_dtype=T.TQ1_0R_1024)
    w.add_tensor("blk.0.ffn_up_exps.weight", samples["TQ1_0"][0], raw_dtype=T.TQ1_0)
    w.add_tensor("blk.0.attn_norm.weight", np.arange(2048, dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    shutil.copyfile(src, dst)
    rs = GGUFReader(str(src))
    ts = {t.name: t for t in rs.tensors}
    rng = np.random.default_rng(1)
    new, expected_ = {}, {}
    for name in ("blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight"):
        t = ts[name]
        p = position_scale(t)
        blocks = np.asarray(t.data).reshape(-1, p["byte_block"])
        s0 = blocks[:, p["off_d"]:p["off_d"] + 2 * p["k"]].copy().view(np.float16).astype(np.float32)
        s1 = s0 * np.exp(rng.normal(0, 0.05, s0.shape)).astype(np.float32)
        new[name] = s1
        expected_[name] = (blocks.copy(), s1.astype(np.float16).astype(np.float32))
    changed = patch_scale(dst, new)
    rep = verify_patch(src, dst, list(new))
    rep_src = hashlib.sha256(src.read_bytes()).hexdigest()
    print(f"[T4] scales changed {changed} · bytes outside the d fields identical · {rep}")
    rd = GGUFReader(str(dst))
    td = {t.name: t for t in rd.tensors}
    for name, (blocks_src, s16) in expected_.items():
        t = td[name]
        p = position_scale(t)
        blocks_dst = np.asarray(t.data).reshape(-1, p["byte_block"])
        trit = trit_from_blocks(torch.from_numpy(np.ascontiguousarray(blocks_src)), p["type_"]).numpy()
        if p["type_"] == "TQ1_0":
            deq = GQ_TQ1_0.dequantize_blocks(np.ascontiguousarray(blocks_dst))
            expected = (trit.astype(np.float32).reshape(-1, 1, 256) * s16.reshape(-1, 1, 1)).reshape(-1, 256)
        else:
            deq = deq_r1024(np.ascontiguousarray(blocks_dst))
            expected = weights_from_trit_scale(torch.from_numpy(trit), torch.from_numpy(s16), "TQ1_0R_1024", S).numpy()
        equal = np.array_equal(deq, expected)
        ok &= equal
        print(f"[T4] {name} {p['type_']}: dequant of the patched file == trit(source) x fp16(new scale) bit-exact: "
              f"{equal} · shapes/types/offset unchanged: "
              f"{[int(x) for x in t.shape] == [int(x) for x in ts[name].shape] and t.data_offset == ts[name].data_offset}")
    print(f"[T4] sha256 source {rep_src[:16]}… (the source file is never opened for writing)")
    print("ESITO:", "ALL GREEN" if ok else "⛔ SOMETHING DOES NOT ADD UP")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    if a.test:
        sys.exit(0 if _test(Path(os.getenv("E2E_SCRATCH", str(HERE / "_e2e_scratch")))) else 1)
    ap.print_help()
