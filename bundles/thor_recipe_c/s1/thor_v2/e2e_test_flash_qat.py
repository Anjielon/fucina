#!/usr/bin/env python3
"""Tests of e2e_flash_qat against an fp64 reference (causal GQA attention, dO with a wide dynamic range).
CPU: TRITON_INTERPRET=1 HIP_VISIBLE_DEVICES= python3 e2e_test_flash_qat.py      GPU: --gpu (real shapes)
F1 forward o, F2 dq/dk/dv, for SPLIT 3 (fp32-accurate: max error / max |ref| <= 1e-5 asserted) and SPLIT 1 (reported);
also today's path (fp32 SDPA math with the additive mask, transformers) against the same fp64 reference."""
import json
import os
import sys
from pathlib import Path

if "--gpu" not in sys.argv:
    os.environ.setdefault("TRITON_INTERPRET", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import e2e_studente  # noqa: E402,F401  (interpreter patch for numpy 2)
import e2e_flash_qat as FQ  # noqa: E402


def rif64(q, k, v, do, scale):
    grp = q.shape[1] // k.shape[1]
    q, k, v = (t.double().detach().requires_grad_(True) for t in (q, k, v))
    kk, vv = k.repeat_interleave(grp, 1), v.repeat_interleave(grp, 1)
    T = q.shape[2]
    m = torch.full((T, T), float("-inf"), dtype=torch.float64, device=q.device).triu(1)
    o = torch.softmax(q @ kk.transpose(-1, -2) * scale + m, -1) @ vv          # (B, H, T, D)
    o = o.transpose(1, 2)
    (o * do.double()).sum().backward()
    return o.detach(), q.grad, k.grad, v.grad


def oggi(q, k, v, do, scale):
    grp = q.shape[1] // k.shape[1]
    q, k, v = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
    T = q.shape[2]
    m = torch.full((T, T), float("-inf"), device=q.device).triu(1)
    o = F.scaled_dot_product_attention(q, k.repeat_interleave(grp, 1), v.repeat_interleave(grp, 1), attn_mask=m,
                                       scale=scale).transpose(1, 2)
    o.backward(do)
    return o.detach(), q.grad, k.grad, v.grad


def main():
    gpu = "--gpu" in sys.argv
    dev = "cuda" if gpu else "cpu"
    B, H, HK, T, D = (2, 16, 2, 4096, 256) if gpu else (1, 4, 2, 80, 256)
    g = torch.Generator(device=dev).manual_seed(0)
    q = torch.randn(B, T, H, D, device=dev, generator=g).transpose(1, 2)        # strided like transformers
    k = torch.randn(B, T, HK, D, device=dev, generator=g).transpose(1, 2)
    v = torch.randn(B, T, HK, D, device=dev, generator=g).transpose(1, 2)
    mag = torch.exp2(-torch.randint(10, 30, (B, T, 1, 1), device=dev, generator=g).float())
    do = torch.randn(B, T, H, D, device=dev, generator=g) * mag
    scale = 1 / 16
    if gpu:   # fp64 reference on a slice (row 0, query heads 0 and 8 need all kv: keep full but T 1024)
        T2 = 1024
        q, k, v, do = q[:, :, :T2], k[:, :, :T2], v[:, :, :T2], do[:, :T2]
    R = rif64(q, k, v, do, scale)
    out = {}
    for nome, split in (("flash3", 3), ("flash1", 1)):
        qq, kk, vv = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
        o = FQ.FlashQAT.apply(qq, kk, vv, scale, split)
        o.backward(do)
        e = {}
        for n, a, r in (("o", o, R[0]), ("dq", qq.grad, R[1]), ("dk", kk.grad, R[2]), ("dv", vv.grad, R[3])):
            e[n] = float((a.double() - r).abs().max() / r.abs().max())
        out[nome] = e
        print(nome, e, flush=True)
    for nome, split in (("attn3", 3), ("attn1", 1)):
        FQ.QB = 32 if not gpu else 512
        qq, kk, vv = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
        o = FQ.AttnBlocchi.apply(qq, kk, vv, scale, split)
        o.backward(do)
        out[nome] = {n: float((a.double() - r).abs().max() / r.abs().max())
                     for n, a, r in (("o", o, R[0]), ("dq", qq.grad, R[1]), ("dk", kk.grad, R[2]), ("dv", vv.grad, R[3]))}
        print(nome, out[nome], flush=True)
    FQ.QB = 32 if not gpu else 512
    qq, kk, vv = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
    o = FQ.AttnBlocchi2.apply(qq, kk, vv, scale)
    o.backward(do)
    out["attn3_v2"] = {n: float((a.double() - r).abs().max() / r.abs().max())
                       for n, a, r in (("o", o, R[0]), ("dq", qq.grad, R[1]), ("dk", kk.grad, R[2]), ("dv", vv.grad, R[3]))}
    print("attn3_v2", out["attn3_v2"], flush=True)
    for nome, split in (("attn3_v3", 3), ("attn1_v3", 1)):
        qq, kk, vv = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
        o = FQ.AttnBlocchi3.apply(qq, kk, vv, scale, split)
        o.backward(do)
        out[nome] = {n: float((a.double() - r).abs().max() / r.abs().max())
                     for n, a, r in (("o", o, R[0]), ("dq", qq.grad, R[1]), ("dk", kk.grad, R[2]), ("dv", vv.grad, R[3]))}
        print(nome, out[nome], flush=True)
    O = oggi(q, k, v, do, scale)
    out["oggi_sdpa_math_fp32"] = {n: float((a.double() - r).abs().max() / r.abs().max())
                                  for n, a, r in zip(("o", "dq", "dk", "dv"), O, R)}
    print("oggi", out["oggi_sdpa_math_fp32"], flush=True)
    assert all(x <= 1e-5 for x in out["flash3"].values()), out["flash3"]
    assert all(x <= 1e-5 for x in out["attn3"].values()), out["attn3"]
    assert all(x <= 1e-5 for x in out["attn3_v2"].values()), out["attn3_v2"]
    assert all(x <= 1e-5 for x in out["attn3_v3"].values()), out["attn3_v3"]
    if len(sys.argv) > 2:
        Path(sys.argv[-1]).write_text(json.dumps(out, indent=1))
    print("F1 F2 ok")


if __name__ == "__main__":
    main()
