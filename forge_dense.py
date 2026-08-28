#!/usr/bin/env python3
"""FORGE DENSE — ternary FFN for a dense (non-MoE) model, from bf16.

The MoE forge quantizes stacked expert tensors and needs a router, an imatrix
of expert counts, a hot-first permutation and a partial second plane. A dense
model has none of that: three plain 2-D tensors per layer (`ffn_gate`,
`ffn_up`, `ffn_down`), used by every token. What carries over unchanged is the
part that makes the ternary good: the optimal-scale two-level search and true
GPTQ against the layer's real input Hessian.

Recipe (v1, deliberately conservative):
  - `blk.{0..N-1}.ffn_{gate,up,down}.weight` → TQ1_0, single dedicated plane,
    GPTQ with the layer Hessian when available.
  - everything else — attention, DeltaNet/ssm apparatus, norms, embeddings,
    the MTP block — is taken VERBATIM from a donor GGUF (a good shelf quant),
    because those tensors are known-sensitive (ssm_* are kept f32 even in
    Q4_K_XL shelf quants) and re-quantizing them buys nothing.
  - the MTP block (last blk index) keeps its donor FFN too: it never runs in
    the normal forward pass, so a Hessian for it does not exist.

Two-model heritage worth stating: on both MoE models we measured, a *partial*
second plane helps only in the final quarter of the depth. The dense v1 ships
with NO second plane — that experiment (does the 73% rule hold without
experts?) is run afterwards against this file as the reference.

    forge_dense.py --bf16 model-bf16.gguf --donor model-Q4_K_XL.gguf \
                   --output out.gguf --hessians DIR [--chunk 3000000]

No resume in v1 — a broken resume is worse than none. The `.journal` file is a
progress marker only; if the run dies, rerun it whole (~1-2 h at 27B).
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/angelo/build-llamacpp-tq1/gguf-py")
sys.path.insert(0, str(Path(__file__).parent))
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType as T  # noqa: E402
from gguf import quants as GQ  # noqa: E402
import ternary_gpu as TG  # noqa: E402
import two_planes as D  # noqa: E402

BLOCK = 256


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def pack_tq1(d: np.ndarray, q: np.ndarray, out_rows: int, n_in: int) -> np.ndarray:
    """(rows, nb, 1) scales + (rows, n_in) signs → raw TQ1_0 bytes."""
    W = (d.reshape(out_rows, -1, 1) * q.reshape(out_rows, -1, BLOCK)).reshape(out_rows, n_in)
    return GQ.quantize(W, T.TQ1_0)  # exact: values are already on the ternary grid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16", required=True, help="bf16 GGUF (FFN source)")
    ap.add_argument("--donor", required=True, help="shelf GGUF for every non-FFN tensor")
    ap.add_argument("--output", required=True)
    ap.add_argument("--hessians", default=None, help="dir of H_XX.npy per layer")
    ap.add_argument("--chunk", type=int, default=3_000_000)
    ap.add_argument("--ternary-parts", default="gate,up,down",
        help="which FFN parts get ternarized; the rest stay verbatim from the "
             "donor. down is the published bottleneck (D2Quant; super weights "
             "live in early down_proj) — 'gate,up' keeps it at donor precision")
    ap.add_argument("--foem-beta", type=float, default=0.0,
        help="FOEM first-order correction (arXiv 2507.11017); 0 = plain GPTQ")
    A = ap.parse_args()

    src = GGUFReader(A.bf16)
    don = GGUFReader(A.donor)
    src_t = {t.name: t for t in src.tensors}
    don_t = {t.name: t for t in don.tensors}
    arch = next(str(bytes(f.parts[-1]), "utf8")
                for f in don.fields.values() if f.name.endswith(".architecture"))

    # FFN layers that actually run in the forward pass: all but the MTP block.
    layers = sorted({int(n.split(".")[1]) for n in src_t
                     if re.match(r"blk\.\d+\.ffn_gate\.weight$", n)})
    mtp = max(int(n.split(".")[1]) for n in don_t if n.startswith("blk."))
    forge_layers = [L for L in layers if L != mtp]
    log(f"arch={arch} · dense FFN layers to forge: {len(forge_layers)} "
        f"(MTP block {mtp} stays donor)")

    def hchol(L: int):
        if not A.hessians:
            return None
        f = Path(A.hessians) / f"H_{L:02d}.npy"
        if not f.exists():
            return None
        return torch.from_numpy(
            D.prepare_hessian(np.load(f).astype(np.float64)).astype(np.float32))

    # ── plan: donor order, FFN of forge layers replaced by TQ1_0 ────────
    w = GGUFWriter(path=None, arch=arch)
    for f in don.fields.values():
        if f.name.startswith(("GGUF.", "split.")):
            continue
        try:
            w.add_key_value(f.name, f.contents(), f.types[0],
                            sub_type=f.types[-1] if len(f.types) > 1 else None)
        except Exception:
            pass

    parts = {x.strip() for x in A.ternary_parts.split(",") if x.strip()}
    assert parts <= {"gate", "up", "down"}, parts

    def is_forged(name: str) -> bool:
        m = re.match(r"blk\.(\d+)\.ffn_(gate|up|down)\.weight$", name)
        return bool(m) and m.group(2) in parts and int(m.group(1)) in forge_layers

    plan = []
    for name in sorted(don_t):
        if is_forged(name):
            shp = [int(x) for x in src_t[name].shape]     # (n_in, n_out) GGUF order
            fb = [shp[1], shp[0] // BLOCK * 54]           # numpy order, last dim in bytes
            w.add_tensor_info(name, fb, np.dtype(np.uint8), int(np.prod(fb)),
                              raw_dtype=T.TQ1_0)
            plan.append((name, "forge"))
        else:
            t = don_t[name]
            d = np.asarray(t.data)
            w.add_tensor_info(name, list(d.shape), d.dtype, int(d.nbytes),
                              raw_dtype=t.tensor_type)
            plan.append((name, "copy"))

    # ⛔ No resume in v1. A resume path that silently rewrites from scratch —
    # or worse, misaligns offsets — is exactly the class of bug that cost this
    # project a night. The 27B forge is ~1-2 h; if it dies, rerun it whole.
    journal = Path(A.output + ".journal")   # progress marker only, not resume
    w.open_output_file(Path(A.output))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()

    t_start = time.time()
    for i, (name, kind) in enumerate(plan):
        if kind == "copy":
            w.write_tensor_data(np.asarray(don_t[name].data))
        else:
            L = int(name.split(".")[1])
            t = src_t[name]
            shp = [int(x) for x in t.shape]               # (n_in, n_out)
            n_in, n_out = shp[0], shp[1]
            W = GQ.dequantize(np.asarray(t.data), t.tensor_type).reshape(n_out, n_in)
            # the Hessian is measured on the residual stream (hidden size):
            # it applies to gate/up, NOT to down, whose input is the
            # intermediate activation. Same rule as the MoE forge.
            H = hchol(L)
            if H is not None and H.shape[0] != n_in:
                H = None
            d1, q1 = TG.quantize_one_plane(
                torch.from_numpy(W.astype(np.float32)), H, chunk=A.chunk,
                foem_beta=A.foem_beta)
            raw = pack_tq1(np.asarray(d1), np.asarray(q1), n_out, n_in)
            w.write_tensor_data(raw)
            err = float(np.linalg.norm(
                (np.asarray(d1).reshape(n_out, -1, 1)
                 * np.asarray(q1).reshape(n_out, -1, BLOCK)).reshape(n_out, n_in) - W)
                / max(np.linalg.norm(W), 1e-12)) * 100
            log(f"  [{i+1}/{len(plan)}] {name}: err {err:.2f}%"
                + ("" if H is not None else "  (no Hessian)"))
        journal.write_text(f"{i} {name}")
        if i % 50 == 0:
            el = time.time() - t_start
            log(f"  {i}/{len(plan)} · {el/60:.0f} min")
    w.close()
    journal.unlink(missing_ok=True)
    log(f"done -> {A.output}")


if __name__ == "__main__":
    main()
