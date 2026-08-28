#!/usr/bin/env python3
"""QERA-approx low-rank correction, exported as a llama.cpp LoRA-GGUF.

The closed form (QERA, arXiv 2410.06040, ICLR 2025): the rank-k correction
that minimises the OUTPUT error of a quantized layer is

    C_k = SVD_k( (W - What) · diag(s) ) · diag(s)^-1 ,   s_i = sqrt(E[x_i^2])

i.e. weight the error's input columns by the activation RMS — the diagonal
of the same Hessian the GPTQ forge already measured — take the top-k SVD,
unscale. Served with `llama-server --lora out.gguf`: zero engine changes.

Scope here: the dense forge's ternary parts (gate/up). `down` is verbatim
from the donor (error zero, skipped). alpha is set equal to the rank so the
engine's alpha/rank scaling is exactly 1.

Usage:
  qera_lora.py --bf16 B.gguf --quant Q.gguf --hessians DIR \
               --out lora.gguf --rank 64 [--parts gate,up]
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, "/home/angelo/build-llamacpp-tq1/gguf-py")
from gguf import GGUFReader, GGUFWriter, quants as GQ

def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16", required=True)
    ap.add_argument("--quant", required=True)
    ap.add_argument("--hessians", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--parts", default="gate,up")
    A = ap.parse_args()
    parts = tuple(x.strip() for x in A.parts.split(","))

    rb = GGUFReader(A.bf16); rq = GGUFReader(A.quant)
    tb = {t.name: t for t in rb.tensors}; tq = {t.name: t for t in rq.tensors}
    arch = None
    for f in rq.fields.values():
        if f.name == "general.architecture":
            arch = str(f.contents())
    w = GGUFWriter(path=None, arch=arch)
    w.add_string("adapter.type", "lora")
    w.add_float32("adapter.lora.alpha", float(A.rank))

    todo = sorted(n for n in tq
                  if any(n.endswith(f"ffn_{p}.weight") for p in parts) and n in tb)
    assert todo, "nessun tensore da correggere"
    log(f"{len(todo)} tensori · rank {A.rank}")

    pairs = []
    for name in todo:
        L = int(name.split(".")[1])
        Wq = GQ.dequantize(np.asarray(tq[name].data), tq[name].tensor_type)
        Wf = GQ.dequantize(np.asarray(tb[name].data), tb[name].tensor_type).astype(np.float32)
        n_out, n_in = Wf.shape
        assert Wq.shape == Wf.shape, (name, Wq.shape, Wf.shape)
        H = np.load(Path(A.hessians) / f"H_{L:02d}.npy")
        assert H.shape[0] == n_in, (name, H.shape, n_in)
        s = np.sqrt(np.abs(np.diag(H)) + 1e-12).astype(np.float32)
        E = (Wf - Wq) * s[None, :]                    # scaled error (out, in)
        # top-k SVD; economy on the smaller side
        U, S, Vt = np.linalg.svd(E, full_matrices=False)
        k = min(A.rank, len(S))
        rootS = np.sqrt(S[:k])
        Bmat = (U[:, :k] * rootS[None, :])            # (out, k)  -> lora_b
        Amat = (rootS[:, None] * Vt[:k]) / s[None, :] # (k, in)   -> lora_a
        cover = float((S[:k] ** 2).sum() / max((S ** 2).sum(), 1e-30))
        log(f"  {name}: errore coperto dal rank-{k}: {cover*100:.1f}%")
        pairs.append((name, Amat.astype(np.float32), Bmat.astype(np.float32)))

    for name, Amat, Bmat in pairs:
        w.add_tensor_info(f"{name}.lora_a", list(Amat.shape), Amat.dtype, Amat.nbytes)
        w.add_tensor_info(f"{name}.lora_b", list(Bmat.shape), Bmat.dtype, Bmat.nbytes)
    w.open_output_file(Path(A.out))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
    for _, Amat, Bmat in pairs:
        w.write_tensor_data(Amat); w.write_tensor_data(Bmat)
    w.close()
    log(f"🏁 LoRA-GGUF scritto → {A.out}")


if __name__ == "__main__":
    main()
