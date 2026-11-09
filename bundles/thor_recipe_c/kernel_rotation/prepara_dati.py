"""
Prepara weights.bin/x.bin per il banco Vulkan, in DUE varianti (base / ruotata),
da un tensore VERO di Ornith-1.5 (lo stesso di correttezza_cpu.py), e salva
anche il riferimento numpy (y_true, y_hat_numpy) per il confronto con l'uscita
della GPU.
"""
import sys
import json
import numpy as np

sys.path.insert(0, "@@ROOT:home@@/build-llamacpp-tq1/gguf-py")
import gguf  # noqa: E402

from tq1_codec import quantize_block_tq1_0, dequantize_block_tq1_0  # noqa: E402
from fwht import fwht_shader_convention  # noqa: E402

GGUF_PATH = "@@ROOT:models@@/gguf/ornith-1.5/Ornith-1.5-35B-A3B-APEX-MTP-I-Quality.gguf"
TENSOR_NAME = "blk.10.attn_gate.weight"
SEED = 20260919


def carica_tensore(nome: str) -> np.ndarray:
    reader = gguf.GGUFReader(GGUF_PATH)
    for t in reader.tensors:
        if t.name == nome:
            print(f"[verifica] {t.name}  quant={t.tensor_type.name}  shape={t.shape}")
            w = gguf.quants.dequantize(t.data, t.tensor_type)
            return np.asarray(w, dtype=np.float32)
    raise SystemExit(f"tensore {nome} non trovato")


def encode_matrix(W: np.ndarray, ruota: bool) -> bytes:
    R, C = W.shape
    assert C % 256 == 0
    out = bytearray()
    for r in range(R):
        row = W[r]
        for b in range(0, C, 256):
            blk = row[b:b + 256]
            if ruota:
                blk = fwht_shader_convention(blk)
            out += quantize_block_tq1_0(blk)
    return bytes(out)


def numpy_reference_matvec(W: np.ndarray, x: np.ndarray, ruota: bool) -> np.ndarray:
    """Riproduce ESATTAMENTE quello che il kernel dovrebbe calcolare:
    dequant [+ derot se ruota] poi dot con x, per ogni riga."""
    R, C = W.shape
    y = np.zeros(R, dtype=np.float64)
    for r in range(R):
        row = W[r]
        acc = 0.0
        for b in range(0, C, 256):
            blk = row[b:b + 256]
            if ruota:
                blk_r = fwht_shader_convention(blk)
                enc = quantize_block_tq1_0(blk_r)
                dec = dequantize_block_tq1_0(enc)
                dec = fwht_shader_convention(dec)  # de-rotazione
            else:
                enc = quantize_block_tq1_0(blk)
                dec = dequantize_block_tq1_0(enc)
            acc += float(np.dot(dec.astype(np.float64), x[b:b + 256].astype(np.float64)))
        y[r] = acc
    return y


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=512, help="righe da usare (sottocampione)")
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    W_full = carica_tensore(TENSOR_NAME)
    Rfull, C = W_full.shape
    rng = np.random.default_rng(SEED)
    rows_idx = np.sort(rng.choice(Rfull, size=min(args.rows, Rfull), replace=False))
    W = W_full[rows_idx].copy()
    R = W.shape[0]
    blocks_per_row = C // 256
    x = rng.standard_normal(C).astype(np.float32)

    import os
    os.makedirs(args.out, exist_ok=True)

    for tag, ruota in (("base", False), ("rot", True)):
        wbytes = encode_matrix(W, ruota=ruota)
        with open(f"{args.out}/weights_{tag}.bin", "wb") as f:
            f.write(wbytes)
        y_ref = numpy_reference_matvec(W, x, ruota=ruota)
        np.save(f"{args.out}/y_ref_{tag}.npy", y_ref)
        print(f"{tag}: {len(wbytes)} byte pesi, y_ref shape={y_ref.shape}")

    with open(f"{args.out}/x.bin", "wb") as f:
        f.write(x.tobytes())

    meta = {"R": R, "C": C, "blocks_per_row": blocks_per_row, "tensor": TENSOR_NAME}
    with open(f"{args.out}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("meta:", meta)


if __name__ == "__main__":
    main()
