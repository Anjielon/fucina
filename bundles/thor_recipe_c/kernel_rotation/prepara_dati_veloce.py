"""
Generatore VELOCE (vettorizzato numpy) di blocchi TQ1_0 sintetici, per un
banco di scala piena (R=4096, C=2048, le dimensioni vere di blk.N.attn_gate
di Ornith-1.5). Serve SOLO per il banco di prestazioni (i valori non vengono
dal GGUF: la correttezza e il confronto coi pesi veri sono gia' coperti da
correttezza_cpu.py e prepara_dati.py su 512 righe vere). Qui verifichiamo che
il rapporto plain/derot misurato su piccola scala regga anche su scala piena,
dove il riuso di cache e' meno probabile.
"""
import argparse
import numpy as np
from fwht import fwht_shader_convention

POW3_5 = np.array([1, 3, 9, 27, 81], dtype=np.int64)


def quantize_blocks_vec(W: np.ndarray) -> bytes:
    """W: (N, 256) float32 -> N blocchi da 54 byte, vettorizzato su N.
    Stesso algoritmo bit-esatto di tq1_codec.quantize_block_tq1_0."""
    N = W.shape[0]
    amax = np.max(np.abs(W), axis=1)  # (N,)
    d = amax
    idd = np.where(d != 0, 1.0 / np.where(d == 0, 1, d), 0.0)

    def trit_group(vals):  # vals: (N, 32or16, 5)
        xi = np.round(vals * idd[:, None, None]).astype(np.int64) + 1
        return np.clip(xi, 0, 2)

    qs = np.zeros((N, 48), dtype=np.uint8)
    # primo gruppo: m=0..32, n=0..5 -> x[m + n*32]
    idxs = np.arange(32)[None, :, None] + np.arange(5)[None, None, :] * 32  # (1,32,5)
    idxs = np.broadcast_to(idxs, (N, 32, 5))
    vals = np.take_along_axis(W[:, None, :].repeat(32, axis=1), idxs, axis=2)  # (N,32,5)
    xi = trit_group(vals)  # (N,32,5), n-major digit order (n=0 -> weight 3^4)
    q = np.zeros((N, 32), dtype=np.int64)
    for n in range(5):
        q = q * 3 + xi[:, :, n]
    q = (q * 256 + 242) // 243
    qs[:, 0:32] = (q & 0xFF).astype(np.uint8)

    # secondo gruppo: m=0..16, n=0..5 -> x[160 + m + n*16]
    base2 = 160
    idxs2 = np.arange(16)[None, :, None] + np.arange(5)[None, None, :] * 16 + base2
    idxs2 = np.broadcast_to(idxs2, (N, 16, 5))
    vals2 = np.take_along_axis(W[:, None, :].repeat(16, axis=1), idxs2, axis=2)
    xi2 = trit_group(vals2)
    q2 = np.zeros((N, 16), dtype=np.int64)
    for n in range(5):
        q2 = q2 * 3 + xi2[:, :, n]
    q2 = (q2 * 256 + 242) // 243
    qs[:, 32:48] = (q2 & 0xFF).astype(np.uint8)

    # qh: j=0..4, m=0..4 -> x[240 + j + m*4]
    base3 = 240
    idxs3 = np.arange(4)[None, :, None] + np.arange(4)[None, None, :] * 4 + base3
    idxs3 = np.broadcast_to(idxs3, (N, 4, 4))
    vals3 = np.take_along_axis(W[:, None, :].repeat(4, axis=1), idxs3, axis=2)
    xi3 = trit_group(vals3)
    q3 = np.zeros((N, 4), dtype=np.int64)
    for m in range(4):
        q3 = q3 * 3 + xi3[:, :, m]
    q3 = q3 * 3
    q3 = (q3 * 256 + 242) // 243
    qh = (q3 & 0xFF).astype(np.uint8)

    d16 = d.astype(np.float16)
    out = np.zeros((N, 54), dtype=np.uint8)
    out[:, 0:48] = qs
    out[:, 48:52] = qh
    out[:, 52:54] = d16.view(np.uint8).reshape(N, 2)
    return out.tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=int, default=4096)
    ap.add_argument("--C", type=int, default=2048)
    ap.add_argument("--out", default="data_full")
    args = ap.parse_args()
    import os
    os.makedirs(args.out, exist_ok=True)

    rng = np.random.default_rng(7)
    blocks_per_row = args.C // 256
    N = args.R * blocks_per_row
    # pesi con code pesanti (simile ai densi veri: curtosi>0), non gaussiani puri
    W = (rng.standard_normal((N, 256)) * (0.02 + 0.05 * rng.random((N, 1)))).astype(np.float32)

    Wb = quantize_blocks_vec(W)
    with open(f"{args.out}/weights_base.bin", "wb") as f:
        f.write(Wb)

    Wrot = fwht_shader_convention(W)  # applica per blocco (ultima dim=256)
    Wr = quantize_blocks_vec(Wrot)
    with open(f"{args.out}/weights_rot.bin", "wb") as f:
        f.write(Wr)

    x = rng.standard_normal(args.C).astype(np.float32)
    with open(f"{args.out}/x.bin", "wb") as f:
        f.write(x.tobytes())

    print(f"R={args.R} C={args.C} blocks_per_row={blocks_per_row}  "
          f"weights_base={len(Wb)}B weights_rot={len(Wr)}B")


if __name__ == "__main__":
    main()
