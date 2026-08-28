"""TCQ1_7 pre-test F1: exact Viterbi encoder on a REAL tensor (CPU-only, numpy).

Bitshift trellis, L=12 (4096 states), schedule (2,2,1)x80 + (2)x16 = 432 bits
= 54 payload bytes per 256-weight block + 2B fp16 scale (1.75 bpw exact).
Codebook: QTIP 1MAD computed code (a=34038481, b=76625530), empirically
standardized over the 4096 reachable states (fixed deterministic table).

Tail-biting: iterative constrained Viterbi (free pass -> constrain the 10
initial wrap bits to the previous pass's final-state low bits, up to
--tb-passes). Blocks that don't reach the fixed point are declared; the
DECODED rel_err (from the packed 54B+2B payload, closed-form offsets with
wraparound) is the honest end-to-end number and includes their cost.

Baseline: per-block EXACT optimal 3-level ternary (scan all support sizes,
d = mean of top-m |w|), i.e. the scalar bound RTN we measure at ~43.5%.

Usage:
  python3 tcq_encode_real.py --blocks 4096 [--tensor PATH.bin] [--seed 0]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

L = 12
S = 1 << L
MASK = S - 1
T = 256
KS = np.array([(2, 2, 1)[j % 3] for j in range(240)] + [2] * 16, dtype=np.int64)
OFFS = np.concatenate([np.zeros(1, np.int64), np.cumsum(KS)[:-1]])  # start bit of c_j
NBITS = int(KS.sum())  # 432
assert NBITS == 432 and NBITS // 8 == 54


def closed_form_offset(j):
    if j < 240:
        return 5 * (j // 3) + (0, 2, 4)[j % 3]
    return 400 + 2 * (j - 240)


def codebook_1mad():
    a, b = np.uint64(34038481), np.uint64(76625530)
    s = np.arange(S, dtype=np.uint64)
    x = (a * s + b) & np.uint64(0xFFFFFFFF)
    bs = ((x & np.uint64(0xFF)) + ((x >> np.uint64(8)) & np.uint64(0xFF))
          + ((x >> np.uint64(16)) & np.uint64(0xFF)) + ((x >> np.uint64(24)) & np.uint64(0xFF)))
    c = bs.astype(np.float64)
    c = (c - c.mean()) / c.std()
    return c.astype(np.float32)


def load_q8(bin_path):
    meta = json.loads(Path(str(bin_path).replace(".bin", ".meta")).read_text())
    assert meta["tipo"] == 8, "expected Q8_0"
    ne0, ne1 = meta["dims"]
    raw = np.fromfile(bin_path, dtype=np.uint8).reshape(-1, 34)
    scales = raw[:, :2].copy().view(np.float16).astype(np.float32)
    quants = raw[:, 2:].view(np.int8).astype(np.float32)
    w = (quants * scales).reshape(ne1, ne0)
    return w, (ne0, ne1)


def viterbi(w, d, C, tail=None):
    """Exact Viterbi. w: (B,T) float32, d: (B,) scale, C: (S,) codebook.
    tail: (B,) low-10-bit initial constraint (tail-biting) or None (free).
    Returns states (B,T) int32, sse (B,) float64 (at scale d)."""
    B = w.shape[0]
    dC = d[:, None].astype(np.float32) * C[None, :]  # (B,S)
    args = []
    # step 0
    err = (w[:, 0, None] - dC) ** 2
    if tail is None:
        V = err
    else:
        base_ok = (np.arange(S) >> 2)[None, :] == tail[:, None]
        V = np.where(base_ok, err, np.float32(np.inf))
    for t in range(1, T):
        k = int(KS[t])
        g = 1 << k
        Vr = V.reshape(B, g, S >> k)
        A = Vr.argmin(axis=1).astype(np.uint8)  # (B, S>>k) choice of top bits
        M = np.take_along_axis(Vr, A[:, None, :].astype(np.int64), axis=1)[:, 0, :]
        err = (w[:, t, None] - dC) ** 2
        V = np.repeat(M, g, axis=1) + err
        args.append(A)
    states = np.empty((B, T), dtype=np.int32)
    s = V.argmin(axis=1).astype(np.int32)
    sse = V[np.arange(B), s].astype(np.float64)
    states[:, T - 1] = s
    rows = np.arange(B)
    for t in range(T - 1, 0, -1):
        k = int(KS[t])
        base = states[:, t] >> k
        ctop = args[t - 1][rows, base].astype(np.int32)
        states[:, t - 1] = (ctop << (L - k)) | base
    return states, sse


def refit_scale(w, states, C):
    F = C[states]
    den = (F * F).sum(axis=1)
    num = (w * F).sum(axis=1)
    d = np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)
    return d.astype(np.float16).astype(np.float32)  # stored as fp16


def encode_blocks(w, C, tb_passes=3, log=print):
    """Full TCQ1_7 encode with tail-biting iteration + fp16 scale refit."""
    B = w.shape[0]
    t0 = time.time()
    d = w.std(axis=1).astype(np.float32)
    d = np.maximum(d, 1e-8)
    states, _ = viterbi(w, d, C, tail=None)  # pass 1: free boundary
    d = np.maximum(refit_scale(w, states, C), 1e-8)
    tails = (states[:, T - 1] & 0x3FF).astype(np.int32)
    todo = np.arange(B)
    npass = 1
    for it in range(tb_passes):
        npass += 1
        st, _ = viterbi(w[todo], d[todo], C, tail=tails[todo])
        states[todo] = st
        new_tail = (st[:, T - 1] & 0x3FF).astype(np.int32)
        ok = new_tail == tails[todo]
        tails[todo] = new_tail
        todo = todo[~ok]
        log(f"  tb pass {it + 1}: {B - todo.size}/{B} blocks at fixed point")
        if todo.size == 0:
            break
    d = refit_scale(w, states, C)  # final fp16 scale
    dt = time.time() - t0
    F = C[states]
    sse = ((w - d[:, None] * F) ** 2).sum(axis=1, dtype=np.float64)
    return states, d, sse, todo.size, dt, npass


# ── packing: [2B fp16 d][54B stream], MSB-first bits, c_j at bits o(j)..o(j)+k-1 ──

def pack(states, d):
    B = states.shape[0]
    bits = np.zeros((B, NBITS), dtype=np.uint8)
    for j in range(T):
        k, o = int(KS[j]), int(OFFS[j])
        c = states[:, j] & ((1 << k) - 1)
        for i in range(k):
            bits[:, o + i] = (c >> (k - 1 - i)) & 1
    payload = np.packbits(bits, axis=1)  # (B,54)
    dbytes = d.astype(np.float16).view(np.uint8).reshape(B, 2)
    return np.concatenate([dbytes, payload], axis=1)  # (B,56)


def decode(packed, C):
    B = packed.shape[0]
    d = packed[:, :2].copy().view(np.float16).astype(np.float32).ravel()
    bits = np.unpackbits(packed[:, 2:], axis=1)  # (B,432)
    idx = np.empty((T, L), dtype=np.int64)
    for j in range(T):
        end = int(OFFS[j]) + int(KS[j])
        idx[j] = [(end - L + i) % NBITS for i in range(L)]
    pw = (1 << np.arange(L - 1, -1, -1)).astype(np.int64)
    states = (bits[:, idx].astype(np.int64) * pw[None, None, :]).sum(axis=2)
    return d[:, None] * C[states], states


def rtn_ternary_optimal(w):
    """Exact optimal 3-level {-d,0,+d} per block: scan all support sizes."""
    a = np.sort(np.abs(w), axis=1)[:, ::-1].astype(np.float64)
    c = np.cumsum(a, axis=1)
    m = np.arange(1, w.shape[1] + 1, dtype=np.float64)
    gain = (c * c) / m
    best = gain.max(axis=1)
    sq = (w.astype(np.float64) ** 2).sum(axis=1)
    return sq - best  # per-block SSE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor", default="/mnt/models/gguf/odino-q8/blk.29.attn_gate.weight.bin")
    ap.add_argument("--blocks", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--tb-passes", type=int, default=3)
    ap.add_argument("--gauss-control", type=int, default=0,
                    help="also run N blocks with a seeded gaussian codebook")
    args = ap.parse_args()

    for j in range(T):  # closed-form offset must match cumulative schedule
        assert closed_form_offset(j) == int(OFFS[j]), j

    w2d, (ne0, ne1) = load_q8(args.tensor)
    total_blocks = w2d.size // T
    print(f"tensor {args.tensor} dims [{ne0},{ne1}] -> {total_blocks} blocks of {T}")
    wall = w2d.reshape(-1, T).astype(np.float32)

    rng = np.random.default_rng(args.seed)
    sel = np.sort(rng.choice(total_blocks, size=min(args.blocks, total_blocks), replace=False))
    w = wall[sel]
    B = w.shape[0]
    ref_sq_sub = float((w.astype(np.float64) ** 2).sum())

    # RTN ternary baseline: full tensor + subset
    t0 = time.time()
    sse_rtn_full = rtn_ternary_optimal(wall)
    t_rtn = time.time() - t0
    rel_rtn_full = np.sqrt(sse_rtn_full.sum() / (wall.astype(np.float64) ** 2).sum())
    rel_rtn_sub = np.sqrt(sse_rtn_full[sel].sum() / ref_sq_sub)
    print(f"RTN ternary (exact per-block optimal): rel_err full={rel_rtn_full:.4f} "
          f"subset={rel_rtn_sub:.4f}  ({t_rtn:.1f}s full tensor)")

    C = codebook_1mad()
    print(f"1MAD codebook: {np.unique(C).size} distinct values, "
          f"min={C.min():.2f} max={C.max():.2f}")

    # TCQ encode in batches
    states = np.empty((B, T), dtype=np.int32)
    dsc = np.empty(B, dtype=np.float32)
    sse = np.empty(B, dtype=np.float64)
    unconv = 0
    t0 = time.time()
    for i in range(0, B, args.batch):
        sl = slice(i, min(i + args.batch, B))
        st, dd, ss, nu, dt, npass = encode_blocks(w[sl], C, args.tb_passes)
        states[sl], dsc[sl], sse[sl] = st, dd, ss
        unconv += nu
        print(f"batch {i // args.batch}: {sl.stop - sl.start} blocks, {npass} viterbi passes, "
              f"{dt:.1f}s ({1000 * dt / (sl.stop - sl.start):.1f} ms/block), {nu} unconverged")
    t_enc = time.time() - t0
    rel_tcq = np.sqrt(sse.sum() / ref_sq_sub)
    print(f"\nTCQ1_7 encoder rel_err (Viterbi states + fp16 refit scale): {rel_tcq:.4f}")
    print(f"tail-biting unconverged: {unconv}/{B} ({100 * unconv / B:.2f}%)")
    print(f"encode time: {t_enc:.1f}s -> {1000 * t_enc / B:.2f} ms/block")

    # pack -> decode -> honest end-to-end error from the 56B payload
    packed = pack(states, dsc)
    assert packed.shape == (B, 56)
    recon, st_dec = decode(packed, C)
    mismatch_blocks = int((st_dec != states).any(axis=1).sum())
    sse_dec = ((w - recon) ** 2).sum(dtype=np.float64)
    rel_dec = np.sqrt(sse_dec / ref_sq_sub)
    print(f"payload: {packed.shape[1]}B/block = {8 * packed.shape[1] / T} bpw")
    print(f"decoded-from-payload rel_err: {rel_dec:.4f} "
          f"(state mismatch vs encoder in {mismatch_blocks}/{B} blocks)")

    # projection: this whole tensor + per-weight rate
    ms_blk = 1000 * t_enc / B
    print(f"\nprojection: this tensor whole ({total_blocks} blocks) ~ "
          f"{ms_blk * total_blocks / 1000 / 60:.1f} min; "
          f"per-GiB-of-bf16 ({2**30 / 2 / T:.0f} blocks) ~ "
          f"{ms_blk * (2**30 / 2 / T) / 1000 / 3600:.2f} h")

    if args.gauss_control:
        Cg = np.random.default_rng(0).standard_normal(S).astype(np.float32)
        n = min(args.gauss_control, B)
        st, dd, ss, nu, dt, npass = encode_blocks(w[:n], Cg, args.tb_passes)
        ref = float((w[:n].astype(np.float64) ** 2).sum())
        print(f"gaussian-codebook control ({n} blocks): rel_err {np.sqrt(ss.sum() / ref):.4f} "
              f"(same blocks, 1MAD gives "
              f"{np.sqrt(sse[:n].sum() / ref):.4f})")


if __name__ == "__main__":
    main()
