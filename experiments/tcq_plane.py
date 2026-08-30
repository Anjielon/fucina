"""TCQ1_7 forge encoder — VECTORISED exact Viterbi, device-agnostic (CPU/GPU).

Same quantizer as `tcq_encode_real.py` (the reference implementation), same
bits, same numbers — but restructured so that the whole trellis update is a
handful of large fused tensor ops instead of a pile of numpy temporaries:

  * one fused `min(dim=1)` per step instead of argmin + take_along_axis
    (one pass over V instead of two);
  * no `np.repeat` of the survivor metrics: the broadcast is folded into the
    add via the (B, S>>k, 2^k) view of the next V (s = low*2^k + c);
  * every buffer (V, V', err, d*C, survivors, backpointers) preallocated once
    per workspace and reused across tail-biting passes and batches — zero
    allocator churn, which is what makes it usable on GPU;
  * torch when available (any device: cpu / cuda / hip), numpy fallback with
    the same broadcast trick.

Bit-for-bit compatible with the reference: the arithmetic order inside a step
is identical (`(w_t - d*C)^2` in float32, survivor add, first-index argmin),
so the emitted trellis states are the SAME, not merely equivalent.

Trellis recap (see docs/TCQ1_7_DESIGN.md):
  L=12 -> 4096 states, T=256 weights/block,
  schedule (2,2,1)x80 + (2)x16 = 432 bits = 54 payload bytes,
  + 2B fp16 scale = 56 B/block = 1.75 bpw exact, tail-biting on 10 wrap bits.

Usage:
  python3 tcq_plane.py --verify --blocks 64          # vs tcq_encode_real.py
  python3 tcq_plane.py --bench --blocks 2048 --batch 512
  python3 tcq_plane.py --bench --device cuda --batch 4096
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

try:
    import torch

    _HAS_TORCH = True
except Exception:  # pragma: no cover - torch optional
    torch = None
    _HAS_TORCH = False

# ── trellis constants (must mirror tcq_encode_real.py exactly) ──────────────
L = 12
S = 1 << L
MASK = S - 1
T = 256
KS = np.array([(2, 2, 1)[j % 3] for j in range(240)] + [2] * 16, dtype=np.int64)
OFFS = np.concatenate([np.zeros(1, np.int64), np.cumsum(KS)[:-1]])
NBITS = int(KS.sum())
assert NBITS == 432 and NBITS % 8 == 0
PAYLOAD_B = NBITS // 8  # 54
BLOCK_B = PAYLOAD_B + 2  # 56 -> 1.75 bpw
K0 = int(KS[0])


def closed_form_offset(j: int) -> int:
    if j < 240:
        return 5 * (j // 3) + (0, 2, 4)[j % 3]
    return 400 + 2 * (j - 240)


def codebook_1mad() -> np.ndarray:
    """QTIP 1MAD computed code, standardised over the 4096 reachable states."""
    a, b = np.uint64(34038481), np.uint64(76625530)
    s = np.arange(S, dtype=np.uint64)
    x = (a * s + b) & np.uint64(0xFFFFFFFF)
    bs = ((x & np.uint64(0xFF)) + ((x >> np.uint64(8)) & np.uint64(0xFF))
          + ((x >> np.uint64(16)) & np.uint64(0xFF)) + ((x >> np.uint64(24)) & np.uint64(0xFF)))
    c = bs.astype(np.float64)
    c = (c - c.mean()) / c.std()
    return c.astype(np.float32)


# ── I/O: read only the blocks we need out of the Q8_0 dump (mmap, no copy) ──

def q8_total_blocks(bin_path) -> int:
    meta = json.loads(Path(str(bin_path).replace(".bin", ".meta")).read_text())
    assert meta["tipo"] == 8, "expected Q8_0"
    ne0, ne1 = meta["dims"]
    assert (ne0 * ne1) % T == 0
    return (ne0 * ne1) // T


def load_q8_blocks(bin_path, sel: np.ndarray) -> np.ndarray:
    """Dequantise ONLY the selected 256-weight blocks. Row-major flat order is
    identical to `(quants*scales).reshape(ne1, ne0).reshape(-1, 256)` because a
    TCQ block spans exactly 8 consecutive Q8_0 blocks (256 = 8 x 32)."""
    raw = np.memmap(bin_path, dtype=np.uint8, mode="r").reshape(-1, 34)
    sel = np.asarray(sel, dtype=np.int64)
    idx = (sel[:, None] * 8 + np.arange(8, dtype=np.int64)[None, :]).ravel()
    sub = np.ascontiguousarray(raw[idx])
    scales = sub[:, :2].copy().view(np.float16).astype(np.float32)
    quants = sub[:, 2:].view(np.int8).astype(np.float32)
    return (quants * scales).reshape(sel.size, T)


# ── backend abstraction ─────────────────────────────────────────────────────

class _NumpyWS:
    """Preallocated workspace, numpy backend."""

    xp = np
    is_torch = False

    def __init__(self, bmax: int, C: np.ndarray, device=None, dtype=None):
        self.bmax = bmax
        self.C = C
        self.dC = np.empty((bmax, S), np.float32)
        self.E = np.empty((bmax, S), np.float32)
        self.V = np.empty((bmax, S), np.float32)
        self.Vn = np.empty((bmax, S), np.float32)
        self.M = {k: np.empty((bmax, S >> k), np.float32) for k in (1, 2)}
        n = {k: int((KS[1:] == k).sum()) for k in (1, 2)}
        self.bp = {k: np.empty((n[k], bmax, S >> k), np.uint8) for k in (1, 2)}
        self.slot = _slot_map(n)
        self.arangeS = np.arange(S, dtype=np.int64)

    def to_dev(self, a):
        return np.ascontiguousarray(a)

    def to_host(self, a):
        return a


class _TorchWS:
    """Preallocated workspace, torch backend (device-agnostic)."""

    is_torch = True

    def __init__(self, bmax: int, C: np.ndarray, device="cpu", dtype=None):
        self.bmax = bmax
        self.device = torch.device(device)
        f32 = torch.float32
        self.C = torch.from_numpy(C).to(self.device)
        mk = lambda *sh, dt=f32: torch.empty(sh, dtype=dt, device=self.device)
        self.dC = mk(bmax, S)
        self.E = mk(bmax, S)
        self.V = mk(bmax, S)
        self.Vn = mk(bmax, S)
        self.M = {k: mk(bmax, S >> k) for k in (1, 2)}
        # survivor metrics broadcast to full state width: writing the 2^k-fold
        # repeat once and then doing a FLAT add beats letting torch broadcast
        # over the innermost axis (measured 123 us vs 178 us per step, B=512).
        self.Mrep = mk(bmax, S)
        # scratch for the survivor tournament (see _survivor_torch)
        self.t_m01 = mk(bmax, S >> 2)
        self.t_m23 = mk(bmax, S >> 2)
        self.t_s01 = mk(bmax, S >> 2, dt=torch.bool)
        self.t_s23 = mk(bmax, S >> 2, dt=torch.bool)
        n = {k: int((KS[1:] == k).sum()) for k in (1, 2)}
        # backpointers as BIT PLANES: k=2 -> (slot, plane, B, S>>2) with
        # plane0 = high decision bit, plane1 = low bit; k=1 -> single plane.
        # Writing the two comparisons straight into storage removes the
        # copy_/mul_/add_ trio that packing them into a uint8 would cost.
        self.bp = {2: mk(n[2], 2, bmax, S >> 2, dt=torch.bool),
                   1: mk(n[1], bmax, S >> 1, dt=torch.bool)}
        self.slot = _slot_map(n)

    def to_dev(self, a):
        return torch.from_numpy(np.ascontiguousarray(a)).to(self.device)

    def to_host(self, a):
        return a.detach().cpu().numpy()


def _survivor_torch(ws, V, B, k, Mv, slot):
    """min over the 2^k predecessor top-bit choices + first-index argmin,
    the decision written straight into the backpointer bit planes.

    Rationale (measured on this box, B=256): `torch.min(dim=1, out=(v,i))`
    costs 492 us/step and `torch.argmin` 26 ms — both fall off the vectorised
    path because they must materialise int64 indices. The same reduction
    expressed as a comparison tournament over contiguous (B, S>>k) slices
    costs 60-180 us and reproduces numpy's *first* minimum on ties exactly
    (all comparisons strict, so equal costs keep the lower index).

    The low decision bit is `select(hi, s23, s01)`, but `torch.where` on bool
    is catastrophically unvectorised on CPU (204 us/step at B=512, ~1.5 GB/s).
    The algebraic identity  lo = s01 XOR (hi AND (s01 XOR s23))  gives the
    SAME bits in three bitwise ops for 20.8 us — a 10x cut on what was the
    single most expensive op in the trellis step.
    """
    g = 1 << k
    Sk = S >> k
    Vg = V.view(B, g, Sk)
    if g == 2:
        a, b = Vg[:, 0], Vg[:, 1]
        torch.minimum(a, b, out=Mv)
        torch.lt(b, a, out=ws.bp[1][slot, :B])
        return
    a, b, c, d = Vg[:, 0], Vg[:, 1], Vg[:, 2], Vg[:, 3]
    m01, m23 = ws.t_m01[:B], ws.t_m23[:B]
    s01, s23 = ws.t_s01[:B], ws.t_s23[:B]
    hi = ws.bp[2][slot, 0, :B]
    lo = ws.bp[2][slot, 1, :B]
    torch.minimum(a, b, out=m01)
    torch.lt(b, a, out=s01)
    torch.minimum(c, d, out=m23)
    torch.lt(d, c, out=s23)
    torch.lt(m23, m01, out=hi)           # strict: ties keep the low group
    torch.minimum(m01, m23, out=Mv)
    torch.bitwise_xor(s01, s23, out=lo)  # lo = s01 ^ (hi & (s01 ^ s23))
    lo.bitwise_and_(hi)
    lo.bitwise_xor_(s01)


def _slot_map(n):
    """t -> (k, slot index inside bp[k]).  Steps 1..T-1 emit a backpointer."""
    cnt = {1: 0, 2: 0}
    out = {}
    for t in range(1, T):
        k = int(KS[t])
        out[t] = (k, cnt[k])
        cnt[k] += 1
    assert cnt == n
    return out


def make_workspace(bmax: int, C: np.ndarray, device: str = "cpu"):
    """device: 'cpu' (default) | 'cuda' (also HIP/ROCm) | 'numpy' | 'auto'.

    The default is deliberately NOT 'auto': this box shares one UMA pool
    between CPU and GPU, and silently grabbing the GPU would collide with
    whatever inference is running. The GPU must be asked for explicitly.
    """
    if device == "auto":
        device = "cuda" if (_HAS_TORCH and torch.cuda.is_available()) else "cpu"
    if device == "numpy" or not _HAS_TORCH:
        return _NumpyWS(bmax, C)
    return _TorchWS(bmax, C, device)


def default_threads() -> int:
    """Physical cores, not SMT siblings.

    Measured on the Corsair (16C/32T): 16 threads = 0.485 ms/block/pass,
    32 threads = 1.565 ms/block/pass. Torch's default (os.cpu_count() = 32)
    is 3.2x SLOWER than the right answer — hence this override.
    """
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:  # pragma: no cover
        n = os.cpu_count() or 1
    return max(1, n // 2)


# ── the vectorised exact Viterbi ────────────────────────────────────────────

def viterbi(ws, w, d, tail=None):
    """Exact Viterbi over the bitshift trellis, batched over blocks.

    w   : (B,T) float32 on ws device — the weights of B blocks
    d   : (B,)  float32 on ws device — per-block scale
    tail: (B,)  int64 or None — tail-biting constraint on the 10 wrap bits
          (state 0 must satisfy s0 >> K0 == tail)
    -> states (B,T) int64 on device, sse (B,) float64 on device.

    Work per step is O(B*S); the survivor broadcast is folded into the add.
    """
    B = int(w.shape[0])
    tch = ws.is_torch
    inf = float("inf")

    dC = ws.dC[:B]
    E = ws.E[:B]
    V = ws.V[:B]
    Vn = ws.Vn[:B]
    Mrep = ws.Mrep[:B] if tch else None

    if tch:
        torch.mul(d.unsqueeze(1), ws.C.unsqueeze(0), out=dC)
        torch.sub(w[:, 0:1], dC, out=E)
        E.pow_(2)
        if tail is None:
            V.copy_(E)
        else:
            V.fill_(inf)
            g0 = 1 << K0
            rows = torch.arange(B, device=ws.device)
            V.view(B, S >> K0, g0)[rows, tail] = E.view(B, S >> K0, g0)[rows, tail]
    else:
        np.multiply(d[:, None], ws.C[None, :], out=dC)
        np.subtract(w[:, 0:1], dC, out=E)
        np.square(E, out=E)
        if tail is None:
            V[...] = E
        else:
            V.fill(np.float32(inf))
            g0 = 1 << K0
            rows = np.arange(B)
            V.reshape(B, S >> K0, g0)[rows, tail] = E.reshape(B, S >> K0, g0)[rows, tail]

    for t in range(1, T):
        k = int(KS[t])
        g = 1 << k
        Sk = S >> k
        kk, slot = ws.slot[t]
        Mv = ws.M[k][:B]
        if tch:
            _survivor_torch(ws, V, B, k, Mv, slot)
            torch.sub(w[:, t:t + 1], dC, out=E)
            E.pow_(2)
            # same floats as `Mv.unsqueeze(2) + E.view(B,Sk,g)`, but the
            # 2^k-fold repeat is materialised once and the add stays flat.
            Mrep.view(B, Sk, g).copy_(Mv.unsqueeze(2))
            torch.add(Mrep, E, out=Vn)
        else:
            bp = ws.bp[k][slot, :B]
            Vr = V.reshape(B, g, Sk)
            A = Vr.argmin(axis=1)
            np.copyto(bp, A.astype(np.uint8))
            Mv[...] = np.take_along_axis(Vr, A[:, None, :], axis=1)[:, 0, :]
            np.subtract(w[:, t:t + 1], dC, out=E)
            np.square(E, out=E)
            np.add(Mv[:, :, None], E.reshape(B, Sk, g), out=Vn.reshape(B, Sk, g))
        V, Vn = Vn, V

    # traceback
    if tch:
        states = torch.empty((B, T), dtype=torch.int64, device=ws.device)
        s = V.argmin(dim=1)
        sse = V.gather(1, s.unsqueeze(1)).squeeze(1).to(torch.float64)
        states[:, T - 1] = s
        rows = torch.arange(B, device=ws.device)
        for t in range(T - 1, 0, -1):
            k = int(KS[t])
            _, slot = ws.slot[t]
            base = states[:, t] >> k
            if k == 2:
                hi = ws.bp[2][slot, 0, :B][rows, base].to(torch.int64)
                lo = ws.bp[2][slot, 1, :B][rows, base].to(torch.int64)
                ctop = (hi << 1) | lo
            else:
                ctop = ws.bp[1][slot, :B][rows, base].to(torch.int64)
            states[:, t - 1] = (ctop << (L - k)) | base
    else:
        states = np.empty((B, T), dtype=np.int64)
        s = V.argmin(axis=1)
        sse = V[np.arange(B), s].astype(np.float64)
        states[:, T - 1] = s
        rows = np.arange(B)
        for t in range(T - 1, 0, -1):
            k = int(KS[t])
            _, slot = ws.slot[t]
            base = states[:, t] >> k
            ctop = ws.bp[k][slot, :B][rows, base].astype(np.int64)
            states[:, t - 1] = (ctop << (L - k)) | base

    # V/Vn were swapped an odd/even number of times; nothing to restore since
    # both belong to the workspace and are fully overwritten on entry.
    return states, sse


def refit_scale(ws, w, states):
    """Least-squares scale, stored (and returned) as fp16 — matches reference."""
    if ws.is_torch:
        F = ws.C[states]
        den = (F * F).sum(dim=1)
        num = (w * F).sum(dim=1)
        d = torch.where(den > 0, num / den.clamp_min(1e-12),
                        torch.zeros((), dtype=torch.float32, device=ws.device))
        return d.to(torch.float16).to(torch.float32)
    F = ws.C[states]
    den = (F * F).sum(axis=1)
    num = (w * F).sum(axis=1)
    d = np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)
    return d.astype(np.float16).astype(np.float32)


def encode_blocks(ws, w, tb_passes: int = 3):
    """Full TCQ1_7 encode of one batch: free pass -> tail-biting fixed point
    -> final fp16 scale. Mirrors tcq_encode_real.encode_blocks step by step.

    w: (B,T) on the ws device. Returns states (B,T), d (B,), sse (B,) f64,
    n_unconverged, n_passes."""
    tch = ws.is_torch
    B = int(w.shape[0])

    if tch:
        d = torch.std(w, dim=1, correction=0).clamp_min(1e-8)
    else:
        d = np.maximum(w.std(axis=1).astype(np.float32), 1e-8)

    states, _ = viterbi(ws, w, d, tail=None)
    d = refit_scale(ws, w, states)
    d = d.clamp_min(1e-8) if tch else np.maximum(d, 1e-8)
    tails = states[:, T - 1] & 0x3FF

    if tch:
        todo = torch.arange(B, device=ws.device)
    else:
        todo = np.arange(B)
    npass = 1
    for _ in range(tb_passes):
        npass += 1
        st, _ = viterbi(ws, w[todo], d[todo], tail=tails[todo])
        states[todo] = st
        new_tail = st[:, T - 1] & 0x3FF
        ok = new_tail == tails[todo]
        tails[todo] = new_tail
        todo = todo[~ok]
        if int(todo.shape[0]) == 0:
            break

    d = refit_scale(ws, w, states)
    F = ws.C[states]
    if tch:
        sse = ((w - d.unsqueeze(1) * F) ** 2).sum(dim=1, dtype=torch.float64)
    else:
        sse = ((w - d[:, None] * F) ** 2).sum(axis=1, dtype=np.float64)
    return states, d, sse, int(todo.shape[0]), npass


# ── packing / decoding (vectorised, byte-identical to the reference) ────────

_J2 = np.nonzero(KS == 2)[0]
_J1 = np.nonzero(KS == 1)[0]
_O2 = OFFS[_J2]
_O1 = OFFS[_J1]


def pack(states: np.ndarray, d: np.ndarray) -> np.ndarray:
    """(B,T) int states + (B,) scale -> (B,56) uint8 payload."""
    B = states.shape[0]
    st = np.asarray(states, dtype=np.int64)
    bits = np.zeros((B, NBITS), dtype=np.uint8)
    c2 = st[:, _J2] & 3
    bits[:, _O2] = (c2 >> 1).astype(np.uint8)
    bits[:, _O2 + 1] = (c2 & 1).astype(np.uint8)
    bits[:, _O1] = (st[:, _J1] & 1).astype(np.uint8)
    payload = np.packbits(bits, axis=1)
    dbytes = np.asarray(d, dtype=np.float16).view(np.uint8).reshape(B, 2)
    return np.concatenate([dbytes, payload], axis=1)


_DEC_IDX = np.empty((T, L), dtype=np.int64)
for _j in range(T):
    _end = int(OFFS[_j]) + int(KS[_j])
    _DEC_IDX[_j] = [(_end - L + _i) % NBITS for _i in range(L)]
_DEC_PW = (1 << np.arange(L - 1, -1, -1)).astype(np.int32)


def decode(packed: np.ndarray, C: np.ndarray):
    """(B,56) uint8 -> (recon (B,256) float32, states (B,256) int64)."""
    B = packed.shape[0]
    d = packed[:, :2].copy().view(np.float16).astype(np.float32).ravel()
    bits = np.unpackbits(packed[:, 2:], axis=1)
    states = (bits[:, _DEC_IDX].astype(np.int32) * _DEC_PW[None, None, :]).sum(axis=2)
    states = states.astype(np.int64)
    return d[:, None] * C[states], states


# ── driver: encode an arbitrary flat weight array ───────────────────────────

def encode_all(wflat: np.ndarray, batch: int = 512, device: str = "auto",
               tb_passes: int = 3, C: np.ndarray = None, log=None):
    """wflat: (N,256) float32. Returns (packed (N,56) uint8, sse (N,) f64,
    n_unconverged, encode_seconds)."""
    C = codebook_1mad() if C is None else C
    N = wflat.shape[0]
    ws = make_workspace(min(batch, N), C, device)
    packed = np.empty((N, BLOCK_B), np.uint8)
    sse = np.empty(N, np.float64)
    unconv = 0
    t0 = time.time()
    for i in range(0, N, batch):
        j = min(i + batch, N)
        wb = ws.to_dev(wflat[i:j])
        st, dd, ss, nu, npass = encode_blocks(ws, wb, tb_passes)
        packed[i:j] = pack(ws.to_host(st), ws.to_host(dd))
        sse[i:j] = ws.to_host(ss)
        unconv += nu
        if log:
            log(f"  batch {i // batch}: {j - i} blocks, {npass} passes, {nu} unconverged")
    if _HAS_TORCH and getattr(ws, "is_torch", False) and ws.device.type != "cpu":
        torch.cuda.synchronize()
    return packed, sse, unconv, time.time() - t0


# ── CLI ─────────────────────────────────────────────────────────────────────

def _time(fn):
    t = time.time()
    fn()
    return time.time() - t


def _select(total, n, seed):
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total, size=min(n, total), replace=False))


def cmd_verify(args):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import tcq_encode_real as ref

    total = q8_total_blocks(args.tensor)
    sel = _select(total, args.blocks, args.seed)
    w = load_q8_blocks(args.tensor, sel)
    B = w.shape[0]
    ref_sq = float((w.astype(np.float64) ** 2).sum())
    C = codebook_1mad()
    assert np.array_equal(C, ref.codebook_1mad())
    for j in range(T):
        assert closed_form_offset(j) == int(OFFS[j])

    print(f"verify: {B} blocks from {args.tensor} (seed {args.seed}), "
          f"{total} blocks total in tensor")

    t0 = time.time()
    r_st, r_d, r_sse, r_un, _, r_np = ref.encode_blocks(w, C, args.tb_passes, log=lambda *a: None)
    t_ref = time.time() - t0
    rel_ref = float(np.sqrt(r_sse.sum() / ref_sq))

    ws = make_workspace(B, C, args.device)
    dev = "numpy" if not ws.is_torch else str(ws.device)
    wd = ws.to_dev(w)
    encode_blocks(ws, wd[:min(B, 16)], args.tb_passes)  # warm: threads + page faults
    t0 = time.time()
    st, d, sse, un, npass = encode_blocks(ws, wd, args.tb_passes)
    t_new = time.time() - t0
    st_h, d_h, sse_h = ws.to_host(st), ws.to_host(d), ws.to_host(sse)
    rel_new = float(np.sqrt(sse_h.sum() / ref_sq))

    same_states = int((st_h.astype(np.int64) != r_st.astype(np.int64)).any(axis=1).sum())
    dmax = float(np.abs(d_h - r_d).max())
    print(f"  reference  rel_err = {rel_ref:.10f}   ({t_ref:.2f}s, "
          f"{1000 * t_ref / B:.2f} ms/block, {r_un} unconverged)")
    print(f"  vectorised rel_err = {rel_new:.10f}   ({t_new:.2f}s, "
          f"{1000 * t_new / B:.2f} ms/block, {un} unconverged, backend {dev})")
    print(f"  |delta rel_err| = {abs(rel_new - rel_ref):.3e}")
    print(f"  blocks with differing states: {same_states}/{B}; max |dscale| = {dmax:.3e}")

    # payload identity
    p_ref = ref.pack(r_st, r_d)
    p_new = pack(st_h, d_h)
    print(f"  packed payload identical: {bool(np.array_equal(p_ref, p_new))}")
    rec_r, sdec_r = ref.decode(p_ref, C)
    rec_n, sdec_n = decode(p_new, C)
    rel_dec_r = float(np.sqrt(((w - rec_r) ** 2).sum(dtype=np.float64) / ref_sq))
    rel_dec_n = float(np.sqrt(((w - rec_n) ** 2).sum(dtype=np.float64) / ref_sq))
    print(f"  decoded-from-payload rel_err: ref {rel_dec_r:.10f}  vec {rel_dec_n:.10f}"
          f"  (|delta| {abs(rel_dec_n - rel_dec_r):.3e})")

    ok = abs(rel_new - rel_ref) <= args.tol
    print(f"\nVERDICT: {'PASS' if ok else 'FAIL'} (tolerance {args.tol:g}) — "
          f"speedup {t_ref / max(t_new, 1e-9):.1f}x on {dev}")
    return 0 if ok else 1


def cmd_bench(args):
    total = q8_total_blocks(args.tensor)
    sel = _select(total, args.blocks, args.seed)
    w = load_q8_blocks(args.tensor, sel)
    C = codebook_1mad()
    N = w.shape[0]
    print(f"bench: {N} blocks, batch {args.batch}, device {args.device}, "
          f"threads {torch.get_num_threads() if _HAS_TORCH else 'n/a'}")

    # warm-up on one batch (allocator + threads) + single-pass micro timing
    ws = make_workspace(min(args.batch, N), C, args.device)
    nwarm = min(args.batch, N)
    ww = ws.to_dev(w[:nwarm])
    dd = (torch.std(ww, dim=1, correction=0).clamp_min(1e-8) if ws.is_torch
          else np.maximum(ww.std(axis=1).astype(np.float32), 1e-8))
    viterbi(ws, ww, dd)
    tp = min(_time(lambda: viterbi(ws, ww, dd)) for _ in range(3))
    print(f"  one Viterbi pass: {1000 * tp / nwarm:.3f} ms/block "
          f"({1e6 * tp / nwarm / (T - 1):.2f} us/block/step)")
    del ws, ww, dd

    dt = float("inf")
    for _ in range(max(1, args.repeat)):
        packed, sse, unconv, dti = encode_all(w, args.batch, args.device, args.tb_passes, C)
        dt = min(dt, dti)
    ref_sq = float((w.astype(np.float64) ** 2).sum())
    rel = float(np.sqrt(sse.sum() / ref_sq))
    recon, _ = decode(packed, C)
    rel_dec = float(np.sqrt(((w - recon) ** 2).sum(dtype=np.float64) / ref_sq))
    msb = 1000 * dt / N
    print(f"  rel_err {rel:.4f} (decoded {rel_dec:.4f}), unconverged {unconv}/{N}")
    print(f"  encode {dt:.2f}s -> {msb:.3f} ms/block, {N / dt:,.0f} blocks/s, "
          f"{N * T / dt / 1e3:,.0f} kweights/s")
    nb = args.project_blocks
    print(f"  projection {nb / 1e9:.2f}G blocks ({nb * T / 1e9:.0f}G weights): "
          f"{msb * nb / 1000 / 3600:,.1f} process-hours "
          f"({msb * nb / 1000 / 86400:,.1f} process-days)")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensor", default="/mnt/models/gguf/odino-q8/blk.29.attn_gate.weight.bin")
    ap.add_argument("--blocks", type=int, default=64)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tb-passes", type=int, default=3)
    ap.add_argument("--device", default="cpu", help="cpu (default) | cuda | numpy | auto")
    ap.add_argument("--threads", type=int, default=0,
                    help="torch CPU threads (0 = physical cores; SMT hurts, see default_threads)")
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--project-blocks", type=float, default=1.5e9)
    ap.add_argument("--repeat", type=int, default=1, help="bench repeats, best wins")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--bench", action="store_true")
    args = ap.parse_args()
    if _HAS_TORCH:
        torch.set_num_threads(args.threads if args.threads else default_threads())
    if not (args.verify or args.bench):
        args.verify = True
    rc = 0
    if args.verify:
        rc |= cmd_verify(args)
    if args.bench:
        rc |= cmd_bench(args)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
