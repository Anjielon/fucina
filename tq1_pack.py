"""TQ1_0 — GPU bit-packing and hot-first expert ordering.

TQ1_0 (llama.cpp): 256 ternary weights per block in 54 bytes — 52 of data
(base 3: five values per byte in the first 48, four per byte in the last 4)
plus a 2-byte f16 scale. 1.69 bits per weight.

The self-test at the bottom is not optional: it checks the packing against the
reference decoder. A well-known third-party fork silently truncated to 8 bits,
and only a round-trip test catches that class of bug.
"""
from __future__ import annotations
import numpy as np
import torch


def pack(d: np.ndarray, q: np.ndarray) -> np.ndarray:
    """d:(n,1) f16 scales, q:(n,256) in {-1,0,1} -> (n,54) uint8 TQ1_0.

    Chunked on the GPU: in numpy this cost 60-100 s for a 2.1B-weight tensor;
    in torch on GPU it is seconds (measured while forging a 397B model).
    """
    n_r = q.shape[0]
    out = np.empty((n_r, 54), np.uint8)
    passo = 8_000_000
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    p5 = torch.tensor([81, 27, 9, 3, 1], dtype=torch.int32, device=dev).view(1, 1, 5, 1)
    p4 = torch.tensor([81, 27, 9, 3], dtype=torch.int32, device=dev).view(1, 1, 4, 1)
    for i in range(0, n_r, passo):
        j = min(i + passo, n_r); m = j - i
        qs = torch.from_numpy(q[i:j].astype(np.int8)).to(dev, torch.int32) + 1
        a = (qs[:, :160].reshape(m, -1, 5, 32) * p5).sum(-2).reshape(m, -1)
        b = (qs[:, 160:240].reshape(m, -1, 5, 16) * p5).sum(-2).reshape(m, -1)
        c = (qs[:, 240:].reshape(m, -1, 4, 4) * p4).sum(-2).reshape(m, -1)
        t = torch.cat([a, b, c], -1)
        t = ((t * 256 + 242) // 243).to(torch.uint8)
        out[i:j, :52] = t.cpu().numpy()
        out[i:j, 52:] = d[i:j].astype(np.float16).view(np.uint8)
    return out


def hot_first_order(hot: list[int], n_experts: int) -> np.ndarray:
    """[hot experts] + [all others in order] — the ordering the two-plane engine
    assumes: the second plane covers the expert PREFIX 0..k-1.

    Hard-won: the engine assumed this ordering, the forge did not apply it, and
    the second-plane corrections landed on the wrong experts. Apply it when the
    file is WRITTEN, router rows included."""
    perm = np.array(list(hot) + [e for e in range(n_experts) if e not in set(hot)])
    assert len(perm) == n_experts and len(set(perm.tolist())) == n_experts, "invalid permutation"
    return perm


def reorder_experts(blocks: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """blocks (E*nb, 54) uint8 -> experts reordered according to perm."""
    n_experts = len(perm)
    v = blocks.reshape(n_experts, -1, 54)
    return np.ascontiguousarray(v[perm]).reshape(-1, 54)


def _self_test() -> None:
    """packing must match the reference decoder, on random values."""
    import sys
    sys.path.insert(0, "/home/angelo/build-llamacpp/gguf-py")
    from gguf import quants as GQ
    from gguf import GGMLQuantizationType as T
    rng = np.random.default_rng(7)
    q = rng.integers(-1, 2, size=(8, 256)).astype(np.int8)
    d = rng.random((8, 1)).astype(np.float32) * 0.1
    p = pack(d, q)
    back = GQ.dequantize(p, T.TQ1_0).reshape(8, 256)
    want = q.astype(np.float32) * d
    assert np.allclose(back, want, atol=1e-3), "pack() disagrees with the reference decoder"
    print("✓ tq1_pack self-test: packing matches the reference decoder")


if __name__ == "__main__":
    _self_test()
