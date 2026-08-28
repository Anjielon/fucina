"""Bitshift-trellis TCQ on i.i.d. Gaussian, exact Viterbi, torch.
State = last L bits of the stream. Step t consumes k_t bits, emits value
f(new_state) from a fixed pseudorandom Gaussian codebook (RPTC-style).
Validates against QTIP Table 2 (L=12, k=2, T=256 -> MSE 0.0733) then
measures fractional-rate patterns."""
import torch, math
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
L = 12; S = 1 << L; mask = S - 1
C = torch.randn(S, device=dev)              # node values ~ N(0,1)
T = 256; BATCH = 1024; NBATCH = 8

def viterbi(seqs, pattern):
    B = seqs.shape[0]
    INF = torch.tensor(float("inf"), device=dev)
    V = torch.zeros(B, S, device=dev)       # free initial state (tail bits stored)
    # backpointers: store predecessor choice per step
    bp = []
    for t in range(T):
        k = pattern[t % len(pattern)]
        # predecessors of s': top L-k bits of s' = bottom L-k bits of s
        # s = c_top * 2^(L-k) + (s' >> k)  for c_top in [0, 2^k)
        sprime = torch.arange(S, device=dev)
        base = sprime >> k                   # (S,)
        preds = (torch.arange(1 << k, device=dev)[:, None] << (L - k)) | base[None, :]  # (2^k, S)
        cand = V[:, preds]                   # (B, 2^k, S)
        Vmin, arg = cand.min(dim=1)          # (B,S)
        err = (seqs[:, t:t+1] - C[None, :]) ** 2   # (B,S)
        V = Vmin + err
        bp.append((preds, arg.to(torch.int16)))
    # terminal: best state, backtrack
    tot, s_end = V.min(dim=1)
    # backtrack to get reconstruction error already in tot
    return tot

results = {}
for name, pattern in [("k2", [2]), ("k5o3", [2, 2, 1]), ("k3o2", [2, 1]), ("k1", [1])]:
    se = 0.0; n = 0
    for b in range(NBATCH):
        torch.manual_seed(100 + b)
        seqs = torch.randn(BATCH, T, device=dev)
        tot = viterbi(seqs, pattern)
        se += tot.sum().item(); n += BATCH * T
    rate = sum(pattern) / len(pattern)
    mse = se / n
    dr = 2.0 ** (-2 * rate)
    results[name] = (rate, mse, dr)
    print(f"{name}: rate={rate:.4f} bpw  MSE={mse:.4f}  D_R={dr:.4f}  eccesso={mse/dr:.3f}x  rel_err={math.sqrt(mse):.4f}", flush=True)
