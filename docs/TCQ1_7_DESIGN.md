# TCQ1_7 — fractional-rate trellis quantization for the ternary class

*Design note, 2026-08-28 evening. Derivation + Viterbi simulations in
`experiments/tcq_sim.py`; published anchors: QTIP (arXiv 2406.11235v4),
GPTQ≡Babai (2507.18553), PTQTP (2509.16989).*

## Why: plain ternary is finished

Our per-256-block optimal-scale ternary measures **43.54%** relative error on
real expert tensors; the Lloyd bound for a fixed {−d,0,+d} alphabet on a
gaussian is **43.62%** — we are AT the scalar bound (and the match validates
the per-block gaussian model). No trit-level code (coset, reordering) can
improve it: with a fixed reproduction alphabet, independent nearest-level
assignment is already optimal per coordinate. Gains require **alphabet
expansion at constant rate** — exactly what a bit-window trellis provides.

Also closed today: the globally-optimal two-plane quantizer (exact 2-D grid
search) is only 3.4% MSE better than our alternating scheme — nothing left
in that family either.

## The quantizer: bitshift trellis, computed codebook, fractional rate

Per 256-weight block, one bit-stream. State at step j = last L=12 bits; step
j consumes k_j new bits:

    s_j = ((s_{j-1} << k_j) | c_j) & (2^L − 1),  c_j ∈ {0..2^{k_j}−1}
    ŵ_j = d · f(s_j)

f = QTIP's 1MAD computed code (a=34038481, b=76625530: LCG, sum the 4 bytes,
affine to ~N(0,1)) — **zero codebook**, 4-5 ALU ops. The 12-bit window gives
4096 reachable values per position at unchanged rate.

**Fractional rate via a time-variant schedule** (novel application): pattern
(2,2,1)×80 for steps 0..239, then (2)×16 → 432 bits = **54 payload bytes**,
1.6875 bit/weight — the same payload as TQ1_0. Closed-form offset (no scan):

    j<240:  o(j)=5⌊j/3⌋+(0,2,4)[j%3],  k_j=(2,2,1)[j%3]
    j≥240:  o(j)=400+2(j−240),          k_j=2

Tail-biting within the block (~0.2% MSE cost per QTIP Tab.2).
Encoding = exact Viterbi over 4096 states (the quadratic metric is additive
on the path — this *is* the exact intra-block CVP solver, subsuming any
Babai/act-order trick).

## Measured/simulated numbers (Viterbi, gaussian, L=12, T=256)

| payload rate | MSE | D_R=2^{−2R} | excess | rel err |
|---|---|---|---|---|
| 2.0 | 0.0714 | 0.0625 | 1.14× | 26.7% |
| **1.6875 (ours)** | **≈0.108** | 0.0964 | ~1.12× | **≈33%** |
| 1.5 | 0.1381 | 0.1250 | 1.10× | 37.2% |

(Validation: k=2 sim 0.0714 vs QTIP published 0.0733 tail-biting.)

Against our baselines:
- **Single plane** (cold experts): 43.5% → ~33% raw = **−42% MSE** for
  +0.0625 bpw. With the GPTQ wrapper unchanged, multiplicative transfer
  projects 28.13% (H-metric) → **~21.5%** [extrapolation, to be measured].
- **Hot experts**: uniform k=3 (3.0625 bpw) gives 14.1% raw vs our joint
  two-plane 17.26% at 3.38 bpw — better AND smaller by 0.32 bpw.
- Aggressive option: (2,1) schedule = 1.5625 bpw total, 37.2% — beats plain
  ternary while being SMALLER.

## Hessian integration (GPTQ stays)

Segmented Viterbi inside BlockLDLQ: diagonal-weighted intra-block metric
(c_j = Hchol[j,j]²), solve 64-step segments, push exact error onto remaining
columns with the existing Hchol rows, resume from the same trellis state;
inter-block propagation and FOEM hooks unchanged. Scale d: RMS init + one
least-squares refit + second Viterbi pass.

## Packing: GGML type "TCQ1_7"

    block = 256 weights = 56 bytes → 1.75 bpw exact
      [ 2B fp16 d ][ 54B tail-biting stream, schedule (2,2,1)×80+(2)×16 ]

## Vulkan decode sketch

Closed-form offset (~4 ALU) → funnel-shift 12-bit window from two u32 loads
→ 1MAD (1 MAD + 4-6 ALU byte-sum) → affine. ~12-15 ALU/weight vs ~4 for
TQ1_0; the matvec stays bandwidth-bound at 1.75 bpw (QTIP measures 2-bit
trellis decode faster than fp16 on GPU).

## Implementation plan (≈1 week)

1. **Forge** (`tcq_plane.py`, torch, 2-3 d): precomputed C[4096] table,
   vectorised Viterbi (rows × 4096 states, 2-bit backpointers, checkpoints),
   diag-H metric, 64-step segments + reused inter-block propagation.
   Validate on the SAME 397B tensor that measures 43.54/28.13 — expect ~33%.
2. **Format + CPU reference** (1 d): 56B pack, ggml scalar dequant,
   bit-exact roundtrip harness (the TQ1_B160 discipline).
3. **Vulkan kernel** (2-3 d): clone of the TQ1_0 shader with window
   extraction + 1MAD; t/s bench vs TQ1_0; then the k=3 hot-expert variant.

Target: ODINO v5 — same 84-86 GiB budget, roughly half the expert error.
