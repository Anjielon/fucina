# The Lever Registry — every technique, its measured effect, its source

> Methodology: no lever is applied on faith. Each carries (a) a **prior** —
> what its authors measured, (b) our **measurement on real weights** of
> Ornith-1.5-397B (bf16, true Hessians from 16,640 samples/layer), and (c) a
> **minutes-scale test protocol** so it can be re-judged on *any* new model
> before being applied. "Rejected" means *rejected for Ornith* — priors are
> model-dependent, which is exactly why the registry re-tests everything.
>
> Error metric below: relative output error ‖(Ŵ−W)x‖/‖Wx‖ under the true
> input distribution (Hessian-weighted), unless stated otherwise.

## 1. Scale (the largest single factor)

| lever | our measurement | verdict | source |
|---|---|---|---|
| **Optimal ternary scale** (least-squares vs abs-max) | abs-max zeroes ~87% of weights: **81% → 43.5%** error | ✅ mandatory | ours; cf. TWN (Li & Liu, arXiv 1605.04711) |
| **Scale from the compensated block** (inside the GPTQ loop) | 28.43 → 28.13 | ✅ free, small | ours |
| **Signed-scale grids** | better than abs-max on NF4 (authors') | 🧪 test per model | BOF4-S, arXiv 2505.06653 |

## 2. Compensation (error feedback)

| lever | our measurement | verdict | source |
|---|---|---|---|
| **Dedicated-per-plane GPTQ** | joint plane-1 used alone loses 10 points: **37.95% → 28.13%** when re-quantized dedicated. The co-adaptation trap. | ✅✅ the key insight | ours |
| **GPTQ** (full Hessian, cross-block propagation) | the backbone of every number above | ✅ | Frantar, Ashkboos, Hoefler, Alistarh — arXiv 2210.17323 |
| **Coordinate-descent refinement on ternary** | authors: −15% vs GPTQ; untested on ternary planes before us | 🧪 planned (v4) | QuantEase, arXiv 2309.01885 |
| **Asymmetric GPTAQ** | **worse** on ternary: 32.6 vs 28.1 symmetric | ⛔ rejected for Ornith | GPTAQ, arXiv 2504.02692 |
| **Cross-layer error propagation** (calibrate layer *l* on the *quantized* output of *l−1*) | authors: largest gains at low bits | 🧪 planned (v4) | QEP, arXiv 2504.09629 |
| **Babai (nearest-plane) ordering** | last-to-first GPTQ = Babai's algorithm, free | ✅ free | arXiv 2507.18553 |

## 3. Foldable transforms (zero runtime cost)

| lever | our measurement | verdict | source |
|---|---|---|---|
| **Free rotation optimized for the ternary objective** | **−35%** on-tensor (vs 2.4% for Kronecker/Hadamard) | 🧪 v4 candidate — folding requires QuaRot-style surgery | ours; cf. SpinQuant (Liu et al., arXiv 2405.16406), QuaRot (Ashkboos et al., arXiv 2404.00456) |
| **Rotation to Hessian eigenbasis** | **−14%** (it *gaussianizes* blocks — helps codebook methods like QuIP, hurts scalar-scale ternary) | ⛔ rejected | ours; cf. QuIP (Chee et al., arXiv 2307.13304) |
| **MagR outlier reduction** | 39% → 46% — MagR serves abs-max quantizers, not least-squares scales | ⛔ rejected (conditional: only with abs-max) | MagR, arXiv 2406.00800 |
| **Local Haar transform** | 0.1% — nothing | ⛔ rejected | HBLLM, arXiv 2512.00862 |
| **Similarity-reordered blocks (SSR)** | 0.6–0.7% on our blocks | ⛔ marginal | PT²-LLM |
| **Intra-expert permutation** (gate/up rows ↔ down columns, same P — exact, free) | never measured — the expert's inner dimension is private | 🧪 v4 candidate | cf. PermuQuant-class results |

## 4. Bit allocation (where to spend the budget)

| lever | our measurement | verdict | source |
|---|---|---|---|
| **Second plane for hot experts only** | 3% hottest: 37.4% → 32.0%; steep curve at the start; full 2-plane everywhere costs 18× the work for the tail | ✅ core design | ours; cf. DynaExq |
| **Frequency beats sensitivity** as the hot-set criterion | 32.6% vs 38.2% (correlation between the two rankings: +0.06). MoPEQ claims the opposite — **re-measure per model** | ✅ (re-test always) | ours vs MoPEQ |
| **down_proj above gate/up** | three independent sources agree | 🧪 v4 | DeepSeek-V3-class configs, Unsloth GGUF headers (read directly), MXMoE |
| **Protect first/last layers** | Hessian trace varies **35×** across layers; on Ornith the top-10 are exactly the last ten | ✅ | ours; cf. Unsloth dynamic quants |
| **Veto for rare-but-critical experts** (rarity ≠ expendability) | checkpoint-only | 🧪 v4 | arXiv 2604.06515 |
| **Linear attention at Q8** | inherited sub-2-bit GatedDeltaNet projections were 47–51% wrong; Q8 restore: model error **25.4% → 17.1%** for +3 GiB — best lever per byte (2.77 pts/GiB) | ✅✅ | ours; cf. Quamba (arXiv 2410.13229): no sub-4-bit SSM scan has ever been demonstrated |
| **Super-weight restore** | losing a single super weight collapses a model; coordinates published only for Llama/Mistral/OLMo/Phi — scan yours | 🧪 v3.2 | Yu et al., "The Super Weight in Large Language Models", arXiv 2411.07191 |

## 5. Repair (post-forge, ternary planes untouched)

| lever | prior | verdict | source |
|---|---|---|---|
| **QESC — router selection correction** | expert-selection bias is the *principal* factor in low-bit MoE degradation (authors) | 🧪 v3.2, TopK-MSE on logits | EAC-MoE, arXiv 2508.01625 |
| **Norm Tweaking** | GLM-130B at W2 near-fp quality; **Iters=1 is mandatory** (5 iterations = collapse) | 🧪 v3.2 | arXiv 2309.02784 |
| **QZO zeroth-order scale refinement** | teacher-free, 18× less memory | 🧪 optional | QZO, arXiv 2505.13430 |
| **EoRA low-rank error correction** | +6.4% quality at rank 4 (our measurement); hot-only cuts 80% of the byte cost | 🧪 v4 | EoRA, arXiv 2410.21271 |
| **ROMER hot-expert copies** | −58% ppl under weight noise (authors) | 🧪 test | ROMER-class, arXiv 2605.11800 |

## 6. Runtime (no file changes)

| lever | measurement | verdict | source |
|---|---|---|---|
| **Low-bit sampling recipe** | temp 0.6 · min_p 0.03 · top_p 0.9 · presence 1.5; thinking ON for hard reasoning; **never XTC**. At temp 0.2 a healthy sub-2-bit model degenerates into token loops — our test harness initially *failed a working model* because of this | ✅ | ours + Unsloth guidance + 3 papers (doc §23) |
| **MTP speculation** | +10% at draft-length 1 on the Ornith family (the MTP head predicts exactly one token; longer drafts *hurt*: acceptance 0.65 → 0.39) | ✅ per model | ours |
| **Self-draft from plane-1** (plane-1 as its own draft model) | expected routing overlap ρ ≈ 0.15–0.30 < 0.5 threshold → probably not viable; QSpec-class gains only at high ρ | 🧪 measure ρ first | doc §24-25 |

## The three bugs that shipped a perfect file and a broken model

Quantization research lives or dies on integration. Our v3.1 forge produced a
structurally perfect 88 GiB file — and a model that scored 0/4 on verifiable
questions. Three independent defects, none of them in the math:

1. **Prefix vs true IDs.** The forge stored second-plane slots under the
   experts' true indices; the engine summed them onto the expert *prefix*
   0..27. Right corrections, wrong experts.
2. **Orphaned plane pair.** The stored plane-2 came from a joint optimization
   whose plane-1 had been discarded in favour of the dedicated one — a couple
   that had never been optimized together.
3. **Double offset division.** The Vulkan TQ1_0 *mat-vec* kernel divided an
   offset that the caller already provides in block units — every generated
   token multiplied real weights of the *wrong expert*, while prompt
   processing (a different shader) was correct. The stock `test-backend-ops`
   matrix does not include TQ1_0, and its MUL_MAT_ID cases never hit the
   small-n vector path with non-zero offsets. 27/27 + 80/80 green after the
   one-line fix.

The seven engineering rules distilled from this day are in
[LESSONS.md](LESSONS.md); they are now *code* in this repository (boundary
asserts, in-forge verification, backend-op test requirements), not prose.

### Haar transform, re-tested properly (rejected with cause)

The original rejection ("0.1% — nothing") was reached by a flawed experiment:
wrong axis, no band grouping. Re-implemented correctly on the expert's private
axis, the verdict is unchanged but the reason is now understood:

| | layer 10 | layer 45 |
|---|---|---|
| baseline, one plane | 43.82% | 43.55% |
| Haar + bands, best level | 44.53% (**−1.63%**) | 43.48% (+0.14%) |
| baseline, two planes | 17.90% | 17.82% |
| Haar + bands, two planes | 18.62% (**−4.04%**) | 17.76% (+0.33%) |

The result oscillates around zero and changes sign between layers: noise, not
signal. Three structural reasons, all measured:

- **adjacent-weight correlation on the private axis: 0.0002–0.0005.** Haar is a
  local low-pass filter and assumes spatial regularity; these weights are white
  noise.
- **per-band energy is exactly proportional to band width** (0.2 / 0.4 / 0.8 /
  1.6 / 3.1 / 6.3 / 12.5 / 25 / 49.9%). The transform concentrates nothing —
  the premise of the technique does not hold here.
- Haar sums independent variables and therefore **gaussianizes** (kurtosis
  3.90 → 3.43), which hurts ternary quantization specifically, since it lives
  on heavy tails.

Two methodological notes worth keeping. On this model the bands on the private
axis are *already* pure at 256-wide blocks, so the grouping step is free —
there was nothing to gain there either. And dequantizing an already-ternary
GGUF to re-quantize it yields a fake ~0 error: measure against the original
weights, never against your own output.
