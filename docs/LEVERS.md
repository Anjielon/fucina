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
| **Coordinate-descent refinement after GPTQ** | authors: layer error −12% median (up to −30%), OPT-350M 3-bit ppl 33.6 → 31.5. Their paper covers **uniform grids only**; the ternary extension is stated and never demonstrated | 🧪 v4 — closed form below | QuantEase, arXiv 2309.01885 |
| **Joint two-plane CD *under the true Hessian*** | PTQTP does joint two-plane with plain Frobenius loss; QuantEase does Hessian-weighted CD on one plane. **Nobody has published the combination** — this is the open slot our pipeline sits in | 🧪🔥 v4 — our distinctive contribution | see below |
| **Asymmetric calibration (GPTAQ/GPTQv2)** | our 32.6 vs 28.1 was a **mis-implementation**, retracted (see LESSONS). Authors at W2A16 on LLaMA2-7B: ppl **20.7 → 9.02** — our exact bit regime | 🧪🔥 **highest-priority v4 lever** | GPTAQ, arXiv 2504.02692 |
| **First-order compensation (FOEM)** | −17.3% ppl *over GPTAQ* at 3-bit; MMLU 53.8 → 56.1. Supersedes GPTAQ | 🧪 v4, after GPTAQ | FOEM, arXiv 2507.11017 (AAAI 2026) |
| **Cross-layer error propagation** (calibrate layer *l* on the *quantized* output of *l−1*) | authors: largest gains at low bits | 🧪 planned (v4) | QEP, arXiv 2504.09629 |
| **Babai (nearest-plane) ordering** | last-to-first GPTQ = Babai's algorithm, free | ✅ free | arXiv 2507.18553 |

### The exact open problem, stated honestly

We previously claimed that joint two-plane ternary optimization was unpublished.
**That was wrong and is retracted.** PTQTP (arXiv 2509.16989) does exactly that:
alternating ridge regression for the two scales, then an exhaustive 9-way search
over `(T¹,T²) ∈ {−1,0,+1}²` per weight, up to 50 iterations. What PTQTP does
*not* do is weight the objective by the calibration Hessian — its loss is plain
Frobenius on the weights, calibration-free.

Conversely, every Hessian-aware ternary method we found (QuantEase, ExTernD, and
GPTQ itself) is **sequential**: one plane, or greedy deflation.

So the precise open slot is:

```
min ‖(W − d₁T₁ − d₂T₂) X‖²_F        with H = XᵀX from real activations
     ↑ joint over both planes        ↑ Hessian-weighted, not Frobenius
```

The blueprint to fill it is a hybrid: QuantEase's per-weight closed form

```
β̃ = −[ Σ_{k≠j} H_{j,k}·Ŵ_{i,k} − (WH)_{i,j} ] / H_{j,j}
```

with the scalar snap `q(β̃)` replaced by PTQTP's joint 9-way enumeration over the
two planes. AQLM (arXiv 2401.06118) subsumes this in principle — two codebooks,
alphabet 3, group size 1 — and its beam search degenerates to exact enumeration
at that alphabet size; but no ternary configuration of AQLM has been published
or benchmarked.

One published warning worth carrying: ExTernD (arXiv 2607.13511) reports that
joint fitting of multiple ternary factors "fails completely" — but for a
*multiplicative* low-rank form, not our *additive* same-shape planes. It is not
evidence against this design; it is evidence that the distinction matters.

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
| **Second plane for hot experts only** | 3% hottest: 37.4% → 32.0% on weights. **End-to-end, measured**: perplexity 8.7204 → **8.3493** with the plane on `up` and `gate` (28 of 256 experts). The `down` plane is currently defective in our engine and must stay off — see below | ✅ core design, ✅ **now measured end-to-end** | ours; cf. DynaExq |
| **Frequency beats sensitivity** as the hot-set criterion | 32.6% vs 38.2% (correlation between the two rankings: +0.06). MoPEQ claims the opposite — **re-measure per model** | ✅ (re-test always) | ours vs MoPEQ |
| **down_proj above gate/up** | three independent sources agree | 🧪 v4 | DeepSeek-V3-class configs, Unsloth GGUF headers (read directly), MXMoE |
| **Protect first/last layers** | Hessian trace varies **35×** across layers; on Ornith the top-10 are exactly the last ten | ✅ | ours; cf. Unsloth dynamic quants |
| **Veto for rare-but-critical experts** (rarity ≠ expendability) | checkpoint-only | 🧪 v4 | arXiv 2604.06515 |
| **Attention QKV at Q8** | **not distinguishable**: paired sign test on 30 chunks, 17/30 wins, p = 0.58, mean gap 0.013 against a ±0.087 spread. Costs 0.78 GiB. Discarded | ⛔ rejected (measured, paired) | ours |
| **Linear attention at Q8** | inherited sub-2-bit GatedDeltaNet projections were 47–51% wrong; Q8 restore: model error **25.4% → 17.1%** for +3 GiB — best lever per byte (2.77 pts/GiB) | ✅✅ | ours; cf. Quamba (arXiv 2410.13229): no sub-4-bit SSM scan has ever been demonstrated |
| **Super-weight restore** | losing a single super weight collapses a model; coordinates published only for Llama/Mistral/OLMo/Phi — scan yours | 🧪 v3.2 | Yu et al., "The Super Weight in Large Language Models", arXiv 2411.07191 |

### The asymmetric-calibration family, stated precisely

Plain GPTQ matches the quantized layer against *its own* replayed input: it never
sees what the full-precision model actually produced there. Asymmetric
calibration changes the target to the full-precision output, so the layer also
absorbs the drift accumulated upstream:

```
GPTQ    min ‖(W+ΔW)X − W X‖²        X  = activations through the ALREADY-QUANTIZED prefix
GPTAQ   min ‖(W+ΔW)X − W X̃‖²        X̃ = activations through the FULL-PRECISION prefix
```

Three implementation facts, each of which we got wrong or did not know:

1. The Hessian stays `H = XXᵀ` from the **quantized** path. Only the *target*
   changes.
2. The extra statistic is the cross term `(X̃ − X)·Xᵀ`, cached once and folded
   in column by column — **not** `X̃Xᵀ`, and it must be routed through `W`
   (`R = W(X̃−X)`) rather than symmetrized.
3. `X` and `X̃` must be captured on the **same token sequences**. Independent
   calibration sets break the derivation, since `ΔX` is meant to be a per-token
   residual. This forces two paired forward passes and a sequential,
   layer-by-layer schedule.

Cost: <10% extra time below dim 4096, 30–40% above. Neither paper reports
ternary or MoE results — the transfer to our regime is unproven and is exactly
what the Tony testbench exists to measure.

### The second plane, measured end to end — and where it still fails

Weight-level error tells you a technique *should* work; only the model tells
you it *does*. Bisecting the three projections independently, on a 9.8 GiB
two-plane MoE, 12 chunks:

| configuration | perplexity | |
|---|---|---|
| second plane off | 8.7204 | reference |
| on `up` only | **8.4226** | −0.30 |
| on `gate` only | **8.5187** | −0.20 |
| on `up` + `gate` | **8.3493** | **−0.37** |
| on `down` only | 11.2405 | **+2.52** |
| all three | 10.9106 | +2.19 |

Two of the three projections deliver what the weights promised, and their
gains add. The third, alone, does more harm than the other two do good — while
its stored weights reconstruct the source *better* than the others' do
(44.5% → 18.45%). So the file is right and something in the engine's `down`
path is not.

The methodological point is the one worth carrying: we had verified the file
against the original checkpoint and verified the graph node by node, and both
verifications were correct — yet the whole was wrong. **Verifying the parts is
not verifying the composition.** What localised it was neither reading nor
reasoning but a five-line bisection with one environment variable per branch,
which is why those switches now ship in the engine rather than being added
when needed.

## 5. Repair (post-forge, ternary planes untouched)

| lever | prior | verdict | source |
|---|---|---|---|
| **EAQuant — expert calibration balance** | authors: +1.15–1.37% at W4A4, +1.33–2.28% at W3A4 on three MoE models. Three parts: smoothing aggregated across experts *and* router; router-logit KL alignment; **injecting tokens for under-served experts** (`count < r·kN/n`) | 🧪 v4 — the calibration-balance part is cheap and MoE-specific | EAQuant, arXiv 2506.13329 |
| **QESC — router selection correction** | expert-selection bias is the *principal* factor in low-bit MoE degradation (authors) | 🧪 v3.2, TopK-MSE on logits | EAC-MoE, arXiv 2508.01625 |
| **Norm Tweaking** | GLM-130B at W2 near-fp quality; **Iters=1 is mandatory** (5 iterations = collapse) | 🧪 v3.2 | arXiv 2309.02784 |
| **QZO zeroth-order scale refinement** | teacher-free, 18× less memory | 🧪 optional | QZO, arXiv 2505.13430 |
| **EoRA low-rank error correction** | +6.4% quality at rank 4 (our measurement); hot-only cuts 80% of the byte cost | 🧪 v4 | EoRA, arXiv 2410.21271 |
| **ROMER hot-expert copies** | −58% ppl under weight noise (authors) | 🧪 test | ROMER-class, arXiv 2605.11800 |

### What the QKV result tells us (a useful negative)

Promoting the 45 `attn_qkv` tensors to Q8_0 changed nothing measurable, while
the *linear-attention* projections on the same model gave 25.4% → 17.1%. The two
results together locate the remaining error: **standard attention is not the
bottleneck on this model; the state-space projections were, and the experts
still are.**

That is worth more than the 0.78 GiB it cost to learn. It also sets a floor on
what this benchmark can adjudicate: with σ ≈ 0.087 over 30 chunks, resolving a
0.013 effect at three sigma would need roughly 4,000 chunks. Levers of that size
are permanently below our measurement floor and should be judged on a different
instrument — or not attempted.

## 5b. Calibration data (chosen, not inherited)

| lever | measurement / prior | verdict | source |
|---|---|---|---|
| **Reasoning-trace self-calibration (AYOT)** | ternary Qwen3-4B: **+8.97 points** over conventional calibration on math/code, with only 4M tokens. Same regime as ours: 1.58-bit, reasoning model | 🧪🔥 v4 top-3 | AYOT / ScaleQ-1.58, arXiv 2608.01078 |
| **Expert-balanced sampling (MoEQuant EBSS)** | routing is power-law: a generic corpus leaves niche experts uncalibrated. Mixtral at 3-bit: **+4.72 points**; DeepSeek-MoE HumanEval +10 | 🧪🔥 v4 — mandatory at 512 experts | MoEQuant, arXiv 2505.03804 |
| **Calibration set size** | saturates at **64–128 sequences** across three independent studies; past that, nothing | ✅ settled — stop spending here | arXiv 2311.09755, 2510.10618, 2410.17170 |
| **Domain match** | math calibration → +5.92 on math; code → +7.49 on code. Generic-corpus *choice* among C4/Wikipedia/RedPajama is inconsistent — the lever is domain match, not corpus brand | ✅ | COLA, arXiv 2510.10618 |
| **Sequence-length diversity** | fixed-length chunks misestimate the Hessian because activation statistics depend on length; mix short and long | 🧪 cheap to try | MaCa, arXiv 2602.07465 |

Honest contradiction, recorded: COLA found self-generated calibration slightly
*worse* than a curated corpus for quantization (44.23 vs 43.61), while
Self-Calibration (arXiv 2410.17170) found it better. Both are at moderate bit
widths on general tasks. AYOT's large win is at **ternary, on reasoning tasks** —
so the reconciliation is that the self-calibration advantage grows as bits fall
and as the task depends on generation quality. We should treat it as promising,
not proven, for our regime, and measure it on Tony.

### Building the overthinking-token list without damaging prose

Two practical notes, both of which cost us a wrong list before we caught them:

**Include the space-prefixed forms.** In BPE vocabularies a word inside running
text is a *different token* from the same word at the start of a line — the
former carries a leading-space marker (`Ġ` in Qwen-family vocabularies).
Searching for the bare forms found 10 tokens; adding the prefixed forms found
**30**. A list without them penalizes almost nothing, because the model
practically always emits the prefixed variant.

**Split the list by how ordinary the word is.** The published marker set
includes `but` and `however`, which is sound for a math-reasoning benchmark and
harmful for general prose — and worse in languages where the equivalent is a
core connective (`ma` in Italian). We keep two sets:

| set | tokens | when |
|---|---|---|
| always safe | `wait`, `alternatively`, `hmm` + capitalized + space-prefixed (10) | any generation |
| thinking-block only | `but`, `however`, `actually`, `ma`, `tuttavia`, `invece`, … (20) | between `<think>` and `</think>`, where they do signal hesitation |

Start the bias around −1.5 to −2.0 and **measure**: this lever changes what the
model says, and a penalty strong enough to stop the loop is also strong enough
to break a sentence.

## 6. Runtime (no file changes)

| lever | measurement | verdict | source |
|---|---|---|---|
| **Low-bit sampling recipe** | temp 0.6 · min_p 0.03 · top_p 0.9 · presence 1.5. Note the ceiling: the loop-rescue authors measured a **90.2% loop rate at temp 0.6 and 94.6% greedy** — temperature alone does not fix severe collapse; thinking ON for hard reasoning; **never XTC**. At temp 0.2 a healthy sub-2-bit model degenerates into token loops — our test harness initially *failed a working model* because of this | ✅ | ours + Unsloth guidance + 3 papers (doc §23) |
| **MTP speculation** | +10% at draft-length 1 on the Ornith family (the MTP head predicts exactly one token; longer drafts *hurt*: acceptance 0.65 → 0.39) | ✅ per model | ours |
| **Loop detection + commit** | detector: any 20-gram repeated ≥4× within the last 1024 tokens. On trigger: if a parseable answer already appeared *before* the loop, emit it and stop. Authors' ablation: loop rescue alone is **+57 of the +59 total points** (17.2 → 74.2 on Qwen3-8B MATH-500) | ✅✅ **runtime, single model** — apply first | arXiv 2606.02011 |
| **Overthinking-token logit penalty** | quantized reasoning models over-emit "wait / but / alternatively" at high-KL positions; in up to **52% of failures the correct answer had already appeared** and was not committed to. Training-free logit penalty: −12–23% chain length, −58% overthinking errors, accuracy preserved, across 5 models × 3 quantizers. **Split the token list — see below** | ✅✅ **cheapest real fix** — a logit-bias list | arXiv 2606.00206 |
| **Forced `</think>` closure (reasoning budget)** | structurally removes budget-exhaustion and unclosed-reasoning failures: soft warning at a fraction of budget, then hard close | 🧪 llama.cpp PR #25961, unmerged — validate before trusting | — |
| **DRY sampler** | sequence-aware repetition penalty already in llama.cpp (`--dry-multiplier`, off by default). Community reports it fixes thinking loops where flat repetition-penalty *causes* them | 🧪 cheap knob, anecdotal evidence only | llama.cpp mainline |
| ~~FP16 planning hybrid~~ | requires a second **full-precision copy of the same model** — ~800 GiB for a 397B. Rejected on feasibility, not on merit; the authors are silent on its memory cost | ⛔ not portable to a single-checkpoint deployment | arXiv 2606.02011 |
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
3. **A destructive op read as if it were pure.** `ggml_clamp` returns a view
   into its input and writes through it; the mask branch read the same tensor
   and got post-clamp values, so the hot-expert mask was uniformly 1 and every
   cold expert received a correction that was not its own. Found by printing
   the tensors, not by reading the code.
4. **Double offset division.** The Vulkan TQ1_0 *mat-vec* kernel divided an
   offset that the caller already provides in block units — every generated
   token multiplied real weights of the *wrong expert*, while prompt
   processing (a different shader) was correct. The stock `test-backend-ops`
   matrix does not include TQ1_0, and its MUL_MAT_ID cases never hit the
   small-n vector path with non-zero offsets. 27/27 + 80/80 green after the
   one-line fix.

The engineering rules distilled from these days are in
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
