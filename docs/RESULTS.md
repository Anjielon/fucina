# Measurements

Every number this project rests on, with the conditions it was taken under.
Figures are only comparable within a block — mixing recipes is the easiest way
to fool yourself, and we did it once.

## Models

| | 397B | 35B |
|---|---|---|
| architecture | 512 experts × 60 layers, hybrid GatedDeltaNet attention | 256 experts × 41 layers |
| source | bf16 checkpoint (740 GiB), non-expert tensors from an existing quantization | an existing high-quality GGUF |
| output | 88.16 GiB with the second plane, **84.00 GiB** without | 9.19 GiB |
| second plane | 28 hot experts, all 60 layers | 28 hot experts, all 41 layers |
| Hessians at forge time | yes (16,640 samples/layer, mean anisotropy α = 1.17) | no |

## Perplexity — 30 chunks of wikitext-2, ctx 2048

| configuration | 397B | |
|---|---|---|
| one plane | **6.1845 ± 0.08705** | shipped build |
| one plane, second-plane tensors stripped | **6.1845 ± 0.08705** | identical, −4.16 GiB |
| `up` + `gate` second plane | 6.3509 ± 0.09050 | worse |
| all three projections | 7.1871 ± 0.10660 | worse |

## Perplexity — 12 chunks, 35B model

| configuration | perplexity | |
|---|---|---|
| second plane off | 8.7204 | reference |
| on `up` only | **8.4226** | −0.30 |
| on `gate` only | **8.5187** | −0.20 |
| on `up` + `gate` | **8.3493** | **−0.37, best** |
| on `down` only | 11.2405 | +2.52 |
| all three | 10.9106 | +2.19 |
| all three + cold-expert scale compensation ×1.093 | 11.6068 | +2.89 |
| `down` only + the same compensation | 11.1027 | −0.14 against `down` alone |

The compensation helps in one configuration and harms in another, which is why
the scale-mismatch explanation was abandoned.

## Reconstruction fidelity, against the original checkpoint

397B, layer 59, expert in file position 0 (verified to be the imatrix's hottest,
so the hot-first permutation is confirmed):

| projection | plane 1 | plane 1 + 2 |
|---|---|---|
| gate | 46.21% | **22.32%** |
| up | 46.20% | **22.31%** |
| down | 45.30% | **19.25%** |

35B, layer 20, both positions checked, all three projections: 44.5% → 18.5%.
Cold experts (positions 30, 100, 200): 44.5%, plane 1 only.

Output error under **real captured activations**, not weights: the same
improvement, 44.9% → 18.6% on `up`, 44.5% → 18.5% on `down`, every layer
sampled (0, 1, 2, 5, 10, 20, 30, 40) within ±0.3.

Weighted sum of the eight routed experts, which is what the layer actually
emits: 44.6% → 42.2% on the 35B (9.5% scale mismatch) and 51.4% → 48.9% on the
397B (16% mismatch). Better in both.

**So fidelity improves at every level we can measure, and the model gets
worse.** See [LESSONS](LESSONS.md).

## Expert routing

The measurement that eight fidelity metrics could not make, because all of them
compared the *same* experts in both configurations. This one asks **which**
experts get chosen.

Captured from `ffn_moe_probs` — the router's own output, F32 and present in
both configurations — with the top-k recomputed offline. 1,883 tokens, 35B
model, 40 layers carrying a second plane.

| | tokens whose expert *set* changes |
|---|---|
| layer 0 | **0.0%** |
| layer 1 | 28.1% |
| layer 5 | 65.9% |
| layer 10 | 70.8% |
| layer 20 | 92.2% |
| layer 30 | 91.8% |
| **all layers** | **78.67%** |
| first third / last third | 57.90% / 89.67% |

Layer 0 at exactly 0.0% is the control: its router reads the embedding, which
no correction touches. The number is what it must be, so the capture is sound.

The boundary is thin enough to explain the sensitivity. The median probability
gap between the 8th and 9th ranked expert is **0.0003**, against a mean
per-expert probability of 1/256 = 0.0039 — the top-k cutoff sits in a gap about
ten times narrower than an average expert's share.

### Does the drift move toward the original, or away?

Both configurations compared against the **source checkpoint** the model was
forged from, layer by layer, as the fraction of the 8 selected experts held in
common.

| | agreement with source |
|---|---|
| one plane | **74.36%** |
| one plane + second | **74.33%** |
| chance (8 of 256) | 3.12% |

Layer 0 gives 100.00%, again as it must. Agreement decays with depth — 91.2% at
layer 1, 65.6% at layer 20 — and the second plane does not consistently move it
in either direction: it is ahead at layers 10, 20 and 30, behind at 1, 2, 5 and
39.

**So the correction reshuffles 78.67% of expert sets and buys back nothing.**
It pays the full price of routing perturbation for no gain in routing fidelity.
That is the first account consistent with every earlier measurement: weight
error halves, single-expert output error halves, and the model still degrades,
because what the model needs is not accurate experts but a routing pattern its
remaining components agree with.

⚠️ This required aligning expert indices first. The forge reorders experts
hot-first, so index *i* means different things in the two files; comparing them
raw gives 4.4% agreement — indistinguishable from the 3.12% chance floor — and
reads convincingly as "the routes are unrelated". The permutation is recovered
from the data rather than from the forge's source: the router's own rows are
permuted alongside the experts, so matching rows between the two files by
correlation recovers it exactly (median match **1.0000**, bijective on all 41
layers). The script refuses to print any routing number if that match degrades.

## The projection factor

`c = ⟨Ŵ,W⟩/⟨W,W⟩`, the fraction of weight energy retained.

| fit | c | 1 − ε² | orthogonality |
|---|---|---|---|
| Q8_0 | 1.0000 | — | error 0.10% |
| reference abs-max ternary | 0.7678 | — | error 81% |
| dedicated single plane (ours) | 0.8072 | 0.8072 | 0 |
| joint plane-1, used alone | 0.8821 | 0.8018 | −0.18 |
| joint pair | 0.9659 | 0.9659 | 0 |

The identity holds wherever the fit is genuinely least-squares optimal. The
production scale search was checked against a 60-point grid with fixed-point
refinement: 43.905% against 43.897%, so there is no headroom in it.

## Error concentration

Top 1% of weights carry **18%** of the ternary error; top 10% carry **63%**.
Identical in `up` and `down` (18.3% against 18.6%), so it does not explain why
they behave differently.

## Input statistics

| | effective rank | kurtosis |
|---|---|---|
| residual stream (input of gate/up) | 23.2% of dimension | 6–32 |
| SwiGLU output (input of down) | 69.6% | 45–150, and 38,770 at layer 0 |

Measured on 1,883 and 15,064 samples respectively. An earlier attempt with two
tokens produced meaningless numbers and was discarded — with n samples the
covariance has rank at most n−1.

## Standardised benchmarks

Run on the stripped 84.00 GiB build with llama.cpp's built-in scorer.

| benchmark | score | interval | tasks | chance |
|---|---|---|---|---|
| HellaSwag | **74.67%** | [69.45, 79.26] | 300 | 25% |
| Winogrande | **70.33%** | ± 2.64 | 300 | **50%** |

Winogrande is the more informative of the two here, for two reasons. Its chance
floor is 50%, not 25 — a model that has stopped understanding scores 50, so
70.33% is unambiguous evidence of real pronoun-resolution ability rather than a
partially-degraded prior. And its interval is half as wide, since it is a
two-choice task.

⚠️ **Two caveats, both material.** These come from `llama-perplexity`'s own
implementation, which the project states is "linearly correlated but not the
same numbers" as the leaderboard, and which reports raw accuracy where lm-eval
reports the length-normalised figure by default — so this belongs in a column
of its own, not beside a published score. And 300 tasks gives ±5 points, which
is honest but wide; the number distinguishes "the model works" from "the model
is broken", not one good model from another.

**Where that sits.** Scored with the *same tool and protocol*, a modern MoE
(Qwen3.5-35B-A3B) holds 83.0% at ~8 bpw, 81.0% at ~2.8, and 80.3% at ~2.6. We
are six points below the 2.6-bpw point with almost a full bit less — a further
step down the same gentle curve, not a cliff.

The cliff is documented elsewhere and looks different. Published 2-bit
post-training quantization of LLaMA3-8B and 70B collapses to **50–64%** with
*every* method tried, good or bad (arXiv 2404.14047). A broken model sits at
25% or below. BitNet b1.58 2B4T, a model *trained from scratch* for ternary
weights, reaches 68.4%.

So: 74.67% at 1.69 bpw, from post-training quantization of weights never
intended for it, is above both the published 2-bit PTQ band and a
natively-trained ternary model — while remaining below what a well-executed
2.6-bpw quant achieves. That is the honest position.

**No published number with this scorer exists below 2 bpw for a model of this
size.** This is new data rather than a comparison against an existing point.

⚠️ Read as an indication, not a fine measurement: ±5 points at 300 tasks means
only differences of ten points or more are safely interpretable, and HellaSwag
has documented validity problems of its own (arXiv 2504.07825).

## Behaviour

Five verifiable questions, sampling `temp 0.6 · top_p 0.9 · min_p 0.03 ·
presence 1.5`, 800 tokens, on the stripped 84 GiB build: **5/5**, including a
two-step discount problem solved with working shown, in 6–27 seconds per answer
and with an empty thinking block — it answers directly.

⚠️ With `temp 0` and 40 tokens the same harness returns empty strings from a
healthy model. Below two bits the sampling recipe is part of the measurement.

## Cost

| | |
|---|---|
| forge, 397B | 7.6 h end-to-end on one consumer GPU, fed from a NAS at ~50 MB/s |
| decode | ~10 tok/s, fully resident in 96 GiB of unified memory |
| what limits it | **memory bandwidth, not compute** — during a HellaSwag run the GPU reports 99% utilisation while accumulating only 5 minutes of CPU-side work over 22 minutes of wall clock. With 512 experts, every forward pass gathers the routed experts from unified memory, and that gather is the bottleneck. It is also why multiple-choice benchmarks, which are prefill-bound in theory and should be fast, are not as fast as the theory promises on this hardware |
| stripping the second plane | 84.00 GiB written in ~8 minutes |


## What this evidence does not yet support

Written down before anyone else has to point it out.

**Two models is not enough for the mixture result.** Recent MoE-quantization
papers use three or four base models; we have one 397B and one 35B, and both
are the *same architecture family* (256 and 512 experts, 41 and 60 layers,
identical tokenizer at 248,320 tokens) forged by the same pipeline. A reader is
entitled to ask whether the effect is a property of these two checkpoints.

The obvious third point is a model with a **different architecture**, not just
a different size: 128 experts across 94 layers instead of 256-512 across 41-60.
Same routing mechanism, very different shape — which is exactly what
distinguishes "a property of this family" from "a property of mixtures". The
forge would need one run — **and one engine change that is easy to miss.** The
second-plane tensors are declared once per architecture, in that architecture's
model file. A different family does not load them: the forge would write a
correct file and the engine would ignore it in silence, so the measurement would
read "no difference" when the truth is "never loaded". Three lines, copied from
the existing architecture, but they have to be written before the run. Same
failure shape as the environment switches earlier here — an absence of effect
that looks like a finding.

**Perplexity and five questions are not a downstream evaluation.** Every
comparable paper pairs perplexity with a task suite — commonsense (PIQA, ARC,
HellaSwag, WinoGrande), knowledge (MMLU), reasoning (GSM8K).

⚠️ **And `llama-perplexity --hellaswag` numbers are not the published ones.**
The project's own discussion (ggml-org/llama.cpp#2321) states results are
"linearly correlated but are not the same numbers" as the leaderboard, and the
metric differs — `acc` against the length-normalised `acc_norm` that lm-eval
reports by default. Such a figure is valid for comparing two of our own builds
under one recipe; it cannot be placed in a column beside a published score.

The defensible route is one consistent toolchain: `llama-server` plus
lm-evaluation-harness through the `local-completions` adapter, with the harness
version, the metric and the shot count all pinned and stated. Two practical
facts make it cheaper than it looks:

- **Multiple-choice tasks are prefill-bound, not decode-bound.** Each question
  generates a single token; cost is dominated by prompt processing, which is far
  faster than our ~10 tok/s generation figure, and the shared few-shot prefix is
  cached across questions. MMLU, ARC, PIQA and BoolQ are hours, not days.
- **GSM8K is the expensive one** — it generates full reasoning chains, so at
  ~10 tok/s and a few hundred tokens per question it is 11-15 hours. Budget for
  it separately or subsample it.

If subsampling, use a **named** method: tinyBenchmarks (arXiv 2402.14992) ships
100-item IRT-selected subsets as lm-eval tasks with a published estimation error
under 2%. "We evaluated a random thousand" is the single most likely reviewer
objection; `tinyMMLU` is a citation.

**The obvious alternative explanation has not been excluded.** The field's
default account of unexpected MoE quantization damage is routing shift —
rank flips at the top-k boundary. Our own leading hypothesis *is* routing
shift, which means the correct framing is not "an anomaly with routing held
fixed" but "routing-mediated damage caused by a *correction* rather than by
quantization". Either way it has to be measured, not asserted: count the
selection changes layer by layer, and compute the Route-Mediated Fraction
(arXiv 2608.11212).

**Ablations a reader would rightly demand**, none of which are done: vary
*which* experts receive the extra precision (random against
importance-ranked), hold the calibration set fixed across configurations,
sweep more than one bit budget, and plot local proxy error against task
accuracy across many configurations rather than comparing two points.

**On the projection identity.** `c = 1 − ε²` follows from the orthogonality
principle, which is decades-old vector-quantization theory. The contribution
is measuring it on ternary LLM weights and drawing the consequence — not
deriving it. It belongs as a lemma inside the pipeline work, not as a claim of
its own.
