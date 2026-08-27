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
