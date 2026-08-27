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
thirteen times narrower than an average expert's share.

⚠️ **That thinness is typical, not anomalous, and saying otherwise would be the
weaker claim.** The one published aggregate on fine-grained MoE routing
(arXiv 2602.02443, measured on Qwen3-30B-A3B, Ling-Lite-1.5, GPT-OSS-20B and —
closest to us — Qwen3-Next-80B-A3B at 256 experts) reports the score gap from
rank 5 to rank 32 as under 1.5% in total; back-computing gives a mean per-rank
tail gap around 14× thinner than the mean expert share, against our 13×. Those
agree. So the mechanism is a property of fine-grained MoE routing generally,
not an artefact of these two checkpoints — which is what makes it worth
reporting.

The comparison is suggestive rather than a replication: theirs is a mean over
ranks 5–32 of a token-averaged curve, ours a median of the specific 8↔9 gap
across tokens, on different models. What is genuinely absent from the
literature is the *distribution*: that paper plots a mean curve and reports no
median, no histogram, no percentiles. Searches for "routing margin", "expert
selection margin" and "router logit gap" return nothing; "router margin"
returns exactly one paper, which computes the quantity per token and collapses
it to a single AUC. Note also that we measure in **probability** space while
that paper measures in **logit** space — a difference to state explicitly
rather than let a reader discover.

### Where the damage lives — a narrow mountain, not an accumulation

The correction enabled on **one layer at a time**, 35B model, 12 chunks:

| layer | perplexity | | layer | perplexity |
|---|---|---|---|---|
| off | 8.7204 | | 25 | **22.5109** |
| 0 | **10.9524** | | 30 | **8.6860** ← better than off |
| 5 | 8.7277 | | 35 | 8.7743 |
| 10 | 8.8125 | | 39 | 8.7370 |
| 15 | **20.1161** | | | |
| 20 | **37.8030** | | | |

The damage is a narrow peak centred on layer 20, plus a secondary bump on layer
0, and nothing anywhere else. Band 20-39 measures 37.6144 — indistinguishable
from layer 20 alone, so the other nineteen layers contribute nothing.

This falsifies the natural reading of the routing result. We predicted damage
*decreasing* with depth, since a deeper layer has fewer downstream routers left
to disturb. Layer 5 has thirty-five below it and costs 0.007.

By band:

| band | perplexity | |
|---|---|---|
| off | 8.7204 | reference |
| 1-12 | 9.0246 | |
| 13-27 | 25.3425 | the mountain |
| 16-24 | 19.0623 | its core |
| 19-21 | 10.0830 | |
| 28-39 | 8.7077 | |
| **30-39** | **8.3576** | **best, on a quarter of the layers** |

### Perturbations cancel instead of accumulating

The result we cannot yet explain, and the sharpest one:

| | perplexity |
|---|---|
| layer 20 alone | **37.8030** |
| layers 19-21 | **10.0830** |
| all 40 layers | **10.9106** |

Perturbing the two neighbours of the worst layer removes three quarters of its
damage. Perturbing everything is three and a half times better than perturbing
one thing. **Adding error reduces damage**, reproducibly — every figure here
repeats to four decimals across runs.

⚠️ **This phenomenon is already published and must not be presented as a
discovery.** EvoPress (arXiv 2410.14649, ICML 2025) opens on it: *"current
methods rely on estimating the importance of a given layer, implicitly assuming
that layers contribute independently to the overall compression error… this
independence assumption does not generally hold for LLM compression: pruning a
model further may even significantly recover performance."* Their Table 1
reports depth pruning that is not monotone — removing strictly more blocks can
improve perplexity — and that models with a *lower* sum of per-layer errors can
perform worse.

What remains ours is narrower and should be stated as such: the **magnitude**
(a 29-perplexity single-layer effect that thirty-nine further perturbed layers
reduce to 2.2, against their block-level non-monotonicity), the
**localisation** (EvoPress explicitly does not isolate individual catastrophic
layers), and any **mechanism** — they report the phenomenon and stop.

The available mechanistic candidate, untested here and not previously connected
to quantization: self-repair. Rushing & Nanda (arXiv 2402.15390, ICML 2024)
identify its two mechanisms as *"changes in the final LayerNorm scaling
factor"* and *"sparse sets of neurons implementing Anti-Erasure"*, and note it
is imperfect and noisy — which matches a cancellation that is partial rather
than complete. The Hydra Effect (arXiv 2307.15771) reports the same shape:
ablating one layer causes another to compensate.

Note also that this cuts directly against the founding assumption of the
error-propagation line: *Quantization Error Propagation* (arXiv 2504.09629)
frames the whole problem as errors that accumulate and grow across layers.

That looked like the signature of a normalisation absorbing a change that is
*consistent* across depth while an isolated one stands out — which would revive,
at layer granularity, the scale explanation refuted at expert granularity. The
prediction separating it was sharp: compensation must help *a lot* on layer 20
alone and *little* on all layers.

| configuration | perplexity | with compensation |
|---|---|---|
| layer 20 alone | 37.8030 | 37.6701 (×1.093) · 37.6473 (×1.045) |
| layer 30 alone | 8.6860 | 8.6818 |
| all 40 layers | 10.9106 | **11.6068** |

It removes 0.13 of 29 points where it was meant to remove most, and harms the
uniform case. Inert where predicted decisive, harmful where predicted neutral.
**Refuted.** The cancellation remains real and unexplained.

### The number of route changes does not predict the damage

If routing mediates the damage, the count of changed expert sets should order
the configurations the way perplexity does. It does not — it very nearly
inverts them.

| configuration | tokens whose expert set changes | perplexity |
|---|---|---|
| layers 30-39 | 11.37% | **8.3576** — the best |
| **layer 20 alone** | **14.84%** | **37.8030** — the worst |
| layers 19-21 | 21.80% | 10.0830 |
| **all 40 layers** | **78.67%** | 10.9106 |

All forty layers cause **five times** the route churn of layer 20 alone and
the model is **three and a half times better**. Correlation −0.294, orderings
different.

Read per *affected* layer the contrast is sharper still, not weaker: layer 20
can only disturb the nineteen routers below it, so its 14.84% of the whole
stack is roughly 30% of the layers it can reach; the 30-39 band reaches nine
layers, so its 11.37% is roughly 45% of them — more churn where it reaches,
and it is the best configuration measured.

**So it is not how many routes change, it is which.** That is consistent with
the one published attempt to tell harmful flips from benign ones: a local
margin heuristic predicts *whether* a flip happened at AUC 0.772 and whether it
was *harmful* at AUC 0.490 — chance. Flip rate is not a proxy for damage, and
any method that optimises it is optimising the wrong quantity.

### Routing concentration, which does not explain it

The correction covers the 28 hottest experts of 256, so an obvious candidate
for the layer profile is that traffic is spread more evenly on the damaged
layers — 28 experts would then cover less of it, leaving the inconsistency
larger. Computable from the imatrix's per-expert counts alone, before forging
anything, in one minute and with no GPU.

| layer | share carried by the 28 hottest | normalised entropy | damage |
|---|---|---|---|
| 0 | 23.84% | 0.9778 | 10.9524 |
| 10 | 26.44% | 0.9597 | 8.8125 |
| 15 | 29.59% | 0.9524 | 20.1161 |
| **20** | **32.04%** | **0.9440** | **37.8030** |
| 25 | 33.40% | 0.9420 | 22.5109 |
| **30** | **30.95%** | **0.9481** | **8.6860** |
| 39 | 44.93% | 0.8912 | 8.7370 |

Correlation with damage: **+0.013** for the hot share, **+0.037** for entropy.
The most-damaged and least-damaged layers carry almost exactly the same share
(32.04% against 30.95%) with four times the difference in damage. **Refuted.**

One real finding falls out sideways, independent of our problem: **routing
concentrates with depth.** The 28 hottest experts carry 23.8% of traffic at
layer 0 and 44.9% at layer 39, with entropy falling monotonically. Deeper
layers rely on a narrower set of experts. It does not predict damage, but it is
a property of the trained model worth reporting — and it means a fixed *K*
hot experts per layer covers very different fractions of the traffic at
different depths.

### The correction is the same size everywhere

The last trivial explanation for the layer profile: perhaps the correction is
simply larger where it does harm. Read straight from the file, no GPU — the
norm of the second plane over the norm of the first, averaged over hot experts.

| layer | up | gate | down | damage |
|---|---|---|---|---|
| 0 | 0.4147 | 0.4151 | 0.4137 | 10.9524 |
| 15 | 0.4174 | 0.4172 | 0.4141 | 20.1161 |
| **20** | **0.4147** | **0.4139** | **0.4119** | **37.8030** |
| 25 | 0.4177 | 0.4188 | 0.4163 | 22.5109 |
| **30** | **0.4177** | **0.4225** | **0.4161** | **8.6860** |
| 39 | 0.4165 | 0.4183 | 0.4208 | 8.7370 |

The ratio sits between 0.412 and 0.423 at every layer of the model — a 2%
spread — while the damage varies four-fold. The correlations are *negative*
(-0.18, -0.56, -0.57): where it hurts most, the correction is marginally
*smaller*. **Refuted**, and the refutation sharpens the result rather than
weakening it: an identical perturbation, applied at the same relative size
everywhere, has incomparable consequences depending only on depth.

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

### The shape of the damage, and of the gain

Perplexity is a mean, and a mean of 37.80 against 8.72 can come from every
chunk getting worse or from two chunks catching fire. The cure differs.
Per-chunk contributions, 12 chunks:

| chunk | off | layer 20 only | all layers | layers 30-39 |
|---|---|---|---|---|
| 1 | 7.74 | **88.36** | 10.21 | **7.45** |
| 4 | 10.38 | **120.68** | 12.70 | **10.12** |
| 9 | 12.03 | 26.07 | 16.14 | **11.12** |
| 12 | 12.93 | 34.40 | 16.75 | **12.20** |

| configuration | worst chunk | median chunk |
|---|---|---|
| layer 20 only | ×11.62 | ×3.20 |
| all layers | ×1.39 | ×1.28 |
| layers 30-39 | ×0.98 | ×0.96 |

The damage is **diffuse**, with a heavy tail: layer 20 makes every chunk worse
by a median factor of 3.2, and the worst by 11.6. It is not a handful of tokens
catching fire — the model is genuinely and uniformly worse.

The gain is diffuse too, which matters more: the 30-39 band improves **all
twelve chunks**, none excepted. The improvement is not luck on a slice of
corpus.

### One projection does all of it

Decomposing the worst layer answers what eleven refuted hypotheses could not.
Layer 20 alone, one projection at a time:

| at layer 20 only | perplexity |
|---|---|
| reference, correction off | 8.7204 |
| `up` alone | **8.6984** — better |
| `gate` alone | **8.7163** — better |
| `up` + `gate` | 8.7400 |
| **`down` alone** | **37.9258** |
| all three | 37.8030 |

`down` accounts for the entire catastrophe — marginally more than all three
together, so `up` and `gate` even mitigate it slightly. Both of them, applied
alone at the worst layer in the model, measure *better* than not applying the
correction at all.

So the whole result is one sentence: **the second plane on the `down`
projection is catastrophic on the mountain layers and helpful in the tail;
`up` and `gate` are harmless everywhere.** That also explains the global
projection profile measured much earlier — `down` on every layer read 11.2405
only because "every layer" includes the mountain.

### The same phenomenon, unremarked, in someone else's published table

A deliberate sweep of the MoE mixed-precision literature found it uniformly
monotone in its *assumptions* — every expert-wise bit-allocation paper frames
the only failure mode as mis-estimating importance, never as being right and
still losing. But the phenomenon is visible in published data:

- **GEMQ (arXiv 2605.23078, ICML 2026), Table 1, Qwen3-30B-A3B.** Expert-level
  mixed precision loses to plain uniform quantization at matched bits-per-expert
  across all three budgets, and at 2.0 bpe on all three metrics simultaneously
  (−9.89 accuracy). The paper covers it with a single hedging adverb
  ("generally") and never returns to it. **That is our result, in someone
  else's experiment, unremarked.**
- **HiFloat4 (arXiv 2607.26515).** Verbatim: *"Counterintuitively, restoring
  the training policy to higher precision while keeping the rollout in FP4
  makes accuracy worse than full FP4 baseline"* — 78.01% against 82.03% for
  uniform FP4. Different setting (training/rollout rather than expert
  granularity) and the mechanism they name is mismatch rather than noise, which
  is the same shape as ours.
- **TurboAngle (arXiv 2603.27467)** reports a strict-superset violation, though
  as a single-author preprint with effect sizes near its own noise floor.

⚠️ The framing that follows is **not** "more bits hurt" — that is false and
easy to refute. It is **heterogeneous fidelity hurts**: three independent
literatures converge on consistency across components mattering more than
fidelity of any one of them. Our contribution is a controlled, localised
instance with a mechanism to test, not the discovery of the effect.

**Documented absence:** no paper applies a residual or second plane
*selectively* to a subset of MoE experts, which is the primitive here.

### The alternative explanation, and why the data already excludes it

The sharpest objection a reader can raise: the literature localises *massive
activations* and *super weights* at layers 1-4 (arXiv 2402.17762, 2411.07191)
and *super experts* at layers 1-3 in MoE (arXiv 2507.23279). Those scalars are
created early and propagate unchanged through the residual stream. A damage
metric sensitive to propagated state would naturally peak somewhere downstream
of the source — possibly mid-depth — for reasons entirely about propagation
rather than about mid-depth weights being special. Our layer-0 bump plus a
layer-20 peak is exactly the pattern that alternative predicts.

The projection decomposition excludes it. In a SwiGLU block, `up` and `gate`
read the **residual stream** — where a propagated massive activation lives —
while `down` reads the **SwiGLU product**, which is local to that layer. If the
damage were mediated by propagated state, the projections reading that state
would be the damaged ones. They are the harmless ones: at the worst layer in
the model, `up` alone (8.6984) and `gate` alone (8.7163) both measure *better*
than applying no correction, while `down` alone measures 37.9258.

The damage is therefore attached to the layer's own SwiGLU output, not to what
passes through it.

Two further facts point the same way. The correction's relative magnitude is
constant with depth (0.412-0.423), so the peak is not a bigger perturbation.
And the `down` input's kurtosis is 38,770 at layer 0 against 45-150 elsewhere —
so the layer with the extreme activation statistics is *not* the layer with the
extreme damage, which is the opposite of what the propagation account needs.

⚠️ Still to do: a per-layer curve of the `down` input statistics, so that
"layer 20 is unremarkable in its activations" is a measurement rather than an
inference from two sampled layers.

### The recipe, written into the file

The two profiles compose. `up` and `gate` help at every depth; `down` helps
only in the tail. No engine switch expresses that, and none is needed — the
second-plane tensors are optional per layer, so the recipe goes into the file:

    strip_tensors.py in.gguf out.gguf --match ffn_down_exps2 --keep-layers 30-39

| configuration | perplexity | |
|---|---|---|
| second plane off | 8.7204 | reference |
| `up`+`gate` every layer | 8.3493 | previous best |
| all three, layers 30-39 | 8.3576 | |
| **`up`+`gate` everywhere, `down` on 30-39 — in the file** | **8.2501** | **−5.4%** |

The file carries the recipe rather than the operator having to remember it,
which is also the only form in which a recipe survives being handed to someone
else.

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

⚠️ **Caveats, and they are worse than "not the same numbers".** Read from the
scorer's own source rather than taken on trust:

- `hellaswag_score()` is labelled **`acc_norm`** but normalises the ending
  log-probability by **token count**; lm-eval normalises by **character
  length**. Same name, two different quantities, and the gap is
  tokeniser-dependent. An earlier version of this document said we report raw
  `acc` — that was wrong in a way that would have understated the
  incomparability.
- Its preprocessing is pinned to a **2023-era harness commit** (`df3da98`), and
  the source itself says all 10,042 tasks should be used "to keep the results
  standardized like other implementations". Our 300 is below what the author
  considered sufficient, drawn with a hardcoded seed (`std::mt19937 rng(1)`).
- Winogrande is worse: it reads a **custom CSV** rather than
  `allenai/winogrande`, and carries an in-source
  `// FIXME: this uses the wrong first logits when not skipping the choice
  word`. lm-eval's winogrande has no `acc_norm` at all.

So the honest column header is
`HellaSwag (llama.cpp acc_norm — token-normalised, 300-task subset, seed=1)`,
with one caption sentence saying these are not leaderboard-comparable and why.
And 300 tasks gives ±5 points: the figure distinguishes "the model works" from
"the model is broken", not one good model from another.

**The metric to move to is not a better accuracy — it is KL divergence.**
*Accuracy is Not All You Need* (arXiv 2407.09141, NeurIPS 2024) shows compressed
models are behaviourally different from their baseline even at equal accuracy,
and argues for KL divergence and token *flips* instead. llama.cpp already
computes both: `--kl-divergence` reports mean KLD, mean Δp and same-top-p%, and
its own README states the reading rule that matches this project's argument —
symmetric percentiles mean the quantization is adding noise, while negative
values larger than positive means the model is genuinely worse.

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

**Two incompatible metric conventions exist, and almost nobody declares which
one they used.** A survey of 25 recent quantization papers found only 4 that
state `acc` against `acc_norm` explicitly, and only 4 that pin an lm-eval
version. The split runs by community rather than by year: the PTQ-compression
lineage reports raw `acc`, the model-release lineage reports `acc_norm` on
length-varying multiple choice. WinoGrande and MMLU are plain `acc` everywhere.

The cautionary case is worth stating because it looks like rigour: "we follow
the evaluation setup of GPTQ" is **not** reproducible. Two papers from the same
group, carrying that identical sentence, differ by 7 points on the
*unquantized* ARC-e baseline — a silent lm-eval v0.3 → v0.4 bump sits inside
the same prose. Naming the protocol you followed is not the same as pinning it.

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
