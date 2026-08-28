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

#### The distribution

75,320 token·layer observations, 40 layers, plane off — the model as served.

| percentile | margin | times narrower than 1/256 |
|---|---|---|
| 1% | 0.000005 | **806×** |
| 5% | 0.000026 | 150× |
| 10% | 0.000054 | 72× |
| 25% | 0.000148 | 26× |
| **50%** | **0.000385** | **10.1×** |
| 75% | 0.000863 | 4.5× |
| 90% | 0.001617 | 2.4× |
| 99% | 0.004241 | 0.9× |
| mean | 0.000678 | 5.8× |

**The distribution is skewed — mean/median = 1.76 — which is exactly why the
mean is the wrong statistic and why this measurement is worth making.** The one
published aggregate reports a token-averaged curve; reporting the mean in place
of the median makes the margin look 1.8× wider than it is for half of all
tokens.

Fraction of tokens below a threshold: **17.7%** under 1e-4, **42.4%** under
3e-4, **79.1%** under 1e-3.

**2.95% of tokens sit below one bf16 ULP** at the typical probability scale
(1.53e-5). That is the quantitative half of what sglang PR #35916 asserts
qualitatively on this same 256-expert top-8 topology — and it asserts it on
*synthetic* weights, its author writing that a real comparison would be
stronger but that they lack the hardware. This is that comparison.

The margin also narrows with depth: median 0.00064 at layer 0, 0.00020 at
layer 38.

#### What is ours in that number, precisely

Three independent sweeps of ~110 papers found no median, percentile, histogram
or CDF of the top-k boundary gap anywhere. But the *quantity* is defined twice
and never measured, so the claim must be the measurement, not the concept:

- **ReMoE (arXiv 2605.27081), Appendix C.2**, "Top-K Stability Under a
  Probability Margin", defines **γ = q_(K) − q_(K+1)** on post-softmax
  probabilities — our exact quantity, in our exact space, at our exact
  boundary. It appears only inside a stability lemma and its proof, as an
  unmeasured premise.
- **arXiv 2608.11212** names the same thing in *logit* space ("weakest-selected
  minus strongest-unselected") and reduces it to a single detector AUC (0.772).
- **arXiv 2602.02443, Fig. 2(c)** gives a token-*averaged* mean curve over
  ranks on a 128-expert model.

So the field has the concept, the name and the mean, and lacks the
distribution — which is the part that matters, because a mean gap on a skewed
distribution is not the median and cannot tell you that half of all tokens sit
at 0.0003. The honest sentence is *"ReMoE assumes γ is nontrivial; we measure
it, and the median is 0.0003"*. **"We introduce the boundary margin" would be
false.**

⚠️ **And someone said the qualitative half of it in public seven days ago.**
sglang PR #35916 (2026-08-21), on the same 256-expert top-8 topology: *"adjacent
expert scores routinely sit within one bf16 ULP, so the narrower store changes
which experts run — a different answer, not a small numeric difference."* It
reports flip rates only (0.4-2.7% of tokens) on **synthetic** weights, and its
author writes that a real comparison would be stronger but they lack the
hardware for it. Our measurement is precisely what that PR is missing. Cite it,
and move.

⚠️ One reported opportunity **did not survive checking**: a brief suggested
arXiv 2608.07911 had published per-request traces carrying a `margin` field,
making a second-model cross-check "one numpy line away". The Zenodo record
holds only the manuscript and two metadata files, and the repository is 3 MB of
tooling. The traces are not published as data.

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

⚠️ **The projection asymmetry itself is not new, and claiming it would be
caught.** D²Quant (arXiv 2602.02546, Jan 2026) — weight-only PTQ at sub-4-bit,
our exact regime — states it as settled: *"down-projection matrices are a
well-known quantization bottleneck, but maintaining their fidelity often
requires extra bit-width"*, and designs a dual-scale quantizer specifically for
them. What no one does is *derive* it from the gated-activation statistics;
D²Quant asserts the bottleneck and engineers around it.

⚠️ **And the depth localisation is contradicted by four independent sources,
one of them GLU-specific.** *Mitigating Quantization Errors Due to Activation
Spikes in GLU-Based LLMs* (arXiv 2405.14428, EMNLP 2024) reports the opposite
band: *"The activation spikes occur in the FFN of specific layers, particularly
in the early and late layers"* — tested on LLaMA-2/3, Mistral, Mixtral, SOLAR
and Gemma. llama.cpp's own `use_more_bits` heuristic spends extra bits on the
first and last eighth. Two further analyses agree on edges.

Our band is 31.7-65.9% of depth on the 35B and 0-65% on the 397B: the middle,
not the edges. The available distinction is that those works measure
*activation spike magnitude* in a dense GLU FFN while we measure
*weight-perturbation damage* in a 1.69-bit MoE — two genuinely different
quantities. But that is a claim to demonstrate, not to assert, and it is why
the per-layer effective-rank measurement below is not optional.

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

### Where this sits: consistency, not smallness, is the safety property

A sweep of the perturbation literature found the pieces of this argument
scattered across three fields and assembled by nobody. Stated honestly: the
individual facts are published, the synthesis is ours, and the step the
synthesis actually needs is *not* proved anywhere.

**Locality decides damage, not magnitude — and the cleanest number is nine
years old.** LLM.int8() (arXiv 2208.07339, §4) zeroes 0.1% of features (the
outlier dimensions) and perplexity degrades **600-1000%**; zeroing **seven
random** features costs **0.1%**. Same operation, same coordinate count, four
orders of magnitude of difference in consequence. Independently, an L2-matched
study (arXiv 2602.11169) finds angular perturbation does up to **42.9×** more
damage to LM loss than a magnitude perturbation of identical Euclidean size.

**A consistent multiplicative perturbation is mathematically free.** SmoothQuant
(arXiv 2211.10438) fuses its per-channel scale into the preceding LayerNorm
"offline"; SliceGPT (arXiv 2401.15024) proves `RMSNorm(XQ)Qᵀ = RMSNorm(X)` for
the entire orthogonal group and absorbs LayerNorm's `diag(α)` into `W_in`.
RMSNorm is stated outright to *discard* input norm information (arXiv
2510.22777).

**And normalisation actively repairs damage**: final-LayerNorm rescaling alone
undoes about **30%** of an ablated head's direct effect (arXiv 2402.15390).

**The non-monotonic shape is published too, twice.** QDrop (arXiv 2203.05740)
shows randomly *dropping* activation quantization half the time beats both no
quantization and full quantization during calibration. Quant-Noise (arXiv
2004.07320) is sharper still on a language model: int8 with noise on a random
5% subset gives **18.7** perplexity, no noise gives **19.6**, and *full*
quantization-aware training gives **21.0** — worse than doing nothing. Partial
beats both ends, which is the shape of our cancellation.

⚠️ **What none of them prove, and what our argument needs.** SmoothQuant and
SliceGPT establish the invariance for *exact* transforms. Nobody has shown it
degrades gracefully for *approximate* consistent perturbations — which is
precisely the step from "an exact rescale is free" to "a roughly-uniform
perturbation is cheap". That is an assumption, and it should be labelled one.

⚠️ **Two facts that cut against us**, both from the papers themselves: the
L2-matched study finds magnitude perturbation *does* disproportionately damage
fine-grained syntax (20.4% against 1.6% on subject-verb agreement), so
"absorbed" holds for aggregate loss and not everywhere; and scale invariance is
noted to hold only "approximately in normalization-heavy networks".

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

⚠️ **But the residual stream is not ordinary at the damaged layer, and that
weakens this defence.** Measured from captured activations, effective rank of
the residual stream per layer: **15.58% at layer 20** against **10.39% at layer
30** — a 50% difference in the wrong direction, correlating with damage at
**+0.651**. The curve rises from 5.0% at layer 0 to a plateau of ~15% across
layers 15-27 — which is our damage band — then falls monotonically to 5.5% at
layer 39.

So the layer where the correction is catastrophic *is* distinguished in its
activations, and an earlier version of this document said the propagation
account was excluded. It is not excluded; it is weakened by the projection
decomposition and left standing by this measurement. Both facts belong in the
paper.

The remaining distinction is still real but now carries the burden of proof:
`up` and `gate` read this residual stream and are harmless at that very layer,
while `down` reads the layer-local SwiGLU product and carries the entire
catastrophe. If the residual stream's rank were the mediator, the projections
reading it should suffer. The measurement that resolves this is the same
effective-rank curve computed on the `down` input, which is pending.

Two further facts point the same way. The correction's relative magnitude is
constant with depth (0.412-0.423), so the peak is not a bigger perturbation.
And the `down` input's kurtosis is 38,770 at layer 0 against 45-150 elsewhere —
so the layer with the extreme activation statistics is *not* the layer with the
extreme damage, which is the opposite of what the propagation account needs.

⚠️ Still to do: a per-layer curve of the `down` input statistics, so that
"layer 20 is unremarkable in its activations" is a measurement rather than an
inference from two sampled layers.

### Every local explanation is excluded by measurement

The chain of pre-registered predictions, each written before the run:

| candidate | prediction | measured | verdict |
|---|---|---|---|
| correction magnitude | larger where it hurts | 0.412-0.423 at **every** layer, ρ = −0.56 | refuted |
| routing concentration | lower where it hurts | ρ = **+0.013** | refuted |
| residual-stream rank | peaks with damage | ρ = +0.651 on 35B, **opposite sign on 397B** | not replicated |
| `down`-input kurtosis | peaks with damage | ρ = **−0.186** | refuted |
| `down`-input effective rank | peaks with damage | ρ = **+0.284** | refuted |
| hot-first permutation | wrong at damaged layers | **28/28 correct on all 40 layers** | refuted |
| **projected error ‖ΔW·X‖** | larger where it hurts | see below | **refuted** |

The last one is the decisive negative, because it is the quantity the model
actually experiences. Under the layer's own captured activations:

| layer | ‖ΔW‖/‖W₁‖ | ‖ΔW·X‖/‖W₁X‖ | ratio | damage |
|---|---|---|---|---|
| 10 | 0.4154 | 0.4158 | 1.001 | 8.81 |
| 15 | 0.4140 | 0.4144 | 1.001 | 20.12 |
| **20** | **0.4115** | **0.4120** | **1.001** | **37.80** |
| 30 | 0.4168 | 0.4168 | 1.000 | 8.69 |

**The perturbation delivered to the output is identical at every layer** —
ratio between 1.000 and 1.002 throughout, so the activation covariance neither
amplifies nor attenuates it differentially. Correlation with damage is
**−0.599**: where the damage is greatest, the delivered perturbation is
slightly *smaller*.

So the statement the evidence supports is not "we do not understand it" but
something sharper and testable: **an identical perturbation, delivered
identically to the layer's output, produces damage that varies by a factor of
1,579 depending only on which layer receives it.** Nothing local to the layer —
its weights, its activations, or the projection of one onto the other —
distinguishes the layer that breaks the model from the one that improves it.

Whatever mediates this is downstream of the perturbation, and the obvious
downstream candidate has already been measured and does not order the
configurations correctly either (see the route-flip count above). That is the
open problem, stated as narrowly as the measurements allow.

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

## The paired verdict on the 397B tail

The absolute comparison — 6.1591 against 6.1845 at 30 chunks — sits inside the
±0.087 absolute error bar three times over, so it cannot support the claim on
its own. The paired estimator can: same chunks, reference logits stored, the
corpus variance cancelled.

**Mean Δp = +0.048% ± 0.014%** — the tail-44-59 build assigns more probability
to the correct token than the plane-off build, at **3.4σ** with uncertainty
propagated from the logits. Median Δp is also positive. The gain is real.

⚠️ Operational lesson recorded rather than hidden: the harness filtered the
scorer's output through a `grep` that discarded the perplexity-ratio lines; the
verdict survived only because the Δp lines happened to match the filter. Store
raw output, filter afterwards.

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
  log-probability by **token count**; lm-eval's `acc_norm` normalises by
  **character length**. Same name, two different quantities, and the gap is
  tokeniser-dependent. So our figures are neither `acc` nor `acc_norm` but a
  **third quantity**. An earlier version of this document said we report raw
  `acc` — wrong, and wrong in the direction that understated the
  incomparability.
- Worse for the comparison we wanted to draw: **lm-eval's Winogrande has `acc`
  only — no `acc_norm` exists** — and the same is true of MMLU. The sentence
  "lm-eval reports the length-normalised figure by default" is false for both
  tasks we ran.
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

⛔ **And the deeper problem is statistical power, not naming.** At n = 300 the
standard error is ±2.51 pp on HellaSwag and ±2.64 pp on Winogrande, so the
95% intervals are about 10 points wide and the smallest difference resolvable
between two builds is **6.9 pp**. Quantization damage is typically 1-4 pp.
**This setup cannot see the effect it was run to measure.** A ±1 pp half-width
needs roughly 7,200 items; the full HellaSwag validation set is 10,042, which
the scorer's own source says to use.

These two numbers therefore stay in this document as evidence that the model is
not broken — which is what they can support — and must not appear in a results
table as a measurement of quantization damage.

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


## ⛔ The two threats that decide whether any of this is publishable

Found by deliberate adversarial search, and both are more serious than anything
in the section below.

### 1. Every number here is perplexity, and perplexity inverts orderings

This is the threat that would sink the paper, and it is not speculative.
arXiv 2607.16721 §5.4, titled *"Perplexity is not a viable gate for pruning
studies"*, measures a randomly-pruned model scoring **better held-out code
perplexity than the base model — 4.82 against 5.38 — while losing 57.9 points
of pass@1**. On its second model the perplexity ranking across criteria
*anti-correlates* with the pass@1 ranking. Corroborated three ways: EvoPress
Table 3 (an allocation 2.4× worse on perplexity is *better* on zero-shot
average); MoPEQ Table 2 (8-bit beats 16-bit on MME-Perception); arXiv 2606.09864
(Mistral-7B loses 15.2% of refusals at 1.03× perplexity, because safety features
occupy a low-dimensional subspace 10²-10³× more vulnerable than the average
perplexity measures).

And the closest prior art in our own method family says it outright — Tied
Trit-Planes (arXiv 2608.08910): *"higher weight-reconstruction error and worse
perplexity, a measured dissociation between proxy metrics and reference
fidelity."*

**Required before publication:** at least one functional evaluation on the
configurations that carry the claim. If the effect survives on a behavioural
metric, it is a result; if not, it is a perplexity artefact.

### 2. The expert-selection criterion is the leading alternative explanation

We select the 28 corrected experts by **routing frequency**. That is precisely
the criterion MoPEQ (arXiv 2509.02512) abandoned — *"instead of relying on the
activation frequency of the expert"* — and that AlphaQ (arXiv 2606.04980)
independently rejected.

AWQ's Table 1 shows what happens when the selection criterion is wrong, at
identical budget and identical mechanism:

| model, INT3-g128 | unprotected | 1% by activation | 1% by weight magnitude | 1% random |
|---|---|---|---|---|
| OPT-6.7B | 23.54 | **11.39** | 22.37 | **24.23** |
| OPT-13B | 46.04 | **10.43** | **48.96** | 42.00 |

**Protecting the wrong 1% is worse than protecting nothing** — 24.23 against
23.54, and 48.96 against 46.04. Same budget, same mechanism; the *sign* of the
effect is set entirely by whether the criterion tracks true sensitivity.

That is structurally our 397B result. So the parsimonious hypothesis is not
"28 of 512 is below a coverage threshold" but **"routing frequency is a good
sensitivity proxy at 256 experts and a bad one at 512"**.

⚠️ The coverage hypothesis is further undermined by the only relevant published
sweep: SqueezeLLM (arXiv 2306.07629, Table D.2) finds the benefit of protecting
sensitive values **saturates at 0.05%** — both 5.5% and 11% are two orders of
magnitude past it, so they should be on the same side of any threshold. And no
paper anywhere reports a threshold on the fraction of experts protected; the
literature is unanimous that efficacy is governed by selection *accuracy*, not
set size (a single super weight moves perplexity three orders of magnitude;
three experts of 6,144 are catastrophic).

**Measured, without reforging anything.** The expected impact of correcting
expert *e* is not its frequency but frequency × ‖W‖² — the injected correction
scales with the expert's size. Comparing the top-28 by frequency against the
top-28 by impact:

| | overlap (frequency vs impact) | impact lost by choosing on frequency |
|---|---|---|
| 35B, layers 0-39 | **24.8/28 mean** | 0.2-6.2% |
| **397B layer 0** | **2/28** | **89.5%** |
| 397B layer 15 | 11/28 | 37.8% |
| 397B layer 30 | 27/28 | 0.2% |
| 397B layer 45 | 26/28 | 1.2% |
| 397B layer 59 | 19/28 | 18.1% |

On the 35B, frequency picks nearly the right set everywhere — and the
correction helps nearly everywhere. On the 397B it misses badly in the early
layers — 2 of 28 right at layer 0, 89.5% of impact lost — **which is exactly
where the damage is worst** (band 0-19: 6.7850), and it is nearly perfect at
layers 30-45, **which is exactly the beneficial tail** (44-59: 6.1591 against
6.1845). Selection quality tracks the damage profile of the 397B layer by
layer, and its difference between the two models tracks the sign flip.

The full 60-layer curve sharpens this and bounds it honestly:

| 397B band | mean selection overlap | mean impact lost | damage (12-chunk ppl) |
|---|---|---|---|
| 0-19 | **9.9/28** | **42.6%** | 6.7850 |
| 20-39 | 25.0/28 | 1.5% | 6.3542 |
| 40-59 | 25.1/28 | 3.4% | 6.1357 |

Per-layer impact-lost against band damage correlates at **+0.788**. And the
limit is visible in the same table: bands 20-39 and 40-59 have essentially
identical selection (25.0 against 25.1) and different damage. **Selection
explains the catastrophic head; the centre-to-tail gradient that remains is a
depth effect** — consistent with the mid-depth fragility measured independently
on the 35B. Two factors, both now quantified, neither reducible to the other.

⛔ **The causal form of this hypothesis is refuted — the eighth
pre-registered refutation.** The prediction was written before the run: band
30-45, whose selection is the best in the entire model (25-27/28 overlap,
0.2-1.5% impact lost), must beat the reference if selection is the mechanism.
Measured: **6.3002 against 6.1845 — it harms.** Selection quality is not
sufficient: with the targets chosen almost perfectly, correcting layers 30-45
still damages the model. The +0.788 correlation was confounded with depth —
the 397B's head is both badly selected *and* shallow, and the experiment that
separates the two has spoken.

**What survives on both models is depth alone: only the final quarter
tolerates the correction.** 35B: layers 30-39 of 41 — from 73% of depth. 397B:
layers 44-59 of 60 — from 73% of depth. The same fraction, at a 10× size
difference. Selection quality remains a real, measured property of the forge
(and choosing by frequency at 512 experts does waste 42.6% of the impact in
the head) — but it does not decide the sign.

Coverage, measured on the same imatrices, moves in the same direction but
weakly: the top-28 carry a median 30.77% of traffic on the 35B against 19.66%
on the 397B — a reduction, not a collapse, and per-layer coverage was already
shown uncorrelated with damage (ρ = +0.013).

⚠️ **Stated plainly: selection explains the 397B profile and the sign flip. It
does not explain the 35B's layer-20 mountain** — that layer's selection is
nearly perfect (25/28, 0.6% lost) and its damage is catastrophic. Two distinct
phenomena, now separable: a selection-criterion failure that grows with expert
count, and a mid-depth fragility that survives every local explanation.

**The confirming experiment stays cheap:** re-select the 397B's 28 experts by
Hessian trace instead of hotness, holding everything else fixed. The Hessians
already exist (16,640 samples per layer). If the sign flips back, coverage is
dead and the mechanism is selection.

⛔ **Do not cite arXiv 2604.18128.** It gives the most seductive account of our
`up`/`gate` versus `down` asymmetry — a reader/generator decomposition of
SwiGLU — and it has been **withdrawn by its authors following internal review**.
The idea that `down` writes into the residual stream while `up` and `gate` only
read from it remains the natural reading of our data, but it has to be
established here, not borrowed.

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

⛔ **Do not use tinyBenchmarks for these numbers, despite the citation being
tempting.** An earlier version of this document recommended it. Checking the
published gp-IRT blend weights shows the reported figure is mostly *not* your
model: the weight on responses actually observed is 11.7% for `tinyMMLU`, 12.9%
for `tinyHellaswag`, 30.5% for `tinyArc` — the rest comes from an IRT model
fitted in January 2024 on 395 full-precision leaderboard models with everything
below MMLU 0.30 filtered out. A quantized model in the near-chance region is
outside the calibration pool by construction, and simulation on the real
published parameters shows it *shrinks* measured quantization damage — 2.95 pp
of true damage reported as 1.69 pp — in the direction that flatters us. At 100
items it also buys no variance advantage over a plain random sample for
measuring a delta.

The defensible alternatives are **metabench** (arXiv 2407.12844, ~850 items,
1.24% RMSE per benchmark) or **MINCE** (arXiv 2606.22826), which is built for
quantized and edge variants and reports 12× lower drift than tinyBenchmarks on
MMLU.

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

## 2026-08-28 — the day's additions (FOEM · impact selection · TQ1_B160 · dense v2)

**FOEM (arXiv 2507.11017) wired into the forge** behind `--foem-beta` /
`FORGE_FOEM_BETA`. Verification protocol and numbers, on a fully-coupled
synthetic Hessian (dense random orthogonal mixing, eigenvalue decay k^-0.6,
1024 dims, 256 rows — the first synthetic, with fast-decaying column scales,
had zero cross-block coupling and showed nothing):

| beta | output error ‖ΔW·X‖/‖W·X‖ | trits changed vs GPTQ |
|---|---|---|
| 0 (plain GPTQ) | 41.523% | — (bit-identical, asserted) |
| 0.0003 (repo default) | 41.349% | 29,890 |
| 0.003 | 40.674% | 44,885 |
| 0.03 | 40.626% | 67,285 |
| 0.1 | 54.459% | 119,944 |
| 0.2 | 56.422% | 125,222 |

The good region is beta ∈ [0.003, 0.03]; beyond it the pull toward the
originals overwhelms the Hessian compensation and the error *rises* past
plain GPTQ. Recipe value: **0.003** (conservative edge of the plateau).
Implementation note: the correction is `((W−W₀) @ U^T) @ U · beta` — two
rows·r² triangular products; forming `U^T U` (r³) would cost more than the
quantization itself at 17k dims.

**Impact selection for the second plane** (`--hot-select impact`,
`FORGE_IMPACT_JSON`): experts ranked by the quantization error the residual
stream actually feels rather than by routing frequency. The full score
trace(H·ΔWᵀΔW) costs 35 TFLOP/layer on the 397B (measured: ~25 min/layer on
CPU); the ranking uses the diagonal weighting ‖ΔW·√diag(H)‖² (QERA-approx's
closed-form), which reduces the cost 1000× and preserves the ordering signal
the selection needs. Ranking runs offline in daylight because the forge
meets `down` before gate/up but needs one hot set per layer.

**TQ1_B160 closed**: GET_ROWS validated GPU==CPU on Vulkan
(test-backend-ops, rows 160·400 and 320·1000), byte-identical against the
numpy reference encoder. The Flash-Next n-gram table path (26.8 GiB IQ4_NL →
~10.1 GiB) is now an engine capability awaiting its model-level evaluation.

**Dense 27B, second verdict.** Full-ternary FFN with true GPTQ: ppl 48.70
(no-GPTQ v1: 54.28) — alive but far past usable; the per-tensor Frobenius
error *rose* (57% vs 43.7% RTN) while ppl fell, which is GPTQ behaving as
designed (H-weighted error traded against flat error), not a defect. v3
(down_proj verbatim from the Q4 donor + FOEM 0.003 on gate/up) is forging.
Literature context gathered today: MoQE measures expert FFNs surviving
2-bit where dense FFNs die (cross-expert redundancy) — our 397B-lives /
27B-dies pair is that measurement's field confirmation — and the
Ternary-Mamba line warns that recurrent-state hybrids (this 27B is
GatedDeltaNet) accumulate FFN error across generation, which post-hoc
per-layer correction cannot cancel.

### ⛔ Correction, same day (12:40): the FOEM sweep above was CONTAMINATED

The morning sweep called the quantizer repeatedly on the same tensor. On CPU
`Tensor.to(device)` returns *the same storage* when dtype and device already
match, and the GPTQ loop compensates **in place** — so every call after the
first started from weights the previous call had already destroyed. (GPU
callers were never affected: `.to("cuda")` always copies. All forged files
stand.) A defensive `.clone()` now guards every chunk load, an
input-unchanged assert is part of the self-tests, and the clean sweep says
the opposite of the dirty one:

| beta | output error (clean) | verdict |
|---|---|---|
| 0 (plain GPTQ) | 35.012% | reference |
| 0.0003 | 35.016% | neutral |
| 0.003 | 35.142% | slightly worse |
| 0.03 | 66.957% | harmful |
| 0.1 | 2880% | catastrophic |

**FOEM does not pay at 1.69 bpw on this synthetic.** The published gains are
at 2-4 bit, where the latent drift is small relative to the grid; at ternary
the rounding is so coarse that the pull toward the originals mostly fights
the Hessian compensation. The flag stays in the forge (beta=0 default,
bit-identical to GPTQ — asserted); the recipe for tonight's 397B v4 drops
FOEM and keeps depth-in-file + impact selection. The 27B v3 forge launched
before this correction runs with beta=0.003 — a ≤0.15% perturbation on the
synthetic, noted for the record rather than worth a restart.

### Impact selection: routing mass is part of the impact (same day, 12:34)

First ranking omitted p_e and produced a top-16 DISJOINT from the frequency
top-16 (0/16): rarely-routed experts with large weights, whose error the
model never feels. With impact_e = p_e·‖ΔW_e·√diag(H)‖² the tail-band
ranking nearly coincides with frequency (layer 44 top-4 identical set) —
independently confirming the earlier finding that frequency selection is
near-optimal in the good band and wrong only in the head.

### FOEM post-mortem: beta is not scale-invariant (the mathematics, 13:15)

The FOEM term is `d ← d·(1 − β·λ)` per application, in the eigenbasis of the
sliced `H⁻¹`: divergence whenever `β·λmax(H⁻¹) > 2`. With 1% mean-diagonal
damping and a rank-deficient sample Hessian, `λmin(H) ≈ 0.01·mean(diag H)`,
so `λmax(H⁻¹) ≈ 100/mean(diag H)` — **the effective β scales inversely with
the activation energy of the layer**, and the paper never normalises it
(its experiments live at 3-4 bit on one activation scale). Measured on the
27B's own Hessians, against the damage the verify had already localised:

| layer | mean diag(H) | β·λmax(H⁻¹) at β=0.003 | prediction | observed |
|---|---|---|---|---|
| 9 | 0.0502 | **5.98** | diverges (>2) | up.9 error 16,649% |
| 32 | 0.4147 | 0.72 | stable | gate.32 error 53.4% |

One number predicts exactly which tensors died. A safe re-enable would
require per-layer `β ≤ 1/λmax(H⁻¹)` (i.e. β normalised by the damping
floor), which is a different method from the paper's — left as a note, not
pursued: the clean sweep shows no gain even where stable.

### FOEM, third and final round: the repo diverges from its own paper; the paper's formula still does not pay at ternary

Investigation (issue #2 on the FOEM repo shows the same NaN failure on
Qwen3-30B-A3B; the author replies "the code in this repo has problems", no
mechanism given): the standalone repo scales the correction by **H⁻¹**
(`(W−W₀)·HinvᵀHinv·β`) — but the paper's Eq. 19 derives that H and H⁻¹
*cancel*, leaving plain `−β(W−W₀)`, which is what the GPTQModel integration
implements (β∈[0.1,0.25], 5% damping). Our divergence measurement
(β·λmax(H⁻¹)=5.98 → 16,649% error) therefore diagnoses **a bug in the
deprecated repo**, previously unexplained. Re-measured with the paper's
correct formula on the clean synthetic:

| beta (Eq. 19) | output error | vs plain GPTQ 35.012% |
|---|---|---|
| 0.05 | 35.058% | worse |
| 0.1 | 35.190% | worse |
| 0.2 | 35.718% | worse |
| 0.25 | 36.056% | worse |

Monotonically harmful across the paper's own recommended range. The negative
result stands on the correct formula: at 1.69 bpw the drift FOEM corrects is
not the binding error source, and pulling latents toward the originals only
discards Hessian compensation. Paper section material: one primitive, two
implementations, one diverges with a one-number predictor, the correct one
is measurably non-beneficial in the ternary regime the paper never tested.

### Dense 27B, v3b verdict: ppl 19.63 — alive

gate+up ternary (plain GPTQ) + down_proj verbatim Q4 donor: WikiText ppl
**19.63** against 48.70 (all-ternary GPTQ) and 54.28 (all-ternary RTN).
The single flag `--ternary-parts gate,up` recovered 29 points — the
down-projection bottleneck (D²Quant) and super-weight (2411.07191)
literature, confirmed on a hybrid dense at 27B. Next: QERA-approx low-rank
correction (closed-form, LoRA-GGUF servable) and the 73-fixture functional
bench.
