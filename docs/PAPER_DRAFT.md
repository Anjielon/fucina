# Depth, Not Fidelity: What Decides Whether a Ternary Correction Helps a Mixture-of-Experts

*Draft for TMLR. Working title. Every number in this draft traces to
[RESULTS](RESULTS.md); every claim of novelty traces to a deliberate
literature sweep recorded in [LESSONS](LESSONS.md).*

## Abstract (draft)

Post-training ternary quantization (1.69 bits/weight) makes very large
mixture-of-experts models runnable on consumer unified-memory hardware — we
quantize a 397B-parameter, 512-expert MoE to 84 GiB and a 35B sibling to
9.2 GiB, both served by a modified llama.cpp with Vulkan ternary kernels. We
then study a *corrective* mechanism: a second ternary plane on the hottest
experts, which halves weight reconstruction error (46% → 22%) and halves the
single-expert output error under real activations. **Applied uniformly, the
correction makes both models worse.** We localise the damage with
per-layer, per-projection switches and find that everything is decided by
*where* the correction lands, not by *how faithful* it is: the identical
perturbation, delivered at identical relative magnitude to every layer's
output, improves the model in the final quarter of the depth and is
catastrophic near the middle (a single layer moves perplexity from 8.72 to
37.8; the same correction at the last layers *improves* it). The fraction of
depth at which the sign flips is the same — 73% — on both models, at a 10×
size difference. We refute eight candidate explanations with pre-registered
predictions — correction magnitude, routing concentration, expert-selection
quality, input kurtosis, effective rank, permutation errors, delivered
output error, and route-flip counts, the last of which *anti-correlates*
with damage. The beneficial configuration is confirmed by a paired
logit-level comparison (mean Δp = +0.048% ± 0.014%), ships as a file rather
than a runtime flag, and improves every evaluation chunk. Alongside, we
report the first measured *distribution* of the MoE top-k routing margin:
on 256 experts, the median gap between the 8th and 9th expert is 10× narrower
than a uniform share, and 2.95% of tokens sit within one bf16 ULP of a
routing flip — quantifying an assumption that stability lemmas and
determinism patches currently take on faith.

## Contributions

1. **A depth rule, measured twice.** A partial ternary correction helps only
   in the final ~quarter of the network, with the sign flipping at the same
   relative depth (73%) on two MoE models 10× apart in size. The recipe that
   follows — correction baked into the file for the tail only — improves the
   35B by 5.4% and the 397B measurably (paired, 3.4σ), with no functional
   regression on verifiable-answer probes.
2. **Eight pre-registered refutations.** Every local account of the damage is
   excluded by measurement, including the decisive one: the perturbation
   *delivered to the layer's output*, ‖ΔW·X‖/‖W X‖ under that layer's own
   captured activations, is constant across depth (1.000-1.002 ratio to the
   weight-space figure) and slightly *smaller* where damage is greatest
   (ρ = −0.599). Route-flip counts nearly invert the damage ordering
   (all-layers: 5× the flips of the worst single layer, 3.5× less damage).
3. **The routing-margin distribution.** Defined twice in prior work (ReMoE's
   γ in probability space; a logit-space "router margin" reduced to one AUC)
   and never measured: we give percentiles. Median 10.1× narrower than the
   uniform expert share; 1st percentile 806× narrower; 2.95% of tokens
   within one bf16 ULP. The distribution is skewed (mean/median 1.76), which
   is precisely why the published token-averaged mean misleads.
4. **An engineering result.** TQ1_0 ternary kernels for Vulkan (upstream PR),
   a forge that produces two-plane ternary MoE files from bf16 on one
   consumer GPU (7.6 h for 397B), a strip tool that bakes measured layer
   profiles into files, and — new with this paper — a dense-model forge
   whose first target tests whether the depth rule needs experts at all.

## What we explicitly do not claim

- Not "more bits hurt": the claim is *heterogeneous* fidelity hurts, and the
  literature contains the same phenomenon unremarked (GEMQ Table 1: expert-
  level mixed precision losing to uniform at matched bits; HiFloat4:
  restoring one component to high precision scoring below uniform FP4).
- Not the concept of the routing margin (ReMoE, arXiv 2605.27081, defines
  it), nor the down-projection bottleneck (D²Quant calls it "well-known"),
  nor non-additivity of compression error (EvoPress, ICML 2025, opens on
  it). Our claims are the measurements, the localisation, and the rule.
- Not a mechanism. Eleven hypotheses died; the depth rule and the
  representational-transition literature (five methods placing a boundary at
  0.47-0.53 of depth; our worst layer sits at 0.488) are consistent, but
  consistency is not identification. We state the open problem as narrowly
  as the measurements allow.

## Structure (planned sections, with source material)

1. Introduction — the deployment setting; the correction paradox in one
   figure (fidelity better at every level, model worse).
2. Setup — models, forge, engine switches; evaluation protocol and its
   honesty caveats (llama.cpp scorer ≠ lm-eval; token- vs char-normalised;
   n=300 power analysis). [RESULTS §Standardised benchmarks]
3. The paradox and the per-layer anatomy. [RESULTS §Expert routing →
   §one projection does all of it]
4. Eight refutations, each with its pre-registered prediction. [RESULTS
   tables; LESSONS for the two that were "confirmed" then killed]
5. The depth rule and the recipe-in-the-file; paired confirmation.
6. The routing margin distribution. [RESULTS §The distribution]
7. Related work — positioned against Tied Trit-Planes (arXiv 2608.08910,
   same primitive, no per-layer analysis by design), PTQTP, EvoPress,
   D²Quant, GEMQ, the GLU activation-spike line (which places damage at the
   *edges* — the tension is stated, not hidden), and sglang #35916.
8. Limitations — two models, one family; perplexity-first evaluation with
   functional probes only; selection-criterion confound quantified but not
   yet causally closed (the Hessian-selection reforge is future work);
   the residual-stream rank correlation that failed to replicate across
   models, reported as a cautionary tale.

## Section 1 draft — Introduction

A 397-billion-parameter mixture-of-experts model does not fit in the memory
of any consumer machine at the precisions its authors shipped. At 1.69
bits/weight it does: 84 GiB, resident in the unified memory of a desktop
APU, served by ternary Vulkan kernels we contribute upstream. This paper is
about what happened when we tried to make that model *better* — and measured,
twice, that the intuitive way to do so makes it worse.

The corrective mechanism is a second ternary plane on the most-routed
experts: a residual quantization that demonstrably halves weight
reconstruction error (46% → 22%) and halves the single-expert output error
under the layer's own captured activations. Every local metric of fidelity
improves. Applied uniformly across the network, perplexity degrades — on
both a 35B and a 397B model of the same family. This is the correction
paradox, and it is not a bug: we verified the file layout, the expert
permutations (28/28 correct on all 40 layers), the scales, and the delivered
error, and then spent the balance of the work localising *where* the paradox
lives.

The answer is depth. Per-layer engine switches (enabling the correction for
arbitrary layer ranges at load time, without re-forging) show that the same
perturbation, delivered at the same relative magnitude to every layer's
output, improves the model in the final quarter of the network and is
catastrophic near the middle: one single mid-depth layer moves perplexity
from 8.72 to 37.8, while the correction restricted to the tail improves on
the uncorrected baseline. The fraction of depth at which the sign flips is
the same on both models — 73% — at a 10× parameter difference. Eight
candidate explanations, each pre-registered as a falsifiable prediction
before its measurement ran, are refuted in Section 4; what survives is a
statement about position, not fidelity. Section 5 turns the statement into a
recipe that ships inside the file. Section 6 reports a measurement the
routing-stability literature has so far assumed: the distribution of the
top-k routing margin, whose skew explains why the one published mean curve
misleads.

We write from a deployment setting the quantization literature rarely
occupies — one machine, no cluster, no fp16 reference for the largest model
— and we are explicit throughout about what that setting does to
measurement: paired estimators where absolute error bars would drown the
effect, functional probes where benchmark harnesses disagree with each
other, and a stated evaluation debt where neither suffices.

## Section 5 draft — The depth rule and the recipe in the file

**The rule.** On both models, the second plane helps if and only if it is
restricted to the final ~quarter of the depth. On the 35B (41 layers,
counted 0-40) the beneficial band is 30-39: 73.2% of depth. On the 397B (60
MoE layers with routed experts, band 44-59): 73.3%. Between the bands and
the mountains sits no gradual transition on the 35B — layer 20 (0.488 of
depth) alone accounts for most of the uniform-application damage — while
the 397B decays monotonically from the head (catastrophic, 2/28 selection
overlap) through the middle (harmful) to the tail (beneficial). The
coincidence of the two flip points at 73% of depth, across a 10×
size difference and two different routing topologies, is the paper's
central empirical fact. We claim the measurement, not a law: two models,
one family (Section 8).

**The recipe.** Because the engine switches that made the anatomy measurable
are ours, we could ship the rule as a runtime flag — and explicitly do not.
A flag is a claim about every future file; a file is a claim about itself.
The forge takes `--plane2-layers LO-HI` and bakes the tail-only correction
in: plane-2 tensors exist only for the beneficial band, the expert
permutation and router reordering are applied only where the plane exists
(one predicate feeds all three coupled decisions), and a strip tool
retrofits the same profile onto already-forged files, reclaiming the dead
plane-2 bytes (4.1 GiB on the 397B). The resulting file runs on the
unmodified engine at full speed with no configuration.

**Confirmation, 35B.** Tail-only (30-39) perplexity 8.2501 against 8.72
uncorrected — a 5.4% improvement — where uniform application had *degraded*
the same model. Functional probes (verifiable-answer arithmetic and
string-manipulation set) unchanged or improved; no regression observed.

**Confirmation, 397B, paired.** At 397B the absolute error bar at 30 chunks
(±0.087) is larger than the expected effect, so the confirmation is paired
(`--kl-divergence-base`, same chunks, reference logits stored): mean
Δp = +0.048% ± 0.014% (3.4σ) for tail-only over uncorrected, improvement in
every evaluation chunk, and the verifiable-answer probes intact. The gain is
small and real; the point is its *sign*, which uniform application gets
wrong at 10× the magnitude.

**Cost.** The tail-only file is strictly smaller than the uniform one (the
correction exists on 16/60 rather than 40/60 layer-equivalents), loads
nothing extra at the head where it would do harm, and the depth rule
transfers across the two models tested at zero search cost — against
EvoPress-style whole-model search, which subsumes the selection problem but
must re-run per model.

## Section 6 draft — The routing-margin distribution

**Definition.** For a token routed to k of E experts, the margin is the gap
between the k-th and (k+1)-th router score — the distance to the nearest
routing flip. ReMoE defines it in probability space as the premise of a
stability lemma; arXiv 2608.11212 defines it in logit space and collapses it
to a single AUC. Neither reports its distribution; arXiv 2602.02443 reports
a token-averaged mean over depth. We measure the distribution on the 397B's
captured router probabilities (16,640 tokens × 60 layers, the same capture
that feeds the forge's Hessians).

**The distribution.** The median margin is 10.1× narrower than the uniform
share 1/E; the 1st percentile is 806× narrower; 2.95% of tokens sit within
one bf16 ULP of the 8th/9th boundary — for these, the routing decision is
below the numerical noise floor of the arithmetic that computes it. The
distribution is right-skewed, mean/median 1.76: the mean curve published in
prior work sits 76% above the typical token, which is precisely the
regime where a mean reassures and a median warns.

**What follows and what does not.** The qualitative half of this measurement
was asserted, on synthetic weights, by the author of sglang PR #35916 —
batch-invariant routing motivated by adjacent scores within one ULP on this
very topology — who lacked hardware for the real model; this section
supplies the measurement. What does *not* follow is the tempting link to
Section 4's damage profile: route-flip counts anti-correlate with damage
(the all-layers configuration flips 5× more routes than the worst single
layer at 3.5× less damage), and the one published flip classifier sits at
chance (AUC 0.490). Fragile margins are a fact of the router; they are not,
by count alone, the mechanism of the correction paradox.

## Section 2 draft — Setup and measurement honesty

**Hardware.** Everything runs on one consumer machine: Ryzen AI Max+ 395,
128 GB unified memory (~96 GiB exposed as Vulkan VRAM), Fedora, our llama.cpp
fork with TQ1_0 Vulkan kernels (scalar, coopmat1 and coopmat2 paths; the
coopmat2 decode functions were later validated on an RTX 4060, the first
hardware to exercise them). A second consumer laptop (RTX 4060 Laptop 8 GB)
serves as an auxiliary bench for the 35B model.

**Models and forge.** The 397B (512 experts × 60 layers, hybrid GatedDeltaNet
attention) is forged from the bf16 checkpoint: expert tensors ternarized with
a two-level optimal-scale search plus true GPTQ against per-layer input
Hessians (16,640 samples each); non-expert tensors from an existing shelf
quantization. 7.6 h end-to-end. The 35B sibling is forged the same way from
a shelf GGUF. Engine switches allow enabling the second plane per layer range
and per projection at load time, which is what makes the per-layer anatomy
measurable without re-forging.

**What our perplexity numbers are, and are not.** All perplexity is
llama.cpp's, wikitext-2, ctx 2048, chunk counts stated per table. Absolute
error bars at 30 chunks are ±0.087, so any comparison inside that band is
resolved with the *paired* estimator (`--kl-divergence-base`): same chunks,
reference logits stored, uncertainty propagated from the logits. The headline
confirmation (tail-only correction on the 397B) is paired: mean
Δp = +0.048% ± 0.014%.

**What our task numbers are, and are not.** llama.cpp's built-in scorers
label their output `acc_norm` but normalise endings by **token count**;
lm-eval's `acc_norm` normalises by **character length**, and for Winogrande
and MMLU lm-eval has no `acc_norm` at all. Our HellaSwag/Winogrande figures
are therefore a third quantity, comparable only within this paper. At n=300
their resolvable difference between two builds is 6.9 pp against typical
quantization effects of 1-4 pp: we use them solely as "the model is not
broken" evidence and never as a measurement of quantization damage. The
final evaluation (pinned lm-eval via a server that implements prompt-token
logprobs; `acc` and `acc_norm` both; full task sets) is listed in the debt
section, and we explicitly do not use IRT-subsampled minibenchmarks: the
published gp-IRT weights put only 12-30% of the reported figure on responses
actually observed, calibrated on a 2024 pool of full-precision models that
excludes the near-chance region a sub-2-bit model can occupy — a simulation
on the published parameters shrinks true damage in the flattering direction.

**Pre-registration discipline.** Every hypothesis in Section 4 was written as
a falsifiable prediction before its measurement ran, in the project log that
ships with the artefact. Two hypotheses were "confirmed" by a first
measurement and killed by a second (residual-stream rank; selection quality);
both stages are kept in the record.

## Section 4 draft — Eight refutations, each with its pre-registered prediction

The method note that carries the section: every hypothesis was written down as
a falsifiable prediction *before* its measurement ran, and two of the eight
were "confirmed" by a first measurement and then killed by a second. We keep
both stages in the record, because the difference between a correlation and a
mechanism is precisely what this section is about.

| # | hypothesis | pre-registered prediction | measurement | verdict |
|---|---|---|---|---|
| 1 | Correction magnitude | ‖ΔW‖/‖W‖ larger where damage is larger | 0.412–0.423 at every layer (2% spread); ρ = −0.56 | refuted |
| 2 | Routing concentration | hot-28 traffic share lower where damage is larger | ρ = +0.013; worst and best layers within 1.1 points of share | refuted |
| 3 | Scale mismatch (expert level) | cold-expert compensation helps wherever plane-2 is on | helps `down`-only (−0.14), harms all-three (+0.70) | refuted |
| 4 | Scale mismatch (layer level) | compensation helps a lot on the single worst layer, little on all layers | 37.80 → 37.67 (0.5% of the damage); *harms* the uniform case | refuted |
| 5 | Input pathology (kurtosis / effective rank of the `down` input) | statistic peaks at the damaged layers | kurtosis ρ = −0.19; effective rank ρ = +0.28; layers 10 and 15 nearly identical rank, 2.3× different damage | refuted |
| 6 | Hot-first permutation error | wrong expert order at damaged layers | 28/28 correct on all 40 layers, no exceptions | refuted |
| 7 | Delivered output error | ‖ΔW·X‖/‖WX‖ larger where damage is larger | ratio to weight-space error 1.000–1.002 at every layer; ρ = −0.599 | refuted — the decisive one |
| 8 | Expert-selection quality (frequency vs impact) | bands with near-perfect selection must benefit from the correction | 397B band 30-45: overlap 25-27/28, best in the model — and it *harms* (6.3002 vs 6.1845) | refuted in causal form |

Two cautionary subplots, reported rather than smoothed away:

- **The residual-stream rank correlation that did not replicate.** On the 35B,
  effective rank of the residual stream correlates with damage at +0.651 —
  publishable-looking. On the 397B the sign *reverses* (damage highest where
  rank is lowest). One model would have made this a mechanism; the second
  model made it a coincidence. We keep it as the section's epigraph.
- **The selection hypothesis died twice.** Its correlational form is strong
  (impact lost by frequency-selection tracks the 397B damage profile at
  +0.788, and the head — 2/28 correct, 89.5% of impact lost — is exactly the
  catastrophic band). Its causal form failed the pre-registered test above.
  Selection quality is a real, quantified property of the forge and it does
  not decide the sign. Both statements are true; conflating them is how the
  literature acquires mechanisms it does not have.

What survives all eight is a statement about *position*: an identical
perturbation, delivered identically, has consequences that depend only on the
depth at which it lands — beneficial in the final quarter (from 73% of depth
on both models), catastrophic near the middle of the 35B (its worst layer sits
at 0.488 of depth, on the representational boundary that five independent
methods place at 0.47–0.53), and progressively less harmful with depth on the
397B. Route-flip *counts* do not order the damage (they nearly invert it), so
if routing mediates the effect it does so through *which* routes change, not
how many — consistent with the one published attempt to classify flips, which
achieves chance (AUC 0.490) at telling harmful from benign.

## Section 7 draft — Related work

**Ternary post-training quantization.** PTQTP (arXiv 2509.16989) introduces
the dual trit-plane decomposition we build on, applied uniformly, with no
per-layer analysis. Tied Trit-Planes (arXiv 2608.08910) is the closest work:
two ternary planes on a disk-streamed MoE, same primitive, same deployment
constraint — and it reports, without explanation, the same proxy/fidelity
dissociation we measure (its higher-reconstruction-error variant scores
*better* on MMLU). Its ladder is over tensor classes and, by its own
statement, "localizes sensitivity to the final cumulative bundle, not to a
single component"; per-layer localisation is precisely what we add. BitNet
(arXiv 2402.17764) trains ternary from scratch and keeps embeddings
high-precision; our setting is post-hoc, on weights never meant for ternary.

**Non-additivity of compression error.** EvoPress (ICML 2025, arXiv
2410.14649) opens on the observation that per-layer independence fails and
that removing *more* blocks can help; its search machinery (fitness = whole-
model KL) subsumes our selection procedure. What it explicitly does not do is
isolate single catastrophic layers or offer a mechanism. QDrop (ICLR 2022)
and Quant-Noise (ICLR 2021) establish the non-monotone shape at calibration
time — full quantization-aware noise *worse* than none, partial better than
both — which is the same shape as our cancellation, in a different regime.

**Mixed precision within MoE.** The expert-wise bit-allocation literature
(MoPEQ, AlphaQ, MC-MoE, GEMQ, BitsMoE) is uniformly monotone in its
assumptions: the only failure mode contemplated is mis-estimating importance.
The phenomenon we isolate is visible, unremarked, in GEMQ's own Table 1
(expert-level mixed precision losing to uniform at matched bits on
Qwen3-30B-A3B, −9.89 acc at 2.0 bpe) and named in HiFloat4 (arXiv 2607.26515:
restoring one component to higher precision scores *below* uniform FP4).
AWQ's Table 1 shows the selection-criterion analogue: protecting the wrong 1%
is worse than protecting nothing. Our framing — heterogeneous fidelity, not
insufficient fidelity — is the synthesis these scattered rows point to.

**Routing fragility.** The margin at the top-k boundary is defined twice and
measured never: ReMoE (arXiv 2605.27081, App. C.2) as the unmeasured premise
of a stability lemma, in probability space; arXiv 2608.11212 in logit space,
collapsed to one AUC (and its margin→harmful classifier sits at chance,
0.490 — consistent with our flip-count anti-correlation). A token-averaged
mean curve exists (arXiv 2602.02443); the distribution does not, and the
mean-vs-median gap (1.76×) is exactly why it matters. sglang PR #35916
asserts the qualitative half — adjacent scores within one bf16 ULP on this
very topology — on synthetic weights, its author lacking hardware for the
real measurement; we supply it.

**Depth structure.** The representational-transition literature places a
boundary at 0.47–0.53 of depth by five independent methods (semantic hub,
activation patching — concept written at exactly 0.500 — probing centers of
gravity, and their encoder controls); our worst layer sits at 0.488. The
intervention literature, by contrast, is dominated by edge-sensitivity: GLU
activation spikes at early *and late* layers (arXiv 2405.14428), llama.cpp's
own extra-bit heuristic on the first and last eighths, and layer-pruning work
finding deep layers most redundant. The tension is real and stated: those
works measure activation magnitude or deletion tolerance; we measure
*corruption* tolerance, and deletion-vs-corruption is exactly the distinction
the reconciliation requires (a deleted layer passes the residual through; a
corrupted one writes an actively wrong signal where the representation is
being formed). One prior intervention study reports a mid-depth sensitivity
peak (arXiv 2511.17194, activation injection, mean peak 54.9%).

**Down-projection difficulty.** Known and named: D²Quant (arXiv 2602.02546)
calls it "a well-known quantization bottleneck" and engineers a dedicated
dual-scale quantizer. Community evidence on our very model family: unsloth's
v3 recipe placing 20 large FFN tensors at 2 bits measurably degraded coding.
What is not in the literature is the sign flip — `down` catastrophic on the
mountain, *helpful* in the tail — nor any derivation from gated-activation
statistics (our attempt at one is refuted by measurement: kurtosis, effective
rank and delivered error all fail to track the damage).

## Evaluation debt (must clear before submission)

- lm-eval via llama-cpp-python server (llama-server's `echo` gap documented),
  version pinned, `acc` and `acc_norm` both, full task sets.
- KL-divergence + top-token agreement against a Q8 proxy reference on the
  35B (fp16-reference validation of the method at 35B scale).
- The dense-model experiment: does the 73% rule survive without experts?
