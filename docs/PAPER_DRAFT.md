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

## Evaluation debt (must clear before submission)

- lm-eval via llama-cpp-python server (llama-server's `echo` gap documented),
  version pinned, `acc` and `acc_norm` both, full task sets.
- KL-divergence + top-token agreement against a Q8 proxy reference on the
  35B (fp16-reference validation of the method at 35B scale).
- The dense-model experiment: does the 73% rule survive without experts?
