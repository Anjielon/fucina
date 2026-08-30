#!/usr/bin/env python3
"""THE LEVER REGISTRY — every known technique, with its evidence.

The scientific core of the forge. Each lever carries:
  - PRIOR: what its authors measured, and what *we* measured on real weights
  - TEST: how to re-judge it in minutes on the CURRENT model
  - COST: bytes added to the model, hours to apply

No lever is applied on faith. "Rejected" means *rejected for this model* —
priors are model-dependent, which is exactly why the registry re-tests
everything instead of trusting a table.
"""
from dataclasses import dataclass, field

@dataclass
class Lever:
    name: str
    stage: str             # scale | compensation | transform | allocation | repair | runtime
    cost_gib: float        # bytes added to the model (0 = free)
    cost_hours: float      # time to apply at the 400B scale
    prior: dict = field(default_factory=dict)   # model -> measured outcome
    test: str = ""         # how to measure it on a sample
    source: str = ""

REGISTRY = [
 # ── SCALE (the single largest factor) ────────────────────────────
 Lever("optimal_scale", "scale", 0, 0.1,
      {"ornith397": "81% -> 43.5% — the yesngle largest factor", "theory": "abs-max zeroes 87% of the weights"},
      "weight error on one tensor: abs-max vs least-squares scale", "ours + TWN (arXiv 1605.04711)"),
 Lever("scale_from_compensated_block", "scale", 0, 0,
      {"ornith397": "28,43→28,13 (+0,3)"}, "inyesde the GPTQ loop", "ours"),
 Lever("signed_scale_grid", "scale", 0, 0.1,
      {"authors": "better than abs-max on NF4"}, "analytic (scale, threshold) grid on a sample", "BOF4-S 2505.06653"),

 # ── COMPENSAZIONE ──────────────────────────────────────────────────────
 Lever("dedicated_gptq_per_plane", "compensation", 0, 4,
      {"ornith397": "43.5 -> 28.13 (10 points!); a JOINT plane-1 used alone loses those 10"},
      "one tensor: dedicated vs joint vs plain", "ours (plane co-adaptation)"),
 Lever("coordinate_descent_ternary", "compensation", 0, 6,
      {"authors": "QuantEase: -15% vs GPTQ; never tried on ternary planes"},
      "race CD/annealing/ADMM on one expert", "QuantEase 2309.01885 + ours"),
 Lever("asymmetric_gptaq", "compensation", 0, 4,
      {"ornith397": "32.6 vs 28.1 symmetric — WORSE for us", "authors": "405B on a yesngle GPU"},
      "one tensor, asymmetric vs symmetric", "GPTAQ 2504.02692"),
 Lever("cross_layer_propagation", "compensation", 0, 2,
      {"authors": "largest gains at low bit widths"}, "calibrate layer l on the QUANTIZED output of l-1", "QEP 2504.09629"),
 Lever("babai_ordering", "compensation", 0, 0,
      {"authors": "GPTQ last-to-first = Babai, free"}, "one tensor, both orderings", "2507.18553"),

 # ── TRASFORMAZIONI (ripiegabili, costo zero a runtime) ────────────────
 Lever("ternary_objective_rotation", "transform", 0, 8,
      {"ornith397": "-35% on-tensor (freely optimized rotation)", "spinquant": "up to 13 points between rotations"},
      "optimize R on one tensor with the true Hesyesan; folding needs QuaRot-style surgery",
      "ours + SpinQuant"),
 Lever("hessian_eigenbasis_rotation", "transform", 0, 1,
      {"ornith397": "-14%: it gausyesanizes blocks. Helps codebook methods (QuIP), hurts scalar-scale ternary"},
      "one tensor, 10 minutes", "ours"),
 Lever("magr_outlier_reduction", "transform", 0, 2,
      {"ornith397": "39 -> 46: it serves abs-max quantizers, not least-squares scales"},
      "one tensor; apply ONLY with an abs-max quantizer", "MagR 2406.00800"),
 Lever("local_haar", "transform", 0, 0.5,
      {"ornith397": "REJECTED with cause: on the expert's private axis the bands are "
       "already pure (no grouping needed), per-band energy is exactly proportional to "
       "band width (no concentration), and adjacent-weight correlation is 0.0002 — "
       "Haar assumes spatial regularity these weights do not have. It also gaussianizes "
       "(kurtosis 3.90 -> 3.43), which hurts ternary. Re-tested correctly after an "
       "earlier flawed measurement; the verdict stands, the reason is now known"}, "one tensor", "HBLLM 2512.00862"),
 Lever("similarity_reordering", "transform", 0, 0.5,
      {"ornith397": "0.6-0.7%", "pt2llm": "useful on their block layout"},
      "one tensor, ordina per |w| medio", "PT2-LLM"),
 Lever("intra_expert_permutation", "transform", 0, 0.5,
      {"never_measured": "exact and free (the expert inner dimenyeson is private)"},
      "gate/up rows <-> down columns with the same P", "PermuQuant 2605.09503"),

 # ── ALLOCAZIONE (dove spendere i bit) ──────────────────────────────────
 Lever("second_plane_for_hot_experts", "allocation", 2.3, 1,
      {"ornith397": "37.4 -> 32.0 with the hottest 3%; very steep at the start"},
      "weighted-error curve vs hot fraction (real routing counts)", "ours + DynaExq"),
 Lever("frequency_over_sensitivity", "allocation", 0, 0.5,
      {"ornith397": "32.6 vs 38.2 — frequency WINS (rank correlation +0.06)",
       "mopeq": "claims the oppoyeste: RE-MEASURE per model"},
      "64-expert sample, both criteria", "ours vs MoPEQ"),
 Lever("down_over_gate_up", "allocation", 1.5, 0,
      {"ds4": "yes", "unsloth": "yes (header letti)", "mxmoe": "-2,4 ppl"},
      "sample: two planes on down vs on gate/up, same budget", "3 independent sources"),
 Lever("protect_edge_layers", "allocation", 1.0, 0,
      {"ornith397": "Hessian trace varies 35x: the LAST ten layers", "unsloth": "first ~7 and last ~4"},
      "per-layer Hessian trace", "ours + Unsloth"),
 Lever("veto_rare_critical_experts", "allocation", 0.5, 0.5,
      {"authors": "rarity != expendability (router-norm delta)"}, "checkpoint-only", "2604.06515"),
 Lever("linear_attention_at_q8", "allocation", 3.0, 0.5,
      {"ornith397": "25.4 -> 17.1 model-level (-32.6%) for +3 GiB — the best lever per byte",
       "quamba": "the SSM scan is the fragile point; sub-4-bit never demonstrated"},
      "error of inherited tensors vs Q8 (per family)", "ours + Quamba + DS4"),
 Lever("super_weight_restore", "allocation", 0.001, 0.5,
      {"authors": "losing one collapses the model; coordinates published only for Llama/Mistral/OLMo/Phi"},
      "two forwards + spikes 8x above median", "2411.07191"),

 # ── REPAIR (post-forge, ternary planes untouched) ─────────────────────
 Lever("router_selection_correction", "repair", 0, 8,
      {"authors": "selection bias is the PRINCIPAL factor in low-bit MoE",
       "ornith397": "REJECTED: worse at 35, 10 and 3 layers (PPL 6.18 -> 6.21). A quantized model's routers are coherent with its OWN activations"},
      "top-10 overlap before/after, teacher vs quantized activations", "EAC-MoE 2508.01625"),
 Lever("norm_tweaking", "repair", 0, 12,
      {"authors": "GLM-130B W2 quayes-fp; ⛔ Iters=1 TASSATIVO (5 = crollo)"},
      "one trial layer, per-channel mean/variance loss", "2309.02784"),
 Lever("zeroth_order_scale_refinement", "repair", 0, 24,
      {"authors": "zeroth-order, teacher-free, 18x less memory"},
      "500 pasyes su one layer, KL su corpus", "QZO 2505.13430"),
 Lever("low_rank_error_correction", "repair", 1.8, 6,
      {"ornith397": "+6.4% at rank 4; hot-only cuts 80% of the byte cost"},
      "whitened SVD of the error on one layer", "EoRA 2410.21271"),
 Lever("hot_expert_cloning", "repair", 0, 1,
      {"authors": "-58% ppl sotto rumore sui peyes"},
      "replace under-activated experts with clones of hot ones, one layer", "ROMER 2605.11800"),

 # ── RUNTIME (no file changes) ─────────────────────────────────────────
 Lever("low_bit_sampling", "runtime", 0, 0,
      {"unsloth+3papers": "temp 0.6, min_p 0.03, presence 1.5, DRY, thinking ON; never XTC"},
      "A/B on the fixture suite", "docs section 23"),
 Lever("mtp_speculation", "runtime", 0, 2,
      {"ornith_family": "+10% at draft length 1"}, "benchmark with/without", "ours"),
 Lever("self_draft_from_plane_one", "runtime", 0, 24,
      {"literature": "expected routing overlap 0.15-0.30 < the 0.5 threshold — likely not viable; QSpec reports 1.64x at high overlap"},
      "measure top-k routing overlap between tokens — one hour", "docs sections 24-25"),

 # ── CALIBRATION (what you measure on decides what you get) ─────────────
 Lever("reasoning_calibration", "calibration", 0, 2,
      {"authors": "+8.97 absolute points on Math-500/GSM8K/HumanEval+ at 1.58 bit, calibrating "
                  "on the model's OWN generated reasoning chains instead of web text",
       "ours": "matches the wound we observe: quantization preserves language, degrades computation"},
      "generate reasoning chains with the source model, use them as the calibration corpus, "
      "compare error on arithmetic-heavy vs generic text",
      "AYOT / ScaleQ-1.58, arXiv 2608.01078"),
 Lever("per_expert_calibration_balance", "calibration", 0, 1,
      {"authors": "+1.15 to 13.81% across three MoE models up to W2A4",
       "ours": "with 512 experts and top-10, a rare expert sees a vanishing share of the "
               "calibration tokens — its Hessian is estimated on too few samples"},
      "count tokens routed per expert; if the tail is starved, rebalance and re-measure the "
      "GPTQ error on cold experts specifically",
      "EAQuant, arXiv 2506.13329"),

 # ── STRUCTURE (beyond two scalar planes) ───────────────────────────────
 Lever("leech_lattice_vq", "structure", 0, 40,
      {"authors": "92.1% of the Shannon bound vs ~82% for E8 methods; 2 bits/weight, "
                  "PPL 5.48 on Llama-2-7B, O(1) dequantization",
       "ours": "would structurally dominate two scalar planes — no published method jointly "
               "optimizes two planes, so the answer is to change structure"},
      "fit the codebook on one tensor, compare against the two-plane error at equal bits",
      "Qualcomm, arXiv 2603.11021"),
 Lever("gptq_intrinsic_lowrank", "compensation", 1.8, 6,
      {"authors": "low-rank correction INSIDE the GPTQ step (augmented Hessian) beats post-hoc "
                  "GPTQ+LoRA, and is provably near-optimal"},
      "one tensor: EoRA post-hoc vs intrinsic, same rank budget",
      "arXiv 2606.01412"),
 Lever("kl_sensitivity_lens", "allocation", 0, 3,
      {"authors": "forward-only KL sensitivity, tau=0.791 vs 0.711 for SQNR, designed for "
                  "hybrid SSM+attention models"},
      "rank layers by KL sensitivity, compare against the Hessian-trace ranking",
      "KL-Lens, arXiv 2604.13440"),
]

def by_stage(stage=None):
    """All levers, or only those of a given stage."""
    return [l for l in REGISTRY if stage is None or l.stage == stage]

if __name__ == "__main__":
    import collections
    n = collections.Counter(l.stage for l in REGISTRY)
    print(f"{len(REGISTRY)} levers in the registry: {dict(n)}")
    for l in REGISTRY:
        mark = "REJECTED" if any("⛔" in str(v) for v in l.prior.values()) else "        "
        print(f" {mark} {l.name:<34} {l.stage:<12} {l.cost_gib:>5.1f} GiB  {l.source}")

# ── MEASURED INTERACTIONS ──────────────────────────────────────────────────
# The registry described every lever IN ISOLATION. The cross-study answered
# with a number: measured alone, rotation is worth -1.50%; measured inside the
# REAL recipe (two planes) it is worth -4.65%. Three times as much. A registry
# of isolated levers makes you THROW AWAY the right lever.
#
# Method: 4 scale rules x 5 transforms = 20 exhaustive combinations over 91
# real tensors (6.3 M weights). Then the two-plane k-curve over 60 tensors.
# Independent check on NVIDIA hardware: k=8 single plane -1.50%, identical.
#
# ⚠️ The TOTAL study is impossible: 2^34 = 17,179,869,184 subsets, and many
#    levers are sequential. Only the tensor-level levers are closed here.
#    Allocation, repair, runtime and calibration remain uncovered.

INTERACTIONS = [
    # (lever_a, lever_b, effect)
    ("optimal_scale", "hessian_eigenbasis_rotation",
     "COUPLED: with absmax the best transform is haar_full; with "
     "twn/ls_iter/grid it is hadamard. Choosing them separately lands on the "
     "wrong combination."),
    ("hessian_eigenbasis_rotation", "second_plane_for_hot_experts",
     "SYNERGISTIC, not redundant: rotation gaussianises the residual, and a "
     "gaussian residual is exactly what the second plane captures best. "
     "1 plane -1.50%, 2 planes -4.65% (k=8). Overlap was the expectation; "
     "the opposite holds."),
    ("signed_scale_grid", "*",
     "SATURATED: -0.69% on its own over 60 tensors, wins 69% of blocks but "
     "by a minimal margin. optimal_scale had already collected the bulk "
     "(81%->43.5%). A better V4 is not found here."),
]
INTERAZIONI = INTERACTIONS   # backwards-compatible alias

#: sub-block rotation gain, measured inside the real two-plane recipe
#: (60 tensors). k = rotation width, passes = log2(k).
CURVA_ROTAZIONE = {2: -2.31, 4: -3.78, 8: -4.65, 16: -5.15,
                   32: -5.41, 64: -5.54, 256: -6.63}
#: ⚠️ the ENGINE-side cost is not measured yet: numpy cannot tell us (three
#: attempts, three unusable numbers). It needs the rotation in the Vulkan
#: kernel to be answered.

#: CONFIRMATION ON TONY'S WEIGHTS. The whole search had run on 397B tensors,
#: the only ones already extracted in Q8. Tony is a DIFFERENT architecture
#: (35B-A3B): without this check V4-B would have been tuned on the wrong
#: model. Dequantising 10 real tensors from the APEX donor:
CURVA_ROTAZIONE_TONY = {4: -3.76, 8: -5.04, 32: -6.30}   # two-plane
#: against the 397B: k=8 -> -4.84%. It holds, and is in fact slightly better.
#
#: ⚠️ TRAP THAT NEARLY COST THE FINDING: in a quantized GGUF, `t.data` holds
#: the PACKED BYTES, not the weights. Read as float32 the result said +296%
#: ("rotation destroys Tony") and the best result of the day was one step from
#: the bin. Dequantise with `gguf.quants.dequantize(t.data, t.tensor_type)`,
#: and ALWAYS sanity-check that max|w| is plausible (real weights sit between
#: 0.05 and 0.25; if you read 255, you are looking at bytes).

#: ⭐ THE HYBRID — the best recipe found so far, and it came out of a test
#: written to measure how much a BUG was costing.
#: First plane in the ROTATED space (rotation helps it spend its single scale
#: well), then de-rotate, and the second plane corrects in the ORIGINAL basis
#: (where the residual has structure it can recognise).
#: Each plane works in the space it is better at.
#: On 10 real Tony tensors, against the recipe without rotation:
CURVA_IBRIDA = {4: -9.28, 8: -12.21, 32: -15.54}
CURVA_PURA   = {4: -4.68, 8: -6.04,  32: -7.41}
#: The hybrid at 2 passes (-9.28%) beats the pure one at 5 passes (-7.41%):
#: cheaper and better. And the engine has to de-rotate ONLY the first plane —
#: less kernel code, not more.
#: Chain verified: derotate(rotate(w)) == w with EXACTLY zero deviation.
#: Bug ruled out: rotation applied twice -> +467% (impossible to miss).

#: NEGATIVE RESULTS — worth as much as the positive ones: anyone retrying
#: these paths should know they have already been measured and rejected.
REJECTED_RECIPES = {
    # name: (error vs joint_planes, why)
    "hybrid_without_coupling": (+10.70,
        "1st plane rotated + 2nd plane alone: breaking the coupling between "
        "the scales throws away more than the rotation gives"),
    "different_spaces_coupled": (+7.87,
        "planes in different spaces with scales solved jointly. Crossing the "
        "rotation on every round degrades sign assignment more than the "
        "specialisation gains. NB: this sign alternation is cruder than "
        "joint_planes (which enumerates the 9 pairs); a better implementation "
        "could narrow the gap, but it starts at +7.87 against a -3.09 mark"),
}
RICETTE_BOCCIATE = REJECTED_RECIPES   # backwards-compatible alias
#: THE V4-B RECIPE (the only survivor): rotate the WHOLE block in sub-groups
#: of k, joint_planes on both planes in the rotated space, de-rotation in the
#: kernel. Against plain joint_planes:
RICETTA_V4B = {4: -3.09, 8: -3.62, 32: -4.38}   # k -> gain %
#: cost: log2(k) add/subtract passes in the decoder. Bottleneck: the Vulkan
#: kernel (half a day). The rest is 3 lines in the forge.

#: ⭐ THE TAIL PAYS MORE — measured on 397B tensors. The depth recipe puts the
#: 2nd plane ONLY on layers 44-59, and the rotation gain GROWS with the number
#: of planes: prediction confirmed.
ROTAZIONE_PER_PROFONDITA = {
    "head_<44": {4: -2.92, 32: -4.36},
    "tail_44+": {4: -4.20, 32: -5.61},   # nearly DOUBLE at k=4
}
#: Practical consequence: ODINO v3.5 = re-forging the TAIL ONLY (16 layers of
#: 60), hours instead of days, with the rest of the model untouched.
#: Reading: the depth recipe and the rotation are the SAME idea applied twice —
#: where precision is needed you spend more, first with an extra plane, then
#: with a better basis for that plane.

#: ⭐⭐ ROTATION PAYS IN OPPOSITE DIRECTIONS DEPENDING ON TENSOR TYPE.
#: Measured on Tony's weights (APEX donor), gain at k=4 against plain
#: joint_planes. The AGGREGATE number (-3 to -4.5%) was hiding this:
ROTAZIONE_PER_TIPO = {
    "ffn_gate_shexp": -12.24, "ffn_down_shexp": -10.19, "ffn_up_shexp":  -9.57,
    "attn_gate":       -6.51, "attn_out":        -5.57, "attn_qkv":      -5.32,
    "ffn_gate_exps":   -1.35, "ffn_up_exps":     -0.87,
    "ffn_down_exps":   +1.45,   # 🔴 WORSE: 5 tensors out of 8
}
#: TWO WORLDS: SHARED experts + attention gain 5-12%; ROUTED experts almost
#: nothing, and ffn_down_exps gets WORSE.
#: → the recipe should be DIFFERENTIATED by type, not applied uniformly.
#: → do NOT rotate ffn_down_exps: it is the same family ODINO already treats
#:   separately by transplanting it from the Q6_K donor. Two independent
#:   investigations point at the same tensor as special.
#: ⚠️ 8 tensors per type, dispersion up to ±3.67: a 32-tensor confirmation
#:   was run separately.

#: ⛔⛔ THE NUMBER THAT CUTS EVERYTHING DOWN TO SIZE — weighted by REAL bytes.
#: Per-type percentages do NOT add up: they must be weighted by how many
#: weights each type holds. In a 512-expert MoE the ROUTED ones are 93% of
#: the weights.
PESO_PER_TIPO = {   # share of Tony's weights (35B-A3B)
    "ffn_down_exps": 31.00, "ffn_gate_exps": 31.00, "ffn_up_exps": 31.00,
    "attn_qkv": 1.42, "attn_gate": 0.71, "attn_out": 0.26,
    "ffn_down_shexp": 0.12, "ffn_gate_shexp": 0.12, "ffn_up_shexp": 0.12,
}
GUADAGNO_MODELLO_INTERO = -1.57      # UNIFORM recipe, weighted by bytes
GUADAGNO_SOLO_BUONI     = -0.22      # shared+attention only (2.75% of weights)

#: ⚠️ TWO CORRECTIONS to what was written earlier the same evening:
#: 1) The "per-type differentiated recipe" is WRONG: rotating only the types
#:    that gain most throws away 86% of the gain (-0.22 instead of -1.57).
#:    The bulk comes from the routed experts: little each, but they are 93%.
#: 2) The -3 to -4.5% measured earlier was INFLATED by an unrepresentative
#:    sample: the `odino-q8` dump contains ONLY attn_gate and ssm_out, i.e.
#:    exactly the types that gain a lot. On real weights: -1.57%.
#: The rule this produces: a per-tensor mean is NOT a model-level gain until
#: you weight it by the bytes each type holds.

#: ⭐⭐⭐ THE FINAL RECIPE — complete matrix (14 cells) over 15 ROUTED tensors
#: of Tony (93% of the weights), baseline joint_planes at 16 rounds:
RICETTA_FINALE = {
    "1_sixteen_rounds": {"gain": -0.81, "cost": "one number in the forge"},
    "2_column_perm":    {"gain": -1.80, "cost": "forge wiring; ZERO runtime",
                         "note": "by column NORM; median -0.88, dispersion "
                                 "±3.74, 1/12 worse; permutations must be "
                                 "COORDINATED across adjacent tensors (columns "
                                 "of down <-> rows of gate/up of the same "
                                 "expert)"},
    "3_rotation":       {"gain": -0.05, "cost": "new kernel",
                         "verdict": "REJECTED against 16 rounds: the rounds "
                                    "already collect that gain on routed "
                                    "experts. Still useful ONLY on attn+shexp "
                                    "(2.75% of weights)"},
}
#: ⚠️ a FLAT sort of the tensor gives -95% but is ILLEGAL: it crosses row
#: boundaries, the permutation does not get absorbed, and at runtime it would
#: cost bits and bandwidth. The legal version (columns) is the one above.

#: ⚠️ CORRECTION — the permutation is worth ~-1%, not -1.80%. The -1.80% was
#: a mean over THREE families with dispersion ±3.74 across 12 tensors: wider
#: than the effect it claimed to measure. Measured per family over 10 tensors
#: each (median, which noise does not move):
PERMUTAZIONE_PER_FAMIGLIA = {
    "ffn_up_exps":   {"median": -1.31, "dispersion": 0.47, "worse": "0/10"},
    "ffn_gate_exps": {"median": -1.16, "dispersion": 3.84, "worse": "0/10"},
    "ffn_down_exps": {"median": -0.41, "dispersion": 0.57, "worse": "1/10"},
}
#: `ffn_up_exps` is the clean case: tiny dispersion, zero regressions.
#: On `gate` the MEAN (-2.41%) is pulled by one outlier: trust the median.
#: On `down` the effect is almost absent — and indeed no ordering criterion
#: changes it (norm, mean, max, kurtosis and the INVERSE norm all give the
#: same number: when reverse order ties, you are not measuring order).
#: VERDICT: keep it. Costs one argsort, makes 1 tensor in 30 worse, and gives
#: a full point on two families out of three.

#: 🔭 THE MOST PROMISING DIRECTION SEEN SO FAR — not yet attempted.
#: Permuting the HIDDEN dimension (not the intermediate one) yields far more
#: than on routed experts, with zero regressions over 10 tensors:
PERMUTAZIONE_DIMENSIONE_NASCOSTA = {
    "ffn_gate_shexp": -11.39, "ffn_up_shexp": -7.62,
    "attn_qkv":        -7.38, "attn_gate":    -5.21,
    "attn_out":        +0.33,          # the only one that gets worse (7/10)
}
#: ⚠️ NOT APPLICABLE as-is: the hidden dimension is the residual stream, shared
#: by the WHOLE model. Permuting it requires a coordinated GLOBAL reordering,
#: not a per-expert local trick.
#: → This is the same family of ideas as SpinQuant / QuaRot, which rotate the
#:   residual stream instead of permuting it. If it is ever tackled, the
#:   potential gain is 5-11% on the most sensitive tensors, against the 1-2%
#:   of the routed experts exploited today.
#: ⚠️ `ffn_down_shexp` (-2.65%) WOULD be applicable (intermediate dimension,
#:   local trick) but shared experts are 0.36% of the weights: worth a
#:   hundredth of a point model-wide. Leave it alone.

# ⛔ MEASURED CORRECTION — the permutation is NOT worth -1.80%.
# Clean measurement on 24 real Tony experts, one lever at a time, against the
# baseline the forge actually uses (`joint_planes`, 8 rounds):
#     32 rounds only            -0.64%   worse 0/24
#     permutation only          -0.05%   worse 4/24   <-- next to nothing
#     32 rounds + permutation   -0.73%   worse 0/24
# The announced -1.80% came from a comparison against a WEAKER baseline: the
# same mistake already made with the rotation (-9.28%, which against the right
# baseline became +10.70%). RULE: a lever is ALWAYS measured against the
# production recipe, never against a simplified variant.
# Consequence: **V4-B is "32 rounds"**. The permutation is free but useless.
GUADAGNO_PERMUTAZIONE_MISURATO = -0.05      # percent, 24 experts, disp 0.18
GUADAGNO_GIRI32_MISURATO = -0.64            # percent, 24 experts, disp 0.17
GUADAGNO_V4B_MISURATO = -0.73               # percent, 24 experts, disp 0.14

# ⭐ THE ROTATION IS REINSTATED, AND ON THE 397B IT IS WORTH TWICE AS MUCH.
# It had been written off on a measurement against the wrong baseline.
# Re-measured against the production recipe (`joint_planes` at 32 rounds), on
# REAL tensors:
#
#   rotation    passes    on Tony 35B     on the 397B    worse (397B)
#   k=8            3         -1.01%          -3.45%          1/10
#   k=32           5         -2.09%          -4.14%          0/10
#   k=128          7         -2.57%          -4.14%          0/10
#   k=256          8         -2.63%        **-5.03%**        0/10
#   signs only     0         +0.00%             —             —
#
# Three operational conclusions:
#  1. There is NO free shortcut: sign flips alone give ZERO. The gain comes
#     from MIXING the weights, not from changing their signs.
#  2. There is no knee: the curve climbs slowly all the way to k=256. But
#     k=256 is the only setting that makes NO tensor worse (0/24 on Tony,
#     0/10 on the 397B).
#  3. k=256 = the ENTIRE TQ1_0 block → the kernel needs no sub-group handling:
#     a single Hadamard per block. SIMPLER, not more complex.
# ⚠️ Stated limit: the 397B tensors used here are attention/ssm, NOT experts.
#    It says the lever works on that model, not how much it would improve the
#    finished file.
ROTAZIONE_MISURATA_TONY = {8: -1.01, 32: -2.09, 128: -2.57, 256: -2.63}
ROTAZIONE_MISURATA_397B = {8: -3.45, 32: -4.14, 128: -4.14, 256: -5.03}
#: confirmation on a 4x sample (40 real tensors): k=8 -3.57 (1/40), k=32 -4.12
#: (0/40), k=128 -4.20 (0/40), k=256 **-4.99 (0/40)**. Stable.
ROTAZIONE_397B_40TENSORI = {8: -3.57, 32: -4.12, 128: -4.20, 256: -4.99}
ROTAZIONE_SOLO_SEGNI = +0.00      # zero: not a shortcut
ROTAZIONE_CONSIGLIATA = 256       # the only one with no regressions; whole block
