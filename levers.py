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
      {"ornith397": "0.1% — nothing"}, "one tensor", "HBLLM 2512.00862"),
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

 # ── RIPARAZIONE (post-forgia, senza toccare i ternari) ────────────────
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

 # ── RUNTIME (non toccano il file) ──────────────────────────────────────
 Lever("low_bit_sampling", "runtime", 0, 0,
      {"unsloth+3papers": "temp 0.6, min_p 0.03, presence 1.5, DRY, thinking ON; never XTC"},
      "A/B on the fixture suite", "docs section 23"),
 Lever("mtp_speculation", "runtime", 0, 2,
      {"ornith_family": "+10% at draft length 1"}, "benchmark with/without", "ours"),
 Lever("self_draft_from_plane_one", "runtime", 0, 24,
      {"literature": "expected routing overlap 0.15-0.30 < the 0.5 threshold — likely not viable; QSpec reports 1.64x at high overlap"},
      "measure top-k routing overlap between tokens — one hour", "docs sections 24-25"),
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
