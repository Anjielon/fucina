"""Objective C — single ternary plane per tensor (BitNet convention): post-training quantizers of the dense tensors
and light distillation, developed and measured on Thor-T on 12-13 Oct 2026.

Every module is OPT-IN (``default_off``): automatic runs skip it until ``fucina enable <name>``, so the measured
recipes of objectives A and B are unchanged. Each one wraps a validated lab driver (``paths.<driver>``, see
``examples/thor/config.yaml``) and carries its measured status (``measured``): MMLU-2000 int8, paired McNemar against
the stated reference, answer-KL4 on held-out questions. Lab paths in ``evidence`` are relative to the lab root.

Reference hosts (same file size, 8,631,199,776 B, every arm written at the exact size of its host):
``(c)`` = balanced GPTQ K=1 + JARQ = 63.80 · ``(c)+DRQ`` = 64.65 · C0B (dense tensors trained end to end) = 65.20."""
from __future__ import annotations

from ..registry import register
from ._base import DriverTechnique, result

DENSE = ("attn", "gdn", "shexp")
_OFF = "objective C (single ternary plane), measured on Thor-T only"
_PTQ2 = "stato/cicli.tsv 2026-10-12T22:59 · OBIETTIVO_C/PTQ2/"


class _SinglePlanePtq(DriverTechnique):
    """Common ground of the single-plane dense quantizers (one arm of the PTQ battery driver)."""
    phase = "F1"
    pieces = DENSE
    group = "dense_quantizer_single_plane"
    per_byte = True
    gpu_cost_h = 0.4                  # MMLU-2000 int8 ~12-20 min per arm; construction is CPU (~2 h per arm, 4 threads)
    driver = "driver_c_ptq"
    default_off = _OFF


@register
class GptqK1(_SinglePlanePtq):
    """GPTQ with a single ternary plane (min-max grid): the control every single-plane quantizer is judged against"""
    name = "gptq_k1"
    variant = "control"
    source = "arXiv 2210.17323 (GPTQ)"
    params = {"arm": "a"}
    lab_status = "FAIL"
    measured = (result("C", "C0B experts, dense from bf16", "23.60", "", "", "", "collapse when used alone",
                       "stato/cicli.tsv 2026-10-12T02:01 (reference arm (a), Dell engine)"),)


@register
class JarqK1(_SinglePlanePtq):
    """JARQ (faithful): joint scale refit + Babai rounding after GPTQ, single plane"""
    name = "jarq_k1"
    variant = "faithful"
    source = "arXiv 2609.38599 (JARQ, Alg. 1)"
    params = {"arm": "b"}
    lab_status = "PASS"
    measured = (result("C", "GPTQ K=1 (a)", "31.65", "23.60", "273/112", "1.3e-16", "dKL4 -1.342 z -34.6",
                       "stato/cicli.tsv 2026-10-12T02:01 (Dell engine)"),)


@register
class BalancedGptqK1Jarq(_SinglePlanePtq):
    """Balanced GPTQ (uniform grid, single plane) followed by faithful JARQ: the single-plane base recipe (c)"""
    name = "bil_k1_jarq"
    variant = "faithful"
    source = "arXiv 2609.05641 (balanced grid) + arXiv 2609.38599 (JARQ)"
    params = {"arm": "c"}
    lab_status = "PASS"
    measured = (
        result("C", "GPTQ K=1 (a)", "63.90", "23.60", "971/165", "2.4e-139", "Dell engine",
               "stato/cicli.tsv 2026-10-12T04:01"),
        result("C", "C0B (dense trained end to end)", "63.80", "65.20", "174/202", "0.16", "Corsair int8",
               "stato/cicli.tsv 2026-10-12T09:27 · E2E_C2/mmlu_C-bil1jarq.log"),
        result("C", "balanced GPTQ K=1 without JARQ (b1)", "63.80", "58.25", "309/198", "9.4e-07",
               "JARQ contribution +5.55", _PTQ2 + "mmlu_C-p_bil1.log"),
    )


@register
class Osaq(_SinglePlanePtq):
    """OSAQ phase 2 (outlier-safe rotation of the inputs) before balanced GPTQ K=1 + JARQ"""
    name = "osaq"
    variant = "faithful"
    deviations = "hyper-parameters not given in the paper: centre of the Fig. 5 grid"
    source = "arXiv 2605.04738 (OSAQ)"
    params = {"arm": "d"}
    lab_status = "FAIL"
    measured = (result("C", "(c)", "60.70", "63.80", "202/264", "0.0047", "dKL4 +0.048 z +3.6; Dell 60.75 p 0.004",
                       "stato/cicli.tsv 2026-10-12T09:54, 10:06"),)


@register
class SchurOptK1(_SinglePlanePtq):
    """SchurOpt scale/code refit between balanced GPTQ K=1 and JARQ"""
    name = "schuropt_k1"
    variant = "faithful"
    source = "arXiv 2608.15567 (SchurQuant / SchurOpt)"
    params = {"arm": "e"}
    lab_status = "FAIL"
    measured = (result("C", "(c)", "61.90", "63.80", "193/231", "0.072", "dKL4 +0.025 z +2.3",
                       "stato/cicli.tsv 2026-10-12T10:07"),)


@register
class Foem(_SinglePlanePtq):
    """FOEM: first-order error compensation inside GPTQ (reference form of the authors' code), optional JARQ after"""
    name = "foem"
    variant = "faithful"
    deviations = ("blocks of 128 columns on scale groups of 256 (0.4 % of codes); the earlier 'work form' (beta "
                  "applied to every remaining column of the block, 25.30) was an implementation error, now cured")
    source = "arXiv 2507.11017 (FOEM); bit-identical to GPTQModel foem.py (audit 13 Oct)"
    params = {"arm": "f_ref", "beta": 0.1, "then_jarq": True}
    lab_status = "NEUTRAL"
    measured = (
        result("C", "balanced GPTQ K=1 (b1)", "57.40", "58.25", "208/225", "0.44", "FOEM alone, dKL4 +0.067",
               _PTQ2 + "mmlu_C-p_bil1foemR.log"),
        result("C", "(c)", "63.40", "63.80", "179/187", "0.71", "FOEM + JARQ, dKL4 +0.008 z +0.8",
               _PTQ2 + "mmlu_C-p_bil1foemRjarq.log"),
    )


@register
class ActOrderSign(_SinglePlanePtq):
    """Static group act-order + sign criterion in balanced GPTQ K=1 (optional JARQ after)"""
    name = "act_order_sign"
    variant = "faithful"
    source = "arXiv 2605.18933 (sign criterion) + static act-order (arXiv 2210.17323 §4)"
    params = {"arm": "g", "then_jarq": True}
    lab_status = "NEUTRAL"
    measured = (
        result("C", "balanced GPTQ K=1 (b1)", "62.75", "58.25", "313/223", "1.2e-4", "without JARQ",
               _PTQ2 + "mmlu_C-p_bil1aos.log"),
        result("C", "(c)", "62.10", "63.80", "189/223", "0.10", "with JARQ, dKL4 +0.030 z +2.4",
               _PTQ2 + "mmlu_C-p_bil1aosjarq.log"),
    )


@register
class Drq(_SinglePlanePtq):
    """DRQ: robust re-quantization against a held-out control set of Hessians, after JARQ — best single-plane PTQ"""
    name = "drq"
    variant = "faithful"
    deviations = ("(alpha, beta) applied on the raw Hessian scale: DRQ barely acts on attn_output, ssm_out and "
                  "the shared-expert down projection; cure = per-family alpha/beta (prepared, not measured)")
    source = "arXiv 2610.11226 (DRQ; paper +0.91 on a 3-bit MoE); 13/13 closed-form tests (audit 13 Oct)"
    params = {"arm": "h", "tolerance_pct": 1.0}
    lab_status = "PASS"
    measured = (
        result("C", "balanced GPTQ K=1 (b1)", "60.85", "58.25", "101/49", "2.6e-05", "DRQ without JARQ",
               _PTQ2 + "mmlu_C-p_bil1drq.log"),
        result("C", "(c)", "64.65", "63.80", "56/39", "0.10", "(c)+DRQ, dKL4 -0.0077 z -2.8; new C host",
               _PTQ2 + "mmlu_C-p_bil1jarqdrq.log"),
    )


@register
class ScaleGroups64(DriverTechnique):
    """Scale groups redistributed at equal bytes (64 on sensitive tensors, 256 on the rest), same recipe as the host"""
    name = "scale_groups_g64"
    phase = "F1"
    pieces = DENSE
    variant = "original"
    source = "no paper: Thor-T scale-group redistribution (engine types TQ1_0R_1024_G64/_G256)"
    params = {"plan": "piano_g64.json"}
    driver = "driver_c_g64"
    gpu_cost_h = 0.8
    slows_down = True
    default_off = _OFF
    lab_status = "FAIL"
    measured = (
        result("C", "(c)+DRQ", "tg128 -5.15 %", "", "", "", "speed gate on the probe GGUF (ABBAAB)",
               "OBIETTIVO_C/G64_VELOCITA/velocita_1013_1146/esito.json"),
        result("C", "(c)+DRQ", "acc 62.97 (held-out 883)", "64.67", "52/67", "", "dKL4 +0.058 z +4.6: not adopted",
               "stato/cicli.tsv 2026-10-13T13:12 · OBIETTIVO_C/G64_BRACCIO_1013/corsa.log"),
    )


@register
class SsmDtBias(DriverTechnique):
    """GatedDeltaNet ssm_dt.bias chosen per head in the exact metric of the recurrent memory (zero bytes)"""
    name = "ssm_dt_bias"
    phase = "F1"
    pieces = ("gdn",)
    variant = "original"
    source = "original (derived from bias correction, arXiv 1906.04721, and Gated DeltaNet, arXiv 2412.06464)"
    params = {"corpus": "wikitext2", "rows": "32+16", "ctx": 2048}
    driver = "driver_c_ssm_dt"
    gpu_cost_h = 0.5
    default_off = _OFF
    lab_status = "FAIL"
    measured = (result("C", "E2E-QP p3 (64.65)", "acc 65.01 (held-out 883)", "64.78", "14/12", "",
                       "memory metric -58.9 %, 873/960 heads; dKL4 +0.0025 z +0.9: gate not passed, no bench",
                       "stato/cicli.tsv 2026-10-13T12:31 · OBIETTIVO_C/M1_SSMDT_1013/corsa.log"),)


@register
class FairCalib(DriverTechnique):
    """FAIR-Calib: calibration Hessians re-captured with token weights from teacher rollouts, then the C recipe"""
    name = "fair_calib"
    phase = "F1"
    pieces = DENSE
    variant = "faithful"
    deviations = "extra per-token reliability-weighted arm (A_F2) declared as Thor variant"
    source = "arXiv 2606.06547 (FAIR-Calib, ICML 2026)"
    params = {"rollouts": 512, "drq_arms": "A_F2 B_F1"}
    driver = "driver_c_fair_calib"
    gpu_cost_h = 3.0
    default_off = _OFF
    lab_status = "RUNNING"
    measured = (result("C", "(c), (c)+DRQ", "not measured yet", "", "", "",
                       "CPU test: uniform weights rebuild (c) and (c)+DRQ bit for bit",
                       "stato/cicli.tsv 2026-10-13T19:12"),)


@register
class HSvdQuant(DriverTechnique):
    """H-SVDQuant: Hessian-weighted low-rank branch + ternary residual on the dense tensors"""
    name = "hsvdquant"
    phase = "F1"
    pieces = DENSE
    variant = "faithful"
    deviations = "Thor variant: K-plane ternary residual (K=2) instead of the paper's 4-bit residual"
    source = "arXiv 2610.11315 (H-SVDQuant)"
    params = {"rank": 4}
    driver = "driver_hsvdquant"
    est_extra_bytes = 9_584_640
    gpu_cost_h = 8.0
    default_off = "prepared, not run; the rank-4 branch exceeds the objective-A cap by 7.45 MB"
    lab_status = "PREPARED"
    measured = (result("A/B", "-", "CPU only", "", "", "",
                       "attn_gate inputs: rank 4 -57.5 % error vs SVDQuant r4; K=2 residual -38 % vs ternary GPTQ; "
                       "tests 34/34 + 15/15", "stato/cicli.tsv 2026-10-13T02:54"),)


# ── home plan for C (sequence S1 -> S1b -> S2+S3 -> S4 -> S5 -> S7), prepared 13 Oct, not run yet ──

@register
class FlowStatistics(DriverTechnique):
    """Statistics of the quantized model's own flow (asymmetric targets, router-weighted expert Hessians)"""
    name = "flow_statistics"
    phase = "F0"
    pieces = DENSE + ("experts_gate_up", "experts_down")
    judged_by = "diagnosis"
    variant = "faithful"
    deviations = "expert Hessians shrunk towards the layer Hessian (k ~ 8000), path Fisher (Thor variant)"
    source = ("arXiv 2504.02692 (GPTAQ) + arXiv 2504.09629 (QEP) + arXiv 2505.03804 (MoEQuant AGQ) + "
              "arXiv 2609.31009 (G2PTQ gradients)")
    driver = "driver_c_flow_stats"
    gpu_cost_h = 3.0
    default_off = _OFF
    lab_status = "PREPARED"
    measured = (result("C", "-", "not run", "", "", "", "stopped by the disk precondition (needs 502 GiB free)",
                       "stato/cicli.tsv 2026-10-13T18:37"),)


@register
class ZeroByteScales(DriverTechnique):
    """Closed-form scale refit at frozen trits on the asymmetric objective (dense rows/columns, expert gate/up/down)"""
    name = "zero_byte_scales"
    phase = "F1"
    pieces = DENSE + ("experts_gate_up", "experts_down")
    variant = "faithful"
    deviations = "column scales absorbed by the upstream row scales (DFQ-style rescaling) with H-weighted steps; held-out guard per tensor"
    source = ("arXiv 2510.03267 (PT2-LLM AGA) + arXiv 2410.03129 (ARB-LLM_RC) + arXiv 2609.38599 (JARQ eq. 6) + "
              "arXiv 1906.04721 (DFQ) + arXiv 2402.11295 (OneBit)")
    driver = "driver_c_zero_byte"
    gpu_cost_h = 1.0
    default_off = _OFF
    lab_status = "PREPARED"
    measured = (result("C", "(c)+DRQ", "not run", "", "", "", "dense prototype -18.6 % weighted error (CPU)",
                       "ricerche/C_PIANO_CASALINGO_1013.md §2 S1b"),)


@register
class SequentialRePtq(DriverTechnique):
    """Sequential layer-by-layer re-PTQ on the quantized flow: dense tensors, then experts, inputs re-captured"""
    name = "sequential_re_ptq"
    phase = "F1"
    pieces = DENSE + ("experts_gate_up", "experts_down")
    variant = "faithful"
    deviations = "QEP alpha in {0.5, 1} chosen per tensor on a held-out half; knowledge-weighted Hessian mix"
    source = "arXiv 2504.09629 (QEP) with the C recipe (2609.05641 + 2609.38599 + 2610.11226)"
    driver = "driver_c_sequential"
    gpu_cost_h = 5.0
    default_off = _OFF
    lab_status = "PREPARED"
    measured = (result("C", "-", "not run", "", "", "", "QEP alpha=1 -38.8 % vs GPTAQ -31.4 % weighted error (CPU)",
                       "ricerche/C_PIANO_CASALINGO_1013.md §2 S2+S3"),)


@register
class TesseraQ(DriverTechnique):
    """TesseraQ: block reconstruction with progressive adaptive rounding (PAR) + dequantization scale tuning (DST)"""
    name = "tesseraq"
    phase = "F4"
    pieces = DENSE + ("experts_gate_up", "experts_down")
    judged_by = "training"
    variant = "faithful"
    deviations = "Thor arm: start from the host codes, harden the most confident first, AdamW on the scales"
    source = "arXiv 2410.19103 (TesseraQ, Alg. 1)"
    params = {"rounds": 20, "steps": 250, "lr": 1e-3}
    driver = "driver_c_tesseraq"
    driver_training = True
    gpu_cost_h = 6.0
    default_off = _OFF
    lab_status = "PREPARED"
    measured = (result("C", "-", "not run", "", "", "", "prototype -3.6 % / -2.7 % squared error (CPU)",
                       "ricerche/C_LEVE_DISTILLAZIONE_LEGGERA_1013.md §1"),)


# ── light distillation on the single-plane model ──

@register
class E2eQp(DriverTechnique):
    """EfficientQAT E2E-QP: end-to-end training of the quantization parameters on a generic corpus (codes PV-tuned)"""
    name = "e2e_qp"
    phase = "F4"
    pieces = DENSE + ("experts_gate_up", "experts_down")
    judged_by = "training"
    variant = "faithful"
    deviations = ("dose 3 accepted updates of 128 in the paper (2.3 %); beta2 0.95, no clip, no warm-up; "
                  "patience 2 on MMLU-val without letter preference")
    source = "arXiv 2407.11062 (EfficientQAT E2E-QP) + arXiv 2405.14852 (PV-Tuning) + arXiv 2402.10631 (CAKLD)"
    params = {"lr": 1e-3, "lr_f32": 1e-5, "schedule": "cosine_steps", "patience": 2, "corpus": "generic"}
    driver = "driver_c_e2e_qp"
    driver_training = True
    gpu_cost_h = 10.0
    default_off = _OFF
    lab_status = "NEUTRAL"
    measured = (
        result("C", "(c)", "64.65", "63.80", "66/49", "0.135", "best step 3; dKL4 -0.018 z -5.1",
               "stato/cicli.tsv 2026-10-13T09:07 · E2E_C3/mmlu_E2E-C3-p3.log"),
        result("C", "(c)", "63.60", "63.80", "161/165", "0.87",
               "Thor variant (multiple-choice rows + letter step): moves letter preferences, not knowledge — refuted",
               "stato/cicli.tsv 2026-10-12T19:58, 21:27 · E2E_C2/mmlu_E2E-C2-p2.log"),
    )


@register
class OnPolicyDistillation(DriverTechnique):
    """Quantization-aware on-policy distillation: the student generates, the teacher scores its own trajectories"""
    name = "on_policy_distillation"
    phase = "F4"
    judged_by = "training"
    variant = "faithful"
    deviations = ("faithful driver: official strict reward, fresh rollouts every round, 2 optimizer steps per round "
                  "(trainer constraint); first round ran with stale rollouts and a non-strict reward")
    source = "arXiv 2609.26708 (QAOPD, official reward math_mixed_reward.py)"
    params = {"rounds": 10, "steps_per_round": 2, "rows": 4, "lr": 1.5e-4}
    driver = "driver_c_on_policy"
    driver_training = True
    gpu_cost_h = 6.4
    default_off = _OFF
    lab_status = "NEUTRAL"
    measured = (result("C", "E2E-QP p3", "64.60", "64.65", "56/57", "1.0",
                       "spoken probe 10/21 vs 5/21 (after the probe-count fix); GSM8K held-out 64.1 -> 67.2 n.s.",
                       "stato/cicli.tsv 2026-10-13T14:42 · OBIETTIVO_C/ONPOLICY/corsa.log"),)


@register
class CardanExpertsK1(DriverTechnique):
    """CARDAN (faithful loss and init) on single-plane expert gate/up: KD of a shared low-rank correction in the
    ternary flow"""
    name = "cardan_experts_k1"
    phase = "F2"
    pieces = ("experts_gate_up",)
    model_types = ("moe", "hybrid")
    judged_by = "training"
    variant = "faithful"
    deviations = ("tau/lambda not given in the paper (1.0 / 0.5); rank 16; q8 teacher. The published numbers used an "
                  "optional closed post-processing hook (not part of Fucina) that keeps the file size unchanged")
    source = "arXiv 2609.21137 (CARDAN §3.3 Eq. 3 + App. A.1 Eq. 5-6)"
    params = {"rank": 16, "steps": 300, "tau": 1.0, "lambda_ce": 0.5, "topk": 256, "init_c": "a1"}
    driver = "driver_c_cardan_experts"
    driver_training = True
    correction_hook = True
    gpu_cost_h = 6.0
    default_off = _OFF
    lab_status = "NEUTRAL"
    measured = (result("C", "(c)+DRQ", "65.25", "64.65", "89/77", "0.39",
                       "KL24 -8.62 % (-27.3 se, 24/24 rows); dKL4 -0.006 z -1.3; tg128 87.03 -> 87.48; new C record",
                       "stato/cicli.tsv 2026-10-13T18:33 · OBIETTIVO_C/CARDAN_ESPERTI_1013/kd/mmlu_CARDAN.log"),)
