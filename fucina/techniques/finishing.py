"""Objectives A (two-plane experts at the Q2_K size) and B (< 9.49 GB): finishing levers developed and measured on
Thor-T on 12-13 Oct 2026, plus the per-family lever search (all on / all but one / without the negative ones).

All modules are OPT-IN (``default_off``): automatic runs keep the recorded recipes until ``fucina enable <name>``.
Hosts: A record = CARDAN faithful on PIENO, MMLU-2000 int8 80.30 (13,083,647,680 B); B record = SEL_pred + CARDAN on
eb882K, 76.05 (9,470,184,864 B). Lab paths in ``evidence`` are relative to the lab root."""
from __future__ import annotations

from ..registry import register
from ._base import DriverTechnique, result

DENSE = ("attn", "gdn", "shexp")
_OFF_A = "objective A finishing lever, measured on Thor-T only"
_LOO = "mmlu_logprob/aragorn/B_LOO/"


@register
class DenseResidualKl4(DriverTechnique):
    """Rank-16 ternary residuals on the dense tensors trained on the answer-token KL (KL4) against the teacher"""
    name = "dense_residual_kl4"
    phase = "F4"
    pieces = DENSE
    judged_by = "training"
    variant = "thor"
    deviations = ("RILQ form with the answer-KL4 loss (V1: full KL + KL4; V2: KL4 only). The step is selected on the "
                  "same held-out questions as the gate (optimistic gate, declared). Measured with an optional closed "
                  "post-processing hook (not part of Fucina) that keeps the file size unchanged")
    source = "arXiv 2412.01129 (RILQ) as the form; the loss is ours"
    params = {"rank": 16, "variant": "V1", "steps": 200, "lr": "1e-5/1e-4"}
    driver = "driver_a_dense_kl4"
    driver_training = True
    correction_hook = True
    gpu_cost_h = 4.0
    default_off = _OFF_A
    lab_status = "NEUTRAL"
    measured = (result("A", "A record (80.30)", "80.20", "80.30", "30/32", "0.90",
                       "held-out gate dKL4 -0.053 z -9.7; KL24 -5.82 %; tg -1.7 %: the KL4 gain does not transfer",
                       "stato/cicli.tsv 2026-10-12T09:08 · CARDAN/densi_kl4/mmlu_V1.log"),)


@register
class KScaleAnswerKl4(DriverTechnique):
    """K scale on single-plane experts (fp16 scales x k, zero bytes) with k tuned on the answer KL4"""
    name = "k_scale_answer_kl4"
    phase = "F2"
    pieces = ("experts_gate_up", "experts_down")
    model_types = ("moe", "hybrid")
    variant = "original"
    deviations = "one k for every single-plane expert; per-family k not covered"
    source = "original (Thor-T lever K, R029 item 30)"
    params = {"grid": "0.98,1.00,1.02,1.03,1.045,1.06"}
    driver = "driver_a_k_scale"
    gpu_cost_h = 2.7
    default_off = _OFF_A
    lab_status = "PREPARED"
    measured = (result("A", "A record", "not run", "", "", "", "dry run green; earlier pv12K: KL24 -0.0040 only",
                       "stato/cicli.tsv 2026-10-13T19:27"),)


@register
class E2mStage2(DriverTechnique):
    """E2M stage II: closed-form joint refit of the dense scales over every plane at frozen trits (zero bytes)"""
    name = "e2m_stage2"
    phase = "F1"
    pieces = DENSE
    variant = "faithful"
    deviations = "one scale per 128-group per plane instead of scale+offset per row (symmetric engine format)"
    source = "arXiv 2606.13054 (TWLA §3.2 / Alg. 3)"
    params = {"arms": "all_planes,first_plane_fixed"}
    driver = "driver_a_e2m"
    gpu_cost_h = 1.5
    default_off = _OFF_A
    lab_status = "PREPARED"
    measured = (result("A", "A record", "not run", "", "", "", "CPU: blk.0.attn_gate error 0.0233 -> 0.0200",
                       "stato/cicli.tsv 2026-10-13T19:27"),)


@register
class DensePtqBeforeExpertKd(DriverTechnique):
    """Order rule: faithful dense PTQ (SchurQuant or G2PTQ) on the host BEFORE the expert distillation, which is then
    redone on the winner"""
    name = "dense_ptq_before_expert_kd"
    phase = "F1"
    pieces = DENSE
    variant = "faithful"
    deviations = "Thor arm in the host's ternary format (same bytes); faithful-format arm kept as reference only"
    source = "arXiv 2608.15567 (SchurQuant) + arXiv 2609.31009 (G2PTQ), then arXiv 2609.21137 (CARDAN)"
    params = {"methods": "schurquant,g2ptq"}
    driver = "driver_a_dense_then_kd"
    driver_training = True
    gpu_cost_h = 35.0
    default_off = "prepared, not run (30-45 h GPU, estimate)"
    lab_status = "PREPARED"
    measured = (result("A", "PIENO (79.40)", "not run", "", "", "", "faithful cores bit-exact vs official code (CPU)",
                       "stato/cicli.tsv 2026-10-13T19:27"),)


@register
class FamilyLeverSearch(DriverTechnique):
    """Per-family lever search on top of a selection: all on, all but one, and all without the negative levers"""
    name = "family_lever_search"
    phase = "F1"
    pieces = DENSE
    judged_by = "kl"
    variant = "original"
    deviations = "K-FAC predicted dKL drives the selection; the paired MMLU decides (predictor over-estimates)"
    source = "original (K-FAC curvature, arXiv 1503.05671, as the predictor)"
    params = {"mode": "without_negatives", "levers": ("tjarq_p1fermo,tied_jarq,schur_p2k,bil_tjarq_p1fermo,"
                                                      "schur_pk,jarq_pk,bil_foem,bil_tjarq,jarq_p2k")}
    driver = "driver_b_lever_search"
    gpu_cost_h = 1.0
    default_off = "objective B lever search, Dell bench engine"
    lab_status = "RUNNING"
    measured = (
        result("B", "eb882K (74.90)", "76.05", "74.90", "111/88", "0.12", "all on (SEL_pred + CARDAN); KL -6.65 %",
               "stato/cicli.tsv 2026-10-12T07:05 · COMBINAZIONE_eb882K/misura/mmlu_SEL_pred_CARDAN.log"),
        result("B", "all on, Dell (75.95)", "76.30", "75.95", "59/52", "0.57", "without tjarq_p1fermo: -0.35",
               "stato/cicli.tsv 2026-10-12T21:44 · " + _LOO + "B-LOO-tjarq_p1fermo.log"),
        result("B", "all on, Dell (75.95)", "76.15", "75.95", "35/31", "0.71", "without tied_jarq: -0.20",
               "stato/cicli.tsv 2026-10-13T00:46 · " + _LOO + "B-LOO-tied_jarq.log"),
        result("B", "all on, Dell (75.95)", "76.10", "75.95", "32/29", "0.80", "without schur_p2k: -0.15",
               "stato/cicli.tsv 2026-10-13T03:50 · " + _LOO + "B-LOO-schur_p2k.log"),
        result("B", "all on, Dell (75.95)", "76.00", "75.95", "", "", "without bil_tjarq: -0.05 (log only)",
               _LOO + "B-LOO-bil_tjarq.log"),
        result("B", "all on, Dell (75.95)", "75.85", "75.95", "33/35", "0.90", "without bil_tjarq_p1fermo: +0.10",
               "stato/cicli.tsv 2026-10-13T07:17 · " + _LOO + "B-LOO-bil_tjarq_p1fermo.log"),
        result("B", "all on, Dell (75.95)", "75.70", "75.95", "25/30", "0.59", "without jarq_pk: +0.25",
               "stato/cicli.tsv 2026-10-13T13:20 · " + _LOO + "B-LOO-jarq_pk.log"),
        result("B", "all on, Dell (75.95)", "75.65", "75.95", "23/29", "0.49", "without schur_pk: +0.30",
               "stato/cicli.tsv 2026-10-13T10:24 · " + _LOO + "B-LOO-schur_pk.log"),
        result("B", "all on, Dell (75.95)", "75.45", "75.95", "15/25", "0.15", "without bil_foem: +0.50",
               "stato/cicli.tsv 2026-10-13T16:22 · " + _LOO + "B-LOO-bil_foem.log"),
        result("B", "-", "additivity ratio 0.12", "", "", "", "sum of contributions -0.0089 vs total dKL4 -0.0762",
               "stato/cicli.tsv 2026-10-12T07:25 · COMBINAZIONE_eb882K/additivita.log"),
    )
