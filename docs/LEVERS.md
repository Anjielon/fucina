# Lever census

Every lever Fucina knows is a **module**: a self-registering technique (`fucina/techniques/*.py`) with its paper
reference, declared variant, parameters, switch and measured status. This page is generated from the registry —
do not edit the module sections by hand:

```bash
fucina --config examples/thor/config.yaml techniques --census          # module sections of this page
fucina --config examples/thor/config.yaml techniques --census --short  # module table of the README
```

## How the modules are searched

- **Switch.** Each module can be turned on or off without rebuilding anything else: `fucina enable <name>` /
  `fucina disable <name>`. Modules marked *opt-in* are skipped by automatic runs until they are enabled, so adding a
  module never changes a recipe that is already measured.
- **All on / all but one / best combination.** Per-byte levers are combined on top of the consolidated structure and
  measured all on, then with each lever removed in turn (leave-one-out); levers whose removal does not hurt are
  dropped and the combination without them is measured again (`family_lever_search`, mode `without_negatives`).
  The paired MMLU (McNemar) decides; KL figures only order the queue.
- **Variants.** `faithful` = the paper's method and protocol (declared deviations listed); `thor` = the faithful
  method plus a declared extension; `original` = no paper to reproduce; `control` = a reference arm.
  A published lever that underperforms its paper stays **to verify** until its number is reproduced or the cause is
  shown.
- **Post-processing.** Modules flagged with a correction write it as extra tensors. An optional closed external hook,
  not part of Fucina, may post-process it; the measured numbers below say when it was used.

## Status legend

`PASS` adopted · `NEUTRAL` measured, not significant · `FAIL` measured worse or gate not passed · `PREPARED` driver
and CPU tests ready, not run · `RUNNING` measurement in progress. Numbers are MMLU-2000 (int8 engine unless stated),
paired McNemar against the stated reference, and answer-KL4 on held-out questions; `evidence` points to the
activity-log line and result file of the reference lab (paths relative to the lab root). Objectives: **A** two-plane
experts at the Q2_K size (≤ 12.19 GiB) · **B** best MMLU under 9,492,293,427 B · **C** single ternary plane per tensor.

## Modules

### `flow_statistics` — Statistics of the quantized model's own flow (asymmetric targets, router-weighted expert Hessians)

- **Phase / pieces**: F0 · attn, gdn, shexp, experts_gate_up, experts_down
- **Reference**: arXiv 2504.02692 (GPTAQ) + arXiv 2504.09629 (QEP) + arXiv 2505.03804 (MoEQuant AGQ) + arXiv 2609.31009 (G2PTQ gradients)
- **Variant**: faithful — declared deviations: expert Hessians shrunk towards the layer Hessian (k ~ 8000), path Fisher (Thor variant)
- **Parameters**: none
- **Switch**: opt-in: `fucina enable flow_statistics`; off: `fucina disable flow_statistics` · file: same size and tensor set -> switched off by going back to the starting GGUF
- **Driver**: `paths.driver_c_flow_stats` (configuration)

Status: **PREPARED**

| obj. | reference host | value | vs | wins/losses | McNemar p | notes | evidence |
|---|---|---|---|---|---|---|---|
| C | - | not run | — | — | — | stopped by the disk precondition (needs 502 GiB free) | stato/cicli.tsv 2026-10-13T18:37 |

### `measured_dkl` — Whole-model dKL measured per expert (+1 plane) on the teacher's held-out rows

- **Phase / pieces**: F0 · experts_gate_up, experts_down
- **Reference**: rule second_plane_from_dkl (RD1/ALLOC1 from an imatrix proxy: -1.8 MMLU)
- **Parameters**: `kl_rows=32`, `block=3`, `seed=2`, `min_rows=64`
- **Switch**: on in automatic runs; off: `fucina disable measured_dkl` · (diagnosis: no switch)

### `act_order_sign` — Static group act-order + sign criterion in balanced GPTQ K=1 (optional JARQ after)

- **Phase / pieces**: F1 · attn, gdn, shexp
- **Reference**: arXiv 2605.18933 (sign criterion) + static act-order (arXiv 2210.17323 §4)
- **Variant**: faithful
- **Parameters**: `arm=g`, `then_jarq=True`
- **Switch**: opt-in: `fucina enable act_order_sign`; off: `fucina disable act_order_sign` · file: same size and tensor set -> switched off by going back to the starting GGUF
- **Driver**: `paths.driver_c_ptq` (configuration)

Status: **NEUTRAL**

| obj. | reference host | value | vs | wins/losses | McNemar p | notes | evidence |
|---|---|---|---|---|---|---|---|
| C | balanced GPTQ K=1 (b1) | 62.75 | 58.25 | 313/223 | 1.2e-4 | without JARQ | stato/cicli.tsv 2026-10-12T22:59 · OBIETTIVO_C/PTQ2/mmlu_C-p_bil1aos.log |
| C | (c) | 62.10 | 63.80 | 189/223 | 0.10 | with JARQ, dKL4 +0.030 z +2.4 | stato/cicli.tsv 2026-10-12T22:59 · OBIETTIVO_C/PTQ2/mmlu_C-p_bil1aosjarq.log |

### `bil` — Balanced ternary (scales 9D/3D/D, uniform 27-level grid) + GPTQ on the dense tensors and the head

- **Phase / pieces**: F1 · attn, gdn, shexp, head
- **Reference**: arXiv 2609.05641 (nuclear limit -> uniform grid); offline weighted error 0.054 -> 0.026
- **Parameters**: none
- **Switch**: on in automatic runs; off: `fucina disable bil` · file: same size and format -> switched off by going back to the starting GGUF

### `bil_k1_jarq` — Balanced GPTQ (uniform grid, single plane) followed by faithful JARQ: the single-plane base recipe (c)

- **Phase / pieces**: F1 · attn, gdn, shexp
- **Reference**: arXiv 2609.05641 (balanced grid) + arXiv 2609.38599 (JARQ)
- **Variant**: faithful
- **Parameters**: `arm=c`
- **Switch**: opt-in: `fucina enable bil_k1_jarq`; off: `fucina disable bil_k1_jarq` · file: same size and tensor set -> switched off by going back to the starting GGUF
- **Driver**: `paths.driver_c_ptq` (configuration)

Status: **PASS**

| obj. | reference host | value | vs | wins/losses | McNemar p | notes | evidence |
|---|---|---|---|---|---|---|---|
| C | GPTQ K=1 (a) | 63.90 | 23.60 | 971/165 | 2.4e-139 | Dell engine | stato/cicli.tsv 2026-10-12T04:01 |
| C | C0B (dense trained end to end) | 63.80 | 65.20 | 174/202 | 0.16 | Corsair int8 | stato/cicli.tsv 2026-10-12T09:27 · E2E_C2/mmlu_C-bil1jarq.log |
| C | balanced GPTQ K=1 without JARQ (b1) | 63.80 | 58.25 | 309/198 | 9.4e-07 | JARQ contribution +5.55 | stato/cicli.tsv 2026-10-12T22:59 · OBIETTIVO_C/PTQ2/mmlu_C-p_bil1.log |

### `dense_ptq_before_expert_kd` — Order rule: faithful dense PTQ (SchurQuant or G2PTQ) on the host BEFORE the expert distillation, which is then

- **Phase / pieces**: F1 · attn, gdn, shexp
- **Reference**: arXiv 2608.15567 (SchurQuant) + arXiv 2609.31009 (G2PTQ), then arXiv 2609.21137 (CARDAN)
- **Variant**: faithful — declared deviations: Thor arm in the host's ternary format (same bytes); faithful-format arm kept as reference only
- **Parameters**: `methods=schurquant,g2ptq`
- **Switch**: opt-in: `fucina enable dense_ptq_before_expert_kd`; off: `fucina disable dense_ptq_before_expert_kd` · file: same size and tensor set -> switched off by going back to the starting GGUF
- **Driver**: `paths.driver_a_dense_then_kd` (configuration)

Status: **PREPARED**

| obj. | reference host | value | vs | wins/losses | McNemar p | notes | evidence |
|---|---|---|---|---|---|---|---|
| A | PIENO (79.40) | not run | — | — | — | faithful cores bit-exact vs official code (CPU) | stato/cicli.tsv 2026-10-13T19:27 |

### `drq` — DRQ: robust re-quantization against a held-out control set of Hessians, after JARQ — best single-plane PTQ

- **Phase / pieces**: F1 · attn, gdn, shexp
- **Reference**: arXiv 2610.11226 (DRQ; paper +0.91 on a 3-bit MoE); 13/13 closed-form tests (audit 13 Oct)
- **Variant**: faithful — declared deviations: (alpha, beta) applied on the raw Hessian scale: DRQ barely acts on attn_output, ssm_out and the shared-expert down projection; cure = per-family alpha/beta (prepared, not measured)
- **Parameters**: `arm=h`, `tolerance_pct=1.0`
- **Switch**: opt-in: `fucina enable drq`; off: `fucina disable drq` · file: same size and tensor set -> switched off by going back to the starting GGUF
- **Driver**: `paths.driver_c_ptq` (configuration)

Status: **PASS**

| obj. | reference host | value | vs | wins/losses | McNemar p | notes | evidence |
|---|---|---|---|---|---|---|---|
| C | balanced GPTQ K=1 (b1) | 60.85 | 58.25 | 101/49 | 2.6e-05 | DRQ without JARQ | stato/cicli.tsv 2026-10-12T22:59 · OBIETTIVO_C/PTQ2/mmlu_C-p_bil1drq.log |
| C | (c) | 64.65 | 63.80 | 56/39 | 0.10 | (c)+DRQ, dKL4 -0.0077 z -2.8; new C host | stato/cicli.tsv 2026-10-12T22:59 · OBIETTIVO_C/PTQ2/mmlu_C-p_bil1jarqdrq.log |

### `e2m_stage2` — E2M stage II: closed-form joint refit of the dense scales over every plane at frozen trits (zero bytes)

- **Phase / pieces**: F1 · attn, gdn, shexp
- **Reference**: arXiv 2606.13054 (TWLA §3.2 / Alg. 3)
- **Variant**: faithful — declared deviations: one scale per 128-group per plane instead of scale+offset per row (symmetric engine format)
- **Parameters**: `arms=all_planes,first_plane_fixed`
- **Switch**: opt-in: `fucina enable e2m_stage2`; off: `fucina disable e2m_stage2` · file: same size and tensor set -> switched off by going back to the starting GGUF
- **Driver**: `paths.driver_a_e2m` (configuration)

Status: **PREPARED**

| obj. | reference host | value | vs | wins/losses | McNemar p | notes | evidence |
|---|---|---|---|---|---|---|---|
| A | A record | not run | — | — | — | CPU: blk.0.attn_gate error 0.0233 -> 0.0200 | stato/cicli.tsv 2026-10-13T19:27 |

### `fair_calib` — FAIR-Calib: calibration Hessians re-captured with token weights from teacher rollouts, then the C recipe

- **Phase / pieces**: F1 · attn, gdn, shexp
- **Reference**: arXiv 2606.06547 (FAIR-Calib, ICML 2026)
- **Variant**: faithful — declared deviations: extra per-token reliability-weighted arm (A_F2) declared as Thor variant
- **Parameters**: `rollouts=512`, `drq_arms=A_F2 B_F1`
- **Switch**: opt-in: `fucina enable fair_calib`; off: `fucina disable fair_calib` · file: same size and tensor set -> switched off by going back to the starting GGUF
- **Driver**: `paths.driver_c_fair_calib` (configuration)

Status: **RUNNING**

| obj. | reference host | value | vs | wins/losses | McNemar p | notes | evidence |
|---|---|---|---|---|---|---|---|
| C | (c), (c)+DRQ | not measured yet | — | — | — | CPU test: uniform weights rebuild (c) and (c)+DRQ bit for bit | stato/cicli.tsv 2026-10-13T19:12 |

### `family_lever_search` — Per-family lever search on top of a selection: all on, all but one, and all without the negative levers

- **Phase / pieces**: F1 · attn, gdn, shexp
- **Reference**: original (K-FAC curvature, arXiv 1503.05671, as the predictor)
- **Variant**: original — declared deviations: K-FAC predicted dKL drives the selection; the paired MMLU decides (predictor over-estimates)
- **Parameters**: `mode=without_negatives`, `levers=tjarq_p1fermo,tied_jarq,schur_p2k,bil_tjarq_p1fermo,schur_pk,jarq_pk,bil_foem,bil_tjarq,jarq_p2k`
- **Switch**: opt-in: `fucina enable family_lever_search`; off: `fucina disable family_lever_search` · file: same size and tensor set -> switched off by going back to the starting GGUF
- **Driver**: `paths.driver_b_lever_search` (configuration)

Status: **RUNNING**

| obj. | reference host | value | vs | wins/losses | McNemar p | notes | evidence |
|---|---|---|---|---|---|---|---|
| B | eb882K (74.90) | 76.05 | 74.90 | 111/88 | 0.12 | all on (SEL_pred + CARDAN); KL -6.65 % | stato/cicli.tsv 2026-10-12T07:05 · COMBINAZIONE_eb882K/misura/mmlu_SEL_pred_CARDAN.log |
| B | all on, Dell (75.95) | 76.30 | 75.95 | 59/52 | 0.57 | without tjarq_p1fermo: -0.35 | stato/cicli.tsv 2026-10-12T21:44 · mmlu_logprob/aragorn/B_LOO/B-LOO-tjarq_p1fermo.log |
| B | all on, Dell (75.95) | 76.15 | 75.95 | 35/31 | 0.71 | without tied_jarq: -0.20 | stato/cicli.tsv 2026-10-13T00:46 · mmlu_logprob/aragorn/B_LOO/B-LOO-tied_jarq.log |
| B | all on, Dell (75.95) | 76.10 | 75.95 | 32/29 | 0.80 | without schur_p2k: -0.15 | stato/cicli.tsv 2026-10-13T03:50 · mmlu_logprob/aragorn/B_LOO/B-LOO-schur_p2k.log |
| B | all on, Dell (75.95) | 76.00 | 75.95 | — | — | without bil_tjarq: -0.05 (log only) | mmlu_logprob/aragorn/B_LOO/B-LOO-bil_tjarq.log |
| B | all on, Dell (75.95) | 75.85 | 75.95 | 33/35 | 0.90 | without bil_tjarq_p1fermo: +0.10 | stato/cicli.tsv 2026-10-13T07:17 · mmlu_logprob/aragorn/B_LOO/B-LOO-bil_tjarq_p1fermo.log |
| B | all on, Dell (75.95) | 75.70 | 75.95 | 25/30 | 0.59 | without jarq_pk: +0.25 | stato/cicli.tsv 2026-10-13T13:20 · mmlu_logprob/aragorn/B_LOO/B-LOO-jarq_pk.log |
| B | all on, Dell (75.95) | 75.65 | 75.95 | 23/29 | 0.49 | without schur_pk: +0.30 | stato/cicli.tsv 2026-10-13T10:24 · mmlu_logprob/aragorn/B_LOO/B-LOO-schur_pk.log |
| B | all on, Dell (75.95) | 75.45 | 75.95 | 15/25 | 0.15 | without bil_foem: +0.50 | stato/cicli.tsv 2026-10-13T16:22 · mmlu_logprob/aragorn/B_LOO/B-LOO-bil_foem.log |
| B | - | additivity ratio 0.12 | — | — | — | sum of contributions -0.0089 vs total dKL4 -0.0762 | stato/cicli.tsv 2026-10-12T07:25 · COMBINAZIONE_eb882K/additivita.log |

### `foem` — FOEM: first-order error compensation inside GPTQ (reference form of the authors' code), optional JARQ after

- **Phase / pieces**: F1 · attn, gdn, shexp
- **Reference**: arXiv 2507.11017 (FOEM); bit-identical to GPTQModel foem.py (audit 13 Oct)
- **Variant**: faithful — declared deviations: blocks of 128 columns on scale groups of 256 (0.4 % of codes); the earlier 'work form' (beta applied to every remaining column of the block, 25.30) was an implementation error, now cured
- **Parameters**: `arm=f_ref`, `beta=0.1`, `then_jarq=True`
- **Switch**: opt-in: `fucina enable foem`; off: `fucina disable foem` · file: same size and tensor set -> switched off by going back to the starting GGUF
- **Driver**: `paths.driver_c_ptq` (configuration)

Status: **NEUTRAL**

| obj. | reference host | value | vs | wins/losses | McNemar p | notes | evidence |
|---|---|---|---|---|---|---|---|
| C | balanced GPTQ K=1 (b1) | 57.40 | 58.25 | 208/225 | 0.44 | FOEM alone, dKL4 +0.067 | stato/cicli.tsv 2026-10-12T22:59 · OBIETTIVO_C/PTQ2/mmlu_C-p_bil1foemR.log |
| C | (c) | 63.40 | 63.80 | 179/187 | 0.71 | FOEM + JARQ, dKL4 +0.008 z +0.8 | stato/cicli.tsv 2026-10-12T22:59 · OBIETTIVO_C/PTQ2/mmlu_C-p_bil1foemRjarq.log |

### `g2ptq` — G2PTQ: GPTQ guided by the gradients of the KL (planes 2-3)

- **Phase / pieces**: F1 · attn, gdn, shexp
- **Reference**: G2PTQ (official code, audit G2 #17)
- **Parameters**: none
- **Switch**: on in automatic runs; off: `fucina disable g2ptq` · file: same size and format -> switched off by going back to the starting GGUF
- **Suspended**: suspended by the owner on 8 Oct (needs the raff-grad gradients)

### `gptq_k1` — GPTQ with a single ternary plane (min-max grid): the control every single-plane quantizer is judged against

- **Phase / pieces**: F1 · attn, gdn, shexp
- **Reference**: arXiv 2210.17323 (GPTQ)
- **Variant**: control
- **Parameters**: `arm=a`
- **Switch**: opt-in: `fucina enable gptq_k1`; off: `fucina disable gptq_k1` · file: same size and tensor set -> switched off by going back to the starting GGUF
- **Driver**: `paths.driver_c_ptq` (configuration)

Status: **FAIL**

| obj. | reference host | value | vs | wins/losses | McNemar p | notes | evidence |
|---|---|---|---|---|---|---|---|
| C | C0B experts, dense from bf16 | 23.60 | — | — | — | collapse when used alone | stato/cicli.tsv 2026-10-12T02:01 (reference arm (a), Dell engine) |

### `hsvdquant` — H-SVDQuant: Hessian-weighted low-rank branch + ternary residual on the dense tensors

- **Phase / pieces**: F1 · attn, gdn, shexp
- **Reference**: arXiv 2610.11315 (H-SVDQuant)
- **Variant**: faithful — declared deviations: Thor variant: K-plane ternary residual (K=2) instead of the paper's 4-bit residual
- **Parameters**: `rank=4`
- **Switch**: opt-in: `fucina enable hsvdquant`; off: `fucina disable hsvdquant` · file: same size and tensor set -> switched off by going back to the starting GGUF
- **Driver**: `paths.driver_hsvdquant` (configuration)

Status: **PREPARED**

| obj. | reference host | value | vs | wins/losses | McNemar p | notes | evidence |
|---|---|---|---|---|---|---|---|
| A/B | - | CPU only | — | — | — | attn_gate inputs: rank 4 -57.5 % error vs SVDQuant r4; K=2 residual -38 % vs ternary GPTQ; tests 34/34 + 15/15 | stato/cicli.tsv 2026-10-13T02:54 |

### `jarq` — JARQ: joint scales + Babai rounding on the composite of the 3 planes

- **Phase / pieces**: F1 · attn, gdn, shexp
- **Reference**: arXiv 2609.38599 (JARQ)
- **Parameters**: none
- **Switch**: on in automatic runs; off: `fucina disable jarq` · file: same size and format -> switched off by going back to the starting GGUF

### `jarq_k1` — JARQ (faithful): joint scale refit + Babai rounding after GPTQ, single plane

- **Phase / pieces**: F1 · attn, gdn, shexp
- **Reference**: arXiv 2609.38599 (JARQ, Alg. 1)
- **Variant**: faithful
- **Parameters**: `arm=b`
- **Switch**: opt-in: `fucina enable jarq_k1`; off: `fucina disable jarq_k1` · file: same size and tensor set -> switched off by going back to the starting GGUF
- **Driver**: `paths.driver_c_ptq` (configuration)

Status: **PASS**

| obj. | reference host | value | vs | wins/losses | McNemar p | notes | evidence |
|---|---|---|---|---|---|---|---|
| C | GPTQ K=1 (a) | 31.65 | 23.60 | 273/112 | 1.3e-16 | dKL4 -1.342 z -34.6 | stato/cicli.tsv 2026-10-12T02:01 (Dell engine) |

### `osaq` — OSAQ phase 2 (outlier-safe rotation of the inputs) before balanced GPTQ K=1 + JARQ

- **Phase / pieces**: F1 · attn, gdn, shexp
- **Reference**: arXiv 2605.04738 (OSAQ)
- **Variant**: faithful — declared deviations: hyper-parameters not given in the paper: centre of the Fig. 5 grid
- **Parameters**: `arm=d`
- **Switch**: opt-in: `fucina enable osaq`; off: `fucina disable osaq` · file: same size and tensor set -> switched off by going back to the starting GGUF
- **Driver**: `paths.driver_c_ptq` (configuration)

Status: **FAIL**

| obj. | reference host | value | vs | wins/losses | McNemar p | notes | evidence |
|---|---|---|---|---|---|---|---|
| C | (c) | 60.70 | 63.80 | 202/264 | 0.0047 | dKL4 +0.048 z +3.6; Dell 60.75 p 0.004 | stato/cicli.tsv 2026-10-12T09:54, 10:06 |

### `qtea` — QTEA: planes 2 and 3 redone with ternary GPTQ + error decay

- **Phase / pieces**: F1 · attn, gdn, shexp
- **Reference**: arXiv 2609.00224 (QTEA)
- **Parameters**: none
- **Switch**: on in automatic runs; off: `fucina disable qtea` · file: same size and format -> switched off by going back to the starting GGUF

### `scale_groups_g64` — Scale groups redistributed at equal bytes (64 on sensitive tensors, 256 on the rest), same recipe as the host

- **Phase / pieces**: F1 · attn, gdn, shexp
- **Reference**: no paper: Thor-T scale-group redistribution (engine types TQ1_0R_1024_G64/_G256)
- **Variant**: original
- **Parameters**: `plan=piano_g64.json`
- **Switch**: opt-in: `fucina enable scale_groups_g64`; off: `fucina disable scale_groups_g64` · file: same size and tensor set -> switched off by going back to the starting GGUF
- **Driver**: `paths.driver_c_g64` (configuration)

Status: **FAIL**

| obj. | reference host | value | vs | wins/losses | McNemar p | notes | evidence |
|---|---|---|---|---|---|---|---|
| C | (c)+DRQ | tg128 -5.15 % | — | — | — | speed gate on the probe GGUF (ABBAAB) | OBIETTIVO_C/G64_VELOCITA/velocita_1013_1146/esito.json |
| C | (c)+DRQ | acc 62.97 (held-out 883) | 64.67 | 52/67 | — | dKL4 +0.058 z +4.6: not adopted | stato/cicli.tsv 2026-10-13T13:12 · OBIETTIVO_C/G64_BRACCIO_1013/corsa.log |

### `schur` — SchurOpt on the composite of planes 2-3

- **Phase / pieces**: F1 · attn, gdn, shexp
- **Reference**: arXiv 2608.15567 (SchurOpt)
- **Parameters**: none
- **Switch**: on in automatic runs; off: `fucina disable schur` · file: same size and format -> switched off by going back to the starting GGUF
- **Suspended**: suspended by the owner on 8 Oct (relaunch script: stato/coda_sospesa_2026-10-08.sh)

### `schuropt_k1` — SchurOpt scale/code refit between balanced GPTQ K=1 and JARQ

- **Phase / pieces**: F1 · attn, gdn, shexp
- **Reference**: arXiv 2608.15567 (SchurQuant / SchurOpt)
- **Variant**: faithful
- **Parameters**: `arm=e`
- **Switch**: opt-in: `fucina enable schuropt_k1`; off: `fucina disable schuropt_k1` · file: same size and tensor set -> switched off by going back to the starting GGUF
- **Driver**: `paths.driver_c_ptq` (configuration)

Status: **FAIL**

| obj. | reference host | value | vs | wins/losses | McNemar p | notes | evidence |
|---|---|---|---|---|---|---|---|
| C | (c) | 61.90 | 63.80 | 193/231 | 0.072 | dKL4 +0.025 z +2.3 | stato/cicli.tsv 2026-10-12T10:07 |

### `sequential_re_ptq` — Sequential layer-by-layer re-PTQ on the quantized flow: dense tensors, then experts, inputs re-captured

- **Phase / pieces**: F1 · attn, gdn, shexp, experts_gate_up, experts_down
- **Reference**: arXiv 2504.09629 (QEP) with the C recipe (2609.05641 + 2609.38599 + 2610.11226)
- **Variant**: faithful — declared deviations: QEP alpha in {0.5, 1} chosen per tensor on a held-out half; knowledge-weighted Hessian mix
- **Parameters**: none
- **Switch**: opt-in: `fucina enable sequential_re_ptq`; off: `fucina disable sequential_re_ptq` · file: same size and tensor set -> switched off by going back to the starting GGUF
- **Driver**: `paths.driver_c_sequential` (configuration)

Status: **PREPARED**

| obj. | reference host | value | vs | wins/losses | McNemar p | notes | evidence |
|---|---|---|---|---|---|---|---|
| C | - | not run | — | — | — | QEP alpha=1 -38.8 % vs GPTAQ -31.4 % weighted error (CPU) | ricerche/C_PIANO_CASALINGO_1013.md §2 S2+S3 |

### `shq8` — Shared expert in rotated Q8 (RotaryQuant): NOT fully ternary

- **Phase / pieces**: F1 · shexp
- **Reference**: RotaryQuant (lever #18)
- **Parameters**: none
- **Switch**: on in automatic runs; off: `fucina disable shq8` · file: switched off by going back to the starting GGUF

### `ssm_dt_bias` — GatedDeltaNet ssm_dt.bias chosen per head in the exact metric of the recurrent memory (zero bytes)

- **Phase / pieces**: F1 · gdn
- **Reference**: original (derived from bias correction, arXiv 1906.04721, and Gated DeltaNet, arXiv 2412.06464)
- **Variant**: original
- **Parameters**: `corpus=wikitext2`, `rows=32+16`, `ctx=2048`
- **Switch**: opt-in: `fucina enable ssm_dt_bias`; off: `fucina disable ssm_dt_bias` · file: same size and tensor set -> switched off by going back to the starting GGUF
- **Driver**: `paths.driver_c_ssm_dt` (configuration)

Status: **FAIL**

| obj. | reference host | value | vs | wins/losses | McNemar p | notes | evidence |
|---|---|---|---|---|---|---|---|
| C | E2E-QP p3 (64.65) | acc 65.01 (held-out 883) | 64.78 | 14/12 | — | memory metric -58.9 %, 873/960 heads; dKL4 +0.0025 z +0.9: gate not passed, no bench | stato/cicli.tsv 2026-10-13T12:31 · OBIETTIVO_C/M1_SSMDT_1013/corsa.log |

### `three_planes` — Three greedy ternary planes on the dense tensors (R2B3m recipe): the starting structure

- **Phase / pieces**: F1 · attn, gdn, shexp, head
- **Reference**: Thor-T recipe (R2B3m 8,846,968,608 B, MMLU-2000 74.60)
- **Parameters**: none
- **Switch**: on in automatic runs; off: `fucina disable three_planes` · (base structure)

### `tied` — Tied Trit-Planes: 27 uniform levels per group (official solver)

- **Phase / pieces**: F1 · attn, gdn, shexp
- **Reference**: arXiv 2608.08910 (Tied Trit-Planes)
- **Parameters**: none
- **Switch**: on in automatic runs; off: `fucina disable tied` · file: same size and format -> switched off by going back to the starting GGUF

### `zero_byte_scales` — Closed-form scale refit at frozen trits on the asymmetric objective (dense rows/columns, expert gate/up/down)

- **Phase / pieces**: F1 · attn, gdn, shexp, experts_gate_up, experts_down
- **Reference**: arXiv 2510.03267 (PT2-LLM AGA) + arXiv 2410.03129 (ARB-LLM_RC) + arXiv 2609.38599 (JARQ eq. 6) + arXiv 1906.04721 (DFQ) + arXiv 2402.11295 (OneBit)
- **Variant**: faithful — declared deviations: column scales absorbed by the upstream row scales (DFQ-style rescaling) with H-weighted steps; held-out guard per tensor
- **Parameters**: none
- **Switch**: opt-in: `fucina enable zero_byte_scales`; off: `fucina disable zero_byte_scales` · file: same size and tensor set -> switched off by going back to the starting GGUF
- **Driver**: `paths.driver_c_zero_byte` (configuration)

Status: **PREPARED**

| obj. | reference host | value | vs | wins/losses | McNemar p | notes | evidence |
|---|---|---|---|---|---|---|---|
| C | (c)+DRQ | not run | — | — | — | dense prototype -18.6 % weighted error (CPU) | ricerche/C_PIANO_CASALINGO_1013.md §2 S1b |

### `cardan` — CARDAN: shared rank-r basis across experts + per-expert coefficients (lr branches in the cardan engine)

- **Phase / pieces**: F2 · experts_gate_up
- **Reference**: lever #20 CARDAN (G4)
- **Parameters**: `rank=16`
- **Switch**: on in automatic runs; off: `fucina disable cardan` · THOR_SPEGNI_LR=1

### `cardan_experts_k1` — CARDAN (faithful loss and init) on single-plane expert gate/up: KD of a shared low-rank correction in the

- **Phase / pieces**: F2 · experts_gate_up
- **Reference**: arXiv 2609.21137 (CARDAN §3.3 Eq. 3 + App. A.1 Eq. 5-6)
- **Variant**: faithful — declared deviations: tau/lambda not given in the paper (1.0 / 0.5); rank 16; q8 teacher. The published numbers used an optional closed post-processing hook (not part of Fucina) that keeps the file size unchanged
- **Parameters**: `rank=16`, `steps=300`, `tau=1.0`, `lambda_ce=0.5`, `topk=256`, `init_c=a1`
- **Switch**: opt-in: `fucina enable cardan_experts_k1`; off: `fucina disable cardan_experts_k1` · file: back to the starting GGUF (the correction is a separate tensor set unless post-processed)
- **Driver**: `paths.driver_c_cardan_experts` (configuration)
- **Post-processing**: the correction is written as extra tensors; an optional closed external hook (not part of Fucina) may post-process it

Status: **NEUTRAL**

| obj. | reference host | value | vs | wins/losses | McNemar p | notes | evidence |
|---|---|---|---|---|---|---|---|
| C | (c)+DRQ | 65.25 | 64.65 | 89/77 | 0.39 | KL24 -8.62 % (-27.3 se, 24/24 rows); dKL4 -0.006 z -1.3; tg128 87.03 -> 87.48; new C record | stato/cicli.tsv 2026-10-13T18:33 · OBIETTIVO_C/CARDAN_ESPERTI_1013/kd/mmlu_CARDAN.log |

### `k_scale` — K scale: fp16 scales of single-plane experts multiplied by k (zero bytes)

- **Phase / pieces**: F2 · experts_gate_up, experts_down
- **Reference**: R029 item 30 (k 1.045)
- **Parameters**: `k=1.045`
- **Switch**: on in automatic runs; off: `fucina disable k_scale` · file: k=1 (back to the starting GGUF, zero bytes)

### `k_scale_answer_kl4` — K scale on single-plane experts (fp16 scales x k, zero bytes) with k tuned on the answer KL4

- **Phase / pieces**: F2 · experts_gate_up, experts_down
- **Reference**: original (Thor-T lever K, R029 item 30)
- **Variant**: original — declared deviations: one k for every single-plane expert; per-family k not covered
- **Parameters**: `grid=0.98,1.00,1.02,1.03,1.045,1.06`
- **Switch**: opt-in: `fucina enable k_scale_answer_kl4`; off: `fucina disable k_scale_answer_kl4` · file: same size and tensor set -> switched off by going back to the starting GGUF
- **Driver**: `paths.driver_a_k_scale` (configuration)

Status: **PREPARED**

| obj. | reference host | value | vs | wins/losses | McNemar p | notes | evidence |
|---|---|---|---|---|---|---|---|
| A | A record | not run | — | — | — | dry run green; earlier pv12K: KL24 -0.0040 only | stato/cicli.tsv 2026-10-13T19:27 |

### `prism` — Additive PrismQuant: exact rank-k term along the eigenvectors of H_ffn (expert gate/up)

- **Phase / pieces**: F2 · experts_gate_up
- **Reference**: arXiv 2609.32429 (PrismQuant)
- **Parameters**: `k=16`, `projections=gate,up`, `mode=add`
- **Switch**: on in automatic runs; off: `fucina disable prism` · THOR_SPEGNI=prism

### `second_plane_dkl` — Second ternary plane on the experts chosen by measured dKL (se-k 1), up to the objective's size cap

- **Phase / pieces**: F2 · experts_gate_up, experts_down
- **Reference**: rule second_plane_from_dkl (PV12P 78.80 at 12.13 GiB; pv12 78.50)
- **Parameters**: `se_k=1`, `plans=['piano_parte0.json', 'piano_parte1.json', 'piano_parte2.json']`
- **Switch**: on in automatic runs; off: `fucina disable second_plane_dkl` · THOR_SPEGNI=p2d,p2gu

### `e2_rank1` — E2 rank1: debias of the letter rows of the output head (rank-1 bias direction)

- **Phase / pieces**: F3 · head
- **Reference**: arXiv 2409.18857 (E2/BNP: ARC 52.3 -> 56.7)
- **Parameters**: `letter_ids=357,417,351,414`
- **Switch**: on in automatic runs; off: `fucina disable e2_rank1` · file: back to the starting GGUF

### `dense_residual_kl4` — Rank-16 ternary residuals on the dense tensors trained on the answer-token KL (KL4) against the teacher

- **Phase / pieces**: F4 · attn, gdn, shexp
- **Reference**: arXiv 2412.01129 (RILQ) as the form; the loss is ours
- **Variant**: thor — declared deviations: RILQ form with the answer-KL4 loss (V1: full KL + KL4; V2: KL4 only). The step is selected on the same held-out questions as the gate (optimistic gate, declared). Measured with an optional closed post-processing hook (not part of Fucina) that keeps the file size unchanged
- **Parameters**: `rank=16`, `variant=V1`, `steps=200`, `lr=1e-5/1e-4`
- **Switch**: opt-in: `fucina enable dense_residual_kl4`; off: `fucina disable dense_residual_kl4` · file: back to the starting GGUF (the correction is a separate tensor set unless post-processed)
- **Driver**: `paths.driver_a_dense_kl4` (configuration)
- **Post-processing**: the correction is written as extra tensors; an optional closed external hook (not part of Fucina) may post-process it

Status: **NEUTRAL**

| obj. | reference host | value | vs | wins/losses | McNemar p | notes | evidence |
|---|---|---|---|---|---|---|---|
| A | A record (80.30) | 80.20 | 80.30 | 30/32 | 0.90 | held-out gate dKL4 -0.053 z -9.7; KL24 -5.82 %; tg -1.7 %: the KL4 gain does not transfer | stato/cicli.tsv 2026-10-12T09:08 · CARDAN/densi_kl4/mmlu_V1.log |

### `e2e_qp` — EfficientQAT E2E-QP: end-to-end training of the quantization parameters on a generic corpus (codes PV-tuned)

- **Phase / pieces**: F4 · attn, gdn, shexp, experts_gate_up, experts_down
- **Reference**: arXiv 2407.11062 (EfficientQAT E2E-QP) + arXiv 2405.14852 (PV-Tuning) + arXiv 2402.10631 (CAKLD)
- **Variant**: faithful — declared deviations: dose 3 accepted updates of 128 in the paper (2.3 %); beta2 0.95, no clip, no warm-up; patience 2 on MMLU-val without letter preference
- **Parameters**: `lr=0.001`, `lr_f32=1e-05`, `schedule=cosine_steps`, `patience=2`, `corpus=generic`
- **Switch**: opt-in: `fucina enable e2e_qp`; off: `fucina disable e2e_qp` · file: same size and tensor set -> switched off by going back to the starting GGUF
- **Driver**: `paths.driver_c_e2e_qp` (configuration)

Status: **NEUTRAL**

| obj. | reference host | value | vs | wins/losses | McNemar p | notes | evidence |
|---|---|---|---|---|---|---|---|
| C | (c) | 64.65 | 63.80 | 66/49 | 0.135 | best step 3; dKL4 -0.018 z -5.1 | stato/cicli.tsv 2026-10-13T09:07 · E2E_C3/mmlu_E2E-C3-p3.log |
| C | (c) | 63.60 | 63.80 | 161/165 | 0.87 | Thor variant (multiple-choice rows + letter step): moves letter preferences, not knowledge — refuted | stato/cicli.tsv 2026-10-12T19:58, 21:27 · E2E_C2/mmlu_E2E-C2-p2.log |

### `on_policy_distillation` — Quantization-aware on-policy distillation: the student generates, the teacher scores its own trajectories

- **Phase / pieces**: F4 · whole model
- **Reference**: arXiv 2609.26708 (QAOPD, official reward math_mixed_reward.py)
- **Variant**: faithful — declared deviations: faithful driver: official strict reward, fresh rollouts every round, 2 optimizer steps per round (trainer constraint); first round ran with stale rollouts and a non-strict reward
- **Parameters**: `rounds=10`, `steps_per_round=2`, `rows=4`, `lr=0.00015`
- **Switch**: opt-in: `fucina enable on_policy_distillation`; off: `fucina disable on_policy_distillation` · file: same size and tensor set -> switched off by going back to the starting GGUF
- **Driver**: `paths.driver_c_on_policy` (configuration)

Status: **NEUTRAL**

| obj. | reference host | value | vs | wins/losses | McNemar p | notes | evidence |
|---|---|---|---|---|---|---|---|
| C | E2E-QP p3 | 64.60 | 64.65 | 56/57 | 1.0 | spoken probe 10/21 vs 5/21 (after the probe-count fix); GSM8K held-out 64.1 -> 67.2 n.s. | stato/cicli.tsv 2026-10-13T14:42 · OBIETTIVO_C/ONPOLICY/corsa.log |

### `rilq` — RILQ-T: rank-64 ternary residuals on attn+gdn(+shared expert) trained on the full KL against the teacher

- **Phase / pieces**: F4 · attn, gdn, shexp
- **Reference**: arXiv 2412.01129 (RILQ: -51 % of the log-PPL gap at W2); reproduced small-scale -52.3 % (audit A5)
- **Parameters**: `hours=10`, `families=attn,gdn,shexp`, `min_steps=200`
- **Switch**: on in automatic runs; off: `fucina disable rilq` · THOR_SPEGNI=lr (lrattn,lrgdn,lrshexp)

### `tesseraq` — TesseraQ: block reconstruction with progressive adaptive rounding (PAR) + dequantization scale tuning (DST)

- **Phase / pieces**: F4 · attn, gdn, shexp, experts_gate_up, experts_down
- **Reference**: arXiv 2410.19103 (TesseraQ, Alg. 1)
- **Variant**: faithful — declared deviations: Thor arm: start from the host codes, harden the most confident first, AdamW on the scales
- **Parameters**: `rounds=20`, `steps=250`, `lr=0.001`
- **Switch**: opt-in: `fucina enable tesseraq`; off: `fucina disable tesseraq` · file: same size and tensor set -> switched off by going back to the starting GGUF
- **Driver**: `paths.driver_c_tesseraq` (configuration)

Status: **PREPARED**

| obj. | reference host | value | vs | wins/losses | McNemar p | notes | evidence |
|---|---|---|---|---|---|---|---|
| C | - | not run | — | — | — | prototype -3.6 % / -2.7 % squared error (CPU) | ricerche/C_LEVE_DISTILLAZIONE_LEGGERA_1013.md §1 |

### `int8` — int8 activations in the engine (ternary mat-vec with Q8 activations)

- **Phase / pieces**: F5 · whole model
- **Reference**: in-house lever (GGML_VK_TQ1R_Q8): MMLU 74.65 vs 74.60, tg 50.4 -> 56.3
- **Parameters**: none
- **Switch**: on in automatic runs; off: `fucina disable int8` · GGML_VK_TQ1R_Q8=0

### `mtp` — MTP head: speculative decoding (acceptance and tok/s)

- **Phase / pieces**: F5 · mtp
- **Reference**: plan of 8 Oct: MTP head to be measured (acceptance and tok/s) on PV12P/RILQ
- **Parameters**: none
- **Switch**: on in automatic runs; off: `fucina disable mtp` · llama-server without --draft (MTP off)

