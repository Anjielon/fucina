<h1 align="center">FUCINA</h1>
<p align="center"><em>Measured, per-tensor-family ternary recipes for pre-trained language models</em></p>

<p align="center">
  <a href="#license"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
  <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-3776AB.svg">
  <img alt="Status: research" src="https://img.shields.io/badge/status-research-orange.svg">
</p>

---

> **Name.** This Fucina is a quantization *forge* for ternary language models. It is unrelated to
> [matteo-grella/fucina](https://github.com/matteo-grella/fucina), an LLM inference engine with the same name.

**Fucina** (Italian for *forge*) builds a ternary version (`T ∈ {-1, 0, +1}`) of a pre-trained language model by
deciding, **for each family of tensors**, which quantization techniques measurably help, in which order, and with
which parameters — and by recording every measurement so that the final recipe can be replayed with one command on a
single 128 GB machine.

It is not one more quantizer. It is the *engine around* quantizers: it knows the pipeline phases, asks a database of
past measurements before spending GPU time, queues the experiments that are still missing behind a machine-wide GPU
lock, judges each technique against explicit thresholds (realised KL against the teacher, paired MMLU with McNemar,
generation speed, file size), and asks you when a result is ambiguous instead of applying it silently.

The full design is in [`docs/SPEC.md`](docs/SPEC.md).

## What it does

Given a model (dense, MoE, hybrid attention/GatedDeltaNet, ...) and an objective such as
`mmlu>=80.05,file<=13085779616,tg>=50` or `ram<=9GiB,tg>=50`, Fucina produces:

- a **recipe per tensor family** — techniques, parameters, order — in `recipe.yaml`;
- the **job queue** that builds and measures the forged GGUF model, one GPU job at a time;
- a **measurement log** (`measurements.sqlite`) that justifies every choice, failures included;
- a **forecast** of the final quality through a fitted KL→MMLU law, when the model has one.

Three modes of execution, for every command that queues work:

| mode | flag | what happens |
|---|---|---|
| simulate | `--simulate` (default) | no process at all; outcomes come from the measurement database |
| dry run | `--dry-run` | the launchable jobs are written, in dependency order, to `dry_run_commands.sh` |
| run | `--run` | jobs are launched as `systemd-run --user` units; GPU jobs take the machine-wide lock |

## Concepts

- **Piece** — a family of tensors with the same role, selected by GGUF tensor name: `attn`, `gdn` (GatedDeltaNet),
  `ffn_dense`, `shexp` (shared expert), `experts_gate_up`, `experts_down`, `router`, `embedding`, `head`, `mtp`.
- **Technique** — a self-registering module (`fucina/techniques/`) with a declared scope (phase, pieces, model types,
  architectures), cost, thresholds, provenance and a **switch** that turns it off without rebuilding the model
  (an engine environment flag, or "go back to the previous GGUF"). Existing, validated scripts are *wrapped*, not
  rewritten.
- **Phase** — `F0 diagnosis` → `F1 dense` → `F2 experts` → `F3 head` → `F4 realignment` (run once, on top of the
  final structure) → `F5 runtime` → `F6 evaluation`.
- **Candidate** — the current model plus the techniques switched on. Structures (three planes, a dKL-selected second
  plane) are consolidated at the end of their phase; per-byte levers are combined on top of the structure and measured
  all-on and all-but-one to expose interactions.
- **Verdict** — `PASS` (consolidated), `FAIL` (switched off with its cause) or `AMBIGUOUS` (a question for you, with
  the numbers and an advice). A published technique that underperforms its paper is marked **to verify**, never
  silently rejected.
- **Rules** — generalisations learned from past measurements (`fucina/rules/*.yaml`), with scope, evidence and
  confidence; they weight priorities and can consolidate a technique from strong evidence.

## Technique modules

Every lever is a module with a paper reference, a declared variant, parameters, a switch and its measured status.
The full census — parameters, deviations from the paper and every measured number with its evidence file — is in
[`docs/LEVERS.md`](docs/LEVERS.md), generated with `fucina techniques --census`. *Opt-in* modules (most of them
added on 12-13 Oct 2026 for the single-plane objective C and the finishing levers of A and B) are skipped by
automatic runs until `fucina enable <name>`.

| module | phase | pieces | reference | variant | status | default |
|---|---|---|---|---|---|---|
| `flow_statistics` | F0 | attn, gdn, shexp, experts_gate_up, experts_down | arXiv 2504.02692 (GPTAQ) + arXiv 2504.09629 (QEP) + arXiv 2505.03804 (MoEQuant AGQ) + arXiv 2609.31009 (G2PTQ gradients) | faithful | PREPARED | opt-in |
| `measured_dkl` | F0 | experts_gate_up, experts_down | rule second_plane_from_dkl (RD1/ALLOC1 from an imatrix proxy: -1.8 MMLU) | — | — | on |
| `act_order_sign` | F1 | attn, gdn, shexp | arXiv 2605.18933 (sign criterion) + static act-order (arXiv 2210.17323 §4) | faithful | NEUTRAL | opt-in |
| `bil` | F1 | attn, gdn, shexp, head | arXiv 2609.05641 (nuclear limit -> uniform grid); offline weighted error 0.054 -> 0.026 | — | — | on |
| `bil_k1_jarq` | F1 | attn, gdn, shexp | arXiv 2609.05641 (balanced grid) + arXiv 2609.38599 (JARQ) | faithful | PASS | opt-in |
| `dense_ptq_before_expert_kd` | F1 | attn, gdn, shexp | arXiv 2608.15567 (SchurQuant) + arXiv 2609.31009 (G2PTQ), then arXiv 2609.21137 (CARDAN) | faithful | PREPARED | opt-in |
| `drq` | F1 | attn, gdn, shexp | arXiv 2610.11226 (DRQ; paper +0.91 on a 3-bit MoE); 13/13 closed-form tests (audit 13 Oct) | faithful | PASS | opt-in |
| `e2m_stage2` | F1 | attn, gdn, shexp | arXiv 2606.13054 (TWLA §3.2 / Alg. 3) | faithful | PREPARED | opt-in |
| `fair_calib` | F1 | attn, gdn, shexp | arXiv 2606.06547 (FAIR-Calib, ICML 2026) | faithful | RUNNING | opt-in |
| `family_lever_search` | F1 | attn, gdn, shexp | original (K-FAC curvature, arXiv 1503.05671, as the predictor) | original | RUNNING | opt-in |
| `foem` | F1 | attn, gdn, shexp | arXiv 2507.11017 (FOEM); bit-identical to GPTQModel foem.py (audit 13 Oct) | faithful | NEUTRAL | opt-in |
| `g2ptq` | F1 | attn, gdn, shexp | G2PTQ (official code, audit G2 #17) | — | SUSPENDED | suspended |
| `gptq_k1` | F1 | attn, gdn, shexp | arXiv 2210.17323 (GPTQ) | control | FAIL | opt-in |
| `hsvdquant` | F1 | attn, gdn, shexp | arXiv 2610.11315 (H-SVDQuant) | faithful | PREPARED | opt-in |
| `jarq` | F1 | attn, gdn, shexp | arXiv 2609.38599 (JARQ) | — | — | on |
| `jarq_k1` | F1 | attn, gdn, shexp | arXiv 2609.38599 (JARQ, Alg. 1) | faithful | PASS | opt-in |
| `osaq` | F1 | attn, gdn, shexp | arXiv 2605.04738 (OSAQ) | faithful | FAIL | opt-in |
| `qtea` | F1 | attn, gdn, shexp | arXiv 2609.00224 (QTEA) | — | — | on |
| `scale_groups_g64` | F1 | attn, gdn, shexp | no paper: Thor-T scale-group redistribution (engine types TQ1_0R_1024_G64/_G256) | original | FAIL | opt-in |
| `schur` | F1 | attn, gdn, shexp | arXiv 2608.15567 (SchurOpt) | — | SUSPENDED | suspended |
| `schuropt_k1` | F1 | attn, gdn, shexp | arXiv 2608.15567 (SchurQuant / SchurOpt) | faithful | FAIL | opt-in |
| `sequential_re_ptq` | F1 | attn, gdn, shexp, experts_gate_up, experts_down | arXiv 2504.09629 (QEP) with the C recipe (2609.05641 + 2609.38599 + 2610.11226) | faithful | PREPARED | opt-in |
| `shq8` | F1 | shexp | RotaryQuant (lever #18) | — | — | on |
| `ssm_dt_bias` | F1 | gdn | original (derived from bias correction, arXiv 1906.04721, and Gated DeltaNet, arXiv 2412.06464) | original | FAIL | opt-in |
| `three_planes` | F1 | attn, gdn, shexp, head | Thor-T recipe (R2B3m 8,846,968,608 B, MMLU-2000 74.60) | — | — | on |
| `tied` | F1 | attn, gdn, shexp | arXiv 2608.08910 (Tied Trit-Planes) | — | — | on |
| `zero_byte_scales` | F1 | attn, gdn, shexp, experts_gate_up, experts_down | arXiv 2510.03267 (PT2-LLM AGA) + arXiv 2410.03129 (ARB-LLM_RC) + arXiv 2609.38599 (JARQ eq. 6) + arXiv 1906.04721 (DFQ) + arXiv 2402.11295 (OneBit) | faithful | PREPARED | opt-in |
| `cardan` | F2 | experts_gate_up | lever #20 CARDAN (G4) | — | — | on |
| `cardan_experts_k1` | F2 | experts_gate_up | arXiv 2609.21137 (CARDAN §3.3 Eq. 3 + App. A.1 Eq. 5-6) | faithful | NEUTRAL | opt-in |
| `k_scale` | F2 | experts_gate_up, experts_down | R029 item 30 (k 1.045) | — | — | on |
| `k_scale_answer_kl4` | F2 | experts_gate_up, experts_down | original (Thor-T lever K, R029 item 30) | original | PREPARED | opt-in |
| `prism` | F2 | experts_gate_up | arXiv 2609.32429 (PrismQuant) | — | — | on |
| `second_plane_dkl` | F2 | experts_gate_up, experts_down | rule second_plane_from_dkl (PV12P 78.80 at 12.13 GiB; pv12 78.50) | — | — | on |
| `e2_rank1` | F3 | head | arXiv 2409.18857 (E2/BNP: ARC 52.3 -> 56.7) | — | — | on |
| `dense_residual_kl4` | F4 | attn, gdn, shexp | arXiv 2412.01129 (RILQ) as the form; the loss is ours | thor | NEUTRAL | opt-in |
| `e2e_qp` | F4 | attn, gdn, shexp, experts_gate_up, experts_down | arXiv 2407.11062 (EfficientQAT E2E-QP) + arXiv 2405.14852 (PV-Tuning) + arXiv 2402.10631 (CAKLD) | faithful | NEUTRAL | opt-in |
| `on_policy_distillation` | F4 | whole model | arXiv 2609.26708 (QAOPD, official reward math_mixed_reward.py) | faithful | NEUTRAL | opt-in |
| `rilq` | F4 | attn, gdn, shexp | arXiv 2412.01129 (RILQ: -51 % of the log-PPL gap at W2); reproduced small-scale -52.3 % (audit A5) | — | — | on |
| `tesseraq` | F4 | attn, gdn, shexp, experts_gate_up, experts_down | arXiv 2410.19103 (TesseraQ, Alg. 1) | faithful | PREPARED | opt-in |
| `int8` | F5 | whole model | in-house lever (GGML_VK_TQ1R_Q8): MMLU 74.65 vs 74.60, tg 50.4 -> 56.3 | — | — | on |
| `mtp` | F5 | mtp | plan of 8 Oct: MTP head to be measured (acceptance and tok/s) on PV12P/RILQ | — | — | on |

## Replaying a published chain

`fucina replay` re-runs a recorded chain from public inputs and checks every step against the record. The reference is
the chain from the released frozen root of Thor-T to **C**, the trunk of `Thor-T-35B-A3B-XS`
([`examples/thor/recipe_c.yaml`](examples/thor/recipe_c.yaml), documentation in [`docs/REPLAY.md`](docs/REPLAY.md)):

```bash
fucina replay examples/thor/recipe_c.yaml --dry-run --set work=/data/thor-c --set engine=/opt/llama.cpp/build/bin \
    --set bundle_src=/data/thor-c-code --set venv=/data/venv      # plan + checks, nothing runs
fucina replay examples/thor/recipe_c.yaml --run   --set ...      # run, verify, resume
```

The dry run shows, per step, whether it runs from code in this package, from the frozen research code bundle, or is
still blocked (and why), with the estimated GPU hours.

## Installation

```bash
git clone https://github.com/Anjielon/fucina.git
cd fucina
python -m pip install -e ".[tui,test]"     # core needs only pyyaml + rich; textual for the TUI
```

Python 3.10+ on Linux. Simulation and dry runs need no GPU. Real runs need the inference engine and the wrapped
scripts referenced by your configuration, plus `systemd --user`.

Without installing, the development launcher works from the checkout: `bin/fucina ...` (uses `$FUCINA_PY` or
`python3`, at `nice 15`).

## Configuration

Nothing machine-specific lives in the package. Every location (engines, scripts, teacher caches, output directory,
GPU lock) comes from a YAML file, found via `--config FILE`, `$FUCINA_CONFIG` or `./fucina.yaml`.

- [`config.example.yaml`](config.example.yaml) documents every key;
- [`examples/thor/`](examples/thor) is the complete configuration of the reference machine (Ryzen AI Max+ 395,
  gfx1151, Vulkan, 128 GB UMA) and the model description of **Thor-T** (Ornith-1.5-35B-A3B, `qwen35moe` hybrid
  GatedDeltaNet + MoE), the model the default phase order and rule base were learned on.

A model file describes the model type, architecture, starting checkpoint, pieces, named objectives, the candidates
already built (used to recognise historic measurements) and, optionally, its fitted KL→MMLU law.

## Quickstart

```bash
# 1. list the techniques and what each one still misses on this machine
fucina --config examples/thor/config.yaml techniques

# 2. pre-load the measurement database from the lab notebook (Thor-T sources)
fucina --config examples/thor/config.yaml import

# 3. plan objective A without running anything: the full job queue is written to dry_run_commands.sh
fucina --config examples/thor/config.yaml auto --objective A --dry-run

# 4. inspect, answer the ambiguous questions, override the engine
fucina --config examples/thor/config.yaml status
fucina --config examples/thor/config.yaml decide                       # interactive
fucina --config examples/thor/config.yaml disable prism                # switches win over the engine
fucina --config examples/thor/config.yaml tui                          # tree of phases → techniques

# 5. launch for real (GPU jobs wait for the machine-wide lock), then replay the final recipe elsewhere
fucina --config examples/thor/config.yaml auto --objective A --run
fucina --config examples/thor/config.yaml forge --recipe runs/thor-A-run/recipe.yaml --dry-run
```

Manual mode advises on a single piece and queues techniques in series or in parallel:

```bash
export FUCINA_CONFIG=examples/thor/config.yaml
fucina manual --piece experts_gate_up --parallel prism,cardan --dry-run
```

Every run directory is resumable: `state.json`, `queue.json`, `questions.json`, `switches.json`, `report.json`,
`recipe.yaml`. Running `fucina auto` again recomputes the decisions from the database and the finished jobs.

## Project layout

```
fucina/
  cli.py            command-line interface (auto, manual, status, decide, enable/disable/release, rules,
                    measurements, import, techniques, forge, tui)
  engine.py         decision engine: phases, structures, combination, ablations, forecast, report, recipe
  decision.py       pure rules: objective parsing, threshold verdicts, KL→MMLU law, priority
  oracle.py         reads past measurements against the current candidate
  measure.py        paired measurements and readers of the measurement scripts' outputs
  executor.py       job queue, simulate / dry-run / systemd-run execution, GPU-lock aware
  registry.py       technique contract and self-registering registry
  candidate.py      candidate and resumable run state
  db.py             measurement database and rule base
  config.py         configuration loading (no hard-coded paths)
  phases.py, pieces.py
  techniques/       wrappers: diagnosis (F0), dense (F1), experts (F2), head/realignment/runtime (F3-F5),
                    single_plane (objective C), finishing (A/B levers and the per-family lever search)
  census.py         lever census renderer (docs/LEVERS.md)
  importers/        loaders of existing lab records into the measurement database
  rules/            packaged rule base (learned on Thor-T)
  tools/            small CLI adapters used inside jobs
  view.py, tui.py   rich report and textual UI
  kernels/          Triton GPU kernels of the ternary student (bf16x3 and hi/lo split products, flash attention,
                    GDN recurrence, direct convolution, folded dense planes, fused trit sweep), with CPU fallbacks
  student/          end-to-end ternary student (scales trained against the teacher) and plane-code optimizer
  ternary/          TQ1_0 / TQ1_0R_1024 codecs, rotation, GGUF dequantization, GGUF <-> HF map (data/)
  sequential/       layer-by-layer re-quantization of dense tensors and experts (stages S2+S3) and GGUF writer
  extdeps.py        environment variables locating external dependencies and data (no hard-coded paths)
examples/thor/      reference machine configuration and model files
docs/SPEC.md        design specification
docs/LEVERS.md      lever census (generated)
docs/GLOSSARY_IT_EN.md  names of the original (Italian) code base -> names in this package
tests/              unit tests + Thor-T integration tests (skipped when its sources are absent)
```

## Tests

```bash
python -m pytest            # or: PYTHONDONTWRITEBYTECODE=1 nice -n 15 python -m pytest -q
```

The unit tests need nothing else. The Thor-T integration tests (simulated runs, dry run, TUI pilot, CLI) read the
reference lab notebook and GGUF files and are skipped on machines that do not have them; they freeze the activity log
at a fixed timestamp so that they test the engine, not the latest experiments.

### Port equivalence

`kernels/`, `student/`, `ternary/` and `sequential/` are a pure rename of the code that produced the Thor-T results
(`docs/GLOSSARY_IT_EN.md`). Two checks prove it against the original tree: `scripts/check_port_ast.py` (syntax
trees identical after the rename) and `tests/test_kernels_equivalence.py` (bit-identical outputs on CPU and GPU).
Both are run by `scripts/run_equivalence_after_passata.sh`, configured through `scripts/equivalence_paths.env`
(copy `scripts/equivalence_paths.env.example`); the numerical tests are skipped when the original tree is absent.

## Status and limitations

- The technique wrappers call scripts validated on `qwen35moe`; other architectures are recognised and excluded with
  a reason until a wrapper exists (see `examples/thor/models/qwen3-0.6b.yaml`).
- The bundled importers parse the Thor-T lab notebook; other projects will write their own importer or record
  measurements through the engine.
- The final-evaluation step relies on the same measurement script; MTP acceptance measurement is not written yet.

## License

[MIT](LICENSE) © 2026 Angelo Mogavero.
