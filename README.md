<h1 align="center">FUCINA</h1>
<p align="center"><em>Measured, per-tensor-family ternary recipes for pre-trained language models</em></p>

<p align="center">
  <a href="#license"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
  <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-3776AB.svg">
  <img alt="Status: research" src="https://img.shields.io/badge/status-research-orange.svg">
</p>

---

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
  techniques/       wrappers: diagnosis (F0), dense (F1), experts (F2), head/realignment/runtime (F3-F5)
  importers/        loaders of existing lab records into the measurement database
  rules/            packaged rule base (learned on Thor-T)
  tools/            small CLI adapters used inside jobs
  view.py, tui.py   rich report and textual UI
examples/thor/      reference machine configuration and model files
docs/SPEC.md        design specification
tests/              unit tests + Thor-T integration tests (skipped when its sources are absent)
```

## Tests

```bash
python -m pytest            # or: PYTHONDONTWRITEBYTECODE=1 nice -n 15 python -m pytest -q
```

The unit tests need nothing else. The Thor-T integration tests (simulated runs, dry run, TUI pilot, CLI) read the
reference lab notebook and GGUF files and are skipped on machines that do not have them; they freeze the activity log
at a fixed timestamp so that they test the engine, not the latest experiments.

## Status and limitations

- The technique wrappers call scripts validated on `qwen35moe`; other architectures are recognised and excluded with
  a reason until a wrapper exists (see `examples/thor/models/qwen3-0.6b.yaml`).
- The bundled importers parse the Thor-T lab notebook; other projects will write their own importer or record
  measurements through the engine.
- The final-evaluation step relies on the same measurement script; MTP acceptance measurement is not written yet.

## License

[MIT](LICENSE) © 2026 Angelo Mogavero.
