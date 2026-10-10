# Fucina — design specification

Fucina ("forge" in Italian) builds a ternary version of a pre-trained language model by choosing, for each family
of tensors, the techniques that measurably help, in the order that works, and by recording every measurement so
that the recipe can be replayed on a single 128 GB machine.

## 1. Goals
Given a model (dense, MoE, hybrid attention/GatedDeltaNet, ...) and a target (for example `MMLU >= 80.05`,
`resident RAM <= 9 GiB`, `generation >= 50 tok/s`), Fucina produces:
- a **recipe per tensor family** (techniques, parameters, order);
- the **forged model** (GGUF) that implements it;
- a **measurement log** that justifies every choice.

Three modes:
- **Automatic** — techniques are chosen per piece from the phase, the model type, the piece type and the rule base.
- **Manual** — the user picks techniques per piece, in series or in parallel; Fucina advises.
- **Ambiguous results** never get applied silently: Fucina states its advice and the numbers, the user decides.

## 2. Concepts
- **Piece** — a family of tensors with the same role, selected by GGUF tensor name (attention, GatedDeltaNet, dense FFN,
  shared expert, expert gate/up, expert down, router, embedding, output head, MTP head).
- **Technique** — a self-registering module with a declared scope, cost and thresholds, and a **switch** (an engine flag
  or optional tensors) that turns it off without rebuilding the model.
- **Phase** — a step of the pipeline. Default order, learned on Thor-T:
  1. `F0 diagnosis` — per-piece sensitivity (measured KL gradient, importance matrix, kurtosis, routing frequency);
  2. `F1 dense` — quantizer for dense tensors (plane count, uniform grids, joint refinement);
  3. `F2 experts` — extra capacity on experts (KL-selected second planes, shared bases, corrections);
  4. `F3 head` — embedding, output head, MTP head;
  5. `F4 realignment` — whole-model realignment against the teacher (ternary low-rank adapters, distillation),
     run **once**, on top of the final structure;
  6. `F5 runtime` — int8 activations, MTP acceptance and throughput;
  7. `F6 evaluation` — spoken probe, paired MMLU, reasoning benchmarks, speed, resident RAM.
- **Candidate** — the current model plus the list of switched-on techniques.

## 3. Technique contract
Each technique lives in `fucina/techniques/<name>.py` and registers itself. It declares:
- scope: phase, admissible pieces, model types, architectures;
- constraints: stays ternary, estimated extra bytes, GPU cost, speed impact;
- thresholds: minimum KL reduction, significance (standard errors), maximum speed loss;
- provenance: paper reference and reported number.

It implements `cheap_test()` (CPU, minutes), `apply(piece)` and `switch()`. Existing, validated scripts are wrapped
rather than rewritten.

## 4. Decision engine
For each phase and piece — sequentially across phases, in parallel across alternative techniques:
1. filter admissible techniques (scope and hard constraints: RAM, ternary, speed);
2. rank them by the rule base and by expected gain per GPU hour;
3. run the cheap test and drop techniques that carry no information;
4. apply the survivors in parallel, each behind its switch, and measure each one **alone** against the current
   candidate: realised whole-model KL on held-out teacher text, generation/prompt speed, resident RAM;
5. verdict by threshold: clear pass → consolidated; clear fail → switched off with its cause; ambiguous → question to the user;
6. combinations: passing techniques are switched on together, then all-on and all-but-one are measured to expose interactions;
7. at the end of a phase the consolidated candidate gets a paired MMLU evaluation.

No long training before the structure is fixed.

## 5. Rule base
- **Measurements** (`measurements.sqlite`) — one row per experiment: model, piece, technique, parameters, KL, MMLU,
  McNemar p-value, speed, RAM, file hash, command, duration. Nothing is discarded, failures included.
- **Rules** (`fucina/rules/*.yaml`, plus the configured `rules_dirs`) — generalisations with scope, evidence (measurement rows), confidence and exceptions,
  for example: local quantizer refinements do not move MMLU without whole-model realignment; extra expert planes must be
  selected by measured KL, not by proxies; realignment quality depends on the number of steps.
- A published technique that underperforms its reported number is marked **to verify**, not rejected, until its number is
  reproduced or the difference is explained.

## 6. Execution
- Single GPU lock; light measurements may share the GPU within a memory budget; never two trainings at once;
  CPU-only file writing happens outside the lock.
- Resumable state after every measurement.
- The final recipe is exported as `recipe.yaml` and replayed with one command, with measured timings.

## 7. Interface
- CLI: `fucina auto`, `fucina manual`, `fucina status`, `fucina rules`, `fucina decide`.
- Terminal UI: a tree of phases → pieces → techniques with their state (queued, testing, passed, off, to decide),
  progress, latest numbers (KL, MMLU, speed, RAM) and the GPU queue; keys to toggle techniques and answer questions.
