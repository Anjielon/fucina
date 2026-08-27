# Design

Fucina is an automated ternary quantization pipeline for Mixture-of-Experts
language models. Given a model and a memory budget, it derives a quantization
plan for *that specific model* by empirically evaluating each candidate
technique before committing to it.

This document describes the architecture and the reasoning behind it. For the
catalogue of techniques and their measurements see [LEVERS.md](LEVERS.md); for
the engineering constraints discovered during development see
[LESSONS.md](LESSONS.md).

---

## 1. Motivation

Published quantization results are reported on the models their authors
tested. Transferring a technique to a different architecture is not reliable:
in our own measurements on a 397B hybrid-attention MoE model, four techniques
with strong published results were actively harmful, while one unpublished
combination produced the single largest improvement.

| technique | published result | measured here |
|---|---|---|
| rotation into the Hessian eigenbasis | large gains for codebook quantizers | **−14%** (harmful) |
| MagR outlier reduction | consistent gains with abs-max quantizers | **39% → 46%** (harmful) |
| sensitivity-based expert allocation | preferred over frequency (MoPEQ) | **loses to frequency** (32.6% vs 38.2%) |
| reusing a joint plane in isolation | not discussed in the literature | **−10 points** |

None of these results contradict the original papers. They reflect a different
weight distribution, a different quantizer, and a different bit regime. The
practical consequence is that a fixed recipe cannot be optimal across models —
which is the problem this pipeline addresses.

**Design principle.** Every technique carries an executable test that runs in
minutes on a sample of the current model. A technique is applied only if it
wins that test, and the result is recorded so that the registry accumulates
evidence across models rather than assumptions.

---

## 2. Pipeline

```
  ┌─────────┐   ┌─────────┐   ┌──────┐   ┌──────┐   ┌───────┐   ┌──────────┐
  │ inspect │──►│ measure │──►│ race │──►│ plan │──►│ forge │──►│ validate │
  └─────────┘   └─────────┘   └──────┘   └──────┘   └───────┘   └──────────┘
    minutes      hours          hours     seconds     hours       hours
                 (once per
                  model)
       │             │            │          │           │            │
       ▼             ▼            ▼          ▼           ▼            ▼
   topology     Hessians,     per-lever   tensor    two-plane    perplexity,
   and budget   routing       gain on a   type map  TQ1_0 with   task suite,
   arithmetic   counts,       sample                a resumable  routing
                super weights                       journal      agreement
```

Each stage persists `forge-state.json`, making the pipeline resumable and
inspectable. This is not a convenience: a full forge of a 397B model takes
several hours, and an unrecoverable failure at hour six is unacceptable.

### 2.1 Inspect

Reads the model topology (expert count, top-k, shared experts, attention
type per layer, fused tensors, auxiliary prediction heads) and derives the
budget arithmetic: available bits per weight given the target memory, after
accounting for tensors that must remain high precision.

### 2.2 Measure

Collected once per model and reused by every subsequent stage:

- **Per-layer Hessians** `H = XᵀX`, accumulated from residual-stream
  activations, with the spectral anisotropy α reported per layer. A flat
  spectrum indicates the Hessian carries little exploitable structure.
- **Routing counts** per expert, from the importance matrix or router
  telemetry.
- **Per-layer sensitivity trace** and a **super-weight scan** (two forward
  passes, looking for activations an order of magnitude above the median).
- A calibration corpus. Size matters more than is usually acknowledged: with
  2,000 tokens, 35 of 60 layers appeared to benefit from a repair; with 7,500
  tokens, only 22 did, and applying all of them degraded the model.

### 2.3 Race

For each applicable lever, on a representative sample (several layers × both
matrix families × hot, warm and cold experts):

```
gain = output_error(without lever) − output_error(with lever)
```

evaluated under the true Hessian. Levers are ranked by gain per GiB and
selected along the Pareto front. Where multiple solvers exist for the same
objective, they compete directly.

### 2.4 Plan

| profile | objective |
|---|---|
| `quality` | maximum fidelity within budget: second plane on hot experts, protected tensors at Q8 |
| `speed` | fully resident, minimum bit width, speculative head retained |
| `streaming` | both planes retained, hot/cold partition sized for the expert cache |

Constraints applied unconditionally, each supported by our measurements and at
least two independent sources:

- routers and normalization parameters are never quantized;
- the recurrent and attention path is never taken below Q6–Q8 (no sub-4-bit
  state-space scan has been demonstrated to work by anyone, ourselves included);
- `down` projections are allocated one step above `gate`/`up`; first and last
  layers one step above the middle;
- identified super weights are restored at half precision.

### 2.5 Forge

Layer-by-layer streaming reads (two concurrent streams; source bandwidth is
measured first, since more streams often reduce throughput), GPU quantization,
and sequential writes guarded by a journal recording tensor index and byte
offset. A crash costs one tensor, never the run. Memory limits are enforced at
the service level rather than trusted to the process.

### 2.6 Validate

The outcome is measured, never inferred:

1. KL divergence against the source model as a fast filter;
2. task fixtures with verifiable answers;
3. divergence at 32 tokens, routing-flip agreement, calibration under
   uncertainty, differential long-context retrieval.

Perplexity is reported **with its uncertainty**, and comparisons are only made
at equal corpus and equal chunk count. A repair is accepted only when the
improvement exceeds that uncertainty — a rule adopted after a change was
briefly credited with a 0.0026 improvement against a measurement noise of
±0.0874.

---

## 3. Dense models and edge deployment

The pipeline is not limited to frontier-scale MoE models. Dense models in the
20–30B range are effectively un-ternarizable with stock tooling (default TQ1_0
conversion yields roughly 73% weight error); with this stack they become
viable on mobile hardware.

Worked example, a 27B dense model (27.32B parameters):

| component | parameters | assigned type |
|---|---|---|
| FFN down/gate/up | 17.4B (64%) | ternary (TQ2_0; ARM kernels are upstream) |
| attention + gated delta net | 7.3B | Q4_K–Q6 (recurrence constraint) |
| embeddings and output head | 2.5B | Q6 |

Resulting size: **9.5–11 GiB**, within a 16 GB device, where the standard Q4
build (16.5 GiB) is not. Throughput follows memory bandwidth: at 50–70 GB/s
and roughly 10 GiB read per token, **5–7 tok/s**, or **7–10 tok/s** with the
model's speculative head.

Adaptations required for dense architectures: expert-frequency levers do not
apply, so allocation shifts to per-layer sensitivity and per-family rules; the
super-weight scan becomes mandatory, as super weights concentrate in early
`down` projections; the second plane reduces to a sum of two matrix products,
simpler than the masked expert path. On ARM, ternary weights pay twice — in
memory and in compute — since the kernels avoid multiplication entirely.

Dense models carry less redundancy than MoE models, so a larger relative
quality loss should be expected. The lever race is correspondingly more
important, not less.

---

## 4. Status and scope

The pipeline has produced one complete model: a 397B MoE quantized to 88 GiB,
all 512 experts retained, outperforming the best published sub-2-bit
quantization of the same model on perplexity. The stages are implemented at
different depths — quantization, forging and validation are production code;
the orchestrator currently coordinates them rather than fully automating the
race. Architecture support is limited to the families we have tested.

Contributions of new model families are the most useful thing an outside user
can provide: every model measured enriches the registry, and levers win or
lose per model.
