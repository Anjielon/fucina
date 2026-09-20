<p align="center"><img src="assets/odino_logo.png" alt="ODINO" width="260"/></p>

<h1 align="center">FUCINA</h1>
<p align="center"><em>A general ternary forge for Mixture-of-Experts models</em></p>

<p align="center">
  <a href="#license"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
  <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-3776AB.svg">
  <img alt="Built on llama.cpp" src="https://img.shields.io/badge/built%20on-llama.cpp%20%2F%20ggml-lightgrey.svg">
  <img alt="Backend: Vulkan" src="https://img.shields.io/badge/backend-Vulkan-A41E22.svg">
  <a href="paper/main.pdf"><img alt="Paper: draft" src="https://img.shields.io/badge/paper-draft%20in%20repo-informational.svg"></a>
  <img alt="Preprint pending" src="https://img.shields.io/badge/arXiv-pending-inactive.svg">
</p>

<!-- PLACEHOLDER: when the preprint is posted, replace the "arXiv — pending"
     badge above with
     [![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
     and add the DOI/OpenReview link next to it. -->

---

**Fucina** (Italian for *forge*) turns a GGUF Mixture-of-Experts model into a
**two-plane ternary** model — `T ∈ {-1, 0, +1}`, 1.69–1.75 bits/weight — that
runs on llama.cpp with Vulkan.

> A **397B-parameter, 512-expert MoE** was quantized from a 740 GiB bf16
> checkpoint down to **84.00 GiB** (single plane) — **all expert weights
> ternary** — and served end to end on **one consumer desktop** (Ryzen AI
> Max+ 395, 96 GiB unified memory) at **~10 tok/s**. The forge itself took
> **7.6 hours** on that same desktop.
>
> The **published two-plane file** (ODINO-397B-v3.4a, **88.17 GiB**, 1.91
> bits/weight at the model level) resolves **78.3%** of a 143-trial paired
> coding-and-agentic suite against **55.2%** for the closest-size public
> IQ1_M quantization of the same checkpoint (85.09 GiB — the rival is 3.5%
> *smaller*, not size-matched; exact McNemar *p* = 5.42·10⁻⁷), and on ten
> public benchmark corpora run whole (cloud, 2026-09-19) it wins BoolQ by
> **+7.99** points, HellaSwag by **+2.33** and MMLU by **+2.13** with
> disjoint 95% intervals, loses none, and is **indistinguishable on
> perplexity** (7.5180 vs 7.5173). Full tables, conditions and commands:
> [paper/main.tex](paper/main.tex) §6 and [docs/RESULTS.md](docs/RESULTS.md).

Every technique here carries its **measured** effect on real weights — not the
paper's promise, our number — and credits the research it came from. The
project's most useful finding is a *negative* one, and it is
[described below](#the-result-we-did-not-go-looking-for).

---

## Key results

All figures below are reproduced with their conditions in
[docs/RESULTS.md](docs/RESULTS.md). Numbers are only comparable within a block.

### The models

| | 397B | 35B |
|---|---|---|
| architecture | 512 experts × 60 layers, hybrid GatedDeltaNet attention | 256 experts × 41 layers |
| source | bf16 checkpoint, 740 GiB | an existing high-quality GGUF |
| **output** | **84.00 GiB** (1.69 bpw expert weights, single plane); published two-plane file v3.4a **88.17 GiB** (second-plane tensors present on all 60 layers) | **9.19 GiB** |
| forge cost | 7.6 h, one consumer GPU, source streamed at ~50 MB/s | — |
| decode | ~10 tok/s, fully resident in 96 GiB UMA | — |

### Perplexity — wikitext-2, ctx 2048

| model | configuration | perplexity |
|---|---|---|
| 397B (30 chunks) | shipped build, one plane | **6.1845 ± 0.087** |
| 397B (30 chunks) | second plane on `up`+`gate`, every layer | 6.3509 — *worse* |
| 397B (30 chunks) | second plane on all three projections | 7.1871 — *worse* |
| 397B (12 chunks) | recipe in the file: tail band 44–59 only | **6.1291** |
| 35B (12 chunks) | second plane off | 8.7204 |
| 35B (12 chunks) | `up`+`gate` everywhere, `down` on 30–39, baked in | **8.2501 (−5.4%)** |

The 397B tail gain survives a paired logit comparison that the absolute error
bar cannot resolve: **mean Δp = +0.048% ± 0.014%, 3.4σ**.

### Task level, at closest public size

Same llama.cpp server build, same 16,384-token context budget, reasoning off
for both, two fixed seeds (7, 42), 72 fixtures → 144 scheduled pairs, 143
valid (frozen record of 2026-08-29). The suite is internal and not released;
aggregate results only, and no reproducibility claim attaches to it. The
ternary file was served with its second plane *disabled* (`ODINO_NO_P2=1` in
the launcher; see the paper, §6.2).

| | resolved | |
|---|---|---|
| **ODINO v3.4a, 88.17 GiB, 1.91 bpw (this work)** | **112/143 = 78.3%** | |
| Ornith-1.5-397B **IQ1_M**, 85.09 GiB (3.5% smaller) | 79/143 = 55.2% | +23.1 points |

Discordant pairs 39 vs 6; **exact McNemar p = 5.42·10⁻⁷**; paired bootstrap
(20k resamples) CI95 of the delta **[+14.7, +31.5] points**. The rival is not
size-matched: the ternary file is 3.6% larger, so the size handicap favours
the baseline. For context — not a controlled comparison — a same-family 35B
served at ~6.5 bits/weight scores 75% on an earlier version of that suite.
(An earlier version of this block reported 109/146 vs 81/146 for the 85.1 GiB
v3.3 file on the pre-freeze CSV; it is superseded by the frozen record above.)

### Public benchmarks, whole corpora (cloud, 2026-09-19)

ODINO v3.4a (88.17 GiB) against the closest-size public quantization of the
same checkpoint, `bartowski/Ornith-1.5-397B-GGUF` IQ1_M (85.09 GiB), scored
with llama.cpp's own `llama-perplexity` on NVIDIA RTX PRO 6000 (Vulkan), every
item of every corpus except PIQA (a declared 1,000-item subset of the 1,838
validation items). Wilson 95% intervals; the second plane of the ternary file
was applied on all 60 layers in these runs.

| benchmark | n | ODINO v3.4a | IQ1_M | Δ |
|---|---:|---|---|---:|
| HellaSwag | 10,042 | **76.40** [75.56, 77.22] | 74.07 [73.20, 74.92] | +2.33 |
| MMLU | 14,042 | **82.71** [82.07, 83.33] | 80.58 [79.92, 81.23] | +2.13 |
| BoolQ | 3,270 | **89.27** [88.16, 90.28] | 81.28 [79.91, 82.58] | +7.99 |
| ARC-Easy | 2,376 | 79.00 [77.31, 80.59] | 76.52 [74.77, 78.18] | +2.48 |
| ARC-Challenge | 1,172 | 57.85 [55.00, 60.65] | 55.63 [52.77, 58.45] | +2.22 |
| PIQA (1,000-item subset) | 1,000 | 80.90 [78.35, 83.22] | 80.90 [78.35, 83.22] | 0.00 |
| OpenBookQA | 500 | 44.40 [40.10, 48.78] | 43.60 [39.32, 47.98] | +0.80 |
| Winogrande | 1,267 | 66.22 [63.57, 68.77] | 61.64 [58.93, 64.28] | +4.58 |
| TruthfulQA MC1 | 817 | 38.43 [35.16, 41.82] | 36.72 [33.48, 40.08] | +1.71 |
| Perplexity, WikiText-2 (72 ch., ctx 4096) | 295k tok | 7.5180 ± 0.0519 | 7.5173 ± 0.0510 | +0.0007 |

Bold: disjoint 95% intervals. No corpus shows a negative difference beyond its
interval; perplexity cannot tell the two files apart. The 35B pair (a
single-plane 15.47 GiB forge against its 16.24 GiB public rival) ties on all
nine multiple-choice corpora and wins perplexity (7.6275 vs 8.1023). Exact
commands, prompt templates, per-item paired tests and the cross-check between
GPU vendors: [paper/main.tex](paper/main.tex) §6; per-piece logs in
[docs/RESULTS.md](docs/RESULTS.md#whole-corpus-public-benchmarks-cloud-2026-09-19).

⚠️ These are llama.cpp's per-token-normalised scores, zero-shot: comparable
between files run with the same protocol, **not** with published leaderboards
(lm-eval's `acc_norm` normalises by characters and its MMLU is 5-shot). The
earlier 300-task figures (74.67% HellaSwag, 70.33% Winogrande, on the 84.00 GiB
single-plane build) are superseded as measurements and kept in
[RESULTS](docs/RESULTS.md#standardised-benchmarks) only as "not broken"
evidence.

### Reconstruction fidelity, against the original checkpoint

397B, layer 59, hottest expert:

| projection | plane 1 | plane 1 + 2 |
|---|---|---|
| gate | 46.21% | **22.32%** |
| up | 46.20% | **22.31%** |
| down | 45.30% | **19.25%** |

### TCQ1_7 — the next quantizer, pre-tested on a real tensor

A fractional-rate trellis quantizer for the ternary class
([design](docs/TCQ1_7_DESIGN.md)), encoded with exact Viterbi over a
4096-state bitshift trellis on `blk.29.attn_gate.weight` of the 397B:

| quantizer | bpw | rel. error ‖e‖₂/‖w‖₂ | MSE vs RTN |
|---|---|---|---|
| RTN ternary, optimal per-256 scale | 1.6875 | 0.4420 | — |
| **TCQ1_7, Viterbi + fp16 refit** | **1.75** | **0.3473** | **−38.3%** |

The design's −42% gaussian projection lands at −38.3% on real weights; the
packed 56-byte payload round-trips bit-exact. This is a **single-tensor
pre-test**, not a model: at 26.5 ms/block the reference CPU encoder is ~460
process-days for the full expert complement, which is why the forge needs the
vectorized path before this becomes a build.

---

## The result we did not go looking for

The measurement that took longest to accept, and the one most likely to be
useful to someone else:

> **A correction that improves fidelity at every level we can measure can
> still make the model worse.**

A second ternary plane, correctly forged and correctly applied, halves the
weight error (46.2% → 22.3%), halves the single expert's output error under
real captured activations, and improves the weighted sum of the eight routed
experts — and degrades the model end to end (perplexity 6.1845 → 6.3509 on the
397B). Nine explanations were proposed; eight were refuted by measurement and
are recorded in [LESSONS](docs/LESSONS.md) with the numbers that killed them.

What decides the outcome is *where* the correction lands, not how faithful it
is. The identical perturbation, delivered at identical relative magnitude to
every layer's output, helps in the final third of the depth and is
catastrophic near the middle. **On both models the sign flips in the final
third of the depth — between 0.63 and 0.73 on the 35B and between 0.50 and
0.73 on the 397B, at the resolution of the band search — across a 10× size
difference, and the shipped bands begin at 73% of depth on both** (the
two-decimal agreement of the band edges is a property of the bands we chose
to ship, not a measured coincidence). The surviving
explanation is that a mixture-of-experts is governed by expert *selection*
more than by expert fidelity, so a correction that improves every expert can
still move the model by moving the routing; three independent 2025–26 papers
document that mechanism as a *cause* of quantization damage, and here it would
be the side effect of a *correction*.

The practical rule this repository now enforces: **decide on the model, never
on the reconstruction.** Every lever in the registry carries an end-to-end
number, or it carries a note saying it does not.

---

## The method

A weight matrix is approximated as the sum of two ternary planes:

```
W ≈ d₁·T₁ + d₂·T₂        T ∈ {-1, 0, +1},  256-weight blocks,  f16 scales
```

- **Hot experts get both planes; cold experts get one.** Routing frequency
  comes from the imatrix `counts`: on the 397B the hottest 3% of experts carry
  ~18% of the traffic, and frequency beats sensitivity as an allocation
  criterion (32.6% vs 38.2% output error, measured).
- **Cold experts get a *dedicated* single-plane GPTQ quantization.** The first
  plane of a joint two-plane optimization is co-adapted to its partner and
  loses ~10 points when used alone (37.95% → 28.13%, measured). This is the
  single most important trap in multi-plane ternary quantization.
- **GPTQ with true Hessians** — H = XXᵀ accumulated from the residual stream
  on the target model, 16,640 samples/layer — with full-matrix error
  propagation across blocks. Scales are computed from the *compensated* block.
- **Fragile tensors stay high-precision.** On hybrid-attention models
  (GatedDeltaNet), the linear-attention projections inherited at ~1.7 bit were
  47–51% wrong; promoting them to Q8_0 cut model-level output error from 25.4%
  to 17.1% for +3 GiB — the best byte-for-byte lever we measured
  (2.77 points/GiB). No sub-4-bit state-space scan has ever been demonstrated
  (cf. Quamba); do not try.
- **Experts are written hot-first at birth**, with router rows permuted to
  match: the inference engine assumes the second plane covers the expert
  prefix `0..K-1`. Assuming this without an assert cost us a full day — see
  [LESSONS](docs/LESSONS.md).
- **Per-layer boundary verification.** The hottest expert, reconstructed from
  the two written planes, must match the source within threshold — an assert
  inside the forge, not a hope.
- **The recipe goes into the file, not into a flag.** Second-plane tensors are
  optional per layer, so `strip_tensors.py` bakes a measured layer profile into
  the GGUF. A recipe that lives in the operator's memory does not survive being
  handed to someone else.

---

## Usage

Full walkthrough: [docs/USAGE.md](docs/USAGE.md).

### 1. Point the tools at your llama.cpp checkout

The scripts import `gguf-py` from a llama.cpp tree. Default is
`~/build-llamacpp`; override once:

```bash
export GGUF_PY="$HOME/llama.cpp/gguf-py"
export FP_CHECKPOINT_DIR="/path/to/original-bf16-checkpoint"   # forge-from-bf16 only
```

### 2. Forge from an existing GGUF

```bash
python3 forge_gguf.py \
    --source   model.gguf \
    --output   model-ternary.gguf \
    --hot      28 \
    --imatrix  imatrix.gguf \
    --hessians H_DIR/
```

| argument | meaning |
|---|---|
| `--source` | any GGUF MoE; weights are dequantized per expert |
| `--hot` | number of experts that receive the second plane |
| `--imatrix` | must contain per-expert routing **counts** |
| `--hessians` | optional; enables GPTQ on the `gate`/`up` projections |

Optional flags, each backed by a measurement (see `--help` and
[docs/USAGE.md](docs/USAGE.md)):

| flag | what it does |
|---|---|
| `--hot-select {freq,impact}` | pick hot experts by Hessian-weighted impact instead of routing frequency (requires `--hessians`) |
| `--plane2-layers 44-59` | restrict the second plane to the layer band where it actually helps — the final ~quarter of the depth on both models measured |
| `--foem-beta` | FOEM first-order correction (arXiv 2507.11017); default 0 = plain GPTQ |
| `--chunk` | rows per GPU chunk |

### 3. Forge from the original bf16 checkpoint

This is the path that produced the case study.

```bash
python3 forge_from_bf16.py        # configured through FORGE_* environment variables
```

### 4. Forge a dense (non-MoE) model

```bash
python3 forge_dense.py \
    --bf16     model-bf16.gguf \
    --donor    model-Q4_K_XL.gguf \
    --output   model-ternary.gguf \
    --hessians H_DIR/
```

`--bf16` is the FFN source; `--donor` supplies every non-FFN tensor verbatim.
Add `--ternary-parts gate,up` to keep `down_proj` at donor precision.

### 5. Bake a measured layer profile into the file

```bash
python3 strip_tensors.py in.gguf out.gguf --match ffn_down_exps2 --keep-layers 30-39
```

### Runtime requirements

A llama.cpp build with **TQ1_0 Vulkan kernels** *and* the two-plane extension
(optional `*_exps2` tensors summed inside `build_moe_ffn`; metadata key
`<arch>.expert_count2`). The single-plane TQ1_0 Vulkan kernels are in upstream
llama.cpp since [ggml-org/llama.cpp#27765](https://github.com/ggml-org/llama.cpp/pull/27765)
(merged 2026-09-07, commit `8fe90e1fbfc065f17a0b233c9df239423cd24a75`); the
two-plane extension is fork-only and a two-plane file does **not** load on an
unmodified upstream build (`wrong number of tensors; expected 1278, got 1098`).
The engine used for every paper number is released with the paper's
supplementary material as a diff against that upstream commit.

Kernel checklist before trusting any output: `test-backend-ops` must pass
`MUL_MAT` **and** `MUL_MAT_ID` for TQ1_0 with non-zero view offsets and n=1 —
the matrix and vector paths are *different shaders* with different offset
conventions (LESSONS, bug #3).

Sampling below 2 bits (measured; converges with Unsloth's advice):

```
temp 0.6 · min_p 0.03 · top_p 0.9 · presence_penalty 1.5
```

thinking enabled per request; never XTC. With `temp 0` and a short token budget
the same harness returns empty strings *from a healthy model* — below two bits
the sampling recipe is part of the measurement.

---

## What we learned

[**docs/LESSONS.md**](docs/LESSONS.md) is the part of this repository we would
read first if it were someone else's. It records, with the numbers:

- the **eight refuted explanations** for the correction paradox, including the
  two that were "confirmed" and then killed;
- the **joint-plane trap** (a plane co-adapted to its partner loses 10 points
  alone) — the bug that silently costs the most;
- three **engine bugs** that produced structurally perfect files which decoded
  to nonsense, and the assert that now catches each one;
- the day lost to assuming the **hot-first permutation** instead of checking it
  (misaligning router and experts moved perplexity 8.25 → 124.5);
- a **contaminated sweep** whose verdict reversed once in-place CPU mutation
  was removed — recorded because the first, wrong verdict had already been
  written down;
- why `--kl-divergence`, not accuracy, is the metric to move to.

[**docs/LEVERS.md**](docs/LEVERS.md) is the companion registry: 34 techniques,
each with its prior from the literature, the test protocol, the measured
outcome, the cost, and the citation. Its negative results are the point —
rotation into H's eigenvectors *hurt* (−14%), MagR *hurt* (39% → 46%),
sensitivity-based allocation *lost* to plain frequency. Those doors are closed
**for these models**; the forge reopens them for the next one, because every
lever is re-tested on a sample of the current model before it is applied.

---

## Repository layout

### The forge

| file | role |
|---|---|
| `forge.py` | orchestrator: applies each lever *only if it wins its own test on the current model* |
| `forge_gguf.py` | forge from an already-quantized GGUF → two-plane TQ1_0, resumable via journal |
| `forge_from_bf16.py` | forge from the original bf16 checkpoint — **this produced the case study** |
| `forge_dense.py` | forge for a *dense* (non-MoE) model: ternary FFN from bf16, every other tensor verbatim from a donor GGUF |
| `ternary_gpu.py` | GPU quantizer: joint two-plane optimization + full-propagation GPTQ |
| `two_planes.py` | Hessian preparation (damping, Cholesky) |
| `tq1_pack.py` | GPU TQ1_0 bit-packing + hot-first permutation, self-tested against the reference decoder |
| `build_hessians.py` | activation dumps → per-layer Hessians (reports anisotropy α) |
| `levers.py` | the lever registry: 34 techniques with prior, test protocol, cost, source |
| `qera_lora.py` | QERA closed-form low-rank correction of the quantization error, exported as a llama.cpp LoRA-GGUF |

### File surgery

| file | role |
|---|---|
| `promote_tensors.py` | selective precision promotion by file rewrite — **read its warning first** |
| `strip_tensors.py` | remove tensors a model no longer needs, and the metadata that declares them |
| `transplant_tensors.py` | replace selected tensors with a donor's copy: cross-file, size-changing, with `--drop` for stale residual planes |
| `selective_rollback.py` | tensor-level diff / guard / restore between two builds |
| `gguf_surgeon.py` | in-place tensor rewrite with automatic backups (no file rewrite) |

### Measurement and forensics

| file | role |
|---|---|
| `safe_repair.py` | the gate: a repair is kept only if its gain exceeds the benchmark's own noise |
| `paired_compare.py` | per-chunk paired sign test — resolves effects smaller than the absolute error bar |
| `generation_health.py` | free-running repetition probe — the number that goes *next to* every perplexity (teacher-forced windows never see the collapse) |
| `diagnose_assembly.py` | forensic check that a written file reconstructs its source weights |
| `permutation_check.py` | verifies the hot-first expert permutation against the imatrix |
| `routing_agreement.py`, `routing_concentration.py`, `routing_margin_distribution.py` | expert-selection instrumentation: route flips, concentration, and the top-k margin distribution |
| `selection_divergence.py`, `hot_coverage.py`, `impact_rank_odino.py` | hot-expert selection: frequency vs Hessian-weighted impact |
| `projected_error.py`, `correction_magnitude.py` | is the perturbation really the same size at every depth? (it is) |
| `layer_band_search.sh`, `layer_damage_profile.sh`, `damage_shape.sh`, `projection_decomposition.sh`, `tools_routing_flip.sh` | the sweeps that produced the depth rule |
| `no_miopen.py` | depthwise-conv fallback for ROCm GPUs without MIOpen kernels (gfx1151) |

### Documentation and research

| path | contents |
|---|---|
| [`docs/RESULTS.md`](docs/RESULTS.md) | every number the project rests on, with its conditions |
| [`docs/LESSONS.md`](docs/LESSONS.md) | the engineering rules we paid for |
| [`docs/LEVERS.md`](docs/LEVERS.md) | 34 techniques: prior, test, verdict, citation |
| [`docs/DESIGN.md`](docs/DESIGN.md) · [`docs/USAGE.md`](docs/USAGE.md) | architecture and end-to-end walkthrough |
| [`docs/TCQ1_7_DESIGN.md`](docs/TCQ1_7_DESIGN.md) · [`docs/TQ1_B160_DESIGN.md`](docs/TQ1_B160_DESIGN.md) | next-generation ternary formats |
| [`docs/QAT_PILOT_PLAN.md`](docs/QAT_PILOT_PLAN.md) | EfficientQAT ternary pilot (planned; no training executed) |
| [`docs/PAPER_DRAFT.md`](docs/PAPER_DRAFT.md) · [`paper/`](paper/) | the write-up and its LaTeX source ([current PDF](paper/main.pdf)) |
| [`experiments/`](experiments) | trellis encoder, Haar bands, and other single-question probes |

---

## Where this sits in the literature

At the time of writing we could find **no published result for a 400B-class
MoE at ~1.7 bits/weight**, and no published method that optimizes two ternary
planes *jointly* under a true calibration Hessian: PTQTP (arXiv 2509.16989)
does the joint part without the Hessian; QuantEase (arXiv 2309.01885) does the
Hessian part on one plane. The nearest published sub-2-bit MoE work is BitsMoE
(arXiv 2606.00079) on a 30B model, and the historical existence proof is QMoE
(arXiv 2310.16795), which compressed a 1.6T Switch Transformer below 1
bit/param — a non-reasoning model, in 2023.

For scale, published 2-bit post-training quantization of LLaMA3-8B and 70B
collapses to 50–64% HellaSwag with *every* method tried (arXiv 2404.14047),
and BitNet b1.58 2B4T — trained from scratch for ternary weights — reaches
68.4%. Our 74.67% at 1.69 bpw, from post-training quantization of weights
never intended for it, sits above both, and below what a well-executed 2.6-bpw
quant of a modern MoE achieves (80.3% with the same scorer). That is the
honest position.

What this repository documents is not a new theory: it is a complete,
measured, reproducible path from a 740 GiB bf16 MoE to a working 84 GiB model
on one desktop, and a rule about *where* corrections may be spent.

---

## Credits

This work stands on GPTQ (Frantar et al.), QuantEase, GPTAQ, QEP, the Babai
ordering result, SpinQuant/QuaRot, MagR, PT²-LLM, PTQTP, BOF4, DynaExq, MoPEQ,
MXMoE, EAC-MoE (QESC), Norm Tweaking, QZO, EoRA, ROMER, Super Weight, Quamba,
HOBBIT, QMoE, EvoPress, ScaleQ and EfficientQAT — and on the
[llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) project, whose TQ1_0
format (Francis Couture-Harpin) is the primitive everything here is built on.
Quantized source models from the community — Unsloth and bartowski in
particular — were the donors and the baselines throughout.

Full per-technique citations with arXiv identifiers:
[docs/LEVERS.md](docs/LEVERS.md); bibliography in
[`paper/references.bib`](paper/references.bib).

## Citing

A preprint is pending. Until it is posted, please cite this repository:

```bibtex
@software{fucina,
  title  = {Fucina: a general ternary forge for Mixture-of-Experts models},
  year   = {2026},
  url    = {https://github.com/Anjielon/fucina},
  note   = {Preprint pending}
}
```

## License

MIT — see [LICENSE](LICENSE). The source model of the case study
(Ornith-1.5) is MIT-licensed.
