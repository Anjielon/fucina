# FUCINA — A General Ternary Forge for Mixture-of-Experts Models

**Fucina** (Italian for *forge*) turns any GGUF Mixture-of-Experts model into a
**two-plane ternary** model (TQ1_0, 1.69–3.38 bits/weight) that runs on
llama.cpp with Vulkan — built and battle-tested by fitting **ODINO**
(Ornith-1.5-397B, all 512 experts, no pruning) into the 96 GiB unified memory
of a single AMD Ryzen AI Max+ 395 desktop, at ~10 tok/s.

Every technique in this repository carries its **measured** effect on real
weights — not the paper's promise, our number — and every technique credits
the research it came from. See [docs/LEVERS.md](docs/LEVERS.md) for the full
catalogue (34 levers, each with prior, cost, verdict and citation) and
[docs/LESSONS.md](docs/LESSONS.md) for the engineering rules we paid for.
Every number the project rests on, with the conditions it was taken under, is
collected in [docs/RESULTS.md](docs/RESULTS.md).

## The method

A weight matrix is approximated as the sum of two ternary planes:

```
W ≈ d₁·T₁ + d₂·T₂        T ∈ {-1, 0, +1},  256-weight blocks,  f16 scales
```

- **Hot experts get both planes; cold experts get one.** Routing frequency is
  taken from the imatrix `counts`: on Ornith-397B the hottest 3% of experts
  capture ~18% of the traffic, and frequency beats sensitivity as an
  allocation criterion (32.6% vs 38.2% output error, measured).
- **Cold experts get a *dedicated* single-plane GPTQ quantization.** The first
  plane of a joint two-plane optimization is co-adapted to its partner and
  loses ~10 points when used alone (37.95% → 28.13% measured). This is the
  single most important trap in multi-plane ternary quantization.
- **GPTQ with true Hessians** (H = XXᵀ accumulated from the residual stream on
  the target model, 16,640 samples/layer), full-matrix error propagation
  across blocks. Scales are computed from the *compensated* block.
- **Fragile tensors are kept high-precision.** On hybrid-attention models
  (GatedDeltaNet), the linear-attention projections inherited at ~1.7 bit were
  47–51% wrong; promoting them to Q8_0 cut model-level output error from
  25.4% to 17.1% for +3 GiB — the best byte-for-byte lever we measured
  (2.77 points/GiB). No sub-4-bit state-space scan has ever been demonstrated
  (cf. Quamba); do not try.
- **Experts are written hot-first at birth** (router rows permuted to match):
  the inference engine assumes the second plane covers the expert prefix
  `0..K-1`. See LESSONS — assuming this without an assert cost us a full day.
- **Per-layer boundary verification**: the hottest expert, reconstructed from
  the two written planes, must match the source within threshold — an assert
  inside the forge, not a hope.

## What's in the box

| file | role |
|---|---|
| `forge.py` | orchestrator: applies each lever *only if it wins its own test on the current model* |
| `forge_gguf.py` | forge from an already-quantized GGUF → two-plane TQ1_0, resumable via journal |
| `forge_from_bf16.py` | forge from the original bf16 checkpoint — **this is what produced the case study** |
| `ternary_gpu.py` | GPU quantizer: joint two-plane optimization + full-propagation GPTQ |
| `two_planes.py` | Hessian preparation (damping, Cholesky) |
| `tq1_pack.py` | GPU TQ1_0 bit-packing + hot-first permutation (self-tested against the reference decoder) |
| `build_hessians.py` | activation dumps → per-layer Hessians (reports anisotropy α) |
| `levers.py` | the lever registry: 34 techniques with prior, test protocol, cost, source |
| `safe_repair.py` | the gate: a repair is kept only if its gain exceeds the benchmark's own noise |
| `paired_compare.py` | per-chunk paired sign test — resolves effects smaller than the absolute error bar |
| `promote_tensors.py` | selective precision promotion by file rewrite — **read its warning first** |
| `strip_tensors.py` | remove tensors a model no longer needs, and the metadata that declares them |
| `selective_rollback.py` | tensor-level diff / guard / restore between two builds |
| `gguf_surgeon.py` | in-place tensor rewrite with automatic backups (no file rewrite) |
| `diagnose_assembly.py` | forensic check that a written file reconstructs its source weights |
| `no_miopen.py` | depthwise-conv fallback for ROCm GPUs without MIOpen kernels (gfx1151) |

## Usage

```bash
python3 forge_gguf.py \
    --source    model.gguf          # any GGUF MoE (weights are dequantized per expert)
    --output    model-ternary.gguf
    --hot       28                  # experts that receive the second plane
    --imatrix   imatrix.gguf        # must contain per-expert routing counts
    --hessians  H_DIR/              # optional: enables GPTQ on gate/up projections
```

Runtime requirements: a llama.cpp build with TQ1_0 Vulkan kernels **and** the
two-plane extension (optional `*_exps2` tensors summed inside `build_moe_ffn`,
metadata key `<arch>.expert_count2`). Kernel checklist before trusting any
output: `test-backend-ops` must pass MUL_MAT **and** MUL_MAT_ID for TQ1_0
with non-zero view offsets and n=1 — matrix and vector paths are *different
shaders* with different offset conventions (see LESSONS, bug #3).

Recommended sampling below 2 bits (measured, converges with Unsloth's advice):
`temp 0.6 · min_p 0.03 · top_p 0.9 · presence_penalty 1.5`, thinking enabled
per request; never XTC.

## Case study: ODINO

| | |
|---|---|
| source | Ornith-1.5-397B-A17B (MIT), 512 experts × 60 layers, hybrid GatedDeltaNet attention |
| output | 88.17 GiB (vs ~740 GiB bf16). Experts: 484 cold @ one dedicated plane, 28 hot @ two joint planes — 75.9 GiB for the first plane, 4.1 GiB for the second. Everything else inherited from the source and left alone: the state-space projections at Q8_0 (3.0 GiB, the lever that paid best per byte), standard attention at Q4_K/Q6_K, embeddings Q8_0, output head Q5_K, routers F32 |
| forge time | 7.6 h end-to-end on one consumer GPU (Radeon 8060S, Vulkan), NAS-fed at ~50 MB/s |
| quality | multi-step arithmetic and logic traps solved with reasoning enabled; per-expert weight error 21.5% (2-plane) / 28.1% (1-plane dedicated) |
| ⚠️ caveat | the published 88 GiB build predates the joint-pair fix, so its second plane is orphaned and must stay disabled — it is a single-plane model, and the `*_exps2` tensors are 4.1 GiB of dead weight. The two-plane numbers above are the quantizer's, measured on weights; a file that realises them requires a forge run with the current code |
| speed | ~10 tok/s decode, fully resident in 96 GiB UMA |

## The result we did not go looking for

The measurement that took the longest to accept, and the one most likely to be
useful to someone else:

> **A correction that improves fidelity at every level we can measure can still
> make the model worse.**

A second ternary plane, correctly forged and correctly applied, halves the
weight error (46.2% → 22.3%), halves the single expert's output error under
real captured activations, and improves the weighted sum of the eight routed
experts — and degrades the model end to end (perplexity 6.1845 → 6.3509 on the
397B). Nine explanations were proposed; eight were refuted by measurement and
are recorded in [LESSONS](docs/LESSONS.md) with the numbers that killed them.

The surviving candidate is that a mixture-of-experts is governed by expert
*selection* more than by expert fidelity, so a correction that improves every
expert can still move the model by moving the routing. Three independent
2025-26 papers document that mechanism as a *cause* of quantization damage
(arXiv 2606.05688, 2506.13329, 2605.23078); here it would be the side effect of
a *correction*. Another project reports the same dissociation from the other
direction: Tied Trit-Planes (arXiv 2608.08910) finds its higher-reconstruction-
error variant scoring 86 against 84 on MMLU.

The practical rule this repository now enforces: **decide on the model, never
on the reconstruction.** Every lever in the registry carries an end-to-end
number, or it carries a note saying it does not.

## Where this sits in the literature

At the time of writing we could find **no published result for a 400B-class MoE
at ~1.7 bits/weight**, and no published method that optimizes two ternary planes
*jointly* under a true calibration Hessian — PTQTP (arXiv 2509.16989) does the
joint part without the Hessian, QuantEase (arXiv 2309.01885) does the Hessian
part on one plane. The nearest published sub-2-bit MoE work is BitsMoE
(arXiv 2606.00079) on a 30B model, and the historical existence proof is QMoE
(arXiv 2310.16795), which compressed a 1.6T Switch Transformer below 1
bit/param — a non-reasoning model, in 2023. The only independent data point at
our exact scale is an industry report that a 397B-class MoE at 2 bit "corrupts
structured output", which matches the failure mode we measured before fixing
our engine.

That is the gap this repository documents: not a new theory, but a complete,
measured, reproducible path from a 740 GiB bf16 MoE to a working 88 GiB model
on one desktop.

## Credits

This work stands on: GPTQ (Frantar et al.), QuantEase, GPTAQ, QEP, the Babai
ordering result, SpinQuant/QuaRot, MagR, PT²-LLM, PTQTP, BOF4, DynaExq, MoPEQ,
MXMoE, EAC-MoE (QESC), Norm Tweaking, QZO, EoRA, ROMER, Super Weight, Quamba,
HOBBIT, and the llama.cpp/ggml project (TQ1_0 format by Francis Couture-Harpin).
Full per-technique citations with arXiv identifiers: [docs/LEVERS.md](docs/LEVERS.md).

## License

MIT. The source model of the case study (Ornith-1.5) is MIT-licensed.
