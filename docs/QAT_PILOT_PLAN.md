# EfficientQAT ternary pilot — plan (no training executed)

> Written 2026-08-28. Code reading + local checks only (GGUF metadata, HF
> config via web, transformers modules installed in
> `/home/angelo/venv-catq`). No training, no GPU was used to produce this
> document.

## 0. Goal

PTQ (post-training, our `forge_dense.py`) breaks generation on the hybrid
dense 27B when ternarization is pushed beyond `ffn_{gate,up,down}`.
EfficientQAT (Block-AP + E2E-QP, ACL 2025) does block-wise QAT with MSE
reconstruction against the fp output — potentially recovering where pure PTQ
collapses. This document plans a **pilot** (2–3 blocks, not the whole model)
to decide whether the full port is worth the investment.

## 1. The model — what it really is (resolved)

Our `qwen38-27b-bf16.gguf` (54.66 GB, `/mnt/models/gguf/qwen38-tern/`) has
GGUF architecture `qwen35` with `qwen35.ssm.*` keys +
`qwen35.full_attention_interval` + `qwen35.nextn_predict_layers` → it is the
text backbone of **`Qwen/Qwen3.8-27B`**, officially released on HuggingFace
(mirrored as `unsloth/Qwen3.8-27B`, the origin of the UD-Q4_K_XL GGUFs
already in our fleet as `mogavis-qwen38-exec`).

**Actual config** (read from `huggingface.co/Qwen/Qwen3.8-27B/raw/main/config.json`):
```
architectures: ["Qwen3_5ForConditionalGeneration"]
model_type: "qwen3_5"
hidden_size: 5120
num_hidden_layers: 64
layer_types: hybrid, full_attention_interval=4 (16 full-attention layers, 48 GatedDeltaNet)
```
64 layers matches **exactly** the 64 Hessian files already on disk
(`/mnt/models/gguf/qwen38-tern/hessiane/H_00.npy`…`H_63.npy`, shape
`(5120, 5120) float32` — consistent with `hidden_size=5120`). The Hessians are
reusable for calibration/validation without recomputing them.

### ⚠️ Trap: the HF checkpoint is a VLM, not a plain CausalLM
`Qwen3_5ForConditionalGeneration` (the class declared in the config) is the
multimodal wrapper: `Qwen3_5Model` has `self.visual` (vision tower) +
`self.language_model` (the text backbone, a generic `Qwen3_5TextModel` via
`AutoModel.from_config`). There is **also** a text-only class,
`Qwen3_5ForCausalLM` (`transformers/models/qwen3_5/modeling_qwen3_5.py:1613`)
with `self.model = Qwen3_5TextModel(config)` — the one with `embed_tokens`,
`layers`, `norm`, `rotary_emb` in the Llama style that EfficientQAT expects.

**Practical implication**: downloading the `Qwen/Qwen3.8-27B` repo also pulls
the vision tower (extra weight, useless to us — our GGUF does not contain it
at all: llama.cpp keeps mmproj separate). To use EfficientQAT one must:
1. download the full checkpoint (bf16, ~54–58 GB including the vision tower);
2. extract ONLY the `model.language_model.*` weights from the state dict,
   rename them `model.*`, and load them into `Qwen3_5ForCausalLM` with
   `config.text_config` (a `Qwen3_5TextConfig`) — EfficientQAT **does not do
   this automatically**: `main_block_ap.py:128` calls
   `AutoModelForCausalLM.from_pretrained(args.model)` with the repo's raw
   config (which declares `ForConditionalGeneration`) → it fails or loads the
   wrong class. A state-dict extraction script is needed before invoking
   `main_block_ap.py` (half a day of work, the same pattern
   `gguf_surgeon.py`/`forge_dense.py` already implement on the GGUF side, but
   on the HF safetensors side).

There is no official gguf→HF converter for this hybrid architecture in
llama.cpp: **the only clean path is re-downloading the HF original**, not
converting our GGUF backwards.

## 2. Quantizer — where the fake-quant lives, exact file:line

Call chain (from the point where Block-AP replaces the Linears):
```
quantize/block_ap.py:166-169   → for each nn.Linear in the block:
                                  quantlinear = int_linear_fake.QuantLinear(module, args.wbits, args.group_size)
quantize/int_linear_fake.py:34 → self.weight_quantizer = UniformAffineQuantizer(wbits, group_size, weight=org_module.weight)
quantize/int_linear_fake.py:41 → weight = self.weight_quantizer(self.weight)   # in the forward
quantize/quantizer.py:23-86    → class UniformAffineQuantizer  ← THE ACTUAL FAKE-QUANT
  quantize/quantizer.py:40-50  →   per-group scale/zero_point init from weight min-max (asymmetric)
  quantize/quantizer.py:58-74  →   fake_quant(): affine INT round-clamp-dequant with zero_point
  quantize/quantizer.py:35-36  →   group_size (assert weight.shape[-1] % group_size == 0)
```

**Replacement for our TQ1_0 semantics** (ternary {-1,0,+1}, per-block-256
scale, **symmetric, no zero_point**):
- Rewrite `UniformAffineQuantizer.__init__` (lines 24-50): no `qmin/qmax`
  derived from bit-width, no `zero_point` (the parameter must be removed, not
  merely zeroed — a trainable `zero_point` would break the ternary symmetry);
  `group_size` fixed at 256 (our BLOCK, consistent with `forge_dense.py:47`
  `BLOCK = 256`); initial scale = `absmax(group)` (the same heuristic as our
  `ternary_gpu.py`, not asymmetric min-max).
- Rewrite `fake_quant()` (lines 58-74): `x_int = round_ste(x / scale).clamp(-1, 1)`
  (the STE is already in `quantizer.py:10-14`, reusable) instead of
  `round-add(zero_point)-clamp(qmin,qmax)-sub(zero_point)`; dequant =
  `x_int * scale` (no offset).
- `int_linear_fake.py` needs no structural change: the forward is already
  agnostic to the quantizer type (`weight = self.weight_quantizer(self.weight)`);
  the new class just has to expose the same interface (`forward`, `.scale` as
  a trainable `nn.Parameter` for E2E-QP).
- `quant_parameters`/`weight_parameters` in `quantize/utils.py` iterate by
  parameter name (`scale`, `zero_point` explicitly filtered somewhere?) — to
  be verified line by line at implementation time: if they filter on the name
  `"zero_point"` the reference must go, otherwise the optimizer will try to
  train a parameter that no longer exists.

### ⚠️ Second problem, bigger than the quantizer: WHICH Linears to touch
`block_ap.py:166-167` does `for name, module in qlayer.named_modules(): if
isinstance(module, torch.nn.Linear)` — **generic, it touches EVERY Linear in
the block**, including the internal projections of the GatedDeltaNet
(`in_proj_qkvz`, `in_proj_ba`, `out_proj` — names verified in
`transformers/models/qwen3_5/modeling_qwen3_5.py`, class
`Qwen3_5GatedDeltaNet`). The truly recurrent part (state-space scan, `A_log`,
`dt_bias`, `conv1d`) is NOT `nn.Linear` → it stays untouched by default, which
is correct. But the **GDN linear projections are**, and our
`docs/LEVERS.md:92` already measures that ternarized GatedDeltaNet projections
are **47–51% wrong**, while at Q8 the model-level error drops from 25.4% to
17.1%. Block-AP's out-of-the-box behaviour would ternarize them along with
`mlp.{gate,up,down}_proj`, **contradicting our most expensive finding**.
**A per-module-name allowlist is needed** in `block_ap.py:166-169` (e.g. regex
`r"\.mlp\.(gate|up|down)_proj$"`, or, to replicate `forge_dense.py` exactly,
also the `q/k/v/o_proj` of the 16 full-attention layers) before the pilot is
even launched — otherwise QAT gets measured on the wrong tensor combination
and the signal is thrown away.

## 3. Model-support verdict

**It does not run out of the box.** Two concrete incompatibilities found by
reading the *installed* transformers 5.15.1 code against EfficientQAT's
assumptions (written against transformers==4.40.1, pre-RoPE-refactor):

1. **Decoder-layer signature** — `Qwen3_5DecoderLayer.forward()`
   (`modeling_qwen3_5.py:756+`, same pattern as `Qwen3NextDecoderLayer`,
   verified at `modeling_qwen3_next.py:840-848`) requires
   `position_embeddings: tuple[Tensor, Tensor]` as a positional argument
   **without a default**, computed once upstream by
   `Qwen3_5TextModel.forward()` (`modeling_qwen3_5.py:1198`,
   `self.rotary_emb(hidden_states, position_ids)`) and propagated to every
   layer. `quantize/block_ap.py` calls the layers with `layer(inps,
   attention_mask=attention_mask, position_ids=position_ids)` in TWO places
   (`update_dataset()` lines 22-30 and the `Catcher.forward` lines 79-95,
   also used to capture the inputs) — **never** `position_embeddings` → an
   immediate `TypeError` on the first block forward. Fix: compute
   `position_embeddings = model.model.rotary_emb(hidden_states, position_ids)`
   once and thread it into the layer call at both sites (a small but
   non-optional patch, specific to post-refactor "RoPE-once" architectures:
   Llama-2 does not suffer from it, which is why the repo never needed it).
2. **VLM nesting** — see §1: `model.model.layers` (used by `block_ap.py:49`)
   only exists when loading `Qwen3_5ForCausalLM`, not
   `Qwen3_5ForConditionalGeneration` (what the repo's config declares by
   default). The state-dict extraction described above is required.

Neither is an impassable wall — together roughly one day of patching — but
**"out of the box" is false**: running `main_block_ap.py --model
Qwen/Qwen3.8-27B` without these two fixes gets a crash at the first block, not
a slow or degraded training run.

The rest of the iteration is genuinely generic: `layers = model.model.layers`
(no per-architecture dispatch), `qlayer.named_modules()` to collect the
Linears (no hardcoded Llama name list), `update_dataset()` calls the layers as
black boxes — with the two fixes above and the §2 allowlist, the rest of the
Block-AP pipeline (input capture, per-block MSE, separate weight/quant-param
optimizers, periodic saving) needs no further structural change.

## 4. Dependency verdict

| Item | requirements.txt | venv-catq | Verdict |
|---|---|---|---|
| torch | `2.2.2` (CUDA) | `2.11.0+rocm7.13.0a20260426` | No `torch==2.2.2` build supports ROCm gfx1151 on this stack: we use **venv-catq**, not the pinned requirements. |
| transformers | `4.40.1` | `5.15.1` | **Structural incompatibility, not just a version gap**: `qwen3_5`/`qwen3_next` do not exist at all in transformers 4.40.1 (Qwen3.8/Qwen3-Next are far more recent). No transformers version satisfies both "requirements.txt as published" and "Qwen3.8 support" — installing the original requirements is not an option; we use **venv-catq as is**, which is the direct cause of the §3.1 blocker (the repo was never updated to the new decoder-layer contract). |
| triton | `2.2.0` | `3.6.0+rocm7.13.0a20260426` (+ `pytorch-triton-rocm 3.5.1`) | Used ONLY in `quantize/int_linear_real.py` (real INT packing post-training, 2/3/4/8 bit) — **not needed for the pilot** (Block-AP fake-quant does not import it; the final TQ1_0 export is done with our own GGUF toolchain in `fucina/`, not with `model_transfer/`). Present and working in venv-catq anyway. |
| flash-attn | not required | not installed | **Zero hits** for `flash_attn` in the whole repo (grep over `*.py`, `quantize/*.py`, `model_transfer/*.py`) — no ROCm problem; the model uses sdpa/eager by default. |
| accelerate | `0.28.0` | `1.14.0` | APIs used (`infer_auto_device_map`, `dispatch_model`, `init_empty_weights`, `load_checkpoint_in_model`) are stable across the two versions — low risk. |
| bitsandbytes | `0.41.0` | not verified in venv-catq | Not referenced in the core Block-AP/E2E-QP path (`main_block_ap.py`, `quantize/block_ap.py`) — check only if the accessory scripts get used. |

**Summary**: we work inside **venv-catq**, ignoring the repo's
`requirements.txt`. That is also the only way to have Qwen3.8 supported — but
it is exactly what breaks the decoder-layer signature (§3.1), because
EfficientQAT has not been touched since transformers introduced
`position_embeddings`.

## 5. Pilot design (2–3 blocks, NOT the whole model)

Purpose: verify whether Block-AP recovers where PTQ collapses, BEFORE
investing in the full port (§3 patches + §2 allowlist + a real training loop).

1. **Minimal patches** (§3.1 position_embeddings, §3.2 text-only extraction,
   §2 Linear allowlist) — needed even to run a single block.
2. **Block selection**: 2–3 full-attention layers (simpler, no GDN
   projections to exclude) or 2–3 GatedDeltaNet layers with the allowlist
   active — better to start from full-attention layers, to separate the
   quantizer problem from the GDN problem.
3. **Calibration**: reuse the existing Hessians
   (`/mnt/models/gguf/qwen38-tern/hessiane/H_XX.npy`, 64 files, one per layer,
   `(5120,5120) float32`) as a control/validation term for the error
   direction, PLUS a small wikitext-2 set (natively supported by
   `datautils_block.py::get_loaders`) for the real block inputs during
   Block-AP (the Hessians alone are not enough for Block-AP, which needs
   batches of input/output hidden states, not just the covariance matrix).
4. **Quantizer**: ternary replacement per §2, `group_size=256`.
5. **Training**: low `--epochs` (2–4, not the hundreds used for Llama-2 in
   the paper — this is a pilot, not the final recipe), only on the 2–3
   chosen blocks, rest of the model untouched (fp16/bf16, or from the
   existing Q4_K_XL donor for the remainder).
6. **Patching the blocks into the real model**: after Block-AP on the 2–3
   blocks, substitute ONLY those layers in the loaded model (full fp16, or
   dequantized from the GGUF) and run free-running generation (not
   teacher-forced).

### GO/NO-GO criteria (from our earlier research)
| Criterion | Threshold | Source |
|---|---|---|
| Block reconstruction convergence | Block-AP MSE decreases monotonically and stabilises within the planned epochs (no NaN, no oscillation) | standard EfficientQAT practice |
| Free-running generation with ONLY those blocks patched | does not collapse (no degenerate repetition, no garbage) on at least 3 long prompts (>200 tokens) | explicit task requirement; our `generation_health.py` already measures that "teacher-forced perplexity never sees the error that free-running accumulates" — use it as a guard BUT not as the only judge |
| GDN state norm stable | no growth/explosion of the recurrent-state norm along free-running generation (compare against the fp16 checkpoint on the same prompt) | consistent with `docs/RESULTS.md:1100` ("GatedDeltaNet accumulates FFN error in the state during generation") — if the state diverges, the problem is in the recurrence, not the quantizer |
| Comparison against pure PTQ on the same blocks | Block-AP must beat direct ternary PTQ (same layer, same scale) both in reconstruction MSE and in generation health, otherwise the QAT cost is not justified | the pilot's business criterion |

If even just the "free-running generation does not collapse" criterion fails
on the 2–3 pilot blocks, **NO-GO**: the problem is not the absence of QAT but
something more structural about ternary applied to the GDN (consistent with
`docs/LEVERS.md:92`), and the full port (patches + allowlist + E2E-QP loop)
would not be worth the investment.

## 6. Wall-clock estimate (pilot, 2–3 blocks, not the whole model)

Estimates, NOT measurements (no training executed):
- §3.1 + §3.2 patches + §2 allowlist: **0.5–1 day** of work (code, not
  compute).
- Download of `Qwen/Qwen3.8-27B` bf16+vision (~54–58 GB estimated, consistent
  with the 54.66 GB of our text-only GGUF + vision tower) over the home
  network: depends on available bandwidth, on the order of **1–3 hours** at
  typical HF hub speeds.
- Text-only state-dict extraction: minutes (local I/O, no GPU).
- Block-AP on 2–3 blocks, few epochs, small batch: on our hardware (Ryzen AI
  Max+ 395, 96 GiB Vulkan/ROCm VRAM) the per-block order of magnitude is
  **tens of minutes** for a few epochs on a dense model (no MoE/router
  overhead), so **1–2 hours total** for 2–3 blocks — an optimistic figure, to
  be recalibrated at the first real run (Block-AP does a full
  forward+backward per block, not forward-only like PTQ).
- Validation (free-running generation + state norm + PTQ comparison):
  **30–60 minutes**.
- **Estimated pilot total: half a day of compute + 1 day of code porting**,
  excluding the network-dependent download time.

## 7. The biggest blocker

**It is not the quantizer** (a localised replacement, ~50 lines in
`quantize/quantizer.py`, behind an already-abstract interface). **It is that
the official HF checkpoint is a VLM (`Qwen3_5ForConditionalGeneration`) while
EfficientQAT assumes a flat Llama-style CausalLM** — a text-only state-dict
extraction (no existing tool in the repo does it) *and* a decoder-layer
signature patch for the post-refactor `position_embeddings` contract of
transformers are both required BEFORE a single block can even be loaded.
Neither patch exists upstream, and without the second one (the §2 allowlist on
the GatedDeltaNet Linears) the pilot would measure the wrong thing anyway,
repeating the mistake already paid for in `docs/LEVERS.md`.
