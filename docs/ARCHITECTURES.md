# Architectures seen by `fucina analyze` (22 Oct 2026)

Every GGUF on the forge machine (144 files, 10 architectures) was read with `fucina analyze`, with no model-specific
configuration. After adding five generic pieces (`attn_mla`, `sparse_indexer`, `hyper_conn`, `draft_fc`, `norm`) every
tensor family of every architecture is classified.

| architecture (GGUF) | example | type | layers | attention / linear | experts | notes |
|---|---|---|---|---|---|---|
| qwen35moe | Thor-T (Ornith-1.5 35B-A3B), 118 files | hybrid MoE | 41 (40 + MTP) | 10 / 31 GatedDeltaNet | 256 (8 active) + shared | reference model of the forge |
| qwen35 | Ternary-Bonsai-27B | hybrid dense | 64 | 16 / 48 | — | same mixer as qwen35moe, dense FFN |
| qwen3 | Ternary-Bonsai-8B | dense | 36 | 36 / 0 | — | |
| qwen3vl | Qwen3-VL-8B (text encoder) | dense | 36 | 36 / 0 | — | vision tower in a separate mmproj |
| qwen4exp | Qwen3.8-Flash-Next, MTP head only (main GGUF deleted) | hybrid MoE | 1 block (#48) | — | 512 | hyper-connections `hc_*`, sparse-attention indexer `indexer.*` |
| muse-glimmer | Muse Glimmer-30B | dense | 52 | 52 / 0 | — | extra `post_ffw_norm` |
| dflash | DFlash draft for Glimmer | dense | 5 | 5 / 0 | — | `fc` fusion of target hidden states |
| deepseek2-ocr | Unlimited-OCR | MoE | 12 | 12 / 0 | 64 + shared | DeepSeek-V2 family |
| llama | Llama-3.1-8B | dense | 32 | 32 / 0 | — | `rope_freqs` |
| spark2_5 | Spark x2.5 4B | dense | 36 | 36 / 0 | — | |

## What is general and what is not

* **Structure: general.** Layer mixer (attention vs linear attention), FFN kind (dense / MoE / shared expert), router,
  MTP, MLA, sparse indexer, hyper-connections and norms are recognised from tensor names and metadata alone.
* **Levers: not yet general.** Every registered technique declares `architectures = ("qwen35moe",)` because its wrapped
  scripts were validated only there; on the other nine architectures `fucina analyze` therefore lists **zero
  applicable levers**. Most of the mathematics is architecture-independent (GPTQ/JARQ/OBQ on any dense linear, CARDAN on
  any expert bank, folding on any ternary tensor): what is missing is, per lever, a validation on a second architecture
  (small model, minutes: Bonsai-8B `qwen3` for dense, DeepSeek-OCR `deepseek2` for MoE) and then
  `architectures = ("*",)` with a per-piece guard.
* **New mixers to study before forging them:** `qwen4exp` (hyper-connections + sparse indexer + 512 experts) and
  DeepSeek-V3 MLA (`attn_kv_a_mqa`/`attn_kv_b`): the low-rank latent projections and the indexer are dense linears
  (levers apply), the hyper-connection matrices are tiny and are kept in full precision.
