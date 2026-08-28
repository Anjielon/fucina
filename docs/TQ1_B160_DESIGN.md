# TQ1_B160 — a ternary type for block-size-160 tensors

*Design note, 2026-08-28. Target: Qwen3.8-Flash-Next's 51.2B-parameter
per-layer n-gram embedding table (`per_layer_token_embd`, shape [160, 320M]).*

## Why a new type at all

No existing GGUF type below 4.5 bpw can encode this tensor: its rows are 160
elements, and every sub-4-bit type has an incompatible block size (K-quants,
IQ*, TQ1_0, TQ2_0: 256 · Q2_0: 64 · Q1_0: 128). unsloth's "4-bit minimum" on
this tensor coincides exactly with the format floor — their stated
sensitivity rationale ("will damage the model") has no published measurement
behind it, and Qwen's own tech report ablations (Tables 7-9) show the table's
contribution is small and saturating. The compression margin is plausibly
nearly free; nobody can currently reach it.

## The arithmetic gift

160 = 5 × 32, and TQ1_0's packing puts **5 trits per byte** (3^5 = 243 < 256).
So one 160-element block packs into exactly **32 bytes** of base-3 payload,
plus one f16 scale:

    block_tq1_b160 = { uint8_t qs[32]; ggml_half d; }   // 34 bytes / 160 el
    = 1.70 bpw — the same density as TQ1_0 itself, no padding, no waste.

A general block-32 ternary would cost 2.25 bpw (7 bytes payload + f16 per 32);
block-160 is strictly better *and* matches the target tensor's row length, so
per-block scale = per-row scale, which is the natural granularity for an
embedding row anyway (one lookup = one row = one scale).

The table drops from 26.8 GiB (IQ4_NL) to **~10.1 GiB**. On a 128 GiB
unified-memory machine serving the 125B Flash-Next, that is the difference
between the n-gram table crowding the transformer and fitting beside it —
and for SSD-streamed serving it halves the disk bandwidth per lookup.

## Scope of the engine work (small, by construction)

The tensor is touched by **one op only**: `ggml_get_rows` (verified: the PR
routes row indices host-side; the K/V n-gram projections are separate small
tensors). No matmul kernels, no mat-vec, no cm2. Required:

1. `GGML_TYPE_TQ1_B160` in ggml: enum, type traits (block 160, 34 bytes),
   CPU dequant row + get_rows case. (~100 lines)
2. Vulkan: `get_rows_tq1_b160.comp` — a trivial shader next to the existing
   get_rows family, reusing the shared `tq1_0_trit` helpers from types.glsl
   (the digit extraction is identical; only the block geometry differs).
3. The quantizer: encode is exact and closed-form once the per-row optimal
   scale is chosen — same two-level scale search as the forge, on rows of
   160. GPTQ does not apply (no accumulation to compensate; each row is
   looked up whole), which also removes the Hessian requirement entirely.

## Evaluation plan (measure, don't assert)

- Reference: the model with the IQ4_NL table. Candidate: TQ1_B160 table,
  everything else identical.
- Primary: paired Δp/KLD on the usual corpus. Secondary and **mandatory**:
  Chinese-heavy and knowledge-heavy tasks — the tech report says those are
  where the n-gram table actually earns its keep, so that is where damage
  would hide from wikitext perplexity.
- If per-row ternary is too coarse: fall back positions exist at 2.25 bpw
  (block-32, finer scales) and ~2.5 bpw (two-plane per row) before
  conceding back to 4 bits.

## Reference encoder

`fucina/tq1_b160.py` — pure-numpy encode/decode with the optimal per-row
scale, round-trip exact by construction. The engine work validates against
it byte-for-byte.
