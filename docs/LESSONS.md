# Engineering Lessons — paid for in full on 2026-08-26

A structurally perfect quantized file and a working model are **two different
claims**. These rules exist because we shipped the first without the second.

## The seven rules

1. **Every convention shared by two components becomes an assert at the
   boundary.** Our forge and our engine "agreed" that hot experts come first —
   in a comment. The forge didn't do it; the engine assumed it; the model
   deliriated. A one-line assert would have caught it at write time.

2. **The smoke test is part of the forge, not a step after it.** Four
   questions with verifiable answers (a discount trap, a character reversal, a
   riddle, one open sentence) run immediately after the file is written.
   Structural validity says nothing about semantics.

3. **Never patch a file in place without saving the bytes you touch.** We
   zeroed 163 MB of second-plane scales without a backup; recomputing them
   cost a 49-minute GPU pass that a 163 MB copy would have avoided.

4. **Test with the deployment sampling recipe.** Below 2 bits, `temp 0.2`
   degenerates into token loops that `temp 0.6 + presence 1.5` never shows.
   Our first test harness *failed a healthy model* because it sampled
   greedily.

5. **Judge reasoning mode separately.** Sub-2-bit models can have a broken
   thinking stream and sound direct answers, or vice versa. One combined
   verdict hides which half is damaged (ours: thinking looped `</think>`
   forever while no-think answered "Paris" in 3 s — then after the kernel
   fix, thinking solved equations the no-think path could not).

6. **A new quantization type enters `test-backend-ops` the same day**, in
   `base_types`, with cases at the *target model's shapes*: real expert
   counts, real k, and **n=1** (single-token decode). Our 872 green tests
   covered everything except the code path the model actually generates with.

7. **Matrix and vector are different shaders.** mul_mm, mul_mat_vec and
   mul_mat_vec_id have separate implementations and separate offset
   conventions (block units vs element units). Validating one says nothing
   about the others. Test all three, with non-zero view offsets.

## The forensic method that found bug #3

When the model still failed after the file was proven correct, the search
narrowed by *exclusion with measurements*, one hour end-to-end:

1. File exonerated: hot expert at position 0, reconstructed from both written
   planes, matched the bf16 source at 26.9% (expected).
2. Engine graph read line-by-line: structurally correct.
3. Key deduction: with plane-2 scales at zero, the same kernel path ran and
   produced *zeros* — so a **numerical** kernel bug would be invisible until
   the scales became real. That inverted the suspicion from "our new code"
   to "the kernel path we never tested".
4. TQ1_0 enabled in `test-backend-ops`: instant garbage (err ≈ 1.9) on
   MUL_MAT_ID at n=1 — and on MUL_MAT only with non-zero view offsets.
5. Diff against the working TQ2_0 shader: `a_offset / QUANT_K` on an offset
   already expressed in blocks. One line, two files, everything green.

Silence in a passing test suite is not evidence — coverage is.

## Auditing your own rejections

Seven techniques had been rejected on measurements. A critical re-reading of
the *experiment code* — not the results — invalidated three of them.

| rejected technique | what the code actually did |
|---|---|
| GPTAQ (asymmetric calibration) | never built the required cross-Hessian; the "asymmetric" flag disabled GPTQ's intra-block propagation. The measured 32.6% sits exactly between no-GPTQ (38.9) and GPTQ (28.1) — a mutilated GPTQ was rejected, not GPTAQ |
| Haar transform | applied to the wrong axis (the residual dimension instead of the expert's private one) and without the band grouping that is half the method. With 256-wide blocks the first block mixes seven bands: 0.1% was the guaranteed outcome |
| MagR | the proximal step was an iterated clip at (1−α)·rowmax — geometric shrinkage of the maximum, not the ℓ∞ proximal operator; no null-space projection and no GPTQ afterwards. The conclusion may still hold, the evidence does not |

A fourth finding was procedural: four of the seven experiments used the same
`gate_up` tensor as a proxy for the model, while the techniques under test are
aimed at `down` — the one matrix that receives no error compensation.

Two rules follow:

1. **A rejection is a claim, and claims need the same scrutiny as successes.**
   Re-read the experiment code before trusting a negative result, especially
   when it contradicts a published one.
2. **Test a technique on the tensor it was designed for.** A convenient proxy
   makes the experiment cheap and the answer meaningless.

There is a third, uncomfortable one: our own rejection of router recalibration
had a margin of 0.028 against a stated uncertainty of ±0.087. By the rule in
this repository, that number does not establish the rejection either. Use a
paired per-chunk comparison when the effect is small in either direction.

## Know what your measurement can see

Before attempting a repair, compare its expected gain against the resolution
of your own benchmark. Ours reports perplexity as `6.2091 ± 0.0874` over 30
chunks — roughly 1.4% relative. A technique whose published benefit is 0.1–0.3
perplexity is therefore invisible at 30 chunks: it sits inside one and a half
standard deviations.

Since σ falls as 1/√n, resolving a 0.1 gain at three sigma needs about 210
chunks — seven times the measurement cost. That is a legitimate choice, but it
must be made deliberately, before spending hours on a repair whose outcome the
benchmark cannot report.

The alternative is a paired comparison: evaluate both models on the same
chunks and test the per-chunk differences, which cancels the corpus variance
that dominates the absolute figure. Use it whenever the expected effect is
small in either direction — including when you are trying to *reject* a
technique.

## The standard of proof for a rejection

Rejecting a technique is a scientific claim and carries the same burden as
accepting one. Before a lever is marked rejected in the registry, all four
must hold:

1. **Faithful implementation.** The code must implement the technique as
   specified, verified against the source paper step by step. Three of our own
   rejections failed here: an "asymmetric GPTQ" that never built the required
   cross-Hessian, a Haar transform on the wrong axis without band grouping, a
   proximal operator that was an iterated clip.
2. **Correct target.** The technique must be tested on the tensor family it
   was designed for, not on whichever one is convenient to load.
3. **Sufficient resolution.** The measured margin must exceed the
   measurement's own uncertainty; if the expected effect is small, use a
   paired per-chunk comparison or enlarge the sample first.
4. **Literature checked.** Search for later work that extends or corrects the
   technique. A 2024 method rejected on its original form may have a 2026
   variant that removes the limitation you hit.

A rejection that fails any of these is provisional and must be recorded as
such. The registry is more valuable for its negative results than its positive
ones — which is exactly why they must be trustworthy.
