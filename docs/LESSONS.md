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
