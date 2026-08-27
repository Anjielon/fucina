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

### Bug #4 — a destructive op read as if it were pure

The second plane, once its file was proven correct and its graph read three
times without finding fault, still destroyed the model: perplexity 9.46 → 70,614
on GPU and 248,320 on CPU. Identical failure on both backends ruled out the
kernels and pointed at graph construction.

The graph did this:

```cpp
mask = ggml_step(ggml_scale_bias(f, -1.0f, n_exp2 - 0.5f));   // reads f
ids2 = ggml_cast(ggml_clamp(f, 0.0f, n_exp2 - 1), I32);        // WRITES INTO f
```

`ggml_clamp` returns `ggml_view_tensor(ctx, a)` — a view sharing `a`'s buffer —
and the CLAMP kernel writes through it. The scheduler is free to run CLAMP
first, and it did. By the time SCALE read `f`, every expert id had already been
clamped to `n_exp2-1`, so `n_exp2 - 0.5 - id` was `+0.5` for every expert and
the mask was **uniformly 1**. Every cold expert received the correction
belonging to the last hot slot, in every layer, for every token.

Two things made it invisible:

- **It reads as correct code.** The mask formula is right, the clamp is right,
  the multiply is right. Nothing is wrong except an aliasing relationship that
  is not visible at the call site.
- **Reading the source did not settle it.** Three careful reads found the graph
  correct. What settled it was `llama-eval-callback`: the pre-threshold mask
  values printed as `0.5, 0.5, 19.5, …` — all positive, when most should have
  been negative. A single line of real numbers ended a search that hours of
  reading had not.

The fix removes the destructive op rather than working around it — the indices
can be derived from the mask itself, writing to nothing:

```cpp
ggml_tensor * m1 = ggml_step(ggml_scale_bias(f, -1.0f, n_exp2 - 0.5f));
ids2 = ggml_cast(ggml_mul(f, m1), I32);   // hot -> id, cold -> 0 (then masked)
```

One number in that failure is worth reading carefully. On CPU the broken model
scored **248320.0000 ± 0.00556**, and the model's vocabulary is **248,320
tokens**. Perplexity equal to the vocabulary size is the exact signature of a
**uniform** output distribution: maximum entropy, zero information. The model
was not degraded, it was annulled — the misapplied corrections, compounded over
41 layers, saturated the network into flat noise. The vanishing error bar says
the same thing: every chunk scored identically, because none of them carried
any signal.

Keep the ratio `perplexity / vocabulary size` in your head. At 1.0 the model is
a random generator and the fault is structural, not a matter of quality. It is
a faster diagnosis than any amount of staring at weights.

Why it had never surfaced upstream: every other call site in llama.cpp uses
the reassignment idiom — `qkv = ggml_clamp(ctx0, qkv, …)` — so the pre-clamp
value is never read again. And `ggml_clamp` is the **only** operation in ggml.c
that returns `ggml_view_tensor(ctx, a)` unconditionally; every other aliasing
op selects with `inplace ? ggml_view_tensor : ggml_dup_tensor`. A lone
exception in an otherwise consistent API is precisely the shape a trap takes.

**Rule: in ggml, know which ops return views into their input.** `ggml_clamp`,
the `*_inplace` family, and anything built on `ggml_view_tensor` write through
to the source. If a tensor feeds two consumers and one of them is destructive,
the graph has a race whose outcome depends on scheduling order — and it will
not announce itself.

**Rule: when reading disagrees with behaviour, print the tensors.** Static
reading is blind to aliasing and execution order. `llama-eval-callback` costs
one run.

Verify the fix the same way you found the bug — at the tensor level, not by
inference from a downstream metric:

| | before | after |
|---|---|---|
| pre-threshold values | `0.5, 0.5, 19.5, …` — all positive | `−97.5, −216.5, 19.5, …` — mostly negative |
| mask | `1,1,1,1,1,1,1,1` (16/16) | `0,0,1,0,…,1,0` (3/16) |
| Tony perplexity, plane 2 on | 70,614 | 12.11 |

Three hot selections out of sixteen is about 19%, which is what 28 hot experts
out of 256 should give once routing frequency is taken into account. A
perplexity that merely *improves* would not have told us the mask was right —
only that something got better.

### Retraction: the date argument was wrong

The section that follows reasoned from provenance — both artefacts predated a
fix, therefore both carried the defect — and concluded that the second plane
could not work on either file. **That conclusion was wrong, and it is retracted
here rather than edited away**, because the way it failed is the useful part.

The direct measurement, made afterwards, reads one expert from the original
checkpoint and compares it against the two planes actually written. On the
397B, layer 59, three positions:

| | plane 1 alone | plane 1 + plane 2 |
|---|---|---|
| gate | 46.21% | **22.32%** |
| up | 46.20% | **22.31%** |
| down | 45.30% | **19.25%** |

And on the small twin, whose file predates the fix by 56 minutes, layer 20:
44.6% → 18.6%, 44.5% → 18.5%, 44.5% → 18.5%. Both files also proved to carry
the hot-first permutation: position 0 holds the expert the imatrix ranks
hottest, while the unpermuted candidates were uncorrelated past 90% and the
test discarded them on its own.

**Both files are sound.** The second plane halves the error on all three
projections in both models. So the defect is entirely in the engine, and the
contradiction is what proves it: with the plane disabled the hot experts carry
44-46% error, with it enabled 18-22%, and yet disabled measures *better*
(6.1845 against 7.1871). No correct engine can produce that ordering.

Three rules, paid for with an hour:

1. **A coherent explanation is not a true one.** The dates lined up perfectly —
   both artefacts before the fix, two independent confirmations, a mechanism
   that explained the sign and roughly the magnitude. Every part of it was
   checkable and none of it was checked against the thing itself.
2. **Measure the artefact, not its history.** Provenance tells you what
   *probably* happened; reading the bytes tells you what *did*. The direct
   measurement took six seconds once the slicing was right.
3. **When ruling something out, prefer the test that could show you wrong.**
   The date check could only ever confirm the hypothesis. The reconstruction
   test could refute it — and did.

### Closing the residual by date, not by measurement (superseded — see above)

After the fix the small model still sat at 12.11 against 9.46 with the plane
off — better by a factor of 5,800, but not yet a win. The obvious move was
another round of measurement. The cheaper one was `git log`:

    model forged            26/08 22:38
    joint-pair fix commit   26/08 23:34   "lessons from the first production night"

The test model predates the fix by 56 minutes, and the commit that fixed the
orphaned plane pair is named after the very run that produced it. Its second
plane was optimized against a plane-1 that the file no longer contains — bug #2
of this document, frozen into a file we then used as a reference.

Two rules: **date your artefacts against your fixes before re-measuring them**,
and **a test bench built before a fix is not a test bench** — it must be rebuilt
with the corrected tool, or every result it produces carries the old defect.

The same check then explained the main model too, and it is worth following
because the conclusion inverted twice:

    397B model forged     26/08 18:59
    small twin forged     26/08 22:38
    joint-pair fix        26/08 23:34

**Both** predate the fix — the large one by four and a half hours. So both carry
the orphaned pair, and the second plane degrades both: 9.46 → 12.11 on the twin,
6.1845 → 7.1871 on the 397B (11.5 sigma, no test needed). Two different models,
same direction, same magnitude relative to their baselines.

What the pre-fix forge actually wrote is the useful part: every expert got the
*dedicated* single-plane quantization (the good one, 28.13%), and the hot ones
additionally got a plane-2 belonging to the *joint* pair. So the shipped file is
a **well-made single-plane model carrying 4.1 GiB of dead weight** that actively
hurts when switched on. That is why disabling the plane is not a fallback here —
it is the correct configuration for this file, and stripping the `*_exps2`
tensors recovers 4.1 GiB at exactly zero quality cost.

The methodological point: we had a prior in-file verification that appeared to
show the pair reconstructing the source well. It could not be reproduced, and
the date evidence is both simpler and stronger. **When a measurement you cannot
re-run disagrees with a provenance check you can, trust the provenance.**

The practical consequence was a shippable improvement rather than a setback.
Stripping the 180 dead tensors and the metadata key that declares them gave a
model measuring **6.1845 ± 0.08705** — identical to the original to the last
digit — at **84.00 GiB instead of 88.16**. Removing weight that does nothing is
not a consolation prize: it is 4.16 GiB of VRAM returned, and it makes it
impossible for a later engine to switch the broken correction back on.

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
| GPTAQ (asymmetric calibration) | never built the required cross term; the "asymmetric" flag disabled GPTQ's intra-block propagation. The measured 32.6% sits exactly between no-GPTQ (38.9) and GPTQ (28.1) — a mutilated GPTQ was rejected, not GPTAQ. Reading the paper properly afterwards showed the requirement is `(X̃−X)Xᵀ` on **paired** token batches, with the Hessian unchanged — a design our capture pipeline never had |
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

### The instrument was already in the toolchain

We hand-rolled a per-chunk sign test before discovering that
`llama-perplexity` already does the paired comparison properly:

    llama-perplexity --kl-divergence-base baseline.kld   # once, on the reference
    llama-perplexity --kl-divergence-base baseline.kld --kl-divergence   # each candidate

On the same chunks it reports the **perplexity ratio with uncertainty
propagated from the logits** — the paired estimator, not two independent error
bars — plus KL divergence, top-1 agreement, and mean/RMS/percentiles of Δp.
The reference does not have to be full precision: pointing it at the previous
build answers "did this change help?" directly.

Two statistical points worth stating plainly, because we got the first wrong:

1. **Two independent ± values are the wrong error bar for a comparison.** The
   variance of a paired difference is `Var(A) + Var(B) − 2·Cov(A,B)`, and when
   both models read the same text that covariance is large and positive. The
   correct interval on the difference can be several times tighter — for free,
   from data already collected. This is the common-random-numbers principle,
   and the paired bootstrap that formalises it has been standard in machine
   translation evaluation since Koehn (2004).
2. **Stratify rather than enlarge.** If you already know per-chunk difficulty
   from the baseline run, stratified selection has provably lower variance than
   taking the next N chunks in file order — again at no extra compute.

### And know where even the better instrument goes blind

`Displacement Is Not Direction` (arXiv 2606.19558) evaluated 28 quantization
variants of Qwen3.6-35B-A3B and 41 of Devstral-Small-24B. KL divergence tracks
downstream score well across the *whole* damage range (ρ ≈ −0.72 / −0.86), and
**loses all correlation near the baseline** (ρ ≈ 0.00 / −0.24, n.s.) — the
"silent zone". The reason is structural: KL measures the *volume* of
disagreement, not its *direction*, and near baseline the helpful and harmful
flips cancel. So the scalar will not adjudicate repair-sized effects either;
what does is the Δp decomposition, which separates symmetric noise from skewed
damage.

Two published reminders that perplexity is the wrong instrument in a second,
independent way — it can move in the *right* direction while real damage
accumulates:

- Gemma-2-2B at INT7: perplexity **improves** while 18.7% of active
  sparse-autoencoder features are damaged; at INT6 only 51.3% survive
  (arXiv 2606.03002).
- Models matched on accuracy after compression can differ sharply in **flip
  rate** and in generative quality (arXiv 2407.09141) — the average over
  thousands of mostly-fine tokens hides a small number of catastrophic ones.

The practical rule: report the paired ratio and the Δp decomposition, keep
perplexity as context, and never let a single scalar close a question on its
own.

## On unified memory, VRAM and RAM are one pool — and the watchdog does not warn

Running a benchmark on the 84 GiB model and a CPU-side activation capture at
the same time reset the machine. There is no out-of-memory message to find
afterwards: the journal simply stops mid-line, with no shutdown sequence. That
absence *is* the diagnosis — it is the signature of a hardware watchdog reset,
fired because init could not answer its ping while the system thrashed.

On a unified-memory machine the GPU allocation is not a separate budget. An
84 GiB model does not leave 30 GiB of host RAM behind it; it leaves whatever
remains of the *single* pool. Two jobs that each fit comfortably alone will
take the machine down together, and they will do it without leaving evidence.

The fix is not vigilance, because vigilance is exactly what fails at midnight.
It is a precondition check that refuses to start:

    spazio_per.sh 84 gpu   →  ⛔ needs 92 GiB (84 + 8 margin), 29 available

Two rules:

1. **One large job at a time, enforced by a gate rather than by intention.**
   The gate takes the requested size, reads the actual free pool, adds a fixed
   margin, and exits non-zero. Wire it as the first line of every heavy script.
2. **A journal that stops mid-line is evidence, not missing evidence.** A clean
   OOM leaves a kill message; a watchdog reset leaves nothing. If the last line
   is ordinary and the next is a boot banner, stop looking for the error and
   start looking at what was resident.

## A tool must not act when it is merely imported

This class of defect appeared twice in one evening, and the second instance
was dangerous.

The first was mild: `build_hessians.py` had top-level code with hardcoded
default paths, so importing it created a directory and announced "Hessians
written". Harmless there, but a tool that acts on import is a defect, not a
style preference.

The second was not mild. The bf16 forge had the same shape — module-level code
that reads the base model, opens the *output* GGUF for writing, writes a
header, and seeks to a resume point. Importing it to check the translation
started a forge: it created a directory and an 88 MB partial file before dying
on an unrelated error. Nothing was lost, because the path it chose happened to
be empty. Had a real model been there, it would have been overwritten.

Two rules:

1. **Every script gets `if __name__ == "__main__": main()`.** Not for style —
   because importing is how you inspect, test, and document a module, and
   inspection must never mutate the world.
2. **A tool that writes should not know its own destination.** Defaults that
   point at a real artefact turn a stray import into data loss. Require the
   path, or default to somewhere provably empty.

The verification that caught it is worth keeping too: after any refactor,
import every module in the package and see what happens. It is one loop, it
found five defects here, and three of them were invisible to reading.

### Never edit a shell script that is currently running

A related trap, hit the same evening. A long pipeline was executing from a
shell script, and extending that script to add a final stage seemed harmless —
the new lines came after the current position. It is not harmless: the shell
keeps a byte offset into the file and re-reads from it, so inserting text
shifts everything after and the next read can start mid-line.

The fix is not to be careful about where you insert. It is to **not edit the
file at all**: restore it byte-for-byte and add the new stage as a separate
process that waits for the first to finish. Orchestration belongs outside the
script being orchestrated.

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

### Duplicate expert ids: tested, and not the culprit

`test-backend-ops` builds MUL_MAT_ID indices as `data[i] = i % n_experts`
followed by a shuffle, which makes every id in a row **distinct**. A masked
second-plane path clamps most slots onto the same index, so the entire suite
could pass while that regime was never exercised.

We added `MOGAVIS_DUP_IDS=<cap>` to generate rows that repeat the same expert,
and re-ran the TQ1_0 cases: **83/83 pass**. Duplicate ids are handled
correctly; the hypothesis is closed, and the environment variable stays as
permanent coverage for anyone porting a masked expert path.

Recording this matters as much as a positive result would: the next person to
suspect duplicate ids can read that it was measured, not assumed.
