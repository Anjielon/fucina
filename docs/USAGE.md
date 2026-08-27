# Usage

This guide walks through quantizing a Mixture-of-Experts model end to end,
then repairing it safely. Commands are real and runnable; the numbers in the
examples come from an actual 397B run.

## Requirements

- a llama.cpp build with TQ1_0 support. CPU kernels are upstream; GPU (Vulkan)
  kernels are proposed in ggml-org/llama.cpp#27765 and available meanwhile
  from the fork linked in the README.
- PyTorch with a working GPU backend (CUDA or ROCm)
- the model in GGUF form, plus an importance matrix (`llama-imatrix`) that
  includes per-expert routing counts
- disk: roughly 1× the output size for the model, plus room for activations

## 1. Prepare the ingredients

```bash
# routing counts and activation statistics
llama-imatrix -m model.gguf -f calibration.txt -ngl 999 --ctx-size 2048 \
              -o imatrix.gguf

# per-layer Hessians, from captured residual-stream activations
NDIM=4096 python build_hessians.py dumps/ hessians/
#   layer  0:  16640 samples · spectral anisotropy alpha = 1.170
```

A note on the calibration corpus: **size changes the answer**. With 2,000
tokens, 35 of 60 layers appeared to benefit from a repair; with 7,500 tokens
only 22 did, and the difference was not noise — it was the smaller corpus
producing false positives.

## 2. Forge

```bash
python forge_gguf.py \
    --sorgente model.gguf \
    --uscita   model-ternary.gguf \
    --caldi    28 \
    --imatrix  imatrix.gguf \
    --hessiane hessians/
```

Output, one line per expert tensor:

```
[712/1278] blk.7.ffn_up_exps.weight · boundary 18.5% · 37.9 M/s
```

`boundary` is the per-layer verification: the hottest expert is rebuilt from
the bytes actually written and compared against the source. If it exceeds the
threshold the forge stops rather than producing a plausible-looking bad file.

The run is resumable. `<output>.giornale` records the tensor index and byte
offset after every write; re-running the same command continues from there.

## 3. Validate

```bash
python safe_repair.py model-ternary.gguf --corpus wiki.test.raw --chunks 30
# PPL = 6.2091 +/- 0.0874
```

Always report the uncertainty. Compare two models only at equal corpus and
equal chunk count — the absolute value moves with both.

## 4. Repair, with a gate

Repairs are the dangerous part: a technique that improves its own internal
metric can degrade the model. The gate exists to make that impossible to miss.

```python
from safe_repair import repair_with_gate

def my_repair(path):
    ...  # modify the copy in place

repair_with_gate(
    "model-ternary.gguf", my_repair,
    corpus="wiki.test.raw",
    engine="/path/to/llama.cpp/build",
    label="router-fix",
)
```

```
measuring the outcome BEFORE the repair...
  PPL before: 6.2091 +/- 0.0874
copying the model (88.2 GiB)...
  PPL after:  6.1904 +/- 0.0871 · margin +0.0187 vs noise 0.0874
⛔ repair DISCARDED (gain smaller than the measurement noise, original untouched)
```

### Repairs in fractions

When a repair has many parts, do not apply them all. Measured on a 60-layer
model, applying the router correction to every layer that showed a gain:

| layers applied | perplexity (threshold 5.8100) |
|---|---|
| 35 (small corpus) | 6.8784 |
| 10 (best-ranked) | 5.8740 |
| **3 (best-ranked)** | **5.8074** |

The benefit concentrates in the first few layers; the damage accumulates with
every layer touched. Sweep the fraction rather than assuming a yes/no answer.

## 5. Rolling back without copying

When a repair touched only a few small tensors, restoring them is seconds:

```bash
python selective_rollback.py good.gguf current.gguf --filter ffn_gate_inp
# differing tensors: 35
# guard (unexpected tensors differing): 0 — OK

python selective_rollback.py good.gguf current.gguf --filter ffn_gate_inp --apply
# ✅ restored 35 · residual differences: 0
```

The guard is the important part: it samples tensors that should *not* have
changed. If any of them differs, the diagnosis is wrong and the rollback
refuses to run.

## 6. When a rebuilt model behaves strangely

If you load quantized weights back into a framework (for teacher-student
repairs) and the block output is orders of magnitude off, correlate every
tensor against the original weights:

```bash
python diagnose_assembly.py
#   ✅ input_layernorm.weight        1.000
#   ⛔ mlp.experts.gate_up_proj      0.000   ← loaded wrong
```

Four traps this catches, all encountered in practice: reversed axis order,
experts reordered hot-first by the forge itself, unit axes omitted by the
GGUF, and values stored already exponentiated.

One caveat: some tensors legitimately differ by an internal head reordering
performed by the converter. If the scale and the value histogram match but the
point-wise error is large, you are looking at a permutation, not a defect.

## Sampling

Below two bits per weight, sampling settings change the verdict:

```
--temp 0.6 --min-p 0.03 --top-p 0.9 --presence-penalty 1.5
```

Never XTC. Enable reasoning for hard tasks — it solves problems the direct
path does not. Greedy decoding can degenerate into token loops on a model that
is otherwise healthy; a test harness using it will fail a working model.
