# Publishing plan

Written from a deliberate survey rather than guesswork, and kept here so the
next session does not have to re-derive it.

## The paper next to ours

**Tied Trit-Planes** (arXiv 2608.08910, 9 Aug 2026) — a single unaffiliated
author, two ternary planes, disk-streamed MoE, a negative result in its own
abstract. Three weeks old and adjacent on every axis: same primitive, same
deployment constraint, same kind of finding.

It does **not** pre-empt this work — it applies the ladder over *tensor
classes* rather than layers and states outright that its design "localizes
sensitivity to the final cumulative bundle, not to a single component", and it
contains no per-layer analysis and no activation statistics. But two of our
framings are weakened by it: proxy/fidelity dissociation is already claimed in
this exact setting, and a mild non-monotonicity is already noted (0.01 at 2,048
tokens). Position against it explicitly, in the first paragraph of related
work, before a reviewer does it for us.

Also cite **PTQTP** (arXiv 2509.16989) for the dual trit-plane method itself —
it applies the decomposition uniformly and has no per-layer analysis.

## Framing

The finding is the **localisation**, not the failure. Write the abstract so the
result is "a correction that halves reconstruction error is catastrophic on one
projection of a narrow band of layers and beneficial everywhere else, and the
proxy cannot see it" — not "we tried something and it did not work".

The claim is **not** "more bits hurt", which is false and refutable in a line.
It is **heterogeneous fidelity hurts**: consistency across components matters
more than fidelity of any one of them. Three independent literatures converge on
this, and two published tables already contain the effect (GEMQ arXiv
2605.23078 Table 1, unremarked; HiFloat4 arXiv 2607.26515, remarked and named).

Precedents worth imitating, both accepted at TMLR, neither apologetic:
*Oscillations Make Neural Networks Robust to Quantization* and *Amdahl's Law
for LLMs*. They name the phenomenon and sell the ceiling as the deliverable.

## Venue

- **TMLR** — the target. Rolling submission, no page limit, and its criteria
  explicitly exclude novelty and SOTA in favour of correctness and clarity,
  which is the right bar for a careful negative/localisation result.
- **MLSys 2027**, deadline 30 Oct — the archival shot, if the third-model
  experiment lands.
- **arXiv** — start the submission now, purely to generate the endorsement
  code: it is the longest-latency item on the list and nothing else depends on
  its outcome. For an unaffiliated author the practical route is the "Which
  authors of this paper are endorsers?" link on any abs page, which identifies
  who qualifies without needing to know the (unpublished) threshold.

## Evaluation — what has to change before submission

⛔ **The lm-eval path through `llama-server` does not work.** The `gguf` backend
POSTs `{"echo": true, ...}` to `/v1/completions` and llama-server rejects
`echo` (ggml-org/llama.cpp#12591; the PR that would add it, #15189, is open and
unmerged). Loglikelihood tasks therefore fail. The working path:

    pip install "llama-cpp-python[server]"
    python3 -m llama_cpp.server --model model.gguf --n_gpu_layers -1 --n_ctx 4096 --port 8000
    lm_eval --model gguf --model_args base_url=http://localhost:8000,max_length=4096 \
            --tasks hellaswag,winogrande,arc_challenge

Do **not** use `--model hf --model_args gguf_file=...`: it dequantises into
torch, so it measures the weights rather than llama.cpp's kernels.

Pin the harness version and state the metric. The corpus practice is poor —
of 25 recent quantization papers, 4 state `acc` against `acc_norm` and 4 pin a
version — which is an argument for doing it, not for skipping it.

Move the headline metric to **KL divergence** (*Accuracy is Not All You Need*,
arXiv 2407.09141, NeurIPS 2024). llama.cpp computes it natively with
`--kl-divergence`, reporting mean KLD, mean Δp and same-top-p%.

For subsampling use a **named** method — tinyBenchmarks (arXiv 2402.14992),
100-item IRT-selected subsets, published estimation error under 2%. "We
evaluated a random thousand" is the easiest objection to raise.

## Artefact release

- **Zenodo, FigShare or Dryad** for the archival deposit. MLSys runs formal
  artifact evaluation under ACM badging, and the cTuning rules it applies state
  that GitHub, GitLab and personal pages are *not* acceptable archival
  locations. Hugging Face alone does not earn "Artifacts Available".
- **Ship a runnable reduced-config forge**, not only the weights. The
  contribution is the transform; an artifact that cannot be re-run fails
  "Functional". Include a small-calibration, single-size, eval-subset mode a
  reviewer finishes in an afternoon, and declare a tolerance on every number.
- **Publish the imatrix and the KLD base logits** alongside the model, so the
  numbers are independently re-derivable. Exemplar:
  `eaddario/gemma-3-12b-it-GGUF`.
- **License is inherited from the base model** and the obligations differ:
  Apache-2.0 is trivial; the Llama Community License requires the repo name to
  begin with `Llama`, a NOTICE with Meta's attribution, "Built with Llama", and
  propagation of the acceptable-use policy. Set `base_model_relation:
  quantized` explicitly rather than letting it be inferred.

## Known weaknesses, to state rather than hide

Two models, same architecture family, one calibration set, no ablation over
*which* experts receive the extra precision. The third model — a different
architecture, not merely a different size — is the single experiment that moves
the paper most, and it needs three lines of engine change written *before* the
run or it silently measures nothing.

## Pinned lm-eval protocol (researched 2026-08-28 — run before submission)

Verified against master source: llama-server still rejects `echo` on
/v1/completions ("Only no echo is supported" in server-common.cpp), so the
path is lm-eval's `gguf` model against `python -m llama_cpp.server`, with
our fork's libllama loaded via `LLAMA_CPP_LIB` (build shared from
build-llamacpp-tq1; llama-cpp-python's ctypes loader honours it).

Pin: `lm-eval==0.4.12` (re-check PyPI at run time). Flags that matter:
`max_length=4096` (GGUFLM default 2048 silently truncates MMLU 5-shot),
`tokenizer=<HF path>` explicit, chat template OFF (loglikelihood = raw),
`--seed 1234 --log_samples`, acc+acc_norm where upstream defines them
(winogrande and mmlu have NO acc_norm upstream — state it, don't invent it).
Wall-clock: serial HTTP → HellaSwag full ~3-6h, MMLU 5-shot the worst;
schedule overnight or on the second bench.

```bash
export FORK=/home/angelo/build-llamacpp-tq1
cmake -S $FORK -B $FORK/build-shared -DGGML_VULKAN=ON -DBUILD_SHARED_LIBS=ON && cmake --build $FORK/build-shared -j
export LLAMA_CPP_LIB=$FORK/build-shared/bin/libllama.so
pip install "lm-eval==0.4.12" llama-cpp-python
python -m llama_cpp.server --model MODEL.gguf --n_gpu_layers -1 --port 8899 &
for T in hellaswag winogrande arc_easy arc_challenge; do
  lm_eval --model gguf --model_args base_url=http://127.0.0.1:8899,max_length=4096 \
    --tasks $T --num_fewshot 0 --seed 1234 --batch_size 1 --output_path results/$T.json --log_samples
done
lm_eval --model gguf --model_args base_url=http://127.0.0.1:8899,max_length=4096 \
  --tasks mmlu --num_fewshot 5 --seed 1234 --batch_size 1 --output_path results/mmlu.json --log_samples
```
