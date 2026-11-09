# Replaying a recorded chain: `fucina replay`

`fucina replay` re-runs a recorded quantization chain on another machine, from public inputs, and checks every step
against the record. The reference recipe is [`examples/thor/recipe_c.yaml`](../examples/thor/recipe_c.yaml): the
chain from the released frozen root of Thor-T (R2B3m) to **C**, the single-plane checkpoint whose trunk is the released
`Thor-T-35B-A3B-XS` (733 of its 753 tensors are byte-identical to C; the other 20 are the MTP head, block 40).

## Quick start

```bash
pip install -e ".[test]" huggingface_hub          # the fetch step uses the `hf` CLI
git clone --branch thor-t-v1 https://github.com/Anjielon/llama.cpp && cmake -S llama.cpp -B llama.cpp/build -DGGML_VULKAN=ON && cmake --build llama.cpp/build -j

# 1. plan and check everything, run nothing (CPU, seconds)
fucina replay examples/thor/recipe_c.yaml --dry-run \
    --set work=/data/thor-c --set engine=$PWD/llama.cpp/build/bin \
    --set bundle_src=/data/thor-c-code --set venv=/data/venv --script /data/thor-c/commands.sh

# 2. run (resumable; a verified step is skipped)
fucina replay examples/thor/recipe_c.yaml --run --set ...      # same --set values
fucina replay examples/thor/recipe_c.yaml --run --from I1 --to I3   # a segment
```

Roots can also come from a `replay:` section of the fucina configuration or from `FUCINA_REPLAY_<ROOT>` variables.

## What the dry run proves

For every selected step it resolves the commands and checks, without writing anything:

* every placeholder resolves and every absolute path lies under a configured root (or `/usr`, `/bin`, `/dev/null`):
  the replay reads nothing else on the machine. `--forbid PREFIX` (repeatable) additionally fails if a command
  contains that prefix — on the author's machine, `--forbid /home/<author> --forbid /mnt/` proves that no private
  directory is reached even where it exists;
* every input is either external with a pinned locator (`repo@revision`) and a recorded hash, or produced by an
  earlier step;
* every code reference exists (`package:` modules of this package, `bundle:` files of the code bundle);
* each step's status: `ported` (code in this package), `bundle` (frozen research code), `blocked` (with the reason),
  and the estimated GPU and CPU hours.

## The recipe format

See the docstring of [`fucina/replay/recipe.py`](../fucina/replay/recipe.py). In short: `roots` (the only
machine-specific values), `inputs` (`hf` pinned files, `package` data, `derived` artifacts), `steps` (needs,
produces + verification, bash commands with `{root}` / `{path.KEY}` placeholders, code references, status,
determinism class, estimated hours) and a `target`.

Verification kinds: `sha256` (file), `manifest` (per-file sha256 of a directory, recorded under
`examples/thor/recipe_c/recorded/`), `tensors` (sha256 over every GGUF tensor's name, type, shape and bytes:
[`gguf_hash.py`](../fucina/replay/gguf_hash.py)), `mmlu` (the `Final result` of `llama-perplexity` within a band),
`exists`.

## The code bundle

Most steps of recipe C run research code that is not yet part of this package (about 21,000 lines in 56 modules,
written in Italian; 27 of the 83 modules the chain imports are already ported to English and checked by AST
equivalence, see `scripts/check_port_ast.py`). The replay runs that code **as recorded**, from a *code bundle*:

* `fucina replay-bundle build examples/thor/recipe_c/bundle_sources.yaml --out DIR --root models=... --root forge=...`
  (on the recording machine) copies the listed source trees, keeps code and small provenance files only, replaces
  every recorded absolute root by a token, scans for secrets and writes `BUNDLE.json` (sha256 of every file, every
  substitution, the absolute paths that are left);
* step `L0` of the recipe installs it (`fucina replay-bundle install`) with the replay roots in place of the tokens,
  and links the trees at the places of the recorded directory layout under `{work}/layout`.

Nothing else is needed: no private repository snapshot, no mount namespace, no development engine build.
Because the paths change, the two writers that embed input paths in a GGUF metadata note produce files that differ
from the record in that note only; chain GGUFs are therefore checked by tensor digest.

## Thor-T helpers in this package

* `fucina.replay.unpack_planes` — the released root is the *packed* GGUF (multi-plane families as
  `TQ1_0R_1024_X3` / `TQ1_0_X3`, the form the release engine loads). Step `U0` unpacks it into the separate-plane
  layout the recorded code reads; it is the exact inverse of the engine's `tools/ternary/pack_planes.py` (same file
  bytes, checked by sha256). `--hash-only` verifies without writing the 8.8 GB file.
* `fucina.replay.thor.s7_onpolicy` — step 27, the on-policy distillation rounds with the student server and the
  probe as child processes (the trainer argv of every round is identical to the recorded launcher's).
* `fucina.replay.thor.post_export` — second half of step 31: export of the POST checkpoint (scales, F32 tensors),
  zero-byte fold of the dense residuals, byte checks, MMLU-2000 on the mixed bench and McNemar against the replay's T400.

## Determinism classes

| class | meaning | check |
|---|---|---|
| `bitwise` | CPU step, same inputs give the same bytes | sha256 / manifest |
| `tolerance` | GPU reductions or training: not bit-reproducible | recorded manifest reported; the next CPU step and the final target decide |
| `stochastic` | sampled rollouts (S7) | final target only |
| `measurement` | an engine measurement | value within a band |

The target of recipe C: tensor digest `bc36d18b…` (exact replay) or, as expected after GPU training, MMLU-2000
within 1.0 point of 73.15 and exact McNemar p ≥ 0.05 against the recorded per-question log.
