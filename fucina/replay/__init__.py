"""Replay of recorded quantization chains from public inputs.

* :mod:`fucina.replay.recipe` — the YAML schema (roots, inputs, steps, target) and its validation;
* :mod:`fucina.replay.runner` — dry run (resolution and private-path checks), run, verification, state;
* :mod:`fucina.replay.bundle` — the frozen research code of a recorded chain, made relocatable;
* :mod:`fucina.replay.gguf_hash` — tensor-level identity of GGUF files;
* :mod:`fucina.replay.inputs` — inputs regenerated from public data (calibration selection, corpora).

See ``docs/REPLAY.md`` and ``examples/thor/recipe_c.yaml``.
"""
