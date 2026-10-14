"""GPU kernels (Triton) and their PyTorch fallbacks used by the end-to-end ternary student.

Every module is a renamed, translated port of the code that produced the Thor-T results; the computation is
unchanged (see ``docs/GLOSSARY_IT_EN.md`` and ``tests/test_kernels_equivalence.py``)."""
