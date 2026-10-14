"""Sequential re-quantization (stages S2+S3 of the home-GPU plan): layer-by-layer statistics on the clean and
quantized streams, closed-form / JARQ / FastKron solvers for dense tensors and experts, and the GGUF writer.

Ported from the original code base with identical computation; see ``docs/GLOSSARY_IT_EN.md``."""
