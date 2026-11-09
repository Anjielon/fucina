# DEPRECATED CPU engines in this directory (2026-11-03)

Frozen code: files are NOT edited (SHA_CONGELATO.txt). Deprecated, kept as extra CPU lanes and fallback:
- cardan_piega.py (serial original) -> use cardan_piega_veloce.py with FORGIA_BABAI_TRITON=1, layer-parallel (byte-identical; Triton accepted 2026-10-22)
- cardan_piega_lotti.py (superseded) -> cardan_piega_veloce.py
- cardan_piega_densi.py --device cpu -> cardan_piega_densi_gpu_veloce.py --device cuda, early from the base

Registry: @@ROOT:forge@@/fucina/docs/DEPRECATED_CPU_ENGINES.md
