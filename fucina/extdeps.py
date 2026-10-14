"""Location of external dependencies and data used by the GPU kernels, the student and the sequential solver.

Nothing machine-specific is hard-coded in these modules: everything they need from outside the package is found
through environment variables, so the same code runs on any host.

=========================  ===================================================================================
variable                   meaning
=========================  ===================================================================================
``FUCINA_GGUF_PY``         ``gguf-py`` directory of the llama.cpp build that knows the rotated ternary types
                           (``TQ1_0R_1024``); unset: the installed ``gguf`` package is used.
``FUCINA_RD_TERNARY_DIR``  directory containing the ``rd_ternary`` package (plane reader/writer for GGUF);
                           unset: it must be importable already.
``FUCINA_REFINE_DIR``      directory of the refinement levers (module ``comune``), used by the CARDAN fold;
                           unset: it must be importable already.
``FUCINA_DATA_DIR``        base directory for default data files (calibration corpora, reference GGUF files,
                           Hugging Face configuration); default: the current directory.
=========================  ===================================================================================

Each module calls :func:`add_path` exactly where the original code extended ``sys.path``, so the import order is
unchanged.
"""
from __future__ import annotations

import os
import sys
from typing import Optional

ENV_GGUF_PY = "FUCINA_GGUF_PY"
ENV_RD_TERNARY = "FUCINA_RD_TERNARY_DIR"
ENV_REFINE = "FUCINA_REFINE_DIR"
ENV_DATA = "FUCINA_DATA_DIR"


def add_path(env_var: str) -> Optional[str]:
    """Insert the directory named by ``env_var`` at the front of ``sys.path`` (once). Returns it, or None if unset."""
    path = os.environ.get(env_var)
    if not path:
        return None
    if path not in sys.path:
        sys.path.insert(0, path)
    return path


def data_path(*parts: str) -> str:
    """Default location of a data file: ``$FUCINA_DATA_DIR/<parts...>`` (``./<parts...>`` when unset)."""
    return os.path.join(os.environ.get(ENV_DATA, "."), *parts)
