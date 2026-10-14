"""F0 — measured dKL per group (expert/tensor, option +1 plane): the diagnosis the second plane is chosen from.
Wraps ``rd_ternary/dkl_raccogli.py``. If the merged dKL table already exists, no job is queued.

The command-line options and environment variables in the job below (``--strati``, ``--blocco``, ``--righe-kl``,
``--solo-esperti``, ``--seme``, ``E2E_MOTORE``, ``E2E_CONV_DIRETTA``, ...) are the interface of that external,
not yet ported tool and must keep its spelling; their English names are listed in ``docs/GLOSSARY_IT_EN.md``.
``measurement_aliases`` matches the wording of the existing lab notebook."""
from __future__ import annotations

import os
from typing import Any

from ..executor import Job, bash
from ..registry import Outcome, register
from ._base import ScriptTechnique, cfg, lock_prelude


@register
class MeasuredDkl(ScriptTechnique):
    """Whole-model dKL measured per expert (+1 plane) on the teacher's held-out rows"""
    name = "measured_dkl"
    phase = "F0"
    pieces = ("experts_gate_up", "experts_down")
    model_types = ("moe", "hybrid")
    judged_by = "diagnosis"
    gpu_cost_h = 6.0
    params = {"kl_rows": 32, "block": 3, "seed": 2, "min_rows": 64}
    tools = ("rd_ternary_dkl", "recovery", "teacher_cache_tt6")
    measurement_aliases = (r"dKL preciso",)
    source = "rule second_plane_from_dkl (RD1/ALLOC1 from an imatrix proxy: -1.8 MMLU)"

    def tsv(self) -> str:
        return cfg().path("dkl_merged")

    def cheap_test(self, ctx: Any) -> Outcome:
        if os.path.exists(self.tsv()) or getattr(ctx, "simulate", False):
            return Outcome("PASS", {"tsv": self.tsv(), "rows": 64},
                           "merged dKL already measured (64 rows, seeds 1+2): no GPU needed")
        return super().cheap_test(ctx)

    def product_gguf(self, ctx: Any, piece: str) -> str:
        return ""

    def jobs(self, ctx: Any, piece: str) -> list:
        if os.path.exists(self.tsv()) or getattr(ctx, "simulate", False):
            return []
        c = cfg()
        o = f"{ctx.outputs_dir}/dkl"
        p = lambda k: self.param(ctx, k)
        rec = c.path("recovery")
        s = (f"mkdir -p {o}; {lock_prelude('FUCINA-DKL', 80, 'none')}cd {c.path('rd_ternary_dkl')}; "
             f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True E2E_GGUF={ctx.candidate.gguf} E2E_GDN=veloce "
             f"E2E_GDN_MODULO=gdn_veloce_corsa_v5 E2E_MOTORE=triton2 E2E_CONV_DIRETTA=1 E2E_SENZA_COESISTENZA=1 "
             f"E2E_CACHE_DEQUANT={c.path('dequant_cache')} PYTHONPATH={rec}/thor_v2 "
             f"{c['python']} dkl_raccogli.py --thor-v2 {rec}/thor_v2 --teacher-cache {c.path('teacher_cache_tt6')} "
             f"--out {o} --strati 0-39 --blocco {p('block')} --righe-kl {p('kl_rows')} --righe-fisher 8 --gruppi esperto "
             f"--solo-esperti --opzioni +1 --seme {p('seed')}")
        return [Job(id="F0_dkl", name="measured dKL of the experts (+1 plane)", command=bash(s), gpu=True,
                    est_h=self.gpu_cost_h, output_dir=o)]

    def switch(self) -> str:
        return "(diagnosis: no switch)"
