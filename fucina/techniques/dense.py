"""F1 — quantizer of the dense tensors (attention, GatedDeltaNet, shared expert, output head).

Mutually exclusive alternatives (group ``dense_quantizer``): Thor-T's base is the three-plane recipe of R2B3m; the
refinement levers change the same bytes (same format, unchanged engine) and are judged by realised KL against the
base. Wrapped scripts: ``levers_g4/bil_gptq.py`` + ``sostituisci.py`` (BIL), ``refine/applica.py`` + ``patch_gguf.py``
(JARQ/Tied/Schur/QTEA/G2PTQ); shQ8 is a known product (not ternary)."""
from __future__ import annotations

from typing import Any

from ..executor import Job, bash
from ..registry import register
from ._base import ScriptTechnique, cfg, lock_prelude, lock_release

DENSE = ("attn", "gdn", "shexp", "head")


@register
class ThreePlanes(ScriptTechnique):
    """Three greedy ternary planes on the dense tensors (R2B3m recipe): the starting structure"""
    name = "three_planes"
    phase = "F1"
    pieces = DENSE
    group = "dense_quantizer"
    judged_by = "structure"
    gpu_cost_h = 0.0
    measurement_aliases = (r"^R2B3m$",)
    source = "Thor-T recipe (R2B3m 8,846,968,608 B, MMLU-2000 74.60)"

    def product_gguf(self, ctx: Any, piece: str) -> str:
        return cfg().path("structure_three_planes")

    def jobs(self, ctx: Any, piece: str) -> list:
        return [Job(id="F1_three_planes", name="rebuild R2B3m from bf16 (recipe steps 1-9, not automated)",
                    command=bash("echo 'R2B3m is rebuilt with the Thor-T recipe (steps 1-9)'; exit 3"),
                    gpu=True, est_h=6.0)]

    def switch(self) -> str:
        return "(base structure)"


@register
class Bil(ScriptTechnique):
    """Balanced ternary (scales 9D/3D/D, uniform 27-level grid) + GPTQ on the dense tensors and the head"""
    name = "bil"
    phase = "F1"
    pieces = DENSE
    group = "dense_quantizer"
    per_byte = True
    gpu_cost_h = 1.5
    expected_kl_gain_pct = -3.0
    measurement_aliases = (r"G4 BIL", r"\bBIL\b")
    source = "arXiv 2609.05641 (nuclear limit -> uniform grid); offline weighted error 0.054 -> 0.026"
    tools = ("levers_g4",)

    def jobs(self, ctx: Any, piece: str) -> list:
        c = cfg()
        o = self.output_dir(ctx, piece)
        g = self.product_gguf(ctx, piece)
        py = c["python"]
        s = (f"mkdir -p {o}; {lock_prelude('FUCINA-BIL', 20)}cd {c.path('levers_g4')}; timeout 3600 nice -n 5 {py} "
             f"bil_gptq.py --tensori densi+testa --device cuda --gguf {ctx.candidate.gguf} --out {o}/tensori; "
             f"{lock_release()}nice -n 15 {py} sostituisci.py --src {ctx.candidate.gguf} --out {g} {o}/tensori")
        return [Job(id="F1_bil", name="BIL/Tied 27 levels + GPTQ: build (GPU 1 h) and GGUF (CPU)",
                    command=bash(s), gpu=True, est_h=self.gpu_cost_h, output_dir=o)]

    def switch(self) -> str:
        return "file: same size and format -> switched off by going back to the starting GGUF"


class _Refine(ScriptTechnique):
    """Refinement of the dense quantizer (refine/applica.py)"""
    phase = "F1"
    pieces = ("attn", "gdn", "shexp")
    group = "dense_quantizer"
    per_byte = True
    gpu_cost_h = 1.0
    lever = ""
    tools = ("refine",)

    def jobs(self, ctx: Any, piece: str) -> list:
        c = cfg()
        o = self.output_dir(ctx, piece)
        g = self.product_gguf(ctx, piece)
        py, r = c["python"], c.path("refine")
        s = (f"mkdir -p {o}; export LEVE_DT=float32 LEVE_OUT={o}; "
             f"{lock_prelude(f'FUCINA-REFINE-{self.lever}', 20, 'fail')}"
             f"nice -n 5 {py} -u {r}/applica.py --leva {self.lever} --device cuda; {lock_release()}"
             f"nice -n 15 {py} {r}/patch_gguf.py --base {ctx.candidate.gguf} --dir {o}/{self.lever}/tensori --out {g}")
        return [Job(id=f"F1_{self.name}", name=f"{self.name}: dense refinement (GPU ~1 h) + GGUF patch (CPU)",
                    command=bash(s), gpu=True, est_h=self.gpu_cost_h, output_dir=o)]

    def switch(self) -> str:
        return "file: same size and format -> switched off by going back to the starting GGUF"


@register
class Jarq(_Refine):
    """JARQ: joint scales + Babai rounding on the composite of the 3 planes"""
    name, lever = "jarq", "jarq_p123"
    expected_kl_gain_pct = -2.0
    measurement_aliases = (r"jarq_p123", r"\bJARQ\b")
    source = "arXiv 2609.38599 (JARQ)"


@register
class Tied(_Refine):
    """Tied Trit-Planes: 27 uniform levels per group (official solver)"""
    name, lever = "tied", "tied_uff"
    expected_kl_gain_pct = -2.0
    measurement_aliases = (r"tied_uff", r"tied_jarq", r"\bTied\b")
    source = "arXiv 2608.08910 (Tied Trit-Planes)"


@register
class Qtea(_Refine):
    """QTEA: planes 2 and 3 redone with ternary GPTQ + error decay"""
    name, lever = "qtea", "qtea_p23"
    expected_kl_gain_pct = -1.0
    measurement_aliases = (r"qtea_p23", r"\bQTEA\b")
    source = "arXiv 2609.00224 (QTEA)"


@register
class Schur(_Refine):
    """SchurOpt on the composite of planes 2-3"""
    name, lever = "schur", "schur_p23"
    measurement_aliases = (r"schur_p", r"SchurQuant")
    source = "arXiv 2608.15567 (SchurOpt)"
    suspended = "suspended by the owner on 8 Oct (relaunch script: stato/coda_sospesa_2026-10-08.sh)"


@register
class G2ptq(_Refine):
    """G2PTQ: GPTQ guided by the gradients of the KL (planes 2-3)"""
    name, lever = "g2ptq", "g2ptq_p23"
    measurement_aliases = (r"g2ptq", r"G2PTQ")
    source = "G2PTQ (official code, audit G2 #17)"
    suspended = "suspended by the owner on 8 Oct (needs the raff-grad gradients)"


@register
class ShQ8(ScriptTechnique):
    """Shared expert in rotated Q8 (RotaryQuant): NOT fully ternary"""
    name = "shq8"
    phase = "F1"
    pieces = ("shexp",)
    ternary = False
    gpu_cost_h = 0.7
    est_extra_bytes = 300_000_000
    measurement_aliases = (r"shQ8",)
    source = "RotaryQuant (lever #18)"
    tools = ("rot",)
    engine = "bin_prism"

    def switch(self) -> str:
        return "file: switched off by going back to the starting GGUF"
