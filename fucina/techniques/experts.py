"""F2 — extra capacity on the experts. Wrapped scripts: ``rd_ternary.unisci`` + ``rd_ternary.scrivi_piano_gguf``
(second plane chosen by measured dKL), ``combo_trials/leva_k.py`` (K scale), ``levers_g4/cardan_costruisci.py`` +
``fucina.tools.write_residuals`` (CARDAN), ``rot/scrivi_prism.py`` (PrismQuant)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..executor import Job, bash
from ..registry import Outcome, register
from ._base import ScriptTechnique, cfg, lock_prelude

EXPERTS = ("experts_gate_up", "experts_down")
_WRITE_RESIDUALS = "fucina.tools.write_residuals"


def _package_root() -> str:
    """Directory containing the ``fucina`` package (jobs run ``python -m fucina.tools...`` from there)."""
    return str(Path(__file__).resolve().parents[2])


@register
class SecondPlaneDkl(ScriptTechnique):
    """Second ternary plane on the experts chosen by measured dKL (se-k 1), up to the objective's size cap"""
    name = "second_plane_dkl"
    phase = "F2"
    pieces = EXPERTS
    model_types = ("moe", "hybrid")
    slows_down = True
    structural = True                 # changes the structure: consolidated at the end of the phase
    gpu_cost_h = 3.0
    expected_kl_gain_pct = -12.0
    est_extra_bytes = 645_000_000
    params = {"se_k": 1, "plans": ["piano_parte0.json", "piano_parte1.json", "piano_parte2.json"]}
    measurement_aliases = (r"^PV12P\b", r"pv12\b", r"^PRECISO\b", r"dkl64_esperti", r"oggi9", r"\bpv9\b")
    source = "rule second_plane_from_dkl (PV12P 78.80 at 12.13 GiB; pv12 78.50)"
    tools = ("rd_ternary", "dkl_merged", "rd_plans")

    def cheap_test(self, ctx: Any) -> Outcome:
        e = super().cheap_test(ctx)
        if e.verdict != "INFORMATIVE":
            return e
        return Outcome("INFORMATIVE", {"cap_bytes": ctx.cap_bytes},
                       f"plan from the merged dKL under {ctx.cap_bytes} B (CPU, minutes)")

    def new_gguf(self, ctx: Any, piece: str) -> str:
        return f"{self.output_dir(ctx, piece)}/{ctx.candidate.model}-second-plane-{ctx.cap_bytes}.gguf"

    def jobs(self, ctx: Any, piece: str) -> list:
        c = cfg()
        o = self.output_dir(ctx, piece)
        g = self.product_gguf(ctx, piece)
        cap = ctx.cap_bytes
        py, rd = c["python"], c.path("rd_ternary")
        plans = " ".join(f"{c.path('rd_plans')}/{p}" for p in self.param(ctx, "plans"))
        merge = (f"mkdir -p {o}; cd {rd}; RD_TOLL_DIST=0.15 nice -n 15 {py} -m rd_ternary.unisci {plans} "
                 f"--expect-layers 0-40 --delta-kl {c.path('dkl_merged')} --unmeasured cur --loadable-only "
                 f"--motore piani-variabili --gguf {ctx.candidate.gguf} --max-file-bytes {cap} --scale-cur sorgente "
                 f"--se-k {self.param(ctx, 'se_k')} --out {o}/piano.json")
        w = lambda src, layers, dst, n: (
            f"RD_TOLL_DIST=0.15 {py} -m rd_ternary.scrivi_piano_gguf --gguf {src} --piano {o}/piano.json "
            f"--motore piani-variabili --max-bytes {cap} --device auto --strati {layers} --out {dst} "
            f"> {o}/scrivi_{n}.log 2>&1")
        write = (f"cd {rd}; {lock_prelude('FUCINA-2PLANE', 40)}"
                 f"{w(ctx.candidate.gguf, '0-13', o + '/g1.gguf', 1)} && {w(o + '/g1.gguf', '14-27', o + '/g2.gguf', 2)} && "
                 f"rm -f {o}/g1.gguf && {w(o + '/g2.gguf', '28-40', g, 3)} && rm -f {o}/g2.gguf")
        return [Job(id="F2_second_plane_merge", name=f"second plane: plan from the dKL (CPU), cap {cap} B",
                    command=bash(merge), est_h=0.1, output_dir=o),
                Job(id="F2_second_plane_write", name="second plane: GGUF written in 3 passes", command=bash(write),
                    gpu=True, est_h=self.gpu_cost_h, depends_on=["F2_second_plane_merge"], output_dir=o)]

    def switch(self) -> str:
        return "THOR_SPEGNI=p2d,p2gu"


@register
class KScale(ScriptTechnique):
    """K scale: fp16 scales of single-plane experts multiplied by k (zero bytes)"""
    name = "k_scale"
    phase = "F2"
    pieces = EXPERTS
    model_types = ("moe", "hybrid")
    gpu_cost_h = 0.6
    expected_kl_gain_pct = -1.0
    params = {"k": 1.045}
    measurement_aliases = (r"leva K\b", r"PRECISO_K", r"PV12P_K", r"K1\.045")
    source = "R029 item 30 (k 1.045)"
    tools = ("combo_trials",)

    def jobs(self, ctx: Any, piece: str) -> list:
        c = cfg()
        o = self.output_dir(ctx, piece)
        g = self.product_gguf(ctx, piece)
        s = (f"mkdir -p {o}; cp --reflink=auto {ctx.candidate.gguf} {g}.part && "
             f"nice -n 15 {c['python']} {c.path('combo_trials')}/leva_k.py {g}.part {self.param(ctx, 'k')} "
             f"> {o}/k_scale.txt && mv {g}.part {g}")
        return [Job(id="F2_k_scale", name=f"K scale {self.param(ctx, 'k')} on single-plane experts (CPU, copy)",
                    command=bash(s), est_h=0.2, output_dir=o)]

    def switch(self) -> str:
        return "file: k=1 (back to the starting GGUF, zero bytes)"


@register
class Cardan(ScriptTechnique):
    """CARDAN: shared rank-r basis across experts + per-expert coefficients (lr branches in the cardan engine)"""
    name = "cardan"
    phase = "F2"
    pieces = ("experts_gate_up",)
    model_types = ("moe", "hybrid")
    per_byte = True
    gpu_cost_h = 1.0
    expected_kl_gain_pct = -2.0
    params = {"rank": 16}
    est_extra_bytes = 120_000_000
    measurement_aliases = (r"CARDAN",)
    source = "lever #20 CARDAN (G4)"
    tools = ("levers_g4", "bin_cardan")
    engine = "bin_cardan"

    def jobs(self, ctx: Any, piece: str) -> list:
        c = cfg()
        o = self.output_dir(ctx, piece)
        g = self.product_gguf(ctx, piece)
        r = self.param(ctx, "rank")
        py = c["python"]
        s = (f"mkdir -p {o}; export HIP_VISIBLE_DEVICES=-1 CUDA_VISIBLE_DEVICES=-1; cd {c.path('levers_g4')}; "
             f"nice -n 15 {py} cardan_costruisci.py --rango {r} --gguf {ctx.candidate.gguf} --out {o}/r{r} --thread 8 && "
             f"cd {_package_root()} && nice -n 15 {py} -m {_WRITE_RESIDUALS} --writer-dir {c.path('e1_letter')} "
             f"--src {ctx.candidate.gguf} --dir {o}/r{r} --out {g}")
        return [Job(id="F2_cardan", name=f"CARDAN r{r}: build and GGUF (CPU, 8 threads)", command=bash(s),
                    est_h=0.0, output_dir=o)]

    def switch(self) -> str:
        return "THOR_SPEGNI_LR=1"


@register
class Prism(ScriptTechnique):
    """Additive PrismQuant: exact rank-k term along the eigenvectors of H_ffn (expert gate/up)"""
    name = "prism"
    phase = "F2"
    pieces = ("experts_gate_up",)
    model_types = ("moe", "hybrid")
    per_byte = True
    gpu_cost_h = 1.5
    expected_kl_gain_pct = -2.0
    params = {"k": 16, "projections": "gate,up", "mode": "add"}
    est_extra_bytes = 340_795_872
    measurement_aliases = (r"\bPA16\b", r"\bPR16\b", r"Prism")
    source = "arXiv 2609.32429 (PrismQuant)"
    tools = ("rot", "bin_prism")
    engine = "bin_prism"

    def jobs(self, ctx: Any, piece: str) -> list:
        c = cfg()
        o = self.output_dir(ctx, piece)
        g = self.product_gguf(ctx, piece)
        p = lambda k: self.param(ctx, k)
        rot = c.path("rot")
        # scrivi_prism.py reads its source from a constant: over another candidate a COPY with SRC replaced is run
        # (PYTHONPATH on the original directory for its local imports)
        src = ctx.candidate.gguf
        copy = f"{o}/scrivi_prism_fucina.py"
        s = (f"mkdir -p {o}; sed \"s|^SRC = .*|SRC = '{src}'|\" {rot}/scrivi_prism.py > {copy} && "
             f"grep -qF \"SRC = '{src}'\" {copy} || {{ echo 'copy of scrivi_prism failed'; exit 2; }}; "
             f"cd {rot}; export HIP_VISIBLE_DEVICES=-1 CUDA_VISIBLE_DEVICES=-1 PYTHONPATH={rot}; "
             f"nice -n 15 {c['python']} {copy} {g} {p('k')} {p('projections')} {p('mode')} > {o}/write.log 2>&1")
        return [Job(id="F2_prism", name=f"Prism k={p('k')} {p('mode')}: GGUF write (CPU)", command=bash(s),
                    est_h=0.0, output_dir=o)]

    def switch(self) -> str:
        return "THOR_SPEGNI=prism"
