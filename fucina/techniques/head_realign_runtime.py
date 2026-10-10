"""F3 head/embedding/MTP, F4 realignment (ONCE, on top of the final structure), F5 runtime.
Wrapped scripts: ``letter/e2_scrivi_gguf.py`` (E2 rank1), the RILQ-T runs ``corsa.sh`` / ``corsa_shexp.sh``,
engine environment variables (int8 activations)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from ..executor import Job, bash
from ..measure import Measurement, read_bench, read_kl, read_mcnemar
from ..registry import Outcome, Thresholds, register
from ._base import ScriptTechnique, cfg


@register
class E2Rank1(ScriptTechnique):
    """E2 rank1: debias of the letter rows of the output head (rank-1 bias direction)"""
    name = "e2_rank1"
    phase = "F3"
    pieces = ("head",)
    gpu_cost_h = 1.2
    expected_kl_gain_pct = 0.0
    params = {"letter_ids": "357,417,351,414"}
    measurement_aliases = (r"E2 rank1",)
    source = "arXiv 2409.18857 (E2/BNP: ARC 52.3 -> 56.7)"
    tools = ("letter",)

    def jobs(self, ctx: Any, piece: str) -> list:
        c = cfg()
        o = self.output_dir(ctx, piece)
        g = self.product_gguf(ctx, piece)
        u = c.path("letter_run")
        s = (f"mkdir -p {o}; cd {c.path('letter')}; HIP_VISIBLE_DEVICES=-1 CUDA_VISIBLE_DEVICES=-1 nice -n 15 "
             f"{c['python']} e2_scrivi_gguf.py --src {ctx.candidate.gguf} --out {g} "
             f"--letter-ids {self.param(ctx, 'letter_ids')} --method rank1 --bias {u}/e2_rank1/bias_direction.pt "
             f"--hidden {u}/hid")
        return [Job(id="F3_e2_rank1", name="E2 rank1 on the output head (CPU)", command=bash(s), est_h=0.0,
                    output_dir=o)]

    def switch(self) -> str:
        return "file: back to the starting GGUF"


@register
class Rilq(ScriptTechnique):
    """RILQ-T: rank-64 ternary residuals on attn+gdn(+shared expert) trained on the full KL against the teacher"""
    name = "rilq"
    phase = "F4"
    pieces = ("attn", "gdn", "shexp")
    judged_by = "training"
    slows_down = True
    gpu_cost_h = 10.0
    expected_kl_gain_pct = -15.0
    est_extra_bytes = 48_000_000
    params = {"hours": 10, "families": "attn,gdn,shexp", "min_steps": 200}
    measurement_aliases = (r"RILQ-THOR", r"RILQ")
    source = "arXiv 2412.01129 (RILQ: -51 % of the log-PPL gap at W2); reproduced small-scale -52.3 % (audit A5)"
    tools = ("rilq_shexp_run", "bin_shexp")
    engine = "bin_shexp"
    thresholds = Thresholds(kl_min_pct=-5.0, sigma=2.0, tg_max_loss_pct=10.0)
    self_measuring = True             # the run scripts measure paired KL, tg and MMLU against their start

    def own_measurement(self, ctx: Any, piece: str) -> Optional[Measurement]:
        o = Path(self.output_dir(ctx, piece))
        k = o / "kl" / "kl_rilq.json"
        if not k.exists():
            return None
        m = read_kl(str(k), str(o / "kl" / "kl_partenza.json"))
        if (o / "bench_rilq.txt").exists():
            m.tg = read_bench(str(o / "bench_rilq.txt"))["tg"]
        if (o / "bench_partenza.txt").exists():
            m.tg_base = read_bench(str(o / "bench_partenza.txt"))["tg"]
        if (o / "mcnemar.json").exists():
            mm = read_mcnemar(str(o / "mcnemar.json"))
            m.mmlu, m.mmlu_base, m.p_mcnemar, m.wins, m.losses = mm.mmlu, mm.mmlu_base, mm.p_mcnemar, mm.wins, mm.losses
        g = Path(self.new_gguf(ctx, piece))
        m.file_bytes = g.stat().st_size if g.exists() else None
        return m.complete()

    def shexp(self, ctx: Any) -> bool:
        return "shexp" in self.param(ctx, "families")

    def output_dir(self, ctx: Any, piece: str) -> str:
        return f"{ctx.outputs_dir}/rilq" + ("_shexp" if self.shexp(ctx) else "")

    def new_gguf(self, ctx: Any, piece: str) -> str:
        return (f"{self.output_dir(ctx, piece)}/{ctx.candidate.model}-FUCINA-RILQ"
                + ("-shexp" if self.shexp(ctx) else "") + ".gguf")

    def jobs(self, ctx: Any, piece: str) -> list:
        c = cfg()
        o = self.output_dir(ctx, piece)
        script = c.path("rilq_shexp_run") if self.shexp(ctx) else f"{c.path('rilqt')}/corsa.sh"
        copy = f"{o}/corsa_fucina.sh"
        # the validated run only knows its own starting points: a COPY with a FUCINA entry (start = current
        # candidate) inserted before the script's catch-all case (whose message reads "nome sconosciuto")
        entry = f"  FUCINA) ST={ctx.candidate.gguf}; LIMITE={ctx.cap_bytes} ;;"
        s = (f"mkdir -p {o}; sed 's|^  \\*) echo \"nome sconosciuto|{entry}\\n&|' {script} > {copy} && "
             f"grep -qF 'FUCINA) ST={ctx.candidate.gguf}' {copy} || {{ echo 'copy of the run script failed'; exit 2; }}; "
             f"RQ_NOME=FUCINA RQ_ORE={self.param(ctx, 'hours')} RQ_OUT={o} bash {copy}")
        return [Job(id="F4_rilq", name=f"RILQ-T ({self.param(ctx, 'families')}) on top of the final structure, teacher "
                    f"corpus, {self.param(ctx, 'hours')} h budget (training + evaluation)", command=bash(s), gpu=True,
                    training=True, est_h=float(self.param(ctx, "hours")) + 2.0, output_dir=o, memory_max="60G")]

    def switch(self) -> str:
        return "THOR_SPEGNI=lr (lrattn,lrgdn,lrshexp)"


@register
class Int8(ScriptTechnique):
    """int8 activations in the engine (ternary mat-vec with Q8 activations)"""
    name = "int8"
    phase = "F5"
    pieces = ()
    judged_by = "speed"
    gpu_cost_h = 1.2
    expected_kl_gain_pct = 0.0
    measurement_aliases = (r"LEVA int8", r"int8 attivazioni")
    source = "in-house lever (GGML_VK_TQ1R_Q8): MMLU 74.65 vs 74.60, tg 50.4 -> 56.3"
    thresholds = Thresholds(kl_min_pct=0.0, sigma=2.0, tg_max_loss_pct=0.0)

    def product_gguf(self, ctx: Any, piece: str) -> str:
        return ctx.candidate.gguf

    def switch(self) -> str:
        return "GGML_VK_TQ1R_Q8=0"

    def env(self) -> dict:
        return {"GGML_VK_TQ1R_Q8": "1"}


@register
class Mtp(ScriptTechnique):
    """MTP head: speculative decoding (acceptance and tok/s)"""
    name = "mtp"
    phase = "F5"
    pieces = ("mtp",)
    judged_by = "speed"
    gpu_cost_h = 0.5
    tools = ("mtp_measure",)
    source = "plan of 8 Oct: MTP head to be measured (acceptance and tok/s) on PV12P/RILQ"

    def cheap_test(self, ctx: Any) -> Outcome:
        if self.missing():
            return Outcome("UNINFORMATIVE", {"missing": self.missing()}, "the MTP measurement tool is missing (to be written)")
        return Outcome("INFORMATIVE")

    def product_gguf(self, ctx: Any, piece: str) -> str:
        return ctx.candidate.gguf

    def switch(self) -> str:
        return "llama-server without --draft (MTP off)"
