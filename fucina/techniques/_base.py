"""Common base of the wrappers: a technique = one or more jobs (commands of validated scripts) + a switch.
Registers nothing (no @register here)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from .. import config
from ..executor import Job
from ..measure import Measurement
from ..registry import Change, Outcome, Technique, Thresholds


def cfg() -> config.Config:
    return config.current()


def lock_prelude(who: str, gib: int, ram_guard: str = "wait") -> str:
    """Shell prelude that waits for RAM, takes the machine-wide GPU lock and releases it on exit.

    ``ram_guard``: ``wait`` (poll every 2 minutes), ``fail`` (exit 3 when RAM is short) or ``none``."""
    c = cfg()
    guard = {"wait": f"until {c.lock('ram_guard')}; do sleep 120; done; ",
             "fail": f"{c.lock('ram_guard')} || exit 3; ", "none": ""}[ram_guard]
    return (f". {c.lock_source()}; {guard}{c.lock('acquire')} '{who} $$' {gib}; "
            f"trap {c.lock('release')} EXIT; ")


def lock_release() -> str:
    return f"{cfg().lock('release')}; trap - EXIT; "


class ScriptTechnique(Technique):
    measurement_aliases: tuple = ()   # regexes (case-insensitive) matching the labels used for it in historic records
    group: str = ""                   # mutually exclusive alternatives within a phase (e.g. the dense quantizer)
    judged_by: str = "kl"             # kl | speed | diagnosis | training | structure
    expected_kl_gain_pct: float = -1.0   # prior estimate (offline test or paper) used for ordering, in % of KL
    suspended: str = ""               # why it does not run automatically (owner's decision)
    params: dict = {}
    tools: tuple = ()                 # config path keys (or literal paths) that must exist to really launch it
    engine: str = "bin_planes"        # config path key of the inference engine it must be measured with
    per_byte: bool = False            # "per-byte" lever: adds capacity at almost constant size
    structural: bool = False          # changes the structure: consolidated at the end of its phase
    architectures: tuple = ("qwen35moe",)   # GGUF architectures for which the wrapped scripts are validated
    self_measuring: bool = False      # True = the last job measures by itself (paired KL/tg/MMLU): no extra measurement

    # ── preflight ──
    def missing(self) -> list:
        out = []
        for s in self.tools:
            if "/" not in s and s not in cfg().paths:
                out.append(f"<paths.{s} not configured>")
                continue
            p = cfg().resolve(s)
            if not Path(p).exists():
                out.append(p)
        return out

    def cheap_test(self, ctx: Any) -> Outcome:
        miss = self.missing()
        if miss and not getattr(ctx, "simulate", False):
            return Outcome("UNINFORMATIVE", {"missing": miss}, "missing tools: " + ", ".join(miss[:3]))
        return Outcome("INFORMATIVE", reason="tools present" if not miss else "simulation")

    # ── construction ──
    def output_dir(self, ctx: Any, piece: str) -> str:
        return f"{ctx.outputs_dir}/{self.name}"

    def known_products(self) -> list:
        """GGUF files already written by a validated run, from the configuration:
        ``techniques.<name>.known_products: [{base, product, cap_min?, cap_max?}]`` (base/product: path keys or
        paths; the optional caps restrict the entry to a range of the objective's size cap)."""
        return list(cfg().technique(self.name).get("known_products") or [])

    def product_gguf(self, ctx: Any, piece: str) -> str:
        for k in self.known_products():
            if cfg().resolve(k["base"]) != ctx.candidate.gguf:
                continue
            if "cap_min" in k and ctx.cap_bytes < int(k["cap_min"]):
                continue
            if "cap_max" in k and ctx.cap_bytes > int(k["cap_max"]):
                continue
            return cfg().resolve(k["product"])
        return self.new_gguf(ctx, piece)

    def new_gguf(self, ctx: Any, piece: str) -> str:
        return f"{self.output_dir(ctx, piece)}/{ctx.candidate.model}-{self.name}.gguf"

    def jobs(self, ctx: Any, piece: str) -> list:
        """Jobs that build the GGUF (dependencies in order). Default: none (engine-level lever)."""
        return []

    def apply(self, ctx: Any, piece: str) -> Change:
        product = self.product_gguf(ctx, piece)
        ids = []
        if not (product and os.path.exists(product)):            # cache: a GGUF already written is not rewritten
            suffix = getattr(ctx, "suffix", "")          # the same job over different candidates = different ids
            for j in self.jobs(ctx, piece):
                j.phase, j.technique, j.piece = self.phase, self.name, piece
                if suffix:
                    j.id = f"{j.id}_{suffix}"
                    j.depends_on = [f"{d}_{suffix}" for d in j.depends_on]
                j.depends_on = list(j.depends_on) + list(getattr(ctx, "depends_on", []) or [])
                j.placeholder = bool(getattr(ctx, "placeholder", False))
                j = ctx.executor.enqueue(j)
                ids.append(j.id)
        return Change(gguf=product or ctx.candidate.gguf, switch=self.switch(), extra_bytes=self.est_extra_bytes,
                      job_ids=ids)

    def switch(self) -> str:
        return ""

    def own_measurement(self, ctx: Any, piece: str) -> Optional[Measurement]:
        """With self_measuring: read the results written by the last job (Measurement or None)."""
        return None

    def param(self, ctx: Any, key: str, default: Any = None) -> Any:
        conf = {**self.params, **(cfg().technique(self.name).get("params") or {})}
        return (getattr(ctx, "params", {}) or {}).get(self.name, {}).get(key, conf.get(key, default))

    def description(self) -> str:
        return (self.__doc__ or "").strip().splitlines()[0] if self.__doc__ else self.name


def thresholds(**kw: Any) -> Thresholds:
    return Thresholds(**kw)
