"""Lever census: a markdown view of the registered techniques, their provenance, switch and measured status.

``fucina techniques --census`` renders ``docs/LEVERS.md``; ``--census --short`` renders the module table of the
README. Nothing is measured here: the numbers are the ``measured`` records declared by each module."""
from __future__ import annotations

from typing import Iterable


def _cell(s: object) -> str:
    return str(s if s not in (None, "") else "—").replace("|", "\\|").replace("\n", " ")


def _description(t) -> str:
    d = getattr(t, "description", None)
    return d() if callable(d) else ((t.__doc__ or t.name).strip().splitlines()[0])


def _switch(t) -> str:
    on = f"opt-in: `fucina enable {t.name}`" if getattr(t, "default_off", "") else "on in automatic runs"
    return f"{on}; off: `fucina disable {t.name}` · {t.switch() or 'engine level'}"


def census_markdown(techniques: Iterable, short: bool = False) -> str:
    ts = sorted(techniques, key=lambda t: (t.phase, t.name))
    if short:
        out = ["| module | phase | pieces | reference | variant | status | default |",
               "|---|---|---|---|---|---|---|"]
        for t in ts:
            out.append("| " + " | ".join(_cell(x) for x in (
                f"`{t.name}`", t.phase, ", ".join(t.pieces) or "whole model", t.source,
                getattr(t, "variant", ""), getattr(t, "lab_status", "") or getattr(t, "suspended", "") and "SUSPENDED",
                "opt-in" if getattr(t, "default_off", "") else "suspended" if getattr(t, "suspended", "") else "on"))
                + " |")
        return "\n".join(out)
    out = []
    for t in ts:
        out += [f"### `{t.name}` — {_description(t)}", ""]
        out.append(f"- **Phase / pieces**: {t.phase} · {', '.join(t.pieces) or 'whole model'}")
        out.append(f"- **Reference**: {_cell(t.source)}")
        if getattr(t, "variant", ""):
            out.append(f"- **Variant**: {t.variant}" + (f" — declared deviations: {t.deviations}"
                                                         if getattr(t, "deviations", "") else ""))
        conf = dict(getattr(t, "params", {}) or {})
        out.append("- **Parameters**: " + (", ".join(f"`{k}={v}`" for k, v in conf.items()) or "none"))
        out.append(f"- **Switch**: {_switch(t)}")
        if getattr(t, "driver", ""):
            out.append(f"- **Driver**: `paths.{t.driver}` (configuration)")
        if getattr(t, "correction_hook", False):
            out.append("- **Post-processing**: the correction is written as extra tensors; an optional closed external "
                       "hook (not part of Fucina) may post-process it")
        if getattr(t, "suspended", ""):
            out.append(f"- **Suspended**: {t.suspended}")
        st = getattr(t, "lab_status", "")
        ms = getattr(t, "measured", ())
        if st or ms:
            out += ["", f"Status: **{st or 'n/a'}**", ""]
        if ms:
            out += ["| obj. | reference host | value | vs | wins/losses | McNemar p | notes | evidence |",
                    "|---|---|---|---|---|---|---|---|"]
            for m in ms:
                out.append("| " + " | ".join(_cell(m.get(k)) for k in (
                    "objective", "host", "value", "vs", "wins_losses", "p", "extra", "evidence")) + " |")
        out.append("")
    return "\n".join(out)
