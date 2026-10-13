"""Lever census invariants: every lever added on 12-13 Oct is an opt-in module with provenance, switch and evidence."""
from __future__ import annotations

import re

from fucina.census import census_markdown
from fucina.registry import load_techniques

NEW = ("gptq_k1", "jarq_k1", "bil_k1_jarq", "osaq", "schuropt_k1", "foem", "act_order_sign", "drq",
       "scale_groups_g64", "ssm_dt_bias", "fair_calib", "hsvdquant", "flow_statistics", "zero_byte_scales",
       "sequential_re_ptq", "tesseraq", "e2e_qp", "on_policy_distillation", "cardan_experts_k1",
       "dense_residual_kl4", "k_scale_answer_kl4", "e2m_stage2", "dense_ptq_before_expert_kd", "family_lever_search")


def test_new_levers_are_registered_opt_in_modules():
    ts = load_techniques()
    for n in NEW:
        t = ts[n]
        assert t.default_off, n                                   # automatic runs keep the recorded recipes
        assert t.lab_status in ("PASS", "FAIL", "NEUTRAL", "PREPARED", "RUNNING"), n
        assert t.variant in ("faithful", "thor", "original", "control"), n
        assert t.measured and all(m["evidence"] for m in t.measured), n
        assert re.search(r"arXiv \d{4}\.\d{5}", t.source) or t.variant == "original", n
        assert t.switch() and t.driver, n


def test_census_renders_every_module_without_proprietary_terms():
    ts = load_techniques().values()
    full, short = census_markdown(ts), census_markdown(ts, short=True)
    for t in ts:
        assert f"`{t.name}`" in full and f"`{t.name}`" in short
    for text in (full, short):
        assert not re.search(r"\bfold|piega", text, re.I)
