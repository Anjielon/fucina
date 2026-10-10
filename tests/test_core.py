"""Core tests (CPU, seconds): pieces, phases, registry, resumable state, database, rules, configuration."""
import tempfile

import pytest

from fucina import candidate, config, db, phases, pieces, registry


def test_pieces():
    assert pieces.classify("blk.3.ffn_down_exps2") == "experts_down"
    assert pieces.classify("blk.7.ffn_gate_shexp") == "shexp"
    assert pieces.classify("output3") == "head"
    assert pieces.classify("blk.40.attn_k", mtp_layer=40) == "mtp"


def test_phase_order():
    assert [p.code for p in phases.PHASES] == ["F0", "F1", "F2", "F3", "F4", "F5", "F6"]
    assert phases.phase("F4").training


def test_registry_and_admissibility():
    @registry.register
    class Fake(registry.Technique):
        name, phase, pieces = "fake_test", "F2", ("experts_down",)

        def apply(self, ctx, piece):
            return registry.Change("x.gguf", "THOR_SPEGNI=fake")

        def switch(self):
            return "THOR_SPEGNI=fake"
    t = registry._REGISTRY["fake_test"]
    assert t.admits("F2", "experts_down", "moe", {})
    assert not t.admits("F1", "experts_down", "moe", {})


def test_state_resume():
    with tempfile.TemporaryDirectory() as d:
        s = candidate.RunState(d)
        c = candidate.Candidate("Thor-T", "hybrid", "a.gguf", {"mmlu_min": 80.05})
        c.choices.append(candidate.Choice("F2", "experts_down", "x", "to_decide"))
        s.save(c)
        c2 = s.load()
        assert c2.to_decide()[0].technique == "x"


def test_db_and_rules():
    with tempfile.TemporaryDirectory() as d:
        m = db.Measurements(f"{d}/m.sqlite")
        m.add(model="Thor-T", technique="int8", mmlu=74.65, mmlu_base=74.60, params={"q8": 1})
        assert m.by_technique("int8")[0]["mmlu"] == 74.65
    ids = [r["id"] for r in db.rules_for("F4", "attn", "hybrid")]
    assert "realignment_steps" in ids


def test_config_paths_and_defaults(tmp_path):
    c = config.Config({"paths": {"a": "/x"}, "runs_dir": "runs"}, str(tmp_path / "cfg.yaml"))
    assert c.path("a") == "/x" and c.resolve("b/c") == "b/c"
    assert c["runs_dir"] == str(tmp_path / "runs")
    with pytest.raises(config.ConfigError, match="paths: {missing"):
        c.path("missing")
    assert c.lock("acquire") == "acquire_gpu"
    with pytest.raises(config.ConfigError):
        c.lock_source()
    assert config.Config().get("cycle_log") is None


def test_no_home_paths_in_the_package():
    """Machine-specific locations live in the configuration, never in the package."""
    import fucina
    from pathlib import Path
    root = Path(fucina.__file__).parent
    for f in root.rglob("*.py"):
        text = f.read_text()
        for needle in ("/home/", "/mnt/", "tony-forgia", "tony-tern"):
            assert needle not in text, f"{f.name} hard-codes {needle}"
