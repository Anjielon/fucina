"""Offline checks of run_recipe_s.py and recipe_s/recorded (no GPU, no multi-GB files needed)."""
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_recipe_s as R  # noqa: E402

REC = HERE / "recipe_s" / "recorded"
HEX64 = re.compile(r"\b[0-9a-f]{64}\b")


def test_steps_ordered_and_classified():
    ids = [s.id for s in R.STEPS]
    assert ids == [f"{i:02d}" for i in range(13)]
    ok = {"bitwise-verified", "bitwise-expected", "tolerance", "measure", "frozen", "gap"}
    assert all(s.determinism in ok for s in R.STEPS)
    assert all(s.cmd for s in R.STEPS if s.kind in ("cpu", "gpu", "measure"))


def test_recorded_gguf_hashes_in_chain_file_and_doc():
    chain = (REC / "gguf_chain.sha256").read_text()
    doc = (HERE / "RECIPE_S.md").read_text()
    for key, (_, sha, _) in R.G.items():
        assert HEX64.fullmatch(sha), key
        assert sha in chain, key
        assert sha[:16] in doc, key


def test_release_hash_consistent():
    assert R.G["S"][1] in (REC / "release_s.sha256").read_text()


@pytest.mark.parametrize("name,n", [("t100_tensors.manifest", 1576), ("release_s_tensors.manifest", 969),
                                    ("sel_pred_cardan_tensori.manifest", 829), ("cardan_fold_tensori.manifest", 152),
                                    ("sel_pred_tensori.manifest", 677)])
def test_manifest_counts(name, n):
    lines = [l for l in (REC / name).read_text().splitlines() if l.strip() and not l.startswith("#") and HEX64.match(l) and l.split()[1] != ".NON_CANCELLARE"]
    assert len(lines) == n


def test_recorded_launchers_and_logs_present():
    for f in ("bpost_b_1101.sh", "cardan_kd_comando_20261010_0053.sh", "comando_combinazione.sh"):
        assert (REC / "launchers" / f).is_file()
    assert (REC / "logs" / "provenienza.txt").is_file()


def test_every_step_renders_without_placeholders():
    for s in R.STEPS:
        if s.cmd:
            for v in ("verify", "run"):
                assert not re.search(r"\{[A-Za-z0-9_]+\}", R.render(s, v)), (s.id, v)


def test_run_mode_hides_record_and_verify_never_writes_it():
    for s in R.STEPS:
        if s.kind in ("gpu", "measure"):
            c = R.render(s, "run")
            assert c.startswith("unshare --user --map-root-user --mount"), s.id
            assert all(root in c for root in R.MIRRORED[:3])
    for s in R.STEPS:
        if s.rerun:
            c = R.render(s, "verify")
            assert str(R.OUT) in c.split("--out")[-1] or str(R.OUT) in c


def test_gpu_measure_step_takes_the_lock():
    for s in (x for x in R.STEPS if x.kind == "measure" or x.id == "09"):
        assert "lucchetto.sh" in R.render(s, "run") and "prendi_gpu" in R.render(s, "run"), s.id


def test_launchers_use_the_lock_themselves():
    for f in ("bpost_b_1101.sh", "cardan_kd_comando_20261010_0053.sh", "grad_eb882K_kl4.sh"):
        assert "prendi_gpu" in (REC / "launchers" / f).read_text() or "lucchetto" in (REC / "launchers" / f).read_text(), f


def test_cli_dry_run_and_tamper(tmp_path, monkeypatch):
    out = subprocess.run([sys.executable, str(HERE / "run_recipe_s.py"), "--dry-run"], capture_output=True, text=True,
                         env={"PATH": "/usr/bin:/bin", "FORGIA_OUT": str(tmp_path)})
    assert out.returncode == 0 and "9d5afbf85afd0613" in out.stdout
    # a wrong recorded hash must make --verify fail
    f = tmp_path / "x.gguf"
    f.write_bytes(b"abc")
    res = []
    R.check_file("gguf X", str(f), "0" * 64, res)
    assert res[0][1] == "DIFFER"
    res = []
    R.check_file("gguf X", str(f), R.sha_of(f), res)
    assert res[0][1] == "MATCH"
