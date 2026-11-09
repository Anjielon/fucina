"""Replay machinery: recipe validation, dry run of recipe C, GGUF tensor identity, code bundle, input tools.

CPU only, no network, no model files: the dry run of recipe C uses a stand-in bundle made of empty files."""
from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import pytest
import yaml

from fucina.replay import bundle, gguf_hash, merge_shards, recipe as R, runner
from fucina.replay.inputs import multiple_choice as mc

ROOT = Path(__file__).resolve().parents[1]
RECIPE_C = ROOT / "examples" / "thor" / "recipe_c.yaml"
PRIVATE = ("/home/angelo", "/mnt/models", "/mnt/deposito", "tony-forgia", "odino-lab")


def _fake_bundle(r: R.Recipe, where: Path) -> Path:
    for s in r.steps:
        for c in s.code:
            kind, _, target = c.partition(":")
            if kind == "bundle":
                p = where / target
                p.parent.mkdir(parents=True, exist_ok=True)
                p.touch()
    (where / "BUNDLE.json").write_text("{}")
    return where


def _roots(r: R.Recipe, tmp: Path) -> dict:
    (tmp / "engine").mkdir(exist_ok=True)
    (tmp / "venv" / "bin").mkdir(parents=True, exist_ok=True)
    sets = [f"work={tmp / 'work'}", f"engine={tmp / 'engine'}", f"bundle_src={_fake_bundle(r, tmp / 'bsrc')}",
            f"venv={tmp / 'venv'}"]
    return runner.configured_roots(r, sets)


def test_recipe_c_loads_and_names_its_target():
    r = R.load(RECIPE_C)
    assert r.target["expect"] == "bc36d18b7fd8257403653132fa0fa3db353b51c000304646322ef38c731b036b"
    ids = [s.id for s in r.steps]
    assert ids.index("13") < ids.index("25") < ids.index("29") < ids.index("31")
    # every external input is pinned: HF inputs have a revision and a recorded hash
    for k, i in r.inputs.items():
        if i.kind == "hf":
            assert i.revision and (i.sha256 or i.manifest), k
    assert r.inputs["bf16"].revision.startswith("10fbf86f")


def test_recipe_c_has_no_private_path():
    text = RECIPE_C.read_text()
    for p in PRIVATE:
        assert p not in text, p


def test_recipe_c_dry_run_resolves_everything(tmp_path):
    r = R.load(RECIPE_C)
    roots = _roots(r, tmp_path)
    sel = runner.select(r, None, None, None, with_optional=True)
    script = tmp_path / "commands.sh"
    res = runner.plan(r, roots, sel, forbid=list(PRIVATE), show=False, script=script)
    assert res["problems"] == []
    assert set(res["blocked"]) == {"I9", "27", "31"}
    body = script.read_text().replace(str(RECIPE_C.parent), "RECIPE").replace(str(ROOT / "fucina"), "PACKAGE")
    for p in PRIVATE:
        assert p not in body
    assert not (tmp_path / "work").exists(), "a dry run must not create the work tree"


def test_dry_run_reports_a_path_outside_the_roots(tmp_path):
    r = R.load(RECIPE_C)
    roots = _roots(r, tmp_path)
    roots["engine"] = "/opt/some/engine"          # not under work: commands then point outside the roots... allowed
    res = runner.plan(r, roots, r.steps[:1], forbid=["/opt/some"], show=False)
    assert any("forbidden prefix /opt/some" in p for p in res["problems"])


@pytest.mark.parametrize("bad, message", [
    ({"run": ["cat /etc/passwd"]}, "literal absolute path"),
    ({"run": ["echo {nowhere}"]}, "unknown placeholder"),
    ({"needs": ["ghost"]}, "needs 'ghost'"),
    ({"status": "blocked"}, "blocked without blockers"),
])
def test_validation_rejects(tmp_path, bad, message):
    step = {"id": "1", "title": "t", "kind": "cpu", "status": "ported", "run": ["true"], **bad}
    p = tmp_path / "r.yaml"
    p.write_text(yaml.safe_dump({"name": "x", "roots": {"work": {}}, "inputs": {}, "steps": [step]}))
    with pytest.raises(R.RecipeError, match=message):
        R.load(p)


def test_run_verifies_and_resumes(tmp_path):
    out = tmp_path / "work"
    content = b"hello replay\n"
    sha = hashlib.sha256(content).hexdigest()
    doc = {"name": "mini", "roots": {"work": {}}, "inputs": {}, "steps": [
        {"id": "a", "title": "write", "kind": "cpu", "status": "ported",
         "run": ["printf 'hello replay\\n' > {work}/a.txt", "echo ran >> {work}/count"],
         "produces": [{"key": "a", "path": "{work}/a.txt", "verify": "sha256", "expect": sha}]},
        {"id": "b", "title": "copy", "kind": "cpu", "status": "ported", "needs": ["a"],
         "run": ["cp {path.a} {work}/b.txt"], "produces": [{"key": "b", "path": "{work}/b.txt", "verify": "exists"}]},
    ]}
    p = tmp_path / "mini.yaml"
    p.write_text(yaml.safe_dump(doc))
    r = R.load(p)
    roots = runner.configured_roots(r, [f"work={out}"])
    assert runner.run(r, roots, r.steps) == 0
    assert runner.run(r, roots, r.steps) == 0                      # resumed: nothing re-run
    assert (out / "count").read_text() == "ran\n"
    assert json.loads((out / "state" / "a.json").read_text())["ok"]


def _write_gguf(path: Path, tensors: list[tuple[str, int, tuple, bytes]], meta: dict) -> None:
    def s(x: str) -> bytes:
        b = x.encode()
        return struct.pack("<Q", len(b)) + b
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(meta))
    for k, v in meta.items():
        head += s(k) + struct.pack("<I", 8) + s(v)
    off, infos, datas = 0, b"", b""
    for name, tid, shape, data in tensors:
        infos += s(name) + struct.pack("<I", len(shape)) + b"".join(struct.pack("<Q", d) for d in shape)
        infos += struct.pack("<IQ", tid, off)
        pad = (-len(data)) % 32
        datas += data + b"\0" * pad
        off += len(data) + pad
    raw = head + infos
    raw += b"\0" * ((-len(raw)) % 32)
    path.write_bytes(raw + datas)


def test_gguf_tensor_identity_ignores_metadata(tmp_path):
    t = [("a.weight", 0, (4, 2), bytes(range(32))), ("b.weight", 8, (32,), bytes(34))]
    _write_gguf(tmp_path / "x.gguf", t, {"note": "written at /path/one"})
    _write_gguf(tmp_path / "y.gguf", t, {"note": "written at /another/path"})
    hx = gguf_hash.tensor_hashes(tmp_path / "x.gguf")
    hy = gguf_hash.tensor_hashes(tmp_path / "y.gguf")
    assert gguf_hash.file_sha256(tmp_path / "x.gguf") != gguf_hash.file_sha256(tmp_path / "y.gguf")
    assert gguf_hash.tensor_digest(hx) == gguf_hash.tensor_digest(hy)
    t2 = [t[0], ("b.weight", 8, (32,), bytes(33) + b"\1")]
    _write_gguf(tmp_path / "z.gguf", t2, {"note": "x"})
    assert gguf_hash.compare(hx, gguf_hash.tensor_hashes(tmp_path / "z.gguf"))["different"] == ["b.weight"]


def test_bundle_build_and_install_relocates_paths(tmp_path):
    src = tmp_path / "rec" / "models" / "code"
    src.mkdir(parents=True)
    (src / "tool.py").write_text('DATA = "/rec/models/gguf/x.gguf"\nOTHER = "/rec/forge/stato/log.tsv"\n'.replace(
        "/rec", str(tmp_path / "rec")))
    (src / "old.py.bak-1").write_text("ignored")
    (src / "__pycache__").mkdir()
    (src / "__pycache__" / "x.pyc").write_bytes(b"\0")
    sources = tmp_path / "sources.yaml"
    sources.write_text(yaml.safe_dump({"trees": [{"from": "{models}/code", "to": "code"}]}))
    rec = {"models": str(tmp_path / "rec" / "models"), "forge": str(tmp_path / "rec" / "forge")}
    meta = bundle.build(sources, tmp_path / "b", rec)
    assert list(meta["files"]) == ["code/tool.py"]
    assert "@@ROOT:models@@/gguf/x.gguf" in (tmp_path / "b" / "code" / "tool.py").read_text()
    assert bundle.verify(tmp_path / "b") == []
    bundle.install(tmp_path / "b", tmp_path / "inst", {"models": "/data/m", "forge": "/data/f"})
    text = (tmp_path / "inst" / "code" / "tool.py").read_text()
    assert '"/data/m/gguf/x.gguf"' in text and '"/data/f/stato/log.tsv"' in text
    (tmp_path / "b" / "code" / "tool.py").write_text("tampered")
    with pytest.raises(ValueError):
        bundle.install(tmp_path / "b", tmp_path / "inst2", {"models": "/m", "forge": "/f"})


def test_bundle_build_flags_secrets(tmp_path):
    src = tmp_path / "t"
    src.mkdir()
    (src / "k.py").write_text('TOKEN = "hf_' + "a" * 34 + '"\n')
    sources = tmp_path / "s.yaml"
    sources.write_text(yaml.safe_dump({"trees": [{"from": str(src), "to": "t"}]}))
    meta = bundle.build(sources, tmp_path / "b", {})
    assert meta["secrets_found"]


def test_multiple_choice_roundtrip(tmp_path):
    tasks = [mc.task(mc.mmlu_prompt("high_school_physics", " Q? ", ["a", "b", "c", "d"]), list("ABCD"), 2),
             mc.task("second", list("ABCD"), 0)]
    mc.write(tasks, tmp_path / "t.bin")
    qs = mc.read_questions(tmp_path / "t.bin")
    assert qs[0].startswith("The following are multiple choice questions (with answers) about high school physics.")
    assert qs[0].endswith("\nAnswer:") and qs[1] == "second"


def test_merge_shards(tmp_path):
    for i, names in ((1, ["x"]), (2, ["y", "z"])):
        d = tmp_path / f"scheggia{i}" / "v" / "tensori"
        d.mkdir(parents=True)
        for n in names:
            (d / f"{n}.npy").write_bytes(n.encode())
        (d / "manifesto.json").write_text(json.dumps({n: {"ok": 1} for n in names}))
    assert merge_shards.merge(tmp_path, "v") == 3
    assert sorted(json.loads((tmp_path / "v" / "tensori" / "manifesto.json").read_text())) == ["x", "y", "z"]
