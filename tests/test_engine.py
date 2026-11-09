"""Tests of the decision engine (verdicts, objective, law, questions, switches) and of the ``--simulate`` acceptance run
on Thor-T (no GPU, no process; the database is rebuilt from the real sources into a temporary file)."""
import asyncio
import json
import subprocess
from pathlib import Path

import pytest

from fucina import config
from fucina import decision as D
from fucina.measure import Measurement
from fucina.registry import Thresholds

OB_A = D.parse_objective("mmlu>=80.05,file<=13085779616,tg>=50")
OB_NO_TG = D.parse_objective("file<=13085779616")


def _paths():
    return config.current().paths


def _sources_present() -> bool:
    src = config.current()["import_sources"]
    return all(Path(p).exists() for p in (src["cycles"], src["results"])) and Path(_paths()["gguf_r2b3m"]).exists()


# ── objective and law ──
def test_objective():
    o = D.parse_objective("mmlu>=80.05,ram<=9GiB,tg>=50")
    assert o.mmlu_min == 80.05 and o.tg_min == 50 and o.cap_bytes == D.DEFAULT_FILE_CAP_BYTES
    assert D.parse_objective("file<=13085779616").cap_bytes == 13085779616
    a = D.parse_objective("A", {"A": "mmlu>=80.05,file<=13085779616,tg>=50"})
    assert a.mmlu_min == 80.05 and a.text.startswith("A:")
    with pytest.raises(ValueError):
        D.parse_objective("speed>=3")


def test_kl_mmlu_law():
    law = D.Law()
    assert abs(law.mmlu(-43.3) - 78.09) < 0.02            # pv12: 78.50 measured
    assert abs(law.drop_for(80.05) + 54.1) < 0.2           # plan: 80.05 needs ~-54 %
    assert law.mmlu(+10) == law.base                       # a worsening does not "gain"
    assert abs(D.compose_drops(-43.3, -11.7) - (-49.93)) < 0.01
    assert D.needs_mmlu(-30) and not D.needs_mmlu(-3) and not D.needs_mmlu(None)


# ── threshold verdicts ──
def m(**k):
    return Measurement(**k).complete()


def test_clear_kl_pass():
    v = D.judge(m(kl=0.39, kl_base=0.41539, kl_se=0.002), Thresholds(), OB_NO_TG)
    assert v.verdict == "PASS" and v.advice == "enable"


def test_worse_and_flat_kl_fail():
    assert D.judge(m(kl=0.42, kl_base=0.41539, kl_se=0.001), Thresholds(), OB_NO_TG).verdict == "FAIL"
    v = D.judge(m(kl=0.4155, kl_base=0.41539, kl_se=0.002), Thresholds(), OB_NO_TG)
    assert v.verdict == "FAIL" and "no measurable effect" in v.reason


def test_weak_kl_is_ambiguous():
    v = D.judge(m(kl=0.4120, kl_base=0.41539, kl_se=0.002), Thresholds(), OB_NO_TG)    # -1.7 se
    assert v.verdict == "AMBIGUOUS" and v.advice == "remeasure"


def test_significant_negative_mmlu_beats_kl():
    v = D.judge(m(kl=0.39, kl_base=0.41539, kl_se=0.002, mmlu=72.75, mmlu_base=74.6, p_mcnemar=0.005), Thresholds(),
                OB_NO_TG)
    assert v.verdict == "FAIL"


def test_published_lever_to_verify():
    v = D.judge(m(kl_pct=0.37, kl_base=0.41539, kl_se=0.00073), Thresholds(), OB_NO_TG, source="arXiv 2501.00001")
    assert v.verdict == "FAIL" and v.to_verify and "to verify" in v.reason
    assert not D.judge(m(kl_pct=0.37, kl_base=0.41539, kl_se=0.00073), Thresholds(), OB_NO_TG,
                       source="in-house lever").to_verify


def test_min_tg_makes_it_ambiguous():
    v = D.judge(m(kl=0.37, kl_base=0.41539, kl_se=0.002, tg=45.0, tg_base=67.9), Thresholds(), OB_A)
    assert v.verdict == "AMBIGUOUS" and "slows down" in v.reason
    assert D.judge(m(kl=0.37, kl_base=0.41539, kl_se=0.002, tg=53.5, tg_base=67.9), Thresholds(), OB_A).verdict == "PASS"


def test_size_gate_and_tg_collapse():
    assert D.judge(m(kl=0.3, kl_base=0.41539, kl_se=0.002, file_bytes=14_000_000_000), Thresholds(), OB_A).verdict == "FAIL"
    collapse = m(kl=0.37, kl_base=0.41539, kl_se=0.002, tg=31.6, tg_base=67.5)
    assert D.judge(collapse, Thresholds(), OB_A).verdict == "FAIL"
    assert D.judge(collapse, Thresholds(), OB_A, slows_down=True).verdict == "AMBIGUOUS"     # slow lever: remeasure


def test_speed_judgement_int8():
    v = D.judge(m(tg=56.33, tg_base=50.4, mmlu=74.65, mmlu_base=74.60, p_mcnemar=1.0), Thresholds(), OB_A, "speed")
    assert v.verdict == "PASS"
    assert D.judge(m(tg=49.0, tg_base=50.4), Thresholds(), OB_A, "speed").verdict == "FAIL"


def test_significant_positive_mmlu_passes_without_kl():
    v = D.judge(m(mmlu=78.80, mmlu_base=74.60, tg=53.5), Thresholds(), OB_A)
    assert v.verdict == "PASS"


# ── questions and switches on disk ──
def test_question_queue(tmp_path):
    from fucina.engine import QuestionQueue, Switches
    q = QuestionQueue(tmp_path)
    assert q.ask("x@R2B3m", text="?", advice="remeasure") is None
    assert q.ask("x@R2B3m", text="?") is None and len(q.items) == 1      # idempotent
    q.answer("x@R2B3m", "enable")
    assert QuestionQueue(tmp_path).find("x@R2B3m")["answer"] == "enable" and not q.open()
    with pytest.raises(ValueError):
        q.answer("x@R2B3m", "maybe")
    s = Switches(tmp_path)
    s.set("prism", "disable")
    assert Switches(tmp_path).get("prism") == "disable"


# ── oracle ──
def test_oracle_referred_and_nearby(tmp_path):
    from fucina.db import Measurements
    from fucina.oracle import Oracle, Signature
    from fucina.registry import load_techniques
    db = Measurements(str(tmp_path / "m.sqlite"))
    db.add(model="Thor-T", technique="bil", kl=0.40, kl_base=0.41539, kl_pct=-3.7, file_bytes=8_900_000_000,
           params=json.dumps({"base": "R2B3m", "kl_se": 0.001}), source="fucina:test")
    db.add(model="Thor-T", technique="bil", mmlu=78.1, mmlu_base=78.50, file_bytes=13_000_000_000, source="fucina:test")
    db.add(model="Other", technique="bil", kl_pct=-50, params=json.dumps({"base": "R2B3m"}), source="fucina:test")
    o = Oracle(db, "Thor-T", load_techniques())
    r2 = Signature("R2B3m", mmlu=[74.60, 74.65], kl=0.41539)
    pv = Signature("pv12", mmlu=[78.50])
    a = o.referred("bil", r2, 13_085_779_616)
    assert a is not None and abs(a.kl_pct + 3.7) < 1e-9 and a.z_kl is not None
    b = o.referred("bil", pv, 13_085_779_616)
    assert b is not None and b.mmlu == 78.1
    assert o.referred("bil", pv, 9_000_000_000) is None                # above the cap it does not count
    vm, base = o.nearby("bil", Signature("PV12P", mmlu=[78.80]), 13_085_779_616, (r2, pv))
    assert base in ("R2B3m", "pv12")


# ── acceptance: Thor-T from bf16 with the pre-loaded database ──
@pytest.fixture
def thor_db(tmp_path_factory):
    if not _sources_present():
        pytest.skip("Thor-T sources are not present on this machine")
    from fucina.db import Measurements
    from fucina.importers.cycles import import_cycles
    from fucina.importers.thor_notes import import_all
    p = str(tmp_path_factory.mktemp("db") / "measurements.sqlite")
    mm = Measurements(p)
    import_cycles(mm)
    import_all(mm)
    return p


@pytest.fixture
def no_processes(monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError(f"--simulate must not launch processes: {a[:1]}")
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)


def _states(r):
    return {v["technique"]: v["state"] for v in r["evaluations"]}


def test_simulate_thor_objective_A(tmp_path, thor_db, no_processes):
    from fucina.engine import Engine
    e = Engine("thor", "A", str(tmp_path / "run"), "simulate", db=thor_db)
    r = e.run()
    assert e.launch() == []
    st = _states(r)
    # known recipe: bf16 -> R2B3m (three planes) -> PV12P (second plane from the dKL)
    assert [s["label"] for s in r["history"]] == ["bf16", "R2B3m", "PV12P"]
    assert r["candidate"]["gguf"] == _paths()["gguf_pv12p"]
    assert st["measured_dkl"] == "consolidated" and st["three_planes"] == "consolidated"
    assert st["second_plane_dkl"] == "passed"
    # today's queue: per-byte levers being tested -> combination (placeholder) -> RILQ -> final evaluation
    for t in ("bil", "jarq", "tied", "cardan", "prism", "k_scale"):
        assert st[t] == "testing", t
    assert st["combination"] == "waiting" and st["rilq"] == "waiting" and st["final_eval"] == "waiting"
    assert st["qtea"] == "off" and st["e2_rank1"] == "off" and st["shq8"] == "excluded"
    assert st["schur"] == "suspended" and st["g2ptq"] == "suspended" and st["int8"] == "consolidated"
    assert r["engine_env"] == {"GGML_VK_TQ1R_Q8": "1"}
    queue = {j["id"]: j for j in r["queue"]}
    assert all(j["state"] == "simulated" for j in queue.values())        # nothing launched
    combo = next(j for j in queue.values() if j["id"].startswith("COMBO_"))
    rilq = next(j for j in queue.values() if j["technique"] == "rilq")
    assert combo["placeholder"] and rilq["placeholder"] and rilq["training"]
    assert any(d.startswith("M_COMBO_") for d in rilq["depends_on"])     # RILQ on top of the combination
    assert any(d.startswith("M_F1_bil") for d in combo["depends_on"])
    # forecast: just under 80.05
    p = r["forecast"]
    assert p["reaches"] is False and 79.0 < p["mmlu_expected"] < 80.05
    assert (tmp_path / "run" / "recipe.yaml").exists() and (tmp_path / "run" / "state.json").exists()


def test_simulate_resume_is_idempotent(tmp_path, thor_db, no_processes):
    from fucina.candidate import RunState
    from fucina.engine import Engine
    c = str(tmp_path / "run")
    r1 = Engine("thor", "A", c, "simulate", db=thor_db).run()
    r2 = Engine("thor", "A", c, "simulate", db=thor_db).run()
    assert [j["id"] for j in r1["queue"]] == [j["id"] for j in r2["queue"]]
    assert RunState(c).load().label == "PV12P"


def test_simulate_thor_objective_B_question(tmp_path, thor_db, no_processes):
    from fucina.engine import Engine, QuestionQueue
    c = tmp_path / "run"
    r = Engine("thor", "B", str(c), "simulate", db=thor_db).run()
    st = _states(r)
    assert st["second_plane_dkl"] == "to_decide"
    q = r["open_questions"]
    assert q and q[0]["technique"] == "second_plane_dkl" and q[0]["advice"] == "remeasure"
    assert [s["label"] for s in r["history"]] == ["bf16", "R2B3m"]
    QuestionQueue(c).answer(q[0]["id"], "enable")
    r2 = Engine("thor", "B", str(c), "simulate", db=thor_db).run()
    assert _states(r2)["second_plane_dkl"] == "passed" and not r2["open_questions"]
    assert r2["history"][-1]["phase"] == "F2"


def test_user_switch(tmp_path, thor_db, no_processes):
    from fucina.engine import Engine, Switches
    c = tmp_path / "run"
    c.mkdir()
    Switches(c).set("prism", "disable")
    r = Engine("thor", "A", str(c), "simulate", db=thor_db).run()
    assert _states(r)["prism"] == "off"
    combo = next(v for v in r["evaluations"] if v["technique"] == "combination")
    assert "prism" not in combo["numbers"]["levers"]


def test_job_outcome_enters_the_database(tmp_path, thor_db, no_processes):
    """A finished measurement job (faked here) is read, written to the database and judged on the next pass."""
    from fucina.db import Measurements
    from fucina.engine import Engine
    from fucina.executor import Executor
    c = tmp_path / "run"
    Engine("thor", "A", str(c), "simulate", db=thor_db).run()
    ex = Executor(str(c), "simulate")
    j = next(x for x in ex.jobs if x.id.startswith("M_F1_bil_"))
    ref = json.loads(Path(_paths()["kl_ref"]).read_text())
    better = {**ref, "kl_media": ref["kl_media"] * 0.95,
              "per_riga": {k: {**v, "kl_media": v["kl_media"] * 0.95} for k, v in ref["per_riga"].items()}}
    (Path(j.output_dir) / "kl").mkdir(parents=True)
    (Path(j.output_dir) / "kl" / "kl_bil.json").write_text(json.dumps(better))
    j.state = "done"
    ex.save()
    r = Engine("thor", "A", str(c), "simulate", db=thor_db).run()
    v = next(v for v in r["evaluations"] if v["technique"] == "bil")
    assert v["state"] == "passed" and abs(v["numbers"]["kl_pct"] + 5.0) < 0.01
    assert any(x["source"].startswith("fucina:run:") and x["technique"] == "bil" for x in Measurements(thor_db).all())
    # the alternatives of the same group still being tested stay; the combination is still a placeholder
    assert _states(r)["combination"] == "waiting"


def test_cli_simulate_and_decide(tmp_path, thor_db, no_processes, capsys):
    from fucina.cli import main
    c = tmp_path / "runB"
    main(["--db", thor_db, "auto", "--objective", "B", "--simulate", "--run-dir", str(c)])
    out = capsys.readouterr().out
    assert "R2B3m" in out and "questions for you" in out
    main(["decide", "--run-dir", str(c), "--list"])
    assert "second_plane_dkl@R2B3m" in capsys.readouterr().out
    main(["decide", "--run-dir", str(c), "second_plane_dkl@R2B3m", "disable"])
    main(["--db", thor_db, "status", "--run-dir", str(c)])
    main(["--db", thor_db, "manual", "--piece", "experts_gate_up", "--parallel", "prism,cardan",
          "--run-dir", str(tmp_path / "man")])
    out = capsys.readouterr().out
    assert "prism" in out and "cardan" in out


def test_cli_dry_run_writes_the_queue(tmp_path, thor_db, no_processes, capsys):
    """--dry-run writes every launchable job, in dependency order, without starting anything."""
    from fucina.cli import main
    c = tmp_path / "dry"
    main(["--db", thor_db, "auto", "--dry-run", "--run-dir", str(c)])
    text = (c / "dry_run_commands.sh").read_text()
    ids = [line[2:].split(":")[0] for line in text.splitlines() if line.startswith("# ")]
    assert ids[0].startswith("F1_bil_") and ids[-1].startswith("M_ref_PV12P_") and len(ids) >= 11  # depends on which wrapped tools exist on disk
    assert "prendi_gpu" in text and "FUCINA-BIL" in text and "C_RIF_NOME=R2B3m" in text
    assert "dry_run_commands.sh" in capsys.readouterr().out


def test_tui_pilot_headless(tmp_path, thor_db):
    """The textual UI opens, shows the tree, switches techniques off and answers questions (headless pilot)."""
    pytest.importorskip("textual")
    from fucina.engine import Engine, QuestionQueue, Switches
    from fucina.tui import build_app
    c = tmp_path / "run"
    Engine("thor", "B", str(c), "simulate", db=thor_db).run()

    async def session():
        app = build_app()(c)
        async with app.run_test(size=(160, 50)) as pilot:
            tr = app.query_one("#tree")
            nodes = [n for f in tr.root.children for n in f.children]
            assert any(n.data["technique"] == "second_plane_dkl" for n in nodes)
            sp = next(n for n in nodes if n.data["technique"] == "prism")
            tr.move_cursor(sp)
            await pilot.pause()
            await pilot.press("d")
            await pilot.pause()
            assert Switches(c).get("prism") == "disable"
            nodes = [n for f in tr.root.children for n in f.children]
            d = next(n for n in nodes if n.data["technique"] == "second_plane_dkl")
            tr.move_cursor(d)
            await pilot.pause()
            await pilot.press("1")
            await pilot.pause()
        return app
    asyncio.run(session())
    assert QuestionQueue(c).find("second_plane_dkl@R2B3m")["answer"] == "enable"
    r = json.loads((c / "report.json").read_text())
    assert {v["technique"]: v["state"] for v in r["evaluations"]}["prism"] == "off"


def test_ablations_switch_off_the_lever_that_brings_nothing(tmp_path, thor_db, no_processes):
    """Measured combination + all-but-one: the lever without which the KL does not worsen is switched off."""
    from fucina.engine import Engine, Evaluation
    from fucina.executor import Job
    e = Engine("thor", "A", str(tmp_path / "run"), "simulate", db=thor_db)
    e.run()
    ref = _paths()["kl_ref"]
    b = json.loads(Path(ref).read_text())
    out = str(tmp_path / "combo")

    def write(name, f):
        d = Path(out) / f"measure_without_{name}" / "kl"
        d.mkdir(parents=True)
        (d / f"kl_COMBO-{name}.json").write_text(json.dumps(
            {**b, "kl_media": b["kl_media"] * f, "per_riga": {k: {**v, "kl_media": v["kl_media"] * f}
                                                              for k, v in b["per_riga"].items()}}))
        e.executor.jobs.append(Job(id=f"M_COMBO_x_without_{name}", name="", command=[], state="done",
                                   output_dir=f"{out}/measure_without_{name}"))
    write("prism", 0.899)       # without prism the KL IMPROVES: prism is harmful
    write("cardan", 0.95)       # without cardan it worsens by 5.6 %: kept
    t = e.techniques
    v = Evaluation("combination", "COMBO", [], "passed", reason="ok", jobs=["COMBO_x"])
    e._ablations(v, [(t["prism"], None), (t["cardan"], None)], Measurement(kl=b["kl_media"] * 0.9), out, ref,
                 e.signature(e.cand))
    assert v.numbers["kl_contribution_pct"]["prism"] < 0 < v.numbers["kl_contribution_pct"]["cardan"]
    assert v.numbers["env_disable"] == {"THOR_SPEGNI": "prism"} and "prism switched off" in v.reason


def test_dense_model_without_wrappers_queues_nothing(tmp_path, thor_db, no_processes):
    """Genericity: a dense model of another architecture gets no Thor command (techniques excluded with a reason)."""
    from fucina.engine import Engine
    r = Engine("qwen3-0.6b", "A", str(tmp_path / "run"), "simulate", db=thor_db).run()
    st = _states(r)
    assert set(st.values()) == {"excluded"} and r["queue"] == []
    assert [s["label"] for s in r["history"]] == ["bf16"]
    assert r["forecast"]["mmlu_expected"] is None


def test_generic_config_without_paths_excludes_nothing_silently(tmp_path, monkeypatch):
    """With the built-in defaults (no tool paths) the techniques report what is missing instead of crashing."""
    from fucina.registry import load_techniques
    monkeypatch.setattr(config, "_current", config.Config())
    miss = {t.name: t.missing() for t in load_techniques().values() if getattr(t, "tools", ())}
    assert miss["bil"] == ["<paths.levers_g4 not configured>"]
    assert all(m for m in miss.values())
