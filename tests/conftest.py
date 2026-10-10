"""Shared fixtures: every test runs against the Thor-T example configuration (examples/thor/config.yaml).

The Thor-T integration tests read the private lab notebook of the reference machine and are skipped elsewhere. The
activity log keeps growing, so it is frozen at SNAPSHOT (the moment the expected recipe below was measured): the
assertions then test the engine, not the latest experiments."""
from pathlib import Path

import pytest
import yaml

from fucina import config

ROOT = Path(__file__).resolve().parents[1]
THOR_CONFIG = ROOT / "examples" / "thor" / "config.yaml"
SNAPSHOT = "2026-10-08T11:00"


@pytest.fixture(scope="session", autouse=True)
def thor_config(tmp_path_factory):
    data = yaml.safe_load(THOR_CONFIG.read_text())
    live = Path(data["import_sources"]["cycles"])
    if live.exists():
        frozen = tmp_path_factory.mktemp("notebook") / "cycles_frozen.tsv"
        frozen.write_text("".join(line for line in live.open(encoding="utf-8", errors="replace")
                                  if line.split("\t", 1)[0] < SNAPSHOT), encoding="utf-8")
        data["import_sources"]["cycles"] = str(frozen)
    config._current = config.Config(data, str(THOR_CONFIG))
    yield config._current


@pytest.fixture(autouse=True)
def no_cycle_log(monkeypatch):
    monkeypatch.setenv("FUCINA_NO_CYCLE_LOG", "1")
