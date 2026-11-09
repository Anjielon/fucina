"""Machine configuration: where tools, engines, teacher caches and outputs live.

Nothing machine-specific is hard-coded in the package. Every path the technique wrappers and the engine need comes
from a YAML file, looked up in this order:

1. the path given to :func:`activate` (the CLI ``--config`` option);
2. the ``FUCINA_CONFIG`` environment variable;
3. ``./fucina.yaml`` in the current directory;
4. built-in defaults (no tool paths: wrappers that need one raise :class:`ConfigError`).

See ``config.example.yaml`` for the documented keys and ``examples/thor/config.yaml`` for a complete, real setup.
Environment variables (``$VAR`` or ``${VAR}``; use ``$VAR`` inside YAML flow mappings) are expanded in every string value, so the example setup names
its machine paths through ``THOR_LAB``, ``THOR_GGUF``, ``THOR_BUILD``, ``THOR_HF``, ``THOR_POD``, ``THOR_DEPOT`` and ``THOR_PYTHON``.
Relative paths in the top-level file-system keys (``runs_dir``, ``database``, ``outputs``, ``tmpdir``,
``models_dirs``, ``rules_dirs``, ``cycle_log``, ``lock_script``, ``lock_owner_file``) are resolved against the
directory of the configuration file.
"""
from __future__ import annotations

import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

import yaml

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_RULES_DIR = PACKAGE_DIR / "rules"

_FS_KEYS = ("runs_dir", "database", "outputs", "tmpdir", "cycle_log", "lock_script", "lock_owner_file")
_FS_LIST_KEYS = ("models_dirs", "rules_dirs")

DEFAULTS: dict[str, Any] = {
    "python": sys.executable,                 # interpreter used inside job scripts
    "runs_dir": "runs",                       # one sub-directory per run (state, queue, questions, recipe)
    "database": "measurements.sqlite",        # measurement log
    "outputs": "forge-outputs",               # where real/dry-run jobs write GGUF files and measurements
    "tmpdir": None,                           # TMPDIR exported to jobs (None: inherit)
    "models_dirs": ["models"],                # where `--model NAME` looks for NAME.yaml
    "rules_dirs": [],                         # extra rule files, on top of the packaged rule base
    "default_model": None,
    "default_objective": "A",
    "cycle_log": None,                        # optional TSV activity log (date<TAB>STATUS<TAB>text)
    "cycle_log_status": "ACTIVE",
    "lock_script": None,                      # shell file defining the GPU lock functions below
    "lock_owner_file": None,                  # file naming the current holder of the GPU lock (read-only, for the UI)
    "lock_commands": {"acquire": "acquire_gpu", "release": "release_gpu", "ram_guard": "ram_guard"},
    "job_env": {"OMP_NUM_THREADS": "8", "PYTHONDONTWRITEBYTECODE": "1"},
    "measure_architectures": ["qwen35moe"],   # architectures for which the measurement script is validated
    "paths": {},                              # tool and asset paths used by the technique wrappers
    "techniques": {},                         # per-technique overrides: {name: {params: {...}, known_products: [...]}}
    "import_sources": {},                     # lab-notebook files read by `fucina import`
}


class ConfigError(KeyError):
    """A path or setting needed by a wrapper is missing from the configuration."""

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else "configuration error"


class Paths(dict):
    """Tool/asset paths. A missing key raises :class:`ConfigError` with the key to add."""

    def __missing__(self, key: str) -> str:
        raise ConfigError(f"path '{key}' is not configured: add `paths: {{{key}: ...}}` to your fucina config")

    def __getitem__(self, key: str) -> str:
        v = super().__getitem__(key)
        if isinstance(v, str) and "$" in v:
            raise ConfigError(f"path '{key}' = {v}: an environment variable it names is not set")
        return v


class Config:
    def __init__(self, data: Optional[dict] = None, source: Optional[str] = None):
        d = deepcopy(DEFAULTS)
        d.update(_expand(data or {}))   # a key given in the file replaces the default as a whole (order matters for job_env)
        base = Path(source).resolve().parent if source else Path.cwd()
        for k in _FS_KEYS:
            if d.get(k):
                d[k] = _resolve(base, d[k])
        for k in _FS_LIST_KEYS:
            d[k] = [_resolve(base, p) for p in (d.get(k) or [])]
        self.data = d
        self.source = source
        self.paths = Paths(d.get("paths") or {})

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    # ── conveniences used by the wrappers ──
    def path(self, key: str) -> str:
        return self.paths[key]

    def resolve(self, key_or_path: str) -> str:
        """A key of ``paths`` or a literal path (old ``AMB.get(x, x)`` idiom)."""
        return self.paths.get(key_or_path, key_or_path)

    def technique(self, name: str) -> dict:
        return (self.data.get("techniques") or {}).get(name) or {}

    def rules_dirs(self) -> list:
        return [str(DEFAULT_RULES_DIR)] + list(self.data.get("rules_dirs") or [])

    def lock(self, which: str) -> str:
        return (self.data.get("lock_commands") or {}).get(which) or DEFAULTS["lock_commands"][which]

    def lock_source(self) -> str:
        s = self.data.get("lock_script")
        if not s:
            raise ConfigError("lock_script is not configured: GPU jobs need a shell file defining "
                              f"{', '.join(DEFAULTS['lock_commands'].values())}")
        return s


def _expand(v: Any) -> Any:
    """Expand ``$VAR``/``${VAR}`` in every string of the file, so a shared config names no machine path."""
    if isinstance(v, str):
        return os.path.expandvars(v)
    if isinstance(v, dict):
        return {k: _expand(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_expand(x) for x in v]
    return v


def _resolve(base: Path, p: str) -> str:
    if "$" in str(p):
        raise ConfigError(f"{p}: an environment variable it names is not set")
    q = Path(os.path.expanduser(str(p)))
    return os.path.normpath(str(q if q.is_absolute() else (base / q)))


def load(path: Optional[str] = None) -> Config:
    p = path or os.getenv("FUCINA_CONFIG") or ("fucina.yaml" if Path("fucina.yaml").exists() else None)
    if not p:
        return Config()
    f = Path(p)
    if not f.exists():
        raise FileNotFoundError(f"fucina config not found: {f}")
    return Config(yaml.safe_load(f.read_text()) or {}, str(f.resolve()))


_current: Optional[Config] = None


def activate(path: Optional[str] = None) -> Config:
    """Load a configuration and make it the current one for the whole package."""
    global _current
    _current = load(path)
    return _current


def is_active() -> bool:
    """True once a configuration has been loaded (explicitly or lazily)."""
    return _current is not None


def current() -> Config:
    global _current
    if _current is None:
        _current = load()
    return _current
