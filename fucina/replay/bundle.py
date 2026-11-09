"""The code bundle: the frozen research code of a recorded chain, made relocatable.

A recorded chain was produced by research scripts that are not (yet) part of this package. Re-running it exactly
needs those scripts as they were. The bundle is a copy of them, built once on the recording machine
(:func:`build`) and installed on any machine (:func:`install`):

* **build** copies the source trees listed in a sources file (``bundle_sources.yaml``), keeps only code and small
  provenance files, replaces every recorded absolute root (the models volume, the research checkout, ...) by a token
  ``@@ROOT:<name>@@``, scans for secrets, and writes ``BUNDLE.json`` (sha256 of every file, every substitution, every
  absolute path that is left and was not mapped to a root). Nothing is executed.
* **install** copies the bundle into the replay work directory and replaces the tokens with the user's roots, so the
  scripts read and write under the replay tree. It refuses a bundle whose files do not match ``BUNDLE.json``.

The token rewrite changes path strings only. Two GGUF writers of the recorded chain embed the paths of their inputs in
a metadata note; with other roots those notes differ, which is why the replay checks GGUF files by tensor digest
(:mod:`fucina.replay.gguf_hash`) and not by file hash.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Optional

import yaml

TOKEN = "@@ROOT:{}@@"
TOKEN_RE = re.compile(r"@@ROOT:([a-z_]+)@@")
KEEP = ("*.py", "*.sh", "*.cpp", "*.h", "*.txt", "*.json", "*.toml", "*.cfg", "*.md")
SKIP_DIRS = ("__pycache__", ".git", "collaudo_*", "dense_fuori_*", "grafi_hip", "fix_ric", "fix_vivo")
SKIP_FILES = ("*.bak*", "*.orig*", "*.pyc", "*.log", "*.prima_*", "*.tmp")
MAX_BYTES = 2 * 1024 * 1024
SECRETS = [re.compile(p) for p in (r"hf_[A-Za-z0-9]{30,}", r"sk-[A-Za-z0-9_\-]{20,}", r"ghp_[A-Za-z0-9]{30,}",
                                   r"github_pat_[A-Za-z0-9_]{30,}", r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
                                   r"AKIA[0-9A-Z]{16}")]
ABS = re.compile(r"(?<![\w.@])(/(?:home|mnt|opt|srv|data|media|share|run/media)/[\w.+\-/]*)")


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _wanted(rel: Path) -> bool:
    if any(fnmatch.fnmatch(part, pat) for part in rel.parts[:-1] for pat in SKIP_DIRS):
        return False
    name = rel.name
    if any(fnmatch.fnmatch(name, p) for p in SKIP_FILES):
        return False
    return any(fnmatch.fnmatch(name, p) for p in KEEP)


def rewrite(text: str, roots: list[tuple[str, str]]) -> tuple[str, dict]:
    """Replace recorded absolute roots (longest first) by tokens; returns the text and the per-root counts."""
    counts: dict[str, int] = {}
    for prefix, name in sorted(roots, key=lambda x: -len(x[0])):
        pat = re.compile(r"(?<![\w.@])" + re.escape(prefix.rstrip("/")) + r"(?=/|\b|$)")
        text, n = pat.subn(TOKEN.format(name), text)
        if n:
            counts[name] = counts.get(name, 0) + n
    return text, counts


def build(sources_file: str | Path, out: str | Path, recorded_roots: dict[str, str]) -> dict:
    """Build the bundle. ``recorded_roots`` maps a root name to its absolute value on the recording machine."""
    spec = yaml.safe_load(Path(sources_file).read_text())
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty")
    out.mkdir(parents=True, exist_ok=True)
    root_list = [(v, k) for k, v in recorded_roots.items()]
    files, subs, leftover, secrets = {}, {}, {}, []

    def expand(t: str) -> str:
        return re.sub(r"\{(\w+)\}", lambda m: recorded_roots[m.group(1)], t)

    for entry in spec["trees"]:
        src = Path(expand(entry["from"]))
        dst = out / entry["to"]
        if src.is_file():
            pairs = [(src, Path(entry.get("name", src.name)))]
        else:
            pairs = [(p, p.relative_to(src)) for p in sorted(src.rglob("*")) if p.is_file()]
            pairs = [(p, r) for p, r in pairs if _wanted(r) and p.stat().st_size <= MAX_BYTES]
            pairs = [(p, r) for p, r in pairs if not any(fnmatch.fnmatch(r.name, x) for x in spec.get("exclude", []))]
            for pat in entry.get("exclude", []) + spec.get("exclude", []):
                pairs = [(p, r) for p, r in pairs if not fnmatch.fnmatch(str(r), pat)]
        for p, rel in pairs:
            raw = p.read_bytes()
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                continue
            for s in SECRETS:
                if s.search(text):
                    secrets.append(f"{entry['to']}/{rel}: {s.pattern}")
            new, c = rewrite(text, root_list)
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(new, encoding="utf-8")
            if os.access(p, os.X_OK):
                target.chmod(0o755)
            key = str(target.relative_to(out))
            files[key] = {"sha256": _sha(new.encode()), "source": rewrite(str(p), root_list)[0],
                         "source_sha256": _sha(raw)}
            if c:
                subs[key] = c
            rest = sorted(set(ABS.findall(TOKEN_RE.sub("", new))))
            if rest:
                leftover[key] = rest
    meta = {"built": time.strftime("%FT%T"), "sources_file": Path(sources_file).name,
            "recorded_roots": sorted(recorded_roots), "files": files, "substitutions": subs,
            "unmapped_absolute_paths": leftover, "secrets_found": secrets}
    (out / "BUNDLE.json").write_text(json.dumps(meta, indent=1, sort_keys=True))
    return meta


def verify(bundle: str | Path) -> list[str]:
    """Files whose content differs from BUNDLE.json (or are missing)."""
    bundle = Path(bundle)
    meta = json.loads((bundle / "BUNDLE.json").read_text())
    bad = []
    for rel, e in meta["files"].items():
        p = bundle / rel
        if not p.exists() or _sha(p.read_bytes()) != e["sha256"]:
            bad.append(rel)
    return bad


def install(bundle: str | Path, into: str | Path, roots: dict[str, str]) -> dict:
    """Copy the bundle into ``into`` with every ``@@ROOT:name@@`` token replaced by ``roots[name]``."""
    bundle, into = Path(bundle), Path(into)
    bad = verify(bundle)
    if bad:
        raise ValueError(f"bundle files differ from BUNDLE.json: {bad[:10]}")
    meta = json.loads((bundle / "BUNDLE.json").read_text())
    needed = set()
    for rel in meta["files"]:
        needed |= set(TOKEN_RE.findall((bundle / rel).read_text(encoding="utf-8")))
    missing = sorted(needed - set(roots))
    if missing:
        raise KeyError(f"roots needed by the bundle and not given: {missing}")
    if into.exists():
        shutil.rmtree(into)
    for rel in meta["files"]:
        src = bundle / rel
        dst = into / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(TOKEN_RE.sub(lambda m: roots[m.group(1)].rstrip("/"), src.read_text(encoding="utf-8")),
                       encoding="utf-8")
        shutil.copymode(src, dst)
    receipt = {"installed": time.strftime("%FT%T"), "bundle": str(bundle), "roots": {k: roots[k] for k in needed},
               "files": len(meta["files"])}
    (into / "INSTALLED.json").write_text(json.dumps(receipt, indent=1))
    return receipt


def report(meta: dict) -> str:
    lines = [f"{len(meta['files'])} files, {sum(sum(c.values()) for c in meta['substitutions'].values())} root "
             f"substitutions in {len(meta['substitutions'])} files"]
    if meta["secrets_found"]:
        lines.append("SECRETS FOUND (the bundle must not be published):")
        lines += [f"  {s}" for s in meta["secrets_found"]]
    if meta["unmapped_absolute_paths"]:
        lines.append(f"absolute paths not under a recorded root ({len(meta['unmapped_absolute_paths'])} files):")
        for f, ps in sorted(meta["unmapped_absolute_paths"].items()):
            lines.append(f"  {f}: {', '.join(ps[:4])}{' ...' if len(ps) > 4 else ''}")
    return "\n".join(lines)


def main(argv: Optional[list] = None) -> int:
    import argparse
    p = argparse.ArgumentParser(prog="fucina replay-bundle", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build the bundle from the recorded source trees (recording machine)")
    b.add_argument("sources")
    b.add_argument("--out", required=True)
    b.add_argument("--root", action="append", default=[], metavar="NAME=PATH",
                   help="recorded root, e.g. models=/data/models (repeat)")
    i = sub.add_parser("install", help="install a bundle with your roots")
    i.add_argument("bundle")
    i.add_argument("--into", required=True)
    i.add_argument("--root", action="append", default=[], metavar="NAME=PATH")
    v = sub.add_parser("verify", help="check a bundle against its BUNDLE.json")
    v.add_argument("bundle")
    a = p.parse_args(argv)
    if a.cmd == "verify":
        bad = verify(a.bundle)
        print("bundle OK" if not bad else f"{len(bad)} files differ: {bad[:20]}")
        return 1 if bad else 0
    roots = dict(x.split("=", 1) for x in a.root)
    if a.cmd == "build":
        meta = build(a.sources, a.out, roots)
        print(report(meta))
        return 2 if meta["secrets_found"] else 0
    print(json.dumps(install(a.bundle, a.into, roots), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
