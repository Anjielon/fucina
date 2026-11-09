"""Write the replay verdict file ``REPLAY_STATUS``.

The paper's switch that says "the chain was replayed from the released artifacts" reads this file and accepts exactly
one form of line::

    REPLAY VERIFIED <YYYY-MM-DD> <fucina commit>

It is written only when the replayed C passes the target of the recipe:

* exact: the tensor digest of the replayed C equals the recorded one; or
* within noise: MMLU-2000 of the replayed C within ``--band`` points of the recorded C and an exact McNemar test
  against the recorded per-question log with p >= 0.05.

Otherwise the file says ``REPLAY NOT VERIFIED <date> <commit>: <reason>``, which the switch rejects.

    python -m fucina.replay.status --gguf C.gguf --expect-digest DIGEST --mcnemar-json MCNEMAR.json \\
        --recorded-mmlu 73.15 --out REPLAY_STATUS
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from . import gguf_hash


COMMIT_FILE = Path(__file__).resolve().parent / "FORGE_COMMIT"


def _git(*args: str) -> str:
    r = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), *args], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ""


def forge_commit(release: str | None = None, strict: bool = True) -> tuple[str, str]:
    """(commit, how). In a release tarball ``git archive`` has written the commit into FORGE_COMMIT (export-subst);
    in a git clone the commit is HEAD, which must be the commit of the recipe's release tag and have no local changes.
    With ``strict`` false nothing is raised: an empty commit and the reason are returned (used by the dry run)."""
    def fail(why: str) -> tuple[str, str]:
        if strict:
            raise SystemExit(why)
        return "", why
    text = COMMIT_FILE.read_text().strip() if COMMIT_FILE.exists() else ""
    if text and not text.startswith("$Format"):
        if not all(c in "0123456789abcdef" for c in text) or len(text) < 7:
            return fail(f"FORGE_COMMIT holds {text!r}, not a commit hash")
        return text[:12], "release tarball (FORGE_COMMIT written by git archive)"
    head = _git("rev-parse", "HEAD")
    if not head:
        return fail("cannot name the forge commit: not a git clone and FORGE_COMMIT was not filled by git archive")
    if release:
        tagged = _git("rev-parse", f"{release}^{{commit}}")
        if not tagged:
            return fail(f"the release tag {release} of the recipe does not exist in this clone")
        if tagged != head:
            return fail(f"HEAD {head[:12]} is not the recipe's release {release} ({tagged[:12]}): check out the tag")
    if _git("status", "--porcelain", "--", str(Path(__file__).resolve().parents[2])):
        return fail("the forge checkout has local changes: a verified replay must name a clean commit")
    return head[:12], f"git clone at {release or 'HEAD'}"


def verdict(gguf: Path, expect: str, mcnemar: Path | None, recorded: float, band: float) -> tuple[bool, str]:
    digest = gguf_hash.tensor_digest(gguf_hash.tensor_hashes(gguf))
    if digest == expect:
        return True, f"tensor digest {digest} equals the record"
    if mcnemar is None or not mcnemar.exists():
        return False, f"tensor digest {digest[:16]} differs and no McNemar result is available"
    m = json.loads(mcnemar.read_text())
    acc, p, n = float(m["acc_B"]), float(m["p_mcnemar_esatto"]), int(m.get("n", 0))
    if n != 2000:
        return False, f"McNemar on {n} questions instead of 2000"
    if abs(acc - recorded) > band:
        return False, f"MMLU-2000 {acc:.2f} outside {recorded} +/- {band}"
    if p < 0.05:
        return False, f"McNemar p {p:.4f} < 0.05 against the recorded C"
    return True, f"MMLU-2000 {acc:.2f} (recorded {recorded}), McNemar p {p:.4f}, tensor digest {digest[:16]}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gguf", type=Path, required=True)
    ap.add_argument("--expect-digest", required=True)
    ap.add_argument("--mcnemar-json", type=Path)
    ap.add_argument("--recorded-mmlu", type=float, required=True)
    ap.add_argument("--band", type=float, default=1.0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--forge-release", help="release tag of the forge named by the recipe")
    a = ap.parse_args(argv)
    ok, why = verdict(a.gguf, a.expect_digest, a.mcnemar_json, a.recorded_mmlu, a.band)
    commit, _ = forge_commit(a.forge_release)
    day = time.strftime("%Y-%m-%d")
    line = f"REPLAY VERIFIED {day} {commit}" if ok else f"REPLAY NOT VERIFIED {day} {commit}: {why}"
    a.out.write_text(line + "\n" + ("" if not ok else f"# {why}\n"))
    print(line, "|", why)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
