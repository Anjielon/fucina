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


def fucina_commit() -> str:
    here = Path(__file__).resolve().parent
    r = subprocess.run(["git", "-C", str(here), "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True)
    if r.returncode or not r.stdout.strip():
        raise SystemExit("REPLAY_STATUS needs the fucina commit: run from a git clone of the forge")
    dirty = subprocess.run(["git", "-C", str(here), "status", "--porcelain", "--", str(here.parents[1])],
                           capture_output=True, text=True).stdout.strip()
    if dirty:
        raise SystemExit("the fucina checkout has local changes: a verified replay must name a clean commit")
    return r.stdout.strip()


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
    a = ap.parse_args(argv)
    ok, why = verdict(a.gguf, a.expect_digest, a.mcnemar_json, a.recorded_mmlu, a.band)
    commit = fucina_commit()
    day = time.strftime("%Y-%m-%d")
    line = f"REPLAY VERIFIED {day} {commit}" if ok else f"REPLAY NOT VERIFIED {day} {commit}: {why}"
    a.out.write_text(line + "\n" + ("" if not ok else f"# {why}\n"))
    print(line, "|", why)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
