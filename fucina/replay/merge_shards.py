"""Merge the per-shard outputs of the dense PTQ step into one tensor directory with one manifest.

Each shard writes ``<variants>/scheggiaN/<variant>/tensori/{<tensor>.npy, manifesto.json}``. The merge copies every
tensor into ``<variants>/<variant>/tensori`` and writes the union of the shard manifests (the recorded run did the
same with an inline script). Usage: python -m fucina.replay.merge_shards VARIANTS_DIR VARIANT
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


def merge(variants: Path, variant: str) -> int:
    out = variants / variant / "tensori"
    out.mkdir(parents=True, exist_ok=True)
    merged = {}
    for shard in sorted(variants.glob("scheggia*")):
        man = shard / variant / "tensori" / "manifesto.json"
        if not man.exists():
            continue
        for name, entry in json.loads(man.read_text()).items():
            shutil.copy2(shard / variant / "tensori" / f"{name}.npy", out / f"{name}.npy")
            merged[name] = entry
    (out / "manifesto.json").write_text(json.dumps(merged))
    return len(merged)


if __name__ == "__main__":
    print(sys.argv[2], merge(Path(sys.argv[1]), sys.argv[2]))
