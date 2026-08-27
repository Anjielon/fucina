#!/usr/bin/env python3
"""GGUF SURGEON — rewrite one tensor's bytes IN PLACE, without rewriting the file.

A GGUF stores each tensor at a fixed offset with a fixed size, so a tensor whose
byte count does not change can be replaced where it lies. This matters at scale:
repairing a 30 MB router inside a 93 GiB model should not cost a 93 GiB rewrite.

What it is for: routers (F32), norms (F32) and per-block scales — anything you
want to patch, measure, and keep or revert. What it is NOT for: changing a
tensor's precision, which changes its size and requires a full rewrite (see
promote_tensors.py).

⚠️ Safety, because a mistake here corrupts hours of forging:
  - the replacement must occupy EXACTLY the same number of bytes;
  - its dtype is checked against the type declared in the GGUF;
  - the original bytes are saved under <file>.backups/ before the first write,
    and the FIRST original always wins, so repeated experiments cannot lose it;
  - restore() puts the tensor back exactly as it was.

The hard-won rule behind that third point: never patch a file in place without
saving the bytes you touch. We once zeroed 163 MB of second-plane scales with
no backup, and recomputing them cost a 49-minute GPU pass that a 163 MB copy
would have avoided.

Self-test:
    gguf_surgeon.py model.gguf [--tensor NAME]
writes ones over one F32 tensor, verifies, restores, and verifies again.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "/home/angelo/build-llamacpp/gguf-py")
from gguf import GGUFReader


class GGUFSurgeon:
    """In-place tensor editor for a GGUF file, with automatic backups."""

    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.reader = GGUFReader(str(path))              # read-only mmap
        self.index = {t.name: t for t in self.reader.tensors}
        self.backups = self.path.parent / (self.path.name + ".backups")

    def read(self, name: str) -> np.ndarray:
        """A copy of the tensor's raw bytes, as stored."""
        return np.array(self.index[name].data)

    def _position(self, name: str) -> tuple[int, int]:
        t = self.index[name]
        return int(t.data_offset), int(t.n_bytes)

    def write(self, name: str, data: np.ndarray) -> None:
        """Replace the tensor's bytes. Same dtype and same size, or it refuses."""
        old = np.array(self.index[name].data)
        new = np.ascontiguousarray(data)
        assert new.dtype == old.dtype, f"{name}: dtype {new.dtype} != {old.dtype}"
        assert new.nbytes == old.nbytes, f"{name}: {new.nbytes} bytes != {old.nbytes}"
        offset, _ = self._position(name)
        self.backups.mkdir(exist_ok=True)
        backup = self.backups / (name.replace("/", "_") + ".bin")
        if not backup.exists():                          # the FIRST original wins
            backup.write_bytes(old.tobytes())
        with open(self.path, "r+b") as f:
            f.seek(offset)
            f.write(new.tobytes())

    def restore(self, name: str) -> None:
        """Put the tensor back exactly as it was before the first write."""
        backup = self.backups / (name.replace("/", "_") + ".bin")
        assert backup.exists(), f"no backup for {name}"
        offset, n_bytes = self._position(name)
        data = backup.read_bytes()
        assert len(data) == n_bytes, f"{name}: backup is {len(data)} bytes, tensor is {n_bytes}"
        with open(self.path, "r+b") as f:
            f.seek(offset)
            f.write(data)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("gguf")
    ap.add_argument("--tensor", default=None, help="tensor to test on (default: first F32)")
    a = ap.parse_args()

    s = GGUFSurgeon(a.gguf)
    name = a.tensor or next(n for n, t in s.index.items() if t.tensor_type.name == "F32")
    before = s.read(name)
    print(f"round-trip test on {name}: {before.shape} {before.dtype}")
    s.write(name, before * 0 + 1.0)
    assert np.allclose(GGUFSurgeon(a.gguf).read(name), 1.0), "write did not take effect"
    s.restore(name)
    assert np.array_equal(GGUFSurgeon(a.gguf).read(name), before), "restore did not recover the original"
    print("✅ write + restore are exact")
