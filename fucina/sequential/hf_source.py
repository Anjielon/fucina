"""Source of the weights to ternarize: the FULL bf16 model (HF safetensors, directory thor-src-hf under
FUCINA_DATA_DIR, or TT_HF), brought into GGUF ORDER with the bit-exact map (gguf_hf_map.json + gguf_hf_perm.npz,
expected sha256 6114b1f4...). Plus the LoRA Delta W of EB3 (lora.npz of snapshot passo_000020) for the tensors that
EB3 trained (Delta W = s (B A - B0 A0), in fp64 then fp32, same formula as e2e_lora.delta_file_da).

Chunked reads: one tensor at a time with os.pread over the exact range given by the safetensors header, then
posix_fadvise(DONTNEED) so as not to fill the page cache (RAM shared with the EB3 training).
No GPU: numpy/torch on CPU only.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
from pathlib import Path
from typing import Optional

import numpy as np

from fucina.extdeps import data_path

_DATA = Path(__file__).resolve().parent.parent / "ternary" / "data"
HF = Path(os.getenv("TT_HF", data_path("thor-src-hf")))
MAP_JSON = _DATA / "gguf_hf_map.json"
MAP_NPZ = _DATA / "gguf_hf_perm.npz"
SHA_MAP_NPZ = "6114b1f4500b364e13c2b8b0f2d93836080ec75c37158a697265471360e14b16"
LORA_NPZ = Path(os.getenv("TT_LORA", data_path("e2e_snapshots", "EB3", "passo_000020", "lora.npz")))
SHA_LORA = "002ad8146bea6cdd26603bc9424575ef40d3969d4a6dd6e7a88e158b110b87b7"   # from meta.json of passo_000020


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def bf16_a_f32(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.uint32) << 16).view(np.float32)


class SourceHF:
    def __init__(self, verify_map: bool = True):
        cfg = json.load(open(HF / "config.json"))
        tc = cfg.get("text_config", cfg)
        self.types_layer = ["linear" if t == "linear_attention" else "full" for t in tc["layer_types"]]
        self.map_ = json.load(open(MAP_JSON))
        if verify_map:
            s = sha256_file(MAP_NPZ)
            assert s == SHA_MAP_NPZ, f"perm map with sha {s} != expected {SHA_MAP_NPZ}"
        self.perm = dict(np.load(MAP_NPZ))
        self.wmap = json.load(open(HF / "model.safetensors.index.json"))["weight_map"]
        self._hdr: dict[str, tuple[int, dict]] = {}
        self._lora = None

    # ─────────────────────────── raw read ───────────────────────────
    def _header(self, fname: str):
        if fname not in self._hdr:
            with open(HF / fname, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                h = json.loads(f.read(n))
            self._hdr[fname] = (8 + n, h)
        return self._hdr[fname]

    def read_hf(self, name_hf: str) -> tuple[np.ndarray, str]:
        """HF tensor as raw bf16 bits (uint16; bf16 -> f32 is exact, no bits lost) + sha256 of the bf16 bytes read."""
        fname = self.wmap[name_hf]
        base, h = self._header(fname)
        e = h[name_hf]
        assert e["dtype"] == "BF16", (name_hf, e["dtype"])
        a, b = e["data_offsets"]
        fd = os.open(HF / fname, os.O_RDONLY)
        try:
            buf = bytearray(b - a)
            mv = memoryview(buf)
            read = 0
            while read < len(buf):
                n = os.preadv(fd, [mv[read:]], base + a + read)
                assert n > 0
                read += n
            os.posix_fadvise(fd, base + a, b - a, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
        sha = hashlib.sha256(buf).hexdigest()
        u16 = np.frombuffer(buf, dtype=np.uint16).reshape(e["shape"])
        return u16, sha

    # ─────────────────────────── GGUF name -> HF + relation ───────────────────────────
    def entry(self, name_gguf: str) -> dict:
        if name_gguf == "token_embd.weight":
            return {"hf": "model.language_model.embed_tokens.weight", "rel": "identity"}
        if name_gguf == "output.weight":
            return {"hf": "lm_head.weight", "rel": "identity"}
        p = name_gguf.split(".")
        assert p[0] == "blk", name_gguf
        L = int(p[1])
        fam = ".".join(p[2:])
        type_ = self.types_layer[L]
        v = dict(self.map_[type_][fam])
        v["hf"] = f"model.language_model.layers.{L}.{v['hf']}"
        return v

    def u16_order_gguf(self, name_gguf: str, apply_perm: bool = True) -> tuple[np.ndarray, dict]:
        v = self.entry(name_gguf)
        assert v.get("slice_") in (None,), (name_gguf, v)          # whole matrices only (the dense tensors); no experts here
        u16, sha = self.read_hf(v["hf"])
        rel = v["rel"]
        if apply_perm and rel == "perm_rows":
            u16 = u16[self.perm[v["perm_npz"]]]
        elif apply_perm and rel == "perm_columns":
            u16 = u16[:, self.perm[v["perm_npz"]]]
        else:
            assert rel == "identity" or not apply_perm, (name_gguf, rel)
        prov = {"hf_file": self.wmap[v["hf"]], "hf_tensor": v["hf"], "sha256_bf16_hf": sha, "relation": rel,
                "perm_npz": v.get("perm_npz"), "sha256_map_npz": SHA_MAP_NPZ}
        return np.ascontiguousarray(u16), prov

    # ─────────────────────────── EB3 LoRA (Delta W) ───────────────────────────
    def delta_lora(self, name_gguf: str) -> Optional[np.ndarray]:
        """Delta W in GGUF order, fp32, or None if the tensor has no adapter or Delta W is exactly zero."""
        import torch
        if self._lora is None:
            s = sha256_file(LORA_NPZ)
            assert s == SHA_LORA, f"lora.npz sha {s} != {SHA_LORA}"
            self._lora = np.load(LORA_NPZ)
            self._lora_meta = json.load(open(LORA_NPZ.parent / "meta.json"))["lora"]
        k = name_gguf + "|A"
        if k not in self._lora.files:
            return None
        A = torch.as_tensor(self._lora[name_gguf + "|A"]).to(torch.float64)
        B = torch.as_tensor(self._lora[name_gguf + "|B"]).to(torch.float64)
        A0 = torch.as_tensor(self._lora[name_gguf + "|A0"]).to(torch.float64)
        B0 = torch.as_tensor(self._lora[name_gguf + "|B0"]).to(torch.float64)
        s = float(self._lora_meta["scale"])
        d_hf = ((B @ A - B0 @ A0) * s).to(torch.float32).numpy()     # HF order, as in e2e_lora.delta_file_da
        if not np.any(d_hf):
            return None
        v = self._lora_meta["entries"][name_gguf]
        if v["rel"] == "perm_rows":
            d_hf = d_hf[self.perm[v["perm_npz"]]]
        elif v["rel"] == "perm_columns":
            d_hf = d_hf[:, self.perm[v["perm_npz"]]]
        return np.ascontiguousarray(d_hf)

    def weight_target(self, name_gguf: str) -> tuple[np.ndarray, dict]:
        """The weight to ternarize, fp32, GGUF order: full bf16 (+ EB3 LoRA Delta W if trained)."""
        u16, prov = self.u16_order_gguf(name_gguf)
        W = bf16_a_f32(u16)
        d = self.delta_lora(name_gguf)
        if d is not None:
            assert d.shape == W.shape, (name_gguf, d.shape, W.shape)
            W = W + d
            prov["delta_lora"] = {"file": str(LORA_NPZ), "sha256": SHA_LORA,
                                  "norm_rel": float(np.linalg.norm(d) / max(np.linalg.norm(W - d), 1e-30))}
        else:
            prov["delta_lora"] = None
        return W, prov
