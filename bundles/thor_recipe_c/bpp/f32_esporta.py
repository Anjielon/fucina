#!/usr/bin/env python3
"""f32_esporta.py — F32-E2E (26/10/2026): a pack f32_<passo>.pt of cardan_congiunto_scale.py --allena-f32 -> the tensor
directory that raffina/patch_gguf.py writes into the HOST GGUF (same names, types, shapes, offsets: only the bytes of the
F32 tensors change; trits, fp16 scales, header and alignment identical to the byte). CPU only, seconds.

uso: f32_esporta.py --base OSPITE.gguf --pacco f32_400.pt --uscita DIR [--tutti]
     f32_esporta.py --base OSPITE.gguf --da-gguf --uscita DIR --tutti      (identity: the pack is the base itself)
  DIR/tensori/<nome>.npy   uint8 [n_bytes] = the fp32 values, little endian, C order of the GGUF tensor (numpy shape)
  DIR/esporta.json         tensors, changed values per family, max relative change, sha256 of every written buffer
Only the tensors whose bytes differ from the base are written (--tutti: all of the pack). Exit 3 = nothing differs
(the patched GGUF would BE the base: the chain then measures the base). Checks: name in the base, type F32, element
count, n_bytes, finite values, and (--da-gguf) every written buffer equal to the base bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "@@ROOT:models@@/build/llamacpp-tq1-k1024/gguf-py")
from gguf import GGUFReader  # noqa: E402


def famiglia(nome: str) -> str:
    m = re.match(r"blk\.\d+\.(.+?)(\.weight|\.bias)?$", nome)
    return m.group(1) if m else nome.replace(".weight", "")


def byte_base(t) -> np.ndarray:
    return np.asarray(t.data).reshape(-1).view(np.uint8)


def tensori(base: str) -> dict:
    """{name: ReaderTensor} of the GGUF. ONE reader per process: this gguf-py fork costs ~470 MB of RSS per reader."""
    return {t.name: t for t in GGUFReader(base, "r").tensors}


def f32_della_base(tens: dict) -> dict:
    """The F32 tensors the student trains (layers 0-39 + output_norm; the MTP layer 40 is never loaded)."""
    return {n: np.asarray(t.data, dtype=np.float32) for n, t in tens.items() if t.tensor_type.name == "F32"
            and (n == "output_norm.weight" or not n.startswith("blk.40."))}


def esporta(base: str, valori: dict, uscita: Path, tutti: bool = False, tens: dict = None) -> dict:
    """valori: {nome: fp32 array (any shape with the tensor's element count)} -> DIR/tensori/*.npy; returns the report.
    `tens`: the {name: ReaderTensor} of `base` if already open (else it is opened here)."""
    tens = tensori(base) if tens is None else tens
    dt = uscita / "tensori"
    dt.mkdir(parents=True, exist_ok=True)
    rep = {"base": base, "tensori_pacco": len(valori), "scritti": 0, "uguali_alla_base": 0, "per_famiglia": {},
           "sha256": {}}
    for nome in sorted(valori):
        if nome not in tens:
            raise KeyError(f"{nome}: non e' nel GGUF base")
        t = tens[nome]
        if t.tensor_type.name != "F32":
            raise TypeError(f"{nome}: tipo {t.tensor_type.name} nel GGUF base, non F32")
        v = np.ascontiguousarray(np.asarray(valori[nome], dtype=np.float32))
        if v.size != int(t.n_elements):
            raise ValueError(f"{nome}: {v.size} valori, il GGUF ne ha {int(t.n_elements)}")
        if not np.isfinite(v).all():
            raise ValueError(f"{nome}: valori non finiti")
        b = v.reshape(-1).astype("<f4", copy=False).view(np.uint8)
        if b.nbytes != int(t.n_bytes):
            raise ValueError(f"{nome}: {b.nbytes} byte, il GGUF ne ha {int(t.n_bytes)}")
        vb = byte_base(t)
        w0 = vb.view("<f4").astype(np.float64)
        diversi = int((b.view("<u4") != vb.view("<u4")).sum()) if not np.array_equal(b, vb) else 0   # bit, not value
        f = rep["per_famiglia"].setdefault(famiglia(nome), {"tensori": 0, "cambiati": 0, "valori_cambiati": 0,
                                                          "rel_max": 0.0, "d2": 0.0, "n2": 0.0})
        f["tensori"] += 1
        d = b.view("<f4").astype(np.float64) - w0
        f["d2"] += float((d * d).sum()); f["n2"] += float((w0 * w0).sum())
        if diversi:
            f["cambiati"] += 1; f["valori_cambiati"] += diversi
            f["rel_max"] = max(f["rel_max"], float(np.abs(d).max() / max(np.abs(w0).max(), 1e-30)))
        else:
            rep["uguali_alla_base"] += 1
        if diversi or tutti:
            np.save(dt / f"{nome}.npy", b)
            rep["sha256"][nome] = hashlib.sha256(b.tobytes()).hexdigest()
            rep["scritti"] += 1
    for f in rep["per_famiglia"].values():
        f["rel_norma"] = (f.pop("d2") / f.pop("n2")) ** 0.5 if f["n2"] > 0 else 0.0
    return rep


def identita(rep: dict, tens: dict) -> str:
    """--da-gguf: every written buffer must be the base's own bytes."""
    for n in rep["sha256"]:
        if hashlib.sha256(byte_base(tens[n]).tobytes()).hexdigest() != rep["sha256"][n]:
            raise ValueError(f"identita': {n} scritto diverso dai byte del GGUF base")
    return f"{len(rep['sha256'])} tensori scritti = byte del GGUF base (sha256 uguali)"


def main(argv=None, tens: dict = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--pacco", default=None)
    ap.add_argument("--da-gguf", action="store_true", help="identity test: the pack = the F32 tensors of the base")
    ap.add_argument("--uscita", required=True)
    ap.add_argument("--tutti", action="store_true")
    a = ap.parse_args(argv)
    uscita = Path(a.uscita)
    uscita.mkdir(parents=True, exist_ok=True)
    tens = tensori(a.base) if tens is None else tens
    if a.da_gguf:
        valori = f32_della_base(tens)
        meta = {"da_gguf": True}
    else:
        import torch
        d = torch.load(a.pacco, map_location="cpu", weights_only=False)
        valori = {n: v.numpy() for n, v in d["valori"].items()}
        for n, s in d.get("sha256_fp32", {}).items():   # the pack's own checksum (bytes as the trainer saved them)
            if hashlib.sha256(np.ascontiguousarray(valori[n], dtype=np.float32).tobytes()).hexdigest() != s:
                raise ValueError(f"{n}: sha256 diverso da quello scritto nel pacco")
        meta = d.get("meta", {})
    rep = esporta(a.base, valori, uscita, a.tutti, tens)
    rep["meta"] = meta
    if a.da_gguf:
        rep["identita"] = identita(rep, tens)
    (uscita / "esporta.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ("tensori_pacco", "scritti", "uguali_alla_base")}
                     | ({"identita": rep["identita"]} if "identita" in rep else {})))
    return 3 if rep["scritti"] == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
