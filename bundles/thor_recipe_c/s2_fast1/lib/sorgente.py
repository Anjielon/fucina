"""Sorgente dei pesi da ternarizzare: il modello PIENO in bf16 (safetensors HF, @@ROOT:models@@/gguf/thor-src-hf),
riportato nell'ORDINE DEL GGUF con la mappa bit-esatta (mappa_completa_gguf_hf.json + _perm.npz, sha256 atteso
6114b1f4...). Piu' il Delta W della LoRA di EB3 (lora.npz dell'istantanea passo_000020) per i tensori che EB3 ha
allenato (Delta W = s (B A - B0 A0), in fp64 poi fp32, stessa formula di e2e_lora.delta_file_da).

Lettura a pezzi: un tensore alla volta con os.pread sull'intervallo esatto dell'header safetensors, poi
posix_fadvise(DONTNEED) per non riempire la cache delle pagine (RAM condivisa con l'allenamento EB3).
Nessuna GPU: solo numpy/torch CPU.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
from pathlib import Path
from typing import Optional

import numpy as np

HF = Path(os.getenv("TT_HF", "@@ROOT:models@@/gguf/thor-src-hf"))
MAPPA_JSON = Path("@@ROOT:forge@@/thor_v2/mappa_completa_gguf_hf.json")
MAPPA_NPZ = Path("@@ROOT:forge@@/thor_v2/mappa_completa_gguf_hf_perm.npz")
SHA_MAPPA_NPZ = "6114b1f4500b364e13c2b8b0f2d93836080ec75c37158a697265471360e14b16"
LORA_NPZ = Path(os.getenv("TT_LORA", "@@ROOT:depot@@/e2e_istantanee/EB3/passo_000020/lora.npz"))
SHA_LORA = "002ad8146bea6cdd26603bc9424575ef40d3969d4a6dd6e7a88e158b110b87b7"   # meta.json di passo_000020


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def bf16_a_f32(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.uint32) << 16).view(np.float32)


class SorgenteHF:
    def __init__(self, verifica_mappa: bool = True):
        cfg = json.load(open(HF / "config.json"))
        tc = cfg.get("text_config", cfg)
        self.tipi_strato = ["linear" if t == "linear_attention" else "full" for t in tc["layer_types"]]
        self.mappa = json.load(open(MAPPA_JSON))
        if verifica_mappa:
            s = sha256_file(MAPPA_NPZ)
            assert s == SHA_MAPPA_NPZ, f"mappa perm con sha {s} != atteso {SHA_MAPPA_NPZ}"
        self.perm = dict(np.load(MAPPA_NPZ))
        self.wmap = json.load(open(HF / "model.safetensors.index.json"))["weight_map"]
        self._hdr: dict[str, tuple[int, dict]] = {}
        self._lora = None

    # ─────────────────────────── lettura grezza ───────────────────────────
    def _header(self, fname: str):
        if fname not in self._hdr:
            with open(HF / fname, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                h = json.loads(f.read(n))
            self._hdr[fname] = (8 + n, h)
        return self._hdr[fname]

    def leggi_hf(self, nome_hf: str) -> tuple[np.ndarray, str]:
        """Tensore HF come float32 (esatto: bf16 -> f32 non perde bit) + sha256 dei byte bf16 letti."""
        fname = self.wmap[nome_hf]
        base, h = self._header(fname)
        e = h[nome_hf]
        assert e["dtype"] == "BF16", (nome_hf, e["dtype"])
        a, b = e["data_offsets"]
        fd = os.open(HF / fname, os.O_RDONLY)
        try:
            buf = bytearray(b - a)
            mv = memoryview(buf)
            letti = 0
            while letti < len(buf):
                n = os.preadv(fd, [mv[letti:]], base + a + letti)
                assert n > 0
                letti += n
            os.posix_fadvise(fd, base + a, b - a, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
        sha = hashlib.sha256(buf).hexdigest()
        u16 = np.frombuffer(buf, dtype=np.uint16).reshape(e["shape"])
        return u16, sha

    # ─────────────────────────── nome GGUF -> HF + relazione ───────────────────────────
    def voce(self, nome_gguf: str) -> dict:
        if nome_gguf == "token_embd.weight":
            return {"hf": "model.language_model.embed_tokens.weight", "rel": "identita"}
        if nome_gguf == "output.weight":
            return {"hf": "lm_head.weight", "rel": "identita"}
        p = nome_gguf.split(".")
        assert p[0] == "blk", nome_gguf
        L = int(p[1])
        fam = ".".join(p[2:])
        tipo = self.tipi_strato[L]
        v = dict(self.mappa[tipo][fam])
        v["hf"] = f"model.language_model.layers.{L}.{v['hf']}"
        return v

    def u16_ordine_gguf(self, nome_gguf: str, applica_perm: bool = True) -> tuple[np.ndarray, dict]:
        v = self.voce(nome_gguf)
        assert v.get("fetta") in (None,), (nome_gguf, v)          # solo matrici intere (i densi); niente esperti qui
        u16, sha = self.leggi_hf(v["hf"])
        rel = v["rel"]
        if applica_perm and rel == "perm_righe":
            u16 = u16[self.perm[v["perm_npz"]]]
        elif applica_perm and rel == "perm_colonne":
            u16 = u16[:, self.perm[v["perm_npz"]]]
        else:
            assert rel == "identita" or not applica_perm, (nome_gguf, rel)
        prov = {"hf_file": self.wmap[v["hf"]], "hf_tensore": v["hf"], "sha256_bf16_hf": sha, "relazione": rel,
                "perm_npz": v.get("perm_npz"), "sha256_mappa_npz": SHA_MAPPA_NPZ}
        return np.ascontiguousarray(u16), prov

    # ─────────────────────────── LoRA di EB3 (Delta W) ───────────────────────────
    def delta_lora(self, nome_gguf: str) -> Optional[np.ndarray]:
        """Delta W nell'ordine del GGUF, fp32, o None se il tensore non ha adattatore o Delta W e' zero esatto."""
        import torch
        if self._lora is None:
            s = sha256_file(LORA_NPZ)
            assert s == SHA_LORA, f"lora.npz sha {s} != {SHA_LORA}"
            self._lora = np.load(LORA_NPZ)
            self._lora_meta = json.load(open(LORA_NPZ.parent / "meta.json"))["lora"]
        k = nome_gguf + "|A"
        if k not in self._lora.files:
            return None
        A = torch.as_tensor(self._lora[nome_gguf + "|A"]).to(torch.float64)
        B = torch.as_tensor(self._lora[nome_gguf + "|B"]).to(torch.float64)
        A0 = torch.as_tensor(self._lora[nome_gguf + "|A0"]).to(torch.float64)
        B0 = torch.as_tensor(self._lora[nome_gguf + "|B0"]).to(torch.float64)
        s = float(self._lora_meta["scala"])
        d_hf = ((B @ A - B0 @ A0) * s).to(torch.float32).numpy()     # ordine HF, come e2e_lora.delta_file_da
        if not np.any(d_hf):
            return None
        v = self._lora_meta["voci"][nome_gguf]
        if v["rel"] == "perm_righe":
            d_hf = d_hf[self.perm[v["perm_npz"]]]
        elif v["rel"] == "perm_colonne":
            d_hf = d_hf[:, self.perm[v["perm_npz"]]]
        return np.ascontiguousarray(d_hf)

    def peso_bersaglio(self, nome_gguf: str) -> tuple[np.ndarray, dict]:
        """Il peso da ternarizzare, fp32, ordine GGUF: bf16 pieno (+ Delta W LoRA di EB3 se allenato)."""
        u16, prov = self.u16_ordine_gguf(nome_gguf)
        W = bf16_a_f32(u16)
        d = self.delta_lora(nome_gguf)
        if d is not None:
            assert d.shape == W.shape, (nome_gguf, d.shape, W.shape)
            W = W + d
            prov["delta_lora"] = {"file": str(LORA_NPZ), "sha256": SHA_LORA,
                                  "norma_rel": float(np.linalg.norm(d) / max(np.linalg.norm(W - d), 1e-30))}
        else:
            prov["delta_lora"] = None
        return W, prov
