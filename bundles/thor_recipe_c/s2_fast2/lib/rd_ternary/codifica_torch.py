"""Codifica dei piani PIENI in torch, su qualunque dispositivo (CPU, CUDA, ROCm/HIP),
a lotti: stessi byte, bit per bit, dei codificatori numpy di riferimento
(formato.encode_full e gguf_piani.encode_rows), che restano il percorso di ripiego.

Serve al controllo di rimaterializzazione: con decine di migliaia di esperti la
codifica tensore per tensore in numpy (e il passaggio GPU -> CPU di ogni piano)
diventa il collo di bottiglia della verifica. Equivalenza in tests/test_percorso_veloce.py.

Formati (vedi formato.py e gguf_piani.py per il layout):
  tq1      : per gruppo da 256  [48 B base 3 (cifra 0 meno significativa) | 4 B a 2 bit | scala fp16]
  tq2      : per gruppo da 256  [64 B a 2 bit | scala fp16]
  tq1r1024 : blocco GGML TQ1_0R_1024 da 1024 (4 quarti TQ1_0 da 52 B + 8 scale fp16) = 224 B
  (cifre {-1,0,1}: tq1/tq2 -> {2,0,1};  GGML -> {0,1,2} con cifra 0 PIU' significativa)
"""
from __future__ import annotations

import torch

from .formato import GROUP


def _scale_bytes(scales: torch.Tensor) -> torch.Tensor:
    """fp16 [..., n] -> uint8 [..., 2n] little-endian (come numpy '<f2')."""
    s = scales.to(torch.float16).contiguous()
    u = s.view(torch.int16).to(torch.int32) & 0xFFFF
    lo, hi = u & 0xFF, (u >> 8) & 0xFF
    return torch.stack([lo, hi], dim=-1).reshape(*s.shape[:-1], -1).to(torch.uint8)


def _pack2(u: torch.Tensor) -> torch.Tensor:
    """[..., 4m] valori 0..3 -> [..., m] byte (elemento 0 nei bit bassi)."""
    q = u.reshape(*u.shape[:-1], -1, 4)
    return (q[..., 0] | (q[..., 1] << 2) | (q[..., 2] << 4) | (q[..., 3] << 6)).to(torch.uint8)


def encode_full_torch(planes: torch.Tensor, scales: torch.Tensor, packing: str = "tq1") -> torch.Tensor:
    """planes int8 [..., o, i] + scales fp16 [..., o, ng] -> uint8 [..., n_byte]
    (n_byte = formato.full_plane_bytes(o, i, packing)); identico a encode_full riga per riga."""
    *lead, o, i = planes.shape
    if packing == "tq1r1024":
        return encode_tq1_0r_1024_torch(planes, scales)
    if packing == "tq1g64":
        return encode_tq1_0_g64_torch(planes, scales)
    if packing == "tq1r1024g64":
        return encode_tq1_0r_1024_g64_torch(planes, scales)
    ng = i // GROUP
    t = planes.reshape(*lead, o * ng, GROUP).to(torch.int32)
    u = torch.where(t < 0, torch.full_like(t, 2), t)                 # {-1,0,1} -> {2,0,1}
    if packing == "tq2":
        packed = _pack2(u)                                          # [..., n_g, 64]
    elif packing == "tq1":
        a = u[..., :240].reshape(*lead, o * ng, 48, 5)
        w = torch.tensor([1, 3, 9, 27, 81], dtype=torch.int32, device=u.device)
        p5 = ((a * w).sum(-1) & 0xFF).to(torch.uint8)               # come numpy: uint16 -> uint8
        p4 = _pack2(u[..., 240:])
        packed = torch.cat([p5, p4], dim=-1)
    else:
        raise ValueError(packing)
    sc = _scale_bytes(scales.reshape(*lead, o * ng, 1))
    return torch.cat([packed, sc], dim=-1).reshape(*lead, -1)


def _encode_tq1_ggml(t: torch.Tensor):
    """trit [..., N, 256] in {-1,0,1} -> (qs [..., N, 48], qh [..., N, 4]) come quantize_row_tq1_0_ref."""
    u = t.to(torch.int32) + 1
    w5 = torch.tensor([81, 27, 9, 3, 1], dtype=torch.int32, device=u.device)
    a = u[..., :160].reshape(*u.shape[:-1], 5, 32)
    b = u[..., 160:240].reshape(*u.shape[:-1], 5, 16)
    c = u[..., 240:].reshape(*u.shape[:-1], 4, 4)
    qa = (a * w5[:, None]).sum(-2)
    qb = (b * w5[:, None]).sum(-2)
    qc = (c * w5[:4, None]).sum(-2)
    qs = torch.cat([qa, qb], dim=-1)
    qs = torch.div(qs * 256 + 242, 243, rounding_mode="floor").to(torch.uint8)
    qh = torch.div(qc * 256 + 242, 243, rounding_mode="floor").to(torch.uint8)
    return qs, qh


def encode_tq1_0_torch(planes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """GGML TQ1_0: planes [..., o, i] + scales [..., o, i//256] -> uint8 [..., o*(i//256)*54]."""
    *lead, o, i = planes.shape
    nb = o * (i // 256)
    qs, qh = _encode_tq1_ggml(planes.reshape(*lead, nb, 256))
    dd = _scale_bytes(scales.reshape(*lead, nb, 1))
    return torch.cat([qs, qh, dd], dim=-1).reshape(*lead, -1)


def encode_tq1_0r_1024_torch(planes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """GGML TQ1_0R_1024: planes [..., o, i] (base ruotata) + scales [..., o, i//128]
    -> uint8 [..., o*(i//1024)*224]."""
    *lead, o, i = planes.shape
    nb = o * (i // 1024)
    qs, qh = _encode_tq1_ggml(planes.reshape(*lead, nb * 4, 256))
    dd = _scale_bytes(scales.reshape(*lead, nb, 8))
    return torch.cat([qs.reshape(*lead, nb, 192), qh.reshape(*lead, nb, 16), dd], dim=-1).reshape(*lead, -1)


def encode_tq1_0_g64_torch(planes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """GGML TQ1_0_G64 (6/10): planes [..., o, i] + scales [..., o, i//64] -> uint8 [..., o*(i//256)*60]."""
    *lead, o, i = planes.shape
    nb = o * (i // 256)
    qs, qh = _encode_tq1_ggml(planes.reshape(*lead, nb, 256))
    dd = _scale_bytes(scales.reshape(*lead, nb, 4))
    return torch.cat([qs, qh, dd], dim=-1).reshape(*lead, -1)


def encode_tq1_0r_1024_g64_torch(planes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """GGML TQ1_0R_1024_G64 (6/10): planes [..., o, i] (base ruotata) + scales [..., o, i//64]
    -> uint8 [..., o*(i//1024)*240]."""
    *lead, o, i = planes.shape
    nb = o * (i // 1024)
    qs, qh = _encode_tq1_ggml(planes.reshape(*lead, nb * 4, 256))
    dd = _scale_bytes(scales.reshape(*lead, nb, 16))
    return torch.cat([qs.reshape(*lead, nb, 192), qh.reshape(*lead, nb, 16), dd], dim=-1).reshape(*lead, -1)
