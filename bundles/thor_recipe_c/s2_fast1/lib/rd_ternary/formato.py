"""Formato dei piani ternari: chiavi, dequantizzazione, conteggio byte ESATTO,
codifica reale (per verificare il conteggio) e rotazione di Hadamard a blocchi.

Convenzioni (valgono in tutto il pacchetto):
  * un peso W ha forma [out, in] (come nn.Linear di HF);
  * i gruppi sono da GROUP=256 lungo la dimensione di INGRESSO;
  * W ~= sum_k s_k * T_k, con T_k in {-1,0,+1} (int8) e s_k fp16 di forma [out, in//256];
  * se un tensore ha l'ingresso ruotato con Hadamard-1024, tutto (pesi teacher,
    piani, H_I) viene espresso nella BASE MEMORIZZATA: W' = W H, H_I' = H H_I H.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

GROUP = 256
SCALE_BYTES = 2  # una scala fp16 per gruppo
HADAMARD_BLOCK = 1024

# byte dei soli trit impaccati per un gruppo da 256 (la scala fp16 si aggiunge a parte)
#  tq1: 240 trit in base 3 (5 per byte -> 48 B) + 16 trit a 2 bit (4 B) = 52 B  (1.6875 bpw con scala)
#  tq2: 256 trit a 2 bit = 64 B                                                (2.0625 bpw con scala)
PACKED_TRIT_BYTES = {"tq1": 52, "tq2": 64}

# packing -> (pesi per blocco, byte per blocco SCALE INCLUSE, pesi per scala).
#  tq1r1024: layout GGML TQ1_0R_1024 (4 quarti TQ1 da 52 B + 8 scale fp16) = 224 B / 1024
#            (1.75 bpw), scala ogni 128 pesi, trit nella base ruotata H(S w).
PACKINGS = {"tq1": (256, 54, 256), "tq2": (256, 66, 256), "tq1r1024": (1024, 224, 128),
            # 6/10 (ramo thor-llama cloud/scala-g64): scala fp16 ogni 64 pesi, stessi trit dei tipi madre
            #  tq1g64      : layout GGML TQ1_0_G64 (52 B di trit + 4 scale) = 60 B / 256   (1.875 bpw)
            #  tq1r1024g64 : layout GGML TQ1_0R_1024_G64 (4 quarti + 16 scale) = 240 B / 1024 (1.875 bpw)
            "tq1g64": (256, 60, 64), "tq1r1024g64": (1024, 240, 64)}
G64 = 64
G64_PACKING = {"tq1": "tq1g64", "tq1r1024": "tq1r1024g64"}   # packing madre -> variante a scala 64

SPARSE_CODINGS = ("local8", "bitmap", "auto")
LOCAL8_MAX_PER_GROUP = 255  # il contatore per gruppo e' un uint8


# --------------------------------------------------------------------------- chiavi
@dataclass(frozen=True, order=True)
class TensorKey:
    """Identifica un tensore: layer, esperto (-1 = tensore denso), tipo (gate/up/down/q/...)."""
    layer: int
    expert: int
    kind: str

    def name(self) -> str:
        e = f"E{self.expert}" if self.expert >= 0 else "D"
        return f"L{self.layer}.{e}.{self.kind}"

    @staticmethod
    def parse(s: str) -> "TensorKey":
        lay, e, kind = s.split(".", 2)
        return TensorKey(int(lay[1:]), -1 if e == "D" else int(e[1:]), kind)


@dataclass
class TernaryTensor:
    """Rappresentazione a piani di un tensore quantizzato.

    Il gruppo di scala si deduce dalle forme (in // scales.shape[-1]): 256 di default,
    128 per TQ1_0R_1024. sign: segni fissi S (+-1, lunghezza HADAMARD_BLOCK) applicati
    PRIMA di H nella rotazione (base memorizzata = H(S w)); None = solo H.
    packing: formato dei byte (chiave di PACKINGS); None = quello del MenuConfig."""
    planes: torch.Tensor  # int8 [K, out, in]
    scales: torch.Tensor  # fp16 [K, out, in//group]
    rotated: bool = False
    sign: Optional[torch.Tensor] = None
    packing: Optional[str] = None

    @property
    def n_planes(self) -> int:
        return int(self.planes.shape[0])

    @property
    def group(self) -> int:
        return int(self.planes.shape[-1]) // int(self.scales.shape[-1])


# --------------------------------------------------------------------------- dequantizzazione
def dequant(planes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """planes [...,K,o,i] int8 + scales [...,K,o,ng] -> pesi fp32 [...,o,i]."""
    *lead, k, o, i = planes.shape
    ng = int(scales.shape[-1])
    # accumulo piano per piano: niente temporaneo fp32 K volte grande quanto il tensore
    w = None
    for j in range(k):
        pj = planes[..., j, :, :].reshape(*lead, o, ng, i // ng).float()
        pj.mul_(scales[..., j, :, :].float().unsqueeze(-1))
        w = pj if w is None else w.add_(pj)
    return w.reshape(*lead, o, i)


def dequant_one(plane: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Un solo piano: plane [...,o,i] + scales [...,o,ng] -> fp32 [...,o,i]."""
    return dequant(plane.unsqueeze(-3), scales.unsqueeze(-3))


# --------------------------------------------------------------------------- conteggio byte
def full_plane_bytes(out: int, in_: int, packing: str = "tq1") -> int:
    """Byte esatti di un piano pieno: blocchi di trit impaccati + scale fp16."""
    blk, bbytes, _ = PACKINGS[packing]
    if in_ % blk:
        raise ValueError(f"in={in_} non multiplo di {blk}")
    return out * (in_ // blk) * bbytes


def packing_group(packing: str) -> int:
    return PACKINGS[packing][2]


def packing_for_scales(base_packing: str, in_: int, n_scale_cols: int) -> str:
    """Packing REALE di un piano dalle sue scale: se le colonne di scala sono in/64 e il
    packing madre ha la variante a scala 64, e' quella; altrimenti il packing madre."""
    if n_scale_cols * G64 == in_ and base_packing in G64_PACKING:
        return G64_PACKING[base_packing]
    return base_packing


def group_counts(plane: torch.Tensor, group: int = GROUP) -> torch.Tensor:
    """Numero di non-zeri per gruppo: [...,o,i] -> [...,o,ng]."""
    *lead, o, i = plane.shape
    return (plane != 0).reshape(*lead, o, i // group, group).sum(-1)


def sparse_plane_bytes(plane: torch.Tensor, coding: str = "local8", group: int = GROUP) -> torch.Tensor:
    """Byte esatti di un piano SPARSO (layout struct-of-arrays, vedi encode_sparse).

    local8 : contatori uint8 per gruppo (o*ng B) + scala fp16 per gruppo non vuoto
             + 1 B di indice locale (0..255) per non-zero + segni impaccati ceil(nnz/8)
    bitmap : bitmap 32 B per gruppo + scala per gruppo non vuoto + segni ceil(nnz/8)
    auto   : 1 B di intestazione + il minore dei due
    Ritorna un tensore int64 con le dimensioni di testa (batch).
    """
    *lead, o, i = plane.shape
    cnt = group_counts(plane, group)
    n_groups = o * (i // group)
    nonempty = (cnt > 0).sum((-1, -2)).long()
    nnz = cnt.sum((-1, -2)).long()
    signs = (nnz + 7) // 8
    local8 = n_groups + SCALE_BYTES * nonempty + nnz + signs
    bitmap = n_groups * (group // 8) + SCALE_BYTES * nonempty + signs
    if coding == "local8":
        if bool((cnt > LOCAL8_MAX_PER_GROUP).any()):
            raise ValueError("local8: un gruppo supera 255 non-zeri (contatore uint8)")
        return local8
    if coding == "bitmap":
        return bitmap
    if coding == "auto":
        too_many = (cnt > LOCAL8_MAX_PER_GROUP).flatten(-2).any(-1)
        local8 = torch.where(too_many, torch.full_like(local8, 1 << 62), local8)
        return 1 + torch.minimum(local8, bitmap)
    raise ValueError(coding)


# --------------------------------------------------------------------------- codifica reale
def _trits_u(plane: np.ndarray) -> np.ndarray:
    """{-1,0,1} -> {2,0,1} (cifre in base 3)."""
    return np.where(plane < 0, 2, plane).astype(np.uint8)


def encode_full(plane: torch.Tensor, scales: torch.Tensor, packing: str = "tq1") -> bytes:
    """Codifica a blocchi: per ogni gruppo [trit impaccati][scala fp16].
    tq1r1024: i byte REALI del tipo GGML TQ1_0R_1024 (block_tq1_0r_1024)."""
    if packing in ("tq1r1024", "tq1g64", "tq1r1024g64"):
        from .gguf_piani import PACKING_TYPE, encode_rows
        return encode_rows(plane.cpu().numpy().astype(np.int8), scales.cpu().numpy().astype(np.float16),
                           PACKING_TYPE[packing]).tobytes()
    o, i = plane.shape
    ng = i // GROUP
    u = _trits_u(plane.cpu().numpy().reshape(o * ng, GROUP))
    if packing == "tq2":
        q = u.reshape(-1, 64, 4)
        packed = (q[..., 0] | (q[..., 1] << 2) | (q[..., 2] << 4) | (q[..., 3] << 6)).astype(np.uint8)
    elif packing == "tq1":
        a = u[:, :240].reshape(-1, 48, 5).astype(np.uint16)
        p5 = (a * np.array([1, 3, 9, 27, 81], dtype=np.uint16)).sum(-1).astype(np.uint8)
        b = u[:, 240:].reshape(-1, 4, 4)
        p4 = (b[..., 0] | (b[..., 1] << 2) | (b[..., 2] << 4) | (b[..., 3] << 6)).astype(np.uint8)
        packed = np.concatenate([p5, p4], axis=1)
    else:
        raise ValueError(packing)
    sc = scales.cpu().numpy().astype("<f2").reshape(-1, 1).view(np.uint8)  # [n_g, 2]
    return np.concatenate([packed, sc], axis=1).tobytes()


def decode_full(buf: bytes, out: int, in_: int, packing: str = "tq1"):
    if packing in ("tq1r1024", "tq1g64", "tq1r1024g64"):
        from .gguf_piani import PACKING_TYPE, decode_rows
        p, sc = decode_rows(np.frombuffer(buf, dtype=np.uint8).reshape(out, -1), PACKING_TYPE[packing], in_)
        return torch.from_numpy(p.copy()), torch.from_numpy(sc.astype(np.float16))
    ng = in_ // GROUP
    nb = PACKED_TRIT_BYTES[packing]
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(out * ng, nb + SCALE_BYTES)
    packed, sc = arr[:, :nb], arr[:, nb:].copy().view("<f2").reshape(out, ng)
    if packing == "tq2":
        u = np.stack([(packed >> (2 * k)) & 3 for k in range(4)], -1).reshape(-1, GROUP)
    else:
        p5 = packed[:, :48].astype(np.int32)
        d = np.stack([(p5 // 3 ** k) % 3 for k in range(5)], -1).reshape(-1, 240)
        p4 = packed[:, 48:]
        e = np.stack([(p4 >> (2 * k)) & 3 for k in range(4)], -1).reshape(-1, 16)
        u = np.concatenate([d, e], axis=1)
    u = u.astype(np.int8)
    t = np.where(u == 2, np.int8(-1), u).astype(np.int8).reshape(out, in_)
    return torch.from_numpy(t), torch.from_numpy(sc.astype(np.float16))


def encode_sparse(plane: torch.Tensor, scales: torch.Tensor, coding: str = "local8", group: int = GROUP) -> bytes:
    """Layout struct-of-arrays (decodificabile, nessuna intestazione oltre a 'auto'):
       [contatori o bitmap][scale fp16 dei gruppi non vuoti][indici locali uint8][segni bit]
    Ordine: gruppi in row-major (riga, gruppo), posizioni crescenti; bit segno 1 = negativo.
    """
    o, i = plane.shape
    ng = i // group
    p = plane.cpu().numpy().reshape(o * ng, group)
    if coding == "auto":
        b_l = int(sparse_plane_bytes(plane, "auto", group))
        b_b = 1 + int(sparse_plane_bytes(plane, "bitmap", group))
        use = "bitmap" if b_l == b_b else "local8"
        return bytes([0 if use == "local8" else 1]) + encode_sparse(plane, scales, use, group)
    nzm = p != 0
    cnt = nzm.sum(1)
    grp, pos = np.nonzero(nzm)
    signs = np.packbits((p[grp, pos] < 0).astype(np.uint8)).tobytes()
    sc = scales.cpu().numpy().reshape(-1)[cnt > 0].astype("<f2").tobytes()
    if coding == "local8":
        if (cnt > LOCAL8_MAX_PER_GROUP).any():
            raise ValueError("local8: gruppo con >255 non-zeri")
        return cnt.astype(np.uint8).tobytes() + sc + pos.astype(np.uint8).tobytes() + signs
    if coding == "bitmap":
        return np.packbits(nzm.astype(np.uint8), axis=1).tobytes() + sc + signs
    raise ValueError(coding)


def decode_sparse(buf: bytes, out: int, in_: int, coding: str = "local8", group: int = GROUP):
    ng = in_ // group
    n_g = out * ng
    gb = group // 8
    b = np.frombuffer(buf, dtype=np.uint8)
    if coding == "auto":
        return decode_sparse(buf[1:], out, in_, "local8" if b[0] == 0 else "bitmap", group)
    if coding == "local8":
        cnt = b[:n_g].astype(np.int64)
        off = n_g
    else:
        mask = np.unpackbits(b[: n_g * gb].reshape(n_g, gb), axis=1).astype(bool)
        cnt = mask.sum(1)
        off = n_g * gb
    n_ne = int((cnt > 0).sum())
    sc_ne = b[off: off + 2 * n_ne].copy().view("<f2")
    off += 2 * n_ne
    nnz = int(cnt.sum())
    if coding == "local8":
        pos = b[off: off + nnz].astype(np.int64)
        grp = np.repeat(np.arange(n_g), cnt)
        off += nnz
    else:
        grp, pos = np.nonzero(mask)
    neg = np.unpackbits(b[off: off + (nnz + 7) // 8])[:nnz].astype(bool)
    plane = np.zeros((n_g, group), dtype=np.int8)
    plane[grp, pos] = np.where(neg, -1, 1)
    sc = np.zeros(n_g, dtype=np.float16)
    sc[cnt > 0] = sc_ne
    return torch.from_numpy(plane.reshape(out, in_)), torch.from_numpy(sc.reshape(out, ng))


# --------------------------------------------------------------------------- Hadamard
def fwht(x: torch.Tensor, block: int = HADAMARD_BLOCK) -> torch.Tensor:
    """Walsh-Hadamard normalizzata (ordine di Sylvester, simmetrica e ortonormale)
    applicata a blocchi di `block` lungo l'ultima dimensione."""
    *lead, n = x.shape
    if n % block:
        raise ValueError(f"dimensione {n} non multipla di {block}")
    nb = n // block
    y = x.reshape(-1, nb, block)
    h = 1
    while h < block:
        y = y.reshape(-1, nb, block // (2 * h), 2, h)
        a, b = y[..., 0, :], y[..., 1, :]
        y = torch.stack((a + b, a - b), dim=-2)
        h *= 2
    return y.reshape(*lead, n) / math.sqrt(block)


def _tile_sign(sign: torch.Tensor, n: int, like: torch.Tensor) -> torch.Tensor:
    s = sign.to(device=like.device, dtype=like.dtype).reshape(-1)
    return s.repeat(n // s.numel())


def rotate_weight_in(w: torch.Tensor, block: int = HADAMARD_BLOCK,
                     sign: Optional[torch.Tensor] = None) -> torch.Tensor:
    """W -> W S H per righe (base memorizzata q ~= H(S w)); senza sign: W -> W H.
    (W S H)(H S x) = W x: H simmetrica ortonormale, S^2 = I."""
    if sign is not None:
        w = w * _tile_sign(sign, w.shape[-1], w)
    return fwht(w, block)


def unrotate_weight_in(w: torch.Tensor, block: int = HADAMARD_BLOCK,
                       sign: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Inversa di rotate_weight_in: W' -> S H(W') per righe (come il dequant C)."""
    y = fwht(w, block)
    if sign is not None:
        y = y * _tile_sign(sign, y.shape[-1], y)
    return y


def rotate_fisher_in(a: torch.Tensor, block: int = HADAMARD_BLOCK,
                     sign: Optional[torch.Tensor] = None) -> torch.Tensor:
    """H_I -> H S H_I S H (covarianza dell'ingresso ruotato); senza sign: H H_I H."""
    if sign is not None:
        s = _tile_sign(sign, a.shape[-1], a)
        a = a * s[:, None] * s[None, :]
    return fwht(fwht(a, block).transpose(-1, -2), block).transpose(-1, -2)
