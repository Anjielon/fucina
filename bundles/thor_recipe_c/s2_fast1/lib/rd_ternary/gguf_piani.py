"""Lettore dei piani ternari dai GGUF del ramo `velocita-ternario` (solo CPU, numpy).

Layout byte (fonte: ggml/src/ggml-common.h + ggml-quants.c del ramo):

  TQ1_0 (tipo 34), blocco 256 pesi, 54 B:
      qs[48]  5 trit/byte (base 3, cifra piu' significativa = elemento piu' basso)
      qh[4]   4 trit/byte
      d       fp16, UNA scala per i 256 pesi
    Ordine degli elementi (dequantize_row_tq1_0):
      byte m in 0..31  cifra n in 0..4  -> elemento n*32 + m
      byte 32+m (m<16) cifra n in 0..4  -> elemento 160 + n*16 + m
      qh[j]            cifra n in 0..3  -> elemento 240 + n*4 + j
    Cifra n di un byte b:  ((uint8)(b * 3^n) * 3) >> 8   (codifica a virgola fissa:
    b = ceil(v * 256 / 243), v = valore base 3 delle 5 cifre, cifre {0,1,2} -> {-1,0,+1}).

  TQ1_0R_1024 (tipo 45), blocco 1024 pesi, 224 B:
      qs[4][48]  qh[4][4]   4 "quarti" da 256 impacchettati ESATTAMENTE come TQ1_0
      d[8]       fp16, una scala ogni 128 pesi (d[e // 128])
    I trit vivono nella BASE RUOTATA: q ~= H(S*w) per blocco 1024 (S = segni fissi
    S1024, H = Walsh-Hadamard di Sylvester normalizzata). Dequant C:
      w_approx = S * H(d[g] * q)       (prima H, poi S)

Piani: ogni piano e' un tensore GGUF SEPARATO con lo stesso tipo e la stessa forma:
  `blk.0.attn_qkv.weight`, `blk.0.attn_qkv2.weight`, `blk.0.attn_qkv3.weight`
  (suffisso numerico prima di `.weight`; per i tensori fuori dai blocchi
  `output.weight`, `output2.weight`, ...). Gli esperti sono tensori 3D
  [ne0=in, ne1=out, ne2=E] (in numpy [E, out, byte_riga]).

La ricostruzione nella base memorizzata e'  W' = sum_k s_k T_k  (scale per gruppo);
nella base originale  W = S * H(W')  per i tipi ruotati.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .formato import HADAMARD_BLOCK, TensorKey, TernaryTensor
from .sorgenti import FisherFactors, FisherSource, PlaneInfo, PlaneSource, TeacherSource

GGUF_PY_DEFAULT = "@@ROOT:models@@/build/thor-velocita/gguf-py"
_LOG = logging.getLogger("rd_ternary.gguf_piani")

# id dei tipi ggml (ggml/include/ggml.h del ramo)
T_F32, T_F16, T_BF16 = 0, 1, 30
T_TQ1_0, T_TQ1_0R_1024 = 34, 45
T_TQ1_0_G64, T_TQ1_0R_1024_G64 = 46, 47      # ramo thor-llama cloud/scala-g64 (6/10): scala ogni 64
G64_OF = {T_TQ1_0: T_TQ1_0_G64, T_TQ1_0R_1024: T_TQ1_0R_1024_G64}   # tipo madre -> variante a scala 64
GGUF_PY_G64 = "@@ROOT:models@@/build/thor-g64/gguf-py"                  # gguf-py che conosce 46 e 47

SIGN_SEED = 20260920   # ~/tony-forgia/kernel-rotazione/gen_sign_1024.py


@dataclass(frozen=True)
class TernaryFormat:
    type_id: int
    name: str
    block: int          # pesi per blocco
    block_bytes: int    # byte per blocco (scale incluse)
    group: int          # pesi per scala
    rotated: bool
    packing: str        # nome del packing nel pacchetto (formato.full_plane_bytes)


FORMATS: Dict[int, TernaryFormat] = {
    T_TQ1_0: TernaryFormat(T_TQ1_0, "TQ1_0", 256, 54, 256, False, "tq1"),
    T_TQ1_0R_1024: TernaryFormat(T_TQ1_0R_1024, "TQ1_0R_1024", 1024, 224, 128, True, "tq1r1024"),
    T_TQ1_0_G64: TernaryFormat(T_TQ1_0_G64, "TQ1_0_G64", 256, 60, 64, False, "tq1g64"),
    T_TQ1_0R_1024_G64: TernaryFormat(T_TQ1_0R_1024_G64, "TQ1_0R_1024_G64", 1024, 240, 64, True, "tq1r1024g64"),
}
PACKING_TYPE = {f.packing: tid for tid, f in FORMATS.items()}


# --------------------------------------------------------------------------- segni S1024
_SIGN_CACHE: Dict[int, np.ndarray] = {}


def sign1024() -> np.ndarray:
    """S1024 (float32 +-1), stesso generatore di gen_sign_1024.py (verificato nei test
    contro il letterale C in ggml-quants.c)."""
    if 1024 not in _SIGN_CACHE:
        rng = np.random.default_rng(SIGN_SEED)
        _SIGN_CACHE[1024] = rng.choice([-1.0, 1.0], size=1024).astype(np.float32)
    return _SIGN_CACHE[1024]


# --------------------------------------------------------------------------- tabelle cifre
def _digit_lut(n_digits: int) -> np.ndarray:
    """LUT [256, n_digits] int8: cifra n del byte b secondo il decoder ggml, in {-1,0,1}."""
    b = np.arange(256, dtype=np.uint16)
    out = np.empty((256, n_digits), dtype=np.int8)
    for n in range(n_digits):
        q = (b * (3 ** n)) & 0xFF             # moltiplicazione uint8 con overflow
        out[:, n] = ((q * 3) >> 8).astype(np.int8) - 1
    return out


_LUT5 = _digit_lut(5)
_LUT4 = _digit_lut(4)


def _decode_tq1_trits(qs: np.ndarray, qh: np.ndarray) -> np.ndarray:
    """qs [N,48] uint8, qh [N,4] uint8 -> trit int8 [N,256] in ordine naturale."""
    n = qs.shape[0]
    a = _LUT5[qs[:, :32]].transpose(0, 2, 1).reshape(n, 160)      # [N,32,5] -> [N,5,32]
    b = _LUT5[qs[:, 32:48]].transpose(0, 2, 1).reshape(n, 80)     # [N,16,5] -> [N,5,16]
    c = _LUT4[qh].transpose(0, 2, 1).reshape(n, 16)               # [N,4,4]  -> [N,4(n),4(j)]
    return np.concatenate([a, b, c], axis=1)


def _encode_tq1_trits(t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """trit int8 [N,256] in {-1,0,1} -> (qs [N,48], qh [N,4]) come quantize_row_tq1_0_ref."""
    n = t.shape[0]
    u = (t.astype(np.int16) + 1).astype(np.uint16)                 # {0,1,2}
    w5 = np.array([81, 27, 9, 3, 1], dtype=np.uint16)              # cifra 0 piu' significativa
    a = u[:, :160].reshape(n, 5, 32)
    b = u[:, 160:240].reshape(n, 5, 16)
    c = u[:, 240:].reshape(n, 4, 4)                                # [N, m(cifra), j(byte)]
    qa = (a * w5[None, :, None]).sum(1)
    qb = (b * w5[None, :, None]).sum(1)
    qc = (c * np.array([81, 27, 9, 3], dtype=np.uint16)[None, :, None]).sum(1)   # = (sum*3) come in C
    qs = np.concatenate([qa, qb], axis=1)
    qs = ((qs * 256 + 242) // 243).astype(np.uint8)
    qh = ((qc * 256 + 242) // 243).astype(np.uint8)
    return qs, qh


# --------------------------------------------------------------------------- blocchi
def decode_tq1_0_blocks(raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """raw uint8 [N,54] -> (trit int8 [N,256], d fp16 [N,1])."""
    raw = np.ascontiguousarray(raw).reshape(-1, 54)
    t = _decode_tq1_trits(raw[:, :48], raw[:, 48:52])
    d = raw[:, 52:54].copy().view("<f2")
    return t, d


def encode_tq1_0_blocks(trits: np.ndarray, d: np.ndarray) -> np.ndarray:
    """trit [N,256] + d [N] (o [N,1]) -> uint8 [N,54] nel layout block_tq1_0."""
    trits = trits.reshape(-1, 256)
    qs, qh = _encode_tq1_trits(trits)
    dd = np.asarray(d, dtype="<f2").reshape(-1, 1).view(np.uint8)
    return np.concatenate([qs, qh, dd], axis=1)


def decode_tq1_0r_1024_blocks(raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """raw uint8 [N,224] -> (trit int8 [N,1024] base ruotata, d fp16 [N,8])."""
    raw = np.ascontiguousarray(raw).reshape(-1, 224)
    n = raw.shape[0]
    qs = raw[:, :192].reshape(n * 4, 48)        # quarti consecutivi
    qh = raw[:, 192:208].reshape(n * 4, 4)
    t = _decode_tq1_trits(qs, qh).reshape(n, 1024)
    d = raw[:, 208:224].copy().view("<f2").reshape(n, 8)
    return t, d


def encode_tq1_0r_1024_blocks(trits: np.ndarray, d: np.ndarray) -> np.ndarray:
    """trit [N,1024] (base ruotata) + d [N,8] -> uint8 [N,224] (block_tq1_0r_1024)."""
    trits = trits.reshape(-1, 1024)
    n = trits.shape[0]
    qs, qh = _encode_tq1_trits(trits.reshape(n * 4, 256))
    dd = np.asarray(d, dtype="<f2").reshape(n, 8).view(np.uint8)
    return np.concatenate([qs.reshape(n, 192), qh.reshape(n, 16), dd], axis=1)


def decode_tq1_0_g64_blocks(raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """raw uint8 [N,60] -> (trit int8 [N,256], d fp16 [N,4]) — block_tq1_0_g64."""
    raw = np.ascontiguousarray(raw).reshape(-1, 60)
    t = _decode_tq1_trits(raw[:, :48], raw[:, 48:52])
    d = raw[:, 52:60].copy().view("<f2").reshape(-1, 4)
    return t, d


def encode_tq1_0_g64_blocks(trits: np.ndarray, d: np.ndarray) -> np.ndarray:
    """trit [N,256] + d [N,4] -> uint8 [N,60] (block_tq1_0_g64: qs 48, qh 4, d[4])."""
    trits = trits.reshape(-1, 256)
    qs, qh = _encode_tq1_trits(trits)
    dd = np.asarray(d, dtype="<f2").reshape(-1, 4).view(np.uint8)
    return np.concatenate([qs, qh, dd], axis=1)


def decode_tq1_0r_1024_g64_blocks(raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """raw uint8 [N,240] -> (trit int8 [N,1024] base ruotata, d fp16 [N,16])."""
    raw = np.ascontiguousarray(raw).reshape(-1, 240)
    n = raw.shape[0]
    t = _decode_tq1_trits(raw[:, :192].reshape(n * 4, 48), raw[:, 192:208].reshape(n * 4, 4)).reshape(n, 1024)
    d = raw[:, 208:240].copy().view("<f2").reshape(n, 16)
    return t, d


def encode_tq1_0r_1024_g64_blocks(trits: np.ndarray, d: np.ndarray) -> np.ndarray:
    """trit [N,1024] (base ruotata) + d [N,16] -> uint8 [N,240] (block_tq1_0r_1024_g64)."""
    trits = trits.reshape(-1, 1024)
    n = trits.shape[0]
    qs, qh = _encode_tq1_trits(trits.reshape(n * 4, 256))
    dd = np.asarray(d, dtype="<f2").reshape(n, 16).view(np.uint8)
    return np.concatenate([qs.reshape(n, 192), qh.reshape(n, 16), dd], axis=1)


def decode_rows(raw_rows: np.ndarray, type_id: int, in_: int) -> Tuple[np.ndarray, np.ndarray]:
    """Righe grezze [out, byte_riga] -> (piano int8 [out,in], scale fp16 [out, in//group])."""
    f = FORMATS[type_id]
    out = raw_rows.shape[0]
    if in_ % f.block:
        raise ValueError(f"{f.name}: in={in_} non multiplo di {f.block}")
    if type_id == T_TQ1_0:
        t, d = decode_tq1_0_blocks(raw_rows)
    elif type_id == T_TQ1_0_G64:
        t, d = decode_tq1_0_g64_blocks(raw_rows)
    elif type_id == T_TQ1_0R_1024_G64:
        t, d = decode_tq1_0r_1024_g64_blocks(raw_rows)
    else:
        t, d = decode_tq1_0r_1024_blocks(raw_rows)
    return t.reshape(out, in_), d.reshape(out, in_ // f.group)


def encode_rows(plane: np.ndarray, scales: np.ndarray, type_id: int) -> np.ndarray:
    """Inverso di decode_rows: piano [out,in] + scale [out,in//group] -> uint8 [out, byte_riga]."""
    f = FORMATS[type_id]
    out, in_ = plane.shape
    nb = in_ // f.block
    if type_id == T_TQ1_0:
        enc = encode_tq1_0_blocks(plane.reshape(-1, 256), scales.reshape(-1))
    elif type_id == T_TQ1_0_G64:
        enc = encode_tq1_0_g64_blocks(plane.reshape(-1, 256), scales.reshape(-1, 4))
    elif type_id == T_TQ1_0R_1024_G64:
        enc = encode_tq1_0r_1024_g64_blocks(plane.reshape(-1, 1024), scales.reshape(-1, 16))
    else:
        enc = encode_tq1_0r_1024_blocks(plane.reshape(-1, 1024), scales.reshape(-1, 8))
    return enc.reshape(out, nb * f.block_bytes)


# --------------------------------------------------------------------------- dequant numpy
def unrotate_rows(w_stored: np.ndarray) -> np.ndarray:
    """Base memorizzata -> originale per TQ1_0R_1024:  w = S * H(w')  per blocco 1024."""
    x = torch.from_numpy(np.ascontiguousarray(w_stored, dtype=np.float32))
    from .formato import fwht
    y = fwht(x, HADAMARD_BLOCK).numpy()
    s = sign1024()
    return (y.reshape(*y.shape[:-1], -1, 1024) * s).reshape(y.shape)


def reconstruct(tt: TernaryTensor, original_basis: bool = True) -> np.ndarray:
    """W ~= sum_k s_k T_k (fp32 [out,in]); se original_basis e il tensore e' ruotato,
    applica S*H come il dequant C."""
    from .formato import dequant
    w = dequant(tt.planes, tt.scales).numpy()
    if original_basis and tt.rotated:
        w = unrotate_rows(w)
    return w


# --------------------------------------------------------------------------- GGUF
def _import_gguf(gguf_py: Optional[str]):
    """Un gguf-py che conosca i tipi ternari: TQ1_0R_1024 sempre; i tipi a scala 64 (46, 47)
    solo il ramo cloud/scala-g64 (GGUF_PY_G64), che si preferisce se esiste — e' un
    soprainsieme (stessi tipi + 46/47), quindi legge anche i GGUF di prima."""
    import importlib
    import os
    import sys
    try:
        import gguf  # noqa: F401
        if hasattr(gguf, "GGUFReader") and _has_type(gguf, T_TQ1_0R_1024) and (
                _has_type(gguf, T_TQ1_0R_1024_G64) or not os.path.isdir(GGUF_PY_G64)):
            return gguf
    except ImportError:
        pass
    p = gguf_py or (GGUF_PY_G64 if os.path.isdir(GGUF_PY_G64) else GGUF_PY_DEFAULT)
    if p in sys.path:
        sys.path.remove(p)
    sys.path.insert(0, p)
    for m in [m for m in list(sys.modules) if m == "gguf" or m.startswith("gguf.")]:
        del sys.modules[m]
    return importlib.import_module("gguf")


def _has_type(gguf, tid: int) -> bool:
    try:
        gguf.GGMLQuantizationType(tid)
        return True
    except ValueError:
        return False


_PLANE_RE = re.compile(r"^(?:blk\.(\d+)\.)?(.+?)\.weight$")
_EXPERT_KIND = {"ffn_gate_exps": "gate", "ffn_up_exps": "up", "ffn_down_exps": "down"}
_KIND_EXPERT = {v: k for k, v in _EXPERT_KIND.items()}


@dataclass
class _Entry:
    layer: int
    base: str                 # nome GGUF senza blk.N. e senza .weight (es. attn_qkv)
    tensors: List[object]     # ReaderTensor per piano (1, 2, 3, ...)
    fmt: TernaryFormat
    out: int
    in_: int
    n_experts: int            # 0 = denso
    # esperti con 2° piano PARZIALE: slot2[e] = indice nel tensore *_exps2, -1 = nessuno.
    # Da `blk.N.<base>2_map` (F32 [1, n_expert], ramo cloud/piani-variabili) oppure, senza
    # mappa ma con meno slot che esperti, il prefisso (expert_count2, ODINO).
    slot2: Optional[np.ndarray] = None

    def n_planes_of(self, expert: int) -> int:
        if self.slot2 is not None and expert >= 0:
            return len(self.tensors) - (1 if self.slot2[expert] < 0 else 0)
        return len(self.tensors)

    def slot(self, plane: int, expert: int) -> int:
        if plane == 1 and self.slot2 is not None:
            s = int(self.slot2[expert])
            if s < 0:
                raise KeyError(f"esperto {expert} senza 2° piano")
            return s
        return expert


def gguf_name(layer: int, base: str, plane: int = 1) -> str:
    suf = "" if plane == 1 else str(plane)
    pre = f"blk.{layer}." if layer >= 0 else ""
    return f"{pre}{base}{suf}.weight"


def _key_to_base(key: TensorKey) -> str:
    return _KIND_EXPERT.get(key.kind, key.kind) if key.expert >= 0 else key.kind


class GGUFPlaneSource(PlaneSource):
    """PlaneSource sui piani ternari di un GGUF (mmap, sola lettura, un tensore alla volta).

    Chiavi: dense  -> TensorKey(layer, -1, base)          (es. L3.D.attn_q; fuori blocco layer=-1)
            esperti-> TensorKey(layer, e, gate|up|down)    (da ffn_{gate,up,down}_exps)
    I piani si riconoscono dal suffisso numerico (base, base2, base3, ...) e devono
    avere stesso tipo e stessa forma. Le scale escono fp16 [K,out,in//group] con
    group = 256 (TQ1_0) o 128 (TQ1_0R_1024); i piani ruotati portano il segno S1024.
    """

    def __init__(self, path: str, layers: Optional[Iterable[int]] = None,
                 bases: Optional[Iterable[str]] = None, include_embeddings: bool = False,
                 max_experts: Optional[int] = None, gguf_py: Optional[str] = None):
        gguf = _import_gguf(gguf_py)
        self.path = path
        self._reader = gguf.GGUFReader(path, "r")          # np.memmap mode='r': mai scritto
        self._max_e = max_experts
        layers = set(layers) if layers is not None else None
        bases = set(bases) if bases is not None else None
        by_name = {t.name: t for t in self._reader.tensors if int(t.tensor_type) in FORMATS}
        self._entries: Dict[Tuple[int, str], _Entry] = {}
        for name, t in by_name.items():
            m = _PLANE_RE.match(name)
            if not m:
                continue
            layer = int(m.group(1)) if m.group(1) is not None else -1
            base = m.group(2)
            mm = re.match(r"^(.*?)(\d+)$", base)
            if mm and gguf_name(layer, mm.group(1)) in by_name and int(mm.group(2)) >= 2:
                continue  # piano 2+, raccolto dal piano 1
            if layers is not None and layer not in layers:
                continue
            if bases is not None and base not in bases:
                continue
            if not include_embeddings and base == "token_embd":
                continue
            planes = [t]
            k = 2
            while gguf_name(layer, base, k) in by_name:
                planes.append(by_name[gguf_name(layer, base, k)])
                k += 1
            fmt = FORMATS[int(t.tensor_type)]
            shp = [int(x) for x in t.shape]
            n_exp = shp[2] if len(shp) == 3 else 0
            if n_exp and base not in _EXPERT_KIND:
                raise ValueError(f"tensore 3D non riconosciuto come esperti: {name}")
            slot2 = None
            if n_exp and len(planes) >= 2:
                slot2 = self._slot_map(layer, base, planes[1], n_exp)
                if len(planes) > 2:
                    raise ValueError(f"{name}: piu' di 2 piani sugli esperti (il motore non li carica)")
            for j, p in enumerate(planes):
                ps = [int(x) for x in p.shape]
                same = ps == shp if (slot2 is None or j == 0) else ps[:2] == shp[:2]
                if int(p.tensor_type) != fmt.type_id or not same:
                    raise ValueError(f"{name}: piano {p.name} con tipo/forma diversi")
            self._entries[(layer, base)] = _Entry(layer, base, planes, fmt, shp[1], shp[0], n_exp, slot2)
        self._keys: List[TensorKey] = []
        for (layer, base), e in self._entries.items():
            if e.n_experts:
                ne = e.n_experts if max_experts is None else min(max_experts, e.n_experts)
                self._keys += [TensorKey(layer, x, _EXPERT_KIND[base]) for x in range(ne)]
            else:
                self._keys.append(TensorKey(layer, -1, base))
        self._keys.sort()

    def _slot_map(self, layer: int, base: str, t2, n_exp: int) -> Optional[np.ndarray]:
        n2 = int(t2.shape[2])
        mname = f"blk.{layer}.{base}2_map"
        mt = next((t for t in self._reader.tensors if t.name == mname), None)
        if mt is not None:
            m = np.asarray(mt.data, dtype=np.float32).reshape(-1)
            if m.size != n_exp:
                raise ValueError(f"{mname}: {m.size} voci, attese {n_exp}")
            sl = np.rint(m).astype(np.int64)
            sl[m < 0] = -1
            used = sl[sl >= 0]
            if used.size and (used.max() >= n2 or len(np.unique(used)) != used.size):
                raise ValueError(f"{mname}: slot fuori intervallo o ripetuti (n2={n2})")
            return sl
        if n2 < n_exp:      # prefisso (expert_count2)
            return np.where(np.arange(n_exp) < n2, np.arange(n_exp), -1)
        return None

    # ---------------------------------------------------------------- ABC
    def keys(self) -> List[TensorKey]:
        return list(self._keys)

    def entry(self, key: TensorKey) -> _Entry:
        return self._entries[(key.layer, _key_to_base(key))]

    def info(self, key: TensorKey) -> PlaneInfo:
        e = self.entry(key)
        return PlaneInfo(e.n_planes_of(key.expert), e.out, e.in_, e.fmt.rotated, e.fmt.group, e.fmt.packing)

    def raw_rows(self, key: TensorKey, plane: int) -> np.ndarray:
        """Byte grezzi del piano (0-based) per la chiave: uint8 [out, byte_riga] (vista mmap)."""
        e = self.entry(key)
        data = e.tensors[plane].data
        if e.n_experts:
            data = data[e.slot(plane, key.expert)]
        return data.reshape(e.out, -1)

    def load(self, key: TensorKey) -> TernaryTensor:
        e = self.entry(key)
        P, S = [], []
        for k in range(e.n_planes_of(key.expert)):
            p, s = decode_rows(self.raw_rows(key, k), e.fmt.type_id, e.in_)
            P.append(p)
            S.append(s)
        sign = torch.from_numpy(sign1024().copy()) if e.fmt.rotated else None
        return TernaryTensor(torch.from_numpy(np.stack(P)), torch.from_numpy(np.stack(S)),
                             e.fmt.rotated, sign=sign, packing=e.fmt.packing)

    def type_id(self, key: TensorKey) -> int:
        return self.entry(key).fmt.type_id


# --------------------------------------------------------------------------- teacher bf16 GGUF
def _to_f32(arr: np.ndarray, type_id: int) -> np.ndarray:
    if type_id == T_BF16:
        return (np.ascontiguousarray(arr).view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    if type_id == T_F16:
        return np.asarray(arr).view(np.float16).astype(np.float32)
    if type_id == T_F32:
        return np.asarray(arr, dtype=np.float32)
    raise ValueError(f"teacher: tipo {type_id} non supportato (serve bf16/f16/f32)")


class GGUFTeacherSource(TeacherSource):
    """Pesi del teacher da GGUF bf16 (anche a frammenti), base ORIGINALE, [out,in] fp32."""

    def __init__(self, paths: Sequence[str], gguf_py: Optional[str] = None):
        gguf = _import_gguf(gguf_py)
        self._t = {}
        self._readers = []
        for p in paths:
            r = gguf.GGUFReader(p, "r")
            self._readers.append(r)
            for t in r.tensors:
                self._t[t.name] = t

    def load(self, key: TensorKey) -> torch.Tensor:
        t = self._t[gguf_name(key.layer, _key_to_base(key))]
        shp = [int(x) for x in t.shape]
        data = t.data
        if key.expert >= 0:
            data = data[key.expert]
        w = _to_f32(data, int(t.tensor_type)).reshape(shp[1], shp[0])
        return torch.from_numpy(w.copy())


# --------------------------------------------------------------------------- Fisher proxy (imatrix)
class ImatrixFisherSource(FisherSource):
    """PROXY di Fisher da un imatrix GGUF (llama-imatrix): H_I = diag(in_sum2/counts)
    (solo diagonale), H_O = I / out. Serve al dry run sui dati veri; la Fisher
    K-FAC completa (H_I piena + H_O) si raccoglie con fisher.KFACCollector (GPU).
    out_of: chiave -> numero di righe (di norma lambda k: plane_source.info(k).out)."""

    """Voci MANCANTI nell'imatrix (es. lo strato MTP blk.40, mai attraversato da
    llama-imatrix): invece del KeyError si usa, nell'ordine,
      1. fallback_source (una FisherSource qualunque, es. la Fisher K-FAC di quel layer);
      2. fallback="uniform": H_I = c * I, con c = media della diagonale dello STESSO
         tensore nello strato piu' vicino che ce l'ha (stessa scala delle distorsioni
         degli altri strati, cosi' il knapsack globale resta confrontabile); se nessuno
         strato ce l'ha, la media dei tensori con la stessa larghezza d'ingresso; poi 1.
    fallback="error" ripristina il comportamento rigido. Ogni ripiego e' registrato una
    volta nel log e in self.missing (nome -> ripiego), che finisce nel piano.

    dense=False (default) restituisce i fattori in forma diagonale (diag_I/diag_O):
    niente matrici [in,in] e [out,out] costruite su CPU per ogni esperto."""

    def __init__(self, path: str, out_of: Callable[[TensorKey], int], gguf_py: Optional[str] = None,
                 floor_rel: float = 1e-6, fallback: str = "uniform",
                 fallback_source: Optional[FisherSource] = None, dense: bool = False,
                 log: Optional[Callable[[str], None]] = None,
                 in_of: Optional[Callable[[TensorKey], int]] = None):
        if fallback not in ("uniform", "error"):
            raise ValueError(f"fallback {fallback!r}: atteso 'uniform' o 'error'")
        gguf = _import_gguf(gguf_py)
        self._r = gguf.GGUFReader(path, "r")
        self._t = {t.name: t for t in self._r.tensors}
        self._out_of = out_of
        self._floor = floor_rel
        self._fallback = fallback
        self._fb_source = fallback_source
        self._dense = dense
        self._in_fn = in_of
        self._log = log or (lambda m: _LOG.warning(m))
        self._uniform_cache: Dict[Tuple[str, int], float] = {}
        self.missing: Dict[str, str] = {}

    # ---------------------------------------------------------------- voci presenti
    def _diag_from(self, name: str, expert: int) -> Tuple[np.ndarray, float]:
        s2 = np.asarray(self._t[name + ".in_sum2"].data, dtype=np.float64)
        cn = np.asarray(self._t[name + ".counts"].data, dtype=np.float64)
        if expert >= 0:
            s2, cn = s2.reshape(-1, s2.shape[-1])[expert], cn.reshape(-1)[expert]
        else:
            s2, cn = s2.reshape(-1), cn.reshape(-1)[0]
        diag = s2 / max(float(cn), 1.0)
        diag = np.maximum(diag, self._floor * max(float(diag.mean()), 1e-30))
        return diag, float(cn)

    def has(self, key: TensorKey) -> bool:
        name = gguf_name(key.layer, _key_to_base(key))
        return name + ".in_sum2" in self._t and name + ".counts" in self._t

    def _pack(self, diag: np.ndarray, out: int, n: int, source: str = "") -> FisherFactors:
        d = torch.from_numpy(diag.astype(np.float32))
        if self._dense:
            return FisherFactors(torch.diag(d), torch.eye(out, dtype=torch.float32) / out, n, source=source)
        return FisherFactors(None, None, n, diag_I=d, diag_O=torch.full((out,), 1.0 / out), source=source)

    # ---------------------------------------------------------------- ripieghi
    def _uniform_level(self, key: TensorKey, in_: int) -> Tuple[float, str]:
        base = _key_to_base(key)
        ck = (base, key.layer)
        if ck in self._uniform_cache:
            return self._uniform_cache[ck], "cache"
        layers = sorted({int(m.group(1)) for n in self._t
                         for m in [re.match(rf"^blk\.(\d+)\.{re.escape(base)}\.weight\.in_sum2$", n)] if m},
                        key=lambda L: (abs(L - key.layer), L))
        level, how = None, ""
        if layers:
            name = gguf_name(layers[0], base)
            s2 = np.asarray(self._t[name + ".in_sum2"].data, dtype=np.float64)
            cn = np.asarray(self._t[name + ".counts"].data, dtype=np.float64).reshape(-1)
            per = s2.reshape(len(cn), -1) / np.maximum(cn, 1.0)[:, None]
            level, how = float(per.mean()), f"uniforme da blk.{layers[0]}.{base}"
        else:
            vals = []
            for n, t in self._t.items():
                if n.endswith(".in_sum2"):
                    cn = np.asarray(self._t[n[: -len(".in_sum2")] + ".counts"].data, dtype=np.float64).reshape(-1)
                    s2 = np.asarray(t.data, dtype=np.float64)
                    if s2.size // max(len(cn), 1) == in_:
                        vals.append(float((s2.reshape(len(cn), -1) / np.maximum(cn, 1.0)[:, None]).mean()))
            if vals:
                level, how = float(np.mean(vals)), f"uniforme, media dei tensori con in={in_}"
            else:
                level, how = 1.0, "uniforme unitaria"
        self._uniform_cache[ck] = level
        return level, how

    def _fallback_load(self, key: TensorKey, name: str) -> FisherFactors:
        if self._fb_source is not None:
            try:
                ff = self._fb_source.load(key)
                ff.source = "fallback_source"
                self._note(key, name, "fallback_source")
                return ff
            except (KeyError, FileNotFoundError, OSError) as e:  # anche il ripiego non ce l'ha
                self._note(key, name, f"fallback_source assente ({type(e).__name__})")
        if self._fallback == "error":
            raise KeyError(f"{name}.in_sum2 assente nell'imatrix (fallback='error')")
        out = int(self._out_of(key))
        in_ = self._in_of(key)
        level, how = self._uniform_level(key, in_)
        self._note(key, name, how)
        return self._pack(np.full(in_, max(level, 1e-30)), out, 0, source=how)

    def _in_of(self, key: TensorKey) -> int:
        if self._in_fn is None:
            raise KeyError(f"{key.name()}: larghezza d'ingresso ignota per il ripiego uniforme (passare in_of)")
        return int(self._in_fn(key))

    def _note(self, key: TensorKey, name: str, how: str):
        tag = f"blk.{key.layer}.{_key_to_base(key)}" if key.layer >= 0 else _key_to_base(key)
        if tag not in self.missing:
            self.missing[tag] = how
            self._log(f"[imatrix] {name}: voce assente -> {how}")

    # ---------------------------------------------------------------- ABC
    def load(self, key: TensorKey) -> FisherFactors:
        name = gguf_name(key.layer, _key_to_base(key))
        if not self.has(key):
            return self._fallback_load(key, name)
        diag, cn = self._diag_from(name, key.expert)
        return self._pack(diag, int(self._out_of(key)), int(cn))
