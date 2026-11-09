"""Interfacce astratte per le tre sorgenti di dati (DIP: il resto del codice
dipende solo da queste ABC) + implementazioni in memoria (test) e safetensors.

  * PlaneSource   : piani ternari del modello quantizzato corrente
  * TeacherSource : pesi bf16 del teacher (base ORIGINALE, non ruotata)
  * FisherSource  : fattori di Kronecker H_I (ingresso) e H_O (gradiente d'uscita)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch

from .formato import GROUP, TensorKey, TernaryTensor


@dataclass
class PlaneInfo:
    n_planes: int
    out: int
    in_: int
    rotated: bool
    group: int = GROUP              # pesi per scala (256; 128 per TQ1_0R_1024)
    packing: Optional[str] = None   # None = MenuConfig.packing


@dataclass
class FisherFactors:
    """Fattori K-FAC gia' normalizzati (vedi fisher.py):
       H_I = media di x x^T sui token instradati all'esperto        [in, in]
       H_O = somma di g g^T diviso il numero TOTALE di token        [out, out]
    cosi' 0.5 tr(H_O D H_I D^T) ~ aumento medio di KL per token.

    Forma diagonale (proxy imatrix): H_I/H_O possono essere None e i fattori stare
    in diag_I [in] / diag_O [out]; dense_in()/dense_out() li materializzano (sul
    dispositivo richiesto) solo quando servono davvero (es. ingresso ruotato)."""
    H_I: Optional[torch.Tensor] = None
    H_O: Optional[torch.Tensor] = None
    n_tokens: int = 0
    diag_I: Optional[torch.Tensor] = None
    diag_O: Optional[torch.Tensor] = None
    source: str = ""                 # "" = sorgente primaria; altrimenti il ripiego usato

    def __post_init__(self):
        if self.H_I is None and self.diag_I is None:
            raise ValueError("FisherFactors: serve H_I oppure diag_I")
        if self.H_O is None and self.diag_O is None:
            raise ValueError("FisherFactors: serve H_O oppure diag_O")

    def dense_in(self, device=None) -> torch.Tensor:
        if self.H_I is not None:
            return self.H_I.to(device) if device is not None else self.H_I
        return torch.diag(self.diag_I.to(device).float())

    def dense_out(self, device=None) -> torch.Tensor:
        if self.H_O is not None:
            return self.H_O.to(device) if device is not None else self.H_O
        return torch.diag(self.diag_O.to(device).float())


# --------------------------------------------------------------------------- ABC
class PlaneSource(ABC):
    @abstractmethod
    def keys(self) -> List[TensorKey]: ...

    @abstractmethod
    def info(self, key: TensorKey) -> PlaneInfo: ...

    @abstractmethod
    def load(self, key: TensorKey) -> TernaryTensor: ...


class TeacherSource(ABC):
    @abstractmethod
    def load(self, key: TensorKey) -> torch.Tensor:
        """Peso [out, in] nella base ORIGINALE (la rotazione la applica il chiamante)."""


class FisherSource(ABC):
    @abstractmethod
    def load(self, key: TensorKey) -> FisherFactors: ...


# --------------------------------------------------------------------------- in memoria
class DictPlaneSource(PlaneSource):
    def __init__(self, data: Dict[TensorKey, TernaryTensor]):
        self._d = data

    def keys(self):
        return sorted(self._d)

    def info(self, key):
        t = self._d[key]
        k, o, i = t.planes.shape
        return PlaneInfo(k, o, i, t.rotated, t.group, t.packing)

    def load(self, key):
        return self._d[key]


class DictTeacherSource(TeacherSource):
    def __init__(self, data: Dict[TensorKey, torch.Tensor]):
        self._d = data

    def load(self, key):
        return self._d[key]


class DictFisherSource(FisherSource):
    def __init__(self, data: Dict[TensorKey, FisherFactors]):
        self._d = data

    def load(self, key):
        return self._d[key]


# --------------------------------------------------------------------------- safetensors
@dataclass
class HFName:
    """Dove trovare un tensore nel checkpoint HF.
    index: indice dell'esperto se il tensore e' fuso [E, ...]; rows: fetta di righe
    (per gate_up fusi); transpose: True se il layout e' [in, out]."""
    tensor: str
    index: Optional[int] = None
    rows: Optional[Tuple[int, int]] = None
    transpose: bool = False


def qwen_moe_names(key: TensorKey) -> HFName:
    """Esempio di mappatura per checkpoint stile Qwen3-MoE (esperti separati)."""
    if key.expert >= 0:
        return HFName(f"model.layers.{key.layer}.mlp.experts.{key.expert}.{key.kind}_proj.weight")
    return HFName(f"model.layers.{key.layer}.{key.kind}.weight")


class SafetensorsTeacher(TeacherSource):
    def __init__(self, files: Iterable[str], name_fn: Callable[[TensorKey], HFName] = qwen_moe_names):
        from safetensors import safe_open
        self._open = safe_open
        self._name_fn = name_fn
        self._where: Dict[str, str] = {}
        for f in files:
            with safe_open(f, "pt") as h:
                for n in h.keys():
                    self._where[n] = f

    def load(self, key):
        spec = self._name_fn(key)
        with self._open(self._where[spec.tensor], "pt") as h:
            sl = h.get_slice(spec.tensor)
            t = sl[spec.index] if spec.index is not None else sl[:]
        if spec.transpose:
            t = t.transpose(-1, -2)
        if spec.rows is not None:
            t = t[spec.rows[0]: spec.rows[1]]
        return t.contiguous()


class SafetensorsPlaneSource(PlaneSource):
    """File per layer `planes_L{layer:03d}.safetensors` con, per ogni tensore,
    `{nome}.planes` int8 [K,o,i], `{nome}.scales` fp16 [K,o,ng], `{nome}.rotated` uint8 [1].
    (Adattatore di riferimento: un lettore GGUF reale implementa la stessa ABC.)"""

    def __init__(self, files: Iterable[str]):
        from safetensors import safe_open
        self._open = safe_open
        self._where: Dict[TensorKey, str] = {}
        for f in files:
            with safe_open(f, "pt") as h:
                for n in h.keys():
                    if n.endswith(".planes"):
                        self._where[TensorKey.parse(n[: -len(".planes")])] = f

    def keys(self):
        return sorted(self._where)

    def info(self, key):
        with self._open(self._where[key], "pt") as h:
            k, o, i = h.get_slice(f"{key.name()}.planes").get_shape()
            rot = bool(h.get_tensor(f"{key.name()}.rotated")[0])
        return PlaneInfo(k, o, i, rot)

    def load(self, key):
        n = key.name()
        with self._open(self._where[key], "pt") as h:
            return TernaryTensor(h.get_tensor(f"{n}.planes"), h.get_tensor(f"{n}.scales"),
                                 bool(h.get_tensor(f"{n}.rotated")[0]))


def default_in_share(key: TensorKey) -> Tuple[str, bool]:
    """gate e up vedono lo stesso ingresso: una sola H_I (accumulata dal gate)."""
    if key.kind in ("gate", "up"):
        return TensorKey(key.layer, key.expert, "gate_up").name(), key.kind == "gate"
    return key.name(), True


class SafetensorsFisherSource(FisherSource):
    """File per layer `fisher_L{layer:03d}.safetensors` scritti da KFACCollector.save()."""

    def __init__(self, directory: str, in_share: Callable = default_in_share):
        from safetensors import safe_open
        self._open, self._dir, self._share = safe_open, directory, in_share

    def load(self, key):
        f = f"{self._dir}/fisher_L{key.layer:03d}.safetensors"
        in_name, _ = self._share(key)
        with self._open(f, "pt") as h:
            return FisherFactors(h.get_tensor(f"{in_name}.H_I").float(),
                                 h.get_tensor(f"{key.name()}.H_O").float(),
                                 int(h.get_tensor(f"{in_name}.n")[0]))
