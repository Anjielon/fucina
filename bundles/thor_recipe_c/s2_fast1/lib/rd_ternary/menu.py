"""Menu di opzioni per ogni tensore: (byte esatti, distorsione K-FAC).

Opzioni:
  cur      : piani attuali (scale eventualmente rifittate: gratis in byte)
  +1       : un piano PIENO in piu', fittato sul residuo nella metrica di Fisher
  +sNN     : un piano SPARSO in piu' a densita' NN% (supporto = posizioni col
             massimo peso di Fisher G_ii * A_jj * R_ij^2), codifica a indici locali
  -1       : un piano in meno (solo se K >= min_planes_to_drop), con rifit delle scale
  g64      : (cfg.allow_g64) stessi piani, scale ogni 64 pesi (tipi TQ1_0_G64 / TQ1_0R_1024_G64 del
             ramo thor-llama cloud/scala-g64): rifit esatto delle scale a gruppo 64, trit invariati
  +1g64    : (cfg.allow_g64 e allow_plus_full) un piano pieno in piu' fittato a gruppo 64, poi
             rifit di TUTTE le scale a gruppo 64
  Le opzioni g64 cambiano il TIPO del tensore GGUF: per gli esperti valgono per tutto il tensore
  (strato, proiezione) — l'allocatore le accoppia (allocazione.allocate_regimi).

Mai fit verso il bf16 grezzo: il nuovo piano approssima R = W_teacher - W_cur
nella metrica di A, poi le scale di TUTTI i piani si rifittano insieme.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch

from .adattamento import (block_diag_size, damp, damp_diag, diag_blocks, diag_if_diagonal, error_of,
                          gptq_factor, gptq_factor_blocks, gptq_ternary, gptq_ternary_diag,
                          kfac_distortion_struct, refit_scales, refit_scales_blocks, refit_scales_diag)
from .formato import (G64, GROUP, LOCAL8_MAX_PER_GROUP, TensorKey, full_plane_bytes, packing_for_scales,
                      rotate_fisher_in, rotate_weight_in, sparse_plane_bytes)
from .sorgenti import FisherSource, PlaneSource, TeacherSource


@dataclass
class MenuConfig:
    densities: Sequence[float] = (0.10, 0.25, 0.50)
    packing: str = "tq1"
    sparse_coding: str = "local8"
    damp: float = 0.01
    refit_existing: bool = True      # rifit delle scale dei piani esistenti (0 byte)
    min_planes_to_drop: int = 2      # "-1" ammesso solo se K >= questo
    allow_plus_full: bool = True
    safeguard: bool = True           # un'opzione "+" non puo' peggiorare cur
    device: str = "cpu"
    batch: int = 32                  # tensori per lotto
    refit_chunk_rows: int = 128
    mm_dtype: torch.dtype = torch.float32
    max_batch_weights: Optional[int] = None  # tetto di out*in*B per lotto (RAM): None = solo `batch`
    fast_diag: bool = True           # fattori diagonali (proxy imatrix) -> percorsi O(o*i) equivalenti
    max_batch_bytes: Optional[int] = None    # tetto di memoria STIMATA per lotto (vedi per_tensor_bytes)
    extra_bytes_per_weight: int = 0          # R015: byte in piu' per peso (Taylor: pile g/f/g2 fp32 + temporanei fp64)
    allow_g64: bool = False                  # 6/10: opzioni "g64" e "+1g64" (scala ogni 64 pesi)

    def chunk_size(self, out: int, in_: int) -> int:
        b = self.batch
        if self.max_batch_weights is not None:
            b = min(b, self.max_batch_weights // (out * in_))
        if self.max_batch_bytes is not None:
            b = min(b, self.max_batch_bytes // (per_tensor_bytes(out, in_) + self.extra_bytes_per_weight * out * in_))
        return max(1, b)

    def option_names(self, n_planes: int) -> List[str]:
        names = ["cur"]
        if self.allow_plus_full:
            names.append("+1")
        names += [f"+s{round(d * 100):02d}" for d in self.densities]
        if n_planes >= self.min_planes_to_drop:
            names.append("-1")
        if self.allow_g64:
            names.append("g64")
            if self.allow_plus_full:
                names.append("+1g64")
        return names

    def density_of(self, name: str) -> float:
        return int(name[2:]) / 100.0


def per_tensor_bytes(out: int, in_: int, n_planes: int = 4, chunk_rows: int = 128, group: int = 128) -> int:
    """Stima PRUDENTE della memoria di picco per tensore in un lotto (fp32):
    ~16 copie [o,i] (teacher, D, R, cloni di GPTQ, piani float), 4 matrici [i,i]
    (A, A smorzata, U, temporaneo di Cholesky), una [o,o] (G densa) e, due volte,
    il blocco Y del rifit denso [K*ng, righe, i] (gruppo 128 = caso peggiore)."""
    y = 2 * n_planes * (in_ // group) * min(chunk_rows, out) * in_
    return 4 * (16 * out * in_ + 4 * in_ * in_ + out * out + y) + n_planes * out * in_


@dataclass
class Option:
    name: str
    bytes: int
    distortion: float
    nnz: int = 0
    extra: Dict = field(default_factory=dict)


@dataclass
class TensorMenu:
    key: TensorKey
    n_planes: int
    shape: tuple
    options: List[Option]

    def get(self, name: str) -> Option:
        return next(o for o in self.options if o.name == name)


@dataclass
class FitResult:
    planes: torch.Tensor                  # int8 [B,K',o,i] piani pieni
    scales: torch.Tensor                  # fp16 [B,K',o,ng]
    D: torch.Tensor                       # errore rispetto al teacher [B,o,i]
    sparse_plane: Optional[torch.Tensor] = None   # int8 [B,o,i]
    sparse_scales: Optional[torch.Tensor] = None  # fp16 [B,o,ng]
    dropped: Optional[torch.Tensor] = None        # indice del piano tolto [B]
    d: Optional[torch.Tensor] = None              # distorsione K-FAC [B] (calcolata una volta)


class TensorBatch:
    """Lotto di tensori con stessa forma, stesso K e stessa rotazione, gia' nella
    base memorizzata, con le quantita' derivate condivise da tutte le opzioni."""

    _TENSORS = ("w_t", "planes", "scales0", "A", "G", "Ad", "U", "cdiag", "a_diag", "g_diag",
                "A_blocks", "Ad_blocks", "d_orig", "scales_cur", "d_cur", "D_cur")

    def __init__(self, keys, w_t, planes, scales, A, G, rotated: bool, cfg: MenuConfig,
                 packing: Optional[str] = None, a_diag: Optional[torch.Tensor] = None,
                 g_diag: Optional[torch.Tensor] = None):
        """A/G dense [B,i,i]/[B,o,o] oppure (a_diag/g_diag) [B,i]/[B,o] in forma diagonale.
        Con cfg.fast_diag la forma diagonale viene anche RICONOSCIUTA nelle dense;
        senza, tutto torna denso (percorso di riferimento)."""
        self.keys, self.w_t, self.planes, self.scales0 = list(keys), w_t, planes, scales
        self.rotated, self.cfg = rotated, cfg
        self.group = int(planes.shape[-1]) // int(scales.shape[-1])   # 256, o 128 (TQ1_0R_1024)
        self.packing = packing or cfg.packing
        if cfg.fast_diag:
            a_diag = a_diag if a_diag is not None else diag_if_diagonal(A)
            g_diag = g_diag if g_diag is not None else diag_if_diagonal(G)
        else:
            A = A if A is not None else torch.diag_embed(a_diag.float())
            G = G if G is not None else torch.diag_embed(g_diag.float())
            a_diag = g_diag = None
        self.a_diag, self.g_diag = a_diag, g_diag
        self.G = None if g_diag is not None else G
        self.A_blocks = self.Ad_blocks = None
        self.block: Optional[int] = None
        if a_diag is not None:          # Hessiana diagonale: niente Cholesky ne' U
            self.A = self.Ad = self.U = None
            self.cdiag = damp_diag(a_diag, cfg.damp)
        else:
            Ad = damp(A, cfg.damp)
            self.cdiag = Ad.diagonal(dim1=-2, dim2=-1).contiguous()
            self.block = block_diag_size(A, self.group) if cfg.fast_diag else None
            if self.block:              # diagonale a blocchi (H_I diagonale ruotata): per blocco
                self.A_blocks, self.Ad_blocks = diag_blocks(A, self.block), diag_blocks(Ad, self.block)
                self.U = dedup_apply(self.Ad_blocks, gptq_factor_blocks)
                self.A = self.Ad = None
            else:
                self.A, self.Ad = A, Ad
                self.U = dedup_apply(Ad, gptq_factor)
        d0 = self.dist(error_of(planes, scales, w_t))
        self.d_orig = d0
        if cfg.refit_existing:
            sc = self.refit(planes, w_t)
            d1 = self.dist(error_of(planes, sc, w_t))
            better = (d1 < d0).view(-1, 1, 1, 1)
            self.scales_cur = torch.where(better, sc, scales)
            self.d_cur = torch.minimum(d0, d1)
        else:
            self.scales_cur, self.d_cur = scales, d0
        self.D_cur = error_of(planes, self.scales_cur, w_t)

    # ------------------------------------------------------------------ operatori (dispatch)
    def dist(self, D: torch.Tensor) -> torch.Tensor:
        """Distorsione K-FAC [B] col percorso adatto alla struttura dei fattori."""
        return kfac_distortion_struct(D, self.A, self.G, self.a_diag, self.g_diag, self.cfg.mm_dtype,
                                      a_blocks=self.A_blocks)

    def refit(self, planes: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.a_diag is not None:
            return refit_scales_diag(planes, target, self.cdiag, self.cfg.refit_chunk_rows, self.group)
        if self.Ad_blocks is not None:
            return refit_scales_blocks(planes, target, self.Ad_blocks, self.cfg.refit_chunk_rows,
                                       self.cfg.mm_dtype, self.group)
        return refit_scales(planes, target, self.Ad, self.cfg.refit_chunk_rows, self.cfg.mm_dtype, self.group)

    def fit_plane(self, R: torch.Tensor, mask: Optional[torch.Tensor] = None, group: Optional[int] = None):
        g = group or self.group
        if self.a_diag is not None:
            return gptq_ternary_diag(R, self.cdiag, mask, group=g)
        return gptq_ternary(R, self.U, self.cdiag, mask, group=g)

    def refit_group(self, planes: torch.Tensor, target: torch.Tensor, group: int) -> torch.Tensor:
        """Come refit, a un gruppo di scala diverso da quello del tensore (opzioni g64)."""
        if self.a_diag is not None:
            return refit_scales_diag(planes, target, self.cdiag, self.cfg.refit_chunk_rows, group)
        if self.Ad_blocks is not None:
            return refit_scales_blocks(planes, target, self.Ad_blocks, self.cfg.refit_chunk_rows,
                                       self.cfg.mm_dtype, group)
        return refit_scales(planes, target, self.Ad, self.cfg.refit_chunk_rows, self.cfg.mm_dtype, group)

    def scales_to_group(self, scales: torch.Tensor, group: int) -> torch.Tensor:
        """Scale [...,ng] al gruppo `group` (sottomultiplo): ripetute, ricostruzione IDENTICA."""
        return scales.repeat_interleave(self.group // group, dim=-1)

    def a_diagonal(self) -> torch.Tensor:
        if self.a_diag is not None:
            return self.a_diag
        if self.A_blocks is not None:
            return self.A_blocks.diagonal(dim1=-2, dim2=-1).reshape(self.A_blocks.shape[0], -1)
        return self.A.diagonal(dim1=-2, dim2=-1)

    def g_diagonal(self) -> torch.Tensor:
        return self.g_diag if self.g_diag is not None else self.G.diagonal(dim1=-2, dim2=-1)

    @property
    def size(self) -> int:
        return len(self.keys)

    def subset(self, idx: List[int]) -> "TensorBatch":
        t = object.__new__(TensorBatch)
        ix = torch.tensor(idx, device=self.w_t.device)
        t.keys = [self.keys[j] for j in idx]
        for name in self._TENSORS:
            v = getattr(self, name)
            setattr(t, name, None if v is None else v.index_select(0, ix))
        t.rotated, t.cfg, t.group, t.packing, t.block = self.rotated, self.cfg, self.group, self.packing, self.block
        return t


def dedup_apply(x: torch.Tensor, fn) -> torch.Tensor:
    """fn(x) calcolata UNA volta per ogni elemento distinto del lotto (confronto col
    precedente). gate e up dello stesso esperto vedono lo stesso ingresso (stessa H_I,
    bit per bit) e nei lotti stanno uno accanto all'altro (chiavi ordinate): la
    Cholesky da i^3 si dimezza. Risultato identico a fn(x) (fn agisce per elemento)."""
    B = x.shape[0]
    if B < 2:
        return fn(x)
    same = (x[1:] == x[:-1]).flatten(1).all(1)              # uguale al precedente?
    if not bool(same.any()):
        return fn(x)
    first = torch.ones(B, dtype=torch.bool, device=x.device)
    first[1:] = ~same
    rep = torch.cumsum(first.long(), 0) - 1                  # indice del rappresentante
    return fn(x[first]).index_select(0, rep)


def gptq_factor_dedup(ad: torch.Tensor) -> torch.Tensor:
    return dedup_apply(ad, gptq_factor)


def _stack_to(xs: List[torch.Tensor], dev: torch.device, dtype=None) -> torch.Tensor:
    """Impila su CPU e trasferisce UNA volta (un trasferimento per lotto, non per tensore)."""
    t = torch.stack([x.cpu() if x.device.type != "cpu" else x for x in xs])
    if dtype is not None:
        t = t.to(dtype)
    return t.to(dev, non_blocking=False)


@dataclass
class LoadedBatch:
    """Il lotto letto dalle sorgenti, ancora su CPU (lo si puo' preparare in un thread
    mentre il dispositivo lavora sul lotto precedente)."""
    keys: List[TensorKey]
    tts: list
    ffs: list
    teach: List[torch.Tensor]


def load_batch_cpu(keys: Sequence[TensorKey], ps: PlaneSource, ts: TeacherSource,
                   fs: FisherSource) -> LoadedBatch:
    keys = list(keys)
    return LoadedBatch(keys, [ps.load(k) for k in keys], [fs.load(k) for k in keys],
                       [ts.load(k).float() for k in keys])


def prepare_batch(keys: Sequence[TensorKey], ps: PlaneSource, ts: TeacherSource, fs: FisherSource,
                  cfg: MenuConfig, loaded: Optional[LoadedBatch] = None) -> TensorBatch:
    """Carica il lotto (CPU), lo trasferisce sul dispositivo e ruota TUTTO il lotto in
    un colpo (base memorizzata: W S H, H S H_I S H). I fattori diagonali restano
    vettori finche' possono: una H_I diagonale ruotata diventa densa (si costruisce
    sul dispositivo), H_O diagonale resta diagonale. loaded: lotto gia' letto."""
    dev = torch.device(cfg.device)
    lb = loaded if loaded is not None else load_batch_cpu(keys, ps, ts, fs)
    tts, ffs = lb.tts, lb.ffs
    rot = {tt.rotated for tt in tts}
    pk = {(tt.group, tt.packing) for tt in tts}
    assert len(rot) == 1, "un lotto deve avere rotazione omogenea"
    assert len(pk) == 1, "un lotto deve avere gruppo di scala e packing omogenei"
    rotated = rot.pop()
    sign = tts[0].sign
    if rotated and sign is not None:
        assert all(tt.sign is not None and torch.equal(tt.sign, sign) for tt in tts), "segni S diversi nel lotto"
    W = _stack_to(lb.teach, dev, torch.float32)
    P = _stack_to([tt.planes for tt in tts], dev)
    S = _stack_to([tt.scales for tt in tts], dev)
    a_diag = g_diag = A = G = None
    if all(ff.H_I is None for ff in ffs):
        a_diag = _stack_to([ff.diag_I.float() for ff in ffs], dev)
    else:
        A = _stack_to([ff.dense_in().float() for ff in ffs], dev)
    if all(ff.H_O is None for ff in ffs):
        g_diag = _stack_to([ff.diag_O.float() for ff in ffs], dev)
    else:
        G = _stack_to([ff.dense_out().float() for ff in ffs], dev)
    if rotated:  # tutto nella base memorizzata (H(S w) se il tensore porta i segni)
        W = rotate_weight_in(W, sign=sign)
        if a_diag is not None:
            A, a_diag = torch.diag_embed(a_diag), None
        A = rotate_fisher_in(A, sign=sign)
    return TensorBatch(keys, W, P, S, A, G, rotated, cfg, packing=pk.pop()[1], a_diag=a_diag, g_diag=g_diag)


def sparse_support(R: torch.Tensor, a_diag: torch.Tensor, g_diag: torch.Tensor, density: float,
                   group: int = GROUP) -> torch.Tensor:
    """Maschera con esattamente round(density*o*i) posizioni (tetto 255 per gruppo),
    scelte per salienza di Fisher diagonale  G_ii * A_jj * R_ij^2."""
    B, o, i = R.shape
    score = g_diag[:, :, None].clamp_min(0) * a_diag[:, None, :].clamp_min(0) * R * R
    k = max(1, int(round(density * o * i)))
    idx = torch.topk(score.reshape(B, -1), k, dim=1).indices
    mask = torch.zeros(B, o * i, dtype=torch.bool, device=R.device)
    mask.scatter_(1, idx, True)
    mask = mask.view(B, o, i)
    # tetto per gruppo (contatore uint8): si tengono le 255 piu' salienti
    mg = mask.view(B, o, i // group, group)
    if bool((mg.sum(-1) > LOCAL8_MAX_PER_GROUP).any()):
        sg = torch.where(mg, score.view(B, o, i // group, group), torch.full_like(mg, -1.0, dtype=score.dtype))
        rank = sg.argsort(-1, descending=True).argsort(-1)
        mg = mg & (rank < LOCAL8_MAX_PER_GROUP)
        mask = mg.view(B, o, i)
    return mask


def _with_new_plane(tb: TensorBatch, T: torch.Tensor, S: torch.Tensor):
    """Aggiunge il piano T e rifitta le scale; ritorna (planes, scales) completi."""
    allp = torch.cat([tb.planes, T[:, None]], dim=1)
    if tb.cfg.refit_existing:
        sc = tb.refit(allp, tb.w_t)
    else:  # scale esistenti fisse: si rifitta solo il nuovo piano sul residuo
        s_new = tb.refit(T[:, None], -tb.D_cur)
        sc = torch.cat([tb.scales_cur, s_new], dim=1)
    return allp, sc


@torch.no_grad()
def apply_option(tb: TensorBatch, name: str) -> FitResult:
    """Calcolo DETERMINISTICO di un'opzione (lo stesso codice serve al menu e alla
    materializzazione, cosi' il piano esportato si ricostruisce identico)."""
    cfg = tb.cfg
    if name == "cur":
        return FitResult(tb.planes, tb.scales_cur, tb.D_cur, d=tb.d_cur)
    R = -tb.D_cur
    if name == "-1":
        K = tb.planes.shape[1]
        best_d, best = None, None
        for k in range(K):
            keep = [j for j in range(K) if j != k]
            p = tb.planes[:, keep]
            sc = tb.refit(p, tb.w_t)
            d = tb.dist(error_of(p, sc, tb.w_t))
            if best is None:
                best_d, best = d, (p, sc, torch.full_like(d, k, dtype=torch.long))
            else:
                m = d < best_d
                mm = m.view(-1, 1, 1, 1)
                best = (torch.where(mm, p, best[0]), torch.where(mm, sc, best[1]),
                        torch.where(m, torch.full_like(best[2], k), best[2]))
                best_d = torch.minimum(d, best_d)
        p, sc, dropped = best
        return FitResult(p, sc, error_of(p, sc, tb.w_t), dropped=dropped, d=best_d)
    if name == "+1":
        T, S = tb.fit_plane(R)
        allp, sc = _with_new_plane(tb, T, S)
        D = error_of(allp, sc, tb.w_t)
        d = tb.dist(D)
        if cfg.safeguard:
            worse = d > tb.d_cur
            if bool(worse.any()):  # piano memorizzato ma a scala zero: d = d_cur
                w4 = worse.view(-1, 1, 1, 1)
                sc_fb = torch.cat([tb.scales_cur, torch.zeros_like(sc[:, -1:])], dim=1)
                sc = torch.where(w4, sc_fb, sc)
                D = error_of(allp, sc, tb.w_t)
                d = torch.where(worse, tb.d_cur, d)
        return FitResult(allp, sc, D, d=d)
    if name.startswith("+s"):
        mask = sparse_support(R, tb.a_diagonal(), tb.g_diagonal(),
                              cfg.density_of(name), tb.group)
        T, S = tb.fit_plane(R, mask)
        allp, sc = _with_new_plane(tb, T, S)
        D = error_of(allp, sc, tb.w_t)
        d = tb.dist(D)
        if cfg.safeguard:
            worse = d > tb.d_cur
            if bool(worse.any()):  # piano sparso vuoto: costa solo i contatori
                w3, w4 = worse.view(-1, 1, 1), worse.view(-1, 1, 1, 1)
                T = torch.where(w3, torch.zeros_like(T), T)
                allp = torch.cat([tb.planes, T[:, None]], dim=1)
                sc_fb = torch.cat([tb.scales_cur, torch.zeros_like(sc[:, -1:])], dim=1)
                sc = torch.where(w4, sc_fb, sc)
                D = error_of(allp, sc, tb.w_t)
                d = torch.where(worse, tb.d_cur, d)
        # le scale dei gruppi vuoti non vengono memorizzate: azzerarle non cambia nulla
        cnt = (T != 0).view(*T.shape[:2], -1, tb.group).sum(-1)
        sp_sc = torch.where(cnt > 0, sc[:, -1], torch.zeros_like(sc[:, -1]))
        return FitResult(tb.planes, sc[:, :-1], D, sparse_plane=T, sparse_scales=sp_sc, d=d)
    if name in ("g64", "+1g64"):
        return _apply_g64(tb, name)
    raise ValueError(name)


def _apply_g64(tb: TensorBatch, name: str) -> FitResult:
    """Opzioni a scala 64. Punto di partenza (e salvaguardia) = scale correnti RIPETUTE a 64:
    stessa ricostruzione di "cur" nel formato g64, quindi g64 non peggiora mai cur e +1g64 mai g64."""
    g = G64
    if tb.group % g:
        raise ValueError(f"gruppo {tb.group} non multiplo di {g}")
    sc_rep = tb.scales_to_group(tb.scales_cur, g)
    sc = tb.refit_group(tb.planes, tb.w_t, g)
    D = error_of(tb.planes, sc, tb.w_t)
    d = tb.dist(D)
    worse = d > tb.d_cur
    if bool(worse.any()):
        sc = torch.where(worse.view(-1, 1, 1, 1), sc_rep, sc)
        D = error_of(tb.planes, sc, tb.w_t)
        d = torch.where(worse, tb.d_cur, d)
    if name == "g64":
        return FitResult(tb.planes, sc, D, d=d)
    T, _ = tb.fit_plane(-D, group=g)
    allp = torch.cat([tb.planes, T[:, None]], dim=1)
    sc2 = tb.refit_group(allp, tb.w_t, g)
    D2 = error_of(allp, sc2, tb.w_t)
    d2 = tb.dist(D2)
    if tb.cfg.safeguard:
        worse = d2 > d
        if bool(worse.any()):   # piano memorizzato a scala zero: torna g64
            sc_fb = torch.cat([sc, torch.zeros_like(sc2[:, -1:])], dim=1)
            sc2 = torch.where(worse.view(-1, 1, 1, 1), sc_fb, sc2)
            D2 = error_of(allp, sc2, tb.w_t)
            d2 = torch.where(worse, d, d2)
    return FitResult(allp, sc2, D2, d=d2)


def option_bytes(fr: FitResult, n_planes_full: int, out: int, in_: int, cfg: MenuConfig,
                 packing: Optional[str] = None, group: int = GROUP) -> torch.Tensor:
    """Byte totali del tensore sotto l'opzione (piani pieni + eventuale sparso).
    packing/group: quelli del tensore (TensorBatch.packing/.group); default dal cfg."""
    pk = packing_for_scales(packing or cfg.packing, in_, int(fr.scales.shape[-1]))   # g64: packing a scala 64
    full = full_plane_bytes(out, in_, pk) * fr.planes.shape[1]
    b = torch.full((fr.D.shape[0],), full, dtype=torch.long, device=fr.D.device)
    if fr.sparse_plane is not None:
        b = b + sparse_plane_bytes(fr.sparse_plane, cfg.sparse_coding, group).to(b.device)
    return b


def iter_batches(groups: Dict[tuple, List[TensorKey]], cfg: MenuConfig, ps: PlaneSource, ts: TeacherSource,
                 fs: FisherSource, prefetch: bool = True):
    """Genera (chiave_gruppo, chunk, TensorBatch). Con prefetch la lettura dal disco e la
    decodifica del lotto successivo (CPU, numpy) si sovrappongono al calcolo del lotto
    corrente sul dispositivo (gli operatori torch rilasciano il GIL)."""
    chunks = []
    for gk, ks in groups.items():
        o, i = gk[1], gk[2]
        bs = cfg.chunk_size(o, i)
        chunks += [(gk, ks[b0: b0 + bs]) for b0 in range(0, len(ks), bs)]
    if not prefetch or len(chunks) < 2:
        for gk, chunk in chunks:
            yield gk, chunk, prepare_batch(chunk, ps, ts, fs, cfg)
        return
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(load_batch_cpu, chunks[0][1], ps, ts, fs)
        for n, (gk, chunk) in enumerate(chunks):
            loaded = fut.result()
            if n + 1 < len(chunks):
                fut = pool.submit(load_batch_cpu, chunks[n + 1][1], ps, ts, fs)
            yield gk, chunk, prepare_batch(chunk, ps, ts, fs, cfg, loaded=loaded)


def build_menus(ps: PlaneSource, ts: TeacherSource, fs: FisherSource, cfg: MenuConfig,
                keys: Optional[Sequence[TensorKey]] = None, log=None, prefetch: bool = True) -> List[TensorMenu]:
    keys = list(keys) if keys is not None else ps.keys()
    groups: Dict[tuple, List[TensorKey]] = defaultdict(list)
    for k in keys:
        inf = ps.info(k)
        groups[(inf.n_planes, inf.out, inf.in_, inf.rotated, inf.group, inf.packing)].append(k)
    menus: Dict[TensorKey, TensorMenu] = {}
    for (K, o, i, rot, _grp, _pk), chunk, tb in iter_batches(groups, cfg, ps, ts, fs, prefetch):
        per = {k: [] for k in chunk}
        for name in cfg.option_names(K):
            fr = apply_option(tb, name)
            d = fr.d.cpu()
            by = option_bytes(fr, K, o, i, cfg, tb.packing, tb.group).cpu()
            nnz = (fr.sparse_plane != 0).sum((-1, -2)).cpu() if fr.sparse_plane is not None else None
            for j, k in enumerate(chunk):
                extra = {"d_orig_scales": float(tb.d_orig[j])} if name == "cur" else {}
                if fr.dropped is not None:
                    extra["dropped_plane"] = int(fr.dropped[j])
                per[k].append(Option(name, int(by[j]), float(d[j]),
                                     int(nnz[j]) if nnz is not None else 0, extra))
        for k in chunk:
            menus[k] = TensorMenu(k, K, (o, i), per[k])
        if log:
            log(f"lotto K={K} {o}x{i} rot={rot}: {len(chunk)} tensori")
        del tb
    return [menus[k] for k in keys]
