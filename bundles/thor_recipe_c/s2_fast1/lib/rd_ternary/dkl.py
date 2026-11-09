"""dKL MISURATA sul modello intero, per gruppo di esperti, con UN passaggio avanti+indietro (R001).

Perche': il proxy imatrix ha predetto male il danno (ALLOC1 73,79 vs 75,00; RD1 72,75 vs
74,55, p = 0,005). Qui il danno non e' piu' un proxy locale: e' lo sviluppo di Taylor
della KL(maestro || studente) calcolata sul MODELLO INTERO, nella base memorizzata dei pesi
(la stessa in cui lavora rd_ternary: per i tensori ruotati H(S w)):

    dKL(opzione) ~= g . dW  +  1/2 sum_ij F_ij dW_ij^2          dW = W_opzione - W_attuale

  g  = gradiente della KL media per token rispetto ai pesi DEQUANTIZZATI dello studente
       (un passaggio avanti+indietro sul modello intero; `dkl_raccogli.py`);
  F  = Fisher DIAGONALE per token. Con token campionati dallo studente
       (`fisher="campionato"`) e' il vero Fisher = Hessiana della KL ai logit (Gauss-Newton),
       non una Fisher empirica; l'alternativa a un solo passaggio e' `fisher="empirica"`.
  dW = esattamente cio' che il codificatore di rd_ternary scrive: `apply_option` ->
       fr.D (errore rispetto al maestro, base memorizzata) meno l'errore dei piani ATTUALI
       con le scale del file. Anche "cur" ha il suo dW (le scale rifittate): nel TSV si
       scrive, e misurato.py lo sottrae come linea di base (misure appaiate).

Il primo ordine e' ESATTAMENTE additivo fra gruppi; il secondo ordine diagonale ignora i
termini misti fra pesi (dentro il gruppo e fra gruppi): e' la stima, non la verita'. Per
questo il file porta anche `se` (errore standard dalla differenza fra le due meta' delle righe, se g2 e' fornito) e il
protocollo di verifica a ciclo chiuso e' in COME_LANCIARE.md.

Uscita: dkl.tsv  =  gruppo <TAB> opzione <TAB> delta_kl <TAB> se <TAB> primo <TAB> secondo
(`unisci.py --delta-kl dkl.tsv` legge le prime tre colonne; le altre sono ignorate).
Gruppi: `tensore` (L12.E37.gate), `esperto` (L12.E37: gate+up+down; l'opzione deve esistere
in tutti e tre), `blocco:N` (L12.E0-31.down: N esperti consecutivi, un tipo), `strato`
(L12.E*.gate ...). I densi sono sempre un gruppo per tensore (L12.D.attn_qkv).
"""
from __future__ import annotations

import argparse
import re
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from .adattamento import error_of
from .formato import TensorKey
from .menu import MenuConfig, TensorBatch, apply_option, iter_batches
from .sorgenti import FisherSource, PlaneSource, TeacherSource


# --------------------------------------------------------------------------- statistiche del gradiente
@dataclass
class GradStat:
    """Per UN tensore, nella base memorizzata (stessa forma [out, in] del tensore):
    g  gradiente della KL media per token;
    f  Fisher diagonale per token (>= 0);
    g2 media di UNA sola meta' delle righe (g e' la media di tutte), facoltativo: |g - g2| . dW e' `se`."""
    g: torch.Tensor
    f: torch.Tensor
    g2: Optional[torch.Tensor] = None


class GradSource(ABC):
    @abstractmethod
    def has(self, key: TensorKey) -> bool: ...

    @abstractmethod
    def load(self, key: TensorKey) -> GradStat: ...


class DictGradSource(GradSource):
    def __init__(self, data: Dict[TensorKey, GradStat]):
        self._d = data

    def has(self, key):
        return key in self._d

    def load(self, key):
        return self._d[key]


class SafetensorsGradSource(GradSource):
    """File per strato `grad_L{layer:03d}.safetensors` con `{nome}.g`, `{nome}.f` (e `{nome}.g2`),
    scritti da `dkl_raccogli.py --salva` (bf16 o fp32)."""

    def __init__(self, directory: str):
        import glob
        import os
        from safetensors import safe_open
        self._open, self._where = safe_open, {}
        for f in sorted(glob.glob(os.path.join(directory, "grad_L*.safetensors"))):
            with safe_open(f, "pt") as h:
                for n in h.keys():
                    if n.endswith(".g"):
                        self._where[TensorKey.parse(n[:-2])] = f

    def has(self, key):
        return key in self._where

    def load(self, key):
        n = key.name()
        with self._open(self._where[key], "pt") as h:
            g2 = h.get_tensor(f"{n}.g2").float() if f"{n}.g2" in h.keys() else None
            return GradStat(h.get_tensor(f"{n}.g").float(), h.get_tensor(f"{n}.f").float(), g2)


# --------------------------------------------------------------------------- Taylor su un lotto
TAYLOR_BYTE_PER_PESO = 64   # 3 pile fp32 (12 B) + temporanei fp64 di taylor_lotto (~5 x 8 B) + margine

@dataclass
class Taylor:
    primo: float
    secondo: float
    se: float = 0.0

    @property
    def dkl(self) -> float:
        return self.primo + self.secondo


def taylor_lotto(tb: TensorBatch, dW: torch.Tensor, g: torch.Tensor, f: torch.Tensor,
                 g2: Optional[torch.Tensor] = None, secondo: str = "diag") -> List[Taylor]:
    """dW, g, f, g2: [B,o,i] nella base memorizzata del lotto. -> un Taylor per tensore.
    secondo="diag": 1/2 sum F_ij dW_ij^2 (Fisher diagonale raccolta). secondo="kfac": 0.5 tr(H_O dW H_I dW^T)
    con i fattori della FisherSource del lotto (`tb.dist`): con una K-FAC VERA (--fisher-dir) cattura le
    correlazioni dentro il tensore che la diagonale ignora; con il proxy imatrix NON e' una misura."""
    dW = dW.double()
    primo = (g.double() * dW).sum((-1, -2))
    if secondo == "kfac":
        secondo_t = tb.dist(dW.float()).double()
    elif secondo == "diag":
        secondo_t = 0.5 * (f.double().clamp_min(0.0) * dW * dW).sum((-1, -2))
    else:
        raise ValueError(secondo)
    # g = media di tutte le righe, g2 = media di UNA meta': g - g2 = (gA - gB)/2 = errore standard di g
    se = ((g2.double() - g.double()) * dW).sum((-1, -2)).abs() if g2 is not None else torch.zeros_like(primo)
    return [Taylor(float(a), float(b), float(c)) for a, b, c in zip(primo.cpu(), secondo_t.cpu(), se.cpu())]


def _stack(xs: List[torch.Tensor], dev) -> torch.Tensor:
    return torch.stack([x.to(dev, torch.float32) for x in xs])


@torch.no_grad()
def taylor_tensori(ps: PlaneSource, ts: TeacherSource, fs: FisherSource, gs: GradSource, cfg: MenuConfig,
                   keys: Optional[Sequence[TensorKey]] = None, options: Optional[Sequence[str]] = None,
                   log: Optional[Callable[[str], None]] = None, prefetch: bool = True,
                   secondo: str = "diag") -> Dict[TensorKey, Dict[str, Taylor]]:
    """Per ogni tensore con gradiente e per ogni opzione del menu: il Taylor di dKL.

    g/f DEVONO essere nella base memorizzata (lo studente in ricetta B deriva rispetto a trit x d
    nel dominio ruotato: e' gia' cosi'). Un gradiente nella base originale si ruota con
    `formato.rotate_weight_in` (il prodotto scalare si conserva, H e' ortonormale), ma una
    Fisher DIAGONALE non si ruota: va raccolta nella base memorizzata."""
    keys = [k for k in (list(keys) if keys is not None else ps.keys()) if gs.has(k)]
    # R015: il Taylor tiene in piu' le pile g/f/g2 (3 x fp32) e i temporanei fp64 di taylor_lotto (~5 x 8 B per peso):
    # non sono in per_tensor_bytes, e senza contarli il tetto di memoria del lotto era troppo ottimista.
    cfg = replace(cfg, extra_bytes_per_weight=max(cfg.extra_bytes_per_weight, TAYLOR_BYTE_PER_PESO))
    groups: Dict[tuple, List[TensorKey]] = defaultdict(list)
    for k in keys:
        inf = ps.info(k)
        groups[(inf.n_planes, inf.out, inf.in_, inf.rotated, inf.group, inf.packing)].append(k)
    out: Dict[TensorKey, Dict[str, Taylor]] = {k: {} for k in keys}
    for (K, o, i, rot, _grp, _pk), chunk, tb in iter_batches(groups, cfg, ps, ts, fs, prefetch):
        dev = tb.w_t.device
        stats = [gs.load(k) for k in chunk]
        g = _stack([s.g for s in stats], dev)
        f = _stack([s.f for s in stats], dev)
        g2 = _stack([s.g2 for s in stats], dev) if all(s.g2 is not None for s in stats) else None
        d_att = error_of(tb.planes, tb.scales0, tb.w_t)        # i pesi che lo studente HA (scale del file)
        for name in cfg.option_names(K):
            if options is not None and name not in options:
                continue
            fr = apply_option(tb, name)
            ts_ = taylor_lotto(tb, fr.D - d_att, g, f, g2, secondo)
            for k, t in zip(chunk, ts_):
                out[k][name] = t
        if log:
            log(f"dkl lotto K={K} {o}x{i} rot={rot}: {len(chunk)} tensori")
        del tb, g, f, g2, d_att
    return out


# --------------------------------------------------------------------------- gruppi
def gruppo_di(key: TensorKey, modo: str) -> str:
    """Nome di gruppo (sintassi di misurato.parse_selector) per la chiave. Per `blocco:N` il
    gruppo e' un selettore con intervallo: L12.E0-31.down."""
    if key.expert < 0:
        return f"L{key.layer}.D.{key.kind}"
    if modo == "tensore":
        return key.name()
    if modo == "esperto":
        return f"L{key.layer}.E{key.expert}"
    if modo == "strato":
        return f"L{key.layer}.E*.{key.kind}"
    m = re.fullmatch(r"blocco:(\d+)", modo)
    if m:
        n = int(m.group(1))
        lo = (key.expert // n) * n
        return f"L{key.layer}.E{lo}-{lo + n - 1}.{key.kind}"
    raise ValueError(f"modo di raggruppamento sconosciuto: {modo!r}")


def aggrega(per_tensore: Dict[TensorKey, Dict[str, Taylor]], modo: str = "esperto",
            attesi: Optional[Dict[str, int]] = None) -> Dict[str, Dict[str, Taylor]]:
    """Somma i Taylor dei tensori di ogni gruppo. Un'opzione entra nel gruppo solo se esiste in
    TUTTI i membri misurati (misurato.py la applica a ogni tensore del gruppo). Il se si somma in
    quadratura (stime indipendenti fra tensori). `attesi`: {gruppo: n. membri}; un gruppo con meno
    membri misurati (es. tensore senza gradiente) viene scartato — mai un dKL parziale."""
    membri: Dict[str, List[TensorKey]] = defaultdict(list)
    for k in per_tensore:
        membri[gruppo_di(k, modo)].append(k)
    res: Dict[str, Dict[str, Taylor]] = {}
    for g, ks in sorted(membri.items()):
        if attesi is not None and len(ks) != attesi.get(g, len(ks)):
            continue
        comuni = set.intersection(*[set(per_tensore[k]) for k in ks])
        d = {}
        for o in sorted(comuni):
            ts = [per_tensore[k][o] for k in ks]
            d[o] = Taylor(sum(t.primo for t in ts), sum(t.secondo for t in ts),
                          float(sum(t.se ** 2 for t in ts) ** 0.5))
        if d:
            res[g] = d
    return res


_ORDINE = {"cur": 0, "-1": 1, "+1": 2}


def scrivi_tsv(gruppi: Dict[str, Dict[str, Taylor]], path: str, intestazione: str = "") -> int:
    """dkl.tsv per `unisci.py --delta-kl`. Ritorna il numero di righe."""
    n = 0
    with open(path, "w") as f:
        if intestazione:
            for ln in intestazione.splitlines():
                f.write(f"# {ln}\n")
        f.write("group\toption\tdelta_kl\tse\tprimo\tsecondo\n")
        for g, opts in gruppi.items():
            for o, t in sorted(opts.items(), key=lambda x: (_ORDINE.get(x[0], 3), x[0])):
                f.write(f"{g}\t{o}\t{t.dkl:.10e}\t{t.se:.4e}\t{t.primo:.10e}\t{t.secondo:.10e}\n")
                n += 1
    return n


def aggiungi_tsv(gruppi: Dict[str, Dict[str, Taylor]], path: str, nuovo: bool) -> int:
    """Come scrivi_tsv ma in coda (un blocco di strati alla volta: il lavoro non si perde)."""
    n = 0
    with open(path, "w" if nuovo else "a") as f:
        if nuovo:
            f.write("group\toption\tdelta_kl\tse\tprimo\tsecondo\n")
        for g, opts in gruppi.items():
            for o, t in sorted(opts.items(), key=lambda x: (_ORDINE.get(x[0], 3), x[0])):
                f.write(f"{g}\t{o}\t{t.dkl:.10e}\t{t.se:.4e}\t{t.primo:.10e}\t{t.secondo:.10e}\n")
                n += 1
    return n


def membri_attesi(chiavi_piani: Iterable[TensorKey], modo: str) -> Dict[str, int]:
    c: Dict[str, int] = defaultdict(int)
    for k in chiavi_piani:
        c[gruppo_di(k, modo)] += 1
    return dict(c)


# --------------------------------------------------------------------------- CLI (gradienti su disco)
def main(argv=None) -> int:
    from .dispositivo import batch_memory_budget, resolve_device
    from .dry_run_gguf import GGUF, IMATRIX, TEACHER
    from .gguf_piani import GGUFPlaneSource, GGUFTeacherSource, ImatrixFisherSource
    from .unisci import parse_layers
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grad-dir", required=True, help="grad_L*.safetensors (dkl_raccogli.py --salva)")
    ap.add_argument("--out", required=True, help="dkl.tsv")
    ap.add_argument("--gguf", default=GGUF)
    ap.add_argument("--teacher", nargs="+", default=TEACHER)
    ap.add_argument("--imatrix", default=IMATRIX)
    ap.add_argument("--layers", nargs="+", default=["0-39"])
    ap.add_argument("--secondo", choices=("diag", "kfac"), default="diag",
                    help="secondo ordine: Fisher diagonale raccolta | K-FAC della sorgente (--fisher-dir)")
    ap.add_argument("--fisher-dir", default=None, help="fisher_L*.safetensors (K-FAC vera) per --secondo kfac")
    ap.add_argument("--gruppi", default="esperto", help="tensore | esperto | strato | blocco:N")
    ap.add_argument("--options", nargs="*", default=None, help="solo queste opzioni (default: tutte + cur)")
    ap.add_argument("--densities", nargs="*", type=float, default=[0.10, 0.25])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--mem-frac", type=float, default=0.4)
    a = ap.parse_args(argv)
    dev = resolve_device(a.device)
    on_gpu = dev.type == "cuda"
    cfg = MenuConfig(densities=tuple(a.densities), batch=a.batch or (256 if on_gpu else 8), device=str(dev),
                     max_batch_weights=None if on_gpu else 8 * 1024 * 1024,
                     max_batch_bytes=batch_memory_budget(dev, a.mem_frac) if on_gpu else None)
    ps = GGUFPlaneSource(a.gguf, layers=parse_layers(a.layers), max_experts=256)
    ts = GGUFTeacherSource(a.teacher)
    if a.fisher_dir:
        from .sorgenti import SafetensorsFisherSource
        fs = SafetensorsFisherSource(a.fisher_dir)
    else:
        fs = ImatrixFisherSource(a.imatrix, out_of=lambda k: ps.info(k).out, in_of=lambda k: ps.info(k).in_,
                                 fallback="uniform", log=lambda m: None)
    gs = SafetensorsGradSource(a.grad_dir)
    opts = None if a.options is None else set(a.options) | {"cur"}
    per = taylor_tensori(ps, ts, fs, gs, cfg, ps.keys(), opts, log=lambda m: print(m, flush=True),
                         secondo=a.secondo)
    gr = aggrega(per, a.gruppi, membri_attesi([k for k in ps.keys() if gs.has(k)], a.gruppi))
    n = scrivi_tsv(gr, a.out, f"dKL di Taylor, gruppi={a.gruppi}, gradienti={a.grad_dir}")
    print(f"{len(gr)} gruppi, {n} righe -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
