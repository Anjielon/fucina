"""Raccolta dei fattori di Kronecker (K-FAC) della KL del modello INTERO.

Perche' etichette campionate: la KL(teacher || studente) ha gradiente nullo nel
punto teacher; quello che conta e' l'Hessiana, che coincide con la Fisher VERA
F = E_x E_{y~p(.|x)} [grad log p(y|x) grad^T]. La si stima facendo il backward
di -log p(y) con y CAMPIONATO dal modello (non dal testo: quella sarebbe la
Fisher empirica, distorta). Allora  KL(p_W || p_{W+D}) ~ 0.5 vec(D)^T F vec(D).

Per un Linear (esperto e) con ingresso x_t e gradiente d'uscita g_t:
   sum_t (g_t^T D x_t)^2  ~  tr( [sum_t g g^T] D [sum_t x x^T / n_e] D^T )      (K-FAC)
Normalizzazione scelta (per avere KL media PER TOKEN di calibrazione):
   H_I = sum_t x x^T / n_e        (media sui soli token instradati all'esperto)
   H_O = sum_t g g^T / N_totale   (N_totale = token di calibrazione * campioni)
Il peso di routing entra da solo in g (l'uscita dell'esperto e' moltiplicata da r_e)
e la frequenza d'uso dell'esperto entra nella somma di H_O: gli esperti poco usati
pesano poco, come devono.

Uso tipico (un sottoinsieme di layer per passata, per la memoria):
    coll = KFACCollector(targets)            # {TensorKey: nn.Linear}
    coll.attach(model)
    collect_fisher(model, batches, coll)
    coll.save(out_dir); coll.detach()
"""
from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .formato import TensorKey
from .sorgenti import FisherFactors, default_in_share


class KFACCollector:
    """Hook di forward (ingressi) e hook sul tensore d'uscita (gradienti).

    shared_out_kinds: tipi per cui H_O e' approssimata da una matrice CONDIVISA nel
      layer (es. "down": il gradiente d'uscita e' quello del flusso residuo, uguale
      per tutti gli esperti a meno del peso di routing). Per l'esperto e:
      H_O_e = G_layer / tr(G_layer) * sum_t ||g_t||^2 / N  -> 256x meno memoria.
    """

    def __init__(self, targets: Dict[TensorKey, nn.Module],
                 in_share: Callable[[TensorKey], Tuple[str, bool]] = default_in_share,
                 shared_out_kinds: Sequence[str] = (), device: Optional[str] = None):
        self.targets = targets
        self.in_share = in_share
        self.shared_out = set(shared_out_kinds)
        self.device = device
        self.acc_in: Dict[str, torch.Tensor] = {}
        self.n_in: Dict[str, int] = defaultdict(int)
        self.acc_out: Dict[TensorKey, torch.Tensor] = {}
        self.pool_out: Dict[Tuple[int, str], torch.Tensor] = {}
        self.trace_out: Dict[TensorKey, float] = defaultdict(float)
        self.n_total = 0
        self._handles = []

    # ------------------------------------------------------------------ hook
    def _acc(self, store, name, m: torch.Tensor):
        dev = self.device or m.device
        m = m.to(dev)
        if name in store:
            store[name].add_(m)
        else:
            store[name] = m.clone()

    def _make_hook(self, key: TensorKey):
        in_name, primary = self.in_share(key)

        def fwd(module, inputs, output):
            x = inputs[0].detach()
            x = x.reshape(-1, x.shape[-1]).float()
            if x.shape[0] == 0:
                return
            if primary:
                self._acc(self.acc_in, in_name, x.T @ x)
                self.n_in[in_name] += x.shape[0]
            if output.requires_grad:
                output.register_hook(lambda g: self._grad(key, g))
        return fwd

    def _grad(self, key: TensorKey, g: torch.Tensor):
        g = g.detach().reshape(-1, g.shape[-1]).float()
        if g.shape[0] == 0:
            return
        if key.kind in self.shared_out:
            self._acc(self.pool_out, (key.layer, key.kind), g.T @ g)
            self.trace_out[key] += float((g * g).sum())
        else:
            self._acc(self.acc_out, key, g.T @ g)

    def attach(self, model: nn.Module):
        for key, mod in self.targets.items():
            self._handles.append(mod.register_forward_hook(self._make_hook(key)))
        emb = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
        if emb is not None:
            # parametri congelati: serve un nodo che richieda gradiente per costruire il grafo
            self._handles.append(emb.register_forward_hook(lambda m, i, o: o.requires_grad_(True)))

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    # ------------------------------------------------------------------ risultati
    def finalize(self) -> Dict[TensorKey, FisherFactors]:
        n = max(self.n_total, 1)
        res = {}
        for key, mod in self.targets.items():
            in_name, _ = self.in_share(key)
            ni = max(self.n_in.get(in_name, 0), 1)
            h_i = self.acc_in[in_name] / ni
            if key.kind in self.shared_out:
                pool = self.pool_out[(key.layer, key.kind)]
                h_o = pool / pool.trace().clamp_min(1e-30) * (self.trace_out[key] / n)
            elif key in self.acc_out:
                h_o = self.acc_out[key] / n
            else:  # esperto mai attivato: Fisher nulla (distorsione zero, verra' ignorato)
                h_o = torch.zeros(mod.weight.shape[0], mod.weight.shape[0])
            res[key] = FisherFactors(h_i.cpu(), h_o.cpu(), self.n_in.get(in_name, 0))
        return res

    def save(self, directory: str, dtype: torch.dtype = torch.float16):
        """Un file per layer. fp16 basta per la memorizzazione (accumulo in fp32);
        H_O e' riscalata dal numero di token quindi il range e' contenuto."""
        from safetensors.torch import save_file
        per_layer: Dict[int, Dict[str, torch.Tensor]] = defaultdict(dict)
        for key, ff in self.finalize().items():
            in_name, _ = self.in_share(key)
            d = per_layer[key.layer]
            d[f"{in_name}.H_I"] = ff.H_I.to(dtype).contiguous()
            d[f"{in_name}.n"] = torch.tensor([ff.n_tokens], dtype=torch.int64)
            d[f"{key.name()}.H_O"] = ff.H_O.to(dtype).contiguous()
        for layer, d in per_layer.items():
            save_file(d, f"{directory}/fisher_L{layer:03d}.safetensors")


def collect_fisher(model: nn.Module, batches: Iterable[dict], collector: KFACCollector,
                   n_label_samples: int = 1, logits_fn: Optional[Callable] = None) -> int:
    """Forward + backward con etichette campionate. `batches`: dict con input_ids
    (e opz. attention_mask). Ritorna il numero di token (x campioni) accumulati."""
    model.requires_grad_(False)
    model.eval()
    for batch in batches:
        out = model(**batch) if logits_fn is None else logits_fn(model, batch)
        logits = (out.logits if hasattr(out, "logits") else out).float()
        v = logits.shape[-1]
        flat = logits.reshape(-1, v)
        valid = batch.get("attention_mask")
        valid = valid.reshape(-1).bool() if valid is not None else torch.ones(flat.shape[0], dtype=torch.bool,
                                                                               device=flat.device)
        with torch.no_grad():
            probs = flat.softmax(-1)
        for s in range(n_label_samples):
            y = torch.multinomial(probs, 1).squeeze(-1)
            loss = F.cross_entropy(flat[valid], y[valid], reduction="sum")
            loss.backward(retain_graph=s < n_label_samples - 1)
            collector.n_total += int(valid.sum())
    return collector.n_total


def fisher_bytes_per_layer(n_experts: int, hidden: int, inter: int, shared_down: bool,
                           bytes_per_el: int = 4) -> int:
    """Memoria degli accumulatori di un layer MoE (gate/up condividono H_I)."""
    per_exp = hidden ** 2 + 2 * inter ** 2 + inter ** 2 + (0 if shared_down else hidden ** 2)
    pooled = hidden ** 2 if shared_down else 0
    return (n_experts * per_exp + pooled) * bytes_per_el
