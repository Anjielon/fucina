"""Ricostruzione a GRADIENTE (stile CAT-Q, arXiv 2606.26650) per TQ1_0R_1024,
al posto del GPTQ greedy sequenziale di `gptq_1024_lib.gptq_1024`.

Motivazione (21/9 sera, richiesta di Angelo — "scrivi il trainer mentre aspetti"):
CAT-Q Tab.4 mostra che sotto i 2 bit il GPTQ greedy e' la famiglia PEGGIORE
(GPTQ 34.38 vs il loro metodo 72.72, media 5 benchmark, Llama2-70B ~2bit).
La leva vera non e' "GPTQ con una Hessiana migliore" (provato oggi pomeriggio,
ROMPE la sonda anche quando l'errore pesato-H isolato migliora) ma un
ALGORITMO diverso: ottimizzazione congiunta di scala+soglia a gradiente, con
ternarizzazione "ammorbidita" che si irrigidisce nel tempo, invece della
correzione sequenziale colonna-per-colonna di GPTQ.

⚠️ NIENTE shift (mu): il nostro formato TQ1_0R_1024 e' scala+trit puro
({-1,0,1}), non (W-mu)/alfa come CAT-Q generico — la modulazione qui e' SOLO
alfa (scala) e Delta (soglia), niente mu.

Riusa la STESSA rotazione fissa (R_1024 + segno S1024) e lo STESSO GROUP=128
di `gptq_1024_lib` — cambia SOLO come si decidono alfa/trit dentro il gruppo,
non il formato su disco. Output (D, Q) nello stesso shape/dtype di
`gptq_1024()`, cosi' `pack_da_d_q` e il writer GGUF esistenti non cambiano.

Verifica PRIMA di usarlo in una forgia vera: `verifica_softern.py` nello
stesso albero (loss GPTQ vs softern sotto la STESSA H, su un tensore vero).
"""
from __future__ import annotations
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gptq_1024_lib import GROUP, _scala_un_gruppo  # noqa: E402


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def _f_transizione(What: torch.Tensor, s: float, delta_norm: torch.Tensor) -> torch.Tensor:
    """f(What; s, delta) = [tanh(s(What-delta)) + tanh(s(What+delta))] / [2 tanh(s)]
    — Eq.5 di CAT-Q. What e' il peso NORMALIZZATO (W/alfa), delta_norm la
    soglia nella STESSA unita' normalizzata (Delta/alfa nel paper). Per s
    piccolo ~ identita', per s grande -> ternarizzazione dura {-1,0,1}."""
    s_t = torch.as_tensor(s, dtype=What.dtype, device=What.device)
    num = torch.tanh(s_t * (What - delta_norm)) + torch.tanh(s_t * (What + delta_norm))
    den = 2.0 * torch.tanh(s_t)
    return num / den.clamp(min=1e-12)


def gptq_1024_softern(
    W: torch.Tensor,
    H: torch.Tensor,
    n_epochi: int = 300,
    gamma: float = 0.6,
    s0: float = 30.0,
    lr: float = 0.02,
    smorzamento: float = 0.01,
    device: str = "cuda",
    log_ogni: int = 0,
):
    """Ricostruzione congiunta a gradiente di (alfa, Delta) per ogni (riga,
    gruppo-di-128), stessa forma di `gptq_1024_lib.gptq_1024`.

    Segue lo schema a DUE STADI di CAT-Q Eq.6 (non un tanh sempre piu' ripido
    all'infinito, che nel primo tentativo faceva DIVERGERE la loss quando la
    sharpness saliva oltre ~15):
      · stadio 2 (0 < t <= gamma): ternarizzazione MORBIDA, sharpness
        s(t) = gamma * t * s0 — CRESCE LINEARE, tetto a gamma^2*s0 (non
        esponenziale fino a 40-60 come nel primo tentativo).
      · stadio 3 (gamma < t <= 1): ternarizzazione DURA vera (Q reale,
        {-1,0,1}), gradiente **straight-through** (backward = identita').
        Qui il gradiente guida SOLO alfa/Delta con la discretizzazione gia'
        attiva — e' la fase che il primo tentativo saltava del tutto.

    W: (n_righe, n_in) GIA' ruotata. H: (n_in, n_in) Hessiana ruotata GREZZA
    (non il fattore Cholesky — qui serve la forma quadratica vera).
    Ritorna (D, Q) nello stesso contratto di gptq_1024, drop-in per pack_da_d_q.
    """
    n_righe, n_in = W.shape
    assert n_in % GROUP == 0
    ng = n_in // GROUP

    Wt = W.to(device=device, dtype=torch.float32)
    Ht = np.array(H, dtype=np.float64, copy=True)
    Ht[np.diag_indices_from(Ht)] += smorzamento * np.mean(np.diag(Ht))
    Ht = torch.as_tensor(Ht, dtype=torch.float32, device=device)

    # warm-start: stessa scala LS della RTN/GPTQ esistente.
    Wg = Wt.reshape(n_righe * ng, GROUP)
    alfa0 = _scala_un_gruppo(Wg).reshape(n_righe, ng, 1).clamp(min=1e-6)
    log_alfa = torch.nn.Parameter(alfa0.log().clone())
    # delta parte a 0.5 (TWN/absmean classico): Delta_norm = sigmoid(raw)*1.5.
    _init_delta = 0.5
    _raw0 = float(np.log(_init_delta / 1.5 / (1 - _init_delta / 1.5)))
    delta_raw = torch.nn.Parameter(torch.full((n_righe, ng, 1), _raw0, device=device))

    opt = torch.optim.Adam([log_alfa, delta_raw], lr=lr)
    n_stadio2 = max(1, int(n_epochi * gamma))
    _fresh_opt_fatto = False

    for epoca in range(n_epochi):
        t = (epoca + 1) / n_epochi
        alfa = log_alfa.exp()
        delta_norm = torch.sigmoid(delta_raw) * 1.5
        What = Wt.reshape(n_righe, ng, GROUP) / alfa
        if epoca < n_stadio2:
            s = gamma * t * s0
            T = _f_transizione(What, s, delta_norm)
        else:
            if not _fresh_opt_fatto:
                # momento di Adam accumulato nella fase morbida non ha senso
                # nella fase dura (il paesaggio cambia carattere di netto):
                # ottimizzatore fresco, lr piu' basso, la fase dura corregge
                # solo di poco cio' che la fase morbida ha gia' impostato.
                opt = torch.optim.Adam([log_alfa, delta_raw], lr=lr * 0.1)
                _fresh_opt_fatto = True
            # stadio 3: DURA con straight-through (round-through trick,
            # standard per la ternarizzazione: forward = hard, backward = identita').
            hard = torch.where(What > delta_norm, torch.ones_like(What),
                                torch.where(What < -delta_norm, -torch.ones_like(What), torch.zeros_like(What)))
            T = hard + (What - What.detach())  # forward=hard, d(T)/d(What)=1 (STE)
            s = float("inf")
        Wq = (alfa * T).reshape(n_righe, n_in)
        diff = Wt - Wq
        loss = ((diff @ Ht) * diff).sum(dim=-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if log_ogni and (epoca % log_ogni == 0 or epoca == n_epochi - 1):
            log(f"  epoca {epoca+1}/{n_epochi} · s={s if s==s else 0:.2f} · loss={loss.item():.6g}"
                if s != float("inf") else f"  epoca {epoca+1}/{n_epochi} · DURA(STE) · loss={loss.item():.6g}")

    with torch.no_grad():
        alfa = log_alfa.exp()
        delta_norm = torch.sigmoid(delta_raw) * 1.5
        What = Wt.reshape(n_righe, ng, GROUP) / alfa
        Q = torch.where(What > delta_norm, torch.ones_like(What),
                         torch.where(What < -delta_norm, -torch.ones_like(What), torch.zeros_like(What)))
        D = alfa.squeeze(-1).unsqueeze(-1).cpu()
        Qout = Q.reshape(n_righe, n_in).to(torch.int8).cpu()
    return D, Qout


def loss_quadratica(W: torch.Tensor, H: torch.Tensor, D: torch.Tensor, Q: torch.Tensor,
                     smorzamento: float = 0.01, device: str = "cuda") -> float:
    """Stessa forma quadratica usata dal trainer, per confrontare GPTQ vs
    softern SOTTO LA STESSA H — la metrica che decide chi vince, non la sonda
    (quella si fa dopo, sul modello vero)."""
    n_righe, n_in = W.shape
    Ht = np.array(H, dtype=np.float64, copy=True)
    Ht[np.diag_indices_from(Ht)] += smorzamento * np.mean(np.diag(Ht))
    Ht = torch.as_tensor(Ht, dtype=torch.float32, device=device)
    ng = n_in // GROUP
    Wt = W.to(device=device, dtype=torch.float32)
    Dd = D.to(device).reshape(n_righe, ng, 1)
    Qd = Q.to(device, torch.float32).reshape(n_righe, ng, GROUP)
    Wq = (Dd * Qd).reshape(n_righe, n_in)
    diff = Wt - Wq
    with torch.no_grad():
        return ((diff @ Ht) * diff).sum(dim=-1).mean().item()
