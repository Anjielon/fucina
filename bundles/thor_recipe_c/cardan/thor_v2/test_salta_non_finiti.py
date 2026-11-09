"""24/9 23:0x: passo_ottimizzazione(salta_non_finiti=True, diagnostica=False).
1) passo normale: parametri identici al bit a salta_non_finiti=False;
2) loss non finita (un peso ternario messo a NaN): passo SALTATO, parametri invariati, niente stato Adam;
3) gradiente non finito con loss finita (gancio che avvelena un gradiente): SALTATO, parametri invariati."""
import copy
import sys

import torch

torch.set_num_threads(1)
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from finestra_globale import costruisci_finestra_giocattolo, parametri_ternari, passo_ottimizzazione  # noqa: E402


def _finestra(seed=0):
    t_fn, q_fn, pesi, h0, _ = costruisci_finestra_giocattolo(2, seed=seed)
    params = [p for strato in pesi for p in parametri_ternari(strato)]
    opt = torch.optim.AdamW(params, lr=1e-2, weight_decay=0.0)
    return t_fn, q_fn, params, h0, opt


guasti = 0
# 1) passo normale identico
a = _finestra(); b = _finestra()
passo_ottimizzazione(a[0], a[1], a[3], a[4], salta_non_finiti=False, diagnostica=False)
r = passo_ottimizzazione(b[0], b[1], b[3], b[4], salta_non_finiti=True, diagnostica=False)
uguali = all(torch.equal(x, y) for x, y in zip(a[2], b[2])) and not r.get("saltato")
print("1) passo normale identico al bit:", uguali); guasti += not uguali

# 2) loss non finita
t_fn, q_fn, params, h0, opt = _finestra(1)
with torch.no_grad():
    params[0].view(-1)[0] = float("nan")
prima = [p.detach().clone() for p in params]
r = passo_ottimizzazione(t_fn, q_fn, h0, opt, salta_non_finiti=True, diagnostica=False)
ok2 = r.get("saltato") == "loss" and all(torch.equal(x, y) or (torch.isnan(x) == torch.isnan(y)).all() and torch.equal(torch.nan_to_num(x), torch.nan_to_num(y)) for x, y in zip(prima, params)) and len(opt.state) == 0
print("2) loss non finita -> saltato:", r.get("saltato"), "· parametri invariati, Adam vuoto:", ok2); guasti += not ok2

# 3) gradiente non finito, loss finita
t_fn, q_fn, params, h0, opt = _finestra(2)
params[1].register_hook(lambda g: g * float("inf"))
prima = [p.detach().clone() for p in params]
r = passo_ottimizzazione(t_fn, q_fn, h0, opt, salta_non_finiti=True, diagnostica=False)
ok3 = r.get("saltato") == "gradienti" and all(torch.equal(x, y) for x, y in zip(prima, params)) and len(opt.state) == 0
print("3) gradiente non finito -> saltato:", r.get("saltato"), "· parametri invariati, Adam vuoto:", ok3); guasti += not ok3

print("OK" if not guasti else f"FALLITO ({guasti})")
sys.exit(1 if guasti else 0)
