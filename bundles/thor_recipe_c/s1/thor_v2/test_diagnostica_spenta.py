"""24/9 20:0x (mandato forgia < 2 ore): errore_lemma1_finestra(diagnostica=False)
toglie il forward diagnostico F̂(h) e il controllo algebrico. Deve dare la STESSA
loss e lo STESSO gradiente della versione con diagnostica (F̂(h) era sotto no_grad
e si cancella in e = a + q). Scala giocattolo, CPU, un thread.

Tolleranze: la loss differisce solo per l'ordine delle somme in fp32
(a + q contro la differenza diretta), quindi relativa <= 1e-5; gradienti idem.
"""
import sys

import torch

torch.set_num_threads(1)
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from finestra_globale import (costruisci_finestra_giocattolo, errore_lemma1_finestra,  # noqa: E402
                              parametri_ternari)


def _una(diag: bool, seed: int) -> tuple:
    t_fn, q_fn, pesi, h0, _ = costruisci_finestra_giocattolo(2, seed=seed)
    e0 = 0.05 * torch.randn_like(h0, generator=torch.Generator().manual_seed(seed + 7))
    params = [p for strato in pesi for p in parametri_ternari(strato)]
    for p in params:
        p.grad = None
    ris = errore_lemma1_finestra(t_fn, q_fn, h0, e0, diagnostica=diag)
    ris["loss_somma"].backward()
    return ris["loss_somma"].item(), [p.grad.detach().clone() for p in params]


def main() -> int:
    guasti = 0
    for seed in (0, 1, 2):
        l1, g1 = _una(True, seed)
        l0, g0 = _una(False, seed)
        rel_l = abs(l1 - l0) / abs(l1)
        rel_g = max(((a - b).norm() / a.norm().clamp_min(1e-30)).item() for a, b in zip(g1, g0))
        ok = rel_l <= 1e-5 and rel_g <= 1e-5
        guasti += not ok
        print(f"seme {seed}: loss con {l1:.8e} senza {l0:.8e} (rel {rel_l:.2e}) · "
              f"gradiente rel max {rel_g:.2e} · {'OK' if ok else 'DIVERSI'}")
    print("OK: diagnostica spenta = stessa loss e stesso gradiente" if not guasti else f"FALLITO ({guasti})")
    return 1 if guasti else 0


if __name__ == "__main__":
    sys.exit(main())
