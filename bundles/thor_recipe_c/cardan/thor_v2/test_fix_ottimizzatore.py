#!/usr/bin/env python3
"""Test delle correzioni dell'audit del 24/9 sull'ottimizzatore (CPU, giocattolo):

(a) B2 — con weight_decay=0 un parametro che non riceve gradiente resta
    IDENTICO AL BIT dopo n passi di AdamW; col decadimento di default (1e-2)
    invece si muove (il test sa fallire sul codice vecchio).
(b) B1 — `passo_ottimizzazione` fa il backward sulla SOMMA: il gradiente e'
    esattamente numel volte quello della media (cablaggio verificato).
(c) B15 — calcolare F̂(h) fuori dal grafo NON cambia il gradiente: confronto
    col gradiente della forma "collassata" e_{l+1} = F̂(h_l+e_l) − F(h_l).
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from finestra_globale import costruisci_finestra_giocattolo, errore_lemma1_finestra, passo_ottimizzazione  # noqa: E402
from quant_ternaria_globale_batch_v2 import TernarioDiffBatchFuso  # noqa: E402


def _parametri(pesi_quant_per_strato):
    return [p for strato in pesi_quant_per_strato for coppia in strato for m in coppia for p in m.parameters()]


def test_a_decadimento() -> bool:
    torch.manual_seed(0)
    W = torch.randn(4, 8, 16) * 0.02
    esiti = {}
    for wd in (0.0, 1e-2):
        t = TernarioDiffBatchFuso(W.clone(), group=8)
        opt = torch.optim.AdamW(t.parameters(), lr=0.05, weight_decay=wd)
        iniziali = {n: p.detach().clone() for n, p in t.named_parameters()}
        for _ in range(50):
            opt.zero_grad(set_to_none=True)
            out = t.forward(4.0)
            (out[:2] ** 2).sum().backward()   # solo gli esperti 0-1 hanno gradiente
            opt.step()
        esiti[wd] = all(torch.equal(p[2:], iniziali[n][2:]) for n, p in t.named_parameters())
        mossi = any(not torch.equal(p[:2], iniziali[n][:2]) for n, p in t.named_parameters())
        print(f"  (a) wd={wd}: esperti senza gradiente identici al bit = {esiti[wd]} · "
              f"esperti con gradiente mossi = {mossi}")
    ok = esiti[0.0] and not esiti[1e-2]
    print(f"  (a) {'✅' if ok else '⛔'} decadimento 0 preserva gli esperti non instradati "
          f"(e il test riconosce il decadimento di default)")
    return ok


def test_b_somma() -> bool:
    tf, qf, pesi, h0, _ = costruisci_finestra_giocattolo(2, seed=1)
    ps = _parametri(pesi)
    e0 = torch.randn_like(h0) * 0.05
    grad = {}
    for rid in ("media", "somma"):
        opt = torch.optim.SGD(ps, lr=0.0)   # lr 0: si guardano solo i gradienti
        passo_ottimizzazione(tf, qf, h0, opt, e0, riduzione=rid)
        grad[rid] = torch.cat([p.grad.reshape(-1) for p in ps if p.grad is not None]).clone()
    numel = h0.numel()                     # stessa forma per ogni strato della finestra
    rel = ((grad["somma"] - numel * grad["media"]).norm() / grad["somma"].norm()).item()
    ok = rel < 1e-5
    print(f"  (b) {'✅' if ok else '⛔'} grad(somma) = {numel}·grad(media): scarto relativo {rel:.2e}")
    return ok


def test_c_fq_senza_grafo() -> bool:
    tf, qf, pesi, h0, _ = costruisci_finestra_giocattolo(3, seed=2)
    ps = _parametri(pesi)
    e0 = torch.randn_like(h0) * 0.05

    for p in ps:
        p.grad = None
    errore_lemma1_finestra(tf, qf, h0, e0)["loss_somma"].backward()
    g_kernel = torch.cat([p.grad.reshape(-1) for p in ps])

    for p in ps:
        p.grad = None
    h, e, loss = h0, e0, torch.zeros(())
    for l in range(len(qf)):
        with torch.no_grad():
            Ft = tf[l](h)
        e = qf[l](h + e) - Ft                 # forma collassata, un solo forward quantizzato
        loss = loss + e.pow(2).sum()
        h = Ft
    loss.backward()
    g_rif = torch.cat([p.grad.reshape(-1) for p in ps])

    rel = ((g_kernel - g_rif).norm() / g_rif.norm()).item()
    ok = rel < 1e-5
    print(f"  (c) {'✅' if ok else '⛔'} gradiente del kernel (F̂(h) senza grafo) = forma collassata: "
          f"scarto relativo {rel:.2e}")
    return ok


if __name__ == "__main__":
    risultati = [test_a_decadimento(), test_b_somma(), test_c_fq_senza_grafo()]
    print("✅ TUTTI I TEST DELLE CORREZIONI PASSANO" if all(risultati) else "⛔ QUALCHE TEST FALLISCE")
    raise SystemExit(0 if all(risultati) else 1)
