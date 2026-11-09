#!/usr/bin/env python3
"""Versione V2 (FUSA) di `TernarioDiffBatch` (`quant_ternaria_globale_batch.py`,
SOLA LETTURA — non una riga toccata li'): stessa identica matematica, stessa
classe base (eredita da `TernarioDiffBatch`: init, `_delta_norm`,
`frazione_ternaria` INVARIATI), ma `forward`/`forward_duro` passano da un
NUCLEO COMPILATO (`torch.compile`, backend Inductor->Triton) invece che dalla
catena di ~9 operazioni elementwise separate di `_f_transizione`.

24/9/2026. DIAGNOSI (misurata, non supposta — vedi `profila_batch_linee.py`
in scratchpad e la sezione "PROVA" in fondo a questo file): `TernarioDiffBatch`
NON era piu' lento per un errore di forma o un reshape che copia. Era piu'
lento perche' e' PIU' VELOCIZZANTE del previsto in un modo contro-intuitivo:

  - Ogni singola operazione elementwise di `_f_transizione` (sub, mul, tanh,
    add, mul, tanh, somma, div — ~9 kernel PyTorch separati, ciascuno alloca
    e MATERIALIZZA un tensore di output della STESSA taglia dell'input) sul
    tensore gate_up completo (E=256, shape (256,1024,2048)=536.9M elementi,
    2.000 GiB fp32) costa ~19 ms CIASCUNA — misurato: 8 operazioni di questo
    tipo fanno ~160ms, + altri 19ms per `s*T` = 180ms totali, che COMBACIA
    con la misura originale (batch forward-solo gate_up: 180.229 ms).
  - Bandwidth implicita: ogni op legge ~2GiB e scrive ~2GiB = ~4GiB di
    traffico; 4GiB / 19ms = ~209 GB/s — VICINISSIMO al picco reale di banda
    di questa APU UMA (Ryzen AI Max+ 395, LPDDR5X). La versione batch e'
    quindi BANDWIDTH-BOUND: la catena di 9 operazioni NON FUSE fa ~9 giri
    completi di lettura+scrittura in memoria di sistema per calcolare
    un'espressione che richiederebbe, fusa in UN kernel, un solo giro.
  - Il loop per-esperto NON e' piu' veloce perche' "vettorizza meglio": e'
    piu' veloce perche' ogni tensore per-esperto (8 MiB, shape
    (1024,16,128)) e' cosi' PICCOLO che l'INTERA catena di 9 operazioni resta
    calda in cache (L2/Infinity Cache) invece di rimbalzare 9 volte sulla
    RAM di sistema — misurato: `_f_transizione` su UN esperto costa 0.19 ms;
    x256 = 48.6 ms, contro i 160 ms della stessa identica catena fatta in UN
    colpo sul tensore da 2 GiB (3.3x piu' lenta nonostante ZERO overhead di
    lancio kernel python aggiuntivo — la differenza e' TUTTA banda/cache).
  - Conferma diretta (torch.compile fonde tanh+tanh+add su tensore 2GiB):
    eager 121.7 ms -> compilato 18.7 ms (6.5x) per SOLO 2 delle 9 operazioni
    — la fusione elimina i giri intermedi in RAM, riportando il costo al
    singolo giro di banda (~19ms), esattamente la cifra attesa.

CURA: fondere l'INTERA catena (_delta_norm + _f_transizione + s*T) in UN
kernel Triton generato da `torch.compile`, cosi' il tensore da 2 GiB viene
letto UNA volta e scritto UNA volta, non 9. Stesso principio, stesso stile
gia' in uso in questo progetto stanotte per un problema simmetrico (troppi
LANCI piccoli invece di troppi GIRI di RAM): vedi `kernel_blocco_fuso.py`
(Triton su ROCm gia' verificato funzionante su questa macchina).

⛔ beta passato come TENSORE 0-dim (mai un float python nudo) al nucleo
compilato: un float python diventa una COSTANTE di compilazione per Dynamo
-> un ricompilo per ogni valore diverso di beta (lo schedule ne usa ~9-16
per finestra, moltiplicati per 40 strati = centinaia di ricompili, ognuno
costoso quanto l'intero vantaggio della fusione). Un tensore 0-dim e' un
input RUNTIME: un solo compilo, riusato per ogni beta e per OGNI strato
successivo con la stessa forma (E, n_out, n_in, group) — la stessa forma
ricorre 40 volte (una per strato), quindi il costo di compilazione (one-off,
~pochi secondi) si ammortizza su tutta la forgia.

⛔ QUESTO FILE E' NUOVO: ne' `quant_ternaria_globale.py` ne'
`quant_ternaria_globale_batch.py` sono toccati. Verificato CONTRO entrambi
(vedi `_verifica_equivalenza_v2` sotto, stessa identica disciplina di
`_verifica_equivalenza` in `quant_ternaria_globale_batch.py`: init, forward
morbido/duro a piu' beta, traiettoria di training completa, parametri
finali — bit-esatto a tolleranza macchina, MAI un pareggio "circa uguale").

⛔ VINCOLO DI SESSIONE: verificare `rocm-smi --showmeminfo vram` libera PRIMA
di ogni test GPU. Memoria liberata a fine test.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quant_ternaria_globale import GROUP, schedula_beta  # noqa: E402 — riferimento, SOLA LETTURA
from quant_ternaria_globale_batch import (  # noqa: E402 — riferimento, SOLA LETTURA
    TernarioDiffBatch, _costruisci_loop, _forward_loop, _max_abs_diff, _tempo,
)


# ══════════════════════ nuclei compilati (Inductor -> Triton) ══════════════════════
# Un SOLO kernel fuso per l'intera espressione: niente `_f_transizione` separata,
# niente `_delta_norm` separata — tutto dentro la regione compilata, cosi'
# Inductor puo' fondere OGNI passo in un unico giro di lettura/scrittura.
# `beta_t` e' un tensore 0-dim: vedi motivazione anti-ricompilo sopra.

@torch.compile
def _nucleo_morbido_compilato(z: torch.Tensor, log_s: torch.Tensor, delta_raw: torch.Tensor,
                               delta_max: float, beta_t: torch.Tensor) -> torch.Tensor:
    s = log_s.exp()
    delta = torch.sigmoid(delta_raw) * delta_max
    num = torch.tanh(beta_t * (z - delta)) + torch.tanh(beta_t * (z + delta))
    den = 2.0 * torch.tanh(beta_t)
    T = num / den.clamp(min=1e-12)
    return s * T


@torch.compile
def _nucleo_duro_compilato(z: torch.Tensor, log_s: torch.Tensor, delta_raw: torch.Tensor,
                            delta_max: float, beta_t: torch.Tensor) -> torch.Tensor:
    s = log_s.exp()
    delta = torch.sigmoid(delta_raw) * delta_max
    num = torch.tanh(beta_t * (z - delta)) + torch.tanh(beta_t * (z + delta))
    den = 2.0 * torch.tanh(beta_t)
    T_soft = num / den.clamp(min=1e-12)
    with torch.no_grad():
        z_d, delta_d = z.detach(), delta.detach()
        hard = torch.where(z_d > delta_d, torch.ones_like(z_d),
                            torch.where(z_d < -delta_d, -torch.ones_like(z_d), torch.zeros_like(z_d)))
    T = hard + (T_soft - T_soft.detach())  # forward=hard, d(T)/dz = d(T_soft)/dz (STE)
    return s * T


@torch.compile
def _nucleo_duro_compilato_bf16(z: torch.Tensor, log_s: torch.Tensor, delta_raw: torch.Tensor,
                                 delta_max: float, beta_t: torch.Tensor) -> torch.Tensor:
    """Come _nucleo_duro_compilato, ma scrive DIRETTAMENTE bf16 (24/9 20:2x, radiografia:
    copie fp32<->bf16 22% del passo). Stessa formula, cast fuso nello stesso kernel:
    identico al bit a `_nucleo_duro_compilato(...).to(torch.bfloat16)`, meta' dei byte
    scritti e un passaggio di memoria in meno, avanti e indietro."""
    s = log_s.exp()
    delta = torch.sigmoid(delta_raw) * delta_max
    num = torch.tanh(beta_t * (z - delta)) + torch.tanh(beta_t * (z + delta))
    den = 2.0 * torch.tanh(beta_t)
    T_soft = num / den.clamp(min=1e-12)
    with torch.no_grad():
        z_d, delta_d = z.detach(), delta.detach()
        hard = torch.where(z_d > delta_d, torch.ones_like(z_d),
                            torch.where(z_d < -delta_d, -torch.ones_like(z_d), torch.zeros_like(z_d)))
    T = hard + (T_soft - T_soft.detach())
    return (s * T).to(torch.bfloat16)


class TernarioDiffBatchFuso(TernarioDiffBatch):
    """Come `TernarioDiffBatch` (STESSO `__init__`, STESSA `_delta_norm`,
    STESSA `frazione_ternaria` — ereditati, non riscritti), ma `forward` e
    `forward_duro` chiamano il nucleo COMPILATO invece della catena elementwise
    non fusa. Nessuna nuova matematica: stessa formula, un solo giro di
    memoria invece di ~9."""

    def forward(self, beta: float) -> torch.Tensor:
        beta_t = torch.as_tensor(beta, dtype=self.z.dtype, device=self.z.device)
        out = _nucleo_morbido_compilato(self.z, self.log_s, self.delta_raw, self.delta_max, beta_t)
        return out.reshape(self.shape)

    def forward_duro(self, beta_surrogato: float, uscita_bf16: bool = False) -> torch.Tensor:
        beta_t = torch.as_tensor(beta_surrogato, dtype=self.z.dtype, device=self.z.device)
        nucleo = _nucleo_duro_compilato_bf16 if uscita_bf16 else _nucleo_duro_compilato
        out = nucleo(self.z, self.log_s, self.delta_raw, self.delta_max, beta_t)
        return out.reshape(self.shape)


# ══════════════════════ verifica di CORRETTEZZA (loop-di-riferimento vs fuso) ══════════════════════
def _verifica_equivalenza_v2(device: str, E: int = 16, n_out: int = 32, n_in: int = 128,
                              group: int = 32, seed: int = 0, n_passi_training: int = 30,
                              lr: float = 0.05, tol: float = 1e-6) -> bool:
    """IDENTICA disciplina di `_verifica_equivalenza` in
    quant_ternaria_globale_batch.py (stessa firma, stessi passi 1-4), ma
    confronta il loop-di-riferimento contro `TernarioDiffBatchFuso` invece
    che contro `TernarioDiffBatch`."""
    print(f"\n══ verifica di equivalenza V2 FUSA (loop-riferimento vs fuso), device={device} ══")
    print(f"  E={E}  n_out={n_out}  n_in={n_in}  group={group}  (ng={n_in // group})")
    torch.manual_seed(seed)
    W = (torch.randn(E, n_out, n_in) * 0.02).to(device)

    loop = _costruisci_loop(W, group, device)
    fuso = TernarioDiffBatchFuso(W, group=group).to(device)

    ok = True

    # 1) parametri iniziali
    for e, m in enumerate(loop):
        d_log_s = _max_abs_diff(m.log_s.reshape(-1), fuso.log_s[e].reshape(-1))
        d_z = _max_abs_diff(m.z.reshape(-1), fuso.z[e].reshape(-1))
        d_delta = _max_abs_diff(m.delta_raw.reshape(-1), fuso.delta_raw[e].reshape(-1))
        if max(d_log_s, d_z, d_delta) > tol:
            ok = False
            print(f"    ⛔ esperto {e}: init diverge (log_s={d_log_s:.2e} z={d_z:.2e} delta_raw={d_delta:.2e})")
    print(f"    init (parametri appena costruiti): {'✅ combaciano' if ok else '⛔ DIVERGONO'} (tol={tol:.0e})")

    # 2) forward morbido e duro, piu' beta
    ok2 = True
    for beta, duro, nome in [(1.0, False, "morbido beta=1"), (8.0, False, "morbido beta=8"),
                              (16.0, True, "duro beta=16 (STE)")]:
        Wl = _forward_loop(loop, beta, duro)
        Wf = fuso.forward_duro(beta) if duro else fuso.forward(beta)
        d = _max_abs_diff(Wl, Wf)
        bene = d <= tol
        ok2 = ok2 and bene
        print(f"    forward [{nome}]: max|diff|={d:.3e}  ({'✅' if bene else '⛔'})")
    ok = ok and ok2

    # 3) traiettoria di training identica
    parametri_loop: list = []
    for m in loop:
        parametri_loop += list(m.parameters())
    opt_loop = torch.optim.AdamW(parametri_loop, lr=lr)
    opt_fuso = torch.optim.AdamW(fuso.parameters(), lr=lr)

    n_dura = max(1, int(n_passi_training * 0.2))
    n_morbida = n_passi_training - n_dura
    betas_morbidi = schedula_beta(1.0, 16.0, max(1, n_morbida))

    ok3 = True
    perdite_loop, perdite_fuso = [], []
    for t in range(n_passi_training):
        duro = t >= n_morbida
        beta = 16.0 if duro else betas_morbidi[min(t, len(betas_morbidi) - 1)]

        opt_loop.zero_grad(set_to_none=True)
        Wl = _forward_loop(loop, beta, duro)
        loss_l = F.mse_loss(Wl, W)
        loss_l.backward()
        opt_loop.step()

        opt_fuso.zero_grad(set_to_none=True)
        Wf = fuso.forward_duro(beta) if duro else fuso.forward(beta)
        loss_f = F.mse_loss(Wf, W)
        loss_f.backward()
        opt_fuso.step()

        perdite_loop.append(loss_l.item())
        perdite_fuso.append(loss_f.item())
        d_loss = abs(loss_l.item() - loss_f.item())
        if d_loss > tol * 10:
            ok3 = False
            print(f"    ⛔ passo {t}: loss diverge (loop={loss_l.item():.6e} fuso={loss_f.item():.6e} diff={d_loss:.2e})")
    print(f"    training ({n_passi_training} passi: {n_morbida} morbidi + {n_dura} duri): "
          f"loss loop {perdite_loop[0]:.4e}->{perdite_loop[-1]:.4e}  "
          f"loss fuso {perdite_fuso[0]:.4e}->{perdite_fuso[-1]:.4e}  "
          f"({'✅ traiettorie combacianti' if ok3 else '⛔ traiettorie DIVERGENTI'})")

    # 4) parametri finali, esperto per esperto
    ok4 = True
    for e, m in enumerate(loop):
        d = max(_max_abs_diff(m.log_s.reshape(-1), fuso.log_s[e].reshape(-1)),
                _max_abs_diff(m.z.reshape(-1), fuso.z[e].reshape(-1)),
                _max_abs_diff(m.delta_raw.reshape(-1), fuso.delta_raw[e].reshape(-1)))
        if d > tol * 10:
            ok4 = False
            print(f"    ⛔ esperto {e}: parametri finali divergono (max|diff|={d:.2e})")
    print(f"    parametri finali (dopo training): {'✅ combaciano' if ok4 else '⛔ DIVERGONO'} esperto-per-esperto")

    ok = ok and ok3 and ok4
    print(f"  {'✅ EQUIVALENZA VERIFICATA (bit-esatta a tolleranza macchina)' if ok else '⛔ EQUIVALENZA FALLITA'}")
    return ok


# ══════════════════════ misura di VELOCITA' (loop vs batch-naive vs fuso, scala realistica) ══════════════════════
def _misura_famiglia_v2(nome: str, device: str, E: int, n_out: int, n_in: int, group: int,
                         n_ripetizioni: int = 20) -> dict:
    print(f"\n── famiglia {nome}: E={E} n_out={n_out} n_in={n_in} group={group} ──")
    torch.manual_seed(0)
    W = (torch.randn(E, n_out, n_in, device=device) * 0.02)

    loop = _costruisci_loop(W, group, device)
    batch_naive = TernarioDiffBatch(W, group=group).to(device)
    fuso = TernarioDiffBatchFuso(W, group=group).to(device)

    opt_loop = torch.optim.AdamW([p for m in loop for p in m.parameters()], lr=0.05)
    opt_naive = torch.optim.AdamW(batch_naive.parameters(), lr=0.05)
    opt_fuso = torch.optim.AdamW(fuso.parameters(), lr=0.05)

    beta = 8.0

    def _fwd_loop():
        with torch.no_grad():
            _forward_loop(loop, beta, duro=False)

    def _fwd_naive():
        with torch.no_grad():
            batch_naive.forward(beta)

    def _fwd_fuso():
        with torch.no_grad():
            fuso.forward(beta)

    def _fwdbwd_loop():
        opt_loop.zero_grad(set_to_none=True)
        Wl = _forward_loop(loop, beta, duro=False)
        F.mse_loss(Wl, W).backward()
        opt_loop.step()

    def _fwdbwd_naive():
        opt_naive.zero_grad(set_to_none=True)
        Wb = batch_naive.forward(beta)
        F.mse_loss(Wb, W).backward()
        opt_naive.step()

    def _fwdbwd_fuso():
        opt_fuso.zero_grad(set_to_none=True)
        Wf = fuso.forward(beta)
        F.mse_loss(Wf, W).backward()
        opt_fuso.step()

    # riscaldamento SEPARATO e piu' lungo per il fuso (compilazione al primo giro:
    # non deve contaminare la misura di velocita' a regime)
    print("    (compilazione torch.compile in corso, primo giro puo' essere lento...)")
    t_compile0 = time.perf_counter()
    _fwd_fuso()
    _fwdbwd_fuso()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"    tempo di compilazione (una tantum, ammortizzato su 40 strati): {time.perf_counter()-t_compile0:.2f}s")

    t_fwd_loop = _tempo(_fwd_loop, n_ripetizioni)
    t_fwd_naive = _tempo(_fwd_naive, n_ripetizioni)
    t_fwd_fuso = _tempo(_fwd_fuso, n_ripetizioni)
    t_fb_loop = _tempo(_fwdbwd_loop, n_ripetizioni)
    t_fb_naive = _tempo(_fwdbwd_naive, n_ripetizioni)
    t_fb_fuso = _tempo(_fwdbwd_fuso, n_ripetizioni)

    risultato = {
        "nome": nome, "E": E, "n_out": n_out, "n_in": n_in, "group": group,
        "fwd_loop_ms": t_fwd_loop * 1e3, "fwd_naive_ms": t_fwd_naive * 1e3, "fwd_fuso_ms": t_fwd_fuso * 1e3,
        "fwd_speedup_vs_loop": t_fwd_loop / max(t_fwd_fuso, 1e-12),
        "fwd_speedup_vs_naive": t_fwd_naive / max(t_fwd_fuso, 1e-12),
        "fwdbwd_loop_ms": t_fb_loop * 1e3, "fwdbwd_naive_ms": t_fb_naive * 1e3, "fwdbwd_fuso_ms": t_fb_fuso * 1e3,
        "fwdbwd_speedup_vs_loop": t_fb_loop / max(t_fb_fuso, 1e-12),
        "fwdbwd_speedup_vs_naive": t_fb_naive / max(t_fb_fuso, 1e-12),
    }
    print(f"    forward-solo:      loop {risultato['fwd_loop_ms']:8.3f} ms   "
          f"naive {risultato['fwd_naive_ms']:8.3f} ms   fuso {risultato['fwd_fuso_ms']:8.3f} ms   "
          f"-> fuso/loop {risultato['fwd_speedup_vs_loop']:6.2f}x   fuso/naive {risultato['fwd_speedup_vs_naive']:6.2f}x")
    print(f"    forward+backward:  loop {risultato['fwdbwd_loop_ms']:8.3f} ms   "
          f"naive {risultato['fwdbwd_naive_ms']:8.3f} ms   fuso {risultato['fwdbwd_fuso_ms']:8.3f} ms   "
          f"-> fuso/loop {risultato['fwdbwd_speedup_vs_loop']:6.2f}x   fuso/naive {risultato['fwdbwd_speedup_vs_naive']:6.2f}x")

    del loop, batch_naive, fuso, opt_loop, opt_naive, opt_fuso, W
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return risultato


def _misura_velocita_v2(device: str) -> list:
    print("\n══ misura di velocita' V2 (scala realistica Thor: E=256, loop vs naive vs fuso) ══")
    risultati = []
    risultati.append(_misura_famiglia_v2("gate_up", device, E=256, n_out=1024, n_in=2048, group=128))
    risultati.append(_misura_famiglia_v2("down", device, E=256, n_out=2048, n_in=512, group=256))
    return risultati


def _test(device: str) -> bool:
    print(f"quant_ternaria_globale_batch_v2.py — TernarioDiffBatchFuso, device={device}")
    r1 = _verifica_equivalenza_v2(device, E=16, n_out=32, n_in=128, group=32)
    r2 = _verifica_equivalenza_v2(device, E=8, n_out=64, n_in=64, group=64, seed=7)
    ok_correttezza = r1 and r2
    if not ok_correttezza:
        print("\n⛔ Equivalenza NON verificata — salto la misura di velocita'.")
        return False
    if device == "cuda":
        _misura_velocita_v2(device)
    else:
        print("\n(misura di velocita' saltata: significativa solo su GPU, --device cuda)")
    return ok_correttezza


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⛔ --device cuda richiesto ma torch non vede una GPU. Uso cpu.")
        args.device = "cpu"
    esito = _test(device=args.device)
    if args.device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        print(f"\nVRAM allocata dopo pulizia: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
    raise SystemExit(0 if esito else 1)
