#!/usr/bin/env python3
"""Versione BATCHATA di `TernarioDiff` (`quant_ternaria_globale.py`, SOLA
LETTURA — non una riga toccata li'): STESSA identica matematica (init
Eq.8-stile CAT-Q + latente `z` libero per-peso, soft-to-hard via
`_f_transizione`, `forward_duro` STE), ma con una dimensione `E` (esperti) IN
TESTA — UN modulo invece di un `ModuleList` di E istanze ricostruite con un
loop Python ad ogni chiamata.

24/9/2026. Nato dalla misura sul PRIMO test reale di
`valida_finestra_8_12.py::ottimizza_finestra_JOINT` (2 strati, 200 passi):
oltre 87 minuti senza completarsi, fermato — proiettato a 20+ ore su 40
strati, contro il mandato esplicito di <2 ore totali.

Sospetto (letto nel codice, verificato sotto): `_stratofn_quant_reale` fa
    gu_rot = torch.stack([fn(m) for m in ternari_gu], dim=0)
un loop Python di 256 chiamate a modulo, per 2 famiglie (gate_up/down), per 2
rami (h/h+e) per passo — fino a 1024 forward separati per passo, 200 passi =
~200k chiamate. E' lo STESSO anti-pattern gia' trovato e corretto stanotte
altrove nello stesso progetto:
  - `gptq_1024_refine.py::raffina_scale_joint_batch` — docstring: "chiamata
    256 volte per round ... e' un ciclo Python di chiamate CPU<->GPU che a
    scala e' diventato il vero collo di bottiglia (misurato: 1.1x di
    accelerazione nonostante il trit_sweep batchato fosse gia' perfetto)."
  - `kernel_blocco_fuso.py` — stesso principio per il trit-sweep.
In entrambi i casi la cura e' la stessa: una dimensione BATCH in piu' su
tensori/parametri, zero loop Python, un einsum/broadcast piu' grande al posto
di N chiamate piccole.

⛔ QUESTO FILE E' NUOVO: `quant_ternaria_globale.py` resta la versione di
riferimento VERIFICATA stanotte (4 prove passate in `_test`) — non si tocca.
Questo file si verifica CONTRO quello (vedi `_verifica_equivalenza` sotto),
non lo sostituisce di autorita'.

⛔ VINCOLO DI SESSIONE (GPU condivisa, uguale a quant_ternaria_globale.py):
verificare `rocm-smi --showmeminfo vram` libera PRIMA di ogni test GPU. Test
di correttezza su scala GIOCATTOLO (8-16 "esperti", hidden 64-128). UN solo
test a scala realistica (256 esperti, hidden=2048/intermedio=512 — le shape
VERE di gate_up_proj/down_proj di Thor, lette da
`valida_finestra_8_12.py::_costruisci_ternari_layer` + `verifica_memoria_finestra_reale.py`)
SOLO sulla ricostruzione del peso (forward/forward+backward di TernarioDiff),
MAI il forward dell'intero layer Qwen3_5MoeDecoderLayer (richiederebbe
caricare Thor — fuori mandato per questo file). Memoria liberata a fine test.
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
from gptq_1024_softern import _f_transizione  # noqa: E402 — riusata, MAI duplicata (Eq.5 CAT-Q)
from quant_ternaria_globale import GROUP, TernarioDiff, _logit, schedula_beta  # noqa: E402 — riferimento, SOLA LETTURA


class TernarioDiffBatch(nn.Module):
    """Come `TernarioDiff` ma per un BATCH di `E` tensori (E, n_out, n_in) —
    stessa scala `s` e soglia `delta` per (esperto, riga, gruppo di `group`
    colonne), stesso latente `z` libero per singolo peso — ORA con una
    dimensione `E` in testa su ogni parametro: zero loop Python, tutto
    broadcast/vettoriale. Nessuna riduzione (mean/sum) attraversa MAI la
    dimensione E: ogni esperto resta matematicamente indipendente dagli
    altri, esattamente come nel `ModuleList` originale — `E` e' solo un asse
    di batching per il motore, non un accoppiamento fra esperti.

    `forward(beta)`/`forward_duro(beta_surrogato)` — identici a TernarioDiff,
    vedi la' per la spiegazione della matematica (Eq.5 CAT-Q, STE)."""

    def __init__(self, W: torch.Tensor, group: int = GROUP, beta_0: float = 1.0,
                 clip: float = 0.95, delta_init: float = 0.5, delta_max: float = 2.0,
                 eps: float = 1e-8):
        super().__init__()
        assert W.dim() == 3, "atteso (E, n_out, n_in) — batch di E esperti, non un tensore solo"
        E, n_out, n_in = W.shape
        assert n_in % group == 0, f"n_in={n_in} non multiplo di group={group}"
        assert 0.0 < delta_init < delta_max, "delta_init deve stare in (0, delta_max)"
        ng = n_in // group
        self.n_esperti = E
        self.shape = (E, n_out, n_in)
        self.group = group
        self.beta_0 = beta_0
        self.clip = clip
        self.delta_max = delta_max

        # ── STESSA identica init Eq.8 di TernarioDiff (s0 = media|W| per
        # gruppo, z0 = atanh(clip(W/s0))/beta_0), eseguita per TUTTI gli E
        # esperti insieme: la riduzione (mean) e' lungo l'ultimo asse
        # `group`, indipendente da E — aggiungere E in testa non cambia UN
        # solo numero del risultato, lo calcola solo tutto in un colpo.
        Wg = W.detach().to(torch.float32).reshape(E, n_out, ng, group)
        s0 = Wg.abs().mean(dim=-1, keepdim=True).clamp_min(eps)      # (E, n_out, ng, 1)
        What0 = (Wg / s0).clamp(-clip, clip)
        z0 = torch.atanh(What0) / beta_0                              # (E, n_out, ng, group)

        self.log_s = nn.Parameter(s0.log().clone())
        self.z = nn.Parameter(z0.clone())
        raw0 = _logit(delta_init / delta_max)
        self.delta_raw = nn.Parameter(torch.full((E, n_out, ng, 1), raw0))

    def _delta_norm(self) -> torch.Tensor:
        """Identico a TernarioDiff._delta_norm, per (esperto, riga, gruppo)."""
        return torch.sigmoid(self.delta_raw) * self.delta_max

    def forward(self, beta: float) -> torch.Tensor:
        """Fase MORBIDA: W_hat = s * f_transizione(z, beta, delta) — per TUTTI
        gli E esperti in una sola chiamata (broadcast su `group`, come nella
        versione singola; qui anche su `E`)."""
        s = self.log_s.exp()                          # (E, n_out, ng, 1)
        delta = self._delta_norm()                     # (E, n_out, ng, 1)
        T = _f_transizione(self.z, beta, delta)         # (E, n_out, ng, group)
        return (s * T).reshape(self.shape)

    def forward_duro(self, beta_surrogato: float) -> torch.Tensor:
        """Identico a TernarioDiff.forward_duro (STE), per tutti gli E insieme."""
        s = self.log_s.exp()
        delta = self._delta_norm()
        T_soft = _f_transizione(self.z, beta_surrogato, delta)
        with torch.no_grad():
            z_d, delta_d = self.z.detach(), delta.detach()
            hard = torch.where(z_d > delta_d, torch.ones_like(z_d),
                                torch.where(z_d < -delta_d, -torch.ones_like(z_d), torch.zeros_like(z_d)))
        T = hard + (T_soft - T_soft.detach())           # forward=hard, d(T)/dz = d(T_soft)/dz (STE)
        return (s * T).reshape(self.shape)

    @torch.no_grad()
    def frazione_ternaria(self, beta: float, soglia_ambiguo: float = 0.1) -> dict:
        """Identico a TernarioDiff.frazione_ternaria, mediato su TUTTI gli E
        esperti insieme (diagnostica aggregata, non usata dalla verifica di
        equivalenza sotto — quella confronta forward/forward_duro/gradienti
        esperto-per-esperto)."""
        delta = self._delta_norm()
        z, d = self.z, delta
        frac_zero = (z.abs() <= d).float().mean().item()
        frac_piu = (z > d).float().mean().item()
        frac_meno = (z < -d).float().mean().item()
        T_soft = _f_transizione(z, beta, d)
        T_hard = torch.where(z > d, torch.ones_like(z), torch.where(z < -d, -torch.ones_like(z), torch.zeros_like(z)))
        ambiguo = ((T_soft - T_hard).abs() > soglia_ambiguo).float().mean().item()
        return {"beta": beta, "vicino_meno1": frac_meno, "vicino_zero": frac_zero,
                "vicino_piu1": frac_piu, "ambiguo": ambiguo}


# ══════════════════════ verifica di CORRETTEZZA (loop-di-riferimento vs batch) ══════════════════════
def _costruisci_loop(W: torch.Tensor, group: int, device: str) -> torch.nn.ModuleList:
    """E istanze SEPARATE della classe ORIGINALE (di riferimento), una per
    esperto, sugli STESSI pesi iniziali di W[e] — replica ESATTA di
    `_costruisci_ternari_layer` in valida_finestra_8_12.py (stessa chiamata,
    stessi argomenti default)."""
    moduli = [TernarioDiff(W[e], group=group).to(device) for e in range(W.shape[0])]
    return torch.nn.ModuleList(moduli)


def _forward_loop(moduli: torch.nn.ModuleList, beta: float, duro: bool) -> torch.Tensor:
    """Replica ESATTA del punto di chiamata reale in
    `_stratofn_quant_reale`: `torch.stack([fn(m) for m in ternari], dim=0)`."""
    fn = (lambda m: m.forward_duro(beta)) if duro else (lambda m: m.forward(beta))
    return torch.stack([fn(m) for m in moduli], dim=0)


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


def _verifica_equivalenza(device: str, E: int = 16, n_out: int = 32, n_in: int = 128,
                           group: int = 32, seed: int = 0, n_passi_training: int = 30,
                           lr: float = 0.05, tol: float = 1e-6) -> bool:
    """Verifica RIGOROSA: batch vs E-istanze-in-loop devono dare risultati
    identici a tolleranza macchina (1e-6), su:
      1. i parametri APPENA INIZIALIZZATI (stessa Eq.8 per ciascun esperto);
      2. forward(beta) e forward_duro(beta) per piu' beta (fase morbida E dura);
      3. l'intera traiettoria di training: N passi di AdamW con LO STESSO lr,
         LA STESSA loss (MSE contro W originale, come nei test 3/4 del file
         di riferimento), applicati IN PARALLELO alle due versioni — se la
         matematica e' davvero la stessa, gradienti e aggiornamenti Adam
         restano identici passo dopo passo (nessuna sorgente di casualita' in
         mezzo: niente dropout, niente RNG dopo l'init)."""
    print(f"\n══ verifica di equivalenza (loop-riferimento vs batch), device={device} ══")
    print(f"  E={E}  n_out={n_out}  n_in={n_in}  group={group}  (ng={n_in // group})")
    torch.manual_seed(seed)
    W = (torch.randn(E, n_out, n_in) * 0.02).to(device)

    loop = _costruisci_loop(W, group, device)
    batch = TernarioDiffBatch(W, group=group).to(device)

    ok = True

    # 1) parametri iniziali — devono combaciare esperto per esperto
    for e, m in enumerate(loop):
        d_log_s = _max_abs_diff(m.log_s.reshape(-1), batch.log_s[e].reshape(-1))
        d_z = _max_abs_diff(m.z.reshape(-1), batch.z[e].reshape(-1))
        d_delta = _max_abs_diff(m.delta_raw.reshape(-1), batch.delta_raw[e].reshape(-1))
        if max(d_log_s, d_z, d_delta) > tol:
            ok = False
            print(f"    ⛔ esperto {e}: init diverge (log_s={d_log_s:.2e} z={d_z:.2e} delta_raw={d_delta:.2e})")
    print(f"    init (parametri appena costruiti): {'✅ combaciano' if ok else '⛔ DIVERGONO'} (tol={tol:.0e})")

    # 2) forward morbido e duro, piu' beta (fase morbida e dura)
    ok2 = True
    for beta, duro, nome in [(1.0, False, "morbido beta=1"), (8.0, False, "morbido beta=8"),
                              (16.0, True, "duro beta=16 (STE)")]:
        Wl = _forward_loop(loop, beta, duro)
        Wb = batch.forward_duro(beta) if duro else batch.forward(beta)
        d = _max_abs_diff(Wl, Wb)
        bene = d <= tol
        ok2 = ok2 and bene
        print(f"    forward [{nome}]: max|diff|={d:.3e}  ({'✅' if bene else '⛔'})")
    ok = ok and ok2

    # 3) traiettoria di training identica — STESSO ottimizzatore/stessi
    # gradienti attesi: un AdamW per la versione loop (su tutti i parametri
    # di tutte le E istanze, come fa il codice reale) e un AdamW indipendente
    # per la versione batch, stesso lr, stesso schedule di beta, stessa loss.
    parametri_loop: list = []
    for m in loop:
        parametri_loop += list(m.parameters())
    opt_loop = torch.optim.AdamW(parametri_loop, lr=lr)
    opt_batch = torch.optim.AdamW(batch.parameters(), lr=lr)

    n_dura = max(1, int(n_passi_training * 0.2))
    n_morbida = n_passi_training - n_dura
    betas_morbidi = schedula_beta(1.0, 16.0, max(1, n_morbida))

    ok3 = True
    perdite_loop, perdite_batch = [], []
    for t in range(n_passi_training):
        duro = t >= n_morbida
        beta = 16.0 if duro else betas_morbidi[min(t, len(betas_morbidi) - 1)]

        opt_loop.zero_grad(set_to_none=True)
        Wl = _forward_loop(loop, beta, duro)
        loss_l = F.mse_loss(Wl, W)
        loss_l.backward()
        opt_loop.step()

        opt_batch.zero_grad(set_to_none=True)
        Wb = batch.forward_duro(beta) if duro else batch.forward(beta)
        loss_b = F.mse_loss(Wb, W)
        loss_b.backward()
        opt_batch.step()

        perdite_loop.append(loss_l.item())
        perdite_batch.append(loss_b.item())
        d_loss = abs(loss_l.item() - loss_b.item())
        if d_loss > tol * 10:   # loss e' una riduzione su tutto il tensore: tolleranza 10x
            ok3 = False
            print(f"    ⛔ passo {t}: loss diverge (loop={loss_l.item():.6e} batch={loss_b.item():.6e} diff={d_loss:.2e})")
    print(f"    training ({n_passi_training} passi: {n_morbida} morbidi + {n_dura} duri): "
          f"loss loop {perdite_loop[0]:.4e}->{perdite_loop[-1]:.4e}  "
          f"loss batch {perdite_batch[0]:.4e}->{perdite_batch[-1]:.4e}  "
          f"({'✅ traiettorie combacianti' if ok3 else '⛔ traiettorie DIVERGENTI'})")

    # parametri finali, esperto per esperto (non solo la loss aggregata)
    ok4 = True
    for e, m in enumerate(loop):
        d = max(_max_abs_diff(m.log_s.reshape(-1), batch.log_s[e].reshape(-1)),
                _max_abs_diff(m.z.reshape(-1), batch.z[e].reshape(-1)),
                _max_abs_diff(m.delta_raw.reshape(-1), batch.delta_raw[e].reshape(-1)))
        if d > tol * 10:
            ok4 = False
            print(f"    ⛔ esperto {e}: parametri finali divergono (max|diff|={d:.2e})")
    print(f"    parametri finali (dopo training): {'✅ combaciano' if ok4 else '⛔ DIVERGONO'} esperto-per-esperto")

    ok = ok and ok3 and ok4
    print(f"  {'✅ EQUIVALENZA VERIFICATA (bit-esatta a tolleranza macchina)' if ok else '⛔ EQUIVALENZA FALLITA'}")
    return ok


# ══════════════════════ misura di VELOCITA' (loop vs batch, scala realistica) ══════════════════════
def _tempo(fn, n_ripetizioni: int, n_scaldamento: int = 3) -> float:
    """Media di n_ripetizioni chiamate sincronizzate (CUDA), dopo n_scaldamento
    passate di riscaldamento (compilazione kernel/allocator caching — MAI
    contate, altrimenti il primo giro penalizzerebbe ingiustamente il batch,
    che ha UN kernel invece di 256 da scaldare)."""
    for _ in range(n_scaldamento):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_ripetizioni):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_ripetizioni


def _misura_famiglia(nome: str, device: str, E: int, n_out: int, n_in: int, group: int,
                      n_ripetizioni: int = 20) -> dict:
    """Misura forward-solo e forward+backward, loop-di-riferimento vs batch,
    a shape REALE (letta da valida_finestra_8_12.py/verifica_memoria_finestra_reale.py:
    gate_up_proj (E,2*intermedio,hidden)=(256,1024,2048) gruppo 128;
    down_proj (E,hidden,intermedio)=(256,2048,512) gruppo 256)."""
    print(f"\n── famiglia {nome}: E={E} n_out={n_out} n_in={n_in} group={group} ──")
    torch.manual_seed(0)
    W = (torch.randn(E, n_out, n_in) * 0.02).to(device)

    loop = _costruisci_loop(W, group, device)
    batch = TernarioDiffBatch(W, group=group).to(device)
    parametri_loop = []
    for m in loop:
        parametri_loop += list(m.parameters())
    opt_loop = torch.optim.AdamW(parametri_loop, lr=0.05)
    opt_batch = torch.optim.AdamW(batch.parameters(), lr=0.05)

    beta = 8.0  # fase morbida (il ramo piu' frequente: 80% dei passi nel kernel reale)

    def _fwd_loop():
        with torch.no_grad():
            _forward_loop(loop, beta, duro=False)

    def _fwd_batch():
        with torch.no_grad():
            batch.forward(beta)

    def _fwdbwd_loop():
        opt_loop.zero_grad(set_to_none=True)
        Wl = _forward_loop(loop, beta, duro=False)
        F.mse_loss(Wl, W).backward()
        opt_loop.step()

    def _fwdbwd_batch():
        opt_batch.zero_grad(set_to_none=True)
        Wb = batch.forward(beta)
        F.mse_loss(Wb, W).backward()
        opt_batch.step()

    t_fwd_loop = _tempo(_fwd_loop, n_ripetizioni)
    t_fwd_batch = _tempo(_fwd_batch, n_ripetizioni)
    t_fb_loop = _tempo(_fwdbwd_loop, n_ripetizioni)
    t_fb_batch = _tempo(_fwdbwd_batch, n_ripetizioni)

    risultato = {
        "nome": nome, "E": E, "n_out": n_out, "n_in": n_in, "group": group,
        "fwd_loop_ms": t_fwd_loop * 1e3, "fwd_batch_ms": t_fwd_batch * 1e3,
        "fwd_speedup": t_fwd_loop / max(t_fwd_batch, 1e-12),
        "fwdbwd_loop_ms": t_fb_loop * 1e3, "fwdbwd_batch_ms": t_fb_batch * 1e3,
        "fwdbwd_speedup": t_fb_loop / max(t_fb_batch, 1e-12),
    }
    print(f"    forward-solo:      loop {risultato['fwd_loop_ms']:8.3f} ms  "
          f"batch {risultato['fwd_batch_ms']:8.3f} ms  →  {risultato['fwd_speedup']:6.2f}x")
    print(f"    forward+backward:  loop {risultato['fwdbwd_loop_ms']:8.3f} ms  "
          f"batch {risultato['fwdbwd_batch_ms']:8.3f} ms  →  {risultato['fwdbwd_speedup']:6.2f}x")

    del loop, batch, opt_loop, opt_batch, W
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return risultato


def _misura_velocita(device: str) -> list:
    print(f"\n══ misura di velocita' (scala realistica Thor: E=256) ══")
    risultati = []
    # gate_up_proj: (256, 2*512, 2048) — group=128 (GRUPPO_GATEUP_REALE, ruotato)
    risultati.append(_misura_famiglia("gate_up", device, E=256, n_out=1024, n_in=2048, group=128))
    # down_proj: (256, 2048, 512) — group=256 (GRUPPO_DOWN_REALE, non ruotato)
    risultati.append(_misura_famiglia("down", device, E=256, n_out=2048, n_in=512, group=256))
    return risultati


def _test(device: str) -> bool:
    print(f"quant_ternaria_globale_batch.py — TernarioDiffBatch, device={device}")
    print(f"GROUP={GROUP} (fonte gptq_1024_lib, via quant_ternaria_globale) · "
          f"_f_transizione riusata da gptq_1024_softern")
    r1 = _verifica_equivalenza(device, E=16, n_out=32, n_in=128, group=32)
    r2 = _verifica_equivalenza(device, E=8, n_out=64, n_in=64, group=64, seed=7)
    ok_correttezza = r1 and r2
    if not ok_correttezza:
        print("\n⛔ Equivalenza NON verificata — salto la misura di velocita' "
              "(un numero veloce ma sbagliato non serve a nulla).")
        return False
    if device == "cuda":
        _misura_velocita(device)
    else:
        print("\n(misura di velocita' saltata: significativa solo su GPU, --device cuda)")
    return ok_correttezza


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                     help="cuda = ROCm-as-cuda su venv-catq. La verifica di correttezza e' "
                          "minuscola su entrambi; la misura di velocita' gira SOLO su cuda "
                          "(a scala realistica E=256) — memoria liberata a fine script.")
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⛔ --device cuda richiesto ma torch non vede una GPU (serve venv-catq ROCm). Uso cpu.")
        args.device = "cpu"
    esito = _test(device=args.device)
    if args.device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        print(f"\nVRAM allocata dopo pulizia: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
    raise SystemExit(0 if esito else 1)
