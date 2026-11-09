#!/usr/bin/env python3
"""Fusione di `quant_differenziabile.py` (SegnoDiff, latente libero per-peso,
arXiv 2607.14630 Eq.7-9) con la soglia esplicita di `gptq_1024_softern.py`
(CAT-Q, arXiv 2606.26650 Eq.5) — per ottenere una parametrizzazione TERNARIA
VERA (zeri inclusi), non solo il segno binario {-1,+1}.

24/9/2026. Nato dal "PROSSIMO PASSO" scritto in cima a `quant_differenziabile.py`
(sezione "CORREZIONE DI LETTERATURA"): quel file verifica (con test, non solo
teoria) che la Eq.7-9 del paper 2607.14630 e' un sign/tanh BINARIO — nessuna
zona morta, nessuno zero esplicito. Qui si fonde con `_f_transizione`
(gia' scritta, gia' in uso nella forgia vera via `gptq_1024_softern.py`, NON
duplicata: importata) per ottenere una zona morta esplicita, controllata da
una soglia `delta` LIBERA (per-gruppo, come CAT-Q), mentre il latente `z`
resta LIBERO per-peso (come SegnoDiff) — cosa che CAT-Q da solo non ha: li'
il "peso normalizzato" What=W/alfa e' DATO fisso, solo alfa/delta si allenano.
Per l'ottimizzazione GLOBALE cross-strato (Eq.9, L_E+L_GE) serve che OGNI
peso possa muoversi liberamente sotto la loss cross-strato, non solo la
scala/soglia del suo gruppo — da qui la necessita' del latente `z` libero.

⛔ VINCOLO DI SESSIONE (GPU condivisa, uguale a quant_differenziabile.py):
questo file lavora SOLO su tensori piccoli, sintetici (un "esperto" giocattolo
~8k pesi). Mai il modello Thor intero ne' un training reale su di esso.
Verifica SOLO la matematica del meccanismo di fusione.

## La matematica della fusione

`_f_transizione(What, s, delta_norm)` (gia' in gptq_1024_softern.py, Eq.5 CAT-Q):

    f(What; s, delta) = [tanh(s(What-delta)) + tanh(s(What+delta))] / [2 tanh(s)]

Verificato analiticamente (e nei test sotto) che a `s` (sharpness, qui
rinominato `beta` per coerenza col resto del progetto) grande:
    - f(0; beta, delta) = 0 SEMPRE (per qualunque beta>0: i due tanh sono
      opposti esatti in 0) — a differenza del tanh singolo di SegnoDiff, che
      vale 0 in 0 solo per costruzione banale ma NON ha una zona morta: per
      |z| piccolo ma non zero, tanh(beta*z) tende comunque a +-1 crescendo beta.
    - per |What| < delta: f -> 0 (zona morta, LARGHEZZA = delta, indipendente
      da beta — beta controlla solo quanto e' NETTA la transizione ai bordi).
    - per |What| > delta: f -> sign(What) (ternario vero {-1,0,+1}).

Qui `What` viene sostituito dal latente LIBERO `z` (non piu' dato fisso W/alfa):
    W_hat = s * f_transizione(z, beta, delta_norm(delta_raw))
con `s` (scala, per-gruppo, come SegnoDiff/CAT-Q) e `delta_norm` (soglia,
per-gruppo, sigmoid-mappata come CAT-Q) ENTRAMBE libere, ESATTAMENTE come
CAT-Q — la sola differenza (quella che serve alla via globale) e' che anche
`z` e' libero per-peso invece di essere `W/alfa` fisso.

## Test di firma: come si distingue un VERO ternario da un binario mascherato

La differenza decisiva (verificata sotto, non assunta): nel binario
(SegnoDiff) la banda "vicino a zero" SI RESTRINGE al crescere di beta (ogni
z!=0 finisce per saturare a +-1). Nel ternario fuso qui, la popolazione che
cade in "zero vero" e' decisa da |z| vs delta e NON dipende da beta — solo la
NITIDEZZA del confine dipende da beta. Il test `_prova_convergenza_ternaria`
verifica esattamente questo: la frazione di pesi "morti" (|z|<delta) resta
STABILE al crescere di beta, mentre l'ambiguita' ai bordi si restringe.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gptq_1024_lib import GROUP  # noqa: E402 — fonte UNICA di GROUP=128, mai ridichiarato
from gptq_1024_softern import _f_transizione  # noqa: E402 — riusata, MAI duplicata (Eq.5 CAT-Q)


def _logit(p: float) -> float:
    """Inversa della sigmoid, per inizializzare delta_raw da un delta_init
    desiderato (stessa idea di gptq_1024_softern, generalizzata a delta_max
    arbitrario invece del fisso 1.5 di CAT-Q)."""
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


class TernarioDiff(nn.Module):
    """Parametrizzazione soft-to-hard TERNARIA (zeri veri) di UN tensore
    (n_out, n_in): scala `s` e soglia `delta` per (riga, gruppo di `group`
    colonne) — come CAT-Q; latente `z` LIBERO per singolo peso — come SegnoDiff.

    `forward(beta)` = fase MORBIDA: W_hat = s * f_transizione(z, beta, delta).
    `forward_duro(beta_surrogato)` = fase DURA finale (STE): forward usa i
    codici REALI {-1,0,+1} (confronto esatto |z| vs delta), il backward scorre
    attraverso il surrogato f_transizione (straight-through), stesso trucco
    gia' in uso in gptq_1024_softern.py e in SegnoDiff.forward_duro.
    """

    def __init__(self, W: torch.Tensor, group: int = GROUP, beta_0: float = 1.0,
                 clip: float = 0.95, delta_init: float = 0.5, delta_max: float = 2.0,
                 eps: float = 1e-8):
        super().__init__()
        assert W.dim() == 2, "atteso (n_out, n_in) — un tensore solo, non un batch di esperti"
        n_out, n_in = W.shape
        assert n_in % group == 0, f"n_in={n_in} non multiplo di group={group}"
        assert 0.0 < delta_init < delta_max, "delta_init deve stare in (0, delta_max)"
        ng = n_in // group
        self.shape = (n_out, n_in)
        self.group = group
        self.beta_0 = beta_0
        self.clip = clip
        self.delta_max = delta_max

        # ── stessa init Eq.8 di SegnoDiff (quant_differenziabile.py): s0 =
        # media|W| per gruppo, z0 = atanh(clip(W/s0))/beta_0 — qui z0 e' SOLO
        # il punto di PARTENZA del latente libero, non un legame permanente a W.
        Wg = W.detach().to(torch.float32).reshape(n_out, ng, group)
        s0 = Wg.abs().mean(dim=-1, keepdim=True).clamp_min(eps)      # (n_out, ng, 1)
        What0 = (Wg / s0).clamp(-clip, clip)
        z0 = torch.atanh(What0) / beta_0                              # (n_out, ng, group)

        self.log_s = nn.Parameter(s0.log().clone())
        self.z = nn.Parameter(z0.clone())
        # soglia LIBERA per-gruppo, sigmoid-mappata come CAT-Q (mai negativa,
        # mai oltre delta_max): raw0 scelto perche' sigmoid(raw0)*delta_max == delta_init.
        raw0 = _logit(delta_init / delta_max)
        self.delta_raw = nn.Parameter(torch.full((n_out, ng, 1), raw0))

    def _delta_norm(self) -> torch.Tensor:
        """Soglia per-gruppo, sempre in (0, delta_max) — mai 0 esatto (garantirebbe
        instabilita' numerica in _f_transizione a delta=0, dove degenera al
        singolo-tanh di SegnoDiff) ne' delta_max esatto (collasserebbe tutto a zero)."""
        return torch.sigmoid(self.delta_raw) * self.delta_max

    def forward(self, beta: float) -> torch.Tensor:
        """Fase MORBIDA: W_hat = s * f_transizione(z, beta, delta)."""
        s = self.log_s.exp()                          # (n_out, ng, 1)
        delta = self._delta_norm()                     # (n_out, ng, 1)
        T = _f_transizione(self.z, beta, delta)         # (n_out, ng, group) — broadcast su group
        return (s * T).reshape(self.shape)

    def forward_duro(self, beta_surrogato: float) -> torch.Tensor:
        """Fase DURA finale (STE): forward = codici REALI ternari {-1,0,+1}
        (confronto ESATTO |z| vs delta, gradiente bloccato su questo ramo);
        backward = attraverso il surrogato morbido f_transizione(beta_surrogato)."""
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
        """Diagnostica pura: frazione di pesi DECISI a -1/0/+1 (confronto esatto
        |z| vs delta — la categoria vera, indipendente da beta) e frazione
        ANCORA ambigua nell'uscita morbida (|T|-|T_hard più vicino| > soglia_ambiguo,
        cioe' l'uscita morbida non ha ancora quasi raggiunto il valore duro
        a cui e' destinata). Per un TERNARIO vero: le frazioni -1/0/+1 NON
        dipendono da beta (dipendono solo da z vs delta); solo 'ambiguo' scende."""
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


def schedula_beta(beta_min: float = 1.0, beta_max: float = 16.0, passi: int = 9) -> list:
    """Identica a quant_differenziabile.py (stesso annealing esponenziale del paper)."""
    if passi <= 1:
        return [beta_max]
    r = (beta_max / beta_min) ** (1.0 / (passi - 1))
    return [beta_min * (r ** i) for i in range(passi)]


# ══════════════════════════════ TEST 1: firma ternaria vera ══════════════════════════════
def _prova_convergenza_ternaria(device: str, seed: int = 0) -> bool:
    """z e delta FISSI (nessun training): sweep di beta. Verifica la firma che
    distingue un TERNARIO VERO da un binario mascherato: le frazioni -1/0/+1
    (decise da |z| vs delta) restano STABILI (non "vicino_zero si restringe"
    come nel binario di quant_differenziabile.py) mentre SOLO l'ambiguita'
    ai bordi scende con beta crescente."""
    print("\n── esperimento 1: firma TERNARIA (z,delta fissi, solo beta cresce) ──")
    ok = True
    for n_out, n_in, ng_attese in [(64, 128, 1), (16, 384, 3)]:
        torch.manual_seed(seed)
        W = (torch.randn(n_out, n_in) * 0.02).to(device)
        mod = TernarioDiff(W).to(device)
        assert n_in // GROUP == ng_attese
        betas = schedula_beta()
        righe = [mod.frazione_ternaria(b) for b in betas]
        print(f"  tensore {n_out}x{n_in} (GROUP={GROUP}, {ng_attese} gruppi/riga):")
        for r in righe:
            print(f"    beta={r['beta']:6.2f}  -1={r['vicino_meno1']:.3f}  0={r['vicino_zero']:.3f}  "
                  f"+1={r['vicino_piu1']:.3f}  ambiguo={r['ambiguo']:.3f}")
        # firma ternaria: -1/0/+1 STABILI (deviazione max sotto 1e-6 — sono
        # calcolati da z/delta, INDIPENDENTI da beta per costruzione: se
        # cambiassero sarebbe un bug, non rumore) mentre l'ambiguita' scende.
        zeri = [r["vicino_zero"] for r in righe]
        stabile = (max(zeri) - min(zeri)) < 1e-6
        esiste_zero_vero = zeri[0] > 0.01        # deve esistere una popolazione di zeri REALI
        ambiguo_scende = righe[-1]["ambiguo"] < righe[0]["ambiguo"]
        esito = stabile and esiste_zero_vero and ambiguo_scende
        print(f"    zero stabile su beta: {stabile} (min={min(zeri):.4f} max={max(zeri):.4f})  |  "
              f"popolazione zero reale: {zeri[0]:.3f}  |  ambiguo scende: {ambiguo_scende}  "
              f"({'✅ firma TERNARIA confermata' if esito else '⛔ non atteso'})")
        ok = ok and esito
    return ok


def _prova_zeri_veri(device: str, seed: int = 3) -> bool:
    """Verifica DIRETTA e cruciale: forward_duro produce zeri ESATTI (0.0, non
    solo 'piccolo') per i pesi con |z|<delta, e valori esattamente +-s per gli
    altri — questa e' la proprieta' che il binario di quant_differenziabile.py
    NON HA (verificato la' che non esiste soglia/zona morta)."""
    print("\n── esperimento 2: zeri VERI in forward_duro (proprieta' che il binario non ha) ──")
    torch.manual_seed(seed)
    W = (torch.randn(64, 128) * 0.02).to(device)
    mod = TernarioDiff(W, delta_init=0.5, delta_max=2.0).to(device)
    Wq = mod.forward_duro(16.0)
    delta = mod._delta_norm()
    attesi_zero = (mod.z.abs() <= delta)
    n_zeri_attesi = int(attesi_zero.sum().item())
    n_zeri_reali = int((Wq == 0.0).sum().item())
    # tolleranza: forward_duro e' hard + STE_residuo; il residuo (T_soft-T_soft.detach())
    # e' ESATTAMENTE zero in valore per costruzione di autograd (detach sottrae
    # se stesso) — quindi Wq deve avere ESATTAMENTE n_zeri_attesi zeri, non "circa".
    esito = (n_zeri_reali == n_zeri_attesi) and (n_zeri_attesi > 0)
    print(f"    pesi con |z|<=delta (attesi zero): {n_zeri_attesi}/{Wq.numel()}  |  "
          f"zeri ESATTI in W_hat: {n_zeri_reali}  "
          f"({'✅ combaciano esattamente, zeri VERI confermati' if esito else '⛔ mismatch'})")
    # e i non-zero devono valere esattamente +-s (per riga/gruppo), non un valore intermedio
    s = mod.log_s.exp().reshape(-1, 1).expand(-1, mod.group).reshape(mod.shape)
    non_zero = Wq != 0.0
    ok_valori = torch.allclose(Wq[non_zero].abs(), s[non_zero], atol=1e-5)
    print(f"    |W_hat| dei non-zero == s del gruppo (esatto): "
          f"{'✅' if ok_valori else '⛔'}")
    return esito and ok_valori


# ══════════════════════════════ TEST 2: salute del gradiente (3 parametri) ══════════════════════════════
def _prova_gradiente_ternaria(device: str, seed: int = 1) -> bool:
    """Come _prova_gradiente di quant_differenziabile.py, ma per TRE parametri
    (z, log_s, delta_raw): nessun NaN/Inf, gradiente vivo lungo tutto lo sweep
    di beta per OGNI parametro — in particolare delta_raw, che e' il pezzo
    nuovo rispetto al binario e deve ricevere segnale per potersi allenare."""
    print("\n── esperimento 3: salute del gradiente su z, log_s, delta_raw ──")
    torch.manual_seed(seed)
    W = (torch.randn(64, 128) * 0.02).to(device)
    mod = TernarioDiff(W).to(device)
    betas = schedula_beta()
    ok = True
    soglia_viva = 1e-8
    for beta in betas:
        mod.zero_grad(set_to_none=True)
        loss = F.mse_loss(mod(beta), W)
        loss.backward()
        sano = True
        vivi = {}
        for nome, p in [("z", mod.z), ("log_s", mod.log_s), ("delta_raw", mod.delta_raw)]:
            g = p.grad
            n_nan_inf = int((~torch.isfinite(g)).sum().item())
            vivi[nome] = (g.abs() > soglia_viva).float().mean().item()
            sano = sano and (n_nan_inf == 0) and (g.abs().max().item() < 1e4)
        print(f"    beta={beta:6.2f}  vivi: z={vivi['z']:.3f} log_s={vivi['log_s']:.3f} "
              f"delta_raw={vivi['delta_raw']:.3f}  ({'✅' if sano else '⛔'})")
        ok = ok and sano
    # a beta massimo, delta_raw deve ANCORA ricevere gradiente non-nullo da
    # qualche parte della batch (altrimenti la soglia non potrebbe mai allenarsi
    # nella via globale) — verificato sul valor medio assoluto, non sulla frazione
    # "viva" per elemento (delta_raw e' per-gruppo: un solo scalare per gruppo,
    # la sua statistica di frazione e' meno informativa che la norma totale).
    mod.zero_grad(set_to_none=True)
    F.mse_loss(mod(betas[-1]), W).backward()
    norma_delta = mod.delta_raw.grad.abs().sum().item()
    delta_vivo = norma_delta > 0.0
    print(f"    a beta={betas[-1]:.2f}: |grad delta_raw|.sum()={norma_delta:.3e}  "
          f"({'✅ soglia ancora allenabile' if delta_vivo else '⛔ soglia morta'})")
    return ok and delta_vivo


# ══════════════════════════════ TEST 3: due fasi (morbida -> dura/STE) ══════════════════════════════
def _prova_due_fasi_ternaria(device: str, seed: int = 2, n_passi_totali: int = 200,
                              frazione_dura: float = 0.2, beta_max: float = 16.0,
                              lr: float = 0.05) -> bool:
    """Come _prova_due_fasi di quant_differenziabile.py: 80% morbida (beta
    anneala), 20% finale dura (STE). In PIU' verifica che, a fine training, il
    modello TERNARIO produca una popolazione di zeri ESATTI non banale — cosa
    che il test gemello binario non puo' verificare per costruzione (non ha
    zone morte)."""
    print("\n── esperimento 4: due fasi (morbida->dura) + emergenza zeri reali ──")
    torch.manual_seed(seed)
    W = (torch.randn(64, 128) * 0.02).to(device)
    mod = TernarioDiff(W).to(device)
    opt = torch.optim.Adam(mod.parameters(), lr=lr)

    n_dura = max(1, int(n_passi_totali * frazione_dura))
    n_morbida = n_passi_totali - n_dura
    gradiente_mai_esploso = True

    def _controlla_finiti():
        nonlocal gradiente_mai_esploso
        for p in mod.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                gradiente_mai_esploso = False

    perdita_iniziale = perdita_fine_morbida = None
    for t in range(n_morbida):
        beta = 1.0 * (beta_max / 1.0) ** (t / max(1, n_morbida - 1))
        opt.zero_grad(set_to_none=True)
        loss = F.mse_loss(mod(beta), W)
        loss.backward()
        _controlla_finiti()
        opt.step()
        if perdita_iniziale is None:
            perdita_iniziale = loss.item()
        perdita_fine_morbida = loss.item()
    print(f"    fase morbida ({n_morbida} passi, beta 1->{beta_max:g}): "
          f"loss {perdita_iniziale:.4e} -> {perdita_fine_morbida:.4e}")

    segno_prec = None
    flip_rates = []
    for _ in range(n_dura):
        opt.zero_grad(set_to_none=True)
        Wq = mod.forward_duro(beta_max)
        loss = F.mse_loss(Wq, W)
        loss.backward()
        _controlla_finiti()
        opt.step()
        with torch.no_grad():
            delta = mod._delta_norm()
            segno_ora = torch.where(mod.z > delta, torch.ones_like(mod.z),
                                     torch.where(mod.z < -delta, -torch.ones_like(mod.z), torch.zeros_like(mod.z)))
            if segno_prec is not None:
                flip_rates.append((segno_ora != segno_prec).float().mean().item())
            segno_prec = segno_ora.clone()
    print(f"    fase dura ({n_dura} passi, STE): flip-rate {flip_rates[0]:.3e} -> {flip_rates[-1]:.3e}")

    with torch.no_grad():
        Wq_finale = mod.forward_duro(beta_max)
        n_zeri = int((Wq_finale == 0.0).sum().item())
        frazione_zeri = n_zeri / Wq_finale.numel()
    print(f"    zeri ESATTI a fine training: {n_zeri}/{Wq_finale.numel()} ({frazione_zeri:.1%})")

    meta = max(1, len(flip_rates) // 2)
    flip_prima = sum(flip_rates[:meta]) / meta
    flip_dopo = sum(flip_rates[meta:]) / max(1, len(flip_rates) - meta)
    loss_scende = perdita_fine_morbida < perdita_iniziale
    codici_si_assestano = flip_dopo <= flip_prima
    zeri_non_banali = frazione_zeri > 0.0   # a differenza del binario, DEVE poter essere > 0
    esito = loss_scende and codici_si_assestano and zeri_non_banali and gradiente_mai_esploso
    print(f"    gradiente sempre finito: {'✅' if gradiente_mai_esploso else '⛔ NO — visto NaN/Inf'}")
    print(f"    {'✅' if esito else '⛔'} loss scende={loss_scende}, "
          f"codici si assestano={codici_si_assestano}, zeri non banali={zeri_non_banali}")
    return esito


def _test(device: str = "cpu") -> bool:
    print(f"quant_ternaria_globale.py — fusione SegnoDiff+CAT-Q, device={device}")
    print(f"GROUP={GROUP} (fonte gptq_1024_lib) · _f_transizione riusata da gptq_1024_softern")
    r1 = _prova_convergenza_ternaria(device)
    r2 = _prova_zeri_veri(device)
    r3 = _prova_gradiente_ternaria(device)
    r4 = _prova_due_fasi_ternaria(device)
    ok = r1 and r2 and r3 and r4
    print(f"\n{'✅ TUTTE LE PROVE PASSANO' if ok else '⛔ ALMENO UNA PROVA FALLISCE'} "
          f"(firma_ternaria={r1}, zeri_veri={r2}, gradiente={r3}, due_fasi={r4})")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                     help="cuda = ROCm-as-cuda su venv-catq; prova MINUSCOLA (~8k pesi, <1s), "
                          "memoria liberata subito a fine script.")
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⛔ --device cuda richiesto ma torch non vede una GPU (serve venv-catq ROCm). Uso cpu.")
        args.device = "cpu"
    esito = _test(device=args.device)
    if args.device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        print(f"VRAM allocata dopo pulizia: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
    raise SystemExit(0 if esito else 1)
