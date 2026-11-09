#!/usr/bin/env python3
"""ANCORA ai pesi di riferimento nella forgia (THOR_ANCORA_RAPPORTO, THOR_ANCORA_A) — 26/9/2026.

## Perche'

Fatti misurati su Thor-T (35B-A3B MoE interamente ternario, 8,69 GiB): MMLU della base senza
forgia globale 75,55 > forgia B (150 passi, 32 lotti) 71,95 > forgia A (300 passi) 69,10 >
B-150 (150 lotti della stessa miscela) 65,85, mentre la PPL wikitext MIGLIORA con la forgia
(9,43-9,49 contro 10,04 della v20). Sul testo MMLU l'errore per strato cresce in tutti i 40
strati, soprattutto negli strati 1-6, e l'held-out di calibrazione (cot, stessa famiglia) NON
lo predice (B-150: -14% sull'held-out mentre l'MMLU perdeva 6 punti).

Diagnosi: la perdita della forgia (Lemma 1 sull'uscita della finestra, lotti di calibrazione)
misura l'errore SOLO nelle direzioni che la calibrazione attiva; le altre (la conoscenza delle
57 materie) valgono zero e l'ottimizzatore le spende gratis. Cura nota in letteratura: un'ANCORA
ai pesi di riferimento — lo smorzamento (damping) di GPTQ, L2-SP (Li et al. 2018, ancora al
punto di partenza), EWC con Fisher identita' —

    perdita = perdita_dati + λ · A
    A = Σ_{L nella finestra} Σ_{fam ∈ gate_up, down} ||deq(θ_{L,fam}) - W_rif||²_F / ||W_rif||²_F

gate/up e' UNA famiglia (il tensore fuso gate_up, come lo ottimizza la forgia), down l'altra.

## λ (THOR_ANCORA_RAPPORTO = r)

Al primo passo di ogni chunk (di norma t = 0) con L0 = perdita dati DERIVATA di quel passo
(loss_somma: e' lei che va nel backward) e A0 = ancora ai parametri di quel passo (prima
dell'aggiornamento): λ = r · L0 / A0, fisso per tutto il chunk. Cosi' al passo 0 il termine
d'ancora vale esattamente r volte il termine dati, qualunque sia la scala della finestra (la
perdita dati cresce di ordini di grandezza con la profondita'). r = 0 (default) = spento: la
forgia non costruisce nulla di questo modulo e resta identica al bit a c316fe3.
Se il passo 0 viene saltato (perdita non finita, THOR_SALTA_NON_FINITI) λ si fissa al primo
passo che arriva al backward, e il log dice quale.

## Riferimento (THOR_ANCORA_A)

- `teacher` (default): i pesi originali bf16 (`strati_teacher`), NELLO STESSO SPAZIO in cui la
  forgia ottimizza quella famiglia: gate/up ruotato con la STESSA funzione di W_gu_rif
  dell'init LS (`_ruota_se_possibile` di valida_finestra_8_12, passata qui come `ruota_gu`),
  down non ruotato. Nessuna copia residente: il riferimento di un blocco di esperti si
  ricalcola dal parametro del teacher a ogni uso (niente 2 GiB fp32 in piu' per strato).
- `init`: i pesi dequantizzati all'inizio del chunk (dopo l'init LS o la ripartenza dello
  strato portato): exp(log_s0) · Q0, fotografati qui (Q0 int8 + scale: 0,75 GiB per strato).
  ⛔ Con la fase dura dal passo 0 (THOR_VALIDA_FRAZIONE_DURA=1.0, la ricetta di produzione)
  i pesi al passo 0 SONO quelli dell'inizializzazione: A0 = 0 per costruzione e λ = r·L0/A0
  non e' definito. Rifiuto rumoroso (`verifica_ricetta`), mai una normalizzazione inventata.

## Gradiente

Passa dallo STESSO nucleo compilato del dato (quant_ternaria_globale_batch_v2): nella fase
dura `_nucleo_duro_compilato` (valore duro, gradiente dal surrogato morbido: lo STE), nella
fase morbida `_nucleo_morbido_compilato`, con lo stesso beta del passo. (Il ramo dati in
bf16 usa `_nucleo_duro_compilato_bf16`: stessa formula, uscita arrotondata a bf16; qui il
termine si calcola in fp32.) Si lavora a BLOCCHI di esperti (BLOCCO_ESPERTI): per ogni blocco
foglie staccate che condividono la memoria dei parametri, `torch.autograd.grad`, e il
gradiente del blocco SOMMATO sul posto nella sua fetta di `.grad`. Non si usa il backward di
una fetta del parametro: SliceBackward allocherebbe a ogni blocco un gradiente pieno (2 GiB
per z di gate_up su Thor) per poi sommarlo. Il risultato e' la stessa somma per elemento.

## ⚠️ Misurato sul giocattolo il 26/9 (test_ancora.py, CPU) — leggere prima della prova su GPU

Il gradiente del gancio e' ESATTO (= autograd di dati + λ·A a tensore intero, scarto relativo <= 4e-7,
fase dura e morbida, rotazione attiva). Ma con la sola fase DURA (STE) e AdamW:
- partendo dall'init LS (minimo dell'errore duro sui pesi), la discesa della SOLA ancora col gradiente
  STE fa SALIRE l'ancora dura: AdamW lr 0,01 a larghezza vera 0,1874 -> 0,1974 in 9 passi (lr 0,001:
  0,1874 -> 0,1876; SGD a passo piccolo: 0,1874 -> 0,1875). Lo STE non e' una direzione di discesa li';
- nella forgia giocattolo di produzione (9 passi) l'ancora finale della finestra 0 e' risultata PIU'
  ALTA con r = 1 che con r = 0 (0,663 contro 0,603), e quasi uguale per r da 0,1 a 1000: Adam normalizza
  per parametro, dove l'ancora domina il passo vale ~lr·segno(∇A) qualunque sia λ (la lezione di
  Loshchilov-Hutter: L2 nella perdita sotto Adam non e' un decadimento). Finestre 1-3: fra -1,5% e -2,8%.
- nella ricetta base (fase morbida) invece l'ancora scende nettamente (0,21-0,58x di r = 0).
Quindi r agisce solo dove i due gradienti sono confrontabili, e sui pesi che i dati non muovono
l'ancora li MUOVE (a r = 0 un gradiente nullo lascia il peso dov'e'). La prova su GPU (prova_ancora.sh)
dice cosa ne resta su Thor; un'ancora DISACCOPPIATA (tirare i latenti verso quelli dell'init LS del
teacher dopo il passo di Adam, come il decadimento di AdamW verso un riferimento invece che verso 0)
non ha questo difetto per costruzione — non e' implementata qui.

## Memoria (stima dal codice, da MISURARE alla prima prova su GPU)

Per blocco di 32 esperti di gate_up (32x1024x2048 fp32 = 256 MiB a tensore): ricostruzione,
riferimento (copia fp32 del teacher + rotazione), differenza, gradiente e i salvati del
nucleo per il backward: ~5-6 tensori da 256 MiB transitori, liberati a fine blocco. Nulla di
residente con `teacher`; con `init` la fotografia (0,75 GiB per strato).

Chi lo usa: `ottimizza_finestra_JOINT` (valida_finestra_8_12.py) attraverso il gancio
`dopo_backward` di `passo_ottimizzazione` (finestra_globale.py). Collaudo: test_ancora.py.
"""
from __future__ import annotations

import math
import os
import sys
from typing import Callable

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quant_ternaria_globale_batch_v2 import (  # noqa: E402
    TernarioDiffBatchFuso, _nucleo_duro_compilato, _nucleo_morbido_compilato,
)

VAR_RAPPORTO = "THOR_ANCORA_RAPPORTO"
VAR_RIFERIMENTO = "THOR_ANCORA_A"
RIFERIMENTI = ("teacher", "init")
# esperti per blocco: 32 su 256 (8 blocchi) tiene il transitorio a ~1,5 GiB (vedi §Memoria).
# Costante di codice, non variabile d'ambiente: cambia solo l'ordine delle somme del valore.
BLOCCO_ESPERTI = 32
NOMI_PARAMETRI = ("z", "log_s", "delta_raw")     # ordine degli argomenti dei nuclei compilati


def configurazione() -> "tuple[float, str]":
    """(r, riferimento) dall'ambiente, letti a OGNI chiamata (un chunk alla volta). Valori non
    validi: errore rumoroso, mai un ripiego."""
    testo = os.getenv(VAR_RAPPORTO, "0")
    try:
        r = float(testo)
    except ValueError:
        raise ValueError(f"{VAR_RAPPORTO}={testo!r}: atteso un numero >= 0") from None
    if not math.isfinite(r) or r < 0:
        raise ValueError(f"{VAR_RAPPORTO}={testo!r}: atteso un numero finito >= 0")
    rif = os.getenv(VAR_RIFERIMENTO, "teacher")
    if rif not in RIFERIMENTI:
        raise ValueError(f"{VAR_RIFERIMENTO}={rif!r}: attesi {RIFERIMENTI}")
    return r, rif


def verifica_ricetta(n_morbida: int, r: "float | None" = None, rif: "str | None" = None) -> None:
    """Rifiuta `init` quando il primo passo del chunk e' gia' in fase dura: li' i pesi del passo 0
    sono ESATTAMENTE quelli dell'inizializzazione, A0 = 0 per costruzione e λ = r·L0/A0 non esiste."""
    if r is None or rif is None:
        r, rif = configurazione()
    if r > 0 and rif == "init" and n_morbida == 0:
        raise ValueError(
            f"{VAR_RIFERIMENTO}=init con la fase dura dal passo 0 (THOR_VALIDA_FRAZIONE_DURA=1.0, nessun passo "
            "morbido): al passo 0 i pesi dequantizzati SONO quelli dell'inizializzazione, l'ancora vale 0 per "
            f"costruzione e λ = r·L0/A0 non e' definito. Usare {VAR_RIFERIMENTO}=teacher o una ricetta con passi "
            "morbidi.")


class _Voce:
    """Un termine dell'ancora: (strato, famiglia), l'oggetto ternario e il suo riferimento a blocchi."""

    def __init__(self, etichetta: str, obj: TernarioDiffBatchFuso, rif_blocco: Callable[[int, int], torch.Tensor]):
        self.etichetta = etichetta
        self.obj = obj
        self.rif_blocco = rif_blocco
        self.norma2 = 0.0


class Ancora:
    """L'ancora di UN chunk. Costruita DOPO l'inizializzazione dei ternari e prima del primo passo.

    `ternari_gu`/`ternari_dw`: {L: TernarioDiffBatchFuso | None} del chunk (quelli che l'ottimizzatore
    aggiorna). `pesi_teacher`: {L: (gate_up_proj, down_proj)} originali (solo con riferimento teacher).
    `ruota_gu`: la rotazione dei pesi di gate/up usata dalla forgia (identita' se non si ruota)."""

    def __init__(self, r: float, riferimento: str, chunk: list, ternari_gu: dict, ternari_dw: dict,
                 pesi_teacher: "dict | None", ruota_gu: Callable[[torch.Tensor], torch.Tensor],
                 log: Callable[..., None] = print):
        if not (r > 0 and math.isfinite(r)):
            raise ValueError(f"Ancora: r={r} (serve un numero finito > 0; r = 0 e' la forgia senza ancora)")
        if riferimento not in RIFERIMENTI:
            raise ValueError(f"Ancora: riferimento {riferimento!r}, attesi {RIFERIMENTI}")
        self.r, self.riferimento, self.log = float(r), riferimento, log
        self.chunk = list(chunk)
        self.voci: list = []
        for L in self.chunk:
            for fam, obj in (("gu", ternari_gu[L]), ("dw", (ternari_dw or {}).get(L))):
                if obj is None:
                    continue
                if not isinstance(obj, TernarioDiffBatchFuso):
                    # i nuclei chiamati qui sono quelli di TernarioDiffBatchFuso: un'altra classe potrebbe
                    # ricostruire i pesi con un'altra formula e l'ancora non misurerebbe piu' cio' che vede il dato
                    raise TypeError(f"Ancora: strato {L} {fam}: atteso TernarioDiffBatchFuso, ricevuto {type(obj).__name__}")
                self.voci.append(_Voce(f"L{L}.{fam}", obj, self._riferimento(L, fam, obj, pesi_teacher, ruota_gu)))
        if not self.voci:
            raise ValueError(f"Ancora: nessun ternario nel chunk {self.chunk}")
        for v in self.voci:
            v.norma2 = self._norma2(v)
            if not (v.norma2 > 0 and math.isfinite(v.norma2)):
                raise ValueError(f"Ancora: ||W_rif||² di {v.etichetta} = {v.norma2} (serve finito > 0)")
        self.lam: "float | None" = None
        self.L0 = self.A0 = None
        self.passo_lam: "int | None" = None
        self.ultimo: "dict | None" = None
        self.storia: list = []                    # (t, dati, A) di ogni passo arrivato al backward
        self.finale: "dict | None" = None

    # ── riferimenti ──
    def _riferimento(self, L: int, fam: str, obj, pesi_teacher, ruota_gu):
        E, n_out, n_in = obj.shape
        if self.riferimento == "teacher":
            if pesi_teacher is None or L not in pesi_teacher:
                raise ValueError(f"Ancora teacher: pesi del teacher dello strato {L} assenti")
            W = pesi_teacher[L][0 if fam == "gu" else 1]
            if tuple(W.shape) != tuple(obj.shape):
                raise ValueError(f"Ancora teacher: {fam} dello strato {L} forma {tuple(W.shape)} != ternario "
                                 f"{tuple(obj.shape)}")
            if fam == "gu":
                # STESSA espressione di W_gu_rif in ottimizza_finestra_JOINT (float(), poi la rotazione della forgia)
                return lambda e0, e1, _W=W: ruota_gu(_W[e0:e1].detach().float().to(obj.z.device))
            return lambda e0, e1, _W=W: _W[e0:e1].detach().float().to(obj.z.device)
        # init: fotografia del codice duro e delle scale al passo 0 (stessa regola di _estrai_famiglia)
        with torch.no_grad():
            delta = torch.sigmoid(obj.delta_raw) * obj.delta_max
            Q0 = torch.where(obj.z > delta, 1, torch.where(obj.z < -delta, -1, 0)).to(torch.int8)
            s0 = obj.log_s.exp()
            del delta
        return lambda e0, e1, _Q=Q0, _s=s0: (_s[e0:e1] * _Q[e0:e1].to(torch.float32)).reshape(e1 - e0, n_out, n_in)

    @staticmethod
    def _blocchi(E: int):
        for e0 in range(0, E, BLOCCO_ESPERTI):
            yield e0, min(E, e0 + BLOCCO_ESPERTI)

    def _norma2(self, v: _Voce) -> float:
        acc = torch.zeros((), dtype=torch.float64, device=v.obj.z.device)
        with torch.no_grad():
            for e0, e1 in self._blocchi(v.obj.shape[0]):
                W = v.rif_blocco(e0, e1)
                acc += (W * W).sum().double()
                del W
        return float(acc)

    # ── valore e gradiente ──
    @staticmethod
    def _nucleo(duro: bool):
        return _nucleo_duro_compilato if duro else _nucleo_morbido_compilato

    def _termine(self, v: _Voce, beta: float, duro: bool, coeff: "float | None") -> torch.Tensor:
        """||deq - W_rif||² / ||W_rif||² di una voce (tensore fp64 0-dim sul device). Con `coeff`
        somma nei .grad dei parametri il gradiente di coeff · termine (STE come il dato)."""
        obj = v.obj
        E, n_out, n_in = obj.shape
        nucleo = self._nucleo(duro)
        beta_t = torch.as_tensor(beta, dtype=obj.z.dtype, device=obj.z.device)   # come TernarioDiffBatchFuso
        acc = torch.zeros((), dtype=torch.float64, device=obj.z.device)
        for e0, e1 in self._blocchi(E):
            W = v.rif_blocco(e0, e1)
            if coeff is None:
                with torch.no_grad():
                    deq = nucleo(obj.z[e0:e1], obj.log_s[e0:e1], obj.delta_raw[e0:e1], obj.delta_max, beta_t)
                    d = deq.reshape(e1 - e0, n_out, n_in) - W
                    acc += (d * d).sum().double()
            else:
                foglie = [getattr(obj, n).detach()[e0:e1].requires_grad_(True) for n in NOMI_PARAMETRI]
                deq = nucleo(foglie[0], foglie[1], foglie[2], obj.delta_max, beta_t)
                d = deq.reshape(e1 - e0, n_out, n_in) - W
                s = (d * d).sum()
                gradi = torch.autograd.grad(s * (coeff / v.norma2), foglie)
                with torch.no_grad():
                    for n, g in zip(NOMI_PARAMETRI, gradi):
                        p = getattr(obj, n)
                        if p.grad is None:
                            p.grad = torch.zeros_like(p)
                        p.grad[e0:e1].add_(g)
                acc += s.detach().double()
                del foglie, gradi, s
            del W, deq, d
        return acc / v.norma2

    def valore(self, beta: float, duro: bool) -> "tuple[float, dict]":
        """A ai parametri correnti (senza gradiente) e i termini per voce."""
        termini = {v.etichetta: self._termine(v, beta, duro, None) for v in self.voci}
        valori = {k: float(t) for k, t in termini.items()}
        return sum(valori.values()), valori

    def gancio(self, t: int, beta: float, duro: bool) -> Callable[[torch.Tensor], dict]:
        """Il `dopo_backward` di passo_ottimizzazione per il passo t: fissa λ al primo passo, poi somma
        nei .grad il gradiente di λ·A (stesso beta, stessa fase del dato) e ritorna i due termini."""
        def _dopo_backward(dati: torch.Tensor) -> dict:
            L = float(dati)
            if self.lam is None:
                A0, per_voce = self.valore(beta, duro)
                if not math.isfinite(L):
                    raise FloatingPointError(f"ancora: perdita dati non finita al passo {t} ({L}): λ = r·L0/A0 "
                                             "non si puo' fissare")
                if not math.isfinite(A0):
                    raise FloatingPointError(f"ancora: A0 non finita al passo {t} ({A0})")
                if A0 <= 0:
                    raise ValueError(f"ancora ({self.riferimento}): A0 = {A0} al passo {t}: i pesi del passo {t} "
                                     "coincidono col riferimento, λ = r·L0/A0 non e' definito")
                self.lam, self.L0, self.A0, self.passo_lam = self.r * L / A0, L, A0, t
                self.log(f"  ancora ({self.riferimento}) sul chunk {self.chunk}: λ = r·L0/A0 = {self.r:g} x {L:.4e} / "
                         f"{A0:.4e} = {self.lam:.4e} (fissato al passo {t}) · A0 per voce: "
                         + " · ".join(f"{k} {x:.4e}" for k, x in per_voce.items()))
            A = float(sum(self._termine(v, beta, duro, self.lam) for v in self.voci))
            info = {"t": t, "dati": L, "ancora": A, "lambda": self.lam, "termine": self.lam * A}
            self.ultimo = info
            self.storia.append((t, L, A))
            return info
        return _dopo_backward

    def riga_crono(self) -> str:
        """Coda della riga del cronometro: i due termini della perdita dell'ultimo passo, separati."""
        u = self.ultimo
        if u is None:
            return " · ancora: nessun passo arrivato al backward"
        return (f" · dati {u['dati']:.4e} · ancora λ·A {u['termine']:.4e} (A {u['ancora']:.4e}, "
                f"λ {u['lambda']:.4e}, t={u['t'] + 1})")

    def chiudi(self, beta_duro: float) -> dict:
        """A a fine chunk sui codici DURI (quelli che si committano), log e diagnostica."""
        A, per_voce = self.valore(beta_duro, True)
        self.finale = {"r": self.r, "riferimento": self.riferimento, "lambda": self.lam, "L0": self.L0,
                       "A0": self.A0, "passo_lambda": self.passo_lam, "A_fine": A, "A_fine_per_voce": per_voce,
                       "storia": list(self.storia)}
        rapporto = (f"{A / self.A0:.3f}x" if self.A0 else "n/d")
        self.log(f"  ancora ({self.riferimento}, r {self.r:g}) sul chunk {self.chunk}: ancora FINALE A {A:.4e} "
                 f"(codici duri) · al passo {self.passo_lam}: {self.A0 if self.A0 is None else f'{self.A0:.4e}'} "
                 f"({rapporto}) · λ {self.lam if self.lam is None else f'{self.lam:.4e}'} · per voce: "
                 + " · ".join(f"{k} {x:.4e}" for k, x in per_voce.items()))
        return self.finale
