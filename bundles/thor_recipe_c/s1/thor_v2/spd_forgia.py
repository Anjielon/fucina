#!/usr/bin/env python3
"""SELECTIVE PROJECTION DECAY nella forgia (THOR_SPD_LAMBDA, THOR_SPD_SEMPRE, THOR_SPD_GRANULARITA) — 26/9/2026.

## Perche'

Fatti misurati su Thor-T (35B-A3B MoE interamente ternario, 8,69 GiB): MMLU della base senza forgia globale
(i latenti di partenza) 75,55 > forgia B 71,95 > forgia A (300 passi) 69,10 > B-150 65,85, mentre la PPL
wikitext migliora (9,43-9,49 contro 10,04 della v20). Piu' la forgia si adatta alla calibrazione, piu' si
allontana dal punto di partenza e perde conoscenza.

L'ancora come penalita' L2 DENTRO la perdita (ancora_forgia.py, THOR_ANCORA_RAPPORTO) e' FALLITA il 26/9
(commit 4b4553f, risultati/verifica_ancora.md): nella ricetta di produzione (fase dura STE + AdamW) non
riavvicina i pesi. Sotto Adam il passo vale ~lr·segno per parametro qualunque sia λ (Loshchilov-Hutter:
L2 nella perdita sotto Adam non e' un decadimento), e sulla GPU i codici cambiati salgono del 5-22%.
La cura disaccoppiata, dalla letteratura, e' questa: si agisce sui PARAMETRI dopo il passo, non sul gradiente.

## L'algoritmo (Tian et al., «Rethinking Weight Decay for Robust Fine-Tuning of Foundation Models», NeurIPS
## 2024, arXiv 2411.01713, Alg. 2, eq. 3, 4, 9; codice ufficiale GT-RIPL/Selective-Projection-Decay adamSPD.py)

Per ogni UNITA' (vedi §Granularita'), con θ0 = i latenti all'inizio del chunk:

    PRIMA di opt.step():  g = p.grad · d = p - θ0 · c = -(g · d)            (eq. 3: la condizione)
                          γ_prev = ||d||                                      (= ||θ_{t-1} - θ0||)
    opt.step()          -> p~                                                  (AdamW della forgia, invariato)
    DOPO:  se c < 0:  γ = ||p~ - θ0|| · r = max(0, γ - γ_prev) / γ              (eq. 9)
                      p <- p~ - λ·r·(p~ - θ0)                                   (eq. 4)

c < 0 vuol dire che il gradiente spinge VERSO θ0 la deviazione accumulata (la direzione di discesa -g e'
contraria a d): li' la regolarizzazione non contrasta la perdita e si applica; dove c >= 0 la perdita chiede
di allontanarsi e il parametro resta libero. r e' la crescita relativa della deviazione in QUESTO passo: con
λ = 1 la proiezione riporta ||p - θ0|| esattamente a γ_prev (la deviazione non cresce), con λ < 1 la frena,
con λ > 1 la contrae (l'articolo: 0 < λ <= 1 debole, λ > 1 forte; valori usati 0,5-2,2, «partire da λ = 1»).
c e γ_prev sono scalari (uno per unita') calcolati PRIMA del passo: nessuna copia di p.
Casi limite (non trattati nell'articolo, dichiarati): γ = 0 (p~ = θ0) -> r = 0, niente da proiettare (il
codice ufficiale li' darebbe 0/0); al primo passo del chunk d = 0, quindi c = 0, non < 0: niente proiezione
(stesso comportamento dell'articolo e del codice ufficiale).

THOR_SPD_SEMPRE=1 (controllo): proiezione senza la condizione c < 0, con lo stesso r — la L2-SP
disaccoppiata con il rapporto dell'articolo. Serve a separare l'effetto della SELEZIONE da quello del
decadimento. ⚠️ Con λ = 1 e' DEGENERE: al primo passo del chunk γ_prev = 0, r = 1 e il passo si annulla
(p = θ0), e cosi' ogni passo dopo — i pesi non si muovono (misurato sul giocattolo il 26/9: distanza finale
1e-11..1e-12, zero codici cambiati, in ogni finestra e in entrambe le ricette). Come controllo va usato con
λ < 1 (a λ 0,5 sul giocattolo di produzione: distanza 0,53x di λ = 0, codici cambiati 16-28 contro 29-46).

## θ0

I latenti allenabili del chunk — TUTTI i parametri passati all'ottimizzatore (z, log_s, delta_raw di gate_up e
di down per ogni strato del chunk; il costruttore verifica che siano esattamente quelli dell'AdamW) —
fotografati prima del primo passo, dopo l'init (teacher + LS per lo strato entrante, stato della finestra
precedente per lo strato portato). Come chiede il compito del 26/9: l'init da cui la forgia parte in QUESTO
chunk. ⚠️ Conseguenza da tenere presente: lo strato portato riparte da una θ0 nuova a ogni finestra (la sua
fine della finestra precedente), quindi in due finestre puo' allontanarsi due volte dai latenti dell'init LS.

## Granularita' (THOR_SPD_GRANULARITA = esperto | tensore)

Il codice ufficiale lavora per TENSORE dell'ottimizzatore («per layer» nell'articolo: un tensore = una matrice di
un Linear). Qui un tensore impila i 256 esperti di uno strato (z di gate_up: 256 x 1024 x 2048): e' il modo in
cui Qwen3.5-MoE FONDE esperti che nella forma non fusa (Qwen3-MoE/Mixtral di HF: un nn.Linear per esperto)
sarebbero 256 tensori separati, cioe' 256 «layer» per l'algoritmo ufficiale.
- `tensore`: c, γ, γ_prev sommati su tutti gli esperti del tensore; UNA decisione e UN r per tensore.
- `esperto`: una decisione e un r per esperto (asse 0 di ogni tensore); gate e up dello stesso esperto
  restano UNA unita' (la forgia li ottimizza come UNA famiglia fusa, come l'ancora).
Il costo e' lo stesso (le riduzioni per esperto si fanno comunque: il valore per tensore e' la loro somma).
Per misurare quanto la scelta conta, il log riporta SEMPRE entrambe le viste: frazione di tensori con c < 0,
frazione di esperti con c < 0 e frazione di esperti il cui segno e' in DISACCORDO con quello del loro tensore
(la parte di selettivita' che la granularita' per tensore perde).
DEFAULT `esperto`, per MISURA (test_spd.py, forgia giocattolo vera a 4 esperti, λ = 1, 26/9):
- il 12-21% degli esperti ha il segno di c OPPOSTO a quello del proprio tensore (in ogni finestra, entrambe le
  ricette): per tensore quegli esperti vengono proiettati quando la loro perdita chiede di allontanarsi, o lasciati
  liberi quando chiede di tornare — l'opposto della selezione dell'articolo;
- a parita' di λ, ricetta di produzione, finestre da 40 passi: per esperto distanza finale 0,93-0,95x di λ = 0 e
  codici cambiati minori in OGNI finestra (446 contro 480 in totale); per tensore 0,97-0,99x e 475 contro 480 (alla
  finestra 0 piu' codici: 152 contro 149). A 9 passi stesso ordine (0,85-0,92x contro 0,92-0,97x).
Su Thor (256 esperti per tensore invece di 4) la misura la da' la prima corsa su GPU: righe cronometro e FINALE.

## Memoria (dal formato, ESATTA per i tensori; il picco va MISURATO sulla GPU)

θ0 si tiene in fp32 sullo STESSO device dei parametri (come il codice ufficiale: `pre` e' una copia dei
parametri). Per strato di Thor (hidden 2048, 256 esperti, intermedio 512, gruppi 128/256):
  z gate_up 256x1024x2048 fp32 = 2 GiB · z down 256x2048x512 = 1 GiB · log_s + delta_raw 40 MiB
  -> 3,039 GiB per strato, 6,078 GiB per chunk a W = 2 (residenti dal primo passo a fine ciclo, poi liberati
  PRIMA delle valutazioni finali del chunk). Transitorio: un blocco di BLOCCO_ESPERTI esperti (256 MiB per z di
  gate_up) per la differenza p - θ0, piu' le riduzioni per esperto (KiB).
Perche' non un dtype piu' stretto: in bf16 θ0 si sposta fino a 2^-8·|z| (~0,4·lr per |z| ~ 1 a lr 0,01): la
proiezione tirerebbe anche i parametri FERMI (p = θ0 esatta, d = 0) verso il punto arrotondato, e vicino alla
soglia cambierebbe codici — cioe' proprio la grandezza che SPD deve ridurre. MISURATO (test_spd.py, 26/9, λ = 1,
θ0 arrotondata a bf16 alla fotografia): ricetta base 95 codici finali su 2880 diversi da θ0 fp32 e codici cambiati
per finestra 141/136/148/118 contro 158/145/169/126 — una differenza PIU' GRANDE dell'effetto di SPD stesso
(158/145/169/126 contro 167/156/172/136 senza SPD); produzione 15 su 2880, per finestra da -4 a +3. Non trascurabile.
Perche' non fuori dalla GPU: RAM di sistema ~31 GiB condivisa con la casa (MemAvailable misurata 3-18 GiB il
26/9 durante la forgia); da disco servirebbero ~6 GiB letti per passo (~1 s a passo contro ~1,3 s del passo).

## Costo in tempo (STIMA dal numero di passaggi in memoria, da MISURARE sulla GPU)

Prima del passo: differenza (leggi p, θ0, scrivi d), norma (leggi d), prodotto con g sul posto (leggi d, g,
scrivi d), somma (leggi d). Dopo: differenza, norma, proiezione (leggi p, d, scrivi p). ~15 volte i 6,08 GiB
per passo ≈ 90 GiB di traffico: ~0,4-0,5 s a passo a 200 GB/s. In modo `tensore` la proiezione e' un
secondo giro solo sui tensori con r > 0.

## Dove si aggancia

`passo_ottimizzazione` (finestra_globale.py), argomento `proiezione`: `prima_del_passo()` subito prima di
`opt.step()` (dopo il backward, dopo il gancio dell'ancora e dopo il controllo dei non finiti, quindi c vede il
gradiente TOTALE che il passo usera') e `dopo_il_passo()` subito dopo. Un passo saltato (non finiti) o un
micro-passo di accumulo non chiama nulla. Costruita e chiusa da `ottimizza_finestra_JOINT`
(valida_finestra_8_12.py) per ogni chunk. THOR_SPD_LAMBDA=0 (default): nessun oggetto, nessuna fotografia,
nessun calcolo: la forgia resta identica al bit a 4b4553f (test_spd.py, prova (a)).
"""
from __future__ import annotations

import math
import os
from typing import Callable

import torch

VAR_LAMBDA = "THOR_SPD_LAMBDA"
VAR_SEMPRE = "THOR_SPD_SEMPRE"
VAR_GRANULARITA = "THOR_SPD_GRANULARITA"
GRANULARITA = ("esperto", "tensore")
GRANULARITA_DEFAULT = "esperto"
# esperti per blocco nei giri su θ0: 32 su 256 tiene il transitorio a 256 MiB (z di gate_up). Costante di codice:
# cambia solo l'ordine dei blocchi, non le riduzioni (ogni esperto sta per intero in un blocco).
BLOCCO_ESPERTI = 32
NOMI_PARAMETRI = ("z", "log_s", "delta_raw")
# campi della statistica di un passo (un vettore float64 sul device: nessuna sincronizzazione per passo)
_CAMPI = ("tensori", "tensori_neg", "esperti", "esperti_neg", "disaccordo", "unita", "unita_cond",
          "somma_r_cond", "unita_proiettate", "dist2", "norma02")


def configurazione() -> "tuple[float, bool, str]":
    """(λ, sempre, granularita') dall'ambiente, letti a OGNI chiamata (un chunk alla volta). Valori non validi:
    errore rumoroso, mai un ripiego. SEMPRE o GRANULARITA impostate con λ = 0 sono rifiutate: non farebbero
    nulla ma cambierebbero il nome della ricetta (la ripresa le confronta)."""
    testo = os.getenv(VAR_LAMBDA, "0")
    try:
        lam = float(testo)
    except ValueError:
        raise ValueError(f"{VAR_LAMBDA}={testo!r}: atteso un numero >= 0") from None
    if not math.isfinite(lam) or lam < 0:
        raise ValueError(f"{VAR_LAMBDA}={testo!r}: atteso un numero finito >= 0")
    s = os.getenv(VAR_SEMPRE, "0")
    if s not in ("0", "1"):
        raise ValueError(f"{VAR_SEMPRE}={s!r}: attesi '0' o '1'")
    g_env = os.getenv(VAR_GRANULARITA)
    g = GRANULARITA_DEFAULT if g_env is None else g_env
    if g not in GRANULARITA:
        raise ValueError(f"{VAR_GRANULARITA}={g!r}: attese {GRANULARITA}")
    if lam == 0 and (s == "1" or g_env is not None):
        raise ValueError(f"{VAR_SEMPRE}/{VAR_GRANULARITA} impostate con {VAR_LAMBDA}=0: SPD e' spenta e non farebbero "
                         "nulla (ma entrerebbero nella ricetta registrata). Toglierle o accendere SPD con λ > 0.")
    return lam, s == "1", g


def descrizione(lam: float, sempre: bool, granularita: str) -> str:
    return (f"λ {lam:g} · granularita' {granularita} · "
            + ("SEMPRE (senza la condizione c<0: controllo L2-SP disaccoppiata)" if sempre else "condizione c<0"))


def _blocchi(E: int):
    for e0 in range(0, E, BLOCCO_ESPERTI):
        yield e0, min(E, e0 + BLOCCO_ESPERTI)


def _norme_righe(x: torch.Tensor) -> torch.Tensor:
    """||x_e|| per esperto (asse 0), in float64 (la norma si calcola nel dtype di x, fp32)."""
    return torch.linalg.vector_norm(x.reshape(x.shape[0], -1), dim=1).double()


def coefficienti(gam: torch.Tensor, gam_prev: torch.Tensor, c: torch.Tensor, lam: float,
                 sempre: bool) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
    """(k = λ·r·cond, r, cond) per unita' (tensori float64 della stessa forma). r = max(0, γ - γ_prev)/γ
    (eq. 9), 0 dove γ = 0. cond = c < 0 (eq. 3), o sempre vera con THOR_SPD_SEMPRE."""
    cond = torch.ones_like(c, dtype=torch.bool) if sempre else (c < 0)
    r = torch.where(gam > 0, (gam - gam_prev).clamp_min(0) / gam, torch.zeros_like(gam))
    return lam * r * cond.to(r.dtype), r, cond


class _Voce:
    """Un tensore dell'ottimizzatore: il parametro, la sua θ0 (copia esatta) e le grandezze del passo."""

    def __init__(self, L: int, fam: str, nome: str, p: torch.nn.Parameter):
        self.etichetta = f"L{L}.{fam}.{nome}"
        self.L, self.fam, self.nome, self.p = L, fam, nome, p
        with torch.no_grad():
            self.p0 = p.detach().clone()                 # stesso dtype, stesso device: θ0 esatta
            self.norma02_e = _norme_righe(self.p0).square()
        self.E = p.shape[0]
        self.attiva = False
        self.gam_prev_e = self.c_e = None


class ProiezioneSPD:
    """SPD di UN chunk. Costruita dopo l'init dei ternari e dopo l'AdamW, prima del primo passo.

    `ternari_gu`/`ternari_dw`: {L: TernarioDiffBatchFuso | None} del chunk. `opt`: l'ottimizzatore del chunk;
    i suoi parametri devono essere ESATTAMENTE quelli dei ternari (controllo per identita')."""

    def __init__(self, lam: float, sempre: bool, granularita: str, chunk: list, ternari_gu: dict,
                 ternari_dw: dict, opt: torch.optim.Optimizer, log: Callable[..., None] = print):
        if not (lam > 0 and math.isfinite(lam)):
            raise ValueError(f"ProiezioneSPD: λ={lam} (serve un numero finito > 0; λ = 0 e' la forgia senza SPD)")
        if granularita not in GRANULARITA:
            raise ValueError(f"ProiezioneSPD: granularita' {granularita!r}, attese {GRANULARITA}")
        self.lam, self.sempre, self.granularita, self.log = float(lam), bool(sempre), granularita, log
        self.chunk = list(chunk)
        self.voci: list = []
        self.oggetti: dict = {}                          # (L, fam) -> TernarioDiffBatchFuso (per i codici duri)
        for L in self.chunk:
            for fam, obj in (("gu", ternari_gu[L]), ("dw", (ternari_dw or {}).get(L))):
                if obj is None:
                    continue
                self.oggetti[(L, fam)] = obj
                for nome in NOMI_PARAMETRI:
                    self.voci.append(_Voce(L, fam, nome, getattr(obj, nome)))
        if not self.voci:
            raise ValueError(f"ProiezioneSPD: nessun ternario nel chunk {self.chunk}")
        nostri = {id(v.p) for v in self.voci}
        dell_opt = [p for g in opt.param_groups for p in g["params"]]
        if nostri != {id(p) for p in dell_opt} or len(dell_opt) != len(self.voci):
            raise RuntimeError(f"ProiezioneSPD: i parametri dell'ottimizzatore ({len(dell_opt)}) non sono esattamente "
                               f"i latenti dei ternari del chunk ({len(self.voci)}): θ0 non coprirebbe TUTTO cio' che "
                               "il passo muove")
        dev = self.voci[0].p.device
        self._dev = dev
        self.norma02 = float(sum(v.norma02_e.sum() for v in self.voci))
        if not (self.norma02 > 0 and math.isfinite(self.norma02)):
            raise ValueError(f"ProiezioneSPD: ||θ0||² = {self.norma02} (serve finito > 0)")
        self.byte_foto = sum(v.p0.numel() * v.p0.element_size() for v in self.voci)
        self.t: "int | None" = None
        self._storia: list = []                          # (t, vettore _CAMPI) di ogni passo proiettato
        self._dal_crono = 0
        self.log(f"  SPD sul chunk {self.chunk}: {descrizione(self.lam, self.sempre, self.granularita)} · "
                 f"fotografia dei latenti di partenza θ0: {len(self.voci)} tensori, "
                 f"{self.byte_foto / 2**30:.3f} GiB {self.voci[0].p0.dtype} su {dev}")

    # ── il passo ──
    def al_passo(self, t: int) -> "ProiezioneSPD":
        """Il passo t del chunk (solo per la storia): ritorna se stessa, da passare a passo_ottimizzazione."""
        self.t = t
        return self

    @torch.no_grad()
    def prima_del_passo(self) -> None:
        """c = -(g·(p - θ0)) e γ_prev = ||p - θ0|| per esperto, PRIMA di opt.step() (p = θ_{t-1})."""
        for v in self.voci:
            g = v.p.grad
            # come AdamW (e come adamSPD.py: `if param.grad is None: continue`): senza gradiente il passo non
            # muove il parametro, e SPD non lo tocca
            v.attiva = g is not None
            if g is None:
                continue
            gam = torch.empty(v.E, dtype=torch.float64, device=self._dev)
            c = torch.empty(v.E, dtype=torch.float64, device=self._dev)
            for e0, e1 in _blocchi(v.E):
                d = v.p[e0:e1] - v.p0[e0:e1]
                gam[e0:e1] = _norme_righe(d)
                d.mul_(g[e0:e1])                          # g·d sul posto: nessun secondo transitorio
                c[e0:e1] = -d.reshape(e1 - e0, -1).sum(dim=1).double()
                del d
            v.gam_prev_e, v.c_e = gam, c

    @torch.no_grad()
    def dopo_il_passo(self) -> None:
        """γ = ||p~ - θ0||, r, e la proiezione p <- p~ - λ·r·(p~ - θ0) dove vale la condizione."""
        stat = torch.zeros(len(_CAMPI), dtype=torch.float64, device=self._dev)
        i = {k: j for j, k in enumerate(_CAMPI)}
        for v in self.voci:
            if not v.attiva:
                continue
            gam = torch.empty(v.E, dtype=torch.float64, device=self._dev)
            forma_k = (-1,) + (1,) * (v.p.dim() - 1)
            if self.granularita == "esperto":
                k_e = torch.empty(v.E, dtype=torch.float64, device=self._dev)
                r_e = torch.empty_like(k_e)
                cond_e = torch.empty(v.E, dtype=torch.bool, device=self._dev)
                for e0, e1 in _blocchi(v.E):
                    d = v.p[e0:e1] - v.p0[e0:e1]
                    gam[e0:e1] = _norme_righe(d)
                    k, r, cond = coefficienti(gam[e0:e1], v.gam_prev_e[e0:e1], v.c_e[e0:e1], self.lam, self.sempre)
                    # p~ - k·d, un esperto alla volta (k = 0: p + (-0) = p, invariato)
                    v.p[e0:e1].addcmul_(d, k.to(v.p.dtype).view(forma_k), value=-1.0)
                    k_e[e0:e1], r_e[e0:e1], cond_e[e0:e1] = k, r, cond
                    del d
                stat[i["unita"]] += v.E
                stat[i["unita_cond"]] += cond_e.sum()
                stat[i["somma_r_cond"]] += (r_e * cond_e).sum()
                stat[i["unita_proiettate"]] += (k_e > 0).sum()
                stat[i["dist2"]] += ((1 - k_e) * gam).square().sum()        # ||p - θ0||² dopo la proiezione
            else:
                for e0, e1 in _blocchi(v.E):
                    d = v.p[e0:e1] - v.p0[e0:e1]
                    gam[e0:e1] = _norme_righe(d)
                    del d
                G = gam.square().sum().sqrt()
                Gp = v.gam_prev_e.square().sum().sqrt()
                k, r, cond = coefficienti(G, Gp, v.c_e.sum(), self.lam, self.sempre)
                kf = float(k)                    # una sincronizzazione per tensore: il secondo giro solo se serve
                if kf != 0.0:
                    kt = k.to(v.p.dtype)
                    for e0, e1 in _blocchi(v.E):
                        d = v.p[e0:e1] - v.p0[e0:e1]
                        v.p[e0:e1].addcmul_(d, kt, value=-1.0)
                        del d
                stat[i["unita"]] += 1
                stat[i["unita_cond"]] += cond.to(torch.float64)
                stat[i["somma_r_cond"]] += r * cond
                stat[i["unita_proiettate"]] += float(kf > 0)
                stat[i["dist2"]] += (1 - k).square() * gam.square().sum()
            # le due viste, sempre (misura della granularita'): segno del tensore e segno di ogni esperto
            neg_t = v.c_e.sum() < 0
            neg_e = v.c_e < 0
            stat[i["tensori"]] += 1
            stat[i["tensori_neg"]] += neg_t.to(torch.float64)
            stat[i["esperti"]] += v.E
            stat[i["esperti_neg"]] += neg_e.sum()
            stat[i["disaccordo"]] += (neg_e != neg_t).sum()
            stat[i["norma02"]] += v.norma02_e.sum()
            v.gam_prev_e = v.c_e = None
        self._storia.append((self.t, stat))

    # ── misure ──
    @staticmethod
    def _riassunto(stat: dict) -> str:
        def pc(a, b):
            return f"{100 * a / b:.1f}%" if b else "n/d"
        r_medio = (f"{stat['somma_r_cond'] / stat['unita_cond']:.3e}" if stat["unita_cond"] else "n/d")
        return (f"c<0 tensori {pc(stat['tensori_neg'], stat['tensori'])} · "
                f"esperti {pc(stat['esperti_neg'], stat['esperti'])}"
                f" (in disaccordo col segno del loro tensore {pc(stat['disaccordo'], stat['esperti'])}) · condizione "
                f"vera {pc(stat['unita_cond'], stat['unita'])} delle unita', r medio {r_medio} · proiettate (r>0) "
                f"{pc(stat['unita_proiettate'], stat['unita'])}")

    def _somma(self, voci_storia: list) -> dict:
        if not voci_storia:
            return {}
        tot = torch.stack([s for _, s in voci_storia]).sum(0).tolist()
        return dict(zip(_CAMPI, tot))

    def riga_crono(self) -> str:
        """Coda della riga del cronometro: statistiche dei passi dall'ultima riga, distanza a fine ultimo passo."""
        nuovi = self._storia[self._dal_crono:]
        self._dal_crono = len(self._storia)
        if not nuovi:
            return " · SPD: nessun passo arrivato a opt.step()"
        s = self._somma(nuovi)
        ultimo = dict(zip(_CAMPI, nuovi[-1][1].tolist()))
        dist = math.sqrt(ultimo["dist2"] / ultimo["norma02"]) if ultimo["norma02"] else float("nan")
        return (f" · SPD ({len(nuovi)} passi): {self._riassunto(s)} · distanza ||p-θ0||/||θ0|| {dist:.4e}")

    @torch.no_grad()
    def chiudi(self) -> dict:
        """Distanza FINALE dai latenti di partenza (giro esatto su p e θ0) e frazione dei codici DURI cambiati
        rispetto all'init del chunk, su TUTTI gli esperti (stessa regola di _estrai_famiglia: z > δ -> +1,
        z < -δ -> -1, δ = sigmoid(delta_raw)·delta_max). Poi libera θ0 e i riferimenti ai parametri."""
        dist2_voce, norma02_voce = {}, {}
        for v in self.voci:
            acc = torch.zeros((), dtype=torch.float64, device=self._dev)
            for e0, e1 in _blocchi(v.E):
                acc += _norme_righe(v.p[e0:e1] - v.p0[e0:e1]).square().sum()
            dist2_voce[v.etichetta] = float(acc)
            norma02_voce[v.etichetta] = float(v.norma02_e.sum())
        per_nome = {n: math.sqrt(sum(x for k, x in dist2_voce.items() if k.endswith("." + n))
                                 / max(sum(x for k, x in norma02_voce.items() if k.endswith("." + n)), 1e-300))
                    for n in NOMI_PARAMETRI}
        dist = math.sqrt(sum(dist2_voce.values()) / self.norma02)
        voce_di = {(v.L, v.fam, v.nome): v for v in self.voci}
        codici, tot_c, tot_n = {}, 0, 0
        for (L, fam), obj in self.oggetti.items():
            vz, vd = voce_di[(L, fam, "z")], voce_di[(L, fam, "delta_raw")]
            cambiati = torch.zeros((), dtype=torch.int64, device=self._dev)
            for e0, e1 in _blocchi(vz.E):
                d0 = torch.sigmoid(vd.p0[e0:e1]) * obj.delta_max
                d1 = torch.sigmoid(vd.p[e0:e1]) * obj.delta_max
                z0, z1 = vz.p0[e0:e1], vz.p[e0:e1]
                cambiati += ((z0 > d0) != (z1 > d1)).logical_or_((z0 < -d0) != (z1 < -d1)).sum()
                del d0, d1
            n = vz.p.numel()
            codici[f"L{L}.{fam}"] = int(cambiati) / n
            tot_c += int(cambiati)
            tot_n += n
        s = self._somma(self._storia)
        storia = [(t, dict(zip(_CAMPI, x.tolist()))) for t, x in self._storia]
        misure = {"lambda": self.lam, "sempre": self.sempre, "granularita": self.granularita,
                  "passi": len(self._storia), "distanza_fine": dist, "distanza_fine_per_tipo": per_nome,
                  "distanza_fine_per_voce": {k: math.sqrt(dist2_voce[k] / norma02_voce[k]) if norma02_voce[k] else 0.0
                                             for k in dist2_voce},
                  "codici_cambiati": tot_c / tot_n, "codici_cambiati_per_voce": codici,
                  "totali": s, "storia": storia, "byte_fotografia": self.byte_foto}
        self.log(f"  SPD ({descrizione(self.lam, self.sempre, self.granularita)}) sul chunk {self.chunk}: "
                 f"distanza FINALE "
                 f"dai latenti di partenza {dist:.4e} (||p-θ0||/||θ0||, tutti i tensori) · z {per_nome['z']:.4e} · "
                 f"log_s {per_nome['log_s']:.4e} · delta_raw {per_nome['delta_raw']:.4e} · codici duri cambiati "
                 f"rispetto all'init del chunk {100 * tot_c / tot_n:.4f}% (tutti gli esperti: "
                 + " · ".join(f"{k} {100 * x:.4f}%" for k, x in codici.items()) + f") · {len(self._storia)} passi: "
                 + (self._riassunto(s) if s else "nessun passo"))
        self.libera()
        return misure

    def libera(self) -> None:
        """Scioglie θ0 (6,08 GiB su Thor a W = 2) e ogni riferimento ai parametri e agli oggetti del chunk."""
        for v in self.voci:
            v.p = v.p0 = v.norma02_e = v.gam_prev_e = v.c_e = None
        self.voci = []
        self.oggetti = {}
