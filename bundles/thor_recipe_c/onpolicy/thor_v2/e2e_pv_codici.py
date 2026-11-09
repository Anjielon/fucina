#!/usr/bin/env python3
"""Passo V (PV-Tuning) sui CODICI ternari di Thor-T, senza stato per codice — prototipo del progetto
`e2e_progetto_pv_codici.md` (26/9/2026 sera).

File NUOVO. Nessun file esistente e' modificato: si importano in SOLA LETTURA `e2e_studente` (Piano,
DatiFamiglia, motori, `_trit_da_byte`, `trit_esperto`), `e2e_codici_scale` (LAYOUT) e `e2e_campi_scala` (decode
torch dai byte). Supera il latente fp32 per codice (488 GiB) di `e2e_progetto_fase2_codici.md`.

Idea (progetto §1). Nel kernel `_k_dd` dello studente il riquadro `acc` (BR x GS, fp32, registri) vale
sum_s dY[s, r] X_rot[s, c] = dL/dW_rot[r, c] per UNA riga di UN gruppo; oggi lo si riduce in dL/dd e si butta.
Qui lo stesso riquadro, nello stesso kernel (`_k_dd_v`, costante MODO_V), da' anche:
  MODO_V = 0   il kernel di oggi, nient'altro (prova al bit contro `_k_dd`);
  MODO_V = 1   PROPOSTA (micro-lotti 1-2): per riga del riquadro, fra i codici con salto di UN livello fattibile
               (Delta = -sign(acc * d0), t + Delta in {-1,0,1}, d0 != 0) quello di |acc| massimo -> `cand` (uint8,
               colonna nel gruppo) e `prop` (fp32, acc col segno); fra i due micro-lotti vince il |.| maggiore;
  MODO_V = 2   VERIFICA (micro-lotti 3-16, dati DISGIUNTI dalla proposta): v = acc[riga, cand] -> S += v, Q += v^2.
  MODO_V = 3   CURVATURA (27/9, cura del difetto «solo primo ordine»): l'indietro porta un vettore CASUALE con
               covarianza = Hessiana della perdita rispetto alle uscite (Fisher campionato = Gauss-Newton: per la KL
               sui logit g = sqrt(c) (p - e_y), y ~ p dello studente); acc2 = sum_s dY^2 X^2 -> H += acc2[riga, cand]
               = stima NON distorta della diagonale Gauss-Newton sul codice candidato. dL/dd = 0 in questo modo
               (theta.grad non si tocca: e' un indietro di misura, non di allenamento).
Nessuna atomica (un programma per (esperto, riquadro di righe, gruppo)), 17 byte per GRUPPO (cand 1 + prop 4 +
S 4 + Q 4 + H 4), nulla per codice. L'uscita dL/dd e' la stessa nei modi 0-2 (prova al bit in T1).

A fine passo (dopo `opt.step`, il passo P sulle scale): per ogni gruppo con candidato, t di Student sui 14
micro-lotti di verifica (guardia contro il rumore), Benjamini-Hochberg q = 0,1 PER MATRICE; poi il criterio di
SECONDO ORDINE (27/9, revisione: il solo primo ordine faceva SALIRE la perdita con un maestro fuori griglia):
guadagno previsto del salto di ampiezza d = g_passo |d| - H_passo d^2 / 2, con g_passo = (N/n_ver) S (proiettato
sulla direzione della proposta) e H_passo = (N/n_curv) H; si salta solo se > 0 (e' l'arrotondamento al livello
piu' vicino del minimo quadratico locale). Fra i convenienti, per esperto, in ordine di guadagno, finche' la norma
EFFETTIVA dei salti scelti tiene ||Delta W_e|| <= rho ||W_e|| (sum d_g^2 dei scelti <= rho^2 sum_g d_g^2 nnz_g:
trust ratio di PV §4.2, garantito, non piu' in media). Salti di UN livello; le scale (theta, Adam) non si toccano.

I salti si scrivono sui BYTE (in place su `fam.blocchi`, lo stesso tensore che i kernel leggono): q del byte,
q += sum Delta 3^(4-n) (piu' salti nello stesso byte si sommano: cifre diverse, nessun riporto), byte =
ceil(256 q / 243). Ogni salto va nel `Registro` (9 byte: L, f, e, r, c, t_vecchio, t_nuovo): annullamento
(regione di fiducia), rigioco (ripresa = GGUF + registri), export (`patcha_codici` su una COPIA del GGUF).

Cosa c'e':
  1. BufferV / StatoV                  buffer per (strato, famiglia), modo del micro-lotto corrente
  2. _k_dd_v / dd_v_triton             kernel (GPU; su CPU nell'interprete di Triton)
  3. dd_v_torch                        riferimento CON dW materializzato per esperto (CPU, prove)
  4. MotoreTorchV / MotoreTritonV      sottoclassi dei motori dello studente: cambia SOLO `dd`; `installa`
  5. sf_t / p_valori / bh / tau / seleziona
  6. posizione_t / salta_byte / applica_salti / annulla / rigioca / Registro
  7. patcha_codici / verifica_patch_codici (export su una COPIA del GGUF)
  8. RegioneFiducia                    sonda: KL peggiore -> annulla e rho/2; taratura di rho sui primi passi;
                                       27/9 `adattivo`: regione di fiducia vera (r = calo effettivo / previsto)
  9. PassoV                            l'orchestratore (buffer + motori + selezione + salti + registro)
  1b. ConfigBop / StatoBop / seleziona_bop   27/9 «V-Bop a gruppo» (--pv-modo vbop, TECNICHE_E2E_PV.md §21): stato
                                       persistente per gruppo (candidato con isteresi, m e v in EMA, H in EMA, eta'),
                                       salto se guadagno N (m|d| - H d^2/2) > 0 e z = m / sqrt(v / n_eff) passa BH,
                                       budget rho e sonda invariati, correzione alla ECO dopo il salto. Default (bop
                                       None) = PV di prima al bit.
  1c. BufferV.HT / _k_ht / ht_triton / ht_torch   27/9 ramo newton-theta (--ottimizzatore-theta sophia): curvatura
                                       Gauss-Newton delle SCALE per gruppo, HT += sum_s dY^2 (t . X_g)^2, nei micro-lotti
                                       di curvatura del passo (StatoV(curv_theta=True); spenta = nessun buffer, al bit).
  5b. ConfigV.accetta / z_min / z_guadagno_min   27/9 --pv-accetta guadagno (TECNICHE_E2E_PV.md §38): candidato
                                       accettato se il guadagno di secondo ordine stimato e' > 0 (e z > z_min sulla
                                       direzione, t del guadagno > z_guadagno_min), SENZA i due test BH;
                                       ordinamento, budget rho, sonda e rho adattivo invariati. Default "bh" = al bit.
  1d. 28/9 --densi-ternari (e2e_studente.FAMIGLIE_DENSE): le famiglie dense a esperto unico (E = 1) sono in
                                       `StratoEsperti.fam`, quindi `installa` le vede come gli esperti (buffer, motore
                                       V, selezione con budget rho sull'intera matrice, salti, annullamento); nel
                                       registro f = 3 + indice in FAMIGLIE_DENSE, e = 0; `patcha_codici` /
                                       `verifica_patch_codici` leggono i tensori 2D come (1, R, NBR); `ricarica_codici`
                                       ricostruisce gli studenti coi densi dei registri. Senza densi: tutto al bit.
  8b. ConfigDosi / conteggi_dosi / PassoV.seleziona_e_prova_dosi / StatoV.prepara_ricentro /
      RegioneFiducia.esito_dosi   27/9 ramo v-interazione-b (TECNICHE_E2E_PV.md §32.5 azioni 1-2): i salti scelti si
                                       provano a DOSI (prefissi annidati in ordine di guadagno) sulla sonda A, la dose
                                       migliore si conferma sulla sonda B; poi (--pv-ricentra) lascia-uno-fuori dallo
                                       stato raggiunto con UN indietro della sonda A (modo 2 sulle colonne dei salti) e
                                       annullamento dei dannosi a meta' per volta finche' B conferma. La dose parziale
                                       tenuta NON dimezza rho. Metodi nuovi: `seleziona_e_salta` e `esito` invariati
                                       (spento = identico al bit; `applica_salti` = `righe_salti` + `salta_registro`).
  8c. PRIMITIVE_PM / prezzo_punto_medio / scegli_primitive / PassoV._prezzi_punto_medio / ConfigDosi.punto_medio
      29/9 ramo punto-medio (arXiv 2609.09867 Prop. 2 e §6(a); matematica/punto_medio_pv.md): nella sonda a dosi i salti
      scelti si raggruppano in primitive, le pm_max col guadagno previsto piu' alto si prezzano al PUNTO MEDIO PROPRIO
      con la regola del trapezio sugli stati interi (un indietro della sonda per primitiva + uno in q0), e le dosi sono
      prefissi delle primitive col prezzo < 0 (con obiettivo A e vincolo B: `scegli_primitive`); conferma sulla sonda B,
      ricentro e (facoltativa) conferma dell'obiettivo su dati disgiunti. Spento (default): identico al bit.

⛔ CPU e memmap: `famiglia_da_tensore` su un tensore intero (o con la sola fetta di esperti) restituisce byte
che sono una VISTA del memmap del GGUF in sola lettura; scriverci farebbe cadere il processo. Su CPU i salti
si fanno solo su famiglie passate da `famiglia_scrivibile` (copia); sulla GPU i byte sono gia' una copia.

Prove: `e2e_test_pv_codici.py` (T1-T4 obbligatorie + T5 compilazione gfx1151 + T6 studente vero), solo CPU.
"""
from __future__ import annotations

import hashlib
import io
import math
import os
import shutil
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
import torch

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))

import e2e_studente as S  # noqa: E402  (sola lettura)
import e2e_codici_scale as CS  # noqa: E402  (sola lettura)
import e2e_campi_scala as CAMPI  # noqa: E402  (sola lettura)

if S._TRITON:
    import triton
    import triton.language as tl
    _trit_da_byte = S._trit_da_byte          # la STESSA funzione jit del kernel dello studente

FAMIGLIE_IDX = {"gate": 0, "up": 1, "down": 2}
# 28/9 --densi-ternari: le famiglie dense (esperto unico, e = 0) nel registro, con indici DOPO quelli degli esperti (i tre
# di sempre invariati: i registri gia' scritti si rileggono uguali). 3 + posizione in e2e_studente.FAMIGLIE_DENSE.
FAMIGLIE_IDX.update({n: 3 + i for i, n in enumerate(S.FAMIGLIE_DENSE)})
# 29/9 secondo piano dei densi (<nome>2): famiglie PROPRIE nel registro, indici dopo i primi piani (quelli di sempre
# invariati): 3 + len(FAMIGLIE_DENSE) + posizione in e2e_studente.FAMIGLIE_DENSE2.
FAMIGLIE_IDX.update({n: 3 + len(S.FAMIGLIE_DENSE) + i for i, n in enumerate(S.FAMIGLIE_DENSE2)})
# 30/9 terzo piano (<nome>3): famiglie PROPRIE, indici dopo i secondi piani (primi e secondi invariati: i registri gia'
# scritti si rileggono uguali): 3 + 2 len(FAMIGLIE_DENSE) + posizione in e2e_studente.FAMIGLIE_DENSE3 (max 32 < 256, u1).
FAMIGLIE_IDX.update({n: 3 + 2 * len(S.FAMIGLIE_DENSE) + i for i, n in enumerate(S.FAMIGLIE_DENSE3)})
NOMI_IDX = {v: k for k, v in FAMIGLIE_IDX.items()}
P3 = (1, 3, 9, 27, 81)
BYTE_PER_GRUPPO = 17                         # cand uint8 + prop fp32 + S fp32 + Q fp32 + H fp32


# ═══════════════════════════════════ 1. buffer ═══════════════════════════════════
@dataclass
class BufferV:
    """Stato del passo V di UNA famiglia di UNO strato, per gruppo (E, R, NG): nulla per codice.
    cand: colonna del candidato nel gruppo (uint8: GS <= 256); prop: acc della proposta col segno (0 = nessun
    candidato); S, Q: somma e somma dei quadrati di acc[riga, cand] sui micro-lotti di verifica; H: somma di
    acc2[riga, cand] (curvatura Gauss-Newton campionata) sui micro-lotti di curvatura."""
    cand: torch.Tensor
    prop: torch.Tensor
    S: torch.Tensor
    Q: torch.Tensor
    H: torch.Tensor
    # 27/9 ramo newton-theta (--ottimizzatore-theta sophia): curvatura Gauss-Newton delle SCALE per gruppo, senza d^2:
    # HT += sum_s dY_{s,r}^2 u_{s,r,g}^2, u = sum_{c in g} X_{s,c} t_{r,c}, nei micro-lotti di curvatura (modo 3) del
    # PASSO (azzerata a ogni passo, anche dentro una finestra di accumulo). None = spenta (default: nulla cambia).
    HT: Optional[torch.Tensor] = None

    @staticmethod
    def crea(fam: S.DatiFamiglia, device=None, curv_theta: bool = False) -> "BufferV":
        dev = device if device is not None else fam.blocchi.device
        assert fam.gs <= 256, "cand e' uint8"
        forma = (fam.E, fam.R, fam.NG)
        return BufferV(torch.zeros(forma, dtype=torch.uint8, device=dev),
                       torch.zeros(forma, dtype=torch.float32, device=dev),
                       torch.zeros(forma, dtype=torch.float32, device=dev),
                       torch.zeros(forma, dtype=torch.float32, device=dev),
                       torch.zeros(forma, dtype=torch.float32, device=dev),
                       torch.zeros(forma, dtype=torch.float32, device=dev) if curv_theta else None)

    def azzera(self) -> None:
        for t in (self.cand, self.prop, self.S, self.Q, self.H):
            t.zero_()

    def byte(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.cand, self.prop, self.S, self.Q, self.H, self.HT)
                   if t is not None)


def chiave(fam: S.DatiFamiglia) -> tuple:
    return (int(fam.strato), fam.nome)


@contextmanager
def cronometro(tempi: Optional[dict], nome: str):
    """2/10 ramo tt5-fase-v (--pv-tempi-v): somma in `tempi[nome]` i secondi del blocco, con una sincronizzazione del
    dispositivo prima e dopo (solo orologio: nessun numero cambia). `tempi` None: niente (nessuna sincronizzazione)."""
    if tempi is None:
        yield
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        tempi[nome] = tempi.get(nome, 0.0) + time.perf_counter() - t0
        tempi["n_" + nome] = tempi.get("n_" + nome, 0) + 1


CONTI_DD = {"saltati": 0}                    # 2/10 --pv-dd-solo-salti: dL/dd non calcolati (diagnostica, nessun effetto)


def dd_superfluo(mot, fam) -> bool:
    """2/10 ramo tt5-fase-v (--pv-dd-solo-salti, StatoV.salta_dd_superflui): True se il dL/dd di `fam` in QUESTO indietro
    non serve a nessuno: siamo in un gradiente della fase V (`prepara_ricentro` ha messo il modo 2 e `famiglie_ricentro`)
    e la famiglia non ha salti preparati (prop = 0 ovunque: S e Q non cambierebbero; il dL/dd andrebbe solo nel
    gradiente di theta, buttato da chi chiama). Fuori da quei gradienti, o con l'opzione spenta: False (di sempre)."""
    st = getattr(mot, "stato", None)
    if st is None or not getattr(st, "salta_dd_superflui", False) or int(getattr(st, "modo", 0) or 0) != 2:
        return False
    fr = getattr(st, "famiglie_ricentro", None)
    if fr is None or chiave(fam) in fr:
        return False
    CONTI_DD["saltati"] += 1
    return True


# ═══════════════════════════════════ 1b. V-Bop a gruppo (27/9, TECNICHE_E2E_PV.md §21) ═══════════════════════════════════
# Stato PERSISTENTE fra i passi per gruppo (solo con `--pv-modo vbop`): candidato con isteresi, momento m e dispersione
# v del gradiente proiettato sul candidato (EMA con `gamma`), curvatura H in EMA, eta' e verso della proposta.
# Unita': m = media per MICRO-LOTTO di acc[riga, cand] (quella di S/n del passo V), v = varianza fra micro-lotti
# (divisore n-1), H = curvatura per micro-lotto di curvatura (H/n_curv); il guadagno del PASSO e' N (m|d| - H d^2/2).
# Memoria persistente: 8 B/gruppo (cand u8 + m fp16 + sqrt(v) fp16 + H fp16 + eta' u8 con il bit 7 = verso), piu'
# 12 B/esperto di scale fp32 (m, sqrt(v), H in fp16 con una potenza di 2 per esperto: niente sottoflusso). Il buffer
# del passo (17 B/gruppo) resta quello di prima. Thor-T (~208 M gruppi): +1,7 GB; ODINO (~16,7 G gruppi): +133 GB.
BYTE_PER_GRUPPO_VBOP = 8
ETA_MAX = 127                                 # 7 bit; il bit 7 dell'eta' e' il verso (1 = proposta con acc < 0)
_CAMPI_BOP = ("cand", "m", "r", "H", "eta", "sm", "sr", "sH")


@dataclass
class ConfigBop:
    gamma: float = 0.5             # EMA di m, v, H: x <- (1 - gamma) x + gamma x_passo (gamma = 1: nessuna memoria)
    curvatura_ogni: int = 1        # curvatura misurata ogni N passi V (1 = ogni passo, come PV); in mezzo la sua EMA
    isteresi: float = 2.0          # il candidato cambia solo se |prop nuova| > isteresi |m| (o se non e' piu' fattibile)
    nu_max: int = 200              # gradi di liberta' massimi del t di Student (oltre, t ~ normale)

    def __post_init__(self):
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError(f"gamma V-Bop {self.gamma} fuori da (0, 1]")
        if int(self.curvatura_ogni) < 1:
            raise ValueError(f"curvatura_ogni {self.curvatura_ogni} < 1")


def _codifica16(x: torch.Tensor) -> tuple:
    """x (ne, ...) float64 -> (fp16, scala fp32 per esperto): scala = 2^(ceil(log2 max|x|) - 14), max|x/scala| <= 2^14
    (le divisioni per potenze di 2 sono esatte: stesso risultato ovunque, ripresa al bit)."""
    ne = x.shape[0]
    amax = x.abs().reshape(ne, -1).amax(1)
    esp = torch.where(amax > 0, torch.ceil(torch.log2(torch.where(amax > 0, amax, torch.ones_like(amax)))) - 14,
                      torch.zeros_like(amax)).clamp(-120, 120)
    s = torch.pow(torch.full_like(esp, 2.0), esp)
    forma = (ne,) + (1,) * (x.dim() - 1)
    return (x / s.reshape(forma)).clamp(-65504.0, 65504.0).to(torch.float16), s.to(torch.float32)


def _decodifica16(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    return q.to(torch.float64) * s.to(torch.float64).reshape((q.shape[0],) + (1,) * (q.dim() - 1))


@dataclass
class StatoBop:
    """Stato V-Bop di UNA famiglia (E, R, NG) + scale per esperto (E,). eta' = 0: nessun candidato con statistiche."""
    cand: torch.Tensor
    m: torch.Tensor
    r: torch.Tensor
    H: torch.Tensor
    eta: torch.Tensor
    sm: torch.Tensor
    sr: torch.Tensor
    sH: torch.Tensor

    @staticmethod
    def crea(fam, device=None) -> "StatoBop":
        dev = device if device is not None else fam.blocchi.device
        forma = (fam.E, fam.R, fam.NG)
        z16 = lambda: torch.zeros(forma, dtype=torch.float16, device=dev)  # noqa: E731
        u1 = lambda: torch.ones(fam.E, dtype=torch.float32, device=dev)  # noqa: E731
        return StatoBop(torch.zeros(forma, dtype=torch.uint8, device=dev), z16(), z16(), z16(),
                        torch.zeros(forma, dtype=torch.uint8, device=dev), u1(), u1(), u1())

    def eta_verso(self, e0: int = 0, e1: Optional[int] = None) -> tuple:
        """(eta' int64, verso +-1 float64) degli esperti [e0, e1): verso = segno di acc della proposta adottata."""
        et = self.eta[e0:e1].to(torch.int64)
        return et & ETA_MAX, torch.where((et & 128) != 0, -1.0, 1.0).to(torch.float64)

    def byte(self) -> int:
        return sum(getattr(self, n).numel() * getattr(self, n).element_size() for n in _CAMPI_BOP)


def _correzione_eta(gamma: float, eta: torch.Tensor) -> tuple:
    """(c, somma dei pesi^2) della EMA dopo `eta` aggiornamenti da zero: c = 1 - (1-gamma)^eta (correzione della
    distorsione, come Adam), sum w^2 dei pesi normalizzati w_t = gamma (1-gamma)^(eta-t) / c. gamma = 1: (1, 1)."""
    a = eta.to(torch.float64)
    b = 1.0 - gamma
    c = 1.0 - torch.pow(torch.full_like(a, b), a)
    c = torch.where(eta > 0, c, torch.ones_like(c))
    den = 1.0 - b * b
    sw2 = gamma * gamma * (1.0 - torch.pow(torch.full_like(a, b * b), a)) / (den * c * c)
    return c, torch.where(eta > 0, sw2, torch.ones_like(sw2))


class StatoV:
    """Buffer di tutte le famiglie + il modo del micro-lotto in corso (lo legge il motore dentro `dd`).
    Il modo lo decide chi allena: `inizia_passo()`, poi `microlotto(k)` prima di ogni indietro della perdita, poi
    `curvatura()` prima di ogni indietro di misura della curvatura, poi `fine_raccolta()` (modo 0: qualunque altro
    indietro, p.es. il cancello G1, resta quello di oggi)."""

    def __init__(self, n_proposta: int = 2, passi_accumulo: int = 1, bop: Optional[ConfigBop] = None,
                 curv_theta: bool = False):
        # 27/9 V-Bop (`bop`): stato persistente per gruppo fra i passi (StatoBop); None = PV di prima, al bit
        self.bop = bop
        # 27/9 newton-theta: con `curv_theta` i buffer hanno HT (curvatura delle scale, per PASSO) e si contano i
        # micro-lotti di curvatura del passo (`n_curvatura_passo`); spenta = nessun buffer in piu', stessi numeri
        self.curv_theta = bool(curv_theta)
        self.n_curvatura_passo = 0
        self.bop_: dict = {}
        self.passi_bop = 0
        if bop is not None and int(passi_accumulo) > 1:
            raise ValueError("V-Bop e finestra di accumulo insieme: la memoria fra i passi e' gia' la EMA di V-Bop")
        self.buffer_: dict = {}
        self.modo = 0
        self.n_proposta = n_proposta
        self.n_verifica = 0
        self.n_propose = 0
        self.n_curvatura = 0
        # 27/9: finestra di accumulo (pv_passi_accumulo). Con 1 (default) ogni passo azzera come prima, al bit. Con N > 1
        # le statistiche (S, Q, H, conteggi) si sommano su N passi a codici FERMI: la proposta (cand, prop) si fa solo
        # nel primo passo della finestra, gli altri passi usano TUTTI i micro-lotti per la verifica (piu' potenza ai
        # test). Chi allena chiude la finestra (`chiudi_finestra`) dopo la decisione. `d_finestra`: scala fp16 di
        # ogni gruppo all'inizio della finestra (staleness: vedi `seleziona(d_rif=...)`).
        self.passi_accumulo = max(1, int(passi_accumulo))
        self.passi_nella_finestra = 0
        self.d_finestra: dict = {}
        # 27/9 ramo dominio-strati (--strati-allenati): chiavi (strato, famiglia) delle famiglie ALLENATE; None = tutte
        # (di sempre). Le altre hanno buffer e stato V-Bop (ripresa al bit) ma nessuna proposta/verifica (dL/dd non si
        # calcola negli strati congelati), nessun aggiornamento V-Bop e nessun salto (`PassoV.seleziona`)
        self.attive: Optional[set] = None
        # 2/10 ramo tt5-fase-v (--pv-dd-solo-salti, FASE_V.md cura V1). Spento (default) = di sempre al bit. Acceso:
        # negli indietro dei GRADIENTI della fase V (`prepara_ricentro` -> modo 2: prezzi al punto medio e ricentro) il
        # dL/dd delle famiglie SENZA salti preparati non si calcola (`dd_superfluo`): quel dL/dd va solo nel gradiente
        # di theta, che gradiente_ricentro/gradiente_lettere buttano e rimettono; S (letto solo sui salti preparati) e dX
        # (che porta il gradiente agli strati sotto) restano identici, cosi' come il controllo di trabocco (guarda dX e
        # gli ingressi fp16, non dL/dd). `famiglie_ricentro`: chiavi con almeno un salto preparato (None fuori dai
        # gradienti della fase V: la verifica del passo P in modo 2 calcola tutto come sempre).
        self.salta_dd_superflui = False
        self.famiglie_ricentro: Optional[set] = None

    def _attiva(self, k) -> bool:
        return self.attive is None or k in self.attive

    def prepara(self, famiglie: Iterable[S.DatiFamiglia]) -> None:
        for f in famiglie:
            k = chiave(f)
            if k in self.buffer_:
                raise KeyError(f"famiglia {k} due volte")
            if getattr(f, "E", 1) > 256 or getattr(f, "R", 0) > 65536 or getattr(f, "C", 0) > 65536:
                # 28/9: il registro scrive e in u1, r e c in u2 (REG_DTYPE): fuori campo si troncherebbero in silenzio
                # (p.es. token_embd/output della variante "B", 248.320 righe)
                raise ValueError(f"famiglia {k} ({f.E}x{f.R}x{f.C}) fuori dal campo del registro dei salti")
            self.buffer_[k] = BufferV.crea(f, curv_theta=self.curv_theta)
            if self.bop is not None:
                self.bop_[k] = StatoBop.crea(f)

    def buffer(self, fam: S.DatiFamiglia) -> BufferV:
        k = chiave(fam)
        if k not in self.buffer_:
            raise KeyError(f"nessun buffer V per {k}: `prepara` non l'ha vista")
        return self.buffer_[k]

    def inizia_passo(self) -> None:
        if self.passi_accumulo <= 1 or self.passi_nella_finestra == 0:
            for b in self.buffer_.values():
                b.azzera()
            self.modo, self.n_verifica, self.n_propose, self.n_curvatura = 0, 0, 0, 0
            self.passi_nella_finestra = 0
            self.d_finestra = {}
        self.passi_nella_finestra += 1
        self.modo = 0
        self.famiglie_ricentro = None
        self.n_curvatura_passo = 0                       # newton-theta: HT e' del PASSO, anche dentro una finestra
        if self.curv_theta:
            for b in self.buffer_.values():
                b.HT.zero_()

    def chiudi_finestra(self) -> None:
        """Dopo la decisione (salti tenuti o annullati): il prossimo `inizia_passo` apre una finestra nuova."""
        self.passi_nella_finestra = 0

    def microlotto(self, k: int) -> int:
        """k = indice del micro-lotto nel passo (0-based). 0..n_proposta-1 -> proposta, poi verifica. In una finestra
        di accumulo solo il PRIMO passo propone: i passi successivi verificano con tutti i micro-lotti."""
        primo = self.passi_nella_finestra <= 1
        self.modo = 1 if (primo and k < self.n_proposta) else 2
        if self.modo == 1:
            self.n_propose += 1
        else:
            if self.bop is not None and self.n_verifica == 0:
                self._fondi_candidati()                  # V-Bop: prima verifica del passo = candidati del passo
            self.n_verifica += 1
        return self.modo

    # ─── V-Bop ───
    def _fondi_candidati(self) -> None:
        """Fra proposta e verifica: il candidato persistente resta (isteresi) salvo |prop nuova| > isteresi |m|; se
        cambia (o non c'era) m, v, H e l'eta' si azzerano e il verso e' quello della proposta nuova. Nel buffer del passo
        (letto dal kernel nei modi 2 e 3) va il candidato del passo: cand persistente e prop = +-1 (il suo verso)."""
        ist = float(self.bop.isteresi)
        for k, b in self.buffer_.items():
            if not self._attiva(k):                      # dominio-strati: stato V-Bop degli strati congelati al bit
                continue
            z = self.bop_[k]
            eta, verso = z.eta_verso()
            ha = eta >= 1
            c, _ = _correzione_eta(self.bop.gamma, eta)
            m_hat = _decodifica16(z.m, z.sm) / c
            nuovo = b.prop != 0
            sost = nuovo & (~ha | (b.prop.to(torch.float64).abs() > ist * m_hat.abs()))
            tieni = ha & ~sost
            z.cand.copy_(torch.where(sost, b.cand, z.cand))
            z.eta.copy_(torch.where(sost, torch.where(b.prop < 0, 128, 0).to(torch.uint8), z.eta))
            zero = torch.zeros((), dtype=torch.float16, device=z.m.device)
            for n in ("m", "r", "H"):
                getattr(z, n).copy_(torch.where(sost, zero, getattr(z, n)))
            b.cand.copy_(torch.where(tieni, z.cand, b.cand))
            b.prop.copy_(torch.where(tieni, verso.to(torch.float32), b.prop))

    def serve_curvatura(self) -> bool:
        """V-Bop con `curvatura_ogni` N: la curvatura si misura ai passi V 0, N, 2N, ... (in mezzo vale la sua EMA)."""
        return self.bop is None or self.passi_bop % int(self.bop.curvatura_ogni) == 0

    def aggiorna_bop(self, pezzo: int = 32) -> None:
        """Fine raccolta V-Bop: EMA di m (media per micro-lotto di acc sul candidato), v (varianza fra i micro-lotti di
        verifica) e H (curvatura per micro-lotto, solo se misurata in questo passo; la prima misura entra intera) per i
        gruppi con candidato nel passo; eta' + 1 (tetto ETA_MAX). Nuove scale per esperto."""
        n, nc, gam = self.n_verifica, self.n_curvatura, float(self.bop.gamma)
        if n < 2:
            raise ValueError(f"V-Bop: servono almeno 2 micro-lotti di verifica (ce ne sono {n})")
        for k, b in self.buffer_.items():
            if not self._attiva(k):                      # dominio-strati: stato V-Bop degli strati congelati al bit
                continue
            z = self.bop_[k]
            E = b.prop.shape[0]
            for e0 in range(0, E, pezzo):
                e1 = min(E, e0 + pezzo)
                ha = b.prop[e0:e1] != 0
                Sd, Qd = b.S[e0:e1].to(torch.float64), b.Q[e0:e1].to(torch.float64)
                xm = Sd / n
                s2 = ((Qd - Sd * Sd / n) / (n - 1)).clamp_min(0.0)
                m = _decodifica16(z.m[e0:e1], z.sm[e0:e1])
                v = _decodifica16(z.r[e0:e1], z.sr[e0:e1]) ** 2
                H = _decodifica16(z.H[e0:e1], z.sH[e0:e1])
                m = torch.where(ha, (1.0 - gam) * m + gam * xm, m)
                v = torch.where(ha, (1.0 - gam) * v + gam * s2, v)
                if nc > 0:
                    Ht = b.H[e0:e1].to(torch.float64) / nc
                    H = torch.where(ha, torch.where(H > 0, (1.0 - gam) * H + gam * Ht, Ht), H)
                eta = z.eta[e0:e1].to(torch.int64)
                eta_n = torch.where(ha, torch.clamp((eta & ETA_MAX) + 1, max=ETA_MAX) | (eta & 128), eta)
                z.eta[e0:e1] = eta_n.to(torch.uint8)
                for campo, sc, x in (("m", "sm", m), ("r", "sr", torch.sqrt(v)), ("H", "sH", H)):
                    q, s = _codifica16(x)
                    getattr(z, campo)[e0:e1] = q
                    getattr(z, sc)[e0:e1] = s
        self.passi_bop += 1

    def byte_bop(self) -> int:
        return sum(z.byte() for z in self.bop_.values())

    def istantanea_bop(self) -> dict:
        """Stato V-Bop per la ripresa (tensori clonati su CPU)."""
        return {"gamma": float(self.bop.gamma), "curvatura_ogni": int(self.bop.curvatura_ogni),
                "passi_bop": self.passi_bop,
                "stato": {f"{k[0]}.{k[1]}": {n: getattr(z, n).detach().cpu().clone() for n in _CAMPI_BOP}
                          for k, z in self.bop_.items()}}

    def ripristina_bop(self, ist: dict) -> None:
        nomi = {f"{k[0]}.{k[1]}": k for k in self.bop_}
        if set(ist["stato"]) != set(nomi):
            raise ValueError("stato V-Bop salvato con famiglie diverse")
        for kn, d in ist["stato"].items():
            z = self.bop_[nomi[kn]]
            for n in _CAMPI_BOP:
                t = getattr(z, n)
                if d[n].shape != t.shape or d[n].dtype != t.dtype:
                    raise ValueError(f"stato V-Bop {kn}.{n}: forma/tipo {tuple(d[n].shape)} {d[n].dtype}")
                t.copy_(d[n].to(t.device))
        self.passi_bop = int(ist["passi_bop"])

    def curvatura(self) -> int:
        """Prima di un indietro di MISURA (vettore casuale con covarianza = Hessiana delle uscite): modo 3."""
        if self.n_propose == 0:
            raise RuntimeError("curvatura prima della proposta: i candidati non ci sono ancora")
        self.modo = 3
        self.n_curvatura += 1
        self.n_curvatura_passo += 1
        return self.modo

    @property
    def n_perdita(self) -> int:
        """Micro-lotti della PERDITA del passo (proposta + verifica): il gradiente del passo li somma tutti."""
        return self.n_propose + self.n_verifica

    def fine_raccolta(self) -> None:
        self.modo = 0
        self.famiglie_ricentro = None

    def prepara_ricentro(self, salti: dict) -> None:
        """27/9 ramo v-interazione-b (--pv-ricentra): i buffer del passo, DOPO la decisione (nessuno li legge piu' fino
        al prossimo `inizia_passo`, che li azzera), servono a leggere il gradiente della sonda nello stato raggiunto
        sulla colonna di ogni salto applicato: cand := colonna del salto, prop := 1 (gruppi senza salto: prop = 0,
        ignorati dal kernel), S = Q = 0, modo 2 (verifica). `salti`: {chiave: (e, r, g, cand)}. Dopo l'indietro
        S[e, r, g] = dL/dW_rot sul codice del salto; chi chiama rimette il modo 0 (`fine_raccolta`). I conteggi del
        passo (n_verifica, ...) non si toccano; H e HT restano quelli del passo.
        2/10 (`salta_dd_superflui`): `famiglie_ricentro` = chiavi con almeno un salto (lette da `dd_superfluo`)."""
        for k, b in self.buffer_.items():
            for t in (b.cand, b.prop, b.S, b.Q):
                t.zero_()
            if k in salti:
                e, r, g, c = salti[k]
                if e.numel():
                    dev = b.cand.device
                    e, r, g = e.to(dev), r.to(dev), g.to(dev)
                    b.cand[e, r, g] = c.to(dev, torch.uint8)
                    b.prop[e, r, g] = 1.0
        self.famiglie_ricentro = {k for k, t in salti.items() if t[0].numel()}
        self.modo = 2

    def byte(self) -> int:
        return sum(b.byte() for b in self.buffer_.values())

    def istantanea(self) -> dict:
        """Stato della finestra per la ripresa (tensori su CPU): buffer, istantanea delle scale, conteggi."""
        return {"passi_accumulo": self.passi_accumulo, "passi_nella_finestra": self.passi_nella_finestra,
                "n_verifica": self.n_verifica, "n_propose": self.n_propose, "n_curvatura": self.n_curvatura,
                "buffer": {f"{k[0]}.{k[1]}": {n: getattr(b, n).detach().cpu() for n in ("cand", "prop", "S", "Q", "H")}
                           for k, b in self.buffer_.items()},
                "d_finestra": {f"{k[0]}.{k[1]}": v.detach().cpu() for k, v in self.d_finestra.items()}}

    def ripristina(self, ist: dict) -> None:
        """Inverso di `istantanea`: stessa finestra, stessi numeri al bit (i buffer tornano sul loro dispositivo)."""
        if int(ist["passi_accumulo"]) != self.passi_accumulo:
            raise ValueError(f"finestra salvata con passi_accumulo {ist['passi_accumulo']}, ora {self.passi_accumulo}")
        nomi = {f"{k[0]}.{k[1]}": k for k in self.buffer_}
        if set(ist["buffer"]) != set(nomi):
            raise ValueError("finestra salvata con famiglie diverse")
        for kn, d in ist["buffer"].items():
            b = self.buffer_[nomi[kn]]
            for n in ("cand", "prop", "S", "Q", "H"):
                getattr(b, n).copy_(d[n].to(getattr(b, n).device))
        self.d_finestra = {nomi[kn]: v.to(self.buffer_[nomi[kn]].prop.device) for kn, v in ist["d_finestra"].items()}
        self.passi_nella_finestra = int(ist["passi_nella_finestra"])
        self.n_verifica, self.n_propose, self.n_curvatura = (int(ist["n_verifica"]), int(ist["n_propose"]),
                                                             int(ist["n_curvatura"]))
        self.modo = 0


# ═══════════════════════════════════ 2. kernel ═══════════════════════════════════
if S._TRITON:
    @triton.jit
    def _k_dd_v(dy_ptr, sdy_s, sdy_r, x_ptr, sx_s, sx_c,
                pb, sb_e, NBR, dd_ptr, sdd_e, NG, offs_ptr, R, C,
                d0_ptr, sd0_e, cand_ptr, prop_ptr, s_ptr, q_ptr, h_ptr, sv_e,
                GS: tl.constexpr, QK: tl.constexpr, NB: tl.constexpr, ROT: tl.constexpr,
                BR: tl.constexpr, BS: tl.constexpr, MODO_V: tl.constexpr,
                dylo_ptr=None, inv_ptr=None, sinv_e=0, DIVISO: tl.constexpr = 0):
        """`_k_dd` dello studente (righe identiche fino allo store di dL/dd) + epilogo del passo V sullo stesso
        riquadro acc = dY^T X (BR x GS) nei registri. MODO_V 0: solo dL/dd. 1: proposta. 2: verifica.
        3: curvatura (acc = (dY^2)^T (X^2) in fp32, dL/dd = 0, H += acc[riga, cand]).
        27/9 (ramo indietro-bf16x3, `e2e_indietro_hilo` via "fp16x2"): DIVISO=1 -> dY = (hi + lo) * inv[e, r], hi in
        dy_ptr e lo in dylo_ptr (fp16, stessi passi), inv (E, R) fp32 potenze di due: acc_lo e acc_hi in due
        accumulatori, acc = (acc_lo + acc_hi) * inv PRIMA dell'epilogo (dL/dd e statistiche V vedono dY^T X intero).
        Non con MODO_V 3. DIVISO=0 (default): il kernel di prima (ramo costante eliminato alla compilazione)."""
        e = tl.program_id(0)
        pid_r = tl.program_id(1)
        g = tl.program_id(2)
        s_ini = tl.load(offs_ptr + e)
        s_fine = tl.load(offs_ptr + e + 1)
        rr = pid_r * BR + tl.arange(0, BR)
        cc = g * GS + tl.arange(0, GS)
        mr = rr < R
        acc = tl.zeros((BR, GS), dtype=tl.float32)
        if DIVISO:
            acc_lo = tl.zeros((BR, GS), dtype=tl.float32)
        for s0 in range(s_ini, s_fine, BS):
            rs = s0 + tl.arange(0, BS)
            ms = rs < s_fine
            dyt = tl.load(dy_ptr + rs[None, :].to(tl.int64) * sdy_s + rr[:, None] * sdy_r,
                          mask=ms[None, :] & mr[:, None], other=0.0)
            x = tl.load(x_ptr + rs[:, None].to(tl.int64) * sx_s + cc[None, :] * sx_c,
                        mask=ms[:, None], other=0.0)
            if MODO_V == 3:
                d32 = dyt.to(tl.float32)
                x32 = x.to(tl.float32)
                acc += tl.dot(d32 * d32, x32 * x32)
            elif DIVISO:
                dyl = tl.load(dylo_ptr + rs[None, :].to(tl.int64) * sdy_s + rr[:, None] * sdy_r,
                              mask=ms[None, :] & mr[:, None], other=0.0)
                acc_lo += tl.dot(dyl, x)
                acc += tl.dot(dyt, x)
            else:
                acc += tl.dot(dyt, x)
        e64 = e.to(tl.int64)
        if DIVISO:
            inv = tl.load(inv_ptr + e64 * sinv_e + rr, mask=mr, other=1.0)
            acc = (acc_lo + acc) * inv[:, None]
        if MODO_V == 3:
            tl.store(dd_ptr + e64 * sdd_e + rr * NG + g, tl.zeros((BR,), dtype=tl.float32), mask=mr)
            ov3 = e64 * sv_e + rr * NG + g
            loc3 = tl.arange(0, GS)
            pr3 = tl.load(prop_ptr + ov3, mask=mr, other=0.0)
            ci3 = tl.load(cand_ptr + ov3, mask=mr, other=0).to(tl.int32)
            v3 = tl.sum(tl.where(loc3[None, :] == ci3[:, None], acc, 0.0), axis=1)
            v3 = tl.where(pr3 != 0.0, v3, 0.0)
            hv = tl.load(h_ptr + ov3, mask=mr, other=0.0)
            tl.store(h_ptr + ov3, hv + v3, mask=mr)
        else:
            t, m2 = _trit_da_byte(pb + e64 * sb_e, rr[:, None], cc[None, :], NBR, R, C, QK, NB, ROT)
            val = tl.sum(acc * t.to(tl.float32), axis=1)
            tl.store(dd_ptr + e64 * sdd_e + rr * NG + g, val, mask=mr)
            if MODO_V != 0:
                ov = e64 * sv_e + rr * NG + g
                loc = tl.arange(0, GS)
                if MODO_V == 1:
                    dv = tl.load(d0_ptr + e64 * sd0_e + rr * NG + g, mask=mr, other=0.0)
                    sd = tl.where(dv > 0.0, 1.0, tl.where(dv < 0.0, -1.0, 0.0))
                    pend = acc * sd[:, None]                 # dL/dt = acc * d: stesso segno di acc * sign(d)
                    dz = tl.where(pend > 0.0, -1, tl.where(pend < 0.0, 1, 0))
                    tn = t + dz
                    fatt = (dz != 0) & (tn >= -1) & (tn <= 1) & m2
                    punt = tl.where(fatt, tl.abs(acc), -1.0)
                    ci = tl.argmax(punt, axis=1)             # a parita', la colonna piu' bassa (come torch.argmax)
                    best = tl.max(punt, axis=1)
                    pv = tl.sum(tl.where(loc[None, :] == ci[:, None], acc, 0.0), axis=1)   # esatto: acc + zeri
                    pn = tl.where(best > 0.0, pv, 0.0)
                    po = tl.load(prop_ptr + ov, mask=mr, other=0.0)
                    co = tl.load(cand_ptr + ov, mask=mr, other=0)
                    rep = tl.abs(pn) > tl.abs(po)
                    tl.store(prop_ptr + ov, tl.where(rep, pn, po), mask=mr)
                    tl.store(cand_ptr + ov, tl.where(rep, ci.to(tl.uint8), co), mask=mr)
                else:
                    pr = tl.load(prop_ptr + ov, mask=mr, other=0.0)
                    ci = tl.load(cand_ptr + ov, mask=mr, other=0).to(tl.int32)
                    v = tl.sum(tl.where(loc[None, :] == ci[:, None], acc, 0.0), axis=1)
                    v = tl.where(pr != 0.0, v, 0.0)
                    sv = tl.load(s_ptr + ov, mask=mr, other=0.0)
                    qv = tl.load(q_ptr + ov, mask=mr, other=0.0)
                    tl.store(s_ptr + ov, sv + v, mask=mr)
                    tl.store(q_ptr + ov, qv + v * v, mask=mr)


if S._TRITON:
    @triton.jit
    def _k_ht(dy_ptr, sdy_s, sdy_r, x_ptr, sx_s, sx_c, pb, sb_e, NBR, offs_ptr, R, C, ht_ptr, sv_e, NG,
              GS: tl.constexpr, QK: tl.constexpr, NB: tl.constexpr, ROT: tl.constexpr,
              BR: tl.constexpr, BS: tl.constexpr, GSB: tl.constexpr):
        """27/9 ramo newton-theta: curvatura Gauss-Newton delle SCALE per gruppo (senza d^2), kernel A PARTE (dentro
        `_k_dd_v` modo 3 riversava 177-1061 VGPR su gfx1151): HT[e, r, g] += sum_s dY_{s,r}^2 u_{s,r}^2,
        u_{s,r} = sum_{c in g} t_{r,c} X_{s,c} (il contributo del gruppo all'uscita, a scala 1). Stessa griglia di
        `_k_dd_v` (esperto, riquadro di righe, gruppo); X letto gia' trasposto; nessuna atomica. GSB = colonne per
        sotto-blocco: GSB == GS (R1024, gruppo 128) -> trit decodificati UNA volta prima del ciclo; GSB < GS (TQ1_0,
        gruppo 256) -> u a sotto-blocchi con i trit ridecodificati a ogni riquadro di token (con GS intero nei
        registri il kernel riversava 46-60 VGPR)."""
        e = tl.program_id(0)
        pid_r = tl.program_id(1)
        g = tl.program_id(2)
        s_ini = tl.load(offs_ptr + e)
        s_fine = tl.load(offs_ptr + e + 1)
        rr = pid_r * BR + tl.arange(0, BR)
        mr = rr < R
        e64 = e.to(tl.int64)
        h = tl.zeros((BR,), dtype=tl.float32)
        if GSB == GS:
            cc = g * GS + tl.arange(0, GS)
            t, mt = _trit_da_byte(pb + e64 * sb_e, rr[:, None], cc[None, :], NBR, R, C, QK, NB, ROT)
            tf = tl.where(mt, t.to(tl.float32), 0.0)
            for s0 in range(s_ini, s_fine, BS):
                rs = s0 + tl.arange(0, BS)
                ms = rs < s_fine
                dyt = tl.load(dy_ptr + rs[None, :].to(tl.int64) * sdy_s + rr[:, None] * sdy_r,
                              mask=ms[None, :] & mr[:, None], other=0.0).to(tl.float32)
                xt = tl.load(x_ptr + rs[None, :].to(tl.int64) * sx_s + cc[:, None] * sx_c,
                             mask=ms[None, :], other=0.0).to(tl.float32)        # (GS, BS)
                u = tl.dot(tf, xt)                                               # (BR, BS)
                h += tl.sum(dyt * dyt * u * u, axis=1)
        else:
            for s0 in range(s_ini, s_fine, BS):
                rs = s0 + tl.arange(0, BS)
                ms = rs < s_fine
                dyt = tl.load(dy_ptr + rs[None, :].to(tl.int64) * sdy_s + rr[:, None] * sdy_r,
                              mask=ms[None, :] & mr[:, None], other=0.0).to(tl.float32)
                u = tl.zeros((BR, BS), dtype=tl.float32)
                for j in range(GS // GSB):
                    cc = g * GS + j * GSB + tl.arange(0, GSB)
                    t, mt = _trit_da_byte(pb + e64 * sb_e, rr[:, None], cc[None, :], NBR, R, C, QK, NB, ROT)
                    xt = tl.load(x_ptr + rs[None, :].to(tl.int64) * sx_s + cc[:, None] * sx_c,
                                 mask=ms[None, :], other=0.0).to(tl.float32)    # (GSB, BS)
                    u += tl.dot(tl.where(mt, t.to(tl.float32), 0.0), xt)
                h += tl.sum(dyt * dyt * u * u, axis=1)
        ov = e64 * sv_e + rr * NG + g
        hv = tl.load(ht_ptr + ov, mask=mr, other=0.0)
        tl.store(ht_ptr + ov, hv + h, mask=mr)


# Taglie di `_k_ht`: compilazione anticipata per gfx1151 (27/9, prova N5 di e2e_test_sophia.py): BR 16, BS 16, 8 warp,
# sotto-blocchi da 128 colonne -> 0 VGPR riversati con R1024 e TQ1_0, ingressi fp16 e fp32. Il tempo va misurato sulla GPU.
BLOCCHI_HT = S.Blocchi(BR_dd=16, BS_dd=16, num_warps=8)
GSB_HT = 128


def ht_triton(dY: torch.Tensor, X: torch.Tensor, fam: S.DatiFamiglia, piano: S.Piano, HT: torch.Tensor,
              blk: Optional[S.Blocchi] = None) -> None:
    """HT += curvatura delle scale del micro-lotto (in place). dY e X come in `dd_v_triton`."""
    b = blk or BLOCCHI_HT
    dY, X = dY.contiguous(), X.contiguous()
    griglia = (fam.E, triton.cdiv(fam.R, b.BR_dd), fam.NG)
    _k_ht[griglia](dY, dY.stride(0), dY.stride(1), X, X.stride(0), X.stride(1),
                   fam.blocchi, fam.blocchi.stride(0), fam.NBR, piano.offs, fam.R, fam.C, HT, fam.R * fam.NG, fam.NG,
                   fam.gs, fam.qk, fam.nb, int(fam.ruotata), b.BR_dd, b.BS_dd, min(fam.gs, GSB_HT),
                   num_warps=b.num_warps)


def ht_torch(gy: torch.Tensor, x: torch.Tensor, fam: S.DatiFamiglia, piano: S.Piano, HT: torch.Tensor) -> None:
    """Riferimento di `_k_ht` esperto per esperto (u materializzato (s, R, NG)); fp64 se gli ingressi lo sono."""
    acc = torch.float64 if gy.dtype == torch.float64 else torch.float32
    o = piano.offs.tolist()
    for e in range(fam.E):
        if o[e + 1] > o[e]:
            g2 = gy[o[e]:o[e + 1]].to(acc) ** 2                                  # (s, R)
            xe = x[o[e]:o[e + 1]].to(acc).view(-1, fam.NG, fam.gs)               # (s, NG, gs)
            t = S.trit_esperto(fam, e).to(acc).view(fam.R, fam.NG, fam.gs)
            u = torch.einsum("sgj,rgj->srg", xe, t)
            HT[e] += torch.einsum("sr,srg->rg", g2, u * u).to(HT.dtype)


def dd_v_triton(dY: torch.Tensor, X: torch.Tensor, fam: S.DatiFamiglia, piano: S.Piano, blk: S.Blocchi,
                buf: Optional[BufferV], modo: int, dy_lo: Optional[torch.Tensor] = None,
                inv: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Come `S._dd_triton` (stessa griglia, stesse taglie) col kernel `_k_dd_v`. modo 0: `buf` puo' essere None.
    27/9: `dy_lo` + `inv` (E, R) = dY diviso (fp16x2 di `e2e_indietro_hilo`): dY vero = (dY + dy_lo) * inv[e, r]."""
    dY, X = dY.contiguous(), X.contiguous()
    assert dY.dtype == X.dtype
    diviso = dy_lo is not None
    if diviso:
        if modo == 3 or inv is None:
            raise ValueError("dd_v_triton diviso: serve inv, e non in modo 3 (curvatura)")
        dy_lo = dy_lo.contiguous()
        inv = inv.to(torch.float32).contiguous()
        if dy_lo.shape != dY.shape or dy_lo.dtype != dY.dtype or dy_lo.stride() != dY.stride() \
                or tuple(inv.shape) != (fam.E, fam.R):
            raise ValueError("dd_v_triton diviso: dy_lo/inv con forma, tipo o passi sbagliati")
    dd = torch.empty(fam.E, fam.R, fam.NG, dtype=torch.float32, device=dY.device)
    d0 = fam.d0.contiguous()
    if buf is None:
        assert modo == 0
        z8 = torch.zeros(1, dtype=torch.uint8, device=dY.device)
        z32 = torch.zeros(1, dtype=torch.float32, device=dY.device)
        cand, prop, s_, q_, h_, sv_e = z8, z32, z32, z32, z32, 0
    else:
        cand, prop, s_, q_, h_, sv_e = buf.cand, buf.prop, buf.S, buf.Q, buf.H, fam.R * fam.NG
    griglia = (fam.E, triton.cdiv(fam.R, blk.BR_dd), fam.NG)
    _k_dd_v[griglia](dY, dY.stride(0), dY.stride(1), X, X.stride(0), X.stride(1),
                     fam.blocchi, fam.blocchi.stride(0), fam.NBR, dd, dd.stride(0), fam.NG, piano.offs,
                     fam.R, fam.C, d0, d0.stride(0), cand, prop, s_, q_, h_, sv_e,
                     fam.gs, fam.qk, fam.nb, int(fam.ruotata), blk.BR_dd, blk.BS_dd, int(modo),
                     **({"dylo_ptr": dy_lo, "inv_ptr": inv, "sinv_e": inv.stride(0), "DIVISO": 1} if diviso else {}),
                     num_warps=blk.num_warps)
    if modo == 3 and buf is not None and buf.HT is not None:     # newton-theta: curvatura delle scale
        ht_triton(dY, X, fam, piano, buf.HT)
    return dd


# ═══════════════════════════════════ 3. riferimento torch (dW materializzato) ═══════════════════════════════════
def proposta_da_dw(buf: BufferV, e: int, dW: torch.Tensor, q: torch.Tensor, d0e: torch.Tensor, gs: int) -> None:
    """Proposta dal dW (R, C) ESPLICITO dell'esperto e (riferimento del modo 1): stessa regola del kernel."""
    R, C = dW.shape
    A = dW.view(R, C // gs, gs)
    t = q.to(torch.int32).view(R, C // gs, gs)
    pend = A * torch.sign(d0e.to(A.dtype)).unsqueeze(-1)
    dz = -torch.sign(pend).to(torch.int32)
    tn = t + dz
    fatt = (dz != 0) & (tn >= -1) & (tn <= 1)
    punt = torch.where(fatt, A.abs(), torch.full_like(A, -1.0))
    best = punt.amax(-1)
    ci = punt.argmax(-1)                          # a parita' il primo indice (come tl.argmax, tie_break_left)
    pv = A.gather(-1, ci.unsqueeze(-1)).squeeze(-1)
    pn = torch.where(best > 0, pv, torch.zeros_like(pv)).to(torch.float32)
    rep = pn.abs() > buf.prop[e].abs()
    buf.prop[e] = torch.where(rep, pn, buf.prop[e])
    buf.cand[e] = torch.where(rep, ci.to(torch.uint8), buf.cand[e])


def verifica_da_dw(buf: BufferV, e: int, dW: torch.Tensor, gs: int) -> None:
    R, C = dW.shape
    A = dW.view(R, C // gs, gs)
    v = A.gather(-1, buf.cand[e].to(torch.int64).unsqueeze(-1)).squeeze(-1).to(torch.float32)
    v = torch.where(buf.prop[e] != 0, v, torch.zeros_like(v))
    buf.S[e] += v
    buf.Q[e] += v * v


def curvatura_da_dw2(buf: BufferV, e: int, dW2: torch.Tensor, gs: int) -> None:
    """Modo 3 dal (dY^2)^T (X^2) (R, C) ESPLICITO dell'esperto e: H += valore sul candidato (dove c'e')."""
    R, C = dW2.shape
    A = dW2.view(R, C // gs, gs)
    v = A.gather(-1, buf.cand[e].to(torch.int64).unsqueeze(-1)).squeeze(-1).to(torch.float32)
    buf.H[e] += torch.where(buf.prop[e] != 0, v, torch.zeros_like(v))


def dd_v_torch(gy, x, fam: S.DatiFamiglia, piano: S.Piano, d_dtype, buf: Optional[BufferV], modo: int):
    """`S.MotoreTorch.dd` (stesse operazioni, stessa uscita al bit) + proposta/verifica dal dW (R, C) esplicito
    di ogni esperto: e' il riferimento «materializzato» contro cui si prova il kernel. Transitorio: UN dW.
    Modo 3 (curvatura): dL/dd = 0 e H dal (dY^2)^T (X^2) esplicito."""
    acc = torch.float64 if gy.dtype == torch.float64 else torch.float32
    dd = torch.zeros(fam.E, fam.R, fam.NG, dtype=d_dtype, device=gy.device)
    o = piano.offs.tolist()
    if modo == 3:
        for e in range(fam.E):
            if o[e + 1] > o[e]:
                g2 = gy[o[e]:o[e + 1]].to(torch.float64 if acc == torch.float64 else torch.float32) ** 2
                x2 = x[o[e]:o[e + 1]].to(g2.dtype) ** 2
                curvatura_da_dw2(buf, e, torch.matmul(g2.T, x2), fam.gs)
        if buf.HT is not None:                                   # newton-theta: curvatura delle scale
            ht_torch(gy, x, fam, piano, buf.HT)
        return dd
    for e in range(fam.E):
        if o[e + 1] > o[e]:
            dW = torch.matmul(gy[o[e]:o[e + 1]].to(acc).T, x[o[e]:o[e + 1]].to(acc))       # (R, C)
            q = S.trit_esperto(fam, e).to(acc)
            dd[e] = (q * dW).view(fam.R, fam.NG, fam.gs).sum(-1).to(d_dtype)
            if modo == 1:
                proposta_da_dw(buf, e, dW, q, fam.d0[e], fam.gs)
            elif modo == 2:
                verifica_da_dw(buf, e, dW, fam.gs)
        # esperto senza token: acc = 0 -> proposta nulla (nessun cambio), verifica v = 0 (S, Q invariati)
    return dd


# ═══════════════════════════════════ 4. motori ═══════════════════════════════════
class MotoreTorchV(S.MotoreTorch):
    """Motore torch dello studente; `dd` in modo V raccoglie anche proposta/verifica (dW per esperto)."""

    def __init__(self, stato: StatoV):
        self.stato = stato
        self.nome = "torch+V"

    def dd(self, gy, x, fam, piano, d_dtype):
        m = self.stato.modo
        if m == 0:
            return super().dd(gy, x, fam, piano, d_dtype)
        return dd_v_torch(gy, x, fam, piano, d_dtype, self.stato.buffer(fam), m)


# Taglie del kernel V (T5, compilazione per gfx1151): coi Blocchi di produzione (BR 64, 8 warp) il modo 1 riversa
# 76-530 VGPR; con BR 32 e 8 warp: 0 per R1024 fp16/fp32 e TQ1_0 fp16, 4 (modo 1) e 1 (modo 2) per TQ1_0 fp32, dove
# il `_k_dd` di produzione di oggi ne riversa gia' 171. Il tempo lo decide V-M0 sulla GPU.
BLOCCHI_V = S.Blocchi(BR_dd=32)

if S._TRITON:
    class MotoreTritonV(S.MotoreTriton):
        """Motore Triton dello studente (isinstance MotoreTriton: la combinazione fusa resta accesa); `dd` in modo
        V usa `_k_dd_v` con `blk_v` (default BLOCCHI_V); il modo 0 resta il dL/dd del motore di base (`_k_dd`, o
        `_k_dd2` se la base e' «triton2»: v2=True, che vale anche per avanti e dX). I modi V 1-3 usano SEMPRE
        `_k_dd_v`, identico per entrambe le basi."""

        def __init__(self, stato: StatoV, denso: bool = False, blk: Optional[S.Blocchi] = None,
                     blk_v: Optional[S.Blocchi] = None, v2: bool = False):
            super().__init__(denso=denso, blk=blk, v2=v2)
            self.stato = stato
            self.blk_v = blk_v or BLOCCHI_V
            self.nome = self.nome + "+V"

        def dd(self, gy, x, fam, piano, d_dtype):
            m = self.stato.modo
            if m == 0:
                return super().dd(gy, x, fam, piano, d_dtype)
            return dd_v_triton(gy, x, fam, piano, self.blk_v, self.stato.buffer(fam), m).to(d_dtype)


def installa(st, stato: StatoV, motore: str = "auto", blk_v: Optional[S.Blocchi] = None):
    """Mette il motore V in TUTTI gli strati di uno `S.Studente` (stesso oggetto per tutti: un solo `stato`),
    prepara i buffer delle sue famiglie e ritorna {(L, nome): DatiFamiglia}. Non tocca theta ne' i byte."""
    if motore == "auto":
        motore = "triton" if isinstance(st.mot, S.MotoreTriton) else "torch"
    if motore == "torch":
        mv = MotoreTorchV(stato)
    else:
        base = st.mot
        mv = MotoreTritonV(stato, denso=getattr(base, "denso", False), blk=getattr(base, "blk", None), blk_v=blk_v,
                           v2=getattr(base, "v2", False))
    fams = {}
    for L in st.strati:
        es = st.esperti(L)
        es.mot = mv
        for n, f in es.fam.items():
            fams[(L, n)] = f
    st.mot = mv
    stato.prepara(fams.values())
    return fams


# ═══════════════════════════════════ 5. selezione ═══════════════════════════════════
@dataclass
class ConfigV:
    q_fdr: float = 0.1             # Benjamini-Hochberg per matrice
    n_proposta: int = 2            # micro-lotti di proposta (gli altri verificano)
    passi_accumulo: int = 1        # finestra di accumulo delle statistiche V (1 = decisione a ogni passo, come prima)
    pezzo_esperti: int = 32        # esperti per pezzo nella selezione (transitorio limitato)
    pezzo_decodifica: int = 8      # esperti per pezzo nel conteggio dei non nulli
    soglia_staleness: float = 0.02 # finestra di accumulo: un gruppo la cui scala si e' mossa di oltre il 2% dall'inizio
                                   # della finestra non si testa (il suo segnale e' stato raccolto con un'altra scala)
    q_guadagno: float = 0.1        # BH per matrice sull'ipotesi «il salto NON riduce la perdita» (guadagno di secondo
                                   # ordine <= 0): col valore puntuale un maestro fuori griglia (residuo 0,4 d) faceva
                                   # salire la perdita (maledizione del vincitore fra migliaia di codici)
    # 27/9 --pv-accetta (ramo v-senza-bh, TECNICHE_E2E_PV.md §38). "bh" (default) = i due test BH di sopra, identico al
    # bit. "guadagno" = accettato ogni candidato col guadagno di secondo ordine STIMATO > 0 (|g| d > d^2 H / 2; V-Bop:
    # N (m^ |d| - H d^2 / 2) > 0) SENZA i test BH (q_fdr e q_guadagno non si leggono); restano invariati ordinamento per
    # guadagno, budget rho garantito, sonda con annullamento al bit e rho adattivo. `z_min` (solo con "guadagno"):
    # in piu' z > z_min, z = t di Student del gradiente nella direzione della proposta (lo stesso z del test sulla
    # direzione; col guadagno > 0 e' sempre z > 0, quindi z > z_min == |z| > z_min). 0 = nessuna soglia.
    # `z_guadagno_min` (solo con "guadagno"): in piu' t_g > z_guadagno_min, t_g = t del GUADAGNO (quello del test
    # q_guadagno, `t_guadagno`; V-Bop: (m^ verso - H|d|/2) / se) = limite inferiore di confidenza del guadagno > 0 a
    # z_guadagno_min errori standard, SENZA correzione per molteplicita'. 0 = nessuna soglia (t_g > 0 <=> guadagno > 0).
    accetta: str = "bh"
    z_min: float = 0.0
    z_guadagno_min: float = 0.0

    def __post_init__(self):
        if self.accetta not in ("bh", "guadagno"):
            raise ValueError(f"accetta {self.accetta!r}: 'bh' o 'guadagno'")
        for nome in ("z_min", "z_guadagno_min"):
            v = getattr(self, nome)
            if not v >= 0.0:
                raise ValueError(f"{nome} {v} < 0")
            if self.accetta == "bh" and v != 0.0:
                raise ValueError(f"{nome} > 0 solo con accetta 'guadagno' (con 'bh' la soglia la fanno i test BH)")


def sf_t(t: torch.Tensor, nu: int) -> torch.Tensor:
    """P(T > t), T di Student con nu (intero >= 1) gradi: forma chiusa di Abramowitz-Stegun 26.7.3 (nu dispari)
    e 26.7.4 (nu pari), in float64, sul dispositivo del tensore (nessuna scipy). t = +-inf ammessi."""
    t = t.to(torch.float64)
    a = t.abs()
    th = torch.atan(a / math.sqrt(nu))
    c, s = torch.cos(th), torch.sin(th)
    c2 = c * c
    if nu % 2 == 1:
        if nu == 1:
            A = 2.0 * th / math.pi
        else:
            term = c.clone()
            somma = c.clone()
            for k in range(1, (nu - 3) // 2 + 1):
                term = term * c2 * (2.0 * k) / (2.0 * k + 1.0)
                somma = somma + term
            A = (2.0 / math.pi) * (th + s * somma)
    else:
        term = torch.ones_like(c)
        somma = torch.ones_like(c)
        for k in range(1, (nu - 2) // 2 + 1):
            term = term * c2 * (2.0 * k - 1.0) / (2.0 * k)
            somma = somma + term
        A = s * somma
    p = (0.5 * (1.0 - A)).clamp(0.0, 1.0)
    return torch.where(t >= 0, p, 1.0 - p)


def p_valori(prop: torch.Tensor, Ssum: torch.Tensor, Qsum: torch.Tensor, n: int) -> tuple:
    """t di Student UNILATERALE nella direzione della proposta: z = S sign(prop) / sqrt(n s^2), s^2 = varianza
    fra gli n micro-lotti di verifica (n-1 gradi). Ritorna (p, z, S_dir) in float64."""
    if n < 2:
        raise ValueError(f"servono almeno 2 micro-lotti di verifica (ce ne sono {n})")
    Sd = Ssum.to(torch.float64)
    Qd = Qsum.to(torch.float64)
    s_dir = Sd * torch.sign(prop.to(torch.float64))
    var = ((Qd - Sd * Sd / n) / (n - 1)).clamp_min(0.0)
    den = torch.sqrt(n * var)
    inf = torch.full_like(s_dir, math.inf)
    z = torch.where(den > 0, s_dir / den.clamp_min(1e-300),
                    torch.where(s_dir > 0, inf, torch.where(s_dir < 0, -inf, torch.zeros_like(s_dir))))
    return sf_t(z, n - 1), z, s_dir


def bh(p: torch.Tensor, testati: torch.Tensor, q: float) -> torch.Tensor:
    """Benjamini-Hochberg riga per riga (una riga = una matrice). p (n, M) float64, `testati` (n, M) bool: le
    ipotesi (i gruppi con candidato); m = testati per riga. Ritorna la maschera dei rifiuti (scoperte)."""
    pp = torch.where(testati, p, torch.full_like(p, math.inf))
    ps, _ = torch.sort(pp, dim=1)
    M = pp.shape[1]
    k = torch.arange(1, M + 1, dtype=torch.float64, device=p.device)
    m = testati.sum(1).to(torch.float64).clamp_min(1.0)
    ok = ps <= k[None, :] * q / m[:, None]
    kmax = (ok.to(torch.float64) * k[None, :]).amax(1).to(torch.int64)
    taglio = ps.gather(1, (kmax - 1).clamp_min(0).unsqueeze(1)).squeeze(1)
    taglio = torch.where(kmax > 0, taglio, torch.full_like(taglio, -math.inf))
    return testati & (pp <= taglio[:, None])


def nnz_gruppi(fam: S.DatiFamiglia, e0: int, e1: int, pezzo: int = 8) -> torch.Tensor:
    """Trit non nulli per gruppo (e1-e0, R, NG) int32, dai byte (decode di e2e_campi_scala), a pezzi di esperti."""
    out = []
    for a in range(e0, e1, pezzo):
        b = min(e1, a + pezzo)
        t = CAMPI.trit_torch(fam.blocchi[a:b].reshape(-1, fam.nb), fam.tipo)
        out.append((t != 0).view(b - a, fam.R, fam.NG, fam.gs).sum(-1, dtype=torch.int32))
        del t
    return torch.cat(out)


def budget_esperti(d: torch.Tensor, nnz: torch.Tensor, rho: float) -> torch.Tensor:
    """Budget del trust ratio per esperto: rho^2 ||W_e||^2 = rho^2 sum_g d_g^2 nnz_g (float64, (n,)). I salti scelti
    devono avere sum d_g^2 <= budget, cioe' ||Delta W_e|| / ||W_e|| <= rho (PV §4.2), garantito per costruzione."""
    ne = d.shape[0]
    d2 = d.to(torch.float64).reshape(ne, -1) ** 2
    return rho * rho * (d2 * nnz.to(torch.float64).reshape(ne, -1)).sum(1)


@dataclass
class Selezione:
    """Salti scelti per UNA famiglia: indici (e, r, g), colonna nel gruppo, direzione, guadagno previsto (secondo
    ordine, unita' della perdita del passo); e per esperto: testati, scoperte BH, convenienti (guadagno > 0 fra le
    scoperte), budget (rho^2 ||W_e||^2), scelti, costo (sum d^2 dei scelti <= budget)."""
    e: torch.Tensor
    r: torch.Tensor
    g: torch.Tensor
    cand: torch.Tensor
    dir: torch.Tensor
    guadagno: torch.Tensor
    testati: torch.Tensor
    scoperte: torch.Tensor
    convenienti: torch.Tensor
    budget: torch.Tensor
    scelti: torch.Tensor
    costo: torch.Tensor

    def __len__(self) -> int:
        return int(self.e.numel())

    def colonne(self, gs: int) -> torch.Tensor:
        return self.g * gs + self.cand


def guadagno_secondo_ordine(S_: torch.Tensor, prop: torch.Tensor, H: torch.Tensor, d: torch.Tensor, n_perdita: int,
                            n_verifica: int, n_curvatura: int, Q_: Optional[torch.Tensor] = None,
                            t_lcb: float = 0.0) -> torch.Tensor:
    """Guadagno previsto (diminuzione della perdita del PASSO) del salto di UN livello sul candidato:
    g_passo |d| - H_passo d^2 / 2, g_passo = (N/n_ver) S sign(prop) (la somma dei micro-lotti della perdita, stimata
    dai soli di verifica: stima non distorta), H_passo = (N/n_curv) H (curvatura Gauss-Newton campionata, stessi
    token per costruzione della perdita). E' positivo solo se il minimo quadratico locale sta oltre meta' livello:
    esattamente l'arrotondamento al livello piu' vicino. Con `Q_` e `t_lcb` > 0 il gradiente e' il suo LIMITE
    INFERIORE di confidenza: S_dir - t_lcb sqrt(n s^2) (s^2 = varianza fra i micro-lotti di verifica). float64."""
    if n_curvatura < 1:
        raise ValueError("criterio di secondo ordine senza curvatura: servono micro-lotti di curvatura (modo 3)")
    N = float(n_perdita)
    Sd = S_.to(torch.float64) * torch.sign(prop.to(torch.float64))
    if Q_ is not None and t_lcb > 0:
        n = float(n_verifica)
        var = ((Q_.to(torch.float64) - S_.to(torch.float64) ** 2 / n) / (n - 1)).clamp_min(0.0)
        Sd = Sd - t_lcb * torch.sqrt(n * var)
    gp = (N / n_verifica) * Sd
    Hp = (N / n_curvatura) * H.to(torch.float64)
    ad = d.to(torch.float64).abs()
    return gp * ad - 0.5 * Hp * ad * ad


def t_guadagno(prop: torch.Tensor, S_: torch.Tensor, Q_: torch.Tensor, H: torch.Tensor, d: torch.Tensor,
               n_verifica: int, n_curvatura: int) -> torch.Tensor:
    """t di Student del guadagno: t = (S_dir - soglia) / sqrt(n s^2), soglia = (n_ver / n_curv) H |d| / 2 (in unita'
    della somma dei micro-lotti di verifica); t > 0 <=> guadagno di secondo ordine stimato > 0. float64.
    (27/9: estratta da `p_guadagno`, stesse operazioni; la usa anche --pv-accetta guadagno con z_guadagno_min.)"""
    n = n_verifica
    Sd = S_.to(torch.float64) * torch.sign(prop.to(torch.float64))
    soglia = 0.5 * (n / float(n_curvatura)) * H.to(torch.float64) * d.to(torch.float64).abs()
    var = ((Q_.to(torch.float64) - S_.to(torch.float64) ** 2 / n) / (n - 1)).clamp_min(0.0)
    den = torch.sqrt(n * var)
    x = Sd - soglia
    inf = torch.full_like(x, math.inf)
    return torch.where(den > 0, x / den.clamp_min(1e-300), torch.where(x > 0, inf, -inf))


def p_guadagno(prop: torch.Tensor, S_: torch.Tensor, Q_: torch.Tensor, H: torch.Tensor, d: torch.Tensor,
               n_verifica: int, n_curvatura: int) -> torch.Tensor:
    """p-valore (t di Student unilaterale, n-1 gradi) dell'ipotesi nulla «il salto non riduce la perdita del passo»:
    guadagno > 0  <=>  S_dir > soglia = (n_ver / n_curv) H |d| / 2 (in unita' della somma dei micro-lotti di verifica).
    t = (S_dir - soglia) / sqrt(n s^2) (`t_guadagno`). Poi BH per matrice: sotto la nulla globale nessun salto con
    probabilita' >= 1 - q; fra i salti scelti la quota di inutili e' <= q in media."""
    return sf_t(t_guadagno(prop, S_, Q_, H, d, n_verifica, n_curvatura), n_verifica - 1)


def seleziona(buf: BufferV, fam: S.DatiFamiglia, d_eff: torch.Tensor, n_verifica: int, rho: float,
              cfg: ConfigV = ConfigV(), *, n_perdita: int, n_curvatura: int,
              d_rif: Optional[torch.Tensor] = None) -> Selezione:
    """Selezione di fine passo per UNA famiglia. `d_eff`: la scala CORRENTE (E, R, NG), quella del forward dopo il
    passo P (fp16 esatta). Scoperte = BH sul t di Student (guardia contro il rumore) nella direzione della proposta;
    convenienti = scoperte con guadagno di SECONDO ORDINE > 0; poi per esperto in ordine di guadagno finche' la
    somma dei d^2 scelti sta nel budget del trust ratio (prefisso: niente salti di un intervallo). A pezzi.
    27/9 `cfg.accetta` "guadagno": scoperte = testati con z > z_min nella direzione della proposta, convenienti =
    scoperte con guadagno stimato > 0 e t del guadagno > z_guadagno_min (nessun BH); il resto invariato."""
    E, R, NG = fam.E, fam.R, fam.NG
    G = R * NG
    parti = {k: [] for k in ("e", "r", "g", "cand", "dir", "guadagno")}
    cont = {k: [] for k in ("testati", "scoperte", "convenienti", "budget", "scelti", "costo")}
    for e0 in range(0, E, cfg.pezzo_esperti):
        e1 = min(E, e0 + cfg.pezzo_esperti)
        ne = e1 - e0
        pr = buf.prop[e0:e1]
        d = d_eff[e0:e1].to(torch.float64)
        testati = ((pr != 0) & (d != 0)).reshape(ne, G)
        if d_rif is not None:                    # finestra di accumulo: fuori chi ha cambiato scala oltre la soglia
            dr = d_rif[e0:e1].to(torch.float64)
            fermi = (dr != 0) & ((d / torch.where(dr != 0, dr, torch.ones_like(dr)) - 1).abs() <= cfg.soglia_staleness)
            testati = testati & fermi.reshape(ne, G)
        p, z, s_dir = p_valori(pr, buf.S[e0:e1], buf.Q[e0:e1], n_verifica)
        if cfg.accetta == "bh":
            scop = bh(p.reshape(ne, G), testati, cfg.q_fdr) & (s_dir.reshape(ne, G) > 0)
            gq = guadagno_secondo_ordine(buf.S[e0:e1], pr, buf.H[e0:e1], d, n_perdita, n_verifica, n_curvatura)
            p_g = p_guadagno(pr, buf.S[e0:e1], buf.Q[e0:e1], buf.H[e0:e1], d, n_verifica, n_curvatura)
            conv = bh(p_g.reshape(ne, G), testati, cfg.q_guadagno) & (gq.reshape(ne, G) > 0)
        else:                                    # --pv-accetta guadagno: guadagno stimato > 0, senza BH
            scop = testati & (s_dir.reshape(ne, G) > 0) & (z.reshape(ne, G) > cfg.z_min)
            gq = guadagno_secondo_ordine(buf.S[e0:e1], pr, buf.H[e0:e1], d, n_perdita, n_verifica, n_curvatura)
            conv = scop & (gq.reshape(ne, G) > 0)
            if cfg.z_guadagno_min > 0:
                t_g = t_guadagno(pr, buf.S[e0:e1], buf.Q[e0:e1], buf.H[e0:e1], d, n_verifica, n_curvatura)
                conv = conv & (t_g.reshape(ne, G) > cfg.z_guadagno_min)
        bud = budget_esperti(d, nnz_gruppi(fam, e0, e1, cfg.pezzo_decodifica), rho)
        gu = torch.where(conv, gq.reshape(ne, G), torch.full((ne, G), -math.inf, dtype=torch.float64,
                                                             device=d.device))
        val, idx = torch.sort(gu, dim=1, descending=True, stable=True)
        costo = (d.reshape(ne, G) ** 2).gather(1, idx)
        fin = torch.isfinite(val)
        cum = torch.cumsum(torch.where(fin, costo, torch.zeros_like(costo)), dim=1)
        tieni = fin & (cum <= bud[:, None])
        ee, jj = torch.nonzero(tieni, as_tuple=True)
        gi = idx[ee, jj]
        e_abs, r_, g_ = ee + e0, gi // NG, gi % NG
        parti["e"].append(e_abs)
        parti["r"].append(r_)
        parti["g"].append(g_)
        parti["cand"].append(buf.cand[e_abs, r_, g_].to(torch.int64))
        sp = torch.sign(buf.prop[e_abs, r_, g_].to(torch.float64) * d_eff[e_abs, r_, g_].to(torch.float64))
        parti["dir"].append((-sp).to(torch.int8))
        parti["guadagno"].append(val[ee, jj])
        cont["testati"].append(testati.sum(1))
        cont["scoperte"].append(scop.sum(1))
        cont["convenienti"].append(conv.sum(1))
        cont["budget"].append(bud)
        cont["scelti"].append(tieni.sum(1))
        cont["costo"].append(torch.where(tieni, costo, torch.zeros_like(costo)).sum(1))
    dev = buf.prop.device
    vuoto = {"e": torch.int64, "r": torch.int64, "g": torch.int64, "cand": torch.int64, "dir": torch.int8,
             "guadagno": torch.float64}
    out = {k: (torch.cat(v) if v else torch.zeros(0, dtype=vuoto[k], device=dev)) for k, v in parti.items()}
    return Selezione(**out, **{k: torch.cat(v) for k, v in cont.items()})


def _sf_t_per_nu(t: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:
    """sf_t con gradi di liberta' per elemento (nu int64 >= 1): un calcolo per valore distinto."""
    p = torch.empty_like(t, dtype=torch.float64)
    for v in torch.unique(nu).tolist():
        mk = nu == v
        p[mk] = sf_t(t[mk], int(v))
    return p


def seleziona_bop(z: StatoBop, buf: BufferV, fam: S.DatiFamiglia, d_eff: torch.Tensor, n_verifica: int, rho: float,
                  cfg: ConfigV, bop: ConfigBop, *, n_perdita: int) -> Selezione:
    """Selezione V-Bop per UNA famiglia (dopo `StatoV.aggiorna_bop`). Per gruppo con candidato (eta' >= 1, H > 0):
    m^ = m / c, v^ = v / c (c = 1 - (1-gamma)^eta'), n_eff = n_verifica / sum w^2, errore standard sqrt(v^ / n_eff),
    z = m^ verso / se (t di Student con nu = round((n_verifica - 1) / sum w^2) gradi, tetto nu_max) -> BH (q_fdr);
    guadagno del passo N (m^ verso |d| - H d^2 / 2) > 0 con il suo test (m^ verso - H |d| / 2) / se -> BH (q_guadagno);
    poi budget rho garantito per esperto come `seleziona`. Con gamma = 1 e' il test di PV (stesso z, stessi gradi).
    27/9 `cfg.accetta` "guadagno": scoperte = testati con m^ verso > 0 e z > z_min, convenienti = scoperte col
    guadagno N (m^ verso |d| - H d^2 / 2) > 0 e (m^ verso - H|d|/2) / se > z_guadagno_min (nessun BH); budget,
    ordinamento e il resto invariati."""
    E, R, NG = fam.E, fam.R, fam.NG
    G = R * NG
    gam = float(bop.gamma)
    N = float(n_perdita)
    parti = {k: [] for k in ("e", "r", "g", "cand", "dir", "guadagno")}
    cont = {k: [] for k in ("testati", "scoperte", "convenienti", "budget", "scelti", "costo")}
    for e0 in range(0, E, cfg.pezzo_esperti):
        e1 = min(E, e0 + cfg.pezzo_esperti)
        ne = e1 - e0
        d = d_eff[e0:e1].to(torch.float64)
        eta, verso = z.eta_verso(e0, e1)
        c, sw2 = _correzione_eta(gam, eta)
        m_dir = _decodifica16(z.m[e0:e1], z.sm[e0:e1]) / c * verso
        v_hat = _decodifica16(z.r[e0:e1], z.sr[e0:e1]) ** 2 / c
        Hh = _decodifica16(z.H[e0:e1], z.sH[e0:e1])
        testati = ((eta >= 1) & (Hh > 0) & (d != 0)).reshape(ne, G)
        se = torch.sqrt(v_hat * sw2 / n_verifica)
        nu = torch.round((n_verifica - 1) / sw2).clamp(1, bop.nu_max).to(torch.int64)
        ad = d.abs()

        def z_di(x):
            inf = torch.full_like(x, math.inf)
            return torch.where(se > 0, x / se.clamp_min(1e-300),
                               torch.where(x > 0, inf, torch.where(x < 0, -inf, torch.zeros_like(x))))
        if cfg.accetta == "bh":
            p = _sf_t_per_nu(z_di(m_dir), nu)
            scop = bh(p.reshape(ne, G), testati, cfg.q_fdr) & (m_dir.reshape(ne, G) > 0)
            gq = N * (m_dir * ad - 0.5 * Hh * ad * ad)
            p_g = _sf_t_per_nu(z_di(m_dir - 0.5 * Hh * ad), nu)
            conv = scop & bh(p_g.reshape(ne, G), testati, cfg.q_guadagno) & (gq.reshape(ne, G) > 0)
        else:                                    # --pv-accetta guadagno: guadagno stimato > 0, senza BH
            scop = testati & (m_dir.reshape(ne, G) > 0) & (z_di(m_dir).reshape(ne, G) > cfg.z_min)
            gq = N * (m_dir * ad - 0.5 * Hh * ad * ad)
            conv = scop & (gq.reshape(ne, G) > 0)
            if cfg.z_guadagno_min > 0:
                conv = conv & (z_di(m_dir - 0.5 * Hh * ad).reshape(ne, G) > cfg.z_guadagno_min)
        bud = budget_esperti(d, nnz_gruppi(fam, e0, e1, cfg.pezzo_decodifica), rho)
        gu = torch.where(conv, gq.reshape(ne, G), torch.full((ne, G), -math.inf, dtype=torch.float64,
                                                             device=d.device))
        val, idx = torch.sort(gu, dim=1, descending=True, stable=True)
        costo = (d.reshape(ne, G) ** 2).gather(1, idx)
        fin = torch.isfinite(val)
        cum = torch.cumsum(torch.where(fin, costo, torch.zeros_like(costo)), dim=1)
        tieni = fin & (cum <= bud[:, None])
        ee, jj = torch.nonzero(tieni, as_tuple=True)
        gi = idx[ee, jj]
        e_abs, r_, g_ = ee + e0, gi // NG, gi % NG
        parti["e"].append(e_abs)
        parti["r"].append(r_)
        parti["g"].append(g_)
        parti["cand"].append(z.cand[e_abs, r_, g_].to(torch.int64))
        sp = verso[ee, r_, g_] * torch.sign(d_eff[e_abs, r_, g_].to(torch.float64))
        parti["dir"].append((-sp).to(torch.int8))
        parti["guadagno"].append(val[ee, jj])
        cont["testati"].append(testati.sum(1))
        cont["scoperte"].append(scop.sum(1))
        cont["convenienti"].append(conv.sum(1))
        cont["budget"].append(bud)
        cont["scelti"].append(tieni.sum(1))
        cont["costo"].append(torch.where(tieni, costo, torch.zeros_like(costo)).sum(1))
    dev = buf.prop.device
    vuoto = {"e": torch.int64, "r": torch.int64, "g": torch.int64, "cand": torch.int64, "dir": torch.int8,
             "guadagno": torch.float64}
    out = {k: (torch.cat(v) if v else torch.zeros(0, dtype=vuoto[k], device=dev)) for k, v in parti.items()}
    return Selezione(**out, **{k: torch.cat(v) for k, v in cont.items()})


# ═══════════════════════════════════ 6. salti sui byte e registro ═══════════════════════════════════
def posizione_t(c: torch.Tensor, tipo: str) -> tuple:
    """Colonna c di una riga -> (blocco nella riga, byte nel blocco, cifra n). Inverso di `_trit_da_byte`
    (vettoriale; `e2e_pv_misure.posizione` e' la versione scalare, T2 le confronta su tutte le colonne)."""
    nb, qk, _, _ = CS.LAYOUT[tipo]
    c = c.to(torch.int64)
    blk, p = c // qk, c % qk
    if tipo == "TQ1_0R_1024":
        qu, j = p // 256, p % 256
        qs0, qh0 = qu * 48, 192 + qu * 4
    else:
        j = p
        qs0, qh0 = torch.zeros_like(p), torch.full_like(p, 48)
    in1, in2 = j < 160, j < 240
    j2, j3 = (j - 160).clamp_min(0), (j - 240).clamp_min(0)
    byte = torch.where(in1, qs0 + j % 32, torch.where(in2, qs0 + 32 + j2 % 16, qh0 + j3 % 4))
    n = torch.where(in1, j // 32, torch.where(in2, j2 // 16, j3 // 4))
    return blk, byte, n


def _cifra(v: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    """Cifra n (0 = piu' significativa) del byte v in {0,1,2}: aritmetica di dequantize_row_tq1_0 (senza il -1)."""
    pw = torch.tensor(P3, dtype=torch.int32, device=v.device)[n]
    return (((v.to(torch.int32) * pw) & 255) * 3) >> 8


def _in_qh(ab: torch.Tensor, tipo: str) -> torch.Tensor:
    """ab = byte nella riga -> True se sta in una regione qh (4 cifre + cifra nulla)."""
    nb = CS.LAYOUT[tipo][0]
    bb = ab % nb
    if tipo == "TQ1_0R_1024":
        return (bb >= 192) & (bb < 208)
    return (bb >= 48) & (bb < 52)


def _valore_q(v: torch.Tensor, qh: torch.Tensor) -> torch.Tensor:
    """q del byte = sum_n cifra_n 3^(4-n) sulle cifre che il decoder legge (5 in qs, 4 in qh)."""
    q = torch.zeros_like(v, dtype=torch.int32)
    for n in range(5):
        cn = _cifra(v, torch.full_like(v, n, dtype=torch.int64))
        if n == 4:
            cn = torch.where(qh, torch.zeros_like(cn), cn)
        q = q + cn * (3 ** (4 - n))
    return q


def salta_byte(B: torch.Tensor, tipo: str, e: torch.Tensor, r: torch.Tensor, c: torch.Tensor,
               t_attesi: torch.Tensor, t_nuovi: torch.Tensor) -> int:
    """Scrive in place sui byte B (E, R, NBR) uint8 i trit t_nuovi alle posizioni (e, r, c), controllando che quelli
    ATTUALI siano t_attesi (altrimenti solleva, nulla e' scritto). Un solo cambio di cifra per posizione, di un
    livello; piu' posizioni nello stesso byte si sommano su q. Ritorna i byte riscritti."""
    if e.numel() == 0:
        return 0
    assert B.dtype == torch.uint8 and B.dim() == 3 and B.is_contiguous()
    E, R, NBR = B.shape
    nb, qk = CS.LAYOUT[tipo][0], CS.LAYOUT[tipo][1]
    C = NBR // nb * qk
    dev = B.device
    e, r, c = e.to(dev, torch.int64), r.to(dev, torch.int64), c.to(dev, torch.int64)
    ta, tn = t_attesi.to(dev, torch.int32), t_nuovi.to(dev, torch.int32)
    if bool(((e < 0) | (e >= E) | (r < 0) | (r >= R) | (c < 0) | (c >= C)).any()):
        raise IndexError("posizione fuori dal tensore")
    pos = (e * R + r) * C + c                                          # posizione di codice (unicita')
    if torch.unique(pos).numel() != pos.numel():
        raise ValueError("due salti sullo stesso codice")
    dt = tn - ta
    if bool((dt.abs() != 1).any()) or bool((tn.abs() > 1).any()):
        raise ValueError("salto non di UN livello dentro {-1,0,1}")
    blk, byte, n = posizione_t(c, tipo)
    ab = blk * nb + byte
    piatto = (e * R + r) * NBR + ab
    Bf = B.view(-1)
    ora = _cifra(Bf[piatto], n) - 1
    if not torch.equal(ora, ta):
        raise RuntimeError(f"trit attuali diversi dagli attesi in {int((ora != ta).sum())} posizioni: registro e "
                           "byte non sono allineati")
    u, inv = torch.unique(piatto, return_inverse=True)
    pw = torch.tensor([81, 27, 9, 3, 1], dtype=torch.int32, device=dev)[n]
    dq = torch.zeros(u.numel(), dtype=torch.int32, device=dev).index_add_(0, inv, dt * pw)
    vu = Bf[u].to(torch.int32)
    qh = _in_qh(u % NBR, tipo)
    q = _valore_q(vu, qh) + dq
    if bool(((q < 0) | (q > 242) | (qh & (q % 3 != 0)) | (qh & (q > 240))).any()):
        raise RuntimeError("q fuori dal campo dopo i salti")
    Bf[u] = (((q * 256 + 242) // 243) & 255).to(torch.uint8)
    return int(u.numel())


REG_DTYPE = np.dtype([("L", "u1"), ("f", "u1"), ("e", "u1"), ("r", "<u2"), ("c", "<u2"),
                      ("t_vecchio", "i1"), ("t_nuovo", "i1")])
assert REG_DTYPE.itemsize == 9


@dataclass
class Registro:
    """Salti in ordine di applicazione: 9 byte a salto (L, f, e, r, c, t_vecchio, t_nuovo)."""
    righe: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=REG_DTYPE))

    def __len__(self) -> int:
        return int(self.righe.shape[0])

    @staticmethod
    def unisci(regs: list) -> "Registro":
        """Concatenazione per LETTURA (conteggi, verifica d'export: primo t_vecchio / ultimo t_nuovo per posizione).
        ⛔ Non si APPLICA: lo stesso codice puo' saltare in passi diversi e `salta_byte` rifiuta i doppioni in un
        solo registro (revisione 26/9). Rigioco, annullamento ed export prendono la LISTA dei registri, in ordine."""
        return Registro(np.concatenate([r.righe for r in regs]) if regs else np.zeros(0, dtype=REG_DTYPE))

    def inverso(self) -> "Registro":
        inv = self.righe[::-1].copy()
        inv["t_vecchio"], inv["t_nuovo"] = self.righe[::-1]["t_nuovo"], self.righe[::-1]["t_vecchio"]
        return Registro(inv)

    def chiavi(self) -> list:
        return sorted({(int(a), int(b)) for a, b in zip(self.righe["L"], self.righe["f"])})

    def di(self, L: int, f: int) -> np.ndarray:
        return self.righe[(self.righe["L"] == L) & (self.righe["f"] == f)]

    def salva(self, p: Path) -> str:
        """Scrittura ATOMICA (revisione 26/9: un crollo a meta' lasciava 256 B al posto del registro valido): file
        temporaneo accanto, fsync, rinomina, fsync della cartella. Ritorna lo sha256 del file."""
        buf = io.BytesIO()
        np.save(buf, self.righe, allow_pickle=False)
        return _scrivi_atomico(Path(p), buf.getvalue())

    @staticmethod
    def salva_lista(p: Path, registri: list) -> str:
        """La LISTA dei registri (uno per passo) in un .npz atomico: righe concatenate + fine di ogni registro."""
        fine = np.cumsum([len(r) for r in registri], dtype=np.int64) if registri else np.zeros(0, dtype=np.int64)
        righe = np.concatenate([r.righe for r in registri]) if registri else np.zeros(0, dtype=REG_DTYPE)
        buf = io.BytesIO()
        np.savez(buf, righe=righe, fine=fine)
        return _scrivi_atomico(Path(p), buf.getvalue())

    @staticmethod
    def carica_lista(p: Path, sha256: Optional[str] = None) -> list:
        p = Path(p)
        dati = p.read_bytes()
        if sha256 is not None and hashlib.sha256(dati).hexdigest() != sha256:
            raise RuntimeError(f"{p}: sha256 diverso da quello della ripresa")
        with np.load(io.BytesIO(dati), allow_pickle=False) as z:
            righe, fine = z["righe"], z["fine"]
        if righe.dtype != REG_DTYPE:
            raise TypeError(f"{p}: dtype {righe.dtype}")
        inizi = np.concatenate([[0], fine[:-1]]) if fine.size else np.zeros(0, dtype=np.int64)
        return [Registro(righe[a:b].copy()) for a, b in zip(inizi, fine)]

    @staticmethod
    def carica(p: Path, sha256: Optional[str] = None) -> "Registro":
        p = Path(p)
        if sha256 is not None and hashlib.sha256(p.read_bytes()).hexdigest() != sha256:
            raise RuntimeError(f"{p}: sha256 diverso da quello della ripresa")
        r = np.load(p, allow_pickle=False)
        if r.dtype != REG_DTYPE:
            raise TypeError(f"{p}: dtype {r.dtype}")
        return Registro(r)


def _scrivi_atomico(p: Path, dati: bytes) -> str:
    tmp = p.with_name(p.name + ".parziale")
    try:
        with open(tmp, "wb") as fh:
            fh.write(dati)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, p)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    fd = os.open(str(p.parent), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return hashlib.sha256(dati).hexdigest()


def famiglia_scrivibile(fam: S.DatiFamiglia) -> S.DatiFamiglia:
    """Su CPU: copia dei byte (i byte di `famiglia_da_tensore` possono essere una vista del memmap in sola
    lettura). Marca la famiglia come scrivibile: `applica_salti` rifiuta le altre su CPU."""
    fam.blocchi = fam.blocchi.clone()
    fam._pv_scrivibile = True
    return fam


def _controlla_scrivibile(fam: S.DatiFamiglia) -> None:
    if fam.blocchi.device.type == "cpu" and not getattr(fam, "_pv_scrivibile", False):
        raise PermissionError("byte su CPU non passati da `famiglia_scrivibile`: potrebbero essere il memmap del "
                              "GGUF in sola lettura")


def salta_registro(fam: S.DatiFamiglia, righe: np.ndarray) -> int:
    """Applica righe di registro (gia' filtrate per questa famiglia) ai byte: t_vecchio -> t_nuovo."""
    _controlla_scrivibile(fam)
    if righe.shape[0] == 0:
        return 0
    f = lambda k, dt: torch.from_numpy(righe[k].astype(dt))  # noqa: E731
    return salta_byte(fam.blocchi, fam.tipo, f("e", np.int64), f("r", np.int64), f("c", np.int64),
                      f("t_vecchio", np.int32), f("t_nuovo", np.int32))


def applica_salti(fam: S.DatiFamiglia, sel: Selezione, L: Optional[int] = None) -> Registro:
    """I salti di una Selezione sui byte della famiglia; ritorna il Registro. t_vecchio si legge dai byte."""
    _controlla_scrivibile(fam)
    L = int(fam.strato) if L is None else int(L)
    if len(sel) == 0:
        return Registro()
    righe = righe_salti(fam, sel, L)
    salta_registro(fam, righe)
    return Registro(righe)


def righe_salti(fam: S.DatiFamiglia, sel: Selezione, L: Optional[int] = None) -> np.ndarray:
    """Righe di registro (REG_DTYPE) dei salti di una Selezione, SENZA scriverli: t_vecchio letto dai byte di ora,
    t_nuovo = t_vecchio + dir (27/9 ramo v-interazione-b: le dosi applicano sottoinsiemi di queste righe). Stesse
    operazioni che `applica_salti` faceva prima di scrivere (applica_salti = righe_salti + salta_registro)."""
    L = int(fam.strato) if L is None else int(L)
    if len(sel) == 0:
        return np.zeros(0, dtype=REG_DTYPE)
    c = sel.colonne(fam.gs)
    blk, byte, n = posizione_t(c.to(fam.blocchi.device), fam.tipo)
    piatto = (sel.e.to(fam.blocchi.device) * fam.R + sel.r.to(fam.blocchi.device)) * fam.NBR + blk * fam.nb + byte
    tv = (_cifra(fam.blocchi.view(-1)[piatto], n) - 1).to(torch.int32)
    tn = tv + sel.dir.to(fam.blocchi.device, torch.int32)
    if bool((tn.abs() > 1).any()):
        raise RuntimeError("selezione con un salto fuori da {-1,0,1}: la fattibilita' della proposta e' saltata")
    righe = np.zeros(len(sel), dtype=REG_DTYPE)
    righe["L"], righe["f"] = L, FAMIGLIE_IDX[fam.nome]
    righe["e"], righe["r"], righe["c"] = sel.e.cpu().numpy(), sel.r.cpu().numpy(), c.cpu().numpy()
    righe["t_vecchio"], righe["t_nuovo"] = tv.cpu().numpy(), tn.cpu().numpy()
    return righe


def annulla(famiglie: dict, reg: Registro) -> None:
    """Annulla un registro (ordine inverso, controllando che i byte portino i t_nuovo)."""
    rigioca(famiglie, [reg.inverso()])


def rigioca(famiglie: dict, registri: list) -> None:
    """Applica i registri in ordine. famiglie: {(L, nome): DatiFamiglia}. Ripresa = byte del GGUF + rigioco."""
    for reg in registri:
        for (L, fi) in reg.chiavi():
            salta_registro(famiglie[(L, NOMI_IDX[fi])], reg.di(L, fi))


# ═══════════════════════════════════ 7. export ═══════════════════════════════════
def nome_tensore(L: int, fi: int) -> str:
    return f"blk.{L}.{S.nome_tensore_famiglia(NOMI_IDX[fi])}.weight"


def _byte_3d(raw) -> torch.Tensor:
    """Byte di un tensore del file come (E, R, NBR): un denso 2D (--densi-ternari) e' la famiglia di UN esperto.
    Vista (nessuna copia): le scritture di `salta_byte` arrivano al memmap."""
    B = torch.from_numpy(raw)
    return B.unsqueeze(0) if B.dim() == 2 else B


def patcha_codici(dst: Path, registri: list, nome: Callable = nome_tensore) -> dict:
    """Scrive i registri nei byte dei codici di `dst` (una COPIA del GGUF: mai il sorgente), memmap r+, stessa
    aritmetica di `salta_byte`, UN registro dopo l'altro (salti ripetuti dello stesso codice fra passi ammessi).
    Ritorna {tensore: byte riscritti}. Chi esporta usa `esporta_codici` (copia .parziale, verifica, rinomina)."""
    from gguf import GGUFReader
    r = GGUFReader(str(dst), mode="r+")
    tens = {t.name: t for t in r.tensors}
    scritti: dict = {}
    for reg in registri:
        for (L, fi) in reg.chiavi():
            t = tens[nome(L, fi)]
            raw = t.data                                          # vista r+ (E, R, NBR) del file ((R, NBR) se denso)
            B = _byte_3d(raw)
            righe = reg.di(L, fi)
            f = lambda k, dt: torch.from_numpy(righe[k].astype(dt))  # noqa: E731
            scritti[t.name] = scritti.get(t.name, 0) + salta_byte(
                B, t.tensor_type.name, f("e", np.int64), f("r", np.int64), f("c", np.int64),
                f("t_vecchio", np.int32), f("t_nuovo", np.int32))
    r.data.flush()
    del r, tens
    return scritti


def verifica_patch_codici(src: Path, dst: Path, registri: list, nome: Callable = nome_tensore,
                          pezzo: int = 1 << 26) -> dict:
    """Controlli d'export, rileggendo i FILE (non i tensori in memoria):
    (1) stessa taglia; (2) confronto a pezzi di `pezzo` byte: ogni byte diverso sta fra i byte dei codici del
    registro (mai un campo d, mai un altro tensore, mai l'intestazione); (3) nei blocchi toccati, trit di dst ==
    trit di src con lo stato FINALE del registro (per ogni posizione: il primo t_vecchio deve essere quello di src,
    l'ultimo t_nuovo quello di dst), tutti gli altri trit identici; (4) campi d dei blocchi toccati identici e
    codice C di llama.cpp sui blocchi toccati di dst == trit x d (de-ruotato per TQ1_0R_1024), al bit."""
    from gguf import GGUFReader
    import e2e_esporta as EX                                         # sola lettura: _deq_c (codice C)
    from tq1_0r_1024_codec import S1024
    src, dst = Path(src), Path(dst)
    if src.stat().st_size != dst.stat().st_size:
        raise RuntimeError("taglie diverse")
    rs, rd = GGUFReader(str(src)), GGUFReader(str(dst))
    ts, td = {t.name: t for t in rs.tensors}, {t.name: t for t in rd.tensors}
    reg = Registro.unisci(registri)
    permessi, blocchi_c, trit_cambiati = [], 0, 0
    for (L, fi) in reg.chiavi():
        righe = reg.di(L, fi)
        t, u = ts[nome(L, fi)], td[nome(L, fi)]
        tipo = t.tensor_type.name
        nb, qk, off, k = CS.LAYOUT[tipo]
        forma = np.asarray(t.data).shape
        E, R, NBR = (1,) + tuple(forma) if len(forma) == 2 else forma     # 28/9: denso 2D = un esperto
        C, bpr = NBR // nb * qk, NBR // nb
        e_, r_, c_ = (righe[x].astype(np.int64) for x in ("e", "r", "c"))
        blk, byte, _ = (x.numpy() for x in posizione_t(torch.from_numpy(c_), tipo))
        permessi.append(int(t.data_offset) + (e_ * R + r_) * NBR + blk * nb + byte)
        chiave_pos = (e_ * R + r_) * C + c_
        uk, i_primo = np.unique(chiave_pos, return_index=True)
        _, i_rev = np.unique(chiave_pos[::-1], return_index=True)
        i_ultimo = len(chiave_pos) - 1 - i_rev
        gb = (e_ * R + r_) * bpr + blk                                # blocco nel tensore
        bi = np.unique(gb)
        Bs = np.ascontiguousarray(np.asarray(t.data).reshape(-1, nb)[bi])
        Bd = np.ascontiguousarray(np.asarray(u.data).reshape(-1, nb)[bi])
        Ts = CS.trit_da_blocchi(torch.from_numpy(Bs), tipo)
        Td = CS.trit_da_blocchi(torch.from_numpy(Bd), tipo)
        riga_b = torch.from_numpy(np.searchsorted(bi, gb[i_primo]))
        col_b = torch.from_numpy(c_[i_primo] % qk)
        if not torch.equal(Ts[riga_b, col_b].to(torch.int64), torch.from_numpy(righe["t_vecchio"][i_primo]
                                                                                  .astype(np.int64))):
            raise RuntimeError(f"{nome(L, fi)}: trit del SORGENTE != primo t_vecchio del registro")
        atteso = Ts.clone()
        atteso[riga_b, col_b] = torch.from_numpy(righe["t_nuovo"][i_ultimo].astype(np.int8))
        if not torch.equal(Td, atteso):
            raise RuntimeError(f"{nome(L, fi)}: trit dei blocchi toccati != sorgente + registro "
                               f"({int((Td != atteso).sum())} diversi)")
        if not np.array_equal(Bs[:, off:off + 2 * k], Bd[:, off:off + 2 * k]):
            raise RuntimeError(f"{nome(L, fi)}: campi d cambiati")
        dd = torch.from_numpy(Bd[:, off:off + 2 * k].copy().view(np.float16).astype(np.float32))
        W = CS.pesi_da_trit_scale(Td, dd, tipo, torch.from_numpy(S1024) if tipo != "TQ1_0" else None).numpy()
        Wc = EX._deq_c(Bd, tipo)
        if not np.array_equal(W, Wc):
            raise RuntimeError(f"{nome(L, fi)}: codice C di llama.cpp != trit nuovi x d (max {np.abs(W - Wc).max()})")
        blocchi_c += int(bi.size)
        trit_cambiati += int((Ts != Td).sum())
    del rs, rd, ts, td
    permessi = np.unique(np.concatenate(permessi)) if permessi else np.zeros(0, dtype=np.int64)
    a = np.memmap(src, dtype=np.uint8, mode="r")
    b = np.memmap(dst, dtype=np.uint8, mode="r")
    diversi = 0
    for i0 in range(0, a.shape[0], pezzo):
        i1 = min(a.shape[0], i0 + pezzo)
        dv = np.flatnonzero(np.asarray(a[i0:i1]) != np.asarray(b[i0:i1])) + i0
        if dv.size:
            fuori = dv[~np.isin(dv, permessi)]
            if fuori.size:
                raise RuntimeError(f"{fuori.size} byte diversi FUORI dai byte del registro (primo {int(fuori[0])})")
            diversi += int(dv.size)
    del a, b
    return {"byte_diversi": diversi, "byte_permessi": int(permessi.size), "blocchi_codice_c": blocchi_c,
            "trit_cambiati": trit_cambiati}


def esporta_codici(scalato: Path, dst: Path, registri: list, nome: Callable = nome_tensore) -> dict:
    """Export dei CODICI sopra un GGUF gia' esportato con le scale del checkpoint (`e2e_esporta`, che verifica le
    scale al bit e decodifica i trit del sorgente, invariati): copia in .parziale, `patcha_codici` registro per
    registro, `verifica_patch_codici` contro `scalato` (campi d identici, byte diversi solo dove i registri dicono,
    trit finali = primo t_vecchio / ultimo t_nuovo, codice C di llama.cpp == trit x d), poi rinomina. Mai sul
    sorgente; un errore lascia solo il .parziale (cancellato)."""
    scalato, dst = Path(scalato), Path(dst)
    if dst.resolve() == scalato.resolve():
        raise ValueError("esporta_codici: la destinazione e' il sorgente")
    tmp = dst.with_name(dst.name + ".parziale")
    shutil.copyfile(scalato, tmp)
    try:
        scritti = patcha_codici(tmp, registri, nome)
        ver = verifica_patch_codici(scalato, tmp, registri, nome)
        with open(tmp, "rb+") as fh:
            os.fsync(fh.fileno())
        os.replace(tmp, dst)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return {"byte_scritti": scritti, **ver, "registri": len(registri), "salti": sum(len(r) for r in registri)}


def densi_dei_registri(registri: list) -> tuple:
    """28/9: famiglie dense (--densi-ternari) che compaiono nei registri (f >= 3), ordine canonico."""
    fi = set()
    for r in registri:
        fi.update(int(x) for x in np.unique(r.righe["f"]) if int(x) >= 3)
    return S.famiglie_dense([NOMI_IDX[i] for i in sorted(fi)])


def ricarica_codici(scalato: Path, dst: Path, registri: list, strati: list, token: int = 64,
                    densi_ternari: Optional[tuple] = None) -> dict:
    """Controllo di RICARICA del file finale (revisione 26/9: `ricarica` di e2e_esporta ricostruisce da sorgente +
    theta, senza rigioco): per ogni strato, studente dal file `scalato` + rigioco dei registri sui suoi byte contro
    studente dal file `dst`: byte dei codici identici e uscita dello strato identica AL BIT (CPU, motore torch).
    28/9 `densi_ternari` (None = quelli dei registri, `densi_dei_registri`): i due studenti con le stesse famiglie dense
    a esperto unico, cosi' i salti sui densi si rigiocano e si confrontano come quelli degli esperti."""
    densi = densi_dei_registri(registri) if densi_ternari is None else S.famiglie_dense(densi_ternari)
    kw = {"densi_ternari": densi} if densi else {}
    esiti = {}
    for L in strati:
        sa = S.Studente(gguf=Path(scalato), strati=[L], ricetta="B", motore_nome="torch", device="cpu",
                        log=lambda *_: None, **kw)
        sb = S.Studente(gguf=Path(dst), strati=[L], ricetta="B", motore_nome="torch", device="cpu",
                        log=lambda *_: None, **kw)
        fams = {}
        for n, f in sa.esperti(L).fam.items():
            fams[(L, n)] = famiglia_scrivibile(f)
        regs_L = [Registro(r.righe[r.righe["L"] == L]) for r in registri]
        rigioca(fams, [r for r in regs_L if len(r)])
        byte_ok = all(torch.equal(sa.esperti(L).fam[n].blocchi, sb.esperti(L).fam[n].blocchi) for n in fams_n(sa, L))
        g = torch.Generator().manual_seed(20260927 + L)
        x = (torch.randn(1, token, sa.cfg.hidden_size, generator=g) * 0.5).to(sa.ric.dtype_residuo)
        with torch.no_grad():
            ya = sa.avanti(torch.zeros(1, token, dtype=torch.long), x0=x)["xL"]
            yb = sb.avanti(torch.zeros(1, token, dtype=torch.long), x0=x)["xL"]
        esiti[L] = {"byte_codici_uguali": byte_ok, "uscita_al_bit": bool(torch.equal(ya, yb)),
                    "salti_nello_strato": sum(len(r) for r in regs_L)}
        del sa, sb
        if not (esiti[L]["byte_codici_uguali"] and esiti[L]["uscita_al_bit"]):
            raise RuntimeError(f"ricarica_codici strato {L} FALLITA: {esiti[L]}")
    return esiti


def fams_n(st, L: int) -> list:
    return list(st.esperti(L).fam.keys())


# ═══════════════════════════════════ 8. regione di fiducia ═══════════════════════════════════
class RegioneFiducia:
    """Sonda del progetto §3: i salti si tengono solo se la perdita sulla sonda (un micro-lotto mai visto) NON
    peggiora; altrimenti si annullano dal registro e rho si dimezza. I primi len(prova) passi provano i rho di
    `prova` (taratura) e poi si tiene quello che ha fatto scendere di piu' la perdita; se nessuno l'ha fatta
    scendere, min(prova)/2.
    27/9 `adattivo` (--pv-rho-adattivo, TECNICHE_E2E_PV.md §19): regione di fiducia vera. r = calo effettivo della
    sonda / calo previsto (`guadagno_previsto`, stesse unita' della sonda); r > 0,75 -> rho x 2 (tetto rho_max),
    r < 0,25 -> rho / 2 (pavimento rho_min), altrimenti invariato; senza salti (r non definito) invariato. I salti si
    annullano ancora quando la sonda sale (r < 0 -> rho / 2). Spento: identico a prima."""

    def __init__(self, prova=(0.01, 0.03, 0.1), rho_min: float = 1e-4, adattivo: bool = False,
                 rho_max: float = 0.1):
        self.prova = list(prova)
        self.rho: Optional[float] = None
        self.rho_min = rho_min
        self.adattivo = bool(adattivo)
        self.rho_max = float(rho_max)
        self.storia: list = []

    def rho_passo(self) -> float:
        n = len(self.storia)
        return self.prova[n] if n < len(self.prova) else float(self.rho)

    def _adatta(self, rho: float, r: Optional[float], tiene: bool) -> float:
        if r is None:
            return rho if tiene else max(self.rho_min, rho / 2)
        if r > 0.75:
            return max(rho, min(self.rho_max, 2 * rho))
        if r < 0.25:
            return max(self.rho_min, rho / 2)
        return rho

    def esito(self, rho: float, perdita_prima: float, perdita_dopo: float, n_salti: int,
              guadagno_previsto: Optional[float] = None) -> bool:
        tiene = perdita_dopo <= perdita_prima
        voce = {"rho": rho, "prima": perdita_prima, "dopo": perdita_dopo, "salti": n_salti,
                "delta": perdita_dopo - perdita_prima, "tenuto": tiene}
        r = None
        if self.adattivo:
            if n_salti > 0 and guadagno_previsto is not None and guadagno_previsto > 0:
                r = (perdita_prima - perdita_dopo) / guadagno_previsto
            voce["r"] = r
        self.storia.append(voce)
        n = len(self.storia)
        if self.adattivo and n >= len(self.prova) and (len(self.prova) == 1 or n > len(self.prova)):
            self.rho = self._adatta(rho, r, tiene)
            voce["rho_dopo"] = self.rho
        elif n == len(self.prova):
            buoni = [s for s in self.storia if s["tenuto"] and s["delta"] < 0]
            self.rho = min(buoni, key=lambda s: s["delta"])["rho"] if buoni else min(self.prova) / 2
        elif n > len(self.prova) and not tiene:
            self.rho = max(self.rho_min, self.rho / 2)
        return tiene

    def esito_dosi(self, rho: float, perdita_prima: float, perdita_dopo: float, n_salti: int, tiene: bool,
                   parziale: bool, guadagno_previsto: Optional[float] = None) -> bool:
        """27/9 ramo v-interazione-b (--pv-sonda-dosi): la decisione (`tiene`) la prende la sonda a dosi con conferma,
        non il confronto prima/dopo di qui. Regole di rho, come `esito` salvo:
          - `tiene` e dose PARZIALE (tenuti meno dei salti scelti, per dose o ricentro): rho NON scende (la dose ha gia'
            separato i salti che interagiscono; dimezzare rho svuoterebbe il V senza curare la causa, §32.4); col rho
            adattivo puo' ancora salire (r > 0,75), cioe' rho = max(rho, regola(r));
          - `tiene` e dose piena: regola di prima (adattivo: r; non adattivo: invariato);
          - non `tiene` (nessuna dose migliora la sonda A, o la seconda sonda non conferma): rho / 2 come oggi.
        r = calo effettivo / calo previsto dei salti TENUTI (`guadagno_previsto` dei soli tenuti), misurato da chi chiama
        sulla sonda di conferma (non su quella che ha scelto la dose: niente maledizione del vincitore)."""
        voce = {"rho": rho, "prima": perdita_prima, "dopo": perdita_dopo, "salti": n_salti,
                "delta": perdita_dopo - perdita_prima, "tenuto": bool(tiene), "dose_parziale": bool(parziale)}
        r = None
        if self.adattivo:
            if tiene and n_salti > 0 and guadagno_previsto is not None and guadagno_previsto > 0:
                r = (perdita_prima - perdita_dopo) / guadagno_previsto
            voce["r"] = r
        self.storia.append(voce)
        n = len(self.storia)
        if self.adattivo and n >= len(self.prova) and (len(self.prova) == 1 or n > len(self.prova)):
            nuovo = self._adatta(rho, r, tiene)
            self.rho = max(rho, nuovo) if (tiene and parziale) else nuovo
            voce["rho_dopo"] = self.rho
        elif n == len(self.prova):
            buoni = [s for s in self.storia if s["tenuto"] and s["delta"] < 0]
            self.rho = min(buoni, key=lambda s: s["delta"])["rho"] if buoni else min(self.prova) / 2
        elif n > len(self.prova) and not tiene:
            self.rho = max(self.rho_min, self.rho / 2)
        return bool(tiene)


# ═══════════════════════ 8b. sonda a dosi e ricentro (27/9 ramo v-interazione-b, TECNICHE_E2E_PV.md §32) ═══════════════════════
FRAZIONI_DOSI = (1 / 256, 1 / 64, 1 / 16, 1 / 4, 1.0)
# 29/9 ramo punto-medio: granularita' delle primitive. famiglia = (strato, famiglia); tipo = famiglia su tutti gli strati;
# strato = tutte le famiglie di uno strato; esperto = (strato, famiglia, esperto). Il costo e' UN indietro della sonda per
# primitiva prezzata (al massimo pm_max) e per funzionale.
PRIMITIVE_PM = ("famiglia", "tipo", "strato", "esperto")


def prezzo_punto_medio(dw: torch.Tensor, g0: torch.Tensor, g1: torch.Tensor) -> torch.Tensor:
    """Prezzo di ogni componente di una mossa al punto medio, sugli stati INTERI (regola del trapezio):
    v = 1/2 dw (g0 + g1), g0 = grad F nello stato di partenza, g1 = grad F nello stato con la mossa applicata. A curvatura
    costante (F quadratica) grad F(q0 + d/2) = (grad F(q0) + grad F(q0 + d)) / 2 esattamente, quindi sum v = F(q0 + d) -
    F(q0) ESATTO (arXiv 2609.09867 Prop. 2; errore del trapezio <= L_H |d|^3 / 12 in generale). float64."""
    return 0.5 * dw.to(torch.float64) * (g0.to(torch.float64) + g1.to(torch.float64))


def scegli_primitive(s_obj: torch.Tensor, s_vin: Optional[torch.Tensor] = None) -> list:
    """Scelta delle primitive (indici, in ORDINE di dose) dai prezzi al punto medio. Un funzionale: quelle con s_obj < 0,
    dal piu' negativo. Due (obiettivo A, vincolo B, arXiv 2609.09867 §6(a): min sum s_A soggetto a sum s_B <= 0): prima le
    «sicure» (s_A < 0 e s_B <= 0) per s_A crescente, poi le «costose» (s_A < 0, s_B > 0) per s_A / s_B crescente (piu'
    obiettivo per unita' di vincolo) finche' la somma di s_B resta <= 0 (greedy dello zaino frazionario: esatto al
    rilassamento lineare salvo l'ultima). Prezzi non finiti = primitiva non prezzata: esclusa."""
    so = s_obj.to(torch.float64)
    ok = torch.isfinite(so) & (so < 0)
    if s_vin is None:
        idx = torch.nonzero(ok).reshape(-1)
        return idx[torch.argsort(so[idx], stable=True)].tolist()
    sv = s_vin.to(torch.float64)
    ok = ok & torch.isfinite(sv)
    sicure = torch.nonzero(ok & (sv <= 0)).reshape(-1)
    sicure = sicure[torch.argsort(so[sicure], stable=True)].tolist()
    costose = torch.nonzero(ok & (sv > 0)).reshape(-1)
    costose = costose[torch.argsort(so[costose] / sv[costose], stable=True)].tolist()
    scelte, margine = list(sicure), -float(sv[sicure].sum()) if sicure else 0.0
    for i in costose:
        if float(sv[i]) <= margine:
            scelte.append(i)
            margine -= float(sv[i])
    return scelte


def ordina_primitive(s_obj: torch.Tensor, s_vin: Optional[torch.Tensor] = None) -> list:
    """29/9 ramo pv-lettere-dosi (`ConfigDosi.pm_golosa`): lo stesso ordine di `scegli_primitive` (sicure per s_A
    crescente, poi costose per s_A / s_B crescente) ma SENZA il taglio del bilancio previsto sum s_B <= 0: il vincolo si
    verifica MISURANDO la sonda generica dopo ogni primitiva (i prezzi al punto medio non vedono l'interazione FRA
    primitive, solo quella dentro). Un funzionale: identico a `scegli_primitive`."""
    if s_vin is None:
        return scegli_primitive(s_obj)
    so, sv = s_obj.to(torch.float64), s_vin.to(torch.float64)
    ok = torch.isfinite(so) & (so < 0) & torch.isfinite(sv)
    sicure = torch.nonzero(ok & (sv <= 0)).reshape(-1)
    costose = torch.nonzero(ok & (sv > 0)).reshape(-1)
    return (sicure[torch.argsort(so[sicure], stable=True)].tolist()
            + costose[torch.argsort(so[costose] / sv[costose], stable=True)].tolist())


@dataclass
class ConfigDosi:
    """--pv-sonda-dosi: frazioni (crescenti, in (0, 1]) dei salti scelti, prefissi annidati in ordine di guadagno;
    --pv-ricentra: V ricentrato dopo la dose, al massimo `ricentra_giri` dimezzamenti confermati dalla sonda B."""
    frazioni: tuple = FRAZIONI_DOSI
    ricentra: bool = False
    ricentra_giri: int = 3
    # 29/9 ramo punto-medio (--pv-punto-medio, arXiv 2609.09867 Prop. 2 e §6(a); matematica/punto_medio_pv.md). Spento
    # (default) = il flusso di sempre al bit. Acceso: i salti scelti si raggruppano in PRIMITIVE (`pm_primitiva`); le
    # `pm_max` primitive col guadagno previsto piu' alto si prezzano al PUNTO MEDIO PROPRIO con la regola del trapezio
    # sugli stati interi (s_i = 1/2 d_i^T (grad F(q0) + grad F(q0 + d_i)), esatta a curvatura costante: il mezzo trit non
    # esiste nel formato), un indietro della sonda per primitiva (+ uno in q0); dentro ogni primitiva, con `pm_filtra`, si
    # tengono i soli salti col prezzo al centro della primitiva < 0; le primitive col prezzo < 0 (con due funzionali:
    # obiettivo A, vincolo sum s_B <= 0) sono le DOSI (prefissi di primitive), poi conferma e ricentro come sempre.
    punto_medio: bool = False
    pm_primitiva: str = "famiglia"
    pm_max: int = 12
    pm_filtra: bool = True
    # 29/9 ramo pv-lettere-dosi (--pv-pm-golosa, solo con punto_medio). Spento (default) = identico al bit. Acceso: le dosi
    # non sono piu' prefissi della scala `frazioni` giudicati dall'argmin della sonda GENERICA A, ma una prova GOLOSA
    # primitiva per primitiva nell'ordine di `ordina_primitive`: si applica la primitiva, si misura il funzionale
    # OBIETTIVO della decisione (`misura_obiettivo_dosi`, le 4 lettere su M1; senza, la sonda A) e la sonda A; si tiene
    # se l'obiettivo scende rispetto allo stato corrente E la sonda A resta <= quella di partenza (il vincolo sum s_B <= 0
    # di arXiv 2609.09867 §6(a), MISURATO invece che previsto), altrimenti si annulla al bit e si passa alla successiva.
    # Nessuna soglia nuova: i confronti sono gli stessi della sonda a dosi (<, <=). Poi conferma B, ricentro e conferma
    # dell'obiettivo su M2 come sempre.
    pm_golosa: bool = False
    # 29/9 ramo tt3-cure (--pv-ricentra-obiettivo, solo con ricentra e pm_golosa). Spento (default) = identico al bit.
    # Causa (TT3 passo V 2): la dose golosa e' decisa dalle LETTERE (M1) ma il ricentro toglie i salti «dannosi» per la
    # sonda GENERICA e ne accetta i giri guardando solo B; la conferma finale sulle lettere M2 e' poi fallita (0.33433 ->
    # 0.34973) e TUTTO e' stato annullato, anche la dose che le lettere avevano scelto. Acceso: ogni giro del ricentro si
    # tiene solo se, oltre a B, anche il funzionale obiettivo della decisione (`misura_obiettivo_dosi`, M1: gli stessi dati
    # della scelta, M2 resta intatto per la conferma) NON sale rispetto al valore raggiunto dalla dose; e la conferma M2 si
    # misura anche PRIMA del ricentro (solo registro: dice chi ha fatto salire le lettere, la decisione non la legge).
    ricentra_obiettivo: bool = False
    # 2/10 ramo tt5-fase-v (FASE_V.md). Esecuzione, non decisione: spenti (default) = di sempre al bit; accesi = STESSE
    # decisioni, meno lavoro.
    # V2 `pm_ordine_strati` (--pv-pm-ordine-strati): i gradienti g1 dei prezzi al punto medio (uno per primitiva, ognuno
    # in q0 + d_p, indipendente dagli altri) si calcolano in ordine di strato DECRESCENTE invece che di guadagno previsto:
    # col riuso (--pv-riuso) l'avanti riparte dallo strato piu' basso cambiato dall'ultima misura, cioe' min(strato della
    # primitiva di prima, strato di questa); in ordine decrescente quel minimo e' sempre lo strato di questa. Ogni g1 e
    # ogni prezzo restano gli stessi numeri (stesso stato dei byte, stesso lotto), salvati per indice come prima.
    pm_ordine_strati: bool = False
    # V3 `golosa_pigra` (--pv-golosa-pigra, solo con pm_golosa): nella prova golosa la primitiva si tiene solo se
    # (obiettivo < corrente) E (sonda A <= partenza). Se la sonda A e' gia' sopra la partenza la primitiva e' rifiutata
    # QUALUNQUE sia l'obiettivo: la sua misura (un avanti delle lettere su M1) si salta. Registro: «obiettivo» None,
    # «obiettivo_saltato» True per quella voce.
    golosa_pigra: bool = False

    def __post_init__(self):
        if self.golosa_pigra and not self.pm_golosa:
            raise ValueError("golosa_pigra solo con pm_golosa")
        f = [float(x) for x in self.frazioni]
        if not f or any(not 0.0 < x <= 1.0 for x in f) or any(b <= a for a, b in zip(f, f[1:])):
            raise ValueError(f"frazioni delle dosi {self.frazioni}: crescenti, in (0, 1]")
        if int(self.ricentra_giri) < 1:
            raise ValueError(f"ricentra_giri {self.ricentra_giri} < 1")
        if self.pm_primitiva not in PRIMITIVE_PM:
            raise ValueError(f"pm_primitiva {self.pm_primitiva!r}: una fra {PRIMITIVE_PM}")
        if int(self.pm_max) < 1:
            raise ValueError(f"pm_max {self.pm_max} < 1")
        if self.pm_golosa and not self.punto_medio:
            raise ValueError("pm_golosa solo con punto_medio (le primitive vengono dai prezzi al punto medio)")
        if self.ricentra_obiettivo and not (self.ricentra and self.pm_golosa):
            raise ValueError("ricentra_obiettivo solo con ricentra e pm_golosa (serve l'obiettivo della decisione, M1)")


def conteggi_dosi(n: int, frazioni) -> list:
    """Salti di ogni dose: max(1, ceil(f n)) (tetto n), crescenti, senza doppioni (con pochi salti le dosi piccole
    coincidono: una misura sola)."""
    out = []
    for f in frazioni:
        c = min(n, max(1, math.ceil(round(float(f) * n, 9))))
        if not out or c > out[-1]:
            out.append(c)
    return out


def cambi_instradamento(idx_a: torch.Tensor, idx_b: torch.Tensor) -> float:
    """Quota di token il cui top-k (come insieme) differisce fra due instradamenti dello stesso strato."""
    return float((torch.sort(idx_a, -1).values != torch.sort(idx_b, -1).values).any(-1).double().mean())


# ═══════════════════════════════════ 9. orchestratore ═══════════════════════════════════
class PassoV:
    """Buffer + modo per micro-lotto + selezione + salti + registro per un insieme di famiglie
    {(L, nome): DatiFamiglia}. Uso per passo:
        pv.inizia_passo(); for k, m in enumerate(micro): pv.microlotto(k); <avanti/indietro di m>
        opt.step()                                                       # passo P (scale)
        for m in curv: pv.curvatura(); <avanti, indietro di un vettore con cov. = Hessiana delle uscite>
        pv.fine_raccolta()
        reg, diag = pv.seleziona_e_salta(scale_correnti, rho)            # passo V
        tenuto = regione.esito(rho, kl_prima, kl_dopo_sulla_sonda, len(reg)); if not tenuto: pv.annulla(reg)"""

    def __init__(self, famiglie: dict, cfg: ConfigV = ConfigV(), stato: Optional[StatoV] = None,
                 bop: Optional[ConfigBop] = None):
        self.fam = famiglie
        self.cfg = cfg
        if stato is None:
            self.stato = StatoV(cfg.n_proposta, cfg.passi_accumulo, bop=bop)
            self.stato.prepara(famiglie.values())
        else:
            if bop is not None and stato.bop is None:
                raise ValueError("PassoV V-Bop con uno StatoV senza V-Bop")
            self.stato = stato
        self.registri: list = []
        self._bop_prima: Optional[tuple] = None      # (registro, {chiave: (e, r, g, valori)}) per l'annullamento
        self.tempi: Optional[dict] = None             # 2/10 --pv-tempi-v: secondi per sotto-fase (None = niente orologio)

    def inizia_passo(self) -> None:
        self.stato.inizia_passo()

    def microlotto(self, k: int) -> int:
        return self.stato.microlotto(k)

    def curvatura(self) -> int:
        return self.stato.curvatura()

    def fine_raccolta(self) -> None:
        self.stato.fine_raccolta()

    def registra_scale_finestra(self, scale: dict) -> None:
        """Scale all'INIZIO della finestra di accumulo (prima del passo P del suo primo passo): riferimento della
        staleness. Con finestra di 1 non serve (nessun filtro: comportamento di prima al bit)."""
        if self.stato.passi_accumulo > 1:
            self.stato.d_finestra = {k: scale[k].detach().to(torch.float16).clone() for k in self.fam}

    def famiglie_attive(self) -> dict:
        """Le famiglie su cui il passo V decide (27/9 dominio-strati: solo quelle degli strati allenati, `StatoV.attive`;
        None = tutte, come sempre). Annullamento e rigioco restano su TUTTE (`self.fam`: registri di prima)."""
        a = getattr(self.stato, "attive", None)
        return self.fam if a is None else {k: f for k, f in self.fam.items() if k in a}

    def seleziona(self, scale: dict, rho: float) -> dict:
        fam = self.famiglie_attive()
        if self.stato.bop is not None:
            return {k: seleziona_bop(self.stato.bop_[k], self.stato.buffer(f), f, scale[k], self.stato.n_verifica,
                                     rho, self.cfg, self.stato.bop, n_perdita=self.stato.n_perdita)
                    for k, f in fam.items()}
        return {k: seleziona(self.stato.buffer(f), f, scale[k], self.stato.n_verifica, rho, self.cfg,
                             n_perdita=self.stato.n_perdita, n_curvatura=self.stato.n_curvatura,
                             d_rif=self.stato.d_finestra.get(k) if self.stato.passi_accumulo > 1 else None)
                for k, f in fam.items()}

    def seleziona_e_salta(self, scale: dict, rho: float) -> tuple:
        """Tutto o niente: se i salti di una famiglia falliscono, quelli gia' scritti nelle altre si annullano.
        V-Bop: prima l'aggiornamento dello stato persistente (EMA), dopo i salti la correzione alla ECO."""
        if self.stato.bop is not None:
            self.stato.aggiorna_bop()
        sel = self.seleziona(scale, rho)
        regs = []
        try:
            for k, s in sel.items():
                regs.append(applica_salti(self.fam[k], s))
        except Exception:
            annulla(self.fam, Registro.unisci(regs))
            raise
        reg = Registro.unisci(regs)
        self.registri.append(reg)
        diag = {f"{k[0]}.{k[1]}": {"testati": int(s.testati.sum()), "scoperte": int(s.scoperte.sum()),
                                   "convenienti": int(s.convenienti.sum()), "salti": len(s),
                                   "guadagno_previsto": float(s.guadagno.sum())} for k, s in sel.items()}
        if self.stato.bop is not None:
            self._bop_prima = (reg, {k: self._dopo_salto_bop(k, s, scale[k], r)
                                     for (k, s), r in zip(sel.items(), regs)})
            for k in sel:
                z = self.stato.bop_[k]
                diag[f"{k[0]}.{k[1]}"]["con_candidato"] = int(((z.eta & ETA_MAX) >= 1).sum())
        return reg, diag

    def _dopo_salto_bop(self, k, sel: Selezione, d_eff: torch.Tensor, reg: Registro) -> tuple:
        """Correzione alla ECO sui gruppi saltati: m^ <- m^ - H |d| sign(m^) (il gradiente dopo il salto, nel modello
        quadratico); se il verso non e' piu' fattibile dal trit nuovo (bordo di {-1,0,1}) lo stato si azzera. Ritorna i
        valori di prima (annullamento al bit)."""
        z = self.stato.bop_[k]
        e, r, g = sel.e, sel.r, sel.g
        prima = {n: getattr(z, n)[e, r, g].clone() for n in ("m", "r", "H", "eta")}
        if len(sel) == 0:
            return e, r, g, prima
        eta, _ = z.eta_verso()
        eta = eta[e, r, g]
        c, _ = _correzione_eta(float(self.stato.bop.gamma), eta)
        sm = z.sm[e].to(torch.float64)
        m_hat = z.m[e, r, g].to(torch.float64) * sm / c
        Hh = z.H[e, r, g].to(torch.float64) * z.sH[e].to(torch.float64)
        m_hat = m_hat - Hh * d_eff[e, r, g].to(torch.float64).abs() * torch.sign(m_hat)
        z.m[e, r, g] = (m_hat * c / sm).clamp(-65504.0, 65504.0).to(torch.float16)
        tn = torch.from_numpy(reg.righe["t_nuovo"].astype(np.int64)).to(sel.dir.device)
        fuori = (tn + sel.dir.to(torch.int64)).abs() > 1
        if bool(fuori.any()):
            ef, rf, gf = e[fuori], r[fuori], g[fuori]
            for n in ("m", "r", "H"):
                getattr(z, n)[ef, rf, gf] = 0
            z.eta[ef, rf, gf] = 0
        return e, r, g, prima

    # ─── 27/9 ramo v-interazione-b: sonda a dosi con conferma (--pv-sonda-dosi) e V ricentrato (--pv-ricentra) ───
    def _curvatura_salti(self, k, s: Selezione) -> torch.Tensor:
        """Curvatura Gauss-Newton PER MICRO-LOTTO sul codice di ogni salto scelto (float64): PV = H del passo / n_curv
        (la curvatura e' misurata sul candidato, che e' la colonna del salto); V-Bop = la sua EMA (gia' per micro-lotto).
        Letta PRIMA della correzione alla ECO (che puo' azzerare lo stato)."""
        if len(s) == 0:
            return torch.zeros(0, dtype=torch.float64, device=s.e.device)
        if self.stato.bop is not None:
            z = self.stato.bop_[k]
            return z.H[s.e, s.r, s.g].to(torch.float64) * z.sH[s.e].to(torch.float64)
        nc = self.stato.n_curvatura
        if nc < 1:
            raise ValueError("ricentro senza curvatura del passo (servono micro-lotti di curvatura, modo 3)")
        return self.stato.buffer(self.fam[k]).H[s.e, s.r, s.g].to(torch.float64) / nc

    @staticmethod
    def _sotto(s: Selezione, li: torch.Tensor) -> Selezione:
        """Le righe `li` (indici locali) di una Selezione; i conteggi per esperto restano quelli della selezione."""
        li = li.to(s.e.device)
        return Selezione(s.e[li], s.r[li], s.g[li], s.cand[li], s.dir[li], s.guadagno[li], s.testati, s.scoperte,
                         s.convenienti, s.budget, s.scelti, s.costo)

    def seleziona_e_prova_dosi(self, scale: dict, rho: float, misura_a: Callable, misura_b: Callable,
                               cfg: "ConfigDosi", gradiente_ricentro: Optional[Callable] = None,
                               gradienti_pm: Optional[dict] = None, fattori_pm: Optional[dict] = None,
                               misura_obiettivo: Optional[Callable] = None,
                               misura_obiettivo_dosi: Optional[Callable] = None) -> tuple:
        """Passo V con SONDA A DOSI e CONFERMA (TECNICHE_E2E_PV.md §32.5 azione 1) e, con `cfg.ricentra`, V RICENTRATO
        (azione 2). Al posto di `seleziona_e_salta` + sonda tutto-o-niente.
          1. (V-Bop) EMA dello stato; selezione come sempre (stessi criteri, budget rho, ordinamento per esperto).
          2. Tutti i salti scelti in ordine GLOBALE di guadagno previsto (decrescente, stabile); dosi = prefissi annidati
             di max(1, ceil(f n)) salti per f in `cfg.frazioni` (duplicati tolti). `misura_a()` prima di tutto e dopo
             ogni dose (i salti si aggiungono per incrementi di registro: nessun salto riapplicato).
          3. Dose migliore = argmin della sonda A (a parita' la piu' piccola). Se non scende sotto la sonda A di
             partenza, o se sulla sonda B (`misura_b`, micro-lotto disgiunto: niente maledizione del vincitore della
             scelta fra 5 dosi) non e' <= a B di partenza: TUTTO annullato al bit, come oggi.
          4. Ricentro (se `cfg.ricentra` e `gradiente_ricentro`): nello stato raggiunto q_S, `gradiente_ricentro()`
             esegue UN indietro della sonda A con lo StatoV in `prepara_ricentro` (ritorna False se il gradiente non e'
             valido: il ricentro si salta); per ogni salto tenuto il lascia-uno-fuori (§32.4 (ii))
                 f(S) - f(S \\ i) ~ dw_i g1_i - 1/2 H_ii dw_i^2,   dw_i = (t_nuovo - t_vecchio) d_i,
             g1 = S del buffer (dL/dW_rot sul codice, unita' di un micro-lotto della perdita del passo), H_ii = curvatura
             per micro-lotto (`_curvatura_salti`). > 0 = il salto, dati gli altri, fa danno. Dannosi in ordine di danno;
             si annulla la META' piu' dannosa, poi la sonda B: se non sale si tiene e si passa alla meta' del resto,
             altrimenti si rimette e ci si ferma (Jacobi smorzato: gli annullamenti simultanei interagiscono anch'essi);
             al massimo `cfg.ricentra_giri` giri.
          5. Registro dei salti TENUTI (ordine di guadagno) in `self.registri`; V-Bop: correzione alla ECO SOLO sui tenuti
             (un salto provato e annullato lascia lo stato come dopo l'EMA, come l'annullamento di oggi).
        29/9 `cfg.punto_medio` (ramo punto-medio): fra 1 e 2 l'ordine e le dosi vengono dai prezzi al punto medio delle
        primitive (`_prezzi_punto_medio`, `scegli_primitive`); `gradienti_pm` = {"B": indietro della sonda generica, e
        facoltativo "A": indietro del funzionale obiettivo (KL delle 4 lettere)}, ognuno come `gradiente_ricentro` (lo
        StatoV e' in `prepara_ricentro`; False = gradiente non valido); `fattori_pm` = {nome: unita' del gradiente -> per
        token} (solo registro); `misura_obiettivo` (facoltativa, dati DISGIUNTI da quelli dei prezzi): alla fine il
        funzionale obiettivo dev'essere sceso, o tutto si annulla. Spento: nessuna di queste strade, identico al bit.
        29/9 `cfg.pm_golosa` (ramo pv-lettere-dosi): il passo 2-3 diventa `_dosi_golose` (primitiva per primitiva,
        decisa dall'obiettivo `misura_obiettivo_dosi` col vincolo della sonda A misurato).
        Ritorna (registro tenuto o None, diag come `seleziona_e_salta`, esito)."""
        if cfg.punto_medio and not (gradienti_pm and "B" in gradienti_pm):
            raise ValueError("punto medio senza il gradiente della sonda generica (gradienti_pm['B'])")
        tempi = self.tempi
        if self.stato.bop is not None:
            self._bop_prima = None
            with cronometro(tempi, "aggiorna_bop"):
                self.stato.aggiorna_bop()
        with cronometro(tempi, "seleziona"):
            sel = self.seleziona(scale, rho)
            diag = {f"{k[0]}.{k[1]}": {"testati": int(s.testati.sum()), "scoperte": int(s.scoperte.sum()),
                                       "convenienti": int(s.convenienti.sum()), "salti": len(s),
                                       "guadagno_previsto": float(s.guadagno.sum())} for k, s in sel.items()}
        chiavi = list(sel.keys())
        righe_f, gua, Hs, dws, fid, loc = [], [], [], [], [], []
        _cr = cronometro(tempi, "prepara_salti")
        _cr.__enter__()
        for i, k in enumerate(chiavi):
            s = sel[k]
            righe_f.append(righe_salti(self.fam[k], s))
            gua.append(s.guadagno.detach().to("cpu", torch.float64))
            Hs.append(self._curvatura_salti(k, s).to("cpu"))
            d = scale[k][s.e, s.r, s.g].to(torch.float64) if len(s) else torch.zeros(0, dtype=torch.float64)
            dws.append((s.dir.to(torch.float64).to(d.device) * d).to("cpu"))
            fid.append(torch.full((len(s),), i, dtype=torch.int64))
            loc.append(torch.arange(len(s), dtype=torch.int64))
        righe = np.concatenate(righe_f) if righe_f else np.zeros(0, dtype=REG_DTYPE)
        gua_t = torch.cat(gua) if gua else torch.zeros(0, dtype=torch.float64)
        H_t = torch.cat(Hs) if Hs else torch.zeros(0, dtype=torch.float64)
        dw_t = torch.cat(dws) if dws else torch.zeros(0, dtype=torch.float64)
        fid_t = torch.cat(fid) if fid else torch.zeros(0, dtype=torch.int64)
        loc_t = torch.cat(loc) if loc else torch.zeros(0, dtype=torch.int64)
        _cr.__exit__(None, None, None)
        n = int(righe.shape[0])
        esito = {"salti": n, "sonda_prima_a": misura_a(), "sonda_prima_b": None, "dosi": [], "dose": None,
                 "sonda_dopo_a": None, "sonda_b_dose": None, "sonda_dopo_b": None, "confermata": None,
                 "tenuto": False, "salti_tenuti": 0, "parziale": False, "guadagno_previsto": float(gua_t.sum()),
                 "guadagno_previsto_tenuti": 0.0, "ricentro": None}
        if n == 0:                                       # come oggi: registro vuoto tenuto, nessuna misura in piu'
            reg = Registro()
            self.registri.append(reg)
            if self.stato.bop is not None:
                self._bop_prima = (reg, {})
            esito.update(tenuto=True, sonda_dopo_a=esito["sonda_prima_a"])
            self._diag_bop(diag)
            return reg, diag, esito
        obj0 = None
        if cfg.punto_medio:                              # 29/9 ramo punto-medio: ordine e dosi dai prezzi al punto medio
            ordine, conti, esito["punto_medio"] = self._prezzi_punto_medio(
                chiavi, sel, righe, fid_t, loc_t, dw_t, gua_t, cfg, gradienti_pm, fattori_pm or {})
            if ordine.numel() == 0:                      # nessuna primitiva conveniente: nessun salto, come n == 0
                reg = Registro()
                self.registri.append(reg)
                if self.stato.bop is not None:
                    self._bop_prima = (reg, {})
                esito.update(tenuto=True, sonda_dopo_a=esito["sonda_prima_a"])
                self._diag_bop(diag)
                return reg, diag, esito
            if misura_obiettivo is not None:
                obj0 = misura_obiettivo()
        else:
            ordine = torch.argsort(gua_t, descending=True, stable=True)
            conti = conteggi_dosi(n, cfg.frazioni)
        righe_o = righe[ordine.numpy()]
        cum = torch.cumsum(gua_t[ordine], 0)
        esito["sonda_prima_b"] = misura_b()
        if cfg.punto_medio and cfg.pm_golosa:           # 29/9 ramo pv-lettere-dosi: prova golosa per primitiva
            tenuti = self._dosi_golose(ordine, conti, righe_o, cum, n, misura_a, misura_b, misura_obiettivo_dosi, esito,
                                       pigra=cfg.golosa_pigra)
            if tenuti is None:
                self._diag_bop(diag)
                return None, diag, esito
        else:
            incrementi, a = [], 0
            try:
                for c in conti:
                    inc = Registro(righe_o[a:c].copy())
                    rigioca(self.fam, [inc])
                    incrementi.append(inc)
                    a = c
                    esito["dosi"].append({"frazione": c / n, "salti": c, "sonda_a": misura_a(),
                                          "guadagno_previsto": float(cum[c - 1])})
            except BaseException:
                for inc in reversed(incrementi):
                    annulla(self.fam, inc)
                raise
            j = min(range(len(conti)), key=lambda i: esito["dosi"][i]["sonda_a"])
            esito["sonda_dopo_a"] = esito["dosi"][j]["sonda_a"]
            if esito["dosi"][j]["sonda_a"] < esito["sonda_prima_a"]:
                for inc in reversed(incrementi[j + 1:]):
                    annulla(self.fam, inc)
                esito["sonda_dopo_b"] = esito["sonda_b_dose"] = misura_b()
                esito["confermata"] = esito["sonda_dopo_b"] <= esito["sonda_prima_b"]
            else:
                esito["confermata"] = False
            esito["dose"] = {"frazione": conti[j] / n, "salti": conti[j], "sonda_a": esito["dosi"][j]["sonda_a"]}
            if not esito["confermata"]:
                for inc in reversed(incrementi[:j + 1] if esito["sonda_dopo_b"] is not None else incrementi):
                    annulla(self.fam, inc)
                self._diag_bop(diag)
                return None, diag, esito
            tenuti = ordine[:conti[j]].clone()
        kw_obj = {}
        if cfg.ricentra_obiettivo and cfg.ricentra and gradiente_ricentro is not None \
                and misura_obiettivo_dosi is not None and esito.get("golosa"):
            kw_obj = {"misura_obj": misura_obiettivo_dosi, "obiettivo": esito["golosa"]["obiettivo_dopo"]}
            if obj0 is not None:                         # solo registro: le lettere M2 alla dose, prima del ricentro
                esito["punto_medio"]["obiettivo_dose"] = misura_obiettivo()
        if cfg.ricentra and gradiente_ricentro is not None:
            tenuti, esito["ricentro"], esito["sonda_dopo_b"] = self._ricentra(
                chiavi, sel, righe, fid_t, loc_t, dw_t, H_t, tenuti, gradiente_ricentro, misura_b,
                esito["sonda_dopo_b"], int(cfg.ricentra_giri), **kw_obj)
        if tenuti.numel() == 0:                          # il ricentro ha tolto tutto: come un annullamento
            esito.update(tenuto=False, salti_tenuti=0)
            self._diag_bop(diag)
            return None, diag, esito
        if obj0 is not None:                             # 29/9 punto-medio: conferma dell'OBIETTIVO su dati disgiunti
            obj1 = misura_obiettivo()
            esito["punto_medio"]["obiettivo"] = {"prima": obj0, "dopo": obj1, "confermato": bool(obj1 < obj0)}
            if not obj1 < obj0:
                annulla(self.fam, Registro(righe[tenuti.numpy()].copy()))
                esito.update(tenuto=False, salti_tenuti=0, confermata=False)
                self._diag_bop(diag)
                return None, diag, esito
        reg = Registro(righe[tenuti.numpy()].copy())
        self.registri.append(reg)
        if self.stato.bop is not None:
            prima = {}
            for i, k in enumerate(chiavi):
                li = loc_t[tenuti][fid_t[tenuti] == i]
                sk = self._sotto(sel[k], li)
                prima[k] = self._dopo_salto_bop(k, sk, scale[k], Registro(righe_f[i][li.numpy()].copy()))
            self._bop_prima = (reg, prima)
            self._diag_bop(diag)
        esito.update(tenuto=True, salti_tenuti=int(tenuti.numel()), parziale=int(tenuti.numel()) < n,
                     guadagno_previsto_tenuti=float(gua_t[tenuti].sum()))
        return reg, diag, esito

    # ─── 29/9 ramo punto-medio (--pv-punto-medio): prezzi delle primitive al punto medio proprio ───
    def _gradiente_sui_salti(self, chiavi, sel, fid_t, loc_t, idx: torch.Tensor, gradiente: Callable):
        """dL/dW_rot sulla colonna di ciascun salto `idx` (indici globali nell'elenco dei salti scelti) nello stato ATTUALE
        dei byte: `prepara_ricentro` sui soli salti `idx` (cand := colonna del salto, modo 2), UN indietro (`gradiente`),
        lettura di S. None se il gradiente non e' valido (trabocco). float64 su CPU, nell'ordine di `idx`."""
        salti, pos = {}, {}
        for i, k in enumerate(chiavi):
            m = fid_t[idx] == i
            if not bool(m.any()):
                continue
            s = sel[k]
            li = loc_t[idx][m].to(s.e.device)
            salti[k] = (s.e[li], s.r[li], s.g[li], s.cand[li])
            pos[k] = torch.nonzero(m).reshape(-1)
        self.stato.prepara_ricentro(salti)
        try:
            valido = gradiente()
        finally:
            self.stato.fine_raccolta()
        if valido is False:
            return None
        g = torch.zeros(idx.numel(), dtype=torch.float64)
        for k, (e, r, gg, _) in salti.items():
            g[pos[k]] = self.stato.buffer(self.fam[k]).S[e, r, gg].to("cpu", torch.float64)
        return g

    def _prezzi_punto_medio(self, chiavi, sel, righe, fid_t, loc_t, dw_t, gua_t, cfg: "ConfigDosi",
                            gradienti: dict, fattori: dict) -> tuple:
        """Passo 1b di `seleziona_e_prova_dosi` con `cfg.punto_medio` (arXiv 2609.09867 Prop. 2 e §6(a),
        matematica/punto_medio_pv.md). Ritorna (ordine dei salti tenuti per le dosi, conteggi cumulativi delle dosi in
        salti, info).
          1. Primitive: salti raggruppati per `cfg.pm_primitiva`; si prezzano le `cfg.pm_max` col guadagno previsto (del
             criterio di sempre) piu' alto, le altre si scartano per questo passo V.
          2. Per ogni funzionale (B = sonda generica; A = obiettivo, facoltativo): g0 = gradiente in q0 sulle colonne di
             TUTTI i salti prezzati (UN indietro: un salto per gruppo, colonne distinte); per ogni primitiva p: salti di p
             applicati (rigioco), g1 = gradiente in q0 + d_p sulle sue colonne (UN indietro), annullamento al bit.
          3. v_j = 1/2 dw_j (g0_j + g1_j) (`prezzo_punto_medio`): s_p = sum_{j in p} v_j = F(q0 + d_p) - F(q0) esatto a
             curvatura costante (compresa TUTTA l'interazione dentro la primitiva, che il criterio diagonale ignora);
             v_j e' il prezzo di j al centro della primitiva (Eq. 4: sbaglia di 1/2 x^T beta (1 - x) sui soli salti dentro).
          4. `cfg.pm_filtra`: dentro ogni primitiva restano i salti con v_obj < 0 (s_p ricalcolato su di essi, lettura al
             centro); poi `scegli_primitive`; dosi = prefissi di primitive (`conteggi_dosi` sul NUMERO di primitive)."""
        n = int(righe.shape[0])
        fid, loc = fid_t, loc_t
        e_all = torch.cat([sel[k].e.to("cpu", torch.int64) for k in chiavi]) if chiavi else torch.zeros(0, dtype=torch.int64)
        tipo = cfg.pm_primitiva
        if tipo == "famiglia":
            chiave_p = [(int(chiavi[int(f)][0]), chiavi[int(f)][1]) for f in fid.tolist()]
        elif tipo == "tipo":
            chiave_p = [chiavi[int(f)][1] for f in fid.tolist()]
        elif tipo == "strato":
            chiave_p = [int(chiavi[int(f)][0]) for f in fid.tolist()]
        else:
            chiave_p = [(int(chiavi[int(f)][0]), chiavi[int(f)][1], int(e)) for f, e in zip(fid.tolist(), e_all.tolist())]
        nomi_p, pid = {}, torch.empty(n, dtype=torch.int64)
        for j, c in enumerate(chiave_p):
            pid[j] = nomi_p.setdefault(c, len(nomi_p))
        chiavi_p = list(nomi_p)
        P = len(chiavi_p)
        g_prev = torch.zeros(P, dtype=torch.float64).index_add_(0, pid, gua_t.to(torch.float64))
        prezzate = torch.argsort(g_prev, descending=True, stable=True)[:int(cfg.pm_max)]
        info = {"primitiva": tipo, "primitive": P, "prezzate": int(prezzate.numel()), "indietro": 0, "funzionali":
                sorted(gradienti), "valido": True, "scartati_non_prezzati": 0}
        membri = {int(p): torch.nonzero(pid == p).reshape(-1) for p in prezzate.tolist()}
        tutti = torch.cat([membri[int(p)] for p in prezzate.tolist()])
        info["scartati_non_prezzati"] = n - int(tutti.numel())
        vuoto = (torch.zeros(0, dtype=torch.int64), [], info)
        v = {nome: torch.full((n,), math.nan, dtype=torch.float64) for nome in gradienti}
        corrente = {nome: torch.full((n,), math.nan, dtype=torch.float64) for nome in gradienti}
        for nome, grad in gradienti.items():
            g0 = self._gradiente_sui_salti(chiavi, sel, fid, loc, tutti, grad)
            info["indietro"] += 1
            if g0 is None:                               # gradiente in q0 non valido: nessun prezzo, nessun salto
                info["valido"] = False
                return vuoto
            g0_pieno = torch.zeros(n, dtype=torch.float64)
            g0_pieno[tutti] = g0
            corrente[nome][tutti] = dw_t[tutti] * g0
            giro_p = prezzate.tolist()
            if cfg.pm_ordine_strati:                     # 2/10 V2: strato (il piu' basso della primitiva) decrescente
                giro_p = sorted(giro_p, key=lambda p: -int(righe["L"][membri[int(p)].numpy()].min()))
            for p in giro_p:
                idx = membri[int(p)]
                reg_p = Registro(righe[idx.numpy()].copy())
                rigioca(self.fam, [reg_p])
                try:
                    g1 = self._gradiente_sui_salti(chiavi, sel, fid, loc, idx, grad)
                finally:
                    annulla(self.fam, reg_p)
                info["indietro"] += 1
                if g1 is not None:
                    v[nome][idx] = prezzo_punto_medio(dw_t[idx], g0_pieno[idx], g1)
        obj = "A" if "A" in gradienti else "B"
        vin = "B" if obj == "A" else None
        tenuti_p, s_obj, s_vin, righe_info = [], [], [], []
        for p in prezzate.tolist():
            idx = membri[int(p)]
            vo = v[obj][idx]
            dentro = idx[vo < 0] if cfg.pm_filtra else idx
            if not bool(torch.isfinite(vo).all()):
                dentro = idx[:0]                         # g1 non valido: primitiva non prezzata
            dentro = dentro[torch.argsort(v[obj][dentro], stable=True)]
            tenuti_p.append(dentro)
            so = float(v[obj][dentro].sum()) if dentro.numel() else math.nan
            sv = float(v[vin][dentro].sum()) if (vin and dentro.numel()) else math.nan
            s_obj.append(so)
            s_vin.append(sv)
            f = lambda nome: float(fattori.get(nome, 1.0))  # noqa: E731
            righe_info.append({"primitiva": str(chiavi_p[int(p)]), "salti": int(idx.numel()), "tenuti": int(dentro.numel()),
                               "guadagno_previsto": float(g_prev[int(p)]),
                               **{f"s_{nome}": float(v[nome][idx].sum()) * f(nome) for nome in gradienti},
                               **{f"corrente_{nome}": float(corrente[nome][idx].sum()) * f(nome) for nome in gradienti},
                               **{f"s_{nome}_filtrata": (float(v[nome][dentro].sum()) * f(nome) if dentro.numel()
                                                         else None) for nome in gradienti}})
        sc = scegli_primitive(torch.tensor(s_obj, dtype=torch.float64),
                              torch.tensor(s_vin, dtype=torch.float64) if vin else None)
        if cfg.pm_golosa:                                # 29/9 pv-lettere-dosi: il bilancio si misura, non si prevede
            info["scelte_previste"] = [righe_info[i]["primitiva"] for i in sc]
            sc = ordina_primitive(torch.tensor(s_obj, dtype=torch.float64),
                                  torch.tensor(s_vin, dtype=torch.float64) if vin else None)
        info["per_primitiva"] = righe_info
        info["scelte"] = [righe_info[i]["primitiva"] for i in sc]
        if not sc:
            return vuoto
        ordine = torch.cat([tenuti_p[i] for i in sc])
        taglio = np.cumsum([int(tenuti_p[i].numel()) for i in sc]).tolist()
        conti = taglio if cfg.pm_golosa else [taglio[c - 1] for c in conteggi_dosi(len(sc), cfg.frazioni)]
        info["salti_scelti"] = int(ordine.numel())
        for nome in gradienti:
            info[f"previsto_{nome}"] = float(v[nome][ordine].sum()) * float(fattori.get(nome, 1.0))
            info[f"previsto_dosi_{nome}"] = [float(v[nome][ordine[:c]].sum()) * float(fattori.get(nome, 1.0))
                                             for c in conti]
        return ordine, conti, info

    def _dosi_golose(self, ordine, conti, righe_o, cum, n: int, misura_a: Callable, misura_b: Callable,
                     misura_obj: Optional[Callable], esito: dict, pigra: bool = False) -> Optional[torch.Tensor]:
        """29/9 ramo pv-lettere-dosi (`ConfigDosi.pm_golosa`), al posto dei passi 2-3 di `seleziona_e_prova_dosi`.
        `conti` = confini cumulativi delle primitive in `ordine` (una primitiva per segmento). Per ogni primitiva: rigioco,
        misura della sonda A (vincolo) e del funzionale obiettivo della DECISIONE `misura_obj` (le 4 lettere su M1; None =
        la sonda A stessa, nessun vincolo separato); tenuta se obiettivo < obiettivo corrente e sonda A <= sonda A di
        partenza, altrimenti annullata al bit. Poi sonda B (disgiunta) <= B di partenza sui soli tenuti, o tutto annullato.
        Ritorna gli indici (in `ordine`) dei salti tenuti, o None (nessun salto: byte come in partenza)."""
        a0 = esito["sonda_prima_a"]
        separato = misura_obj is not None
        o0 = misura_obj() if separato else a0
        cur_o = o0
        cur_a = a0
        tenuti, applicati, pendente, a = [], [], None, 0
        try:
            for c in conti:
                inc = Registro(righe_o[a:c].copy())
                rigioca(self.fam, [inc])
                pendente = inc
                ma = misura_a()
                if separato and pigra and not bool(ma <= a0):     # 2/10 V3: rifiutata comunque, l'obiettivo non serve
                    mo, tiene = None, False
                else:
                    mo = misura_obj() if separato else ma
                    tiene = bool(mo < cur_o) and (not separato or bool(ma <= a0))
                esito["dosi"].append({"frazione": c / n, "salti": c - a, "sonda_a": ma,
                                      "obiettivo": mo if separato else None, "tenuta": tiene,
                                      "guadagno_previsto": float(cum[c - 1]) - (float(cum[a - 1]) if a else 0.0)})
                if separato and pigra and mo is None:
                    esito["dosi"][-1]["obiettivo_saltato"] = True
                if tiene:
                    applicati.append(inc)
                    tenuti.append(ordine[a:c])
                    cur_o, cur_a = mo, ma
                else:
                    annulla(self.fam, inc)
                pendente = None
                a = c
        except BaseException:
            if pendente is not None:
                annulla(self.fam, pendente)
            for inc in reversed(applicati):
                annulla(self.fam, inc)
            raise
        esito["golosa"] = {"obiettivo_prima": o0 if separato else None, "obiettivo_dopo": cur_o if separato else None,
                           "primitive": len(conti), "tenute": len(applicati)}
        esito["sonda_dopo_a"] = cur_a
        k = int(sum(int(t.numel()) for t in tenuti))
        esito["dose"] = {"frazione": k / n, "salti": k, "sonda_a": cur_a}
        if not applicati:
            esito["confermata"] = False
            return None
        esito["sonda_dopo_b"] = esito["sonda_b_dose"] = misura_b()
        esito["confermata"] = esito["sonda_dopo_b"] <= esito["sonda_prima_b"]
        if not esito["confermata"]:
            for inc in reversed(applicati):
                annulla(self.fam, inc)
            return None
        return torch.cat(tenuti).clone()

    def _diag_bop(self, diag: dict) -> None:
        """V-Bop: gruppi con candidato per famiglia nel diag (come `seleziona_e_salta`, dopo la correzione alla ECO)."""
        if self.stato.bop is None:
            return
        for k, z in self.stato.bop_.items():
            if f"{k[0]}.{k[1]}" in diag:
                diag[f"{k[0]}.{k[1]}"]["con_candidato"] = int(((z.eta & ETA_MAX) >= 1).sum())

    def _ricentra(self, chiavi, sel, righe, fid_t, loc_t, dw_t, H_t, tenuti, gradiente, misura_b, sonda_b: float,
                  giri: int, misura_obj: Optional[Callable] = None, obiettivo: Optional[float] = None) -> tuple:
        """Passo 4 di `seleziona_e_prova_dosi`. Ritorna (tenuti, info, sonda B finale). 29/9 tt3-cure: con `misura_obj`
        (ConfigDosi.ricentra_obiettivo) un giro si tiene solo se anche l'obiettivo non sale oltre `obiettivo`."""
        salti = {}
        for i, k in enumerate(chiavi):
            s = sel[k]
            li = loc_t[tenuti][fid_t[tenuti] == i].to(s.e.device)
            salti[k] = (s.e[li], s.r[li], s.g[li], s.cand[li])
        self.stato.prepara_ricentro(salti)
        try:
            valido = gradiente()
        finally:
            self.stato.fine_raccolta()
        info = {"valido": bool(valido is not False), "dannosi": 0, "giri": [], "annullati": 0,
                "tenuti": int(tenuti.numel())}
        if valido is False:
            return tenuti, info, sonda_b
        g1 = torch.zeros(tenuti.numel(), dtype=torch.float64)
        for i, k in enumerate(chiavi):
            m = fid_t[tenuti] == i
            if not bool(m.any()):
                continue
            e, r, g, _ = salti[k]
            g1[m] = self.stato.buffer(self.fam[k]).S[e, r, g].to("cpu", torch.float64)
        dw, H = dw_t[tenuti], H_t[tenuti]
        loo = dw * g1 - 0.5 * H * dw * dw
        info["loo_somma"] = float(loo.sum())
        info["loo_max"] = float(loo.max())
        dan = torch.nonzero(loo > 0).reshape(-1)
        dan = dan[torch.argsort(loo[dan], descending=True, stable=True)]
        info["dannosi"] = int(dan.numel())
        info["loo_dannosi"] = float(loo[dan].sum()) if dan.numel() else 0.0
        vivo = torch.ones(tenuti.numel(), dtype=torch.bool)
        rimasti = dan
        for _ in range(max(0, giri)):
            if rimasti.numel() == 0:
                break
            h = (int(rimasti.numel()) + 1) // 2
            tolti = rimasti[:h]
            reg_t = Registro(righe[tenuti[tolti].numpy()].copy())
            annulla(self.fam, reg_t)
            nuova = misura_b()
            accetta = nuova <= sonda_b
            giro = {"annullati": h, "loo_previsto": float(loo[tolti].sum()), "sonda_b_prima": sonda_b,
                    "sonda_b_dopo": nuova, "accettato": bool(accetta)}
            if misura_obj is not None:                   # 29/9 tt3-cure: il giro non deve far salire l'obiettivo
                o = misura_obj()
                giro.update(obiettivo_prima=obiettivo, obiettivo_dopo=o)
                accetta = accetta and o <= obiettivo
                giro["accettato"] = bool(accetta)
            info["giri"].append(giro)
            if not accetta:
                rigioca(self.fam, [reg_t])
                break
            vivo[tolti] = False
            sonda_b = nuova
            if misura_obj is not None:
                obiettivo = o
            rimasti = rimasti[h:]
        tenuti = tenuti[vivo]
        info["annullati"] = int((~vivo).sum())
        info["tenuti"] = int(tenuti.numel())
        return tenuti, info, sonda_b

    def annulla(self, reg: Registro) -> None:
        annulla(self.fam, reg)
        for i in range(len(self.registri) - 1, -1, -1):
            if self.registri[i] is reg:
                del self.registri[i]
                break
        if self._bop_prima is not None and self._bop_prima[0] is reg:
            for k, (e, r, g, prima) in self._bop_prima[1].items():
                z = self.stato.bop_[k]
                for n, v in prima.items():
                    getattr(z, n)[e, r, g] = v
            self._bop_prima = None


# ═══════════════════════════════════ 10. riga di comando ═══════════════════════════════════
def main(argv=None) -> int:
    """`esporta`: GGUF finale = GGUF con le scale del checkpoint (da `e2e_esporta`) + codici dei registri PV del
    checkpoint (`migliore_*/registri_pv.npz`), verificato su file e ricaricato strato per strato."""
    import argparse
    import json
    ap = argparse.ArgumentParser(description=main.__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("esporta")
    p.add_argument("--scalato", required=True, help="GGUF gia' esportato con le scale del checkpoint")
    p.add_argument("--dst", required=True)
    p.add_argument("--registri", required=True, help="registri_pv.npz del checkpoint (lista, atomico)")
    p.add_argument("--sha256", default=None)
    p.add_argument("--strati-ricarica", default="0,3,39")
    a = ap.parse_args(argv)
    regs = Registro.carica_lista(Path(a.registri), a.sha256)
    rep = esporta_codici(Path(a.scalato), Path(a.dst), regs)
    rep["ricarica"] = ricarica_codici(Path(a.scalato), Path(a.dst), regs,
                                      [int(x) for x in a.strati_ricarica.split(",") if x != ""])
    print(json.dumps(rep, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
