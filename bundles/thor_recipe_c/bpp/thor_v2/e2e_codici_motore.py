#!/usr/bin/env python3
"""Fase 2 E2E, punti 2+3 del compito del 26/9 sera: motore VERO (non un giocattolo a parte) che rende i
CODICI ternari differenziabili DENTRO il grafo autograd che lo studente usa davvero (Piano/instradamento,
Ricetta, ruota/de_ruota), cosi' da poter essere agganciato allo `Studente` reale con un monkeypatch mirato
(vedi `e2e_allena_codici.py`), non un secondo studente parallelo.

26/9/2026 sera. Continua `e2e_codici_ste.py` (T1-T5, il PROGETTO: `e2e_progetto_fase2_codici.md`, gia'
committato). Questo file e' NUOVO: **NON tocca `e2e_studente.py` ne' `e2e_allena.py`**, li importa in SOLA
LETTURA (`e2e_studente as S`, `e2e_allena as AL`) e riusa `CodiceTernarioSTE`/`latente_da_trit` da
`e2e_codici_ste` (sola lettura: una sola STE, non una seconda). Conseguenza diretta e VERIFICATA (T1): la
fase 1 resta byte per byte quella di oggi, perche' nessuna riga di `e2e_studente.py`/`e2e_allena.py` cambia
-- l'"interruttore spento" non e' un flag da controllare a runtime, e' l'assenza stessa di ogni modifica.

Struttura (stessa forma O(E) di `S.MotoreTorch`/`S.FamigliaEsperti`: mai un tensore (E,R,C) di PRODOTTO
intermedio, solo il parametro `w` stesso, che e' il costo dichiarato in
`e2e_progetto_fase2_codici.md` §1.1):

  `_peso_esperto_codici`  peso (R,C) dell'esperto e, dal LATENTE w[e] (STE) invece che dai byte fissi
  `_avanti/_indietro_x`   stessa matematica di S.MotoreTorch, un esperto alla volta
  `_dd`                   dL/dtheta (scala): come oggi, ma il trit e' quello CORRENTE di w (non i byte fissi)
  `_dw`                   dL/dw, PER-ELEMENTO (non per-gruppo come dd): NUOVO, non esiste nella fase 1
  `FamigliaEspertiConCodici`  torch.autograd.Function, differenziabile in x, d E w (S.FamigliaEsperti lo e'
                          solo in x e d) -- stessa API di posto, un parametro `w` e un `clip` in piu'
  `w_da_famiglia`         latente iniziale = trit CORRENTE, vettorizzato (nessun ciclo Python su E)
  `cancello_g1_codici`    cancello sui gradienti dw azzerati dal cast fp16 (come G1 per dtheta, S6 di
                          `cancello_g1` in e2e_allena.py), qui su dati SINTETICI (bersaglio: un'altra
                          esperta vera) perche' non c'e' un teacher/corpus vero da CPU -- la versione che
                          lavora sul gradiente VERO del teacher e' `cancello_g1_codici_vivo` in
                          `e2e_allena_codici.py` (riusa QUESTA stessa `_dw`, non un secondo calcolo)

Test (`_test()`), TUTTI su CPU, sui BYTE VERI di Thor-T, strati 0 e 3 (come fa `e2e_test_studente.py`):
  T1  nessuna regressione: FamigliaEspertiConCodici con w = trit CORRENTE (nessun passo fatto) produce
      uscita, dx e dd IDENTICI AL BIT a S.FamigliaEsperti (byte fissi) -- prova che il nuovo motore e' un
      SOVRAINSIEME del vecchio nel punto operativo, non una reimplementazione divergente.
  T2  dw contro un riferimento INDIPENDENTE (autograd nativo di torch attraverso la STESSA STE, ricostruita
      a mano fuori da FamigliaEspertiConCodici -- stesso stile di `riferimento_famiglia` in
      `e2e_test_studente.py` S3), ricette B64 (ruotata) e A64 (de-ruotata): copre ENTRAMBI i rami
      dell'aggiustamento per rotazione in `_dw` (il down non ruota mai, il gate/up si', a seconda della
      ricetta).
  T3  ciclo minimo instradato (Piano vero, non un caso degenere a un esperto): SGD riduce la loss verso il
      bersaglio (un'altra esperta vera), Adam con le beta di theta no (conferma T3 di `e2e_codici_ste.py`
      ora sul motore VERO, non sul giocattolo isolato).
  T4  cancello G1 per i codici: quota di dL/dw che sparisce per il cast fp16 (come fa davvero
      FamigliaEspertiConCodici.backward con la ricetta di produzione B), convergenza di K con
      `AL.regola_k` (RIUSATA, sola lettura) fino a 3 giri come `cancello_g1`.

Uso (CPU confinata):
    export XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
    systemd-run --user --scope -p CPUQuota=800% -p MemoryMax=10G nice -n 10 \
      env OMP_NUM_THREADS=4 python3 e2e_codici_motore.py --test
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
sys.path.insert(0, "@@ROOT:models@@/build/llamacpp-tq1-k1024/gguf-py")
sys.path.insert(0, "@@ROOT:forge@@/kernel-rotazione")

from gguf import GGUFReader  # noqa: E402
import e2e_studente as S  # noqa: E402  (sola lettura)
import e2e_allena as AL  # noqa: E402  (sola lettura: regola_k)
import e2e_campi_scala as CAMPI  # noqa: E402  (sola lettura: trit_torch)
import e2e_codici_scale as CS  # noqa: E402  (sola lettura: LAYOUT)
from e2e_codici_ste import CodiceTernarioSTE  # noqa: E402  (sola lettura: LA STE, una sola fonte)

GGUF_THOR_T = S.GGUF_THOR_T


# ═══════════════════════════ 1. peso dal latente, O(E) come S.MotoreTorch ═══════════════════════════
def _peso_esperto_codici(fam: "S.DatiFamiglia", e: int, w: torch.Tensor, d: torch.Tensor,
                         ric: "S.Ricetta", clip: float) -> torch.Tensor:
    """Come `S.peso_esperto_torch`, ma il trit dell'esperto e viene dal LATENTE w[e] (STE) invece che dai
    byte fissi (`S.trit_esperto`). w: (E, R, C), foglia allenabile in dtype qualunque (il cast a `calc` e'
    voluto: e' lo STESSO cast di `S.peso_esperto_torch`, cosi' T1 confronta al bit nella STESSA precisione)."""
    calc = torch.float64 if ric.dtype_gemm == torch.float64 else torch.float32
    trit = CodiceTernarioSTE.apply(w[e].to(calc), clip)
    W = (trit.view(fam.R, fam.NG, fam.gs) * d[e].to(calc).unsqueeze(-1)).view(fam.R, fam.C)
    if fam.ruotata and not ric.ruota:
        W = S.de_ruota(W)
    return W.to(ric.dtype_gemm)


def _avanti(x: torch.Tensor, d: torch.Tensor, w: torch.Tensor, fam: "S.DatiFamiglia", piano: "S.Piano",
           ric: "S.Ricetta", clip: float) -> torch.Tensor:
    out = torch.empty(x.shape[0], fam.R, dtype=ric.dtype_uscita, device=x.device)
    o = piano.offs.tolist()
    for e in range(fam.E):
        if o[e + 1] > o[e]:
            W = _peso_esperto_codici(fam, e, w, d, ric, clip)
            xs = x[o[e]:o[e + 1]]
            if ric.dtype_gemm == torch.float16:
                out[o[e]:o[e + 1]] = torch.matmul(xs.float(), W.float().T)
            else:
                out[o[e]:o[e + 1]] = torch.matmul(xs, W.T).to(out.dtype)
    return out


def _indietro_x(gy: torch.Tensor, d: torch.Tensor, w: torch.Tensor, fam: "S.DatiFamiglia", piano: "S.Piano",
                ric: "S.Ricetta", clip: float) -> torch.Tensor:
    dx = torch.empty(gy.shape[0], fam.C, dtype=gy.dtype, device=gy.device)
    o = piano.offs.tolist()
    for e in range(fam.E):
        if o[e + 1] > o[e]:
            W = _peso_esperto_codici(fam, e, w, d, ric, clip)
            if ric.dtype_gemm == torch.float16:
                dx[o[e]:o[e + 1]] = torch.matmul(gy[o[e]:o[e + 1]].float(), W.float()).to(dx.dtype)
            else:
                dx[o[e]:o[e + 1]] = torch.matmul(gy[o[e]:o[e + 1]], W)
    return dx


def _dd(gy: torch.Tensor, xr: torch.Tensor, w: torch.Tensor, fam: "S.DatiFamiglia", piano: "S.Piano",
       d_dtype: torch.dtype) -> torch.Tensor:
    """dL/dtheta (scala): STESSA formula di `S.MotoreTorch.dd`, ma il trit e' quello CORRENTE del latente
    (round-clamp(w), non i byte fissi -- puo' essere cambiato dai passi precedenti su w). `xr` e' gia' nel
    dominio giusto (ruotato o no): la stessa convenzione di `S.FamigliaEsperti.backward`, il chiamante la
    prepara UNA volta per dd e dw insieme."""
    acc = torch.float64 if gy.dtype == torch.float64 else torch.float32
    dd = torch.zeros(fam.E, fam.R, fam.NG, dtype=d_dtype, device=gy.device)
    o = piano.offs.tolist()
    for e in range(fam.E):
        if o[e + 1] > o[e]:
            dW = torch.matmul(gy[o[e]:o[e + 1]].to(acc).T, xr[o[e]:o[e + 1]].to(acc))
            with torch.no_grad():
                trit_corrente = torch.clamp(torch.round(w[e]), -1.0, 1.0).to(acc)
            dd[e] = (trit_corrente * dW).view(fam.R, fam.NG, fam.gs).sum(-1).to(d_dtype)
    return dd


def _dw(gy: torch.Tensor, xr: torch.Tensor, d: torch.Tensor, fam: "S.DatiFamiglia", piano: "S.Piano",
       w: torch.Tensor, clip: float) -> torch.Tensor:
    """dL/dw, PER-ELEMENTO (non per-gruppo come dd): dW_e = gy_e^T @ xr_e (R,C), gia' nel dominio del
    latente (xr pre-ruotata dal chiamante, STESSA convenzione di dd -- vedi `e2e_progetto_fase2_codici.md`
    §5 per la verifica che rotazione e contrazione sul token commutano: e' lo stesso motivo per cui dd
    ruota x invece di ruotare dW, qui si riusa senza riscriverlo); dL/dw_e = scala_e (broadcast sul gruppo)
    * dW_e * passa_STE(w_e) -- lo STE si applica A MANO (identita' dentro [-clip,clip], zero fuori, come
    `CodiceTernarioSTE.backward`) perche' il motore CPU non richiama autograd dentro il ciclo per-esperto
    (stessa scelta O(E) di `S.MotoreTorch.dd`, che gia' non richiama autograd per calcolare dd)."""
    acc = torch.float64 if gy.dtype == torch.float64 else torch.float32
    dw = torch.zeros_like(w, dtype=acc)
    o = piano.offs.tolist()
    for e in range(fam.E):
        if o[e + 1] > o[e]:
            dW = torch.matmul(gy[o[e]:o[e + 1]].to(acc).T, xr[o[e]:o[e + 1]].to(acc))       # (R, C)
            scala_bc = d[e].to(acc).unsqueeze(-1).expand(fam.R, fam.NG, fam.gs).reshape(fam.R, fam.C)
            passa = (w[e].to(acc).abs() <= clip).to(acc)
            dw[e] = scala_bc * dW * passa
    return dw.to(w.dtype)


# ═══════════════════════════ 2. autograd Function ═══════════════════════════
class FamigliaEspertiConCodici(torch.autograd.Function):
    """Come `S.FamigliaEsperti`, ma differenziabile ANCHE in w (i codici, via STE): un parametro (`w`) e un
    iperparametro (`clip`) in piu' nella firma, nessun altro cambiamento di contratto. Motore "torch" SOLO
    (CPU e GPU: `torch.matmul` e' portabile, non serve Triton -- la versione Triton fusa resta un passo
    successivo dichiarato in `e2e_progetto_fase2_codici.md` §7, DOPO che questo motore e' confermato
    corretto, stesso ordine gia' seguito dalla fase 1)."""

    @staticmethod
    def forward(ctx, x, d, w, fam: "S.DatiFamiglia", piano: "S.Piano", ric: "S.Ricetta", clip: float):
        y = _avanti(x, d, w, fam, piano, ric, clip)
        ctx.save_for_backward(x, d, w)
        ctx.fam, ctx.piano, ctx.ric, ctx.clip = fam, piano, ric, clip
        return y

    @staticmethod
    def backward(ctx, gy):
        x, d, w = ctx.saved_tensors
        fam, piano, ric, clip = ctx.fam, ctx.piano, ctx.ric, ctx.clip
        di = ric.dtype_indietro or x.dtype          # STESSA scelta di S.FamigliaEsperti.backward
        gy = gy.to(di).contiguous()
        dx = None
        if ctx.needs_input_grad[0]:
            dx = _indietro_x(gy, d, w, fam, piano, ric, clip).to(x.dtype)
        dd = dw = None
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            xr = S.ruota(x.to(torch.float64 if x.dtype == torch.float64 else torch.float32)) \
                if (fam.ruotata and not ric.ruota) else x
            xr = xr.to(di).contiguous()
            if ctx.needs_input_grad[1]:
                dd = _dd(gy, xr, w, fam, piano, d.dtype)
            if ctx.needs_input_grad[2]:
                dw = _dw(gy, xr, d, fam, piano, w, clip)
        return dx, dd, dw, None, None, None, None


def w_da_famiglia(fam: "S.DatiFamiglia", dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Latente iniziale = trit CORRENTE di TUTTI gli esperti della famiglia, VETTORIZZATO: nessun ciclo
    Python su E (fam.blocchi.reshape(-1, nb) e' gia' flat su (E*R*blocchi_riga, nb), CAMPI.trit_torch
    lavora su blocchi flat qualunque sia N -- lo stesso decoder che usa S.trit_esperto, ma senza il ciclo
    per-esperto che quella funzione fa apposta per le prove piccole)."""
    b = fam.blocchi.reshape(-1, fam.nb)
    trit = CAMPI.trit_torch(b, fam.tipo).to(dtype)
    return trit.reshape(fam.E, fam.R, fam.C)


# ═══════════════════════════ 3. dati veri + riferimento indipendente ═══════════════════════════
def carica_famiglia(nome_gguf_tensore: str, nome: str, esperti: slice, gguf: Path = GGUF_THOR_T) -> "S.DatiFamiglia":
    r = GGUFReader(str(gguf))
    t = {x.name: x for x in r.tensors}[nome_gguf_tensore]
    return S.famiglia_da_tensore(t, nome, esperti=esperti)


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double(), b.double()
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-300))


def riferimento_famiglia_codici(x: torch.Tensor, d: torch.Tensor, w0: torch.Tensor, fam: "S.DatiFamiglia",
                                piano: "S.Piano", ruota_peso: bool, clip: float, G: torch.Tensor) -> tuple:
    """Riferimento INDIPENDENTE per dL/dw (stesso stile di `riferimento_famiglia` in
    `e2e_test_studente.py` S3, che e' il riferimento gia' fidato per dd/dx): la STESSA STE
    (`CodiceTernarioSTE`, gia' verificata in `e2e_codici_ste.py` T2) composta A MANO con l'algebra
    dell'esperto e l'autograd NATIVO di torch, non le funzioni interne (`_dd`/`_dw`) di questo modulo --
    se avessero lo stesso bug, qui non lo vedrebbero, essendo un percorso di calcolo diverso."""
    x = x.detach().clone().requires_grad_(True)
    w = w0.detach().clone().requires_grad_(True)
    o = piano.offs.tolist()
    ys = []
    for e in range(fam.E):
        trit = CodiceTernarioSTE.apply(w[e], clip)
        W = (trit.view(fam.R, fam.NG, fam.gs) * d[e].unsqueeze(-1)).view(fam.R, fam.C)
        if ruota_peso:
            W = S.de_ruota(W)
        ys.append(x[o[e]:o[e + 1]] @ W.T)
    L = (torch.cat(ys) * G).sum()
    L.backward()
    return float(L.detach()), x.grad, w.grad


# ═══════════════════════════ T1 — nessuna regressione ═══════════════════════════
def t1_nessuna_regressione(fam: "S.DatiFamiglia", ricn: str = "B64", seed: int = 101) -> dict:
    torch.manual_seed(seed)
    ric = S.RICETTE[ricn]
    idx = torch.randint(0, fam.E, (48, 3))
    piano = S.Piano(idx, fam.E)
    x = torch.randn(piano.S, fam.C, dtype=torch.float64)
    d = fam.d0.double()
    w0 = w_da_famiglia(fam, torch.float64)

    x_rif, d_rif = x.clone().requires_grad_(True), d.clone().requires_grad_(True)
    y_rif = S.FamigliaEsperti.apply(x_rif, d_rif, fam, piano, ric, S.motore("torch"))
    G = torch.randn_like(y_rif)
    (y_rif * G).sum().backward()

    x_new, d_new, w_new = x.clone().requires_grad_(True), d.clone().requires_grad_(True), w0.clone().requires_grad_(True)
    y_new = FamigliaEspertiConCodici.apply(x_new, d_new, w_new, fam, piano, ric, 1.5)
    (y_new * G).sum().backward()

    return {"ricetta": ricn, "N": int(w0.numel()),
            "avanti_identico_al_bit": bool(torch.equal(y_rif, y_new)),
            "dx_identico_al_bit": bool(torch.equal(x_rif.grad, x_new.grad)),
            "dd_identico_al_bit": bool(torch.equal(d_rif.grad, d_new.grad))}


# ═══════════════════════════ T2 — dw contro riferimento indipendente ═══════════════════════════
def t2_gradiente_indipendente(fam: "S.DatiFamiglia", clip: float = 1.5, seed: int = 202) -> dict:
    torch.manual_seed(seed)
    out = {}
    for ricn in ("B64", "A64"):        # B64: ruotata (down non ruota mai comunque); A64: de-ruotata
        ric = S.RICETTE[ricn]
        idx = torch.randint(0, fam.E, (48, 3))
        piano = S.Piano(idx, fam.E)
        x = torch.randn(piano.S, fam.C, dtype=torch.float64)
        d0 = fam.d0.double()
        w0 = w_da_famiglia(fam, torch.float64)
        G = torch.randn(piano.S, fam.R, dtype=torch.float64)
        ruota_peso = fam.ruotata and not ric.ruota
        Lr, gxr, gwr = riferimento_famiglia_codici(x, d0, w0, fam, piano, ruota_peso, clip, G)

        xx, ww, dd_ = x.clone().requires_grad_(True), w0.clone().requires_grad_(True), d0.clone().requires_grad_(True)
        y = FamigliaEspertiConCodici.apply(xx, dd_, ww, fam, piano, ric, clip)
        (y * G).sum().backward()
        out[ricn] = {"ruotata_nel_backward": ruota_peso, "err_rel_dx": _rel(xx.grad, gxr),
                    "err_rel_dw": _rel(ww.grad, gwr), "loss_riferimento": Lr}
    return out


# ═══════════════════════════ T3 — ciclo minimo instradato ═══════════════════════════
def _ciclo(w0, d0, fam, piano, x, y_target, ric, clip, lr, passi, ottimizzatore) -> dict:
    w = w0.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([w], lr=lr, betas=(0.9, 0.95)) if ottimizzatore == "adam" else None
    l0 = None
    for _ in range(passi):
        y = FamigliaEspertiConCodici.apply(x, d0, w, fam, piano, ric, clip)
        L = (y - y_target).pow(2).sum()
        (opt.zero_grad() if opt else setattr(w, "grad", None))
        L.backward()
        if l0 is None:
            l0 = float(L)
        (opt.step() if opt else w.data.sub_(lr * w.grad))
    with torch.no_grad():
        n_flip = int((torch.clamp(torch.round(w), -1.0, 1.0) != torch.clamp(torch.round(w0), -1.0, 1.0)).sum())
        quota_clip = float((w.abs() > clip).double().mean())
    return {"ottimizzatore": ottimizzatore, "lr": lr, "passi": passi, "loss_iniziale": l0,
            "loss_finale": float(L), "flip": n_flip, "quota_oltre_clip": quota_clip, "scende": float(L) < l0}


def t3_ciclo_minimo(fam: "S.DatiFamiglia", fam_bersaglio: "S.DatiFamiglia", clip: float = 1.5,
                    seed: int = 303) -> dict:
    """Bersaglio: un'ALTRA esperta vera (stessa forma), instradata da un Piano VERO (non un caso degenere a
    un esperto solo -- copre il costrutto O(E) sotto instradamento reale)."""
    torch.manual_seed(seed)
    ric = S.RICETTE["B64"]
    idx = torch.randint(0, fam.E, (64, 4))
    piano = S.Piano(idx, fam.E)
    x = torch.randn(piano.S, fam.C, dtype=torch.float64)
    d0 = fam.d0.double()
    w0 = w_da_famiglia(fam, torch.float64)
    w_bersaglio = w_da_famiglia(fam_bersaglio, torch.float64)
    with torch.no_grad():
        y_target = FamigliaEspertiConCodici.apply(x, d0, w_bersaglio, fam_bersaglio, piano, ric, clip)
    quota_gia_uguale = float((torch.round(w0).clamp(-1, 1) == torch.round(w_bersaglio).clamp(-1, 1)).double().mean())
    sgd = _ciclo(w0, d0, fam, piano, x, y_target, ric, clip, lr=0.5, passi=60, ottimizzatore="sgd")
    adam = _ciclo(w0, d0, fam, piano, x, y_target, ric, clip, lr=0.03, passi=60, ottimizzatore="adam")
    return {"quota_gia_uguale_al_bersaglio": quota_gia_uguale, "n_codici": int(w0.numel()), "sgd": sgd, "adam": adam}


# ═══════════════════════════ T4 — cancello G1 per i codici ═══════════════════════════
def cancello_g1_codici(fam: "S.DatiFamiglia", K: float, ric_prod: "S.Ricetta" = None, clip: float = 1.5,
                       divisore: float = 131072.0, seed: int = 404) -> dict:
    """Come `AL.cancello_g1`, ma per i codici. Misura DUE cose distinte, come fa davvero il G1 di fase 1
    (non una sola: sono fenomeni diversi):
    1. `quota_gy_azzerati_fp16` -- ESATTAMENTE quello che misura `Osservatore.gradiente`/`aggrega_dy` di
       fase 1 ("dY azzerati"): quota di ELEMENTI di gy che sono != 0 in fp64 e diventano ESATTAMENTE 0 col
       cast `.to(fp16)` che `FamigliaEspertiConCodici.backward` fa davvero (`gy = gy.to(di)`, ricetta B:
       di = fp16). E' la quantita' che il gate `max_quota_dy_azzerati` (1e-4) protegge in fase 1.
    2. `quota_dw_rotti_o_azzerati` -- come "dtheta rotti"/`confronto_fp32` di `cancello_g1`: dopo il cast,
       il dL/dw AGGREGATO (somma su ~R x C_gruppo token) e' 0 o di segno sbagliato rispetto al riferimento
       fp64. E' una quantita' PIU' rara della prima (la somma su molti token puo' restare corretta anche se
       alcuni ADDENDI si sono azzerati): la differenza fra le due e' essa stessa informativa.
    Bersaglio: un'altra esperta vera (come T3), non rumore puro -- un gy che viene da un passo indietro
    vero, non inventato. `divisore`: STESSA normalizzazione per-token di `gradiente_passo`
    (`kl * (scala_k / n_tok)`, default 131.072 = E-B, §2.5 del progetto di fase 1) -- SENZA, G ~ N(0,1)
    produce un gy di grandezza ~1, lontanissimo dalla soglia fp16 (~6e-8: la prima versione di questo
    cancello, senza divisore, misurava zero azzeramenti SEMPRE, per costruzione del giocattolo, non perche'
    il rischio non ci sia). Dati SINTETICI (nessun teacher/corpus: non c'e' GPU oggi) -- la versione che
    lavora sul gradiente VERO del teacher e' `cancello_g1_codici_vivo` in `e2e_allena_codici.py`, che riusa
    QUESTA stessa `_dw`."""
    ric_prod = ric_prod or S.RICETTE["B"]
    torch.manual_seed(seed)
    idx = torch.randint(0, fam.E, (64, 4))
    piano = S.Piano(idx, fam.E)
    x64 = torch.randn(piano.S, fam.C, dtype=torch.float64)
    d64 = fam.d0.double()
    w64 = w_da_famiglia(fam, torch.float64)
    G64 = torch.randn(piano.S, fam.R, dtype=torch.float64)

    esito: dict = {"K0": K, "divisore": divisore, "giri": []}
    for giro in range(3):
        scala_k = K / divisore
        # riferimento fp64: nessun cast in nessun punto, K/n_tok*L come farebbe davvero il ciclo
        x_, w_ = x64.clone().requires_grad_(True), w64.clone().requires_grad_(True)
        y = FamigliaEspertiConCodici.apply(x_, d64, w_, fam, piano, S.RICETTE["B64"], 1.5)
        (scala_k * (y * G64).sum()).backward()
        dw_rif = w_.grad.clone()

        # gy allo STESSO punto operativo (stesso avanti, stessa K), per isolare SOLO il cast nell'indietro
        x2 = x64.clone().requires_grad_(True)
        y2 = FamigliaEspertiConCodici.apply(x2, d64, w64.clone(), fam, piano, S.RICETTE["B64"], 1.5)
        gy64 = torch.autograd.grad((scala_k * (y2 * G64).sum()), y2)[0]

        with torch.no_grad():
            xr64 = S.ruota(x64) if (fam.ruotata and not ric_prod.ruota) else x64
            di = ric_prod.dtype_indietro or torch.float16     # ricetta B: di = x.dtype = fp16 in produzione
            gy16 = gy64.to(di)
            xr16 = xr64.to(di)
            # 1. quota_gy_azzerati_fp16 (== "dY azzerati" di Osservatore/aggrega_dy)
            nz_gy = gy64 != 0
            n_nz_gy = max(int(nz_gy.sum()), 1)
            quota_gy_azzerati = int((nz_gy & (gy16 == 0)).sum()) / n_nz_gy
            # 2. dw aggregato dopo il cast, contro il riferimento fp64 (== "dtheta rotti"/confronto_fp32)
            dw_fp16 = _dw(gy16.to(torch.float64), xr16.to(torch.float64), d64, fam, piano, w64, 1.5)

        vivo = dw_rif != 0
        azzerati = vivo & (dw_fp16 == 0)
        rotti = vivo & (dw_fp16 != 0) & (torch.sign(dw_fp16) != torch.sign(dw_rif))
        n_vivo = max(int(vivo.sum()), 1)
        quota_dw_rotti_o_azzerati = int((azzerati | rotti).sum()) / n_vivo
        # STESSA normalizzazione di `statistiche_g(st, K)`: si riporta il gradiente al valore "come se K=1"
        # PRIMA di misurare q01, altrimenti il K di prova si mescolerebbe con la scala naturale che si sta
        # cercando -- `regola_k` confronta contro soglie assolute (1e-8), non contro un rapporto.
        g_abs = (dw_rif.abs() / K).reshape(-1)
        g_abs = g_abs[g_abs > 0]
        q01 = float(torch.kthvalue(g_abs, max(1, math.ceil(0.01 * g_abs.numel()))).values)
        r = AL.regola_k(q01)
        esito["giri"].append({"K_usata": K, "n_gy": int(gy64.numel()), "vivo_dw": n_vivo,
                              "quota_gy_azzerati_fp16": quota_gy_azzerati,
                              "quota_dw_rotti_o_azzerati": quota_dw_rotti_o_azzerati, "q01_dw": q01, **r})
        if r["K"] == K:
            break
        K = r["K"]
    esito.update(esito["giri"][-1])
    return esito


# ═══════════════════════════ test runner ═══════════════════════════
def _test() -> bool:
    torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "4")))
    ok = True
    t0 = time.time()
    fonti = [("blk.0.ffn_gate_exps.weight", "gate", slice(0, 4), slice(4, 8)),
            ("blk.3.ffn_down_exps.weight", "down", slice(0, 4), slice(4, 8))]
    print(f"GGUF: {GGUF_THOR_T}")
    for nome_t, nome, es, es_bersaglio in fonti:
        fam = carica_famiglia(nome_t, nome, es)
        fam_b = carica_famiglia(nome_t, nome, es_bersaglio)
        print(f"\n── {nome_t} ({fam.tipo}, esperti {es}, E={fam.E} R={fam.R} C={fam.C}; "
              f"bersaglio T3/T4: esperti {es_bersaglio}) ──")

        for ricn in ("B64", "A64") if fam.ruotata else ("B64",):
            r1 = t1_nessuna_regressione(fam, ricn)
            ok &= r1["avanti_identico_al_bit"] and r1["dx_identico_al_bit"] and r1["dd_identico_al_bit"]
            print(f"[T1 {ricn}] avanti={r1['avanti_identico_al_bit']} dx={r1['dx_identico_al_bit']} "
                  f"dd={r1['dd_identico_al_bit']} ({r1['N']:,} codici)")

        r2 = t2_gradiente_indipendente(fam)
        for ricn, v in r2.items():
            passa = v["err_rel_dx"] < 1e-9 and v["err_rel_dw"] < 1e-9
            ok &= passa
            print(f"[T2 {ricn}] ruotata_nel_backward={v['ruotata_nel_backward']} · err_rel dx="
                  f"{v['err_rel_dx']:.3e} dw={v['err_rel_dw']:.3e} · passa: {passa}")

        r3 = t3_ciclo_minimo(fam, fam_b)
        ok &= r3["sgd"]["scende"]
        print(f"[T3] bersaglio: un'altra esperta vera, {r3['quota_gia_uguale_al_bersaglio']:.3f} dei codici "
              f"gia' uguali · {r3['n_codici']:,} codici, instradati da un Piano vero")
        for tag in ("sgd", "adam"):
            e = r3[tag]
            print(f"  [{tag:4s}] lr {e['lr']}, {e['passi']} passi: loss {e['loss_iniziale']:.3f} -> "
                  f"{e['loss_finale']:.3f} (scende: {e['scende']}) · flip {e['flip']:,}/{r3['n_codici']:,} · "
                  f"oltre il clip: {e['quota_oltre_clip']:.4f}")

        r4a = cancello_g1_codici(fam, K=1.0)
        r4b = cancello_g1_codici(fam, K=2.0 ** 20)
        conv_a = r4a["giri"][-1]["K"] == r4a["giri"][-1]["K_usata"]
        conv_b = r4b["giri"][-1]["K"] == r4b["giri"][-1]["K_usata"]
        ok &= conv_a and conv_b
        print(f"[T4] K0=1: dY azzerati in fp16 {r4a['quota_gy_azzerati_fp16']:.4f} ({r4a['n_gy']:,} gy) · "
              f"dw rotti/azzerati {r4a['quota_dw_rotti_o_azzerati']:.2e} ({r4a['vivo_dw']:,} vivi) -> "
              f"K converge a 2^{math.log2(r4a['K']):.0f} in {len(r4a['giri'])} giri (converge: {conv_a})")
        print(f"     K0=2^20 (K0 di fase 1): dY azzerati {r4b['quota_gy_azzerati_fp16']:.2e} · "
              f"dw rotti/azzerati {r4b['quota_dw_rotti_o_azzerati']:.2e} -> "
              f"K converge a 2^{math.log2(r4b['K']):.0f} in {len(r4b['giri'])} giri (converge: {conv_b})")

    print(f"\n{time.time() - t0:.1f} s totali")
    print("ESITO:", "TUTTO VERDE" if ok else "⛔ QUALCOSA NON TORNA")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    if a.test:
        sys.exit(0 if _test() else 1)
    ap.print_help()
