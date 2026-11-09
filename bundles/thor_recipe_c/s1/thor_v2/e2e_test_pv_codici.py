#!/usr/bin/env python3
"""Prove di `e2e_pv_codici.py` (passo V sui codici, 26/9/2026 sera). SOLO CPU: la GPU e' di EB2. Il kernel
`_k_dd_v` gira nell'interprete di Triton (TRITON_INTERPRET=1); T5 lo COMPILA per gfx1151 senza eseguirlo.
Tutte le prove leggono i BYTE VERI di Thor-T (`Thor-v2-ThorT-scor150.gguf`).

  T1  segnale per codice SENZA dW (kernel `_k_dd_v`, riquadri nei registri) == segnale CON dW materializzato
      (`dd_v_torch`, dW (R, C) per esperto) entro fp32, su 16 micro-lotti (2 proposta + 14 verifica), esperti veri
      piccoli (sottoinsiemi di righe), gate TQ1_0R_1024 con righe a d0 = 0 e down TQ1_0, ingressi fp32 (B_ind32) e
      fp16 (B); dL/dd identico AL BIT in tutti i modi e a `_k_dd`; scelta del kernel ottima entro fp32 contro float64.
  T2  passo V su famiglie vere intere (strato 39, 8 esperti x 512/2048 righe): dopo i salti i codici restano
      ternari e canonici, cambiano SOLO i codici scelti (uno per gruppo, <= tau_e per esperto, == min(tau, scoperte)),
      byte cambiati solo quelli dei salti, campi d intatti; quattro decoder (due torch, Triton, codice C di
      llama.cpp) == trit nuovi x d al bit; annullamento e rigioco al bit; registro 9 B/salto con sha256; andata e
      ritorno attraverso un GGUF giocattolo coi blocchi veri (`patcha_codici` + `verifica_patch_codici`).
  T3  giocattolo con bersaglio raggiungibile (esperti veri, 1% dei codici spostati di un livello): passi V ripetuti
      fanno scendere la perdita held-out e non la fanno mai salire; P+V contro sole scale (P) sugli stessi dati;
      regione di fiducia: salti cattivi forzati -> sonda peggiore -> annullati al bit, rho dimezzato.
  T4  memoria transitoria di picco per uno strato vero intero (3 famiglie x 256 esperti): buffer, raccolta del
      segnale (contro il `dd` di oggi), selezione, salti; processo figlio, VmHWM azzerato per fase.
  T5  (in piu') `_k_dd_v` compilato per gfx1151 (MODO_V 0/1/2, R1024 e TQ1_0, ingressi fp16 e fp32): v_wmma e
      registri riversati, contro `_k_dd`.
  T6  (in piu') innesto nello `Studente` vero (2 strati, 64 esperti, ricetta B_ind32, CPU): theta.grad col motore V
      == senza AL BIT; buffer pieni; salti -> uscita cambia; annullamento -> uscita identica al bit.
  T11/T12 (27/9, GGUF vero) finestra di accumulo: potenza a pari token; ripresa a meta' finestra al bit.
  T13 come T12 su una famiglia TQ1_0 SINTETICA (gguf-py): l'unica prova dell'accumulo che non legge /mnt.
  T14-T18 (27/9, famiglie SINTETICHE, niente /mnt) V-Bop a gruppo (--pv-modo vbop) e rho adattivo: T14 default
      identico al bit al modulo di git 6bcec55; T15 V-Bop contro PV col maestro fuori griglia (piu' salti giusti,
      zero sbagliati); T16 isteresi, correzione alla ECO, azzeramento, annullamento; T17 ripresa al bit a meta'
      stato; T18 regione di fiducia che raddoppia/dimezza nei casi costruiti e nel passo vero.
  T19-T21 (27/9, ramo v-senza-bh, famiglie SINTETICHE) --pv-accetta guadagno (senza i due test BH): T19 default
      identico al bit al modulo di master 767b2a7 (PV e V-Bop+rho adattivo); T20 guadagno contro bh col maestro fuori
      griglia (misura di salti giusti/sbagliati/annullati; controlli: insiemi annidati sugli stessi buffer, budget,
      annullamento al bit); T21 ripresa al bit con l'opzione accesa (PV e V-Bop).

Uso (4 fili, nice, limiti di memoria):
    export XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
    systemd-run --user --scope -p CPUQuota=400% -p MemoryMax=8G nice -n 10 env CUDA_VISIBLE_DEVICES= \\
        HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=4 TRITON_INTERPRET=1 @@ROOT:venv@@/bin/python3 \\
        e2e_test_pv_codici.py [--solo T1,T2,...]
    esito: $E2E_PV_DIR/esito_test_pv_codici.json (default @@ROOT:lavoro@@/e2e/progetto_pv)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("TRITON_INTERPRET", "1")
import numpy as np  # noqa: E402
import torch  # noqa: E402

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import e2e_studente as S  # noqa: E402
import e2e_codici_scale as CS  # noqa: E402
import e2e_campi_scala as CAMPI  # noqa: E402
import e2e_pv_codici as PV  # noqa: E402
from gguf import GGUFReader  # noqa: E402

DIR = Path(os.getenv("E2E_PV_DIR", "@@ROOT:lavoro@@/e2e/progetto_pv"))
FILI = int(os.getenv("OMP_NUM_THREADS", "4"))
torch.set_num_threads(FILI)
ESITO: dict = {"data": time.strftime("%Y-%m-%d %H:%M:%S"), "gguf": str(S.GGUF_THOR_T), "prove": {}}
_TENS = None


def tens() -> dict:
    global _TENS
    if _TENS is None:
        _TENS = {t.name: t for t in GGUFReader(str(S.GGUF_THOR_T)).tensors}
    return _TENS


def registra(nome: str, ok: bool, **dati) -> bool:
    ESITO["prove"][nome] = {"ok": bool(ok), **dati}
    print(f"[{nome}] {'✅' if ok else '⛔'} {json.dumps(dati, ensure_ascii=False, default=str)[:1500]}", flush=True)
    return ok


def rel(a, b) -> float:
    a, b = a.double(), b.double()
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-300))


def famiglia(L: int, n: str, esperti=None, righe=None) -> S.DatiFamiglia:
    return S.famiglia_da_tensore(tens()[f"blk.{L}.{S.FAMIGLIE[n]}.weight"], n, esperti=esperti, righe=righe)


def trit_famiglia(fam: S.DatiFamiglia) -> torch.Tensor:
    return CAMPI.trit_torch(fam.blocchi.reshape(-1, fam.nb), fam.tipo).view(fam.E, fam.R, fam.C)


# ─────────────────────────────── T1 ───────────────────────────────
def _finestra_con_morti(L: int, n: str, esperti: slice, alt: int) -> slice:
    """Righe [r0, r0+alt) che contengono sia gruppi con d0 = 0 sia gruppi vivi (esclusione della proposta)."""
    f = famiglia(L, n, esperti=esperti)
    morti = (f.d0 == 0).any(-1).any(0)                     # (R,)
    idx = torch.nonzero(morti).reshape(-1).tolist()
    if not idx:
        return slice(0, alt)
    r0 = max(0, min(idx[0] - alt // 3, f.R - alt))
    return slice(r0, r0 + alt)


def prova_t1() -> bool:
    torch.manual_seed(11)
    ok_tot, dati = True, {}
    b64, b32 = S.Blocchi(), S.Blocchi(BR_dd=32)                  # produzione di oggi / consigliati per V (T5)
    casi = [("gate_L0_fp32", 0, "gate", slice(0, 3), None, 96, torch.float32, b64),
            ("down_L0_fp32", 0, "down", slice(0, 3), slice(0, 80), None, torch.float32, b64),
            ("gate_L39_fp16", 39, "gate", slice(5, 7), slice(32, 96), None, torch.float16, b64),
            ("gate_L39_fp32_BR32", 39, "gate", slice(0, 2), slice(0, 80), None, torch.float32, b32),
            ("down_L3_fp32_BR32", 3, "down", slice(10, 12), slice(0, 80), None, torch.float32, b32)]
    for etichetta, L, n, esp, righe, alt, dt, blk in casi:
        t0 = time.time()
        if righe is None:
            righe = _finestra_con_morti(L, n, esp, alt)
        fam = famiglia(L, n, esperti=esp, righe=righe)
        E = fam.E
        bk, bt, b64 = PV.BufferV.crea(fam), PV.BufferV.crea(fam), PV.BufferV.crea(fam)
        dd_bit_modi, dd_bit_kdd, dd_bit_torch, err_dd = True, True, True, 0.0
        dW_prop = []                                        # dW float64 dei micro-lotti di proposta (per la scelta)
        for k in range(16):
            modo = 1 if k < 2 else 2
            T = 40
            idx = torch.randint(0, E, (T, 2))
            if k == 5:
                idx = torch.randint(1, E, (T, 2))           # esperto 0 senza token in un micro-lotto di verifica
            piano = S.Piano(idx, E)
            x = torch.randn(piano.S, fam.C).to(dt)
            gy = (torch.randn(piano.S, fam.R) * 1e-3).to(dt)
            dd_k = PV.dd_v_triton(gy, x, fam, piano, blk, bk, modo)
            dd_0 = PV.dd_v_triton(gy, x, fam, piano, blk, None, 0)
            dd_s = S._dd_triton(gy, x, fam, piano, blk)
            dd_t = PV.dd_v_torch(gy, x, fam, piano, torch.float32, bt, modo)
            dd_m = S.MotoreTorch().dd(gy, x, fam, piano, torch.float32)
            PV.dd_v_torch(gy.double(), x.double(), fam, piano, torch.float64, b64, modo)
            dd_bit_modi &= torch.equal(dd_k, dd_0)
            dd_bit_kdd &= torch.equal(dd_0, dd_s)
            dd_bit_torch &= torch.equal(dd_t, dd_m)
            err_dd = max(err_dd, rel(dd_k, dd_t))
            if modo == 1:
                o = piano.offs.tolist()
                dW_prop.append([gy[o[e]:o[e + 1]].double().T @ x[o[e]:o[e + 1]].double() for e in range(E)])
        # modo 3 (curvatura): dL/dd = 0 in ogni motore; H del kernel contro (dY^2)^T (X^2) esplicito
        dd3_zero = True
        H_atteso = torch.zeros(E, fam.R, fam.NG, dtype=torch.float64)
        for k in range(2):
            idx = torch.randint(0, E, (40, 2))
            piano = S.Piano(idx, E)
            x = torch.randn(piano.S, fam.C).to(dt)
            gy = (torch.randn(piano.S, fam.R) * 1e-3).to(dt)
            dd3_zero &= bool((PV.dd_v_triton(gy, x, fam, piano, blk, bk, 3) == 0).all())
            dd3_zero &= bool((PV.dd_v_torch(gy, x, fam, piano, torch.float32, bt, 3) == 0).all())
            PV.dd_v_torch(gy.double(), x.double(), fam, piano, torch.float64, b64, 3)
            o = piano.offs.tolist()
            for e in range(E):
                if o[e + 1] > o[e]:
                    A2 = (gy[o[e]:o[e + 1]].double() ** 2).T @ (x[o[e]:o[e + 1]].double() ** 2)
                    v = A2.view(fam.R, fam.NG, fam.gs).gather(-1, bk.cand[e].long().unsqueeze(-1)).squeeze(-1)
                    H_atteso[e] += torch.where(bk.prop[e] != 0, v, torch.zeros_like(v))
        # (a) kernel (niente dW) contro torch (dW materializzato, fp32)
        stesso = bk.cand == bt.cand
        con_cand = bt.prop != 0
        m = stesso & (bk.prop != 0) & con_cand
        frac_cand = float((stesso | ~con_cand).double().mean())
        err = {"prop": rel(bk.prop[m], bt.prop[m]), "S": rel(bk.S[m], bt.S[m]), "Q": rel(bk.Q[m], bt.Q[m]),
               "H": rel(bk.H[m], bt.H[m]), "H_contro_esplicito_fp64": rel(bk.H.double(), H_atteso)}
        # (b) torch fp32 contro float64 (dW in float64, stessi buffer fp32)
        m64 = (bt.cand == b64.cand) & con_cand & (b64.prop != 0)
        err64 = {"prop": rel(bt.prop[m64], b64.prop[m64]), "S": rel(bt.S[m64], b64.S[m64]),
                 "Q": rel(bt.Q[m64], b64.Q[m64]), "H": rel(bt.H[m64], b64.H[m64])}
        # (c) la scelta del kernel e' ottima entro fp32: |prop_k| == max_{micro 1-2, codici fattibili} |acc| (float64)
        q = PV.S.trit_esperto
        migliore = torch.zeros(E, fam.R, fam.NG, dtype=torch.float64)
        for dws in dW_prop:
            for e in range(E):
                A = dws[e].view(fam.R, fam.NG, fam.gs)
                t = q(fam, e).to(torch.int32).view(fam.R, fam.NG, fam.gs)
                dz = -torch.sign(A * torch.sign(fam.d0[e].double()).unsqueeze(-1)).to(torch.int32)
                fatt = (dz != 0) & ((t + dz).abs() <= 1)
                migliore[e] = torch.maximum(migliore[e], torch.where(fatt, A.abs(), torch.zeros_like(A)).amax(-1))
        tol = 1e-5 * float(migliore.max())
        scelta_ottima = bool(((bk.prop.double().abs() - migliore).abs() <= tol).all())
        # (d) fattibilita' della scelta del kernel: t + dir in {-1,0,1}, dir = -sign(prop d0)
        fattibile = True
        for e in range(E):
            t = q(fam, e).to(torch.int64).view(fam.R, fam.NG, fam.gs)
            tc = t.gather(-1, bk.cand[e].to(torch.int64).unsqueeze(-1)).squeeze(-1)
            dr = -torch.sign(bk.prop[e].double() * fam.d0[e].double()).to(torch.int64)
            ha = bk.prop[e] != 0
            fattibile &= bool(((tc + dr).abs()[ha] <= 1).all()) and bool((dr[ha] != 0).all())
        morti = fam.d0 == 0
        esclusi = bool((bk.prop[morti] == 0).all()) and bool((bt.prop[morti] == 0).all())
        ok = (dd_bit_modi and dd_bit_kdd and dd_bit_torch and dd3_zero and err_dd < 1e-5 and frac_cand == 1.0
              and max(err.values()) < 1e-5 and max(err64.values()) < 1e-5 and scelta_ottima and fattibile
              and esclusi and int(con_cand.sum()) > 0)
        ok_tot &= ok
        dati[etichetta] = {
            "ok": ok, "tipo": fam.tipo, "esperti": E, "righe": [righe.start, righe.stop], "C": fam.C, "gs": fam.gs,
            "ingressi": str(dt).replace("torch.", ""), "BR": blk.BR_dd, "gruppi": int(bt.prop.numel()),
            "gruppi_con_candidato": int(con_cand.sum()), "gruppi_d0_zero": int(morti.sum()),
            "d0_zero_senza_candidato": esclusi, "cand_uguali_kernel_torch": frac_cand,
            "err_rel_kernel_contro_dW_fp32": err, "err_rel_dW_fp32_contro_fp64": err64,
            "scelta_kernel_ottima_entro_fp32": scelta_ottima, "scelta_kernel_fattibile": fattibile,
            "dd_modi_1_2_uguale_modo_0_al_bit": dd_bit_modi, "dd_modo_0_uguale_k_dd_al_bit": dd_bit_kdd,
            "dd_modo_3_zero": dd3_zero,
            "dd_v_torch_uguale_MotoreTorch_al_bit": dd_bit_torch, "err_rel_dd_kernel_contro_torch": err_dd,
            "secondi": round(time.time() - t0, 1)}
        print(f"  T1 {etichetta}: {json.dumps(dati[etichetta], default=str)}", flush=True)
    return registra("T1_segnale_senza_dW", ok_tot, casi=dati,
                    nota="kernel nell'interprete di Triton (CPU); micro-lotti 1-2 proposta, 3-16 verifica")


# ─────────────────────────────── bersaglio giocattolo (T2, T3) ───────────────────────────────
RIC = S.RICETTE["B32"]          # fp32, pesi trit x d nel dominio dei byte: x e' gia' «ruotata» (casuale)


def pianta_errori(fam: S.DatiFamiglia, frazione: float, seme: int) -> int:
    """Sposta di UN livello una `frazione` dei codici di ogni esperto (0 -> +-1 a caso, +-1 -> 0), sui byte."""
    g = torch.Generator().manual_seed(seme)
    T = trit_famiglia(fam)
    n = int(frazione * fam.R * fam.C)
    es, rs, cs, ta, tn = [], [], [], [], []
    for e in range(fam.E):
        pos = torch.randperm(fam.R * fam.C, generator=g)[:n]
        r, c = pos // fam.C, pos % fam.C
        t = T[e, r, c].to(torch.int32)
        su = torch.randint(0, 2, (n,), generator=g, dtype=torch.int32) * 2 - 1
        nuovo = torch.where(t == 0, su, torch.zeros_like(t))
        es.append(torch.full((n,), e, dtype=torch.int64))
        rs.append(r)
        cs.append(c)
        ta.append(t)
        tn.append(nuovo)
    return PV.salta_byte(fam.blocchi, fam.tipo, torch.cat(es), torch.cat(rs), torch.cat(cs), torch.cat(ta),
                         torch.cat(tn))


class Dati:
    """Micro-lotti deterministici per (passo, k): instradamento casuale, x correlata (x = z M)."""

    def __init__(self, fam: S.DatiFamiglia, token: int, k: int, seme: int, corr: float = 0.3):
        g = torch.Generator().manual_seed(seme)
        self.fam, self.token, self.k, self.seme = fam, token, k, seme
        self.M = torch.eye(fam.C) + corr * torch.randn(fam.C, fam.C, generator=g) / math.sqrt(fam.C)

    def lotto(self, passo: int, k: int, token=None):
        g = torch.Generator().manual_seed(self.seme * 100003 + passo * 101 + k)
        T = token or self.token
        idx = torch.randint(0, self.fam.E, (T, self.k), generator=g)
        piano = S.Piano(idx, self.fam.E)
        x = torch.randn(piano.S, self.fam.C, generator=g) @ self.M
        return x, piano


def perdita(fam_s, d, fam_t, x, piano, mot, n_tot):
    with torch.no_grad():
        y_t = S.MotoreTorch().avanti(x, fam_t.d0, fam_t, piano, RIC)
    y = S.FamigliaEsperti.apply(x, d, fam_s, piano, RIC, mot)
    return ((y - y_t) ** 2).sum() / n_tot


def scala(fam, theta):
    return S.ArrotondaFP16.apply(fam.d0 * torch.exp(theta), None)


def curvatura_mse(pv: PV.PassoV, fam_s, theta, dati: Dati, passo: int, n_tot: int, n_curv: int = 2,
                  mot=None) -> None:
    """Micro-lotti di CURVATURA per la perdita sum (y - y_t)^2 / n_tot: Hessiana rispetto a y = (2/n_tot) I, quindi
    vettore Rademacher x sqrt(2/n_tot) (covarianza = Hessiana: Fisher/Gauss-Newton campionato). theta.grad non cambia."""
    mot = mot or PV.MotoreTorchV(pv.stato)
    for k in range(n_curv):
        pv.curvatura()
        x, piano = dati.lotto(passo, 1000 + k)
        y = S.FamigliaEsperti.apply(x, scala(fam_s, theta), fam_s, piano, RIC, mot)
        g = torch.Generator().manual_seed(dati.seme * 7919 + passo * 31 + k)
        v = (torch.randint(0, 2, y.shape, generator=g).to(y.dtype) * 2 - 1) * math.sqrt(2.0 / n_tot)
        y.backward(v)


def raccogli(pv: PV.PassoV, fam_s, fam_t, theta, dati: Dati, passo: int, n_micro: int = 16, con_v: bool = True):
    """Un passo di raccolta: 16 micro-lotti, indietro su theta (P) e, se con_v, segnale V nei buffer + 2 micro-lotti
    di curvatura (criterio di secondo ordine)."""
    mot = PV.MotoreTorchV(pv.stato)
    pv.inizia_passo()
    theta.grad = None
    n_tot = n_micro * dati.token * dati.k
    tot = 0.0
    for k in range(n_micro):
        if con_v:
            pv.microlotto(k)
        x, piano = dati.lotto(passo, k)
        L = perdita(fam_s, scala(fam_s, theta), fam_t, x, piano, mot, n_tot)
        L.backward()
        tot += float(L)
    if con_v:
        gp = theta.grad.clone()
        curvatura_mse(pv, fam_s, theta, dati, passo, n_tot, mot=mot)
        if not torch.equal(theta.grad, gp):
            raise AssertionError("la curvatura ha toccato theta.grad")
    pv.fine_raccolta()
    return tot


# ─────────────────────────────── T2 ───────────────────────────────
def _canonici(fam: S.DatiFamiglia) -> int:
    """Byte dei codici NON canonici (valore != ceil(256 q / 243) delle cifre che il decoder legge): devono restare 0."""
    off = CS.LAYOUT[fam.tipo][2]                                   # prima del campo d: qs + qh
    B = fam.blocchi.reshape(-1, fam.nb)[:, :off].to(torch.int32)
    qh = PV._in_qh(torch.arange(off), fam.tipo).expand_as(B)
    q = PV._valore_q(B, qh)
    return int(((((q * 256 + 242) // 243) & 255) != B).sum())


def _quattro_decoder(fam: S.DatiFamiglia, T_atteso: torch.Tensor) -> dict:
    """trit x d (de-ruotato per R1024) dei byte con: CS.trit_da_blocchi, CAMPI.trit_torch, kernel Triton
    (`S.decodifica_pesi`, interprete) e codice C di llama.cpp. Tutti al bit contro T_atteso x d."""
    import e2e_esporta as EX
    from tq1_0r_1024_codec import S1024
    Bn = fam.blocchi.reshape(-1, fam.nb)
    t_cs = CS.trit_da_blocchi(Bn, fam.tipo).view(fam.E, fam.R, fam.C)
    t_ca = CAMPI.trit_torch(Bn, fam.tipo).view(fam.E, fam.R, fam.C)
    W_tr = S.decodifica_pesi(fam, fam.d0, torch.float32)
    W_at = (T_atteso.float().view(fam.E, fam.R, fam.NG, fam.gs) * fam.d0.unsqueeze(-1)).view(fam.E, fam.R, fam.C)
    Sgn = torch.from_numpy(S1024) if fam.ruotata else None
    W_rif = CS.pesi_da_trit_scale(T_atteso.reshape(-1, fam.qk), CS.scale_da_blocchi(Bn, fam.tipo).float(),
                                  fam.tipo, Sgn).numpy()
    W_c = EX._deq_c(Bn.numpy(), fam.tipo)
    return {"trit_codici_scale": bool(torch.equal(t_cs, T_atteso)), "trit_campi_scala": bool(torch.equal(t_ca, T_atteso)),
            "triton_trit_x_d": bool(torch.equal(W_tr, W_at)), "codice_c_llama": bool(np.array_equal(W_c, W_rif))}


def _gguf_giocattolo(p: Path, fams: dict) -> None:
    from gguf import GGUFWriter, GGMLQuantizationType as QT
    w = GGUFWriter(str(p), arch="qwen35moe")
    w.add_uint32("e2e.pv.prova", 1)
    for (L, n), f in sorted(fams.items()):
        w.add_tensor(f"blk.{L}.{S.FAMIGLIE[n]}.weight", f.blocchi.numpy().copy(), raw_dtype=QT[f.tipo])
    w.add_tensor("blk.0.attn_norm.weight", np.arange(2048, dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def _controlli_passo(fam, sel: PV.Selezione, reg: PV.Registro, B_prima, T_prima) -> dict:
    T_dopo = trit_famiglia(fam)
    tern = bool(((T_dopo >= -1) & (T_dopo <= 1)).all())
    cambiati = torch.nonzero(T_dopo != T_prima)
    righe = reg.righe
    pos_reg = set(zip(righe["e"].tolist(), righe["r"].tolist(), righe["c"].tolist()))
    pos_cam = set(map(tuple, cambiati.tolist()))
    solo_scelti = pos_reg == pos_cam and len(pos_reg) == len(reg)
    e_, r_, c_ = (torch.from_numpy(righe[k].astype(np.int64)) for k in ("e", "r", "c"))
    valori = bool(torch.equal(T_dopo[e_, r_, c_].to(torch.int64), torch.from_numpy(righe["t_nuovo"].astype(np.int64))))
    vecchi = bool(torch.equal(T_prima[e_, r_, c_].to(torch.int64),
                              torch.from_numpy(righe["t_vecchio"].astype(np.int64))))
    un_livello = bool((np.abs(righe["t_nuovo"].astype(int) - righe["t_vecchio"].astype(int)) == 1).all())
    g_ = c_ // fam.gs
    uno_per_gruppo = len(set(zip(e_.tolist(), r_.tolist(), g_.tolist()))) == len(reg)
    per_e = torch.bincount(e_, minlength=fam.E)
    d_sc = fam.d0.new_zeros(0)
    costo = torch.zeros(fam.E, dtype=torch.float64).index_add_(0, e_, _d_scelti(fam, sel).double() ** 2)
    budget_ok = bool((costo <= sel.budget + 1e-12 * sel.budget.abs().clamp_min(1)).all())
    costo_ok = bool(torch.allclose(costo, sel.costo))
    scelti_ok = bool(torch.equal(per_e, sel.scelti)) and bool((sel.scelti <= sel.convenienti).all())
    # byte: diversi solo dove cadono i salti; campi d intatti
    blk, byte, _ = PV.posizione_t(c_, fam.tipo)
    permessi = set(((e_ * fam.R + r_) * fam.NBR + blk * fam.nb + byte).tolist())
    diversi = set(torch.nonzero(fam.blocchi.view(-1) != B_prima.view(-1)).reshape(-1).tolist())
    off, k = CS.LAYOUT[fam.tipo][2], CS.LAYOUT[fam.tipo][3]
    d_intatti = bool(torch.equal(fam.blocchi.reshape(-1, fam.nb)[:, off:off + 2 * k],
                                 B_prima.reshape(-1, fam.nb)[:, off:off + 2 * k]))
    return {"salti": len(reg), "ternari": tern, "cambiano_solo_i_scelti": solo_scelti, "t_nuovo_letto": valori,
            "t_vecchio_letto": vecchi, "un_livello": un_livello, "uno_per_gruppo": uno_per_gruppo,
            "somma_d2_scelti_le_budget": budget_ok, "costo_registrato_uguale": costo_ok, "scelti_le_convenienti": scelti_ok,
            "byte_diversi": len(diversi), "byte_diversi_sottoinsieme_salti": diversi <= permessi,
            "byte_con_piu_salti": len(reg) - len(permessi), "campi_d_intatti": d_intatti,
            "byte_non_canonici": _canonici(fam), "convenienti_per_esperto": sel.convenienti.tolist()[:8],
            "scoperte_per_esperto": sel.scoperte.tolist()[:8], "testati": int(sel.testati.sum())}


_D_SCELTI = {}


def _d_scelti(fam, sel) -> torch.Tensor:
    """d usato dalla selezione per i salti scelti (registrato da chi chiama `seleziona` con d_eff)."""
    return _D_SCELTI[id(sel)]


def _prova_sf_t() -> dict:
    """sf_t contro (a) quantili noti della t di Student (tavole, 6 cifre) e (b) integrazione numerica della densita'
    (Simpson in u = atan(x / sqrt(nu)), integranda C sqrt(nu) cos^(nu-1) u, liscia), piu' +-inf."""
    tavola = [(1, 6.313752, 0.05), (1, 1.0, 0.25), (2, 2.919986, 0.05), (4, 2.131847, 0.05), (13, 1.770933, 0.05),
              (13, 2.160369, 0.025), (13, 3.012276, 0.005), (14, 1.761310, 0.05), (30, 2.042272, 0.025)]
    f64 = lambda t: torch.tensor([t], dtype=torch.float64)  # noqa: E731  (in fp32 0.7 non e' 0.7: +3.6e-9)
    err_tav = max(abs(float(PV.sf_t(f64(t), nu)[0]) - pa) for nu, t, pa in tavola)
    err_int = 0.0
    for nu in (1, 2, 3, 4, 5, 13, 14):
        C = math.exp(math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2)) / math.sqrt(nu * math.pi)
        for t in (-3.0, -0.5, 0.0, 0.7, 2.0, 5.0):
            u0 = math.atan(t / math.sqrt(nu))
            u = np.linspace(u0, math.pi / 2, 4001)
            f = C * math.sqrt(nu) * np.cos(u) ** (nu - 1)
            h = (u[-1] - u[0]) / 4000
            num = h / 3 * (f[0] + f[-1] + 4 * f[1:-1:2].sum() + 2 * f[2:-1:2].sum())
            err_int = max(err_int, abs(float(PV.sf_t(f64(t), nu)[0]) - num))
    inf = PV.sf_t(torch.tensor([math.inf, -math.inf]), 13).tolist()
    return {"err_tavole": err_tav, "err_integrazione": err_int, "sf_piu_meno_inf": inf,
            "ok": err_tav < 2e-6 and err_int < 1e-12 and inf == [0.0, 1.0]}


def _selezione_indipendente(buf: PV.BufferV, fam: S.DatiFamiglia, d_eff: torch.Tensor, n: int, rho: float,
                            sel: PV.Selezione, n_perdita: int, n_curv: int, q: float = 0.1, q_g: float = 0.1) -> dict:
    """Ricalcolo INDIPENDENTE (numpy float64, decoder di e2e_codici_scale, BH a ciclo per esperto) di scoperte,
    convenienti (BH sull'ipotesi «guadagno di secondo ordine <= 0» e guadagno > 0), budget del trust ratio e
    insieme scelto (prefisso per guadagno con sum d^2 <= budget); confronto con la Selezione."""
    T = CS.trit_da_blocchi(fam.blocchi.reshape(-1, fam.nb), fam.tipo).view(fam.E, fam.R, fam.NG, fam.gs)
    nnz = (T != 0).sum(-1).numpy().astype(np.float64)
    d = d_eff.double().numpy()
    ad, d2 = np.abs(d), d ** 2
    budget = rho * rho * (d2 * nnz).reshape(fam.E, -1).sum(1)
    Sd, Qd, pr, Hd = buf.S.double().numpy(), buf.Q.double().numpy(), buf.prop.double().numpy(), buf.H.double().numpy()
    s_dir = Sd * np.sign(pr)
    var = np.maximum((Qd - Sd * Sd / n) / (n - 1), 0.0)
    den = np.sqrt(n * var)

    def tstat(x):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(den > 0, x / den, np.where(x > 0, np.inf, np.where(x < 0, -np.inf, 0.0)))
    p = PV.sf_t(torch.from_numpy(tstat(s_dir)), n - 1).numpy()
    soglia = 0.5 * (n / n_curv) * Hd * ad
    x_g = s_dir - soglia
    with np.errstate(divide="ignore", invalid="ignore"):
        tg = np.where(den > 0, x_g / den, np.where(x_g > 0, np.inf, -np.inf))
    pg = PV.sf_t(torch.from_numpy(tg), n - 1).numpy()
    gq = (n_perdita / n) * s_dir * ad - 0.5 * (n_perdita / n_curv) * Hd * d2
    testati = (pr != 0) & (d != 0)

    def bh_np(te, pe, qq):
        m = int(te.sum())
        dis = np.zeros_like(te)
        if m:
            ps = np.sort(pe[te], kind="stable")
            ks = np.nonzero(ps <= np.arange(1, m + 1) * qq / m)[0]
            if ks.size:
                dis = te & (pe <= ps[ks.max()])
        return dis
    atteso = set()
    scop = np.zeros(fam.E, dtype=np.int64)
    conv_n = np.zeros(fam.E, dtype=np.int64)
    for e in range(fam.E):
        te = testati[e].reshape(-1)
        dis = bh_np(te, p[e].reshape(-1), q) & (s_dir[e].reshape(-1) > 0)
        scop[e] = int(dis.sum())
        conv = bh_np(te, pg[e].reshape(-1), q_g) & (gq[e].reshape(-1) > 0)
        conv_n[e] = int(conv.sum())
        idx = np.nonzero(conv)[0]
        idx = idx[np.argsort(-gq[e].reshape(-1)[idx], kind="stable")]
        cum = np.cumsum(d2[e].reshape(-1)[idx])
        idx = idx[cum <= budget[e]]
        atteso |= {(e, int(i) // fam.NG, int(i) % fam.NG) for i in idx}
    scelto = set(zip(sel.e.tolist(), sel.r.tolist(), sel.g.tolist()))
    return {"budget_uguale": bool(np.allclose(budget, sel.budget.numpy())), "scoperte_uguali": bool(np.array_equal(
        scop, sel.scoperte.numpy())), "convenienti_uguali": bool(np.array_equal(conv_n, sel.convenienti.numpy())),
        "insieme_scelto_uguale": scelto == atteso, "convenienti_indipendenti": conv_n.tolist()[:8]}


def prova_t2() -> bool:
    import e2e_pv_misure as MI                                   # sola lettura: `posizione` scalare
    t0 = time.time()
    ok, dati = True, {}
    dati["sf_t"] = _prova_sf_t()
    ok &= dati["sf_t"]["ok"]
    # (0) posizione vettoriale == scalare, tutte le colonne dei due formati
    for tipo, C in (("TQ1_0R_1024", 2048), ("TQ1_0", 512)):
        c = torch.arange(C)
        vb, vy, vn = PV.posizione_t(c, tipo)
        sc = [MI.posizione(i, tipo) for i in range(C)]
        uguale = [tuple(x) for x in zip(vb.tolist(), vy.tolist(), vn.tolist())] == sc
        dati[f"posizione_{tipo}"] = uguale
        ok &= uguale
    L = 39
    fams_t, fams_s = {}, {}
    for n in ("gate", "down"):
        fams_t[(L, n)] = PV.famiglia_scrivibile(famiglia(L, n, esperti=slice(0, 8)))
        fams_s[(L, n)] = PV.famiglia_scrivibile(famiglia(L, n, esperti=slice(0, 8)))
        dati[f"piantati_{n}_byte"] = pianta_errori(fams_s[(L, n)], 0.005, 7 + PV.FAMIGLIE_IDX[n])
    base = {k: f.blocchi.clone() for k, f in fams_s.items()}      # studente dopo la piantatura (sorgente del rigioco)
    scratch = DIR / "scratch_t2"
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    src = scratch / "t2_sorgente.gguf"
    _gguf_giocattolo(src, fams_s)
    registri = []
    for giro, rho in enumerate((0.01, 0.3)):
        for (_, n), fs in fams_s.items():
            ft = fams_t[(L, n)]
            pv = PV.PassoV({(L, n): fs})
            theta = torch.zeros_like(fs.d0, requires_grad=True)
            dt = Dati(fs, token=256, k=2, seme=100 + giro)
            raccogli(pv, fs, ft, theta, dt, passo=giro)
            B_prima, T_prima = fs.blocchi.clone(), trit_famiglia(fs)
            d_eff = scala(fs, theta).detach()
            st_v = pv.stato
            sel = PV.seleziona(st_v.buffer(fs), fs, d_eff, st_v.n_verifica, rho, n_perdita=st_v.n_perdita,
                               n_curvatura=st_v.n_curvatura)
            _D_SCELTI[id(sel)] = d_eff[sel.e, sel.r, sel.g]
            ind = _selezione_indipendente(st_v.buffer(fs), fs, d_eff, st_v.n_verifica, rho, sel, st_v.n_perdita,
                                          st_v.n_curvatura)
            reg = PV.applica_salti(fs, sel)
            c = _controlli_passo(fs, sel, reg, B_prima, T_prima)
            c["ricalcolo_indipendente"] = {k2: v2 for k2, v2 in ind.items() if not k2.endswith("indipendenti")}
            ok &= ind["budget_uguale"] and ind["scoperte_uguali"] and ind["convenienti_uguali"] and \
                ind["insieme_scelto_uguale"]
            c["n_verifica"] = pv.stato.n_verifica
            atteso = T_prima.clone()                              # trit attesi SENZA rileggere i byte nuovi
            ri = reg.righe
            atteso[torch.from_numpy(ri["e"].astype(np.int64)), torch.from_numpy(ri["r"].astype(np.int64)),
                   torch.from_numpy(ri["c"].astype(np.int64))] = torch.from_numpy(ri["t_nuovo"].astype(np.int8))
            dec = _quattro_decoder(fs, atteso)
            dec_ok = all(dec.values())
            c["decoder_al_bit"] = dec
            # annullamento al bit, poi riapplicazione (rigioco dello stesso registro)
            B_dopo = fs.blocchi.clone()
            PV.annulla({(L, n): fs}, reg)
            ann = bool(torch.equal(fs.blocchi, B_prima))
            PV.rigioca({(L, n): fs}, [reg])
            rig = bool(torch.equal(fs.blocchi, B_dopo))
            c["annulla_al_bit"], c["rigioca_al_bit"] = ann, rig
            chiave_ok = all(v for k2, v in c.items() if isinstance(v, bool))
            ok &= chiave_ok and dec_ok and c["salti"] > 0 and c["byte_non_canonici"] == 0
            dati[f"giro{giro}_rho{rho}_{n}"] = c
            print(f"  T2 giro {giro} rho {rho} {n}: {json.dumps(c, default=str)}", flush=True)
            registri.append(reg)
    # registro su disco: la LISTA (un registro per passo), 9 B/salto, sha256, scrittura atomica
    tutti = PV.Registro.unisci(registri)
    sha = PV.Registro.salva_lista(scratch / "registri.npz", registri)
    back = PV.Registro.carica_lista(scratch / "registri.npz", sha)
    reg_ok = len(back) == len(registri) and all(np.array_equal(a_.righe, b_.righe) for a_, b_ in zip(back, registri)) \
        and tutti.righe.itemsize == 9
    try:
        PV.Registro.carica_lista(scratch / "registri.npz", "0" * 64)
        sha_rifiuta = False
    except RuntimeError:
        sha_rifiuta = True
    dati["registro"] = {"salti": len(tutti), "registri": len(registri), "byte_per_salto": int(tutti.righe.itemsize),
                        "file_byte": (scratch / "registri.npz").stat().st_size, "sha256": sha[:16],
                        "rilettura_uguale": reg_ok, "sha_sbagliato_rifiutato": sha_rifiuta}
    ok &= reg_ok and sha_rifiuta
    # rigioco dalla base (ripresa = byte sorgente + registri) == stato finale
    rifatti = {k: PV.famiglia_scrivibile(S.DatiFamiglia(f.nome, f.tipo, base[k].clone(), f.d0, f.strato))
               for k, f in fams_s.items()}
    PV.rigioca(rifatti, registri)
    rig_base = all(torch.equal(rifatti[k].blocchi, fams_s[k].blocchi) for k in fams_s)
    dati["rigioco_dalla_base_al_bit"] = rig_base
    ok &= rig_base
    # andata e ritorno col FORMATO DEL GGUF: copia, patch dei codici, verifica sui file, rilettura dei tensori
    dst = scratch / "t2_patch.gguf"
    shutil.copyfile(src, dst)
    scritti = PV.patcha_codici(dst, registri)
    ver = PV.verifica_patch_codici(src, dst, registri)
    r = GGUFReader(str(dst))
    td = {t.name: t for t in r.tensors}
    riletti = all(np.array_equal(np.asarray(td[PV.nome_tensore(L, PV.FAMIGLIE_IDX[n])].data),
                                 fams_s[(L, n)].blocchi.numpy()) for (_, n) in fams_s)
    altri = np.array_equal(np.asarray(td["blk.0.attn_norm.weight"].data), np.arange(2048, dtype=np.float32))
    del r, td
    dati["gguf"] = {"byte_scritti": scritti, **ver, "tensori_riletti_uguali_ai_byte_in_memoria": riletti,
                    "altro_tensore_intatto": altri}
    ok &= riletti and altri and ver["byte_diversi"] > 0
    dati["secondi"] = round(time.time() - t0, 1)
    return registra("T2_passo_V_sui_byte_veri", ok, **dati)


# ─────────────────────────────── T3 ───────────────────────────────
def _wrong(fs, ft) -> int:
    return int((trit_famiglia(fs) != trit_famiglia(ft)).sum())


def _held(fs, ft, theta, held) -> float:
    x, piano = held
    with torch.no_grad():
        return float(perdita(fs, scala(fs, theta), ft, x, piano, S.MotoreTorch(), piano.S))


def _corsa(tipo: str, passi: int, fam_t, fam_s0, lr: float, seme: int) -> dict:
    fs = PV.famiglia_scrivibile(S.DatiFamiglia(fam_s0.nome, fam_s0.tipo, fam_s0.blocchi.clone(), fam_s0.d0,
                                               fam_s0.strato))
    ft = fam_t
    pv = PV.PassoV({(fs.strato, fs.nome): fs})
    theta = torch.zeros_like(fs.d0, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=lr) if "P" in tipo else None
    dati = Dati(fs, token=64, k=2, seme=seme)
    held = dati.lotto(10 ** 6, 0, token=2048)
    reg_f = PV.RegioneFiducia()
    storia = [{"passo": 0, "held": _held(fs, ft, theta, held), "codici_sbagliati": _wrong(fs, ft)}]
    for t in range(passi):
        raccogli(pv, fs, ft, theta, dati, t, con_v="V" in tipo)
        riga = {"passo": t + 1}
        if opt is not None:
            opt.step()
        if "V" in tipo:
            rho = reg_f.rho_passo()
            d_eff = scala(fs, theta).detach()
            xs, ps = dati.lotto(t + 1, 0)                             # sonda: primo micro-lotto del passo dopo
            n_tot = ps.S
            with torch.no_grad():
                prima = float(perdita(fs, d_eff, ft, xs, ps, S.MotoreTorch(), n_tot))
            reg, diag = pv.seleziona_e_salta({(fs.strato, fs.nome): d_eff}, rho)
            with torch.no_grad():
                dopo = float(perdita(fs, d_eff, ft, xs, ps, S.MotoreTorch(), n_tot))
            tiene = reg_f.esito(rho, prima, dopo, len(reg))
            if not tiene:
                pv.annulla(reg)
            riga.update(rho=rho, salti=len(reg), tenuto=tiene, sonda_prima=prima, sonda_dopo=dopo,
                        scoperte=diag[f"{fs.strato}.{fs.nome}"]["scoperte"],
                        convenienti=diag[f"{fs.strato}.{fs.nome}"]["convenienti"])
        riga["held"] = _held(fs, ft, theta, held)
        riga["codici_sbagliati"] = _wrong(fs, ft)
        storia.append(riga)
    return {"storia": storia, "fam": fs, "theta": theta, "pv": pv, "regione": reg_f, "dati": dati}


def prova_t3() -> bool:
    t0 = time.time()
    L = 39
    ft = PV.famiglia_scrivibile(famiglia(L, "gate", esperti=slice(0, 4), righe=slice(0, 64)))
    fs0 = PV.famiglia_scrivibile(famiglia(L, "gate", esperti=slice(0, 4), righe=slice(0, 64)))
    pianta_errori(fs0, 0.01, 31)
    # 27/9: 30 passi (erano 12). Le soglie restano quelle di prima; il passo V ora e' piu' prudente per costruzione
    # (trust ratio GARANTITO: prima tau lo superava fino a 2,9x; criterio di secondo ordine: salta solo se il
    # minimo quadratico sta oltre meta' livello) -> ~150-200 salti a passo invece di ~550: stessa strada, piu' passi.
    passi = 30
    corse = {tipo: _corsa(tipo, passi, ft, fs0, 1e-2, 5) for tipo in ("V", "P", "PV")}
    st = {k: v["storia"] for k, v in corse.items()}
    hv = [r["held"] for r in st["V"]]
    salite_v = [i for i in range(1, len(hv)) if hv[i] > hv[i - 1] * (1 + 1e-12)]
    fin = {k: v[-1]["held"] for k, v in st.items()}
    sbagliati = {k: (v[0]["codici_sbagliati"], v[-1]["codici_sbagliati"]) for k, v in st.items()}
    # (c) regione di fiducia: salti CATTIVI forzati (200 codici GIUSTI spostati di un livello) -> sonda peggiore ->
    #     annullati dal registro al bit, rho dimezzato
    c = corse["V"]
    fs, pv, reg_f, dati, theta = c["fam"], c["pv"], c["regione"], c["dati"], c["theta"]
    rho_tarato = reg_f.rho                                   # scelto dalla taratura, prima della prova cattiva
    d_eff = scala(fs, theta).detach()
    Ts, Tt = trit_famiglia(fs), trit_famiglia(ft)
    giusti = torch.nonzero(Ts == Tt)
    g = torch.Generator().manual_seed(77)
    scelti = giusti[torch.randperm(giusti.shape[0], generator=g)[:200]]
    e_, r_, c_ = scelti[:, 0], scelti[:, 1], scelti[:, 2]
    tc = Ts[e_, r_, c_].to(torch.int32)
    su = torch.randint(0, 2, (200,), generator=g, dtype=torch.int32) * 2 - 1
    dr = torch.where(tc == 0, su, -tc).to(torch.int8)
    zero = torch.zeros(0)
    cattiva = PV.Selezione(e_, r_, c_ // fs.gs, c_ % fs.gs, dr, zero, zero, zero, zero, zero, zero, zero)
    xs, ps = dati.lotto(passi + 1, 0)
    B0 = fs.blocchi.clone()
    rho_prima = reg_f.rho
    with torch.no_grad():
        prima = float(perdita(fs, d_eff, ft, xs, ps, S.MotoreTorch(), ps.S))
    reg = PV.applica_salti(fs, cattiva)
    with torch.no_grad():
        dopo = float(perdita(fs, d_eff, ft, xs, ps, S.MotoreTorch(), ps.S))
    tiene = reg_f.esito(reg_f.rho_passo(), prima, dopo, len(reg))
    if not tiene:
        PV.annulla({(fs.strato, fs.nome): fs}, reg)
    ripristino = bool(torch.equal(fs.blocchi, B0))
    fiducia = {"salti_cattivi": len(reg), "sonda_prima": prima, "sonda_dopo": dopo, "tenuto": tiene,
               "byte_ripristinati_al_bit": ripristino, "rho_prima": rho_prima, "rho_dopo": reg_f.rho}
    ok = (not salite_v and hv[-1] < 0.2 * hv[0] and fin["PV"] < 0.5 * fin["P"]
          and sbagliati["V"][1] < 0.5 * sbagliati["V"][0] and sbagliati["P"][1] == sbagliati["P"][0]
          and len(reg) > 0 and not tiene and ripristino and reg_f.rho == rho_prima / 2)
    riassunto = {k: [{kk: (round(vv, 6) if isinstance(vv, float) else vv) for kk, vv in r.items()} for r in v]
                 for k, v in st.items()}
    return registra("T3_giocattolo_V_contro_scale", ok, held_finale=fin, held_iniziale=hv[0],
                    codici_sbagliati_inizio_fine=sbagliati, salite_held_V=salite_v,
                    rho_scelto_dalla_taratura=rho_tarato, regione_di_fiducia=fiducia,
                    storia=riassunto, secondi=round(time.time() - t0, 1),
                    nota="strato 39 gate, esperti 0-3 righe 0-63 VERI come bersaglio; studente = stessi byte con 1% "
                         "dei codici spostati di un livello; x correlata; 16 micro-lotti da 128 posti; Adam lr 1e-2")


# ─────────────────────────────── T4 ───────────────────────────────
_T4_FIGLIO = r"""
import sys, os, json, time
sys.path.insert(0, %r)
import torch, numpy as np
torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "4")))
import e2e_studente as S, e2e_pv_codici as PV
from gguf import GGUFReader
L = int(sys.argv[1])

def mem():
    v = {}
    for l in open("/proc/self/status"):
        if l.startswith(("VmHWM", "VmRSS")):
            v[l.split(":")[0]] = int(l.split()[1]) * 1024
    return v

def azzera():
    open("/proc/self/clear_refs", "w").write("5")

class Fase:
    def __init__(self, nome, out):
        self.nome, self.out = nome, out
    def __enter__(self):
        azzera(); self.m0 = mem(); self.t0 = time.time(); return self
    def __exit__(self, *a):
        m1 = mem()
        self.out[self.nome] = {"picco_sopra_inizio_MiB": round((m1["VmHWM"] - self.m0["VmRSS"]) / 2**20, 2),
                               "resta_MiB": round((m1["VmRSS"] - self.m0["VmRSS"]) / 2**20, 2),
                               "secondi": round(time.time() - self.t0, 2)}

out = {}
tens = {t.name: t for t in GGUFReader(str(S.GGUF_THOR_T)).tensors}
with Fase("carica_3_famiglie", out):
    fams = {(L, n): PV.famiglia_scrivibile(S.famiglia_da_tensore(tens[f"blk.{L}.{S.FAMIGLIE[n]}.weight"], n))
            for n in S.FAMIGLIE}
out["byte_codici_e_scale_MiB"] = round(sum(f.blocchi.numel() for f in fams.values()) / 2**20, 2)
with Fase("buffer_V", out):
    pv = PV.PassoV(fams)
out["buffer_V_byte_esatti"] = pv.stato.byte()
out["gruppi"] = sum(f.E * f.R * f.NG for f in fams.values())
g = torch.Generator().manual_seed(4)
mot_v, mot_0 = PV.MotoreTorchV(pv.stato), S.MotoreTorch()
pv.inizia_passo()
picchi = {"dd_oggi": 0.0, "dd_V_proposta": 0.0, "dd_V_verifica": 0.0}
t_dd = {"dd_oggi": 0.0, "dd_V_proposta": 0.0, "dd_V_verifica": 0.0}
for k in range(16):
    modo = pv.microlotto(k)
    for (_, n), f in fams.items():
        idx = torch.randint(0, f.E, (256, 8), generator=g)
        piano = S.Piano(idx, f.E)
        x = torch.randn(piano.S, f.C, generator=g)
        gy = torch.randn(piano.S, f.R, generator=g) * 1e-3
        for nome, fn in (("dd_oggi", lambda: mot_0.dd(gy, x, f, piano, torch.float32)),
                         ("dd_V_proposta" if modo == 1 else "dd_V_verifica",
                          lambda: mot_v.dd(gy, x, f, piano, torch.float32))):
            if nome == "dd_oggi" and k not in (0, 2):
                continue
            azzera(); m0 = mem(); t0 = time.time()
            r = fn(); del r
            m1 = mem()
            picchi[nome] = max(picchi[nome], (m1["VmHWM"] - m0["VmRSS"]) / 2**20)
            t_dd[nome] += time.time() - t0
for k in range(2):
    pv.curvatura()
    for (_, n), f in fams.items():
        idx = torch.randint(0, f.E, (256, 8), generator=g)
        piano = S.Piano(idx, f.E)
        x = torch.randn(piano.S, f.C, generator=g)
        gy = torch.randn(piano.S, f.R, generator=g) * 1e-3
        azzera(); m0 = mem(); t0 = time.time()
        r = mot_v.dd(gy, x, f, piano, torch.float32); del r
        m1 = mem()
        picchi["dd_V_curvatura"] = max(picchi.get("dd_V_curvatura", 0.0), (m1["VmHWM"] - m0["VmRSS"]) / 2**20)
        t_dd["dd_V_curvatura"] = t_dd.get("dd_V_curvatura", 0.0) + time.time() - t0
pv.fine_raccolta()
out["raccolta_segnale"] = {"picco_per_chiamata_MiB": {k: round(v, 2) for k, v in picchi.items()},
                           "secondi_totali": {k: round(v, 1) for k, v in t_dd.items()},
                           "n_verifica": pv.stato.n_verifica, "posti_per_micro_lotto": 2048}
d_eff = {k: f.d0.clone() for k, f in fams.items()}
with Fase("selezione_rho_0.01", out):
    sel = {k: PV.seleziona(pv.stato.buffer(f), f, d_eff[k], pv.stato.n_verifica, 0.01,
                           n_perdita=pv.stato.n_perdita, n_curvatura=pv.stato.n_curvatura) for k, f in fams.items()}
out["selezione_esito"] = {f"{k[0]}.{k[1]}": {"testati": int(s.testati.sum()), "scoperte": int(s.scoperte.sum()),
                          "convenienti": int(s.convenienti.sum()), "salti": len(s)} for k, s in sel.items()}
# salti di taglia REALISTICA: sotto l'ipotesi nulla (dati casuali) la BH non scopre quasi nulla, quindi per misurare
# la memoria dei salti si prendono, per esperto, i gruppi per |prop| finche' sum d^2 sta nel budget (fattibili)
fitte = {}
for k, f in fams.items():
    s = sel[k]
    B = pv.stato.buffer(f)
    a = B.prop.abs().reshape(f.E, -1)
    v, i = torch.sort(a, dim=1, descending=True, stable=True)
    cum = torch.cumsum(f.d0.double().reshape(f.E, -1).gather(1, i) ** 2, dim=1)
    tieni = (cum <= s.budget[:, None]) & (v > 0)
    ee, jj = torch.nonzero(tieni, as_tuple=True)
    gi = i[ee, jj]
    r_, g_ = gi // f.NG, gi %% f.NG
    dr = (-torch.sign(B.prop[ee, r_, g_].double() * f.d0[ee, r_, g_].double())).to(torch.int8)
    z = torch.zeros(0)
    fitte[k] = PV.Selezione(ee, r_, g_, B.cand[ee, r_, g_].long(), dr, v[ee, jj].double(), s.testati, s.scoperte,
                            s.convenienti, s.budget, tieni.sum(1), z)
with Fase("salti", out):
    regs = [PV.applica_salti(fams[k], s) for k, s in fitte.items()]
out["salti_realistici"] = sum(len(r) for r in regs)
with Fase("annulla", out):
    for r in regs:
        PV.annulla(fams, r)
print(json.dumps(out))
""" % str(QUI)


def prova_t4() -> bool:
    L = int(os.getenv("E2E_PV_T4_STRATO", "3"))
    env = {**os.environ, "TRITON_INTERPRET": "0", "MALLOC_MMAP_THRESHOLD_": "65536", "MALLOC_TRIM_THRESHOLD_": "0",
           "CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": ""}
    t0 = time.time()
    r = subprocess.run([sys.executable, "-c", _T4_FIGLIO, str(L)], capture_output=True, text=True, env=env)
    if r.returncode != 0:
        return registra("T4_memoria_strato_vero", False, errore=r.stderr[-1500:])
    out = json.loads(r.stdout.strip().splitlines()[-1])
    gruppi = out["gruppi"]
    atteso = PV.BYTE_PER_GRUPPO * gruppi
    buf_ok = out["buffer_V_byte_esatti"] == atteso
    rac = out["raccolta_segnale"]["picco_per_chiamata_MiB"]
    extra_v = max(rac["dd_V_proposta"], rac["dd_V_verifica"]) - rac["dd_oggi"]
    ok = buf_ok and gruppi == 5_242_880 and out["salti_realistici"] > 0
    return registra("T4_memoria_strato_vero", ok, strato=L, gruppi=gruppi, byte_attesi_17_per_gruppo=atteso,
                    buffer_byte_misurati=out["buffer_V_byte_esatti"],
                    buffer_MiB=round(out["buffer_V_byte_esatti"] / 2 ** 20, 2),
                    extra_V_nel_dd_per_chiamata_MiB=round(extra_v, 2), misure=out,
                    secondi=round(time.time() - t0, 1),
                    nota="CPU, prototipo torch: RSS del processo figlio (VmHWM azzerato per fase, malloc con mmap "
                         "da 64 KiB). Sulla GPU il segnale sta nei registri di _k_dd_v: resta solo il buffer.")


# ─────────────────────────────── T5 ───────────────────────────────
_T5_FIGLIO = r"""
import sys, re, json
sys.path.insert(0, %r)
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
import e2e_studente as S, e2e_pv_codici as PV
tgt = GPUTarget("hip", "gfx1151", 32)
def info(fn, sig, cex, nw):
    k = triton.compile(ASTSource(fn=fn, signature=sig, constexprs=cex), target=tgt, options={"num_warps": nw})
    a = k.asm["amdgcn"]
    return {"v_wmma": len(re.findall(r"v_wmma", a)), "spill": int(re.findall(r"vgpr_spill_count:\s+(\d+)", a)[0]),
            "vgpr": int(re.findall(r"\.vgpr_count:\s+(\d+)", a)[0]) if re.findall(r"\.vgpr_count:\s+(\d+)", a) else -1}
out = {}
for dt in ("fp16", "fp32"):
    gd = {"dy_ptr": "*" + dt, "sdy_s": "i32", "sdy_r": "i32", "x_ptr": "*" + dt, "sx_s": "i32", "sx_c": "i32",
          "pb": "*u8", "sb_e": "i32", "NBR": "i32", "dd_ptr": "*fp32", "sdd_e": "i32", "NG": "i32", "offs_ptr": "*i64",
          "R": "i32", "C": "i32", "GS": "constexpr", "QK": "constexpr", "NB": "constexpr", "ROT": "constexpr",
          "BR": "constexpr", "BS": "constexpr"}
    gv = dict(gd)
    for k2, v2 in (("d0_ptr", "*fp32"), ("sd0_e", "i32"), ("cand_ptr", "*u8"), ("prop_ptr", "*fp32"),
                   ("s_ptr", "*fp32"), ("q_ptr", "*fp32"), ("h_ptr", "*fp32"), ("sv_e", "i32")):
        gv[k2] = v2
    gv = {k2: gv[k2] for k2 in list(gd)[:15] + ["d0_ptr", "sd0_e", "cand_ptr", "prop_ptr", "s_ptr", "q_ptr", "h_ptr",
                                                "sv_e"] + list(gd)[15:]}
    gv["MODO_V"] = "constexpr"
    for nome, rot, gs, qk, nb in (("R1024", 1, 128, 1024, 224), ("TQ1_0", 0, 256, 256, 54)):
        for br, nw in ((64, 8), (32, 8), (32, 4), (16, 4)):
            c = {"GS": gs, "QK": qk, "NB": nb, "ROT": rot, "BR": br, "BS": 16}
            out[f"k_dd_{nome}_{dt}_BR{br}_w{nw}"] = info(S._k_dd, gd, c, nw)
            for m in (0, 1, 2, 3):
                out[f"k_dd_v_{nome}_{dt}_BR{br}_w{nw}_M{m}"] = info(PV._k_dd_v, gv, {**c, "MODO_V": m}, nw)
print(json.dumps(out))
""" % str(QUI)


def prova_t5() -> bool:
    env = {**os.environ, "TRITON_INTERPRET": "0", "CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "",
           "ROCR_VISIBLE_DEVICES": ""}                           # solo compilazione: nessun dispositivo visibile
    t0 = time.time()
    r = subprocess.run([sys.executable, "-c", _T5_FIGLIO], capture_output=True, text=True, env=env)
    if r.returncode != 0:
        return registra("T5_compilazione_gfx1151", False, errore=r.stderr[-1500:])
    out = json.loads(r.stdout.strip().splitlines()[-1])
    blk = S.Blocchi()
    prod = f"BR{blk.BR_dd}_w{blk.num_warps}"                       # configurazione di produzione di oggi (EB2)
    default = {k: v for k, v in out.items() if f"_{prod}" in k}
    # (a) MODO_V = 0 compila come `_k_dd` (stessi v_wmma, VGPR, spill) in OGNI configurazione
    m0_uguale = all(out[k.replace("k_dd_", "k_dd_v_", 1) + "_M0"] == v for k, v in out.items()
                    if k.startswith("k_dd_") and not k.startswith("k_dd_v_"))
    # (b) per formato x dtype: una configurazione dove i modi 1 e 2 non riversano PIU' del `_k_dd` di produzione
    configs = sorted({k.rsplit("_", 3)[-3] + "_" + k.rsplit("_", 3)[-2] for k in out if k.startswith("k_dd_v_")})
    scelte, ok_b = {}, True
    for nome in ("R1024", "TQ1_0"):
        for dt in ("fp16", "fp32"):
            oggi = out[f"k_dd_{nome}_{dt}_{prod}"]["spill"]
            buone = []
            for cfg in configs:
                s1 = out[f"k_dd_v_{nome}_{dt}_{cfg}_M1"]["spill"]
                s2 = out[f"k_dd_v_{nome}_{dt}_{cfg}_M2"]["spill"]
                if s1 <= oggi and s2 <= oggi:
                    buone.append((max(s1, s2), -int(cfg.split("_")[0][2:]), cfg, s1, s2))
            buone.sort()
            scelte[f"{nome}_{dt}"] = {"spill_k_dd_produzione": oggi,
                                      "migliore_V": ({"blocchi": buone[0][2], "spill_M1": buone[0][3],
                                                      "spill_M2": buone[0][4]} if buone else None)}
            ok_b &= bool(buone)
    ok = m0_uguale and ok_b
    return registra("T5_compilazione_gfx1151", ok, modo0_compila_come_k_dd=m0_uguale, scelte_per_formato=scelte,
                    produzione=default, tutte=out, secondi=round(time.time() - t0, 1),
                    nota="compilazione anticipata (nessuna GPU usata); spill = VGPR riversati in memoria. Con i "
                         "Blocchi di produzione (BR 64, 8 warp) il modo 1 riversa: il kernel V vuole i suoi blocchi")


# ─────────────────────────────── T6 ───────────────────────────────
def prova_t6() -> bool:
    t0 = time.time()
    torch.manual_seed(6)
    strati = [38, 39]

    def nuovo():
        st = S.Studente(strati=strati, ricetta="B_ind32", motore_nome="torch", device="cpu",
                        esperti=slice(0, 64), log=lambda *_: None)
        for L in strati:
            for f in st.esperti(L).fam.values():
                PV.famiglia_scrivibile(f)
        return st

    T = 24
    ids = torch.zeros(1, T, dtype=torch.long)
    # segnale COERENTE fra i micro-lotti (stesso bersaglio lineare, ingressi quasi uguali): con un bersaglio nuovo a
    # ogni micro-lotto il gradiente e' rumore puro e la BH, giustamente, non scopre nulla (misurato: 0 su 594k)
    base = torch.randn(1, T, 2048) * 0.05
    x0s = [base + 0.005 * torch.randn(1, T, 2048) for _ in range(16)]
    G = torch.randn(T, 2048)
    Gs = [G] * 16

    def passo(st, pv=None):
        for p in st.parametri():
            p.grad = None
        if pv is not None:
            pv.inizia_passo()
        for k in range(16):
            if pv is not None:
                pv.microlotto(k)
            st.gradiente(ids, lambda h: (h[0] * Gs[k]).sum() / (16 * T), x0=x0s[k])
        if pv is not None:
            # perdita LINEARE in h: la sua Hessiana e' 0, quindi il vettore di curvatura e' zero (H = 0); il modo 3
            # deve girare e NON toccare theta.grad (confronto al bit qui sotto con lo studente senza V)
            for k in range(2):
                pv.curvatura()
                st.gradiente(ids, lambda h: (h[0] * 0.0).sum(), x0=x0s[k])
            pv.fine_raccolta()
        return [p.grad.clone() for p in st.parametri()]

    st0 = nuovo()
    g0 = passo(st0)
    del st0
    st = nuovo()
    stato = PV.StatoV()
    fams = PV.installa(st, stato, motore="torch")
    pv = PV.PassoV(fams, stato=stato)
    g1 = passo(st, pv)
    grad_al_bit = all(torch.equal(a, b) for a, b in zip(g0, g1))
    pieni = {f"{k[0]}.{k[1]}": int((stato.buffer(f).prop != 0).sum()) for k, f in fams.items()}
    xt = base + 0.005 * torch.randn(1, T, 2048)
    with torch.no_grad():
        a0 = st.avanti(ids, x0=xt)
    scale = {k: st.esperti(k[0]).scala(k[1]).detach() for k in fams}
    reg, diag = pv.seleziona_e_salta(scale, 0.3)
    with torch.no_grad():
        a1 = st.avanti(ids, x0=xt)
    cambia = not torch.equal(a0["xL"], a1["xL"])
    L0 = float((st.esce(a0["xL"])[0] * G).sum())
    L1 = float((st.esce(a1["xL"])[0] * G).sum())
    scende = L1 < L0
    instr = PV.cambi_instradamento(a0["instr"][1], a1["instr"][1])
    pv.annulla(reg)
    with torch.no_grad():
        a2 = st.avanti(ids, x0=xt)
    ritorno = torch.equal(a0["xL"], a2["xL"])
    ok = grad_al_bit and all(v > 0 for v in pieni.values()) and len(reg) > 0 and cambia and ritorno \
        and stato.n_verifica == 14 and stato.n_curvatura == 2 and scende
    return registra("T6_innesto_studente_vero", ok, strati=strati, esperti=64, ricetta="B_ind32",
                    theta_grad_col_motore_V_uguale_al_bit=grad_al_bit, gruppi_con_candidato=pieni,
                    n_verifica=stato.n_verifica, salti=len(reg), diagnostica=diag, uscita_cambia_coi_salti=cambia, perdita_sonda_prima_dopo=[L0, L1],
                    quota_token_con_top8_cambiato_strato_39=instr, annulla_uscita_al_bit=ritorno,
                    secondi=round(time.time() - t0, 1),
                    nota="dati sintetici (x0 casuale, perdita lineare): prova l'innesto, non la qualita'")



# ─────────────────────────────── T7-T10: cure della revisione del 26/9 ───────────────────────────────
def prova_t7() -> bool:
    """Difetto 1 (solo primo ordine): maestro FUORI griglia (residuo 0,4 d, ingressi scorrelati, byte veri strato 39
    gate) -> nessun salto e la perdita held-out non sale; controprova residuo 0,9 d -> salti e la perdita scende."""
    t0 = time.time()
    tens_ = tens()

    def corsa(residuo: float) -> dict:
        fs = PV.famiglia_scrivibile(S.famiglia_da_tensore(tens_["blk.39.ffn_gate_exps.weight"], "gate",
                                                          esperti=slice(0, 4), righe=slice(0, 64)))
        E, R, C, gs = fs.E, fs.R, fs.C, fs.gs
        g = torch.Generator().manual_seed(3)
        Ws = torch.stack([S.peso_esperto_torch(fs, e, fs.d0, RIC) for e in range(E)])
        segno = torch.randint(0, 2, (E, R, C), generator=g).float() * 2 - 1
        Wt = Ws + residuo * fs.d0.repeat_interleave(gs, -1) * segno

        def lotto(seme, T=512, k=2):
            gg = torch.Generator().manual_seed(seme)
            piano = S.Piano(torch.randint(0, E, (T, k), generator=gg), E)
            return torch.randn(piano.S, C, generator=gg), piano

        def perdita_(x, piano, d, mot, n_tot):
            y = S.FamigliaEsperti.apply(x, d, fs, piano, RIC, mot)
            o = piano.offs.tolist()
            yt = torch.cat([x[o[e]:o[e + 1]] @ Wt[e].T for e in range(E)])
            return ((y - yt) ** 2).sum() / n_tot

        pv = PV.PassoV({(fs.strato, fs.nome): fs})
        mot = PV.MotoreTorchV(pv.stato)
        theta = torch.zeros_like(fs.d0, requires_grad=True)
        pv.inizia_passo()
        n_tot = 16 * 1024
        for k in range(16):
            pv.microlotto(k)
            x, piano = lotto(100 + k)
            perdita_(x, piano, scala(fs, theta), mot, n_tot).backward()
        gp = theta.grad.clone()
        for k in range(2):
            pv.curvatura()
            x, piano = lotto(500 + k)
            y = S.FamigliaEsperti.apply(x, scala(fs, theta), fs, piano, RIC, mot)
            gr = torch.Generator().manual_seed(900 + k)
            y.backward((torch.randint(0, 2, y.shape, generator=gr).float() * 2 - 1) * math.sqrt(2.0 / n_tot))
        pv.fine_raccolta()
        xh, ph = lotto(10 ** 6, T=8192)
        d = scala(fs, theta).detach()
        with torch.no_grad():
            prima = float(perdita_(xh, ph, d, S.MotoreTorch(), ph.S))
        out = {"theta_grad_intatto": bool(torch.equal(theta.grad, gp))}
        for rho in (0.01, 0.1):
            reg, diag = pv.seleziona_e_salta({(fs.strato, fs.nome): d}, rho)
            with torch.no_grad():
                dopo = float(perdita_(xh, ph, d, S.MotoreTorch(), ph.S))
            out[str(rho)] = {**diag[f"{fs.strato}.{fs.nome}"], "held_prima": prima, "held_dopo": dopo}
            pv.annulla(reg)
        return out
    a, b = corsa(0.4), corsa(0.9)
    ok = (a["theta_grad_intatto"] and b["theta_grad_intatto"]
          and all(a[r]["held_dopo"] <= a[r]["held_prima"] for r in ("0.01", "0.1"))
          and all(b[r]["held_dopo"] < b[r]["held_prima"] and b[r]["salti"] > 0 for r in ("0.01", "0.1")))
    return registra("T7_secondo_ordine_maestro_fuori_griglia", ok, residuo_0_4=a, residuo_0_9=b,
                    secondi=round(time.time() - t0, 1),
                    nota="prima della cura (26/9): 607 salti a rho 0,1 e held-out 4,715604 -> 4,722180 (sale)")


def prova_t8() -> bool:
    """Difetto 2 (tau): ||Delta W_e|| / ||W_e|| <= rho GARANTITO, su d0 e nnz VERI (strati 0 e 39, gate e down,
    tutti gli esperti), con ogni gruppo scoperta conveniente, |S| iid e caso peggiore |S| ~ |d|."""
    t0 = time.time()
    tens_ = tens()
    g = torch.Generator().manual_seed(0)
    ok, dati = True, {}
    for rho in (0.01, 0.1):
        for L in (0, 39):
            for n in ("gate", "down"):
                f = S.famiglia_da_tensore(tens_[f"blk.{L}.{S.FAMIGLIE[n]}.weight"], n)
                E = f.E
                nnz = PV.nnz_gruppi(f, 0, E).double()
                d = f.d0
                W2 = (d.double().reshape(E, -1) ** 2 * nnz.reshape(E, -1)).sum(1)
                for caso in ("iid", "peggiore"):
                    buf = PV.BufferV.crea(f, device=torch.device("cpu"))
                    mu = (torch.randn(d.shape, generator=g, dtype=torch.float64).abs() if caso == "iid"
                          else d.double().abs())
                    buf.prop.fill_(1.0)
                    buf.S.copy_((14 * mu).float())
                    buf.Q.copy_((14 * mu * mu).float())
                    sel = PV.seleziona(buf, f, d, 14, rho, PV.ConfigV(), n_perdita=16, n_curvatura=2)
                    dd2 = torch.zeros(E, dtype=torch.float64).index_add_(0, sel.e, d[sel.e, sel.r, sel.g].double() ** 2)
                    rap = torch.sqrt(dd2 / W2)[W2 > 0] / rho
                    r_max = float(rap.max())
                    ok &= r_max <= 1.0 + 1e-12 and bool(torch.allclose(sel.costo, dd2))
                    dati[f"rho{rho}_L{L}_{n}_{caso}"] = {"salti": len(sel), "mediana": round(float(rap.median()), 4),
                                                        "massimo": r_max}
    return registra("T8_trust_ratio_garantito", ok, casi=dati, secondi=round(time.time() - t0, 1),
                    nota="prima della cura (26/9): mediana 1,12-1,61 x rho, massimo 2,94 x rho")


def prova_t9() -> bool:
    """Difetti 3-4: lo STESSO codice che salta in passi diversi (registri distinti) si rigioca, si annulla e si
    esporta dalla LISTA; l'unione applicata e' rifiutata (documentato); il salvataggio e' atomico (un errore a meta'
    lascia intatto il file valido di prima, nessun .parziale)."""
    t0 = time.time()
    fs = PV.famiglia_scrivibile(famiglia(39, "gate", esperti=slice(0, 4), righe=slice(0, 64)))
    base = fs.blocchi.clone()
    g = torch.Generator().manual_seed(9)
    T0 = trit_famiglia(fs)

    def sel_da(pos, dirs):
        e_, r_, c_ = pos[:, 0], pos[:, 1], pos[:, 2]
        z = torch.zeros(0)
        return PV.Selezione(e_, r_, c_ // fs.gs, c_ % fs.gs, dirs.to(torch.int8), z, z, z, z, z, z, z)
    tutte = torch.stack(torch.meshgrid(torch.arange(fs.E), torch.arange(fs.R), torch.arange(fs.C), indexing="ij"),
                        -1).reshape(-1, 3)
    scelte = tutte[torch.randperm(tutte.shape[0], generator=g)[:80]]
    t_ = T0[scelte[:, 0], scelte[:, 1], scelte[:, 2]].to(torch.int32)
    su = torch.randint(0, 2, (80,), generator=g, dtype=torch.int32) * 2 - 1
    dir1 = torch.where(t_ == 0, su, -t_)
    p1 = scelte[:50]
    reg1 = PV.applica_salti(fs, sel_da(p1, dir1[:50]))
    T1_ = trit_famiglia(fs)
    rip = p1[:20]                                                   # 20 codici di reg1 saltano DI NUOVO in reg2
    t_r = T1_[rip[:, 0], rip[:, 1], rip[:, 2]].to(torch.int32)
    dir_r = torch.where(t_r == 0, torch.randint(0, 2, (20,), generator=g, dtype=torch.int32) * 2 - 1, -t_r)
    p2 = torch.cat([rip, scelte[50:80]])
    reg2 = PV.applica_salti(fs, sel_da(p2, torch.cat([dir_r, dir1[50:80]])))
    finale = fs.blocchi.clone()
    rif = PV.famiglia_scrivibile(S.DatiFamiglia(fs.nome, fs.tipo, base.clone(), fs.d0, fs.strato))
    PV.rigioca({(39, "gate"): rif}, [reg1, reg2])
    lista_ok = bool(torch.equal(rif.blocchi, finale))
    rif2 = PV.famiglia_scrivibile(S.DatiFamiglia(fs.nome, fs.tipo, base.clone(), fs.d0, fs.strato))
    try:
        PV.rigioca({(39, "gate"): rif2}, [PV.Registro.unisci([reg1, reg2])])
        unione_rifiutata = False
    except ValueError:
        unione_rifiutata = True
    PV.annulla({(39, "gate"): fs}, reg2)
    PV.annulla({(39, "gate"): fs}, reg1)
    annulla_ok = bool(torch.equal(fs.blocchi, base))
    scratch = DIR / "scratch_t9"
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    pz = scratch / "registri.npz"
    sha = PV.Registro.salva_lista(pz, [reg1])
    vero_replace = PV.os.replace

    def rotto(*a, **k):
        raise OSError("crollo simulato a meta' del salvataggio")
    PV.os.replace = rotto
    try:
        PV.Registro.salva_lista(pz, [reg1, reg2])
        crollo = False
    except OSError:
        crollo = True
    finally:
        PV.os.replace = vero_replace
    dopo = PV.Registro.carica_lista(pz, sha)
    atomico = crollo and len(dopo) == 1 and np.array_equal(dopo[0].righe, reg1.righe) and \
        not pz.with_name(pz.name + ".parziale").exists()
    # export della lista su un GGUF giocattolo (copia .parziale, verifica, rinomina) + guardia sul sorgente
    src = scratch / "t9.gguf"
    fb = PV.famiglia_scrivibile(S.DatiFamiglia(fs.nome, fs.tipo, base.clone(), fs.d0, fs.strato))
    _gguf_giocattolo(src, {(39, "gate"): fb})
    dst = scratch / "t9_codici.gguf"
    rep_ = PV.esporta_codici(src, dst, [reg1, reg2])
    r = GGUFReader(str(dst))
    letto = np.asarray({t.name: t for t in r.tensors}["blk.39.ffn_gate_exps.weight"].data)
    export_ok = np.array_equal(letto, finale.numpy()) and not dst.with_name(dst.name + ".parziale").exists()
    del r
    try:
        PV.esporta_codici(src, src, [reg1])
        guardia = False
    except ValueError:
        guardia = True
    ok = lista_ok and unione_rifiutata and annulla_ok and atomico and export_ok and guardia
    return registra("T9_registri_ripetuti_e_atomici", ok, salti_reg1=len(reg1), salti_reg2=len(reg2),
                    ripetuti=20, rigioco_lista_al_bit=lista_ok, unione_applicata_rifiutata=unione_rifiutata,
                    annulla_al_bit=annulla_ok, salvataggio_atomico=atomico, export_lista=rep_,
                    export_uguale_ai_byte=export_ok, guardia_sorgente=guardia, secondi=round(time.time() - t0, 1))


def prova_t10() -> bool:
    """Difetto 5 (export scale+codici): GGUF con tensori VERI (strati 0 e 3), registri di due passi sullo strato 0
    (gate e down, con ripetizioni) -> esporta_codici + ricarica_codici (studente dal file finale == studente dal
    file di partenza + rigioco, byte e uscita AL BIT); un registro manomesso fa fallire l'export senza lasciare file."""
    import e2e_test_allena as TA
    t0 = time.time()
    scratch = DIR / "scratch_t10"
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    src = scratch / "ridotto.gguf"
    TA.gguf_ridotto(src, (0, 3))
    st = S.Studente(gguf=src, strati=[0], ricetta="B", motore_nome="torch", device="cpu", log=lambda *_: None)
    fams = {(0, n): PV.famiglia_scrivibile(f) for n, f in st.esperti(0).fam.items()}
    g = torch.Generator().manual_seed(10)
    regs = []
    for passo in range(2):
        parti = []
        for n in ("gate", "down"):
            f = fams[(0, n)]
            T_ = CS.trit_da_blocchi(f.blocchi.reshape(-1, f.nb), f.tipo).view(f.E, f.R, f.C)
            e_ = torch.randint(0, f.E, (200,), generator=g)
            r_ = torch.randint(0, f.R, (200,), generator=g)
            c_ = torch.randint(0, f.C, (200,), generator=g)
            chiave = (e_ * f.R + r_) * f.C + c_
            _, pr = np.unique(chiave.numpy(), return_index=True)
            e_, r_, c_ = e_[pr], r_[pr], c_[pr]
            if passo == 1:                                        # ripete i codici del passo 0
                e0, r0, c0 = regs[0].righe["e"], regs[0].righe["r"], regs[0].righe["c"]
                m0 = regs[0].righe["f"] == PV.FAMIGLIE_IDX[n]
                e_ = torch.cat([torch.from_numpy(e0[m0][:30].astype(np.int64)), e_])
                r_ = torch.cat([torch.from_numpy(r0[m0][:30].astype(np.int64)), r_])
                c_ = torch.cat([torch.from_numpy(c0[m0][:30].astype(np.int64)), c_])
                chiave = (e_ * f.R + r_) * f.C + c_
                _, pr = np.unique(chiave.numpy(), return_index=True)
                e_, r_, c_ = e_[np.sort(pr)], r_[np.sort(pr)], c_[np.sort(pr)]
            t_ = T_[e_, r_, c_].to(torch.int32)
            dr = torch.where(t_ == 0, torch.randint(0, 2, t_.shape, generator=g, dtype=torch.int32) * 2 - 1, -t_)
            z = torch.zeros(0)
            parti.append(PV.applica_salti(f, PV.Selezione(e_, r_, c_ // f.gs, c_ % f.gs, dr.to(torch.int8), z, z, z,
                                                           z, z, z, z)))
        regs.append(PV.Registro.unisci(parti))                    # un registro per passo (niente doppioni dentro)
    dst = scratch / "finale.gguf"
    rep_ = PV.esporta_codici(src, dst, regs)
    ric = PV.ricarica_codici(src, dst, regs, [0, 3])
    rotto = PV.Registro(regs[1].righe.copy())
    rotto.righe["t_vecchio"][0] = -rotto.righe["t_vecchio"][0] if rotto.righe["t_vecchio"][0] != 0 else 1
    dst2 = scratch / "finale_rotto.gguf"
    try:
        PV.esporta_codici(src, dst2, [regs[0], rotto])
        manomesso_rifiutato = False
    except (RuntimeError, ValueError):
        manomesso_rifiutato = not dst2.exists() and not dst2.with_name(dst2.name + ".parziale").exists()
    ok = all(v["byte_codici_uguali"] and v["uscita_al_bit"] for v in ric.values()) and manomesso_rifiutato and \
        ric[0]["salti_nello_strato"] > 0 and ric[3]["salti_nello_strato"] == 0
    return registra("T10_export_scale_piu_codici_ricaricato", ok, export=rep_, ricarica=ric,
                    registro_manomesso_rifiutato=manomesso_rifiutato, secondi=round(time.time() - t0, 1))


# ─────────────────────────────── T11 / T12: finestra di accumulo (27/9) ───────────────────────────────
def _corsa_acc(N: int, passi: int, fam_t, fam_s0, token: int, seme: int) -> dict:
    """Solo passo V (theta fermo), decisione ogni N passi con le statistiche sommate sulla finestra. Per decisione:
    salti tenuti, giusti (il codice diventa quello del maestro) e falsi (diverso dal maestro dopo il salto)."""
    fs = PV.famiglia_scrivibile(S.DatiFamiglia(fam_s0.nome, fam_s0.tipo, fam_s0.blocchi.clone(), fam_s0.d0,
                                               fam_s0.strato))
    k = (fs.strato, fs.nome)
    pv = PV.PassoV({k: fs}, PV.ConfigV(passi_accumulo=N))
    theta = torch.zeros_like(fs.d0, requires_grad=True)
    dati = Dati(fs, token=token, k=2, seme=seme)
    reg_f = PV.RegioneFiducia(prova=(0.03,))
    Tt = trit_famiglia(fam_t)
    dec = []
    for t in range(passi):
        raccogli(pv, fs, fam_t, theta, dati, t, con_v=True)
        if pv.stato.passi_nella_finestra == 1:
            pv.registra_scale_finestra({k: scala(fs, theta).detach()})
        if (t + 1) % N:
            continue
        rho = reg_f.rho_passo()
        d_eff = scala(fs, theta).detach()
        xs, ps = dati.lotto(t + 1, 0)
        T0 = trit_famiglia(fs)
        with torch.no_grad():
            prima = float(perdita(fs, d_eff, fam_t, xs, ps, S.MotoreTorch(), ps.S))
        reg, diag = pv.seleziona_e_salta({k: d_eff}, rho)
        with torch.no_grad():
            dopo = float(perdita(fs, d_eff, fam_t, xs, ps, S.MotoreTorch(), ps.S))
        tiene = reg_f.esito(rho, prima, dopo, len(reg))
        if not tiene:
            pv.annulla(reg)
        T1 = trit_famiglia(fs)
        cambiati = T0 != T1
        dec.append({"passo": t + 1, "salti": len(reg), "tenuto": tiene,
                    "giusti": int((cambiati & (T1 == Tt)).sum()), "falsi": int((cambiati & (T1 != Tt)).sum()),
                    "n_verifica": pv.stato.n_verifica, "convenienti": diag[f"{k[0]}.{k[1]}"]["convenienti"]})
        pv.stato.chiudi_finestra()
    return {"decisioni": dec, "sbagliati_fine": _wrong(fs, fam_t), "fam": fs, "pv": pv}


def prova_t11() -> bool:
    """Finestra di accumulo a pari token (16 passi, micro-lotti piccoli = test con poca potenza): N=4 decide 4 volte con
    ~4x micro-lotti di verifica; deve fare piu' salti GIUSTI per decisione e non piu' salti falsi di N=1."""
    t0 = time.time()
    L = 39
    ft = PV.famiglia_scrivibile(famiglia(L, "gate", esperti=slice(0, 4), righe=slice(0, 64)))
    fs0 = PV.famiglia_scrivibile(famiglia(L, "gate", esperti=slice(0, 4), righe=slice(0, 64)))
    pianta_errori(fs0, 0.01, 31)
    sbagliati0 = _wrong(fs0, ft)
    r = {N: _corsa_acc(N, 16, ft, fs0, token=16, seme=11) for N in (1, 4)}
    tot = {N: {"decisioni": len(v["decisioni"]), "giusti": sum(d["giusti"] for d in v["decisioni"]),
               "falsi": sum(d["falsi"] for d in v["decisioni"]), "sbagliati_fine": v["sbagliati_fine"]}
           for N, v in r.items()}
    for v in tot.values():
        v["giusti_per_decisione"] = v["giusti"] / max(v["decisioni"], 1)
        v["precisione"] = v["giusti"] / max(v["giusti"] + v["falsi"], 1)
    ok = (tot[4]["giusti_per_decisione"] > tot[1]["giusti_per_decisione"] and tot[4]["falsi"] <= tot[1]["falsi"]
          and tot[4]["giusti"] > 0 and tot[4]["sbagliati_fine"] < sbagliati0)
    return registra("T11_finestra_di_accumulo_potenza", ok, sbagliati_iniziali=sbagliati0, riassunto=tot,
                    decisioni={N: v["decisioni"] for N, v in r.items()}, secondi=round(time.time() - t0, 1),
                    nota="strato 39 gate, esperti 0-3 righe 0-63; 1% codici piantati; micro-lotti da 32 posti; "
                         "16 passi solo V; rho fisso 0,03")


def prova_t12() -> bool:
    """Ripresa a meta' finestra: 4 passi di fila contro 2 passi -> istantanea -> torch.save/load -> PassoV nuovo con
    `ripristina` -> 2 passi: buffer, conteggi e salti della decisione identici AL BIT."""
    return _ripresa_a_meta_finestra("T12_ripresa_a_meta_finestra_al_bit",
                                    lambda: famiglia(39, "gate", esperti=slice(0, 4), righe=slice(0, 64)))


def famiglia_sintetica(E: int = 4, R: int = 64, C: int = 1024, seme: int = 13) -> S.DatiFamiglia:
    """Famiglia TQ1_0 SINTETICA (pesi gaussiani quantizzati da gguf-py, canonici come llama.cpp): niente GGUF."""
    from gguf import GGMLQuantizationType as GT, quants
    g = np.random.default_rng(seme)
    raw = quants.quantize(g.standard_normal((E * R, C)).astype(np.float32), GT.TQ1_0).reshape(E, R, -1)
    B = torch.from_numpy(np.ascontiguousarray(raw))
    nb = CS.LAYOUT["TQ1_0"][0]
    d0 = CS.scale_da_blocchi(B.reshape(-1, nb), "TQ1_0").to(torch.float32).reshape(E, R, -1)
    return S.DatiFamiglia("gate", "TQ1_0", B, d0, 0)


def prova_t13() -> bool:
    """Come T12 (ripresa a meta' finestra di accumulo AL BIT) su una famiglia TQ1_0 sintetica: non legge /mnt."""
    return _ripresa_a_meta_finestra("T13_ripresa_a_meta_finestra_sintetica_al_bit", famiglia_sintetica)


def _ripresa_a_meta_finestra(nome: str, fam_fn) -> bool:
    t0 = time.time()
    ft = PV.famiglia_scrivibile(fam_fn())
    fs0 = PV.famiglia_scrivibile(fam_fn())
    pianta_errori(fs0, 0.01, 31)
    k = (fs0.strato, fs0.nome)

    def nuova():
        fs = PV.famiglia_scrivibile(S.DatiFamiglia(fs0.nome, fs0.tipo, fs0.blocchi.clone(), fs0.d0, fs0.strato))
        return fs, PV.PassoV({k: fs}, PV.ConfigV(passi_accumulo=4))
    theta = torch.zeros_like(fs0.d0, requires_grad=True)
    dati = Dati(fs0, token=16, k=2, seme=12)

    def passi(pv, fs, da, a):
        for t in range(da, a):
            raccogli(pv, fs, ft, theta, dati, t, con_v=True)
            if pv.stato.passi_nella_finestra == 1:
                pv.registra_scale_finestra({k: scala(fs, theta).detach()})
    fsA, pvA = nuova()
    passi(pvA, fsA, 0, 4)
    fsB, pvB = nuova()
    passi(pvB, fsB, 0, 2)
    p_ist = DIR / "t12_istantanea.pt"
    torch.save(pvB.stato.istantanea(), p_ist)
    fsC, pvC = nuova()
    pvC.stato.ripristina(torch.load(p_ist, map_location="cpu", weights_only=True))
    passi(pvC, fsC, 2, 4)
    bA, bC = pvA.stato.buffer(fsA), pvC.stato.buffer(fsC)
    buffer_uguali = all(torch.equal(getattr(bA, n), getattr(bC, n)) for n in ("cand", "prop", "S", "Q", "H"))
    conteggi = [(s.n_verifica, s.n_propose, s.n_curvatura, s.passi_nella_finestra) for s in (pvA.stato, pvC.stato)]
    d_eff = scala(fs0, theta).detach()
    regA, _ = pvA.seleziona_e_salta({k: d_eff}, 0.03)
    regC, _ = pvC.seleziona_e_salta({k: d_eff}, 0.03)
    salti_uguali = len(regA) > 0 and np.array_equal(regA.righe, regC.righe) and torch.equal(fsA.blocchi, fsC.blocchi)
    ok = buffer_uguali and conteggi[0] == conteggi[1] and salti_uguali
    return registra(nome, ok, buffer_uguali=buffer_uguali, conteggi=conteggi,
                    salti=len(regA), salti_uguali=salti_uguali, secondi=round(time.time() - t0, 1))


# ─────────────────────── 27/9: V-Bop a gruppo e regione di fiducia adattiva ───────────────────────
BASE_PRIMA_PV = "6bcec55"          # e2e_pv_codici.py prima di V-Bop e del rho adattivo


def _pv_prima(base: str = BASE_PRIMA_PV):
    """e2e_pv_codici.py del commit `base` (git show), caricato come modulo a parte (niente /mnt: scrive in DIR)."""
    import importlib.util
    testo = subprocess.run(["git", "-C", str(QUI), "show", f"{base}:thor_v2/e2e_pv_codici.py"], check=True,
                           capture_output=True, text=True).stdout
    nome = f"e2e_pv_codici_{base}"
    DIR.mkdir(parents=True, exist_ok=True)
    p = DIR / f"{nome}.py"
    p.write_text(testo)
    spec = importlib.util.spec_from_file_location(nome, p)
    m = importlib.util.module_from_spec(spec)
    sys.modules[nome] = m
    spec.loader.exec_module(m)
    return m


def _copia_fam(fs0: S.DatiFamiglia) -> S.DatiFamiglia:
    return PV.famiglia_scrivibile(S.DatiFamiglia(fs0.nome, fs0.tipo, fs0.blocchi.clone(), fs0.d0, fs0.strato))


def _corsa_mod(mod, fs0, ft, passi: int, pv_kw: dict, reg_kw: dict, adatt: bool = False, token: int = 16,
               seme: int = 12, fs=None, pv=None, reg_f=None, da: int = 0, cfg_kw=None):
    """Passi V completi (raccolta + curvatura + sonda + esito + annullamento) col modulo `mod` (nuovo o di git).
    `cfg_kw` (27/9, ramo v-senza-bh): campi di ConfigV (None = ConfigV() come prima)."""
    k = (fs0.strato, fs0.nome)
    fs = fs if fs is not None else _copia_fam(fs0)
    pv = pv if pv is not None else mod.PassoV({k: fs}, mod.ConfigV(**(cfg_kw or {})), **pv_kw)
    reg_f = reg_f if reg_f is not None else mod.RegioneFiducia(**reg_kw)
    theta = torch.zeros_like(fs0.d0, requires_grad=True)
    dati = Dati(fs0, token=token, k=2, seme=seme)
    d = scala(fs0, theta).detach()
    out = []
    for t in range(da, passi):
        mot = mod.MotoreTorchV(pv.stato)
        pv.inizia_passo()
        n_tot = 16 * dati.token * dati.k
        for kk in range(16):
            pv.microlotto(kk)
            x, piano = dati.lotto(t, kk)
            perdita(fs, scala(fs, theta), ft, x, piano, mot, n_tot).backward()
        if getattr(pv.stato, "bop", None) is None or pv.stato.serve_curvatura():
            curvatura_mse(pv, fs, theta, dati, t, n_tot, mot=mot)
        pv.fine_raccolta()
        xs, ps = dati.lotto(t + 1, 999, token=128)
        prima = float(perdita(fs, d, ft, xs, ps, S.MotoreTorch(), ps.S))
        rho = reg_f.rho_passo()
        reg, diag = pv.seleziona_e_salta({k: d}, rho)
        dopo = float(perdita(fs, d, ft, xs, ps, S.MotoreTorch(), ps.S))
        gq = sum(v["guadagno_previsto"] for v in diag.values())
        tiene = reg_f.esito(rho, prima, dopo, len(reg), guadagno_previsto=gq) if adatt else \
            reg_f.esito(rho, prima, dopo, len(reg))
        if not tiene:
            pv.annulla(reg)
        out.append({"passo": t + 1, "salti": len(reg), "tenuto": tiene, "rho": rho, "diag": diag})
    return fs, pv, reg_f, out


def prova_t14() -> bool:
    """Default IDENTICO AL BIT al modulo di prima (git 6bcec55): 4 passi V completi (sonda, annullamenti, taratura
    di rho) su una famiglia sintetica con 1% di codici piantati; PassoV() e PassoV(bop=None), RegioneFiducia() e
    RegioneFiducia(adattivo=False): byte, registri, buffer, storia di rho uguali al bit; niente stato V-Bop."""
    t0 = time.time()
    VEC = _pv_prima()
    ft = PV.famiglia_scrivibile(famiglia_sintetica(R=32))
    fs0 = PV.famiglia_scrivibile(famiglia_sintetica(R=32))
    pianta_errori(fs0, 0.01, 31)
    ris = {"prima": _corsa_mod(VEC, fs0, ft, 4, {}, {}),
           "nuovo": _corsa_mod(PV, fs0, ft, 4, {}, {}),
           "nuovo_esplicito": _corsa_mod(PV, fs0, ft, 4, {"bop": None}, {"adattivo": False, "rho_max": 0.1})}
    fp, pp, rp, op = ris["prima"]
    uguali = {}
    for nome in ("nuovo", "nuovo_esplicito"):
        fn, pn, rn, on = ris[nome]
        bp, bn = pp.stato.buffer(fp), pn.stato.buffer(fn)
        uguali[nome] = {
            "byte": torch.equal(fp.blocchi, fn.blocchi),
            "registri": len(pp.registri) == len(pn.registri) and all(
                np.array_equal(a.righe, b.righe) for a, b in zip(pp.registri, pn.registri)),
            "buffer": all(torch.equal(getattr(bp, n), getattr(bn, n)) for n in ("cand", "prop", "S", "Q", "H")),
            "storia_rho": rp.storia == rn.storia and rp.rho == rn.rho,
            "diag": [o["diag"] for o in op] == [o["diag"] for o in on],
            "senza_bop": pn.stato.bop is None and pn.stato.bop_ == {} and pn.stato.byte_bop() == 0}
    salti = [o["salti"] for o in op]
    tenuti = [o["tenuto"] for o in op]
    ok = all(all(v.values()) for v in uguali.values()) and sum(salti) > 0
    return registra("T14_default_identico_al_bit", ok, uguali=uguali, salti=salti, tenuti=tenuti,
                    rho=[o["rho"] for o in op], base=BASE_PRIMA_PV, secondi=round(time.time() - t0, 1))


def _maestro_fuori_griglia(fs, seme: int, q_giusti: float = 0.25, q_trappola: float = 0.25, res_g: float = 0.9,
                           res_t: float = 0.4):
    """Maestro FUORI GRIGLIA (come T7): in un gruppo su 4 un codice e' spostato di 0,9 d verso un livello vicino
    (salto GIUSTO: dopo il salto il residuo e' 0,1 d), in uno su 4 di 0,4 d (TRAPPOLA: saltare porta il residuo a
    0,6 d, peggio); gli altri gruppi sono esatti. Ritorna pesi del maestro (E, R, C), tipo (0/1/2), colonna, verso."""
    E, R, NG, gs = fs.E, fs.R, fs.NG, fs.gs
    g = torch.Generator().manual_seed(seme)
    T = trit_famiglia(fs).view(E, R, NG, gs)
    Wt = torch.stack([S.peso_esperto_torch(fs, e, fs.d0, RIC) for e in range(E)]).view(E, R, NG, gs).clone()
    tipo = torch.zeros(E, R, NG, dtype=torch.int64)
    col = torch.zeros_like(tipo)
    verso = torch.zeros_like(tipo)
    u = torch.rand(E, R, NG, generator=g)
    for e in range(E):
        for r in range(R):
            for gg in range(NG):
                k = 1 if u[e, r, gg] < q_giusti else (2 if u[e, r, gg] < q_giusti + q_trappola else 0)
                if k == 0 or fs.d0[e, r, gg] == 0:
                    continue
                c = int(torch.randint(0, gs, (1,), generator=g))
                t = int(T[e, r, gg, c])
                dz = (1 if bool(torch.rand(1, generator=g) < 0.5) else -1) if t == 0 else -t
                Wt[e, r, gg, c] += (res_g if k == 1 else res_t) * dz * float(fs.d0[e, r, gg])
                tipo[e, r, gg], col[e, r, gg], verso[e, r, gg] = k, c, dz
    return Wt.view(E, R, -1), tipo, col, verso


def _corsa_maestro(modo: str, passi: int, fs0, mae, token: int = 8, seme: int = 11, rho: float = 0.2,
                   adatt: bool = False, tok_curv: int = 256, bop_kw=None, cfg_kw=None, solo_raccolta: bool = False):
    """Passi di SOLO V (theta fermo) contro il maestro fuori griglia; curvatura su `tok_curv` token (riscalata ai
    micro-lotti del passo); sonda su 512 token mai visti. Conta ogni salto: GIUSTO = gruppo giusto, colonna e verso
    del maestro, dal codice di partenza; ogni altro salto e' SBAGLIATO (trappole, gruppi esatti, colonne sbagliate).
    27/9 (ramo v-senza-bh): `cfg_kw` = campi di ConfigV (None = ConfigV() come prima); per passo anche gli sbagliati
    per tipo di gruppo (0 esatto, 1 giusto ma codice/verso sbagliato, 2 trappola), il delta della sonda e se
    l'annullamento rimette i byte al bit; perdita iniziale/finale su 2048 token fissi mai visti (solo misura)."""
    Wt, tipo, col, verso = mae
    fs = _copia_fam(fs0)
    E, R, NG, gs, C = fs.E, fs.R, fs.NG, fs.gs, fs.C
    k = (fs.strato, fs.nome)
    bop = PV.ConfigBop(**(bop_kw or {})) if modo == "vbop" else None
    pv = PV.PassoV({k: fs}, PV.ConfigV(**(cfg_kw or {})), bop=bop)
    reg_f = PV.RegioneFiducia(prova=(rho,), adattivo=adatt)
    d = scala(fs, torch.zeros_like(fs.d0)).detach()
    mot = PV.MotoreTorchV(pv.stato)

    def lotto(passo, kk, T):
        gg = torch.Generator().manual_seed(seme * 100003 + passo * 101 + kk)
        piano = S.Piano(torch.randint(0, E, (T, 2), generator=gg), E)
        return torch.randn(piano.S, C, generator=gg), piano

    def perd(x, piano, mot_, n_tot, dd):
        y = S.FamigliaEsperti.apply(x, dd, fs, piano, RIC, mot_)
        o = piano.offs.tolist()
        yt = torch.cat([x[o[e]:o[e + 1]] @ Wt[e].T for e in range(E)])
        return ((y - yt) ** 2).sum() / n_tot
    T0 = trit_famiglia(fs).clone().view(E, R, NG, gs)
    xh, ph = lotto(10 ** 6, 0, 2048)
    with torch.no_grad():
        perdita_iniziale = float(perd(xh, ph, S.MotoreTorch(), ph.S, d))
    storia, giusti, sbagliati = [], 0, 0
    for t in range(passi):
        pv.inizia_passo()
        n_tot = 16 * token * 2
        th = torch.zeros_like(fs.d0, requires_grad=True)
        for kk in range(16):
            pv.microlotto(kk)
            x, piano = lotto(t, kk, token)
            perd(x, piano, mot, n_tot, scala(fs, th)).backward()
        if pv.stato.serve_curvatura():
            for kk in range(2):
                pv.curvatura()
                x, piano = lotto(t, 1000 + kk, tok_curv)
                y = S.FamigliaEsperti.apply(x, scala(fs, th), fs, piano, RIC, mot)
                gr = torch.Generator().manual_seed(seme * 7919 + t * 31 + kk)
                y.backward((torch.randint(0, 2, y.shape, generator=gr).float() * 2 - 1)
                           * math.sqrt(2.0 / n_tot * token / tok_curv))
        pv.fine_raccolta()
        if solo_raccolta:                         # T20: buffer del primo passo, per selezioni sugli stessi numeri
            return {"pv": pv, "fs": fs, "d": d, "k": k}
        xs, ps = lotto(t + 1, 0, 512)
        with torch.no_grad():
            prima = float(perd(xs, ps, S.MotoreTorch(), ps.S, d))
        Tb = trit_famiglia(fs).clone().view(E, R, NG, gs)
        Bb = fs.blocchi.clone()
        rho_t = reg_f.rho_passo()
        reg, diag = pv.seleziona_e_salta({k: d}, rho_t)
        with torch.no_grad():
            dopo = float(perd(xs, ps, S.MotoreTorch(), ps.S, d))
        gq = sum(v["guadagno_previsto"] for v in diag.values())
        Ta = trit_famiglia(fs).view(E, R, NG, gs)
        g_t = s_t = 0
        s_tipo = [0, 0, 0]
        for a, b, c_, dc in zip(*[x.tolist() for x in torch.nonzero(Ta != Tb, as_tuple=True)]):
            ok = (tipo[a, b, c_] == 1 and col[a, b, c_] == dc and int(Ta[a, b, c_, dc] - Tb[a, b, c_, dc])
                  == verso[a, b, c_] and Tb[a, b, c_, dc] == T0[a, b, c_, dc])
            g_t += int(ok)
            s_t += int(not ok)
            if not ok:
                s_tipo[int(tipo[a, b, c_])] += 1
        tiene = reg_f.esito(rho_t, prima, dopo, len(reg), guadagno_previsto=gq) if adatt else \
            reg_f.esito(rho_t, prima, dopo, len(reg))
        al_bit = None
        if not tiene:
            pv.annulla(reg)
            al_bit = torch.equal(fs.blocchi, Bb)
        else:
            giusti += g_t
            sbagliati += s_t
        storia.append({"passo": t + 1, "salti": len(reg), "giusti": g_t, "sbagliati": s_t, "tenuto": tiene,
                       "rho": rho_t, **({"r": reg_f.storia[-1]["r"], "rho_dopo": reg_f.storia[-1]["rho_dopo"]}
                                        if adatt else {}),
                       "sbagliati_per_tipo": s_tipo, "delta_sonda": dopo - prima, "annullato_al_bit": al_bit,
                       "convenienti": sum(v["convenienti"] for v in diag.values()),
                       "scoperte": sum(v["scoperte"] for v in diag.values())})
    Tf = trit_famiglia(fs).view(E, R, NG, gs)
    fine = Tf != T0
    g_f = sum(int(tipo[a, b, c_] == 1 and col[a, b, c_] == dc and int(Tf[a, b, c_, dc] - T0[a, b, c_, dc])
                  == verso[a, b, c_]) for a, b, c_, dc in zip(*[x.tolist() for x in torch.nonzero(fine, as_tuple=True)]))
    with torch.no_grad():
        perdita_finale = float(perd(xh, ph, S.MotoreTorch(), ph.S, d))
    return {"giusti": giusti, "sbagliati": sbagliati, "cambiati_fine": int(fine.sum()), "giusti_fine": g_f,
            "sbagliati_fine": int(fine.sum()) - g_f, "storia": storia, "reg_f": reg_f, "pv": pv,
            "perdita_iniziale": perdita_iniziale, "perdita_finale": perdita_finale}


def prova_t15() -> bool:
    """V-Bop contro PV a parita' di passi (12 passi di solo V, micro-lotti da 16 posti = test con poca potenza) su
    uno studente finto (famiglia sintetica 4 x 32 x 1024) con maestro FUORI GRIGLIA (salti giusti e trappole):
    V-Bop accumula l'evidenza fra i passi (EMA di m e v sul candidato persistente) e deve trovare PIU' salti giusti
    di PV, con ZERO salti sbagliati (tenuti) e zero codici sbagliati alla fine."""
    t0 = time.time()
    fs0 = PV.famiglia_scrivibile(famiglia_sintetica(E=4, R=32, C=1024, seme=13))
    mae = _maestro_fuori_griglia(fs0, 5)
    r = {m: _corsa_maestro(m, 12, fs0, mae) for m in ("pv", "vbop")}
    rias = {m: {k: v[k] for k in ("giusti", "sbagliati", "cambiati_fine", "giusti_fine", "sbagliati_fine")}
            for m, v in r.items()}
    for m, v in r.items():
        rias[m]["salti_per_passo"] = [s["salti"] for s in v["storia"]]
        rias[m]["annullati"] = sum(not s["tenuto"] for s in v["storia"])
    rias["vbop"]["byte_stato"] = r["vbop"]["pv"].stato.byte_bop()
    ok = (rias["vbop"]["giusti_fine"] > rias["pv"]["giusti_fine"] and rias["vbop"]["giusti"] > rias["pv"]["giusti"]
          and rias["vbop"]["sbagliati"] == 0 and rias["vbop"]["sbagliati_fine"] == 0)
    return registra("T15_vbop_piu_salti_giusti_del_pv", ok, gruppi_giusti=int((mae[1] == 1).sum()),
                    gruppi_trappola=int((mae[1] == 2).sum()), riassunto=rias, secondi=round(time.time() - t0, 1),
                    nota="solo V, 12 passi, 16 micro-lotti x 8 token x top-2, rho fisso 0,2, q 0,05; curvatura su "
                         "256 token (riscalata) per entrambi; residuo giusto 0,9 d, trappola 0,4 d")


def prova_t16() -> bool:
    """Isteresi, correzione alla ECO, azzeramento per non fattibilita', annullamento.
    (a) casi costruiti (gamma 0,5, eta' 3, m^ = 1): proposta nuova 1,5 <= 2|m^| -> candidato TENUTO (nel buffer del
        passo: cand persistente, prop = verso); 2,5 > 2|m^| -> SOSTITUITO (m, v, H, eta' azzerati, verso nuovo);
        senza proposta nuova -> tenuto; senza candidato -> adottato.
    (b) passo vero (1% codici piantati): per ogni gruppo saltato m^ dopo = m^ prima - H|d| sign(m^) (entro fp16) se
        il verso resta fattibile dal trit nuovo, altrimenti stato azzerato; annullamento -> stato V-Bop al bit."""
    t0 = time.time()
    fs = PV.famiglia_scrivibile(famiglia_sintetica(R=32))
    k = (fs.strato, fs.nome)
    st = PV.StatoV(bop=PV.ConfigBop(gamma=0.5))
    st.prepara([fs])
    z, b = st.bop_[k], st.buffer_[k]
    c, _ = PV._correzione_eta(0.5, torch.tensor(3))
    gruppi = {"tiene_1.5": (0, 0, 0), "sostituisce_2.5": (0, 1, 0), "senza_proposta": (0, 2, 0), "adotta": (0, 3, 0)}
    m = torch.zeros(z.m.shape, dtype=torch.float64)
    for nome in ("tiene_1.5", "sostituisce_2.5", "senza_proposta"):
        g = gruppi[nome]
        z.cand[g], z.eta[g] = 5, 3
        m[g] = float(c)                                       # m^ = m / c = 1
    q, s = PV._codifica16(m)
    z.m.copy_(q)
    z.sm.copy_(s)
    z.r.fill_(0.5)
    z.H.fill_(0.25)
    st.inizia_passo()
    st.microlotto(0)
    st.microlotto(1)
    for nome, prop in (("tiene_1.5", 1.5), ("sostituisce_2.5", -2.5), ("adotta", 0.3)):
        b.cand[gruppi[nome]], b.prop[gruppi[nome]] = 9, prop
    st.microlotto(2)                                          # prima verifica: fusione dei candidati
    v = lambda g: {"z_cand": int(z.cand[g]), "eta": int(z.eta[g]), "m": float(z.m[g]),  # noqa: E731
                   "r": float(z.r[g]), "b_cand": int(b.cand[g]), "b_prop": float(b.prop[g])}
    casi = {n: v(g) for n, g in gruppi.items()}
    ok_a = (casi["tiene_1.5"] == {"z_cand": 5, "eta": 3, "m": float(z.m[gruppi["tiene_1.5"]]), "r": 0.5,
                                  "b_cand": 5, "b_prop": 1.0} and casi["tiene_1.5"]["m"] != 0
            and casi["sostituisce_2.5"] == {"z_cand": 9, "eta": 128, "m": 0.0, "r": 0.0, "b_cand": 9, "b_prop": -2.5}
            and casi["senza_proposta"]["b_cand"] == 5 and casi["senza_proposta"]["b_prop"] == 1.0
            and casi["adotta"] == {"z_cand": 9, "eta": 0, "m": 0.0, "r": 0.0, "b_cand": 9, "b_prop": float(torch.tensor(0.3))}
            and int(((b.prop != 0)).sum()) == 4)

    # (b) passo vero
    ft = PV.famiglia_scrivibile(famiglia_sintetica(R=32))
    fs0 = PV.famiglia_scrivibile(famiglia_sintetica(R=32))
    pianta_errori(fs0, 0.01, 31)
    fs1 = _copia_fam(fs0)
    pv = PV.PassoV({k: fs1}, PV.ConfigV(), bop=PV.ConfigBop())
    theta = torch.zeros_like(fs0.d0, requires_grad=True)
    dati = Dati(fs0, token=16, k=2, seme=12)
    raccogli(pv, fs1, ft, theta, dati, 0)
    foto = {}
    orig = pv.stato.aggiorna_bop

    def aggiorna_e_fotografa(*a, **kw):
        orig(*a, **kw)
        foto.update({n: getattr(pv.stato.bop_[k], n).clone() for n in PV._CAMPI_BOP})
    pv.stato.aggiorna_bop = aggiorna_e_fotografa
    d = scala(fs0, theta).detach()
    Tb = trit_famiglia(fs1).view(fs.E, fs.R, fs.NG, fs.gs).clone()
    reg, diag = pv.seleziona_e_salta({k: d}, 0.1)
    zb = pv.stato.bop_[k]
    Ta = trit_famiglia(fs1).view(fs.E, fs.R, fs.NG, fs.gs)
    n_eco = n_azz = 0
    err_eco = 0.0
    ok_b = len(reg) > 0
    for e, r_, g, cc in zip(*[x.tolist() for x in torch.nonzero(Ta != Tb, as_tuple=True)]):
        tn, dz = int(Ta[e, r_, g, cc]), int(Ta[e, r_, g, cc] - Tb[e, r_, g, cc])
        if abs(tn + dz) > 1:
            n_azz += 1
            ok_b &= int(zb.eta[e, r_, g]) == 0 and float(zb.m[e, r_, g]) == 0 and float(zb.r[e, r_, g]) == 0
        else:
            n_eco += 1
            eta = int(foto["eta"][e, r_, g]) & PV.ETA_MAX
            cg = float(PV._correzione_eta(0.5, torch.tensor(eta))[0])
            m0 = float(foto["m"][e, r_, g]) * float(foto["sm"][e]) / cg
            H0 = float(foto["H"][e, r_, g]) * float(foto["sH"][e])
            atteso = m0 - H0 * abs(float(d[e, r_, g])) * math.copysign(1.0, m0)
            m1 = float(zb.m[e, r_, g]) * float(zb.sm[e]) / cg
            err_eco = max(err_eco, abs(m1 - atteso) / max(abs(atteso), abs(m0)))
            ok_b &= int(zb.eta[e, r_, g]) == int(foto["eta"][e, r_, g]) and \
                float(zb.r[e, r_, g]) == float(foto["r"][e, r_, g])
    ok_b &= err_eco < 2e-3
    pv.annulla(reg)
    ripristinato = all(torch.equal(getattr(zb, n), foto[n]) for n in PV._CAMPI_BOP) and torch.equal(fs1.blocchi,
                                                                                                    fs0.blocchi)
    ok = ok_a and ok_b and ripristinato
    return registra("T16_isteresi_eco_annullamento", ok, casi=casi, ok_casi=ok_a, salti=len(reg), eco=n_eco,
                    azzerati_non_fattibili=n_azz, err_rel_eco=err_eco, annullamento_al_bit=ripristinato,
                    secondi=round(time.time() - t0, 1))


def prova_t17() -> bool:
    """Ripresa AL BIT a meta' stato V-Bop (gamma 0,5, curvatura ogni 2 passi, rho adattivo): 5 passi di fila contro
    3 passi -> istantanea_bop + registri + regione -> torch.save/load -> famiglia di partenza + rigioco + PassoV
    nuovo con `ripristina_bop` -> 2 passi: byte, registri, stato V-Bop (8 campi) e storia di rho identici."""
    t0 = time.time()
    ft = PV.famiglia_scrivibile(famiglia_sintetica(R=32))
    fs0 = PV.famiglia_scrivibile(famiglia_sintetica(R=32))
    pianta_errori(fs0, 0.01, 31)
    k = (fs0.strato, fs0.nome)
    cfg = dict(gamma=0.5, curvatura_ogni=2)
    reg_kw = dict(prova=(0.03,), adattivo=True, rho_max=0.1)
    fA, pA, rA, oA = _corsa_mod(PV, fs0, ft, 5, {"bop": PV.ConfigBop(**cfg)}, reg_kw, adatt=True)
    fB, pB, rB, oB = _corsa_mod(PV, fs0, ft, 3, {"bop": PV.ConfigBop(**cfg)}, reg_kw, adatt=True)
    p = DIR / "t17_ripresa.pt"
    torch.save({"bop": pB.stato.istantanea_bop(), "rho": rB.rho, "storia": rB.storia}, p)
    sha = PV.Registro.salva_lista(DIR / "t17_registri.pt", pB.registri)
    ist = torch.load(p, map_location="cpu", weights_only=False)
    fC = _copia_fam(fs0)
    pC = PV.PassoV({k: fC}, PV.ConfigV(), bop=PV.ConfigBop(**cfg))
    regs = PV.Registro.carica_lista(DIR / "t17_registri.pt", sha)
    PV.rigioca({k: fC}, regs)
    pC.registri = list(regs)
    pC.stato.ripristina_bop(ist["bop"])
    rC = PV.RegioneFiducia(**reg_kw)
    rC.rho, rC.storia = ist["rho"], list(ist["storia"])
    fC, pC, rC, oC = _corsa_mod(PV, fs0, ft, 5, {}, {}, adatt=True, fs=fC, pv=pC, reg_f=rC, da=3)
    zA, zC = pA.stato.bop_[k], pC.stato.bop_[k]
    uguali = {"byte": torch.equal(fA.blocchi, fC.blocchi),
              "registri": len(pA.registri) == len(pC.registri) and all(
                  np.array_equal(a.righe, b.righe) for a, b in zip(pA.registri, pC.registri)),
              "stato_bop": all(torch.equal(getattr(zA, n), getattr(zC, n)) for n in PV._CAMPI_BOP),
              "passi_bop": pA.stato.passi_bop == pC.stato.passi_bop,
              "storia_rho": rA.storia == rC.storia and rA.rho == rC.rho}
    curv = [pA.stato.passi_bop]
    ok = all(uguali.values()) and sum(o["salti"] for o in oA[3:]) > 0 and any(o["salti"] for o in oA[:3])
    return registra("T17_ripresa_vbop_a_meta_stato_al_bit", ok, uguali=uguali,
                    salti=[o["salti"] for o in oA], tenuti=[o["tenuto"] for o in oA],
                    rho=[o["rho"] for o in oA], passi_bop=curv, byte_stato=pA.stato.byte_bop(),
                    byte_per_gruppo=pA.stato.byte_bop() / (fs0.E * fs0.R * fs0.NG), secondi=round(time.time() - t0, 1))


def prova_t18() -> bool:
    """Regione di fiducia vera per rho (adattivo). (a) casi costruiti: r = calo effettivo / calo previsto; r > 0,75
    -> x2, 0,25 <= r <= 0,75 -> invariato, r < 0,25 -> /2; tetto rho_max, pavimento rho_min; senza salti invariato;
    sonda che sale -> annullato e /2; con taratura (3 rho di prova) si adatta solo dopo; spento = regola di prima.
    (b) nel passo V vero contro il maestro fuori griglia (V-Bop, rho iniziale 0,05): ogni rho_dopo segue la regola."""
    t0 = time.time()
    R = PV.RegioneFiducia(prova=(0.01,), adattivo=True, rho_max=0.1, rho_min=1e-3)
    casi = [  # (prima, dopo, salti, guadagno previsto, rho atteso dopo, tenuto atteso)
        (1.0, 0.99, 5, 0.01, 0.02, True),        # r = 1   -> x2
        (1.0, 0.995, 5, 0.01, 0.02, True),       # r = 0,5 -> invariato
        (1.0, 0.999, 5, 0.01, 0.01, True),       # r = 0,1 -> /2
        (1.0, 0.992, 5, 0.01, 0.02, True),       # r = 0,8 -> x2
        (1.0, 0.99, 5, 0.01, 0.04, True),
        (1.0, 0.99, 5, 0.01, 0.08, True),
        (1.0, 0.99, 5, 0.01, 0.1, True),         # tetto
        (1.0, 0.99, 5, 0.01, 0.1, True),
        (1.0, 1.0, 0, 0.0, 0.1, True),           # niente salti: invariato
        (1.0, 1.01, 5, 0.01, 0.05, False),       # sonda sale: annullato, r = -1 -> /2
        (1.0, 1.0025, 5, 0.01, 0.025, False),
        (1.0, 0.997, 5, 0.01, 0.025, True),      # r = 0,3 -> invariato
        (1.0, 0.99, 5, 0.01, 0.05, True),
    ]
    trovati, ok_a = [], True
    for prima, dopo, n, gq, atteso, t_att in casi:
        rho = R.rho_passo()
        tiene = R.esito(rho, prima, dopo, n, guadagno_previsto=gq)
        trovati.append((R.storia[-1]["r"], R.rho, tiene))
        ok_a &= math.isclose(R.rho, atteso, rel_tol=1e-12) and tiene == t_att and R.storia[-1]["rho_dopo"] == R.rho
    Rp = PV.RegioneFiducia(prova=(0.004,), adattivo=True, rho_min=1e-3)
    for _ in range(4):
        Rp.esito(Rp.rho_passo(), 1.0, 1.001, 3, guadagno_previsto=0.01)
    pavimento = Rp.rho == 1e-3
    Rt = PV.RegioneFiducia(adattivo=True)                      # taratura (0,01 0,03 0,1) poi adattivo
    for dd in (-0.01, -0.03, -0.02):
        Rt.esito(Rt.rho_passo(), 1.0, 1.0 + dd, 4, guadagno_previsto=0.02)
    tar = Rt.rho == 0.03 and "rho_dopo" not in Rt.storia[-1]
    Rt.esito(Rt.rho_passo(), 1.0, 0.98, 4, guadagno_previsto=0.02)
    tar &= Rt.rho == 0.06
    Rs = PV.RegioneFiducia(prova=(0.01,))                      # spento: regola di prima
    Rs.esito(0.01, 1.0, 0.9, 5)
    Rs.esito(0.01, 1.0, 0.9, 5, guadagno_previsto=0.001)
    spento = Rs.rho == 0.01 and "r" not in Rs.storia[-1]
    Rs.esito(0.01, 1.0, 1.1, 5)
    spento &= Rs.rho == 0.005
    ok_a &= pavimento and tar and spento

    fs0 = PV.famiglia_scrivibile(famiglia_sintetica(E=4, R=32, C=1024, seme=13))
    mae = _maestro_fuori_griglia(fs0, 5)
    rv = _corsa_maestro("vbop", 10, fs0, mae, rho=0.05, adatt=True)
    regola, visti = True, set()
    for s in rv["storia"]:
        r, rho, rd = s["r"], s["rho"], s["rho_dopo"]
        if r is None:
            att = rho if s["tenuto"] else max(1e-4, rho / 2)
        elif r > 0.75:
            att, _ = max(rho, min(0.1, 2 * rho)), visti.add("x2")
        elif r < 0.25:
            att, _ = max(1e-4, rho / 2), visti.add("/2")
        else:
            att, _ = rho, visti.add("=")
        regola &= math.isclose(rd, att, rel_tol=1e-12)
    ok_b = regola and "x2" in visti and rv["sbagliati"] == 0
    ok = ok_a and ok_b
    return registra("T18_regione_di_fiducia_adattiva", ok, casi_costruiti=trovati, pavimento=pavimento,
                    taratura_poi_adattivo=tar, spento_come_prima=spento,
                    passo_vero=[(s["passo"], s["salti"], s["tenuto"], None if s["r"] is None else round(s["r"], 3),
                                 s["rho"], s["rho_dopo"]) for s in rv["storia"]],
                    regole_viste=sorted(visti), giusti=rv["giusti"], sbagliati=rv["sbagliati"],
                    secondi=round(time.time() - t0, 1))


# ─────────────────── 27/9 ramo v-senza-bh: --pv-accetta guadagno (senza i due test BH) ───────────────────
BASE_PRIMA_ACCETTA = "767b2a7"     # master prima di --pv-accetta (e2e_pv_codici.py del merge 413d518)
_CFG_ACCETTA = {"bh": {"accetta": "bh"}, "guadagno": {"accetta": "guadagno"},
                "guadagno_z2": {"accetta": "guadagno", "z_min": 2.0},
                "guadagno_zg1": {"accetta": "guadagno", "z_guadagno_min": 1.0},
                "guadagno_zg2": {"accetta": "guadagno", "z_guadagno_min": 2.0}}
# scenari del maestro fuori griglia: quota di gruppi con salto giusto (0,9 d) e di trappole (0,4 d)
_SCENARI_MAESTRO = {"base": (0.25, 0.25), "raro": (0.03, 0.25), "nullo": (0.0, 0.25)}


def _uguali_corse(a, b) -> dict:
    """Confronto al bit di due corse di `_corsa_mod` (fam, PassoV, RegioneFiducia, esiti)."""
    fa, pa, ra, oa = a
    fb, pb, rb, ob = b
    ba, bb = pa.stato.buffer(fa), pb.stato.buffer(fb)
    u = {"byte": torch.equal(fa.blocchi, fb.blocchi),
         "registri": len(pa.registri) == len(pb.registri) and all(
             np.array_equal(x.righe, y.righe) for x, y in zip(pa.registri, pb.registri)),
         "buffer": all(torch.equal(getattr(ba, n), getattr(bb, n)) for n in ("cand", "prop", "S", "Q", "H")),
         "storia_rho": ra.storia == rb.storia and ra.rho == rb.rho,
         "diag": [o["diag"] for o in oa] == [o["diag"] for o in ob]}
    if pa.stato.bop is not None or pb.stato.bop is not None:
        u["stato_bop"] = (pa.stato.bop is not None and pb.stato.bop is not None and set(pa.stato.bop_) == set(pb.stato.bop_)
                          and all(torch.equal(getattr(pa.stato.bop_[k], c), getattr(pb.stato.bop_[k], c))
                                  for k in pa.stato.bop_ for c in PV._CAMPI_BOP)
                          and pa.stato.passi_bop == pb.stato.passi_bop)
    return u


def prova_t19() -> bool:
    """Default IDENTICO AL BIT al modulo di master prima di --pv-accetta (git 767b2a7), sulla famiglia sintetica con
    1% di codici piantati (come T14): (a) PV, 4 passi V completi; (b) V-Bop con curvatura ogni 2 e rho adattivo, 5
    passi; ciascuno col modulo di prima, col nuovo a ConfigV() e col nuovo a ConfigV(accetta="bh", z_min=0): byte,
    registri, buffer, storia di rho, diagnostica e stato V-Bop uguali al bit. (c) Configurazioni rifiutate: accetta
    sconosciuto, z_min > 0 con bh, z_min < 0."""
    t0 = time.time()
    VEC = _pv_prima(BASE_PRIMA_ACCETTA)
    ft = PV.famiglia_scrivibile(famiglia_sintetica(R=32))
    fs0 = PV.famiglia_scrivibile(famiglia_sintetica(R=32))
    pianta_errori(fs0, 0.01, 31)
    casi = {"pv": (4, lambda m: {}, {}, False),
            "vbop_curv2_rho_adattivo": (5, lambda m: {"bop": m.ConfigBop(gamma=0.5, curvatura_ogni=2)},
                                        {"prova": (0.03,), "adattivo": True, "rho_max": 0.1}, True)}
    uguali, salti = {}, {}
    for nome, (passi, pv_kw, reg_kw, adatt) in casi.items():
        prima = _corsa_mod(VEC, fs0, ft, passi, pv_kw(VEC), reg_kw, adatt=adatt)
        for v, ckw in (("nuovo", None), ("nuovo_esplicito", {"accetta": "bh", "z_min": 0.0})):
            uguali[f"{nome}/{v}"] = _uguali_corse(prima, _corsa_mod(PV, fs0, ft, passi, pv_kw(PV), reg_kw,
                                                                    adatt=adatt, cfg_kw=ckw))
        salti[nome] = [o["salti"] for o in prima[3]]
    rifiuti = []
    for kw in ({"accetta": "gain"}, {"z_min": 1.0}, {"accetta": "guadagno", "z_min": -1.0}):
        try:
            PV.ConfigV(**kw)
            rifiuti.append(False)
        except ValueError:
            rifiuti.append(True)
    ok = (all(all(v.values()) for v in uguali.values()) and all(sum(s) > 0 for s in salti.values()) and all(rifiuti)
          and PV.ConfigV() == PV.ConfigV(accetta="bh", z_min=0.0))
    return registra("T19_accetta_default_identico_al_bit", ok, uguali=uguali, salti=salti, rifiuti=rifiuti,
                    base=BASE_PRIMA_ACCETTA, secondi=round(time.time() - t0, 1))


def _riassumi_maestro(r: dict) -> dict:
    st = r["storia"]
    ann = [s for s in st if not s["tenuto"]]
    return {"proposti_giusti": sum(s["giusti"] for s in st), "proposti_sbagliati": sum(s["sbagliati"] for s in st),
            "proposti_sbagliati_per_tipo": [sum(s["sbagliati_per_tipo"][i] for s in st) for i in range(3)],
            "tenuti_giusti": r["giusti"], "tenuti_sbagliati": r["sbagliati"],
            "passi_annullati": len(ann), "giusti_persi_per_annullamento": sum(s["giusti"] for s in ann),
            "sbagliati_tolti_dalla_sonda": sum(s["sbagliati"] for s in ann),
            "giusti_fine": r["giusti_fine"], "sbagliati_fine": r["sbagliati_fine"],
            "perdita_fine_su_inizio": r["perdita_finale"] / r["perdita_iniziale"],
            "rho_finale": r["reg_f"].rho, "convenienti": sum(s["convenienti"] for s in st),
            "annullamenti_al_bit": all(s["annullato_al_bit"] for s in ann)}


def _somma_riassunti(rr: list) -> dict:
    out = {}
    for k in rr[0]:
        v = [x[k] for x in rr]
        if k == "proposti_sbagliati_per_tipo":
            out[k] = [sum(x[i] for x in v) for i in range(3)]
        elif k == "annullamenti_al_bit":
            out[k] = all(v)
        elif k in ("perdita_fine_su_inizio", "rho_finale"):
            out[k] = [round(x, 6) for x in v]
        else:
            out[k] = sum(v)
    return out


def prova_t20() -> bool:
    """--pv-accetta guadagno contro bh A PARITA' DI PASSI col maestro FUORI GRIGLIA (come T15/T18: famiglia sintetica
    4 x 32 x 1024; salto giusto a 0,9 d, trappola a 0,4 d; 12 passi di SOLO V, 16 micro-lotti x 8 token x top-2,
    curvatura su 256 token). Tre scenari: base (25% giusti, 25% trappole, come T15), raro (3% giusti, 25% trappole),
    nullo (nessun giusto, 25% trappole). MISURA, per PV e V-Bop, a rho 0,2 (budget largo, come T15) e 0,03 (budget
    stretto come EB2), su 3 maestri (semi 5, 6, 7): salti GIUSTI e SBAGLIATI proposti (prima della sonda; sbagliati
    per tipo di gruppo: esatto / giusto ma codice o verso sbagliato / trappola) e tenuti, passi annullati dalla sonda
    (giusti persi, sbagliati tolti), codici giusti/sbagliati alla fine, perdita su 2048 token fissi mai visti (fine /
    inizio), rho finale; varianti bh, guadagno, guadagno con z_min 2 (direzione), con z_guadagno_min 1 e 2 (limite
    inferiore del guadagno); piu' V-Bop con rho adattivo (0,05 come T18) per bh e guadagno nello scenario base; piu'
    PV a rho 0,03 con curvatura POVERA (16 token, ~8 per esperto come in A33 sullo strato vero) in tutti gli scenari.
    VERDETTO solo sui controlli: (a) sugli STESSI buffer (primo passo) convenienti annidati bh ⊆ zg1 ⊆ guadagno,
    zg2 ⊆ zg1, z2 ⊆ guadagno (a budget illimitato i scelti sono i convenienti: inclusione degli insiemi), costo <=
    budget per esperto in ogni variante a rho 0,2 e 0,03; (b) ogni annullamento rimette i byte al bit; (c) bh
    esplicito == default (== T15) sugli esiti."""
    t0 = time.time()
    semi = (5, 6, 7)
    fam0 = PV.famiglia_scrivibile(famiglia_sintetica(E=4, R=32, C=1024, seme=13))
    maestri = {(sc, s): _maestro_fuori_griglia(fam0, s, q_giusti=qg, q_trappola=qt)
               for sc, (qg, qt) in _SCENARI_MAESTRO.items() for s in semi}
    # (a) stessi buffer
    ok_a, dett_a = True, {}
    for (sc, s), mae in maestri.items():
        for modo in ("pv", "vbop"):
            rc = _corsa_maestro(modo, 1, fam0, mae, seme=11 + s - 5, solo_raccolta=True)
            pv, d, k = rc["pv"], rc["d"], rc["k"]
            if modo == "vbop":
                pv.stato.aggiorna_bop()
            ins, conv, cb = {}, {}, True
            for nome, kw in _CFG_ACCETTA.items():
                pv.cfg = PV.ConfigV(**kw)
                sl = pv.seleziona({k: d}, 1e3)[k]
                ins[nome] = set(zip(sl.e.tolist(), sl.r.tolist(), sl.g.tolist()))
                conv[nome] = int(sl.convenienti.sum())
                cb &= len(sl) == conv[nome]
                for rho in (0.2, 0.03):
                    sr = pv.seleziona({k: d}, rho)[k]
                    cb &= bool((sr.costo <= sr.budget * (1 + 1e-12)).all()) and int(sr.scelti.sum()) == len(sr)
            ann = (ins["bh"] <= ins["guadagno_zg1"] <= ins["guadagno"] and ins["guadagno_zg2"] <= ins["guadagno_zg1"]
                   and ins["guadagno_z2"] <= ins["guadagno"])
            ok_a &= ann and cb
            dett_a[f"{sc}/{modo}/{s}"] = {"convenienti": conv, "annidati": ann, "budget_e_scelti": cb}
    # (b) corse
    ris, per_seme = {}, {}
    varianti = [(sc, modo, rho, nome, kw, False, 256) for sc in _SCENARI_MAESTRO for modo in ("pv", "vbop")
                for rho in (0.2, 0.03) for nome, kw in _CFG_ACCETTA.items()]
    varianti += [("base", "vbop", 0.05, f"{nome}+rho_adattivo", _CFG_ACCETTA[nome], True, 256)
                 for nome in ("bh", "guadagno")]
    # curvatura POVERA: 16 token (~8 per esperto con top-2 su 4, come i ~8 per esperto di A33 sullo strato vero)
    varianti += [(sc, "pv", 0.03, f"{nome}+curv16", _CFG_ACCETTA[nome], False, 16) for sc in _SCENARI_MAESTRO
                 for nome in ("bh", "guadagno", "guadagno_zg1", "guadagno_zg2")]
    for sc, modo, rho, nome, kw, adatt, tc in varianti:
        rr = []
        for s in semi:
            r = _corsa_maestro(modo, 12, fam0, maestri[(sc, s)], seme=11 + s - 5, rho=rho, adatt=adatt, cfg_kw=kw,
                               tok_curv=tc)
            rr.append(_riassumi_maestro(r))
            per_seme[f"{sc}/{modo}/rho{rho}/{nome}/s{s}"] = {k_: rr[-1][k_] for k_ in (
                "proposti_giusti", "proposti_sbagliati", "tenuti_giusti", "tenuti_sbagliati", "passi_annullati",
                "giusti_fine", "sbagliati_fine", "perdita_fine_su_inizio")}
        ris[f"{sc}/{modo}/rho{rho}/{nome}"] = _somma_riassunti(rr)
    ok_b = all(v["annullamenti_al_bit"] for v in ris.values())
    # (c) bh esplicito == default (T15: seme 5, scenario base)
    ok_c = True
    for modo in ("pv", "vbop"):
        rd = _corsa_maestro(modo, 12, fam0, maestri[("base", 5)])
        rx = _corsa_maestro(modo, 12, fam0, maestri[("base", 5)], cfg_kw={"accetta": "bh"})
        ok_c &= (rd["storia"] == rx["storia"] and rd["giusti_fine"] == rx["giusti_fine"]
                 and rd["perdita_finale"] == rx["perdita_finale"])
    gruppi = {f"{sc}/{s}": (int((m[1] == 1).sum()), int((m[1] == 2).sum())) for (sc, s), m in maestri.items()}
    for chiave, v in ris.items():
        print(f"   {chiave:40s} prop g/s {v['proposti_giusti']:4d}/{v['proposti_sbagliati']:4d} "
              f"(tipi {v['proposti_sbagliati_per_tipo']})  tenuti g/s {v['tenuti_giusti']:4d}/{v['tenuti_sbagliati']:4d}"
              f"  annullati {v['passi_annullati']:2d} (persi g {v['giusti_persi_per_annullamento']}, tolti s "
              f"{v['sbagliati_tolti_dalla_sonda']})  fine g/s {v['giusti_fine']}/{v['sbagliati_fine']}  "
              f"perdita {v['perdita_fine_su_inizio']}", flush=True)
    ok = ok_a and ok_b and ok_c
    return registra("T20_accetta_guadagno_contro_bh_maestro_fuori_griglia", ok, a=ok_a, b_annullamenti_al_bit=ok_b,
                    c_bh_esplicito_uguale_default=ok_c, gruppi_giusti_trappola=gruppi, riassunto=ris,
                    a_stessi_buffer=dett_a, per_seme=per_seme, secondi=round(time.time() - t0, 1),
                    nota="solo V, 12 passi, 16 micro-lotti x 8 token x top-2; curvatura 256 token; residuo giusto "
                         "0,9 d, trappola 0,4 d; somme su 3 maestri (semi 5,6,7)")


def prova_t21() -> bool:
    """Ripresa AL BIT con --pv-accetta guadagno acceso (come T17): 5 passi di fila contro 3 passi -> registri +
    regione (+ istantanea V-Bop) -> torch.save/load -> famiglia di partenza + rigioco + PassoV NUOVO con la stessa
    ConfigV -> 2 passi: byte, registri, storia di rho (e stato V-Bop) identici. Due varianti: PV con guadagno;
    V-Bop (gamma 0,5, curvatura ogni 2, rho adattivo) con guadagno e z_min 1. Salti in entrambe le meta'."""
    t0 = time.time()
    ft = PV.famiglia_scrivibile(famiglia_sintetica(R=32))
    fs0 = PV.famiglia_scrivibile(famiglia_sintetica(R=32))
    pianta_errori(fs0, 0.01, 31)
    k = (fs0.strato, fs0.nome)
    varianti = {"pv_guadagno": ({"accetta": "guadagno"}, None, dict(prova=(0.03,)), False),
                "vbop_guadagno_z1_curv2_rho_adattivo": ({"accetta": "guadagno", "z_min": 1.0},
                                                        dict(gamma=0.5, curvatura_ogni=2),
                                                        dict(prova=(0.03,), adattivo=True, rho_max=0.1), True)}
    esiti, ok = {}, True
    for nome, (ckw, bkw, reg_kw, adatt) in varianti.items():
        pv_kw = (lambda: {"bop": PV.ConfigBop(**bkw)}) if bkw else (lambda: {})
        A_ = _corsa_mod(PV, fs0, ft, 5, pv_kw(), reg_kw, adatt=adatt, cfg_kw=ckw)
        fB, pB, rB, oB = _corsa_mod(PV, fs0, ft, 3, pv_kw(), reg_kw, adatt=adatt, cfg_kw=ckw)
        p = DIR / f"t21_{nome}.pt"
        torch.save({"bop": pB.stato.istantanea_bop() if bkw else None, "rho": rB.rho, "storia": rB.storia}, p)
        sha = PV.Registro.salva_lista(DIR / f"t21_{nome}_registri.pt", pB.registri)
        ist = torch.load(p, map_location="cpu", weights_only=False)
        fC = _copia_fam(fs0)
        pC = PV.PassoV({k: fC}, PV.ConfigV(**ckw), **pv_kw())
        regs = PV.Registro.carica_lista(DIR / f"t21_{nome}_registri.pt", sha)
        PV.rigioca({k: fC}, regs)
        pC.registri = list(regs)
        if bkw:
            pC.stato.ripristina_bop(ist["bop"])
        rC = PV.RegioneFiducia(**reg_kw)
        rC.rho, rC.storia = ist["rho"], list(ist["storia"])
        C_ = _corsa_mod(PV, fs0, ft, 5, {}, {}, adatt=adatt, fs=fC, pv=pC, reg_f=rC, da=3)
        u = _uguali_corse(A_, C_)
        u.pop("diag")          # gli esiti della meta' ripresa sono solo i passi 4-5
        u["diag_passi_4_5"] = [o["diag"] for o in A_[3][3:]] == [o["diag"] for o in C_[3]]
        salti = [o["salti"] for o in A_[3]]
        e_ok = all(u.values()) and sum(salti[3:]) > 0 and sum(salti[:3]) > 0 and pC.cfg.accetta == "guadagno"
        # stessa famiglia, stessi dati, bh: quanti salti in meno (misura)
        bh_ = _corsa_mod(PV, fs0, ft, 5, pv_kw(), reg_kw, adatt=adatt)
        esiti[nome] = {"uguali": u, "salti": salti, "tenuti": [o["tenuto"] for o in A_[3]],
                       "rho": [o["rho"] for o in A_[3]], "salti_con_bh": [o["salti"] for o in bh_[3]], "ok": e_ok}
        ok &= e_ok
    return registra("T21_ripresa_con_accetta_guadagno_al_bit", ok, varianti=esiti, secondi=round(time.time() - t0, 1))


# ─────────────────────────────── main ───────────────────────────────
PROVE = {"T1": prova_t1, "T2": prova_t2, "T3": prova_t3, "T4": prova_t4, "T5": prova_t5, "T6": prova_t6,
         "T7": prova_t7, "T8": prova_t8, "T9": prova_t9, "T10": prova_t10, "T11": prova_t11, "T12": prova_t12,
         "T13": prova_t13, "T14": prova_t14, "T15": prova_t15, "T16": prova_t16, "T17": prova_t17,
         "T18": prova_t18, "T19": prova_t19, "T20": prova_t20, "T21": prova_t21}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo", default=",".join(PROVE))
    a = ap.parse_args()
    DIR.mkdir(parents=True, exist_ok=True)
    esito_p = DIR / "esito_test_pv_codici.json"
    if esito_p.exists():
        try:
            ESITO["prove"].update(json.loads(esito_p.read_text()).get("prove", {}))
        except Exception as ex:  # noqa: BLE001
            print(f"esito precedente illeggibile ({ex}): si riparte", flush=True)
    ok = True
    for nome in a.solo.split(","):
        for k in [k for k in ESITO["prove"] if k.startswith(nome + "_")]:
            del ESITO["prove"][k]                                   # esiti vecchi della stessa prova
        t0 = time.time()
        try:
            ok &= PROVE[nome]()
        except Exception as ex:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            ok &= registra(f"{nome}_eccezione", False, errore=repr(ex))
        print(f"  ({nome}: {time.time() - t0:.1f} s)", flush=True)
        esito_p.write_text(json.dumps(ESITO, indent=1, default=str))
    tutte = all(v["ok"] for k, v in ESITO["prove"].items() if not k.endswith("_eccezione")) and \
        not any(k.endswith("_eccezione") for k in ESITO["prove"])
    print("ESITO:", "TUTTO VERDE" if (ok and tutte) else "⛔ QUALCOSA NON TORNA", "->", esito_p, flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
