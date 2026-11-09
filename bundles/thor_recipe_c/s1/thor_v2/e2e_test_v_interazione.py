#!/usr/bin/env python3
"""Prove del ramo `v-interazione-b` (27/9/2026): sonda a DOSI con conferma (`--pv-sonda-dosi`) e V RICENTRATO
(`--pv-ricentra`) del passo V sui codici (e2e_pv_codici.PassoV.seleziona_e_prova_dosi; TECNICHE_E2E_PV.md §31, §32).
SOLO CPU (la GPU e' di EB2), niente @@ROOT:lavoro@@/e2e/allena: i file vanno in $E2E_VI_DIR.

  V1  default al bit, modulo PV: PassoV/RegioneFiducia del ramo == modulo di `integrata` (git 39d514a) su famiglia
      sintetica (1% dei codici piantati): PV 4 passi; V-Bop curvatura 2 + rho adattivo 5 passi; V-Bop + accetta guadagno
      t_g > 1: byte, registri, buffer, storia di rho, diagnostica, stato V-Bop uguali al bit; `applica_salti` ==
      `righe_salti` + `salta_registro`; conteggi delle dosi; configurazioni rifiutate.
  V2  GIOCATTOLO CON INTERAZIONE (salti che da soli migliorano e insieme peggiorano). Famiglia sintetica TQ1_0,
      scale portate a 1, ingressi nulli fuori dalle colonne attive. Riga «coppia»: colonna c1 (gruppo 0) e c2 (gruppo
      1) con x_c2 = 0,3 x_c1; residuo -0,55 su c1: da solo il salto su c1 porta il residuo efficace a 0,45, quello su
      c2 a -0,25, insieme a 0,75 (peggio di 0,55). Il criterio diagonale vede due salti buoni (guadagni 0,10 e 0,24).
      32 righe coppia, passo V VERO (raccolta, curvatura, selezione, accetta guadagno): la sonda di prima annulla TUTTO
      (la perdita sale); la sonda a dosi sceglie una dose PARZIALE che fa scendere la perdita su A e su B, tiene solo
      salti c2 (i migliori per guadagno), e nessuna coppia con entrambi i salti.
  V3  RICENTRO: 16 righe «buone» (residuo 0,9, salto che porta a 0,1) + 16 righe coppia, dose unica (frazioni (1,)):
      la dose intera migliora (i buoni pesano di piu') e si tiene; il lascia-uno-fuori dallo stato raggiunto trova
      dannosi TUTTI e soli i salti delle coppie (c1 piu' dannosi dei c2: 0,50 contro 0,36 teorici), il primo giro
      annulla i 16 c1 (accettato), il secondo prova 8 c2 e la sonda B lo RIFIUTA (rimessi): restano buoni + c2, la
      perdita minima raggiungibile; previsto contro misurato.
  V4  composizione con V-Bop: dose + ricentro con stato per gruppo; i salti provati e annullati lasciano lo stato come
      dopo l'EMA (m, v, H, eta' uguali al bit a un V-Bop che ha selezionato e non ha saltato); i tenuti hanno la
      correzione alla ECO; `annulla` del registro tenuto rimette byte e stato al bit.
  V5  regione di fiducia `esito_dosi`: dose parziale tenuta con r < 0,25 -> rho NON scende; r > 0,75 -> sale; dose
      intera -> regola di prima; non tenuto -> rho / 2; non adattivo: tenuto = invariato, non tenuto = /2.
  V6  ciclo vero (`allena`, studente VERO minuscolo di e2e_test_integrata: V-Bop + rho adattivo + CAKLD + F32 +
      LoRA-GA + Sophia + pesi kl + accetta guadagno + EMA + attribuzione) con --pv-sonda-dosi --pv-ricentra: 3 passi;
      curva dose -> sonda nel registro, nella storia e in registro_componenti.jsonl; ripresa 1 + 2 == 3 AL BIT;
      L_tutto(t) == L0(t+1) dell'attribuzione (il registro tenuto e' quello giusto); gradienti di theta intatti dopo
      il ricentro; accensione alla ripresa = decisione registrata; configurazioni rifiutate; CLI.
  V7  default al bit, ciclo: `allena` del ramo con i campi nuovi ai default == `allena` di integrata (git 39d514a)
      sullo stesso studente vero minuscolo e la stessa configurazione di EB3 in piccolo (theta, F32, LoRA, byte,
      registri, ottimizzatore, stato senza i campi nuovi).

Uso (4 fili, nice 19, niente GPU):
    systemd-run --user --scope -p CPUQuota=400% -p MemoryMax=6G nice -n 19 env CUDA_VISIBLE_DEVICES= \\
        HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \\
        E2E_VI_DIR=@@ROOT:home@@/claude-pod-lavoro/v-interazione-b-prove \\
        @@ROOT:venv@@/bin/python3 e2e_test_v_interazione.py [V1 V2 ...]
"""
from __future__ import annotations

import copy
import json
import math
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("TRITON_INTERPRET", "1")
_VI = os.environ.setdefault("E2E_VI_DIR", "@@ROOT:home@@/claude-pod-lavoro/v-interazione-b-prove")
os.environ.setdefault("E2E_PV_DIR", str(Path(_VI) / "pv"))
os.environ.setdefault("E2E_ALLENA_PROVE", str(Path(_VI) / "allena"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import e2e_studente as S  # noqa: E402
import e2e_pv_codici as PV  # noqa: E402
import e2e_test_pv_codici as TP  # noqa: E402

torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "4")))
DIR = Path(_VI)
# e2e_pv_codici.py / e2e_allena.py di `integrata` prima di questo ramo (dopo un merge: il commit di integrata di prima)
BASE_INTEGRATA = os.environ.get("E2E_VI_BASE", "39d514a")
ESITO: dict = {"data": time.strftime("%Y-%m-%d %H:%M:%S"), "prove": {}}
RIC = TP.RIC


def registra(nome: str, ok: bool, **dati) -> bool:
    ESITO["prove"][nome] = {"ok": bool(ok), **dati}
    print(f"[{nome}] {'✅' if ok else '❌'} {json.dumps(dati, ensure_ascii=False, default=str)[:2500]}", flush=True)
    return bool(ok)


# ─────────────────────────────────────────── V1 ───────────────────────────────────────────
def prova_v1() -> bool:
    t0 = time.time()
    VEC = TP._pv_prima(BASE_INTEGRATA)
    ft = PV.famiglia_scrivibile(TP.famiglia_sintetica(R=32))
    fs0 = PV.famiglia_scrivibile(TP.famiglia_sintetica(R=32))
    TP.pianta_errori(fs0, 0.01, 31)
    casi = {"pv": (4, lambda m: {}, {}, False, None),
            "vbop_curv2_rho_adattivo": (5, lambda m: {"bop": m.ConfigBop(gamma=0.5, curvatura_ogni=2)},
                                        {"prova": (0.03,), "adattivo": True, "rho_max": 0.1}, True, None),
            "vbop_guadagno_tg1": (4, lambda m: {"bop": m.ConfigBop(gamma=0.5)}, {"prova": (0.03,)}, False,
                                  {"accetta": "guadagno", "z_guadagno_min": 1.0})}
    uguali, salti = {}, {}
    for nome, (passi, pv_kw, reg_kw, adatt, ckw) in casi.items():
        prima = TP._corsa_mod(VEC, fs0, ft, passi, pv_kw(VEC), reg_kw, adatt=adatt, cfg_kw=ckw)
        nuovo = TP._corsa_mod(PV, fs0, ft, passi, pv_kw(PV), reg_kw, adatt=adatt, cfg_kw=ckw)
        uguali[nome] = TP._uguali_corse(prima, nuovo)
        salti[nome] = [o["salti"] for o in prima[3]]
    # applica_salti == righe_salti + salta_registro (stessi byte, stesse righe)
    fa, fb = TP._copia_fam(fs0), TP._copia_fam(fs0)
    k = (fa.strato, fa.nome)
    pv = PV.PassoV({k: fa}, PV.ConfigV())
    th = torch.zeros_like(fs0.d0, requires_grad=True)
    TP.raccogli(pv, fa, ft, th, TP.Dati(fs0, token=16, k=2, seme=12), 0)
    d = TP.scala(fa, th).detach()
    sel = PV.seleziona(pv.stato.buffer(fa), fa, d, pv.stato.n_verifica, 0.05, n_perdita=pv.stato.n_perdita,
                       n_curvatura=pv.stato.n_curvatura)
    righe = PV.righe_salti(fb, sel)
    PV.salta_registro(fb, righe)
    reg = PV.applica_salti(fa, sel)
    applica = len(sel) > 0 and np.array_equal(reg.righe, righe) and torch.equal(fa.blocchi, fb.blocchi)
    conti = {n: PV.conteggi_dosi(n, PV.FRAZIONI_DOSI) for n in (1, 3, 5, 64, 1000, 87697)}
    conti_ok = (conti[1] == [1] and conti[3] == [1, 3] and conti[64] == [1, 4, 16, 64]
                and conti[1000] == [4, 16, 63, 250, 1000] and conti[87697] == [343, 1371, 5482, 21925, 87697])
    rifiuti = []
    for kw in ({"frazioni": ()}, {"frazioni": (0.5, 0.25)}, {"frazioni": (0.0, 1.0)}, {"frazioni": (1.5,)},
               {"ricentra_giri": 0}):
        try:
            PV.ConfigDosi(**kw)
            rifiuti.append(False)
        except ValueError:
            rifiuti.append(True)
    ok = (all(all(v.values()) for v in uguali.values()) and all(sum(s) > 0 for s in salti.values()) and applica
          and conti_ok and all(rifiuti))
    return registra("V1_default_modulo_pv_al_bit", ok, uguali=uguali, salti=salti, applica_uguale=applica,
                    conteggi_dosi=conti, rifiuti=rifiuti, base=BASE_INTEGRATA, secondi=round(time.time() - t0, 1))


# ─────────────────────────────────── giocattolo con interazione ───────────────────────────────────
class Giocattolo:
    """Studente = UNA famiglia TQ1_0 sintetica (1 esperto, top-1), scale d = fp16(d0 exp(theta)) ~ 1 (theta = -log d0),
    maestro W_t = W_s + residui costruiti, ingressi x NULLI fuori dalle colonne attive (nessun candidato altrove).
    Perdita del passo = sum_t |y_t - yt_t|^2 / n_tot (16 micro-lotti da `T` token: unita' di un micro-lotto come nel
    ciclo vero); sonda = media per token su `T` token mai visti (A e B con semi diversi)."""

    A_C2 = 0.3                               # x_c2 = a x_c1
    RES_COPPIA = 0.55                        # residuo su c1 (in unita' di d): soli 0,45 / -0,25, insieme 0,75
    RES_BUONO = 0.9                          # residuo della riga buona: il salto lo porta a 0,1

    def __init__(self, n_coppie: int, n_buoni: int = 0, T: int = 128, seme: int = 7):
        R = n_coppie + n_buoni
        self.fs0 = PV.famiglia_scrivibile(TP.famiglia_sintetica(E=1, R=R, C=1024, seme=seme))
        self.T, self.seme = T, seme
        fs = self.fs0
        gs = fs.gs
        assert fs.NG >= 2 and bool((fs.d0 > 0).all())
        self.theta0 = torch.log(1.0 / fs.d0)
        self.d = TP.scala(fs, self.theta0).detach()
        g = torch.Generator().manual_seed(seme)
        Tt = TP.trit_famiglia(fs)[0]                                     # (R, C)
        W = (Tt.to(torch.float64).view(R, fs.NG, gs) * self.d[0].to(torch.float64).unsqueeze(-1)).view(R, fs.C)
        self.Wt = W.clone()
        self.fonte = torch.full((fs.C,), -1, dtype=torch.int64)          # colonna attiva: di chi e' la z
        self.scala_x = torch.zeros(fs.C, dtype=torch.float64)
        usate = set()
        self.coppie, self.buoni = [], []

        def libera(r, gruppo):
            cand = [c for c in range(gruppo * gs, (gruppo + 1) * gs) if int(Tt[r, c]) == 0 and c not in usate]
            c = cand[int(torch.randint(0, len(cand), (1,), generator=g))]
            usate.add(c)
            return c
        for r in range(R):
            s = 1 if bool(torch.rand(1, generator=g) < 0.5) else -1
            if r < n_coppie:
                c1, c2 = libera(r, 0), libera(r, 1)
                self.Wt[r, c1] += self.RES_COPPIA * s * float(self.d[0, r, c1 // gs])
                self.fonte[c1], self.scala_x[c1] = c1, 1.0
                self.fonte[c2], self.scala_x[c2] = c1, self.A_C2
                self.coppie.append((r, c1, c2, s))
            else:
                c = libera(r, 0)
                self.Wt[r, c] += self.RES_BUONO * s * float(self.d[0, r, c // gs])
                self.fonte[c], self.scala_x[c] = c, 1.0
                self.buoni.append((r, c, s))
        self.k = (fs.strato, fs.nome)

    def lotto(self, passo: int, kk: int, T: int = None):
        T = T or self.T
        g = torch.Generator().manual_seed(self.seme * 1000003 + passo * 1009 + kk)
        z = torch.randn(T, self.fs0.C, generator=g, dtype=torch.float64)
        x = torch.zeros_like(z)
        att = self.fonte >= 0
        x[:, att] = z[:, self.fonte[att]] * self.scala_x[att]
        return x.to(torch.float32), S.Piano(torch.zeros(T, 1, dtype=torch.int64), 1)

    def perdita(self, fs, x, piano, mot, n_tot, d):
        y = S.FamigliaEsperti.apply(x, d, fs, piano, RIC, mot)
        yt = (x.to(torch.float64) @ self.Wt.T).to(y.dtype)
        return ((y - yt) ** 2).sum() / n_tot

    def raccogli(self, pv, fs, passo: int, n_micro: int = 16, n_curv: int = 2) -> int:
        """Passo V completo fino a `fine_raccolta` (theta fermo: solo V). Ritorna n_tot."""
        mot = PV.MotoreTorchV(pv.stato)
        pv.inizia_passo()
        n_tot = n_micro * self.T
        th = self.theta0.clone().requires_grad_(True)
        for kk in range(n_micro):
            pv.microlotto(kk)
            x, piano = self.lotto(passo, kk)
            self.perdita(fs, x, piano, mot, n_tot, TP.scala(fs, th)).backward()
        if pv.stato.serve_curvatura():
            for kk in range(n_curv):
                pv.curvatura()
                x, piano = self.lotto(passo, 1000 + kk)
                y = S.FamigliaEsperti.apply(x, TP.scala(fs, th), fs, piano, RIC, mot)
                gr = torch.Generator().manual_seed(self.seme * 7919 + passo * 31 + kk)
                y.backward((torch.randint(0, 2, y.shape, generator=gr).to(y.dtype) * 2 - 1) * math.sqrt(2.0 / n_tot))
        pv.fine_raccolta()
        return n_tot

    def sonde(self, pv, fs, passo: int, n_tot: int):
        """(misura_a, misura_b, gradiente_ricentro) come nel ciclo: media per token su T token; indietro della sonda A
        con la scala di un micro-lotto del passo (1 / n_tot) nel motore V (lo StatoV e' in `prepara_ricentro`)."""
        xa, pa = self.lotto(passo + 1, 0)
        xb, pb = self.lotto(passo + 1, 1)
        mot = PV.MotoreTorchV(pv.stato)

        def misura(x, p):
            with torch.no_grad():
                return float(self.perdita(fs, x, p, S.MotoreTorch(), x.shape[0], self.d))

        def grad():
            th = self.theta0.clone().requires_grad_(True)
            self.perdita(fs, xa, pa, mot, n_tot, TP.scala(fs, th)).backward()
            return True
        return (lambda: misura(xa, pa)), (lambda: misura(xb, pb)), grad

    def trit(self, fs) -> torch.Tensor:
        return TP.trit_famiglia(fs)[0]

    def conta(self, fs) -> dict:
        """Quali salti delle righe costruite sono applicati ora (rispetto ai trit di partenza)."""
        T0, T1 = self.trit(self.fs0), self.trit(fs)
        cambiati = int((T0 != T1).sum())
        c1 = sum(int(T1[r, a] != T0[r, a]) for r, a, _, _ in self.coppie)
        c2 = sum(int(T1[r, b] != T0[r, b]) for r, _, b, _ in self.coppie)
        entrambi = sum(int(T1[r, a] != T0[r, a] and T1[r, b] != T0[r, b]) for r, a, b, _ in self.coppie)
        buoni = sum(int(T1[r, c] != T0[r, c]) for r, c, _ in self.buoni)
        return {"cambiati": cambiati, "c1": c1, "c2": c2, "coppie_entrambi": entrambi, "buoni": buoni,
                "altri": cambiati - c1 - c2 - buoni}


def _pv_gioco(gio: Giocattolo, bop=None):
    fs = TP._copia_fam(gio.fs0)
    pv = PV.PassoV({gio.k: fs}, PV.ConfigV(accetta="guadagno"), bop=bop)
    return fs, pv


# ─────────────────────────────────────────── V2 ───────────────────────────────────────────
def prova_v2() -> bool:
    t0 = time.time()
    gio = Giocattolo(n_coppie=32, seme=7)
    rho = 1.0
    # (a) la sonda di prima (tutto o niente) sugli stessi buffer
    fs_v, pv_v = _pv_gioco(gio)
    n_tot = gio.raccogli(pv_v, fs_v, 0)
    ma, mb, _ = gio.sonde(pv_v, fs_v, 0, n_tot)
    prima_a = ma()
    reg_v, diag_v = pv_v.seleziona_e_salta({gio.k: gio.d}, rho)
    tutti = gio.conta(fs_v)
    dopo_a = ma()
    rf = PV.RegioneFiducia(prova=(rho,))
    rf.storia = [{"ripresa": True}]
    rf.rho = rho
    tiene_vecchio = rf.esito(rho, prima_a, dopo_a, len(reg_v))
    if not tiene_vecchio:
        pv_v.annulla(reg_v)
    vecchio_bit = torch.equal(fs_v.blocchi, gio.fs0.blocchi)
    # (b) sonda a dosi (stessa raccolta: stessi dati, stessa selezione)
    fs, pv = _pv_gioco(gio)
    n_tot = gio.raccogli(pv, fs, 0)
    ma, mb, gr = gio.sonde(pv, fs, 0, n_tot)
    B0 = fs.blocchi.clone()
    reg, diag, es = pv.seleziona_e_prova_dosi({gio.k: gio.d}, rho, ma, mb, PV.ConfigDosi())
    dopo = gio.conta(fs)
    curva = [(x["salti"], round(x["sonda_a"], 6)) for x in es["dosi"]]
    registro_ok = reg is not None and len(pv.registri) == 1 and pv.registri[0] is reg and len(reg) == es["salti_tenuti"]
    # il registro tenuto rigiocato sulla partenza da' gli stessi byte; annullato torna alla partenza
    fr = TP._copia_fam(gio.fs0)
    PV.rigioca({gio.k: fr}, [reg]) if reg is not None else None
    rigioco = reg is not None and torch.equal(fr.blocchi, fs.blocchi)
    fa = TP._copia_fam(fs)
    PV.annulla({gio.k: fa}, reg) if reg is not None else None
    annulla_bit = torch.equal(fa.blocchi, B0)
    n = es["salti"]
    # (c) nessuna dose migliora (sonda A finta che cresce coi salti applicati): tutto annullato al bit
    fs_c, pv_c = _pv_gioco(gio)
    n_c = gio.raccogli(pv_c, fs_c, 0)
    ma_c, mb_c, _ = gio.sonde(pv_c, fs_c, 0, n_c)
    Bc = fs_c.blocchi.clone()
    peggio = lambda: 1.0 + float((gio.trit(fs_c) != gio.trit(gio.fs0)).sum())  # noqa: E731
    reg_c, _, es_c = pv_c.seleziona_e_prova_dosi({gio.k: gio.d}, rho, peggio, mb_c, PV.ConfigDosi())
    tutto_annullato = reg_c is None and torch.equal(fs_c.blocchi, Bc) and pv_c.registri == [] and not es_c["tenuto"]
    # (d) A migliora ma B non conferma: annullato
    fs_d, pv_d = _pv_gioco(gio)
    n_d = gio.raccogli(pv_d, fs_d, 0)
    ma_d, _, _ = gio.sonde(pv_d, fs_d, 0, n_d)
    Bd = fs_d.blocchi.clone()
    peggio_d = lambda: 1.0 + float((gio.trit(fs_d) != gio.trit(gio.fs0)).sum())  # noqa: E731
    reg_d, _, es_d = pv_d.seleziona_e_prova_dosi({gio.k: gio.d}, rho, ma_d, peggio_d, PV.ConfigDosi())
    non_confermata = (reg_d is None and es_d["confermata"] is False and es_d["sonda_dopo_b"] is not None
                      and torch.equal(fs_d.blocchi, Bd) and pv_d.registri == [])
    ok = (not tiene_vecchio and vecchio_bit and tutti["coppie_entrambi"] > 0
          and es["tenuto"] and es["parziale"] and 0 < es["salti_tenuti"] < n
          and es["dose"]["frazione"] < 1.0
          and es["sonda_dopo_a"] < es["sonda_prima_a"] and es["sonda_dopo_b"] <= es["sonda_prima_b"]
          and es["dosi"][-1]["sonda_a"] > es["sonda_prima_a"]                 # la dose intera peggiora
          and dopo["coppie_entrambi"] == 0 and dopo["c2"] == es["salti_tenuti"] and dopo["altri"] == 0
          and registro_ok and rigioco and annulla_bit and tutto_annullato and non_confermata)
    return registra("V2_dosi_evitano_il_peggioramento", ok, salti_scelti=n, diag=diag,
                    sonda_di_prima={"prima": prima_a, "dopo": dopo_a, "tenuto": tiene_vecchio, "salti": len(reg_v),
                                    "applicati": tutti, "annullato_al_bit": vecchio_bit},
                    dosi=curva, dose=es["dose"], sonda_a=(es["sonda_prima_a"], es["sonda_dopo_a"]),
                    sonda_b=(es["sonda_prima_b"], es["sonda_dopo_b"]), tenuti=es["salti_tenuti"], applicati=dopo,
                    registro=registro_ok, rigioco=rigioco, annulla_al_bit=annulla_bit,
                    ogni_dose_peggio_annulla_tutto=tutto_annullato, b_non_conferma_annulla=non_confermata,
                    secondi=round(time.time() - t0, 1))


# ─────────────────────────────────────────── V3 ───────────────────────────────────────────
def _loo_teorico(gio: Giocattolo) -> dict:
    """Lascia-uno-fuori teorico nello stato «tutti i salti» (unita' di d^2 x varianza): c1 0,50; c2 0,36; buoni -0,80."""
    a, e = gio.A_C2, gio.RES_COPPIA
    r = 1.0 + a - e
    return {"c1": 2 * r - 1.0, "c2": 2 * r * a - a * a, "buono": 2 * (1.0 - gio.RES_BUONO) - 1.0}


def prova_v3() -> bool:
    t0 = time.time()
    gio = Giocattolo(n_coppie=16, n_buoni=16, seme=11, T=4096)
    fs, pv = _pv_gioco(gio)
    n_tot = gio.raccogli(pv, fs, 0)
    ma, mb, gr = gio.sonde(pv, fs, 0, n_tot)
    orig_dosi = pv.seleziona_e_prova_dosi
    stato_qs = {}

    def spia_ricentra(*a, **k):                  # lo stato q_S (dopo la dose, prima del ricentro)
        stato_qs.update(gio.conta(fs))
        return orig_ricentra(*a, **k)
    orig_ricentra = pv._ricentra
    pv._ricentra = spia_ricentra
    reg, diag, es = orig_dosi({gio.k: gio.d}, 1.0, ma, mb,
                              PV.ConfigDosi(frazioni=(1.0,), ricentra=True, ricentra_giri=3), gr)
    dopo = gio.conta(fs)
    ri = es["ricentro"] or {}
    giri = ri.get("giri", [])
    # dose intera senza ricentro (stesso stato di partenza): per il confronto della perdita
    fs2, pv2 = _pv_gioco(gio)
    n2 = gio.raccogli(pv2, fs2, 0)
    ma2, mb2, _ = gio.sonde(pv2, fs2, 0, n2)
    _, _, es2 = pv2.seleziona_e_prova_dosi({gio.k: gio.d}, 1.0, ma2, mb2, PV.ConfigDosi(frazioni=(1.0,)))
    teo = _loo_teorico(gio)
    entrambi = stato_qs.get("coppie_entrambi", -1)
    ok = (es["tenuto"] and es2["tenuto"] and es2["salti_tenuti"] == es["salti"] and entrambi > 0
          and ri.get("valido")
          # dannosi = TUTTI e soli i salti delle coppie con entrambi i salti (teoria: c1 +0,50, c2 +0,36; i salti soli
          # e i buoni hanno lascia-uno-fuori < 0)
          and ri.get("dannosi") == 2 * entrambi
          # primo giro: la meta' piu' dannosa = i c1 di quelle coppie (0,50 > 0,36), accettato; secondo giro: meta'
          # dei c2 rimasti (prezzi ormai stantii: da soli sono buoni), la sonda B peggiora e lo RIFIUTA
          and len(giri) == 2 and giri[0]["accettato"] and giri[0]["annullati"] == entrambi
          and not giri[1]["accettato"] and giri[1]["sonda_b_dopo"] > giri[1]["sonda_b_prima"]
          and dopo["coppie_entrambi"] == 0 and dopo["c1"] == stato_qs["c1"] - entrambi and dopo["c2"] == stato_qs["c2"]
          and dopo["buoni"] == len(gio.buoni) == stato_qs["buoni"] and dopo["altri"] == 0
          and es["sonda_dopo_b"] < es2["sonda_dopo_b"] and len(reg) == es["salti_tenuti"])
    # rapporto previsto/misurato del primo giro (unita' di un micro-lotto -> per token della sonda: x n_tot / T)
    prev = giri[0]["loo_previsto"] * n_tot / gio.T if giri else None
    mis = (giri[0]["sonda_b_prima"] - giri[0]["sonda_b_dopo"]) if giri else None
    return registra("V3_ricentro_trova_e_annulla_i_dannosi", ok, salti_scelti=es["salti"], ricentro=ri,
                    applicati_in_q_S=stato_qs, applicati_dopo=dopo, sonda_b_senza_ricentro=es2["sonda_dopo_b"], sonda_b_con_ricentro=es["sonda_dopo_b"],
                    loo_teorico_per_salto=teo, primo_giro_previsto_per_token=prev, primo_giro_misurato_su_B=mis,
                    secondi=round(time.time() - t0, 1))


# ─────────────────────────────────────────── V4 ───────────────────────────────────────────
def prova_v4() -> bool:
    t0 = time.time()
    gio = Giocattolo(n_coppie=16, n_buoni=16, seme=13, T=4096)
    bop = PV.ConfigBop(gamma=0.5)
    # riferimento: V-Bop che aggiorna e seleziona ma non salta (lo stato dopo l'EMA)
    fr, pr = _pv_gioco(gio, bop)
    gio.raccogli(pr, fr, 0)
    pr.stato.aggiorna_bop()
    rif = {k: {c: getattr(z, c).clone() for c in PV._CAMPI_BOP} for k, z in pr.stato.bop_.items()}
    fs, pv = _pv_gioco(gio, bop)
    n_tot = gio.raccogli(pv, fs, 0)
    ma, mb, gr = gio.sonde(pv, fs, 0, n_tot)
    B0 = fs.blocchi.clone()
    reg, diag, es = pv.seleziona_e_prova_dosi({gio.k: gio.d}, 1.0, ma, mb,
                                              PV.ConfigDosi(frazioni=(1.0,), ricentra=True), gr)
    z = pv.stato.bop_[gio.k]
    zr = rif[gio.k]
    tenuti = set()
    if reg is not None:
        for e_, r_, c_ in zip(reg.righe["e"], reg.righe["r"], reg.righe["c"]):
            tenuti.add((int(e_), int(r_), int(c_) // fs.gs))
    # i provati e annullati dal ricentro (salti delle coppie non tenuti): stato == dopo l'EMA, al bit
    provati_annullati = [(0, r, c // fs.gs) for r, c1, c2, _ in gio.coppie for c in (c1, c2)
                         if (0, r, c // fs.gs) not in tenuti and int(zr["eta"][0, r, c // fs.gs] & 127) >= 1]
    annullati_bit = len(provati_annullati) > 0 and all(
        all(torch.equal(getattr(z, c)[e, r, g], zr[c][e, r, g]) for c in ("cand", "m", "r", "H", "eta"))
        for e, r, g in provati_annullati)
    # i tenuti: m corretto alla ECO (diverso dall'EMA) salvo stato azzerato per verso non piu' fattibile
    corretti = sum(int(not torch.equal(z.m[e, r, g], zr["m"][e, r, g])) for e, r, g in tenuti)
    # altrove nulla cambia
    maschera = torch.ones_like(z.m, dtype=torch.bool)
    for e, r, g in tenuti:
        maschera[e, r, g] = False
    altrove = all(torch.equal(getattr(z, c)[maschera], zr[c][maschera]) for c in ("cand", "m", "r", "H", "eta"))
    # annullamento del registro tenuto: byte e stato al bit (dopo l'EMA)
    pv.annulla(reg)
    torna = torch.equal(fs.blocchi, B0) and all(torch.equal(getattr(z, c), zr[c]) for c in PV._CAMPI_BOP)
    ok = (es["tenuto"] and (es["ricentro"] or {}).get("annullati", 0) > 0 and annullati_bit
          and corretti == len(tenuti) and altrove and torna and "con_candidato" in next(iter(diag.values())))
    return registra("V4_vbop_stato_coi_salti_annullati", ok, tenuti=len(tenuti), annullati_dal_ricentro=len(provati_annullati),
                    stato_annullati_al_bit=annullati_bit, tenuti_corretti_alla_eco=corretti, altrove_invariato=altrove,
                    annulla_torna_al_bit=torna, ricentro=es["ricentro"], secondi=round(time.time() - t0, 1))


# ─────────────────────────────────────────── V5 ───────────────────────────────────────────
def prova_v5() -> bool:
    def rf(adatt=True):
        r = PV.RegioneFiducia(prova=(0.01,), adattivo=adatt, rho_max=0.1)
        r.storia, r.rho = [{"ripresa": True}], 0.01
        return r
    casi = {}
    r = rf()
    r.esito_dosi(0.01, 1.0, 0.99, 10, True, True, guadagno_previsto=1.0)      # r = 0,01 < 0,25, parziale
    casi["parziale_r_basso_non_scende"] = r.rho == 0.01
    r = rf()
    r.esito_dosi(0.01, 1.0, 0.1, 10, True, True, guadagno_previsto=1.0)       # r = 0,9 > 0,75, parziale
    casi["parziale_r_alto_sale"] = r.rho == 0.02
    r = rf()
    r.esito_dosi(0.01, 1.0, 0.99, 10, True, False, guadagno_previsto=1.0)     # dose intera, r basso
    casi["intera_r_basso_scende"] = r.rho == 0.005
    r = rf()
    r.esito_dosi(0.01, 1.0, 1.2, 10, False, False, guadagno_previsto=1.0)     # non tenuto
    casi["non_tenuto_dimezza"] = r.rho == 0.005 and r.storia[-1]["r"] is None
    r = rf(False)
    r.esito_dosi(0.01, 1.0, 0.99, 10, True, True)
    casi["non_adattivo_tenuto_invariato"] = r.rho == 0.01
    r = rf(False)
    r.esito_dosi(0.01, 1.0, 0.98, 10, False, True)
    casi["non_adattivo_non_tenuto_dimezza"] = r.rho == 0.005
    # taratura di una corsa nuova (prova di un solo rho): tenuto con calo -> quel rho
    r = PV.RegioneFiducia(prova=(0.02,))
    r.esito_dosi(0.02, 1.0, 0.9, 5, True, True)
    casi["taratura"] = r.rho == 0.02
    return registra("V5_regione_di_fiducia_dosi", all(casi.values()), casi=casi)


# ─────────────────────────────────────────── V6 ───────────────────────────────────────────
def prova_v6() -> bool:
    import e2e_allena as A
    import e2e_test_integrata as TI
    import e2e_test_lora as TL
    t0 = time.time()
    src, WS = TL.sorgente_lora(n_train=12, n_held=4)
    src = TI.con_domini(src)
    fr = (1 / 1024, 1 / 256, 1 / 64, 1 / 16, 1 / 4, 1.0)            # anche la griglia da riga di comando
    ca = TI._ca_integrata(pv_sonda_dosi=True, pv_ricentra=True, pv_dosi_frazioni=fr)
    d_f, d_r, d_s = DIR / "v6_filata", DIR / "v6_ripresa", DIR / "v6_accesa_alla_ripresa"
    TI._pulisci(d_f, d_r, d_s)
    reg: list = []
    grad_ok = []
    orig = A.gradiente_ricentro

    def spia(st, lot, ca_, WS_, WT_, sc):                # gradienti di theta intatti dopo il ricentro
        prima = [None if p.grad is None else p.grad.clone() for p in st.parametri()]
        v = orig(st, lot, ca_, WS_, WT_, sc)
        grad_ok.append(all((a is None and p.grad is None) or (a is not None and p.grad is not None
                                                              and torch.equal(a, p.grad))
                           for a, p in zip(prima, st.parametri())))
        return v
    A.gradiente_ricentro = spia
    try:
        st_f = TI._studente_int()
        z_f = A.allena(st_f, src, d_f, ca, WS, WS, log=reg.append)
        A.allena(TI._studente_int(), src, d_r, ca, WS, WS, ferma_a=1, log=TI.muto)
        st_r = TI._studente_int()
        z_r = A.allena(st_r, src, d_r, ca, WS, WS, riprendi=True, log=TI.muto)
        # accesa alla ripresa: spenta fino al passo 1, poi accesa (decisione registrata)
        A.allena(TI._studente_int(), src, d_s, TI._ca_integrata(), WS, WS, ferma_a=1, log=TI.muto)
        reg_s: list = []
        z_s = A.allena(TI._studente_int(), src, d_s, ca, WS, WS, riprendi=True, log=reg_s.append)
    finally:
        A.gradiente_ricentro = orig
    storia = z_f["storia"]
    ev = [x["pv"] for x in storia if x.get("pv")]
    voci = {"passi": z_f["passo"] == 3 and len(ev) == 2,
            "curva": all(len(e["dosi"]) >= 1 and all({"frazione", "salti", "sonda_a"} <= set(x) for x in e["dosi"])
                         and [x["salti"] for x in e["dosi"]] == PV.conteggi_dosi(e["salti"], fr)
                         for e in ev if e["salti"] > 0),
            "conferma": all("conferma" in e and "salti_tenuti" in e for e in ev),
            "log_dosi": any("passo V" in m and "dosi " in m for m in reg) and any("sonda a DOSI" in m for m in reg),
            "ricentro_nel_log": any("RICENTRO" in m for m in reg),
            "grad_intatti": len(grad_ok) > 0 and all(grad_ok),
            "contatori": z_f["pv"]["salti_tenuti"] == sum(e["salti_tenuti"] for e in ev if e["tenuto"])}
    righe = TI._righe_jsonl(d_f)
    jl = len(righe) == 3 and all("dosi" in r["pv"] for r in righe if "pv" in r) and any("pv" in r for r in righe)
    catena = all(storia[i]["attribuzione"]["L_tutto"]["perdita"] == storia[i + 1]["attribuzione"]["L0"]["perdita"]
                 for i in range(len(storia) - 1))
    rf, rr = A.carica_ripresa(d_f, "cpu"), A.carica_ripresa(d_r, "cpu")
    rip = {"theta": TI._uguali_lista(st_f.parametri(), st_r.parametri()),
           "f32": TI._uguali_lista(st_f.parametri_f32(), st_r.parametri_f32()),
           "lora": TI._uguali_lora_ga(st_f, st_r), "byte": TI._uguali_lista(TI._byte(st_f), TI._byte(st_r)),
           "opt": TI._uguali_stato_opt(rf["opt"], rr["opt"]),
           "stato": TI._stato_confrontabile(z_f) == TI._stato_confrontabile(z_r),
           "bop_sha": z_f["pv"].get("bop", {}).get("sha") == z_r["pv"].get("bop", {}).get("sha") is not None,
           "registri_sha": z_f["pv"]["sha"] == z_r["pv"]["sha"]}
    cambi = next((m for m in reg_s if m.startswith("ripresa con campi cambiati")), "")
    switch = ("pv_sonda_dosi None -> True" in cambi or "pv_sonda_dosi False -> True" in cambi) and z_s["passo"] == 3 \
        and any("dosi" in (x.get("pv") or {}) for x in z_s["storia"][1:])
    rifiuti = []
    for kw in ({"pv_ricentra": True}, {"pv_sonda_dosi": True, "pv_ricentra": True, "pv_ricentra_giri": 0},
               {"pv_sonda_dosi": True, "righe_per_passo": 1, "righe_per_micro": 1},
               {"pv_dosi_frazioni": (0.25, 1.0)}, {"pv_sonda_dosi": True, "pv_dosi_frazioni": (0.5, 0.25)},
               {"pv_sonda_dosi": True, "pv_dosi_frazioni": (0.0, 1.0)}, {"pv_sonda_dosi": True, "pv_dosi_frazioni": [1.0]}):
        try:
            TI._ca_integrata(**kw)
            rifiuti.append(False)
        except ValueError:
            rifiuti.append(True)
    testo = (QUI / "e2e_allena.py").read_text()
    cli = (all(f'"--{x}"' in testo for x in ("pv-sonda-dosi", "pv-ricentra", "pv-ricentra-giri", "pv-dosi-frazioni"))
           and A._frazioni_da_arg("1/4096, 1/256,1/16,1") == (1 / 4096, 1 / 256, 1 / 16, 1.0)
           and A._frazioni_da_arg("") == () and A._frazioni_da_arg(None) == ())
    campi = {"pv_sonda_dosi", "pv_ricentra", "pv_ricentra_giri", "pv_dosi_frazioni"} <= A._CAMPI_DI_DECISIONE
    dflt = A.ConfigAllena()
    dflt_ok = (dflt.pv_sonda_dosi, dflt.pv_ricentra, dflt.pv_ricentra_giri, dflt.pv_dosi_frazioni) == (False, False, 3, ())
    ok = (all(voci.values()) and jl and catena and all(rip.values()) and switch and all(rifiuti) and cli and campi
          and dflt_ok)
    return registra("V6_ciclo_vero_dosi_e_ricentro", ok, voci=voci, jsonl=jl, L_tutto_uguale_L0_dopo=catena,
                    ripresa=rip, accesa_alla_ripresa=switch, rifiuti=rifiuti, cli=cli, campi_decisione=campi,
                    passi_v=[{k: e[k] for k in ("salti", "salti_tenuti", "tenuto", "rho", "sonda_prima", "sonda_dopo",
                                                 "dose", "conferma", "ricentro") if k in e} for e in ev],
                    curve=[e.get("dosi") for e in ev], secondi=round(time.time() - t0, 1))


# ─────────────────────────────────────────── V7 ───────────────────────────────────────────
def prova_v7() -> bool:
    import e2e_allena as A
    import e2e_test_allena as TA
    import e2e_test_integrata as TI
    import e2e_test_lora as TL
    t0 = time.time()
    VEC = TA._modulo_prima(BASE_INTEGRATA)
    src, WS = TL.sorgente_lora(n_train=12, n_held=4)
    src = TI.con_domini(src)
    ris = {}
    for nome, mod in (("prima", VEC), ("nuovo", A)):
        d = DIR / f"v7_{nome}"
        TI._pulisci(d)
        ca = mod.ConfigAllena(**{k: v for k, v in TI._ca_integrata().__dict__.items()
                                 if k in mod.ConfigAllena.__dataclass_fields__})
        st = TI._studente_int()
        z = mod.allena(st, src, d, ca, WS, WS, log=TI.muto)
        ris[nome] = (st, z, mod.carica_ripresa(d, "cpu"))
    (sp, zp, rp), (sn, zn, rn) = ris["prima"], ris["nuovo"]
    a_, b_ = TI._stato_confrontabile(zn), TI._stato_confrontabile(zp)
    nuovi = {k: a_["config"].pop(k) for k in set(a_["config"]) - set(b_["config"])}
    a_.pop("versione", None)
    b_.pop("versione", None)
    u = {"theta": TI._uguali_lista(sn.parametri(), sp.parametri()),
         "f32": TI._uguali_lista(sn.parametri_f32(), sp.parametri_f32()),
         "lora": TI._uguali_lora_ga(sn, sp), "byte": TI._uguali_lista(TI._byte(sn), TI._byte(sp)),
         "opt": TI._uguali_stato_opt(rn["opt"], rp["opt"]),
         "stato": a_ == b_, "registri_sha": zn["pv"]["sha"] == zp["pv"]["sha"],
         # campi nuovi rispetto alla base: TUTTI ai loro default (i miei 4 fra questi; dopo un merge anche quelli di
         # altri rami arrivati in integrata dopo la base)
         "campi_nuovi": ({"pv_sonda_dosi", "pv_ricentra", "pv_ricentra_giri", "pv_dosi_frazioni"} <= set(nuovi)
                         and nuovi == {k: A._CONFIG_DEFAULT[k] for k in nuovi})}
    return registra("V7_default_ciclo_al_bit", all(u.values()), uguali=u, campi_nuovi=nuovi,
                    pv=[(x["pv"]["salti"], x["pv"]["tenuto"]) for x in zn["storia"] if x.get("pv")],
                    base=BASE_INTEGRATA, secondi=round(time.time() - t0, 1))


# ─────────────────────────────────────────── V8 ───────────────────────────────────────────
def prova_v8() -> bool:
    """MISURA sullo strato 0 VERO di Thor-T (legge il GGUF; impostazione di A33: maestro = theta* + 0,5% dei codici
    spostati di un livello, V-Bop, --pv-accetta guadagno t_g > 1, rho 0,05, motore torch su CPU, 6 passi da 6 righe):
    stessi dati, (a) la sonda tutto-o-niente di prima (A33 misurava: 4 passi V su 5 annullati, sonda fino a x57, rho
    0,05 -> 0,003); (b) --pv-sonda-dosi --pv-ricentra. Per passo V: salti scelti, dose, tenuti, sonde A/B, ricentro,
    rho; held-out ai passi 2, 4, 6. Verde se entrambe le corse finiscono e, nella (b), i byte dello studente ==
    GGUF + rigioco dei registri salvati (il registro tenuto e' quello giusto)."""
    import e2e_allena as A
    import e2e_campi_scala as CAMPI  # noqa: F401
    import e2e_codici_scale as CS
    import e2e_test_allena as TA
    t0 = time.time()
    st_t = TA.studente((0,))
    for f in st_t.esperti(0).fam.values():
        PV.famiglia_scrivibile(f)
    g = torch.Generator().manual_seed(15)
    for n, f in st_t.esperti(0).fam.items():
        T_ = CS.trit_da_blocchi(f.blocchi.reshape(-1, f.nb), f.tipo).view(f.E, f.R, f.C)
        k = int(0.005 * f.R * f.C)
        for e in range(f.E):
            pos = torch.randperm(f.R * f.C, generator=g)[:k]
            r_, c_ = pos // f.C, pos % f.C
            t_ = T_[e, r_, c_].to(torch.int32)
            nuovo_t = torch.where(t_ == 0, torch.randint(0, 2, (k,), generator=g, dtype=torch.int32) * 2 - 1,
                                  torch.zeros_like(t_))
            PV.salta_byte(f.blocchi, f.tipo, torch.full((k,), e), r_, c_, t_, nuovo_t)
    th = TA.theta_stella(st_t)
    passi = 6
    src = TA.sorgente(st_t, 6 * passi, 4, th)
    del st_t
    base = dict(pv_codici=True, righe_per_passo=6, righe_per_micro=1, heldout_ogni=2, ripresa_ogni=2, pv_rho=0.05,
                pv_modo="vbop", pv_accetta="guadagno", pv_z_guadagno_min=1.0)
    corse = {"prima": TA._ca(**base), "dosi_ricentro": TA._ca(**base, pv_sonda_dosi=True, pv_ricentra=True)}
    ris = {}
    for nome, ca in corse.items():
        d = DIR / f"v8_{nome}"
        shutil.rmtree(d, ignore_errors=True)
        st = TA.studente((0,))
        WS = TA.testa_ristretta(st)
        lg: list = []
        z = A.allena(st, src, d, ca, WS, WS, log=lg.append)
        ris[nome] = (st, z, d, lg)
    st_d, z_d, d_d, _ = ris["dosi_ricentro"]
    st_g = TA.studente((0,))
    fams = {(0, n): PV.famiglia_scrivibile(f) for n, f in st_g.esperti(0).fam.items()}
    regs = PV.Registro.carica_lista(d_d / z_d["pv"]["file"], z_d["pv"]["sha"])[:z_d["pv"]["registri"]]
    PV.rigioca(fams, regs)
    rigioco = all(torch.equal(fams[(0, n)].blocchi, st_d.esperti(0).fam[n].blocchi) for n in st_d.esperti(0).fam)
    campi = ("salti", "salti_tenuti", "tenuto", "rho", "rho_dopo", "sonda_prima", "sonda_dopo", "r_fiducia", "dose",
             "conferma", "convenienti")
    misura = {nome: [{k: (round(v, 6) if isinstance(v, float) else v) for k, v in x["pv"].items() if k in campi}
                     for x in z["storia"] if x.get("pv")] for nome, (_, z, _, _) in ris.items()}
    curve = [x["pv"].get("dosi") for x in z_d["storia"] if x.get("pv")]
    ricentri = [x["pv"].get("ricentro") for x in z_d["storia"] if x.get("pv")]
    held = {nome: [(x["passo"], round(x["metrica"], 6)) for x in z["storia_heldout"]] for nome, (_, z, _, _) in ris.items()}
    ok = all(z["passo"] == passi for _, z, _, _ in ris.values()) and rigioco
    return registra("V8_misura_strato0_vero_dosi_contro_tutto_o_niente", ok, rigioco_al_bit=rigioco,
                    salti_tenuti={n: z["pv"]["salti_tenuti"] for n, (_, z, _, _) in ris.items()},
                    rho_finale={n: z["pv"]["rho"] for n, (_, z, _, _) in ris.items()}, heldout=held,
                    per_passo_v=misura, curve_dosi=curve, ricentri=ricentri, secondi=round(time.time() - t0, 1))


PROVE = {"V1": prova_v1, "V2": prova_v2, "V3": prova_v3, "V4": prova_v4, "V5": prova_v5, "V6": prova_v6,
         "V7": prova_v7, "V8": prova_v8}


def main() -> int:
    DIR.mkdir(parents=True, exist_ok=True)
    Path(os.environ["E2E_PV_DIR"]).mkdir(parents=True, exist_ok=True)
    scelte = [a for a in sys.argv[1:] if a in PROVE] or list(PROVE)
    esito_p = DIR / "esito_test_v_interazione.json"
    if esito_p.exists():
        try:
            ESITO["prove"].update(json.loads(esito_p.read_text()).get("prove", {}))
        except Exception as ex:  # noqa: BLE001
            print(f"esito precedente illeggibile ({ex}): si riparte", flush=True)
    ok = True
    for nome in scelte:
        for k in [k for k in ESITO["prove"] if k.startswith(nome + "_")]:
            del ESITO["prove"][k]
        t0 = time.time()
        try:
            ok &= PROVE[nome]()
        except Exception as ex:  # noqa: BLE001
            traceback.print_exc()
            ok &= registra(f"{nome}_eccezione", False, errore=repr(ex))
        print(f"  ({nome}: {time.time() - t0:.1f} s)", flush=True)
        esito_p.write_text(json.dumps(ESITO, indent=1, default=str))
    print("ESITO:", "TUTTO VERDE" if ok else "QUALCOSA NON TORNA", "->", esito_p, flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
