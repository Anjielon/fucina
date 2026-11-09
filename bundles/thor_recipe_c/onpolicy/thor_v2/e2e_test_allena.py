#!/usr/bin/env python3
"""Prove di `e2e_allena.py`, `e2e_esporta.py`, `e2e_fedelta.py` (26/9/2026), SOLO CPU (la GPU e' del teacher):
i test del prototipo (T4, T7, T8, T9, T10) portati sui moduli DEFINITIVI e sui byte VERI di Thor-T
(`Thor-v2-ThorT-scor150.gguf`, strati 0 = GDN e 3 = attenzione piena, tutti i 256 esperti), piu' le prove nuove.

Uso (limiti del flusso: 8 core, 12G, nice 10):
    systemd-run --user --scope -p CPUQuota=800% -p MemoryMax=12G nice -n 10 env CUDA_VISIBLE_DEVICES= \\
        HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=8 @@ROOT:venv@@/bin/python3 e2e_test_allena.py [A1 A5 ...]
    esito: $E2E_ALLENA_PROVE/esito_test_allena[_<prove>].json (default @@ROOT:lavoro@@/e2e/prove_allena)

Il MAESTRO delle prove e' SINTETICO e noto: lo studente stesso con theta* (spostamento per famiglia + rumore),
nascosto finale salvato in bf16 come fa il teacher vero. Cosi' la KL a theta = 0 e' > 0 e il ciclo DEVE
scendere verso theta*: si prova la meccanica (codici fermi, scale fp16 esatte, G1, ripresa al bit, held-out,
arresto, blocchi rotanti), non il guadagno su Thor. Dichiarato: righe da 128 token (il codice non dipende da T)
e, SOLO nelle prove del ciclo, la testa ristretta alle prime 8192 righe del vocabolario vero (la KL a vocabolario
INTERO, 248.320, e' provata a parte in A2 con le due teste vere).

  A1  KLFusa == riferimento autograd (float64, gradcheck; pezzi che non dividono i token) e CE della valutazione
  A2  teste vere: testa del teacher sul dispositivo == bf16 -> fp32 dei byte (sha256 uguale); KLFusa a
      vocabolario intero (student Q5_K + teacher bf16 originale) == autograd non a pezzi
  A3  (T7) cancello G1 su strato vero, ricetta B (indietro fp16): K dalla regola, dY azzerati, confronto con
      l'indietro fp32 (nulli / segno sbagliato), primo passo pieno >= 99%; e senza K la quota di dY a zero
  A4  (T8) ciclo: codici identici al bit (sha256 prima/dopo, nel file e in memoria), unici parametri theta,
      scale del migliore == fp16(d0 exp theta) al bit, segni, zeri nuovi 0, held-out che SCENDE
  A5  (T9) ripresa: 3 passi + ricarica da disco (studente NUOVO) + 3 passi == 6 filati, al bit (theta, Adam,
      storia, held-out, migliore e la sua istantanea su disco)
  A6  logica pura: arresto (2 salite con train in discesa), ordine delle epoche, blocchi consumabili, bersagli,
      regola di K
  A7  held-out per dominio: metrica = media dei domini SENZA «wiki» == ricalcolo indipendente riga per riga
  A8  (R3) invarianza di scala: K * perdita con eps * K == perdita con eps (indietro fp32), 3 passi
  A9  blocchi rotanti: si consuma SOLO dopo una ripresa che copre il blocco; attesa del teacher
  A10 (T10) export su GGUF con i tensori VERI (strati 0 e 3): file intero verificato, codice C di llama.cpp == trit
      x d nuove, ricarica al bit, logit al bit dello studente ricaricato SOLO dal file; manomissione scoperta;
      cartelle vietate
  A11 fedelta': strumento `e2e_dump_logit` sul modello minuscolo (CPU) letto da `esegui_strumento`; metriche
      (errore per token, accordo top-k, KL dei logit) contro formule dirette; verdetto F0b; rho; F0a
  A12 adattatore della cache VERA del teacher (sola lettura): id == corpus, forma e tipo di h, domini, modo
  A13 (solo se chiesta) export a GRANDEZZA VERA: 120 tensori, copia da 8,69 GiB su @@ROOT:models@@, tempi per fase
  A14 fase 2 PV nel ciclo (strato 0 vero): ripresa al bit con i registri dei codici
  27/9 CAKLD (BitDistiller Eq. 5), SOLO CPU e senza file su /mnt (studente finto `StudenteFinto`):
  A15 gradiente CAKLD fuso a pezzi == autograd float64 (V=1000, P=64, gamma 0 / 0,3 / 1), errore <= 1e-10
  A16 gamma=None AL BIT come il codice del commit a1f23cb: kl_pezzo, KLFusa e il ciclo intero (allena)
  A17 gamma misurato == media di p_T(y vero) sui bersagli (ricalcolo float64)
  A18 ciclo cakld: gamma misurato una volta e salvato, ripresa al bit con lo stesso gamma, held-out KL
      avanti/inversa/CAKLD, metrica = macro CAKLD, ripresa kl -> cakld e gamma deciso registrate
  A19 (strato 0 VERO, legge il GGUF) finestra di accumulo PV (pv_passi_accumulo=4) nel ciclo, ripresa a meta'
      finestra al bit (era A15 nel ramo pv-accumulo-wip, rinominata al merge)
  A20 (studente finto, senza /mnt) sonda del passo V con la perdita della corsa (kl al bit come c56df09, cakld ==
      fp64; passo_v tiene/annulla secondo la perdita della corsa) e top1_uguale nel held-out (macro e per dominio)
  27/9 velocita' (studente finto, senza /mnt):
  A21 rumore del gradiente (McCandlish app. A): norme^2 per micro-lotto e del passo (anche precondizionate con sqrt(v)
      di Adam) e stime |G|^2, tr(Sigma), B_simple contro i gradienti per micro-lotto calcolati a parte (fp64, <= 1e-10);
      theta.grad e ciclo intero identici al bit con e senza la misura
  A22 default identico al bit a d22c2cf (anche i file del migliore); EMA di theta (ricorrenza, metrica_ema, istantanee
      migliore_ema_* e migliore_*/ema, ripresa al bit); held-out rapido con migliore solo sulle misure piene; --beta2
  A28 (studente finto, senza /mnt) finestra di accumulo PV nel CICLO come A19 (StatoV vero, motore/salti finti):
      ripresa a meta' finestra salvata e riletta al bit, decisioni ai passi 4 e 6 anche con fermata e ripresa
      (la fermata `ferma_a` non chiude la finestra), theta/contatori/esiti V al bit
  A29 (studente finto, senza /mnt) V-Bop a gruppo (--pv-modo vbop, curvatura ogni 2) e rho adattivo nel CICLO:
      stato V-Bop nella ripresa (sha) riletto al bit, curvatura ai passi V alterni, r/rho_dopo negli esiti, ripresa
      al bit; configurazioni incompatibili rifiutate
  A30 (studente finto, senza /mnt) PASSAGGIO della corsa vera da accumulo 4 (finestra a meta' nella ripresa) a V-Bop alla
      ripresa: statistiche della finestra scartate (mai lette: D2 senza il file == D1 al bit), V-Bop vuoto, rho tarato
      ripreso, decisioni a ogni passo dopo, niente accumulo_pv_* rimasti; varianti vbop e vbop+curvatura 2+rho adattivo
  A31 (strato 0 VERO, legge il GGUF) come A30 col codice vero del passo V (StatoBop sulle famiglie vere, salti sui byte),
      memoria dello stato = 8 B/gruppo + 12 B/esperto
  A32 (studente finto, senza /mnt; ramo v-senza-bh) --pv-accetta guadagno / --pv-z-min / --pv-z-guadagno-min nel CICLO:
      ConfigV passata a PassoV, ripresa al bit con l'opzione accesa, accensione alla ripresa registrata, rifiuti, CLI
  A33 (strato 0 VERO, legge il GGUF; ramo v-senza-bh) --pv-accetta guadagno (V-Bop, z_guadagno_min 1) col codice vero
      del passo V: ripresa al bit (theta, byte, registri, stato V-Bop) e misura salti/sonda contro bh sugli stessi dati
  A23-A26 (--allena-f32) stanno in e2e_test_f32.py (erano A21-A24 nel ramo allena-f32, rinominate al merge).
Esito del 26/9 11:4x (log_finale.txt): TUTTO VERDE, 13 prove, 30 min.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import e2e_allena as A  # noqa: E402
import e2e_esporta as X  # noqa: E402
import e2e_fedelta as F  # noqa: E402
import e2e_studente as S  # noqa: E402
import e2e_codici_scale as CS  # noqa: E402
from gguf import GGMLQuantizationType as GT, GGUFReader, GGUFWriter  # noqa: E402

DIR = Path(os.getenv("E2E_ALLENA_PROVE", "@@ROOT:lavoro@@/e2e/prove_allena"))
TEACHER_VERO = Path(os.getenv("E2E_TEACHER_DIR", "@@ROOT:lavoro@@/e2e/teacher/q8-tutte-riga"))
CORPUS_HELD = Path("@@ROOT:models@@/gguf/tony-tern/calibrazione/e2e/e2e_heldout_4096_tutte.pt")
MINUSCOLO = Path("@@ROOT:lavoro@@/e2e/teacher/collaudo/minuscolo.gguf")
torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "8")))
T_RIGA = 128
V_PROVA = 8192
ESITO: dict = {"data": time.strftime("%Y-%m-%d %H:%M:%S"), "gguf": str(S.GGUF_THOR_T), "prove": {}}
muto = lambda *a, **k: None  # noqa: E731


def registra(nome: str, ok: bool, **dati) -> bool:
    ESITO["prove"][nome] = {"ok": bool(ok), **dati}
    print(f"[{nome}] {'✅' if ok else '❌'} {json.dumps(dati, default=str)[:1500]}", flush=True)
    return bool(ok)


def rel(a, b) -> float:
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-300))


# ─────────────────────────────── dati e maestro sintetico ───────────────────────────────
_IDS = None


def righe_vere(n: int, da: int = 0) -> list:
    """n righe da T_RIGA token VERI (held-out del corpus E2E, righe consecutive tagliate a T_RIGA)."""
    global _IDS
    if _IDS is None:
        d = torch.load(CORPUS_HELD, weights_only=True)
        tutte = []
        for dom in d["nomi_domini"]:
            X_ = d["domini"][dom]
            for r in range(X_.shape[0]):
                for c in range(0, X_.shape[1] - T_RIGA + 1, 1024):
                    tutte.append((dom, X_[r, c:c + T_RIGA].numpy().astype(np.int32)))
        _IDS = tutte
    if da + n > len(_IDS):
        raise ValueError(f"righe_vere: chieste {da}..{da + n}, ce ne sono {len(_IDS)}")
    return _IDS[da:da + n]


def testa_ristretta(st, V: int = V_PROVA) -> torch.Tensor:
    from dequant_gguf_torch import _dequant_raw
    t = st._tens["output.weight"]
    return torch.from_numpy(_dequant_raw(np.asarray(t.data)[:V], t)).to(torch.float32)


def theta_stella(st, seme: int = 7) -> dict:
    g = torch.Generator().manual_seed(seme)
    spost = {"gate": 0.08, "up": -0.06, "down": 0.10}
    return {k: spost[k[1]] + 0.05 * torch.randn(p.shape, generator=g) for k, p in zip(st.chiavi(), st.parametri())}


@torch.no_grad()
def maestro_sintetico(st, righe: list, th: dict) -> list:
    """h del «teacher» = uscita dello studente con theta* (dopo output_norm), in bf16 come la cache vera."""
    for k, p in zip(st.chiavi(), st.parametri()):
        p.copy_(th[k])
    out = []
    for _, ids in righe:
        out.append(st.nascosto(torch.from_numpy(ids.astype(np.int64))[None]).to(torch.bfloat16).clone())
    for p in st.parametri():
        p.zero_()
    return out


class SorgenteLenta(A.SorgenteMemoria):
    """Il «teacher» sigilla le righe di train da `pronte` in poi solo 1 s DOPO la prima richiesta (prova dell'attesa)."""
    t_richiesta = None

    def disponibile(self, lato, i):
        ok = super().disponibile(lato, i)
        if not ok and self.pronte is not None and lato == "train" and i >= self.pronte:
            if self.t_richiesta is None:
                self.t_richiesta = time.time()
            if time.time() - self.t_richiesta > 1.0:
                self.pronte = None
                ok = super().disponibile(lato, i)
        return ok


def sorgente(st, n_train: int, n_held: int, th: dict, blocchi_train=None, domini_held=None,
             classe=A.SorgenteMemoria) -> A.SorgenteMemoria:
    tr = righe_vere(n_train, 0)
    he = righe_vere(n_held, 200)                             # 256 righe da 128: train 0.., held-out 200..
    ht, hh = maestro_sintetico(st, tr, th), maestro_sintetico(st, he, th)
    dh = domini_held or [d for d, _ in he]
    righe = {"train": [(ids, h, d, None) for (d, ids), h in zip(tr, ht)],
             "heldout": [(ids, h, dd, None) for (_, ids), h, dd in zip(he, hh, dh)]}
    return classe(righe, {"train": blocchi_train} if blocchi_train else {})


def studente(strati=(0,), ricetta="B") -> "S.Studente":
    return S.Studente(strati=list(strati), ricetta=ricetta, motore_nome="torch", device="cpu", log=muto)


def sha_blocchi(st) -> dict:
    return {f"{L}.{n}": hashlib.sha256(st.esperti(L).fam[n].blocchi.numpy().tobytes()).hexdigest()
            for L, n in st.chiavi()}


def sha_blocchi_file(strati) -> dict:
    t = {x.name: x for x in GGUFReader(str(S.GGUF_THOR_T)).tensors}
    return {f"{L}.{n}": hashlib.sha256(np.ascontiguousarray(np.asarray(t[f"blk.{L}.{S.FAMIGLIE[n]}.weight"].data))
                                       .tobytes()).hexdigest() for L in strati for n in S.FAMIGLIE}


# ─────────────────────────────────────────── A1 ───────────────────────────────────────────
def prova_a1() -> bool:
    g = torch.Generator().manual_seed(1)
    N, H, V = 13, 16, 64
    hS = torch.randn(N, H, generator=g, dtype=torch.float64, requires_grad=True)
    WS = torch.randn(V, H, generator=g, dtype=torch.float64) * 0.5
    hT = torch.randn(N, H, generator=g, dtype=torch.float64)
    WT = torch.randn(V, H, generator=g, dtype=torch.float64) * 0.5
    ok_gc = torch.autograd.gradcheck(lambda h: A.KLFusa.apply(h, WS, hT, WT, 5), (hS,))
    v = A.KLFusa.apply(hS, WS, hT, WT, 5)
    (v * 3.0).backward()
    g1 = hS.grad.clone()
    hS.grad = None
    r = A.kl_diretta(hS, WS, hT, WT)
    (r * 3.0).backward()
    e_v, e_g = abs(float(v) - float(r)) / abs(float(r)), rel(g1, hS.grad)
    # CE della valutazione == formula diretta (bersagli con -1)
    b = torch.randint(0, V, (N,), generator=g)
    b[3] = -1
    out = A.kl_pezzo(hS.detach(), WS, hT, WT, False, b)
    lS = torch.log_softmax(hS.detach() @ WS.T, -1)
    lT = torch.log_softmax(hT @ WT.T, -1)
    m = b >= 0
    ce_s = float(-lS[m].gather(1, b[m][:, None]).sum())
    ce_t = float(-lT[m].gather(1, b[m][:, None]).sum())
    kl_riga = (lT.exp() * (lT - lS)).sum(-1)
    ok_ce = abs(float(out["ce_s"]) - ce_s) < 1e-10 and abs(float(out["ce_t"]) - ce_t) < 1e-10 and int(out["n_ce"]) == 12
    ok_kl = float((out["kl"] - kl_riga).abs().max()) < 1e-12
    return registra("A1_kl_fusa", ok_gc and e_v < 1e-13 and e_g < 1e-13 and ok_ce and ok_kl, gradcheck=ok_gc,
                    err_valore=e_v, err_gradiente=e_g, ce=ok_ce, kl_per_token=ok_kl)


# ─────────────────────────────────────────── A2 ───────────────────────────────────────────
def prova_a2() -> bool:
    import e2e_teacher as T
    t0 = time.time()
    WT, sha = A.testa_maestro(torch.device("cpu"))
    t = T._tensore_gguf(T.TESTA_BF16, "output.weight")
    raw = np.asarray(t.data)
    h = hashlib.sha256()
    u16 = raw.view(np.uint16).reshape(raw.shape[0], -1)
    for a in range(0, u16.shape[0], 50000):
        h.update(np.ascontiguousarray(u16[a:a + 50000]).tobytes())
    ok_sha = h.hexdigest() == sha
    ok_val = True
    for a in (0, 16383, 16384, 123457, u16.shape[0] - 5):
        rif = torch.from_numpy(T.bf16_a_f32(np.ascontiguousarray(u16[a:a + 5])))
        ok_val &= torch.equal(WT[a:a + 5], rif)
    del raw, u16
    st = studente((0,))
    WS = st.testa(torch.float32)
    g = torch.Generator().manual_seed(2)
    hS = (torch.randn(16, 2048, generator=g) * 1.5).requires_grad_(True)
    hT = (torch.randn(16, 2048, generator=g) * 1.5).to(torch.bfloat16)
    v = A.KLFusa.apply(hS, WS, hT, WT, 7)
    v.backward()
    gf = hS.grad.clone()
    hS.grad = None
    r = A.kl_diretta(hS, WS, hT, WT)
    r.backward()
    e_v, e_g = abs(float(v) - float(r)) / abs(float(r)), rel(gf, hS.grad)
    return registra("A2_teste_vere_vocabolario_intero", ok_sha and ok_val and e_v < 1e-5 and e_g < 1e-5,
                    sha256_uguale=ok_sha, valori_bf16_fp32_esatti=ok_val, V=int(WS.shape[0]), kl=float(v),
                    err_valore=e_v, err_gradiente=e_g, secondi=round(time.time() - t0, 1))


# ─────────────────────────────────────────── A3 + A4 ───────────────────────────────────────────
def _ca(**kw) -> A.ConfigAllena:
    base = dict(lr=1e-2, righe_per_passo=2, righe_per_micro=1, righe_per_micro_valuta=2, heldout_ogni=4,
                ripresa_ogni=2, pezzo_kl=100, attesa_ogni_s=0.2)
    base.update(kw)
    return A.ConfigAllena(**base)


def prova_a3_a4() -> bool:
    t0 = time.time()
    st = studente((0,))
    WS = testa_ristretta(st)
    th = theta_stella(st)
    src = sorgente(st, 24, 4, th)
    sha0_mem, sha0_file = sha_blocchi(st), sha_blocchi_file([0])
    dir_c = DIR / "a4"
    if dir_c.exists():
        shutil.rmtree(dir_c)
    ca = _ca()
    stato = A.allena(st, src, dir_c, ca, WS, WS, log=muto)
    g1, pp = stato["g1"], stato["primo_passo"]
    # A3 — senza K (perdita media) la quota di dY che va a zero in fp16 e' visibile; con K no
    car = A.Caricatore(src, st.device)
    st1 = studente((0,))
    st1.diagnostica_indietro(True)
    A.gradiente_passo(st1, car, "train", [0, 1], ca, WS, WS, 1.0)
    senza_k = A.aggrega_dy(st1.diagnostica_indietro(False))
    car.chiudi()
    del st1
    cf = g1.get("confronto_fp32", {})
    ok3 = (pp["quota_passo_pieno"] >= 0.99 and pp["quota_verso_giusto"] >= 0.99 and abs(pp["mediana_passo_su_lr"] - 1) < 0.01
           and cf.get("quota_rotti", 1) <= 1e-3 and all(v["quota_azzerati"] <= 1e-4 for v in g1["dy"].values())
           and g1["eps_adam"] == g1["eps_g1"] * g1["K"] and g1["K"] >= 1)
    registra("A3_G1_scala_K", ok3, K=g1["K"], eps_g1=g1["eps_g1"], eps_adam=g1["eps_adam"], q01=g1["q01"],
             giri=[x["K_usata"] for x in g1["giri"]], dy_con_K={f: {k: v[k] for k in ("quota_azzerati", "quota_subnormali")}
                                                                for f, v in g1["dy"].items()},
             dy_senza_K={f: {k: v[k] for k in ("quota_azzerati", "quota_subnormali")} for f, v in senza_k.items()},
             confronto_fp32=cf, primo_passo=pp)
    # A4 (T8) — codici, parametri, scale del migliore, held-out
    sha1_mem = sha_blocchi(st)
    solo_theta = all(p.dtype == torch.float32 and p.requires_grad for p in st.parametri()) and all(
        (not st.esperti(L).fam[n].blocchi.requires_grad) and st.esperti(L).fam[n].blocchi.dtype == torch.uint8
        for L, n in st.chiavi())
    mig = stato["migliore"]
    ist = A.carica_istantanea(dir_c / mig["dir"])
    theta_m = torch.load(dir_c / mig["dir"] / "theta.pt", weights_only=True)
    uguali = segni = True
    zeri = cambiate = 0
    for L, n in st.chiavi():
        fam = st.esperti(L).fam[n]
        ric = (fam.d0 * torch.exp(theta_m[f"{L}.{n}"])).to(torch.float16).reshape(-1).numpy()
        es = ist["scale"][f"blk.{L}.{S.FAMIGLIE[n]}.weight"].reshape(-1)
        uguali &= np.array_equal(ric.view(np.uint16), es.view(np.uint16))
        d0 = fam.d0.reshape(-1).numpy()
        segni &= bool((np.sign(es.astype(np.float32)) == np.sign(d0)).all())
        zeri += int(((es == 0) & (d0 != 0)).sum())
        cambiate += int((es.view(np.uint16) != d0.astype(np.float16).view(np.uint16)).sum())
    h = stato["storia_heldout"]
    ok4 = (sha1_mem == sha0_mem and all(sha0_mem[k] == sha0_file[k] for k in sha0_mem) and solo_theta and uguali
           and segni and zeri == 0 and mig["metrica"] < 0.9 * h[0]["metrica"] and stato["passo"] == 12)
    return registra("A4_ciclo_T8", ok3 and ok4, passi=stato["passo"], codici_identici=sha1_mem == sha0_mem,
                    codici_uguali_al_file=all(sha0_mem[k] == sha0_file[k] for k in sha0_mem), solo_theta=solo_theta,
                    scale_migliore_al_bit=uguali, segni=segni, zeri_nuovi=zeri, scale_cambiate=cambiate,
                    heldout=[(x["passo"], round(x["metrica"], 6)) for x in h], migliore=mig["passo"],
                    riduzione=f"{100 * (1 - mig['metrica'] / h[0]['metrica']):.1f}%",
                    cambi_instr=stato["cambi_instradamento"], secondi=round(time.time() - t0, 1))


# ─────────────────────────────────────────── A5 ───────────────────────────────────────────
def _senza_tempi(stato: dict) -> dict:
    s = json.loads(json.dumps(stato, default=str))
    for k in ("attesa_teacher_s", "tempi_ultimi"):
        s.pop(k, None)
    if s.get("g1"):
        s["g1"].pop("secondi", None)
    return s


def prova_a5() -> bool:
    t0 = time.time()
    strati = (0, 3)
    base = studente(strati)
    WS = testa_ristretta(base)
    th = theta_stella(base)
    src = sorgente(base, 12, 3, th)
    del base
    ca = _ca(heldout_ogni=2, ripresa_ogni=3)
    d_f, d_r = DIR / "a5_filata", DIR / "a5_ripresa"
    for d in (d_f, d_r):
        if d.exists():
            shutil.rmtree(d)
    st_f = studente(strati)
    filata = A.allena(st_f, src, d_f, ca, WS, WS, log=muto)
    st_a = studente(strati)
    A.allena(st_a, src, d_r, ca, WS, WS, ferma_a=3, log=muto)
    del st_a
    st_b = studente(strati)                                   # processo «nuovo»: tutto dal disco
    ripresa = A.allena(st_b, src, d_r, ca, WS, WS, riprendi=True, log=muto)
    th_ok = all(torch.equal(a, b) for a, b in zip(st_f.parametri(), st_b.parametri()))
    rf = A.carica_ripresa(d_f, "cpu")
    rr = A.carica_ripresa(d_r, "cpu")
    so, sr = rf["opt"]["state"], rr["opt"]["state"]
    adam = all(torch.equal(so[i][n], sr[i][n]) for i in so for n in ("exp_avg", "exp_avg_sq", "step"))
    stati = _senza_tempi(filata) == _senza_tempi(ripresa)
    mf, mr = filata["migliore"], ripresa["migliore"]
    zf = np.load(d_f / mf["dir"] / "scale.npz")
    zr = np.load(d_r / mr["dir"] / "scale.npz")
    ist = mf["passo"] == mr["passo"] and all(np.array_equal(zf[n], zr[n]) for n in zf.files)
    return registra("A5_ripresa_al_bit_T9", th_ok and adam and stati and ist and filata["passo"] == 6,
                    theta=th_ok, adam=adam, stato_del_ciclo=stati, istantanea_migliore=ist, passi=filata["passo"],
                    migliore=mf["passo"], riprese_su_disco=sorted(p.name for p in d_r.glob("ripresa_*.pt")),
                    secondi=round(time.time() - t0, 1))


# ─────────────────────────────────────────── A6 ───────────────────────────────────────────
def prova_a6() -> bool:
    h = lambda m, t: {"metrica": m, "kl_train_intervallo": t}  # noqa: E731
    casi = {
        "due_salite_train_giu": (A.regola_arresto([h(1.0, 5), h(1.1, 4), h(1.2, 3)], 2), True),
        "una_salita": (A.regola_arresto([h(1.0, 5), h(0.9, 4), h(1.2, 3)], 2), False),
        "salite_ma_train_su": (A.regola_arresto([h(1.0, 5), h(1.1, 6), h(1.2, 7)], 2), False),
        "poche_misure": (A.regola_arresto([h(1.0, 5), h(1.1, 4)], 2), False),
    }
    ok = all(v == atteso for v, atteso in casi.values())
    righe = list(range(10))
    o0, o1, o1b = A.ordine_epoca(righe, 0, 5), A.ordine_epoca(righe, 1, 5), A.ordine_epoca(righe, 1, 5)
    ok_ord = o0 == righe and o1 == o1b and sorted(o1) == righe and o1 != righe
    src = A.SorgenteMemoria({"train": [None] * 6}, {"train": [("b0", 0, 2), ("b1", 2, 4), ("b2", 4, 6)]})
    ok_bl = A.blocchi_consumabili(src, {0, 1, 2}) == ["b0"] and A.blocchi_consumabili(src, set(range(6))) == ["b0", "b1", "b2"]
    ids = torch.tensor([[5, 6, 7, 8, 9]])
    iz = torch.tensor([[1, 0, 1, 0, 0]], dtype=torch.int32)
    ok_b = A._bersagli(ids, None).tolist() == [6, 7, 8, 9, -1] and A._bersagli(ids, iz).tolist() == [6, -1, 8, 9, -1]
    r = A.regola_k(3.1e-11)
    ok_k = r["eps_g1"] == 3.1e-14 and r["K"] == 2.0 ** round(np.log2(1e-8 / 3.1e-14)) and r["eps_adam"] == r["eps_g1"] * r["K"]
    r2 = A.regola_k(1.0)
    ok_k &= r2["eps_g1"] == 1e-8 and r2["K"] == 1.0
    return registra("A6_logica", ok and ok_ord and ok_bl and ok_b and ok_k, arresto={k: v[0] for k, v in casi.items()},
                    ordine=ok_ord, blocchi=ok_bl, bersagli=ok_b, regola_k=ok_k, K_esempio=r["K"])


# ─────────────────────────────────────────── A7 ───────────────────────────────────────────
def prova_a7() -> bool:
    st = studente((0,))
    WS = testa_ristretta(st)
    th = theta_stella(st)
    src = sorgente(st, 2, 6, th, domini_held=["edu", "edu", "cot", "wiki", "it_chat", "wiki"])
    car = A.Caricatore(src, st.device)
    v = A.valuta(st, car, list(range(6)), _ca(righe_per_micro_valuta=4), WS, WS)
    car.chiudi()
    per_riga = []
    with torch.no_grad():
        for ids, hh, dom, _ in src.r["heldout"]:
            hs = st.nascosto(torch.from_numpy(ids.astype(np.int64))[None])
            per_riga.append((dom, float(A.kl_diretta(hs, WS, hh, WS)) / len(ids)))
    dom = {}
    for d, x in per_riga:
        dom.setdefault(d, []).append(x)
    kd = {d: float(np.mean(x)) for d, x in dom.items()}
    metrica = float(np.mean([kd[d] for d in kd if d != "wiki"]))
    # fp32 (logit e log-softmax): la KL piccola e' differenza di numeri grandi, 1e-5 relativo e' il rumore atteso;
    # la prova e' sull'aggregazione (domini, esclusione di «wiki», media macro), non sull'aritmetica (A1, A2)
    ok = abs(v["metrica"] - metrica) / metrica < 1e-4 and "wiki" not in v["domini_metrica"] and \
        all(abs(v["kl_per_dominio"][d] - kd[d]) / kd[d] < 1e-4 for d in kd) and set(v["domini_metrica"]) == set(kd) - {"wiki"}
    return registra("A7_heldout_per_dominio", ok, metrica=v["metrica"], ricalcolo=metrica, per_dominio=v["kl_per_dominio"],
                    domini_metrica=v["domini_metrica"], ce_studente=v["ce_studente"], ce_maestro=v["ce_maestro"])


# ─────────────────────────────────────────── A8 ───────────────────────────────────────────
def prova_a8() -> bool:
    st_a, st_b = studente((0,), "B32"), studente((0,), "B32")
    WS = testa_ristretta(st_a)
    src = sorgente(st_a, 6, 1, theta_stella(st_a))
    ca = _ca(lr=3e-3)
    eps_g1 = 2.0 ** -40
    K = 2.0 ** round(np.log2(1e-8 / eps_g1))
    oa = torch.optim.AdamW(st_a.parametri(), lr=ca.lr, betas=(0.9, 0.95), eps=eps_g1 * K, weight_decay=0.0)
    ob = torch.optim.AdamW(st_b.parametri(), lr=ca.lr, betas=(0.9, 0.95), eps=eps_g1, weight_decay=0.0)
    car = A.Caricatore(src, st_a.device)
    for t in range(3):
        righe = [2 * t, 2 * t + 1]
        for st, o, k in ((st_a, oa, K), (st_b, ob, 1.0)):
            o.zero_grad(set_to_none=True)
            A.gradiente_passo(st, car, "train", righe, ca, WS, WS, k)
            o.step()
    car.chiudi()
    diff = max(float((a - b).abs().max()) for a, b in zip(st_a.parametri(), st_b.parametri())) / ca.lr
    sa, sb = st_a.scale_fp16(), st_b.scale_fp16()
    n = sum(v.numel() for v in sa.values())
    uguali = sum(int((sa[k].view(torch.int16) == sb[k].view(torch.int16)).sum()) for k in sa)
    return registra("A8_invarianza_scala_K", diff < 1e-4 and uguali / n > 0.9999, K=K, max_diff_theta_su_lr=diff,
                    scale_fp16_identiche=f"{uguali}/{n}")


# ─────────────────────────────────────────── A9 ───────────────────────────────────────────
def prova_a9() -> bool:
    st = studente((0,))
    WS = testa_ristretta(st)
    blocchi = [("train/b0", 0, 4), ("train/b1", 4, 8), ("train/b2", 8, 12)]
    src = sorgente(st, 12, 2, theta_stella(st), blocchi_train=blocchi, classe=SorgenteLenta)
    registro = []
    orig = src.consuma

    def consuma(nome):
        man = json.loads((dir_c / "ripresa_ultima.json").read_text())
        registro.append((nome, man["passo"]))
        orig(nome)
    src.consuma = consuma
    src.pronte = 2                                            # il teacher ha sigillato solo 2 righe
    dir_c = DIR / "a9"
    if dir_c.exists():
        shutil.rmtree(dir_c)
    ca = _ca(rotazione=True, ripresa_ogni=3, heldout_ogni=6)
    stato = A.allena(st, src, dir_c, ca, WS, WS, log=muto)
    # blocco b consumato solo quando la ripresa al momento copriva tutte le sue righe (passo * 2 >= fine)
    coperti = all(passo * ca.righe_per_passo >= dict((n, b) for n, _, b in blocchi)[nome] for nome, passo in registro)
    ok = (coperti and [n for n, _ in registro] == ["train/b0", "train/b1", "train/b2"] and stato["attesa_teacher_s"] > 0.5
          and stato["consumati"] == ["train/b0", "train/b1", "train/b2"] and stato["passo"] == 6)
    return registra("A9_blocchi_rotanti", ok, consumi=registro, attesa_teacher_s=round(stato["attesa_teacher_s"], 2),
                    passi=stato["passo"])


# ─────────────────────────────────────────── A10 ───────────────────────────────────────────
def gguf_ridotto(p: Path, strati=(0, 3)) -> None:
    """GGUF coi tensori VERI (byte identici) degli strati `strati` + output_norm, output e token_embd di Thor-T."""
    r = GGUFReader(str(S.GGUF_THOR_T))
    w = GGUFWriter(str(p), arch="qwen35moe")
    for t in r.tensors:
        nome = t.name
        if not (any(nome.startswith(f"blk.{L}.") for L in strati) or nome in ("output_norm.weight", "output.weight",
                                                                                "token_embd.weight")):
            continue
        dati = np.ascontiguousarray(np.asarray(t.data))
        if dati.dtype != np.uint8:                         # byte grezzi: la forma logica la ricava il writer
            dati = dati.view(np.uint8)
        w.add_tensor(nome, dati, raw_dtype=GT[t.tensor_type.name])
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def prova_a10() -> bool:
    t0 = time.time()
    d = DIR / "a10"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    src = d / "ridotto.gguf"
    gguf_ridotto(src)
    # i byte dei tensori copiati sono quelli di Thor-T
    ra, rb = {t.name: t for t in GGUFReader(str(S.GGUF_THOR_T)).tensors}, {t.name: t for t in GGUFReader(str(src)).tensors}
    copia_ok = all(np.array_equal(np.asarray(ra[n].data), np.asarray(rb[n].data)) and ra[n].tensor_type == rb[n].tensor_type
                   for n in rb)
    del ra, rb
    st = S.Studente(gguf=src, strati=[0, 3], ricetta="B", motore_nome="torch", device="cpu", log=muto)
    g = torch.Generator().manual_seed(11)
    with torch.no_grad():
        for p in st.parametri():
            p.copy_(0.05 * torch.randn(p.shape, generator=g))
    nome = A.salva_istantanea(d, st, 7, {"prova": True})
    ist = A.carica_istantanea(d / nome)
    theta = torch.load(d / nome / "theta.pt", weights_only=True)
    dst = d / "esportato.gguf"
    prov = X.esporta(src, dst, ist["scale"], ist["meta"], theta=theta, strati_ricarica=(0, 3), device="cpu", log=muto)
    # T10: studente ricaricato SOLO dal file (theta = 0) == studente allenato (theta), logit al bit su id veri
    sf = S.Studente(gguf=dst, strati=[0, 3], ricetta="B", motore_nome="torch", device="cpu", log=muto)
    WS = testa_ristretta(st)
    ids = torch.from_numpy(np.stack([x for _, x in righe_vere(2, 40)]).astype(np.int64))
    with torch.no_grad():
        za = st.nascosto(ids) @ WS.T
        zb = sf.nascosto(ids) @ WS.T
    logit_ok = bool(torch.equal(za, zb))
    # manomissione di un byte di CODICE: la verifica del file intero deve accorgersene
    man = d / "manomesso.gguf"
    shutil.copyfile(dst, man)
    pos = [t for t in GGUFReader(str(man)).tensors if t.name == "blk.3.ffn_up_exps.weight"][0].data_offset + 5
    with open(man, "r+b") as f:
        f.seek(pos)
        b = f.read(1)
        f.seek(pos)
        f.write(bytes([b[0] ^ 1]))
    try:
        CS.verifica_patch(src, man, sorted(ist["scale"]), pezzo=1 << 24)
        scoperta = False
    except RuntimeError as e:
        scoperta = "FUORI dai campi d" in str(e)
    man.unlink()
    vietati = []
    for p in (Path("/tmp/x.gguf"), Path("@@ROOT:home@@/x.gguf")):
        try:
            X.controlla_destinazione(src, p)
            vietati.append((str(p), "accettata"))
        except (ValueError, OSError) as e:
            vietati.append((str(p), type(e).__name__))
    ok_v = all(x[1] != "accettata" for x in vietati)
    ric = prov["ricarica"]
    ok = (copia_ok and logit_ok and scoperta and ok_v and prov["controllo_scale"]["totali"]["zeri_nuovi"] == 0
          and all(v["scale_al_bit"] and v["uscita_al_bit"] for v in ric.values())
          and all(v["codice_c_al_bit"] for v in prov["controllo_scale"]["per_tensore"].values())
          and prov["sorgente"]["byte"] == prov["esportato"]["byte"])
    return registra("A10_export_T10", ok, byte=prov["esportato"]["byte"], copia_tensori_veri=copia_ok,
                    scale=prov["controllo_scale"]["totali"], istogramma=prov["controllo_scale"]["istogramma_rel"],
                    ricarica=ric, logit_ricaricato_al_bit=logit_ok, manomissione_scoperta=scoperta, cartelle_vietate=vietati,
                    secondi=round(time.time() - t0, 1))


# ─────────────────────────────────────────── A11 ───────────────────────────────────────────
def prova_a11() -> bool:
    t0 = time.time()
    d = DIR / "a11"
    if d.exists():
        shutil.rmtree(d)
    ids = righe_vere(1, 60)[0][1][:64]
    u = F.esegui_strumento(MINUSCOLO, d / "seq", ids, None, vulkan=False, thread=4)
    n_l = sum(1 for k in u._seq(0)["tensori"] if k.startswith("l_out-"))
    x = torch.from_numpy(np.array(u.tensore(0, "l_out-1"), dtype=np.float32))
    tk = torch.from_numpy(np.array(u.tensore(0, "ffn_moe_topk-1"), dtype=np.int32))
    pos, zl = u.logit(0)
    forme = x.shape == (64, 256) and tk.shape == (64, 4) and zl.shape == (64, 248320) and len(pos) == 64 and n_l >= 4
    e0 = F.errore_per_token(x, x)
    e1 = F.errore_per_token(x * 1.02, x)
    tk2 = tk.clone()
    tk2[5] = torch.flip(tk2[5], [0])                          # stesso insieme, altro ordine
    tk3 = tk.clone()
    tk3[7, 0] = next(e for e in range(16) if e not in tk[7].tolist())
    ok_m = float(e0.max()) == 0.0 and float((e1 - 0.02).abs().max()) < 1e-6 and F.accordo_topk(tk, tk2) == 1.0 and \
        abs(F.accordo_topk(tk, tk3) - 63 / 64) < 1e-12
    # KL dei logit: formula diretta su numeri piccoli
    g = torch.Generator().manual_seed(3)
    zlm = np.asarray(torch.randn(9, 50, generator=g) * 2, dtype=np.float32)
    hS, WS = torch.randn(9, 8, generator=g), torch.randn(50, 8, generator=g)
    hT, WT = torch.randn(9, 8, generator=g).to(torch.bfloat16), torch.randn(50, 8, generator=g)
    k = F.kl_logit(zlm, hS, WS, hT, WT, pezzo=4)
    ll = torch.log_softmax(torch.from_numpy(zlm).double(), -1)
    ls = torch.log_softmax((hS @ WS.T).double(), -1)
    lt = torch.log_softmax((hT.float() @ WT.T).double(), -1)
    kd = lambda a, b: (a.exp() * (a - b)).sum(-1)  # noqa: E731
    ok_kl = all(float((k[n] - r).abs().max()) < 1e-12 for n, r in
                (("llama_torch", kd(ll, ls)), ("teacher_llama", kd(lt, ll)), ("teacher_torch", kd(lt, ls))))
    # verdetto F0b su strutture sintetiche: strato 2 oltre il 3%
    per_seq = []
    for s_ in range(2):
        strati = {L: {"errore": {"mediana": 0.01}, "_err": torch.full((10,), 0.01 if L != 2 else 0.05),
                      "accordo_topk": 0.9} for L in range(4)}
        per_seq.append({"strati": strati, "_kl_grezze": {"llama_torch": torch.full((10,), 0.001),
                                                          "teacher_llama": torch.full((10,), 0.1),
                                                          "teacher_torch": torch.full((10,), 0.1)}})
    v = F.verdetto_f0b(per_seq)
    ok_v = v["primo_strato_del_salto"] == 2 and not v["passa"] and v["criteri"]["kl_llama_torch_<=_1/10_kl_teacher_llama"]
    for s_ in per_seq:
        s_["strati"][2]["_err"] = torch.full((10,), 0.02)
    ok_v &= F.verdetto_f0b(per_seq)["passa"]
    r = F.rho({"llama_prima": 0.5, "llama_dopo": 0.41, "torch_prima": 0.5, "torch_dopo": 0.4})
    ok_r = abs(r["rho"] - 0.9) < 1e-12 and r["passa"] and not F.rho({"llama_prima": 0.5, "llama_dopo": 0.48,
                                                                    "torch_prima": 0.5, "torch_dopo": 0.4})["passa"]
    # F0a: criteri sulle uscite (sintetiche) di e2e_teacher f0a-cache / f0a-ppl
    dt = d / "teacher_finto"
    (dt / "f0a").mkdir(parents=True)
    (dt / "f0a" / "ppl.json").write_text(json.dumps({"q8_meno_bf16_in_sigma": 0.3, "identita_4_decimali": True}))
    (dt / "f0a" / "cache.json").write_text(json.dumps({"per_riga": [
        {"h_cache|testa_bf16": {"kl_diretti_vs_ricalcolati_media": 1e-3}, "h_cache|testa_q8": {"kl_diretti_vs_ricalcolati_media": 1e-5}},
        {"h_cache|testa_bf16": {"kl_diretti_vs_ricalcolati_media": 3e-3}, "h_cache|testa_q8": {"kl_diretti_vs_ricalcolati_media": 1e-5}}]}))
    ok_a = F.f0a_verdetto(dt, 0.1)["passa"] and not F.f0a_verdetto(dt, 0.02)["passa"]
    shutil.rmtree(d)
    return registra("A11_fedelta", forme and ok_m and ok_kl and ok_v and ok_r and ok_a, strumento_forme=forme,
                    secondi_strumento=round(u.secondi, 1), metriche=ok_m, kl_logit=ok_kl, verdetto_f0b=ok_v, rho=ok_r,
                    f0a=ok_a, secondi=round(time.time() - t0, 1))


# ─────────────────────────────────────────── A12 ───────────────────────────────────────────
def prova_a12() -> bool:
    src = A.SorgenteCache(TEACHER_VERO)
    d = torch.load(CORPUS_HELD, weights_only=True)
    nomi = list(d["nomi_domini"])
    righe = src.righe_previste("heldout")
    disp = [i for i in righe if src.disponibile("heldout", i)]
    ok = len(righe) == 64 and len(disp) > 0 and len(src.righe_previste("train")) == src.piano["corpus"]["train"]["n_righe"]
    controllate = 0
    for i in disp[:: max(1, len(disp) // 8)]:
        di, r = d["ordine"][i].tolist()
        atteso = d["domini"][nomi[di]][r].numpy().astype(np.int32)
        h = src.h("heldout", i)
        ok &= bool(np.array_equal(src.ids("heldout", i), atteso)) and src.dominio("heldout", i) == nomi[di]
        ok &= tuple(h.shape) == (4096, 2048) and h.dtype == torch.bfloat16 and bool(torch.isfinite(h.float()).all())
        ok &= (src.inizi("heldout", i) is None) == (src.modo == "riga")
        controllate += 1
    return registra("A12_cache_vera_del_teacher", ok and controllate > 0, modo=src.modo, righe_heldout=len(righe),
                    sigillate=len(disp), controllate=controllate, train_previste=len(src.righe_previste("train")))


# ─────────────────────────────────────────── A13 ───────────────────────────────────────────
def prova_a13() -> bool:
    """Export a GRANDEZZA VERA (collaudo del percorso di produzione, non nella lista di default): tutti i 120
    tensori di esperti di Thor-T con scale fp16(d0 exp theta), theta ~ N(0, 0,05) col seme; copia da 8,69 GiB,
    verifica del file intero, codice C, ricarica degli strati 0 e 39 al bit; tempi di ogni fase. La copia si
    cancella alla fine (su @@ROOT:models@@, mai /tmp)."""
    t0 = time.time()
    d = Path(os.getenv("E2E_ALLENA_PROVE_PIENO", "@@ROOT:models@@/gguf/tony-tern/e2e_allena_prove"))
    d.mkdir(parents=True, exist_ok=True)
    dst = d / "ThorT-export-prova.gguf"
    for q in (dst, Path(str(dst) + ".provenienza.json"), dst.with_name(dst.name + ".parziale")):
        if q.exists():
            q.unlink()
    r = GGUFReader(str(S.GGUF_THOR_T))
    g = torch.Generator().manual_seed(26)
    scale, theta = {}, {}
    for L in range(40):
        for n, nome_t in S.FAMIGLIE.items():
            t = next(x for x in r.tensors if x.name == f"blk.{L}.{nome_t}.weight")
            tipo = t.tensor_type.name
            nb = CS.LAYOUT[tipo][0]
            raw = np.asarray(t.data)
            E, R = raw.shape[0], raw.shape[1]
            d0 = CS.scale_da_blocchi(torch.from_numpy(np.ascontiguousarray(raw).reshape(-1, nb)), tipo).float().reshape(E, R, -1)
            th = 0.05 * torch.randn(d0.shape, generator=g)
            theta[f"{L}.{n}"] = th
            scale[t.name] = (d0 * torch.exp(th)).to(torch.float16).numpy().reshape(-1, CS.LAYOUT[tipo][3])
    del r
    t1 = time.time()
    prov = X.esporta(S.GGUF_THOR_T, dst, scale, {"prova": "A13", "seme": 26}, theta=theta, strati_ricarica=(0, 39),
                     device="cpu", log=lambda m: print("   " + m, flush=True))
    t2 = time.time()
    tot = prov["controllo_scale"]["totali"]
    ok = (tot["zeri_nuovi"] == 0 and tot["segni_cambiati"] == 0 and len(prov["scale_cambiate_per_tensore"]) == 120
          and all(v["codice_c_al_bit"] for v in prov["controllo_scale"]["per_tensore"].values())
          and all(v["scale_al_bit"] and v["uscita_al_bit"] for v in prov["ricarica"].values())
          and prov["sorgente"]["byte"] == prov["esportato"]["byte"])
    esito = registra("A13_export_grandezza_vera", ok, byte=prov["esportato"]["byte"], scale=tot,
                     istogramma=prov["controllo_scale"]["istogramma_rel"], ricarica=prov["ricarica"],
                     secondi_preparazione=round(t1 - t0, 1), secondi_esporta=round(t2 - t1, 1), tempi=prov["tempi"],
                     sha_sorgente=prov["sorgente"]["sha256"][:16], sha_esportato=prov["esportato"]["sha256"][:16])
    dst.unlink()
    Path(str(dst) + ".provenienza.json").unlink()
    return esito



# ─────────────────────────────────────────── A14 ───────────────────────────────────────────
def prova_a14() -> bool:
    """27/9 fase 2 PV innestata nel ciclo: strato 0 VERO, maestro = theta* + 0,5% dei codici spostati di un livello
    (lo studente parte coi codici originali). (a) corsa intera P+V; (b) la stessa interrotta al passo 3 e ripresa:
    theta, byte dei codici, registri e contatori PV identici AL BIT; (c) byte dello studente == byte del file +
    rigioco dei registri della ripresa; (d) il migliore porta i suoi registri; (e) corsa con PV spento: codici intatti."""
    import e2e_pv_codici as PV
    t0 = time.time()
    st_t = studente((0,))
    for f in st_t.esperti(0).fam.values():
        PV.famiglia_scrivibile(f)
    g = torch.Generator().manual_seed(14)
    piantati = 0
    for n, f in st_t.esperti(0).fam.items():
        T_ = CS.trit_da_blocchi(f.blocchi.reshape(-1, f.nb), f.tipo).view(f.E, f.R, f.C)
        k = int(0.005 * f.R * f.C)
        for e in range(f.E):
            pos = torch.randperm(f.R * f.C, generator=g)[:k]
            r_, c_ = pos // f.C, pos % f.C
            t_ = T_[e, r_, c_].to(torch.int32)
            nuovo = torch.where(t_ == 0, torch.randint(0, 2, (k,), generator=g, dtype=torch.int32) * 2 - 1,
                                torch.zeros_like(t_))
            piantati += PV.salta_byte(f.blocchi, f.tipo, torch.full((k,), e), r_, c_, t_, nuovo)
    th = theta_stella(st_t)
    passi = 6
    src = sorgente(st_t, 6 * passi, 4, th)
    del st_t
    ca = _ca(pv_codici=True, righe_per_passo=6, righe_per_micro=1, heldout_ogni=2, ripresa_ogni=2, pv_rho=0.05)

    def corsa(dir_c, **kw):
        st = studente((0,))
        WS = testa_ristretta(st)
        return st, A.allena(st, src, dir_c, ca, WS, WS, log=muto, **kw)
    dA, dB = DIR / "a14_intera", DIR / "a14_ripresa"
    for d in (dA, dB):
        if d.exists():
            shutil.rmtree(d)
    stA, sA = corsa(dA)
    _, sB1 = corsa(dB, ferma_a=3)
    stB, sB = corsa(dB, riprendi=True)
    theta_uguale = all(torch.equal(a, b) for a, b in zip(stA.parametri(), stB.parametri()))
    byte_uguali = all(torch.equal(stA.esperti(0).fam[n].blocchi, stB.esperti(0).fam[n].blocchi)
                      for n in stA.esperti(0).fam)
    pvA, pvB = sA["pv"], sB["pv"]
    contatori = all(pvA[k] == pvB[k] for k in ("registri", "salti_tenuti", "passi_v", "rho", "sha"))
    # (c) file + rigioco dei registri della ULTIMA ripresa == byte in memoria
    regs = PV.Registro.carica_lista(dA / pvA["file"], pvA["sha"])
    st_f = studente((0,))
    fams_f = {(0, n): PV.famiglia_scrivibile(f) for n, f in st_f.esperti(0).fam.items()}
    PV.rigioca(fams_f, regs)
    rigioco = all(torch.equal(fams_f[(0, n)].blocchi, stA.esperti(0).fam[n].blocchi) for n in stA.esperti(0).fam)
    # (d) il migliore ha i suoi registri (prefisso della lista finale)
    mig = sA["migliore"]
    reg_m = PV.Registro.carica_lista(dA / mig["dir"] / "registri_pv.npz", mig["registri_sha"]) \
        if mig.get("registri_sha") else None
    migliore_ok = reg_m is not None and len(reg_m) == mig["registri"] and all(
        np.array_equal(a_.righe, b_.righe) for a_, b_ in zip(reg_m, regs[:len(reg_m)]))
    # (e) PV spento sulla stessa sorgente: codici intatti (fase 1 identica)
    dC = DIR / "a14_spento"
    if dC.exists():
        shutil.rmtree(dC)
    stC = studente((0,))
    sha0 = sha_blocchi(stC)
    sC = A.allena(stC, src, dC, _ca(righe_per_passo=6, righe_per_micro=1, heldout_ogni=2, ripresa_ogni=2),
                  testa_ristretta(stC), testa_ristretta(stC), log=muto, ferma_a=2)
    spento_intatto = sha_blocchi(stC) == sha0 and "pv" not in sC and not list(dC.glob("registri_pv_*.npz"))
    storia_v = [x.get("pv") for x in sA["storia"] if x.get("pv")]
    ok = (theta_uguale and byte_uguali and contatori and rigioco and migliore_ok and spento_intatto
          and pvA["passi_v"] == passi - 1 and sB1["passo"] == 3)
    return registra("A14_fase2_PV_nel_ciclo", ok, codici_piantati_nel_maestro=piantati,
                    ripresa_theta_al_bit=theta_uguale, ripresa_byte_al_bit=byte_uguali, contatori_uguali=contatori,
                    file_piu_rigioco_uguale_memoria=rigioco, migliore_coi_registri=migliore_ok,
                    pv_spento_codici_intatti=spento_intatto, pv=pvA,
                    passi_v=[{k: (round(v, 6) if isinstance(v, float) else v) for k, v in x.items()} for x in storia_v],
                    heldout=[(x["passo"], round(x["metrica"], 6)) for x in sA["storia_heldout"]],
                    secondi=round(time.time() - t0, 1))

# ─────────────────────────────── 27/9 CAKLD (BitDistiller Eq. 5) ───────────────────────────────
# Il codice di PRIMA (commit a1f23cb), copiato qui: riferimento AL BIT del percorso gamma=None.
BASE_PRIMA = "a1f23cb"
# 27/9 V-Bop e rho adattivo: campi nuovi della configurazione e loro default (con questi: comportamento di prima)
_CFG_VBOP_DEFAULT = {"pv_modo": "pv", "pv_gamma": 0.5, "pv_curvatura_ogni": 1, "pv_rho_adattivo": False,
                     "pv_rho_max": 0.1,
                     # 27/9 ramo v-senza-bh (--pv-accetta, --pv-z-min): ai default = i due test BH di prima
                     "pv_accetta": "bh", "pv_z_min": 0.0, "pv_z_guadagno_min": 0.0}
# 27/9 LoRA-GA (ramo lora-ga): campi nuovi della configurazione ai default (init zero = LoRA di prima)
_CFG_LORA_GA_DEFAULT = {"lora_init": "zero", "lora_ga_gamma": 16.0, "lora_ga_righe": 0}
# 27/9 --peso-token (ramo kl-pesata): campi nuovi e loro default (uniforme: comportamento di prima, al bit)
_CFG_PESO_DEFAULT = {"peso_token": "uniforme", "peso_alfa": 1.0, "peso_max": 5.0, "peso_frazione": 0.6}
# 27/9 ramo newton-theta (--ottimizzatore-theta): campi nuovi e loro default (adamw: comportamento di prima)
_CFG_SOPHIA_DEFAULT = {"ottimizzatore_theta": "adamw", "sophia_gamma": 0.01, "sophia_beta2": 0.99, "sophia_eps": 1e-15}
# 27/9 ramo integrata: i campi nuovi di TUTTI i rami uniti (kl-pesata, newton-theta, ...) ai loro default; le prove
# «default al bit contro un commit di prima dei rami» li vedono tutti come nuovi
_CFG_ACCETTA_DEFAULT = {k: _CFG_VBOP_DEFAULT[k] for k in ("pv_accetta", "pv_z_min", "pv_z_guadagno_min")}  # v-senza-bh
# 27/9 campi nati nel ramo integrata: --attribuzione (e2e_attribuzione.py, esecuzione: spenta = allenamento di prima)
_CFG_INTEGRATA_PROPRI = {"attribuzione": False, "attribuzione_righe": 2, "attribuzione_solo_ogni": 1,
                         # e --istantanea-ogni-passo (esecuzione: spenta = nessun file in piu')
                         "istantanea_ogni_passo": False, "istantanee_dir": "", "istantanee_tieni": 2,
                         "istantanee_libero_min_gib": 5.0,
                         # e --non-consumare (esecuzione: di default i blocchi si consumano come prima)
                         "consuma_blocchi": True,
                         # e (ramo dominio-strati-b) --peso-dominio / --strati-allenati (vuoti = spenti, perdita e
                         # strati di prima)
                         "peso_dominio": [], "strati_allenati": [],
                         # 27/9 ramo v-interazione-b (--pv-sonda-dosi, --pv-ricentra): spenti = sonda di prima al bit
                         "pv_sonda_dosi": False, "pv_ricentra": False, "pv_ricentra_giri": 3, "pv_dosi_frazioni": [],
                         # 28/9 --f32-ripristina (vuota = nessuna famiglia riportata al GGUF: ripresa di prima al bit)
                         "f32_ripristina": [],
                         # 28/9 --lora-ripristina (vuoto = LoRA della ripresa caricata come sempre, al bit)
                         "lora_ripristina": "",
                         # 28/9 --righe-base-da (0 = il flusso base dalla riga 0, piano di sempre al bit; EB4)
                         "righe_base_da": 0,
                         # 28/9 --densi-ternari (vuoto = nessun denso a esperto unico: studente di prima al bit)
                         "densi_ternari": [],
                         # 1/10 tt5-ottimizzato --tempi-fasi (esecuzione: solo registro)
                         "tempi_fasi": False, "scritture_asincrone": False}
# 27/9 bf16x3fix (merge in integrata): teste di default fp32
_CFG_BF16X3_DEFAULT = {"testa_kl": "fp32", "logit_maestro": "fp32"}
# 27/9 LoTA-QAF (--lota-*, ramo lota-qaf): campi nuovi della configurazione ai default (LoTA spenta: comportamento di prima)
_CFG_LOTA_DEFAULT = {"lota_rango": 0, "lota_omega": None, "lota_top_p": 0.05, "lota_top_p_fine": 1e-3,
                     "lota_top_p_coda": 1e-4, "lota_lr": 1.0, "lota_tau": 1e-9, "lota_sonda": True, "lota_init": "paper",
                     "lota_sonda_tolleranza": 0.0, "lota_sonda_finestra": 0}
# 27/9 righe a scelta multipla (ramo trainer-mc): campi nuovi e loro default (spenti: comportamento di prima, al bit)
_CFG_MC_DEFAULT = {"righe_mc": 0, "beta_risposta": 0.1, "gamma_lettere": 0.0,
                   # integrata: --righe-mc-permutate (spento: ordine del teacher mc nel primo giro, come prima)
                   "righe_mc_permutate": False}
_CFG_INTEGRATA_DEFAULT = {**_CFG_PESO_DEFAULT, **_CFG_SOPHIA_DEFAULT, **_CFG_LORA_GA_DEFAULT, **_CFG_ACCETTA_DEFAULT,
                          **_CFG_INTEGRATA_PROPRI, **_CFG_BF16X3_DEFAULT,
                          **_CFG_LOTA_DEFAULT, **_CFG_MC_DEFAULT}


def _kl_pezzo_prima(hS, WS, hT, WT, gradiente, bersagli=None):
    dt = hS.dtype
    lT = A._logit_maestro(hT, WT, dt)
    lT.sub_(torch.logsumexp(lT, -1, keepdim=True))
    lS = hS @ WS.T
    lS.sub_(torch.logsumexp(lS, -1, keepdim=True))
    out = {}
    if bersagli is not None:
        ok = bersagli >= 0
        b = bersagli.clamp(min=0).unsqueeze(-1)
        out["ce_s"] = -(lS.gather(-1, b).squeeze(-1).double() * ok).sum()
        out["ce_t"] = -(lT.gather(-1, b).squeeze(-1).double() * ok).sum()
        out["n_ce"] = ok.sum()
    pT = lT.exp()
    lT.sub_(lS).mul_(pT)
    out["kl"] = lT.sum(-1, dtype=torch.float64)
    del lT
    if gradiente:
        pS = lS.exp_()
        pS.sub_(pT)
        out["dh"] = pS @ WS
    return out


class _KLFusaPrima(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hS, WS, hT, WT, pezzo):
        tot = torch.zeros((), dtype=torch.float64, device=hS.device)
        dh = torch.empty_like(hS)
        for i in range(0, hS.shape[0], pezzo):
            r = _kl_pezzo_prima(hS[i:i + pezzo], WS, hT[i:i + pezzo], WT, True)
            tot += r["kl"].sum()
            dh[i:i + pezzo] = r["dh"]
        ctx.save_for_backward(dh)
        return tot

    @staticmethod
    def backward(ctx, g):
        (dh,) = ctx.saved_tensors
        return dh * g.to(dh.dtype), None, None, None, None


def _modulo_prima(base: str = BASE_PRIMA):
    """e2e_allena.py del commit `base` (git show), caricato come modulo a parte."""
    import importlib.util
    import subprocess
    testo = subprocess.run(["git", "-C", str(QUI), "show", f"{base}:thor_v2/e2e_allena.py"], check=True,
                           capture_output=True, text=True).stdout
    nome = "e2e_allena_prima" if base == BASE_PRIMA else f"e2e_allena_{base}"
    p = DIR / f"{nome}.py"
    p.write_text(testo)
    spec = importlib.util.spec_from_file_location(nome, p)
    m = importlib.util.module_from_spec(spec)
    sys.modules[nome] = m
    spec.loader.exec_module(m)
    return m


class _FamFinta:
    def __init__(self, forma, dtype=torch.float32):
        self.d0 = torch.ones(forma, dtype=dtype)
        self.blocchi = torch.zeros(64, dtype=torch.uint8)
        self.E = 1


class StudenteFinto:
    """Studente minuscolo con la STESSA interfaccia che `allena` usa (prove del ciclo CAKLD senza GGUF): h =
    emb[id] + tanh(emb[id] @ sum_f W_f exp(theta_f)), theta per famiglia (H x H), avanti e indietro esatti fp32."""
    FAM = ("gate", "up", "down")

    def __init__(self, V: int = 512, H: int = 16, seme: int = 3, dtype=torch.float32):
        g = torch.Generator().manual_seed(seme)
        self.emb = torch.randn(V, H, generator=g).to(dtype)
        self.W = {n: (0.3 * torch.randn(H, H, generator=g)).to(dtype) for n in self.FAM}
        self.theta = {n: torch.nn.Parameter(torch.zeros(H, H, dtype=dtype)) for n in self.FAM}
        self.fam = {n: _FamFinta((H, H), dtype) for n in self.FAM}
        self.E = 1
        self.strati = [0]
        self.device = torch.device("cpu")
        self.gguf = str(Path(__file__).resolve())
        from types import SimpleNamespace
        self.ric = SimpleNamespace(nome="finto", dtype_gemm=torch.float32, dtype_indietro=None)

    def chiavi(self):
        return [(0, n) for n in self.FAM]

    def parametri(self):
        return [self.theta[n] for n in self.FAM]

    def esperti(self, L):
        return self

    def _f(self, x):
        M = sum(self.W[n] * torch.exp(self.theta[n]) for n in self.FAM)
        return x + torch.tanh(x @ M)

    def avanti(self, ids, inizi=None):
        x = self.emb[ids.reshape(-1)]
        with torch.no_grad():
            return {"x": x, "xL": self._f(x)}

    def esce(self, xL):
        return xL * 1.0

    def indietro(self, stato, g):
        with torch.enable_grad():
            self._f(stato["x"]).backward(g)

    @torch.no_grad()
    def nascosto(self, ids, inizi=None):
        return self.esce(self.avanti(ids)["xL"])

    def controlla(self, tollera_trabocco=False):
        return {"trabocco_fp16": False, "cambi_instradamento": 0}

    def azzera_cambi(self):
        pass

    def diagnostica_indietro(self, acceso):
        return [] if acceso else [{"famiglia": n, "non_nulli": 1, "a_zero_in_fp16": 0, "subnormali_in_fp16": 0,
                                   "max_abs": 1e-3} for n in self.FAM]

    def scale_per_export(self):
        return {f"blk.0.{n}": (self.W[n] * torch.exp(self.theta[n].detach())).to(torch.float16).numpy()
                for n in self.FAM}


def sorgente_finta(V: int = 512, T: int = 32, n_train: int = 12, n_held: int = 4, seme: int = 5):
    """Maestro = lo studente finto con theta* (bf16, come la cache vera); domini held-out a, b, a, wiki."""
    st = StudenteFinto(V)
    g = torch.Generator().manual_seed(seme)
    with torch.no_grad():
        for p in st.parametri():
            p.copy_(0.3 * torch.randn(p.shape, generator=g))
    doms = ["a", "b", "a", "wiki"]
    righe = {}
    for lato, n in (("train", n_train), ("heldout", n_held)):
        rr = []
        for i in range(n):
            ids = torch.randint(0, V, (T,), generator=g).numpy().astype(np.int32)
            h = st.nascosto(torch.from_numpy(ids.astype(np.int64))[None]).to(torch.bfloat16).clone()
            rr.append((ids, h, doms[i % 4] if lato == "heldout" else "tr", None))
        righe[lato] = rr
    WS = torch.randn(V, 16, generator=g) * 0.7
    return A.SorgenteMemoria(righe, {}), WS


def _ca_finta(**kw):
    base = dict(lr=3e-2, righe_per_passo=2, righe_per_micro=1, righe_per_micro_valuta=2, heldout_ogni=2,
                ripresa_ogni=2, pezzo_kl=24, attesa_ogni_s=0.01)
    base.update(kw)
    return A.ConfigAllena(**base)


# ─────────────────────────────────────────── A15 ───────────────────────────────────────────
def prova_a15() -> bool:
    """Gradiente CAKLD fuso a pezzi == autograd (float64, V=1000, P=64, pezzi da 17 che non dividono P) per
    gamma 0, 0,3, 1 (valore, per token KL avanti/inversa/perdita, dL/dhS): errore <= 1e-10; gradcheck a 0,3."""
    g = torch.Generator().manual_seed(15)
    P, H, V = 64, 24, 1000
    hS = torch.randn(P, H, generator=g, dtype=torch.float64, requires_grad=True)
    WS = torch.randn(V, H, generator=g, dtype=torch.float64) * 0.4
    hT = torch.randn(P, H, generator=g, dtype=torch.float64)
    WT = torch.randn(V, H, generator=g, dtype=torch.float64) * 0.4
    lS = torch.log_softmax(hS.detach() @ WS.T, -1)
    lT = torch.log_softmax(hT @ WT.T, -1)
    kf_rif = (lT.exp() * (lT - lS)).sum(-1)
    ki_rif = (lS.exp() * (lS - lT)).sum(-1)
    esiti, ok = {}, True
    for gamma in (0.0, 0.3, 1.0):
        v, tf, ti = A.KLFusa.apply(hS, WS, hT, WT, 17, gamma)
        (v * 3.0).backward()
        gf = hS.grad.clone()
        hS.grad = None
        r = A.cakld_diretta(hS, WS, hT, WT, gamma)
        (r * 3.0).backward()
        gr = hS.grad.clone()
        hS.grad = None
        pz = A.kl_pezzo(hS.detach(), WS, hT, WT, True, gamma=gamma)
        # dL/dz_S dalla formula chiusa contro autograd sui logit
        z = (hS.detach() @ WS.T).requires_grad_(True)
        lz = torch.log_softmax(z, -1)
        (gamma * (lz.exp() * (lz - lT)).sum() + (1 - gamma) * (lT.exp() * (lT - lz)).sum()).backward()
        pS = lS.exp()
        dz = gamma * pS * (lS - lT - ki_rif[:, None]) + (1 - gamma) * (pS - lT.exp())
        e = {"valore": abs(float(v) - float(r)) / abs(float(r)), "grad_rel": rel(gf, gr),
             "grad_max_abs": float((gf - gr).abs().max()),
             "kl_fwd_somma": abs(float(tf) - float(kf_rif.sum())), "kl_inv_somma": abs(float(ti) - float(ki_rif.sum())),
             "per_token": max(float((pz["kl"] - kf_rif).abs().max()), float((pz["kl_inv"] - ki_rif).abs().max()),
                              float((pz["perdita"] - (gamma * ki_rif + (1 - gamma) * kf_rif)).abs().max())),
             "dh_pezzo": float((pz["dh"] * 3.0 - gr).abs().max()), "dz_formula": float((dz - z.grad).abs().max())}
        esiti[str(gamma)] = e
        ok &= all(x <= 1e-10 for x in e.values())
    h2 = hS.detach()[:9].clone().requires_grad_(True)
    ok_gc = torch.autograd.gradcheck(lambda h: A.KLFusa.apply(h, WS[:50], hT[:9], WT[:50], 4, 0.3)[0], (h2,))
    # gamma=0 == KL in avanti (valore al bit per token)
    ok_g0 = torch.equal(A.kl_pezzo(hS.detach(), WS, hT, WT, False, gamma=0.0)["kl"],
                        A.kl_pezzo(hS.detach(), WS, hT, WT, False)["kl"])
    return registra("A15_cakld_gradiente_fuso", ok and ok_gc and ok_g0, P=P, V=V, pezzo=17, errori=esiti,
                    gradcheck=ok_gc, gamma0_kl_al_bit=ok_g0)


# ─────────────────────────────────────────── A16 ───────────────────────────────────────────
def prova_a16() -> bool:
    """gamma=None identico AL BIT al codice di prima: (a) kl_pezzo (kl, dh, CE) e KLFusa (valore, gradiente) in
    fp32 (percorso vero, logit del teacher da e2e_teacher) e fp64 contro la copia del commit a1f23cb; con gamma la
    "kl" e' la stessa al bit; (b) il CICLO intero (allena, perdita kl, studente finto, 4 passi + held-out) del
    nuovo modulo == quello del modulo del commit a1f23cb (theta, Adam, storia, held-out, migliore)."""
    t0 = time.time()
    g = torch.Generator().manual_seed(16)
    ok_a, dett = True, {}
    for dt in (torch.float32, torch.float64):
        P, H, V = 70, 32, 1000
        hS = (torch.randn(P, H, generator=g) * 1.5).to(dt)
        WS = (torch.randn(V, H, generator=g) * 0.5).to(dt)
        hT = (torch.randn(P, H, generator=g) * 1.5).to(torch.bfloat16 if dt == torch.float32 else dt)
        WT = (torch.randn(V, H, generator=g) * 0.5).to(dt)
        b = torch.randint(0, V, (P,), generator=g)
        b[5] = -1
        n, o = A.kl_pezzo(hS, WS, hT, WT, True, b), _kl_pezzo_prima(hS, WS, hT, WT, True, b)
        c = A.kl_pezzo(hS, WS, hT, WT, True, b, gamma=0.3)
        u = all(torch.equal(n[k], o[k]) for k in ("kl", "dh", "ce_s", "ce_t", "n_ce")) and set(n) == set(o)
        u_c = torch.equal(c["kl"], o["kl"]) and all(torch.equal(c[k], o[k]) for k in ("ce_s", "ce_t", "n_ce"))
        x1 = hS.clone().requires_grad_(True)
        x2 = hS.clone().requires_grad_(True)
        v1 = A.KLFusa.apply(x1, WS, hT, WT, 23)
        v2 = _KLFusaPrima.apply(x2, WS, hT, WT, 23)
        (v1 * 2.0 ** 20 / 4096).backward()
        (v2 * 2.0 ** 20 / 4096).backward()
        u_f = torch.equal(v1, v2) and torch.equal(x1.grad, x2.grad) and isinstance(v1, torch.Tensor)
        dett[str(dt)] = {"kl_pezzo": u, "cakld_kl_uguale": u_c, "KLFusa": u_f}
        ok_a &= u and u_c and u_f
    # (b) ciclo intero, nuovo contro vecchio
    src, WS = sorgente_finta()
    ca = _ca_finta()
    esiti = {}
    for nome, mod in (("nuovo", A), ("prima", _modulo_prima())):
        d = DIR / f"a16_{nome}"
        if d.exists():
            shutil.rmtree(d)
        st = StudenteFinto()
        cfg = ca if mod is A else _config_per(mod, ca)
        stato = mod.allena(st, src, d, cfg, WS, WS, ferma_a=4, log=muto)
        esiti[nome] = (st, stato, mod.carica_ripresa(d, "cpu"))
    (sn, zn, rn), (sp, zp, rp) = esiti["nuovo"], esiti["prima"]
    th = all(torch.equal(a, b) for a, b in zip(sn.parametri(), sp.parametri()))
    so, sr = rn["opt"]["state"], rp["opt"]["state"]
    adam = all(torch.equal(so[i][k], sr[i][k]) for i in so for k in ("exp_avg", "exp_avg_sq", "step"))
    a_, b_ = _senza_tempi(zn), _senza_tempi(zp)
    cfg_nuovi = {k: a_["config"].pop(k) for k in set(a_["config"]) - set(b_["config"])}
    a_.pop("versione", None)
    b_.pop("versione", None)
    rumore_presente = _togli_rumore(a_)                # 27/9 velocita': rumore del gradiente, solo registro
    # 27/9 top1_uguale (A20): solo registro nel held-out, tolto qui; tutto il resto al bit
    top1_presente = all("top1_uguale" in x for x in a_["storia_heldout"])
    for x in a_["storia_heldout"]:
        x.pop("top1_uguale", None)
        x.pop("top1_uguale_per_dominio", None)
    stati = a_ == b_ and top1_presente and rumore_presente
    ok_b = th and adam and stati and zn["passo"] == 4 and "gamma" not in zn
    return registra("A16_gamma_none_al_bit", ok_a and ok_b, pezzi=dett, ciclo_theta=th, ciclo_adam=adam,
                    ciclo_stato=stati, config_nuovi=cfg_nuovi, base=BASE_PRIMA,
                    heldout=[(x["passo"], round(x["metrica"], 6)) for x in zn["storia_heldout"]],
                    secondi=round(time.time() - t0, 1))


def dataclasses_asdict(ca) -> dict:
    import dataclasses
    return dataclasses.asdict(ca)


def _config_per(mod, ca):
    """ConfigAllena del modulo `mod` (un commit di prima) coi soli campi che conosce, presi da `ca`."""
    import dataclasses
    noti = {f.name for f in dataclasses.fields(mod.ConfigAllena)}
    return mod.ConfigAllena(**{k: v for k, v in dataclasses_asdict(ca).items() if k in noti})


def _togli_rumore(s: dict) -> bool:
    """27/9 velocita': toglie (in posto) il rumore del gradiente (solo registro) dallo stato JSON di `allena`; vero se
    c'era in ogni voce della storia e nel G1."""
    ok = all("rumore" in x for x in s["storia"]) and "rumore" in (s.get("g1") or {})
    for x in s["storia"]:
        x.pop("rumore", None)
    (s.get("g1") or {}).pop("rumore", None)
    return ok


# ─────────────────────────────────────────── A17 ───────────────────────────────────────────
def prova_a17() -> bool:
    """gamma misurato == media di p_T(y vero) sui bersagli (token successivo, `_bersagli`: niente bersaglio a fine
    riga ne' fra segmenti; bersagli fuori dalla testa esclusi e contati), ricalcolata in float64 riga per riga."""
    g = torch.Generator().manual_seed(17)
    V, H, T, n = 1000, 16, 40, 6
    WT = torch.randn(V, H, generator=g) * 0.8
    righe = []
    for i in range(n):
        ids = torch.randint(0, V + 50, (T,), generator=g).numpy().astype(np.int32)   # alcuni id oltre la testa
        hT = (torch.randn(T, H, generator=g) * 2).to(torch.bfloat16)
        iz = np.zeros(T, np.int32)                  # modo segmenti (la sorgente ha un solo modo)
        iz[[0, 7 + i, 23 + 2 * i]] = 1
        righe.append((ids, hT, "tr", iz))
    src = A.SorgenteMemoria({"train": righe, "heldout": righe[:1]}, {})
    car = A.Caricatore(src, torch.device("cpu"))
    ca = _ca_finta(pezzo_kl=13, righe_per_micro=4)
    m = A.misura_gamma(car, "train", list(range(n)), ca, WT)
    car.chiudi()
    tot, cnt, fuori = 0.0, 0, 0
    for ids, hT, _, iz in righe:
        lp = torch.log_softmax(hT.double() @ WT.double().T, -1)
        for t in range(T - 1):
            if iz[t + 1] == 1:
                continue
            y = int(ids[t + 1])
            if y >= V:
                fuori += 1
                continue
            tot += float(lp[t, y].exp())
            cnt += 1
    rif = tot / cnt
    # logit del teacher in fp32 (percorso vero): 1e-5 relativo
    ok = abs(m["gamma"] - rif) <= 1e-5 * rif and m["token"] == cnt and m["bersagli_fuori_testa"] == fuori
    return registra("A17_gamma_misurato", ok, gamma=m["gamma"], ricalcolo_fp64=rif, err_rel=abs(m["gamma"] - rif) / rif,
                    token=m["token"], atteso_token=cnt, fuori_testa=m["bersagli_fuori_testa"], atteso_fuori=fuori)


# ─────────────────────────────────────────── A18 ───────────────────────────────────────────
def prova_a18() -> bool:
    """Ciclo con perdita cakld (studente finto): (a) gamma misurato UNA volta sulle righe del passo 0 (== misura
    diretta), nello stato; (b) 3 passi + ripresa da disco + 3 == 6 filati AL BIT (theta, Adam, stato, gamma) e
    alla ripresa il gamma NON si rimisura; (c) held-out: KL avanti, inversa, CAKLD per dominio e metrica = macro
    CAKLD senza wiki; il migliore porta la metrica CAKLD; (d) ripresa kl -> cakld e gamma deciso accettate
    (campi di decisione) e registrate; (e) CLI --perdita --gamma."""
    t0 = time.time()
    src, WS = sorgente_finta()
    ca = _ca_finta(perdita="cakld")
    conta = {"n": 0}
    orig = A.misura_gamma

    def misura_contata(*a, **k):
        conta["n"] += 1
        return orig(*a, **k)
    A.misura_gamma = misura_contata
    try:
        d_f, d_r, d_k = DIR / "a18_filata", DIR / "a18_ripresa", DIR / "a18_kl_poi_cakld"
        for d in (d_f, d_r, d_k):
            if d.exists():
                shutil.rmtree(d)
        st_f = StudenteFinto()
        filata = A.allena(st_f, src, d_f, ca, WS, WS, log=muto)
        n_filata = conta["n"]
        A.allena(StudenteFinto(), src, d_r, ca, WS, WS, ferma_a=3, log=muto)
        n_prima = conta["n"]
        registro = []
        st_b = StudenteFinto()
        ripresa = A.allena(st_b, src, d_r, ca, WS, WS, riprendi=True, log=registro.append)
        n_ripresa = conta["n"] - n_prima
        # (d) kl per 2 passi, poi ripresa con cakld (gamma misurato allora) e poi con gamma deciso 0,25
        A.allena(StudenteFinto(), src, d_k, _ca_finta(), WS, WS, ferma_a=2, log=muto)
        reg_k = []
        s_k = A.allena(StudenteFinto(), src, d_k, ca, WS, WS, riprendi=True, ferma_a=4, log=reg_k.append)
        reg_k2 = []
        s_k2 = A.allena(StudenteFinto(), src, d_k, _ca_finta(perdita="cakld", gamma=0.25), WS, WS, riprendi=True,
                        ferma_a=6, log=reg_k2.append)
    finally:
        A.misura_gamma = orig
    car = A.Caricatore(src, torch.device("cpu"))
    diretta = A.misura_gamma(car, "train", [0, 1], ca, WS)["gamma"]
    car.chiudi()
    ga = filata["gamma"]
    ok_a = ga["misurato"] and ga["valore"] == diretta and n_filata == 1 and ga["righe"] == [0, 1]
    th = all(torch.equal(a, b) for a, b in zip(st_f.parametri(), st_b.parametri()))
    rf, rr = A.carica_ripresa(d_f, "cpu"), A.carica_ripresa(d_r, "cpu")
    so, sr = rf["opt"]["state"], rr["opt"]["state"]
    adam = all(torch.equal(so[i][k], sr[i][k]) for i in so for k in ("exp_avg", "exp_avg_sq", "step"))
    ok_b = th and adam and _senza_tempi(filata) == _senza_tempi(ripresa) and ripresa["gamma"] == ga \
        and n_ripresa == 0 and any("gamma CAKLD dallo stato" in x for x in registro) and filata["passo"] == 6
    # (c) held-out
    h = filata["storia_heldout"][-1]
    usati = [d for d in h["perdita_per_dominio"] if d != "wiki"]
    macro = sum(h["perdita_per_dominio"][d] for d in usati) / len(usati)
    per_dom_ok = all(abs(h["perdita_per_dominio"][d] - (ga["valore"] * h["kl_inv_per_dominio"][d] +
                                                        (1 - ga["valore"]) * h["kl_per_dominio"][d])) < 1e-9
                     for d in h["perdita_per_dominio"])
    mig = filata["migliore"]
    mig_h = next(x for x in filata["storia_heldout"] if x["passo"] == mig["passo"])
    ok_c = (h["metrica"] == macro and h["metrica_kl"] == sum(h["kl_per_dominio"][d] for d in usati) / len(usati)
            and per_dom_ok and "wiki" not in h["domini_metrica"] and h["gamma"] == ga["valore"]
            and mig["perdita"] == "cakld" and mig["metrica"] == mig_h["metrica"]
            and mig["metrica"] == min(x["metrica"] for x in filata["storia_heldout"])
            and all("kl_inv_train" in x and "kl_fwd_train" in x for x in filata["storia"]))
    # (d)
    ok_d = (s_k["config"]["perdita"] == "cakld" and s_k["gamma"]["misurato"] and s_k["passo"] == 4
            and any("perdita kl -> cakld" in x for x in reg_k) and s_k2["gamma"] == {"valore": 0.25, "misurato": False}
            and s_k2["storia_heldout"][-1]["gamma"] == 0.25 and s_k2["passo"] == 6
            and any("gamma" in x and "0.25" in x for x in reg_k2))
    # (e) CLI
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--perdita", choices=("kl", "cakld"), default=A.ConfigAllena.perdita)
    p.add_argument("--gamma", type=float, default=A.ConfigAllena.gamma)
    ok_e = "perdita" in A._CAMPI_DI_DECISIONE and "gamma" in A._CAMPI_DI_DECISIONE
    try:
        A.ConfigAllena(perdita="xx")
        ok_e = False
    except ValueError:
        pass
    testo = (QUI / "e2e_allena.py").read_text()
    ok_e &= '"--perdita"' in testo and '"--gamma"' in testo and "perdita=a.perdita, gamma=a.gamma" in testo
    return registra("A18_cakld_ciclo_e_ripresa", ok_a and ok_b and ok_c and ok_d and ok_e, gamma=ga,
                    gamma_diretto=diretta, misure_filata=n_filata, misure_alla_ripresa=n_ripresa, a=ok_a,
                    ripresa_al_bit=ok_b, heldout=ok_c, kl_poi_cakld=ok_d, cli=ok_e,
                    ultima_heldout={k: h[k] for k in ("kl", "kl_inv", "perdita", "metrica", "metrica_kl")},
                    secondi=round(time.time() - t0, 1))


# ─────────────────────────────────────────── A20 ───────────────────────────────────────────
BASE_CAKLD = "c56df09"          # commit CAKLD, prima di sonda con la perdita e top1_uguale


def _pv_finto(st, salto: dict):
    """PassoV minimo per `passo_v`: il "salto" somma `salto` a theta (annullamento = sottrazione esatta: si salva
    theta prima). Niente famiglie: le scale non servono."""
    from types import SimpleNamespace

    class PVFinto:
        fam, registri = {}, []

        def __init__(self):
            self.stato = SimpleNamespace(passi_nella_finestra=1, n_verifica=0, n_curvatura=0,
                                         chiudi_finestra=lambda: None)
            self.prima = None

        def fine_raccolta(self):
            pass

        def seleziona_e_salta(self, scale, rho):
            self.prima = {n: p.detach().clone() for n, p in st.theta.items()}
            with torch.no_grad():
                for n, p in st.theta.items():
                    p.add_(salto[n])
            return [0], {"x": {"testati": 1, "scoperte": 1, "convenienti": 1, "salti": 1, "guadagno_previsto": 0.0}}

        def annulla(self, reg):
            with torch.no_grad():
                for n, p in st.theta.items():
                    p.copy_(self.prima[n])
    return PVFinto()


def prova_a20() -> bool:
    """27/9 (a) sonda della regione di fiducia del passo V con la perdita dell'allenamento: perdita kl -> `kl_sonda`
    AL BIT come il commit c56df09; perdita cakld -> media per token della perdita CAKLD col gamma della corsa (==
    ricalcolo float64, == somma di kl_pezzo["perdita"] al bit). (b) `passo_v` (PV finto, salti = spostamenti di theta)
    decide tieni/annulla sulla perdita della corsa: con un salto che fa SCENDERE la KL e SALIRE la CAKLD, kl tiene e
    cakld annulla (theta rimesso al bit); e l'inverso. (c) held-out: top1_uguale (argmax studente == argmax maestro)
    per dominio e macro senza wiki == ricalcolo diretto, in entrambe le perdite; tutte le altre uscite di `valuta`
    AL BIT come c56df09 (metrica e migliore invariati)."""
    import e2e_pv_codici as PV
    t0 = time.time()
    src, WS = sorgente_finta()
    vec = _modulo_prima(BASE_CAKLD)
    car = A.Caricatore(src, torch.device("cpu"))
    gamma = 0.8
    ca_kl, ca_ck = _ca_finta(), _ca_finta(perdita="cakld", gamma=gamma)
    righe = [0, 1, 2]

    def rif64(st, rr):
        num, n = 0.0, 0
        for i in rr:
            ids, hT, _, _ = src.r["train"][i]
            hS = st.nascosto(torch.from_numpy(np.asarray(ids).astype(np.int64))[None]).double()
            lS = torch.log_softmax(hS @ WS.double().T, -1)
            lT = torch.log_softmax(hT.double().reshape(hS.shape) @ WS.double().T, -1)
            kf = (lT.exp() * (lT - lS)).sum(-1)
            ki = (lS.exp() * (lS - lT)).sum(-1)
            num += float((gamma * ki + (1 - gamma) * kf).sum())
            n += hS.shape[0]
        return num / n
    st = StudenteFinto()
    ca_kl_v = _config_per(vec, ca_kl)
    s_kl = A.kl_sonda(st, car, "train", righe, ca_kl, WS, WS)
    s_kl_v = vec.kl_sonda(st, car, "train", righe, ca_kl_v, WS, WS)
    s_ck = A.kl_sonda(st, car, "train", righe, ca_ck, WS, WS)
    tot, n = 0.0, 0
    for j in range(0, len(righe), ca_ck.righe_per_micro):
        lot = car.prendi("train", righe[j:j + ca_ck.righe_per_micro])
        hS = st.nascosto(lot.ids, inizi=lot.inizi)
        for i in range(0, hS.shape[0], ca_ck.pezzo_kl):
            tot += float(A.kl_pezzo(hS[i:i + ca_ck.pezzo_kl], WS, lot.hT[i:i + ca_ck.pezzo_kl], WS, False,
                                    gamma=gamma)["perdita"].sum())
        n += hS.shape[0]
    r64 = rif64(st, righe)
    ok_a = s_kl == s_kl_v and s_ck == tot / n and s_ck != s_kl and abs(s_ck - r64) <= 1e-5 * abs(r64)

    # (b) salti con KL e CAKLD in disaccordo sulla sonda (primo micro-lotto delle righe "prossime")
    prossime, sonda = [3, 4], [3]
    orig_curv = A.curvatura_passo
    A.curvatura_passo = lambda *a, **k: None
    trovati, esiti_b = {}, {}
    try:
        g = torch.Generator().manual_seed(20)
        for tent in range(1200):
            salto = {n_: (0.02, 0.05, 0.15)[tent % 3] * (1 - 2 * (tent // 3 % 2)) * torch.randn(16, 16, generator=g)
                     for n_ in StudenteFinto.FAM}
            st_ = StudenteFinto()
            pk = A.kl_sonda(st_, car, "train", sonda, ca_kl, WS, WS)
            pc = A.kl_sonda(st_, car, "train", sonda, ca_ck, WS, WS)
            with torch.no_grad():
                for n_, p in st_.theta.items():
                    p.add_(salto[n_])
            dk = A.kl_sonda(st_, car, "train", sonda, ca_kl, WS, WS) - pk
            dc = A.kl_sonda(st_, car, "train", sonda, ca_ck, WS, WS) - pc
            caso = "kl_giu_cakld_su" if dk < 0 < dc else "kl_su_cakld_giu" if dc < 0 < dk else None
            if caso and caso not in trovati:
                trovati[caso] = (tent, salto)
            if len(trovati) == 2:
                break
        for caso, (tent, salto) in trovati.items():
            for nome, ca_ in (("kl", ca_kl), ("cakld", ca_ck)):
                st_ = StudenteFinto()
                th0 = [p.detach().clone() for p in st_.parametri()]
                ev = A.passo_v(st_, src, car, righe, prossime, [0], ca_, WS, WS, 1.0, 1, _pv_finto(st_, salto),
                               PV.RegioneFiducia(prova=(0.05,)), 0, muto)
                ripristinato = all(torch.equal(a, b) for a, b in zip(th0, st_.parametri()))
                atteso = (caso == "kl_giu_cakld_su") == (nome == "kl")
                esiti_b[f"{caso}/{nome}"] = {"tenuto": ev["tenuto"], "atteso": atteso,
                                             "sonda": (ev["sonda_prima"], ev["sonda_dopo"]),
                                             "theta_al_bit_se_annullato": ripristinato}
    finally:
        A.curvatura_passo = orig_curv
    ok_b = len(trovati) == 2 and all(e["tenuto"] == e["atteso"] and (e["tenuto"] or e["theta_al_bit_se_annullato"])
                                     for e in esiti_b.values())

    # (c) top1_uguale in valuta
    held = list(range(len(src.r["heldout"])))
    ok_c, dett_c = True, {}
    st = StudenteFinto()
    with torch.no_grad():
        for p in st.parametri():
            p.add_(0.1)
    for nome, ca_ in (("kl", ca_kl), ("cakld", ca_ck)):
        v = A.valuta(st, car, held, ca_, WS, WS)
        cav = _config_per(vec, ca_)
        w = vec.valuta(st, car, held, cav, WS, WS)
        per_dom: dict = {}
        for i in held:
            lot = car.prendi("heldout", [i])
            hS = st.nascosto(lot.ids, inizi=lot.inizi)
            u = ((hS @ WS.T).argmax(-1) == (lot.hT.to(hS.dtype) @ WS.T).argmax(-1)).double()
            s_ = per_dom.setdefault(lot.domini[0], [0.0, 0])
            s_[0] += float(u.sum())
            s_[1] += u.numel()
        rif = {d: a / b for d, (a, b) in sorted(per_dom.items())}
        macro = sum(rif[d] for d in rif if d != "wiki") / len([d for d in rif if d != "wiki"])
        resto = {k: x for k, x in v.items() if not k.startswith("top1_")}
        u_c = (resto == w and set(v) - set(w) == {"top1_uguale", "top1_uguale_per_dominio"}
               and all(abs(v["top1_uguale_per_dominio"][d] - rif[d]) < 1e-12 for d in rif)
               and set(v["top1_uguale_per_dominio"]) == set(rif) and abs(v["top1_uguale"] - macro) < 1e-12
               and 0.0 < v["top1_uguale"] < 1.0)
        dett_c[nome] = {"top1_uguale": v["top1_uguale"], "per_dominio": v["top1_uguale_per_dominio"],
                        "ricalcolo": rif, "resto_al_bit_come_c56df09": resto == w, "metrica": v["metrica"]}
        ok_c &= u_c
    car.chiudi()
    return registra("A20_sonda_con_la_perdita_e_top1", ok_a and ok_b and ok_c, sonda_kl=s_kl, sonda_kl_c56df09=s_kl_v,
                    sonda_cakld=s_ck, sonda_cakld_fp64=r64, a=ok_a, casi=sorted(trovati),
                    tentativi={c: t for c, (t, _) in trovati.items()}, passo_v=esiti_b, b=ok_b, top1=dett_c, c=ok_c,
                    secondi=round(time.time() - t0, 1))


def prova_a19() -> bool:
    """27/9 finestra di accumulo PV (pv_passi_accumulo=4) nel ciclo: strato 0 VERO, maestro con 0,5% dei codici spostati.
    (a) corsa intera di 6 passi (finestra 2-4 decisa al passo 4, finestra 5-6 chiusa all'ultimo passo); (b) la stessa
    fermata al passo 3 (finestra A META': la ripresa salva accumulo_pv_000003.pt) e ripresa: theta, byte, registri,
    contatori identici AL BIT; (c) nei passi in accumulo nessun salto (i byte cambiano solo a fine finestra)."""
    import e2e_pv_codici as PV
    t0 = time.time()
    st_t = studente((0,))
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
    th = theta_stella(st_t)
    passi = 6
    src = sorgente(st_t, 6 * passi, 4, th)
    del st_t
    ca = _ca(pv_codici=True, righe_per_passo=6, righe_per_micro=1, heldout_ogni=2, ripresa_ogni=2, pv_rho=0.05,
             pv_passi_accumulo=4)

    def corsa(dir_c, **kw):
        st = studente((0,))
        WS = testa_ristretta(st)
        return st, A.allena(st, src, dir_c, ca, WS, WS, log=muto, **kw)
    dA, dB = DIR / "a19_intera", DIR / "a19_ripresa"
    for d in (dA, dB):
        if d.exists():
            shutil.rmtree(d)
    stA, sA = corsa(dA)
    _, sB1 = corsa(dB, ferma_a=3)
    acc_salvato = bool(sB1["pv"].get("accumulo")) and (dB / sB1["pv"]["accumulo"]["file"]).exists()
    stB, sB = corsa(dB, riprendi=True)
    theta_uguale = all(torch.equal(a, b) for a, b in zip(stA.parametri(), stB.parametri()))
    byte_uguali = all(torch.equal(stA.esperti(0).fam[n].blocchi, stB.esperti(0).fam[n].blocchi)
                      for n in stA.esperti(0).fam)
    pvA, pvB = sA["pv"], sB["pv"]
    contatori = all(pvA[k] == pvB[k] for k in ("registri", "salti_tenuti", "passi_v", "rho", "sha"))
    decisioni = [x["passo"] for x in sA["storia"] if x.get("pv")]
    ok = (theta_uguale and byte_uguali and contatori and acc_salvato and sB1["passo"] == 3
          and decisioni == [4, 6] and pvA["passi_v"] == 2)
    return registra("A19_finestra_di_accumulo_ripresa_al_bit", ok, accumulo_salvato_a_meta=acc_salvato,
                    ripresa_theta_al_bit=theta_uguale, ripresa_byte_al_bit=byte_uguali, contatori_uguali=contatori,
                    passi_con_decisione=decisioni, pv=pvA,
                    passi_v=[{k: (round(v, 6) if isinstance(v, float) else v) for k, v in x["pv"].items()}
                             for x in sA["storia"] if x.get("pv")],
                    heldout=[(x["passo"], round(x["metrica"], 6)) for x in sA["storia_heldout"]],
                    secondi=round(time.time() - t0, 1))


# ─────────────────────────────────────────── A28 ───────────────────────────────────────────
class _StudenteFintoV(StudenteFinto):
    def scala(self, n):
        return self.W[n] * torch.exp(self.theta[n])


def _pv_finto_accumulo():
    """`installa` e `PassoV` finti per `allena` con lo studente finto: StatoV VERO (finestra, conteggi, istantanea,
    ripristina) su buffer (1, 16, 16) per famiglia; il "motore V" somma in prop/S/Q/H numeri che dipendono da theta
    del passo e dal micro-lotto; il "salto" sposta theta di rho * 1e-2 * tanh(S + H accumulati) (annullamento =
    copia). `ultimo["pv"]`: l'ultimo PassoV creato dal ciclo."""
    from types import SimpleNamespace
    ultimo: dict = {}

    def installa(st, stato):
        fams = {(0, n): SimpleNamespace(nome=n, strato=0, E=1, R=16, NG=16, gs=16, blocchi=st.fam[n].blocchi)
                for n in st.FAM}
        stato.prepara(fams.values())
        ultimo["st"] = st
        return fams

    class PassoVFinto:
        def __init__(self, fam, cfg, stato):
            self.fam, self.cfg, self.stato, self.registri = fam, cfg, stato, []
            self.st, self.prima = ultimo["st"], None
            ultimo["pv"] = self

        def inizia_passo(self):
            self.stato.inizia_passo()

        def registra_scale_finestra(self, scale):
            self.stato.d_finestra = {k: scale[k].detach().to(torch.float16).clone() for k in self.fam}

        def _somma(self, campo, peso):
            for (L, n), f in self.fam.items():
                getattr(self.stato.buffer(f), campo).add_(
                    peso * torch.sin(self.st.theta[n].detach() * 997.0 + peso)[None])

        def microlotto(self, k):
            modo = self.stato.microlotto(k)
            if modo == 1:
                self._somma("prop", 1.0 + k)
            else:
                self._somma("S", 1.0 + k)
                self._somma("Q", 0.5 + k)
            return modo

        def curvatura(self):
            m = self.stato.curvatura()
            self._somma("H", 0.25)
            return m

        def fine_raccolta(self):
            self.stato.fine_raccolta()

        def seleziona_e_salta(self, scale, rho):
            s = self.stato
            self.prima = {n: p.detach().clone() for n, p in self.st.theta.items()}
            with torch.no_grad():
                for (L, n), f in self.fam.items():
                    self.st.theta[n].add_(rho * 1e-2 * torch.tanh(s.buffer(f).S[0] + s.buffer(f).H[0]))
            g = float(sum(float(s.buffer(f).S.double().sum()) for f in self.fam.values()))
            return [0], {"x": {"testati": s.n_verifica, "scoperte": s.n_propose, "convenienti": s.n_curvatura,
                               "salti": s.passi_nella_finestra, "guadagno_previsto": g}}

        def annulla(self, reg):
            with torch.no_grad():
                for n, p in self.st.theta.items():
                    p.copy_(self.prima[n])
    return installa, PassoVFinto, ultimo


def prova_a28() -> bool:
    """27/9 finestra di accumulo PV nel CICLO con lo studente finto (CPU, senza /mnt): stessa sequenza di A19
    (pv_passi_accumulo=4, 6 passi, fermata al passo 3 = finestra a meta', ripresa) con StatoV VERO e motore/salti
    finti. (a) la ripresa del passo 3 contiene la finestra (accumulo_pv_000003.pt, 2 passi nella finestra) e la
    rilegge AL BIT (buffer, scale d'inizio finestra, conteggi); (b) decisioni ai passi 4 e 6 nella corsa intera E
    nella corsa ripresa (nessuna prima della fermata); (c) theta, contatori PV, esiti dei passi V e buffer finali
    identici AL BIT."""
    import e2e_pv_codici as PV
    t0 = time.time()
    passi = 6
    src, WS = sorgente_finta(n_train=2 * passi)
    ca = _ca_finta(pv_codici=True, pv_passi_accumulo=4, pv_rho=0.5, pv_micro_curvatura=1)
    orig = (PV.installa, PV.PassoV, PV.famiglia_scrivibile)
    installa, PVF, ultimo = _pv_finto_accumulo()
    PV.installa, PV.PassoV, PV.famiglia_scrivibile = installa, PVF, (lambda f: f)

    def corsa(d, **kw):
        st = _StudenteFintoV()
        return st, A.allena(st, src, d, ca, WS, WS, log=muto, **kw), ultimo["pv"]
    try:
        dA, dB = DIR / "a28_intera", DIR / "a28_ripresa"
        for d in (dA, dB):
            if d.exists():
                shutil.rmtree(d)
        stA, sA, pvA = corsa(dA)
        _, sB1, pvB1 = corsa(dB, ferma_a=3)
        acc = sB1.get("pv", {}).get("accumulo")
        acc_ok = bool(acc) and (dB / acc["file"]).exists() and acc.get("passi_nella_finestra") == 2
        ist_B1 = pvB1.stato.istantanea()
        ist_ripresa = {}
        orig_rip = PV.StatoV.ripristina

        def ripristina(self, ist):
            orig_rip(self, ist)
            i_ = self.istantanea()                    # su CPU .cpu() non copia: clone, i buffer cresceranno
            i_["buffer"] = {f: {n: x.clone() for n, x in d.items()} for f, d in i_["buffer"].items()}
            i_["d_finestra"] = {f: x.clone() for f, x in i_["d_finestra"].items()}
            ist_ripresa.update(i_)
        PV.StatoV.ripristina = ripristina
        try:
            stB, sB, pvB = corsa(dB, riprendi=True)
        finally:
            PV.StatoV.ripristina = orig_rip
    finally:
        PV.installa, PV.PassoV, PV.famiglia_scrivibile = orig

    def uguali(a, b):
        return (bool(a) and bool(b) and all(a[k] == b[k] for k in ("passi_accumulo", "passi_nella_finestra",
                                                                   "n_verifica", "n_propose", "n_curvatura"))
                and set(a["buffer"]) == set(b["buffer"]) and set(a["d_finestra"]) == set(b["d_finestra"])
                and all(torch.equal(a["buffer"][f][n], b["buffer"][f][n]) for f in a["buffer"] for n in a["buffer"][f])
                and all(torch.equal(a["d_finestra"][f], b["d_finestra"][f]) for f in a["d_finestra"]))
    riletta_al_bit = acc_ok and uguali(ist_B1, ist_ripresa)
    theta_uguale = all(torch.equal(a, b) for a, b in zip(stA.parametri(), stB.parametri()))
    buffer_finali = uguali(pvA.stato.istantanea(), pvB.stato.istantanea())
    contatori = all(sA["pv"][k] == sB["pv"][k] for k in ("registri", "salti_tenuti", "passi_v", "rho", "tarato"))
    dec = {nome: [x["passo"] for x in s_["storia"] if x.get("pv")]
           for nome, s_ in (("intera", sA), ("fino_a_3", sB1), ("ripresa", sB))}
    evA = [x["pv"] for x in sA["storia"] if x.get("pv")]
    evB = [x["pv"] for x in sB["storia"] if x.get("pv")]
    ok = (acc_ok and riletta_al_bit and theta_uguale and buffer_finali and contatori and sB1["passo"] == 3
          and dec == {"intera": [4, 6], "fino_a_3": [], "ripresa": [4, 6]} and evA == evB
          and sA["pv"]["passi_v"] == 2)
    return registra("A28_finestra_di_accumulo_nel_ciclo_studente_finto", ok, accumulo_salvato_a_meta=acc,
                    finestra_riletta_al_bit=riletta_al_bit, ripresa_theta_al_bit=theta_uguale,
                    buffer_finali_al_bit=buffer_finali, contatori_uguali=contatori, passi_con_decisione=dec,
                    pv_intera=sA["pv"], pv_ripresa=sB["pv"], esiti_v_uguali=evA == evB,
                    passi_v=[{k: (round(v, 6) if isinstance(v, float) else v) for k, v in e.items()} for e in evA],
                    secondi=round(time.time() - t0, 1))


def _pv_finto_vbop():
    """Come `_pv_finto_accumulo` (StatoV VERO, motore/salti finti) con V-Bop: la selezione aggiorna lo stato
    persistente VERO (`aggiorna_bop`: fusione dei candidati con isteresi, EMA di m/v/H, eta'), il "salto" sposta anche
    m di chi ha candidato (come la correzione alla ECO) e l'annullamento lo rimette; guadagno previsto > 0."""
    installa, Base, ultimo = _pv_finto_accumulo()

    class PassoVFintoBop(Base):
        def seleziona_e_salta(self, scale, rho):
            self.stato.aggiorna_bop()
            self.prima_bop = {k: z.m.clone() for k, z in self.stato.bop_.items()}
            reg, diag = super().seleziona_e_salta(scale, rho)
            for z in self.stato.bop_.values():
                z.m.mul_(0.5)
            diag["x"]["guadagno_previsto"] = abs(diag["x"]["guadagno_previsto"]) * 1e-4 + 1e-3
            diag["x"]["con_candidato"] = sum(int(((z.eta & 127) >= 1).sum()) for z in self.stato.bop_.values())
            return reg, diag

        def annulla(self, reg):
            super().annulla(reg)
            for k, z in self.stato.bop_.items():
                z.m.copy_(self.prima_bop[k])
    return installa, PassoVFintoBop, ultimo


def prova_a29() -> bool:
    """27/9 V-Bop a gruppo (--pv-modo vbop, curvatura ogni 2 passi V) e rho adattivo (--pv-rho-adattivo) nel CICLO
    con lo studente finto (CPU, senza /mnt), StatoV VERO e motore/salti finti; 6 passi contro fermata al passo 3 e
    ripresa: (a) la ripresa contiene lo stato V-Bop (bop_pv_000003.pt, sha nello stato) e lo rilegge AL BIT;
    (b) curvatura misurata ai passi V 1, 3, 5 (su 5); r e rho_dopo (raddoppi e dimezzamenti) in ogni esito V e nella storia di rho; (c) theta,
    esiti V, contatori, stato V-Bop finale identici AL BIT; (d) la configurazione rifiuta vbop con accumulo > 1 e
    curvatura_ogni > 1 senza vbop."""
    import e2e_pv_codici as PV
    t0 = time.time()
    passi = 6
    src, WS = sorgente_finta(n_train=4 * passi)
    ca = _ca_finta(righe_per_passo=4, pv_codici=True, pv_modo="vbop", pv_curvatura_ogni=2, pv_rho_adattivo=True, pv_rho=0.5,
                   pv_rho_max=4.0, pv_micro_curvatura=1)
    orig = (PV.installa, PV.PassoV, PV.famiglia_scrivibile)
    installa, PVF, ultimo = _pv_finto_vbop()
    PV.installa, PV.PassoV, PV.famiglia_scrivibile = installa, PVF, (lambda f: f)
    righe_log: list = []

    def corsa(d, **kw):
        st = _StudenteFintoV()
        return st, A.allena(st, src, d, ca, WS, WS, log=righe_log.append, **kw), ultimo["pv"]
    try:
        dA, dB = DIR / "a29_intera", DIR / "a29_ripresa"
        for d in (dA, dB):
            if d.exists():
                shutil.rmtree(d)
        stA, sA, pvA = corsa(dA)
        _, sB1, pvB1 = corsa(dB, ferma_a=3)
        sb = sB1.get("pv", {}).get("bop")
        bop_ok = bool(sb) and (dB / sb["file"]).exists() and sb["file"] == "bop_pv_000003.pt" \
            and A.sha256_file(dB / sb["file"]) == sb["sha"]
        ist_B1 = pvB1.stato.istantanea_bop()
        ist_ripresa = {}
        orig_rip = PV.StatoV.ripristina_bop

        def ripristina_bop(self, ist):
            orig_rip(self, ist)
            i_ = self.istantanea_bop()
            ist_ripresa.update(i_)
        PV.StatoV.ripristina_bop = ripristina_bop
        try:
            stB, sB, pvB = corsa(dB, riprendi=True)
        finally:
            PV.StatoV.ripristina_bop = orig_rip
    finally:
        PV.installa, PV.PassoV, PV.famiglia_scrivibile = orig

    def uguali(a, b):
        return (bool(a) and bool(b) and a["passi_bop"] == b["passi_bop"] and a["gamma"] == b["gamma"]
                and set(a["stato"]) == set(b["stato"])
                and all(torch.equal(a["stato"][f][n], b["stato"][f][n]) for f in a["stato"] for n in a["stato"][f]))
    riletto = bop_ok and uguali(ist_B1, ist_ripresa)
    theta_uguale = all(torch.equal(a, b) for a, b in zip(stA.parametri(), stB.parametri()))
    finale = uguali(pvA.stato.istantanea_bop(), pvB.stato.istantanea_bop())
    evA = [x["pv"] for x in sA["storia"] if x.get("pv")]
    evB = [x["pv"] for x in sB["storia"] if x.get("pv")]
    contatori = all(sA["pv"][k] == sB["pv"][k] for k in ("registri", "salti_tenuti", "passi_v", "rho", "tarato"))
    curv = [e.get("curvatura") for e in evA]
    campi = all("r_fiducia" in e and "rho_dopo" in e and "con_candidato" in e for e in evA)
    regola = True
    for e in evA:
        r, rho = e["r_fiducia"], e["rho"]
        att = (rho if e["tenuto"] else rho / 2) if r is None else \
            (min(4.0, 2 * rho) if r > 0.75 else (rho / 2 if r < 0.25 else rho))
        regola &= math.isclose(e["rho_dopo"], max(att, 1e-4), rel_tol=1e-12)
    log_ok = any("V-Bop a gruppo" in r and "B/gruppo" in r for r in righe_log) and \
        any("stato V-Bop ripreso" in r for r in righe_log)
    rifiuti = []
    # 1/10 TT5-ottimizzato: vbop con pv_passi_accumulo > 1 e' ora AMMESSO (decisione ogni N passi, prova A34)
    for kw in ({"pv_curvatura_ogni": 2}, {"pv_modo": "bop"}, {"pv_modo": "vbop", "righe_per_passo": 3}):
        try:
            _ca_finta(pv_codici=True, **kw)
            rifiuti.append(False)
        except ValueError:
            rifiuti.append(True)
    n_v = sA["pv"]["passi_v"]                  # il passo 1 e' del cancello G1: passi V 2..6
    ok = (riletto and theta_uguale and finale and evA == evB and contatori and n_v == len(evA) == passi - 1
          and curv == [i % 2 == 0 for i in range(n_v)] and campi and regola and log_ok and all(rifiuti)
          and {e["rho_dopo"] > e["rho"] for e in evA} == {True, False}
          and sB1["passo"] == 3)
    return registra("A29_vbop_e_rho_adattivo_nel_ciclo_studente_finto", ok, bop_salvato=sb,
                    stato_bop_riletto_al_bit=riletto, theta_al_bit=theta_uguale, stato_bop_finale_al_bit=finale,
                    esiti_v_uguali=evA == evB, contatori_uguali=contatori, curvatura=curv, campi_nuovi=campi,
                    regola_rho=regola, log=log_ok, rifiuti_config=rifiuti,
                    passi_v=[{k: (round(v, 6) if isinstance(v, float) else v) for k, v in e.items()} for e in evA],
                    secondi=round(time.time() - t0, 1))


def prova_a30() -> bool:
    """27/9 (ramo vbop2) PASSAGGIO della corsa vera da finestra di accumulo a V-Bop, alla ripresa (studente finto,
    CPU, senza /mnt; StatoV VERO, motore/salti finti come A28/A29). Fase 1: pv_passi_accumulo=4 (come EB2), 8 passi,
    decisione al passo 4, fermata al passo 6 con la finestra A META' (passi 5 e 6: accumulo_pv_000006.pt nella
    ripresa). Fase 2: ripresa con pv_modo vbop e pv_passi_accumulo 1, in due varianti (V-Bop semplice; V-Bop con
    curvatura ogni 2 e rho adattivo) e su due copie della cartella: D1 intatta, D2 SENZA il file della finestra.
    Si prova: (a) la ripresa parte senza errori, cambi registrati (pv_modo, pv_passi_accumulo) e i due messaggi
    («finestra salvata con accumulo 4, ora 1: si riparte da una finestra nuova», «V-Bop acceso alla ripresa: stato
    per gruppo nuovo (vuoto)»); (b) le statistiche della finestra sono SCARTATE: `StatoV.ripristina` mai chiamato, il
    file mai letto (D2 senza file == D1 al bit: theta, esiti V, stato V-Bop finale), lo stato V-Bop alla prima
    selezione e' vuoto (eta' 0, m = v = H = 0); (c) decisioni V a OGNI passo dopo la ripresa (7 e 8), rho di partenza
    = quello salvato (tarato) dalla fase 1; (d) dopo la ripresa: stato['pv']['accumulo'] None, nessun accumulo_pv_*.pt
    rimasto (tieni_riprese 1), bop_pv_000008.pt con sha."""
    import dataclasses
    import e2e_pv_codici as PV
    t0 = time.time()
    passi = 8
    src, WS = sorgente_finta(n_train=4 * passi)
    ca1 = _ca_finta(righe_per_passo=4, ripresa_ogni=1, pv_codici=True, pv_passi_accumulo=4, pv_rho=0.5,
                    pv_micro_curvatura=1)
    varianti = {"vbop": dict(pv_modo="vbop", pv_passi_accumulo=1),
                "vbop_curv2_rho_adattivo": dict(pv_modo="vbop", pv_passi_accumulo=1, pv_curvatura_ogni=2,
                                                pv_rho_adattivo=True, pv_rho_max=4.0)}
    orig = (PV.installa, PV.PassoV, PV.famiglia_scrivibile)
    orig_rip = PV.StatoV.ripristina
    chiamate_ripristina = []

    def ripristina(self, ist):
        chiamate_ripristina.append(ist.get("passi_nella_finestra"))
        return orig_rip(self, ist)
    esiti, ok = {}, True
    try:
        PV.StatoV.ripristina = ripristina
        # ── fase 1: accumulo 4, fermata a meta' finestra ──
        installa, PVF, ultimo = _pv_finto_accumulo()
        PV.installa, PV.PassoV, PV.famiglia_scrivibile = installa, PVF, (lambda f: f)
        d0 = DIR / "a30_fase1"
        if d0.exists():
            shutil.rmtree(d0)
        s1 = A.allena(_StudenteFintoV(), src, d0, ca1, WS, WS, log=muto, ferma_a=6)
        acc = s1.get("pv", {}).get("accumulo")
        fase1 = {"passo": s1["passo"], "accumulo": acc, "passi_v": s1["pv"]["passi_v"], "rho": s1["pv"]["rho"],
                 "tarato": s1["pv"]["tarato"], "decisioni": [x["passo"] for x in s1["storia"] if x.get("pv")]}
        f1_ok = (s1["passo"] == 6 and bool(acc) and acc.get("passi_nella_finestra") == 2
                 and (d0 / acc["file"]).exists() and fase1["decisioni"] == [4] and s1["pv"]["tarato"])
        ok &= f1_ok
        # ── fase 2: ripresa con V-Bop, su D1 (intatta) e D2 (senza il file della finestra) ──
        installa, PVF, ultimo = _pv_finto_vbop()
        prima_sel: list = []

        class PVFSonda(PVF):
            def seleziona_e_salta(self, scale, rho):
                if not prima_sel:
                    prima_sel.append({"eta_zero": all(bool(((z.eta & 127) == 0).all()) for z in self.stato.bop_.values()),
                                      "m_v_H_zero": all(bool((z.m == 0).all() and (z.r == 0).all() and (z.H == 0).all())
                                                        for z in self.stato.bop_.values()),
                                      "passi_bop": self.stato.passi_bop, "rho": rho})
                return super().seleziona_e_salta(scale, rho)
        PV.installa, PV.PassoV = installa, PVFSonda
        for nome, kw in varianti.items():
            ca2 = dataclasses.replace(ca1, **kw)
            ris = {}
            for tipo in ("D1_intatta", "D2_senza_file_finestra"):
                d = DIR / f"a30_{nome}_{tipo}"
                if d.exists():
                    shutil.rmtree(d)
                shutil.copytree(d0, d)
                if tipo.startswith("D2"):
                    (d / acc["file"]).unlink()
                del chiamate_ripristina[:], prima_sel[:]
                righe_log: list = []
                st = _StudenteFintoV()
                s2 = A.allena(st, src, d, ca2, WS, WS, riprendi=True, log=righe_log.append)
                pvx = ultimo["pv"]
                ris[tipo] = {"st": st, "s": s2, "pv": pvx, "log": righe_log, "ripristina": list(chiamate_ripristina),
                             "prima_sel": dict(prima_sel[0]) if prima_sel else None,
                             "file_acc": sorted(f.name for f in d.glob("accumulo_pv_*.pt")),
                             "file_bop": sorted(f.name for f in d.glob("bop_pv_*.pt"))}
            a_, b_ = ris["D1_intatta"], ris["D2_senza_file_finestra"]
            s2 = a_["s"]
            ev = [x["pv"] for x in s2["storia"] if x.get("pv")]
            evD2 = [x["pv"] for x in b_["s"]["storia"] if x.get("pv")]
            ib, ib2 = a_["pv"].stato.istantanea_bop(), b_["pv"].stato.istantanea_bop()
            bop_uguale = ib["passi_bop"] == ib2["passi_bop"] and all(
                torch.equal(ib["stato"][f][n], ib2["stato"][f][n]) for f in ib["stato"] for n in ib["stato"][f])
            theta_uguale = all(torch.equal(x, y) for x, y in zip(a_["st"].parametri(), b_["st"].parametri()))
            sb = s2["pv"].get("bop") or {}
            cambi = next((r for r in a_["log"] if r.startswith("ripresa con campi cambiati")), "")
            e = {
                "cambi_registrati": "pv_modo pv -> vbop" in cambi and "pv_passi_accumulo 4 -> 1" in cambi,
                "msg_finestra_scartata": any("finestra salvata con accumulo 4, ora 1: si riparte da una finestra nuova"
                                             in r for r in a_["log"]),
                "msg_vbop_vuoto": any("V-Bop acceso alla ripresa: stato per gruppo nuovo (vuoto)" in r for r in a_["log"]),
                "ripristina_chiamato": a_["ripristina"] + b_["ripristina"],
                "stato_vbop_alla_prima_selezione": a_["prima_sel"],
                "rho_di_partenza_uguale_al_salvato": a_["prima_sel"] is not None and a_["prima_sel"]["rho"] == fase1["rho"],
                "decisioni": [x["passo"] for x in s2["storia"] if x.get("pv")],
                "passi_v": s2["pv"]["passi_v"], "passo": s2["passo"],
                "D2_uguale_D1_al_bit": theta_uguale and ev == evD2 and bop_uguale and b_["s"]["pv"]["passi_v"] == s2["pv"]["passi_v"],
                "accumulo_dopo": s2["pv"].get("accumulo"), "file_accumulo_rimasti": a_["file_acc"],
                "file_bop": a_["file_bop"],
                "bop_sha_ok": bool(sb) and sb.get("file") == "bop_pv_000008.pt" and A.sha256_file(DIR / f"a30_{nome}_D1_intatta" / sb["file"]) == sb["sha"],
                "esiti_v": [{k: (round(v, 6) if isinstance(v, float) else v) for k, v in x.items()} for x in ev],
            }
            e_ok = (e["cambi_registrati"] and e["msg_finestra_scartata"] and e["msg_vbop_vuoto"]
                    and e["ripristina_chiamato"] == [] and a_["prima_sel"] is not None
                    and a_["prima_sel"]["eta_zero"] and a_["prima_sel"]["m_v_H_zero"] and a_["prima_sel"]["passi_bop"] == 0
                    and e["rho_di_partenza_uguale_al_salvato"] and e["decisioni"] == [4, 7, 8] and e["passi_v"] == 3
                    and e["passo"] == passi and e["D2_uguale_D1_al_bit"] and e["accumulo_dopo"] is None
                    and e["file_accumulo_rimasti"] == [] and e["bop_sha_ok"] and all("con_candidato" in x for x in ev[1:]))
            if "rho_adattivo" in nome:
                e_ok = e_ok and all("r_fiducia" in x and "rho_dopo" in x for x in ev[1:])
            e["ok"] = e_ok
            ok &= e_ok
            esiti[nome] = e
    finally:
        PV.installa, PV.PassoV, PV.famiglia_scrivibile = orig
        PV.StatoV.ripristina = orig_rip
    return registra("A30_passaggio_accumulo4_a_vbop_alla_ripresa", ok, fase1=fase1, fase1_ok=f1_ok, varianti=esiti,
                    secondi=round(time.time() - t0, 1))


def prova_a31() -> bool:
    """27/9 (ramo vbop2) come A30 ma col CODICE VERO del passo V: strato 0 VERO di Thor-T (legge il GGUF), motore
    torch su CPU, maestro con 0,5% dei codici spostati (come A19). Fase 1: accumulo 4, fermata al passo 6 con la
    finestra 5-6 aperta (accumulo_pv_000006.pt). Fase 2: ripresa con --pv-modo vbop --pv-passi-accumulo 1 su D1
    (intatta) e D2 (senza il file della finestra): StatoBop vero sulle 3 famiglie vere, fusione dei candidati,
    aggiorna_bop, seleziona_bop, salti sui byte veri, sonda, annullamento, bop_pv_000008.pt. Si prova: ripresa senza
    errori, `StatoV.ripristina` mai chiamato, stato V-Bop vuoto alla prima raccolta, D2 == D1 AL BIT (theta, byte,
    sha dei registri, stato V-Bop finale), decisioni ai passi 4, 7, 8, memoria dello stato = 8 B/gruppo + 12 B/esperto."""
    import dataclasses
    import e2e_pv_codici as PV
    t0 = time.time()
    st_t = studente((0,))
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
    th = theta_stella(st_t)
    passi = 8
    src = sorgente(st_t, 6 * passi, 4, th)
    del st_t
    ca1 = _ca(pv_codici=True, righe_per_passo=6, righe_per_micro=1, heldout_ogni=2, ripresa_ogni=1, pv_rho=0.05,
              pv_passi_accumulo=4)
    ca2 = dataclasses.replace(ca1, pv_modo="vbop", pv_passi_accumulo=1)
    orig_rip, orig_agg = PV.StatoV.ripristina, PV.StatoV.aggiorna_bop
    chiamate, prima_agg = [], []

    def ripristina(self, ist):
        chiamate.append(ist.get("passi_nella_finestra"))
        return orig_rip(self, ist)

    def aggiorna_bop(self, *a, **k):
        if not prima_agg:
            zs = list(self.bop_.values())
            prima_agg.append({"eta_zero": all(bool(((z.eta & 127) == 0).all()) for z in zs),
                              "m_v_H_zero": all(bool((z.m == 0).all() and (z.r == 0).all() and (z.H == 0).all())
                                                for z in zs),
                              "passi_bop": self.passi_bop, "byte_bop": self.byte_bop(),
                              "gruppi": sum(z.cand.numel() for z in zs), "esperti": sum(z.sm.numel() for z in zs)})
        return orig_agg(self, *a, **k)

    def corsa(dir_c, ca, **kw):
        st = studente((0,))
        WS = testa_ristretta(st)
        righe_log: list = []
        return st, A.allena(st, src, dir_c, ca, WS, WS, log=righe_log.append, **kw), righe_log
    PV.StatoV.ripristina, PV.StatoV.aggiorna_bop = ripristina, aggiorna_bop
    try:
        d0 = DIR / "a31_fase1"
        if d0.exists():
            shutil.rmtree(d0)
        _, s1, _ = corsa(d0, ca1, ferma_a=6)
        acc = s1["pv"].get("accumulo")
        fase1 = {"passo": s1["passo"], "accumulo": acc, "passi_v": s1["pv"]["passi_v"], "rho": s1["pv"]["rho"],
                 "decisioni": [x["passo"] for x in s1["storia"] if x.get("pv")]}
        f1_ok = (s1["passo"] == 6 and bool(acc) and acc.get("passi_nella_finestra") == 2
                 and (d0 / acc["file"]).exists() and fase1["decisioni"] == [4])
        ris = {}
        for tipo in ("D1", "D2"):
            d = DIR / f"a31_{tipo}"
            if d.exists():
                shutil.rmtree(d)
            shutil.copytree(d0, d)
            if tipo == "D2":
                (d / acc["file"]).unlink()
            del chiamate[:], prima_agg[:]
            st, s2, lg = corsa(d, ca2, riprendi=True)
            ris[tipo] = {"st": st, "s": s2, "log": lg, "ripristina": list(chiamate),
                         "prima_agg": dict(prima_agg[0]) if prima_agg else None,
                         "file_acc": sorted(f.name for f in d.glob("accumulo_pv_*.pt")),
                         "bop": s2["pv"].get("bop"),
                         "ist_bop": torch.load(d / s2["pv"]["bop"]["file"], weights_only=True)
                         if s2["pv"].get("bop") else None}
    finally:
        PV.StatoV.ripristina, PV.StatoV.aggiorna_bop = orig_rip, orig_agg
    a_, b_ = ris["D1"], ris["D2"]
    s2 = a_["s"]
    ev = [x["pv"] for x in s2["storia"] if x.get("pv")]
    evD2 = [x["pv"] for x in b_["s"]["storia"] if x.get("pv")]
    theta = all(torch.equal(x, y) for x, y in zip(a_["st"].parametri(), b_["st"].parametri()))
    byte = all(torch.equal(a_["st"].esperti(0).fam[n].blocchi, b_["st"].esperti(0).fam[n].blocchi)
               for n in a_["st"].esperti(0).fam)
    ia, ib = a_["ist_bop"], b_["ist_bop"]
    bop_uguale = ia is not None and ib is not None and ia["passi_bop"] == ib["passi_bop"] and all(
        torch.equal(ia["stato"][f][n], ib["stato"][f][n]) for f in ia["stato"] for n in ia["stato"][f])
    pa = a_["prima_agg"] or {}
    mem_ok = pa.get("byte_bop") == PV.BYTE_PER_GRUPPO_VBOP * pa.get("gruppi", -1) + 12 * pa.get("esperti", -1)
    d1_ok = (a_["ripristina"] == [] and b_["ripristina"] == [] and pa.get("eta_zero") and pa.get("m_v_H_zero")
             and pa.get("passi_bop") == 0 and mem_ok)
    uguali = theta and byte and ev == evD2 and bop_uguale and s2["pv"]["sha"] == b_["s"]["pv"]["sha"]
    msg = (any("si riparte da una finestra nuova" in r for r in a_["log"])
           and any("V-Bop acceso alla ripresa: stato per gruppo nuovo (vuoto)" in r for r in a_["log"]))
    dec = [x["passo"] for x in s2["storia"] if x.get("pv")]
    sb = a_["bop"] or {}
    fine_ok = (s2["passo"] == passi and dec == [4, 7, 8] and s2["pv"].get("accumulo") is None and a_["file_acc"] == []
               and sb.get("file") == "bop_pv_000008.pt"
               and A.sha256_file(DIR / "a31_D1" / sb["file"]) == sb.get("sha"))
    ok = bool(f1_ok and d1_ok and uguali and msg and fine_ok)
    return registra("A31_passaggio_accumulo4_a_vbop_strato0_vero", ok, fase1=fase1, fase1_ok=f1_ok,
                    ripristina_chiamato=a_["ripristina"] + b_["ripristina"], stato_vbop_alla_prima_raccolta=pa,
                    memoria_8B_gruppo_12B_esperto=mem_ok, D2_uguale_D1_al_bit=uguali, theta=theta, byte=byte,
                    bop_finale=bop_uguale, messaggi=msg, decisioni=dec, bop=sb,
                    passi_v=[{k: (round(v, 6) if isinstance(v, float) else v) for k, v in x.items()} for x in ev],
                    secondi=round(time.time() - t0, 1))


# ─────────────────────────────────────────── A32 ───────────────────────────────────────────
def _pv_finto_accetta(vbop: bool):
    """27/9 (ramo v-senza-bh) PassoV finto (StatoV VERO, motore/salti finti: `_pv_finto_vbop` o `_pv_finto_accumulo`)
    che REGISTRA la ConfigV ricevuta dal ciclo e la usa: con accetta "guadagno" il "salto" e' (3 + z_min + 2 z_guadagno_min)
    volte quello di bh, cosi' l'opzione cambia davvero i numeri della corsa."""
    installa, Base, ultimo = _pv_finto_vbop() if vbop else _pv_finto_accumulo()
    ultimo["cfg"] = []

    class PassoVFintoAccetta(Base):
        def __init__(self, fam, cfg, stato):
            super().__init__(fam, cfg, stato)
            ultimo["cfg"].append((cfg.accetta, cfg.z_min, cfg.z_guadagno_min, cfg.q_fdr, cfg.q_guadagno))

        def seleziona_e_salta(self, scale, rho):
            c = self.cfg
            f = 1.0 if c.accetta == "bh" else 3.0 + c.z_min + 2.0 * c.z_guadagno_min
            return super().seleziona_e_salta(scale, rho * f)
    return installa, PassoVFintoAccetta, ultimo


def prova_a32() -> bool:
    """27/9 (ramo v-senza-bh) --pv-accetta guadagno / --pv-z-min / --pv-z-guadagno-min nel CICLO con lo studente finto
    (CPU, senza /mnt; StatoV VERO, motore/salti finti che leggono la ConfigV), per PV e V-Bop: (a) la ConfigV che il
    ciclo passa a PassoV porta accetta/z_min/z_guadagno_min della config (e bh ai default); log della scelta; (b)
    corsa intera (6 passi) == fermata al passo 3 + ripresa AL BIT con l'opzione accesa (theta, esiti V, contatori,
    stato V-Bop); la corsa con l'opzione differisce da quella bh (l'opzione arriva al passo V); (c) accenderla alla
    RIPRESA (bh fino al passo 3, guadagno dopo) e' una decisione registrata («pv_accetta bh -> guadagno») e dalla
    ripresa la ConfigV e' quella nuova; (d) configurazioni rifiutate; CLI e campi di decisione."""
    from types import SimpleNamespace
    import e2e_pv_codici as PV
    t0 = time.time()
    passi = 6
    src, WS = sorgente_finta(n_train=4 * passi)
    orig = (PV.installa, PV.PassoV, PV.famiglia_scrivibile)
    esiti, ok = {}, True
    try:
        for modo in ("pv", "vbop"):
            base = dict(righe_per_passo=4, pv_codici=True, pv_modo=modo, pv_rho=0.5, pv_micro_curvatura=1)
            ca_bh = _ca_finta(**base)
            ca_on = _ca_finta(**base, pv_accetta="guadagno", pv_z_min=1.0, pv_z_guadagno_min=0.5)
            installa, PVF, ultimo = _pv_finto_accetta(modo == "vbop")
            PV.installa, PV.PassoV, PV.famiglia_scrivibile = installa, PVF, (lambda f: f)

            def corsa(d, ca, **kw):
                st = _StudenteFintoV()
                lg: list = []
                s = A.allena(st, src, d, ca, WS, WS, log=lg.append, **kw)
                return st, s, ultimo["pv"], lg
            dirs = {n: DIR / f"a32_{modo}_{n}" for n in ("on", "on_ripresa", "bh", "switch")}
            for d in dirs.values():
                if d.exists():
                    shutil.rmtree(d)
            del ultimo["cfg"][:]
            stA, sA, pvA, lgA = corsa(dirs["on"], ca_on)
            cfg_on = list(ultimo["cfg"])
            del ultimo["cfg"][:]
            _, sB1, _, _ = corsa(dirs["on_ripresa"], ca_on, ferma_a=3)
            del ultimo["cfg"][:]
            stB, sB, pvB, lgB = corsa(dirs["on_ripresa"], ca_on, riprendi=True)
            cfg_rip = list(ultimo["cfg"])
            del ultimo["cfg"][:]
            stC, sC, _, lgC = corsa(dirs["bh"], ca_bh)
            cfg_bh = list(ultimo["cfg"])
            corsa(dirs["switch"], ca_bh, ferma_a=3)
            del ultimo["cfg"][:]
            stD, sD, _, lgD = corsa(dirs["switch"], ca_on, riprendi=True)
            cfg_sw = list(ultimo["cfg"])
            atteso_on = ("guadagno", 1.0, 0.5, ca_on.pv_q_fdr, ca_on.pv_q_guadagno)
            atteso_bh = ("bh", 0.0, 0.0, ca_bh.pv_q_fdr, ca_bh.pv_q_guadagno)
            evA = [x["pv"] for x in sA["storia"] if x.get("pv")]
            evB = [x["pv"] for x in sB["storia"] if x.get("pv")]
            evC = [x["pv"] for x in sC["storia"] if x.get("pv")]
            theta_rip = all(torch.equal(a, b) for a, b in zip(stA.parametri(), stB.parametri()))
            theta_div = not all(torch.equal(a, b) for a, b in zip(stA.parametri(), stC.parametri()))
            cont = all(sA["pv"][k] == sB["pv"][k] for k in ("registri", "salti_tenuti", "passi_v", "rho", "tarato"))
            bop = True
            if modo == "vbop":
                ia, ib = pvA.stato.istantanea_bop(), pvB.stato.istantanea_bop()
                bop = ia["passi_bop"] == ib["passi_bop"] and all(
                    torch.equal(ia["stato"][f][n], ib["stato"][f][n]) for f in ia["stato"] for n in ia["stato"][f])
            cambi = next((r for r in lgD if r.startswith("ripresa con campi cambiati")), "")
            e = {"cfg_on": cfg_on == [atteso_on], "cfg_ripresa": cfg_rip == [atteso_on], "cfg_bh": cfg_bh == [atteso_bh],
                 "cfg_switch": cfg_sw == [atteso_on], "log_on": any("SENZA i test BH" in r for r in lgA),
                 "log_bh_assente": not any("SENZA i test BH" in r for r in lgC) and sB1["passo"] == 3,
                 "ripresa_theta_al_bit": theta_rip, "ripresa_esiti_v": evA == evB, "ripresa_contatori": cont,
                 "ripresa_bop": bop, "diverso_da_bh": theta_div and evA != evC,
                 "switch_registrato": ("pv_accetta bh -> guadagno" in cambi and "pv_z_min 0.0 -> 1.0" in cambi
                                       and "pv_z_guadagno_min 0.0 -> 0.5" in cambi),
                 "switch_config_salvata": sD["config"]["pv_accetta"] == "guadagno" and sD["passo"] == passi,
                 "passi_v": len(evA)}
            e["ok"] = all(v for k, v in e.items() if k != "passi_v") and e["passi_v"] == passi - 1
            ok &= e["ok"]
            esiti[modo] = e
    finally:
        PV.installa, PV.PassoV, PV.famiglia_scrivibile = orig
    rifiuti = []
    for kw in ({"pv_accetta": "gain"}, {"pv_z_min": 1.0}, {"pv_z_guadagno_min": 1.0},
               {"pv_accetta": "guadagno", "pv_z_min": -1.0}, {"pv_accetta": "guadagno", "pv_z_guadagno_min": -0.5}):
        try:
            _ca_finta(pv_codici=True, **kw)
            rifiuti.append(False)
        except ValueError:
            rifiuti.append(True)
    testo = (QUI / "e2e_allena.py").read_text()
    cli = all(f'"--{x}"' in testo for x in ("pv-accetta", "pv-z-min", "pv-z-guadagno-min"))
    arg = dict(lr=1e-3, righe_passo=2, righe_micro=1, righe_micro_valuta=2, epoche=1, heldout_ogni=2, ripresa_ogni=2,
               pezzo_kl=10, rotazione=False, K0=A.ConfigAllena.K0, pv_codici=False, pv_rho=0.01, pv_micro_curvatura=2,
               pv_n_proposta=2, pv_q_fdr=0.1, pv_q_guadagno=0.1, pv_passi_accumulo=1, pv_modo="pv", pv_gamma=0.5,
               pv_curvatura_ogni=1, pv_rho_adattivo=False, pv_rho_max=0.1, perdita="kl", gamma=None, beta2=0.95,
               ema_theta=0.0, heldout_righe_rapide=0, heldout_pieno_ogni=0, allena_f32="", lr_f32=1e-5,
               escludi_domini="wiki")
    c0 = A._config_da_arg(SimpleNamespace(**arg))
    c1 = A._config_da_arg(SimpleNamespace(**arg, pv_accetta="guadagno", pv_z_min=2.0, pv_z_guadagno_min=1.0))
    cli &= ((c0.pv_accetta, c0.pv_z_min, c0.pv_z_guadagno_min) == ("bh", 0.0, 0.0)
            and (c1.pv_accetta, c1.pv_z_min, c1.pv_z_guadagno_min) == ("guadagno", 2.0, 1.0))
    campi = {"pv_accetta", "pv_z_min", "pv_z_guadagno_min"} <= A._CAMPI_DI_DECISIONE
    dflt = A.ConfigAllena()
    ok &= all(rifiuti) and cli and campi and (dflt.pv_accetta, dflt.pv_z_min, dflt.pv_z_guadagno_min) == ("bh", 0.0, 0.0)
    return registra("A32_pv_accetta_guadagno_nel_ciclo_studente_finto", ok, esiti=esiti, rifiuti_config=rifiuti, cli=cli,
                    campi_decisione=campi, secondi=round(time.time() - t0, 1))


def prova_a33() -> bool:
    """27/9 (ramo v-senza-bh) --pv-accetta guadagno col CODICE VERO del passo V nel ciclo: strato 0 VERO di Thor-T
    (legge il GGUF), motore torch su CPU, maestro con 0,5% dei codici spostati (come A19/A31), V-Bop, z_guadagno_min 1,
    rho 0,05. (a) corsa intera di 6 passi contro fermata al passo 3 + ripresa: theta, byte, sha dei registri, stato
    V-Bop ed esiti V identici AL BIT; (b) MISURA: la stessa corsa con bh (stessi dati): testati, scoperte,
    convenienti, salti e tenuti per passo V, e sonda (KL prima/dopo) nei due modi."""
    import e2e_pv_codici as PV
    t0 = time.time()
    st_t = studente((0,))
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
    th = theta_stella(st_t)
    passi = 6
    src = sorgente(st_t, 6 * passi, 4, th)
    del st_t
    base = dict(pv_codici=True, righe_per_passo=6, righe_per_micro=1, heldout_ogni=2, ripresa_ogni=2, pv_rho=0.05,
                pv_modo="vbop")
    ca_on = _ca(**base, pv_accetta="guadagno", pv_z_guadagno_min=1.0)
    ca_bh = _ca(**base)

    def corsa(dir_c, ca, **kw):
        st = studente((0,))
        WS = testa_ristretta(st)
        lg: list = []
        return st, A.allena(st, src, dir_c, ca, WS, WS, log=lg.append, **kw), lg
    dA, dB, dC = DIR / "a33_intera", DIR / "a33_ripresa", DIR / "a33_bh"
    for d in (dA, dB, dC):
        if d.exists():
            shutil.rmtree(d)
    stA, sA, lgA = corsa(dA, ca_on)
    _, sB1, _ = corsa(dB, ca_on, ferma_a=3)
    stB, sB, _ = corsa(dB, ca_on, riprendi=True)
    stC, sC, _ = corsa(dC, ca_bh)
    theta = all(torch.equal(a, b) for a, b in zip(stA.parametri(), stB.parametri()))
    byte = all(torch.equal(stA.esperti(0).fam[n].blocchi, stB.esperti(0).fam[n].blocchi) for n in stA.esperti(0).fam)
    evA = [x["pv"] for x in sA["storia"] if x.get("pv")]
    evB = [x["pv"] for x in sB["storia"] if x.get("pv")]
    evC = [x["pv"] for x in sC["storia"] if x.get("pv")]
    cont = all(sA["pv"][k] == sB["pv"][k] for k in ("registri", "salti_tenuti", "passi_v", "rho", "sha"))
    ia = torch.load(dA / sA["pv"]["bop"]["file"], weights_only=True)
    ib = torch.load(dB / sB["pv"]["bop"]["file"], weights_only=True)
    bop = ia["passi_bop"] == ib["passi_bop"] and all(torch.equal(ia["stato"][f][n], ib["stato"][f][n])
                                                     for f in ia["stato"] for n in ia["stato"][f])
    log_ok = any("SENZA i test BH" in r for r in lgA)
    misura = {nome: [{k: (round(v, 6) if isinstance(v, float) else v) for k, v in x.items()
                      if k in ("testati", "scoperte", "convenienti", "salti", "tenuto", "rho", "sonda_prima", "sonda_dopo",
                               "guadagno_previsto", "con_candidato")} for x in ev]
              for nome, ev in (("guadagno_zg1", evA), ("bh", evC))}
    ok = (theta and byte and evA == evB and cont and bop and log_ok and sB1["passo"] == 3 and len(evA) == passi - 1
          and sum(x["salti"] for x in evA) > 0)
    return registra("A33_pv_accetta_guadagno_strato0_vero_ripresa_al_bit", ok, ripresa_theta_al_bit=theta,
                    ripresa_byte_al_bit=byte, esiti_v_uguali=evA == evB, contatori_uguali=cont, stato_bop_al_bit=bop,
                    log=log_ok, pv_guadagno=sA["pv"], pv_bh=sC["pv"], misura_per_passo_v=misura,
                    heldout={"guadagno_zg1": [(x["passo"], round(x["metrica"], 6)) for x in sA["storia_heldout"]],
                             "bh": [(x["passo"], round(x["metrica"], 6)) for x in sC["storia_heldout"]]},
                    secondi=round(time.time() - t0, 1))


# ─────────────────────────────────────────── A21 ───────────────────────────────────────────
class _SenzaRumore:
    """Al posto di `MisuraRumore`: nessun gancio, nessuna misura (prova 'con e senza')."""
    def __init__(self, *a, **k):
        pass

    def aggancia(self):
        pass

    def micro(self, n):
        pass

    def fine(self, n, r):
        pass

    def risultato(self):
        return None


def _rumore_diretto(st, src, WS, gruppi: list, K: float, opt=None) -> dict:
    """Riferimento: gradiente di OGNI micro-lotto calcolato da solo con autograd (kl_diretta, non a pezzi), float64,
    scale vive (d0 != 0); |G_k|^2 del gradiente della perdita media per token del micro-lotto, |G|^2 del passo e le
    stesse col precondizionatore di Adam in unita' vere P = K / (sqrt(v / (1 - beta2^passo)) + eps)."""
    par = st.parametri()
    vivi = [(st.fam[n].d0 != 0) for n in st.FAM]
    gk, nk = [], []
    for rr in gruppi:
        ids = torch.from_numpy(np.stack([np.asarray(src.r["train"][i][0], dtype=np.int64) for i in rr]))
        hT = torch.cat([src.r["train"][i][1].reshape(-1, WS.shape[1]) for i in rr])
        with torch.enable_grad():
            h = st.esce(st._f(st.emb[ids.reshape(-1)]))
            gg = torch.autograd.grad(A.kl_diretta(h, WS, hT, WS), par)
        gk.append([torch.where(v, g, torch.zeros_like(g)).double() for g, v in zip(gg, vivi)])
        nk.append(ids.numel())
    N = sum(nk)
    tot = [sum(g[j] for g in gk) / N for j in range(len(par))]
    out = {"g2_micro": [sum(float((g / n).pow(2).sum()) for g in gg) for gg, n in zip(gk, nk)],
           "g2_passo": sum(float(g.pow(2).sum()) for g in tot), "n": nk, "N": N}
    if opt is not None:
        gr = opt.param_groups[0]
        den = []
        for p in par:
            s_ = opt.state[p]
            bc = (1 - gr["betas"][1] ** float(s_["step"])) ** 0.5
            den.append((s_["exp_avg_sq"].double().sqrt() / bc + gr["eps"]) / K)
        out["p_micro"] = [sum(float(((g / n).pow(2) / d).sum()) for g, d in zip(gg, den)) for gg, n in zip(gk, nk)]
        out["p_passo"] = sum(float((g.pow(2) / d).sum()) for g, d in zip(tot, den))
    return out


def _stima_rif(g2k: list, nk: list, g2b: float, N: int) -> dict:
    """McCandlish app. A scritta a parte: E|G_B|^2 = |G|^2 + S/B con B_piccolo = 1/media(1/n_k), B_grande = N."""
    bs = len(nk) / sum(1 / n for n in nk)
    ms = sum(g2k) / len(g2k)
    G2 = (N * g2b - bs * ms) / (N - bs)
    S_ = (ms - g2b) / (1 / bs - 1 / N)
    return {"G2": G2, "trSigma": S_, "B_simple_token": S_ / G2}


def prova_a21() -> bool:
    """27/9 velocita' — rumore del gradiente (McCandlish 2018, app. A) in `gradiente_passo`: (a) studente finto in
    float64 (scale morte comprese: d0 = 0 su 5 scale), 4 micro-lotti da 1 riga e 2 disuguali (3 + 1), K = 2^10, Adam
    gia' con stato: |G_k|^2, |G_passo|^2, versione precondizionata e stime (|G|^2, tr(Sigma), B_simple) contro il
    gradiente di ogni micro-lotto calcolato da solo con autograd, errore relativo fp64 <= 1e-10; theta.grad IDENTICO
    al bit con e senza la misura. (b) ciclo (allena, fp32) identico AL BIT con e senza la misura (theta, Adam, stato
    tolto il rumore); rumore nella storia di ogni passo (G1 senza Adam, poi precondizionato) e nel registro."""
    t0 = time.time()
    src, WS = sorgente_finta(n_train=12)
    WS64 = WS.double()
    car = A.Caricatore(src, torch.device("cpu"))
    K = 2.0 ** 10
    err, dett = 0.0, {}
    for nome, rpm in (("4x1", 1), ("3+1", 3)):
        st = StudenteFinto(dtype=torch.float64)
        st.fam["gate"].d0[0, :5] = 0.0
        ca = _ca_finta(righe_per_passo=4, righe_per_micro=rpm)
        opt = torch.optim.AdamW(st.parametri(), lr=3e-2, betas=(0.9, 0.95), eps=1e-6 * K, weight_decay=0.0)
        A.gradiente_passo(st, car, "train", [4, 5, 6, 7], ca, WS64, WS64, K)
        opt.step()
        opt.zero_grad(set_to_none=True)
        rum = A.MisuraRumore(st, opt, K)
        A.gradiente_passo(st, car, "train", [0, 1, 2, 3], ca, WS64, WS64, K, rumore=rum)
        r = rum.risultato()
        g_con = [p.grad.clone() for p in st.parametri()]
        opt.zero_grad(set_to_none=True)
        A.gradiente_passo(st, car, "train", [0, 1, 2, 3], ca, WS64, WS64, K)
        grad_al_bit = all(torch.equal(a, p.grad) for a, p in zip(g_con, st.parametri()))
        gruppi = [[0, 1, 2, 3][j:j + rpm] for j in range(0, 4, rpm)]
        d = _rumore_diretto(st, src, WS64, gruppi, K, opt)
        e = {"g2_micro": max(abs(a - b) / b for a, b in zip(r["g2_micro"], d["g2_micro"])),
             "g2_passo": abs(r["g2_passo"] - d["g2_passo"]) / d["g2_passo"],
             "p_micro": max(abs(a - b) / b for a, b in zip(r["precondizionata"]["g2_micro"], d["p_micro"])),
             "p_passo": abs(r["precondizionata"]["g2_passo"] - d["p_passo"]) / d["p_passo"]}
        for chiave, g2k, g2b in (("stima", d["g2_micro"], d["g2_passo"]), ("stima_p", d["p_micro"], d["p_passo"])):
            rif = _stima_rif(g2k, d["n"], g2b, d["N"])
            mis = r["stima"] if chiave == "stima" else r["precondizionata"]["stima"]
            for k_ in rif:
                e[f"{chiave}.{k_}"] = abs(mis[k_] - rif[k_]) / abs(rif[k_])
        err = max(err, max(e.values()))
        dett[nome] = {"errori": {k_: f"{v:.1e}" for k_, v in e.items()}, "grad_al_bit": grad_al_bit,
                      "micro": r["micro"], "token_micro": r["token_micro"],
                      "B_simple_token": r["stima"]["B_simple_token"],
                      "B_simple_p_token": r["precondizionata"]["stima"]["B_simple_token"],
                      "contributi_multipli": r["contributi_multipli"]}
        ok_a_ = grad_al_bit and not r["contributi_multipli"] and r["micro"] == len(gruppi)
        dett[nome]["ok"] = ok_a_
        err = err if ok_a_ else float("inf")
    car.chiudi()
    ok_a = err <= 1e-10
    # (b) ciclo con e senza la misura
    src, WS = sorgente_finta()
    ca = _ca_finta()
    esiti = {}
    orig = A.MisuraRumore
    try:
        for nome, cls in (("con", orig), ("senza", _SenzaRumore)):
            A.MisuraRumore = cls
            d_ = DIR / f"a21_{nome}"
            if d_.exists():
                shutil.rmtree(d_)
            st = StudenteFinto()
            reg = []
            stato = A.allena(st, src, d_, ca, WS, WS, log=reg.append)
            esiti[nome] = (st, stato, A.carica_ripresa(d_, "cpu"), reg)
    finally:
        A.MisuraRumore = orig
    (sc, zc, rc, regc), (ss, zs, rs, _) = esiti["con"], esiti["senza"]
    th = all(torch.equal(a, b) for a, b in zip(sc.parametri(), ss.parametri()))
    so, sr = rc["opt"]["state"], rs["opt"]["state"]
    adam = all(torch.equal(so[i][k], sr[i][k]) for i in so for k in ("exp_avg", "exp_avg_sq", "step"))
    a_, b_ = _senza_tempi(zc), _senza_tempi(zs)
    storia_r = [x["rumore"] for x in a_["storia"]]
    presente = _togli_rumore(a_)
    _togli_rumore(b_)
    ok_b = (th and adam and a_ == b_ and presente and storia_r[0]["precondizionata"] is None
            and all(x["precondizionata"]["stima"] is not None for x in storia_r[1:])
            and all(x["stima"] is not None and x["micro"] == 2 for x in storia_r)
            and sum("rumore del gradiente: |G|^2" in x for x in regc) == zc["passo"])
    return registra("A21_rumore_del_gradiente", ok_a and ok_b, errore_max_fp64=err, soglia=1e-10, a=ok_a, fp64=dett,
                    ciclo_theta=th, ciclo_adam=adam, ciclo_stato=a_ == b_, b=ok_b,
                    B_simple_token_per_passo=[round(x["stima"]["B_simple_token"], 2) for x in storia_r],
                    B_simple_p_token_per_passo=[None if x["precondizionata"] is None else
                                                round(x["precondizionata"]["stima"]["B_simple_token"], 2)
                                                for x in storia_r],
                    riga_registro=next(x for x in regc if "rumore del gradiente" in x and "precond" in x)[:300],
                    secondi=round(time.time() - t0, 1))


# ─────────────────────────────────────────── A22 ───────────────────────────────────────────
BASE_VELOCITA = "d22c2cf"       # ramo velocita' nato qui: default identico AL BIT a questo commit


def _sha_cartella(d: Path) -> dict:
    """{percorso relativo: sha256} dei file di una corsa (senza registri di testo con le ore)."""
    return {str(x.relative_to(d)): A.sha256_file(x) for x in sorted(d.rglob("*")) if x.is_file()}


def prova_a22() -> bool:
    """27/9 velocita' — (a) DEFAULT identico AL BIT al commit d22c2cf: ciclo intero (studente finto): theta, Adam,
    stato (tolto il rumore, solo registro), file della corsa (stessi nomi), istantanee del migliore AL BYTE, riprese
    senza "ema"; nessuna chiave nuova nello stato. (b) EMA (--ema-theta 0,5): theta e Adam IDENTICI a EMA spenta;
    theta_EMA == ricorrenza lerp fp32 ricalcolata dalle theta di ogni passo, al bit; metrica_ema == valuta con
    theta = theta_EMA; migliore_ema = minimo delle metrica_ema; migliore_<p>/ema/ e migliore_ema_<p>/ leggibili
    (carica_istantanea) con scale == fp16(W exp(theta_EMA)); 3 passi + ripresa + 3 == 6 filati AL BIT (theta, Adam,
    theta_EMA, stato, istantanee). (c) held-out ridotto (2 righe rapide ogni passo, piene ogni 3): misure piene ai
    passi 0, 3, 6, rapide a 1, 2, 4, 5 su 2 righe; migliore (e istantanee) SOLO fra le piene; (d) --beta2 in Adam,
    cambiato alla ripresa (decisione registrata); campi nuovi fra esecuzione/decisione; CLI."""
    t0 = time.time()
    src, WS = sorgente_finta()
    ca = _ca_finta()
    # (a)
    vec = _modulo_prima(BASE_VELOCITA)
    es = {}
    for nome, mod in (("nuovo", A), ("prima", vec)):
        d = DIR / f"a22_default_{nome}"
        if d.exists():
            shutil.rmtree(d)
        st = StudenteFinto()
        stato = mod.allena(st, src, d, ca if mod is A else _config_per(mod, ca), WS, WS, log=muto)
        es[nome] = (st, stato, mod.carica_ripresa(d, "cpu"), d)
    (sn, zn, rn, dn), (sp, zp, rp, dp) = es["nuovo"], es["prima"]
    th = all(torch.equal(a, b) for a, b in zip(sn.parametri(), sp.parametri()))
    so, sr = rn["opt"]["state"], rp["opt"]["state"]
    adam = all(torch.equal(so[i][k], sr[i][k]) for i in so for k in ("exp_avg", "exp_avg_sq", "step")) \
        and rn["opt"]["param_groups"] == rp["opt"]["param_groups"]
    a_, b_ = _senza_tempi(zn), _senza_tempi(zp)
    nuovi_cfg = {k: a_["config"].pop(k) for k in set(a_["config"]) - set(b_["config"])}
    _togli_rumore(a_)
    stati = a_ == b_ and set(zn) == set(zp)
    fn, fp_ = _sha_cartella(dn), _sha_cartella(dp)
    ist = {k: v for k, v in fn.items() if k.startswith("migliore_") and not k.endswith("meta.json")}
    file_ok = (set(fn) == set(fp_) and all(fn[k] == fp_[k] for k in ist) and len(ist) > 0
               and set(rn) == set(rp) and "ema" not in rn)
    ok_a = th and adam and stati and file_ok and nuovi_cfg == {"heldout_righe_rapide": 0, "heldout_pieno_ogni": 0,
                                                                "ema_theta": 0.0, "allena_f32": [], "lr_f32": 1e-5,
                                                                **_CFG_VBOP_DEFAULT, **_CFG_PESO_DEFAULT,
                                                                **_CFG_MC_DEFAULT,
                                                                # 27/9 --lora-nonesperti (ai default: spenta)
                                                                "lora_rango": 0, "lora_alpha": None,
                                                                "lora_famiglie": [], "lr_lora": 1e-4, "lora_tipi": [],
                                                                **_CFG_SOPHIA_DEFAULT, **_CFG_LORA_GA_DEFAULT,
                                                                **_CFG_INTEGRATA_PROPRI, **_CFG_BF16X3_DEFAULT,
                                                                **_CFG_LOTA_DEFAULT}
    # (b) EMA
    dec = 0.5
    ca_e = _ca_finta(ema_theta=dec, ripresa_ogni=1, tieni_riprese=10)
    d_e, d_r = DIR / "a22_ema", DIR / "a22_ema_ripresa"
    for d in (d_e, d_r):
        if d.exists():
            shutil.rmtree(d)
    st_e = StudenteFinto()
    z_e = A.allena(st_e, src, d_e, ca_e, WS, WS, log=muto)
    uguale_spenta = all(torch.equal(a, b) for a, b in zip(st_e.parametri(), sn.parametri()))
    r_e = A.carica_ripresa(d_e, "cpu")
    adam_e = all(torch.equal(r_e["opt"]["state"][i][k], so[i][k]) for i in so for k in ("exp_avg", "exp_avg_sq", "step"))
    chiavi = [A._chiave(k) for k in st_e.chiavi()]
    ema_rif = {c: torch.zeros(16, 16) for c in chiavi}
    ricorrenza = True
    ema_per_passo = {}
    for p_ in range(1, z_e["passo"] + 1):
        rp_ = torch.load(d_e / f"ripresa_{p_:06d}.pt", weights_only=False)
        for c in chiavi:
            ema_rif[c].lerp_(rp_["theta"][c], 1.0 - dec)
            ricorrenza &= torch.equal(ema_rif[c], rp_["ema"][c])
        ema_per_passo[p_] = {c: rp_["ema"][c].clone() for c in chiavi}
    h_e = z_e["storia_heldout"]
    st_v = StudenteFinto()
    with torch.no_grad():
        for k, p in zip(st_v.chiavi(), st_v.parametri()):
            p.copy_(r_e["ema"][A._chiave(k)])
    car = A.Caricatore(src, torch.device("cpu"))
    v_ema = A.valuta(st_v, car, list(range(len(src.r["heldout"]))), ca_e, WS, WS)
    car.chiudi()
    metrica_ok = h_e[-1]["metrica_ema"] == v_ema["metrica"] and h_e[-1]["ema"] == v_ema
    me = z_e["migliore_ema"]
    mig_ema_ok = me["metrica"] == min(x["metrica_ema"] for x in h_e) and all("metrica_ema" in x for x in h_e)
    ist_ok = True
    for nome_d, passo_d in ((me["dir"], me["passo"]), (f"{z_e['migliore']['dir']}/ema", z_e["migliore"]["passo"])):
        if passo_d == 0:
            continue
        ii = A.carica_istantanea(d_e / nome_d)
        th_i = torch.load(d_e / nome_d / "theta.pt", weights_only=True)
        for n in StudenteFinto.FAM:
            atteso = (st_e.W[n] * torch.exp(ema_per_passo[passo_d][f"0.{n}"])).to(torch.float16).numpy()
            ist_ok &= np.array_equal(ii["scale"][f"blk.0.{n}"].view(np.uint16), atteso.view(np.uint16))
            ist_ok &= torch.equal(th_i[f"0.{n}"], ema_per_passo[passo_d][f"0.{n}"])
    citata = (d_e / me["dir"]).is_dir() and json.loads((d_e / "ripresa_ultima.json").read_text()).get("migliore_ema") == me["dir"]
    # ripresa al bit con EMA
    A.allena(StudenteFinto(), src, d_r, ca_e, WS, WS, ferma_a=3, log=muto)
    st_r = StudenteFinto()
    z_r = A.allena(st_r, src, d_r, ca_e, WS, WS, riprendi=True, log=muto)
    r_r = A.carica_ripresa(d_r, "cpu")
    rip_ok = (all(torch.equal(a, b) for a, b in zip(st_e.parametri(), st_r.parametri()))
              and all(torch.equal(r_e["opt"]["state"][i][k], r_r["opt"]["state"][i][k]) for i in so
                      for k in ("exp_avg", "exp_avg_sq", "step"))
              and all(torch.equal(r_e["ema"][c], r_r["ema"][c]) for c in chiavi)
              and _senza_tempi(z_e) == _senza_tempi(z_r)
              and all(A.sha256_file(d_e / me["dir"] / f) == A.sha256_file(d_r / me["dir"] / f)
                      for f in ("scale.npz",)))
    ok_b = uguale_spenta and adam_e and ricorrenza and metrica_ok and mig_ema_ok and ist_ok and citata and rip_ok
    # (c) held-out ridotto
    ca_h = _ca_finta(heldout_ogni=1, heldout_righe_rapide=2, heldout_pieno_ogni=3)
    d_h = DIR / "a22_rapido"
    if d_h.exists():
        shutil.rmtree(d_h)
    reg_h = []
    z_h = A.allena(StudenteFinto(), src, d_h, ca_h, WS, WS, log=reg_h.append)
    pieni = [x["passo"] for x in z_h["storia_heldout"]]
    rapidi = [x["passo"] for x in z_h["storia_heldout_rapido"]]
    T_ = int(np.asarray(src.r["heldout"][0][0]).shape[0])
    mig_h = z_h["migliore"]
    dir_mig = sorted(x.name for x in d_h.glob("migliore_*"))
    ok_c = (pieni == [0, 3, 6] and rapidi == [1, 2, 4, 5]
            and all(x["token"] == 2 * T_ and x["righe"] == 2 for x in z_h["storia_heldout_rapido"])
            and all(x["token"] == 4 * T_ for x in z_h["storia_heldout"])
            and mig_h["passo"] in pieni and mig_h["metrica"] == min(x["metrica"] for x in z_h["storia_heldout"])
            and all(int(n.split("_")[1]) in pieni for n in dir_mig)
            and A.righe_rapide([0, 1, 2, 3], 2) == [0, 2] and A.righe_rapide([0, 1, 2, 3], 0) == [0, 1, 2, 3]
            and sum("held-out RAPIDO" in x for x in reg_h) == 4)
    # (d) beta2
    d_b = DIR / "a22_beta2"
    if d_b.exists():
        shutil.rmtree(d_b)
    A.allena(StudenteFinto(), src, d_b, _ca_finta(beta2=0.9), WS, WS, ferma_a=2, log=muto)
    b1 = A.carica_ripresa(d_b, "cpu")["opt"]["param_groups"][0]["betas"]
    reg_b = []
    A.allena(StudenteFinto(), src, d_b, _ca_finta(beta2=0.99), WS, WS, riprendi=True, ferma_a=4, log=reg_b.append)
    b2 = A.carica_ripresa(d_b, "cpu")["opt"]["param_groups"][0]["betas"]
    try:
        A.ConfigAllena(beta2=1.0)
        err_b = False
    except ValueError:
        err_b = True
    testo = (QUI / "e2e_allena.py").read_text()
    cli = all(f'"--{x}"' in testo for x in ("beta2", "ema-theta", "heldout-righe-rapide", "heldout-pieno-ogni"))
    campi = ({"beta2", "ema_theta"} <= A._CAMPI_DI_DECISIONE
             and {"heldout_righe_rapide", "heldout_pieno_ogni"} <= A._CAMPI_DI_ESECUZIONE)
    dflt = A.ConfigAllena()
    ok_d = (tuple(b1) == (0.9, 0.9) and tuple(b2) == (0.9, 0.99) and any("betas" in x for x in reg_b) and err_b
            and cli and campi and (dflt.beta2, dflt.ema_theta, dflt.heldout_righe_rapide, dflt.heldout_pieno_ogni)
            == (0.95, 0.0, 0, 0) and "0,95^(r/32)" in A.ConfigAllena.__doc__)
    return registra("A22_heldout_ridotto_ema_beta2", ok_a and ok_b and ok_c and ok_d, base=BASE_VELOCITA,
                    a_default_al_bit=ok_a, theta=th, adam=adam, stato=stati, file_e_istantanee=file_ok,
                    file_corsa=sorted(fn), b_ema=ok_b, ema_uguale_spenta=uguale_spenta, ricorrenza_lerp=ricorrenza,
                    metrica_ema=metrica_ok, migliore_ema=me, istantanee_ema=ist_ok, ripresa_ema_al_bit=rip_ok,
                    heldout_ema=[(x["passo"], round(x["metrica"], 6), round(x["metrica_ema"], 6)) for x in h_e],
                    c_rapido=ok_c, pieni=pieni, rapidi=rapidi, migliore_rapido=mig_h["passo"], cartelle=dir_mig,
                    d_beta2=ok_d, betas=(b1, b2), secondi=round(time.time() - t0, 1))


def prova_a34() -> bool:
    """1/10 TT5-ottimizzato: V-Bop con --pv-passi-accumulo N (= passo V ogni N passi) nel CICLO con lo studente finto
    (StatoV VERO con V-Bop, motore/salti finti come A29). 7 passi, N = 3, contro fermata al passo 4 (a meta' fra due
    decisioni) e ripresa: (a) la configurazione e' accettata; (b) decisioni (selezione + salti + sonda) SOLO ai passi
    3, 6 e 7 (ultimo della corsa); negli altri passi V solo statistiche: (c) la EMA di V-Bop si aggiorna a OGNI passo V
    (passi_bop = 6 = passi V 2..7), la curvatura a ogni passo; (d) ripresa: theta, esiti V, contatori, stato V-Bop
    finale AL BIT contro la corsa intera; (e) con N = 1 i passi V decidono tutti (6 decisioni), come A29."""
    import e2e_pv_codici as PV
    t0 = time.time()
    passi = 7
    src, WS = sorgente_finta(n_train=4 * passi)
    esiti = {}
    ok = True
    for N in (3, 1):
        ca = _ca_finta(righe_per_passo=4, pv_codici=True, pv_modo="vbop", pv_passi_accumulo=N, pv_rho_adattivo=True,
                       pv_rho=0.5, pv_rho_max=4.0, pv_micro_curvatura=1)
        orig = (PV.installa, PV.PassoV, PV.famiglia_scrivibile)
        installa, PVF, ultimo = _pv_finto_vbop()
        n_agg = {"n": 0}
        orig_agg = PV.StatoV.aggiorna_bop

        def agg(self, *a_, **k_):
            n_agg["n"] += 1
            return orig_agg(self, *a_, **k_)
        PV.installa, PV.PassoV, PV.famiglia_scrivibile = installa, PVF, (lambda f: f)
        PV.StatoV.aggiorna_bop = agg
        righe_log: list = []

        def corsa(d, **kw):
            st = _StudenteFintoV()
            return st, A.allena(st, src, d, ca, WS, WS, log=righe_log.append, **kw), ultimo["pv"]
        try:
            dA, dB = DIR / f"a34_intera_{N}", DIR / f"a34_ripresa_{N}"
            for d in (dA, dB):
                if d.exists():
                    shutil.rmtree(d)
            stA, sA, pvA = corsa(dA)
            agg_intera = n_agg["n"]
            _, sB1, _ = corsa(dB, ferma_a=4)
            stB, sB, pvB = corsa(dB, riprendi=True)
        finally:
            PV.installa, PV.PassoV, PV.famiglia_scrivibile = orig
            PV.StatoV.aggiorna_bop = orig_agg

        def uguali(a, b):
            return (a["passi_bop"] == b["passi_bop"] and set(a["stato"]) == set(b["stato"])
                    and all(torch.equal(a["stato"][f][n], b["stato"][f][n]) for f in a["stato"] for n in a["stato"][f]))
        passi_dec = [x["passo"] for x in sA["storia"] if x.get("pv")]
        evA = [x["pv"] for x in sA["storia"] if x.get("pv")]
        evB = [x["pv"] for x in sB["storia"] if x.get("pv")]
        theta = all(torch.equal(a, b) for a, b in zip(stA.parametri(), stB.parametri()))
        bop_fin = uguali(pvA.stato.istantanea_bop(), pvB.stato.istantanea_bop())
        cont = all(sA["pv"][k] == sB["pv"][k] for k in ("registri", "salti_tenuti", "passi_v", "rho", "tarato"))
        n_solo = sum(1 for r in righe_log if "solo statistiche V-Bop" in r)
        attese = [3, 6, 7] if N == 3 else [2, 3, 4, 5, 6, 7]
        e = {"passi_con_decisione": passi_dec, "attese": attese, "passi_bop_finale": pvA.stato.passi_bop,
             "aggiorna_bop_chiamate_corsa_intera": agg_intera, "passi_v_contati": sA["pv"]["passi_v"],
             "theta_ripresa_al_bit": theta, "bop_finale_al_bit": bop_fin, "esiti_v_uguali": evA == evB,
             "contatori_uguali": cont, "righe_solo_statistiche": n_solo, "fermata": sB1["passo"],
             "curvatura": [x.get("curvatura") for x in evA]}
        esiti[f"N{N}"] = e
        ok &= (passi_dec == attese and pvA.stato.passi_bop == passi - 1 and agg_intera == passi - 1
               and sA["pv"]["passi_v"] == len(attese) and theta and bop_fin and evA == evB and cont
               and sB1["passo"] == 4 and all(e["curvatura"]))
        ok &= (n_solo > 0) if N == 3 else (n_solo == 0)
    return registra("A34_vbop_decisione_ogni_N_passi", ok, **esiti, secondi=round(time.time() - t0, 1))


def _uguali_profondi(a, b) -> bool:
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.dtype == b.dtype
                and a.shape == b.shape and torch.equal(a.cpu(), b.cpu()))
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_uguali_profondi(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(_uguali_profondi(x, y) for x, y in zip(a, b))
    return a == b


def prova_a35() -> bool:
    """1/10 tt5-ottimizzato --scritture-asincrone (studente finto, V-Bop finto come A29, decisione ogni 2 passi):
    (a) GEMELLO: a ogni salvataggio, dagli stessi oggetti nello stesso istante, la via SINCRONA scrive in una cartella
        gemella; a fine corsa ogni ripresa_*.pt e bop_pv_*.pt della corsa asincrona, riletto, e' identico AL BIT a quello
        sincrono (theta, Adam, EMA, stato del ciclo con lo sha del file V-Bop), stessi file tenuti;
    (b) RIPRESA: corsa asincrona fermata al passo 3 e ripresa == corsa sincrona intera (theta, stato V-Bop, esiti V,
        contatori AL BIT);
    (c) un errore di scrittura nel filo si rilancia nel ciclo (la corsa si ferma, non continua senza riprese)."""
    import copy
    import e2e_pv_codici as PV
    t0 = time.time()
    passi = 6
    src, WS = sorgente_finta(n_train=4 * passi)
    base = dict(righe_per_passo=4, pv_codici=True, pv_modo="vbop", pv_passi_accumulo=2, pv_rho_adattivo=True,
                pv_rho=0.5, pv_rho_max=4.0, pv_micro_curvatura=1, ripresa_ogni=1, ema_theta=0.9)
    ca_s = _ca_finta(**base)
    ca_a = _ca_finta(**base, scritture_asincrone=True)
    orig = (PV.installa, PV.PassoV, PV.famiglia_scrivibile)
    orig_salva = A.salva_ripresa
    esiti = {}

    def corsa(ca, d, **kw):
        installa, PVF, ultimo = _pv_finto_vbop()
        PV.installa, PV.PassoV, PV.famiglia_scrivibile = installa, PVF, (lambda f: f)
        st = _StudenteFintoV()
        return st, A.allena(st, src, d, ca, WS, WS, log=lambda *_: None, **kw), ultimo["pv"]
    try:
        dA, dB, dG, dS = (DIR / f"a35_{x}" for x in ("asinc", "asinc_ripresa", "gemello", "sinc"))
        for d in (dA, dB, dG, dS):
            if d.exists():
                shutil.rmtree(d)
        dG.mkdir(parents=True)

        def doppio(dir_c, st, opt, stato, tieni, ema=None, scrittore=None, bop=None, dopo=None):
            if scrittore is not None:
                s2 = copy.deepcopy(stato)
                if bop is not None:
                    sha = A._torch_save_atomico(bop[1], dG / bop[0])
                    s2["pv"]["bop"] = {"file": bop[0], "sha": sha, **bop[2]}
                orig_salva(dG, st, opt, s2, tieni, ema=ema)
            return orig_salva(dir_c, st, opt, stato, tieni, ema=ema, scrittore=scrittore, bop=bop, dopo=dopo)
        A.salva_ripresa = doppio
        try:
            stA, sA, pvA = corsa(ca_a, dA)
        finally:
            A.salva_ripresa = orig_salva
        rip_a = sorted(f.name for f in dA.glob("ripresa_*.pt"))
        rip_g = sorted(f.name for f in dG.glob("ripresa_*.pt"))
        uguali = rip_a == rip_g and len(rip_a) > 0 and all(
            _uguali_profondi(torch.load(dA / f, weights_only=False), torch.load(dG / f, weights_only=False))
            for f in rip_a)
        bop_a = sorted(f.name for f in dA.glob("bop_pv_*.pt"))
        bop_ok = len(bop_a) > 0 and all(
            _uguali_profondi(torch.load(dA / f, weights_only=False), torch.load(dG / f, weights_only=False))
            for f in bop_a)
        sha_bop = torch.load(dA / rip_a[-1], weights_only=False)["stato"]["pv"]["bop"]
        sha_ok = sha_bop["sha"] == A.sha256_file(dA / sha_bop["file"])
        esiti["a_gemello"] = {"riprese": rip_a, "riprese_al_bit": uguali, "bop": bop_a, "bop_al_bit": bop_ok,
                              "sha_bop_nella_ripresa_giusto": sha_ok}
        # (b)
        stS, sS, pvS = corsa(ca_s, dS)
        _, sB1, _ = corsa(ca_a, dB, ferma_a=3)
        stB, sB, pvB = corsa(ca_a, dB, riprendi=True)
        ib, isn = pvB.stato.istantanea_bop(), pvS.stato.istantanea_bop()
        evS = [x["pv"] for x in sS["storia"] if x.get("pv")]
        evB = [x["pv"] for x in sB["storia"] if x.get("pv")]
        esiti["b_ripresa"] = {
            "theta_al_bit": all(torch.equal(a, b) for a, b in zip(stS.parametri(), stB.parametri())),
            "theta_asinc_intera_al_bit": all(torch.equal(a, b) for a, b in zip(stS.parametri(), stA.parametri())),
            "bop_al_bit": _uguali_profondi(ib, isn), "esiti_v_uguali": evS == evB,
            "contatori": all(sS["pv"][k] == sB["pv"][k] for k in ("registri", "salti_tenuti", "passi_v", "rho")),
            "bop_nello_stato_finale": sB["pv"].get("bop", {}).get("file"), "fermata": sB1["passo"]}
        # (c)
        orig_t = A._torch_save_atomico

        def rotto(o, p):
            if p.name.startswith("ripresa_"):
                raise OSError("disco finto rotto")
            return orig_t(o, p)
        A._torch_save_atomico = rotto
        try:
            dC = DIR / "a35_errore"
            if dC.exists():
                shutil.rmtree(dC)
            try:
                corsa(ca_a, dC)
                rilanciato = False
            except RuntimeError as e:
                rilanciato = "scrittura asincrona fallita" in str(e)
        finally:
            A._torch_save_atomico = orig_t
        esiti["c_errore_rilanciato"] = rilanciato
    finally:
        PV.installa, PV.PassoV, PV.famiglia_scrivibile = orig
        A.salva_ripresa = orig_salva
    e = esiti
    ok = (e["a_gemello"]["riprese_al_bit"] and e["a_gemello"]["bop_al_bit"] and e["a_gemello"]["sha_bop_nella_ripresa_giusto"]
          and all(v for k, v in e["b_ripresa"].items() if k not in ("bop_nello_stato_finale", "fermata"))
          and e["b_ripresa"]["fermata"] == 3 and e["b_ripresa"]["bop_nello_stato_finale"] == "bop_pv_000006.pt"
          and e["c_errore_rilanciato"])
    return registra("A35_scritture_asincrone_al_bit", ok, **esiti, secondi=round(time.time() - t0, 1))


PROVE = {"A35": prova_a35, "A34": prova_a34, "A33": prova_a33, "A32": prova_a32, "A31": prova_a31, "A30": prova_a30, "A29": prova_a29, "A28": prova_a28, "A22": prova_a22, "A21": prova_a21, "A20": prova_a20, "A19": prova_a19, "A18": prova_a18, "A17": prova_a17, "A16": prova_a16, "A15": prova_a15,
         "A14": prova_a14, "A13": prova_a13, "A1": prova_a1, "A2": prova_a2, "A3": prova_a3_a4, "A4": prova_a3_a4, "A5": prova_a5, "A6": prova_a6,
         "A7": prova_a7, "A8": prova_a8, "A9": prova_a9, "A10": prova_a10, "A11": prova_a11, "A12": prova_a12}


def main() -> int:
    DIR.mkdir(parents=True, exist_ok=True)
    scelte = [a.upper() for a in sys.argv[1:]] or ["A1", "A6", "A11", "A12", "A7", "A8", "A3", "A5", "A9", "A10", "A2"]
    fatte, ok = set(), True
    t0 = time.time()
    for k in scelte:
        fn = PROVE[k]
        if fn in fatte:
            continue
        fatte.add(fn)
        t1 = time.time()
        try:
            ok &= fn()
        except Exception as e:                       # una prova che solleva e' ROSSA, con la traccia
            ok = False
            registra(f"{k}_eccezione", False, errore=repr(e), traccia=traceback.format_exc()[-3000:])
        print(f"   ({k}: {time.time() - t1:.1f} s)", flush=True)
    ESITO["tutto_verde"] = ok
    ESITO["secondi"] = round(time.time() - t0, 1)
    suff = "_" + "_".join(scelte) if sys.argv[1:] else ""
    (DIR / f"esito_test_allena{suff}.json").write_text(json.dumps(ESITO, indent=1, default=str))
    print(f"ESITO: {'TUTTO VERDE' if ok else 'ROSSO'} ({ESITO['secondi']} s)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
