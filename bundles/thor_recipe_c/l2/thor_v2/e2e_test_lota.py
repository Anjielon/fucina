#!/usr/bin/env python3
"""Prove di `e2e_lota.py` e dell'innesto `--lota-*` in `e2e_allena.py` (27/9/2026, ramo lota-qaf). SOLO CPU (la GPU e'
di EB2/EB3). Famiglie SINTETICHE (gguf-py, niente /mnt) salvo T1b/T9, che leggono i byte VERI di Thor-T in sola lettura.

  T1  gradiente del gancio (forma fattorizzata, niente dW) == riferimento con G = d (.) dY^T X materializzato ==
      autograd sul peso continuo d (.) (T + B A) (il STE del paper), float64, su TQ1_0 sintetica con gruppi morti e
      (T1b) su sottoinsiemi VERI gate TQ1_0R_1024 e down TQ1_0; dL/dtheta col motore avvolto == motore di base AL BIT;
      raccolta spenta = nessun accumulo
  T2  t-SignSGD == regola di `t_signSGD.py` del paper (torch.quantile, where, add, clamp, round) AL BIT su A e B
      (esperti insieme), con zeri, pareggi e p diversi; quantile_lineare == torch.quantile AL BIT; schedula == quella
      del codice (300 passi); media del gradiente con lr < 1 == ricorrenza m <- (1 - lr) m + lr g
  T3  codici: dopo `sincronizza` byte == clamp(T0 + W^ vivo), trit in {-1,0,1}, byte canonici, campi d intatti, gruppi
      morti intatti, salti di due livelli (W^ da -1 a +1 su T0 = 0), B = 0 -> byte della base AL BIT, quattro decoder
      (compreso il codice C di llama.cpp) == trit x d, annullamento AL BIT, registro netto di UN livello che rigiocato
      sulla base da' i byte, export su un GGUF giocattolo con `PV.esporta_codici` (verifica al bit) e rilettura
  T4  giocattolo: maestro RAGGIUNGIBILE (codici = clamp(T0 + W^*) di un adattatore ternario nascosto) e maestro FUORI
      dalla famiglia di LoTA (1% di codici a caso): passi LoTA (gradiente dal gancio, t-SignSGD, sonda) -> la perdita
      held-out SCENDE; omega 0,75 r / 0,5 r / 0,25 r: passi prima del primo codice mosso, codici mossi, perdita finale
  T5  ripresa del modulo AL BIT: 3 passi + istantanea (A, B a 2 bit, u con lr < 1) + stato NUOVO + 3 passi == 6 filati
      (A, B, u, byte, moltiplicatore), con lr 1 e 0,5
  T6  ciclo `allena` (studente MoE finto a 2 strati con famiglie TQ1_0 vere, motore torch): LoTA SPENTA identica AL BIT
      al modulo di master (git ba9eeb3) — theta, Adam, stato (tolte le sole chiavi nuove, tutte ai default), file del
      migliore; LoTA ACCESA: held-out che scende, codici mossi, registri del migliore che rigiocati sulla base danno i
      byte, ripresa 3 + 3 == 6 AL BIT (theta, Adam, A, B, byte, stato), export del migliore su un GGUF giocattolo
  T7  registri PV di una fase precedente CONGELATI sotto la base (installa con registri, registri del modello = PV +
      netto, rigioco ed export al bit) e, nel ciclo, PV -> LoTA alla ripresa (stato["pv"] rigiocato)
  T8  configurazioni rifiutate (con --pv-codici, omega fuori campo, omega senza rango, spegnerla o cambiarne omega dopo
      l'allenamento), accensione alla ripresa (decisione, codici invariati), riga di comando -> ConfigAllena
  T9  (legge il GGUF vero) strato 0 di Thor-T, 256 esperti: tempo e memoria del gancio per famiglia e micro-lotto con
      256 token per esperto (quelli di un micro-lotto di EB2), `sincronizza` su famiglie intere, controlli; stima GPU

Uso (4 fili, nice):
    nice -n 19 env CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \\
        OPENBLAS_NUM_THREADS=4 @@ROOT:venv@@/bin/python3 e2e_test_lota.py [T1 T2 ...]
    esito: $E2E_LOTA_PROVE/esito_test_lota[_<prove>].json (default <albero>/prove_lota_out)
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import resource
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("TRITON_INTERPRET", "1")
QUI = Path(__file__).resolve().parent
DIR = Path(os.getenv("E2E_LOTA_PROVE", str(QUI.parent / "prove_lota_out")))
DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("E2E_PV_DIR", str(DIR / "pv"))
os.environ.setdefault("E2E_ALLENA_PROVE", str(DIR / "allena"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(QUI))
sys.path.insert(0, "@@ROOT:models@@/build/llamacpp-tq1-k1024/gguf-py")
import e2e_studente as S  # noqa: E402
import e2e_codici_scale as CS  # noqa: E402
import e2e_campi_scala as CAMPI  # noqa: E402
import e2e_pv_codici as PV  # noqa: E402
import e2e_lota as LT  # noqa: E402
import e2e_allena as A  # noqa: E402

FILI = int(os.getenv("OMP_NUM_THREADS", "4"))
torch.set_num_threads(FILI)
RIC = S.RICETTE["B32"]
BASE_MASTER = os.getenv("E2E_LOTA_BASE", "ba9eeb3")      # e2e_allena.py di master prima di LoTA
ESITO: dict = {"data": time.strftime("%Y-%m-%d %H:%M:%S"), "prove": {}}
muto = lambda *a, **k: None  # noqa: E731


def registra(nome: str, ok: bool, **dati) -> bool:
    ESITO["prove"][nome] = {"ok": bool(ok), **dati}
    print(f"[{nome}] {'✅' if ok else '⛔'} {json.dumps(dati, ensure_ascii=False, default=str)[:2500]}", flush=True)
    return ok


def rel(a, b) -> float:
    a, b = a.double(), b.double()
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-300))


# ─────────────────────────────── famiglie ───────────────────────────────
def famiglia_sintetica(E: int = 4, R: int = 64, C: int = 512, seme: int = 13, std: float = 1.0, nome: str = "gate",
                       strato: int = 0, morti: float = 0.0) -> S.DatiFamiglia:
    """Famiglia TQ1_0 SINTETICA (pesi gaussiani quantizzati da gguf-py, byte canonici come llama.cpp), scrivibile. `morti`:
    quota di gruppi con d = 0 scritta nei byte (d0 = 0, come le righe morte di Thor)."""
    from gguf import GGMLQuantizationType as GT, quants
    g = np.random.default_rng(seme)
    raw = quants.quantize((g.standard_normal((E * R, C)) * std).astype(np.float32), GT.TQ1_0).reshape(E, R, -1)
    raw = np.ascontiguousarray(raw)
    nb, _, off, _ = CS.LAYOUT["TQ1_0"]
    if morti > 0:
        blocchi = raw.reshape(-1, nb)
        scelti = g.random(blocchi.shape[0]) < morti
        blocchi[scelti, off:off + 2] = 0
    B = torch.from_numpy(raw.copy())
    d0 = CS.scale_da_blocchi(B.reshape(-1, nb), "TQ1_0").to(torch.float32).reshape(E, R, -1)
    f = S.DatiFamiglia(nome, "TQ1_0", B, d0, strato)
    return PV.famiglia_scrivibile(f)


def copia_fam(f: S.DatiFamiglia) -> S.DatiFamiglia:
    return PV.famiglia_scrivibile(S.DatiFamiglia(f.nome, f.tipo, f.blocchi.clone(), f.d0.clone(), f.strato))


_TENS = None


def fam_vera(L: int, n: str, esperti=None, righe=None) -> S.DatiFamiglia:
    global _TENS
    if _TENS is None:
        from gguf import GGUFReader
        _TENS = {t.name: t for t in GGUFReader(str(S.GGUF_THOR_T)).tensors}
    f = S.famiglia_da_tensore(_TENS[f"blk.{L}.{S.FAMIGLIE[n]}.weight"], n, esperti=esperti, righe=righe)
    if esperti is None and righe is None:
        f.blocchi = f.blocchi.clone()                       # il tensore intero e' una vista del memmap: copia
    return PV.famiglia_scrivibile(f)


def trit_fam(f) -> torch.Tensor:
    return CAMPI.trit_torch(f.blocchi.reshape(-1, f.nb), f.tipo).view(f.E, f.R, f.C)


def cfg_lota(r: int = 8, **kw) -> LT.ConfigLoTA:
    return LT.ConfigLoTA(rango=r, **kw)


def ad_casuale(fam, cfg, seme: int, dens_B: float = 0.6) -> LT.AdattatoreFamiglia:
    """Adattatore con A del paper e B ternario casuale (densita' dens_B): Delta W grandi, molti W^ non nulli."""
    ad = LT.AdattatoreFamiglia((fam.strato, fam.nome), fam, cfg)
    g = torch.Generator().manual_seed(seme)
    u = torch.rand(ad.B.shape, generator=g)
    s = torch.randint(0, 2, ad.B.shape, generator=g, dtype=torch.int8) * 2 - 1
    ad.B.copy_(torch.where(u < dens_B, s, torch.zeros_like(s)))
    return ad


def lotto(fam, T: int, k: int, seme: int, dtype=torch.float64):
    g = torch.Generator().manual_seed(seme)
    piano = S.Piano(torch.randint(0, fam.E, (T, k), generator=g), fam.E)
    x = torch.randn(piano.S, fam.C, generator=g, dtype=dtype)
    return x, piano


# ─────────────────────────────── T1 ───────────────────────────────
def _riferimenti(ad: LT.AdattatoreFamiglia, gy, x, piano, d):
    """(gB, gA) con G materializzato e con autograd sul peso continuo d (.) (T + B A) (y = x W^T, L = <gy, y>)."""
    fam = ad.fam
    T = trit_fam(fam).double()
    o = piano.offs.tolist()
    gB_m = torch.zeros(ad.B.shape, dtype=torch.float64)
    gA_m = torch.zeros(ad.A.shape, dtype=torch.float64)
    gB_a = torch.zeros_like(gB_m)
    gA_a = torch.zeros_like(gA_m)
    for e in range(fam.E):
        a, b = o[e], o[e + 1]
        if b <= a:
            continue
        D = d[e].double().repeat_interleave(fam.gs, dim=-1)          # (R, C)
        G = D * (gy[a:b].double().T @ x[a:b].double())
        gB_m[e] = G @ ad.A[e].double().T
        gA_m[e] = ad.B[e].double().T @ G
        Ac = ad.A[e].double().clone().requires_grad_(True)
        Bc = ad.B[e].double().clone().requires_grad_(True)
        W = D * (T[e] + Bc @ Ac)
        L = (gy[a:b].double() * (x[a:b].double() @ W.T)).sum()
        L.backward()
        gB_a[e], gA_a[e] = Bc.grad, Ac.grad
    return (gB_m, gA_m), (gB_a, gA_a)


def _t1_su(fam, nome: str, seme: int) -> dict:
    cfg = cfg_lota(8, dtype_u=torch.float64)
    ad = ad_casuale(fam, cfg, seme)
    ris = {}
    for k_mic in range(3):                                 # accumulo su 3 micro-lotti
        x, piano = lotto(fam, 24, 2, seme * 10 + k_mic)
        gy = torch.randn(piano.S, fam.R, generator=torch.Generator().manual_seed(seme * 77 + k_mic), dtype=torch.float64)
        d = fam.d0.double()
        ad.raccogli(gy, x, piano, d)
        (gBm, gAm), (gBa, gAa) = _riferimenti(ad, gy, x, piano, d)
        if k_mic == 0:
            acc = [gBm, gAm, gBa, gAa]
        else:
            acc = [a + b for a, b in zip(acc, (gBm, gAm, gBa, gAa))]
    ris["B_contro_materializzato"] = rel(ad.uB, acc[0])
    ris["A_contro_materializzato"] = rel(ad.uA, acc[1])
    ris["B_contro_autograd"] = rel(ad.uB, acc[2])
    ris["A_contro_autograd"] = rel(ad.uA, acc[3])
    vivi = fam.d0 != 0
    ris["gruppi_morti"] = int((~vivi).sum())
    return ris


def prova_t1() -> bool:
    t0 = time.time()
    esiti, ok = {}, True
    fam = famiglia_sintetica(E=4, R=48, C=512, seme=3, morti=0.1)
    esiti["sintetica_TQ1_0"] = _t1_su(fam, "sintetica", 5)
    try:
        g = fam_vera(39, "gate", esperti=slice(0, 3), righe=slice(0, 40))
        dn = fam_vera(39, "down", esperti=slice(0, 3), righe=slice(0, 40))
        esiti["vera_gate_TQ1_0R_1024"] = _t1_su(g, "gate", 7)
        esiti["vera_down_TQ1_0"] = _t1_su(dn, "down", 9)
    except (FileNotFoundError, OSError) as e:
        esiti["vere"] = f"saltate: {e}"
    for k, v in esiti.items():
        if isinstance(v, dict):
            ok &= all(v[x] <= 1e-12 for x in v if x.startswith(("A_", "B_")))
    # dL/dtheta col motore avvolto == motore di base AL BIT; raccolta spenta = uA, uB fermi
    fam2 = famiglia_sintetica(E=4, R=32, C=512, seme=21)
    cfg = cfg_lota(4)
    stato = LT.StatoLoTA({(0, "gate"): fam2}, cfg, {(0, "gate"): lambda: fam2.d0})
    mot_b = S.MotoreTorch()
    mot_l = LT.avvolgi_motore(mot_b, stato)
    x, piano = lotto(fam2, 20, 2, 99, torch.float32)
    th = torch.zeros_like(fam2.d0, requires_grad=True)
    gth = {}
    for nome, mot, att in (("base", mot_b, False), ("lota_spenta", mot_l, False), ("lota_accesa", mot_l, True)):
        th.grad = None
        stato.attivo = att
        d = S.ArrotondaFP16.apply(fam2.d0 * torch.exp(th), None)
        y = S.FamigliaEsperti.apply(x, d, fam2, piano, RIC, mot)
        (y.pow(2).sum() * 1e-3).backward()
        gth[nome] = th.grad.clone()
        if nome == "lota_spenta":
            spenta_ferma = bool((stato.ad[(0, "gate")].uB == 0).all())
    stato.attivo = False
    theta_bit = torch.equal(gth["base"], gth["lota_spenta"]) and torch.equal(gth["base"], gth["lota_accesa"])
    accesa_raccoglie = bool((stato.ad[(0, "gate")].uB != 0).any())
    sottoclasse = isinstance(mot_l, S.MotoreTorch) and mot_l.nome == "torch+LoTA"
    ok &= theta_bit and spenta_ferma and accesa_raccoglie and sottoclasse
    return registra("T1_gradiente_gancio", ok, errori=esiti, dtheta_al_bit=theta_bit, spenta_non_raccoglie=spenta_ferma,
                    accesa_raccoglie=accesa_raccoglie, motore_sottoclasse=sottoclasse, secondi=round(time.time() - t0, 1))


# ─────────────────────────────── T2 ───────────────────────────────
def _paper_passo(X: torch.Tensor, g: torch.Tensor, sigma: float) -> torch.Tensor:
    """Il corpo di `tSignSGD.step` (fase 1, upper False) su UN parametro, copiato da t_signSGD.py del paper."""
    p = X.to(torch.float32).clone()
    sigma_t = torch.quantile(g.abs(), q=sigma)
    p.add_(torch.where((g.abs() > sigma_t) & (g.abs() > 1e-9), -torch.sign(g), torch.zeros_like(g)))
    p.clamp_(min=-1, max=1).round_()
    return p.to(torch.int8)


def _paper_sigma(passo: int, tot: int, thr=0.95, ming=0.999, filt=0.9999) -> float:
    """sigma del codice del paper al passo `passo` (1-based, dopo `current_step += 1`)."""
    f1 = tot * 0.8
    if passo <= f1:
        return thr + (passo / f1) * (ming - thr)
    s = ming + ((passo - f1) / (tot - f1)) * (filt - ming)
    return filt if passo >= tot else s


def prova_t2() -> bool:
    t0 = time.time()
    ok, esiti = True, {}
    # quantile_lineare == torch.quantile al bit
    g = torch.Generator().manual_seed(2)
    q_ok = []
    for n in (1, 2, 7, 1000, 4097, 1 << 20):
        v = torch.randn(n, generator=g).abs()
        if n > 10:
            v[: n // 10] = 0.0                                   # zeri esatti (esperti senza token)
            v[n // 10: n // 5] = 0.5                             # pareggi
        for q in (0.0, 0.5, 0.95, 0.999, 0.9999, 1.0):
            q_ok.append(bool(torch.equal(LT.quantile_lineare(v, q), torch.quantile(v, q))))
    esiti["quantile_al_bit"] = all(q_ok)
    ok &= all(q_ok)
    # passo == paper su A e su B (esperti insieme = UN tensore)
    fam = famiglia_sintetica(E=4, R=64, C=512, seme=31)
    passi_ok = []
    for p in (0.05, 0.01, 0.001, 0.3):
        cfg = cfg_lota(8)
        ad = ad_casuale(fam, cfg, 17)
        gA = torch.randn(ad.uA.shape, generator=g)
        gB = torch.randn(ad.uB.shape, generator=g)
        gA[0] = 0.0                                              # esperto senza token
        gB[1, :5] = 3.0                                          # pareggi sopra la soglia
        ad.uA.copy_(gA)
        ad.uB.copy_(gB)
        A0, B0 = ad.A.clone(), ad.B.clone()
        m = ad.passo(p, 1e-9)
        attA = _paper_passo(A0, gA, 1.0 - p)
        attB = _paper_passo(B0, gB, 1.0 - p)
        uguale = torch.equal(ad.A, attA) and torch.equal(ad.B, attB)
        quota = m["B"][2] / ad.B.numel()
        passi_ok.append({"p": p, "al_bit": uguale, "quota_B_scelte": quota,
                         "cambiate_B": m["B"][3], "cambiate_A": m["A"][3]})
        ok &= uguale and abs(quota - p) <= max(0.1 * p, 2.0 / ad.B.numel()) + 20 / ad.B.numel()
        ad.annulla_passo(m)
        ok &= torch.equal(ad.A, A0) and torch.equal(ad.B, B0)
    esiti["passi"] = passi_ok
    # schedula == codice del paper (300 passi)
    cfg = cfg_lota(8)
    err = max(abs(LT.top_p_passo(cfg, k, 300) - (1.0 - _paper_sigma(k, 300))) for k in range(1, 301))
    esiti["schedula_err_max"] = err
    ok &= err < 1e-12 and LT.top_p_passo(cfg, 300, 300) == cfg.top_p_coda and LT.top_p_passo(cfg, 5, 0) == cfg.top_p
    # media del gradiente: lr 0,5, tre passi -> m = lr u == ricorrenza
    cfgm = cfg_lota(4, lr=0.5, dtype_u=torch.float64)
    fam3 = famiglia_sintetica(E=2, R=16, C=256 * 2, seme=5)
    st = LT.StatoLoTA({(0, "gate"): fam3}, cfgm, {(0, "gate"): lambda: fam3.d0})
    ad3 = st.ad[(0, "gate")]
    m_rif = torch.zeros_like(ad3.uB)
    for t in range(3):
        st.inizia_passo()
        gg = torch.randn(ad3.uB.shape, generator=g, dtype=torch.float64)
        ad3.uB += gg
        m_rif = (1 - 0.5) * m_rif + 0.5 * gg
    esiti["media_err"] = rel(0.5 * ad3.uB, m_rif)
    ok &= esiti["media_err"] < 1e-14
    return registra("T2_tsignsgd_come_il_paper", ok, **esiti, secondi=round(time.time() - t0, 1))


# ─────────────────────────────── T3 ───────────────────────────────
def _canonici(fam) -> int:
    off = CS.LAYOUT[fam.tipo][2]
    B = fam.blocchi.reshape(-1, fam.nb)[:, :off].to(torch.int32)
    qh = PV._in_qh(torch.arange(off), fam.tipo).expand_as(B)
    q = PV._valore_q(B, qh)
    return int(((((q * 256 + 242) // 243) & 255) != B).sum())


def _campi_d(fam) -> torch.Tensor:
    off, k = CS.LAYOUT[fam.tipo][2], CS.LAYOUT[fam.tipo][3]
    return fam.blocchi.reshape(-1, fam.nb)[:, off:off + 2 * k].clone()


def _gguf_giocattolo(p: Path, fams: dict) -> None:
    from gguf import GGUFWriter, GGMLQuantizationType as QT
    w = GGUFWriter(str(p), arch="qwen35moe")
    w.add_uint32("e2e.lota.prova", 1)
    for (L, n), f in sorted(fams.items()):
        w.add_tensor(f"blk.{L}.{S.FAMIGLIE[n]}.weight", f.blocchi.numpy().copy(), raw_dtype=QT[f.tipo])
    w.add_tensor("blk.0.attn_norm.weight", np.arange(512, dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def _quattro_decoder(fam, T_atteso: torch.Tensor) -> dict:
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


def _t3_su(fam, seme: int, cartella: Path) -> dict:
    cfg = cfg_lota(8, omega=3.0)
    B_base = fam.blocchi.clone()
    D_base = _campi_d(fam)
    T0 = trit_fam(fam)
    ad = ad_casuale(fam, cfg, seme, dens_B=0.7)
    ris = {}
    c1 = ad.sincronizza()
    T1 = trit_fam(fam)
    Wh = ad.w_hat(0, fam.E)
    atteso = (T0.to(torch.int16) + Wh.to(torch.int16)).clamp(-1, 1).to(torch.int8)
    vivo = (fam.d0 != 0).repeat_interleave(fam.gs, dim=-1)
    ris.update({"cambiati": c1["cambiati"], "w_hat_non_nulli": c1["w_hat_non_nulli"],
                "diversi_da_base": c1["diversi_da_base"],
                "codici_bersaglio": bool(torch.equal(T1, atteso)), "ternari": bool((T1.abs() <= 1).all()),
                "al_piu_un_livello_da_base": bool(((T1.to(torch.int16) - T0).abs() <= 1).all()),
                "gruppi_morti_intatti": bool(torch.equal(T1[~vivo], T0[~vivo])),
                "byte_non_canonici": _canonici(fam), "campi_d_intatti": bool(torch.equal(_campi_d(fam), D_base))})
    ad.controlla()
    ris["quattro_decoder"] = _quattro_decoder(fam, T1)
    # salti di DUE livelli: W^ = -1 su tutti i T0 = 0 vivi, poi +1
    zeri = (T0 == 0) & vivo
    with torch.no_grad():
        r = cfg.rango
        ad.A.fill_(1)
        ad.B.fill_(-1)
        c_meno = ad.sincronizza()
        meno_ok = bool((trit_fam(fam)[zeri] == -1).all())
        ad.B.fill_(1)
        c_piu = ad.sincronizza()
        piu_ok = bool((trit_fam(fam)[zeri] == 1).all())
    ris.update({"due_livelli_cambiati": c_piu["cambiati"], "due_livelli_ok": meno_ok and piu_ok and c_meno["cambiati"] > 0,
                "byte_non_canonici_dopo_due_livelli": _canonici(fam)})
    ad.controlla()
    # B = 0 -> base AL BIT
    ad.B.zero_()
    ad.sincronizza()
    ris["b_zero_base_al_bit"] = bool(torch.equal(fam.blocchi, B_base))
    # annullamento AL BIT
    ad2 = ad_casuale(fam, cfg, seme + 1, dens_B=0.7)
    ad2.base = ad.base
    ad2.sincronizza()
    Bp, Ap, Bb = ad2.B.clone(), ad2.A.clone(), fam.blocchi.clone()
    g = torch.Generator().manual_seed(seme)
    ad2.uA.copy_(torch.randn(ad2.uA.shape, generator=g))
    ad2.uB.copy_(torch.randn(ad2.uB.shape, generator=g))
    mosse = ad2.passo(0.2, 0.0)
    cm = ad2.sincronizza()
    ad2.annulla_passo(mosse)
    ad2.sincronizza()
    ris["annullamento"] = {"codici_cambiati_dal_passo": cm["cambiati"],
                           "al_bit": bool(torch.equal(fam.blocchi, Bb) and torch.equal(ad2.A, Ap) and torch.equal(ad2.B, Bp))}
    # registro netto: un livello, rigiocato sulla base == byte; export su GGUF giocattolo
    net = ad2.registro_netto()
    un_liv = bool((np.abs(net.righe["t_nuovo"].astype(int) - net.righe["t_vecchio"].astype(int)) == 1).all())
    base_copia = PV.famiglia_scrivibile(S.DatiFamiglia(fam.nome, fam.tipo, ad2.base.clone(), fam.d0, fam.strato))
    PV.rigioca({(fam.strato, fam.nome): base_copia}, [net])
    ris["registro_netto"] = {"salti": len(net), "un_livello": un_liv,
                             "rigiocato_sulla_base_al_bit": bool(torch.equal(base_copia.blocchi, fam.blocchi))}
    src = cartella / f"t3_{fam.nome}_{fam.tipo}.gguf"
    dst = cartella / f"t3_{fam.nome}_{fam.tipo}_lota.gguf"
    _gguf_giocattolo(src, {(fam.strato, fam.nome): base_copia.__class__(fam.nome, fam.tipo, ad2.base.clone(), fam.d0,
                                                                         fam.strato)})
    rep = PV.esporta_codici(src, dst, [net])
    from gguf import GGUFReader
    t = {x.name: x for x in GGUFReader(str(dst)).tensors}[f"blk.{fam.strato}.{S.FAMIGLIE[fam.nome]}.weight"]
    ris["export"] = {"trit_cambiati": rep["trit_cambiati"], "byte_diversi": rep["byte_diversi"],
                     "file_uguale_ai_byte": bool(np.array_equal(np.asarray(t.data), fam.blocchi.numpy()))}
    return ris


def prova_t3() -> bool:
    t0 = time.time()
    cart = DIR / "t3"
    cart.mkdir(parents=True, exist_ok=True)
    esiti = {"sintetica_TQ1_0": _t3_su(famiglia_sintetica(E=4, R=32, C=512, seme=41, morti=0.1), 43, cart)}
    try:
        esiti["vera_gate_TQ1_0R_1024"] = _t3_su(fam_vera(20, "gate", esperti=slice(0, 3), righe=slice(0, 24)), 47, cart)
    except (FileNotFoundError, OSError) as e:
        esiti["vera"] = f"saltata: {e}"
    ok = True
    for v in esiti.values():
        if not isinstance(v, dict):
            continue
        ok &= (v["codici_bersaglio"] and v["ternari"] and v["al_piu_un_livello_da_base"] and v["gruppi_morti_intatti"]
               and v["byte_non_canonici"] == 0 and v["campi_d_intatti"] and all(v["quattro_decoder"].values())
               and v["due_livelli_ok"] and v["byte_non_canonici_dopo_due_livelli"] == 0 and v["b_zero_base_al_bit"]
               and v["annullamento"]["al_bit"] and v["annullamento"]["codici_cambiati_dal_passo"] > 0
               and v["registro_netto"]["un_livello"] and v["registro_netto"]["rigiocato_sulla_base_al_bit"]
               and v["export"]["file_uguale_ai_byte"] and v["cambiati"] > 0)
    return registra("T3_codici_nel_formato", ok, **esiti, secondi=round(time.time() - t0, 1))


# ─────────────────────────────── T4 ───────────────────────────────
class Dati:
    """Micro-lotti deterministici (instradamento casuale, x correlata x = z M) per UNA famiglia."""

    def __init__(self, fam, token: int, k: int, seme: int, corr: float = 0.3):
        g = torch.Generator().manual_seed(seme)
        self.fam, self.token, self.k, self.seme = fam, token, k, seme
        self.M = torch.eye(fam.C) + corr * torch.randn(fam.C, fam.C, generator=g) / math.sqrt(fam.C)

    def lotto(self, passo: int, kk: int, token=None):
        g = torch.Generator().manual_seed(self.seme * 100003 + passo * 101 + kk)
        T = token or self.token
        piano = S.Piano(torch.randint(0, self.fam.E, (T, self.k), generator=g), self.fam.E)
        return torch.randn(piano.S, self.fam.C, generator=g) @ self.M, piano


def _y_maestro(mae, x, piano):
    """Uscita del maestro: una DatiFamiglia (codici sulla griglia) oppure pesi CONTINUI (E, R, C) (come il Q8 davanti al
    ternario: l'errore dello studente e' errore di quantizzazione)."""
    with torch.no_grad():
        if isinstance(mae, torch.Tensor):
            out = torch.empty(x.shape[0], mae.shape[1])
            o = piano.offs.tolist()
            for e in range(mae.shape[0]):
                if o[e + 1] > o[e]:
                    out[o[e]:o[e + 1]] = x[o[e]:o[e + 1]] @ mae[e].T
            return out
        return S.MotoreTorch().avanti(x, mae.d0, mae, piano, RIC)


def perdita_mse(fam_s, mae, x, piano, mot, n_tot, d=None):
    y_t = _y_maestro(mae, x, piano)
    if d is None:                                        # d con gradiente: senza, il dd (e il gancio) non gira
        d = fam_s.d0.detach().clone().requires_grad_(True)
    y = S.FamigliaEsperti.apply(x, d, fam_s, piano, RIC, mot)
    return ((y - y_t) ** 2).sum() / n_tot


def fam_da_pesi(W: np.ndarray, nome: str = "gate", strato: int = 0) -> S.DatiFamiglia:
    """Famiglia TQ1_0 = quantizzazione di gguf-py (come llama.cpp) dei pesi W (E, R, C)."""
    from gguf import GGMLQuantizationType as GT, quants
    E, R, C = W.shape
    raw = np.ascontiguousarray(quants.quantize(W.reshape(E * R, C).astype(np.float32), GT.TQ1_0).reshape(E, R, -1))
    B = torch.from_numpy(raw.copy())
    nb = CS.LAYOUT["TQ1_0"][0]
    d0 = CS.scale_da_blocchi(B.reshape(-1, nb), "TQ1_0").to(torch.float32).reshape(E, R, -1)
    return PV.famiglia_scrivibile(S.DatiFamiglia(nome, "TQ1_0", B, d0, strato))


def maestro_raggiungibile(fs0, r: int, omega: float, seme: int, dens_B: float = 0.5):
    """Maestro = codici clamp(T0 + W^*) di un adattatore ternario NASCOSTO (A* del paper, B* casuale): LoTA lo puo'
    raggiungere esattamente (con rango 2r e omega 2 omega*: B = [B*, B*], A = [A*; A*])."""
    ft = copia_fam(fs0)
    ad = ad_casuale(ft, LT.ConfigLoTA(rango=r, omega=omega, seme=seme), seme, dens_B)
    ad.sincronizza()
    return ft, ad


def _corsa_giocattolo(fs0, mae, passi: int, cfg: LT.ConfigLoTA, sonda: bool = True, token: int = 32, seme: int = 5) -> dict:
    """Passi LoTA soli (scale ferme) su UNA famiglia: gradiente dal gancio del dd (8 micro-lotti), t-SignSGD, codici, sonda
    (128 token mai visti) come in `allena`; held-out su 512 token. Conta i codici mossi e quanti coincidono col maestro."""
    fs = copia_fam(fs0)
    k = (fs.strato, fs.nome)
    st = LT.StatoLoTA({k: fs}, cfg, {k: lambda: fs.d0})
    mot = LT.avvolgi_motore(S.MotoreTorch(), st)
    dati = Dati(fs, token, 2, seme)
    n_micro = 8
    n_tot = n_micro * token * 2
    xh, ph = dati.lotto(10 ** 6, 0, token=512)
    held = lambda: float(perdita_mse(fs, mae, xh, ph, S.MotoreTorch(), ph.S).detach())  # noqa: E731
    curva = [held()]
    primo, tenuti, annullati, mosse_cod, ritorni = None, 0, 0, [], 0
    T0 = trit_fam(fs0)
    xf, pf = dati.lotto(10 ** 6 + 7, 0, token=128)              # sonda FISSA (solo con la finestra)
    for t in range(passi):
        st.inizia_passo()
        with st.raccolta():
            for kk in range(n_micro):
                x, piano = dati.lotto(t, kk)
                perdita_mse(fs, mae, x, piano, mot, n_tot).backward()
        mosse = st.passo(st.p_passo(t + 1, passi), 1.0)
        n = st.sincronizza(applica=False)["cambiati"]
        if n and primo is None:
            primo = t + 1
        if cfg.finestra > 0 and sonda:                        # sonda a finestra: si tiene sempre, si torna al migliore
            if n:
                st.sincronizza()
            v = float(perdita_mse(fs, mae, xf, pf, S.MotoreTorch(), pf.S).detach())
            if st.esito_finestra(v, t + 1)["azione"] == "ritorno":
                ritorni += 1
        elif n and sonda:
            xs, ps = dati.lotto(t + 1, 999, token=128)
            prima = float(perdita_mse(fs, mae, xs, ps, S.MotoreTorch(), ps.S).detach())
            st.sincronizza()
            dopo = float(perdita_mse(fs, mae, xs, ps, S.MotoreTorch(), ps.S).detach())
            if st.esito(prima, dopo, n):
                tenuti += 1
            else:
                st.annulla(mosse)
                annullati += 1
        elif n:
            st.sincronizza()
        mosse_cod.append(n)
        curva.append(held())
    Tf = trit_fam(fs)
    out = {"curva": [round(c, 5) for c in curva], "iniziale": curva[0], "finale": curva[-1], "minimo": min(curva),
           "rapporto": curva[-1] / curva[0], "primo_passo_con_codici": primo, "tenuti": tenuti, "annullati": annullati,
           "codici_mossi_per_passo": mosse_cod, "diversi_da_base_finale": int((Tf != T0).sum()),
           "ternari": bool((Tf.abs() <= 1).all()), "mult_finale": st.mult, "ritorni_al_migliore": ritorni}
    if not isinstance(mae, torch.Tensor):
        Tt = trit_fam(mae)
        mossi = Tf != T0
        out["codici_giusti_su_mossi"] = float(((Tf == Tt) & mossi).sum() / max(int(mossi.sum()), 1))
        # a caso: una mossa di un livello da T0 e' giusta se il maestro si e' mosso li' (verso obbligato da +-1, 1/2 da 0)
        cambia = (Tt != T0).double()
        out["giusti_per_caso"] = float((cambia * torch.where(T0 == 0, 0.5, 1.0).double()).mean())
    return out


def prova_t4() -> bool:
    """Giocattolo (una famiglia, 4 esperti 64 x 512, scale ferme), la DINAMICA di LoTA. (a) maestro RAGGIUNGIBILE (11,7 % dei
    codici mossi da un adattatore nascosto), init dal gradiente, omega 2 (0,25 r): senza sonda, sonda stretta, sonda con
    tolleranza 0,1, sonda a finestra 12; init del paper senza sonda. (b) maestro a pesi CONTINUI (lo studente e' la sua
    quantizzazione TQ1_0: nessuna mossa aiuta davvero): senza sonda, sonda stretta, sonda a finestra 12."""
    t0 = time.time()
    passi = int(os.getenv("E2E_LOTA_T4_PASSI", "40"))
    fs0 = famiglia_sintetica(E=4, R=64, C=512, seme=61, std=1.0 / math.sqrt(512))
    ft, _ = maestro_raggiungibile(fs0, r=4, omega=1.0, seme=62)
    esiti = {"passi": passi, "raggiungibile_codici_mossi_dal_maestro": int((trit_fam(ft) != trit_fam(fs0)).sum()),
             "codici": fs0.E * fs0.R * fs0.C}
    kw = dict(rango=8, omega=2.0, top_p=0.2, top_p_fine=0.02, top_p_coda=0.005)
    esiti["raggiungibile_gradiente_senza_sonda"] = _corsa_giocattolo(fs0, ft, passi, LT.ConfigLoTA(**kw, init="gradiente"),
                                                                     sonda=False)
    esiti["raggiungibile_gradiente_sonda_stretta"] = _corsa_giocattolo(fs0, ft, passi, LT.ConfigLoTA(**kw, init="gradiente"))
    esiti["raggiungibile_gradiente_sonda_tolleranza_0_1"] = _corsa_giocattolo(
        fs0, ft, passi, LT.ConfigLoTA(**kw, init="gradiente", tolleranza=0.1))
    esiti["raggiungibile_paper_senza_sonda"] = _corsa_giocattolo(fs0, ft, passi, LT.ConfigLoTA(**kw), sonda=False)
    esiti["raggiungibile_gradiente_sonda_finestra_12"] = _corsa_giocattolo(
        fs0, ft, passi, LT.ConfigLoTA(**kw, init="gradiente", finestra=12))
    g = np.random.default_rng(81)
    W = (g.standard_normal((4, 64, 512)) / math.sqrt(512)).astype(np.float32)
    fq = fam_da_pesi(W)
    Wt = torch.from_numpy(W)
    kwf = dict(rango=8, omega=2.0, top_p=0.05, init="gradiente")
    esiti["continuo_senza_sonda"] = _corsa_giocattolo(fq, Wt, passi, LT.ConfigLoTA(**kwf), sonda=False)
    esiti["continuo_sonda_stretta"] = _corsa_giocattolo(fq, Wt, passi, LT.ConfigLoTA(**kwf), sonda=True)
    esiti["continuo_sonda_finestra_12"] = _corsa_giocattolo(fq, Wt, passi, LT.ConfigLoTA(**kwf, finestra=12), sonda=True)
    for v in esiti.values():                                      # le curve intere restano nell'esito, qui le ogni-5
        if isinstance(v, dict):
            v["curva_ogni_5"] = v["curva"][::5]
    ok = all(v["ternari"] for v in esiti.values() if isinstance(v, dict))
    senza = esiti["raggiungibile_gradiente_senza_sonda"]
    stretta = esiti["raggiungibile_gradiente_sonda_stretta"]
    toll = esiti["raggiungibile_gradiente_sonda_tolleranza_0_1"]
    # la leva sul giocattolo: senza sonda LoTA (init dal gradiente) si muove in massa e verso il maestro
    ok &= senza["rapporto"] < 0.8 and senza["codici_giusti_su_mossi"] > 5 * senza["giusti_per_caso"]
    # la sonda stretta non peggiora ma tronca; la tolleranza sta in mezzo
    ok &= stretta["finale"] <= stretta["iniziale"] and toll["finale"] <= stretta["finale"]
    ok &= stretta["codici_giusti_su_mossi"] > 2 * stretta["giusti_per_caso"]
    # dove nessuna mossa aiuta la sonda impedisce il danno
    ok &= esiti["continuo_sonda_stretta"]["finale"] <= esiti["continuo_senza_sonda"]["finale"]
    ok &= esiti["continuo_sonda_stretta"]["finale"] <= esiti["continuo_sonda_stretta"]["iniziale"] * 1.001
    # la sonda a finestra fissa: sul raggiungibile lascia passare la valle (come senza sonda), sul continuo torna al migliore
    fin = esiti["raggiungibile_gradiente_sonda_finestra_12"]
    ok &= fin["rapporto"] < 0.8 and fin["ritorni_al_migliore"] == 0
    ok &= esiti["continuo_sonda_finestra_12"]["finale"] <= esiti["continuo_sonda_finestra_12"]["iniziale"] * 1.001
    return registra("T4_giocattolo_dinamica", ok, **esiti, secondi=round(time.time() - t0, 1))


# ─────────────────────────────── T5 ───────────────────────────────
def prova_t5() -> bool:
    t0 = time.time()
    esiti, ok = {}, True
    fs0 = famiglia_sintetica(E=4, R=32, C=512, seme=71, std=1.0 / math.sqrt(512))
    ft, _ = maestro_raggiungibile(fs0, r=4, omega=1.0, seme=72)
    for lr, init in ((1.0, "paper"), (0.5, "paper"), (1.0, "gradiente")):
        cfg = LT.ConfigLoTA(rango=8, omega=2.0, top_p=0.2, lr=lr, init=init)
        dati = Dati(fs0, 16, 2, 73)

        def nuovo():
            fs = copia_fam(fs0)
            st = LT.StatoLoTA({(0, "gate"): fs}, cfg, {(0, "gate"): lambda: fs.d0})
            return fs, st, LT.avvolgi_motore(S.MotoreTorch(), st)

        def passi(fs, st, mot, da, a):
            for t in range(da, a):
                st.inizia_passo()
                with st.raccolta():
                    for kk in range(4):
                        x, piano = dati.lotto(t, kk)
                        perdita_mse(fs, ft, x, piano, mot, 4 * 32).backward()
                st.passo(st.p_passo(t + 1, 6), 1.0)
                st.sincronizza()
                st.passi = t + 1
                if t == 2:
                    st.mult = 0.5                                # stato non banale del moltiplicatore
        fa, sa, ma = nuovo()
        passi(fa, sa, ma, 0, 6)
        fb, sb, mb = nuovo()
        passi(fb, sb, mb, 0, 3)
        p = DIR / f"t5_lota_lr{lr}_{init}.pt"
        torch.save(sb.istantanea(), p)
        fc, sc, mc = nuovo()
        rip = sc.ripristina(torch.load(p, map_location="cpu", weights_only=True))
        byte_dopo_ripresa = torch.equal(fc.blocchi, fb.blocchi)
        passi(fc, sc, mc, 3, 6)
        ada, adc = sa.ad[(0, "gate")], sc.ad[(0, "gate")]
        e = {"byte_alla_ripresa": byte_dopo_ripresa, "ricostruiti": rip["diversi_da_base"],
             "A": torch.equal(ada.A, adc.A), "B": torch.equal(ada.B, adc.B), "byte": torch.equal(fa.blocchi, fc.blocchi),
             "u": torch.equal(ada.uA, adc.uA) and torch.equal(ada.uB, adc.uB), "mult": sa.mult == sc.mult,
             "passi": sa.passi == sc.passi, "diversi_da_base": int((trit_fam(fa) != trit_fam(fs0)).sum()),
             "byte_file": p.stat().st_size, "inizializzato": sa.inizializzato == sc.inizializzato is True}
        esiti[f"lr_{lr}_{init}"] = e
        ok &= all(v for k2, v in e.items() if isinstance(v, bool)) and e["diversi_da_base"] > 0
    return registra("T5_ripresa_modulo_al_bit", ok, **esiti, secondi=round(time.time() - t0, 1))


# ─────────────────────────────── studente MoE finto (T6-T8) ───────────────────────────────
_FIRMA_FINTA = DIR / "studente_moe_finto.firma"
if not _FIRMA_FINTA.exists():
    _FIRMA_FINTA.write_text("studente MoE finto di e2e_test_lota.py (file fisso: firma dello studente nelle riprese)\n")


class StudenteMoEFinto:
    """Due strati MoE con famiglie TQ1_0 VERE (sintetiche, 4 esperti x 512 x 512, top-2), stessa interfaccia che `allena`
    usa: h_{l+1} = h_l + sum_j w_j down_e(silu(gate_e h) * up_e h), esperti con `S.FamigliaEsperti` + `S.StratoEsperti`
    (theta, scala, motore) -> il `dd` dello studente e il gancio LoTA sono quelli veri. Router fisso (instradamento vivo,
    deterministico: il ricalcolo dell'indietro ritrova le stesse scelte)."""
    FAM = ("gate", "up", "down")

    def __init__(self, V: int = 256, H: int = 512, E: int = 4, k: int = 2, strati=(0, 1), seme: int = 3, fams=None):
        g = torch.Generator().manual_seed(seme)
        self.V, self.H, self.E, self.k = V, H, E, k
        self.emb = torch.randn(V, H, generator=g)
        self.router = {L: torch.randn(E, H, generator=g) * (2.0 / math.sqrt(H)) for L in strati}
        self.strati = list(strati)
        self.device = torch.device("cpu")
        self.gguf = str(_FIRMA_FINTA)                     # file FISSO: la firma (byte, mtime) non cambia se si edita la prova
        self.ric = RIC
        self.mot = S.MotoreTorch()
        self.es = {}
        for L in strati:
            if fams is not None:
                fam = {n: copia_fam(fams[(L, n)]) for n in self.FAM}
            else:
                fam = {n: famiglia_sintetica(E=E, R=H, C=H, seme=seme * 100 + 10 * L + i, std=1.0 / math.sqrt(H), nome=n,
                                             strato=L) for i, n in enumerate(self.FAM)}
            self.es[L] = S.StratoEsperti(fam, RIC, self.mot)

    def famiglie(self) -> dict:
        return {(L, n): self.es[L].fam[n] for L in self.strati for n in self.FAM}

    def chiavi(self):
        return [(L, n) for L in self.strati for n in self.FAM]

    def parametri(self):
        return [self.es[L].theta[n] for (L, n) in self.chiavi()]

    def esperti(self, L):
        return self.es[L]

    def _strato(self, L, h):
        es = self.es[L]
        prob = torch.softmax(h @ self.router[L].T, -1)
        idx = torch.topk(prob, self.k, -1).indices
        w = prob.gather(-1, idx)
        w = (w / w.sum(-1, keepdim=True)).detach()
        piano = S.Piano(idx, self.E)
        x_ord = S._Raccogli.apply(h, piano.ordine, piano.inversa, self.k)
        gg = S.FamigliaEsperti.apply(x_ord, es.scala("gate"), es.fam["gate"], piano, RIC, es.mot)
        uu = S.FamigliaEsperti.apply(x_ord, es.scala("up"), es.fam["up"], piano, RIC, es.mot)
        a = torch.nn.functional.silu(gg) * uu
        o = S.FamigliaEsperti.apply(a, es.scala("down"), es.fam["down"], piano, RIC, es.mot)
        o = o * w.reshape(-1)[piano.ordine].unsqueeze(-1)
        return h + S._Disperdi.apply(o, piano.ordine, piano.inversa, self.k)

    def _f(self, x):
        h = x
        for L in self.strati:
            h = self._strato(L, h)
        return h

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
        return {f"blk.{L}.{S.FAMIGLIE[n]}": self.es[L].scala(n).detach().to(torch.float16).numpy() for (L, n) in self.chiavi()}


def sorgente_moe(fams_maestro: dict, V: int = 256, T: int = 32, n_train: int = 28, n_held: int = 4, seme: int = 5):
    """Maestro = lo studente MoE finto coi codici `fams_maestro` (bf16 come la cache vera); domini held-out a, b, a, wiki."""
    st = StudenteMoEFinto(V, fams=fams_maestro)
    g = torch.Generator().manual_seed(seme)
    doms = ["a", "b", "a", "wiki"]
    righe = {}
    for lato, n in (("train", n_train), ("heldout", n_held)):
        rr = []
        for i in range(n):
            ids = torch.randint(0, V, (T,), generator=g).numpy().astype(np.int32)
            h = st.nascosto(torch.from_numpy(ids.astype(np.int64))[None]).to(torch.bfloat16).clone()
            rr.append((ids, h, doms[i % 4] if lato == "heldout" else "tr", None))
        righe[lato] = rr
    WS = torch.randn(V, st.H, generator=g) * (3.0 / math.sqrt(st.H))
    return A.SorgenteMemoria(righe, {}), WS


def fams_maestro_moe(seme_st: int = 3, r: int = 4, omega: float = 1.0) -> dict:
    """Codici del maestro: quelli dello studente MoE finto + un adattatore ternario nascosto per famiglia."""
    base = StudenteMoEFinto(seme=seme_st)
    out = {}
    for (L, n), f in base.famiglie().items():
        ft = copia_fam(f)
        ad = ad_casuale(ft, LT.ConfigLoTA(rango=r, omega=omega, seme=900 + L), 900 + 10 * L + len(n), 0.5)
        ad.sincronizza()
        out[(L, n)] = ft
    return out


def _ca_moe(**kw):
    base = dict(lr=3e-2, righe_per_passo=4, righe_per_micro=1, righe_per_micro_valuta=2, heldout_ogni=2,
                ripresa_ogni=2, pezzo_kl=16, attesa_ogni_s=0.01)
    base.update(kw)
    return A.ConfigAllena(**base)


def _modulo_master(base: str = BASE_MASTER):
    import importlib.util
    testo = subprocess.run(["git", "-C", str(QUI), "show", f"{base}:thor_v2/e2e_allena.py"], check=True,
                           capture_output=True, text=True).stdout
    nome = f"e2e_allena_{base}"
    p = DIR / f"{nome}.py"
    p.write_text(testo)
    spec = importlib.util.spec_from_file_location(nome, p)
    m = importlib.util.module_from_spec(spec)
    sys.modules[nome] = m
    spec.loader.exec_module(m)
    return m


def _config_per(mod, ca):
    noti = {f.name for f in dataclasses.fields(mod.ConfigAllena)}
    return mod.ConfigAllena(**{k: v for k, v in dataclasses.asdict(ca).items() if k in noti})


def _senza_tempi(stato: dict) -> dict:
    s = json.loads(json.dumps(stato, default=str))
    for k in ("attesa_teacher_s", "tempi_ultimi"):           # tempi a parete (come `_senza_tempi` di e2e_test_allena)
        s.pop(k, None)
    if s.get("g1"):
        s["g1"].pop("secondi", None)
    for x in s.get("storia", []):
        x.pop("rumore", None)
        if "lota" in x:
            x["lota"].pop("secondi", None)
    (s.get("g1") or {}).pop("rumore", None)
    return s


def _sha_cartella(d: Path) -> dict:
    import hashlib
    return {str(p.relative_to(d)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(d.rglob("*")) if p.is_file()}


_LOTA_DEFAULT = {"lota_rango": 0, "lota_omega": None, "lota_top_p": 0.05, "lota_top_p_fine": 1e-3, "lota_top_p_coda": 1e-4,
                 "lota_lr": 1.0, "lota_tau": 1e-9, "lota_sonda": True, "lota_init": "paper", "lota_sonda_tolleranza": 0.0,
                 "lota_sonda_finestra": 0}


def _lota_e_integrata_default() -> dict:
    """Ramo integrata: contro ba9eeb3 sono nuovi anche i campi degli altri rami uniti, tutti ai loro default."""
    import e2e_test_allena as TA
    return {**_LOTA_DEFAULT, **TA._CFG_INTEGRATA_DEFAULT}


def prova_t6() -> bool:
    t0 = time.time()
    esiti, ok = {}, True
    fm = fams_maestro_moe()
    src, WS = sorgente_moe(fm)
    # (a) LoTA spenta == master AL BIT
    vec = _modulo_master()
    ca = _ca_moe()
    es = {}
    for nome, mod in (("nuovo", A), ("master", vec)):
        d = DIR / f"t6_spenta_{nome}"
        if d.exists():
            shutil.rmtree(d)
        st = StudenteMoEFinto()
        stato = mod.allena(st, src, d, ca if mod is A else _config_per(mod, ca), WS, WS, log=muto)
        es[nome] = (st, stato, mod.carica_ripresa(d, "cpu"), d)
    (sn, zn, rn, dn), (sp, zp, rp, dp) = es["nuovo"], es["master"]
    th = all(torch.equal(a, b) for a, b in zip(sn.parametri(), sp.parametri()))
    so, sr = rn["opt"]["state"], rp["opt"]["state"]
    adam = all(torch.equal(so[i][k], sr[i][k]) for i in so for k in ("exp_avg", "exp_avg_sq", "step"))
    a_, b_ = _senza_tempi(zn), _senza_tempi(zp)
    nuovi = {k: a_["config"].pop(k) for k in set(a_["config"]) - set(b_["config"])}
    stati = a_ == b_ and set(zn) == set(zp)
    if not stati:
        esiti["spenta_diff_chiavi"] = [k for k in set(a_) | set(b_) if a_.get(k) != b_.get(k)]
    fn, fp_ = _sha_cartella(dn), _sha_cartella(dp)
    ist = {k: v for k, v in fn.items() if k.startswith("migliore_") and not k.endswith("meta.json")}
    file_ok = set(fn) == set(fp_) and all(fn[k] == fp_[k] for k in ist) and len(ist) > 0
    byte = all(torch.equal(a.blocchi, b.blocchi) for a, b in zip(sn.famiglie().values(), sp.famiglie().values()))
    esiti["spenta_come_master"] = {"theta": th, "adam": adam, "stato": stati, "file_migliore": file_ok, "byte": byte,
                                   "chiavi_nuove_ai_default": nuovi == _lota_e_integrata_default(), "base": BASE_MASTER}
    ok &= th and adam and stati and file_ok and byte and nuovi == _lota_e_integrata_default()
    # (b) LoTA accesa: 6 passi filati
    cal = dict(lota_rango=8, lota_omega=2.0, lota_top_p=0.2, lota_top_p_fine=0.05, lota_top_p_coda=0.02,
               ripresa_ogni=1, tieni_riprese=10, heldout_ogni=1)
    d6 = DIR / "t6_accesa_6"
    if d6.exists():
        shutil.rmtree(d6)
    st6 = StudenteMoEFinto()
    base6 = {k: f.blocchi.clone() for k, f in st6.famiglie().items()}
    z6 = A.allena(st6, src, d6, _ca_moe(**cal), WS, WS, ferma_a=6, log=muto)
    passi_l = [v.get("lota") for v in z6["storia"] if v.get("lota")]
    held = [v["metrica"] for v in z6["storia_heldout"]]
    cod = sum(int((trit_fam(f) != CAMPI.trit_torch(base6[k].reshape(-1, f.nb), f.tipo).view(f.E, f.R, f.C)).sum())
              for k, f in st6.famiglie().items())
    mig = z6["migliore"]
    if not mig.get("registri_sha"):
        return registra("T6_ciclo_allena", False, **esiti, errore=f"migliore al passo {mig['passo']} senza registri")
    regs = PV.Registro.carica_lista(d6 / mig["dir"] / "registri_pv.npz", mig["registri_sha"])
    esiti["accesa"] = {"passi_lota": len(passi_l), "held_out": held, "codici_diversi_da_base": cod,
                       "tenuti": z6["lota"]["tenuti"], "annullati": z6["lota"]["annullati"],
                       "codici_mossi_per_passo": [v["codici_cambiati"] for v in passi_l],
                       "migliore": mig["passo"], "salti_del_migliore": sum(len(r) for r in regs)}
    ok &= len(passi_l) == 5 and min(held[1:]) < held[0] and cod > 0
    # stessa corsa con LoTA spenta (solo scale): per attribuire la discesa (registrato, non un cancello)
    doff = DIR / "t6_spenta_6"
    if doff.exists():
        shutil.rmtree(doff)
    zoff = A.allena(StudenteMoEFinto(), src, doff, _ca_moe(ripresa_ogni=1, tieni_riprese=10, heldout_ogni=1), WS, WS,
                    ferma_a=6, log=muto)
    esiti["accesa"]["held_out_con_lota_spenta"] = [v["metrica"] for v in zoff["storia_heldout"]]
    # registri del migliore rigiocati sulla base == byte dello studente al passo del migliore: si rifa' la corsa fino a li'
    stm = StudenteMoEFinto()
    dm = DIR / "t6_accesa_fino_al_migliore"
    if dm.exists():
        shutil.rmtree(dm)
    A.allena(stm, src, dm, _ca_moe(**cal), WS, WS, ferma_a=mig["passo"], log=muto)
    rig = {k: copia_fam(StudenteMoEFinto().famiglie()[k]) for k in stm.famiglie()}
    PV.rigioca(rig, regs)
    esiti["registri_migliore_rigiocati_al_bit"] = all(torch.equal(rig[k].blocchi, f.blocchi) for k, f in stm.famiglie().items())
    ok &= esiti["registri_migliore_rigiocati_al_bit"]
    # (c) ripresa 3 + 3 == 6 AL BIT
    d3 = DIR / "t6_accesa_3p3"
    if d3.exists():
        shutil.rmtree(d3)
    A.allena(StudenteMoEFinto(), src, d3, _ca_moe(**cal), WS, WS, ferma_a=3, log=muto)
    st3 = StudenteMoEFinto()
    z3 = A.allena(st3, src, d3, _ca_moe(**cal), WS, WS, riprendi=True, ferma_a=6, log=muto)
    r6, r3 = A.carica_ripresa(d6, "cpu"), A.carica_ripresa(d3, "cpu")
    th_ok = all(torch.equal(a, b) for a, b in zip(st6.parametri(), st3.parametri()))
    ad_ok = all(torch.equal(st6.esperti(L).mot._lota.ad[(L, n)].A, st3.esperti(L).mot._lota.ad[(L, n)].A) and
                torch.equal(st6.esperti(L).mot._lota.ad[(L, n)].B, st3.esperti(L).mot._lota.ad[(L, n)].B)
                for (L, n) in st6.chiavi())
    by_ok = all(torch.equal(a.blocchi, b.blocchi) for a, b in zip(st6.famiglie().values(), st3.famiglie().values()))
    so, sr = r6["opt"]["state"], r3["opt"]["state"]
    adam_ok = all(torch.equal(so[i][k], sr[i][k]) for i in so for k in ("exp_avg", "exp_avg_sq", "step"))
    a6, a3 = _senza_tempi(z6), _senza_tempi(z3)
    stato_ok = a6 == a3
    if not stato_ok:
        diff = [k for k in set(a6) | set(a3) if a6.get(k) != a3.get(k)]
        esiti["ripresa_diff_chiavi"] = diff
    esiti["ripresa_3p3"] = {"theta": th_ok, "A_B": ad_ok, "byte": by_ok, "adam": adam_ok, "stato": stato_ok}
    ok &= th_ok and ad_ok and by_ok and adam_ok and stato_ok
    # (d) export del migliore su un GGUF giocattolo coi byte della base
    src_g = DIR / "t6_base.gguf"
    dst_g = DIR / "t6_lota.gguf"
    _gguf_giocattolo(src_g, StudenteMoEFinto().famiglie())
    rep = PV.esporta_codici(src_g, dst_g, regs)
    from gguf import GGUFReader
    tt = {x.name: x for x in GGUFReader(str(dst_g)).tensors}
    exp_ok = all(np.array_equal(np.asarray(tt[f"blk.{L}.{S.FAMIGLIE[n]}.weight"].data), f.blocchi.numpy())
                 for (L, n), f in stm.famiglie().items())
    esiti["export_migliore"] = {"trit_cambiati": rep["trit_cambiati"], "salti": rep["salti"], "file_uguale_ai_byte": exp_ok}
    ok &= exp_ok
    return registra("T6_ciclo_allena", ok, **esiti, secondi=round(time.time() - t0, 1))


# ─────────────────────────────── T7 ───────────────────────────────
def prova_t7() -> bool:
    t0 = time.time()
    esiti, ok = {}, True
    st = StudenteMoEFinto()
    base0 = {k: f.blocchi.clone() for k, f in st.famiglie().items()}
    # un registro PV «di prima»: 1% dei codici di ogni famiglia, un livello, righe canoniche del registro
    parti = []
    g = torch.Generator().manual_seed(7)
    for (L, n), f in st.famiglie().items():
        T = trit_fam(f)
        pos = torch.randperm(f.E * f.R * f.C, generator=g)[: f.E * f.R * f.C // 100]
        e, r, c = pos // (f.R * f.C), (pos // f.C) % f.R, pos % f.C
        tv = T[e, r, c].to(torch.int64)
        tn = torch.where(tv == 0, torch.ones_like(tv), torch.zeros_like(tv))
        righe = np.zeros(pos.numel(), dtype=PV.REG_DTYPE)
        righe["L"], righe["f"] = L, PV.FAMIGLIE_IDX[n]
        righe["e"], righe["r"], righe["c"] = e.numpy(), r.numpy(), c.numpy()
        righe["t_vecchio"], righe["t_nuovo"] = tv.numpy(), tn.numpy()
        parti.append(righe)
    reg_pv = PV.Registro(np.concatenate(parti))
    lota = LT.installa(st, LT.ConfigLoTA(rango=8, omega=2.0), registri_congelati=[reg_pv])
    for ad in lota.ad.values():
        g2 = torch.Generator().manual_seed(11)
        u = torch.rand(ad.B.shape, generator=g2)
        ad.B.copy_(torch.where(u < 0.6, (torch.randint(0, 2, ad.B.shape, generator=g2) * 2 - 1).to(torch.int8),
                               torch.zeros_like(ad.B)))
    c = lota.sincronizza()
    regs = [reg_pv] + lota.registri_netti()
    rig = {k: PV.famiglia_scrivibile(S.DatiFamiglia(n, "TQ1_0", base0[k].clone(), st.famiglie()[k].d0, k[0]))
           for k in st.famiglie() for n in [k[1]]}
    PV.rigioca(rig, regs)
    rig_ok = all(torch.equal(rig[k].blocchi, f.blocchi) for k, f in st.famiglie().items())
    base_ok = True
    rig_pv = {k: PV.famiglia_scrivibile(S.DatiFamiglia(k[1], "TQ1_0", base0[k].clone(), st.famiglie()[k].d0, k[0]))
              for k in st.famiglie()}
    PV.rigioca(rig_pv, [reg_pv])
    base_ok = all(torch.equal(rig_pv[k].blocchi, lota.ad[k].base) for k in st.famiglie())
    src_g, dst_g = DIR / "t7_base.gguf", DIR / "t7_pv_lota.gguf"
    _gguf_giocattolo(src_g, {k: PV.famiglia_scrivibile(S.DatiFamiglia(k[1], "TQ1_0", base0[k].clone(),
                                                                       st.famiglie()[k].d0, k[0])) for k in base0})
    rep = PV.esporta_codici(src_g, dst_g, regs)
    esiti["unita"] = {"codici_lota": c["diversi_da_base"], "salti_pv": len(reg_pv), "base_e_gguf_piu_pv": base_ok,
                      "pv_piu_netto_rigiocati_al_bit": rig_ok, "export_trit_cambiati": rep["trit_cambiati"]}
    ok &= rig_ok and base_ok and c["diversi_da_base"] > 0
    # nel ciclo: corsa PV (finta: stato["pv"] scritto nella ripresa come lo scrive la fase 2) -> LoTA alla ripresa
    fm = fams_maestro_moe()
    src, WS = sorgente_moe(fm)
    d = DIR / "t7_pv_poi_lota"
    if d.exists():
        shutil.rmtree(d)
    A.allena(StudenteMoEFinto(), src, d, _ca_moe(), WS, WS, ferma_a=2, log=muto)
    rip = A.carica_ripresa(d, "cpu")
    nome = f"registri_pv_{rip['stato']['passo']:06d}.npz"
    sha = PV.Registro.salva_lista(d / nome, [reg_pv])
    rip["stato"]["pv"] = {"registri": 1, "file": nome, "sha": sha, "rho": 0.01, "tarato": True, "salti_tenuti": len(reg_pv),
                          "passi_v": 2}
    rip["stato"]["config"]["pv_codici"] = True
    man = json.loads((d / "ripresa_ultima.json").read_text())
    shaR = A._torch_save_atomico(rip, d / man["file"])
    man["sha256"] = shaR
    A._scrivi_json(d / "ripresa_ultima.json", man)
    righe_log = []
    stL = StudenteMoEFinto()
    zL = A.allena(stL, src, d, _ca_moe(lota_rango=8, lota_omega=2.0, lota_top_p=0.2, heldout_ogni=1), WS, WS,
                  riprendi=True, ferma_a=4, log=righe_log.append)
    lotaL = stL.mot._lota
    pv_nella_base = all(torch.equal(rig_pv[k].blocchi, lotaL.ad[k].base) for k in stL.famiglie())
    mig = zL["migliore"]
    regs_m = PV.Registro.carica_lista(d / mig["dir"] / "registri_pv.npz", mig["registri_sha"]) if mig.get("registri_sha") \
        else None
    esiti["ciclo"] = {"pv_rigiocato_nella_base": pv_nella_base, "passi_lota": zL["lota"]["passi"],
                      "log_congelati": any("congelati" in x for x in righe_log),
                      "migliore": mig["passo"], "registri_migliore": None if regs_m is None else len(regs_m),
                      "primo_registro_migliore_e_pv": regs_m is not None and np.array_equal(regs_m[0].righe, reg_pv.righe)}
    ok &= pv_nella_base and esiti["ciclo"]["log_congelati"] and zL["lota"]["passi"] == 2
    if mig["passo"] > 2:
        ok &= esiti["ciclo"]["primo_registro_migliore_e_pv"]
    return registra("T7_pv_congelato_sotto_lota", ok, **esiti, secondi=round(time.time() - t0, 1))


# ─────────────────────────────── T8 ───────────────────────────────
def prova_t8() -> bool:
    t0 = time.time()
    esiti, ok = {}, True
    rifiuti = {}
    for nome, kw in (("con_pv", dict(lota_rango=8, pv_codici=True)), ("omega_ge_rango", dict(lota_rango=8, lota_omega=8.0)),
                     ("omega_senza_rango", dict(lota_omega=2.0)), ("top_p_disordinati", dict(lota_rango=8, lota_top_p=1e-4)),
                     ("lr_fuori", dict(lota_rango=8, lota_lr=0.0)), ("rango_negativo", dict(lota_rango=-1))):
        try:
            _ca_moe(**kw)
            rifiuti[nome] = False
        except ValueError:
            rifiuti[nome] = True
    esiti["rifiutate"] = rifiuti
    ok &= all(rifiuti.values())
    fm = fams_maestro_moe()
    src, WS = sorgente_moe(fm)
    cal = dict(lota_rango=8, lota_omega=2.0, lota_top_p=0.2, heldout_ogni=1)
    d = DIR / "t8_ripresa"
    if d.exists():
        shutil.rmtree(d)
    A.allena(StudenteMoEFinto(), src, d, _ca_moe(**cal), WS, WS, ferma_a=3, log=muto)
    pr = {}
    for nome, kw in (("spenta", dict()), ("omega_cambiato", dict(cal, lota_omega=3.0))):
        try:
            A.allena(StudenteMoEFinto(), src, d, _ca_moe(**kw), WS, WS, riprendi=True, ferma_a=4, log=muto)
            pr[nome] = False
        except RuntimeError:
            pr[nome] = True
    esiti["ripresa_rifiutata"] = pr
    ok &= all(pr.values())
    # top-p cambiato alla ripresa: decisione ammessa
    righe = []
    A.allena(StudenteMoEFinto(), src, d, _ca_moe(**dict(cal, lota_top_p=0.1)), WS, WS, riprendi=True, ferma_a=4,
             log=righe.append)
    esiti["top_p_cambiato_ammesso"] = any("lota_top_p" in x for x in righe)
    ok &= esiti["top_p_cambiato_ammesso"]
    # accesa alla ripresa: codici invariati al momento dell'accensione
    d2 = DIR / "t8_accesa_alla_ripresa"
    if d2.exists():
        shutil.rmtree(d2)
    A.allena(StudenteMoEFinto(), src, d2, _ca_moe(), WS, WS, ferma_a=2, log=muto)
    st2 = StudenteMoEFinto()
    b0 = {k: f.blocchi.clone() for k, f in st2.famiglie().items()}
    righe2 = []
    z2 = A.allena(st2, src, d2, _ca_moe(**cal), WS, WS, riprendi=True, ferma_a=2, log=righe2.append)
    esiti["accesa_alla_ripresa"] = {"log": any("LoTA accesa al passo 2" in x for x in righe2),
                                    "codici_invariati": all(torch.equal(b0[k], f.blocchi) for k, f in st2.famiglie().items()),
                                    "inizio": z2["lota"]["inizio"]}
    ok &= esiti["accesa_alla_ripresa"]["log"] and esiti["accesa_alla_ripresa"]["codici_invariati"] and z2["lota"]["inizio"] == 2
    # riga di comando
    import argparse
    ns = argparse.Namespace(lota_rango=16, lota_omega=None, lota_top_p=0.05, lota_top_p_fine=1e-3, lota_top_p_coda=1e-4,
                            lota_lr=0.5, lota_tau=1e-9, lota_senza_sonda=True, lota_init="gradiente")
    campi = A._lota_da_arg(ns)
    esiti["cli"] = campi
    ok &= campi["lota_rango"] == 16 and campi["lota_lr"] == 0.5 and campi["lota_sonda"] is False \
        and campi["lota_init"] == "gradiente" and A.config_lota(A.ConfigAllena(**campi)).init == "gradiente"
    ok &= A._lota_da_arg(argparse.Namespace(lota_rango=0)) == {}
    return registra("T8_configurazioni_e_ripresa", ok, **esiti, secondi=round(time.time() - t0, 1))


# ─────────────────────────────── T9 ───────────────────────────────
def _vmhwm_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def prova_t9() -> bool:
    """Strato 0 VERO (256 esperti) sulla CPU: gancio (256 token per esperto = un micro-lotto di EB2: 2 x 4096 token x
    top-8 / 256), sincronizza e controlli su famiglie intere. Memoria degli adattatori == formula. Tempi CPU [M] e stima
    GPU [S] dai FLOP."""
    t0 = time.time()
    esiti, ok = {}, True
    r = int(os.getenv("E2E_LOTA_T9_RANGO", "16"))
    cfg = LT.ConfigLoTA(rango=r)
    for n in ("gate", "down"):
        fam = fam_vera(0, n)
        base_prima = fam.blocchi.clone()
        m0 = _vmhwm_mb()
        tc = time.time()
        ad = LT.AdattatoreFamiglia((0, n), fam, cfg)
        t_crea = time.time() - tc
        per_e = 256
        idx = torch.arange(fam.E).repeat_interleave(per_e).view(-1, 1)
        piano = S.Piano(idx, fam.E)
        g = torch.Generator().manual_seed(9)
        x = torch.randn(piano.S, fam.C, generator=g)
        gy = torch.randn(piano.S, fam.R, generator=g) * 1e-3
        u = torch.rand(ad.B.shape, generator=g)                  # B non nullo: il ramo dL/dA lavora davvero
        ad.B.copy_(torch.where(u < 0.3, torch.ones_like(ad.B), torch.zeros_like(ad.B)))
        tg = time.time()
        ad.raccogli(gy, x, piano, fam.d0)
        t_g = time.time() - tg
        flop = LT.flop_gradiente(fam.R, fam.C, fam.NG, r) * piano.S
        ad.B.zero_()
        ts = time.time()
        c0 = ad.sincronizza(applica=False)
        t_s0 = time.time() - ts
        # B casuale denso (omega 0,75 r): quanti codici si muovono e costo di sincronizza con scritture vere
        u = torch.rand(ad.B.shape, generator=g)
        s = (torch.randint(0, 2, ad.B.shape, generator=g) * 2 - 1).to(torch.int8)
        ad.B.copy_(torch.where(u < 0.9, s, torch.zeros_like(s)))
        ts = time.time()
        c1 = ad.sincronizza()
        t_s1 = time.time() - ts
        tk = time.time()
        ad.controlla()
        t_k = time.time() - tk
        ad.B.zero_()
        ad.sincronizza()
        base_bit = torch.equal(fam.blocchi, base_prima)
        by = ad.byte()
        atteso_ab = fam.E * r * (fam.R + fam.C)
        esiti[n] = {"tipo": fam.tipo, "forma": [fam.E, fam.R, fam.C], "NG": fam.NG, "crea_s": round(t_crea, 2),
                    "gancio_s_cpu": round(t_g, 2), "gancio_gflop": round(flop / 1e9, 2),
                    "gancio_gflops_cpu": round(flop / 1e9 / t_g, 1), "sincronizza_b0_s": round(t_s0, 2),
                    "sincronizza_scritture_s": round(t_s1, 2), "controlla_s": round(t_k, 2),
                    "codici_mossi_b_denso": c1["cambiati"], "quota_mossi": c1["cambiati"] / (fam.E * fam.R * fam.C),
                    "b_zero_torna_base_al_bit": base_bit, "sincronizza_b0_cambiati": c0["cambiati"],
                    "byte": by, "AB_come_formula": by["AB"] == atteso_ab, "vmhwm_mb_delta": round(_vmhwm_mb() - m0, 1)}
        ok &= base_bit and by["AB"] == atteso_ab and c0["cambiati"] == 0 and c1["cambiati"] > 0
        del ad, fam, x, gy, base_prima
    # stima GPU [S]: FLOP del gancio per passo di EB2 (131.072 token x top-8, 40 strati) a 1,34 TFLOPS fp32 (forma vera
    # misurata, TECNICHE §15) e a 0,3 TFLOPS (GEMM piccole per esperto)
    slot = 131072 * 8
    tf = sum(LT.flop_gradiente(R, C, NG, r) * slot for (R, C, NG) in ((512, 2048, 16), (512, 2048, 16), (2048, 512, 2)))
    esiti["stima_gpu"] = {"flop_gancio_per_passo": tf * 40, "secondi_a_1_34_tflops": round(tf * 40 / 1.34e12, 1),
                          "secondi_a_0_3_tflops": round(tf * 40 / 0.3e12, 1), "rango": r}
    return registra("T9_strato_vero_tempi_memoria", ok, **esiti, secondi=round(time.time() - t0, 1))



# ─────────────────────────────── T10 ───────────────────────────────
def prova_t10() -> bool:
    """Sonda a FINESTRA (--lota-sonda-finestra): (a) logica pura su una famiglia sintetica: il migliore si fotografa, dopo N
    passi senza batterlo A e B (e i byte) tornano AL BIT alla foto e p si dimezza; tetto con tolleranza; (b) nel ciclo
    (studente MoE finto): esiti «finestra» nel registro, ripresa 3 + 3 == 6 AL BIT con la foto del migliore nello stato."""
    t0 = time.time()
    esiti, ok = {}, True
    fam = famiglia_sintetica(E=4, R=32, C=512, seme=101)
    st = LT.StatoLoTA({(0, "gate"): fam}, LT.ConfigLoTA(rango=8, omega=2.0, finestra=2), {(0, "gate"): lambda: fam.d0})
    ad = st.ad[(0, "gate")]
    g = torch.Generator().manual_seed(3)

    def mossa():
        ad.uA.copy_(torch.randn(ad.uA.shape, generator=g))
        ad.uB.copy_(torch.randn(ad.uB.shape, generator=g))
        st.passo(0.3, 0.0)
        st.sincronizza()

    mossa()
    e1 = st.esito_finestra(5.0, 1)
    A1, B1, by1 = ad.A.clone(), ad.B.clone(), fam.blocchi.clone()
    mossa()
    e2 = st.esito_finestra(6.0, 2)
    mossa()
    cambiato = not torch.equal(fam.blocchi, by1)
    e3 = st.esito_finestra(7.0, 3)
    tornato = torch.equal(ad.A, A1) and torch.equal(ad.B, B1) and torch.equal(fam.blocchi, by1)
    ad.controlla()
    esiti["logica"] = {"azioni": [e1["azione"], e2["azione"], e3["azione"]], "mult": st.mult, "byte_cambiati_prima": cambiato,
                       "tornato_al_bit": tornato}
    ok &= [e1["azione"], e2["azione"], e3["azione"]] == ["migliore", "attesa", "ritorno"] and st.mult == 0.5 and tornato
    st2 = LT.StatoLoTA({(0, "gate"): copia_fam(fam)}, LT.ConfigLoTA(rango=8, omega=2.0, finestra=5, tolleranza=0.1),
                       {(0, "gate"): lambda: fam.d0})
    a1 = st2.esito_finestra(5.0, 1)["azione"]
    a2 = st2.esito_finestra(5.4, 2)["azione"]
    a3 = st2.esito_finestra(5.6, 3)
    esiti["tetto"] = [a1, a2, a3["azione"], a3.get("motivo")]
    ok &= esiti["tetto"] == ["migliore", "attesa", "ritorno", "tetto"]
    # (b) ciclo
    fm = fams_maestro_moe()
    src, WS = sorgente_moe(fm)
    cal = dict(lota_rango=8, lota_omega=2.0, lota_top_p=0.2, lota_sonda_finestra=2, lota_init="gradiente",
               ripresa_ogni=1, tieni_riprese=10, heldout_ogni=1)
    d6, d3 = DIR / "t10_6", DIR / "t10_3p3"
    for d in (d6, d3):
        if d.exists():
            shutil.rmtree(d)
    st6 = StudenteMoEFinto()
    z6 = A.allena(st6, src, d6, _ca_moe(**cal), WS, WS, ferma_a=6, log=muto)
    A.allena(StudenteMoEFinto(), src, d3, _ca_moe(**cal), WS, WS, ferma_a=3, log=muto)
    st3 = StudenteMoEFinto()
    z3 = A.allena(st3, src, d3, _ca_moe(**cal), WS, WS, riprendi=True, ferma_a=6, log=muto)
    ev = [v["lota"] for v in z6["storia"] if v.get("lota")]
    l6, l3 = st6.mot._lota, st3.mot._lota
    ad_ok = all(torch.equal(l6.ad[k].A, l3.ad[k].A) and torch.equal(l6.ad[k].B, l3.ad[k].B) for k in l6.ad)
    by_ok = all(torch.equal(a.blocchi, b.blocchi) for a, b in zip(st6.famiglie().values(), st3.famiglie().values()))
    th_ok = all(torch.equal(a, b) for a, b in zip(st6.parametri(), st3.parametri()))
    mig_ok = (l6.migliore is None) == (l3.migliore is None) and (l6.migliore is None or (
        l6.migliore["valore"] == l3.migliore["valore"] and all(np.array_equal(l6.migliore["A"][k], l3.migliore["A"][k])
                                                               for k in l6.ad)))
    a6, a3 = _senza_tempi(z6), _senza_tempi(z3)
    esiti["ciclo"] = {"azioni": [e.get("finestra", {}).get("azione") for e in ev], "init": [e["init"] for e in ev],
                      "codici_cambiati": [e["codici_cambiati"] for e in ev], "held_out": [v["metrica"] for v in z6["storia_heldout"]],
                      "ripresa": {"theta": th_ok, "A_B": ad_ok, "byte": by_ok, "migliore_della_sonda": mig_ok,
                                  "stato": a6 == a3}}
    ok &= all(e.get("finestra") is not None for e in ev if not e["init"]) and th_ok and ad_ok and by_ok and mig_ok and a6 == a3
    return registra("T10_sonda_a_finestra", ok, **esiti, secondi=round(time.time() - t0, 1))


PROVE = {"T1": prova_t1, "T2": prova_t2, "T3": prova_t3, "T4": prova_t4, "T5": prova_t5, "T6": prova_t6, "T7": prova_t7,
         "T8": prova_t8, "T9": prova_t9, "T10": prova_t10}


def main() -> int:
    scelte = [a for a in sys.argv[1:] if a in PROVE] or list(PROVE)
    ok = True
    for nome in scelte:
        try:
            ok &= PROVE[nome]()
        except Exception as e:                                   # noqa: BLE001 — la prova rossa si registra, non si tace
            traceback.print_exc()
            ok &= registra(nome, False, eccezione=f"{type(e).__name__}: {e}")
    ESITO["tutte_verdi"] = ok
    suff = "" if scelte == list(PROVE) else "_" + "_".join(scelte)
    (DIR / f"esito_test_lota{suff}.json").write_text(json.dumps(ESITO, indent=1, default=str))
    print("TUTTO VERDE" if ok else "ROSSO", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
