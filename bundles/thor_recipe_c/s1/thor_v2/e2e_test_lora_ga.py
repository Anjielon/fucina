#!/usr/bin/env python3
"""Prove di `--lora-init ga` (27/9/2026, ramo lora-ga): inizializzazione LoRA-GA (arXiv 2407.05000) degli adattatori
sui k-quant NON esperti. SOLO CPU, nessun file su /mnt: lo stesso ambiente sintetico di `e2e_test_lora.py` (config
Qwen3.5-MoE ridotta, 2 strati, k-quant veri Q8_0/Q4_0/Q4_K/Q6_K nelle famiglie LoRA).

Uso:
    nice -n 15 env CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=4 TRITON_INTERPRET=1 \\
        E2E_ALLENA_PROVE=/tmp/lora-ga-prove @@ROOT:venv@@/bin/python3 e2e_test_lora_ga.py [G1 ... G6]
    esito: $E2E_ALLENA_PROVE/lora_ga/esito_test_lora_ga[_<prove>].json

  G1 passo 0 AL BIT (ricette A32 e A bf16): dopo `inizializza_ga` avanti, gradiente di theta, valore e instradamento
     == studente SENZA LoRA; A == A0 e B == B0 al bit, dL/dA e dL/dB finiti e non nulli; il gancio di cattura dW ==
     gradiente autograd del peso (A64, <= 1e-12); A0 A0^T = c^2 I, B0^T B0 = c^2 I; Teorema 3.1: s^2 (dW A0^T A0 +
     B0 B0^T dW) == s^2 c^2 * (troncata di rango 2r di dW) (A64, <= 1e-10); export griglia del passo 0 == sorgente al byte
  G2 forma GA lontano dal passo 0 (A != A0, B != B0, A64): uscita == strato denso con W + s (B A - B0 A0) (<= 1e-12),
     dL/dA e dL/dB == differenze finite centrate (err. rel. <= 1e-5), delta_file == s (B A - B0 A0) nell'ordine del file
  G3 discesa a LOTTO PIENO: maestro = studente con un adattatore di rango 4 a seme DIVERSO (direzione che nessuna init
     conosce), theta* = 0; 8 righe x 32 token ripetute (ogni passo le vede tutte), held-out = le stesse righe; lr theta
     1e-6, 8 passi, zero contro ga a lr_lora 1e-4 e 3e-4: con ga la KL e' piu' bassa a OGNI passo; controllo: ga a
     1e-4 batte zero a 1e-3 (non e' solo un lr effettivo piu' alto)
  G4 ripresa: 3 passi + ripresa da disco (studente NUOVO) + 3 == 6 filati AL BIT con ga (theta, A, B, A0, B0, Adam,
     stato con "lora_ga", lora.npz); LoRA-GA accesa alla ripresa (passo 2): theta dopo il passo 3 == corsa SENZA LoRA
     al bit (il modello non cambia), Adam LoRA al passo 1; init cambiata su LoRA gia' allenata = errore (zero->ga,
     ga->zero); ConfigAllena e CLI (--lora-init, --lora-ga-gamma, --lora-ga-righe)
  G5 export dopo allenamento con ga: byte dei tensori LoRA == riquantizza(W + s (B A - B0 A0)), ricarica al bit,
     delta dall'istantanea == delta dello studente
  G6 `svd_alti`: iterazione di sottospazio (matrice 1536 x 768 > 2^20) contro la SVD esatta: valori singolari e
     sottospazio dei primi 2r (errore <= 1e-6), deterministica al bit con lo stesso seme
"""
from __future__ import annotations

import functools
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
os.environ.setdefault("E2E_ALLENA_PROVE", "/tmp/lora-ga-prove")
import e2e_test_lora as TL  # noqa: E402  (GGUF sintetico con k-quant veri, studente, _passo, _b_casuale)
import e2e_test_f32 as TF  # noqa: E402
from e2e_test_f32 import A, S, X, GGUFReader  # noqa: E402
import e2e_lora as LR  # noqa: E402

DIR = Path(os.environ["E2E_ALLENA_PROVE"]) / "lora_ga"
torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "4")))
ESITO: dict = {"data": time.strftime("%Y-%m-%d %H:%M:%S"), "prove": {}}
muto = TL.muto
TUTTE = TL.TUTTE


def registra(nome: str, ok: bool, **dati) -> bool:
    ESITO["prove"][nome] = {"ok": bool(ok), **dati}
    print(f"[{nome}] {'✅' if ok else '❌'} {json.dumps(dati, default=str)[:2400]}", flush=True)
    return bool(ok)


def _perdita(R):
    return lambda h: (h.to(R.dtype) * R).sum()


def _inizializza(st, ids, x, R, gamma: float = 16.0) -> dict:
    """LoRA-GA con la perdita sintetica sum(h * R) su (ids, x): lo stesso `inizializza_ga` che usa e2e_allena."""
    def calcola():
        for p in st.parametri() + st.parametri_lora():
            p.grad = None
        st.gradiente(ids, _perdita(R), x0=x)
    info = LR.inizializza_ga(st, calcola, gamma, muto)
    for p in st.parametri() + st.parametri_lora():
        p.grad = None
    st.azzera_cambi()
    return info


def _dw_autograd(st, ids, x, R) -> dict:
    """dW di riferimento INDIPENDENTE: gradiente autograd del peso base di ogni LinearLoRA (requires_grad solo qui),
    e in parallelo il gancio di cattura (accumulatori a mano): {nome: (autograd, gancio)}."""
    lins = {}
    for L in st.strati:
        for m in st.moduli[L].modules():
            if isinstance(m, LR.LinearLoRA):
                lins[m.ad.nome] = m
    for m in lins.values():
        m.base.weight.requires_grad_(True)
        m.base.weight.grad = None
        m.ad.cattura = torch.zeros(m.ad.forma, dtype=torch.float64)
    try:
        st.gradiente(ids, _perdita(R), x0=x)
        out = {n: (m.base.weight.grad.detach().clone(), m.ad.cattura.clone()) for n, m in lins.items()}
    finally:
        for m in lins.values():
            m.base.weight.requires_grad_(False)
            m.base.weight.grad = None
            m.ad.cattura = None
    return out


# ─────────────────────────────── G1 ───────────────────────────────
def prova_g1() -> bool:
    t0 = time.time()
    det, ok = {}, True
    for ric in ("A32", "A"):
        sn, sg = TL.studente(ric), TL.studente(ric, TL.cfg_lora(4))
        for st in (sn, sg):
            TF._theta_casuale(st)
        ids_c, x_c, R_c = TF._ingresso(sg, seme=11)                 # lotto di calibrazione (diverso da quello di prova)
        info = _inizializza(sg, ids_c, x_c, R_c)
        ids, x, R = TF._ingresso(sn)
        yn, gn, vn, in_ = TL._passo(sn, ids, x, R)
        yg, gg, vg, ig = TL._passo(sg, ids, x, R)
        uguali = lambda a, b: all(torch.equal(u, v) for u, v in zip(a, b))  # noqa: E731
        d = {"uscita_theta_valore_instr_al_bit": torch.equal(yn, yg) and uguali(gn, gg) and vn == vg and uguali(in_, ig),
             "A_uguale_A0_B_uguale_B0": all(torch.equal(a.A, a.A0) and torch.equal(a.B, a.B0) for a in sg.lora.values()),
             "B_non_nullo": all(float(a.B.abs().max()) > 0 for a in sg.lora.values()),
             "dA_dB_finiti_non_nulli": all(bool(torch.isfinite(p.grad).all()) and float(p.grad.abs().max()) > 0
                                           for p in sg.parametri_lora()),
             "cambi": sg.controlla()["cambi_instradamento"], "adattatori": len(sg.lora),
             "energia_2r": info["quota_energia_2r"]}
        det[ric] = d
        ok &= d["uscita_theta_valore_instr_al_bit"] and d["A_uguale_A0_B_uguale_B0"] and d["B_non_nullo"] \
            and d["dA_dB_finiti_non_nulli"] and d["cambi"] == 0 and d["adattatori"] == len(TL.TIPO_SINTETICO)
    # A64: gancio == autograd; ortonormalita' e Teorema 3.1 contro il dW di riferimento
    s64 = TL.studente("A64", TL.cfg_lora(4, alpha=8.0))
    TF._theta_casuale(s64)
    with TF.tutto_fp64():
        ids_c, x_c, R_c = TF._ingresso(s64, seme=11)
        rif = _dw_autograd(s64, ids_c, x_c, R_c)
        _inizializza(s64, ids_c, x_c, R_c, gamma=16.0)
    err_gancio = max(float((g - a).abs().max() / a.abs().max()) for a, g in rif.values())
    s = s64.lora_cfg.scala
    err_orto = err_t31 = 0.0
    for n, a in s64.lora.items():
        G = rif[n][0].double()
        r = a.rango
        c = LR.scala_ga(a.forma[0], 16.0)
        I = torch.eye(r, dtype=torch.float64)
        err_orto = max(err_orto, float((a.A0 @ a.A0.T - c * c * I).abs().max()) / c ** 2,
                       float((a.B0.T @ a.B0 - c * c * I).abs().max()) / c ** 2)
        U, Sv, Vh = torch.linalg.svd(G, full_matrices=False)
        tronca = (U[:, :2 * r] * Sv[:2 * r]) @ Vh[:2 * r]
        primo = s * s * (G @ a.A0.T @ a.A0 + a.B0 @ a.B0.T @ G)        # -lr * questo = primo passo SGD di s B A
        err_t31 = max(err_t31, float((primo - s * s * c * c * tronca).abs().max() / (s * s * c * c * tronca).abs().max()))
    teorema = err_gancio <= 1e-12 and err_orto <= 1e-12 and err_t31 <= 1e-10
    # export dell'istantanea del passo 0 (griglia): Delta W = 0 esatto -> file == sorgente al byte
    sg = TL.studente("A32", TL.cfg_lora(4))
    ids_c, x_c, R_c = TF._ingresso(sg, seme=11)
    _inizializza(sg, ids_c, x_c, R_c)
    d0 = DIR / "g1_ist"
    shutil.rmtree(d0, ignore_errors=True)
    d0.mkdir(parents=True)
    nome = A.salva_istantanea(d0, sg, 0, {"metrica": 0.0})
    ist = A.carica_istantanea(d0 / nome)
    dst = DIR / "g1_passo0.gguf"
    for p in (dst, Path(str(dst) + ".provenienza.json")):
        p.unlink(missing_ok=True)
    prov = X.esporta(TL.gguf_lora(), dst, ist["scale"], ist["meta"], controlla_disco=False, log=muto,
                     lora=ist["lora"], lora_meta=ist["lora_meta"], lora_modo="griglia")
    npz_ga = all(set(v) == {"A", "B", "A0", "B0"} for v in ist["lora"].values()) and ist["lora_meta"].get("init") == "ga"
    delta0 = all(float(np.abs(a.delta_file()).max()) == 0.0 for a in sg.lora.values())
    b0 = TL._sha(dst) == TL._sha(TL.gguf_lora()) and sum(prov["lora"]["verifica"]["byte_cambiati_per_tensore"].values()) == 0
    ok &= teorema and npz_ga and delta0 and b0
    return registra("G1_passo0_al_bit", ok, ricette=det, err_gancio_vs_autograd=err_gancio, err_ortonormalita=err_orto,
                    err_teorema_3_1=err_t31, npz_con_A0_B0=npz_ga, delta_passo0_zero=delta0,
                    export_passo0_uguale_al_sorgente=b0, secondi=round(time.time() - t0, 1))


# ─────────────────────────────── G2 ───────────────────────────────
def prova_g2() -> bool:
    t0 = time.time()
    st = TL.studente("A64", TL.cfg_lora(3, alpha=6.0))
    TF._theta_casuale(st)
    with TF.tutto_fp64():
        ids_c, x_c, R_c = TF._ingresso(st, seme=11)
        _inizializza(st, ids_c, x_c, R_c)
        g = torch.Generator().manual_seed(5)
        with torch.no_grad():                                        # lontano dal passo 0: entrambi i termini vivi
            for a in st.lora.values():
                a.A.add_(0.1 * a.A0.abs().mean() * torch.randn(a.A.shape, generator=g, dtype=torch.float64))
                a.B.add_(0.1 * a.B0.abs().mean() * torch.randn(a.B.shape, generator=g, dtype=torch.float64))
        # uscita del modulo == Linear denso con W + s (B A - B0 A0)
        err_mod = 0.0
        for L in st.strati:
            for m in st.moduli[L].modules():
                if isinstance(m, LR.LinearLoRA):
                    a = m.ad
                    xx = torch.randn(5, a.forma[1], generator=g, dtype=torch.float64)
                    W = m.base.weight.detach() + a.scala * (a.B @ a.A - a.B0 @ a.A0).detach()
                    with torch.no_grad():
                        err_mod = max(err_mod, float((m(xx) - xx @ W.T).abs().max() / (xx @ W.T).abs().max()))
        # delta_file == s (B A - B0 A0) riportato nell'ordine del file
        err_delta = max(float(np.abs(a.delta_file().astype(np.float64) - LR.hf_a_file(
            (a.scala * (a.B @ a.A - a.B0 @ a.A0)).detach().float().numpy(), a.voce).astype(np.float64)).max())
            for a in st.lora.values())
        ids, x, R = TF._ingresso(st, T=16)

        def perdita_nograd():
            with torch.no_grad():
                out = st.avanti(ids, x0=x)
                return float((st.esce(out["xL"]).to(torch.float64) * R).sum()), [o.clone() for o in out["instr"]]
        _, _, _, instr0 = TL._passo(st, ids, x, R)
        gr = {(n, k): getattr(a, k).grad.clone() for n, a in st.lora.items() for k in ("A", "B")}
        st.azzera_cambi()
        gen = torch.Generator().manual_seed(12)
        peggiore, cambi = 0.0, 0
        for n, a in st.lora.items():
            for k in ("A", "B"):
                m = getattr(a, k)
                idx = torch.randperm(m.numel(), generator=gen)[:3]
                num = den = 0.0
                for i in idx.tolist():
                    v0 = m.detach().reshape(-1)[i].item()
                    h = 1e-6 * max(1.0, abs(v0))
                    vals = []
                    for sgn in (+1, -1):
                        with torch.no_grad():
                            m.view(-1)[i] = v0 + sgn * h
                        f, ins = perdita_nograd()
                        cambi += sum(int((torch.sort(u, -1).values != torch.sort(v, -1).values).any(-1).sum())
                                     for u, v in zip(ins, instr0))
                        vals.append(f)
                    with torch.no_grad():
                        m.view(-1)[i] = v0
                    fd = (vals[0] - vals[1]) / (2 * h)
                    an = gr[(n, k)].reshape(-1)[i].item()
                    num = max(num, abs(fd - an))
                    den = max(den, abs(an), abs(fd))
                peggiore = max(peggiore, num / max(den, 1e-12))
    ok = err_mod <= 1e-12 and err_delta <= 1e-6 and peggiore < 1e-5 and cambi == 0
    return registra("G2_forma_ga_gradienti_e_delta", ok, err_modulo_vs_denso=err_mod, err_delta_file=err_delta,
                    err_rel_diff_finite=peggiore, cambi_instradamento=cambi, secondi=round(time.time() - t0, 1))


# ─────────────────────────────── G3 ───────────────────────────────
def sorgente_lotto_pieno(n_righe: int = 8, ripeti: int = 8, T: int = 32, seme: int = 5, scala_b: float = 0.08):
    """Maestro = studente minuscolo con theta* = 0 e un adattatore di rango 4 col seme 777 (A* diverso dall'A dello
    studente, seme della corsa 20260926) e B* casuale: l'unico divario e' un Delta W* di rango 4 che nessuna init conosce.
    Lotto PIENO deterministico: le stesse `n_righe` ripetute `ripeti` volte (epoca 0 = ordine del piano: ogni passo di
    `n_righe` righe vede esattamente quelle), held-out = le stesse righe (domini a/b, niente wiki): la metrica e' la
    perdita di addestramento a lotto pieno. E' la domanda del paper (velocita' di convergenza), senza il rumore del lotto."""
    st = TL.studente("A32", TL.cfg_lora(4, seme=777))
    TL._b_casuale(st, scala_b, seme + 1)
    g = torch.Generator().manual_seed(seme)
    base = []
    for _ in range(n_righe):
        ids = torch.randint(0, st.cfg.vocab_size, (T,), generator=g)
        with torch.no_grad():
            h = st.esce(st.avanti(ids[None])["xL"]).to(torch.bfloat16).clone()
        base.append((ids.numpy().astype(np.int32), h))
    train = [(i, h, "tr", None) for _ in range(ripeti) for (i, h) in base]
    held = [(i, h, "ab"[k % 2], None) for k, (i, h) in enumerate(base)]
    return A.SorgenteMemoria({"train": train, "heldout": held}, {}), st.testa(torch.float32)


def _curva_g3(src, WS, ini: str, lr_l: float, n_righe: int = 8) -> list:
    d = DIR / f"g3_{ini}_{lr_l:g}"
    shutil.rmtree(d, ignore_errors=True)
    kw = {"lora_init": "ga"} if ini == "ga" else {}
    ca = TL._ca_lora(4, lr=1e-6, lr_lora=lr_l, heldout_ogni=1, ripresa_ogni=100, righe_per_passo=n_righe,
                     righe_per_micro=n_righe // 2, righe_per_micro_valuta=n_righe // 2, **kw)
    z = A.allena(TL.studente("A32", TL.cfg_lora(4)), src, d, ca, WS, WS, log=muto)
    return [round(v["metrica"], 6) for v in z["storia_heldout"]]


G3_LR = (1e-4, 3e-4)
G3_CONTROLLO = ((1e-4, 1e-3),)          # (lr di ga, lr di zero): zero con lr 10x (piu' del rapporto |dW1| GA/zero stimato)


def prova_g3() -> bool:
    t0 = time.time()
    src, WS = sorgente_lotto_pieno()
    per_lr, ok, curve = {}, True, {}
    for lr_l in sorted(set(G3_LR) | {x for c in G3_CONTROLLO for x in c}):
        for ini in ("zero", "ga"):
            if ini == "ga" and lr_l not in G3_LR and lr_l not in {c[0] for c in G3_CONTROLLO}:
                continue
            curve[(ini, lr_l)] = _curva_g3(src, WS, ini, lr_l)
    for lr_l in G3_LR:
        cz, cg = curve[("zero", lr_l)], curve[("ga", lr_l)]
        m14 = (sum(cz[1:5]) / 4, sum(cg[1:5]) / 4)
        esito = cz[0] == cg[0] and m14[1] < m14[0] and cg[-1] < cz[-1] and all(b < a for a, b in zip(cz[1:], cg[1:]))
        per_lr[f"{lr_l:g}"] = {"zero": cz, "ga": cg, "media_passi_1_4": m14,
                               "calo_rel_passo_8": ((cz[0] - cz[-1]) / cz[0], (cg[0] - cg[-1]) / cg[0]), "ga_meglio": esito}
        ok &= esito
    controllo = {}
    for lg, lz in G3_CONTROLLO:
        cg, cz = curve[("ga", lg)], curve[("zero", lz)]
        controllo[f"ga {lg:g} contro zero {lz:g}"] = {"zero": cz, "ga": cg, "ga_meglio_al_passo_8": cg[-1] < cz[-1]}
        ok &= cg[-1] < cz[-1]
    return registra("G3_discesa_ga_vs_zero_lotto_pieno", ok, per_lr=per_lr, controllo_lr_zero_maggiore=controllo,
                    secondi=round(time.time() - t0, 1))


# ─────────────────────────────── G4 ───────────────────────────────
def prova_g4() -> bool:
    import e2e_test_allena as TA
    t0 = time.time()
    src, WS = TL.sorgente_lora()
    ca = TL._ca_lora(lora_init="ga")
    d_f, d_r = DIR / "g4_filata", DIR / "g4_ripresa"
    for d in (d_f, d_r):
        shutil.rmtree(d, ignore_errors=True)
    st_f = TL.studente("A32", TL.cfg_lora(4))
    filata = A.allena(st_f, src, d_f, ca, WS, WS, log=muto)
    A.allena(TL.studente("A32", TL.cfg_lora(4)), src, d_r, ca, WS, WS, ferma_a=3, log=muto)
    st_b = TL.studente("A32", TL.cfg_lora(4))
    ripresa = A.allena(st_b, src, d_r, ca, WS, WS, riprendi=True, log=muto)
    rf, rr = A.carica_ripresa(d_f, "cpu"), A.carica_ripresa(d_r, "cpu")
    th = all(torch.equal(a, b) for a, b in zip(st_f.parametri(), st_b.parametri()))
    lo = TL._uguali_lora(st_f, st_b) and all(torch.equal(st_f.lora[n].A0, st_b.lora[n].A0)
                                              and torch.equal(st_f.lora[n].B0, st_b.lora[n].B0) for n in st_f.lora)
    mossi = sum(1 for a in st_f.lora.values() if not torch.equal(a.B, a.B0))
    adam = TF._uguali_adam(rf, rr) and len(rf["opt"]["param_groups"]) == 2
    stato = TA._senza_tempi(filata) == TA._senza_tempi(ripresa) and filata["passo"] == 6 and "lora_ga" in filata \
        and filata["lora_ga"]["passo"] == 0
    file_ga = sorted(p.name for p in d_f.glob("lora_ga_passo*.json")) == ["lora_ga_passo000000.json"] and \
        sorted(p.name for p in d_r.glob("lora_ga_passo*.json")) == ["lora_ga_passo000000.json"]
    ist = A.carica_istantanea(d_f / filata["migliore"]["dir"])
    ist_ok = ist["lora_meta"].get("init") == "ga" and all(set(v) == {"A", "B", "A0", "B0"} for v in ist["lora"].values())
    ripresa_a0 = all("A0" in v for v in rf["lora"].values())
    # LoRA-GA accesa alla ripresa: theta dopo il passo 3 == corsa senza LoRA (il modello non cambia al passo 3)
    d_x, d_n = DIR / "g4_accesa", DIR / "g4_senza"
    for d in (d_x, d_n):
        shutil.rmtree(d, ignore_errors=True)
    A.allena(TL.studente("A32"), src, d_x, TL._ca(), WS, WS, ferma_a=2, log=muto)
    reg = []
    s_x = TL.studente("A32", TL.cfg_lora(4))
    z_x = A.allena(s_x, src, d_x, TL._ca_lora(lora_init="ga"), WS, WS, riprendi=True, ferma_a=3, log=reg.append)
    s_n = TL.studente("A32")
    A.allena(s_n, src, d_n, TL._ca(), WS, WS, ferma_a=3, log=muto)
    r_x = A.carica_ripresa(d_x, "cpu")
    n_th = len(s_n.parametri())
    passi_lora = [int(r_x["opt"]["state"][n_th + j]["step"]) for j in range(2 * len(r_x["lora_nomi"]))]
    accesa = (all(torch.equal(a, b) for a, b in zip(s_x.parametri(), s_n.parametri()))
              and z_x["lora_ga"]["passo"] == 2 and passi_lora == [1] * len(passi_lora)
              and any("LoRA-GA prima del prossimo passo" in x for x in reg) and LR.ha_ga(s_x)
              and (d_x / "lora_ga_passo000002.json").exists())
    # init cambiata su LoRA gia' allenata = errore; config e CLI
    errori = {}
    for nome, d, ca_e in (("zero_poi_ga", DIR / "g4_zero", TL._ca_lora()), ("ga_poi_zero", d_f, TL._ca_lora())):
        if nome == "zero_poi_ga":
            shutil.rmtree(d, ignore_errors=True)
            A.allena(TL.studente("A32", TL.cfg_lora(4)), src, d, ca_e, WS, WS, ferma_a=1, log=muto)
            ca_e = TL._ca_lora(lora_init="ga")
        try:
            A.allena(TL.studente("A32", TL.cfg_lora(4)), src, d, ca_e, WS, WS, riprendi=True, ferma_a=7, log=muto)
            errori[nome] = False
        except RuntimeError as e:
            errori[nome] = "init" in str(e)
    for nome, kw in (("ga_senza_rango", {"lora_init": "ga"}), ("init_ignota", {"lora_rango": 4, "lora_famiglie": TUTTE,
                                                                             "lora_init": "pissa"}),
                     ("gamma_senza_ga", {"lora_rango": 4, "lora_famiglie": TUTTE, "lora_ga_gamma": 64.0}),
                     ("gamma_zero", {"lora_rango": 4, "lora_famiglie": TUTTE, "lora_init": "ga", "lora_ga_gamma": 0.0})):
        try:
            A.ConfigAllena(**kw)
            errori[nome] = False
        except ValueError:
            errori[nome] = True
    c = A._config_da_arg(TF._arg(lora_nonesperti=4, lora_alpha=None, lora_famiglie="tutte", lr_lora=1e-4,
                                 lora_init="ga", lora_ga_gamma=64.0, lora_ga_righe=1))
    c0 = A._config_da_arg(TF._arg(lora_nonesperti=4, lora_alpha=None, lora_famiglie="tutte", lr_lora=1e-4,
                                  lora_init="zero", lora_ga_gamma=64.0, lora_ga_righe=1))
    c_off = A._config_da_arg(TF._arg(lora_nonesperti=0, lora_alpha=None, lora_famiglie="tutte", lr_lora=1e-4,
                                     lora_init="ga"))
    cli = ((c.lora_init, c.lora_ga_gamma, c.lora_ga_righe) == ("ga", 64.0, 1)
           and (c0.lora_init, c0.lora_ga_gamma, c0.lora_ga_righe) == ("zero", 16.0, 0)
           and c_off.lora_init == "zero" and A._config_da_arg(TF._arg()).lora_init == "zero"
           and {"lora_init", "lora_ga_gamma", "lora_ga_righe"} <= A._CAMPI_DI_DECISIONE)
    # --lora-ga-righe 1: dW sulla sola prima riga del passo
    d_1 = DIR / "g4_una_riga"
    shutil.rmtree(d_1, ignore_errors=True)
    z_1 = A.allena(TL.studente("A32", TL.cfg_lora(4)), src, d_1, TL._ca_lora(lora_init="ga", lora_ga_righe=1), WS, WS,
                   ferma_a=1, log=muto)
    una = len(z_1["lora_ga"]["righe"]) == 1 and len(filata["lora_ga"]["righe"]) == ca.righe_per_passo
    ok = th and lo and mossi > 0 and adam and stato and file_ga and ist_ok and ripresa_a0 and accesa \
        and all(errori.values()) and cli and una
    return registra("G4_ripresa_al_bit_ga", ok, theta=th, lora_A_B_A0_B0=lo, adattatori_mossi=mossi, adam=adam,
                    stato=stato, file_lora_ga=file_ga, istantanea=ist_ok, ripresa_con_A0=ripresa_a0,
                    accesa_alla_ripresa=accesa, passi_adam_lora=passi_lora[:4], errori=errori, cli=cli,
                    lora_ga_righe_1=una, lora_ga_stato={k: v for k, v in filata["lora_ga"].items() if k != "righe"},
                    heldout=[(v["passo"], round(v["metrica"], 6)) for v in filata["storia_heldout"]],
                    secondi=round(time.time() - t0, 1))


# ─────────────────────────────── G5 ───────────────────────────────
def prova_g5() -> bool:
    t0 = time.time()
    src = TL.gguf_lora()
    st = TL.studente("A32", TL.cfg_lora(4))
    TF._theta_casuale(st, seme=4, s=0.1)
    ids_c, x_c, R_c = TF._ingresso(st, seme=11)
    _inizializza(st, ids_c, x_c, R_c)
    g = torch.Generator().manual_seed(31)
    with torch.no_grad():                                          # «dopo l'allenamento»: B lontano da B0, dell'ordine
        for a in st.lora.values():                                  # del passo della griglia (in parte sopravvive)
            a.B.add_(0.5 * torch.randn(a.B.shape, generator=g))
            a.A.add_(0.05 * torch.randn(a.A.shape, generator=g))
    d0 = DIR / "g5_ist"
    shutil.rmtree(d0, ignore_errors=True)
    d0.mkdir(parents=True)
    nome = A.salva_istantanea(d0, st, 1, {"metrica": 0.0})
    ist = A.carica_istantanea(d0 / nome)
    theta = torch.load(d0 / nome / "theta.pt", map_location="cpu", weights_only=True)
    delta_ok = all(np.array_equal(LR.delta_file_da(v["A"], v["B"], ist["lora_meta"]["scala"], ist["lora_meta"]["voci"][n],
                                                   v["A0"], v["B0"]), st.lora[n].delta_file())
                   for n, v in ist["lora"].items())
    delta_non_nullo = all(float(np.abs(st.lora[n].delta_file()).max()) > 0 for n in st.lora)
    ric0 = X.ricarica
    X.ricarica = functools.partial(ric0, ricetta="A32", token=24)
    try:
        dst = DIR / "g5_griglia.gguf"
        for p in (dst, Path(str(dst) + ".provenienza.json"), dst.with_name(dst.name + ".parziale")):
            p.unlink(missing_ok=True)
        prov = X.esporta(src, dst, ist["scale"], ist["meta"], theta=theta, strati_ricarica=(0, 1), controlla_disco=False,
                         log=muto, lora=ist["lora"], lora_meta=ist["lora_meta"], lora_modo="griglia")
    finally:
        X.ricarica = ric0
    ric = prov["ricarica"]
    ricarica_ok = all(r["scale_al_bit"] and r["uscita_al_bit"] for r in ric.values())
    ra, rb = GGUFReader(str(src)), GGUFReader(str(dst))
    ta, tb = {t.name: t for t in ra.tensors}, {t.name: t for t in rb.tensors}
    tens_ok, cambiati = True, 0
    for n in ist["lora"]:
        va, vb = np.asarray(ta[n].data), np.asarray(tb[n].data)
        tipo = ta[n].tensor_type.name
        atteso = LR.riquantizza(va, LR.dequantizza(va, tipo) + st.lora[n].delta_file(), tipo, "griglia")
        tens_ok &= np.array_equal(vb, atteso)
        cambiati += int((va != vb).sum())
    del ra, rb, ta, tb
    uscita = {L: r.get("lora_uscita") for L, r in ric.items()}
    ok = delta_ok and delta_non_nullo and ricarica_ok and tens_ok and cambiati > 0 and \
        all(u is not None and u["mantenuto"] == u["mantenuto"] for u in uscita.values())
    return registra("G5_export_dopo_ga", ok, delta_istantanea_uguale=delta_ok, delta_non_nullo=delta_non_nullo,
                    ricarica_al_bit=ricarica_ok, tensori=tens_ok, byte_cambiati=cambiati, uscita=uscita,
                    secondi=round(time.time() - t0, 1))


# ─────────────────────────────── G6 ───────────────────────────────
def prova_g6() -> bool:
    t0 = time.time()
    g = torch.Generator().manual_seed(9)
    m, n, k = 1536, 768, 16
    assert m * n > LR.GA_ESATTA_MAX
    U0 = torch.linalg.qr(torch.randn(m, n, generator=g, dtype=torch.float64)).Q
    V0 = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=torch.float64)).Q
    sv = 0.93 ** torch.arange(n, dtype=torch.float64)                   # spettro che decade (come un gradiente)
    M = ((U0 * sv) @ V0.T).float()
    Ue, Se, Vhe = torch.linalg.svd(M.double(), full_matrices=False)
    ris = {}
    ok = True
    for dt in (torch.float32, torch.float64):
        Mm = M.to(dt)
        U, Sv, V = LR.svd_alti(Mm, k, 123)
        U2, S2, V2 = LR.svd_alti(Mm, k, 123)
        det = torch.equal(U, U2) and torch.equal(Sv, S2) and torch.equal(V, V2)
        e_s = float(((Sv - Se[:k]).abs() / Se[:k]).max())
        e_u = float((U @ U.T - Ue[:, :k] @ Ue[:, :k].T).abs().max())
        e_v = float((V @ V.T - Vhe[:k].T @ Vhe[:k]).abs().max())
        orto = float((U.T @ U - torch.eye(k, dtype=torch.float64)).abs().max())
        soglia = 1e-5 if dt == torch.float32 else 1e-8
        ris[str(dt)] = {"deterministica": det, "err_sigma": e_s, "err_sottospazio_U": e_u, "err_sottospazio_V": e_v,
                        "ortonormalita": orto}
        ok &= det and e_s <= soglia and e_u <= soglia and e_v <= soglia and orto <= (1e-6 if dt == torch.float32 else 1e-12)
    Up, Sp, Vp = LR.svd_alti(M[:512, :256], k, 1)                        # piccola: via esatta
    Ue2, Se2, _ = torch.linalg.svd(M[:512, :256].double(), full_matrices=False)
    esatta = torch.equal(Sp, Se2[:k].contiguous()) and torch.equal(Up, Ue2[:, :k].contiguous())
    ok &= esatta
    return registra("G6_svd_alti", ok, per_dtype=ris, via_esatta_uguale_linalg=esatta, secondi=round(time.time() - t0, 1))


PROVE = {"G1": prova_g1, "G2": prova_g2, "G3": prova_g3, "G4": prova_g4, "G5": prova_g5, "G6": prova_g6}


def main() -> int:
    DIR.mkdir(parents=True, exist_ok=True)
    TL.DIR.mkdir(parents=True, exist_ok=True)
    scelte = [a.upper() for a in sys.argv[1:]] or list(PROVE)
    ok, t0 = True, time.time()
    for k in scelte:
        t1 = time.time()
        try:
            ok &= PROVE[k]()
        except Exception as e:
            ok = False
            registra(f"{k}_eccezione", False, errore=repr(e), traccia=traceback.format_exc()[-3000:])
        print(f"   ({k}: {time.time() - t1:.1f} s)", flush=True)
    ESITO["tutto_verde"] = ok
    ESITO["secondi"] = round(time.time() - t0, 1)
    suff = "_" + "_".join(scelte) if sys.argv[1:] else ""
    (DIR / f"esito_test_lora_ga{suff}.json").write_text(json.dumps(ESITO, indent=1, default=str))
    print(f"ESITO: {'TUTTO VERDE' if ok else 'ROSSO'} ({ESITO['secondi']} s)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
