"""Applica UNA leva di raffinamento ai 250 densi multi-piano di R2B3m e scrive i byte dei piani nuovi
(<OUT>/<leva>/tensori/<nome_piano>.npy, uint8 [righe, byte_riga]) + registro per tensore (registro.jsonl).
Riprendibile: un tensore con tutti i piani gia' scritti e la riga nel registro si salta.

uso: applica.py --leva LEVA [--strati 0-39] [--solo nome,...] [--device cpu|cuda]
leve sui densi:
  jarq_p123   JARQ (2609.38599) sui 3 piani: scale congiunte + Babai sul composito, ospite = R2B3m
  jarq_p23    idem con il piano 1 (quello allenato end-to-end da TT2) FERMO
  tied_uff    Tied Trit-Planes (2608.08910), risolutore UFFICIALE di fucina (ptqtp.zig solveGroupTied, K=3):
              quantizzatore uniforme a 27 livelli per gruppo, passo s da una spazzata di 65 divisori dell'absmax,
              MSE non pesato, scale (9s,3s,s) in fp16
  tied_jarq   stesso vincolo (s_k = 9s,3s,s) ma ottimizzato pesato-H con il risolutore JARQ (scala congiunta legata)
  schur_p23 / schur_p123   SchurOpt (2608.15567) sul composito, avvio dall'ospite, R=4 passi, lambda = LEVE_LAM
  qtea_p23    QTEA (2609.00224): piani 2 e 3 rifatti con GPTQ ternario + error decay (lambda_decay=1, B=128),
              poi scale congiunte del piano (ricetta di casa); la riscalatura per colonna NON e' caricabile sui
              tensori ruotati (vedi LEGGIMI)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import comune as CM  # noqa: E402
import solutori as SO  # noqa: E402

LEVE = ("jarq_p123", "jarq_p23", "tied_uff", "tied_jarq", "schur_p23", "schur_p123", "qtea_p23",
        "g2ptq_p23", "g2ptq0_p23", "fastkron_p23", "tied_jarq_g2ptq", "g2ptq_jarq", "g2ptq_auto_p23",
        "bil_p123", "bil_tjarq", "bil_foem", "bil_tjarq_p1fermo",
        "jarq_pk", "jarq_p2k", "schur_pk", "schur_p2k", "tjarq_p1fermo")
# 9/10 AUDIT_G2PTQ_THOR: alpha fisso a 1 raddoppiava la KL (0,88 contro 0,416 con alpha=0, stesse A e quantizzatore).
# Il codice ufficiale sceglie alpha per blocco su una griglia [0,3] con arresto al primo peggioramento della perdita
# VERA (App. C.2, Alg. 3). Qui l'arbitro e' l'errore pesato con l'Hessiana del manifesto H' (indipendente da A).
ALPHA_GRIGLIA = tuple(float(x) for x in os.getenv("LEVE_ALPHA_GRIGLIA", "0,0.03,0.1,0.3,1,3").split(","))
GRAD = os.getenv("LEVE_GRAD", "@@ROOT:models@@/gguf/tony-tern/LEVE_RAFF/grad")
_PROV_CORRENTE: dict = {}   # provenienza del tensore in lavoro (bersaglio_e_H), letta da bil_foem


def tied_ufficiale(W, Wfisso, K, ng, legame):
    """Port esatto di fucina/src/ptqtp.zig::solveGroupTied (planes=3): per gruppo, s = absmax/div con
    div = (L-0.75) + i*(2/64), i=0..64, L=13; codice c = clamp(round(x/s), -L, L); si tiene l'MSE minimo."""
    R, C = W.shape
    G = C // ng
    X = (W - Wfisso).to(SO.F64).reshape(R, ng, G)
    L = (3 ** K - 1) // 2
    am = X.abs().amax(-1, keepdim=True)
    best_s = am / L
    best_e = torch.full_like(am, float("inf"))
    for i in range(65):
        div = (L - 0.75) + i * (2.0 / 64.0)
        s = am / div
        c = torch.clamp(torch.round(X / torch.where(s > 0, s, torch.ones_like(s))), -L, L)
        e = ((X - s * c) ** 2).sum(-1, keepdim=True)
        m = e < best_e
        best_e = torch.where(m, e, best_e)
        best_s = torch.where(m, s, best_s)
    c = torch.clamp(torch.round(X / torch.where(best_s > 0, best_s, torch.ones_like(best_s))), -L, L)
    c = torch.where(am > 0, c, torch.zeros_like(c)).reshape(R, C).long()
    T = SO.trit_bilanciati(c, K)
    S = SO.fp16(legame.to(SO.F64).view(K, 1, 1) * best_s.reshape(1, R, ng))
    return T, S


def applica_leva(leva, d, W, H):
    T, S = d.T.to(W.device), d.S.to(W.device).to(SO.F64)
    K, R, C = T.shape
    ng = S.shape[-1]
    legame = torch.tensor([3.0 ** (K - 1 - k) for k in range(K)], dtype=SO.F64, device=W.device)
    lam = float(os.getenv("LEVE_LAM", "0.0"))
    # 9/10 COMBINAZIONE_LEVE: varianti GENERICHE in K (PV12P ha famiglie a 2, 3 e 4 piani). jarq_p23/schur_p23 hanno
    # i liberi [1, 2] fissi: con K=2 indicizzano un piano che non c'e', con K=4 lasciano fuori il 4o. "_pk" = tutti i
    # piani liberi, "_p2k" = piano 1 (codici TT2 allenati end-to-end) FERMO al bit, liberi i piani 2..K.
    if leva in ("jarq_pk", "jarq_p2k"):
        return SO.jarq(W, H, T, S, list(range(K)) if leva == "jarq_pk" else list(range(1, K)), giri=3, nu2=1e-4)
    if leva in ("schur_pk", "schur_p2k"):
        return SO.schuropt(W, H, T, S, list(range(K)) if leva == "schur_pk" else list(range(1, K)), ng, passi=4, lam=lam)
    if leva == "tjarq_p1fermo":
        # Tied/JARQ con il piano 1 dell'ospite fermo al bit: griglia legata 3^(K-2-k) (RTN ufficiale di Tied) sul residuo
        # W - W1, poi JARQ legato sui soli piani 2..K. Gemello di bil_tjarq_p1fermo con l'avvio RTN invece di GPTQ.
        W1 = SO.ricostruisci(T[:1], S[:1])
        leg_l = torch.tensor([3.0 ** (K - 2 - k) for k in range(K - 1)], dtype=SO.F64, device=W.device)
        Tl, Sl = tied_ufficiale(W - W1, torch.zeros_like(W), K - 1, ng, leg_l)
        T1, S1 = T.clone(), S.clone().to(SO.F64)
        T1[1:], S1[1:] = Tl.to(T1.dtype), Sl.to(SO.F64)
        T2, S2, cj = SO.jarq(W, H, T1, S1, list(range(1, K)), giri=3, nu2=1e-4, legame=leg_l)
        cj["ospite"] = float(SO.obiettivo(W, SO.ricostruisci(T, S), H).sum())
        return T2, S2, cj
    if leva == "jarq_p123":
        return SO.jarq(W, H, T, S, [0, 1, 2], giri=3, nu2=1e-4)
    if leva == "jarq_p23":
        return SO.jarq(W, H, T, S, [1, 2], giri=3, nu2=1e-4)
    if leva == "tied_uff":
        T2, S2 = tied_ufficiale(W, torch.zeros_like(W), K, ng, legame)
        f0 = float(SO.obiettivo(W, SO.ricostruisci(T, S), H).sum())
        f1 = float(SO.obiettivo(W, SO.ricostruisci(T2, S2), H).sum())
        return T2, S2.float(), {"ospite": f0, "finale": f1}
    if leva == "tied_jarq":
        # 9/10: TUTTI i K piani dell'ospite sono liberi e legati 3^(K-1-k) (K=3: 9:3:1; K=4, famiglie col 4o piano di
        # PV12P: 27:9:3:1 = griglia uniforme a 81 livelli; K=2: 3:1). Prima i liberi erano [0, 1, 2] fissi: con K=4 il
        # legame a 4 voci incontrava 3 piani (RuntimeError view [3,1,1,1] in scale_congiunte) e il 4o piano restava fuori.
        T0, S0 = tied_ufficiale(W, torch.zeros_like(W), K, ng, legame)
        T2, S2, cat = SO.jarq(W, H, T0, S0, list(range(K)), giri=3, nu2=1e-4, legame=legame)
        cat["ospite_R2B3m"] = float(SO.obiettivo(W, SO.ricostruisci(T, S), H).sum())
        return T2, S2, cat
    if leva in ("schur_p23", "schur_p123"):
        lib = [1, 2] if leva == "schur_p23" else [0, 1, 2]
        return SO.schuropt(W, H, T, S, lib, ng, passi=4, lam=lam)
    if leva == "qtea_p23":
        W1 = SO.ricostruisci(T[:1], S[:1])
        T2, S2 = T.clone(), S.clone()
        res = W - W1
        for k in (1, 2):
            tk, sk, _ = SO.gptq_piano_decay(res, H, ng, lam_decay=1.0)
            fisso = W - res
            sk = SO.fp16(SO.scale_congiunte(W, H, tk.unsqueeze(0), fisso, ng)[0])
            T2[k], S2[k] = tk, sk
            res = res - SO.ricostruisci(tk.unsqueeze(0), sk.unsqueeze(0))
        f0 = float(SO.obiettivo(W, SO.ricostruisci(T, S), H).sum())
        f1 = SO.obiettivo(W, SO.ricostruisci(T2, S2), H)
        return T2, S2.float(), {"ospite": f0, "finale": float(f1.sum())}
    if leva == "tied_jarq_g2ptq":
        # 8/10 (Angelo: in serie, non al posto): Tied/JARQ sceglie il piano 1 e la scala legata, poi G2PTQ risolve i piani 2-3
        T0, S0, _ = applica_leva("tied_jarq", d, W, H)
        return _g2ptq_su(d, W, T0.to(W.device), S0.to(W.device).to(SO.F64), "g2ptq_p23")
    if leva == "g2ptq_jarq":
        # ordine inverso: G2PTQ sui piani 2-3 di R2B3m, poi raffinamento congiunto JARQ (pesato H) su tutti i piani
        T1, S1, _ = _g2ptq_su(d, W, T, S, "g2ptq_p23")
        return SO.jarq(W, H, T1.to(W.device), S1.to(W.device).to(SO.F64), [0, 1, 2], giri=3, nu2=1e-4)
    if leva in ("g2ptq_p23", "g2ptq0_p23", "fastkron_p23"):
        return _g2ptq_su(d, W, T, S, leva)
    if leva == "g2ptq_auto_p23":
        return _g2ptq_su(d, W, T, S, leva, Harb=H)
    if leva == "bil_foem":
        # 9/10 FOEM (2507.11017) sul GPTQ della griglia bilanciata (= Tied: stessi 3^K livelli, scale 3^(K-1-k) D).
        # Forma del LAVORO (Alg. 1, Eq. 19): invariante alla scala di H, beta = 0,1 come nel lavoro (LEVE_FOEM_BETA).
        # LEVE_FOEM_FORMA=codice: forma del repository, beta riportato alla scala della nostra H; sui tensori con
        # rho = beta·lambda_max(H^-1) >= 1 (causa dell'esplosione del 2/9: ssm_out rho 2,4, attn_output 6,8) FOEM resta spento.
        # NIENTE JARQ dopo: JARQ accetta solo cio' che abbassa l'errore del secondo ordine e cancellerebbe proprio la
        # correzione del primo ordine che FOEM introduce.
        beta = float(os.getenv("LEVE_FOEM_BETA", "0.1"))
        forma = os.getenv("LEVE_FOEM_FORMA", "lavoro")
        cat0 = {"beta_lavoro": beta, "forma": forma}
        be = beta
        if forma == "codice":
            dbar = _PROV_CORRENTE.get("diag_media_grezza")
            assert dbar, f"{d.nome}: manca la diagonale grezza dell'Hessiana"
            be = SO.foem_beta_efficace(beta, dbar)
            rho = SO.foem_rho(H, be)
            cat0.update({"diag_media_grezza": dbar, "rho": rho})
            if rho >= 1.0:
                # fuori dalla regione di stabilita' del passo (rho >= 1 oscilla, >= 2 diverge): il codice ufficiale qui
                # esploderebbe. Il tensore resta BIL puro e il registro lo dice (nessun beta inventato).
                be = 0.0
                cat0["foem_spento_rho"] = True
        T1, S1, cat = SO.gptq_bilanciato(W, H, K, ng, foem_beta=be, foem_forma=forma)
        cat.update(cat0)
        cat.update({"ospite": float(SO.obiettivo(W, SO.ricostruisci(T, S), H).sum()),
                    "finale": float(SO.obiettivo(W, SO.ricostruisci(T1, S1), H).sum())})
        return T1, S1.float(), cat
    if leva == "bil_tjarq_p1fermo":
        # 9/10 (diagnosi BIL, LEVE_RAFF_PV12P/diagnosi): bil_tjarq e COMBO A riscrivono il PIANO 1, che in PV12P (famiglie
        # a 3 piani, byte = R2B3m = R2) porta i codici ALLENATI end-to-end da TT2 (+9,3 MMLU): il 21 % dei trit del piano 1
        # cambia, e la KL-4 della risposta MMLU peggiora contro PV12P (z +5,1) mentre la KL delle 24 righe migliora.
        # Qui il piano 1 dell'ospite resta FERMO al bit (codici e scala); BIL rifa' i piani 2..K sul residuo W - W1
        # (griglia bilanciata a 3^(K-1) livelli), poi JARQ legato 3^(K-2-k) sui soli piani 2..K.
        W1 = SO.ricostruisci(T[:1], S[:1])
        Tl, Sl, cat = SO.gptq_bilanciato(W - W1, H, K - 1, ng)
        T1, S1 = T.clone(), S.clone()
        T1[1:], S1[1:] = Tl, Sl
        cat["ospite"] = float(SO.obiettivo(W, SO.ricostruisci(T, S), H).sum())
        cat["bil_p2k"] = float(SO.obiettivo(W, SO.ricostruisci(T1, S1), H).sum())
        leg_l = torch.tensor([3.0 ** (K - 2 - k) for k in range(K - 1)], dtype=SO.F64, device=W.device)
        T2, S2, cj = SO.jarq(W, H, T1, S1, list(range(1, K)), giri=3, nu2=1e-4, legame=leg_l)
        cj.update({f"bil_{k}": v for k, v in cat.items()})
        return T2, S2, cj
    if leva in ("bil_p123", "bil_tjarq"):
        # 9/10 BIL: GPTQ sulla griglia bilanciata a 3^K livelli su TUTTI i piani dell'ospite (K=3, o 4 dove PV12P ha il
        # piano 4). bil_tjarq = BIL in SERIE con Tied/JARQ: l'avvio di JARQ (2609.38599 Alg. 1, ospite GPTQ come nel
        # lavoro) e' BIL invece del RTN tied_uff; JARQ col legame 3^(K-1-k) tiene la griglia uniforme e non puo'
        # peggiorare nessuna riga rispetto a BIL (le righe peggiori tornano all'ospite).
        T1, S1, cat = SO.gptq_bilanciato(W, H, K, ng)
        cat["ospite"] = float(SO.obiettivo(W, SO.ricostruisci(T, S), H).sum())
        cat["bil"] = float(SO.obiettivo(W, SO.ricostruisci(T1, S1), H).sum())
        if leva == "bil_p123":
            return T1, S1.float(), cat
        T2, S2, cj = SO.jarq(W, H, T1, S1, list(range(K)), giri=3, nu2=1e-4, legame=legame)
        cj.update({f"bil_{k}": v for k, v in cat.items()})
        return T2, S2, cj
    raise ValueError(leva)


def _g2ptq_su(d, W, T, S, leva, Harb=None):
    """Piani 2-3 risolti con G2PTQ/FastKron sul residuo del piano 1 dell'ospite (T, S)."""
    K, R, C = T.shape
    ng = S.shape[-1]
    if True:
        nome = d.nome
        A = torch.from_numpy(np.load(f"{GRAD}/A_{nome}.npy").astype(np.float32)).to(W.device).to(SO.F64)
        A = 0.5 * (A + A.T)
        W1 = SO.ricostruisci(T[:1], S[:1])
        res = W - W1
        if leva == "fastkron_p23":
            Bm = torch.from_numpy(np.load(f"{GRAD}/B_{nome}.npy").astype(np.float32)).to(W.device).to(SO.F64)
            Tl, Sl, cat = SO.fastkron_composito(res, A, 0.5 * (Bm + Bm.T), ng, Kl=2)
        else:
            g = torch.from_numpy(np.load(f"{GRAD}/g_{nome}.npy")).to(W.device).to(SO.F64)
            tsv = open(f"{GRAD}/scale_kron.tsv").read()
            scA = float(tsv.split(f"A\t{nome}\t")[1].split("\t")[0])
            scB, ntok = (float(x) for x in tsv.split(f"B\t{nome}\t")[1].split("\n")[0].split("\t")[:2])
            Bd = torch.from_numpy(np.load(f"{GRAD}/B_{nome}.npy").astype(np.float32)).diagonal().to(W.device).to(SO.F64)
            Tr = ntok / float(os.getenv("LEVE_GRAD_RIGHE", "32"))
            # Hessiana della KL per la riga j (G2PTQ Teor. 2, Fisher con canali indipendenti, eq. 8):
            #   H_j ~ T^2 E_t[gy_j^2] E_t[x xᵀ] = T^2 * scB*B_jj * scA*A   (A, B salvate a diag media 1)
            hj = (Tr ** 2) * scB * Bd.clamp_min(1e-6) * scA
            if leva == "g2ptq_auto_p23":
                # alpha per tensore (App. C.2): griglia crescente, arresto al primo peggioramento dell'arbitro H'
                migliore, prove = None, []
                for alpha in ALPHA_GRIGLIA:
                    Tl, Sl, cat = SO.g2ptq_composito(res, A, g / hj.unsqueeze(1), ng, Kl=2, alpha=alpha)
                    T2, S2 = T.clone(), S.clone()
                    T2[1:], S2[1:] = Tl, SO.fp16(Sl)
                    f = float(SO.obiettivo(W, SO.ricostruisci(T2, S2), Harb).sum())
                    prove.append([alpha, f])
                    if migliore is not None and f > migliore[0]:
                        break
                    if migliore is None or f <= migliore[0]:
                        migliore = (f, alpha, Tl, Sl, cat)
                _, alpha, Tl, Sl, cat = migliore
                cat = dict(cat, alpha_scelto=alpha, prove_alpha=prove)
            else:
                alpha = float(os.getenv("LEVE_ALPHA", "1.0")) if leva == "g2ptq_p23" else 0.0
                Tl, Sl, cat = SO.g2ptq_composito(res, A, g / hj.unsqueeze(1), ng, Kl=2, alpha=alpha)
        T2, S2 = T.clone(), S.clone()
        T2[1:], S2[1:] = Tl, Sl
        f0 = float(SO.obiettivo(W, SO.ricostruisci(T, S), A).sum())
        cat.update({"ospite_A": f0, "finale_A": float(SO.obiettivo(W, SO.ricostruisci(T2, S2), A).sum())})
        return T2, S2.float(), cat


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--leva", required=True, choices=LEVE)
    ap.add_argument("--strati", default="0-39")
    ap.add_argument("--solo", default="")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    a0, b0 = (int(x) for x in a.strati.split("-"))
    strati = list(range(a0, b0 + 1))
    out = CM.OUT_BASE / a.leva
    td = out / "tensori"
    td.mkdir(parents=True, exist_ok=True)
    reg = out / "registro.jsonl"
    fatti = set()
    if reg.exists():
        for r in reg.read_text().splitlines():
            if r.strip():
                fatti.add(json.loads(r)["nome"])
    dev = torch.device(a.device)
    src = CM.gp.GGUFPlaneSource(CM.OSPITE, layers=strati)
    nomi = [n for n in CM.elenco_densi(src) if int(n.split(".")[1]) in strati]
    if a.solo:
        nomi = [n for n in nomi if n in set(a.solo.split(","))]
    CM.log(f"ospite {CM.OSPITE}, uscita {out}")
    CM.log(f"leva {a.leva}: {len(nomi)} tensori, gia' fatti {len(fatti & set(nomi))}, device {dev}")
    t_tot = time.time()
    for nome in nomi:
        if nome in fatti:
            continue
        t0 = time.time()
        d = CM.Denso(src, nome)
        W, H, prov = CM.bersaglio_e_H(nome, d.rotated)
        _PROV_CORRENTE.clear(); _PROV_CORRENTE.update(prov)
        assert H is not None, f"{nome}: manca l'Hessiana"
        W, H = W.to(dev).to(SO.F64), H.to(dev).to(SO.F64)
        e0 = CM.err_pesato(W.cpu().double(), d.ricostruisci(), H.cpu().double())
        T2, S2, cat = applica_leva(a.leva, d, W, H)
        T2, S2 = T2.cpu(), S2.cpu().float()
        assert T2.shape == d.T.shape and S2.shape == d.S.shape
        assert torch.equal(S2, S2.to(torch.float16).float()), "scale non fp16"
        e1 = CM.err_pesato(W.cpu().double(), SO.ricostruisci(T2, S2).cpu(), H.cpu().double())
        byte = d.byte_piani(T2, S2)
        for nm, bb in zip(d.nomi_piani, byte):
            np.save(td / f"{nm}.npy", bb)
        riga = {"nome": nome, "piani": d.nomi_piani, "errH_prima": e0[0], "errH_dopo": e1[0], "fro_prima": e0[1],
                "fro_dopo": e1[1], "catena": cat, "secondi": round(time.time() - t0, 1)}
        with open(reg, "a") as f:
            f.write(json.dumps(riga) + "\n")
        CM.log(f"{nome}: errH {e0[0]:.4f} -> {e1[0]:.4f}  fro {e0[1]:.4f} -> {e1[1]:.4f}  ({riga['secondi']} s)")
    CM.log(f"leva {a.leva} finita in {time.time() - t_tot:.0f} s")


if __name__ == "__main__":
    main()
