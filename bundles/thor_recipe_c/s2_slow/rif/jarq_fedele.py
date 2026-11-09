#!/usr/bin/env python3
"""Obiettivo C — sequenza FEDELE di JARQ (arXiv 2609.38599) sui 250 densi a UN piano (11/10/2026), SOLO CPU.

Perche' esiste: jarq1/bil1jarq del 9/10 (c_stadio3.py) hanno dato MMLU 61,60 / 62,90 contro C0B 65,20. Regola di
Angelo: una leva pubblicata che rende meno e' un errore nostro finche' non e' applicata nel momento e nel modo del
lavoro. Differenze trovate (dettaglio in ~/tony-forgia/ricerche/JARQ_C_SEQUENZA_1011.md):
  D1 jarq1 partiva dai codici TT2 GIA' allenati end-to-end sulla KL (il lavoro: ospite PTQ da pesi pieni, prima di
     ogni allenamento);  D2 bersaglio bf16 + dW LoRA EB3 (il lavoro: pesi originali);  D3 nu2 = 1e-4 fisso sulla H
     normalizzata (il lavoro: nu = 0,6 su A = [X; nu I], cioe' nu^2/(N·dbar) nella nostra scala);  D4 ordine del ciclo
     (il lavoro: scale -> L -> arresto PER CANALE -> codici; il nostro: scale -> codici -> arresto GLOBALE);
  D5 scale arrotondate a fp16 a ogni giro (il lavoro: FP32 e cast solo alla fine);  D6 righe riportate all'ospite se
     peggiori (non nel lavoro);  D7 ridge 1e-4·diag su fallimento di Cholesky (il lavoro: 1e-4·I se spettro non
     finito, lambda_min <= 1e-10·lambda_max o Cholesky fallita).
Qui D1-D7 sono tolte. Restano dichiarate (non eliminabili senza propagazione sequenziale sul modello intero):
  R1 ingressi X del bf16 (manifesto R2B3, corpus 20/9, N = 196.608) invece di X~ propagati dal modello quantizzato;
  R2 formato: gruppi di 128 contigui nella base ruotata (TQ1_0R_1024) / 256 (TQ1_0), scala fp16, niente act-order.

Varianti, tutte dallo STESSO bersaglio (bf16 puro) e dalla STESSA H:
  a  f_gptq1      GPTQ K=1 fedele: fasterquant di IST-DASLab/gptq, quantizzatore trits (livelli 0, ±max|w| del gruppo,
                  soglia meta'), groupsize = gruppo del formato, blocksize 128, percdamp 0,01, ordine naturale, FP32
  b  f_gptq1jarq  (a) + JARQ Alg. 1 fedele (K = 3, nu = 0,6, tau = 1e-5)
  c  f_bil1jarq   GPTQ bilanciato K=1 (solutori.gptq_bilanciato, griglia di Tied + GPTQ) + JARQ Alg. 1 fedele
Uscita: <out>/<variante>/tensori/<primo piano>.npy + manifesto.json (formato di c0_scrivi.py) + registro.jsonl comune.
uso: jarq_fedele.py [--out DIR] [--solo nome,...] [--prova]   (--prova: aggiunge d = TT2 + JARQ fedele e le misure di D)
"""
import argparse, hashlib, json, os, sys, time
sys.path.insert(0, "@@ROOT:models@@/build/leve/raffina")
import numpy as np
import torch
import comune as CM
import solutori as SO

F64 = torch.float64
N_CALIB = 196_608            # 96 seq x 2048 (cattura del 20/9: H = X^T X / N)
NU = 0.6                     # 2609.38599 §4 e App. A.1/E.4/F.2
TAU = 1e-5                   # App. E.2 eq. 18
RIDGE = 1e-4                 # App. E.2 (lambda, unita' del lavoro)
RCOND = 1e-10
EPS32 = float(torch.finfo(torch.float32).eps)


def bersaglio_bf16_e_H(nome, rotated):
    """bf16 PURO (senza dW LoRA) e H del manifesto, normalizzata come comune.bersaglio_e_H; ritorna anche c = N·dbar."""
    _, H, prov = CM.bersaglio_e_H(nome, rotated)
    from sorgente import bf16_a_f32
    u16, _ = CM._SORG.u16_ordine_gguf(nome)
    W = torch.from_numpy(bf16_a_f32(u16)).double()
    if rotated:
        W = CM.ruota(W)
    return W, H, N_CALIB * prov["diag_media_grezza"], prov


# ─────────────────────────── (a) GPTQ fedele: IST-DASLab/gptq fasterquant + Quantizer(trits=True) ───────────────────────────
def gptq_trits(W, H, ng, percdamp=0.01, blocksize=128, scala_fp16=True):
    R, C = W.shape
    G = C // ng
    Wf = W.float().clone()
    Hf = H.float().clone()
    dead = torch.diag(Hf) == 0
    Hf[dead, dead] = 1
    Wf[:, dead] = 0
    Hf += percdamp * torch.mean(torch.diag(Hf)) * torch.eye(C)
    L = torch.linalg.cholesky(Hf)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(L), upper=True)
    T = torch.zeros(1, R, C, dtype=torch.int8)
    S = torch.zeros(1, R, ng, dtype=F64)
    scale = zero = None
    for i1 in range(0, C, blocksize):
        i2 = min(i1 + blocksize, C)
        W1 = Wf[:, i1:i2].clone()
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            j = i1 + i
            if j % G == 0:                                      # find_params(W[:, j:j+groupsize]) — sym, perchannel
                x = Wf[:, j:j + G]                              # == W1 all'inizio del gruppo (G = blocksize allineati)
                xmin = torch.minimum(x.min(1)[0], torch.zeros(R))
                xmax = torch.maximum(x.max(1)[0], torch.zeros(R))
                xmax = torch.maximum(xmin.abs(), xmax)
                neg = xmin < 0
                xmin[neg] = -xmax[neg]
                tmp = (xmin == 0) & (xmax == 0)
                xmin[tmp] = -1
                xmax[tmp] = 1
                scale = SO.fp16(xmax.double()).float() if scala_fp16 else xmax.clone()   # formato: fp16 (dichiarato)
                zero = -scale * (xmin < 0).float()              # trits: zero = xmin (= -scale o 0)
                S[0, :, j // G] = scale.double()
            w = W1[:, i]
            q = (w > scale / 2).float() * scale + (w < zero / 2).float() * zero
            T[0, :, j] = torch.where(w > scale / 2, 1, torch.where((w < zero / 2) & (zero < 0), -1, 0)).to(torch.int8)
            d = Hinv1[i, i]
            err1 = (w - q) / d
            W1[:, i:] -= err1.unsqueeze(1) @ Hinv1[i, i:].unsqueeze(0)
            Err1[:, i] = err1
        Wf[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    return T, S


# ─────────────────────────── JARQ Algorithm 1 fedele (K = 3, per canale) ───────────────────────────
def _scale_righe(Wt, Ha, T, ng, c, lotto=256):
    """scale per tutte le righe date, a lotti (memoria: [lotto, ng, C] in F64)."""
    out, nrid = [], 0
    for a in range(0, Wt.shape[0], lotto):
        s, n = _scale_riga_lotto(Wt[a:a + lotto], Ha, T[:, a:a + lotto], ng, c)
        out.append(s); nrid += n
    return torch.cat(out, 1), nrid


def _scale_riga_lotto(Wt, Ha, T, ng, c):
    """Alg. 1 righe 5-6, eq. 6, App. E.2: Gram G = B^T B (unita' del lavoro = c·nostra), simmetrizzata, FP32;
    ridge lambda = 1e-4 SOLO se spettro non finito, lambda_min <= rcond·max(|lambda_max|, eps) o Cholesky fallisce."""
    R, C = Wt.shape
    G = C // ng
    Tt = T[0].to(F64).reshape(R, ng, G)
    Hg = Ha.reshape(ng, G, C)
    BH = torch.empty(R, ng, C, dtype=F64)
    for g in range(ng):
        BH[:, g, :] = Tt[:, g, :] @ Hg[g]
    M = (BH.reshape(R, ng, ng, G) * Tt.unsqueeze(1)).sum(-1)          # [R, ng, ng]
    rhs = torch.einsum("rgj,rj->rg", BH, Wt)
    Mp = (c * 0.5 * (M + M.transpose(1, 2))).float()                 # unita' del lavoro, FP32
    bp = (c * rhs).float()
    ev = torch.linalg.eigvalsh(Mp)
    lmax = ev.abs().amax(1).clamp_min(EPS32)
    instab = ~torch.isfinite(ev).all(1) | (ev[:, 0] <= RCOND * lmax)
    Lc, info = torch.linalg.cholesky_ex(Mp)
    instab |= info != 0
    I = torch.eye(ng).expand_as(Mp)
    Mp = torch.where(instab.view(-1, 1, 1), Mp + RIDGE * I, Mp)
    Lc, info = torch.linalg.cholesky_ex(Mp)
    if (info != 0).any():
        raise RuntimeError("JARQ: Cholesky fallita anche con ridge (App. E.2: errore numerico)")
    s = torch.cholesky_solve(bp.unsqueeze(-1), Lc).squeeze(-1).double()
    return s.unsqueeze(0), int(instab.sum())


def _babai(W, Ha, T, S, attive):
    """Alg. 1 righe 9-12, eq. 7-10: gruppi in ordine, proposta di Babai sul fattore triangolare del gruppo
    (R_i^T R_i = A_i^T A_i = H_ii + nu^2 I), arrotondata e tagliata a {-1,0,1}; accettata solo se finita e
    strettamente migliore. Solo righe attive e gruppi con |s| > eps FP32."""
    _, R, C = T.shape
    ng = S.shape[-1]
    G = C // ng
    T = T.clone()
    What = SO.ricostruisci(T, S)
    HE = (W - What) @ Ha
    cambi = 0
    for g in range(ng):
        sl = slice(g * G, (g + 1) * G)
        Hgg = Ha[sl, sl]
        Rch = torch.linalg.cholesky(Hgg).T.contiguous()
        s = S[0, :, g]
        ok = attive & (s.abs() > EPS32)
        if not ok.any():
            continue
        zcur = What[:, sl]
        y = zcur + torch.cholesky_solve(HE[:, sl].T.contiguous(), Rch.T.contiguous()).T
        z = torch.empty_like(y)
        q = torch.empty(R, G, dtype=F64)
        ss = torch.where(ok, s, torch.ones_like(s))
        for l in range(G - 1, -1, -1):
            corr = ((z[:, l + 1:] - y[:, l + 1:]) * Rch[l, l + 1:]).sum(1) / Rch[l, l] if l < G - 1 else 0.0
            v = (y[:, l] - corr) / ss
            q[:, l] = torch.clamp(torch.round(v), -1, 1)
            z[:, l] = q[:, l] * ss
        dn, do = z - y, zcur - y
        e_new = ((dn @ Hgg) * dn).sum(1)
        e_old = ((do @ Hgg) * do).sum(1)
        acc = ok & torch.isfinite(e_new) & (e_new < e_old)
        n = int(acc.sum())
        if n:
            cambi += n
            T[0, acc, sl] = q[acc].to(torch.int8)
            dl = torch.zeros(R, G, dtype=F64)
            dl[acc] = z[acc] - zcur[acc]
            What[acc, sl] = z[acc]
            HE -= dl @ Ha[sl, :]
    return T, cambi


def jarq_fedele(W, H, T0, S0, c, K=3, nu=NU, tau=TAU):
    """Ritorna (T, S fp16 in F64, catena). Obiettivo F = ||X(W-Ŵ)||^2 + nu^2 ||W-Ŵ||^2 = c·[E (H + nu^2/c I) E^T]."""
    R, C = W.shape
    ng = S0.shape[-1]
    Ha = H + (nu * nu / c) * torch.eye(C, dtype=F64)
    T, S = T0.clone(), S0.clone().to(F64)
    attive = torch.ones(R, dtype=torch.bool)
    Lprec = None
    cat = {"ospite": float(SO.obiettivo(W, SO.ricostruisci(T, S), Ha).sum()), "nu2_rel": nu * nu / c}
    for k in range(1, K + 1):
        Snew, nrid = _scale_righe(W[attive], Ha, T[:, attive], ng, c)
        S[:, attive] = Snew
        Lk = SO.obiettivo(W, SO.ricostruisci(T, S), Ha) * c
        if k >= 2:
            d = Lprec - Lk
            stop = attive & ((d / Lprec.abs().clamp_min(EPS32) <= tau) | (d.abs() <= tau))
            attive &= ~stop
        Lprec = Lk
        cat[f"L_scale_{k}"] = float(Lk.sum() / c)
        cat[f"ridge_{k}"] = nrid
        cat[f"attive_{k}"] = int(attive.sum())
        if not attive.any():
            break
        T, cambi = _babai(W, Ha, T, S, attive)
        cat[f"cambi_{k}"] = cambi
        cat[f"L_codici_{k}"] = float(SO.obiettivo(W, SO.ricostruisci(T, S), Ha).sum())
    S = SO.fp16(S)                                              # cast al dtype del formato, solo alla fine
    cat["finale_fp16"] = float(SO.obiettivo(W, SO.ricostruisci(T, S), Ha).sum())
    return T, S, cat


def errH(W, T, S, H):
    return CM.err_pesato(W, SO.ricostruisci(T, S), H)[0]


def normaH(D, H):
    return float(((D @ H) * D).sum()) ** 0.5


VARIANTI = ("f_gptq1", "f_gptq1jarq", "f_bil1jarq")


def una(d, prova=False):
    W, H, c, _ = bersaglio_bf16_e_H(d.nome, d.rotated)
    ng = d.S.shape[-1]
    T1, S1 = d.T[:1].clone(), d.S[:1].clone().double()                 # TT2 (C0B), allenato E2E
    t0 = time.time()
    Ta, Sa = gptq_trits(W, H, ng)
    Tb, Sb, cb = jarq_fedele(W, H, Ta, Sa, c)
    Tc0, Sc0, _ = SO.gptq_bilanciato(W, H, 1, ng)
    Tc, Sc, cc = jarq_fedele(W, H, Tc0, Sc0, c)
    out = {"f_gptq1": (Ta, Sa), "f_gptq1jarq": (Tb, Sb), "f_bil1jarq": (Tc, Sc)}
    m = {"errH_tt2": errH(W, T1, S1, H), "errH_gptq1": errH(W, Ta, Sa, H), "errH_gptq1jarq": errH(W, Tb, Sb, H),
         "errH_bil1": errH(W, Tc0, Sc0, H), "errH_bil1jarq": errH(W, Tc, Sc, H),
         "trit_cambiati_jarq_da_gptq": float((Tb != Ta).double().mean()),
         "trit_cambiati_jarq_da_bil": float((Tc != Tc0).double().mean()),
         "trit_diversi_tt2_gptq1jarq": float((Tb != T1).double().mean()),
         "nu2_rel": cb["nu2_rel"], "catena_b": cb, "catena_c": cc}
    if prova:
        Td, Sd, cd = jarq_fedele(W, H, T1, S1, c)                       # d: JARQ fedele applicata DOPO l'E2E
        WT = SO.ricostruisci(T1, S1)
        D = SO.ricostruisci(Td, Sd) - WT
        ET = W - WT
        nW = normaH(W, H)
        m.update({"errH_tt2jarq": errH(W, Td, Sd, H), "trit_cambiati_jarq_da_tt2": float((Td != T1).double().mean()),
                  "D_H_rel": normaH(D, H) / nW, "ET_H_rel": normaH(ET, H) / nW,
                  "cos_H_ET_D": float(((ET @ H) * D).sum()) / max(normaH(ET, H) * normaH(D, H), 1e-30),
                  "dist_H_tt2_gptq1jarq_rel": normaH(SO.ricostruisci(Tb, Sb) - WT, H) / nW,
                  "dist_H_tt2_bil1jarq_rel": normaH(SO.ricostruisci(Tc, Sc) - WT, H) / nW,
                  "catena_d": cd})
    m["secondi"] = round(time.time() - t0, 1)
    return out, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="@@ROOT:models@@/gguf/tony-tern/OBIETTIVO_C/varianti_fedeli")
    ap.add_argument("--solo", default="")
    ap.add_argument("--prova", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    reg = os.path.join(a.out, "registro_prova.jsonl" if a.prova else "registro.jsonl")
    fatti = {json.loads(l)["nome"] for l in open(reg)} if os.path.exists(reg) else set()
    src = CM.gp.GGUFPlaneSource(CM.R2B3M, layers=list(range(40)))
    nomi = CM.elenco_densi(src)
    if a.solo:
        nomi = [n for n in nomi if n in a.solo.split(",")]
    CM.log(f"jarq_fedele: {len(nomi)} densi, gia' fatti {len(fatti)}, prova={a.prova}")
    mans = {}
    for v in VARIANTI:
        os.makedirs(os.path.join(a.out, v, "tensori"), exist_ok=True)
        mp = os.path.join(a.out, v, "tensori", "manifesto.json")
        mans[v] = json.load(open(mp)) if os.path.exists(mp) else {}
    for i, nome in enumerate(nomi):
        if nome in fatti:
            continue
        d = CM.Denso(src, nome)
        out, m = una(d, a.prova)
        if not a.prova:
            p1 = d.nomi_piani[0]
            for v, (T, S) in out.items():
                b = d.byte_piani(T, S.float())[0]
                td = os.path.join(a.out, v, "tensori")
                np.save(os.path.join(td, p1 + ".npy"), b)
                mans[v][p1] = {"tipo": CM.gp._import_gguf(None).GGMLQuantizationType(d.tid).name,
                               "forma_ggml": [int(d.in_), int(d.out)],
                               "sha256_npy": hashlib.sha256(np.ascontiguousarray(b).tobytes()).hexdigest(),
                               "variante": v, "errH": m["errH_" + v[2:]]}
                mp = os.path.join(td, "manifesto.json")
                json.dump(mans[v], open(mp + ".tmp", "w")); os.replace(mp + ".tmp", mp)
        m["nome"] = nome
        with open(reg, "a") as f:
            f.write(json.dumps(m) + "\n")
        CM.log(f"{i+1}/{len(nomi)} {nome}: errH tt2 {m['errH_tt2']:.4f} | gptq1 {m['errH_gptq1']:.4f} -> +jarq "
               f"{m['errH_gptq1jarq']:.4f} | bil1 {m['errH_bil1']:.4f} -> +jarq {m['errH_bil1jarq']:.4f}"
               + (f" | tt2+jarq {m['errH_tt2jarq']:.4f} |D|H {m['D_H_rel']:.4f} cos(ET,D) {m['cos_H_ET_D']:+.3f}"
                  if a.prova else "") + f", {m['secondi']} s")
    CM.log("fine jarq_fedele")


if __name__ == "__main__":
    main()
