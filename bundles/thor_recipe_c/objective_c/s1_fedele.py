#!/usr/bin/env python3
"""Obiettivo C, RAMO B, stadio S1 FEDELE sui 250 densi (+ i 60 ssm_alpha/beta che condividono attn_norm), SOLO CPU.
11/10/2026. Poi S2 rifatto SOPRA S1 con le tre ricette di varianti_fedeli (a GPTQ K=1, b a+JARQ, c BIL K=1+JARQ).
Nota completa: ~/tony-forgia/ricerche/S1_C_1011.md

Leve S1 a ZERO byte (niente cambia nel motore: tutto piegato nei pesi o nelle norme F32):
  (3) OSAQ, arXiv 2605.04738, §5 eq. 8-14: W' = W + beta N, N = autovettori di H con gli autovalori |lambda| piu' piccoli
      la cui energia cumulata raggiunge gamma·sum|lambda| (eq. 8); per riga b_i = -A_i^-1 rho_i (eq. 12-14) con
      A_i = sum_j s_ij n_j n_j^T + mu1 I + mu2 v v^T, rho_i = sum_j s_ij W_ij n_j, v = N 1, s_ij = softmax_j(|W_ij|/tau)
      (eq. 10, sulla W di partenza). Iperparametri NON specificati nel lavoro: centro della griglia di Fig. 5
      (gamma 1e-4, tau 0,1, mu1 = mu2 = 1e-3). Ordine (App. B): OSAQ PRIMA di AWQ/GPTQ/rotazioni.
      Non e' un'identita': cambia la perdita di calibrazione di tr(dW H dW^T) = sum_i b_i^T diag(lambda_K) b_i (misurato).
  (2) Riequilibrio dei canali, criterio AWQ arXiv 2306.00978 (weight-only; SmoothQuant 2211.10438 e' W8A8 e non
      guarda l'errore dei pesi quantizzati): eq. 4-5, codice ufficiale llm-awq auto_scale.py _search_module_scale:
      n_grid 20, ratio = i/20 (i = 0..19), s = x_mag^ratio .clamp(1e-4), s /= sqrt(max·min), Q = RTN del formato,
      perdita = MSE dell'uscita, il migliore (strettamente minore) vince; piega scale_ln_fcs (norma /= s, fc *= s) e
      scale_fc_fc (up righe /= s, down colonne *= s). Gruppi:
        attn_norm -> {attn_qkv, attn_gate, ssm_alpha, ssm_beta} (GDN) | {attn_q, attn_k, attn_v} (full attention)
        ffn_up_shexp -> ffn_down_shexp  (candidati: griglia AWQ U {regola L2 di 2609.21450 eq. 17,
            lambda_k = sqrt(||X_k|| / ||W_k||), che quel lavoro applica proprio all'ingresso di down}; decide la
            perdita AWQ eq. 4, la stessa per tutti i candidati)
      ESCLUSI (dimostrato in test e nota): post_attention_norm (alimenta anche i 256 esperti di C0B, fermi per regola:
      l'unica scala che li lascia rappresentabili e' costante su ogni blocco da 1024 e di potenza di 2, e quella non
      cambia il problema di quantizzazione: leva nulla); v -> o (llm-awq lo fa solo se le forme coincidono: GQA 2/16 no).
      Deviazioni dichiarate: D-A1 x_mag = sqrt(diag H) (RMS) invece di mean|x| (le attivazioni non sono salvate: solo
      H = X^T X / N); D-A2 perdita = somma delle MSE d'uscita dei lineari del gruppo (tr(E H E^T)) invece dell'uscita del
      blocco (attenzione/GDN non lineari: servirebbe X); D-A3 niente auto_clip (il clipping e' del quantizzatore: lo
      fa S2). Q = RTN ternario min-max del formato (stesso Quantizer(trits=True) della variante a), nella base ruotata.
      Scala effettiva s_eff = gamma_f32 / fl32(gamma_f32 / s): l'identita' vale ESATTA con la norma F32 scritta.
  (4) Segni Hadamard multipli, arXiv 2609.21450 §4.6/§5: piegabili SOLO nel gruppo attn_norm (D = ±1 nella norma e
      nelle colonne: segni effettivi S·D). Criterio del lavoro = perplexity globale (RTN, poi le 3 migliori con
      GPTAQ): GPU. Qui solo il filtro CPU (variante Thor dichiarata): N = 10 semi Rademacher, UNO per tutta la
      categoria (stesso vettore in tutti i 40 strati, come nel lavoro), perdita relativa RTN sommata sugli strati,
      poi le 3 migliori con GPTQ trits; si sceglie fra {D = 1} U 10 semi.
Hessiana: dopo S1 il quantizzatore vede X' = X diag(s)^-1, quindi H' = diag(s)^-1 H diag(s)^-1 (con D = diag(s)^-1
la forma D H D); l'errore e' invariante: tr(E' H' E'^T) = tr(E H E^T) con E' = E diag(s). errH si riporta nella base
ORIGINALE contro il bf16 PURO (senza OSAQ), stessa definizione di varianti_fedeli.
uso: s1_fedele.py [--out DIR] [--strati 0,1] [--senza-s2] [--senza-segni]
"""
import argparse, hashlib, json, math, os, sys, time
sys.path.insert(0, "@@ROOT:models@@/gguf/tony-tern/OBIETTIVO_C/codice")
import numpy as np
import torch
import jarq_fedele as JF

CM, SO = JF.CM, JF.SO
F64 = torch.float64
C0B = "@@ROOT:models@@/gguf/tony-tern/OBIETTIVO_C/C0B/Thor-T-C0B.gguf"
HATTN = "@@ROOT:models@@/gguf/tony-tern/hessiane-attn/H_{:02d}.npy"
OSAQ_GAMMA, OSAQ_TAU, OSAQ_MU1, OSAQ_MU2 = 1e-4, 0.1, 1e-3, 1e-3
AWQ_NGRID = 20
N_SEMI = 10
T_TQ1_0, T_TQ1_0R = 34, 45


# ─────────────────────────── dati ───────────────────────────
def sorgente():
    if CM._SORG is None:
        from sorgente import SorgenteHF
        CM._SORG = SorgenteHF()
        CM._MANIF = json.load(open(CM.MANIF_R2B3))
    return CM._SORG


def W_bf16(nome):
    from sorgente import bf16_a_f32
    u16, _ = sorgente().u16_ordine_gguf(nome)
    return torch.from_numpy(bf16_a_f32(u16)).double()


def H_grezza(f):
    """H = X^T X / N (base originale, ordine GGUF), simmetrizzata; colonne morte -> 1 (come comune.bersaglio_e_H)."""
    H = torch.from_numpy(np.load(f)).double()
    H = 0.5 * (H + H.T)
    d = torch.diagonal(H).clone()
    m = d <= 0
    if m.any():
        H[m, m] = 1.0
    return H


def normalizza(H):
    d = torch.diagonal(H)
    return H / d.clamp_min(1e-30).mean(), float(d.clamp_min(1e-30).mean())


def file_H(nome):
    sorgente()
    L = int(nome.split(".")[1])
    if nome.endswith((".ssm_alpha.weight", ".ssm_beta.weight")):
        return HATTN.format(L)
    return CM._MANIF[nome.replace(".weight", "3.weight")]["hessiana"]["file"]


def gruppi_strato(L):
    full = (L % 4 == 3)
    b = f"blk.{L}."
    attn = [b + x + ".weight" for x in (("attn_q", "attn_k", "attn_v") if full else
                                         ("attn_qkv", "attn_gate", "ssm_alpha", "ssm_beta"))]
    altri = [b + x + ".weight" for x in (("attn_output",) if full else ("ssm_out",))]
    shexp = [b + x + ".weight" for x in ("ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp")]
    return attn, altri, shexp


def ruotato(nome):
    return not nome.endswith("ffn_down_shexp.weight")


# ─────────────────────────── rotazione con segni a scelta (leva 4) ───────────────────────────
def segni_motore():
    return torch.from_numpy(CM.gp.sign1024().astype(np.float64))


def ruota_s(W, sgn):
    C = W.shape[-1]
    x = (W.reshape(*W.shape[:-1], C // 1024, 1024) * sgn.to(W.dtype)).reshape(W.shape)
    return CM.fwht(x, 1024)


def deruota_s(Wp, sgn):
    y = CM.fwht(Wp, 1024)
    C = Wp.shape[-1]
    return (y.reshape(*Wp.shape[:-1], C // 1024, 1024) * sgn.to(Wp.dtype)).reshape(Wp.shape)


# ─────────────────────────── Q = RTN ternario min-max del formato (Quantizer(trits=True)) ───────────────────────────
def rtn_trits(Wst, G):
    R, C = Wst.shape
    x = Wst.reshape(R, C // G, G)
    xmin = torch.minimum(x.min(-1)[0], torch.zeros(1, dtype=x.dtype))
    xmax = torch.maximum(x.max(-1)[0], torch.zeros(1, dtype=x.dtype))
    xmax = torch.maximum(xmin.abs(), xmax)
    neg = xmin < 0
    xmin = torch.where(neg, -xmax, xmin)
    tmp = (xmin == 0) & (xmax == 0)
    xmin = torch.where(tmp, -torch.ones_like(xmin), xmin)
    xmax = torch.where(tmp, torch.ones_like(xmax), xmax)
    scale = SO.fp16(xmax).unsqueeze(-1)
    zero = (-scale * (xmin < 0).to(x.dtype).unsqueeze(-1))
    q = (x > scale / 2).to(x.dtype) * scale + (x < zero / 2).to(x.dtype) * zero
    return q.reshape(R, C)


def Q_formato(W, rot, sgn=None):
    """W nella base ORIGINALE -> dequantizzato RTN del formato, di nuovo nella base originale."""
    if rot:
        sgn = segni_motore() if sgn is None else sgn
        return deruota_s(rtn_trits(ruota_s(W, sgn), 128), sgn)
    return rtn_trits(W, 256)


def trHE(E, H):
    return float(((E @ H) * E).sum())


# ─────────────────────────── (3) OSAQ ───────────────────────────
def osaq(W, H, gamma=OSAQ_GAMMA, tau=OSAQ_TAU, mu1=OSAQ_MU1, mu2=OSAQ_MU2, lotto=None):
    lam, V = torch.linalg.eigh(H)
    a = lam.abs()
    ordn = torch.argsort(a)
    cum = torch.cumsum(a[ordn], 0)
    tot = float(a.sum())
    K = int((cum < gamma * tot).sum()) + 1                        # min k: sum_{i<=k} |lambda_i| >= gamma sum |lambda|
    idx = ordn[:K]
    Nn = V[:, idx].T.contiguous()                                 # [K, C], righe = autovettori
    lamK = lam[idx]
    s = torch.softmax(W.abs() / tau, dim=1)                       # eq. 10, per riga, sulla W di partenza
    v = Nn.sum(1)                                                 # N 1_N
    rho = (s * W) @ Nn.T                                          # [M, K]
    M, C = W.shape
    if lotto is None:
        lotto = max(1, min(M, int(4e8 // max(1, K * C * 8))))
    B = torch.empty(M, K, dtype=F64)
    I = torch.eye(K, dtype=F64)
    vv = torch.outer(v, v)
    for a0 in range(0, M, lotto):
        sq = s[a0:a0 + lotto].sqrt()
        X = Nn.unsqueeze(0) * sq.unsqueeze(1)                     # [b, K, C]
        A = X @ X.transpose(1, 2) + mu1 * I + mu2 * vv            # eq. 13
        Lc = torch.linalg.cholesky(A)                             # Remark 1: A >= mu1 I > 0
        B[a0:a0 + lotto] = -torch.cholesky_solve(rho[a0:a0 + lotto].unsqueeze(-1), Lc).squeeze(-1)   # eq. 14
    dW = B @ Nn
    den = trHE(W, H)
    info = {"K": K, "gamma": gamma, "tau": tau, "mu1": mu1, "mu2": mu2, "lam_K_max": float(a[ordn[K - 1]]),
            "lam_max": float(a.max()), "dL_rel": trHE(dW, H) / den,
            "dL_rel_da_beta": float((B * B * lamK).sum()) / den,
            "linf_prima": float(W.abs().amax(1).mean()), "linf_dopo": float((W + dW).abs().amax(1).mean()),
            "dW_fro_rel": float(dW.norm() / W.norm())}
    return dW, info, (Nn, s, v, B)


# ─────────────────────────── (2) AWQ: ricerca della scala (llm-awq _search_module_scale) ───────────────────────────
def awq_cerca(Wlist, H, xmag, rot, ngrid=AWQ_NGRID):
    Wc = torch.cat(Wlist, 0)
    storia, best, best_s, best_r = [], float("inf"), None, None
    for i in range(ngrid):
        r = i * 1 / ngrid
        sc = xmag.pow(r).clamp(min=1e-4).view(-1)
        sc = sc / (sc.max() * sc.min()).sqrt()
        Wq = Q_formato(Wc * sc, rot) / sc
        loss = trHE(Wq - Wc, H)
        storia.append(loss)
        if loss < best:
            best, best_s, best_r = loss, sc.clone(), r
    return best_s, best_r, storia


def l2_regola(Wd, H):
    """2609.21450 eq. 17 (riferimento): lambda_k = sqrt(||X_k||_2 / ||W_k||_2) sull'ingresso di down."""
    return (torch.diagonal(H).clamp_min(0).sqrt() / Wd.norm(dim=0).clamp_min(1e-30)).sqrt()


# ─────────────────────────── gamma della norma (C0B, F32) ───────────────────────────
_C0B_R = None


def tensore_c0b(nome):
    global _C0B_R
    if _C0B_R is None:
        sys.path.insert(0, "@@ROOT:models@@/build/thor-rexppiani2/gguf-py")
        from gguf import GGUFReader
        _C0B_R = {t.name: t for t in GGUFReader(C0B).tensors}
    return _C0B_R[nome]


def gamma_c0b(L):
    return np.array(tensore_c0b(f"blk.{L}.attn_norm.weight").data, dtype=np.float32).copy()


# ─────────────────────────── S1 di uno strato ───────────────────────────
def s1_strato(L, D=None, fissi=None, solo_attn=False):
    """Ritorna dict con: W (bf16 orig), Wt (trasformati, base orig), Ht (H grezza trasformata), H (grezza orig),
    misc: s_eff attn, s down, gnew (F32), info OSAQ/AWQ, test d'identita'."""
    attn, altri, shexp = gruppi_strato(L)
    nomi = attn if solo_attn else attn + altri + shexp
    W = {n: W_bf16(n) for n in nomi}
    Hf = {n: file_H(n) for n in nomi}
    Hc = {}
    for f in set(Hf.values()):
        Hc[f] = H_grezza(f)
    assert len({Hf[n] for n in attn}) == 1, "il gruppo attn_norm deve condividere la stessa H"
    info = {"strato": L, "osaq": {}, "awq": {}, "test": {}}
    # (3) OSAQ per tensore (prima di tutto, App. B)
    Wo = {}
    for n in nomi:
        dW, inf_, _ = osaq(W[n], Hc[Hf[n]])
        Wo[n] = W[n] + dW
        info["osaq"][n] = inf_
    # (2) AWQ attn_norm
    Ha = Hc[Hf[attn[0]]]
    xa = torch.diagonal(Ha).clamp_min(0).sqrt()
    if fissi is None:
        s_a, r_a, st_a = awq_cerca([Wo[n] for n in attn], Ha, xa, True)
    else:
        s_a, r_a, st_a = fissi["s_a"], fissi["r_a"], fissi["st_a"]
    gam = gamma_c0b(L)
    gnew = (torch.from_numpy(gam).double() / s_a).float().numpy()
    if D is not None:
        gnew = (gnew * D.repeat(gnew.size // 1024).float().numpy()).astype(np.float32)   # (4) segni nella norma, esatti in F32
    s_eff = torch.from_numpy(gam).double() / torch.from_numpy(gnew).double()   # include D (s_eff = s·D)
    info["awq"]["attn_norm"] = {"ratio": r_a, "perdita_rtn": st_a, "guadagno_rtn": 1 - min(st_a) / st_a[0],
                                "s_min": float(s_a.min()), "s_max": float(s_a.max()),
                                "rel_s_eff_s": float(((s_eff.abs() - s_a) / s_a).abs().max())}
    if solo_attn:                                                 # passo 2 (segni): basta il gruppo attn_norm
        Wt = {n: Wo[n] * s_eff for n in attn}
        Ht = {n: Hc[Hf[n]] / torch.outer(s_eff, s_eff) for n in attn}
        return {"Wt": Wt, "Ht": Ht, "attn": attn, "info": info}
    # (2) AWQ up -> down
    nd, nu = shexp[2], shexp[1]
    Hd = Hc[Hf[nd]]
    xd = torch.diagonal(Hd).clamp_min(0).sqrt()
    lam = l2_regola(W[nd], Hd)
    l2_loss = trHE(Q_formato(Wo[nd] * lam, False) / lam - Wo[nd], Hd)
    if fissi is None:
        s_d, r_d, st_d = awq_cerca([Wo[nd]], Hd, xd, False)
        if l2_loss < min(st_d):                                   # candidato in piu': regola L2 di 2609.21450 eq. 17,
            s_d, r_d = lam.clone(), "L2-2609"                     # giudicato con la stessa perdita AWQ (eq. 4)
    else:
        s_d, r_d, st_d = fissi["s_d"], fissi["r_d"], fissi["st_d"]
    info["awq"]["up_down"] = {"ratio": r_d, "perdita_rtn": st_d, "guadagno_rtn": 1 - min(min(st_d), l2_loss) / st_d[0],
                              "s_min": float(s_d.min()), "s_max": float(s_d.max()),
                              "l2_2609_perdita_rtn": l2_loss, "l2_2609_guadagno_rtn": 1 - l2_loss / st_d[0]}
    # trasformati
    Wt, Ht, cols, rows = {}, {}, {}, {}
    for n in nomi:
        Wt[n], Ht[n] = Wo[n], Hc[Hf[n]]
    for n in attn:
        Wt[n] = Wo[n] * s_eff
        Ht[n] = Hc[Hf[n]] / torch.outer(s_eff, s_eff)              # H' = diag(s)^-1 H diag(s)^-1
        cols[n] = s_eff
    Wt[nd] = Wo[nd] * s_d
    Ht[nd] = Hd / torch.outer(s_d, s_d)
    cols[nd] = s_d
    Wt[nu] = Wo[nu] / s_d.unsqueeze(1)
    rows[nu] = s_d
    # test d'identita' fp64 (uscita del blocco invariata, prima della quantizzazione)
    g = torch.Generator().manual_seed(1000 + L)
    xh = torch.randn(64, Wo[attn[0]].shape[1], generator=g, dtype=F64)
    gam64, gn64 = torch.from_numpy(gam).double(), torch.from_numpy(gnew).double()
    y0 = torch.cat([(xh * gam64) @ Wo[n].T for n in attn], 1)
    y1 = torch.cat([(xh * gn64) @ Wt[n].T for n in attn], 1)
    info["test"]["identita_attn_rel"] = float((y0 - y1).abs().max() / y0.abs().max())
    x = torch.randn(64, Wo[nu].shape[1], generator=g, dtype=F64) * gam64.abs().mean()
    ng_, nu_ = shexp[0], shexp[1]
    sw0 = (torch.nn.functional.silu(x @ Wo[ng_].T) * (x @ Wo[nu_].T)) @ Wo[nd].T
    sw1 = (torch.nn.functional.silu(x @ Wt[ng_].T) * (x @ Wt[nu_].T)) @ Wt[nd].T
    info["test"]["identita_shexp_rel"] = float((sw0 - sw1).abs().max() / sw0.abs().max())
    # invarianza dell'errore sotto H' (con un errore qualsiasi)
    E = torch.randn(8, Wo[attn[0]].shape[1], generator=g, dtype=F64)
    info["test"]["invarianza_H_rel"] = abs(trHE(E * s_eff, Ht[attn[0]]) - trHE(E, Ha)) / trHE(E, Ha)
    return {"W": W, "Wo": Wo, "Wt": Wt, "Ht": Ht, "H": {n: Hc[Hf[n]] for n in nomi}, "cols": cols, "rows": rows,
            "gnew": gnew, "s_attn": s_eff, "s_down": s_d, "info": info, "attn": attn,
            "fissi": {"s_a": s_a, "r_a": r_a, "st_a": st_a, "s_d": s_d, "r_d": r_d, "st_d": st_d}}


def a_originale(n, What_t, S):
    if n in S["cols"]:
        What_t = What_t / S["cols"][n]
    if n in S["rows"]:
        What_t = What_t * S["rows"][n].unsqueeze(1)
    return What_t


# ─────────────────────────── (4) filtro CPU dei semi di segno ───────────────────────────
def perdita_segni(S, sgn, gptq=False):
    tot = 0.0
    if True:
        for n in S["attn"]:
            Wt, Ht = S["Wt"][n], S["Ht"][n]
            if gptq:
                Wr = ruota_s(Wt, sgn)
                Hr = ruota_s(ruota_s(Ht, sgn).T.contiguous(), sgn).T.contiguous()
                Hr = 0.5 * (Hr + Hr.T)
                Hn, _ = normalizza(Hr)
                T, Sc = JF.gptq_trits(Wr, Hn, Wt.shape[1] // 128)
                Wq = deruota_s(SO.ricostruisci(T, Sc), sgn)
            else:
                Wq = Q_formato(Wt, True, sgn)
            tot += trHE(Wq - Wt, Ht) / trHE(Wt, Ht)
    return tot


# ─────────────────────────── S2 (ricette di varianti_fedeli) ───────────────────────────
def s2_tensore(n, S, W0):
    Wt, Ht = S["Wt"][n], S["Ht"][n]
    rot = ruotato(n)
    if rot:
        Wr, Hr = CM.ruota(Wt), CM.ruota_H(Ht)
    else:
        Wr, Hr = Wt.clone(), Ht.clone()
    Hr = 0.5 * (Hr + Hr.T)
    Hn, dbar = normalizza(Hr)
    c = JF.N_CALIB * dbar
    G = 128 if rot else 256
    ng = Wr.shape[1] // G
    t0 = time.time()
    Ta, Sa = JF.gptq_trits(Wr, Hn, ng)
    Tb, Sb, cb = JF.jarq_fedele(Wr, Hn, Ta, Sa, c)
    Tc0, Sc0, _ = SO.gptq_bilanciato(Wr, Hn, 1, ng)
    Tc, Sc, cc = JF.jarq_fedele(Wr, Hn, Tc0, Sc0, c)
    Hon, _ = normalizza(S["H"][n])
    out, m = {}, {"nome": n}
    for v, (T, Sx) in {"a": (Ta, Sa), "b": (Tb, Sb), "c": (Tc, Sc), "bil1": (Tc0, Sc0)}.items():
        Wq = SO.ricostruisci(T, Sx)
        Wq_t = CM.deruota(Wq) if rot else Wq
        Wq_o = a_originale(n, Wq_t, S)
        m[f"errH_{v}"] = CM.err_pesato(W0, Wq_o, Hon)[0]                       # base ORIGINALE, bf16 puro
        m[f"errH_{v}_base_s1"] = CM.err_pesato(Wr, Wq, Hn)[0]                 # nella base trasformata, contro W'
        if v != "bil1":
            out[v] = (T, Sx)
    m["errH_osaq_solo"] = CM.err_pesato(W0, S["Wo"][n], Hon)[0]
    m["nu2_rel"] = cb["nu2_rel"]
    m["trit_cambiati_jarq_da_gptq"] = float((Tb != Ta).double().mean())
    m["trit_cambiati_jarq_da_bil"] = float((Tc != Tc0).double().mean())
    m["secondi"] = round(time.time() - t0, 1)
    return out, m


def scrivi(out_dir, v, n, T, Sx, tid, forma, extra):
    td = os.path.join(out_dir, v, "tensori")
    os.makedirs(td, exist_ok=True)
    b = CM.gp.encode_rows(T[0].numpy().astype(np.int8), Sx[0].to(torch.float16).numpy(), tid)
    np.save(os.path.join(td, n + ".npy"), b)
    mp = os.path.join(td, "manifesto.json")
    man = json.load(open(mp)) if os.path.exists(mp) else {}
    man[n] = dict({"tipo": "TQ1_0R_1024" if tid == T_TQ1_0R else "TQ1_0", "forma_ggml": forma,
                   "sha256_npy": hashlib.sha256(np.ascontiguousarray(b).tobytes()).hexdigest(), "variante": v}, **extra)
    json.dump(man, open(mp + ".tmp", "w")); os.replace(mp + ".tmp", mp)


def scrivi_norma(out_dir, L, gnew, info):
    td = os.path.join(out_dir, "norme")
    os.makedirs(td, exist_ok=True)
    n = f"blk.{L}.attn_norm.weight"
    b = np.ascontiguousarray(gnew.astype(np.float32)).view(np.uint8)
    np.save(os.path.join(td, n + ".npy"), b)
    mp = os.path.join(td, "manifesto_norme.json")
    man = json.load(open(mp)) if os.path.exists(mp) else {}
    man[n] = {"tipo": "F32", "forma_ggml": [int(gnew.size)], "sha256_npy": hashlib.sha256(b.tobytes()).hexdigest(),
              "piega": "S1 (2) AWQ attn_norm [+ (4) segni]: gamma' = fl32(gamma_C0B / s)", "awq": info}
    json.dump(man, open(mp + ".tmp", "w")); os.replace(mp + ".tmp", mp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="@@ROOT:models@@/gguf/tony-tern/OBIETTIVO_C/varianti_s1")
    ap.add_argument("--strati", default="")
    ap.add_argument("--senza-s2", action="store_true")
    ap.add_argument("--senza-segni", action="store_true")
    ap.add_argument("--solo-osaq", action="store_true",
                    help="isola la leva 3: niente AWQ ne' segni (s = 1, norme invariate), alpha/beta di C0B tenuti")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    strati = [int(x) for x in a.strati.split(",")] if a.strati else list(range(40))
    reg_s1 = os.path.join(a.out, "registro_s1.jsonl")
    reg = os.path.join(a.out, "registro.jsonl")
    # ── passo 1: S1 di ogni strato (OSAQ + ricerca AWQ) + filtro RTN dei semi di segno, uno strato alla volta ──
    CM.log(f"S1: strati {strati}")
    base = segni_motore()
    cand = {"motore": base}
    for k in range(N_SEMI):
        g = torch.Generator().manual_seed(21450 + k)
        cand[f"seme{k}"] = base * (torch.randint(0, 2, (1024,), generator=g) * 2 - 1).double()
    fissi, info1, rtn = {}, {}, {k: 0.0 for k in cand}
    for L in strati:
        t0 = time.time()
        if a.solo_osaq:
            uno = {"s_a": torch.ones(2048, dtype=F64), "r_a": "spento", "st_a": [1.0],
                   "s_d": torch.ones(512, dtype=F64), "r_d": "spento", "st_d": [1.0]}
            S = s1_strato(L, fissi=uno)
        else:
            S = s1_strato(L)
        fissi[L], info1[L] = S["fissi"], S["info"]
        if not a.senza_segni:
            for k, v in cand.items():
                rtn[k] += perdita_segni(S, v)
        i = S["info"]
        CM.log(f"S1 strato {L}: AWQ attn ratio {i['awq']['attn_norm']['ratio']} guad. RTN "
               f"{i['awq']['attn_norm']['guadagno_rtn']*100:.2f} % | up->down ratio {i['awq']['up_down']['ratio']} "
               f"{i['awq']['up_down']['guadagno_rtn']*100:.2f} % (L2 2609: {i['awq']['up_down']['l2_2609_guadagno_rtn']*100:.2f} %)"
               f" | OSAQ K {[v['K'] for v in i['osaq'].values()]} | identita' {i['test']['identita_attn_rel']:.1e} "
               f"{i['test']['identita_shexp_rel']:.1e} H {i['test']['invarianza_H_rel']:.1e} | {time.time()-t0:.0f} s")
        del S
    scelta, D = {"seme": None}, None
    if a.solo_osaq:
        a.senza_segni = True
    if not a.senza_segni:
        top = sorted(rtn, key=rtn.get)[:3]
        gq = {k: 0.0 for k in set(top) | {"motore"}}
        for L in strati:                                            # passo 2: le 3 migliori (+ motore) con GPTQ
            S = s1_strato(L, fissi=fissi[L], solo_attn=True)
            for k in gq:
                gq[k] += perdita_segni(S, cand[k], gptq=True)
            del S
        best = min(gq, key=gq.get)
        scelta = {"rtn": rtn, "top3": top, "gptq": gq, "scelto": best,
                  "criterio": "Thor CPU: somma delle perdite relative per strato (RTN, poi GPTQ trits); fedele = PPL (GPU)"}
        CM.log(f"segni: RTN {json.dumps({k: round(v, 6) for k, v in rtn.items()})}; GPTQ {json.dumps(gq)}; scelto {best}")
        if best != "motore":
            D = cand[best] / base
            scelta["D"] = D.tolist()
    json.dump(scelta, open(os.path.join(a.out, "segni.json"), "w"))
    with open(reg_s1, "w") as f:
        for L in strati:
            f.write(json.dumps(info1[L]) + "\n")
    if a.senza_s2:
        return
    # ── passo 3: S1 definitivo (con i segni scelti) + norme F32 + S2 sopra S1 ──
    fatti = {json.loads(l)["nome"] for l in open(reg)} if os.path.exists(reg) else set()
    reg_t = os.path.join(a.out, "registro_s1_definitivo.jsonl")
    for L in strati:
        S = s1_strato(L, D=D, fissi=fissi[L])
        if a.solo_osaq:
            assert np.array_equal(S["gnew"], gamma_c0b(L)), "solo OSAQ: la norma deve restare quella di C0B al bit"
        if not a.solo_osaq:
            scrivi_norma(a.out, L, S["gnew"], {"ratio": S["info"]["awq"]["attn_norm"]["ratio"], "segni": scelta.get("scelto")})
        os.makedirs(os.path.join(a.out, "norme"), exist_ok=True)
        np.savez(os.path.join(a.out, "norme", f"s1_blk{L}.npz"), s_attn_eff=S["s_attn"].numpy(), s_down=S["s_down"].numpy())
        with open(reg_t, "a") as f:
            f.write(json.dumps({"strato": L, "test": S["info"]["test"]}) + "\n")
        for n in S["Wt"]:
            if n in fatti or (a.solo_osaq and n.endswith((".ssm_alpha.weight", ".ssm_beta.weight"))):
                continue
            t = tensore_c0b(n)
            tid = int(t.tensor_type)
            forma = [int(x) for x in t.shape]
            out, m = s2_tensore(n, S, S["W"][n])
            for v, (T, Sx) in out.items():
                scrivi(a.out, v, n, T, Sx, tid, forma, {"errH_originale": m[f"errH_{v}"]})
            with open(reg, "a") as f:
                f.write(json.dumps(m) + "\n")
            CM.log(f"S2 {n}: errH(orig) a {m['errH_a']:.4f} b {m['errH_b']:.4f} c {m['errH_c']:.4f} "
                   f"(bil1 {m['errH_bil1']:.4f}, OSAQ solo {m['errH_osaq_solo']:.5f}), {m['secondi']} s")
        del S
    CM.log("fine s1_fedele")


if __name__ == "__main__":
    main()
