#!/usr/bin/env python3
"""Obiettivo C / C0T (9/10, ordine di Angelo: TUTTO ternario, scale simmetriche {-s,0,+s}): rende ternari i ~91 MB che C0
lasciava in F32. SOLO CPU, fp64. Ogni tensore -> <out>/tensori/<nome>.npy (byte del tipo ternario) + voce del manifesto
(tipo, forma_ggml memorizzata, sha256, misure, e 'dol' = [ne0, ne1, modo] per i tensori che il motore dequantizza al
caricamento). Registro per tensore: <out>/registro.jsonl. Riprende: salta i .npy gia' scritti.

  norme RMSNorm (attn_norm, post_attention_norm, ssm_norm, output_norm, nextn.enorm/hnorm): PIEGATE nei consumatori
      (c0t_piega: bersaglio Q_C0 diag(g), metrica H senza guadagno, vie J/F scelte per errore) -> la norma diventa il
      vettore di UNI, ternario ESATTO (trit +1, scala 1.0). Per confronto si misura anche la via 'diretta' (g ternario).
      Gli esperti instradati (ffn_gate/up_exps 0-39) si piegano sulla GPU (c0t_piega_esperti.py), qui solo blk.40.
  attn_q_norm / attn_k_norm: la piega del prodotto g_q*g_k su k e' esatta sulle dimensioni non ruotate e sulle coppie
      IMROPE solo se g_q e' uguale nella coppia (T3/T3b): per strato si sceglie la via con l'errore dei logit minore
      fra (i) q=1, k=ternario(g_q*g_k) e (ii) g_q, g_k ternari.
  router ffn_gate_inp e ffn_gate_inp_shexp: F32 piegati per g_post e ternari (TQ1_0R_1024, H dell'ingresso FFN);
      si misura la concordanza dei top-8 col router F32 su ingressi gaussiani con la covarianza vera (surrogato).
  ssm_a, ssm_dt.bias (32 per strato): ternario con UNA scala per tensore; codici scelti per testa (9 combinazioni) e scale
      su griglia fp16 per minimizzare l'errore sul DECADIMENTO exp(A softplus(alpha+dt)) (alpha ~ N(0, W H W^T));
      dominio di A lineare o A_log (= il parametro originale di HF; il motore fa -exp al caricamento): vince l'errore minore.
  ssm_conv1d: memorizzato trasposto [8192 canali, 4 tap], TQ1_0 (scala ogni 256 canali per tap), ternario pesato per
      la varianza del canale d'ingresso (diag(W_qkv H W_qkv^T)); confronto con TQ1_0R_1024 (rotazione lungo i canali).
  nextn.shared_head_norm: la testa piegata per g_out vuole g_sh/g_out (T5, esatto) -> ternario diretto (solo MTP).
"""
import argparse, hashlib, json, os, sys, time
sys.path.insert(0, "@@ROOT:models@@/gguf/tony-tern/OBIETTIVO_C/codice")
import numpy as np
import torch
import c0t_piega as P
from c0t_ternario_vettori import ternario, ternario_gruppo
CM, gp, TQ = P.CM, P.gp, P.TQ
from gguf import GGUFReader

TT = "@@ROOT:models@@/gguf/tony-tern"
H_ATTN, H_FFN = f"{TT}/hessiane-attn/H_{{:02d}}.npy", f"{TT}/hessiane/H_{{:02d}}.npy"
H_SSMOUT, H_OUT = "@@ROOT:depot@@/tutto_ternario/hessiane/ssm_out/H_{:02d}.npy", f"{TT}/hessiane-output/H.npy"
TID = {"TQ1_0": 34, "TQ1_0R_1024": 45}
rng = np.random.default_rng(20261009)


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


class Uscita:
    def __init__(self, out):
        self.td = os.path.join(out, "tensori"); os.makedirs(self.td, exist_ok=True)
        self.mp = os.path.join(self.td, "manifesto_c0t.json")
        self.man = json.load(open(self.mp)) if os.path.exists(self.mp) else {}
        self.reg = open(os.path.join(out, "registro.jsonl"), "a")

    def fatto(self, nome):
        return nome in self.man and os.path.exists(os.path.join(self.td, nome + ".npy"))

    def scrivi(self, nome, arr, tipo, forma, misure, dol=None):
        arr = np.ascontiguousarray(arr, dtype=np.uint8)
        np.save(os.path.join(self.td, nome + ".npy"), arr)
        v = {"tipo": tipo, "forma_ggml": list(map(int, forma)), "sha256_npy": hashlib.sha256(arr.tobytes()).hexdigest(),
             "misure": misure}
        if dol:
            v["dol"] = list(dol)
        self.man[nome] = v
        json.dump(self.man, open(self.mp + ".tmp", "w"), indent=1); os.replace(self.mp + ".tmp", self.mp)
        self.reg.write(json.dumps({"tensore": nome, "tipo": tipo, "forma": v["forma_ggml"], "dol": dol, **misure}) + "\n")
        self.reg.flush()
        log(nome, tipo, json.dumps({k: (round(x, 5) if isinstance(x, float) else x) for k, x in misure.items()})[:300])


def tq1_0_vettore(q, s_per_gruppo):
    """trit int8 [N] (N%256==0) + scala per gruppo da 256 -> byte TQ1_0 [N/256*54] (1 riga)."""
    N = len(q)
    S = np.repeat(np.asarray(s_per_gruppo, np.float64), 1)[None, :].astype(np.float16)
    return gp.encode_rows(q.reshape(1, N).astype(np.int8), S, TID["TQ1_0"])


def vettore_dol(U, nome, v_q, s_g, n_vero, modo, misure):
    """scrive un vettore ternario memorizzato in TQ1_0 (riempito a 256 con zeri) da dequantizzare al caricamento."""
    N = ((n_vero + 255) // 256) * 256
    q = np.zeros(N, np.int8); q[:len(v_q)] = v_q
    s = np.zeros(N // 256); s[:len(s_g)] = s_g
    b = tq1_0_vettore(q, s)
    # verifica indipendente col dequantizzatore di gguf-py (TQ1_0 standard)
    import gguf
    dq = gguf.quants.dequantize(b.reshape(1, -1), gguf.GGMLQuantizationType.TQ1_0).reshape(-1)
    atteso = (q.astype(np.float64).reshape(-1, 256) * np.float16(s).astype(np.float64)[:, None]).reshape(-1)
    assert np.array_equal(dq.astype(np.float64), atteso.astype(np.float32).astype(np.float64)), nome
    misure["valori_dequant_primi"] = [float(x) for x in dq[:4]]
    U.scrivi(nome, b, "TQ1_0", [N, 1], misure, dol=[n_vero, 1, modo])
    return dq[:n_vero]


def uni(U, nome, n, motivo):
    if U.fatto(nome):
        return
    q = np.ones(n, np.int8)
    vettore_dol(U, nome, q, [1.0] * (((n + 255) // 256)), n, "tronca" if n % 256 else "copia",
                {"metodo": "piegata nei consumatori -> vettore di uni ESATTO (trit +1, scala 1.0)", "consumatori": motivo})


class Sorgente:
    def __init__(self):
        self.r = GGUFReader(CM.R2B3M)
        self.t = {t.name: t for t in self.r.tensors}
        self.src = {}

    def f32(self, nome):
        t = self.t[nome]
        return np.array(t.data, np.float64).reshape([int(x) for x in t.shape][::-1])

    def piano0(self, nome):
        """(T0 int8 [R,C], S0 fp64 [R,ng], ruotato, tid) del PRIMO piano del tensore (quello che C0 tiene)."""
        t = self.t[nome]
        tid = int(t.tensor_type.value)
        ne = [int(x) for x in t.shape]
        raw = np.array(t.data).reshape(int(np.prod(ne[1:])), -1)
        T, S = gp.decode_rows(raw, tid, ne[0])
        return torch.from_numpy(T.astype(np.int8)), torch.from_numpy(S.astype(np.float32)).double(), tid == TID["TQ1_0R_1024"], tid, ne


def rif_diretto(T0, S0, g, Hh, ruotato):
    """errore della via DIRETTA (g ternario, consumatore invariato), stessa metrica della piega."""
    gq = ternario(g)[0]
    R, C = T0.shape
    Q0 = (T0.double().reshape(R, S0.shape[1], -1) * S0.unsqueeze(-1)).reshape(R, C)
    Wo = CM.deruota(Q0) if ruotato else Q0
    gt, gqt = torch.from_numpy(g), torch.from_numpy(gq)
    return P.err_rel(Wo * gt, Wo * gqt, Hh)


def piega_consumatore(U, S, nome, g, Hraw, H_none=False):
    if U.fatto(nome):
        return
    T0, S0, ruotato, tid, ne = S.piano0(nome)
    t0 = time.time()
    if H_none:
        R, C = T0.shape
        Q0 = (T0.double().reshape(R, S0.shape[1], -1) * S0.unsqueeze(-1)).reshape(R, C)
        Wo = Q0 * torch.from_numpy(g)
        d16, t = TQ.ternario_esatto(Wo.float().numpy().reshape(-1, 256))
        T = torch.from_numpy(t.reshape(R, C).astype(np.int8)); Sn = torch.from_numpy(d16.reshape(R, -1).astype(np.float64))
        I = torch.eye(C, dtype=torch.float64)
        mis = {"via": "ottimo_esatto_non_pesato (nessuna Hessiana per blk.40)", "err_rel_fro": P.err_rel(Wo, (T.double().reshape(R, -1, 256) * Sn.unsqueeze(-1)).reshape(R, C), I)}
    else:
        Hh = P.H_senza_guadagno(Hraw, g)
        Tj, Sj, mj = P.piega_J(T0, S0, g, Hh, ruotato)
        Hr = Hraw / g[:, None] / g[None, :]
        Tf, Sf, mf = P.piega_F(T0, S0, g, Hr, ruotato, tid)
        ef = P.err_in_base(Tf, Sf, T0, S0, g, Hh, ruotato)
        ej = mj["err_rel_H"]
        T, Sn = (Tj, Sj) if ej <= ef else (Tf, Sf)
        mis = {"via": "J" if ej <= ef else "F", "err_rel_H_J": ej, "err_rel_H_F": ef, "err_rel_H": min(ej, ef),
               "err_rel_H_via_diretta_g_ternario": rif_diretto(T0, S0, g, Hh, ruotato)}
    mis["secondi"] = round(time.time() - t0, 1)
    U.scrivi(nome, P.codifica(T, Sn, tid), "TQ1_0R_1024" if ruotato else "TQ1_0", ne, mis)


def da_f32(U, S, nome, g, Hraw, tipo="TQ1_0R_1024"):
    """tensore F32 di C0 (ssm_alpha/beta, router, gate shexp): W diag(g) quantizzato da zero (ricetta di C0)."""
    if U.fatto(nome):
        return
    W = S.f32(nome)
    W = W.reshape(1, -1) if W.ndim == 1 else W
    Wg = (W * g[None, :]).astype(np.float32)
    Hr = None if Hraw is None else Hraw / g[:, None] / g[None, :]
    if tipo == "TQ1_0R_1024":
        packed, mis = TQ.quantizza_1024(Wg, Hr, nome)
    else:
        packed, mis = TQ.quantizza_tq1_0(Wg, Hr, nome)
    ne = [int(x) for x in S.t[nome].shape]
    mis = {k: mis[k] for k in mis if k.startswith("err") or k in ("secondi", "frazione_zeri", "ricetta")}
    if "ffn_gate_inp.weight" in nome and Hraw is not None:
        mis.update(topk_router(W, Wg, packed, g, Hraw, tipo))
    U.scrivi(nome, packed.reshape(Wg.shape[0], -1), tipo, ne, mis)


def topk_router(W, Wg, packed, g, Hraw, tipo, n=4000, k=8):
    import gguf
    R, C = Wg.shape
    if tipo == "TQ1_0R_1024":
        Wq = TQ.dequantize_blocks(packed).reshape(R, C)
    else:
        Wq = gguf.quants.dequantize(packed.reshape(R, -1), gguf.GGMLQuantizationType.TQ1_0).reshape(R, C)
    lam, U_ = np.linalg.eigh(0.5 * (Hraw + Hraw.T)); lam = np.clip(lam, 0, None)
    x = rng.standard_normal((n, C)) @ (U_ * np.sqrt(lam)).T                  # x = ingresso CON guadagno
    a = x @ W.T; b = (x / g) @ Wq.T.astype(np.float64)
    ta = np.argsort(-a, 1)[:, :k]; tb = np.argsort(-b, 1)[:, :k]
    inter = np.mean([len(set(ta[i]) & set(tb[i])) for i in range(n)]) / k
    pa = np.exp(a - a.max(1, keepdims=True)); pa /= pa.sum(1, keepdims=True)
    massa = float(np.mean(np.take_along_axis(pa, tb, 1).sum(1) / np.take_along_axis(pa, ta, 1).sum(1)))
    return {"router_top8_concordanza_surrogato": float(inter), "router_massa_top8_catturata_surrogato": massa}


def sp(x):
    return np.logaddexp(0, x)


N_PASSI = 2048


def Sfun(c):
    """sum_{n=0}^{N-1} exp(n c) per c <= 0 (stabile vicino a 0)."""
    c = np.minimum(c, 0.0)
    den = np.expm1(c)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.expm1(N_PASSI * c) / den
    return np.where(np.abs(c) < 1e-12, float(N_PASSI), v)


def ssm_a_dt(U, S, L):
    na, nd = f"blk.{L}.ssm_a", f"blk.{L}.ssm_dt.bias"
    if U.fatto(na) and U.fatto(nd):
        return
    A, dt = S.f32(na).reshape(-1), S.f32(nd).reshape(-1)
    Wa = S.f32(f"blk.{L}.ssm_alpha.weight").reshape(32, 2048)
    H = np.load(H_ATTN.format(L)).astype(np.float64)
    C_ = Wa @ H @ Wa.T; lam, Uv = np.linalg.eigh(0.5 * (C_ + C_.T)); lam = np.clip(lam, 0, None)
    al = rng.standard_normal((4000, 32)) @ (Uv * np.sqrt(lam)).T
    d0 = np.exp(A * sp(al + dt)); den = float(np.mean(d0 ** 2))
    # METRICA (corretta 21:30): errore L2 relativo della RISPOSTA IMPULSIVA del decadimento su N passi, per testa:
    # lo stato pesa il token di n passi fa con exp(n*l); sum_n (e^{n l1} - e^{n l0})^2 = S(2l1) - 2S(l1+l0) + S(2l0),
    # S(c) = sum_{n<N} e^{n c} = expm1(N c)/expm1(c). Il solo errore sul decadimento di un passo trascurava che
    # l'errore si COMPONE sulle teste a memoria lunga (es. 0.94 contro 0.9995 a un passo = 0.002 contro 0.95 a 100 passi).
    l0 = A * sp(al + dt)
    S0 = Sfun(2 * l0); den_h = S0.mean(0)                                  # [32]
    codici = np.array([-1, 0, 1])
    migliore = None

    def valuta(dominio, gA, gD):
        """per ogni (sA, sD) della griglia: per testa la migliore delle 9 combinazioni di codici; totale minimo."""
        best = None
        vD = gD[:, None] * codici[None, :]                                   # [nD, 3]
        SP = sp(al[None, None] + vD[:, :, None, None])                       # [nD, 3, n, 32]
        for sA in gA:
            vA = sA * codici
            Av = vA if dominio == "lineare" else -np.exp(vA)
            E = np.full((len(gD), 3, 3, 32), np.inf)
            for i in range(3):
                if dominio == "lineare" and Av[i] > 0:                       # A deve restare <= 0 (decadimento)
                    continue
                l1 = Av[i] * SP                                                   # [nD, 3, n, 32]
                E[:, i] = np.mean(Sfun(2 * l1) - 2 * Sfun(l1 + l0[None, None]) + S0[None, None], 2) / den_h
            flat = E.transpose(0, 3, 1, 2).reshape(len(gD), 32, 9)
            kb = flat.argmin(2); tot = flat.min(2).sum(1)                    # [nD]
            j = int(tot.argmin())
            if best is None or tot[j] < best[0]:
                best = (float(tot[j]), dominio, float(sA), float(gD[j]), kb[j] // 3, kb[j] % 3)
        return best

    def griglia(c, lo, hi, n):
        return np.unique(np.float16(np.concatenate([np.linspace(lo, hi, n), np.geomspace(max(lo, 1e-4 * hi), hi, n)]) * c)
                         .astype(np.float64))

    for dominio in ("lineare", "log"):
        base_A = A if dominio == "lineare" else np.log(-A)
        sA0 = np.abs(base_A).max(); sD0 = np.abs(dt).max()
        b = valuta(dominio, griglia(sA0, 0, 1, 30), griglia(sD0, 0, 1, 30))            # grossolana
        for _ in range(2):                                                               # due affinamenti locali
            b2 = valuta(dominio, griglia(b[2], 0.7, 1.4, 25) if b[2] > 0 else griglia(sA0, 0, 0.05, 25),
                        griglia(b[3], 0.7, 1.4, 25) if b[3] > 0 else griglia(sD0, 0, 0.05, 25))
            b = min(b, b2, key=lambda x: x[0])
        if migliore is None or b[0] < migliore[0]:
            migliore = b
    tot, dominio, sA, sD, iA, iD = migliore
    qA, qD = codici[iA].astype(np.int8), codici[iD].astype(np.int8)
    Aq = (sA * qA) if dominio == "lineare" else -np.exp(sA * qA)
    l1 = Aq * sp(al + sD * qD)
    num_h = np.mean(Sfun(2 * l1) - 2 * Sfun(l1 + l0) + S0, 0) / den_h
    mis = {"dominio_A": dominio, "metrica": f"risposta impulsiva L2 relativa su {N_PASSI} passi, media sulle teste",
           "err_rel_risposta_impulsiva": float(np.sqrt(np.mean(num_h))),
           "err_rel_risposta_impulsiva_testa_peggiore": float(np.sqrt(num_h.max())),
           "err_rel_decadimento_un_passo": float(np.sqrt(np.mean((np.exp(l1) - d0) ** 2) / den)),
           "err_rel_log_decadimento": float(np.sqrt(np.mean((l1 - l0) ** 2) / np.mean(l0 ** 2))),
           "rif_ternario_mse_err_decadimento": None}
    Am = ternario(A, 256)[0]; dm = ternario(dt, 256)[0]
    lm = Am * sp(al + dm)
    mis["rif_ternario_mse_err_risposta_impulsiva"] = float(np.sqrt(np.mean(np.mean(Sfun(2 * lm) - 2 * Sfun(lm + l0) + S0, 0) / den_h)))
    mis.pop("rif_ternario_mse_err_decadimento", None)
    vettore_dol(U, na, qA, [sA], 32, "tronca" if dominio == "lineare" else "tronca_negexp",
                {"metodo": f"ternario 1 scala, codici per testa e scale su griglia fp16 che minimizzano l'errore della risposta impulsiva del decadimento; A in dominio {dominio}", **mis})
    vettore_dol(U, nd, qD, [sD], 32, "tronca", {"metodo": "ternario 1 scala, ottimizzato insieme a ssm_a (vedi ssm_a)", **mis})


def conv1d(U, S, L, Tq_qkv):
    nome = f"blk.{L}.ssm_conv1d.weight"
    if U.fatto(nome):
        return
    W = S.f32(nome).reshape(8192, 4)                                    # [canale, tap]
    H = np.load(H_ATTN.format(L)).astype(np.float64)
    Wq = Tq_qkv                                                          # (8192, 2048) base originale (C0)
    var = np.einsum("ij,jk,ik->i", Wq, H, Wq); var = var / var.mean()
    Wt = W.T.copy()                                                      # [tap, canale] = memoria [8192, 4]
    out = np.zeros_like(Wt); Q = np.zeros(Wt.shape, np.int8); Sg = np.zeros((4, 32))
    for k in range(4):
        out[k], Sg[k], Q[k] = ternario(Wt[k], 256, var)
    e_g = float(np.sqrt(np.sum(var[None] * (out - Wt) ** 2) / np.sum(var[None] * Wt ** 2)))
    packed, m2 = TQ.quantizza_1024(Wt.astype(np.float32), np.diag(var), nome)
    Wr = TQ.dequantize_blocks(packed).reshape(4, 8192).astype(np.float64)
    e_r = float(np.sqrt(np.sum(var[None] * (Wr - Wt) ** 2) / np.sum(var[None] * Wt ** 2)))
    mis = {"err_rel_pesato_varianza_TQ1_0": e_g, "err_rel_pesato_varianza_TQ1_0R_1024": e_r}
    if e_g <= e_r:
        b = gp.encode_rows(Q, Sg.astype(np.float16), TID["TQ1_0"])
        mis["scelto"] = "TQ1_0"
        U.scrivi(nome, b, "TQ1_0", [8192, 4], mis, dol=[4, 8192, "trasposta"])
    else:
        mis["scelto"] = "TQ1_0R_1024"
        U.scrivi(nome, packed.reshape(4, -1), "TQ1_0R_1024", [8192, 4], mis, dol=[4, 8192, "trasposta"])


def qk(U, S, L):
    nq, nk = f"blk.{L}.attn_q_norm.weight", f"blk.{L}.attn_k_norm.weight"
    if U.fatto(nq) and U.fatto(nk):
        return
    gq, gk = S.f32(nq).reshape(-1), S.f32(nk).reshape(-1)
    D, NR, HALF = 256, 64, 32
    q = rng.standard_normal((6000, D)); k = rng.standard_normal((6000, D)); th = rng.uniform(0, 2 * np.pi, (6000, HALF))

    def rope(x):
        y = x.copy(); c, s = np.cos(th), np.sin(th)
        y[:, :HALF] = x[:, :HALF] * c - x[:, HALF:NR] * s; y[:, HALF:NR] = x[:, :HALF] * s + x[:, HALF:NR] * c; return y
    a = np.sum(rope(q * gq) * k * gk, 1)

    def err(gq2, gk2):
        b = np.sum(rope(q * gq2) * k * gk2, 1); return float(np.sqrt(np.mean((a - b) ** 2) / np.mean(a ** 2)))
    c_t, c_s, c_q = ternario(gq * gk)
    e1 = err(np.ones(D), c_t)
    gq_t, gq_s, gq_q = ternario(gq); gk_t, gk_s, gk_q = ternario(gk)
    e2 = err(gq_t, gk_t)
    base = {"err_rel_logit_via_i_q1_k_prodotto": e1, "err_rel_logit_via_ii_qk_ternari": e2}
    if e1 <= e2:
        vettore_dol(U, nq, np.ones(D, np.int8), [1.0], D, "copia", {"metodo": "via (i): guadagno di q piegato su k (uni esatti)", **base})
        vettore_dol(U, nk, c_q, c_s, D, "copia", {"metodo": "via (i): k = ternario(g_q*g_k)", **base})
    else:
        vettore_dol(U, nq, gq_q, gq_s, D, "copia", {"metodo": "via (ii): g_q ternario", **base})
        vettore_dol(U, nk, gk_q, gk_s, D, "copia", {"metodo": "via (ii): g_k ternario", **base})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{TT}/OBIETTIVO_C/C0T")
    ap.add_argument("--strati", default="0-40")
    a = ap.parse_args()
    torch.set_num_threads(6)
    U = Uscita(a.out); S = Sorgente()
    lo, _, hi = a.strati.partition("-")
    for L in range(int(lo), int(hi or lo) + 1):
        if L == 40:
            continue
        ga = S.f32(f"blk.{L}.attn_norm.weight").reshape(-1); gpn = S.f32(f"blk.{L}.post_attention_norm.weight").reshape(-1)
        Ha = np.load(H_ATTN.format(L)).astype(np.float64); Hf = np.load(H_FFN.format(L)).astype(np.float64)
        log(f"strato {L}")
        if (L + 1) % 4:
            for f in ("attn_qkv", "attn_gate"):
                piega_consumatore(U, S, f"blk.{L}.{f}.weight", ga, Ha)
            for f in ("ssm_alpha", "ssm_beta"):
                da_f32(U, S, f"blk.{L}.{f}.weight", ga, Ha)
            gs = S.f32(f"blk.{L}.ssm_norm.weight").reshape(-1)
            piega_consumatore(U, S, f"blk.{L}.ssm_out.weight", np.tile(gs, 32), np.load(H_SSMOUT.format(L)).astype(np.float64))
            uni(U, f"blk.{L}.ssm_norm.weight", 128, "ssm_out (guadagno ripetuto sulle 32 teste)")
            ssm_a_dt(U, S, L)
            if not U.fatto(f"blk.{L}.ssm_conv1d.weight"):
                T0, S0, ruot, tid, ne = S.piano0(f"blk.{L}.attn_qkv.weight")
                Q0 = (T0.double().reshape(8192, S0.shape[1], -1) * S0.unsqueeze(-1)).reshape(8192, 2048)
                conv1d(U, S, L, CM.deruota(Q0).numpy())
        else:
            for f in ("attn_q", "attn_k", "attn_v"):
                piega_consumatore(U, S, f"blk.{L}.{f}.weight", ga, Ha)
            qk(U, S, L)
        uni(U, f"blk.{L}.attn_norm.weight", 2048, "attn_qkv/attn_gate/ssm_alpha/ssm_beta o attn_q/k/v")
        for f in ("ffn_gate_shexp", "ffn_up_shexp"):
            piega_consumatore(U, S, f"blk.{L}.{f}.weight", gpn, Hf)
        da_f32(U, S, f"blk.{L}.ffn_gate_inp.weight", gpn, Hf)
        da_f32(U, S, f"blk.{L}.ffn_gate_inp_shexp.weight", gpn, Hf)
        uni(U, f"blk.{L}.post_attention_norm.weight", 2048, "ffn_gate/up_exps (GPU), ffn_gate/up_shexp, router, gate shexp")
    if int(hi or lo) >= 40:
        blocco40(U, S)
        testa(U, S)
    log("FINE")


def blocco40(U, S):
    L = 40
    ga = S.f32("blk.40.attn_norm.weight").reshape(-1); gpn = S.f32("blk.40.post_attention_norm.weight").reshape(-1)
    for f in ("attn_q", "attn_k", "attn_v"):
        piega_consumatore(U, S, f"blk.40.{f}.weight", ga, None, H_none=True)
    qk(U, S, 40)
    uni(U, "blk.40.attn_norm.weight", 2048, "attn_q/k/v di blk.40")
    for f in ("ffn_gate_shexp", "ffn_up_shexp"):
        piega_consumatore(U, S, f"blk.40.{f}.weight", gpn, None, H_none=True)
    da_f32(U, S, "blk.40.ffn_gate_inp.weight", gpn, None, tipo="TQ1_0")
    da_f32(U, S, "blk.40.ffn_gate_inp_shexp.weight", gpn, None, tipo="TQ1_0")
    for f in ("ffn_gate_exps", "ffn_up_exps"):
        nome = f"blk.40.{f}.weight"
        if U.fatto(nome):
            continue
        t = S.t[nome]; ne = [int(x) for x in t.shape]                    # [2048, 512, 256]
        raw = np.array(t.data).reshape(ne[2] * ne[1], -1)
        T, Sc = gp.decode_rows(raw, TID["TQ1_0"], ne[0])
        Wo = (T.astype(np.float64).reshape(-1, ne[0] // 256, 256) * Sc.astype(np.float64)[:, :, None]).reshape(-1, ne[0]) * gpn
        d16, tq = TQ.ternario_esatto(Wo.astype(np.float32).reshape(-1, 256))
        Wn = (tq.astype(np.float64) * d16.astype(np.float64)[:, None]).reshape(Wo.shape)
        mis = {"via": "ottimo_esatto_non_pesato (nessuna Hessiana per blk.40)",
               "err_rel_fro": float(np.linalg.norm(Wn - Wo) / np.linalg.norm(Wo))}
        b = gp.encode_rows(tq.reshape(-1, ne[0]).astype(np.int8), d16.reshape(-1, ne[0] // 256).astype(np.float16), TID["TQ1_0"])
        U.scrivi(nome, b.reshape(ne[2], ne[1], -1), "TQ1_0", ne, mis)
    uni(U, "blk.40.post_attention_norm.weight", 2048, "esperti/shexp/router di blk.40")


def testa(U, S):
    go = S.f32("output_norm.weight").reshape(-1)
    if not U.fatto("output.weight"):
        T0, S0, ruotato, tid, ne = S.piano0("output.weight")
        Hraw = np.load(H_OUT).astype(np.float64); Hh = P.H_senza_guadagno(Hraw, go)
        t0 = time.time()
        Tj, Sj, mj = P.piega_J(T0, S0, go, Hh, ruotato, righe=8192)
        mis = {"via": "J", "err_rel_H": mj["err_rel_H"], "err_rel_H_senza_piega": mj["err_rel_H_senza_piega"],
               "err_rel_H_via_diretta_g_ternario": rif_diretto(T0[:16384], S0[:16384], go, Hh, ruotato),
               "nota": "via F non provata sulla testa (248320 righe): J dai codici di C0", "secondi": round(time.time() - t0, 1)}
        U.scrivi("output.weight", P.codifica(Tj, Sj, tid), "TQ1_0R_1024", ne, mis)
    uni(U, "output_norm.weight", 2048, "output (testa)")
    # MTP: shared_head_norm' = g_sh / g_out (T5, esatto) -> ternario diretto
    gsh = S.f32("blk.40.nextn.shared_head_norm.weight").reshape(-1)
    if not U.fatto("blk.40.nextn.shared_head_norm.weight"):
        r_ = gsh / go; vt, s_, q_ = ternario(r_)
        vettore_dol(U, "blk.40.nextn.shared_head_norm.weight", q_, s_, 2048, "copia",
                    {"metodo": "testa piegata per g_out -> serve g_sh/g_out (esatto), poi ternario diretto (solo MTP)",
                     "err_rel": float(np.linalg.norm(vt - r_) / np.linalg.norm(r_))})
    # eh_proj: colonne [0,2048) = enorm(e), [2048,4096) = hnorm(h); g_h' = g_h * g_out * rho (T7, rho dal surrogato)
    if not U.fatto("blk.40.nextn.eh_proj.weight"):
        ge = S.f32("blk.40.nextn.enorm.weight").reshape(-1); gh = S.f32("blk.40.nextn.hnorm.weight").reshape(-1)
        Hraw = np.load(H_OUT).astype(np.float64); lam, Uv = np.linalg.eigh(0.5 * (Hraw + Hraw.T)); lam = np.clip(lam, 0, None)
        h = rng.standard_normal((4000, 2048)) @ (Uv * np.sqrt(lam)).T
        rr = np.sqrt(np.mean((h / go) ** 2, 1)) / np.sqrt(np.mean(h ** 2, 1)); rho = float(rr.mean())
        gcat = np.concatenate([ge, gh * go * rho])
        t = S.t["blk.40.nextn.eh_proj.weight"]; ne = [int(x) for x in t.shape]
        raw = np.array(t.data).reshape(ne[1], -1)
        T, Sc = gp.decode_rows(raw, TID["TQ1_0"], ne[0])
        Wo = (T.astype(np.float64).reshape(ne[1], -1, 256) * Sc.astype(np.float64)[:, :, None]).reshape(ne[1], ne[0]) * gcat
        d16, tq = TQ.ternario_esatto(Wo.astype(np.float32).reshape(-1, 256))
        Wn = (tq.astype(np.float64) * d16.astype(np.float64)[:, None]).reshape(Wo.shape)
        a_ = h / np.sqrt(np.mean(h ** 2, 1, keepdims=True)) * gh; hp = h / go
        b_ = hp / np.sqrt(np.mean(hp ** 2, 1, keepdims=True)) * (gh * go * rho)
        mis = {"via": "ottimo_esatto_non_pesato", "err_rel_fro": float(np.linalg.norm(Wn - Wo) / np.linalg.norm(Wo)),
               "rho": rho, "cv_rapporto_rms": float(rr.std() / rho),
               "err_rel_hnorm_mtp_non_esatto": float(np.sqrt(np.mean((a_ - b_) ** 2) / np.mean(a_ ** 2)))}
        b = gp.encode_rows(tq.reshape(ne[1], ne[0]).astype(np.int8), d16.reshape(ne[1], -1).astype(np.float16), TID["TQ1_0"])
        U.scrivi("blk.40.nextn.eh_proj.weight", b, "TQ1_0", ne, mis)
    uni(U, "blk.40.nextn.enorm.weight", 2048, "eh_proj colonne 0-2047")
    uni(U, "blk.40.nextn.hnorm.weight", 2048, "eh_proj colonne 2048-4095 (con g_out e rho)")


if __name__ == "__main__":
    main()
