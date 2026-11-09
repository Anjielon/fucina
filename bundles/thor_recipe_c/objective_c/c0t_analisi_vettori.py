"""Obiettivo C / C0T (9/10): misura degli errori delle vie per rendere ternari i vettori rimasti (CPU, fp64).
q/k norm: (i) gain q=1, k-gain c=g_q*g_k ternario  (ii) g_q e g_k ternari al loro posto; errore relativo dei logit
q~.k~ con q^,k^ gaussiani unitari, IMROPE coppie (i, i+32) su 64 dimensioni, angoli casuali.
ssm_a / dt_bias: log-decadimento l = A*softplus(alpha+dt), alpha ~ N(0, w_h^T H w_h) (H dell'ingresso attn);
vie: A lineare ternario, A_log ternario (A=-exp), dt ternario; errore relativo su l e su exp(l)."""
import json, sys
sys.path.insert(0, "@@ROOT:models@@/build/thor-rexppiani2/gguf-py"); sys.path.insert(0, "@@ROOT:models@@/gguf/tony-tern/OBIETTIVO_C/codice")
import numpy as np
from gguf import GGUFReader
from c0t_ternario_vettori import ternario
r = GGUFReader("@@ROOT:models@@/gguf/tony-tern/R2B3m/Thor-T-R2B3m.gguf")
T = {t.name: t for t in r.tensors}
def v(n): return np.array(T[n].data, dtype=np.float64).reshape(-1)
rng = np.random.default_rng(0)
out = {"qk": {}, "ssm": {}}
NR, HALF = 64, 32
def logit_err(gq, gk, gq2, gk2, n=4000):
    D = len(gq); q = rng.standard_normal((n, D)); k = rng.standard_normal((n, D)); th = rng.uniform(0, 2*np.pi, (n, HALF))
    def rope(x):
        y = x.copy(); c, s = np.cos(th), np.sin(th)
        y[:, :HALF] = x[:, :HALF]*c - x[:, HALF:NR]*s; y[:, HALF:NR] = x[:, :HALF]*s + x[:, HALF:NR]*c; return y
    a = np.sum(rope(q*gq) * k*gk, 1)       # rope relativa: ruota solo q (equivale a R_m^T R_n)
    b = np.sum(rope(q*gq2) * k*gk2, 1)
    return float(np.sqrt(np.mean((a-b)**2)/np.mean(a**2)))
for L in [3,7,11,15,19,23,27,31,35,39,40]:
    gq, gk = v(f"blk.{L}.attn_q_norm.weight"), v(f"blk.{L}.attn_k_norm.weight")
    pair = float(np.max(np.abs(gq[:HALF]-gq[HALF:NR])/gq[:HALF]))
    c = gq*gk; ct, _, _ = ternario(c)
    e1 = logit_err(gq, gk, np.ones_like(gq), ct)                 # (i)
    e1x = logit_err(gq, gk, np.ones_like(gq), c)                 # (i) senza ternario: solo l'effetto delle coppie
    gqt, _, _ = ternario(gq); gkt, _, _ = ternario(gk)
    e2 = logit_err(gq, gk, gqt, gkt)                             # (ii)
    # (iii) q ternario solo sulle 64 ruotate (resto 1), k-gain = c sulle non ruotate, g_k sulle ruotate
    gq3 = np.ones_like(gq); gq3[:NR] = ternario(gq[:NR], 64)[0]; gk3 = c.copy(); gk3[:NR] = gk[:NR]; gk3t = ternario(gk3)[0]
    e3 = logit_err(gq, gk, gq3, gk3t)
    out["qk"][L] = {"coppie_diverse_max_rel": pair, "i_kgain_c_ternario": e1, "i_solo_coppie_fp": e1x, "ii_qk_ternari": e2,
                    "iii_qrot_ternario_k_c": e3}
    print("qk", L, json.dumps({k: round(x, 4) for k, x in out["qk"][L].items()}), flush=True)
def sp(x): return np.logaddexp(0, x)
for L in [l for l in range(40) if (l+1) % 4]:
    A, dt = v(f"blk.{L}.ssm_a"), v(f"blk.{L}.ssm_dt.bias")
    Wa = np.array(T[f"blk.{L}.ssm_alpha.weight"].data, dtype=np.float64).reshape(32, 2048)
    H = np.load(f"@@ROOT:models@@/gguf/tony-tern/hessiane-densi-grande/H_attn_{L:02d}.npy").astype(np.float64)
    C = Wa @ H @ Wa.T; C = 0.5*(C+C.T)
    lam, U = np.linalg.eigh(C); lam = np.clip(lam, 0, None)
    al = rng.standard_normal((4000, 32)) @ (U*np.sqrt(lam)).T
    l0 = A*sp(al+dt)
    At = ternario(A, 256)[0]; Alt = -np.exp(ternario(np.log(-A), 256)[0]); dtt = ternario(dt, 256)[0]
    def err(Ax, dx):
        l1 = Ax*sp(al+dx); return (float(np.sqrt(np.mean((l1-l0)**2)/np.mean(l0**2))),
                                   float(np.sqrt(np.mean((np.exp(l1)-np.exp(l0))**2)/np.mean(np.exp(l0)**2))))
    out["ssm"][L] = {"A_lin": err(At, dt), "A_log": err(Alt, dt), "dt": err(A, dtt), "A_lin+dt": err(At, dtt), "A_log+dt": err(Alt, dtt)}
    print("ssm", L, json.dumps(out["ssm"][L]), flush=True)
json.dump(out, open("@@ROOT:models@@/gguf/tony-tern/OBIETTIVO_C/C0T/analisi/vettori.json", "w"), indent=1)
