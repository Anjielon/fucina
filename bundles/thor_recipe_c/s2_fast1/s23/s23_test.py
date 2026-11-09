#!/usr/bin/env python3
"""S2+S3 — prove CPU pure (fp64, niente file grandi, niente GPU): i porting di s23_algebra contro gli ORIGINALI congelati
in ../rif (solutori.gptq_bilanciato, jarq_fedele.jarq_fedele, ptq2_costruttori.drq, fk_scale_date, s1b passo_colonne),
identita' matematiche (QEP = simmetrico attorno a W*, Teorema 2.1 di FastKron con B = I), piega K2 esatta, ganci con arresto,
andata e ritorno dei byte. Stampa OK/KO per riga; codice d'uscita 1 se un KO."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("LEVE_DT", "float64")
import numpy as np  # noqa: E402
import torch  # noqa: E402

QUI = Path(__file__).resolve().parent
for p in (str(QUI), str(QUI.parent / "lib"), str(QUI.parent / "s1b"), str(QUI.parent / "rif")):
    if p not in sys.path:
        sys.path.insert(0, p)
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
import s23_algebra as AL  # noqa: E402
import s23_risolvi as RS  # noqa: E402
import s1b_comune as K  # noqa: E402
import s1_comune as SC  # noqa: E402
import comune  # noqa: E402,F401   (rif: raffina/comune.py congelato, prima di jarq_fedele)
import solutori as SO  # noqa: E402   (rif: raffina/solutori.py congelato)
import jarq_fedele as JF  # noqa: E402
import ptq2_costruttori as PT  # noqa: E402
import fk_scale_date as FK  # noqa: E402
import cardan_piega as CP  # noqa: E402
import s23_passata as SP  # noqa: E402

F64 = torch.float64
g = torch.Generator().manual_seed(20261013)
esiti = []


def prova(nome, cond, info=""):
    esiti.append(bool(cond))
    print(f"{'OK' if cond else 'KO'} {nome} {info}", flush=True)


def spd(n, N=None, scala=True):
    X = torch.randn(N or 3 * n, n, generator=g, dtype=F64)
    if scala:
        X = X * torch.linspace(0.2, 3, n, dtype=F64)
    return X.T @ X / X.shape[0], X


def norm(H):
    return H / torch.diagonal(H).mean()


# 1 GPTQ bilanciato K=1 a lotti == solutori.gptq_bilanciato (al bit, fp64 CPU), due tensori con H diverse
R, C, ng = 24, 256, 2
Ws = [torch.randn(R, C, generator=g, dtype=F64) * 0.02 for _ in range(2)]
Hs = [norm(spd(C)[0]) for _ in range(2)]
U = torch.stack([AL.cholesky_smorzata(H)[1] for H in Hs])
T, S = AL.gptq_bil_lotto(torch.stack(Ws), U, ng)
ok = True
for b in range(2):
    To, So, _ = SO.gptq_bilanciato(Ws[b], Hs[b], 1, ng)
    ok &= torch.equal(To[0], T[b]) and torch.equal(SO.fp16(So[0]), S[b])
prova("gptq_bil_lotto == solutori.gptq_bilanciato K=1 (al bit)", ok)

# 2 JARQ fedele portato == jarq_fedele.jarq_fedele (fp64 CPU)
W, H = Ws[0], Hs[0]
c = 196608.0 * 0.7
Tj, Sj, cj = AL.jarq_fedele_dev(W, H, T[0], S[0], c)
Tjo, Sjo, cjo = JF.jarq_fedele(W, H, T[0].unsqueeze(0), S[0].unsqueeze(0), c)
prova("jarq_fedele_dev == jarq_fedele (al bit)", torch.equal(Tj, Tjo[0]) and torch.equal(Sj, Sjo[0]),
      f"finale {cj['finale_fp16']:.6g} vs {cjo['finale_fp16']:.6g}")

# 3 Lanczos contro eigh
M = spd(200, 600)[0]
lam, v = AL.lanczos_max(M)
ev = torch.linalg.eigvalsh(M)
prova("lanczos_max = lambda_max di eigh", abs(lam / float(ev[-1]) - 1) < 1e-10, f"{lam:.12g} vs {float(ev[-1]):.12g}")

# 4 DRQ portato contro ptq2_costruttori.drq (stessi codici finali o stessa J a 1e-9)
Hraw = spd(C, 4 * C)[0]
Hc = spd(C, 4 * C)[0]
Td, cd = AL.drq_dev(W, Hraw, [Hc], Tj, Sj)
Tdo, cdo = PT.drq(W, Hraw, [Hc], Tj.unsqueeze(0), Sj.unsqueeze(0))
prova("drq_dev == ptq2.drq", torch.equal(Td, Tdo[0]) or abs(cd["J"] / cdo["J"] - 1) < 1e-9,
      f"J {cd['J']:.9g} vs {cdo['J']:.9g}; trit cambiati {cd['trit_cambiati']:.4f}; Lanczos {cd['lanczos']} ripieghi {cd['ripieghi_eigh']}")

# 5 FastKron a scale date: == fk_scale_date originale; con B = I == GPTQ a scale date (Teorema 2.1)
Rk, Ck = 16, 256
Wk = torch.randn(Rk, Ck, generator=g, dtype=F64) * 0.02
A_ = norm(spd(Ck)[0])
B_ = norm(spd(Rk, 64)[0])
Sd = SO.fp16(Wk.abs().reshape(Rk, 2, 128).amax(-1) * 0.6)
Tf = AL.fastkron_sd(Wk, A_, B_, Sd)
Tfo, _, _ = FK.fastkron_scale_date(Wk, A_, B_, 2, Sd.unsqueeze(0), Kl=1)
prova("fastkron_sd == fk_scale_date (al bit)", torch.equal(Tf, Tfo[0]))
TfI = AL.fastkron_sd(Wk, A_, torch.eye(Rk, dtype=F64), Sd)
LA = AL.L_unit(A_)
Wr = Wk.clone()
Tg = torch.zeros(Rk, Ck, dtype=torch.int8)
alf = torch.stack([-Sd, torch.zeros_like(Sd), Sd], -1)
for j in range(Ck):
    a_ = alf[:, j // 128]
    jx = (a_ - Wr[:, j].unsqueeze(1)).abs().argmin(1)
    vq = a_.gather(1, jx.unsqueeze(1)).squeeze(1)
    Tg[:, j] = (jx - 1).to(torch.int8)
    Wr[:, j:] += (vq - Wr[:, j]).unsqueeze(1) * LA[j:, j].unsqueeze(0)
prova("FastKron con B = I == GPTQ a un lato a scale date (Teorema 2.1)", torch.equal(TfI, Tg))

# 6 scale a due lati: equazioni normali soddisfatte
s2 = AL.scale_due_lati(Wk, Tf, Sd, A_, B_, it=400)
Tt = Tf.to(F64)


def grad_s(s):
    E = Wk - AL.espandi(s, Ck) * Tt
    return ((B_ @ E @ A_) * Tt).reshape(Rk, 2, 128).sum(-1)


prova("scale_due_lati: gradiente nullo all'ottimo", float(grad_s(s2).norm() / grad_s(Sd).norm()) < 1e-6,
      f"{float(grad_s(s2).norm() / grad_s(Sd).norm()):.2e}")

# 7 identita' QEP: E||W x_f - Q x_q||^2 = ||(W* - Q)||_Hq^2 + cost, W* = W Mx Hq^-1 (a = 1, damp 0)
Cq = 64
Hq_, Xq = spd(Cq, 400)
Xf = Xq + 0.1 * torch.randn(Xq.shape, generator=g, dtype=F64)
Mx = Xf.T @ Xq / Xq.shape[0]
Wq = torch.randn(8, Cq, generator=g, dtype=F64)
Wst = Wq @ Mx @ torch.linalg.inv(Hq_)
dif = []
for _ in range(3):
    Q = torch.randn(8, Cq, generator=g, dtype=F64)
    asim = float(((Xf @ Wq.T - Xq @ Q.T) ** 2).sum() / Xq.shape[0])
    sym = float((((Wst - Q) @ Hq_) * (Wst - Q)).sum())
    dif.append(asim - sym)
prova("QEP a=1: obiettivo asimmetrico = simmetrico attorno a W* + costante", max(dif) - min(dif) < 1e-9 * abs(dif[0]) + 1e-9,
      f"costanti {dif}")
prova("J_asim coerente con l'obiettivo asimmetrico", abs((RS.J_asim(Q, Hq_, Wq @ Mx) + float(((Xf @ Wq.T) ** 2).sum()) / Xq.shape[0])
                                                       - asim) < 1e-8 * abs(asim))

# 8 passo_colonne portato == s1b_comune.passo_colonne (con e senza legame GQA)
Qo = torch.randn(32, 64, generator=g, dtype=F64)
Hq2 = spd(64)[0]
Cl = torch.randn(32, 64, generator=g, dtype=F64)
P = AL.legame_gqa(4, 2, 16)
prova("passo_colonne == s1b (libero)", torch.allclose(AL.passo_colonne(Qo, Hq2, Cl), K.passo_colonne(Qo, Hq2, Cl), rtol=1e-12, atol=1e-14))
prova("passo_colonne == s1b (GQA)", torch.allclose(AL.passo_colonne(Qo, Hq2, Cl, P), K.passo_colonne(Qo, Hq2, Cl, K.legame_gqa(4, 2, 16)),
                                                    rtol=1e-12, atol=1e-14))

# 9 piega K2 esatta: up (righe x c) poi down (colonne / c) == down con colonna c sull'ingresso originale
Wu = torch.randn(12, 32, generator=g, dtype=F64)
Wdn = torch.randn(32, 12, generator=g, dtype=F64)
cc = torch.rand(12, generator=g, dtype=F64) + 0.5
x = torch.randn(5, 32, generator=g, dtype=F64)
prova("piega K2 (righe a monte x c) == colonna c a valle", torch.allclose((x @ (Wu * cc.unsqueeze(1)).T) @ Wdn.T, (x @ Wu.T) @ (Wdn * cc).T))

# 10 andata e ritorno dei byte (TQ1_0R_1024 e TQ1_0)
for tid, Cb, G in ((45, 2048, 128), (34, 512, 256)):
    Tb = torch.randint(-1, 2, (8, Cb), generator=g).to(torch.int8).numpy()
    Sb = (torch.rand(8, Cb // G, generator=g) * 0.01 + 1e-3).to(torch.float16).numpy()
    bb = K.gp.encode_rows(Tb, Sb, tid)
    t2, s2_ = K.gp.decode_rows(bb, tid, Cb)
    prova(f"andata e ritorno tipo {tid}", np.array_equal(t2, Tb) and np.array_equal(s2_.view(np.uint16), Sb.view(np.uint16)))

# 11 ganci con arresto su un modulo giocattolo


class Strato(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = torch.nn.Identity()
        self.post_attention_layernorm = torch.nn.Identity()
        self.linear_attn = torch.nn.Module()
        self.linear_attn.out_proj = torch.nn.Linear(4, 4)
        self.mlp = torch.nn.Module()
        self.mlp.shared_expert = torch.nn.Module()
        self.mlp.shared_expert.down_proj = torch.nn.Linear(4, 4)
        self.passi = 0

    def forward(self, x):
        h = self.input_layernorm(x)
        self.passi += 1
        h = self.linear_attn.out_proj(h + 1)
        self.passi += 1
        h = self.post_attention_layernorm(h)
        self.passi += 1
        return self.mlp.shared_expert.down_proj(h)


s_ = Strato()
gg = SP.Gancio(s_, False, ("attn", "att_out"), ferma="att_out")
fermato = False
try:
    s_(torch.ones(2, 4))
except SP.Ferma:
    fermato = True
finally:
    gg.chiudi()
prova("gancio: arresto al punto, prima del resto", fermato and s_.passi == 1 and torch.equal(gg.d["att_out"], torch.full((2, 4), 2.0))
      and len(s_.input_layernorm._forward_hooks) == 0 and len(s_.linear_attn.out_proj._forward_pre_hooks) == 0)

# 12 U_Minv_lotto == cholesky_smorzata; spacchetta_lotto == inversa di impacchetta
Hb = torch.stack([norm(spd(64)[0]) for _ in range(3)])
Mi, Ub, nr = SP.U_Minv_lotto(Hb)
ok = all(torch.allclose(Ub[b], AL.cholesky_smorzata(Hb[b])[1], rtol=1e-10, atol=1e-12) and
         torch.allclose(Mi[b], AL.cholesky_smorzata(Hb[b])[0], rtol=1e-10, atol=1e-12) for b in range(3))
prova("U_Minv_lotto == cholesky_smorzata", ok and nr == 0)
prova("spacchetta_lotto(impacchetta) == identita'", torch.equal(SP.spacchetta_lotto(SC.impacchetta(Hb), 64, "cpu", F64), 0.5 * (Hb + Hb.transpose(1, 2)))
      or torch.allclose(SP.spacchetta_lotto(SC.impacchetta(Hb), 64, "cpu", F64), Hb, atol=1e-15))

# 13 JARQ a lotti di CARDAN (esperti): obiettivo non peggiore della partenza, scale fp16
Wl = torch.stack(Ws)
Hl = torch.stack(Hs)
T3, S3, cat3 = CP.jarq_lotto(Wl, Hl, T.unsqueeze(1), S.unsqueeze(1), giri=3, nu2=1e-4)
f0 = CP.obiettivo_lotto(Wl, CP.ricostruisci_lotto(T.unsqueeze(1), S.unsqueeze(1)), Hl + 1e-4 * torch.eye(C, dtype=F64)).sum()
f1 = CP.obiettivo_lotto(Wl, CP.ricostruisci_lotto(T3, S3), Hl + 1e-4 * torch.eye(C, dtype=F64)).sum()
prova("jarq_lotto (CARDAN) non peggiora e arrotonda a fp16", float(f1) <= float(f0) and torch.equal(S3, AL.fp16(S3)), f"{float(f1):.6g} <= {float(f0):.6g}")

# 14 risolvi_denso di prova: sceglie un candidato valido e non peggiore della partenza sulla meta' B
Rd, Cd = 16, 1024
Wd = torch.randn(Rd, Cd, generator=g, dtype=F64) * 0.02


def acc_finto():
    out = {}
    for s_n, n in (("M", 900), ("Kv", 600), ("Kw", 500)):
        X = torch.randn(n, Cd, generator=g, dtype=F64) * torch.linspace(0.3, 2, Cd, dtype=F64)
        Xf_ = X + 0.05 * torch.randn(n, Cd, generator=g, dtype=F64)
        out[s_n] = (X.T @ X, Xf_.T @ X, n)
    return out


accA, accB = acc_finto(), acc_finto()
Hm_, _, _ = RS.mescola(accA, 1.0)
T0p, S0p, _ = SO.gptq_bilanciato(K.ruota(Wd), norm(K.ruota_H(Hm_)), 1, 8)
opz = {"alfe": [0.5, 1.0], "w": 1.0, "damp_qep": 0.01, "drq": True, "k1": (), "giri_k1": 2, "k2": False, "giri_k2": 2,
       "limite_c": 8.0}
r = RS.risolvi_denso("blk.0.attn_gate.weight", "attn_gate", 45, Wd, accA, accB, T0p[0], SO.fp16(S0p[0]), opz)
JB = {k: v["JB"] for k, v in r["registro"]["candidati"].items()}
prova("risolvi_denso: trit validi, scale fp16, scelta = minimo sulla meta' B e non peggiore della partenza",
      bool(torch.isin(r["T"], torch.tensor([-1, 0, 1], dtype=torch.int8)).all()) and torch.equal(r["S"], AL.fp16(r["S"]))
      and JB[r["scelto"]] <= JB["partenza"]
      and JB[r["scelto"]] == min(v["JB"] for k, v in r["registro"]["candidati"].items() if k == "partenza" or v["valido"]),
      f"scelto {r['scelto']} JB {JB}")

# 15 (13/10) DRQ per famiglia: griglia di 2610.11226 App. C, coppia di prima al bit, unita' «norm» invarianti di scala,
#    taratura dai registri chiusi (ripresa senza doppi conti), risolvi_denso tarato/fisso
gp, gpn, gs = RS.griglia_drq("paper"), RS.griglia_drq("paper+norm"), RS.griglia_drq("spenta")
prova("griglia DRQ: 20 (paper) / 40 (paper+norm) / 1 (spenta), prima = coppia di prima, nessun doppione",
      len(gp) == 20 and len(gpn) == 40 and gs == [RS.COMBO_FISSA] and gp[0] == gpn[0] == RS.COMBO_FISSA
      and len(set(gpn)) == 40 and {RS.combo_valori(k)[1] for k in gp} == set(RS.GRIGLIA_ALFA)
      and {RS.combo_valori(k)[2] for k in gp} == set(RS.GRIGLIA_BETA), f"{gp[:3]} ...")
Tf_, cf_ = RS.drq_combo(W, Hraw, Hc, Tj, Sj, RS.COMBO_FISSA)
prova("drq_combo(coppia di prima) == drq_dev di prima (al bit)", torch.equal(Tf_, Td) and cf_["J"] == cd["J"])
s2 = 2.0 ** -14                                       # potenza di 2: in fp64 la scala e' esatta
Tn1, cn1 = RS.drq_combo(W, Hraw, Hc, Tj, Sj, "norm:a0.3:b0.1")
Tn2, cn2 = RS.drq_combo(W, Hraw * s2, Hc * s2, Tj, Sj, "norm:a0.3:b0.1")
Tg1, _ = RS.drq_combo(W, Hraw, Hc, Tj, Sj, "grezza:a0.3:b0.1")
Tg2, _ = RS.drq_combo(W, Hraw * s2, Hc * s2, Tj, Sj, "grezza:a0.3:b0.1")
prova("DRQ «norm» invariante alla scala degli ingressi (H x 2^-14: stessi codici); «grezza» no (atteso, e' il difetto)",
      torch.equal(Tn1, Tn2), f"trit norm {cn1['trit_cambiati']:.4f}; grezza: scala cambia i codici {not torch.equal(Tg1, Tg2)}")
reg_t = lambda rel, modo="paper+norm": {"densi": {"blk.0.ssm_out.weight": {"drq_griglia": {"tarato": True, "modo": modo,
                                                                                             "rel": rel}}}}
r1 = {k: 0.0 for k in gpn}; r1["norm:a0.5:b0.25"] = -3e-3; r1["grezza:a0.1:b1"] = -2e-3
r2 = {k: 0.0 for k in gpn}; r2["norm:a0.5:b0.25"] = +2e-3; r2["grezza:a0.1:b1"] = -2e-3
t0_ = RS.taratura_famiglia([], "ssm_out", 2, "paper+norm")
t1_ = RS.taratura_famiglia([(0, reg_t(r1))], "ssm_out", 2, "paper+norm")
t2_ = RS.taratura_famiglia([(0, reg_t(r1)), (1, reg_t(r2))], "ssm_out", 2, "paper+norm")
t3_ = RS.taratura_famiglia([(0, reg_t(r1)), (1, reg_t(r2, "paper"))], "ssm_out", 2, "paper+norm")
t4_ = RS.taratura_famiglia([(0, reg_t(r1)), (1, reg_t(r2))], "attn_gate", 2, "paper+norm")
prova("taratura_famiglia: tara finche' < n strati (somme portate), poi FISSA = argmin delle somme; altro modo/famiglia non conta",
      t0_["tarato"] and t0_["somme_prec"] == {} and t1_["tarato"] and t1_["somme_prec"]["norm:a0.5:b0.25"] == -3e-3
      and not t2_["tarato"] and t2_["combo"] == "grezza:a0.1:b1" and t3_["tarato"] and t3_["strati_tarati"] == [0]
      and t4_["tarato"] and RS.taratura_famiglia([], "x", 2, "spenta")["combo"] == RS.COMBO_FISSA,
      f"dopo 2 strati {t2_['combo']}")
prova("scegli_combo: a parita' vince l'ordine della griglia (la coppia di prima)",
      RS.scegli_combo({k: 0.0 for k in gpn}, gpn) == RS.COMBO_FISSA)
opz_f = {**opz, "drq_taratura": {"tarato": False, "combo": RS.COMBO_FISSA, "modo": "paper+norm", "strati_tarati": [0, 1]}}
rf = RS.risolvi_denso("blk.0.attn_gate.weight", "attn_gate", 45, Wd, accA, accB, T0p[0], SO.fp16(S0p[0]), opz_f)
prova("risolvi_denso con la coppia FISSA di prima == stesura di prima (al bit)",
      torch.equal(rf["T"], r["T"]) and torch.equal(rf["S"], r["S"]) and rf["scelto"] == r["scelto"])
opz_t = dict(opz)
opz_t["drq_taratura"] = {**RS.taratura_famiglia([], "attn_gate", 1, "paper+norm"), "somme_prec": {"norm:a1:b1": -1.0}}
rt = RS.risolvi_denso("blk.0.attn_gate.weight", "attn_gate", 45, Wd, accA, accB, T0p[0], SO.fp16(S0p[0]), opz_t)
dg = rt["registro"]["drq_griglia"]
JBt = {k: v["JB"] for k, v in rt["registro"]["candidati"].items()}
prova("risolvi_denso TARATO: 40 combinazioni, scelta = argmin(somme precedenti + rel), guardia invariata",
      dg["tarato"] and len(dg["rel"]) == 40 and dg["scelta"] == "norm:a1:b1"
      and all(v["cat"]["drq_combo"] == "norm:a1:b1" for k, v in rt["registro"]["candidati"].items() if k.startswith("qep"))
      and JBt[rt["scelto"]] <= JBt["partenza"], f"{dg['secondi']} s; rel coppia di prima {dg['rel_fissa']:+.3e}")

# K-best di Klein (OJBKQ 2602.08376, S23_KBEST): K<=1 = Babai di prima al bit; K>1 riproducibile (seme fisso), non peggiore
_kb0 = AL.KBEST
AL.KBEST = 1
Tk1, Sk1, _ = AL.jarq_fedele_dev(W, H, T[0], S[0], c)
AL.KBEST = 5
Tk5, Sk5, ck5 = AL.jarq_fedele_dev(W, H, T[0], S[0], c)
Tk5b, Sk5b, _ = AL.jarq_fedele_dev(W, H, T[0], S[0], c)
Hk = H + (AL.NU ** 2 / c) * torch.eye(H.shape[0], dtype=H.dtype)
fk1 = float(AL.obiettivo(W, AL.ricostruisci(Tk1, Sk1), Hk).sum()); fk5 = float(AL.obiettivo(W, AL.ricostruisci(Tk5, Sk5), Hk).sum())
f0k = float(AL.obiettivo(W, AL.ricostruisci(T[0], S[0]), Hk).sum())
prova("K-best: K=1 == JARQ di prima al bit; K=5 riproducibile (seme), trit validi, non peggiore della partenza",
      torch.equal(Tk1, Tj) and torch.equal(Sk1, Sj) and torch.equal(Tk5, Tk5b) and torch.equal(Sk5, Sk5b)
      and int(Tk5.abs().max()) <= 1 and fk5 <= f0k, f"K1 {fk1:.6g} K5 {fk5:.6g} partenza {f0k:.6g}")
Wl, Hl = torch.stack(Ws[:2]), torch.stack(Hs[:2])
T5l, S5l, c5l = AL.jarq_lotto_kbest_da(CP)(Wl, Hl, T[:2].unsqueeze(1), S[:2].unsqueeze(1), giri=3, nu2=1e-4)
Ha_l = Hl + 1e-4 * torch.eye(Hl.shape[-1], dtype=Hl.dtype)
fl0 = CP.obiettivo_lotto(Wl, CP.ricostruisci_lotto(T[:2].unsqueeze(1), S[:2].unsqueeze(1)), Ha_l)
fl5 = CP.obiettivo_lotto(Wl, CP.ricostruisci_lotto(T5l, S5l), Ha_l)
prova("K-best a lotti (esperti): per riga non peggiore della partenza, scale fp16", bool((fl5 <= fl0).all())
      and torch.equal(S5l, AL.fp16(S5l)), f"{float(fl5.sum()):.6g} <= {float(fl0.sum()):.6g}")
AL.KBEST = _kb0

# 14/10 (C_S23_CORRETTO_1014) G1-G3: guardia simmetrica + energia, H corta
accA3 = {k: tuple(v) for k, v in accA.items()}
H1a, M1a, N1a = RS.mescola(accA3, 1.0)
nM, nKv, nKw = accA3["M"][2], accA3["Kv"][2], accA3["Kw"][2]
H1v = (accA3["M"][0] / nM + (accA3["Kv"][0] + accA3["Kw"][0]) / (nKv + nKw)) / 2
H3a, _, _ = RS.mescola(accA3, 1.0, 3.0)
H3v = (accA3["M"][0] / nM + (accA3["Kv"][0] + 3 * accA3["Kw"][0]) / (nKv + 3 * nKw)) / 2
H1b, _, _ = RS.mescola(accA3, 1.0, 1.0)
prova("mescola: peso_kw = 1 identico al bit al miscuglio di prima; peso_kw = 3 = Kw contato 3 volte dentro H_K",
      torch.equal(H1a, H1b) and torch.allclose(H1a, H1v, rtol=1e-13, atol=0) and torch.allclose(H3a, H3v, rtol=1e-13, atol=0)
      and not torch.allclose(H1a, H3a))
Qg = Wd + 0.003 * torch.randn(Wd.shape, generator=g, dtype=F64)
Hg = accB["M"][0] / accB["M"][2]
prova("errore_sim = ||(W-Q)x||^2; rapporto_energia = ||Qx||/||Wx||; energia_ok rifiuta fuori [0,9; 1,1]",
      abs(RS.errore_sim(Qg, Wd, Hg) - float((((Wd - Qg) @ Hg) * (Wd - Qg)).sum())) < 1e-15
      and abs(RS.rapporto_energia(0.5 * Wd, Hg, float(((Wd @ Hg) * Wd).sum())) - 0.5) < 1e-12
      and not RS.energia_ok(0.5, 0.9, 1.1) and RS.energia_ok(1.0, 0.9, 1.1) and not RS.energia_ok(float("nan"), 0.9, 1.1)
      and RS.energia_ok(0.899, 0.9, 1.1, 0.884) and not RS.energia_ok(0.88, 0.9, 1.1, 0.884)
      and not RS.energia_ok(0.6, 0.9, 1.1, 0.95))
Qb = torch.stack([Qg, 0.6 * Wd]); Wb = torch.stack([Wd, Wd]); Hb_ = torch.stack([Hg, Hg])
prova("J_sim_lotto / N_lotto per riga coerenti con errore_sim / energia",
      abs(float(RS.J_sim_lotto(Qb, Hb_, Wb)[0].sum()) - RS.errore_sim(Qg, Wd, Hg)) < 1e-12
      and abs(float((RS.N_lotto(Qb, Hb_)[1].sum() / RS.N_lotto(Wb, Hb_)[1].sum()).sqrt()) - 0.6) < 1e-12)


def acc_rho(rho):
    """x_q = rho x_f + rumore indipendente (correlazione maestro/studente < 1): il bersaglio QEP a = 1 attenua W di ~rho."""
    out = {}
    for s_n, n in (("M", 1500), ("Kv", 1200), ("Kw", 900)):
        Xf_ = torch.randn(n, Cd, generator=g, dtype=F64) * torch.linspace(0.3, 2, Cd, dtype=F64)
        Xq_ = rho * Xf_ + (1 - rho ** 2) ** 0.5 * torch.randn(n, Cd, generator=g, dtype=F64) * torch.linspace(0.3, 2, Cd, dtype=F64)
        out[s_n] = (Xq_.T @ Xq_, Xf_.T @ Xq_, n)
    return out


aA, aB = acc_rho(0.55), acc_rho(0.55)
HmR, _, _ = RS.mescola(aA, 1.0)
T0r, S0r, _ = SO.gptq_bilanciato(K.ruota(Wd), norm(K.ruota_H(HmR)), 1, 8)
opz_r = {**opz, "alfe": [0.0, 0.5, 1.0], "drq": False}
rr_ = RS.risolvi_denso("blk.0.attn_gate.weight", "attn_gate", 45, Wd, aA, aB, T0r[0], SO.fp16(S0r[0]), opz_r)
cr = rr_["registro"]["candidati"]
vecchia = min(cr, key=lambda k: cr[k]["JB_asim"])       # la scelta della guardia di prima (obiettivo asimmetrico)
prova("rho = 0,55: la guardia asimmetrica di prima sceglierebbe un QEP a > 0 con energia < 0,9 (attenuato); la nuova lo "
      "RIFIUTA: scelta con energia in [0,9; 1,1] e errore simmetrico <= partenza",
      vecchia in ("qep0.5", "qep1") and cr[vecchia]["energia"] < 0.9 and not cr[vecchia]["valido"]
      and cr["qep1"]["energia"] < 0.9 and rr_["scelto"] not in ("qep0.5", "qep1") and (rr_["scelto"] == "partenza" or 0.9 <= cr[rr_["scelto"]]["energia"] <= 1.1)
      and cr[rr_["scelto"]]["sim_B"] <= cr["partenza"]["sim_B"],
      f"asim {[round(cr[k]['JB_asim'],4) for k in cr]} sim {[round(cr[k]['sim_B'],4) for k in cr]} energia qep0/0,5/1 {cr['qep0']['energia']:.3f}/{cr['qep0.5']['energia']:.3f}/{cr['qep1']['energia']:.3f}; scelto {rr_['scelto']}")

# 14/10 G1/G2 esperti: risolvi_esperti_gu / risolvi_esperti_down eseguiti davvero (CPU, dati finti, rho = 0,6)
import tempfile  # noqa: E402
Ee, Ie, He_, Ne = 3, 256, 1024, 2400
gE = torch.Generator().manual_seed(20261014)
Wgu_e = (torch.randn(Ee, 2 * Ie, He_, generator=gE) * 0.02).float()
Wd_e = (torch.randn(Ee, He_, Ie, generator=gE) * 0.02).float()
sc_e = torch.linspace(0.3, 2, He_, dtype=F64)
acc3_e = SP.AccDensi(He_, torch.device("cpu"))
dl_e = Path(tempfile.mkdtemp(prefix="s23_test_esperti_"))
adn_e = SP.AccDown(Ee, Ie, torch.device("cpu"))
for h in "AB":
    agu_e = SP.AccGateUp(Ee, Ie, He_, torch.device("cpu"))
    Xf_e = torch.randn(Ne, He_, generator=gE, dtype=F64) * sc_e
    Xq_e = 0.6 * Xf_e + 0.8 * torch.randn(Ne, He_, generator=gE, dtype=F64) * sc_e
    for i_, s_ in enumerate(SC.SORGENTI):
        sl_ = slice(i_ * Ne // 3, (i_ + 1) * Ne // 3)
        acc3_e.add(h, s_, Xq_e[sl_], Xf_e[sl_])
    esp_e = torch.randint(0, Ee, (Ne,), generator=gE)
    ordine_e = torch.argsort(esp_e, stable=True)
    offs_e = torch.cat([torch.zeros(1, dtype=torch.long), torch.bincount(esp_e, minlength=Ee).cumsum(0)])
    w_e = torch.ones(Ne, 1)
    agu_e.add(Xq_e.float(), Xf_e.float(), w_e, 1, ordine_e, offs_e, Wgu_e)
    agu_e.scarica(dl_e / f"gu_{h}")
    Af_e = torch.randn(Ne, Ie, generator=gE, dtype=F64)
    Aq_e = 0.6 * Af_e + 0.8 * torch.randn(Ne, Ie, generator=gE, dtype=F64)
    for e in range(Ee):
        m_ = esp_e == e
        adn_e.Hd[h][e] += (Aq_e[m_].T @ Aq_e[m_]).float()
        adn_e.Md[h][e] += (Af_e[m_].T @ Aq_e[m_]).float()
        adn_e.conta[h][e] += int(m_.sum())
        adn_e.peso[h][e] += float(m_.sum())


def _q_semplice(Wr, gr):
    R_, C_ = Wr.shape
    Wg = Wr.reshape(R_, C_ // gr, gr)
    Sg = Wg.abs().mean(-1)
    Tg = torch.where(Wg.abs() > 0.5 * Sg.unsqueeze(-1), Wg.sign(), torch.zeros_like(Wg))
    return Tg.reshape(R_, C_).to(torch.int8), AL.fp16(Sg.double())


def codifica_e(T_, S_, tid_, C_):
    return K.gp.encode_rows(T_.cpu().numpy().astype(np.int8), S_.cpu().to(torch.float16).numpy(), tid_)


def decodifica_e(raw_rows, tid_, C_):
    t_, d_ = K.gp.decode_rows(np.ascontiguousarray(raw_rows), tid_, C_)
    return torch.from_numpy(t_.astype(np.int8)), torch.from_numpy(d_.astype(np.float64))


raw_e = {}
for f_, q_ in (("gate", 0), ("up", 1)):
    rr_ = []
    for e in range(Ee):
        T_, S_ = _q_semplice(K.ruota(Wgu_e[e, q_ * Ie:(q_ + 1) * Ie].double()), 128)
        rr_.append(codifica_e(T_, S_, 45, He_))
    raw_e[f"blk.0.ffn_{f_}_exps.weight"] = np.stack(rr_)
rr_ = []
for e in range(Ee):
    T_, S_ = _q_semplice(Wd_e[e].double(), 256)
    rr_.append(codifica_e(T_, S_, 34, Ie))
raw_e["blk.0.ffn_down_exps.weight"] = np.stack(rr_)
nuovi_e = {}
gd_e = {"emin": 0.9, "emax": 1.1, "peso_kw": 1.0}
rgu = SP.risolvi_esperti_gu(0, dl_e, acc3_e, Wgu_e, nuovi_e, lambda *x: None, codifica_e, decodifica_e, lambda n: raw_e[n],
                            list(range(Ee)), [0.0, 0.5, 1.0], 8000.0, 2, CP.jarq_lotto, torch.device("cpu"), F64, Ee, Ie, He_, gd_e)
rdn = SP.risolvi_esperti_down(0, adn_e, Wd_e, nuovi_e, lambda *x: None, codifica_e, decodifica_e, lambda n: raw_e[n],
                              list(range(Ee)), [0.0, 0.5], 8000.0, 2, CP.jarq_lotto, torch.device("cpu"), F64, Ee, Ie, He_, gd_e)
en_ok = all(0.9 <= x <= 1.1 for x in [v for v in rgu["energia_mediana_scelti"].values() if v is not None]
            + ([rdn["energia_mediana_scelti"]] if rdn["energia_mediana_scelti"] is not None else []))
prova("esperti (rho 0,6): gate/up e down girano coi candidati a, accettati solo con energia in [0,9; 1,1] e errore simmetrico "
      "piu' basso (dSim mediana <= 0), mai a = 1 attenuato",
      en_ok and rgu["accettati_gate"] + rgu["accettati_up"] > 0 and rdn["accettati"] > 0
      and all(v is None or v <= 0 for v in rgu["dJB_rel_mediana"].values()) and rdn["dJB_rel_mediana"] <= 0
      and "1" not in rgu["alfa_scelti"]["gate"] and "1" not in rgu["alfa_scelti"]["up"],
      f"gate/up {rgu['accettati_gate']}/{rgu['accettati_up']} alfa {rgu['alfa_scelti']} en {rgu['energia_mediana_scelti']} "
      f"part {rgu['energia_mediana_partenza']}; down {rdn['accettati']} alfa {rdn['alfa_scelti']} en {rdn['energia_mediana_scelti']}")
import shutil as _sh  # noqa: E402
_sh.rmtree(dl_e, ignore_errors=True)

print(f"{sum(esiti)}/{len(esiti)} verdi", flush=True)
sys.exit(0 if all(esiti) else 1)
