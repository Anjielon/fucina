"""SELRISP (10/10): KL4 on the calibration set (teacher q8 MCLP vs candidate MCLP), split in the questions used for the
gradients (kl4_torch.tsv of the gradient run) and the HELD-OUT rest. Paired difference candidate - reference with z.
usage: kl4_calib.py TEACHER.log GRAD_DIR REF.log [CAND.log ...]"""
import math, sys
import numpy as np


def mclp(p):
    out = {}
    for r in open(p, errors="replace"):
        i = r.find("MCLP\t")
        if i < 0:
            continue
        c = r[i:].rstrip("\n").split("\t"); n = int(c[4])
        out[int(c[1])] = (int(c[2]), int(c[3]), np.array([float(x) for x in c[5:5 + n]]))
    return out


def lp4(x):
    x = x - x.max(); return x - np.log(np.exp(x).sum())


T = mclp(sys.argv[1])
usate = {int(r.split("\t")[0]) for r in list(open(f"{sys.argv[2]}/kl4_torch.tsv"))[1:]}


def kl(p):
    R = mclp(p); ks = {}
    for i, (g, s, lp) in R.items():
        if i in T:
            lt = lp4(T[i][2]); ks[i] = (float((np.exp(lt) * (lt - lp4(lp))).sum()), s == g)
    return ks


rif = kl(sys.argv[3])
for parte, filtro in (("held-out", lambda i: i not in usate), ("usate", lambda i: i in usate), ("tutte", lambda i: True)):
    ids = sorted(i for i in rif if filtro(i))
    kr = np.array([rif[i][0] for i in ids]); ar = np.mean([rif[i][1] for i in ids]) * 100
    print(f"[{parte}] rif {sys.argv[3]}: n {len(ids)} acc {ar:.2f} KL4 {kr.mean():.5f}")
    for c in sys.argv[4:]:
        K = kl(c); ids2 = [i for i in ids if i in K]
        d = np.array([K[i][0] - rif[i][0] for i in ids2]); z = d.mean() / (d.std(ddof=1) / math.sqrt(len(d)))
        ok = np.array([K[i][1] for i in ids2]); okr = np.array([rif[i][1] for i in ids2])
        print(f"[{parte}] {c}: acc {ok.mean()*100:.2f} KL4 {kr.mean()+d.mean():.5f} dKL4 {d.mean():+.5f} z {z:+.1f} "
              f"vinte/perse {int((ok & ~okr).sum())}/{int((okr & ~ok).sum())}")
