"""CARDAN-DENSI-KL4 (10/10): KL4 on the HELD-OUT calibration questions (those NOT used by --kl4-n in training:
same split as gradienti_densi.pacchi_kl4 -> filter on teacher MCLP, Random(7).shuffle, first N = used), paired
candidate - reference with z and win/loss. usage: kl4_heldout.py TEACHER.log CALIB.jsonl N REF.log CAND.log"""
import json, math, random, sys
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
Q = [json.loads(x) for x in open(sys.argv[2])]
Q = [q for q in Q if q["i"] in T]
random.Random(7).shuffle(Q)
usate = {q["i"] for q in Q[: int(sys.argv[3])]}
held = [q["i"] for q in Q if q["i"] not in usate]
R, K = mclp(sys.argv[4]), mclp(sys.argv[5])
ids = [i for i in held if i in R and i in K]
kl = lambda M, i: float((np.exp(lp4(T[i][2])) * (lp4(T[i][2]) - lp4(M[i][2]))).sum())
d = np.array([kl(K, i) - kl(R, i) for i in ids])
z = d.mean() / (d.std(ddof=1) / math.sqrt(len(d)))
okc = np.array([K[i][1] == K[i][0] for i in ids]); okr = np.array([R[i][1] == R[i][0] for i in ids])
print(f"held-out n {len(ids)}: KL4 rif {np.mean([kl(R, i) for i in ids]):.5f} cand {np.mean([kl(K, i) for i in ids]):.5f} "
      f"dKL4 {d.mean():+.5f} z {z:+.1f} acc {okr.mean()*100:.2f} -> {okc.mean()*100:.2f} vinte/perse "
      f"{int((okc & ~okr).sum())}/{int((okr & ~okc).sum())}")
print(f"DKL4 {d.mean():.6f}")
