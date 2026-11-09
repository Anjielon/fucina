import sys, json, numpy as np
sys.path.insert(0, "@@ROOT:forge@@/thor_v2")
from e2e_dump_logit_leggi import leggi_tensore
L = [357, 417, 351, 414]
zs = json.load(open("token/zero_shot.json"))
def mclp(fn):
    R = {}
    for r in open(fn, errors='ignore'):
        if r.startswith('MCLP\t'):
            q = r.rstrip('\n').split('\t'); n=int(q[4]); R[int(q[1])] = (int(q[2]), int(q[3]), [float(x) for x in q[5:5+n]])
    return R
A = '@@ROOT:models@@/gguf/tony-tern/mmlu_logprob/'
dell = {"maestro": mclp(A + 'teacher.log'), "base": mclp(A + 'aragorn/EB2-migliore_ema_000016.log'),
        "k01": mclp(A + 'aragorn/rifint-R2039d4v-K01.log')}
modelli = [m for m in sys.argv[1].split(",")]
ris = {m: [] for m in modelli}
for j, z in enumerate(zs):
    for m in modelli:
        lg = np.asarray(leggi_tensore(f"zero_{m}/seq_{j:04d}/logit.f32"))
        for c in z["compiti"]:
            x = lg[c["ultima"]].astype(np.float64); x = x - x.max(); ls = x - np.log(np.exp(x).sum())
            lp = ls[L]
            d = dell[m][c["compito"]]
            ris[m].append({"compito": c["compito"], "tipo": c["tipo"], "giusta": c["giusta"], "scelta_cpu": int(lp.argmax()),
                           "scelta_banco": d[1], "lp_cpu": lp.round(3).tolist(), "lp_banco": [round(v, 3) for v in d[2]]})
for m in modelli:
    r = ris[m]
    acc = np.mean([x["scelta_cpu"] == x["giusta"] for x in r]); accb = np.mean([x["scelta_banco"] == x["giusta"] for x in r])
    ug = np.mean([x["scelta_cpu"] == x["scelta_banco"] for x in r])
    dlp = np.array([np.array(x["lp_cpu"]) - np.array(x["lp_banco"]) for x in r])
    lpc = np.array([x["lp_cpu"] for x in r])
    print(f"{m:8s} n={len(r)} acc cpu {acc:.3f} banco {accb:.3f} · scelte uguali cpu/banco {ug:.3f} · |Δlogp| cpu-banco medio {np.abs(dlp).mean():.3f} max {np.abs(dlp).max():.3f} · scelte cpu {np.bincount([x['scelta_cpu'] for x in r],minlength=4)} · logp medio A..D {lpc.mean(0).round(2)} · massa4 mediana {np.median(np.exp(lpc).sum(1)):.3f}")
json.dump(ris, open("esito_zero.json", "w"), indent=1)
