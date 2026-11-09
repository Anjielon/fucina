"""Riassunto per posizione dei logit (logp delle 4 lettere, KL contro il maestro, NLL del token dopo) prima di cancellare
i logit completi (1 GB a sequenza): il disco di @@ROOT:depot@@ serve alle istantanee di EB3."""
import sys, json, numpy as np
sys.path.insert(0, "@@ROOT:forge@@/thor_v2")
from e2e_dump_logit_leggi import leggi_tensore
LET = [357, 417, 351, 414]
def lsm(x):
    x = x.astype(np.float64); x = x - x.max(-1, keepdims=True); return x - np.log(np.exp(x).sum(-1, keepdims=True))
for pref, n, fid in (("llama", 4, lambda i: json.load(open("token/token.json"))["sequenze"][i]["nome"]),
                     ("zero", 2, lambda i: f"zs{i}")):
    for i in range(n):
        ids = np.fromfile(f"token/{fid(i)}.i32", dtype=np.int32)
        nxt = np.append(ids, 0).astype(np.int64)[1:]
        out = {}
        lg = {m: leggi_tensore(f"{pref}_{m}/seq_{i:04d}/logit.f32") for m in ("maestro", "base", "k01")}
        for m in lg: out[f"lettere_{m}"] = np.zeros((len(ids), 4), np.float32); out[f"nll_{m}"] = np.zeros(len(ids), np.float32)
        for m in ("base", "k01"): out[f"kl_{m}"] = np.zeros(len(ids), np.float32)
        for a in range(0, len(ids), 64):
            b = min(len(ids), a + 64)
            lm = lsm(np.asarray(lg["maestro"][a:b])); pm = np.exp(lm)
            for m in lg:
                ls = lm if m == "maestro" else lsm(np.asarray(lg[m][a:b]))
                out[f"lettere_{m}"][a:b] = ls[:, LET]; out[f"nll_{m}"][a:b] = -ls[np.arange(b - a), nxt[a:b]]
                if m != "maestro": out[f"kl_{m}"][a:b] = (pm * (lm - ls)).sum(1)
        np.savez(f"riassunto_logit_{pref}_{fid(i)}.npz", **out)
        print(pref, fid(i), "ok", flush=True)
