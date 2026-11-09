"""KL al completo (vocabolario intero) e KL ristretta alle 4 lettere (rinormalizzata) nelle posizioni di risposta del
banco zero-shot, base e K01 contro il maestro; stessa misura sulle posizioni di risposta dei dati di calibrazione."""
import sys, json, numpy as np
sys.path.insert(0, "@@ROOT:forge@@/thor_v2")
from e2e_dump_logit_leggi import leggi_tensore
import crollo_k01 as CK
LET = [357, 417, 351, 414]
def lsm(x):
    x = x.astype(np.float64); x = x - x.max(-1, keepdims=True); return x - np.log(np.exp(x).sum(-1, keepdims=True))
def misure(lm, lb, lk, giuste):
    out = {}
    pm = np.exp(lm)
    for n, ls in (("base", lb), ("k01", lk)):
        kl = (pm * (lm - ls)).sum(1)
        a, b = lm[:, LET], ls[:, LET]
        a4, b4 = a - np.log(np.exp(a).sum(1, keepdims=True)), b - np.log(np.exp(b).sum(1, keepdims=True))
        kl4 = (np.exp(a4) * (a4 - b4)).sum(1)
        out[n] = {"kl_vocabolario": float(kl.mean()), "kl_4_lettere": float(kl4.mean()),
                  "acc": float((b.argmax(1) == giuste).mean()), "accordo_col_maestro": float((b.argmax(1) == a.argmax(1)).mean()),
                  "massa_lettere": float(np.median(np.exp(b).sum(1)))}
    out["maestro"] = {"acc": float((lm[:, LET].argmax(1) == giuste).mean()), "massa_lettere": float(np.median(np.exp(lm[:, LET]).sum(1)))}
    return out
# banco zero-shot
zs = json.load(open("token/zero_shot.json"))
M = {m: [] for m in ("maestro", "base", "k01")}; G = []
for j, z in enumerate(zs):
    ult = [c["ultima"] for c in z["compiti"]]; G += [c["giusta"] for c in z["compiti"]]
    for m in M:
        M[m].append(lsm(np.asarray(leggi_tensore(f"zero_{m}/seq_{j:04d}/logit.f32"))[ult]))
r = {"banco_zero_shot": misure(*(np.concatenate(M[m]) for m in ("maestro", "base", "k01")), np.array(G))}
# calibrazione (few-shot impacchettato): posizioni di risposta delle 4 sequenze
info = json.load(open("token/token.json"))
for gruppo, sel in (("calibrazione_mc_heldout", ["ho_mc"]), ("calibrazione_mc_addestramento", ["tr_mc"]), ("conoscenza_mmlu_val_impacchettata", ["con"])):
    M = {m: [] for m in ("maestro", "base", "k01")}; G = []
    for i, s in enumerate(info["sequenze"]):
        if s["nome"] not in sel: continue
        ids = np.fromfile(f"token/{s['nome']}.i32", dtype=np.int32); pr = CK.posizioni_risposta(ids)
        G += [LET.index(int(ids[t + 1])) for t in pr]
        for m in M:
            M[m].append(lsm(np.asarray(leggi_tensore(f"llama_{m}/seq_{i:04d}/logit.f32"))[pr]))
    r[gruppo] = misure(*(np.concatenate(M[m]) for m in ("maestro", "base", "k01")), np.array(G)); r[gruppo]["n"] = len(G)
r["banco_zero_shot"]["n"] = len(zs[0]["compiti"]) + len(zs[1]["compiti"])
print(json.dumps(r, indent=1))
json.dump(r, open("esito_zero3.json", "w"), indent=1)
