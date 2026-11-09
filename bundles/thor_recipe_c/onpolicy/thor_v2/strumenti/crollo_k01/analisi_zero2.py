"""Banco zero-shot (24 compiti MMLU del Dell, impacchettati con --segmenti): errore del flusso contro il maestro negli
strati 19-39 alla posizione di risposta e sulle altre posizioni dei compiti, lente sulle lettere, scelte."""
import sys, json, numpy as np
sys.path.insert(0, "@@ROOT:forge@@/thor_v2"); sys.path.insert(1, "@@ROOT:models@@/build/llamacpp-tq1-k1024/gguf-py")
sys.path.insert(2, "@@ROOT:depot@@/e2e/codice/riforgia-9d50452/thor_v2")
from e2e_dump_logit_leggi import leggi_tensore
from gguf import GGUFReader
from dequant_gguf_torch import _dequant_raw
LET = [357, 417, 351, 414]
zs = json.load(open("token/zero_shot.json"))
t = {x.name: x for x in GGUFReader("@@ROOT:models@@/gguf/tony-tern/Thor-T-diagnosi-ema16.gguf").tensors}
W = _dequant_raw(np.asarray(t["output.weight"].data)[LET], t["output.weight"]).astype(np.float64)
g = np.asarray(t["output_norm.weight"].data, dtype=np.float64)
def lett(h):
    h = h.astype(np.float64); return (h / np.sqrt((h ** 2).mean(-1, keepdims=True) + 1e-6) * g) @ W.T
def ll(m, j, n): return np.array(leggi_tensore(f"zero_{m}/seq_{j:04d}/{n}.f32"), dtype=np.float32)
STRATI = [19, 20, 21, 22, 23, 25, 27, 29, 31, 33, 35, 37, 39]
ris = {"per_strato": {}, "lente": {}, "compiti": []}
acc = {L: {"base_r": [], "k01_r": [], "base_q": [], "k01_q": [], "kb_r": [], "hb_r": []} for L in STRATI}
lens = {L: [] for L in STRATI}
for j, z in enumerate(zs):
    ult = np.array([c["ultima"] for c in z["compiti"]])
    seg = np.fromfile(f"token/{z['file']}_seg.i32", dtype=np.int32)
    inizi = np.where(seg == 1)[0]
    # posizioni "domanda": tutte quelle dei compiti tranne l'ultima di ciascuno (esclude il riempitivo finale)
    dom = np.concatenate([np.arange(a, u) for a, u in zip(inizi[:len(ult)], ult)])
    fin_b = ll("base", j, "l_out-39")[ult]
    base_l = lett(fin_b)
    for L in STRATI:
        hm, hb, hk = ll("maestro", j, f"l_out-{L}"), ll("base", j, f"l_out-{L}"), ll("k01", j, f"l_out-{L}")
        a = acc[L]
        a["base_r"].append(((hb[ult] - hm[ult]) ** 2).sum(1)); a["k01_r"].append(((hk[ult] - hm[ult]) ** 2).sum(1))
        a["base_q"].append(((hb[dom] - hm[dom]) ** 2).sum(1)); a["k01_q"].append(((hk[dom] - hm[dom]) ** 2).sum(1))
        a["kb_r"].append(((hk[ult] - hb[ult]) ** 2).sum(1)); a["hb_r"].append((hm[ult] ** 2).sum(1))
        d = hk[ult] - hb[ult]
        lens[L].append(lett(fin_b + d) - base_l)
    for c, lb, lk in zip(z["compiti"], ll("base", j, "l_out-39")[ult], ll("k01", j, "l_out-39")[ult]):
        pass
for L in STRATI:
    a = {k: np.concatenate(v) for k, v in acc[L].items()}
    ris["per_strato"][L] = {
        "risposta_err_rel_base": float(np.sqrt(a["base_r"].sum() / a["hb_r"].sum())),
        "risposta_err_rel_k01": float(np.sqrt(a["k01_r"].sum() / a["hb_r"].sum())),
        "risposta_mse_k01_su_base": float(a["k01_r"].sum() / a["base_r"].sum()),
        "risposta_compiti_k01_peggio": int((a["k01_r"] > a["base_r"]).sum()),
        "domanda_mse_k01_su_base": float(a["k01_q"].sum() / a["base_q"].sum()),
        "risposta_k01_vs_base_rel": float(np.sqrt(a["kb_r"].sum() / a["hb_r"].sum()))}
    dl = np.concatenate(lens[L])
    ris["lente"][L] = {"delta_logit_medio_A_D": dl.mean(0).round(3).tolist(),
                       "delta_BC_meno_AD": float((dl[:, 1] + dl[:, 2] - dl[:, 0] - dl[:, 3]).mean() / 2)}
    print(L, json.dumps(ris["per_strato"][L]), "lente", ris["lente"][L])
json.dump(ris, open("esito_zero2.json", "w"), indent=1)
