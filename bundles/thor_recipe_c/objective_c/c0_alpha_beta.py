#!/usr/bin/env python3
"""Obiettivo C (9/10/2026): ssm_alpha / ssm_beta dei 30 strati GDN ternari a UN piano (TQ1_0R_1024).
In R2B3m (= TT2 p6 per tutto cio' che non e' piano 2/3) sono F32 [32, 2048] allenati da TT2: bersaglio = quei valori.
Hessiana = H dell'ingresso dello strato (stesso ingresso di attn_qkv/attn_gate: uscita di attn_norm), hessiane-attn/H_LL,
la stessa del manifesto R2B3 per attn_qkv. Ricetta = tutto_ternario/tt_quant.quantizza_1024 (GPTQ + coordinate descent
esatta trit/scale congiunte pesate H + scale fp16 + ultimo giro di trit). SOLO CPU.
Uscita: <out>/blk.L.ssm_{alpha,beta}.weight.npy (uint8 [32, 448]) + manifesto_alpha_beta.json.
"""
import hashlib, json, os, sys, time
os.environ.setdefault("TT_DEVICE", "cpu")
sys.path.insert(0, "@@ROOT:depot@@/tutto_ternario/codice")
sys.path.insert(0, "@@ROOT:models@@/build/thor-rexppiani2/gguf-py")
import numpy as np
from gguf import GGUFReader
import tt_quant as TQ

BASE = "@@ROOT:models@@/gguf/tony-tern/R2B3m/Thor-T-R2B3m.gguf"
OUT = sys.argv[1] if len(sys.argv) > 1 else "@@ROOT:models@@/gguf/tony-tern/OBIETTIVO_C/C0/tensori"
HD = "@@ROOT:models@@/gguf/tony-tern/hessiane-attn"

r = GGUFReader(BASE)
man = {}
for t in r.tensors:
    if not (t.name.endswith(".ssm_alpha.weight") or t.name.endswith(".ssm_beta.weight")):
        continue
    L = int(t.name.split(".")[1])
    p = os.path.join(OUT, t.name + ".npy")
    if os.path.exists(p):
        continue
    W = np.array(t.data, dtype=np.float32).reshape(int(t.shape[1]), int(t.shape[0]))   # (32, 2048) = (R, C)
    hf = f"{HD}/H_{L:02d}.npy"
    H = np.load(hf).astype(np.float64)
    packed, mis = TQ.quantizza_1024(W, H, t.name)
    arr = packed.reshape(W.shape[0], -1)
    np.save(p, arr)
    man[t.name] = {"tipo": "TQ1_0R_1024", "forma_ggml": [int(t.shape[0]), int(t.shape[1])], "bersaglio": "F32 di R2B3m (TT2 p6)",
                   "hessiana": hf, "misure": mis, "sha256_npy": hashlib.sha256(arr.tobytes()).hexdigest()}
    print(time.strftime("%H:%M:%S"), t.name, {k: mis[k] for k in ("err_rel", "err_rel_pesato_H", "frazione_zeri", "secondi")}, flush=True)
mp = os.path.join(OUT, "manifesto_alpha_beta.json")
old = json.load(open(mp)) if os.path.exists(mp) else {}
old.update(man)
json.dump(old, open(mp, "w"), indent=1)
print("fatti", len(old))
