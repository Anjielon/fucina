#!/usr/bin/env python3
"""Test (f) dell'audit (B3/B4) su DATI VERI, CPU, sottoinsieme di esperti.

Errore relativo sui pesi rispetto all'originale bf16 (spazio ruotato per
gate/up, come nel formato TQ1_0R_1024; spazio reale per down), per:
  (i)   la v20 stessa (dequantizzata dal GGUF)
  (ii)  il VECCHIO punto di partenza: codice duro di TernarioDiffBatch
        costruito sulla v20 (s0=media|W|, delta=0.5)
  (iii) il NUOVO punto di partenza: init LS dai pesi originali
Cancello: (iii) < (i) e (iii) < (ii); disaccordo codice duro vs LS ~0.
Conta anche le righe gate+up entrambe nulle nella v20 (neuroni morti).
Uso: test_init_ls_dati_veri.py [strato] [n_esperti]"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.set_num_threads(4)   # una corsia pesante per volta: la forgia di controllo sta girando

from safetensors import safe_open  # noqa: E402
import json  # noqa: E402
from carica_strato_gguf import pesi_strato_da_gguf  # noqa: E402
from init_ls import reinit_ls_  # noqa: E402
from quant_ternaria_globale_batch_v2 import TernarioDiffBatchFuso  # noqa: E402
from rotazione_gpu import _R_1024_torch, S1024  # noqa: E402

SRC = "@@ROOT:models@@/gguf/thor-src-hf"
V20 = "@@ROOT:models@@/gguf/tony-tern/Thor-v20-downaggr3.gguf"
L = int(sys.argv[1]) if len(sys.argv) > 1 else 0
N_ESP = int(sys.argv[2]) if len(sys.argv) > 2 else 32
PASSO = 256 // N_ESP
S = torch.from_numpy(S1024).float()


def rel(a, b):
    return ((a - b).norm() / b.norm()).item()


def duro(t):
    with torch.no_grad():
        dlt = t._delta_norm()
        q = torch.where(t.z > dlt, 1.0, torch.where(t.z < -dlt, -1.0, 0.0))
        return (t.log_s.exp() * q).reshape(t.shape)


idx = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]
pref = f"model.language_model.layers.{L}.mlp.experts."
teacher = {}
for nome in ("gate_up_proj", "down_proj"):
    with safe_open(f"{SRC}/{idx[pref + nome]}", "pt") as f:
        teacher[nome] = f.get_tensor(pref + nome)[::PASSO].float()
v20_tutto = pesi_strato_da_gguf(V20, L)
v20 = {n: v20_tutto[f"mlp.experts.{n}"][::PASSO].float().clone() for n in ("gate_up_proj", "down_proj")}
del v20_tutto

meta = teacher["gate_up_proj"].shape[1] // 2
def _morti(X):   # |w|max < 1e-20: l'originale ha valori ~1e-37, non zero esatto
    return ((X[:, :meta].abs().amax(-1) < 1e-20) & (X[:, meta:].abs().amax(-1) < 1e-20)).float().mean().item()
morti, morti_teacher = _morti(v20["gate_up_proj"]), _morti(teacher["gate_up_proj"])
print(f"strato {L}, {teacher['gate_up_proj'].shape[0]} esperti: neuroni morti (gate e up nulli) "
      f"v20 {morti*100:.1f}% · originale {morti_teacher*100:.1f}%")

ok = True
for nome, group, ruota in (("gate_up_proj", 128, True), ("down_proj", 256, False)):
    T = _R_1024_torch(teacher[nome], S) if ruota else teacher[nome]
    V = _R_1024_torch(v20[nome], S) if ruota else v20[nome]
    e_v20 = rel(V, T)
    vecchio = TernarioDiffBatchFuso(V.clone(), group=group)
    e_vecchio = rel(duro(vecchio), T)
    nuovo = TernarioDiffBatchFuso(T.clone(), group=group)
    diag = reinit_ls_(nuovo, T)
    e_nuovo = rel(duro(nuovo), T)
    # gate/up della v20 sono gia' ternari: confronto alla pari. Il down della v20
    # e' a 4 bit (IQ4_XS/Q4_K): lo batte per costruzione, vale solo "nuovo < vecchio".
    esito = (e_nuovo < e_vecchio and diag["frazione_disaccordo_ls"] < 1e-3
             and (e_nuovo < e_v20 if ruota else True))
    ok &= esito
    print(f"  {nome}: errore vs originale — v20 {e_v20:.4f} · vecchio init {e_vecchio:.4f} · "
          f"NUOVO init LS {e_nuovo:.4f} · disaccordo duro/LS {diag['frazione_disaccordo_ls']*100:.4f}% "
          f"{'✅' if esito else '⛔'}")
print("✅ INIT LS DALL'ORIGINALE: MIGLIORE DI v20 E DEL VECCHIO INIT" if ok else "⛔ CANCELLO NON SUPERATO")
raise SystemExit(0 if ok else 1)
