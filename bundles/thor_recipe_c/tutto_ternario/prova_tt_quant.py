"""Prove CPU del quantizzatore (28/9), prima di usarlo sui tensori veri. Ogni prova sa FALLIRE (controllo negativo).
 P1 scala_esatta == forza bruta su tutti i supporti (g=8) e batte la griglia storica a 6 soglie.
 P2 ternario_esatto: nessun (d16, t) vicino fa meglio (perturbazioni di un trit / della scala).
 P3 quantizza_1024 con H: catena non crescente (GPTQ -> round -> fp16 -> trit finali), round-trip dai byte == 0,
    pesato-H migliore del non pesato; negativo: con un segno S sbagliato il round-trip NON torna.
 P4 quantizza_tq1_0: il dequantizzatore indipendente di gguf-py ridà esattamente t*d16.
"""
from __future__ import annotations

import itertools
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "@@ROOT:models@@/build/llamacpp-tq1-k1024/gguf-py")
import tt_quant as TQ  # noqa: E402
import gptq_1024_lib as G  # noqa: E402

rng = np.random.default_rng(0)
esito = {}

# P1
X = rng.standard_normal((200, 8)).astype(np.float32) * rng.uniform(0.1, 3, (200, 1)).astype(np.float32)
d = TQ.scala_esatta(torch.from_numpy(X)).numpy()[:, 0].astype(np.float64)
e_es = ((X - d[:, None] * np.sign(X) * (np.abs(X) >= 0)) ** 2)  # placeholder non usato
best = np.full(200, np.inf)
for t in itertools.product((-1, 0, 1), repeat=8):
    t = np.array(t, dtype=np.float64)
    n = (t * t).sum()
    if n == 0:
        e = (X.astype(np.float64) ** 2).sum(1)
    else:
        dd = (X @ t) / n
        e = ((X - dd[:, None] * t) ** 2).sum(1)
    best = np.minimum(best, e)
ab = np.abs(X.astype(np.float64))
k = np.argsort(-ab, 1)
e_es = np.empty(200)
for i in range(200):
    s = np.sort(ab[i])[::-1]; cs = np.cumsum(s); kk = int(np.argmax(cs * cs / np.arange(1, 9)))
    e_es[i] = (ab[i] ** 2).sum() - cs[kk] ** 2 / (kk + 1)
    assert abs(d[i] - cs[kk] / (kk + 1)) < 1e-6
gr = G.__dict__.get("_scala_un_gruppo_orig")
esito["P1_esatta_eq_bruta"] = bool(np.allclose(e_es, best, rtol=1e-9, atol=1e-9))
# controllo negativo: la griglia storica (6 soglie) NON e' esatta su questo campione
from tq1_0r_1024_codec import scala_ls  # noqa: E402
Xg = rng.standard_normal((500, 1, 128)).astype(np.float32)
dg = scala_ls(Xg)[:, 0].astype(np.float64)
tg = np.clip(np.round(Xg[:, 0] / np.where(dg == 0, 1, dg)[:, None]), -1, 1)
eg = ((Xg[:, 0] - dg[:, None] * tg) ** 2).sum(1)
de = TQ.scala_esatta(torch.from_numpy(Xg[:, 0])).numpy()[:, 0].astype(np.float64)
te = np.clip(TQ._lround(Xg[:, 0] / de[:, None]), -1, 1)
ee = ((Xg[:, 0] - de[:, None] * te) ** 2).sum(1)
esito["P1_esatta_mai_peggio_griglia"] = bool((ee <= eg + 1e-9).all())
esito["P1_griglia_peggiore_su_alcuni(negativo)"] = int((ee < eg - 1e-9).sum())

# P2
Xb = rng.standard_normal((300, 128)).astype(np.float32)
d16, t = TQ.ternario_esatto(Xb)
base = ((Xb - d16[:, None] * t) ** 2).sum(1)
peggio = 0
for j in range(0, 128, 7):
    for c in (-1, 0, 1):
        t2 = t.copy(); t2[:, j] = c
        peggio += int((((Xb - d16[:, None] * t2) ** 2).sum(1) < base - 1e-6).sum())
for f in (0.999, 1.001):
    peggio += int((((Xb - (d16 * f)[:, None] * t) ** 2).sum(1) < base - 1e-4 * base).sum())
esito["P2_nessun_vicino_migliore"] = peggio == 0
esito["P2_d16_esatto_fp16"] = bool((d16.astype(np.float16).astype(np.float32) == d16).all())

# P3
R, C = 96, 2048
A = rng.standard_normal((C, 3 * C)).astype(np.float64) * rng.uniform(0.2, 2, (C, 1))
H = A @ A.T / (3 * C)
W = (rng.standard_normal((R, C)) * 0.02).astype(np.float32)
TQ.N_ROUND = 3
packed, mis = TQ.quantizza_1024(W, H, "prova")
c = mis["loss_pesata_rel_catena"]
seq = [c["gptq"]] + [c[f"round_{k+1}"] for k in range(3)]
esito["P3_catena_non_crescente"] = all(b <= a + 1e-12 for a, b in zip(seq, seq[1:])) and c["finale"] <= c["scale_fp16"] + 1e-12
esito["P3_roundtrip_zero"] = mis["roundtrip_max_abs"] == 0.0
esito["P3_pesato_meglio_del_nonpesato"] = mis["err_rel_pesato_H"] < mis["rif_nonpesato_err_rel_pesato_H"]
esito["P3_numeri"] = {k: mis[k] for k in ("err_rel", "err_rel_pesato_H", "rif_nonpesato_err_rel", "rif_nonpesato_err_rel_pesato_H")} | {"catena": c}
# negativo: dequantizzazione con segno sbagliato
import tq1_0r_1024_codec as CO  # noqa: E402
S_ok = CO.S1024.copy()
CO.S1024[:] = -S_ok
Wd_neg = CO.dequantize_blocks(packed).reshape(R, C)
CO.S1024[:] = S_ok
Wd_ok = CO.dequantize_blocks(packed).reshape(R, C)
esito["P3_negativo_segno_sbagliato_diverso"] = float(np.abs(Wd_neg - Wd_ok).max()) > 1e-3
# P4
W4 = (rng.standard_normal((64, 512)) * 0.03).astype(np.float32)
p4, m4 = TQ.quantizza_tq1_0(W4, "prova4")
esito["P4_roundtrip_gguf_py_zero"] = m4["roundtrip_max_abs"] == 0.0
esito["P4_err_rel"] = m4["err_rel"]
import json  # noqa: E402
print(json.dumps(esito, indent=1))
ok = all(v for k, v in esito.items() if isinstance(v, bool))
print("TUTTO VERDE" if ok else "⛔ ROSSO")
sys.exit(0 if ok else 1)
