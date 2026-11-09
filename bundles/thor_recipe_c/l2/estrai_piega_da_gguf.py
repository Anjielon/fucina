#!/usr/bin/env python3
"""11/10: ricostruisce piega/tensori nel formato ESATTO di cardan_piega.py (righe 263-314: raw1 = dato del piano 1
reshape(n_esperti, out, -1); raw2 = piano 2 reshape(shape[2], out, -1)) leggendo il GGUF piegato con lo stesso lettore
(rd_ternary.gguf_piani), che contiene quei byte identici (patch). Serve dopo la pulizia del disco delle 09:58 (errore mio:
piega/tensori cancellati, ma SEL_pred_CARDAN di B li legge). Si scrivono solo gli strati diversi dall'ospite.
Uso: estrai_piega_da_gguf.py PIEGATO.gguf OSPITE.gguf OUT_DIR"""
import sys, os, numpy as np
sys.path.insert(0, "@@ROOT:models@@/build/leve/raffina")
import comune as CM
from rd_ternary.formato import TensorKey
gp = CM.gp
pg, osp, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
scritti = uguali = 0
for L in range(40):
    sp = gp.GGUFPlaneSource(pg, layers=[L], bases=["ffn_gate_exps", "ffn_up_exps"])
    so = gp.GGUFPlaneSource(osp, layers=[L], bases=["ffn_gate_exps", "ffn_up_exps"])
    for f in ("gate", "up"):
        ep, eo = sp.entry(TensorKey(L, 0, f)), so.entry(TensorKey(L, 0, f))
        a1 = np.array(ep.tensors[0].data); b1 = np.array(eo.tensors[0].data)
        cambiato = not np.array_equal(a1, b1)
        r2 = None
        if len(ep.tensors) > 1:
            a2 = np.array(ep.tensors[1].data); cambiato |= not np.array_equal(a2, np.array(eo.tensors[1].data))
            r2 = a2.reshape(int(ep.tensors[1].shape[2]), ep.out, -1).copy()
        if not cambiato:
            uguali += 1; continue
        np.save(os.path.join(out, f"blk.{L}.ffn_{f}_exps.weight.npy"), a1.reshape(ep.n_experts, ep.out, -1).copy())
        if r2 is not None:
            np.save(os.path.join(out, f"blk.{L}.ffn_{f}_exps2.weight.npy"), r2)
        scritti += 1
print(f"famiglie piegate scritte {scritti}, uguali all'ospite {uguali}, in {out}")
