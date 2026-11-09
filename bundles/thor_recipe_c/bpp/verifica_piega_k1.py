#!/usr/bin/env python3
"""verifica_piega_k1.py — 11/10 CARDAN-C0B: prova a secco della piega a UN piano (K=1) su CPU.
Controlla: (1) nessun tensore exps2 (C0B non ne ha), (2) gli esperti NON piegati sono identici al byte all'ospite,
(3) gli esperti piegati hanno codici a UN piano e scale fp16 che si ridecodificano al bit (encode->decode), (4) taglia
dei tensori = taglia nell'ospite. uso: verifica_piega_k1.py OSPITE.gguf CARTELLA_PIEGA N_ESPERTI_PIEGATI"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "@@ROOT:models@@/build/leve/raffina")
import comune as CM  # noqa: E402
gp = CM.gp
g, d, n = sys.argv[1], Path(sys.argv[2]) / "tensori", int(sys.argv[3])
gguf = gp._import_gguf(None)
rd = gguf.GGUFReader(g, "r"); tens = {t.name: t for t in rd.tensors}
assert not list(d.glob("*exps2*")), "piega K=1 ha scritto tensori exps2"
for f in sorted(d.glob("*.npy")):
    nome = f.name[:-4]; t = tens[nome]; a = np.load(f)
    o = np.asarray(t.data).reshape(a.shape) if np.asarray(t.data).size == a.size else np.asarray(t.data).view(np.uint8).reshape(a.shape)
    assert a.dtype == np.uint8 and a.nbytes == t.n_bytes, (nome, a.nbytes, t.n_bytes)
    assert np.array_equal(a[n:], o[n:]), f"{nome}: esperti >= {n} cambiati"
    cambiati = [e for e in range(n) if not np.array_equal(a[e], o[e])]
    L = int(nome.split(".")[1]); fam = "gate" if "gate" in nome else "up"
    from rd_ternary.formato import TensorKey
    src = gp.GGUFPlaneSource(g, layers=[L], bases=["ffn_gate_exps", "ffn_up_exps"])
    e_ = src.entry(TensorKey(L, 0, fam)); tid = e_.fmt.type_id
    for e in range(n):
        assert e_.n_planes_of(e) == 1, (nome, e, e_.n_planes_of(e))
        tp, sp = gp.decode_rows(a[e], tid, e_.in_)
        b = gp.encode_rows(tp.astype(np.int8), sp, tid)
        assert np.array_equal(np.asarray(b).view(np.uint8).reshape(-1), a[e].reshape(-1)), (nome, e, "ricodifica non al bit")
        assert set(np.unique(tp)).issubset({-1, 0, 1}), (nome, e)
    print(f"OK {nome}: {a.nbytes} B = ospite, esperti piegati cambiati {cambiati}/{n}, esperti {n}..{a.shape[0]-1} identici al byte, "
          f"un piano, ricodifica al bit")
print("VERIFICA PIEGA K=1 OK")
