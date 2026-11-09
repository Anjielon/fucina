#!/usr/bin/env python3
"""verifica_congiunto_secco.py — 18/10, dry run of cardan_congiunto.py: byte check of the TWO folds (experts with
cardan_piega.py, dense with cardan_piega_densi.py) against the host GGUF. For every written tensor:
  name exists in the host · dtype uint8 · n_bytes == host n_bytes · shape == host raw shape · decode->encode at the bit
  (codes in {-1,0,1}, fp16 scales) · experts: one plane per folded expert (K=1), experts >= N identical to the host ·
  dense: same plane names / count as the host, at least one byte changed when the correction is non-zero.
uso: verifica_congiunto_secco.py OSPITE.gguf PIEGA_ESPERTI PIEGA_DENSI N_ESPERTI_PIEGATI"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "@@ROOT:models@@/build/leve/raffina")
import comune as CM  # noqa: E402
gp = CM.gp
g, de, dd, n = sys.argv[1], Path(sys.argv[2]) / "tensori", Path(sys.argv[3]) / "tensori", int(sys.argv[4])
gguf = gp._import_gguf(None)
tens = {t.name: t for t in gguf.GGUFReader(g, "r").tensors}


def grezzo(t, forma):
    o = np.asarray(t.data)
    return (o if o.dtype == np.uint8 else o.view(np.uint8)).reshape(forma)


def comune(f):
    nome = f.name[:-4]; t = tens[nome]; a = np.load(f)
    assert a.dtype == np.uint8 and a.nbytes == t.n_bytes, (nome, a.dtype, a.nbytes, t.n_bytes)
    o = grezzo(t, a.shape)
    assert o.shape == a.shape, (nome, o.shape, a.shape)
    return nome, t, a, o


from rd_ternary.formato import TensorKey  # noqa: E402
for f in sorted(de.glob("*.npy")):
    nome, t, a, o = comune(f)
    L = int(nome.split(".")[1]); fam = "gate" if "gate" in nome else "up"
    e_ = gp.GGUFPlaneSource(g, layers=[L], bases=["ffn_gate_exps", "ffn_up_exps"]).entry(TensorKey(L, 0, fam))
    assert np.array_equal(a[n:], o[n:]), f"{nome}: esperti >= {n} cambiati"
    for e in range(n):
        assert e_.n_planes_of(e) == 1, (nome, e)
        tp, sp = gp.decode_rows(a[e], e_.fmt.type_id, e_.in_)
        b = gp.encode_rows(tp.astype(np.int8), sp, e_.fmt.type_id)
        assert np.array_equal(np.asarray(b).view(np.uint8).reshape(-1), a[e].reshape(-1)), (nome, e)
        assert set(np.unique(tp)).issubset({-1, 0, 1}), (nome, e)
    cam = [e for e in range(n) if not np.array_equal(a[e], o[e])]
    print(f"OK esperti {nome}: {a.nbytes} B = ospite, forma {a.shape} uint8, K=1, ricodifica al bit, "
          f"cambiati {cam}/{n}, esperti {n}..{a.shape[0] - 1} identici al byte")
for f in sorted(dd.glob("*.npy")):
    nome, t, a, o = comune(f)
    L = int(nome.split(".")[1])
    d = CM.Denso(gp.GGUFPlaneSource(g, layers=[L]), nome.replace("3.weight", ".weight").replace("2.weight", ".weight"))
    assert nome in d.nomi_piani, (nome, d.nomi_piani)
    tp, sp = gp.decode_rows(a, d.tid, d.in_)
    b = gp.encode_rows(tp.astype(np.int8), sp, d.tid)
    assert np.array_equal(np.asarray(b).view(np.uint8).reshape(-1), a.reshape(-1)), nome
    assert set(np.unique(tp)).issubset({-1, 0, 1}), nome
    print(f"OK densi {nome}: {a.nbytes} B = ospite, forma {a.shape} uint8, piani dell'ospite {len(d.nomi_piani)}, "
          f"ricodifica al bit, byte cambiati {int((a != o).sum())}")
print("VERIFICA PIEGHE CONGIUNTO OK")
