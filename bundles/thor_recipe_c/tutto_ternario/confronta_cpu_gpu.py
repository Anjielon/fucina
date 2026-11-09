"""Stessa ricetta, CPU contro GPU: byte uguali, trit diversi, scale diverse (tensori presenti in entrambe le cartelle).
Uso: confronta_cpu_gpu.py <cartella_cpu> <cartella_gpu>"""
import json
import sys

import numpy as np

sys.path.insert(0, "@@ROOT:depot@@/tutto_ternario/codice")
from tq1_0r_1024_codec import _unpack_quarter  # noqa: E402

a_dir, b_dir = sys.argv[1], sys.argv[2]
ma = json.load(open(f"{a_dir}/manifesto.json")); mb = json.load(open(f"{b_dir}/manifesto.json"))
comuni = sorted(set(ma) & set(mb), key=lambda n: (int(n.split(".")[1]), n))
tot_t = tot_n = 0
for n in comuni:
    fa = ma[n]["misure"]["formato"]
    A = np.load(f"{a_dir}/{n}.npy"); B = np.load(f"{b_dir}/{n}.npy")
    if fa == "TQ1_0R_1024":
        A = A.reshape(-1, 224); B = B.reshape(-1, 224)
        tA = np.concatenate([_unpack_quarter(A[:, q*48:(q+1)*48], A[:, 192+q*4:192+(q+1)*4]) for q in range(4)], 1)
        tB = np.concatenate([_unpack_quarter(B[:, q*48:(q+1)*48], B[:, 192+q*4:192+(q+1)*4]) for q in range(4)], 1)
        dA = A[:, 208:].copy().view(np.float16).astype(np.float32); dB = B[:, 208:].copy().view(np.float16).astype(np.float32)
    else:
        A = A.reshape(-1, 54); B = B.reshape(-1, 54)
        tA = _unpack_quarter(A[:, :48], A[:, 48:52]); tB = _unpack_quarter(B[:, :48], B[:, 48:52])
        dA = A[:, 52:].copy().view(np.float16).astype(np.float32); dB = B[:, 52:].copy().view(np.float16).astype(np.float32)
    dt = int((tA != tB).sum()); tot_t += dt; tot_n += tA.size
    ea, eb = ma[n]["misure"].get("err_rel_pesato_H"), mb[n]["misure"].get("err_rel_pesato_H")
    print(f"{n:30s} byte uguali {np.array_equal(A, B)} · trit diversi {dt / tA.size:.2e} · scale diverse {(dA != dB).mean():.2e}"
          + (f" · err pesato cpu {ea:.5f} gpu {eb:.5f}" if ea is not None else f" · err cpu {ma[n]['misure']['err_rel']:.5f} gpu {mb[n]['misure']['err_rel']:.5f}"))
print(f"TOTALE {len(comuni)} tensori · trit diversi {tot_t}/{tot_n} = {tot_t / max(tot_n, 1):.2e}")
