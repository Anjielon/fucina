"""Collaudo AL BIT della ricetta 20set: i byte dei tensori costruiti senza LoRA (tensori_rif/) contro quelli della forgia
del 20/9 dentro Thor-v20-downaggr3 (61 densi TQ1_0R_1024 da Thor-ibrido-L = GPTQ + scale congiunte, H piccole).
Per ogni tensore: byte uguali, blocchi uguali, trit diversi, scale diverse (in unita' fp16)."""
import sys, json, numpy as np
sys.path.insert(0, '@@ROOT:models@@/build/llamacpp-tq1-k1024/gguf-py'); sys.path.insert(0, '.')
from gguf import GGUFReader
from tq1_0r_1024_codec import _unpack_quarter
man = json.load(open('@@ROOT:depot@@/tutto_ternario/tensori_rif/manifesto.json'))
v20 = {t.name: t for t in GGUFReader('@@ROOT:models@@/gguf/tony-tern/Thor-v20-downaggr3.gguf').tensors}
def trit(P):
    return np.concatenate([_unpack_quarter(P[:, q*48:(q+1)*48], P[:, 192+q*4:192+(q+1)*4]) for q in range(4)], 1)
for nome in man:
    a = np.load(f'@@ROOT:depot@@/tutto_ternario/tensori_rif/{nome}.npy').reshape(-1, 224)
    b = np.asarray(v20[nome].data).reshape(-1, 224)
    assert v20[nome].tensor_type.name == 'TQ1_0R_1024'
    ta, tb = trit(a), trit(b)
    da = a[:, 208:].copy().view(np.float16).astype(np.float32); db = b[:, 208:].copy().view(np.float16).astype(np.float32)
    print(f"{nome:30s} byte uguali {np.array_equal(a, b)} · blocchi uguali {(a == b).all(1).mean():.4f} · trit diversi {(ta != tb).mean():.2e}"
          f" · scale diverse {(da != db).mean():.2e} (max |Δ|/|d| {np.max(np.abs(da - db) / np.maximum(np.abs(db), 1e-12)):.2e})")
