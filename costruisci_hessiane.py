"""Dai dump grezzi di attn_post_norm alle Hessiane H = XXᵀ per i 60 strati."""
import glob, os, re, sys
import numpy as np
D = sys.argv[1] if len(sys.argv) > 1 else "/mnt/models/gguf/odino-dumps"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/mnt/models/gguf/odino-hessiane"
NDIM = 4096
os.makedirs(OUT, exist_ok=True)
for f in sorted(glob.glob(f"{D}/*attn_post_norm*.f32")):
    m = re.search(r"attn_post_norm-(\d+)", f) or re.search(r"(\d+)", os.path.basename(f))
    if not m: continue
    strato = int(m.group(1))
    X = np.fromfile(f, dtype=np.float32)
    n = X.size // NDIM
    if n < NDIM // 4:
        print(f"   ⚠️ strato {strato}: solo {n} campioni per {NDIM} dimensioni — H sara' povera")
    X = X[:n * NDIM].reshape(n, NDIM).astype(np.float64)
    H = (X.T @ X) / max(n, 1)
    np.save(f"{OUT}/H_{strato:02d}.npy", H.astype(np.float32))
    ev = np.linalg.eigvalsh(H)[::-1]
    ev = ev[ev > 0]
    alfa = -np.polyfit(np.log(np.arange(1, len(ev) + 1)), np.log(ev / ev[0]), 1)[0]
    print(f"   strato {strato:2d}: {n:6d} campioni · anisotropia alfa = {alfa:.3f}")
print(f"✓ Hessiane in {OUT}")
