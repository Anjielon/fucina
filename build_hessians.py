"""Activation dumps -> per-layer Hessians H = XᵀX, one file per layer.

The Hessian is what lets GPTQ push quantization error where it costs least.
Two practical notes learned in the field:
  - accumulate from the RESIDUAL STREAM (post-attention norm), which is the
    actual input of the expert gate/up projections;
  - report the spectrum's anisotropy (alpha): a flat spectrum means the
    Hessian carries little information and GPTQ will gain little.

Set NDIM to the model's hidden size (default 4096).
"""
import glob, os, re, sys
import numpy as np


def main() -> None:
    """Read the activation dumps and write one Hessian per layer."""
    DUMPS = sys.argv[1] if len(sys.argv) > 1 else "/mnt/models/gguf/odino-dumps"
    OUT = sys.argv[2] if len(sys.argv) > 2 else "/mnt/models/gguf/odino-hessians"
    NDIM = int(__import__("os").getenv("NDIM", "4096"))
    os.makedirs(OUT, exist_ok=True)
    for f in sorted(glob.glob(f"{DUMPS}/*attn_post_norm*.f32")):
        m = re.search(r"attn_post_norm-(\d+)", f) or re.search(r"(\d+)", os.path.basename(f))
        if not m: continue
        layer = int(m.group(1))
        X = np.fromfile(f, dtype=np.float32)
        n = X.size // NDIM
        if n < NDIM // 4:
            print(f"   warning: layer {layer}: only {n} samples for {NDIM} dimensions — weak Hessian")
        X = X[:n * NDIM].reshape(n, NDIM).astype(np.float64)
        H = (X.T @ X) / max(n, 1)
        np.save(f"{OUT}/H_{layer:02d}.npy", H.astype(np.float32))
        ev = np.linalg.eigvalsh(H)[::-1]
        ev = ev[ev > 0]
        alpha = -np.polyfit(np.log(np.arange(1, len(ev) + 1)), np.log(ev / ev[0]), 1)[0]
        print(f"   layer {layer:2d}: {n:6d} samples · spectral anisotropy alpha = {alpha:.3f}")
    print(f"✓ Hessians written to {OUT}")


if __name__ == "__main__":
    main()
