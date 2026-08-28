#!/usr/bin/env python3
"""Pre-compute the impact ranking of the 397B's experts for the tail band.

Why offline: in the forge the FIRST tensor of each layer is `down`, but the
hot set must be identical across gate/up/down (one router permutation), and
the impact score needs gate/up weights — which the forge has not read yet at
that point. Computing the ranking here, in daylight, on an idle NAS, adds
ZERO minutes to the night forge: forge_from_bf16 reads the JSON via
FORGE_IMPACT_JSON.

Score per expert e:  sum over gate,up of  ||dW_e @ L||_F^2  with H = L L^T
(the layer's residual-stream Hessian) and dW_e the dedicated-plane RTN error
— trace(H dW^T dW), what the residual stream actually feels (MoPEQ line).

Usage: impact_rank_odino.py LO-HI OUT.json
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, torch
torch.set_num_threads(4)   # la forgia v3 ha la precedenza sulla CPU
sys.path.insert(0, str(Path(__file__).parent))
import ternary_gpu as G
sys.path.insert(0, "/home/angelo/build-llamacpp-tq1/gguf-py")

NAS = Path("/mnt/nas/CACHEDEV1_DATA/modelli-fp/Ornith-1.5-397B")
IMATRIX = Path("/home/angelo/odino-lab/imatrix/Ornith-1.5-397B-imatrix.gguf")
HESS = Path("/mnt/models/gguf/odino-hessiane")
E = 512

def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def main() -> None:
    lo, hi = (int(x) for x in sys.argv[1].split("-"))
    out_path = Path(sys.argv[2])
    done = json.loads(out_path.read_text()) if out_path.exists() else {}
    from safetensors import safe_open
    idx = json.load(open(NAS / "model.safetensors.index.json"))["weight_map"]
    # ⛔ La massa di instradamento E' parte dell'impatto: un esperto mai chiamato
    # non consegna il suo errore. Primo giro SENZA p_e: overlap con la frequenza
    # 0/16 — classifica dominata dagli esperti freddi a pesi grossi. Score:
    #   impact_e = p_e * ||dW_e sqrt(diag H)||^2
    from gguf import GGUFReader
    counts = {}
    for tt in GGUFReader(str(IMATRIX)).tensors:
        if tt.name.endswith(".ffn_gate_exps.weight.counts"):
            counts[int(tt.name.split(".")[1])] = np.array(tt.data).astype(np.float64).ravel()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for L in range(lo, hi + 1):
        if str(L) in done:
            log(f"layer {L}: gia' in cassa"); continue
        Hraw = np.load(HESS / f"H_{L:02d}.npy").astype(np.float64)
        n = Hraw.shape[0]
        # ⚡ DIAGONAL score, not the full trace(H dW^T dW): the full matmul is
        # 35 TFLOP/layer (measured: 25 min/layer on CPU — 6.7 h for the band).
        # For a RANKING the QERA-approx weighting ||dW sqrt(diag H)||^2 is the
        # published closed-form stand-in (activation RMS per input channel).
        wcol = torch.from_numpy(np.sqrt(np.abs(np.diag(Hraw))).astype(np.float32)).to(dev)
        key = f"model.language_model.layers.{L}.mlp.experts.gate_up_proj"
        t0 = time.time()
        score = np.zeros(E)
        with safe_open(str(NAS / idx[key]), "pt") as f:
            sl = f.get_slice(key)                 # (E, 2*out, hidden) fused
            shp = sl.get_shape()
            half = shp[1] // 2                    # gate = righe [:half], up = [half:]
            assert shp[2] == n, (shp, n)          # colonne = residuo = dim Hessiana
            for e in range(E):
                fused = sl[e].to(torch.float32).numpy()   # (2*out, hidden)
                for part in (fused[:half, :], fused[half:, :]):   # gate, up — (out, in)
                    W = torch.from_numpy(np.ascontiguousarray(part)).to(dev)
                    d, q = G._scale_one_plane(W.reshape(-1, 256))
                    dW = W - (d * q).reshape(W.shape)
                    score[int(e)] += float((dW * wcol).square().sum())
                    del W, dW
                if e % 128 == 0:
                    log(f"  layer {L}: esperto {e}/512")
        pe = counts[L] / max(counts[L].sum(), 1.0)
        score = score * pe
        done[str(L)] = [int(x) for x in np.argsort(-score)]
        out_path.write_text(json.dumps(done))
        log(f"layer {L}: classificato in {(time.time()-t0)/60:.1f} min · "
            f"top-4 impact: {done[str(L)][:4]}")
    log(f"🏁 IMPACT RANK COMPLETO → {out_path}")


if __name__ == "__main__":
    main()
