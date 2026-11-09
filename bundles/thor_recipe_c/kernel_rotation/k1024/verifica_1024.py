"""
Verifiche (a) e (b) del tipo TQ1_0R_1024 su un TENSORE VERO (Ornith-1.5-35B-A3B,
blk.20.attn_qkv.weight bf16 -> f32 esatto), attraverso le API ggml del fork
@@ROOT:models@@/build/llamacpp-tq1-k1024 (harness_ggml_1024):
  (a) round-trip CPU bit-esatto contro numpy:
        byte quantizzati C == byte del codec numpy (tutti)
        float dequantizzati C == float del codec numpy (tutti, bit per bit)
  (b) numpy vs kernel GPU (Vulkan, lo stesso ggml_mul_mat di llama-server):
        y_gpu = W_hat x  contro  y_ref = dequant_numpy(W) . x in float64,
        per n=1 (mul_mat_vec cooperativo), n=4 (NUM_COLS), n=32/512 (mul_mm,
        forma chiusa). Errore relativo = ||y_gpu - y_ref|| / ||y_ref|| e
        max|dy|/max|y|; con float32 e ~2048 termini ci si aspetta ~1e-6.
      Piu' il CPU backend di ggml (vec_dot f16) come terzo riferimento.
"""
import os
import subprocess
import sys
import glob
import numpy as np

sys.path.insert(0, "@@ROOT:models@@/build/llamacpp-tq1-k1024/gguf-py")
sys.path.insert(0, "@@ROOT:forge@@/kernel-rotazione")
from gguf import GGUFReader  # noqa: E402
from tq1_0r_1024_codec import quantize_rows, dequantize_blocks, BLOCK_BYTES  # noqa: E402

HARNESS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "harness_ggml_1024")
SRC_DIR = "@@ROOT:models@@/gguf/thor-src/Ornith-1.5-35B-A3B-bf16"
TENSOR = os.environ.get("K1024_TENSOR", "blk.20.attn_qkv.weight")
N_ROWS = int(os.environ.get("K1024_ROWS", "1024"))
TMP = os.environ.get("K1024_TMP", "/tmp/claude-1000/-home-angelo-MOGAVIS/12e4589a-507e-45a0-b690-66b786e2b26f/scratchpad/k1024")
os.makedirs(TMP, exist_ok=True)
ENV = dict(os.environ, LD_LIBRARY_PATH="@@ROOT:models@@/build/llamacpp-tq1-k1024/build/bin:/lib64",
           GGML_BACKEND_PATH="@@ROOT:models@@/build/llamacpp-tq1-k1024/build/bin/libggml-vulkan.so")


def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def carica():
    for p in sorted(glob.glob(SRC_DIR + "/*.gguf")):
        r = GGUFReader(p)
        for t in r.tensors:
            if t.name == TENSOR:
                W = bf16_to_f32(np.asarray(t.data).view(np.uint16))
                return np.ascontiguousarray(W[:N_ROWS], dtype=np.float32)
    raise SystemExit(f"{TENSOR} non trovato in {SRC_DIR}")


def run(*args):
    r = subprocess.run([HARNESS, *map(str, args)], env=ENV, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout, r.stderr)
        raise SystemExit(f"harness fallito: {args}")
    return r.stdout.strip()


def main():
    W = carica()
    R, C = W.shape
    print(f"tensore {TENSOR}: {R}x{C} (righe usate {N_ROWS}), blocchi/riga {C//1024}")
    fW = f"{TMP}/W.f32"; W.tofile(fW)

    # (a1) quantizzazione: C vs numpy, byte per byte
    fP = f"{TMP}/packed_c.bin"
    print(run("quant", fW, R, C, fP))
    P_c = np.fromfile(fP, dtype=np.uint8).reshape(-1, BLOCK_BYTES)
    P_np = quantize_rows(W)
    same_bytes = np.array_equal(P_c, P_np)
    n_diff = int((P_c != P_np).sum())
    print(f"(a1) quant C == numpy: {same_bytes}  (byte diversi: {n_diff} su {P_c.size})")
    if not same_bytes:
        # dove divergono? trit (0..207) o scale (208..223)?
        d = (P_c != P_np)
        print("     byte diversi nei trit:", int(d[:, :208].sum()), " nelle scale:", int(d[:, 208:].sum()))

    # (a2) dequantizzazione: C vs numpy, float per float (partendo dagli STESSI byte C)
    fD = f"{TMP}/dequant_c.f32"
    print(run("dequant", fP, R, C, fD))
    D_c = np.fromfile(fD, dtype=np.float32).reshape(R, C)
    D_np = dequantize_blocks(P_c).reshape(R, C)
    same_f = np.array_equal(D_c.view(np.uint32), D_np.view(np.uint32))
    print(f"(a2) dequant C == numpy (bit-esatto): {same_f}  max|diff| = {np.abs(D_c - D_np).max():.3e}")
    mse_q = float(np.mean((W - D_np) ** 2)); mse0 = float(np.mean(W ** 2))
    print(f"     MSE quantizzazione = {mse_q:.6e}  (MSE/energia = {mse_q/mse0*100:.2f}%)")

    # (b) matmul: GPU (Vulkan) e CPU ggml contro numpy float64 su W_hat
    rng = np.random.default_rng(20260920)
    esiti = []
    for n in (1, 4, 32, 512):
        X = rng.standard_normal((n, C)).astype(np.float32)
        fX = f"{TMP}/X_{n}.f32"; X.tofile(fX)
        y_ref = (X.astype(np.float64) @ D_np.astype(np.float64).T)        # (n, R)
        for be in ("vulkan", "cpu"):
            fY = f"{TMP}/Y_{n}_{be}.f32"
            out = run("matmul", fP, R, C, fX, n, fY, be)
            Y = np.fromfile(fY, dtype=np.float32).reshape(n, R).astype(np.float64)
            rel_norm = np.linalg.norm(Y - y_ref) / np.linalg.norm(y_ref)
            rel_max = np.abs(Y - y_ref).max() / np.abs(y_ref).max()
            esiti.append((n, be, rel_norm, rel_max))
            print(f"(b) n={n:3d} {be:6s} {out.splitlines()[0]:24s} ||dy||/||y|| = {rel_norm:.3e}   max|dy|/max|y| = {rel_max:.3e}"
                  + ("   [prec f32]" if os.environ.get("K1024_PREC_F32") else ""))
    peggio = max(e[2] for e in esiti)
    print(f"ESITO (a): {'OK' if same_bytes and same_f else 'FALLITO'}   ESITO (b): errore relativo peggiore {peggio:.3e} {'(<=1e-6 OK)' if peggio <= 1e-6 else '(> 1e-6: vedi sopra)'}")


if __name__ == "__main__":
    main()
