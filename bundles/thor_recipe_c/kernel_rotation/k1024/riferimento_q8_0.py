"""Baseline di precisione del percorso mul_mm fp16acc di ggml-vulkan con un tipo
di serie (Q8_0), STESSO tensore e STESSE x della verifica (b): se l'errore di
TQ1_0R_1024 a n>=32 coincide con questo, il decode a forma chiusa e' corretto e
l'errore e' del percorso (accumulo fp16 + B in f16), non del tipo nuovo."""
import os, subprocess, numpy as np
TMP = "/tmp/claude-1000/-home-angelo-MOGAVIS/12e4589a-507e-45a0-b690-66b786e2b26f/scratchpad/k1024"
H = os.path.join(os.path.dirname(os.path.abspath(__file__)), "harness_ggml_1024")
ENV = dict(os.environ, K1024_TYPE="q8_0", LD_LIBRARY_PATH="@@ROOT:models@@/build/llamacpp-tq1-k1024/build/bin:/lib64",
           GGML_BACKEND_PATH="@@ROOT:models@@/build/llamacpp-tq1-k1024/build/bin/libggml-vulkan.so")
R, C = 1024, 2048
def run(*a):
    r = subprocess.run([H, *map(str, a)], env=ENV, capture_output=True, text=True); assert r.returncode == 0, r.stderr; return r.stdout
run("quant", f"{TMP}/W.f32", R, C, f"{TMP}/packed_q8.bin")
run("dequant", f"{TMP}/packed_q8.bin", R, C, f"{TMP}/dequant_q8.f32")
D = np.fromfile(f"{TMP}/dequant_q8.f32", dtype=np.float32).reshape(R, C).astype(np.float64)
for n in (1, 32, 512):
    X = np.fromfile(f"{TMP}/X_{n}.f32", dtype=np.float32).reshape(n, C)
    y_ref = X.astype(np.float64) @ D.T
    for prec in ("", "f32"):
        env = dict(ENV); 
        if prec: env["K1024_PREC_F32"] = "1"
        r = subprocess.run([H, "matmul", f"{TMP}/packed_q8.bin", str(R), str(C), f"{TMP}/X_{n}.f32", str(n), f"{TMP}/Y_q8_{n}.f32", "vulkan"], env=env, capture_output=True, text=True); assert r.returncode == 0, r.stderr
        Y = np.fromfile(f"{TMP}/Y_q8_{n}.f32", dtype=np.float32).reshape(n, R).astype(np.float64)
        print(f"q8_0  n={n:3d} vulkan {'prec f32' if prec else 'default '}  ||dy||/||y|| = {np.linalg.norm(Y-y_ref)/np.linalg.norm(y_ref):.3e}   max|dy|/max|y| = {np.abs(Y-y_ref).max()/np.abs(y_ref).max():.3e}")
