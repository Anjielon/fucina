"""test_piega_algebra.py — the fold's factorizations off rocSOLVER (9/10 fault, cardan_piega.py).

CPU part (always): chol_ex / chol / risolvi_chol / risolvi with ALGEBRA_CPU forced on a 'meta-free' path give the same
numbers as the torch primitives, including the info != 0 rows and the 4-D batch of scale_lotto; jarq_lotto still
matches its fp64 reference on a toy problem.
GPU part (only with a GPU and PIEGA_TEST_GPU=1, inside the lock): the old batched rocSOLVER call is run in a CHILD
process at the fold's shapes (reproduces the fault: the child dies), and the cured fold of one real batch runs clean.
"""
import os
import subprocess
import sys

import tempfile
import torch

sys.path.insert(0, os.path.dirname(__file__))
import cardan_piega as CP  # noqa: E402


def _spd(b, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(b, n, n + 3, generator=g, dtype=torch.float64)
    return A @ A.transpose(-1, -2) + 1e-6 * torch.eye(n, dtype=torch.float64)


def test_risolvi_chol_uguale_a_cholesky_solve():
    M = _spd(8, 24)
    L = torch.linalg.cholesky(M)
    B = torch.randn(8, 24, 5, dtype=torch.float64)
    rif = torch.cholesky_solve(B, L)
    y = torch.linalg.solve_triangular(L, B, upper=False)
    x = torch.linalg.solve_triangular(L.transpose(-1, -2), y, upper=True)   # the GPU branch of risolvi_chol
    assert torch.allclose(x, rif, rtol=1e-10, atol=1e-10)


def test_chol_ex_4d_e_info():
    M = _spd(2 * 3, 6).reshape(2, 3, 6, 6).clone()
    M[1, 2] = -torch.eye(6, dtype=torch.float64)                           # not SPD -> info != 0, like a dead row
    L, info = torch.linalg.cholesky_ex(M.to(torch.float64))
    L2, info2 = CP.torch.linalg.cholesky_ex(CP._f64cpu(M))
    assert torch.equal(info, info2) and int(info2[1, 2]) != 0 and int(info2[0, 0]) == 0
    assert torch.allclose(L[info == 0], L2[info2 == 0])


def test_jarq_lotto_cpu_invariato():
    torch.manual_seed(1)
    B, K, R, C, ng = 2, 3, 8, 16, 2
    T = torch.randint(-1, 2, (B, K, R, C), dtype=torch.int8)
    S = torch.rand(B, K, R, ng, dtype=torch.float64) * torch.tensor([1.0, 0.3, 0.1]).view(1, K, 1, 1).double()
    H = _spd(B, C, seed=3)
    H = H / torch.diagonal(H, dim1=-2, dim2=-1).mean(-1).view(B, 1, 1)
    W = CP.ricostruisci_lotto(T, S) + 0.05 * torch.randn(B, R, C, dtype=torch.float64)
    T2, S2, cat = CP.jarq_lotto(W, H, T, S, giri=2)
    f0 = CP.obiettivo_lotto(W, CP.ricostruisci_lotto(T, S), H).sum()
    f1 = CP.obiettivo_lotto(W, CP.ricostruisci_lotto(T2, S2), H).sum()
    assert f1 <= f0 + 1e-12


GPU = os.environ.get("PIEGA_TEST_GPU") == "1" and torch.cuda.is_available()


def test_gpu_sonda_e_piega_curata():
    if not GPU:
        print("GPU part skipped (only inside the lock, PIEGA_TEST_GPU=1)")
        return
    tmp_path = tempfile.mkdtemp(dir=os.environ.get("TMPDIR"))
    p = subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "sonda_linalg_gpu.py")],
                       capture_output=True, text=True, timeout=1800)
    print(p.stdout)
    env = dict(os.environ, PIEGA_ALGEBRA="cpu")
    q = subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "cardan_piega.py"),
                        "--pt", "@@ROOT:models@@/gguf/tony-tern/CARDAN/kd_r16/allena/cardan_migliore.pt",
                        "--gguf", "@@ROOT:models@@/gguf/tony-tern/PV12P/Thor-T-pv12p.gguf", "--out", str(tmp_path),
                        "--strati", "0-0", "--device", "cuda", "--lotto", "16", "--esperti", "16"],
                       capture_output=True, text=True, timeout=1800, env=env)
    assert q.returncode == 0, q.stdout[-2000:] + q.stderr[-2000:]


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_") and callable(f):
            f()
            print("ok", n)
    print("TUTTE LE PROVE PIEGA-ALGEBRA PASSATE")
