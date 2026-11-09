#!/usr/bin/env python3
"""sonda_linalg_gpu.py — which batched linear-algebra primitive of cardan_piega.py faults on gfx1151 (9/10).

The fold died at 15:31 with "Memory access fault by GPU node-1 on address (nil)" before finishing its first batch
(layer 0, 16 experts). Each primitive runs ALONE in a child process at the fold's real shapes (B=16 experts, R=512
rows, C=2048, ng=2 groups of G=1024, K=3 planes -> n=K*ng=6), so a fault kills only that child and is named.
Usage (inside the GPU lock): python3 sonda_linalg_gpu.py   -> one line per primitive: ok / FAULT (return code)
"""
import subprocess
import sys

PRIMITIVE = {
    # scale_lotto: B*r_ = 16*128 small SPD systems n=6
    "cholesky_ex_6x6_b2048": "M=spd(2048,6); L,i=torch.linalg.cholesky_ex(M); s(L)",
    "cholesky_solve_6x6_b2048": "M=spd(2048,6); L=torch.linalg.cholesky(M.cpu()).to(dev); "
                                "x=torch.cholesky_solve(torch.randn(2048,6,1,device=dev),L); s(x)",
    "solve_6x6_b37": "M=spd(37,6); x=torch.linalg.solve(M,torch.randn(37,6,1,device=dev)); s(x)",
    # scale_lotto as the fold calls it: 4-D batch [B, r_, n, n]
    "cholesky_ex_4d_16x128x6x6": "M=spd(2048,6).reshape(16,128,6,6); L,i=torch.linalg.cholesky_ex(M); s(L)",
    "cholesky_solve_4d_16x128x6x6": "M=spd(2048,6).reshape(16,128,6,6); L=torch.linalg.cholesky(M.cpu()).to(dev); "
                                    "x=torch.cholesky_solve(torch.randn(16,128,6,1,device=dev),L); s(x)",
    # babai_lotto: per group G=1024, batch 16
    "cholesky_ex_1024_b16": "M=spd(16,1024); L,i=torch.linalg.cholesky_ex(M); s(L)",
    "cholesky_1024_b16": "M=spd(16,1024); L=torch.linalg.cholesky(M); s(L)",
    "cholesky_solve_1024x512_b16": "M=spd(16,1024); L=torch.linalg.cholesky(M.cpu()).to(dev); "
                                   "x=torch.cholesky_solve(torch.randn(16,1024,512,device=dev),L); s(x)",
    "solve_triangular_1024x512_b16": "M=spd(16,1024); L=torch.linalg.cholesky(M.cpu()).to(dev); "
                                     "y=torch.linalg.solve_triangular(L,torch.randn(16,1024,512,device=dev),upper=False); "
                                     "x=torch.linalg.solve_triangular(L.transpose(-1,-2),y,upper=True); s(x)",
}

PRE = ("import torch; dev=torch.device('cuda'); torch.manual_seed(0)\n"
       "def spd(b,n):\n A=torch.randn(b,n,n+4,device=dev); return A@A.transpose(-1,-2)+1e-3*torch.eye(n,device=dev)\n"
       "def s(t):\n torch.cuda.synchronize(); assert torch.isfinite(t).all(), 'non finito'\n")


def main():
    esiti = {}
    for nome, corpo in PRIMITIVE.items():
        p = subprocess.run([sys.executable, "-c", PRE + corpo + "\nprint('ok')"], capture_output=True, text=True,
                           timeout=600)
        ok = p.returncode == 0 and p.stdout.strip().endswith("ok")
        coda = (p.stderr.strip().splitlines() or [""])[-1][:160]
        esiti[nome] = "ok" if ok else f"FAULT rc={p.returncode} {coda}"
        print(f"{nome:34s} {esiti[nome]}", flush=True)
    return 0 if all(v == "ok" for v in esiti.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
