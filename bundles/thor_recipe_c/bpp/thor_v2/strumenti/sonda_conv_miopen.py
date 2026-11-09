"""2/10 TT6-veloce: quale aritmetica usa F.conv1d (depthwise, causale, fp32) del GDN sulla GPU? Confronto AL BIT con
candidate scritte nella disposizione (B, T, C) senza trasposte. Piccola: 2 x 4096 x 8192 fp32 (256 MiB), pochi secondi."""
import sys
import torch
import torch.nn.functional as F

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
B, T, C, K = (2, 4096, 8192, 4) if dev == "cuda" else (1, 64, 256, 4)
x = torch.randn(B, T, C, device=dev) * 0.7
w = torch.randn(C, K, device=dev) * 0.3
g = torch.randn(B, T, C, device=dev)


def rif(x, w):                         # come transformers: (B, C, T) vista trasposta -> F.conv1d -> [..., :T]
    xt = x.transpose(1, 2)
    return F.conv1d(xt, w.unsqueeze(1), None, padding=K - 1, groups=C)[:, :, :T].transpose(1, 2)


def cand(x, w, acc, ordine):
    xp = F.pad(x.to(acc), (0, 0, K - 1, 0))
    y = torch.zeros(B, T, C, device=dev, dtype=acc)
    ks = range(K) if ordine == "su" else reversed(range(K))
    for k in ks:
        y = y + xp[:, k:k + T] * w[:, k].to(acc)
    return y.to(torch.float32)


xr = x.clone().requires_grad_(True)
y0 = rif(xr, w)
y0.backward(g)
dx0 = xr.grad
for acc in (torch.float64, torch.float32):
    for ordine in ("su", "giu"):
        y = cand(x, w, acc, ordine)
        xc = x.clone().requires_grad_(True)
        yc = cand(xc, w, acc, ordine)
        yc.backward(g)
        print(f"{dev} {str(acc):14s} {ordine:4s} avanti diversi {int((y != y0).sum())}/{y.numel()}  max {float((y - y0).abs().max()):.3e}"
              f" | indietro diversi {int((xc.grad != dx0).sum())}  max {float((xc.grad - dx0).abs().max()):.3e}", flush=True)
sys.exit(0)
