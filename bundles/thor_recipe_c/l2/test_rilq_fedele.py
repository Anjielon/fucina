"""RILQ-FEDELE: perdita_rilq (value and manual gradient) against torch autograd, CPU, small random head."""
import sys, torch
sys.argv = ["x"]
import importlib.util
spec = importlib.util.spec_from_file_location("rf", "@@ROOT:models@@/gguf/tony-tern/CARDAN/codice/rilq_fedele.py")
rf = importlib.util.module_from_spec(spec); spec.loader.exec_module(rf)
torch.manual_seed(0)
T, H, V = 37, 16, 50
W = torch.randn(V, H, dtype=torch.float64)
h = torch.randn(T, H, dtype=torch.float64, requires_grad=True)
hT = torch.randn(T, H, dtype=torch.float64)
tgt = torch.randint(0, V, (T,)); ok = torch.rand(T) > 0.2; ok[-1] = False
wm, wc = 0.5, 0.5
lp = torch.log_softmax(h @ W.T, -1)
ce = -(lp.gather(1, tgt[:, None])[:, 0] * ok).sum() / ok.sum()
L = wm * ((h - hT) ** 2).mean() + wc * ce
L.backward()
mse, ce2, g = rf.perdita_rilq(h.detach(), hT, tgt, ok, lambda x: x.double() @ W.T, lambda q: q.double() @ W,
                             wm, wc, 8, True)
v = wm * mse + wc * ce2
assert abs(v - float(L)) < 1e-6, (v, float(L))
err = float((g.double() - h.grad).abs().max())
assert err < 1e-5, err
print(f"perdita_rilq OK: valore {v:.6f} = autograd {float(L):.6f}, gradiente max|diff| {err:.2e}")
