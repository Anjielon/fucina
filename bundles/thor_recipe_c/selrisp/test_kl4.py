"""SELRISP (10/10): CPU test of gradienti_densi.passo_kl4 / pacchi_kl4 with a mock student.
Checks: segments start at each question, loss positions = last prompt token, dL/dh equals autograd of
mean KL(pT4 || softmax(h W_letters^T)), MCLP parsing, label check, packing bound."""
import json, sys, tempfile, types
from pathlib import Path
import torch
sys.path.insert(0, "@@ROOT:models@@/build/leve/raffina")
_m = types.ModuleType("dkl_raccogli"); _m.Raccoglitore = object; sys.modules.setdefault("dkl_raccogli", _m)
import gradienti_densi as G  # noqa: E402

torch.manual_seed(0)
V, H = 50, 8
WS = torch.randn(V, H)
E = torch.randn(V, H)
class Rac:
    def __init__(self): self.n = 0
    def fine_passata(self, tag): self.n += 1
class St:
    def avanti(self, ids, inizi=None):
        self.ids, self.inizi = ids, inizi
        return {"xL": E[ids[0]].clone()}
    def esce(self, xL): return torch.tanh(xL) * 1.5
    def indietro(self, stato, g): self.g = g

d = Path(tempfile.mkdtemp())
Q, log = [], []
for i in range(7):
    ids = torch.randint(0, V, (5 + 3 * i,)).tolist()
    lab = i % 4
    Q.append({"i": i, "label": lab, "ids": ids, "lettere": [3, 7, 11, 13]})
    lp = torch.log_softmax(torch.randn(4), 0).tolist()
    log.append("x MCLP\t%d\t%d\t%d\t4\t%s\t3\t7\t11\t13" % (i, lab, 0, "\t".join(f"{v:.6f}" for v in lp)))
(d / "c.jsonl").write_text("\n".join(json.dumps(q) for q in Q))
(d / "t.log").write_text("\n".join(log) + "\n")
a = types.SimpleNamespace(calib_maestro=str(d / "t.log"), calib_jsonl=str(d / "c.jsonl"), seme=7, calib_n=0, calib_pacco=40)
pacchi, info = G.pacchi_kl4(a, print)
assert all(sum(len(q["ids"]) for q in p) <= 40 or len(p) == 1 for p in pacchi)
assert sum(len(p) for p in pacchi) == 7
st, rac, out = St(), Rac(), []
for p in pacchi:
    G.passo_kl4(st, WS, p, "cpu", rac, out)
    # reference with autograd
    ids = st.ids
    xL = E[ids[0]].clone().requires_grad_(True)
    h = torch.tanh(xL) * 1.5
    ult, o = [], 0
    for q in p:
        assert st.inizi[0, o] == 1
        o += len(q["ids"]); ult.append(o - 1)
    assert int(st.inizi.sum()) == len(p)
    z = h[ult] @ WS[[3, 7, 11, 13]].T
    pT = torch.stack([q["pT"] for q in p]).float()
    L = (pT * (pT.log() - torch.log_softmax(z, -1))).sum(-1).mean()
    L.backward()
    err = (xL.grad - st.g).abs().max().item()
    assert err < 1e-5, err
assert rac.n == len(pacchi) and len(out) == 7
print("OK", len(pacchi), "pacchi, gradiente = autograd, segmenti e posizioni corretti")
