#!/usr/bin/env python3
"""test_cardan_kd.py — CPU proofs (fp64 references) of the CARDAN distillation branch and of its fold (9/10).

  K1  grouped low-rank product: forward and backward (z, C) equal a dense fp64 reference built expert by expert;
      torch.autograd.gradcheck on a small case.
  K2  student hook (MoEStudente.cardan, B64 recipe = rotated activations, fp64): with C = 0 the MoE output is the
      student's AT THE BIT; with random C, D the output equals the ENGINE-semantics reference (test_rilq_piani_extra)
      whose expert weights are W_e + C_e D^T R (R = the student's rotation, x_rot = R x), with the second plane of the
      hot experts present; gradients w.r.t. x, C and D equal the autograd fp64 reference.
  K3  rotation: the student's `ruota` and raffina/comune.ruota are the same R (the fold's basis = the training basis).
  K4  batched JARQ (cardan_piega.jarq_lotto) == raffina/solutori.jarq expert by expert (fp64): same objective chain,
      same trits, same fp16 scales; K = 1 and K = 2 planes.
  K5  fold on REAL PV12P data (layer 10, 3 experts per family, Gq of the quantized flow): bytes re-encoded at the bit
      (asserted inside piega_strato), experts with C = 0 untouched byte for byte, captured fraction > 0, no row worse
      than the host.
Run: HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= OMP_NUM_THREADS=6 nice -n 15 \\
       @@ROOT:venv@@/bin/python3 test_cardan_kd.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

for _v in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
    os.environ[_v] = ""
os.environ["LEVE_DT"] = "float64"
import numpy as np  # noqa: E402
import torch  # noqa: E402

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import test_rilq_piani_extra as TX  # noqa: E402  (synthetic MoE fixture + engine-semantics reference)
import cardan_kd_moduli as CK  # noqa: E402
import cardan_piega as CP  # noqa: E402

S = TX.S
torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "4")))
F64 = torch.float64


def _rel(a, b):
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-300))


# ═══════════════════════════════ K1 ═══════════════════════════════
def test_gruppi_contro_denso():
    g = torch.Generator().manual_seed(1)
    E, R, r = 5, 12, 3
    cnt = [3, 0, 4, 1, 2]
    offs = [0] + list(np.cumsum(cnt))
    Sn = offs[-1]
    zg = torch.randn(Sn, r, generator=g, dtype=F64, requires_grad=True)
    zu = torch.randn(Sn, r, generator=g, dtype=F64, requires_grad=True)
    Cg = torch.randn(E, R, r, generator=g, dtype=F64, requires_grad=True)
    Cu = torch.randn(E, R, r, generator=g, dtype=F64, requires_grad=True)
    yg, yu = CK._GroupedLowRank.apply(zg, zu, Cg, Cu, offs)
    eid = torch.repeat_interleave(torch.arange(E), torch.tensor(cnt))
    rg = torch.einsum("srk,sk->sr", Cg[eid], zg)
    ru = torch.einsum("srk,sk->sr", Cu[eid], zu)
    assert _rel(yg, rg) < 1e-14 and _rel(yu, ru) < 1e-14
    gy = (torch.randn_like(yg), torch.randn_like(yu))
    gr = torch.autograd.grad((yg * gy[0]).sum() + (yu * gy[1]).sum(), (zg, zu, Cg, Cu))
    rr = torch.autograd.grad((rg * gy[0]).sum() + (ru * gy[1]).sum(), (zg, zu, Cg, Cu))
    for a, b in zip(gr, rr):
        assert _rel(a, b) < 1e-13, _rel(a, b)
    assert torch.autograd.gradcheck(lambda a, b, c, d: CK._GroupedLowRank.apply(a, b, c, d, offs), (zg, zu, Cg, Cu))
    print("K1 grouped low-rank: forward/backward == dense fp64, gradcheck OK")


# ═══════════════════════════════ K2 ═══════════════════════════════
def _R():
    """R with x_rot = R x for the student's rotation (rows of ruota(I) are columns of R)."""
    return S.ruota(torch.eye(TX.H, dtype=F64)).T.contiguous()


def test_gancio_studente():
    secondi = {"gate": [1, -1, 0, -1], "up": [-1, 0, -1, 1], "down": [2, 0, -1, 1]}
    tens, W, W2, mappe = TX._strato(11, secondi)
    moe, router, sh, shg = TX._moe(tens, 3)
    x = torch.randn(12, TX.H, generator=torch.Generator().manual_seed(5), dtype=F64)
    with torch.no_grad():
        y0 = moe(x[None])[0].clone()
    g = torch.Generator().manual_seed(7)
    r = 3
    Dg0 = torch.randn(TX.H, r, generator=g, dtype=F64) / TX.H ** 0.5
    Du0 = torch.randn(TX.H, r, generator=g, dtype=F64) / TX.H ** 0.5
    m = CK.CardanStrato(0, TX.E, TX.FF, TX.H, r, Dg0, Du0, dtype=F64)
    moe.cardan = m
    with torch.no_grad():
        y1 = moe(x[None])[0]
    assert torch.equal(y0, y1), "C = 0 must leave the student at the bit"
    with torch.no_grad():
        m.C_gate.copy_(torch.randn(TX.E, TX.FF, r, generator=g, dtype=F64) * 0.02)
        m.C_up.copy_(torch.randn(TX.E, TX.FF, r, generator=g, dtype=F64) * 0.02)
    xs = x.clone().requires_grad_(True)
    y = moe(xs[None])[0]
    # engine-semantics reference with W_e(orig) + C_e D^T R  (D in the rotated domain)
    Rm = _R()
    W_ = {n: v.copy() for n, v in W.items()}
    Cg_r, Cu_r = m.C_gate.detach().clone().requires_grad_(True), m.C_up.detach().clone().requires_grad_(True)
    Dg_r, Du_r = m.D_gate.detach().clone().requires_grad_(True), m.D_up.detach().clone().requires_grad_(True)
    xr = x.clone().requires_grad_(True)
    dWg = torch.einsum("erk,ck->erc", Cg_r, Dg_r) @ Rm
    dWu = torch.einsum("erk,ck->erc", Cu_r, Du_r) @ Rm
    # the reference takes numpy weights: rebuild it in torch to keep the graph through C and D
    ref = _riferimento_torch(xr, router, sh, shg, W_, W2, mappe, {"gate": dWg, "up": dWu})
    e_y = _rel(y.detach(), ref.detach())
    torch.manual_seed(11)
    gy = torch.randn_like(ref.detach())
    (y * gy).sum().backward()
    (ref * gy).sum().backward()
    errs = {"x": _rel(xs.grad, xr.grad), "C_gate": _rel(m.C_gate.grad, Cg_r.grad), "C_up": _rel(m.C_up.grad, Cu_r.grad),
            "D_gate": _rel(m.D_gate.grad, Dg_r.grad), "D_up": _rel(m.D_up.grad, Du_r.grad)}
    print(f"K2 hook: |student+CARDAN - engine(W + C D^T R)| rel {e_y:.2e}; gradients rel " +
          ", ".join(f"{k} {v:.1e}" for k, v in errs.items()))
    assert e_y < 1e-10 and max(errs.values()) < 1e-9, (e_y, errs)


def _riferimento_torch(x, router, sh, shg, W, W2, mappe, dW):
    """test_rilq_piani_extra._riferimento_motore in torch, with gate/up weights W_e + dW_e (graph kept)."""
    import torch.nn.functional as F
    prob = torch.softmax(x @ router.T, -1)
    idx = torch.topk(prob, TX.K, -1).indices
    w = prob.gather(-1, idx)
    w = w / w.sum(-1, keepdim=True)
    Wt = {n: torch.from_numpy(v) for n, v in W.items()}
    W2t = {n: torch.from_numpy(v) for n, v in W2.items()}
    righe = []
    for t in range(x.shape[0]):
        acc = torch.zeros(TX.H, dtype=F64)
        for j in range(TX.K):
            e = int(idx[t, j])

            def mm(n, v):
                Wn = Wt[n][e] + (dW[n][e] if n in dW else 0)
                y = Wn @ v
                if n in W2t and mappe[n][e] >= 0:
                    y = y + W2t[n][int(mappe[n][e])] @ v
                return y
            a = F.silu(mm("gate", x[t])) * mm("up", x[t])
            acc = acc + w[t, j] * mm("down", a)
        righe.append(acc)
    return torch.stack(righe) + torch.sigmoid(shg(x)) * sh(x)


# ═══════════════════════════════ K3 ═══════════════════════════════
def test_rotazione_coerente():
    sys.path.insert(0, CP.RAFF)
    import comune as CM
    v = torch.randn(7, 2048, generator=torch.Generator().manual_seed(3), dtype=F64)
    e = _rel(S.ruota(v), CM.ruota(v))
    print(f"K3 rotation student vs raffina/comune: rel {e:.1e}")
    assert e < 1e-12


# ═══════════════════════════════ K4 ═══════════════════════════════
def _caso(seed, K, R=24, C=512, ng=4):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(C, C, generator=g, dtype=F64)
    H = A @ A.T / C + 0.05 * torch.eye(C, dtype=F64)
    H = H / torch.diagonal(H).mean()
    T = torch.randint(-1, 2, (K, R, C), generator=g).to(torch.int8)
    S_ = (torch.rand(K, R, ng, generator=g, dtype=F64) * 0.02 + 0.005).half().double()
    import solutori as SO
    W = SO.ricostruisci(T, S_) + 0.004 * torch.randn(R, C, generator=g, dtype=F64)
    return W, H, T, S_


def test_jarq_lotto_uguale():
    sys.path.insert(0, CP.RAFF)
    import solutori as SO
    assert SO.F64 == torch.float64
    for K in (1, 2):
        casi = [_caso(10 * K + i, K) for i in range(3)]
        W = torch.stack([c[0] for c in casi]); H = torch.stack([c[1] for c in casi])
        T = torch.stack([c[2] for c in casi]); S_ = torch.stack([c[3] for c in casi])
        T2, S2, cat = CP.jarq_lotto(W, H, T, S_, giri=3, nu2=1e-4)
        for j, (Wj, Hj, Tj, Sj) in enumerate(casi):
            Tr, Sr, catr = SO.jarq(Wj, Hj, Tj, Sj, list(range(K)), giri=3, nu2=1e-4, tau=-1.0)
            same_t = float((T2[j] == Tr).float().mean())
            e_s = _rel(S2[j], Sr.double())
            fr = float(SO.obiettivo(Wj, SO.ricostruisci(Tr, Sr.double()), Hj + 1e-4 * torch.eye(Wj.shape[1], dtype=F64)).sum())
            fl = float(SO.obiettivo(Wj, SO.ricostruisci(T2[j], S2[j]), Hj + 1e-4 * torch.eye(Wj.shape[1], dtype=F64)).sum())
            assert same_t == 1.0 and e_s < 1e-12 and abs(fl / fr - 1) < 1e-12, (K, j, same_t, e_s, fl, fr)
        print(f"K4 batched JARQ K={K}: trits/scales/objective identical to solutori.jarq on 3 cases "
              f"(host {cat['ospite']:.4e} -> giro3 {cat['giro_3']:.4e})")


# ═══════════════════════════════ K5 ═══════════════════════════════
def test_piega_dati_veri():
    import types
    sys.path.insert(0, CP.RAFF)
    import comune as CM
    from rd_ternary.formato import TensorKey
    G = "@@ROOT:models@@/gguf/tony-tern/PV12P/Thor-T-pv12p.gguf"
    L, r, ne = 10, 4, 3
    g = torch.Generator().manual_seed(9)
    stato = {}
    for f in ("gate", "up"):
        D = torch.randn(2048, r, generator=g, dtype=F64) / 2048 ** 0.5
        C = torch.zeros(256, 512, r, dtype=F64)
        C[:ne] = torch.randn(ne, 512, r, generator=g, dtype=F64) * 0.01
        stato[f"D_{f}"], stato[f"C_{f}"] = D, C
    with tempfile.TemporaryDirectory(dir=os.getenv("CARDAN_TEST_TMP", "@@ROOT:models@@/gguf/tony-tern/CARDAN")) as td:
        td = Path(td)
        with open(td / "reg.jsonl", "w") as reg:
            CP.piega_strato(L, stato, G, torch.device("cpu"), F64, 4, td, reg, esperti=ne)
        righe = [__import__("json").loads(x) for x in open(td / "reg.jsonl")]
        src = CM.gp.GGUFPlaneSource(G, layers=[L], bases=["ffn_gate_exps", "ffn_up_exps"])
        for f in ("gate", "up"):
            e_ = src.entry(TensorKey(L, 0, f))
            host = np.asarray(e_.tensors[0].data).reshape(256, 512, -1)
            new = np.load(td / f"blk.{L}.ffn_{f}_exps.weight.npy")
            assert new.shape == host.shape and new.dtype == np.uint8
            assert np.array_equal(new[ne:], host[ne:]), "experts with C = 0 must be untouched"
            assert not np.array_equal(new[:ne], host[:ne])
            if len(e_.tensors) > 1:
                h2 = np.asarray(e_.tensors[1].data).reshape(int(e_.tensors[1].shape[2]), 512, -1)
                n2 = np.load(td / f"blk.{L}.ffn_{f}_exps2.weight.npy")
                toccati = {int(e_.slot2[e]) for e in range(ne) if e_.slot2[e] >= 0}
                for s in range(h2.shape[0]):
                    if s not in toccati:
                        assert np.array_equal(n2[s], h2[s])
        for rr in righe:
            print(f"K5 fold L{rr['strato']} {rr['famiglia']}: {rr['esperti_piegati']} experts, captured "
                  f"{100 * rr['catturata']:.1f} % (scales only {100 * rr['catturata_sole_scale']:.1f} %), H {rr['hessiane']}")
            assert rr["catturata"] > 0.0 and rr["residuo_dopo_piega_H"] <= rr["energia_correzione_H"] * (1 + 1e-9)
        _ = types


# ═══════════════════════════════ K6 ═══════════════════════════════
def _driver(out, passi, extra=()):
    """cardan_kd.main on a fake 2-layer student whose MoE blocks are REAL MoEStudente (synthetic bytes, B64 recipe)."""
    import types
    import cardan_kd as KD
    import test_rilq_thor_shexp as TS
    V, T_ = 64, 24
    secondi = {"gate": [1, -1, 0, -1], "up": [-1, 0, -1, 1]}

    class _Blocco(torch.nn.Module):
        def __init__(s_, mlp):
            super().__init__()
            s_.mlp = mlp

        def forward(s_, x):
            return x + s_.mlp(x[None])[0]

    def fabbrica(seme):
        class Stud:
            def __init__(s_, **kw):
                g = torch.Generator().manual_seed(seme)
                s_.device = torch.device("cpu")
                s_.ric = TX.RIC
                s_.strati = [0, 1]
                s_.E = torch.randn(V, TX.H, generator=g, dtype=F64)
                s_.moduli = {}
                for L in s_.strati:
                    tens, *_ = TX._strato(11 + L + seme, secondi)
                    s_.moduli[L] = _Blocco(TX._moe(tens, 3 + L + seme)[0])
                s_.strati_allenati = None

            def parametri(s_):
                return [p for m in s_.moduli.values() for p in m.parameters()]

            def _avanti(s_, ids):
                x = s_.E[ids.reshape(-1) % V]
                ing = []
                for L in s_.strati:
                    ing.append(x)
                    x = s_.moduli[L](x)
                return x, ing

            def nascosto(s_, ids, inizi=None):
                return s_._avanti(ids)[0]

            def avanti(s_, ids, inizi=None):
                with torch.no_grad():
                    x, ing = s_._avanti(ids)
                return {"xL": x, "ingressi": ing}

            def indietro(s_, stato, g):
                i0 = min(s_.strati_allenati)
                for L in reversed(s_.strati):
                    if L < i0:
                        break
                    xin = stato["ingressi"][L].detach().requires_grad_(L > i0)
                    with torch.enable_grad():
                        s_.moduli[L](xin).backward(g)
                    g = xin.grad

            def esce(s_, x):
                return x
        return Stud

    g = torch.Generator().manual_seed(5)
    WS = torch.randn(V, TX.H, generator=g) / TX.H ** 0.5
    WT = WS + 0.3 * torch.randn(V, TX.H, generator=g) / TX.H ** 0.5
    maestro = fabbrica(99)()
    fS = types.SimpleNamespace(Studente=fabbrica(7))
    fA = TS._allena_finto(WS, WT, maestro)
    vecchio_T = TS.T
    TS.T = T_
    old = {k: sys.modules.get(k) for k in ("e2e_studente", "e2e_allena")}
    sys.modules["e2e_studente"], sys.modules["e2e_allena"] = fS, fA
    cicli_vero = KD.CICLI
    os.makedirs(out, exist_ok=True)
    KD.CICLI = os.path.join(out, "cicli_finto.tsv")
    try:
        return KD.main(["--uscita", out, "--gguf", "/dev/null", "--nome", "SINT", "--teacher", "finto",
                        "--strati", "0-1", "--rango", "3", "--passi", str(passi), "--heldout-righe", "3",
                        "--heldout-ogni", "1", "--lr-c", "3e-2", "--lr-d", "1e-3", "--riscaldamento", "1",
                        "--device", "cpu", "--pezzo-kl", "64", *extra])
    finally:
        KD.CICLI = cicli_vero
        TS.T = vecchio_T
        for k, v in old.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def test_driver_cardan_kd():
    import json
    with tempfile.TemporaryDirectory(dir=os.getenv("CARDAN_TEST_TMP", "@@ROOT:models@@/gguf/tony-tern/CARDAN")) as td:
        c = _driver(td + "/a", 4)
        reg = open(td + "/a/registro.log").read()
        assert c == 0 and "AL BIT" in reg, reg[-500:]
        st = json.load(open(td + "/a/storia.json"))
        kl = [r["kl_held"] for r in st["storia"] if "kl_held" in r]
        assert kl[-1] < kl[0], kl
        d = torch.load(td + "/a/cardan_migliore.pt", weights_only=False)
        assert set(d["stato"]) == {0, 1} and float(d["stato"][0]["C_gate"].abs().max()) > 0
        print(f"K6 driver: identity gate AL BIT, held-out KL {kl[0]:.5f} -> {min(kl):.5f} in 4 steps, cardan_migliore.pt OK")
        c2 = _driver(td + "/b", 2, ["--kl-partenza-motore", str(kl[0] * 1.10), "--soglia-coerenza", "0.03"])
        assert c2 == 2 and "lo studente non e' il GGUF" in open(td + "/b/registro.log").read()
        c3 = _driver(td + "/c", 1, ["--kl-partenza-motore", str(kl[0] * 1.01), "--soglia-coerenza", "0.03"])
        assert c3 == 0
        print("K6 coherence gate: +10 % -> stop (code 2) before step 1; +1 % -> passes")


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_") and callable(f):
            f()
    print("TUTTE LE PROVE CARDAN-KD PASSATE")
