#!/usr/bin/env python3
"""test_cardan_down.py — 25/10 CPU proofs (synthetic, small dims, fp64) of the lever "CARDAN low-rank on expert DOWN".

  D1  rank 0 = unchanged: on a tiny synthetic MoE layer (real MoEStudente, synthetic TQ1_0R_1024/TQ1_0 bytes, 2nd planes,
      gate/up CARDAN active with random C) the MoE output and the gradients w.r.t. x, C_gate/up, D_gate/up of THIS copy
      without `cardan_down` are BITWISE equal to the ORIGINAL code tree (CARDAN_C0B/codice/thor_v2/e2e_studente.py +
      CARDAN_C0B/codice/cardan_kd_moduli.py); with `cardan_down` installed at C = 0 the output is still bitwise equal.
      Plus: argparse defaults of cardan_congiunto_scale.py leave the lever off.
  D2  gradient check: _GroupedLowRank1 == dense fp64 reference (forward/backward) + torch.autograd.gradcheck; the hooked
      MoE with random C_down, D_down equals the engine-semantics reference with down weights W_e + C_e D_down^T (plain
      basis, TQ1_0 is unrotated), gradients w.r.t. x, C_down, D_down (and the gate/up branch) equal autograd's.
  D3  fold roundtrip on synthetic TQ1_0 experts (K = 1 and K = 2 planes, exps2 slot map, one expert with C = 0, with and
      without trained scales theta): bytes re-encode at the bit, dequant(fold(W, C, D)) is closer to W + C D^T than
      dequant(W) for every corrected expert; C = 0 experts are byte-identical (or theta-only re-encoded).
  D4  (optional) _GruppiLowRank1 under TRITON_INTERPRET=1 on CPU == loop path.
Run: HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= OMP_NUM_THREADS=4 nice -n 15 \\
       @@ROOT:venv@@/bin/python3 test_cardan_down.py
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path

for _v in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
    os.environ[_v] = ""
os.environ["LEVE_DT"] = "float64"
import numpy as np  # noqa: E402
import torch  # noqa: E402

QUI = Path(__file__).resolve().parent
ORIG = Path("@@ROOT:models@@/gguf/tony-tern/OBIETTIVO_C/CARDAN_C0B/codice")
sys.path.insert(0, str(QUI))
import test_rilq_piani_extra as TX  # noqa: E402  (synthetic MoE fixture, this copy's e2e_studente as TX.S)
import test_cardan_kd as TK  # noqa: E402  (engine-semantics torch reference with dW)
import cardan_kd_moduli as CK  # noqa: E402
import cardan_piega_down as CPD  # noqa: E402

S = TX.S
F64 = torch.float64
torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "4")))
assert Path(S.__file__).resolve().parent == (QUI / "thor_v2").resolve(), S.__file__


def _rel(a, b):
    return float((a.detach() - b.detach()).abs().max() / b.detach().abs().max().clamp_min(1e-300))


def _carica(nome: str, path: Path):
    spec = importlib.util.spec_from_file_location(nome, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[nome] = m                     # dataclasses need the module registered
    spec.loader.exec_module(m)
    return m


SECONDI = {"gate": [1, -1, 0, -1], "up": [-1, 0, -1, 1], "down": [2, 0, -1, 1]}


def _moe_con(Smod, tens, seme_mod: int):
    """= TX._moe with an arbitrary e2e_studente module (original or this copy)."""
    fam = {n: Smod.famiglia_da_tensore(tens[f"blk.0.{Smod.FAMIGLIE[n]}.weight"], n) for n in Smod.FAMIGLIE}
    es = Smod.StratoEsperti(fam, Smod.RICETTE["B64"], Smod.motore("torch"))
    for n, (f2, m2) in Smod.secondi_piani_esperti(tens, 0, fam, "cpu").items():
        es.fam2[n], es.mappa2[n] = f2, m2
    g = torch.Generator().manual_seed(seme_mod)
    router = torch.randn(TX.E, TX.H, generator=g, dtype=F64) * 0.05
    sh = torch.nn.Linear(TX.H, TX.H, bias=False).double()
    shg = torch.nn.Linear(TX.H, 1, bias=False).double()
    with torch.no_grad():
        sh.weight.copy_(torch.randn(TX.H, TX.H, generator=g, dtype=F64) * 0.01)
        shg.weight.copy_(torch.randn(1, TX.H, generator=g, dtype=F64) * 0.05)
    for p in (*sh.parameters(), *shg.parameters()):
        p.requires_grad_(False)
    return Smod.MoEStudente(router, sh, shg, es, TX.K), router, sh, shg


def _cardan_gu(CKmod, seme=7, r=3):
    g = torch.Generator().manual_seed(seme)
    Dg0 = torch.randn(TX.H, r, generator=g, dtype=F64) / TX.H ** 0.5
    Du0 = torch.randn(TX.H, r, generator=g, dtype=F64) / TX.H ** 0.5
    m = CKmod.CardanStrato(0, TX.E, TX.FF, TX.H, r, Dg0, Du0, dtype=F64)
    with torch.no_grad():
        m.C_gate.copy_(torch.randn(TX.E, TX.FF, r, generator=g, dtype=F64) * 0.02)
        m.C_up.copy_(torch.randn(TX.E, TX.FF, r, generator=g, dtype=F64) * 0.02)
    return m


def _passo(moe, x, m, gy=None):
    xs = x.clone().requires_grad_(True)
    y = moe(xs[None])[0]
    if gy is None:
        gy = torch.randn(y.shape, generator=torch.Generator().manual_seed(11), dtype=F64)
    (y * gy).sum().backward()
    gr = {"x": xs.grad.clone(), **{n: p.grad.clone() for n, p in m.named_parameters()}}
    return y.detach().clone(), gr


# ═══════════════════════════════ D1 ═══════════════════════════════
def test_rango_zero_invariato():
    So = _carica("e2e_studente_originale", ORIG / "thor_v2" / "e2e_studente.py")
    CKo = _carica("cardan_kd_moduli_originale", ORIG / "cardan_kd_moduli.py")
    tens, _, _, _ = TX._strato(11, SECONDI)
    x = torch.randn(12, TX.H, generator=torch.Generator().manual_seed(5), dtype=F64)
    moe_o, *_ = _moe_con(So, tens, 3)
    moe_n, *_ = _moe_con(S, tens, 3)
    mo, mn = _cardan_gu(CKo), _cardan_gu(CK)
    moe_o.cardan, moe_n.cardan = mo, mn
    yo, go = _passo(moe_o, x, mo)
    yn, gn = _passo(moe_n, x, mn)
    assert torch.equal(yo, yn), "copy without cardan_down must equal the original at the bit"
    for k in go:
        assert torch.equal(go[k], gn[k]), k
    # cardan_down installed with C = 0: forward still at the bit
    dn = CK.CardanDown(0, TX.E, TX.H, TX.FF, 4, torch.randn(TX.FF, 4, dtype=F64) / TX.FF ** 0.5, dtype=F64)
    moe_n.cardan_down = dn
    with torch.no_grad():
        y0 = moe_n(x[None])[0]
    assert torch.equal(y0, yo), "C_down = 0 must leave the student at the bit"
    # trainer flags: default off
    sys.argv = ["x"]
    import cardan_congiunto_scale as CCS
    a = CCS.argomenti(["--uscita", "/nonexistent", "--gguf", "/nonexistent"])
    assert a.down_lowrank_rango == 0 and a.base_a1_down is None and a.down_lowrank_lr is None
    print(f"D1 rank 0: output + {len(go)} gradients BITWISE equal to the original tree; C_down = 0 at the bit; "
          f"trainer default --down-lowrank-rango 0")


# ═══════════════════════════════ D2 ═══════════════════════════════
def test_gradiente():
    g = torch.Generator().manual_seed(1)
    E, R, r = 5, 12, 3
    cnt = [3, 0, 4, 1, 2]
    offs = [0] + list(np.cumsum(cnt))
    z = torch.randn(offs[-1], r, generator=g, dtype=F64, requires_grad=True)
    C = torch.randn(E, R, r, generator=g, dtype=F64, requires_grad=True)
    y = CK._GroupedLowRank1.apply(z, C, offs)
    eid = torch.repeat_interleave(torch.arange(E), torch.tensor(cnt))
    ref = torch.einsum("srk,sk->sr", C[eid], z)
    assert _rel(y, ref) < 1e-14
    gy = torch.randn_like(ref)
    ga = torch.autograd.grad((y * gy).sum(), (z, C))
    gb = torch.autograd.grad((ref * gy).sum(), (z, C))
    assert max(_rel(a, b) for a, b in zip(ga, gb)) < 1e-13
    assert torch.autograd.gradcheck(lambda a, b: CK._GroupedLowRank1.apply(a, b, offs), (z, C))
    # hooked MoE vs engine-semantics reference with W_down_e + C_e D_down^T (and the gate/up branch on top)
    tens, W, W2, mappe = TX._strato(11, SECONDI)
    moe, router, sh, shg = TX._moe(tens, 3)
    mgu = _cardan_gu(CK)
    moe.cardan = mgu
    rd = 4
    gd = torch.Generator().manual_seed(21)
    dn = CK.CardanDown(0, TX.E, TX.H, TX.FF, rd, torch.randn(TX.FF, rd, generator=gd, dtype=F64) / TX.FF ** 0.5,
                       dtype=F64)
    with torch.no_grad():
        dn.C_down.copy_(torch.randn(TX.E, TX.H, rd, generator=gd, dtype=F64) * 0.02)
    moe.cardan_down = dn
    x = torch.randn(12, TX.H, generator=torch.Generator().manual_seed(5), dtype=F64)
    xs = x.clone().requires_grad_(True)
    yv = moe(xs[None])[0]
    Rm = TK._R()
    xr = x.clone().requires_grad_(True)
    Cg, Cu = mgu.C_gate.detach().clone().requires_grad_(True), mgu.C_up.detach().clone().requires_grad_(True)
    Dg, Du = mgu.D_gate.detach().clone().requires_grad_(True), mgu.D_up.detach().clone().requires_grad_(True)
    Cd, Dd = dn.C_down.detach().clone().requires_grad_(True), dn.D_down.detach().clone().requires_grad_(True)
    dW = {"gate": torch.einsum("erk,ck->erc", Cg, Dg) @ Rm, "up": torch.einsum("erk,ck->erc", Cu, Du) @ Rm,
          "down": torch.einsum("erk,ck->erc", Cd, Dd)}
    ref = TK._riferimento_torch(xr, router, sh, shg, W, W2, mappe, dW)
    e_y = _rel(yv, ref)
    gyy = torch.randn(ref.shape, generator=torch.Generator().manual_seed(13), dtype=F64)
    (yv * gyy).sum().backward()
    (ref * gyy).sum().backward()
    errs = {"x": _rel(xs.grad, xr.grad), "C_down": _rel(dn.C_down.grad, Cd.grad), "D_down": _rel(dn.D_down.grad, Dd.grad),
            "C_gate": _rel(mgu.C_gate.grad, Cg.grad), "D_up": _rel(mgu.D_up.grad, Du.grad)}
    assert e_y < 1e-10 and max(errs.values()) < 1e-9, (e_y, errs)
    # gradcheck of the module itself on a tiny routed case
    pl = types.SimpleNamespace(offs=torch.tensor(offs))
    dm = CK.CardanDown(0, E, R, 6, r, torch.randn(6, r, generator=g, dtype=F64), dtype=F64)
    with torch.no_grad():
        dm.C_down.normal_(0, 0.3, generator=g)
    a_ = torch.randn(offs[-1], 6, generator=g, dtype=F64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda a, c, d: _cd_fun(dm, a, c, d, pl), (a_, dm.C_down, dm.D_down))
    print(f"D2 gradients: grouped == dense + gradcheck OK; MoE+down vs engine(W_down + C D^T) rel {e_y:.1e}, grads " +
          ", ".join(f"{k} {v:.1e}" for k, v in errs.items()) + "; CardanDown gradcheck OK")


def _cd_fun(m, a, c, d, pl):
    z = a @ d
    return CK._GroupedLowRank1.apply(z, c, pl.offs.tolist())


# ═══════════════════════════════ D3 ═══════════════════════════════
def _gp():
    sys.path.insert(0, "@@ROOT:models@@/build/leve/raffina")
    import comune as CM
    return CM.gp


class _FalsaSorgente:
    """Duck-typed GGUFPlaneSource for ONE layer of synthetic TQ1_0 down experts (+ optional exps2 with a slot map)."""

    def __init__(self, raw1, raw2, slot2, R, C):
        gp = _gp()
        self.gp = gp
        self.raw1, self.raw2, self.slot2 = raw1, raw2, slot2
        tensori = [types.SimpleNamespace(data=raw1, shape=[C, R, raw1.shape[0]])]
        if raw2 is not None:
            tensori.append(types.SimpleNamespace(data=raw2, shape=[C, R, raw2.shape[0]]))
        self.en = gp._Entry(0, "ffn_down_exps", tensori, gp.FORMATS[gp.T_TQ1_0], R, C, raw1.shape[0], slot2)

    def entry(self, key):
        return self.en

    def load(self, key):
        from rd_ternary.formato import TernaryTensor
        e = key.expert
        P, Sc = [], []
        for k in range(self.en.n_planes_of(e)):
            raw = self.raw1[e] if k == 0 else self.raw2[self.en.slot(1, e)]
            p, s = self.gp.decode_rows(raw, self.gp.T_TQ1_0, self.en.in_)
            P.append(p); Sc.append(s)
        return TernaryTensor(torch.from_numpy(np.stack(P)), torch.from_numpy(np.stack(Sc)), False, sign=None,
                             packing=self.en.fmt.packing)


def _ternarizza(W: np.ndarray):
    """naive TQ1_0 (absmean per 256-block): trits int8 + fp16 scales."""
    R, C = W.shape
    Wb = W.reshape(R, C // 256, 256)
    s = np.abs(Wb).mean(-1)
    T = np.clip(np.rint(Wb / np.maximum(s[..., None], 1e-12)), -1, 1).astype(np.int8).reshape(R, C)
    return T, s.astype(np.float16)


def _deq(gp, raw, C):
    t, s = gp.decode_rows(raw, gp.T_TQ1_0, C)
    R = t.shape[0]
    return (t.astype(np.float64).reshape(R, -1, 256) * s.astype(np.float64)[..., None]).reshape(R, C)


def test_piega_tq1_0():
    gp = _gp()
    rng = np.random.default_rng(3)
    E, R, C, r = 4, 64, 512, 4
    raw1 = np.stack([gp.encode_rows(*_ternarizza(rng.standard_normal((R, C)) * 0.05), gp.T_TQ1_0) for _ in range(E)])
    slot2 = np.array([0, -1, 1, -1])
    raw2 = np.stack([gp.encode_rows(*_ternarizza(rng.standard_normal((R, C)) * 0.02), gp.T_TQ1_0) for _ in range(2)])
    g = torch.Generator().manual_seed(4)
    D = torch.randn(C, r, generator=g, dtype=F64) / C ** 0.5
    Cd = torch.randn(E, R, r, generator=g, dtype=F64) * 0.01
    Cd[3] = 0.0                                                                     # expert 3: no correction
    for con_theta in (False, True):
        theta = (torch.randn(E, R, C // 256, generator=g, dtype=torch.float32) * 0.05) if con_theta else None
        src = _FalsaSorgente(raw1.copy(), raw2.copy(), slot2, R, C)
        with tempfile.TemporaryDirectory() as td:
            riga = CPD.piega_strato_down(0, {"D_down": D, "C_down": Cd}, src, torch.device("cpu"), F64, 2, Path(td),
                                         None, theta=theta)
            o1 = np.load(Path(td) / "blk.0.ffn_down_exps.weight.npy")
            o2 = np.load(Path(td) / "blk.0.ffn_down_exps2.weight.npy")
        assert o1.shape == raw1.shape and o1.dtype == np.uint8 and o2.shape == raw2.shape
        migl = []
        for e in range(E):
            Q = _deq(gp, raw1[e], C)
            F_ = _deq(gp, o1[e], C)
            if slot2[e] >= 0:
                Q = Q + _deq(gp, raw2[slot2[e]], C)
                F_ = F_ + _deq(gp, o2[slot2[e]], C)
            if theta is not None:                                                   # host = trained scales
                _, s0 = gp.decode_rows(raw1[e], gp.T_TQ1_0, C)
                t0, _ = gp.decode_rows(raw1[e], gp.T_TQ1_0, C)
                s1 = (s0.astype(np.float64) * np.exp(theta[e].double().numpy())).astype(np.float16).astype(np.float64)
                Q = Q - _deq(gp, raw1[e], C) + (t0.astype(np.float64).reshape(R, -1, 256) * s1[..., None]).reshape(R, C)
            tgt = Q + (Cd[e] @ D.T).numpy()
            if e == 3:
                if theta is None:
                    assert np.array_equal(o1[e], raw1[e]), "C = 0 expert must keep its bytes"
                else:
                    assert np.abs(F_ - Q).max() == 0.0, "C = 0 expert with theta = trained scales, same trits"
                continue
            e_host = np.linalg.norm(Q - tgt)
            e_fold = np.linalg.norm(F_ - tgt)
            migl.append(1 - (e_fold / e_host) ** 2)
            assert e_fold < e_host, (e, e_fold, e_host)
        print(f"D3 fold TQ1_0 (theta {'on' if con_theta else 'off'}, K=1/2 planes): captured "
              f"{', '.join(f'{100 * m:.1f}%' for m in migl)} of |C D^T|^2 per corrected expert; registry captured "
              f"{100 * riga['catturata']:.1f} %; C=0 expert {'byte-identical' if not con_theta else 'theta-only'}")


# ═══════════════════════════════ D4 (optional) ═══════════════════════════════
_D4 = r'''
import os, sys, numpy as np, torch
sys.path.insert(0, sys.argv[1])
import cardan_kd_moduli as CK
torch.manual_seed(0)
E, R, r, cnt = 6, 40, 5, [70, 0, 3, 130, 1, 20]
offs = torch.tensor([0] + list(np.cumsum(cnt)))
z = torch.randn(int(offs[-1]), r, requires_grad=True); C = torch.randn(E, R, r)      # C frozen: forward + dz only
tab = CK.tabella_riquadri(offs, z.shape[0], CK._GR_BM)
y1 = CK._GruppiLowRank1.apply(z, C, offs, tab); y0 = CK._GroupedLowRank1.apply(z, C, offs.tolist())
g = torch.randn_like(y0)
a = torch.autograd.grad((y1 * g).sum(), z)[0]; b = torch.autograd.grad((y0 * g).sum(), z)[0]
e = max(float((y1 - y0).abs().max()), float((a - b).abs().max()))
print(f"D4 gruppi (TRITON_INTERPRET, forward + dz; the dC reduction kernel of the ORIGINAL gate/up path also cannot run "
      f"under the interpreter) vs loop: max abs diff {e:.2e}"); assert e < 1e-4
'''


def test_gruppi_interprete():
    env = dict(os.environ, TRITON_INTERPRET="1", HIP_VISIBLE_DEVICES="", CUDA_VISIBLE_DEVICES="", ROCR_VISIBLE_DEVICES="")
    p = subprocess.run([sys.executable, "-c", _D4, str(QUI)], env=env, capture_output=True, text=True, timeout=600)
    if p.returncode != 0:
        print(f"D4 SKIPPED/FAILED under the Triton interpreter (rc {p.returncode}): {p.stderr.strip()[-400:]}")
        return False
    print(p.stdout.strip())
    return True


# ═══════════════════════════════ D5 ═══════════════════════════════
def test_base_installa_salva():
    """basis builder (pure part) lowers the residual; installa_down reads the npz layout it writes (D from file, C = 0,
    Gaussian fallback for a missing layer); CK.salva/carica round-trip of the down state."""
    import cardan_costruisci_down as CCD
    g = torch.Generator().manual_seed(8)
    E, Hh, FF, r = 6, 32, 24, 3
    U = torch.randn(Hh, r, generator=g, dtype=F64)
    B = torch.randn(FF, r, generator=g, dtype=F64)
    Rs = (torch.einsum("hk,ek,fk->ehf", U, torch.randn(E, r, generator=g, dtype=F64), B)
          + 0.01 * torch.randn(E, Hh, FF, generator=g, dtype=F64)).float()
    for Sig in (torch.eye(FF, dtype=F64), (lambda A: A @ A.T / FF + 0.1 * torch.eye(FF, dtype=F64))(
            torch.randn(FF, FF, generator=g, dtype=F64))):
        D16, C16, E0, E1 = CCD.base_down(Rs, Sig, r, lotto=4)
        assert tuple(D16.shape) == (FF, r) and tuple(C16.shape) == (E, Hh, r) and E1 < 0.05 * E0, (E0, E1)
    with tempfile.TemporaryDirectory() as td:
        np.savez(Path(td) / "strato_00.npz", **{"blk.0.ffn_down_lrd.weight": D16.T.contiguous().to(torch.float16)
                                                .numpy().view(np.uint8)})
        fam = types.SimpleNamespace(E=E, R=Hh, C=FF)
        mlp0, mlp1 = types.SimpleNamespace(esperti=types.SimpleNamespace(fam={"down": fam})), \
            types.SimpleNamespace(esperti=types.SimpleNamespace(fam={"down": fam}))
        st = types.SimpleNamespace(moduli={0: types.SimpleNamespace(mlp=mlp0), 1: types.SimpleNamespace(mlp=mlp1)},
                                   ric=types.SimpleNamespace(dtype_gemm=torch.float16))
        mods = CK.installa_down(st, [0, 1], r, Path(td), "cpu")
        assert mlp0.cardan_down is mods[0] and torch.equal(mods[0].D_down.double(), D16.float().double())
        assert float(mods[0].C_down.abs().max()) == 0 and mods[1].D_down.shape == (FF, r)
        with torch.no_grad():
            mods[1].C_down.normal_(generator=g)
        CK.salva(mods, Path(td) / "cardan_down_x.pt", {"passo": 7})
        ref = {L: {k: v.clone() for k, v in m.state_dict().items()} for L, m in mods.items()}
        with torch.no_grad():
            for m in mods.values():
                m.C_down.zero_()
        assert CK.carica(mods, Path(td) / "cardan_down_x.pt")["passo"] == 7
        assert all(torch.equal(ref[L][k], v) for L, m in mods.items() for k, v in m.state_dict().items())
    print(f"D5 basis builder: residual energy {E1 / E0:.2e} of the start; installa_down reads lrd, C = 0, Gaussian "
          f"fallback; salva/carica round-trip OK")


if __name__ == "__main__":
    test_rango_zero_invariato()
    test_gradiente()
    test_piega_tq1_0()
    test_base_installa_salva()
    test_gruppi_interprete()
    print("ALL DONE")
