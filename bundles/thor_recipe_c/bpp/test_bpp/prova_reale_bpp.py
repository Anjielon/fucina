#!/usr/bin/env python3
"""prova_reale_bpp.py — arm B++ on the REAL host GGUF of B+ (CARDAN_DOWN_1025/Thor-T-C-S7g10-DOWNH.gguf), CPU only, no
GPU, RSS < 2 GB per process (26/10/2026). Complements test_bpp.py (synthetic layers) with the real tensors:

  RR1 real Studente, layers 3 (full attention) and 4 (Gated DeltaNet), recipe B32, torch engine, CPU, densi_ternari tutte,
      allena_f32 tutte; residuals installed as the trainer does (attn/gdn + --residui-shexp, continuous, rank 16): the shexp
      bases are LineareTernaria (gate/up TQ1_0R_1024 rotated, down TQ1_0); start at the bit; presence of every shexp
      family; checkpointed avanti/indietro: every shexp B and every F32 master has a gradient; R-CAMMINO hook
      (router_distilla, KL_U 16+8 + mass at risk 4-9) on the REAL MoEStudente: value == the term on plain hooks, and
      gradient(W) - gradient(0) on the F32 masters (ffn_gate_inp, upstream norms) == full-graph autograd of the term
  RR2 basis convention (fold + LoRA ceiling): LineareTernaria(x) == x W_file^T with W_file = comune.deruota(stored planes) for
      gate/up_shexp (TQ1_0R_1024) and W_file = stored for down_shexp (TQ1_0); the other convention is far off
  RR3 dense fold cardan_piega_densi_gpu_veloce.py (--device cpu, fp64, subprocess) of a small rank-4 dW on the 3 shexp tensors
      of layer 3 + attn_q of layer 3: planes written, round trip at the bit (inside the fold), captured fraction > 0
      (measured 26/10: attn_q 0.52, gate/up_shexp 0.28/0.24, down_shexp 0.02). WARNING: the fold subprocess peaks at
      2.7 GB RSS (above the usual 2 GB cap): run RR3 only when that is allowed (PROVE="rr2 rr1 rr4" skips it)
  RR4 LoRA ceiling exporter ~/tony-forgia/stato/tetto_cardan_lora_down_1026_bpp.py on the host with a tiny synthetic tappa
      (subprocess): shexp lora_a = A, lora_b = diag(gamma) B, ne [in, r] / [r, out], host types in the summary JSON
Run: HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 nice -n 19 @@ROOT:venv@@/bin/python3 \\
       test_bpp/prova_reale_bpp.py [GGUF]
"""
from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(4)
QUI = Path(__file__).resolve().parent
COD = QUI.parent
sys.path.insert(0, str(COD))
os.environ.setdefault("E2E_GDN", "veloce"); os.environ.setdefault("E2E_GDN_MODULO", "gdn_veloce_corsa")
os.environ.setdefault("E2E_CONV_DIRETTA", "1"); os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
OSPITE = "@@ROOT:models@@/gguf/tony-tern/OBIETTIVO_C/CARDAN_DOWN_1025/Thor-T-C-S7g10-DOWNH.gguf"
GGUF = sys.argv[1] if len(sys.argv) > 1 else OSPITE
os.environ["LEVE_OSPITE"] = GGUF
os.environ.setdefault("LEVE_THREADS", "4")              # raffina/comune sets torch threads from it (default 8)
import cardan_kd_paper as KDP                       # noqa: E402,F401  (path setup of the thor_v2 tree)
import e2e_studente as S                            # noqa: E402
import e2e_allena as A                              # noqa: E402
import f32_blocco as FB                             # noqa: E402
import residui_ternari as RT                        # noqa: E402
import router_distilla as RDm                       # noqa: E402

PY = "@@ROOT:venv@@/bin/python3"
ESPORTA = "@@ROOT:forge@@/stato/tetto_cardan_lora_down_1026_bpp.py"
TMP = Path("@@ROOT:models@@/gguf/tony-tern/OBIETTIVO_C/SINTESI_1025/BPP/collaudo") / time.strftime("reale_%Y%m%d_%H%M%S")
TMP.mkdir(parents=True, exist_ok=True)
ESITI = []
SENZA_GPU = {"HIP_VISIBLE_DEVICES": "", "CUDA_VISIBLE_DEVICES": "", "ROCR_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "4",
             "MKL_NUM_THREADS": "4", "PYTHONDONTWRITEBYTECODE": "1"}


def rss() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20


def prova(nome):
    def dec(f):
        def run():
            t0 = time.time()
            try:
                m = f()
                ESITI.append((nome, True, m or ""))
                print(f"PASS {nome} {m or ''} [{time.time() - t0:.1f} s, RSS max {rss():.2f} GiB]", flush=True)
            except Exception as e:
                import traceback
                ESITI.append((nome, False, f"{type(e).__name__}: {e}"))
                print(f"FAIL {nome}: {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
        run.__name__ = f.__name__
        return run
    return dec


A.scegli_gdn(lambda m: print(m, flush=True))
ST = S.Studente(gguf=Path(GGUF), strati=[3, 4], ricetta="B32", motore_nome="torch", device="cpu", log=lambda *m: None,
                densi_ternari="tutte", allena_f32=FB.famiglie("tutte"))
for p_ in ST.parametri():
    p_.requires_grad_(False)                            # as the trainer
BASI = {f: ST.moduli[3].mlp.shared_expert.get_submodule(p.split(".")[-1])
        for f, p in (("ffn_gate_shexp", "gate_proj"), ("ffn_up_shexp", "up_proj"), ("ffn_down_shexp", "down_proj"))}
IDS = torch.tensor([[760, 6511, 314, 9338, 369, 11751, 13, 576, 3974, 7111, 11, 323, 4996, 279, 1879, 25]], dtype=torch.long)
print(f"Studente vero: strati {ST.strati}, F32 {ST.f32_conteggi}; RSS {rss():.2f} GiB", flush=True)


@prova("RR2 convenzione di base shexp (piega + LoRA): LineareTernaria(x) == x W_file^T, W_file = deruota(memorizzato) per R")
def rr2():
    sys.path.insert(0, "@@ROOT:models@@/build/leve/raffina")
    import comune as CM
    src = CM.gp.GGUFPlaneSource(GGUF, layers=[3])
    g = torch.Generator().manual_seed(3)
    out = {}
    for f, lin in BASI.items():
        d = CM.Denso(src, f"blk.3.{f}.weight")
        atteso_r = f != "ffn_down_shexp"
        assert d.rotated == atteso_r and lin.es.fam[lin.nome].ruotata == atteso_r, (f, d.rotated, lin.nome)
        Q0 = d.ricostruisci()                                       # stored basis, all planes, fp64 [out, in]
        W_file = CM.deruota(Q0) if d.rotated else Q0
        x = torch.randn(5, lin.in_features, generator=g)
        with torch.no_grad():
            y = lin(x).double()
        e_ok = float((y - x.double() @ W_file.T).norm() / y.norm())
        # the other convention: stored planes read as if they were in the original basis (only for the rotated types;
        # TQ1_0 down_shexp has 512 input columns, not even a multiple of the 1024-block of the rotation)
        e_no = float((y - x.double() @ Q0.T).norm() / y.norm()) if d.rotated else float("nan")
        assert e_ok < 2e-3 and (not d.rotated or e_no > 0.5), (f, e_ok, e_no)
        out[f] = (d.K, e_ok, e_no)
    return "(" + ", ".join(f"{f} K={k} giusta {a:.1e} sbagliata {b:.2f}" for f, (k, a, b) in out.items()) + ")"


@prova("RR1 Studente vero: residui shexp installati come il trainer, al bit, presenti, con gradiente; router sul MoE vero")
def rr1():
    st = ST
    with torch.no_grad():
        h0 = st.nascosto(IDS).clone()
    for f, lin in BASI.items():
        assert isinstance(lin, S.LineareTernaria), (f, type(lin))
    cfgD = RT.ConfigResidui(rango=16, piani=1, strati=[3, 4], famiglie=RT.famiglie_da("attn,gdn"), seme=20261007,
                            continuo=True)
    resD = RT.installa(st, cfgD, device=torch.device("cpu"))
    cfgS = RT.ConfigResidui(rango=16, piani=1, strati=[3, 4], famiglie=RT.famiglie_da("shexp"), seme=20261007,
                            continuo=True)
    resS = RT.installa(st, cfgS, device=torch.device("cpu"))
    assert len(resS) == 6 and all(isinstance(m.base, S.LineareTernaria) for m in
                                  (st.moduli[3].mlp.shared_expert.gate_proj, st.moduli[4].mlp.shared_expert.down_proj))
    with torch.no_grad():
        h1 = st.nascosto(IDS)
    assert torch.equal(h0, h1), "partenza non al bit"
    for n in [f"blk.3.{f}" for f in ("ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp")]:   # presence (the trainer's rule)
        with torch.no_grad():
            resS[n].B.normal_(0.0, 1e-3); h2 = st.nascosto(IDS)
            resS[n].B.zero_(); h3 = st.nascosto(IDS)
        assert not torch.equal(h2, h1) and torch.equal(h3, h1), n
    for p_ in RT.parametri(resD) + RT.parametri(resS):
        p_.requires_grad_(True)
    r = torch.randn(h1.shape, generator=torch.Generator().manual_seed(7), dtype=h1.dtype)

    def passo(rd=None):
        for p_ in RT.parametri(resD) + RT.parametri(resS) + [t.master for t in st.f32.values()]:
            p_.grad = None
        stato = st.avanti(IDS)
        xL = stato["xL"].detach().requires_grad_(True)
        with torch.enable_grad():
            ((st.esce(xL) * r).sum() / h1.shape[0]).backward()
        if rd is None:
            st.indietro(stato, xL.grad)
        else:
            with rd.su("train", 0, IDS[0]):
                st.indietro(stato, xL.grad)
        return {n: t.master.grad.detach().clone() for n, t in st.f32.items()}, \
            {n: resS[n].B.grad.detach().clone() for n in resS}

    g0, gB = passo()
    assert all(bool((g != 0).any()) for g in gB.values()), "residuo shexp senza gradiente"
    assert all(g is not None and bool((g != 0).any()) for g in g0.values()), "F32 senza gradiente"
    # router: teacher logits = the student's own logits + noise (a stand-in for the cache; same tokens)
    zs = {}
    hs = [st.moduli[L].mlp.register_forward_hook(lambda m, a, o, L=L: zs.__setitem__(L, RDm.router_logit(
        m, a[0].reshape(-1, a[0].shape[-1])).detach())) for L in (3, 4)]
    with torch.no_grad():
        st.nascosto(IDS)
    for h in hs:
        h.remove()
    gz = torch.Generator().manual_seed(9)
    zT = torch.stack([zs[L] + 0.5 * torch.randn(zs[L].shape, generator=gz) for L in (3, 4)])

    class Stub:
        n_token, n_expert = IDS.shape[1], zT.shape[-1]
        strati = [3, 4]

        def disponibile(self, lato, i):
            return True

        def logit(self, lato, i, strati, device):
            return zT.to(device), IDS[0].numpy().astype(np.int64)
    W = 3.0
    rd = RDm.DistillaRouter(st, Stub(), [3, 4], W)
    gW, _ = passo(rd)
    kl_trainer = rd.ultimo
    rd.rimuovi()
    # independent full-graph reference of the term
    for t in st.f32.values():
        t.master.grad = None
    xs = {}
    hs = [st.moduli[L].mlp.register_forward_hook(lambda m, a, o, L=L: xs.__setitem__(L, (m, a[0]))) for L in (3, 4)]
    st.nascosto(IDS)
    for h in hs:
        h.remove()
    tot = 0.0
    for k, L in enumerate((3, 4)):
        m, x = xs[L]
        xf = x.reshape(-1, x.shape[-1])
        z = (torch.nn.functional.linear(xf.float(), m.router_w32) if (m.router_w32 is not None and xf.dtype not in
             (torch.float32, torch.float64)) else torch.nn.functional.linear(xf, m.router_w)).float()
        zt_ = zT[k].float()                             # R-CAMMINO with the module's loss functions (checked in test_bpp R0)
        tot = tot + RDm.kl_zona(z, zt_, 16, 8) + RDm.massa_a_rischio(z, zt_, (4, 9), RDm.tau_adattiva(z, zt_), 8)
    (W / 2 * tot).backward()
    kl_rif = float(tot.detach()) / 2
    assert abs(kl_trainer - kl_rif) <= 1e-5 * max(1.0, kl_rif), (kl_trainer, kl_rif)
    err, viste = {}, []
    for n, t in st.f32.items():
        rif = t.master.grad
        if rif is None or not bool((rif != 0).any()):
            continue
        d_ = (gW[n] - g0[n]).double(); r_ = rif.double()
        err[n] = float((d_ - r_).norm() / r_.norm())
        viste.append(n)
    chiave = [n for n in viste if "ffn_gate_inp.weight" in n and ".3." in n or "ffn_gate_inp.weight" in n and ".4." in n]
    assert len(chiave) == 2 and any("attn_norm" in n for n in viste), viste
    assert max(err.values()) < 1e-3, {n: e for n, e in err.items() if e >= 1e-3}
    return (f"(6 residui shexp su LineareTernaria, al bit, presenti, con gradiente; F32 {len(g0)} con gradiente; router KL "
            f"{kl_rif:.4f} = riferimento; gradiente del termine su {len(viste)} F32 (router 3-4, norme a monte): errore max "
            f"{max(err.values()):.1e})")


@prova("RR3 piega densi (veloce, CPU fp64) dei 3 shexp + attn_q dello strato 3: piani scritti al bit, catturata > 0")
def rr3():
    sys.path.insert(0, "@@ROOT:models@@/build/leve/raffina")
    import comune as CM
    src = CM.gp.GGUFPlaneSource(GGUF, layers=[3])
    g = torch.Generator().manual_seed(5)
    dd = {}
    for f in ("ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp", "attn_q"):
        d = CM.Denso(src, f"blk.3.{f}.weight")
        Q0 = d.ricostruisci()
        W_file = CM.deruota(Q0) if d.rotated else Q0
        U = torch.randn(d.out, 4, generator=g, dtype=torch.float64); Vt = torch.randn(4, d.in_, generator=g, dtype=torch.float64)
        dW = U @ Vt
        dd[f"blk.3.{f}"] = (0.05 * dW * W_file.norm() / dW.norm()).float()
    fd = TMP / "piega" / "residui_delta.pt"
    fd.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"delta_file": dd, "rango": 4, "famiglie": ["attn_q", "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp"],
                "strati": [3], "passo": 0, "nome": "prova_reale_bpp"}, fd)
    cmd = [PY, str(COD / "cardan_piega_densi_gpu_veloce.py"), "--delta", str(fd), "--gguf", GGUF,
           "--out", str(TMP / "piega" / "out"), "--device", "cpu", "--lotto", "2", "--fili-lettura", "1"]
    with open(TMP / "piega" / "piega.out", "w") as lg:
        rc = subprocess.run(["/usr/bin/time", "-f", "RSS %M KB", "nice", "-n", "19"] + cmd, stdout=lg,
                            stderr=subprocess.STDOUT, env={**os.environ, **SENZA_GPU}, cwd=str(COD)).returncode
    coda = (TMP / "piega" / "piega.out").read_text()
    assert rc == 0, coda[-1500:]
    rec = [json.loads(r) for r in open(TMP / "piega" / "out" / "registro.jsonl") if r.strip()]
    rec = [r.get("record", r) for r in rec]
    cat = {r["nome"]: r["catturata"] for r in rec if "catturata" in r}
    assert set(cat) == {f"{n}.weight" for n in dd}, cat
    assert min(cat.values()) > 0.0, cat         # mechanics: every tensor folded and captured something (26/10 run: attn_q
    # 0.52, gate/up_shexp 0.28/0.24, down_shexp 0.02 for a RANDOM rank-4 dW of 5 % on K = 1 planes: folding is weak there)
    npy = sorted(p.name for p in (TMP / "piega" / "out" / "tensori").glob("*.npy"))
    assert all(any(n.startswith(f"{k}") for n in npy) for k in dd), npy
    rss_f = [r for r in coda.splitlines() if r.startswith("RSS")]
    return "(" + ", ".join(f"{n.split('.')[2]} {100 * c:.1f} %" for n, c in sorted(cat.items())) + f"; {len(npy)} piani; " \
        f"{rss_f[-1] if rss_f else ''})"


@prova("RR4 esportatore tetto _bpp sull'ospite: LoRA shexp = (A, diag(gamma) B), forme ne, tipi nel riassunto")
def rr4():
    g = torch.Generator().manual_seed(11)
    r = 2
    st_ = {3: {f"D_{f}": torch.randn(2048, r, generator=g) for f in ("gate", "up")}}
    st_[3].update({f"C_{f}": torch.randn(256, 512, r, generator=g) * 1e-3 for f in ("gate", "up")})
    cpt = TMP / "lora" / "cardan_7.pt"; cpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"rango": r, "stato": st_, "meta": {"passo": 7}}, cpt)
    forme = {"blk.3.attn_q": (8192, 2048), "blk.4.attn_qkv": (8192, 2048), "blk.3.ffn_gate_shexp": (512, 2048),
             "blk.3.ffn_up_shexp": (512, 2048), "blk.3.ffn_down_shexp": (2048, 512), "blk.4.ffn_down_shexp": (2048, 512)}
    lat = {n: {"A": torch.randn(r, i, generator=g), "B": torch.randn(o, r, generator=g),
               "gamma": 1 + 0.1 * torch.randn(o, generator=g)} for n, (o, i) in forme.items()}
    rpt = TMP / "lora" / "residui_7.pt"
    torch.save({"meta": {"passo": 7}, "cfg": {"continuo": True, "rango": r}, "latenti": lat}, rpt)
    out = TMP / "lora" / "lora_7.gguf"
    with open(TMP / "lora" / "lora.out", "w") as lg:
        rc = subprocess.run(["/usr/bin/time", "-f", "RSS %M KB", "nice", "-n", "19", PY, ESPORTA, str(cpt), str(rpt), GGUF,
                             str(out)], stdout=lg, stderr=subprocess.STDOUT, env={**os.environ, **SENZA_GPU}).returncode
    txt = (TMP / "lora" / "lora.out").read_text()
    assert rc == 0, txt[-1500:]
    sys.path.insert(0, "@@ROOT:models@@/build/thor-c0t/gguf-py")
    import gguf
    rd = gguf.GGUFReader(str(out))
    T = {t.name: t for t in rd.tensors}
    for n, sd in lat.items():
        a_, b_ = T[f"{n}.weight.lora_a"], T[f"{n}.weight.lora_b"]
        o, i = forme[n]
        assert [int(x) for x in a_.shape] == [i, r] and [int(x) for x in b_.shape] == [r, o], (n, a_.shape, b_.shape)
        A_ = np.asarray(a_.data, dtype=np.float32).reshape(r, i)
        B_ = np.asarray(b_.data, dtype=np.float32).reshape(o, r)
        assert np.array_equal(A_, sd["A"].numpy()), f"{n}: lora_a != A (nessuna rotazione attesa)"
        assert np.array_equal(B_, (sd["B"].double() * sd["gamma"].double().view(-1, 1)).float().numpy()), n
    ria = json.load(open(str(out) + ".json"))["densi_per_famiglia"]
    assert ria["ffn_gate_shexp"]["tipi"] == ["TQ1_0R_1024"] and ria["ffn_down_shexp"]["tipi"] == ["TQ1_0"], ria
    assert ria["ffn_down_shexp"]["tensori"] == 2 and "di cui 4 shexp" in txt, (ria, txt[-300:])
    return f"({len(lat)} densi di cui 4 shexp al bit; tipi {ria['ffn_gate_shexp']['tipi']}/{ria['ffn_down_shexp']['tipi']}; " \
        f"{[l for l in txt.splitlines() if l.startswith('RSS')][-1]})"


if __name__ == "__main__":
    print(f"GGUF {GGUF}\ncartella {TMP}", flush=True)
    scelte = os.environ.get("PROVE", "rr2 rr1 rr3 rr4").split()      # e.g. PROVE="rr1" to rerun one
    for f in [g for g in (rr2, rr1, rr3, rr4) if g.__name__ in scelte]:
        f()
    ok = sum(1 for _, b, _ in ESITI if b)
    print(f"\n{ok}/{len(ESITI)} PASS (cartella {TMP}, RSS max {rss():.2f} GiB)")
    sys.exit(0 if ok == len(ESITI) else 1)
