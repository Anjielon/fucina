#!/usr/bin/env python3
"""Prove CPU degli esperti di recupero (e2e_rexp). Minuscole: <= 4 fili, pochi MB."""
import sys
import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(4)
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import e2e_rexp as R


def test_piano_ottimo_esatto():
    torch.manual_seed(0)
    w = torch.randn(3, 512)
    T, s = R.piano_ternario(w)
    assert T.dtype == torch.int8 and set(T.unique().tolist()) <= {-1, 0, 1}
    assert torch.equal(s, s.half().float())
    err = (w - R.ricostruisci(T, s)).pow(2).sum()
    # forza bruta per gruppo: ogni k, s = media dei k |w| piu' grandi
    tot = 0.0
    for gb in w.reshape(-1, 256):
        a = gb.abs().sort(descending=True).values
        best = min(((a[:k].sum() / k - a[:k]).pow(2).sum() + a[k:].pow(2).sum()).item() for k in range(1, 257))
        tot += best
    assert err.item() <= tot * 1.001 + 1e-3, (err.item(), tot)


def test_piani_riducono_errore():
    torch.manual_seed(1)
    w = torch.randn(4, 1024)
    e = [float((w - sum(R.ricostruisci(T, s) for T, s in R.piani(w, n))).pow(2).mean()) for n in (1, 2, 3)]
    assert e[0] > e[1] > e[2], e
    assert 0.15 < e[0] / float(w.pow(2).mean()) < 0.25        # ~0,19: pavimento Lloyd-Max del ternario gaussiano


def _mod(n_r=4, k_r=2, F_=256, H=512, pe=1, pr=3, cond=True):
    cfg = R.ConfigRexp(n_r=n_r, k_r=k_r, F=F_, piani_esperti=pe, piani_router=pr, condiviso=cond, strati=(0,))
    return R.EspertiRecupero(H, cfg, seme=3)


def test_partenza_esatta_zero():
    m = _mod()
    with torch.no_grad():
        m.R.normal_()
    x = torch.randn(64, 512, dtype=torch.bfloat16)
    y = m(x)
    assert y.dtype == x.dtype and torch.count_nonzero(y) == 0     # gamma = 0 (D casuale) => blocco + 0 = R2B3m al bit
    assert torch.count_nonzero(m.D) > 0


def _riferimento_denso(m, x):
    q = {k: v.detach() for k, v in m.pesi_q().items()}
    xf = x.float()
    prob = torch.softmax(xf @ q["R"].T, -1)
    idx = torch.topk(prob, m.cfg.k_r, -1).indices
    w = torch.zeros_like(prob).scatter(-1, idx, prob.gather(-1, idx))
    w = w / w.sum(-1, keepdim=True)
    tutti = torch.stack([(F.silu(xf @ q["G"][e].T) * (xf @ q["U"][e].T)) @ q["D"][e].T for e in range(m.cfg.n_r)], 1)
    y = (tutti * w.unsqueeze(-1)).sum(1)
    if m.cfg.condiviso:
        y = y + (F.silu(xf @ q["Gs"][0].T) * (xf @ q["Us"][0].T)) @ q["Ds"][0].T
    return y


def test_avanti_uguale_al_denso_e_gradienti():
    torch.manual_seed(2)
    m = _mod(pe=2)
    with torch.no_grad():
        m.R.normal_(); m.D.normal_(0, 0.05); m.gamma.uniform_(0.5, 1.5); m.gamma_s.fill_(0.7)
    x = torch.randn(96, 512)
    y = m(x)
    assert torch.allclose(y, _riferimento_denso(m, x), atol=1e-4, rtol=1e-4)
    y.pow(2).sum().backward()
    for n, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, n


def test_byte_tq1_rilettura():
    sys.path.insert(0, '@@ROOT:models@@/build/llamacpp-qat2/gguf-py')
    from gguf import quants, GGMLQuantizationType as QT
    torch.manual_seed(4)
    w = torch.randn(2, 3, 512) * 0.01
    for T, s in R.piani(w, 2):
        b = R._byte_tq1(T, s)
        assert b.shape == (2, 3, 2 * 54) and b.dtype == np.uint8
        d = quants.dequantize(b.reshape(-1, b.shape[-1]), QT.TQ1_0).reshape(w.shape)
        assert np.array_equal(d, R.ricostruisci(T, s).numpy()), np.abs(d - R.ricostruisci(T, s).numpy()).max()
    z = R._byte_tq1(*R.piano_ternario(torch.zeros(1, 256)))
    assert not quants.dequantize(z, QT.TQ1_0).any()


def test_tensori_gguf_nomi():
    m = _mod(pe=2, pr=3)
    with torch.no_grad():
        m.D.normal_()
    t = R.tensori_gguf({7: m})
    assert set(t) == {"blk.7.ffn_gate_inp_rexp.weight", "blk.7.ffn_gate_inp_rexp2.weight", "blk.7.ffn_gate_inp_rexp3.weight",
                      "blk.7.ffn_gate_rexps.weight", "blk.7.ffn_gate_rexps2.weight", "blk.7.ffn_up_rexps.weight",
                      "blk.7.ffn_up_rexps2.weight", "blk.7.ffn_down_rexps.weight", "blk.7.ffn_down_rexps2.weight",
        "blk.7.ffn_gate_rshexp.weight", "blk.7.ffn_gate_rshexp2.weight", "blk.7.ffn_up_rshexp.weight",
        "blk.7.ffn_up_rshexp2.weight", "blk.7.ffn_down_rshexp.weight", "blk.7.ffn_down_rshexp2.weight"}
    assert t["blk.7.ffn_down_rexps.weight"].shape == (4, 512, 54)    # (n_r, H, F/256*54)
    assert t["blk.7.ffn_down_rshexp.weight"].shape == (512, 54)      # condiviso: 2D


def test_init_gradiente_allinea():
    """Con D = 0 il primo gradiente su D e' sum w g act^T: l'init da C_e deve dare ||dL/dD|| >> init casuale."""
    torch.manual_seed(5)
    H, N = 512, 4096
    X = torch.randn(N, H)
    A = torch.randn(H, 8) @ torch.randn(8, H) / H                  # residuo di rango basso (H2)
    Gd = -(X @ A.T)                                                 # dL/dy = -(residuo) per perdita 1/2||r - y||^2
    norme = {}
    for modo in ("caso", "gradiente"):
        m = _mod(n_r=4, k_r=2, F_=256, H=H, cond=False)
        with torch.no_grad():                                       # misura con gamma = 1, D = 0: dL/dD = gamma * sum w g act^T
            m.D.zero_()
        if modo == "gradiente":
            d = R.inizializza_da_gradiente(m, X, Gd)
            assert sum(d["token_per_gruppo"]) == N
        with torch.no_grad():
            m.R.copy_(torch.randn_like(m.R) if modo == "caso" else m.R)
        with torch.no_grad():
            m.gamma.fill_(1.0)
        y = m(X)
        (y * Gd).sum().backward()
        q = m.pesi_q()
        with torch.no_grad():                                       # norma delle feature (a pari ampiezza d'ingresso)
            act = torch.cat([F.silu(X @ q["G"][e].T) * (X @ q["U"][e].T) for e in range(m.cfg.n_r)], 1)
        norme[modo] = float(m.D.grad.norm() / act.norm())
    assert norme["gradiente"] > 1.5 * norme["caso"], norme      # misurato 1,83x (G = U: piu basso)


def test_giocattolo_kl_scende():
    """Maestro = MLP a 2 strati; studente = maestro ternarizzato (pesi fermi) + esperti di recupero sul blocco nascosto,
    allenati con KL sui logit (obiettivo del MODELLO). La KL deve scendere e partire esattamente da quella dello studente."""
    torch.manual_seed(6)
    H, V, N = 512, 64, 2048
    W1, W2 = torch.randn(H, H) / H ** 0.5, torch.randn(V, H) / H ** 0.5
    W1s = R.ricostruisci(*R.piano_ternario(W1))
    X = torch.randn(N, H)
    lT = torch.log_softmax(torch.tanh(X @ W1.T) @ W2.T * 4, -1)
    m = _mod(n_r=4, k_r=2, F_=256, H=H)

    def kl():
        h = torch.tanh(X @ W1s.T) + m(X)
        lS = torch.log_softmax(h @ W2.T * 4, -1)
        return (lT.exp() * (lT - lS)).sum(-1).mean()
    with torch.no_grad():
        k0_rif = float((lT.exp() * (lT - torch.log_softmax(torch.tanh(X @ W1s.T) @ W2.T * 4, -1))).sum(-1).mean())
    with torch.no_grad():
        assert abs(float(kl()) - k0_rif) < 1e-7
    # init dal gradiente (stessa raccolta che fa il driver), poi Adam
    h = (torch.tanh(X @ W1s.T)).requires_grad_(True)
    ((lT.exp() * (lT - torch.log_softmax(h @ W2.T * 4, -1))).sum(-1).mean()).backward()
    R.inizializza_da_gradiente(m, X, h.grad)
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    for _ in range(60):
        opt.zero_grad(); v = kl(); v.backward(); opt.step()
    k1 = float(kl())
    assert k1 < 0.8 * k0_rif, (k0_rif, k1)
    print(f"giocattolo: KL {k0_rif:.4f} -> {k1:.4f}")


def test_costruttore_gguf_minuscolo(tmp=None):
    """Base minuscola (un tensore F32) + esperti di recupero: forme ggml, chiave k_r, byte riletti uguali."""
    import tempfile, os
    sys.path.insert(0, '@@ROOT:models@@/build/llamacpp-qat2/gguf-py')
    from gguf import GGUFWriter, GGUFReader
    import costruisci_recupero as CR
    d = tempfile.mkdtemp(dir='@@ROOT:depot@@/e2e/recupero_prove') if os.path.isdir('@@ROOT:depot@@/e2e/recupero_prove') \
        else tempfile.mkdtemp()
    base = os.path.join(d, 'base.gguf')
    w = GGUFWriter(base, arch='qwen35moe'); w.add_tensor('blk.0.attn_norm.weight', np.ones(512, np.float32))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    m = _mod(n_r=4, F_=256, H=512, pe=1, pr=3)
    with torch.no_grad():
        m.D.normal_(0, 0.01); m.R.normal_(); m.gamma.fill_(0.5)
    t = R.tensori_gguf({0: m})
    ver = CR.costruisci(base, t, 2, os.path.join(d, 'x.gguf'), sha_base=None)
    assert ver['ok'], ver
    r = GGUFReader(os.path.join(d, 'x.gguf'))
    tt = {x.name: x for x in r.tensors}
    assert list(tt['blk.0.ffn_down_rexps.weight'].shape) == [256, 512, 4]       # ggml: ne0 = ingresso F, ne1 = H, ne2 = n_r
    assert list(tt['blk.0.ffn_gate_rexps.weight'].shape) == [512, 256, 4]
    assert list(tt['blk.0.ffn_gate_inp_rexp3.weight'].shape) == [512, 4]
    assert list(tt['blk.0.ffn_down_rshexp.weight'].shape) == [256, 512]
    assert int(r.fields[CR.CHIAVE_K].contents()) == 2 and int(r.fields[CR.CHIAVE_N].contents()) == 4


def test_gamma_fuso_nelle_scale_esatto():
    """I byte esportati di D (gamma fuso nelle scale fp16) riletti == il D che l'avanti usa, al bit."""
    sys.path.insert(0, '@@ROOT:models@@/build/llamacpp-qat2/gguf-py')
    from gguf import quants, GGMLQuantizationType as QT
    m = _mod(pe=2)
    with torch.no_grad():
        m.gamma.copy_(torch.tensor([0.3, -1.7, 0.0, 2.5])); m.gamma_s.fill_(-0.4)
    q = m.pesi_q()
    t = R.tensori_gguf({0: m})
    for k, nome in (("D", "ffn_down_rexps"), ("Ds", "ffn_down_rshexp")):
        d = sum(quants.dequantize(t[f"blk.0.{nome}{s}.weight"].reshape(-1, 108), QT.TQ1_0) for s in ("", "2"))
        assert np.array_equal(d.reshape(-1), q[k].detach().numpy().reshape(-1)), k


def test_perdite_ausiliarie_router():
    """Con peso_aux > 0 il router riceve il gradiente di bilanciamento + z-loss anche con uscita nulla (gamma = 0)."""
    torch.manual_seed(7)
    m = _mod()
    with torch.no_grad():
        m.R.normal_(0, 0.05)
    x = torch.randn(256, 512)
    m(x).sum().backward()
    assert m.R.grad is None or m.R.grad.abs().sum() == 0              # gamma = 0, niente aux: router fermo
    m.zero_grad(); m.peso_aux = 1.0
    m(x).sum().backward()
    assert m.R.grad.abs().sum() > 0 and {"bil", "z", "carico_max"} <= set(m.stat)
    assert 0.99 <= m.stat["bil"] and m.stat["z"] > 0                   # bil >= 1 (uguale a 1 col carico uniforme)


def test_taylor_segno():
    """taylor = -gamma dL/dgamma: positivo se togliere gli esperti peggiora la perdita."""
    m = _mod(cond=False)
    with torch.no_grad():
        m.gamma.fill_(1.0)
    x = torch.randn(64, 512)
    y0 = m(x).detach()
    ((m(x) - 0.0) * y0).sum().backward()          # perdita che PREMIA l'uscita: dL/dgamma > 0 ... togliere migliora
    assert m.taylor() < 0


def test_kl_top_esatta_col_secchio():
    """KLTop == formula diretta (autograd) sulla distribuzione a k token + secchio; con k = V coincide con la KL piena."""
    torch.manual_seed(8)
    P, H, V = 37, 16, 50
    W = torch.randn(V, H); hT = torch.randn(P, H); hS = torch.randn(P, H, requires_grad=True)
    lT = hT @ W.T
    for k in (8, V):
        idx, lpT = R.topk_maestro(lT, k)
        v = R.KLTop.apply(hS, idx, lpT, lambda h: h @ W.T, lambda p: p @ W, 7)
        g, = torch.autograd.grad(v, hS)
        h2 = hS.detach().clone().requires_grad_(True)
        lS = torch.log_softmax(h2 @ W.T, -1)
        pt = lpT.exp(); ls = lS.gather(-1, idx.long())
        rT = (1 - pt.sum(-1)).clamp_min(1e-12); rS = (1 - ls.exp().sum(-1)).clamp_min(1e-12)
        rif = ((pt * (lpT - ls)).sum(-1) + rT * (rT.log() - rS.log())).sum()
        g2, = torch.autograd.grad(rif, h2)
        assert abs(float(v) - float(rif)) < 1e-4 * max(1.0, abs(float(rif))), (k, float(v), float(rif))
        assert torch.allclose(g, g2, atol=1e-4, rtol=1e-3), k
        if k == V:
            lTn = torch.log_softmax(lT, -1)
            piena = (lTn.exp() * (lTn - torch.log_softmax(hS.detach() @ W.T, -1))).sum()
            assert abs(float(v) - float(piena)) < 1e-3


def test_driver_su_studente_finto():
    """e2e_recupero.main di capo a coda su uno studente FINTO (2 strati, vocabolario 64) al posto di e2e_studente/e2e_allena:
    cancello d'identita', init, sonda lr, allenamento, ablazione, esportazione. Prova la colla, non i numeri."""
    import types, tempfile, os, json
    import e2e_recupero as DRV
    torch.manual_seed(9)
    H, V, T, NL = 512, 64, 32, 2
    W = [torch.randn(H, H) / H ** 0.5 for _ in range(NL)]
    Wout_T = torch.randn(V, H) / H ** 0.5
    E = torch.randn(V, H)

    class Mlp(torch.nn.Module):
        def __init__(s_, Wl):
            super().__init__(); s_.Wl = Wl; s_.router_w = torch.zeros(4, H)
        def forward(s_, h):
            y = torch.tanh(h @ R.ricostruisci(*R.piano_ternario(s_.Wl)).T)
            r = s_._modules.get("rexp")
            return y + r(h) if r is not None else y

    class Strato(torch.nn.Module):
        def __init__(s_, Wl):
            super().__init__(); s_.mlp = Mlp(Wl)
        def forward(s_, x):
            return x + s_.mlp(x)

    class Stud:
        def __init__(s_, **kw):
            s_.device = torch.device("cpu"); s_.strati = list(range(NL)); s_.moduli = {L: Strato(W[L]) for L in s_.strati}
            s_.strati_allenati = None
        def parametri(s_):
            return []
        def avanti(s_, ids, inizi=None):
            with torch.no_grad():
                x = E[ids].reshape(-1, H); ing = []
                for L in s_.strati:
                    ing.append(x); x = s_.moduli[L](x)
            return {"xL": x, "ingressi": ing}
        def indietro(s_, stato, g):
            i0 = min(s_.strati_allenati)
            for L in reversed(s_.strati):
                if L < i0: break
                xin = stato["ingressi"][L].detach().requires_grad_(L > i0)
                with torch.enable_grad():
                    y = s_.moduli[L](xin); y.backward(g)
                g = xin.grad
        def esce(s_, x):
            return x
        def nascosto(s_, ids, inizi=None):
            return s_.avanti(ids)["xL"]

    def teacher_h(ids):
        x = E[ids].reshape(-1, H)
        for L in range(NL):
            x = x + torch.tanh(x @ W[L].T)
        return x

    ids_tr = {i: torch.randint(0, V, (1, T)) for i in range(40)}
    ids_ho = {i: torch.randint(0, V, (1, T)) for i in range(4)}

    class Lot:
        def __init__(s_, ids):
            s_.ids, s_.inizi, s_.hT = ids, None, teacher_h(ids)

    class Car:
        def __init__(s_, src, dev): pass
        def prendi(s_, lato, righe):
            return Lot((ids_tr if lato == "train" else ids_ho)[righe[0]])

    class Src:
        def __init__(s_, d): pass
        def righe_previste(s_, lato):
            return list((ids_tr if lato == "train" else ids_ho).keys())

    def kl_pezzo(hS, WS, hT, WT, grad):
        lT = torch.log_softmax(hT @ WT.T, -1); lS = torch.log_softmax(hS @ WS.T, -1)
        return {"kl": (lT.exp() * (lT - lS)).sum(-1)}

    fA = types.SimpleNamespace(scegli_gdn=lambda log: "finto", testa_studente=lambda st, t: Wout_T,
                               testa_maestro=lambda dev, dt: (Wout_T, "x"), SorgenteCache=Src, Caricatore=Car,
                               kl_pezzo=kl_pezzo, _logit_maestro=lambda h, W_, dt: h.float() @ W_.T,
                               _logit_studente=lambda h, W_: h @ W_.T, _per_testa=lambda p, W_: p @ W_)
    fS = types.SimpleNamespace(Studente=Stud)
    vecchi = {k: sys.modules.get(k) for k in ("e2e_studente", "e2e_allena")}
    sys.modules["e2e_studente"], sys.modules["e2e_allena"] = fS, fA
    try:
        d = tempfile.mkdtemp(dir='@@ROOT:depot@@/e2e/recupero_prove') if os.path.isdir('@@ROOT:depot@@/e2e/recupero_prove') \
            else tempfile.mkdtemp()
        rc = DRV.main(["--uscita", d, "--device", "cpu", "--strati", "0-1", "--n-r", "4", "--k-r", "2", "--F", "256",
                       "--passi", "6", "--righe-passo", "2", "--righe-init", "2", "--heldout-righe", "4",
                       "--heldout-ogni", "3", "--sonda-lr", "3e-4,1e-3", "--passi-sonda", "2", "--heldout-sonda", "2",
                       "--cache-topk", d + "/cache", "--topk", "16", "--gemm", "fp32", "--pezzo-kl", "16"])
        assert rc == 0
        e = json.load(open(d + "/esportazione.json")); st = json.load(open(d + "/storia.json"))
        assert e["n_tensori"] == 2 * (3 + 3 + 3) and len(st["sonda"]) == 2 and st["allenamento"]
        assert "AL BIT" in open(d + "/registro.log").read()
        print("driver finto: KL", e["kl0"], "->", e["kl_held"], "passo", e["passo"], "lr", st["lr_latenti"],
              "ablazione", st["ablazione"])
    finally:
        for k, v in vecchi.items():
            if v is None: sys.modules.pop(k, None)
            else: sys.modules[k] = v


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("ok", n)
