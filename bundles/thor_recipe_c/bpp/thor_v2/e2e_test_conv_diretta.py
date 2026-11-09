"""Prove di e2e_conv_diretta (2/10 TT6-veloce).
CPU (TRITON_INTERPRET=1, nice 19, <= 4 fili):
  K1 kernel Triton == aritmetica torch fp64 (avanti e dX) AL BIT, su forme con bordi (T, C non multipli dei blocchi)
  K2 autograd della Function == autograd della via torch fp64 di riferimento, AL BIT
  K3 ripiego: peso che chiede gradiente / bias / dtype diverso -> funzione originale di transformers (chiamata contata)
  K4 installa / disinstalla idempotenti; spento (niente installa) = transformers di sempre
GPU (--gpu, piccola):
  KG1 forma vera 2 x 4096 x 8192: diretta == F.conv1d di MIOpen AL BIT, avanti e dX (con silu come transformers)
  KG2 strato 0 vero di R2B3m (GDN) avanti + ricalcolo + indietro con e senza E2E_CONV_DIRETTA: h e dL/dtheta AL BIT
Uso: TRITON_INTERPRET=1 python e2e_test_conv_diretta.py [--gpu]"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import e2e_conv_diretta as CD  # noqa: E402


def _rif_mio(x_bct, w, act=None):
    """Transformers di sempre (la funzione originale), su (B, C, T)."""
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as M
    f = CD._ORIGINALE or M.causal_conv1d_fn
    return f(x_bct, w, None, activation=act)


def k1(dev="cpu"):
    torch.manual_seed(1)
    for (B, T, C) in ((1, 37, 130), (2, 70, 256), (1, 5, 3)):
        x = torch.randn(B, T, C, device=dev)
        w = torch.randn(C, 4, device=dev)
        for verso in (0, 1):
            yt = CD._conv_torch(x, w, verso)
            for xin, bct in ((x, False), (x.transpose(1, 2).contiguous().transpose(1, 2), True)):
                yk = torch.empty(B, C, T).transpose(1, 2) if bct else torch.empty_like(x)
                CD._k_conv[(B, (T + 31) // 32, (C + 127) // 128)](xin, *xin.stride(), w, yk, *yk.stride(), T, C, 4,
                                                                  verso, 32, 128)
                assert torch.equal(yk, yt), (B, T, C, verso, bct, (yk - yt).abs().max())
    print("K1 ok: kernel Triton == torch fp64 al bit (avanti e dX, bordi)")


def k2():
    torch.manual_seed(2)
    x = torch.randn(2, 50, 64, dtype=torch.float32)
    w = torch.randn(64, 4)
    g = torch.randn(2, 50, 64)
    xa = x.clone().requires_grad_(True)
    CD.ConvCausaleBTC.apply(xa, w).backward(g)
    xb = x.clone().requires_grad_(True)       # riferimento: stessa formula fp64 differenziata da autograd
    xp = F.pad(xb.to(torch.float64), (0, 0, 3, 0))
    y = sum(xp[:, k:k + 50] * w[:, k].double() for k in range(4)).to(torch.float32)
    y.backward(g)
    assert torch.equal(xa.grad, xb.grad), (xa.grad - xb.grad).abs().max()
    assert torch.equal(CD.ConvCausaleBTC.apply(x, w), y.detach())
    print("K2 ok: Function == autograd della formula fp64 al bit")


def k3():
    CD.installa()
    os.environ["E2E_CONV_DIRETTA_CPU"] = "1"
    try:
        x = torch.randn(1, 64, 20).transpose(1, 2).contiguous().transpose(1, 2)   # (B, C, T) vista trasposta
        w = torch.randn(64, 4)
        n0 = dict(CD.CONTI)
        CD.causal_conv1d_fn_diretta(x, w.clone().requires_grad_(True), None, activation="silu")
        CD.causal_conv1d_fn_diretta(x, w, torch.zeros(64), activation="silu")
        CD.causal_conv1d_fn_diretta(x.double(), w.double(), None, activation="silu")
        assert CD.CONTI["ripiego"] - n0["ripiego"] == 3 and CD.CONTI["diretta"] == n0["diretta"]
        y = CD.causal_conv1d_fn_diretta(x, w, None, activation="silu")
        assert CD.CONTI["diretta"] == n0["diretta"] + 1
        rif = _rif_mio(x, w, "silu")
        assert y.shape == x.shape and y.is_contiguous() and y.stride() == rif.stride(), (y.stride(), rif.stride())
        os.environ.pop("E2E_CONV_DIRETTA_CPU")
        CD.causal_conv1d_fn_diretta(x, w, None, activation="silu")                 # CPU senza la variabile: ripiego
        assert CD.CONTI["ripiego"] - n0["ripiego"] == 4
    finally:
        os.environ.pop("E2E_CONV_DIRETTA_CPU", None)
        CD.disinstalla()
    print("K3 ok: ripieghi (peso con gradiente, bias, fp64, CPU) contati; uscita (B, C, T) contigua coi passi di transformers")


def k4():
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as M
    orig = M.causal_conv1d_fn
    CD.installa(); CD.installa()
    assert M.causal_conv1d_fn is CD.causal_conv1d_fn_diretta and CD._ORIGINALE is orig
    CD.disinstalla(); CD.disinstalla()
    assert M.causal_conv1d_fn is orig
    print("K4 ok: installa/disinstalla idempotenti")


def kg1():
    torch.manual_seed(0)
    B, T, C = 2, 4096, 8192
    h = torch.randn(B, T, C, device="cuda") * 0.7
    w = torch.randn(C, 4, device="cuda") * 0.3
    g = torch.randn(B, T, C, device="cuda")
    CD.installa()
    try:
        out = {}
        for nome, f in (("rif", CD._ORIGINALE), ("diretta", CD.causal_conv1d_fn_diretta)):
            x = h.clone().requires_grad_(True)
            y = f(x.transpose(1, 2), w, None, activation="silu").transpose(1, 2)
            y.backward(g)
            torch.cuda.synchronize()
            out[nome] = (y.detach(), x.grad)
        assert torch.equal(out["rif"][0], out["diretta"][0]), "avanti diverso"
        assert torch.equal(out["rif"][1], out["diretta"][1]), "dX diverso"
        xa = h.clone().requires_grad_(True)
        ya = CD._ORIGINALE(xa.transpose(1, 2), w, None, activation="silu")
        yb = CD.causal_conv1d_fn_diretta(xa.transpose(1, 2), w, None, activation="silu")
        assert ya.stride() == yb.stride(), (ya.stride(), yb.stride())          # stessi passi: stesso torch.compile dopo
        import time
        for nome, f in (("rif", CD._ORIGINALE), ("diretta", CD.causal_conv1d_fn_diretta)):
            x = h.clone().requires_grad_(True)
            torch.cuda.synchronize(); t0 = time.time()
            for _ in range(3):
                y = f(x.transpose(1, 2), w, None, activation="silu").transpose(1, 2)
                y.backward(g)
            torch.cuda.synchronize()
            print(f"  {nome}: avanti+dX {(time.time() - t0) / 3 * 1e3:.1f} ms")
    finally:
        CD.disinstalla()
    print("KG1 ok: diretta == MIOpen al bit (avanti con silu e dX), forma vera")


def kg2():
    import e2e_allena as A
    import e2e_studente as S
    os.environ.setdefault("E2E_GDN", "veloce")
    os.environ.setdefault("E2E_GDN_MODULO", "gdn_veloce_corsa_v5")
    A.scegli_gdn()
    st = S.Studente(gguf="@@ROOT:models@@/gguf/tony-tern/R2B3m/Thor-T-R2B3m.gguf", strati=[0], ricetta="B3_ind2",
                    motore_nome="triton2", device="cuda", densi_ternari="tutte")
    ids = S._ids_reali(2, 4096).cuda()
    ris = {}
    for acceso in (False, True):
        (CD.installa if acceso else CD.disinstalla)()
        for p in st.parametri():
            p.grad = None
        h = {}
        val = st.gradiente(ids, lambda x: (h.setdefault("h", x.detach().clone()), x.float().pow(2).mean() * 1e3)[1])
        torch.cuda.synchronize()
        ris[acceso] = (h["h"], [p.grad.clone() for p in st.parametri()], val)
    CD.disinstalla()
    assert torch.equal(ris[False][0], ris[True][0]), "h diverso"
    for a, b in zip(ris[False][1], ris[True][1]):
        assert torch.equal(a, b), "dL/dtheta diverso"
    print(f"KG2 ok: strato 0 vero, h e {len(ris[True][1])} gradienti al bit (conti {CD.CONTI})")


if __name__ == "__main__":
    torch.set_num_threads(4)
    if "--gpu" in sys.argv:
        kg1()
        kg2()
    else:
        k1(); k2(); k3(); k4()
