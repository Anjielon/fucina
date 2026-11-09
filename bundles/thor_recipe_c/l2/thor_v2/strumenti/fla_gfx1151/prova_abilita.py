#!/usr/bin/env python3
"""Collaudo di gdn_fla_gfx1151.abilita_sicuro come lo userebbe la corsa (27/9/2026).
transformers importato con fla NON importabile -> abilita_sicuro(M) -> un Qwen3_5MoeGatedDeltaNet vero (config Thor:
hidden 2048, 32 teste valore, 16 teste chiave, dk = dv = 128) a B=2, T=4096, fp32, avanti e indietro:
  1. il riferimento fp64 (G.ripiego_fp64) controllato contro una ricorrenza token per token scritta da capo;
  2. tempo dello STRATO intero (mediana di N) con fla e col ripiego, stesse condizioni -> ms/token;
  3. errore dello strato (uscita, gradiente d'ingresso, gradiente di ogni parametro) di fla e del ripiego fp32
     contro lo strato in fp64 (pesi e ingressi identici, GDN = ripiego_fp64).
Lanciare con lancia_prova.sh (PROVA=prova_abilita.py)."""
import statistics
import sys
import time
from pathlib import Path

import torch

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as M  # noqa: E402  (fla non importabile qui)
import gdn_fla_gfx1151 as G  # noqa: E402
import riferimento64 as R  # noqa: E402


def metodo_fp64(metodo, sostituzioni):
    """Il metodo di transformers con i cast a fp32 portati a fp64 (sorgente originale, sostituzioni verificate):
    nello strato GDN `g` passa da `.float()` e RMSNormGated da `.to(torch.float32)` — in uno strato fp64 sarebbero
    arrotondamenti fp32 dentro il riferimento."""
    import inspect
    import textwrap
    f = metodo                                       # force_accelerate_hooks non usa functools.wraps: l'originale
    for cella in (f.__closure__ or ()):              # sta nella chiusura del wrapper
        v = cella.cell_contents
        if inspect.isfunction(v) and v.__name__ == metodo.__name__.replace("wrapped", "forward"):
            f = v
    if f.__name__ != "forward":
        f = next(c.cell_contents for c in metodo.__closure__ if inspect.isfunction(c.cell_contents)
                 and c.cell_contents.__name__ == "forward")
    src = textwrap.dedent(inspect.getsource(f))
    src = src[src.index("def "):]
    for vecchio, nuovo in sostituzioni:
        assert vecchio in src, vecchio
        src = src.replace(vecchio, nuovo)
    ns = dict(vars(M))
    exec(compile(src, f"<{metodo.__qualname__} fp64>", "exec"), ns)
    return ns["forward"]


def rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm())


def main() -> int:
    rip64 = G.ripiego_fp64(M)                        # PRIMA di abilita_sicuro: dopo, il globale e' l'involucro
    t0 = time.perf_counter()
    r = G.abilita_sicuro(M)
    print(f"abilita_sicuro in {time.perf_counter() - t0:.1f}s: {r}", flush=True)
    if not r["attivo"]:
        return 2
    gdn = M.torch_chunk_gated_delta_rule

    # 1. il riferimento stesso
    gen = torch.Generator(device="cpu").manual_seed(5)
    qq, kk, vv = (torch.randn(1, 1024, 8, 128, generator=gen).cuda() for _ in range(3))
    g_ = (-torch.nn.functional.softplus(torch.randn(1, 1024, 8, generator=gen))).cuda()
    b_ = torch.sigmoid(torch.randn(1, 1024, 8, generator=gen)).cuda()
    with torch.no_grad():
        o_rec = R.ricorrenza_fp64(qq, kk, vv, g_, b_)
        o_r64, _ = rip64(qq.double(), kk.double(), vv.double(), g=g_.double(), beta=b_.double(),
                         use_qk_l2norm_in_kernel=True)
        o_r32, _ = gdn.ripiego(qq, kk, vv, g=g_, beta=b_, use_qk_l2norm_in_kernel=True)
        o_fal, _ = gdn.ripiego(qq.double(), kk.double(), vv.double(), g=g_.double(), beta=b_.double(),
                               use_qk_l2norm_in_kernel=True)
    print(f"riferimento: ripiego_fp64~ricorrenza_fp64 {rel(o_r64, o_rec):.2e} · ripiego(fp64 in ingresso, falso)"
          f"~ricorrenza {rel(o_fal, o_rec):.2e} · ripiego fp32~ricorrenza {rel(o_r32, o_rec):.2e}", flush=True)
    del qq, kk, vv, g_, b_, o_rec, o_r64, o_r32, o_fal

    # 2-3. strato vero
    cfg = M.Qwen3_5MoeTextConfig(hidden_size=2048, linear_num_value_heads=32, linear_num_key_heads=16,
                                 linear_key_head_dim=128, linear_value_head_dim=128, linear_conv_kernel_dim=4)
    torch.manual_seed(0)
    strato = M.Qwen3_5MoeGatedDeltaNet(cfg, layer_idx=0).cuda().float()
    with torch.no_grad():                             # gate realistici: A_log e dt_bias come all'init di HF
        strato.dt_bias.fill_(1.0)
    B, T = 2, 4096
    x = torch.randn(B, T, cfg.hidden_size, device="cuda")
    dy = torch.randn(B, T, cfg.hidden_size, device="cuda")

    def passo(impl, s, xx0, dy0):
        M.torch_chunk_gated_delta_rule = impl
        s.zero_grad(set_to_none=True)
        xx = xx0.clone().requires_grad_(True)
        torch.cuda.synchronize()
        t = time.perf_counter()
        y = s(xx)
        torch.cuda.synchronize()
        ta = time.perf_counter() - t
        y.backward(dy0)
        torch.cuda.synchronize()
        return (y.detach(), xx.grad.detach(), {n: p.grad.detach().clone() for n, p in s.named_parameters()},
                ta, time.perf_counter() - t)

    uscite, tempi = {}, {}
    for nome, impl in (("fla", gdn), ("ripiego", gdn.ripiego)):
        uscite[nome] = passo(impl, strato, x, dy)          # primo passo (compilazione, se c'e')
        mis = [passo(impl, strato, x, dy)[3:] for _ in range(5)]
        ta = statistics.median(m[0] for m in mis)
        tt = statistics.median(m[1] for m in mis)
        tempi[nome] = (ta, tt)
        print(f"{nome:8s} strato GDN: avanti {ta * 1e3:.1f} ms · avanti+indietro {tt * 1e3:.1f} ms · "
              f"{tt * 1e3 / (B * T):.4f} ms/token (mediana di 5, EB2 accanto)", flush=True)
    print(f"accelerazione dello strato (avanti+indietro): x{tempi['ripiego'][1] / tempi['fla'][1]:.2f}", flush=True)
    s64 = M.Qwen3_5MoeGatedDeltaNet(cfg, layer_idx=0).cuda().double()
    s64.load_state_dict({k_: v_.double() for k_, v_ in strato.state_dict().items()})
    import types
    s64.forward = types.MethodType(metodo_fp64(M.Qwen3_5MoeGatedDeltaNet.forward, [(".float()", ".double()")]), s64)
    s64.norm.forward = types.MethodType(metodo_fp64(M.Qwen3_5MoeRMSNormGated.forward,
                                                    [("torch.float32", "torch.float64")]), s64.norm)
    s64.forward.__func__.__globals__["torch_chunk_gated_delta_rule"] = rip64   # il forward copiato legge QUESTO
    u64 = passo(rip64, s64, x.double(), dy.double())
    M.torch_chunk_gated_delta_rule = gdn
    peggio = 0.0
    righe = [("uscita", 0, None), ("grad ingresso", 1, None)] + [(f"grad {n}", 2, n) for n in uscite["fla"][2]]
    for etichetta, i, n in righe:
        a = uscite["fla"][i] if n is None else uscite["fla"][2][n]
        b = uscite["ripiego"][i] if n is None else uscite["ripiego"][2][n]
        c = u64[i] if n is None else u64[2][n]
        ef, et = rel(a, c), rel(b, c)
        peggio = max(peggio, ef / et)
        print(f"  {etichetta:26s} fla~fp64 {ef:.2e} · ripiego~fp64 {et:.2e} · rapporto {ef / et:.2f}", flush=True)
    print(f"ESITO strato: rapporto peggiore {peggio:.2f} -> {'NEL RUMORE fp32' if peggio <= 2 else 'FUORI'}", flush=True)
    print(f"stato involucro: {gdn.stato}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
