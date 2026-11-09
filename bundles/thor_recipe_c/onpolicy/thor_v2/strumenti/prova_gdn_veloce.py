#!/usr/bin/env python3
"""Prova di strumenti/gdn_veloce.py contro il ripiego di transformers (torch_chunk_gated_delta_rule), 27/9/2026.

Stessa matematica? Avanti e gradienti dq, dk, dv, dg, dbeta, ciascuno contro il RIFERIMENTO fp64 vero: una copia del
ripiego con il dtype di calcolo parametrico (il ripiego di transformers fa .to(torch.float32) dentro, quindi
chiamarlo con ingressi fp64 NON da' fp64: le prove precedenti misuravano cosi' un "rumore fp32" che era fp32 contro
fp32). Criterio: errore della versione nuova verso fp64 <= 2 x quello del ripiego fp32, in avanti e in TUTTI i gradienti.
Poi i tempi (mediana di R ripetizioni, avanti e indietro separati) e, con --profilo, la tabella torch.profiler.

GPU: lanciare come unita' systemd --user, un processo di prova alla volta accanto alla corsa (vedi strumenti/).
CPU: --dev cpu con dimensioni piccole (--B --T --H --D) per l'equivalenza algebrica."""
import argparse
import os
import statistics
import sys
import time

import torch
import torch.nn.functional as F

sys.modules["fla"] = None                 # il ripiego PyTorch: transformers lo sceglie se fla non si importa
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as M  # noqa: E402
del sys.modules["fla"]
ripiego = M.torch_chunk_gated_delta_rule
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gdn_veloce import chunk_gated_delta_rule_veloce, l2norm  # noqa: E402


def ripiego_dtype(query, key, value, g, beta, chunk_size=64, initial_state=None, output_final_state=False,
                  use_qk_l2norm_in_kernel=False, dt=torch.float64, **kwargs):
    """Copia TESTUALE di transformers torch_chunk_gated_delta_rule (5.x, modeling_qwen3_5_moe.py:249) con il solo
    `torch.float32` sostituito da `dt`: il riferimento fp64 della stessa algebra."""
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [x.transpose(1, 2).contiguous().to(dt) for x in (query, key, value, beta, g)]
    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().to(dt)).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
        if initial_state is None else initial_state.to(value))
    core_attn_out = torch.zeros_like(value)
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new)
    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def candidato(versione, dt=torch.float32):
    def f(*a, **kw):
        return chunk_gated_delta_rule_veloce(*a, _dtype_calcolo=dt, _versione=versione, **kw)
    return f


# scala del log-decadimento per passo, peso di una direzione COMUNE in q e k (chiavi correlate: coseno ~0.9, la
# soluzione triangolare (I + L)^-1 lavora vicino al suo caso peggiore), spostamento di beta verso 1
SCENARI = {"normale": (1.0, 0.0, 0.0), "forte": (4.0, 0.0, 0.0), "debole": (0.05, 0.0, 0.0),
           "correlate": (0.05, 3.0, 2.0)}


def ingressi(seed, B, T, H, D, dev, scenario, stato):
    scala_g, comune, spinta_beta = scenario
    gen = torch.Generator(device="cpu").manual_seed(seed)
    q, k, v = (torch.randn(B, T, H, D, generator=gen) for _ in range(3))
    if comune:
        base = torch.randn(B, 1, H, D, generator=gen)
        q, k = q + comune * base, k + comune * base
    gg = -F.softplus(torch.randn(B, T, H, generator=gen)) * scala_g           # log-decadimento <= 0
    beta = torch.sigmoid(torch.randn(B, T, H, generator=gen) + spinta_beta)   # (0, 1)
    do = torch.randn(B, T, H, D, generator=gen)
    s0 = torch.randn(B, H, D, D, generator=gen) * 0.1 if stato else None
    dS = torch.randn(B, H, D, D, generator=gen) if stato else None
    return [t.to(dev, torch.float32) for t in (q, k, v, gg, beta, do)], (
        None if s0 is None else s0.to(dev, torch.float32), None if dS is None else dS.to(dev, torch.float32))


def sinc(dev):
    if dev != "cpu":
        torch.cuda.synchronize()


def corri(f, dati, stato, dt=torch.float32, ripeti=1):
    """Avanti + indietro; ritorna uscita, 5 gradienti (+ gradiente dello stato iniziale se c'e'), tempi."""
    q, k, v, gg, beta, do = [t.to(dt) for t in dati]
    s0, dS = stato
    ta, ti = [], []
    for _ in range(ripeti):
        xs = [t.clone().requires_grad_(True) for t in (q, k, v, gg, beta)]
        s = None if s0 is None else s0.to(dt).clone().requires_grad_(True)
        sinc(q.device.type); t0 = time.perf_counter()
        o, SN = f(xs[0], xs[1], xs[2], g=xs[3], beta=xs[4], initial_state=s, output_final_state=s is not None,
                  use_qk_l2norm_in_kernel=True)
        sinc(q.device.type); t1 = time.perf_counter()
        perdita = (o * do).sum() if SN is None else (o * do).sum() + (SN * dS.to(dt)).sum()
        perdita.backward()
        sinc(q.device.type); t2 = time.perf_counter()
        ta.append(t1 - t0); ti.append(t2 - t1)
    grads = [x.grad.detach().double() for x in xs] + ([] if s is None else [s.grad.detach().double()])
    uscite = [o.detach().double()] + ([] if SN is None else [SN.detach().double()])
    return uscite, grads, ta, ti


def rel(a, b):
    a, b = a.detach(), b.detach()
    return float((a - b).norm() / b.norm().clamp_min(1e-300))


NOMI = ("dq", "dk", "dv", "dg", "dbeta", "dS0")


def confronta(args, scenario, seed):
    dati, stato = ingressi(seed, args.B, args.T, args.H, args.D, args.dev, SCENARI[scenario], args.stato)
    u64, g64, _, _ = corri(ripiego_dtype, dati, stato, torch.float64)            # riferimento fp64 della stessa algebra
    ur, gr, _, _ = corri(ripiego, dati, stato)
    ok = True
    for ver in args.versioni.split(","):
        ok &= confronta_uno(args, scenario, seed, int(ver), dati, stato, u64, g64, ur, gr)
    return ok


def confronta_uno(args, scenario, seed, ver, dati, stato, u64, g64, ur, gr):
    un64, gn64, _, _ = corri(candidato(ver, torch.float64), dati, stato, torch.float64)
    un, gn, _, _ = corri(candidato(ver), dati, stato)
    nomi_u = ("avanti", "S_fin")
    ok = True
    print(f"[{scenario} seme {seed} v{ver}] B{args.B} T{args.T} H{args.H} D{args.D} {args.dev} stato={args.stato}")
    print(f"  {'':7s} {'rip32-fp64':>11s} {'nuova32-fp64':>12s} {'rapporto':>8s} {'nuova64-rip64':>13s} {'nuova32-rip32':>13s}")
    for nome, a_r, a_n, a64, an64 in list(zip(nomi_u, ur, un, u64, un64)) + list(zip(NOMI, gr, gn, g64, gn64)):
        er, en = rel(a_r, a64), rel(a_n, a64)
        alg = rel(an64, a64)
        rapporto = en / max(er, 1e-300)
        passa = en <= 2 * er or en < 1e-12
        ok &= passa
        print(f"  {nome:7s} {er:11.3e} {en:12.3e} {rapporto:8.2f} {alg:13.3e} {rel(a_n, a_r):13.3e} "
              f"{'ok' if passa else 'FUORI'}", flush=True)
    return ok


def tempi(args):
    dati, stato = ingressi(1, args.B, args.T, args.H, args.D, args.dev, SCENARI["normale"], False)
    riga = {}
    cand = ([] if args.senza_ripiego else [("ripiego", ripiego)]) + [(f"v{v}", candidato(int(v)))
                                                                      for v in args.versioni.split(",")]
    for nome, f in cand:
        t0 = time.perf_counter()
        corri(f, dati, stato)                                                   # riscaldamento (compilazione)
        print(f"  {nome:8s} primo avanti+indietro (con compilazione) {time.perf_counter() - t0:.1f} s", flush=True)
        if args.dev != "cpu":
            torch.cuda.reset_peak_memory_stats()
        _, _, ta, ti = corri(f, dati, stato, ripeti=args.R)
        picco = torch.cuda.max_memory_allocated() / 2**30 if args.dev != "cpu" else float("nan")
        riga[nome] = (statistics.median(ta), statistics.median(ti), picco)
        print(f"  {nome:8s} avanti {riga[nome][0]*1e3:8.1f} ms  indietro {riga[nome][1]*1e3:8.1f} ms  "
              f"totale {(riga[nome][0]+riga[nome][1])*1e3:8.1f} ms  picco memoria {picco:.2f} GiB", flush=True)
    r = riga.get("ripiego")
    for nome, n in (riga.items() if r else ()):
        if nome != "ripiego":
            print(f"  accelerazione {nome}: avanti x{r[0]/n[0]:.1f}  indietro x{r[1]/n[1]:.1f}  "
                  f"totale x{(r[0]+r[1])/(n[0]+n[1]):.1f}")


def sezioni(args):
    """Tempi dei pezzi della v2/v3 da soli, alle dimensioni vere: la ricorrenza (Function) e (I + L)^-1."""
    import gdn_veloce as G
    dev = args.dev
    N, BH, C, D = args.T // 64, args.B * args.H, 64, args.D
    gen = torch.Generator(device="cpu").manual_seed(3)
    mk = lambda *s: torch.randn(*s, generator=gen).to(dev)             # noqa: E731
    u, w, kw, qg = (mk(N, BH, C, D) * 0.1 for _ in range(4))
    attn = mk(N, BH, C, C).tril() * 0.1
    egl = torch.rand(N, BH, generator=gen).to(dev) * 0.5 + 0.4
    S0 = torch.zeros(BH, D, D, device=dev)
    do = mk(N, BH, C, D)
    Lm = (mk(N, BH, C, C) * 0.05).tril(-1)
    dX = mk(N, BH, C, C)

    def cronometra(f, ripeti=args.R):
        f(); sinc(dev)
        tt = []
        for _ in range(ripeti):
            t0 = time.perf_counter(); f(); sinc(dev); tt.append(time.perf_counter() - t0)
        return statistics.median(tt) * 1e3

    classi = [("ricorrenza v2", G._RicorrenzaUscita), ("ricorrenza v3", G._RicorrenzaUscita3)]
    if G._TRITON and dev != "cpu":
        classi.append(("ricorrenza v4 (Triton)", G._RicorrenzaTriton))
    for nome, cls in classi:
        def solo_avanti():
            with torch.no_grad():
                cls.apply(u, w, kw, egl, qg, attn, S0)

        def av_ind():
            xs = [t.clone().requires_grad_(True) for t in (u, w, kw, egl, qg, attn)]
            o, _ = cls.apply(*xs, S0)
            o.backward(do)
        print(f"  {nome}: avanti {cronometra(solo_avanti):.1f} ms, avanti+indietro {cronometra(av_ind):.1f} ms",
              flush=True)

    def inv_av():
        with torch.no_grad():
            G._Inversa.apply(Lm)

    def inv_ai():
        L = Lm.clone().requires_grad_(True)
        G._Inversa.apply(L).backward(dX)

    def inv_generica():
        L = Lm.clone().requires_grad_(True)
        eye = torch.eye(C, device=dev).expand(Lm.shape)
        torch.linalg.solve_triangular(L, eye, upper=False, unitriangular=True).backward(dX)
    print(f"  (I+L)^-1: avanti {cronometra(inv_av):.1f} ms, avanti+indietro a mano {cronometra(inv_ai):.1f} ms, "
          f"generica {cronometra(inv_generica):.1f} ms", flush=True)


def sweep_triton(args):
    """Ricorrenza Triton da sola, per blocco di colonne BV e numero di warp."""
    import gdn_veloce as G
    dev = args.dev
    N, BH, C, D = args.T // 64, args.B * args.H, 64, args.D
    gen = torch.Generator(device="cpu").manual_seed(3)
    mk = lambda *s: torch.randn(*s, generator=gen).to(dev)             # noqa: E731
    u, w, kw, qg = (mk(N, BH, C, D) * 0.1 for _ in range(4))
    attn = mk(N, BH, C, C).tril() * 0.1
    egl = torch.rand(N, BH, generator=gen).to(dev) * 0.5 + 0.4
    S0 = torch.zeros(BH, D, D, device=dev)
    do = mk(N, BH, C, D)
    rif = None
    griglia = [tuple(map(int, c.split("x"))) for c in args.griglia_triton.split(",")]
    for bv, nw in griglia:
        G._BV, G._NW = bv, nw
        try:
            def av_ind():
                xs = [t.clone().requires_grad_(True) for t in (u, w, kw, egl, qg, attn)]
                o, _ = G._RicorrenzaTriton.apply(*xs, S0)
                o.backward(do)
                return o, xs[1].grad
            o, gw = av_ind(); sinc(dev)
            if rif is None:
                rif = (o, gw)
            err = rel(o.double(), rif[0].double()) + rel(gw.double(), rif[1].double())
            tt = []
            for _ in range(args.R):
                t0 = time.perf_counter(); av_ind(); sinc(dev); tt.append(time.perf_counter() - t0)
            with torch.no_grad():
                ta = []
                for _ in range(args.R):
                    t0 = time.perf_counter(); G._RicorrenzaTriton.apply(u, w, kw, egl, qg, attn, S0); sinc(dev)
                    ta.append(time.perf_counter() - t0)
            print(f"  BV={bv:3d} warp={nw}: avanti {statistics.median(ta)*1e3:6.1f} ms, avanti+indietro "
                  f"{statistics.median(tt)*1e3:6.1f} ms (scarto dal primo {err:.1e})", flush=True)
        except Exception as e:                       # risorse esaurite ecc.: si registra e si prosegue
            print(f"  BV={bv:3d} warp={nw}: ERRORE {type(e).__name__}: {str(e)[:120]}", flush=True)
    G._BV, G._NW = 32, 8


def profilo(args, quale):
    from torch.profiler import ProfilerActivity, profile
    f = ripiego if quale == "ripiego" else candidato(int(quale[1:]))
    dati, stato = ingressi(1, args.B, args.T, args.H, args.D, args.dev, SCENARI["normale"], False)
    corri(f, dati, stato)
    att = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if args.dev != "cpu" else [])
    with profile(activities=att) as p:
        corri(f, dati, stato)
    chiave = "self_cuda_time_total" if args.dev != "cpu" else "self_cpu_time_total"
    print(f"== profilo {quale} ==")
    print(p.key_averages().table(sort_by=chiave, row_limit=args.righe, max_name_column_width=60), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="cuda")
    ap.add_argument("--B", type=int, default=2)
    ap.add_argument("--T", type=int, default=4096)
    ap.add_argument("--H", type=int, default=32)
    ap.add_argument("--D", type=int, default=128)
    ap.add_argument("--R", type=int, default=5)
    ap.add_argument("--stato", action="store_true", help="stato iniziale + stato finale nella perdita")
    ap.add_argument("--scenari", default="normale,forte,debole")
    ap.add_argument("--semi", default="1")
    ap.add_argument("--fasi", default="confronto,tempi")
    ap.add_argument("--righe", type=int, default=25)
    ap.add_argument("--versioni", default="1,2")
    ap.add_argument("--senza-ripiego", action="store_true")
    ap.add_argument("--griglia-triton", default="16x8,16x16,32x16,8x8,8x16")
    ap.add_argument("--fili", type=int, default=4)
    ap.add_argument("--blas", default="", help="torch.backends.cuda.preferred_blas_library (prova)")
    args = ap.parse_args()
    torch.set_num_threads(args.fili)
    if args.blas:
        print("blas:", torch.backends.cuda.preferred_blas_library(args.blas), flush=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    fasi = args.fasi.split(",")
    ok = True
    if "confronto" in fasi:
        for sc in args.scenari.split(","):
            for s in args.semi.split(","):
                ok &= confronta(args, sc, int(s))
        print("EQUIVALENZA:", "PASSA" if ok else "NON PASSA", flush=True)
    if "tempi" in fasi:
        tempi(args)
    if "sezioni" in fasi:
        sezioni(args)
    if "sweep_triton" in fasi:
        sweep_triton(args)
    for q in ["ripiego"] + [f"v{v}" for v in args.versioni.split(",")]:
        if f"profilo_{q}" in fasi:
            profilo(args, q)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
