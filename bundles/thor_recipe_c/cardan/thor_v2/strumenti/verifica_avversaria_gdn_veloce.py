#!/usr/bin/env python3
"""Verifica AVVERSARIA di strumenti/gdn_veloce.py (v5 = v3 + preparazione compilata) e della copia in servizio
strumenti/gdn_veloce_corsa.py (v3 congelata), 27/9/2026. Non riusa niente della prova dell'autore:

  * riferimento fp64 DERIVATO MECCANICAMENTE dal sorgente di transformers (inspect.getsource del ripiego,
    sostituzioni contate: `torch.float32` -> fp64 e `.float()` -> fp64), non una copia a mano; lo stesso per il
    ripiego RICORRENTE token per token (algoritmo diverso: controlla il riferimento stesso, su CPU piccolo);
  * ingressi con la DISPOSIZIONE VERA della corsa: uscita della conv (B, C, T) trasposta, split in q/k/v (viste non
    contigue), q e k da 16 teste ripetute a 32 (repeat_interleave), come Qwen3_5MoeGatedDeltaNet.forward; i
    gradienti dq, dk, dv sono quelli del tensore di partenza (somma sulle ripetizioni compresa);
  * scenari estremi: g molto negativo (fino a -1000 per passo, esponenziali sotto il minimo fp32), g = 0 esatto,
    beta vicino a 0 (~1e-5) e a 1 (anche 1.0 esatto in fp32), chiavi correlate con beta ~1 (caso peggiore di
    (I + L)^-1), estremi mescolati per testa e per token; stato iniziale/finale e T non multiplo di 64;
  * metriche: errore relativo L2 globale verso fp64 (criterio: nuova <= 2 x ripiego fp32), PIU' la testa peggiore e
    l'errore massimo assoluto (per non nascondere un difetto locale nella norma globale), e i non finiti;
  * precisione nascosta: stato di TF32/matmul precision SENZA toccarli (come la corsa), controllo che la
    compilazione della v5 sia davvero attiva (nessun ritorno silenzioso alla v3) e scansione del codice Triton
    generato da Inductor alla ricerca di cast a fp16/bf16, tl.dot, tf32, fast-math;
  * tempi con torch.cuda.synchronize prima e dopo ogni fase, candidati alternati a ogni ripetizione, anche
    l'avanti SENZA grafo (la corsa fa avanti no_grad + ricalcolo con grafo + indietro per ogni strato).

GPU: lanciare come unita' systemd --user (un solo processo di prova accanto alla corsa).
"""
import argparse
import glob
import inspect
import os
import re
import statistics
import sys
import textwrap
import time
import warnings

import torch
import torch.nn.functional as F

sys.modules["fla"] = None                  # il ripiego PyTorch, come nella corsa (fla non importabile nel venv)
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as M  # noqa: E402
del sys.modules["fla"]

QUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, QUI)
import gdn_veloce as G                    # noqa: E402
import gdn_veloce_corsa as GC             # noqa: E402

RIPIEGO = M.torch_chunk_gated_delta_rule   # quello che la corsa chiama col default E2E_GDN=ripiego


def _derivata(fn, sostituzioni):
    """Il sorgente VERO di transformers con le sole sostituzioni di dtype, contate (se transformers cambia, si ferma)."""
    righe = textwrap.dedent(inspect.getsource(fn)).splitlines()
    while righe and righe[0].lstrip().startswith("@"):
        righe.pop(0)
    src = "\n".join(righe)
    for a, b, n in sostituzioni:
        c = src.count(a)
        if c != n:
            raise RuntimeError(f"{fn.__name__}: '{a}' trovato {c} volte, attese {n}: il sorgente e' cambiato")
        src = src.replace(a, b)
    ns = dict(vars(M))
    exec(compile(src, f"<{fn.__name__} derivata>", "exec"), ns)
    return ns[fn.__name__], ns


def riferimento_chunk(dt):
    f, ns = _derivata(RIPIEGO.__wrapped__, [("torch.float32", "_DT", 1), (".float()", ".to(_DT)", 1)])
    ns["_DT"] = dt
    return f


def riferimento_ricorrente(dt):
    f, ns = _derivata(M.torch_recurrent_gated_delta_rule.__wrapped__, [("torch.float32", "_DT", 1)])
    ns["_DT"] = dt

    def g(q, k, v, g, beta, chunk_size=64, initial_state=None, output_final_state=False,
          use_qk_l2norm_in_kernel=False, **kw):
        return f(q, k, v, g=g, beta=beta, initial_state=initial_state, output_final_state=output_final_state,
                 use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel)
    return g


def v_nuova(versione):
    def f(*a, **kw):
        return G.chunk_gated_delta_rule_veloce(*a, _versione=versione, **kw)
    return f


def v_nuova64(versione):
    def f(*a, **kw):
        return G.chunk_gated_delta_rule_veloce(*a, _versione=versione, _dtype_calcolo=torch.float64, **kw)
    return f


def v_corsa(*a, **kw):
    return GC.chunk_gated_delta_rule_veloce(*a, **kw)


def contigui(f):
    """Leva disposizione (prova): q, k, v resi contigui PRIMA della funzione (nella corsa v arriva come vista dello
    split trasposto, con passo T sull'ultima dimensione). Valori identici: cambia solo la disposizione in memoria."""
    def h(q, k, v, *a, **kw):
        return f(q.contiguous(), k.contiguous(), v.contiguous(), *a, **kw)
    return h


# ───────────────────────────── scenari ─────────────────────────────
def _g_beta(nome, B, T, H, gen):
    r = lambda *s: torch.randn(*s, generator=gen)            # noqa: E731
    u = lambda *s: torch.rand(*s, generator=gen)             # noqa: E731
    if nome == "normale":
        return -F.softplus(r(B, T, H)), torch.sigmoid(r(B, T, H))
    if nome == "g_molto_negativo":          # da -1 a ~-200 per passo: decadimenti e^g sotto il minimo fp32
        return -F.softplus(r(B, T, H)) * 60 - 1, torch.sigmoid(r(B, T, H))
    if nome == "g_zero":                   # nessun decadimento: lo stato accumula su 4096 token
        return torch.zeros(B, T, H), torch.sigmoid(r(B, T, H))
    if nome == "beta_vicino_0":
        return -F.softplus(r(B, T, H)), torch.sigmoid(r(B, T, H) - 12)
    if nome == "beta_vicino_1":            # sigmoid(x + 12): da 1-6e-6 fino a 1.0 esatto in fp32
        return -F.softplus(r(B, T, H)) * 0.05, torch.sigmoid(r(B, T, H) + 12)
    if nome == "correlate_beta1":          # chiavi correlate + beta ~1 + decadimento debole: (I + L) mal condizionata
        return -F.softplus(r(B, T, H)) * 0.02, torch.sigmoid(r(B, T, H) + 8)
    if nome == "estremi_misti":            # per testa e per token: g in {0, -1e-3, -5, -1000}, beta in {~0, 1, U}
        scelte_g = torch.tensor([0.0, -1e-3, -5.0, -1000.0])
        ig = torch.randint(0, 4, (B, T, H), generator=gen)
        testa_forte = (torch.arange(H) % 2 == 0).view(1, 1, H)        # meta' teste: solo g estremi
        ig = torch.where(testa_forte, torch.where(ig % 2 == 0, 0, 3), ig)
        g = scelte_g[ig]
        ib = torch.randint(0, 3, (B, T, H), generator=gen)
        beta = torch.where(ib == 0, torch.sigmoid(r(B, T, H) - 12),
                           torch.where(ib == 1, torch.ones(B, T, H), u(B, T, H)))
        return g, beta
    raise ValueError(nome)


COMUNE = {"correlate_beta1": 3.0}           # peso della direzione comune in q e k (coseno ~0.9)


def ingressi(nome, seed, B, T, Hk, H, D, stato):
    gen = torch.Generator().manual_seed(seed)
    Ck, Cv = Hk * D, H * D
    base = torch.randn(B, 2 * Ck + Cv, T, generator=gen)          # uscita della conv: (B, C, T)
    c = COMUNE.get(nome, 0.0)
    if c:
        d = torch.randn(B, Ck, 1, generator=gen)
        base[:, :Ck] += c * d
        base[:, Ck:2 * Ck] += c * d
    g, beta = _g_beta(nome, B, T, H, gen)
    do = torch.randn(B, T, H, D, generator=gen)
    s0 = torch.randn(B, H, D, D, generator=gen) * 0.1 if stato else None
    dS = torch.randn(B, H, D, D, generator=gen) if stato else None
    return dict(base=base, g=g, beta=beta, do=do, s0=s0, dS=dS, Hk=Hk, H=H, D=D)


def viste(base, Hk, H, D):
    """Come Qwen3_5MoeGatedDeltaNet.forward: transpose, split, reshape, repeat_interleave."""
    B, _, T = base.shape
    mixed = base.transpose(1, 2)
    q, k, v = torch.split(mixed, [Hk * D, Hk * D, H * D], dim=-1)
    q, k, v = q.reshape(B, T, -1, D), k.reshape(B, T, -1, D), v.reshape(B, T, -1, D)
    if H // Hk > 1:
        q = q.repeat_interleave(H // Hk, dim=2)
        k = k.repeat_interleave(H // Hk, dim=2)
    return q, k, v


def sinc(dev):
    if dev == "cuda":
        torch.cuda.synchronize()


def corri(f, dati, dev, dt=torch.float32, con_grafo=True):
    """Avanti (+ indietro). Ritorna (uscite, gradienti, t_avanti, t_indietro)."""
    Hk, H, D = dati["Hk"], dati["H"], dati["D"]
    foglia = lambda x: x.to(dev, dt).detach().clone().requires_grad_(con_grafo)      # noqa: E731 (mai il tensore dei dati)
    base, g, beta = foglia(dati["base"]), foglia(dati["g"]), foglia(dati["beta"])
    s = None if dati["s0"] is None else foglia(dati["s0"])
    with torch.set_grad_enabled(con_grafo):
        sinc(dev); t0 = time.perf_counter()
        q, k, v = viste(base, Hk, H, D)
        o, SN = f(q, k, v, g=g, beta=beta, initial_state=s, output_final_state=s is not None,
                  use_qk_l2norm_in_kernel=True, cu_seqlens=None)
        sinc(dev); t1 = time.perf_counter()
    if not con_grafo:
        return None, None, t1 - t0, 0.0
    perdita = (o * dati["do"].to(dev, o.dtype)).sum()
    if SN is not None:
        perdita = perdita + (SN * dati["dS"].to(dev, SN.dtype)).sum()
    sinc(dev); t2 = time.perf_counter()
    perdita.backward()
    sinc(dev); t3 = time.perf_counter()
    Ck = Hk * D
    gb = base.grad.detach()
    grad = {"dq": gb[:, :Ck], "dk": gb[:, Ck:2 * Ck], "dv": gb[:, 2 * Ck:], "dg": g.grad.detach(),
            "dbeta": beta.grad.detach()}
    if s is not None:
        grad["dS0"] = s.grad.detach()
    usc = {"o": o.detach()}
    if SN is not None:
        usc["S_fin"] = SN.detach()
    return {**usc, **grad}, None, t1 - t0, t3 - t2


def per_testa(nome, x, Hk, H, D):
    """(teste, resto) per l'errore della testa peggiore."""
    B = x.shape[0]
    if nome in ("dq", "dk"):
        return x.reshape(B, Hk, D, -1).reshape(B * Hk, -1)
    if nome == "dv":
        return x.reshape(B, H, D, -1).reshape(B * H, -1)
    if nome == "o":
        return x.permute(0, 2, 1, 3).reshape(B * H, -1)
    if nome in ("dg", "dbeta"):
        return x.permute(0, 2, 1).reshape(B * H, -1)
    return x.reshape(B * H, -1)                        # S_fin, dS0 (B, H, D, D)


def metriche(a, rif, nome, Hk, H, D):
    a, rif = a.double(), rif.double()
    e = a - rif
    n_rif = rif.norm().clamp_min(1e-300)
    ta, tr = per_testa(nome, e, Hk, H, D), per_testa(nome, rif, Hk, H, D)
    testa = (ta.norm(dim=1) / tr.norm(dim=1).clamp_min(1e-30))
    vive = tr.norm(dim=1) > 1e-30 * max(1.0, float(n_rif))
    peggiore = float(testa[vive].max()) if bool(vive.any()) else 0.0
    return dict(rel=float(e.norm() / n_rif), testa=peggiore, maxabs=float(e.abs().max()),
                nonfiniti=int((~torch.isfinite(a)).sum()))


def confronta(args, nome, seed, dev, cands):
    dati = ingressi(nome, seed, args.B, args.T, args.Hk, args.H, args.D, args.stato)
    r64, _, _, _ = corri(riferimento_chunk(torch.float64), dati, dev, torch.float64)
    rr, _, _, _ = corri(RIPIEGO, dati, dev)
    ok = True
    nf_rif = sum(int((~torch.isfinite(t)).sum()) for t in r64.values())
    print(f"\n[{nome} seme {seed}] B{args.B} T{args.T} Hk{args.Hk} H{args.H} D{args.D} {dev} stato={args.stato} "
          f"(non finiti nel riferimento fp64: {nf_rif})", flush=True)
    for cn, f in cands:
        rn, _, _, _ = corri(f, dati, dev)
        print(f"  -- {cn}: {'':6s} {'rip32':>9s} {'nuova32':>9s} {'rapp':>6s} | {'testa rip':>9s} {'testa nuova':>11s}"
              f" {'rapp':>6s} | {'maxabs rip':>10s} {'maxabs nuova':>12s} {'rapp':>6s} | nf", flush=True)
        for q in r64:
            mr = metriche(rr[q], r64[q], q, args.Hk, args.H, args.D)
            mn = metriche(rn[q], r64[q], q, args.Hk, args.H, args.D)
            rg = mn["rel"] / max(mr["rel"], 1e-300)
            rt = mn["testa"] / max(mr["testa"], 1e-300)
            rm = mn["maxabs"] / max(mr["maxabs"], 1e-300)
            passa = (mn["rel"] <= 2 * mr["rel"] or mn["rel"] < 1e-12) and mn["nonfiniti"] <= mr["nonfiniti"]
            nota = "ok" if passa else "FUORI"
            if passa and (rt > 2 or rm > 2):
                nota = "ok (locale >2x: GUARDARE)"
            ok &= passa
            print(f"     {q:6s} {mr['rel']:9.2e} {mn['rel']:9.2e} {rg:6.2f} | {mr['testa']:9.2e} {mn['testa']:11.2e} "
                  f"{rt:6.2f} | {mr['maxabs']:10.2e} {mn['maxabs']:12.2e} {rm:6.2f} | {mr['nonfiniti']}/"
                  f"{mn['nonfiniti']} {nota}", flush=True)
        del rn
    if args.algebra64:                               # la stessa algebra in fp64: deve coincidere col riferimento
        a64, _, _, _ = corri(v_nuova64(3), dati, dev, torch.float64)
        peg = max(metriche(a64[q], r64[q], q, args.Hk, args.H, args.D)["rel"] for q in r64)
        print(f"  -- algebra fp64 (v3 in fp64 contro riferimento fp64): peggiore errore relativo {peg:.2e}", flush=True)
        ok &= peg < 1e-10
    return ok


def controllo_riferimento(args):
    """Il riferimento a chunk fp64 contro il ricorrente token per token fp64 (algoritmo diverso), CPU piccolo."""
    ok = True
    for nome in args.scenari.split(","):
        dati = ingressi(nome, 5, 1, 200, 2, 4, 32, True)
        a, _, _, _ = corri(riferimento_chunk(torch.float64), dati, "cpu", torch.float64)
        b, _, _, _ = corri(riferimento_ricorrente(torch.float64), dati, "cpu", torch.float64)
        peg = max(float((a[q] - b[q]).norm() / b[q].norm().clamp_min(1e-300)) for q in a)
        print(f"  riferimento chunk fp64 vs ricorrente fp64 [{nome}]: peggiore {peg:.2e}", flush=True)
        ok &= peg < 1e-9
    return ok


def tempi(args, dev, cands):
    dati = ingressi("normale", 3, args.B, args.T, args.Hk, args.H, args.D, False)
    tutti = [("ripiego", RIPIEGO)] + cands
    for n, f in tutti:                                                   # riscaldamento/compilazione
        t0 = time.perf_counter()
        corri(f, dati, dev)
        corri(f, dati, dev, con_grafo=False)
        print(f"  {n:8s} riscaldamento (compilazione compresa) {time.perf_counter() - t0:.1f} s", flush=True)
    mis = {n: {"av": [], "ind": [], "ng": [], "picco": 0.0} for n, _ in tutti}
    for _ in range(args.R):                                              # alternati a ogni ripetizione
        for n, f in tutti:
            if dev == "cuda":
                torch.cuda.reset_peak_memory_stats()
            _, _, ta, ti = corri(f, dati, dev)
            if dev == "cuda":
                mis[n]["picco"] = max(mis[n]["picco"], torch.cuda.max_memory_allocated() / 2**30)
            _, _, tn, _ = corri(f, dati, dev, con_grafo=False)
            mis[n]["av"].append(ta); mis[n]["ind"].append(ti); mis[n]["ng"].append(tn)
    rip = mis["ripiego"]
    tot_r = statistics.median(rip["ng"]) + statistics.median(rip["av"]) + statistics.median(rip["ind"])
    for n, m in mis.items():
        av, ind, ng = (statistics.median(m[k]) * 1e3 for k in ("av", "ind", "ng"))
        print(f"  {n:8s} avanti {av:7.1f} ms  indietro {ind:7.1f} ms  (av+ind {av + ind:7.1f})  avanti no_grad {ng:7.1f}"
              f" ms  | come nella corsa (no_grad + av + ind) {ng + av + ind:7.1f} ms  x{tot_r * 1e3 / (ng + av + ind):.1f}"
              f"  | min/max av+ind {min(a + b for a, b in zip(m['av'], m['ind'])) * 1e3:.0f}/"
              f"{max(a + b for a, b in zip(m['av'], m['ind'])) * 1e3:.0f} ms  picco {m['picco']:.2f} GiB", flush=True)


def tempi_disposizione(args, dev, cands):
    """Stessi dati, due disposizioni alternate a ogni ripetizione: q/k/v CONTIGUI come foglie (la prova dell'autore)
    e la disposizione VERA (viste dello split + repeat_interleave, gradiente fino al tensore di partenza). Separa il
    costo della disposizione dalla contesa con la corsa."""
    dati = ingressi("normale", 3, args.B, args.T, args.Hk, args.H, args.D, False)
    q0, k0, v0 = (t.contiguous() for t in viste(dati["base"], args.Hk, args.H, args.D))
    do = dati["do"].to(dev)

    def contigua(f):
        xs = [t.to(dev).clone().requires_grad_(True) for t in (q0, k0, v0, dati["g"], dati["beta"])]
        sinc(dev); t0 = time.perf_counter()
        o, _ = f(xs[0], xs[1], xs[2], g=xs[3], beta=xs[4], use_qk_l2norm_in_kernel=True)
        sinc(dev); t1 = time.perf_counter()
        (o * do).sum().backward()
        sinc(dev); t2 = time.perf_counter()
        return t1 - t0, t2 - t1

    def vera(f):
        _, _, ta, ti = corri(f, dati, dev)
        return ta, ti
    tutti = [("ripiego", RIPIEGO)] + cands
    for _, f in tutti:
        contigua(f); vera(f)
    mis = {(n, d): [] for n, _ in tutti for d in ("contigua", "vera")}
    for _ in range(args.R):
        for n, f in tutti:
            for d, fun in (("contigua", contigua), ("vera", vera)):
                mis[(n, d)].append(sum(fun(f)))
    for (n, d), v in mis.items():
        print(f"  {n:8s} {d:9s} avanti+indietro mediana {statistics.median(v) * 1e3:7.1f} ms  "
              f"(min {min(v) * 1e3:6.0f}, max {max(v) * 1e3:6.0f}) "
              f"x{statistics.median(mis[('ripiego', d)]) / statistics.median(v):.1f}", flush=True)


def prep_pezzi(args, dev):
    """Dove nasce lo scarto della v5: ogni uscita di _prep (compilata fp32 e eager fp32) contro _prep eager fp64."""
    for nome in ("normale", "estremi_misti"):
        dati = ingressi(nome, 2, args.B, args.T, args.Hk, args.H, args.D, False)
        N, C = args.T // 64, 64

        def ing(dt):
            q, k, v = (t.to(dev, dt).contiguous() for t in viste(dati["base"], args.Hk, args.H, args.D))
            return q, k, v, dati["g"].to(dev, dt), dati["beta"].to(dev, dt)
        with torch.no_grad():
            r64 = G._prep(*ing(torch.float64), C, N, True)
            e32 = G._prep(*ing(torch.float32), C, N, True)
            c32 = G._prep_compilata()(*ing(torch.float32), C, N, True)
        print(f"  [_prep {nome}] errore relativo L2 / massimo assoluto verso fp64 (eager32 | compilata32):", flush=True)
        for n, a, b, r in zip(("Lm", "v_beta", "kbeg", "attn", "qg", "kw", "egl"), e32, c32, r64):
            m = metriche(a, r, "S_fin", 1, 1, 1)
            mc = metriche(b, r, "S_fin", 1, 1, 1)
            print(f"     {n:6s} {m['rel']:9.2e} {m['maxabs']:9.2e} | {mc['rel']:9.2e} {mc['maxabs']:9.2e}  "
                  f"x{mc['rel'] / max(m['rel'], 1e-300):.2f} / x{mc['maxabs'] / max(m['maxabs'], 1e-300):.2f}",
                  flush=True)
        del r64, e32, c32


def scansione_inductor(cartella):
    """Cast a precisione ridotta, tl.dot/tf32, fast-math nel codice Triton generato per _prep."""
    file = [f for f in glob.glob(os.path.join(cartella, "**", "*.py"), recursive=True)]
    trovati = {}
    pat = re.compile(r"tl\.float16|tl\.bfloat16|float16|bfloat16|tf32|input_precision|tl\.dot|fast_|approx|"
                     r"tl_math\.exp|extern_kernels\.\w+")
    kern = 0
    for f in file:
        try:
            t = open(f, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        if "@triton.jit" not in t and "extern_kernels" not in t:
            continue
        kern += t.count("@triton.jit")
        for m in pat.findall(t):
            trovati[m] = trovati.get(m, 0) + 1
    print(f"  codice Inductor: {len(file)} file, {kern} kernel Triton; occorrenze sensibili: {trovati}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="cuda")
    ap.add_argument("--B", type=int, default=2)
    ap.add_argument("--T", type=int, default=4096)
    ap.add_argument("--Hk", type=int, default=16)
    ap.add_argument("--H", type=int, default=32)
    ap.add_argument("--D", type=int, default=128)
    ap.add_argument("--R", type=int, default=7)
    ap.add_argument("--semi", default="2")
    ap.add_argument("--stato", action="store_true")
    ap.add_argument("--scenari", default="normale,g_molto_negativo,g_zero,beta_vicino_0,beta_vicino_1,"
                                         "correlate_beta1,estremi_misti")
    ap.add_argument("--candidati", default="v5,v3,corsa")
    ap.add_argument("--fasi", default="riferimento,confronto,tempi")
    ap.add_argument("--algebra64", action="store_true")
    ap.add_argument("--fili", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.fili)
    dev = args.dev
    print(f"torch {torch.__version__} | allow_tf32 matmul={torch.backends.cuda.matmul.allow_tf32} "
          f"cudnn={torch.backends.cudnn.allow_tf32} | float32_matmul_precision={torch.get_float32_matmul_precision()}"
          f" | GDN_VELOCE_VERSIONE default={G.VERSIONE} | inductor cache={os.environ.get('TORCHINDUCTOR_CACHE_DIR')}",
          flush=True)
    mappa = {"v5": v_nuova(5), "v3": v_nuova(3), "corsa": v_corsa, "v5c": contigui(v_nuova(5)),
             "v3c": contigui(v_nuova(3)), "ripiegoc": contigui(RIPIEGO)}
    cands = [(c, mappa[c]) for c in args.candidati.split(",")]
    ok = True
    avvisi = []
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        if "riferimento" in args.fasi:
            ok &= controllo_riferimento(args)
        if "confronto" in args.fasi:
            for s in args.semi.split(","):
                for nome in args.scenari.split(","):
                    ok &= confronta(args, nome, int(s), dev, cands)
        if "tempi" in args.fasi:
            tempi(args, dev, cands)
        if "prep" in args.fasi:
            prep_pezzi(args, dev)
        if "disposizione" in args.fasi:
            tempi_disposizione(args, dev, cands)
        avvisi = [str(x.message)[:200] for x in w if "gdn_veloce" in str(x.message)]
    print(f"\ncompilazione v5 fallita? {G._COMPILAZIONE_FALLITA} | avvisi gdn_veloce: {avvisi}", flush=True)
    ok &= not G._COMPILAZIONE_FALLITA
    if os.environ.get("TORCHINDUCTOR_CACHE_DIR"):
        scansione_inductor(os.environ["TORCHINDUCTOR_CACHE_DIR"])
    print("ESITO:", "PASSA" if ok else "NON PASSA", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
