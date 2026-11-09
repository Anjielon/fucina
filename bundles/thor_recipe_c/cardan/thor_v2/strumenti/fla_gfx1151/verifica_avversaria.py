#!/usr/bin/env python3
"""Verifica avversaria della leva fla su gfx1151 (27/9/2026) — rifatta da capo, senza riusare gli esiti del prototipo.

Cosa controlla, in un solo processo GPU (lanciare con lancia_prova.sh: unita' systemd --user, EB2 accanto e basta):
  0. il riferimento: ripiego_fp64 (G) contro una ricorrenza token per token scritta QUI in fp64 (CPU, autograd),
     anche con g molto negativo e beta agli estremi — se il riferimento sbaglia, tutto il resto non vale;
  1. alle dimensioni vere (2, 4096, 32, 128), sei regimi di ingresso con semi diversi da quelli del prototipo:
     g dolce · g molto negativo · beta ~ 0 · beta ~ 1 · g ~ 0 con beta ~ 1 (memoria lunga) · misto per testa;
     errore verso fp64 di fla, ripiego fp32 e gdn_veloce (copia della corsa) su uscita, dq, dk, dv, dg, dbeta
     e su tre RIDUZIONI lungo i token (sum dg, sum dg*g ~ dA_log, sum dbeta), che sono cio' che arriva ai pesi;
  2. tempi avanti+indietro con sincronizzazione, alternati, mediana;
  3. abilita_sicuro(M) vero: involucro attivo, instradamento (T%64 != 0 -> ripiego), gradienti finiti.
Criterio: rapporto errore(candidato)/errore(ripiego fp32) <= 2 su ogni grandezza.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import torch

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
sys.path.insert(0, str(QUI.parent.parent))                  # thor_v2: strumenti.gdn_veloce_corsa
import gdn_fla_gfx1151 as G  # noqa: E402

sys.modules["fla"] = None                                   # transformers lega il ripiego, come nella corsa
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as M  # noqa: E402
from strumenti.gdn_veloce_corsa import chunk_gated_delta_rule_veloce as VELOCE  # noqa: E402

RIP = M.torch_chunk_gated_delta_rule
R64 = G.ripiego_fp64(M)
NOMI = ("avanti", "dq", "dk", "dv", "dg", "dbeta")


def log(*a):
    print(*a, flush=True)


def ricorrenza64(q, k, v, g, beta):
    """Gated delta rule token per token (fp64), scritta da capo: stato [dk, dv], decadimento, poi correzione delta."""
    q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    q = q * q.shape[-1] ** -0.5
    B, T, H, D = q.shape
    S = torch.zeros(B, H, D, v.shape[-1], dtype=q.dtype, device=q.device)
    uscite = []
    for t in range(T):
        S = S * g[:, t].exp()[..., None, None]
        kt = k[:, t]
        err = v[:, t] - torch.einsum("bhk,bhkv->bhv", kt, S)
        S = S + torch.einsum("bhk,bhv->bhkv", kt, err * beta[:, t][..., None])
        uscite.append(torch.einsum("bhk,bhkv->bhv", q[:, t], S))
    return torch.stack(uscite, 1)


def ingressi(seme, B, T, H, D, regime):
    gen = torch.Generator(device="cpu").manual_seed(seme)
    q, k, v, do = (torch.randn(B, T, H, D, generator=gen) for _ in range(4))
    z1, z2 = torch.randn(B, T, H, generator=gen), torch.randn(B, T, H, generator=gen)
    sp = torch.nn.functional.softplus
    if regime == "dolce":
        g, b = -sp(z1), torch.sigmoid(z2)
    elif regime == "g_molto_neg":
        g, b = -30.0 * sp(z1 + 1.0), torch.sigmoid(z2)
    elif regime == "beta_zero":
        g, b = -sp(z1), torch.sigmoid(z2 - 9.0)
    elif regime == "beta_uno":
        g, b = -sp(z1), torch.sigmoid(z2 + 9.0)
    elif regime == "memoria_lunga":
        g, b = -1e-3 * sp(z1), torch.sigmoid(z2 + 9.0)
    elif regime == "misto":                                  # teste diverse, estremi diversi
        h = torch.arange(H)
        scala = torch.where(h % 4 == 0, 40.0, torch.where(h % 4 == 1, 1e-3, 1.0))
        spost = torch.where(h % 3 == 0, 9.0, torch.where(h % 3 == 1, -9.0, 0.0))
        g, b = -scala * sp(z1), torch.sigmoid(z2 + spost)
    else:
        raise ValueError(regime)
    return q, k, v, g, b, do


def passo(f, dati, dev, dt):
    xs = [t.to(dev, dt).requires_grad_(True) for t in dati[:5]]
    o, _ = f(xs[0], xs[1], xs[2], g=xs[3], beta=xs[4], initial_state=None, output_final_state=False,
             use_qk_l2norm_in_kernel=True)
    o.backward(dati[5].to(dev, dt))
    return [o.detach()] + [x.grad.detach() for x in xs]


def rel(a, b) -> float:
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-300))


def riduzioni(u, g):
    dg, db = u[4].double(), u[5].double()
    return {"sum_dg": dg.sum((0, 1)), "dAlog~sum_dg*g": (dg * g).sum((0, 1)), "sum_dbeta": db.sum((0, 1))}


def controllo_riferimento() -> dict:
    """0: ripiego_fp64 contro la ricorrenza scritta qui (CPU fp64), tre regimi."""
    esito = {}
    for regime in ("dolce", "g_molto_neg", "misto"):
        dati = ingressi(101, 1, 256, 4, 128, regime)
        ur = passo(R64, dati, "cpu", torch.float64)
        un = passo(lambda q, k, v, g, beta, **kw: (ricorrenza64(q, k, v, g, beta), None), dati, "cpu",
                   torch.float64)
        esito[regime] = {n: rel(a, b) for n, a, b in zip(NOMI, ur, un)}
        log(f"[0] riferimento {regime}: " + " ".join(f"{n} {e:.1e}" for n, e in esito[regime].items()))
    return esito


def caso(fla_gdn, seme, regime, dims) -> dict:
    B, T, H, D = dims
    dati = ingressi(seme, B, T, H, D, regime)
    g64 = dati[3].double().to("cuda")
    u64 = passo(R64, dati, "cuda", torch.float64)
    red64 = riduzioni(u64, g64)
    ris = {"seme": seme, "regime": regime, "g_min": float(dati[3].min()), "g_max": float(dati[3].max()),
           "beta_min": float(dati[4].min()), "beta_max": float(dati[4].max())}
    errori = {}
    for nome, f in (("ripiego", RIP), ("fla", fla_gdn), ("veloce", VELOCE)):
        u = passo(f, dati, "cuda", torch.float32)
        e = {n: rel(a, b) for n, a, b in zip(NOMI, u, u64)}
        e.update({n: rel(a, red64[n]) for n, a in riduzioni(u, g64).items()})
        e["finito"] = all(bool(torch.isfinite(t).all()) for t in u)
        errori[nome] = e
        del u
        torch.cuda.empty_cache()
    ris["errori"] = errori
    rip = errori["ripiego"]
    for cand in ("fla", "veloce"):
        rap = {n: (errori[cand][n] / rip[n] if rip[n] > 0 else float("inf")) for n in rip if n != "finito"}
        ris[f"rapporto_{cand}"] = rap
        peggio = max(rap, key=rap.get)
        ris[f"esito_{cand}"] = bool(errori[cand]["finito"] and rap[peggio] <= 2.0)
        log(f"[1] {regime:13s} seme {seme} g[{ris['g_min']:.1f},{ris['g_max']:.1e}] "
            f"beta[{ris['beta_min']:.1e},{ris['beta_max']:.4f}] {cand}: "
            + " ".join(f"{n} {v:.2f}" for n, v in rap.items())
            + f" -> peggiore {peggio} {rap[peggio]:.2f} {'OK' if ris[f'esito_{cand}'] else 'FUORI'}")
    log("    errori assoluti vs fp64 (ripiego/fla/veloce): "
        + " ".join(f"{n} {rip[n]:.1e}/{errori['fla'][n]:.1e}/{errori['veloce'][n]:.1e}" for n in rip
                   if n != "finito"))
    del u64, dati
    torch.cuda.empty_cache()
    return ris


def tempi(fla_gdn, dims, rip=5) -> dict:
    dati = ingressi(7, *dims, "dolce")
    dati = [t.to("cuda") for t in dati]
    fns = (("ripiego", RIP), ("fla", fla_gdn), ("veloce", VELOCE))
    mis = {n: [] for n, _ in fns}
    for n, f in fns:                                         # riscaldamento
        passo(f, dati, "cuda", torch.float32)
    for _ in range(rip):
        for n, f in fns:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            passo(f, dati, "cuda", torch.float32)
            torch.cuda.synchronize()
            mis[n].append(time.perf_counter() - t0)
    esito = {n: statistics.median(v) for n, v in mis.items()}
    log("[2] tempi avanti+indietro (mediana, sincronizzati): "
        + " ".join(f"{n} {v:.3f}s" for n, v in esito.items())
        + f" · fla x{esito['ripiego'] / esito['fla']:.2f} · veloce x{esito['ripiego'] / esito['veloce']:.2f}")
    return {"mediane": esito, "tutte": mis}


def collaudo_abilita() -> dict:
    """3: abilita_sicuro vero sul modulo di transformers, poi instradamento e gradienti."""
    r = G.abilita_sicuro(M)
    log(f"[3] abilita_sicuro: attivo {r['attivo']} motivo {r.get('motivo')} controllo {r.get('controllo')}")
    if not r["attivo"]:
        return {"resoconto": {k: v for k, v in r.items() if k != "risorse"}}
    f = M.torch_chunk_gated_delta_rule
    out = {}
    for T in (4096, 1000):                                   # 1000 % 64 != 0 -> deve andare al ripiego
        dati = ingressi(55, 1, T, 32, 128, "dolce")
        prima = dict(f.stato)
        u = passo(f, dati, "cuda", torch.float32)
        out[T] = {"fla": f.stato["chiamate_fla"] - prima["chiamate_fla"],
                  "ripiego": f.stato["chiamate_ripiego"] - prima["chiamate_ripiego"],
                  "finito": all(bool(torch.isfinite(t).all()) for t in u)}
    log(f"[3] instradamento: {out} · guasto {f.stato['guasto']}")
    M.torch_chunk_gated_delta_rule = f.ripiego
    return {"resoconto": {k: v for k, v in r.items() if k != "risorse"}, "instradamento": out}


def main() -> int:
    log(f"torch {torch.__version__} · {torch.cuda.get_device_name(0)} · allow_tf32 matmul "
        f"{torch.backends.cuda.matmul.allow_tf32} cudnn {torch.backends.cudnn.allow_tf32} · "
        f"precisione fp32 matmul {torch.get_float32_matmul_precision()}")
    # --salta-riferimento: il passo 0 e' solo CPU (lento sotto CPUQuota): si puo' eseguire a parte, senza GPU,
    # con `python3 -c "import verifica_avversaria as V; V.controllo_riferimento()"` (27/9: fatto, 5e-16..8e-13)
    esiti = {"riferimento": None if "--salta-riferimento" in sys.argv else controllo_riferimento()}
    fla_gdn = G.configura()
    import os
    log(f"TRITON_F32_DEFAULT={os.environ.get('TRITON_F32_DEFAULT')} · FLA_USE_FAST_OPS="
        f"{os.environ.get('FLA_USE_FAST_OPS')} · impronta copia {(G.PKG / 'IMPRONTA').read_text().strip()}")
    dims = (2, 4096, 32, 128)
    esiti["casi"] = [caso(fla_gdn, s, r, dims) for s, r in ((31, "dolce"), (32, "g_molto_neg"), (33, "beta_zero"),
                                                           (34, "beta_uno"), (35, "memoria_lunga"), (36, "misto"))]
    esiti["tempi"] = tempi(fla_gdn, dims)
    esiti["abilita"] = collaudo_abilita()
    esiti["risorse"] = G.RISORSE
    for n, r in G.RISORSE.items():
        log(f"    risorse {n}: {r}")
    (QUI / "esiti").mkdir(exist_ok=True)
    (QUI / "esiti" / "verifica_avversaria.json").write_text(json.dumps(esiti, indent=1, default=str))
    log("FINE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
