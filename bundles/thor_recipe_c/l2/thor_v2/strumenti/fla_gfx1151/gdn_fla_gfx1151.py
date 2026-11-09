#!/usr/bin/env python3
"""GDN di Thor-T con il kernel Triton di fla (copia in pkg/), su gfx1151 — caricatore unico (27/9/2026).

Tre responsabilita', niente altro:
  1. carica_fla(): importa la COPIA di fla (pkg/, preparata da prepara_copia.sh) e verifica che sia proprio quella,
     senza lasciare pkg/ in sys.path (nessun altro import di fla puo' pescarla per caso);
  2. fissa_config(): sostituisce lo spazio di autotune di ogni kernel con UNA configurazione scelta (tabella
     CONFIG_GFX1151 o regola di ripiego) — su gfx1151 l'autotune fp32 costa decine di minuti di compilazione e
     alcune configurazioni vanno in page fault (vedi prova_equivalenza.py);
  3. abilita_in_transformers(M): sostituisce M.torch_chunk_gated_delta_rule (globale di modulo, letto a ogni
     chiamata dal forward di Qwen3_5MoeGatedDeltaNet) con un involucro che chiama fla SOLO nel caso provato
     (fp32, niente cu_seqlens, niente stato iniziale/finale, q/k l2 nel kernel) e altrimenti il ripiego.

Il decoratore di transformers use_kernel_func_from_hub_with_fallback sceglie fla all'IMPORT del modulo: per questo
carica_fla() va chiamato DOPO l'import di modeling_qwen3_5_moe, e per questo fla non va mai reso importabile nel venv.

Abilitazione nella corsa (da chi possiede e2e_studente.py, al riavvio; qui non si tocca), dopo l'import del modeling:
    if os.environ.get("THOR_GDN_FLA") == "1":
        sys.path.insert(0, "@@ROOT:forge@@/thor_v2/strumenti/fla_gfx1151")
        import gdn_fla_gfx1151
        print("GDN fla:", gdn_fla_gfx1151.abilita_sicuro(M), flush=True)   # M = modeling_qwen3_5_moe
Senza la variabile non cambia nulla; con la variabile, se un controllo non torna resta il ripiego (resoconto stampato).
Preparare prima la copia: strumenti/fla_gfx1151/prepara_copia.sh
"""
from __future__ import annotations

import gc
import os
import sys
from pathlib import Path
from typing import Callable, Optional

QUI = Path(__file__).resolve().parent
PKG = QUI / "pkg"

# Configurazioni fissate per gfx1151 (nome kernel -> kwargs Config + num_warps/num_stages): TUTTE a scratch zero
# (compila_offline.py, 27/9). Regola trovata: fp32 = prodotto FMA che tiene gli operandi interi nei registri, quindi
# 16 warp (512 thread: meno righe/colonne per thread, 256 VGPR ancora disponibili; 32 warp scende a 192 VGPR) e
# blocchi da 16. Un kernel assente usa la regola di ripiego _scegli().
CONFIG_GFX1151: dict[str, dict] = {
    "chunk_fwd_kernel_o": {"BK": 16, "BV": 16, "num_warps": 16, "num_stages": 1},
    "chunk_gated_delta_rule_fwd_kernel_h_blockdim64": {"BV": 16, "num_warps": 16, "num_stages": 1},
    "chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64": {"BV": 16, "num_warps": 16, "num_stages": 1},
    "chunk_gated_delta_rule_fwd_kkt_solve_kernel": {"BK": 16, "num_warps": 16, "num_stages": 1},
    "chunk_bwd_kernel_dqkwg": {"num_warps": 16, "num_stages": 1},
    "chunk_bwd_kernel_dv_local": {"num_warps": 16, "num_stages": 1},
    "recompute_w_u_fwd_kernel": {"num_warps": 16, "num_stages": 1},
}
# blocchi che fla sceglie nei wrapper (non nelle Config): letti a ogni chiamata dalla copia patchata (patch_fla.py)
AMBIENTE_GFX1151 = {"FLA_GFX1151_BLOCCO_WU": "16", "FLA_GFX1151_BLOCCO_WY": "16", "FLA_GFX1151_BLOCCO_O": "16",
                    "FLA_CACHE_MODE": "disabled"}


def carica_fla(pkg: Path = PKG) -> Callable:
    """Importa la copia di fla da pkg/ e ne restituisce chunk_gated_delta_rule."""
    gia = sys.modules.get("fla")
    if gia is not None and pkg.resolve() not in Path(gia.__file__).resolve().parents:
        # Il decoratore di transformers fa importlib.import_module("fla"): nel venv-catq il PACCHETTO fla si importa
        # (solo i sottomoduli vogliono einops), la risoluzione di chunk_gated_delta_rule fallisce e transformers lega
        # il ripiego — ma `fla` del venv resta in sys.modules (misura 27/9, collaudo di abilita_sicuro). Nessuno lo
        # usa: si toglie (con eventuali sottomoduli) e si carica la copia. Rifiuto se qualche sottomodulo GDN del venv
        # fosse davvero carico (vorrebbe dire einops presente nel venv).
        vecchi = [n for n in sys.modules if n == "fla" or n.startswith("fla.")]
        if any(n.startswith("fla.ops.gated_delta_rule") for n in vecchi):
            raise RuntimeError(f"fla del venv ({gia.__file__}) ha il GDN caricato: einops nel venv? non tocco nulla")
        for n in vecchi:
            del sys.modules[n]
        gia = None
    if gia is None:
        sys.modules.pop("fla", None)               # anche un eventuale blocco None messo da chi voleva il ripiego
        sys.path.insert(0, str(pkg))
        try:
            import fla  # noqa: F401
            import einops  # noqa: F401
        finally:
            sys.path.remove(str(pkg))
        f = Path(sys.modules["fla"].__file__).resolve()
        if pkg.resolve() not in f.parents:
            raise RuntimeError(f"importato fla da {f}, atteso sotto {pkg}")
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    return chunk_gated_delta_rule


def _autotuner() -> list:
    from triton.runtime.autotuner import Autotuner
    return [o for o in gc.get_objects() if isinstance(o, Autotuner)         # base_fn = funzione python originale
            and getattr(getattr(o, "base_fn", None), "__module__", "").startswith("fla.")]


def _nome(at) -> str:
    return getattr(at, "base_fn", at.fn).__name__ if hasattr(at, "base_fn") else at.fn.__name__


def _scegli(configs: list):
    """Regola di ripiego: num_warps piu' vicino a 4, blocchi piu' piccoli, num_stages=1 (niente pipeline software:
    con num_stages=2 chunk_fwd_kernel_o in fp32 chiede 98304 B di LDS contro i 65536 di gfx1151 — misura 27/9)."""
    import triton

    def chiave(c):
        blocchi = sum(v for v in c.kwargs.values() if isinstance(v, int))
        return (abs(c.num_warps - 4), blocchi, c.num_stages)
    c = min(configs, key=chiave)
    return triton.Config(dict(c.kwargs), num_warps=c.num_warps, num_stages=1)


def fissa_config(tabella: Optional[dict] = None, verbose: bool = True) -> dict[str, str]:
    """Riduce l'autotune di ogni kernel fla a UNA config (Autotuner con una sola config non misura)."""
    import triton
    tabella = CONFIG_GFX1151 if tabella is None else tabella
    scelte = {}
    for at in _autotuner():
        nome = _nome(at)
        if nome in tabella:
            t = dict(tabella[nome])
            nw, ns = t.pop("num_warps"), t.pop("num_stages", 1)
            c = triton.Config(t, num_warps=nw, num_stages=ns)
        else:
            c = _scegli(at.configs)
        at.configs = [c]
        at.cache.clear()
        scelte[nome] = str(c)
    if verbose:
        for n, c in sorted(scelte.items()):
            print(f"  config {n}: {c}", flush=True)
    return scelte


class ScratchVietato(RuntimeError):
    """Un kernel fla compilato usa memoria scratch (spill di registri): non va lanciato su gfx1151."""


RISORSE: dict[str, dict] = {}        # nome kernel -> {vgpr, scratch, lds} dell'ultima variante caricata


def _risorse_amdgcn(metadata_group: dict) -> tuple[str, dict]:
    import re
    for nome_file, percorso in metadata_group.items():
        if nome_file.endswith(".amdgcn"):
            testo = Path(percorso).read_text()
            def campo(c):
                m = re.search(rf"\.{c}:\s+(\d+)", testo)
                return int(m.group(1)) if m else -1
            return nome_file[:-7], {"vgpr": campo("vgpr_count"), "scratch": campo("private_segment_fixed_size"),
                                    "lds": campo("group_segment_fixed_size")}
    return "?", {"vgpr": -1, "scratch": -1, "lds": -1}


def installa_guardia_scratch(soglia_byte: int = 0) -> None:
    """Prima di CARICARE un kernel (dunque prima del primo lancio) legge il suo .amdgcn: se usa scratch oltre soglia
    solleva ScratchVietato. Misura 27/9: tutte le varianti fp32 (ieee) dei kernel GDN fanno spill (1,1-2,6 KB di
    scratch per corsia) e il page fault cade su chunk_gated_delta_rule_fwd_kernel_h_blockdim64 (1588 B di scratch),
    mentre le varianti bf16 (WMMA) hanno scratch 0 e non vanno mai in fault. Si solleva nel kernel_load_START_hook:
    il modulo resta non caricato, quindi ogni tentativo successivo ripassa di qui (launch_metadata chiama
    _init_handles prima di kernel.run) e nessun lancio parte."""
    import triton

    marca = str(PKG / "fla")

    def di_fla(metadata_group: dict) -> bool:
        # SOLO i kernel della copia di fla: la corsa ha i suoi kernel Triton (motore dello studente) e la guardia
        # non deve mai toccarli. Il .ttir/.source porta i percorsi dei sorgenti (loc).
        for nome_file, percorso in metadata_group.items():
            if nome_file.endswith((".source", ".ttir")):
                try:
                    return marca in Path(percorso).read_text(errors="ignore")[:200000]
                except OSError:
                    return False
        return False

    def guardia(module, function, name, metadata_group, hash_):
        if not di_fla(metadata_group):
            return
        nome, r = _risorse_amdgcn(metadata_group)
        RISORSE[nome] = r
        if r["scratch"] > soglia_byte or r["scratch"] < 0:
            raise ScratchVietato(f"{nome}: scratch {r['scratch']} B, vgpr {r['vgpr']}, lds {r['lds']}")

    triton.knobs.runtime.kernel_load_start_hook.add(guardia)


def risorse_compilate() -> list[tuple]:
    """(kernel, num_warps, num_stages, registri, spill, LDS) di ogni kernel fla compilato — spill = memoria scratch."""
    righe = []
    for at in _autotuner():
        fn = at.fn
        while hasattr(fn, "fn") and not hasattr(fn, "device_caches"):
            fn = fn.fn
        for dev, cache in getattr(fn, "device_caches", {}).items():
            for k in cache[0].values():
                try:
                    k._init_handles()
                    righe.append((k.name, k.metadata.num_warps, k.metadata.num_stages, k.n_regs, k.n_spills,
                                  k.metadata.shared))
                except Exception as e:                  # noqa: BLE001 — diagnostica, mai bloccante
                    righe.append((getattr(k, "name", "?"), None, None, None, None, repr(e)[:60]))
    return righe


def prepare_wy_repr_bwd_torch(k, v, beta, A, dw, du, g=None, cu_seqlens=None, chunk_indices=None):
    """Stessa matematica di fla prepare_wy_repr_bwd_kernel, in torch fp32 a lotti (un blocco = una matrice 64x64).
    Perche': il kernel Triton ha due prodotti registro-registro 64x64x64 (A^T dA A^T) che in FMA fp32 fanno spill su
    gfx1151 a ogni numero di warp (16: 268 B anche con la contrazione spezzata; 32: LDS 80 KB > 64 KB). Blocchi
    indipendenti -> pochi matmul a lotti (rocBLAS fp32 pieno), nessuna ricorsione."""
    import torch
    assert cu_seqlens is None and g is not None
    B, T, H, K = k.shape
    HV, V, BT = v.shape[2], v.shape[-1], A.shape[-1]
    assert H == HV and T % BT == 0
    NC = T // BT

    def blocchi(x):                      # [B, T, HV, ...] -> [B, NC, HV, BT, ...]
        x = x.reshape(B, NC, BT, HV, *x.shape[3:])
        return x.transpose(2, 3).float()
    kc, vc, dwc, duc, Ac = blocchi(k), blocchi(v), blocchi(dw), blocchi(du), blocchi(A)
    bc, gc = blocchi(beta), blocchi(g)                        # [B, NC, HV, BT]
    At = Ac.transpose(-1, -2)                                 # b_A del kernel = A^T
    gexp = torch.exp2(gc)
    kbg = kc * (bc * gexp)[..., None]
    dA = dwc @ kbg.transpose(-1, -2)
    dkbg = At @ dwc
    dk = dkbg * (gexp * bc)[..., None]
    db = (dkbg * kc * gexp[..., None]).sum(-1)
    dg = (dkbg * kbg).sum(-1)
    vb = vc * bc[..., None]
    dA = dA + duc @ vb.transpose(-1, -2)
    dvb = At @ duc
    dv = dvb * bc[..., None]
    db = db + (dvb * vc).sum(-1)
    i = torch.arange(BT, device=k.device)
    m = i[:, None] > i[None, :]
    dA = torch.where(m, dA, 0.0)
    dA = At @ (dA @ At)
    dA = dA * torch.exp2(gc[..., :, None] - gc[..., None, :])
    dA = torch.where(m, -dA, 0.0)
    A2 = kc @ kc.transpose(-1, -2)
    dkb = dA @ kc
    db = db + (dkb * kc).sum(-1)
    kb = kc * bc[..., None]
    dk = dk + dkb * bc[..., None] + dA.transpose(-1, -2) @ kb
    AdA = dA * (A2 * bc[..., :, None])
    dg = dg + AdA.sum(-1) - AdA.sum(-2)

    def righe(x):                        # [B, NC, HV, BT, ...] -> [B, T, HV, ...]
        return x.transpose(2, 3).reshape(B, T, HV, *x.shape[4:]).contiguous()
    return (righe(dk).to(k.dtype), righe(dv).to(v.dtype), righe(db).to(beta.dtype), righe(dg).to(g.dtype))


def cumsum_blocchi_fp64(g, chunk_size, reverse=False, scale=None, cu_seqlens=None, head_first=False,
                        output_dtype=None, chunk_indices=None, **kw):
    """chunk_local_cumsum di fla (scalare, [B, T, H], niente cu_seqlens) calcolata in fp64 ed arrotondata UNA volta.
    Perche' (27/9, prova in corso): il kernel fa tl.cumsum (scansione ad albero) in fp32 e poi moltiplica per
    RCP_LN2 in fp32, due arrotondamenti indipendenti per token su valori |G'| fino a ~2000 nei decadimenti forti; qui
    la somma e' esatta e la costante e' 1/ln2 esatto (RCP_LN2 = 1.4426950216 ha errore relativo 1,3e-8).
    ⛔ REFUTATA come cura (prova_abilita, strato vero, 27/9 09:25): i gradienti di A_log / dt_bias / in_proj_a restano
    5,9x / 5,9x / 4,7x il ripiego fp32 contro fp64 (identici senza): l'errore sta nel dg dei kernel (chunk_bwd_dqkwg),
    non nella somma cumulata. Resta opzionale (FLA_GFX1151_CUMSUM64=1), spenta di default."""
    import math
    import torch
    assert cu_seqlens is None and not head_first and g.dim() == 3, "solo il caso della corsa"
    B, T, H = g.shape
    x = g.double().reshape(B, T // chunk_size, chunk_size, H)
    y = x.flip(2).cumsum(2).flip(2) if reverse else x.cumsum(2)
    if scale is not None:
        y = y * (1.0 / math.log(2.0) if abs(scale - 1.4426950216) < 1e-9 else scale)
    return y.reshape(B, T, H).to(output_dtype or torch.float32)


def installa_sostituzioni() -> None:
    """prepare_wy_repr_bwd di fla -> versione torch (vedi sopra), nel modulo che la chiama; con
    FLA_GFX1151_CUMSUM64=1 anche chunk_local_cumsum -> cumsum_blocchi_fp64."""
    import fla.ops.gated_delta_rule.chunk as C
    C.prepare_wy_repr_bwd = prepare_wy_repr_bwd_torch
    if os.environ.get("FLA_GFX1151_CUMSUM64") == "1":
        C.chunk_local_cumsum = cumsum_blocchi_fp64


def configura(tabella: Optional[dict] = None) -> Callable:
    """Ambiente + copia di fla + config fissate + sostituzioni + guardia scratch: la ricetta gfx1151 intera."""
    for k_, v_ in AMBIENTE_GFX1151.items():
        os.environ.setdefault(k_, v_)
    fla_gdn = carica_fla()
    fissa_config(tabella=tabella, verbose=False)
    installa_sostituzioni()
    installa_guardia_scratch()
    return fla_gdn


def _implementazione_legata(f) -> Optional[str]:
    """Modulo della funzione che il decoratore di transformers ha legato dentro f (None se non si trova)."""
    for cella in getattr(f, "__closure__", None) or ():
        try:
            v = cella.cell_contents
        except ValueError:
            continue
        if callable(v) and getattr(v, "__name__", "") in ("chunk_gated_delta_rule", "torch_chunk_gated_delta_rule"):
            return getattr(v, "__module__", None)
    return None


def ripiego_fp64(M) -> Callable:
    """Il ripiego di transformers in fp64 VERO. Il ripiego converte tutto a float32 dentro (`.to(torch.float32)`,
    `.float()`): chiamato con tensori fp64 calcola in fp32 (solo l2norm resta fp64). Misura 27/9: "ripiego fp64"
    contro la stessa matematica in fp64 differisce di 6,4e-7 (= l'errore fp32 del ripiego), quindi NON e' un
    riferimento. Qui lo stesso sorgente, con float32 -> float64."""
    import inspect
    f = M.torch_chunk_gated_delta_rule
    while hasattr(f, "__wrapped__"):
        f = f.__wrapped__
    sorgente = inspect.getsource(f)
    sorgente = sorgente[sorgente.index("def "):]                     # via il decoratore
    nuovo = sorgente.replace("torch.float32", "torch.float64").replace(".float()", ".double()")
    assert nuovo != sorgente and "float32" not in nuovo
    spazio = dict(vars(M))
    exec(compile(nuovo, f"<ripiego_fp64 da {f.__code__.co_filename}>", "exec"), spazio)
    return spazio[f.__name__]


def _confronto(fla_gdn, ripiego, ripiego64, B=1, T=512, H=32, D=128, seme=11) -> dict:
    """fwd+bwd di fla e ripiego (fp32) contro il ripiego in fp64: rapporto degli errori relativi per 6 grandezze."""
    import torch
    gen = torch.Generator(device="cpu").manual_seed(seme)
    q, k, v, do = (torch.randn(B, T, H, D, generator=gen) for _ in range(4))
    gg = -torch.nn.functional.softplus(torch.randn(B, T, H, generator=gen))
    be = torch.sigmoid(torch.randn(B, T, H, generator=gen))

    def passo(f, dt):
        xs = [t.to("cuda", dt).requires_grad_(True) for t in (q, k, v, gg, be)]
        o, _ = f(xs[0], xs[1], xs[2], g=xs[3], beta=xs[4], initial_state=None, output_final_state=False,
                 use_qk_l2norm_in_kernel=True)
        o.backward(do.to("cuda", dt))
        return [o.detach().double()] + [x.grad.detach().double() for x in xs]

    uf, ut, u64 = passo(fla_gdn, torch.float32), passo(ripiego, torch.float32), passo(ripiego64, torch.float64)
    esito = {}
    for n, a, b, c in zip(("avanti", "dq", "dk", "dv", "dg", "dbeta"), uf, ut, u64):
        ef = float((a - c).norm() / c.norm())
        et = float((b - c).norm() / c.norm())
        esito[n] = ef / et if et > 0 else float("inf")
    esito["finito"] = all(bool(torch.isfinite(t).all()) for t in uf)
    return esito


def abilita_sicuro(M, f32_dot: str = "ieee", soglia: float = 2.0, controlla: bool = True,
                   dims_corsa: tuple = (2, 4096, 32, 128)) -> dict:
    """UNICO ingresso per la corsa: GDN su fla (copia pkg/) solo se tutto torna, altrimenti il ripiego resta.

    Da chiamare DOPO l'import di modeling_qwen3_5_moe (M) e di ogni altro modeling di transformers.
    1. rifiuta se il decoratore di transformers ha gia' legato un fla (fla importabile per errore);
    2. ambiente: cache Triton propria, TRITON_F32_DEFAULT=f32_dot (vale per ogni kernel Triton JIT del processo:
       la corsa non ne ha altri), autotune di fla spento (config fissate);
    3. guardia scratch (nessun kernel con spill viene lanciato) + controllo fwd+bwd contro fp64: ogni rapporto
       errore(fla)/errore(ripiego fp32) <= soglia, tutto finito;
    4. sostituisce M.torch_chunk_gated_delta_rule con un involucro: fla nel caso provato (fp32, niente cu_seqlens,
       niente stato, q/k l2 nel kernel), ripiego altrimenti; un'eccezione in avanti spegne fla per sempre (ripiego).
    Restituisce un resoconto da stampare nel giornale della corsa."""
    import torch
    resoconto: dict = {"attivo": False, "f32_dot": f32_dot}
    legata = _implementazione_legata(M.torch_chunk_gated_delta_rule)
    resoconto["legata_prima"] = legata
    if legata is not None and legata.startswith("fla"):
        resoconto["motivo"] = "transformers ha gia' legato fla: fla era importabile all'import del modeling"
        return resoconto
    if f32_dot != "ieee":
        os.environ["TRITON_F32_DEFAULT"] = f32_dot
    fla_gdn = configura()
    ripiego = M.torch_chunk_gated_delta_rule
    if controlla:
        try:
            esito = _confronto(fla_gdn, ripiego, ripiego_fp64(M))
        except Exception as e:                           # noqa: BLE001 — qualunque guasto = resta il ripiego
            resoconto["motivo"] = f"controllo fallito: {type(e).__name__}: {e}"[:300]
            return resoconto
        resoconto["controllo"] = esito
        peggio = max(v for k, v in esito.items() if k != "finito")
        if not esito["finito"] or peggio > soglia:
            resoconto["motivo"] = f"fuori dal rumore fp32 (rapporto peggiore {peggio:.2f})"
            return resoconto
    try:                                                 # carica ORA ogni variante che la corsa usera' (guardia
        B, T, H, D = dims_corsa                          # scratch compresa): mai un caricamento dentro un backward
        xs = [torch.randn(B, T, H, D, device="cuda", requires_grad=True) for _ in range(3)]
        gg = (-torch.nn.functional.softplus(torch.randn(B, T, H, device="cuda"))).requires_grad_(True)
        be = torch.sigmoid(torch.randn(B, T, H, device="cuda")).requires_grad_(True)
        o, _ = fla_gdn(xs[0], xs[1], xs[2], g=gg, beta=be, use_qk_l2norm_in_kernel=True)
        o.sum().backward()
        torch.cuda.synchronize()
        del xs, gg, be, o
    except Exception as e:                               # noqa: BLE001
        resoconto["motivo"] = f"riscaldamento alle dimensioni della corsa fallito: {type(e).__name__}: {e}"[:300]
        return resoconto
    stato = {"fla": True, "chiamate_fla": 0, "chiamate_ripiego": 0, "guasto": None}

    def gdn(query, key, value, g, beta, chunk_size=64, initial_state=None, output_final_state=False,
            use_qk_l2norm_in_kernel=False, cu_seqlens=None, **kwargs):
        caso_provato = (stato["fla"] and query.dtype == torch.float32 and cu_seqlens is None
                        and initial_state is None and not output_final_state and use_qk_l2norm_in_kernel
                        and chunk_size == 64 and query.is_cuda and query.shape[1] % 64 == 0)
        if caso_provato:
            try:
                o, _ = fla_gdn(query, key, value, g=g, beta=beta, initial_state=None, output_final_state=False,
                               use_qk_l2norm_in_kernel=True)
                stato["chiamate_fla"] += 1
                return o, None
            except Exception as e:                       # noqa: BLE001 — si spegne fla, la corsa continua
                stato["fla"], stato["guasto"] = False, f"{type(e).__name__}: {e}"[:300]
        stato["chiamate_ripiego"] += 1
        return ripiego(query, key, value, g=g, beta=beta, chunk_size=chunk_size, initial_state=initial_state,
                       output_final_state=output_final_state, use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                       cu_seqlens=cu_seqlens, **kwargs)

    gdn.stato = stato
    gdn.ripiego = ripiego
    M.torch_chunk_gated_delta_rule = gdn
    resoconto.update({"attivo": True, "impronta_copia": (PKG / "IMPRONTA").read_text().strip(),
                      "risorse": dict(RISORSE)})
    return resoconto
