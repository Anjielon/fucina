#!/usr/bin/env python3
"""collaudo_cardan_gruppi.py (21/10) - CARDAN_GRUPPI=1 (_GruppiLowRank, grouped GEMM) vs the loop (_GroupedLowRank).
CPU: run with TRITON_INTERPRET=1 (P1 fp64, P2 fp32, gradcheck, P3 C=0). GPU: only when the KFD process count allows it
(checked here: refuses if /sys/class/kfd/kfd/proc has more than --max-kfd entries, this process excluded)."""
import argparse, json, os, sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cardan_kd_moduli as CK  # noqa: E402

if os.environ.get("TRITON_INTERPRET") == "1":
    # numpy >= 2.5 refuses int() on a 1-element 1-d array; the Triton 3.6 interpreter keeps scalars that way, so a
    # runtime-bounded range() inside a kernel fails ONLY in the interpreter. Test-side patch, no effect on the GPU path.
    import triton.runtime.interpreter as _TI
    _orig = _TI._patch_lang_tensor

    def _patch(tensor, scope):
        _orig(tensor, scope)
        scope.set_attr(tensor, "__index__", lambda self: int(self.handle.data.reshape(-1)[0]))
    _TI._patch_lang_tensor = _patch


def piano_sintetico(E, S, seme, zipf=None, vuoti=0):
    g = torch.Generator().manual_seed(seme)
    if zipf:
        p = 1.0 / torch.arange(1, E + 1, dtype=torch.float64) ** zipf
    else:
        p = torch.ones(E, dtype=torch.float64)
    if vuoti:
        p[torch.randperm(E, generator=g)[:vuoti]] = 0
    idx = torch.multinomial(p, S, replacement=True, generator=g)
    cnt = torch.bincount(idx, minlength=E)
    offs = torch.zeros(E + 1, dtype=torch.int64)
    offs[1:] = torch.cumsum(cnt, 0)
    return offs


def dati(E, R, r, S, dt, dev, seme):
    g = torch.Generator().manual_seed(seme + 1)
    f = lambda *s: torch.randn(*s, generator=g, dtype=torch.float64).to(dt).to(dev)
    return f(S, r), f(S, r), f(E, R, r) * 0.1, f(E, R, r) * 0.1, f(S, R), f(S, R)


class _Ctx:
    """Stand-in for the autograd ctx: on CPU the autograd ENGINE opens a KFD context (torch-ROCm starts its device
    threads), so the big CPU comparisons call forward/backward directly and never touch the GPU driver."""
    needs_input_grad = (True, True, True, True, False, False)

    def save_for_backward(self, *t):
        self.saved_tensors = t


def esegui_senza_motore(classe, zg, zu, Cg, Cu, gg, gu, offs, eid):
    c = _Ctx()
    with torch.no_grad():
        if classe == "ciclo":
            yg, yu = CK._GroupedLowRank.forward(c, zg, zu, Cg, Cu, offs.tolist())
            d = CK._GroupedLowRank.backward(c, gg, gu)
        else:
            yg, yu = CK._GruppiLowRank.forward(c, zg, zu, Cg, Cu, offs, eid)
            d = CK._GruppiLowRank.backward(c, gg, gu)
    return {"yg": yg, "yu": yu, "dzg": d[0], "dzu": d[1], "dCg": d[2], "dCu": d[3]}


def esegui(classe, zg, zu, Cg, Cu, gg, gu, offs, eid):
    if zg.device.type == "cpu" and os.environ.get("COLLAUDO_SENZA_MOTORE", "1") == "1":
        return esegui_senza_motore(classe, zg, zu, Cg, Cu, gg, gu, offs, eid)
    zg, zu, Cg, Cu = [t.detach().clone().requires_grad_(True) for t in (zg, zu, Cg, Cu)]
    if classe == "ciclo":
        yg, yu = CK._GroupedLowRank.apply(zg, zu, Cg, Cu, offs.tolist())
    else:
        yg, yu = CK._GruppiLowRank.apply(zg, zu, Cg, Cu, offs, eid)
    torch.autograd.backward((yg, yu), (gg, gu))
    return {"yg": yg.detach(), "yu": yu.detach(), "dzg": zg.grad, "dzu": zu.grad, "dCg": Cg.grad, "dCu": Cu.grad}


def scarti(a, b):
    out = {}
    for k in a:
        d = (a[k].double() - b[k].double()).abs()
        m = b[k].double().abs().max().item()
        out[k] = {"abs": d.max().item(), "rel_max": d.max().item() / max(m, 1e-300),
                  "al_bit": bool(torch.equal(a[k], b[k])), "frac_bit": (a[k] == b[k]).double().mean().item()}
    return out


def kfd_n():
    try:
        return len(os.listdir("/sys/class/kfd/kfd/proc"))
    except OSError:
        return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dispositivo", default="cpu")
    ap.add_argument("--S", default="16384,65536")
    ap.add_argument("--E", type=int, default=256)
    ap.add_argument("--R", type=int, default=512)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--ripeti", type=int, default=20)
    ap.add_argument("--max-kfd", type=int, default=2)
    ap.add_argument("--uscita", default="")
    a = ap.parse_args()
    dev = torch.device(a.dispositivo)
    ris = {"dispositivo": a.dispositivo, "E": a.E, "R": a.R, "r": a.r, "prove": []}
    if dev.type == "cpu":
        assert os.environ.get("TRITON_INTERPRET") == "1", "CPU: TRITON_INTERPRET=1"
    if dev.type == "cpu":
        if os.environ.get("COLLAUDO_GRADCHECK", "0") == "1":   # autograd engine: opens a KFD context
            # gradcheck fp64, small problem with empty experts
            offs = piano_sintetico(4, 21, 3, vuoti=1)
            eid = CK.tabella_riquadri(offs, 21, CK._GR_BM)
            z = [t.requires_grad_(True) for t in dati(4, 8, 3, 21, torch.float64, dev, 3)[:4]]
            ok = torch.autograd.gradcheck(lambda zg, zu, Cg, Cu: CK._GruppiLowRank.apply(zg, zu, Cg, Cu, offs, eid),
                                          tuple(z), eps=1e-6, atol=1e-8, fast_mode=True)
            ris["gradcheck_fp64"] = bool(ok)
            print("gradcheck fp64:", ok, flush=True)
        # P3: C = 0 -> y exactly 0 and identical in both paths
        zg, zu, Cg, Cu, gg, gu = dati(a.E, a.R, a.r, 4096, torch.float32, dev, 5)
        offs = piano_sintetico(a.E, 4096, 5, zipf=1.0)
        eid = CK.tabella_riquadri(offs, 4096, CK._GR_BM)
        n = esegui("gruppi", zg, zu, Cg * 0, Cu * 0, gg, gu, offs, eid)
        o = esegui("ciclo", zg, zu, Cg * 0, Cu * 0, gg, gu, offs, eid)
        ris["P3_C0"] = {"y_zero": bool((n["yg"] == 0).all() and (n["yu"] == 0).all()),
                        "y_al_bit": bool(torch.equal(n["yg"], o["yg"]) and torch.equal(n["yu"], o["yu"]))}
        print("P3:", ris["P3_C0"], flush=True)
    for S in [int(s) for s in a.S.split(",")]:
        for nome, kw in (("uniforme", {}), ("zipf2_vuoti", {"zipf": 2.0, "vuoti": max(1, a.E // 16)})):
            if dev.type == "cuda" and kfd_n() - 1 > a.max_kfd:
                print(f"GPU occupata ({kfd_n()} processi KFD): fermo", flush=True)
                ris["interrotto_kfd"] = kfd_n()
                break
            offs_c = piano_sintetico(a.E, S, 11 + S, **kw)
            offs = offs_c.to(dev)
            eid = CK.tabella_riquadri(offs, S, CK._GR_BM)
            vuoti = int(((offs_c[1:] - offs_c[:-1]) == 0).sum())
            voce = {"S": S, "piano": nome, "esperti_vuoti": vuoti}
            t32 = dati(a.E, a.R, a.r, S, torch.float32, dev, 7)
            if dev.type == "cpu":   # P1 fp64 reference (loop in fp64) vs grouped fp64 / P2 fp32
                t64 = [t.double() for t in t32]
                rif = esegui("ciclo", *t64, offs, eid)
                voce["P1_fp64"] = scarti(esegui("gruppi", *t64, offs, eid), rif)
                o32 = esegui("ciclo", *t32, offs, eid)
                n32 = esegui("gruppi", *t32, offs, eid)
                voce["P2_fp32_gruppi_vs_rif64"] = scarti(n32, rif)
                voce["P2_fp32_ciclo_vs_rif64"] = scarti(o32, rif)
                voce["P2_fp32_gruppi_vs_ciclo"] = scarti(n32, o32)
                vu = (offs_c[1:] - offs_c[:-1]) == 0
                voce["dC_vuoti_zero_esatto"] = bool((n32["dCg"][vu] == 0).all() and (n32["dCu"][vu] == 0).all())
            else:
                rif = {k: v for k, v in esegui("ciclo", *[t.double().cpu() for t in t32], offs_c,
                                                 None).items()}
                n32 = {k: v.cpu() for k, v in esegui("gruppi", *t32, offs, eid).items()}
                o32 = {k: v.cpu() for k, v in esegui("ciclo", *t32, offs, eid).items()}
                voce["gruppi_vs_ciclo_fp32"] = scarti(n32, o32)
                voce["gruppi_vs_rif64"] = scarti(n32, rif)
                vu = (offs_c[1:] - offs_c[:-1]) == 0
                voce["dC_vuoti_zero_esatto"] = bool((n32["dCg"][vu] == 0).all() and (n32["dCu"][vu] == 0).all())
                voce["ciclo_vs_rif64"] = scarti(o32, rif)
                tempi = {}
                for cl in ("ciclo", "gruppi", "ciclo", "gruppi"):
                    esegui(cl, *t32, offs, eid)
                    torch.cuda.synchronize()
                    ts = []
                    for _ in range(a.ripeti):
                        t0 = time.perf_counter()
                        esegui(cl, *t32, offs, eid)
                        torch.cuda.synchronize()
                        ts.append(time.perf_counter() - t0)
                    ts.sort()
                    tempi[cl] = {"mediana_ms": 1e3 * ts[len(ts) // 2], "min_ms": 1e3 * ts[0]}
                voce["tempi_avanti_indietro"] = tempi
                voce["fattore"] = tempi["ciclo"]["mediana_ms"] / tempi["gruppi"]["mediana_ms"]
            ris["prove"].append(voce)
            print(f"S={S} {nome} vuoti={vuoti}", flush=True)
            for kk, vv in voce.items():
                if isinstance(vv, dict) and "yg" in vv:
                    print("  ", kk, " ".join(f"{t}:abs={x['abs']:.2e},rel={x['rel_max']:.2e},bit={x['frac_bit']:.3f}"
                                             for t, x in vv.items()), flush=True)
                elif kk not in ("S", "piano", "esperti_vuoti"):
                    print("  ", kk, vv, flush=True)
    if a.uscita:
        json.dump(ris, open(a.uscita, "w"), indent=1)


if __name__ == "__main__":
    main()
