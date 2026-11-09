#!/usr/bin/env python3
"""profila_strati_gpu.py — prova PICCOLA (1-2 strati) del passo FUSO di cardan_congiunto.py.

Stesso studente (e2e_studente, stessi E2E_* della corsa congiunto S7T550), stessi rami installati (CARDAN esperti
C/D, residui densi continui), stesso ingresso FUSO (riga paper wt103 + pacco KL4 come segmenti). Misura avanti +
indietro con torch.profiler e confronta il miscelatore per segmenti ORIGINALE con le varianti di
cardan_segmenti_gpu.py (uscita e gradienti: identici al bit o scarto dichiarato).

Uso (nice 19, solo con RAM sana):
  E2E_KERNEL_NUOVI=1 E2E_GDN=veloce E2E_GDN_MODULO=gdn_veloce_corsa_v5 E2E_MOTORE=triton2 E2E_CONV_DIRETTA=1 \
  E2E_SENZA_COESISTENZA=1 E2E_DENSI_PIEGATI=x2 E2E_PIEGA_GEMM=triton E2E_DENSI_SSM_AB=1 \
  python profila_strati_gpu.py --gguf G --base-a1 B --strati 0,3 --out CARTELLA_NUOVA
"""
from __future__ import annotations

import argparse
import json
import os
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"  # priorita' 0 (21/10): ogni worker inductor apre /dev/kfd e prende una coda SDMA
import sys
import time
from pathlib import Path

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))


def host():
    m = {l.split(":")[0]: int(l.split()[1]) for l in open("/proc/meminfo")}
    def swin():
        for l in open("/proc/vmstat"):
            if l.startswith("pswpin "):
                return int(l.split()[1])
    a = swin(); time.sleep(1.0); b = swin()
    return m["MemAvailable"] / 2 ** 20, (b - a) * 4096 / 2 ** 20


def guardia(soglia_gib=8.0, attesa_s=600):
    fine = time.time() + attesa_s
    ma, si = host()
    while (ma < soglia_gib or si > 20) and time.time() < fine:
        time.sleep(10); ma, si = host()
    if ma < soglia_gib or si > 20:
        raise SystemExit(f"RAM non sana: MemAvailable {ma:.1f} GiB, swap-in {si:.1f} MB/s: non lancio")
    return ma, si


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True); ap.add_argument("--base-a1", default=None)
    ap.add_argument("--strati", default="0,3"); ap.add_argument("--out", required=True)
    ap.add_argument("--rango-esperti", type=int, default=16); ap.add_argument("--rango-densi", type=int, default=16)
    ap.add_argument("--famiglie", default="attn,gdn"); ap.add_argument("--passo", type=int, default=151)
    ap.add_argument("--ripetizioni", type=int, default=3)
    ap.add_argument("--varianti", default="originale,lotto", help="originale,lotto,riempito (cardan_segmenti_gpu)")
    ap.add_argument("--profilo", action="store_true")
    ap.add_argument("--seme", type=int, default=20261007)
    ap.add_argument("--device", default="cuda", help="cpu: verifica della logica (motore torch), niente HIP")
    a = ap.parse_args(argv)
    os.nice(max(0, 19 - os.getpriority(os.PRIO_PROCESS, 0)))
    out = Path(a.out)
    guardia()
    out.mkdir(parents=True, exist_ok=False)
    reg = open(out / "misure.jsonl", "a")

    def scrivi(d):
        print(json.dumps(d), flush=True); reg.write(json.dumps(d) + "\n"); reg.flush()

    scrivi({"evento": "avvio", "host": guardia(), "argv": sys.argv})
    import cardan_kd_paper as KDP  # noqa: F401  percorsi e E2E_KERNEL_NUOVI come la corsa
    import numpy as np
    import torch
    import e2e_studente as S
    import e2e_allena as A
    import residui_ternari as RT
    import cardan_kd_moduli as CK
    import cardan_congiunto as CG
    sys.path.insert(0, "@@ROOT:models@@/build/leve/raffina")
    import gradienti_densi as GD
    import types
    if os.environ.get("FORGIA_RICOMPILA_LIMITE"):
        import torch._dynamo
        torch._dynamo.config.recompile_limit = int(os.environ["FORGIA_RICOMPILA_LIMITE"])
    A.scegli_gdn(print)
    torch.manual_seed(a.seme)
    dev = torch.device(a.device)
    sinc = torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)
    strati = [int(x) for x in a.strati.split(",")]
    t0 = time.time()
    st = S.Studente(gguf=Path(a.gguf), strati=strati, ricetta="B3_ind2", motore_nome=os.environ.get("E2E_MOTORE", "triton2"),
                    device=a.device, log=print, densi_ternari="tutte")
    for p in st.parametri():
        p.requires_grad_(False)
    resE = CK.installa(st, strati, a.rango_esperti, Path(a.base_a1) if a.base_a1 else None, dev)
    cfgD = RT.ConfigResidui(rango=a.rango_densi, piani=1, strati=strati, famiglie=RT.famiglie_da(a.famiglie),
                            seme=a.seme, continuo=True)
    resD = RT.installa(st, cfgD, device=dev)
    st.strati_allenati = sorted(set(strati) | set(RT.strati_di(resD)))
    g = torch.Generator(device="cpu").manual_seed(a.seme)
    with torch.no_grad():                       # parametri NON nulli: il gradiente attraversa entrambi i rami
        for p in CK.parametri(resE, "C") + [x for n in sorted(resD) for x in (resD[n].B,)]:
            p.copy_((torch.randn(p.shape, generator=g) * 1e-3).to(p))
    pE = CK.parametri(resE, "C") + CK.parametri(resE, "D")
    pD = RT.parametri(resD)
    for p in pE + pD:
        p.requires_grad_(True)
    scrivi({"evento": "studente", "s": round(time.time() - t0, 1), "strati": strati,
            "parametri": sum(p.numel() for p in pE + pD)})
    src_w = A.SorgenteCache(CG.TEACHER_WT); car_w = A.Caricatore(src_w, dev)
    ord_w = src_w.righe_previste("train")[:]
    import random
    random.Random(a.seme).shuffle(ord_w)
    ns = dict(calib_maestro=f"{CG.CALIB}/mclp_maestro.log", calib_jsonl=f"{CG.CALIB}/calib_mmlu.jsonl", seme=7,
              calib_pacco=4096)
    pacchi4, _ = GD.pacchi_kl4(types.SimpleNamespace(calib_n=896, **ns), print)
    lot = car_w.prendi("train", [ord_w[(a.passo - 1) % len(ord_w)]])
    pk = pacchi4[(a.passo - 1) % len(pacchi4)]
    ids4, inizi4, ult = CG.pacco_tensori(pk, dev)
    Tp = lot.ids.shape[1]
    if lot.inizi is not None:
        iP = lot.inizi.to(torch.long).cpu()
    else:
        iP = torch.zeros((1, Tp), dtype=torch.long); iP[0, 0] = 1
    ids = torch.cat([lot.ids.to(dev), ids4], 1); inizi = torch.cat([iP, inizi4], 1)
    nseg = int(inizi.sum())
    scrivi({"evento": "ingresso", "token": int(ids.shape[1]), "segmenti": nseg, "Tp": Tp})
    car_w.chiudi()
    gx = None
    import cardan_segmenti_gpu as SG
    originale = S.Studente._miscela_segmenti
    risultati = {}
    for var in a.varianti.split(","):
        SG.imposta(S, var)
        for opt_p in pE + pD:
            opt_p.grad = None
        tempi = []
        for r in range(a.ripetizioni + 1):
            guardia()
            for opt_p in pE + pD:
                opt_p.grad = None
            sinc(); t1 = time.time()
            stato = st.avanti(ids, inizi=inizi)
            sinc(); t2 = time.time()
            xL = stato["xL"]
            if gx is None:
                gx = (torch.randn(xL.shape, generator=g) * 1e-2).to(xL)
            st.indietro(stato, gx)
            sinc(); t3 = time.time()
            if r > 0:
                tempi.append((t2 - t1, t3 - t2))
        prof_file = None
        if a.profilo:
            from torch.profiler import profile, ProfilerActivity
            for opt_p in pE + pD:
                opt_p.grad = None
            with profile(activities=[ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if dev.type == "cuda" else [])) as pr:
                stato = st.avanti(ids, inizi=inizi); st.indietro(stato, gx); sinc()
            prof_file = out / f"profilo_{var}.txt"
            ka = pr.key_averages()
            prof_file.write_text(ka.table(sort_by="self_cuda_time_total", row_limit=40) + "\n\n" +
                                 ka.table(sort_by="self_cpu_time_total", row_limit=40))
            tot_cuda = sum(e.self_device_time_total for e in ka) / 1e6
            tot_cpu = sum(e.self_cpu_time_total for e in ka) / 1e6
            n_kern = sum(e.count for e in ka if e.self_device_time_total > 0 and e.device_type.name in ("CUDA",))
            scrivi({"evento": "profilo", "variante": var, "gpu_s": round(tot_cuda, 3), "cpu_self_s": round(tot_cpu, 3),
                    "lanci_gpu": n_kern, "file": str(prof_file)})
        with torch.no_grad():
            stato = st.avanti(ids, inizi=inizi)
        xLv = stato["xL"].detach().clone()
        for opt_p in pE + pD:
            opt_p.grad = None
        stato = st.avanti(ids, inizi=inizi); st.indietro(stato, gx)
        grads = [(p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)) for p in pE + pD]
        risultati[var] = (xLv, grads)
        av = sum(t[0] for t in tempi) / len(tempi); ind = sum(t[1] for t in tempi) / len(tempi)
        scrivi({"evento": "tempo", "variante": var, "avanti_s": round(av, 4), "indietro_s": round(ind, 4),
                "tempi": tempi, "vram_picco_GiB": (round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if dev.type == "cuda" else None)})
    S.Studente._miscela_segmenti = originale
    coppie = [("originale", v) for v in risultati if v != "originale"]
    if "lotto" in risultati and "riempito" in risultati:
        coppie.append(("lotto", "riempito"))           # isola l'effetto del lotto di segmenti nel GDN
    for nome_rif, var in coppie:
        if nome_rif in risultati:
            rif = risultati[nome_rif]
            x, gr = risultati[var]
            dx = (x.float() - rif[0].float())
            ug = sum(int(torch.equal(u, v)) for u, v in zip(gr, rif[1]))
            num = sum(float((u.double() - v.double()).pow(2).sum()) for u, v in zip(gr, rif[1]))
            den = sum(float(v.double().pow(2).sum()) for v in rif[1])
            scrivi({"evento": "confronto", "rif": nome_rif, "variante": var, "xL_al_bit": bool(torch.equal(x, rif[0])),
                    "xL_max_abs": float(dx.abs().max()), "xL_rel": float(dx.norm() / rif[0].float().norm()),
                    "xL_elementi_diversi": int((x != rif[0]).sum()), "xL_elementi": x.numel(),
                    "grad_tensori_al_bit": ug, "grad_tensori": len(gr), "grad_rel": (num / den) ** 0.5 if den else 0.0})
    scrivi({"evento": "fine", "host": host()})


if __name__ == "__main__":
    main()
