#!/usr/bin/env python3
"""OBQ fibrato, tappa CPU: nuovi trit dei densi a un piano dell'OSPITE con la metrica M = H (x) I + lam A_cal (x) B_cal
(fib_solutore.obq_fibrato; scale dell'ospite FISSE -> stessi byte, stessa velocita' per costruzione).

GUARDIA PER TENSORE (variante Thor dichiarata; V §4.3.4 e SINTESI §3.3: l'aumento di errH e' la perdita attesa sui
compiti ignoti e non si liquida): il tensore nuovo entra solo se
  (1) KF_ctl(nuovo) < KF_ctl(ospite)   — KL4 al secondo ordine sulla meta' di CONTROLLO, mai usata per adattare;
  (2) errH(nuovo) <= errH(ospite) * (1 + errh_max)   — H di calibrazione (quella di tutte le varianti di C).
Altrimenti resta il tensore dell'ospite (nessun file scritto).
H = Hessiana del manifesto (calib_auto2, la stessa di (c)+DRQ), bersaglio = bf16 puro del maestro nella base memorizzata
(jarq_fedele.bersaglio_bf16_e_H). Fattori: simmetrizzati e riportati a media diagonale 1, senza proiezione PSD (gli
autovalori negativi d'arrotondamento sono troncati a zero dentro il solutore).
Ripresa: una riga per tensore in registro.jsonl (scritta DOPO l'eventuale npy, rinominato atomicamente); i tensori gia'
registrati si saltano.
uso: fib_risolvi.py --ospite G --fattori DIR(cal,ctl) --out DIR [--lam 0.3] [--errh-max 0.10] [--solo a,b] [--fili 6]
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

QUI = Path(__file__).resolve().parent


def carica_fattore(p: Path):
    import torch
    M = torch.from_numpy(np.load(p).astype(np.float64))
    M = 0.5 * (M + M.T)
    return M / torch.diagonal(M).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ospite", required=True)
    ap.add_argument("--fattori", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lam", type=float, default=0.3)
    ap.add_argument("--errh-max", type=float, default=0.10)
    ap.add_argument("--solo", default="")
    ap.add_argument("--fili", type=int, default=6)
    a = ap.parse_args()
    os.environ["LEVE_THREADS"] = str(a.fili)
    for p in (str(QUI), "@@ROOT:models@@/build/leve/raffina", "@@ROOT:models@@/gguf/tony-tern/OBIETTIVO_C/codice",
              "@@ROOT:depot@@/tutto_ternario/codice"):
        if p not in sys.path:
            sys.path.insert(0, p)
    import torch
    import comune as CM
    import jarq_fedele as JF
    import fib_solutore as FS
    torch.set_num_threads(a.fili)
    fat = Path(a.fattori)
    out = Path(a.out)
    (out / "tensori").mkdir(parents=True, exist_ok=True)
    reg = out / "registro.jsonl"
    fatti = set()
    if reg.exists():
        fatti = {json.loads(r)["nome"] for r in open(reg) if r.strip()}
    nomi = sorted({p.name[2:-4] for p in (fat / "cal").glob("B_blk.*.npy")} &
                  {p.name[2:-4] for p in (fat / "ctl").glob("B_blk.*.npy")},
                  key=lambda n: (int(n.split(".")[1]), n))
    if a.solo:
        nomi = [n for n in nomi if n in set(a.solo.split(","))]
    print(f"[fib] {len(nomi)} densi con fattori cal+ctl; gia' fatti {len(fatti & set(nomi))}; lam {a.lam}; "
          f"errH max +{a.errh_max*100:.0f} %; fili {a.fili}", flush=True)
    src = {}
    t_tot = time.time()
    for nome in nomi:
        if nome in fatti:
            continue
        t0 = time.time()
        L = int(nome.split(".")[1])
        if L not in src:
            src.clear(); gc.collect()
            src[L] = CM.gp.GGUFPlaneSource(a.ospite, layers=[L])
        d = CM.Denso(src[L], nome)
        if d.K != 1:
            raise SystemExit(f"{nome}: {d.K} piani nell'ospite (la leva e' per i densi a UN piano)")
        tid, rot = d.tid, d.rotated
        Tq, D = d.T[0].clone(), d.S[0].double().clone()
        del d
        W, H, _, _ = JF.bersaglio_bf16_e_H(nome, rot)
        R, C = W.shape
        assert Tq.shape == (R, C), (nome, Tq.shape, W.shape)
        Eh = W - FS.ricostruisci(Tq, D)
        A, B = carica_fattore(fat / "cal" / f"A_{nome}.npy"), carica_fattore(fat / "cal" / f"B_{nome}.npy")
        kf_cal_h = FS.relKF(Eh, W, A, B)
        ts = time.time()
        T = FS.obq_fibrato(W, H, D, [(a.lam, A, B)])
        t_sol = time.time() - ts
        En = W - FS.ricostruisci(T, D)
        kf_cal_n = FS.relKF(En, W, A, B)
        del A, B; gc.collect()
        A, B = carica_fattore(fat / "ctl" / f"A_{nome}.npy"), carica_fattore(fat / "ctl" / f"B_{nome}.npy")
        kf_ctl_h, kf_ctl_n = FS.relKF(Eh, W, A, B), FS.relKF(En, W, A, B)
        del A, B; gc.collect()
        eh_h, eh_n = FS.rel(Eh, W, H), FS.rel(En, W, H)
        ok = bool(kf_ctl_n < kf_ctl_h and eh_n <= eh_h * (1 + a.errh_max))
        if ok:
            S16 = D.to(torch.float16).numpy()
            Tn = T.numpy().astype(np.int8)
            raw = CM.gp.encode_rows(Tn, S16, tid)
            t2, s2 = CM.gp.decode_rows(raw, tid, C)
            assert np.array_equal(t2, Tn) and np.array_equal(s2.view(np.uint16), S16.view(np.uint16)), "andata e ritorno"
            f = out / "tensori" / f"{nome}.npy"
            tmpf = out / "tensori" / f".{nome}.tmp.npy"
            np.save(tmpf, np.ascontiguousarray(raw.astype(np.uint8)))
            os.replace(tmpf, f)
        rig = {"nome": nome, "R": R, "C": C, "tipo": tid, "ruotato": bool(rot), "lam": a.lam,
               "errH_ospite": eh_h, "errH_nuovo": eh_n, "kf_cal_ospite": kf_cal_h, "kf_cal_nuovo": kf_cal_n,
               "kf_ctl_ospite": kf_ctl_h, "kf_ctl_nuovo": kf_ctl_n,
               "dkf_ctl_%": 100 * (kf_ctl_n / kf_ctl_h - 1), "derrH_%": 100 * (eh_n / eh_h - 1),
               "trit_cambiati": float((T != Tq).double().mean()), "accettato": ok,
               "secondi_solutore": round(t_sol, 1), "secondi": round(time.time() - t0, 1)}
        with open(reg, "a") as fh:
            fh.write(json.dumps(rig) + "\n")
        print(f"[fib] {nome} ({R}x{C}): KF ctl {kf_ctl_h:.4f} -> {kf_ctl_n:.4f} ({rig['dkf_ctl_%']:+.1f} %), cal "
              f"{kf_cal_h:.4f} -> {kf_cal_n:.4f}, errH {rig['derrH_%']:+.1f} %, trit {rig['trit_cambiati']*100:.1f} %: "
              f"{'ACCETTATO' if ok else 'resta ospite'} ({rig['secondi']:.0f} s)", flush=True)
        del W, H, T, Tq, D, Eh, En; gc.collect()
    rr = [json.loads(r) for r in open(reg) if r.strip()]
    rr = [r for r in rr if r["nome"] in set(nomi)]
    acc = [r for r in rr if r["accettato"]]
    med = lambda v: float(np.median(v)) if v else float("nan")
    sint = {"tensori": len(rr), "accettati": len(acc), "lam": a.lam, "errh_max": a.errh_max,
            "dkf_ctl_mediana_%": med([r["dkf_ctl_%"] for r in rr]),
            "dkf_ctl_mediana_accettati_%": med([r["dkf_ctl_%"] for r in acc]),
            "derrH_mediana_accettati_%": med([r["derrH_%"] for r in acc]),
            "per_famiglia": {}, "ore_cpu": round(sum(r["secondi"] for r in rr) / 3600, 2)}
    for r in rr:
        f = r["nome"].split(".")[2]
        x = sint["per_famiglia"].setdefault(f, [0, 0])
        x[0] += 1; x[1] += int(r["accettato"])
    (out / "sintesi.json").write_text(json.dumps(sint, indent=1) + "\n")
    print(f"SINTESI accettati {len(acc)}/{len(rr)}; dKF ctl mediana {sint['dkf_ctl_mediana_%']:+.1f} % (accettati "
          f"{sint['dkf_ctl_mediana_accettati_%']:+.1f} %); errH accettati {sint['derrH_mediana_accettati_%']:+.1f} %; "
          f"famiglie {sint['per_famiglia']}; ore CPU {sint['ore_cpu']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
