#!/usr/bin/env python3
"""Obiettivo C / C0T (9/10): PIEGA di post_attention_norm (g) negli esperti instradati ffn_gate_exps / ffn_up_exps
(strati 0-39, TQ1_0R_1024, primo piano = quello di C0), sulla GPU col lucchetto.
Bersaglio per esperto (base memorizzata): T' = Q'_C0 R diag(g) R^T (esatto, test_c0t_matematica T1/T4); metrica = Hessiana
dell'ingresso dell'esperto SENZA guadagno, H_x^ = D H D (D = diag(1/g)), H = Gq per esperto (flusso quantizzato,
strati 0-31) o H_ffn condivisa (come cardan_piega.Hessiane), poi ruotata.
Due vie per esperto, vince l'errore H minore (scelta MISURATA, registrata):
  J = JARQ batched dai codici di C0 (cardan_piega.jarq_lotto, fattorizzazioni su CPU fp64)
  F = quantizzazione da zero del bersaglio (tt_quant.quantizza_1024, ricetta 20set, la stessa di C0), TT_DEVICE=cuda
Uscita: <out>/blk.L.ffn_{gate,up}_exps.weight.npy (uint8 [256, 512, 448]) + registro.jsonl (per strato e famiglia)."""
import argparse, json, os, sys, time
from pathlib import Path
os.environ.setdefault("TT_DEVICE", "cuda")
sys.path.insert(0, "@@ROOT:models@@/gguf/tony-tern/OBIETTIVO_C/codice")
import numpy as np
import torch
import c0t_piega as P
CM, CP, gp, TQ = P.CM, P.CP, P.gp, P.TQ
from gguf import GGUFReader


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def strato(L, r, dev, dt, lotto, td, reg, usa_F):
    from rd_ternary.formato import TensorKey
    g = np.array({t.name: t for t in r.tensors}[f"blk.{L}.post_attention_norm.weight"].data, np.float64)
    gt = torch.from_numpy(g)
    M = CM.ruota_H(torch.diag(gt))                                        # R diag(g) R^T  (C x C)
    D = (1.0 / gt)
    src = gp.GGUFPlaneSource(CM.R2B3M, layers=[L], bases=["ffn_gate_exps", "ffn_up_exps"])
    hs = CP.Hessiane(L)
    for f in ("gate", "up"):
        out = td / f"blk.{L}.ffn_{f}_exps.weight.npy"
        if out.exists():
            continue
        t0 = time.time()
        e_ = src.entry(TensorKey(L, 0, f)); tid = e_.fmt.type_id
        nE = e_.n_experts
        raw = np.array(e_.tensors[0].data).reshape(nE, e_.out, -1).copy()
        tot = {"J": 0, "F": 0, "num": 0.0, "den": 0.0, "num_J": 0.0, "num_F": 0.0, "fonti": {}}
        for a in range(0, nE, lotto):
            ee = list(range(a, min(nE, a + lotto)))
            Ts, Ss, Hs, Hraw = [], [], [], []
            for e in ee:
                tt = src.load(TensorKey(L, e, f))
                H, fonte = hs.di(e); tot["fonti"][fonte] = tot["fonti"].get(fonte, 0) + 1
                Hx = P.norm_H(H * D[:, None] * D[None, :])
                Hraw.append(Hx.numpy()); Hs.append(CM.ruota_H(Hx))
                Ts.append(tt.planes[:1]); Ss.append(tt.scales[:1].double())
            T = torch.stack(Ts).to(dev); S = torch.stack(Ss).to(dev, dt); H = torch.stack(Hs).to(dev, dt)
            Q0 = CP.ricostruisci_lotto(T, S)
            W = Q0 @ M.to(dev, dt)
            Ha = H + 1e-4 * torch.eye(H.shape[-1], dtype=dt, device=dev)
            T2, S2, _ = CP.jarq_lotto(W, H, T, S, giri=3, nu2=1e-4)
            eJ = CP.obiettivo_lotto(W, CP.ricostruisci_lotto(T2, S2), Ha).sum(-1)
            den = CP.obiettivo_lotto(W, torch.zeros_like(W), Ha).sum(-1)
            T2c, S2c = T2.cpu(), S2.cpu().to(torch.float16)
            for j, e in enumerate(ee):
                scelta, Tb, Sb, en = "J", T2c[j, 0].numpy().astype(np.int8), S2c[j, 0].numpy(), float(eJ[j])
                tot["num_J"] += float(eJ[j])
                if usa_F:
                    Wo = CM.deruota(Q0[j].double().cpu()) * gt
                    packed, _ = TQ.quantizza_1024(Wo.float().numpy(), Hraw[j], f"L{L}{f}{e}")
                    Tf, Sf = gp.decode_rows(packed.reshape(e_.out, -1), tid, e_.in_)
                    Qf = (torch.from_numpy(Tf.astype(np.float32)).double().reshape(e_.out, Sf.shape[1], -1)
                          * torch.from_numpy(Sf.astype(np.float32)).double().unsqueeze(-1)).reshape(e_.out, -1)
                    Ef = W[j].double().cpu() - Qf
                    eF = float(((Ef @ Ha[j].double().cpu()) * Ef).sum())
                    tot["num_F"] += eF
                    if eF < en:
                        scelta, Tb, Sb, en = "F", Tf.astype(np.int8), Sf.astype(np.float16), eF
                tot[scelta] += 1; tot["num"] += en; tot["den"] += float(den[j])
                b = gp.encode_rows(Tb, Sb, tid)
                tp, sp_ = gp.decode_rows(b, tid, e_.in_)
                assert np.array_equal(tp, Tb) and np.array_equal(sp_.view(np.uint16), np.asarray(Sb).view(np.uint16)), (L, f, e)
                raw[e] = b
        np.save(out, raw)
        riga = {"strato": L, "famiglia": f, "esperti": nE, "scelti_J": tot["J"], "scelti_F": tot["F"],
                "err_rel_H": (tot["num"] / tot["den"]) ** 0.5, "err_rel_H_solo_J": (tot["num_J"] / tot["den"]) ** 0.5,
                "err_rel_H_solo_F": ((tot["num_F"] / tot["den"]) ** 0.5) if usa_F else None,
                "hessiane": tot["fonti"], "secondi": round(time.time() - t0, 1)}
        reg.write(json.dumps(riga) + "\n"); reg.flush()
        log(json.dumps(riga))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="@@ROOT:models@@/gguf/tony-tern/OBIETTIVO_C/C0T/esperti")
    ap.add_argument("--strati", default="0-39")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lotto", type=int, default=16)
    ap.add_argument("--senza-F", action="store_true")
    a = ap.parse_args()
    dev = torch.device(a.device); dt = torch.float64 if dev.type == "cpu" else torch.float32
    torch.set_num_threads(6)
    td = Path(a.out); td.mkdir(parents=True, exist_ok=True)
    r = GGUFReader(CM.R2B3M)
    lo, _, hi = a.strati.partition("-")
    with open(td / "registro.jsonl", "a") as reg:
        for L in range(int(lo), int(hi or lo) + 1):
            strato(L, r, dev, dt, a.lotto, td, reg, not a.senza_F)
    log("FINE piega esperti")


if __name__ == "__main__":
    main()
