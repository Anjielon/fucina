#!/usr/bin/env python3
"""Obiettivo C — stadi 2-3 su UN piano (9/10/2026), SOLO CPU. Varianti dei 250 densi a un piano, misurabili con
c0_scrivi.py (stessi nomi/tipi del primo piano dell'ospite R2B3m = TT2 p6):
  s1ls      codici TT2 (allenati end-to-end) FERMI, scale ai minimi quadrati congiunti pesati H (JARQ eq. 6), fp16
  jarq1     JARQ (2609.38599 Alg. 1) con K=1 partendo da TT2: scale congiunte + proposte di Babai, riga mai peggio
  bil1jarq  da zero: GPTQ ternario (griglia bilanciata K=1 = soglia ternaria pesata H) + JARQ K=1 (ricetta A2 moderna)
Bersaglio e H = comune.bersaglio_e_H (bf16 + dW LoRA EB3 p20; H del manifesto R2B3, corpus 20/9).
⚠ Lezione del 9/10 (PROVA_LEVE_VS_PAPER): ridurre l'errore pesato H verso il bf16 NON garantisce MMLU migliore su un
modello allenato; per questo ogni variante si giudica con KL della risposta MMLU + MMLU appaiato, non con errH.
Uscita: <out>/<variante>/tensori/<nome>.npy + manifesto.json + registro.jsonl (errH/fro prima e dopo). Riprendibile.
uso: c_stadio3.py --variante s1ls|jarq1|bil1jarq [--out DIR] [--solo nome,...]
"""
import argparse, hashlib, json, os, sys, time
sys.path.insert(0, "@@ROOT:models@@/build/leve/raffina")
import numpy as np
import torch
import comune as CM
import solutori as SO
GT = CM.gp._import_gguf(None).GGMLQuantizationType


def una(d, var):
    W, H, prov = CM.bersaglio_e_H(d.nome, d.rotated)
    T1, S1 = d.T[:1].clone(), d.S[:1].clone()
    ng = S1.shape[-1]
    e0 = CM.err_pesato(W, SO.ricostruisci(T1, S1), H)
    if var == "s1ls":
        S2 = SO.fp16(SO.scale_congiunte(W, H, T1, torch.zeros_like(W), ng)).float()
        T2, cat = T1, {}
    elif var == "jarq1":
        T2, S2, cat = SO.jarq(W, H, T1, S1, [0], giri=3, nu2=1e-4)
    elif var == "bil1jarq":
        Tb, Sb, cb = SO.gptq_bilanciato(W, H, 1, ng)
        T2, S2, cat = SO.jarq(W, H, Tb, Sb, [0], giri=3, nu2=1e-4)
        cat.update({f"bil_{k}": v for k, v in cb.items() if isinstance(v, (int, float))})
    else:
        raise ValueError(var)
    e1 = CM.err_pesato(W, SO.ricostruisci(T2, S2), H)
    zeri = float((T2[0] == 0).double().mean())
    cambi = float((T2[0] != T1[0]).double().mean())
    return T2, S2, {"errH_tt2": e0[0], "errH": e1[0], "fro_tt2": e0[1], "fro": e1[1], "zeri": zeri,
                    "trit_cambiati": cambi, "catena": {k: v for k, v in cat.items() if isinstance(v, (int, float))}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variante", required=True)
    ap.add_argument("--out", default="@@ROOT:models@@/gguf/tony-tern/OBIETTIVO_C/varianti")
    ap.add_argument("--solo", default="")
    a = ap.parse_args()
    out = os.path.join(a.out, a.variante, "tensori"); os.makedirs(out, exist_ok=True)
    reg = os.path.join(a.out, a.variante, "registro.jsonl")
    fatti = {json.loads(l)["nome"] for l in open(reg)} if os.path.exists(reg) else set()
    src = CM.gp.GGUFPlaneSource(CM.R2B3M, layers=list(range(40)))
    nomi = CM.elenco_densi(src)
    if a.solo:
        nomi = [n for n in nomi if n in a.solo.split(",")]
    CM.log(f"variante {a.variante}: {len(nomi)} densi, gia' fatti {len(fatti)}")
    mp = os.path.join(out, "manifesto.json")
    man = json.load(open(mp)) if os.path.exists(mp) else {}
    for i, nome in enumerate(nomi):
        if nome in fatti:
            continue
        t0 = time.time()
        d = CM.Denso(src, nome)
        T2, S2, mis = una(d, a.variante)
        b = d.byte_piani(T2, S2)[0]
        p1 = d.nomi_piani[0]
        np.save(os.path.join(out, p1 + ".npy"), b)
        man[p1] = {"tipo": GT(d.tid).name, "forma_ggml": [int(d.in_), int(d.out)],
                   "sha256_npy": hashlib.sha256(np.ascontiguousarray(b).tobytes()).hexdigest(),
                   "misure": mis, "variante": a.variante}
        json.dump(man, open(mp + ".tmp", "w")); os.replace(mp + ".tmp", mp)
        mis.update({"nome": nome, "secondi": round(time.time() - t0, 1)})
        with open(reg, "a") as f:
            f.write(json.dumps(mis) + "\n")
        CM.log(f"{i+1}/{len(nomi)} {nome}: errH {mis['errH_tt2']:.4f} -> {mis['errH']:.4f}, trit cambiati {100*mis['trit_cambiati']:.1f} %, {mis['secondi']} s")
    CM.log("fine", a.variante)


if __name__ == "__main__":
    main()
