"""Verifica della SORGENTE (28/9, prima di ternarizzare): il bf16 HF permutato con la mappa e' davvero il tensore
del GGUF, e il GGUF di EB3 p20 e' quel tensore quantizzato (piu' la LoRA dove EB3 l'ha allenata).

Per ogni tensore denso degli strati scelti (+ token_embd, output):
 (1) HF bf16 + perm  ==  GGUF bf16 del convertitore (Ornith-1.5-35B-A3B-bf16), BIT PER BIT;
     controllo negativo: senza perm (famiglie permutate) i bit NON coincidono.
 (2) Q8_0(HF + perm) == byte di EB3 p20 (tensori Q8_0 senza Delta W: il sorgente di EB3 e' Q8_0 del bf16);
     tensori con Delta W: deq(EB3) contro deq(Q8_0(W)) + Delta W, errore in passi di quantizzazione (<= 0,5 + eps);
     k-quant (Q4_K/Q5_K/Q6_K/Q4_0): errore relativo e correlazione per riga contro W, e controllo negativo senza perm.
Uscita: ../log/verifica_sorgente.json
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "@@ROOT:models@@/build/llamacpp-tq1-k1024/gguf-py")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gguf  # noqa: E402
from gguf import GGUFReader, GGMLQuantizationType as T  # noqa: E402
from sorgente import SorgenteHF, bf16_a_f32  # noqa: E402

EB3 = "@@ROOT:models@@/gguf/tony-tern/Thor-T-EB3-passo_000020.gguf"
BF16 = sorted(glob.glob("@@ROOT:models@@/gguf/thor-src/Ornith-1.5-35B-A3B-bf16/*.gguf"))
STRATI = [int(x) for x in os.getenv("TT_STRATI", "0,3,11,21,39").split(",")]
FAM = ["attn_q", "attn_k", "attn_v", "attn_output", "attn_qkv", "attn_gate", "ssm_out",
       "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp"]
RIGHE_PEZZO = 16384


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def deq(t, r0, r1):
    """Dequantizza le righe r0:r1 (numpy, gguf-py) di un tensore del GGUF."""
    raw = np.asarray(t.data)[r0:r1]
    return gguf.quants.dequantize(raw, t.tensor_type).astype(np.float32).reshape(r1 - r0, -1)


def corr_righe(a, b):
    a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
    na = np.linalg.norm(a, axis=1); nb = np.linalg.norm(b, axis=1)
    ok = (na > 0) & (nb > 0)
    return (a[ok] * b[ok]).sum(1) / (na[ok] * nb[ok])


def main():
    src = SorgenteHF()
    eb3 = {t.name: t for t in GGUFReader(EB3).tensors}
    bf = {}
    for p in BF16:
        for t in GGUFReader(p).tensors:
            bf[t.name] = t
    nomi = ["token_embd.weight", "output.weight"]
    for L in STRATI:
        nomi += [f"blk.{L}.{f}.weight" for f in FAM if f"blk.{L}.{f}.weight" in eb3]
    esiti = {}
    for nome in nomi:
        t0 = time.time()
        u16, prov = src.u16_ordine_gguf(nome)
        tb = bf[nome]
        assert tb.tensor_type == T.BF16, (nome, tb.tensor_type)
        ub = np.asarray(tb.data).view(np.uint16).reshape(u16.shape)
        e = {"forma": list(u16.shape), "prov": prov, "tipo_eb3": eb3[nome].tensor_type.name,
             "bit_uguali_bf16": int((u16 == ub).sum()), "n": int(u16.size)}
        e["bf16_identico"] = e["bit_uguali_bf16"] == e["n"]
        if prov["relazione"] != "identita":
            u_np, _ = src.u16_ordine_gguf(nome, applica_perm=False)
            e["neg_senza_perm_bit_uguali"] = int((u_np == ub).sum())
        del ub
        W = bf16_a_f32(u16)
        dW = src.delta_lora(nome)
        e["delta_lora_norma_rel"] = None if dW is None else float(np.linalg.norm(dW) / np.linalg.norm(W))
        te = eb3[nome]
        R = W.shape[0]
        if te.tensor_type == T.Q8_0:
            if dW is None:
                q = gguf.quants.quantize(W, T.Q8_0)
                e["q8_byte_uguali_frazione"] = float((q.reshape(-1) == np.asarray(te.data).reshape(-1)).mean())
            else:
                qd = gguf.quants.dequantize(gguf.quants.quantize(W, T.Q8_0), T.Q8_0).reshape(W.shape) + dW
                de = deq(te, 0, R)
                d = np.asarray(te.data).reshape(R, -1, 34)[:, :, :2].copy().view(np.float16).astype(np.float32)
                passo = np.repeat(d.reshape(R, -1), 32, axis=1)
                r = np.abs(de - qd) / np.maximum(passo, 1e-12)
                e["lora_passi_max"] = float(r.max()); e["lora_passi_p999"] = float(np.quantile(r, 0.999))
                e["lora_err_rel"] = float(np.linalg.norm(de - qd) / np.linalg.norm(qd))
                e["senza_lora_err_rel"] = float(np.linalg.norm(de - (qd - dW)) / np.linalg.norm(qd))
        # errore relativo e correlazione per riga (tutti i tipi), a pezzi di righe
        num = den = 0.0; cmin = 1.0; cneg = []
        u_np = None
        if prov["relazione"] != "identita":
            u_np, _ = src.u16_ordine_gguf(nome, applica_perm=False)
        for r0 in range(0, R, RIGHE_PEZZO):
            r1 = min(R, r0 + RIGHE_PEZZO)
            de = deq(te, r0, r1)
            ref = W[r0:r1] + (dW[r0:r1] if dW is not None else 0)
            num += float(((de - ref) ** 2).sum()); den += float((ref ** 2).sum())
            c = corr_righe(de, ref); cmin = min(cmin, float(c.min()))
            if u_np is not None:
                cneg.append(float(np.median(corr_righe(de, bf16_a_f32(u_np[r0:r1])))))
        e["eb3_err_rel"] = (num / den) ** 0.5
        e["eb3_corr_riga_min"] = cmin
        if cneg:
            e["neg_senza_perm_corr_mediana"] = float(np.median(cneg))
        e["secondi"] = round(time.time() - t0, 1)
        esiti[nome] = e
        log(nome, {k: v for k, v in e.items() if k != "prov"})
    ok = all(v["bf16_identico"] for v in esiti.values())
    neg = all(v.get("neg_senza_perm_bit_uguali", 0) < v["n"] for v in esiti.values())
    out = {"strati": STRATI, "tutti_bf16_identici": ok, "negativi_diversi": neg, "esiti": esiti}
    json.dump(out, open("@@ROOT:depot@@/tutto_ternario/log/verifica_sorgente.json", "w"), indent=1)
    log("RIASSUNTO bf16 identici:", ok, "· controlli negativi diversi:", neg)


if __name__ == "__main__":
    main()
