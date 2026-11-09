"""Ternarizza i tensori non ternari di Thor-T EB3 p20 DAL BF16 PIENO (28/9), uno alla volta, su CPU, con ripresa.

Uso: costruisci.py --insieme A|B [--strati 0,3] [--solo nome1,nome2]
  A = attenzione (attn_q/k/v/output, attn_qkv, attn_gate) + ssm_out + esperto condiviso (gate/up/down), 40 strati
  B = in piu' output.weight e token_embd.weight
Uscita: @@ROOT:depot@@/tutto_ternario/tensori/<nome>.npy (byte impacchettati) + manifesto.json (provenienza con sha256,
formato, misure). Un tensore gia' fatto (npy presente e sha uguale al manifesto) non si rifa.
Hessiane (catturate sul bf16, corpus piccolo 196.608 campioni: quelle della ricetta verificata del 20/9, MMLU 72,30 sui
149 densi): hessiane-attn/H_LL (ingresso input_layernorm: attn_q/k/v, attn_qkv, attn_gate), hessiane/H_LL (ingresso
post_attention_layernorm: ffn_gate_shexp, ffn_up_shexp), hessiane-output/H.npy (dopo output_norm: output.weight).
Senza Hessiana su disco: attn_output, ssm_out, ffn_down_shexp, token_embd (ottimo esatto non pesato).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

QUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, QUI)
sys.path.insert(0, "@@ROOT:models@@/build/llamacpp-tq1-k1024/gguf-py")
from gguf import GGUFReader  # noqa: E402
from sorgente import SorgenteHF, sha256_file  # noqa: E402
import tt_quant as TQ  # noqa: E402

EB3 = "@@ROOT:models@@/gguf/tony-tern/Thor-T-EB3-passo_000020.gguf"
OUT = Path(os.getenv("TT_TENSORI", "@@ROOT:depot@@/tutto_ternario/tensori"))   # una cartella (e un manifesto) per variante
MANIF = OUT / "manifesto.json"
HT = Path("@@ROOT:models@@/gguf/tony-tern")
FAM_A = ["attn_q", "attn_k", "attn_v", "attn_output", "attn_qkv", "attn_gate", "ssm_out",
         "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp"]
H_ATTN = {"attn_q", "attn_k", "attn_v", "attn_qkv", "attn_gate"}
H_FFN = {"ffn_gate_shexp", "ffn_up_shexp"}
H_NUOVE = Path("@@ROOT:depot@@/tutto_ternario/hessiane")   # catturate il 28/9 sul bf16: ingressi veri di o_proj, out_proj, down
H_INGRESSO = {"attn_output", "ssm_out", "ffn_down_shexp"}


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def hessiana(nome: str):
    if nome == "output.weight":
        p = HT / "hessiane-output" / "H.npy"
    elif nome.startswith("blk."):
        L = int(nome.split(".")[1]); fam = nome.split(".")[2]
        if fam in H_ATTN:
            p = HT / "hessiane-attn" / f"H_{L:02d}.npy"
        elif fam in H_FFN:
            p = HT / "hessiane" / f"H_{L:02d}.npy"
        elif fam in H_INGRESSO:
            p = H_NUOVE / fam / f"H_{L:02d}.npy"
            assert p.exists(), f"manca la Hessiana catturata {p}"
        else:
            return None, None
    else:
        return None, None
    return np.load(p).astype(np.float64), {"file": str(p.resolve()), "sha256": sha256_file(p)}


def elenco(insieme: str, strati=None, solo=None, famiglie=None) -> list[str]:
    nomi_eb3 = {t.name for t in GGUFReader(EB3).tensors}
    out = []
    for L in range(40):
        if strati is not None and L not in strati:
            continue
        out += [f"blk.{L}.{f}.weight" for f in FAM_A if f"blk.{L}.{f}.weight" in nomi_eb3
                and (famiglie is None or f in famiglie)]
    if insieme == "B":
        out += ["output.weight", "token_embd.weight"]
    if solo:
        out = [n for n in out if n in solo]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--insieme", default="A", choices=["A", "B"])
    ap.add_argument("--strati", default=None)
    ap.add_argument("--solo", default=None)
    ap.add_argument("--famiglie", default=None, help="elenco di famiglie (default tutte quelle di A)")
    ap.add_argument("--senza-lora", action="store_true", help="solo per il collaudo contro i byte del 20/9: niente Delta W")
    a = ap.parse_args()
    strati = None if a.strati is None else {int(x) for x in a.strati.split(",")}
    solo = None if a.solo is None else set(a.solo.split(","))
    OUT.mkdir(parents=True, exist_ok=True)
    man = json.load(open(MANIF)) if MANIF.exists() else {}
    src = SorgenteHF()
    famiglie = None if a.famiglie is None else set(a.famiglie.split(","))
    nomi = elenco(a.insieme, strati, solo, famiglie)
    if a.senza_lora:
        src.delta_lora = lambda nome: None
    log(f"insieme {a.insieme}: {len(nomi)} tensori -> {OUT}, ricetta {TQ.RICETTA}, device {TQ.DEVICE}, "
        f"thread torch {TQ.torch.get_num_threads()}, senza_lora {a.senza_lora}")
    for i, nome in enumerate(nomi):
        f = OUT / f"{nome}.npy"
        if nome in man and f.exists() and sha256_file(f) == man[nome]["sha256_npy"]:
            continue
        t0 = time.time()
        W, prov = src.peso_bersaglio(nome)
        H, hprov = hessiana(nome)
        if W.shape[1] % 1024 == 0:
            packed, mis = TQ.quantizza_1024(W, H, nome)
        else:
            packed, mis = TQ.quantizza_tq1_0(W, H, nome)
        tmp = OUT / f".{nome}.tmp.npy"
        np.save(tmp, packed)
        os.replace(tmp, f)
        man[nome] = {"forma_ggml": [int(W.shape[1]), int(W.shape[0])], "sorgente": prov, "hessiana": hprov,
                     "misure": mis, "sha256_byte": hashlib.sha256(packed.tobytes()).hexdigest(),
                     "sha256_npy": sha256_file(f), "scritto": time.strftime("%Y-%m-%dT%H:%M:%S")}
        tmpm = OUT / ".manifesto.tmp"
        json.dump(man, open(tmpm, "w"), indent=1)
        os.replace(tmpm, MANIF)
        c = mis.get("catena_pesata_rel")
        log(f"{i+1}/{len(nomi)} {nome} {mis['formato']} err_rel {mis['err_rel']:.4f}"
            + (f" pesato {mis['err_rel_pesato_H']:.4f} (non pesato: {mis['rif_nonpesato_err_rel_pesato_H']:.4f})"
               f" catena {c['gptq']:.4f}->{c['finale']:.4f}" if H is not None else "")
            + (f" LoRA {prov['delta_lora']['norma_rel']:.4f}" if prov.get("delta_lora") else "")
            + f" rt {mis['roundtrip_max_abs']:.1e} {time.time()-t0:.0f}s")
    log("FATTO")


if __name__ == "__main__":
    main()
