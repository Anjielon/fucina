"""Monta il GGUF di una variante (28/9): EB3 p20 con i tensori densi sostituiti dai ternari costruiti dal bf16.
Tutto il resto (esperti ternari con scale/codici di EB3, norme e parametri F32 allenati, router, strato nextn) e'
copiato AL BYTE dal GGUF di EB3 p20.

Uso: assembla.py --variante NOME --da DIR[,DIR...] --attesi N --ricetta TESTO --uscita <file.gguf>
  (i tensori sostituiti sono ESATTAMENTE le voci dei manifesti delle cartelle --da; un nome in due manifesti = errore)
Poi RICARICA il file scritto e verifica: stessi nomi/ordine/forme di EB3; copiati == byte di EB3 (sha256);
riscritti == byte del manifesto (sha256) nel tipo atteso. Scrive <uscita>.provenienza.json (sorgente + sha256 per
ogni tensore) e <uscita>.verifica.json.
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
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType as T  # noqa: E402
from sorgente import sha256_file  # noqa: E402

EB3 = "@@ROOT:models@@/gguf/tony-tern/Thor-T-EB3-passo_000020.gguf"
TENS = Path("@@ROOT:depot@@/tutto_ternario/tensori")
FAM_A = ("attn_q", "attn_k", "attn_v", "attn_output", "attn_qkv", "attn_gate", "ssm_out",
         "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp")
BPB = {T.TQ1_0R_1024: (1024, 224), T.TQ1_0: (256, 54)}


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variante", required=True)
    ap.add_argument("--da", required=True)
    ap.add_argument("--attesi", type=int, required=True)
    ap.add_argument("--ricetta", required=True)
    ap.add_argument("--uscita", required=True)
    a = ap.parse_args()
    man, cartella = {}, {}
    for d in a.da.split(","):
        m = json.load(open(Path(d) / "manifesto.json"))
        for k, v in m.items():
            assert k not in man, f"{k} in due manifesti"
            man[k] = v; cartella[k] = Path(d)
    assert len(man) == a.attesi, f"{len(man)} tensori nei manifesti, attesi {a.attesi}"
    sostituito = lambda nome, _v=None: nome in man
    log("sha256 del GGUF di EB3 p20...")
    sha_eb3 = sha256_file(Path(EB3))
    log("sha256 EB3 p20:", sha_eb3)
    r = GGUFReader(EB3)
    w = GGUFWriter(path=None, arch="qwen35moe")
    for f in r.fields.values():
        if f.name.startswith(("GGUF.", "split.")):
            continue
        w.add_key_value(f.name, f.contents(), f.types[0], sub_type=f.types[-1] if len(f.types) > 1 else None)
    n_sost = sum(1 for t in r.tensors if sostituito(t.name, a.variante))
    assert n_sost == a.attesi, (n_sost, a.attesi)
    w.add_string("thor.tutto_ternario", (
        f"variante {a.variante}: Thor-T EB3 passo_000020 (sha256 {sha_eb3}) con {n_sost} tensori densi TERNARI "
        f"costruiti dal bf16 pieno (thor-src-hf + mappa perm 6114b1f4 + Delta W LoRA di EB3 p20). Ricetta: {a.ricetta}. "
        "Esperti, norme, F32, router, strato nextn: byte di EB3 p20."))
    piano, prov = [], {}
    for t in r.tensors:
        if sostituito(t.name, a.variante):
            m = man[t.name]
            packed = np.load(cartella[t.name] / f"{t.name}.npy", mmap_mode="r")
            assert hashlib.sha256(np.ascontiguousarray(packed).tobytes()).hexdigest() == m["sha256_byte"], t.name
            tipo = T.TQ1_0R_1024 if m["misure"]["formato"] == "TQ1_0R_1024" else T.TQ1_0
            ne0, ne1 = m["forma_ggml"]
            assert [int(x) for x in t.shape] == [ne0, ne1], (t.name, t.shape, m["forma_ggml"])
            qk, nbyte = BPB[tipo]
            forma_byte = [ne1, ne0 // qk * nbyte]
            assert packed.size == ne1 * (ne0 // qk) * nbyte, t.name
            w.add_tensor_info(t.name, forma_byte, np.dtype(np.uint8), int(packed.size), raw_dtype=tipo)
            piano.append((t.name, "ternario", packed.reshape(forma_byte)))
            prov[t.name] = {"azione": "ternarizzato dal bf16", "cartella": str(cartella[t.name]), "tipo": tipo.name, "forma_ggml": [ne0, ne1],
                            "sorgente": m["sorgente"], "hessiana": m["hessiana"], "misure": m["misure"],
                            "sha256_byte": m["sha256_byte"], "tipo_in_eb3": t.tensor_type.name}
        else:
            raw = np.asarray(t.data)
            w.add_tensor_info(t.name, list(raw.shape), raw.dtype, int(raw.nbytes), raw_dtype=t.tensor_type)
            piano.append((t.name, "copia", raw))
            prov[t.name] = {"azione": "copiato al byte da EB3 p20", "tipo": t.tensor_type.name,
                            "sorgente": {"gguf": EB3, "sha256_file": sha_eb3},
                            "sha256_byte": hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest()}
    log(f"piano: {len(piano)} tensori, {n_sost} ternari nuovi, {len(piano) - n_sost} copiati")
    tmp = a.uscita + ".parziale"
    w.open_output_file(Path(tmp))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
    for i, (nome, az, dati) in enumerate(piano):
        w.write_tensor_data(np.ascontiguousarray(dati))
        if i % 100 == 0:
            log(f"  {i+1}/{len(piano)} {nome}")
    w.close()
    os.replace(tmp, a.uscita)
    log(f"scritto {a.uscita}: {os.path.getsize(a.uscita)/2**30:.4f} GiB")
    # ── RICARICA e verifica al byte ──
    r2 = GGUFReader(a.uscita)
    ver = {"n_tensori": len(r2.tensors), "n_eb3": len(r.tensors), "errori": []}
    eb = {t.name: t for t in r.tensors}
    for t2, t1 in zip(r2.tensors, r.tensors):
        if t2.name != t1.name:
            ver["errori"].append(f"ordine: {t2.name} != {t1.name}"); continue
        if [int(x) for x in t2.shape] != [int(x) for x in t1.shape]:
            ver["errori"].append(f"forma {t2.name}")
        s = hashlib.sha256(np.ascontiguousarray(np.asarray(t2.data)).tobytes()).hexdigest()
        if s != prov[t2.name]["sha256_byte"]:
            ver["errori"].append(f"byte {t2.name}")
        if t2.tensor_type.name != prov[t2.name]["tipo"]:
            ver["errori"].append(f"tipo {t2.name}")
    ver["byte_per_tipo"] = {}
    for t2 in r2.tensors:
        ver["byte_per_tipo"][t2.tensor_type.name] = ver["byte_per_tipo"].get(t2.tensor_type.name, 0) + int(t2.n_bytes)
    ver["taglia_file"] = os.path.getsize(a.uscita)
    ver["taglia_eb3"] = os.path.getsize(EB3)
    ver["sha256_file"] = sha256_file(Path(a.uscita))
    ver["ok"] = not ver["errori"] and len(r2.tensors) == len(r.tensors)
    json.dump(prov, open(a.uscita + ".provenienza.json", "w"), indent=1)
    json.dump(ver, open(a.uscita + ".verifica.json", "w"), indent=1)
    log("VERIFICA", "OK" if ver["ok"] else f"⛔ {ver['errori'][:10]}", f"taglia {ver['taglia_file']/2**30:.4f} GiB",
        f"(EB3 {ver['taglia_eb3']/2**30:.4f})", "sha256", ver["sha256_file"])
    sys.exit(0 if ver["ok"] else 1)


if __name__ == "__main__":
    main()
