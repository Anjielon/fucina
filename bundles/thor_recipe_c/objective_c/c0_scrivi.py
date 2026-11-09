#!/usr/bin/env python3
"""Obiettivo C (9/10/2026): GGUF a UN SOLO piano ternario per tensore, da un ospite multi-piano.

Si TOLGONO tutti i piani aggiuntivi (<famiglia>2/3/4 dei densi, token_embd2/3, output2/3, ffn_*_exps2/3 e le loro
mappe); i tensori in --tensori/<nome>.npy (con manifesto*.json: tipo, forma_ggml, sha256_npy) SOSTITUISCONO quelli
dell'ospite (anche cambiando tipo, es. ssm_alpha F32 -> TQ1_0R_1024); tutto il resto e' copiato AL BYTE.
Con R2B3m come ospite il risultato e' TT2 p6 (densi a un piano allenati end-to-end) + token_embd/output a un piano
(primo piano di RB, che per costruzione avida e' l'ottimo a un piano) + blk.40 TQ1_0 + le sostituzioni.
Verifica: riletto, ordine/forme/byte/tipi; NESSUN piano aggiuntivo; elenco dei tensori NON ternari con i byte;
taglia < --tetto. Uso: c0_scrivi.py --base GGUF --tensori DIR --uscita OUT.gguf [--tetto 9470184864] [--solo-resoconto]
"""
import argparse, glob, hashlib, json, os, re, sys, time
from pathlib import Path
sys.path.insert(0, "@@ROOT:models@@/build/thor-rexppiani2/gguf-py")
import numpy as np
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType as T

FAM = ("attn_q", "attn_k", "attn_v", "attn_output", "attn_qkv", "attn_gate", "ssm_out", "ffn_gate_shexp", "ffn_up_shexp",
       "ffn_down_shexp", "token_embd", "output", "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")
TERNARI = {T.TQ1_0, T.TQ1_0R_1024, getattr(T, "TQ2_0", T.TQ1_0)}
BPB = {T.TQ1_0R_1024: (1024, 224), T.TQ1_0: (256, 54)}


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def piano_extra(nome):
    base = nome.split(".", 2)[-1] if nome.startswith("blk.") else nome
    base = base[: -len(".weight")] if base.endswith(".weight") else base
    m = re.fullmatch(r"(.+?)([234])", base)
    if m and m.group(1) in FAM:
        return True
    return bool(re.search(r"exps[234]?_mappa|_mappa[234]?$|exps[234]_map", base))


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="@@ROOT:models@@/gguf/tony-tern/R2B3m/Thor-T-R2B3m.gguf")
    ap.add_argument("--tensori", required=True)
    ap.add_argument("--uscita", required=True)
    ap.add_argument("--tetto", type=int, default=9_470_184_864)
    ap.add_argument("--nota", default="C0: ospite R2B3m senza piani 2/3 (= TT2 p6 a un piano) + ssm_alpha/beta ternari")
    ap.add_argument("--solo-resoconto", action="store_true")
    a = ap.parse_args()
    man = {}
    for mp in glob.glob(os.path.join(a.tensori, "manifesto*.json")):
        man.update(json.load(open(mp)))
    r = GGUFReader(a.base)
    tolti, piano, prov = [], [], {}
    w = GGUFWriter(path=None, arch="qwen35moe")
    for f in r.fields.values():
        if f.name.startswith(("GGUF.", "split.")):
            continue
        w.add_key_value(f.name, f.contents(), f.types[0], sub_type=f.types[-1] if len(f.types) > 1 else None)
    w.add_string("thor.obiettivo_C", f"{a.nota}. Ospite {a.base} ({os.path.getsize(a.base)} B). Tolti tutti i piani "
                 "aggiuntivi; sostituiti i tensori di " + a.tensori + "; il resto copiato al byte.")
    for t in r.tensors:
        if piano_extra(t.name):
            tolti.append(t.name); continue
        if t.name in man:
            m = man[t.name]
            arr = np.load(os.path.join(a.tensori, t.name + ".npy"), mmap_mode="r")
            assert sha(np.ascontiguousarray(arr).tobytes()) == m["sha256_npy"], t.name
            tipo = T[m["tipo"]]
            ne0, ne1 = m["forma_ggml"]
            assert [int(x) for x in t.shape] == [ne0, ne1], (t.name, t.shape)
            qk, nb = BPB[tipo]
            fb = [ne1, ne0 // qk * nb]
            assert arr.size == fb[0] * fb[1], t.name
            w.add_tensor_info(t.name, fb, np.dtype(np.uint8), int(arr.size), raw_dtype=tipo)
            piano.append((t.name, np.asarray(arr).reshape(fb)))
            prov[t.name] = {"azione": "sostituito", "tipo": tipo.name, "tipo_ospite": t.tensor_type.name,
                            "sha256_byte": m["sha256_npy"], "misure": m.get("misure")}
        else:
            raw = np.asarray(t.data)
            w.add_tensor_info(t.name, list(raw.shape), raw.dtype, int(raw.nbytes), raw_dtype=t.tensor_type)
            piano.append((t.name, raw))
            prov[t.name] = {"azione": "copiato al byte dall'ospite", "tipo": t.tensor_type.name, "sha256_byte": None}
    n_dati = sum(int(np.asarray(d).nbytes) for _, d in piano)
    log(f"tensori {len(piano)} (tolti {len(tolti)} piani aggiuntivi, sostituiti {sum(1 for v in prov.values() if v['azione']=='sostituito')}), "
        f"dati {n_dati} B = {n_dati/2**30:.3f} GiB")
    nonter = {}
    for nome, d in piano:
        tp = prov[nome]["tipo"]
        if T[tp] not in TERNARI:
            fam = (nome.split(".", 2)[-1] if nome.startswith("blk.") else nome)
            fam = ("blk40." if nome.startswith("blk.40.") else "") + fam
            e = nonter.setdefault(fam, [0, 0, tp]); e[0] += 1; e[1] += int(np.asarray(d).nbytes)
    log("NON ternari:", json.dumps(nonter))
    if a.solo_resoconto:
        assert n_dati < a.tetto, "oltre il tetto"
        return
    tmp = a.uscita + ".parziale"
    w.open_output_file(Path(tmp))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
    for i, (nome, d) in enumerate(piano):
        w.write_tensor_data(np.ascontiguousarray(d))
    w.close()
    os.replace(tmp, a.uscita)
    sz = os.path.getsize(a.uscita)
    log(f"scritto {a.uscita}: {sz} B = {sz/2**30:.4f} GiB")
    r2 = GGUFReader(a.uscita)
    base = {t.name: t for t in r.tensors}
    err = []
    names2 = [t.name for t in r2.tensors]
    if names2 != [n for n, _ in piano]:
        err.append("ordine/nomi")
    for t2 in r2.tensors:
        if piano_extra(t2.name):
            err.append(f"piano aggiuntivo rimasto {t2.name}")
        b2 = np.ascontiguousarray(np.asarray(t2.data)).tobytes()
        p = prov[t2.name]
        if p["azione"] == "sostituito":
            if sha(b2) != p["sha256_byte"]: err.append(f"byte {t2.name}")
        else:
            if b2 != np.ascontiguousarray(np.asarray(base[t2.name].data)).tobytes(): err.append(f"copia {t2.name}")
        if t2.tensor_type.name != p["tipo"]: err.append(f"tipo {t2.name}")
    ver = {"ok": not err and sz < a.tetto, "errori": err[:20], "taglia": sz, "tetto": a.tetto, "tolti": tolti,
           "non_ternari": nonter, "sha256_file": None}
    h = hashlib.sha256()
    with open(a.uscita, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 24), b""):
            h.update(blk)
    ver["sha256_file"] = h.hexdigest()
    json.dump(prov, open(a.uscita + ".provenienza.json", "w"), indent=1)
    json.dump(ver, open(a.uscita + ".verifica.json", "w"), indent=1)
    log("VERIFICA", "OK" if ver["ok"] else f"ERRORE {err[:5]}", "sha256", ver["sha256_file"])
    sys.exit(0 if ver["ok"] else 1)


if __name__ == "__main__":
    main()
