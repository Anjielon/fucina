#!/usr/bin/env python3
"""Obiettivo C (9/10): scrittore generale delle varianti a UN piano (C0B, C0T; C0 resta su c0_scrivi.py).
Ospite: R2B3m; si TOLGONO tutti i piani aggiuntivi (come c0_scrivi.piano_extra); i tensori dei manifesti (--tensori, in
ordine: l'ultimo vince) SOSTITUISCONO quelli dell'ospite. Voci del manifesto:
  {"tipo", "forma_ggml" (forma MEMORIZZATA, 1-3 dimensioni), "sha256_npy", "dol": [ne0, ne1, modo] facoltativo}
     -> <dir>/<nome>.npy (byte del tipo);
  {"sorgente_gguf": file, "tensore": nome} -> byte copiati AL BYTE da un altro GGUF (es. Q8_0 del maestro per C0B).
Le voci con 'dol' finiscono nella chiave thor.dequant_on_load ("nome|ne0|ne1|modo"): il motore c0t le dequantizza al
caricamento. --ref-f32 scrive invece quei tensori in F32 (valori dequantizzati, trasformati come fa il motore) senza la
chiave: e' il RIFERIMENTO per la prova d'identita' del motore (stesse uscite attese).
Verifica: riletto, ordine/tipi/byte, nessun piano aggiuntivo, elenco dei NON ternari con i byte, taglia < --tetto."""
import argparse, glob, hashlib, json, os, sys, time
from pathlib import Path
sys.path.insert(0, "@@ROOT:models@@/build/thor-rexppiani2/gguf-py")
sys.path.insert(0, "@@ROOT:models@@/gguf/tony-tern/OBIETTIVO_C/codice")
sys.path.insert(0, "@@ROOT:depot@@/tutto_ternario/codice")
os.environ.setdefault("TT_DEVICE", "cpu")
import numpy as np
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType as T
from c0_scrivi import piano_extra, sha, log, TERNARI

BPB = {T.TQ1_0R_1024: (1024, 224), T.TQ1_0: (256, 54), T.Q8_0: (32, 34),
       T.F32: (1, 4)}   # 11/10: norme F32 piegate da S1 (s1_fedele.py), .npy = byte uint8 dei float32


def dequant(tipo, b, n):
    if tipo == T.TQ1_0R_1024:
        import tt_quant as TQ
        return TQ.dequantize_blocks(np.ascontiguousarray(b).reshape(-1, 224)).reshape(-1)[:n]
    import gguf
    return gguf.quants.dequantize(np.ascontiguousarray(b).reshape(1, -1), tipo).reshape(-1)[:n]


def valori_dol(tipo, b, forma, dol):
    ne0, ne1, modo = dol
    tot = int(np.prod(forma))
    v = dequant(tipo, b, tot).astype(np.float32)
    if modo == "trasposta":
        a, bb = forma[0], int(np.prod(forma[1:]))
        return v.reshape(bb, a).T.copy().reshape(-1)          # [canale][tap]
    v = v[: ne0 * ne1]
    if modo == "tronca_negexp":
        v = (-np.exp(v.astype(np.float64))).astype(np.float32)   # come il motore (exp in double): bit-identico
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="@@ROOT:models@@/gguf/tony-tern/R2B3m/Thor-T-R2B3m.gguf")
    ap.add_argument("--tensori", nargs="+", required=True)
    ap.add_argument("--uscita", required=True)
    ap.add_argument("--tetto", type=int, default=9_470_184_864)
    ap.add_argument("--nota", required=True)
    ap.add_argument("--ref-f32", action="store_true")
    a = ap.parse_args()
    man = {}
    for d in a.tensori:
        for mp in sorted(glob.glob(os.path.join(d, "manifesto*.json"))):
            for k, v in json.load(open(mp)).items():
                man[k] = dict(v, _dir=d)
    r = GGUFReader(a.base)
    w = GGUFWriter(path=None, arch="qwen35moe")
    for f in r.fields.values():
        if f.name.startswith(("GGUF.", "split.")):
            continue
        w.add_key_value(f.name, f.contents(), f.types[0], sub_type=f.types[-1] if len(f.types) > 1 else None)
    w.add_string("thor.obiettivo_C", f"{a.nota}. Ospite {a.base}; tolti i piani aggiuntivi; sostituiti da "
                 + ", ".join(a.tensori) + ("; RIFERIMENTO F32 dei tensori dol" if a.ref_f32 else ""))
    piano, prov, tolti, dol = [], {}, [], []
    altri = {}
    for t in r.tensors:
        if piano_extra(t.name):
            tolti.append(t.name); continue
        m = man.get(t.name)
        if m is None:
            raw = np.asarray(t.data)
            w.add_tensor_info(t.name, list(raw.shape), raw.dtype, int(raw.nbytes), raw_dtype=t.tensor_type)
            piano.append((t.name, raw)); prov[t.name] = {"azione": "copiato al byte dall'ospite", "tipo": t.tensor_type.name}
            continue
        if "sorgente_gguf" in m:
            if m["sorgente_gguf"] not in altri:
                altri[m["sorgente_gguf"]] = {x.name: x for x in GGUFReader(m["sorgente_gguf"]).tensors}
            ts = altri[m["sorgente_gguf"]][m["tensore"]]
            assert [int(x) for x in ts.shape] == [int(x) for x in t.shape], t.name
            raw = np.asarray(ts.data)
            w.add_tensor_info(t.name, list(raw.shape), raw.dtype, int(raw.nbytes), raw_dtype=ts.tensor_type)
            piano.append((t.name, raw))
            prov[t.name] = {"azione": f"copiato al byte da {m['sorgente_gguf']}", "tipo": ts.tensor_type.name, "sha256_byte": sha(raw.tobytes())}
            continue
        tipo = T[m["tipo"]]; forma = [int(x) for x in m["forma_ggml"]]
        arr = np.load(os.path.join(m["_dir"], t.name + ".npy"), mmap_mode="r")
        assert sha(np.ascontiguousarray(arr).tobytes()) == m["sha256_npy"], t.name
        if not m.get("dol"):
            assert forma == [int(x) for x in t.shape], (t.name, forma, t.shape)
        if m.get("dol") and a.ref_f32:
            v = valori_dol(tipo, np.asarray(arr), forma, m["dol"])
            ne0, ne1 = m["dol"][0], m["dol"][1]
            v = v.reshape(ne1, ne0) if ne1 > 1 else v.reshape(ne0)
            w.add_tensor_info(t.name, list(v.shape), v.dtype, int(v.nbytes), raw_dtype=T.F32)
            piano.append((t.name, v)); prov[t.name] = {"azione": "riferimento F32 del tensore dol", "tipo": "F32",
                                                       "sha256_byte": sha(v.tobytes()), "dol": m["dol"]}
            continue
        qk, nb = BPB[tipo]
        fb = list(reversed(forma[1:])) + [forma[0] // qk * nb]
        assert arr.size == int(np.prod(fb)), (t.name, arr.shape, fb)
        w.add_tensor_info(t.name, fb, np.dtype(np.uint8), int(arr.size), raw_dtype=tipo)
        piano.append((t.name, np.asarray(arr).reshape(fb)))
        prov[t.name] = {"azione": "sostituito", "tipo": tipo.name, "tipo_ospite": t.tensor_type.name, "sha256_byte": m["sha256_npy"],
                        "forma_memorizzata": forma, "dol": m.get("dol"), "misure": m.get("misure")}
        if m.get("dol"):
            dol.append(f"{t.name}|{m['dol'][0]}|{m['dol'][1]}|{m['dol'][2]}")
    if dol:
        w.add_array("thor.dequant_on_load", dol)
    n_dati = sum(int(np.asarray(d).nbytes) for _, d in piano)
    nonter = {}
    for nome, d in piano:
        if T[prov[nome]["tipo"]] not in TERNARI:
            fam = ("blk40." if nome.startswith("blk.40.") else "") + (nome.split(".", 2)[-1] if nome.startswith("blk.") else nome)
            e = nonter.setdefault(fam, [0, 0, prov[nome]["tipo"]]); e[0] += 1; e[1] += int(np.asarray(d).nbytes)
    log(f"tensori {len(piano)} (tolti {len(tolti)}, dol {len(dol)}), dati {n_dati} B = {n_dati/2**30:.3f} GiB")
    log("NON ternari:", json.dumps(nonter))
    tmp = a.uscita + ".parziale"
    w.open_output_file(Path(tmp)); w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
    for nome, d in piano:
        w.write_tensor_data(np.ascontiguousarray(d))
    w.close(); os.replace(tmp, a.uscita)
    sz = os.path.getsize(a.uscita)
    r2 = GGUFReader(a.uscita); err = []
    if [t.name for t in r2.tensors] != [n for n, _ in piano]:
        err.append("ordine/nomi")
    pdict = dict(piano)
    for t2 in r2.tensors:
        if piano_extra(t2.name):
            err.append(f"piano aggiuntivo {t2.name}")
        if np.ascontiguousarray(np.asarray(t2.data)).tobytes() != np.ascontiguousarray(pdict[t2.name]).tobytes():
            err.append(f"byte {t2.name}")
        if t2.tensor_type.name != prov[t2.name]["tipo"]:
            err.append(f"tipo {t2.name}")
    h = hashlib.sha256()
    with open(a.uscita, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 24), b""):
            h.update(blk)
    ver = {"ok": not err and sz < a.tetto, "errori": err[:20], "taglia": sz, "tetto": a.tetto, "tolti": len(tolti),
           "dequant_on_load": dol, "non_ternari": nonter, "sha256_file": h.hexdigest()}
    json.dump(prov, open(a.uscita + ".provenienza.json", "w"), indent=1)
    json.dump(ver, open(a.uscita + ".verifica.json", "w"), indent=1)
    log(f"scritto {a.uscita}: {sz} B; VERIFICA", "OK" if ver["ok"] else f"ERRORE {err[:5]}")
    sys.exit(0 if ver["ok"] else 1)


if __name__ == "__main__":
    main()
