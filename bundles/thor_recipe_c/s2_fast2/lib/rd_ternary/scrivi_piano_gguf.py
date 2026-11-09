"""Da piano.json a un GGUF che il motore (thor-llama) carica.

Copia il GGUF sorgente BYTE PER BYTE e ricodifica soltanto i tensori che il piano cambia
(con gli stessi codificatori esatti di gguf_piani: TQ1_0 e TQ1_0R_1024), aggiungendo o
togliendo i tensori di piano (base, base2, base3, ...; esperti: *_exps2 e, se il 2° piano
copre solo alcuni esperti, la mappa *_exps2_map del ramo cloud/piani-variabili).
Metadati (KV) copiati tali e quali, testa MTP compresa; informazioni dei tensori e offset
ricalcolati con lo stesso allineamento. Piano nullo -> file identico (sha256 uguale).

I piani nuovi si RICALCOLANO col codice del menu (piano.materialize: deterministico, con
controllo di byte e distorsione contro il piano) oppure si leggono da una cartella gia'
materializzata (rd_L*.safetensors di materialize(out_dir=...)).

Uso tipico (stessa macchina del dry run: teacher bf16 + imatrix):
  python3 -m rd_ternary.scrivi_piano_gguf --gguf Thor-T-R2B3m.gguf --piano piano_globale.json \\
      --out Thor-T-R2B3m-rd.gguf --device auto
  # piani gia' materializzati:
  python3 -m rd_ternary.scrivi_piano_gguf --gguf ... --piano ... --materializzati DIR --out ...
  # solo il resoconto (byte, caricabilita'), senza scrivere:
  python3 -m rd_ternary.scrivi_piano_gguf --gguf ... --piano ... --out X --solo-resoconto
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import gguf_grezzo as gg
from .formato import TensorKey
from .gguf_piani import FORMATS, G64_OF, GGUFPlaneSource, decode_rows, encode_rows, gguf_name
from .motore import (BASE_KIND, EXPERT_BASES, MAPPE, N_MAIN_DEFAULT, PROFILI, base_of, check_plan_loadable,
                     max_planes)

MAX_FILE_BYTES = 9 * 2 ** 30
T_F32 = 0

NewPlanes = Dict[TensorKey, Tuple[np.ndarray, np.ndarray]]   # chiave -> (piani int8 [K,o,i], scale fp16 [K,o,ng])


def n_main_layers(src: gg.GGUFFile, default: int = N_MAIN_DEFAULT) -> int:
    arch = src.kv.get("general.architecture")
    bc = src.kv.get(f"{arch}.block_count")
    if bc is None:
        return default
    return int(bc) - int(src.kv.get(f"{arch}.nextn_predict_layers", 0) or 0)


def _group_names(layer: int, base: str, n_planes: int, expert: bool) -> List[str]:
    names = [gguf_name(layer, base, k) for k in range(1, n_planes + 1)]
    if expert:
        names.append(f"blk.{layer}.{base}2_map")
    return names


class PianoNonCaricabile(ValueError):
    pass


def _is_g64(scales: np.ndarray, in_: int) -> bool:
    return int(np.asarray(scales).shape[-1]) * 64 == in_


def _to_g64(scales: np.ndarray, in_: int) -> np.ndarray:
    """Scale del tipo madre -> scala 64 (ripetute): ricostruzione IDENTICA, solo piu' byte."""
    s = np.asarray(scales)
    rep = in_ // 64 // int(s.shape[-1])
    return s if rep == 1 else np.repeat(s, rep, axis=-1)


def _row_to_g64(raw: np.ndarray, tid: int, in_: int) -> np.ndarray:
    """Righe grezze di un piano del tipo madre -> righe del tipo a scala 64 (stessi trit)."""
    p, s = decode_rows(np.asarray(raw), tid, in_)
    return encode_rows(p, _to_g64(s, in_).astype(np.float16), G64_OF[tid])


def build_tensors(src_path: str, new: NewPlanes, profilo: str = "piani-variabili", forza: bool = False,
                  gguf_py: Optional[str] = None) -> Tuple[gg.GGUFFile, List[gg.OutTensor], Dict]:
    """Lista dei tensori in uscita + resoconto. Solleva PianoNonCaricabile se il risultato
    non si carica col profilo indicato (forza=True lo scrive comunque)."""
    if profilo not in PROFILI:
        raise ValueError(profilo)
    src = gg.read_gguf(src_path)
    info = src.by_name()
    n_main = n_main_layers(src)
    n_expert_kv = src.kv.get(f"{src.kv.get('general.architecture')}.expert_count")
    ps = GGUFPlaneSource(src_path, include_embeddings=True, gguf_py=gguf_py)
    by_entry: Dict[Tuple[int, str], Dict[int, Tuple[np.ndarray, np.ndarray]]] = defaultdict(dict)
    for k, (p, s) in new.items():
        by_entry[(k.layer, base_of(k))][k.expert] = (np.asarray(p), np.asarray(s))
    problems: List[str] = []
    replace: Dict[str, List[gg.OutTensor]] = {}      # nome del piano 1 -> gruppo in uscita
    skip = set()
    rep = {"densi_cambiati": 0, "esperti_cambiati": 0, "piani_aggiunti": 0, "piani_tolti": 0,
           "mappe": 0, "esperti_con_2_piani": {}, "riempiti_a_zero": 0}
    for (layer, base), per in sorted(by_entry.items()):
        e = ps._entries.get((layer, base))
        if e is None:
            raise KeyError(f"blk.{layer}.{base}: tensore ternario assente nel sorgente")
        tid, fmt = e.fmt.type_id, e.fmt
        src_tid = tid
        # 6/10: se un tensore del gruppo arriva con scale a 64 (opzioni g64), TUTTO il gruppo passa
        # al tipo a scala 64 (unico per tensore GGUF); gli altri piani/esperti si convertono senza perdita
        g64 = any(_is_g64(s, e.in_) for (_p, s) in per.values())
        if g64:
            if tid not in G64_OF:
                raise PianoNonCaricabile(f"blk.{layer}.{base}: tipo {fmt.name} senza variante a scala 64")
            if profilo != "scala-g64":
                problems.append(f"blk.{layer}.{base}: scala 64 richiede il motore 'scala-g64' (non {profilo})")
            tid = G64_OF[tid]
            rep.setdefault("tensori_g64", []).append(f"blk.{layer}.{base}")
        p1 = info[e.tensors[0].name]
        old_names = _group_names(layer, base, len(e.tensors), bool(e.n_experts))
        skip.update(n for n in old_names if n in info)
        lim = max_planes(layer, base, n_main)
        group: List[gg.OutTensor] = []
        if not e.n_experts:
            (planes, scales), = per.values()
            K = int(planes.shape[0])
            if K < 1:
                raise ValueError(f"blk.{layer}.{base}: zero piani")
            if K > lim:
                problems.append(f"{gguf_name(layer, base)}: {K} piani, il motore ne carica {lim}")
            if tuple(planes.shape[1:]) != (e.out, e.in_):
                raise ValueError(f"{gguf_name(layer, base)}: forma {planes.shape[1:]} != {(e.out, e.in_)}")
            if g64:
                scales = _to_g64(scales, e.in_)
            for j in range(K):
                rows = encode_rows(planes[j].astype(np.int8), scales[j].astype(np.float16), tid)
                group.append(gg.OutTensor(gguf_name(layer, base, j + 1), p1.dims, tid, data=rows))
            rep["densi_cambiati"] += 1
            rep["piani_aggiunti"] += max(0, K - len(e.tensors))
            rep["piani_tolti"] += max(0, len(e.tensors) - K)
        else:
            E = e.n_experts
            rb = p1.nbytes // (E * e.out)
            t1 = gg.read_tensor(src, p1).reshape(E, e.out, rb)
            rows2: Dict[int, np.ndarray] = {}
            for x in range(E):                       # 2° piano gia' presente e non toccato
                if x not in per and e.n_planes_of(x) == 2:
                    rows2[x] = np.asarray(ps.raw_rows(TensorKey(layer, x, BASE_KIND[base]), 1)).copy()
                    if g64:
                        rows2[x] = _row_to_g64(rows2[x], src_tid, e.in_)
            if g64:                                  # esperti non toccati: stessi trit, scale ripetute
                conv = np.empty((E, e.out, (e.in_ // FORMATS[tid].block) * FORMATS[tid].block_bytes), np.uint8)
                for x in range(E):
                    if x not in per:
                        conv[x] = _row_to_g64(t1[x], src_tid, e.in_)
                t1 = conv
                mixed = [x for x, (_p, s) in per.items() if not _is_g64(s, e.in_)]
                if mixed or len(per) < E:
                    rep.setdefault("esperti_convertiti_a_g64", {})[f"blk.{layer}.{base}"] = E - len(per) + len(mixed)
            for x, (planes, scales) in per.items():
                if g64:
                    scales = _to_g64(scales, e.in_)
                K = int(planes.shape[0])
                if K > lim or K < 1:
                    problems.append(f"blk.{layer}.{base} esperto {x}: {K} piani, il motore ne carica 1..{lim}")
                    K = min(max(K, 1), lim)
                t1[x] = encode_rows(planes[0].astype(np.int8), scales[0].astype(np.float16), tid)
                if K >= 2:
                    rows2[x] = encode_rows(planes[1].astype(np.int8), scales[1].astype(np.float16), tid)
                rep["esperti_cambiati"] += 1
            group.append(gg.OutTensor(p1.name, p1.dims, tid, data=t1))
            S2 = sorted(rows2)
            rep["esperti_con_2_piani"][f"blk.{layer}.{base}"] = len(S2)
            if S2:
                name2 = gguf_name(layer, base, 2)
                if len(S2) == E:
                    t2 = np.stack([rows2[x] for x in range(E)])
                    group.append(gg.OutTensor(name2, (p1.dims[0], p1.dims[1], E), tid, data=t2))
                elif profilo in MAPPE:
                    t2 = np.stack([rows2[x] for x in S2])
                    m = np.full((E,), -1.0, dtype=np.float32)
                    m[S2] = np.arange(len(S2), dtype=np.float32)
                    group.append(gg.OutTensor(name2, (p1.dims[0], p1.dims[1], len(S2)), tid, data=t2))
                    group.append(gg.OutTensor(f"blk.{layer}.{base}2_map", (1, E), T_F32, data=m))
                    rep["mappe"] += 1
                else:   # "oggi": il 2° piano deve coprire tutti gli esperti -> zeri (scala 0) per gli altri
                    zero = encode_rows(np.zeros((e.out, e.in_), np.int8),
                                       np.zeros((e.out, e.in_ // FORMATS[tid].group), np.float16), tid)
                    t2 = np.stack([rows2.get(x, zero) for x in range(E)])
                    group.append(gg.OutTensor(name2, (p1.dims[0], p1.dims[1], E), tid, data=t2))
                    rep["riempiti_a_zero"] += E - len(S2)
                    problems.append(f"blk.{layer}.{base}: 2° piano su {len(S2)}/{E} esperti — in 'oggi' "
                                    f"si scrive pieno con {E - len(S2)} esperti a zero (byte sprecati)")
            n_old = len(e.tensors)
            n_new = 2 if S2 else 1
            rep["piani_aggiunti"] += max(0, n_new - n_old)
            rep["piani_tolti"] += max(0, n_old - n_new)
        replace[p1.name] = group
    real = [p for p in problems if "byte sprecati" not in p]
    if real and not forza:
        raise PianoNonCaricabile("; ".join(real[:10]) + (f" (+{len(real) - 10})" if len(real) > 10 else ""))
    out: List[gg.OutTensor] = []
    for t in src.tensors:
        if t.name in replace:
            out.extend(replace[t.name])
        elif t.name not in skip:
            out.append(gg.OutTensor(t.name, t.dims, t.type_id, src_offset=src.data_start + t.offset))
    names = [t.name for t in out]
    if len(set(names)) != len(names):
        raise ValueError("nomi di tensore duplicati in uscita")
    rep.update({"avvisi": problems, "n_main": n_main, "n_expert": n_expert_kv, "profilo": profilo})
    return src, out, rep


def write_plan_gguf(src_path: str, out_path: str, new: NewPlanes, profilo: str = "piani-variabili",
                    max_bytes: int = MAX_FILE_BYTES, forza: bool = False, note: Optional[Dict] = None,
                    dry: bool = False, gguf_py: Optional[str] = None) -> Dict:
    """Scrive il GGUF con i piani nuovi. note: dizionario salvato nel KV `thor.rd_piano`
    (solo se qualcosa cambia: col piano nullo il file resta identico al sorgente)."""
    src, out, rep = build_tensors(src_path, new, profilo, forza, gguf_py)
    extra = []
    # R022b: scrittura in due giri (--strati): il sorgente del 2° giro ha gia' `thor.rd_piano`; un secondo, con lo stesso nome,
    # renderebbe il file illeggibile (gguf-py: «Duplicate»). La nota del 1° giro resta.
    if new and note is not None and "thor.rd_piano" not in src.kv:
        extra.append(gg.kv_string("thor.rd_piano", json.dumps(note, ensure_ascii=False, sort_keys=True)))
    size = gg.predicted_size(src, out, extra)
    rep.update({"bytes_sorgente": src.file_size, "bytes_uscita": size, "delta_bytes": size - src.file_size,
                "max_bytes": max_bytes, "entro_tetto": size <= max_bytes, "n_tensori": len(out),
                "n_tensori_sorgente": len(src.tensors)})
    if size > max_bytes and not forza:
        raise PianoNonCaricabile(f"file di {size} byte oltre il tetto {max_bytes} ({size / 2**30:.3f} GiB)")
    if dry:
        return rep
    w = gg.write_gguf(src, out, out_path, extra)
    if w["bytes"] != size:
        raise RuntimeError(f"scritti {w['bytes']} byte, previsti {size}")
    rep.update(w)
    return rep


def verify_written(out_path: str, new: NewPlanes, gguf_py: Optional[str] = None) -> Dict:
    """Rilegge il GGUF scritto e confronta, bit per bit, piani e scale dei tensori cambiati."""
    ps = GGUFPlaneSource(out_path, include_embeddings=True, gguf_py=gguf_py)
    bad = []
    for k, (p, s) in new.items():
        tt = ps.load(k)
        ok = (tuple(tt.planes.shape) == tuple(p.shape) and np.array_equal(tt.planes.numpy(), p.astype(np.int8))
              and np.array_equal(tt.scales.numpy().view(np.uint16), np.asarray(s, np.float16).view(np.uint16)))
        if not ok:
            bad.append(k.name())
    return {"verificati": len(new), "diversi": bad}


# --------------------------------------------------------------------------- piano -> piani nuovi
def keys_to_write(plan: Dict, scale_cur: str = "sorgente") -> List[TensorKey]:
    ks = []
    for t in plan["tensors"]:
        if t["option"].startswith("+s"):
            raise PianoNonCaricabile(f"{t['key']}: piano sparso {t['option']} — il motore non ha il formato")
        if t["option"] != "cur" or scale_cur == "rifit":
            ks.append(TensorKey.parse(t["key"]))
    return ks


def planes_from_materialized(dir_: str, keys: Sequence[TensorKey]) -> NewPlanes:
    from safetensors.numpy import load_file
    cache: Dict[int, Dict[str, np.ndarray]] = {}
    out: NewPlanes = {}
    for k in keys:
        if k.layer not in cache:
            p = os.path.join(dir_, f"rd_L{k.layer:03d}.safetensors")
            cache[k.layer] = load_file(p) if os.path.exists(p) else {}
        d = cache[k.layer]
        n = k.name()
        if f"{n}.sparse" in d:
            raise PianoNonCaricabile(f"{n}: piano sparso nei materializzati")
        if f"{n}.planes" not in d:
            raise KeyError(f"{n}: assente in {dir_}")
        out[k] = (d[f"{n}.planes"], d[f"{n}.scales"])
    return out


def planes_from_sources(plan: Dict, keys: Sequence[TensorKey], ps, ts, fs, device: str = "cpu",
                        batch: int = 8, max_batch_weights: Optional[int] = 8 * 1024 * 1024,
                        max_batch_bytes: Optional[int] = None, rel_tol: float = 1e-4) -> Tuple[NewPlanes, Dict]:
    """Ricalcola le opzioni scelte (piano.materialize) e controlla byte e distorsione."""
    from .piano import materialize
    want = {k.name() for k in keys}
    sub = dict(plan)
    sub["tensors"] = [t for t in plan["tensors"] if t["key"] in want]
    r = materialize(sub, ps, ts, fs, device=device, batch=batch, rel_tol=rel_tol,
                    max_batch_weights=max_batch_weights, max_batch_bytes=max_batch_bytes)
    out: NewPlanes = {}
    for k in keys:
        d = r["tensors"][k.layer]
        out[k] = (d[f"{k.name()}.planes"].numpy(), d[f"{k.name()}.scales"].numpy())
    check = {"materializzati": r["n"], "bytes_mismatch": len(r["bytes_mismatch"]),
             "distortion_ok": bool(r["distortion_ok"]), "max_rel_err": r["max_rel_err_distortion"],
             "encoder_mismatch": len(r["encoder_mismatch"])}
    return out, check


def drop_unchanged(src_path: str, new: NewPlanes, gguf_py: Optional[str] = None) -> NewPlanes:
    """Toglie le chiavi i cui piani ricodificati sono identici al sorgente (es. 'cur' senza rifit)."""
    ps = GGUFPlaneSource(src_path, include_embeddings=True, gguf_py=gguf_py)
    keep: NewPlanes = {}
    for k, (p, s) in new.items():
        tt = ps.load(k)
        same = (tuple(tt.planes.shape) == tuple(p.shape) and np.array_equal(tt.planes.numpy(), p)
                and np.array_equal(tt.scales.numpy().view(np.uint16), np.asarray(s, np.float16).view(np.uint16)))
        if not same:
            keep[k] = (p, s)
    return keep


def estimate_size(plan: Dict, src: gg.GGUFFile, profilo: str, max_bytes: int = MAX_FILE_BYTES) -> Dict:
    """Stima ANALITICA (senza ricalcolare i piani) della taglia del file: i byte dei tensori
    del piano sono esatti per costruzione; si aggiungono le mappe e un tetto per le voci
    nuove della tabella dei tensori (nome + forma + tipo + offset) e l'allineamento."""
    delta = sum(int(t["delta_bytes"]) for t in plan["tensors"])
    n_exp = int(src.kv.get(f"{src.kv.get('general.architecture')}.expert_count", 256) or 256)
    k2: Dict[Tuple[int, str], int] = defaultdict(int)
    planes: Dict[Tuple[int, str], int] = {}
    for t in plan["tensors"]:
        k = TensorKey.parse(t["key"])
        K = int(t["n_planes"]) + {"+1": 1, "-1": -1, "+1g64": 1}.get(t["option"], 0)
        if k.expert >= 0:
            k2[(k.layer, base_of(k))] += int(K >= 2)
        else:
            planes[(k.layer, k.kind)] = K - int(t["n_planes"])
    maps = sum(1 for v in k2.values() if 0 < v < n_exp) if profilo in MAPPE else 0
    new_tensors = sum(max(0, v) for v in planes.values()) + sum(1 for v in k2.values() if v) + maps
    upper = delta + maps * (4 * n_exp + src.alignment) + new_tensors * (128 + src.alignment) + 4096
    size = src.file_size + upper
    return {"stima": True, "bytes_sorgente": src.file_size, "delta_bytes_tensori": delta,
            "mappe": maps, "tensori_nuovi_max": new_tensors, "bytes_uscita_max": size,
            "delta_bytes_max": upper, "max_bytes": max_bytes, "entro_tetto": size <= max_bytes,
            "margine_gib": round((max_bytes - size) / 2 ** 30, 4)}


def main(argv=None) -> Dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True, help="GGUF sorgente (quello del dry run)")
    ap.add_argument("--piano", required=True, help="piano.json (dry_run_gguf / unisci)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--motore", choices=PROFILI, default="oggi",
                    help="oggi = velocita-ternario/esperti-recupero; piani-variabili = cloud/piani-variabili")
    ap.add_argument("--max-bytes", type=int, default=MAX_FILE_BYTES, help="tetto del file (default 9 GiB)")
    ap.add_argument("--materializzati", default=None, help="cartella con rd_L*.safetensors (materialize out_dir)")
    ap.add_argument("--teacher", nargs="+", default=None)
    ap.add_argument("--imatrix", default=None)
    ap.add_argument("--fisher-dir", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--scale-cur", choices=("sorgente", "rifit"), default="sorgente",
                    help="tensori 'cur': scale del sorgente (default) o rifittate come nel menu")
    ap.add_argument("--forza", action="store_true", help="scrive anche se non caricabile/oltre il tetto")
    ap.add_argument("--solo-resoconto", action="store_true", help="non scrive: byte e caricabilita'")
    ap.add_argument("--no-verifica", action="store_true")
    ap.add_argument("--gguf-py", default=None)
    ap.add_argument("--strati", default=None,
                    help="R022b: scrive SOLO i tensori di questi strati (es. 0-19); gli altri restano come nel --gguf. Per spezzare "
                         "la scrittura in due giri da <= 0,5 h: giro 1 `--strati 0-19 --out A.gguf`, giro 2 `--gguf A.gguf "
                         "--strati 20-40 --out finale.gguf` (stesso --piano). Identico a un giro solo (provato su CPU)")
    a = ap.parse_args(argv)

    from .piano import load_plan
    t0 = time.time()
    plan = load_plan(a.piano)
    src = gg.read_gguf(a.gguf)
    n_main = n_main_layers(src)
    errs = check_plan_loadable(plan, a.motore, n_main,
                               int(src.kv.get(f"{src.kv.get('general.architecture')}.expert_count", 256) or 256))
    if errs and not a.forza and not a.solo_resoconto:
        raise SystemExit("piano non caricabile col motore '%s':\n  %s" % (a.motore, "\n  ".join(errs[:20])))
    keys = keys_to_write(plan, a.scale_cur)
    if a.strati:
        lo, _, hi = a.strati.partition("-")
        lo, hi = int(lo), int(hi or lo)
        keys = [k for k in keys if lo <= k.layer <= hi]
        plan = dict(plan)
        plan["tensors"] = [t for t in plan["tensors"] if lo <= TensorKey.parse(t["key"]).layer <= hi or t["option"] == "cur"]
    check = {}
    if a.solo_resoconto and not a.materializzati:
        rep = estimate_size(plan, src, a.motore, a.max_bytes)
        cnt: Dict[str, int] = defaultdict(int)
        for t in plan["tensors"]:
            cnt[t["option"]] += 1
        rep.update({"opzioni": dict(cnt), "non_caricabile": errs[:20], "t_s": round(time.time() - t0, 1)})
        print(json.dumps(rep, indent=1))
        return rep
    if not keys:
        new: NewPlanes = {}
    elif a.materializzati:
        new = planes_from_materialized(a.materializzati, keys)
    else:
        from .dispositivo import resolve_device
        from .dry_run_gguf import IMATRIX, TEACHER
        from .gguf_piani import GGUFTeacherSource, ImatrixFisherSource
        from .sorgenti import SafetensorsFisherSource
        dev = resolve_device(a.device)
        ps = GGUFPlaneSource(a.gguf, layers=sorted({k.layer for k in keys}), include_embeddings=True,
                             gguf_py=a.gguf_py)
        ts = GGUFTeacherSource(a.teacher or TEACHER, gguf_py=a.gguf_py)
        if a.fisher_dir:
            fs = SafetensorsFisherSource(a.fisher_dir)
        else:
            fs = ImatrixFisherSource(a.imatrix or IMATRIX, out_of=lambda k: ps.info(k).out,
                                     in_of=lambda k: ps.info(k).in_, gguf_py=a.gguf_py)
        on_gpu = dev.type == "cuda"
        new, check = planes_from_sources(plan, keys, ps, ts, fs, device=str(dev),
                                         batch=a.batch or (256 if on_gpu else 8),
                                         max_batch_weights=None if on_gpu else 8 * 1024 * 1024)
        if (check["bytes_mismatch"] or (not check["distortion_ok"] and not (check.get("encoder_mismatch", 1) == 0 and check.get("max_rel_err", 1) < float(os.getenv("RD_TOLL_DIST", "0.05"))))) and not a.forza:   # casa 5/10: byte e codificatore esatti, scarto solo sulla distorsione PREVISTA (<5%): si scrive
            raise SystemExit(f"rimaterializzazione diversa dal piano: {check}")
    if a.scale_cur == "rifit" and new:
        new = drop_unchanged(a.gguf, new, a.gguf_py)
    note = {"piano": os.path.basename(a.piano), "sorgente": os.path.basename(a.gguf), "motore": a.motore,
            "tensori_cambiati": len(new), "pred_distortion": plan.get("pred_distortion_total"),
            "pred_distortion_base": plan.get("pred_distortion_base"), "scale_cur": a.scale_cur}
    rep = write_plan_gguf(a.gguf, a.out, new, a.motore, a.max_bytes, a.forza, note, dry=a.solo_resoconto,
                          gguf_py=a.gguf_py)
    rep["rimaterializzazione"] = check
    if not a.solo_resoconto and not a.no_verifica and new:
        rep["verifica"] = verify_written(a.out, new, a.gguf_py)
    rep["t_s"] = round(time.time() - t0, 1)
    rep.pop("esperti_con_2_piani", None) if len(rep.get("esperti_con_2_piani", {})) > 50 else None
    print(json.dumps(rep, indent=1, default=str))
    return rep


if __name__ == "__main__":
    r = main()
    sys.exit(0 if not r.get("verifica", {}).get("diversi") else 1)
