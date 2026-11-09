#!/usr/bin/env python3
"""e2e_esporta — export del migliore della fase E2E (progetto `e2e_progetto_scelto.md` §2.7). 26/9/2026.
File NUOVO: nessun file esistente e' modificato; si importano in SOLA LETTURA `e2e_codici_scale`
(`patcha_scale`, `verifica_patch`, layout), `e2e_allena` (lettura dell'istantanea), `e2e_studente` (ricarica).

## Cosa fa, in ordine (ogni passo solleva al primo scarto: meglio fermarsi che scrivere un file sbagliato)
1. **Copia** del GGUF sorgente in una cartella che NON sia `/tmp` (RAM), `/dev/shm`, `/run` ne' il disco di
   sistema `/` (§2.7); spazio libero controllato; il sorgente non si apre MAI in scrittura (sha256 prima e dopo).
2. **Riscrittura sul posto dei soli campi d** (`e2e_codici_scale.patcha_scale`, memmap r+): le scale fp16
   ESATTE dell'istantanea del migliore (`e2e_allena.salva_istantanea`: i bit che l'avanti usava quando e' stato
   valutato, una sola fonte, §2.2), nell'ordine dei blocchi del GGUF.
3. **Verifica del file intero** (`verifica_patch`): stessa taglia, OGNI byte fuori dai campi d identico (il primo
   diverso solleva), sha256 del flusso senza d e dei soli codici.
4. **Rilettura** con GGUFReader: stessi nomi, tipi, forme, offset; per ogni tensore patchato: d del file ==
   istantanea AL BIT, nessuno zero nuovo, segni conservati; istogramma di |d/d0 - 1|.
5. **Dequantizzazione del CODICE C di llama.cpp** (`dequantize_row_tq1_0` / `_tq1_0r_1024` della build k1024,
   ctypes, come la prova R7) su esperti campione di ogni tensore == trit(byte del SORGENTE) x d nuove, al bit.
6. **Ricarica**: studenti costruiti SOLO dal file esportato (theta = 0), strato per strato: scale == istantanea
   al bit e, sullo stesso ingresso, uscita dello strato == quella dello studente del sorgente con theta del
   migliore, al bit. (Strato per strato: ogni strato e' una funzione deterministica del suo ingresso e dei
   suoi pesi, e tutto il resto del file e' identico al byte dal punto 3: l'uguaglianza passa al modello.)
6b. **Tensori F32 allenati** (27/9, `--allena-f32` di e2e_allena: `f32.npz` dell'istantanea): dopo 3-5, i nuovi
   valori fp32 scritti AL POSTO dei vecchi (stesso nome, tipo F32, forma, offset: solo quei `n_bytes`), poi OGNI
   byte fuori da quelle regioni con la stessa sha256 di prima (`verifica_f32`), le regioni == i valori attesi al byte e
   la rilettura GGUFReader == attesi al bit; la ricarica (6) confronta con lo studente del sorgente coi valori F32.
6c. **Adattatori LoRA** (27/9, `--lora-nonesperti` di e2e_allena: `lora.npz` dell'istantanea): W_file + s B A
   (Delta W riportato nell'ordine del file) riquantizzato nello STESSO tipo e taglia (`--lora-modo griglia`: scale
   ferme, codici riarrotondati; `ggml`: quantizzatore di llama.cpp da capo), scritto al posto dei soli byte di quei
   tensori, poi OGNI altro byte con la stessa sha256 e rilettura == attesi; misure di sopravvivenza per tensore
   (`e2e_lora.misura_fusione`) e, nella ricarica, uscita del file == studente del sorgente coi pesi riquantizzati AL
   BIT e quota dell'effetto allenato sull'uscita dello strato che resta dopo la riquantizzazione ("lora_uscita").
   27/9 LoRA-GA (`--lora-init ga`: "|A0", "|B0" in lora.npz, meta lora.init = "ga"): Delta W = s (B A - B0 A0)
   (`e2e_lora.delta_file_da`), zero esatto per l'istantanea del passo 0.
   28/9 `--lora-ripristina` di e2e_allena: un tensore con Delta W = 0 ESATTO tiene i byte del sorgente in ENTRAMBI i
   modi (`e2e_lora.byte_riquantizzati`, "delta_nullo" nelle misure): con Delta W = 0 dappertutto il GGUF e' quello di
   una corsa senza LoRA, al byte (prova R4 di e2e_test_lora_ripristina.py). Con `ferma` l'istantanea non ha lora.npz.
6d. **Densi ternari** (28/9, `--densi-ternari` di e2e_allena): le loro scale sono nell'istantanea come quelle degli
   esperti (tensori 2D TQ1_0R_1024 / TQ1_0): stessa riscrittura dei campi d, stessa verifica del file intero, codice C
   sull'intero tensore (un esperto solo), ricarica con studenti costruiti coi densi dell'istantanea
   (`e2e_studente.famiglie_dense_da_nomi`). I codici dei densi si esportano con `e2e_pv_codici esporta` come quelli degli
   esperti (registro f >= 3).
7. **Provenienza** in `<gguf>.provenienza.json` accanto al GGUF (l'intestazione NON si tocca: una chiave nuova
   sposterebbe tutti gli offset, §2.7): sha256 di sorgente e copia, istantanea (passo, valutazione), corsa
   (config, G1, sorgente del teacher), esiti di 3-6, sha256 dei moduli e2e_*.py, commit della fucina.

Uso:
    python3 e2e_esporta.py --sorgente @@ROOT:models@@/gguf/tony-tern/Thor-v2-ThorT-scor150.gguf \\
        --corsa @@ROOT:lavoro@@/e2e/allena/NOME [--istantanea migliore_000128] \\
        --uscita @@ROOT:models@@/gguf/tony-tern/Thor-v2-ThorT-scor150-E2E.gguf [--strati-ricarica 0,3,39] [--device cuda]
Prove: `e2e_test_allena.py` (A9-A10), `e2e_test_f32.py` (A26: tensori F32 allenati).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
sys.path.insert(0, "@@ROOT:models@@/build/llamacpp-tq1-k1024/gguf-py")
sys.path.insert(0, "@@ROOT:forge@@/kernel-rotazione")

from gguf import GGUFReader  # noqa: E402
import e2e_codici_scale as CS  # noqa: E402

VERSIONE = "e2e_esporta 1 (26/9/2026)"
LIB_GGML = "@@ROOT:models@@/build/llamacpp-tq1-k1024/build/bin/libggml-base.so"
VIETATI = ("/tmp", "/dev/shm", "/run", "/var/tmp")


def adesso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(p: Path, blocco: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(blocco)
            if not b:
                return h.hexdigest()
            h.update(b)


def controlla_destinazione(src: Path, dst: Path, margine_gib: float = 2.0) -> None:
    """§2.7: mai in /tmp (tmpfs = RAM) ne' sul disco di sistema; dst non esiste; spazio sufficiente."""
    dst = dst.resolve()
    if dst.exists():
        raise FileExistsError(dst)
    if dst == src.resolve():
        raise ValueError("destinazione == sorgente")
    s = str(dst)
    if any(s == v or s.startswith(v + "/") for v in VIETATI):
        raise ValueError(f"{dst}: cartella vietata (tmpfs/RAM)")
    if os.stat(dst.parent).st_dev == os.stat("/").st_dev:
        raise ValueError(f"{dst}: sul disco di sistema /, vietato (§2.7)")
    libero = shutil.disk_usage(dst.parent).free
    if libero < src.stat().st_size + margine_gib * 2 ** 30:
        raise OSError(f"{dst.parent}: {libero / 2 ** 30:.1f} GiB liberi, servono "
                      f"{src.stat().st_size / 2 ** 30 + margine_gib:.1f}")


def _deq_c(blocchi: np.ndarray, tipo: str) -> np.ndarray:
    """Dequantizzazione del codice C di llama.cpp (build k1024, backend CPU) via ctypes: il riferimento di R7."""
    import ctypes
    lib = ctypes.CDLL(LIB_GGML)
    fn = getattr(lib, "dequantize_row_tq1_0" if tipo == "TQ1_0" else "dequantize_row_tq1_0r_1024")
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
    fn.restype = None
    nb, qk, _, _ = CS.LAYOUT[tipo]
    b = np.ascontiguousarray(blocchi, dtype=np.uint8).reshape(-1, nb)
    y = np.empty((b.shape[0], qk), dtype=np.float32)
    fn(b.ctypes.data, y.ctypes.data, ctypes.c_int64(b.shape[0] * qk))
    return y


def _tensori(p: Path) -> dict:
    r = GGUFReader(str(p))
    out = {t.name: {"tipo": t.tensor_type.name, "forma": [int(x) for x in t.shape], "offset": int(t.data_offset),
                    "byte": int(t.n_bytes)} for t in r.tensors}
    del r
    return out


def controlla_scale(src: Path, dst: Path, scale: dict, campioni_esperti=(0, 127, 255)) -> dict:
    """Punti 4-5: d del file == istantanea al bit, zeri nuovi, segni, istogramma; codice C sugli esperti campione."""
    from tq1_0r_1024_codec import S1024
    S = torch.from_numpy(S1024)
    rs, rd = GGUFReader(str(src)), GGUFReader(str(dst))
    ts, td = {t.name: t for t in rs.tensors}, {t.name: t for t in rd.tensors}
    per_tensore, bordi = {}, [0, 1e-6, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, float("inf")]
    isto = np.zeros(len(bordi) - 1, dtype=np.int64)
    tot = {"scale": 0, "cambiate": 0, "zeri_nuovi": 0, "segni_cambiati": 0, "d0_zero": 0}
    for nome, attese in sorted(scale.items()):
        a, b = ts[nome], td[nome]
        tipo = a.tensor_type.name
        nb, _, off, k = CS.LAYOUT[tipo]
        ba = np.asarray(a.data).reshape(-1, nb)
        bb = np.asarray(b.data).reshape(-1, nb)
        d0 = ba[:, off:off + 2 * k].copy().view(np.float16).reshape(-1)
        d1 = bb[:, off:off + 2 * k].copy().view(np.float16).reshape(-1)
        att = np.ascontiguousarray(attese, dtype=np.float16).reshape(-1)
        if d1.shape != att.shape or not np.array_equal(d1.view(np.uint16), att.view(np.uint16)):
            raise RuntimeError(f"{nome}: campi d del file diversi dall'istantanea")
        f0, f1 = d0.astype(np.float32), d1.astype(np.float32)
        zeri = int(((f1 == 0) & (f0 != 0)).sum())
        segni = int((np.sign(f1) != np.sign(f0)).sum())
        vive = f0 != 0
        rel = np.abs(f1[vive] / f0[vive] - 1.0)
        isto += np.histogram(rel, bins=bordi)[0]
        cambiate = int((d0.view(np.uint16) != d1.view(np.uint16)).sum())
        tot["scale"] += d1.size
        tot["cambiate"] += cambiate
        tot["zeri_nuovi"] += zeri
        tot["segni_cambiati"] += segni
        tot["d0_zero"] += int((~vive).sum())
        if zeri or segni:
            raise RuntimeError(f"{nome}: {zeri} zeri nuovi, {segni} segni cambiati")
        # codice C sugli esperti campione: trit dai byte del SORGENTE x d nuove == dequantizzazione del file
        # 28/9 --densi-ternari: un tensore denso 2D (R, NBR) e' UN esperto (1, R, NBR): si controlla tutto
        da, db = np.asarray(a.data), np.asarray(b.data)
        if da.ndim == 2:
            da, db = da[None], db[None]
        E = int(da.shape[0])
        c_ok = True
        for e in sorted({min(x, E - 1) for x in campioni_esperti}):
            be = np.ascontiguousarray(db[e]).reshape(-1, nb)
            bsrc = torch.from_numpy(np.ascontiguousarray(da[e]).reshape(-1, nb))
            dn = torch.from_numpy(be[:, off:off + 2 * k].copy().view(np.float16).astype(np.float32).reshape(-1, k))
            W = CS.pesi_da_trit_scale(CS.trit_da_blocchi(bsrc, tipo), dn, tipo, S if tipo != "TQ1_0" else None).numpy()
            Wc = _deq_c(be, tipo)
            if not np.array_equal(W, Wc):
                c_ok = False
                raise RuntimeError(f"{nome} esperto {e}: codice C di llama.cpp != trit x d nuove "
                                   f"(max diff {float(np.abs(W - Wc).max()):.3e})")
        per_tensore[nome] = {"tipo": tipo, "scale": int(d1.size), "cambiate": cambiate,
                             "rel_mediana": float(np.median(rel)) if rel.size else 0.0,
                             "rel_max": float(rel.max()) if rel.size else 0.0, "codice_c_al_bit": c_ok}
    del rs, rd, ts, td
    return {"totali": tot, "istogramma_rel": {f"[{bordi[i]:g}, {bordi[i + 1]:g})": int(isto[i])
                                              for i in range(len(isto))},
            "per_tensore": per_tensore, "esperti_campione": list(campioni_esperti)}


def _regioni_f32(p: Path, f32: dict) -> dict:
    """{nome: (offset assoluto, n_bytes)} dei tensori F32 da riscrivere; solleva se nome/tipo/forma non tornano."""
    r = GGUFReader(str(p))
    ts = {t.name: t for t in r.tensors}
    out = {}
    for n, v in f32.items():
        t = ts.get(n)
        if t is None or t.tensor_type.name != "F32":
            raise ValueError(f"{n}: non e' un tensore F32 del GGUF ({None if t is None else t.tensor_type.name})")
        v = np.asarray(v)
        if v.dtype != np.float32 or v.size != int(t.n_elements) or v.nbytes != int(t.n_bytes):
            raise ValueError(f"{n}: {v.dtype} {v.shape}, il tensore ha {int(t.n_elements)} F32")
        if tuple(v.shape) != tuple(np.asarray(t.data).shape):
            raise ValueError(f"{n}: forma {v.shape} != {tuple(np.asarray(t.data).shape)} del file")
        if not np.isfinite(v).all():
            raise ValueError(f"{n}: valori non finiti")
        out[n] = (int(t.data_offset), int(t.n_bytes))
    del r, ts
    return out


def patcha_f32(p: Path, f32: dict) -> dict:
    """Riscrive sul posto (memmap r+) i soli byte dei tensori F32 in `f32` ({nome: fp32 nella forma del file}).
    Ritorna {nome: valori cambiati}."""
    reg = _regioni_f32(p, f32)
    mm = np.memmap(p, dtype=np.uint8, mode="r+")
    cambiati = {}
    for n, (off, nb) in reg.items():
        nuovi = np.ascontiguousarray(f32[n], dtype=np.float32).reshape(-1).view(np.uint8)
        vecchi = mm[off:off + nb].view(np.uint32)
        cambiati[n] = int((vecchi != nuovi.view(np.uint32)).sum())
        mm[off:off + nb] = nuovi
    mm.flush()
    del mm
    return cambiati


def impronta_fuori_f32(p: Path, f32: dict, pezzo: int = 1 << 24) -> dict:
    """sha256 di OGNI byte del file fuori dalle regioni dei tensori F32 in `f32` (e conteggio dei byte)."""
    reg = sorted(_regioni_f32(p, f32).values())
    a = np.memmap(p, dtype=np.uint8, mode="r")
    h, fuori, pos = hashlib.sha256(), 0, 0
    for off, nb in reg + [(a.size, 0)]:
        if off < pos:
            raise RuntimeError("regioni F32 sovrapposte")
        for s0 in range(pos, off, pezzo):
            s1 = min(off, s0 + pezzo)
            h.update(memoryview(a[s0:s1]))
            fuori += s1 - s0
        pos = off + nb
    del a
    return {"sha256_fuori": h.hexdigest(), "byte_fuori": fuori}


def verifica_f32(p: Path, f32: dict, prima: dict) -> dict:
    """Dopo `patcha_f32`: i byte fuori dalle regioni F32 hanno la STESSA impronta di prima (`impronta_fuori_f32`
    sul file prima della riscrittura), dentro esattamente i byte attesi, e la rilettura GGUFReader == attesi al
    bit (vista uint32). Solleva al primo scarto."""
    dopo = impronta_fuori_f32(p, f32)
    if dopo != prima:
        raise RuntimeError(f"verifica F32: byte fuori dalle regioni F32 cambiati ({prima} -> {dopo})")
    reg = _regioni_f32(p, f32)
    mm = np.memmap(p, dtype=np.uint8, mode="r")
    for n, (off, nb) in reg.items():
        if not np.array_equal(mm[off:off + nb], np.ascontiguousarray(f32[n], dtype=np.float32).reshape(-1).view(np.uint8)):
            raise RuntimeError(f"verifica F32: {n}, byte scritti diversi dagli attesi")
    del mm
    r = GGUFReader(str(p))
    for t in r.tensors:
        if t.name in f32 and not np.array_equal(np.asarray(t.data).view(np.uint32),
                                                np.ascontiguousarray(f32[t.name], dtype=np.float32).view(np.uint32)):
            raise RuntimeError(f"verifica F32: rilettura di {t.name} diversa dagli attesi")
    del r
    return {"tensori": len(reg), "byte_riscritti": int(sum(nb for _, nb in reg.values())), **dopo}


def _regioni_lora(p: Path, nuovi: dict) -> dict:
    """{nome: (offset assoluto, n_bytes)} dei tensori k-quant da riscrivere (--lora-nonesperti); solleva se nome, tipo o
    taglia non tornano (la taglia del file NON cambia: stessi byte, stesso tipo)."""
    import e2e_lora as LR
    r = GGUFReader(str(p))
    ts = {t.name: t for t in r.tensors}
    out = {}
    for n, v in nuovi.items():
        t = ts.get(n)
        if t is None or t.tensor_type.name not in LR.TIPI:
            raise ValueError(f"{n}: non e' un tensore k-quant LoRA del GGUF ({None if t is None else t.tensor_type.name})")
        v = np.asarray(v)
        if v.dtype != np.uint8 or v.nbytes != int(t.n_bytes) or tuple(v.shape) != tuple(np.asarray(t.data).shape):
            raise ValueError(f"{n}: {v.dtype} {v.shape}, il tensore ha {int(t.n_bytes)} byte {np.asarray(t.data).shape}")
        out[n] = (int(t.data_offset), int(t.n_bytes))
    del r, ts
    return out


def _impronta_fuori(p: Path, reg: dict, pezzo: int = 1 << 24) -> dict:
    """sha256 di OGNI byte del file fuori dalle regioni `reg` ({nome: (offset, n_bytes)}) e conteggio dei byte."""
    reg = sorted(reg.values())
    a = np.memmap(p, dtype=np.uint8, mode="r")
    h, fuori, pos = hashlib.sha256(), 0, 0
    for off, nb in reg + [(a.size, 0)]:
        if off < pos:
            raise RuntimeError("regioni sovrapposte")
        for s0 in range(pos, off, pezzo):
            s1 = min(off, s0 + pezzo)
            h.update(memoryview(a[s0:s1]))
            fuori += s1 - s0
        pos = off + nb
    del a
    return {"sha256_fuori": h.hexdigest(), "byte_fuori": fuori}


def patcha_lora(p: Path, nuovi: dict) -> dict:
    """Riscrive sul posto (memmap r+) i soli byte dei tensori in `nuovi` e verifica: ogni byte fuori da quelle regioni
    con la stessa sha256 di prima, le regioni == i byte attesi, rilettura GGUFReader == attesi. Solleva al primo
    scarto. Ritorna {byte cambiati per tensore, verifica}."""
    reg = _regioni_lora(p, nuovi)
    prima = _impronta_fuori(p, reg)
    mm = np.memmap(p, dtype=np.uint8, mode="r+")
    cambiati = {}
    for n, (off, nb) in reg.items():
        nuovo = np.ascontiguousarray(nuovi[n], dtype=np.uint8).reshape(-1)
        cambiati[n] = int((mm[off:off + nb] != nuovo).sum())
        mm[off:off + nb] = nuovo
    mm.flush()
    del mm
    dopo = _impronta_fuori(p, reg)
    if dopo != prima:
        raise RuntimeError(f"verifica LoRA: byte fuori dalle regioni riscritte cambiati ({prima} -> {dopo})")
    r = GGUFReader(str(p))
    for t in r.tensors:
        if t.name in nuovi and not np.array_equal(np.asarray(t.data), np.asarray(nuovi[t.name])):
            raise RuntimeError(f"verifica LoRA: rilettura di {t.name} diversa dai byte attesi")
    del r
    return {"byte_cambiati_per_tensore": cambiati, "tensori": len(reg),
            "byte_riscritti": int(sum(nb for _, nb in reg.values())), **dopo}


def ricarica(src: Path, dst: Path, scale: dict, theta: Optional[dict], strati: list, device: str = "cpu",
             ricetta: str = "B", motore: Optional[str] = None, token: int = 256, esperti: Optional[slice] = None,
             log: Callable = print, f32: Optional[dict] = None, lora_nuovi: Optional[dict] = None,
             lora: Optional[dict] = None, lora_meta: Optional[dict] = None) -> dict:
    """Punto 6. Per ogni strato: studente SOLO dal file esportato (theta = 0) -> scale == istantanea al bit;
    se `theta` (del migliore) c'e', sullo stesso ingresso l'uscita dello strato == studente del SORGENTE con
    theta, al bit (nessuna tolleranza). `f32` (27/9): lo studente del sorgente ha anche quei tensori F32 coi
    valori allenati (`allena_f32` + `imposta_f32`); output_norm si confronta a parte."""
    import e2e_studente as S
    fam_f32 = ()
    if f32:
        per_nome = {v: k for k, v in S.FAMIGLIE_F32.items()}
        fam_f32 = S.famiglie_f32({per_nome[n.split(".", 2)[2]] if n.startswith("blk.") else
                                  {v: k for k, v in S.F32_GLOBALI.items()}[n] for n in f32})
    motore = motore or ("triton" if device.startswith("cuda") else "torch")
    # 28/9 --densi-ternari: i densi ternari allenati sono quelli che hanno scale nell'istantanea; gli studenti di ricarica
    # li costruiscono a esperto unico (stesso avanti dell'allenamento). Nessuno = chiamate di sempre.
    densi = S.famiglie_dense_da_nomi(scale)
    kw_d = {"densi_ternari": densi} if densi else {}
    esiti = {}
    for L in strati:
        sf = S.Studente(gguf=dst, strati=[L], ricetta=ricetta, motore_nome=motore, device=device, esperti=esperti,
                        log=lambda *_: None, **kw_d)
        sc = sf.scale_per_export()
        manca = [n for n in sc if n not in scale]
        uguali = True
        for n in sc:
            if n in manca:
                continue
            denso = np.asarray(sf._tens[n].data).ndim == 2           # 28/9: un esperto solo, nessuna fetta
            E_tot = 1 if denso else int(sf._tens[n].shape[-1])
            att = np.ascontiguousarray(scale[n], dtype=np.float16).reshape(E_tot, -1)
            if esperti is not None and not denso:
                att = att[esperti]
            uguali &= bool(np.array_equal(np.ascontiguousarray(sc[n]).view(np.uint16),
                                          np.ascontiguousarray(att).reshape(sc[n].shape).view(np.uint16)))
        r = {"scale_al_bit": uguali and not manca, "non_nell_istantanea": manca}
        if f32 and "output_norm.weight" in f32:
            r["output_norm_al_bit"] = bool(np.array_equal(
                sf.norma_finale.cpu().numpy().view(np.uint32),
                np.ascontiguousarray(f32["output_norm.weight"], dtype=np.float32).view(np.uint32)))
        if theta is not None or f32 or lora_nuovi:
            ss = S.Studente(gguf=src, strati=[L], ricetta=ricetta, motore_nome=motore, device=device, esperti=esperti,
                            log=lambda *_: None, allena_f32=fam_f32, **kw_d)
            if f32:
                ss.imposta_f32({n: v for n, v in f32.items() if n in ss.f32})
                r["f32_nello_strato"] = sum(1 for n in ss.f32 if n.startswith(f"blk.{L}."))
            # 26/9 sera: il riferimento usa le scale ESATTE dell'allenamento (istantanea), non d0*exp(theta) ricalcolato
            # qui: exp fp32 su CPU e su GPU differiscono all'ultimo bit e ~5 scale su un milione cadono sull'altro lato
            # dell'arrotondamento fp16 (misurato: strato 0 di EB2 passo 2, 28 su 5.242.880, tutte di 1 ulp). Con
            # theta_ref = ln(scala/d0) in fp64, fp16_RNE(d0*exp(theta_ref)) = scala al bit (lo scarto fp32 e' ~1e-7,
            # la spaziatura fp16 ~1e-3). Cosi' il confronto dell'uscita resta AL BIT e prova il posizionamento nel file.
            with torch.no_grad():
                for (LL, n), p in zip(ss.chiavi(), ss.parametri()):
                    d0 = ss.esperti(LL).fam[n].d0.to(torch.float64)
                    sn = np.ascontiguousarray(scale[f"blk.{LL}.{S.nome_tensore_famiglia(n)}.weight"], dtype=np.float16)
                    sn = sn.reshape(-1, d0[0].numel()) if (esperti is None or S.e_densa(n)) \
                        else sn.reshape(-1, d0[0].numel())[esperti]
                    sn = torch.from_numpy(sn.astype(np.float64)).reshape(d0.shape)
                    rif = torch.where(d0 != 0, torch.log(sn / torch.where(d0 != 0, d0, torch.ones_like(d0))),
                                      torch.zeros_like(d0))
                    p.copy_(rif.to(torch.float32))
                ok = all(np.array_equal(np.ascontiguousarray(v).view(np.uint16).ravel(),
                                        np.ascontiguousarray(scale[k], dtype=np.float16).view(np.uint16).ravel()
                                        if esperti is None else np.ascontiguousarray(v).view(np.uint16).ravel())
                         for k, v in ss.scale_per_export().items() if k in scale)
                if not ok:
                    raise RuntimeError(f"ricarica strato {L}: il riferimento non riproduce le scale dell'istantanea")
            g = torch.Generator().manual_seed(20260926 + L)
            x = (torch.randn(1, token, ss.cfg.hidden_size, generator=g) * 0.5).to(device=device, dtype=ss.ric.dtype_residuo)
            if lora_nuovi and any(n.startswith(f"blk.{L}.") for n in lora_nuovi):
                r.update(_lora_nello_strato(ss, src, L, x, lora_nuovi, lora, lora_meta, fam_f32, f32, ricetta, motore,
                                            device, esperti))
            with torch.no_grad():
                ya = ss.avanti(torch.zeros(1, token, dtype=torch.long), x0=x)["xL"]
                yb = sf.avanti(torch.zeros(1, token, dtype=torch.long), x0=x)["xL"]
            r["uscita_al_bit"] = bool(torch.equal(ya, yb))
            r["uscita_max_diff"] = float((ya.double() - yb.double()).abs().max())
            if "_y_base" in r:                  # LoRA: quanta parte dell'effetto sull'uscita dello strato sopravvive
                yb0, ytr = r.pop("_y_base"), r.pop("_y_allenato")
                dv, de = (ytr - yb0).double(), (yb.double() - yb0.double())
                n2 = float((dv * dv).sum())
                m = float((de * dv).sum()) / n2 if n2 > 0 else float("nan")
                r["lora_uscita"] = {"mantenuto": m, "rumore": float(((de - m * dv) ** 2).sum() / n2) ** 0.5 if n2 > 0
                                    else float("nan"), "rel_effetto_allenato": (n2 / float((yb0.double() ** 2).sum())) ** 0.5}
            del ss, ya, yb
        esiti[L] = r
        log(f"ricarica strato {L}: {r}")
        del sf
        if not r["scale_al_bit"] or r.get("uscita_al_bit") is False or r.get("output_norm_al_bit") is False:
            raise RuntimeError(f"ricarica strato {L} FALLITA: {r}")
    return esiti


def _lora_nello_strato(ss, src: Path, L: int, x, lora_nuovi: dict, lora: Optional[dict], lora_meta: Optional[dict],
                       fam_f32, f32, ricetta, motore, device, esperti) -> dict:
    """--lora-nonesperti, strato L. (1) y_base = studente del sorgente (theta, F32) SENZA LoRA; (2) y_allenato = lo
    stesso con gli adattatori A, B dell'istantanea attivi (cio' che l'allenamento vedeva); (3) nel riferimento `ss` i
    pesi dei tensori riscritti diventano la dequantizzazione dei byte nuovi (`e2e_lora.installa_pesi`): il chiamante
    confronta poi `ss` col file esportato AL BIT, e misura quanto di y_allenato - y_base resta in y_esportato - y_base."""
    import e2e_lora as LR
    import e2e_studente as S
    out = {}
    with torch.no_grad():
        out["_y_base"] = ss.avanti(torch.zeros(1, x.shape[1], dtype=torch.long), x0=x)["xL"].clone()
    if lora is not None and lora_meta is not None:
        cfg = LR.ConfigLoRA(rango=int(lora_meta["rango"]), alpha=lora_meta["alpha"], famiglie=tuple(lora_meta["famiglie"]),
                            seme=int(lora_meta["seme"]), tipi=tuple(lora_meta.get("tipi_ammessi") or ()))
        kw_d = {"densi_ternari": ss.densi_ternari} if getattr(ss, "densi_ternari", ()) else {}   # 28/9: come `ss`
        sl = S.Studente(gguf=src, strati=[L], ricetta=ricetta, motore_nome=motore, device=device, esperti=esperti,
                        log=lambda *_: None, allena_f32=fam_f32, lora=cfg, **kw_d)
        with torch.no_grad():
            for p, q in zip(sl.parametri(), ss.parametri()):
                p.copy_(q)
            if f32:
                sl.imposta_f32({n: v for n, v in f32.items() if n in sl.f32})
            sl.imposta_lora({n: v for n, v in lora.items() if n in sl.lora})
            out["_y_allenato"] = sl.avanti(torch.zeros(1, x.shape[1], dtype=torch.long), x0=x)["xL"].clone()
        del sl
    else:
        out["_y_allenato"] = out["_y_base"]
    out["lora_tensori_nello_strato"] = LR.installa_pesi(ss, L, lora_nuovi)
    return out


def _moduli() -> dict:
    return {p.name: sha256_file(p) for p in sorted(QUI.glob("e2e_*.py"))}


def _commit() -> Optional[str]:
    try:
        return subprocess.run(["git", "-C", str(QUI.parent), "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def esporta(src: Path, dst: Path, scale: dict, meta_istantanea: dict, theta: Optional[dict] = None,
            corsa: Optional[Path] = None, strati_ricarica: tuple = (), device: str = "cpu",
            esperti_ricarica: Optional[slice] = None, controlla_disco: bool = True, log: Callable = print,
            f32: Optional[dict] = None, lora: Optional[dict] = None, lora_meta: Optional[dict] = None,
            lora_modo: str = "griglia") -> dict:
    """I punti 1-7. `scale`: {nome del tensore GGUF: fp16 (n_blocchi, k)} (dall'istantanea). `f32` (27/9):
    {nome del tensore F32: fp32 nella forma del file} (`f32.npz` dell'istantanea), punto 6b. `lora` / `lora_meta`
    (27/9, `lora.npz` dell'istantanea): W + s B A riquantizzato nello STESSO tipo (`lora_modo` griglia | ggml,
    e2e_lora.riquantizza), riscritto al posto dei byte vecchi (punto 6c), misure di sopravvivenza per tensore e, nella
    ricarica, sull'uscita dello strato. Ritorna la provenienza (scritta anche accanto al GGUF)."""
    src, dst = Path(src), Path(dst)
    t0 = time.time()
    if controlla_disco:
        controlla_destinazione(src, dst)
    elif dst.exists():
        raise FileExistsError(dst)
    tipi_src = _tensori(src)
    for n, v in scale.items():
        if n not in tipi_src or tipi_src[n]["tipo"] not in CS.LAYOUT:
            raise ValueError(f"{n}: non e' un tensore TQ1_0/TQ1_0R_1024 del sorgente")
        nb, _, _, k = CS.LAYOUT[tipi_src[n]["tipo"]]
        if np.asarray(v).size != tipi_src[n]["byte"] // nb * k:
            raise ValueError(f"{n}: {np.asarray(v).size} scale, il tensore ne ha {tipi_src[n]['byte'] // nb * k}")
    if f32:
        _regioni_f32(src, f32)                         # nomi/tipi/forme/finitezza PRIMA di copiare 16 GB
    lora_nuovi, lora_misure = None, None
    if lora:                                           # --lora-nonesperti: byte riquantizzati PRIMA di copiare
        import e2e_lora as LR
        tl = time.time()
        lora_nuovi, lora_misure = LR.byte_riquantizzati(src, lora, lora_meta, lora_modo)
        _regioni_lora(src, lora_nuovi)
        m = [v["mantenuto"] for v in lora_misure.values() if not v.get("delta_nullo")]
        nulli = sum(1 for v in lora_misure.values() if v.get("delta_nullo"))
        log(f"LoRA: {len(lora_nuovi) - nulli} tensori riquantizzati ({lora_modo}) in {time.time() - tl:.1f} s" +
            (f"; Delta W mantenuto (proiezione) mediano {float(np.median(m)):.3f}, min {min(m):.3f}, max {max(m):.3f}"
             if m else "") +
            (f"; {nulli} con Delta W = 0 esatto: byte del sorgente (28/9 --lora-ripristina)" if nulli else ""))
    tempi = {}
    tc = time.time()

    def fase(nome):
        nonlocal tc
        tempi[nome] = round(time.time() - tc, 1)
        tc = time.time()
    log(f"sha256 del sorgente {src.name} ...")
    sha_src = sha256_file(src)
    fase("sha256_sorgente")
    tmp = dst.with_name(dst.name + ".parziale")
    if tmp.exists():
        tmp.unlink()
    log(f"copia -> {tmp}")
    shutil.copyfile(src, tmp)
    with open(tmp, "rb+") as f:
        os.fsync(f.fileno())
    fase("copia")
    cambiate = CS.patcha_scale(tmp, scale)
    fase("riscrittura_d")
    log(f"campi d riscritti: {sum(cambiate.values())} scale cambiate in {len(cambiate)} tensori; verifica del file intero")
    rep = CS.verifica_patch(src, tmp, sorted(scale), pezzo=1 << 24)
    fase("verifica_file_intero")
    tipi_dst = _tensori(tmp)
    if tipi_dst != tipi_src:
        diversi = [n for n in tipi_src if tipi_dst.get(n) != tipi_src[n]]
        raise RuntimeError(f"rilettura: tensori diversi dal sorgente {diversi[:5]}")
    sc = controlla_scale(src, tmp, scale)
    fase("rilettura_scale_codice_c")
    log(f"scale: {sc['totali']}")
    rep_f32, cambiati_f32 = None, None
    if f32:
        # DOPO verifica_patch/controlla_scale, che vogliono identico al sorgente ogni byte fuori dai campi d
        prima = impronta_fuori_f32(tmp, f32)
        cambiati_f32 = patcha_f32(tmp, f32)
        rep_f32 = verifica_f32(tmp, f32, prima)
        if _tensori(tmp) != tipi_src:
            raise RuntimeError("rilettura dopo F32: tensori diversi dal sorgente")
        fase("tensori_f32")
        log(f"tensori F32: {len(f32)} riscritti, {sum(cambiati_f32.values())} valori cambiati; {rep_f32}")
    rep_lora = None
    if lora_nuovi:
        # 6c, DOPO F32: riscrive i soli byte dei tensori LoRA (stesso tipo e taglia) e verifica il resto del file
        rep_lora = patcha_lora(tmp, lora_nuovi)
        if _tensori(tmp) != tipi_src:
            raise RuntimeError("rilettura dopo LoRA: tensori diversi dal sorgente")
        fase("tensori_lora")
        log(f"tensori LoRA: {rep_lora['tensori']} riscritti, {sum(rep_lora['byte_cambiati_per_tensore'].values())} "
            f"byte cambiati su {rep_lora['byte_riscritti']}")
    with open(tmp, "rb+") as f:
        os.fsync(f.fileno())
    os.replace(tmp, dst)
    fd = os.open(dst.parent, os.O_RDONLY)
    os.fsync(fd)
    os.close(fd)
    ric = ricarica(src, dst, scale, theta, list(strati_ricarica), device=device, esperti=esperti_ricarica,
                   log=log, f32=f32, lora_nuovi=lora_nuovi, lora=lora, lora_meta=lora_meta) if strati_ricarica else {}
    fase("ricarica")
    sha_src2 = sha256_file(src)
    if sha_src2 != sha_src:
        raise RuntimeError("il GGUF SORGENTE e' cambiato durante l'export")
    sha_dst = sha256_file(dst)
    fase("sha256_finali")
    corsa_info = {}
    if corsa is not None:
        corsa = Path(corsa)
        for nome in ("config.json", "esito.json"):
            if (corsa / nome).exists():
                d = json.loads((corsa / nome).read_text())
                if nome == "esito.json":
                    d = {k: d.get(k) for k in ("passo", "migliore", "g1", "primo_passo", "fermato", "sorgente",
                                               "cambi_instradamento", "storia_heldout", "teste", "fine")}
                corsa_info[nome] = d
    prov = {"versione": VERSIONE, "scritto": adesso(), "secondi": round(time.time() - t0, 1), "tempi": tempi,
            "sorgente": {"file": str(src), "sha256": sha_src, "byte": src.stat().st_size},
            "esportato": {"file": str(dst), "sha256": sha_dst, "byte": dst.stat().st_size},
            "istantanea": meta_istantanea, "scale_cambiate_per_tensore": cambiate, "verifica_file_intero": rep,
            "controllo_scale": sc, "ricarica": {str(k): v for k, v in ric.items()}, "corsa": corsa_info,
            "moduli_sha256": _moduli(), "commit_fucina": _commit()}
    if f32:
        prov["tensori_f32"] = {"valori_cambiati_per_tensore": cambiati_f32, "verifica": rep_f32}
    if lora_nuovi:
        prov["lora"] = {"modo": lora_modo, "meta": {k: v for k, v in lora_meta.items() if k != "voci"},
                        "sopravvivenza_per_tensore": lora_misure, "verifica": rep_lora}
    p = Path(str(dst) + ".provenienza.json")
    with open(p, "w") as f:
        json.dump(prov, f, indent=1, default=str)
        f.flush()
        os.fsync(f.fileno())
    log(f"esportato {dst} ({sha_dst[:16]}), provenienza {p.name}, {prov['secondi']} s")
    return prov


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sorgente", required=True)
    ap.add_argument("--corsa", required=True, help="cartella della corsa di e2e_allena")
    ap.add_argument("--istantanea", default=None, help="default: il migliore dell'ultima ripresa")
    ap.add_argument("--uscita", required=True)
    ap.add_argument("--strati-ricarica", default="0,3,39")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--lora-modo", choices=("griglia", "ggml"), default="griglia",
                    help="riquantizzazione di W + s B A (--lora-nonesperti): griglia = scale del file ferme, codici "
                         "riarrotondati (Delta W = 0 -> byte identici); ggml = quantizzatore di llama.cpp da capo")
    a = ap.parse_args(argv)
    import e2e_allena as A
    corsa = Path(a.corsa)
    nome = a.istantanea or json.loads((corsa / "ripresa_ultima.json").read_text())["migliore"]
    ist = A.carica_istantanea(corsa / nome)
    theta = torch.load(corsa / nome / "theta.pt", map_location="cpu", weights_only=True)
    strati = [int(x) for x in a.strati_ricarica.split(",") if x != ""]
    esporta(Path(a.sorgente), Path(a.uscita), ist["scale"], ist["meta"], theta=theta, corsa=corsa,
            strati_ricarica=tuple(strati), device=a.device, f32=ist.get("f32"), lora=ist.get("lora"),
            lora_meta=ist.get("lora_meta"), lora_modo=a.lora_modo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
