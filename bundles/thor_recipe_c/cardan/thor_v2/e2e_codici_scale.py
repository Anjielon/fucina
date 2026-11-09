#!/usr/bin/env python3
"""E2E sulle scale (progetto `e2e_progetto_2.md`) — i tre mattoni «di fedelta'», solo CPU.

25/9/2026. File NUOVO: non modifica e non viene importato da nessun file esistente. Legge
in SOLA LETTURA `rotazione_gpu.py` (FWHT torch gia' verificata) e il codec di casa
`kernel-rotazione/tq1_0r_1024_codec.py` (riferimento op per op di llama.cpp).

Perche' esiste. La fase E2E allena SOLO le scale fp16 degli esperti ternari, a codici
fermi. Tre cose devono essere vere al bit, altrimenti si ottimizza un modello diverso da
quello che llama.cpp esegue (lezione B9):

1. lo studente legge i trit dagli STESSI byte del GGUF che legge llama.cpp
   (`trit_da_blocchi`: porta torch del decode di `dequantize_row_tq1_0`);
2. il peso che lo studente usa, a scale iniziali, e' identico al dequantizzato di
   riferimento (`pesi_da_trit_scale`: s (x) trit per gruppo, poi de-rotazione per
   TQ1_0R_1024, differenziabile in s);
3. l'export riscrive SOLO i campi d dei blocchi (2 byte a blocco per TQ1_0, 16 per
   TQ1_0R_1024) in una COPIA del GGUF: ogni altro byte resta identico
   (`patcha_scale` + `verifica_patch`).

Layout (ggml-common.h della build k1024, verificato):
  TQ1_0        54 byte / 256 pesi: qs[0:48] qh[48:52] d[52:54]            (1 scala / 256)
  TQ1_0R_1024 224 byte / 1024 pesi: qs[0:192] (4 quarti x 48) qh[192:208] d[208:224]
               (8 scale fp16, d[g] per gli elementi g*128..g*128+127 nel dominio RUOTATO)

Uso:  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 19 python3 e2e_codici_scale.py --test
      (scrive un GGUF giocattolo di pochi MB in $E2E_SCRATCH, default ./_e2e_scratch)
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
sys.path.insert(0, "@@ROOT:models@@/build/llamacpp-tq1-k1024/gguf-py")
sys.path.insert(0, "@@ROOT:forge@@/kernel-rotazione")

from gguf import GGUFReader  # noqa: E402

# tipo -> (byte per blocco, pesi per blocco, offset di d, scale per blocco)
LAYOUT = {"TQ1_0": (54, 256, 52, 1), "TQ1_0R_1024": (224, 1024, 208, 8)}
_POW3 = (1, 3, 9, 27, 81)


# ─────────────────────────── 1. decode dei trit (torch) ───────────────────────────
def _quarto(qs: torch.Tensor, qh: torch.Tensor) -> torch.Tensor:
    """(N,48),(N,4) uint8 -> (N,256) int8 in {-1,0,1}: STESSI loop di dequantize_row_tq1_0
    (troncamento a 8 bit compreso), ordine naturale degli elementi."""
    out = torch.empty(qs.shape[0], 256, dtype=torch.int8)
    b = qs[:, 0:32].to(torch.int32)
    for n, p in enumerate(_POW3):
        out[:, n * 32:(n + 1) * 32] = ((((b * p) & 255) * 3) >> 8).to(torch.int8) - 1
    b = qs[:, 32:48].to(torch.int32)
    for n, p in enumerate(_POW3):
        out[:, 160 + n * 16:160 + (n + 1) * 16] = ((((b * p) & 255) * 3) >> 8).to(torch.int8) - 1
    b = qh.to(torch.int32)
    for n, p in enumerate(_POW3[:4]):
        out[:, 240 + n * 4:240 + (n + 1) * 4] = ((((b * p) & 255) * 3) >> 8).to(torch.int8) - 1
    return out


def trit_da_blocchi(blocchi: torch.Tensor, tipo: str) -> torch.Tensor:
    """blocchi (N, byte_per_blocco) uint8 -> trit (N, pesi_per_blocco) int8."""
    nb, qk, _, _ = LAYOUT[tipo]
    assert blocchi.dtype == torch.uint8 and blocchi.shape[-1] == nb
    if tipo == "TQ1_0":
        return _quarto(blocchi[:, 0:48], blocchi[:, 48:52])
    return torch.cat([_quarto(blocchi[:, q * 48:(q + 1) * 48], blocchi[:, 192 + q * 4:192 + (q + 1) * 4])
                      for q in range(4)], dim=1)


def scale_da_blocchi(blocchi: torch.Tensor, tipo: str) -> torch.Tensor:
    """blocchi (N, byte_per_blocco) uint8 -> scale (N, k) float16, i byte d cosi' come sono."""
    nb, _, off, k = LAYOUT[tipo]
    return blocchi[:, off:off + 2 * k].contiguous().view(torch.float16).reshape(-1, k)


# ─────────────────────── 2. peso differenziabile nelle scale ───────────────────────
def pesi_da_trit_scale(trit: torch.Tensor, scale: torch.Tensor, tipo: str,
                       S: "torch.Tensor | None" = None) -> torch.Tensor:
    """trit (N, qk) int8 · scale (N, k) (fp32, con grad se allenate) -> pesi (N, qk) fp32.
    Stessa aritmetica del riferimento: per TQ1_0R_1024 trit*d[g] in fp32, poi FWHT-1024
    (con 1/32 in testa) e segno S — `rotazione_gpu._RT_1024_torch`, riusata."""
    _, qk, _, k = LAYOUT[tipo]
    W = (trit.to(torch.float32).reshape(trit.shape[0], k, qk // k) * scale.to(torch.float32).unsqueeze(-1))
    W = W.reshape(trit.shape[0], qk)
    if tipo == "TQ1_0R_1024":
        from rotazione_gpu import _RT_1024_torch  # sola lettura
        W = _RT_1024_torch(W, S.to(torch.float32))
    return W


def scala_fp16_ste(s: torch.Tensor) -> torch.Tensor:
    """Avanti: la scala arrotondata a fp16 (round-to-nearest-even, come l'export e come
    GGML_FP32_TO_FP16). Indietro: identita' (straight-through). Cosi' lo studente allenato
    E' il modello esportato, a ogni passo."""
    return s + (s.to(torch.float16).to(s.dtype) - s).detach()


# ─────────────────────────────── 3. export al bit ───────────────────────────────
def posizione_scale(t) -> dict:
    """Dove stanno i campi d di un ReaderTensor: offset assoluto nel file, numero di
    blocchi, layout. I blocchi sono in ordine numpy (esperti, righe, blocchi della riga)."""
    tipo = t.tensor_type.name
    nb, qk, off, k = LAYOUT[tipo]
    n_blocchi = int(t.n_bytes) // nb
    assert n_blocchi * nb == int(t.n_bytes)
    return {"tipo": tipo, "inizio": int(t.data_offset), "n_blocchi": n_blocchi, "byte_blocco": nb,
            "off_d": off, "k": k}


def patcha_scale(dst: Path, nuove: dict) -> dict:
    """`nuove`: {nome_tensore: array float (n_blocchi, k) o (…, n_blocchi_riga*k)} — le scale
    nell'ordine dei blocchi del tensore. Scrive fp16(RNE) SOLO nei campi d di `dst`, che deve
    essere una COPIA gia' fatta (mai il file sorgente). Ritorna {nome: numero di scale cambiate}."""
    r = GGUFReader(str(dst))
    tens = {t.name: t for t in r.tensors}
    pos = {n: posizione_scale(tens[n]) for n in nuove}
    del r, tens
    mm = np.memmap(dst, dtype=np.uint8, mode="r+")
    cambiate = {}
    for nome, valori in nuove.items():
        p = pos[nome]
        v16 = np.ascontiguousarray(np.asarray(valori, dtype=np.float32).reshape(p["n_blocchi"], p["k"])
                                   .astype(np.float16))
        if not np.isfinite(v16.astype(np.float32)).all():
            raise FloatingPointError(f"{nome}: scala fuori dal campo fp16")
        vista = mm[p["inizio"]:p["inizio"] + p["n_blocchi"] * p["byte_blocco"]].reshape(p["n_blocchi"], p["byte_blocco"])
        prima = vista[:, p["off_d"]:p["off_d"] + 2 * p["k"]].copy()
        vista[:, p["off_d"]:p["off_d"] + 2 * p["k"]] = v16.view(np.uint8).reshape(p["n_blocchi"], 2 * p["k"])
        cambiate[nome] = int((prima.view(np.uint16) != v16.view(np.uint16).reshape(prima.shape[0], -1)).sum())
    mm.flush()
    del mm
    return cambiate


def verifica_patch(src: Path, dst: Path, nomi: list, pezzo: int = 1 << 26) -> dict:
    """Confronta i due file a pezzi. Tutti i byte FUORI dai campi d dei tensori `nomi` devono
    essere identici (si solleva altrimenti). Ritorna gli sha256 del flusso 'senza campi d'
    (uguale per costruzione se passa) e di quello dei soli codici (qs/qh) dei tensori patchati."""
    if src.stat().st_size != dst.stat().st_size:
        raise RuntimeError(f"taglie diverse: {src.stat().st_size} contro {dst.stat().st_size}")
    r = GGUFReader(str(src))
    tens = {t.name: t for t in r.tensors}
    maschere = []   # (inizio, fine, byte_blocco, off_d, k)
    for n in nomi:
        p = posizione_scale(tens[n])
        maschere.append((p["inizio"], p["inizio"] + p["n_blocchi"] * p["byte_blocco"], p["byte_blocco"],
                         p["off_d"], p["k"]))
    del r, tens
    h_resto, h_codici = hashlib.sha256(), hashlib.sha256()
    a = np.memmap(src, dtype=np.uint8, mode="r")
    b = np.memmap(dst, dtype=np.uint8, mode="r")
    n = a.shape[0]
    for i0 in range(0, n, pezzo):
        i1 = min(n, i0 + pezzo)
        x, y = np.asarray(a[i0:i1]), np.asarray(b[i0:i1])
        dentro_d = np.zeros(i1 - i0, dtype=bool)
        codici = np.zeros(i1 - i0, dtype=bool)
        for (m0, m1, nb, off, k) in maschere:
            j0, j1 = max(i0, m0), min(i1, m1)
            if j0 >= j1:
                continue
            rel = np.arange(j0, j1) - m0
            nel_blocco = rel % nb
            e_d = (nel_blocco >= off) & (nel_blocco < off + 2 * k)
            dentro_d[j0 - i0:j1 - i0] |= e_d
            codici[j0 - i0:j1 - i0] |= ~e_d
        diversi = (x != y) & ~dentro_d
        if diversi.any():
            j = int(np.flatnonzero(diversi)[0]) + i0
            raise RuntimeError(f"byte {j} diverso FUORI dai campi d: {int(a[j])} contro {int(b[j])}")
        h_resto.update(x[~dentro_d].tobytes())
        h_codici.update(x[codici].tobytes())
    return {"sha256_senza_d": h_resto.hexdigest(), "sha256_codici": h_codici.hexdigest(), "byte": n}


# ─────────────────────────────────── test CPU ───────────────────────────────────
def _test(scratch: Path) -> bool:
    from gguf.quants import TQ1_0 as GQ_TQ1_0
    from tq1_0r_1024_codec import dequantize_blocks as deq_r1024, S1024
    torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "2")))
    ok = True
    fonti = {"TQ1_0": ("@@ROOT:models@@/gguf/tony-tern/Thor-v20-densiv5.gguf", ["blk.0.ffn_gate_exps.weight",
                                                                          "blk.39.ffn_up_exps.weight"]),
             "TQ1_0R_1024": ("@@ROOT:models@@/gguf/tony-tern/Thor-T-ibrido-0-19-ThorT-veloce.gguf",
                             ["blk.0.ffn_gate_exps.weight", "blk.0.ffn_down_exps.weight",
                              "blk.19.ffn_up_exps.weight"])}
    S = torch.from_numpy(S1024)
    campioni = {}
    for tipo_atteso, (percorso, nomi) in fonti.items():
        r = GGUFReader(percorso)
        tens = {t.name: t for t in r.tensors}
        for nome in nomi:
            t = tens[nome]
            tipo = t.tensor_type.name
            nb, qk, _, k = LAYOUT[tipo]
            raw = np.asarray(t.data)                       # memmap (E, righe, byte)
            for e in (0, 97, 255):
                blocchi = np.ascontiguousarray(raw[e]).reshape(-1, nb)
                bt = torch.from_numpy(blocchi)
                trit = trit_da_blocchi(bt, tipo)
                sc = scale_da_blocchi(bt, tipo)
                if tipo == "TQ1_0":
                    rif = GQ_TQ1_0.dequantize_blocks(blocchi)                      # gguf-py
                    W = pesi_da_trit_scale(trit, sc.float(), tipo).numpy()
                else:
                    rif = deq_r1024(blocchi)                                     # codec di casa
                    W = pesi_da_trit_scale(trit, sc.float(), tipo, S).numpy()
                uguale = np.array_equal(W, rif)
                dmax = float(np.abs(W - rif).max())
                ok &= uguale
                print(f"[T1/T2] {Path(percorso).name} {nome} {tipo} esperto {e}: {blocchi.shape[0]} blocchi · "
                      f"peso torch == riferimento al bit: {uguale} (max|diff| {dmax:.3e})")
            campioni[tipo] = (np.ascontiguousarray(raw[:2, :4]), t)
        del r, tens

    # T3: gradiente rispetto alle scale = somma per gruppo di q * dL/dW (EfficientQAT eq.: dw/ds = w_q)
    torch.manual_seed(0)
    blocchi = torch.from_numpy(np.ascontiguousarray(campioni["TQ1_0R_1024"][0]).reshape(-1, 224)[:4])
    trit = trit_da_blocchi(blocchi, "TQ1_0R_1024")
    s = scale_da_blocchi(blocchi, "TQ1_0R_1024").double().requires_grad_(True)
    x = torch.randn(1024, 7, dtype=torch.float64)
    Wd = (trit.double().reshape(4, 8, 128) * s.unsqueeze(-1)).reshape(4, 1024)
    from rotazione_gpu import _RT_1024_torch
    y = _RT_1024_torch(Wd, S.double()) @ x
    L = (y ** 2).sum()
    L.backward()
    g = s.grad.clone()
    eps = 1e-6
    num = torch.zeros_like(g)
    with torch.no_grad():
        for i in range(4):
            for j in range(8):
                sp, sm = s.detach().clone(), s.detach().clone()
                sp[i, j] += eps
                sm[i, j] -= eps
                f = lambda ss: ((_RT_1024_torch((trit.double().reshape(4, 8, 128) * ss.unsqueeze(-1))
                                                .reshape(4, 1024), S.double()) @ x) ** 2).sum()
                num[i, j] = (f(sp) - f(sm)) / (2 * eps)
    rel = float(((g - num).abs().max() / num.abs().max()))
    ok &= rel < 1e-6
    print(f"[T3] gradiente autograd contro differenze finite (float64, 32 scale): errore relativo {rel:.2e}")

    # T5: ricetta B (come lo shader rot_y di Vulkan): W_reale @ x == W_ruotato @ (R x), R x = FWHT(S (.) x)
    from rotazione_gpu import _R_1024_torch
    r = GGUFReader(fonti["TQ1_0R_1024"][0])
    t = {q.name: q for q in r.tensors}["blk.19.ffn_up_exps.weight"]
    blocchi = torch.from_numpy(np.ascontiguousarray(np.asarray(t.data)[5]).reshape(-1, 224))
    trit, sc = trit_da_blocchi(blocchi, "TQ1_0R_1024"), scale_da_blocchi(blocchi, "TQ1_0R_1024").double()
    W_rot = (trit.double().reshape(-1, 8, 128) * sc.unsqueeze(-1)).reshape(512, 2048)
    W_reale = _RT_1024_torch(W_rot, S.double())
    x = torch.randn(16, 2048, dtype=torch.float64)
    a, b = x @ W_reale.T, _R_1024_torch(x, S.double()) @ W_rot.T
    rel = float((a - b).abs().max() / a.abs().max())
    ok &= rel < 1e-12
    print(f"[T5] esperto 5 di blk.19.ffn_up: uscita col peso de-ruotato contro attivazione ruotata (float64): "
          f"errore relativo {rel:.2e}")
    del r, t

    # T4: patch su un GGUF giocattolo costruito coi blocchi VERI, poi verifica al bit
    from gguf import GGUFWriter, GGMLQuantizationType as T
    scratch.mkdir(parents=True, exist_ok=True)
    src, dst = scratch / "e2e_giocattolo.gguf", scratch / "e2e_giocattolo_patch.gguf"
    w = GGUFWriter(str(src), arch="qwen35moe")
    w.add_uint32("e2e.prova", 1)
    w.add_tensor("blk.0.ffn_gate_exps.weight", campioni["TQ1_0R_1024"][0], raw_dtype=T.TQ1_0R_1024)
    w.add_tensor("blk.0.ffn_up_exps.weight", campioni["TQ1_0"][0], raw_dtype=T.TQ1_0)
    w.add_tensor("blk.0.attn_norm.weight", np.arange(2048, dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    shutil.copyfile(src, dst)
    rs = GGUFReader(str(src))
    ts = {t.name: t for t in rs.tensors}
    rng = np.random.default_rng(1)
    nuove, attese = {}, {}
    for nome in ("blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight"):
        t = ts[nome]
        p = posizione_scale(t)
        blocchi = np.asarray(t.data).reshape(-1, p["byte_blocco"])
        s0 = blocchi[:, p["off_d"]:p["off_d"] + 2 * p["k"]].copy().view(np.float16).astype(np.float32)
        s1 = s0 * np.exp(rng.normal(0, 0.05, s0.shape)).astype(np.float32)
        nuove[nome] = s1
        attese[nome] = (blocchi.copy(), s1.astype(np.float16).astype(np.float32))
    cambiate = patcha_scale(dst, nuove)
    rep = verifica_patch(src, dst, list(nuove))
    rep_src = hashlib.sha256(src.read_bytes()).hexdigest()
    print(f"[T4] scale cambiate {cambiate} · byte fuori dai campi d identici · {rep}")
    rd = GGUFReader(str(dst))
    td = {t.name: t for t in rd.tensors}
    for nome, (blocchi_src, s16) in attese.items():
        t = td[nome]
        p = posizione_scale(t)
        blocchi_dst = np.asarray(t.data).reshape(-1, p["byte_blocco"])
        trit = trit_da_blocchi(torch.from_numpy(np.ascontiguousarray(blocchi_src)), p["tipo"]).numpy()
        if p["tipo"] == "TQ1_0":
            deq = GQ_TQ1_0.dequantize_blocks(np.ascontiguousarray(blocchi_dst))
            atteso = (trit.astype(np.float32).reshape(-1, 1, 256) * s16.reshape(-1, 1, 1)).reshape(-1, 256)
        else:
            deq = deq_r1024(np.ascontiguousarray(blocchi_dst))
            atteso = pesi_da_trit_scale(torch.from_numpy(trit), torch.from_numpy(s16), "TQ1_0R_1024", S).numpy()
        uguale = np.array_equal(deq, atteso)
        ok &= uguale
        print(f"[T4] {nome} {p['tipo']}: dequant del file patchato == trit(sorgente) x fp16(scala nuova) al bit: "
              f"{uguale} · forme/tipi/offset invariati: "
              f"{[int(x) for x in t.shape] == [int(x) for x in ts[nome].shape] and t.data_offset == ts[nome].data_offset}")
    print(f"[T4] sha256 sorgente {rep_src[:16]}… (il file sorgente non e' mai aperto in scrittura)")
    print("ESITO:", "TUTTO VERDE" if ok else "⛔ QUALCOSA NON TORNA")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    if a.test:
        sys.exit(0 if _test(Path(os.getenv("E2E_SCRATCH", str(QUI / "_e2e_scratch")))) else 1)
    ap.print_help()
