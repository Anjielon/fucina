#!/usr/bin/env python3
"""Collaudo di e2e_dump_logit, SOLO CPU. Scritto il 25/9/2026.

Riferimenti indipendenti (nessuno scritto per questo strumento):
  - `llama-perplexity` della stessa build: file --kl-divergence-base e PPL stampata;
  - `strumenti/dump_strati` (lo strumento esistente): logit dell'ULTIMO token e l_out;
  - numpy float64 sui pesi letti con gguf-py (router, norma finale, testa).
Due modelli: Qwen3-0.6B (vero, denso, 28 strati) e un qwen35moe minuscolo generato
(e2e_modello_minuscolo.py: GDN + attenzione piena + MoE + MTP, tokenizzatore di Thor).

Prove:
  T0  generatore del modello minuscolo deterministico (due generazioni, stesso sha256)
  T1  id di `tokenizza` == id nel file di llama-perplexity (entrambi i modelli); vocabolario
      di Thor-v20-densiv5 (solo vocabolario) == quello del minuscolo, stessi id
  T2  --uscite meta: file KLD IDENTICO AL BYTE a llama-perplexity, in due configurazioni di
      lotto (un micro-batch; 2 lotti x 2 micro-batch); PPL stampata identica
  T3  --uscite tutte: righe >= n/2 contro --uscite meta; KLD contro llama-perplexity
  T4  ultimo token contro dump_strati (result_output) e l_out contro dump_strati
  T5  isolamento fra sequenze (KV e stato GDN azzerati): s1 da solo == s1 dopo s0 == s1 dopo s2,
      logit + tutti gli l_out (+ ffn_moe_topk), al bit; logit con cb_eval == senza
  T6  coerenza interna in float64: router (ffn_moe_logits/probs/topk), norma finale, testa;
      allineamento righe logit <-> righe l_out
  T7  rifiuti: nome inesistente, id fuori vocabolario, file non int32, --lunghezza non
      divisibile, --kld-base con lunghezze diverse o senza posizioni n/2: codice 1, nessun manifesto
  T8  --lunghezza (un file, piu' sequenze) == file separati, al bit
  T9  RSS massimo di OGNI processo < 4 GiB (/usr/bin/time -v); GPU mai vista (0 dispositivi Vulkan)
  T10 lettore: un file troncato solleva
  T11 tensori su 2 lotti x 2 micro-batch == un micro-batch solo (4 pezzi, righe allineate)
  T12 --uscite meta con lotti da n/2: primo lotto senza uscite, pezzo vuoto contato

Uso:  CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 19 python3 e2e_test_dump_logit.py
      cartella: $E2E_DUMP_DIR (default @@ROOT:lavoro@@/e2e/dump_logit/collaudo, su disco, non /tmp);
      E2E_TIENI=1 per non cancellare i file grandi (logit, KLD) alla fine.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
sys.path.insert(0, "@@ROOT:models@@/build/llamacpp-tq1-k1024/gguf-py")
from e2e_dump_logit_leggi import Uscita, leggi_kld_base, leggi_tensore  # noqa: E402
from gguf import GGUFReader  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

BIN = QUI / "e2e_dump_logit"
DUMP_STRATI = QUI / "strumenti" / "dump_strati"
PPL = Path("@@ROOT:models@@/build/llamacpp-tq1-k1024/build/bin/llama-perplexity")
Q3 = Path("@@ROOT:models@@/ollama/.ollama/models/blobs/"
          "sha256-7f4030143c1c477224c5434f8272c662a8b042079a0a584f0a27a1684fe2e1fa")  # Qwen3-0.6B (turbo)
THOR = Path("@@ROOT:models@@/gguf/tony-tern/Thor-v20-densiv5.gguf")
TESTO = Path("@@ROOT:odino@@/collaudo/corpus/wiki.test.raw")
LAVORO = Path(os.environ.get("E2E_DUMP_DIR", "@@ROOT:lavoro@@/e2e/dump_logit/collaudo"))
TETTO_RSS_KIB = 4 * 1024 * 1024
AMBIENTE = dict(os.environ, CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="", GGML_VK_VISIBLE_DEVICES="",
                OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")

ESITI: list[dict] = []
MISURE: list[dict] = []


class Fallito(Exception):
    pass


def prova(nome: str, ok: bool, **dettagli) -> None:
    ESITI.append({"prova": nome, "ok": bool(ok), **dettagli})
    print(f"[{'OK ' if ok else 'NO '}] {nome}  {json.dumps(dettagli, default=str)[:400]}", flush=True)


def esegui(nome: str, argv: list, rc_atteso: int = 0, gpu_log: bool = True) -> str:
    cmd = ["/usr/bin/time", "-v", "nice", "-n", "19", "ionice", "-c3"] + [str(a) for a in argv]
    t0 = time.time()
    p = subprocess.run(cmd, cwd=LAVORO, env=AMBIENTE, capture_output=True, text=True)
    sec = time.time() - t0
    log = p.stdout + p.stderr
    (LAVORO / "log" / f"{nome}.log").write_text(log)
    m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", log)
    rss = int(m.group(1)) if m else -1
    # prova che nessuna GPU e' stata vista: la riga di ggml-vulkan (strumenti nostri) oppure, per
    # llama-perplexity (log di common, che quella riga non la mostra), il suo avviso
    gpu_zero = ("ggml_vulkan: Found 0 Vulkan devices" in log
                or "warning: no usable GPU found, --gpu-layers option will be ignored" in log)
    MISURE.append({"processo": nome, "rc": p.returncode, "secondi": round(sec, 2), "rss_max_kib": rss,
                   "vulkan_0_dispositivi": gpu_zero, "controllo_gpu": gpu_log and rc_atteso == 0})
    if p.returncode != rc_atteso:
        raise Fallito(f"{nome}: rc {p.returncode}, atteso {rc_atteso} (log: {LAVORO / 'log' / (nome + '.log')})")
    if gpu_log and rc_atteso == 0 and not gpu_zero:
        raise Fallito(f"{nome}: nel log manca la prova di 0 GPU ('Found 0 Vulkan devices' / 'no usable GPU found')")
    return log


def ppl_stampata(log: str) -> str:
    m = re.search(r"Final estimate: PPL = ([0-9.]+) \+/- ([0-9.]+)", log)
    if not m:
        raise Fallito("PPL finale non trovata nel log di llama-perplexity")
    return m.group(1)


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def stessi_byte(a: Path, b: Path) -> bool:
    return a.stat().st_size == b.stat().st_size and subprocess.run(["cmp", "-s", str(a), str(b)]).returncode == 0


def diff_max(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def al_bit(a, b) -> bool:
    a = np.asarray(a)
    b = np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def pesi(gguf: Path, nome: str) -> np.ndarray:
    r = GGUFReader(str(gguf))
    for t in r.tensors:
        if t.name == nome:
            if t.tensor_type.name in ("F32", "F16"):
                return np.array(t.data, dtype=np.float32)
            # a pezzi di righe: i temporanei di gguf-py su 150 M valori supererebbero i 4 GiB
            pezzi = [dequantize(np.asarray(t.data[a:a + 8192]), t.tensor_type).astype(np.float32)
                     for a in range(0, t.data.shape[0], 8192)]
            return np.concatenate(pezzi, axis=0)
    raise KeyError(nome)


def ha_tensore(gguf: Path, nome: str) -> bool:
    return any(t.name == nome for t in GGUFReader(str(gguf)).tensors)


def testa_float64(W: np.ndarray, h) -> np.ndarray:
    """h (r, D) x W (V, D)^T in float64, a pezzi di vocabolario (niente copia float64 di W intera)."""
    h = np.asarray(h, dtype=np.float64)
    out = np.empty((h.shape[0], W.shape[0]), dtype=np.float64)
    for a in range(0, W.shape[0], 16384):
        out[:, a:a + 16384] = h @ W[a:a + 16384].astype(np.float64).T
    return out


def tipo_tensore(gguf: Path, nome: str) -> str:
    return next(t.tensor_type.name for t in GGUFReader(str(gguf)).tensors if t.name == nome)


def attivazione_come_cpu(h, tipo_testa: str) -> np.ndarray:
    """L'attivazione che la CPU moltiplica davvero per una testa quantizzata (vec_dot_type di
    ggml-cpu.c della build k1024):
      Q8_0      -> q8_0: quantize_row_q8_0 x86 (arch/x86/quants.c:328-346): d = amax/127 salvato in
                   fp16, id = 127/amax, q = round-half-even(x*id) in float32;
      K-quant   -> q8_K: quantize_row_q8_K_ref (ggml-quants.c:3080-3112): max col segno,
                   iscale = -127/max, q = min(127, nearest_int(iscale*x)), d = 1/iscale."""
    h = np.asarray(h, np.float32)
    r, D = h.shape
    with np.errstate(divide="ignore", invalid="ignore"):
        if tipo_testa == "Q8_0":
            b = h.reshape(r, D // 32, 32)
            amax = np.abs(b).max(-1, keepdims=True)
            d16 = (amax / np.float32(127)).astype(np.float32).astype(np.float16).astype(np.float64)
            idv = np.where(amax != 0, np.float32(127) / amax, np.float32(0)).astype(np.float32)
            q = np.rint((b * idv).astype(np.float32))
            return (q * d16).reshape(r, D)
        if tipo_testa in ("Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K"):
            b = h.reshape(r, D // 256, 256)
            mx = np.take_along_axis(b, np.abs(b).argmax(-1)[..., None], -1)
            iscale = np.where(mx != 0, np.float32(-127) / mx, np.float32(0)).astype(np.float32)
            q = np.minimum(127, np.rint((b * iscale).astype(np.float32)))
            d = np.where(mx != 0, np.float32(1) / iscale, np.float32(0)).astype(np.float32).astype(np.float64)
            return (q * d).reshape(r, D)
    raise ValueError(f"testa {tipo_testa}: quantizzazione dell'attivazione non emulata")


def rmsnorm(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w.astype(np.float64)


# ─────────────────────────────────────────────────────────────────────────────────────────
def collaudo_modello(tag: str, modello: Path, n: int, strato_x: int, n_strati: int, moe: bool, eps: float) -> None:
    ids_dir = LAVORO / f"{tag}_ids"
    esegui(f"{tag}_tokenizza", [BIN, "tokenizza", modello, TESTO, n, ids_dir, 3])
    s = [ids_dir / f"seq_{i:04d}.i32" for i in range(3)]
    ultimo = n_strati - 1

    # ── riferimento: llama-perplexity in due configurazioni di lotto
    confs = {"A": (n, n), "B": (n // 2, n // 4)}
    ppl_rif = {}
    for c, (b, ub) in confs.items():
        log = esegui(f"{tag}_perplexity_{c}", [PPL, "-m", modello, "-f", TESTO, "-c", n, "-b", b, "-ub", ub,
                                               "--chunks", 3, "-t", 2, "-ngl", 0, "--device", "none",
                                               "--kl-divergence-base", LAVORO / f"{tag}_pp_{c}.kld"])
        ppl_rif[c] = ppl_stampata(log)

    # T1: id identici a quelli di llama-perplexity
    kb = leggi_kld_base(LAVORO / f"{tag}_pp_A.kld")
    tutti = np.fromfile(ids_dir / "token_tutti.i32", dtype=np.int32)
    uguali = all(np.array_equal(np.fromfile(s[i], dtype=np.int32), kb["tokens"][i * n:(i + 1) * n]) for i in range(3))
    prova(f"T1 {tag}: id di tokenizza == id di llama-perplexity", uguali and np.array_equal(tutti[:3 * n], kb["tokens"]),
          n_token_testo=int(tutti.size), n_ctx=kb["n_ctx"], n_chunk=kb["n_chunk"], n_vocab=kb["n_vocab"])

    # T2: --uscite meta, KLD al byte
    for c, (b, ub) in confs.items():
        esegui(f"{tag}_meta_{c}", [BIN, "logit", "-m", modello, "-o", LAVORO / f"{tag}_meta_{c}", "-t", 2, "-c", n,
                                   "-b", b, "-ub", ub, "--uscite", "meta", "--kld-base", LAVORO / f"{tag}_mio_{c}.kld", *s])
        man = Uscita(LAVORO / f"{tag}_meta_{c}").man
        mia = f"{man['kld_base']['ppl']:.4f}"
        prova(f"T2 {tag}: KLD al byte con llama-perplexity, lotto {b}/micro {ub}",
              stessi_byte(LAVORO / f"{tag}_mio_{c}.kld", LAVORO / f"{tag}_pp_{c}.kld") and mia == ppl_rif[c],
              byte=(LAVORO / f"{tag}_pp_{c}.kld").stat().st_size, ppl_perplexity=ppl_rif[c], ppl_strumento=mia,
              lotti_per_sequenza=(n + b - 1) // b, micro_batch_per_lotto=(b + ub - 1) // ub)

    # T3: --uscite tutte (grafo con tutte le uscite) contro meta
    esegui(f"{tag}_tutte_A", [BIN, "logit", "-m", modello, "-o", LAVORO / f"{tag}_tutte_A", "-t", 2, "-c", n, "-b", n,
                              "--uscite", "tutte", "--kld-base", LAVORO / f"{tag}_mio_tutte.kld", *s])
    ut, um = Uscita(LAVORO / f"{tag}_tutte_A"), Uscita(LAVORO / f"{tag}_meta_A")
    dmax, bit, rel = 0.0, True, 0.0
    for k in range(3):
        pt, Lt = ut.logit(k)
        pm, Lm = um.logit(k)
        sel = pt >= n // 2
        assert np.array_equal(pt[sel], pm)
        dmax = max(dmax, diff_max(Lt[sel], Lm))
        rel = max(rel, dmax / float(np.max(np.abs(Lm))))
        bit = bit and al_bit(Lt[sel], Lm)
    kld_bit = stessi_byte(LAVORO / f"{tag}_mio_tutte.kld", LAVORO / f"{tag}_pp_A.kld")
    prova(f"T3 {tag}: --uscite tutte, righe >= n/2 == --uscite meta (entro 1e-5 relativo)", rel <= 1e-5,
          al_bit=bit, diff_max_assoluta=dmax, diff_max_relativa=rel, kld_tutte_al_byte_con_perplexity=kld_bit,
          righe=int(ut.logit(0)[1].shape[0]))

    # T4: ultimo token e l_out contro dump_strati (un micro-batch, n_ctx = n + 16)
    ds = LAVORO / f"{tag}_dump_strati"
    ds.mkdir()
    esegui(f"{tag}_dump_strati", [DUMP_STRATI, modello, s[0], ds, 2, "result_output", f"l_out-{strato_x}", f"l_out-{ultimo}"])
    ro = leggi_tensore(ds / "result_output.f32")
    lx_ds = leggi_tensore(ds / f"l_out-{strato_x}.f32")
    lu_ds = leggi_tensore(ds / f"l_out-{ultimo}.f32")
    comuni = ["logit", "-m", modello, "-t", 2, "-c", n + 16, "-b", n, "-ub", n, "--l-out", f"{strato_x},{ultimo}", s[0]]
    esegui(f"{tag}_ultima", [BIN, *comuni[:1], "-o", LAVORO / f"{tag}_ultima", *comuni[1:], "--uscite", "ultima"])
    esegui(f"{tag}_tutte_1ub", [BIN, *comuni[:1], "-o", LAVORO / f"{tag}_tutte_1ub", *comuni[1:], "--uscite", "tutte"])
    uu, ua = Uscita(LAVORO / f"{tag}_ultima"), Uscita(LAVORO / f"{tag}_tutte_1ub")
    _, Lu = uu.logit(0)
    _, La = ua.logit(0)
    prova(f"T4 {tag}: ultimo token (--uscite ultima) == result_output di dump_strati, al bit",
          al_bit(Lu.reshape(-1), ro.reshape(-1)), n_vocab=int(ro.size))
    d_ult = diff_max(La[-1], ro.reshape(-1))
    prova(f"T4 {tag}: ultimo token con TUTTE le uscite == dump_strati (entro 1e-5 relativo)",
          d_ult / float(np.max(np.abs(ro))) <= 1e-5, al_bit=al_bit(La[-1], ro.reshape(-1)), diff_max=d_ult,
          nll_ultimo_non_definito=bool(np.isnan(ua.nll(0)[-1])))
    prova(f"T4 {tag}: l_out-{strato_x} (tutti i token) == dump_strati, al bit, in entrambe le modalita'",
          al_bit(uu.tensore(0, f"l_out-{strato_x}"), lx_ds) and al_bit(ua.tensore(0, f"l_out-{strato_x}"), lx_ds),
          forma=list(lx_ds.shape))
    lu_u, lu_a = uu.tensore(0, f"l_out-{ultimo}"), ua.tensore(0, f"l_out-{ultimo}")
    rig_u = uu.man["sequenze"][0]["tensori"][f"l_out-{ultimo}"]["righe_uguali_a"]
    rig_a = ua.man["sequenze"][0]["tensori"][f"l_out-{ultimo}"]["righe_uguali_a"]
    # dump_strati (solo l'ultimo in uscita) e --uscite ultima fanno lo stesso grafo: stesse righe
    ok_u = al_bit(lu_u, lu_ds)
    ok_a = diff_max(lu_a[-lu_ds.shape[0]:], lu_ds) <= 1e-5 * float(np.max(np.abs(lu_ds)))
    prova(f"T4 {tag}: l_out-{ultimo} (ultimo strato) == dump_strati; righe token/uscite come il grafo",
          ok_u and ok_a and lu_a.shape[0] == n, righe_dump_strati=int(lu_ds.shape[0]), righe_ultima=rig_u,
          righe_tutte=rig_a, al_bit_tutte=al_bit(lu_a[-lu_ds.shape[0]:], lu_ds))

    # T5: isolamento fra sequenze + cb_eval non cambia i conti
    extra = ["--l-out", "tutti"] + (["--topk", "--tensore", "ffn_moe_logits*", "--tensore", "ffn_moe_probs*",
                                     "--tensore", "attn_post_norm*"] if moe else []) + ["--tensore", "result_norm"]
    base = ["logit", "-m", modello, "-t", 2, "-c", n, "-b", n]
    esegui(f"{tag}_iso_012", [BIN, *base, "-o", LAVORO / f"{tag}_iso_012", *extra, s[0], s[1], s[2]])
    esegui(f"{tag}_iso_1", [BIN, *base, "-o", LAVORO / f"{tag}_iso_1", *extra, s[1]])
    esegui(f"{tag}_iso_21", [BIN, *base, "-o", LAVORO / f"{tag}_iso_21", *extra, s[2], s[1]])
    u012, u1, u21 = (Uscita(LAVORO / f"{tag}_iso_{x}") for x in ("012", "1", "21"))
    nomi = sorted(u1.man["sequenze"][0]["tensori"])
    ok = al_bit(u012.logit(1)[1], u1.logit(0)[1]) and al_bit(u21.logit(1)[1], u1.logit(0)[1])
    diversi = []
    for nm in nomi:
        if not (al_bit(u012.tensore(1, nm), u1.tensore(0, nm)) and al_bit(u21.tensore(1, nm), u1.tensore(0, nm))):
            diversi.append(nm)
    prova(f"T5 {tag}: s1 da sola == s1 dopo s0 == s1 dopo s2 (logit + {len(nomi)} tensori), al bit",
          ok and not diversi, tensori=len(nomi), diversi=diversi,
          strati_l_out=u1.man["sequenze"][0]["prefissi"]["l_out-"]["strati"])
    cb_bit = all(al_bit(u012.logit(k)[1], ut.logit(k)[1]) for k in range(3))
    cb_d = max(diff_max(u012.logit(k)[1], ut.logit(k)[1]) for k in range(3))
    prova(f"T5 {tag}: logit con tensori richiesti (cb_eval) == senza (entro 1e-5 relativo)",
          cb_d <= 1e-5 * float(np.max(np.abs(ut.logit(0)[1]))), al_bit=cb_bit, diff_max=cb_d)

    # T11: tensori concatenati su 2 lotti x 2 micro-batch == un micro-batch solo (righe allineate)
    esegui(f"{tag}_micro", [BIN, "logit", "-m", modello, "-t", 2, "-c", n, "-b", n // 2, "-ub", n // 4,
                            "-o", LAVORO / f"{tag}_micro", *extra, s[1]])
    um = Uscita(LAVORO / f"{tag}_micro")
    tm = um.man["sequenze"][0]["tensori"]
    pezzi_ok = all(v["pezzi"] == 4 and v["righe"] == n and v["righe_pezzo"] == [n // 4] * 4 for v in tm.values())
    peggiore, nome_p, sfas_min, bit_tutti = 0.0, None, float("inf"), True
    for nm in nomi:
        a = np.asarray(um.tensore(0, nm))
        b1 = np.asarray(u1.tensore(0, nm))
        if a.dtype.kind == "f":
            d = diff_max(a, b1) / float(np.max(np.abs(b1)))
            sf = diff_max(a[1:], b1[:-1]) / float(np.max(np.abs(b1)))
            sfas_min = min(sfas_min, sf)
            if d > peggiore:
                peggiore, nome_p = d, nm
        bit_tutti = bit_tutti and al_bit(a, b1)
    topk_acc = None
    if moe:
        tot = [(np.sort(np.asarray(um.tensore(0, f"ffn_moe_topk-{il}")), 1) ==
                np.sort(np.asarray(u1.tensore(0, f"ffn_moe_topk-{il}")), 1)).all(1).mean() for il in range(n_strati)]
        topk_acc = float(min(tot))
    dl = diff_max(um.logit(0)[1], u1.logit(0)[1]) / float(np.max(np.abs(u1.logit(0)[1])))
    prova(f"T11 {tag}: 2 lotti x 2 micro-batch == un micro-batch (tensori e logit entro 1e-4 relativo, righe allineate)",
          pezzi_ok and peggiore <= 1e-4 and dl <= 1e-4 and sfas_min > 100 * max(peggiore, 1e-12)
          and (topk_acc is None or topk_acc >= 0.99),
          pezzi_e_righe_ok=pezzi_ok, diff_rel_peggiore=peggiore, tensore_peggiore=nome_p, diff_rel_logit=dl,
          diff_rel_sfasata_min=sfas_min, al_bit=bit_tutti, topk_insieme_uguale_min=topk_acc)

    # T12: --uscite meta con lotti da n/2: il primo lotto non ha uscite (pezzo vuoto)
    esegui(f"{tag}_meta_tagli", [BIN, "logit", "-m", modello, "-t", 2, "-c", n, "-b", n // 2,
                                 "-o", LAVORO / f"{tag}_meta_tagli", "--uscite", "meta", "--l-out", ultimo, s[1]])
    ug = Uscita(LAVORO / f"{tag}_meta_tagli")
    tg = ug.man["sequenze"][0]["tensori"][f"l_out-{ultimo}"]
    lg_ = np.asarray(ug.tensore(0, f"l_out-{ultimo}"))
    rif_ = np.asarray(u1.tensore(0, f"l_out-{ultimo}"))[n // 2:] if tg["righe_uguali_a"] == "uscite" \
        else np.asarray(u1.tensore(0, f"l_out-{ultimo}"))
    dg = diff_max(lg_, rif_) / float(np.max(np.abs(rif_)))
    prova(f"T12 {tag}: --uscite meta, lotti da n/2: l_out-{ultimo} con pezzo iniziale vuoto se tagliato",
          dg <= 1e-4 and tg["pezzi"] == 2, righe=tg["righe"], righe_uguali_a=tg["righe_uguali_a"],
          righe_pezzo=tg["righe_pezzo"], diff_rel=dg)

    # T6: coerenza interna in float64
    w_norm = pesi(modello, "output_norm.weight")
    testa = "output.weight" if ha_tensore(modello, "output.weight") else "token_embd.weight"
    W = pesi(modello, testa)
    rn = u1.tensore(0, "result_norm")
    lo = u1.tensore(0, f"l_out-{ultimo}")
    d_rn = diff_max(rn, rmsnorm(lo, w_norm, eps)) / float(np.max(np.abs(rn)))
    _, L1 = u1.logit(0)
    tipo_t = tipo_tensore(modello, testa)
    righe = np.linspace(1, n - 1, 16).astype(int)

    def errore(rif):
        Lr = np.asarray(L1[righe], np.float64)
        return np.linalg.norm(Lr - rif, axis=1) / np.linalg.norm(rif, axis=1)

    # (a) testa in float64 su result_norm float: e' il conto «alla torch», senza la quantizzazione
    #     dell'attivazione che la CPU fa -> misura quanto pesa quel passo
    err_f = errore(testa_float64(W, rn[righe]))
    # (b) testa su result_norm quantizzato ESATTAMENTE come la CPU (riga giusta e riga sfasata di 1)
    err = errore(testa_float64(W, attivazione_come_cpu(rn[righe], tipo_t)))
    err_sp = errore(testa_float64(W, attivazione_come_cpu(rn[righe - 1], tipo_t)))
    del W
    prova(f"T6 {tag}: result_norm == rmsnorm(l_out-{ultimo}) in float64 (entro 1e-5 relativo)", d_rn <= 1e-5,
          diff_rel=d_rn)
    prova(f"T6 {tag}: logit == testa {tipo_t} x attivazione quantizzata come la CPU (entro 1e-4); righe allineate",
          float(err.max()) <= 1e-4 and float(err_sp.min()) >= 100 * float(err.max()),
          errore_rel_max=float(err.max()), errore_rel_mediano=float(np.median(err)),
          errore_sfasato_min=float(err_sp.min()),
          errore_senza_quantizzare_attivazione_max=float(err_f.max()),
          errore_senza_quantizzare_attivazione_mediano=float(np.median(err_f)))
    if moe:
        d_lg, d_pr, top_ok, top_tot, pareggi, ok_ind, n_ind = 0.0, 0.0, 0, 0, 0, 0, 0
        for il in range(n_strati):
            Wr = pesi(modello, f"blk.{il}.ffn_gate_inp.weight")          # (E, D)
            apn = np.asarray(u1.tensore(0, f"attn_post_norm-{il}"), np.float64)
            lg = np.asarray(u1.tensore(0, f"ffn_moe_logits-{il}"), np.float64)
            pr = np.asarray(u1.tensore(0, f"ffn_moe_probs-{il}"), np.float64)
            tk = np.asarray(u1.tensore(0, f"ffn_moe_topk-{il}"))
            lg64 = apn @ Wr.T.astype(np.float64)
            d_lg = max(d_lg, diff_max(lg, lg64) / float(np.max(np.abs(lg64))))
            pr64 = np.exp(lg64 - lg64.max(1, keepdims=True))
            pr64 /= pr64.sum(1, keepdims=True)
            d_pr = max(d_pr, diff_max(pr, pr64))
            k = tk.shape[1]
            ordine = np.argsort(-pr, axis=1, kind="stable")[:, :k]
            top_ok += int((ordine == tk).all(1).sum())
            top_tot += tk.shape[0]
            srt = -np.sort(-pr, axis=1)
            pareggi += int((srt[:, k - 1] == srt[:, k]).sum())
            ind = np.sort(np.argsort(-pr64, axis=1, kind="stable")[:, :k], 1)
            ok_ind += int((ind == np.sort(tk, 1)).all(1).sum())
            n_ind += tk.shape[0]
        prova("T6 minuscolo: ffn_moe_logits == W_router x attn_post_norm; probs == softmax (float64)",
              d_lg <= 1e-5 and d_pr <= 1e-6, diff_rel_logit=d_lg, diff_prob=d_pr)
        prova("T6 minuscolo: ffn_moe_topk == primi k di ffn_moe_probs (ordine compreso), tutti i token/strati",
              top_ok == top_tot, uguali=top_ok, totale=top_tot, pareggi_k_k1=pareggi)
        prova("T6 minuscolo: insieme top-k == quello del router ricalcolato in float64",
              ok_ind == n_ind, uguali=ok_ind, totale=n_ind)


def rne_bf16(x: np.ndarray) -> np.ndarray:
    """f32 -> bit bf16, arrotondamento al piu' vicino pari: la formula di ggml_compute_fp32_to_bf16
    (ggml-impl.h della build k1024), scritta qui in numpy come riferimento indipendente. Solo finiti."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def collaudo_26_9(mq: Path, s: list) -> None:
    """26/9: --salva-tipo (T13), --segmenti (T14), rifiuti delle due opzioni (T15). Modello minuscolo, CPU."""
    n = 256
    extra = ["--l-out", "tutti", "--topk", "--tensore", "h_nextn", "--tensore", "result_norm"]
    base = [BIN, "logit", "-m", mq, "-t", 2, "-c", n, "-b", n, "--uscite", "tutte"]
    # T13: stessi tensori richiesti, tre tipi su disco
    for tipo in ("f32", "bf16", "f16"):
        esegui(f"st_{tipo}", [*base, "-o", LAVORO / f"st_{tipo}", *extra, "--salva-tipo", tipo, s[1]])
    u32, ubf, uf16 = (Uscita(LAVORO / f"st_{t}") for t in ("f32", "bf16", "f16"))
    t32 = u32.man["sequenze"][0]["tensori"]
    diversi, conv, controllati = [], 0, 0
    for nm, meta in t32.items():
        a = u32.tensore(0, nm, mmap=False)
        b = ubf.tensore(0, nm, mmap=False)
        c = uf16.tensore(0, nm, mmap=False)
        mb, mc = ubf.man["sequenze"][0]["tensori"][nm], uf16.man["sequenze"][0]["tensori"][nm]
        controllati += 1
        if meta["tipo"] == "f32":
            conv += 1
            ok = (mb["tipo"] == "bf16" and mb["file"].endswith(".bf16") and b.dtype == np.uint16
                  and al_bit(b, rne_bf16(a)) and mc["tipo"] == "f16" and mc["file"].endswith(".f16")
                  and al_bit(c, a.astype(np.float16)) and mb["ne"] == meta["ne"] == mc["ne"]
                  and mb["tipo_grafo"] == mc["tipo_grafo"] == "f32")
        else:  # i32 (topk) e f16 del grafo: non si toccano
            ok = al_bit(a, b) and al_bit(a, c) and mb["tipo"] == mc["tipo"] == meta["tipo"]
        if not ok:
            diversi.append(nm)
    logit_uguali = al_bit(u32.logit(0)[1], ubf.logit(0)[1]) and al_bit(u32.logit(0)[1], uf16.logit(0)[1])
    prova("T13 --salva-tipo: bf16 == RNE(f32) e f16 == f32.astype(float16) AL BIT, i32 intatti, logit f32 identici",
          not diversi and logit_uguali and conv >= 2, tensori=controllati, convertiti=conv, diversi=diversi,
          salva_tipo=[ubf.man["contesto"]["salva_tipo"], uf16.man["contesto"]["salva_tipo"]])

    # T14: segmenti [0,70) [70,200) [200,256) == tre sequenze separate, al bit; un solo segmento == senza
    ids = np.fromfile(s[0], dtype=np.int32)
    tagli = [0, 70, 200, n]
    fl = np.zeros(n, np.int32)
    fl[tagli[:-1]] = 1
    seg = LAVORO / "segmenti_3.i32"
    fl.tofile(seg)
    uno = LAVORO / "segmenti_1.i32"
    f1 = np.zeros(n, np.int32)
    f1[0] = 1
    f1.tofile(uno)
    pezzi = []
    for j in range(3):
        p = LAVORO / f"pezzo_{j}.i32"
        ids[tagli[j]:tagli[j + 1]].tofile(p)
        pezzi.append(p)
    esegui("seg_tre", [*base, "-o", LAVORO / "seg_tre", *extra, "--segmenti", seg, s[0]])
    esegui("seg_separati", [*base, "-o", LAVORO / "seg_separati", *extra, *pezzi])
    esegui("seg_uno", [*base, "-o", LAVORO / "seg_uno", *extra, "--segmenti", uno, s[0]])
    esegui("seg_nessuno", [*base, "-o", LAVORO / "seg_nessuno", *extra, s[0]])
    ut, us, u1, u0 = (Uscita(LAVORO / x) for x in ("seg_tre", "seg_separati", "seg_uno", "seg_nessuno"))
    nomi = sorted(ut.man["sequenze"][0]["tensori"])
    L_t = ut.logit(0)[1]
    L_s = np.concatenate([us.logit(j)[1] for j in range(3)])
    diversi = [nm for nm in nomi
               if not al_bit(ut.tensore(0, nm), np.concatenate([us.tensore(j, nm) for j in range(3)], axis=0))]
    prova("T14 --segmenti: una sequenza con 3 segmenti == 3 sequenze separate (logit + tensori), AL BIT",
          al_bit(L_t, L_s) and not diversi and ut.man["sequenze"][0]["segmenti"] == tagli[:-1],
          tensori=len(nomi), diversi=diversi, segmenti=ut.man["sequenze"][0].get("segmenti"))
    diversi1 = [nm for nm in nomi if not al_bit(u1.tensore(0, nm), u0.tensore(0, nm))]
    prova("T14 --segmenti con un solo inizio == senza --segmenti, AL BIT",
          al_bit(u1.logit(0)[1], u0.logit(0)[1]) and not diversi1, diversi=diversi1)
    # controllo che i segmenti agiscano davvero: dal secondo segmento in poi i logit cambiano
    cambia = [int(not al_bit(L_t[tagli[j]:tagli[j + 1]], u0.logit(0)[1][tagli[j]:tagli[j + 1]])) for j in (1, 2)]
    prova("T14 i segmenti 2 e 3 danno logit diversi dalla sequenza intera (i confini agiscono)", all(cambia),
          diff_max_seg2=diff_max(L_t[70:200], u0.logit(0)[1][70:200]))

    # T15 rifiuti delle opzioni nuove
    corto = LAVORO / "segmenti_corto.i32"
    fl[:-1].tofile(corto)
    zero = LAVORO / "segmenti_zero.i32"
    fz = fl.copy()
    fz[0] = 0
    fz.tofile(zero)
    due = LAVORO / "segmenti_due.i32"
    fd = fl.copy()
    fd[9] = 2
    fd.tofile(due)
    casi = {
        "--segmenti di lunghezza diversa": ["--segmenti", corto, s[0]],
        "--segmenti che non inizia con 1": ["--segmenti", zero, s[0]],
        "--segmenti con valore 2": ["--segmenti", due, s[0]],
        "--segmenti: 1 file per 2 file di id": ["--segmenti", seg, s[0], s[1]],
        "--segmenti con --kld-base": ["--segmenti", seg, "--kld-base", LAVORO / "no3.kld", "--uscite", "meta", s[0]],
        "--salva-tipo sconosciuto": ["--salva-tipo", "f8", s[0]],
    }
    for i, (nome, argv) in enumerate(casi.items()):
        out = LAVORO / f"rifiuto26_{i}"
        log = esegui(f"rifiuto26_{i}", [BIN, "logit", "-m", mq, "-o", out, "-t", 2, *argv], rc_atteso=1, gpu_log=False)
        msg = [r for r in log.splitlines() if r.startswith("ERRORE")]
        prova(f"T15 rifiuto: {nome}", not (out / "manifesto.json").exists() and bool(msg), messaggio=msg[:1])


def collaudo_rifiuti(modello: Path, ids: Path, n: int) -> None:
    corti = LAVORO / "corti.i32"
    corti.write_bytes(np.fromfile(ids, dtype=np.int32)[: n // 2].tobytes())
    fuori = LAVORO / "fuori.i32"
    a = np.fromfile(ids, dtype=np.int32).copy()
    a[5] = 10_000_000
    fuori.write_bytes(a.tobytes())
    storto = LAVORO / "storto.i32"
    storto.write_bytes(ids.read_bytes()[:-2])
    casi = {
        "nome inesistente": ["--tensore", "non_esiste-3", ids],
        "id fuori vocabolario": [fuori],
        "file non int32": [storto],
        "--lunghezza non divisibile": ["--lunghezza", n - 1, ids],
        "--kld-base, lunghezze diverse": ["--kld-base", LAVORO / "no.kld", ids, corti],
        "--kld-base con --uscite ultima": ["--kld-base", LAVORO / "no2.kld", "--uscite", "ultima", ids],
        "prefisso senza tensori": ["--tensore", "niente_del_genere*", ids],
    }
    for i, (nome, argv) in enumerate(casi.items()):
        out = LAVORO / f"rifiuto_{i}"
        log = esegui(f"rifiuto_{i}", [BIN, "logit", "-m", modello, "-o", out, "-t", 2, *argv], rc_atteso=1, gpu_log=False)
        msg = [r for r in log.splitlines() if r.startswith("ERRORE")]
        prova(f"T7 rifiuto: {nome}", not (out / "manifesto.json").exists(), messaggio=msg[:1])


def main() -> int:
    t0 = time.time()
    assert LAVORO.name == "collaudo" and str(LAVORO).startswith("@@ROOT:lavoro@@/"), "cartella di collaudo inattesa"
    if LAVORO.exists():
        shutil.rmtree(LAVORO)
    (LAVORO / "log").mkdir(parents=True)
    esito = {"versione_strumento": None, "prove": ESITI, "processi": MISURE}
    try:
        esito["versione_strumento"] = subprocess.run([BIN], capture_output=True, text=True).stderr.splitlines()[0]
        esito["sha256_binario"] = sha(BIN)
        esito["sha256_sorgente"] = sha(QUI / "e2e_dump_logit.cpp")

        # T0 modello minuscolo deterministico
        mq = LAVORO / "minuscolo-qwen35moe.gguf"
        mq2 = LAVORO / "minuscolo-bis.gguf"
        for p in (mq, mq2):
            esegui(f"genera_{p.stem}", [sys.executable, QUI / "e2e_modello_minuscolo.py", p], gpu_log=False)
        prova("T0 modello minuscolo deterministico (due generazioni)", sha(mq) == sha(mq2), sha256=sha(mq),
              byte=mq.stat().st_size)
        mq2.unlink()
        Path(str(mq2) + ".json").unlink()

        eps_q3 = float(GGUFReader(str(Q3)).fields["qwen3.attention.layer_norm_rms_epsilon"].contents())
        collaudo_modello("minuscolo", mq, 256, 1, 4, True, 1e-6)
        collaudo_modello("qwen3", Q3, 512, 10, 28, False, eps_q3)

        # T1 vocabolario di Thor: stessi id del minuscolo (che porta il suo tokenizzatore)
        esegui("thor_tokenizza", [BIN, "tokenizza", THOR, TESTO, 256, LAVORO / "thor_ids", 3])
        uguali = all(stessi_byte(LAVORO / "thor_ids" / f"seq_{i:04d}.i32", LAVORO / "minuscolo_ids" / f"seq_{i:04d}.i32")
                     for i in range(3)) and stessi_byte(LAVORO / "thor_ids" / "token_tutti.i32",
                                                        LAVORO / "minuscolo_ids" / "token_tutti.i32")
        prova("T1 Thor-v20-densiv5 (solo vocabolario): id == minuscolo == llama-perplexity sul minuscolo", uguali,
              rss_kib=MISURE[-1]["rss_max_kib"])

        # T8 --lunghezza
        s = [LAVORO / "minuscolo_ids" / f"seq_{i:04d}.i32" for i in range(3)]
        cat = LAVORO / "tre_seq.i32"
        cat.write_bytes(b"".join(p.read_bytes() for p in s))
        esegui("minuscolo_lunghezza", [BIN, "logit", "-m", mq, "-o", LAVORO / "minuscolo_lunghezza", "-t", 2, "-c", 256,
                                       "-b", 256, "--uscite", "tutte", "--lunghezza", 256, cat])
        ul, ut = Uscita(LAVORO / "minuscolo_lunghezza"), Uscita(LAVORO / "minuscolo_tutte_A")
        prova("T8 --lunghezza: un file da 3 x 256 == tre file, logit al bit",
              ul.n_sequenze == 3 and all(al_bit(ul.logit(k)[1], ut.logit(k)[1]) for k in range(3)),
              offset=[q["offset_token"] for q in ul.man["sequenze"]])

        collaudo_rifiuti(mq, s[0], 256)

        # T10 lettore: file troncato
        src = LAVORO / "minuscolo_iso_1" / "seq_0000" / "l_out-0.f32"
        tronco = LAVORO / "tronco.f32"
        tronco.write_bytes(src.read_bytes()[:-4])
        try:
            leggi_tensore(tronco)
            prova("T10 lettore: file troncato solleva", False)
        except ValueError as e:
            prova("T10 lettore: file troncato solleva", True, messaggio=str(e)[:120])

        # 26/9: --salva-tipo e --segmenti (strumento versione 2)
        collaudo_26_9(mq, s)
    except Fallito as e:
        prova("ESECUZIONE", False, errore=str(e))

    rss = [m["rss_max_kib"] for m in MISURE]
    import resource
    rss_io = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    llama = [m for m in MISURE if m["controllo_gpu"]]
    gpu_mai = all(m["vulkan_0_dispositivi"] for m in llama)
    prova("T9 RSS massimo di ogni processo < 4 GiB (figli e questo script); 0 dispositivi Vulkan in ogni log llama",
          all(0 < r < TETTO_RSS_KIB for r in rss) and rss_io < TETTO_RSS_KIB and gpu_mai,
          rss_max_figli_kib=max(rss) if rss else None, rss_max_script_kib=rss_io, processi=len(rss),
          processi_llama_con_0_vulkan=f"{sum(m['vulkan_0_dispositivi'] for m in llama)}/{len(llama)}")
    esito["tutto_verde"] = all(e["ok"] for e in ESITI)
    esito["secondi"] = round(time.time() - t0, 1)
    if os.environ.get("E2E_TIENI") != "1":
        via = 0
        for p in list(LAVORO.rglob("logit.f32")) + list(LAVORO.glob("*.kld")) + list(LAVORO.rglob("*.gguf")):
            via += p.stat().st_size
            p.unlink()
        esito["byte_cancellati"] = via
    (LAVORO / "esito_collaudo.json").write_text(json.dumps(esito, indent=1, default=str))
    print(f"\nTUTTO VERDE: {esito['tutto_verde']}  ({len(ESITI)} prove, {esito['secondi']} s)  "
          f"-> {LAVORO / 'esito_collaudo.json'}")
    return 0 if esito["tutto_verde"] else 1


if __name__ == "__main__":
    sys.exit(main())
