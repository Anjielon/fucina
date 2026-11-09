#!/usr/bin/env python3
"""Lettore delle uscite di `e2e_dump_logit` (formato documentato in testa a e2e_dump_logit.cpp).
Scritto il 25/9/2026 per la fase E2E di Thor (F0a/F0b/F0c). Solo numpy, nessun import pesante.

    from e2e_dump_logit_leggi import Uscita, leggi_tensore, leggi_kld_base
    u = Uscita("@@ROOT:lavoro@@/e2e/.../corsa")          # solleva se manifesto.json manca
    pos, L = u.logit(0)      # posizioni (int) e logit memmap float32 [righe, n_vocab]
    h = u.tensore(0, "l_out-39")                     # [n_token, n_embd]
    tk = u.tensore(0, "ffn_moe_topk-3")              # int32 [n_token, k]

Un file tensore e' 4 x int64 (ne0 ne1 ne2 ne3) + dati contigui del tipo dell'estensione
(.f32 .f16 .bf16 .i32 .f64). In numpy l'ordine degli assi e' rovesciato (ne3, ne2, ne1, ne0) e gli
assi unitari in testa oltre (ne1, ne0) si tolgono: [x, token] -> (token, x), sempre 2D anche con
una riga sola. La taglia del file e' controllata:
un file troncato solleva invece di restituire dati parziali.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# 26/9: ".bf16" (e2e_dump_logit --salva-tipo bf16) si legge come uint16 GREZZO (numpy non ha bf16):
# bf16_a_f32() lo porta a float32 esatto, torch.from_numpy(a).view(torch.bfloat16) a torch.
_TIPI = {".f32": np.float32, ".f16": np.float16, ".bf16": np.uint16, ".i32": np.int32, ".f64": np.float64}


def bf16_a_f32(a: np.ndarray) -> np.ndarray:
    """bit bf16 (uint16) -> float32 esatto (26/9)."""
    a = np.asarray(a)
    if a.dtype != np.uint16:
        raise TypeError(f"attesi i bit bf16 come uint16, non {a.dtype}")
    return (a.astype(np.uint32) << 16).view(np.float32)


def leggi_tensore(percorso, mmap: bool = True) -> np.ndarray:
    p = Path(percorso)
    if p.suffix not in _TIPI:
        raise ValueError(f"{p}: estensione {p.suffix} sconosciuta")
    dt = np.dtype(_TIPI[p.suffix])
    ne = np.fromfile(p, dtype=np.int64, count=4)
    if ne.size != 4 or (ne < 0).any():
        raise ValueError(f"{p}: intestazione non valida {ne}")
    n = int(np.prod(ne))
    atteso = 32 + n * dt.itemsize
    if p.stat().st_size != atteso:
        raise ValueError(f"{p}: {p.stat().st_size} byte, attesi {atteso} (file troncato o intestazione errata)")
    forma = tuple(int(x) for x in ne[::-1])
    while len(forma) > 2 and forma[0] == 1:  # restano sempre almeno (ne1, ne0)
        forma = forma[1:]
    if mmap:
        return np.memmap(p, dtype=dt, mode="r", offset=32, shape=forma)
    return np.fromfile(p, dtype=dt, offset=32).reshape(forma)


class Uscita:
    """Una corsa completa di `e2e_dump_logit logit` (manifesto.json presente)."""

    def __init__(self, cartella):
        self.dir = Path(cartella)
        m = self.dir / "manifesto.json"
        if not m.exists():
            raise FileNotFoundError(f"{m}: manca — corsa incompleta o fallita")
        self.man = json.loads(m.read_text())
        if self.man.get("esito") != "ok":
            raise RuntimeError(f"{m}: esito {self.man.get('esito')!r}")

    @property
    def n_sequenze(self) -> int:
        return len(self.man["sequenze"])

    def _seq(self, k: int) -> dict:
        return self.man["sequenze"][k]

    def ids(self, k: int) -> np.ndarray:
        return np.fromfile(self.dir / self._seq(k)["cartella"] / "ids.i32", dtype=np.int32)

    def logit(self, k: int, mmap: bool = True):
        s = self._seq(k)
        L = leggi_tensore(self.dir / s["logit"]["file"], mmap=mmap)
        L = L.reshape(-1, self.man["modello"]["n_vocab"])
        if s["logit"]["contigue"]:
            pos = np.arange(s["logit"]["posizione_prima"], s["logit"]["posizione_ultima"] + 1)
        else:  # nessuna modalita' attuale produce posizioni non contigue
            raise NotImplementedError("posizioni non contigue")
        if len(pos) != L.shape[0]:
            raise ValueError("righe dei logit incoerenti col manifesto")
        return pos, L

    def nll(self, k: int) -> np.ndarray:
        return leggi_tensore(self.dir / self._seq(k)["nll"]["file"], mmap=False).reshape(-1)

    def tensore(self, k: int, nome: str, mmap: bool = True) -> np.ndarray:
        t = self._seq(k)["tensori"][nome]
        return leggi_tensore(self.dir / t["file"], mmap=mmap)


def leggi_kld_base(percorso) -> dict:
    """File di `llama-perplexity --kl-divergence-base` (o `e2e_dump_logit --kld-base`):
    "_logits_", int32 n_ctx, int32 n_vocab, int32 n_chunk, int32 token[n_chunk*n_ctx], poi per
    ogni pezzo n_ctx-1-n_ctx/2 record da nv = 2*((n_vocab+1)/2)+4 uint16: [scala f32][min f32]
    [n_vocab uint16 quantizzati]. log p_i = min + q_i*scala per i logit > max-16."""
    p = Path(percorso)
    with open(p, "rb") as f:
        if f.read(8) != b"_logits_":
            raise ValueError(f"{p}: magia errata")
        n_ctx, n_vocab, n_chunk = np.fromfile(f, dtype=np.int32, count=3).tolist()
    off_tok = 8 + 12
    tokens = np.memmap(p, dtype=np.int32, mode="r", offset=off_tok, shape=(n_chunk * n_ctx,))
    nv = 2 * ((n_vocab + 1) // 2) + 4
    n_rec = n_chunk * (n_ctx - 1 - n_ctx // 2)
    off_rec = off_tok + 4 * n_chunk * n_ctx
    atteso = off_rec + 2 * nv * n_rec
    if p.stat().st_size != atteso:
        raise ValueError(f"{p}: {p.stat().st_size} byte, attesi {atteso}")
    rec = np.memmap(p, dtype=np.uint16, mode="r", offset=off_rec, shape=(n_rec, nv))
    return {"n_ctx": n_ctx, "n_vocab": n_vocab, "n_chunk": n_chunk, "tokens": tokens, "record": rec, "nv": nv}
