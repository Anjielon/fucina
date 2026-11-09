#!/usr/bin/env python3
"""Prototipo a scala giocattolo, SOLO CPU, della fase E2E sulle scale fp16 degli esperti
(progetto scelto: `e2e_progetto_scelto.md`). Scritto il 25/9/2026. File NUOVO: nessun file
esistente della forgia e' modificato; si importano in sola lettura il codec di casa
(`kernel-rotazione/tq1_0r_1024_codec.py`), il packer `pack_tq1_0_standard.py`, la FWHT torch
di `rotazione_gpu.py` e i mattoni di `e2e_codici_scale.py` (decode dai byte, patch e verifica).

Cosa riproduce, in piccolo, del progetto:
  - un MoE vero (attenzione causale, router softmax top-k rinormalizzato, esperti SwiGLU,
    esperto condiviso con porta sigmoide) il cui MAESTRO bf16 e' addestrato davvero, a byte,
    su testo pubblico e versionato (i sorgenti della libreria standard di Python 3.14);
  - lo STUDENTE ha gli esperti nei formati di Thor: gate/up `TQ1_0R_1024` (rotazione
    FWHT-1024 col segno S1024, 8 scale fp16 per blocco da 1024, una ogni 128 pesi RUOTATI),
    down `TQ1_0` (una scala fp16 ogni 256 pesi). I codici sono i BYTE dei blocchi, decodificati
    in trit a ogni passata: in memoria non esiste una seconda copia dei codici;
  - si allenano SOLO le scale: d_eff = fp16_RNE(d0 * exp(theta)), theta fp32 a zero, con
    straight-through sull'arrotondamento (lo studente allenato E' il modello esportato);
  - ricetta B («come il motore»): per gate/up si ruota l'ATTIVAZIONE (R x = FWHT(S*x)) e si
    moltiplica per w_rot = d*trit, come fa lo shader Vulkan per n_token > 8;
  - perdita: KL(maestro || studente) in avanti, esatta sul vocabolario intero, a pezzi di
    token, gradiente in forma chiusa; il maestro e' in cache come STATO NASCOSTO finale bf16
    dopo la norma (i logit si ricalcolano con la sua testa bf16);
  - un checkpoint per strato: avanti senza grafo, poi ricalcolo strato per strato con
    l'instradamento top-k SALVATO e riusato (contatore dei cambi di instradamento);
  - AdamW con weight_decay = 0 esplicito, beta (0,9, 0,95), lr COSTANTE, eps scelto dalla
    misura dei gradienti al passo 0 (cancello G1), held-out ogni N passi e si tiene il migliore;
  - ripresa (theta + stato Adam + cursore dei dati + migliore) collaudata;
  - export: copia del GGUF sorgente e riscrittura dei SOLI campi d, verifica byte per byte.

Uso:  CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 19 python3 e2e_prototipo.py
      (i test: e2e_test_prototipo.py). Cartella di lavoro: $E2E_PROTO_DIR.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
sys.path.insert(0, "@@ROOT:forge@@/kernel-rotazione")
sys.path.insert(0, "@@ROOT:models@@/build/llamacpp-tq1-k1024/gguf-py")

from gguf import GGMLQuantizationType as GT, GGUFReader, GGUFWriter  # noqa: E402
from tq1_0r_1024_codec import GRIGLIA_LS, S1024, quantize_rows  # noqa: E402  (sola lettura)
from pack_tq1_0_standard import pack_tq1_0_da_d_q  # noqa: E402  (sola lettura)
from rotazione_gpu import fwht1024_torch  # noqa: E402  (sola lettura)
import e2e_codici_scale as CS  # noqa: E402

DIR_DEFAULT = "/tmp/claude-1000/-home-angelo-MOGAVIS/12e4589a-507e-45a0-b690-66b786e2b26f/scratchpad/e2e/prototipo"
DIR = Path(os.getenv("E2E_PROTO_DIR", DIR_DEFAULT))
FONTE_TESTO = Path("/usr/lib64/python3.14")          # rpm python3-libs-3.14.4-2.fc43
torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "2")))


# ───────────────────────────────────── configurazione ─────────────────────────────────────
@dataclass(frozen=True)
class Config:
    V: int = 256            # vocabolario a byte
    H: int = 1024           # nascosto: multiplo di 1024, e' il blocco di TQ1_0R_1024
    L: int = 2
    teste: int = 4
    d_testa: int = 64
    E: int = 8
    k: int = 2
    I: int = 256            # intermedio esperti: multiplo di 256, e' il blocco di TQ1_0
    I_cond: int = 256       # intermedio dell'esperto condiviso
    ctx: int = 128
    eps_norm: float = 1e-6


@dataclass(frozen=True)
class ConfigAllena:
    lr: float = 3e-3
    beta1: float = 0.9
    beta2: float = 0.95
    seq_per_passo: int = 16
    seq_per_micro: int = 8
    epoche: int = 3
    heldout_ogni: int = 8
    pezzo_kl: int = 256
    seme: int = 20260925


FAMIGLIE = {"gate": ("ffn_gate_exps", "TQ1_0R_1024"), "up": ("ffn_up_exps", "TQ1_0R_1024"),
            "down": ("ffn_down_exps", "TQ1_0")}
NON_ESPERTI = ("attn_norm", "attn_qkv", "attn_output", "post_attention_norm", "ffn_gate_inp",
               "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp", "ffn_gate_inp_shexp")


# ───────────────────────────────────────── dati ─────────────────────────────────────────
def _parte(nome: str) -> str:
    c = hashlib.sha256(nome.encode()).digest()[0] % 10
    return "maestro" if c < 6 else ("calibrazione" if c < 8 else "heldout")


def manifesto_dati() -> dict:
    """File .py della libreria standard, divisi in tre parti DISGIUNTE per nome (sha256)."""
    parti: dict = {"maestro": [], "calibrazione": [], "heldout": []}
    for p in sorted(FONTE_TESTO.glob("*.py")):
        b = p.read_bytes()
        parti[_parte(p.name)].append({"file": p.name, "byte": len(b), "sha256": hashlib.sha256(b).hexdigest()})
    return parti


def _flusso(voci: list) -> np.ndarray:
    return np.frombuffer(b"\x00".join((FONTE_TESTO / v["file"]).read_bytes() for v in voci), dtype=np.uint8)


def finestre(voci: list, n_seq: int, ctx: int) -> torch.Tensor:
    """n_seq finestre NON sovrapposte di ctx+1 byte, distribuite uniformemente sul flusso."""
    f = _flusso(voci)
    n_tot = f.size // (ctx + 1)
    if n_tot < n_seq:
        raise ValueError(f"testo insufficiente: {n_tot} finestre < {n_seq}")
    scelte = np.linspace(0, n_tot - 1, n_seq).round().astype(np.int64)
    return torch.from_numpy(np.stack([f[i * (ctx + 1):(i + 1) * (ctx + 1)] for i in scelte]).astype(np.int64))


# ─────────────────────────────── blocchi funzionali comuni ───────────────────────────────
def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def attenzione(x: torch.Tensor, P: dict, l: int, c: Config) -> torch.Tensor:
    B, T, _ = x.shape
    qkv = (x @ P[f"blk.{l}.attn_qkv"].T).view(B, T, 3, c.teste, c.d_testa).permute(2, 0, 3, 1, 4)
    o = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], is_causal=True)
    return o.transpose(1, 2).reshape(B, T, c.teste * c.d_testa) @ P[f"blk.{l}.attn_output"].T


def swiglu(x: torch.Tensor, Wg: torch.Tensor, Wu: torch.Tensor, Wd: torch.Tensor) -> torch.Tensor:
    return (F.silu(x @ Wg.T) * (x @ Wu.T)) @ Wd.T


def instrada(n: torch.Tensor, router: torch.Tensor, k: int, instr: Optional[torch.Tensor]):
    """Router come Qwen3.5-MoE: softmax su tutti gli esperti, top-k, rinormalizzazione. Con
    `instr` (indici salvati nella prima passata) il top-k NON si ricalcola: i pesi si prendono
    agli indici salvati (differenziabili). Ritorna (indici, pesi, cambi rispetto al top-k vivo,
    probabilita' del router)."""
    p = torch.softmax(n @ router.T, dim=-1)
    with torch.no_grad():
        vivo = torch.topk(p, k, dim=-1).indices
    idx = vivo if instr is None else instr
    cambi = 0 if instr is None else int((torch.sort(vivo, -1).values != torch.sort(instr, -1).values).any(-1).sum())
    w = p.gather(-1, idx)
    return idx, w / w.sum(-1, keepdim=True), cambi, p


def moe(n: torch.Tensor, idx: torch.Tensor, w: torch.Tensor, E: int, esperto) -> torch.Tensor:
    y = torch.zeros_like(n)
    for e in range(E):
        tok, pos = torch.where(idx == e)
        if tok.numel():
            y = y.index_add(0, tok, esperto(e, tok) * w[tok, pos].unsqueeze(-1))
    return y


def conteggi_esperti(idx: torch.Tensor, E: int) -> list:
    return torch.bincount(idx.reshape(-1), minlength=E).tolist()


# ──────────────────────────────────────── maestro ────────────────────────────────────────
def init_maestro(c: Config, seme: int) -> dict:
    g = torch.Generator().manual_seed(seme)
    r = lambda *s, std=0.02: (torch.randn(*s, generator=g) * std)  # noqa: E731
    P = {"token_embd": r(c.V, c.H), "output_norm": torch.ones(c.H), "output": r(c.V, c.H)}
    sd = 0.02 / math.sqrt(2 * c.L)
    for l in range(c.L):
        P |= {f"blk.{l}.attn_norm": torch.ones(c.H), f"blk.{l}.post_attention_norm": torch.ones(c.H),
              f"blk.{l}.attn_qkv": r(3 * c.teste * c.d_testa, c.H),
              f"blk.{l}.attn_output": r(c.H, c.teste * c.d_testa, std=sd),
              f"blk.{l}.ffn_gate_inp": r(c.E, c.H), f"blk.{l}.ffn_gate_exps": r(c.E, c.I, c.H),
              f"blk.{l}.ffn_up_exps": r(c.E, c.I, c.H), f"blk.{l}.ffn_down_exps": r(c.E, c.H, c.I, std=sd),
              f"blk.{l}.ffn_gate_shexp": r(c.I_cond, c.H), f"blk.{l}.ffn_up_shexp": r(c.I_cond, c.H),
              f"blk.{l}.ffn_down_shexp": r(c.H, c.I_cond, std=sd), f"blk.{l}.ffn_gate_inp_shexp": r(1, c.H)}
    return P


def strato_denso(x: torch.Tensor, P: dict, l: int, c: Config, instr=None):
    """Strato del maestro (esperti densi). Ritorna (uscita, indici, termine di bilanciamento)."""
    h = x + attenzione(rmsnorm(x, P[f"blk.{l}.attn_norm"], c.eps_norm), P, l, c)
    n = rmsnorm(h, P[f"blk.{l}.post_attention_norm"], c.eps_norm).reshape(-1, c.H)
    idx, w, _, p = instrada(n, P[f"blk.{l}.ffn_gate_inp"], c.k, instr)
    frazione = torch.bincount(idx.reshape(-1), minlength=c.E).to(p.dtype) / idx.numel()
    bilanciamento = c.E * (frazione * p.mean(0)).sum()
    Wg, Wu, Wd = P[f"blk.{l}.ffn_gate_exps"], P[f"blk.{l}.ffn_up_exps"], P[f"blk.{l}.ffn_down_exps"]
    y = moe(n, idx, w, c.E, lambda e, tok: swiglu(n[tok], Wg[e], Wu[e], Wd[e]))
    y = y + torch.sigmoid(n @ P[f"blk.{l}.ffn_gate_inp_shexp"].T) * swiglu(
        n, P[f"blk.{l}.ffn_gate_shexp"], P[f"blk.{l}.ffn_up_shexp"], P[f"blk.{l}.ffn_down_shexp"])
    return h + y.view_as(h), idx, bilanciamento


def nascosto_maestro(P: dict, ids: torch.Tensor, c: Config) -> torch.Tensor:
    """Stato nascosto finale DOPO la norma, in bf16: e' cio' che la testa del maestro riceve e
    cio' che si mette in cache (4 KiB/token in Thor, 2 KiB qui)."""
    x = P["token_embd"][ids]
    for l in range(c.L):
        x = strato_denso(x, P, l, c)[0]
    return rmsnorm(x, P["output_norm"], c.eps_norm).to(torch.bfloat16).reshape(-1, c.H)


def logit_maestro(hT: torch.Tensor, WT: torch.Tensor) -> torch.Tensor:
    """UNICA definizione dei logit del maestro: bf16 -> fp32, accumulo fp32. La usano sia il
    maestro diretto sia la perdita che parte dalla cache."""
    return hT.float() @ WT.float().T


def maestro_bf16(P32: dict) -> dict:
    return {k: v.detach().to(torch.bfloat16) for k, v in P32.items()}


def maestro_calcolo(Pbf: dict) -> dict:
    """Pesi bf16 portati a fp32 per il calcolo (esatto); il router resta fp32 come in Thor."""
    return {k: v.float() for k, v in Pbf.items()}


def addestra_maestro(c: Config, voci: list, passi: int = 600, seme: int = 7, log=print) -> tuple:
    """Pre-addestramento vero del maestro (fp32), poi congelato in bf16. Perdita: CE sul byte
    successivo + bilanciamento degli esperti (Switch, 0,01)."""
    torch.manual_seed(seme)
    P = {k: v.requires_grad_(True) for k, v in init_maestro(c, seme).items()}
    opt = torch.optim.AdamW(list(P.values()), lr=2e-3, betas=(0.9, 0.95), weight_decay=0.1)
    f = _flusso(voci)
    g = torch.Generator().manual_seed(seme + 1)
    B, storia = 16, []
    for t in range(passi):
        lr = 2e-3 * min(1.0, (t + 1) / 30) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t / passi)))
        for gr in opt.param_groups:
            gr["lr"] = lr
        o = torch.randint(0, f.size - c.ctx - 1, (B,), generator=g).numpy()
        ids = torch.from_numpy(np.stack([f[i:i + c.ctx + 1] for i in o]).astype(np.int64))
        x = P["token_embd"][ids[:, :-1]]
        aux = 0.0
        for l in range(c.L):
            x, _, bil = strato_denso(x, P, l, c)
            aux = aux + bil
        z = rmsnorm(x, P["output_norm"], c.eps_norm) @ P["output"].T
        ce = F.cross_entropy(z.reshape(-1, c.V), ids[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        (ce + 0.01 * aux).backward()
        torch.nn.utils.clip_grad_norm_(list(P.values()), 1.0)
        opt.step()
        storia.append(float(ce.detach()))
        if t % 50 == 0 or t == passi - 1:
            log(f"  maestro passo {t:4d}/{passi}  CE {float(ce.detach()):.4f}  lr {lr:.2e}")
    return maestro_bf16(P), storia


# ─────────────────────────────── ternarizzazione iniziale ───────────────────────────────
def ternarizza_ls(W: np.ndarray, gruppo: int) -> tuple:
    """Scala ai minimi quadrati per gruppo sulla griglia di casa (GRIGLIA_LS, METODO §4.1) e
    trit = clip(lroundf(w * (1/d))): STESSA aritmetica di `tq1_0r_1024_codec.scala_ls` +
    `quantize_rotated_blocks`, qui con gruppo qualsiasi (256 per il down, TQ1_0).
    W (..., n) float32 -> d (..., n/gruppo) float32, trit (..., n) int8."""
    lead, n = W.shape[:-1], W.shape[-1]
    G = np.ascontiguousarray(W, dtype=np.float32).reshape(*lead, n // gruppo, gruppo)
    ab = np.abs(G).astype(np.float64)
    media = ab.sum(-1) / float(gruppo)
    best_s, best_d = np.full(media.shape, -1.0), np.zeros(media.shape)
    for m in GRIGLIA_LS:
        vivi = ab > (m * media)[..., None]
        k = vivi.sum(-1).astype(np.float64)
        somma = np.where(vivi, ab, 0.0).sum(-1)
        s = np.where(k > 0, somma * somma / np.maximum(k, 1.0), 0.0)
        ok = s > best_s
        best_s = np.where(ok, s, best_s)
        best_d = np.where(ok, np.where(k > 0, somma / np.maximum(k, 1.0), 0.0), best_d)
    d = best_d.astype(np.float32)
    idd = np.where(d != 0, np.float32(1.0) / np.where(d == 0, np.float32(1), d), np.float32(0)).astype(np.float32)
    x = (G * idd[..., None]).astype(np.float32).astype(np.float64)
    trit = np.clip(np.sign(x) * np.floor(np.abs(x) + 0.5), -1, 1).astype(np.int8)
    return d, trit.reshape(*lead, n)


def blocchi_iniziali(Pbf: dict, c: Config) -> dict:
    """Codici e scale iniziali (LS per gruppo, nessuna Hessiana) nei byte VERI dei due formati.
    {(strato, famiglia): uint8 (E, righe, byte_per_riga)}."""
    B = {}
    for l in range(c.L):
        for fam in ("gate", "up"):
            W = Pbf[f"blk.{l}.{FAMIGLIE[fam][0]}"].float().numpy()               # (E, I, H)
            B[(l, fam)] = quantize_rows(W.reshape(c.E * c.I, c.H)).reshape(c.E, c.I, (c.H // 1024) * 224)
        W = Pbf[f"blk.{l}.ffn_down_exps"].float().numpy()                          # (E, H, I)
        d, trit = ternarizza_ls(W, 256)
        B[(l, "down")] = pack_tq1_0_da_d_q(d, trit).reshape(c.E, c.H, (c.I // 256) * 54)
    return B


def sha_codici(B: np.ndarray, tipo: str) -> str:
    """sha256 dei soli byte dei CODICI (qs+qh), campi d esclusi."""
    nb, _, off, k = CS.LAYOUT[tipo]
    b = np.ascontiguousarray(B).reshape(-1, nb)
    m = np.ones(nb, dtype=bool)
    m[off:off + 2 * k] = False
    return hashlib.sha256(np.ascontiguousarray(b[:, m]).tobytes()).hexdigest()


# ─────────────────────────────────── studente ternario ───────────────────────────────────
class ArrotondaFP16(torch.autograd.Function):
    """Avanti: il valore fp16 (round-to-nearest-even, come `.half()`, numpy `astype(float16)`
    e GGML_FP32_TO_FP16), riportato al dtype d'ingresso: e' ESATTAMENTE la scala che il GGUF
    conterra'. Indietro: identita' (straight-through, come EfficientQAT tiene s in FP16)."""

    @staticmethod
    def forward(ctx, s: torch.Tensor) -> torch.Tensor:
        r = s.to(torch.float16)
        if not torch.isfinite(r).all():
            raise FloatingPointError("scala fuori dal campo fp16 (|d| > 65504 o NaN)")
        return r.to(s.dtype)

    @staticmethod
    def backward(ctx, g: torch.Tensor):
        return g


class Studente:
    """Non-esperti = pesi bf16 del maestro (in Thor: quelli del GGUF). Esperti = byte dei
    blocchi (mai modificati) + theta (unici parametri). `dtype` e' il dtype di calcolo."""

    def __init__(self, Pbf: dict, blocchi: dict, c: Config, dtype=torch.float32):
        self.c, self.dtype = c, dtype
        self.P = {k: v.float().to(dtype) for k, v in Pbf.items() if not k.endswith("_exps")}
        self.blocchi = {chiave: torch.from_numpy(np.ascontiguousarray(b)) for chiave, b in blocchi.items()}
        self.d0, self.theta = {}, {}
        for (l, fam), b in self.blocchi.items():
            tipo = FAMIGLIE[fam][1]
            nb = CS.LAYOUT[tipo][0]
            d = CS.scale_da_blocchi(b.reshape(-1, nb), tipo).to(torch.float32)
            self.d0[(l, fam)] = d.reshape(b.shape[0], b.shape[1], -1)
            self.theta[(l, fam)] = torch.zeros_like(self.d0[(l, fam)], requires_grad=True)
        self.S = torch.from_numpy(S1024).to(dtype)
        self.cambi_instr = 0
        self.zeri_nuovi = 0

    def parametri(self) -> list:
        return [self.theta[k] for k in sorted(self.theta)]

    def scala(self, l: int, fam: str) -> torch.Tensor:
        d0 = self.d0[(l, fam)]
        r = ArrotondaFP16.apply(d0 * torch.exp(self.theta[(l, fam)]))
        with torch.no_grad():
            self.zeri_nuovi += int(((r == 0) & (d0 != 0)).sum())
        return r

    def scale_fp16(self) -> dict:
        """Le scale ESATTE che il forward usa adesso, in float16: quelle che si esportano."""
        with torch.no_grad():
            return {k: self.scala(*k).to(torch.float16).clone() for k in sorted(self.theta)}

    def trit(self, l: int, fam: str) -> torch.Tensor:
        """Trit decodificati dai byte a OGNI chiamata (int8, (E, righe, colonne))."""
        b = self.blocchi[(l, fam)]
        tipo = FAMIGLIE[fam][1]
        nb = CS.LAYOUT[tipo][0]
        return CS.trit_da_blocchi(b.reshape(-1, nb), tipo).reshape(b.shape[0], b.shape[1], -1)

    def pesi(self, l: int, fam: str) -> torch.Tensor:
        """w = d * trit per gruppo, TUTTI gli esperti: per gate/up e' il peso RUOTATO (w_rot), per
        down il peso. Solo per le prove (T2): lo strato usa `peso_esperto` (vedi sotto)."""
        q = self.trit(l, fam).to(self.dtype)
        d = self.scala(l, fam).to(self.dtype)
        E, R, C = q.shape
        return (q.view(E, R, d.shape[-1], -1) * d.unsqueeze(-1)).view(E, R, C)

    def peso_esperto(self, l: int, fam: str, e: int, d: torch.Tensor) -> torch.Tensor:
        """w = d[e] * trit[e] di UN esperto, trit decodificati dai soli byte di quell'esperto.
        Revisione 25/9: materializzare W intero e poi indicizzare W[e] nel ciclo fa generare ad
        autograd, per OGNI esperto, un gradiente grande quanto tutto W (SelectBackward) da sommare:
        indietro O(E^2 R C), x85-97 a E = 256 (e2e_revisione_prove.py, R1), gradienti identici
        al bit. `d` sono le scale fp16 dello strato (una sola chiamata a `scala` per famiglia)."""
        b = self.blocchi[(l, fam)][e]
        tipo = FAMIGLIE[fam][1]
        q = CS.trit_da_blocchi(b.reshape(-1, CS.LAYOUT[tipo][0]), tipo).reshape(b.shape[0], -1).to(self.dtype)
        de = d[e].to(self.dtype)
        R, C = q.shape
        return (q.view(R, de.shape[-1], -1) * de.unsqueeze(-1)).view(R, C)

    def ruota(self, x: torch.Tensor) -> torch.Tensor:
        """R x = FWHT(S * x) a blocchi di 1024 (ricetta B: si ruota l'attivazione)."""
        xb = x.reshape(*x.shape[:-1], x.shape[-1] // 1024, 1024)
        return fwht1024_torch(xb * self.S).reshape(x.shape)

    def strato(self, x: torch.Tensor, l: int, instr=None):
        c, P = self.c, self.P
        h = x + attenzione(rmsnorm(x, P[f"blk.{l}.attn_norm"], c.eps_norm), P, l, c)
        n = rmsnorm(h, P[f"blk.{l}.post_attention_norm"], c.eps_norm).reshape(-1, c.H)
        idx, w, cambi, _ = instrada(n, P[f"blk.{l}.ffn_gate_inp"], c.k, instr)
        self.cambi_instr += cambi
        nr = self.ruota(n)
        dg, du, dd = self.scala(l, "gate"), self.scala(l, "up"), self.scala(l, "down")
        pe = self.peso_esperto
        y = moe(n, idx, w, c.E, lambda e, tok: (F.silu(nr[tok] @ pe(l, "gate", e, dg).T)
                                                * (nr[tok] @ pe(l, "up", e, du).T)) @ pe(l, "down", e, dd).T)
        y = y + torch.sigmoid(n @ P[f"blk.{l}.ffn_gate_inp_shexp"].T) * swiglu(
            n, P[f"blk.{l}.ffn_gate_shexp"], P[f"blk.{l}.ffn_up_shexp"], P[f"blk.{l}.ffn_down_shexp"])
        return h + y.view_as(h), idx

    def entra(self, ids: torch.Tensor) -> torch.Tensor:
        return self.P["token_embd"][ids]

    def esce(self, x: torch.Tensor) -> torch.Tensor:
        return rmsnorm(x, self.P["output_norm"], self.c.eps_norm).reshape(-1, self.c.H)

    def nascosto(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.entra(ids)
        for l in range(self.c.L):
            x = self.strato(x, l)[0]
        return self.esce(x)

    def logit(self, ids: torch.Tensor) -> torch.Tensor:
        return self.nascosto(ids) @ self.P["output"].T


# ────────────────────────────────── perdita KL a pezzi ──────────────────────────────────
class KLAPezzi(torch.autograd.Function):
    """sum_t KL(p_T || p_S) sul vocabolario intero, a pezzi di token: z_S = hS WS^T,
    z_T = logit_maestro(hT, WT). Indietro in forma chiusa: dL/dhS = (p_S - p_T) WS.
    Niente grafo sul vocabolario: in Thor 8192 x 248.320 logit non si materializzano mai."""

    @staticmethod
    def forward(ctx, hS, WS, hT, WT, pezzo: int):
        tot = torch.zeros((), dtype=torch.float64)
        for i in range(0, hS.shape[0], pezzo):
            lS = torch.log_softmax(hS[i:i + pezzo] @ WS.T, -1)
            lT = torch.log_softmax(_zT(hT[i:i + pezzo], WT, hS.dtype), -1)
            tot += (lT.exp() * (lT - lS)).sum().double()
        ctx.save_for_backward(hS, WS, hT, WT)
        ctx.pezzo = pezzo
        return tot.to(hS.dtype)

    @staticmethod
    def backward(ctx, g):
        hS, WS, hT, WT = ctx.saved_tensors
        dh = torch.empty_like(hS)
        for i in range(0, hS.shape[0], ctx.pezzo):
            pS = torch.softmax(hS[i:i + ctx.pezzo] @ WS.T, -1)
            pT = torch.softmax(_zT(hT[i:i + ctx.pezzo], WT, hS.dtype), -1)
            dh[i:i + ctx.pezzo] = ((pS - pT) * g) @ WS
        return dh, None, None, None, None


def _zT(hT: torch.Tensor, WT: torch.Tensor, dtype) -> torch.Tensor:
    return logit_maestro(hT, WT) if dtype == torch.float32 else hT.to(dtype) @ WT.to(dtype).T


def kl_diretta(hS, WS, hT, WT) -> torch.Tensor:
    """Riferimento non a pezzi, con autograd (solo per i test)."""
    lS = torch.log_softmax(hS @ WS.T, -1)
    lT = torch.log_softmax(_zT(hT, WT, hS.dtype), -1)
    return (lT.exp() * (lT - lS)).sum()


# ─────────────────────────────── passo con checkpoint per strato ───────────────────────────────
def gradiente_micro(st: Studente, ids: torch.Tensor, hT: torch.Tensor, WT: torch.Tensor,
                    pezzo: int, scala_perdita: float) -> float:
    """Avanti senza grafo (si salvano l'ingresso di ogni strato e il top-k), poi ricalcolo
    strato per strato con l'instradamento SALVATO e indietro. Accumula in theta.grad il
    gradiente di scala_perdita * sum KL. Ritorna la somma KL del micro-lotto."""
    c = st.c
    with torch.no_grad():
        x = st.entra(ids)
        ingressi, instr = [], []
        for l in range(c.L):
            ingressi.append(x)
            x, idx = st.strato(x, l)
            instr.append(idx)
    xL = x.detach().requires_grad_(True)
    perdita = KLAPezzi.apply(st.esce(xL), st.P["output"], hT, WT, pezzo)
    (perdita * scala_perdita).backward()
    g = xL.grad
    for l in reversed(range(c.L)):
        xin = ingressi[l].detach().requires_grad_(True)
        y, _ = st.strato(xin, l, instr=instr[l])
        y.backward(g)
        g = xin.grad
    return float(perdita.detach())


def gradiente_grafo_intero(st: Studente, ids, hT, WT, scala_perdita: float) -> float:
    """Riferimento: grafo intero, niente checkpoint, perdita non a pezzi (solo per i test)."""
    perdita = kl_diretta(st.nascosto(ids), st.P["output"], hT, WT)
    (perdita * scala_perdita).backward()
    return float(perdita.detach())


@torch.no_grad()
def valuta(st: Studente, ids: torch.Tensor, hT: torch.Tensor, WT: torch.Tensor, lotto: int = 16) -> dict:
    """KL(maestro||studente) media per token e CE sul byte vero (sola lettura), su dati tenuti fuori."""
    kl = ce_s = ce_t = 0.0
    n = 0
    T = st.c.ctx
    for i in range(0, ids.shape[0], lotto):
        x = ids[i:i + lotto, :-1]
        y = ids[i:i + lotto, 1:].reshape(-1)
        zS = st.logit(x).float()
        zT = logit_maestro(hT[i * T:(i + lotto) * T], WT)
        lS, lT = torch.log_softmax(zS, -1), torch.log_softmax(zT, -1)
        kl += float((lT.exp() * (lT - lS)).sum().double())
        ce_s += float(F.cross_entropy(zS, y, reduction="sum").double())
        ce_t += float(F.cross_entropy(zT, y, reduction="sum").double())
        n += y.numel()
    return {"kl": kl / n, "ce_studente": ce_s / n, "ce_maestro": ce_t / n, "token": n}


# ─────────────────────────────────────── allenamento ───────────────────────────────────────
def statistiche_gradiente(st: Studente, eps: float) -> dict:
    """|dL/dtheta| per famiglia, sulle scale VIVE (d0 != 0) con gradiente non nullo. Un gradiente
    esattamente zero su una scala viva vuol dire gruppo con trit tutti nulli (la scala non conta)
    o esperto mai instradato nel lotto: si conta a parte."""
    out = {}
    for fam in ("gate", "up", "down"):
        g = torch.cat([st.theta[(l, fam)].grad.reshape(-1) for l in range(st.c.L)]).abs().double()
        vive = torch.cat([st.d0[(l, fam)].reshape(-1) for l in range(st.c.L)]) != 0
        gv = g[vive & (g > 0)]
        out[fam] = {"mediana": float(gv.median()), "q01": float(torch.quantile(gv, 0.01)),
                    "p99": float(torch.quantile(gv, 0.99)), "frazione_sotto_10eps": float((gv < 10 * eps).double().mean()),
                    "vive": int(vive.sum()), "vive_con_gradiente_zero": int((vive & (g == 0)).sum())}
    return out


def ordine(n: int, seme: int, epoca: int) -> torch.Tensor:
    return torch.randperm(n, generator=torch.Generator().manual_seed(seme * 1000 + epoca))


def nuovo_ottimizzatore(st: Studente, ca: ConfigAllena, eps: float) -> torch.optim.Optimizer:
    opt = torch.optim.AdamW(st.parametri(), lr=ca.lr, betas=(ca.beta1, ca.beta2), eps=eps, weight_decay=0.0)
    assert all(g["weight_decay"] == 0.0 for g in opt.param_groups)
    return opt


def cancello_g1(st: Studente, ids, hT, WT, ca: ConfigAllena) -> tuple:
    """Gradiente al passo 0 sul primo lotto (cancello G1, lezione del 24/9: «grad < eps»).
    eps = min(1e-8, 1e-3 * q01): anche la scala viva al 1° percentile di |g| fa un primo passo
    di Adam pari a lr/(1 + 1e-3). La prova che conta e' a valle: T7 misura il passo vero."""
    for p in st.parametri():
        p.grad = None
    n_tok = ids.shape[0] * st.c.ctx
    for j in range(0, ids.shape[0], ca.seq_per_micro):
        sl = slice(j * st.c.ctx, (j + ca.seq_per_micro) * st.c.ctx)
        gradiente_micro(st, ids[j:j + ca.seq_per_micro, :-1], hT[sl], WT, ca.pezzo_kl, 1.0 / n_tok)
    q01 = min(v["q01"] for v in statistiche_gradiente(st, 1e-8).values())
    eps = min(1e-8, q01 * 1e-3)
    stat = statistiche_gradiente(st, eps)
    for p in st.parametri():
        p.grad = None
    return eps, stat


def allena(st: Studente, cal: torch.Tensor, cal_hT: torch.Tensor, held: torch.Tensor, held_hT: torch.Tensor,
           WT: torch.Tensor, ca: ConfigAllena, eps: float, ripresa: Optional[dict] = None,
           ferma_a: Optional[int] = None, log=print) -> dict:
    """Ciclo: accumulo su micro-lotti, AdamW (wd 0), lr costante, held-out ogni `heldout_ogni`
    passi, si tiene il migliore (theta + scale fp16 esatte). I codici si controllano a ogni passo."""
    c = st.c
    opt = nuovo_ottimizzatore(st, ca, eps)
    passi_epoca = cal.shape[0] // ca.seq_per_passo
    tot = passi_epoca * ca.epoche
    stato = {"passo": 0, "migliore": None, "storia": [], "cambi_instr": 0}
    cambi0 = st.cambi_instr
    sha0 = {k: hashlib.sha256(b.numpy().tobytes()).hexdigest() for k, b in st.blocchi.items()}
    if ripresa is not None:
        with torch.no_grad():
            for k, v in ripresa["theta"].items():
                st.theta[k].copy_(v)
        opt.load_state_dict(ripresa["opt"])
        stato = ripresa["stato"]
    if stato["passo"] == 0:
        v0 = valuta(st, held, held_hT, WT)
        stato["storia"].append({"passo": 0, **v0})
        stato["migliore"] = {"passo": 0, "kl": v0["kl"], "theta": _copia_theta(st), "scale": st.scale_fp16()}
        log(f"  passo    0  held-out KL {v0['kl']:.5f}  CE {v0['ce_studente']:.4f} (maestro {v0['ce_maestro']:.4f})")
    while stato["passo"] < (tot if ferma_a is None else min(tot, ferma_a)):
        t = stato["passo"]
        ep, pos = divmod(t, passi_epoca)
        scelta = ordine(cal.shape[0], ca.seme, ep)[pos * ca.seq_per_passo:(pos + 1) * ca.seq_per_passo]
        opt.zero_grad(set_to_none=True)
        n_tok, kl_passo = len(scelta) * c.ctx, 0.0
        for j in range(0, len(scelta), ca.seq_per_micro):
            s = scelta[j:j + ca.seq_per_micro]
            hT = cal_hT.view(cal.shape[0], c.ctx, c.H)[s].reshape(-1, c.H)
            kl_passo += gradiente_micro(st, cal[s, :-1], hT, WT, ca.pezzo_kl, 1.0 / n_tok)
        opt.step()
        stato["passo"] = t + 1
        _controlla_codici(st, sha0)
        voce = {"passo": t + 1, "kl_train": kl_passo / n_tok}
        if (t + 1) % ca.heldout_ogni == 0 or t + 1 == tot:
            v = valuta(st, held, held_hT, WT)
            voce |= v
            if v["kl"] < stato["migliore"]["kl"]:
                stato["migliore"] = {"passo": t + 1, "kl": v["kl"], "theta": _copia_theta(st), "scale": st.scale_fp16()}
            log(f"  passo {t + 1:4d}  KL train {voce['kl_train']:.5f}  held-out KL {v['kl']:.5f}  "
                f"CE {v['ce_studente']:.4f}  |theta| mediana {_mediana_theta(st):.2e}")
        stato["storia"].append(voce)
    stato["cambi_instr"] += st.cambi_instr - cambi0
    return {"stato": stato, "opt": opt.state_dict(), "theta": _copia_theta(st), "sha_blocchi": sha0}


def _copia_theta(st: Studente) -> dict:
    return {k: v.detach().clone() for k, v in st.theta.items()}


def _mediana_theta(st: Studente) -> float:
    return float(torch.cat([v.detach().abs().reshape(-1) for v in st.theta.values()]).median())


def _controlla_codici(st: Studente, sha0: dict) -> None:
    for k, b in st.blocchi.items():
        if hashlib.sha256(b.numpy().tobytes()).hexdigest() != sha0[k]:
            raise RuntimeError(f"i byte dei blocchi {k} sono cambiati durante l'allenamento")
        if b.requires_grad:
            raise RuntimeError(f"i byte dei blocchi {k} hanno un gradiente")


def salva_ripresa(percorso: Path, esito: dict) -> None:
    torch.save({"theta": esito["theta"], "opt": esito["opt"], "stato": esito["stato"]}, percorso)


def carica_ripresa(percorso: Path) -> dict:
    return torch.load(percorso, weights_only=False)


# ─────────────────────────────────────────── GGUF ───────────────────────────────────────────
def _bf16_byte(t: torch.Tensor) -> np.ndarray:
    u = t.to(torch.bfloat16).contiguous().view(torch.int16).numpy().view(np.uint8)
    return u.reshape(t.shape[0], -1) if t.dim() == 2 else u


def scrivi_gguf(percorso: Path, Pbf: dict, blocchi: dict, c: Config) -> None:
    """GGUF giocattolo: non-esperti BF16 (norme e router F32, come in Thor), esperti nei byte
    dei loro blocchi (TQ1_0R_1024 / TQ1_0, forma GGUF [in, out, E])."""
    if percorso.exists():
        raise FileExistsError(percorso)
    w = GGUFWriter(str(percorso), arch="e2e_giocattolo")
    for k, v in asdict(c).items():
        (w.add_float32 if isinstance(v, float) else w.add_uint32)(f"e2e_giocattolo.{k}", v)
    for nome, t in Pbf.items():
        if nome.endswith("_exps"):
            continue
        if nome.endswith("norm") or nome.endswith("ffn_gate_inp") or nome.endswith("ffn_gate_inp_shexp"):
            w.add_tensor(nome + ".weight", t.float().numpy())
        else:
            w.add_tensor(nome + ".weight", _bf16_byte(t), raw_dtype=GT.BF16)
    for (l, fam), b in sorted(blocchi.items()):
        nome, tipo = FAMIGLIE[fam]
        w.add_tensor(f"blk.{l}.{nome}.weight", np.ascontiguousarray(b), raw_dtype=GT[tipo])
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def leggi_gguf(percorso: Path, c: Config) -> tuple:
    """-> (Pbf non-esperti in bf16/f32 esatti, blocchi {(l,fam): uint8 (E, righe, byte)})."""
    r = GGUFReader(str(percorso))
    Pbf, blocchi = {}, {}
    for t in r.tensors:
        nome = t.name.removesuffix(".weight")
        dati = np.array(t.data)
        if t.tensor_type.name == "BF16":
            Pbf[nome] = torch.from_numpy(dati.view(np.int16).copy()).view(torch.bfloat16)
        elif t.tensor_type.name == "F32":
            Pbf[nome] = torch.from_numpy(dati.copy()).reshape([int(x) for x in reversed(t.shape)])
        else:
            l = int(nome.split(".")[1])
            fam = next(f for f, (n, _) in FAMIGLIE.items() if nome.endswith(n))
            blocchi[(l, fam)] = dati.reshape(c.E, -1, dati.shape[-1]).copy()
    return Pbf, blocchi


def scale_per_tensore(scale: dict) -> dict:
    """{(l,fam): fp16 (E, righe, ng)} -> {nome GGUF: array (n_blocchi, k)} nell'ordine dei blocchi."""
    out = {}
    for (l, fam), s in scale.items():
        nome, tipo = FAMIGLIE[fam]
        k = CS.LAYOUT[tipo][3]
        out[f"blk.{l}.{nome}.weight"] = s.numpy().astype(np.float16).reshape(-1, k)
    return out


def esporta(src: Path, dst: Path, scale: dict) -> dict:
    """Copia del GGUF sorgente + riscrittura dei SOLI campi d + verifica byte per byte."""
    if dst.exists():
        raise FileExistsError(dst)
    sha_src = hashlib.sha256(src.read_bytes()).hexdigest()
    shutil.copyfile(src, dst)
    nuove = scale_per_tensore(scale)
    cambiate = CS.patcha_scale(dst, nuove)
    rep = CS.verifica_patch(src, dst, list(nuove))
    if hashlib.sha256(src.read_bytes()).hexdigest() != sha_src:
        raise RuntimeError("il GGUF sorgente e' cambiato")
    return {"scale_cambiate": cambiate, **rep, "sha256_sorgente": sha_src,
            "sha256_esportato": hashlib.sha256(dst.read_bytes()).hexdigest()}


# ───────────────────────────────────────── pipeline ─────────────────────────────────────────
def prepara(c: Config, log=print) -> dict:
    """Maestro (dalla cache se c'e', altrimenti addestrato), dati, cache del maestro."""
    DIR.mkdir(parents=True, exist_ok=True)
    man = manifesto_dati()
    firma = hashlib.sha256(json.dumps({"dati": man["maestro"], "cfg": asdict(c)}, sort_keys=True).encode()).hexdigest()
    cache = DIR / "maestro_bf16.pt"
    if cache.exists() and torch.load(cache, weights_only=False)["firma"] == firma:
        m = torch.load(cache, weights_only=False)
        log(f"maestro dalla cache {cache} (CE finale di addestramento {m['storia'][-1]:.4f})")
    else:
        log("addestro il maestro (una volta sola, poi in cache)")
        t0 = time.time()
        Pbf, storia = addestra_maestro(c, man["maestro"], log=log)
        m = {"firma": firma, "P": Pbf, "storia": storia, "secondi": time.time() - t0}
        torch.save(m, cache)
    cal = finestre(man["calibrazione"], 512, c.ctx)
    held = finestre(man["heldout"], 128, c.ctx)
    Pc = maestro_calcolo(m["P"])
    with torch.no_grad():
        cal_hT = torch.cat([nascosto_maestro(Pc, cal[i:i + 16, :-1], c) for i in range(0, cal.shape[0], 16)])
        held_hT = torch.cat([nascosto_maestro(Pc, held[i:i + 16, :-1], c) for i in range(0, held.shape[0], 16)])
    return {"man": man, "Pbf": m["P"], "cal": cal, "held": held, "cal_hT": cal_hT, "held_hT": held_hT,
            "WT": m["P"]["output"], "maestro_storia": m["storia"]}


def main() -> int:
    c, ca = Config(), ConfigAllena()
    D = prepara(c)
    blocchi = blocchi_iniziali(D["Pbf"], c)
    st = Studente(D["Pbf"], blocchi, c)
    eps, g1 = cancello_g1(st, D["cal"][:ca.seq_per_passo], D["cal_hT"][:ca.seq_per_passo * c.ctx], D["WT"], ca)
    print(f"G1: eps = {eps:.3e}  {json.dumps(g1)}")
    esito = allena(st, D["cal"], D["cal_hT"], D["held"], D["held_hT"], D["WT"], ca, eps)
    src, dst = DIR / "studente_iniziale.gguf", DIR / "studente_e2e.gguf"
    for p in (src, dst):
        if p.exists():
            p.unlink()
    scrivi_gguf(src, D["Pbf"], blocchi, c)
    rep = esporta(src, dst, esito["stato"]["migliore"]["scale"])
    print(json.dumps({"migliore": {k: v for k, v in esito["stato"]["migliore"].items() if k in ("passo", "kl")},
                      "export": rep}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
