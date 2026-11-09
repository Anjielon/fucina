#!/usr/bin/env python3
"""e2e_lora — «LoRA sui non-esperti» di Thor-T (27/9/2026, ramo lora-nonesperti). File NUOVO.

## Cosa fa
Oggi i non-esperti di Thor-T (proiezioni dell'attenzione piena q/k/v/o, GDN in_proj_qkv / in_proj_z (attn_gate) /
out_proj (ssm_out), esperto condiviso gate/up/down) sono k-quant del GGUF (Q8_0, Q4_K, Q6_K; Q4_0 solo nello strato
MTP 40, che lo studente non carica) dequantizzati in fp32 e FERMI. Qui ognuno puo' ricevere un adattatore di rango r:

    W_eff = W_deq + s * B A,      s = alpha / r,   A (r, in), B (out, r)
    y = x W_deq^T + s * (x A^T) B^T                      (forma NON fusa nell'avanti: niente dW pieno nell'indietro)
    dL/dB = s * g^T (x A^T),   dL/dA = s * (g B)^T x      (g = dL/dy; costo O(T r (in + out)), non O(T in out))

A ~ U(-1/sqrt(in), 1/sqrt(in)) (PEFT/kaiming a=sqrt(5)), seme per tensore (sha256 del nome + seme della corsa, sulla
CPU: stesso A su CPU e GPU), B = 0: al passo 0 il ramo vale ESATTAMENTE zero e l'avanti e' quello di sempre (prova L1).
A e B sono foglie fp32 (fp64 nelle ricette di prova a 64 bit); il peso del modulo HF resta senza gradiente. Il ramo
sta DENTRO lo strato: avanti senza grafo e ricalcolo dell'indietro calcolano lo stesso valore (instradamento salvato
e contatore dei cambi invariati).

Lo spazio dell'adattatore e' quello del modulo HF (righe/colonne permutate come `carica_strato_gguf._applica`);
l'export riporta Delta W nell'ordine del FILE con l'inversa esatta della relazione (identita', perm_righe:
W_file = W_hf[perm], perm_colonne: W_file = W_hf[:, perm]) e riquantizza nello STESSO tipo (taglia invariata):

  * modo "griglia" (default dell'export): scale e minimi del blocco FERMI, si ri-arrotondano i soli codici,
    q' = clip(rint((W + Delta W - off) / passo)). Con Delta W = 0 restituisce i byte del file AL BIT (prova L5).
  * modo "ggml": il quantizzatore di llama.cpp da capo (gguf-py per Q8_0/Q4_0, che e' al bit con ggml;
    `ggml_quantize_chunk` di libggml-base via ctypes per Q4_K/Q6_K, che gguf-py 0.19 NON sa quantizzare), senza
    imatrix: sceglie scale nuove, quindi sposta anche i pesi con Delta W = 0 (misurato: vedi `sopravvivenza`).

## Il rischio vero (misura `sopravvivenza`)
Il codice cambia solo se |Delta W| supera mezzo passo della griglia: un Delta W piu' piccolo del passo di
quantizzazione SPARISCE nella fusione. `python3 e2e_lora.py sopravvivenza` misura su matrici vere di Thor-T (CPU):
quota di Delta W che resta dopo la riquantizzazione (proiezione e rumore), per tipo, in funzione di
rho = rms(Delta W) / rms(passo), con Delta W = direzione del miglior correttore di rango r (SVD troncata di
W_bf16 - W_deq, il bf16 ORIGINALE di Ornith) e con quel correttore alla sua norma naturale.

MISURATO (27/9, TECNICHE_E2E_PV.md §25): il correttore del proprio errore ha rho 0,04-0,10 e la fusione ne tiene 2-11%
(guadagno verso il bf16 negativo); con dW gaussiano rho 0,05 → 0%, 0,25 → ~46%, 0,5 → 94-97%. rms(passo)/rms(W): Q8_0 0,02,
Q6_K 0,07, Q4_K 0,30: con lr 1e-4 sopravvivono i Q8_0 (`ConfigLoRA.tipi` / `--lora-tipi Q8_0`), non i Q4_K.

## 27/9 LoRA-GA (ramo lora-ga; arXiv 2407.05000 §3.4 + codice ufficiale Outsider565/LoRA-GA, TECNICHE §37)
`inizializza_ga(st, calcola, gamma)`: durante `calcola()` (un avanti+indietro sul lotto di calibrazione) ogni
`LinearLoRA` accumula il gradiente PIENO del suo peso, dW = sum_t g_t^T x_t (gancio sull'uscita, fp32/fp64, nello
spazio HF); poi dW = U S V^T (primi 2r vettori singolari) e

    A0 = V[:, :r]^T * c,   B0 = U[:, r:2r] * c,   c = out^(1/4) / sqrt(gamma)     (direzione ArB2r, scala «stable»)

cosi' il primo passo di SGD su (A, B) muove s B A come -lr * zeta * (troncata di rango 2r di dW), zeta = s^2 c^2
(Teorema 3.1). La correzione del paper W_frozen = W0 - s B0 A0 e' scritta NON fusa:

    y = x W0^T + s [(x (A - A0)^T) B^T + (x A0^T) (B - B0)^T]    (= x (W0 - s B0 A0)^T + s x (B A)^T in aritmetica esatta)

al passo 0 (A = A0, B = B0 bit per bit) entrambi i termini sono ZERO ESATTO: avanti, dL/dx e gradiente di theta
identici al bit alla LoRA spenta (la forma fusa del paper cambierebbe l'arrotondamento di W). All'export
Delta W = s (B A - B0 A0) (fp64, zero esatto al passo 0). A0, B0 nella ripresa e in lora.npz ("|A0", "|B0").

Uso (CPU, nice):
    nice -n 15 env OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES= @@ROOT:venv@@/bin/python3 e2e_lora.py \\
        sopravvivenza [--rango 16] [--tensori blk.0.attn_qkv.weight,...] [--uscita FILE.json]
Prove: e2e_test_lora.py (L1-L5), e2e_test_lora_ga.py (G1-G6).
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import math
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
sys.path.insert(0, "@@ROOT:models@@/build/llamacpp-tq1-k1024/gguf-py")   # gguf-py con TQ1_0R (tipo 45), come gli altri

VERSIONE = "e2e_lora 1 (27/9/2026)"
LIB_GGML = os.getenv("E2E_LIB_GGML", "@@ROOT:models@@/build/llamacpp-tq1-k1024/build/bin/libggml-base.so")
GGUF_BF16 = [Path("@@ROOT:models@@/gguf/thor-src/Ornith-1.5-35B-A3B-bf16") / f"Ornith-1.5-35B-A3B-bf16-0000{i}-of-00002.gguf"
             for i in (1, 2)]

# famiglia -> nome del tensore GGUF senza "blk.N." (ordine canonico: ordine dei parametri nel gruppo di Adam)
FAMIGLIE_LORA = {
    "attn_q": "attn_q.weight", "attn_k": "attn_k.weight", "attn_v": "attn_v.weight", "attn_output": "attn_output.weight",
    "attn_qkv": "attn_qkv.weight", "attn_gate": "attn_gate.weight", "ssm_out": "ssm_out.weight",
    "ffn_gate_shexp": "ffn_gate_shexp.weight", "ffn_up_shexp": "ffn_up_shexp.weight",
    "ffn_down_shexp": "ffn_down_shexp.weight",
}
GRUPPI_LORA = {
    "attn": ("attn_q", "attn_k", "attn_v", "attn_output"),            # attenzione piena (10 strati)
    "gdn": ("attn_qkv", "attn_gate", "ssm_out"),                        # Gated DeltaNet (30 strati)
    "shexp": ("ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp"),      # esperto condiviso (40 strati)
    "tutte": tuple(FAMIGLIE_LORA),
}
# tipo GGUF -> (valori per blocco, byte per blocco); id ggml (ggml.h: Q4_0 2, Q8_0 8, Q4_K 12, Q6_K 14)
BLOCCO = {"Q8_0": (32, 34), "Q4_0": (32, 18), "Q4_K": (256, 144), "Q6_K": (256, 210)}
ID_GGML = {"Q4_0": 2, "Q8_0": 8, "Q4_K": 12, "Q6_K": 14}
TIPI = tuple(BLOCCO)
MODI = ("griglia", "ggml")


def famiglie_lora(voci) -> tuple:
    """"a,b" o iterabile (famiglie o gruppi: attn, gdn, shexp, tutte) -> tupla canonica (ordine di FAMIGLIE_LORA).
    Vuota = nessuna."""
    if isinstance(voci, str):
        voci = voci.split(",")
    chieste = set()
    for v in voci:
        v = v.strip()
        if not v:
            continue
        if v.endswith(".weight"):
            v = v[:-len(".weight")]
        if v in GRUPPI_LORA:
            chieste.update(GRUPPI_LORA[v])
        elif v in FAMIGLIE_LORA:
            chieste.add(v)
        else:
            raise ValueError(f"famiglia LoRA sconosciuta {v!r}: {sorted(FAMIGLIE_LORA) + sorted(GRUPPI_LORA)}")
    return tuple(f for f in FAMIGLIE_LORA if f in chieste)


def tipi_lora(voci) -> tuple:
    """"Q8_0,Q6_K" o iterabile -> tupla canonica (ordine di TIPI). Vuota = tutti i tipi supportati."""
    if isinstance(voci, str):
        voci = voci.split(",")
    chiesti = {v.strip().upper() for v in voci if v.strip()}
    ignoti = chiesti - set(TIPI)
    if ignoti:
        raise ValueError(f"tipi LoRA sconosciuti {sorted(ignoti)}: {TIPI}")
    return tuple(t for t in TIPI if t in chiesti)


@dataclass(frozen=True)
class ConfigLoRA:
    rango: int = 0                        # 0 = spenta (studente identico al bit)
    alpha: Optional[float] = None         # None = rango (s = 1)
    famiglie: tuple = ()
    seme: int = 20260926
    tipi: tuple = ()                      # solo i tensori di questi tipi GGUF (vuota = tutti: Q8_0, Q4_0, Q4_K, Q6_K)

    def __post_init__(self):
        if self.rango < 0:
            raise ValueError(f"rango {self.rango} < 0")
        if self.rango > 0 and not self.famiglie:
            raise ValueError("LoRA accesa senza famiglie")
        if self.alpha is not None and not self.alpha > 0:
            raise ValueError(f"alpha {self.alpha} <= 0")
        if any(t not in TIPI for t in self.tipi):
            raise ValueError(f"tipi {self.tipi}: ammessi {TIPI}")

    @property
    def scala(self) -> float:
        return float(self.alpha if self.alpha is not None else self.rango) / self.rango

    def firma(self) -> dict:
        return {"rango": self.rango, "alpha": self.alpha, "scala": self.scala, "famiglie": list(self.famiglie),
                "seme": self.seme, "tipi_ammessi": list(self.tipi)}


def seme_tensore(nome: str, seme: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{seme}:{nome}".encode()).digest()[:8], "little") & ((1 << 63) - 1)


# ═══════════════════════════════════ relazione HF <-> file ═══════════════════════════════════
def controlla_voce(voce: dict, nome: str) -> None:
    if voce["rel"] not in ("identita", "perm_righe", "perm_colonne") or voce.get("trasposta") or voce.get("fetta") \
            or voce.get("f"):
        raise ValueError(f"{nome}: relazione {voce} non ammessa per la LoRA (attese identita'/perm_righe/perm_colonne)")


def hf_a_file(m, voce: dict):
    """Inversa esatta di `carica_strato_gguf._applica` per le relazioni delle matrici LoRA (numpy o torch)."""
    rel = voce["rel"]
    if rel == "identita":
        return m
    import carica_strato_gguf as CSG  # sola lettura
    perm = CSG.PERM[voce["perm_npz"]]
    if isinstance(m, torch.Tensor):
        perm = torch.from_numpy(np.asarray(perm, dtype=np.int64)).to(m.device)
    return m[perm] if rel == "perm_righe" else m[:, perm]


# ═══════════════════════════════════ adattatore e modulo ═══════════════════════════════════
class AdattatoreLoRA:
    """UN tensore GGUF non-esperto con il suo adattatore. `A`, `B` foglie con gradiente (cio' che Adam aggiorna)."""

    def __init__(self, nome: str, famiglia: str, L: int, voce: dict, fuori: int, dentro: int, cfg: ConfigLoRA,
                 device, dtype_master: torch.dtype, tipo: str):
        controlla_voce(voce, nome)
        self.nome, self.famiglia, self.L, self.voce, self.tipo = nome, famiglia, L, voce, tipo
        self.rango, self.scala = cfg.rango, cfg.scala
        self.forma = (fuori, dentro)
        g = torch.Generator().manual_seed(seme_tensore(nome, cfg.seme))
        lim = 1.0 / math.sqrt(dentro)
        A = (torch.rand(cfg.rango, dentro, generator=g, dtype=torch.float64) * 2.0 - 1.0) * lim
        self.A = A.to(dtype=dtype_master).to(device).requires_grad_(True)
        self.B = torch.zeros(fuori, cfg.rango, dtype=dtype_master, device=device).requires_grad_(True)
        # 27/9 LoRA-GA: valori iniziali FERMI (None = LoRA di sempre, avanti di sempre al bit) e accumulatore del
        # gradiente pieno dW durante la calibrazione (None fuori da `inizializza_ga`)
        self.A0: Optional[torch.Tensor] = None
        self.B0: Optional[torch.Tensor] = None
        self.cattura: Optional[torch.Tensor] = None

    def parametri(self) -> list:
        return [self.A, self.B]

    def valori(self) -> dict:
        """Copie staccate sulla CPU: {"A", "B"} (+ "A0", "B0" con LoRA-GA). Ripresa e istantanea."""
        v = {"A": self.A.detach().cpu().clone(), "B": self.B.detach().cpu().clone()}
        if self.A0 is not None:
            v.update({"A0": self.A0.detach().cpu().clone(), "B0": self.B0.detach().cpu().clone()})
        return v

    @torch.no_grad()
    def imposta(self, v: dict) -> None:
        """Il contrario di `valori`: A, B (e A0, B0 se ci sono; assenti = LoRA di sempre)."""
        for k in ("A", "B"):
            t = torch.as_tensor(v[k])
            p = getattr(self, k)
            if tuple(t.shape) != tuple(p.shape):
                raise ValueError(f"{self.nome}.{k}: forma {tuple(t.shape)} != {tuple(p.shape)} (rango diverso?)")
            p.copy_(t.to(device=p.device, dtype=p.dtype))
        if ("A0" in v) != ("B0" in v):
            raise ValueError(f"{self.nome}: A0 e B0 vanno insieme")
        if "A0" in v:
            a0, b0 = torch.as_tensor(v["A0"]), torch.as_tensor(v["B0"])
            if tuple(a0.shape) != tuple(self.A.shape) or tuple(b0.shape) != tuple(self.B.shape):
                raise ValueError(f"{self.nome}: forme di A0/B0 {tuple(a0.shape)}/{tuple(b0.shape)} diverse da A/B")
            self.A0 = a0.to(device=self.A.device, dtype=self.A.dtype).clone()
            self.B0 = b0.to(device=self.B.device, dtype=self.B.dtype).clone()
        else:
            self.A0 = self.B0 = None

    @torch.no_grad()
    def imposta_ga(self, A: torch.Tensor, B: torch.Tensor) -> None:
        """A, B nuovi e A0, B0 = loro copie AL BIT (clone dopo il cast): al passo 0 A - A0 = B - B0 = 0 esatto."""
        self.A.copy_(A.to(device=self.A.device, dtype=self.A.dtype))
        self.B.copy_(B.to(device=self.B.device, dtype=self.B.dtype))
        self.A0 = self.A.detach().clone()
        self.B0 = self.B.detach().clone()

    @torch.no_grad()
    def azzera_delta(self) -> None:
        """28/9 --lora-ripristina (e2e_allena): Delta W = 0 ESATTO, l'adattatore resta allenabile. LoRA di sempre: B = 0,
        A tenuto (lo stato dell'init: dL/dB = s g^T (x A^T), dL/dA = 0 al primo passo). LoRA-GA: A = A0, B = B0 al bit
        (lo stato dell'init GA e il caso «lora_spenta» di strumenti/prova_strumenti_f32.py, quello misurato): A - A0 e
        B - B0 zero esatti, quindi avanti e Delta W = s (B A - B0 A0) all'export zero esatti; la forma resta GA."""
        if self.A0 is None:
            self.B.zero_()
        else:
            self.A.copy_(self.A0)
            self.B.copy_(self.B0)

    def delta_file(self) -> np.ndarray:
        """Delta W fp32 nell'ordine e nella forma (righe, colonne) del tensore del FILE."""
        if self.A0 is None:
            return delta_file_da(self.A.detach().cpu(), self.B.detach().cpu(), self.scala, self.voce)
        return delta_file_da(self.A.detach().cpu(), self.B.detach().cpu(), self.scala, self.voce,
                             self.A0.detach().cpu(), self.B0.detach().cpu())


@torch.no_grad()
def delta_file_da(A, B, scala: float, voce: dict, A0=None, B0=None) -> np.ndarray:
    """Delta W = s * B A (prodotto in fp64 sulla CPU, portato a fp32 una volta) nell'ordine del FILE. Stessa funzione
    per lo studente (`AdattatoreLoRA.delta_file`) e per l'export dall'istantanea (`lora.npz`): stesso bit.
    27/9 LoRA-GA (A0, B0 dati): Delta W = s * (B A - B0 A0), in fp64: zero ESATTO al passo 0 (A = A0, B = B0)."""
    A = torch.as_tensor(np.asarray(A)).to(torch.float64)
    B = torch.as_tensor(np.asarray(B)).to(torch.float64)
    if A0 is None and B0 is None:
        return np.ascontiguousarray(hf_a_file(((B @ A) * scala).to(torch.float32).numpy(), voce))
    if A0 is None or B0 is None:
        raise ValueError("A0 e B0 vanno insieme")
    A0 = torch.as_tensor(np.asarray(A0)).to(torch.float64)
    B0 = torch.as_tensor(np.asarray(B0)).to(torch.float64)
    return np.ascontiguousarray(hf_a_file(((B @ A - B0 @ A0) * scala).to(torch.float32).numpy(), voce))


@torch.no_grad()
def norma_delta(A, B, scala: float, A0=None, B0=None) -> float:
    """28/9 --lora-ripristina: |Delta W|_F in fp64 sulla CPU senza formare Delta W (matrici 2r x 2r). LoRA di sempre:
    |s B A|^2 = s^2 tr((B^T B)(A A^T)). LoRA-GA: B A - B0 A0 = (B - B0) A + B0 (A - A0) = M N con M = [B - B0 | B0],
    N = [A ; A - A0]: forma per differenze, niente cancellazione vicino all'init (dove B A e B0 A0 quasi si annullano)."""
    t = lambda v: torch.as_tensor(np.asarray(v) if not isinstance(v, torch.Tensor) else v).detach().to(  # noqa: E731
        "cpu", torch.float64)
    A, B = t(A), t(B)
    if (A0 is None) != (B0 is None):
        raise ValueError("A0 e B0 vanno insieme")
    if A0 is None:
        M, N = B, A
    else:
        A0, B0 = t(A0), t(B0)
        M, N = torch.cat([B - B0, B0], dim=1), torch.cat([A, A - A0], dim=0)
    q = float(torch.trace((M.T @ M) @ (N @ N.T))) * float(scala) ** 2
    return math.sqrt(max(q, 0.0))


class LinearLoRA(torch.nn.Module):
    """Sostituisce il `nn.Linear` HF: y = base(x) + s * (x A^T) B^T. A, B NON sono Parameter del modulo
    (restano fuori da `strato.parameters()`, come i master F32)."""

    def __init__(self, base: torch.nn.Linear, ad: AdattatoreLoRA):
        super().__init__()
        self.base = base
        self.ad = ad

    def forward(self, x):
        a = self.ad
        if a.A0 is None:                                   # LoRA di sempre (default, al bit)
            z = F.linear(F.linear(x, a.A.to(x.dtype)), a.B.to(x.dtype))
        else:                                              # LoRA-GA: s (B A - B0 A0), zero esatto al passo 0
            dt = x.dtype
            z = F.linear(F.linear(x, (a.A - a.A0).to(dt)), a.B.to(dt)) + \
                F.linear(F.linear(x, a.A0.to(dt)), (a.B - a.B0).to(dt))
        y = self.base(x) + z * a.scala
        if a.cattura is not None and y.requires_grad:      # LoRA-GA, calibrazione: dW += g^T x (il gradiente non cambia)
            xs = x.detach()
            acc = a.cattura

            def gancio(g, xs=xs, acc=acc):
                acc.addmm_(g.reshape(-1, g.shape[-1]).to(acc.dtype).T, xs.reshape(-1, xs.shape[-1]).to(acc.dtype))
                return None
            y.register_hook(gancio)
        return y


def _modulo_padre(strato, hf: str):
    pref, _, attr = hf.rpartition(".")                 # "linear_attn.in_proj_qkv.weight" -> modulo del Linear
    pref2, _, nome_lin = pref.rpartition(".")
    padre = strato.get_submodule(pref2) if pref2 else strato
    return padre, nome_lin


def _lineare_base(lin) -> bool:
    """nn.Linear, oppure i lineari congelati della ricetta B3 / del braccio rumore (e2e_bf16x3: stessa interfaccia
    in_features/out_features, ingresso e uscita fp32)."""
    return isinstance(lin, torch.nn.Linear) or type(lin).__name__ in ("LineareBF16x3", "LineareK2")


def _dove_lineare(lin) -> tuple:
    """(dispositivo, dtype dell'ingresso) del lineare di base."""
    if isinstance(lin, torch.nn.Linear):
        return lin.weight.device, lin.weight.dtype
    w = getattr(lin, "hi", None) if hasattr(lin, "hi") else getattr(lin, "W")
    return w.device, torch.float32


def installa(st, cfg: ConfigLoRA) -> None:
    """Per ogni strato caricato e famiglia chiesta presente nel suo tipo: adattatore + `LinearLoRA` al posto del
    Linear. Il valore dell'avanti non cambia (B = 0): verificato qui su un ingresso casuale, al bit."""
    import carica_strato_gguf as CSG  # sola lettura
    dt_m = torch.float64 if st.ric.dtype_residuo == torch.float64 else torch.float32
    st.lora, st.lora_cfg = {}, cfg
    st.lora_conteggi = {f: 0 for f in cfg.famiglie}
    for L in st.strati:
        mappa = CSG.MAPPA["full" if st.cfg.layer_types[L] == "full_attention" else "linear"]
        strato = st.moduli[L]
        for fam in cfg.famiglie:
            gg = FAMIGLIE_LORA[fam]
            nome = f"blk.{L}.{gg}"
            if gg not in mappa or nome not in st._tens:
                continue
            t = st._tens[nome]
            tipo = t.tensor_type.name
            if cfg.tipi and tipo not in cfg.tipi:
                continue                                  # --lora-tipi: solo i tipi chiesti (es. Q8_0, §25)
            if tipo not in TIPI:
                raise ValueError(f"{nome}: tipo {tipo} non supportato dall'export LoRA ({TIPI})")
            voce = mappa[gg]
            padre, nome_lin = _modulo_padre(strato, voce["hf"])
            lin = getattr(padre, nome_lin)
            # 27/9 integrata: con la ricetta B3 (bf16x3fix) le proiezioni non-esperte sono gia' `LineareBF16x3` (o
            # `LineareK2` nel braccio rumore del cancello): il ramo LoRA avvolge anche quelle (y = base(x) + s (x A^T) B^T)
            if not _lineare_base(lin) or getattr(lin, "bias", None) is not None:
                raise TypeError(f"{nome}: {voce['hf']} non e' un lineare senza bias ({type(lin).__name__})")
            fuori, dentro = lin.out_features, lin.in_features
            ad = AdattatoreLoRA(nome, fam, L, voce, fuori, dentro, cfg, st.device, dt_m, tipo)
            nuovo = LinearLoRA(lin, ad)
            with torch.no_grad():
                g = torch.Generator().manual_seed(seme_tensore(nome, 1))
                dev_l, dt_l = _dove_lineare(lin)
                x = torch.randn(3, dentro, generator=g).to(device=dev_l, dtype=dt_l)
                if not torch.equal(nuovo(x), lin(x)):
                    raise RuntimeError(f"{nome}: il ramo LoRA a B = 0 cambia l'uscita")
            setattr(padre, nome_lin, nuovo)
            st.lora[nome] = ad
            st.lora_conteggi[fam] += 1


@contextmanager
def fermi(adattatori):
    """Per la durata del blocco A e B non raccolgono gradiente (curvatura del passo V)."""
    tutti = [p for a in adattatori for p in a.parametri()]
    prima = [p.requires_grad for p in tutti]
    try:
        for p in tutti:
            p.requires_grad_(False)
        yield
    finally:
        for p, r in zip(tutti, prima):
            p.requires_grad_(r)


# ═══════════════════════════════════ LoRA-GA (arXiv 2407.05000) ═══════════════════════════════════
GA_GAMMA = 16.0                  # codice ufficiale (LoraGAConfig.stable_gamma) e paper T5; il paper su Llama 2 usa 64
GA_NITER = 8                     # iterazioni di sottospazio (ufficiale: torch.svd_lowrank niter=4): 8 per margine
GA_ESATTA_MAX = 1 << 20          # fino a 2^20 elementi la SVD e' quella ESATTA (LAPACK fp64 sulla CPU)


@torch.no_grad()
def svd_alti(M: torch.Tensor, k: int, seme: int, niter: int = GA_NITER) -> tuple:
    """I primi k valori/vettori singolari di M (m, n): (U (m, k), S (k,), V (n, k)) fp64 sulla CPU.
    Piccole (m n <= GA_ESATTA_MAX o k >= min(m, n) / 2): `torch.linalg.svd` esatta in fp64. Grandi: iterazione di
    sottospazio (Halko-Martinsson-Tropp 2011, Alg. 4.4; e' cio' che fa `torch.svd_lowrank` del codice ufficiale) con
    q = min(2k, m, n) colonne (ufficiale: q = 4r = 2k), rumore gaussiano da un Generator CPU col seme dato
    (riproducibile, niente RNG globale), QR a ogni mezzo giro sulla CPU in fp64 (prodotti per M sul dispositivo, nel suo
    dtype: niente algebra lineare della GPU); poi la SVD esatta fp64 della proiezione q x n."""
    m, n = M.shape
    q = min(2 * k, m, n)
    if m * n <= GA_ESATTA_MAX or q >= min(m, n):
        U, S, Vh = torch.linalg.svd(M.detach().to("cpu", torch.float64), full_matrices=False)
        return U[:, :k].contiguous(), S[:k].contiguous(), Vh[:k].T.contiguous()
    def qr(Y):                        # QR (m o n) x q sulla CPU in fp64: economica, deterministica, niente linalg GPU
        return torch.linalg.qr(Y.to("cpu", torch.float64)).Q

    def su(Q):
        return Q.to(device=M.device, dtype=M.dtype)
    g = torch.Generator().manual_seed(int(seme))
    Q = qr(M @ su(torch.randn(n, q, generator=g, dtype=torch.float64)))
    for _ in range(niter):
        Q = qr(M.T @ su(Q))
        Q = qr(M @ su(Q))
    Ub, S, Vh = torch.linalg.svd((su(Q).T @ M).to("cpu", torch.float64), full_matrices=False)
    return (Q @ Ub)[:, :k].contiguous(), S[:k].contiguous(), Vh[:k].T.contiguous()


def scala_ga(fuori: int, gamma: float) -> float:
    """c = out^(1/4) / sqrt(gamma): scala «stable» del paper (§3.3-3.4, zeta = s^2 sqrt(out) / gamma)."""
    return float(fuori) ** 0.25 / math.sqrt(float(gamma))


def ha_ga(st) -> bool:
    """Gli adattatori dello studente sono gia' in forma LoRA-GA (A0, B0 presenti: inizializzati o ripresi)."""
    return any(a.A0 is not None for a in getattr(st, "lora", {}).values())


def inizializza_ga(st, calcola: Callable[[], object], gamma: float = GA_GAMMA, log: Callable = print,
                   niter: int = GA_NITER) -> dict:
    """LoRA-GA su TUTTI gli adattatori di `st`. `calcola()` fa avanti+indietro sul lotto di calibrazione: durante
    quel solo indietro ogni `LinearLoRA` accumula dW = sum g^T x (gradiente pieno del peso, spazio HF). Poi per
    ogni adattatore dW = U S V^T, A = V[:, :r]^T c, B = U[:, r:2r] c (ArB2r, c = `scala_ga`), A0 = A, B0 = B al bit.
    Richiede 2r <= min(out, in) e dW finito e non nullo (solleva: niente inizializzazione a meta')."""
    ads = getattr(st, "lora", {})
    if not ads:
        raise ValueError("LoRA-GA senza adattatori")
    if ha_ga(st):
        raise RuntimeError("LoRA-GA: adattatori gia' inizializzati (A0 presente)")
    if not gamma > 0:
        raise ValueError(f"gamma {gamma} <= 0")
    for a in ads.values():
        if 2 * a.rango > min(a.forma):
            raise ValueError(f"{a.nome}: 2r = {2 * a.rango} > min(out, in) = {min(a.forma)}: LoRA-GA non definita")
    t0 = time.time()
    try:
        for a in ads.values():
            dt = torch.float64 if a.A.dtype == torch.float64 else torch.float32
            a.cattura = torch.zeros(a.forma, dtype=dt, device=a.A.device)
        calcola()
        dW = {n: a.cattura for n, a in ads.items()}
    finally:
        for a in ads.values():
            a.cattura = None
    t1 = time.time()
    seme = int(getattr(getattr(st, "lora_cfg", None), "seme", 0) or 0)
    per = {}
    for n, a in ads.items():
        G = dW[n]
        if not bool(torch.isfinite(G).all()):
            raise FloatingPointError(f"{n}: gradiente pieno non finito")
        nG2 = float((G.double() ** 2).sum())
        if nG2 == 0.0:
            raise RuntimeError(f"{n}: gradiente pieno nullo sul lotto di calibrazione")
        r = a.rango
        U, S, V = svd_alti(G, 2 * r, seme_tensore(n, seme), niter)
        c = scala_ga(a.forma[0], gamma)
        a.imposta_ga(V[:, :r].T * c, U[:, r:2 * r] * c)
        per[n] = {"quota_energia_2r": float((S ** 2).sum()) / nG2, "sigma_1": float(S[0]),
                  "sigma_2r": float(S[2 * r - 1]), "c": c}
        dW[n] = None
    del dW
    q = sorted(v["quota_energia_2r"] for v in per.values())
    info = {"gamma": float(gamma), "niter": niter, "adattatori": len(per),
            "quota_energia_2r": {"min": q[0], "mediana": q[len(q) // 2], "max": q[-1]},
            "c": {"min": min(v["c"] for v in per.values()), "max": max(v["c"] for v in per.values())}}
    log(f"LoRA-GA: {len(per)} adattatori da dW (indietro {t1 - t0:.1f} s, SVD {time.time() - t1:.1f} s), gamma "
        f"{gamma:g}, c {info['c']['min']:.3f}-{info['c']['max']:.3f}; energia di dW nei primi 2r: min "
        f"{q[0]:.3f} mediana {q[len(q) // 2]:.3f} max {q[-1]:.3f}")
    return {**info, "per_tensore": per}


# ═══════════════════════════════════ riquantizzazione ═══════════════════════════════════
def _gt(tipo: str):
    from gguf import GGMLQuantizationType as GT
    return getattr(GT, tipo)


def dequantizza(raw: np.ndarray, tipo: str) -> np.ndarray:
    """byte (righe, byte per riga) -> fp32 (righe, colonne): la dequantizzazione di gguf-py (= `dequant_np`)."""
    import gguf.quants as GQ
    return np.asarray(GQ.dequantize(np.ascontiguousarray(raw), _gt(tipo)), dtype=np.float32)


def _blocchi(raw: np.ndarray, tipo: str) -> np.ndarray:
    return np.ascontiguousarray(raw, dtype=np.uint8).reshape(-1, BLOCCO[tipo][1])


def _campi(b: np.ndarray, tipo: str) -> tuple:
    """(codici int16 (n, K), passo fp32 (n, K), spostamento fp32 (n, K), minimo, massimo del codice): il valore
    dequantizzato e' passo * codice - spostamento con la STESSA aritmetica fp32 di gguf-py (prova L5: al bit)."""
    n = b.shape[0]
    if tipo == "Q8_0":
        d = b[:, :2].copy().view(np.float16).astype(np.float32)
        c = b[:, 2:].view(np.int8).astype(np.int16)
        return c, np.broadcast_to(d, c.shape), np.zeros(c.shape, np.float32), -128, 127
    if tipo == "Q4_0":
        d = b[:, :2].copy().view(np.float16).astype(np.float32)
        qs = b[:, 2:]
        c = np.concatenate([qs & 0x0F, qs >> 4], axis=1).astype(np.int16)
        # gguf-py: d * (c - 8); come passo*codice - spostamento: d*c - 8d (8d esatto: stesso bit)
        return c, np.broadcast_to(d, c.shape), np.broadcast_to(8.0 * d, c.shape).astype(np.float32), 0, 15
    if tipo == "Q4_K":
        from gguf.quants import Q4_K
        d = b[:, 0:2].copy().view(np.float16).astype(np.float32)
        dmin = b[:, 2:4].copy().view(np.float16).astype(np.float32)
        sc, m = Q4_K.get_scale_min(b[:, 4:16])
        d1 = (d * sc.astype(np.float32)).reshape(n, 8, 1)
        dm = (dmin * m.astype(np.float32)).reshape(n, 8, 1)
        qs = b[:, 16:].reshape(n, 4, 1, 32) >> np.array([0, 4], dtype=np.uint8).reshape(1, 1, 2, 1)
        c = (qs & 0x0F).reshape(n, 8, 32).astype(np.int16)
        return (c.reshape(n, 256), np.broadcast_to(d1, (n, 8, 32)).reshape(n, 256),
                np.broadcast_to(dm, (n, 8, 32)).reshape(n, 256), 0, 15)
    if tipo == "Q6_K":
        ql, qh = b[:, :128], b[:, 128:192]
        sc = b[:, 192:208].view(np.int8).astype(np.float32)
        d = b[:, 208:210].copy().view(np.float16).astype(np.float32)
        ds = (d * sc).reshape(n, 16, 1)
        lo = ql.reshape(n, 2, 1, 64) >> np.array([0, 4], dtype=np.uint8).reshape(1, 1, 2, 1)
        lo = (lo & 0x0F).reshape(n, 8, 32)
        hi = qh.reshape(n, 2, 1, 32) >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape(1, 1, 4, 1)
        hi = (hi & 0x03).reshape(n, 8, 32)
        c = (lo | (hi << 4)).astype(np.int16).reshape(n, 256)
        # gguf-py: (d*sc) * (c - 32); passo*codice - spostamento con spostamento = 32 (d*sc): esatto (6+11 bit)
        passo = np.broadcast_to(ds, (n, 16, 16)).reshape(n, 256)
        return c, passo, (32.0 * passo).astype(np.float32), 0, 63
    raise ValueError(f"tipo {tipo} non supportato ({TIPI})")


def _valori(c: np.ndarray, passo: np.ndarray, sposta: np.ndarray, tipo: str) -> np.ndarray:
    """Valori dai codici con la stessa sequenza di operazioni fp32 di gguf-py."""
    if tipo == "Q8_0":
        return c.astype(np.float32) * passo
    if tipo == "Q4_0":
        return passo * (c.astype(np.int8) - np.int8(8)).astype(np.float32)
    if tipo == "Q4_K":
        return passo * c.astype(np.float32) - sposta
    if tipo == "Q6_K":
        return passo * (c.astype(np.int8) - np.int8(32)).astype(np.float32)
    raise ValueError(tipo)


def impacchetta(b: np.ndarray, c: np.ndarray, tipo: str) -> np.ndarray:
    """Blocchi `b` con i soli bit dei codici sostituiti da `c` (n, K); ogni altro byte (scale, minimi) invariato."""
    out = b.copy()
    n = b.shape[0]
    c = c.astype(np.int16)
    if tipo == "Q8_0":
        out[:, 2:] = c.astype(np.int8).view(np.uint8)
    elif tipo == "Q4_0":
        cu = c.astype(np.uint8)
        out[:, 2:] = cu[:, :16] | (cu[:, 16:] << 4)
    elif tipo == "Q4_K":
        cu = c.astype(np.uint8).reshape(n, 4, 2, 32)
        out[:, 16:] = (cu[:, :, 0, :] | (cu[:, :, 1, :] << 4)).reshape(n, 128)
    elif tipo == "Q6_K":
        cu = c.astype(np.uint8).reshape(n, 2, 2, 2, 32)            # (h, s, parte, l): gruppo g = 4h + 2s + parte
        lo, hi = cu & 0x0F, cu >> 4
        out[:, :128] = (lo[:, :, 0] | (lo[:, :, 1] << 4)).reshape(n, 128)
        hi4 = hi.reshape(n, 2, 4, 32)                              # indice k4 = 2s + parte
        out[:, 128:192] = (hi4[:, :, 0] | (hi4[:, :, 1] << 2) | (hi4[:, :, 2] << 4) | (hi4[:, :, 3] << 6)).reshape(n, 64)
    else:
        raise ValueError(tipo)
    return out


def quantizza_griglia(raw: np.ndarray, W: np.ndarray, tipo: str) -> np.ndarray:
    """Scale e minimi del blocco FERMI, codici q' = clip(rint((W + spostamento) / passo)) (fp64). Passo 0: codice
    invariato (il valore non dipende dal codice). Stessa forma di `raw`."""
    b = _blocchi(raw, tipo)
    c, passo, sposta, lo, hi = _campi(b, tipo)
    x = np.asarray(W, dtype=np.float32).reshape(c.shape).astype(np.float64)
    p = passo.astype(np.float64)
    vivi = p != 0
    q = np.where(vivi, np.rint((x + sposta.astype(np.float64)) / np.where(vivi, p, 1.0)), c)
    q = np.clip(q, lo, hi).astype(np.int16)
    return impacchetta(b, q, tipo).reshape(np.asarray(raw).shape)


_LIB = None


def _lib():
    global _LIB
    if _LIB is None:
        lib = ctypes.CDLL(LIB_GGML)
        lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
                                            ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
        lib.ggml_quantize_chunk.restype = ctypes.c_size_t
        _LIB = lib
    return _LIB


def quantizza_ggml(W: np.ndarray, tipo: str, via: str = "auto") -> np.ndarray:
    """Il quantizzatore di llama.cpp da capo, senza imatrix: (righe, colonne) fp32 -> byte (righe, byte per riga).
    via "auto": gguf-py dove esiste in python (Q8_0, Q4_0: al bit con ggml, prova L5), altrimenti
    `ggml_quantize_chunk` di libggml-base (Q4_K, Q6_K: quantize_row_*_ref, come llama-quantize senza imatrix)."""
    W = np.ascontiguousarray(W, dtype=np.float32)
    righe, col = W.shape
    K, nb = BLOCCO[tipo]
    if col % K:
        raise ValueError(f"{col} colonne non multiple di {K}")
    if via == "gguf" or (via == "auto" and tipo in ("Q8_0", "Q4_0")):
        import gguf.quants as GQ
        return np.asarray(GQ.quantize(W, _gt(tipo)), dtype=np.uint8).reshape(righe, col // K * nb)
    out = np.empty((righe, col // K * nb), dtype=np.uint8)
    n = _lib().ggml_quantize_chunk(ID_GGML[tipo], W.ctypes.data, out.ctypes.data, 0, righe, col, None)
    if n != out.nbytes:
        raise RuntimeError(f"ggml_quantize_chunk {tipo}: {n} byte, attesi {out.nbytes}")
    return out


def riquantizza(raw: np.ndarray, W: np.ndarray, tipo: str, modo: str = "griglia") -> np.ndarray:
    """W (righe, colonne) fp32 nello STESSO tipo e nella STESSA taglia di `raw` (righe, byte per riga)."""
    if modo == "griglia":
        out = quantizza_griglia(raw, W, tipo)
    elif modo == "ggml":
        out = quantizza_ggml(W, tipo).reshape(np.asarray(raw).shape)
    else:
        raise ValueError(f"modo {modo!r}: {MODI}")
    if out.shape != np.asarray(raw).shape or out.dtype != np.uint8:
        raise RuntimeError(f"riquantizzazione {tipo}/{modo}: forma {out.shape} != {np.asarray(raw).shape}")
    return out


def passo_rms(raw: np.ndarray, tipo: str) -> float:
    """rms del passo della griglia (d, d*sc) sui blocchi vivi."""
    _, p, _, _, _ = _campi(_blocchi(raw, tipo), tipo)
    p = p[p != 0].astype(np.float64)
    return float(np.sqrt((p ** 2).mean())) if p.size else 0.0


def misura_fusione(W: np.ndarray, dW: np.ndarray, Wn: np.ndarray, raw: Optional[np.ndarray] = None,
                   raw_n: Optional[np.ndarray] = None, E: Optional[np.ndarray] = None) -> dict:
    """W = W_deq, dW = Delta W voluto, Wn = dequantizzazione di Q(W + dW). Delta reale D = Wn - W.
    mantenuto = <D, dW>/|dW|^2 (quanta parte di dW resta nella sua direzione), rumore = |D - mantenuto dW| / |dW|,
    |D|/|dW|, |Q(W+dW) - Q(W)| / |dW|. Con E = W_bf16 - W: guadagno = (|E|^2 - |E - D|^2) / (|E|^2 - |E - dW|^2)
    (quota del guadagno ideale verso il bf16 che sopravvive; <0 = la fusione peggiora il file)."""
    W64, d64, D = W.astype(np.float64), dW.astype(np.float64), Wn.astype(np.float64) - W.astype(np.float64)
    nd2 = float((d64 ** 2).sum())
    m = float((D * d64).sum()) / nd2 if nd2 > 0 else float("nan")
    r = {"norma_dW_rel_W": math.sqrt(nd2 / float((W64 ** 2).sum())), "mantenuto": m,
         "rumore": float(np.sqrt(((D - m * d64) ** 2).sum() / nd2)) if nd2 > 0 else float("nan"),
         "norma_D_su_dW": float(np.sqrt((D ** 2).sum() / nd2)) if nd2 > 0 else float("nan")}
    if raw is not None and raw_n is not None:
        r["byte_cambiati"] = float((np.asarray(raw) != np.asarray(raw_n)).mean())
    if E is not None:
        E64 = E.astype(np.float64)
        e2 = float((E64 ** 2).sum())
        ideale = e2 - float(((E64 - d64) ** 2).sum())
        reale = e2 - float(((E64 - D) ** 2).sum())
        r.update({"guadagno_ideale_rel": ideale / e2, "guadagno_reale_rel": reale / e2,
                  "guadagno_sopravvive": reale / ideale if ideale != 0 else float("nan")})
    return r


# ═══════════════════════════════════ export (usato da e2e_esporta) ═══════════════════════════════════
def byte_riquantizzati(gguf: Path, lora: dict, meta: dict, modo: str = "griglia") -> tuple:
    """`lora` = {nome: {"A", "B"}} e `meta` = configurazione dell'istantanea (e2e_allena._scrivi_lora). Per ogni
    tensore: W = dequantizzazione dei byte del SORGENTE, Delta W = s B A nell'ordine del file, byte nuovi =
    riquantizza(W + Delta W) nello stesso tipo. Ritorna ({nome: byte (righe, byte per riga)}, {nome: misure})."""
    from gguf import GGUFReader
    if modo not in MODI:
        raise ValueError(f"modo {modo!r}: {MODI}")
    ts = {t.name: t for t in GGUFReader(str(gguf)).tensors}
    nuovi, misure = {}, {}
    for nome in meta["tensori"]:
        t = ts.get(nome)
        if t is None or t.tensor_type.name != meta["tipi"][nome] or t.tensor_type.name not in TIPI:
            raise ValueError(f"{nome}: tipo nel sorgente {None if t is None else t.tensor_type.name}, "
                             f"nell'istantanea {meta['tipi'][nome]}")
        tipo = t.tensor_type.name
        raw = np.ascontiguousarray(np.asarray(t.data))
        W = dequantizza(raw, tipo)
        v = lora[nome]
        dW = delta_file_da(v["A"], v["B"], meta["scala"], meta["voci"][nome], v.get("A0"), v.get("B0"))
        if dW.shape != W.shape:
            raise ValueError(f"{nome}: Delta W {dW.shape}, tensore {W.shape}")
        if not np.isfinite(dW).all():
            raise ValueError(f"{nome}: Delta W non finito")
        # 28/9 --lora-ripristina: Delta W = 0 ESATTO (adattatore azzerato, o strato congelato mai allenato) = i byte del
        # sorgente in ogni modo (il modo ggml riquantizzerebbe da capo e sposterebbe i pesi): lo stesso GGUF di una corsa
        # senza LoRA. Col modo griglia cambia nulla (griglia(W_deq) == byte, prova L5).
        nullo = not dW.any()
        nuovo = raw.copy() if nullo else riquantizza(raw, W + dW, tipo, modo)
        misure[nome] = {"tipo": tipo, "rho": float(np.sqrt((dW.astype(np.float64) ** 2).mean())) / max(passo_rms(raw, tipo), 1e-30),
                        **misura_fusione(W, dW, dequantizza(nuovo, tipo), raw, nuovo)}
        if nullo:
            misure[nome]["delta_nullo"] = True
        nuovi[nome] = nuovo
    return nuovi, misure


@torch.no_grad()
def installa_pesi(st, L: int, nuovi: dict) -> list:
    """Nello strato L dello studente `st`, il peso dei moduli HF dei tensori in `nuovi` (byte riquantizzati)
    diventa la loro dequantizzazione, con la STESSA via del caricatore (`dequant_np` -> `_applica` -> fp32 ->
    dtype dello strato): il riferimento della ricarica. Ritorna i nomi toccati."""
    import carica_strato_gguf as CSG  # sola lettura
    mappa = CSG.MAPPA["full" if st.cfg.layer_types[L] == "full_attention" else "linear"]
    fatti = []
    for nome, raw in nuovi.items():
        if not nome.startswith(f"blk.{L}."):
            continue
        voce = mappa[nome.split(".", 2)[2]]
        padre, nl = _modulo_padre(st.moduli[L], voce["hf"])
        lin = getattr(padre, nl)
        lin = lin.base if isinstance(lin, LinearLoRA) else lin
        W = dequantizza(raw, st._tens[nome].tensor_type.name)
        v = torch.from_numpy(np.array(CSG._applica(voce, W), dtype=np.float32, copy=True))
        if type(lin).__name__ in _BASI_RICETTA:          # 27/9: base bf16x3 / K2 (ricette B3 / Bk2)
            lin.imposta_peso(v)
        else:
            lin.weight.data = v.to(device=lin.weight.device, dtype=lin.weight.dtype)
        fatti.append(nome)
    return fatti


# ═══════════════════════════════════ misura della sopravvivenza (CPU) ═══════════════════════════════════
TENSORI_MISURA = ("blk.0.attn_qkv.weight", "blk.4.attn_qkv.weight", "blk.0.attn_gate.weight", "blk.0.ssm_out.weight",
                  "blk.3.attn_q.weight", "blk.3.attn_v.weight", "blk.3.attn_output.weight",
                  "blk.5.ffn_gate_shexp.weight", "blk.0.ffn_down_shexp.weight")
RHO = (0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0)


def _tensori_bf16() -> dict:
    from gguf import GGUFReader
    out = {}
    for p in GGUF_BF16:
        for t in GGUFReader(str(p)).tensors:
            out[t.name] = t
    return out


def sopravvivenza(tensori=TENSORI_MISURA, rango: int = 16, rho=RHO, gguf: Optional[Path] = None,
                  log: Callable = print) -> dict:
    """Per ogni tensore vero dello studente: E = W_bf16 - W_deq (bf16 ORIGINALE, stesso ordine del file: correlazione
    controllata), U = miglior correttore di rango r di E (SVD troncata, torch.svd_lowrank), poi per ogni modo:
    (a) Delta W = U alla norma naturale, (b) Delta W = U riscalato a rho = rms(Delta W)/rms(passo) per ogni rho;
    misure di `misura_fusione`. Piu' la deriva del modo ggml a Delta W = 0 (|Q(W) - W| / |E|)."""
    import e2e_studente as S
    from dequant_gguf_torch import dequant_np
    from gguf import GGUFReader
    gguf = Path(gguf) if gguf else S.GGUF_THOR_T
    ts = {t.name: t for t in GGUFReader(str(gguf)).tensors}
    tb = _tensori_bf16()
    ris = {"versione": VERSIONE, "gguf": str(gguf), "bf16": [str(p) for p in GGUF_BF16], "rango": rango,
           "rho": list(rho), "tensori": {}}
    for nome in tensori:
        t0 = time.time()
        t = ts[nome]
        tipo = t.tensor_type.name
        raw = np.ascontiguousarray(np.asarray(t.data))
        W = dequant_np(t)
        Wb = dequant_np(tb[nome]).astype(np.float32)
        if Wb.shape != W.shape:
            raise RuntimeError(f"{nome}: forma bf16 {Wb.shape} != {W.shape}")
        corr = float(np.corrcoef(W.ravel()[::7], Wb.ravel()[::7])[0, 1])
        if corr < 0.95:
            raise RuntimeError(f"{nome}: correlazione bf16/studente {corr:.4f}: ordine del file diverso?")
        E = (Wb.astype(np.float64) - W.astype(np.float64)).astype(np.float32)
        torch.manual_seed(20260927)
        Ut, St, Vt = torch.svd_lowrank(torch.from_numpy(E), q=rango + 8, niter=6)
        U = ((Ut[:, :rango] * St[:rango]) @ Vt[:, :rango].T).numpy()
        pr = passo_rms(raw, tipo)
        rms = lambda a: float(np.sqrt((a.astype(np.float64) ** 2).mean()))  # noqa: E731
        r = {"tipo": tipo, "forma": list(W.shape), "corr_bf16": corr, "rms_W": rms(W), "rms_E": rms(E),
             "rms_passo": pr, "rms_E_su_passo": rms(E) / pr, "energia_E_nel_rango": float((U.astype(np.float64) ** 2).sum()
                                                                                        / (E.astype(np.float64) ** 2).sum()),
             "rho_naturale": rms(U) / pr, "modi": {}}
        for modo in MODI:
            m = {}
            if modo == "ggml":
                raw0 = riquantizza(raw, W, tipo, "ggml")
                W0 = dequantizza(raw0, tipo)
                m["deriva_a_dW_zero"] = {"rel_E": float(np.linalg.norm((W0 - W).astype(np.float64))
                                                        / np.linalg.norm(E.astype(np.float64))),
                                         "errore_bf16_prima": float(np.linalg.norm(E.astype(np.float64))),
                                         "errore_bf16_dopo": float(np.linalg.norm((Wb - W0).astype(np.float64))),
                                         "byte_cambiati": float((raw0 != raw).mean())}
            else:
                raw0 = riquantizza(raw, W, tipo, "griglia")
                m["deriva_a_dW_zero"] = {"byte_identici": bool(np.array_equal(raw0, raw))}
            rn = riquantizza(raw, W + U, tipo, modo)
            m["naturale"] = misura_fusione(W, U, dequantizza(rn, tipo), raw, rn, E)
            per_rho = {}
            for x in rho:
                dW = (U * (x * pr / rms(U))).astype(np.float32)
                rn = riquantizza(raw, W + dW, tipo, modo)
                per_rho[str(x)] = misura_fusione(W, dW, dequantizza(rn, tipo), raw, rn, E)
            m["per_rho"] = per_rho
            r["modi"][modo] = m
        r["secondi"] = round(time.time() - t0, 1)
        ris["tensori"][nome] = r
        g, gg = r["modi"]["griglia"], r["modi"]["ggml"]
        log(f"{nome} {tipo} {tuple(W.shape)}: rms E/passo {r['rms_E_su_passo']:.3f}, energia di E nel rango {rango} "
            f"{100 * r['energia_E_nel_rango']:.2f}%, rho naturale {r['rho_naturale']:.4f} | griglia: mantenuto "
            f"{g['naturale']['mantenuto']:.3f} guadagno {g['naturale']['guadagno_sopravvive']:.3f}; per rho " +
            " ".join(f"{k}:{v['mantenuto']:.2f}" for k, v in g["per_rho"].items()) +
            f" | ggml: deriva |Q(W)-W|/|E| {gg['deriva_a_dW_zero']['rel_E']:.3f}, mantenuto naturale "
            f"{gg['naturale']['mantenuto']:.3f} guadagno {gg['naturale']['guadagno_sopravvive']:.3f}; per rho " +
            " ".join(f"{k}:{v['mantenuto']:.2f}" for k, v in gg["per_rho"].items()) + f" ({r['secondi']} s)")
        del W, Wb, E, U, raw
    return ris


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sopravvivenza")
    p.add_argument("--rango", type=int, default=16)
    p.add_argument("--tensori", default=",".join(TENSORI_MISURA))
    p.add_argument("--rho", default=",".join(str(x) for x in RHO))
    p.add_argument("--gguf", default=None)
    p.add_argument("--uscita", default=None)
    a = ap.parse_args(argv)
    torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "4")))
    ris = sopravvivenza([x for x in a.tensori.split(",") if x], a.rango, tuple(float(x) for x in a.rho.split(",")),
                        a.gguf)
    if a.uscita:
        Path(a.uscita).parent.mkdir(parents=True, exist_ok=True)
        Path(a.uscita).write_text(json.dumps(ris, indent=1))
    print(json.dumps(ris, indent=1)[:20000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
