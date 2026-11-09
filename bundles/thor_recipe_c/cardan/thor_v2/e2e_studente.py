#!/usr/bin/env python3
"""Studente Thor-T per la fase E2E sulle sole scale fp16 degli esperti (progetto `e2e_progetto_scelto.md`).

26/9/2026. File NUOVO (modulo «studente» del flusso E2E). Nessun file esistente e' modificato; si importano
in SOLA LETTURA: `carica_strato_gguf` (mappa GGUF->HF verificata, A_log corretto dall'audit B9),
`dequant_gguf_torch`, `e2e_codici_scale` / `e2e_campi_scala` (decode dai byte, layout), `rotazione_gpu` (FWHT),
il codec di casa (`S1024`).

Cosa c'e' qui, e perche' (numeri della revisione avversaria, §9 del progetto):

1. **Studente dal GGUF.** I non-esperti si dequantizzano con la mappa verificata (`carica_strato_gguf.MAPPA`,
   `_applica`, `dequant_np`: la STESSA via di `pesi_strato_da_gguf`, saltando i soli tensori `per_esperto`) e
   si caricano in un `Qwen3_5MoeDecoderLayer` (attenzione piena / GDN come li ha controllati B9). Gli esperti
   restano i BYTE del GGUF (codici fermi) e sono letti solo dai kernel: in memoria non esiste una seconda copia
   dei codici. Le uniche variabili allenate sono theta (E, R, ng) fp32, d_eff = fp16_RNE(d0 * exp(theta)).
2. **Funzione autograd PER FAMIGLIA** (`FamigliaEsperti`, A1). Una chiamata per (strato, famiglia) su tutti i
   token gia' ordinati per esperto. Salva solo l'ingresso e le scale; nell'indietro il peso di ogni esperto si
   RICOSTRUISCE dai byte: dX con GEMM raggruppata, dL/dd = somma_gruppo trit (.) (dY^T X) accumulata in fp32.
   Nessun tensore (E, R, C) salvato, nessun gradiente grande quanto W sommato E volte: indietro O(E), non O(E^2).
3. **Decodifica dei trit FUSA** (A3). Kernel Triton: i trit si ricavano dai byte DENTRO il prodotto (avanti e
   dX) e dentro la riduzione di dL/dd, con la stessa aritmetica intera di `dequantize_row_tq1_0`
   (((b * 3^n) & 255) * 3) >> 8. Il peso decodificato non passa mai dalla memoria globale (motore "triton").
   Variante "triton_denso": un kernel decodifica byte -> peso fp16 e il prodotto lo legge (M0 sceglie il piu'
   veloce). Motore "torch": stessa matematica esperto per esperto, per la CPU e per le prove.
   Motore "triton2" (27/9): kernel _k_gemm2/_k_dx2/_k_dd2 (schema fisso dei pezzi da 32 colonne, scala per
   gruppo, ordine a gruppi GROUP_M per la L2, dL/dd a K variabile con dY letto contiguo, gate+up nella stessa
   passata): avanti e dX AL BIT uguali a "triton", dL/dd salvo l'ordine delle somme fp32 (~2e-7 relativo).
   Prove: e2e_test_triton2.py; cancello GPU: strumenti/cancello_kernel.sh.
4. **Ricette** (§2.6 F0b):
   - **B** («come il motore»): attivazione ruotata R x = FWHT(S (.) x)/32 calcolata in fp32 e portata a fp16
     (come `rot_y_tq1_0r_1024`), peso fp16 = trit x d (ESATTO), accumulo fp32, uscita e residuo fp32;
     non-esperti fp32. Con GEMM fp16 nell'indietro la perdita va scalata per K (A2): lo fa chi allena.
   - **A** («come la forgia»): peso de-ruotato R^T(trit x d) in fp32 portato a bf16, attivazioni bf16,
     non-esperti bf16.
   - **C** (26/9, candidata di velocita'): esperti ESATTAMENTE come B, non-esperti e residuo bf16 come A,
     router fp32 (peso F32 del GGUF). In B le proiezioni non-esperte girano in fp32, senza la via WMMA di
     gfx1151. Scelgono M0 (tempo) e F0b (fedelta' contro Vulkan), come fra A e B.
   - **B3** (27/9, candidata di velocita' che tiene la precisione di B): come B, ma le proiezioni non-esperte
     (>= 256 righe) in bf16x3 (`e2e_bf16x3`, Henry et al. 2019): 3 GEMM bf16 con accumulo fp32, pesi scomposti
     una volta. `B3_ind32` = B3 con l'indietro degli esperti fp32 (come B_ind32). Cancello: `strumenti/cancello_b3.sh`.
5. **Checkpoint per strato** con l'instradamento top-k SALVATO nella prima passata e riusato nel ricalcolo;
   contatore dei cambi (top-k vivo del ricalcolo diverso da quello salvato), tenuto sul dispositivo: nessuna
   sincronizzazione nel ciclo (Piano d'instradamento e tabelle dei riquadri senza .item()).
6. **Contesto come il teacher**: modo «riga» (default: la riga da 4096 e' una sequenza causale) oppure modo
   «segmenti» (`inizi` (B, T) 0/1 dai cu_seqlens: ogni segmento e' una sequenza a se', posizioni da 0, stato
   GDN azzerato), come `e2e_teacher --modo`. Chi allena DEVE usare il modo del teacher (`CacheMaestro.modo`).
7. **Niente copie ne' sincronizzazioni nel ciclo** (26/9): segni S1024 in cache per (dtype, dispositivo) (prima:
   copia host->GPU a ogni rotazione); FWHT compilata con `torch.compile` (aritmetica identica: S11 al bit su CPU,
   `misura` la riprova sulla GPU e la spegne se non e' al bit; E2E_RUOTA_COMPILATA=0 la spegne a mano);
   combinazione degli esperti fusa (`CombinaEsperti`: pesi d'instradamento, raccolta e somma in un kernel,
   indietro in due; S13 contro la via torch; E2E_COMBINA_FUSA=0 la spegne); `controlla` a UNA sincronizzazione;
   `Osservatore`: trabocco fp16 (rischio 8) e quote del gradiente a zero/subnormali in fp16 (cancello G1, S12).

API per chi allena: `Studente(strati=range(40), ricetta="B", motore_nome="triton", device="cuda")`,
`parametri()` (theta), `gradiente(ids, perdita, inizi=...)` oppure `avanti` / `esce` / `indietro`,
`controlla()` una volta per passo (scale non finite, trabocco fp16, cambi d'instradamento),
`diagnostica_indietro()` al passo 0 (cancello G1), `scale_per_export()` per
`e2e_codici_scale.patcha_scale`, `carica_embedding()` e `testa()` (Q5_K del GGUF, fp32).
27/9 `allena_f32=(famiglie)` (`--allena-f32` di e2e_allena; vuota = identico al bit): i tensori F32 del GGUF
di quelle famiglie (norme, router `ffn_gate_inp`, porta dell'esperto condiviso, parametri GDN, output_norm)
diventano `TensoreF32.master` (valori del file, fp32) con gradiente; `parametri_f32()`, `chiavi_f32()`,
`valori_f32()` / `imposta_f32()` (ripresa), `f32_per_export()` (export), `f32_fermi()`; `controlla()` solleva
anche su master non finiti (o ssm_a >= 0). Il parametro HF si ricalcola dal master a ogni chiamata dello
strato: avanti e ricalcolo dell'indietro vedono lo stesso router (instradamento salvato e cambi invariati).
Prove: `e2e_test_f32.py` (A23-A26).
27/9 `lora=e2e_lora.ConfigLoRA(rango, alpha, famiglie)` (`--lora-nonesperti` di e2e_allena; None/rango 0 = identico
al bit, nessun modulo toccato): i nn.Linear non esperti delle famiglie chieste diventano `e2e_lora.LinearLoRA`
(y = x W^T + s (x A^T) B^T, B = 0 al passo 0); `parametri_lora()`, `chiavi_lora()`, `valori_lora()` /
`imposta_lora()` (ripresa), `lora_per_export()` (Delta W nell'ordine del file), `lora_fermi()`. Prove: e2e_test_lora.py.
27/9 LoRA-GA (`e2e_lora.inizializza_ga`, `--lora-init ga` di e2e_allena): con A0, B0 negli adattatori l'avanti e'
y = x W^T + s [(x (A-A0)^T) B^T + (x A0^T) (B-B0)^T] (zero esatto al passo 0); `valori_lora()` li porta, `imposta_lora()`
li rimette (assenti = LoRA di sempre). Prove: e2e_test_lora_ga.py.
28/9 `densi_ternari=(famiglie)` (`--densi-ternari` di e2e_allena; vuota = identico al bit): i tensori DENSI non esperti
gia' ternari nel GGUF (attn_q/k/v/output, attn_qkv, attn_gate, ssm_out, ffn_gate/up/down_shexp in TQ1_0R_1024 / TQ1_0)
diventano famiglie a esperto UNICO (E = 1) dentro `StratoEsperti.fam` dopo gate/up/down: theta, `chiavi()`,
`parametri()`, `scale_per_export()`, il motore V, i registri e l'export li trattano come gli esperti. Il Linear HF diventa
`LineareTernaria` (FamigliaEsperti su un Piano a un esperto, permutazioni file<->HF). Prove: e2e_test_densi_ternari.py.

Uso (prove su CPU, kernel Triton nell'interprete): `e2e_test_studente.py`.
Misura su GPU (M0: solo quando la GPU e' libera, `strumenti/gpu_occupata.sh` a 0):
    python3 e2e_studente.py --misura --strati 0,1,2,3 --token 8192,16384,32768 --ricetta B,C \
        --motori triton,triton_denso --combina fusa,torch --spazza --profila
        # -> @@ROOT:models@@/gguf/tony-tern/e2e_studente/misura_m0.json
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
sys.path.insert(0, "@@ROOT:models@@/build/llamacpp-tq1-k1024/gguf-py")
sys.path.insert(0, "@@ROOT:forge@@/kernel-rotazione")

from gguf import GGUFReader  # noqa: E402
from tq1_0r_1024_codec import S1024  # noqa: E402  (sola lettura)
import e2e_codici_scale as CS  # noqa: E402
import e2e_campi_scala as CAMPI  # noqa: E402
import e2e_densi_piegati as _DP  # noqa: E402  (1/10 densi-piegati; spento di default)

GGUF_THOR_T = Path(os.getenv("E2E_GGUF", "@@ROOT:models@@/gguf/tony-tern/Thor-v2-ThorT-scor150.gguf"))
SRC_HF = Path(os.getenv("THOR_HF_DIR", "@@ROOT:models@@/gguf/thor-src-hf"))     # SOLO la configurazione
FAMIGLIE = {"gate": "ffn_gate_exps", "up": "ffn_up_exps", "down": "ffn_down_exps"}
# 28/9 --densi-ternari (ramo densi-ternari): tensori DENSI non esperti gia' ternari nel GGUF (variante "A" di
# @@ROOT:depot@@/tutto_ternario: TQ1_0R_1024 con ingresso 2048/4096, TQ1_0 per ffn_down_shexp) allenati come un
# «esperto UNICO» (E = 1): stessi byte, stesse theta (StratoEsperti.theta[famiglia]), stessa FamigliaEsperti, stesso
# motore (e motore V), stessi registri e stesso export dei codici. famiglia -> nome del tensore senza "blk.N." e ".weight".
FAMIGLIE_DENSE = {
    "attn_q": "attn_q", "attn_k": "attn_k", "attn_v": "attn_v", "attn_output": "attn_output",
    "attn_qkv": "attn_qkv", "attn_gate": "attn_gate", "ssm_out": "ssm_out",
    "ffn_gate_shexp": "ffn_gate_shexp", "ffn_up_shexp": "ffn_up_shexp", "ffn_down_shexp": "ffn_down_shexp",
}
# 11/10/2026 E2E-C1 (OBIETTIVO C, ospite C0B): in C0B anche ssm_alpha / ssm_beta (in_proj_a / in_proj_b del GDN,
# 2048 -> 32, relazione perm_righe come attn_qkv) sono TQ1_0R_1024 (in TT2 erano F32 e si allenavano con --allena-f32).
# Con E2E_DENSI_SSM_AB=1 diventano due famiglie dense in piu' (esperto unico: theta, codici del passo V, export), messe
# DOPO le dieci di sempre: gli indici di registro dei dieci primi piani restano quelli di sempre. Spento (default) =
# albero d'origine al bit. Prove: e2e_test_ssm_ab.py (CPU).
_SSM_AB = os.environ.get("E2E_DENSI_SSM_AB") == "1"
if _SSM_AB:
    FAMIGLIE_DENSE.update({"ssm_alpha": "ssm_alpha", "ssm_beta": "ssm_beta"})
GRUPPI_DENSI = {
    "attn": ("attn_q", "attn_k", "attn_v", "attn_output"),            # attenzione piena (10 strati)
    "gdn": ("attn_qkv", "attn_gate", "ssm_out") + (("ssm_alpha", "ssm_beta") if _SSM_AB else ()),   # Gated DeltaNet (30)
    "shexp": ("ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp"),      # esperto condiviso (40 strati)
    "tutte": tuple(FAMIGLIE_DENSE),
}
# 29/9 secondo piano ternario dei densi (ramo secondo-piano-e2e; motore: fork llamacpp-tq1-k1024 ramo
# secondo-piano-densi-tutti 8d2506e): tensori blk.N.<nome>2.weight, stesso tipo e forma del primo piano, uscita
# W1 x + W2 x. Non si chiedono a parte: con --densi-ternari, il secondo piano di una famiglia chiesta si allena
# da solo SE e' nel GGUF (scale theta proprie, codici del passo V come famiglia propria, export); senza *2 nel GGUF
# nulla cambia (identico al bit a 7d92c83). Nel cammino dequantizzato (famiglia densa NON chiesta) il Linear HF
# riceve W1 + W2 (somma fp32 delle due dequantizzazioni).
SUFFISSO_P2 = "2"
FAMIGLIE_DENSE2 = {f"{k}{SUFFISSO_P2}": f"{v}{SUFFISSO_P2}" for k, v in FAMIGLIE_DENSE.items()}
# 30/9 TERZO piano dei densi (ramo tt5-terzo-piano da cfb718f; R2B3/R2B3m, motore terzo-piano-densi 0ed1faa): tensori
# blk.N.<nome>3.weight, stesso tipo e forma del primo piano, uscita W1 x + W2 x + W3 x (somme nell'ordine del motore).
# Stessa macchina del secondo piano: famiglia propria (theta, codici del passo V, export), dopo TUTTI i *2 nelle chiavi;
# nel cammino dequantizzato il Linear HF riceve W1 + W2 + W3. Senza *3 nel GGUF nulla cambia (identico al bit a cfb718f).
SUFFISSO_P3 = "3"
FAMIGLIE_DENSE3 = {f"{k}{SUFFISSO_P3}": f"{v}{SUFFISSO_P3}" for k, v in FAMIGLIE_DENSE.items()}
# 9/10 (cura P1 di RILQT/COMBO_A_shexp/DIAGNOSI_DIVARIO.md) QUARTO piano dei densi (R2B4 / PV12P / COMBO A; motore
# thor-e1-shexp qwen35moe.cpp: wq4/wqkv4/wqkv_gate4/ssm_out4/ffn_*_shexp4, `cur += mm(W4, x)` DOPO il terzo piano e
# PRIMA dei residui _lra/_lrb). Stessa macchina del terzo: famiglia propria dopo TUTTI i *3, uscita W1x+W2x+W3x+W4x;
# nel cammino dequantizzato il Linear HF riceve ((W1 + W2) + W3) + W4. Senza *4 nel GGUF nulla cambia (al bit).
SUFFISSO_P4 = "4"
FAMIGLIE_DENSE4 = {f"{k}{SUFFISSO_P4}": f"{v}{SUFFISSO_P4}" for k, v in FAMIGLIE_DENSE.items()}
# 9/10 (cura P1) SECONDO piano degli esperti instradati (PV12 / COMBO A; motore llama-graph.cpp build_moe_ffn
# `second_plane`): ffn_{gate,up,down}_exps2 (E2 slot) + mappa F32 ffn_X_exps2_map [n_expert] (slot dell'esperto nel 2°
# piano, -1 = nessuno). Uscita per assegnazione (token, esperto e): mm(W_e, x) + [map[e] >= 0] mm(W2[map[e]], x), per
# gate, up e down separatamente (mappe proprie), PRIMA di silu*up e PRIMA del peso del router. Senza mappa: prefisso
# dei primi E2 esperti (id < E2). ODINO_P2_COMP (compensa_freddi) non vale con la mappa e non e' usato dalle corse.
SUFFISSO_E2 = "2"


def e_densa(n: str) -> bool:
    """Famiglia densa a esperto unico (primo o secondo piano)."""
    return n in FAMIGLIE_DENSE or n in FAMIGLIE_DENSE2 or n in FAMIGLIE_DENSE3 or n in FAMIGLIE_DENSE4


def primo_piano(n: str) -> str:
    """"attn_q2" -> "attn_q"; una famiglia di primo piano resta se stessa."""
    if n in FAMIGLIE_DENSE2:
        return n[:-len(SUFFISSO_P2)]
    if n in FAMIGLIE_DENSE4:                      # 9/10 quarto piano
        return n[:-len(SUFFISSO_P4)]
    return n[:-len(SUFFISSO_P3)] if n in FAMIGLIE_DENSE3 else n


def famiglie_dense(voci) -> tuple:
    """"a,b" o iterabile (famiglie o gruppi attn/gdn/shexp/tutte) -> tupla canonica (ordine di FAMIGLIE_DENSE).
    Vuota = nessuna (default: studente identico al bit)."""
    if isinstance(voci, str):
        voci = voci.split(",")
    chieste = set()
    for v in voci:
        v = v.strip()
        if not v:
            continue
        if v.endswith(".weight"):
            v = v[:-len(".weight")]
        if v in GRUPPI_DENSI:
            chieste.update(GRUPPI_DENSI[v])
        elif v in FAMIGLIE_DENSE2 or v in FAMIGLIE_DENSE3 or v in FAMIGLIE_DENSE4:   # 29/9 (30/9 terzo, 9/10 quarto)
            chieste.add(primo_piano(v))
        elif v in FAMIGLIE_DENSE:
            chieste.add(v)
        else:
            raise ValueError(f"famiglia densa sconosciuta {v!r}: {sorted(FAMIGLIE_DENSE) + sorted(GRUPPI_DENSI)}")
    return tuple(f for f in FAMIGLIE_DENSE if f in chieste)


def nome_tensore_famiglia(n: str) -> str:
    """Famiglia (esperti gate/up/down o densa) -> nome del tensore GGUF senza "blk.N." e senza ".weight"."""
    return FAMIGLIE[n] if n in FAMIGLIE else FAMIGLIE_DENSE[n] if n in FAMIGLIE_DENSE else \
        FAMIGLIE_DENSE2[n] if n in FAMIGLIE_DENSE2 else FAMIGLIE_DENSE3[n] if n in FAMIGLIE_DENSE3 else \
        FAMIGLIE_DENSE4[n]


def famiglie_dense_da_nomi(nomi) -> tuple:
    """Nomi di tensori GGUF ("blk.12.attn_q.weight", ...) -> famiglie dense presenti (ordine canonico). Serve a chi
    ricostruisce lo studente da un'istantanea o da un registro (export, ricarica): le scale dicono quali densi c'erano."""
    per_suffisso = {f"{v}.weight": k for k, v in FAMIGLIE_DENSE.items()}
    per_suffisso.update({f"{v}.weight": primo_piano(k) for k, v in FAMIGLIE_DENSE2.items()})   # 29/9: *2 -> primo piano
    per_suffisso.update({f"{v}.weight": primo_piano(k) for k, v in FAMIGLIE_DENSE3.items()})   # 30/9: *3 -> primo piano
    per_suffisso.update({f"{v}.weight": primo_piano(k) for k, v in FAMIGLIE_DENSE4.items()})   # 9/10: *4 -> primo piano
    trovate = set()
    for n in nomi:
        parti = str(n).split(".", 2)
        if len(parti) == 3 and parti[0] == "blk" and parti[2] in per_suffisso:
            trovate.add(per_suffisso[parti[2]])
    return tuple(f for f in FAMIGLIE_DENSE if f in trovate)

try:  # Triton c'e' nel venv della forgia; senza, resta il motore torch (CPU)
    import triton
    import triton.language as tl
    _TRITON = True
except Exception:  # pragma: no cover
    _TRITON = False

if _TRITON and os.getenv("TRITON_INTERPRET") == "1":
    # Solo per le prove su CPU (interprete di Triton). 26/9: con numpy 2.5 l'interprete di Triton 3.6 fallisce su
    # `range(0, K, BK)` con K a runtime: i suoi scalari sono array (1,) e `int()` su un array non 0-d ora solleva
    # («only 0-dimensional arrays can be converted to Python scalars»). Si rimette `__index__` sull'unico
    # elemento; nessun effetto fuori dall'interprete (il kernel compilato non passa di qui).
    from triton.runtime import interpreter as _interp

    _orig_patch = _interp._patch_lang_tensor

    def _patch_lang_tensor_np2(tensor, scope):
        _orig_patch(tensor, scope)
        scope.set_attr(tensor, "__index__", lambda self: int(np.asarray(self.handle.data).reshape(-1)[0]))

    _interp._patch_lang_tensor = _patch_lang_tensor_np2


# ═══════════════════════════════════ ricette e formati ═══════════════════════════════════
@dataclass(frozen=True)
class Ricetta:
    """dtype_residuo: flusso residuo e non-esperti. dtype_gemm: ingressi dei prodotti degli esperti (le
    uscite sono fp32 se dtype_gemm e' fp16, altrimenti dtype_gemm). ruota: True = si ruota l'ATTIVAZIONE e il
    peso resta trit x d nel dominio ruotato (ricetta B); False = peso de-ruotato denso (ricetta A)."""
    nome: str
    dtype_residuo: torch.dtype
    dtype_gemm: torch.dtype
    ruota: bool
    dtype_indietro: Optional[torch.dtype] = None     # None = dtype_gemm; fp32 = indietro di riferimento (prove)
    router_fp32: bool = False                        # router in fp32 (peso F32 del GGUF) anche con residuo bf16
    non_esperti_x3: bool = False                     # 27/9: proiezioni non-esperte in bf16x3 (e2e_bf16x3), ricetta B3
    non_esperti_k2: bool = False                     # 27/9: le stesse in fp32 con K in due meta' (braccio rumore, Bk2)
    # 27/9 (ramo indietro-bf16x3): indietro degli ESPERTI a prodotti divisi hi/lo con i kernel fp16/bf16 della
    # ricetta B ("fp16x2" | "bf16x3", `e2e_indietro_hilo`); "" = spento. Vale solo con dtype_indietro None: la
    # ricetta fp32 di riferimento di G1 (dtype_indietro fp32) torna all'indietro fp32 esatto.
    indietro_diviso: str = ""

    @property
    def dtype_uscita(self) -> torch.dtype:
        return torch.float32 if self.dtype_gemm == torch.float16 else self.dtype_gemm


RICETTE = {
    "A": Ricetta("A", torch.bfloat16, torch.bfloat16, False),
    "B": Ricetta("B", torch.float32, torch.float16, True),
    # solo prove: la stessa algebra in float64 / float32, per confronti stretti col riferimento
    "B64": Ricetta("B64", torch.float64, torch.float64, True),
    "A64": Ricetta("A64", torch.float64, torch.float64, False),
    "B32": Ricetta("B32", torch.float32, torch.float32, True),
    "A32": Ricetta("A32", torch.float32, torch.float32, False),
    # stesso avanti di B, indietro (dX e dL/dd) con ingressi fp32: il riferimento di G1 / A2 per la scala K
    "B_ind32": Ricetta("B_ind32", torch.float32, torch.float16, True, torch.float32),
    # 26/9 10:0x, candidata di VELOCITA' per M0/F0b (la sceglie la misura, come A e B): esperti ESATTAMENTE come B
    # (attivazione ruotata fp16, trit x d fp16, accumulo fp32, indietro fp16 con la scala K), non-esperti e
    # residuo bf16 come A (la forgia). Motivo: in B le proiezioni di attenzione/GDN, l'esperto condiviso e il GDN
    # girano in fp32, e gfx1151 non ha la via WMMA per fp32 (§2.6 rev.): sono ~35 M parametri per strato, cioe'
    # tanti FLOP quanti gli esperti attivi. Il router resta fp32 (il suo peso nel GGUF e' F32; bf16 sposterebbe
    # scelte di top-k che llama.cpp non sposta).
    "C": Ricetta("C", torch.bfloat16, torch.float16, True, None, True),
    # 27/9, candidata di VELOCITA' senza toccare la precisione di B: tutto come B (residuo fp32, esperti fp16 esatti,
    # router fp32), ma le proiezioni non-esperte con >= 256 righe (attenzione q/k/v/o, GDN in_proj_qkv/in_proj_z/
    # out_proj, esperto condiviso) in bf16x3 (`e2e_bf16x3`: 3 GEMM bf16 con accumulo fp32, errore ~1e-5 relativo
    # invece di ~1e-6 della GEMM fp32, prova X1). B3_ind32 = B3 con l'indietro degli esperti fp32, come B_ind32 per B.
    # La sceglie il cancello `strumenti/cancello_b3.sh` (tempo, KL(B||B3), dL/dtheta contro B_ind32).
    "B3": Ricetta("B3", torch.float32, torch.float16, True, None, False, True),
    "B3_ind32": Ricetta("B3_ind32", torch.float32, torch.float16, True, torch.float32, False, True),
    # 27/9 pomeriggio, SOLO per il cancello B3 (braccio "rumore"): B_ind32 con le stesse proiezioni non-esperte in
    # fp32 ma la somma su K in due meta' (`e2e_bf16x3.LineareK2`): un arrotondamento fp32 diverso, nient'altro.
    # Misura quanto KL e dL/dtheta di B si spostano da soli sui 40 strati (instradamento top-k + ricorrenze).
    "Bk2_ind32": Ricetta("Bk2_ind32", torch.float32, torch.float16, True, torch.float32, False, False, True),
    # 27/9 (ramo indietro-bf16x3), candidate di VELOCITA' dell'indietro: avanti ESATTAMENTE come B (B_ind*) o B3
    # (B3_ind*), indietro degli esperti a prodotti divisi (`e2e_indietro_hilo`) al posto dei prodotti fp32 di
    # B_ind32 / B3_ind32: "fp16x2" = dY scalato per potenze di due (riga per dX, esperto x riga di W per dL/dd) e
    # diviso in due fp16, W e X gia' esatti in fp16 (2 prodotti WMMA fp16); "bf16x3" = tre prodotti bf16. Le sceglie
    # il cancello `strumenti/cancello_indietro.sh` (tempo, dL/dtheta contro B_ind32 sullo STESSO avanti).
    "B_ind2": Ricetta("B_ind2", torch.float32, torch.float16, True, None, False, False, False, "fp16x2"),
    "B3_ind2": Ricetta("B3_ind2", torch.float32, torch.float16, True, None, False, True, False, "fp16x2"),
    "B_ind3": Ricetta("B_ind3", torch.float32, torch.float16, True, None, False, False, False, "bf16x3"),
    "B3_ind3": Ricetta("B3_ind3", torch.float32, torch.float16, True, None, False, True, False, "bf16x3"),
}


@dataclass
class DatiFamiglia:
    """Una famiglia di esperti di UNO strato: byte (E, R, byte_per_riga) uint8 cosi' come sono nel GGUF
    (ordine numpy: esperto, riga d'uscita, blocchi della riga), scale iniziali d0 (E, R, ng) fp32."""
    nome: str
    tipo: str               # "TQ1_0R_1024" | "TQ1_0"
    blocchi: torch.Tensor   # uint8 (E, R, NBR)
    d0: torch.Tensor        # fp32 (E, R, NG)
    strato: int = -1        # indice dello strato (dal nome del tensore GGUF), per la diagnostica

    @property
    def E(self) -> int:
        return self.blocchi.shape[0]

    @property
    def R(self) -> int:
        return self.blocchi.shape[1]

    @property
    def NBR(self) -> int:
        return self.blocchi.shape[2]

    @property
    def nb(self) -> int:
        return CS.LAYOUT[self.tipo][0]

    @property
    def qk(self) -> int:
        return CS.LAYOUT[self.tipo][1]

    @property
    def gs(self) -> int:
        return CS.LAYOUT[self.tipo][1] // CS.LAYOUT[self.tipo][3]

    @property
    def C(self) -> int:
        return (self.NBR // self.nb) * self.qk

    @property
    def NG(self) -> int:
        return self.C // self.gs

    @property
    def ruotata(self) -> bool:
        return self.tipo == "TQ1_0R_1024"


def famiglia_da_tensore(t, nome: str, device="cpu", esperti: Optional[slice] = None,
                        righe: Optional[slice] = None) -> DatiFamiglia:
    """ReaderTensor GGUF di esperti -> DatiFamiglia. `esperti`/`righe`: sottoinsieme (solo prove).
    28/9 --densi-ternari: un tensore DENSO 2D (R, NBR) diventa la famiglia di UN esperto (1, R, NBR), stessi byte
    (vista del memmap come per gli esperti); `esperti` non si applica (c'e' un esperto solo)."""
    tipo = t.tensor_type.name
    raw = np.asarray(t.data)                                   # memmap (E, R, NBR)
    if raw.ndim == 2:
        raw = raw[None]
    elif esperti is not None:
        raw = raw[esperti]
    if righe is not None:
        raw = raw[:, righe]
    B = torch.from_numpy(np.ascontiguousarray(raw))
    nb = CS.LAYOUT[tipo][0]
    d0 = CS.scale_da_blocchi(B.reshape(-1, nb), tipo).to(torch.float32).reshape(B.shape[0], B.shape[1], -1)
    parti = str(t.name).split(".")
    L = int(parti[1]) if len(parti) > 1 and parti[0] == "blk" and parti[1].isdigit() else -1
    return DatiFamiglia(nome, tipo, B.to(device), d0.to(device), L)


# ═══════════════════════════════════ riferimento torch ═══════════════════════════════════
def trit_esperto(fam: DatiFamiglia, e: int) -> torch.Tensor:
    """Trit (R, C) int8 dell'esperto e, dai soli suoi byte (decoder device-agnostico di e2e_campi_scala)."""
    b = fam.blocchi[e].reshape(-1, fam.nb)
    return CAMPI.trit_torch(b, fam.tipo).reshape(fam.R, fam.C)


_S_CACHE: dict = {}


def _S(dtype, device) -> torch.Tensor:
    """Segni S1024 nel dtype/dispositivo chiesti, UNA volta per coppia (26/9 10:0x: prima si rifaceva
    `from_numpy(S1024).to(device)` a ogni rotazione, cioe' una copia host->GPU da memoria paginabile, sincrona,
    dentro ogni strato del ciclo)."""
    chiave = (dtype, torch.device(device))
    s = _S_CACHE.get(chiave)
    if s is None:
        s = _S_CACHE[chiave] = torch.from_numpy(S1024).to(device=device, dtype=dtype)
    return s


# Rotazione compilata (26/9 10:0x, leva di velocita' A3/«niente copie»): la FWHT eager materializza 10 tensori
# grandi quanto l'attivazione (uno per stadio) in avanti e altrettanti nell'indietro; `torch.compile` la fonde.
# Aritmetica identica (solo somme/differenze e *2^-5 esatto, nessun riassociamento): S11 lo prova AL BIT su CPU
# (inductor), `misura` lo riprova sulla GPU prima di cronometrare. None = automatico (solo su cuda);
# E2E_RUOTA_COMPILATA=0 la spegne.
COMPILA_RUOTA: Optional[bool] = None if os.getenv("E2E_RUOTA_COMPILATA", "1") == "1" else False
_ROT_COMPILATE: dict = {}


def _funzione_rotazione(nome: str, device: torch.device):
    import rotazione_gpu as RG  # sola lettura
    f = getattr(RG, nome)
    compila = COMPILA_RUOTA if COMPILA_RUOTA is not None else device.type == "cuda"
    if not compila:
        return f
    if nome not in _ROT_COMPILATE:
        _ROT_COMPILATE[nome] = torch.compile(f, dynamic=True)
    return _ROT_COMPILATE[nome]


def ruota(x: torch.Tensor) -> torch.Tensor:
    """R x = FWHT(S (.) x) / 32 a blocchi di 1024 lungo l'ultimo asse (differenziabile). Stessa aritmetica
    di `rotazione_gpu._R_1024_torch` e dello shader `rot_y_tq1_0r_1024` (S, poi H, con 1/32 in testa)."""
    return _funzione_rotazione("_R_1024_torch", x.device)(x, _S(x.dtype, x.device))


def de_ruota(w: torch.Tensor) -> torch.Tensor:
    """R^T w (righe di pesi dal dominio ruotato al reale): `rotazione_gpu._RT_1024_torch`."""
    return _funzione_rotazione("_RT_1024_torch", w.device)(w, _S(w.dtype, w.device))


def peso_esperto_torch(fam: DatiFamiglia, e: int, d: torch.Tensor, ric: Ricetta) -> torch.Tensor:
    """Peso (R, C) dell'esperto e nel dtype dei prodotti. Ricetta con `ruota`: trit x d nel dominio ruotato
    (esatto in fp16 perche' d e' fp16). Ricetta senza: R^T(trit x d) in fp32 (o fp64), poi dtype_gemm."""
    q = trit_esperto(fam, e)
    calc = torch.float64 if ric.dtype_gemm == torch.float64 else torch.float32
    W = (q.to(calc).view(fam.R, fam.NG, fam.gs) * d[e].to(calc).unsqueeze(-1)).view(fam.R, fam.C)
    if fam.ruotata and not ric.ruota:
        W = de_ruota(W)
    return W.to(ric.dtype_gemm)


# ═══════════════════════════════════ piano d'instradamento ═══════════════════════════════════
class Piano:
    """Assegnazioni (token, slot) ordinate per esperto, tutto sul dispositivo e SENZA sincronizzazioni:
    ordine (S,) = argsort stabile degli esperti, inversa (S,), offs (E+1,) da searchsorted.
    Le tabelle dei riquadri Triton (una per altezza BS) si costruiscono su richiesta, anch'esse senza sync:
    la griglia e' il limite superiore S//BS + E e i riquadri in piu' escono subito (esperto -1)."""

    def __init__(self, idx: torch.Tensor, E: int):
        self.T, self.k = idx.shape
        self.E = E
        piatto = idx.reshape(-1)
        self.ordine = torch.argsort(piatto, stable=True)
        self.inversa = torch.empty_like(self.ordine)
        self.inversa[self.ordine] = torch.arange(self.ordine.numel(), device=idx.device)
        ordinati = piatto[self.ordine]
        self.offs = torch.searchsorted(ordinati, torch.arange(E + 1, device=idx.device, dtype=ordinati.dtype))
        self.S = piatto.numel()
        self._tabelle: dict = {}
        self._conteggi: Optional[list] = None

    def conteggi(self) -> list:
        """Solo per il motore torch (CPU): una sincronizzazione, una volta per piano."""
        if self._conteggi is None:
            o = self.offs.tolist()
            self._conteggi = [o[e + 1] - o[e] for e in range(self.E)]
        return self._conteggi

    def tabella(self, BS: int) -> tuple:
        if BS not in self._tabelle:
            cnt = self.offs[1:] - self.offs[:-1]
            nt = (cnt + BS - 1) // BS
            cum = torch.cumsum(nt, 0)
            inizio = cum - nt
            n_max = self.S // BS + self.E
            t = torch.arange(n_max, device=self.offs.device)
            e = torch.searchsorted(cum, t, right=True)
            valido = e < self.E
            ec = torch.clamp(e, max=self.E - 1)
            s = self.offs[ec] + (t - inizio[ec]) * BS
            self._tabelle[BS] = (torch.where(valido, e, -1).to(torch.int32).contiguous(),
                                 torch.where(valido, s, 0).to(torch.int32).contiguous(), n_max)
        return self._tabelle[BS]


# 29/9 ramo tt3-cure (--pv-riuso, e2e_riuso_prove.RiusoProve): riuso dentro il passo V, dove theta, F32 e LoRA sono
# FERMI e cambiano solo i byte dei codici. Spento (default) = nessuna di queste strade, identico al bit.
#   piano_unico  il Piano a esperto unico dei densi (tutti i token all'esperto 0) dipende solo da (token, dispositivo):
#                uno per forma invece di uno per chiamata (250 x 2 densi x avanti e ricalcolo);
#   piano_instr  il Piano dell'avanti di uno strato MoE resta attaccato al suo top-k salvato (`ultimo_idx`): il ricalcolo
#                dell'indietro con quello stesso top-k lo riusa invece di rifarlo (stesse tabelle, stessi valori);
#   scala        la scala fp16(d0 exp(theta)) di ogni famiglia, fuori dal grafo (avanti senza gradiente), si ricalcola
#                solo se theta, d0, la sentinella o il dtype cambiano (chiave: identita' e versione dei tensori).
RIUSO = {"attivo": False, "piani_unici": {}, "scale": {}}


def riuso_attivo(on: bool) -> None:
    """Accende/spegne il riuso (e svuota le cache: nessun valore sopravvive a chi l'ha acceso)."""
    RIUSO["attivo"] = bool(on)
    RIUSO["piani_unici"].clear()
    RIUSO["scale"].clear()


def piano_unico(T: int, device) -> "Piano":
    """Piano a esperto unico per T token (riusato se il riuso e' acceso)."""
    if not RIUSO["attivo"]:
        return Piano(torch.zeros(T, 1, dtype=torch.long, device=device), 1)
    k = (int(T), str(device))
    p = RIUSO["piani_unici"].get(k)
    if p is None:
        p = RIUSO["piani_unici"][k] = Piano(torch.zeros(T, 1, dtype=torch.long, device=device), 1)
    return p


# ═══════════════════════════════════ kernel Triton ═══════════════════════════════════
if _TRITON:
    @triton.jit
    def _trit_da_byte(pb, r, c, NBR, R, C, QK: tl.constexpr, NB: tl.constexpr, ROT: tl.constexpr):
        """Trit (int32 in {-1,0,1}, 0 fuori campo) del peso (r, c) di UN esperto (`pb` punta ai suoi byte).
        Layout di ggml-common.h (k1024): TQ1_0 qs[48] qh[4] d; TQ1_0R_1024 = 4 quarti TQ1_0 (qs[4][48],
        qh[4][4]) + d[8]. Nel quarto: byte 0..31 -> elementi n*32+m, 32..47 -> 160+n*16+m, qh -> 240+n*4+m."""
        blk = c // QK
        p = c % QK
        if ROT:
            q = p // 256
            j = p % 256
            qs0 = q * 48
            qh0 = 192 + q * 4
        else:
            j = p
            qs0 = 0
            qh0 = 48
        in1 = j < 160
        in2 = j < 240
        j2 = j - 160
        j3 = j - 240
        bi = tl.where(in1, qs0 + (j % 32), tl.where(in2, qs0 + 32 + (j2 % 16), qh0 + (j3 % 4)))
        n = tl.where(in1, j // 32, tl.where(in2, j2 // 16, j3 // 4))
        pw = tl.where(n == 0, 1, tl.where(n == 1, 3, tl.where(n == 2, 9, tl.where(n == 3, 27, 81))))
        m = (r < R) & (c < C)
        v = tl.load(pb + r.to(tl.int64) * NBR + blk * NB + bi, mask=m, other=0).to(tl.int32)
        t = ((((v * pw) & 255) * 3) >> 8) - 1
        return tl.where(m, t, 0), m

    @triton.jit
    def _k_gemm(a_ptr, sa_s, sa_k,
                pb, sb_e, NBR,
                d_ptr, sd_e, NG,
                w_ptr, sw_e, sw_r,
                o_ptr, so_s, so_n,
                te_ptr, ts_ptr, offs_ptr,
                R, C, K, N,
                GS: tl.constexpr, QK: tl.constexpr, NB: tl.constexpr, ROT: tl.constexpr,
                TRASP: tl.constexpr, DENSO: tl.constexpr,
                BS: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        """Prodotto raggruppato per esperto. TRASP=0: out[s, r] = sum_c A[s, c] W_e[r, c] (avanti);
        TRASP=1: out[s, c] = sum_r A[s, r] W_e[r, c] (indietro dX). DENSO=0: W decodificato dai byte nel
        riquadro (trit x d, esatto); DENSO=1: W letto da un tensore (E, R, C). Accumulo fp32."""
        pid_t = tl.program_id(0)
        pid_n = tl.program_id(1)
        e = tl.load(te_ptr + pid_t)
        if e < 0:
            return
        s0 = tl.load(ts_ptr + pid_t)
        s_fine = tl.load(offs_ptr + e + 1)
        e64 = e.to(tl.int64)
        rs = s0 + tl.arange(0, BS)
        ms = rs < s_fine
        rn = pid_n * BN + tl.arange(0, BN)
        mn = rn < N
        acc = tl.zeros((BS, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            rk = k0 + tl.arange(0, BK)
            mk = rk < K
            a = tl.load(a_ptr + rs[:, None].to(tl.int64) * sa_s + rk[None, :] * sa_k,
                        mask=ms[:, None] & mk[None, :], other=0.0)
            if TRASP:
                r2 = rk[:, None]
                c2 = rn[None, :]
            else:
                r2 = rn[None, :]
                c2 = rk[:, None]
            if DENSO:
                b = tl.load(w_ptr + e64 * sw_e + r2.to(tl.int64) * sw_r + c2, mask=(r2 < R) & (c2 < C), other=0.0)
            else:
                t, m2 = _trit_da_byte(pb + e64 * sb_e, r2, c2, NBR, R, C, QK, NB, ROT)
                dv = tl.load(d_ptr + e64 * sd_e + r2.to(tl.int64) * NG + c2 // GS, mask=m2, other=0.0)
                b = t.to(tl.float32) * dv
            acc += tl.dot(a, b.to(a.dtype))
        tl.store(o_ptr + rs[:, None].to(tl.int64) * so_s + rn[None, :] * so_n,
                 acc.to(o_ptr.dtype.element_ty), mask=ms[:, None] & mn[None, :])

    @triton.jit
    def _k_dd(dy_ptr, sdy_s, sdy_r, x_ptr, sx_s, sx_c,
              pb, sb_e, NBR, dd_ptr, sdd_e, NG, offs_ptr, R, C,
              GS: tl.constexpr, QK: tl.constexpr, NB: tl.constexpr, ROT: tl.constexpr,
              BR: tl.constexpr, BS: tl.constexpr):
        """dL/dd[e, r, g] = sum_{c in g} trit[e, r, c] * sum_{s in e} dY[s, r] X[s, c], fp32. Il blocco
        (BR x GS) di dY^T X non esce mai dai registri: i trit si decodificano dai byte alla fine."""
        e = tl.program_id(0)
        pid_r = tl.program_id(1)
        g = tl.program_id(2)
        s_ini = tl.load(offs_ptr + e)
        s_fine = tl.load(offs_ptr + e + 1)
        rr = pid_r * BR + tl.arange(0, BR)
        cc = g * GS + tl.arange(0, GS)
        mr = rr < R
        acc = tl.zeros((BR, GS), dtype=tl.float32)
        for s0 in range(s_ini, s_fine, BS):
            rs = s0 + tl.arange(0, BS)
            ms = rs < s_fine
            dyt = tl.load(dy_ptr + rs[None, :].to(tl.int64) * sdy_s + rr[:, None] * sdy_r,
                          mask=ms[None, :] & mr[:, None], other=0.0)
            x = tl.load(x_ptr + rs[:, None].to(tl.int64) * sx_s + cc[None, :] * sx_c,
                        mask=ms[:, None], other=0.0)
            acc += tl.dot(dyt, x)
        e64 = e.to(tl.int64)
        t, m2 = _trit_da_byte(pb + e64 * sb_e, rr[:, None], cc[None, :], NBR, R, C, QK, NB, ROT)
        val = tl.sum(acc * t.to(tl.float32), axis=1)
        tl.store(dd_ptr + e64 * sdd_e + rr * NG + g, val, mask=mr)

    @triton.jit
    def _k_decodifica(pb, sb_e, NBR, d_ptr, sd_e, NG, w_ptr, sw_e, sw_r, R, C,
                      GS: tl.constexpr, QK: tl.constexpr, NB: tl.constexpr, ROT: tl.constexpr,
                      BR: tl.constexpr, BC: tl.constexpr):
        """byte + scale -> W (E, R, C) = trit x d nel dtype di w_ptr, un solo passaggio (decodifica fusa)."""
        e64 = tl.program_id(0).to(tl.int64)
        rr = tl.program_id(1) * BR + tl.arange(0, BR)
        cc = tl.program_id(2) * BC + tl.arange(0, BC)
        r2 = rr[:, None]
        c2 = cc[None, :]
        t, m2 = _trit_da_byte(pb + e64 * sb_e, r2, c2, NBR, R, C, QK, NB, ROT)
        dv = tl.load(d_ptr + e64 * sd_e + r2.to(tl.int64) * NG + c2 // GS, mask=m2, other=0.0)
        w = t.to(tl.float32) * dv
        tl.store(w_ptr + e64 * sw_e + r2.to(tl.int64) * sw_r + c2, w.to(w_ptr.dtype.element_ty), mask=m2)

    # ─────────────── motore «triton2» (27/9): stessa matematica, decodifica a pezzi ───────────────
    # Un sottoblocco da 256 colonne (un blocco TQ1_0, o un quarto di TQ1_0R_1024) si divide in 8 PEZZI da 32
    # colonne, con lo schema FISSO di ggml-common.h:
    #   pezzi 0..4: colonne 32p..32p+31 = byte qs[0:32] (contigui) col moltiplicatore 3^p (uno per pezzo);
    #   pezzi 5, 6: colonne 160..223     = byte qs[32:48] (16 contigui, ciascuno due volte), 3^0|3^1 e 3^2|3^3;
    #   pezzo 7   : colonne 224..239 = qs[32:48] x 3^4, 240..255 = qh[m] x 3^n (colonna 240 + 4n + m).
    # In _k_dx2 e _k_dd2 (pezzi lungo le colonne del programma) i 32 byte di qs[0:32] si leggono UNA volta per
    # 5 pezzi (carico contiguo), il moltiplicatore e' una costante o un vettore fisso di 32 e la scala un vettore
    # per riga caricato una volta per gruppo (GS = 128 o 256 contiene 4 o 8 pezzi interi). In _k_gemm2 (pezzi
    # lungo K) gli 8 pezzi srotolati riversavano 150-600 registri su gfx1151: li' il ciclo e' a passi di 32 con
    # lo schema del pezzo calcolato su vettori 1D di 32 (byte e moltiplicatore per colonna, scala per riga una
    # volta per pezzo): per peso restano 1 carico + 5 operazioni intere + conversione e prodotto, senza le
    # divisioni, le catene di where e gli indirizzi int64 per elemento di `_trit_da_byte` (114-165 VGPR, 0 spill).
    # Niente tabella byte -> 5 trit: sulla GPU un carico indicizzato per peso costa piu' di 4 operazioni intere.
    # Il valore del peso e' lo stesso di `_trit_da_byte` x d (t in {-1, 0, 1}: fp16(t d) = t fp16(d), anche
    # nel segno dello zero) e l'ordine dei prodotti lungo K e' quello di `_k_gemm` con BK = 32.
    @triton.jit
    def _dec_trit(v, pw):
        """((((b * 3^n) & 255) * 3) >> 8) - 1 (dequantize_row_tq1_0), v int32, pw scalare o tensore."""
        return ((((v * pw) & 255) * 3) >> 8) - 1

    @triton.jit
    def _lungo_col(v, TR: tl.constexpr):
        """Vettore sulle 32 colonne del pezzo -> forma del riquadro (TR=1: colonne sull'asse 0)."""
        if TR:
            r = v[:, None]
        else:
            r = v[None, :]
        return r

    @triton.jit
    def _lungo_riga(v, TR: tl.constexpr):
        """Vettore sulle righe del peso -> forma del riquadro (TR=1: righe sull'asse 1)."""
        if TR:
            r = v[None, :]
        else:
            r = v[:, None]
        return r

    @triton.jit
    def _t_pezzo(v0, v5, v7, pw5, pw6, pw7, P: tl.constexpr, TR: tl.constexpr):
        """Trit int32 del pezzo P da byte gia' caricati (v0: qs[0:32]; v5: qs[32 + i%16]; v7: pezzo 7)."""
        if P == 0:
            t = _dec_trit(v0, 1)
        elif P == 1:
            t = _dec_trit(v0, 3)
        elif P == 2:
            t = _dec_trit(v0, 9)
        elif P == 3:
            t = _dec_trit(v0, 27)
        elif P == 4:
            t = _dec_trit(v0, 81)
        elif P == 5:
            t = _dec_trit(v5, _lungo_col(pw5, TR))
        elif P == 6:
            t = _dec_trit(v5, _lungo_col(pw6, TR))
        else:
            t = _dec_trit(v7, _lungo_col(pw7, TR))
        return t

    @triton.jit
    def _w_pezzo(v0, v5, v7, dA, dB, pw5, pw6, pw7, P: tl.constexpr, GS: tl.constexpr, TR: tl.constexpr,
                 DT: tl.constexpr):
        """Peso trit x d del pezzo P nel dtype DT. dA: scala del gruppo dei pezzi 0..3 (e 4..7 se GS = 256),
        dB: del gruppo dei pezzi 4..7 se GS = 128. Gia' in DT (conversione esatta, vedi sopra)."""
        t = _t_pezzo(v0, v5, v7, pw5, pw6, pw7, P, TR)
        if GS == 128:
            if P >= 4:
                dv = dB
            else:
                dv = dA
        else:
            dv = dA
        return t.to(DT) * _lungo_riga(dv, TR)

    @triton.jit
    def _schema_sottoblocco(c0, QK: tl.constexpr, NB: tl.constexpr, ROT: tl.constexpr):
        """Offset (byte, dentro la riga) di qs e qh del sottoblocco da 256 che parte alla colonna c0."""
        b0 = (c0 // QK) * NB
        if ROT:
            q = (c0 % QK) // 256
            qs = b0 + q * 48
            qh = b0 + 192 + q * 4
        else:
            qs = b0
            qh = b0 + 48
        return qs, qh

    @triton.jit
    def _ordine_gruppi(pid, NT, NN, GROUP_M: tl.constexpr):
        """pid -> (riquadro di token, riquadro d'uscita): GROUP_M riquadri di token consecutivi (in gran parte
        dello STESSO esperto) percorrono insieme i riquadri d'uscita, cosi' byte, scale e righe di A restano
        nella L2. Solo l'ordine di esecuzione cambia: ogni riquadro fa gli stessi conti."""
        per_g = GROUP_M * NN
        t_ini = (pid // per_g) * GROUP_M
        gm = tl.minimum(NT - t_ini, GROUP_M)
        loc = pid % per_g
        return t_ini + loc % gm, loc // gm

    @triton.jit
    def _k_gemm2(a_ptr, sa_s,
                 pb, sb_e, NBR, d_ptr, sd_e, NG, o_ptr, so_s,
                 pb2, sb2_e, d2_ptr, sd2_e, o2_ptr,
                 te_ptr, ts_ptr, offs_ptr, R, C, NT,
                 GS: tl.constexpr, QK: tl.constexpr, NB: tl.constexpr, ROT: tl.constexpr, DUE: tl.constexpr,
                 BS: tl.constexpr, BN: tl.constexpr, GROUP_M: tl.constexpr):
        """Avanti: out[s, r] = sum_c A[s, c] W_e[r, c], come `_k_gemm` TRASP=0 (un prodotto per passo di 32
        colonne, nello stesso ordine di BK = 32). DUE=1: stessa passata per una seconda famiglia della stessa forma
        (gate e up) sullo STESSO riquadro di A: due accumulatori, A letto una volta. Indirizzi = base scalare
        (int64) + scostamenti int32 del riquadro, invarianti nel ciclo: pochi registri vettoriali."""
        DT = a_ptr.dtype.element_ty
        pid_t, pid_n = _ordine_gruppi(tl.program_id(0), NT, tl.cdiv(R, BN), GROUP_M)
        e = tl.load(te_ptr + pid_t)
        if e < 0:
            return
        s0 = tl.load(ts_ptr + pid_t)
        s_fine = tl.load(offs_ptr + e + 1)
        e64 = e.to(tl.int64)
        ls = tl.arange(0, BS)
        ms = (s0 + ls) < s_fine
        ln = tl.arange(0, BN)
        r0 = pid_n * BN
        mn = (r0 + ln) < R
        i = tl.arange(0, 32)
        pa = a_ptr + s0.to(tl.int64) * sa_s
        offa = (ls * sa_s)[:, None] + i[None, :]
        pbw = pb + e64 * sb_e + r0.to(tl.int64) * NBR
        pbw2 = pb2 + e64 * sb2_e + r0.to(tl.int64) * NBR
        pdd = d_ptr + e64 * sd_e + r0.to(tl.int64) * NG
        pdd2 = d2_ptr + e64 * sd2_e + r0.to(tl.int64) * NG
        offr = (ln * NBR)[None, :]
        offd = ln * NG
        m2 = mn[None, :]
        acc = tl.zeros((BS, BN), dtype=tl.float32)
        acc2 = tl.zeros((BS, BN), dtype=tl.float32)
        for k in range(0, C, 32):
            # schema del pezzo (k % 256) // 32 su 32 colonne: vettori 1D (costo trascurabile rispetto al riquadro)
            qs, qh = _schema_sottoblocco(k, QK, NB, ROT)
            j = (k % 256) + i
            n1 = j // 32
            n2 = (j - 160) // 16
            n3 = (j - 240) // 4
            n = tl.where(j < 160, n1, tl.where(j < 240, n2, n3))
            pw = tl.where(n == 0, 1, tl.where(n == 1, 3, tl.where(n == 2, 9, tl.where(n == 3, 27, 81))))
            ob = tl.where(j < 160, qs + i, tl.where(j < 240, qs + 32 + (i % 16), qh + (i % 4)))
            offb = offr + ob[:, None]
            dv = tl.load(pdd + k // GS + offd, mask=mn, other=0.0).to(DT)
            v = tl.load(pbw + offb, mask=m2, other=0).to(tl.int32)
            a = tl.load(pa + k + offa, mask=ms[:, None], other=0.0)
            acc += tl.dot(a, _dec_trit(v, pw[:, None]).to(DT) * dv[None, :])
            if DUE:
                ev = tl.load(pdd2 + k // GS + offd, mask=mn, other=0.0).to(DT)
                u = tl.load(pbw2 + offb, mask=m2, other=0).to(tl.int32)
                acc2 += tl.dot(a, _dec_trit(u, pw[:, None]).to(DT) * ev[None, :])
        offo = (ls * so_s)[:, None] + ln[None, :]
        mo = ms[:, None] & mn[None, :]
        po = s0.to(tl.int64) * so_s + r0
        tl.store(o_ptr + po + offo, acc.to(o_ptr.dtype.element_ty), mask=mo)
        if DUE:
            tl.store(o2_ptr + po + offo, acc2.to(o2_ptr.dtype.element_ty), mask=mo)

    @triton.jit
    def _k_dx2(a_ptr, sa_s, pb, sb_e, NBR, d_ptr, sd_e, NG, o_ptr, so_s,
               te_ptr, ts_ptr, offs_ptr, R, C, NT,
               GS: tl.constexpr, QK: tl.constexpr, NB: tl.constexpr, ROT: tl.constexpr,
               BS: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr):
        """Indietro dX: out[s, c] = sum_r A[s, r] W_e[r, c], come `_k_gemm` TRASP=1 (stesso ordine lungo r a
        passi di BK). Un programma = riquadro di token x UN sottoblocco da 256 colonne (8 accumulatori da 32):
        A si rilegge C/256 volte invece di C/BN; gli scostamenti dei byte e delle scale sono invarianti nel
        ciclo (cambia solo la base scalare della riga)."""
        DT = a_ptr.dtype.element_ty
        pid_t, pid_n = _ordine_gruppi(tl.program_id(0), NT, C // 256, GROUP_M)
        e = tl.load(te_ptr + pid_t)
        if e < 0:
            return
        s0 = tl.load(ts_ptr + pid_t)
        s_fine = tl.load(offs_ptr + e + 1)
        e64 = e.to(tl.int64)
        ls = tl.arange(0, BS)
        ms = (s0 + ls) < s_fine
        lk = tl.arange(0, BK)
        c0 = pid_n * 256
        qs, qh = _schema_sottoblocco(c0, QK, NB, ROT)
        i = tl.arange(0, 32)
        offr = (lk * NBR)[:, None]
        off0 = offr + (qs + i)[None, :]
        off5 = offr + (qs + 32 + (i % 16))[None, :]
        off7 = offr + tl.where(i < 16, qs + 32 + i, qh + (i % 4))[None, :]
        offd = lk * NG
        offa = (ls * sa_s)[:, None] + lk[None, :]
        pw5 = tl.where(i < 16, 1, 3)
        pw6 = tl.where(i < 16, 9, 27)
        pw7 = tl.where(i < 16, 81, tl.where(i < 20, 1, tl.where(i < 24, 3, tl.where(i < 28, 9, 27))))
        g = c0 // GS
        pe = pb + e64 * sb_e
        pde = d_ptr + e64 * sd_e + g
        pa = a_ptr + s0.to(tl.int64) * sa_s
        acc0 = tl.zeros((BS, 32), dtype=tl.float32)
        acc1 = tl.zeros((BS, 32), dtype=tl.float32)
        acc2 = tl.zeros((BS, 32), dtype=tl.float32)
        acc3 = tl.zeros((BS, 32), dtype=tl.float32)
        acc4 = tl.zeros((BS, 32), dtype=tl.float32)
        acc5 = tl.zeros((BS, 32), dtype=tl.float32)
        acc6 = tl.zeros((BS, 32), dtype=tl.float32)
        acc7 = tl.zeros((BS, 32), dtype=tl.float32)
        for r0 in range(0, R, BK):
            mr = (r0 + lk) < R
            prow = pe + r0 * NBR
            m2 = mr[:, None]
            v0 = tl.load(prow + off0, mask=m2, other=0).to(tl.int32)
            v5 = tl.load(prow + off5, mask=m2, other=0).to(tl.int32)
            v7 = tl.load(prow + off7, mask=m2, other=0).to(tl.int32)
            pdr = pde + r0 * NG
            dA = tl.load(pdr + offd, mask=mr, other=0.0).to(DT)
            if GS == 128:
                dB = tl.load(pdr + 1 + offd, mask=mr, other=0.0).to(DT)
            else:
                dB = dA
            a = tl.load(pa + r0 + offa, mask=ms[:, None] & mr[None, :], other=0.0)
            acc0 += tl.dot(a, _w_pezzo(v0, v5, v7, dA, dB, pw5, pw6, pw7, 0, GS, 0, DT))
            acc1 += tl.dot(a, _w_pezzo(v0, v5, v7, dA, dB, pw5, pw6, pw7, 1, GS, 0, DT))
            acc2 += tl.dot(a, _w_pezzo(v0, v5, v7, dA, dB, pw5, pw6, pw7, 2, GS, 0, DT))
            acc3 += tl.dot(a, _w_pezzo(v0, v5, v7, dA, dB, pw5, pw6, pw7, 3, GS, 0, DT))
            acc4 += tl.dot(a, _w_pezzo(v0, v5, v7, dA, dB, pw5, pw6, pw7, 4, GS, 0, DT))
            acc5 += tl.dot(a, _w_pezzo(v0, v5, v7, dA, dB, pw5, pw6, pw7, 5, GS, 0, DT))
            acc6 += tl.dot(a, _w_pezzo(v0, v5, v7, dA, dB, pw5, pw6, pw7, 6, GS, 0, DT))
            acc7 += tl.dot(a, _w_pezzo(v0, v5, v7, dA, dB, pw5, pw6, pw7, 7, GS, 0, DT))
        OT = o_ptr.dtype.element_ty
        po = o_ptr + s0.to(tl.int64) * so_s + c0
        offo = (ls * so_s)[:, None] + i[None, :]
        mo = ms[:, None]
        tl.store(po + offo, acc0.to(OT), mask=mo)
        tl.store(po + 32 + offo, acc1.to(OT), mask=mo)
        tl.store(po + 64 + offo, acc2.to(OT), mask=mo)
        tl.store(po + 96 + offo, acc3.to(OT), mask=mo)
        tl.store(po + 128 + offo, acc4.to(OT), mask=mo)
        tl.store(po + 160 + offo, acc5.to(OT), mask=mo)
        tl.store(po + 192 + offo, acc6.to(OT), mask=mo)
        tl.store(po + 224 + offo, acc7.to(OT), mask=mo)

    @triton.jit
    def _k_dd2(dy_ptr, sdy_s, x_ptr, sx_s, pb, sb_e, NBR, dd_ptr, sdd_e, NG, offs_ptr, R, C,
               GS: tl.constexpr, QK: tl.constexpr, NB: tl.constexpr, ROT: tl.constexpr,
               BR: tl.constexpr, BS: tl.constexpr):
        """dL/dd come GEMM a K variabile (K = token dell'esperto): programma = (esperto, BR righe, UN
        sottoblocco da 256 colonne); acc_p (BR x 32) = dY^T X per gli 8 pezzi, riquadro sui token BS; dY
        letto per righe contigue (BS x BR, poi trasposto nei registri) una volta per sottoblocco (C/256 volte
        invece di NG); trit (pezzi) e somme per gruppo SOLO nell'epilogo. Rispetto a `_k_dd` cambia solo
        l'ordine delle somme fp32 (passo sui token e riduzione del gruppo)."""
        e = tl.program_id(0)
        pid_r = tl.program_id(1)
        c0 = tl.program_id(2) * 256
        s_ini = tl.load(offs_ptr + e)
        s_fine = tl.load(offs_ptr + e + 1)
        r0 = pid_r * BR
        lr = tl.arange(0, BR)
        mr = (r0 + lr) < R
        ls = tl.arange(0, BS)
        i = tl.arange(0, 32)
        offy = (ls * sdy_s)[:, None] + lr[None, :]
        offx = (ls * sx_s)[:, None] + i[None, :]
        pdy = dy_ptr + r0
        px = x_ptr + c0
        acc0 = tl.zeros((BR, 32), dtype=tl.float32)
        acc1 = tl.zeros((BR, 32), dtype=tl.float32)
        acc2 = tl.zeros((BR, 32), dtype=tl.float32)
        acc3 = tl.zeros((BR, 32), dtype=tl.float32)
        acc4 = tl.zeros((BR, 32), dtype=tl.float32)
        acc5 = tl.zeros((BR, 32), dtype=tl.float32)
        acc6 = tl.zeros((BR, 32), dtype=tl.float32)
        acc7 = tl.zeros((BR, 32), dtype=tl.float32)
        for s0 in range(s_ini, s_fine, BS):
            ms = (s0 + ls) < s_fine
            s64 = s_ini.to(tl.int64) + (s0 - s_ini)       # int64 anche nell'interprete
            dy = tl.load(pdy + s64 * sdy_s + offy, mask=ms[:, None] & mr[None, :], other=0.0)
            dyt = tl.trans(dy)
            pxs = px + s64 * sx_s + offx
            mx = ms[:, None]
            acc0 += tl.dot(dyt, tl.load(pxs, mask=mx, other=0.0))
            acc1 += tl.dot(dyt, tl.load(pxs + 32, mask=mx, other=0.0))
            acc2 += tl.dot(dyt, tl.load(pxs + 64, mask=mx, other=0.0))
            acc3 += tl.dot(dyt, tl.load(pxs + 96, mask=mx, other=0.0))
            acc4 += tl.dot(dyt, tl.load(pxs + 128, mask=mx, other=0.0))
            acc5 += tl.dot(dyt, tl.load(pxs + 160, mask=mx, other=0.0))
            acc6 += tl.dot(dyt, tl.load(pxs + 192, mask=mx, other=0.0))
            acc7 += tl.dot(dyt, tl.load(pxs + 224, mask=mx, other=0.0))
        e64 = e.to(tl.int64)
        qs, qh = _schema_sottoblocco(c0, QK, NB, ROT)
        pw5 = tl.where(i < 16, 1, 3)
        pw6 = tl.where(i < 16, 9, 27)
        pw7 = tl.where(i < 16, 81, tl.where(i < 20, 1, tl.where(i < 24, 3, tl.where(i < 28, 9, 27))))
        prow = pb + e64 * sb_e + r0.to(tl.int64) * NBR
        offr = (lr * NBR)[:, None]
        m2 = mr[:, None]
        v0 = tl.load(prow + offr + (qs + i)[None, :], mask=m2, other=0).to(tl.int32)
        v5 = tl.load(prow + offr + (qs + 32 + (i % 16))[None, :], mask=m2, other=0).to(tl.int32)
        v7 = tl.load(prow + offr + tl.where(i < 16, qs + 32 + i, qh + (i % 4))[None, :], mask=m2,
                     other=0).to(tl.int32)
        p0 = acc0 * _t_pezzo(v0, v5, v7, pw5, pw6, pw7, 0, 0).to(tl.float32)
        p0 += acc1 * _t_pezzo(v0, v5, v7, pw5, pw6, pw7, 1, 0).to(tl.float32)
        p0 += acc2 * _t_pezzo(v0, v5, v7, pw5, pw6, pw7, 2, 0).to(tl.float32)
        p0 += acc3 * _t_pezzo(v0, v5, v7, pw5, pw6, pw7, 3, 0).to(tl.float32)
        p1 = acc4 * _t_pezzo(v0, v5, v7, pw5, pw6, pw7, 4, 0).to(tl.float32)
        p1 += acc5 * _t_pezzo(v0, v5, v7, pw5, pw6, pw7, 5, 0).to(tl.float32)
        p1 += acc6 * _t_pezzo(v0, v5, v7, pw5, pw6, pw7, 6, 0).to(tl.float32)
        p1 += acc7 * _t_pezzo(v0, v5, v7, pw5, pw6, pw7, 7, 0).to(tl.float32)
        g = c0 // GS
        pdd = dd_ptr + e64 * sdd_e + r0.to(tl.int64) * NG + g + lr * NG
        if GS == 128:
            tl.store(pdd, tl.sum(p0, axis=1), mask=mr)
            tl.store(pdd + 1, tl.sum(p1, axis=1), mask=mr)
        else:
            tl.store(pdd, tl.sum(p0 + p1, axis=1), mask=mr)

    # 26/9 10:5x: combinazione degli esperti FUSA (vedi CombinaEsperti). Ordine della somma j = 0..K-1 come
    # `build_moe_ffn` di llama.cpp (moe_out = e0 + e1 + ...); deterministici, nessuna atomica.
    @triton.jit
    def _k_combina(o_ptr, so_s, w_ptr, inv_ptr, y_ptr, sy_t, H, K: tl.constexpr, BH: tl.constexpr):
        """y[t, h] = sum_j w[t, j] * o[inversa[t*K + j], h], fp32."""
        t64 = tl.program_id(0).to(tl.int64)
        h = tl.program_id(1) * BH + tl.arange(0, BH)
        m = h < H
        acc = tl.zeros((BH,), dtype=tl.float32)
        for j in tl.static_range(K):
            r = tl.load(inv_ptr + t64 * K + j).to(tl.int64)
            wj = tl.load(w_ptr + t64 * K + j)
            acc += tl.load(o_ptr + r * so_s + h, mask=m, other=0.0) * wj
        tl.store(y_ptr + t64 * sy_t + h, acc, mask=m)

    @triton.jit
    def _k_combina_do(g_ptr, sg_t, w_ptr, ord_ptr, do_ptr, sdo_s, H, K: tl.constexpr, BH: tl.constexpr):
        """dL/do[s, h] = w[q] * g[q // K, h], q = ordine[s] (riga s dell'ordine per esperto)."""
        s64 = tl.program_id(0).to(tl.int64)
        h = tl.program_id(1) * BH + tl.arange(0, BH)
        m = h < H
        q = tl.load(ord_ptr + s64).to(tl.int64)
        wv = tl.load(w_ptr + q)
        g = tl.load(g_ptr + (q // K) * sg_t + h, mask=m, other=0.0)
        tl.store(do_ptr + s64 * sdo_s + h, g * wv, mask=m)

    @triton.jit
    def _k_combina_dw(g_ptr, sg_t, o_ptr, so_s, inv_ptr, dw_ptr, H, K: tl.constexpr, BH: tl.constexpr):
        """dL/dw[q] = sum_h g[q // K, h] * o[inversa[q], h], fp32."""
        q64 = tl.program_id(0).to(tl.int64)
        r = tl.load(inv_ptr + q64).to(tl.int64)
        t = q64 // K
        acc = tl.zeros((BH,), dtype=tl.float32)
        for h0 in range(0, H, BH):
            h = h0 + tl.arange(0, BH)
            m = h < H
            acc += tl.load(g_ptr + t * sg_t + h, mask=m, other=0.0) * tl.load(o_ptr + r * so_s + h, mask=m, other=0.0)
        tl.store(dw_ptr + q64, tl.sum(acc, axis=0))


@dataclass
class Blocchi:
    """Taglie dei riquadri Triton. Valori di partenza scelti SENZA GPU (26/9): compilazione anticipata per
    gfx1151 (`triton.compile`, GPUTarget("hip", "gfx1151", 32)) e conteggio di v_wmma e dei registri
    riversati in memoria (spill). Con num_warps 4 e BS=BN=64, BK=32 il prodotto fuso riversava 34 registri,
    e il dd con GS=256 fino a 139: questi valori danno 0 spill su tutte le varianti. Il tempo lo misura M0
    (`--misura --blocchi '{...}'`)."""
    BS: int = 64
    BN: int = 64
    BK: int = 32
    BR_dd: int = 64
    BS_dd: int = 16
    BR_dec: int = 32
    BC_dec: int = 128
    num_warps: int = 8
    # motore «triton2» (27/9): riquadri dei kernel _k_gemm2 / _k_dx2 / _k_dd2. Valori scelti con la stessa
    # compilazione anticipata per gfx1151 (e2e_test_triton2 K6): 0 registri riversati con ingressi fp16 in tutte
    # e tre (avanti anche con gate+up insieme); il tempo lo decide strumenti/cancello_kernel.sh (--spazza).
    BS2: int = 64          # token per riquadro dell'avanti (anche gate+up insieme)
    BN2: int = 64          # righe d'uscita per riquadro dell'avanti
    GROUP_M: int = 8       # riquadri di token per gruppo (ordine per la L2), avanti e dX
    BS_dx2: int = 32       # token per riquadro del dX (x 256 colonne: 8 accumulatori da 32)
    BK_dx2: int = 32       # passo lungo R del dX (32 = stesso ordine delle somme di _k_gemm)
    BR_dd2: int = 64       # righe per riquadro di dL/dd (x 256 colonne)
    BS_dd2: int = 32       # token per passo di dL/dd
    num_warps2: int = 8
    stadi_gemm2: int = 0   # num_stages dell'avanti (0 = default di Triton; 1 = senza prefetch, meno registri)
    stadi_dx2: int = 1     # dX e dL/dd: con il prefetch (default) 8 pezzi srotolati riversano registri
    stadi_dd2: int = 1
    gate_up: bool = True   # gate e up nella stessa passata dell'avanti (stesso ingresso)


def decodifica_pesi(fam: DatiFamiglia, d: torch.Tensor, dtype: torch.dtype, blk: Blocchi = Blocchi()) -> torch.Tensor:
    """Kernel fuso byte + d -> W (E, R, C) = trit x d nel dominio dei byte (ruotato per TQ1_0R_1024)."""
    E, R, C = fam.E, fam.R, fam.C
    W = torch.empty(E, R, C, dtype=dtype, device=fam.blocchi.device)
    d = d.contiguous()
    griglia = (E, triton.cdiv(R, blk.BR_dec), triton.cdiv(C, blk.BC_dec))
    _k_decodifica[griglia](fam.blocchi, fam.blocchi.stride(0), fam.NBR, d, d.stride(0), fam.NG,
                           W, W.stride(0), W.stride(1), R, C, fam.gs, fam.qk, fam.nb, int(fam.ruotata),
                           blk.BR_dec, blk.BC_dec, num_warps=blk.num_warps)
    return W


def _gemm_triton(A: torch.Tensor, fam: DatiFamiglia, d: Optional[torch.Tensor], W: Optional[torch.Tensor],
                 piano: Piano, trasp: bool, out_dtype: torch.dtype, blk: Blocchi) -> torch.Tensor:
    A = A.contiguous()
    K = A.shape[1]
    N = fam.C if trasp else fam.R
    assert K == (fam.R if trasp else fam.C), (K, fam.R, fam.C, trasp)
    out = torch.empty(A.shape[0], N, dtype=out_dtype, device=A.device)
    te, ts, n_max = piano.tabella(blk.BS)
    denso = W is not None
    dd_ = d.contiguous() if d is not None else A      # segnaposto non letto quando DENSO
    Wd = W if denso else A
    griglia = (n_max, triton.cdiv(N, blk.BN))
    _k_gemm[griglia](A, A.stride(0), A.stride(1),
                     fam.blocchi, fam.blocchi.stride(0), fam.NBR,
                     dd_, dd_.stride(0) if not denso else 0, fam.NG,
                     Wd, Wd.stride(0) if denso else 0, Wd.stride(1) if denso else 0,
                     out, out.stride(0), out.stride(1),
                     te, ts, piano.offs,
                     fam.R, fam.C, K, N,
                     fam.gs, fam.qk, fam.nb, int(fam.ruotata), int(trasp), int(denso),
                     blk.BS, blk.BN, blk.BK, num_warps=blk.num_warps)
    return out


def _dd_triton(dY: torch.Tensor, X: torch.Tensor, fam: DatiFamiglia, piano: Piano, blk: Blocchi) -> torch.Tensor:
    dY, X = dY.contiguous(), X.contiguous()
    assert dY.dtype == X.dtype
    dd = torch.empty(fam.E, fam.R, fam.NG, dtype=torch.float32, device=dY.device)
    griglia = (fam.E, triton.cdiv(fam.R, blk.BR_dd), fam.NG)
    _k_dd[griglia](dY, dY.stride(0), dY.stride(1), X, X.stride(0), X.stride(1),
                   fam.blocchi, fam.blocchi.stride(0), fam.NBR, dd, dd.stride(0), fam.NG, piano.offs,
                   fam.R, fam.C, fam.gs, fam.qk, fam.nb, int(fam.ruotata), blk.BR_dd, blk.BS_dd,
                   num_warps=blk.num_warps)
    return dd


def _stadi(n: int) -> dict:
    return {"num_stages": n} if n > 0 else {}


def _controlla_v2(fam: DatiFamiglia) -> None:
    assert fam.C % 256 == 0 and fam.gs in (128, 256) and fam.qk % 256 == 0, (fam.tipo, fam.C, fam.gs)


def _avanti2_triton(A: torch.Tensor, fams: list, ds: list, piano: Piano, out_dtype: torch.dtype,
                    blk: Blocchi) -> list:
    """Avanti del motore triton2 per UNA famiglia o per DUE della stessa forma (gate e up, stesso A)."""
    A = A.contiguous()
    f = fams[0]
    _controlla_v2(f)
    assert A.shape[1] == f.C and len(fams) in (1, 2)
    for g in fams[1:]:
        assert (g.tipo, g.E, g.R, g.NBR) == (f.tipo, f.E, f.R, f.NBR)
    ds = [d.contiguous() for d in ds]
    import e2e_kernel_qat as _KQ               # 9/10 E2E_KQAT=caldi: piani 2/3 solo sugli esperti che li hanno
    blk = _KQ.blk_avanti(blk)                  # 9/10 E2E_KQAT=tile: riquadri misurati (stessi valori al bit)
    te, ts, n_t, salta = _KQ.tabella(piano, blk.BS2, fams)
    nuovo = torch.zeros if salta else torch.empty
    outs = [nuovo(A.shape[0], f.R, dtype=out_dtype, device=A.device) for _ in fams]
    f2, d2, o2 = fams[-1], ds[-1], outs[-1]
    griglia = (n_t * triton.cdiv(f.R, blk.BN2),)
    _k_gemm2[griglia](A, A.stride(0),
                      f.blocchi, f.blocchi.stride(0), f.NBR, ds[0], ds[0].stride(0), f.NG, outs[0], outs[0].stride(0),
                      f2.blocchi, f2.blocchi.stride(0), d2, d2.stride(0), o2,
                      te, ts, piano.offs, f.R, f.C, n_t,
                      f.gs, f.qk, f.nb, int(f.ruotata), int(len(fams) == 2),
                      blk.BS2, blk.BN2, blk.GROUP_M, num_warps=blk.num_warps2, **_stadi(blk.stadi_gemm2))
    return outs


def _dx2_triton(gy: torch.Tensor, fam: DatiFamiglia, d: torch.Tensor, piano: Piano, out_dtype: torch.dtype,
                blk: Blocchi) -> torch.Tensor:
    gy, d = gy.contiguous(), d.contiguous()
    _controlla_v2(fam)
    assert gy.shape[1] == fam.R
    import e2e_kernel_qat as _KQ               # 9/10 E2E_KQAT=caldi (vedi _avanti2_triton)
    te, ts, n_t, salta = _KQ.tabella(piano, blk.BS_dx2, [fam])
    out = (torch.zeros if salta else torch.empty)(gy.shape[0], fam.C, dtype=out_dtype, device=gy.device)
    _k_dx2[(n_t * (fam.C // 256),)](gy, gy.stride(0), fam.blocchi, fam.blocchi.stride(0), fam.NBR,
                                    d, d.stride(0), fam.NG, out, out.stride(0), te, ts, piano.offs,
                                    fam.R, fam.C, n_t, fam.gs, fam.qk, fam.nb, int(fam.ruotata),
                                    blk.BS_dx2, blk.BK_dx2, blk.GROUP_M, num_warps=blk.num_warps2,
                                    **_stadi(blk.stadi_dx2))
    return out


def _dd2_triton(dY: torch.Tensor, X: torch.Tensor, fam: DatiFamiglia, piano: Piano, blk: Blocchi) -> torch.Tensor:
    dY, X = dY.contiguous(), X.contiguous()
    assert dY.dtype == X.dtype
    _controlla_v2(fam)
    dd = torch.empty(fam.E, fam.R, fam.NG, dtype=torch.float32, device=dY.device)
    griglia = (fam.E, triton.cdiv(fam.R, blk.BR_dd2), fam.C // 256)
    _k_dd2[griglia](dY, dY.stride(0), X, X.stride(0), fam.blocchi, fam.blocchi.stride(0), fam.NBR,
                    dd, dd.stride(0), fam.NG, piano.offs, fam.R, fam.C,
                    fam.gs, fam.qk, fam.nb, int(fam.ruotata), blk.BR_dd2, blk.BS_dd2, num_warps=blk.num_warps2,
                    **_stadi(blk.stadi_dd2))
    return dd


# ═══════════════════════════════════ motori ═══════════════════════════════════
class MotoreTorch:
    """Stessa matematica dei kernel, esperto per esperto (transitorio = UN esperto). CPU e prove."""
    nome = "torch"

    def pesi(self, fam, d, ric):
        return None

    def avanti(self, x, d, fam, piano, ric, pesi=None):
        """Con dtype_gemm fp16 il prodotto si fa in fp32 sui valori fp16 (esatti): e' cio' che fa un
        accumulo fp32 su ingressi fp16, salvo l'ordine delle somme."""
        out = torch.empty(x.shape[0], fam.R, dtype=ric.dtype_uscita, device=x.device)
        o = piano.offs.tolist()
        for e in range(fam.E):
            if o[e + 1] > o[e]:
                W = peso_esperto_torch(fam, e, d, ric)
                xs = x[o[e]:o[e + 1]]
                if ric.dtype_gemm == torch.float16:
                    out[o[e]:o[e + 1]] = torch.matmul(xs.float(), W.float().T)
                else:
                    out[o[e]:o[e + 1]] = torch.matmul(xs, W.T).to(out.dtype)
        return out

    def indietro_x(self, gy, d, fam, piano, ric, pesi=None):
        dx = torch.empty(gy.shape[0], fam.C, dtype=gy.dtype, device=gy.device)
        o = piano.offs.tolist()
        for e in range(fam.E):
            if o[e + 1] > o[e]:
                W = peso_esperto_torch(fam, e, d, ric)
                if ric.dtype_gemm == torch.float16:
                    dx[o[e]:o[e + 1]] = torch.matmul(gy[o[e]:o[e + 1]].float(), W.float()).to(dx.dtype)
                else:
                    dx[o[e]:o[e + 1]] = torch.matmul(gy[o[e]:o[e + 1]], W)
        return dx

    def dd(self, gy, x, fam, piano, d_dtype):
        acc = torch.float64 if gy.dtype == torch.float64 else torch.float32
        dd = torch.zeros(fam.E, fam.R, fam.NG, dtype=d_dtype, device=gy.device)
        o = piano.offs.tolist()
        for e in range(fam.E):
            if o[e + 1] > o[e]:
                dW = torch.matmul(gy[o[e]:o[e + 1]].to(acc).T, x[o[e]:o[e + 1]].to(acc))       # (R, C)
                q = trit_esperto(fam, e).to(acc)
                dd[e] = (q * dW).view(fam.R, fam.NG, fam.gs).sum(-1).to(d_dtype)
        return dd


class MotoreTriton:
    """Kernel Triton. denso=False: decodifica dentro il prodotto (il peso non passa dalla memoria globale).
    denso=True: un kernel decodifica byte -> peso, il prodotto lo legge (e per la ricetta A il peso de-ruotato
    si costruisce comunque denso, bf16).
    v2=True (motore «triton2», 27/9): kernel _k_gemm2 / _k_dx2 / _k_dd2 (decodifica a pezzi, scala per gruppo,
    ordine a gruppi per la L2, dL/dd a K variabile, gate+up nella stessa passata). Avanti e dX: stessi valori
    di «triton» AL BIT (stessi prodotti, stesso ordine lungo K); dL/dd: stesso valore salvo l'ordine delle somme
    fp32. Dove serve il peso denso (ricetta A sulle famiglie ruotate) resta la via di «triton»."""

    def __init__(self, denso: bool = False, blk: Optional[Blocchi] = None, v2: bool = False):
        if denso and v2:
            raise ValueError("triton2 e' solo a decodifica fusa")
        self.denso = denso
        self.v2 = v2
        self.blk = blk or Blocchi()
        self.nome = "triton_denso" if denso else ("triton2" if v2 else "triton")

    @property
    def gate_up_insieme(self) -> bool:
        return self.v2 and self.blk.gate_up

    def puo_fondere(self, fg, fu, ric) -> bool:
        """Gate e up nella stessa passata: motore triton2, stessa forma, peso decodificato nel kernel."""
        return (self.gate_up_insieme and (fg.tipo, fg.E, fg.R, fg.NBR) == (fu.tipo, fu.E, fu.R, fu.NBR)
                and not (fg.ruotata and not ric.ruota))

    def pesi(self, fam, d, ric, pezzo: int = 32):
        """Peso denso (E, R, C) se serve: ricetta A per le famiglie ruotate (R^T(trit x d) in fp32, poi
        dtype_gemm, a pezzi di `pezzo` esperti: il transitorio fp32 e' 1/8 della famiglia) o motore denso;
        altrimenti None (decodifica dentro il prodotto)."""
        if fam.ruotata and not ric.ruota:
            W = torch.empty(fam.E, fam.R, fam.C, dtype=ric.dtype_gemm, device=fam.blocchi.device)
            for e0 in range(0, fam.E, pezzo):
                e1 = min(fam.E, e0 + pezzo)
                sotto = DatiFamiglia(fam.nome, fam.tipo, fam.blocchi[e0:e1], fam.d0[e0:e1])
                W[e0:e1] = de_ruota(decodifica_pesi(sotto, d[e0:e1], torch.float32, self.blk)).to(ric.dtype_gemm)
            return W
        if self.denso:
            return decodifica_pesi(fam, d, ric.dtype_gemm, self.blk)
        return None

    def avanti(self, x, d, fam, piano, ric, pesi=None):
        W = self.pesi(fam, d, ric) if pesi is None else pesi
        if self.v2 and W is None:
            return _avanti2_triton(x, [fam], [d], piano, ric.dtype_uscita, self.blk)[0]
        return _gemm_triton(x, fam, d, W, piano, False, ric.dtype_uscita, self.blk)

    def avanti_due(self, x, ds, fams, piano, ric):
        """Gate e up sullo stesso x in una passata (solo se `puo_fondere`)."""
        return _avanti2_triton(x, list(fams), list(ds), piano, ric.dtype_uscita, self.blk)

    def indietro_x(self, gy, d, fam, piano, ric, pesi=None):
        W = self.pesi(fam, d, ric) if pesi is None else pesi
        if self.v2 and W is None:
            return _dx2_triton(gy, fam, d, piano, gy.dtype, self.blk)
        return _gemm_triton(gy, fam, d, W, piano, True, gy.dtype, self.blk)

    def dd(self, gy, x, fam, piano, d_dtype):
        if self.v2:
            return _dd2_triton(gy, x, fam, piano, self.blk).to(d_dtype)
        return _dd_triton(gy, x, fam, piano, self.blk).to(d_dtype)


def motore(nome: str, blk: Optional[Blocchi] = None):
    if nome == "torch":
        return MotoreTorch()
    if nome in ("triton", "triton_denso", "triton2"):
        if not _TRITON:
            raise RuntimeError("Triton non disponibile")
        return MotoreTriton(denso=(nome == "triton_denso"), blk=blk, v2=(nome == "triton2"))
    raise ValueError(nome)


# ═══════════════════════════════════ autograd per famiglia ═══════════════════════════════════
class Osservatore:
    """26/9. Guardie dei tensori fp16 della ricetta B/C (rischio 8 del progetto: trabocco a 65504; A2/G1: il
    gradiente sotto 2^-24 diventa zero), SENZA sincronizzazioni nel ciclo:
    - `trabocco`: bool sul dispositivo, acceso se un ingresso fp16 di un prodotto degli esperti (attivazione
      ruotata, silu(g)*u, gradiente scalato per K) o un dX fp16 non e' finito. Si legge in `Studente.controlla`.
    - `diagnostica = True` (solo al passo 0, per il cancello G1): per ogni (strato, famiglia) conta, sul
      gradiente d'uscita in fp32 GIA' scalato per K, gli elementi non nulli che in fp16 vanno a zero (< 2^-24)
      e quelli che diventano subnormali (< 2^-14), piu' il massimo |g|. Tensori raccolti; `riassunto()` li
      legge con una sincronizzazione sola."""

    def __init__(self, device):
        self.trabocco = torch.zeros((), dtype=torch.bool, device=device)
        self.diagnostica = False
        self._righe: list = []

    def guarda(self, t: torch.Tensor) -> None:
        if t.dtype == torch.float16:
            import e2e_kernel_qat as _KQ           # 9/10 E2E_KQAT=guardia: una passata sola, stesso predicato
            if _KQ.attivo("guardia"):
                _KQ.guarda_fp16(self.trabocco, t, self.__dict__.setdefault("_flag", {}))
            else:
                self.trabocco |= ~torch.isfinite(t).all()

    def gradiente(self, fam: "DatiFamiglia", g32: torch.Tensor, g16: torch.Tensor) -> None:
        self.guarda(g16)
        if self.diagnostica:
            a = g32.abs()
            nz = a > 0
            self._righe.append((fam.strato, fam.nome, nz.sum(), (nz & (g16 == 0)).sum(),
                                (nz & (g16 != 0) & (g16.abs() < 2.0 ** -14)).sum(), a.max()))

    def riassunto(self) -> list:
        if not self._righe:
            return []
        numeri = torch.stack([torch.stack([r[2].double(), r[3].double(), r[4].double(), r[5].double()])
                              for r in self._righe]).tolist()
        out = [{"strato": r[0], "famiglia": r[1], "non_nulli": int(n[0]),
                "a_zero_in_fp16": int(n[1]), "subnormali_in_fp16": int(n[2]), "max_abs": n[3],
                "quota_a_zero": n[1] / max(n[0], 1), "quota_subnormali": n[2] / max(n[0], 1)}
               for r, n in zip(self._righe, numeri)]
        self._righe = []
        return out


class FamigliaEsperti(torch.autograd.Function):
    """y = prodotto raggruppato di x (token ordinati per esperto, (S, C)) coi pesi di TUTTI gli esperti di
    una famiglia, W_e = trit_e x d_e. Differenziabile in x e in d (le scale, gia' arrotondate a fp16).

    Salvati per l'indietro: x e d (e nient'altro). Indietro:
      dx      = raggruppato(gy, W_e) con W_e ricostruito dai byte;
      dL/dd   = somma per gruppo di trit (.) (gy_e^T x_rot_e), fp32, senza materializzare dW;
    dove x_rot = x per la ricetta B (x e' gia' ruotata) e per il down; = R x per la ricetta A (x reale).
    Costo O(E): ogni esperto e' toccato una volta per passata."""

    @staticmethod
    def forward(ctx, x, d, fam: DatiFamiglia, piano: Piano, ric: Ricetta, mot, oss: Optional[Osservatore] = None):
        y = mot.avanti(x, d, fam, piano, ric)
        ctx.save_for_backward(x, d)
        ctx.fam, ctx.piano, ctx.ric, ctx.mot, ctx.oss = fam, piano, ric, mot, oss
        return y

    @staticmethod
    def backward(ctx, gy):
        x, d = ctx.saved_tensors
        dx, dd = _indietro_famiglia(gy, x, d, ctx.fam, ctx.piano, ctx.ric, ctx.mot, ctx.oss,
                                    ctx.needs_input_grad[0], ctx.needs_input_grad[1])
        return dx, dd, None, None, None, None, None


def _indietro_famiglia(gy, x, d, fam, piano, ric, mot, oss, serve_x: bool, serve_d: bool):
    """Indietro di UNA famiglia (corpo di `FamigliaEsperti.backward`, condiviso con `FamiglieGateUp`)."""
    if ric.indietro_diviso and ric.dtype_indietro is None:      # 27/9: prodotti divisi (e2e_indietro_hilo)
        import e2e_indietro_hilo as HL
        return HL.indietro_famiglia(gy, x, d, fam, piano, ric, mot, oss, serve_x, serve_d)
    di = ric.dtype_indietro or x.dtype          # dtype degli INGRESSI dei prodotti dell'indietro
    g_in = gy
    gy = gy.to(di).contiguous()
    if oss is not None:
        oss.gradiente(fam, g_in, gy)
    if serve_d and getattr(getattr(mot, "stato", None), "salta_dd_superflui", False):
        import e2e_pv_codici as _PV                # 2/10 --pv-dd-solo-salti: dL/dd che nessuno legge
        if _PV.dd_superfluo(mot, fam):
            serve_d = False
    dx = None
    if serve_x:
        pesi = mot.pesi(fam, d, ric)
        dx = mot.indietro_x(gy, d, fam, piano, ric, pesi).to(x.dtype)
        del pesi
        if oss is not None:
            oss.guarda(dx)
    dd = None
    if serve_d:
        xr = ruota(x.to(torch.float64 if x.dtype == torch.float64 else torch.float32)) \
            if (fam.ruotata and not ric.ruota) else x
        dd = mot.dd(gy, xr.to(di).contiguous(), fam, piano, d.dtype)
    return dx, dd


class FamiglieGateUp(torch.autograd.Function):
    """27/9, motore triton2: (g, u) = gate e up sullo STESSO x in una passata del kernel (A letto una volta).
    Avanti identico al bit a due `FamigliaEsperti`. Indietro: le due famiglie come prima, prima up poi gate
    (l'ordine del motore autograd con due nodi), dx = dx_up + dx_gate nel dtype di x (la somma che autograd
    faceva accumulando il gradiente di x: commutativa, stesso risultato)."""

    @staticmethod
    def forward(ctx, x, dg, du, fg: DatiFamiglia, fu: DatiFamiglia, piano: Piano, ric: Ricetta, mot,
                oss: Optional[Osservatore] = None):
        g, u = mot.avanti_due(x, (dg, du), (fg, fu), piano, ric)
        ctx.save_for_backward(x, dg, du)
        ctx.fg, ctx.fu, ctx.piano, ctx.ric, ctx.mot, ctx.oss = fg, fu, piano, ric, mot, oss
        return g, u

    @staticmethod
    def backward(ctx, gg, gu):
        x, dg, du = ctx.saved_tensors
        nx, ng, nu = ctx.needs_input_grad[0], ctx.needs_input_grad[1], ctx.needs_input_grad[2]
        dx_u, ddu = _indietro_famiglia(gu, x, du, ctx.fu, ctx.piano, ctx.ric, ctx.mot, ctx.oss, nx, nu)
        dx_g, ddg = _indietro_famiglia(gg, x, dg, ctx.fg, ctx.piano, ctx.ric, ctx.mot, ctx.oss, nx, ng)
        dx = dx_u + dx_g if nx else None
        return dx, ddg, ddu, None, None, None, None, None, None


# 20/10 (E2E_SCALE_FISSE=1, default spento): con theta congelato (requires_grad False, come in CARDAN congiunto/KD) la scala
# fp16(d0*exp(theta)) e' costante: la si calcola una volta per famiglia e la si riusa (stessa funzione, stessi ingressi ->
# valori identici al bit; la firma include le _version di d0 e theta, quindi una modifica la ricalcola). Py-spy 20/10 22:5x:
# 63 % dei campioni del passo congiunto stavano in ArrotondaFP16.forward chiamata da scala().
_SCALE_FISSE = os.environ.get("E2E_SCALE_FISSE", "0") == "1"
_CACHE_SCALE: dict = {}


class ArrotondaFP16(torch.autograd.Function):
    """Avanti: il valore fp16 RNE (== `.half()`, numpy astype(float16), GGML_FP32_TO_FP16), riportato al
    dtype d'ingresso: e' la scala che il GGUF conterra'. Indietro: identita' (straight-through). Il controllo
    di finitezza NON sincronizza: accumula in `sentinella` (tensore bool sul dispositivo); chi allena lo
    legge una volta per passo (`Studente.controlla`). Copia di `e2e_prototipo.ArrotondaFP16` salvo questo."""

    @staticmethod
    def forward(ctx, s: torch.Tensor, sentinella: Optional[torch.Tensor]):
        r = s.to(torch.float16)
        if sentinella is None:
            if not torch.isfinite(r).all():
                raise FloatingPointError("scala fuori dal campo fp16 (|d| > 65504 o NaN)")
        else:
            sentinella |= ~torch.isfinite(r).all()
        return r.to(s.dtype)

    @staticmethod
    def backward(ctx, g):
        return g, None


class _Raccogli(torch.autograd.Function):
    """x_ord = x[ordine // k] (x: (T, H)). Indietro diretto: il gradiente torna all'ordine originale con la
    permutazione INVERSA (un gather) e le k copie di ogni token si sommano in fp32 (come esperti_thor)."""

    @staticmethod
    def forward(ctx, x, ordine, inversa, k):
        ctx.save_for_backward(inversa)
        ctx.k, ctx.T = k, x.shape[0]
        return x[ordine // k]

    @staticmethod
    def backward(ctx, g):
        (inversa,) = ctx.saved_tensors
        acc = torch.float64 if g.dtype == torch.float64 else torch.float32
        dx = g[inversa].view(ctx.T, ctx.k, -1).to(acc).sum(1).to(g.dtype)
        return dx, None, None, None


class _Disperdi(torch.autograd.Function):
    """y[t] = sum_j o[inversa[t*k + j]] (o: (S, H) ordinato per esperto), somma in fp32/fp64."""

    @staticmethod
    def forward(ctx, o, ordine, inversa, k):
        ctx.save_for_backward(ordine)
        ctx.k = k
        acc = torch.float64 if o.dtype == torch.float64 else torch.float32
        return o[inversa].view(-1, k, o.shape[1]).to(acc).sum(1)

    @staticmethod
    def backward(ctx, g):
        (ordine,) = ctx.saved_tensors
        k = ctx.k
        gs = g.unsqueeze(1).expand(-1, k, -1).reshape(-1, g.shape[1])       # riga t*k+j
        return gs[ordine].contiguous(), None, None, None


# Combinazione fusa (26/9 10:5x, leva di velocita' misurabile prima/dopo in M0: `misura --combina fusa,torch`).
# E2E_COMBINA_FUSA=0 torna alla via torch (o*wo, raccolta, somma). Attiva solo col motore Triton e uscite fp32.
COMBINA_FUSA = os.getenv("E2E_COMBINA_FUSA", "1") == "1"
BH_COMBINA = 512


class CombinaEsperti(torch.autograd.Function):
    """y[t] = sum_j w[t, j] * o[inversa[t*k + j]]: o (S, H) fp32 = uscite NON pesate del down, in ordine per
    esperto; w (T, k) fp32 = pesi d'instradamento rinormalizzati. Sostituisce `o * wo` + `_Disperdi`, che a
    8192 token e H = 2048 muovono (S, H) fp32 = 512 MiB quattro volte in avanti e altrettante indietro (copia
    pesata, raccolta, somma; nell'indietro espansione, raccolta e prodotto). Qui: avanti legge o e scrive y;
    indietro scrive dL/do (una passata) e legge o per dL/dw (un prodotto scalare per assegnazione). Salva o, w e
    le permutazioni (la via torch salvava o pesata e wo: stessa memoria)."""

    @staticmethod
    def forward(ctx, o, w, ordine, inversa, k):
        o = o.contiguous()
        w = w.contiguous()
        T, H = w.shape[0], o.shape[1]
        y = torch.empty(T, H, dtype=torch.float32, device=o.device)
        _k_combina[(T, triton.cdiv(H, BH_COMBINA))](o, o.stride(0), w, inversa, y, y.stride(0), H, k, BH_COMBINA,
                                                    num_warps=4)
        ctx.save_for_backward(o, w, ordine, inversa)
        ctx.k = k
        return y

    @staticmethod
    def backward(ctx, gy):
        o, w, ordine, inversa = ctx.saved_tensors
        k = ctx.k
        g = gy.to(torch.float32).contiguous()
        S_, H = o.shape
        do = dw = None
        if ctx.needs_input_grad[0]:
            do = torch.empty(S_, H, dtype=torch.float32, device=o.device)
            _k_combina_do[(S_, triton.cdiv(H, BH_COMBINA))](g, g.stride(0), w, ordine, do, do.stride(0), H, k,
                                                             BH_COMBINA, num_warps=4)
        if ctx.needs_input_grad[1]:
            dw = torch.empty(w.shape, dtype=torch.float32, device=o.device)
            _k_combina_dw[(S_,)](g, g.stride(0), o, o.stride(0), inversa, dw, H, k, BH_COMBINA, num_warps=4)
        return do, dw, None, None, None


# ═══════════════════════════════════ MoE dello studente ═══════════════════════════════════
class MoEStudente(torch.nn.Module):
    """Sostituisce `Qwen3_5MoeSparseMoeBlock`: router (softmax fp32, top-k, rinormalizzazione, come
    Qwen3.5-MoE e llama.cpp), esperto condiviso con porta sigmoide (moduli HF coi pesi del GGUF), esperti
    instradati con `FamigliaEsperti`. `instr` (impostato da chi chiama) = top-k salvato da riusare;
    `ultimo_idx` = top-k usato nell'ultima chiamata; `cambi` = tensore (dispositivo) dei token il cui top-k
    vivo differisce da quello salvato."""

    def __init__(self, router_w, shared, shared_gate, strato: "StratoEsperti", top_k: int):
        super().__init__()
        self.router_w = router_w          # (E, H), nn.Parameter senza gradiente
        self.shared_expert = shared
        self.shared_expert_gate = shared_gate
        self.esperti = strato
        self.top_k = top_k
        self.instr: Optional[torch.Tensor] = None
        self.ultimo_idx: Optional[torch.Tensor] = None
        self.cambi: Optional[torch.Tensor] = None
        self.router_w32: Optional[torch.Tensor] = None   # attributo semplice (NON buffer: .to(bf16) non lo tocca)

    def _piano2(self, nome: str, idx: torch.Tensor, piano: "Piano", cache: dict):
        """9/10 (cura P1): (Piano del 2° piano, perm_12, perm_21) per la proiezione `nome`, o None se non c'e'.
        idx2 = mappa2[idx] (esperti senza 2° piano -> slot nullo E2). perm_12[q] = riga nell'ordine del primo piano
        dell'assegnazione q del secondo (x2 = x_ord[perm_12]); perm_21[p] = riga del secondo per la riga p del primo
        (y += y2[perm_21]). Stesso significato del `second_plane` del motore (maschera = slot nullo, uscita 0 esatto)."""
        st = self.esperti
        m = st.mappa2.get(nome)
        if m is None:
            return None
        k = id(m)
        if k not in cache:
            p2 = Piano(m[idx], st.fam2[nome].E)
            cache[k] = (p2, piano.inversa[p2.ordine], p2.inversa[piano.ordine])
        return cache[k]

    def _secondo_piano(self, nome: str, x_ord: torch.Tensor, idx, piano, cache, oss):
        """Contributo del 2° piano (0 se assente) nell'ordine del primo piano: mm(W2[map[e]], x) per assegnazione."""
        st = self.esperti
        t = self._piano2(nome, idx, piano, cache)
        if t is None:
            return None
        p2, perm_12, perm_21 = t
        y2 = FamigliaEsperti.apply(x_ord[perm_12].contiguous(), st.scala2(nome), st.fam2[nome], p2, st.ric, st.mot,
                                   oss)
        return y2[perm_21]

    def forward(self, hidden_states):
        B, T, H = hidden_states.shape
        x = hidden_states.reshape(-1, H)
        st = self.esperti
        ric = st.ric
        if self.router_w32 is not None and x.dtype not in (torch.float32, torch.float64):
            logit = F.linear(x.float(), self.router_w32)
        else:
            logit = F.linear(x, self.router_w)
        prob = torch.softmax(logit, dim=-1, dtype=torch.float32 if x.dtype != torch.float64 else torch.float64)
        with torch.no_grad():
            vivo = torch.topk(prob, self.top_k, dim=-1).indices
        if self.instr is None:
            idx = vivo
        else:
            idx = self.instr
            with torch.no_grad():
                diversi = (torch.sort(vivo, -1).values != torch.sort(idx, -1).values).any(-1).sum()
                self.cambi = diversi if self.cambi is None else self.cambi + diversi
        self.ultimo_idx = idx.detach()
        w = prob.gather(-1, idx)
        w = w / w.sum(-1, keepdim=True)
        condiviso = torch.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)

        piano = getattr(idx, "_piano_riuso", None) if RIUSO["attivo"] else None   # 29/9 tt3-cure (--pv-riuso)
        if piano is None:
            piano = Piano(idx, st.E)
            if RIUSO["attivo"]:
                self.ultimo_idx._piano_riuso = piano
        if ric.ruota:
            xg = ruota(x.to(torch.float64 if ric.dtype_gemm == torch.float64 else torch.float32)).to(ric.dtype_gemm)
        else:
            xg = x.to(ric.dtype_gemm)
        oss = st.oss
        if oss is not None:
            oss.guarda(xg)
        x_ord = _Raccogli.apply(xg, piano.ordine, piano.inversa, self.top_k)
        dg, du, dd = st.scala("gate"), st.scala("up"), st.scala("down")
        if getattr(st.mot, "gate_up_insieme", False) and st.mot.puo_fondere(st.fam["gate"], st.fam["up"], ric):
            g, u = FamiglieGateUp.apply(x_ord, dg, du, st.fam["gate"], st.fam["up"], piano, ric, st.mot, oss)
        else:
            g = FamigliaEsperti.apply(x_ord, dg, st.fam["gate"], piano, ric, st.mot, oss)
            u = FamigliaEsperti.apply(x_ord, du, st.fam["up"], piano, ric, st.mot, oss)
        c2: dict = {}
        if st.fam2:                         # 9/10 (cura P1) 2° piano degli esperti: up += p2(up), gate += p2(gate)
            u2 = self._secondo_piano("up", x_ord, idx, piano, c2, oss)
            if u2 is not None:
                u = u + u2.to(u.dtype)
            g2 = self._secondo_piano("gate", x_ord, idx, piano, c2, oss)
            if g2 is not None:
                g = g + g2.to(g.dtype)
        cardan = self._modules.get("cardan")   # 9/10 CARDAN-KD (cardan_kd.py): absent = bit-identical
        if cardan is not None:
            cg, cu = cardan(x_ord, piano)
            g = g + cg.to(g.dtype)
            u = u + cu.to(u.dtype)
        a = (F.silu(g) * u).to(ric.dtype_gemm)
        if oss is not None:
            oss.guarda(a)
        o = FamigliaEsperti.apply(a, dd, st.fam["down"], piano, ric, st.mot, oss)
        if st.fam2:                         # 9/10: down += p2(down) PRIMA del peso del router (come il motore)
            o2 = self._secondo_piano("down", a, idx, piano, c2, oss)
            if o2 is not None:
                o = o + o2.to(o.dtype)
        if COMBINA_FUSA and isinstance(st.mot, MotoreTriton) and o.dtype == torch.float32 \
                and w.dtype == torch.float32:
            y = CombinaEsperti.apply(o, w, piano.ordine, piano.inversa, self.top_k)
        else:
            wo = w.reshape(-1)[piano.ordine].to(o.dtype if o.dtype != torch.bfloat16 else torch.float32)
            o = o.to(wo.dtype) * wo.unsqueeze(-1)
            y = _Disperdi.apply(o, piano.ordine, piano.inversa, self.top_k)
        rexp = self._modules.get("rexp")       # 3/10 esperti di recupero (e2e_rexp): assente = identico al bit
        if rexp is not None:
            return (y.to(x.dtype) + condiviso + rexp(x)).reshape(B, T, H)
        return (y.to(x.dtype) + condiviso).reshape(B, T, H)


class StratoEsperti:
    """Le tre famiglie di UNO strato: byte + d0 + theta. theta fp32 (E, R, NG), unici parametri allenati."""

    def __init__(self, fam: dict, ric: Ricetta, mot, sentinella: Optional[torch.Tensor] = None,
                 oss: Optional[Osservatore] = None):
        self.fam = fam
        self.ric = ric
        self.mot = mot
        self.E = fam["gate"].E
        self.theta = {n: torch.zeros_like(f.d0, dtype=torch.float32, requires_grad=True) for n, f in fam.items()}
        self.sentinella = sentinella
        self.oss = oss
        # 9/10 (cura P1) secondo piano degli esperti: {gate/up/down: DatiFamiglia con E2 + 1 esperti (l'ultimo e' NULLO:
        # byte 0, d0 0, riceve le assegnazioni degli esperti senza 2° piano)} e {gate/up/down: LongTensor (E,) esperto ->
        # slot (E2 = nullo)}. FUORI da fam/theta: congelati (nessun parametro, nessuna chiave, nessun export dei codici).
        self.fam2: dict = {}
        self.mappa2: dict = {}

    def scala2(self, nome: str) -> torch.Tensor:
        """Scala del 2° piano degli esperti: fp16(d0) (= d0, gia' fp16 nel file), nel dtype dei prodotti."""
        d = self.fam2[nome].d0
        return d.to(torch.float64) if self.ric.dtype_gemm == torch.float64 else d

    def scala(self, nome: str) -> torch.Tensor:
        if _SCALE_FISSE and not self.theta[nome].requires_grad:   # 20/10: theta congelato -> scala costante, calcolata una volta
            f, th = self.fam[nome], self.theta[nome]
            k = (id(self), nome)
            firma = (id(f.d0), f.d0._version, id(th), th._version, id(self.sentinella), self.ric.dtype_gemm)
            v = _CACHE_SCALE.get(k)
            if v is None or v[0] != firma:
                with torch.no_grad():
                    v = _CACHE_SCALE[k] = (firma, self._scala(nome))
            return v[1]
        if RIUSO["attivo"] and not torch.is_grad_enabled():   # 29/9 tt3-cure (--pv-riuso): fuori dal grafo, riusata
            f, th = self.fam[nome], self.theta[nome]
            k = (id(self), nome)
            firma = (id(f.d0), f.d0._version, id(th), th._version, id(self.sentinella), self.ric.dtype_gemm)
            v = RIUSO["scale"].get(k)
            if v is None or v[0] != firma:
                v = RIUSO["scale"][k] = (firma, self._scala(nome))
            return v[1]
        return self._scala(nome)

    def _scala(self, nome: str) -> torch.Tensor:
        f = self.fam[nome]
        s = f.d0 * torch.exp(self.theta[nome])
        r = ArrotondaFP16.apply(s, self.sentinella)
        return r.to(torch.float64) if self.ric.dtype_gemm == torch.float64 else r


# ═══════════════════════════════════ densi ternari (28/9, --densi-ternari) ═══════════════════════════════════
def _voce_densa_ammessa(voce: dict, nome: str) -> None:
    """Relazioni file <-> HF che la LineareTernaria sa fare: identita', righe permutate (teste di attn_gate/attn_qkv),
    colonne permutate (ssm_out). Nessuna trasposta, fetta o trasformata (nessun tensore denso di Thor ne ha)."""
    if voce["rel"] not in ("identita", "perm_righe", "perm_colonne") or voce.get("trasposta") or voce.get("fetta") \
            or voce.get("f"):
        raise ValueError(f"{nome}: relazione {voce} non ammessa per un denso ternario")


class LineareTernaria(torch.nn.Module):
    """28/9 --densi-ternari. Sostituisce il `nn.Linear` HF di UN tensore denso ternario del GGUF con la macchina degli
    esperti a esperto UNICO: y_hf = [x_file W_file(d)^T] riportato nell'ordine HF, con
      x_file = x_hf[:, perm] (relazione perm_colonne, ssm_out), altrimenti x_hf;
      W_file = R^T (trit x d) per TQ1_0R_1024 (ingresso ruotato come gli esperti: ricetta B = R x in fp32 -> fp16),
               trit x d per TQ1_0 (ingresso nel dtype dei prodotti, come il down degli esperti);
      y_hf = y_file[:, inv] (relazione perm_righe, attn_gate/attn_qkv: W_hf = W_file[inv]).
    Il prodotto e' `FamigliaEsperti` su un Piano con UN esperto (tutti i token all'esperto 0): stessi kernel, stesso dL/dd
    (somma per gruppo di trit (.) dY^T X_rot), stesso motore dello strato (`es.mot`, letto a ogni chiamata: il motore V
    installato da e2e_pv_codici vale anche qui), stessa scala d = fp16(d0 exp(theta)) (`es.scala`), stessa ricetta
    (`es.ric`, cambiabile da `imposta_ricetta`). Nessun parametro del modulo (theta vive in StratoEsperti)."""

    def __init__(self, es: "StratoEsperti", nome: str, voce: dict, fuori: int, dentro: int,
                 nome2: Optional[str] = None, nome3: Optional[str] = None, nome4: Optional[str] = None):
        super().__init__()
        _voce_densa_ammessa(voce, nome)
        self.es = es
        self.nome = nome
        # 29/9 secondo piano (tensore <nome>2 nel GGUF): y_file = x W1(d1)^T + x W2(d2)^T, stessa relazione file <-> HF
        # (il motore lo scrive nello stesso ordine del primo piano), scale e codici propri. None = di sempre.
        self.nome2 = nome2
        self.nome3 = nome3          # 30/9 terzo piano: + x W3(d3)^T dopo il secondo (ordine del motore); None = di sempre
        self.nome4 = nome4          # 9/10 quarto piano: + x W4(d4)^T dopo il terzo (ordine del motore); None = di sempre
        self.in_features, self.out_features = int(dentro), int(fuori)
        px = iy = None
        if voce["rel"] in ("perm_righe", "perm_colonne"):
            import carica_strato_gguf as CSG  # sola lettura
            perm = np.asarray(CSG.PERM[voce["perm_npz"]], dtype=np.int64)
            if voce["rel"] == "perm_colonne":
                px = torch.from_numpy(perm.copy())                 # x_file[:, c] = x_hf[:, perm[c]]
            else:
                inv = np.empty_like(perm)
                inv[perm] = np.arange(len(perm))
                iy = torch.from_numpy(inv)                         # y_hf[:, j] = y_file[:, inv[j]]
        # buffer interi non persistenti: `.to(dtype)` dello strato non li tocca, `.to(device)` li sposta
        self.register_buffer("perm_x", px, persistent=False)
        self.register_buffer("inv_y", iy, persistent=False)

    @staticmethod
    def _ingresso(xf, fam, ric):
        if fam.ruotata and ric.ruota:
            return ruota(xf.to(torch.float64 if ric.dtype_gemm == torch.float64 else torch.float32)).to(ric.dtype_gemm)
        return xf.to(ric.dtype_gemm)

    def forward(self, x):
        es = self.es
        fam = es.fam[self.nome]
        ric = es.ric
        forma = x.shape
        xf = x.reshape(-1, forma[-1])
        if self.perm_x is not None:
            xf = xf[:, self.perm_x]
        xg = self._ingresso(xf, fam, ric)
        if es.oss is not None:
            es.oss.guarda(xg)
        if (self.nome2 is not None or self.nome4 is not None) and _DP.attivo():   # 1/10 densi-piegati: spento = di sempre
            nomi = [self.nome] + [n for n in (self.nome2, self.nome3, self.nome4) if n is not None]
            fams = [es.fam[n] for n in nomi]
            if _DP.ammesso(fams, ric, es.mot, xg):
                import e2e_kernel_qat as _KQ          # 9/10 E2E_KQAT=cache_densi: chiave theta del peso piegato
                _KQ.CHIAVE_THETA = tuple((id(es.theta[n]), es.theta[n]._version) for n in nomi) \
                    if _KQ.attivo("cache_densi") else None
                try:
                    y = _DP.applica(xg, [es.scala(n) for n in nomi], fams, ric, es.mot, es.oss)
                finally:
                    _KQ.CHIAVE_THETA = None
                if self.inv_y is not None:
                    y = y[:, self.inv_y]
                return y.to(x.dtype).reshape(*forma[:-1], fam.R)
            _DP.CONTI["ripiego"] += 1
        piano = piano_unico(xg.shape[0], xg.device)            # 29/9 tt3-cure: spento = Piano nuovo, come sempre
        y = FamigliaEsperti.apply(xg.contiguous(), es.scala(self.nome), fam, piano, ric, es.mot, es.oss)
        if self.nome2 is not None:                 # 29/9 secondo piano: + x W2(d2)^T (stesso piano, stesso motore)
            fam2 = es.fam[self.nome2]
            xg2 = xg if fam2.ruotata == fam.ruotata else self._ingresso(xf, fam2, ric)
            if xg2 is not xg and es.oss is not None:
                es.oss.guarda(xg2)
            y = y + FamigliaEsperti.apply(xg2.contiguous(), es.scala(self.nome2), fam2, piano, ric, es.mot, es.oss)
        if self.nome3 is not None:                 # 30/9 terzo piano: + x W3(d3)^T (stesso piano, stesso motore)
            fam3 = es.fam[self.nome3]
            xg3 = xg if fam3.ruotata == fam.ruotata else self._ingresso(xf, fam3, ric)
            if xg3 is not xg and es.oss is not None:
                es.oss.guarda(xg3)
            y = y + FamigliaEsperti.apply(xg3.contiguous(), es.scala(self.nome3), fam3, piano, ric, es.mot, es.oss)
        if self.nome4 is not None:                 # 9/10 quarto piano: + x W4(d4)^T (stesso piano, stesso motore)
            fam4 = es.fam[self.nome4]
            xg4 = xg if fam4.ruotata == fam.ruotata else self._ingresso(xf, fam4, ric)
            if xg4 is not xg and es.oss is not None:
                es.oss.guarda(xg4)
            y = y + FamigliaEsperti.apply(xg4.contiguous(), es.scala(self.nome4), fam4, piano, ric, es.mot, es.oss)
        if self.inv_y is not None:
            y = y[:, self.inv_y]
        return y.to(x.dtype).reshape(*forma[:-1], fam.R)

    def extra_repr(self) -> str:
        p2 = f", secondo_piano={self.nome2}" if self.nome2 is not None else ""
        p2 += f", terzo_piano={self.nome3}" if getattr(self, "nome3", None) is not None else ""
        p2 += f", quarto_piano={self.nome4}" if getattr(self, "nome4", None) is not None else ""
        return f"in_features={self.in_features}, out_features={self.out_features}, ternaria={self.nome}{p2}"


def voci_dense(tens: dict, L: int, pieno: bool, densi) -> dict:
    """{famiglia: (nome GGUF, voce della mappa)} delle famiglie dense chieste PRESENTI nello strato L (una famiglia
    dell'altro tipo di strato, p.es. attn_q in uno strato GDN, si salta). Solleva se il tensore non e' ternario
    (TQ1_0R_1024 / TQ1_0) o non e' 2D: con --densi-ternari un k-quant non si allena in silenzio al posto di un ternario."""
    if not densi:
        return {}
    import carica_strato_gguf as CSG  # sola lettura
    mappa = CSG.MAPPA["full" if pieno else "linear"]
    out = {}
    for n in densi:
        gg = f"{FAMIGLIE_DENSE[n]}.weight"
        nome = f"blk.{L}.{gg}"
        if gg not in mappa or nome not in tens:
            continue
        t = tens[nome]
        tipo = t.tensor_type.name
        if tipo not in CS.LAYOUT:
            raise ValueError(f"{nome}: tipo {tipo}, non ternario (--densi-ternari vuole TQ1_0R_1024 o TQ1_0)")
        if np.asarray(t.data).ndim != 2:
            raise ValueError(f"{nome}: tensore non 2D {np.asarray(t.data).shape}")
        _voce_densa_ammessa(mappa[gg], nome)
        out[n] = (nome, mappa[gg])
    # 29/9 secondo piano: <nome>2 accanto a ogni famiglia chiesta presente, DOPO tutti i primi piani (le chiavi e i
    # parametri di sempre restano al loro posto). Stesso tipo e stessa forma del primo piano, o si solleva.
    for n in list(out):
        n2 = f"{n}{SUFFISSO_P2}"
        nome2 = f"blk.{L}.{FAMIGLIE_DENSE2[n2]}.weight"
        if nome2 not in tens:
            continue
        t1, t2 = tens[out[n][0]], tens[nome2]
        if t2.tensor_type.name != t1.tensor_type.name or np.asarray(t2.data).shape != np.asarray(t1.data).shape:
            raise ValueError(f"{nome2}: tipo {t2.tensor_type.name} {np.asarray(t2.data).shape}, il primo piano e' "
                             f"{t1.tensor_type.name} {np.asarray(t1.data).shape}")
        out[n2] = (nome2, out[n][1])
    # 30/9 terzo piano: <nome>3 accanto a ogni famiglia chiesta presente, DOPO tutti i *2 (chiavi e parametri dei primi e
    # secondi piani restano al loro posto). Stesso tipo e stessa forma del primo piano, o si solleva.
    for n in [m for m in out if m in FAMIGLIE_DENSE]:
        n3 = f"{n}{SUFFISSO_P3}"
        nome3 = f"blk.{L}.{FAMIGLIE_DENSE3[n3]}.weight"
        if nome3 not in tens:
            continue
        t1, t3 = tens[out[n][0]], tens[nome3]
        if t3.tensor_type.name != t1.tensor_type.name or np.asarray(t3.data).shape != np.asarray(t1.data).shape:
            raise ValueError(f"{nome3}: tipo {t3.tensor_type.name} {np.asarray(t3.data).shape}, il primo piano e' "
                             f"{t1.tensor_type.name} {np.asarray(t1.data).shape}")
        out[n3] = (nome3, out[n][1])
    # 9/10 quarto piano: <nome>4 accanto a ogni famiglia chiesta presente, DOPO tutti i *3. Stesso tipo e forma del primo.
    for n in [m for m in out if m in FAMIGLIE_DENSE]:
        n4 = f"{n}{SUFFISSO_P4}"
        nome4 = f"blk.{L}.{FAMIGLIE_DENSE4[n4]}.weight"
        if nome4 not in tens:
            continue
        t1, t4 = tens[out[n][0]], tens[nome4]
        if t4.tensor_type.name != t1.tensor_type.name or np.asarray(t4.data).shape != np.asarray(t1.data).shape:
            raise ValueError(f"{nome4}: tipo {t4.tensor_type.name} {np.asarray(t4.data).shape}, il primo piano e' "
                             f"{t1.tensor_type.name} {np.asarray(t1.data).shape}")
        out[n4] = (nome4, out[n][1])
    return out


def nome_p2(nome_g: str) -> str:
    """"blk.3.attn_q.weight" -> "blk.3.attn_q2.weight" (e "attn_q.weight" -> "attn_q2.weight")."""
    return f"{nome_g[:-len('.weight')]}{SUFFISSO_P2}.weight"


def nome_p3(nome_g: str) -> str:
    """"blk.3.attn_q.weight" -> "blk.3.attn_q3.weight" (e "output.weight" -> "output3.weight")."""
    return f"{nome_g[:-len('.weight')]}{SUFFISSO_P3}.weight"


def nome_p4(nome_g: str) -> str:
    """"blk.3.attn_q.weight" -> "blk.3.attn_q4.weight" (9/10 quarto piano; "output.weight" -> "output4.weight")."""
    return f"{nome_g[:-len('.weight')]}{SUFFISSO_P4}.weight"


def piani_di(tens: dict, nome_g: str) -> list:
    """30/9: [nome_g] + i piani in piu' PRESENTI nel GGUF, nell'ordine di somma del motore (<nome>2, poi <nome>3).
    Vale per i densi (cammino dequantizzato) e per le tabelle token_embd / output (motore secondo-piano-embd-output
    0b81365: get_rows(te) + get_rows(te2) + get_rows(te3); mm(out) + mm(out2) + mm(out3))."""
    if not nome_g.endswith(".weight"):
        return [nome_g]
    # 9/10: + <nome>4 (motore thor-e1-shexp: output4 `cur += mm(output4, h)` dopo output3; densi *4 dopo *3)
    return [nome_g] + [p for p in (nome_p2(nome_g), nome_p3(nome_g), nome_p4(nome_g)) if p in tens]


def _padre_lineare(strato, hf: str) -> tuple:
    """"linear_attn.in_proj_qkv.weight" -> (modulo che contiene il Linear, nome del Linear)."""
    pref = hf.rpartition(".")[0]
    padre_n, _, lin = pref.rpartition(".")
    return (strato.get_submodule(padre_n) if padre_n else strato), lin


# ═══════════════════════════════════ tensori F32 allenabili (27/9, --allena-f32) ═══════════════════════════════════
# famiglia -> nome del tensore GGUF senza "blk.N." (ordine canonico: e' l'ordine dei parametri nel gruppo di Adam)
FAMIGLIE_F32 = {
    "attn_norm": "attn_norm.weight", "post_attention_norm": "post_attention_norm.weight",
    "attn_q_norm": "attn_q_norm.weight", "attn_k_norm": "attn_k_norm.weight", "ssm_norm": "ssm_norm.weight",
    "ffn_gate_inp": "ffn_gate_inp.weight", "ffn_gate_inp_shexp": "ffn_gate_inp_shexp.weight",
    "ssm_a": "ssm_a", "ssm_alpha": "ssm_alpha.weight", "ssm_beta": "ssm_beta.weight",
    "ssm_conv1d": "ssm_conv1d.weight", "ssm_dt": "ssm_dt.bias",
}
F32_GLOBALI = {"output_norm": "output_norm.weight"}


def famiglie_f32(voci) -> tuple:
    """"a,b" o iterabile -> tupla canonica (ordine di FAMIGLIE_F32, poi output_norm). Vuota = nessuna (default:
    comportamento identico al bit). Accetta anche il nome col suffisso (ssm_dt.bias, attn_norm.weight)."""
    if isinstance(voci, str):
        voci = [v for v in voci.split(",")]
    chieste = set()
    for v in voci:
        v = v.strip()
        if not v:
            continue
        for suf in (".weight", ".bias"):
            if v.endswith(suf):
                v = v[:-len(suf)]
        if v not in FAMIGLIE_F32 and v not in F32_GLOBALI:
            raise ValueError(f"famiglia F32 sconosciuta {v!r}: {sorted(FAMIGLIE_F32) + sorted(F32_GLOBALI)}")
        chieste.add(v)
    return tuple(f for f in list(FAMIGLIE_F32) + list(F32_GLOBALI) if f in chieste)


def famiglia_del_tensore_f32(nome: str) -> str:
    """28/9: nome del tensore GGUF ("blk.12.ffn_gate_inp.weight", "output_norm.weight") -> famiglia F32
    ("ffn_gate_inp", "output_norm"); solleva per un nome che non e' di una famiglia F32 allenabile."""
    for fam, suf in F32_GLOBALI.items():
        if nome == suf:
            return fam
    parti = nome.split(".", 2)
    if len(parti) == 3 and parti[0] == "blk" and parti[1].isdigit():
        for fam, suf in FAMIGLIE_F32.items():
            if parti[2] == suf:
                return fam
    raise ValueError(f"{nome!r}: non e' il nome di un tensore F32 allenabile")


def _trasforma_torch(m: torch.Tensor, voce: Optional[dict], inv: Optional[torch.Tensor]) -> torch.Tensor:
    """Da valore GGUF (ordine numpy del file) a parametro HF: la STESSA algebra di `carica_strato_gguf._applica`
    in torch (differenziabile). Sottrazione di 1 e permutazioni sono esatte: stesso bit di numpy (verificato alla
    costruzione); log(-v) (ssm_a) passa da `_ValoreNumpy` per il valore e usa questa solo nell'indietro."""
    if voce is None:
        return m
    rel = voce["rel"]
    if rel == "identita":
        out = m
    elif rel == "trasformata":
        out = m - 1.0 if voce["f"] == "piu_uno" else torch.log(-m)
    elif rel in ("perm_righe", "perm_1d"):
        out = m[inv]
        if voce.get("f") == "neg_exp":
            out = torch.log(-out)
    elif rel == "perm_colonne":
        out = m[:, inv]
    else:
        raise ValueError(f"relazione sconosciuta: {rel}")
    if voce.get("trasposta"):
        out = out.T
    if voce.get("fetta") == "squeeze":
        out = out[:, None, :]
    if voce["hf"].endswith("shared_expert_gate.weight") and out.ndim == 1:
        out = out[None, :]
    return out


class _ValoreNumpy(torch.autograd.Function):
    """Valore = `carica_strato_gguf._applica` (numpy, il caricatore del GGUF esportato: stesso bit alla
    ricarica anche su GPU, dove log fp32 puo' differire da numpy di 1 ulp); gradiente = quello di `_trasforma_torch`."""

    @staticmethod
    def forward(ctx, m, valore, voce, inv):
        ctx.save_for_backward(m)
        ctx.voce, ctx.inv = voce, inv
        return valore.clone()

    @staticmethod
    def backward(ctx, g):
        (m,) = ctx.saved_tensors
        with torch.enable_grad():
            mm = m.detach().requires_grad_(True)
            (gm,) = torch.autograd.grad(_trasforma_torch(mm, ctx.voce, ctx.inv), mm, g)
        return gm, None, None, None


class TensoreF32:
    """UN tensore F32 del GGUF allenato. `master` = i valori del file (ordine numpy del GGUF, fp32; fp64 solo con
    le ricette di prova a 64 bit), foglia con gradiente: e' cio' che Adam aggiorna e che l'export scrive AL POSTO
    dei byte vecchi. Il parametro HF del modulo non e' piu' un nn.Parameter: `installa()` lo ricalcola da `master`
    (con grafo se il gradiente e' acceso) prima di ogni chiamata dello strato, quindi l'avanti senza grafo e il
    ricalcolo dell'indietro vedono lo STESSO valore (instradamento salvato e contatore dei cambi invariati)."""

    def __init__(self, nome: str, famiglia: str, L: int, voce: Optional[dict], valore_np: np.ndarray, device,
                 dtype_master: torch.dtype, modulo=None, attributo: Optional[str] = None,
                 dtype_modulo: Optional[torch.dtype] = None, righe: Optional[slice] = None):
        self.nome, self.famiglia, self.L, self.voce = nome, famiglia, L, voce
        self.forma_file = tuple(valore_np.shape)
        self.righe = righe
        v = valore_np if righe is None else valore_np[righe]
        self.master = torch.from_numpy(np.array(v, dtype=np.float32, copy=True)).to(device=device,
                                                                                    dtype=dtype_master)
        self.master.requires_grad_(True)
        self.modulo, self.attributo, self.dtype_modulo = modulo, attributo, dtype_modulo
        self.inv = None
        if voce is not None and voce["rel"] in ("perm_righe", "perm_1d", "perm_colonne"):
            import carica_strato_gguf as CSG  # sola lettura
            perm = CSG.PERM[voce["perm_npz"]]
            inv = np.empty_like(perm)
            inv[perm] = np.arange(len(perm))
            self.inv = torch.from_numpy(inv.astype(np.int64)).to(device)
        self._numpy = voce is not None and voce.get("f") == "neg_exp"
        self._cache = (None, None)

    def _valore_numpy(self) -> torch.Tensor:
        ver = self.master._version
        if self._cache[0] != ver:
            import carica_strato_gguf as CSG  # sola lettura
            a = CSG._applica(self.voce, self.master.detach().cpu().numpy())
            self._cache = (ver, torch.from_numpy(np.array(a, copy=True)).to(self.master.device))
        return self._cache[1]

    def valore_hf(self) -> torch.Tensor:
        if self._numpy:
            if torch.is_grad_enabled() and self.master.requires_grad:
                return _ValoreNumpy.apply(self.master, self._valore_numpy(), self.voce, self.inv)
            return self._valore_numpy()
        return _trasforma_torch(self.master, self.voce, self.inv)

    def installa(self) -> None:
        v = self.valore_hf()
        if self.attributo == "router_w":
            self.modulo.router_w = v.to(self.dtype_modulo)
            if self.modulo.router_w32 is not None:
                self.modulo.router_w32 = v.to(torch.float32)
        elif self.modulo is not None:
            setattr(self.modulo, self.attributo, v.to(self.dtype_modulo))

    def valore_file(self) -> np.ndarray:
        """fp32 numpy con la forma del tensore nel GGUF (per l'export)."""
        if self.righe is not None:
            raise RuntimeError(f"{self.nome}: allenato su un sottoinsieme di esperti (prove), non esportabile")
        return self.master.detach().to(torch.float32).cpu().numpy().reshape(self.forma_file)


def _modulo_e_attributo(strato, hf: str):
    """Nome HF relativo allo strato -> (modulo, attributo) dove l'avanti lo legge."""
    if hf == "mlp.gate.weight":
        return strato.mlp, "router_w"                   # MoEStudente: il router e' `router_w` (+ `router_w32`)
    pref, _, attr = hf.rpartition(".")
    return strato.get_submodule(pref), attr


# ═══════════════════════════════════ caricamento dal GGUF ═══════════════════════════════════
def config_testo():
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(str(SRC_HF)).text_config
    cfg._attn_implementation = "sdpa"
    if any(v in os.getenv("E2E_KQAT", "") for v in ("flash1", "flash3", "attn1", "attn3")):   # 9/10 (spento: sdpa)
        import e2e_flash_qat as _FQ
        _FQ.installa(cfg)
    return cfg


def pesi_non_esperti(tens: dict, L: int, pieno: bool, salta=()) -> dict:
    """{nome HF relativo allo strato: tensore fp32} per TUTTI i tensori dello strato tranne gli esperti
    instradati. Stesso ciclo, stessa mappa e stesse funzioni di `carica_strato_gguf.pesi_strato_da_gguf`
    (il caricatore verificato), saltando le voci `per_esperto`. 28/9 `salta`: nomi GGUF senza "blk.N." da NON
    dequantizzare (--densi-ternari: quei Linear diventano `LineareTernaria` sui byte); vuoto = di sempre."""
    import carica_strato_gguf as CSG  # sola lettura
    from dequant_gguf_torch import dequant_np  # sola lettura
    mappa = CSG.MAPPA["full" if pieno else "linear"]
    out = {}
    for fam, voce in mappa.items():
        nome_g = f"blk.{L}.{fam}"
        if nome_g not in tens or voce.get("per_esperto") or fam in salta:
            continue
        W = dequant_np(tens[nome_g])
        if nome_g.endswith(".weight") and nome_p2(nome_g) in tens:      # 29/9 secondo piano: W1 + W2 (fp32)
            W = np.asarray(W, dtype=np.float32) + np.asarray(dequant_np(tens[nome_p2(nome_g)]), dtype=np.float32)
        if nome_g.endswith(".weight") and nome_p3(nome_g) in tens:      # 30/9 terzo piano: (W1 + W2) + W3 (fp32)
            W = np.asarray(W, dtype=np.float32) + np.asarray(dequant_np(tens[nome_p3(nome_g)]), dtype=np.float32)
        if nome_g.endswith(".weight") and nome_p4(nome_g) in tens:      # 9/10 quarto piano: ((W1+W2)+W3) + W4 (fp32)
            W = np.asarray(W, dtype=np.float32) + np.asarray(dequant_np(tens[nome_p4(nome_g)]), dtype=np.float32)
        # copia: per i tensori F32 `_applica` puo' restituire la vista del memmap (sola lettura)
        out[voce["hf"]] = torch.from_numpy(np.array(CSG._applica(voce, W), dtype=np.float32, copy=True))
    return out


def mappa_secondo_piano(tens: dict, L: int, nome_fam: str, E: int, E2: int) -> np.ndarray:
    """9/10 (cura P1): (E,) int64 esperto -> slot del 2° piano, -1 = nessuno. Con ffn_X_exps2_map (F32 [E]) la si legge;
    senza, prefisso dei primi E2 esperti (semantica del motore). Solleva su valori non interi o fuori campo."""
    nm = f"blk.{L}.{nome_fam}{SUFFISSO_E2}_map"
    if nm not in tens:
        return np.where(np.arange(E) < E2, np.arange(E), -1).astype(np.int64)
    v = np.asarray(tens[nm].data, dtype=np.float64).reshape(-1)
    if v.size != E or not np.all(np.isfinite(v)) or not np.all(v == np.round(v)) or v.min() < -1 or v.max() > E2 - 1:
        raise ValueError(f"{nm}: mappa non valida (len {v.size}, attesi {E}; campo [{v.min()}, {v.max()}], E2 {E2})")
    m = v.astype(np.int64)
    usati = m[m >= 0]
    if len(np.unique(usati)) != len(usati):
        raise ValueError(f"{nm}: due esperti sullo stesso slot del 2° piano")
    return m


def secondi_piani_esperti(tens: dict, L: int, fam: dict, device, esperti: Optional[slice] = None) -> dict:
    """9/10 (cura P1): {gate/up/down: (DatiFamiglia del 2° piano con E2 + 1 esperti, LongTensor (E,) esperto -> slot)}
    per i tensori ffn_X_exps2 PRESENTI nello strato L. L'esperto in piu' (indice E2) e' NULLO (byte 0, d0 0: peso 0
    esatto) e riceve le assegnazioni degli esperti senza 2° piano: uscita 0, come `mask` nel motore. Stesso tipo del
    primo piano (stessa rotazione dell'ingresso) e stesse righe/colonne, o si solleva."""
    out = {}
    for n, nome_fam in FAMIGLIE.items():
        nome2 = f"blk.{L}.{nome_fam}{SUFFISSO_E2}.weight"
        nm = f"blk.{L}.{nome_fam}{SUFFISSO_E2}_map"
        if nome2 not in tens:
            if nm in tens:
                raise ValueError(f"{nm} presente senza {nome2}")
            continue
        f1 = fam[n]
        f2 = famiglia_da_tensore(tens[nome2], f"{n}{SUFFISSO_E2}", "cpu")
        E2 = f2.E
        if f2.tipo != f1.tipo or (f2.R, f2.NBR) != (f1.R, f1.NBR):
            raise ValueError(f"{nome2}: {f2.tipo} ({f2.R}, {f2.NBR}), il primo piano e' {f1.tipo} ({f1.R}, {f1.NBR})")
        E_file = int(np.asarray(tens[f"blk.{L}.{nome_fam}.weight"].data).shape[0])
        m = mappa_secondo_piano(tens, L, nome_fam, E_file, E2)
        if esperti is not None:
            m = m[esperti]
        m = np.where(m >= 0, m, E2)                                  # nessun 2° piano -> esperto nullo
        blocchi = torch.cat([f2.blocchi, torch.zeros_like(f2.blocchi[:1])], 0)
        d0 = torch.cat([f2.d0, torch.zeros_like(f2.d0[:1])], 0)
        out[n] = (DatiFamiglia(f2.nome, f2.tipo, blocchi.to(device), d0.to(device), f2.strato),
                  torch.from_numpy(m).to(device))
    return out


def costruisci_strato(cfg, tens: dict, L: int, ric: Ricetta, mot, device, sentinella=None,
                      esperti: Optional[slice] = None, oss: Optional[Osservatore] = None, densi=()):
    """Qwen3_5MoeDecoderLayer coi non-esperti del GGUF (nessun peso bf16 originale: se ne manca uno si
    solleva) e `mlp` sostituito da `MoEStudente` sui byte degli esperti.
    28/9 `densi` (--densi-ternari; vuoto = di sempre, al bit): le famiglie dense ternarie chieste presenti nello strato
    NON si dequantizzano: i loro byte diventano famiglie a esperto unico in `StratoEsperti.fam` (dopo gate/up/down,
    ordine di FAMIGLIE_DENSE: theta, chiavi e registri le vedono come gli esperti) e il loro Linear una `LineareTernaria`."""
    from accelerate import init_empty_weights
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeDecoderLayer
    with init_empty_weights():
        strato = Qwen3_5MoeDecoderLayer(cfg, layer_idx=L)
    pieno = cfg.layer_types[L] == "full_attention"
    dense = voci_dense(tens, L, pieno, densi)
    if dense:
        sd = pesi_non_esperti(tens, L, pieno, salta={nome.split(".", 2)[2] for nome, _ in dense.values()})
    else:
        sd = pesi_non_esperti(tens, L, pieno)
    manca, inattesi = strato.load_state_dict(sd, strict=False, assign=True)
    hf_densi = {v["hf"] for _, v in dense.values()}
    manca = [k for k in manca if not k.startswith("mlp.experts.") and k not in hf_densi]
    if manca or inattesi:
        raise RuntimeError(f"strato {L}: pesi GGUF mancanti {manca} / inattesi {inattesi}")
    mlp = strato.mlp
    fam = {n: famiglia_da_tensore(tens[f"blk.{L}.{FAMIGLIE[n]}.weight"], n, device, esperti=esperti)
           for n in FAMIGLIE}
    for n, (nome_g, _) in dense.items():
        fam[n] = famiglia_da_tensore(tens[nome_g], n, device)
    router = mlp.gate.weight if esperti is None else torch.nn.Parameter(mlp.gate.weight[esperti].clone())
    es = StratoEsperti(fam, ric, mot, sentinella, oss)
    for n, (f2, m2) in secondi_piani_esperti(tens, L, fam, device, esperti).items():   # 9/10 cura P1
        es.fam2[n], es.mappa2[n] = f2, m2
    moe = MoEStudente(router, mlp.shared_expert, mlp.shared_expert_gate, es, cfg.num_experts_per_tok)
    router32 = router.detach().to(torch.float32).clone() if ric.router_fp32 else None   # F32 del GGUF, esatto
    strato.mlp = moe
    for n, (nome_g, voce) in dense.items():      # PRIMA di .to: il Linear da sostituire ha ancora il peso sul meta
        if n in FAMIGLIE_DENSE2 or n in FAMIGLIE_DENSE3 or n in FAMIGLIE_DENSE4:   # vivono nella LineareTernaria del primo
            continue
        padre, nome_lin = _padre_lineare(strato, voce["hf"])
        lin = getattr(padre, nome_lin)
        f = fam[n]
        if not isinstance(lin, torch.nn.Linear) or lin.bias is not None or (f.R, f.C) != (lin.out_features,
                                                                                         lin.in_features):
            raise RuntimeError(f"{nome_g}: {voce['hf']} ({type(lin).__name__} {getattr(lin, 'out_features', '?')}x"
                               f"{getattr(lin, 'in_features', '?')}) non e' il Linear del tensore ({f.R}x{f.C})")
        n2 = f"{n}{SUFFISSO_P2}"
        n3 = f"{n}{SUFFISSO_P3}"
        n4 = f"{n}{SUFFISSO_P4}"
        setattr(padre, nome_lin, LineareTernaria(es, n, voce, lin.out_features, lin.in_features,
                                                 nome2=n2 if n2 in dense else None,
                                                 nome3=n3 if n3 in dense else None,
                                                 nome4=n4 if n4 in dense else None))
    strato = strato.to(device=device, dtype=ric.dtype_residuo)
    if ric.non_esperti_x3:                       # 27/9 B3: DOPO .to (hi/lo bf16 non sono buffer: .to non li tocca)
        import e2e_bf16x3 as X3
        strato.lineari_x3 = X3.sostituisci_lineari(strato)
    elif ric.non_esperti_k2:                     # 27/9 Bk2 (braccio rumore del cancello B3)
        import e2e_bf16x3 as X3
        strato.lineari_x3 = X3.sostituisci_lineari(strato, classe=X3.LineareK2)
    if router32 is not None:
        strato.mlp.router_w32 = router32.to(device)
    for p in strato.parameters():
        p.requires_grad_(False)
    return strato.eval()


class TensoriTracciati(dict):
    """9/10 (cura P1): dizionario nome -> ReaderTensor che ANNOTA ogni lettura (`[]`, `get`). `in` non conta: chi chiede
    solo se un tensore esiste non lo usa. Serve al cancello di copertura (`controlla_copertura`)."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.letti: set = set()

    def __getitem__(self, k):
        v = super().__getitem__(k)
        self.letti.add(k)
        return v

    def get(self, k, default=None):
        if k in self:
            self.letti.add(k)
        return super().get(k, default)


# 9/10 (cura P1) LISTA BIANCA ESPLICITA del cancello di copertura: tensori del GGUF che il forward dello studente NON
# deve usare perche' il motore non li usa nel grafo misurato (logit del decodificatore principale). Ogni voce: (regola,
# motivo). Nient'altro e' estraneo: i residui _lra/_lrb gia' presenti nella PARTENZA non sono in lista (il motore li
# applica, lo studente no: si solleva) e nemmeno i piani in piu' sconosciuti (es. *5, exps3).
LISTA_BIANCA_COPERTURA = (
    ("blk.L.* con L >= n_strati del grafo principale", "strato nextn/MTP (blk.40): fuori dal forward dei logit"),
    ("blk.L.* con L non caricato", "studente parziale (prove su 1-2 strati): la copertura vale per gli strati caricati"),
)


def controlla_copertura(tens: dict, letti, strati, n_strati: int, usati_dopo=()) -> list:
    """9/10 (cura P1, fail-closed): ogni tensore del GGUF deve essere LETTO dalla costruzione dello studente, o letto piu'
    tardi da un cammino dichiarato (`usati_dopo`: embedding, testa, norma finale), o in LISTA_BIANCA_COPERTURA. Solleva
    ValueError con l'elenco degli orfani: un tensore ignorato in silenzio vuol dire allenare un modello diverso dal GGUF
    (DIAGNOSI_DIVARIO.md: 308 tensori, 3,06 GiB ignorati). Ritorna i nomi messi da parte dalla lista bianca."""
    letti = set(letti) | set(usati_dopo)
    strati = set(strati)
    orfani, bianchi = [], []
    for n in tens:
        if n in letti:
            continue
        m = re.match(r"blk\.(\d+)\.", n)
        if m and (int(m.group(1)) >= n_strati or int(m.group(1)) not in strati):
            bianchi.append(n)
            continue
        orfani.append(n)
    if orfani:
        raise ValueError(f"studente: {len(orfani)} tensori del GGUF non rappresentati nel forward (es. "
                         f"{sorted(orfani)[:8]}): il modello allenato non sarebbe il GGUF. Lista bianca: "
                         f"{[r for r, _ in LISTA_BIANCA_COPERTURA]}")
    return bianchi


class Studente:
    """Studente Thor-T dal GGUF. `strati`: gli indici caricati (tutti per l'addestramento; 1-2 per le prove).
    Checkpoint per strato: `avanti` (senza grafo) salva l'ingresso di ogni strato e il top-k; `indietro`
    ricalcola strato per strato con l'instradamento SALVATO e propaga il gradiente, che si accumula in theta."""

    def __init__(self, gguf: Path = GGUF_THOR_T, strati=range(40), ricetta: str = "B", motore_nome: str = "triton",
                 device: str = "cuda", blk: Optional[Blocchi] = None, esperti: Optional[slice] = None,
                 log: Callable = print, allena_f32=(), lora=None, densi_ternari=()):
        self.ric = RICETTE[ricetta]
        self.mot = motore(motore_nome, blk)
        self.device = torch.device(device)
        self._log = log                                 # 1/10: registro della cache di dequantizzazione
        self.cfg = config_testo()
        self.gguf = Path(gguf)
        self.sentinella = torch.zeros((), dtype=torch.bool, device=self.device)
        self.oss = Osservatore(self.device)
        r = GGUFReader(str(self.gguf))
        self._tens = TensoriTracciati({t.name: t for t in r.tensors})   # 9/10: letture annotate (copertura)
        self.strati = list(strati)
        self.moduli = {}
        # 28/9 --densi-ternari: famiglie dense ternarie allenate come esperto unico (vuota = identico al bit)
        self.densi_ternari = famiglie_dense(densi_ternari)
        kw_densi = {"densi": self.densi_ternari} if self.densi_ternari else {}
        t0 = time.time()
        for L in self.strati:
            self.moduli[L] = costruisci_strato(self.cfg, self._tens, L, self.ric, self.mot, self.device,
                                               self.sentinella, esperti, self.oss, **kw_densi)
        log(f"studente: {len(self.strati)} strati da {self.gguf.name}, ricetta {self.ric.nome}, motore "
            f"{self.mot.nome}, {time.time() - t0:.1f} s")
        self.densi_conteggi = {f: sum(1 for L in self.strati if f in self.esperti(L).fam) for f in self.densi_ternari}
        # 29/9 secondo piano: quanti *2 per famiglia (vuoto senza *2 nel GGUF: attributo nuovo, nessun effetto)
        self.p2_conteggi = {f"{f}{SUFFISSO_P2}": c for f in self.densi_ternari
                            if (c := sum(1 for L in self.strati if f"{f}{SUFFISSO_P2}" in self.esperti(L).fam))}
        if self.densi_ternari:
            n_g = sum(int(self.esperti(L).fam[f].d0.numel()) for L in self.strati for f in self.densi_ternari
                      if f in self.esperti(L).fam)
            log(f"studente: densi ternari (esperto unico) {self.densi_conteggi}, {n_g} gruppi di scala")
            if self.p2_conteggi:
                n_g2 = sum(int(self.esperti(L).fam[f].d0.numel()) for L in self.strati for f in self.p2_conteggi
                           if f in self.esperti(L).fam)
                log(f"studente: secondo piano dei densi {self.p2_conteggi}, {n_g2} gruppi di scala in piu'")
        # 30/9 terzo piano: quanti *3 per famiglia (vuoto senza *3 nel GGUF: nessun effetto)
        self.p3_conteggi = {f"{f}{SUFFISSO_P3}": c for f in self.densi_ternari
                            if (c := sum(1 for L in self.strati if f"{f}{SUFFISSO_P3}" in self.esperti(L).fam))}
        if self.p3_conteggi:
            n_g3 = sum(int(self.esperti(L).fam[f].d0.numel()) for L in self.strati for f in self.p3_conteggi
                       if f in self.esperti(L).fam)
            log(f"studente: terzo piano dei densi {self.p3_conteggi}, {n_g3} gruppi di scala in piu'")
        # 9/10 (cura P1) quarto piano dei densi e secondo piano degli esperti: conteggi nel registro
        self.p4_conteggi = {f"{f}{SUFFISSO_P4}": c for f in self.densi_ternari
                            if (c := sum(1 for L in self.strati if f"{f}{SUFFISSO_P4}" in self.esperti(L).fam))}
        if self.p4_conteggi:
            log(f"studente: quarto piano dei densi {self.p4_conteggi}")
        self.e2_conteggi = {n: c for n in FAMIGLIE if (c := sum(1 for L in self.strati if n in self.esperti(L).fam2))}
        if self.e2_conteggi:
            n_e2 = {n: sum(int((self.esperti(L).mappa2[n] < self.esperti(L).fam2[n].E - 1).sum())
                           for L in self.strati if n in self.esperti(L).fam2) for n in self.e2_conteggi}
            log(f"studente: secondo piano degli esperti, strati {self.e2_conteggi}, esperti coperti {n_e2}")
        self.norma_finale = torch.from_numpy(np.array(self._tens["output_norm.weight"].data, dtype=np.float32)) \
            .to(self.device)
        self.eps = float(self.cfg.rms_norm_eps)
        self._emb = None
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextRotaryEmbedding
        self.rotary = Qwen3_5MoeTextRotaryEmbedding(config=self.cfg).to(self.device)
        self.allena_f32 = famiglie_f32(allena_f32)
        self.f32: dict = {}                       # nome GGUF -> TensoreF32, in ordine canonico (strato, famiglia)
        self._f32_strato: dict = {}
        self.f32_conteggi = {f: 0 for f in self.allena_f32}
        if self.allena_f32:
            self._prepara_f32(esperti)
            log(f"studente: F32 allenati {self.f32_conteggi} ({sum(t.master.numel() for t in self.f32.values())} "
                f"valori)")
        # 27/9 --lora-nonesperti (e2e_lora.py): adattatori B A sui k-quant non esperti. None o rango 0 = nessuno
        # (nessun modulo sostituito: identico al bit); con rango > 0, B = 0 e l'avanti resta quello di sempre al passo 0.
        self.lora: dict = {}                      # nome GGUF -> e2e_lora.AdattatoreLoRA, ordine canonico (strato, famiglia)
        self.lora_cfg = None
        self.lora_conteggi: dict = {}
        if lora is not None and lora.rango > 0:
            import e2e_lora as LR
            LR.installa(self, lora)
            log(f"studente: LoRA rango {lora.rango} (scala {lora.scala:g}) su {self.lora_conteggi} "
                f"({sum(p.numel() for a in self.lora.values() for p in a.parametri())} valori)")
        # 27/9 ramo dominio-strati (--strati-allenati, `congela_strati`): None = tutti gli strati allenati (di sempre)
        self.strati_allenati: Optional[list] = None
        # 9/10 (cura P1) cancello di copertura, fail-closed: embedding e testa si leggono piu' tardi (piani_di)
        dopo = piani_di(self._tens, "token_embd.weight") + piani_di(self._tens, "output.weight") + ["output_norm.weight"]
        bianchi = controlla_copertura(self._tens, self._tens.letti, self.strati, int(self.cfg.num_hidden_layers), dopo)
        log(f"studente: copertura del GGUF completa ({len(self._tens) - len(bianchi)} tensori usati, {len(bianchi)} in "
            f"lista bianca)")

    # ── strati congelati (27/9, ramo dominio-strati, --strati-allenati)
    def congela_strati(self, allenati=None) -> None:
        """`allenati` = indici degli strati allenati; None, vuoto o TUTTI gli strati caricati = nessuno strato congelato
        (requires_grad di tutto torna vero: lo studente di sempre, identico al bit). Negli strati congelati theta, i master
        F32 e A/B della LoRA hanno requires_grad falso: nessun gradiente (grad resta None, AdamW e Sophia li saltano:
        stato dell'ottimizzatore al bit), nessuna statistica del passo V (dL/dd non si calcola: `_indietro_famiglia`
        serve_d falso) e l'indietro si ferma al primo strato allenato (`indietro`: niente dL/dx sotto). I tensori F32
        GLOBALI (output_norm, sopra l'ultimo strato) non sono di nessuno strato: restano allenati se chiesti."""
        s = sorted({int(L) for L in (allenati or ())})
        fuori = [L for L in s if L not in self.strati]
        if fuori:
            raise ValueError(f"strati allenati {fuori} non caricati nello studente (strati {self.strati})")
        self.strati_allenati = s if s and set(s) != set(self.strati) else None
        on = set(self.strati if self.strati_allenati is None else self.strati_allenati)
        for L in self.strati:
            for t in self.esperti(L).theta.values():
                t.requires_grad_(L in on)
            for tf in self._f32_strato.get(L, ()):
                tf.master.requires_grad_(L in on)
        for a in self.lora.values():
            for p in a.parametri():
                p.requires_grad_(a.L in on)

    def allenato(self, L: int) -> bool:
        """Lo strato L e' allenato (--strati-allenati; senza congelamento: tutti; L = -1, i F32 globali: sempre)."""
        return self.strati_allenati is None or L < 0 or L in self.strati_allenati

    # ── LoRA sui non-esperti (e2e_lora.py)
    def chiavi_lora(self) -> list:
        return list(self.lora)

    def parametri_lora(self) -> list:
        """[A, B] per adattatore, nell'ordine di `chiavi_lora`: e' l'ordine dello stato di Adam del gruppo LoRA."""
        return [p for a in self.lora.values() for p in a.parametri()]

    def valori_lora(self) -> dict:
        """{nome GGUF: {"A": ..., "B": ...}} (copie staccate, CPU): ripresa e istantanea. 27/9 LoRA-GA: anche "A0",
        "B0" (valori iniziali fermi) per gli adattatori in forma GA; senza GA il dizionario e' quello di prima."""
        return {n: a.valori() for n, a in self.lora.items()}

    @torch.no_grad()
    def imposta_lora(self, valori: dict) -> None:
        """A, B (e A0, B0 della LoRA-GA se presenti: la forma dell'avanti la decidono i DATI) di ogni adattatore."""
        for n, v in valori.items():
            self.lora[n].imposta(v)

    def lora_per_export(self) -> dict:
        """{nome del tensore GGUF: Delta W = s B A fp32 nella forma e nell'ordine del FILE}."""
        return {n: a.delta_file() for n, a in self.lora.items()}

    @contextmanager
    def lora_fermi(self):
        import e2e_lora as LR
        with LR.fermi(self.lora.values()):
            yield

    def _prepara_f32(self, esperti: Optional[slice]) -> None:
        """--allena-f32: per ogni strato caricato e famiglia chiesta presente nel suo tipo, il valore del file diventa
        `TensoreF32.master` e il parametro del modulo si ricalcola da li'. Alla costruzione il valore ricalcolato e'
        verificato AL BIT contro quello del caricatore di sempre (stessa via di `pesi_non_esperti`)."""
        import carica_strato_gguf as CSG  # sola lettura
        from dequant_gguf_torch import dequant_np  # sola lettura
        dt_m = torch.float64 if self.ric.dtype_residuo == torch.float64 else torch.float32
        for L in self.strati:
            mappa = CSG.MAPPA["full" if self.cfg.layer_types[L] == "full_attention" else "linear"]
            strato = self.moduli[L]
            for fam in self.allena_f32:
                if fam in F32_GLOBALI or FAMIGLIE_F32[fam] not in mappa:
                    continue
                nome = f"blk.{L}.{FAMIGLIE_F32[fam]}"
                t = self._tens.get(nome)
                if t is None:
                    continue
                if t.tensor_type.name != "F32":
                    raise ValueError(f"{nome}: tipo {t.tensor_type.name}, non F32")
                voce = mappa[FAMIGLIE_F32[fam]]
                mod, attr = _modulo_e_attributo(strato, voce["hf"])
                vecchio = getattr(mod, attr)
                righe = esperti if fam == "ffn_gate_inp" else None
                tf = TensoreF32(nome, fam, L, voce, dequant_np(t), self.device, dt_m, mod, attr, vecchio.dtype, righe)
                with torch.no_grad():
                    if tf._numpy:
                        a = CSG._applica(voce, tf.master.detach().to(torch.float32).cpu().numpy())
                        rif = torch.from_numpy(np.array(a, copy=True))
                    else:
                        rif = _trasforma_torch(tf.master.detach().to(torch.float32), voce, tf.inv)
                    if not torch.equal(rif.to(device=vecchio.device, dtype=vecchio.dtype), vecchio.detach()):
                        raise RuntimeError(f"{nome}: il valore ricalcolato dal master non e' quello del caricatore")
                    if attr == "router_w" and mod.router_w32 is not None and not torch.equal(
                            rif.to(mod.router_w32.device), mod.router_w32):
                        raise RuntimeError(f"{nome}: router fp32 ricalcolato diverso")
                if attr in mod._parameters:
                    del mod._parameters[attr]              # da qui e' un attributo semplice, ricalcolato da master
                tf.installa()
                self.f32[nome] = tf
                self._f32_strato.setdefault(L, []).append(tf)
                self.f32_conteggi[fam] += 1
        if "output_norm" in self.allena_f32:
            nome = F32_GLOBALI["output_norm"]
            tf = TensoreF32(nome, "output_norm", -1, None, np.asarray(self._tens[nome].data), self.device, dt_m)
            if not torch.equal(tf.master.detach().to(torch.float32), self.norma_finale):
                raise RuntimeError(f"{nome}: master diverso da norma_finale")
            self.norma_finale = tf.master                   # esce() lo legge: gradiente diretto
            self.f32[nome] = tf
            self.f32_conteggi["output_norm"] += 1

    def chiavi_f32(self) -> list:
        return list(self.f32)

    def parametri_f32(self) -> list:
        return [t.master for t in self.f32.values()]

    def valori_f32(self) -> dict:
        """{nome GGUF: master (copia staccata, CPU)}: ripresa e istantanea."""
        return {n: t.master.detach().cpu().clone() for n, t in self.f32.items()}

    @torch.no_grad()
    def imposta_f32(self, valori: dict) -> None:
        """Valori dei master: dalla ripresa (forma del master) o dall'export (forma del file; con un sottoinsieme
        di esperti il router si affetta come alla costruzione)."""
        for n, v in valori.items():
            tf = self.f32[n]
            v = torch.as_tensor(np.asarray(v) if isinstance(v, np.ndarray) else v)
            if tf.righe is not None and tuple(v.shape) == tf.forma_file:
                v = v[tf.righe]
            tf.master.copy_(v.reshape(tf.master.shape).to(device=tf.master.device, dtype=tf.master.dtype))

    def f32_per_export(self) -> dict:
        """{nome del tensore GGUF: fp32 numpy nella forma del file}: i byte che l'export scrive al posto dei vecchi."""
        return {n: t.valore_file() for n, t in self.f32.items()}

    def valori_f32_gguf(self, nomi=None) -> dict:
        """28/9 --f32-ripristina: {nome: valore del FILE GGUF} dei tensori F32 allenabili (tutti, o `nomi`), letto come
        alla costruzione (`_prepara_f32`: dequant_np per i tensori di strato, i dati grezzi per output_norm), nella forma
        del file: `imposta_f32` lo riporta nel master esattamente come la costruzione (stesso bit)."""
        from dequant_gguf_torch import dequant_np  # sola lettura
        out = {}
        for n in (self.f32 if nomi is None else nomi):
            famiglia_del_tensore_f32(n)                  # solleva per un nome che non e' di una famiglia F32
            t = self._tens.get(n)
            if t is None or t.tensor_type.name != "F32":
                raise KeyError(f"{n}: tensore F32 assente dal GGUF dello studente")
            out[n] = np.asarray(t.data) if n in F32_GLOBALI.values() else dequant_np(t)
        return out

    @contextmanager
    def f32_fermi(self):
        """Per la durata del blocco i master non raccolgono gradiente (curvatura del passo V: dL/dd = 0 anche qui)."""
        prima = [t.master.requires_grad for t in self.f32.values()]
        try:
            for t in self.f32.values():
                t.master.requires_grad_(False)
            yield
        finally:
            for t, r in zip(self.f32.values(), prima):
                t.master.requires_grad_(r)

    # ── parametri
    def chiavi(self) -> list:
        """(strato, famiglia) nell'ordine dei parametri: gate, up, down di ogni strato (di sempre) e, con --densi-ternari,
        subito dopo le famiglie dense presenti nello strato (ordine di StratoEsperti.fam, cioe' di FAMIGLIE_DENSE)."""
        return [(L, n) for L in self.strati for n in self.esperti(L).fam]

    def esperti(self, L: int) -> StratoEsperti:
        return self.moduli[L].mlp.esperti

    def parametri(self) -> list:
        return [self.esperti(L).theta[n] for (L, n) in self.chiavi()]

    @torch.no_grad()
    def scale_fp16(self) -> dict:
        """{(strato, famiglia): scale fp16 (E, R, NG)} ESATTE del forward attuale: quelle da esportare
        (ordine dei blocchi del GGUF: esperto, riga, blocco della riga, scala nel blocco)."""
        return {(L, n): self.esperti(L).scala(n).to(torch.float16).clone() for (L, n) in self.chiavi()}

    def scale_per_export(self) -> dict:
        """{nome del tensore GGUF: scale fp16 numpy (n_blocchi, scale_per_blocco)}: la forma che
        `e2e_codici_scale.patcha_scale` si aspetta (stesso ordine dei blocchi del file, prova S9)."""
        out = {}
        for (L, n), s in self.scale_fp16().items():
            fam = self.esperti(L).fam[n]
            out[f"blk.{L}.{nome_tensore_famiglia(n)}.weight"] = s.cpu().numpy().reshape(-1, CS.LAYOUT[fam.tipo][3])
        return out

    def controlla(self, tollera_trabocco: bool = False) -> dict:
        """UNA sincronizzazione per passo (26/9: prima una per strato): scale non finite (solleva), trabocco
        fp16 negli ingressi/gradienti degli esperti (solleva, salvo `tollera_trabocco`: chi allena salta il
        passo e dimezza K; il gradiente di quel passo NON e' valido) e cambi d'instradamento. Azzera il
        trabocco dopo averlo letto."""
        cs = [self.moduli[L].mlp.cambi for L in self.strati if self.moduli[L].mlp.cambi is not None]
        cambi = torch.stack(cs).sum() if cs else torch.zeros((), device=self.device, dtype=torch.long)
        voci = [self.sentinella.long(), self.oss.trabocco.long(), cambi.to(self.sentinella.device).long()]
        if self.f32:                       # --allena-f32: valori finiti (e ssm_a < 0: A_log = log(-ssm_a)), stessa sync
            ko = [(~torch.isfinite(t.master)).any() | ((t.master >= 0).any() if t.famiglia == "ssm_a" else False)
                  for t in self.f32.values()]
            voci.append(torch.stack(ko).any().to(self.sentinella.device).long())
        i_lora = None
        if self.lora:                      # --lora-nonesperti: A e B finiti, stessa sync
            i_lora = len(voci)
            voci.append(torch.stack([~torch.isfinite(p).all() for p in self.parametri_lora()]).any()
                        .to(self.sentinella.device).long())
        letti = torch.stack(voci).tolist()
        scala_ko, trab, n_cambi = letti[:3]
        self.oss.trabocco.zero_()
        if scala_ko:
            raise FloatingPointError("una scala e' uscita dal campo fp16")
        if self.f32 and letti[3]:
            cattivi = [n for n, t in self.f32.items() if not bool(torch.isfinite(t.master).all())
                       or (t.famiglia == "ssm_a" and bool((t.master >= 0).any()))]
            raise FloatingPointError(f"tensori F32 allenati non finiti (o ssm_a >= 0): {cattivi[:8]}")
        if i_lora is not None and letti[i_lora]:
            cattivi = [n for n, a in self.lora.items() if not all(bool(torch.isfinite(p).all()) for p in a.parametri())]
            raise FloatingPointError(f"adattatori LoRA non finiti: {cattivi[:8]}")
        if trab and not tollera_trabocco:
            raise FloatingPointError("trabocco fp16 negli ingressi o nei gradienti degli esperti (K troppo grande?)")
        return {"cambi_instradamento": int(n_cambi), "trabocco_fp16": bool(trab)}

    def cambi(self) -> int:
        cs = [self.moduli[L].mlp.cambi for L in self.strati if self.moduli[L].mlp.cambi is not None]
        return int(torch.stack(cs).sum()) if cs else 0

    def azzera_cambi(self) -> None:
        for L in self.strati:
            self.moduli[L].mlp.cambi = None

    def diagnostica_indietro(self, accesa: bool = True) -> list:
        """Cancello G1 (passo 0): accende/spegne la raccolta dell'Osservatore e restituisce le righe raccolte
        finora (quote a zero / subnormali in fp16 del gradiente scalato per K, per strato e famiglia)."""
        righe = self.oss.riassunto()
        self.oss.diagnostica = accesa
        return righe

    # ── ingresso e uscita
    def embedding(self, ids: torch.Tensor) -> torch.Tensor:
        """Righe di token_embd (Q5_K) dequantizzate come get_rows di llama.cpp (fp32), poi dtype_residuo.
        Con `carica_embedding()` la tabella intera sta sul dispositivo; altrimenti si dequantizzano le sole
        righe presenti (CPU, prove)."""
        if self._emb is not None:
            return self._emb[ids.to(self.device)].to(self.ric.dtype_residuo)
        from dequant_gguf_torch import _dequant_raw
        unici = torch.unique(ids.cpu())
        piani = piani_di(self._tens, "token_embd.weight")
        t = self._tens[piani[0]]
        righe = torch.from_numpy(_dequant_raw(np.asarray(t.data)[unici.numpy()], t))
        for nome in piani[1:]:                  # 30/9: + token_embd2/3 (get_rows del motore in fp32, sommati in ordine)
            t = self._tens[nome]
            righe = righe.to(torch.float32) + torch.from_numpy(
                np.asarray(_dequant_raw(np.asarray(t.data)[unici.numpy()], t), dtype=np.float32))
        pos = torch.searchsorted(unici, ids.cpu())
        return righe[pos].to(device=self.device, dtype=self.ric.dtype_residuo)

    def carica_embedding(self) -> None:
        from dequant_gguf_torch import dequant_np
        self._emb = torch.from_numpy(self._somma_piani("token_embd.weight")).to(self.device)

    def _somma_piani(self, nome_g: str) -> np.ndarray:
        """30/9: dequantizzazione fp32 di una tabella con i suoi piani in piu' (<nome>2, <nome>3) sommati nell'ordine del
        motore. Senza piani in piu' e' `dequant_np` di sempre (stesso array)."""
        from dequant_gguf_torch import dequant_np
        piani = piani_di(self._tens, nome_g)

        def calcola():
            W = dequant_np(self._tens[piani[0]])
            for p in piani[1:]:
                W = np.asarray(W, dtype=np.float32) + np.asarray(dequant_np(self._tens[p]), dtype=np.float32)
            return W
        cache = os.environ.get("E2E_CACHE_DEQUANT", "").strip()
        if cache:                               # 1/10 tt5-ottimizzato: stesso array dalla cache (e2e_cache_dequant)
            import e2e_cache_dequant as CD
            return CD.somma_piani(self._tens, nome_g, piani, Path(cache), calcola, log=getattr(self, "_log", print))
        return calcola()

    def testa(self, dtype=torch.float32) -> torch.Tensor:
        """output.weight (Q5_K) dequantizzata: la testa dello STUDENTE (quella del teacher e' la bf16)."""
        return torch.from_numpy(self._somma_piani("output.weight")).to(device=self.device, dtype=dtype)   # 30/9: + output2/3

    def esce(self, x: torch.Tensor) -> torch.Tensor:
        """output_norm come llama.cpp: rmsnorm in fp32 per il peso del GGUF (che contiene gia' 1 + w)."""
        acc = torch.float64 if x.dtype == torch.float64 else torch.float32
        xf = x.to(acc)
        h = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps) * self.norma_finale.to(acc)
        return h.reshape(-1, x.shape[-1])

    # ── avanti e indietro con checkpoint per strato
    def _contesto(self, B: int, T: int, dtype, inizi: Optional[torch.Tensor] = None) -> dict:
        """Contesto posizionale. `inizi` None = modo «riga» (la riga e' UNA sequenza causale, posizioni 0..T-1,
        come `e2e_teacher --modo riga` e la forgia). `inizi` (B, T) 0/1 con 1 all'inizio di ogni segmento dei
        cu_seqlens = modo «segmenti» (come `e2e_teacher --modo segmenti` / `e2e_dump_logit --segmenti`): ogni
        segmento e' una sequenza a se', posizioni da 0 e memoria (GDN) azzerata. Chi allena usa lo STESSO modo
        del teacher (`CacheMaestro.modo`)."""
        from transformers.masking_utils import create_causal_mask

        def pos_e_maschera(b: int, t: int):
            pos4 = torch.arange(t, device=self.device).view(1, 1, -1).expand(4, b, -1)
            testo, pos = pos4[0], pos4[1:]
            finto = torch.zeros(b, t, self.cfg.hidden_size, device=self.device, dtype=dtype)
            pe = self.rotary(finto, pos)
            m = create_causal_mask(config=self.cfg, inputs_embeds=finto, attention_mask=None,
                                   past_key_values=None, position_ids=testo)
            return pe, m, testo

        if inizi is None:
            pe, m, testo = pos_e_maschera(B, T)
            return {"modo": "riga", "pe": pe, "maschera": m, "testo": testo}
        inizi = inizi.cpu()
        if inizi.shape != (B, T) or not bool((inizi[:, 0] == 1).all()):
            raise ValueError(f"inizi: forma {tuple(inizi.shape)} (attesa {(B, T)}), il primo token deve aprire")
        segmenti, per_lunghezza = [], {}
        for b in range(B):
            cu = torch.nonzero(inizi[b]).reshape(-1).tolist() + [T]
            for s0, s1 in zip(cu[:-1], cu[1:]):
                segmenti.append((b, s0, s1))
                if s1 - s0 not in per_lunghezza:
                    per_lunghezza[s1 - s0] = pos_e_maschera(1, s1 - s0)
        return {"modo": "segmenti", "segmenti": segmenti, "per_lunghezza": per_lunghezza, "B": B, "T": T}

    def _miscela_segmenti(self, modulo, tipo: str, h, ctx):
        """Il miscelatore di token (attenzione piena o GDN) segmento per segmento; il resto dello strato
        (norme, MoE, residui) resta su tutti i token insieme."""
        righe = []
        for b in range(ctx["B"]):
            pezzi = []
            for (bb, s0, s1) in ctx["segmenti"]:
                if bb != b:
                    continue
                hs = h[b:b + 1, s0:s1]
                if tipo == "linear_attention":
                    o = modulo.linear_attn(hidden_states=hs, cache_params=None, attention_mask=None)
                else:
                    pe, m, testo = ctx["per_lunghezza"][s1 - s0]
                    o, _ = modulo.self_attn(hidden_states=hs, attention_mask=m, position_ids=testo,
                                            past_key_values=None, position_embeddings=pe)
                pezzi.append(o)
            righe.append(torch.cat(pezzi, dim=1))
        return torch.cat(righe, dim=0)

    def _strato(self, L: int, x, ctx, instr=None):
        modulo = self.moduli[L]
        for tf in self._f32_strato.get(L, ()):
            tf.installa()                  # --allena-f32: parametri ricalcolati dai master (con grafo nell'indietro)
        modulo.mlp.instr = instr
        tipo = self.cfg.layer_types[L]
        try:
            if ctx["modo"] == "riga":
                o = modulo(x, position_embeddings=ctx["pe"],
                           attention_mask=ctx["maschera"] if tipo == "full_attention" else None,
                           position_ids=ctx["testo"])
                return o[0] if isinstance(o, tuple) else o
            # modo segmenti: STESSA sequenza di Qwen3_5MoeDecoderLayer.forward (transformers 5.15), col solo
            # miscelatore spezzato (prova S10: un segmento per riga == modo riga, al bit)
            h = modulo.input_layernorm(x)
            x = x + self._miscela_segmenti(modulo, tipo, h, ctx)
            h = modulo.mlp(modulo.post_attention_layernorm(x))
            return x + (h[0] if isinstance(h, tuple) else h)
        finally:
            modulo.mlp.instr = None

    @torch.no_grad()
    def avanti(self, ids: torch.Tensor, x0: Optional[torch.Tensor] = None,
               inizi: Optional[torch.Tensor] = None, grafo: bool = False) -> dict:
        """ids (B, T). Nessun grafo. Ritorna {"xL": uscita dell'ultimo strato, "ingressi": [...],
        "instr": [top-k per strato], "ctx": contesto}. `x0`: ingresso esplicito (prove). `inizi`: vedi
        `_contesto` (modo segmenti)."""
        x = self.embedding(ids) if x0 is None else x0
        B, T = x.shape[:2]
        ctx = self._contesto(B, T, x.dtype, inizi)
        ingressi, instr = [], []
        if grafo and os.environ.get("E2E_SENZA_RICALCOLO") == "1" and self.strati_allenati is not None:
            # 21/10 (Angelo: velocizza): il grafo degli strati allenati si TIENE nell'avanti invece di ricalcolarlo
            # nell'indietro (profilo: indietro 82 s = ricalcolo + indietro, avanti 20 s; VRAM usata 25 di 88 GiB).
            # Stessi calcoli, salvati invece che rifatti. Sotto il primo strato allenato: nessun grafo, come prima.
            # E2E_GRAFO_STRATI=N (21/10 20:1x, dopo OOM a 52 GiB col grafo di tutti gli strati): grafo tenuto solo per gli
            # N strati PIU' ALTI; sotto (da i0 a is_-1) ingressi salvati e ricalcolo come _indietro_troncato.
            on = set(self.strati_allenati)
            i0 = min(i for i, L in enumerate(self.strati) if L in on)
            n_g = int(os.environ.get("E2E_GRAFO_STRATI", "0")) or len(self.strati)
            is_ = max(i0, len(self.strati) - n_g)
            ingressi = [None] * len(self.strati)
            for i, L in enumerate(self.strati):
                if i < is_:
                    ingressi[i] = x
                    x = self._strato(L, x, ctx)
                else:
                    if i == is_ and is_ > i0:
                        ingressi[i] = x
                        x = x.detach().requires_grad_(True)     # foglia: qui arriva dL/dx per il ricalcolo sotto
                        xs = x
                    with torch.enable_grad():
                        x = self._strato(L, x, ctx)
                instr.append(self.moduli[L].mlp.ultimo_idx)
            return {"xL": x, "ingressi": ingressi, "instr": instr, "ctx": ctx, "grafo": True,
                    "is_": is_, "i0": i0, "xs": xs if is_ > i0 else None}
        for L in self.strati:
            ingressi.append(x)
            x = self._strato(L, x, ctx)
            instr.append(self.moduli[L].mlp.ultimo_idx)
        return {"xL": x, "ingressi": ingressi, "instr": instr, "ctx": ctx}

    def indietro(self, stato: dict, g: torch.Tensor) -> torch.Tensor:
        """g = dL/dxL. Ricalcolo strato per strato con il top-k salvato; accumula dL/dtheta. Ritorna dL/dx0.
        27/9 --strati-allenati (`congela_strati`): `_indietro_troncato` (si ferma al primo strato allenato, ritorna None)."""
        if stato.get("grafo"):                     # E2E_SENZA_RICALCOLO: backward sul grafo tenuto, poi ricalcolo sotto
            if stato["xL"].requires_grad:
                stato["xL"].backward(g)
            if stato.get("xs") is None:
                return None
            g = stato["xs"].grad
            if g is None:
                raise RuntimeError("E2E_GRAFO_STRATI: dL/dx assente al confine del grafo")
            i0, is_ = stato["i0"], stato["is_"]
            for i in reversed(range(i0, is_)):            # stesso corpo di _indietro_troncato
                L = self.strati[i]
                xin = stato["ingressi"][i].detach()
                if i > i0:
                    xin.requires_grad_(True)
                with torch.enable_grad():
                    y = self._strato(L, xin, stato["ctx"], instr=stato["instr"][i])
                    if y.requires_grad:
                        y.backward(g)
                if i > i0:
                    g = xin.grad
                    if g is None:
                        raise RuntimeError(f"strato {L}: dL/dx assente nell'indietro troncato")
            return None
        if self.strati_allenati is not None:
            return self._indietro_troncato(stato, g)
        for i in reversed(range(len(self.strati))):
            L = self.strati[i]
            xin = stato["ingressi"][i].detach().requires_grad_(True)
            with torch.enable_grad():
                y = self._strato(L, xin, stato["ctx"], instr=stato["instr"][i])
                y.backward(g)
            g = xin.grad
        return g

    def _indietro_troncato(self, stato: dict, g: torch.Tensor) -> None:
        """27/9 --strati-allenati: come `indietro` ma dall'ultimo strato fino al PRIMO strato allenato (in ordine di
        avanti) e non oltre. Gli strati congelati sopra di lui propagano solo dL/dx (i loro parametri non chiedono
        gradiente); nel primo strato allenato l'ingresso non chiede gradiente (niente dL/dx: nessuno sotto lo userebbe);
        sotto, nessun ricalcolo. I gradienti degli strati allenati sono gli STESSI calcoli dell'indietro intero (prova
        D3 di e2e_test_dominio_strati: al bit). Ritorna None (dL/dx0 non esiste piu')."""
        on = set(self.strati_allenati)
        i0 = min(i for i, L in enumerate(self.strati) if L in on)
        for i in reversed(range(i0, len(self.strati))):
            L = self.strati[i]
            xin = stato["ingressi"][i].detach()
            if i > i0:
                xin.requires_grad_(True)
            with torch.enable_grad():
                y = self._strato(L, xin, stato["ctx"], instr=stato["instr"][i])
                if y.requires_grad:
                    y.backward(g)
            if i > i0:
                g = xin.grad
                if g is None:                      # nessun cammino verso l'ingresso (non accade: c'e' il residuo)
                    raise RuntimeError(f"strato {L}: dL/dx assente nell'indietro troncato")
        return None

    def gradiente(self, ids: torch.Tensor, perdita: Callable, x0: Optional[torch.Tensor] = None,
                  inizi: Optional[torch.Tensor] = None) -> float:
        """Passo completo: avanti senza grafo, perdita(h) su h = esce(xL) (scalare, GIA' scalata da chi
        chiama: la scala K di G1), indietro con checkpoint. Ritorna il valore della perdita."""
        stato = self.avanti(ids, x0, inizi)
        xL = stato["xL"].detach().requires_grad_(True)
        with torch.enable_grad():
            val = perdita(self.esce(xL))
            val.backward()
        self.indietro(stato, xL.grad)
        return float(val.detach())

    def nascosto(self, ids: torch.Tensor, x0: Optional[torch.Tensor] = None,
                 inizi: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Uscita dopo output_norm, con grafo intero (solo prove: riferimento del checkpoint)."""
        x = self.embedding(ids) if x0 is None else x0
        ctx = self._contesto(x.shape[0], x.shape[1], x.dtype, inizi)
        for L in self.strati:
            x = self._strato(L, x, ctx)
        return self.esce(x)


# ═══════════════════════════════════ misura (M0, GPU) ═══════════════════════════════════
def _sincronizza():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# Configurazioni dei riquadri con 0 registri riversati su TUTTE le varianti (compilazione anticipata gfx1151,
# 26/9, e2e_test_studente S8 per la prima): le candidate della spazzata di M0.
CANDIDATE_GEMM = [(64, 64, 32, 8), (64, 32, 32, 4), (128, 64, 32, 8), (128, 64, 16, 4), (128, 128, 16, 8),
                  (64, 128, 16, 8), (128, 32, 32, 4)]
CANDIDATE_DD = [(64, 16, 8), (32, 16, 8), (16, 32, 8), (16, 16, 4)]


def _ids_reali(B: int, T: int) -> torch.Tensor:
    """Righe vere del corpus E2E (l'instradamento, cioe' il carico per esperto, dipende dal testo)."""
    c = torch.load("@@ROOT:models@@/gguf/tony-tern/calibrazione/e2e/e2e_corpus_4096_tutte.pt", weights_only=True)
    righe = torch.cat([c["domini"][d] for d in c["nomi_domini"]])
    return righe[:B, :T].clone()


def _misura_passo(st: "Studente", ids: torch.Tensor, ripetizioni: int) -> dict:
    """Avanti senza grafo + ricalcolo e indietro, strato per strato, con sincronizzazioni SOLO qui (misura)."""
    per = {L: {"avanti": [], "indietro": []} for L in st.strati}
    tot = []
    for rip in range(ripetizioni + 1):
        _sincronizza()
        t_in = time.time()
        with torch.no_grad():
            x = st.embedding(ids)
            ctx = st._contesto(x.shape[0], x.shape[1], x.dtype)
            ingressi, instr = [], []
            for L in st.strati:
                ingressi.append(x)
                _sincronizza()
                t0 = time.time()
                x = st._strato(L, x, ctx)
                _sincronizza()
                per[L]["avanti"].append(time.time() - t0)
                instr.append(st.moduli[L].mlp.ultimo_idx)
        xL = x.detach().requires_grad_(True)
        with torch.enable_grad():
            (st.esce(xL).float().pow(2).mean() * 1e3).backward()
        g = xL.grad
        for i in reversed(range(len(st.strati))):
            L = st.strati[i]
            _sincronizza()
            t0 = time.time()
            xin = ingressi[i].detach().requires_grad_(True)
            with torch.enable_grad():
                y = st._strato(L, xin, ctx, instr=instr[i])
                y.backward(g)
            g = xin.grad
            _sincronizza()
            per[L]["indietro"].append(time.time() - t0)
        tot.append(time.time() - t_in)
        if rip == 0:                                              # la prima compila i kernel: si scarta
            for L in st.strati:
                per[L]["avanti"].clear()
                per[L]["indietro"].clear()
            tot.clear()
        for p_ in st.parametri():
            p_.grad = None
    return {"per_strato_s": {L: {k: min(v) for k, v in d.items()} for L, d in per.items()}, "passo_s": min(tot)}


def _spazza_blocchi(st: "Studente", ids: torch.Tensor) -> dict:
    """Tempo di UNA chiamata avanti + dX + dL/dd per famiglia (gate e down dello strato st.strati[0]) con
    l'instradamento vero, per ogni configurazione candidata."""
    L = st.strati[0]
    with torch.no_grad():
        x = st.embedding(ids)
        ctx = st._contesto(x.shape[0], x.shape[1], x.dtype)
        st._strato(L, x, ctx)
    piano = Piano(st.moduli[L].mlp.ultimo_idx, st.esperti(L).E)
    out = {}
    for nome in ("gate", "down"):
        fam = st.esperti(L).fam[nome]
        d = st.esperti(L).scala(nome).detach()
        xs = torch.randn(piano.S, fam.C, device=st.device, dtype=torch.float16)
        gy = torch.randn(piano.S, fam.R, device=st.device, dtype=torch.float16)
        righe = []
        for (BS, BN, BK, w) in CANDIDATE_GEMM:
            blk = Blocchi(BS=BS, BN=BN, BK=BK, num_warps=w)
            try:
                for _ in range(2):
                    _gemm_triton(xs, fam, d, None, piano, False, torch.float32, blk)
                    _gemm_triton(gy, fam, d, None, piano, True, torch.float16, blk)
                _sincronizza()
                t0 = time.time()
                for _ in range(5):
                    _gemm_triton(xs, fam, d, None, piano, False, torch.float32, blk)
                _sincronizza()
                t1 = time.time()
                for _ in range(5):
                    _gemm_triton(gy, fam, d, None, piano, True, torch.float16, blk)
                _sincronizza()
                righe.append({"BS": BS, "BN": BN, "BK": BK, "warps": w, "avanti_ms": (t1 - t0) / 5 * 1e3,
                              "dx_ms": (time.time() - t1) / 5 * 1e3})
            except Exception as ex:  # una configurazione che non gira si registra, non si nasconde
                righe.append({"BS": BS, "BN": BN, "BK": BK, "warps": w, "errore": repr(ex)[:200]})
        dd_righe = []
        for (BR, BSd, w) in CANDIDATE_DD:
            blk = Blocchi(BR_dd=BR, BS_dd=BSd, num_warps=w)
            try:
                for _ in range(2):
                    _dd_triton(gy, xs, fam, piano, blk)
                _sincronizza()
                t0 = time.time()
                for _ in range(5):
                    _dd_triton(gy, xs, fam, piano, blk)
                _sincronizza()
                dd_righe.append({"BR": BR, "BS": BSd, "warps": w, "dd_ms": (time.time() - t0) / 5 * 1e3})
            except Exception as ex:
                dd_righe.append({"BR": BR, "BS": BSd, "warps": w, "errore": repr(ex)[:200]})
        # confronto: prodotto denso di torch sullo stesso lavoro (bmm per esperto non c'e': mm per esperto)
        out[nome] = {"gemm": righe, "dd": dd_righe, "assegnazioni": piano.S}
    return out


def _profila(st: "Studente", ids: torch.Tensor, n: int = 30) -> dict:
    """26/9: un passo intero (avanti senza grafo + ricalcolo + indietro sugli strati caricati) sotto
    torch.profiler, a kernel gia' compilati: i `n` nomi col piu' tempo PROPRIO sul dispositivo. Serve a scegliere
    la leva successiva sui numeri (esperti, non-esperti, attenzione, elementari). Se il profilatore non gira su
    ROCm si registra l'errore, non un numero."""
    try:
        from torch.profiler import ProfilerActivity, profile
        _sincronizza()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            st.gradiente(ids, lambda h: h.float().pow(2).mean() * 1e3)
            _sincronizza()
        for p_ in st.parametri():
            p_.grad = None
        medie = list(prof.key_averages())
        campo = "self_device_time_total"
        if sum(float(getattr(e, campo, 0.0)) for e in medie) <= 0:      # nessun tempo di dispositivo: CPU
            campo = "self_cpu_time_total"
        ev = sorted(medie, key=lambda e: -float(getattr(e, campo, 0.0)))
        tot = sum(float(getattr(e, campo, 0.0)) for e in ev)
        return {"token": int(ids.numel()), "campo": campo, "ms_totale": tot / 1e3,
                "top": [{"nome": e.key[:140], "chiamate": e.count, "ms": float(getattr(e, campo, 0.0)) / 1e3,
                         "quota": float(getattr(e, campo, 0.0)) / max(tot, 1e-9)} for e in ev[:n]]}
    except Exception as ex:  # pragma: no cover  (dipende dal profilatore della build ROCm)
        return {"errore": repr(ex)[:300]}


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double(), b.double()
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-300))


def _prova_combina(st: "Studente", ids: torch.Tensor) -> dict:
    """26/9: sul dispositivo vero e coi prodotti del motore scelto, lo strato st.strati[0] con la combinazione
    fusa contro la via torch: stesso ingresso, stesso instradamento, stesso gradiente d'uscita (x 256, come
    una scala K). Uscita, dL/dx e dL/dtheta (errore relativo al massimo). Sulla CPU (interprete, S13) erano
    identici al bit; qui si tollera l'ordine delle somme fp32 della GPU: uscita e dL/dx <= 1e-5,
    dL/dtheta <= 1e-4. Se non torna, la fusa NON si usa (COMBINA_FUSA=False) e lo si scrive nell'esito."""
    global COMBINA_FUSA
    vecchio = COMBINA_FUSA
    L = st.strati[0]
    with torch.no_grad():
        x = st.embedding(ids)
        ctx = st._contesto(x.shape[0], x.shape[1], x.dtype)
    ris, instr, G = {}, None, None
    try:
        for fusa in (False, True):
            COMBINA_FUSA = fusa
            xin = x.detach().clone().requires_grad_(True)
            with torch.enable_grad():
                y = st._strato(L, xin, ctx, instr=instr)
                if instr is None:
                    instr = st.moduli[L].mlp.ultimo_idx
                if G is None:
                    gen = torch.Generator(device=y.device).manual_seed(0)
                    G = torch.randn(y.shape, generator=gen, device=y.device, dtype=torch.float32)
                (y.float() * G).sum().mul(256.0).backward()
            ris[fusa] = (y.detach(), xin.grad, [st.esperti(L).theta[n].grad.clone() for n in FAMIGLIE])
            for n in FAMIGLIE:
                st.esperti(L).theta[n].grad = None
    finally:
        COMBINA_FUSA = vecchio
    st.azzera_cambi()
    e = {"uscita": _rel(ris[True][0], ris[False][0]), "dx": _rel(ris[True][1], ris[False][1]),
         "dtheta": max(_rel(a, b) for a, b in zip(ris[True][2], ris[False][2]))}
    ok = e["uscita"] <= 1e-5 and e["dx"] <= 1e-5 and e["dtheta"] <= 1e-4
    if not ok:
        COMBINA_FUSA = False
    return {**e, "ok": ok, "in_uso": "fusa" if ok and vecchio else "torch"}


def _prova_rotazione(device, T: int = 8192, ripetizioni: int = 5) -> dict:
    """26/9: rotazione eager contro compilata sul dispositivo vero, avanti e indietro, fp32 (la ricetta B/C
    ruota in fp32). Uguaglianza AL BIT richiesta; se manca, la compilata NON si usa (COMPILA_RUOTA=False) e
    lo si scrive nell'esito. Tempi per 1 avanti + 1 indietro di (T, 2048)."""
    global COMPILA_RUOTA
    vecchio = COMPILA_RUOTA
    gen = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(T, 2048, device=device, generator=gen)
    gy = torch.randn(T, 2048, device=device, generator=gen)
    ris = {}
    for modo, flag in (("eager", False), ("compilata", True)):
        COMPILA_RUOTA = flag
        for _ in range(2):                                            # la prima compila
            xx = x.clone().requires_grad_(True)
            ruota(xx).backward(gy)
        _sincronizza()
        t0 = time.time()
        for _ in range(ripetizioni):
            xx = x.clone().requires_grad_(True)
            y = ruota(xx)
            y.backward(gy)
        _sincronizza()
        ris[modo] = {"ms": (time.time() - t0) / ripetizioni * 1e3, "y": y.detach(), "g": xx.grad}
    uguale = torch.equal(ris["eager"]["y"], ris["compilata"]["y"]) and torch.equal(ris["eager"]["g"],
                                                                                  ris["compilata"]["g"])
    COMPILA_RUOTA = vecchio if uguale else False
    return {"T": T, "ms_eager": ris["eager"]["ms"], "ms_compilata": ris["compilata"]["ms"],
            "uguale_al_bit": bool(uguale), "in_uso": "compilata" if uguale and vecchio is not False else "eager"}


def misura(argv=None) -> dict:
    """M0 dello studente (GPU). Per ogni micro-lotto richiesto: ms/token di avanti + ricalcolo + indietro sugli
    strati VERI caricati (default 0-3: tre GDN e un'attenzione piena, catena vera), per strato e per tipo;
    proiezione a 40 strati DAI TEMPI MISURATI (30 GDN + 10 pieni; esclusi embedding, testa/KL e teacher, che
    sono di altri moduli); picco di memoria; decodifica fusa da sola; confronto dei motori; spazzata dei riquadri.
    Si rifiuta se la GPU e' occupata (strumenti/gpu_occupata.sh)."""
    import argparse
    import subprocess
    ap = argparse.ArgumentParser()
    ap.add_argument("--misura", action="store_true")
    ap.add_argument("--strati", default="0,1,2,3")
    ap.add_argument("--token", default="8192", help="micro-lotti da provare, es. 8192,16384,32768")
    ap.add_argument("--l-seq", type=int, default=4096)
    ap.add_argument("--ricetta", default="B,C", help="una o piu' ricette separate da virgola (A, B, C, B_ind32, B3, B3_ind32)")
    ap.add_argument("--motori", default="triton,triton_denso")
    ap.add_argument("--ripetizioni", type=int, default=2)
    ap.add_argument("--blocchi", default="", help="json dei campi di Blocchi")
    ap.add_argument("--spazza", action="store_true", help="spazzata delle configurazioni dei riquadri")
    ap.add_argument("--profila", action="store_true", help="torch.profiler su un passo, per ricetta e motore")
    ap.add_argument("--combina", default="fusa", help="fusa, torch o entrambe (prima/dopo della leva)")
    ap.add_argument("--uscita", default=str(Path("@@ROOT:models@@/gguf/tony-tern/e2e_studente") / "misura_m0.json"))
    ap.add_argument("--dispositivo", default="cuda", help="cpu = prova a secco del codice (motore torch)")
    a = ap.parse_args(argv)
    gpu = a.dispositivo == "cuda"
    if gpu:
        rc = subprocess.run([str(QUI / "strumenti" / "gpu_occupata.sh")], capture_output=True, text=True)
        if rc.returncode != 0:
            raise SystemExit(f"GPU occupata, non misuro:\n{rc.stdout}")
    blk = Blocchi(**json.loads(a.blocchi)) if a.blocchi else Blocchi()
    strati = [int(s) for s in a.strati.split(",")]
    esito = {"data": time.strftime("%Y-%m-%d %H:%M:%S"), "gguf": str(GGUF_THOR_T), "strati": strati,
             "ricette": a.ricetta, "blocchi": blk.__dict__, "corse": []}
    if gpu:
        libera, totale = torch.cuda.mem_get_info()
        esito["vram_all_avvio_GiB"] = {"libera": libera / 2 ** 30, "totale": totale / 2 ** 30}
        esito["rotazione"] = _prova_rotazione(torch.device(a.dispositivo), T=min(8192, int(a.token.split(",")[0])))
        print(json.dumps({"rotazione": esito["rotazione"]}), flush=True)
    global COMBINA_FUSA
    for ricn, nome_mot, comb in [(r, m, c) for r in a.ricetta.split(",") for m in a.motori.split(",")
                                 for c in a.combina.split(",")]:
        COMBINA_FUSA = comb == "fusa"
        if comb == "fusa" and esito.get("combina_fusa_scartata"):
            continue                                        # la prova sul dispositivo l'ha bocciata: non si misura
        alloc0 = torch.cuda.memory_allocated() if gpu else 0
        st = Studente(strati=strati, ricetta=ricn, motore_nome=nome_mot, device=a.dispositivo, blk=blk)
        residente_strato = (torch.cuda.memory_allocated() - alloc0) / len(strati) if gpu else 0
        theta_strato = sum(st.esperti(L).theta[n].numel() for (L, n) in st.chiavi()) * 4 / len(strati)
        if gpu:
            st.carica_embedding()
        fam = st.esperti(strati[0]).fam["gate"]
        d = st.esperti(strati[0]).scala("gate").detach()
        if gpu and "decodifica_fusa_gate" not in esito:
            for _ in range(2):
                decodifica_pesi(fam, d, torch.float16, blk)
            _sincronizza()
            t0 = time.time()
            for _ in range(5):
                decodifica_pesi(fam, d, torch.float16, blk)
            _sincronizza()
            t_dec = (time.time() - t0) / 5
            esito.setdefault("decodifica_fusa_gate", {"ms": t_dec * 1e3,
                                                      "GB_s_scritti": fam.E * fam.R * fam.C * 2 / t_dec / 1e9})
        if comb == "fusa" and isinstance(st.mot, MotoreTriton):
            n0 = int(a.token.split(",")[0])
            prova = _prova_combina(st, _ids_reali(max(1, n0 // a.l_seq), min(n0, a.l_seq)))
            esito.setdefault("prova_combina", {})[f"{ricn}_{nome_mot}"] = prova
            print(json.dumps({"prova_combina": {f"{ricn}_{nome_mot}": prova}}), flush=True)
            if not prova["ok"]:                             # nessuna corsa «fusa» che misuri la via torch
                esito["combina_fusa_scartata"] = True
                COMBINA_FUSA = False
                del st
                if gpu:
                    torch.cuda.empty_cache()
                continue
        for n_tok in [int(v) for v in a.token.split(",")]:
            B = max(1, n_tok // a.l_seq)
            T = n_tok // B
            ids = _ids_reali(B, T)
            if gpu:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated() if gpu else 0
            try:
                m = _misura_passo(st, ids, a.ripetizioni)
            except torch.OutOfMemoryError as ex:
                esito["corse"].append({"ricetta": ricn, "motore": nome_mot, "token": B * T,
                                       "errore": "OOM " + str(ex)[:200]})
                continue
            tipi = {L: st.cfg.layer_types[L] for L in strati}
            per_tipo = {}
            for L, t in m["per_strato_s"].items():
                per_tipo.setdefault(tipi[L], []).append(t["avanti"] + t["indietro"])
            media = {k: sum(v) / len(v) for k, v in per_tipo.items()}
            proiezione = None
            if {"linear_attention", "full_attention"} <= set(media):
                n_tipo = {k: sum(1 for L in range(40) if st.cfg.layer_types[L] == k) for k in media}
                proiezione = sum(media[k] * n_tipo[k] for k in media) / (B * T) * 1e3
            corsa = {"ricetta": ricn, "motore": nome_mot, "combina": comb, "token": B * T, "B": B, "T": T,
                     "ms_token_per_strato": {str(L): (t["avanti"] + t["indietro"]) / (B * T) * 1e3
                                             for L, t in m["per_strato_s"].items()},
                     "per_strato_s": {str(L): t for L, t in m["per_strato_s"].items()},
                     "ms_token_passo_strati_caricati": m["passo_s"] / (B * T) * 1e3,
                     "proiezione_40_strati_ms_token": proiezione,
                     "picco_GiB": (torch.cuda.max_memory_allocated() - base) / 2 ** 30 if gpu else None,
                     "cambi_instradamento": st.cambi()}
            if gpu:
                # 26/9: proiezione della memoria dello STUDENTE a 40 strati, per componenti (esclusi testa, KL e
                # teacher, che sono di altri moduli). Transitorio di uno strato = picco del passo meno i
                # checkpoint e i gradienti di theta degli strati caricati (stima per differenza, dichiarata).
                ck = B * T * st.cfg.hidden_size * torch.tensor([], dtype=st.ric.dtype_residuo).element_size()
                picco = torch.cuda.max_memory_allocated() - base
                trans = picco - len(strati) * (ck + theta_strato)
                comp = {"residente_per_strato_GiB": residente_strato / 2 ** 30,
                        "adam_grad_m_v_per_strato_GiB": 3 * theta_strato / 2 ** 30,
                        "checkpoint_per_strato_GiB": ck / 2 ** 30,
                        "transitorio_uno_strato_GiB": trans / 2 ** 30,
                        "embedding_fp32_GiB": st._emb.numel() * 4 / 2 ** 30 if st._emb is not None else 0.0}
                comp["totale_40_strati_GiB"] = (40 * (residente_strato + 3 * theta_strato + ck) + trans) / 2 ** 30 \
                    + comp["embedding_fp32_GiB"]
                corsa["proiezione_memoria"] = comp
            esito["corse"].append(corsa)
            print(json.dumps(corsa), flush=True)
        if a.spazza and nome_mot == "triton" and "spazzata" not in esito:
            ids = _ids_reali(max(1, int(a.token.split(",")[0]) // a.l_seq), a.l_seq)
            esito["spazzata"] = _spazza_blocchi(st, ids)
        if a.profila:
            n0 = int(a.token.split(",")[0])
            esito.setdefault("profilo", {})[f"{ricn}_{nome_mot}_{comb}"] = _profila(
                st, _ids_reali(max(1, n0 // a.l_seq), min(n0, a.l_seq)))
        del st
        if gpu:
            torch.cuda.empty_cache()
    Path(a.uscita).parent.mkdir(parents=True, exist_ok=True)
    Path(a.uscita).write_text(json.dumps(esito, indent=1))
    print(json.dumps(esito, indent=1))
    return esito


if __name__ == "__main__":
    if "--misura" in sys.argv:
        misura(sys.argv[1:])
    else:
        print(__doc__)
