#!/usr/bin/env python3
"""e2e_allena — il ciclo della fase E2E sulle sole scale fp16 degli esperti di Thor-T (progetto
`e2e_progetto_scelto.md` §2.3, §2.5, §3, §4, §9). 26/9/2026. File NUOVO: nessun file esistente e' modificato.
Si importano in SOLA LETTURA `e2e_studente` (lo studente dai byte del GGUF, funzione autograd per famiglia),
`e2e_teacher` (cache del teacher, `logit_maestro`, testa bf16 originale).

## Cosa fa (e da dove viene ogni scelta)
- **Unici parametri: theta** (fp32, una per scala), d_eff = fp16_RNE(d0 * exp(theta)) (§2.2; e' dello studente).
- **Perdita = sum_t KL(p_T || p_S)** in avanti, vocabolario intero, sui logit ricalcolati dal nascosto del teacher
  (bf16, dopo output_norm) con la testa bf16 ORIGINALE e la stessa funzione `e2e_teacher.logit_maestro` (§2.3).
  `KLFusa` calcola valore E gradiente dL/dh nello STESSO passaggio (la perdita e' l'ultimo nodo, l'indietro
  segue subito): 3 prodotti V x H a token invece dei 5 del prototipo (che ricalcolava p_S e p_T
  nell'indietro), pezzi da 512 token (A7: ~1,5 GiB vivi, non 4,7), logit fp32, somme in fp64.
- **G1 come SCALA DELLA PERDITA K** (A2, R3): perdita = K * (somma KL / token del passo), K = 2^round(log2(1e-8 /
  eps_G1)), eps_G1 = min(1e-8, 1e-3 * q01|g|) sulle scale vive con g != 0 al passo 0, eps di Adam = eps_G1 * K.
  Il cancello misura anche, nel dtype VERO dell'indietro (fp16 nella ricetta B), la quota di dL/dY che diventa
  zero o subnormale entrando nei prodotti (dL/dW = dY^T X si accumula in fp32 dentro il kernel: l'underflow
  entra solo da dY), e confronta dL/dtheta dell'indietro fp16 con quello di un indietro fp32 (ricetta
  «B_ind32», stesso avanti) sul primo micro-lotto: quota di scale con gradiente nullo o di SEGNO SBAGLIATO.
  Dopo il passo 1: >= 99% delle scale vive con g != 0 mosse di >= 0,99 lr nel verso -g (prova T7).
- **AdamW, weight_decay = 0.0 scritto e VERIFICATO** (assert sui param_groups alla creazione, dopo ogni
  ripresa e prima di ogni passo: `load_state_dict` ripristina gli iperparametri salvati), beta (0,9, 0,95),
  lr COSTANTE (§1.2), `fused=True` su GPU.
- **Held-out per dominio** ogni `heldout_ogni` passi e alla fine: KL media per token per dominio; la metrica
  che sceglie il migliore e' la media dei domini (macro) esclusi quelli in `escludi_domini` (default la wiki
  del verdetto, «wiki»: §2.4, rischio 6). Si tiene il **migliore** (istantanea su DISCO, mai in VRAM: A7:
  scale fp16 ESATTE dell'avanti + theta) e ci si **ferma** se la metrica sale in 2 misure di fila mentre la KL
  di addestramento scende (§2.5, P1).
- **Blocchi rotanti del teacher** (§3.1): epoca 0 nell'ordine del piano del teacher (ogni prefisso ha la
  composizione U), si aspetta il teacher se una riga non e' ancora sigillata, e con `--rotazione` un blocco
  si consuma SOLO dopo che un punto di ripresa che ne ha usato tutte le righe e' su disco.
- **Punti di ripresa collaudati al bit** (lezione del 17/9, T9): theta, stato di Adam, stato del ciclo
  (passo, storia, migliore, contatori, G1), scritti in `.parziale`, fsync, rename atomico, manifesto con
  sha256; due punti tenuti. `e2e_test_allena.py` A5: 3 passi + salva/carica + 3 == 6 passi filati.
- **27/9 velocita'**: rumore del gradiente a due lotti (McCandlish 2018, app. A) a ogni passo, micro-lotti contro
  passo intero, anche precondizionato con sqrt(v) di Adam (`MisuraRumore`, voce "rumore" della storia, riga di
  registro; non cambia nessun numero); held-out rapido su N righe con misura piena ogni M passi
  (`--heldout-righe-rapide`, `--heldout-pieno-ogni`); media esponenziale di theta (`--ema-theta`, migliore_ema_*,
  export con `e2e_esporta.py --istantanea migliore_ema_<passo>`); `--beta2` (regola in `ConfigAllena`).
- **bf16x3** (`e2e_bf16x3.py`, Henry et al. 2019): ricetta `B3_ind32` (proiezioni non-esperte in bf16x3,
  indietro degli esperti fp32), `--testa-kl bf16x3` (logit dello studente e dh), `--logit-maestro bf16` (UNA
  GEMM bf16 con accumulo fp32: W_T e h_T sono gia' bf16). Default fp32: B identica al bit. Cancello GPU:
  `strumenti/cancello_b3.sh` (`e2e_cancello_b3.py`); prove CPU `e2e_test_bf16x3.py`.
- **Niente sincronizzazioni nel ciclo** salvo UNA per passo (valore della perdita, sentinella delle scale,
  cambi d'instradamento: `Studente.controlla`); il caricamento del teacher e' anticipato in un filo a parte
  (memoria bloccata, copia asincrona).
- **27/9 `--allena-f32 fam1,fam2 [--lr-f32 1e-5]`** (default vuoto = identico al bit, A23 di e2e_test_f32): i tensori F32 del GGUF
  di quelle famiglie (norme, router, GDN; costo zero di dimensione, vincolo ternario intatto) in un SECONDO gruppo
  AdamW col proprio lr (default 1e-5, Norm Tweaking). Nella ripresa i master e l'ordine dei nomi; famiglie
  aggiunte alla ripresa = decisione (Adam nuovo per i nuovi), tolte = errore (28/9: salvo con `--f32-ripristina
  fam,...`, che le riporta al valore del GGUF in tutti gli strati, anche congelati; prova A28); `f32.npz` nell'istantanea, che
  `e2e_esporta` scrive al posto dei vecchi valori. Prove: `e2e_test_f32.py` (A23-A26, A27 = integrazione con --beta2/--ema-theta/E2E_GDN;
  erano A21-A24 nel ramo allena-f32, rinominate al merge: A21/A22 sono il rumore e la EMA in e2e_test_allena).
- **27/9 `--lora-nonesperti R [--lora-alpha A] [--lora-famiglie attn,gdn,shexp|tutte] [--lr-lora 1e-4] [--lora-tipi Q8_0]`** (default 0 =
  identico al bit, L1 di e2e_test_lora): adattatori B A (e2e_lora.py) sui k-quant NON esperti, y = x W^T + s (x A^T) B^T,
  B = 0 al passo 0; gruppo AdamW proprio (in coda: theta, F32, LoRA). Ripresa: A, B e ordine; accenderla alla ripresa
  = decisione, spegnerla o cambiarla = errore. `lora.npz` nell'istantanea; `e2e_esporta --lora-modo griglia|ggml`
  riquantizza W + s B A nello stesso tipo (taglia invariata). Prove: e2e_test_lora.py (L1-L5).
- **27/9 `--peso-token {uniforme,kl,topk}` [--peso-alfa 1] [--peso-max 5] [--peso-frazione 0.6]** (default uniforme =
  identico al bit, W1 di e2e_test_peso_token.py; prove W1-W5): la perdita del micro-lotto (kl o cakld) diventa
  sum_t w_t l_t con w FISSI (senza gradiente) a media 1 nel micro-lotto (`pesi_token`): kl = (l_t / media l)^alfa col
  tetto, topk = Rho-1. dL/dh_t = w_t dL_t/dh_t (gradiente esatto a w fisso). Held-out (`valuta`), sonda del passo V
  (`kl_sonda`) e curvatura restano UNIFORMI; "kl_train" del registro resta la somma uniforme, la pesata e i pesi nella
  voce "pesi". TECNICHE §35.
- **27/9 `--ottimizzatore-theta {adamw,sophia} [--sophia-gamma 0.01] [--sophia-beta2 0.99] [--sophia-eps 1e-15]`**
  (default adamw = identico al bit, N1 di e2e_test_sophia): Newton diagonale stile Sophia sulle scale (e2e_sophia.py,
  arXiv 2305.14342) con la curvatura Gauss-Newton per gruppo misurata nei micro-lotti di curvatura del passo V (serve
  --pv-codici); F32 e LoRA restano AdamW. Passaggio alla ripresa in entrambi i versi = decisione. Diagnosi e numeri:
  TECNICHE_E2E_PV.md §36. Prove: e2e_test_sophia.py (N1-N6).
- **27/9 `--lora-init ga [--lora-ga-gamma 16] [--lora-ga-righe 0]`** (default zero = identico al bit): LoRA-GA (arXiv
  2407.05000): prima del primo passo con la LoRA, un avanti+indietro sulle righe di quel passo da' il gradiente pieno
  dW di ogni tensore LoRA; A = V[:, :r]^T c, B = U[:, r:2r] c (c = out^(1/4)/sqrt(gamma)), A0/B0 fermi e l'avanti
  y = x W^T + s [(x (A-A0)^T) B^T + (x A0^T) (B-B0)^T]: modello identico al bit al passo 0. Export: s (B A - B0 A0).
  Prove: e2e_test_lora_ga.py (G1-G6); TECNICHE_E2E_PV.md §37 (era §33 nel ramo).
- **27/9 ramo integrata (TECNICHE §39)**: `--lora-reinit-ga` (alla ripresa, LoRA a init zero scartata e re-inizializzata
  GA; resto della ripresa invariato; idempotente) e `--attribuzione [--attribuzione-righe 2] [--attribuzione-solo-ogni 1]`
  (e2e_attribuzione.py: sonda fissa, lascia-uno-fuori/solo-uno per theta/F32/LoRA/codici al bit, passo di AdamW contro
  Sophia, diagnostica e tempi; riga JSON per passo in <corsa>/registro_componenti.jsonl). Prove: e2e_test_integrata.py.
- **28/9 `--lora-ripristina {zero,ferma}`** (default vuoto = identico al bit; prova_strumenti_f32 sul passo 20 di EB3:
  LoRA spenta = KL strumenti 1,044 -> 0,750): alla ripresa Delta W di OGNI adattatore torna a 0 esatto (anche negli strati
  congelati) e il suo |Delta W|_F va in stato["lora_ripristini"] e nel registro, per adattatore. zero: B = 0 con A tenuto
  (LoRA-GA: A = A0, B = B0), Adam della LoRA da zero, la LoRA continua (tipicamente con --lr-lora piu' basso); avviene
  quando il modo compare (la stessa riga rilanciata non rifa' nulla). ferma: LoRA fuori dall'ottimizzatore e senza
  gradiente finche' la riga lo dice, niente A/B in riprese e istantanee (export = GGUF senza LoRA). Lo stato di Adam
  della ripresa perde il gruppo LoRA (`_senza_gruppo_lora`): theta per indice, F32 per nome. e2e_esporta: un tensore
  con Delta W = 0 esatto tiene i byte del sorgente in entrambi i modi. Prove: e2e_test_lora_ripristina.py (R1-R5).
- **27/9 `--lota-rango R [--lota-omega W] [--lota-top-p P] [--lota-lr LR] [--lota-init paper|gradiente]
  [--lota-sonda-tolleranza T | --lota-senza-sonda]`** (default 0 = identico al bit, T6 di e2e_test_lota): LoTA-QAF
  (arXiv 2505.18724, e2e_lota.py) sui CODICI degli esperti: adattatori ternari B A per esperto,
  codici = clamp(T0 + sign(B A) 1[|B A| > omega]) dentro {-1,0,1}, t-SignSGD sul top-p dei gradienti (gancio nel dd:
  nessun indietro in piu'), sonda della regione di fiducia (annulla e p/2 se la perdita sale oltre la tolleranza;
  ⚠️ sui giocattoli la sonda stretta blocca la traiettoria di LoTA, che sale prima di scendere: LEGGIMI_lota-qaf.md). ESCLUSIVA col passo V
  (--pv-codici); i registri PV di una fase precedente si rigiocano congelati sotto la base. Ripresa: A, B a 2 bit
  (`lota_NNNNNN.pt`, sha nello stato), byte ricostruiti e controllati; migliore: `registri_pv.npz` = PV congelati +
  registro netto LoTA -> export dei codici con `e2e_pv_codici.py esporta` invariato. Prove: e2e_test_lota.py (T1-T9).
  voce "pesi". TECNICHE §31.
- **27/9 righe a scelta multipla (ramo trainer-mc)**, tutto spento di default (identico al bit, prova M1 di
  e2e_test_mc.py; prove M1-M7): `--teacher-mc DIR` (cache del maestro sulle righe mc, STESSO formato di --teacher,
  con `risposte.pt` accanto al file del corpus che il suo piano cita; lato «heldout» o `--teacher-mc-heldout DIR` = held-out
  mc) e `--righe-mc M` (M righe mc su `righe_per_passo`, intercalate; le altre dal teacher base, nell'ORDINE di sempre:
  `PianoRighe`, cursori nello stato). La ripresa accetta l'aggiunta (o il cambio) del teacher mc senza toccare theta,
  codici, Adam, V-Bop, EMA: la firma del teacher base resta quella di prima, quella mc sta in stato["mc"].
  `--peso-token risposta [--beta-risposta 0.1]`: KL piena su tutti i token (peso 1) e peso 1 + lambda alle posizioni-
  risposta, lambda tale che le posizioni-risposta portino la frazione B della massa dei PESI della perdita del passo
  (`lambda_risposta`). `--gamma-lettere G`: in piu' G * w_a * KL(p4_T || p4_S) alle posizioni-risposta, p4 = softmax
  rinormalizzata sui 4 id delle lettere (`kl_lettere`). Held-out mc a ogni misura (rapida e piena): accuratezza sulle 4
  lettere, p(C | giusta non C), KL4, margine (`valuta_mc`); solo registro, mai nella scelta del migliore.
- **27/9 ramo dominio-strati** (analisi_ripidita.md §1.2-1.3, §3 leve 3 e 4; LEGGIMI_dominio-strati-b.md), entrambe spente di
  default (identico al bit, prova D1 di e2e_test_dominio_strati.py):
  `--peso-dominio "it_chat=0.5,strumenti=0.75,edu=1.5,..."`: la perdita della riga (kl o cakld, anche pesata per token:
  moltiplicativo) pesata per il DOMINIO della riga (righe mc del ramo trainer-mc, indice >= 2^30: dominio `mc`), pesi > 0
  (vincolo: nessun dominio abbandonato), normalizzati sulle righe del PASSO (media 1: la scala della perdita, la K di G1 e
  l'eps di Adam restano quelli di prima); non elencati = 1. dL/dh_t = w_riga dl_t/dh_t (gancio su hS, esatto). Curvatura
  del passo V (e di Sophia) coerente: vettore campionato per sqrt(w_riga). Held-out, sonda del passo V, attribuzione e
  "kl_train" restano UNIFORMI; nel registro la voce "peso_dominio" (costante di normalizzazione, pesi effettivi, righe).
  `--strati-allenati 20-39`: scale, F32, LoRA e codici V allenati SOLO in quegli strati (Studente.congela_strati:
  requires_grad falso altrove, grad None, AdamW/Sophia li saltano: stato al bit; V senza statistiche ne' salti ne'
  aggiornamenti V-Bop; EMA ferma; curvatura di Sophia ferma); l'indietro si ferma al primo strato allenato. Accendere,
  spegnere o cambiare alla ripresa = decisione (una ripresa che li allenava tutti e' accettata).
- **27/9 ramo v-interazione-b `--pv-sonda-dosi [--pv-ricentra [--pv-ricentra-giri 3]]`** (default spenti = sonda
  tutto-o-niente di prima, identica al bit; TECNICHE §32.5 azioni 1-2, LEGGIMI_v-interazione-b.md): i salti scelti
  si provano a dosi annidate 1/256 1/64 1/16 1/4 1 in ordine di guadagno sulla sonda A, la dose migliore si conferma
  sulla sonda B (secondo micro-lotto del passo dopo); ricentro = un indietro della sonda A nello stato raggiunto,
  lascia-uno-fuori f(S) - f(S\\i) ~ dw g1 - H dw^2 / 2 per salto, annullamento dei dannosi a meta' per volta finche' B
  conferma. La dose parziale tenuta non dimezza rho. Curva dose -> sonda in voce["pv"]["dosi"] (storia, log e
  registro_componenti.jsonl). Prove: e2e_test_v_interazione.py.

## Uso (GPU solo quando e' libera: `strumenti/gpu_occupata.sh` a 0)
    python3 e2e_allena.py allena --corsa NOME --teacher @@ROOT:lavoro@@/e2e/teacher/q8-tutte-riga [--lr 1e-3]
        [--righe-passo 32] [--righe-micro 2] [--heldout-ogni 16] [--ripresa-ogni 4] [--rotazione] [--riprendi]
        [--beta2 0.95] [--ema-theta 0] [--heldout-righe-rapide 0] [--heldout-pieno-ogni 0]
        [--allena-f32 attn_norm,ffn_gate_inp,...] [--lr-f32 1e-5]
        [--lora-nonesperti 16 --lora-famiglie tutte --lr-lora 1e-4 [--lora-alpha 16] [--lora-tipi Q8_0]
         [--lora-init ga [--lora-ga-gamma 16] [--lora-ga-righe 0]] [--lora-ripristina zero|ferma]]
        [--peso-token kl [--peso-alfa 1] [--peso-max 5] | --peso-token topk [--peso-frazione 0.6]]
        [--lota-rango 16 [--lota-omega 12] [--lota-top-p 0.05] [--lota-lr 1.0] [--lota-senza-sonda]]
        [--teacher-mc DIR [--teacher-mc-heldout DIR] [--righe-mc 8] [--peso-token risposta [--beta-risposta 0.1]]
         [--gamma-lettere 1]]
        [--peso-dominio "it_chat=0.5,strumenti=0.75,edu=1.5,..."] [--strati-allenati 20-39]
    python3 e2e_allena.py g1 --corsa NOME --teacher ...      # solo il cancello G1 e il primo passo (M0)
    python3 e2e_allena.py misura --teacher ... --strati 0 --strati 0,1 --strati 0,3   # ms/token e picco (M0)
    python3 e2e_allena.py lancia --corsa NOME -- <argomenti di allena>     # unita' systemd --user e2e-allena-NOME
    python3 e2e_allena.py stato --corsa NOME
Uscite: @@ROOT:lavoro@@/e2e/allena/<corsa>/ (config.json, registro.jsonl, ripresa_*.pt, migliore_*/, esito.json).
Prove (CPU, byte veri di Thor-T): `e2e_test_allena.py`.
"""
from __future__ import annotations

import argparse
import concurrent.futures as _cf
import dataclasses
import datetime as _dt
import fcntl
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

import e2e_sentinella as SE                                # 28/9 blocco di EB3: sentinella di progresso (stdlib)

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
sys.path.insert(0, "@@ROOT:models@@/build/llamacpp-tq1-k1024/gguf-py")

VERSIONE = "e2e_allena 1 (26/9/2026)"
RADICE = Path(os.getenv("E2E_ALLENA_DIR", "@@ROOT:lavoro@@/e2e/allena"))
GUARDIA = QUI / "strumenti" / "gpu_occupata.sh"
PYTHON = "@@ROOT:venv@@/bin/python3"
# 26/9: registro di coesistenza col teacher live (e2e_teacher.py) — vedi `_coesistenza_gpu` e `_guardia_gpu`.
REGISTRO_COESISTENZA = QUI / "strumenti" / ".gpu_coesistenza"


def adesso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


# ═══════════════════════════════════════ configurazione ═══════════════════════════════════════
PESI_TOKEN = ("uniforme", "kl", "topk", "risposta", "generati")   # 12/10 "generati" = OPD (e2e_onpolicy.py)        # 27/9 --peso-token (vedi ConfigAllena e `pesi_token`)
_PESO_DEFAULT = {"peso_alfa": 1.0, "peso_max": 5.0, "peso_frazione": 0.6}
_BETA_RISPOSTA_DEFAULT = 0.1                               # 27/9 trainer-mc: --beta-risposta (vedi `lambda_risposta`)
LETTERA_C = 2                                              # indice di « C» in id_lettere (A, B, C, D)
LORA_RIPRISTINI = ("", "zero", "ferma")                    # 28/9 --lora-ripristina (vedi ConfigAllena e `ripristino_lora`)


@dataclass(frozen=True)
class ConfigAllena:
    """Default = E-B del progetto (§2.5): 32 x 4096 = 131.072 token a passo, held-out ogni 16 passi, 1 epoca.
    lr: centro della griglia {3e-4, 1e-3, 3e-3} di E-A; E-A non si fa (decisione del 26/9): una sola corsa.
    27/9 regola per passi PIU' PICCOLI di 32 righe (r = `righe_per_passo`, CLI `--righe-passo`, `--beta2`, `--lr`):
    beta2 = 0,95^(r/32) (stessa memoria di v in RIGHE viste: r = 16 -> 0,9747, r = 8 -> 0,9873, r = 4 -> 0,9936);
    lr invariato (Adam normalizza il gradiente: il passo per aggiornamento non dipende dal lotto) oppure
    lr * sqrt(r/32) (regola della radice: stesso rumore per riga vista). Si sceglie col rumore del gradiente del
    registro (B_simple, McCandlish 2018): con r * 4096 token << B_simple il passo e' dominato dal rumore -> sqrt."""
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.95
    righe_per_passo: int = 32
    righe_per_micro: int = 2
    righe_per_micro_valuta: int = 4
    epoche: int = 1
    seconda_epoca_se: float = 1.3          # §2.5: la 2a epoca solo se held-out scende e KL held/train <= 1,3
    heldout_ogni: int = 16
    ripresa_ogni: int = 4
    pezzo_kl: int = 512
    seme: int = 20260926
    K0: float = 2.0 ** 20                  # scala PROVVISORIA per misurare g al passo 0 (poi K dalla regola)
    soglia_passo_pieno: float = 0.99
    quota_min_passo_pieno: float = 0.99
    max_quota_dy_azzerati: float = 1e-4    # G1: dL/dY != 0 in fp32 che diventano 0 in fp16 (quota)
    max_quota_dtheta_rotti: float = 1e-3   # G1: dL/dtheta nullo o di segno sbagliato contro l'indietro fp32
    margine_trabocco: float = 16.0         # G1: K * max|dY| resta sotto 65504 / margine (altre righe, altri passi)
    max_giri_g1: int = 8
    # 27/9 fase 2 PV (e2e_pv_codici, progetto e2e_progetto_pv_codici.md): passo V sui CODICI ternari dopo il passo P
    pv_codici: bool = False                # spento = fase 1 identica (EB2)
    pv_rho: float = 0.01                   # trust ratio ||Delta W_e|| / ||W_e|| per passo (PV §4.2); dimezzato se la sonda peggiora
    pv_micro_curvatura: int = 2            # micro-lotti del passo rifatti per la curvatura (Fisher campionato)
    pv_n_proposta: int = 2                 # micro-lotti di proposta (gli altri della perdita verificano)
    pv_q_fdr: float = 0.1                  # BH sulla direzione
    pv_q_guadagno: float = 0.1             # BH sul guadagno di secondo ordine
    pv_passi_accumulo: int = 1             # finestra: statistiche V sommate su N passi, salti decisi ogni N (1 = come prima)
    # 1/10 TT5-ottimizzato: con pv_modo "vbop" N > 1 vuol dire «passo V ogni N passi»: proposta, fusione dei candidati,
    # verifica, curvatura (ogni pv_curvatura_ogni) e EMA di V-Bop (aggiorna_bop) a OGNI passo, come con N = 1; selezione,
    # salti, sonda a dosi, punto medio e ricentro SOLO ai passi t con (t + 1) % N == 0 (e all'ultimo della corsa). La
    # memoria fra i passi e' la EMA di V-Bop (lo StatoV resta a finestra 1: nessuna somma di S/Q fra passi). N = 1 = di
    # sempre, al bit.
    # 27/9 V-Bop a gruppo (TECNICHE_E2E_PV.md §21): "pv" = PV di prima (default, identico al bit); "vbop" = stato
    # persistente per gruppo fra i passi (candidato con isteresi, m e v del gradiente proiettato in EMA con pv_gamma,
    # H in EMA, eta'), curvatura ogni pv_curvatura_ogni passi V (1 = ogni passo; >1 solo con vbop)
    pv_modo: str = "pv"
    pv_gamma: float = 0.5
    pv_curvatura_ogni: int = 1
    # 27/9 regione di fiducia vera per rho (§19): r = calo effettivo della sonda / calo previsto; r > 0,75 -> rho x 2
    # (tetto pv_rho_max), r < 0,25 -> rho / 2; spenta = rho solo dimezzato quando la sonda sale (come prima)
    pv_rho_adattivo: bool = False
    pv_rho_max: float = 0.1
    # 27/9 --pv-accetta (ramo v-senza-bh, TECNICHE_E2E_PV.md §38): "bh" = i due test Benjamini-Hochberg (q_fdr sulla
    # direzione, q_guadagno sul guadagno), default, identico al bit; "guadagno" = candidato accettato se il guadagno di
    # secondo ordine STIMATO e' > 0, senza BH (budget rho, ordinamento, sonda e rho adattivo invariati). pv_z_min (solo
    # con "guadagno"): in piu' z > pv_z_min, z = t del gradiente nella direzione della proposta (0 = nessuna soglia);
    # pv_z_guadagno_min (solo con "guadagno"): in piu' t del GUADAGNO > soglia = limite inferiore di confidenza del
    # guadagno > 0 a quegli errori standard, senza correzione per molteplicita' (0 = nessuna soglia)
    pv_accetta: str = "bh"
    pv_z_min: float = 0.0
    pv_z_guadagno_min: float = 0.0
    # 27/9 ramo v-interazione-b (TECNICHE_E2E_PV.md §32.5 azioni 1-2; e2e_pv_codici.seleziona_e_prova_dosi). Spenti = sonda
    # tutto-o-niente di prima, identica al bit. pv_sonda_dosi: i salti scelti si provano a dosi crescenti (prefissi
    # annidati in ordine di guadagno previsto, frazioni 1/256 1/64 1/16 1/4 1) sul primo micro-lotto del passo dopo
    # (sonda A), si tiene la dose migliore se la conferma il SECONDO micro-lotto del passo dopo (sonda B, disgiunto);
    # nessuna dose che migliora (o B che non conferma) = annullamento e rho / 2 come oggi; una dose PARZIALE tenuta
    # non dimezza rho. pv_ricentra (solo con pv_sonda_dosi): dopo la dose UN indietro della sonda A nello stato raggiunto,
    # lascia-uno-fuori per ogni salto tenuto, annullamento dei dannosi a meta' per volta (al massimo pv_ricentra_giri
    # giri) finche' la sonda B conferma.
    pv_sonda_dosi: bool = False
    pv_ricentra: bool = False
    pv_ricentra_giri: int = 3
    # frazioni delle dosi (crescenti, in (0, 1]); vuota = la griglia di sempre 1/256, 1/64, 1/16, 1/4, 1
    # (e2e_pv_codici.FRAZIONI_DOSI). Con molti salti scelti (rho alto, --pv-accetta guadagno) servono dosi piu' piccole.
    pv_dosi_frazioni: tuple = ()
    # 29/9 ramo punto-medio (matematica/punto_medio_pv.md; e2e_pv_codici.PassoV._prezzi_punto_medio). Spento = identico al
    # bit. pv_punto_medio (solo con pv_sonda_dosi): le pv_pm_max primitive (pv_pm_primitiva: famiglia | tipo | strato |
    # esperto) col guadagno previsto piu' alto si prezzano al PUNTO MEDIO PROPRIO sulla sonda A (regola del trapezio sugli
    # stati interi, un indietro per primitiva + uno in q0); le dosi sono prefissi delle primitive col prezzo < 0, poi
    # conferma su B e ricentro come sempre. pv_pm_filtra: dentro la primitiva solo i salti col prezzo al centro < 0.
    # pv_pm_lettere (serve righe_mc > 0): obiettivo = KL delle 4 lettere su un micro-lotto di righe mc del passo dopo
    # (M1), vincolo = sonda generica (sum s_B <= 0), conferma dell'obiettivo su un secondo micro-lotto mc DISGIUNTO (M2).
    pv_punto_medio: bool = False
    pv_pm_primitiva: str = "famiglia"
    pv_pm_max: int = 12
    pv_pm_filtra: bool = True
    pv_pm_lettere: bool = False
    # 29/9 ramo pv-lettere-dosi (e2e_pv_codici.ConfigDosi.pm_golosa; solo con pv_punto_medio). Spento = identico al bit.
    # Acceso: le dosi sono una prova GOLOSA primitiva per primitiva, decisa dal funzionale obiettivo (con pv_pm_lettere:
    # KL delle 4 lettere su M1) col vincolo della sonda generica A MISURATO (<= partenza) invece dell'argmin della sonda
    # generica su prefissi della scala di frazioni (regola della fucina E73: il funzionale che sceglie porta il termine
    # della decisione). Conferma su B e su M2 come sempre.
    pv_pm_golosa: bool = False
    # 29/9 ramo tt3-cure. pv_riuso (e2e_riuso_prove.RiusoProve): nella fase V a dosi, prefisso per strato riusato fra le
    # prove sugli stessi micro-lotti (si ricalcola solo dallo strato cambiato in giu'), indietro troncato al salto piu'
    # basso, piani e scale riusati. NON e' una decisione: sonde, lettere e gradienti letti sono gli stessi al bit (prove
    # e2e_test_riuso_prove.py); `pv_riuso_lotti` micro-lotti tenuti (~2,7 GB di VRAM l'uno a 8192 token).
    # pv_ricentra_obiettivo (ConfigDosi.ricentra_obiettivo, solo con pv_ricentra, pv_pm_golosa e pv_pm_lettere): i giri
    # del ricentro si tengono solo se anche le lettere su M1 non salgono. Decisione. Spenti = identico al bit.
    pv_riuso: bool = False
    pv_riuso_lotti: int = 2
    pv_ricentra_obiettivo: bool = False
    # 2/10 ramo tt5-fase-v (~/tony-forgia/thor_v2/ottimizzazioni_tt5/FASE_V.md). ESECUZIONE, non decisione: spenti =
    # di sempre al bit; accesi = stesse decisioni (prove e2e_test_fase_v.py), meno lavoro nella fase V.
    # pv_dd_solo_salti (V1): negli indietro dei gradienti della fase V (prezzi al punto medio, ricentro) niente dL/dd per
    # le famiglie senza salti preparati (va solo nel gradiente di theta, che si butta).
    # pv_pm_ordine_strati (V2): i g1 dei prezzi al punto medio in ordine di strato decrescente (prefisso del riuso).
    # pv_golosa_pigra (V3): nella prova golosa, sonda A gia' sopra la partenza -> primitiva rifiutata senza misurare
    # l'obiettivo (le lettere su M1).
    # pv_tempi_v: secondi per sotto-fase del passo V nel registro (sincronizza il dispositivo ai confini: solo orologio).
    pv_dd_solo_salti: bool = False
    pv_pm_ordine_strati: bool = False
    pv_golosa_pigra: bool = False
    pv_tempi_v: bool = False
    # 27/9 perdita: "kl" = KL(p_T || p_S) (EB2, default: identica al bit); "cakld" = BitDistiller Eq. 5,
    # gamma KL(S||T) + (1 - gamma) KL(T||S); gamma None = misurato (media di p_T(y vero) sulle righe del passo 0)
    perdita: str = "kl"
    gamma: Optional[float] = None
    # 27/9 --peso-token (ramo kl-pesata, TECNICHE_E2E_PV.md §35, era §31 nel ramo): peso per token della perdita (kl o cakld) nel
    # micro-lotto, calcolato SENZA gradiente dalla perdita del token l_t. "uniforme" = sum_t l_t (default, identico al
    # bit); "kl" = w_t = min(c (l_t / media l)^peso_alfa, peso_max), c tale che la media di w nel micro-lotto sia 1;
    # "topk" (Rho-1, arXiv 2404.07965 Eq. 4) = w_t = N/k sui k = ceil(peso_frazione N) token a perdita piu' alta, 0
    # altrove. Held-out, sonda del passo V e curvatura restano UNIFORMI; kl_train nel registro resta la perdita uniforme.
    peso_token: str = "uniforme"
    peso_alfa: float = _PESO_DEFAULT["peso_alfa"]
    peso_max: float = _PESO_DEFAULT["peso_max"]
    peso_frazione: float = _PESO_DEFAULT["peso_frazione"]
    # 27/9 trainer-mc (righe a scelta multipla, `SorgenteConMC`): righe_mc = righe mc per passo (su righe_per_passo; 0 =
    # spento, identico al bit); peso_token "risposta" = peso 1 a ogni token e 1 + lambda alle posizioni-risposta, lambda
    # tale che sum_{t in A} w_t = beta_risposta * sum_t w_t sul passo (`lambda_risposta`); gamma_lettere = G del termine
    # in piu' G * sum_{a in A} w_a KL(p4_T || p4_S) sui 4 id delle lettere (`kl_lettere`; 0 = spento).
    righe_mc: int = 0
    beta_risposta: float = _BETA_RISPOSTA_DEFAULT
    gamma_lettere: float = 0.0
    # 12/10 OPD (e2e_onpolicy.py, arXiv 2609.26708): con peso_token "generati" la perdita del micro-lotto e' il termine
    # OPD alle posizioni generate dallo studente (vantaggio GRPO + KL col maestro, modo opd_kl); opd_pivot = PIVOT-Q
    # (arXiv 2610.05745, "" = spento); opd_eakld_k = K di EdgeRazor (arXiv 2605.04062) con opd_kl "eakld".
    opd_beta: float = 1.0
    opd_kl: str = "campionata"
    opd_eakld_k: int = 16
    opd_pivot: str = ""
    # 27/9 integrata (nota del corpus mc, LEGGIMI_corpus-mc §8: le righe mc sono a QUARTI di rotazione, le prime 192 tutte
    # rotazione 0): righe_mc_permutate = anche il PRIMO giro delle righe mc permutato col seme (come i giri dopo), invece
    # dell'ordine del teacher mc. Spento = identico al bit. Sta nel segmento del piano ("permutate": True).
    righe_mc_permutate: bool = False
    # 28/9 --righe-base-da N (EB4 dal GGUF riforgiato, TECNICHE §41): SOLO in una corsa NUOVA il flusso base parte dalla
    # riga N dell'ordine del teacher invece che da 0. Serve quando il teacher e' condiviso e le sue prime righe sono state
    # CONSUMATE da un'altra corsa (q8-EB-pieno: blocchi 0000-0003 = righe 0-511 consumati da EB2): col piano di sempre la
    # corsa nuova aspetterebbe per sempre righe che il teacher non rifa' piu' (marcatore .consumato). Sta nel primo
    # segmento del piano ("cursore_base": N, `_segmenti_mc`); il gamma misurato e i blocchi consumabili partono da N.
    # 0 = identico al bit. Campo di configurazione: alla ripresa vale quello della corsa (cambiarlo solleva).
    righe_base_da: int = 0
    # 27/9 --allena-f32: famiglie di tensori F32 del GGUF allenate insieme a theta (vuota = identico al bit), gruppo
    # AdamW proprio con lr_f32. Default 1e-5: lr iniziale di Norm Tweaking (arXiv 2309.02784, Adam sulle sole norme,
    # «un lr grande danneggia»); Router KD (arXiv 2603.02217) usa 5e-5 sul solo router ma per UNA epoca.
    allena_f32: tuple = ()
    lr_f32: float = 1e-5
    # 28/9 --densi-ternari (ramo densi-ternari, TECNICHE_E2E_PV.md §47): famiglie DENSE non esperte gia' ternarie nel GGUF
    # (variante "A": attn_q/k/v/output, attn_qkv, attn_gate, ssm_out, ffn_gate/up/down_shexp in TQ1_0R_1024 / TQ1_0)
    # allenate come esperto UNICO: scale theta nel gruppo di theta (stesso lr, Adam/Sophia, EMA), codici col passo V
    # (stessi criteri, budget rho per matrice), registro ed export come gli esperti. Vuota = identico al bit. Campo del
    # MODELLO: cambiarlo alla ripresa solleva (e la firma dello studente lo porta).
    densi_ternari: tuple = ()
    # 28/9 --f32-ripristina (thor_v2/analisi_strumenti.md: con un lr_f32 unico il router, il gate dell'esperto condiviso e i
    # gate SSM si spostano dell'1-3 % relativo e il dominio strumenti sale da 0,65 a 1,04): famiglie F32 RIPORTATE al
    # valore del GGUF alla ripresa, in TUTTI gli strati (anche quelli congelati da --strati-allenati). Vuota = identico al
    # bit. Decisione da qui in avanti: una famiglia che compare qui per la prima volta (rispetto alla config della ripresa)
    # torna al GGUF; se resta in --allena-f32 riparte da li' con Adam da zero, se e' TOLTA da --allena-f32 la toglierla
    # e' ammesso (senza --f32-ripristina togliere una famiglia allenata solleva, come sempre) e il modello usa il GGUF.
    # Rilanciare con la stessa riga NON la riporta di nuovo (idempotente); in stato["f32_ripristini"] passo e norme.
    f32_ripristina: tuple = ()
    # 27/9 --lora-nonesperti R (e2e_lora.py): adattatori B A di rango R sui k-quant NON esperti delle famiglie
    # `lora_famiglie` (q/k/v/o, GDN qkv/gate/out, esperto condiviso), in un gruppo AdamW proprio con lr_lora. 0 = spenta
    # (identico al bit). alpha None = R (s = alpha/R = 1). lr 1e-4: quello tipico di LoRA/QLoRA con Adam (NON misurato
    # qui). All'export Delta W = s B A si riquantizza nello stesso tipo (e2e_esporta --lora-modo): vedi TECNICHE §25
    # per quanta parte di Delta W sopravvive.
    lora_rango: int = 0
    lora_alpha: Optional[float] = None
    lora_famiglie: tuple = ()
    lr_lora: float = 1e-4
    lora_tipi: tuple = ()                  # solo i tensori di questi tipi GGUF (vuota = tutti); §25: Q8_0 sopravvive
    # 27/9 --lora-init (ramo lora-ga, TECNICHE §37): "zero" = A casuale, B = 0 (default, identico al bit); "ga" = LoRA-GA
    # (arXiv 2407.05000 §3.4, codice ufficiale: direzione ArB2r, scala «stable»): prima del primo passo con la LoRA un
    # avanti+indietro sulle righe di quel passo (le prime `lora_ga_righe`, 0 = tutte) da' il gradiente PIENO dW di ogni
    # tensore; A = V[:, :r]^T c, B = U[:, r:2r] c, c = out^(1/4)/sqrt(lora_ga_gamma). Il modello resta identico al bit
    # (W_eff = W + s (B A - B0 A0), A0/B0 fermi). Costo: un passo in piu' (indietro fp32 se quello della ricetta e' a 16 bit).
    lora_init: str = "zero"
    lora_ga_gamma: float = 16.0
    lora_ga_righe: int = 0
    # 28/9 --lora-ripristina (prova_strumenti_f32 sul passo 20 di EB3: LoRA spenta = KL strumenti 1,044 -> 0,750, costo
    # piccolo sulla conoscenza): "" = nulla (identico al bit). Decisione ALLA RIPRESA, su TUTTI gli adattatori (anche degli
    # strati congelati), registrata in stato["lora_ripristini"] (passo, |Delta W| scartato per adattatore) e nel registro:
    #   "zero"  Delta W = 0 esatto (`e2e_lora.AdattatoreLoRA.azzera_delta`: B = 0 con A tenuto; LoRA-GA A = A0, B = B0),
    #           Adam della LoRA da zero, la LoRA CONTINUA ad allenarsi (tipicamente con --lr-lora piu' basso). Avviene
    #           quando "zero" compare (config della ripresa diversa): rilanciare con la stessa riga NON riazzera.
    #   "ferma" Delta W = 0 e LoRA NON piu' allenata: fuori dall'ottimizzatore (nessun gruppo LoRA), niente gradiente, niente
    #           LoRA-GA, niente A/B in riprese e istantanee (l'export e' quello senza LoRA). E' uno STATO finche' la riga lo
    #           dice (anche in una corsa nuova); toglierlo = LoRA accesa alla ripresa (decisione di sempre).
    lora_ripristina: str = ""
    # 27/9 --lota-rango R (e2e_lota.py, LoTA-QAF arXiv 2505.18724): adattatori TERNARI B A di rango R per esperto sui
    # CODICI (gate/up/down di ogni strato), codici = clamp(T0 + sign(B A) 1[|B A| > omega]); 0 = spenta (identico al bit).
    # omega None = 0,75 R (paper). top_p: quota delle voci di A e di B aggiornate per tensore, schedula del codice del paper
    # top_p -> top_p_fine (80% dei passi) -> top_p_coda. lota_lr: tasso della media del gradiente (1 = t-SignSGD del paper,
    # senza memoria). lota_sonda: regione di fiducia (sonda prima/dopo i codici nuovi; sale -> annulla, p / 2).
    lota_rango: int = 0
    lota_omega: Optional[float] = None
    lota_top_p: float = 0.05
    lota_top_p_fine: float = 1e-3
    lota_top_p_coda: float = 1e-4
    lota_lr: float = 1.0
    lota_tau: float = 1e-9
    lota_sonda: bool = True
    lota_init: str = "paper"               # "gradiente": A dallo schizzo Omega^T G del primo passo LoTA (variante nostra)
    lota_sonda_tolleranza: float = 0.0     # la sonda tiene i codici se la perdita non sale oltre questa quota relativa
    lota_sonda_finestra: int = 0           # > 0: sonda FISSA a finestra (ritorno al migliore dopo N passi senza meglio)
    escludi_domini: tuple = ("wiki",)
    salite_per_arresto: int = 2
    rotazione: bool = False
    tieni_riprese: int = 1           # 27/9: 1 (ripresa+accumulo ~7,7 GB a passo, disco pieno); la vecchia si cancella DOPO la nuova
    attesa_ogni_s: float = 20.0
    # 27/9 velocita': held-out ridotto. 0 = ogni misura e' piena (le 64 righe, come prima). Con N > 0: ogni
    # `heldout_ogni` passi una misura RAPIDA su N righe equispaziate (solo registro), ogni `heldout_pieno_ogni`
    # passi (0 = heldout_ogni) la misura PIENA: migliore, arresto e 2a epoca SOLO sulle misure piene.
    heldout_righe_rapide: int = 0
    heldout_pieno_ogni: int = 0
    # 27/9 integrata (richiesta di Angelo 16:15, e2e_attribuzione.py): attribuzione per componente a OGNI passo su una
    # sonda fissa di `attribuzione_righe` righe del held-out (lascia-uno-fuori per theta/F32/LoRA/codici e, ogni
    # `attribuzione_solo_ogni` passi, solo-uno), diagnostica per componente e tempi per fase; riga JSON per passo in
    # <corsa>/registro_componenti.jsonl. Campi di ESECUZIONE: misura e ripristina al bit, l'allenamento non cambia.
    attribuzione: bool = False
    attribuzione_righe: int = 2
    attribuzione_solo_ogni: int = 1
    # 27/9 integrata (dato delle 16:37: MMLU dell'EMA +1,0 in 4 passi mentre la KL held-out peggiorava: il migliore per
    # metrica non basta): istantanea ESPORTABILE a ogni passo, theta e theta_EMA (sottocartella ema/), cartella passo_<N>
    # in `istantanee_dir` (vuoto = la cartella della corsa), le ultime `istantanee_tieni`. Campi di ESECUZIONE. Un errore
    # (disco pieno) non ferma la corsa: si registra e si prosegue.
    istantanea_ogni_passo: bool = False
    # 1/10 tt5-ottimizzato: tempi per fase in OGNI passo (P = gradiente + Adam fino alla sincronizzazione del passo, V,
    # valutazioni, salvataggi) nel registro, senza --attribuzione. Campo di ESECUZIONE: nessun calcolo in piu'.
    tempi_fasi: bool = False
    # 1/10 tt5-ottimizzato (AUDIT cura 8): ripresa e stato V-Bop scritti da un filo a parte (copia CPU al momento del
    # salvataggio, stessi file). Campo di ESECUZIONE.
    scritture_asincrone: bool = False
    # 2/10 tt5-fase-v (FASE_V.md §6): con --scritture-asincrone la copia della ripresa e dello stato V-Bop si fa sul
    # DISPOSITIVO (clone in VRAM) invece che sulla CPU, e il filo dello scrittore la porta in RAM un tensore alla volta
    # mentre scrive (torch.save copia ogni storage del dispositivo sulla CPU subito prima di scriverlo): picco di RAM host
    # da ~6 GB a un solo tensore. File IDENTICI al bit: le storage in sosta si etichettano «cpu» (`_tag_sosta`), come
    # quelle della copia CPU. Senza VRAM libera sufficiente (margine 8 GiB) si torna alla copia CPU di prima. ESECUZIONE.
    scritture_sosta_dispositivo: bool = False
    # 27/9 integrata: con --rotazione i blocchi del teacher usati si CANCELLANO dopo la ripresa che li copre (default, come
    # prima). False (--non-consumare) = non si toccano: EB3 legge lo stesso teacher di EB2, che deve restare intatto.
    consuma_blocchi: bool = True
    istantanee_dir: str = ""
    istantanee_tieni: int = 2
    istantanee_libero_min_gib: float = 5.0
    # 27/9 velocita': media esponenziale di theta (0 = spenta): theta_EMA <- d theta_EMA + (1 - d) theta dopo ogni
    # passo (fp32, da theta = 0). L'held-out pieno misura anche theta_EMA ("metrica_ema"; migliore e arresto
    # restano su theta). Nelle riprese (al bit), in migliore_<passo>/ema/ e in migliore_ema_<passo>/ (il migliore
    # per metrica_ema): export da EMA con `e2e_esporta.py --istantanea migliore_ema_<passo>` (o migliore_<passo>/ema).
    ema_theta: float = 0.0
    # 27/9 ramo newton-theta (TECNICHE_E2E_PV.md §19, §36, era §31/§35 nel ramo): ottimizzatore del gruppo theta. "adamw" = come sempre
    # (default, identico al bit); "sophia" = Newton diagonale stile Sophia (e2e_sophia.py, arXiv 2305.14342):
    # theta -= lr clip(m / max(sophia_gamma h, sophia_eps K), 1), h = EMA (sophia_beta2) della curvatura Gauss-Newton
    # per gruppo misurata nei micro-lotti di curvatura del passo V (serve pv_codici). F32 e LoRA restano AdamW.
    # gamma 0.01 = valore del README di Sophia; la diagnosi del 27/9 (§36) dice che qui |m/h| ha mediana >> 1: con
    # 0.01 il clip scatta ovunque (passo = lr segno(m)); per il 10-50% di coordinate non clippate serve gamma ~ quantili
    # 0,1-0,5 di |m/h| (registrati nel log: "sophia: non clippati").
    ottimizzatore_theta: str = "adamw"
    sophia_gamma: float = 0.01
    sophia_beta2: float = 0.99
    sophia_eps: float = 1e-15              # in unita' vere: il denominatore minimo e' sophia_eps * K
    # 27/9 velocita' (e2e_bf16x3): testa KL dello studente (logit e dL/dh) "fp32" (default: identica al bit) o
    # "bf16x3" (3 GEMM bf16, accumulo fp32, WS scomposta una volta); logit del maestro "fp32" (default) o "bf16"
    # (W_T e h_T sono gia' bf16: UNA GEMM bf16 con accumulo e uscita fp32, esatta salvo l'ordine delle somme).
    testa_kl: str = "fp32"
    logit_maestro: str = "fp32"
    # 27/9 ramo dominio-strati. peso_dominio: ((dominio, peso > 0), ...) in ordine di nome, pesi della perdita per DOMINIO
    # della riga normalizzati sul passo (`pesi_dominio_passo`); vuota = spento (identico al bit). strati_allenati: indici
    # ordinati degli strati allenati (`Studente.congela_strati`); vuota = tutti (identico al bit).
    peso_dominio: tuple = ()
    strati_allenati: tuple = ()
    # Integrazione allena-f32 + velocita' (27/9): --beta2 vale per ENTRAMBI i gruppi AdamW (theta e F32, anche alla
    # ripresa); la EMA resta SOLO su theta: metrica_ema = theta_EMA coi F32 CORRENTI (quelli che ema/ e migliore_ema_*
    # scrivono in f32.npz). Ragione: coi F32 a lr_f32 = 1e-5 la loro media conta poco, e una EMA anche dei F32 cambierebbe
    # il formato delle riprese (ema + f32_nomi, famiglie aggiunte alla ripresa) e l'export per un guadagno non misurato.


    def __post_init__(self):
        if isinstance(self.righe_base_da, bool) or not isinstance(self.righe_base_da, int) or self.righe_base_da < 0:
            raise ValueError(f"righe_base_da {self.righe_base_da!r}: intero >= 0")
        if self.istantanee_tieni < 1 or self.istantanee_libero_min_gib < 0:
            raise ValueError(f"istantanee_tieni {self.istantanee_tieni} >= 1, istantanee_libero_min_gib >= 0")
        if self.attribuzione_righe < 1 or self.attribuzione_solo_ogni < 0:
            raise ValueError(f"attribuzione_righe {self.attribuzione_righe} >= 1, attribuzione_solo_ogni "
                             f"{self.attribuzione_solo_ogni} >= 0")
        # 27/9 dominio-strati: forme canoniche (quelle che config.json scrive e la ripresa confronta)
        if not isinstance(self.peso_dominio, tuple) or not all(
                isinstance(x, tuple) and len(x) == 2 and isinstance(x[0], str) and x[0] and isinstance(x[1], float)
                and math.isfinite(x[1]) and x[1] > 0.0 for x in self.peso_dominio) or \
                [x[0] for x in self.peso_dominio] != sorted({x[0] for x in self.peso_dominio}):
            raise ValueError(f"peso_dominio {self.peso_dominio!r}: ((dominio, peso > 0 finito), ...) in ordine di nome, "
                             f"senza doppi (peso 0 = dominio abbandonato: vietato)")
        if not isinstance(self.strati_allenati, tuple) or not all(
                isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in self.strati_allenati) or \
                list(self.strati_allenati) != sorted(set(self.strati_allenati)):
            raise ValueError(f"strati_allenati {self.strati_allenati!r}: indici >= 0 crescenti, senza doppi")
        if self.perdita not in ("kl", "cakld"):
            raise ValueError(f"perdita {self.perdita!r}: 'kl' o 'cakld'")
        if self.gamma is not None and not 0.0 <= self.gamma <= 1.0:
            raise ValueError(f"gamma {self.gamma} fuori da [0, 1]")
        if self.peso_token not in PESI_TOKEN:
            raise ValueError(f"peso_token {self.peso_token!r}: uno di {PESI_TOKEN}")
        if not (math.isfinite(self.peso_alfa) and self.peso_alfa >= 0.0):
            raise ValueError(f"peso_alfa {self.peso_alfa}: finito e >= 0")
        if not (math.isfinite(self.peso_max) and self.peso_max >= 1.0):
            raise ValueError(f"peso_max {self.peso_max}: finito e >= 1 (la media dei pesi e' 1)")
        if not 0.0 < self.peso_frazione <= 1.0:
            raise ValueError(f"peso_frazione {self.peso_frazione} fuori da (0, 1]")
        # un parametro che non entra nella perdita scelta resta al suo default (config.json dice cio' che si calcola)
        usati = {"uniforme": (), "kl": ("peso_alfa", "peso_max"), "topk": ("peso_frazione",),
                 "risposta": (), "generati": ()}[self.peso_token]
        fermi = [k for k in ("peso_alfa", "peso_max", "peso_frazione")
                 if k not in usati and getattr(self, k) != _PESO_DEFAULT[k]]
        if fermi:
            raise ValueError(f"{', '.join(fermi)} senza effetto con peso_token {self.peso_token!r}")
        # 27/9 trainer-mc
        if not isinstance(self.righe_mc, int) or not 0 <= self.righe_mc < self.righe_per_passo:
            raise ValueError(f"righe_mc {self.righe_mc!r}: intero in [0, righe_per_passo = {self.righe_per_passo})")
        if not (math.isfinite(self.beta_risposta) and 0.0 < self.beta_risposta < 1.0):
            raise ValueError(f"beta_risposta {self.beta_risposta} fuori da (0, 1)")
        if self.peso_token != "risposta" and self.beta_risposta != _BETA_RISPOSTA_DEFAULT:
            raise ValueError(f"beta_risposta senza effetto con peso_token {self.peso_token!r}")
        if not (math.isfinite(self.gamma_lettere) and self.gamma_lettere >= 0.0):
            raise ValueError(f"gamma_lettere {self.gamma_lettere}: finito e >= 0")
        if not isinstance(self.righe_mc_permutate, bool) or (self.righe_mc_permutate and self.righe_mc == 0):
            raise ValueError("righe_mc_permutate: booleano, solo con righe_mc > 0")
        # 12/10 OPD
        import e2e_onpolicy as _OPD
        if self.opd_kl not in _OPD.MODI_KL or not (math.isfinite(self.opd_beta) and self.opd_beta >= 0.0) \
                or self.opd_eakld_k < 2:
            raise ValueError(f"opd_kl {self.opd_kl!r} in {_OPD.MODI_KL}, opd_beta {self.opd_beta} >= 0, opd_eakld_k >= 2")
        _OPD.pivot_da_testo(self.opd_pivot)
        if self.peso_token == "generati":
            if self.perdita != "cakld" or self.gamma != 1.0 or self.righe_mc != 0 or self.gamma_lettere != 0:
                raise ValueError("peso_token generati (OPD): perdita cakld con gamma 1 (KL inversa: sonda del passo V e "
                                 "held-out), senza righe mc ne' lettere")
        elif (self.opd_beta, self.opd_kl, self.opd_eakld_k, self.opd_pivot) != (1.0, "campionata", 16, ""):
            raise ValueError("opd_* senza effetto senza --peso-token generati")
        if (self.peso_token == "risposta" or self.gamma_lettere > 0) and self.righe_mc == 0:
            raise ValueError("peso_token risposta / gamma_lettere > 0 senza righe mc nel passo (righe_mc = 0)")
        if self.heldout_righe_rapide < 0 or self.heldout_pieno_ogni < 0:
            raise ValueError("heldout_righe_rapide e heldout_pieno_ogni >= 0")
        if not 0.0 <= self.ema_theta < 1.0:
            raise ValueError(f"ema_theta {self.ema_theta} fuori da [0, 1)")
        if not 0.0 < self.beta2 < 1.0:
            raise ValueError(f"beta2 {self.beta2} fuori da (0, 1)")
        if self.testa_kl not in ("fp32", "bf16x3"):
            raise ValueError(f"testa_kl {self.testa_kl!r}: 'fp32' o 'bf16x3'")
        if self.logit_maestro not in ("fp32", "bf16"):
            raise ValueError(f"logit_maestro {self.logit_maestro!r}: 'fp32' o 'bf16'")
        if not isinstance(self.allena_f32, tuple) or not all(isinstance(x, str) for x in self.allena_f32):
            raise ValueError(f"allena_f32 {self.allena_f32!r}: tupla di nomi di famiglia")
        if not isinstance(self.f32_ripristina, tuple) or not all(isinstance(x, str) for x in self.f32_ripristina):
            raise ValueError(f"f32_ripristina {self.f32_ripristina!r}: tupla di nomi di famiglia")
        if not self.lr_f32 > 0:
            raise ValueError(f"lr_f32 {self.lr_f32} <= 0")
        if self.ottimizzatore_theta not in ("adamw", "sophia"):
            raise ValueError(f"ottimizzatore_theta {self.ottimizzatore_theta!r}: 'adamw' o 'sophia'")
        if self.ottimizzatore_theta == "sophia" and not self.pv_codici:
            raise ValueError("ottimizzatore_theta sophia: la curvatura delle scale si misura nei micro-lotti di curvatura "
                             "del passo V (serve --pv-codici)")
        if not self.sophia_gamma > 0 or not 0.0 < self.sophia_beta2 < 1.0 or not self.sophia_eps > 0:
            raise ValueError(f"sophia: gamma {self.sophia_gamma} > 0, beta2 {self.sophia_beta2} in (0, 1), "
                             f"eps {self.sophia_eps} > 0")
        if self.pv_modo not in ("pv", "vbop"):
            raise ValueError(f"pv_modo {self.pv_modo!r}: 'pv' o 'vbop'")
        if not 0.0 < self.pv_gamma <= 1.0:
            raise ValueError(f"pv_gamma {self.pv_gamma} fuori da (0, 1]")
        if self.pv_curvatura_ogni < 1 or (self.pv_curvatura_ogni > 1 and self.pv_modo != "vbop"):
            raise ValueError(f"pv_curvatura_ogni {self.pv_curvatura_ogni}: >= 1, e > 1 solo con pv_modo vbop (serve "
                             "la EMA della curvatura)")
        # 1/10 TT5-ottimizzato: vbop con pv_passi_accumulo N > 1 = decisione ogni N passi, EMA di V-Bop a ogni passo
        # (prima sollevava: «la memoria fra i passi e' gia' la EMA di V-Bop»; vedi il commento del campo)
        if not self.pv_rho_max > 0:
            raise ValueError(f"pv_rho_max {self.pv_rho_max} <= 0")
        if self.pv_accetta not in ("bh", "guadagno"):
            raise ValueError(f"pv_accetta {self.pv_accetta!r}: 'bh' o 'guadagno'")
        for nome in ("pv_z_min", "pv_z_guadagno_min"):
            v = getattr(self, nome)
            if not v >= 0.0 or (self.pv_accetta == "bh" and v != 0.0):
                raise ValueError(f"{nome} {v}: >= 0, e > 0 solo con pv_accetta 'guadagno'")
        if self.pv_ricentra and not self.pv_sonda_dosi:
            raise ValueError("pv_ricentra solo con pv_sonda_dosi (il ricentro parte dalla dose confermata sulla sonda B)")
        if int(self.pv_ricentra_giri) < 1:
            raise ValueError(f"pv_ricentra_giri {self.pv_ricentra_giri} < 1")
        if not isinstance(self.pv_dosi_frazioni, tuple):
            raise ValueError(f"pv_dosi_frazioni {self.pv_dosi_frazioni!r}: tupla di frazioni")
        if self.pv_dosi_frazioni:
            f = [float(x) for x in self.pv_dosi_frazioni]
            if not self.pv_sonda_dosi or any(not 0.0 < x <= 1.0 for x in f) or any(b <= a for a, b in zip(f, f[1:])):
                raise ValueError(f"pv_dosi_frazioni {self.pv_dosi_frazioni}: crescenti in (0, 1], solo con pv_sonda_dosi")
        if self.pv_punto_medio and not self.pv_sonda_dosi:
            raise ValueError("pv_punto_medio solo con pv_sonda_dosi (i prezzi al punto medio ordinano le dosi)")
        if self.pv_pm_primitiva not in ("famiglia", "tipo", "strato", "esperto") or int(self.pv_pm_max) < 1:
            raise ValueError(f"pv_pm_primitiva {self.pv_pm_primitiva!r} / pv_pm_max {self.pv_pm_max}: "
                             "famiglia|tipo|strato|esperto, pv_pm_max >= 1")
        if self.pv_pm_lettere and not (self.pv_punto_medio and int(self.righe_mc) > 0):
            raise ValueError("pv_pm_lettere solo con pv_punto_medio e righe_mc > 0 (obiettivo sulle righe mc del passo dopo)")
        if self.pv_pm_golosa and not self.pv_punto_medio:
            raise ValueError("pv_pm_golosa solo con pv_punto_medio (le dosi golose sono primitive al punto medio)")
        if self.pv_ricentra_obiettivo and not (self.pv_ricentra and self.pv_pm_golosa and self.pv_pm_lettere):
            raise ValueError("pv_ricentra_obiettivo solo con pv_ricentra, pv_pm_golosa e pv_pm_lettere")
        if self.pv_riuso and not self.pv_sonda_dosi:
            raise ValueError("pv_riuso solo con pv_sonda_dosi (riusa fra le prove della fase V a dosi)")
        if self.pv_golosa_pigra and not self.pv_pm_golosa:
            raise ValueError("pv_golosa_pigra solo con pv_pm_golosa")
        if self.pv_pm_ordine_strati and not self.pv_punto_medio:
            raise ValueError("pv_pm_ordine_strati solo con pv_punto_medio")
        if self.scritture_sosta_dispositivo and not self.scritture_asincrone:
            raise ValueError("scritture_sosta_dispositivo solo con scritture_asincrone")
        if int(self.pv_riuso_lotti) < 1:
            raise ValueError(f"pv_riuso_lotti {self.pv_riuso_lotti} < 1")
        if self.pv_codici and self.pv_sonda_dosi and self.righe_per_passo < 2 * self.righe_per_micro:
            raise ValueError(f"pv_sonda_dosi: servono due micro-lotti del passo dopo (sonde A e B): righe_per_passo "
                             f"{self.righe_per_passo} < 2 x righe_per_micro {self.righe_per_micro}")
        if self.pv_codici and self.pv_modo == "vbop" and \
                -(-self.righe_per_passo // self.righe_per_micro) < self.pv_n_proposta + 2:
            raise ValueError(f"pv_modo vbop: servono almeno pv_n_proposta + 2 = {self.pv_n_proposta + 2} micro-lotti a "
                             f"passo (varianza di verifica), ce ne sono {-(-self.righe_per_passo // self.righe_per_micro)}")
        if not isinstance(self.lora_rango, int) or self.lora_rango < 0:
            raise ValueError(f"lora_rango {self.lora_rango!r}: intero >= 0")
        if not isinstance(self.lora_famiglie, tuple) or not all(isinstance(x, str) for x in self.lora_famiglie):
            raise ValueError(f"lora_famiglie {self.lora_famiglie!r}: tupla di nomi di famiglia")
        if self.lora_rango == 0 and (self.lora_famiglie or self.lora_alpha is not None or self.lora_tipi):
            raise ValueError("lora_famiglie/lora_alpha/lora_tipi senza lora_rango > 0")
        if not isinstance(self.lora_tipi, tuple) or not all(isinstance(x, str) for x in self.lora_tipi):
            raise ValueError(f"lora_tipi {self.lora_tipi!r}: tupla di tipi GGUF")
        if self.lora_rango > 0 and not self.lora_famiglie:
            raise ValueError("lora_rango > 0 senza famiglie")
        if self.lora_alpha is not None and not self.lora_alpha > 0:
            raise ValueError(f"lora_alpha {self.lora_alpha} <= 0")
        if not self.lr_lora > 0:
            raise ValueError(f"lr_lora {self.lr_lora} <= 0")
        if self.lora_init not in ("zero", "ga"):
            raise ValueError(f"lora_init {self.lora_init!r}: 'zero' o 'ga'")
        if self.lora_init == "ga" and self.lora_rango == 0:
            raise ValueError("lora_init 'ga' senza lora_rango > 0")
        if not self.lora_ga_gamma > 0:
            raise ValueError(f"lora_ga_gamma {self.lora_ga_gamma} <= 0")
        if not isinstance(self.lora_ga_righe, int) or self.lora_ga_righe < 0:
            raise ValueError(f"lora_ga_righe {self.lora_ga_righe!r}: intero >= 0 (0 = tutte le righe del passo)")
        if self.lora_init != "ga" and (self.lora_ga_gamma != 16.0 or self.lora_ga_righe != 0):
            raise ValueError("lora_ga_gamma/lora_ga_righe senza lora_init 'ga'")
        if self.lora_ripristina not in LORA_RIPRISTINI:
            raise ValueError(f"lora_ripristina {self.lora_ripristina!r}: {LORA_RIPRISTINI}")
        if self.lora_ripristina and self.lora_rango == 0:
            raise ValueError(f"lora_ripristina {self.lora_ripristina!r} senza lora_rango > 0")
        if not isinstance(self.lota_rango, int) or self.lota_rango < 0:
            raise ValueError(f"lota_rango {self.lota_rango!r}: intero >= 0")
        if self.lota_rango == 0 and self.lota_omega is not None:
            raise ValueError("lota_omega senza lota_rango > 0")
        if self.lota_rango > 0:
            if self.pv_codici:
                raise ValueError("lota_rango > 0 con pv_codici: LoTA e il passo V muovono gli stessi codici con basi diverse "
                                 "(e2e_lota.py, «esclusiva»); i registri PV di una fase precedente si rigiocano congelati")
            if self.strati_allenati:
                # integrata (merge dominio-strati-b): i due rami non si conoscevano; LoTA sugli strati congelati (senza
                # indietro sotto il primo allenato) non e' definita ne' provata: rifiutata invece di un comportamento a caso
                raise ValueError("lota_rango > 0 con strati_allenati: combinazione non provata (LoTA muove i codici di "
                                 "tutti gli strati, l'indietro si ferma al primo allenato)")
            config_lota(self)                  # solleva se omega/top-p/lr/tau sono fuori campo


# ═══════════════════════════════════════ sorgenti del teacher ═══════════════════════════════════════
class SorgenteCache:
    """Adattatore di `e2e_teacher.CacheMaestro` (formato su disco del teacher). Solo lettura, salvo `consuma`."""

    def __init__(self, dir_corsa):
        import e2e_teacher as T  # sola lettura
        self.T = T
        self.c = T.CacheMaestro(dir_corsa)
        self.dir = Path(dir_corsa)
        self.modo = self.c.modo
        self.piano = self.c.piano
        self._blocco = {}
        for b in self.piano["blocchi"]:
            for i in range(b["da"], b["a"]):
                self._blocco[(b["lato"], i)] = b["nome"]

    def firma(self) -> dict:
        p = self.dir / "piano.json"
        return {"teacher": str(self.dir), "piano_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                "modo": self.modo, "modello": self.piano["modello"], "corpus": self.piano["corpus"]}

    def righe_previste(self, lato: str) -> list:
        return [i for b in self.piano["blocchi"] if b["lato"] == lato for i in range(b["da"], b["a"])]

    def blocchi(self, lato: str) -> list:
        return [(b["nome"], b["da"], b["a"]) for b in self.piano["blocchi"] if b["lato"] == lato]

    def blocco_di(self, lato: str, i: int) -> str:
        return self._blocco[(lato, i)]

    def aggiorna(self) -> None:
        self.c.aggiorna()

    def disponibile(self, lato: str, i: int) -> bool:
        return self.c.disponibile(lato, i)

    def ids(self, lato: str, i: int) -> np.ndarray:
        return self.c.ids(lato, i)

    def inizi(self, lato: str, i: int) -> Optional[np.ndarray]:
        """None nel modo «riga» (la riga e' UNA sequenza, come il teacher); 0/1 nel modo «segmenti»."""
        if self.modo == "riga":
            return None
        cu = self.c.cu_seqlens(lato, i)
        f = np.zeros(cu[-1], np.int32)
        f[cu[:-1]] = 1
        return f

    def h(self, lato: str, i: int) -> torch.Tensor:
        return self.c.h(lato, i)

    def dominio(self, lato: str, i: int) -> str:
        return self.c.info(lato, i)["dominio"]

    def consuma(self, nome: str) -> None:
        self.c.consuma(nome)


class SorgenteMemoria:
    """Sorgente in memoria con la STESSA interfaccia (prove): righe {lato: [(ids, h bf16, dominio, inizi|None)]},
    blocchi {lato: [(nome, da, a)]}. `consumati` registra i blocchi consumati, `pronte` quante righe del lato
    train sono «sigillate» (per provare l'attesa)."""

    def __init__(self, righe: dict, blocchi: dict, modo: str = "riga"):
        self.r = righe
        self.b = blocchi
        self.modo = modo
        self.consumati: list = []
        self.pronte: Optional[int] = None

    def firma(self) -> dict:
        h = hashlib.sha256()
        for lato in sorted(self.r):
            for ids, hh, dom, _ in self.r[lato]:
                h.update(np.ascontiguousarray(ids).tobytes())
                h.update(hh.contiguous().view(torch.int16).numpy().tobytes())
                h.update(dom.encode())
        return {"teacher": "memoria", "sha256": h.hexdigest(), "modo": self.modo}

    def righe_previste(self, lato: str) -> list:
        return list(range(len(self.r[lato])))

    def blocchi(self, lato: str) -> list:
        return list(self.b.get(lato, [(f"{lato}/tutto", 0, len(self.r[lato]))]))

    def blocco_di(self, lato: str, i: int) -> str:
        return next(n for n, a, b in self.blocchi(lato) if a <= i < b)

    def aggiorna(self) -> None:
        pass

    def disponibile(self, lato: str, i: int) -> bool:
        if self.blocco_di(lato, i) in self.consumati:
            return False
        return self.pronte is None or lato != "train" or i < self.pronte

    def ids(self, lato, i):
        return self.r[lato][i][0]

    def inizi(self, lato, i):
        return self.r[lato][i][3]

    def h(self, lato, i):
        return self.r[lato][i][1]

    def dominio(self, lato, i):
        return self.r[lato][i][2]

    def consuma(self, nome: str) -> None:
        self.consumati.append(nome)


# ═══════════════════════════════════════ righe a scelta multipla (trainer-mc) ═══════════════════════════════════════
OFFSET_MC = 1 << 30          # indice virtuale nel lato "train" delle righe mc: OFFSET_MC + indice nel lato del teacher mc
LATO_HELD_MC = "heldout_mc"  # lato virtuale del held-out mc


@dataclass
class RisposteMC:
    """`risposte.pt` di UN lato mc, letto e verificato (`leggi_risposte`). per_riga: {indice della riga nel lato (= indice
    del teacher = posizione in `ordine` del corpus): (t int64 [k], giusta int64 [k])}; i logit in t predicono il token in
    t + 1, che e' la lettera id_lettere[giusta]. id_lettere int64 [4] (A, B, C, D)."""
    per_riga: dict
    id_lettere: torch.Tensor
    fonte: list
    sha256: str
    n: int

    def conta(self, i: int) -> int:
        r = self.per_riga.get(i)
        return 0 if r is None else int(r[0].numel())


def leggi_risposte(p, ids_di: Callable, righe: list, lunghezza: int) -> RisposteMC:
    """Legge `risposte.pt` (dict: "pos" int [n, 2] = (riga, t), "giusta" int [n] in 0..3, "id_lettere" int [4], "fonte"
    lista di n stringhe) e lo VERIFICA contro gli id delle righe (`ids_di(riga)`): ogni riga nel lato, 0 <= t < L - 1,
    coppie (riga, t) uniche, 4 lettere distinte, e soprattutto id[riga][t + 1] == id_lettere[giusta] per OGNI risposta
    (la lettera giusta sta davvero in t + 1: controlla anche il significato di «riga»). Qualunque difetto: ValueError."""
    p = Path(p)
    d = torch.load(p, map_location="cpu", weights_only=True)
    mancano = {"pos", "giusta", "id_lettere", "fonte"} - set(d)
    if mancano:
        raise ValueError(f"{p}: mancano {sorted(mancano)}")
    pos, giusta, id4 = (torch.as_tensor(d[k]).long() for k in ("pos", "giusta", "id_lettere"))
    fonte = list(d["fonte"])
    n = int(giusta.numel())
    if pos.shape != (n, 2) or giusta.shape != (n,) or id4.shape != (4,) or len(fonte) != n:
        raise ValueError(f"{p}: forme pos {tuple(pos.shape)}, giusta {tuple(giusta.shape)}, id_lettere "
                         f"{tuple(id4.shape)}, fonte {len(fonte)}: attese [n, 2], [n], [4], n")
    if len(set(id4.tolist())) != 4:
        raise ValueError(f"{p}: id_lettere {id4.tolist()} non distinti")
    if n == 0:
        raise ValueError(f"{p}: nessuna risposta")
    if bool(((giusta < 0) | (giusta > 3)).any()):
        raise ValueError(f"{p}: giusta fuori da 0..3")
    insieme = set(int(x) for x in righe)
    fuori = [int(r) for r in pos[:, 0].tolist() if int(r) not in insieme]
    if fuori:
        raise ValueError(f"{p}: {len(fuori)} risposte su righe fuori dal lato (prima {fuori[0]})")
    if bool(((pos[:, 1] < 0) | (pos[:, 1] >= lunghezza - 1)).any()):
        raise ValueError(f"{p}: t fuori da [0, {lunghezza - 1})")
    if len(set(map(tuple, pos.tolist()))) != n:
        raise ValueError(f"{p}: coppie (riga, t) ripetute")
    per_riga: dict = {}
    sbagliate, esempio = 0, None
    for r in sorted(set(pos[:, 0].tolist())):
        sel = (pos[:, 0] == r).nonzero().squeeze(1)
        t = pos[sel, 1]
        ordine = torch.argsort(t)
        t, g = t[ordine].contiguous(), giusta[sel][ordine].contiguous()
        ids = torch.from_numpy(np.asarray(ids_di(int(r)), dtype=np.int64))
        if ids.numel() != lunghezza:
            raise ValueError(f"{p}: riga {r} di {ids.numel()} token, attesi {lunghezza}")
        ko = ids[t + 1] != id4[g]
        if bool(ko.any()):
            sbagliate += int(ko.sum())
            if esempio is None:
                k = int(ko.nonzero()[0, 0])
                esempio = {"riga": int(r), "t": int(t[k]), "id_in_t+1": int(ids[t[k] + 1]), "atteso": int(id4[g[k]])}
        per_riga[int(r)] = (t, g)
    if sbagliate:
        raise ValueError(f"{p}: {sbagliate}/{n} risposte con la lettera giusta NON in t + 1 (primo caso {esempio}); "
                         "«riga» deve essere l'indice della riga nel lato (posizione in `ordine` del corpus)")
    return RisposteMC(per_riga=per_riga, id_lettere=id4, fonte=fonte, sha256=sha256_file(p), n=n)


def _lunghezza_lato(src, lato: str) -> int:
    """Token per riga del lato: dal piano del teacher (le righe consumate dalla rotazione non ci sono piu'), altrimenti
    dalla prima riga (sorgenti in memoria delle prove). Tutte le righe di un lato hanno la stessa lunghezza."""
    piano = getattr(src, "piano", None)
    if piano is not None:
        return int(piano["corpus"][lato]["L"])
    return int(np.asarray(src.ids(lato, src.righe_previste(lato)[0])).shape[0])


class SorgenteConMC:
    """27/9 trainer-mc. Sorgente del ciclo con la STESSA interfaccia di `SorgenteCache`: il teacher base per tutti i suoi
    lati, piu' le righe mc nel lato "train" agli indici OFFSET_MC + j (j = indice nel lato "train" del teacher mc) e il
    held-out mc nel lato virtuale LATO_HELD_MC. `righe_previste("train")`, `blocchi`, `consuma` e `firma()` sono QUELLI
    del teacher base (l'ordine dei blocchi base, i blocchi rotanti e il controllo della ripresa non cambiano); la firma
    mc e' `firma_mc()` e va nello stato a parte. Le righe mc non si consumano (si riusano a giri, `PianoRighe`).
    mc = (sorgente, RisposteMC) del lato "train" o None; held = (sorgente, lato, RisposteMC) o None."""

    def __init__(self, base, mc: Optional[tuple] = None, held: Optional[tuple] = None, firma_mc: Optional[dict] = None):
        self.base = base
        self.dir = getattr(base, "dir", None)
        self.modo = base.modo
        self.mc, self.risposte_train = mc if mc is not None else (None, None)
        self.held, self.lato_held, self.risposte_held = held if held is not None else (None, None, None)
        self._firma_mc = firma_mc
        lettere = [r.id_lettere for r in (self.risposte_train, self.risposte_held) if r is not None]
        if not lettere:
            raise ValueError("SorgenteConMC senza righe mc ne' held-out mc")
        if len(lettere) == 2 and not torch.equal(lettere[0], lettere[1]):
            raise ValueError(f"id_lettere diversi fra train mc {lettere[0].tolist()} e held-out mc {lettere[1].tolist()}")
        self.id_lettere = lettere[0]
        for s in (self.mc, self.held):
            if s is not None and s.modo != self.modo:
                raise ValueError(f"teacher mc in modo {s.modo!r}, base {self.modo!r}: devono vedere lo stesso contesto")
        L = _lunghezza_lato(base, "train")
        for nome, s, lato in (("train mc", self.mc, "train"), ("held-out mc", self.held, self.lato_held)):
            if s is not None and _lunghezza_lato(s, lato) != L:
                raise ValueError(f"{nome}: righe di {_lunghezza_lato(s, lato)} token, il teacher base {L}")
        if self.mc is not None and self.held is not None:        # held-out DISGIUNTO dalle righe mc di addestramento
            imp = {hashlib.sha256(np.ascontiguousarray(self.mc.ids("train", i), np.int32).tobytes()).hexdigest()
                   for i in self.mc.righe_previste("train") if self.mc.disponibile("train", i)}
            doppie = [i for i in self.held.righe_previste(self.lato_held) if self.held.disponibile(self.lato_held, i) and
                      hashlib.sha256(np.ascontiguousarray(self.held.ids(self.lato_held, i), np.int32).tobytes())
                      .hexdigest() in imp]
            if doppie:
                raise ValueError(f"held-out mc NON disgiunto: righe {doppie[:5]} uguali a righe mc di addestramento")

    # --- chi risponde per (lato, i) ---
    def _chi(self, lato: str, i: int) -> tuple:
        if lato == LATO_HELD_MC:
            if self.held is None:
                raise KeyError("nessun held-out mc")
            return self.held, self.lato_held, i
        if lato == "train" and i >= OFFSET_MC:
            if self.mc is None:
                raise KeyError(f"riga mc {i} senza teacher mc")
            return self.mc, "train", i - OFFSET_MC
        return self.base, lato, i

    def firma(self) -> dict:
        return self.base.firma()

    def firma_mc(self) -> dict:
        return json.loads(json.dumps(self._firma_mc, default=str))

    def righe_previste(self, lato: str) -> list:
        if lato == LATO_HELD_MC:
            return [] if self.held is None else list(self.held.righe_previste(self.lato_held))
        return self.base.righe_previste(lato)

    def righe_mc(self) -> list:
        return [] if self.mc is None else [OFFSET_MC + j for j in self.mc.righe_previste("train")]

    def blocchi(self, lato: str) -> list:
        return self.base.blocchi(lato)

    def blocco_di(self, lato: str, i: int) -> str:
        s, lt, j = self._chi(lato, i)
        return s.blocco_di(lt, j) if s is self.base else f"mc:{lt}/{s.blocco_di(lt, j)}"

    def aggiorna(self) -> None:
        for s in {id(x): x for x in (self.base, self.mc, self.held) if x is not None}.values():
            s.aggiorna()

    def disponibile(self, lato: str, i: int) -> bool:
        s, lt, j = self._chi(lato, i)
        return s.disponibile(lt, j)

    def ids(self, lato: str, i: int):
        s, lt, j = self._chi(lato, i)
        return s.ids(lt, j)

    def inizi(self, lato: str, i: int):
        s, lt, j = self._chi(lato, i)
        return s.inizi(lt, j)

    def h(self, lato: str, i: int):
        s, lt, j = self._chi(lato, i)
        return s.h(lt, j)

    def dominio(self, lato: str, i: int) -> str:
        s, lt, j = self._chi(lato, i)
        return s.dominio(lt, j)

    def consuma(self, nome: str) -> None:
        self.base.consuma(nome)

    def risposte(self, lato: str, i: int) -> Optional[tuple]:
        """(t, giusta) delle risposte della riga, o None (righe base, righe mc senza risposte)."""
        if lato == LATO_HELD_MC:
            return self.risposte_held.per_riga.get(i) if self.risposte_held is not None else None
        if lato == "train" and i >= OFFSET_MC and self.risposte_train is not None:
            return self.risposte_train.per_riga.get(i - OFFSET_MC)
        return None


def _risposte_accanto(cache, lato: str) -> Path:
    """risposte.pt accanto al file del corpus del lato (quello che il piano del teacher cita, sha256 verificato)."""
    c = cache.piano["corpus"][lato]
    f = Path(c["file"])
    if sha256_file(f) != c["sha256"]:
        raise RuntimeError(f"{f}: sha256 diverso da quello del piano del teacher mc")
    return f.parent / "risposte.pt"


def _ids_corpus(cache, lato: str) -> np.ndarray:
    """id del corpus del lato nell'ORDINE del teacher (`e2e_teacher.righe_lato`): indice i = riga i del teacher."""
    import e2e_teacher as T
    c = cache.piano["corpus"][lato]
    r = T.righe_lato(lato, Path(c["file"]).parent, cache.piano["variante"])
    if Path(r["file"]).resolve() != Path(c["file"]).resolve() or r["sha256"] != c["sha256"]:
        raise RuntimeError(f"{cache.dir} {lato}: il corpus riletto ({r['file']}) non e' quello del piano ({c['file']})")
    return r["ids"]


def apri_sorgente_mc(base, dir_mc: Optional[Path], dir_held: Optional[Path] = None, log: Callable = print):
    """--teacher-mc DIR [--teacher-mc-heldout DIR2] -> `SorgenteConMC`. Train mc = lato "train" di DIR (se c'e'); held-out
    mc = DIR2 (il suo lato "train", o "heldout" se ha solo quello: e' la corsa del teacher sulla sottocartella heldout/ del
    corpus mc, che ha il suo risposte.pt), altrimenti il lato "heldout" di DIR SOLO se il suo file del corpus sta in
    un'altra cartella di quello del train (altrimenti risposte.pt sarebbe lo stesso file: lato ignorato, registrato). Per
    ogni lato: risposte.pt accanto al file del corpus del piano (sha256 verificato), verificato contro gli id del corpus
    e contro gli id sigillati del teacher. Stesso modello del teacher base (sha256), stesso tipo di h; strumento diverso
    registrato. Senza DIR2: se DIR/heldout/piano.json esiste (corsa del teacher sul held-out mc messa DENTRO DIR), e' DIR2.
    """
    if dir_held is None and dir_mc is not None and (Path(dir_mc) / "heldout" / "piano.json").exists():
        dir_held = Path(dir_mc) / "heldout"
        log(f"held-out mc: {dir_held} (teacher dentro --teacher-mc)")
    firma = {"teacher_mc": None, "teacher_mc_heldout": None, "train": None, "heldout": None}
    mc = held = None
    fonti = []
    if dir_mc is not None:
        fonti.append(("mc", Path(dir_mc)))
    if dir_held is not None:
        fonti.append(("held", Path(dir_held)))
    aperte = {n: SorgenteCache(d) for n, d in fonti}
    for n, s in aperte.items():
        if isinstance(base, SorgenteCache):
            for k in ("sha256",):
                if s.piano["modello"][k] != base.piano["modello"][k]:
                    raise RuntimeError(f"teacher {n} {s.dir}: modello {s.piano['modello'][k][:12]} diverso dal teacher "
                                       f"base {base.piano['modello'][k][:12]}")
            if s.c.salva_tipo != base.c.salva_tipo:
                raise RuntimeError(f"teacher {n}: h in {s.c.salva_tipo}, base {base.c.salva_tipo}")
            if s.piano["strumento"]["sha256"] != base.piano["strumento"]["sha256"]:
                log(f"teacher {n}: strumento e2e_dump_logit diverso dal teacher base (sha256 "
                    f"{s.piano['strumento']['sha256'][:12]} contro {base.piano['strumento']['sha256'][:12]}): registrato")

    def voce(s, lato):
        p = _risposte_accanto(s, lato)
        ids = _ids_corpus(s, lato)
        righe = s.righe_previste(lato)
        risp = leggi_risposte(p, lambda i: ids[i], righe, int(s.piano["corpus"][lato]["L"]))
        for i in righe:                                       # i sigillati del teacher == corpus (al bit)
            if s.disponibile(lato, i) and not np.array_equal(np.asarray(s.ids(lato, i)), ids[i]):
                raise RuntimeError(f"{s.dir} {lato}[{i}]: id del teacher diversi dal corpus")
        f = {"teacher": str(s.dir), "lato": lato, "piano_sha256": s.firma()["piano_sha256"],
             "risposte": str(p), "risposte_sha256": risp.sha256, "righe": len(righe), "risposte_n": risp.n,
             "id_lettere": risp.id_lettere.tolist()}
        return risp, f

    if "mc" in aperte:
        s = aperte["mc"]
        firma["teacher_mc"] = str(s.dir)
        if "train" in s.piano["corpus"]:
            r, firma["train"] = voce(s, "train")
            mc = (s, r)
        cp = s.piano["corpus"]
        if "heldout" in cp and "held" not in aperte:
            if "train" in cp and Path(cp["heldout"]["file"]).parent == Path(cp["train"]["file"]).parent:
                log(f"teacher mc {s.dir}: lato heldout nella stessa cartella del train (risposte.pt condiviso): ignorato; "
                    "held-out mc con --teacher-mc-heldout (corsa del teacher su heldout/)")
            else:
                r, firma["heldout"] = voce(s, "heldout")
                held = (s, "heldout", r)
    if "held" in aperte:
        s = aperte["held"]
        firma["teacher_mc_heldout"] = str(s.dir)
        lato = "heldout" if "heldout" in s.piano["corpus"] else "train"
        r, firma["heldout"] = voce(s, lato)
        held = (s, lato, r)
    src = SorgenteConMC(base, mc, held, firma)
    log(f"teacher mc: train {firma['train'] and (firma['train']['righe'], firma['train']['risposte_n'])} (righe, "
        f"risposte), held-out {firma['heldout'] and (firma['heldout']['righe'], firma['heldout']['risposte_n'])}, "
        f"id lettere {src.id_lettere.tolist()}")
    return src


# ═══════════════════════════════════════ caricamento dei micro-lotti ═══════════════════════════════════════
@dataclass
class Lotto:
    righe: list
    ids: torch.Tensor                  # (B, T) int64, sul dispositivo
    hT: torch.Tensor                   # (B*T, H) bf16, sul dispositivo
    inizi: Optional[torch.Tensor]      # (B, T) int32 (CPU) o None
    domini: list


def _carica_cpu(src, lato: str, righe: list, pin: bool) -> tuple:
    ids = torch.from_numpy(np.stack([np.asarray(src.ids(lato, i), dtype=np.int64) for i in righe]))
    hs = [src.h(lato, i) for i in righe]
    h = torch.cat([x.reshape(-1, x.shape[-1]) for x in hs]) if len(hs) > 1 else hs[0].reshape(-1, hs[0].shape[-1]).clone()
    iz = [src.inizi(lato, i) for i in righe]
    inizi = None if iz[0] is None else torch.from_numpy(np.stack(iz).astype(np.int32))
    if pin:
        ids, h = ids.pin_memory(), h.pin_memory()
    return ids, h, inizi, [src.dominio(lato, i) for i in righe]


class Caricatore:
    """Anticipa il caricamento (disco -> memoria bloccata) del micro-lotto successivo in un filo a parte; la copia
    sul dispositivo e' asincrona (l'allocatore della memoria bloccata tiene il buffer finche' la copia non e'
    finita). Sul CPU: niente memoria bloccata, stessa sequenza."""

    def __init__(self, src, device: torch.device):
        self.src = src
        self.device = device
        self.pin = device.type == "cuda"
        self._pool = _cf.ThreadPoolExecutor(max_workers=1)
        self._fut: dict = {}

    def prenota(self, lato: str, righe: list) -> None:
        k = (lato, tuple(righe))
        if k not in self._fut:
            self._fut[k] = self._pool.submit(_carica_cpu, self.src, lato, list(righe), self.pin)

    def prendi(self, lato: str, righe: list) -> Lotto:
        k = (lato, tuple(righe))
        self.prenota(lato, righe)
        ids, h, inizi, dom = self._fut.pop(k).result()
        nb = self.pin
        SE.segno()                                     # 28/9: un micro-lotto = un'unita' di progresso (sentinella)
        return Lotto(list(righe), ids.to(self.device, non_blocking=nb), h.to(self.device, non_blocking=nb), inizi, dom)

    def chiudi(self) -> None:
        self._pool.shutdown(wait=True)


def attendi_righe(src, lato: str, righe: list, log: Callable, ogni_s: float = 20.0) -> float:
    """Aspetta che il teacher abbia sigillato le righe (blocchi rotanti: il teacher corre avanti). Ritorna i
    secondi di attesa (si registrano: un'attesa lunga e' tempo perso, non va nascosta)."""
    t0 = time.time()
    avvisato = False
    while True:
        mancano = [i for i in righe if not src.disponibile(lato, i)]
        if not mancano:
            return time.time() - t0
        if not avvisato:
            log(f"attesa del teacher: {len(mancano)} righe {lato} non ancora sigillate (prima {mancano[0]})")
            avvisato = True
        time.sleep(ogni_s)
        src.aggiorna()


# ═══════════════════════════════════════ perdita KL (fusa) ═══════════════════════════════════════
def _logit_maestro(hT: torch.Tensor, WT: torch.Tensor, dtype) -> torch.Tensor:
    """In fp32 e' ESATTAMENTE `e2e_teacher.logit_maestro` (l'unica definizione, §2.3); fp64 solo nelle prove.
    27/9 `--logit-maestro bf16`: WT bf16 (i byte della testa del teacher) e hT bf16 (la cache): UNA GEMM bf16 con
    accumulo e uscita fp32 (`e2e_bf16x3.mm_bf16`), gli stessi prodotti esatti di `logit_maestro` salvo l'ordine
    delle somme fp32 (prova X5)."""
    if WT.dtype == torch.bfloat16:
        if dtype != torch.float32 or hT.dtype != torch.bfloat16:
            raise TypeError(f"logit del maestro bf16: hT {hT.dtype}, calcolo {dtype} (attesi bf16 e float32)")
        import e2e_bf16x3 as X3
        return X3.mm_bf16(hT, WT.T)
    if dtype == torch.float32:
        import e2e_teacher as T
        return T.logit_maestro(hT, WT)
    return hT.to(dtype) @ WT.to(dtype).T


def _logit_studente(hS: torch.Tensor, WS) -> torch.Tensor:
    """hS @ WS^T; con `--testa-kl bf16x3` WS e' un `e2e_bf16x3.TestaX3` (scomposta una volta)."""
    if isinstance(WS, torch.Tensor):
        return hS @ WS.T
    return WS.logit(hS)


def _per_testa(p: torch.Tensor, WS) -> torch.Tensor:
    """p @ WS (dL/dh dai logit); bf16x3 con una TestaX3."""
    if isinstance(WS, torch.Tensor):
        return p @ WS
    return WS.per(p)


def testa_studente(st, testa_kl: str):
    """La testa dello studente nella forma chiesta: tensore fp32 (Q5_K dequantizzata) o TestaX3 (hi/lo bf16: la
    copia fp32 si libera)."""
    W = st.testa(torch.float32)
    if testa_kl == "fp32":
        return W
    import e2e_kernel_qat as KQ                 # 9/10 livello fp16 (E2E_KQAT=testa1): un prodotto fp16 per verso
    if KQ.attivo("testa1"):
        return KQ.TestaX1(W)
    if KQ.attivo("testa3"):                     # 9/10 livello esatto: tre prodotti fp16 divisi in UN lancio
        return KQ.TestaX3K(W)
    import e2e_bf16x3 as X3
    return X3.TestaX3(W)


def teste(st, dev, ca: "ConfigAllena") -> tuple:
    """(WS, WT, sha256 della testa del teacher) secondo `ca.testa_kl` e `ca.logit_maestro`."""
    WS = testa_studente(st, ca.testa_kl)
    WT, sha_t = testa_maestro(dev, torch.bfloat16 if ca.logit_maestro == "bf16" else torch.float32)
    return WS, WT, sha_t


def kl_pezzo(hS: torch.Tensor, WS: torch.Tensor, hT: torch.Tensor, WT: torch.Tensor, gradiente: bool,
             bersagli: Optional[torch.Tensor] = None, gamma: Optional[float] = None, top1: bool = False) -> dict:
    """UN pezzo di token. KL(p_T || p_S) per token (fp64), e se `gradiente` dL/dhS = (p_S - p_T) WS del
    pezzo (non scalato). `bersagli` (P,) int64, -1 = nessun bersaglio: anche le CE (sola valutazione).
    Buffer vivi (P x V fp32): log p_T, log p_S, p_T -> 3 (in-place dove si puo').
    27/9 `gamma` (CAKLD di BitDistiller, arXiv 2402.10631 Eq. 5): None = KL in avanti, STESSO codice di prima.
    Con gamma anche "kl_inv" = KL(p_S || p_T) e "perdita" = gamma KL(S||T) + (1 - gamma) KL(T||S) per token
    (fp64; "kl" = "kl_fwd" AL BIT come senza gamma) e dL/dz_S = gamma p_S (log p_S - log p_T - KL(S||T)) +
    (1 - gamma)(p_S - p_T), poi @ WS. Un solo buffer P x V in piu' (4).
    27/9 `top1` (sola valutazione): anche "top1" (P,) bool = argmax dello studente == argmax del maestro; non tocca
    le altre uscite."""
    dt = hS.dtype
    lT = _logit_maestro(hT, WT, dt)
    lT.sub_(torch.logsumexp(lT, -1, keepdim=True))
    lS = _logit_studente(hS, WS)
    lS.sub_(torch.logsumexp(lS, -1, keepdim=True))
    out = {}
    if top1:
        out["top1"] = lS.argmax(-1) == lT.argmax(-1)
    if bersagli is not None:
        ok = bersagli >= 0
        b = bersagli.clamp(min=0).unsqueeze(-1)
        out["ce_s"] = -(lS.gather(-1, b).squeeze(-1).double() * ok).sum()
        out["ce_t"] = -(lT.gather(-1, b).squeeze(-1).double() * ok).sum()
        out["n_ce"] = ok.sum()
    pT = lT.exp()
    if gamma is not None:
        return _cakld_pezzo(lT, lS, pT, WS, gradiente, float(gamma), out)
    lT.sub_(lS).mul_(pT)                                    # p_T (log p_T - log p_S)
    out["kl"] = lT.sum(-1, dtype=torch.float64)             # (P,)
    del lT
    if gradiente:
        pS = lS.exp_()
        pS.sub_(pT)
        out["dh"] = _per_testa(pS, WS)
    return out


def _cakld_pezzo(lT: torch.Tensor, lS: torch.Tensor, pT: torch.Tensor, WS: torch.Tensor, gradiente: bool,
                 gamma: float, out: dict) -> dict:
    """Coda CAKLD di `kl_pezzo` (consuma lT, lS, pT). Buffer P x V: lT -> N = log p_T - log p_S, lS -> p_S, pT,
    piu' UNO temporaneo per i prodotti delle somme fp64 (liberato prima del gradiente)."""
    N = lT.sub_(lS)                                         # log p_T - log p_S (= -(log p_S - log p_T) esatto)
    tmp = torch.mul(N, pT)                                  # il buffer in piu'
    out["kl"] = out["kl_fwd"] = tmp.sum(-1, dtype=torch.float64)          # KL(p_T || p_S), AL BIT come gamma=None
    pS = lS.exp_()
    torch.mul(N, pS, out=tmp)
    kl_inv = tmp.sum(-1, dtype=torch.float64).neg_()        # KL(p_S || p_T) = sum p_S (log p_S - log p_T)
    del tmp
    out["kl_inv"] = kl_inv
    out["perdita"] = gamma * kl_inv + (1.0 - gamma) * out["kl"]
    if gradiente:
        # gamma p_S (log p_S - log p_T - KL(S||T)) + (1 - gamma)(p_S - p_T), in N e poi in p_S (in-place)
        N.add_(kl_inv.to(N.dtype).unsqueeze(-1)).neg_().mul_(pS).mul_(gamma)
        pS.sub_(pT).mul_(1.0 - gamma).add_(N)
        del N
        out["dh"] = _per_testa(pS, WS)
    return out


def peso_di(ca: "ConfigAllena") -> Optional[tuple]:
    """27/9 --peso-token: la configurazione dei pesi per `KLFusa`. None con "uniforme" (il percorso di prima, al bit);
    altrimenti (modo, alfa, massimo, frazione)."""
    if ca.peso_token == "uniforme":
        return None
    if ca.peso_token == "risposta":                     # 27/9 trainer-mc: i pesi del micro-lotto li fa `pesi_risposta`
        return ("risposta", float(ca.beta_risposta))
    if ca.peso_token == "generati":                     # 12/10 OPD: i pesi del micro-lotto li fa `e2e_onpolicy.pesi_generati`
        return ("generati",)
    return (ca.peso_token, float(ca.peso_alfa), float(ca.peso_max), float(ca.peso_frazione))


def pesato_di(ca: "ConfigAllena") -> bool:
    """27/9: la perdita del micro-lotto passa dal percorso pesato di `KLFusa` (--peso-token != uniforme, o
    --gamma-lettere > 0): somma pesata e statistiche nella voce "pesi" del registro."""
    return peso_di(ca) is not None or ca.gamma_lettere > 0


@torch.no_grad()
def pesi_token(l: torch.Tensor, peso: tuple) -> tuple:
    """27/9 --peso-token. Pesi per token w (fp64, forma di l, SENZA gradiente) dalla perdita per token l (fp64) del
    micro-lotto, N token. In entrambi i modi sum_t w_t = N (media 1, a meno dell'arrotondamento fp64).
    "kl": r_t = (max(l_t, 0) / media(l))^alfa, w_t = min(c r_t, M), c > 0 l'unica soluzione di (1/N) sum_t min(c r_t, M)
      = 1 (riempimento a tetto: sotto il tetto w resta proporzionale a r, i tagliati valgono M; senza tagli c = 1/media r,
      con alfa = 1 w = l / media l). Forma chiusa: s = r in ordine decrescente, con i k piu' grandi al tetto
      c_k = (N - k M) / sum_{j>=k} s_j; si prende il k piu' piccolo con c_k s_k <= M. Se nessun c esiste (meno di N/M
      token con l > 0) o la media di l non e' positiva e finita: w = 1 (uniforme) e ripiego = 1.
    "topk" (Rho-1, arXiv 2404.07965 Eq. 4): i k = ceil(frazione N) token a l piu' alta (ordinamento STABILE: a pari
      perdita l'indice minore, deterministico) hanno w = N/k, gli altri 0: la somma divisa per N k% di Rho-1.
    "risposta" (27/9 trainer-mc, peso = ("risposta", w, marcati)): i pesi sono GIA' decisi da `pesi_risposta` (1 ovunque,
      1 + lambda alle posizioni-risposta `marcati`, bool): qui solo le statistiche; la media NON e' 1 (KL piena resta a
      peso 1), "tagliati" = posizioni-risposta, in piu' "pesata_marcati" = sum_{marcati} w l.
    Ritorna (w, info), info = tensori 0-d sul dispositivo (nessuna sincronizzazione): n, somma_w, somma_w2, max_w,
    tagliati (kl: token al tetto; topk: token tenuti), ripiego."""
    modo = peso[0]
    if modo != "risposta":
        _, alfa, M, frazione = peso
    N = l.numel()
    dev = l.device
    if modo == "topk":
        k = min(N, max(1, math.ceil(frazione * N)))
        w = torch.zeros_like(l)
        w[torch.sort(l, descending=True, stable=True).indices[:k]] = N / k
        tagliati = torch.tensor(k, dtype=torch.int64, device=dev)
        ripiego = torch.zeros((), dtype=torch.int64, device=dev)
    elif modo == "risposta":                                          # 27/9 trainer-mc: pesi GIA' decisi (`pesi_risposta`)
        w, marcati = peso[1], peso[2]
        if w.shape != l.shape or marcati.shape != l.shape:
            raise ValueError(f"pesi risposta di forma {tuple(w.shape)}, perdita {tuple(l.shape)}")
        tagliati = marcati.sum()
        ripiego = torch.zeros((), dtype=torch.int64, device=dev)
        info_extra = {"pesata_marcati": (w * l * marcati).sum()}
    elif modo == "kl":
        m = l.mean()
        r = (l.clamp_min(0.0) / m).pow(alfa)
        s = torch.sort(r, descending=True).values
        coda = s.flip(0).cumsum(0).flip(0)                          # coda[k] = sum_{j>=k} s_j
        num = N - torch.arange(N, dtype=l.dtype, device=dev) * M
        c = num / coda
        ok = (coda > 0) & (num > 0) & (c * s <= M)
        esiste = ok.any() & (m > 0) & torch.isfinite(m)
        cs = torch.where(esiste, c[torch.argmax(ok.to(torch.int8))], torch.ones((), dtype=l.dtype, device=dev))
        cr = cs * r
        w = torch.where(esiste, torch.clamp(cr, max=M), torch.ones_like(l))
        tagliati = ((cr >= M) & esiste).sum()
        ripiego = (~esiste).to(torch.int64)
    else:
        raise ValueError(f"peso_token {modo!r}")
    info = {"n": torch.tensor(N, dtype=torch.int64, device=dev), "somma_w": w.sum(), "somma_w2": (w * w).sum(),
            "max_w": w.max(), "tagliati": tagliati, "ripiego": ripiego,
            # 27/9 integrata (registro per componente): minimo e mediana dei pesi del micro-lotto (solo registro)
            "min_w": w.min(), "mediana_w": w.median()}
    if modo == "risposta":
        info.update(info_extra)
    return w, info


def lambda_risposta(beta: float, n_token: int, n_risposte: int) -> float:
    """27/9 --peso-token risposta. Peso 1 a ogni token, 1 + lambda alle n_risposte posizioni-risposta del passo (n_token
    token): lambda >= 0 tale che sum_{t in A} w_t / sum_t w_t = beta, cioe' (1 + lambda) a = beta (N + lambda a) ->
    lambda = (beta N - a) / ((1 - beta) a); 0 se le risposte portano gia' >= beta della massa dei pesi (o se a = 0).
    beta e' la quota della massa dei PESI (= della massa della perdita se ogni token avesse la stessa perdita); la quota
    effettiva della perdita (sum_A w l / sum w l) si registra a parte ("quota_massa_risposte")."""
    if n_risposte <= 0:
        return 0.0
    return max(0.0, (beta * n_token - n_risposte) / ((1.0 - beta) * n_risposte))


def _risposte_di(src, lato: str, i: int) -> Optional[tuple]:
    f = getattr(src, "risposte", None)
    return None if f is None else f(lato, i)


def _lambda_del_passo(src, lato: str, righe: list, ca: "ConfigAllena") -> tuple:
    """(lambda, posizioni-risposta del passo, token per riga): lambda di `lambda_risposta` sulle righe INTERE del passo
    (0 senza --peso-token risposta: allora le risposte pesano 1 e servono solo al termine delle lettere)."""
    T0 = int(np.asarray(src.ids(lato, righe[0])).shape[0])
    a = sum(0 if r is None else int(r[0].numel()) for r in (_risposte_di(src, lato, i) for i in righe))
    lam = lambda_risposta(ca.beta_risposta, len(righe) * T0, a) if ca.peso_token == "risposta" else 0.0
    return lam, a, T0


def pesi_risposta(src, lato: str, m: list, Tn: int, lam: float, ca: "ConfigAllena", device) -> tuple:
    """27/9 trainer-mc. Per il micro-lotto m (righe, Tn token ciascuna, token appiattiti b * Tn + t come hS): (peso,
    lettere) per `KLFusa`. peso: con --peso-token risposta ("risposta", w, marcati) con w = 1 ovunque e 1 + lambda
    alle posizioni-risposta t (i logit in t predicono la lettera in t + 1); altrimenti `peso_di(ca)` (None o kl/topk).
    lettere: con --gamma-lettere G > 0 (pos, id_lettere, coef = G * w_pos; pos anche vuoto: il percorso pesato resta
    lo stesso per ogni micro-lotto), altrimenti None."""
    pos = [b * Tn + r[0] for b, r in enumerate(_risposte_di(src, lato, i) for i in m) if r is not None]
    pos = torch.cat(pos).to(device) if pos else torch.zeros(0, dtype=torch.int64, device=device)
    if ca.peso_token == "risposta":
        marcati = torch.zeros(len(m) * Tn, dtype=torch.bool, device=device)
        marcati[pos] = True
        w = torch.ones(len(m) * Tn, dtype=torch.float64, device=device)
        w[pos] = 1.0 + lam
        peso = ("risposta", w, marcati)
        w_pos = 1.0 + lam
    else:
        peso, w_pos = peso_di(ca), 1.0
    lettere = None
    if ca.gamma_lettere > 0:
        coef = torch.full((pos.numel(),), ca.gamma_lettere * w_pos, dtype=torch.float64, device=device)
        lettere = (pos, src.id_lettere.to(device), coef)
    return peso, lettere


@torch.no_grad()
def righe_testa(WS, idx: torch.Tensor) -> torch.Tensor:
    """27/9 integrata (trainer-mc + bf16x3fix): le righe `idx` della testa dello studente come tensore fp32. Con
    `--testa-kl bf16x3` WS e' una `e2e_bf16x3.TestaX3` (hi + lo bf16, niente indicizzazione): W[idx] = hi[idx] + lo[idx] in
    fp32 (la scomposizione di `scomponi`: errore relativo ~2^-16 contro la W fp32, come i logit bf16x3)."""
    if isinstance(WS, torch.Tensor):
        return WS[idx]
    return WS.hi[idx].to(torch.float32) + WS.lo[idx].to(torch.float32)


def kl_lettere(hS: torch.Tensor, WS: torch.Tensor, hT: torch.Tensor, WT: torch.Tensor, pos: torch.Tensor,
               id4: torch.Tensor) -> dict:
    """27/9 trainer-mc. Alle posizioni `pos` (righe di hS/hT): logit delle 4 lettere (id4) dello studente (hS @ WS[id4]^T,
    dtype di hS) e del maestro (`_logit_maestro` sulle stesse 4 righe della testa), log-softmax RINORMALIZZATA sulle 4,
    KL4 = KL(p4_T || p4_S) per posizione (fp64). Ritorna zS, zT, lS, lT (log p4), pS, pT, kl4."""
    W4S = righe_testa(WS, id4)                            # integrata: anche con la testa bf16x3 (TestaX3)
    zS = hS[pos] @ W4S.T
    zT = _logit_maestro(hT[pos], WT[id4], hS.dtype)
    lS = torch.log_softmax(zS, -1)
    lT = torch.log_softmax(zT, -1)
    pS, pT = lS.exp(), lT.exp()
    return {"zS": zS, "zT": zT, "lS": lS, "lT": lT, "pS": pS, "pT": pT, "W4S": W4S,
            "kl4": (pT * (lT - lS)).sum(-1, dtype=torch.float64)}


# ═══════════════════════════════════════ pesi per dominio (27/9, ramo dominio-strati) ═══════════════════════════════════════
# Righe mc del ramo trainer-mc: nel lato "train" agli indici OFFSET_MC + j, OFFSET_MC = 2^30 (stesso valore): dominio `mc`
OFFSET_MC_PESO = OFFSET_MC   # unificato al merge in integrata: e' l'OFFSET_MC di trainer-mc (2^30)
DOMINIO_MC = "mc"


def peso_dominio_da_testo(testo: str) -> tuple:
    """"strumenti=0.5,it_chat=0.5,edu=1.5" -> ((dominio, peso), ...) in ordine di nome (la forma canonica della config;
    i controlli sui valori sono in ConfigAllena). Vuoto = ()."""
    out: dict = {}
    for voce in (x.strip() for x in (testo or "").split(",") if x.strip()):
        nome, uguale, v = voce.partition("=")
        nome = nome.strip()
        if not nome or not uguale or not v.strip():
            raise ValueError(f"--peso-dominio: voce {voce!r}, attesa dominio=peso")
        if nome in out:
            raise ValueError(f"--peso-dominio: dominio {nome!r} due volte")
        out[nome] = float(v)
    return tuple(sorted(out.items()))


def strati_da_testo(testo: str) -> tuple:
    """"20-39" / "0-3,30-39" / "20,21" -> indici crescenti senza doppi. Vuoto = () (tutti gli strati)."""
    s: set = set()
    for voce in (x.strip() for x in (testo or "").split(",") if x.strip()):
        a, trattino, b = voce.partition("-")
        try:
            lo, hi = (int(a), int(b)) if trattino else (int(a), int(a))
        except ValueError:
            raise ValueError(f"--strati-allenati: voce {voce!r}, attesi N o N-M") from None
        if lo < 0 or hi < lo:
            raise ValueError(f"--strati-allenati: intervallo {voce!r} vuoto o negativo")
        s.update(range(lo, hi + 1))
    return tuple(sorted(s))


def _testo_strati(strati) -> str:
    """(20, 21, ..., 39) -> "20-39" (l'inverso di `strati_da_testo`)."""
    s = sorted(strati)
    pezzi, i = [], 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[j] + 1:
            j += 1
        pezzi.append(str(s[i]) if i == j else f"{s[i]}-{s[j]}")
        i = j + 1
    return ",".join(pezzi)


def dominio_per_peso(src, lato: str, i: int) -> str:
    """Il dominio che --peso-dominio usa per la riga: quello del teacher, `mc` per le righe mc (indice >= 2^30)."""
    return DOMINIO_MC if i >= OFFSET_MC_PESO else src.dominio(lato, i)


def pesi_dominio_passo(src, lato: str, righe: list, ca: "ConfigAllena") -> Optional[dict]:
    """--peso-dominio: {riga: peso} per le righe del PASSO, w_r = p_dominio(r) * c con c = n / sum_r p_dominio(r): media 1
    sulle righe (le righe di un passo hanno lo stesso numero di token: media 1 anche per token), cosi' la scala della
    perdita (e la K di G1, l'eps di Adam) resta quella della perdita uniforme. Domini non elencati: 1. None se spento o se
    i pesi del passo sono tutti uguali (normalizzati = 1: il calcolo di prima, al bit)."""
    if not ca.peso_dominio or not righe:
        return None
    pd = dict(ca.peso_dominio)
    grezzi = [float(pd.get(dominio_per_peso(src, lato, i), 1.0)) for i in righe]
    if all(w == grezzi[0] for w in grezzi):
        return None
    c = len(grezzi) / math.fsum(grezzi)
    return {i: w * c for i, w in zip(righe, grezzi)}


def riassunto_peso_dominio(src, lato: str, righe: list, ca: "ConfigAllena") -> Optional[dict]:
    """Voce "peso_dominio" del registro del passo: costante di normalizzazione c, per dominio (righe, peso effettivo)."""
    if not ca.peso_dominio:
        return None
    pd = dict(ca.peso_dominio)
    doms = [dominio_per_peso(src, lato, i) for i in righe]
    grezzi = [float(pd.get(d, 1.0)) for d in doms]
    c = len(grezzi) / math.fsum(grezzi)
    per: dict = {}
    for d, w in zip(doms, grezzi):
        x = per.setdefault(d, {"righe": 0, "peso": w * c})
        x["righe"] += 1
    return {"c": c, "per_dominio": dict(sorted(per.items())),
            "quota_perdita": {d: x["righe"] * x["peso"] / len(righe) for d, x in sorted(per.items())}}


def aggancia_peso_dominio(hS: torch.Tensor, lot, pesi: Optional[dict]) -> None:
    """--peso-dominio sul micro-lotto: la perdita diventa sum_riga w_riga sum_t l_t. Siccome l_t dipende solo da h_t,
    dL/dh_t = w_riga dl_t/dh_t ESATTO: basta un gancio sul gradiente di hS (B*T, H; righe contigue, `Studente.esce`),
    che vale per theta, F32 (anche output_norm), LoRA e per le statistiche V (proposta/verifica). Si moltiplica ai pesi
    per token (--peso-token, gia' dentro dh). pesi None = nessun gancio (il calcolo di prima, al bit)."""
    if pesi is None or not hS.requires_grad:
        return
    T = hS.shape[0] // len(lot.righe)
    wv = torch.tensor([pesi[i] for i in lot.righe], dtype=hS.dtype, device=hS.device).repeat_interleave(T).unsqueeze(-1)
    hS.register_hook(lambda g, wv=wv: g * wv.to(g.dtype))


def _allenata(st, L: int) -> bool:
    """27/9 dominio-strati: lo strato L e' allenato (studenti senza `allenato`, p.es. quelli finti delle prove: si')."""
    f = getattr(st, "allenato", None)
    return True if f is None else f(L)


def _congelati(st) -> bool:
    return getattr(st, "strati_allenati", None) is not None


class KLFusa(torch.autograd.Function):
    """sum_t KL(p_T || p_S) (fp64, scalare) con hS = uscita dello studente dopo output_norm. Il gradiente si
    calcola NEL passaggio in avanti (la perdita e' l'ultimo nodo e l'indietro la segue subito): l'indietro
    moltiplica soltanto per g (la scala della perdita K / token, potenza di 2: esatto).
    27/9 con `gamma` (CAKLD): ritorna (sum perdita CAKLD, sum KL(T||S), sum KL(S||T)); le ultime due senza
    gradiente (solo registro). gamma=None: stesso percorso e stessa uscita (un tensore) di prima.
    27/9 con `peso` (--peso-token, `peso_di`): perdita PESATA sum_t w_t l_t, w = `pesi_token`(l) FISSI (senza gradiente),
    dL/dhS_t = w_t (dL_t/dhS_t); ritorna (sum pesata, sum uniforme[, sum KL(T||S), sum KL(S||T) con gamma]), dalla
    seconda in poi senza gradiente; la somma uniforme e' calcolata come nel percorso senza pesi (al bit). `diag` (dict,
    facoltativo) riceve le statistiche dei pesi (`pesi_token`). peso=None: il percorso di prima.
    27/9 trainer-mc `lettere` = (pos int64 [a], id4 int64 [4], coef fp64 [a]): in piu' sum_a coef_a KL(p4_T || p4_S)
    (`kl_lettere`) alle posizioni pos, nel valore pesato e nel gradiente (dL/dz4_S = coef (p4_S - p4_T), sulle 4 righe
    della testa); con peso=None i pesi sono tutti 1. `diag` riceve anche "kl4" (sum KL4), "kl4_pesata" (sum coef KL4),
    "n_lettere". Stessa forma delle uscite del percorso pesato."""

    @staticmethod
    def forward(ctx, hS, WS, hT, WT, pezzo: int, gamma: Optional[float] = None, peso: Optional[tuple] = None,
                diag: Optional[dict] = None, lettere: Optional[tuple] = None):
        tot = torch.zeros((), dtype=torch.float64, device=hS.device)
        dh = torch.empty_like(hS)
        if peso is not None or lettere is not None:
            return KLFusa._pesata(ctx, hS, WS, hT, WT, pezzo, gamma, peso, diag, tot, dh, lettere)
        if gamma is None:
            for i in range(0, hS.shape[0], pezzo):
                r = kl_pezzo(hS[i:i + pezzo], WS, hT[i:i + pezzo], WT, True)
                tot += r["kl"].sum()
                dh[i:i + pezzo] = r["dh"]
            ctx.save_for_backward(dh)
            return tot
        tf = torch.zeros((), dtype=torch.float64, device=hS.device)
        ti = torch.zeros((), dtype=torch.float64, device=hS.device)
        for i in range(0, hS.shape[0], pezzo):
            r = kl_pezzo(hS[i:i + pezzo], WS, hT[i:i + pezzo], WT, True, gamma=gamma)
            tot += r["perdita"].sum()
            tf += r["kl"].sum()
            ti += r["kl_inv"].sum()
            dh[i:i + pezzo] = r["dh"]
        ctx.save_for_backward(dh)
        ctx.mark_non_differentiable(tf, ti)
        return tot, tf, ti

    @staticmethod
    def _pesata(ctx, hS, WS, hT, WT, pezzo: int, gamma: Optional[float], peso: Optional[tuple], diag: Optional[dict],
                tot, dh, lettere: Optional[tuple] = None):
        """Percorso di `forward` con i pesi per token: l_t e dh_t per pezzo (stesso `kl_pezzo`), poi i pesi sull'INTERO
        micro-lotto e dh_t <- w_t dh_t (w fisso: gradiente esatto della somma pesata con w costante). 27/9 trainer-mc:
        peso None (solo `lettere`) = pesi 1; poi il termine delle lettere (`lettere`, vedi la classe)."""
        chiave = "kl" if gamma is None else "perdita"
        l = torch.empty(hS.shape[0], dtype=torch.float64, device=hS.device)
        tf = ti = None
        if gamma is not None:
            tf = torch.zeros((), dtype=torch.float64, device=hS.device)
            ti = torch.zeros((), dtype=torch.float64, device=hS.device)
        for i in range(0, hS.shape[0], pezzo):
            r = kl_pezzo(hS[i:i + pezzo], WS, hT[i:i + pezzo], WT, True, gamma=gamma)
            tot += r[chiave].sum()                              # somma uniforme, come il percorso senza pesi
            l[i:i + pezzo] = r[chiave]
            if gamma is not None:
                tf += r["kl"].sum()
                ti += r["kl_inv"].sum()
            dh[i:i + pezzo] = r["dh"]
        if diag is not None and diag.get("vuole_l"):         # integrata --attribuzione: perdita per token (uniforme)
            diag["l"] = l
        if peso is not None:
            w, info = pesi_token(l, peso)
            dh.mul_(w.to(dh.dtype).unsqueeze(-1))
            tot_w = (w * l).sum()
        else:                                                   # 27/9 solo --gamma-lettere: pesi tutti 1
            n = torch.tensor(l.numel(), dtype=torch.int64, device=l.device)
            info = {"n": n, "somma_w": n.to(torch.float64), "somma_w2": n.to(torch.float64),
                    "max_w": torch.ones((), dtype=torch.float64, device=l.device), "tagliati": torch.zeros_like(n),
                    "ripiego": torch.zeros_like(n)}
            tot_w = l.sum()
        if lettere is not None:
            pos, id4, coef = lettere
            k = kl_lettere(hS, WS, hT, WT, pos, id4)
            dz = (k["pS"] - k["pT"]).mul_(coef.to(k["pS"].dtype).unsqueeze(-1))
            dh.index_add_(0, pos, (dz @ k["W4S"]).to(dh.dtype))
            k4w = (coef * k["kl4"]).sum()
            tot_w = tot_w + k4w
            info.update({"kl4": k["kl4"].sum(), "kl4_pesata": k4w,
                         "n_lettere": torch.tensor(pos.numel(), dtype=torch.int64, device=l.device)})
        opd = diag.pop("opd", None) if diag is not None else None
        if opd is not None:                                     # 12/10 OPD (e2e_onpolicy.termine_opd): gradiente fuso
            import e2e_onpolicy as _OPD
            tot_w = tot_w + _OPD.termine_opd(hS, WS, hT, WT, opd, pezzo, dh, _logit_studente, _logit_maestro,
                                             _per_testa, info)
            info.pop("_pesi_opd", None)
            info.pop("_lambda_opd", None)
        if diag is not None:
            diag.update(info)
        ctx.save_for_backward(dh)
        if gamma is None:
            ctx.mark_non_differentiable(tot)
            return tot_w, tot
        ctx.mark_non_differentiable(tot, tf, ti)
        return tot_w, tot, tf, ti

    @staticmethod
    def backward(ctx, g, *_):
        (dh,) = ctx.saved_tensors
        return dh * g.to(dh.dtype), None, None, None, None, None, None, None, None


def kl_diretta(hS, WS, hT, WT) -> torch.Tensor:
    """Riferimento non a pezzi, con autograd (solo prove)."""
    lS = torch.log_softmax(hS @ WS.T, -1)
    lT = torch.log_softmax(_logit_maestro(hT, WT, hS.dtype), -1)
    return (lT.exp() * (lT - lS)).sum()


def cakld_diretta(hS, WS, hT, WT, gamma: float) -> torch.Tensor:
    """Riferimento CAKLD non a pezzi, con autograd (solo prove): sum_t gamma KL(S||T) + (1 - gamma) KL(T||S)."""
    lS = torch.log_softmax(hS @ WS.T, -1)
    lT = torch.log_softmax(_logit_maestro(hT, WT, hS.dtype), -1)
    return gamma * (lS.exp() * (lS - lT)).sum() + (1.0 - gamma) * (lT.exp() * (lT - lS)).sum()


def gamma_di(ca: "ConfigAllena") -> Optional[float]:
    """Il gamma da passare a `kl_pezzo`/`KLFusa`: None con la KL in avanti; con CAKLD il valore gia' risolto."""
    if ca.perdita == "kl":
        return None
    if ca.gamma is None:
        raise RuntimeError("perdita cakld con gamma non ancora misurato (lo risolve `risolvi_gamma`)")
    return float(ca.gamma)


@torch.no_grad()
def misura_gamma(car: "Caricatore", lato: str, righe: list, ca: "ConfigAllena", WT) -> dict:
    """gamma di CAKLD (BitDistiller §3.2): media sul corpus di p_T(y vero), y = token successivo (`_bersagli`),
    sulle `righe` (i primi micro-lotti di addestramento). Somma fp64; bersagli fuori dalla testa esclusi e contati."""
    tot = torch.zeros((), dtype=torch.float64, device=WT.device)
    n = torch.zeros((), dtype=torch.int64, device=WT.device)
    fuori = torch.zeros((), dtype=torch.int64, device=WT.device)
    for j in range(0, len(righe), ca.righe_per_micro):
        lot = car.prendi(lato, righe[j:j + ca.righe_per_micro])
        bers = _bersagli(lot.ids, lot.inizi)
        f = bers >= WT.shape[0]
        fuori += f.sum()
        bers = torch.where(f, torch.full_like(bers, -1), bers)
        for i in range(0, bers.shape[0], ca.pezzo_kl):
            b = bers[i:i + ca.pezzo_kl]
            ok = b >= 0
            lT = _logit_maestro(lot.hT[i:i + ca.pezzo_kl], WT, torch.float32)
            lT.sub_(torch.logsumexp(lT, -1, keepdim=True))
            tot += (lT.gather(-1, b.clamp(min=0).unsqueeze(-1)).squeeze(-1).double().exp() * ok).sum()
            n += ok.sum()
            del lT
    return {"gamma": float(tot) / max(int(n), 1), "token": int(n), "bersagli_fuori_testa": int(fuori),
            "righe": [righe[0], righe[-1]] if righe else []}


def risolvi_gamma(stato: dict, ca: "ConfigAllena", src, car: "Caricatore", righe0: list, WT,
                  log: Callable) -> "ConfigAllena":
    """Con perdita cakld: gamma deciso (ca.gamma) oppure misurato UNA volta sulle righe del passo 0 e salvato in
    stato["gamma"]; alla ripresa il gamma misurato si rilegge dallo stato (stesso gamma). Ritorna la config con
    ca.gamma risolto (quella dei calcoli; stato["config"] resta quella chiesta). Con perdita kl: ca invariata."""
    if ca.perdita == "kl":
        return ca
    salvato = stato.get("gamma")
    if ca.gamma is not None:
        if salvato is not None and salvato["valore"] != ca.gamma:
            log(f"gamma CAKLD: {salvato['valore']:.6f} -> {ca.gamma:.6f} (deciso, da qui in avanti)")
        stato["gamma"] = {"valore": float(ca.gamma), "misurato": False}
    elif salvato is not None and salvato.get("misurato"):
        log(f"gamma CAKLD dallo stato della corsa: {salvato['valore']:.6f} (misurato su {salvato['token']} token)")
    else:
        consumati = set(stato.get("consumati") or [])
        if any(src.blocco_di("train", i) in consumati for i in righe0):
            raise RuntimeError("gamma CAKLD da misurare ma le righe del passo 0 sono gia' consumate (rotazione): "
                               "dare --gamma esplicito")
        stato["attesa_teacher_s"] = stato.get("attesa_teacher_s", 0.0) + attendi_righe(src, "train", righe0, log,
                                                                                       ca.attesa_ogni_s)
        m = misura_gamma(car, "train", righe0, ca, WT)
        stato["gamma"] = {"valore": m["gamma"], "misurato": True, "token": m["token"], "righe": m["righe"],
                          "bersagli_fuori_testa": m["bersagli_fuori_testa"]}
        log(f"gamma CAKLD misurato: {m['gamma']:.6f} = media di p_T(y vero) su {m['token']} token "
            f"(righe {m['righe']})")
    return dataclasses.replace(ca, gamma=stato["gamma"]["valore"])


# ═══════════════════════════════════════ teste ═══════════════════════════════════════
def testa_maestro(device, dtype=torch.float32) -> tuple:
    """output.weight del bf16 ORIGINALE (§2.3) -> dispositivo, poi fp32 (esatto): la conversione avviene SUL
    dispositivo (in RAM di sistema restano solo i byte bf16 del memmap). Stessi valori di
    `e2e_teacher.TestaMaestro("bf16").W` (prova A2). Ritorna (W, sha256 dei byte)."""
    import e2e_teacher as T
    t = T._tensore_gguf(T.TESTA_BF16, "output.weight")
    if t.tensor_type.name != "BF16":
        raise ValueError(f"testa del teacher {t.tensor_type.name}, attesa BF16")
    V, H = int(t.shape[1]), int(t.shape[0])                 # GGUF [H, V] -> righe V
    off, nbyte = int(t.data_offset), int(t.n_bytes)
    if nbyte != V * H * 2:
        raise ValueError(f"testa del teacher: {nbyte} byte per {V} x {H} bf16")
    mm = np.memmap(T.TESTA_BF16, dtype=np.int16, mode="r", offset=off, shape=(V, H))
    h = hashlib.sha256()
    W = torch.empty(V, H, dtype=dtype, device=device)
    passo = 16384                                            # 64 MiB alla volta: in RAM di sistema solo questo
    for a in range(0, V, passo):
        pezzo = np.array(mm[a:a + passo])
        h.update(pezzo.tobytes())
        W[a:a + passo] = torch.from_numpy(pezzo).to(device).view(torch.bfloat16).to(dtype)
    del mm
    return W, h.hexdigest()


# ═══════════════════════════════════════ G1 ═══════════════════════════════════════
def aggrega_dy(righe: list) -> dict:
    """Righe dell'`Osservatore` dello studente (`Studente.diagnostica_indietro`: una per strato, famiglia e
    chiamata dell'indietro, sul dL/dY fp32 GIA' scalato per K che entra nei prodotti fp16) -> per famiglia:
    non nulli, azzerati in fp16, subnormali in fp16, max |dY|. dL/dW = dY^T X si accumula in fp32 dentro il
    kernel: l'underflow dell'indietro entra solo da qui."""
    out: dict = {}
    for r in righe:
        a = out.setdefault(r["famiglia"], {"non_nulli_fp32": 0, "azzerati": 0, "subnormali": 0, "max_abs_dy": 0.0})
        a["non_nulli_fp32"] += r["non_nulli"]
        a["azzerati"] += r["a_zero_in_fp16"]
        a["subnormali"] += r["subnormali_in_fp16"]
        a["max_abs_dy"] = max(a["max_abs_dy"], float(r["max_abs"]))
    for a in out.values():
        a["quota_azzerati"] = a["azzerati"] / max(a["non_nulli_fp32"], 1)
        a["quota_subnormali"] = a["subnormali"] / max(a["non_nulli_fp32"], 1)
    return out


def _kth(v: torch.Tensor, q: float) -> float:
    """Statistica d'ordine k = ceil(q n) (1-based) di un vettore 1-D (torch.quantile non accetta > 2^24 valori)."""
    k = min(v.numel(), max(1, math.ceil(q * v.numel())))
    return float(torch.kthvalue(v, k).values)


def confronta_dtheta(d0: list, g_prova: list, g_rif: list) -> dict:
    """dL/dtheta di prova contro il riferimento (G1: indietro fp16 contro fp32; cancello B3: B3_ind32 contro
    B_ind32), sulle scale vive (d0 != 0) col gradiente di riferimento non nullo: quante vanno a zero, quante hanno
    il segno opposto, errore relativo mediano e p99. (I nomi dei campi sono quelli di G1.)"""
    vive_nz = rotti = nulli = 0
    rel = []
    for d, a, b in zip(d0, g_prova, g_rif):
        a = a.to(b.device)
        v = (d.to(b.device) != 0) & (b != 0)
        vive_nz += int(v.sum())
        nulli += int((v & (a == 0)).sum())
        rotti += int((v & (a != 0) & (torch.sign(a) != torch.sign(b))).sum())
        rr = ((a - b).abs() / b.abs())[v]
        if rr.numel():
            rel.append(rr.float())
    rel_t = torch.cat(rel) if rel else torch.zeros(1)
    return {"vive_con_g32_non_nullo": vive_nz, "nulli_in_fp16": nulli, "segno_sbagliato": rotti,
            "quota_rotti": (nulli + rotti) / max(vive_nz, 1), "err_rel_mediano": _kth(rel_t, 0.5),
            "err_rel_p99": _kth(rel_t, 0.99)}


def statistiche_g(st, divisore: float) -> dict:
    """|dL/dtheta| / divisore per famiglia, sulle scale VIVE (d0 != 0) con gradiente non nullo (§2.6 G1)."""
    out = {}
    for n in ("gate", "up", "down"):
        gs, vv = [], []
        for L in st.strati:
            if not _allenata(st, L):                 # 27/9 dominio-strati: gli strati congelati non hanno gradiente
                continue
            e = st.esperti(L)
            gs.append(e.theta[n].grad.reshape(-1))
            vv.append(e.fam[n].d0.reshape(-1) != 0)
        g = torch.cat(gs).abs().double() / divisore
        vive = torch.cat(vv)
        gv = g[vive & (g > 0)]
        out[n] = {"vive": int(vive.sum()), "vive_con_gradiente_zero": int((vive & (g == 0)).sum()),
                  "mediana": _kth(gv, 0.5), "q01": _kth(gv, 0.01), "p99": _kth(gv, 0.99),
                  "quota_sotto_10x_eps_1e-8": float((gv < 1e-7).double().mean())}
        del g, gv
    return out


def regola_k(q01: float) -> dict:
    """§2.5 (rev.) + A2: eps_G1 = min(1e-8, 1e-3 q01), K = 2^round(log2(1e-8 / eps_G1)), eps di Adam = eps_G1 K."""
    eps_g1 = min(1e-8, 1e-3 * q01)
    k = 2.0 ** round(math.log2(1e-8 / eps_g1))
    return {"eps_g1": eps_g1, "K": k, "eps_adam": eps_g1 * k}


def scegli_k(K: float, dy: dict, trabocco: bool, ca: ConfigAllena) -> float:
    """26/9 sera. La scala K dell'indietro fp16 si sceglie dal gradiente d'uscita dY (sottoflusso sotto 2^-24,
    trabocco oltre 65504), NON dalla regola dell'eps: AdamW e' invariante alla scala del gradiente se l'eps scala
    con lei (eps_adam = eps_G1 * K da' lo stesso passo per QUALUNQUE K). Con K da `regola_k` (eps_adam ~ 1e-8)
    K scendeva a 2^13 e i dY azzerati arrivavano a 2,6e-2 (gate): cancello G1 fallito alle 17:15 del 26/9.
    Qui: con sottoflusso oltre soglia K sale fino al massimo che tiene max|dY| * K * margine <= 65504; se c'e'
    trabocco scende; se la soglia e' rispettata resta (scende solo se viola il margine). Stessa K = convergenza."""
    if trabocco:
        return K / 16.0
    max_dy = max(v["max_abs_dy"] for v in dy.values())
    quota = max(v["quota_azzerati"] for v in dy.values())
    if max_dy <= 0:
        return K
    j = math.floor(math.log2(65504.0 / (max_dy * ca.margine_trabocco)))
    if quota <= ca.max_quota_dy_azzerati:
        return K * 2.0 ** min(j, 0)
    return K * 2.0 ** j if j > 0 else K


def _parametri_f32(st) -> list:
    """Master dei tensori F32 allenati (--allena-f32); [] per lo studente di sempre (e per gli studenti finti)."""
    f = getattr(st, "parametri_f32", None)
    return f() if f is not None else []


def _parametri_lora(st) -> list:
    """[A, B] degli adattatori LoRA (--lora-nonesperti); [] per lo studente di sempre (e per gli studenti finti).
    28/9 --lora-ripristina ferma (`st.lora_ferma`): [] anche con gli adattatori installati (Delta W = 0 esatto): per
    l'ottimizzatore, i gradienti, l'attribuzione, le riprese e le istantanee la LoRA non c'e'."""
    if getattr(st, "lora_ferma", False):
        return []
    f = getattr(st, "parametri_lora", None)
    return f() if f is not None else []


def _blocca_lora(st) -> None:
    """28/9 --lora-ripristina ferma: A e B senza gradiente (dopo `congela_strati`, che li riaccende negli strati
    allenati, va richiamata)."""
    for a in getattr(st, "lora", {}).values():
        for p in a.parametri():
            p.requires_grad_(False)
            p.grad = None


def _nomi_param_lora(chiavi: list) -> list:
    """Nomi dei parametri del gruppo LoRA nell'ordine di Adam: A poi B per adattatore."""
    return [f"{n}#{k}" for n in chiavi for k in ("A", "B")]


def firma_lora_config(ca) -> tuple:
    return (ca.lora_rango, ca.lora_alpha, tuple(ca.lora_famiglie), tuple(ca.lora_tipi))


def firma_lora_studente(st) -> tuple:
    c = getattr(st, "lora_cfg", None)
    return (0, None, (), ()) if c is None else (c.rango, c.alpha, tuple(c.famiglie), tuple(c.tipi))


@contextmanager
def _fermi_extra(st):
    """Master F32 e adattatori LoRA senza gradiente per la durata del blocco (curvatura del passo V)."""
    from contextlib import ExitStack
    with ExitStack() as pila:
        if _parametri_f32(st):
            pila.enter_context(st.f32_fermi())
        if _parametri_lora(st):
            pila.enter_context(st.lora_fermi())
        yield


def _azzera(st) -> None:
    for p in st.parametri() + _parametri_f32(st) + _parametri_lora(st):
        p.grad = None


def _copia_grad(st) -> list:
    """Copia dei gradienti di theta; None per le scale degli strati congelati (27/9 dominio-strati: grad None)."""
    return [None if p.grad is None else p.grad.detach().clone() for p in st.parametri()]


def ricetta_indietro_fp32(ric):
    """Stessa ricetta (stesso avanti: B, C...), ma gli ingressi dei prodotti dell'INDIETRO in fp32: il
    riferimento di G1/A2 (per B e' `RICETTE["B_ind32"]`)."""
    return dataclasses.replace(ric, nome=ric.nome + "_ind32", dtype_indietro=torch.float32)


def imposta_ricetta(st, ric) -> None:
    """Cambio PERMANENTE della ricetta dello studente (G1 che passa da solo all'indietro fp32)."""
    st.ric = ric
    for L in st.strati:
        st.esperti(L).ric = ric


@contextmanager
def ricetta_temporanea(st, ric):
    """Per la sola durata del blocco, le famiglie di esperti usano `ric` (solo G1: indietro fp32 di riferimento).
    L'avanti degli esperti dipende da ruota/dtype_gemm, che restano quelli della ricetta vera."""
    vecchie = {L: st.esperti(L).ric for L in st.strati}
    try:
        for L in st.strati:
            st.esperti(L).ric = ric
        yield
    finally:
        for L, r in vecchie.items():
            st.esperti(L).ric = r


def stima_due_lotti(g2_piccoli: list, n_piccoli: list, g2_grande: float, n_grande: int) -> Optional[dict]:
    """McCandlish et al. 2018 (arXiv 1812.06162) app. A: E|G_B|^2 = |G|^2 + tr(Sigma)/B (B in token). Lotti piccoli =
    i micro-lotti (media dei |G_k|^2, B_piccolo = 1 / media(1/n_k)), lotto grande = il passo intero. Stime (non
    distorte, rumorose: possono uscire negative) di |G|^2, tr(Sigma) e B_simple = tr(Sigma) / |G|^2. None con 1 micro."""
    if len(g2_piccoli) < 2:
        return None
    inv_p = sum(1.0 / n for n in n_piccoli) / len(n_piccoli)
    b_p = 1.0 / inv_p
    if b_p >= n_grande:
        return None
    m = sum(g2_piccoli) / len(g2_piccoli)
    G2 = (n_grande * g2_grande - b_p * m) / (n_grande - b_p)
    S = (m - g2_grande) / (inv_p - 1.0 / n_grande)
    return {"G2": G2, "trSigma": S, "B_simple_token": S / G2 if G2 != 0 else float("inf"),
            "g2_micro_media": m, "B_piccolo_token": b_p}


class MisuraRumore:
    """27/9 velocita'. Rumore del gradiente a due lotti (McCandlish 2018, app. A) DENTRO `gradiente_passo`, senza
    cambiare nessun numero dell'allenamento: un gancio (`register_hook`) su ogni theta riceve il contributo g del
    micro-lotto PRIMA che autograd lo accumuli in .grad; lo scarto del .grad accumulato e' allora
    fp32(grad + g) - grad (la stessa somma che fa AccumulateGrad), calcolato in fp64 per quel solo tensore (niente
    copia di tutto theta per micro-lotto). Il gancio ritorna None: il gradiente non cambia. Solo scale vive (d0 != 0).
    Unita' vere (divise per K): |G_k|^2 = (N/n_k)^2 |scarto_k|^2 / K^2 (gradiente della perdita media per token del
    micro-lotto), |G_passo|^2 = |theta.grad|^2 / K^2. Precondizionata (`opt` con stato): P = 1/(sqrt(v^)+eps) di Adam
    in unita' vere (v^ = exp_avg_sq / (1 - beta2^passo) PRIMA del passo, cioe' dal passo precedente): |g|_P^2 =
    sum g^2 / (K (sqrt(v^) + eps_adam)). Tutto sul dispositivo; `risultato` sincronizza una volta.
    Presupposto: ogni theta riceve UN contributo per micro-lotto (la scala si calcola una volta per strato e avanti);
    se no "contributi_multipli" e' vero e le stime per micro-lotto sono None."""

    def __init__(self, st, opt=None, K: float = 1.0):
        self.K = float(K)
        # 27/9 dominio-strati: solo le scale degli strati ALLENATI (le congelate non hanno gradiente ne' ganci)
        self.coppie = [(p, st.esperti(L).fam[n].d0) for (L, n), p in zip(st.chiavi(), st.parametri())
                       if _allenata(st, L)]
        self.opt = opt
        self.pre = None
        if opt is not None and all("exp_avg_sq" in opt.state.get(p, {}) for p, _ in self.coppie):
            gr = opt.param_groups[0]
            passi = {float(opt.state[p]["step"]) for p, _ in self.coppie}   # una lettura per passo, non per gancio
            b2 = float(gr["betas"][1])
            if len(passi) == 1:
                self.pre = (math.sqrt(1.0 - b2 ** passi.pop()), float(gr["eps"]))
            else:
                # 27/9 dominio-strati: strati scongelati alla ripresa -> passi di Adam diversi fra i parametri (Adam li
                # corregge uno per uno): la stessa correzione per parametro
                self._bc = {id(p): math.sqrt(1.0 - b2 ** float(opt.state[p]["step"])) for p, _ in self.coppie}
                self.pre = (None, float(gr["eps"]))
        self.g2, self.g2p, self.n_tok, self.conta = [], [], [], []
        self._ganci: list = []
        self.tot: list = []
        self.n_passo = None
        self.n_righe = None

    def _denom(self, p) -> torch.Tensor:
        bc, eps = self.pre
        if bc is None:                                   # 27/9 dominio-strati: passi di Adam diversi (vedi __init__)
            bc = self._bc[id(p)]
        return self.opt.state[p]["exp_avg_sq"].double().sqrt().div_(bc).add_(eps)

    def _gancio(self, i: int, p, d0):
        def f(g):
            k = len(self.g2) - 1
            if k < 0:
                return None
            prev = p.grad
            d = g.to(torch.float64, copy=True) if prev is None else (prev + g).to(torch.float64, copy=True).sub_(prev.double())
            d.masked_fill_((d0 == 0).reshape(d.shape), 0.0)
            d.mul_(d)
            self.g2[k] += d.sum()
            if self.pre is not None:
                self.g2p[k] += d.div_(self._denom(p)).sum()
            self.conta[k][i] += 1
            return None
        return f

    def aggancia(self) -> None:
        self._ganci = [p.register_hook(self._gancio(i, p, d0)) for i, (p, d0) in enumerate(self.coppie)]

    def micro(self, n_token: int) -> None:
        dev = self.coppie[0][0].device
        self.g2.append(torch.zeros((), dtype=torch.float64, device=dev))
        self.g2p.append(torch.zeros((), dtype=torch.float64, device=dev))
        self.n_tok.append(int(n_token))
        self.conta.append([0] * len(self.coppie))

    def fine(self, n_passo: int, n_righe: int) -> None:
        for h in self._ganci:
            h.remove()
        self._ganci = []
        self.n_passo, self.n_righe = int(n_passo), int(n_righe)
        dev = self.coppie[0][0].device
        t = torch.zeros((), dtype=torch.float64, device=dev)
        tp = torch.zeros((), dtype=torch.float64, device=dev)
        for p, d0 in self.coppie:
            if p.grad is None:
                continue
            a = p.grad.to(torch.float64, copy=True).masked_fill_((d0 == 0).reshape(p.grad.shape), 0.0)
            a.mul_(a)
            t += a.sum()
            if self.pre is not None:
                tp += a.div_(self._denom(p)).sum()
        self.tot = [t, tp]

    def risultato(self) -> dict:
        n = len(self.g2)
        vals = torch.stack(self.g2 + self.g2p + self.tot).tolist()        # la sincronizzazione
        K, N = self.K, self.n_passo
        multipli = any(c != 1 for cc in self.conta for c in cc)
        g2k = [(N / nk) ** 2 * x / K ** 2 for x, nk in zip(vals[:n], self.n_tok)]
        out = {"micro": n, "token_micro": self.n_tok, "token_passo": N, "token_riga": N / self.n_righe,
               "g2_micro": None if multipli else g2k, "g2_passo": vals[2 * n] / K ** 2,
               "contributi_multipli": multipli, "stima": None, "precondizionata": None}
        if not multipli:
            out["stima"] = stima_due_lotti(g2k, self.n_tok, out["g2_passo"], N)
        if self.pre is not None and not multipli:
            g2kp = [(N / nk) ** 2 * x / K for x, nk in zip(vals[n:2 * n], self.n_tok)]
            out["precondizionata"] = {"g2_micro": g2kp, "g2_passo": vals[2 * n + 1] / K,
                                      "stima": stima_due_lotti(g2kp, self.n_tok, vals[2 * n + 1] / K, N)}
        for s in (out["stima"], (out["precondizionata"] or {}).get("stima")):
            if s is not None:
                s["B_simple_righe"] = s["B_simple_token"] / out["token_riga"]
        return out


def testo_rumore(r: Optional[dict]) -> str:
    """Riga di registro del rumore del gradiente."""
    if r is None or r["stima"] is None:
        return "rumore del gradiente: non stimabile (" + ("nessuna misura" if r is None else
                                                         f"{r['micro']} micro-lotti") + ")"
    s = r["stima"]
    t = (f"rumore del gradiente: |G|^2 {s['G2']:.4e}, tr(Sigma) {s['trSigma']:.4e}, B_simple {s['B_simple_token']:.4g} "
         f"token ({s['B_simple_righe']:.4g} righe); passo {r['token_passo']} token, |G_passo|^2 {r['g2_passo']:.4e}")
    p = (r.get("precondizionata") or {}).get("stima")
    if p is not None:
        t += f"; precondizionato sqrt(v): |G|^2 {p['G2']:.4e}, tr(Sigma) {p['trSigma']:.4e}, B_simple " \
             f"{p['B_simple_token']:.4g} token ({p['B_simple_righe']:.4g} righe)"
    return t


def gradiente_passo(st, car: Caricatore, lato: str, righe: list, ca: ConfigAllena, WS, WT, scala_k: float,
                    prossime: Optional[list] = None, pv=None, somme: Optional[dict] = None,
                    rumore: Optional[MisuraRumore] = None) -> torch.Tensor:
    """Accumula in theta.grad il gradiente di K * (sum_t KL) / token del passo, micro-lotto per micro-lotto:
    avanti senza grafo, KL fusa, indietro con checkpoint per strato e instradamento salvato (studente).
    Ritorna la somma KL (fp64, sul dispositivo: nessuna sincronizzazione).
    27/9 con perdita cakld: stessa cosa sulla perdita CAKLD (ca.gamma gia' risolto); in `somme` (se dato) si
    accumulano anche "kl" = sum KL(T||S) e "kl_inv" = sum KL(S||T) del passo (tensori fp64).
    27/9 `rumore` (MisuraRumore): norme^2 fp64 del gradiente di ogni micro-lotto e del passo (ganci che non toccano
    il gradiente); i risultati con `rumore.risultato()` dopo la sincronizzazione del passo.
    27/9 --peso-token: il gradiente e' quello della perdita PESATA (`_micro_lotti_pesati`), il valore ritornato resta la
    somma UNIFORME; la somma pesata e le statistiche dei pesi vanno in `somme` (se dato)."""
    gamma = gamma_di(ca)
    micro = [righe[j:j + ca.righe_per_micro] for j in range(0, len(righe), ca.righe_per_micro)]
    seguenti = micro[1:] + ([prossime[:ca.righe_per_micro]] if prossime else [])
    n_tok = None
    tot = None
    pesi_d = pesi_dominio_passo(car.src, lato, righe, ca)   # 27/9 --peso-dominio (spento o uniforme: None)
    if rumore is not None:
        rumore.aggancia()
    try:
        if pesato_di(ca):                            # 27/9 --peso-token / --gamma-lettere: somma pesata in `somme`
            tot, n_tok = _micro_lotti_pesati(st, car, lato, righe, micro, seguenti, ca, WS, WT, scala_k, gamma, pv,
                                             {} if somme is None else somme, rumore, pesi_d=pesi_d)
        else:
            tot, n_tok = _micro_lotti(st, car, lato, righe, micro, seguenti, ca, WS, WT, scala_k, gamma, pv, somme,
                                      rumore, pesi_d=pesi_d)
    finally:
        if rumore is not None:
            rumore.fine(n_tok or 0, len(righe))
    return tot


def _micro_lotti(st, car, lato, righe, micro, seguenti, ca, WS, WT, scala_k, gamma, pv, somme, rumore, pesi_d=None):
    """Il ciclo sui micro-lotti di `gradiente_passo` (stesso codice di prima, piu' `rumore.micro`; 27/9 `pesi_d` =
    --peso-dominio, `aggancia_peso_dominio`)."""
    n_tok = None
    tot = None
    for k, m in enumerate(micro):
        lot = car.prendi(lato, m)
        if k < len(seguenti) and seguenti[k]:
            car.prenota(lato, seguenti[k])
        B, Tn = lot.ids.shape
        if n_tok is None:
            n_tok = len(righe) * Tn
        stato = st.avanti(lot.ids, inizi=lot.inizi)
        xL = stato["xL"].detach().requires_grad_(True)
        with torch.enable_grad():
            hS = st.esce(xL)
            aggancia_peso_dominio(hS, lot, pesi_d)
            if gamma is None:
                kl = KLFusa.apply(hS, WS, lot.hT, WT, ca.pezzo_kl)
            else:
                kl, kf, ki = KLFusa.apply(hS, WS, lot.hT, WT, ca.pezzo_kl, gamma)
                if somme is not None:
                    somme["kl"] = kf if "kl" not in somme else somme["kl"] + kf
                    somme["kl_inv"] = ki if "kl_inv" not in somme else somme["kl_inv"] + ki
            (kl * (scala_k / n_tok)).backward()
        g = xL.grad
        del hS, xL
        if pv is not None:
            pv.microlotto(k)                                     # fase 2: proposta (1-2) / verifica (3-16) nel `dd`
        if rumore is not None:
            rumore.micro(B * Tn)
        st.indietro(stato, g)
        del stato, g
        tot = kl.detach() if tot is None else tot + kl.detach()
    return tot, n_tok


def _micro_lotti_pesati(st, car, lato, righe, micro, seguenti, ca, WS, WT, scala_k, gamma, pv, somme, rumore,
                        pesi_d=None):
    """27/9 --peso-token: come `_micro_lotti`, ma l'indietro parte dalla perdita PESATA del micro-lotto
    (K * sum_t w_t l_t / token del passo, w a media 1 nel micro-lotto: stessa scala della perdita uniforme). Ritorna la
    somma UNIFORME (quella del registro, confrontabile col held-out); in `somme` (obbligatorio): "pesata", le
    statistiche dei pesi (`accumula_pesi`) e con cakld "kl", "kl_inv" come sempre.
    27/9 trainer-mc: con --peso-token risposta e/o --gamma-lettere i pesi e il termine delle lettere del micro-lotto li
    decide `pesi_risposta` dalle risposte della sorgente (lambda sul passo INTERO, `lambda_risposta`). 27/9 `pesi_d`
    (--peso-dominio): il peso della riga moltiplica quello del token e il termine delle lettere (gancio sul gradiente
    di hS); "pesata" resta quella dei soli token."""
    peso = peso_di(ca)
    mc = peso is not None and peso[0] == "risposta" or ca.gamma_lettere > 0
    if mc:
        lam, a_passo, T0 = _lambda_del_passo(car.src, lato, righe, ca)
        somme["risposte_lambda"] = lam
        somme["risposte_n"] = somme.get("risposte_n", 0) + a_passo
        somme["risposte_token"] = somme.get("risposte_token", 0) + len(righe) * T0
    n_tok = None
    tot = None
    for k, m in enumerate(micro):
        lot = car.prendi(lato, m)
        if k < len(seguenti) and seguenti[k]:
            car.prenota(lato, seguenti[k])
        B, Tn = lot.ids.shape
        if n_tok is None:
            n_tok = len(righe) * Tn
        stato = st.avanti(lot.ids, inizi=lot.inizi)
        xL = stato["xL"].detach().requires_grad_(True)
        dg = {"vuole_l": True} if (ca.attribuzione and ca.righe_mc > 0) else {}
        if peso is not None and peso[0] == "generati":     # 12/10 OPD: pesi 0 alla KL di base + termine OPD in dg
            import e2e_onpolicy as _OPD
            peso_m, dg["opd"] = _OPD.pesi_generati(car.src, lato, m, lot.ids, ca, lot.hT.device)
            lettere_m = None
        else:
            peso_m, lettere_m = (pesi_risposta(car.src, lato, m, Tn, lam, ca, lot.hT.device) if mc else (peso, None))
        with torch.enable_grad():
            hS = st.esce(xL)
            aggancia_peso_dominio(hS, lot, pesi_d)
            uscite = KLFusa.apply(hS, WS, lot.hT, WT, ca.pezzo_kl, gamma, peso_m, dg, lettere_m)
            kw, kl = uscite[0], uscite[1]
            if gamma is not None:
                somme["kl"] = uscite[2] if "kl" not in somme else somme["kl"] + uscite[2]
                somme["kl_inv"] = uscite[3] if "kl_inv" not in somme else somme["kl_inv"] + uscite[3]
            (kw * (scala_k / n_tok)).backward()
        accumula_pesi(somme, kw.detach(), dg)
        if "l" in dg:                                     # integrata --attribuzione: perdita per origine (base / mc)
            accumula_origine(somme, dg.pop("l"), m, Tn)
        g = xL.grad
        del hS, xL
        if pv is not None:
            pv.microlotto(k)
        if rumore is not None:
            rumore.micro(B * Tn)
        st.indietro(stato, g)
        del stato, g
        tot = kl.detach() if tot is None else tot + kl.detach()
    return tot, n_tok


def accumula_origine(somme: dict, l: torch.Tensor, righe: list, T: int) -> None:
    """27/9 integrata --attribuzione con righe mc: somma della perdita della corsa (uniforme, per token) per origine della
    riga (base = teacher di sempre, mc = indice >= OFFSET_MC), sul dispositivo (niente sincronizzazione)."""
    lr = l.detach().reshape(len(righe), T).sum(1)
    mc = torch.tensor([i >= OFFSET_MC for i in righe], device=l.device)
    for chiave, maschera in (("origine_mc", mc), ("origine_base", ~mc)):
        s = (lr * maschera).sum()
        somme[chiave] = s if chiave not in somme else somme[chiave] + s
        n = int(maschera.sum()) * T
        somme[chiave + "_token"] = somme.get(chiave + "_token", 0) + n


def per_origine(somme: dict) -> Optional[dict]:
    """Voce del registro: perdita di allenamento media per token delle righe base e delle righe mc del passo."""
    if "origine_base" not in somme:
        return None
    out = {}
    for o in ("base", "mc"):
        n = somme.get(f"origine_{o}_token", 0)
        out[o] = float(somme[f"origine_{o}"]) / n if n else None
        out[f"{o}_token"] = n
    return out


def accumula_pesi(somme: dict, kw: torch.Tensor, dg: dict) -> None:
    """Somma pesata e statistiche dei pesi di un micro-lotto in `somme` (tensori sul dispositivo, nessuna
    sincronizzazione): "pesata", "pesi_n", "pesi_somma", "pesi_ess" (somma sui micro-lotti di (sum w)^2 / (N sum w^2),
    la quota di token EFFETTIVI), "pesi_max", "pesi_tagliati", "pesi_ripieghi", "pesi_micro"."""
    ess = dg["somma_w"] * dg["somma_w"] / (dg["n"] * dg["somma_w2"])
    for chiave, v in (("pesata", kw), ("pesi_n", dg["n"]), ("pesi_somma", dg["somma_w"]), ("pesi_ess", ess),
                      ("pesi_tagliati", dg["tagliati"]), ("pesi_ripieghi", dg["ripiego"])):
        somme[chiave] = v if chiave not in somme else somme[chiave] + v
    somme["pesi_max"] = dg["max_w"] if "pesi_max" not in somme else torch.maximum(somme["pesi_max"], dg["max_w"])
    if "min_w" in dg and somme.get("con_min_mediana"):  # 27/9 integrata (--attribuzione): minimo e somma delle mediane
        somme["pesi_min"] = dg["min_w"] if "pesi_min" not in somme else torch.minimum(somme["pesi_min"], dg["min_w"])
        somme["pesi_mediana"] = dg["mediana_w"] if "pesi_mediana" not in somme else somme["pesi_mediana"] + dg["mediana_w"]
    somme["pesi_micro"] = somme.get("pesi_micro", 0) + 1
    for chiave, k2 in (("pesata_marcati", "pesi_marcati"), ("kl4", "lettere_kl4"), ("kl4_pesata", "lettere_kl4_pesata"),
                       ("n_lettere", "lettere_n")):    # 27/9 trainer-mc (solo se ci sono)
        if chiave in dg:
            somme[k2] = dg[chiave] if k2 not in somme else somme[k2] + dg[chiave]


def riassunto_pesi(somme: dict) -> dict:
    """Voce del registro (UNA sincronizzazione, dopo il passo): perdita pesata media per token, media dei pesi (== 1),
    massimo, quota media di token effettivi (ESS/N), quota di token al tetto (kl) o tenuti (topk), ripieghi."""
    v = {k: float(somme[k]) for k in ("pesata", "pesi_n", "pesi_somma", "pesi_ess", "pesi_max", "pesi_tagliati",
                                      "pesi_ripieghi")}
    out = {"perdita_pesata": v["pesata"] / v["pesi_n"], "media_w": v["pesi_somma"] / v["pesi_n"], "max_w": v["pesi_max"],
           "ess_quota": v["pesi_ess"] / somme["pesi_micro"], "tagliati_quota": v["pesi_tagliati"] / v["pesi_n"],
           "ripieghi": int(v["pesi_ripieghi"]), "micro": int(somme["pesi_micro"])}
    if "pesi_min" in somme:                             # 27/9 integrata: minimo e mediana (media delle mediane per micro)
        out["min_w"] = float(somme["pesi_min"])
        out["mediana_w"] = float(somme["pesi_mediana"]) / somme["pesi_micro"]
    out.update(_riassunto_mc(somme, v))              # 27/9 trainer-mc
    return out


def _riassunto_mc(somme: dict, v: dict) -> dict:
    """27/9 trainer-mc: la parte del riassunto sulle risposte (vuota senza --peso-token risposta / --gamma-lettere).
    risposte = posizioni-risposta del passo, peso_risposta = 1 + lambda, quota_peso_risposte = sum_A w / sum w (== beta
    se lambda > 0), quota_massa_risposte = sum_A w l / sum w l (la quota EFFETTIVA della perdita pesata, senza lettere);
    lettere: KL4 media per posizione, termine delle lettere per token del passo, posizioni."""
    out: dict = {}
    if "risposte_lambda" in somme:
        a, lam = int(somme["risposte_n"]), float(somme["risposte_lambda"])
        out.update({"risposte": a, "peso_risposta": 1.0 + lam,
                    "quota_peso_risposte": a * (1.0 + lam) / max(v["pesi_somma"], 1e-300)})
    if "pesi_marcati" in somme:
        pes = v["pesata"] - (float(somme["lettere_kl4_pesata"]) if "lettere_kl4_pesata" in somme else 0.0)
        out["quota_massa_risposte"] = float(somme["pesi_marcati"]) / pes if pes > 0 else float("nan")
    if "lettere_n" in somme:
        n4 = int(somme["lettere_n"])
        out.update({"lettere_n": n4, "kl4_media": float(somme["lettere_kl4"]) / n4 if n4 else float("nan"),
                    "perdita_lettere": float(somme["lettere_kl4_pesata"]) / v["pesi_n"]})
    return out


def testo_pesi(r: dict) -> str:
    t = (f"perdita pesata {r['perdita_pesata']:.5f}, media w {r['media_w']:.12f}, " +
            (f"min w {r['min_w']:.3f}, mediana w {r['mediana_w']:.3f}, " if "min_w" in r else "") +
            f"max w {r['max_w']:.3f}, token "
            f"effettivi {100 * r['ess_quota']:.1f}%, al tetto/tenuti {100 * r['tagliati_quota']:.2f}%, ripieghi "
            f"{r['ripieghi']}/{r['micro']}")
    if "risposte" in r:                                   # 27/9 trainer-mc
        t += (f"; risposte {r['risposte']}, peso {r['peso_risposta']:.3f}, quota del peso "
              f"{100 * r['quota_peso_risposte']:.2f}%")
    if "quota_massa_risposte" in r:
        t += f", quota della perdita {100 * r['quota_massa_risposte']:.2f}%"
    if "lettere_n" in r:
        t += f"; lettere: KL4 media {r['kl4_media']:.5f} su {r['lettere_n']}, termine {r['perdita_lettere']:.6f}/token"
    return t


def curvatura_passo(st, car: Caricatore, lato: str, righe: list, ca: ConfigAllena, WS, scala_c: float, pv,
                    seme: int, pesi_d: Optional[dict] = None) -> None:
    """Fase 2 PV, criterio di secondo ordine: micro-lotti di MISURA della curvatura. Stesso avanti; all'indietro, al
    posto di dL/dh, un vettore casuale con covarianza = Hessiana Gauss-Newton della perdita del passo rispetto ai
    logit (KL e CE hanno la stessa: diag(p) - p p^T): g = sqrt(c) (p - e_y) WS, y ~ p dello studente (Fisher
    campionato), c = K / token del passo come la perdita. Nel `dd` (modo 3): dL/dd = 0 (theta.grad non cambia),
    H += (dY^2)^T (X^2) sul codice candidato. A pezzi di `pezzo_kl` token (logit P x V fp32 transitori).
    27/9: con perdita cakld resta QUESTA curvatura (Gauss-Newton della KL in avanti / CE): e' un'approssimazione,
    la Gauss-Newton di gamma KL(S||T) non e' diag(p) - p p^T (dipende anche da log p_S - log p_T).
    27/9 --peso-token: resta UNIFORME (quella della perdita pesata sarebbe sum_t w_t GN_t, con w dai logit del maestro
    che qui non si calcolano): approssimazione dichiarata, TECNICHE §35.
    27/9 --peso-dominio (`pesi_d`, i pesi del PASSO): ESATTA, g della riga per sqrt(w_riga) -> E[g g^T] = c w_riga GN_riga,
    la curvatura della perdita pesata (coerente col gradiente del passo P, con la verifica V e con Sophia)."""
    if _parametri_f32(st) or _parametri_lora(st):        # --allena-f32 / LoRA: la curvatura non tocca i loro gradienti
        with _fermi_extra(st):
            return _curvatura_passo(st, car, lato, righe, ca, WS, scala_c, pv, seme, pesi_d)
    return _curvatura_passo(st, car, lato, righe, ca, WS, scala_c, pv, seme, pesi_d)


def _curvatura_passo(st, car: Caricatore, lato: str, righe: list, ca: ConfigAllena, WS, scala_c: float, pv,
                     seme: int, pesi_d: Optional[dict] = None) -> None:
    micro = [righe[j:j + ca.righe_per_micro] for j in range(0, len(righe), ca.righe_per_micro)]
    gen = torch.Generator(device=st.device).manual_seed(seme)
    rad_c = math.sqrt(scala_c)
    for m in micro:
        lot = car.prendi(lato, m)
        stato = st.avanti(lot.ids, inizi=lot.inizi)
        xL = stato["xL"].detach().requires_grad_(True)
        with torch.enable_grad():
            hS = st.esce(xL)
        gh = torch.empty(hS.shape, dtype=torch.float32, device=hS.device)
        with torch.no_grad():
            for i in range(0, hS.shape[0], ca.pezzo_kl):
                lg = _logit_studente(hS[i:i + ca.pezzo_kl].detach().to(torch.float32), WS)
                p = torch.softmax(lg, dim=-1)
                del lg
                y = torch.multinomial(p, 1, generator=gen).squeeze(1)
                p[torch.arange(p.shape[0], device=p.device), y] -= 1.0
                gh[i:i + ca.pezzo_kl] = _per_testa(p, WS) * rad_c
                del p, y
            if pesi_d is not None:                           # 27/9 --peso-dominio: sqrt(w_riga) per riga (righe contigue)
                T = gh.shape[0] // len(lot.righe)
                gh.mul_(torch.tensor([math.sqrt(pesi_d[i]) for i in lot.righe], dtype=gh.dtype, device=gh.device)
                        .repeat_interleave(T).unsqueeze(-1))
        with torch.enable_grad():
            hS.backward(gh.to(hS.dtype))
        g = xL.grad
        del hS, xL, gh
        pv.curvatura()
        st.indietro(stato, g)
        del stato, g


@torch.no_grad()
def kl_sonda(st, car: Caricatore, lato: str, righe: list, ca: ConfigAllena, WS, WT) -> float:
    """Perdita dell'allenamento media per token su poche righe (sonda della regione di fiducia del passo V):
    KL(teacher || studente) con perdita kl (stesso codice di prima, al bit); 27/9 con perdita cakld la perdita
    CAKLD col gamma della corsa (ca.gamma gia' risolto), cosi' i salti si tengono solo se scende CIO' che si ottimizza.
    27/9 --peso-token: la sonda resta UNIFORME (media per token, nessun peso): i salti si tengono se scende la perdita
    uniforme, la stessa misura del held-out."""
    return perdita_sonda_lotti(st, (car.prendi(lato, righe[j:j + ca.righe_per_micro])
                                    for j in range(0, len(righe), ca.righe_per_micro)), ca, WS, WT)


@torch.no_grad()
def perdita_sonda_lotti(st, lotti, ca: ConfigAllena, WS, WT) -> float:
    """Il corpo di `kl_sonda` (stesse operazioni, stesso ordine: `kl_sonda` lo chiama con i lotti caricati uno alla
    volta) su lotti gia' caricati: 27/9 ramo v-interazione-b, la sonda a dosi misura gli stessi due micro-lotti fino a
    otto volte senza ricaricarli dal disco."""
    gamma = gamma_di(ca)
    chiave = "kl" if gamma is None else "perdita"
    tot, n = 0.0, 0
    for lot in lotti:
        SE.segno()                                     # 28/9: la sonda a dosi rimisura lotti gia' caricati (niente prendi)
        hS = st.nascosto(lot.ids, inizi=lot.inizi)
        for i in range(0, hS.shape[0], ca.pezzo_kl):
            if gamma is None:
                tot += float(kl_pezzo(hS[i:i + ca.pezzo_kl], WS, lot.hT[i:i + ca.pezzo_kl], WT, False)["kl"].sum())
            else:
                tot += float(kl_pezzo(hS[i:i + ca.pezzo_kl], WS, lot.hT[i:i + ca.pezzo_kl], WT, False,
                                      gamma=gamma)[chiave].sum())
        n += hS.shape[0]
        del hS
    return tot / max(n, 1)


def gradiente_ricentro(st, lot, ca: ConfigAllena, WS, WT, scala_c: float) -> bool:
    """27/9 ramo v-interazione-b (--pv-ricentra): UN indietro della perdita della sonda (KL o CAKLD UNIFORME, come la
    sonda; niente pesi per token) sul lotto `lot`, nello stato dei codici raggiunto dalla dose, con la scala di un
    micro-lotto del passo (`scala_c` = K / token del passo: stesse unita' di S e H del passo V, stesso rischio di
    trabocco dell'allenamento). Lo StatoV e' gia' in `prepara_ricentro` (modo 2): i buffer raccolgono dL/dW_rot sulle
    colonne dei salti. Nessuno stato dell'allenamento cambia: theta.grad torna quello di prima (i gradienti di questo
    indietro si buttano; il riferimento si rimette al suo posto), F32 e LoRA fermi (nessun gradiente), il trabocco fp16
    del ricentro non entra nel contatore del passo dopo (si rilegge prima e dopo, e si rimette). Ritorna False se il
    ricentro ha traboccato (gradiente non valido: chi chiama salta il ricentro)."""
    from contextlib import nullcontext
    SE.segno()                                         # 28/9: il ricentro usa il lotto gia' caricato (niente prendi)
    params = st.parametri()
    salvati = [p.grad for p in params]
    for p in params:
        p.grad = None
    oss = getattr(st, "oss", None)                     # (gli studenti finti delle prove non hanno l'osservatore)
    trab0 = oss.trabocco.clone() if oss is not None else None
    gamma = gamma_di(ca)
    valido = False
    try:
        with (_fermi_extra(st) if (_parametri_f32(st) or _parametri_lora(st)) else nullcontext()):
            stato = st.avanti(lot.ids, inizi=lot.inizi)
            xL = stato["xL"].detach().requires_grad_(True)
            with torch.enable_grad():
                hS = st.esce(xL)
                if gamma is None:
                    kl = KLFusa.apply(hS, WS, lot.hT, WT, ca.pezzo_kl)
                else:
                    kl, _, _ = KLFusa.apply(hS, WS, lot.hT, WT, ca.pezzo_kl, gamma)
                (kl * scala_c).backward()
            g = xL.grad
            del hS, xL
            st.indietro(stato, g)
            del stato, g
        valido = trab0 is None or bool(torch.equal(oss.trabocco, trab0))
    finally:
        if trab0 is not None:
            oss.trabocco.copy_(trab0)
        for p, gp in zip(params, salvati):
            p.grad = gp
    return valido


def inizializza_lora_ga(st, car: Caricatore, lato: str, righe: list, ca: ConfigAllena, WS, WT, scala_k: float,
                        log: Callable) -> dict:
    """27/9 --lora-init ga (e2e_lora.inizializza_ga): un avanti+indietro della perdita della corsa (KL o CAKLD, gamma
    gia' risolto) sulle righe date, con la scala `scala_k` (la direzione di dW non dipende dalla scala). Se l'indietro
    della ricetta e' a 16 bit si usa quello fp32 esatto (stesso avanti, `ricetta_temporanea`: niente trabocco con la K
    provvisoria). Dopo: gradienti di theta/F32/LoRA di questo lotto SCARTATI, contatori dello studente azzerati. Il
    modello non cambia (A0 = A, B0 = B al bit). Ritorna le misure (senza tempi: finiscono nello stato della corsa)."""
    import e2e_lora as LR
    ind = st.ric.dtype_indietro or st.ric.dtype_gemm
    basso = ind in (torch.float16, torch.bfloat16)

    def calcola():
        _azzera(st)
        if basso:
            with ricetta_temporanea(st, ricetta_indietro_fp32(st.ric)):
                gradiente_passo(st, car, lato, righe, ca, WS, WT, scala_k)
        else:
            gradiente_passo(st, car, lato, righe, ca, WS, WT, scala_k)
    bersaglio = st
    if _congelati(st):
        # 27/9 dominio-strati: LoRA-GA solo sugli adattatori degli strati ALLENATI (quelli congelati non ricevono
        # gradiente: restano come sono, al bit)
        from types import SimpleNamespace
        bersaglio = SimpleNamespace(lora={n: a for n, a in st.lora.items() if _allenata(st, a.L)},
                                    lora_cfg=getattr(st, "lora_cfg", None))
    info = LR.inizializza_ga(bersaglio, calcola, ca.lora_ga_gamma, log)
    ctl = st.controlla(tollera_trabocco=True)
    st.azzera_cambi()
    _azzera(st)
    return {**info, "righe": list(righe), "K": float(scala_k), "indietro_fp32": basso,
            "cambi_instradamento": ctl["cambi_instradamento"], "trabocco_fp16": ctl["trabocco_fp16"]}


def cancello_g1(st, car: Caricatore, lato: str, righe: list, ca: ConfigAllena, WS, WT, log: Callable) -> dict:
    """Cancello G1 sul passo 0 (§2.6, A2, R3). Solleva se fallisce. Lascia theta.grad = gradiente del passo 0
    con la scala K SCELTA (lo usa il primo passo). Nessun parametro cambia qui."""
    t0 = time.time()
    esito: dict = {"K0": ca.K0, "giri": [], "escalation": None}
    K = ca.K0
    while True:
        fp16 = st.ric.dtype_gemm == torch.float16 and (st.ric.dtype_indietro in (None, torch.float16)) \
            and not getattr(st.ric, "indietro_diviso", "")     # 27/9: indietro diviso = gamma di fp32, come ind32
        convergente, provate = False, []
        for giro in range(ca.max_giri_g1):
            _azzera(st)
            st.diagnostica_indietro(True)
            somme0 = {} if (ca.perdita == "cakld" or pesato_di(ca)) else None
            rum0 = MisuraRumore(st, None, K)                   # 27/9: rumore del gradiente del passo 0 (senza Adam)
            kl0 = gradiente_passo(st, car, lato, righe, ca, WS, WT, K, somme=somme0, rumore=rum0)
            dy = aggrega_dy(st.diagnostica_indietro(False))
            ctl = st.controlla(tollera_trabocco=True)
            st.azzera_cambi()
            stat = statistiche_g(st, K)
            for v in dy.values():
                v["trabocco_fp16"] = ctl["trabocco_fp16"]
            q01 = min(v["q01"] for v in stat.values())
            eps_g1 = min(1e-8, 1e-3 * q01)                     # dal gradiente VERO (diviso per K): non dipende da K
            K_nuova = scegli_k(K, dy, ctl["trabocco_fp16"], ca)
            r = {"eps_g1": eps_g1, "K": K_nuova, "eps_adam": eps_g1 * K_nuova}
            esito["giri"].append({"K_usata": K, "statistiche_g_perdita_media": stat, "dy": dy, "q01": q01, **r})
            log(f"G1 giro {giro}: K usata 2^{math.log2(K):.0f} -> q01 {q01:.3e}, eps_G1 {eps_g1:.3e}, "
                f"K 2^{math.log2(K_nuova):.0f}, trabocco {ctl['trabocco_fp16']}; dY azzerati " +
                ", ".join(f"{f} {v['quota_azzerati']:.2e}" for f, v in dy.items()) + "; max|dY| " +
                ", ".join(f"{f} {v['max_abs_dy']:.3e}" for f, v in dy.items()))
            if K_nuova == K:
                convergente = True
                break
            if K_nuova in provate:                             # oscillazione: nessuna K fissa in questa ricetta
                break
            provate.append(K)
            K = K_nuova
        quota_ko = any(v["quota_azzerati"] > ca.max_quota_dy_azzerati for v in dy.values())
        if fp16 and (not convergente or quota_ko or ctl["trabocco_fp16"]):
            # 26/9 sera (Angelo: nessun errore si ferma senza la soluzione): con l'indietro fp16 nessuna K tiene
            # insieme sottoflusso e trabocco -> si passa da soli all'indietro fp32 ESATTO (il riferimento di G1,
            # stesso avanti) e si rifa' G1 da capo. Registrato nell'esito: la ripresa lo riapplica.
            ric32 = ricetta_indietro_fp32(st.ric)
            log(f"G1: con l'indietro fp16 nessuna K rispetta le soglie (convergente {convergente}, dY sopra soglia "
                f"{quota_ko}, trabocco {ctl['trabocco_fp16']}) -> indietro fp32 esatto ({ric32.nome}), G1 da capo")
            imposta_ricetta(st, ric32)
            esito["escalation"] = ric32.nome
            K = ca.K0
            continue
        if not convergente:
            raise RuntimeError(f"G1: K non converge dopo {ca.max_giri_g1} giri ({[g['K'] for g in esito['giri']]})")
        break
    esito.update({"eps_g1": r["eps_g1"], "K": r["K"], "eps_adam": r["eps_adam"], "q01": q01,
                  "statistiche_g_perdita_media": stat, "dy": dy, "kl_somma_passo0": float(kl0),
                  "rumore": rum0.risultato()})
    if ca.perdita == "cakld":                            # cakld: kl_somma_passo0 e' la perdita CAKLD
        esito.update({"kl_fwd_somma_passo0": float(somme0["kl"]), "kl_inv_somma_passo0": float(somme0["kl_inv"])})
    if pesato_di(ca):                                    # --peso-token: kl_somma_passo0 resta la somma UNIFORME
        esito["pesi_passo0"] = riassunto_pesi(somme0)
        log(f"G1 con pesi per token {ca.peso_token}: " + testo_pesi(esito["pesi_passo0"]))
    grad_passo0 = _copia_grad(st)
    grad_f32_0 = [None if p.grad is None else p.grad.detach().clone() for p in _parametri_f32(st)]
    grad_lora_0 = [None if p.grad is None else p.grad.detach().clone() for p in _parametri_lora(st)]
    # riferimento: indietro fp32 sul PRIMO micro-lotto (stesso avanti), stessa K
    if fp16:
        m0 = righe[:ca.righe_per_micro]
        _azzera(st)
        gradiente_passo(st, car, lato, m0, dataclasses.replace(ca, righe_per_micro=len(m0)), WS, WT, K)
        g16 = _copia_grad(st)
        _azzera(st)
        with ricetta_temporanea(st, ricetta_indietro_fp32(st.ric)):
            gradiente_passo(st, car, lato, m0, dataclasses.replace(ca, righe_per_micro=len(m0)), WS, WT, K)
        g32 = _copia_grad(st)
        vivi = [i for i, (L, n) in enumerate(st.chiavi()) if _allenata(st, L)]   # 27/9 dominio-strati: solo gli allenati
        cf = confronta_dtheta([st.esperti(L).fam[n].d0 for i, (L, n) in enumerate(st.chiavi()) if i in vivi],
                              [g16[i] for i in vivi], [g32[i] for i in vivi])
        nulli, rotti, vive_nz = cf["nulli_in_fp16"], cf["segno_sbagliato"], cf["vive_con_g32_non_nullo"]
        esito["confronto_fp32"] = {"righe": m0, **cf}
        del g16, g32
        log(f"G1 contro l'indietro fp32 (primo micro-lotto): nulli {nulli}, segno sbagliato {rotti} su {vive_nz} "
            f"(quota {esito['confronto_fp32']['quota_rotti']:.2e}), errore relativo mediano "
            f"{esito['confronto_fp32']['err_rel_mediano']:.2e}")
    for p, g in zip(st.parametri(), grad_passo0):
        p.grad = g
    for p, g in zip(_parametri_f32(st), grad_f32_0):     # --allena-f32: anche il loro gradiente del passo 0
        p.grad = g
    for p, g in zip(_parametri_lora(st), grad_lora_0):   # --lora-nonesperti: idem per A e B
        p.grad = g
    del grad_passo0, grad_f32_0, grad_lora_0
    # cancelli
    probl = []
    for f, v in dy.items():
        if v["trabocco_fp16"]:
            probl.append(f"trabocco fp16 negli ingressi o nei gradienti degli esperti (K = {esito['K']:g})")
            break
    for f, v in dy.items():
        if v["quota_azzerati"] > ca.max_quota_dy_azzerati:
            probl.append(f"{f}: quota dL/dY azzerati {v['quota_azzerati']:.2e} > {ca.max_quota_dy_azzerati}")
    cf = esito.get("confronto_fp32")
    if cf and cf["quota_rotti"] > ca.max_quota_dtheta_rotti:
        probl.append(f"dL/dtheta nulli o di segno sbagliato {cf['quota_rotti']:.2e} > {ca.max_quota_dtheta_rotti}")
    esito["secondi"] = round(time.time() - t0, 1)
    esito["problemi"] = probl
    if probl:
        raise RuntimeError("cancello G1 FALLITO: " + "; ".join(probl))
    return esito


@torch.no_grad()
def verifica_primo_passo(st, lr: float, soglia: float) -> dict:
    """Dopo il passo 1 da theta = 0 (prova T7): quota delle scale vive con g != 0 mosse di >= soglia * lr nel
    verso -g, mediana di |passo|/lr, verso. theta.grad e' ancora quello del passo."""
    n = pieni = verso_ok = 0
    passi = []
    for (L, f), p in zip(st.chiavi(), st.parametri()):
        g = p.grad
        if g is None and not _allenata(st, L):          # 27/9 dominio-strati: strato congelato, nessun passo
            continue
        v = (st.esperti(L).fam[f].d0 != 0) & (g != 0)
        mosso = -torch.sign(g) * p
        n += int(v.sum())
        pieni += int((v & (mosso >= soglia * lr)).sum())
        verso_ok += int((v & (mosso > 0)).sum())
        passi.append(p.abs()[v].float())
    pa = torch.cat(passi)
    return {"vive_con_g": n, "quota_passo_pieno": pieni / max(n, 1), "quota_verso_giusto": verso_ok / max(n, 1),
            "mediana_passo_su_lr": _kth(pa, 0.5) / lr}


# ═══════════════════════════════════════ ottimizzatore ═══════════════════════════════════════
def nuovo_ottimizzatore(st, ca: ConfigAllena, eps: float, K: Optional[float] = None):
    """AdamW su theta (gruppo 0, come sempre). Con --allena-f32 un secondo gruppo coi master F32 e lr_f32 (stessi
    beta ed eps, weight_decay 0: una norma non va tirata verso 0); senza, la lista di parametri di sempre.
    27/9 --ottimizzatore-theta sophia: theta con SophiaTheta (e2e_sophia), F32/LoRA con lo stesso AdamW di prima
    (`OttimizzatoreSophia`); `K` serve al denominatore minimo sophia_eps * K."""
    if ca.ottimizzatore_theta == "sophia":
        return _nuovo_sophia(st, ca, eps, K)
    fused = st.device.type == "cuda"
    f32 = _parametri_f32(st)
    lo = _parametri_lora(st)                            # 27/9 --lora-nonesperti: gruppo in coda (dopo i F32)
    if not f32 and not lo:
        gruppi = st.parametri()
    else:
        gruppi = [{"params": st.parametri()}]
        if f32:
            gruppi.append({"params": f32, "lr": ca.lr_f32})
        if lo:
            gruppi.append({"params": lo, "lr": ca.lr_lora})
    opt = torch.optim.AdamW(gruppi, lr=ca.lr, betas=(ca.beta1, ca.beta2), eps=eps, weight_decay=0.0,
                            fused=fused)
    controlla_ottimizzatore(opt, ca, eps)
    return opt


def _nuovo_sophia(st, ca: ConfigAllena, eps: float, K: Optional[float]):
    import e2e_sophia as SO
    if K is None:
        raise ValueError("sophia: serve la scala K (denominatore minimo sophia_eps * K)")
    so = SO.SophiaTheta(st.parametri(), lr=ca.lr, beta1=ca.beta1, gamma=ca.sophia_gamma, beta_h=ca.sophia_beta2,
                        eps=ca.sophia_eps * float(K),
                        ombra_adam=(ca.beta2, eps) if ca.attribuzione else None)   # integrata: passo di AdamW nel registro
    f32, lo = _parametri_f32(st), _parametri_lora(st)
    adam = None
    if f32 or lo:
        gruppi = ([{"params": f32, "lr": ca.lr_f32}] if f32 else []) + ([{"params": lo, "lr": ca.lr_lora}] if lo else [])
        adam = torch.optim.AdamW(gruppi, lr=ca.lr, betas=(ca.beta1, ca.beta2), eps=eps, weight_decay=0.0,
                                 fused=st.device.type == "cuda")
    opt = SO.OttimizzatoreSophia(so, adam)
    controlla_ottimizzatore(opt, ca, eps, K)
    return opt


def betas_del_gruppo(ca: ConfigAllena, i: int) -> tuple:
    """(beta1, beta2) attesi del gruppo i: con sophia il gruppo theta ha (beta1, sophia_beta2) (EMA della curvatura)."""
    if i == 0 and ca.ottimizzatore_theta == "sophia":
        return (ca.beta1, ca.sophia_beta2)
    return (ca.beta1, ca.beta2)


def nomi_gruppi(ca: ConfigAllena) -> list:
    """Gruppi di AdamW nell'ordine: theta, F32 (--allena-f32), LoRA (--lora-nonesperti; 28/9 non con --lora-ripristina
    ferma: la LoRA e' fuori dall'ottimizzatore)."""
    return ["theta"] + (["f32"] if ca.allena_f32 else []) + \
        (["lora"] if ca.lora_rango > 0 and ca.lora_ripristina != "ferma" else [])


def lr_del_gruppo(ca: ConfigAllena, i: int) -> float:
    return {"theta": ca.lr, "f32": ca.lr_f32, "lora": ca.lr_lora}[nomi_gruppi(ca)[i]]


def _suffisso_gruppo(ca: ConfigAllena, i: int) -> str:
    n = nomi_gruppi(ca)[i]
    return "" if n == "theta" else f"_{n}"


def controlla_ottimizzatore(opt, ca: ConfigAllena, eps: float, K: Optional[float] = None) -> None:
    assert len(opt.param_groups) == len(nomi_gruppi(ca)), \
        f"{len(opt.param_groups)} gruppi, allena_f32 {ca.allena_f32}, lora_rango {ca.lora_rango}"
    sophia = ca.ottimizzatore_theta == "sophia"
    assert sophia == (getattr(opt, "tipo", None) == "sophia"), f"ottimizzatore {type(opt)} con {ca.ottimizzatore_theta}"
    for i, g in enumerate(opt.param_groups):
        assert g["weight_decay"] == 0.0, f"weight_decay {g['weight_decay']} != 0"
        if sophia and i == 0:
            assert g["lr"] == ca.lr and g["gamma"] == ca.sophia_gamma and tuple(g["betas"]) == betas_del_gruppo(ca, 0) \
                and (K is None or g["eps"] == ca.sophia_eps * float(K)), g
            continue
        assert g["lr"] == lr_del_gruppo(ca, i) and g["eps"] == eps and tuple(g["betas"]) == (ca.beta1, ca.beta2), g


def ripristino_f32(st, nomi_prima: list, valori_prima: dict, ca: "ConfigAllena", cfg_prima: dict, passo: int,
                   log: Callable) -> dict:
    """28/9 --f32-ripristina, alla RIPRESA, prima di installare i F32 della ripresa e di caricare l'ottimizzatore.
    `nomi_prima`/`valori_prima` = f32_nomi/f32 della ripresa; `cfg_prima` = config della ripresa. Ritorna
      "valori"        i valori della ripresa da installare (`imposta_f32`), senza quelli riportati al GGUF;
      "tolti_ammessi" tensori TOLTI da --allena-f32 le cui famiglie sono in --f32-ripristina: il modello usa il GGUF
                      (lo studente che non li allena li ha dal caricatore), il loro Adam si butta;
      "adam_da_zero"  tensori ANCORA allenati delle famiglie NUOVE in --f32-ripristina (assenti dalla config della
                      ripresa): tornano al valore del GGUF e ripartono con Adam da zero;
      "voce"          per stato["f32_ripristini"] (None se non si e' riportato nulla).
    Idempotente: rilanciare con la stessa riga non riporta di nuovo una famiglia ancora allenata (non e' piu' nuova).
    Tolta senza --f32-ripristina: solleva dopo, in `carica_stato_ottimizzatore` (come sempre)."""
    import e2e_studente as S
    S.famiglie_f32(ca.f32_ripristina)                     # solleva per un nome di famiglia sconosciuto
    ripr = set(ca.f32_ripristina)
    nuove = ripr - set(cfg_prima.get("f32_ripristina") or ())
    fam = S.famiglia_del_tensore_f32
    f32_ora = getattr(st, "f32", {})
    tolti = [n for n in nomi_prima if n not in f32_ora and fam(n) in ripr]
    da_zero = [n for n in nomi_prima if n in f32_ora and fam(n) in nuove]
    via = set(da_zero)
    out = {"valori": {n: v for n, v in valori_prima.items() if n in f32_ora and n not in via},
           "tolti_ammessi": tolti, "adam_da_zero": da_zero, "voce": None}
    if not (tolti or da_zero):
        if nuove:
            log(f"ripresa: --f32-ripristina {','.join(sorted(nuove))}: nessun tensore allenato nella ripresa (gia' al "
                f"valore del GGUF), nulla da riportare")
        return out
    gguf = st.valori_f32_gguf(tolti + da_zero)
    norme: dict = {}
    for n in tolti + da_zero:
        a = np.asarray(valori_prima[n], dtype=np.float64).reshape(-1)
        b = np.asarray(gguf[n], dtype=np.float64).reshape(-1)
        if a.size != b.size:                              # studente con un sottoinsieme di esperti (solo prove)
            continue
        s = norme.setdefault(fam(n), [0.0, 0.0, 0])
        s[0] += float(((a - b) ** 2).sum())
        s[1] += float((b ** 2).sum())
        s[2] += 1
    per_fam = {f: {"tensori": k, "norma": math.sqrt(d2), "relativa": math.sqrt(d2 / w2) if w2 > 0 else None}
               for f, (d2, w2, k) in sorted(norme.items())}
    out["voce"] = {"passo": int(passo), "famiglie": sorted({fam(n) for n in tolti + da_zero}),
                   "tolti": len(tolti), "ancora_allenati": len(da_zero), "scartato_per_famiglia": per_fam}
    log(f"ripresa: --f32-ripristina: {len(tolti)} tensori F32 TOLTI da --allena-f32 e {len(da_zero)} ancora allenati "
        f"riportati al valore del GGUF (decisione, da qui in avanti; Adam buttato / da zero), in tutti gli strati; "
        f"|W_ripresa - W_GGUF|/|W_GGUF| scartato: " +
        ", ".join(f"{f} {x['relativa']:.3e}" for f, x in per_fam.items() if x["relativa"] is not None))
    return out


def _kw_tolti(tolti_ammessi) -> dict:
    """28/9: `tolti_ammessi` passato SOLO se non vuoto: senza --f32-ripristina le chiamate restano quelle di prima (anche
    per chi le avvolge, come le spie delle prove)."""
    return {"tolti_ammessi": list(tolti_ammessi)} if tolti_ammessi else {}


def carica_stato_ottimizzatore(opt, sd: dict, n_theta: int, nomi_prima: list, nomi_ora: list, log: Callable,
                               lora_prima=(), lora_ora=(), tolti_ammessi=()) -> None:
    """Ripresa dello stato di Adam. Stesse famiglie F32 (e stessi adattatori LoRA): `load_state_dict` (al bit).
    Famiglie AGGIUNTE alla ripresa (decisione): theta e i tensori gia' allenati ritrovano il loro stato, i nuovi
    partono da Adam vuoto. Togliere una famiglia gia' allenata non e' una decisione (i suoi valori sono nel modello):
    solleva, SALVO i tensori in `tolti_ammessi` (28/9 --f32-ripristina: tolti E riportati al valore del GGUF, il loro
    stato di Adam si butta). 27/9 LoRA (`lora_*` = nomi dei parametri A/B, `_nomi_param_lora`): accesa alla ripresa =
    Adam vuoto per A e B; adattatori gia' allenati tolti o cambiati: solleva."""
    lora_prima, lora_ora = list(lora_prima), list(lora_ora)
    if nomi_prima == nomi_ora and lora_prima == lora_ora:
        opt.load_state_dict(sd)
        return
    ammessi = set(tolti_ammessi)
    tolti = [n for n in nomi_prima if n not in nomi_ora and n not in ammessi]
    if tolti:
        raise RuntimeError(f"ripresa: tensori F32 gia' allenati tolti da --allena-f32 ({tolti[:4]}...): nuova corsa "
                           f"(oppure --f32-ripristina con le loro famiglie: tornano al valore del GGUF)")
    if lora_prima and lora_prima != lora_ora:
        raise RuntimeError("ripresa: adattatori LoRA gia' allenati tolti o cambiati: nuova corsa")
    ora = opt.state_dict()
    stato = {i: v for i, v in sd["state"].items() if i < n_theta}
    for j, n in enumerate(nomi_ora):
        if n in nomi_prima and n_theta + nomi_prima.index(n) in sd["state"]:
            stato[n_theta + j] = sd["state"][n_theta + nomi_prima.index(n)]
    if lora_prima:                                                 # stessi adattatori, F32 aggiunti prima di loro
        o_p, o_o = n_theta + len(nomi_prima), n_theta + len(nomi_ora)
        for j in range(len(lora_ora)):
            if o_p + j in sd["state"]:
                stato[o_o + j] = sd["state"][o_p + j]
    gruppi = ora["param_groups"]
    for k, v in sd["param_groups"][0].items():                     # iperparametri salvati del gruppo theta
        if k != "params":
            gruppi[0][k] = v
    opt.load_state_dict({"state": stato, "param_groups": gruppi})
    n_tolti = sum(1 for n in nomi_prima if n not in nomi_ora)
    log(f"ripresa: {sum(1 for n in nomi_ora if n not in nomi_prima)} tensori F32 nuovi (Adam da zero, valori del GGUF), "
        f"{sum(1 for n in nomi_ora if n in nomi_prima)} ripresi" +
        (f"; {n_tolti} tolti e riportati al GGUF (--f32-ripristina, Adam buttato)" if n_tolti else "") +
        (f"; LoRA accesa ora ({len(lora_ora)} parametri, Adam da zero)" if lora_ora and not lora_prima else ""))


def _e_stato_sophia(sd) -> bool:
    return isinstance(sd, dict) and sd.get("tipo") == "sophia"


def _carica_parte_adam(adam, parte: dict, nomi_prima: list, f32_ora: list, lora_p: list, lora_o: list,
                       log: Callable, tolti_ammessi=()) -> None:
    """I gruppi AdamW (F32, LoRA) di un OttimizzatoreSophia: stessa regola di `carica_stato_ottimizzatore` a n_theta 0."""
    if nomi_prima == f32_ora and lora_p == lora_o:
        adam.load_state_dict(parte)
    else:
        carica_stato_ottimizzatore(adam, parte, 0, nomi_prima, f32_ora, log, lora_prima=lora_p, lora_ora=lora_o,
                                   **_kw_tolti(tolti_ammessi))


def carica_ottimizzatore_sophia(opt, sd: dict, st, nomi_prima: list, lora_prima: list, log: Callable,
                                tolti_ammessi=()) -> None:
    """27/9 ramo newton-theta. Ripresa quando Sophia e' accesa ORA o lo era nella ripresa (con adamw -> adamw si usa
    il percorso di prima, invariato). sophia -> sophia: stato al bit. adamw -> sophia (decisione): m = exp_avg di Adam
    (stessa EMA, stesso beta1), step ripreso, curvatura vuota (fino alla prima misura il passo e' lr segno(m)).
    sophia -> adamw (decisione): Adam da zero per theta (Sophia non ha v). F32/LoRA: come sempre."""
    import e2e_sophia as SO
    n_theta = len(st.parametri())
    f32_ora = st.chiavi_f32() if _parametri_f32(st) else []
    lora_p = _nomi_param_lora(lora_prima)
    lora_o = _nomi_param_lora(st.chiavi_lora()) if _parametri_lora(st) else []
    if getattr(opt, "tipo", None) == "sophia":
        if SO.e_sophia(sd):
            opt.sophia.load_state_dict(sd["sophia"])
            parte = sd["adam"]
        else:
            opt.sophia.load_state_dict(SO.stato_sophia_da_adam(sd, n_theta, opt.sophia))
            parte = SO.parte_adam(sd, n_theta)
            log("ripresa: ottimizzatore di theta adamw -> sophia (decisione, da qui in avanti): m = exp_avg di Adam, "
                "curvatura da misurare (fino alla prima misura il passo e' lr * segno(m))")
        if opt.adam is not None:
            if parte is None:
                log("ripresa: gruppi AdamW (F32/LoRA) accesi ora: Adam da zero")
            else:
                _carica_parte_adam(opt.adam, parte, nomi_prima, f32_ora, lora_p, lora_o, log, **_kw_tolti(tolti_ammessi))
        elif parte is not None and not (tolti_ammessi and not lora_p and set(nomi_prima) <= set(tolti_ammessi)):
            raise RuntimeError("ripresa: stato AdamW di F32/LoRA salvato, lo studente non li allena")
        return
    sd_a = SO.adam_da_parte(sd["adam"], n_theta, opt.state_dict()["param_groups"][0], sd_sophia=sd.get("sophia"))
    ripresi = sum(1 for i in range(n_theta) if i in sd_a["state"])
    log(f"ripresa: ottimizzatore di theta sophia -> adamw (decisione, da qui in avanti): " +
        (f"Adam RIPRENDE per theta da m di Sophia e v dell'ombra ({ripresi}/{n_theta} tensori)" if ripresi
         else "Adam da zero per theta"))
    if nomi_prima or f32_ora or lora_prima or lora_o:
        carica_stato_ottimizzatore(opt, sd_a, n_theta, nomi_prima, f32_ora, log, lora_prima=lora_p, lora_ora=lora_o,
                                   **_kw_tolti(tolti_ammessi))
    else:
        opt.load_state_dict(sd_a)


def aggiorna_curvatura_sophia(st, opt, pv, n_micro: int) -> Optional[int]:
    """27/9 newton-theta: dopo la curvatura del passo V, h_misurata = (N / n_c) d^2 HT per ogni theta (unita' K come
    m; d = scala fp16 del forward della curvatura). Ritorna n_c (None se in questo passo la curvatura non e' stata
    misurata: V-Bop con curvatura_ogni > 1)."""
    nc = int(getattr(pv.stato, "n_curvatura_passo", 0))
    if nc == 0:
        return None
    with torch.no_grad():
        for (L, n), p in zip(st.chiavi(), st.parametri()):
            if not _allenata(st, L):                     # 27/9 dominio-strati: h degli strati congelati ferma al bit
                continue
            f = pv.fam[(L, n)]
            HT = pv.stato.buffer(f).HT
            d = st.esperti(L).scala(n).detach().to(torch.float32).reshape(HT.shape)
            opt.sophia.aggiorna_curvatura(p, (float(n_micro) / nc) * d * d * HT)
    return nc


# ═══════════════════════════════════════ valutazione held-out ═══════════════════════════════════════
def _bersagli(ids: torch.Tensor, inizi: Optional[torch.Tensor]) -> torch.Tensor:
    """Bersaglio della posizione t = id[t+1] se t+1 e' nella stessa sequenza (riga o segmento), altrimenti -1."""
    b = torch.full_like(ids, -1)
    b[:, :-1] = ids[:, 1:]
    if inizi is not None:
        nuovo = inizi[:, 1:].to(ids.device) == 1
        b[:, :-1] = torch.where(nuovo, torch.full_like(b[:, :-1], -1), b[:, :-1])
    return b.reshape(-1)


@torch.no_grad()
def valuta(st, car: Caricatore, righe: list, ca: ConfigAllena, WS, WT) -> dict:
    """KL(teacher || studente) media per token, per dominio e (macro) sui domini non esclusi; CE sul token vero
    di studente e teacher. Una sincronizzazione alla fine.
    27/9 con perdita cakld (ca.gamma risolto): anche KL(studente || teacher) e la perdita CAKLD, per dominio e
    medie; "metrica" (migliore, arresto) = macro della perdita CAKLD sui domini; "metrica_kl" = macro della KL.
    27/9 (entrambe le perdite): "top1_uguale" = quota di token con argmax studente == argmax teacher, macro sui
    domini della metrica, e "top1_uguale_per_dominio"; solo registro (non entra in metrica ne' migliore)."""
    gamma = gamma_di(ca)
    per_riga = []
    per_riga_top1 = []
    per_riga_inv, per_riga_p = [], []
    ce_s = ce_t = n_ce = n_fuori = None
    micro = [righe[j:j + ca.righe_per_micro_valuta] for j in range(0, len(righe), ca.righe_per_micro_valuta)]
    domini = []
    Tn = None
    for k, m in enumerate(micro):
        lot = car.prendi("heldout", m)
        if k + 1 < len(micro):
            car.prenota("heldout", micro[k + 1])
        Tn = lot.ids.shape[1]
        hS = st.nascosto(lot.ids, inizi=lot.inizi)
        bers = _bersagli(lot.ids, lot.inizi)
        fuori = bers >= WS.shape[0]                  # solo con una testa ristretta (prove): contati, non taciuti
        n_fuori = fuori.sum() if n_fuori is None else n_fuori + fuori.sum()
        bers = torch.where(fuori, torch.full_like(bers, -1), bers)
        kl = torch.empty(hS.shape[0], dtype=torch.float64, device=hS.device)
        t1 = torch.empty(hS.shape[0], dtype=torch.float64, device=hS.device)
        if gamma is not None:
            kli = torch.empty_like(kl)
            per = torch.empty_like(kl)
        for i in range(0, hS.shape[0], ca.pezzo_kl):
            r = kl_pezzo(hS[i:i + ca.pezzo_kl], WS, lot.hT[i:i + ca.pezzo_kl], WT, False, bers[i:i + ca.pezzo_kl],
                         gamma=gamma, top1=True)
            kl[i:i + ca.pezzo_kl] = r["kl"]
            t1[i:i + ca.pezzo_kl] = r["top1"]
            if gamma is not None:
                kli[i:i + ca.pezzo_kl] = r["kl_inv"]
                per[i:i + ca.pezzo_kl] = r["perdita"]
            ce_s = r["ce_s"] if ce_s is None else ce_s + r["ce_s"]
            ce_t = r["ce_t"] if ce_t is None else ce_t + r["ce_t"]
            n_ce = r["n_ce"] if n_ce is None else n_ce + r["n_ce"]
        per_riga.append(kl.view(len(m), Tn).sum(1))
        per_riga_top1.append(t1.view(len(m), Tn).sum(1))
        if gamma is not None:
            per_riga_inv.append(kli.view(len(m), Tn).sum(1))
            per_riga_p.append(per.view(len(m), Tn).sum(1))
            del kli, per
        domini += lot.domini
        del hS, kl, t1
    tot = torch.cat(per_riga).cpu().tolist()                   # la sincronizzazione
    tot_t1 = torch.cat(per_riga_top1).cpu().tolist()

    def per_dominio(valori: list) -> dict:
        per_dom: dict = {}
        for d, v in zip(domini, valori):
            s = per_dom.setdefault(d, [0.0, 0])
            s[0] += v
            s[1] += Tn
        return {d: s / n for d, (s, n) in sorted(per_dom.items())}
    kl_dom = per_dominio(tot)
    usati = [d for d in kl_dom if d not in ca.escludi_domini]
    out = {"kl": sum(tot) / (len(tot) * Tn), "metrica": sum(kl_dom[d] for d in usati) / len(usati),
           "kl_per_dominio": kl_dom, "domini_metrica": usati, "ce_studente": float(ce_s) / int(n_ce),
           "ce_maestro": float(ce_t) / int(n_ce), "token": len(tot) * Tn, "ce_token": int(n_ce),
           "ce_bersagli_fuori_testa": int(n_fuori)}
    t1_dom = per_dominio(tot_t1)
    out.update({"top1_uguale": sum(t1_dom[d] for d in usati) / len(usati), "top1_uguale_per_dominio": t1_dom})
    if gamma is not None:
        tot_i = torch.cat(per_riga_inv).cpu().tolist()
        tot_p = torch.cat(per_riga_p).cpu().tolist()
        inv_dom, p_dom = per_dominio(tot_i), per_dominio(tot_p)
        out.update({"perdita_nome": "cakld", "gamma": gamma, "metrica_kl": out["metrica"],
                    "metrica": sum(p_dom[d] for d in usati) / len(usati),
                    "kl_inv": sum(tot_i) / (len(tot_i) * Tn), "perdita": sum(tot_p) / (len(tot_p) * Tn),
                    "metrica_kl_inv": sum(inv_dom[d] for d in usati) / len(usati),
                    "kl_inv_per_dominio": inv_dom, "perdita_per_dominio": p_dom})
    return out


@torch.no_grad()
def valuta_mc(st, car: Caricatore, righe: list, ca: ConfigAllena, WS, WT) -> dict:
    """27/9 trainer-mc. Held-out mc (lato LATO_HELD_MC): alle posizioni-risposta, logit delle 4 lettere dello studente e
    del maestro (`kl_lettere`). Per lo studente (e, stesso calcolo, per il maestro in "maestro"): n, accuratezza (argmax
    sulle 4, primo massimo == giusta), p(C | giusta non C) = media di p4[C] sulle risposte con giusta != C, scelta_C =
    quota di argmax == C sulle stesse, KL4 media = media di KL(p4_T || p4_S), margine medio = media di z[g] -
    max_{k != g} z[k] (logit, = differenza di log p sulle 4 e sul vocabolario), scelte = conteggio degli argmax per
    lettera. Solo registro: non entra in metrica, migliore, arresto. Una sincronizzazione alla fine."""
    src = car.src
    id4 = src.id_lettere.to(WS.device)
    zS_, zT_, g_, kl_ = [], [], [], []
    micro = [righe[j:j + ca.righe_per_micro_valuta] for j in range(0, len(righe), ca.righe_per_micro_valuta)]
    for k, m in enumerate(micro):
        lot = car.prendi(LATO_HELD_MC, m)
        if k + 1 < len(micro):
            car.prenota(LATO_HELD_MC, micro[k + 1])
        Tn = lot.ids.shape[1]
        rr = [(b, src.risposte(LATO_HELD_MC, i)) for b, i in enumerate(m)]
        rr = [(b, r) for b, r in rr if r is not None]
        if not rr:
            continue
        pos = torch.cat([b * Tn + r[0] for b, r in rr]).to(lot.hT.device)
        hS = st.nascosto(lot.ids, inizi=lot.inizi)
        r4 = kl_lettere(hS, WS, lot.hT, WT, pos, id4)
        zS_.append(r4["zS"].double())
        zT_.append(r4["zT"].double())
        kl_.append(r4["kl4"])
        g_.append(torch.cat([r[1] for _, r in rr]).to(lot.hT.device))
        del hS
    if not zS_:
        return {"n": 0}
    zS, zT, g, kl4 = torch.cat(zS_), torch.cat(zT_), torch.cat(g_), torch.cat(kl_)

    def misure(z: torch.Tensor) -> dict:
        p = torch.softmax(z, -1)
        scelta = z.argmax(-1)                             # primo massimo (come l'MMLU di casa)
        zg = z.gather(1, g[:, None]).squeeze(1)
        altri = z.masked_fill(torch.nn.functional.one_hot(g, 4).bool(), float("-inf")).max(-1).values
        nonC = g != LETTERA_C
        return {"accuratezza": (scelta == g).double().mean(), "p_C_se_giusta_non_C": p[nonC, LETTERA_C].mean(),
                "scelta_C_se_giusta_non_C": (scelta[nonC] == LETTERA_C).double().mean(),
                "margine_medio": (zg - altri).mean(), "scelte": torch.bincount(scelta, minlength=4)}
    s_, t_ = misure(zS), misure(zT)
    out = {k: (v.tolist() if k == "scelte" else float(v)) for k, v in s_.items()}   # la sincronizzazione
    out.update({"n": int(g.numel()), "kl4_media": float(kl4.mean()), "giuste": torch.bincount(g, minlength=4).tolist(),
                "n_giusta_non_C": int((g != LETTERA_C).sum()), "righe": len(righe),
                "maestro": {k: (v.tolist() if k == "scelte" else float(v)) for k, v in t_.items()}})
    return out


def testo_mc(v: dict) -> str:
    if not v.get("n"):
        return "nessuna risposta"
    m = v["maestro"]
    return (f"{v['n']} risposte: accuratezza {100 * v['accuratezza']:.2f}% (maestro {100 * m['accuratezza']:.2f}%), "
            f"p(C | giusta non C) {v['p_C_se_giusta_non_C']:.4f} (maestro {m['p_C_se_giusta_non_C']:.4f}), scelta C "
            f"{100 * v['scelta_C_se_giusta_non_C']:.2f}%, KL4 {v['kl4_media']:.5f}, margine {v['margine_medio']:.4f} "
            f"(maestro {m['margine_medio']:.4f}), scelte A-D {v['scelte']}")


def regola_arresto(storia_h: list, salite: int) -> bool:
    """Vero se la metrica held-out e' salita in `salite` misure di fila mentre la KL di addestramento (media
    dell'intervallo) scendeva (§2.5, P1). `storia_h`: [{"metrica", "kl_train_intervallo"}] in ordine."""
    if len(storia_h) < salite + 1:
        return False
    for a, b in zip(storia_h[-salite - 1:-1], storia_h[-salite:]):
        if not (b["metrica"] > a["metrica"] and b["kl_train_intervallo"] < a["kl_train_intervallo"]):
            return False
    return True


# ═══════════════════════════════════════ disco: istantanee e riprese ═══════════════════════════════════════
def _fsync_dir(p: Path) -> None:
    fd = os.open(p, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _scrivi_json(p: Path, dati) -> None:
    tmp = p.with_name(p.name + ".parziale")
    with open(tmp, "w") as f:
        json.dump(dati, f, indent=1, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)
    _fsync_dir(p.parent)


def scrivi_riga_componenti(dir_c: Path, riga: dict) -> None:
    """27/9 integrata: una riga JSON per passo in <corsa>/registro_componenti.jsonl (in coda, fsync). Un passo rifatto
    dopo una ripresa riscrive la sua riga: chi legge tiene l'ULTIMA per passo (strumenti/leggi_componenti.py)."""
    with open(dir_c / "registro_componenti.jsonl", "a") as f:
        f.write(json.dumps(riga, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def sha256_file(p: Path, blocco: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(blocco)
            if not b:
                return h.hexdigest()
            h.update(b)


# 30/9 (TT4 morto alle 11:58 scrivendo ripresa_000002.pt: @@ROOT:depot@@ al 100 %, Errno 28, passo 2 perso): il
# salvataggio non muore piu' per disco pieno. Prima di scrivere: spazio libero >= 2 x la taglia attesa (il file gemello
# piu' recente nella stessa cartella, altrimenti 0) + 1 GiB; se non c'e', ⛔ nel registro di sistema e attesa di
# SALVA_ATTESA_S, finche' lo spazio torna (lo stato resta in memoria). Un ENOSPC durante la scrittura: si toglie il
# .parziale e si riprova dopo l'attesa. Nessun'altra eccezione cambia comportamento.
SALVA_ATTESA_S = float(os.getenv("E2E_SALVA_ATTESA_S", "120"))
SALVA_MARGINE_B = 1 << 30


def _taglia_attesa(p: Path) -> int:
    radice = p.name.split("_")[0]
    gemelli = [q for q in p.parent.glob(f"{radice}_*") if q.is_file() and not q.name.endswith(".parziale")]
    return max((q.stat().st_size for q in gemelli), default=0)


def _attendi_spazio(p: Path, serve: int, motivo: str) -> None:
    import errno  # noqa: F401  (documenta il caso)
    while True:
        libero = shutil.disk_usage(p.parent).free
        if libero >= serve:
            return
        print(f"[{adesso()}] ⛔ disco: {p.parent} ha {libero / 2**30:.2f} GiB liberi, ne servono {serve / 2**30:.2f} "
              f"per {p.name} ({motivo}); attendo {SALVA_ATTESA_S:.0f} s e riprovo (stato in memoria)", file=sys.stderr,
              flush=True)
        time.sleep(SALVA_ATTESA_S)


def _torch_save_atomico(oggetto, p: Path) -> str:
    import errno
    tmp = p.with_name(p.name + ".parziale")
    serve = 2 * _taglia_attesa(p) + SALVA_MARGINE_B
    falsi = 0
    while True:
        _attendi_spazio(p, serve, "prima della scrittura")
        try:
            with open(tmp, "wb") as f:
                torch.save(oggetto, f)
                f.flush()
                os.fsync(f.fileno())
            break
        except (OSError, RuntimeError) as e:          # torch avvolge l'ENOSPC in RuntimeError (inline_container)
            pieno = (isinstance(e, OSError) and e.errno == errno.ENOSPC) or "No space left" in str(e) or \
                "unexpected pos" in str(e)
            tmp.unlink(missing_ok=True)
            if not pieno:
                raise
            if "unexpected pos" in str(e) and shutil.disk_usage(p.parent).free >= serve:
                falsi += 1                                   # errore di torch con spazio c'e': non e' il disco
                if falsi >= 3:
                    raise
            serve = max(serve, 2 * max(_taglia_attesa(p), 1) + SALVA_MARGINE_B)
            print(f"[{adesso()}] ⛔ disco pieno scrivendo {p.name}: {str(e)[:160]}; .parziale tolto, riprovo",
                  file=sys.stderr, flush=True)
            time.sleep(SALVA_ATTESA_S)
    os.replace(tmp, p)
    _fsync_dir(p.parent)
    return sha256_file(p)


def _chiave(k) -> str:
    return f"{k[0]}.{k[1]}"


def passo0_riusato(dir_c: Path, st, src, ca: ConfigAllena, held: list, sorgente: Path, log: Callable):
    """26/9 sera. La valutazione held-out del passo 0 (theta = 0, cioe' Thor-T non ancora allenato) e'
    deterministica e costa ~23 min: misurata tre volte identica (KL 0,39907) prima del cancello G1 fallito.
    La si riusa SOLO se tutto coincide ai byte: impronte sha256 dei file, theta salvato tutto zero, scale fp16
    dell'istantanea == quelle dello studente di ora (stesso GGUF, stesso export), stessi parametri di
    valutazione e stesso teacher registrati nel config.json della corsa che l'ha prodotta, stesso numero di
    token held-out. Qualunque differenza -> eccezione (niente ripiego silenzioso). Ritorna (v0, nome)."""
    meta = json.loads((sorgente / "meta.json").read_text())
    if meta["passo"] != 0:
        raise RuntimeError(f"{sorgente}: istantanea del passo {meta['passo']}, non 0")
    for f, k in (("theta.pt", "sha256_theta"), ("scale.npz", "sha256_scale")):
        if sha256_file(sorgente / f) != meta[k]:
            raise RuntimeError(f"{sorgente / f}: sha256 diverso da meta.json")
    theta = torch.load(sorgente / "theta.pt", map_location="cpu")
    if set(theta) != {_chiave(k) for k in st.chiavi()} or any(bool((t != 0).any()) for t in theta.values()):
        raise RuntimeError(f"{sorgente}: theta non e' lo zero dello stesso studente")
    if any(bool((p != 0).any()) for p in st.parametri()):
        raise RuntimeError("theta dello studente non e' zero")
    ora = st.scale_per_export()
    with np.load(sorgente / "scale.npz") as z:
        if set(z.files) != set(ora):
            raise RuntimeError(f"{sorgente}: tensori delle scale diversi")
        for n, v in ora.items():
            if not np.array_equal(z[n], np.ascontiguousarray(v).view(np.uint16)):
                raise RuntimeError(f"{sorgente}: scale di {n} diverse ai byte")
    cfg = json.loads((sorgente.parent / "config.json").read_text())
    for campo in ("pezzo_kl", "righe_per_micro_valuta", "escludi_domini"):
        prima = cfg["config"][campo]
        adesso_ = json.loads(json.dumps(getattr(ca, campo), default=str))
        if prima != adesso_:
            raise RuntimeError(f"{sorgente}: {campo} {prima} != {adesso_}")
    if Path(cfg["teacher"]).resolve() != Path(src.dir).resolve():
        raise RuntimeError(f"{sorgente}: teacher {cfg['teacher']} != {src.dir}")
    v0 = meta["valutazione"]
    if v0.get("gamma") != gamma_di(ca):                 # 27/9: la metrica del passo 0 dipende dalla perdita (CAKLD)
        raise RuntimeError(f"{sorgente}: perdita/gamma della valutazione {v0.get('perdita_nome', 'kl')}/"
                           f"{v0.get('gamma')} != {ca.perdita}/{gamma_di(ca)}")
    tok = sum(lotto_T(src, i) for i in held)
    if v0["token"] != tok:
        raise RuntimeError(f"{sorgente}: {v0['token']} token held-out != {tok} di ora")
    nome = "migliore_000000"
    if (dir_c / nome).resolve() != sorgente.resolve():
        if (dir_c / nome).exists():
            shutil.rmtree(dir_c / nome)
        shutil.copytree(sorgente, dir_c / nome)
    log(f"passo 0 RIUSATO da {sorgente} (theta 0, scale identiche ai byte, sha256 e config di valutazione "
        f"verificati): KL {v0['kl']:.5f}")
    return v0, nome


@contextmanager
def theta_scambiato(st, nuovi: list):
    """27/9: per la durata del blocco i parametri theta dello studente puntano ai tensori `nuovi` (theta_EMA): nessuna
    copia, e all'uscita tornano gli STESSI tensori di prima (al bit, anche per Adam che li tiene per oggetto)."""
    vecchi = [p.data for p in st.parametri()]
    try:
        for p, e in zip(st.parametri(), nuovi):
            p.data = e
        yield
    finally:
        for p, v in zip(st.parametri(), vecchi):
            p.data = v


def salva_istantanea(dir_c: Path, st, passo: int, valutazione: dict, prefisso: str = "migliore",
                     ema: Optional[tuple] = None) -> str:
    """Il migliore su DISCO (A7): scale fp16 ESATTE che l'avanti usa ora (nell'ordine dei blocchi del GGUF,
    `Studente.scale_per_export`, come bit uint16) + theta. Cartella migliore_<passo>, scritta in .parziale e
    rinominata. Ritorna il nome.
    27/9: `prefisso` (migliore_ema per il migliore di theta_EMA, scritto con theta scambiato); `ema` = (theta_EMA,
    valutazione di theta_EMA): anche la sottocartella ema/ nello STESSO formato (esportabile) dentro l'istantanea."""
    nome = f"{prefisso}_{passo:06d}"
    fin = dir_c / nome
    tmp = dir_c / (nome + ".parziale")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    scale = st.scale_per_export()
    np.savez(tmp / "scale.npz", **{n: np.ascontiguousarray(v).view(np.uint16) for n, v in scale.items()})
    theta = {_chiave(k): p.detach().cpu() for k, p in zip(st.chiavi(), st.parametri())}
    torch.save(theta, tmp / "theta.pt")
    meta = {"passo": passo, "valutazione": valutazione, "scritto": adesso(),
            "sha256_scale": sha256_file(tmp / "scale.npz"), "sha256_theta": sha256_file(tmp / "theta.pt"),
            "tensori": sorted(scale)}
    f32 = st.f32_per_export() if _parametri_f32(st) else {}
    if f32:                                  # --allena-f32: i valori da scrivere nel GGUF (fp32, forma del file)
        np.savez(tmp / "f32.npz", **f32)
        meta["sha256_f32"] = sha256_file(tmp / "f32.npz")
        meta["tensori_f32"] = sorted(f32)
    if _parametri_lora(st):                  # --lora-nonesperti: A e B; Delta W = s B A si ricalcola all'export
        _scrivi_lora(tmp, st, meta)
    with open(tmp / "meta.json", "w") as f:
        json.dump(meta, f, indent=1, default=str)
        f.flush()
        os.fsync(f.fileno())
    for x in tmp.iterdir():
        with open(x, "rb") as f:
            os.fsync(f.fileno())
    if ema is not None:
        with theta_scambiato(st, ema[0]):
            _scrivi_ema(tmp / "ema", st, passo, ema[1])
        _fsync_dir(tmp)
    if fin.exists():
        shutil.rmtree(fin)
    os.replace(tmp, fin)
    _fsync_dir(dir_c)
    return nome


def salva_istantanea_passo(dir_i: Path, st, passo: int, ema: Optional[list], pv, valutazione: dict, tieni: int,
                           libero_min_gib: float, log: Callable) -> dict:
    """27/9 integrata --istantanea-ogni-passo: `passo_<N>` in `dir_i` nel formato del migliore (esportabile con
    `e2e_esporta --corsa dir_i --istantanea passo_<N>` e `.../passo_<N>/ema`), con i registri dei codici in entrambe
    (`registri_pv.npz`, per l'export dei salti). Prima si controlla lo spazio (taglia dell'ultima istantanea x 1,1 +
    `libero_min_gib` liberi dopo); poi si tengono le ultime `tieni`. Qualunque errore: niente istantanea, la corsa continua
    (la riga del registro lo dice)."""
    t0 = time.time()
    nome = f"passo_{passo:06d}"
    try:
        dir_i.mkdir(parents=True, exist_ok=True)
        vecchie = sorted(d for d in dir_i.glob("passo_[0-9][0-9][0-9][0-9][0-9][0-9]") if d.is_dir())
        stima = (sum(f.stat().st_size for f in vecchie[-1].rglob("*") if f.is_file()) if vecchie else 3 << 30) * 1.1
        libero = shutil.disk_usage(dir_i).free
        if libero - stima < libero_min_gib * 2 ** 30:
            log(f"istantanea del passo {passo} SALTATA: {libero / 2 ** 30:.1f} GiB liberi in {dir_i}, ne servono "
                f"{stima / 2 ** 30:.1f} + {libero_min_gib:g}")
            return {"saltata": "spazio", "libero_gib": round(libero / 2 ** 30, 2)}
        salva_istantanea(dir_i, st, passo, valutazione, prefisso="passo",
                         ema=None if ema is None else (ema, {**valutazione, "theta": "ema"}))
        if pv is not None:
            import e2e_pv_codici as PV
            sha = PV.Registro.salva_lista(dir_i / nome / "registri_pv.npz", pv.registri)
            if ema is not None:
                shutil.copyfile(dir_i / nome / "registri_pv.npz", dir_i / nome / "ema" / "registri_pv.npz")
        tenute = sorted(d for d in dir_i.glob("passo_[0-9][0-9][0-9][0-9][0-9][0-9]") if d.is_dir())
        for d in tenute[:-tieni]:
            shutil.rmtree(d, ignore_errors=True)
        byte = sum(f.stat().st_size for f in (dir_i / nome).rglob("*") if f.is_file())
        return {"dir": str(dir_i / nome), "ema": ema is not None, "registri": None if pv is None else len(pv.registri),
                "byte": byte, "secondi": round(time.time() - t0, 1)}
    except Exception as e:                                    # disco pieno o altro: la corsa NON si ferma
        shutil.rmtree(dir_i / (nome + ".parziale"), ignore_errors=True)
        log(f"istantanea del passo {passo} NON salvata ({e!r}): la corsa continua")
        return {"errore": repr(e)}


def _scrivi_ema(d: Path, st, passo: int, valutazione: dict) -> None:
    """Sottocartella ema/ dell'istantanea (theta gia' scambiato con theta_EMA): scale.npz, theta.pt, meta.json."""
    d.mkdir()
    scale = st.scale_per_export()
    np.savez(d / "scale.npz", **{n: np.ascontiguousarray(v).view(np.uint16) for n, v in scale.items()})
    torch.save({_chiave(k): p.detach().cpu() for k, p in zip(st.chiavi(), st.parametri())}, d / "theta.pt")
    meta = {"passo": passo, "valutazione": valutazione, "scritto": adesso(), "theta": "ema",
            "sha256_scale": sha256_file(d / "scale.npz"), "sha256_theta": sha256_file(d / "theta.pt"),
            "tensori": sorted(scale)}
    f32 = st.f32_per_export() if _parametri_f32(st) else {}
    if f32:                                  # --allena-f32: i F32 CORRENTI (la EMA e' solo su theta, vedi ConfigAllena)
        np.savez(d / "f32.npz", **f32)
        meta["sha256_f32"] = sha256_file(d / "f32.npz")
        meta["tensori_f32"] = sorted(f32)
    if _parametri_lora(st):                  # --lora-nonesperti: gli adattatori CORRENTI (EMA solo su theta)
        _scrivi_lora(d, st, meta)
    with open(d / "meta.json", "w") as f:
        json.dump(meta, f, indent=1, default=str)
    for x in d.iterdir():
        with open(x, "rb") as f:
            os.fsync(f.fileno())
    _fsync_dir(d)


def _scrivi_lora(d: Path, st, meta: dict) -> None:
    """lora.npz ("<nome GGUF>|A", "<nome GGUF>|B", dtype dei master) + in meta: configurazione, voci della mappa
    (relazione HF -> file, per riportare Delta W nell'ordine del file) e tipi GGUF."""
    v = st.valori_lora()
    np.savez(d / "lora.npz", **{f"{n}|{k}": t[k].numpy() for n, t in v.items() for k in ("A", "B", "A0", "B0")
                                if k in t})
    meta["sha256_lora"] = sha256_file(d / "lora.npz")
    meta["lora"] = {**st.lora_cfg.firma(), "tensori": list(v), "voci": {n: a.voce for n, a in st.lora.items()},
                    "tipi": {n: a.tipo for n, a in st.lora.items()}}
    if any("A0" in t for t in v.values()):  # 27/9 LoRA-GA: Delta W = s (B A - B0 A0) all'export (e2e_lora.delta_file_da)
        meta["lora"]["init"] = "ga"


def carica_istantanea(dir_ist: Path) -> dict:
    """{nome del tensore GGUF: np.float16 (n_blocchi, k)} + meta; verifica gli sha256."""
    meta = json.loads((dir_ist / "meta.json").read_text())
    if sha256_file(dir_ist / "scale.npz") != meta["sha256_scale"]:
        raise RuntimeError(f"{dir_ist}/scale.npz: sha256 diverso dal meta")
    z = np.load(dir_ist / "scale.npz")
    out = {"scale": {n: z[n].view(np.float16) for n in z.files}, "meta": meta}
    if "sha256_f32" in meta:
        if sha256_file(dir_ist / "f32.npz") != meta["sha256_f32"]:
            raise RuntimeError(f"{dir_ist}/f32.npz: sha256 diverso dal meta")
        z = np.load(dir_ist / "f32.npz")
        out["f32"] = {n: np.asarray(z[n], dtype=np.float32) for n in z.files}
    if "sha256_lora" in meta:                # --lora-nonesperti: {nome: {"A", "B"}} + configurazione e voci
        if sha256_file(dir_ist / "lora.npz") != meta["sha256_lora"]:
            raise RuntimeError(f"{dir_ist}/lora.npz: sha256 diverso dal meta")
        z = np.load(dir_ist / "lora.npz")
        lo: dict = {}
        for k in z.files:
            n, p = k.rsplit("|", 1)
            lo.setdefault(n, {})[p] = np.asarray(z[k])
        out["lora"] = {n: lo[n] for n in meta["lora"]["tensori"]}
        out["lora_meta"] = meta["lora"]
    return out


def _indice_riprese(dir_c: Path) -> list:
    p = dir_c / "riprese.json"
    return json.loads(p.read_text()) if p.exists() else []


def _copia_cpu(x):
    """1/10 tt5-ottimizzato (scritture asincrone): copia PROFONDA staccata sulla CPU (tensori clonati, contenitori
    ricostruiti, il resto deepcopy): l'istantanea di cio' che la ripresa salverebbe ADESSO, indipendente dai passi dopo."""
    import copy
    if isinstance(x, torch.Tensor):
        return x.detach().to("cpu", copy=True)
    if isinstance(x, dict):
        return {k: _copia_cpu(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_copia_cpu(v) for v in x]
    if isinstance(x, tuple):
        return tuple(_copia_cpu(v) for v in x)
    return copy.deepcopy(x)


_SOSTA: set = set()                          # 2/10: _cdata delle storage in sosta sul dispositivo (etichetta «cpu»)
_SOSTA_REGISTRATA = {"fatto": False}
SOSTA_MARGINE_B = 8 << 30


def _tag_sosta(storage):
    """Etichetta di torch.save per le storage della copia in sosta: «cpu» (come la copia CPU di `_copia_cpu`), cosi' il
    file e' identico al bit; None per tutte le altre (decidono le etichette di torch, come sempre)."""
    try:
        return "cpu" if storage._cdata in _SOSTA else None
    except AttributeError:
        return None


def _registra_tag_sosta() -> None:
    if not _SOSTA_REGISTRATA["fatto"]:
        torch.serialization.register_package(1, _tag_sosta, lambda storage, location: None)
        _SOSTA_REGISTRATA["fatto"] = True


def _copia_sosta(x, cdata: set):
    """2/10 --scritture-sosta-dispositivo: come `_copia_cpu` ma i tensori clonati SUL LORO DISPOSITIVO (stessi dtype,
    forme e passi di `.to("cpu", copy=True)`: entrambi preserve_format); le storage non-CPU finiscono in `cdata`."""
    import copy
    if isinstance(x, torch.Tensor):
        c = x.detach().clone()
        if c.device.type != "cpu":
            cdata.add(c.untyped_storage()._cdata)
        return c
    if isinstance(x, dict):
        return {k: _copia_sosta(v, cdata) for k, v in x.items()}
    if isinstance(x, list):
        return [_copia_sosta(v, cdata) for v in x]
    if isinstance(x, tuple):
        return tuple(_copia_sosta(v, cdata) for v in x)
    return copy.deepcopy(x)


def _byte_tensori(x) -> int:
    if isinstance(x, torch.Tensor):
        return x.numel() * x.element_size() if x.device.type != "cpu" else 0
    if isinstance(x, dict):
        return sum(_byte_tensori(v) for v in x.values())
    if isinstance(x, (list, tuple)):
        return sum(_byte_tensori(v) for v in x)
    return 0


def sosta_possibile(*oggetti) -> bool:
    """VRAM libera >= byte da clonare + margine (8 GiB). Senza CUDA: False (la copia CPU di prima)."""
    if not torch.cuda.is_available():
        return False
    try:
        libero, _ = torch.cuda.mem_get_info()
    except Exception:  # noqa: BLE001
        return False
    return libero >= sum(_byte_tensori(o) for o in oggetti) + SOSTA_MARGINE_B


class ScrittoreAsincrono:
    """1/10 tt5-ottimizzato (--scritture-asincrone, AUDIT TT5 cura 8): UN filo che scrive su disco la ripresa (e lo stato
    V-Bop) dalla copia CPU fatta al momento del salvataggio, mentre il passo dopo calcola. Una scrittura alla volta, in
    ordine: `lancia` aspetta la precedente; `attendi` si chiama prima delle valutazioni del passo dopo (il migliore nuovo
    nasce li' e la potatura delle istantanee non deve vederlo a meta'), prima di ogni nuova ripresa e alla fine. Un
    errore della scrittura si rilancia al primo `attendi` (la corsa si ferma come si fermerebbe la scrittura sincrona).
    I file sono gli stessi della via sincrona (stessi oggetti salvati, tensori sulla CPU: `carica_ripresa` li porta sul
    dispositivo con map_location come prima)."""

    def __init__(self, log: Callable = print):
        import threading
        self._threading = threading
        self._t = None
        self._err = None
        self.log = log
        self.risultati: dict = {}

    def attendi(self) -> None:
        if self._t is not None:
            self._t.join()
            self._t = None
        if self._err is not None:
            e, self._err = self._err, None
            raise RuntimeError(f"scrittura asincrona fallita: {e!r}") from e

    def lancia(self, fn: Callable, descr: str) -> None:
        self.attendi()

        def corpo():
            t0 = time.time()
            try:
                fn()
                self.log(f"scrittura asincrona: {descr} in {time.time() - t0:.0f} s")
            except BaseException as e:  # noqa: BLE001 — si rilancia nel filo principale ad `attendi`
                self._err = e
        self._t = self._threading.Thread(target=corpo, name="scrittore-ripresa", daemon=False)
        self._t.start()

    def prendi_risultati(self) -> dict:
        r, self.risultati = self.risultati, {}
        return r


def salva_ripresa(dir_c: Path, st, opt, stato: dict, tieni: int, ema: Optional[list] = None,
                  scrittore: Optional[ScrittoreAsincrono] = None, bop: Optional[tuple] = None,
                  dopo: Optional[Callable] = None, sosta: bool = False) -> Path:
    """Punto di ripresa: theta + stato di Adam + stato del ciclo, in .parziale, fsync, rename, poi l'indice
    `riprese.json` (le ultime `tieni`) e il puntatore `ripresa_ultima.json`. Solo DOPO si cancellano le riprese
    e le istantanee del migliore che nessuna ripresa tenuta cita piu'. 27/9: con `ema` anche theta_EMA ("ema").
    1/10 `scrittore` (--scritture-asincrone): stessa ripresa, scritta da un filo a parte dalla copia CPU fatta QUI;
    `bop` = (nome, istantanea V-Bop gia' su CPU, campi) scritto PRIMA della ripresa nello stesso filo (il suo sha entra
    nello stato della ripresa come nella via sincrona); `dopo` gira nel filo dopo la ripresa (potature)."""
    passo = stato["passo"]
    p = dir_c / f"ripresa_{passo:06d}.pt"
    ogg = {"versione": VERSIONE, "theta": {_chiave(k): t.detach() for k, t in zip(st.chiavi(), st.parametri())},
           "opt": opt.state_dict(), "stato": stato}
    if ema is not None:
        ogg["ema"] = {_chiave(k): t for k, t in zip(st.chiavi(), ema)}
    if _parametri_f32(st):                   # --allena-f32: i master (valori e ordine: l'ordine dello stato di Adam)
        ogg["f32"] = st.valori_f32()
        ogg["f32_nomi"] = st.chiavi_f32()
    if _parametri_lora(st):                  # --lora-nonesperti: A, B (dtype dei master) e ordine degli adattatori
        ogg["lora"] = st.valori_lora()
        ogg["lora_nomi"] = st.chiavi_lora()
    if scrittore is None:
        return _scrivi_ripresa(dir_c, ogg, tieni)
    cdata: set = set()
    if sosta and sosta_possibile(ogg, bop[1] if bop is not None else None):
        _registra_tag_sosta()                # 2/10 --scritture-sosta-dispositivo: copia in VRAM, RAM un tensore alla volta
        ogg = _copia_sosta(ogg, cdata)
        if bop is not None:
            bop = (bop[0], _copia_sosta(bop[1], cdata), bop[2])
    else:
        if sosta:
            print(f"[{adesso()}] scrittura in sosta sul dispositivo NON possibile (VRAM libera): copia CPU di prima",
                  file=sys.stderr, flush=True)
            if bop is not None:
                bop = (bop[0], _copia_cpu(bop[1]), bop[2])
        ogg = _copia_cpu(ogg)

    def lavoro():
        _SOSTA.update(cdata)
        try:
            if bop is not None:
                nome, ist, campi = bop
                sha_b = _torch_save_atomico(ist, dir_c / nome)
                ogg["stato"]["pv"]["bop"] = {"file": nome, "sha": sha_b, **campi}
                scrittore.risultati["bop"] = dict(ogg["stato"]["pv"]["bop"])
            _scrivi_ripresa(dir_c, ogg, tieni)
        finally:
            _SOSTA.difference_update(cdata)
        if dopo is not None:
            dopo()
    scrittore.lancia(lavoro, f"ripresa del passo {passo}" + (" + stato V-Bop" if bop is not None else ""))
    return p


def _scrivi_ripresa(dir_c: Path, ogg: dict, tieni: int) -> Path:
    """La scrittura di `salva_ripresa` (sincrona, o nel filo dello scrittore): file, indice, puntatore, potature."""
    stato = ogg["stato"]
    passo = stato["passo"]
    p = dir_c / f"ripresa_{passo:06d}.pt"
    sha = _torch_save_atomico(ogg, p)
    voce = {"file": p.name, "sha256": sha, "passo": passo, "migliore": stato["migliore"]["dir"], "scritto": adesso()}
    if stato.get("migliore_ema"):
        voce["migliore_ema"] = stato["migliore_ema"]["dir"]
    indice = [v for v in _indice_riprese(dir_c) if v["file"] != p.name] + [voce]
    tenute = indice[-tieni:]
    _scrivi_json(dir_c / "riprese.json", tenute)
    _scrivi_json(dir_c / "ripresa_ultima.json", voce)
    for v in indice[:-tieni]:
        (dir_c / v["file"]).unlink(missing_ok=True)
    pota_istantanee(dir_c, _citate(tenute))
    return p


def _citate(voci: list) -> set:
    """Istantanee citate dalle riprese: il migliore e (27/9, EMA accesa) il migliore di theta_EMA."""
    return {v["migliore"] for v in voci} | {v["migliore_ema"] for v in voci if v.get("migliore_ema")}


def pota_istantanee(dir_c: Path, tenere: set) -> None:
    """Cancella le istantanee non citate (anche quelle «dal futuro» di un crollo fra istantanea e ripresa)."""
    for m in dir_c.glob("migliore_*"):
        if m.is_dir() and m.name not in tenere:
            shutil.rmtree(m)


def carica_ripresa(dir_c: Path, device) -> Optional[dict]:
    man = dir_c / "ripresa_ultima.json"
    if not man.exists():
        return None
    m = json.loads(man.read_text())
    p = dir_c / m["file"]
    if sha256_file(p) != m["sha256"]:
        raise RuntimeError(f"{p}: sha256 diverso dal manifesto (ripresa corrotta)")
    return torch.load(p, map_location=device, weights_only=False)


# ═══════════════════════════════════════ ordine dei dati ═══════════════════════════════════════
def ordine_epoca(righe: list, epoca: int, seme: int) -> list:
    """Epoca 0: l'ordine del piano del teacher (ogni prefisso ha la composizione U; permette i blocchi rotanti).
    Epoche successive: permutazione col seme (serve tutto il lato: niente rotazione)."""
    if epoca == 0:
        return list(righe)
    g = torch.Generator().manual_seed(seme * 1000 + epoca)
    return [righe[i] for i in torch.randperm(len(righe), generator=g).tolist()]


def intercala(base: list, mc: list) -> list:
    """27/9 trainer-mc: le righe mc sparse in modo uniforme nel passo (Bresenham): la posizione k (0-based, R = righe del
    passo, M = mc) e' mc se floor((k + 1) M / R) > floor(k M / R). Base e mc restano nel loro ordine."""
    R, M = len(base) + len(mc), len(mc)
    out, ib, im = [], 0, 0
    for k in range(R):
        if (k + 1) * M // R > k * M // R:
            out.append(mc[im])
            im += 1
        else:
            out.append(base[ib])
            ib += 1
    return out


class PianoRighe:
    """Quali righe fa il passo t. Senza segmenti (`segmenti` None): il calcolo di SEMPRE (epoca, posizione = divmod(t,
    passi per epoca), `ordine_epoca`, resto dell'epoca scartato), identico al bit. 27/9 trainer-mc, con segmenti
    [{"passo", "cursore_base", "cursore_mc", "righe_mc"}] (stato["mc"]["segmenti"], l'ultimo vale da `passo` in poi):
    - flusso base = concatenazione delle epoche e di `ordine_epoca(train, e, seme)` (epoca 0 = ordine del teacher, lo
      stesso di sempre); il passo t prende le R - M righe base dal cursore c(t) = cursore_base + (t - passo) (R - M);
    - flusso mc = giri delle righe mc (giro 0 nell'ordine del teacher mc, giri successivi permutati col seme); il passo
      prende M righe dal cursore m(t) = cursore_mc + (t - passo) M;
    - righe del passo = `intercala`(base, mc); passi totali: finche' il flusso base ha R - M righe entro le epoche.
    Quando l'mc si accende alla ripresa il cursore base e' quello del calcolo di sempre (e n + pos R): le righe base
    restano quelle, nello stesso ordine."""

    def __init__(self, train: list, ca: ConfigAllena, righe_mc: Optional[list] = None,
                 segmenti: Optional[list] = None):
        self.train, self.ca, self.n = list(train), ca, len(train)
        self.R = ca.righe_per_passo
        self.pe = self.n // self.R
        self.mc = list(righe_mc or [])
        self.seg = segmenti
        self._ord: dict = {}
        self._perm: dict = {}

    @property
    def classico(self) -> bool:
        return self.seg is None

    def _ordine(self, e: int) -> list:
        if e not in self._ord:
            self._ord = {k: v for k, v in self._ord.items() if k == e - 1}   # al piu' due epoche in memoria
            self._ord[e] = ordine_epoca(self.train, e, self.ca.seme)
        return self._ord[e]

    def _giro_mc(self, e: int, permutato: bool = False) -> list:
        """Il giro e delle righe mc: giro 0 nell'ordine del teacher mc, i giri dopo permutati col seme; con `permutato`
        (segmento con "permutate", --righe-mc-permutate) anche il giro 0, con la stessa formula del seme."""
        k = (e, bool(permutato and e == 0))
        if k not in self._perm:
            if e == 0 and not permutato:
                self._perm = {k: list(self.mc)}
            else:
                g = torch.Generator().manual_seed(self.ca.seme * 7919 + 13 * e + 1)
                self._perm = {k: [self.mc[i] for i in torch.randperm(len(self.mc), generator=g).tolist()]}
        return self._perm[k]

    def _segmento(self, t: int) -> dict:
        return [s for s in self.seg if s["passo"] <= t][-1]

    def cursori(self, t: int) -> tuple:
        """(cursore base, cursore mc, M) all'INIZIO del passo t (= dopo t passi)."""
        if self.classico:
            ep, pos = divmod(t, self.pe)
            return ep * self.n + pos * self.R, 0, 0
        s = self._segmento(t)
        M = int(s["righe_mc"])
        return int(s["cursore_base"]) + (t - s["passo"]) * (self.R - M), int(s["cursore_mc"]) + (t - s["passo"]) * M, M

    def totale(self) -> int:
        if self.classico:
            return self.pe * self.ca.epoche
        s = self.seg[-1]
        return int(s["passo"]) + max(0, (self.n * self.ca.epoche - int(s["cursore_base"])) // (self.R - int(s["righe_mc"])))

    def righe(self, t: int) -> tuple:
        """(righe del passo t, epoca della prima riga base)."""
        if self.classico:
            ep, pos = divmod(t, self.pe)
            return self._ordine(ep)[pos * self.R:(pos + 1) * self.R], ep
        cb, cm, M = self.cursori(t)
        base = [self._ordine(c // self.n)[c % self.n] for c in range(cb, cb + self.R - M)]
        if M and not self.mc:
            raise RuntimeError(f"passo {t}: {M} righe mc chieste e nessuna riga mc nella sorgente")
        perm = bool(self._segmento(t).get("permutate", False))
        mc = [self._giro_mc(c // len(self.mc), perm)[c % len(self.mc)] for c in range(cm, cm + M)]
        return intercala(base, mc), cb // self.n

    def fine_epoca(self, t: int) -> bool:
        """Il passo t chiude un'epoca del flusso base."""
        if self.classico:
            return (t + 1) % self.pe == 0
        cb, _, M = self.cursori(t)
        return (cb + self.R - M) // self.n > cb // self.n

    def fine_prima_epoca(self, t: int) -> bool:
        if self.classico:
            return t + 1 == self.pe
        cb, _, M = self.cursori(t)
        return cb < self.n <= cb + self.R - M

    def base_consumate(self, passo: int) -> int:
        """Righe base dell'epoca 0 gia' usate dopo `passo` passi (blocchi rotanti)."""
        if self.classico:
            return passo * self.R
        return min(self.cursori(passo)[0], self.n)

    def base_iniziale(self) -> int:
        """28/9 --righe-base-da: prima riga base dell'epoca 0 che QUESTA corsa legge. 0 salvo una corsa nata con il primo
        segmento al passo 0 e cursore > 0: le righe prima non le ha mai lette, i loro blocchi non sono suoi da consumare.
        (Righe mc accese alla ripresa, EB2 -> EB3: primo segmento al passo > 0, le righe base prima le ha lette: 0.)"""
        if self.classico or int(self.seg[0]["passo"]) != 0:
            return 0
        return min(int(self.seg[0]["cursore_base"]), self.n)


def blocchi_consumabili(src, righe_usate: set) -> list:
    return [n for n, a, b in src.blocchi("train") if all(i in righe_usate for i in range(a, b))]


# ═══════════════════════════════════════ il ciclo ═══════════════════════════════════════
def _uguale_ai_byte(st, impronte: dict) -> None:
    for (L, n), v in impronte.items():
        b = st.esperti(L).fam[n].blocchi
        if int(b.view(torch.int32).sum(dtype=torch.int64)) != v:
            raise RuntimeError(f"i byte dei blocchi ({L}, {n}) sono cambiati durante l'allenamento")
        if b.requires_grad:
            raise RuntimeError(f"i byte dei blocchi ({L}, {n}) hanno un gradiente")


def impronte_byte(st) -> dict:
    """Impronta sul dispositivo dei byte dei codici (somma int32 -> int64): guardia a ogni held-out. La prova
    AL BIT dei codici e' nell'export (`verifica_patch` sul file intero) e nelle prove (sha256)."""
    return {(L, n): int(st.esperti(L).fam[n].blocchi.view(torch.int32).sum(dtype=torch.int64))
            for (L, n) in st.chiavi()}


def passo_v(st, src, car: Caricatore, righe: list, prossime: Optional[list], held: list, ca: ConfigAllena, WS, WT,
            scala_k: float, n_tok: int, pv, reg_f, seme: int, log: Callable, chiudi: bool = True) -> dict:
    """Fase 2 PV dopo il passo P: curvatura sui primi micro-lotti del passo, selezione di secondo ordine con trust
    ratio, salti sui byte (registro), sonda della regione di fiducia sul primo micro-lotto del passo DOPO (mai visto;
    held-out solo all'ultimo passo): se la KL della sonda sale, i salti si annullano al bit e rho si dimezza."""
    import e2e_pv_codici as PV
    import e2e_cancello_fase_v as CV
    if CV.attivo() and CV.modo() == "esci":                    # 2/10 cancello GPU in una COPIA della corsa
        chiudi = True
    righe_c = righe[:ca.pv_micro_curvatura * ca.righe_per_micro]
    curv = ca.pv_modo != "vbop" or pv.stato.serve_curvatura()   # V-Bop: ogni pv_curvatura_ogni passi V
    pv.tempi = {} if ca.pv_tempi_v else None                   # 2/10 --pv-tempi-v (spento: None, nessun orologio)
    t_v0 = time.time()
    if curv:
        with PV.cronometro(pv.tempi, "curvatura"):
            curvatura_passo(st, car, "train", righe_c, ca, WS, scala_k / n_tok, pv, seme,
                            pesi_d=pesi_dominio_passo(src, "train", righe, ca))    # 27/9 --peso-dominio: pesi del PASSO
    pv.fine_raccolta()
    try:
        out = _passo_v_corpo(st, src, car, righe, prossime, held, ca, WS, WT, scala_k, n_tok, pv, reg_f, log, chiudi,
                             curv)
    finally:
        tempi, pv.tempi = pv.tempi, None
    if tempi is not None:
        tempi["totale_passo_v"] = time.time() - t_v0
        out["tempi_v"] = tempi
    return out


def _passo_v_corpo(st, src, car: Caricatore, righe: list, prossime: Optional[list], held: list, ca: ConfigAllena, WS,
                   WT, scala_k: float, n_tok: int, pv, reg_f, log: Callable, chiudi: bool, curv: bool) -> dict:
    """Il corpo di `passo_v` dopo la curvatura (2/10: diviso per l'orologio di --pv-tempi-v; stesse righe di prima)."""
    import e2e_pv_codici as PV
    if not chiudi and ca.pv_modo == "vbop":     # 1/10: V-Bop ogni N passi: la EMA si aggiorna ADESSO, nessuna decisione
        with PV.cronometro(pv.tempi, "aggiorna_bop"):
            pv.stato.aggiorna_bop()
        return {"accumulo": True, "solo_statistiche_vbop": True, "passi_bop": pv.stato.passi_bop,
                "n_verifica": pv.stato.n_verifica, "n_curvatura": pv.stato.n_curvatura, "curvatura": curv}
    if not chiudi:                              # finestra di accumulo aperta: statistiche sommate, nessun salto
        return {"accumulo": True, "passi_nella_finestra": pv.stato.passi_nella_finestra,
                "n_verifica": pv.stato.n_verifica, "n_curvatura": pv.stato.n_curvatura}
    scale = {k: st.esperti(k[0]).scala(k[1]).detach() for k in pv.fam}
    if ca.pv_sonda_dosi:                        # 27/9 ramo v-interazione-b: sonda a dosi con conferma (+ ricentro)
        import e2e_cancello_fase_v as CV         # 2/10 cancello GPU delle cure V1-V3 (E2E_CANCELLO_V; spento: niente)
        if CV.cure_spente():
            ca = dataclasses.replace(ca, **{c: False for c in CV.CURE})
        if CV.attivo():
            out = CV.esegui(lambda c: _passo_v_dosi(st, src, car, prossime, held, c, WS, WT, scala_k, n_tok, pv, reg_f,
                                                    scale, log), ca, pv, reg_f, lambda: impronte_byte(st), log)
        else:
            out = _passo_v_dosi(st, src, car, prossime, held, ca, WS, WT, scala_k, n_tok, pv, reg_f, scale, log)
        if ca.pv_modo == "vbop":
            out["curvatura"] = curv
        return out
    if prossime:
        lato, sonda = "train", prossime[:ca.righe_per_micro]
        attendi_righe(src, lato, sonda, log, ca.attesa_ogni_s)
    else:
        lato, sonda = "heldout", held[:ca.righe_per_micro]
    prima = kl_sonda(st, car, lato, sonda, ca, WS, WT)
    rho = reg_f.rho_passo()
    reg, diag = pv.seleziona_e_salta(scale, rho)
    dopo = kl_sonda(st, car, lato, sonda, ca, WS, WT) if len(reg) else prima
    gq = sum(d["guadagno_previsto"] for d in diag.values())
    if ca.pv_rho_adattivo:                      # calo previsto per token = guadagno del passo / K (la perdita e' K x media)
        tiene = reg_f.esito(rho, prima, dopo, len(reg), guadagno_previsto=gq / scala_k)
    else:
        tiene = reg_f.esito(rho, prima, dopo, len(reg))
    if not tiene:
        pv.annulla(reg)
    pv.stato.chiudi_finestra()
    tot = {k2: sum(d[k2] for d in diag.values()) for k2 in ("testati", "scoperte", "convenienti", "salti")}
    out = {"rho": rho, "sonda_prima": prima, "sonda_dopo": dopo, "tenuto": tiene, **tot,
           "guadagno_previsto": gq, "registri": len(pv.registri)}
    if ca.pv_rho_adattivo:
        out["r_fiducia"] = reg_f.storia[-1].get("r")
        out["rho_dopo"] = reg_f.rho
    if ca.pv_modo == "vbop":
        out["curvatura"] = curv
        out["con_candidato"] = sum(d.get("con_candidato", 0) for d in diag.values())
    return out


def _posizioni_lettere(src, lot, righe: list) -> Optional[torch.Tensor]:
    """29/9 punto-medio: posizioni-risposta (righe di hS appiattito) delle righe mc `righe` del lato train nel lotto `lot`
    (come `valuta_mc`); None se nessuna riga ha risposte."""
    risp = getattr(src, "risposte", None)
    if risp is None:
        return None
    Tn = lot.ids.shape[1]
    rr = [(b, risp("train", i)) for b, i in enumerate(righe)]
    rr = [(b, r) for b, r in rr if r is not None]
    if not rr:
        return None
    return torch.cat([b * Tn + r[0] for b, r in rr]).to(lot.hT.device)


@torch.no_grad()
def misura_lettere(st, lot, pos: torch.Tensor, id4: torch.Tensor, WS, WT) -> float:
    """29/9 punto-medio: KL delle 4 lettere (KL(p4_T || p4_S), `kl_lettere`) media sulle posizioni-risposta del lotto."""
    SE.segno()
    hS = st.nascosto(lot.ids, inizi=lot.inizi)
    k = float(kl_lettere(hS, WS, lot.hT, WT, pos, id4)["kl4"].mean())
    del hS
    return k


def gradiente_lettere(st, lot, pos: torch.Tensor, id4: torch.Tensor, WS, WT, scala_c: float) -> bool:
    """29/9 punto-medio: UN indietro della KL delle 4 lettere (media sulle posizioni-risposta x token del lotto x `scala_c`,
    cioe' le stesse unita' della sonda generica di `gradiente_ricentro`) nello stato attuale dei codici; lo StatoV e' gia'
    in `prepara_ricentro`. Nessuno stato dell'allenamento cambia (stesse garanzie di `gradiente_ricentro`: grad di theta
    rimesso, F32/LoRA fermi, trabocco rimesso). False se ha traboccato."""
    from contextlib import nullcontext
    SE.segno()
    params = st.parametri()
    salvati = [p.grad for p in params]
    for p in params:
        p.grad = None
    oss = getattr(st, "oss", None)
    trab0 = oss.trabocco.clone() if oss is not None else None
    valido = False
    try:
        with (_fermi_extra(st) if (_parametri_f32(st) or _parametri_lora(st)) else nullcontext()):
            stato = st.avanti(lot.ids, inizi=lot.inizi)
            xL = stato["xL"].detach().requires_grad_(True)
            with torch.enable_grad():
                hS = st.esce(xL)
                k4 = kl_lettere(hS, WS, lot.hT, WT, pos, id4)["kl4"]
                (k4.mean() * (float(lot.ids.numel()) * scala_c)).backward()
            g = xL.grad
            del hS, xL
            st.indietro(stato, g)
            del stato, g
        valido = trab0 is None or bool(torch.equal(oss.trabocco, trab0))
    finally:
        if trab0 is not None:
            oss.trabocco.copy_(trab0)
        for p, gp in zip(params, salvati):
            p.grad = gp
    return valido


def _cronometrato(tempi: dict, nome: str, fn: Callable) -> Callable:
    """2/10 --pv-tempi-v: `fn` con l'orologio di `e2e_pv_codici.cronometro` (stesso valore restituito)."""
    import e2e_pv_codici as PV

    def f():
        with PV.cronometro(tempi, nome):
            return fn()
    return f


def _passo_v_dosi(st, src, car: Caricatore, prossime: Optional[list], held: list, ca: ConfigAllena, WS, WT,
                  scala_k: float, n_tok: int, pv, reg_f, scale: dict, log: Callable) -> dict:
    """27/9 ramo v-interazione-b: il passo V con `PassoV.seleziona_e_prova_dosi` (TECNICHE_E2E_PV.md §32.5 azioni 1-2).
    Sonda A = primo micro-lotto del passo dopo (la stessa della sonda di prima), sonda B = il secondo (disgiunto, mai
    visto; all'ultimo passo: i primi due micro-lotti del held-out). A e B si caricano UNA volta e restano sul
    dispositivo (2 x ~32 MB). Il ricentro (--pv-ricentra) fa UN indietro della sonda A con la scala di un micro-lotto del
    passo (`gradiente_ricentro`). rho: `RegioneFiducia.esito_dosi` (la dose parziale tenuta non dimezza rho; r sulla
    sonda B coi soli salti tenuti)."""
    import e2e_pv_codici as PV
    rpm = ca.righe_per_micro
    if prossime:
        lato, sa, sb = "train", prossime[:rpm], prossime[rpm:2 * rpm]
        attendi_righe(src, lato, sa + sb, log, ca.attesa_ogni_s)
    else:
        lato, sa, sb = "heldout", held[:rpm], held[rpm:2 * rpm]
    if not sa or not sb:
        raise ValueError(f"pv_sonda_dosi: sonde A {sa} e B {sb}: servono due micro-lotti ({lato})")
    lot_a, lot_b = car.prendi(lato, sa), car.prendi(lato, sb)
    cfg_d = PV.ConfigDosi(frazioni=tuple(float(x) for x in ca.pv_dosi_frazioni) or PV.FRAZIONI_DOSI,
                          ricentra=ca.pv_ricentra, ricentra_giri=ca.pv_ricentra_giri)
    grad = (lambda: gradiente_ricentro(st, lot_a, ca, WS, WT, scala_k / n_tok)) if ca.pv_ricentra else None
    rho = reg_f.rho_passo()
    kw_pm = {}
    if ca.pv_punto_medio:                        # 29/9 ramo punto-medio: prezzi delle primitive al punto medio
        cfg_d = PV.ConfigDosi(frazioni=cfg_d.frazioni, ricentra=cfg_d.ricentra, ricentra_giri=cfg_d.ricentra_giri,
                              punto_medio=True, pm_primitiva=ca.pv_pm_primitiva, pm_max=ca.pv_pm_max,
                              pm_filtra=ca.pv_pm_filtra, pm_golosa=ca.pv_pm_golosa,
                              ricentra_obiettivo=ca.pv_ricentra_obiettivo,
                              pm_ordine_strati=ca.pv_pm_ordine_strati, golosa_pigra=ca.pv_golosa_pigra)
        gr_pm = {"B": lambda: gradiente_ricentro(st, lot_a, ca, WS, WT, scala_k / n_tok)}
        fat = {"B": n_tok / (scala_k * max(int(lot_a.ids.numel()), 1))}
        mis_obj = mis_obj_dosi = None
        if ca.pv_pm_lettere:                     # obiettivo sulle lettere: M1 per i prezzi, M2 (disgiunto) per la conferma
            mc_ = [i for i in (prossime or []) if i >= OFFSET_MC and i not in sa and i not in sb]
            m1, m2 = mc_[:rpm], mc_[rpm:2 * rpm]
            if m1 and m2:
                lot_m1, lot_m2 = car.prendi(lato, m1), car.prendi(lato, m2)
                p1, p2 = _posizioni_lettere(car.src, lot_m1, m1), _posizioni_lettere(car.src, lot_m2, m2)
                if p1 is not None and p2 is not None:
                    id4 = car.src.id_lettere.to(lot_m1.hT.device)
                    gr_pm["A"] = lambda: gradiente_lettere(st, lot_m1, p1, id4, WS, WT, scala_k / n_tok)
                    fat["A"] = n_tok / (scala_k * max(int(lot_m1.ids.numel()), 1))
                    mis_obj = lambda: misura_lettere(st, lot_m2, p2, id4, WS, WT)  # noqa: E731
                    if ca.pv_pm_golosa:          # 29/9 pv-lettere-dosi: le dosi si decidono sulle lettere (M1)
                        mis_obj_dosi = lambda: misura_lettere(st, lot_m1, p1, id4, WS, WT)  # noqa: E731
            if "A" not in gr_pm:
                log(f"punto medio: obiettivo lettere NON disponibile a questo passo (righe mc del passo dopo {len(mc_)} < "
                    f"{2 * rpm} o senza risposte): solo la sonda generica")
        kw_pm = {"gradienti_pm": gr_pm, "fattori_pm": fat, "misura_obiettivo": mis_obj}
        if mis_obj_dosi is not None:
            kw_pm["misura_obiettivo_dosi"] = mis_obj_dosi
    from contextlib import nullcontext
    if ca.pv_riuso:                              # 29/9 tt3-cure: riuso fra le prove (stessi numeri, meno lavoro)
        import e2e_riuso_prove as RP
        riuso = RP.RiusoProve(st, pv.stato, max_lotti=ca.pv_riuso_lotti)
    else:
        riuso = nullcontext()
    mis_a = lambda: perdita_sonda_lotti(st, [lot_a], ca, WS, WT)  # noqa: E731
    mis_b = lambda: perdita_sonda_lotti(st, [lot_b], ca, WS, WT)  # noqa: E731
    tempi = pv.tempi if ca.pv_tempi_v else None            # 2/10 --pv-tempi-v: orologio per sotto-fase (passo_v)
    if tempi is not None:
        mis_a, mis_b = _cronometrato(tempi, "misura_A", mis_a), _cronometrato(tempi, "misura_B", mis_b)
        grad = _cronometrato(tempi, "gradiente_ricentro", grad) if grad is not None else None
        if kw_pm:
            kw_pm["gradienti_pm"] = {k: _cronometrato(tempi, f"gradiente_pm_{k}", f)
                                     for k, f in kw_pm["gradienti_pm"].items()}
            for k, nome in (("misura_obiettivo", "lettere_M2"), ("misura_obiettivo_dosi", "lettere_M1")):
                if kw_pm.get(k) is not None:
                    kw_pm[k] = _cronometrato(tempi, nome, kw_pm[k])
    pv.stato.salta_dd_superflui = bool(ca.pv_dd_solo_salti)    # 2/10 V1 (spento: False, come sempre)
    PV.CONTI_DD["saltati"] = 0
    try:
        with riuso:
            with PV.cronometro(tempi, "seleziona_e_prova_dosi"):
                reg, diag, es = pv.seleziona_e_prova_dosi(scale, rho, mis_a, mis_b, cfg_d, grad, **kw_pm)
    finally:
        pv.stato.salta_dd_superflui = False
    if ca.pv_dd_solo_salti:
        es["dd_saltati"] = PV.CONTI_DD["saltati"]
    if ca.pv_riuso:
        es["riuso"] = dict(riuso.conta)
    if es.get("ricentro"):                      # lascia-uno-fuori: unita' di un micro-lotto -> per token della sonda
        es["ricentro"]["fattore_per_token"] = n_tok / (scala_k * max(int(lot_a.ids.numel()), 1))
    del lot_a, lot_b
    tiene = es["tenuto"]
    if es["confermata"]:                           # decisione e r sulla sonda di conferma (B), coi soli salti tenuti
        prima_r, dopo_r = es["sonda_prima_b"], es["sonda_dopo_b"]
    else:
        prima_r, dopo_r = es["sonda_prima_a"], es["sonda_dopo_a"]
    reg_f.esito_dosi(rho, prima_r, dopo_r, es["salti_tenuti"], tiene, es["parziale"],
                     guadagno_previsto=(es["guadagno_previsto_tenuti"] / scala_k) if ca.pv_rho_adattivo else None)
    pv.stato.chiudi_finestra()
    tot = {k2: sum(d[k2] for d in diag.values()) for k2 in ("testati", "scoperte", "convenienti", "salti")}
    out = {"rho": rho, "sonda_prima": es["sonda_prima_a"], "sonda_dopo": es["sonda_dopo_a"], "tenuto": tiene, **tot,
           "guadagno_previsto": es["guadagno_previsto"], "registri": len(pv.registri),
           "salti_tenuti": es["salti_tenuti"], "guadagno_previsto_tenuti": es["guadagno_previsto_tenuti"],
           "dose_parziale": es["parziale"], "dosi": es["dosi"], "dose": es["dose"],
           "conferma": {"prima": es["sonda_prima_b"], "dose": es["sonda_b_dose"], "dopo": es["sonda_dopo_b"],
                        "confermata": es["confermata"]},
           "ricentro": es["ricentro"]}
    if "punto_medio" in es:                      # 29/9 ramo punto-medio (spento: chiave assente, stato al bit)
        out["punto_medio"] = es["punto_medio"]
    if "riuso" in es:                            # 29/9 tt3-cure (spento: chiave assente)
        out["riuso"] = es["riuso"]
    if "dd_saltati" in es:                       # 2/10 tt5-fase-v V1 (spento: chiave assente)
        out["dd_saltati"] = es["dd_saltati"]
    if ca.pv_rho_adattivo:
        out["r_fiducia"] = reg_f.storia[-1].get("r")
        out["rho_dopo"] = reg_f.rho
    if ca.pv_modo == "vbop":
        out["con_candidato"] = sum(d.get("con_candidato", 0) for d in diag.values())
    return out


def testo_tempi_v(tv: dict, ev: Optional[dict] = None) -> str:
    """2/10 --pv-tempi-v: riga di registro dei secondi per sotto-fase del passo V (e numero di chiamate)."""
    voci = sorted((k for k in tv if not k.startswith("n_") and k != "totale_passo_v"), key=lambda k: -tv[k])
    t = f"tempi V (totale {tv.get('totale_passo_v', 0.0):.0f} s): " + ", ".join(
        f"{k} {tv[k]:.1f} s" + (f" x{tv['n_' + k]}" if tv.get("n_" + k, 1) > 1 else "") for k in voci)
    if ev is not None and ev.get("riuso"):
        r = ev["riuso"]
        t += (f"; riuso: strati avanti {r.get('strati_avanti')} (+{r.get('strati_saltati')} saltati), indietro "
              f"{r.get('strati_indietro')} (+{r.get('strati_indietro_saltati')})")
    if ev is not None and "dd_saltati" in ev:
        t += f"; dL/dd saltati {ev['dd_saltati']}"
    return t


def testo_punto_medio(pm: dict) -> str:
    """29/9 ramo punto-medio: riga di registro dei prezzi al punto medio (per token della sonda)."""
    if not pm.get("valido", True):
        return "punto medio: gradiente in q0 NON valido (trabocco): nessun salto"
    t = (f"punto medio ({pm['primitiva']}): {pm['prezzate']}/{pm['primitive']} primitive prezzate, {pm['indietro']} "
         f"indietro, scelte {len(pm.get('scelte', []))}")
    for r in pm.get("per_primitiva", [])[:12]:
        t += f"; {r['primitiva']} n{r['salti']} prev {r['guadagno_previsto']:.4g} corr " + \
             "/".join(f"{r[k]:+.4g}" for k in sorted(r) if k.startswith("corrente_")) + " medio " + \
             "/".join(f"{r[k]:+.4g}" for k in sorted(r) if k.startswith("s_") and not k.endswith("filtrata")) + \
             f" tenuti {r['tenuti']}"
    if "previsto_B" in pm:
        t += f"; previsto B {pm['previsto_B']:+.5f}"
    if pm.get("obiettivo"):
        o = pm["obiettivo"]
        t += f"; lettere M2 {o['prima']:.5f} -> " + \
            (f"{pm['obiettivo_dose']:.5f} alla dose -> " if pm.get("obiettivo_dose") is not None else "") + \
            f"{o['dopo']:.5f} ({'confermato' if o['confermato'] else 'NON confermato'})"
    return t


def testo_dosi(ev: dict) -> str:
    """Riga di registro della sonda a dosi e del ricentro (27/9 ramo v-interazione-b)."""
    if ev["dosi"] and "tenuta" in ev["dosi"][0]:    # 29/9 pv-lettere-dosi: dosi golose (una primitiva per voce)
        t = "dosi golose " + ", ".join(
            f"{x['salti']}{'+' if x['tenuta'] else '-'}:A {x['sonda_a']:.5f}" +
            (f"/L {x['obiettivo']:.5f}" if x.get("obiettivo") is not None else "") for x in ev["dosi"])
    else:
        t = "dosi " + ", ".join(f"{x['salti']}:{x['sonda_a']:.5f}" for x in ev["dosi"])
    if ev.get("dose"):
        t += f"; scelta {ev['dose']['salti']} ({100 * ev['dose']['frazione']:.2f}%)"
    c = ev["conferma"]
    if c["prima"] is not None:
        conf_b = c.get("dose") is not None and c["dose"] <= c["prima"]   # 29/9 tt3-cure: la conferma di B e' questa
        t += f"; sonda B {c['prima']:.5f} -> " + (f"{c['dose']:.5f}" if c.get("dose") is not None else "(non misurata)") + \
             (" confermata" if conf_b else " NON confermata")
        if conf_b and not c["confermata"]:
            t += " (poi tutto annullato dalla conferma dell'obiettivo, vedi 'lettere M2')"
        if c.get("dose") is not None and c["dopo"] != c["dose"]:
            t += f", dopo il ricentro {c['dopo']:.5f}"
    ri = ev.get("ricentro")
    if ri:
        t += (f"; ricentro: {ri['dannosi']} dannosi, {ri['annullati']} annullati in {len(ri['giri'])} giri, "
              f"{ri['tenuti']} tenuti" if ri["valido"] else "; ricentro SALTATO (gradiente non valido)")
        if ri["valido"] and ri["giri"] and "obiettivo_dopo" in ri["giri"][0]:   # 29/9 tt3-cure --pv-ricentra-obiettivo
            t += " (lettere M1 per giro " + ", ".join(
                f"{g['obiettivo_prima']:.5f}->{g['obiettivo_dopo']:.5f}{'+' if g['accettato'] else '-'}"
                for g in ri["giri"]) + ")"
    return t + f"; tenuti {ev['salti_tenuti']}/{ev['salti']}"


def _finestra_statov(ca: ConfigAllena) -> int:
    """1/10 TT5-ottimizzato: la finestra dello StatoV. PV: pv_passi_accumulo (somma di S/Q su N passi). V-Bop: 1 sempre
    (pv_passi_accumulo e' la cadenza delle decisioni; la memoria fra i passi e' la EMA di V-Bop)."""
    return 1 if ca.pv_modo == "vbop" else int(ca.pv_passi_accumulo)


def _salva_registri_pv(dir_c: Path, pv, stato: dict) -> None:
    """Registri PV della ripresa (atomico, sha nello stato che la ripresa salva) e potatura di quelli non piu' citati."""
    import e2e_pv_codici as PV
    nome = f"registri_pv_{stato['passo']:06d}.npz"
    sha = PV.Registro.salva_lista(dir_c / nome, pv.registri)
    stato["pv"].update({"file": nome, "sha": sha, "registri": len(pv.registri)})


def _salva_accumulo_pv(dir_c: Path, pv, stato: dict) -> None:
    """Finestra di accumulo a meta' (27/9): le statistiche V sommate finora vanno nella ripresa (atomico, sha nello stato),
    altrimenti un riavvio le perderebbe. Finestra chiusa (o accumulo 1): niente file."""
    st_v = pv.stato
    if st_v.passi_accumulo > 1 and st_v.passi_nella_finestra > 0:
        nome = f"accumulo_pv_{stato['passo']:06d}.pt"
        sha = _torch_save_atomico(st_v.istantanea(), dir_c / nome)
        stato["pv"]["accumulo"] = {"file": nome, "sha": sha, "passi_accumulo": st_v.passi_accumulo,
                                   "passi_nella_finestra": st_v.passi_nella_finestra}
    else:
        stato["pv"]["accumulo"] = None


def _salva_bop_pv(dir_c: Path, pv, stato: dict) -> None:
    """27/9 V-Bop: lo stato persistente per gruppo (candidati, m, v, H, eta', scale) nella ripresa (atomico, sha nello
    stato). Solo con pv_modo vbop: col PV di prima nessun file e nessuna chiave nuova."""
    st_v = pv.stato
    if st_v.bop is None:
        return
    nome = f"bop_pv_{stato['passo']:06d}.pt"
    sha = _torch_save_atomico(st_v.istantanea_bop(), dir_c / nome)
    stato["pv"]["bop"] = {"file": nome, "sha": sha, "passi_bop": st_v.passi_bop, "byte": st_v.byte_bop()}


def _bop_da_scrivere(pv, stato: dict, sosta: bool = False) -> Optional[tuple]:
    """1/10 (--scritture-asincrone): (nome, istantanea V-Bop su CPU, campi) da scrivere nel filo dello scrittore; None
    senza V-Bop. Stessi nome, contenuto e campi di `_salva_bop_pv`. 2/10 `sosta`: l'istantanea coi tensori VIVI del
    dispositivo (nessuna copia qui: `salva_ripresa` la clona in VRAM, o sulla CPU se la VRAM non basta)."""
    import e2e_pv_codici as PV
    st_v = pv.stato
    if st_v.bop is None:
        return None
    if sosta:
        ist = {"gamma": float(st_v.bop.gamma), "curvatura_ogni": int(st_v.bop.curvatura_ogni),
               "passi_bop": st_v.passi_bop,
               "stato": {f"{k[0]}.{k[1]}": {n: getattr(z, n).detach() for n in PV._CAMPI_BOP}
                         for k, z in st_v.bop_.items()}}
    else:
        ist = st_v.istantanea_bop()
    return (f"bop_pv_{stato['passo']:06d}.pt", ist, {"passi_bop": st_v.passi_bop, "byte": st_v.byte_bop()})


def _pota_registri_pv(dir_c: Path) -> None:
    tenuti = {f"registri_pv_{v['passo']:06d}.npz" for v in _indice_riprese(dir_c)}
    for f in dir_c.glob("registri_pv_*.npz"):
        if f.name not in tenuti:
            f.unlink(missing_ok=True)
    tenuti_a = {f"accumulo_pv_{v['passo']:06d}.pt" for v in _indice_riprese(dir_c)}
    for f in dir_c.glob("accumulo_pv_*.pt"):
        if f.name not in tenuti_a:
            f.unlink(missing_ok=True)
    tenuti_b = {f"bop_pv_{v['passo']:06d}.pt" for v in _indice_riprese(dir_c)}
    for f in dir_c.glob("bop_pv_*.pt"):
        if f.name not in tenuti_b:
            f.unlink(missing_ok=True)


# ═══════════════════════════════════════ 27/9 LoTA-QAF (e2e_lota.py) ═══════════════════════════════════════
def passo_lota(st, src, car: Caricatore, prossime: Optional[list], held: list, ca: ConfigAllena, WS, WT,
               scala_k: float, lota, n_lota: int, log: Callable) -> dict:
    """Passo LoTA dopo il passo P: t-SignSGD su A e B (gradiente raccolto nel dd dei micro-lotti del passo), poi i
    codici = clamp(T0 + W^). Se cambia almeno un codice e la sonda e' accesa: perdita sulla sonda (primo micro-lotto del
    passo DOPO, mai visto; held-out all'ultimo passo) prima e dopo; se sale, A e B tornano indietro (byte al bit) e p si
    dimezza per i passi dopo. Senza codici cambiati niente sonda (A e B si caricano sotto omega senza costo di avanti)."""
    t0 = time.time()
    k = lota.passi + 1
    p = lota.p_passo(k, n_lota)
    mosse = lota.passo(p, scala_k)
    if ca.lota_sonda and ca.lota_sonda_finestra > 0:
        return _passo_lota_finestra(st, src, car, ca, WS, WT, lota, mosse, k, p, t0, log)
    conta = lota.sincronizza(applica=False)
    n = conta["cambiati"]
    sonda = ca.lota_sonda and n > 0
    prima = dopo = None
    if sonda:
        if prossime:
            lato, righe_s = "train", prossime[:ca.righe_per_micro]
            attendi_righe(src, lato, righe_s, log, ca.attesa_ogni_s)
        else:
            lato, righe_s = "heldout", held[:ca.righe_per_micro]
        prima = kl_sonda(st, car, lato, righe_s, ca, WS, WT)
    fatto = lota.sincronizza() if n else conta
    if sonda:
        dopo = kl_sonda(st, car, lato, righe_s, ca, WS, WT)
    tiene = lota.esito(prima, dopo, n) if sonda else True
    if not tiene:
        fatto = lota.annulla(mosse)
    lota.passi = k
    voci = {nome: {x: sum(m[nome][i] for m in mosse.values()) for x, i in (("scelte", 2), ("cambiate", 3))}
            for nome in ("A", "B")}
    return {"passo_lota": k, "p": p, "init": lota.ultimo_init, "mult_dopo": lota.mult, "voci_A": voci["A"], "voci_B": voci["B"],
            "codici_cambiati": n, "tenuto": tiene, "sonda_prima": prima, "sonda_dopo": dopo,
            "codici_diversi_da_base": fatto["diversi_da_base"], "w_hat_non_nulli": fatto["w_hat_non_nulli"],
            "secondi": round(time.time() - t0, 2)}


def _passo_lota_finestra(st, src, car: Caricatore, ca: ConfigAllena, WS, WT, lota, mosse: dict, k: int, p: float,
                         t0: float, log: Callable) -> dict:
    """--lota-sonda-finestra N: i codici del passo si scrivono sempre; la sonda e' FISSA (le prime righe del lato train,
    le stesse a ogni passo: valori confrontabili) e misurata dopo; `StatoLoTA.esito_finestra` fotografa il migliore o, dopo
    N passi senza batterlo, riporta A e B (e i byte) al migliore e dimezza p. Un avanti a passo invece di due."""
    n = lota.sincronizza()["cambiati"]
    righe_s = src.righe_previste("train")[:ca.righe_per_micro]
    attendi_righe(src, "train", righe_s, log, ca.attesa_ogni_s)
    v = kl_sonda(st, car, "train", righe_s, ca, WS, WT)
    esf = lota.esito_finestra(v, k)
    fatto = lota.sincronizza(applica=False)
    lota.passi = k
    voci = {nome: {x: sum(m[nome][i] for m in mosse.values()) for x, i in (("scelte", 2), ("cambiate", 3))}
            for nome in ("A", "B")}
    return {"passo_lota": k, "p": p, "init": lota.ultimo_init, "mult_dopo": lota.mult, "voci_A": voci["A"], "voci_B": voci["B"],
            "codici_cambiati": n, "tenuto": esf["azione"] != "ritorno", "sonda_prima": None, "sonda_dopo": v,
            "finestra": esf, "codici_diversi_da_base": fatto["diversi_da_base"], "w_hat_non_nulli": fatto["w_hat_non_nulli"],
            "secondi": round(time.time() - t0, 2)}


def _registri_codici(pv, lota, congelati: list) -> Optional[list]:
    """Registri dei CODICI del modello di adesso, nell'ordine d'applicazione sul GGUF: PV (fase 2) oppure PV congelati +
    registro netto LoTA. None se i codici sono quelli del GGUF (niente PV, niente LoTA)."""
    if pv is not None:
        return list(pv.registri)
    if lota is not None:
        return list(congelati) + lota.registri_netti()
    return None


def _salva_lota(dir_c: Path, lota, stato: dict, impr: dict) -> None:
    """A, B (2 bit), u (solo con lr < 1), passi, moltiplicatore nella ripresa (atomico, sha nello stato) e l'impronta dei
    byte che la ripresa deve ritrovare dopo `sincronizza`."""
    nome = f"lota_{stato['passo']:06d}.pt"
    sha = _torch_save_atomico(lota.istantanea(), dir_c / nome)
    stato["lota"].update({"file": nome, "sha": sha, "passi": lota.passi, "mult": lota.mult,
                          "impronte": {f"{L}.{n}": v for (L, n), v in impr.items()}})


def _pota_lota(dir_c: Path) -> None:
    tenuti = {f"lota_{v['passo']:06d}.pt" for v in _indice_riprese(dir_c)}
    for f in dir_c.glob("lota_*.pt"):
        if f.name not in tenuti:
            f.unlink(missing_ok=True)


# 26/9 sera: campi che cambiano COME la GPU raggruppa le stesse righe, non cosa si calcola. Il passo somma
# kl * (K / token del passo) su micro-lotti di righe indipendenti: stessa somma per qualunque taglio (a meno
# dell'ordine di somma in virgola mobile). La ripresa li accetta cambiati e lo scrive nel registro.
# 26/9 sera: il lr si puo' cambiare alla ripresa per DECISIONE (una schedula a gradini: m e v di Adam sono statistiche del
# gradiente e non dipendono dal lr). Cambia la matematica dei passi SUCCESSIVI, non quella gia' fatta: va scritto nel registro.
# 27/9: anche la perdita (kl | cakld) e il gamma di CAKLD: decisioni registrate alla ripresa (il gamma MISURATO sta
# nello stato della corsa e si rilegge, vedi `risolvi_gamma`).
# 27/9 velocita': beta2 (come il lr: gli iperparametri di Adam si riscrivono dopo load_state_dict) e la media
# esponenziale di theta (accesa alla ripresa: parte da theta di quel passo; spenta: non si aggiorna piu').
# 27/9 --allena-f32: allena_f32 (aggiungere famiglie; toglierne solleva, vedi `carica_stato_ottimizzatore`) e lr_f32.
# 28/9 --f32-ripristina: famiglie riportate al GGUF (e, se tolte da allena_f32, toglierle e' ammesso): `ripristino_f32`.
# 27/9 --lora-nonesperti: accenderla alla ripresa e' una decisione (B = 0: il modello non cambia, Adam nuovo per A e B);
# cambiare rango/alpha/famiglie di una LoRA gia' allenata o spegnerla solleva (Delta W e' nel modello); lr_lora libero.
# 28/9 --lora-ripristina zero|ferma: Delta W della ripresa SCARTATO (norme nel registro), zero = la LoRA continua da Delta W
# = 0 con Adam nuovo, ferma = fuori dall'ottimizzatore finche' la riga lo dice (`modo_ripristino_lora`, `ripristino_lora`).
# 27/9 --peso-token: i pesi per token (modo e parametri) come la perdita: decisione, da qui in avanti (i pesi non hanno
# stato: si ricalcolano a ogni micro-lotto dalla perdita del token).
_CAMPI_DI_DECISIONE = {"lr", "pv_codici", "pv_rho", "pv_micro_curvatura", "pv_n_proposta", "pv_q_fdr", "pv_q_guadagno",
                       "pv_passi_accumulo", "perdita", "gamma", "beta2", "ema_theta", "allena_f32", "lr_f32",
                       "f32_ripristina",   # 28/9: famiglie F32 riportate al GGUF alla ripresa (vedi `ripristino_f32`)
                       "pv_modo", "pv_gamma", "pv_curvatura_ogni", "pv_rho_adattivo", "pv_rho_max",
                       "pv_accetta", "pv_z_min", "pv_z_guadagno_min",
                       # 27/9 ramo v-interazione-b: come si decide sui salti (dose, ricentro) dal passo V successivo
                       "pv_sonda_dosi", "pv_ricentra", "pv_ricentra_giri", "pv_dosi_frazioni",
                       # 29/9 ramo punto-medio: prezzi al punto medio delle primitive (decisione, dal passo V successivo)
                       "pv_punto_medio", "pv_pm_primitiva", "pv_pm_max", "pv_pm_filtra", "pv_pm_lettere",
                       "pv_pm_golosa",     # 29/9 ramo pv-lettere-dosi
                       "pv_ricentra_obiettivo",   # 29/9 ramo tt3-cure (pv_riuso NON e' una decisione: stessi numeri)
                       "lora_rango", "lora_alpha", "lora_famiglie", "lr_lora", "lora_tipi",
                       "lora_ripristina",  # 28/9: LoRA azzerata (zero) o tolta dall'ottimizzatore (ferma), `ripristino_lora`
                       "peso_token", "peso_alfa", "peso_max", "peso_frazione",
                       "ottimizzatore_theta", "sophia_gamma", "sophia_beta2", "sophia_eps",
                       # 27/9 LoRA-GA: contano solo quando la LoRA si inizializza (corsa nuova o accesa alla ripresa); una
                       # LoRA gia' allenata con l'altra init solleva (vedi `allena`), gamma/righe dopo l'init non entrano piu'
                       "lora_init", "lora_ga_gamma", "lora_ga_righe",
                       # 27/9: aritmetica delle teste (bf16x3 / bf16): cambia l'ordine di grandezza 1e-5 dei logit
                       "testa_kl", "logit_maestro",
                       # 27/9 LoTA: accenderla alla ripresa = decisione (B = 0: codici invariati); rango/omega di adattatori
                       # gia' allenati o spegnerla sollevano (i codici sono nel modello); top-p, lr, tau, sonda liberi
                       "lota_rango", "lota_omega", "lota_top_p", "lota_top_p_fine", "lota_top_p_coda", "lota_lr",
                       "lota_tau", "lota_sonda", "lota_init", "lota_sonda_tolleranza", "lota_sonda_finestra",
                       # 27/9 trainer-mc: righe mc e pesi delle risposte, decisioni da qui in avanti (vedi `_segmenti_mc`)
                       "righe_mc", "beta_risposta", "gamma_lettere", "righe_mc_permutate",
                       # 27/9 dominio-strati: pesi per dominio (senza stato) e strati allenati (i congelati tengono il loro
                       # stato al bit: accenderli, spegnerli o cambiarli alla ripresa e' una decisione, da qui in avanti)
                       "peso_dominio", "strati_allenati"}
_CONFIG_DEFAULT = json.loads(json.dumps(asdict(ConfigAllena()), default=str))
# parametri del SOLO cancello G1: dopo che G1 e' passato (stato["g1"] nella ripresa) non entrano piu' in nessun calcolo
_CAMPI_SOLO_G1 = {"max_giri_g1", "margine_trabocco", "max_quota_dy_azzerati", "max_quota_dtheta_rotti",
                  "soglia_passo_pieno", "quota_min_passo_pieno"}
_CAMPI_DI_ESECUZIONE = {"righe_per_micro", "righe_per_micro_valuta", "attesa_ogni_s", "tieni_riprese",
                        # 27/9 integrata: l'attribuzione misura e ripristina al bit (non cambia l'allenamento)
                        "attribuzione", "attribuzione_righe", "attribuzione_solo_ogni",
                        # 27/9 integrata: istantanee a ogni passo (solo scrittura su disco)
                        "istantanea_ogni_passo", "istantanee_dir", "istantanee_tieni", "istantanee_libero_min_gib",
                        "consuma_blocchi",
                        # 29/9 tt3-cure: riuso fra le prove della fase V (stessi numeri, meno lavoro)
                        "pv_riuso", "pv_riuso_lotti",
                        # sorveglianza: QUANDO si misura il held-out e si salva, non cosa si calcola nei passi
                        "heldout_ogni", "ripresa_ogni", "heldout_righe_rapide", "heldout_pieno_ogni",
                        "tempi_fasi",         # 1/10 tt5-ottimizzato: solo orologio nel registro
                        "scritture_asincrone",   # 1/10: stessi file, scritti da un filo a parte
                        # 2/10 tt5-fase-v: stesse decisioni, meno lavoro (FASE_V.md); l'orologio della fase V
                        "pv_dd_solo_salti", "pv_pm_ordine_strati", "pv_golosa_pigra", "pv_tempi_v",
                        "scritture_sosta_dispositivo"}   # 2/10: stessi file, copia in VRAM


def _segmenti_mc(stato: dict, ca: ConfigAllena, src: Optional["SorgenteConMC"], train: list,
                 log: Callable) -> Optional[list]:
    """27/9 trainer-mc: i segmenti del piano delle righe (`PianoRighe`) in stato["mc"] = {"firme": [firma del teacher mc,
    ...], "segmenti": [{"passo", "cursore_base", "cursore_mc", "righe_mc", "firma"}]}. None = piano di sempre (nessuna
    riga mc mai usata: lo stato non ha la chiave "mc", identico al bit). Decisioni registrate:
    - mc accesa (corsa nuova o ripresa senza segmenti): primo segmento al passo di ora, cursore base = quello del piano di
      sempre (le righe base gia' fatte e le prossime restano quelle, nello stesso ordine), cursore mc 0;
    - righe_mc cambiato (anche a 0 = mc spenta) o teacher mc diverso (firma): nuovo segmento dai cursori di ora (teacher
      diverso: cursore mc da 0, la firma nuova si aggiunge a "firme"). Nulla dello stato del modello cambia."""
    info = stato.get("mc")
    seg = info.get("segmenti") if info else None
    t, M = stato["passo"], ca.righe_mc
    firma = src.firma_mc() if src is not None and src.righe_mc() else None
    if seg is None:
        N0 = int(ca.righe_base_da)                         # 28/9 --righe-base-da (0 = il piano di sempre, al bit)
        if M == 0 and N0 == 0:
            return None
        if N0 and t != 0:
            raise RuntimeError(f"--righe-base-da {N0}: vale solo per una corsa NUOVA (passo {t}, piano senza segmenti)")
        if N0 >= len(train):
            raise ValueError(f"--righe-base-da {N0}: il teacher ha {len(train)} righe di addestramento")
        cb = N0 if N0 else PianoRighe(train, ca).cursori(t)[0]
        primo = {"passo": t, "cursore_base": cb, "cursore_mc": 0, "righe_mc": M, "firma": 0}
        if ca.righe_mc_permutate:                          # integrata: la chiave c'e' solo se accesa (stato di prima al bit)
            primo["permutate"] = True
        stato["mc"] = {"firme": [firma], "segmenti": [primo]}
        if N0:
            log(f"flusso base dalla riga {N0} dell'ordine del teacher (--righe-base-da, corsa nuova): le righe 0-{N0 - 1} "
                f"non si leggono ne' si consumano; righe base per passo {ca.righe_per_passo - M}")
        if M:
            log(f"righe mc ACCESE al passo {t}: {M} su {ca.righe_per_passo} a passo (intercalate), cursore base {cb} "
                f"(stesse righe base, stesso ordine), righe mc {len(src.righe_mc())}, teacher mc {firma.get('teacher_mc')}" +
                (", righe mc PERMUTATE col seme anche nel primo giro" if ca.righe_mc_permutate else ""))
        return stato["mc"]["segmenti"]
    ult = seg[-1]
    i_f = int(ult["firma"])
    if firma is not None and firma != info["firme"][i_f]:
        info["firme"].append(firma)
        i_f = len(info["firme"]) - 1
    perm_prima = bool(ult.get("permutate", False))
    if M != int(ult["righe_mc"]) or i_f != int(ult["firma"]) or perm_prima != ca.righe_mc_permutate:
        cb, cm, _ = PianoRighe(train, ca, src.righe_mc() if firma is not None else [], seg).cursori(t)
        if i_f != int(ult["firma"]):
            cm = 0
        nuovo = {"passo": t, "cursore_base": cb, "cursore_mc": cm, "righe_mc": M, "firma": i_f}
        if ca.righe_mc_permutate:
            nuovo["permutate"] = True
        if perm_prima != ca.righe_mc_permutate:
            log(f"righe mc alla ripresa (passo {t}): ordine {'PERMUTATO' if ca.righe_mc_permutate else 'del teacher'} "
                f"dal cursore mc {cm} (decisione: righe gia' viste nel giro possono tornare prima della fine del giro)")
        if int(ult["passo"]) == t:
            seg[-1] = nuovo
        else:
            seg.append(nuovo)
        log(f"righe mc alla ripresa (passo {t}): {ult['righe_mc']} -> {M} a passo" +
            (f", teacher mc NUOVO ({firma.get('teacher_mc')}), cursore mc da 0" if i_f != int(ult["firma"]) else "") +
            f"; cursore base {cb}, cursore mc {cm} (decisione, da qui in avanti)")
    return seg


def righe_rapide(held: list, n: int) -> list:
    """27/9: le n righe della misura rapida, equispaziate nel held-out (i domini sono a blocchi: tutti presenti)."""
    if n <= 0 or n >= len(held):
        return list(held)
    return [held[(i * len(held)) // n] for i in range(n)]


def ricetta_cambiabile(prima: Optional[str], ora) -> bool:
    """27/9 integrata: la ricetta della ripresa `prima` (nome) si puo' cambiare in `ora` (Ricetta) alla ripresa se le due
    differiscono SOLO nell'aritmetica delle proiezioni non-esperte (non_esperti_x3 / non_esperti_k2): stesso residuo,
    stessi prodotti ed stesso indietro degli esperti (quindi la K di G1 resta valida), stessa rotazione, stesso router."""
    import e2e_studente as S
    if prima is None or prima not in S.RICETTE or ora is None:
        return False
    a, b = dataclasses.asdict(S.RICETTE[prima]), dataclasses.asdict(ora)
    for k in ("nome", "non_esperti_x3", "non_esperti_k2"):
        a.pop(k, None)
        b.pop(k, None)
    return a == b


def lora_scartata(lora_rip: dict, st) -> dict:
    """27/9 integrata (--lora-reinit-ga): quanto vale la LoRA della ripresa che si scarta, |s B A|_F per adattatore
    (fp64, con |B A|_F^2 = tr((B^T B)(A A^T)): matrici r x r, niente Delta W pieno), sommata per famiglia."""
    per_fam: dict = {}
    tot = 0.0
    for n, v in lora_rip.items():
        a = st.lora[n]
        A_ = torch.as_tensor(v["A"]).to(torch.float64)
        B_ = torch.as_tensor(v["B"]).to(torch.float64)
        q = float(torch.trace((B_.T @ B_) @ (A_ @ A_.T))) * a.scala ** 2
        per_fam[a.famiglia] = per_fam.get(a.famiglia, 0.0) + q
        tot += q
    return {"norma_totale": math.sqrt(max(tot, 0.0)),
            "norma_per_famiglia": {f: math.sqrt(max(x, 0.0)) for f, x in sorted(per_fam.items())}}


def modo_ripristino_lora(ca: ConfigAllena, cfg_prima: dict) -> Optional[str]:
    """28/9 --lora-ripristina: il ripristino da fare a QUESTA ripresa = il modo della riga se e' cambiato rispetto alla
    config della ripresa (None altrimenti, o se vuoto). Idempotente: rilanciata con la stessa riga, la config della ripresa
    ha gia' quel modo (scritto dalla prima ripresa dopo il ripristino) e non si rifa' nulla; una ripresa fallita PRIMA di
    salvare lo rifa' identico (deterministico)."""
    m = ca.lora_ripristina
    return m if m and m != (cfg_prima.get("lora_ripristina") or "") else None


def ripristino_lora(st, lora_rip: dict, modo: str, passo: int, ca: ConfigAllena, log: Callable) -> dict:
    """28/9 --lora-ripristina, alla RIPRESA: |Delta W|_F della LoRA della ripresa che si scarta, per adattatore (fp64 sulla
    CPU, `e2e_lora.norma_delta`: s B A, o s (B A - B0 A0) nella forma GA), per famiglia e totale: la voce di
    stato["lora_ripristini"]. Scrive nel registro il riassunto e una riga per strato. Non tocca lo studente."""
    import e2e_lora as LR
    per: dict = {}
    per_fam: dict = {}
    for n, v in lora_rip.items():
        a = st.lora[n]
        x = LR.norma_delta(v["A"], v["B"], a.scala, v.get("A0"), v.get("B0"))
        per[n] = x
        per_fam[a.famiglia] = per_fam.get(a.famiglia, 0.0) + x * x
    tot = math.sqrt(sum(x * x for x in per.values()))
    # forma per adattatore: "ga" (A0, B0) o "zero"; "mista" quando convivono (--lora-reinit-ga con strati congelati)
    n_ga = sum(1 for v in lora_rip.values() if "A0" in v)
    forma = "ga" if n_ga == len(lora_rip) else ("zero" if n_ga == 0 else "mista")
    voce = {"passo": int(passo), "modo": modo, "forma": forma, "adattatori": len(per), "adattatori_ga": n_ga,
            "norma_totale": tot, "norma_per_famiglia": {f: math.sqrt(q) for f, q in sorted(per_fam.items())},
            "per_adattatore": per}
    n_max = max(per, key=per.get) if per else None
    come = {"ga": "A = A0, B = B0", "zero": "B = 0, A tenuto",
            "mista": f"B = 0, A tenuto sui {len(per) - n_ga} di forma zero; A = A0, B = B0 sui {n_ga} in forma GA"}[forma]
    cosa = ({"zero": come + f"; Adam della LoRA da zero, la LoRA CONTINUA ad allenarsi (lr_lora {ca.lr_lora:g})",
             "ferma": "adattatori al valore di costruzione (B = 0); LoRA FUORI dall'ottimizzatore, niente gradiente, "
                      "niente A/B in riprese e istantanee"}[modo])
    log(f"ripresa: --lora-ripristina {modo} al passo {passo} (decisione, da qui in avanti): Delta W = 0 esatto su "
        f"{len(per)} adattatori ({cosa}); |Delta W|_F scartato totale {tot:.4e}, per famiglia " +
        ", ".join(f"{f} {x:.3e}" for f, x in voce["norma_per_famiglia"].items()) +
        (f"; massimo {n_max} {per[n_max]:.3e}" if n_max else ""))
    per_strato: dict = {}
    for n, x in per.items():
        per_strato.setdefault(st.lora[n].L, []).append(f"{st.lora[n].famiglia} {x:.3e}")
    for L in sorted(per_strato):
        log(f"  |Delta W| scartato strato {L}: " + ", ".join(per_strato[L]))
    return voce


def _senza_gruppo_lora(sd: dict, n_param: int) -> dict:
    """28/9 --lora-ripristina: lo stato dell'ottimizzatore della ripresa SENZA il gruppo della LoRA (l'ultimo: theta, F32,
    LoRA; con Sophia l'ultimo del suo AdamW, che diventa None se restava solo lui). Il resto si carica poi come da una
    ripresa senza LoRA: theta per indice, F32 per NOME (`carica_stato_ottimizzatore`, anche con gli indici spostati da
    --f32-ripristina); A e B ripartono da Adam vuoto (zero) o non sono nell'ottimizzatore (ferma)."""
    def togli(a: dict) -> dict:
        g = a["param_groups"][-1]
        if len(g["params"]) != n_param:
            raise RuntimeError(f"ripresa: l'ultimo gruppo dell'ottimizzatore ha {len(g['params'])} parametri, la LoRA "
                               f"della ripresa {n_param}: stato non riconosciuto")
        via = set(g["params"])
        return {"state": {i: v for i, v in a["state"].items() if i not in via},
                "param_groups": list(a["param_groups"][:-1])}
    if _e_stato_sophia(sd):
        if sd.get("adam") is None:
            raise RuntimeError("ripresa: LoRA allenata ma nessun gruppo AdamW nello stato di Sophia")
        a = togli(sd["adam"])
        return {**sd, "adam": a if a["param_groups"] else None}
    return togli(sd)


def _riprendi_da_ema(st, opt, rip: dict, stato: dict, log: Callable) -> None:
    """29/9 ramo tt3-cure (--riprendi-da-ema, solo con --riprendi). Causa (TT3 passo 2): held-out grezzo 0,50574, theta_EMA
    0,43636 (meglio del passo 0, 0,45982): il passo intero di Adam e' ~7 volte quello utile (rumore del gradiente), la
    EMA no. Alla ripresa:
      - theta := theta_EMA salvata nella ripresa (copia al bit); la EMA resta quella (theta e EMA coincidono);
      - F32 e LoRA: la EMA non li ha (e' solo di theta), restano quelli GREZZI della ripresa. E' esattamente il modello
        che il held-out «theta_EMA» ha misurato (`theta_scambiato` scambia la sola theta);
      - Adam: momenti di theta AZZERATI (lo stato per parametro si toglie: primo e secondo momento e contatore del passo
        ripartono da zero, con la correzione del bias come al passo 0; i momenti erano del punto grezzo, non della EMA).
        Momenti di F32 e LoRA INVARIATI (i loro valori non cambiano). Codici, registri PV, stato V-Bop invariati.
    Una volta sola per corsa: se lo stato ha gia' la voce `ripresa_da_ema`, la richiesta si ignora (un riavvio
    automatico dell'unita' con la stessa riga di comando non rifa' lo scambio su una ripresa successiva)."""
    fatte = stato.get("ripresa_da_ema") or []
    if fatte:
        log(f"--riprendi-da-ema IGNORATO: ripresa dalla EMA gia' fatta al passo {fatte[-1]['passo']} (una volta per corsa)")
        return
    if rip.get("ema") is None:
        raise RuntimeError("--riprendi-da-ema: la ripresa non ha theta_EMA (corsa senza --ema-theta)")
    diff = 0.0
    with torch.no_grad():
        for k, p in zip(st.chiavi(), st.parametri()):
            e = rip["ema"][_chiave(k)].to(device=p.device, dtype=p.dtype)
            diff = max(diff, float((p - e).abs().max()))
            p.copy_(e)
    if getattr(opt, "tipo", None) == "sophia":        # theta vive in SophiaTheta: non previsto (TT3 usa adamw)
        raise RuntimeError("--riprendi-da-ema con --ottimizzatore-theta sophia: non previsto")
    st_adam = opt.state
    azzerati = sum(1 for p in st.parametri() if st_adam.pop(p, None) is not None)
    voce = {"passo": int(stato["passo"]), "max_diff_theta": diff, "adam_theta": "azzerato",
            "parametri_azzerati": azzerati, "f32_lora": "grezzi (la EMA e' solo di theta), Adam invariato"}
    stato["ripresa_da_ema"] = fatte + [voce]
    log(f"ripresa DALLA EMA al passo {voce['passo']}: theta := theta_EMA (max |theta - EMA| prima {diff:.4e}), momenti "
        f"di Adam di theta azzerati ({azzerati} parametri), F32/LoRA grezzi della ripresa con il loro Adam")


def allena(st, src, dir_c: Path, ca: ConfigAllena, WS, WT, riprendi: bool = False,
           ferma_a: Optional[int] = None, log: Callable = print, riusa_passo0: Optional[str] = None,
           reinit_lora_ga: bool = False, da_ema: bool = False, cambia_ricetta: bool = False) -> dict:
    """Il ciclo completo. Il passo 0 passa dal cancello G1 (e usa il gradiente che G1 lascia, gia' con la K
    scelta). Ritorna lo stato finale (lo stesso che si salva nelle riprese).
    27/9 integrata `reinit_lora_ga` (CLI --lora-reinit-ga, solo con --riprendi e --lora-init ga): se la LoRA della
    ripresa e' a init zero la si scarta e la si re-inizializza con LoRA-GA (vedi sopra); gia' GA = nessun effetto
    (idempotente: la stessa riga di lancio si puo' riusare alle riprese successive)."""
    con_mc = isinstance(src, SorgenteConMC)             # 27/9 trainer-mc
    righe_mc_src = src.righe_mc() if con_mc else []
    if ca.righe_mc > 0 and not righe_mc_src:
        raise ValueError(f"righe_mc {ca.righe_mc} senza righe mc nella sorgente (--teacher-mc con il lato train)")
    dir_c.mkdir(parents=True, exist_ok=True)
    car = Caricatore(src, st.device)
    scrittore = ScrittoreAsincrono(log) if ca.scritture_asincrone else None   # 1/10 --scritture-asincrone
    train = src.righe_previste("train")
    held = src.righe_previste("heldout")
    passi_epoca = len(train) // ca.righe_per_passo
    if passi_epoca == 0:
        raise ValueError(f"{len(train)} righe di addestramento < {ca.righe_per_passo} a passo")
    tot = passi_epoca * ca.epoche
    if tuple(getattr(st, "allena_f32", ())) != tuple(ca.allena_f32):
        raise ValueError(f"studente con allena_f32 {getattr(st, 'allena_f32', ())}, config {ca.allena_f32}")
    if tuple(getattr(st, "densi_ternari", ())) != tuple(ca.densi_ternari):      # 28/9 --densi-ternari
        raise ValueError(f"studente con densi_ternari {getattr(st, 'densi_ternari', ())}, config {ca.densi_ternari}")
    if firma_lora_studente(st) != firma_lora_config(ca):
        raise ValueError(f"studente con LoRA {firma_lora_studente(st)}, config {firma_lora_config(ca)}")
    # 27/9 dominio-strati: strati congelati (vuoto = nessuno; tutti gli strati elencati = nessuno, come congela_strati)
    strati_fermi: set = set()
    if ca.strati_allenati:
        if not hasattr(st, "congela_strati"):
            raise ValueError("--strati-allenati: lo studente non sa congelare gli strati (congela_strati)")
        fuori = sorted(set(ca.strati_allenati) - set(st.strati))
        if fuori:
            raise ValueError(f"--strati-allenati {fuori}: strati non caricati nello studente ({st.strati[0]}..{st.strati[-1]})")
        strati_fermi = set(st.strati) - set(ca.strati_allenati)
    congela = bool(strati_fermi)
    if da_ema and not riprendi:
        raise ValueError("--riprendi-da-ema solo con --riprendi (una corsa nuova non ha una EMA da cui ripartire)")
    lora_adam_da_zero: list = []                    # dominio-strati + --lora-reinit-ga: Adam da zero SOLO per questi
    if getattr(st, "lora", None):                   # 28/9 --lora-ripristina ferma: uno STATO finche' la riga lo dice
        st.lora_ferma = ca.lora_ripristina == "ferma"
        if st.lora_ferma:
            _blocca_lora(st)
            for a in st.lora.values():              # studente appena costruito: B = 0 gia' (no-op); per sicurezza
                a.azzera_delta()
            log(f"LoRA FERMA (--lora-ripristina ferma): {len(st.lora)} adattatori installati con Delta W = 0 esatto "
                f"(l'avanti e' quello senza LoRA; i valori della ripresa NON si caricano), fuori dall'ottimizzatore, niente "
                f"gradiente ne' LoRA-GA, niente A/B in riprese e istantanee (export = senza LoRA)")
    rip = carica_ripresa(dir_c, st.device) if riprendi else None
    if rip is None and ca.f32_ripristina:
        log(f"--f32-ripristina {','.join(ca.f32_ripristina)} senza effetto: corsa nuova, i F32 partono gia' dal GGUF")
    if rip is None and ca.lora_ripristina == "zero":
        log("--lora-ripristina zero senza effetto: corsa nuova, la LoRA parte gia' da Delta W = 0")
    ripreso = rip is not None
    impr = impronte_byte(st)
    if rip is not None:
        stato = rip["stato"]
        ora_cfg = json.loads(json.dumps(asdict(ca), default=str))
        # un campo che la ripresa non conosce (nato dopo) vale il suo default
        diversi = {k for k in set(stato["config"]) | set(ora_cfg)
                   if stato["config"].get(k, _CONFIG_DEFAULT.get(k)) != ora_cfg.get(k)}
        ammessi = _CAMPI_DI_ESECUZIONE | _CAMPI_DI_DECISIONE | (_CAMPI_SOLO_G1 if stato.get("g1") else set())
        if diversi - ammessi:
            raise RuntimeError(f"la ripresa ha una configurazione diversa ({sorted(diversi)}): nuova corsa con un altro nome")
        cp = stato["config"]
        lora_cfg_prima = (int(cp.get("lora_rango", 0) or 0), cp.get("lora_alpha"), tuple(cp.get("lora_famiglie") or ()),
                          tuple(cp.get("lora_tipi") or ()))
        if lora_cfg_prima[0] > 0 and lora_cfg_prima != firma_lora_config(ca):
            raise RuntimeError(f"ripresa: LoRA gia' allenata {lora_cfg_prima}, ora {firma_lora_config(ca)} (spegnerla o "
                               f"cambiarne rango/alpha/famiglie cambia il modello): nuova corsa con un altro nome")
        lota_prima = (int(cp.get("lota_rango", 0) or 0), cp.get("lota_omega"))
        if lota_prima[0] > 0 and (stato.get("lota") or {}).get("file") and lota_prima != (ca.lota_rango, ca.lota_omega):
            raise RuntimeError(f"ripresa: LoTA gia' allenata (rango, omega) {lota_prima}, ora {(ca.lota_rango, ca.lota_omega)} "
                               "(spegnerla o cambiarne rango/omega cambia i codici): nuova corsa con un altro nome")
        if diversi:
            log(f"ripresa con campi cambiati (esecuzione: stessa matematica; decisione: da qui in avanti): " +
                ", ".join(f"{k} {stato['config'].get(k)} -> {ora_cfg.get(k)}" for k in sorted(diversi)))
            if "pv_rho" in diversi:
                stato["pv_rho_deciso"] = True           # la regione di fiducia riparte dal rho DECISO, non da quello salvato
            stato["config"] = ora_cfg
        if json.loads(json.dumps(stato["sorgente"], default=str)) != json.loads(json.dumps(src.firma(), default=str)):
            raise RuntimeError("la ripresa viene da un altro teacher (piano/corpus/modello diversi)")
        ricetta_cambiata = False
        if stato.get("studente") != _firma_studente(st):
            prima_f, ora_f = dict(stato.get("studente") or {}), _firma_studente(st)
            r_prima = prima_f.pop("ricetta", None)
            if {k: v for k, v in ora_f.items() if k != "ricetta"} == prima_f and ricetta_cambiabile(r_prima, st.ric):
                # 27/9 integrata (cancello bf16x3 v2 AMMESSO): cambiare ricetta alla ripresa e' una DECISIONE solo se cambia
                # l'aritmetica delle proiezioni non-esperte (B_ind32 <-> B3_ind32): stesso modello, stesso indietro degli
                # esperti (la K di G1 resta valida), stessi parametri
                log(f"ripresa: ricetta {r_prima} -> {st.ric.nome} (decisione, da qui in avanti: stesso modello e stesso "
                    f"indietro degli esperti, cambia l'aritmetica delle proiezioni non-esperte)")
                stato.setdefault("ricette_cambiate", []).append({"passo": stato["passo"], "da": r_prima,
                                                                 "a": st.ric.nome})
                stato["studente"] = ora_f
            elif cambia_ricetta and ricetta_sostituibile(stato.get("studente"), ora_f):
                # 1/10 TT5-ottimizzato (porto di indietro-bf16x3 22321a9): --cambia-ricetta, stesso studente e stesso
                # formato (residuo, gemm, rotazione, router), cambia l'aritmetica dell'indietro (B3_ind32 -> B3_ind2)
                ricetta_cambiata = True
                vecchia = stato["studente"]["ricetta"]
                stato.setdefault("cambi_ricetta", []).append({"passo": stato.get("passo"), "da": vecchia,
                                                              "a": st.ric.nome, "data": adesso()})
                stato["studente"] = ora_f
                log(f"ripresa: ricetta {vecchia} -> {st.ric.nome} (--cambia-ricetta: stesso GGUF, strati ed esperti, "
                    f"stesso formato di residuo/esperti; cambia solo l'aritmetica)")
            else:
                raise RuntimeError(f"la ripresa viene da un altro studente: {stato.get('studente')} != {ora_f}")
        esc = (stato.get("g1") or {}).get("escalation")
        if esc and st.ric.nome != esc and not ricetta_cambiata:
            imposta_ricetta(st, ricetta_indietro_fp32(st.ric))
            if st.ric.nome != esc:
                raise RuntimeError(f"ripresa: G1 era passato a {esc}, lo studente e' {st.ric.nome}")
            log(f"ripresa: riapplico l'indietro fp32 scelto da G1 ({esc})")
        with torch.no_grad():
            for k, p in zip(st.chiavi(), st.parametri()):
                p.copy_(rip["theta"][_chiave(k)])
        nomi_prima = list(rip.get("f32_nomi", []))
        rip_f32 = {"tolti_ammessi": [], "adam_da_zero": []}
        if nomi_prima:
            # 28/9 --f32-ripristina: le famiglie riportate al GGUF non ricevono il valore della ripresa (vuota = al bit)
            rip_f32 = ripristino_f32(st, nomi_prima, rip["f32"], ca, cp, stato["passo"], log)
            if not _parametri_f32(st) and len(rip_f32["tolti_ammessi"]) < len(nomi_prima):
                raise RuntimeError(f"ripresa: {len(nomi_prima)} tensori F32 allenati, studente senza --allena-f32")
            if _parametri_f32(st):
                st.imposta_f32(rip_f32["valori"])
                if rip_f32["adam_da_zero"]:
                    st.imposta_f32(st.valori_f32_gguf(rip_f32["adam_da_zero"]))
            if rip_f32["voce"] is not None:
                stato.setdefault("f32_ripristini", []).append(rip_f32["voce"])
        lora_prima = list(rip.get("lora_nomi", []))
        sd_opt = rip["opt"]                             # 28/9 --lora-ripristina: senza il gruppo LoRA (`_senza_gruppo_lora`)
        modo_lr = modo_ripristino_lora(ca, cp)          # 28/9 None = nessun ripristino della LoRA a questa ripresa
        if lora_prima:                                  # --lora-nonesperti: A e B della ripresa, stessi adattatori
            if not getattr(st, "lora", None):           # (28/9: con --lora-ripristina ferma `_parametri_lora` e' [])
                raise RuntimeError(f"ripresa: {len(lora_prima)} adattatori LoRA allenati, studente senza LoRA")
            if lora_prima != st.chiavi_lora():
                raise RuntimeError("ripresa: adattatori LoRA diversi da quelli allenati (strati/famiglie): nuova corsa")
            ga_prima = any("A0" in v for v in rip["lora"].values())      # 27/9 LoRA-GA: A0/B0 nella ripresa
            if modo_lr is not None:
                # 28/9 --lora-ripristina zero|ferma: Delta W della ripresa SCARTATO (norme registrate) su TUTTI gli
                # adattatori, anche degli strati congelati; Adam della LoRA buttato. zero: A/B della ripresa e poi
                # `azzera_delta` (la LoRA continua); ferma: A/B NON caricati (restano con Delta W = 0), fuori dall'ottimizzatore
                if reinit_lora_ga and not ga_prima and ca.lora_init == "ga":
                    raise RuntimeError(f"ripresa: --lora-ripristina {modo_lr} e --lora-reinit-ga insieme su una LoRA a init "
                                       "zero: entrambi scartano la LoRA della ripresa, sceglierne uno")
                if modo_lr == "zero" and ga_prima != (ca.lora_init == "ga"):
                    raise RuntimeError(f"ripresa: LoRA gia' allenata con init {'ga' if ga_prima else 'zero'}, ora "
                                       f"--lora-init {ca.lora_init}: con --lora-ripristina zero la LoRA resta nella sua forma")
                if reinit_lora_ga:
                    log("ripresa: --lora-reinit-ga senza effetto (" + ("LoRA gia' in forma GA" if ga_prima else
                                                                        "--lora-init non e' ga") + ")")
                stato.setdefault("lora_ripristini", []).append(ripristino_lora(st, rip["lora"], modo_lr, stato["passo"],
                                                                               ca, log))
                if modo_lr == "zero":
                    st.imposta_lora(rip["lora"])
                    for a in st.lora.values():
                        a.azzera_delta()
                sd_opt = _senza_gruppo_lora(sd_opt, 2 * len(lora_prima))
                lora_prima = []                         # per l'ottimizzatore: LoRA della ripresa senza stato di Adam
            elif reinit_lora_ga and not ga_prima and ca.lora_init == "ga" and congela:
                # 27/9 dominio-strati + --lora-reinit-ga: si scarta (e si re-inizializza GA, piu' sotto) SOLO la LoRA degli
                # strati ALLENATI; negli strati congelati la LoRA della ripresa resta com'e', col suo stato di Adam (al
                # bit). Adam del gruppo LoRA si carica intero e poi si azzera per i soli adattatori scartati.
                ferme = {n: v for n, v in rip["lora"].items() if st.lora[n].L in strati_fermi}
                scartata = lora_scartata({n: v for n, v in rip["lora"].items() if n not in ferme}, st)
                st.imposta_lora(ferme)
                lora_adam_da_zero = [p for n, a in st.lora.items() if n not in ferme for p in a.parametri()]
                stato["lora_reinit_ga"] = {"passo": stato["passo"], "adattatori": len(lora_prima) - len(ferme),
                                           "conservati_strati_congelati": len(ferme), **scartata}
                log(f"ripresa: --lora-reinit-ga con strati congelati: LoRA della ripresa SCARTATA nei soli strati allenati "
                    f"({len(lora_prima) - len(ferme)} adattatori, |s B A|_F {scartata['norma_totale']:.4e}), CONSERVATA al "
                    f"bit (valori e Adam) nei {len(ferme)} degli strati congelati; LoRA-GA prima del prossimo passo")
            elif reinit_lora_ga and not ga_prima and ca.lora_init == "ga":
                # 27/9 integrata --lora-reinit-ga: la LoRA della ripresa (init zero, gia' allenata) si SCARTA e si
                # re-inizializza con LoRA-GA sulle righe del passo della ripresa (piu' sotto, come «accesa ora»); theta,
                # codici, F32, Adam di theta/F32, V-Bop, EMA restano quelli della ripresa, Adam della LoRA riparte da zero
                scartata = lora_scartata(rip["lora"], st)
                stato["lora_reinit_ga"] = {"passo": stato["passo"], "adattatori": len(lora_prima), **scartata}
                log(f"ripresa: --lora-reinit-ga: LoRA della ripresa (init zero, {len(lora_prima)} adattatori) SCARTATA "
                    f"(|s B A|_F totale {scartata['norma_totale']:.4e}; per famiglia " +
                    ", ".join(f"{f} {x:.3e}" for f, x in scartata["norma_per_famiglia"].items()) +
                    "): LoRA-GA prima del prossimo passo, Adam da zero per A e B; theta, codici, F32, V-Bop, EMA e "
                    "Adam di theta/F32 invariati")
                lora_prima = []                         # per l'ottimizzatore: LoRA «accesa ora»
            else:
                if reinit_lora_ga:
                    log("ripresa: --lora-reinit-ga senza effetto (" + ("LoRA gia' in forma GA" if ga_prima else
                                                                        "--lora-init non e' ga") + ")")
                if ga_prima != (ca.lora_init == "ga"):
                    raise RuntimeError(f"ripresa: LoRA gia' allenata con init {'ga' if ga_prima else 'zero'}, ora "
                                       f"--lora-init {ca.lora_init}: nuova corsa con un altro nome (o, da zero a ga, "
                                       f"--lora-reinit-ga)")
                st.imposta_lora(rip["lora"])
        elif _parametri_lora(st):
            log(f"ripresa: LoRA accesa ora (decisione): {len(st.chiavi_lora())} adattatori, " +
                ("LoRA-GA prima del prossimo passo (modello invariato al bit)" if ca.lora_init == "ga"
                 else "B = 0 (modello invariato)") + ", Adam da zero per A e B" +
                (f" (--lora-ripristina {modo_lr}: nessuna LoRA allenata nella ripresa, nulla da scartare)"
                 if modo_lr else ""))
        elif modo_lr is not None:                       # 28/9 ferma e la ripresa non ha LoRA allenata: nulla da scartare
            log(f"ripresa: --lora-ripristina {modo_lr}: nessuna LoRA allenata nella ripresa, nulla da scartare")
        opt = nuovo_ottimizzatore(st, ca, stato["g1"]["eps_adam"], K=stato["g1"]["K"])
        if getattr(opt, "tipo", None) == "sophia" or _e_stato_sophia(sd_opt):   # 27/9 newton-theta
            carica_ottimizzatore_sophia(opt, sd_opt, st, nomi_prima, lora_prima, log,
                                        **_kw_tolti(rip_f32["tolti_ammessi"]))
        elif nomi_prima or _parametri_f32(st) or lora_prima or _parametri_lora(st):
            carica_stato_ottimizzatore(opt, sd_opt, len(st.parametri()), nomi_prima,
                                       st.chiavi_f32() if _parametri_f32(st) else [], log,
                                       lora_prima=_nomi_param_lora(lora_prima),
                                       lora_ora=_nomi_param_lora(st.chiavi_lora()) if _parametri_lora(st) else [],
                                       **_kw_tolti(rip_f32["tolti_ammessi"]))
        else:
            opt.load_state_dict(sd_opt)
        if rip_f32["adam_da_zero"]:                     # 28/9 --f32-ripristina: riportati al GGUF e ancora allenati
            st_adam = opt.adam.state if getattr(opt, "tipo", None) == "sophia" else opt.state
            for n in rip_f32["adam_da_zero"]:
                st_adam.pop(st.f32[n].master, None)
        if lora_adam_da_zero:                           # 27/9 dominio-strati: adattatori scartati = Adam da zero
            st_adam = opt.adam.state if getattr(opt, "tipo", None) == "sophia" else opt.state
            for p in lora_adam_da_zero:
                st_adam.pop(p, None)
        for i, g in enumerate(opt.param_groups):        # il lr della ripresa e' quello della config di ORA
            lr_i = lr_del_gruppo(ca, i)
            if g["lr"] != lr_i:
                log(f"ripresa: lr{_suffisso_gruppo(ca, i)} {g['lr']:g} -> {lr_i:g} (decisione, da qui in avanti)")
                g["lr"] = lr_i
            b_i = betas_del_gruppo(ca, i)
            if tuple(g["betas"]) != b_i:   # 27/9: anche beta2, su OGNI gruppo (theta, F32, LoRA)
                log(f"ripresa: betas{_suffisso_gruppo(ca, i)} {tuple(g['betas'])} -> {b_i} "
                    f"(decisione, da qui in avanti)")
                g["betas"] = b_i
            if "gamma" in g and g["gamma"] != ca.sophia_gamma:          # 27/9 newton-theta: gamma di Sophia
                log(f"ripresa: sophia_gamma {g['gamma']:g} -> {ca.sophia_gamma:g} (decisione, da qui in avanti)")
                g["gamma"] = ca.sophia_gamma
        controlla_ottimizzatore(opt, ca, stato["g1"]["eps_adam"], stato["g1"]["K"])
        if da_ema:                                      # 29/9 tt3-cure --riprendi-da-ema (una volta sola per corsa)
            _riprendi_da_ema(st, opt, rip, stato, log)
        tieni_ist = {stato["migliore"]["dir"]} | _citate(_indice_riprese(dir_c))
        if stato.get("migliore_ema"):
            tieni_ist.add(stato["migliore_ema"]["dir"])
        pota_istantanee(dir_c, tieni_ist)
        log(f"ripresa dal passo {stato['passo']} (migliore {stato['migliore']['passo']})")
        ema = None
        if ca.ema_theta > 0:
            if rip.get("ema") is not None:
                ema = [rip["ema"][_chiave(k)].clone() for k in st.chiavi()]
            else:
                ema = [p.detach().clone() for p in st.parametri()]
                log(f"media esponenziale di theta accesa alla ripresa (decadimento {ca.ema_theta:g}): parte da theta "
                    f"del passo {stato['passo']}")
        del rip
    else:
        stato = {"versione": VERSIONE, "config": json.loads(json.dumps(asdict(ca), default=str)), "passo": 0,
                 "storia": [], "storia_heldout": [], "migliore": None, "g1": None, "primo_passo": None,
                 "cambi_instradamento": 0, "fermato": None, "consumati": [], "attesa_teacher_s": 0.0,
                 "sorgente": src.firma(), "studente": _firma_studente(st), "passi_totali": tot, "kl_intervallo": []}
        if any(t.abs().max() != 0 for t in st.parametri()):
            raise RuntimeError("theta non e' zero all'inizio di una corsa nuova")
        opt = None
        ema = [p.detach().clone() for p in st.parametri()] if ca.ema_theta > 0 else None
    # 27/9 trainer-mc: righe mc nel passo (segmenti del piano, nello stato) e held-out mc; senza teacher mc e con
    # righe_mc = 0 il piano e' quello di sempre (`PianoRighe` classico)
    held_mc = src.righe_previste(LATO_HELD_MC) if con_mc else []
    piano = PianoRighe(train, ca, righe_mc_src, _segmenti_mc(stato, ca, src if con_mc else None, train, log))
    if not piano.classico:
        tot = piano.totale()
        if stato.get("passi_totali") != tot:
            log(f"piano con righe mc: passi totali {stato.get('passi_totali')} -> {tot} (passo {stato['passo']}, cursori "
                f"{piano.cursori(stato['passo'])[:2]})")
            stato["passi_totali"] = tot
    if held_mc:                                         # il held-out mc si legge senza attese: deve essere tutto sigillato
        stato["attesa_teacher_s"] = stato.get("attesa_teacher_s", 0.0) + attendi_righe(src, LATO_HELD_MC, held_mc, log,
                                                                                       ca.attesa_ogni_s)
    if congela:                                  # 27/9 dominio-strati: da qui (G1, LoRA-GA, passi) solo gli strati allenati
        st.congela_strati(ca.strati_allenati)
        if getattr(st, "lora_ferma", False):     # 28/9 --lora-ripristina ferma: congela_strati riaccende A/B degli allenati
            _blocca_lora(st)
        n_th = sum(int(p.numel()) for (L, _), p in zip(st.chiavi(), st.parametri()) if _allenata(st, L))
        n_f = sum(1 for t in getattr(st, "f32", {}).values() if t.master.requires_grad)
        n_l = sum(1 for a in getattr(st, "lora", {}).values() if a.A.requires_grad)
        L0 = min(ca.strati_allenati)
        log(f"strati allenati {_testo_strati(ca.strati_allenati)} ({len(ca.strati_allenati)} di {len(st.strati)}): scale "
            f"{n_th} ({100 * n_th / max(1, sum(int(p.numel()) for p in st.parametri())):.1f}%), tensori F32 {n_f} di "
            f"{len(getattr(st, 'f32', {}))}, adattatori LoRA {n_l} di {len(getattr(st, 'lora', {}))}; negli altri "
            f"strati nessun gradiente (stato di Adam/Sophia, V-Bop, EMA al bit), indietro fermo allo strato {L0}")
    pv = reg_f = None
    if ca.pv_codici:
        import e2e_pv_codici as PV
        bop = (PV.ConfigBop(gamma=ca.pv_gamma, curvatura_ogni=ca.pv_curvatura_ogni) if ca.pv_modo == "vbop"
               else None)
        acc_v = _finestra_statov(ca)                      # 1/10: vbop -> 1 (pv_passi_accumulo = decisione ogni N passi)
        stato_v = (PV.StatoV(ca.pv_n_proposta, acc_v) if bop is None
                   else PV.StatoV(ca.pv_n_proposta, acc_v, bop=bop))
        if ca.ottimizzatore_theta == "sophia":           # 27/9 newton-theta: HT nei buffer (curvatura delle scale)
            stato_v = PV.StatoV(ca.pv_n_proposta, acc_v, bop=bop, curv_theta=True)
        fams_v = PV.installa(st, stato_v)
        if st.device.type == "cpu":                       # su CPU i byte possono essere il memmap in sola lettura
            for f in fams_v.values():
                PV.famiglia_scrivibile(f)
        if congela:                                       # 27/9 dominio-strati: il passo V decide solo sugli allenati
            stato_v.attive = {k for k in fams_v if k[0] not in strati_fermi}
            log(f"fase 2 PV: salti e stato V-Bop solo sulle {len(stato_v.attive)} famiglie degli strati allenati (le "
                f"{len(fams_v) - len(stato_v.attive)} degli strati congelati: codici e stato V-Bop al bit)")
        cfg_v = (PV.ConfigV(q_fdr=ca.pv_q_fdr, n_proposta=ca.pv_n_proposta, q_guadagno=ca.pv_q_guadagno,
                            passi_accumulo=acc_v) if ca.pv_accetta == "bh"
                 else PV.ConfigV(q_fdr=ca.pv_q_fdr, n_proposta=ca.pv_n_proposta, q_guadagno=ca.pv_q_guadagno,
                                 passi_accumulo=acc_v, accetta=ca.pv_accetta, z_min=ca.pv_z_min,
                                 z_guadagno_min=ca.pv_z_guadagno_min))
        pv = PV.PassoV(fams_v, cfg_v, stato_v)
        info = stato.setdefault("pv", {"registri": 0, "file": None, "sha": None, "rho": ca.pv_rho, "tarato": False,
                                       "salti_tenuti": 0, "passi_v": 0})
        reg_f = (PV.RegioneFiducia(prova=(ca.pv_rho,)) if not ca.pv_rho_adattivo
                 else PV.RegioneFiducia(prova=(ca.pv_rho,), adattivo=True, rho_max=ca.pv_rho_max))
        if info.get("tarato"):
            reg_f.storia = [{"ripresa": True}]
            reg_f.rho = info["rho"]
            if stato.pop("pv_rho_deciso", False):
                reg_f.rho = info["rho"] = ca.pv_rho
                log(f"fase 2 PV: rho deciso alla ripresa {ca.pv_rho:g} (al posto di quello tarato)")
        if info.get("registri"):
            regs = PV.Registro.carica_lista(dir_c / info["file"], info["sha"])[:info["registri"]]
            PV.rigioca(pv.fam, regs)
            pv.registri = list(regs)
            log(f"fase 2 PV: rigiocati {len(regs)} registri ({sum(len(r) for r in regs)} salti) sui byte dello studente")
        acc = info.get("accumulo")
        if acc:
            if acc_v > 1 and int(acc.get("passi_accumulo", 0)) == acc_v:
                p_acc = dir_c / acc["file"]
                if sha256_file(p_acc) != acc["sha"]:
                    raise RuntimeError(f"{p_acc}: sha256 della finestra PV diverso dalla ripresa")
                stato_v.ripristina(torch.load(p_acc, map_location="cpu", weights_only=True))
                log(f"fase 2 PV: finestra di accumulo ripresa ({stato_v.passi_nella_finestra}/{ca.pv_passi_accumulo} "
                    f"passi, {stato_v.n_verifica} micro-lotti di verifica)")
            else:
                log(f"fase 2 PV: finestra salvata con accumulo {acc.get('passi_accumulo')}, ora {ca.pv_passi_accumulo}: "
                    f"si riparte da una finestra nuova")
        sb = info.get("bop")
        if bop is not None and sb:
            p_b = dir_c / sb["file"]
            if sha256_file(p_b) != sb["sha"]:
                raise RuntimeError(f"{p_b}: sha256 dello stato V-Bop diverso dalla ripresa")
            ist_b = torch.load(p_b, map_location="cpu", weights_only=True)
            stato_v.ripristina_bop(ist_b)
            log(f"fase 2 PV: stato V-Bop ripreso ({stato_v.passi_bop} passi V" +
                (f"; gamma salvato {ist_b['gamma']:g}, ora {bop.gamma:g}: decisione, da qui in avanti)"
                 if ist_b["gamma"] != bop.gamma else ")"))
        elif bop is not None and info.get("passi_v"):
            log("fase 2 PV: V-Bop acceso alla ripresa: stato per gruppo nuovo (vuoto)")
        elif bop is None and sb:
            log("fase 2 PV: V-Bop spento alla ripresa: lo stato per gruppo salvato non si usa")
            info.pop("bop", None)
        if bop is not None:
            log(f"fase 2 PV: V-Bop a gruppo, gamma {bop.gamma:g}, curvatura ogni {bop.curvatura_ogni} passi V, "
                f"stato persistente {stato_v.byte_bop() / 2 ** 20:.1f} MiB ({PV.BYTE_PER_GRUPPO_VBOP} B/gruppo)")
        if ca.pv_sonda_dosi:
            log("fase 2 PV: sonda a DOSI (1/256, 1/64, 1/16, 1/4, 1 dei salti scelti, in ordine di guadagno) sul primo "
                "micro-lotto del passo dopo, conferma sul secondo; dose parziale tenuta = rho non dimezzato" +
                (f"; RICENTRO dopo la dose (lascia-uno-fuori, al massimo {ca.pv_ricentra_giri} giri)"
                 if ca.pv_ricentra else ""))
        if ca.pv_punto_medio:
            log(f"fase 2 PV: PUNTO MEDIO (arXiv 2609.09867): {ca.pv_pm_max} primitive ({ca.pv_pm_primitiva}) prezzate "
                f"al punto medio proprio sulla sonda A (trapezio sugli stati interi, un indietro per primitiva + uno), "
                f"filtro dentro la primitiva {'acceso' if ca.pv_pm_filtra else 'spento'}" +
                ("; obiettivo KL delle 4 lettere su righe mc del passo dopo (M1), conferma su M2" if ca.pv_pm_lettere
                 else "") +
                ("; dosi GOLOSE: primitiva per primitiva, tenuta se l'obiettivo scende e la sonda A resta <= partenza"
                 if ca.pv_pm_golosa else "") +
                ("; RICENTRO con l'obiettivo: un giro si tiene se B e le lettere M1 non salgono"
                 if ca.pv_ricentra_obiettivo else ""))
        if ca.pv_riuso:
            log(f"fase 2 PV: RIUSO fra le prove (e2e_riuso_prove): prefisso per strato su {ca.pv_riuso_lotti} "
                f"micro-lotti, indietro troncato al salto piu' basso, piani e scale riusati (stessi numeri)")
        if ca.pv_accetta != "bh":
            log(f"fase 2 PV: accettazione per guadagno di secondo ordine stimato > 0 SENZA i test BH (--pv-accetta "
                f"{ca.pv_accetta}, z_min {ca.pv_z_min:g}, z_guadagno_min {ca.pv_z_guadagno_min:g}); budget rho, "
                f"ordinamento e sonda invariati")
        impr = impronte_byte(st)
        log(f"fase 2 PV accesa: {len(fams_v)} famiglie, rho {reg_f.rho_passo():g}, curvatura su "
            f"{ca.pv_micro_curvatura} micro-lotti")
    lota, regs_cong, n_lota = None, [], 0
    if ca.lota_rango > 0:
        import e2e_lota as LT
        import e2e_pv_codici as PV
        info_pv = stato.get("pv") or {}
        if info_pv.get("registri"):                     # fase 2 PV di prima: codici congelati sotto la base di LoTA
            regs_cong = PV.Registro.carica_lista(dir_c / info_pv["file"], info_pv["sha"])[:info_pv["registri"]]
        lota = LT.installa(st, config_lota(ca), registri_congelati=regs_cong)
        info_l = stato.setdefault("lota", {"inizio": stato["passo"], "file": None, "sha": None, "passi": 0,
                                           "mult": 1.0, "tenuti": 0, "annullati": 0})
        if info_l.get("file"):
            p_l = dir_c / info_l["file"]
            if sha256_file(p_l) != info_l["sha"]:
                raise RuntimeError(f"{p_l}: sha256 dello stato LoTA diverso dalla ripresa")
            ris_l = lota.ripristina(torch.load(p_l, map_location="cpu", weights_only=True))
            ora = {f"{L}.{n}": v for (L, n), v in impronte_byte(st).items()}
            if ora != info_l["impronte"]:
                raise RuntimeError("ripresa LoTA: byte dei codici ricostruiti diversi da quelli salvati (impronte)")
            log(f"LoTA ripresa: {info_l['passi']} passi, moltiplicatore {lota.mult:g}, {ris_l['diversi_da_base']} codici "
                f"diversi dalla base ricostruiti e controllati" + (f"; {len(regs_cong)} registri PV congelati rigiocati"
                                                                  if regs_cong else ""))
        else:
            log(f"LoTA accesa al passo {stato['passo']} (B = 0: codici invariati), rango {ca.lota_rango}, omega "
                f"{lota.cfg.omega_eff:g}, top-p {ca.lota_top_p:g} -> {ca.lota_top_p_fine:g} -> {ca.lota_top_p_coda:g}, "
                f"lr {ca.lota_lr:g}, sonda {'si' if ca.lota_sonda else 'no'}" +
                (f"; {len(regs_cong)} registri PV di una fase precedente rigiocati congelati" if regs_cong else ""))
        n_lota = tot - max(int(info_l["inizio"]), 1)
        b = lota.byte()
        log(f"LoTA: {len(lota.ad)} famiglie, memoria persistente A+B {b['AB'] / 2 ** 20:.1f} MiB, gradiente "
            f"{b['u'] / 2 ** 20:.1f} MiB, base {b['base'] / 2 ** 20:.1f} MiB; schedula su {n_lota} passi")
        impr = impronte_byte(st)
    if held_mc and ripreso:          # DOPO i codici rigiocati dal PV; solo registro di testo (lo stato resta quello salvato)
        log(f"held-out mc alla ripresa (passo {stato['passo']}, {len(held_mc)} righe): " +
            testo_mc(valuta_mc(st, car, held_mc, ca, WS, WT)))
    tempi = []
    stato.setdefault("kl_intervallo", [])      # KL di addestramento dall'ultima misura held-out: e' STATO (ripresa)
    try:
        # 27/9 CAKLD: gamma risolto (deciso o misurato sulle righe del passo 0) PRIMA di held-out 0 e G1
        # 28/9 --righe-base-da N: le righe del passo 0 cominciano da N (0 = [:R] di sempre, al bit)
        ca = risolvi_gamma(stato, ca, src, car,
                           ordine_epoca(train, 0, ca.seme)[ca.righe_base_da:ca.righe_base_da + ca.righe_per_passo], WT, log)
        cakld = ca.perdita == "cakld"
        pesato = pesato_di(ca)                             # 27/9 --peso-token (e --gamma-lettere, trainer-mc)
        if pesato:
            log(f"pesi per token {ca.peso_token} (" + {"kl": f"alfa {ca.peso_alfa:g}, tetto {ca.peso_max:g}",
                                                         "topk": f"frazione {ca.peso_frazione:g}",
                                                         "risposta": f"beta {ca.beta_risposta:g}",
                                                         "generati": f"OPD {ca.opd_kl}, beta {ca.opd_beta:g}, "
                                                                     f"K {ca.opd_eakld_k}, PIVOT {ca.opd_pivot or 'spento'}",
                                                         "uniforme": "tutti 1"}[ca.peso_token] +
                (f"; lettere gamma {ca.gamma_lettere:g}" if ca.gamma_lettere > 0 else "") +
                f") sulla perdita {ca.perdita} del micro-lotto; held-out, sonda del passo V e curvatura UNIFORMI; "
                f"KL train del registro uniforme, la pesata a parte")
        if ca.peso_dominio:                                # 27/9 dominio-strati: nomi controllati (un refuso non pesa nulla)
            noti = {src.dominio("heldout", i) for i in held if src.disponibile("heldout", i)}
            noti |= {dominio_per_peso(src, "train", i) for i in train if src.disponibile("train", i)}
            ignoti = sorted({d for d, _ in ca.peso_dominio} - noti - {DOMINIO_MC})
            if ignoti:
                raise ValueError(f"--peso-dominio: domini {ignoti} assenti dal teacher (domini: {sorted(noti)})")
            log("pesi per dominio (--peso-dominio): " + ", ".join(f"{d} {w:g}" for d, w in ca.peso_dominio) +
                " (non elencati 1; normalizzati sulle righe di ogni passo a media 1; x pesi per token se accesi); "
                "curvatura del passo V/Sophia pesata; held-out, sonda del passo V, attribuzione e KL train UNIFORMI")

        def migliore_di(passo: int, v: dict, nome: str) -> dict:
            m = {"passo": passo, "metrica": v["metrica"], "kl": v["kl"], "dir": nome}
            if cakld:
                m.update({"perdita": "cakld", "metrica_kl": v["metrica_kl"], "gamma": v["gamma"]})
            return m

        def meglio(v: dict, mig: dict) -> bool:
            if mig.get("perdita", "kl") == ca.perdita:
                return v["metrica"] < mig["metrica"]
            # perdita cambiata alla ripresa: le metriche non si confrontano, si confronta la KL in avanti (macro)
            return v.get("metrica_kl", v["metrica"]) < mig.get("metrica_kl", mig["metrica"])

        if _parametri_lora(st) and ca.lora_init == "ga" and \
                stato["passo"] < (tot if ferma_a is None else min(tot, ferma_a)) and \
                (not congela or any(_allenata(st, a.L) for a in st.lora.values())):   # dominio-strati: qualcuno da fare
            import e2e_lora as LR
            if not LR.ha_ga(st):
                # 27/9 LoRA-GA, PRIMA della held-out 0 (l'istantanea del passo 0 porta gia' A0, B0; il modello e' lo stesso al
                # bit, quindi la held-out 0 non cambia): dW sulle righe del passo che sta per iniziare (corsa nuova: il passo
                # 0 di G1; LoRA accesa alla ripresa: il passo della ripresa), K di G1 se c'e' gia', altrimenti K0
                t_ga = stato["passo"]
                # integrata: le righe del passo dal piano (con righe mc: base + mc come il passo vero; senza: il calcolo di
                # sempre, al bit)
                righe_ga = piano.righe(t_ga)[0]
                if ca.lora_ga_righe:
                    righe_ga = righe_ga[:ca.lora_ga_righe]
                stato["attesa_teacher_s"] += attendi_righe(src, "train", righe_ga, log, ca.attesa_ogni_s)
                k_ga = stato["g1"]["K"] if stato.get("g1") else ca.K0
                info_ga = inizializza_lora_ga(st, car, "train", righe_ga, ca, WS, WT, k_ga, log)
                _scrivi_json(dir_c / f"lora_ga_passo{t_ga:06d}.json",
                             {"passo": t_ga, **{k: v for k, v in info_ga.items()}})
                stato["lora_ga"] = {"passo": t_ga, **{k: v for k, v in info_ga.items() if k != "per_tensore"}}
        if stato["migliore"] is None:
            v0 = None
            if riusa_passo0 is not None:
                try:
                    v0, nome = passo0_riusato(dir_c, st, src, ca, held, Path(riusa_passo0), log)
                except RuntimeError as e:
                    log(f"passo 0 NON riusato ({e}): lo rimisuro")
            if v0 is None:
                v0 = valuta(st, car, held, ca, WS, WT)
                nome = salva_istantanea(dir_c, st, 0, v0)
            if held_mc:                                   # 27/9 trainer-mc: solo registro
                v0["mc"] = valuta_mc(st, car, held_mc, ca, WS, WT)
                log("held-out mc al passo 0: " + testo_mc(v0["mc"]))
            stato["migliore"] = migliore_di(0, v0, nome)
            voce0 = {"passo": 0, **v0, "kl_train_intervallo": None}
            if ema is not None:                           # al passo 0 theta_EMA == theta (lo zero): stessa istantanea
                voce0["metrica_ema"] = v0["metrica"]
                stato["migliore_ema"] = migliore_di(0, v0, nome)
            stato["storia_heldout"].append(voce0)
            log(f"passo 0: held-out KL {v0['kl']:.5f}, metrica {v0['metrica']:.5f}, CE {v0['ce_studente']:.4f} "
                f"(teacher {v0['ce_maestro']:.4f})" +
                (f"; CAKLD {v0['perdita']:.5f}, KL inversa {v0['kl_inv']:.5f}, gamma {v0['gamma']:.4f}" if cakld else "") +
                (f"; top-1 uguale {100 * v0['top1_uguale']:.2f}%" if "top1_uguale" in v0 else ""))
        limite = tot if ferma_a is None else min(tot, ferma_a)
        pieno_ogni = ca.heldout_pieno_ogni or ca.heldout_ogni
        held_r = righe_rapide(held, ca.heldout_righe_rapide)
        if ca.heldout_righe_rapide > 0:
            stato.setdefault("storia_heldout_rapido", [])
        if ca.rotazione and not ca.consuma_blocchi:
            log("blocchi rotanti del teacher: consumo SPENTO (--non-consumare): i blocchi usati restano sul disco")
        sonda_att = None
        if ca.attribuzione:                                       # 27/9 integrata: sonda fissa dell'attribuzione
            import e2e_attribuzione as AT
            sonda_att = AT.righe_sonda(held, ca.attribuzione_righe)
            log(f"attribuzione per componente accesa: sonda fissa {len(sonda_att)} righe del held-out {sonda_att}, "
                f"solo_i ogni {ca.attribuzione_solo_ogni} passi; registro {dir_c / 'registro_componenti.jsonl'}")
        while stato["passo"] < limite and stato["fermato"] is None:
            t = stato["passo"]
            righe, ep = piano.righe(t)                    # 27/9: `PianoRighe` (classico = il calcolo di sempre)
            # le righe del passo dopo e la chiusura della finestra PV dipendono dalla corsa INTERA (`tot`), non dalla
            # fermata `ferma_a`: una corsa fermata e ripresa fa gli stessi passi V di quella intera, al bit
            if t + 1 < tot:
                prossime = piano.righe(t + 1)[0]
            else:
                prossime = None
            stato["attesa_teacher_s"] += attendi_righe(src, "train", righe, log, ca.attesa_ogni_s)
            t0 = time.time()
            fasi = None
            if ca.attribuzione:                                   # 27/9 integrata: sonda fissa PRIMA del passo
                import e2e_attribuzione as AT
                fasi = {}
                t_a = time.time()
                att_L0 = AT.perdita_sonda(st, car, sonda_att, ca, WS, WT, kl_pezzo, gamma_di(ca))
                att_ini = AT.InizioPasso(st, _parametri_f32(st), _parametri_lora(st))
                fasi["attribuzione"] = time.time() - t_a
            if stato["g1"] is None:
                g1 = cancello_g1(st, car, "train", righe, ca, WS, WT, log)
                stato["g1"] = g1
                opt = nuovo_ottimizzatore(st, ca, g1["eps_adam"], K=g1["K"])
                kl = None                                         # theta.grad = gradiente del passo 0 (G1)
            else:
                controlla_ottimizzatore(opt, ca, stato["g1"]["eps_adam"], stato["g1"]["K"])
                opt.zero_grad(set_to_none=True)
                if pv is not None:
                    pv.inizia_passo()
                    if pv.stato.passi_accumulo > 1 and pv.stato.passi_nella_finestra == 1:
                        pv.registra_scale_finestra({k: st.esperti(k[0]).scala(k[1]).detach() for k in pv.fam})
                somme = {} if (cakld or pesato) else None
                if somme is not None and ca.attribuzione:
                    somme["con_min_mediana"] = True               # integrata: min/mediana dei pesi nel registro
                rum = MisuraRumore(st, opt, stato["g1"]["K"])
                if lota is not None:                              # LoTA: dL/dA, dL/dB raccolti nel dd degli stessi micro-lotti
                    lota.inizia_passo()
                import contextlib
                cm_lota = lota.raccolta() if lota is not None else contextlib.nullcontext()
                if fasi is None:
                    with cm_lota:
                        kl = gradiente_passo(st, car, "train", righe, ca, WS, WT, stato["g1"]["K"], prossime, pv=pv,
                                             somme=somme, rumore=rum)
                else:                                             # 27/9 integrata: tempi di andata e ritorno
                    t_g = time.time()
                    with AT.cronometra(st, fasi), cm_lota:
                        kl = gradiente_passo(st, car, "train", righe, ca, WS, WT, stato["g1"]["K"], prossime, pv=pv,
                                             somme=somme, rumore=rum)
                    _sync(st.device)
                    fasi["gradiente"] = time.time() - t_g
            t_p = time.time()
            opt.step()
            ctl = st.controlla()                                  # UNA sincronizzazione per passo
            tf = {"P": time.time() - t0} if ca.tempi_fasi else None   # 1/10: tempi per fase (solo registro)
            st.azzera_cambi()
            ris_rumore = stato["g1"].get("rumore") if kl is None else (rum.risultato() if rum is not None else None)
            ev = None
            if fasi is not None:
                fasi["passo_P"] = time.time() - t_p
            reg_passo = None
            if pv is not None and kl is not None:
                ntok_v = len(righe) * lotto_T(src, righe[0])
                chiudi = ca.pv_passi_accumulo <= 1 or (t + 1) % ca.pv_passi_accumulo == 0 or t + 1 == tot
                n_reg0, t_v = len(pv.registri), time.time()
                ev = passo_v(st, src, car, righe, prossime, held, ca, WS, WT, stato["g1"]["K"], ntok_v, pv, reg_f,
                             ca.seme * 1000 + t, log, chiudi=chiudi)
                if tf is not None:
                    tf["V" if chiudi else "V_solo_statistiche"] = time.time() - t_v
                if fasi is not None:
                    fasi["passo_V"] = time.time() - t_v
                    if len(pv.registri) == n_reg0 + 1:            # salti TENUTI di questo passo (annullati: tolti)
                        reg_passo = pv.registri[-1]
            sophia_passo = None
            if getattr(opt, "tipo", None) == "sophia":         # 27/9 newton-theta: clip del passo, curvatura misurata
                nc_, viv_ = opt.sophia.ultimo_conteggio
                sophia_passo = {"non_clippati": int(nc_), "vivi": int(viv_),
                                "quota_non_clippati": int(nc_) / max(int(viv_), 1)}
                if pv is not None and kl is not None:
                    sophia_passo["curvatura_micro"] = aggiorna_curvatura_sophia(
                        st, opt, pv, -(-len(righe) // ca.righe_per_micro))
            if ev is not None and ev.get("solo_statistiche_vbop"):    # 1/10: V-Bop ogni N passi
                log(f"passo V {t + 1}: solo statistiche V-Bop (EMA aggiornata, passi V-Bop {ev['passi_bop']}; verifica "
                    f"{ev['n_verifica']} micro-lotti, curvatura {'misurata' if ev['curvatura'] else 'in EMA'}): "
                    f"decisione ogni {ca.pv_passi_accumulo} passi")
                if "tempi_v" in ev:                               # 2/10 --pv-tempi-v
                    log(f"passo V {t + 1}: " + testo_tempi_v(ev["tempi_v"]))
                ev = None
            elif ev is not None and ev.get("accumulo"):
                log(f"passo V {t + 1}: accumulo {ev['passi_nella_finestra']}/{ca.pv_passi_accumulo} (verifica "
                    f"{ev['n_verifica']} micro-lotti, curvatura {ev['n_curvatura']}): decisione a fine finestra")
                ev = None
            if ev is not None:
                stato["pv"]["passi_v"] += 1
                stato["pv"]["salti_tenuti"] += ev.get("salti_tenuti", ev["salti"]) if ev["tenuto"] else 0
                stato["pv"]["rho"], stato["pv"]["tarato"] = reg_f.rho, reg_f.rho is not None
                impr = impronte_byte(st)                          # i byte cambiano SOLO qui: nuova impronta
                log(f"passo V {t + 1}: salti {ev['salti']} ({'tenuti' if ev['tenuto'] else 'ANNULLATI'}), "
                    f"sonda {'CAKLD' if cakld else 'KL'} {ev['sonda_prima']:.5f} -> {ev['sonda_dopo']:.5f}, rho {ev['rho']:g}, "
                    f"convenienti {ev['convenienti']}/{ev['testati']}" +
                    (f", r {ev['r_fiducia']:.3f}" if ev.get("r_fiducia") is not None else "") +
                    (f", rho dopo {ev['rho_dopo']:g}" if "rho_dopo" in ev else "") +
                    (f", V-Bop: {ev['con_candidato']} gruppi con candidato, curvatura "
                     f"{'misurata' if ev['curvatura'] else 'in EMA'}" if "curvatura" in ev else ""))
                if "dosi" in ev:                                  # 27/9 ramo v-interazione-b: curva dose -> sonda
                    log(f"passo V {t + 1}: " + testo_dosi(ev))
                if "punto_medio" in ev:                           # 29/9 ramo punto-medio
                    log(f"passo V {t + 1}: " + testo_punto_medio(ev["punto_medio"]))
                if "tempi_v" in ev:                               # 2/10 --pv-tempi-v
                    log(f"passo V {t + 1}: " + testo_tempi_v(ev["tempi_v"], ev))
            ev_l = None
            if lota is not None and kl is not None:
                ev_l = passo_lota(st, src, car, prossime, held, ca, WS, WT, stato["g1"]["K"], lota, n_lota, log)
                stato["lota"]["passi"], stato["lota"]["mult"] = lota.passi, lota.mult
                stato["lota"]["tenuti" if ev_l["tenuto"] else "annullati"] += 1
                impr = impronte_byte(st)                          # i byte cambiano anche qui: nuova impronta
                log(f"passo LoTA {ev_l['passo_lota']} (passo {t + 1}): p {ev_l['p']:.3g}, voci A {ev_l['voci_A']['cambiate']}"
                    f"/{ev_l['voci_A']['scelte']}, B {ev_l['voci_B']['cambiate']}/{ev_l['voci_B']['scelte']} cambiate/scelte, "
                    f"codici cambiati {ev_l['codici_cambiati']} ({'tenuti' if ev_l['tenuto'] else 'ANNULLATI'})" +
                    (f", sonda {'CAKLD' if cakld else 'KL'} {ev_l['sonda_prima']:.5f} -> {ev_l['sonda_dopo']:.5f}"
                     if ev_l["sonda_prima"] is not None else "") +
                    (f", sonda fissa {ev_l['sonda_dopo']:.5f} ({ev_l['finestra']['azione']}, migliore "
                     f"{ev_l['finestra']['migliore']:.5f})" if "finestra" in ev_l else "") +
                    f", diversi dalla base {ev_l['codici_diversi_da_base']}, W^ non nulli {ev_l['w_hat_non_nulli']}, "
                    f"moltiplicatore {ev_l['mult_dopo']:g}, {ev_l['secondi']:.1f} s" +
                    ("; INIZIALIZZAZIONE dal gradiente (A = ternario di Omega^T G, B = 0)" if ev_l["init"] else ""))
            if ema is not None:                                   # 27/9: theta_EMA dopo il passo (P e V)
                with torch.no_grad():
                    for (L, _), e, p in zip(st.chiavi(), ema, st.parametri()):
                        if congela and not _allenata(st, L):      # dominio-strati: EMA degli strati congelati al bit
                            continue
                        e.lerp_(p.detach(), 1.0 - ca.ema_theta)
            if t == 0:
                pp = verifica_primo_passo(st, ca.lr, ca.soglia_passo_pieno)
                stato["primo_passo"] = pp
                log(f"primo passo: {100 * pp['quota_passo_pieno']:.3f}% delle scale vive con g != 0 mosse di >= "
                    f"{ca.soglia_passo_pieno} lr, verso giusto {100 * pp['quota_verso_giusto']:.3f}%, mediana "
                    f"|passo|/lr {pp['mediana_passo_su_lr']:.4f}")
                if pp["quota_passo_pieno"] < ca.quota_min_passo_pieno:
                    raise RuntimeError(f"G1 sul passo vero FALLITO: {pp}")
            att_voce = None
            if fasi is not None:                                  # 27/9 integrata: di chi e' il passo (e il peggioramento)
                t_a = time.time()
                con_solo = ca.attribuzione_solo_ogni > 0 and (t + 1) % ca.attribuzione_solo_ogni == 0
                ombra = (opt.sophia.theta_adam if getattr(opt, "tipo", None) == "sophia"
                         and getattr(opt.sophia, "ombra", None) is not None else None)
                if ombra is not None and congela:                 # dominio-strati: AdamW non muove gli strati congelati
                    fermi_id = {id(p) for (L, _), p in zip(st.chiavi(), st.parametri()) if not _allenata(st, L)}
                    ombra = (lambda f, fermi: (lambda p, t0: t0 if id(p) in fermi else f(p, t0)))(ombra, fermi_id)
                att_voce = AT.attribuisci(st, car, sonda_att, ca, WS, WT, kl_pezzo, gamma_di(ca), att_ini, att_L0,
                                          _parametri_f32(st), _parametri_lora(st), pv, reg_passo, impronte_byte,
                                          con_solo, theta_adam=ombra)
                diag = {"theta": AT.statistiche_theta(st, att_ini, ca.lr)}
                if _parametri_f32(st):
                    diag["f32"] = AT.statistiche_f32(st, att_ini)
                if _parametri_lora(st):
                    diag["lora"] = AT.statistiche_lora(st, att_ini)
                del att_ini
                fasi["attribuzione"] += time.time() - t_a
                att_voce["diagnostica"] = diag
            ntok = len(righe) * lotto_T(src, righe[0])
            kl_v = (stato["g1"]["kl_somma_passo0"] if kl is None else float(kl)) / ntok
            stato["passo"] = t + 1
            stato["cambi_instradamento"] += ctl["cambi_instradamento"]
            voce = {"passo": t + 1, "epoca": ep, "righe": [righe[0], righe[-1]], "kl_train": kl_v,
                    "cambi_instradamento": ctl["cambi_instradamento"]}
            if cakld:                                             # kl_train = perdita CAKLD (quella ottimizzata)
                g1_ = stato["g1"]
                voce["kl_fwd_train"] = (g1_["kl_fwd_somma_passo0"] if kl is None else float(somme["kl"])) / ntok
                voce["kl_inv_train"] = (g1_["kl_inv_somma_passo0"] if kl is None else float(somme["kl_inv"])) / ntok
            if pesato:                                            # --peso-token: la perdita ottimizzata e i pesi
                pz = stato["g1"].get("pesi_passo0") if kl is None else riassunto_pesi(somme)
                if pz is not None:
                    voce["pesi"] = pz
                po = per_origine(somme) if kl is not None else None   # integrata --attribuzione con righe mc
                if po is not None:
                    voce["per_origine"] = po
            if ca.peso_dominio:                                   # 27/9 --peso-dominio: pesi effettivi del passo
                voce["peso_dominio"] = riassunto_peso_dominio(src, "train", righe, ca)
            if ev is not None:
                voce["pv"] = ev
            if ev_l is not None:
                voce["lota"] = ev_l
            voce["rumore"] = ris_rumore
            if sophia_passo is not None:
                voce["sophia"] = sophia_passo
            if att_voce is not None:
                voce["attribuzione"] = att_voce
            dt = time.time() - t0
            tempi.append({"passo": t + 1, "secondi": round(dt, 2), "ms_token": round(1000 * dt / ntok, 4)})
            stato["kl_intervallo"].append(kl_v)
            stato["storia"].append(voce)
            log(f"passo {t + 1}/{limite}: KL train {kl_v:.5f}  {dt:.1f} s  {1000 * dt / ntok:.3f} ms/token  "
                f"cambi instr. {ctl['cambi_instradamento']}")
            log(f"passo {t + 1}: " + testo_rumore(ris_rumore))
            if "pesi" in voce:
                log(f"passo {t + 1}: pesi per token {ca.peso_token}: " + testo_pesi(voce["pesi"]))
            if sophia_passo is not None:
                log(f"passo {t + 1}: sophia: non clippati {sophia_passo['non_clippati']}/{sophia_passo['vivi']} "
                    f"({100 * sophia_passo['quota_non_clippati']:.2f}%), curvatura " +
                    (f"misurata su {sophia_passo['curvatura_micro']} micro-lotti" if sophia_passo.get("curvatura_micro")
                     else "non misurata (resta la EMA)"))
            if att_voce is not None:
                dg = att_voce["diagnostica"]
                log(f"passo {t + 1}: attribuzione: " + AT.testo_attribuzione(att_voce))
                log(f"passo {t + 1}: theta: |passo|/lr mediana {dg['theta']['mediana_su_lr']:.4f}, q90 "
                    f"{dg['theta']['q90_su_lr']:.4f}, a passo pieno {100 * dg['theta']['quota_passo_pieno']:.2f}%" +
                    ("; F32 |passo| " + ", ".join(f"{f} {x['norma_passo']:.3e}" for f, x in dg["f32"].items())
                     if "f32" in dg else "") +
                    ("; LoRA |dW| passo/totale " + ", ".join(f"{f} {x['norma_passo']:.3e}/{x['norma_totale']:.3e}"
                                                            for f, x in dg["lora"].items()) if "lora" in dg else ""))
                log(f"passo {t + 1}: tempi " + ", ".join(f"{k} {v:.1f} s" for k, v in fasi.items()))
            # 27/9: misura PIENA ogni `heldout_pieno_ogni` (0 = heldout_ogni, come prima), a fine corsa e a fine epoca
            t_val = time.time()
            if scrittore is not None:                             # 1/10: il migliore nuovo nasce qui sotto
                scrittore.attendi()
                if pv is not None:
                    stato["pv"].update(scrittore.prendi_risultati())
            misura_h = (t + 1) % pieno_ogni == 0 or t + 1 == tot or piano.fine_epoca(t)
            misura_r = not misura_h and ca.heldout_righe_rapide > 0 and (t + 1) % ca.heldout_ogni == 0
            if misura_r:                                          # solo registro: non tocca migliore ne' arresto
                _uguale_ai_byte(st, impr)
                vr = valuta(st, car, held_r, ca, WS, WT)
                buoni = [x for x in stato["kl_intervallo"] if x == x]
                vr["kl_train_intervallo"] = sum(buoni) / len(buoni) if buoni else None
                vr["righe"] = len(held_r)
                if held_mc:                                   # 27/9 trainer-mc: solo registro
                    vr["mc"] = valuta_mc(st, car, held_mc, ca, WS, WT)
                stato["storia_heldout_rapido"].append({"passo": t + 1, **vr})
                log(f"held-out RAPIDO al passo {t + 1} ({len(held_r)} righe): KL {vr['kl']:.5f}, metrica "
                    f"{vr['metrica']:.5f}; per dominio " +
                    ", ".join(f"{d} {x:.4f}" for d, x in vr["kl_per_dominio"].items()))
                if held_mc:
                    log(f"held-out mc al passo {t + 1}: " + testo_mc(vr["mc"]))
            if misura_h:
                _uguale_ai_byte(st, impr)
                v = valuta(st, car, held, ca, WS, WT)
                buoni = [x for x in stato["kl_intervallo"] if x == x]
                v["kl_train_intervallo"] = sum(buoni) / len(buoni) if buoni else None
                stato["kl_intervallo"] = []
                if held_mc:                                   # 27/9 trainer-mc: solo registro (mai in `meglio`)
                    v["mc"] = valuta_mc(st, car, held_mc, ca, WS, WT)
                    log(f"held-out mc al passo {t + 1}: " + testo_mc(v["mc"]))
                ve = None
                if ema is not None:                               # 27/9: anche theta_EMA, sulle stesse righe
                    with theta_scambiato(st, ema):
                        ve = valuta(st, car, held, ca, WS, WT)
                    v["metrica_ema"] = ve["metrica"]
                    v["ema"] = ve
                stato["storia_heldout"].append({"passo": t + 1, **v})
                msg = ""
                if meglio(v, stato["migliore"]):
                    nome = salva_istantanea(dir_c, st, t + 1, v, ema=None if ve is None else (ema, ve))
                    stato["migliore"] = migliore_di(t + 1, v, nome)
                    regs_c = _registri_codici(pv, lota, regs_cong)
                    if regs_c is not None:                         # il migliore = scale + codici DI QUEL passo
                        import e2e_pv_codici as PV
                        stato["migliore"]["registri_sha"] = PV.Registro.salva_lista(dir_c / nome / "registri_pv.npz",
                                                                                     regs_c)
                        stato["migliore"]["registri"] = len(regs_c)
                        if lota is not None:
                            stato["migliore"]["lota_codici"] = len(regs_c[-1])
                    msg = " (MIGLIORE)"
                if ve is not None:
                    msg += f"; theta_EMA: KL {ve['kl']:.5f}, metrica {ve['metrica']:.5f}"
                    if stato.get("migliore_ema") is None or meglio(ve, stato["migliore_ema"]):
                        with theta_scambiato(st, ema):
                            nome_e = salva_istantanea(dir_c, st, t + 1, ve, prefisso="migliore_ema")
                        stato["migliore_ema"] = migliore_di(t + 1, ve, nome_e)
                        regs_c = _registri_codici(pv, lota, regs_cong)
                        if regs_c is not None:
                            import e2e_pv_codici as PV
                            stato["migliore_ema"]["registri_sha"] = PV.Registro.salva_lista(
                                dir_c / nome_e / "registri_pv.npz", regs_c)
                            stato["migliore_ema"]["registri"] = len(regs_c)
                        msg += " (MIGLIORE EMA)"
                log(f"held-out al passo {t + 1}: KL {v['kl']:.5f}, metrica {v['metrica']:.5f}{msg}; per dominio " +
                    ", ".join(f"{d} {x:.4f}" for d, x in v["kl_per_dominio"].items()) +
                    (f"; CAKLD {v['perdita']:.5f}, KL inversa {v['kl_inv']:.5f}, KL macro {v['metrica_kl']:.5f}"
                     if cakld else "") + f"; top-1 uguale {100 * v['top1_uguale']:.2f}%")
                # l'arresto confronta solo misure della STESSA perdita (una ripresa puo' averla cambiata)
                storia_ok = [x for x in stato["storia_heldout"] if x["kl_train_intervallo"] is not None
                             and x.get("perdita_nome", "kl") == ca.perdita]
                if regola_arresto(storia_ok, int(os.environ.get("E2E_SALITE_ARRESTO", ca.salite_per_arresto))):
                    stato["fermato"] = {"passo": t + 1, "motivo": "held-out in salita con addestramento in discesa"}
                    log(f"ARRESTO al passo {t + 1}: {stato['fermato']['motivo']}")
                if piano.fine_prima_epoca(t) and ca.epoche > 1 and stato["fermato"] is None:
                    h = stato["storia_heldout"]
                    rap = v["perdita" if cakld else "kl"] / v["kl_train_intervallo"] if v["kl_train_intervallo"] \
                        else float("inf")
                    if not (len(h) >= 2 and h[-1]["metrica"] < h[-2]["metrica"] and rap <= ca.seconda_epoca_se):
                        stato["fermato"] = {"passo": t + 1, "motivo": f"2a epoca non ammessa (held/train {rap:.3f})"}
                        log(f"FINE dopo 1 epoca: {stato['fermato']['motivo']}")
            if tf is not None:
                tf["valutazioni"] = time.time() - t_val
            t_sal = time.time()
            ist_passo = None
            if ca.istantanea_ogni_passo:                          # 27/9 integrata: istantanea esportabile a OGNI passo
                val_i = {"passo": t + 1, "kl_train": kl_v}
                if misura_r:
                    val_i["heldout_rapido"] = {k: stato["storia_heldout_rapido"][-1].get(k) for k in ("kl", "metrica")}
                if misura_h:
                    val_i["heldout"] = {k: stato["storia_heldout"][-1].get(k) for k in ("kl", "metrica", "metrica_ema")}
                ist_passo = salva_istantanea_passo(Path(ca.istantanee_dir) if ca.istantanee_dir else dir_c, st, t + 1,
                                                   ema, pv, val_i, ca.istantanee_tieni, ca.istantanee_libero_min_gib, log)
                if "dir" in ist_passo:
                    log(f"istantanea del passo {t + 1}: {ist_passo['dir']}" + (" (+ ema/)" if ist_passo["ema"] else "") +
                        f", {ist_passo['byte'] / 2 ** 30:.2f} GiB, {ist_passo['secondi']:.0f} s")
            if fasi is not None:                                  # 27/9 integrata: una riga JSON per passo
                riga = {"passo": t + 1, "data": adesso(), "secondi_passo": round(dt, 2),
                        "tempi": {k: round(v, 2) for k, v in fasi.items()}, **{k: voce[k] for k in voce if k != "passo"}}
                if misura_r:
                    riga["heldout_rapido"] = stato["storia_heldout_rapido"][-1]
                if misura_h:
                    riga["heldout"] = stato["storia_heldout"][-1]
                if ist_passo is not None:
                    riga["istantanea"] = ist_passo
                scrivi_riga_componenti(dir_c, riga)
            if misura_h or (t + 1) % ca.ripresa_ogni == 0 or t + 1 == limite or stato["fermato"]:
                if scrittore is not None:                         # 1/10: la ripresa di prima e' scritta
                    scrittore.attendi()
                    if pv is not None:
                        stato["pv"].update(scrittore.prendi_risultati())
                if pv is not None:
                    _salva_registri_pv(dir_c, pv, stato)
                    _salva_accumulo_pv(dir_c, pv, stato)
                    if scrittore is None:
                        _salva_bop_pv(dir_c, pv, stato)
                if lota is not None:
                    _salva_lota(dir_c, lota, stato, impr)
                if scrittore is None:
                    salva_ripresa(dir_c, st, opt, stato, ca.tieni_riprese, ema=ema)
                    if pv is not None:
                        _pota_registri_pv(dir_c)
                else:                                             # 1/10 --scritture-asincrone (stessi file)
                    salva_ripresa(dir_c, st, opt, stato, ca.tieni_riprese, ema=ema, scrittore=scrittore,
                                  bop=_bop_da_scrivere(pv, stato, sosta=ca.scritture_sosta_dispositivo)
                                  if pv is not None else None,
                                  dopo=(lambda: _pota_registri_pv(dir_c)) if pv is not None else None,
                                  sosta=ca.scritture_sosta_dispositivo)
                if lota is not None:
                    _pota_lota(dir_c)
                if ca.rotazione and ca.epoche == 1 and ca.consuma_blocchi:
                    usate = set(train[piano.base_iniziale():piano.base_consumate(stato["passo"])])   # 28/9 da N
                    for nome in blocchi_consumabili(src, usate):
                        if nome not in stato["consumati"]:
                            src.consuma(nome)
                            stato["consumati"].append(nome)
                            log(f"blocco del teacher consumato (dopo la ripresa del passo {stato['passo']}): {nome}")
            if tf is not None:                                    # 1/10 tt5-ottimizzato
                tf["salvataggi"] = time.time() - t_sal
                log(f"passo {t + 1}: tempi per fase " + ", ".join(f"{k} {v:.1f} s" for k, v in tf.items()) +
                    f" (totale {sum(tf.values()):.1f} s)")
        _uguale_ai_byte(st, impr)
        if scrittore is not None:                                 # 1/10: l'ultima ripresa e' su disco prima di uscire
            scrittore.attendi()
            if pv is not None:
                stato["pv"].update(scrittore.prendi_risultati())
    finally:
        if scrittore is not None:
            try:
                scrittore.attendi()
            except RuntimeError as e:                             # gia' in uscita per un altro errore: lo si registra
                log(f"⛔ {e}")
        car.chiudi()
        if congela:                                    # 27/9 dominio-strati: lo studente torna com'era (tutto allenabile)
            st.congela_strati(None)
        if getattr(st, "lora_ferma", False):           # 28/9 --lora-ripristina ferma: idem (A, B di nuovo allenabili)
            st.lora_ferma = False
            for p in st.parametri_lora():
                p.requires_grad_(True)
    stato["tempi_ultimi"] = tempi[-50:]
    return stato


def ricetta_sostituibile(firma_prima: Optional[dict], firma_ora: dict) -> bool:
    """27/9 (--cambia-ricetta, ramo indietro-bf16x3): la ripresa puo' cambiare ricetta se TUTTO il resto della firma
    dello studente e' uguale e le due ricette hanno lo stesso formato (residuo, prodotti degli esperti, rotazione,
    router): cambia solo l'aritmetica (B_ind32 -> B_ind2: stesso avanti, altro indietro; B_ind32 -> B3_ind2: non
    esperti in bf16x3, ~1e-5 sui logit). theta, Adam, F32, LoRA e registri V restano quelli della ripresa."""
    import e2e_studente as S
    if not firma_prima or set(firma_prima) != set(firma_ora):
        return False
    if any(firma_prima[k] != firma_ora[k] for k in firma_ora if k != "ricetta"):
        return False
    a, b = S.RICETTE.get(firma_prima.get("ricetta")), S.RICETTE.get(firma_ora.get("ricetta"))
    if a is None or b is None:
        return False
    return all(getattr(a, c) == getattr(b, c) for c in ("dtype_residuo", "dtype_gemm", "ruota", "router_fp32"))


def _firma_studente(st) -> dict:
    """Quale studente: GGUF (percorso, byte, data), ricetta, strati, esperti per famiglia."""
    p = Path(st.gguf)
    f = {"gguf": str(p), "byte": p.stat().st_size, "mtime": int(p.stat().st_mtime), "ricetta": st.ric.nome,
         "strati": list(st.strati), "esperti": [int(st.esperti(L).E) for L in st.strati[:1]]}
    if getattr(st, "densi_ternari", ()):              # 28/9 --densi-ternari: solo se acceso (firma di sempre altrimenti)
        f["densi_ternari"] = list(st.densi_ternari)
    return f


def lotto_T(src, i: int) -> int:
    return int(np.asarray(src.ids("train", i)).shape[0])


# ═══════════════════════════════════════ misura (M0) ═══════════════════════════════════════
def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def prime_disponibili(src, lato: str, n: int) -> list:
    """27/9: le prime n righe del lato GIA' sigillate e non consumate (i blocchi rotanti consumati mancano)."""
    src.aggiorna()
    righe = [i for i in src.righe_previste(lato) if src.disponibile(lato, i)][:n]
    if len(righe) < n:
        raise RuntimeError(f"{lato}: solo {len(righe)} righe disponibili, ne servono {n}")
    return righe


def _salva_dtheta(st, strati, cartella: Path) -> Path:
    """dL/dtheta del primo micro-lotto (avanti+indietro della ripetizione 0 di `misura`), con le chiavi e le scale
    vive (d0 != 0), per confrontare fra loro le varianti del GDN (E2E_GDN) come fa G1 con l'indietro fp32."""
    etichetta = os.environ.get("E2E_MISURA_ETICHETTA") or os.environ.get("E2E_GDN", "ripiego")
    cartella.mkdir(parents=True, exist_ok=True)
    f = cartella / f"dtheta_{etichetta}_strati{'-'.join(map(str, strati))}.pt"
    torch.save({"chiavi": [list(k) for k in st.chiavi()], "grad": [p.grad.detach().cpu() for p in st.parametri()],
                "vive": [(st.esperti(L).fam[n].d0 != 0).cpu() for L, n in st.chiavi()],
                "gdn": os.environ.get("E2E_GDN", "ripiego")}, f)
    return f


def misura(argv_strati: list, teacher: Path, gguf: Path, ca: ConfigAllena, ripetizioni: int, device: str,
           ricetta: str = "B", log=print, disponibili: bool = False) -> dict:
    """ms/token del passo intero (avanti + KL + indietro + Adam) su 1-2 strati veri, a parti separate, e picco di
    memoria; proiezione a 40 strati: fisso + 30 x GDN + 10 x attenzione piena (tipi da config).
    27/9: teste secondo `ca.testa_kl` / `ca.logit_maestro`; `disponibili` = le prime righe train presenti (non le
    prime del piano, che coi blocchi rotanti possono essere consumate)."""
    import e2e_studente as S
    src = SorgenteCache(teacher)
    if disponibili:
        righe = prime_disponibili(src, "train", ca.righe_per_micro)
    else:
        righe = src.righe_previste("train")[:ca.righe_per_micro]
        attendi_righe(src, "train", righe, log)
    dev = torch.device(device)
    WT, sha_t = testa_maestro(dev, torch.bfloat16 if ca.logit_maestro == "bf16" else torch.float32)
    ris = {"righe": righe, "micro_token": None, "ricetta": ricetta, "testa_kl": ca.testa_kl,
           "logit_maestro": ca.logit_maestro, "configurazioni": {}}
    for strati in argv_strati:
        st = S.Studente(gguf=gguf, strati=strati, ricetta=ricetta,
                        motore_nome=os.environ.get("E2E_MOTORE", "triton") if dev.type == "cuda" else "torch", device=device, log=log,
                        **({"densi_ternari": ca.densi_ternari} if ca.densi_ternari else {}))   # 28/9 --densi-ternari
        if dev.type == "cuda":
            st.carica_embedding()
            torch.cuda.reset_peak_memory_stats()
        WS = testa_studente(st, ca.testa_kl)
        car = Caricatore(src, dev)
        lot = car.prendi("train", righe)
        ntok = lot.ids.numel()
        ris["micro_token"] = ntok
        parti = {"avanti": [], "kl": [], "indietro": [], "adam": []}
        opt = torch.optim.AdamW(st.parametri(), lr=ca.lr, eps=1e-8, weight_decay=0.0, fused=dev.type == "cuda")
        for r in range(ripetizioni + 1):
            _sync(dev); a = time.time()
            stato = st.avanti(lot.ids, inizi=lot.inizi)
            _sync(dev); b = time.time()
            xL = stato["xL"].detach().requires_grad_(True)
            with torch.enable_grad():
                hS = st.esce(xL)
                kl = KLFusa.apply(hS, WS, lot.hT, WT, ca.pezzo_kl)
                (kl * (2.0 ** 20 / ntok)).backward()
            _sync(dev); c = time.time()
            st.indietro(stato, xL.grad)
            _sync(dev); d = time.time()
            if r == 0 and os.environ.get("E2E_MISURA_DTHETA"):      # 27/9: dtheta del 1o micro-lotto, per variante GDN
                _salva_dtheta(st, strati, Path(os.environ["E2E_MISURA_DTHETA"]))
            opt.step()
            opt.zero_grad(set_to_none=True)
            _sync(dev); e = time.time()
            if r:                                           # la prima e' riscaldamento (compilazione Triton)
                for k, v in zip(parti, (b - a, c - b, d - c, e - d)):
                    parti[k].append(v)
            del stato, xL, hS, kl
        med = {k: 1000 * sorted(v)[len(v) // 2] / ntok for k, v in parti.items()}
        picco = torch.cuda.max_memory_allocated() / 2 ** 30 if dev.type == "cuda" else None
        ris["configurazioni"][",".join(map(str, strati))] = {"ms_token": med, "totale_ms_token": sum(med.values()),
                                                             "picco_torch_gib": picco,
                                                             "tipi": [st.cfg.layer_types[L] for L in strati]}
        log(f"strati {strati}: {json.dumps(ris['configurazioni'][','.join(map(str, strati))])}")
        car.chiudi()
        del st, WS, opt
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    import e2e_bf16x3 as X3
    ris["via_bf16x3"] = X3.stato()
    return ris


# ═══════════════════════════════════════ riga di comando ═══════════════════════════════════════
def _famiglie_f32(testo: str) -> tuple:
    if not testo:
        return ()
    import e2e_studente as S
    return S.famiglie_f32(testo)


def _famiglie_dense(testo: str) -> tuple:
    """28/9 --densi-ternari "attn,shexp" -> tupla canonica (vuoto -> ())."""
    if not testo:
        return ()
    import e2e_studente as S
    return S.famiglie_dense(testo)


def _lora_da_arg(a) -> dict:
    """--lora-nonesperti R [--lora-alpha A] [--lora-famiglie ...] [--lr-lora X] -> campi di ConfigAllena (R = 0: i
    campi di default, identici a prima anche se --lora-famiglie/--lora-alpha sono passati)."""
    r = int(getattr(a, "lora_nonesperti", 0) or 0)
    lr = getattr(a, "lr_lora", ConfigAllena.lr_lora)
    if r == 0:
        return {"lora_rango": 0, "lora_alpha": None, "lora_famiglie": (), "lr_lora": lr, "lora_tipi": ()}
    import e2e_lora as LR
    out = {"lora_rango": r, "lora_alpha": a.lora_alpha, "lora_famiglie": LR.famiglie_lora(a.lora_famiglie),
           "lr_lora": lr, "lora_tipi": LR.tipi_lora(getattr(a, "lora_tipi", ""))}
    ini = getattr(a, "lora_init", None) or ConfigAllena.lora_init          # 27/9 --lora-init (LoRA-GA)
    if ini != ConfigAllena.lora_init:
        out.update({"lora_init": ini, "lora_ga_gamma": float(getattr(a, "lora_ga_gamma", ConfigAllena.lora_ga_gamma)),
                    "lora_ga_righe": int(getattr(a, "lora_ga_righe", ConfigAllena.lora_ga_righe))})
    return out


def config_lora(ca: ConfigAllena):
    """ConfigLoRA dello studente (None se spenta); il seme degli A e' il seme della corsa."""
    if ca.lora_rango == 0:
        return None
    import e2e_lora as LR
    return LR.ConfigLoRA(rango=ca.lora_rango, alpha=ca.lora_alpha, famiglie=tuple(ca.lora_famiglie), seme=ca.seme,
                         tipi=tuple(ca.lora_tipi))


def _frazioni_da_arg(testo) -> tuple:
    """--pv-dosi-frazioni "1/1024,1/256,...,1" -> (0.0009765625, 0.00390625, ..., 1.0); vuoto -> ()."""
    from fractions import Fraction
    if isinstance(testo, (tuple, list)):
        return tuple(float(x) for x in testo)
    return tuple(float(Fraction(x.strip())) for x in str(testo or "").split(",") if x.strip())


def _peso_da_arg(a) -> dict:
    """--peso-token/--peso-alfa/--peso-max/--peso-frazione -> campi di ConfigAllena; un Namespace che non li ha (chiamanti
    di prima, prove) vale i default (uniforme, identico al bit). 27/9 dominio-strati: anche --peso-dominio e
    --strati-allenati (assenti = vuoti = spenti)."""
    out = {k: getattr(a, k, getattr(ConfigAllena, k)) for k in ("peso_token", "peso_alfa", "peso_max", "peso_frazione")}
    out["peso_dominio"] = peso_dominio_da_testo(getattr(a, "peso_dominio", "") or "")
    out["strati_allenati"] = strati_da_testo(getattr(a, "strati_allenati", "") or "")
    return out


def config_lota(ca: ConfigAllena):
    """ConfigLoTA della corsa (None se spenta); il seme degli A e' il seme della corsa."""
    if ca.lota_rango == 0:
        return None
    import e2e_lota as LT
    return LT.ConfigLoTA(rango=ca.lota_rango, omega=ca.lota_omega, top_p=ca.lota_top_p, top_p_fine=ca.lota_top_p_fine,
                         top_p_coda=ca.lota_top_p_coda, lr=ca.lota_lr, tau=ca.lota_tau, seme=ca.seme,
                         init=ca.lota_init, tolleranza=ca.lota_sonda_tolleranza, finestra=ca.lota_sonda_finestra)


def _lota_da_arg(a) -> dict:
    """--lota-* -> campi di ConfigAllena (rango 0: tutti ai default, identico a prima)."""
    r = int(getattr(a, "lota_rango", 0) or 0)
    if r == 0:
        return {}
    return {"lota_rango": r, "lota_omega": a.lota_omega, "lota_top_p": a.lota_top_p, "lota_top_p_fine": a.lota_top_p_fine,
            "lota_top_p_coda": a.lota_top_p_coda, "lota_lr": a.lota_lr, "lota_tau": a.lota_tau,
            "lota_sonda": not a.lota_senza_sonda, "lota_init": getattr(a, "lota_init", "paper"),
            "lota_sonda_tolleranza": getattr(a, "lota_sonda_tolleranza", 0.0),
            "lota_sonda_finestra": getattr(a, "lota_sonda_finestra", 0)}


def _mc_da_arg(a) -> dict:
    """27/9 trainer-mc: --righe-mc/--beta-risposta/--gamma-lettere -> campi di ConfigAllena (default se il Namespace non
    li ha: chiamanti di prima, prove)."""
    return {"righe_mc": int(getattr(a, "righe_mc", 0) or 0),
            "beta_risposta": float(getattr(a, "beta_risposta", _BETA_RISPOSTA_DEFAULT)),
            "gamma_lettere": float(getattr(a, "gamma_lettere", 0.0) or 0.0),
            "righe_mc_permutate": bool(getattr(a, "righe_mc_permutate", False)),
            "opd_beta": float(getattr(a, "opd_beta", 1.0)), "opd_kl": str(getattr(a, "opd_kl", "campionata")),
            "opd_eakld_k": int(getattr(a, "opd_eakld_k", 16)), "opd_pivot": str(getattr(a, "opd_pivot", "") or "")}


def _config_da_arg(a) -> ConfigAllena:
    return ConfigAllena(**_lora_da_arg(a), **_lota_da_arg(a), **_mc_da_arg(a), lr=a.lr, righe_per_passo=a.righe_passo, righe_per_micro=a.righe_micro,
                        righe_per_micro_valuta=a.righe_micro_valuta, epoche=a.epoche, heldout_ogni=a.heldout_ogni,
                        ripresa_ogni=a.ripresa_ogni, pezzo_kl=a.pezzo_kl, rotazione=a.rotazione, K0=a.K0,
                        pv_codici=a.pv_codici, pv_rho=a.pv_rho, pv_micro_curvatura=a.pv_micro_curvatura, pv_n_proposta=a.pv_n_proposta, pv_q_fdr=a.pv_q_fdr,
                        pv_q_guadagno=a.pv_q_guadagno, pv_passi_accumulo=a.pv_passi_accumulo,
                        pv_modo=a.pv_modo, pv_gamma=a.pv_gamma, pv_curvatura_ogni=a.pv_curvatura_ogni,
                        pv_rho_adattivo=a.pv_rho_adattivo, pv_rho_max=a.pv_rho_max,
                        pv_accetta=getattr(a, "pv_accetta", ConfigAllena.pv_accetta),
                        pv_z_min=getattr(a, "pv_z_min", ConfigAllena.pv_z_min),
                        pv_z_guadagno_min=getattr(a, "pv_z_guadagno_min", ConfigAllena.pv_z_guadagno_min),
                        pv_sonda_dosi=getattr(a, "pv_sonda_dosi", ConfigAllena.pv_sonda_dosi),
                        pv_ricentra=getattr(a, "pv_ricentra", ConfigAllena.pv_ricentra),
                        pv_ricentra_giri=getattr(a, "pv_ricentra_giri", ConfigAllena.pv_ricentra_giri),
                        pv_dosi_frazioni=_frazioni_da_arg(getattr(a, "pv_dosi_frazioni", "")),
                        pv_punto_medio=getattr(a, "pv_punto_medio", ConfigAllena.pv_punto_medio),
                        pv_pm_primitiva=getattr(a, "pv_pm_primitiva", ConfigAllena.pv_pm_primitiva),
                        pv_pm_max=getattr(a, "pv_pm_max", ConfigAllena.pv_pm_max),
                        pv_pm_filtra=not getattr(a, "pv_pm_non_filtrare", False),
                        pv_pm_lettere=getattr(a, "pv_pm_lettere", ConfigAllena.pv_pm_lettere),
                        pv_pm_golosa=getattr(a, "pv_pm_golosa", ConfigAllena.pv_pm_golosa),
                        pv_riuso=getattr(a, "pv_riuso", ConfigAllena.pv_riuso),
                        pv_riuso_lotti=getattr(a, "pv_riuso_lotti", ConfigAllena.pv_riuso_lotti),
                        pv_ricentra_obiettivo=getattr(a, "pv_ricentra_obiettivo", ConfigAllena.pv_ricentra_obiettivo),
                        pv_dd_solo_salti=getattr(a, "pv_dd_solo_salti", ConfigAllena.pv_dd_solo_salti),
                        pv_pm_ordine_strati=getattr(a, "pv_pm_ordine_strati", ConfigAllena.pv_pm_ordine_strati),
                        pv_golosa_pigra=getattr(a, "pv_golosa_pigra", ConfigAllena.pv_golosa_pigra),
                        pv_tempi_v=getattr(a, "pv_tempi_v", ConfigAllena.pv_tempi_v),
                        scritture_sosta_dispositivo=getattr(a, "scritture_sosta_dispositivo",
                                                            ConfigAllena.scritture_sosta_dispositivo),
                        perdita=a.perdita, gamma=a.gamma, beta2=a.beta2, ema_theta=a.ema_theta,
                        **_peso_da_arg(a),
                        heldout_righe_rapide=a.heldout_righe_rapide, heldout_pieno_ogni=a.heldout_pieno_ogni,
                        # getattr: chi costruisce gli argomenti a mano (prove di --allena-f32) non li ha
                        testa_kl=getattr(a, "testa_kl", ConfigAllena.testa_kl),
                        logit_maestro=getattr(a, "logit_maestro", ConfigAllena.logit_maestro),
                        allena_f32=_famiglie_f32(a.allena_f32), lr_f32=a.lr_f32,
                        densi_ternari=_famiglie_dense(getattr(a, "densi_ternari", "") or ""),   # 28/9
                        f32_ripristina=_famiglie_f32(getattr(a, "f32_ripristina", "") or ""),
                        lora_ripristina=getattr(a, "lora_ripristina", "") or "",
                        ottimizzatore_theta=getattr(a, "ottimizzatore_theta", ConfigAllena.ottimizzatore_theta),
                        sophia_gamma=getattr(a, "sophia_gamma", ConfigAllena.sophia_gamma),
                        sophia_beta2=getattr(a, "sophia_beta2", ConfigAllena.sophia_beta2),
                        sophia_eps=getattr(a, "sophia_eps", ConfigAllena.sophia_eps),
                        attribuzione=getattr(a, "attribuzione", ConfigAllena.attribuzione),
                        attribuzione_righe=getattr(a, "attribuzione_righe", ConfigAllena.attribuzione_righe),
                        attribuzione_solo_ogni=getattr(a, "attribuzione_solo_ogni", ConfigAllena.attribuzione_solo_ogni),
                        istantanea_ogni_passo=getattr(a, "istantanea_ogni_passo", ConfigAllena.istantanea_ogni_passo),
                        tempi_fasi=getattr(a, "tempi_fasi", ConfigAllena.tempi_fasi),
                        scritture_asincrone=getattr(a, "scritture_asincrone", ConfigAllena.scritture_asincrone),
                        consuma_blocchi=not getattr(a, "non_consumare", False),
                        istantanee_dir=getattr(a, "istantanee_dir", ConfigAllena.istantanee_dir),
                        istantanee_tieni=getattr(a, "istantanee_tieni", ConfigAllena.istantanee_tieni),
                        istantanee_libero_min_gib=getattr(a, "istantanee_libero_min_gib",
                                                          ConfigAllena.istantanee_libero_min_gib),
                        righe_base_da=int(getattr(a, "righe_base_da", 0) or 0),      # 28/9 EB4
                        escludi_domini=tuple(x for x in a.escludi_domini.split(",") if x))


def _pid_unita_systemd(unita: str) -> list:
    """PID vivi ORA nel cgroup di un'unita' systemd --user (lista vuota se non attiva/non trovata). Letto
    fresco ad ogni chiamata: il figlio `e2e_dump_logit` del teacher cambia PID ad ogni blocco (26/9, misurato:
    blocco_0015 PID 4185852, il successivo 9990 sei minuti dopo — MAI un elenco statico)."""
    p = Path(f"/sys/fs/cgroup/user.slice/user-{os.getuid()}.slice/user@{os.getuid()}.service"
             f"/app.slice/{unita}.service/cgroup.procs")
    try:
        return [int(x) for x in p.read_text().split()]
    except OSError:
        return []


def _pid_teacher_attivo() -> list:
    """PID di TUTTE le unita' `e2e-teacher*` attive ORA (coda principale + eventuali accodate): usato SOLO
    per escludere il teacher dalla guardia del training (§26/9, coesistenza disegnata: `e2e_allena.py` legge
    in sola lettura, mai un subprocess verso `e2e_dump_logit`). Import lazy di `e2e_teacher`, come `SorgenteCache`.
    Fallisce muto (lista vuota) se il modulo teacher non e' importabile: la guardia resta severa, mai piu' permissiva."""
    try:
        import e2e_teacher as T  # sola lettura, stessa convenzione di SorgenteCache
        unita = T.unita_teacher_attive()
    except Exception:
        return []
    pid = []
    for u in unita:
        pid += _pid_unita_systemd(u)
    return pid


@contextmanager
def _coesistenza_gpu(attivo: bool = True):
    """26/9: mentre il training tiene la GPU, il proprio pid entra in REGISTRO_COESISTENZA — l'UNICO canale
    che raggiunge il teacher gia' vivo: un processo in esecuzione non puo' ricevere un nuovo GPU_ESCLUDI_PID
    nel suo ambiente (rilancerebbe l'unita', vietato), ma `gpu_occupata.sh` e' un file bash riletto FRESCO ad
    OGNI chiamata (sia dal teacher che dal training): editarlo, o il registro che legge, vale da subito senza
    riavviare nessuno. Rimosso all'uscita (anche in errore): la guardia torna severa non appena il training
    finisce. `flock` sul file `.lock` accanto: 'misura' e 'allena' in sequenza non si accavallano scrivendo."""
    if not attivo or os.environ.get("E2E_SENZA_COESISTENZA") == "1":
        # 27/9: senza registrazione il teacher vede la GPU occupata e resta IN PAUSA (voluto: righe gia' pronte per ~96 passi)
        yield
        return
    REGISTRO_COESISTENZA.parent.mkdir(parents=True, exist_ok=True)
    lock = REGISTRO_COESISTENZA.with_suffix(".lock")

    def _aggiorna(aggiungi: bool) -> None:
        with open(lock, "w") as fl:
            fcntl.flock(fl, fcntl.LOCK_EX)
            try:
                vivi = [int(x) for x in REGISTRO_COESISTENZA.read_text().split()]
            except OSError:
                vivi = []
            vivi = [p for p in vivi if p != os.getpid() and Path(f"/proc/{p}").exists()]
            if aggiungi:
                vivi.append(os.getpid())
            REGISTRO_COESISTENZA.write_text("\n".join(map(str, vivi)) + ("\n" if vivi else ""))
            fcntl.flock(fl, fcntl.LOCK_UN)

    _aggiorna(True)
    try:
        yield
    finally:
        _aggiorna(False)


def _guardia_gpu(attendi_s: int) -> None:
    # 26/9 sera: piu' i PID passati a mano (GPU_ESCLUDI_PID) — una misura accanto a EB2 mentre il teacher e' in pausa
    # 27/9: il teacher si esclude SOLO se la coesistenza e' accesa. Con E2E_SENZA_COESISTENZA=1 escluderlo era la
    # meta' mancante della pausa: alle 10:07:51 EB2 e' ripartito SOPRA il blocco 27 del teacher (preso nel buco del
    # riavvio) e alle 10:10:37 il kernel ha resettato la coda di calcolo. Ora aspetta che il blocco finisca (o fallisce
    # rumoroso dopo `attendi_s`), come per qualunque altro processo GPU.
    teacher = _pid_teacher_attivo() if os.environ.get("E2E_SENZA_COESISTENZA") != "1" else []
    esclusi = [os.getpid()] + teacher + [int(x) for x in os.environ.get("GPU_ESCLUDI_PID", "").split()
                                         if x.isdigit()]
    env = dict(os.environ, GPU_ESCLUDI_PID=" ".join(str(p) for p in esclusi))
    r = subprocess.run(["bash", str(GUARDIA), "--attendi", str(attendi_s)], capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"GPU occupata: {r.stdout.strip()[:500]}")


def _chi_sulla_gpu() -> str:
    """28/9 (sentinella): chi ALTRO tiene la GPU adesso, secondo la guardia unica (kfd + DRM fdinfo: Vulkan compreso),
    escluso SOLO questo processo (anche i pid di GPU_ESCLUDI_PID si vedono: e' la diagnosi del blocco). Solo registro."""
    env = dict(os.environ, GPU_ESCLUDI_PID=str(os.getpid()))
    r = subprocess.run(["bash", str(GUARDIA)], capture_output=True, text=True, env=env, timeout=60)
    return "nessuno" if r.returncode == 0 else " ".join(r.stdout.split())[:600]


def scegli_gdn(log: Callable = print) -> str:
    """E2E_GDN = ripiego | fla | veloce (27/9/2026): quale GDN usano i 30 strati Gated DeltaNet, scelto PRIMA del
    primo avanti. Default «ripiego» = comportamento identico a prima (nessun import, nessuna sostituzione).
      veloce  strumenti/gdn_veloce.py (stessa matematica fp32 del ripiego, riscritta: solve_triangular, ricorrenza
              in lotto con indietro a mano)
      fla     strumenti/fla_gfx1151 (kernel Triton di fla 0.5.2 in una COPIA fuori dal venv, ricetta gfx1151 a
              scratch zero); abilita_sicuro controlla contro fp64 e, se qualcosa non torna, lascia il ripiego.
    Il modulo di transformers cerca torch_chunk_gated_delta_rule a ogni chiamata: basta sostituirlo qui."""
    if os.environ.get("E2E_CONV_DIRETTA", "").strip() in ("1", "btc"):   # 2/10 TT6-veloce: conv GDN senza trasposte
        import e2e_conv_diretta as CD
        CD.installa()
        log(f"conv GDN: diretta (e2e_conv_diretta, fp64 come MIOpen; E2E_CONV_DIRETTA={os.environ['E2E_CONV_DIRETTA'].strip()}: "
            f"1 = al bit, btc = piu' veloce, dL/dtheta all'arrotondamento)")
    scelta = os.environ.get("E2E_GDN", "ripiego").strip().lower() or "ripiego"
    if scelta == "ripiego":
        log("GDN: ripiego di transformers (E2E_GDN=ripiego)")
        return scelta
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as M   # fla NON importabile nel venv: ripiego
    if scelta == "veloce":
        import importlib                                    # E2E_GDN_MODULO: copia congelata da usare (default v3 = 8ff7c3f)
        modulo = os.environ.get("E2E_GDN_MODULO", "gdn_veloce_corsa")
        importlib.import_module(f"strumenti.{modulo}").installa(M)
        log(f"GDN: veloce (strumenti/{modulo}.py) -> {M.torch_chunk_gated_delta_rule.__module__}."
            f"{M.torch_chunk_gated_delta_rule.__name__}")
        return scelta
    if scelta == "fla":
        sys.path.insert(0, str(QUI / "strumenti" / "fla_gfx1151"))
        import gdn_fla_gfx1151
        r = gdn_fla_gfx1151.abilita_sicuro(M)
        if not r["attivo"]:
            log(f"GDN: fla NON attivo ({r.get('motivo')}) -> resta il ripiego")
            return "ripiego"
        log(f"GDN: fla (copia {r['impronta_copia']}, controllo fp64 {json.dumps(r.get('controllo'))})")
        return scelta
    raise ValueError(f"E2E_GDN={scelta!r}: attesi ripiego | fla | veloce")


# Con fla i gradienti di A_log, dt_bias e in_proj_a escono 4,7-5,9x il ripiego (TECNICHE_E2E_PV.md, 14c8a6a): con
# queste famiglie F32 allenate serve E2E_GDN=veloce (v3) o ripiego.
_F32_DG_FLA = ("ssm_a", "ssm_dt", "ssm_alpha")


def controlla_gdn_f32(gdn: str, allena_f32: tuple) -> None:
    rotte = [f for f in allena_f32 if f in _F32_DG_FLA]
    if gdn == "fla" and rotte:
        raise ValueError(f"E2E_GDN=fla con --allena-f32 {','.join(rotte)}: gradienti sbagliati nei kernel fla, "
                         "usa E2E_GDN=veloce")


def _registro(dir_c: Path):
    def log(msg: str) -> None:
        SE.segno()                                     # 28/9: ogni riga di registro e' un'unita' di progresso
        riga = f"[{adesso()}] {msg}"
        print(riga, flush=True)
        with open(dir_c / "registro.log", "a") as f:
            f.write(riga + "\n")
    return log


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nome in ("allena", "g1"):
        p = sub.add_parser(nome)
        p.add_argument("--corsa", required=True)
        p.add_argument("--teacher", required=True)
        p.add_argument("--gguf", default=None)
        p.add_argument("--lr", type=float, default=ConfigAllena.lr)
        p.add_argument("--righe-passo", type=int, default=ConfigAllena.righe_per_passo)
        p.add_argument("--righe-micro", type=int, default=ConfigAllena.righe_per_micro)
        p.add_argument("--righe-micro-valuta", type=int, default=ConfigAllena.righe_per_micro_valuta)
        p.add_argument("--epoche", type=int, default=ConfigAllena.epoche)
        p.add_argument("--heldout-ogni", type=int, default=ConfigAllena.heldout_ogni)
        p.add_argument("--ripresa-ogni", type=int, default=ConfigAllena.ripresa_ogni)
        p.add_argument("--pezzo-kl", type=int, default=ConfigAllena.pezzo_kl)
        p.add_argument("--escludi-domini", default=",".join(ConfigAllena.escludi_domini))
        p.add_argument("--rotazione", action="store_true")
        p.add_argument("--riprendi", action="store_true")
        p.add_argument("--ferma-a", type=int, default=None)
        p.add_argument("--device", default="cuda")
        p.add_argument("--ricetta", default="B", help="B, B_ind32, C, B3, B3_ind32 (le sceglie F0b/M0/cancello B3)")
        p.add_argument("--attendi-gpu", type=int, default=600)
        p.add_argument("--K0", type=float, default=ConfigAllena.K0, help="scala di partenza dell'indietro fp16")
        p.add_argument("--riusa-passo0", default=None, help="cartella migliore_000000 di una corsa identica")
        p.add_argument("--cambia-ricetta", action="store_true",
                       help="27/9: la ripresa accetta una --ricetta diversa da quella salvata se il formato e' lo stesso "
                            "(es. B_ind32 -> B_ind2 / B3_ind2); registrato in stato['cambi_ricetta']. Default: solleva")
        p.add_argument("--pv-codici", action="store_true", help="fase 2 PV: passo V sui codici ternari")
        p.add_argument("--pv-rho", type=float, default=ConfigAllena.pv_rho)
        p.add_argument("--pv-micro-curvatura", type=int, default=ConfigAllena.pv_micro_curvatura)
        p.add_argument("--pv-n-proposta", type=int, default=ConfigAllena.pv_n_proposta)
        p.add_argument("--pv-q-fdr", type=float, default=ConfigAllena.pv_q_fdr)
        p.add_argument("--pv-q-guadagno", type=float, default=ConfigAllena.pv_q_guadagno)
        p.add_argument("--pv-passi-accumulo", type=int, default=ConfigAllena.pv_passi_accumulo,
                       help="PV: finestra di N passi (statistiche sommate). Con --pv-modo vbop (1/10): decisione (selezione, "
                            "salti, sonda, punto medio, ricentro) ogni N passi, statistiche e EMA V-Bop a ogni passo")
        p.add_argument("--pv-modo", choices=("pv", "vbop"), default=ConfigAllena.pv_modo,
                       help="pv = passo V di prima (default); vbop = V-Bop a gruppo (stato persistente fra i passi)")
        p.add_argument("--pv-gamma", type=float, default=ConfigAllena.pv_gamma, help="V-Bop: EMA di m, v, H")
        p.add_argument("--pv-curvatura-ogni", type=int, default=ConfigAllena.pv_curvatura_ogni,
                       help="V-Bop: curvatura misurata ogni N passi V (in mezzo la sua EMA)")
        p.add_argument("--pv-rho-adattivo", action="store_true",
                       help="regione di fiducia vera: rho x2 se r > 0,75, /2 se r < 0,25 (r = calo vero / previsto)")
        p.add_argument("--pv-rho-max", type=float, default=ConfigAllena.pv_rho_max)
        p.add_argument("--pv-accetta", choices=("bh", "guadagno"), default=ConfigAllena.pv_accetta,
                       help="bh = due test Benjamini-Hochberg (default); guadagno = guadagno di 2o ordine stimato > 0, "
                            "senza BH (budget rho, sonda invariati)")
        p.add_argument("--pv-z-min", type=float, default=ConfigAllena.pv_z_min,
                       help="solo con --pv-accetta guadagno: in piu' z > soglia (t del gradiente sulla direzione)")
        p.add_argument("--pv-z-guadagno-min", type=float, default=ConfigAllena.pv_z_guadagno_min,
                       help="solo con --pv-accetta guadagno: in piu' t del guadagno > soglia (limite inferiore di "
                            "confidenza del guadagno > 0, senza BH)")
        p.add_argument("--pv-sonda-dosi", action="store_true",
                       help="(v-interazione-b) salti provati a dosi 1/256 1/64 1/16 1/4 1 in ordine di guadagno sulla "
                            "sonda A, dose migliore confermata sulla sonda B; la dose parziale tenuta non dimezza rho")
        p.add_argument("--pv-ricentra", action="store_true",
                       help="(v-interazione-b, con --pv-sonda-dosi) un indietro della sonda A nello stato raggiunto: "
                            "lascia-uno-fuori per salto, annulla i dannosi a meta' per volta finche' la sonda B conferma")
        p.add_argument("--pv-ricentra-giri", type=int, default=ConfigAllena.pv_ricentra_giri,
                       help="giri massimi di annullamento a meta' del ricentro (default 3)")
        p.add_argument("--pv-dosi-frazioni", default="",
                       help="(v-interazione-b, con --pv-sonda-dosi) frazioni crescenti delle dosi, p.es. "
                            "1/4096,1/1024,1/256,1/64,1/16,1/4,1 (vuoto = 1/256,1/64,1/16,1/4,1)")
        p.add_argument("--pv-punto-medio", action="store_true",
                       help="(punto-medio, con --pv-sonda-dosi) prezzi delle primitive al PUNTO MEDIO PROPRIO sulla sonda A "
                            "(arXiv 2609.09867; un indietro per primitiva + uno), dosi = primitive col prezzo < 0")
        p.add_argument("--pv-pm-primitiva", choices=("famiglia", "tipo", "strato", "esperto"),
                       default=ConfigAllena.pv_pm_primitiva, help="granularita' delle primitive (default famiglia)")
        p.add_argument("--pv-pm-max", type=int, default=ConfigAllena.pv_pm_max,
                       help="primitive prezzate per passo V (le prime per guadagno previsto; default 12)")
        p.add_argument("--pv-pm-non-filtrare", action="store_true",
                       help="tiene tutti i salti della primitiva scelta (default: solo quelli col prezzo al centro < 0)")
        p.add_argument("--pv-pm-lettere", action="store_true",
                       help="obiettivo = KL delle 4 lettere su righe mc del passo dopo, vincolo = sonda generica, "
                            "conferma dell'obiettivo su righe mc disgiunte (serve --righe-mc)")
        p.add_argument("--pv-pm-golosa", action="store_true",
                       help="(pv-lettere-dosi, con --pv-punto-medio) dosi golose: primitiva per primitiva, tenuta se il "
                            "funzionale obiettivo (le lettere su M1 con --pv-pm-lettere) scende e la sonda A resta <= "
                            "partenza; al posto dell'argmin della sonda generica sui prefissi")
        p.add_argument("--pv-riuso", action="store_true",
                       help="(tt3-cure) fase V a dosi: prefisso per strato riusato fra le prove, indietro troncato al "
                            "salto piu' basso, piani e scale riusati; stessi numeri al bit, meno lavoro")
        p.add_argument("--pv-riuso-lotti", type=int, default=ConfigAllena.pv_riuso_lotti,
                       help="micro-lotti tenuti dal riuso (~2,7 GB di VRAM l'uno a 8192 token; default 2)")
        p.add_argument("--pv-dd-solo-salti", action="store_true",
                       help="(2/10 tt5-fase-v V1) gradienti della fase V: niente dL/dd per le famiglie senza salti "
                            "preparati (andava solo nel gradiente di theta, buttato); stesse decisioni al bit")
        p.add_argument("--pv-pm-ordine-strati", action="store_true",
                       help="(2/10 V2, con --pv-punto-medio) g1 dei prezzi al punto medio in ordine di strato "
                            "decrescente: col riuso l'avanti riparte dallo strato della primitiva; stessi numeri")
        p.add_argument("--pv-golosa-pigra", action="store_true",
                       help="(2/10 V3, con --pv-pm-golosa) prova golosa: sonda A sopra la partenza -> primitiva "
                            "rifiutata senza misurare le lettere su M1; stesse decisioni")
        p.add_argument("--pv-tempi-v", action="store_true",
                       help="(2/10) secondi per sotto-fase del passo V nel registro (sincronizza ai confini)")
        p.add_argument("--pv-ricentra-obiettivo", action="store_true",
                       help="(tt3-cure, con --pv-ricentra --pv-pm-golosa --pv-pm-lettere) ogni giro del ricentro si "
                            "tiene solo se anche le lettere su M1 non salgono; lettere M2 misurate anche alla dose")
        p.add_argument("--perdita", choices=("kl", "cakld"), default=ConfigAllena.perdita,
                       help="kl = KL(teacher||studente) (default); cakld = BitDistiller Eq. 5")
        p.add_argument("--gamma", type=float, default=ConfigAllena.gamma,
                       help="gamma di CAKLD; assente = misurato (media di p_T(y vero) sulle righe del passo 0)")
        p.add_argument("--peso-token", choices=PESI_TOKEN, default=ConfigAllena.peso_token,
                       help="peso per token della perdita nel micro-lotto: uniforme (default, identico al bit); kl = "
                            "(l_t / media l)^alfa col tetto, a media 1; topk = solo la frazione dei token a perdita piu' "
                            "alta, riscalata (Rho-1); risposta = 1 ovunque, 1 + lambda alle posizioni-risposta delle "
                            "righe mc (--beta-risposta). Held-out e sonda del passo V restano uniformi")
        p.add_argument("--peso-alfa", type=float, default=ConfigAllena.peso_alfa, help="esponente di --peso-token kl")
        p.add_argument("--peso-max", type=float, default=ConfigAllena.peso_max,
                       help="tetto dei pesi di --peso-token kl (>= 1; la media resta 1)")
        p.add_argument("--peso-frazione", type=float, default=ConfigAllena.peso_frazione,
                       help="quota di token tenuti da --peso-token topk (Rho-1: 0,6)")
        p.add_argument("--teacher-mc", default=None,
                       help="27/9: cache del maestro sulle righe a scelta multipla (stesso formato di --teacher; "
                            "risposte.pt accanto al file del corpus del suo piano); lato heldout = held-out mc")
        p.add_argument("--teacher-mc-heldout", default=None,
                       help="cache del maestro sul held-out mc, se non e' il lato heldout di --teacher-mc")
        p.add_argument("--righe-mc", type=int, default=ConfigAllena.righe_mc,
                       help="righe mc per passo su --righe-passo (le altre dal teacher base, nell'ordine di sempre)")
        p.add_argument("--beta-risposta", type=float, default=ConfigAllena.beta_risposta,
                       help="--peso-token risposta: quota della massa dei pesi del passo alle posizioni-risposta")
        p.add_argument("--opd-beta", type=float, default=ConfigAllena.opd_beta,
                       help="(12/10 OPD, --peso-token generati) peso del termine di KL (paper: 1)")
        p.add_argument("--opd-kl", choices=("campionata", "piena", "eakld"), default=ConfigAllena.opd_kl,
                       help="KL dell'OPD: campionata (arXiv 2609.26708, fedele), piena (KL inversa esatta), eakld "
                            "(EdgeRazor arXiv 2605.04062, miscela avanti/inversa per entropia del maestro)")
        p.add_argument("--opd-eakld-k", type=int, default=ConfigAllena.opd_eakld_k, help="K di EdgeRazor (paper 16)")
        p.add_argument("--opd-pivot", default=ConfigAllena.opd_pivot,
                       help="PIVOT-Q (arXiv 2610.05745) 'H=4,gamma=0.9,fasi=4,m=4,sep=2,ancora=0.1'; vuoto = spento")
        p.add_argument("--gamma-lettere", type=float, default=ConfigAllena.gamma_lettere,
                       help="peso G della KL ristretta alle 4 lettere alle posizioni-risposta (0 = spenta)")
        p.add_argument("--righe-mc-permutate", action="store_true",
                       help="(integrata 27/9) anche il PRIMO giro delle righe mc permutato col seme (il corpus mc e' a "
                            "quarti di rotazione: nell'ordine del teacher le prime 192 righe sono tutte rotazione 0)")
        p.add_argument("--righe-base-da", type=int, default=ConfigAllena.righe_base_da, metavar="N",
                       help="(28/9, EB4) solo corsa NUOVA: il flusso base parte dalla riga N dell'ordine del teacher (le "
                            "prime N consumate da un'altra corsa: con 0 la corsa le aspetterebbe per sempre). 0 = al bit")
        p.add_argument("--peso-dominio", default="",
                       help="(dominio-strati 27/9) peso della perdita per DOMINIO della riga, pesi > 0, es. "
                            "\"it_chat=0.5,strumenti=0.75,edu=1.5,cot=1.5,it_web=1.5,it_wiki=1.5,it_pensiero=1.5\" (non "
                            "elencati 1; righe mc: dominio mc; normalizzati a media 1 sulle righe di ogni passo; x "
                            "--peso-token). Vuoto = spento, identico al bit")
        p.add_argument("--strati-allenati", default="",
                       help="(dominio-strati 27/9) strati allenati, es. 20-39 o 0-3,30-39: scale, F32, LoRA e codici V solo "
                            "li'; negli altri nessun gradiente (stato al bit) e l'indietro si ferma al primo allenato. "
                            "Vuoto = tutti, identico al bit")
        p.add_argument("--beta2", type=float, default=ConfigAllena.beta2,
                       help="beta2 di Adam; passi di r righe: 0,95^(r/32) (vedi ConfigAllena)")
        p.add_argument("--ema-theta", type=float, default=ConfigAllena.ema_theta,
                       help="decadimento della media esponenziale di theta (0 = spenta)")
        p.add_argument("--heldout-righe-rapide", type=int, default=ConfigAllena.heldout_righe_rapide,
                       help="righe della misura rapida ogni --heldout-ogni (0 = ogni misura e' piena)")
        p.add_argument("--heldout-pieno-ogni", type=int, default=ConfigAllena.heldout_pieno_ogni,
                       help="passi fra le misure piene (0 = --heldout-ogni): migliore e arresto solo su queste")
        p.add_argument("--allena-f32", default="",
                       help="famiglie F32 del GGUF da allenare, separate da virgola (vuoto = nessuna, identico al "
                            "bit): attn_norm, post_attention_norm, attn_q_norm, attn_k_norm, ssm_norm, ffn_gate_inp "
                            "(router), ffn_gate_inp_shexp, ssm_a, ssm_alpha, ssm_beta, ssm_conv1d, ssm_dt, output_norm")
        p.add_argument("--lr-f32", type=float, default=ConfigAllena.lr_f32,
                       help="lr del gruppo AdamW dei tensori F32 (default 1e-5, Norm Tweaking)")
        p.add_argument("--densi-ternari", default="",
                       help="(28/9) famiglie DENSE gia' ternarie nel GGUF (TQ1_0R_1024/TQ1_0) da allenare come esperto "
                            "unico: scale theta + codici del passo V, export come gli esperti. Famiglie o gruppi: attn "
                            "(attn_q,k,v,output), gdn (attn_qkv, attn_gate, ssm_out), shexp (ffn_gate/up/down_shexp), "
                            "tutte. Vuoto = spento, identico al bit")
        p.add_argument("--f32-ripristina", default="",
                       help="28/9: famiglie F32 RIPORTATE al valore del GGUF alla ripresa, in tutti gli strati (vuoto = "
                            "nessuna, identico al bit). Tolte da --allena-f32: restano al GGUF; ancora in --allena-f32: "
                            "ripartono dal GGUF con Adam da zero. Solo quando compaiono per la prima volta (idempotente)")
        p.add_argument("--lora-nonesperti", type=int, default=ConfigAllena.lora_rango, metavar="R",
                       help="rango degli adattatori LoRA sui k-quant non esperti (0 = spenta, identico al bit)")
        p.add_argument("--lora-alpha", type=float, default=None, help="alpha (s = alpha/R); default R (s = 1)")
        p.add_argument("--lora-famiglie", default="tutte",
                       help="famiglie o gruppi separati da virgola: attn (q,k,v,output), gdn (attn_qkv, attn_gate, "
                            "ssm_out), shexp (ffn_gate/up/down_shexp), tutte; o i nomi (attn_q, ssm_out, ...)")
        p.add_argument("--lr-lora", type=float, default=ConfigAllena.lr_lora,
                       help="lr del gruppo AdamW degli adattatori LoRA (default 1e-4)")
        p.add_argument("--ottimizzatore-theta", choices=("adamw", "sophia"), default=ConfigAllena.ottimizzatore_theta,
                       help="adamw = come sempre (default, identico al bit); sophia = Newton diagonale stile Sophia sulle "
                            "scale (arXiv 2305.14342) con la curvatura Gauss-Newton per gruppo dei micro-lotti di "
                            "curvatura del passo V (serve --pv-codici; F32/LoRA restano AdamW). TECNICHE_E2E_PV.md §36")
        p.add_argument("--sophia-gamma", type=float, default=ConfigAllena.sophia_gamma,
                       help="gamma di Sophia: theta -= lr clip(m / max(gamma h, eps), 1); per il 10-50%% di coordinate non "
                            "clippate (README di Sophia) serve gamma ~ quantili 0,1-0,5 di |m/h| (§36: >> 1 qui)")
        p.add_argument("--sophia-beta2", type=float, default=ConfigAllena.sophia_beta2,
                       help="EMA della curvatura di Sophia (0.99 dell'articolo)")
        p.add_argument("--sophia-eps", type=float, default=ConfigAllena.sophia_eps,
                       help="denominatore minimo di Sophia in unita' vere (x K)")
        p.add_argument("--lora-ripristina", choices=("zero", "ferma"), default="",
                       help="28/9: alla ripresa Delta W di OGNI adattatore LoRA torna a 0 (|Delta W| scartato nel registro). "
                            "zero = B = 0 (LoRA-GA: A = A0, B = B0), Adam della LoRA da zero, la LoRA continua ad allenarsi "
                            "(tipicamente con --lr-lora piu' basso); ferma = LoRA fuori dall'ottimizzatore finche' la riga "
                            "lo dice (export = senza LoRA). Solo quando compare (la stessa riga rilanciata non rifa' nulla)")
        p.add_argument("--lora-tipi", default="",
                       help="solo i tensori di questi tipi GGUF, es. Q8_0 (vuoto = tutti: Q8_0,Q4_0,Q4_K,Q6_K). §25 di "
                            "TECNICHE_E2E_PV.md: con Delta W sotto mezzo passo la riquantizzazione lo cancella")
        p.add_argument("--lora-init", choices=("zero", "ga"), default=ConfigAllena.lora_init,
                       help="zero = A casuale, B = 0 (default, identico al bit); ga = LoRA-GA (arXiv 2407.05000): A, B "
                            "dai primi 2R vettori singolari del gradiente pieno sulle righe del primo passo, modello "
                            "invariato al bit (TECNICHE §37)")
        p.add_argument("--lora-ga-gamma", type=float, default=ConfigAllena.lora_ga_gamma,
                       help="LoRA-GA: gamma della scala stable, c = out^(1/4)/sqrt(gamma) (default 16, codice ufficiale)")
        p.add_argument("--lora-ga-righe", type=int, default=ConfigAllena.lora_ga_righe,
                       help="LoRA-GA: righe del lotto di calibrazione (le prime del passo; 0 = tutte)")
        p.add_argument("--attribuzione", action="store_true",
                       help="(integrata 27/9) a ogni passo: sonda fissa del held-out prima/dopo, lascia-uno-fuori e solo-uno "
                            "per theta/F32/LoRA/codici (ripristino al bit), diagnostica e tempi per fase; riga JSON per "
                            "passo in <corsa>/registro_componenti.jsonl (strumenti/leggi_componenti.py)")
        p.add_argument("--attribuzione-righe", type=int, default=ConfigAllena.attribuzione_righe,
                       help="righe del held-out nella sonda dell'attribuzione (default 2 = 8.192 token)")
        p.add_argument("--attribuzione-solo-ogni", type=int, default=ConfigAllena.attribuzione_solo_ogni,
                       help="solo-uno ogni N passi (1 = sempre, 0 = mai: meta' delle valutazioni in meno)")
        p.add_argument("--non-consumare", action="store_true",
                       help="(integrata 27/9) con --rotazione NON cancellare i blocchi del teacher usati (teacher condiviso "
                            "con un'altra corsa)")
        p.add_argument("--scritture-asincrone", action="store_true",
                       help="1/10: ripresa e stato V-Bop scritti da un filo a parte (stessi file)")
        p.add_argument("--scritture-sosta-dispositivo", action="store_true",
                       help="2/10 (con --scritture-asincrone): copia della ripresa in VRAM, portata in RAM un tensore "
                            "alla volta dal filo dello scrittore; stessi file al bit")
        p.add_argument("--tempi-fasi", action="store_true",
                       help="1/10: tempi per fase a ogni passo nel registro (P, V, valutazioni, salvataggi)")
        p.add_argument("--istantanea-ogni-passo", action="store_true",
                       help="(integrata 27/9) istantanea esportabile passo_<N> (theta + ema/) a OGNI passo, per l'MMLU di "
                            "ogni passo (il migliore per metrica held-out non basta)")
        p.add_argument("--istantanee-dir", default=ConfigAllena.istantanee_dir,
                       help="cartella delle istantanee passo_<N> (vuoto = la cartella della corsa)")
        p.add_argument("--istantanee-tieni", type=int, default=ConfigAllena.istantanee_tieni,
                       help="quante istantanee passo_<N> tenere (le piu' recenti)")
        p.add_argument("--istantanee-libero-min-gib", type=float, default=ConfigAllena.istantanee_libero_min_gib,
                       help="GiB che devono restare liberi dopo l'istantanea, altrimenti la si salta (la corsa continua)")
        p.add_argument("--riprendi-da-ema", action="store_true",
                       help="(tt3-cure 29/9) con --riprendi: theta := theta_EMA della ripresa, momenti di Adam di theta "
                            "azzerati; F32/LoRA grezzi col loro Adam; una volta sola per corsa (ignorato se gia' fatto)")
        p.add_argument("--lora-reinit-ga", action="store_true",
                       help="(integrata 27/9) con --riprendi e --lora-init ga: se la LoRA della ripresa e' a init zero la "
                            "si scarta e la si re-inizializza con LoRA-GA alle righe del passo della ripresa (theta, codici, "
                            "F32, V-Bop, EMA, Adam di theta/F32 invariati; Adam della LoRA da zero); gia' GA: nessun effetto")
        p.add_argument("--lota-rango", type=int, default=ConfigAllena.lota_rango, metavar="R",
                       help="LoTA-QAF sui codici degli esperti: rango degli adattatori ternari (0 = spenta, identico al "
                            "bit; esclusiva con --pv-codici)")
        p.add_argument("--lota-omega", type=float, default=None, help="soglia omega in (0, R) (default 0,75 R, paper)")
        p.add_argument("--lota-top-p", type=float, default=ConfigAllena.lota_top_p,
                       help="quota iniziale delle voci di A e di B aggiornate per tensore (paper: 0,05)")
        p.add_argument("--lota-top-p-fine", type=float, default=ConfigAllena.lota_top_p_fine,
                       help="quota all'80%% dei passi (paper: 0,001)")
        p.add_argument("--lota-top-p-coda", type=float, default=ConfigAllena.lota_top_p_coda,
                       help="quota all'ultimo passo (paper: 0,0001)")
        p.add_argument("--lota-lr", type=float, default=ConfigAllena.lota_lr,
                       help="tasso della media del gradiente in (0, 1] (1 = t-SignSGD del paper, senza memoria)")
        p.add_argument("--lota-tau", type=float, default=ConfigAllena.lota_tau,
                       help="soglia minima sul |gradiente| vero (paper: 1e-9)")
        p.add_argument("--lota-senza-sonda", action="store_true",
                       help="niente regione di fiducia (i codici nuovi si tengono sempre)")
        p.add_argument("--lota-sonda-tolleranza", type=float, default=ConfigAllena.lota_sonda_tolleranza,
                       help="la sonda tiene i codici nuovi se la perdita sale al piu' di questa quota relativa (0 = mai)")
        p.add_argument("--lota-sonda-finestra", type=int, default=ConfigAllena.lota_sonda_finestra, metavar="N",
                       help="sonda FISSA a finestra: i codici si tengono sempre, dopo N passi senza battere il migliore "
                            "A e B tornano al migliore e p si dimezza (0 = sonda passo per passo)")
        p.add_argument("--lota-init", choices=("paper", "gradiente"), default=ConfigAllena.lota_init,
                       help="paper: A kaiming ternarizzata; gradiente: A = ternario di Omega^T G al primo passo LoTA "
                            "(schizzo dello spazio delle righe del gradiente, variante nostra alla LoRA-GA)")
    for p in (sub.choices["allena"], sub.choices["g1"], sub.add_parser("misura")):
        p.add_argument("--testa-kl", choices=("fp32", "bf16x3"), default=ConfigAllena.testa_kl,
                       help="logit e dL/dh dello studente: fp32 (default) o bf16x3 (e2e_bf16x3)")
        p.add_argument("--logit-maestro", choices=("fp32", "bf16"), default=ConfigAllena.logit_maestro,
                       help="logit del teacher: fp32 (default) o UNA GEMM bf16 con accumulo fp32")
    p = sub.choices["misura"]
    p.add_argument("--teacher", required=True)
    p.add_argument("--gguf", default=None)
    p.add_argument("--strati", action="append", required=True, help="es. 0  oppure 0,1 (ripetibile)")
    p.add_argument("--righe-micro", type=int, default=ConfigAllena.righe_per_micro)
    p.add_argument("--ripetizioni", type=int, default=3)
    p.add_argument("--ricetta", default="B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--uscita", default=str(RADICE / "misura_m0.json"))
    p.add_argument("--prime-disponibili", action="store_true",
                   help="le prime righe train presenti (blocchi rotanti: le prime del piano possono essere consumate)")
    p.add_argument("--densi-ternari", default="",
                   help="(28/9) misura con i densi ternari a esperto unico (come in allena); vuoto = di sempre")
    p = sub.add_parser("lancia")
    p.add_argument("--corsa", required=True)
    p.add_argument("resto", nargs=argparse.REMAINDER)
    p = sub.add_parser("stato")
    p.add_argument("--corsa", required=True)
    a = ap.parse_args(argv)

    if a.cmd == "stato":
        d = RADICE / a.corsa
        m = d / "ripresa_ultima.json"
        print(m.read_text() if m.exists() else "nessuna ripresa")
        e = d / "esito.json"
        if e.exists():
            print(e.read_text()[:4000])
        return 0
    if a.cmd == "lancia":
        resto = [x for x in a.resto if x != "--"]
        unita = f"e2e-allena-{a.corsa}"
        cmd = ["systemd-run", "--user", f"--unit={unita}", "--collect", "-p", "Nice=5", "-p", "Restart=on-failure",
               "-p", "RestartSec=60", "-p", "StartLimitBurst=3", "-p", "StartLimitIntervalSec=3600",
               "--working-directory", str(QUI), PYTHON, str(QUI / "e2e_allena.py"), "allena", "--corsa", a.corsa,
               "--riprendi"] + resto
        print(" ".join(cmd))
        return subprocess.run(cmd, env=dict(os.environ, XDG_RUNTIME_DIR="/run/user/1000",
                                            DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/1000/bus")).returncode
    import e2e_studente as S
    gguf = Path(a.gguf) if getattr(a, "gguf", None) else S.GGUF_THOR_T
    # 26/9: `_coesistenza_gpu` copre TUTTO l'uso GPU di questo processo (guardia inclusa), cosi' il registro
    # e' scritto PRIMA che qualunque controllo GPU parta e rimosso solo alla fine — vedi la funzione.
    with _coesistenza_gpu(a.device == "cuda"):
        if a.cmd == "misura":
            if a.device == "cuda":
                _guardia_gpu(600)
            gdn_scelto = scegli_gdn()
            ca = ConfigAllena(righe_per_micro=a.righe_micro, testa_kl=a.testa_kl, logit_maestro=a.logit_maestro,
                              densi_ternari=_famiglie_dense(getattr(a, "densi_ternari", "") or ""))
            strati = [[int(x) for x in s.split(",")] for s in a.strati]
            ris = misura(strati, Path(a.teacher), gguf, ca, a.ripetizioni, a.device, a.ricetta,
                         disponibili=a.prime_disponibili)
            ris["gdn"] = gdn_scelto
            conf = ris["configurazioni"]
            if all(k in conf for k in ("0", "0,1", "0,3")):
                gdn = conf["0,1"]["totale_ms_token"] - conf["0"]["totale_ms_token"]
                piena = conf["0,3"]["totale_ms_token"] - conf["0"]["totale_ms_token"]
                fisso = conf["0"]["totale_ms_token"] - gdn
                ris["proiezione_40"] = {"fisso_ms_token": fisso, "gdn_ms_token": gdn, "piena_ms_token": piena,
                                        "totale_ms_token": fisso + 30 * gdn + 10 * piena,
                                        "nota": "fisso = strato 0 meno uno strato GDN (KL, embedding, uscita); Adam "
                                                "scala coi parametri: stima, la misura vera e' il passo a 40 strati"}
            Path(a.uscita).parent.mkdir(parents=True, exist_ok=True)
            _scrivi_json(Path(a.uscita), ris)
            print(json.dumps(ris, indent=1))
            return 0
        ca = _config_da_arg(a)
        dir_c = RADICE / a.corsa
        dir_c.mkdir(parents=True, exist_ok=True)
        log = _registro(dir_c)
        if a.device == "cuda":
            _guardia_gpu(a.attendi_gpu)
        gdn = scegli_gdn(log)
        controlla_gdn_f32(gdn, ca.allena_f32)
        if shutil.disk_usage(dir_c).free < 16 * 2 ** 30:
            raise RuntimeError(f"{dir_c}: meno di 16 GiB liberi (riprese ~2,5 GiB, istantanee ~1,3 GiB)")
        cfg_json = {"config": asdict(ca), "argv": sys.argv, "gguf": str(gguf), "ricetta": a.ricetta,
                    "teacher": a.teacher, "versione": VERSIONE, "avvio": adesso()}
        if a.teacher_mc or a.teacher_mc_heldout:            # 27/9 trainer-mc (solo se dati: config.json di sempre)
            cfg_json.update({"teacher_mc": a.teacher_mc, "teacher_mc_heldout": a.teacher_mc_heldout})
        _scrivi_json(dir_c / "config.json", cfg_json)
        src = SorgenteCache(a.teacher)
        if ca.peso_token == "generati":                     # 12/10 OPD: posizioni generate e vantaggi accanto al corpus
            import e2e_onpolicy as _OPD
            src = _OPD.apri_sorgente_onpolicy(src, log)
        if a.teacher_mc or a.teacher_mc_heldout:
            src = apri_sorgente_mc(src, Path(a.teacher_mc) if a.teacher_mc else None,
                                   Path(a.teacher_mc_heldout) if a.teacher_mc_heldout else None, log)
        dev = torch.device(a.device)
        if ca.densi_ternari and ca.lota_rango > 0:
            raise ValueError("--densi-ternari con LoTA (--lota-rango): LoTA sa solo gli esperti; codici dei densi = --pv-codici")
        st = S.Studente(gguf=gguf, strati=range(40), ricetta=a.ricetta,
                        motore_nome=os.environ.get("E2E_MOTORE", "triton") if dev.type == "cuda" else "torch", device=a.device, log=log,
                        allena_f32=ca.allena_f32, lora=config_lora(ca),
                        **({"densi_ternari": ca.densi_ternari} if ca.densi_ternari else {}))
        vuote = [f for f, n in getattr(st, "f32_conteggi", {}).items() if n == 0]
        if vuote:
            raise ValueError(f"--allena-f32: famiglie senza tensori F32 nel GGUF {vuote}")
        vuote = [f for f, n in getattr(st, "densi_conteggi", {}).items() if n == 0]
        if vuote:
            raise ValueError(f"--densi-ternari: famiglie senza tensori ternari nel GGUF {vuote}")
        if ca.densi_ternari:
            log(f"--densi-ternari {','.join(ca.densi_ternari)}: {st.densi_conteggi} tensori a esperto unico (scale nel "
                f"gruppo theta" + (", codici col passo V" if ca.pv_codici else ", codici FERMI: manca --pv-codici") + ")")
        if ca.allena_f32:
            log(f"--allena-f32 {','.join(ca.allena_f32)}: {len(st.chiavi_f32())} tensori, lr_f32 {ca.lr_f32:g}")
        if ca.lora_rango:
            vuote = [f for f, n in st.lora_conteggi.items() if n == 0]
            if not st.lora or (vuote and not ca.lora_tipi):
                raise ValueError(f"--lora-famiglie: famiglie senza tensori nel GGUF {vuote} (tipi {ca.lora_tipi})")
            log(f"--lora-nonesperti {ca.lora_rango} (alpha {ca.lora_alpha}, famiglie {','.join(ca.lora_famiglie)}, "
                f"tipi {','.join(ca.lora_tipi) or 'tutti'}; per famiglia {st.lora_conteggi}): "
                f"{len(st.chiavi_lora())} adattatori, {sum(p.numel() for p in st.parametri_lora())} valori, "
                f"lr_lora {ca.lr_lora:g}" +
                (f"; init LoRA-GA (gamma {ca.lora_ga_gamma:g}, righe {ca.lora_ga_righe or ca.righe_per_passo}; "
                 f"accumulatori dW transitori {sum(a.forma[0] * a.forma[1] for a in st.lora.values()) * 4 / 2 ** 30:.2f} "
                 f"GiB fp32)" if ca.lora_init == "ga" else ""))
        if dev.type == "cuda":
            st.carica_embedding()
        WS, WT, sha_t = teste(st, dev, ca)
        log(f"teste: studente {tuple(WS.shape)} ({ca.testa_kl}), teacher bf16 originale sha256 {sha_t[:16]} "
            f"(logit {ca.logit_maestro})")
        if a.cmd == "g1":
            car = Caricatore(src, dev)
            righe = src.righe_previste("train")[ca.righe_base_da:ca.righe_base_da + ca.righe_per_passo]   # 28/9: da N
            attendi_righe(src, "train", righe, log)
            ga = None
            if ca.lora_init == "ga" and _parametri_lora(st):          # 27/9: G1 sul modello con la LoRA-GA vera
                ga = inizializza_lora_ga(st, car, "train", righe[:ca.lora_ga_righe] if ca.lora_ga_righe else righe,
                                         ca, WS, WT, ca.K0, log)
            g1 = cancello_g1(st, car, "train", righe, ca, WS, WT, log)
            if ga is not None:
                g1["lora_ga"] = ga
            car.chiudi()
            _scrivi_json(dir_c / "g1.json", g1)
            print(json.dumps(g1, indent=1, default=str)[:6000])
            return 0
        # 28/9 blocco di EB3 (curvatura del passo 21 ferma su un segnale GPU mai arrivato, GPU 0 %, ~25 min muti):
        # nessun progresso per E2E_SENTINELLA_S (default 1800) -> registro, pile e uscita 75 (e2e_sentinella.py)
        sec_s, marg_s = SE.secondi_da_env()
        with SE.sentinella(dir_c / "registro.log", sec_s if dev.type == "cuda" else 0.0, log,
                           controllo_gpu=_chi_sulla_gpu, margine=marg_s):
            stato = allena(st, src, dir_c, ca, WS, WT, riprendi=a.riprendi, ferma_a=a.ferma_a, log=log,
                           riusa_passo0=a.riusa_passo0, reinit_lora_ga=getattr(a, "lora_reinit_ga", False),
                           da_ema=getattr(a, "riprendi_da_ema", False),
                           cambia_ricetta=getattr(a, "cambia_ricetta", False))
        stato["teste"] = {"teacher_bf16_sha256": sha_t}
        stato["fine"] = adesso()
        _scrivi_json(dir_c / "esito.json", stato)
        log(f"FINE: passo {stato['passo']}, migliore al passo {stato['migliore']['passo']} "
            f"(metrica {stato['migliore']['metrica']:.5f}), istantanea {stato['migliore']['dir']}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
