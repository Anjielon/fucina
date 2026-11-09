#!/usr/bin/env python3
"""cardan_congiunto.py — JOINT CARDAN on C (18/10, feasibility prototype, NOT launched): ONE global realignment step
that moves the routed experts (CARDAN K=1 on gate/up, cardan_kd_paper.py) AND the dense tensors (continuous low-rank
branch on attn/gdn, rilq_thor_kl4.py --continuo) together, on the same loss, instead of the alternating block descent
S6 -> N3 -> D2 -> G3 ... Study: ~/tony-forgia/ricerche/CARDAN_CONGIUNTO_1018.md.

Nothing of the two trainers' mathematics is rewritten; everything is imported:
  student      e2e_studente.Studente (same tree QUI/recupero_7dabd6f/thor_v2, same E2E_* switches as both launchers)
  experts      cardan_kd_moduli (mlp.cardan hook, C_e D^T, A.1 basis, closed-form C init)
  dense        residui_ternari continuous mode (ConResiduo on the attn/gdn linears, B = 0, gamma = 1, file order)
  paper loss   cardan_paper_perdita.perdita_paper (arXiv 2609.21137 Eq. 3: tau^2 KL top-256 + lambda CE)
  KL4          gradienti_densi.pacchi_kl4 packing; the 4-way letter KL is the SAME expression as rilq_thor_kl4.py
  full KL      e2e_allena.KLFusa (optional, --kl-peso; 0 by default, see the study for why)
Joint loss (declared weighted sum): L = paper(1 row of the CARDAN_FEDELE wt103 corpus) + w4 * KL4(1 pack of DISJOINT
calibration questions) [+ wk * KL_full(1 row of q8-maestro)]. Both blocks receive the gradient of the whole L.
Modes:
  congiunto  every step: one backward of L, Adam step on BOTH blocks (two Adam, each block clipped on its own norm,
             exactly as each recipe clips its own parameters)
  alternato  step k: only one block requires grad and steps (densi on odd steps, experts on even by default); the other
             block is frozen in the forward; --adam-cpu-inattivo moves the frozen block's Adam state to the host
Selection: held-out KL4 (disjoint questions, never trained) by default, or paper held-out; a best step is accepted
only if KL24 (q8-maestro held-out, NOT in the loss when --kl-peso 0) did not grow more than --guardia-kl24.
Export (same files the existing folds read, so the folds run unchanged): cardan_migliore.pt -> cardan_piega.py,
residui_delta.pt -> cardan_piega_densi.py; the two folds touch disjoint tensors (ffn_{gate,up}_exps vs attn/ssm).
Dry run on CPU: --prova-secco N (N tokens per row/pack, 1 held-out row/pack, one step, no cicli.tsv).

25/10 SINTESI_1025 (copy L2/codice; ~/tony-forgia/ricerche/SINTESI_1025/L2_L5_trainer.md). Three new switches, ALL OFF by
default (off = the code paths above, unchanged):
  --allinea-peso W      Lever 2: intermediate-state alignment on the paper rows (CKA-QAD 2606.05682, allinea_stati.py):
                        L = L_out + W * sg(L_out / (L_al + eps)) * L_al (--allinea-modo rapporto) or + W * L_al (fisso);
                        teacher states from a cattura_stati_maestro.py cache (--allinea-cache), gradient injected into
                        the student's own backward (no extra forward/backward). Works with --fuso.
  --kl4-rotazioni N     Lever 5: KL4 over N cyclic rotations of the answer options, teacher letter distribution recomputed
                        per rotation (kl4_rotazioni.py: rotated calibration + teacher MCLP logs in --kl4-rotazioni-dir).
  --paper-campionato K  Lever 5b: training KD term = unbiased K-sample estimate of the FULL KL (RS-KD 2503.16870,
                        paper_campionato.py) instead of the top-256 subset; held-out paper metric unchanged.
  (configuration errors of these switches exit with code 5: the launchers retry only on code 2)

25/10 SINTESI_1025 L3 (copy L3/codice; ~/tony-forgia/ricerche/SINTESI_1025/L3_scale_gate_up.md; motivated by R2 §3.5:
the folded CARDAN correction keeps only its projection on the scale directions, and 75-79 % of what training adds IS a
per-group scale change). Two switches, both OFF by default (off = every code path above, at the bit):
  --scale-gateup-lr LR  train theta of the routed-expert ffn_gate_exps / ffn_up_exps fp16 group scales (TQ1_0R_1024,
                        16 scales per 2048-wide stored/rotated row), scale = fp16(d0 * exp(theta)) exactly as
                        StratoEsperti._scala (theta fp32 for the whole run, rounded to fp16 only in the forward and at
                        export: sub-half-ulp updates accumulate). Own Adam block "scale_gu" clipped on its own norm.
                        Saved as scale_gateup_{ultimo,migliore}.pt and tappe/scale_gateup_<passo>.pt (theta + sha256 of
                        the fp16 scales the student actually used, for scale_gateup_esporta.py); Adam state optSgu.
  --rango-esperti 0     the gate/up low-rank CARDAN branch is NOT installed (no mlp.cardan module, no parameters, no
                        compute; cardan_*.pt are written with an empty state, rango 0). Only with --modo congiunto.
                        With both: trained == deployed for gate/up by construction (no fold of gate/up at all).

26/10 SINTESI_1025 L6 (copy L6/codice; ~/tony-forgia/ricerche/SINTESI_1025/L6_prior_lettere.md; motivated by R6 §2-§3,
§7.4a: the re-ternarised expert downs shift the answer-letter prior, the per-question KL4 fixed it but is memorised).
Three switches, the term is OFF by default (off = every code path above, at the bit):
  --prior-lettere-peso W     add W * KL(pbar_T || pbar_S) (prior_lettere.py): pbar_S = mean over the questions of the
                             step's batch of the student's softmax over the 4 letter logits at the answer position, pbar_T
                             = mean of the teacher KL4 targets of the same questions. Batch level (3 degrees of freedom),
                             never per question. Works with --kl4-peso 0 (the KL4 packs are always built), in the plain
                             path and in --fuso; with --kl4-peso > 0 both terms share the forward of the step's KL4 pack.
  --prior-lettere-pacchi N   the batch = N KL4 packs per step (pack offsets k*P//N, k = 0..N-1, P = packs per epoch; the
                             first is the step's own KL4 pack). Plain path: N forwards whose recompute states are kept until
                             the one backward of the term; --fuso: [paper row | N packs] in one sequence.
  --prior-lettere-ema BETA   running estimate of pbar_S and pbar_T over steps (EMA), loss KL(rT || rS), gradient through
                             the current batch only (literal derivative: effective weight W (1 - BETA) on the batch, see
                             prior_lettere.py); state in ottimizzatori_*.pt, bitwise resume.
  Held-out diagnostic (only when the term is on): held_prior = KL of the letter marginal over the KL4 held-out questions,
  held_pS / held_pT; ALLARME line if it grows > 0.003 over the start (R6 §7.3 guard). Not used for selection.
  Also (all modes, no effect on runs without NaN): a NaN gradient norm in ANY block now skips the step; before, max()
  over the block norms hid a NaN that was not in the first block (live KLPPRIOR run, step 166: NaN in "scale").

26/10 SINTESI_1025 F32-E2E (copy F32/codice of L6/codice; ~/tony-forgia/ricerche/SINTESI_1025/ANTAGONISTA_1026.md §2;
f32_blocco.py). All switches OFF by default (off = every code path above, at the bit; test_f32/test_f32.py F*):
  --allena-f32 FAMS     train the F32 tensors of the GGUF of those families ("tutte" = the 11 F32 families of C: 301
                        tensors, 22.2 M values), as e2e_studente.TensoreF32 masters (Studente(allena_f32=...), the
                        --allena-f32 branch of e2e_allena used by S7). Own Adam blocks f32_norme / f32_router / f32_ssm,
                        each clipped on its own norm; --lr-f32-norme / --lr-f32-router / --lr-f32-ssm-a / --lr-f32-ssm;
                        --f32-log CLASSES: those classes trained in log space (w = sign(w0) exp(phi)). Saved as
                        f32_{ultimo,migliore}.pt and tappe/f32_<passo>.pt (the fp32 bytes to deploy, f32_esporta.py ->
                        patch_gguf.py: trained == deployed, no fold), Adam state "optF32" in ottimizzatori_*.pt.
                        Presence gate per family at start; gradient coverage per family at the first step (a family with
                        no gradient at all -> exit code 6: the launcher retries on the reference GDN/conv path).
  --rango-densi 0       the dense continuous low-rank branch is NOT installed (as --rango-esperti 0 for gate/up).
  --solo-f32            refuse (code 5) unless the F32 blocks are the ONLY trainable variables; at every evaluation
                        theta of every ternary tensor must be exactly 0 with no gradient (frozen at the bit), else code 7.
  --prior-lettere-guardia  R6 §7.3 diagnostic without the L6 term: held_prior / held_pS / held_pT on the KL4 held-out
                        and the ALLARME line if it grows > 0.003 over the start (never used for selection).

26/10 SINTESI_1025 BPP = arm "B++" (copy BPP/codice of F32/codice; the two never-trained families of the T400 model map:
the 301 F32 tensors and the SHARED EXPERT, ~126 M ternary values active on every token). One new switch, OFF by default
(off = every code path above, at the bit; test_bpp/test_bpp.py B*):
  --residui-shexp       the dense continuous low-rank branch (residui_ternari continuous mode, ConResiduo, B = 0, gamma = 1,
                        A from the same seeded Gaussian as the dense ones) ALSO on ffn_gate_shexp / ffn_up_shexp /
                        ffn_down_shexp of every --strati layer, same rank (--rango-densi, must be > 0) and same lr
                        (--lr-densi / --lr-gamma). Own Adam block "densi_shexp" (optDsh), clipped on its own norm; it steps
                        with the "densi" block in --modo alternato. Identity gate (start = host at the bit) + presence gate
                        (gate/up/down of the first and last layer). Latents saved IN residui_*.pt next to the dense ones
                        (same file, same format: residui_delta.pt -> cardan_piega_densi_gpu_veloce.py folds them as any
                        other dense tensor, the LoRA ceiling exporter writes them as any other dense LoRA). --riprendi from
                        a checkpoint without them (B+): they start at their init (B = 0) with a fresh Adam.
  --router-cammino W    lever L1 "R-CAMMINO" of ricerche/SINTESI_1025/R7_instradamento.md (router_distilla.py): on every
                        router layer (default: all cached layers the student trains), on the student's OWN propagated
                        state, W * sum_l lambda_l mean_t [KL_U(pT || pS) + beta * M]: KL on the decision zone (teacher
                        top-K_T ∪ student top-K_S + one "rest" bucket, --router-cammino-zona 16,8) + soft mass at risk on the
                        teacher ranks a-b (--router-cammino-ranghi 4-9, --router-cammino-beta 1, --router-cammino-tau 0 =
                        0.25 rms(zS - zT) per layer); lambda_l uniform or --router-cammino-lambda FILE (JSON {layer: w}).
                        Teacher router logits from a cattura_router_maestro.py cache (--router-cache, fp16/fp32, bf16
                        refused); student logits recomputed in a forward hook of every MoEStudente (its own expression):
                        gradient into router_w (F32 master of ffn_gate_inp under --allena-f32) and UPSTREAM into every
                        trained block. --router-strati, --router-token N, --router-parziale (rows without cache: no term),
                        --router-cammino-perdita kl_piena (full-softmax KL, comparison only). Logged: router (sum_l
                        lambda_l term_l), router_ov8 / router_massa (R7 §4.1), held_router* on the held-out paper rows; never
                        part of the selection. With --allinea-peso the ratio weight uses L_out = paper + KL4. Not with
                        --prior-lettere-peso.
  --forza-scelte-maestro CACHE_PAPER,CACHE_KL24   "prova 0" of R7 §5 (no training): measures KL24 and paper held-out with the
                        student free and with the teacher's top-8 IMPOSED at every MoE layer (MoEStudente.instr: gates = the
                        student's softmax renormalised over the teacher's set), writes forza_scelte_maestro.json and exits 0.
                        Caches: cattura_router_maestro.py of the held-out rows of --teacher (paper) and --teacher-kl24 (KL24),
                        full row length (--token 4096); "no" skips one (e.g. no,CACHE_KL24). Works after --riprendi.
  F32 (unchanged switches) now also: --allena-f32 with --fuso / --allinea-peso (f32_blocco.errore_config), and --riprendi
                        from a checkpoint without f32_ultimo.pt / optF32 (B+): the F32 start at the host GGUF values with a
                        fresh Adam (before: exit 4). f32_ultimo.pt present without optF32: values loaded, fresh Adam.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))      # also under python -I
# 24/10 (velocita'): cache persistente di Triton/Inductor (niente ricompilazioni a ogni ripartenza) e limite di
# ricompilazione alzato (col default 8 _prep di gdn_veloce ricadeva in eager per le lunghezze di segmento nuove).
_CC = Path("@@ROOT:models@@/gguf/tony-tern/OBIETTIVO_C/CACHE_COMPILA_1024")
(_CC / "triton").mkdir(parents=True, exist_ok=True); (_CC / "inductor").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TRITON_CACHE_DIR", str(_CC / "triton")); os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(_CC / "inductor"))
import torch._dynamo as _dyn                                                  # noqa: E402
_dyn.config.recompile_limit = 64
if hasattr(_dyn.config, "accumulated_recompile_limit"):
    _dyn.config.accumulated_recompile_limit = max(_dyn.config.accumulated_recompile_limit, 256)
import cardan_kd_paper as KDP            # noqa: E402  path setup (QUI, REC, E2E_KERNEL_NUOVI) + helpers, unchanged
QUI, REC = KDP.QUI, KDP.REC
import residui_ternari as RT             # noqa: E402
import cardan_kd_moduli as CK            # noqa: E402
import cardan_paper_perdita as PP        # noqa: E402
import allinea_stati as AS               # noqa: E402  25/10 Lever 2 (inert unless --allinea-peso > 0)
import kl4_rotazioni as KR               # noqa: E402  25/10 Lever 5 (inert unless --kl4-rotazioni > 1)
import paper_campionato as PC            # noqa: E402  25/10 Lever 5b (inert unless --paper-campionato > 0)
import prior_lettere as PL               # noqa: E402  26/10 L6 (inert unless --prior-lettere-peso > 0)
import f32_blocco as FB                  # noqa: E402  26/10 F32-E2E (inert unless --allena-f32)
import router_distilla as RDm            # noqa: E402  26/10 BPP (inert unless --router-cammino > 0 / --forza-scelte-maestro)

FAM_GU = ("gate", "up")                  # 25/10 L3: routed-expert families whose scales --scale-gateup-lr trains
TEACHER_WT = "@@ROOT:models@@/gguf/tony-tern/CARDAN_FEDELE/teacher/q8-maestro-wt103"
TEACHER_Q8 = "@@ROOT:depot@@/e2e_teacher/q8-maestro"
CALIB = "@@ROOT:models@@/gguf/tony-tern/SELRISP/calib"


def argomenti(argv=None):
    ap = argparse.ArgumentParser()
    a_ = ap.add_argument
    a_("--uscita", required=True); a_("--gguf", required=True); a_("--nome", default="C-congiunto")
    a_("--teacher", default=TEACHER_WT, help="paper-loss corpus cache (CARDAN_FEDELE wt103)")
    a_("--teacher-kl24", default=TEACHER_Q8, help="KL24 guard cache (+ full-KL rows if --kl-peso > 0)")
    a_("--strati-studente", default="0-39"); a_("--strati", default="0-39")
    # experts (cardan_kd_paper.py recipe of S5)
    a_("--rango-esperti", type=int, default=16, help="rank of the gate/up CARDAN branch; 0 = branch not installed (L3)")
    a_("--base-a1", default=None)
    a_("--init-c", choices=("zero", "a1"), default="a1")
    a_("--frazione-c", type=float, default=0.005); a_("--frazione-d", type=float, default=0.001)
    a_("--lr-c", type=float, default=None); a_("--lr-d", type=float, default=None)
    # dense (rilq_thor_kl4.py --continuo recipe of DENSI2, arm thor)
    a_("--famiglie", default="attn,gdn"); a_("--rango-densi", type=int, default=16)
    a_("--lr-densi", type=float, default=1e-5); a_("--lr-gamma", type=float, default=1e-4)
    # loss
    a_("--tau", type=float, default=1.0); a_("--lambda-ce", type=float, default=0.5); a_("--topk", type=int, default=256)
    a_("--kl4-peso", type=float, default=1.0); a_("--kl-peso", type=float, default=0.0)
    a_("--kl4-calib", default=f"{CALIB}/calib_mmlu.jsonl"); a_("--kl4-maestro", default=f"{CALIB}/mclp_maestro.log")
    a_("--kl4-n", type=int, default=896)
    # schedule
    a_("--modo", choices=("congiunto", "alternato"), default="congiunto")
    a_("--primo", choices=("densi", "esperti"), default="densi", help="alternato: block of step 1")
    a_("--adam-cpu-inattivo", action="store_true")
    a_("--passi", type=int, default=300); a_("--riscaldamento", type=int, default=10)
    a_("--clip", type=float, default=1.0); a_("--heldout-righe", type=int, default=24)
    # 24/10 (Angelo, fedele al paper): WSD di arXiv 2609.32902 §3 «holds the peak rate and then applies a cosine cooldown
    # over the final fraction f of training, with f=0.1» (Hu et al. 2024; Hagele et al. 2024: decadimento fino a 0).
    a_("--wsd-frazione", type=float, default=0.0, help="f del cooldown a coseno finale (0 = lr costante, come prima)")
    a_("--tappe-dir", default=None, help="24/10: copia cardan/residui di ogni passo valutato in DIR/{cardan,residui}_<passo>.pt")
    a_("--heldout-ogni", type=int, default=25); a_("--pazienza", type=int, default=3)
    # --selezione (21/10, Angelo): kl24_paper = held KL24 + errore paper (criterio di selezione); mmlu = nessuna scelta ne' arresto
    a_("--selezione", choices=("kl4", "paper", "mmlu", "kl24_paper"), default="kl4"); a_("--guardia-kl24", type=float, default=0.02)
    a_("--pezzo-kl", type=int, default=512); a_("--ore-max", type=float, default=12.0)
    a_("--vram-tetto", type=float, default=88.0, help="GiB: stop after step 1 if the peak is above")
    a_("--kl-partenza-motore", type=float, default=None); a_("--soglia-coerenza", type=float, default=0.03)
    a_("--scale-down-lr", type=float, default=0.0, help="24/10 distillazione delle scale dei down esperti (theta log-scala, RoleQ 2610.21808 §scale distillation); 0 = spento"); a_("--seme", type=int, default=20261007); a_("--riprendi", action="store_true")
    # 25/10 CARDAN low-rank on the routed expert DOWN (cardan_kd_moduli.CardanDown, README_DOWN.md): 0 = off (default,
    # byte-identical behaviour). dW_down_e = C_e D_down^T, D_down [512, r] shared per layer, C_e [2048, r] init 0.
    a_("--down-lowrank-rango", type=int, default=0, help="rank r of the down low-rank correction (0 = off)")
    a_("--down-lowrank-lr", type=float, default=None, help="Adam lr of C_down (default: the gate/up C lr)")
    a_("--down-lowrank-lr-d", type=float, default=None, help="Adam lr of D_down (default: --frazione-d * rms(D_down))")
    a_("--base-a1-down", default=None, help="directory of cardan_costruisci_down.py (strato_LL.npz): D_down init; "
       "absent/missing layer = Gaussian basis")
    a_("--down-lowrank-strati", default=None, help="layers of the down correction (default: --strati)")
    # 25/10 SINTESI_1025 L3: 0 = off (default, byte-identical behaviour)
    a_("--scale-gateup-lr", type=float, default=0.0, help="Adam lr of theta of the routed-expert ffn_gate_exps/"
       "ffn_up_exps fp16 group scales, scale = fp16(d0 * exp(theta)) (theta fp32 for the whole run); 0 = off")
    a_("--device", default="cuda"); a_("--ricetta", default="B3_ind2")
    a_("--prova-secco", type=int, default=0, help="CPU dry run: N tokens per row/pack, one step")
    a_("--fuso", action="store_true", help="18/10: paper row + KL4 pack in ONE forward/backward (segment mode, the "
       "token mixer runs per segment, MoE/norms on all tokens together): same loss, same gradients up to fp order")
    a_("--prova-fuso", type=int, default=0, help="N steps: gradients and time, separate vs fused, then exit (no Adam)")
    # 25/10 SINTESI_1025 Lever 2 (allinea_stati.py): 0 = off (default, byte-identical behaviour)
    a_("--allinea-peso", type=float, default=0.0, help="weight W of the intermediate-state alignment (0 = off)")
    a_("--allinea-strati", default="tutti", help="layer outputs to align (e.g. 0-39, 10-29); tutti = all cached layers")
    a_("--allinea-cache", default=None, help="teacher-state cache of cattura_stati_maestro.py")
    a_("--allinea-metrica", choices=("cka", "mse_rel"), default="cka", help="cka = 1 - linear CKA (paper); mse_rel = Thor")
    a_("--allinea-modo", choices=("rapporto", "fisso"), default="rapporto",
       help="rapporto = W * sg(L_out / (L_al + eps)) (CKA-QAD); fisso = W")
    a_("--allinea-eps", type=float, default=1e-8)
    a_("--allinea-token", type=int, default=0, help="aligned tokens per row (0 = all the cached ones)")
    a_("--allinea-salta-inizi", action="store_true", help="Thor option: drop the first token of every segment")
    a_("--allinea-parziale", action="store_true", help="rows missing from the cache: no alignment term (else: error)")
    # 25/10 SINTESI_1025 Lever 5 (kl4_rotazioni.py, paper_campionato.py): 1 / 0 = off (default)
    a_("--kl4-rotazioni", type=int, default=1, help="N cyclic rotations of the answer options in KL4 (1 = off)")
    a_("--kl4-rotazioni-dir", default=None, help="calib_mmlu_rot<k>.jsonl + mclp_maestro_rot<k>.log, k = 1..N-1")
    a_("--kl4-rotazioni-modo", choices=("ciclo", "tutte"), default="ciclo",
       help="ciclo: one rotation per question per step, cycling over epochs; tutte: all N in every pack")
    a_("--paper-campionato", type=int, default=0, help="K samples per token: unbiased sampled KD (RS-KD); 0 = top-k paper")
    # 26/10 SINTESI_1025 L6 (prior_lettere.py): 0 = off (default, byte-identical behaviour)
    a_("--prior-lettere-peso", type=float, default=0.0, help="W of the batch-level answer-letter prior term "
       "W * KL(pbar_T || pbar_S) over the questions of the step's KL4 packs (0 = off)")
    a_("--prior-lettere-pacchi", type=int, default=1, help="N KL4 calibration packs per step forming the batch of the "
       "prior term (the first is the step's own KL4 pack)")
    a_("--prior-lettere-ema", type=float, default=0.0, help="beta of the running (EMA) estimate of pbar_S / pbar_T over "
       "steps, gradient through the current batch only (0 = batch estimate)")
    # 26/10 SINTESI_1025 F32-E2E (f32_blocco.py): "" / off = byte-identical behaviour
    a_("--allena-f32", default="", help="F32 families of the GGUF to train (comma list of e2e_studente.FAMIGLIE_F32 / "
       "output_norm, or 'tutte' = the 11 F32 families of C); empty = off")
    a_("--lr-f32-norme", type=float, default=1e-4, help="Adam lr of the norms (attn, post_attention, q/k, ssm, output)")
    a_("--lr-f32-router", type=float, default=3e-5, help="Adam lr of ffn_gate_inp and ffn_gate_inp_shexp")
    a_("--lr-f32-ssm-a", type=float, default=1e-4, help="Adam lr of ssm_a (log space by default: phi = A_log)")
    a_("--lr-f32-ssm", type=float, default=3e-5, help="Adam lr of ssm_dt (bias) and ssm_conv1d")
    a_("--f32-log", default="norme,ssm_a", help="F32 classes trained in log space, w = sign(w0) exp(phi) ('' = all linear)")
    a_("--solo-f32", action="store_true", help="the F32 blocks must be the only trainable variables (code 5 otherwise); "
       "frozen check of every theta at every evaluation (code 7)")
    a_("--prior-lettere-guardia", action="store_true", help="R6 §7.3 held-out letter-prior diagnostic + ALLARME line "
       "without the L6 term (logged only)")
    # 26/10 SINTESI_1025 BPP: off = byte-identical behaviour
    a_("--residui-shexp", action="store_true", help="dense continuous low-rank residuals also on ffn_gate_shexp / "
       "ffn_up_shexp / ffn_down_shexp (rank --rango-densi, lr --lr-densi / --lr-gamma, own Adam block densi_shexp)")
    a_("--router-cammino", type=float, default=0.0, help="W of R-CAMMINO (R7 L1, router_distilla.py): W * sum_l lambda_l "
       "[KL_U + beta * mass at risk] on the student's own path, gradient upstream; 0 = off")
    a_("--router-cammino-perdita", choices=("cammino", "kl_piena"), default="cammino")
    a_("--router-cammino-zona", default="16,8", help="K_T,K_S: decision zone = teacher top-K_T ∪ student top-K_S")
    a_("--router-cammino-ranghi", default="4-9", help="teacher ranks a-b of the mass-at-risk term")
    a_("--router-cammino-beta", type=float, default=1.0, help="weight of the mass-at-risk term")
    a_("--router-cammino-tau", type=float, default=0.0, help="sigmoid temperature; 0 = 0.25 rms(zS - zT) per layer")
    a_("--router-cammino-lambda", default=None, help="JSON {layer: weight} (normalised over the router layers)")
    a_("--forza-scelte-maestro", default="", help="CACHE_PAPER,CACHE_KL24: prova 0 (R7 §5), free vs teacher-forced top-8 "
       "KL24/paper held-out, then exit")
    a_("--router-cache", default=None, help="teacher router-logit cache of cattura_router_maestro.py")
    a_("--router-strati", default="tutti", help="layers of the router term (default: cached layers the student trains)")
    a_("--router-token", type=int, default=0, help="tokens per row of the router term (0 = all the cached ones)")
    a_("--router-parziale", action="store_true", help="rows missing from the router cache: no router term (else: error)")
    return ap.parse_args(argv)


def taglia(lot, n: int):
    """dry run only: first n tokens of a row (ids, hT, inizi stay aligned)."""
    if n <= 0 or lot.ids.shape[1] <= n:
        return lot
    lot.ids = lot.ids[:, :n].contiguous(); lot.hT = lot.hT[:n].contiguous()
    lot.inizi = None if lot.inizi is None else lot.inizi[:, :n].contiguous()
    return lot


def bersagli(lot) -> tuple:
    """= cardan_kd_paper.bersagli: next token inside each segment."""
    ids = lot.ids.reshape(-1); T = ids.shape[0]
    ok = torch.ones(T, dtype=torch.bool, device=ids.device); ok[-1] = False
    if lot.inizi is not None:
        ok[:-1] &= lot.inizi.reshape(-1).to(ids.device)[1:] == 0
    tgt = torch.zeros(T, dtype=torch.long, device=ids.device); tgt[:-1] = ids[1:]
    return tgt, ok


def pacco_tensori(p, dev):
    """= rilq_thor_kl4 packing of one KL4 pack: ids, segment starts, last prompt token of each question."""
    ids = torch.tensor([sum((q["ids"] for q in p), [])], dtype=torch.long, device=dev)
    inizi = torch.zeros((1, ids.shape[1]), dtype=torch.long); ult, o = [], 0
    for q in p:
        inizi[0, o] = 1; o += len(q["ids"]); ult.append(o - 1)
    return ids, inizi, ult


def logp_lettere(hh, p, ult, WS):
    """Student log-softmax over the 4 letter logits at the last prompt token of each question of the pack, [Q, 4]
    (26/10 L6: split out of kl4_di with the very same operations, so kl4_di is unchanged at the bit)."""
    let = torch.tensor(p[0]["lettere"], device=hh.device)
    Wl = (WS.hi[let].float() + WS.lo[let].float()) if hasattr(WS, "hi") else WS[let].to(torch.float32)
    return torch.log_softmax(hh[torch.tensor(ult, device=hh.device)].float() @ Wl.T, -1)


def pT_di(p, dev):
    """Teacher 4-letter distributions of the pack's questions, [Q, 4] fp32 (= the KL4 targets)."""
    return torch.stack([q["pT"] for q in p]).to(dev).float()


def kl4_di(hh, p, ult, WS):
    """= rilq_thor_kl4: mean over the pack of KL(pT4 || softmax(letter logits of the student))."""
    lS = logp_lettere(hh, p, ult, WS)
    pT = pT_di(p, hh.device)
    return (pT * (torch.log(pT.clamp_min(1e-30)) - lS)).sum(-1)


def sposta_stato(opt, dev) -> None:
    for s in opt.state.values():
        for k, v in s.items():
            if torch.is_tensor(v) and v.device != dev:
                s[k] = v.to(dev)


def main(argv=None) -> int:
    a = argomenti(argv)
    if a.heldout_ogni > 0 and a.passi % a.heldout_ogni:   # 22/10 (Angelo, second time 540 instead of 550): the last step must be a
        a.passi += a.heldout_ogni - a.passi % a.heldout_ogni   # saved, evaluated tappa -> round --passi UP to a multiple of --heldout-ogni
        print(f"--passi rounded up to {a.passi} (multiple of --heldout-ogni {a.heldout_ogni})", flush=True)
    secco = a.prova_secco > 0
    if secco:
        a.heldout_righe, a.passi = 1, (2 if a.modo == "alternato" else 1)
    import e2e_studente as S
    import e2e_allena as A
    # 24/10: dL/dd veloce per famiglie non ruotate a pochi gruppi (down TQ1_0, NG=2): dd[e,r,g] = sum_t gy[t,r] * (x_g q_g^T)[t,r]
    # = NG prodotti in avanti del motore con scala indicatrice del gruppo + somma per esperto. Verificato contro l'originale
    # alla prima chiamata (errore relativo <= 1e-3, altrimenti resta l'originale).
    _DD = {"ric": None, "ok": None}
    _dd_orig = S.MotoreTriton.dd
    def _dd_veloce(self, gy, x, fam, piano, d_dtype):
        if getattr(fam, "ruotata", False) or fam.NG > 4 or _DD["ric"] is None or _DD["ok"] is False:
            return _dd_orig(self, gy, x, fam, piano, d_dtype)
        offs = piano.offs
        cnt = (offs[1:] - offs[:-1]).long()
        eid = torch.repeat_interleave(torch.arange(fam.E, device=gy.device), cnt, output_size=gy.shape[0])   # niente sync host
        out = torch.zeros(fam.E, fam.R, fam.NG, dtype=torch.float32, device=gy.device)
        for g in range(fam.NG):
            dm = torch.zeros(fam.E, fam.R, fam.NG, dtype=fam.d0.dtype, device=gy.device); dm[..., g] = 1
            yg = self.avanti(x, dm.to(_DD["dt"]), fam, piano, _DD["ric"]).float()
            out[:, :, g].index_add_(0, eid, gy.float() * yg)
        if _DD["ok"] is None:
            ref = _dd_orig(self, gy, x, fam, piano, torch.float32)
            err = float((out - ref).norm() / ref.norm().clamp_min(1e-30))
            _DD["ok"] = err <= 1e-3
            print(f"dd veloce: errore relativo {err:.2e} -> {'attivo' if _DD['ok'] else 'SPENTO, resta originale'}", flush=True)
            if not _DD["ok"]:
                return ref.to(d_dtype)
        return out.to(d_dtype)
    S.MotoreTriton.dd = _dd_veloce
    # 24/10 T5 (ACCELERA_INDIETRO_MOE_1021): dX di gate e up in UN lancio del kernel dX su K = 2R,
    # dX = [gg, gu] [Wg; Wu]. Solo ricetta a prodotto singolo (indietro_diviso vuoto) e motore triton2. Verificato alla
    # prima chiamata contro dx_u + dx_g (errore relativo <= 1e-3, altrimenti resta l'originale).
    import dataclasses as _dc
    _GU = {"ok": None}
    _bw_orig = S.FamiglieGateUp.backward
    def _bw_fuso(ctx, gg, gu):
        x, dg, du = ctx.saved_tensors
        nx, ng_, nu = ctx.needs_input_grad[0], ctx.needs_input_grad[1], ctx.needs_input_grad[2]
        mot, ric, fg, fu = ctx.mot, ctx.ric, ctx.fg, ctx.fu
        if (not nx or _GU["ok"] is False or ric.indietro_diviso or not getattr(mot, "v2", False)
                or fg.tipo != fu.tipo or fg.NBR != fu.NBR):
            return _bw_orig(ctx, gg, gu)
        _, ddu = S._indietro_famiglia(gu, x, du, fu, ctx.piano, ric, mot, None, False, nu) if nu else (None, None)
        _, ddg = S._indietro_famiglia(gg, x, dg, fg, ctx.piano, ric, mot, None, False, ng_) if ng_ else (None, None)
        di = ric.dtype_indietro or x.dtype
        f2 = _dc.replace(fg, blocchi=torch.cat([fg.blocchi, fu.blocchi], 1), d0=torch.cat([fg.d0, fu.d0], 1))
        d2 = torch.cat([dg, du], 1)
        a2 = torch.cat([gg.to(di), gu.to(di)], 1).contiguous()
        if ctx.oss is not None:                          # come _indietro_famiglia: guardie fp16 del gradiente e di dX
            ctx.oss.gradiente(fu, gu, gu.to(di)); ctx.oss.gradiente(fg, gg, gg.to(di))
        dx = S._dx2_triton(a2, f2, d2, ctx.piano, a2.dtype, mot.blk).to(x.dtype)
        if ctx.oss is not None:
            ctx.oss.guarda(dx)
        if _GU["ok"] is None:                            # prova: originale e fuso contro un riferimento fp32 (ingressi fp32)
            r = _bw_orig(ctx, gg, gu)[0]
            ref = (S._dx2_triton(gu.float().contiguous(), fu, du, ctx.piano, torch.float32, mot.blk)
                   + S._dx2_triton(gg.float().contiguous(), fg, dg, ctx.piano, torch.float32, mot.blk))
            nr = ref.norm().clamp_min(1e-30)
            e_or = float((r.float() - ref).norm() / nr); e_fu = float((dx.float() - ref).norm() / nr)
            e_tra = float((dx.float() - r.float()).norm() / nr)
            _GU["ok"] = e_fu <= max(1.5 * e_or, 1e-3)
            print(f"gate+up fusi: vs fp32 originale {e_or:.2e} fuso {e_fu:.2e} (fra loro {e_tra:.2e}) -> {'attivo' if _GU['ok'] else 'SPENTO'}", flush=True)
            if not _GU["ok"]:
                dx = r
        return dx, ddg, ddu, None, None, None, None, None, None
    S.FamiglieGateUp.backward = staticmethod(_bw_fuso)
    # 24/10 T6 (ACCELERA_INDIETRO_MOE_1021): SwiGLU a = silu(g)*u avanti e indietro in UN kernel Triton ciascuno
    # (prima: autograd di silu e del prodotto, 6-10 elementari su S x 512 fp32). Verificato alla prima chiamata.
    import triton, triton.language as tl
    @triton.jit
    def _k_sw_av(g_ptr, u_ptr, a_ptr, n, BL: tl.constexpr):
        i = tl.program_id(0) * BL + tl.arange(0, BL); m = i < n
        g = tl.load(g_ptr + i, mask=m).to(tl.float32); u = tl.load(u_ptr + i, mask=m).to(tl.float32)
        tl.store(a_ptr + i, g * tl.sigmoid(g) * u, mask=m)
    @triton.jit
    def _k_sw_in(g_ptr, u_ptr, d_ptr, dg_ptr, du_ptr, n, BL: tl.constexpr):
        i = tl.program_id(0) * BL + tl.arange(0, BL); m = i < n
        g = tl.load(g_ptr + i, mask=m).to(tl.float32); u = tl.load(u_ptr + i, mask=m).to(tl.float32)
        d = tl.load(d_ptr + i, mask=m).to(tl.float32); sg = tl.sigmoid(g)
        tl.store(dg_ptr + i, d * u * sg * (1 + g * (1 - sg)), mask=m); tl.store(du_ptr + i, d * g * sg, mask=m)
    class _SwiGLU(torch.autograd.Function):
        @staticmethod
        def forward(ctx, g, u):
            g, u = g.contiguous(), u.contiguous(); a = torch.empty_like(g); n = g.numel()
            _k_sw_av[(triton.cdiv(n, 1024),)](g, u, a, n, BL=1024); ctx.save_for_backward(g, u); return a
        @staticmethod
        def backward(ctx, d):
            g, u = ctx.saved_tensors; d = d.contiguous(); dg = torch.empty_like(g); du = torch.empty_like(u); n = g.numel()
            _k_sw_in[(triton.cdiv(n, 1024),)](g, u, d, dg, du, n, BL=1024); return dg, du
    _SW = {"ok": None}
    class _Pigro:
        __slots__ = ("g",)
        def __init__(self, g): self.g = g
        def __mul__(self, u):
            g = self.g
            if _SW["ok"] is False or g.dtype != u.dtype or g.shape != u.shape or not g.is_cuda:
                return torch.nn.functional.silu(g) * u
            if _SW["ok"] is None:
                with torch.enable_grad():
                    g1 = g.detach().clone().requires_grad_(True); u1 = u.detach().clone().requires_grad_(True)
                    g2 = g.detach().clone().requires_grad_(True); u2 = u.detach().clone().requires_grad_(True)
                    r = torch.nn.functional.silu(g1) * u1; f = _SwiGLU.apply(g2, u2); w = torch.randn_like(r)
                    (r * w).sum().backward(); (f * w).sum().backward()
                e = max(float((f - r).norm() / r.norm().clamp_min(1e-30)), float((g2.grad - g1.grad).norm() / g1.grad.norm().clamp_min(1e-30)),
                        float((u2.grad - u1.grad).norm() / u1.grad.norm().clamp_min(1e-30)))
                _SW["ok"] = e <= 1e-5
                print(f"SwiGLU fuso: errore relativo {e:.2e} -> {'attivo' if _SW['ok'] else 'SPENTO'}", flush=True)
                if not _SW["ok"]:
                    return torch.nn.functional.silu(g) * u
            return _SwiGLU.apply(g, u)
    class _FProxy:
        def __getattr__(self, k): return getattr(torch.nn.functional, k)
        @staticmethod
        def silu(g, *a, **kw): return _Pigro(g) if not a and not kw else torch.nn.functional.silu(g, *a, **kw)
    S.F = _FProxy()
    if os.environ.get("E2E_RIUSO") == "1":            # 21/10: riuso piani/scale di e2e_studente (gia' scritto, al bit; spento)
        S.riuso_attivo(True)
    dense_fuori = os.environ.get("E2E_DENSE_FUORI", "").strip()
    if dense_fuori:                                   # 22/10 (DENSE_FUORI_CICLO_1022.md): dense projections out of the
        import cardan_segmenti_gpu as SG              # per-segment loop; unset = original e2e_studente mixer, untouched
        if dense_fuori not in ("1", "lotto", "riempito"):
            raise ValueError(f"E2E_DENSE_FUORI={dense_fuori!r}: expected 1 | lotto | riempito")
        SG.imposta(S, "riempito" if dense_fuori == "riempito" else "lotto")
    sys.path.insert(0, "@@ROOT:models@@/build/leve/raffina")
    import gradienti_densi as GD
    out = Path(a.uscita); out.mkdir(parents=True, exist_ok=True)
    flog = open(out / "registro.log", "a")

    def log(*m):
        r = f"[{KDP.adesso()}] " + " ".join(str(x) for x in m)
        print(r, flush=True); flog.write(r + "\n"); flog.flush()

    if dense_fuori:
        log(f"E2E_DENSE_FUORI={dense_fuori}: segment mixer = cardan_segmenti_gpu."
            f"{'riempito' if dense_fuori == 'riempito' else 'lotto'} (dense projections on all tokens at once)")
    if a.rango_esperti < 0 or a.scale_gateup_lr < 0:     # 25/10 L3: configuration errors exit with code 5
        log("ERRORE: --rango-esperti e --scale-gateup-lr non possono essere negativi"); return 5
    if a.rango_esperti == 0 and a.modo == "alternato":
        log("ERRORE: --rango-esperti 0 (ramo gate/up non installato) vuole --modo congiunto"); return 5
    if a.rango_densi < 0 or (a.rango_densi == 0 and a.modo == "alternato"):   # 26/10 F32: 0 = dense branch not installed
        log("ERRORE: --rango-densi >= 0, e 0 (ramo denso non installato) vuole --modo congiunto"); return 5
    f32_fam = ()                                       # 26/10 F32-E2E: () = off, original paths
    if a.allena_f32.strip():
        try:
            f32_fam = FB.famiglie(a.allena_f32)
        except ValueError as e:
            log(f"ERRORE: --allena-f32: {e}"); return 5
        err = FB.errore_config(a, f32_fam)
        if err is not None:
            log(f"ERRORE: {err}"); return 5
    elif a.solo_f32:
        log("ERRORE: --solo-f32 senza --allena-f32"); return 5
    if a.residui_shexp and a.rango_densi <= 0:          # 26/10 BPP: the shexp residuals share the dense rank
        log("ERRORE: --residui-shexp vuole --rango-densi > 0 (stesso rango dei densi)"); return 5
    if a.residui_shexp and a.solo_f32:
        log("ERRORE: --residui-shexp non si combina con --solo-f32"); return 5
    if a.router_cammino < 0 or (a.router_cammino > 0 and (not a.router_cache or a.prior_lettere_peso != 0
                                                          or a.prova_fuso)):
        log("ERRORE: --router-cammino W >= 0; W > 0 vuole --router-cache, non si combina con --prior-lettere-peso "
            "/ --prova-fuso"); return 5
    try:                                               # 26/10 BPP: R-CAMMINO shape parameters (used only when W > 0)
        zona_r = tuple(int(x) for x in a.router_cammino_zona.split(","))
        ranghi_r = tuple(int(x) for x in a.router_cammino_ranghi.split("-"))
        if len(zona_r) != 2 or min(zona_r) < 1 or len(ranghi_r) != 2 or not 1 <= ranghi_r[0] <= ranghi_r[1] \
                or a.router_cammino_beta < 0 or a.router_cammino_tau < 0:
            raise ValueError("forma")
    except ValueError:
        log("ERRORE: --router-cammino-zona K_T,K_S (>= 1), --router-cammino-ranghi a-b (1 <= a <= b), beta e tau >= 0")
        return 5
    PR = None                                         # 26/10 L6: answer-letter prior term (None = off, original paths)
    if a.prior_lettere_peso != 0:
        err = PL.PriorLettere.errore_config(a.prior_lettere_peso, a.prior_lettere_pacchi, a.prior_lettere_ema)
        if err is None and a.allinea_peso > 0:
            err = "--prior-lettere-peso non si combina con --allinea-peso in questa copia"
        if err is not None:
            log(f"ERRORE: {err}"); return 5
        PR = PL.PriorLettere(a.prior_lettere_peso, a.prior_lettere_pacchi, a.prior_lettere_ema)

    def cicli(stato, testo):
        if not secco:
            KDP.riga_cicli(stato, f"CARDAN-CONGIUNTO {a.nome}: {testo}")

    t_avvio = time.time()
    random.seed(a.seme); np.random.seed(a.seme); torch.manual_seed(a.seme)
    dev = torch.device(a.device)
    json.dump({"argv": sys.argv if argv is None else argv, "avvio": KDP.adesso(), "codice": str(QUI)},
              open(out / ("config_ripresa.json" if a.riprendi else "config.json"), "w"), indent=1)
    A.scegli_gdn(log)
    try:
        kw_f32 = {"allena_f32": f32_fam} if f32_fam else {}    # 26/10 F32: absent = the call of always
        st = S.Studente(gguf=Path(a.gguf), strati=KDP.intervallo(a.strati_studente), ricetta=a.ricetta,
                        motore_nome=os.environ.get("E2E_MOTORE", "torch" if dev.type == "cpu" else "triton"),
                        device=a.device, log=log, densi_ternari="tutte", **kw_f32)
    except ValueError as e:
        log(f"ERRORE: {e}"); cicli("ERRORE", f"lo studente non e' il GGUF ({str(e)[:200]})"); return 2
    for p in st.parametri():
        p.requires_grad_(False)
    WS = A.testa_studente(st, "bf16x3")
    WT, sha_t = A.testa_maestro(dev, torch.bfloat16)
    src_w = A.SorgenteCache(a.teacher); car_w = A.Caricatore(src_w, dev)
    src_q = A.SorgenteCache(a.teacher_kl24); car_q = A.Caricatore(src_q, dev)
    train_w, held_w = src_w.righe_previste("train"), src_w.righe_previste("heldout")[: a.heldout_righe]
    train_q, held_q = src_q.righe_previste("train"), src_q.righe_previste("heldout")[: a.heldout_righe]
    import types as _ty
    ns = dict(calib_maestro=a.kl4_maestro, calib_jsonl=a.kl4_calib, seme=7, calib_pacco=4096)
    pacchi4, _ = GD.pacchi_kl4(_ty.SimpleNamespace(calib_n=a.kl4_n, **ns), log)
    usate4 = {q["i"] for p in pacchi4 for q in p}
    tutte4, _ = GD.pacchi_kl4(_ty.SimpleNamespace(calib_n=0, **ns), log)
    held4, cur, nt = [], [], 0                         # = rilq_thor_kl4: held-out = questions never trained on
    for q in (q for p in tutte4 for q in p if q["i"] not in usate4):
        if cur and nt + len(q["ids"]) > 4096:
            held4.append(cur); cur, nt = [], 0
        cur.append(q); nt += len(q["ids"])
    if cur:
        held4.append(cur)
    if secco:                                          # dry run: packs cut to N tokens, one held-out pack
        def corto(p):
            o, n = [], 0
            for q in p:
                if o and n + len(q["ids"]) > a.prova_secco:
                    break
                o.append(q); n += len(q["ids"])
            return o
        pacchi4, held4 = [corto(p) for p in pacchi4[:1]], [corto(held4[0])]
    log(f"corpus paper {a.teacher}: {len(train_w)} train / {len(held_w)} held; KL24 {a.teacher_kl24}: {len(held_q)} held;"
        f" KL4 {len(pacchi4)} pacchi train, {sum(map(len, held4))} domande held-out; testa maestro sha {sha_t[:16]}")
    rot4 = None
    if a.kl4_rotazioni > 1:                            # 25/10 Lever 5: rotated variants of the SAME training packs
        if a.kl4_peso <= 0 or not a.kl4_rotazioni_dir or a.kl4_rotazioni > 4:
            log("ERRORE: --kl4-rotazioni N (2..4) vuole --kl4-peso > 0 e --kl4-rotazioni-dir"); return 5
        rot4 = KR.Rotazioni.carica(pacchi4, a.kl4_rotazioni, a.kl4_rotazioni_modo, a.kl4_rotazioni_dir, GD.leggi_mclp, log)
        log(f"KL4 rotazioni: {a.kl4_rotazioni} ({a.kl4_rotazioni_modo}), maestro ricalcolato per rotazione da "
            f"{a.kl4_rotazioni_dir}; held-out KL4 resta sulla rotazione 0")

    def pacco_del_passo(passo: int) -> list:
        """The KL4 pack of a step: pacchi4 unchanged without rotations (same object as before)."""
        return pacchi4[(passo - 1) % len(pacchi4)] if rot4 is None else rot4.pacco(passo)

    held_pr = {}                                       # 26/10 L6: held-out letter marginals (filled by kl4_held)
    if PR is not None:
        if PR.n_pacchi > len(pacchi4):
            if not secco:
                log(f"ERRORE: --prior-lettere-pacchi {PR.n_pacchi} > {len(pacchi4)} pacchi KL4 per epoca"); return 5
            PR.n_pacchi = len(pacchi4)                   # dry run: the packs were cut to one
        accT = PL.Accumulo()
        for p_ in pacchi4:
            accT.aggiungi(torch.zeros(len(p_), 4, dtype=torch.float64), pT_di(p_, "cpu"))
        log(f"prior di lettera (L6): peso {PR.peso:g} x KL(pbar_T || pbar_S) per lotto, {PR.n_pacchi} pacchi KL4 per "
            f"passo (offset {PL.offsets(PR.n_pacchi, len(pacchi4))}, ~{accT.n * PR.n_pacchi / len(pacchi4):.0f} domande), "
            f"{'EMA beta ' + format(PR.beta, 'g') + ' (gradiente sul lotto corrente, peso effettivo W(1-beta))' if PR.beta > 0 else 'stima sul lotto'}"
            f"; KL4 peso {a.kl4_peso:g}; pbar maestro sulle {accT.n} domande di allenamento {accT.risultato()['pT']}")

    def prendi(car, lato, i):
        return taglia(car.prendi(lato, [i]), a.prova_secco)

    def perdita_paper(hd, lot, con_grad, passo=None):
        tgt, ok = bersagli(lot)
        if a.paper_campionato > 0 and con_grad:        # 25/10 Lever 5b: training only; the held-out metric stays top-k
            return PC.perdita_paper_campionata(
                hd, WS, lot.hT.reshape(hd.shape[0], -1), WT, tgt, ok, a.tau, a.lambda_ce, a.paper_campionato, a.pezzo_kl,
                con_grad, lambda h: A._logit_studente(h, WS), lambda h: A._logit_maestro(h, WT, torch.float32),
                lambda q: A._per_testa(q, WS), PC.generatore(hd.device, a.seme, passo or 0))
        return PP.perdita_paper(hd, WS, lot.hT.reshape(hd.shape[0], -1), WT, tgt, ok, a.tau, a.lambda_ce, a.topk,
                                a.pezzo_kl, con_grad, lambda h: A._logit_studente(h, WS),
                                lambda h: A._logit_maestro(h, WT, torch.float32), lambda q: A._per_testa(q, WS))

    def kl24() -> float:
        tot, n = 0.0, 0
        with torch.no_grad():
            for i in held_q:
                lot = prendi(car_q, "heldout", i); hS = st.nascosto(lot.ids, inizi=lot.inizi)
                for j in range(0, hS.shape[0], a.pezzo_kl):
                    r = A.kl_pezzo(hS[j:j + a.pezzo_kl], WS, lot.hT[j:j + a.pezzo_kl], WT, False)
                    tot += float(r["kl"].sum()); n += int(r["kl"].numel())
        return tot / n

    inj, src_al, strati_al, held_al = None, None, [], {}   # 25/10 Lever 2 (set below when --allinea-peso > 0)
    RD_, held_rd = None, {}                            # 26/10 BPP --router-cammino (set below when W > 0)

    def router_su(lot, lato: str, i: int):
        """26/10 BPP: router term on the paper row (lato, i) for the forward+backward inside; off / row not cached = no-op."""
        if RD_ is None or not RD_.src.disponibile(lato, i):
            return contextlib.nullcontext()
        return RD_.su(lato, i, lot.ids[0])

    def router_in_val(val: dict, somma_L: bool) -> None:
        """26/10 BPP: the router term of the step (when measured) into the logged values; somma_L adds W * router to L."""
        if RD_ is not None and RD_.ultimo is not None:
            val["router"] = RD_.ultimo
            for k_ in ("ov8", "massa", "kl", "m"):
                if k_ in RD_.ultimo_diag:
                    val[f"router_{k_}"] = RD_.ultimo_diag[k_]
            if somma_L:
                val["L"] += a.router_cammino * val["router"]

    def paper_held() -> float:
        tk, tc, nt_, nv = 0.0, 0.0, 0, 0
        held_rd.clear()
        if inj is not None:                             # 25/10 Lever 2: held-out alignment on the same forwards
            held_al.clear(); va = []
        with torch.no_grad():
            for i in held_w:
                lot = prendi(car_w, "heldout", i)
                if inj is not None:
                    inj.uscite = {}; inj.registra = True
                mis = RD_.misura("heldout", i, lot.ids[0]) if (RD_ is not None and RD_.src.disponibile("heldout", i)) \
                    else contextlib.nullcontext()          # 26/10 BPP: held-out router KL on the same forward
                try:
                    with mis:
                        hS = st.nascosto(lot.ids, inizi=lot.inizi)
                finally:
                    if inj is not None:
                        inj.registra = False
                if RD_ is not None and RD_.src.disponibile("heldout", i) and RD_.ultimo is not None:
                    held_rd.setdefault("v", []).append(RD_.ultimo)
                    held_rd.setdefault("d", []).append(RD_.ultimo_diag)
                kd, ce, T, n, _ = perdita_paper(hS, lot, False); tk += kd; tc += ce; nt_ += T; nv += n
                if inj is not None and src_al.disponibile("heldout", i):
                    Tm, n_ = stati_maestro_riga(lot, "heldout", i)
                    m_ = AS.maschera_inizi(lot.inizi, n_, dev) if a.allinea_salta_inizi else None
                    va.append(AS.valore_allineamento({L: inj.uscite[L][:n_] for L in strati_al}, Tm, a.allinea_metrica, m_))
        if inj is not None and va:
            held_al["allinea"] = sum(va) / len(va)
        return tk / nt_ + a.lambda_ce * tc / nv

    def stati_maestro_riga(lot, lato: str, i: int) -> tuple:
        """25/10 Lever 2: teacher states {layer: [n, H]} of row i, n = min(cached tokens, row tokens); the ids of the
        state cache must be the ids of the paper row (same corpus row, same order), checked every time."""
        Tm_, ids_c = src_al.stati(lato, i, strati_al, dev)
        n_ = min(Tm_.shape[1], lot.ids.shape[1], a.allinea_token or Tm_.shape[1])
        if not np.array_equal(ids_c[:n_], lot.ids[0, :n_].cpu().numpy()):
            raise RuntimeError(f"allinea: ids of the state cache differ from the paper row {lato} {i}")
        return {L: Tm_[k, :n_] for k, L in enumerate(strati_al)}, n_

    def kl4_held() -> float:
        tot, n = 0.0, 0
        acc = PL.Accumulo() if (PR is not None or a.prior_lettere_guardia) else None   # 26/10 L6 (+ F32: guard only)
        with torch.no_grad():
            for p in held4:
                ids, inizi, ult = pacco_tensori(p, dev)
                hS = st.nascosto(ids, inizi=inizi)
                hS = hS.reshape(-1, hS.shape[-1])
                tot += float(kl4_di(hS, p, ult, WS).sum()); n += len(p)
                if acc is not None:
                    acc.aggiungi(logp_lettere(hS, p, ult, WS), pT_di(p, dev))
        if acc is not None:
            held_pr.clear(); held_pr.update(acc.risultato())
        return tot / n

    def valuta() -> dict:
        v = {"kl24": kl24(), "paper": paper_held(), "kl4": kl4_held()}
        v["sel"] = (v["kl4"] if a.selezione == "kl4" else v["kl24"] + v["paper"] if a.selezione == "kl24_paper"
                    else v["paper"])
        if inj is not None and "allinea" in held_al:     # 25/10 Lever 2: logged only, never part of "sel"
            v["allinea"] = held_al["allinea"]
        if (PR is not None or a.prior_lettere_guardia) and held_pr:   # 26/10 L6: logged only, never part of "sel"
            v["prior"], v["pS"], v["pT"] = held_pr["prior"], held_pr["pS"], held_pr["pT"]
        if held_rd.get("v"):                             # 26/10 BPP: logged only, never part of "sel"
            v["router"] = sum(held_rd["v"]) / len(held_rd["v"])
            dd_ = held_rd["d"]
            for k_ in ("ov8", "massa"):
                v[f"router_{k_}"] = sum(d_[k_] for d_ in dd_) / len(dd_)
                v[f"router_{k_}_strati"] = {str(L): round(sum(d_["strati"][L][k_] for d_ in dd_) / len(dd_), 5)
                                            for L in dd_[0]["strati"]}
        return v

    # ── cancello d'identita' con ENTRAMBI i rami installati (C = 0, B = 0): h al bit
    lot = prendi(car_q, "heldout", held_q[0])
    with torch.no_grad():
        h0 = st.nascosto(lot.ids, inizi=lot.inizi).clone()
    strati = KDP.intervallo(a.strati)
    strati_sc = sorted(set(strati))                    # == sorted(resE) whenever the gate/up branch is installed
    # 25/10 L3: --rango-esperti 0 = no gate/up branch at all (no mlp.cardan module: the student's own forward)
    resE = CK.installa(st, strati, a.rango_esperti, Path(a.base_a1) if a.base_a1 else None, dev) \
        if a.rango_esperti > 0 else {}
    if a.rango_densi > 0:
        cfgD = RT.ConfigResidui(rango=a.rango_densi, piani=1, strati=strati, famiglie=RT.famiglie_da(a.famiglie),
                                seme=a.seme, continuo=True)
        resD = RT.installa(st, cfgD, device=dev)
    else:                                              # 26/10 F32: --rango-densi 0 = dense branch not installed
        cfgD, resD = None, {}
    resSh, init_sh, cfgSh = {}, {}, None               # 26/10 BPP: shared-expert residuals ({} = off, original paths)
    if a.residui_shexp:
        cfgSh = RT.ConfigResidui(rango=a.rango_densi, piani=1, strati=strati, famiglie=RT.famiglie_da("shexp"),
                                 seme=a.seme, continuo=True)
        resSh = RT.installa(st, cfgSh, device=dev)
        # the start of every shexp residual (B = 0, gamma = 1, seeded A): a checkpoint without them restores exactly this
        init_sh = {n: {k: v.detach().clone() for k, v in m.state_dict().items()} for n, m in resSh.items()}
    resDT = {**resD, **resSh}                          # 26/10 BPP: what residui_*.pt holds (== resD when off)
    resDn = {}
    if a.down_lowrank_rango > 0:                       # 25/10 down low-rank (C = 0): covered by the identity gate below
        strati_dn = KDP.intervallo(a.down_lowrank_strati) if a.down_lowrank_strati else strati
        resDn = CK.installa_down(st, strati_dn, a.down_lowrank_rango,
                                 Path(a.base_a1_down) if a.base_a1_down else None, dev)
    with torch.no_grad():
        h1 = st.nascosto(lot.ids, inizi=lot.inizi)
    ident = bool(torch.equal(h0, h1))
    nE = sum(p.numel() for p in CK.parametri(resE, "C") + CK.parametri(resE, "D"))
    nD = sum(p.numel() for p in RT.parametri(resD))
    log(f"cancello identita' (esperti C = 0 su {len(resE)} strati, {nE / 1e6:.2f} M; densi B = 0 su {len(resD)} "
        f"proiezioni, {nD / 1e6:.2f} M): {'AL BIT' if ident else 'DIVERSO'}")
    if not ident:
        log("ERRORE: la partenza non e' l'ospite al bit"); return 2
    for nm in ({sorted(resD)[0], sorted(resD)[-1]} if resD else ()):   # = rilq_thor_kl4 cancello di presenza (continuo)
        with torch.no_grad():
            resD[nm].B.normal_(0.0, 1e-3); h2 = st.nascosto(lot.ids, inizi=lot.inizi)
            resD[nm].B.zero_(); h3 = st.nascosto(lot.ids, inizi=lot.inizi)
        if torch.equal(h2, h1) or not torch.equal(h3, h1):
            log(f"ERRORE: ramo denso {nm} fuori dal percorso dello studente"); return 2
    log("cancello di presenza densi: ok" if resD else "ramo denso non installato (--rango-densi 0)")
    if resSh:                                          # 26/10 BPP: presence gate, every family on the first and last layer
        L_sh = RT.strati_di(resSh)
        for L_ in sorted({L_sh[0], L_sh[-1]}):
            for f_ in cfgSh.famiglie:
                nm = f"blk.{L_}.{f_}"
                with torch.no_grad():
                    resSh[nm].B.normal_(0.0, 1e-3); h2 = st.nascosto(lot.ids, inizi=lot.inizi)
                    resSh[nm].B.zero_(); h3 = st.nascosto(lot.ids, inizi=lot.inizi)
                if torch.equal(h2, h1) or not torch.equal(h3, h1):
                    log(f"ERRORE: residuo shexp {nm} fuori dal percorso dello studente"); return 2
        nSh = sum(p.numel() for p in RT.parametri(resSh))
        log(f"residui shexp: rango {a.rango_densi} su {len(resSh)} proiezioni ({', '.join(cfgSh.famiglie)}; strati "
            f"{L_sh[0]}-{L_sh[-1]}), {nSh / 1e6:.2f} M; cancello di presenza (strati {sorted({L_sh[0], L_sh[-1]})}): ok")
    if resDn:
        nDn = sum(p.numel() for p in CK.parametri_down(resDn, "C") + CK.parametri_down(resDn, "D"))
        for L_ in {sorted(resDn)[0], sorted(resDn)[-1]}:  # presence gate of the down branch (same rule as densi)
            with torch.no_grad():
                resDn[L_].C_down.normal_(0.0, 1e-3); h2 = st.nascosto(lot.ids, inizi=lot.inizi)
                resDn[L_].C_down.zero_(); h3 = st.nascosto(lot.ids, inizi=lot.inizi)
            if torch.equal(h2, h1) or not torch.equal(h3, h1):
                log(f"ERRORE: ramo down low-rank L{L_} fuori dal percorso dello studente"); return 2
        log(f"down low-rank: rango {a.down_lowrank_rango} su {len(resDn)} strati ({nDn / 1e6:.2f} M), base "
            f"{a.base_a1_down or 'gaussiana'}; cancello di presenza: ok")
    if not resE:
        log("ramo gate/up CARDAN non installato (--rango-esperti 0): gate/up = solo trit x scale del GGUF")
    if a.scale_gateup_lr > 0:                          # 25/10 L3: presence gate of the gate/up scales (same rule as densi)
        for L_ in sorted({strati_sc[0], strati_sc[-1]}):
            es_ = st.moduli[L_].mlp.esperti
            for f_ in FAM_GU:
                with torch.no_grad():
                    es_.theta[f_].normal_(0.0, 1e-3); h2 = st.nascosto(lot.ids, inizi=lot.inizi)
                    es_.theta[f_].zero_(); h3 = st.nascosto(lot.ids, inizi=lot.inizi)
                if torch.equal(h2, h1) or not torch.equal(h3, h1):
                    log(f"ERRORE: scale {f_} L{L_} fuori dal percorso dello studente"); return 2
        log(f"scale gate/up: cancello di presenza ok (strati {sorted({strati_sc[0], strati_sc[-1]})}, gate e up)")
    F32 = None                                         # 26/10 F32-E2E: None = off, original paths
    if f32_fam:
        try:
            F32 = FB.BloccoF32(st, {"norme": a.lr_f32_norme, "router": a.lr_f32_router, "ssm_a": a.lr_f32_ssm_a,
                                    "ssm": a.lr_f32_ssm}, FB.log_classi(a.f32_log))
        except ValueError as e:
            log(f"ERRORE: {e}"); return 5
        err_ = FB.cancello_presenza(st, F32, lambda: st.nascosto(lot.ids, inizi=lot.inizi), h1)
        if err_:
            for e_ in err_:
                log(f"ERRORE cancello di presenza F32: {e_}")
            return 2
        log(f"{F32.descrizione()}; cancello di presenza F32 (un tensore per famiglia): ok")
    del h0, h1, lot
    st.strati_allenati = sorted(set(strati) | set(RT.strati_di(resD)) | set(resDn)
                                | (set(RT.strati_di(resSh)) if resSh else set()))   # 26/10 BPP
    if a.allinea_peso > 0:                             # 25/10 Lever 2 (allinea_stati.py)
        if not a.allinea_cache:
            log("ERRORE: --allinea-peso > 0 senza --allinea-cache"); return 5
        if a.prova_fuso:
            log("ERRORE: --prova-fuso confronta i passi originali: non si combina con --allinea-peso"); return 5
        src_al = AS.SorgenteStati(a.allinea_cache)
        ok_l = set(src_al.strati) & set(st.strati)
        richiesti = sorted(ok_l) if a.allinea_strati == "tutti" else list(KDP.intervallo(a.allinea_strati))
        if not richiesti or any(L not in ok_l for L in richiesti):
            log(f"ERRORE: --allinea-strati {a.allinea_strati}: strati in cache e nello studente {sorted(ok_l)}"); return 5
        L0 = min(st.strati_allenati)                    # an output below the first trained layer moves no parameter
        strati_al = [L for L in richiesti if L >= L0]
        if len(strati_al) < len(richiesti):
            log(f"allinea: strati {sorted(set(richiesti) - set(strati_al))} sotto il primo allenato ({L0}): tolti")
        sp = src_al.piano
        if (Path(sp["sorgente"]).resolve() != Path(a.teacher).resolve()
                or sp["sorgente_piano_sha256"] != hashlib.sha256((Path(a.teacher) / "piano.json").read_bytes()).hexdigest()):
            log(f"ERRORE: la cache degli stati e' di {sp['sorgente']}, il corpus paper e' {a.teacher}"); return 5
        servono = {o for k, o in enumerate(AS.righe_in_ordine(train_w, a.seme)) if k < a.passi}
        mancano = sorted(i for i in servono if not src_al.disponibile("train", i))
        if mancano and not (a.allinea_parziale or secco):
            log(f"ERRORE: allinea: {len(mancano)} righe dei {a.passi} passi non sono nella cache (prima {mancano[0]}); "
                f"cattura_stati_maestro.py --n-righe {a.passi} o --allinea-parziale"); return 5
        inj = AS.Iniettore(st, strati_al, a.allinea_token or src_al.n_token)
        log(f"allinea: cache {a.allinea_cache} ({src_al.n_token} token/riga, {len(src_al.righe('train'))} train, "
            f"{len(src_al.righe('heldout'))} held), strati {strati_al[0]}..{strati_al[-1]} ({len(strati_al)}), metrica "
            f"{a.allinea_metrica}, modo {a.allinea_modo}, peso {a.allinea_peso:g}, eps {a.allinea_eps:g}, token "
            f"{inj.n}{', senza inizi di segmento' if a.allinea_salta_inizi else ''}; righe senza cache: {len(mancano)}")
    if a.router_cammino > 0:                           # 26/10 BPP (router_distilla.py, R7 L1)
        try:
            src_r = RDm.SorgenteRouter(a.router_cache)
        except (OSError, ValueError, KeyError, RuntimeError) as e:
            log(f"ERRORE: --router-cache {a.router_cache}: {e}"); return 5
        sp = src_r.piano
        if (Path(sp["sorgente"]).resolve() != Path(a.teacher).resolve()
                or sp["sorgente_piano_sha256"] != hashlib.sha256((Path(a.teacher) / "piano.json").read_bytes()).hexdigest()):
            log(f"ERRORE: la cache dei router e' di {sp['sorgente']}, il corpus paper e' {a.teacher}"); return 5
        ok_r = set(src_r.strati) & set(st.strati)
        L0r = min(st.strati_allenati)                   # a layer below the first trained one is never recomputed
        rich = sorted(ok_r) if a.router_strati == "tutti" else list(KDP.intervallo(a.router_strati))
        if not rich or any(L not in ok_r for L in rich):
            log(f"ERRORE: --router-strati {a.router_strati}: strati in cache e nello studente {sorted(ok_r)}"); return 5
        strati_r = [L for L in rich if L >= L0r]
        if not strati_r:
            log(f"ERRORE: --router-strati: nessuno strato >= {L0r} (primo strato allenato)"); return 5
        if src_r.n_expert != int(st.moduli[strati_r[0]].mlp.router_w.shape[0]):
            log(f"ERRORE: router della cache {src_r.n_expert} esperti, studente "
                f"{int(st.moduli[strati_r[0]].mlp.router_w.shape[0])}"); return 5
        servono_r = {o for k, o in enumerate(AS.righe_in_ordine(train_w, a.seme)) if k < a.passi}
        mancano_r = sorted(i for i in servono_r if not src_r.disponibile("train", i))
        if mancano_r and not (a.router_parziale or secco):
            log(f"ERRORE: router: {len(mancano_r)} righe dei {a.passi} passi non sono nella cache (prima {mancano_r[0]}); "
                f"--router-parziale per proseguire senza termine su quelle righe"); return 5
        try:
            lam_r = RDm.leggi_lambda(a.router_cammino_lambda) if a.router_cammino_lambda else None
            RD_ = RDm.DistillaRouter(st, src_r, strati_r, a.router_cammino, a.router_token, a.router_cammino_perdita,
                                     zona_r, a.router_cammino_beta, ranghi_r, a.router_cammino_tau, lam_r)
        except (OSError, ValueError) as e:
            log(f"ERRORE: router: {e}"); return 5
        log(f"router: R-CAMMINO ({a.router_cammino_perdita}) peso {a.router_cammino:g} su {len(strati_r)} strati "
            f"({strati_r[0]}..{strati_r[-1]}), zona top-{zona_r[0]} maestro ∪ top-{zona_r[1]} studente + resto, massa a "
            f"rischio ranghi {ranghi_r[0]}-{ranghi_r[1]} beta {a.router_cammino_beta:g} tau "
            f"{a.router_cammino_tau if a.router_cammino_tau > 0 else '0,25 rms(dz)'}, lambda "
            f"{'da ' + a.router_cammino_lambda if lam_r else 'uniforme'}; {RD_.n_max} token/riga, {src_r.n_expert} "
            f"esperti, cache {a.router_cache} ({src_r.piano.get('salva_tipo')}) "
            f"({len(src_r.righe('train'))} train, {len(src_r.righe('heldout'))} held); righe senza cache: {len(mancano_r)}")
    if a.paper_campionato > 0:
        log(f"paper campionato (RS-KD 2503.16870): K = {a.paper_campionato} campioni/token da p_maestro^(tau), stima non "
            f"distorta della KL piena al posto del top-{a.topk} (solo allenamento; held-out paper invariato)")
    pE = CK.parametri(resE, "C") + CK.parametri(resE, "D")
    pD = RT.parametri(resD)
    for p in pE + pD:
        p.requires_grad_(True)
    rms_d = float(torch.cat([p.detach().reshape(-1) for p in CK.parametri(resE, "D")]).float().pow(2).mean().sqrt()) \
        if resE else 0.0
    rms_c = None
    if a.base_a1 and resE:
        cc = []
        for L in sorted(resE):
            z = np.load(Path(a.base_a1) / f"strato_{L:02d}.npz")
            cc += [z[k].view(np.float16).astype(np.float32).reshape(-1) for k in z.files if k.endswith("_lrc.weight")]
        rms_c = float(np.sqrt(np.mean(np.concatenate(cc).astype(np.float64) ** 2)))
    lr_c = a.lr_c if a.lr_c is not None else a.frazione_c * (rms_c if rms_c else 1e-3)
    lr_d = a.lr_d if a.lr_d is not None else a.frazione_d * rms_d
    optE = torch.optim.Adam([{"params": CK.parametri(resE, "C"), "lr": lr_c},
                             {"params": CK.parametri(resE, "D"), "lr": lr_d}], betas=(0.9, 0.999), weight_decay=0.0)
    optD = torch.optim.Adam([{"params": [p for n in sorted(resD) for p in (resD[n].A, resD[n].B)], "lr": a.lr_densi},
                             {"params": [resD[n].gamma for n in sorted(resD)], "lr": a.lr_gamma}],
                            betas=(0.9, 0.999), weight_decay=0.0)
    blocchi = {"esperti": (optE, pE, [lr_c, lr_d]), "densi": (optD, pD, [a.lr_densi, a.lr_gamma])}
    if not resE:                                          # 25/10 L3: no gate/up branch -> no "esperti" block (optE stays,
        blocchi.pop("esperti")                            # empty, so the checkpoint layout is unchanged)
    if not resD:                                          # 26/10 F32: --rango-densi 0 -> no "densi" block (optD stays, empty)
        blocchi.pop("densi")
    pDsh, optDsh = [], None                               # 26/10 BPP: shexp residuals, own Adam (same lr as the dense)
    if resSh:
        pDsh = RT.parametri(resSh)
        for p in pDsh:
            p.requires_grad_(True)
        optDsh = torch.optim.Adam([{"params": [p for n in sorted(resSh) for p in (resSh[n].A, resSh[n].B)], "lr": a.lr_densi},
                                   {"params": [resSh[n].gamma for n in sorted(resSh)], "lr": a.lr_gamma}],
                                  betas=(0.9, 0.999), weight_decay=0.0)
        blocchi["densi_shexp"] = (optDsh, pDsh, [a.lr_densi, a.lr_gamma])
    pS, optS = [], None
    if a.scale_down_lr > 0:                               # 24/10: theta dei down esperti (scala = fp16(d0*exp(theta)))
        pS = [st.moduli[L].mlp.esperti.theta["down"] for L in strati_sc]
        optS = torch.optim.Adam([{"params": pS, "lr": a.scale_down_lr}], betas=(0.9, 0.999), weight_decay=0.0)
        blocchi["scale"] = (optS, pS, [a.scale_down_lr])
        _DD["ric"] = st.ric; _DD["dt"] = st.moduli[strati_sc[0]].mlp.esperti.scala("down").dtype
        log(f"scale down esperti: {len(pS)} tensori theta {tuple(pS[0].shape)}, lr {a.scale_down_lr:g}")
    pSgu, optSgu = [], None
    if a.scale_gateup_lr > 0:                             # 25/10 L3: theta of the gate/up scales, layer-major (L, gate/up)
        pSgu = [st.moduli[L].mlp.esperti.theta[f] for L in strati_sc for f in FAM_GU]
        optSgu = torch.optim.Adam([{"params": pSgu, "lr": a.scale_gateup_lr}], betas=(0.9, 0.999), weight_decay=0.0)
        blocchi["scale_gu"] = (optSgu, pSgu, [a.scale_gateup_lr])
        f0_ = st.moduli[strati_sc[0]].mlp.esperti.fam["gate"]
        log(f"scale gate/up esperti: {len(pSgu)} tensori theta {tuple(pSgu[0].shape)} ({f0_.tipo}, gruppo {f0_.gs}), "
            f"{sum(p.numel() for p in pSgu) / 1e6:.1f} M, lr {a.scale_gateup_lr:g}; scala = fp16(d0 exp(theta)), theta fp32")
    pDn, optDn = [], None
    if resDn:                                             # 25/10: own Adam block, clipped on its own norm
        pDn = CK.parametri_down(resDn, "C") + CK.parametri_down(resDn, "D")
        for p in pDn:
            p.requires_grad_(True)
        lr_cdn = a.down_lowrank_lr if a.down_lowrank_lr is not None else lr_c
        rms_ddn = CK.statistiche_down(resDn)["rms_D_down"]
        lr_ddn = a.down_lowrank_lr_d if a.down_lowrank_lr_d is not None else a.frazione_d * rms_ddn
        optDn = torch.optim.Adam([{"params": CK.parametri_down(resDn, "C"), "lr": lr_cdn},
                                  {"params": CK.parametri_down(resDn, "D"), "lr": lr_ddn}], betas=(0.9, 0.999),
                                 weight_decay=0.0)
        blocchi["down"] = (optDn, pDn, [lr_cdn, lr_ddn])
        log(f"down low-rank: lr C {lr_cdn:.3g} D {lr_ddn:.3g} (rms D_down {rms_ddn:.3g})")
    if F32 is not None:                                   # 26/10 F32-E2E: own Adam blocks, each clipped on its own norm
        blocchi.update(F32.blocchi())
        if a.solo_f32 and sorted(blocchi) != sorted(F32.nomi()):
            log(f"ERRORE: --solo-f32 ma i blocchi allenati sono {sorted(blocchi)}"); return 5
    if not blocchi and not a.forza_scelte_maestro:        # 26/10 BPP: prova 0 measures, it needs no trainable block
        log("ERRORE: nessun blocco allenabile"); return 5

    def salva_scale(p_):
        if pS:
            torch.save({L: t.detach().cpu() for L, t in zip(strati_sc, pS)}, str(p_) + ".tmp"); os.replace(str(p_) + ".tmp", p_)

    def pacco_scale_gu() -> tuple:
        """25/10 L3: (pack, stats) of the gate/up scales. Pack = theta fp32 (CPU copies: --riprendi and the exporter) + the
        sha256 of the fp16 scales the student uses NOW (es.scala on the training device = the bytes that deploy)."""
        th, sha, n_c, n_t, s2, mx = {}, {}, 0, 0, 0.0, 0.0
        with torch.no_grad():
            for L in strati_sc:
                es_ = st.moduli[L].mlp.esperti
                th[L], sha[L] = {}, {}
                for f in FAM_GU:
                    t = es_.theta[f].detach()
                    th[L][f] = t.to("cpu", copy=True)
                    s16 = es_.scala(f).detach().to(torch.float16)
                    sha[L][f] = hashlib.sha256(s16.cpu().contiguous().numpy().tobytes()).hexdigest()
                    n_c += int((s16.view(torch.int16) != es_.fam[f].d0.to(torch.float16).view(torch.int16)).sum())
                    n_t += s16.numel(); s2 += float(t.double().pow(2).sum()); mx = max(mx, float(t.abs().max()))
        pk = {"theta": th, "sha256_fp16": sha, "dispositivo": str(dev), "gguf": str(a.gguf), "famiglie": list(FAM_GU),
              "formula": "scale = fp16(d0 * exp(theta)), d0 = fp16 scales of the host GGUF, fp32 (StratoEsperti._scala)"}
        return pk, {"theta_gu_rms": math.sqrt(s2 / max(1, n_t)), "theta_gu_max": mx, "scale_gu_cambiate": n_c / max(1, n_t)}
    log(f"lr esperti C {lr_c:.3g} D {lr_d:.3g}; densi {a.lr_densi:g}/{a.lr_gamma:g}; modo {a.modo}; perdita paper "
        f"(tau {a.tau}, lambda {a.lambda_ce}, top-{a.topk}) + {a.kl4_peso} KL4 + {a.kl_peso} KL piena")

    def salva(nome_e, nome_d, meta, pk_gu=None):
        CK.salva(resE, out / f"{nome_e}_tmp.pt", meta); os.replace(out / f"{nome_e}_tmp.pt", out / f"{nome_e}.pt")
        if resD:
            RT.salva_latenti(resDT, out / f"{nome_d}_tmp.pt", meta)   # 26/10 BPP: resDT == resD when off
        else:                                           # 26/10 F32: --rango-densi 0, same file with no latents
            torch.save({"meta": meta, "cfg": None, "latenti": {}}, out / f"{nome_d}_tmp.pt")
        os.replace(out / f"{nome_d}_tmp.pt", out / f"{nome_d}.pt")
        if F32 is not None:                               # 26/10 F32-E2E: f32_{migliore,ultimo}.pt (the bytes to deploy)
            F32.salva(out / f"{nome_e.replace('cardan', 'f32', 1)}.pt", meta)
        if resDn:                                         # 25/10: cardan_down_{migliore,ultimo}.pt next to cardan_*.pt
            nome_dn = nome_e.replace("cardan", "cardan_down", 1)
            CK.salva(resDn, out / f"{nome_dn}_tmp.pt", meta); os.replace(out / f"{nome_dn}_tmp.pt", out / f"{nome_dn}.pt")
        if pSgu:                                          # 25/10 L3: scale_gateup_{migliore,ultimo}.pt
            nome_sg = nome_e.replace("cardan", "scale_gateup", 1)
            torch.save({**(pk_gu if pk_gu is not None else pacco_scale_gu()[0]), "meta": meta}, out / f"{nome_sg}_tmp.pt")
            os.replace(out / f"{nome_sg}_tmp.pt", out / f"{nome_sg}.pt")

    def carica_residui(p) -> list:
        """26/10 BPP: residui_*.pt -> dense + shexp latents; shexp residuals absent from the file (a checkpoint of a run
        without --residui-shexp) go back to their start (init_sh: B = 0, gamma = 1, seeded A). Returns their names.
        A latent in the file that this run does not have is an error (resuming a B++ checkpoint without the switch)."""
        d_ = torch.load(p, map_location="cpu")
        estranei = sorted(set(d_["latenti"]) - set(resDT))
        if estranei:
            raise ValueError(f"{Path(p).name}: residui {estranei[:3]}... non installati in questa corsa "
                             f"({len(estranei)}; manca --residui-shexp?)")
        for n, sd in d_["latenti"].items():
            resDT[n].load_state_dict(sd)
        mancano = [n for n in resSh if n not in d_["latenti"]]
        for n in mancano:
            resSh[n].load_state_dict(init_sh[n])
        return mancano

    if a.riprendi and (out / "ottimizzatori_ultimo.pt").exists():
        CK.carica(resE, out / "cardan_ultimo.pt")
        if resSh:                                         # 26/10 BPP (off: the call of always)
            try:
                mancano_sh = carica_residui(out / "residui_ultimo.pt")
            except ValueError as e:
                log(f"ERRORE: ripresa: {e}"); return 4
        else:
            try:
                RT.carica_latenti(resD, out / "residui_ultimo.pt")
            except KeyError as e:                         # 26/10 BPP: a B++ checkpoint resumed without --residui-shexp
                log(f"ERRORE: ripresa: residuo {e} nel checkpoint ma non in questa corsa (manca --residui-shexp?)"); return 4
        d = torch.load(out / "ottimizzatori_ultimo.pt", map_location=dev, weights_only=False)
        optE.load_state_dict(d["optE"]); optD.load_state_dict(d["optD"])
        if resSh:                                         # 26/10 BPP: latents + Adam of the shexp residuals
            if "optDsh" in d and mancano_sh:
                log(f"ERRORE: ripresa con optDsh ma senza {len(mancano_sh)} residui shexp in residui_ultimo.pt"); return 4
            if "optDsh" in d:
                optDsh.load_state_dict(d["optDsh"])
                log(f"ripresa: residui shexp ({len(resSh)}) e Adam densi_shexp dal checkpoint")
            elif mancano_sh:
                if len(mancano_sh) != len(resSh):
                    log(f"ERRORE: ripresa: {len(resSh) - len(mancano_sh)} residui shexp nel file ma {len(mancano_sh)} "
                        f"assenti e nessun optDsh"); return 4
                log(f"ripresa: la corsa ripresa non aveva i residui shexp: {len(resSh)} partono da B = 0 (init), Adam nuovo")
            else:
                log("ripresa: residui shexp caricati, stato Adam densi_shexp nuovo (assente nel checkpoint)")
        if pS and "optS" in d:
            optS.load_state_dict(d["optS"])
        if resDn:
            if not (out / "cardan_down_ultimo.pt").exists() or "optDn" not in d:
                log("ERRORE: ripresa con --down-lowrank-rango ma senza cardan_down_ultimo.pt / optDn"); return 4
            CK.carica(resDn, out / "cardan_down_ultimo.pt"); optDn.load_state_dict(d["optDn"])
        if pS and (out / "scale_ultimo.pt").exists():
            th = torch.load(out / "scale_ultimo.pt", map_location="cpu")
            with torch.no_grad():
                for L, t in zip(strati_sc, pS):
                    t.copy_(th[L].to(t.device))
        if pSgu:                                          # 25/10 L3: theta + Adam of the gate/up scales
            f_sg = out / "scale_gateup_ultimo.pt"
            if "optSgu" in d and not f_sg.exists():
                log("ERRORE: ripresa con optSgu ma senza scale_gateup_ultimo.pt"); return 4
            if f_sg.exists():
                pk = torch.load(f_sg, map_location="cpu", weights_only=False)
                with torch.no_grad():
                    for L in strati_sc:
                        for f in FAM_GU:
                            t = st.moduli[L].mlp.esperti.theta[f]
                            src_t = pk["theta"][L][f]
                            if tuple(src_t.shape) != tuple(t.shape):
                                log(f"ERRORE: ripresa: theta {f} L{L} {tuple(src_t.shape)} != {tuple(t.shape)}"); return 4
                            t.copy_(src_t.to(t.device))
                if pk.get("dispositivo") == str(dev):     # same device: the fp16 scales must come back at the bit
                    sha_ora = pacco_scale_gu()[0]["sha256_fp16"]
                    diversi = [(L, f) for L in strati_sc for f in FAM_GU if sha_ora[L][f] != pk["sha256_fp16"][L][f]]
                    if diversi:
                        log(f"ERRORE: ripresa: scale gate/up fp16 diverse da quelle salvate in {diversi[:4]}"); return 4
                if "optSgu" in d:
                    optSgu.load_state_dict(d["optSgu"])
                else:
                    log("ripresa: theta gate/up caricate, stato Adam gate/up nuovo (assente nel checkpoint)")
                log(f"ripresa: scale gate/up dal passo {pk.get('meta', {}).get('passo')} ({f_sg.name})")
            else:
                log("ripresa: la corsa ripresa non allenava le scale gate/up: partono da theta = 0, Adam nuovo")
        if F32 is not None:                               # 26/10 F32-E2E: masters, phi and Adam of the F32 blocks
            f_f32 = out / "f32_ultimo.pt"
            if "optF32" in d and not f_f32.exists():
                log("ERRORE: ripresa con optF32 ma senza f32_ultimo.pt"); return 4
            if f_f32.exists():
                try:
                    m_f32 = F32.carica(f_f32)
                    if "optF32" in d:
                        F32.carica_ottimizzatori(d["optF32"])
                except ValueError as e:
                    log(f"ERRORE: {e}"); return 4
                log(f"ripresa: F32 dal passo {m_f32.get('passo')} ({len(F32.voci)} tensori, Adam "
                    f"{F32.nomi() if 'optF32' in d else 'NUOVO (assente nel checkpoint)'})")
            else:                                         # 26/10 BPP: a run without --allena-f32 (B+) resumed with it
                log(f"ripresa: la corsa ripresa non allenava i F32: i {len(F32.voci)} tensori partono dai valori del "
                    f"GGUF ospite, Adam nuovo ({F32.nomi()})")
        if PR is not None and PR.beta > 0:                # 26/10 L6: running letter marginals of the resumed run
            if "prior_lettere" in d:
                PR.carica(d["prior_lettere"], dev)
                log(f"ripresa: prior di lettera EMA dopo {PR.n} passi (beta salvato {d['prior_lettere'].get('beta')})")
            else:
                log("ripresa: la corsa ripresa non aveva la stima EMA del prior: riparte dal primo lotto")
        storia, v0, migl, pegg, p_ini = d["storia"], d["v0"], tuple(d["migliore"]), d["pegg"], d["passo"] + 1
        log(f"RIPRESA dal passo {d['passo']}: migliore {migl}")
        if a.selezione == "kl24_paper":                 # 21/10: criterio cambiato in corsa -> migliore ricalcolato dalla storia
            cand = [(v0["kl24"] + v0["paper"], 0)] + [(r["held_kl24"] + r["held_paper"], r["passo"]) for r in storia
                                                       if "held_kl24" in r and r.get("guardia_kl24", True)]
            nuovo = min(cand)
            pegg = sum(1 for r in storia if "held_kl24" in r and r["passo"] > nuovo[1])
            if nuovo[1] != migl[1]:
                import shutil
                for n_, m_ in (("cardan", "cardan_migliore"), ("residui", "residui_migliore")) + (
                        (("cardan_down", "cardan_down_migliore"),) if resDn else ()) + (
                        (("scale_gateup", "scale_gateup_migliore"),) if pSgu else ()) + (
                        (("f32", "f32_migliore"),) if F32 is not None else ()):
                    src_ = out / "tappe" / f"{n_}_{nuovo[1]}.pt"
                    if not src_.exists():
                        log(f"ERRORE: manca {src_} per il migliore KL24+paper"); cicli("ERRORE", f"manca {src_}"); return 4
                    shutil.copy2(src_, out / f"{m_}.pt")
            migl = nuovo
            log(f"SELEZIONE KL24+paper: migliore ricalcolato {migl}, valutazioni dopo {pegg}")
    else:
        kl0 = kl24()
        if a.kl_partenza_motore is not None:            # = cancello di coerenza torch <-> motore
            sc = kl0 / a.kl_partenza_motore - 1
            log(f"coerenza: KL torch {kl0:.5f} contro motore {a.kl_partenza_motore:.5f} ({100 * sc:+.2f} %)")
            if not (math.isfinite(kl0) and abs(sc) <= a.soglia_coerenza):
                cicli("ERRORE", f"cancello di coerenza fallito ({100 * sc:+.2f} %)"); return 2
        if a.init_c == "a1" and a.base_a1 and resE:     # = cardan_kd_paper --init-c a1 (App. A.1 Eq. 6)
            with torch.no_grad():
                for L, m in resE.items():
                    z = np.load(Path(a.base_a1) / f"strato_{L:02d}.npz")
                    for f in ("gate", "up"):
                        k = [n for n in z.files if n.endswith(f"ffn_{f}_lrc.weight")][0]
                        C = torch.from_numpy(z[k].view(np.float16).astype(np.float32))[..., : a.rango_esperti]
                        getattr(m, f"C_{f}").copy_(C.to(getattr(m, f"C_{f}")))
            log("init C esperti: forma chiusa A.1 caricata")
        v0 = valuta(); v0["kl24_ospite"] = kl0
        log(f"passo 0: {json.dumps(v0)}")
        cicli("ATTIVO", f"passo 0 KL24 ospite {kl0:.5f}, partenza {json.dumps(v0)} -> {out}")
        storia, migl, pegg, p_ini = [{"passo": 0, **v0}], (v0["sel"], 0), 0, 1
        salva("cardan_migliore", "residui_migliore", {"passo": 0, **v0})

    rng = random.Random(a.seme)
    ord_w, ord_q = train_w[:], train_q[:]
    rng.shuffle(ord_w); rng.shuffle(ord_q)

    def attivi(passo: int) -> list:
        if a.modo == "congiunto":                      # 25/10 L3: "scale_gu" block; "esperti" absent with --rango-esperti 0
            return [b for b in ["esperti", "densi"] + (["scale"] if pS else []) + (["scale_gu"] if pSgu else [])
                    + (["down"] if pDn else []) + (["densi_shexp"] if pDsh else [])   # 26/10 BPP
                    + (F32.nomi() if F32 is not None else []) if b in blocchi]
        altro = "esperti" if a.primo == "densi" else "densi"
        b = a.primo if passo % 2 == 1 else altro
        return [b] + (["down"] if pDn and b == "esperti" else []) \
            + (["scale_gu"] if pSgu and b == "esperti" else []) \
            + (["densi_shexp"] if pDsh and b == "densi" else [])   # down / gate-up scales with the experts, shexp with densi

    def un_passo(passo: int, on: list) -> dict:
        for b, (opt, ps, _) in blocchi.items():
            for p in ps:
                p.requires_grad_(b in on)
            if a.adam_cpu_inattivo:
                sposta_stato(opt, dev if b in on else torch.device("cpu"))
        val = {}
        i_r = ord_w[(passo - 1) % len(ord_w)]
        lot = prendi(car_w, "train", i_r)                                    # 1) paper loss, one corpus row
        stato = st.avanti(lot.ids, inizi=lot.inizi)
        xL = stato["xL"].detach().requires_grad_(True)
        with torch.enable_grad():
            hS = st.esce(xL)
            kd, ce, T, n, g = perdita_paper(hS.detach(), lot, True, passo)
            hS.backward(g.to(hS.dtype))
        if RD_ is not None:                                                  # 26/10 BPP: router term in the recompute
            RD_.ultimo = None
        with router_su(lot, "train", i_r):
            st.indietro(stato, xL.grad)
        val["paper"] = kd / T + a.lambda_ce * ce / n
        router_in_val(val, False)                                            # 26/10 BPP
        del stato, xL, lot
        if PR is not None:                                                   # 2') 26/10 L6: KL4 (if w4 > 0) + prior
            val.update(passo_lettere(passo))
        elif a.kl4_peso > 0:                                                 # 2) KL4, one calibration pack
            p = pacco_del_passo(passo)
            ids, inizi, ult = pacco_tensori(p, dev)
            stato = st.avanti(ids, inizi=inizi)
            xL = stato["xL"].detach().requires_grad_(True)
            with torch.enable_grad():
                k4 = kl4_di(st.esce(xL).reshape(-1, xL.shape[-1]), p, ult, WS).mean()
                (a.kl4_peso * k4).backward()
            st.indietro(stato, xL.grad); val["kl4"] = float(k4.detach())
            del stato, xL
        if a.kl_peso > 0:                                                    # 3) optional full KL, one q8 row
            lot = prendi(car_q, "train", ord_q[(passo - 1) % len(ord_q)])
            stato = st.avanti(lot.ids, inizi=lot.inizi)
            xL = stato["xL"].detach().requires_grad_(True)
            with torch.enable_grad():
                kl = A.KLFusa.apply(st.esce(xL), WS, lot.hT, WT, a.pezzo_kl)
                (a.kl_peso * kl / lot.ids.numel()).backward()
            st.indietro(stato, xL.grad); val["kl"] = float(kl.detach()) / lot.ids.numel()
            del stato, xL, lot
        val["L"] = val["paper"] + a.kl4_peso * val.get("kl4", 0.0) + a.kl_peso * val.get("kl", 0.0)
        if PR is not None:                                                   # 26/10 L6
            val["L"] += PR.peso * val["prior_lettere"]
        if "router" in val:                                                  # 26/10 BPP
            val["L"] += a.router_cammino * val["router"]
        return val

    def passo_lettere(passo: int) -> dict:
        """26/10 L6, plain path: forwards of the step's N KL4 packs (recompute states kept), then ONE backward of
        w4 * KL4(first pack, = the original KL4 term) + W * KL(pbar_T || pbar_S)(all packs), then the N student backwards
        in pack order."""
        packs = PR.pacchi(passo, pacco_del_passo, len(pacchi4))
        stati, lS, pT, k4 = [], [], [], None
        for j, p in enumerate(packs):
            ids, inizi, ult = pacco_tensori(p, dev)
            stato = st.avanti(ids, inizi=inizi)
            xL = stato["xL"].detach().requires_grad_(True)
            with torch.enable_grad():
                hh = st.esce(xL).reshape(-1, xL.shape[-1])
                lS.append(logp_lettere(hh, p, ult, WS))
                if j == 0 and a.kl4_peso > 0:
                    k4 = kl4_di(hh, p, ult, WS).mean()
            pT.append(pT_di(p, dev)); stati.append((stato, xL))
            del hh
        with torch.enable_grad():
            kp, info = PR.termine(lS, pT)
            tot = PR.peso * kp if k4 is None else a.kl4_peso * k4 + PR.peso * kp
            tot.backward()
        for stato, xL in stati:
            st.indietro(stato, xL.grad)
        del stati, lS
        out_ = {"prior_lettere": float(kp.detach()), **info}
        if k4 is not None:
            out_["kl4"] = float(k4.detach())
        return out_

    def un_passo_fuso_prior(passo: int, on: list) -> dict:
        """26/10 L6, --fuso path: [paper row | N KL4 packs] in ONE forward/backward (segments), loss = paper +
        w4 * KL4(first pack) + W * KL(pbar_T || pbar_S)(all packs). With N = 1 the sequence is the one of un_passo_fuso."""
        for b, (opt, ps, _) in blocchi.items():
            for p_ in ps:
                p_.requires_grad_(b in on)
        val = {}
        lot = prendi(car_w, "train", ord_w[(passo - 1) % len(ord_w)])
        packs = PR.pacchi(passo, pacco_del_passo, len(pacchi4))
        Tp = lot.ids.shape[1]
        if lot.inizi is not None:
            iP = lot.inizi.to(torch.long).cpu()
        else:
            iP = torch.zeros((1, Tp), dtype=torch.long); iP[0, 0] = 1
        pezzi_ids, pezzi_in, ults, o = [lot.ids.to(dev)], [iP], [], Tp
        for p_ in packs:
            ids4, inizi4, ult = pacco_tensori(p_, dev)
            pezzi_ids.append(ids4); pezzi_in.append(inizi4); ults.append((o, ids4.shape[1], ult)); o += ids4.shape[1]
        ids = torch.cat(pezzi_ids, 1); inizi = torch.cat(pezzi_in, 1)
        stato = st.avanti(ids, inizi=inizi, grafo=True)   # E2E_SENZA_RICALCOLO=1: grafo tenuto (21/10)
        xL = stato["xL"].detach().requires_grad_(True)
        k4 = None
        with torch.enable_grad():
            h2 = st.esce(xL)
            hP = h2[:Tp]
            kd, ce, T, n, g = perdita_paper(hP.detach(), lot, True, passo)
            lS, pT = [], []
            for j, (p_, (o_, T_, ult)) in enumerate(zip(packs, ults)):
                h4 = h2[o_:o_ + T_]
                lS.append(logp_lettere(h4, p_, ult, WS)); pT.append(pT_di(p_, dev))
                if j == 0 and a.kl4_peso > 0:
                    k4 = kl4_di(h4, p_, ult, WS).mean()
            kp, info = PR.termine(lS, pT)
            tot = PR.peso * kp if k4 is None else a.kl4_peso * k4 + PR.peso * kp
            torch.autograd.backward([hP, tot], [g.to(hP.dtype), None])
        st.indietro(stato, xL.grad)
        val["paper"] = kd / T + a.lambda_ce * ce / n
        if k4 is not None:
            val["kl4"] = float(k4.detach())
        val["prior_lettere"] = float(kp.detach()); val.update(info)
        val["L"] = val["paper"] + a.kl4_peso * val.get("kl4", 0.0) + PR.peso * val["prior_lettere"]
        del stato, xL, lot
        return val

    def un_passo_fuso(passo: int, on: list) -> dict:
        """= un_passo with kl_peso 0, the two forwards/backwards concatenated as segments of one sequence."""
        for b, (opt, ps, _) in blocchi.items():
            for p in ps:
                p.requires_grad_(b in on)
        val = {}
        i_r = ord_w[(passo - 1) % len(ord_w)]
        lot = prendi(car_w, "train", i_r)
        p = pacco_del_passo(passo)
        ids4, inizi4, ult = pacco_tensori(p, dev)
        Tp = lot.ids.shape[1]
        if lot.inizi is not None:
            iP = lot.inizi.to(torch.long).cpu()
        else:
            iP = torch.zeros((1, Tp), dtype=torch.long); iP[0, 0] = 1
        ids = torch.cat([lot.ids.to(dev), ids4], 1); inizi = torch.cat([iP, inizi4], 1)
        if RD_ is not None:                               # 26/10 BPP: router term on the first n tokens (= the paper row)
            RD_.ultimo = None
        with router_su(lot, "train", i_r):
            stato = st.avanti(ids, inizi=inizi, grafo=True)   # E2E_SENZA_RICALCOLO=1: grafo tenuto (21/10)
            xL = stato["xL"].detach().requires_grad_(True)
            with torch.enable_grad():
                h2 = st.esce(xL)
                hP, h4 = h2[:Tp], h2[Tp:]
                kd, ce, T, n, g = perdita_paper(hP.detach(), lot, True, passo)
                k4 = kl4_di(h4, p, ult, WS).mean()
                torch.autograd.backward([hP, a.kl4_peso * k4], [g.to(hP.dtype), None])
            st.indietro(stato, xL.grad)
        val["paper"] = kd / T + a.lambda_ce * ce / n; val["kl4"] = float(k4.detach())
        val["L"] = val["paper"] + a.kl4_peso * val["kl4"]
        router_in_val(val, True)                          # 26/10 BPP
        del stato, xL, lot
        return val

    def termine_allinea(lot, i_riga: int, L_out: float) -> tuple:
        """25/10 Lever 2: ({layer: w * dL_al/dH_l [n, H]}, L_al, w) for the paper row just forwarded (outputs recorded by
        the Iniettore). w = --allinea-peso * sg(L_out / (L_al + eps)) (rapporto) or --allinea-peso (fisso)."""
        if not src_al.disponibile("train", i_riga):      # only with --allinea-parziale / dry run (checked at start)
            return {}, 0.0, 0.0
        Tm, n_ = stati_maestro_riga(lot, "train", i_riga)
        m_ = AS.maschera_inizi(lot.inizi, n_, dev) if a.allinea_salta_inizi else None
        L_al, gr, _ = AS.gradienti_allineamento({L: inj.uscite[L][:n_] for L in strati_al}, Tm, a.allinea_metrica, m_)
        w = AS.peso_effettivo(a.allinea_peso, a.allinea_modo, L_out, L_al, a.allinea_eps)
        return {L: w * g_ for L, g_ in gr.items()}, L_al, w

    def indietro_allineato(stato, gx, extra: dict) -> None:
        """st.indietro with the alignment terms added to the gradient of the aligned layer outputs (Iniettore hooks);
        every term must have been added exactly once."""
        inj.extra, inj.attivo, inj.colpi = extra, True, 0
        try:
            st.indietro(stato, gx)
        finally:
            inj.attivo, inj.extra = False, {}
        if inj.colpi != len(extra):
            raise RuntimeError(f"allinea: {inj.colpi} termini aggiunti nell'indietro, attesi {len(extra)}")

    def un_passo_allinea(passo: int, on: list) -> dict:
        """= un_passo + Lever 2. All forwards first (the ratio weight needs L_out of the whole step), then the backwards
        in the ORIGINAL order (paper row + alignment, KL4 pack, full KL): same gradient accumulation order."""
        for b, (opt, ps, _) in blocchi.items():
            for p_ in ps:
                p_.requires_grad_(b in on)
            if a.adam_cpu_inattivo:
                sposta_stato(opt, dev if b in on else torch.device("cpu"))
        val = {}
        i_riga = ord_w[(passo - 1) % len(ord_w)]
        lot = prendi(car_w, "train", i_riga)                                 # 1) paper row, outputs recorded
        inj.uscite, inj.registra = {}, True
        try:
            statoP = st.avanti(lot.ids, inizi=lot.inizi)
        finally:
            inj.registra = False
        xLP = statoP["xL"].detach().requires_grad_(True)
        with torch.enable_grad():
            hSP = st.esce(xLP)
            kd, ce, T, n, gP = perdita_paper(hSP.detach(), lot, True, passo)
        val["paper"] = kd / T + a.lambda_ce * ce / n
        if a.kl4_peso > 0:                                                   # 2) KL4 pack forward (graph kept)
            p = pacco_del_passo(passo)
            ids, inizi, ult = pacco_tensori(p, dev)
            statoK = st.avanti(ids, inizi=inizi)
            xLK = statoK["xL"].detach().requires_grad_(True)
            with torch.enable_grad():
                k4 = kl4_di(st.esce(xLK).reshape(-1, xLK.shape[-1]), p, ult, WS).mean()
            val["kl4"] = float(k4.detach())
        if a.kl_peso > 0:                                                    # 3) full KL forward (graph kept)
            lotQ = prendi(car_q, "train", ord_q[(passo - 1) % len(ord_q)])
            statoQ = st.avanti(lotQ.ids, inizi=lotQ.inizi)
            xLQ = statoQ["xL"].detach().requires_grad_(True)
            with torch.enable_grad():
                kl = A.KLFusa.apply(st.esce(xLQ), WS, lotQ.hT, WT, a.pezzo_kl)
            nQ = lotQ.ids.numel(); val["kl"] = float(kl.detach()) / nQ
        L_out = val["paper"] + a.kl4_peso * val.get("kl4", 0.0) + a.kl_peso * val.get("kl", 0.0)
        extra, L_al, w = termine_allinea(lot, i_riga, L_out)
        with torch.enable_grad():
            hSP.backward(gP.to(hSP.dtype))
        if RD_ is not None:                                                  # 26/10 BPP: router term in the recompute
            RD_.ultimo = None
        with router_su(lot, "train", i_riga):
            indietro_allineato(statoP, xLP.grad, extra)
        router_in_val(val, False)                                            # 26/10 BPP
        del statoP, xLP, hSP, lot, extra
        if a.kl4_peso > 0:
            with torch.enable_grad():
                (a.kl4_peso * k4).backward()
            st.indietro(statoK, xLK.grad)
            del statoK, xLK
        if a.kl_peso > 0:
            with torch.enable_grad():
                (a.kl_peso * kl / nQ).backward()
            st.indietro(statoQ, xLQ.grad)
            del statoQ, xLQ, lotQ
        val["allinea"], val["peso_allinea"] = L_al, w
        val["L"] = L_out + w * L_al
        if "router" in val:                                                  # 26/10 BPP
            val["L"] += a.router_cammino * val["router"]
        return val

    def un_passo_fuso_allinea(passo: int, on: list) -> dict:
        """= un_passo_fuso + Lever 2: one forward of [paper row | KL4 pack]; the alignment is on the first n tokens (paper
        row). Hooks active from the forward on (E2E_SENZA_RICALCOLO keeps the upper layers' graph from the forward)."""
        for b, (opt, ps, _) in blocchi.items():
            for p_ in ps:
                p_.requires_grad_(b in on)
        val = {}
        i_riga = ord_w[(passo - 1) % len(ord_w)]
        lot = prendi(car_w, "train", i_riga)
        p = pacco_del_passo(passo)
        ids4, inizi4, ult = pacco_tensori(p, dev)
        Tp = lot.ids.shape[1]
        if lot.inizi is not None:
            iP = lot.inizi.to(torch.long).cpu()
        else:
            iP = torch.zeros((1, Tp), dtype=torch.long); iP[0, 0] = 1
        ids = torch.cat([lot.ids.to(dev), ids4], 1); inizi = torch.cat([iP, inizi4], 1)
        if RD_ is not None:                               # 26/10 BPP: router term on the first n tokens (= the paper row)
            RD_.ultimo = None
        with router_su(lot, "train", i_riga):
            inj.uscite, inj.extra, inj.registra, inj.attivo, inj.colpi = {}, {}, True, True, 0
            try:
                stato = st.avanti(ids, inizi=inizi, grafo=True)
            except BaseException:
                inj.attivo = False
                raise
            finally:
                inj.registra = False
            xL = stato["xL"].detach().requires_grad_(True)
            with torch.enable_grad():
                h2 = st.esce(xL)
                hP, h4 = h2[:Tp], h2[Tp:]
                kd, ce, T, n, g = perdita_paper(hP.detach(), lot, True, passo)
                k4 = kl4_di(h4, p, ult, WS).mean()
            val["paper"] = kd / T + a.lambda_ce * ce / n; val["kl4"] = float(k4.detach())
            L_out = val["paper"] + a.kl4_peso * val["kl4"]
            lot_p = _ty.SimpleNamespace(ids=lot.ids[:, :Tp], inizi=iP)
            extra, L_al, w = termine_allinea(lot_p, i_riga, L_out)
            inj.extra = extra
            try:
                with torch.enable_grad():
                    torch.autograd.backward([hP, a.kl4_peso * k4], [g.to(hP.dtype), None])
                st.indietro(stato, xL.grad)
            finally:
                inj.attivo, inj.extra = False, {}
        if inj.colpi != len(extra):
            raise RuntimeError(f"allinea: {inj.colpi} termini aggiunti nell'indietro, attesi {len(extra)}")
        val["allinea"], val["peso_allinea"] = L_al, w
        val["L"] = L_out + w * L_al
        router_in_val(val, True)                          # 26/10 BPP
        del stato, xL, lot, extra
        return val

    if a.fuso or a.prova_fuso:                          # 26/10 L6: with the prior term on, KL4 may be 0
        assert a.kl_peso == 0 and (a.kl4_peso > 0 or PR is not None) and a.modo == "congiunto", \
            "--fuso: solo congiunto, KL4 > 0 (o prior di lettera), KL piena 0"
    if a.prova_fuso:                                    # same parameters, same rows: separate vs fused
        tutti = pE + pD + pDn + pSgu + pDsh                 # 26/10 BPP: pDsh empty when off
        for k in range(a.prova_fuso):
            pk = p_ini + k; tt, gg, vv = {}, {}, {}
            for nome, fn in (("separato", un_passo), ("fuso", un_passo_fuso if PR is None else un_passo_fuso_prior)):
                for opt, _, _ in blocchi.values():
                    opt.zero_grad(set_to_none=True)
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t1 = time.time()
                vv[nome] = fn(pk, ["esperti", "densi"] + (["down"] if pDn else []) + (["scale_gu"] if pSgu else [])
                              + (["densi_shexp"] if pDsh else []))
                torch.cuda.synchronize(); tt[nome] = time.time() - t1
                vv[nome]["picco_GiB"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
                gg[nome] = [x.grad.detach().float().clone() if x.grad is not None else torch.zeros_like(x, dtype=torch.float32)
                            for x in tutti]
            num = sum(float((x - y).pow(2).sum()) for x, y in zip(gg["separato"], gg["fuso"]))
            den = sum(float(x.pow(2).sum()) for x in gg["separato"])
            cos = sum(float((x * y).sum()) for x, y in zip(gg["separato"], gg["fuso"])) / math.sqrt(
                den * sum(float(y.pow(2).sum()) for y in gg["fuso"]))
            log(f"PROVA FUSO passo {pk}: separato {tt['separato']:.1f} s {json.dumps(vv['separato'])} | fuso "
                f"{tt['fuso']:.1f} s {json.dumps(vv['fuso'])} | grad: errore relativo {math.sqrt(num / den):.3e}, "
                f"coseno {cos:.8f} | fattore {tt['separato'] / tt['fuso']:.3f}")
            del gg
        return 0
    if a.fuso:
        un_passo = un_passo_fuso if PR is None else un_passo_fuso_prior   # noqa: F811  26/10 L6
        log("modo FUSO: paper + KL4 in un solo avanti/indietro (segmenti)" if PR is None else
            f"modo FUSO: paper + {PR.n_pacchi} pacchi KL4 (KL4 peso {a.kl4_peso:g}, prior di lettera) in un solo "
            f"avanti/indietro (segmenti)")
    if inj is not None:                                 # 25/10 Lever 2
        un_passo = un_passo_fuso_allinea if a.fuso else un_passo_allinea   # noqa: F811
        if ord_w != AS.righe_in_ordine(train_w, a.seme):
            log("ERRORE: ordine delle righe paper diverso da quello della cache"); return 5

    def prova_zero() -> int:
        """26/10 BPP --forza-scelte-maestro (R7 §5 "prova 0"): KL24 and paper held-out of the student as it is now
        (host, or the resumed tappa), free and with the teacher's top-8 imposed at every MoE layer; JSON + exit 0."""
        parti = (a.forza_scelte_maestro.split(",") + ["-"])[:2]
        sorg, rif = {}, {"paper": (a.teacher, held_w), "kl24": (a.teacher_kl24, held_q)}
        for chiave, c_ in zip(("paper", "kl24"), parti):
            if c_.strip() in ("", "-", "no"):
                continue
            try:
                s_ = RDm.SorgenteRouter(c_.strip())
            except (OSError, ValueError, KeyError, RuntimeError) as e:
                log(f"ERRORE: --forza-scelte-maestro {c_}: {e}"); return 5
            if Path(s_.piano["sorgente"]).resolve() != Path(rif[chiave][0]).resolve():
                log(f"ERRORE: forza: la cache {c_} e' di {s_.piano['sorgente']}, il {chiave} usa {rif[chiave][0]}"); return 5
            manca = [i for i in rif[chiave][1] if not s_.disponibile("heldout", i)]
            if manca or any(L not in s_.strati for L in st.strati):
                log(f"ERRORE: forza: cache {c_}: righe held-out mancanti {manca[:4]} o strati mancanti"); return 5
            sorg[chiave] = s_
        if not sorg:
            log("ERRORE: --forza-scelte-maestro senza cache"); return 5
        FZ = RDm.ForzaScelte(st, sorg, st.strati)
        libero = valuta()
        forz, coperti = {}, {}
        with torch.no_grad():
            if "kl24" in sorg:
                tot, n, cop = 0.0, 0, 0
                for i in held_q:
                    lot = prendi(car_q, "heldout", i)
                    with FZ.su("kl24", "heldout", i, lot.ids[0]):
                        hS = st.nascosto(lot.ids, inizi=lot.inizi); cop += FZ.n
                    for j in range(0, hS.shape[0], a.pezzo_kl):
                        r = A.kl_pezzo(hS[j:j + a.pezzo_kl], WS, lot.hT[j:j + a.pezzo_kl], WT, False)
                        tot += float(r["kl"].sum()); n += int(r["kl"].numel())
                forz["kl24"], coperti["kl24"] = tot / n, cop / n
            if "paper" in sorg:
                tk, tc, nt_, nv, cop, ntok = 0.0, 0.0, 0, 0, 0, 0
                for i in held_w:
                    lot = prendi(car_w, "heldout", i)
                    with FZ.su("paper", "heldout", i, lot.ids[0]):
                        hS = st.nascosto(lot.ids, inizi=lot.inizi); cop += FZ.n
                    ntok += lot.ids.numel()
                    kd, ce, T, n, _ = perdita_paper(hS, lot, False); tk += kd; tc += ce; nt_ += T; nv += n
                forz["paper"], coperti["paper"] = tk / nt_ + a.lambda_ce * tc / nv, cop / ntok
        FZ.rimuovi()
        ov = sum(FZ.ov) / max(1, len(FZ.ov))
        ris = {"libero": {k: libero[k] for k in ("kl24", "paper", "kl4") if k in libero}, "forzato": forz,
               "differenza": {k: forz[k] - libero[k] for k in forz}, "token_forzati_frazione": coperti,
               "ov8_studente_libero_sulle_scelte_maestro": ov, "strati": list(st.strati), "gguf": str(a.gguf),
               "ripresa": bool(a.riprendi), "cache": {k: str(s_.dir) for k, s_ in sorg.items()},
               "nota": "R7 §5 prova 0: tetto di ogni leva d'instradamento; porte = softmax dello studente rinormalizzato "
                       "sulle top-8 del maestro (MoEStudente.instr); KL4 non forzata (nessuna cache dei pacchi MMLU)"}
        json.dump(ris, open(out / "forza_scelte_maestro.json", "w"), indent=1)
        log(f"PROVA 0 (scelte del maestro imposte): {json.dumps(ris['libero'])} -> forzato {json.dumps(forz)} "
            f"(differenza {json.dumps(ris['differenza'])}; token forzati {json.dumps(coperti)}; ov8 libero {ov:.4f})")
        cicli("ATTIVO", f"prova 0 forza-scelte: {json.dumps(ris['differenza'])}")
        return 0

    if a.forza_scelte_maestro:                          # 26/10 BPP: prova 0 of R7 §5, then exit (no training)
        return prova_zero()
    t_inizio = time.time(); passo = p_ini - 1
    for passo in range(p_ini, a.passi + 1):
        t1 = time.time()
        if t1 - t_avvio > a.ore_max * 3600:
            log(f"tetto di {a.ore_max} h: mi fermo prima del passo {passo}"); passo -= 1; break
        on = attivi(passo)
        f = min(1.0, passo / max(1, a.riscaldamento))
        if a.wsd_frazione > 0:                            # WSD: picco costante, poi coseno fino a 0 sull'ultimo f*T
            n_cd = max(1, round(a.wsd_frazione * a.passi)); s_cd = a.passi - n_cd
            if passo > s_cd:
                f *= 0.5 * (1 + math.cos(math.pi * (passo - s_cd) / n_cd))
        for b in on:
            opt, _, base = blocchi[b]
            for gr, lr in zip(opt.param_groups, base):
                gr["lr"] = lr * f
        for opt, _, _ in blocchi.values():                # also the frozen block: no stale grads kept in memory
            opt.zero_grad(set_to_none=True)
        if F32 is not None:                               # 26/10 F32: masters of log classes are in no optimiser
            F32.azzera_grad_master()
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        val = un_passo(passo, on)
        if F32 is not None:                               # 26/10 F32: dL/dphi = dL/dw * w, coverage per family
            cop = F32.gradienti_log()
            if passo == p_ini:
                log(f"F32 gradienti non nulli al primo passo (tensori per famiglia): "
                    f"{json.dumps({f: f'{c[0]}/{c[1]}' for f, c in cop.items()})}")
                vuote = [f for f, c in cop.items() if c[0] == 0]
                if vuote:
                    log(f"ERRORE: famiglie F32 senza gradiente {vuote}: il percorso veloce non le vede"); return 6
        norme = {b: float(torch.nn.utils.clip_grad_norm_(blocchi[b][1], a.clip)) for b in on}
        # 26/10 (coordinator fix, same in every trainer copy): max() hides a NaN that is not the first block norm
        # (live KLPPRIOR step 166: "norma_scale": NaN, step taken, theta NaN for good) -> any non-finite norm skips
        n_max = max(norme.values()) if all(math.isfinite(v) for v in norme.values()) else float("nan")
        motivo = KDP.motivo_salto(val["L"], n_max)
        riga = {"passo": passo, "blocchi": on, **{k: round(v, 6) for k, v in val.items()},
                **{f"norma_{b}": round(v, 5) for b, v in norme.items()}}
        if motivo is not None:
            if PR is not None:
                PR.scarta()
            log(f"PASSO SALTATO ({motivo}): {json.dumps(riga)}"); storia.append({**riga, "saltato": motivo}); continue
        for b in on:
            blocchi[b][0].step()
        if F32 is not None:                               # 26/10 F32: w = sign(w0) exp(phi) where phi moved
            F32.dopo_passo()
        if PR is not None:                                # 26/10 L6: the running estimate moves only with a taken step
            PR.conferma()
        riga["s"] = round(time.time() - t1, 1)
        if dev.type == "cuda":
            riga["vram_picco_GiB"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
            if passo == p_ini and riga["vram_picco_GiB"] > a.vram_tetto:
                log(f"ERRORE: picco {riga['vram_picco_GiB']} GiB > tetto {a.vram_tetto}"); cicli("ERRORE", "VRAM oltre il tetto")
                return 3
        if secco:
            riga["grad_nonnulli"] = {
                "esperti": sum(int(p.grad is not None and bool(p.grad.abs().sum() > 0)) for p in pE),
                "densi": sum(int(p.grad is not None and bool(p.grad.abs().sum() > 0)) for p in pD),
                "di": {"esperti": len(pE), "densi": len(pD)}}
            if pDn:
                riga["grad_nonnulli"]["down"] = sum(int(p.grad is not None and bool(p.grad.abs().sum() > 0)) for p in pDn)
                riga["grad_nonnulli"]["di"]["down"] = len(pDn)
            if pSgu:                                      # 25/10 L3
                riga["grad_nonnulli"]["scale_gu"] = sum(int(p.grad is not None and bool(p.grad.abs().sum() > 0)) for p in pSgu)
                riga["grad_nonnulli"]["di"]["scale_gu"] = len(pSgu)
            if pDsh:                                      # 26/10 BPP
                riga["grad_nonnulli"]["densi_shexp"] = sum(int(p.grad is not None and bool(p.grad.abs().sum() > 0))
                                                           for p in pDsh)
                riga["grad_nonnulli"]["di"]["densi_shexp"] = len(pDsh)
            if F32 is not None:                           # 26/10 F32: per family, from the step's coverage
                riga["grad_nonnulli"]["f32"] = {f: c[0] for f, c in F32.copertura.items()}
                riga["grad_nonnulli"]["di"]["f32"] = {f: c[1] for f, c in F32.copertura.items()}
        if passo % a.heldout_ogni == 0 or secco:
            v = valuta(); riga.update({f"held_{k}": v[k] for k in ("kl24", "paper", "kl4", "sel")})
            if "allinea" in v:                            # 25/10 Lever 2 (logged, not selected on)
                riga["held_allinea"] = v["allinea"]
            if "router" in v:                             # 26/10 BPP (logged, not selected on)
                riga.update({f"held_{k_}": v[k_] for k_ in v if k_.startswith("router")})
            if "prior" in v:                              # 26/10 L6 (logged, not selected on) + R6 §7.3 guard
                riga.update({"held_prior": v["prior"], "held_pS": v["pS"], "held_pT": v["pT"]})
                rif_pr = next((r_["prior"] if "prior" in r_ else r_["held_prior"] for r_ in storia
                               if "prior" in r_ or "held_prior" in r_), None)
                if rif_pr is not None and v["prior"] > rif_pr + 0.003:
                    log(f"ALLARME prior di lettera: held {v['prior']:.5f} > partenza {rif_pr:.5f} + 0.003 (pS {v['pS']})")
            if resDn:
                riga.update({k: round(x, 8) for k, x in CK.statistiche_down(resDn).items()})
            pk_gu = None                                  # 25/10 L3: one pass over the gate/up scales per evaluation
            if pSgu:
                pk_gu, st_gu = pacco_scale_gu()
                riga.update({k: round(x, 8) for k, x in st_gu.items()})
            if F32 is not None:                           # 26/10 F32: how far the F32 moved (per class)
                riga.update({k: round(x, 8) for k, x in F32.statistiche().items()})
            if resSh:                                     # 26/10 BPP: size of the shexp correction (rms of gamma B, A)
                with torch.no_grad():
                    gb = [(resSh[n].B * resSh[n].gamma.view(-1, 1)).detach().double().pow(2) for n in sorted(resSh)]
                    riga["shexp_gB_rms"] = round(math.sqrt(sum(float(x.sum()) for x in gb)
                                                           / max(1, sum(x.numel() for x in gb))), 10)
                    riga["shexp_gB_max"] = round(max(float(x.max()) for x in gb) ** 0.5, 10)
            if a.solo_f32:                                # 26/10 F32: every theta exactly 0 and without gradient
                cg = FB.controllo_congelati(st)
                riga["congelati"] = cg["ok"]
                if not cg["ok"]:
                    log(f"ERRORE: --solo-f32 ma theta mosse o con gradiente: {json.dumps(cg)}")
                    cicli("ERRORE", "solo-f32: theta non congelate"); return 7
            guardia = v["kl24"] <= storia[0]["kl24"] * (1 + a.guardia_kl24)
            if a.selezione == "mmlu" or (v["sel"] < migl[0] - 1e-5 and guardia):   # mmlu: «migliore» = ultimo passo
                migl, pegg = (v["sel"], passo), 0
                salva("cardan_migliore", "residui_migliore", {"passo": passo, **v}, pk_gu)
            else:
                pegg += 1
            riga["guardia_kl24"] = guardia
            salva("cardan_ultimo", "residui_ultimo", {"passo": passo}, pk_gu); salva_scale(out / "scale_ultimo.pt")
            if a.tappe_dir:
                Path(a.tappe_dir).mkdir(parents=True, exist_ok=True)   # 25/10: la copia delle scale veniva prima della mkdir
            if a.tappe_dir and pS:
                import shutil
                shutil.copyfile(out / "scale_ultimo.pt", Path(a.tappe_dir) / f"scale_{passo}.pt")
            if a.tappe_dir:
                import shutil
                for n_ in ("cardan", "residui") + (("cardan_down",) if resDn else ()) + (("scale_gateup",) if pSgu else ()) \
                        + (("f32",) if F32 is not None else ()):
                    shutil.copyfile(out / f"{n_}_ultimo.pt", Path(a.tappe_dir) / f"{n_}_{passo}.pt.tmp")
                    os.replace(Path(a.tappe_dir) / f"{n_}_{passo}.pt.tmp", Path(a.tappe_dir) / f"{n_}_{passo}.pt")
            torch.save({"optE": optE.state_dict(), "optD": optD.state_dict(), **({"optS": optS.state_dict()} if optS else {}),
                        **({"optDn": optDn.state_dict()} if optDn else {}),
                        **({"optSgu": optSgu.state_dict()} if optSgu else {}),
                        **({"prior_lettere": PR.stato()} if PR is not None and PR.beta > 0 else {}),   # 26/10 L6
                        **({"optF32": F32.stato_ottimizzatori()} if F32 is not None else {}),         # 26/10 F32
                        **({"optDsh": optDsh.state_dict()} if optDsh is not None else {}),           # 26/10 BPP
                        "storia": storia + [riga], "v0": v0,
                        "migliore": list(migl), "pegg": pegg, "passo": passo}, out / "ottimizzatori_ultimo.pt")
            if a.tappe_dir:                                # 24/10 (Angelo): anche lo stato Adam di OGNI tappa, mai sovrascritto
                import shutil
                shutil.copyfile(out / "ottimizzatori_ultimo.pt", Path(a.tappe_dir) / f"ottimizzatori_{passo}.pt.tmp")
                os.replace(Path(a.tappe_dir) / f"ottimizzatori_{passo}.pt.tmp", Path(a.tappe_dir) / f"ottimizzatori_{passo}.pt")
            cicli("ATTIVO", f"passo {passo} sel {v['sel']:.5f} KL24 {v['kl24']:.5f} KL4 {v['kl4']:.5f}, migliore {migl}")
        storia.append(riga); log(json.dumps(riga))
        if pegg >= a.pazienza:
            log(f"arresto: {pegg} valutazioni senza miglioramento"); break
    json.dump({"storia": storia, "migliore": list(migl), "secondi": time.time() - t_avvio,
               "s_passo_medio": (time.time() - t_inizio) / max(1, passo - p_ini + 1)},
              open(out / "storia.json", "w"), indent=1)
    if secco and migl[1] == 0:                          # dry run: export the trained step anyway (fold test)
        salva("cardan_migliore", "residui_migliore", {"passo": passo, "secco": True}); migl = (migl[0], passo)
    if migl[1] > 0 or a.init_c == "a1":                 # export in the formats the two existing folds read
        CK.carica(resE, out / "cardan_migliore.pt")
        if resSh:                                         # 26/10 BPP: shexp absent from the best file -> their start (B = 0)
            mancano_sh = carica_residui(out / "residui_migliore.pt")
            if mancano_sh:
                log(f"esportazione: il migliore (passo {migl[1]}) non ha i residui shexp: {len(mancano_sh)} a B = 0")
        else:
            RT.carica_latenti(resD, out / "residui_migliore.pt")
        CK.scrivi_json(resE, out / "cardan.json", {"passo": migl[1], "uso": "cardan_piega.py --pt cardan_migliore.pt"})
        if resDn:
            CK.carica(resDn, out / "cardan_down_migliore.pt")
            CK.scrivi_json_down(resDn, out / "cardan_down.json",
                                {"passo": migl[1], "uso": "cardan_piega_down.py --pt cardan_down_migliore.pt"})
        if pSgu:                                          # 25/10 L3: nothing to fold, the scales ARE the deployed bytes
            json.dump({"passo": migl[1], "strati": strati_sc, "famiglie": list(FAM_GU),
                       "uso": "scale_gateup_esporta.py OSPITE.gguf scale_gateup_migliore.pt DIR_TENSORI (trit invariati), "
                              "poi raffina/patch_gguf.py --dir DIR_TENSORI"}, open(out / "scale_gateup.json", "w"), indent=1)
        if F32 is not None:                               # 26/10 F32: nothing to fold, the F32 values ARE the deployed bytes
            json.dump({"passo": migl[1], "tensori": len(F32.voci), "famiglie": list(f32_fam),
                       "uso": "f32_esporta.py --base OSPITE.gguf --pacco f32_migliore.pt --uscita DIR, poi "
                              "raffina/patch_gguf.py --dir DIR/tensori --base OSPITE.gguf --out NUOVO.gguf"},
                      open(out / "f32.json", "w"), indent=1)
        dW = {n: RT.delta_file(m).float() for n, m in resDT.items()}   # 26/10 BPP: + shexp (resDT == resD when off)
        torch.save({"delta_file": dW, "rango": a.rango_densi,
                    "famiglie": (list(cfgD.famiglie) if cfgD else []) + (list(cfgSh.famiglie) if cfgSh else []),
                    "strati": RT.strati_di(resDT) if resDT else [], "passo": migl[1], "nome": a.nome,
                    "nota": "dW = diag(gamma) B A, ordine del FILE, base NON ruotata (cardan_piega_densi.py)"},
                   out / "residui_delta.pt")
        log(f"esportati cardan_migliore.pt (esperti) + residui_delta.pt ({len(dW)} densi), passo {migl[1]}")
    json.dump({"passo": migl[1], "sel": migl[0], "selezione": a.selezione, "modo": a.modo, "nome": a.nome},
              open(out / "esportazione.json", "w"), indent=1)
    log(f"FINE: migliore passo {migl[1]} sel {migl[0]:.5f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
