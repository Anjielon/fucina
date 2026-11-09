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
"""
from __future__ import annotations

import argparse
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
import cardan_kd_paper as KDP            # noqa: E402  path setup (QUI, REC, E2E_KERNEL_NUOVI) + helpers, unchanged
QUI, REC = KDP.QUI, KDP.REC
import residui_ternari as RT             # noqa: E402
import cardan_kd_moduli as CK            # noqa: E402
import cardan_paper_perdita as PP        # noqa: E402

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
    a_("--rango-esperti", type=int, default=16); a_("--base-a1", default=None)
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
    a_("--seme", type=int, default=20261007); a_("--riprendi", action="store_true")
    a_("--device", default="cuda"); a_("--ricetta", default="B3_ind2")
    a_("--prova-secco", type=int, default=0, help="CPU dry run: N tokens per row/pack, one step")
    a_("--fuso", action="store_true", help="18/10: paper row + KL4 pack in ONE forward/backward (segment mode, the "
       "token mixer runs per segment, MoE/norms on all tokens together): same loss, same gradients up to fp order")
    a_("--prova-fuso", type=int, default=0, help="N steps: gradients and time, separate vs fused, then exit (no Adam)")
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


def kl4_di(hh, p, ult, WS):
    """= rilq_thor_kl4: mean over the pack of KL(pT4 || softmax(letter logits of the student))."""
    let = torch.tensor(p[0]["lettere"], device=hh.device)
    Wl = (WS.hi[let].float() + WS.lo[let].float()) if hasattr(WS, "hi") else WS[let].to(torch.float32)
    lS = torch.log_softmax(hh[torch.tensor(ult, device=hh.device)].float() @ Wl.T, -1)
    pT = torch.stack([q["pT"] for q in p]).to(hh.device).float()
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
        st = S.Studente(gguf=Path(a.gguf), strati=KDP.intervallo(a.strati_studente), ricetta=a.ricetta,
                        motore_nome=os.environ.get("E2E_MOTORE", "torch" if dev.type == "cpu" else "triton"),
                        device=a.device, log=log, densi_ternari="tutte")
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

    def prendi(car, lato, i):
        return taglia(car.prendi(lato, [i]), a.prova_secco)

    def perdita_paper(hd, lot, con_grad):
        tgt, ok = bersagli(lot)
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

    def paper_held() -> float:
        tk, tc, nt_, nv = 0.0, 0.0, 0, 0
        with torch.no_grad():
            for i in held_w:
                lot = prendi(car_w, "heldout", i); hS = st.nascosto(lot.ids, inizi=lot.inizi)
                kd, ce, T, n, _ = perdita_paper(hS, lot, False); tk += kd; tc += ce; nt_ += T; nv += n
        return tk / nt_ + a.lambda_ce * tc / nv

    def kl4_held() -> float:
        tot, n = 0.0, 0
        with torch.no_grad():
            for p in held4:
                ids, inizi, ult = pacco_tensori(p, dev)
                hS = st.nascosto(ids, inizi=inizi)
                hS = hS.reshape(-1, hS.shape[-1])
                tot += float(kl4_di(hS, p, ult, WS).sum()); n += len(p)
        return tot / n

    def valuta() -> dict:
        v = {"kl24": kl24(), "paper": paper_held(), "kl4": kl4_held()}
        v["sel"] = (v["kl4"] if a.selezione == "kl4" else v["kl24"] + v["paper"] if a.selezione == "kl24_paper"
                    else v["paper"])
        return v

    # ── cancello d'identita' con ENTRAMBI i rami installati (C = 0, B = 0): h al bit
    lot = prendi(car_q, "heldout", held_q[0])
    with torch.no_grad():
        h0 = st.nascosto(lot.ids, inizi=lot.inizi).clone()
    strati = KDP.intervallo(a.strati)
    resE = CK.installa(st, strati, a.rango_esperti, Path(a.base_a1) if a.base_a1 else None, dev)
    cfgD = RT.ConfigResidui(rango=a.rango_densi, piani=1, strati=strati, famiglie=RT.famiglie_da(a.famiglie),
                            seme=a.seme, continuo=True)
    resD = RT.installa(st, cfgD, device=dev)
    with torch.no_grad():
        h1 = st.nascosto(lot.ids, inizi=lot.inizi)
    ident = bool(torch.equal(h0, h1))
    nE = sum(p.numel() for p in CK.parametri(resE, "C") + CK.parametri(resE, "D"))
    nD = sum(p.numel() for p in RT.parametri(resD))
    log(f"cancello identita' (esperti C = 0 su {len(resE)} strati, {nE / 1e6:.2f} M; densi B = 0 su {len(resD)} "
        f"proiezioni, {nD / 1e6:.2f} M): {'AL BIT' if ident else 'DIVERSO'}")
    if not ident:
        log("ERRORE: la partenza non e' l'ospite al bit"); return 2
    for nm in {sorted(resD)[0], sorted(resD)[-1]}:      # = rilq_thor_kl4 cancello di presenza (modo continuo)
        with torch.no_grad():
            resD[nm].B.normal_(0.0, 1e-3); h2 = st.nascosto(lot.ids, inizi=lot.inizi)
            resD[nm].B.zero_(); h3 = st.nascosto(lot.ids, inizi=lot.inizi)
        if torch.equal(h2, h1) or not torch.equal(h3, h1):
            log(f"ERRORE: ramo denso {nm} fuori dal percorso dello studente"); return 2
    log("cancello di presenza densi: ok")
    del h0, h1, lot
    st.strati_allenati = sorted(set(strati) | set(RT.strati_di(resD)))
    pE = CK.parametri(resE, "C") + CK.parametri(resE, "D")
    pD = RT.parametri(resD)
    for p in pE + pD:
        p.requires_grad_(True)
    rms_d = float(torch.cat([p.detach().reshape(-1) for p in CK.parametri(resE, "D")]).float().pow(2).mean().sqrt())
    rms_c = None
    if a.base_a1:
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
    log(f"lr esperti C {lr_c:.3g} D {lr_d:.3g}; densi {a.lr_densi:g}/{a.lr_gamma:g}; modo {a.modo}; perdita paper "
        f"(tau {a.tau}, lambda {a.lambda_ce}, top-{a.topk}) + {a.kl4_peso} KL4 + {a.kl_peso} KL piena")

    def salva(nome_e, nome_d, meta):
        CK.salva(resE, out / f"{nome_e}_tmp.pt", meta); os.replace(out / f"{nome_e}_tmp.pt", out / f"{nome_e}.pt")
        RT.salva_latenti(resD, out / f"{nome_d}_tmp.pt", meta); os.replace(out / f"{nome_d}_tmp.pt", out / f"{nome_d}.pt")

    if a.riprendi and (out / "ottimizzatori_ultimo.pt").exists():
        CK.carica(resE, out / "cardan_ultimo.pt"); RT.carica_latenti(resD, out / "residui_ultimo.pt")
        d = torch.load(out / "ottimizzatori_ultimo.pt", map_location=dev, weights_only=False)
        optE.load_state_dict(d["optE"]); optD.load_state_dict(d["optD"])
        storia, v0, migl, pegg, p_ini = d["storia"], d["v0"], tuple(d["migliore"]), d["pegg"], d["passo"] + 1
        log(f"RIPRESA dal passo {d['passo']}: migliore {migl}")
        if a.selezione == "kl24_paper":                 # 21/10: criterio cambiato in corsa -> migliore ricalcolato dalla storia
            cand = [(v0["kl24"] + v0["paper"], 0)] + [(r["held_kl24"] + r["held_paper"], r["passo"]) for r in storia
                                                       if "held_kl24" in r and r.get("guardia_kl24", True)]
            nuovo = min(cand)
            pegg = sum(1 for r in storia if "held_kl24" in r and r["passo"] > nuovo[1])
            if nuovo[1] != migl[1]:
                import shutil
                for n_, m_ in (("cardan", "cardan_migliore"), ("residui", "residui_migliore")):
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
        if a.init_c == "a1" and a.base_a1:              # = cardan_kd_paper --init-c a1 (App. A.1 Eq. 6)
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
        if a.modo == "congiunto":
            return ["esperti", "densi"]
        altro = "esperti" if a.primo == "densi" else "densi"
        return [a.primo if passo % 2 == 1 else altro]

    def un_passo(passo: int, on: list) -> dict:
        for b, (opt, ps, _) in blocchi.items():
            for p in ps:
                p.requires_grad_(b in on)
            if a.adam_cpu_inattivo:
                sposta_stato(opt, dev if b in on else torch.device("cpu"))
        val = {}
        lot = prendi(car_w, "train", ord_w[(passo - 1) % len(ord_w)])        # 1) paper loss, one corpus row
        stato = st.avanti(lot.ids, inizi=lot.inizi)
        xL = stato["xL"].detach().requires_grad_(True)
        with torch.enable_grad():
            hS = st.esce(xL)
            kd, ce, T, n, g = perdita_paper(hS.detach(), lot, True)
            hS.backward(g.to(hS.dtype))
        st.indietro(stato, xL.grad); val["paper"] = kd / T + a.lambda_ce * ce / n
        del stato, xL, lot
        if a.kl4_peso > 0:                                                   # 2) KL4, one calibration pack
            p = pacchi4[(passo - 1) % len(pacchi4)]
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
        return val

    def un_passo_fuso(passo: int, on: list) -> dict:
        """= un_passo with kl_peso 0, the two forwards/backwards concatenated as segments of one sequence."""
        for b, (opt, ps, _) in blocchi.items():
            for p in ps:
                p.requires_grad_(b in on)
        val = {}
        lot = prendi(car_w, "train", ord_w[(passo - 1) % len(ord_w)])
        p = pacchi4[(passo - 1) % len(pacchi4)]
        ids4, inizi4, ult = pacco_tensori(p, dev)
        Tp = lot.ids.shape[1]
        if lot.inizi is not None:
            iP = lot.inizi.to(torch.long).cpu()
        else:
            iP = torch.zeros((1, Tp), dtype=torch.long); iP[0, 0] = 1
        ids = torch.cat([lot.ids.to(dev), ids4], 1); inizi = torch.cat([iP, inizi4], 1)
        stato = st.avanti(ids, inizi=inizi, grafo=True)   # E2E_SENZA_RICALCOLO=1: grafo tenuto (21/10)
        xL = stato["xL"].detach().requires_grad_(True)
        with torch.enable_grad():
            h2 = st.esce(xL)
            hP, h4 = h2[:Tp], h2[Tp:]
            kd, ce, T, n, g = perdita_paper(hP.detach(), lot, True)
            k4 = kl4_di(h4, p, ult, WS).mean()
            torch.autograd.backward([hP, a.kl4_peso * k4], [g.to(hP.dtype), None])
        st.indietro(stato, xL.grad)
        val["paper"] = kd / T + a.lambda_ce * ce / n; val["kl4"] = float(k4.detach())
        val["L"] = val["paper"] + a.kl4_peso * val["kl4"]
        del stato, xL, lot
        return val

    if a.fuso or a.prova_fuso:
        assert a.kl_peso == 0 and a.kl4_peso > 0 and a.modo == "congiunto", "--fuso: solo congiunto, KL4 > 0, KL piena 0"
    if a.prova_fuso:                                    # same parameters, same rows: separate vs fused
        tutti = pE + pD
        for k in range(a.prova_fuso):
            pk = p_ini + k; tt, gg, vv = {}, {}, {}
            for nome, fn in (("separato", un_passo), ("fuso", un_passo_fuso)):
                for opt, _, _ in blocchi.values():
                    opt.zero_grad(set_to_none=True)
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t1 = time.time()
                vv[nome] = fn(pk, ["esperti", "densi"]); torch.cuda.synchronize(); tt[nome] = time.time() - t1
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
        un_passo = un_passo_fuso                        # noqa: F811
        log("modo FUSO: paper + KL4 in un solo avanti/indietro (segmenti)")

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
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        val = un_passo(passo, on)
        norme = {b: float(torch.nn.utils.clip_grad_norm_(blocchi[b][1], a.clip)) for b in on}
        motivo = KDP.motivo_salto(val["L"], max(norme.values()))
        riga = {"passo": passo, "blocchi": on, **{k: round(v, 6) for k, v in val.items()},
                **{f"norma_{b}": round(v, 5) for b, v in norme.items()}}
        if motivo is not None:
            log(f"PASSO SALTATO ({motivo}): {json.dumps(riga)}"); storia.append({**riga, "saltato": motivo}); continue
        for b in on:
            blocchi[b][0].step()
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
        if passo % a.heldout_ogni == 0 or secco:
            v = valuta(); riga.update({f"held_{k}": v[k] for k in ("kl24", "paper", "kl4", "sel")})
            guardia = v["kl24"] <= storia[0]["kl24"] * (1 + a.guardia_kl24)
            if a.selezione == "mmlu" or (v["sel"] < migl[0] - 1e-5 and guardia):   # mmlu: «migliore» = ultimo passo
                migl, pegg = (v["sel"], passo), 0
                salva("cardan_migliore", "residui_migliore", {"passo": passo, **v})
            else:
                pegg += 1
            riga["guardia_kl24"] = guardia
            salva("cardan_ultimo", "residui_ultimo", {"passo": passo})
            if a.tappe_dir:
                import shutil
                for n_ in ("cardan", "residui"):
                    shutil.copyfile(out / f"{n_}_ultimo.pt", Path(a.tappe_dir) / f"{n_}_{passo}.pt.tmp")
                    os.replace(Path(a.tappe_dir) / f"{n_}_{passo}.pt.tmp", Path(a.tappe_dir) / f"{n_}_{passo}.pt")
            torch.save({"optE": optE.state_dict(), "optD": optD.state_dict(), "storia": storia + [riga], "v0": v0,
                        "migliore": list(migl), "pegg": pegg, "passo": passo}, out / "ottimizzatori_ultimo.pt")
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
        CK.carica(resE, out / "cardan_migliore.pt"); RT.carica_latenti(resD, out / "residui_migliore.pt")
        CK.scrivi_json(resE, out / "cardan.json", {"passo": migl[1], "uso": "cardan_piega.py --pt cardan_migliore.pt"})
        dW = {n: RT.delta_file(m).float() for n, m in resD.items()}
        torch.save({"delta_file": dW, "rango": a.rango_densi, "famiglie": list(cfgD.famiglie),
                    "strati": RT.strati_di(resD), "passo": migl[1], "nome": a.nome,
                    "nota": "dW = diag(gamma) B A, ordine del FILE, base NON ruotata (cardan_piega_densi.py)"},
                   out / "residui_delta.pt")
        log(f"esportati cardan_migliore.pt (esperti) + residui_delta.pt ({len(dW)} densi), passo {migl[1]}")
    json.dump({"passo": migl[1], "sel": migl[0], "selezione": a.selezione, "modo": a.modo, "nome": a.nome},
              open(out / "esportazione.json", "w"), indent=1)
    log(f"FINE: migliore passo {migl[1]} sel {migl[0]:.5f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
