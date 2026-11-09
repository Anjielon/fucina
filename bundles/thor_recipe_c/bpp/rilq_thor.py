#!/usr/bin/env python3
"""rilq_thor.py — RILQ FEDELE su Thor-T (7/10 notte, priorita' di Angelo). Ricetta riprodotta dall'audit A5
(AUDIT_RIPRODUZIONE.md §A5.7, AUDIT/script/rilq_mini.py) portata sul Thor con gli strumenti di E1:

  adattatori   residui ternari di rango basso (residui_ternari.py, A e B ternari via STE, gamma = 0 => partenza al bit)
               rango 64, 1 piano, su TUTTI gli strati 0-39, famiglie attn (q/k/v/o) + gdn (qkv/gate/ssm_out).
               Posizione: nell'audit tutti i lineari -52,3 %, sola attenzione -50,7 %, solo MLP -49,2 %: equivalenti.
               Il motore (thor-e1 build-rexp) legge _lra/_lrb solo sulle famiglie attn/gdn -> qui la sola attenzione
               (variante iv), su tutta la profondita' come nella ricetta.
  perdita      KL esatta sul vocabolario intero (T||S) sui logit del modello intero contro il maestro Q8 (cache --teacher,
               h dopo output_norm + testa bf16 originale): variante (vi) dell'audit, la migliore (-72,9 % al passo 300,
               -77,1 % al passo 1000).
  passi/lotti  Adam (beta 0,9/0,999, decadimento 0), lr 1e-4 sui latenti A, B con riscaldamento lineare di 10 passi poi
               costante (come rilq_mini.py). gamma: lr 2e-4 = stesso passo effettivo su dW del LoRA alpha/r = 0,25
               (B ternario normalizzato per riga ha scala ~0,15; 1e-4 * 0,25 / 0,15 ~ 1,7e-4). Lotto = 4096 token
               (= 8 x 512 della ricetta) = 1 riga del corpus del maestro per passo. Clip della norma 1,0.
  selezione    KL held-out su --heldout-righe righe (24) del maestro ogni --heldout-ogni passi; arresto dopo --pazienza
               valutazioni senza miglioramento; si esporta il migliore SOLO se batte il passo 0.
  ripresa      ogni valutazione salva residui_ultimo.pt + ottimizzatore + storia: --riprendi riparte da li'.

8/10 --famiglie accetta anche shexp (ffn_gate/up/down_shexp, shared expert denso di OGNI strato; residui_ternari
ramo cloud/lettera-e1-shexp) -> tensori blk.L.ffn_<x>_shexp_lra/_lrb, letti dal motore thor-e1 ramo cloud/rexp-lr-shexp.

Mai dati MMLU: il corpus del maestro (base + conoscenza) e' quello di P2c.

9/10 cure di RILQT/COMBO_A_shexp/DIAGNOSI_DIVARIO.md:
  P1  lo studente legge anche i quarti piani densi *4 e i secondi piani degli esperti exps2 (+ mappe) e fallisce se un
      tensore del GGUF resta fuori dal forward (e2e_studente.controlla_copertura); --kl-partenza-motore: KL realizzata
      della partenza misurata dal motore sulle STESSE righe held-out; se |KL torch passo 0 / KL motore - 1| supera
      --soglia-coerenza (3 %) ci si ferma prima del passo 1 (codice 2).
  P2  KL o norma del gradiente non finite: il passo si SALTA prima di opt.step() (parametri e stato di Adam intatti),
      con messaggi distinti per NaN e overflow; dopo 3 salti di fila si ricarica l'ultimo salvataggio con lr dimezzato.
  P3  le etichette del maestro escono da --teacher (prima "q8-TT6b" era scritto a mano anche quando era q8-maestro).
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

QUI = Path(__file__).resolve().parent
REC = QUI / "recupero_7dabd6f" / "thor_v2"
# 10/10 (CARDAN dei densi): stesso interruttore di cardan_kd.py: E2E_KERNEL_NUOVI=1 -> copia thor_v2_kqat con i kernel
# nuovi (default degli interruttori = quelli provati al bit sul passo reale di CARDAN). Assente = albero vecchio.
if os.environ.get("E2E_KERNEL_NUOVI") == "1":
    REC = QUI / "recupero_7dabd6f" / "thor_v2_kqat"
    os.environ.setdefault("E2E_KQAT", "cache_densi,silu,tile,amax,mm16_grande,guardia")
sys.path.insert(0, str(QUI))
sys.path.insert(1, str(REC))
import residui_ternari as RT                 # noqa: E402

TEACHER = "@@ROOT:depot@@/e2e_teacher/q8-maestro"      # 9/10 P3: la cache usata davvero (prima q8-TT6b)
CICLI = "@@ROOT:forge@@/stato/cicli.tsv"


def intervallo(s: str) -> tuple:
    out = []
    for p in s.split(","):
        if "-" in p:
            a, b = p.split("-"); out += list(range(int(a), int(b) + 1))
        elif p.strip():
            out.append(int(p))
    return tuple(out)


def adesso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def riga_cicli(stato: str, testo: str) -> None:
    t = testo.replace("\t", " ").replace("\n", " ").replace("rc=", "codice ")
    with open(CICLI, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M')}\t{stato}\t{t}\n")


def motivo_salto(kl: float, norma_grad: float) -> "str | None":
    """9/10 P2: None se il passo si puo' fare, altrimenti il motivo (messaggi distinti per NaN e per overflow)."""
    if math.isnan(kl):
        return "NaN nella KL"
    if math.isinf(kl):
        return "overflow della KL (inf)"
    if math.isnan(norma_grad):
        return "NaN nel gradiente"
    if math.isinf(norma_grad):
        return "overflow del gradiente (norma inf)"
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uscita", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--nome", default="R2B3m")
    ap.add_argument("--teacher", default=TEACHER)
    ap.add_argument("--strati", default="0-39")
    ap.add_argument("--famiglie", default="attn,gdn")
    ap.add_argument("--rango", type=int, default=64)
    ap.add_argument("--piani", type=int, default=1)
    ap.add_argument("--passi", type=int, default=1000, help="massimo; decidono arresto e --ore-max")
    ap.add_argument("--righe-passo", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr-gamma", type=float, default=2e-4)
    ap.add_argument("--riscaldamento", type=int, default=10)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--heldout-righe", type=int, default=24)
    ap.add_argument("--heldout-ogni", type=int, default=50)
    ap.add_argument("--pazienza", type=int, default=4)
    ap.add_argument("--pezzo-kl", type=int, default=512)
    ap.add_argument("--gemm", choices=("fp16", "bf16", "fp32"), default="fp32")
    ap.add_argument("--ore-max", type=float, default=7.0, help="tetto dell'allenamento dall'avvio (poi esportazione)")
    ap.add_argument("--seme", type=int, default=20261007)
    ap.add_argument("--riprendi", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ricetta", default="B3_ind2")
    ap.add_argument("--kl-partenza-motore", type=float, default=None,
                    help="9/10 P1: KL realizzata della partenza (motore, stesse righe held-out); None = nessun cancello")
    ap.add_argument("--soglia-coerenza", type=float, default=0.03, help="9/10 P1: scarto relativo massimo torch/motore")
    ap.add_argument("--salti-max", type=int, default=3, help="9/10 P2: salti di fila prima di ricaricare a lr/2")
    ap.add_argument("--continuo", action="store_true",
                    help="10/10 CARDAN dei densi: ramo di rango basso CONTINUO (B=0, gamma=1), esportato come dW nel "
                         "file (residui_delta.pt) da piegare nei piani esistenti con cardan_piega_densi.py")
    a = ap.parse_args(argv)
    etichetta = Path(a.teacher).name                  # 9/10 P3: il nome vero della cache del maestro

    import e2e_studente as S
    import e2e_allena as A
    out = Path(a.uscita)
    out.mkdir(parents=True, exist_ok=True)
    flog = open(out / "registro.log", "a")

    def log(*m):
        r = f"[{adesso()}] " + " ".join(str(x) for x in m)
        print(r, flush=True)
        flog.write(r + "\n"); flog.flush()

    t_avvio = time.time()
    random.seed(a.seme); np.random.seed(a.seme); torch.manual_seed(a.seme)
    dev = torch.device(a.device)
    cfg = RT.ConfigResidui(rango=a.rango, piani=a.piani, strati=intervallo(a.strati),
                           famiglie=RT.famiglie_da(a.famiglie), seme=a.seme, continuo=a.continuo)
    json.dump({"argv": sys.argv if argv is None else argv, "cfg": cfg.__dict__, "avvio": adesso(),
               "codice": str(QUI)}, open(out / ("config_ripresa.json" if a.riprendi else "config.json"), "w"),
              indent=1, default=list)
    A.scegli_gdn(log)
    try:
        st = S.Studente(gguf=Path(a.gguf), strati=range(40), ricetta=a.ricetta,
                        motore_nome=os.environ.get("E2E_MOTORE", "triton"), device=a.device, log=log,
                        densi_ternari="tutte")
    except ValueError as e:                            # 9/10 P1: cancello di copertura (o piani incoerenti) fallito
        log(f"ERRORE: {e}")
        riga_cicli("ERRORE", f"RILQ-THOR {a.nome}: lo studente non e' il GGUF ({str(e)[:200]}): allenamento NON avviato")
        return 2
    for p in st.parametri():
        p.requires_grad_(False)
    WS = A.testa_studente(st, "bf16x3")
    WT, sha_t = A.testa_maestro(dev, torch.bfloat16)
    src = A.SorgenteCache(a.teacher)
    car = A.Caricatore(src, dev)
    held = src.righe_previste("heldout")[: a.heldout_righe]
    train = src.righe_previste("train")
    log(f"maestro {a.teacher} ({etichetta}): {len(train)} righe train, held-out {len(held)} righe; testa maestro sha {sha_t}")

    def kl_righe(righe: list) -> float:
        tot, n = 0.0, 0
        with torch.no_grad():
            for i in righe:
                lot = car.prendi("heldout", [i])
                hS = st.nascosto(lot.ids, inizi=lot.inizi)
                for j in range(0, hS.shape[0], a.pezzo_kl):
                    r = A.kl_pezzo(hS[j:j + a.pezzo_kl], WS, lot.hT[j:j + a.pezzo_kl], WT, False)
                    tot += float(r["kl"].sum()); n += int(r["kl"].numel())
        return tot / n

    # ── 0 cancello d'identita' (gamma = 0): h finale con i residui installati == senza, AL BIT
    lot = car.prendi("heldout", [held[0]])
    with torch.no_grad():
        h0 = st.nascosto(lot.ids, inizi=lot.inizi).clone()
    res = RT.installa(st, cfg, device=dev)
    dt_g = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[a.gemm]
    for m in res.values():
        m.dtype_gemm = dt_g
    with torch.no_grad():
        h1 = st.nascosto(lot.ids, inizi=lot.inizi)
    ident = bool(torch.equal(h0, h1))
    npar = sum(p.numel() for m in res.values() for p in (m.A, m.B))
    log(f"cancello identita' (gamma = 0; {len(res)} proiezioni in {len(RT.strati_di(res))} strati, rango {a.rango}, "
        f"{npar / 1e6:.2f} M latenti): {'AL BIT' if ident else 'DIVERSO'}")
    if not ident:
        log(f"ERRORE: la partenza non e' {a.nome} al bit"); return 2
    # 10/10 cancello di PRESENZA (modo continuo): con B = 0 il cancello d'identita' passerebbe anche se un percorso
    # veloce dello studente scavalcasse il ramo (gradiente nullo, allenamento a vuoto). Si accende un B casuale piccolo
    # sul primo e sull'ultimo residuo: h deve cambiare; poi B torna a 0 e h deve tornare al bit.
    if a.continuo:
        nomi_r = sorted(res)
        for nm in {nomi_r[0], nomi_r[-1]}:
            with torch.no_grad():
                res[nm].B.normal_(0.0, 1e-3)
                h2 = st.nascosto(lot.ids, inizi=lot.inizi)
                res[nm].B.zero_()
                h3 = st.nascosto(lot.ids, inizi=lot.inizi)
            cambia, torna = not torch.equal(h2, h1), bool(torch.equal(h3, h1))
            log(f"cancello di presenza {nm}: ramo acceso cambia h = {cambia}, spento torna al bit = {torna}")
            if not (cambia and torna):
                log(f"ERRORE: il ramo continuo di {nm} non e' nel percorso dello studente (o non torna al bit)")
                riga_cicli("ERRORE", f"RILQ-THOR {a.nome}: cancello di presenza fallito su {nm}: allenamento NON avviato")
                return 2
            del h2, h3
    del h0, h1, lot
    st.strati_allenati = RT.strati_di(res)            # indietro fino al primo strato con residui
    for m in res.values():
        m.requires_grad_(True)

    gruppi = [{"params": [p for n in sorted(res) for p in (res[n].A, res[n].B)], "lr": a.lr},
              {"params": [res[n].gamma for n in sorted(res)], "lr": a.lr_gamma}]
    base = [g["lr"] for g in gruppi]
    opt = torch.optim.Adam(gruppi, betas=(0.9, 0.999), weight_decay=0.0)

    rng = random.Random(a.seme)
    ordine = train[:]
    rng.shuffle(ordine)

    def righe_di(passo: int) -> list:                 # deterministico dal passo: la ripresa rivede le stesse righe
        return [ordine[((passo - 1) * a.righe_passo + j) % len(ordine)] for j in range(a.righe_passo)]

    def passo_kl(righe: list) -> float:
        tot, n_tok = 0.0, 0
        lots = [car.prendi("train", [i]) for i in righe]
        n_tok = sum(int(l.ids.numel()) for l in lots)
        for lot in lots:
            stato = st.avanti(lot.ids, inizi=lot.inizi)
            xL = stato["xL"].detach().requires_grad_(True)
            with torch.enable_grad():
                kl = A.KLFusa.apply(st.esce(xL), WS, lot.hT, WT, a.pezzo_kl)
                (kl / n_tok).backward()
            st.indietro(stato, xL.grad)
            tot += float(kl.detach())
            del stato, xL
        return tot / n_tok

    stato_lr = {"fattore": 1.0}                       # 9/10 P2: dimezzato a ogni ricarica dopo salti di fila

    def salva_ultimo(passo: int, kl0: float, migl: tuple, storia: list, pegg: int) -> None:
        meta = {"passo": passo, "kl0": kl0, "migliore": list(migl), "pegg": pegg, "fattore_lr": stato_lr["fattore"]}
        RT.salva_latenti(res, out / "residui_ultimo_tmp.pt", meta)
        torch.save({"opt": opt.state_dict(), "storia": storia, **meta}, out / "ottimizzatore_ultimo_tmp.pt")
        os.replace(out / "residui_ultimo_tmp.pt", out / "residui_ultimo.pt")
        os.replace(out / "ottimizzatore_ultimo_tmp.pt", out / "ottimizzatore_ultimo.pt")

    if a.riprendi and (out / "ottimizzatore_ultimo.pt").exists():
        RT.carica_latenti(res, out / "residui_ultimo.pt")
        d = torch.load(out / "ottimizzatore_ultimo.pt", map_location=dev, weights_only=False)
        opt.load_state_dict(d["opt"])
        storia, kl0, migl, pegg, p_ini = d["storia"], d["kl0"], tuple(d["migliore"]), d["pegg"], d["passo"] + 1
        stato_lr["fattore"] = float(d.get("fattore_lr", 1.0))
        base = [b * stato_lr["fattore"] for b in base]
        log(f"RIPRESA dal passo {d['passo']}: KL0 {kl0:.5f}, migliore {migl}, fattore lr {stato_lr['fattore']:g}")
    else:
        t_ev = time.time()
        kl0 = kl_righe(held)
        s_eval = time.time() - t_ev
        log(f"held-out {len(held)} righe: KL al passo 0 = {kl0:.5f} ({a.nome}), valutazione {s_eval:.0f} s")
        riga_cicli("ATTIVO", f"RILQ-THOR {a.nome}: KL held-out passo 0 = {kl0:.5f} ({len(held)} righe {etichetta}, "
                             f"valutazione {s_eval:.0f} s); residui ternari r{a.rango} su {len(res)} proiezioni {a.famiglie} "
                             f"strati {cfg.strati[0]}-{cfg.strati[-1]}, perdita KL piena, Adam {a.lr:g}/{a.lr_gamma:g} -> {out}")
        RT.salva_latenti(res, out / "residui_init.pt", {"passo": 0, "kl_held": kl0})
        storia, migl, pegg, p_ini = [{"passo": 0, "kl_held": kl0}], (kl0, 0), 0, 1
        RT.salva_latenti(res, out / "residui_migliore.pt", {"passo": 0, "kl_held": kl0})

    # ── 9/10 P1 cancello di coerenza torch <-> motore: lo studente deve essere il GGUF (R2B3m: 0,5 %; COMBO A senza la
    # cura: +66 %). Vale anche alla ripresa (kl0 salvato).
    if a.kl_partenza_motore is not None:
        rif = float(a.kl_partenza_motore)
        scarto = kl0 / rif - 1 if rif > 0 and math.isfinite(rif) else float("inf")
        log(f"coerenza: KL torch passo 0 {kl0:.5f} contro motore {rif:.5f} ({100 * scarto:+.2f} %, soglia "
            f"{100 * a.soglia_coerenza:.1f} %)")
        if not (math.isfinite(kl0) and abs(scarto) <= a.soglia_coerenza):
            log(f"ERRORE: KL torch passo 0 {kl0:.5f} contro motore {rif:.5f}: lo studente non e' il GGUF")
            riga_cicli("ERRORE", f"RILQ-THOR {a.nome}: cancello di coerenza fallito, KL torch {kl0:.5f} contro motore "
                                 f"{rif:.5f} ({100 * scarto:+.2f} %): allenamento NON avviato")
            return 2
    else:
        log("coerenza: --kl-partenza-motore assente, cancello torch <-> motore NON eseguito")

    def ricarica_dopo_salti(passo: int) -> None:
        """9/10 P2: dopo --salti-max salti di fila, ultimo salvataggio (o la partenza) e lr dimezzato."""
        stato_lr["fattore"] *= 0.5
        for i in range(len(base)):
            base[i] *= 0.5
        if (out / "ottimizzatore_ultimo.pt").exists():
            RT.carica_latenti(res, out / "residui_ultimo.pt")
            d = torch.load(out / "ottimizzatore_ultimo.pt", map_location=dev, weights_only=False)
            opt.load_state_dict(d["opt"])
            da = f"residui_ultimo.pt (passo {d['passo']})"
        else:
            RT.carica_latenti(res, out / "residui_init.pt")
            opt.state.clear()                         # nessun salvataggio di Adam ancora: stato azzerato
            da = "residui_init.pt (passo 0, Adam azzerato)"
        log(f"RICARICA al passo {passo} dopo {a.salti_max} salti di fila: {da}, lr x{stato_lr['fattore']:g}")
        riga_cicli("ATTIVO", f"RILQ-THOR {a.nome}: {a.salti_max} passi non finiti di fila, ricaricato {da}, "
                             f"lr x{stato_lr['fattore']:g}")

    tempi = []
    salti = 0                                         # 9/10 P2: passi saltati di fila
    passo = p_ini - 1
    for passo in range(p_ini, a.passi + 1):
        t1 = time.time()
        if t1 - t_avvio > a.ore_max * 3600:
            log(f"tetto di {a.ore_max} h raggiunto: mi fermo prima del passo {passo}")
            passo -= 1
            break
        f = min(1.0, passo / max(1, a.riscaldamento))
        for g, b in zip(opt.param_groups, base):
            g["lr"] = b * f
        opt.zero_grad(set_to_none=True)
        kl_tr = passo_kl(righe_di(passo))
        gn = torch.nn.utils.clip_grad_norm_(RT.parametri(res), a.clip)
        motivo = motivo_salto(kl_tr, float(gn))
        if motivo is not None:                         # 9/10 P2: niente opt.step(): parametri e Adam intatti
            opt.zero_grad(set_to_none=True)
            salti += 1
            riga = {"passo": passo, "saltato": motivo, "kl_train": kl_tr, "norma_grad": float(gn), "salti_di_fila": salti,
                    "s": round(time.time() - t1, 1)}
            storia.append(riga)
            log(f"PASSO SALTATO ({motivo}): " + json.dumps(riga))
            if salti >= a.salti_max:
                ricarica_dopo_salti(passo)
                salti = 0
            continue
        salti = 0
        opt.step()
        if not all(bool(torch.isfinite(p).all()) for p in RT.parametri(res)):   # non deve accadere: ultima difesa
            log(f"ERRORE: parametri non finiti dopo opt.step() al passo {passo}: ricarico")
            ricarica_dopo_salti(passo)
            continue
        s = time.time() - t1
        tempi.append(s)
        riga = {"passo": passo, "kl_train": round(kl_tr, 6), "norma_grad": round(float(gn), 5), "lr_f": round(f, 3),
                "s": round(s, 1), "gamma_medio": float(np.mean([m.gamma.detach().abs().mean().item()
                                                                for m in res.values()]))}
        if passo == p_ini + 2:
            sp = float(np.mean(tempi[1:3])) if len(tempi) >= 3 else float(np.mean(tempi))
            ore200 = 200 * sp / 3600
            riga_cicli("ATTIVO", f"RILQ-THOR {a.nome}: primi 3 passi {', '.join(f'{x:.0f}' for x in tempi[:3])} s "
                                 f"(1 riga x 4096 token per passo, indietro su 40 strati); a regime ~{sp:.0f} s/passo, "
                                 f"200 passi ~{ore200:.1f} h + valutazioni held-out")
        if passo % a.heldout_ogni == 0:
            t_ev = time.time()
            k = kl_righe(held)
            riga["kl_held"] = round(k, 6)
            riga["s_eval"] = round(time.time() - t_ev, 1)
            if k < migl[0] - 1e-5:
                migl, pegg = (k, passo), 0
                RT.salva_latenti(res, out / "residui_migliore_tmp.pt", {"passo": passo, "kl_held": k})
                os.replace(out / "residui_migliore_tmp.pt", out / "residui_migliore.pt")
            else:
                pegg += 1
            storia.append(riga)
            log(json.dumps(riga))
            salva_ultimo(passo, kl0, migl, storia, pegg)
            riga_cicli("ATTIVO", f"RILQ-THOR {a.nome}: passo {passo} KL held {k:.5f} ({100 * (k / kl0 - 1):+.2f} % "
                                 f"dal passo 0), migliore passo {migl[1]} {100 * (migl[0] / kl0 - 1):+.2f} %")
            if pegg >= a.pazienza:
                log(f"arresto: {pegg} valutazioni senza miglioramento"); break
        else:
            storia.append(riga)
            log(json.dumps(riga))
    # valutazione finale se l'ultimo passo non e' stato valutato
    if passo >= p_ini and storia[-1].get("passo") == passo and "kl_held" not in storia[-1]:
        k = kl_righe(held)
        storia[-1]["kl_held"] = k
        if k < migl[0] - 1e-5:
            migl = (k, passo)
            RT.salva_latenti(res, out / "residui_migliore.pt", {"passo": passo, "kl_held": k})
        salva_ultimo(passo, kl0, migl, storia, pegg)
        log(f"valutazione finale passo {passo}: KL held {k:.5f}")
    json.dump({"storia": storia, "kl0": kl0, "migliore": list(migl), "secondi": time.time() - t_avvio},
              open(out / "storia.json", "w"), indent=1)
    log(f"FINE: migliore passo {migl[1]} KL held {migl[0]:.5f} (passo 0 {kl0:.5f}, {100 * (migl[0] / kl0 - 1):+.2f} %)")

    esp = {"passo": migl[1], "kl_held": migl[0], "kl0": kl0, "nome": a.nome}
    if migl[1] > 0 and a.continuo:
        RT.carica_latenti(res, out / "residui_migliore.pt")
        dW = {n: RT.delta_file(m).float() for n, m in res.items()}
        torch.save({"delta_file": dW, "rango": a.rango, "famiglie": list(cfg.famiglie), "strati": RT.strati_di(res),
                    "passo": migl[1], "kl_held": migl[0], "kl0": kl0, "nome": a.nome,
                    "nota": "dW = diag(gamma) B A, ordine del FILE (tensore GGUF), base NON ruotata"},
                   out / "residui_delta.pt")
        esp.update({"n_tensori": len(dW), "continuo": True,
                    "norma_rel_media": float(np.mean([float(v.norm()) for v in dW.values()]))})
        log(f"esportate {len(dW)} correzioni dense continue -> residui_delta.pt (da piegare: cardan_piega_densi.py)")
    elif migl[1] > 0:
        RT.carica_latenti(res, out / "residui_migliore.pt")
        tr, info = RT.tensori_gguf(res)
        np.savez(out / "residui_tq1.npz", **tr)
        json.dump({"tensori": info, "rango": a.rango, "piani": a.piani, "rp": RT.RP, "famiglie": list(cfg.famiglie),
                   "strati": RT.strati_di(res), "passo": migl[1], "kl_held": migl[0], "kl0": kl0,
                   "uso_motore": "cur = mul_mat(W,x)[+piani] + mul_mat(<fam>_lrb, mul_mat(<fam>_lra, x))"},
                  open(out / "residui.json", "w"), indent=1)
        esp.update({"n_tensori": len(tr), "byte": int(sum(v.nbytes for v in tr.values()))})
        log(f"esportati {len(tr)} tensori residui, {esp['byte'] / 2**20:.2f} MiB -> residui_tq1.npz")
    json.dump(esp, open(out / "esportazione.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
