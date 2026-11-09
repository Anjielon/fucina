#!/usr/bin/env python3
"""cardan_kd.py — the DISTILLATION of CARDAN (arXiv 2609.21137 §3.3, Eq. 3) on Thor-T, in the ternary flow (9/10).

Built on rilq_thor.py (same student, same teacher cache, same loss, same gates), with the RILQ adapters replaced by the
CARDAN branch of the routed experts (cardan_kd_moduli.py): g_e += C_e (D^T x), u_e += C'_e (D'^T x), D shared by the
256 experts of a layer, C_e private, on every layer 0-39.
  student   the whole ternary host (PV12P: experts with exps2, dense *4 planes, cure P1 of 9/10): every block reads the
            output of the ternary blocks below it, so the correction is learned for the TERNARY flow; the target is the
            teacher's distribution (q8-maestro cache), never W_bf16 (lesson of 2/10: a bf16 weight target undoes the
            compensations of the forge).
  loss      exact KL(T||S) on the full vocabulary of the final logits (paper: tau^2 KL on the top-256 + lambda CE;
            the exact full-vocab KL dominates it and is the loss that RILQ validated on this student).
  start     D = App. A.1 basis (cardan_base.py, rotated), C = 0: the student IS the host at the bit at step 0
            (identity gate), and --kl-partenza-motore checks KL torch step 0 vs the engine's realised KL (3 %).
  export    cardan_migliore.pt (best held-out step); the fold into the existing planes is cardan_piega.py.
Everything else (P2 skip of non-finite steps, reload at lr/2, held-out selection, patience, resume) is rilq_thor.py's.
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
# 9/10 E2E_KERNEL_NUOVI=1: the same student with the new kernels (ricerche/KERNEL_QAT_SOTTO_3H.md), copy thor_v2_kqat
# made by kernel_qat/applica_kqat.py; default switches = the ones proven BIT-EXACT on the real CARDAN step (C, D
# gradients 160/160 equal, kernel_qat/confronta_cardan.py). Unset = the old tree, untouched.
if os.environ.get("E2E_KERNEL_NUOVI") == "1":
    REC = QUI / "recupero_7dabd6f" / "thor_v2_kqat"
    os.environ.setdefault("E2E_KQAT", "cache_densi,silu,tile,amax,mm16_grande,guardia")   # 9/10: gradienti di C,D al bit (160/160) sul passo reale
sys.path.insert(0, str(QUI))
sys.path.insert(1, str(REC))
import residui_ternari as RT                 # noqa: E402
import cardan_kd_moduli as CK                 # noqa: E402

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
    ap.add_argument("--rango", type=int, default=16)
    ap.add_argument("--base-a1", default=None, help="directory of cardan_costruisci.py (strato_LL.npz): D init")
    ap.add_argument("--passi", type=int, default=1000, help="massimo; decidono arresto e --ore-max")
    ap.add_argument("--righe-passo", type=int, default=1)
    ap.add_argument("--lr-c", type=float, default=None, help="Adam lr of C (default: --frazione-c x rms of the A.1 C)")
    ap.add_argument("--lr-d", type=float, default=None, help="Adam lr of D (default: --frazione-d x rms of D0)")
    ap.add_argument("--frazione-c", type=float, default=0.005)
    ap.add_argument("--frazione-d", type=float, default=0.001)
    ap.add_argument("--riscaldamento", type=int, default=10)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--heldout-righe", type=int, default=24)
    ap.add_argument("--heldout-ogni", type=int, default=50)
    ap.add_argument("--pazienza", type=int, default=4)
    ap.add_argument("--pezzo-kl", type=int, default=512)
    ap.add_argument("--ore-max", type=float, default=7.0, help="tetto dell'allenamento dall'avvio (poi esportazione)")
    ap.add_argument("--seme", type=int, default=20261007)
    ap.add_argument("--riprendi", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ricetta", default="B3_ind2")
    ap.add_argument("--kl-partenza-motore", type=float, default=None,
                    help="9/10 P1: KL realizzata della partenza (motore, stesse righe held-out); None = nessun cancello")
    ap.add_argument("--soglia-coerenza", type=float, default=0.03, help="9/10 P1: scarto relativo massimo torch/motore")
    ap.add_argument("--salti-max", type=int, default=3, help="9/10 P2: salti di fila prima di ricaricare a lr/2")
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
    strati = intervallo(a.strati)
    json.dump({"argv": sys.argv if argv is None else argv, "strati": strati, "rango": a.rango, "avvio": adesso(),
               "codice": str(QUI)}, open(out / ("config_ripresa.json" if a.riprendi else "config.json"), "w"),
              indent=1, default=list)
    A.scegli_gdn(log)
    try:
        st = S.Studente(gguf=Path(a.gguf), strati=range(40), ricetta=a.ricetta,
                        motore_nome=os.environ.get("E2E_MOTORE", "triton"), device=a.device, log=log,
                        densi_ternari="tutte")
    except ValueError as e:                            # 9/10 P1: cancello di copertura (o piani incoerenti) fallito
        log(f"ERRORE: {e}")
        riga_cicli("ERRORE", f"CARDAN-KD {a.nome}: lo studente non e' il GGUF ({str(e)[:200]}): allenamento NON avviato")
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
    tutti = lambda r: CK.parametri(r, "C") + CK.parametri(r, "D")   # noqa: E731
    res = CK.installa(st, strati, a.rango, Path(a.base_a1) if a.base_a1 else None, dev)
    with torch.no_grad():
        h1 = st.nascosto(lot.ids, inizi=lot.inizi)
    ident = bool(torch.equal(h0, h1))
    npar = sum(p.numel() for p in CK.parametri(res, "C") + CK.parametri(res, "D"))
    log(f"cancello identita' (C = 0; CARDAN su {len(res)} strati, gate+up, rango {a.rango}, {npar / 1e6:.2f} M "
        f"parametri): {'AL BIT' if ident else 'DIVERSO'}")
    if not ident:
        log(f"ERRORE: la partenza non e' {a.nome} al bit"); return 2
    del h0, h1, lot
    st.strati_allenati = sorted(res)                  # backward down to the first layer with a CARDAN branch
    for m in res.values():
        m.requires_grad_(True)
    # learning rates on the scale of the parameters (Adam moves each coordinate by ~lr per step)
    rms_d = float(torch.cat([p.detach().reshape(-1) for p in CK.parametri(res, "D")]).float().pow(2).mean().sqrt())
    rms_c = None
    if a.base_a1:
        cc = []
        for L in sorted(res):
            z = np.load(Path(a.base_a1) / f"strato_{L:02d}.npz")
            cc += [z[k].view(np.float16).astype(np.float32).reshape(-1) for k in z.files if k.endswith("_lrc.weight")]
        rms_c = float(np.sqrt(np.mean(np.concatenate(cc).astype(np.float64) ** 2)))
    lr_c = a.lr_c if a.lr_c is not None else a.frazione_c * (rms_c if rms_c else 1e-3)
    lr_d = a.lr_d if a.lr_d is not None else a.frazione_d * rms_d
    log(f"lr: C {lr_c:.3g} (rms C A.1 {rms_c}), D {lr_d:.3g} (rms D0 {rms_d:.3g})")
    gruppi = [{"params": CK.parametri(res, "C"), "lr": lr_c}, {"params": CK.parametri(res, "D"), "lr": lr_d}]
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
        CK.salva(res, out / "cardan_ultimo_tmp.pt", meta)
        torch.save({"opt": opt.state_dict(), "storia": storia, **meta}, out / "ottimizzatore_ultimo_tmp.pt")
        os.replace(out / "cardan_ultimo_tmp.pt", out / "cardan_ultimo.pt")
        os.replace(out / "ottimizzatore_ultimo_tmp.pt", out / "ottimizzatore_ultimo.pt")

    if a.riprendi and (out / "ottimizzatore_ultimo.pt").exists():
        CK.carica(res, out / "cardan_ultimo.pt")
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
        riga_cicli("ATTIVO", f"CARDAN-KD {a.nome}: KL held-out passo 0 = {kl0:.5f} ({len(held)} righe {etichetta}, "
                             f"valutazione {s_eval:.0f} s); CARDAN r{a.rango} gate+up su {len(res)} strati "
                             f"{strati[0]}-{strati[-1]}, perdita KL piena, Adam C {lr_c:.3g} D {lr_d:.3g} -> {out}")
        CK.salva(res, out / "cardan_init.pt", {"passo": 0, "kl_held": kl0})
        storia, migl, pegg, p_ini = [{"passo": 0, "kl_held": kl0}], (kl0, 0), 0, 1
        CK.salva(res, out / "cardan_migliore.pt", {"passo": 0, "kl_held": kl0})

    # ── 9/10 P1 cancello di coerenza torch <-> motore: lo studente deve essere il GGUF (R2B3m: 0,5 %; COMBO A senza la
    # cura: +66 %). Vale anche alla ripresa (kl0 salvato).
    if a.kl_partenza_motore is not None:
        rif = float(a.kl_partenza_motore)
        scarto = kl0 / rif - 1 if rif > 0 and math.isfinite(rif) else float("inf")
        log(f"coerenza: KL torch passo 0 {kl0:.5f} contro motore {rif:.5f} ({100 * scarto:+.2f} %, soglia "
            f"{100 * a.soglia_coerenza:.1f} %)")
        if not (math.isfinite(kl0) and abs(scarto) <= a.soglia_coerenza):
            log(f"ERRORE: KL torch passo 0 {kl0:.5f} contro motore {rif:.5f}: lo studente non e' il GGUF")
            riga_cicli("ERRORE", f"CARDAN-KD {a.nome}: cancello di coerenza fallito, KL torch {kl0:.5f} contro motore "
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
            CK.carica(res, out / "cardan_ultimo.pt")
            d = torch.load(out / "ottimizzatore_ultimo.pt", map_location=dev, weights_only=False)
            opt.load_state_dict(d["opt"])
            da = f"cardan_ultimo.pt (passo {d['passo']})"
        else:
            CK.carica(res, out / "cardan_init.pt")
            opt.state.clear()                         # nessun salvataggio di Adam ancora: stato azzerato
            da = "cardan_init.pt (passo 0, Adam azzerato)"
        log(f"RICARICA al passo {passo} dopo {a.salti_max} salti di fila: {da}, lr x{stato_lr['fattore']:g}")
        riga_cicli("ATTIVO", f"CARDAN-KD {a.nome}: {a.salti_max} passi non finiti di fila, ricaricato {da}, "
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
        gn = torch.nn.utils.clip_grad_norm_(tutti(res), a.clip)
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
        if not all(bool(torch.isfinite(p).all()) for p in tutti(res)):   # non deve accadere: ultima difesa
            log(f"ERRORE: parametri non finiti dopo opt.step() al passo {passo}: ricarico")
            ricarica_dopo_salti(passo)
            continue
        s = time.time() - t1
        tempi.append(s)
        riga = {"passo": passo, "kl_train": round(kl_tr, 6), "norma_grad": round(float(gn), 5), "lr_f": round(f, 3),
                "s": round(s, 1), **CK.statistiche(res)}
        if passo == p_ini + 2:
            sp = float(np.mean(tempi[1:3])) if len(tempi) >= 3 else float(np.mean(tempi))
            ore200 = 200 * sp / 3600
            riga_cicli("ATTIVO", f"CARDAN-KD {a.nome}: primi 3 passi {', '.join(f'{x:.0f}' for x in tempi[:3])} s "
                                 f"(1 riga x 4096 token per passo, indietro su 40 strati); a regime ~{sp:.0f} s/passo, "
                                 f"200 passi ~{ore200:.1f} h + valutazioni held-out")
        if passo % a.heldout_ogni == 0:
            t_ev = time.time()
            k = kl_righe(held)
            riga["kl_held"] = round(k, 6)
            riga["s_eval"] = round(time.time() - t_ev, 1)
            if k < migl[0] - 1e-5:
                migl, pegg = (k, passo), 0
                CK.salva(res, out / "cardan_migliore_tmp.pt", {"passo": passo, "kl_held": k})
                os.replace(out / "cardan_migliore_tmp.pt", out / "cardan_migliore.pt")
            else:
                pegg += 1
            storia.append(riga)
            log(json.dumps(riga))
            salva_ultimo(passo, kl0, migl, storia, pegg)
            riga_cicli("ATTIVO", f"CARDAN-KD {a.nome}: passo {passo} KL held {k:.5f} ({100 * (k / kl0 - 1):+.2f} % "
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
            CK.salva(res, out / "cardan_migliore.pt", {"passo": passo, "kl_held": k})
        salva_ultimo(passo, kl0, migl, storia, pegg)
        log(f"valutazione finale passo {passo}: KL held {k:.5f}")
    json.dump({"storia": storia, "kl0": kl0, "migliore": list(migl), "secondi": time.time() - t_avvio},
              open(out / "storia.json", "w"), indent=1)
    log(f"FINE: migliore passo {migl[1]} KL held {migl[0]:.5f} (passo 0 {kl0:.5f}, {100 * (migl[0] / kl0 - 1):+.2f} %)")

    esp = {"passo": migl[1], "kl_held": migl[0], "kl0": kl0, "nome": a.nome}
    if migl[1] > 0:
        CK.scrivi_json(res, out / "cardan.json", {"passo": migl[1], "kl_held": migl[0], "kl0": kl0,
                                                  "uso": "cardan_piega.py --pt cardan_migliore.pt"})
    json.dump(esp, open(out / "esportazione.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
