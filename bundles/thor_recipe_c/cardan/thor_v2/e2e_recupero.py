#!/usr/bin/env python3
"""e2e_recupero — addestra gli ESPERTI DI RECUPERO (e2e_rexp) su Thor-T R2B3m con la KL del MODELLO intero.

Tutto il resto dello studente e' FERMO (scale, codici, densi, router originali, norme): theta.requires_grad = False e
l'indietro si ferma al primo strato con esperti di recupero (`strati_allenati`). Dati: la cache del maestro di TT6
(q8-TT6b: h dopo output_norm del Q8, KL esatta sul vocabolario intero con la testa bf16 originale), lato `train`
(1024 righe x 4096: 512 base + 512 conoscenza) e `heldout` (64 righe) per il cancello.

Fasi:
  0 cancello d'identita': logit con gli esperti installati (D = 0) == logit senza, AL BIT (su una riga held-out)
  1 inizializzazione allineata al gradiente: --righe-init righe, raccolta (x, dKL/dy) per strato, k-means + SVD
  2 passi Adam sui latenti (lr separati router / G,U / D), held-out ogni --heldout-ogni passi, migliore tenuto
  3 esportazione: latenti (.pt) + byte TQ1_0 per strato (.npz) -> costruisci_recupero.py fa il GGUF

Uso (GPU, dopo YAQA, col lucchetto):
  E2E_GDN=veloce E2E_GDN_MODULO=gdn_veloce_corsa_v5 E2E_MOTORE=triton2 E2E_CONV_DIRETTA=1 E2E_SENZA_COESISTENZA=1 \\
  python e2e_recupero.py --uscita @@ROOT:depot@@/e2e/recupero/P1 --strati 10-29 --n-r 8 --k-r 2 --passi 32
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2e_rexp as RX                       # noqa: E402

GGUF = "@@ROOT:models@@/gguf/tony-tern/R2B3m/Thor-T-R2B3m.gguf"
TEACHER = "@@ROOT:depot@@/e2e_teacher_tt6/q8-TT6b"


def intervallo(s: str) -> tuple:
    out = []
    for p in s.split(","):
        a, b = p.split("-") if "-" in p else (p, p)
        out += range(int(a), int(b) + 1)
    return tuple(out)


def adesso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uscita", required=True)
    ap.add_argument("--gguf", default=GGUF)
    ap.add_argument("--teacher", default=TEACHER)
    ap.add_argument("--strati", default="10-29")
    ap.add_argument("--n-r", type=int, default=8)
    ap.add_argument("--k-r", type=int, default=2)
    ap.add_argument("--F", type=int, default=512)
    ap.add_argument("--piani-esperti", type=int, default=1)
    ap.add_argument("--piani-router", type=int, default=3)
    ap.add_argument("--passi", type=int, default=32)
    ap.add_argument("--righe-passo", type=int, default=8)
    ap.add_argument("--righe-init", type=int, default=2)
    ap.add_argument("--init", choices=("gradiente", "caso"), default="gradiente")
    ap.add_argument("--lr-r", type=float, default=1e-4, help="router")
    ap.add_argument("--lr-gamma", type=float, default=1e-2,
                    help="gamma per esperto: UNO scalare, Adam lo muove di ~lr a passo -> con 1e-4 in 32 passi |gamma| <= 3e-3 "
                         "(giocattolo, 32 passi, KL 0,499: gamma lr 1e-4 -> 0,312 ; 1e-3 -> 0,128 ; 1e-2 -> 0,066)")
    ap.add_argument("--lr-lat", type=float, default=3e-4, help="latenti G/U/D (se la sonda non sceglie)")
    ap.add_argument("--sonda-lr", default="3e-4,1e-3", help="griglia dei lr dei latenti; vuoto = niente sonda")
    ap.add_argument("--passi-sonda", type=int, default=4)
    ap.add_argument("--heldout-sonda", type=int, default=8)
    ap.add_argument("--riscaldamento", type=float, default=0.03)
    ap.add_argument("--topk", type=int, default=64, help="KL col maestro top-k in cache (+ secchio); 0 = KL piena")
    ap.add_argument("--cache-topk", default="@@ROOT:depot@@/e2e/recupero/cache_topk_q8-TT6b")
    ap.add_argument("--ablazione-righe", type=int, default=2)
    ap.add_argument("--senza-condiviso", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--heldout-righe", type=int, default=16)
    ap.add_argument("--heldout-ogni", type=int, default=8)
    ap.add_argument("--pezzo-kl", type=int, default=512)
    ap.add_argument("--seme", type=int, default=20261003)
    ap.add_argument("--gemm", choices=("fp16", "bf16", "fp32"), default="fp16")
    ap.add_argument("--ricetta", default="B3_ind2")
    ap.add_argument("--ore-max", type=float, default=3.5, help="tetto di tempo dell'allenamento (poi held-out + esportazione)")
    a = ap.parse_args(argv)

    import e2e_studente as S
    import e2e_allena as A
    out = Path(a.uscita)
    out.mkdir(parents=True, exist_ok=True)
    flog = open(out / "registro.log", "a")

    def log(*m):
        r = f"[{adesso()}] " + " ".join(str(x) for x in m)
        print(r, flush=True)
        flog.write(r + "\n"); flog.flush()

    random.seed(a.seme); np.random.seed(a.seme); torch.manual_seed(a.seme)
    dev = torch.device(a.device)
    strati = intervallo(a.strati)
    cfg = RX.ConfigRexp(n_r=a.n_r, k_r=a.k_r, F=a.F, piani_esperti=a.piani_esperti, piani_router=a.piani_router,
                        condiviso=not a.senza_condiviso, strati=strati)
    json.dump({"argv": sys.argv, "cfg": cfg.__dict__, "avvio": adesso()}, open(out / "config.json", "w"), indent=1)
    A.scegli_gdn(log)
    st = S.Studente(gguf=Path(a.gguf), strati=range(40), ricetta=a.ricetta,
                    motore_nome=os.environ.get("E2E_MOTORE", "triton"), device=a.device, log=log, densi_ternari="tutte")
    for p in st.parametri():
        p.requires_grad_(False)
    WS = A.testa_studente(st, "bf16x3")
    WT, sha_t = A.testa_maestro(dev, torch.bfloat16)
    src = A.SorgenteCache(a.teacher)
    car = A.Caricatore(src, dev)
    held = src.righe_previste("heldout")[: a.heldout_righe]
    train = src.righe_previste("train")

    def kl_righe(righe: list) -> float:
        """KL media per token (T||S) su righe held-out, senza grafo."""
        tot, n = 0.0, 0
        with torch.no_grad():
            for i in righe:
                lot = car.prendi("heldout", [i])
                hS = st.nascosto(lot.ids, inizi=lot.inizi)
                for j in range(0, hS.shape[0], a.pezzo_kl):
                    r = A.kl_pezzo(hS[j:j + a.pezzo_kl], WS, lot.hT[j:j + a.pezzo_kl], WT, False)
                    tot += float(r["kl"].sum()); n += int(r["kl"].numel())
        return tot / n

    # ── 0 cancello d'identita'
    lot = car.prendi("heldout", [held[0]])
    with torch.no_grad():
        h0 = st.nascosto(lot.ids, inizi=lot.inizi).clone()
    mods = RX.installa(st, cfg, device=dev, dtype_gemm={"fp16": torch.float16, "bf16": torch.bfloat16,
                                                         "fp32": torch.float32}[a.gemm])
    with torch.no_grad():
        h1 = st.nascosto(lot.ids, inizi=lot.inizi)
    ident = bool(torch.equal(h0, h1))
    log(f"cancello identita' (gamma = 0, {len(strati)} strati x {a.n_r} esperti + condiviso {cfg.condiviso}): {'AL BIT' if ident else 'DIVERSO'}")
    if not ident:
        log("ERRORE: la partenza non e' R2B3m al bit"); return 2
    del h0, h1
    st.strati_allenati = sorted(strati)            # indietro troncato al primo strato di recupero
    for m in mods.values():
        m.requires_grad_(True)

    # ── cache del maestro top-k (una volta, condivisa fra corse): log p_T dei k token piu' probabili per riga di train
    cache_k = Path(a.cache_topk) / f"k{a.topk}"
    cache_k.mkdir(parents=True, exist_ok=True)

    def topk_riga(i: int, lot=None) -> tuple:
        f = cache_k / f"r{i:05d}.pt"
        if f.exists():
            d = torch.load(f, map_location=dev)
            return d["idx"], d["lp"]
        lot = lot if lot is not None else car.prendi("train", [i])
        ii, ll = [], []
        for j in range(0, lot.hT.shape[0], a.pezzo_kl):
            lT = A._logit_maestro(lot.hT[j:j + a.pezzo_kl], WT, torch.float32)
            x, y = RX.topk_maestro(lT, a.topk); ii.append(x); ll.append(y)
        idx, lp = torch.cat(ii), torch.cat(ll)
        tmp = f.with_suffix(".parziale")
        torch.save({"idx": idx.cpu(), "lp": lp.cpu()}, tmp); os.replace(tmp, f)
        return idx, lp

    def passo_kl(righe: list, scala: float) -> float:
        tot = 0.0
        n_tok = None
        for m in mods.values():
            m.peso_aux = a.peso_aux_attivo / len(righe)
        for i in righe:
            lot = car.prendi("train", [i])
            if n_tok is None:
                n_tok = len(righe) * lot.ids.shape[1]
            stato = st.avanti(lot.ids, inizi=lot.inizi)
            xL = stato["xL"].detach().requires_grad_(True)
            with torch.enable_grad():
                if a.topk > 0:
                    idx, lp = topk_riga(i, lot)
                    kl = RX.KLTop.apply(st.esce(xL), idx, lp, lambda h: A._logit_studente(h, WS),
                                        lambda q: A._per_testa(q, WS), a.pezzo_kl)
                else:
                    kl = A.KLFusa.apply(st.esce(xL), WS, lot.hT, WT, a.pezzo_kl)
                (kl * (scala / n_tok)).backward()
            g = xL.grad
            st.indietro(stato, g)
            tot += float(kl.detach())
            del stato, g, xL
        for m in mods.values():
            m.peso_aux = 0.0
        return tot / n_tok

    # ── 1 inizializzazione (+ sensibilita' iniziale per strato: energia del gradiente della KL all'uscita del blocco)
    rng = random.Random(a.seme)
    ordine = train[:]
    rng.shuffle(ordine)
    t0 = time.time()
    a.peso_aux_attivo = 1.0
    sens = {}
    if a.init == "gradiente":
        for m in mods.values():
            m.sonda = {"x": [], "g": []}
        a.peso_aux_attivo = 0.0
        passo_kl(ordine[: a.righe_init], 1.0)
        a.peso_aux_attivo = 1.0
        diag = {}
        for L, m in mods.items():
            X = torch.cat(m.sonda["x"]); G = torch.cat(m.sonda["g"])
            m.sonda = None
            sens[L] = float(G.pow(2).sum(-1).mean())
            diag[L] = RX.inizializza_da_gradiente(m, X, G, seme=a.seme + L)
            diag[L]["energia_gradiente"] = sens[L]
            del X, G
        for m in mods.values():
            m.zero_grad(set_to_none=True)
        json.dump(diag, open(out / "init_diag.json", "w"))
        log(f"init dal gradiente: {a.righe_init} righe, {time.time() - t0:.0f} s; energia top-F media "
            f"{np.mean([np.mean(d['energia_topF']) for d in diag.values() if d['energia_topF']]):.3f}; "
            f"sensibilita' per strato (|dKL/dy|^2) {json.dumps({L: round(v, 9) for L, v in sens.items()})}")
    ordine = ordine[a.righe_init:]
    kl0 = kl_righe(held)
    log(f"held-out {len(held)} righe: KL al passo 0 = {kl0:.5f} (R2B3m)")
    RX.salva_latenti(mods, out / "latenti_init.pt", {"passo": 0, "kl_held": kl0})
    t_avvio = time.time()

    def allena(lr_lat: float, passi: int, righe_di, valuta_righe: list, salva: bool, etichetta: str) -> tuple:
        """passi Adam da latenti_init; ritorna (migliore KL, passo, storia)."""
        RX.carica_latenti(mods, out / "latenti_init.pt")
        gruppi = [{"params": [m.R for m in mods.values()], "lr": a.lr_r},
                  {"params": [p for m in mods.values() for p in (m.gamma,) + ((m.gamma_s,) if cfg.condiviso else ())],
                   "lr": a.lr_gamma},
                  {"params": [p for m in mods.values() for p in (m.G, m.U, m.D) + ((m.Gs, m.Us, m.Ds) if cfg.condiviso
                                                                                     else ())], "lr": lr_lat}]
        base = [g["lr"] for g in gruppi]
        opt = torch.optim.AdamW(gruppi, betas=(0.9, 0.99), weight_decay=0.0)
        k0 = kl0 if valuta_righe is held else kl_righe(valuta_righe)
        storia, migl = [{"passo": 0, "kl_held": k0}], (k0, 0)
        n_risc = max(1, int(a.riscaldamento * passi))
        for passo in range(1, passi + 1):
            t1 = time.time()
            if t1 - t_avvio > a.ore_max * 3600:
                log(f"{etichetta}: tetto di {a.ore_max} h raggiunto al passo {passo - 1}: chiudo")
                k = kl_righe(valuta_righe)
                if k < migl[0]:
                    migl = (k, passo - 1)
                    if salva:
                        RX.salva_latenti(mods, out / "latenti_migliore.pt", {"passo": passo - 1, "kl_held": k})
                break
            f = min(1.0, passo / n_risc) * (0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * max(0, passo - n_risc) /
                                                                             max(1, passi - n_risc))))
            for g, b in zip(opt.param_groups, base):
                g["lr"] = b * f
            righe = righe_di(passo)
            opt.zero_grad(set_to_none=True)
            kl_tr = passo_kl(righe, 1.0)
            tay = {L: round(m.taylor(), 9) for L, m in mods.items()}
            gn = torch.nn.utils.clip_grad_norm_(RX.parametri(mods), a.clip)
            opt.step()
            st_r = [m.stat for m in mods.values() if m.stat]
            riga = {"fase": etichetta, "passo": passo, "kl_train": kl_tr, "norma_grad": float(gn), "lr_f": f,
                    "s": time.time() - t1, "taylor": tay,
                    "gamma_medio": float(np.mean([m.gamma.detach().abs().mean().item() for m in mods.values()])),
                    "carico_max": float(np.max([d["carico_max"] for d in st_r])) if st_r else None,
                    "z": float(np.mean([d["z"] for d in st_r])) if st_r else None}
            if passo % a.heldout_ogni == 0 or passo == passi:
                k = kl_righe(valuta_righe)
                riga["kl_held"] = k
                if k < migl[0]:
                    migl = (k, passo)
                    if salva:
                        RX.salva_latenti(mods, out / "latenti_migliore.pt", {"passo": passo, "kl_held": k})
                if salva:
                    RX.salva_latenti(mods, out / f"latenti_{passo:03d}.pt", {"passo": passo, "kl_held": k})
            storia.append(riga)
            log(json.dumps(riga))
        return migl[0], migl[1], storia

    # ── 2a sonda dei lr dei latenti (passi corti, held-out ridotto, stesse righe per tutti i candidati)
    lrs = [float(x) for x in a.sonda_lr.split(",") if x.strip()]
    lr_lat = lrs[0] if lrs else a.lr_lat
    tutto = {"sonda": {}}
    usate = 0
    if len(lrs) > 1:
        righe_s = ordine[: a.passi_sonda * a.righe_passo]
        usate = len(righe_s)
        val_s = held[: a.heldout_sonda]
        esiti = {}
        for lr in lrs:
            k, p_, sto = allena(lr, a.passi_sonda, lambda pa: righe_s[(pa - 1) * a.righe_passo: pa * a.righe_passo],
                                val_s, False, f"sonda lr {lr:g}")
            esiti[lr] = sto[-1].get("kl_held", k)
            tutto["sonda"][str(lr)] = sto
        lr_lat = min(esiti, key=esiti.get)
        log(f"sonda lr latenti: {json.dumps({str(k): round(v, 6) for k, v in esiti.items()})} -> scelto {lr_lat:g}")
    ordine = ordine[usate:]

    # ── 2b allenamento
    def righe_di(passo):
        return [ordine[(passo - 1) * a.righe_passo + j] for j in range(a.righe_passo)] \
            if passo * a.righe_passo <= len(ordine) else rng.sample(train, a.righe_passo)
    migl_k, migl_p, storia = allena(lr_lat, a.passi, righe_di, held, True, "P")
    tutto["allenamento"] = storia
    if migl_p == 0:                                  # nessun miglioramento: si esporta la partenza (gamma = 0 = R2B3m)
        RX.carica_latenti(mods, out / "latenti_init.pt")
        RX.salva_latenti(mods, out / "latenti_migliore.pt", {"passo": 0, "kl_held": kl0})
    log(f"FINE: migliore passo {migl_p} KL held {migl_k:.5f} (passo 0 {kl0:.5f}, {100 * (migl_k / kl0 - 1):+.2f} %), "
        f"lr latenti {lr_lat:g}")
    RX.carica_latenti(mods, out / "latenti_migliore.pt")

    # ── 2c sensibilita' per strato a fine corsa (per l'allocazione di P2): ablazione esatta gamma = 0 strato per strato
    abl = {}
    if a.ablazione_righe > 0 and migl_p > 0:
        rr = held[: a.ablazione_righe]
        kb = kl_righe(rr)
        for L, m in mods.items():
            salv = [m.gamma.detach().clone()] + ([m.gamma_s.detach().clone()] if cfg.condiviso else [])
            with torch.no_grad():
                m.gamma.zero_()
                if cfg.condiviso:
                    m.gamma_s.zero_()
            abl[L] = kl_righe(rr) - kb
            with torch.no_grad():
                m.gamma.copy_(salv[0])
                if cfg.condiviso:
                    m.gamma_s.copy_(salv[1])
        log(f"ablazione per strato (dKL held {len(rr)} righe togliendo gli esperti di uno strato): "
            f"{json.dumps({L: round(v, 6) for L, v in abl.items()})}")
    tutto.update({"sensibilita_init": sens, "ablazione": abl, "lr_latenti": lr_lat})
    json.dump(tutto, open(out / "storia.json", "w"), indent=1)

    # ── 3 esportazione (latenti del migliore -> byte TQ1_0, gamma fuso nelle scale)
    t = RX.tensori_gguf(mods)
    np.savez(out / "rexp_tq1.npz", **t)
    json.dump({"k_r": a.k_r, "n_r": a.n_r, "condiviso": cfg.condiviso, "strati": list(strati), "passo": migl_p,
               "kl_held": migl_k, "kl0": kl0, "byte": int(sum(v.nbytes for v in t.values())), "n_tensori": len(t)},
              open(out / "esportazione.json", "w"), indent=1)
    log(f"esportati {len(t)} tensori, {sum(v.nbytes for v in t.values()) / 2**20:.1f} MiB -> {out / 'rexp_tq1.npz'}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
