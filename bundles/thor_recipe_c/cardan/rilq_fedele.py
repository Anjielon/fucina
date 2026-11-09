#!/usr/bin/env python3
"""RILQ-FEDELE (10/10 notte): copy of rilq_thor_kl4.py (unchanged for the queued cardan-densi-kl4 run) with
  --perdita rilq   the loss of the RILQ paper (Lee et al., arXiv 2412.01129): W_mse x MSE between the student and
                   teacher LAST hidden state (h after output_norm, the e2e_teacher cache) + W_ce x cross-entropy on the
                   TRUE next token of the text (inside each segment); default 0.5 / 0.5 as in the paper.
  --teacher        the C4 teacher cache (RILQ_FEDELE/teacher/q8-maestro-c4: C4 en 256x512 train, 32x512 held-out)
  --teacher-kl24   the cache used for the KL24 evaluation (default q8-maestro, the usual 24 held-out rows)
  --selezione rilq best step / early stop on the held-out C4 RILQ loss (paper); KL24 and KL4 held-out logged
  --solo-valuta P  diagnosis: identity gate, then (selection value, KL24, KL4 held-out, RILQ held-out) with the
                   correction at zero and with the latents of P loaded (NOT folded); writes solo_valuta.json, exit 0
Below, the original rilq_thor_kl4.py docstring.

SELRISP option (10/10, NOT used by the queued cardan-densi run, which uses rilq_thor.py unchanged):
same as rilq_thor.py plus --kl4-peso W: each step adds W x mean KL4 (MMLU answer token, 4-way softmax vs the q8
teacher MCLP) on one packed row of DISJOINT calibration questions (SELRISP/calib, cais/mmlu validation+dev).
W = 0 (default) = rilq_thor.py exactly. Packing/loss = gradienti_densi.pacchi_kl4 (segments, zero-shot).
rilq_thor.py — RILQ FEDELE su Thor-T (7/10 notte, priorita' di Angelo). Ricetta riprodotta dall'audit A5
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



def perdita_rilq(hd, hT, tgt, ok, logit_fn, per_fn, w_mse: float, w_ce: float, pezzo: int, con_grad: bool) -> tuple:
    """RILQ paper loss on one row: (MSE(hS, hT), CE on true next token, dL/dh or None) with
    L = w_mse * mean((hS - hT)^2) + w_ce * mean_valid(-log p(next token)); logits = logit_fn(h) (h @ W^T), and
    per_fn(q) = q @ W (gradient back through the head). Chunked over tokens so the full logits never live at once."""
    diff = hd.float() - hT.float()
    mse = float((diff * diff).mean())
    g = (w_mse * 2.0 / diff.numel()) * diff if con_grad else None
    nv = int(ok.sum()); ce_tot = 0.0
    for j in range(0, hd.shape[0], pezzo):
        m = ok[j:j + pezzo]
        if not bool(m.any()):
            continue
        lp = torch.log_softmax(logit_fn(hd[j:j + pezzo].float()).float(), -1)
        t = tgt[j:j + pezzo]
        ce_tot += float(-(lp.gather(1, t[:, None])[:, 0] * m).sum())
        if con_grad:
            pr = lp.exp(); pr[torch.arange(pr.shape[0], device=pr.device), t] -= 1.0
            pr *= (m.float() * (w_ce / max(1, nv)))[:, None]
            g[j:j + pezzo] += per_fn(pr).float()
        del lp
    return mse, ce_tot / max(1, nv), g

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
    ap.add_argument("--kl4-peso", type=float, default=0.0)
    ap.add_argument("--kl4-calib", default="@@ROOT:models@@/gguf/tony-tern/SELRISP/calib/calib_mmlu.jsonl")
    ap.add_argument("--kl4-maestro", default="@@ROOT:models@@/gguf/tony-tern/SELRISP/calib/mclp_maestro.log")
    ap.add_argument("--kl4-n", type=int, default=896, help="calibration questions used (the rest stays held-out)")
    ap.add_argument("--kl-peso", type=float, default=1.0, help="weight of the full-text KL (0 = KL4 only)")
    ap.add_argument("--perdita", choices=("kl", "rilq"), default="kl")
    ap.add_argument("--peso-mse", type=float, default=0.5)
    ap.add_argument("--peso-ce", type=float, default=0.5)
    ap.add_argument("--teacher-kl24", default=TEACHER, help="cache for the KL24 evaluation (24 held-out rows)")
    ap.add_argument("--solo-valuta", default=None, help="diagnosis: evaluate zero vs these latents, then exit")
    ap.add_argument("--selezione", choices=("kl24", "kl4", "rilq"), default="kl24",
                    help="best step chosen on held-out KL24 (text) or held-out KL4 (calibration questions never trained on)")
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
    train = src.righe_previste("train")
    held_c4 = src.righe_previste("heldout")              # RILQ-FEDELE: held-out rows of the TRAINING cache
    if Path(a.teacher_kl24).resolve() == Path(a.teacher).resolve():
        src24, car24 = src, car
    else:
        src24 = A.SorgenteCache(a.teacher_kl24); car24 = A.Caricatore(src24, dev)
    held = src24.righe_previste("heldout")[: a.heldout_righe]
    log(f"maestro {a.teacher} ({etichetta}): {len(train)} righe train, held-out {len(held)} righe; testa maestro sha {sha_t}")

    def kl_righe(righe: list) -> float:
        tot, n = 0.0, 0
        with torch.no_grad():
            for i in righe:
                lot = car24.prendi("heldout", [i])
                hS = st.nascosto(lot.ids, inizi=lot.inizi)
                for j in range(0, hS.shape[0], a.pezzo_kl):
                    r = A.kl_pezzo(hS[j:j + a.pezzo_kl], WS, lot.hT[j:j + a.pezzo_kl], WT, False)
                    tot += float(r["kl"].sum()); n += int(r["kl"].numel())
        return tot / n

    # ── 0 cancello d'identita' (gamma = 0): h finale con i residui installati == senza, AL BIT
    lot = car24.prendi("heldout", [held[0]])
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

    pacchi4, held4 = [], []
    if a.kl4_peso > 0 or a.selezione == "kl4" or a.solo_valuta or a.perdita == "rilq":
        import types as _ty
        sys.path.insert(0, "@@ROOT:models@@/build/leve/raffina")
        import gradienti_densi as GD
        ns = dict(calib_maestro=a.kl4_maestro, calib_jsonl=a.kl4_calib, seme=7, calib_pacco=4096)
        tr4, _ = GD.pacchi_kl4(_ty.SimpleNamespace(calib_n=a.kl4_n, **ns), log)
        if a.kl4_peso > 0:
            pacchi4 = tr4
        usate4 = {q["i"] for p in tr4 for q in p}
        tutte4, _ = GD.pacchi_kl4(_ty.SimpleNamespace(calib_n=0, **ns), log)
        cur, nt = [], 0                                # held-out = the calibration questions never trained on
        for q in (q for p in tutte4 for q in p if q["i"] not in usate4):
            if cur and nt + len(q["ids"]) > 4096:
                held4.append(cur); cur, nt = [], 0
            cur.append(q); nt += len(q["ids"])
        if cur:
            held4.append(cur)
        log(f"kl4 held-out: {sum(len(p) for p in held4)} domande in {len(held4)} pacchi (disgiunte dalle {len(usate4)} usate)")
        if a.selezione == "kl4" and not held4:
            log("ERRORE: --selezione kl4 senza domande held-out"); return 2
    giro4 = {"k": 0}
    ultimo_rilq = {"v": float("nan")}

    def kl4_held() -> float:
        tot, n = 0.0, 0
        with torch.no_grad():
            for p in held4:
                ids = torch.tensor([sum((q["ids"] for q in p), [])], dtype=torch.long, device=WS.device)
                inizi = torch.zeros((1, ids.shape[1]), dtype=torch.long); ult, o = [], 0
                for q in p:
                    inizi[0, o] = 1; o += len(q["ids"]); ult.append(o - 1)
                hS = st.nascosto(ids, inizi=inizi)
                hS = hS.reshape(-1, hS.shape[-1])
                let = torch.tensor(p[0]["lettere"], device=WS.device)
                Wl = (WS.hi[let].float() + WS.lo[let].float()) if hasattr(WS, "hi") else WS[let].to(torch.float32)
                lS = torch.log_softmax(hS[torch.tensor(ult, device=WS.device)].float() @ Wl.T, -1)
                pT = torch.stack([q["pT"] for q in p]).to(WS.device).float()
                tot += float((pT * (torch.log(pT.clamp_min(1e-30)) - lS)).sum()); n += len(p)
        return tot / n

    def bersagli(lot) -> tuple:
        """RILQ-FEDELE: next-token targets inside each segment (position t -> ids[t+1] unless t+1 starts a segment)."""
        ids = lot.ids.reshape(-1)
        T = ids.shape[0]
        ok = torch.ones(T, dtype=torch.bool, device=ids.device); ok[-1] = False
        if lot.inizi is not None:
            iz = lot.inizi.reshape(-1).to(ids.device)
            ok[:-1] &= iz[1:] == 0
        tgt = torch.zeros(T, dtype=torch.long, device=ids.device); tgt[:-1] = ids[1:]
        return tgt, ok

    def rilq_grad(hd: torch.Tensor, lot, con_grad: bool) -> tuple:
        tgt, ok = bersagli(lot)
        return perdita_rilq(hd, lot.hT.reshape(hd.shape), tgt, ok, lambda h: A._logit_studente(h, WS),
                            lambda q: A._per_testa(q, WS), a.peso_mse, a.peso_ce, a.pezzo_kl, con_grad)

    def rilq_held() -> float:
        tm, tc, n = 0.0, 0.0, 0
        with torch.no_grad():
            for i in held_c4:
                lot = car.prendi("heldout", [i])
                hS = st.nascosto(lot.ids, inizi=lot.inizi).reshape(-1, lot.hT.shape[-1])
                m_, c_, _ = rilq_grad(hS, lot, False)
                tm += m_; tc += c_; n += 1
        return a.peso_mse * tm / n + a.peso_ce * tc / n

    def valuta() -> tuple:
        """(selection value, KL24, KL4 held-out or nan); RILQ held-out loss stored in ultimo_rilq"""
        k24 = kl_righe(held)
        k4 = kl4_held() if held4 else float("nan")
        kr = rilq_held() if (a.perdita == "rilq" or a.selezione == "rilq") else float("nan")
        ultimo_rilq["v"] = kr
        sel = {"kl4": k4, "rilq": kr}.get(a.selezione, k24)
        return sel, k24, k4

    if a.solo_valuta:
        t_ev = time.time()
        z = valuta(); zr = ultimo_rilq["v"]
        RT.carica_latenti(res, Path(a.solo_valuta))
        c = valuta(); cr = ultimo_rilq["v"]
        d = {"latenti": a.solo_valuta, "zero": {"kl24": z[1], "kl4_held": z[2], "rilq_held": zr},
             "corretto_non_piegato": {"kl24": c[1], "kl4_held": c[2], "rilq_held": cr},
             "dkl4_held": c[2] - z[2], "dkl24_rel": c[1] / z[1] - 1, "secondi": round(time.time() - t_ev, 1),
             "kl4_held_domande": sum(len(p) for p in held4)}
        json.dump(d, open(out / "solo_valuta.json", "w"), indent=1)
        log("SOLO-VALUTA " + json.dumps(d))
        return 0

    def passo_kl4() -> float:
        p = pacchi4[giro4["k"] % len(pacchi4)]; giro4["k"] += 1
        ids = torch.tensor([sum((q["ids"] for q in p), [])], dtype=torch.long, device=WS.device)
        inizi = torch.zeros((1, ids.shape[1]), dtype=torch.long); ult, o = [], 0
        for q in p:
            inizi[0, o] = 1; o += len(q["ids"]); ult.append(o - 1)
        stato = st.avanti(ids, inizi=inizi)
        xL = stato["xL"].detach().requires_grad_(True)
        with torch.enable_grad():
            hh = st.esce(xL).reshape(-1, xL.shape[-1])
            let = torch.tensor(p[0]["lettere"], device=WS.device)
            Wl = (WS.hi[let].float() + WS.lo[let].float()) if hasattr(WS, "hi") else WS[let].to(torch.float32)
            lS = torch.log_softmax(hh[torch.tensor(ult, device=WS.device)].float() @ Wl.T, -1)
            pT = torch.stack([q["pT"] for q in p]).to(WS.device).float()
            k4 = (pT * (torch.log(pT.clamp_min(1e-30)) - lS)).sum(-1).mean()
            (a.kl4_peso * k4).backward()
        st.indietro(stato, xL.grad)
        del stato, xL
        return float(k4.detach())

    def passo_kl(righe: list) -> float:
        k4p = None
        if pacchi4:
            k4p = passo_kl4()
            log(f"KL4 calib (passo, peso {a.kl4_peso}): {k4p:.5f}")
        if a.perdita == "rilq":
            tot = 0.0
            for i in righe:
                lot = car.prendi("train", [i])
                stato = st.avanti(lot.ids, inizi=lot.inizi)
                xL = stato["xL"].detach().requires_grad_(True)
                with torch.enable_grad():
                    hh = st.esce(xL)
                    hd = hh.detach().reshape(-1, hh.shape[-1])
                    m_, c_, g = rilq_grad(hd, lot, True)
                    hh.backward(g.reshape(hh.shape).to(hh.dtype) / len(righe))
                st.indietro(stato, xL.grad)
                tot += a.peso_mse * m_ + a.peso_ce * c_
                del stato, xL, hh, hd, g
            return tot / len(righe)
        if a.kl_peso == 0:
            return k4p
        tot, n_tok = 0.0, 0
        lots = [car.prendi("train", [i]) for i in righe]
        n_tok = sum(int(l.ids.numel()) for l in lots)
        for lot in lots:
            stato = st.avanti(lot.ids, inizi=lot.inizi)
            xL = stato["xL"].detach().requires_grad_(True)
            with torch.enable_grad():
                kl = A.KLFusa.apply(st.esce(xL), WS, lot.hT, WT, a.pezzo_kl)
                (a.kl_peso * kl / n_tok).backward()
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
        sel0, kl0, k40 = valuta()
        s_eval = time.time() - t_ev
        log(f"held-out {len(held)} righe: KL al passo 0 = {kl0:.5f} ({a.nome}), KL4 held-out {k40:.5f}, "
            f"selezione {a.selezione}, valutazione {s_eval:.0f} s")
        riga_cicli("ATTIVO", f"RILQ-THOR {a.nome}: KL held-out passo 0 = {kl0:.5f} ({len(held)} righe {etichetta}, "
                             f"valutazione {s_eval:.0f} s); residui ternari r{a.rango} su {len(res)} proiezioni {a.famiglie} "
                             f"strati {cfg.strati[0]}-{cfg.strati[-1]}, perdita KL piena, Adam {a.lr:g}/{a.lr_gamma:g} -> {out}")
        RT.salva_latenti(res, out / "residui_init.pt", {"passo": 0, "kl_held": kl0, "kl4_held": k40})
        storia, migl, pegg, p_ini = [{"passo": 0, "kl_held": kl0, "kl4_held": k40, "rilq_held": ultimo_rilq["v"]}], (sel0, 0), 0, 1
        RT.salva_latenti(res, out / "residui_migliore.pt", {"passo": 0, "kl_held": kl0, "kl4_held": k40})

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
            k, k24, k4 = valuta()
            riga["kl_held"] = round(k24, 6); riga["kl4_held"] = round(k4, 6); riga["rilq_held"] = round(ultimo_rilq["v"], 6)
            riga["s_eval"] = round(time.time() - t_ev, 1)
            if k < migl[0] - 1e-5:
                migl, pegg = (k, passo), 0
                RT.salva_latenti(res, out / "residui_migliore_tmp.pt", {"passo": passo, "kl_held": k24, "kl4_held": k4})
                os.replace(out / "residui_migliore_tmp.pt", out / "residui_migliore.pt")
            else:
                pegg += 1
            storia.append(riga)
            log(json.dumps(riga))
            salva_ultimo(passo, kl0, migl, storia, pegg)
            s0 = {"kl4": storia[0]["kl4_held"], "rilq": storia[0].get("rilq_held", kl0)}.get(a.selezione, kl0)
            riga_cicli("ATTIVO", f"RILQ-THOR {a.nome}: passo {passo} KL24 held {k24:.5f} ({100 * (k24 / kl0 - 1):+.2f} %), "
                                 f"KL4 held-out {k4:.5f}, selezione {a.selezione}: migliore passo {migl[1]} "
                                 f"{100 * (migl[0] / s0 - 1):+.2f} %")
            if pegg >= a.pazienza:
                log(f"arresto: {pegg} valutazioni senza miglioramento"); break
        else:
            storia.append(riga)
            log(json.dumps(riga))
    # valutazione finale se l'ultimo passo non e' stato valutato
    if passo >= p_ini and storia[-1].get("passo") == passo and "kl_held" not in storia[-1]:
        k, k24, k4 = valuta()
        storia[-1]["kl_held"] = k24; storia[-1]["kl4_held"] = k4
        if k < migl[0] - 1e-5:
            migl = (k, passo)
            RT.salva_latenti(res, out / "residui_migliore.pt", {"passo": passo, "kl_held": k24, "kl4_held": k4})
        salva_ultimo(passo, kl0, migl, storia, pegg)
        log(f"valutazione finale passo {passo}: KL24 held {k24:.5f} KL4 held-out {k4:.5f}")
    json.dump({"storia": storia, "kl0": kl0, "migliore": list(migl), "secondi": time.time() - t_avvio},
              open(out / "storia.json", "w"), indent=1)
    s0 = {"kl4": storia[0].get("kl4_held", kl0), "rilq": storia[0].get("rilq_held", kl0)}.get(a.selezione, kl0)
    log(f"FINE: migliore passo {migl[1]} {a.selezione} held {migl[0]:.5f} (passo 0 {s0:.5f}, {100 * (migl[0] / s0 - 1):+.2f} %)")

    esp = {"passo": migl[1], "sel_held": migl[0], "selezione": a.selezione, "sel0": s0, "kl0": kl0, "nome": a.nome,
           "kl4_peso": a.kl4_peso, "kl_peso": a.kl_peso, "perdita": a.perdita, "peso_mse": a.peso_mse, "peso_ce": a.peso_ce}
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
