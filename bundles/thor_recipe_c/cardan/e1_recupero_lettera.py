#!/usr/bin/env python3
"""e1_recupero_lettera.py — E1 = correzione del LEGAME lettera <-> contenuto sopra Thor-T R2B3m FERMO.

Due capacita' nuove, tutte ternarie, accendibili insieme o da sole (--modo):
  esperti  : ESPERTI DI RECUPERO ternari (e2e_rexp, blocco MoE) — l'E1 di prima, invariato al bit con le stesse opzioni
  residui  : RESIDUI TERNARI DI RANGO BASSO sulle proiezioni dell'attenzione degli strati CENTRALI (residui_ternari.py:
             A, B ternari via STE, rango 16-64, ri-ternarizzati a ogni passo, gamma = 0 => partenza al bit). Motivo:
             la lettera si decide in poche teste di attenzione degli strati centrali (arXiv 2407.15018); lm_head ed
             esperti FFN correggono il priore, non il legame.
  entrambi : le due insieme
Adattatore minimo del driver e2e_recupero.py: stesso studente (e2e_studente), stessa esportazione degli esperti
(RX.tensori_gguf -> rexp_tq1.npz -> costruisci_recupero.py); i residui vanno in residui_tq1.npz + residui.json ->
e1_scrivi_residui_gguf.py (originale al byte + tensori TQ1_0 in piu').

  dati      : items_teacher.jsonl di lettera (4 rotazioni per item, teacher_logprobs delle 4 lettere dal maestro Q8).
              RIFIUTO di ogni riga con fonte MMLU / ARC / OpenBookQA / RACE / MCTest (e1_perdita_legame.FONTI_VIETATE),
              in allenamento E in selezione. Gruppi interi IMPACCHETTATI in modo «segmenti» (`inizi`).
  perdita   : e1_perdita_legame.perdita_legame sui 4 logit lettera dell'ultimo token di ogni prompt
              = KL maestro alla STESSA rotazione (--target raw) + lam * coerenza fra rotazioni riportata al contenuto
              + --peso-pif (0,3) * negativi puntuali PIF (2406.01026). Nessuna perdita LM sul testo.
  selezione : SOLO held-out LogiQA / MedMCQA (--fonti-selezione); mai MMLU, ARC, OBQA, RACE, MCTest.

Codice del recupero: copia congelata in recupero_7dabd6f/thor_v2 (commit 7dabd6f, ramo esperti-recupero, sola lettura).
Prove CPU: test_e1_recupero_lettera.py (esperti, studente finto), test_residui_ternari.py (residui + perdita + GGUF).

Uso (GPU, col lucchetto, dopo E1.2 maestro):
  E2E_GDN=veloce E2E_GDN_MODULO=gdn_veloce_corsa_v5 E2E_MOTORE=triton2 E2E_CONV_DIRETTA=1 E2E_SENZA_COESISTENZA=1 \\
  python e1_recupero_lettera.py --uscita OUT --dati items_teacher.jsonl --modo entrambi --strati 30-39 \\
      --residui-strati 12-27 --residui-rango 32 --passi 150
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

QUI = Path(__file__).resolve().parent
REC = QUI / "recupero_7dabd6f" / "thor_v2"
sys.path.insert(0, str(QUI))
sys.path.insert(1, str(REC))
import e2e_rexp as RX                        # noqa: E402  (copia congelata)
import residui_ternari as RT                 # noqa: E402
import e1_veloce as V                        # noqa: E402  (cache del prefisso, budget di tempo)
from common import read_jsonl                # noqa: E402
from e1_letter_kd import collate_targets     # noqa: E402
from e1_perdita_legame import (FONTI_SELEZIONE, fonte_vietata, metriche_selezione,  # noqa: E402
                               perdita_legame)

GGUF = "@@ROOT:models@@/gguf/tony-tern/R2B3m/Thor-T-R2B3m.gguf"
TOKENIZER = "@@ROOT:models@@/gguf/thor-src-hf"
LETTER_IDS = (357, 417, 351, 414)


def intervallo(s: str) -> tuple:
    out = []
    for p in s.split(","):
        a, b = p.split("-") if "-" in p else (p, p)
        out += range(int(a), int(b) + 1)
    return tuple(out)


def codificatore(path: str, lids) -> callable:
    """encode(testo) -> id, con il controllo del confine del token lettera IN CONTESTO (dopo "Answer:")."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    enc = lambda s: tok(s, add_special_tokens=True).input_ids          # noqa: E731
    p0 = "Q?\nA. x\nB. y\nC. z\nD. w\nAnswer:"
    got = [enc(p0 + s)[len(enc(p0)):] for s in (" A", " B", " C", " D")]
    if got != [[i] for i in lids]:
        raise SystemExit(f"id lettere del tokenizer {got} != {list(lids)}")
    return enc


# ── dati ─────────────────────────────────────────────────────────────────────
def carica_gruppi(path: str, tokenize, letter_ids) -> dict:
    """{split: [gruppo]}; gruppo = 4 righe (rot 0..3) con 'ids' (prompt tokenizzato) e i campi di collate_targets."""
    righe = read_jsonl(path)
    bad = [r["id"] for r in righe if fonte_vietata(r.get("source", ""))]
    if bad:
        raise SystemExit(f"righe MMLU/ARC/OBQA/RACE/MCTest nei dati ({len(bad)}): rifiuto ({bad[:3]})")
    per_g: dict = {}
    for r in righe:
        if "teacher_logprobs" not in r or len(r["teacher_logprobs"]) != 4:
            continue
        per_g.setdefault(r["group_id"], []).append(r)
    out: dict = {"train": [], "heldout": []}
    scartati = 0
    for g, rr in per_g.items():
        if len(rr) != 4 or len({r["rot"] for r in rr}) != 4:
            scartati += 1
            continue
        rr.sort(key=lambda r: r["rot"])
        for r in rr:
            r["ids"] = tokenize(r["prompt"])
        out.setdefault(rr[0]["split"], []).append(rr)
    out["scartati"] = scartati
    return out


def gruppi_selezione(held: list, fonti: tuple, n: int, seme: int) -> list:
    """Held-out di SELEZIONE: solo le fonti chieste (LogiQA/MedMCQA), mescolato a seme fisso, equilibrato per fonte."""
    vietate = [f for f in fonti if fonte_vietata(f)]
    if vietate:
        raise SystemExit(f"fonti di selezione vietate: {vietate}")
    per = {f: [g for g in held if str(g[0].get("source", "")).lower() == f] for f in fonti}
    rng = random.Random(seme)
    for v in per.values():
        rng.shuffle(v)
    out, i = [], 0
    while len(out) < n and any(i < len(v) for v in per.values()):
        out += [v[i] for v in per.values() if i < len(v)][: n - len(out)]
        i += 1
    return out


def impacchetta(gruppi: list, device) -> dict:
    """Gruppi interi -> una riga in modo segmenti: ids (1, T), inizi (1, T), ultimi (n,), bersagli."""
    rows = [r for g in gruppi for r in g]
    ids, inizi, ultimi, t = [], [], [], 0
    for r in rows:
        n = len(r["ids"])
        ids += r["ids"]
        inizi += [1] + [0] * (n - 1)
        t += n
        ultimi.append(t - 1)
    b = collate_targets(rows)
    b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
    return {"ids": torch.tensor([ids], device=device), "inizi": torch.tensor([inizi], device=device),
            "ultimi": torch.tensor(ultimi, device=device), "batch": b, "rows": rows}


def lotti(gruppi: list, token_max: int) -> list:
    """Gruppi in ordine -> liste di gruppi con somma dei token <= token_max (almeno un gruppo per lotto)."""
    out, cur, n = [], [], 0
    for g in gruppi:
        k = sum(len(r["ids"]) for r in g)
        if cur and n + k > token_max:
            out.append(cur); cur, n = [], 0
        cur.append(g); n += k
    if cur:
        out.append(cur)
    return out


def argomenti(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--uscita", required=True)
    ap.add_argument("--dati", required=True, help="items_teacher.jsonl (teacher_logprobs delle 4 lettere)")
    ap.add_argument("--gguf", default=GGUF)
    ap.add_argument("--tokenizer", default=TOKENIZER)
    ap.add_argument("--letter-ids", default=",".join(map(str, LETTER_IDS)))
    ap.add_argument("--modo", choices=("esperti", "residui", "entrambi"), default="esperti")
    # esperti di recupero
    ap.add_argument("--strati", default="20-39", help="strati degli esperti di recupero")
    ap.add_argument("--n-r", type=int, default=8)
    ap.add_argument("--k-r", type=int, default=2)
    ap.add_argument("--F", type=int, default=512)
    ap.add_argument("--piani-esperti", type=int, default=1)
    ap.add_argument("--piani-router", type=int, default=3)
    ap.add_argument("--senza-condiviso", action="store_true")
    # residui ternari di rango basso sull'attenzione
    ap.add_argument("--residui-strati", default="12-27", help="strati CENTRALI dei residui (Thor: 40 strati)")
    ap.add_argument("--residui-famiglie", default="attn,gdn",
                    help="attn (q,k,v,o: strati full_attention), gdn (qkv, gate, ssm_out), o famiglie singole")
    ap.add_argument("--residui-rango", type=int, default=32)
    ap.add_argument("--residui-piani", type=int, default=1)
    ap.add_argument("--lr-res", type=float, default=3e-4, help="latenti A, B dei residui")
    ap.add_argument("--lr-res-gamma", type=float, default=1e-2)
    # perdita
    ap.add_argument("--w-kd", type=float, default=1.0, help="peso della KL col maestro alla stessa rotazione")
    ap.add_argument("--lam", type=float, default=0.5, help="peso della coerenza fra rotazioni (al contenuto)")
    ap.add_argument("--peso-pif", type=float, default=0.3, help="peso dei negativi puntuali PIF (0 = spenti)")
    ap.add_argument("--pif-negativi-soli", action="store_true")
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--target", choices=("raw", "debiased"), default="raw",
                    help="raw = maestro alla STESSA rotazione; debiased = media geometrica delle rotazioni")
    # selezione
    ap.add_argument("--fonti-selezione", default=",".join(FONTI_SELEZIONE))
    ap.add_argument("--criterio", choices=("loss", "kd", "acc"), default="loss",
                    help="loss = perdita intera held-out; kd = KD + coerenza (senza oro); acc = accuratezza held-out")
    ap.add_argument("--heldout-gruppi", type=int, default=250)
    ap.add_argument("--heldout-ogni", type=int, default=20)
    # ottimizzazione
    ap.add_argument("--passi", type=int, default=200)
    ap.add_argument("--gruppi-passo", type=int, default=32, help="gruppi (x4 prompt) per passo")
    ap.add_argument("--token-lotto", type=int, default=4096, help="token per riga impacchettata (micro-lotto)")
    ap.add_argument("--init", choices=("gradiente", "caso"), default="gradiente")
    ap.add_argument("--gruppi-init", type=int, default=32)
    ap.add_argument("--lr-r", type=float, default=1e-4)
    ap.add_argument("--lr-gamma", type=float, default=1e-2)
    ap.add_argument("--lr-lat", type=float, default=3e-4)
    ap.add_argument("--riscaldamento", type=float, default=0.03)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--peso-aux", type=float, default=1.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gemm", choices=("fp16", "bf16", "fp32"), default="fp16")
    ap.add_argument("--ricetta", default="B3_ind2")
    ap.add_argument("--seme", type=int, default=20261003)
    ap.add_argument("--ore-max", type=float, default=3.0, help="tetto DURO: oltre, chiude (il lr non e' arrivato a zero)")
    # velocita' (R007): non cambiano cio' che si calcola, solo quante volte lo si ricalcola
    ap.add_argument("--cache-prefisso", choices=("si", "no"), default="si",
                    help="stati d'ingresso dello primo strato allenato calcolati UNA volta (e1_veloce.CachePrefisso)")
    ap.add_argument("--token-cache", type=int, default=32768, help="token per lotto nel calcolo della cache")
    ap.add_argument("--cache-file", default="", help="se dato: salva/riprende la cache (stessa corsa, stessi dati)")
    ap.add_argument("--tol-cache", type=float, default=5e-2, help="scarto massimo dei logit lettera cache vs avanti pieno")
    ap.add_argument("--token-valuta", type=int, default=16384, help="token per lotto della valutazione held-out (senza grad)")
    ap.add_argument("--budget-ore", type=float, default=0.0,
                    help="se > 0: dopo --passi-misura passi MISURATI fissa il numero di passi perche' lr (coseno) "
                         "arrivi a zero entro il budget (conta dall'avvio, init e cache compresi). --passi = massimo")
    ap.add_argument("--passi-misura", type=int, default=3)
    ap.add_argument("--passi-min", type=int, default=20)
    ap.add_argument("--recompile-limit", type=int, default=64,
                    help="torch._dynamo recompile_limit: nel modo segmenti il GDN e' chiamato PROMPT PER PROMPT, quindi "
                         "_prep (torch.compile dynamic=False) vede un N = ceil(len/64) diverso per ogni lunghezza di "
                         "prompt (R012: limite 8 superato). Con prompt <= ~700 token gli N distinti sono ~12")
    ap.add_argument("--salva-ogni-passo", choices=("si", "no"), default="si",
                    help="residui/latenti 'ultimo' riscritti a ogni passo (scrittura atomica)")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = argomenti(argv)
    t_main0 = time.time()
    import torch._dynamo
    torch._dynamo.config.recompile_limit = max(int(torch._dynamo.config.recompile_limit), a.recompile_limit)
    if hasattr(torch._dynamo.config, "accumulated_recompile_limit"):
        torch._dynamo.config.accumulated_recompile_limit = max(
            int(torch._dynamo.config.accumulated_recompile_limit), 4 * a.recompile_limit)
    import e2e_studente as S
    import e2e_allena as A
    out = Path(a.uscita)
    out.mkdir(parents=True, exist_ok=True)
    flog = open(out / "registro.log", "a")

    def log(*m):
        r = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] " + " ".join(str(x) for x in m)
        print(r, flush=True)
        flog.write(r + "\n"); flog.flush()

    random.seed(a.seme); np.random.seed(a.seme); torch.manual_seed(a.seme)
    dev = torch.device(a.device)
    lids = [int(x) for x in a.letter_ids.split(",")]
    usa_rexp, usa_res = a.modo in ("esperti", "entrambi"), a.modo in ("residui", "entrambi")
    strati = intervallo(a.strati)
    cfg = RX.ConfigRexp(n_r=a.n_r, k_r=a.k_r, F=a.F, piani_esperti=a.piani_esperti, piani_router=a.piani_router,
                        condiviso=not a.senza_condiviso, strati=strati)
    cfg_r = RT.ConfigResidui(rango=a.residui_rango, piani=a.residui_piani, strati=intervallo(a.residui_strati),
                             famiglie=RT.famiglie_da(a.residui_famiglie), seme=a.seme)
    fonti_sel = tuple(f.strip().lower() for f in a.fonti_selezione.split(",") if f.strip())
    json.dump({"argv": sys.argv if argv is None else argv, "modo": a.modo, "cfg": cfg.__dict__ if usa_rexp else None,
               "cfg_residui": cfg_r.__dict__ if usa_res else None, "fonti_selezione": fonti_sel,
               "codice_recupero": str(REC)}, open(out / "config.json", "w"), indent=1, default=list)

    enc = codificatore(a.tokenizer, lids)
    G = carica_gruppi(a.dati, enc, lids)
    rng = random.Random(a.seme)
    train = G["train"]
    rng.shuffle(train)
    held = gruppi_selezione(G["heldout"], fonti_sel, a.heldout_gruppi, a.seme)
    log(f"dati: {len(train)} gruppi train, {len(held)} held-out di selezione {fonti_sel} (x4 rotazioni), "
        f"scartati {G['scartati']}; token medi/prompt {np.mean([len(r['ids']) for g in train[:500] for r in g]):.1f}")
    if not train or not held:
        raise SystemExit(f"servono gruppi train e held-out {fonti_sel} con teacher_logprobs")

    A.scegli_gdn(log)
    st = S.Studente(gguf=Path(a.gguf), strati=range(40), ricetta=a.ricetta,
                    motore_nome=os.environ.get("E2E_MOTORE", "triton"), device=a.device, log=log, densi_ternari="tutte")
    for p in st.parametri():
        p.requires_grad_(False)
    W4 = st.testa(torch.float32)[lids].contiguous()          # righe lettera della testa vera (somma dei piani)

    def perdita(z, b):
        return perdita_legame(z, b, w_kd=a.w_kd, lam=a.lam, w_pif=a.peso_pif, T=a.T, target=a.target,
                              pif_negativi_soli=a.pif_negativi_soli)

    cache = [None]                                   # CachePrefisso dopo il cancello; None = avanti pieno

    def avanti_lot(lot, pieno: bool = False) -> dict:
        if cache[0] is not None and not pieno:
            return cache[0].avanti(lot)
        return st.avanti(lot["ids"], inizi=lot["inizi"])

    def logits4(lot, pieno: bool = False) -> torch.Tensor:
        with torch.no_grad():
            return st.esce(avanti_lot(lot, pieno)["xL"])[lot["ultimi"]].float() @ W4.T

    def valuta(gruppi_l: list) -> dict:
        zs, rows = [], []
        for gl in lotti(gruppi_l, a.token_valuta):
            lot = impacchetta(gl, dev)
            zs.append(logits4(lot).float()); rows += lot["rows"]
        z = torch.cat(zs)
        b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in collate_targets(rows).items()}
        _, s = perdita(z, b)
        s["n"] = len(rows)
        s["per_fonte"] = metriche_selezione(z, b, [str(r.get("source", "")).lower() for r in rows])
        return s

    def chiave(v: dict) -> tuple:                     # piu' piccolo = migliore
        if a.criterio == "kd":                        # solo l'accordo col maestro (KD + coerenza), senza l'oro
            return (v["kd"] * a.w_kd + a.lam * v["rot"],)
        return (v["loss"],) if a.criterio == "loss" else (-v["acc"], v["loss"])

    # ── 0 cancello d'identita' (gamma = 0): logit lettera con TUTTO installato == senza, AL BIT
    lot0 = impacchetta(held[:2], dev)
    z0 = logits4(lot0).clone()
    dt_g = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[a.gemm]
    mods = RX.installa(st, cfg, device=dev, dtype_gemm=dt_g) if usa_rexp else {}
    res = RT.installa(st, cfg_r, device=dev) if usa_res else {}
    for m in res.values():
        m.dtype_gemm = dt_g
    z1 = logits4(lot0)
    ident = bool(torch.equal(z0, z1))
    log(f"cancello identita' (gamma = 0; esperti {len(mods)} strati, residui {len(res)} proiezioni in "
        f"{len(RT.strati_di(res)) if res else 0} strati, rango {a.residui_rango}): {'AL BIT' if ident else 'DIVERSO'}")
    if not ident:
        log("ERRORE: la partenza non e' R2B3m al bit"); return 2
    st.strati_allenati = sorted(set(mods) | set(RT.strati_di(res) if res else ()))
    for m in list(mods.values()) + list(res.values()):
        m.requires_grad_(True)
    t_main = t_main0
    if a.cache_prefisso == "si":
        try:
            L0 = V.primo_strato_allenato(st)
        except ValueError as e:                     # si allena dal primo strato: niente prefisso congelato
            log(f"cache prefisso spenta: {e}")
            a.cache_prefisso = "no"
    if a.cache_prefisso == "si":
        cache[0] = V.CachePrefisso(st, L0)
        meta_cache = {"gguf": str(a.gguf), "L0": L0, "seme": a.seme}
        if a.cache_file and Path(a.cache_file).exists():
            log(f"cache prefisso: riprese {cache[0].carica(a.cache_file, meta_cache)} righe da {a.cache_file}")
        rows_h = [r for g in held for r in g]
        cache[0].riempi(rows_h + lot0["rows"], a.token_cache, dev, log)
        z2 = logits4(lot0)
        scarto = float((z2 - z1).abs().max())
        log(f"cancello cache: prefisso < strato {L0}; scarto massimo dei logit lettera, cache vs avanti pieno {scarto:.3e}")
        if not torch.allclose(z2, z1, rtol=a.tol_cache, atol=a.tol_cache):
            log("ERRORE: la cache del prefisso non riproduce l'avanti pieno"); return 2

    def righe_passo(k: int) -> list:
        i0 = ((k - 1) * a.gruppi_passo) % max(1, len(train) - a.gruppi_passo + 1)
        return [r for g in train[i0:i0 + a.gruppi_passo] for r in g]

    def assicura_cache(k: int, ultimo: int) -> None:
        """Riempie in anticipo i passi k.. (tanti quanti stanno in un lotto di cache): il costo e' dei pesi, non dei token."""
        if cache[0] is None or not cache[0].mancanti(righe_passo(k)):
            return
        per_passo = max(1, sum(len(r["ids"]) for r in righe_passo(k)))
        avanti_k = max(1, a.token_cache // per_passo)
        rows = [r for kk in range(k, min(ultimo, k + avanti_k - 1) + 1) for r in righe_passo(kk)]
        cache[0].riempi(rows, a.token_cache, dev, log)

    def salva(nome: str, meta: dict) -> None:
        if mods:
            RX.salva_latenti(mods, out / f"latenti_{nome}.pt", meta)
        if res:
            RT.salva_latenti(res, out / f"residui_{nome}.pt", meta)

    def salva_atomico(nome: str, meta: dict) -> None:
        """Come salva, ma scrive su <nome>_tmp e rinomina: un'interruzione non lascia un file a meta'."""
        salva(nome + "_tmp", meta)
        for pre in (("latenti_",) if mods else ()) + (("residui_",) if res else ()):
            os.replace(out / f"{pre}{nome}_tmp.pt", out / f"{pre}{nome}.pt")

    def carica(nome: str) -> None:
        if mods:
            RX.carica_latenti(mods, out / f"latenti_{nome}.pt")
        if res:
            RT.carica_latenti(res, out / f"residui_{nome}.pt")

    def passo(gruppi_p: list, peso_aux: float) -> dict:
        """Gradiente della perdita media sui gruppi del passo (micro-lotti impacchettati, accumulo)."""
        ls = lotti(gruppi_p, a.token_lotto)
        n_tot = sum(len(g) for g in gruppi_p) * 4
        for m in mods.values():
            m.peso_aux = peso_aux / len(ls)
        agg = {"loss": 0.0, "kd": 0.0, "rot": 0.0, "pif": 0.0, "acc": 0.0}
        for gl in ls:
            lot = impacchetta(gl, dev)
            stato = avanti_lot(lot)
            xL = stato["xL"].detach().requires_grad_(True)
            with torch.enable_grad():
                z = st.esce(xL)[lot["ultimi"]].float() @ W4.T
                loss, s = perdita(z, lot["batch"])
                w = len(lot["rows"]) / n_tot          # ogni lotto ha gruppi interi -> media pesata esatta
                (loss * w).backward()
            st.indietro(stato, xL.grad)
            for k in agg:
                agg[k] += s[k] * w
            del stato, xL
        for m in mods.values():
            m.peso_aux = 0.0
        return agg

    # ── 1 inizializzazione allineata al gradiente (esperti: come e2e_recupero; residui: SVD di sum g x^T)
    t0 = time.time()
    if a.init == "gradiente":
        for m in mods.values():
            m.sonda = {"x": [], "g": [], "cpu": a.device != "cpu"}
        for m in res.values():
            m.sonda = {"cpu": a.device != "cpu"}
        if cache[0] is not None:
            cache[0].riempi([r for g in train[: a.gruppi_init] for r in g], a.token_cache, dev, log)
        passo(train[: a.gruppi_init], 0.0)
        diag = {"esperti": {}, "residui": {}}
        for L, m in mods.items():
            X = torch.cat(m.sonda["x"]).to(dev); Gd = torch.cat(m.sonda["g"]).to(dev)
            m.sonda = None
            diag["esperti"][L] = RX.inizializza_da_gradiente(m, X, Gd, seme=a.seme + L)
            diag["esperti"][L]["energia_gradiente"] = float(Gd.pow(2).sum(-1).mean())
            del X, Gd
        for n, m in res.items():
            stat, m.sonda = m.sonda, None
            if not stat.get("n"):
                raise RuntimeError(f"{n}: la sonda non ha raccolto gradienti (strato fuori dall'indietro?)")
            diag["residui"][n] = RT.inizializza_da_gradiente(m, stat)
            del stat
        for m in list(mods.values()) + list(res.values()):
            m.zero_grad(set_to_none=True)
        json.dump(diag, open(out / "init_diag.json", "w"))
        log(f"init dal gradiente: {a.gruppi_init} gruppi, {time.time() - t0:.0f} s")
        train = train[a.gruppi_init:]
    t_ev = time.time()
    v0 = valuta(held)
    s_eval = time.time() - t_ev
    log(f"held-out {fonti_sel} passo 0 (R2B3m): {json.dumps(v0)}")
    log(f"dynamo: recompile_limit {torch._dynamo.config.recompile_limit}")
    salva("init", {"passo": 0, "held": v0})

    gruppi = []
    if mods:
        gruppi += [{"params": [m.R for m in mods.values()], "lr": a.lr_r},
                   {"params": [p for m in mods.values() for p in (m.gamma,) + ((m.gamma_s,) if cfg.condiviso else ())],
                    "lr": a.lr_gamma},
                   {"params": [p for m in mods.values() for p in (m.G, m.U, m.D) + ((m.Gs, m.Us, m.Ds)
                                                                                     if cfg.condiviso else ())],
                    "lr": a.lr_lat}]
    if res:
        gruppi += [{"params": [p for n in sorted(res) for p in (res[n].A, res[n].B)], "lr": a.lr_res},
                   {"params": [res[n].gamma for n in sorted(res)], "lr": a.lr_res_gamma}]
    base = [g["lr"] for g in gruppi]
    tutti = RX.parametri(mods) + RT.parametri(res) if res else RX.parametri(mods)
    opt = torch.optim.AdamW(gruppi, betas=(0.9, 0.99), weight_decay=0.0)
    migl = (chiave(v0), 0, v0)
    storia = [{"passo": 0, "held": v0}]
    n_risc = max(1, int(a.riscaldamento * a.passi))
    t_avvio = time.time()
    passi_tot, tempi_passo, k = a.passi, [], 0
    while k < passi_tot:
        k += 1
        if time.time() - t_avvio > a.ore_max * 3600:
            log(f"tetto di {a.ore_max} h al passo {k - 1}: chiudo")
            break
        f = min(1.0, k / n_risc) * (0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * max(0, k - n_risc) /
                                                                     max(1, passi_tot - n_risc))))
        for g, b in zip(opt.param_groups, base):
            g["lr"] = b * f
        i0 = ((k - 1) * a.gruppi_passo) % max(1, len(train) - a.gruppi_passo + 1)
        gp = train[i0:i0 + a.gruppi_passo]
        assicura_cache(k, passi_tot)                   # fuori dal cronometro del passo: costo dei pesi, una volta
        t1 = time.time()
        opt.zero_grad(set_to_none=True)
        s = passo(gp, a.peso_aux)
        gn = torch.nn.utils.clip_grad_norm_(tutti, a.clip)
        opt.step()
        riga = {"passo": k, **{x: round(y, 6) for x, y in s.items()}, "norma_grad": float(gn), "lr_f": round(f, 4),
                "s": round(time.time() - t1, 2)}
        if mods:
            riga["gamma_medio"] = float(np.mean([m.gamma.detach().abs().mean().item() for m in mods.values()]))
        if res:
            riga["gamma_res_medio"] = float(np.mean([m.gamma.detach().abs().mean().item() for m in res.values()]))
        tempi_passo.append(time.time() - t1)
        if a.budget_ore > 0 and k == a.passi_misura:
            s_passo = V.mediana(tempi_passo[1:] or tempi_passo)      # il primo compila i kernel
            nuovo = V.passi_da_budget(a.budget_ore * 3600, time.time() - t_main, s_passo, s_eval, a.heldout_ogni,
                                      k, a.passi, a.passi_min)
            sta = (nuovo - k) * (s_passo + s_eval / a.heldout_ogni) + s_eval
            log(f"budget {a.budget_ore} h: {s_passo:.1f} s/passo, held-out {s_eval:.1f} s ogni {a.heldout_ogni} -> "
                f"passi {a.passi} -> {nuovo} (restano ~{sta / 60:.0f} min)" +
                ("  ATTENZIONE: budget insufficiente anche per il minimo" if nuovo * s_passo > a.budget_ore * 3600 else ""))
            passi_tot = nuovo
        if k % a.heldout_ogni == 0 or k == passi_tot:
            v = valuta(held)
            riga["held"] = v
            if chiave(v) < migl[0]:
                migl = (chiave(v), k, v)
                salva("migliore", {"passo": k, "held": v})
        storia.append(riga)
        log(json.dumps(riga))
        if a.salva_ogni_passo == "si":
            salva_atomico("ultimo", {"passo": k, "riga": json.loads(json.dumps(riga, default=float))})
    if migl[1] == 0:
        carica("init")
        salva("migliore", {"passo": 0, "held": v0})
    carica("migliore")
    log(f"FINE: migliore passo {migl[1]} ({a.criterio}) held loss {migl[2]['loss']:.5f} (passo 0 {v0['loss']:.5f}); "
        f"acc {v0['acc']:.4f} -> {migl[2]['acc']:.4f}")
    json.dump({"storia": storia, "passi_eseguiti": len(storia) - 1, "passi_programmati": passi_tot,
               "secondi_totali": time.time() - t_main, "cache_prefisso_s": cache[0].secondi if cache[0] else None},
              open(out / "storia.json", "w"), indent=1)
    if cache[0] is not None and a.cache_file:
        cache[0].salva(a.cache_file, {"gguf": str(a.gguf), "L0": cache[0].L0, "seme": a.seme})

    # ── esportazione: esperti (identica a e2e_recupero) e/o residui (tensori TQ1_0 in piu', gamma fuso nelle scale di B)
    esp = {"modo": a.modo, "passo": migl[1], "held": migl[2], "held0": v0}
    if mods:
        t = RX.tensori_gguf(mods)
        np.savez(out / "rexp_tq1.npz", **t)
        esp.update({"k_r": a.k_r, "n_r": a.n_r, "condiviso": cfg.condiviso, "strati": list(strati),
                    "byte": int(sum(v.nbytes for v in t.values())), "n_tensori": len(t)})
        log(f"esportati {len(t)} tensori esperti, {sum(v.nbytes for v in t.values()) / 2**20:.1f} MiB -> rexp_tq1.npz")
    if res:
        tr, info = RT.tensori_gguf(res)
        np.savez(out / "residui_tq1.npz", **tr)
        json.dump({"tensori": info, "rango": a.residui_rango, "piani": a.residui_piani, "rp": RT.RP,
                   "famiglie": list(cfg_r.famiglie), "strati": RT.strati_di(res), "passo": migl[1],
                   "uso_motore": "cur = mul_mat(W,x)[+piani] + mul_mat(<fam>_lrb, mul_mat(<fam>_lra, x))"},
                  open(out / "residui.json", "w"), indent=1)
        esp.update({"residui_n_tensori": len(tr), "residui_byte": int(sum(v.nbytes for v in tr.values()))})
        log(f"esportati {len(tr)} tensori residui, {sum(v.nbytes for v in tr.values()) / 2**20:.2f} MiB -> "
            f"residui_tq1.npz (+ residui.json; GGUF: e1_scrivi_residui_gguf.py)")
    json.dump(esp, open(out / "esportazione.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
