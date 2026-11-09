#!/usr/bin/env python3
"""Obiettivo C, stadio 2 — costruttori (e) e (f) sui 250 densi a UN piano (12/10/2026), SOLO CPU.

Stesso bersaglio (bf16 puro), stessa H (manifesto R2B3, N = 196.608) e stesso formato di jarq_fedele.py (a/b/c):
uscita <out>/<variante>/tensori/<primo piano>.npy + manifesto.json, base C0B per patch_gguf. Si confrontano con
(c) f_bil1jarq (63,90 sul Dell), di cui cambiano UNA cosa sola:

  e   f_bil1schurjarq  GPTQ bilanciato K=1 (come (c)) -> SchurOpt (arXiv 2608.15567, Alg. 1; solutori.schuropt con
                       i parametri gia' usati da applica.py: passi = 4, lam = 0, unico piano libero) -> JARQ Alg. 1
                       fedele (jarq_fedele.jarq_fedele, K = 3, nu = 0,6, tau = 1e-5).
                       Ordine: solutore -> raffinamento -> JARQ. JARQ (2609.38599) e' definita come passo subito dopo la
                       PTQ, a partire da qualunque inizializzazione: SchurOpt e' parte della PTQ (raffina i codici del
                       solutore con il suffisso continuo), quindi viene prima; JARQ chiude la catena come in (b)/(c).
  f   f_bil1foemjarq   GPTQ bilanciato K=1 con FOEM (arXiv 2507.11017) nella forma del LAVORO, Alg. 1 / Eq. 19
                       (solutori.gptq_bilanciato, foem_forma="lavoro": invariante alla scala di H, §3.3), beta = 0,1
                       come nel lavoro -> JARQ Alg. 1 fedele.  Variante Thor DICHIARATA: il lavoro non prevede JARQ
                       dopo FOEM; qui la si aggiunge perche' e' la sequenza scelta per lo stadio 2 (c) e isola FOEM.
  f0  f_bil1foem       FOEM fedele e basta (forma del lavoro, beta 0,1, NIENTE JARQ dopo, come nel codice ufficiale):
                       JARQ accetta solo cio' che abbassa l'errore del SECONDO ordine e puo' cancellare la correzione del
                       primo ordine che FOEM introduce (nota gia' in applica.py, leva bil_foem). Serve a decidere se
                       l'eventuale differenza f - c viene da FOEM o se JARQ la annulla.
  Differenze dichiarate rispetto ai lavori (eredita' di jarq_fedele, R1/R2): ingressi X del bf16 invece di X~ propagati;
  gruppi del formato (128 ruotati / 256), scala fp16, niente act-order; FOEM sulla griglia bilanciata a 3 livelli
  (K = 1: {-D, 0, +D}, D scelto per gruppo come in (c)) invece del quantizzatore uniforme del lavoro.
  schuropt riporta all'ingresso le righe che peggiorerebbero (solutori.py, come in applica.py): dettaglio Thor ereditato.
Prova del 12/10 su 2 tensori (blk.0.attn_gate, blk.9.ffn_gate_shexp): bil1 = avvio di (c) al bit (0,1056 / 0,1539);
  errH e 0,0993 / 0,1451 contro (c) 0,0998 / 0,1455; f0 FOEM 0,2092 / 0,2391 e f 0,1074 / 0,1506: FOEM corregge un
  termine del PRIMO ordine (beta (W~ - W0)) che l'errore pesato del secondo ordine non vede, quindi errH non decide; decide
  l'MMLU appaiato sul Dell (lezione S1: l'errH non ha predetto la KL su C).
uso: costruttori_ef.py [--out DIR] [--solo nome,...] [--varianti f_bil1schurjarq,f_bil1foemjarq,f_bil1foem]
     HIP_VISIBLE_DEVICES= LEVE_THREADS=6 nice -n 10 ...
"""
import argparse, hashlib, json, os, sys, time
sys.path.insert(0, "@@ROOT:models@@/build/leve/raffina")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import comune as CM
import solutori as SO
import jarq_fedele as JF

FOEM_BETA = 0.1              # 2507.11017, beta del lavoro (forma invariante alla scala di H)
SCHUR_PASSI = 4              # come applica.py (schur_pk)
TUTTE = ("f_bil1schurjarq", "f_bil1foemjarq", "f_bil1foem")


def una(d, varianti):
    W, H, c, _ = JF.bersaglio_bf16_e_H(d.nome, d.rotated)
    ng = d.S.shape[-1]
    t0 = time.time()
    out, m = {}, {}
    Tc0, Sc0, _ = SO.gptq_bilanciato(W, H, 1, ng)                       # = avvio di (c), deterministico
    m["errH_bil1"] = JF.errH(W, Tc0, Sc0, H)
    if "f_bil1schurjarq" in varianti:
        Ts, Ss, cs = SO.schuropt(W, H, Tc0.clone(), Sc0.clone().to(SO.F64), [0], ng, passi=SCHUR_PASSI, lam=0.0)
        Ts, Ss = Ts[:1].to(torch.int8), SO.fp16(Ss[:1].to(SO.F64))
        m["errH_bil1schur"] = JF.errH(W, Ts, Ss, H)
        Te, Se, ce = JF.jarq_fedele(W, H, Ts, Ss, c)
        out["f_bil1schurjarq"] = (Te, Se)
        m["errH_bil1schurjarq"] = JF.errH(W, Te, Se, H)
        m["trit_cambiati_schur_da_bil"] = float((Ts != Tc0).double().mean())
        m["catena_e"] = ce
    if "f_bil1foemjarq" in varianti or "f_bil1foem" in varianti:
        Tf0, Sf0, cf = SO.gptq_bilanciato(W, H, 1, ng, foem_beta=FOEM_BETA, foem_forma="lavoro")
        m["errH_bil1foem"] = JF.errH(W, Tf0, Sf0, H)
        m["trit_cambiati_foem_da_bil"] = float((Tf0 != Tc0).double().mean())
        if "f_bil1foem" in varianti:
            out["f_bil1foem"] = (Tf0, SO.fp16(Sf0))
        if "f_bil1foemjarq" in varianti:
            Tf, Sf, cfj = JF.jarq_fedele(W, H, Tf0, Sf0, c)
            out["f_bil1foemjarq"] = (Tf, Sf)
            m["errH_bil1foemjarq"] = JF.errH(W, Tf, Sf, H)
            m["catena_f"] = cfj
    for v, (T, S) in out.items():
        assert T.dtype == torch.int8 and int(T.abs().max()) <= 1 and T.shape[0] == 1, f"{d.nome} {v}: codici fuori da {{-1,0,1}}"
        assert torch.isfinite(S).all() and torch.equal(S, SO.fp16(S)), f"{d.nome} {v}: scale non fp16 o non finite"
    m["secondi"] = round(time.time() - t0, 1)
    return out, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="@@ROOT:models@@/gguf/tony-tern/OBIETTIVO_C/varianti_fedeli")
    ap.add_argument("--solo", default="")
    ap.add_argument("--varianti", default=",".join(TUTTE))
    ap.add_argument("--registro", default="registro_ef.jsonl")
    a = ap.parse_args()
    varianti = [v for v in a.varianti.split(",") if v]
    assert all(v in TUTTE for v in varianti), varianti
    reg = os.path.join(a.out, a.registro)
    fatti = {json.loads(l)["nome"] for l in open(reg)} if os.path.exists(reg) else set()
    rif_c = {}
    pc = os.path.join(a.out, "registro.jsonl")                          # errH di (c) dalla corsa dell'11/10
    if os.path.exists(pc):
        for l in open(pc):
            r = json.loads(l); rif_c[r["nome"]] = r.get("errH_bil1jarq")
    src = CM.gp.GGUFPlaneSource(CM.R2B3M, layers=list(range(40)))
    nomi = CM.elenco_densi(src)
    if a.solo:
        nomi = [n for n in nomi if n in a.solo.split(",")]
    CM.log(f"costruttori_ef: {len(nomi)} densi, gia' fatti {len(fatti)}, varianti {varianti}, threads {torch.get_num_threads()}")
    mans = {}
    for v in varianti:
        os.makedirs(os.path.join(a.out, v, "tensori"), exist_ok=True)
        mp = os.path.join(a.out, v, "tensori", "manifesto.json")
        mans[v] = json.load(open(mp)) if os.path.exists(mp) else {}
    for i, nome in enumerate(nomi):
        if nome in fatti:
            continue
        d = CM.Denso(src, nome)
        out, m = una(d, varianti)
        p1 = d.nomi_piani[0]
        for v, (T, S) in out.items():
            b = d.byte_piani(T, S.float())[0]
            td = os.path.join(a.out, v, "tensori")
            np.save(os.path.join(td, p1 + ".npy"), b)
            mans[v][p1] = {"tipo": CM.gp._import_gguf(None).GGMLQuantizationType(d.tid).name,
                           "forma_ggml": [int(d.in_), int(d.out)],
                           "sha256_npy": hashlib.sha256(np.ascontiguousarray(b).tobytes()).hexdigest(),
                           "variante": v, "errH": m["errH_" + v[2:]]}
            mp = os.path.join(td, "manifesto.json")
            json.dump(mans[v], open(mp + ".tmp", "w")); os.replace(mp + ".tmp", mp)
        m["nome"] = nome
        m["errH_bil1jarq_c"] = rif_c.get(nome)
        with open(reg, "a") as f:
            f.write(json.dumps(m) + "\n")
        g = lambda k: f"{m[k]:.4f}" if m.get(k) is not None else "-"
        CM.log(f"{i+1}/{len(nomi)} {nome}: errH bil1 {g('errH_bil1')} (c) {g('errH_bil1jarq_c')} | schur {g('errH_bil1schur')}"
               f" -> +jarq {g('errH_bil1schurjarq')} | foem {g('errH_bil1foem')} -> +jarq {g('errH_bil1foemjarq')}, {m['secondi']} s")
    CM.log("fine costruttori_ef")


if __name__ == "__main__":
    main()
