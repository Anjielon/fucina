#!/usr/bin/env python3
"""S2+S3 — costruzione dei candidati e GUARDIA, per un denso (S2) o per un lotto di esperti (S3).

Statistiche (dal passaggio sequenziale, flusso di C coi tensori a monte GIA' nuovi), meta' A costruisce, meta' B guarda:
  Hq = E[x_q x_q^T] (flusso di C), Mx = E[x_f x_q^T] (x_f = flusso del maestro bf16 sugli stessi token).
DENSI (S2), per ogni tensore, candidati:
  QEP_a   bersaglio asimmetrico QEP (2504.09629): W_a = W + a (W Mx H^-1 - W), a in S23_ALFE (default {0,5; 1}: «a tarato in
          {0,5; 1}» del piano; qui scelto PER TENSORE dalla guardia sulla meta' B — variante Thor dichiarata, invece di due
          strati e della KL4 a blocchi); H = H_manifesto + w H_conoscenza (K6, w = S23_W_CONOSCENZA, default 1), con
          H_manifesto = sorgente M (calib_auto2, quello di ogni H di C) e H_conoscenza = Kv (wiki+edu) + Kw (WikiText-103),
          ciascuna normalizzata per token; stesso miscuglio per Mx. Solutore = ricetta di C: BIL K=1 (GPTQ bilanciato) +
          JARQ fedele (K=3, nu 0,6) + DRQ con insieme di CONTROLLO = H della meta' B; (alpha, beta) di DRQ
          TARATI PER FAMIGLIA sulla meta' B dalla griglia di 2610.11226 App. C (13/10: vedi «DRQ per famiglia» sotto).
  K1      (ssm_out, attn_output, attn_v, ffn_down_shexp) obiettivo a DUE LATI tr(B E A' E^T) con A' = 1/2 (A_KL + H) (AMIX,
          FRONTIERA §3.2; YAQA 2505.22988 / FastKron 2608.06291): partenza dal miglior QEP_a, 2 giri di codici FastKron a scale
          date <-> scale a due lati (CG) = maestro_kron2 (variante Thor dichiarata); A_KL, B = fattori di Kronecker della KL
          di C calcolati da S1 (ospite di S1, NON ricalcolati sul flusso sequenziale: approssimazione dichiarata).
  K2      (ffn_down_shexp -> righe di ffn_up_shexp; attn_output -> righe di attn_v, legame GQA) colonna c piegata a monte,
          ALTERNATA coi codici (SCALE §1.1: c in forma chiusa ARB-LLM_RC 2410.03129 pesata H e asimmetrica; poi codici e scale
          rifatti con BIL K=1 + JARQ fedele sul bersaglio W_a diag(1/c) con H' = D H D), 2 giri.
  GUARDIA: si sceglie il candidato con l'obiettivo piu' basso sulla meta' B (asimmetrico a un lato; per le famiglie K1, se i
          fattori ci sono, a due lati coi fattori di CONTROLLO disgiunti di S1 e A'_B = 1/2 (A_ctl + H_B)) e lo si accetta
          SOLO se batte la partenza (piano §2 S2.5: «altrimenti resta quello dell'ospite»).
ESPERTI (S3, E1 del piano / linea ESPERTI §4), a lotti sul dispositivo:
  H_e(k) = (n_e Gq_e/peso_e + k H_strato)/(n_e + k) (MoEQuant AGQ 2505.03804 + restringimento k = S23_K, F1 di T2), termine
  incrociato ristretto allo stesso modo; bersaglio asimmetrico QEP a = S23_ALFA_ESPERTI (default 0,5 = valore del lavoro);
  GPTQ bilanciato K=1 + JARQ a lotti (cardan_piega.jarq_lotto, quello della piega CARDAN, nu2 1e-4); down sull'ingresso VERO
  del down di C ricatturato DOPO gate/up nuovi, col termine incrociato a_f (maestro) a_q^T. Guardia per esperto e per
  tensore sulla meta' B («mai peggio dell'ospite»).

CORRETTO 14/10/2026 (ricerche/C_S1B_AUDIT_1014.md -> ricerche/C_S23_CORRETTO_1014.md). Copia del codice congelato di S23
(codice/, SHA_CONGELATO invariato) con tre differenze di sostanza:
  G1  GUARDIA INDIPENDENTE dall'obiettivo che si ottimizza. Prima la guardia misurava lo stesso obiettivo ASIMMETRICO
      E||W x_f - Q x_q||^2 (bersaglio QEP a=1) sulla meta' B: con codici fermi o quasi, il suo minimo e' W attenuato del
      fattore di correlazione maestro/studente rho (errori nelle variabili), proprieta' della popolazione -> passa anche
      sulla meta' disgiunta e sceglie sempre a=1 (11/11 densi «scelto qep1» nella passata del 14/10). Ora ogni scelta
      (candidato del denso, base per K1/K2, combinazione DRQ in taratura, alfa e accettazione per esperto) si giudica
      sull'errore d'uscita SIMMETRICO verso il maestro sulla meta' B, ||(W - Q) x_q||^2 = tr((W-Q) H_B (W-Q)^T)
      (S23_GUARDIA=sim, default; S23_GUARDIA=kl_locale: per le famiglie K1 coi fattori, metrica a due lati
      tr(B_ctl E A'_B E^T) con E = Q - W, cioe' KL locale sui fattori di CONTROLLO disgiunti), PIU' un controllo d'energia
      d'uscita: r = ||Q x_q|| / ||W x_q|| (= sqrt(tr(Q H_B Q^T)/tr(W H_B W^T))) fuori da [S23_ENERGIA_MIN, S23_ENERGIA_MAX]
      (default [0,9; 1,1]) -> candidato rifiutato. La partenza non si giudica sull'energia (e' il ripiego).
  G2  candidati a in S23_ALFE (default {0; 0,5; 1}, densi) e S23_ALFE_ESPERTI (default {0; 0,5}, esperti): QEP fedele
      (2504.09629: alpha e' un iperparametro < 1 e i pesi si riquantizzano) con a=0 = GPTQ simmetrico; tutti giudicati da G1.
      K2: bersaglio della colonna Clin = W((1-a) H_A + a Mx_A) col suo a (era W Mx_A, cioe' a = 1 sempre).
  G3  H CORTA (variante Thor dichiarata, misura B di ricerche/C_TARATURA_CORTE_1013.md §4-5): nel miscuglio K6 la
      sorgente Kw (WikiText-103, finestre da 256 token, stato fresco) pesa S23_PESO_KW volte i suoi token dentro la
      conoscenza: H_K = (S_Kv + p S_Kw)/(n_Kv + p n_Kw). p = 1 = miscuglio di prima al bit. Stesso miscuglio per Mx, per la
      H_B della guardia e per la H di strato verso cui si restringono gli esperti (Gq_e e il down restano a token).
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

QUI = Path(__file__).resolve().parent
for p in (str(QUI), str(QUI.parent / "lib"), str(QUI.parent / "s1b"), str(QUI.parent / "rif")):
    if p not in sys.path:
        sys.path.insert(0, p)
import s23_algebra as AL  # noqa: E402
import s1b_comune as K  # noqa: E402   (ruota/deruota/ruota_H, formati: copia congelata di S1b)

F64 = torch.float64
FAM_K1_DEFAULT = ("ssm_out", "attn_output", "attn_v", "ffn_down_shexp")
COPPIE_K2 = {"ffn_down_shexp": "ffn_up_shexp", "attn_output": "attn_v"}
T_TQ1_0R_1024 = 45


def mem(X, rot):
    return K.ruota(X) if rot else X


def memH(H, rot):
    return K.ruota_H(H) if rot else H


def normalizza(H):
    """come comune.bersaglio_e_H: simmetrica, diagonali morte -> 1, media della diagonale 1. Ritorna (Hn, dbar)."""
    H = 0.5 * (H + H.transpose(-1, -2))
    d = torch.diagonal(H, dim1=-2, dim2=-1)
    morti = d <= 0
    if bool(morti.any()):
        H = H.clone()
        if H.dim() == 2:
            H[morti, morti] = 1.0
        else:
            for b in range(H.shape[0]):
                m = morti[b]
                if m.any():
                    H[b, m, m] = 1.0
    dbar = torch.diagonal(H, dim1=-2, dim2=-1).clamp_min(1e-30).mean(-1)
    return H / (dbar.view(*dbar.shape, 1, 1) if H.dim() == 3 else dbar), dbar


def mescola(acc: Dict[str, tuple], w: float, peso_kw: float = 1.0):
    """acc {sorgente: (H_somma, Mx_somma, n)} -> (H, Mx, N) normalizzate per token, H = (H_M + w H_K)/(1+w) (K6).
    14/10 G3 (H corta): dentro H_K i token di Kw contano peso_kw volte (peso_kw = 1: identico al bit a prima)."""
    nM = acc.get("M", (None, None, 0))[2]
    Hk = Mk = None
    nK = 0
    nKe = 0.0
    for s in ("Kv", "Kw"):
        if s in acc and acc[s][2]:
            c = peso_kw if s == "Kw" else 1.0
            hs = acc[s][0] if c == 1.0 else c * acc[s][0]
            ms = acc[s][1] if c == 1.0 else c * acc[s][1]
            Hk = hs if Hk is None else Hk + hs
            Mk = ms if Mk is None else Mk + ms
            nK += acc[s][2]
            nKe += c * acc[s][2]
    parti = []
    if nM:
        parti.append((1.0, acc["M"][0] / nM, acc["M"][1] / nM))
    if nK:
        den_k = nK if nKe == nK else nKe
        parti.append((w, Hk / den_k, Mk / den_k))
    assert parti, "nessuna sorgente"
    pt = sum(p[0] for p in parti)
    H = sum(p[0] * p[1] for p in parti) / pt
    Mx = sum(p[0] * p[2] for p in parti) / pt
    return H, Mx, nM + nK


def J_asim(Qm, Hm, Cm) -> float:
    """tr(Q H Q^T) - 2 tr(Q C^T) (= E||W x_f - Q x_q||^2 a meno della costante), base memorizzata."""
    return float(((Qm @ Hm) * Qm).sum() - 2.0 * (Qm * Cm).sum())


# ───────────── 14/10 G1: guardia simmetrica + energia (indipendente dall'obiettivo ottimizzato) ─────────────
def opz_guardia(opz: dict):
    """(energia_min, energia_max, modo). Default [0,9; 1,1], modo «sim» (vedi docstring G1)."""
    return float(opz.get("energia_min", 0.9)), float(opz.get("energia_max", 1.1)), opz.get("guardia", "sim")


def errore_sim(Qm, Wm, Hm) -> float:
    """tr((Q - W) H (Q - W)^T) = E||(W - Q) x_q||^2 (base memorizzata: Wm, Qm, Hm nella stessa base)."""
    E = Qm - Wm
    return float(((E @ Hm) * E).sum())


def rapporto_energia(Qm, Hm, den_w: float) -> float:
    """||Q x_q|| / ||W x_q|| = sqrt(tr(Q H Q^T) / tr(W H W^T))."""
    num = float(((Qm @ Hm) * Qm).sum())
    return math.sqrt(max(num, 0.0) / max(den_w, 1e-300))


def energia_ok(r: float, emin: float, emax: float, r0: Optional[float] = None) -> bool:
    """r in [emin, emax]; con r0 (energia della PARTENZA, S23_ENERGIA_RELATIVA=1) basta anche non essere piu' lontani da 1
    della partenza (|ln r| <= |ln r0|): misurato 14/10, il down degli esperti a=0 sta a 0,899 con la partenza a 0,884 e
    l'errore simmetrico piu' basso — un rifiuto per soglia assoluta li' scarterebbe un miglioramento vero."""
    if not math.isfinite(r) or r <= 0:
        return False
    if emin <= r <= emax:
        return True
    return r0 is not None and math.isfinite(r0) and r0 > 0 and abs(math.log(r)) <= abs(math.log(r0))


class Kron:
    """fattori di S1 (base memorizzata, diagonale media 1) di un tensore: A, B di calibrazione e di controllo."""

    def __init__(self, cal: Optional[Path], ctl: Optional[Path], nome: str, quale_B: str, dev, dt):
        self.ok = False
        if cal is None or ctl is None:
            return
        f = {k: d / f"{t}_{nome}.npy" for k, d, t in (("Ac", cal, "A"), ("Bc", cal, quale_B), ("Ak", ctl, "A"),
                                                       ("Bk", ctl, quale_B))}
        if not all(p.exists() for p in f.values()):
            return
        for k, p in f.items():
            M = torch.from_numpy(np.load(p).astype(np.float64)).to(dev, dt)
            M, _ = normalizza(M)
            setattr(self, k, M)
        self.ok = True


# ───────────── DRQ: (alpha, beta) tarati PER FAMIGLIA (13/10, cura dell'audit AUDIT_1013/drq_jarq.md) ─────────────
# 2610.11226 App. C «Main parameters»: alpha in {0,1; 0,3; 0,5; 1}, beta in {0,005; 0,1; 0,25; 0,75; 1} (20 combinazioni),
# scelte su 64 campioni di VALIDAZIONE (Pile) per modello e metodo base. Qui, dichiarato:
#   (V1) la scelta e' PER FAMIGLIA (non per modello), sui primi S23_DRQ_STRATI_TAR (default 2) strati in cui la famiglia
#        compare, poi FISSA per gli strati dopo; criterio = obiettivo asimmetrico sulla META' B (la validazione che la
#        passata ha gia', disgiunta dalla meta' A di ricerca), somma su strati tarati del guadagno relativo contro il
#        JARQ senza DRQ, al miglior bersaglio QEP_a. Il lavoro sceglie su perplessita' del modello: qui ricostruzione per
#        strato (stessa logica della guardia, molto meno cara). La meta' B fa anche da insieme di controllo di DRQ (come
#        prima); il lavoro tiene controllo e validazione disgiunti.
#   (V2) unita': «grezza» = la griglia del lavoro sulla scala GREZZA di H (C = (1-b) H + b I, fedele); «norm» = la stessa
#        griglia con H (ricerca e controllo) divise per m = media della diagonale della H di ricerca, cioe' beta e alpha
#        relativi alla scala degli ingressi (variante Thor: l'audit ha misurato che su attn_output, ssm_out,
#        ffn_down_shexp gli ingressi sono minuscoli, C ~ b I, S0/L0 ~ 1e4 e DRQ non agisce). Le tolleranze (L e controlli
#        <= +1 %) sono rapporti: invarianti. S23_DRQ_GRIGLIA = paper+norm (default, 40) | paper (20, fedele) | spenta
#        (la coppia di prima, (1; 0,005) grezza, al bit).
GRIGLIA_ALFA = (0.1, 0.3, 0.5, 1.0)
GRIGLIA_BETA = (0.005, 0.1, 0.25, 0.75, 1.0)
COMBO_FISSA = "grezza:a1:b0.005"          # (alpha, beta) di prima (QuaRot+GPTQ W2 del lavoro), unita' grezze
MODI_GRIGLIA = ("paper+norm", "paper", "spenta")


def combo_chiave(unita: str, a: float, b: float) -> str:
    return f"{unita}:a{a:g}:b{b:g}"


def combo_valori(chiave: str):
    u, a, b = chiave.split(":")
    assert u in ("grezza", "norm") and a[0] == "a" and b[0] == "b", chiave
    return u, float(a[1:]), float(b[1:])


def griglia_drq(modo: str) -> list:
    """le combinazioni valutate in taratura, nell'ordine di spareggio (a parita' vince la prima: COMBO_FISSA e' prima)."""
    if modo not in MODI_GRIGLIA:
        raise ValueError(f"S23_DRQ_GRIGLIA {modo!r} non in {MODI_GRIGLIA}")
    if modo == "spenta":
        return [COMBO_FISSA]
    unita = ("grezza",) if modo == "paper" else ("grezza", "norm")
    g = [combo_chiave(u, a, b) for u in unita for a in GRIGLIA_ALFA[::-1] for b in GRIGLIA_BETA]
    g.remove(COMBO_FISSA)
    return [COMBO_FISSA] + g


def drq_combo(Wa_m, HA_m_raw, HB_m_raw, T, S, chiave: str = COMBO_FISSA):
    """DRQ con la combinazione data. «grezza»: identico a AL.drq_dev(..., alpha, beta). «norm»: H di ricerca e di
    controllo divise per la media della diagonale della H di ricerca (stesso fattore: i confronti restano coerenti)."""
    u, a, b = combo_valori(chiave)
    if u == "norm":
        m = float(torch.diagonal(HA_m_raw).mean())
        if not (math.isfinite(m) and m > 0):
            raise ValueError(f"DRQ norm: media della diagonale di H {m}")
        HA_m_raw, HB_m_raw = HA_m_raw / m, HB_m_raw / m
    T, cd = AL.drq_dev(Wa_m, HA_m_raw, [HB_m_raw], T, S, alpha=a, beta=b)
    cd["combo"] = chiave
    return T, cd


def scegli_combo(somme: Dict[str, float], ordine: list) -> str:
    """argmin delle somme (guadagni relativi, piu' negativo = meglio); spareggio = ordine della griglia."""
    validi = [k for k in ordine if k in somme and math.isfinite(somme[k])]
    if not validi:
        return COMBO_FISSA
    return min(validi, key=lambda k: (somme[k], ordine.index(k)))


def taratura_famiglia(registri: list, fam: str, n_tar: int, modo: str) -> dict:
    """registri = [(strato, registro.json)] degli strati GIA' CHIUSI (prima di quello di ora). Conta gli strati in cui la
    famiglia e' stata TARATA con lo stesso modo; meno di n_tar -> si tara anche ora (somme precedenti portate avanti),
    altrimenti la combinazione e' FISSA = argmin delle somme. Ripresa sicura: le somme si ricalcolano dai registri chiusi,
    uno strato rifatto non conta due volte."""
    ordine = griglia_drq(modo)
    somme: Dict[str, float] = {}
    tarati = set()
    for L, reg in registri:
        for nome, v in reg.get("densi", {}).items():
            if nome.split(".")[2] != fam:
                continue
            g = v.get("drq_griglia") or {}
            if not g.get("tarato") or g.get("modo") != modo:
                continue
            tarati.add(int(L))
            for k, x in g["rel"].items():
                somme[k] = somme.get(k, 0.0) + float(x)
    if modo == "spenta":
        return {"tarato": False, "combo": COMBO_FISSA, "modo": modo, "strati_tarati": []}
    if len(tarati) < n_tar:
        return {"tarato": True, "griglia": ordine, "somme_prec": somme, "modo": modo, "strati_tarati": sorted(tarati)}
    return {"tarato": False, "combo": scegli_combo(somme, ordine), "modo": modo, "strati_tarati": sorted(tarati),
            "somme": somme}


def costruisci_base(Wa_m, HnA_m, cN, ng):
    """BIL K=1 -> JARQ fedele (la ricetta di C senza DRQ). Ritorna (T, S, cat)."""
    _, U, pd = AL.cholesky_smorzata(HnA_m)
    T, S = AL.gptq_bil_lotto(Wa_m.unsqueeze(0), U.unsqueeze(0), ng)
    T, S = T[0], S[0]
    T, S, cj = AL.jarq_fedele_dev(Wa_m, HnA_m, T, S, cN)
    return T, S, {"damp": pd, "jarq_finale": cj.get("finale_fp16")}


def costruisci_os(Wa_m, HnA_m, HA_m_raw, HB_m_raw, cN, ng, drq: bool, combo: str = COMBO_FISSA):
    """ricetta di C sul bersaglio dato: BIL K=1 -> JARQ fedele -> DRQ (controllo = meta' B). Ritorna (T, S, cat).
    combo = COMBO_FISSA: identico al bit alla stesura di prima."""
    T, S, cat = costruisci_base(Wa_m, HnA_m, cN, ng)
    if drq:
        T, cd = drq_combo(Wa_m, HA_m_raw, HB_m_raw, T, S, combo)
        cat["drq_trit_cambiati"] = cd["trit_cambiati"]
        cat["drq_lanczos_ripieghi"] = cd["ripieghi_eigh"]
        cat["drq_combo"] = combo
    return T, S, cat


def risolvi_denso(nome: str, fam: str, tid: int, W: torch.Tensor, accA: dict, accB: dict, T0: torch.Tensor,
                  S0: torch.Tensor, opz: dict, kron: Optional[Kron] = None, P_gqa: Optional[torch.Tensor] = None,
                  monte: Optional[tuple] = None):
    """W [R,C] maestro bf16 nell'ordine GGUF (dt, dispositivo). T0/S0 = partenza (base memorizzata). monte = (T_m, S_m) del
    tensore a monte gia' scelto (solo per le famiglie K2). Ritorna dict con T, S, (S_monte), registro."""
    t0 = time.time()
    dev, dt = W.device, W.dtype
    rot = tid == T_TQ1_0R_1024
    R, C = W.shape
    ng = S0.shape[-1]
    pkw = float(opz.get("peso_kw", 1.0))
    emin, emax, modo_g = opz_guardia(opz)
    HA, MxA, NA = mescola(accA, opz["w"], pkw)                  # fp64 (accumulatori)
    HB, MxB, NB = mescola(accB, opz["w"], pkw)
    W64 = W.to(HA.device, F64)
    if any(a != 0 for a in opz["alfe"]):
        WsA = (W64 @ MxA.to(F64) @ AL.inversa_smorzata(HA.to(F64), opz["damp_qep"])).to(dev, dt)   # QEP a = 1, in fp64
    else:
        WsA = W                                                  # solo a = 0: W_a = W (GPTQ simmetrico)
    HA, MxA, HB, MxB = (x.to(dev, dt) for x in (HA, MxA, HB, MxB))
    HnA, dA = normalizza(HA)
    HnA_m = memH(HnA, rot)
    HA_m = memH(0.5 * (HA + HA.T), rot)
    HB_m = memH(0.5 * (HB + HB.T), rot)
    CB_m = mem(W @ MxB, rot)
    CA_m = mem(W @ MxA, rot)
    Wm = mem(W, rot)
    den_w = float(((W @ HB) * W).sum())                          # tr(W H_B W^T) = ||W x_q||^2 (meta' B)
    den = den_w
    rel_e = bool(opz.get("energia_relativa", True))
    reg = {"tipo": tid, "forma": [R, C], "token_A": NA, "token_B": NB, "peso_kw": pkw, "energia_relativa": rel_e,
           "energia_intervallo": [emin, emax]}
    usa_k1 = fam in opz["k1"] and kron is not None and kron.ok
    kl_locale = usa_k1 and modo_g == "kl_locale"
    if usa_k1:
        HnB, _ = normalizza(HB)
        ApB = 0.5 * (kron.Ak + memH(HnB, rot))
        ApA = 0.5 * (kron.Ac + HnA_m)
        den_kl = max(abs(AL.J_due_lati(Wm, ApB, kron.Bk)), 1e-300)

    def guardia(Qm):
        """14/10 G1: errore d'uscita SIMMETRICO verso il maestro sulla meta' B, relativo a ||W x_q||^2 (o KL locale a due
        lati sui fattori di controllo con E = Q - W, S23_GUARDIA=kl_locale per le famiglie K1)."""
        if kl_locale:
            return AL.J_due_lati(Qm - Wm, ApB, kron.Bk) / den_kl
        return errore_sim(Qm, Wm, HB_m) / max(den_w, 1e-300)

    def diagnosi(Qm):
        d = {"sim_B": errore_sim(Qm, Wm, HB_m) / max(den_w, 1e-300), "energia": rapporto_energia(Qm, HB_m, den_w),
             "JB_asim": J_asim(Qm, HB_m, CB_m) / max(den_w, 1e-300)}
        if usa_k1:
            d["kl_locale"] = AL.J_due_lati(Qm - Wm, ApB, kron.Bk) / den_kl
        return d

    Q0 = AL.ricostruisci(T0.to(dev), S0.to(dev, dt))
    cand = {"partenza": {"T": T0.to(dev), "S": S0.to(dev, dt), "c": None, "JB": guardia(Q0), "valido": True,
                         **diagnosi(Q0)}}
    e0 = cand["partenza"]["energia"] if rel_e else None
    # 13/10 DRQ per famiglia: tar = None -> la coppia di prima (al bit); tarato -> griglia su ogni QEP_a, guadagno relativo
    # per combinazione = (min_a JB_asim(a, combo) - min_a JB_asim(a, senza DRQ)) / ||W||_HB, scelta = argmin delle somme
    # (strati tarati prima + questo); non tarato -> la combinazione fissa della famiglia
    tar = opz.get("drq_taratura") if opz["drq"] else None
    combo = COMBO_FISSA if tar is None else tar.get("combo", COMBO_FISSA)
    if tar is not None and tar["tarato"]:
        den0 = max(abs(float(((W @ HB) * W).sum())), 1e-300)
        tg = time.time()
        basi, jb_g = {}, {}
        for a in opz["alfe"]:
            Wa_m = mem(W + a * (WsA - W), rot)
            Tb, Sb, catb = costruisci_base(Wa_m, HnA_m, NA * float(dA), ng)
            basi[a] = (Wa_m, Tb, Sb, catb)
            jb_g[("senza", a)] = errore_sim(AL.ricostruisci(Tb, Sb), Wm, HB_m)       # 14/10 G1: simmetrico
            for k in tar["griglia"]:
                Tk, _ = drq_combo(Wa_m, HA_m, HB_m, Tb, Sb, k)
                jb_g[(k, a)] = errore_sim(AL.ricostruisci(Tk, Sb), Wm, HB_m)
        j0 = min(jb_g[("senza", a)] for a in opz["alfe"])
        rel = {k: (min(jb_g[(k, a)] for a in opz["alfe"]) - j0) / den0 for k in tar["griglia"]}
        somme = {k: tar["somme_prec"].get(k, 0.0) + rel[k] for k in tar["griglia"]}
        combo = scegli_combo(somme, tar["griglia"])
        reg["drq_griglia"] = {"tarato": True, "modo": tar["modo"], "metrica": "simmetrica_B", "rel": rel, "scelta": combo,
                              "strati_tarati_prima": tar["strati_tarati"], "somma_scelta": somme[combo],
                              "rel_fissa": rel.get(COMBO_FISSA), "secondi": round(time.time() - tg, 1)}
    elif tar is not None:
        reg["drq_griglia"] = {"tarato": False, "modo": tar["modo"], "scelta": combo,
                              "strati_tarati": tar["strati_tarati"]}
    os_migliore = None
    for a in opz["alfe"]:
        tc = time.time()
        Wa = W + a * (WsA - W)
        Wa_m = mem(Wa, rot)
        if tar is not None and tar["tarato"]:
            _, Tb, Sb, cat = basi[a]
            cat = dict(cat)
            T, cd = drq_combo(Wa_m, HA_m, HB_m, Tb, Sb, combo)       # ricalcolata (determinista): niente 40 copie di T
            S = Sb
            cat.update({"drq_trit_cambiati": cd["trit_cambiati"], "drq_lanczos_ripieghi": cd["ripieghi_eigh"],
                        "drq_combo": combo})
        else:
            T, S, cat = costruisci_os(Wa_m, HnA_m, HA_m, HB_m, NA * float(dA), ng, opz["drq"], combo)
        Q = AL.ricostruisci(T, S)
        dg_ = diagnosi(Q)
        cand[f"qep{a:g}"] = {"T": T, "S": S, "c": None, "JB": guardia(Q), "a": a,
                             "valido": energia_ok(dg_["energia"], emin, emax, e0),
                             "JA_asim": J_asim(Q, HA_m, CA_m) / max(den_w, 1e-300), **dg_, "cat": cat,
                             "s": round(time.time() - tc, 1)}
    # 14/10 G1: base per K1/K2 = miglior QEP_a sulla guardia simmetrica fra quelli con energia nell'intervallo
    # (se nessuno lo e', il migliore sulla guardia: K1/K2 ripartono da li' e sono giudicati comunque)
    qeps = [k for k in cand if k.startswith("qep")]
    validi_q = [k for k in qeps if cand[k]["valido"]] or qeps
    os_migliore = min(validi_q, key=lambda k: (cand[k]["JB"], qeps.index(k)))
    base = cand[os_migliore]
    Wa_m = mem(W + base["a"] * (WsA - W), rot)
    if usa_k1:
        tc = time.time()
        T, S = base["T"], base["S"]
        for _ in range(opz["giri_k1"]):
            T = AL.fastkron_sd(Wa_m, ApA, kron.Bc, S)
            S = AL.fp16(AL.scale_due_lati(Wa_m, T, S, ApA, kron.Bc))
        Q = AL.ricostruisci(T, S)
        dg_ = diagnosi(Q)
        cand["k1"] = {"T": T, "S": S, "c": None, "JB": guardia(Q), "a": base["a"], **dg_,
                      "valido": energia_ok(dg_["energia"], emin, emax, e0),
                      "trit_diversi_da_qep": float((T != base["T"]).double().mean()), "s": round(time.time() - tc, 1)}
    if fam in COPPIE_K2 and monte is not None and opz["k2"]:
        tc = time.time()
        a = base["a"]
        Wa = W + a * (WsA - W)
        ClinA = W @ ((1.0 - a) * HA + a * MxA) if a != 1 else W @ MxA      # 14/10 G2: bersaglio col suo a
        T, S = base["T"], base["S"]
        migliore = None
        storia = []
        for _ in range(opz["giri_k2"]):
            Qo = K.deruota(AL.ricostruisci(T, S)) if rot else AL.ricostruisci(T, S)
            c = AL.passo_colonne(Qo, HA, ClinA, P_gqa if fam == "attn_output" else None)
            if not bool(torch.isfinite(c).all()) or float(c.abs().min()) <= 0:
                break
            Hc = HA * c.unsqueeze(0) * c.unsqueeze(1)
            Hcn, dc = normalizza(Hc)
            Hc_m = memH(0.5 * (Hc + Hc.T), rot)
            Wc_m = mem(Wa / c.unsqueeze(0), rot)
            _, U, _ = AL.cholesky_smorzata(memH(Hcn, rot))
            T2, S2 = AL.gptq_bil_lotto(Wc_m.unsqueeze(0), U.unsqueeze(0), ng)
            T2, S2, _ = AL.jarq_fedele_dev(Wc_m, memH(Hcn, rot), T2[0], S2[0], NA * float(dc))
            jA = J_asim(AL.ricostruisci(T2, S2), Hc_m, mem(ClinA * c.unsqueeze(0), rot))
            storia.append(jA)
            if migliore is None or jA < migliore[0]:
                migliore = (jA, T2, S2, c.clone())
            T, S = T2, S2
        if migliore is not None:
            _, T, S, c = migliore
            Qeff_m = mem((K.deruota(AL.ricostruisci(T, S)) if rot else AL.ricostruisci(T, S)) * c.unsqueeze(0), rot)
            Tm, Sm = monte
            if fam == "attn_output":
                cm = (P_gqa.T @ c) / P_gqa.sum(0)
            else:
                cm = c
            Smn = AL.fp16(Sm.to(dt) * cm.unsqueeze(1))
            ok = bool((cm.abs() >= 1 / opz["limite_c"]).all() and (cm.abs() <= opz["limite_c"]).all()
                      and AL.scale_scrivibili(Smn, Sm.to(dt)) and AL.scale_scrivibili(S, S))
            # la colonna effettiva dopo la piega e' quella che le scale fp16 di monte realizzano: si giudica QUELLA
            if ok:
                c_eff = Smn / Sm.to(dt).where(Sm.to(dt) != 0, torch.ones_like(Smn))
                c_eff = (c_eff * (Sm.to(dt) != 0)).sum(1) / (Sm.to(dt) != 0).sum(1).clamp_min(1)
                c_eff = P_gqa @ c_eff if fam == "attn_output" else c_eff
                Qeff_m = mem((K.deruota(AL.ricostruisci(T, S)) if rot else AL.ricostruisci(T, S)) * c_eff.unsqueeze(0), rot)
                dg_ = diagnosi(Qeff_m)
                cand["k2"] = {"T": T, "S": S, "c": c, "S_monte": Smn, "JB": guardia(Qeff_m), "a": a, **dg_,
                              "valido": energia_ok(dg_["energia"], emin, emax, e0), "Qeff": Qeff_m,
                              "storia_JA": storia, "c_min": float(cm.min()), "c_max": float(cm.max()),
                              "s": round(time.time() - tc, 1)}
            else:
                reg["k2_rifiutata"] = {"c_min": float(cm.min()), "c_max": float(cm.max())}
    # 14/10 G1: vince il minimo della guardia simmetrica fra i candidati con energia in [emin, emax]; entra solo se batte
    # la partenza (che resta il ripiego, senza controllo d'energia)
    ordine = list(cand)
    validi = [k for k in ordine if k != "partenza" and cand[k]["valido"] and math.isfinite(cand[k]["JB"])]
    scelto = min(validi, key=lambda k: (cand[k]["JB"], ordine.index(k))) if validi else "partenza"
    if not cand[scelto]["JB"] < cand["partenza"]["JB"]:
        scelto = "partenza"
    reg["rifiutati_energia"] = [k for k in ordine if k != "partenza" and not cand[k]["valido"]]
    reg["candidati"] = {k: {kk: (round(v, 6) if isinstance(v, float) else v) for kk, v in d.items()
                            if kk not in ("T", "S", "c", "S_monte", "Qeff")} for k, d in cand.items()}
    for k, d in reg["candidati"].items():
        d["dJB_rel"] = cand[k]["JB"] - cand["partenza"]["JB"]            # la guardia e' gia' relativa
    reg["metrica_guardia"] = ("kl_locale_ctl" if kl_locale else "simmetrica_B") + "+energia"
    reg["scelto"] = scelto
    if opz.get("diag_sorgenti", True):
        Qs = cand[scelto]["Qeff"] if scelto == "k2" else AL.ricostruisci(cand[scelto]["T"], cand[scelto]["S"])
        reg["sim_per_sorgente"] = {}
        for s_, v_ in accB.items():
            if not v_[2] or Qs is None:
                continue
            Hs_ = memH(0.5 * (v_[0] + v_[0].T).to(dev, dt) / v_[2], rot)
            dws = max(float(((Wm @ Hs_) * Wm).sum()), 1e-300)
            reg["sim_per_sorgente"][s_] = {"partenza": errore_sim(Q0, Wm, Hs_) / dws, "scelto": errore_sim(Qs, Wm, Hs_) / dws}
    reg["trit_cambiati"] = float((cand[scelto]["T"] != cand["partenza"]["T"]).double().mean())
    reg["secondi"] = round(time.time() - t0, 1)
    d = cand[scelto]
    return {"T": d["T"], "S": d["S"], "S_monte": d.get("S_monte"), "scelto": scelto, "registro": reg}


# ───────────────────────────── esperti (S3) ─────────────────────────────
def ruota_H_lotto(H):
    Hr = K.ruota(H)
    return K.ruota(Hr.transpose(-1, -2).contiguous()).transpose(-1, -2).contiguous()


def U_lotto(Hn):
    return torch.stack([AL.cholesky_smorzata(Hn[b])[1] for b in range(Hn.shape[0])])


def inv_lotto(H, damp):
    return torch.stack([AL.inversa_smorzata(H[b], damp) for b in range(H.shape[0])])


def costruisci_esperti_lotto(Wm, Hm, ng, jarq_lotto):
    """GPTQ bilanciato K=1 + JARQ a lotti. Wm [b,R,C], Hm [b,C,C] (media diag 1). Ritorna T int8 [b,R,C], S [b,R,ng]."""
    U = U_lotto(Hm)
    T, S = AL.gptq_bil_lotto(Wm, U, ng)
    T2, S2, cat = jarq_lotto(Wm, Hm, T.unsqueeze(1), S.unsqueeze(1), giri=3, nu2=1e-4)
    return T2[:, 0], S2[:, 0], cat


def J_asim_lotto(Qm, Hm, Cm):
    """per esperto e per riga: [b, R]."""
    return (torch.bmm(Qm, Hm) * Qm).sum(-1) - 2.0 * (Qm * Cm).sum(-1)


def J_sim_lotto(Qm, Hm, Wm):
    """14/10 G1: errore d'uscita simmetrico per esperto e per riga: [b, R] = diag((Q - W) H (Q - W)^T)."""
    E = Qm - Wm
    return (torch.bmm(E, Hm) * E).sum(-1)


def N_lotto(Qm, Hm):
    """energia d'uscita per esperto e per riga: [b, R] = diag(Q H Q^T)."""
    return (torch.bmm(Qm, Hm) * Qm).sum(-1)
