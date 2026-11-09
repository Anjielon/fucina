#!/usr/bin/env python3
"""Revisione AVVERSARIA del progetto E2E sulle scale (`e2e_progetto_scelto.md`), 25/9/2026.
File NUOVO, SOLO CPU. Non modifica nessun file: importa in sola lettura `e2e_prototipo.py`
e `e2e_codici_scale.py`, legge in sola lettura i GGUF del prototipo e (a campione, pochi
tensori) due GGUF veri di casa.

Il prototipo e' tutto verde, ma e' tutto in fp32 e con 8 esperti. Queste prove cercano cio'
che a quella scala non si puo' vedere:

  R1  lo scheletro MoE del prototipo (peso INTERO materializzato, poi `W[e]` nel ciclo) ha un
      indietro che cresce come E^2: ogni `W[e]` genera un gradiente GRANDE QUANTO TUTTO W che
      autograd somma E volte. Con E = 256 (Thor) e' un costo che il conto dei tempi non vede.
      Controprova: costruzione del peso PER ESPERTO, gradienti uguali.
  R2  grandezza di dL/dW e di dL/dtheta con la perdita MEDIA per token del prototipo, contro i
      limiti di fp16 (subnormale minimo 2^-24, normale minimo 2^-14) e la precisione di bf16:
      cosa succede se l'indietro degli esperti gira in fp16 (la «ricetta B» con GEMM fp16).
  R3  Adam e' invariante a (perdita*K, eps*K): la regola G1 (eps piccolissimo) e' la STESSA
      ottimizzazione della perdita scalata di K = 1e-8/eps con eps di default — che e' la cura
      di YAQA («normalize the activation error ... default Adam eps = 1e-8») e non fa sparire i
      gradienti in un indietro a bassa precisione.
  R4  instradamento studente <-> maestro, strato per strato, PRIMA e DOPO l'allenamento
      (dai due GGUF del prototipo): la KL scende, ma l'instradamento?
  R5  scale fp16 subnormali: nel giocattolo dopo l'allenamento, e sui GGUF veri (campione):
      quanto margine in theta prima che le scale piccole diventino subnormali.
  R6  le FORME di Thor: il giocattolo ha un solo blocco per riga (H = 1024, I = 256); Thor ne ha
      due (H = 2048, I = 512). Layout blocco-nella-riga, rotazione a 2 blocchi ed export provati
      contro le librerie su quelle forme.

Uso:  CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 19 python3 e2e_revisione_prove.py
      (richiede che `e2e_test_prototipo.py` o `e2e_prototipo.py` abbiano gia' scritto i due GGUF
      in $E2E_PROTO_DIR). Esito in $E2E_PROTO_DIR/../revisione/esito_revisione.json.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import e2e_prototipo as P  # noqa: E402  (sola lettura)
import e2e_codici_scale as CS  # noqa: E402  (sola lettura)
from gguf import GGUFReader  # noqa: E402

OUT = P.DIR.parent / "revisione"
ESITO: dict = {}
FP16_SUB_MIN = 2.0 ** -24       # 5,96e-8: sotto la meta' si arrotonda a 0
FP16_NORM_MIN = 2.0 ** -14      # 6,10e-5: sotto si perde precisione (subnormale)
FP16_MAX = 65504.0


def scrivi(nome: str, **dati) -> None:
    ESITO[nome] = dati
    print(f"── {nome}: " + json.dumps(dati, default=str), flush=True)


# ─────────────────────────────────────────── R1 ───────────────────────────────────────────
def _tempo_indietro(E: int, R: int, C: int, N: int, k: int, modo: str, g: torch.Generator) -> tuple:
    q = torch.randint(-1, 2, (E, R, C), generator=g, dtype=torch.int8)
    d0 = torch.rand(E, R, C // 128, generator=g) * 0.01 + 1e-3
    x = torch.randn(N, C, generator=g)
    idx = torch.argsort(torch.rand(N, E, generator=g), dim=-1)[:, :k]
    gy = torch.randn(N, k, R, generator=g)
    th = torch.zeros(E, R, C // 128, requires_grad=True)
    tot = torch.zeros(())
    if modo == "intero":                       # come Studente.pesi + moe del prototipo
        W = (q.to(torch.float32).view(E, R, C // 128, 128) * (d0 * torch.exp(th)).unsqueeze(-1)).view(E, R, C)
        for e in range(E):
            tok, pos = torch.where(idx == e)
            if tok.numel():
                tot = tot + ((x[tok] @ W[e].T) * gy[tok, pos]).sum()
    else:                                      # peso costruito PER ESPERTO
        for e in range(E):
            tok, pos = torch.where(idx == e)
            if tok.numel():
                We = (q[e].to(torch.float32).view(R, C // 128, 128)
                      * (d0[e] * torch.exp(th[e])).unsqueeze(-1)).view(R, C)
                tot = tot + ((x[tok] @ We.T) * gy[tok, pos]).sum()
    t0 = time.perf_counter()
    tot.backward()
    return time.perf_counter() - t0, th.grad.clone()


def r1_scheletro() -> None:
    """Stesso lavoro utile (N*k righe instradate, stessa forma per esperto) al crescere di E."""
    R, C, N, k = 64, 1024, 2048, 8
    righe = []
    for E in (16, 32, 64, 128, 256):
        tA, gA = _tempo_indietro(E, R, C, N, k, "intero", torch.Generator().manual_seed(E))
        tB, gB = _tempo_indietro(E, R, C, N, k, "per_esperto", torch.Generator().manual_seed(E))
        diff = float((gA - gB).abs().max() / gB.abs().max())
        righe.append({"E": E, "indietro_peso_intero_s": round(tA, 3), "indietro_per_esperto_s": round(tB, 3),
                      "rapporto": round(tA / tB, 1), "diff_rel_gradiente": f"{diff:.1e}"})
        print(f"   E={E:4d}  intero {tA:7.3f} s   per esperto {tB:7.3f} s   x{tA / tB:6.1f}   diff {diff:.1e}", flush=True)
    a, b = righe[2], righe[4]                                   # E=64 -> E=256
    esp = math.log(a["indietro_peso_intero_s"] / b["indietro_peso_intero_s"]) / math.log(64 / 256)
    esp_b = math.log(a["indietro_per_esperto_s"] / b["indietro_per_esperto_s"]) / math.log(64 / 256)
    scrivi("R1 indietro: peso intero + W[e] nel ciclo (prototipo) contro peso per esperto",
           forma={"R": R, "C": C, "token": N, "top_k": k}, righe=righe,
           esponente_E_peso_intero_64_256=round(esp, 2), esponente_E_per_esperto_64_256=round(esp_b, 2),
           gradienti_uguali=all(float(r["diff_rel_gradiente"]) < 1e-5 for r in righe),
           traffico_extra_thor_stima=("E^2*R*C*~12 byte per famiglia e strato = 256^2*512*2048*12 B = 0,8 TB; "
                                      "x3 famiglie = 2,5 TB a strato a ogni indietro; a 200-256 GB/s ~10-12 s a "
                                      "strato, ~400-500 s ogni 8192 token (~50-60 ms/token) contro i 3,3-5,2 "
                                      "ms/token del conto: STIMA da banda, da misurare in M0"))


# ─────────────────────────────────────────── R2 ───────────────────────────────────────────
class _GradPesi:
    """dL/dW di una famiglia, rimontato (E, righe, colonne) dai pesi dei singoli esperti."""

    def __init__(self, per_esperto: dict, forma: tuple):
        self.per_esperto, self.forma = per_esperto, forma

    @property
    def grad(self) -> torch.Tensor:
        g = torch.zeros(self.forma, dtype=torch.float32)
        for e, W in self.per_esperto.items():
            if W.grad is not None:
                g[e] = W.grad
        return g


class StudenteSonda(P.Studente):
    """Come lo Studente del prototipo, ma tiene il gradiente dei pesi materializzati per esperto."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._vivi: dict = {}

    def peso_esperto(self, l: int, fam: str, e: int, d: torch.Tensor) -> torch.Tensor:
        W = super().peso_esperto(l, fam, e, d)
        if W.requires_grad:
            W.retain_grad()
            self._vivi.setdefault((l, fam), {})[e] = W
        return W

    @property
    def W_vivi(self) -> dict:
        out = {}
        for (l, fam), per in self._vivi.items():
            b = self.blocchi[(l, fam)]
            tipo = P.FAMIGLIE[fam][1]
            forma = (b.shape[0], b.shape[1], b.shape[2] // CS.LAYOUT[tipo][0] * CS.LAYOUT[tipo][1])
            out[(l, fam)] = _GradPesi(per, forma)
        return out


def _arrotonda(t: torch.Tensor, dtype) -> torch.Tensor:
    return t.to(dtype).to(torch.float64)


def _dtheta_da_dW(st: P.Studente, l: int, fam: str, dW: torch.Tensor) -> torch.Tensor:
    """dL/dtheta = sum_gruppo(trit * dL/dW) * d_eff (STE): la stessa catena di autograd."""
    q = st.trit(l, fam).to(torch.float64)
    E, Rr, Cc = q.shape
    d = st.d0[(l, fam)].double() * torch.exp(st.theta[(l, fam)].detach().double())
    ng = d.shape[-1]
    return (q.view(E, Rr, ng, -1) * dW.view(E, Rr, ng, -1)).sum(-1) * d


def r2_grandezze(D: dict, B: dict, c: P.Config, ca: P.ConfigAllena, eps_g1: float) -> None:
    st = StudenteSonda(D["Pbf"], B, c)
    n_seq = ca.seq_per_passo
    ids, hT = D["cal"][:n_seq, :-1], D["cal_hT"][:n_seq * c.ctx]
    P.gradiente_micro(st, ids, hT, D["WT"], ca.pezzo_kl, 1.0 / (n_seq * c.ctx))
    K = 1e-8 / eps_g1
    out = {}
    for fam in ("gate", "up", "down"):
        dW = torch.cat([st.W_vivi[(l, fam)].grad.reshape(-1).double() for l in range(c.L)])
        a = dW.abs()
        a_nz = a[a > 0]
        ref = torch.cat([st.theta[(l, fam)].grad.reshape(-1).double() for l in range(c.L)])
        ric = torch.cat([_dtheta_da_dW(st, l, fam, st.W_vivi[(l, fam)].grad.double()).reshape(-1)
                         for l in range(c.L)])
        coerente = float((ric - ref).norm() / ref.norm())
        riga = {"dW_mediana": f"{float(a_nz.median()):.2e}", "dW_p99": f"{float(torch.quantile(a_nz[torch.randperm(a_nz.numel(), generator=torch.Generator().manual_seed(0))[:4_000_000]], 0.99)):.2e}",
                "dW_max": f"{float(a.max()):.2e}",
                "frazione_dW_a_zero_in_fp16": f"{float((a < FP16_SUB_MIN / 2).double().mean()):.4f}",
                "frazione_dW_subnormale_fp16": f"{float(((a >= FP16_SUB_MIN / 2) & (a < FP16_NORM_MIN)).double().mean()):.4f}",
                "controllo_catena_dtheta": f"{coerente:.1e}"}
        for nome, dtype, scala in (("fp16", torch.float16, 1.0), ("bf16", torch.bfloat16, 1.0),
                                   ("fp16_scalaK", torch.float16, K)):
            dWr = _arrotonda(torch.cat([st.W_vivi[(l, fam)].grad.reshape(-1).double() * scala
                                        for l in range(c.L)]), dtype) / scala
            if not torch.isfinite(dWr).all():
                riga[f"{nome}_trabocco"] = int((~torch.isfinite(dWr)).sum())
                dWr = torch.where(torch.isfinite(dWr), dWr, torch.zeros_like(dWr))
            pezzi, o = [], 0
            for l in range(c.L):
                n = st.W_vivi[(l, fam)].grad.numel()
                pezzi.append(_dtheta_da_dW(st, l, fam, dWr[o:o + n].view_as(st.W_vivi[(l, fam)].grad)).reshape(-1))
                o += n
            g_r = torch.cat(pezzi)
            vive = ref != 0
            riga[f"{nome}_errore_rel_dtheta"] = f"{float((g_r - ref).norm() / ref.norm()):.2e}"
            riga[f"{nome}_frazione_dtheta_zero_o_segno_sbagliato"] = \
                f"{float(((g_r[vive] == 0) | (torch.sign(g_r[vive]) != torch.sign(ref[vive]))).double().mean()):.4f}"
        out[fam] = riga
    scrivi("R2 dL/dW e dL/dtheta (perdita media per token, 2048 token/passo) contro fp16/bf16",
           K_dalla_regola_G1=f"{K:.3e}", per_famiglia=out,
           nota=("Thor: 131.072 token/passo (x64 al denominatore) e 8 esperti su 256 (1/32 dei token "
                 "per esperto contro 1/4): a parita' di contributo per token, dL/dW per elemento e' "
                 "~8-20 volte PIU' PICCOLO del giocattolo (stima d'ordine, non misura)"))


# ─────────────────────────────────────────── R3 ───────────────────────────────────────────
def _corsa_r3(D: dict, B: dict, c: P.Config, ca: P.ConfigAllena, scala: float, eps: float, passi: int) -> dict:
    st = P.Studente(D["Pbf"], B, c)
    opt = torch.optim.AdamW(st.parametri(), lr=ca.lr, betas=(ca.beta1, ca.beta2), eps=eps, weight_decay=0.0)
    for t in range(passi):
        s = slice(t * ca.seq_per_passo, (t + 1) * ca.seq_per_passo)
        ids, hT = D["cal"][s, :-1], D["cal_hT"][s.start * c.ctx:s.stop * c.ctx]
        opt.zero_grad(set_to_none=True)
        n_tok = ca.seq_per_passo * c.ctx
        for j in range(0, ca.seq_per_passo, ca.seq_per_micro):
            P.gradiente_micro(st, ids[j:j + ca.seq_per_micro], hT[j * c.ctx:(j + ca.seq_per_micro) * c.ctx],
                              D["WT"], ca.pezzo_kl, scala / n_tok)
        opt.step()
    par = st.parametri()
    v = torch.cat([opt.state[p]["exp_avg_sq"].reshape(-1) for p in par])
    g = torch.cat([p.grad.reshape(-1) for p in par])
    return {"theta": torch.cat([p.detach().reshape(-1) for p in par]), "v": v, "g": g,
            "scale": torch.cat([x.view(torch.int16).reshape(-1) for x in st.scale_fp16().values()])}


def r3_invarianza(D: dict, B: dict, c: P.Config, ca: P.ConfigAllena, eps_g1: float) -> None:
    """K potenza di 2: moltiplicare per 2^k e' ESATTO in virgola mobile (fuori da subnormali e
    trabocchi), quindi le due corse devono coincidere AL BIT. Ogni scarto e' la firma di un valore
    finito nella zona subnormale di fp32 in una delle due (g, o g^2 in v)."""
    K = 2.0 ** round(math.log2(1e-8 / eps_g1))
    passi = 3
    A = _corsa_r3(D, B, c, ca, 1.0, eps_g1, passi)                  # G1: perdita media, eps piccolo
    Bk = _corsa_r3(D, B, c, ca, K, eps_g1 * K, passi)                # perdita x K, eps x K (~1e-8)
    FP32_NORM_MIN = 2.0 ** -126
    ug = A["theta"] == Bk["theta"]
    diff = (A["theta"] - Bk["theta"]).abs() / ca.lr
    v_sub_A = int(((A["v"] > 0) & (A["v"] < FP32_NORM_MIN)).sum())
    v_zero_g_A = int(((A["v"] == 0) & (A["g"] != 0)).sum())
    v_sub_B = int(((Bk["v"] > 0) & (Bk["v"] < FP32_NORM_MIN)).sum())
    scrivi("R3 Adam: (perdita*K, eps*K) == (perdita, eps), K potenza di 2, 3 passi veri",
           K=f"2^{int(math.log2(K))} = {K:.3e}", eps_A=f"{eps_g1:.3e}", eps_B=f"{eps_g1 * K:.3e}",
           theta_uguali_al_bit=f"{int(ug.sum())}/{ug.numel()}",
           scale_fp16_uguali_al_bit=f"{int((A['scale'] == Bk['scale']).sum())}/{A['scale'].numel()}",
           max_scarto_theta_in_lr=f"{float(diff.max()):.2e}",
           v_subnormale_fp32_corsa_A=v_sub_A, v_zero_con_g_non_zero_corsa_A=v_zero_g_A,
           v_subnormale_fp32_corsa_B=v_sub_B,
           g_min_non_zero_A=f"{float(A['g'].abs()[A['g'] != 0].min()):.2e}",
           conclusione=("se theta coincide al bit, G1 E' la perdita scalata di K con eps ~1e-8 (la cura di "
                        "YAQA: normalizzare la perdita); la forma scalata tiene g e g^2 lontani dalla zona "
                        "subnormale di fp32 e dall'underflow di un indietro fp16/bf16"))


# ─────────────────────────────────────────── R4 ───────────────────────────────────────────
@torch.no_grad()
def _instradamenti_maestro(Pc: dict, ids: torch.Tensor, c: P.Config) -> list:
    x = Pc["token_embd"][ids]
    out = []
    for l in range(c.L):
        x, idx, _ = P.strato_denso(x, Pc, l, c)
        out.append(idx)
    return out


@torch.no_grad()
def _instradamenti_studente(st: P.Studente, ids: torch.Tensor) -> list:
    x = st.entra(ids)
    out = []
    for l in range(st.c.L):
        x, idx = st.strato(x, l)
        out.append(idx)
    return out


def r4_instradamento(D: dict, c: P.Config) -> None:
    Pc = P.maestro_calcolo(D["Pbf"])
    studenti = {}
    for nome, f in (("prima", "studente_iniziale.gguf"), ("dopo", "studente_e2e.gguf")):
        Pbf, B = P.leggi_gguf(P.DIR / f, c)
        studenti[nome] = P.Studente(Pbf, B, c)
    held = D["held"][:, :-1]
    acc = {n: [[0, 0.0] for _ in range(c.L)] for n in studenti}
    tot = 0
    for i in range(0, held.shape[0], 16):
        ids = held[i:i + 16]
        rm = _instradamenti_maestro(Pc, ids, c)
        for n, st in studenti.items():
            rs = _instradamenti_studente(st, ids)
            for l in range(c.L):
                a, b = torch.sort(rm[l], -1).values, torch.sort(rs[l], -1).values
                acc[n][l][0] += int((a == b).all(-1).sum())
                inter = (rm[l].unsqueeze(-1) == rs[l].unsqueeze(-2)).any(-1).sum(-1)
                acc[n][l][1] += float(inter.double().sum()) / c.k
        tot += ids.numel()
    per = {n: [{"strato": l, "insieme_top_k_identico": f"{v[0] / tot:.4f}", "sovrapposizione_media": f"{v[1] / tot:.4f}"}
               for l, v in enumerate(acc[n])] for n in acc}
    kl = {n: round(P.valuta(st, D["held"], D["held_hT"], D["WT"])["kl"], 6) for n, st in studenti.items()}
    scrivi("R4 instradamento studente<->maestro sul held-out, prima e dopo", token=tot, kl_heldout=kl, **per)


# ─────────────────────────────────────────── R5 ───────────────────────────────────────────
def _statistiche_d(d: np.ndarray) -> dict:
    a = np.abs(d.astype(np.float32)).reshape(-1)
    nz = a[a > 0]
    q = float(np.quantile(nz, 1e-3))
    return {"scale": int(a.size), "zero": f"{float((a == 0).mean()):.4f}",
            "subnormali": int(((a > 0) & (a < FP16_NORM_MIN)).sum()), "negative": int((d.astype(np.float32) < 0).sum()),
            "min_non_zero": f"{float(nz.min()):.3e}", "q0001": f"{q:.3e}", "mediana": f"{float(np.median(nz)):.3e}",
            "max": f"{float(nz.max()):.3e}",
            "margine_theta_q0001_prima_del_subnormale": f"{math.log(q / FP16_NORM_MIN):.2f}" if q > FP16_NORM_MIN else "gia' sotto"}


def r5_subnormali(c: P.Config) -> None:
    _, B0 = P.leggi_gguf(P.DIR / "studente_iniziale.gguf", c)
    _, B1 = P.leggi_gguf(P.DIR / "studente_e2e.gguf", c)
    giocattolo = {}
    for k in sorted(B0):
        tipo = P.FAMIGLIE[k[1]][1]
        nb = CS.LAYOUT[tipo][0]
        d0 = CS.scale_da_blocchi(torch.from_numpy(np.ascontiguousarray(B0[k]).reshape(-1, nb)), tipo).float()
        d1 = CS.scale_da_blocchi(torch.from_numpy(np.ascontiguousarray(B1[k]).reshape(-1, nb)), tipo).float()
        giocattolo[f"blk.{k[0]}.{k[1]}"] = {
            "subnormali_prima": int(((d0.abs() > 0) & (d0.abs() < FP16_NORM_MIN)).sum()),
            "subnormali_dopo": int(((d1.abs() > 0) & (d1.abs() < FP16_NORM_MIN)).sum()),
            "min_prima": f"{float(d0.abs()[d0 != 0].min()):.3e}", "min_dopo": f"{float(d1.abs()[d1 != 0].min()):.3e}"}
    veri = {}
    campione = {"@@ROOT:models@@/gguf/tony-tern/Thor-v20-densiv5.gguf":
                ["blk.0.ffn_gate_exps.weight", "blk.20.ffn_up_exps.weight", "blk.39.ffn_gate_exps.weight"],
                "@@ROOT:models@@/gguf/tony-tern/Thor-T-ibrido-0-19-ThorT-veloce.gguf":
                ["blk.0.ffn_gate_exps.weight", "blk.0.ffn_down_exps.weight", "blk.19.ffn_up_exps.weight",
                 "blk.19.ffn_down_exps.weight"]}
    for percorso, nomi in campione.items():
        if not Path(percorso).exists():
            veri[percorso] = "assente"
            continue
        r = GGUFReader(percorso)
        tens = {t.name: t for t in r.tensors}
        for n in nomi:
            t = tens[n]
            tipo = t.tensor_type.name
            nb, _, off, k = CS.LAYOUT[tipo]
            blocchi = np.asarray(t.data).reshape(-1, nb)
            d = np.ascontiguousarray(blocchi[:, off:off + 2 * k]).view(np.float16)
            veri[f"{Path(percorso).name}:{n} ({tipo})"] = _statistiche_d(d)
        del r, tens
    scrivi("R5 scale fp16 subnormali (giocattolo prima/dopo; GGUF veri a campione)", giocattolo=giocattolo,
           gguf_veri=veri, soglia_normale_fp16=FP16_NORM_MIN,
           nota=("margine = ln(q0,001/2^-14): quanto theta puo' scendere prima che lo 0,1% piu' piccolo delle "
                 "scale diventi subnormale; E-A: |theta| <= 64*lr, E-B: <= 128*lr (256*lr con la 2a epoca)"))


# ─────────────────────────────────────────── R6 ───────────────────────────────────────────
def r6_forme_thor() -> None:
    """Il giocattolo ha H = 1024 e I = 256: UN SOLO blocco per riga in entrambi i formati. Thor ha
    H = 2048 (2 blocchi TQ1_0R_1024 per riga di gate/up, 16 scale) e I = 512 (2 blocchi TQ1_0 per
    riga di down). L'ordine blocco-nella-riga, la rotazione a 2 blocchi e l'export non erano provati
    su quelle forme. Pesi casuali (nessun addestramento: qui conta il layout), theta casuali."""
    from gguf.quants import TQ1_0 as GQ_TQ1_0
    from tq1_0r_1024_codec import dequantize_blocks as deq_r1024
    c = P.Config(H=2048, I=512, I_cond=512, E=4, L=1, ctx=32)
    Pbf = P.maestro_bf16(P.init_maestro(c, 123))
    B = P.blocchi_iniziali(Pbf, c)
    st = P.Studente(Pbf, B, c)
    g = torch.Generator().manual_seed(9)
    with torch.no_grad():
        for k in sorted(st.theta):
            st.theta[k].copy_(torch.randn(st.theta[k].shape, generator=g) * 0.05)
    scale = st.scale_fp16()
    src, dst = OUT / "r6_forme_thor_src.gguf", OUT / "r6_forme_thor_dst.gguf"
    for p in (src, dst):
        if p.exists():
            p.unlink()
    P.scrivi_gguf(src, Pbf, B, c)
    rep = P.esporta(src, dst, scale)
    Pbf2, B2 = P.leggi_gguf(dst, c)
    Pdeq = P.maestro_calcolo(Pbf2)
    forme = {}
    for (l, fam), b in B2.items():
        nome, tipo = P.FAMIGLIE[fam]
        blocchi = np.ascontiguousarray(b).reshape(-1, CS.LAYOUT[tipo][0])
        deq = GQ_TQ1_0.dequantize_blocks(blocchi) if tipo == "TQ1_0" else deq_r1024(blocchi)
        Pdeq[f"blk.{l}.{nome}"] = torch.from_numpy(deq).reshape(c.E, b.shape[1], -1)
        forme[f"blk.{l}.{fam}"] = {"forma": list(Pdeq[f"blk.{l}.{nome}"].shape), "blocchi_per_riga": b.shape[-1] // CS.LAYOUT[tipo][0],
                                   "scale_per_riga": int(st.theta[(l, fam)].shape[-1])}
    ids = torch.randint(0, c.V, (4, c.ctx), generator=g)
    with torch.no_grad():
        zS = st.logit(ids)
        x = Pdeq["token_embd"][ids]
        for l in range(c.L):
            x = P.strato_denso(x, Pdeq, l, c)[0]
        zA = P.rmsnorm(x, Pdeq["output_norm"], c.eps_norm).reshape(-1, c.H) @ Pdeq["output"].T
        z_file = P.Studente(Pbf2, B2, c).logit(ids)
    st_th0 = P.Studente(Pbf2, B2, c)          # theta = 0: le scale vengono SOLO dal file esportato
    e_lib = float((zS - zA).abs().max() / zA.abs().max())
    scrivi("R6 forme di Thor (H=2048, I=512: 2 blocchi per riga) — studente == pesi dequantizzati dalle librerie "
           "dal GGUF esportato; ricaricato dal file == studente coi theta",
           forme=forme, byte_file=rep["byte"], scale_cambiate=rep["scale_cambiate"],
           errore_rel_logit_studente_vs_librerie=f"{e_lib:.2e}",
           ricaricato_uguale_al_bit=bool(torch.equal(z_file, zS)),
           ok=bool(e_lib < 1e-4 and torch.equal(z_file, zS) and all(st_th0.theta[k].abs().max() == 0 for k in st_th0.theta)))
    r7_c_riferimento(dst, B2, scale, c)
    for p in (src, dst):
        p.unlink()


# ─────────────────────────────────────────── R7 ───────────────────────────────────────────
_LIB_GGML = "@@ROOT:models@@/build/llamacpp-tq1-k1024/build/bin/libggml-base.so"


def _deq_c(blocchi: np.ndarray, tipo: str) -> np.ndarray:
    """Dequantizzazione del CODICE C di llama.cpp (dequantize_row_tq1_0 / _tq1_0r_1024 della build
    k1024, backend CPU), chiamata con ctypes: il riferimento che T1 non usava (gguf-py e codec)."""
    import ctypes
    lib = ctypes.CDLL(_LIB_GGML)
    fn = getattr(lib, "dequantize_row_tq1_0" if tipo == "TQ1_0" else "dequantize_row_tq1_0r_1024")
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
    fn.restype = None
    nb, qk, _, _ = CS.LAYOUT[tipo]
    b = np.ascontiguousarray(blocchi, dtype=np.uint8).reshape(-1, nb)
    y = np.empty((b.shape[0], qk), dtype=np.float32)
    fn(b.ctypes.data, y.ctypes.data, ctypes.c_int64(b.shape[0] * qk))
    return y


def r7_c_riferimento(dst: Path, B2: dict, scale: dict, c: P.Config) -> None:
    """Il GGUF esportato (forme di Thor) e un campione di tensori VERI, dequantizzati dal codice C,
    contro trit(byte) x d come li usa lo studente (de-ruotati per R1024)."""
    from tq1_0r_1024_codec import S1024
    S = torch.from_numpy(S1024)
    righe = {}
    for (l, fam), b in sorted(B2.items()):
        tipo = P.FAMIGLIE[fam][1]
        nb = CS.LAYOUT[tipo][0]
        bt = torch.from_numpy(np.ascontiguousarray(b).reshape(-1, nb))
        W = CS.pesi_da_trit_scale(CS.trit_da_blocchi(bt, tipo), CS.scale_da_blocchi(bt, tipo).float(), tipo,
                                  S if tipo != "TQ1_0" else None).numpy()
        Wc = _deq_c(bt.numpy(), tipo)
        d_ok = np.array_equal(CS.scale_da_blocchi(bt, tipo).numpy().reshape(-1).view(np.uint16),
                              scale[(l, fam)].numpy().reshape(-1).view(np.uint16))
        righe[f"esportato blk.{l}.{fam} ({tipo})"] = {"uguale_al_bit": bool(np.array_equal(W, Wc)),
                                                     "max_diff_rel": f"{float(np.abs(W - Wc).max() / np.abs(Wc).max()):.1e}",
                                                     "d_nel_file_==_scale_allenate": d_ok}
    percorso = "@@ROOT:models@@/gguf/tony-tern/Thor-T-ibrido-0-19-ThorT-veloce.gguf"
    if Path(percorso).exists():
        r = GGUFReader(percorso)
        tens = {t.name: t for t in r.tensors}
        for n in ("blk.0.ffn_gate_exps.weight", "blk.19.ffn_up_exps.weight", "blk.0.ffn_down_exps.weight"):
            t = tens[n]
            tipo = t.tensor_type.name
            nb = CS.LAYOUT[tipo][0]
            for e in (0, 131, 255):
                bt = torch.from_numpy(np.ascontiguousarray(np.asarray(t.data)[e]).reshape(-1, nb))
                W = CS.pesi_da_trit_scale(CS.trit_da_blocchi(bt, tipo), CS.scale_da_blocchi(bt, tipo).float(), tipo,
                                          S if tipo != "TQ1_0" else None).numpy()
                Wc = _deq_c(bt.numpy(), tipo)
                righe[f"vero {n} esperto {e} ({tipo})"] = {"uguale_al_bit": bool(np.array_equal(W, Wc)),
                                                          "max_diff_rel": f"{float(np.abs(W - Wc).max() / max(np.abs(Wc).max(), 1e-30)):.1e}"}
        del r, tens
    scrivi("R7 codice C di llama.cpp (dequantize_row_*, CPU, ctypes) == trit x d dello studente",
           libreria=_LIB_GGML, righe=righe, tutto_al_bit=all(v["uguale_al_bit"] for v in righe.values()))


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    for f in ("studente_iniziale.gguf", "studente_e2e.gguf"):
        if not (P.DIR / f).exists():
            print(f"manca {P.DIR / f}: lancia prima e2e_test_prototipo.py")
            return 2
    c, ca = P.Config(), P.ConfigAllena()
    print("R1 (sintetico, nessun dato)", flush=True)
    r1_scheletro()
    D = P.prepara(c)
    B = P.blocchi_iniziali(D["Pbf"], c)
    eps_g1, _ = P.cancello_g1(P.Studente(D["Pbf"], B, c), D["cal"][:ca.seq_per_passo],
                              D["cal_hT"][:ca.seq_per_passo * c.ctx], D["WT"], ca)
    r2_grandezze(D, B, c, ca, eps_g1)
    r3_invarianza(D, B, c, ca, eps_g1)
    r4_instradamento(D, c)
    r5_subnormali(c)
    r6_forme_thor()
    ESITO["minuti"] = round((time.time() - t0) / 60, 2)
    (OUT / "esito_revisione.json").write_text(json.dumps(ESITO, indent=1, default=str))
    print(f"esito in {OUT / 'esito_revisione.json'} ({ESITO['minuti']} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
