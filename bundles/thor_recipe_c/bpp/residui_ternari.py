#!/usr/bin/env python3
"""residui_ternari.py — RESIDUI TERNARI DI RANGO BASSO sulle proiezioni dell'attenzione degli strati CENTRALI (E1, 3/10).

Perche' (RICERCA_CLOUD_1820.md): la scelta della lettera nasce in poche teste di attenzione degli strati centrali
(arXiv 2407.15018). Una modifica di lm_head corregge il PRIORE delle lettere ma non ricostruisce il LEGAME
contenuto -> lettera; gli esperti di recupero (e2e_rexp) aggiungono capacita' solo nel blocco FFN/MoE. Qui si aggiunge
capacita' DENTRO l'attenzione, senza toccare un solo byte dei piani ternari esistenti:

    y = W_planes x  +  diag(gamma) . B_q (A_q x)                         (nello spazio del FILE, vedi sotto)

    A (r, in)   ternario, scala fp16 per gruppo di 256 lungo la riga (= TQ1_0, come i piani e gli esperti di recupero)
    B (out, r)  ternario, UNA scala fp16 per riga (gruppo = r); gamma (out,) fuso in quella scala -> resta UNA scala
    r = 16..64 (rango); n piani (1-2) sul residuo, come e2e_rexp

STE: avanti coi pesi ri-ternarizzati a OGNI passo (scala ottima esatta per gruppo, regola di e2e_rexp.piano_ternario),
indietro diritto sul latente fp32. gamma = 0 => uscita esattamente W_planes x (partenza = R2B3m al bit).

## Spazio del FILE (perche' le scale di gruppo restino quelle del file)
Il modulo HF e il tensore GGUF differiscono per una permutazione (mappa_completa_gguf_hf.json):
  identita'    (attn_q/k/v/output)         W_file = W_hf
  perm_righe   (attn_qkv, attn_gate)       W_file = W_hf[perm]      (y_hf = y_file[inv])
  perm_colonne (ssm_out)                   W_file = W_hf[:, perm]   (x_file = x_hf[:, perm])
I latenti vivono nell'ordine del FILE e l'avanti fa la stessa permutazione di e2e_studente.LineareTernaria: cosi' i
gruppi da 256 di A sono gli stessi che il motore vedra' (una permutazione di colonne cambierebbe i gruppi e quindi la
scala ottima) e l'esportazione non permuta nulla.

## Esportazione (scelta implementata: TENSORI TERNARI IN PIU')
Per ogni proiezione e strato due tensori TQ1_0 nuovi, accanto ai piani che restano AL BYTE:
  blk.L.<fam>_lra.weight   ne = (in, RP)   righe 0..r-1 = A_q (piano 1), righe r..RP-1 = zero  (RP = 256)
  blk.L.<fam>_lrb.weight   ne = (RP, out)  colonne 0..r-1 = B_q con scala fp16(s_riga * gamma_riga), resto zero
  (+ <fam>_lra2 / _lrb2 per il secondo piano)
B ha ne0 = r < 256 = QK_K: TQ1_0 vuole ne0 multiplo di 256, quindi si imbottisce a 256 con zeri. E' ESATTO: nel blocco
da 256 ci sono gli r valori della riga e 256-r zeri; la scala ottima del blocco (argmax c_k^2/k sui |w| ordinati) non
cambia aggiungendo zeri, e TQ1_0 rappresenta {-s, 0, +s} senza errore (d = max|w| = s). La moltiplicazione nel motore:
  cur = mul_mat(W, x) [+ piani 2, 3]  +  mul_mat(lrb, mul_mat(lra, x))
(due mul_mat da RP righe: costo trascurabile; l'uscita intermedia RP con righe nulle oltre r). Il file resta TUTTO
ternario: trit + scale fp16, come il formato esistente. ATTENZIONE: serve il motore con quel ramo nel grafo
(qwen35moe.cpp: se il tensore _lra esiste, somma il ramo); llama.cpp di serie rifiuta un GGUF con tensori in piu'
("wrong number of tensors"), esattamente come il GGUF REXP degli esperti di recupero.

## Alternativa documentata (NON implementata): fusione nei piani esistenti
W' = W_planes + B A non e' ternario: andrebbe ri-ternarizzato tutto il piano (discesa a coordinate nella metrica delle
attivazioni, come e2_riternarizza per lm_head). Misurato su lm_head (e2_scrivi_gguf): la sola RTN cancellava il 99,7 %
di una modifica rank-1, perche' un ritocco di rango basso sta sotto la soglia ternaria del piano base. La fusione
quindi passa SOLO per un piano NUOVO di forma piena (<fam>4 = ternarizzazione di B A con scale proprie per gruppo,
errore relativo ~0,19 a 1 piano, ~1 byte ogni 4,7 pesi della matrice piena): costa la taglia di un piano intero e
perde l'esattezza; i due tensori di rango basso costano ~r/out del piano e sono esatti. Per questo l'export e' (A, B).

Prove CPU: test_residui_ternari.py.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

GRUPPO = 256
RP = 256                     # righe/colonne imbottite del rango nel file (ne0 di B deve essere multiplo di 256)
TQ1_0 = 34                   # ggml_type GGML_TYPE_TQ1_0
TQ1_0_BLOCCO = (256, 54)     # (valori, byte) per blocco: qs 48 + qh 4 + d fp16 2

# famiglia -> (tipo di strato, percorso HF del Linear, relazione, chiave della permutazione in mappa_completa_gguf_hf_perm)
FAMIGLIE = {
    "attn_q": ("full_attention", "self_attn.q_proj", "identita", None),
    "attn_k": ("full_attention", "self_attn.k_proj", "identita", None),
    "attn_v": ("full_attention", "self_attn.v_proj", "identita", None),
    "attn_output": ("full_attention", "self_attn.o_proj", "identita", None),
    "attn_qkv": ("linear_attention", "linear_attn.in_proj_qkv", "perm_righe", "linear.attn_qkv.weight"),
    "attn_gate": ("linear_attention", "linear_attn.in_proj_z", "perm_righe", "linear.attn_gate.weight"),
    "ssm_out": ("linear_attention", "linear_attn.out_proj", "perm_colonne", "linear.ssm_out.weight"),
    # 8/10 shared expert (denso, in OGNI strato: tipo None = qualunque). Mappa GGUF<->HF: identita' (mappa_completa).
    # Nome GGUF del residuo: blk.L.ffn_<x>_shexp_lra/_lrb (motore thor-e1 ramo cloud/rexp-lr-shexp, THOR_SPEGNI=lrshexp).
    # down: ingresso n_ff_shexp (512 sul Thor) = multiplo di 256, come serve ad A.
    "ffn_gate_shexp": (None, "mlp.shared_expert.gate_proj", "identita", None),
    "ffn_up_shexp": (None, "mlp.shared_expert.up_proj", "identita", None),
    "ffn_down_shexp": (None, "mlp.shared_expert.down_proj", "identita", None),
}
GRUPPI = {"attn": ("attn_q", "attn_k", "attn_v", "attn_output"), "gdn": ("attn_qkv", "attn_gate", "ssm_out"),
          "shexp": ("ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp")}
GRUPPI["tutte"] = tuple(FAMIGLIE)


def famiglie_da(voci) -> tuple:
    """"attn,gdn" / "attn_q,ssm_out" -> tupla canonica (ordine di FAMIGLIE)."""
    if isinstance(voci, str):
        voci = voci.split(",")
    s = set()
    for v in (x.strip() for x in voci):
        if not v:
            continue
        if v in GRUPPI:
            s.update(GRUPPI[v])
        elif v in FAMIGLIE:
            s.add(v)
        else:
            raise ValueError(f"famiglia sconosciuta {v!r}: {sorted(FAMIGLIE) + sorted(GRUPPI)}")
    return tuple(f for f in FAMIGLIE if f in s)


# ───────────────────────────────────────── quantizzazione ternaria a gruppo variabile ─────────────────────────────────────────
def piano_ternario_g(r: torch.Tensor, g: int) -> tuple:
    """Come e2e_rexp.piano_ternario ma con gruppo g lungo l'ultima dimensione (g = 256 per A, g = r per B).
    Ritorna (T int8 stessa forma, s fp32 (..., C/g) arrotondata a fp16). Ottimo esatto per gruppo: k* = argmax c_k^2/k."""
    C = r.shape[-1]
    assert C % g == 0, (C, g)
    rb = r.reshape(*r.shape[:-1], C // g, g).float()
    a = rb.abs()
    srt = torch.sort(a, dim=-1, descending=True).values
    c = torch.cumsum(srt, -1)
    k = torch.arange(1, g + 1, device=r.device, dtype=c.dtype)
    kstar = torch.argmax(c * c / k, dim=-1, keepdim=True)
    s = (torch.gather(c, -1, kstar) / (kstar + 1).to(c.dtype)).half().float()
    if not torch.isfinite(s).all():
        # 9/10 (cura P2): due diagnosi distinte. NaN = latenti gia' non finiti (gradiente o Adam avvelenati: il caso del
        # passo 196 di COMBO A, che prima usciva col messaggio fuorviante dell'overflow); inf = |scala| > 65504.
        if torch.isnan(r).any() or torch.isnan(s).any():
            raise FloatingPointError("piano ternario: scala NaN (latenti NaN: gradiente/Adam avvelenato)")
        if not torch.isfinite(r).all():
            raise FloatingPointError("piano ternario: latenti infiniti (overflow dei latenti, non della scala fp16)")
        raise FloatingPointError("piano ternario: scala oltre il massimo fp16 (overflow)")
    soglia = torch.gather(srt, -1, kstar)
    T = torch.where(a >= soglia, torch.sign(rb), torch.zeros_like(rb))
    T = torch.where(s > 0, T, torch.zeros_like(T))
    return T.reshape(r.shape).to(torch.int8), s.squeeze(-1)


def ricostruisci_g(T: torch.Tensor, s: torch.Tensor, g: int) -> torch.Tensor:
    C = T.shape[-1]
    return (T.reshape(*T.shape[:-1], C // g, g).float() * s.unsqueeze(-1)).reshape(T.shape)


def piani_g(w: torch.Tensor, n: int, g: int) -> list:
    out, r = [], w.detach().float()
    for _ in range(n):
        T, s = piano_ternario_g(r, g)
        out.append((T, s))
        r = r - ricostruisci_g(T, s, g)
    return out


def quant_ste_g(w: torch.Tensor, n: int, g: int) -> torch.Tensor:
    q = sum(ricostruisci_g(T, s, g) for T, s in piani_g(w, n, g))
    return w + (q - w).detach()


def quant_ste_righe_gamma(w: torch.Tensor, n: int, gamma: torch.Tensor) -> torch.Tensor:
    """B (out, r) a una scala per riga, con gamma (out,) FUSO nella scala fp16: valore esatto sum_p T_p fp16(s_p gamma),
    cioe' cio' che scrive l'esportazione. Gradiente diritto sul latente, esatto su gamma (come e2e_rexp.quant_ste_gamma)."""
    g = w.shape[-1]
    pp = piani_g(w, n, g)
    base = w + (sum(ricostruisci_g(T, s, g) for T, s in pp) - w).detach()
    scal = base * gamma.view(-1, 1)
    gv = gamma.detach().float().view(-1, 1)
    esatto = sum(ricostruisci_g(T, (s * gv).half().float(), g) for T, s in pp)
    return scal + (esatto - scal).detach()


# ───────────────────────────────────────── modulo ─────────────────────────────────────────
@dataclass
class ConfigResidui:
    rango: int = 32
    piani: int = 1
    famiglie: tuple = GRUPPI["attn"] + GRUPPI["gdn"]
    strati: tuple = tuple(range(12, 28))
    seme: int = 20261003
    # 10/10 CARDAN dei densi: continuo=True -> ramo di rango basso CONTINUO (A, B latenti fp32, niente STE ternario),
    # B = 0 e gamma = 1 alla partenza (uscita = ospite al bit). Non si esporta come tensori: la correzione
    # dW = diag(gamma) B A (spazio del FILE) si PIEGA nei piani esistenti (cardan_piega_densi.py). Default False:
    # RILQ e CARDAN degli esperti invariati.
    continuo: bool = False

    def __post_init__(self):
        if not 1 <= self.rango <= RP:
            raise ValueError(f"rango {self.rango} fuori da 1..{RP}")
        if self.continuo:
            return
        if self.piani not in (1, 2):
            raise ValueError(f"piani {self.piani}: 1 o 2")


class ResiduoTernario(torch.nn.Module):
    """Latenti fp32 nell'ordine del FILE: A (r, in), B (out, r), gamma (out,) = 0. Uscita (spazio HF) del solo ramo."""

    def __init__(self, nome: str, fuori: int, dentro: int, cfg: ConfigResidui, perm_x=None, inv_y=None,
                 device=None, seme: int = 0):
        super().__init__()
        if dentro % GRUPPO and not cfg.continuo:
            raise ValueError(f"{nome}: ingresso {dentro} non multiplo di {GRUPPO}")
        self.nome, self.cfg = nome, cfg
        self.fuori, self.dentro = int(fuori), int(dentro)
        g = torch.Generator().manual_seed(seme)
        r = cfg.rango
        self.A = torch.nn.Parameter((torch.randn(r, dentro, generator=g) / math.sqrt(dentro)).to(device))
        self.B = torch.nn.Parameter((torch.randn(fuori, r, generator=g) / math.sqrt(r)).to(device))
        self.gamma = torch.nn.Parameter(torch.zeros(fuori, device=device))
        if cfg.continuo:                              # 10/10: B = 0, gamma = 1 -> ramo nullo ma gradiente su B subito
            with torch.no_grad():
                self.B.zero_()
                self.gamma.fill_(1.0)
        self.register_buffer("perm_x", None if perm_x is None else torch.as_tensor(perm_x, dtype=torch.long,
                                                                                   device=device), persistent=False)
        self.register_buffer("inv_y", None if inv_y is None else torch.as_tensor(inv_y, dtype=torch.long,
                                                                                 device=device), persistent=False)
        self.sonda: Optional[dict] = None
        self.dtype_gemm = torch.float32

    def pesi_q(self) -> tuple:
        if self.cfg.continuo:
            return self.A, self.B * self.gamma.view(-1, 1)
        return quant_ste_g(self.A, self.cfg.piani, GRUPPO), quant_ste_righe_gamma(self.B, self.cfg.piani, self.gamma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        forma = x.shape
        xf = x.reshape(-1, forma[-1]).float()
        if self.perm_x is not None:
            xf = xf[:, self.perm_x]
        Aq, Bq = self.pesi_q()
        dt = self.dtype_gemm
        y = ((xf.to(dt) @ Aq.to(dt).T) @ Bq.to(dt).T).float()
        if self.sonda is not None and torch.is_grad_enabled():
            # init dal gradiente: si ACCUMULANO C = sum g^T x (out, in), sum x x^T, sum x, n (fp64) invece di tenere
            # tutti gli x e i g (migliaia di token x 8192 colonne x decine di proiezioni non stanno in RAM)
            z = torch.zeros_like(y, requires_grad=True)
            xs, sonda = xf.detach().double(), self.sonda

            def _g(gr, xs=xs, sonda=sonda):
                g64 = gr.detach().double()
                for k, v in (("C", g64.T @ xs), ("XX", xs.T @ xs), ("sx", xs.sum(0))):
                    v = v.cpu() if sonda.get("cpu") else v
                    sonda[k] = v if sonda.get(k) is None else sonda[k] + v
                sonda["n"] = sonda.get("n", 0) + xs.shape[0]
            z.register_hook(_g)
            y = y + z
        if self.inv_y is not None:
            y = y[:, self.inv_y]
        return y.reshape(*forma[:-1], self.fuori)


class ConResiduo(torch.nn.Module):
    """Avvolge il lineare di base (nn.Linear / LineareTernaria / qualunque modulo x -> y): y = base(x) + residuo(x)."""

    def __init__(self, base: torch.nn.Module, residuo: ResiduoTernario):
        super().__init__()
        self.base = base
        self.residuo = residuo

    def forward(self, x):
        y = self.base(x)
        return y + self.residuo(x).to(y.dtype)


# ───────────────────────────────────────── installazione ─────────────────────────────────────────
def _padre(strato, percorso: str):
    pref, _, nome = percorso.rpartition(".")
    return (strato.get_submodule(pref) if pref else strato), nome


def _perm(chiave: str, perm_fn):
    if perm_fn is not None:
        return np.asarray(perm_fn(chiave), dtype=np.int64)
    import carica_strato_gguf as CSG  # sola lettura (copia congelata)
    return np.asarray(CSG.PERM[chiave], dtype=np.int64)


def installa(st, cfg: ConfigResidui, device=None, perm_fn=None) -> dict:
    """Per ogni strato chiesto e famiglia presente nel suo tipo (attn_* solo negli strati full_attention, gdn negli
    altri, shexp in tutti): ConResiduo al posto del lineare. Verifica al bit che a gamma = 0 l'uscita non cambi. {nome GGUF: residuo}."""
    out = {}
    dev = device or getattr(st, "device", None)
    tipi = st.cfg.layer_types
    for L in cfg.strati:
        strato = st.moduli[L]
        for fam in cfg.famiglie:
            tipo, percorso, rel, chiave = FAMIGLIE[fam]
            if tipo is not None and tipi[L] != tipo:
                continue
            padre, nome_lin = _padre(strato, percorso)
            lin = getattr(padre, nome_lin)
            fuori, dentro = int(lin.out_features), int(lin.in_features)
            px = iy = None
            if rel == "perm_colonne":
                px = _perm(chiave, perm_fn)
            elif rel == "perm_righe":
                p = _perm(chiave, perm_fn)
                iy = np.empty_like(p)
                iy[p] = np.arange(len(p))
            nome = f"blk.{L}.{fam}"
            res = ResiduoTernario(nome, fuori, dentro, cfg, perm_x=px, inv_y=iy, device=dev,
                                  seme=cfg.seme + 1009 * L + list(FAMIGLIE).index(fam))
            nuovo = ConResiduo(lin, res)
            with torch.no_grad():
                pdev = next((p.device for p in lin.parameters()), None) or next(
                    (b.device for b in lin.buffers() if b is not None), None) or dev or torch.device("cpu")
                pdt = next((p.dtype for p in lin.parameters() if p.is_floating_point()), torch.float32)
                x = torch.randn(3, dentro, generator=torch.Generator().manual_seed(L)).to(pdev, pdt)
                if not torch.equal(nuovo(x), lin(x)):
                    raise RuntimeError(f"{nome}: il residuo a gamma = 0 cambia l'uscita")
            setattr(padre, nome_lin, nuovo)
            out[nome] = res
    if not out:
        raise ValueError(f"nessuna proiezione {cfg.famiglie} negli strati {cfg.strati}")
    return out


def strati_di(mods: dict) -> list:
    return sorted({int(n.split(".")[1]) for n in mods})


def parametri(mods: dict) -> list:
    return [p for n in sorted(mods) for p in mods[n].parameters()]


def statistiche(X: torch.Tensor, Gd: torch.Tensor) -> dict:
    """Le stesse statistiche che la sonda accumula, da X (N, in) e Gd (N, out) nell'ordine del file (per le prove)."""
    X, Gd = X.double(), Gd.double()
    return {"C": Gd.T @ X, "XX": X.T @ X, "sx": X.sum(0), "n": X.shape[0]}


@torch.no_grad()
def inizializza_da_gradiente(res: ResiduoTernario, stat: dict) -> dict:
    """stat = {"C": sum g^T x (out, in), "XX": sum x x^T, "sx": sum x, "n"} (sonda o `statistiche`), ordine del file.
    C e' il gradiente del peso pieno; SVD troncata: A <- V_r^T / std(V_r x) (|A x| ~ 1), B <- U_r S_r^(1/2)
    normalizzato per riga, gamma resta 0 (partenza al bit). Il primo passo su gamma e' allora il piu' ripido possibile
    nel rango r (stessa idea di e2e_rexp.inizializza_da_gradiente / LoRA-GA 2407.05000)."""
    r = res.cfg.rango
    C = stat["C"].double().cpu()
    n = max(int(stat["n"]), 1)
    mx = stat["sx"].double().cpu() / n
    cov = stat["XX"].double().cpu() / n - torch.outer(mx, mx)
    U, S, Vh = torch.linalg.svd(C, full_matrices=False)
    k = min(r, S.numel())
    V = Vh[:k]
    sx = torch.einsum("ki,ij,kj->k", V, cov, V).clamp_min(1e-24).sqrt()
    A = torch.zeros(r, res.dentro, dtype=torch.float64)
    B = torch.zeros(res.fuori, r, dtype=torch.float64)
    A[:k] = V / sx[:, None]
    B[:, :k] = U[:, :k] * S[:k].sqrt()
    B /= B.norm(dim=1, keepdim=True).clamp_min(1e-12)
    if k < r:                                                          # rango del gradiente minore di r: resto casuale
        g = torch.Generator().manual_seed(res.cfg.seme)
        A[k:] = torch.randn(r - k, res.dentro, generator=g, dtype=torch.float64) / math.sqrt(res.dentro)
        B[:, k:] = torch.randn(res.fuori, r - k, generator=g, dtype=torch.float64) / math.sqrt(r)
    res.A.copy_(A.to(res.A))
    res.B.copy_(B.to(res.B))
    res.gamma.zero_()
    tot = float((S ** 2).sum())
    return {"energia_rango": float((S[:k] ** 2).sum() / tot) if tot > 0 else 0.0, "token": n}


# ───────────────────────────────────────── esportazione ─────────────────────────────────────────
@torch.no_grad()
def delta_file(res: ResiduoTernario) -> torch.Tensor:
    """10/10 (modo continuo): dW = diag(gamma) B A (out, in) fp64 nell'ordine del FILE (= tensore GGUF), cioe' la
    correzione che cardan_piega_densi.py piega nei piani esistenti. In modo ternario: il prodotto dei valori quantizzati."""
    A, Bq = res.pesi_q()
    return (Bq.detach().double() @ A.detach().double()).cpu()


def byte_tq1_0(W: np.ndarray) -> np.ndarray:
    """(righe, C) fp32 con valori {-s, 0, +s} per blocco -> byte TQ1_0 (righe, C/256*54). gguf-py (quants.quantize)."""
    from gguf import GGMLQuantizationType as QT, quants
    W = np.ascontiguousarray(W, dtype=np.float32)
    return quants.quantize(W, QT.TQ1_0)


def deq_tq1_0(b: np.ndarray, C: int) -> np.ndarray:
    from gguf import GGMLQuantizationType as QT, quants
    return quants.dequantize(np.ascontiguousarray(b), QT.TQ1_0).reshape(-1, C)


@torch.no_grad()
def matrici_file(res: ResiduoTernario) -> list:
    """[(A_pad (RP, in), B_pad (out, RP))] fp32 per piano, nell'ordine del file, gamma fuso nelle scale di B:
    esattamente i valori dell'avanti (A_q, B_q) imbottiti di zeri."""
    r, n = res.cfg.rango, res.cfg.piani
    pa = piani_g(res.A, n, GRUPPO)
    pb = piani_g(res.B, n, r)
    gv = res.gamma.detach().float().view(-1, 1)
    out = []
    for (Ta, sa), (Tb, sb) in zip(pa, pb):
        A = torch.zeros(RP, res.dentro)
        A[:r] = ricostruisci_g(Ta, sa, GRUPPO).cpu()
        B = torch.zeros(res.fuori, RP)
        B[:, :r] = ricostruisci_g(Tb, (sb * gv).half().float(), r).cpu()
        out.append((A.numpy(), B.numpy()))
    return out


def tensori_gguf(mods: dict) -> tuple:
    """({nome GGUF: byte uint8 (righe, byte_riga)}, {nome: {"type": 34, "ne": [ne0, ne1]}}). Piano p > 1: suffisso p."""
    t, info = {}, {}
    for nome in sorted(mods):
        for p, (A, B) in enumerate(matrici_file(mods[nome]), 1):
            suff = "" if p == 1 else str(p)
            for k, W in (("lra", A), ("lrb", B)):
                n = f"{nome}_{k}{suff}.weight"
                t[n] = byte_tq1_0(W)
                info[n] = {"type": TQ1_0, "ne": [int(W.shape[1]), int(W.shape[0])]}
    return t, info


def salva_latenti(mods: dict, p, meta: dict) -> None:
    torch.save({"meta": meta, "cfg": next(iter(mods.values())).cfg.__dict__,
                "latenti": {n: {k: v.detach().cpu() for k, v in m.state_dict().items()} for n, m in mods.items()}}, p)


def carica_latenti(mods: dict, p) -> dict:
    d = torch.load(p, map_location="cpu")
    for n, sd in d["latenti"].items():
        mods[n].load_state_dict(sd)
    return d["meta"]
