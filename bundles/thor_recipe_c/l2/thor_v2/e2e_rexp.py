#!/usr/bin/env python3
"""e2e_rexp — ESPERTI DI RECUPERO sintetici per Thor-T (idea di Angelo, 3/10/2026).

Per ogni strato MoE scelto: n_r esperti SwiGLU nuovi (stessa forma degli esperti originali: H=2048 -> F=512 -> H) con un
router proprio g_r sullo STESSO ingresso del blocco MoE; la loro uscita si SOMMA al blocco:

    y_L(x) = MoE_R2B3m(x) + sum_{e in topk_r(x)} w_e(x) * D_e( silu(G_e x) * (U_e x) ),   w = softmax(R x) rinormalizzata

TUTTO ternario (decisione di Angelo 3/10): G, U, D e il router R sono somme di 1-3 piani s*T, T in {-1,0,+1}, scala fp16
per gruppo di 256 lungo la riga (formato TQ1_0 di ggml: la scala e' l'unico numero non ternario, come nel formato esistente).

Perche' cosi' (dettaglio in RICETTA_THOR.md §7.9):
- pesi originali FERMI + D latente = 0 all'inizio  =>  y_L = MoE_R2B3m esattamente al passo 0 (come B = 0 in LoRA): la classe
  di funzioni contiene R2B3m, la nuova capacita' puo' solo AGGIUNGERE; la compensazione forgiata fra strati resta intatta.
- obiettivo del MODELLO intero (KL maestro -> studente sui logit), mai l'errore locale dello strato (PERCHE_LOCALE_FALLISCE).
- H2 (fatti = ritocchi di rango basso sotto la soglia ternaria): in un tensore SEPARATO la scala si adatta all'ampiezza del
  ritocco (piccola), quindi un ritocco che nel piano base cade sotto soglia (cancellato al 100 %) qui si rappresenta con la
  distorsione relativa del ternario (~0,19 Lloyd-Max a 1 piano, meno con 2 piani).
- inizializzazione allineata al gradiente (stile LoRA-GA, arXiv 2407.05000) per gruppo di token: k-means sugli ingressi ->
  router e porta G; per ogni gruppo SVD della covarianza incrociata C_e = sum g x^T (g = dKL/dy del blocco) -> righe di U =
  direzioni d'ingresso dominanti (l'esperto parte come mappa lineare di rango basso sul suo gruppo). Con D = 0 il primo gradiente su D e' sum w g act^T: le feature allineate ai vettori singolari di C_e
  massimizzano ||dL/dD|| (discesa iniziale piu' ripida) senza mai usare un bersaglio locale.

Quantizzazione nel passo (STE): per ogni piano, per gruppo di 256 la scala OTTIMA esatta del ternario (k* = argmax c_k^2/k
sui |w| ordinati, s = c_k/k: stessa regola «scala di gruppo = ottimo ternario esatto» della ricetta), arrotondata a fp16;
piano p sul residuo dei piani precedenti. Avanti con i pesi quantizzati, indietro diritto sul latente fp32.
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

GRUPPO = 256
FAMIGLIE_R = ("gate", "up", "down")


# ───────────────────────────────────────── quantizzazione ternaria ─────────────────────────────────────────
def piano_ternario(r: torch.Tensor) -> tuple:
    """Un piano: (T int8 stessa forma, s fp32 (..., C/256) gia' arrotondata a fp16). Ottimo esatto per gruppo."""
    C = r.shape[-1]
    assert C % GRUPPO == 0, C
    rb = r.reshape(*r.shape[:-1], C // GRUPPO, GRUPPO).float()
    a = rb.abs()
    srt = torch.sort(a, dim=-1, descending=True).values
    c = torch.cumsum(srt, -1)
    k = torch.arange(1, GRUPPO + 1, device=r.device, dtype=c.dtype)
    kstar = torch.argmax(c * c / k, dim=-1, keepdim=True)                # 0-based
    s = torch.gather(c, -1, kstar) / (kstar + 1).to(c.dtype)
    s = s.half().float()                                                  # la scala del formato e' fp16
    if not torch.isfinite(s).all():
        raise FloatingPointError("piano ternario: scala oltre il massimo fp16 (latente fuori misura)")
    soglia = torch.gather(srt, -1, kstar)                                 # il k*-esimo |w|
    T = torch.where(a >= soglia, torch.sign(rb), torch.zeros_like(rb))
    T = torch.where(s > 0, T, torch.zeros_like(T))
    # pareggi sulla soglia: il conteggio puo' superare k*; lo scarto e' irrilevante (stessa s, |w| uguali)
    return T.reshape(r.shape).to(torch.int8), s.squeeze(-1)


def ricostruisci(T: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    C = T.shape[-1]
    return (T.reshape(*T.shape[:-1], C // GRUPPO, GRUPPO).float() * s.unsqueeze(-1)).reshape(T.shape)


def piani(w: torch.Tensor, n: int) -> list:
    """n piani sul residuo: [(T1, s1), (T2, s2), ...]."""
    out, r = [], w.detach().float()
    for _ in range(n):
        T, s = piano_ternario(r)
        out.append((T, s))
        r = r - ricostruisci(T, s)
    return out


def quant_ste(w: torch.Tensor, n: int) -> torch.Tensor:
    """Pesi quantizzati a n piani con gradiente diritto sul latente."""
    q = sum(ricostruisci(T, s) for T, s in piani(w, n))
    return w + (q - w).detach()


# ───────────────────────────────────────── modulo ─────────────────────────────────────────
def quant_ste_gamma(w: torch.Tensor, n: int, gamma: torch.Tensor) -> torch.Tensor:
    """(3/10, correzione del coordinatore) uscita D_e * gamma_e con gamma FUSO nelle scale fp16 del formato: il valore
    esatto e' sum_p T_p * fp16(s_p * gamma_e), cioe' cio' che scrive l'esportazione (il file resta ternario + scale fp16).
    Gradiente: diritto sul latente w, esatto su gamma. w (E, R, C), gamma (E,)."""
    pp = piani(w, n)
    qd = sum(ricostruisci(T, s) for T, s in pp)
    base = w + (qd - w).detach()
    gv = gamma.detach().float().view(-1, *([1] * (w.dim() - 1)))
    scal = base * gamma.view(-1, *([1] * (w.dim() - 1)))
    esatto = sum(ricostruisci(T, (s * gv).half().float()) for T, s in pp)
    return scal + (esatto - scal).detach()


class _AggiungiPerdita(torch.autograd.Function):
    """y invariato; nell'indietro inietta d(aux)/d(.) con peso 1 (perdite ausiliarie del router dentro l'indietro per strato)."""

    @staticmethod
    def forward(ctx, y, aux):
        ctx.save_for_backward(aux)
        return y.view_as(y)

    @staticmethod
    def backward(ctx, g):
        (aux,) = ctx.saved_tensors
        return g, torch.ones_like(aux)


@dataclass
class ConfigRexp:
    n_r: int = 8                 # esperti di recupero instradati per strato
    k_r: int = 2                 # quanti per token
    F: int = 512                 # larghezza intermedia (= moe_ff degli esperti originali)
    piani_esperti: int = 1       # 1-2 (il motore li somma come secondo piano)
    piani_router: int = 3        # 1-3
    condiviso: bool = True       # + 1 esperto di recupero CONDIVISO (sempre acceso) per strato
    aux_bil: float = 0.01        # bilanciamento del carico (Switch, 2101.03961)
    aux_z: float = 1e-3          # z-loss del router (ST-MoE, 2202.08906)
    strati: tuple = tuple(range(10, 30))


class EspertiRecupero(torch.nn.Module):
    """Latenti fp32: R (n_r,H) router; G/U (n_r,F,H), D (n_r,H,F) instradati; Gs/Us (F,H), Ds (H,F) condiviso;
    gamma (n_r), gamma_s (1) = 0 => uscita esattamente 0 (partenza = R2B3m al bit), D casuale ternario (niente trappola dello
    zero: con D = 0 la scala ottima e' 0 e G/U non ricevono mai gradiente finche' D resta nullo)."""

    def __init__(self, H: int, cfg: ConfigRexp, device=None, seme: int = 0):
        super().__init__()
        self.cfg = cfg
        self.H = H
        g = torch.Generator(device="cpu").manual_seed(seme)
        sh, sf = 1.0 / math.sqrt(H), 1.0 / math.sqrt(cfg.F)
        rn = lambda *f: torch.randn(*f, generator=g)                  # noqa: E731
        self.R = torch.nn.Parameter(torch.zeros(cfg.n_r, H, device=device))
        self.G = torch.nn.Parameter((rn(cfg.n_r, cfg.F, H) * sh).to(device))
        self.U = torch.nn.Parameter((rn(cfg.n_r, cfg.F, H) * sh).to(device))
        self.D = torch.nn.Parameter((rn(cfg.n_r, H, cfg.F) * sf).to(device))
        self.gamma = torch.nn.Parameter(torch.zeros(cfg.n_r, device=device))
        if cfg.condiviso:
            self.Gs = torch.nn.Parameter((rn(1, cfg.F, H) * sh).to(device))
            self.Us = torch.nn.Parameter((rn(1, cfg.F, H) * sh).to(device))
            self.Ds = torch.nn.Parameter((rn(1, H, cfg.F) * sf).to(device))
            self.gamma_s = torch.nn.Parameter(torch.zeros(1, device=device))
        self.dtype_gemm = torch.float32
        self.sonda: Optional[dict] = None       # inizializzazione: raccoglie (x, dL/dy) nel ricalcolo dell'indietro
        self.ultimo_idx: Optional[torch.Tensor] = None
        self.peso_aux = 0.0                     # impostato dal driver (1 / micro-lotti del passo); 0 = niente perdite ausiliarie
        self.stat: dict = {}

    def pesi_q(self) -> dict:
        c = self.cfg
        q = {"R": quant_ste(self.R, c.piani_router), "G": quant_ste(self.G, c.piani_esperti),
             "U": quant_ste(self.U, c.piani_esperti), "D": quant_ste_gamma(self.D, c.piani_esperti, self.gamma)}
        if c.condiviso:
            q.update({"Gs": quant_ste(self.Gs, c.piani_esperti), "Us": quant_ste(self.Us, c.piani_esperti),
                      "Ds": quant_ste_gamma(self.Ds, c.piani_esperti, self.gamma_s)})
        return q

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x (N, H) -> (N, H) nel dtype di x."""
        q = self.pesi_q()
        xf = x.float()
        logit = xf @ q["R"].T
        prob = torch.softmax(logit, -1)
        idx = torch.topk(prob, self.cfg.k_r, dim=-1).indices
        self.ultimo_idx = idx.detach()
        w = prob.gather(-1, idx)
        w = w / w.sum(-1, keepdim=True)
        dt = self.dtype_gemm
        xg = xf.to(dt)
        y = torch.zeros_like(xf)
        for e in range(self.cfg.n_r):
            tok, slot = (idx == e).nonzero(as_tuple=True)
            if tok.numel() == 0:
                continue
            xe = xg.index_select(0, tok)
            h = F.silu(xe @ q["G"][e].to(dt).T) * (xe @ q["U"][e].to(dt).T)
            o = (h @ q["D"][e].to(dt).T).float() * w[tok, slot].unsqueeze(-1)
            y = y.index_add(0, tok, o)
        if self.cfg.condiviso:
            h = F.silu(xg @ q["Gs"][0].to(dt).T) * (xg @ q["Us"][0].to(dt).T)
            y = y + (h @ q["Ds"][0].to(dt).T).float()
        if self.peso_aux > 0 and torch.is_grad_enabled():
            n = self.cfg.n_r
            f = torch.bincount(idx.reshape(-1), minlength=n).float() / idx.numel()
            bil = n * (f * prob.mean(0)).sum()
            zl = torch.logsumexp(logit, -1).pow(2).mean()
            self.stat = {"bil": float(bil.detach()), "z": float(zl.detach()), "carico_max": float(f.max())}
            y = _AggiungiPerdita.apply(y, (self.cfg.aux_bil * bil + self.cfg.aux_z * zl) * self.peso_aux)
        if self.sonda is not None and torch.is_grad_enabled():
            z = torch.zeros_like(y, requires_grad=True)
            sonda = self.sonda
            sonda["x"].append(x.detach().float().cpu() if sonda.get("cpu") else x.detach().float())

            def _g(gr, sonda=sonda):
                sonda["g"].append(gr.detach().float().cpu() if sonda.get("cpu") else gr.detach().float())
            z.register_hook(_g)
            y = y + z
        return y.to(x.dtype)

    def taylor(self) -> float:
        """Stima al primo ordine di quanto PEGGIOREREBBE la perdita togliendo gli esperti di questo strato:
        -sum gamma * dL/dgamma (l'uscita aggiunta e' lineare in gamma). Sensibilita' per l'allocazione di P2."""
        t = 0.0
        for gm in (self.gamma, getattr(self, "gamma_s", None)):
            if gm is not None and gm.grad is not None:
                t += float(-(gm.detach() * gm.grad).sum())
        return t


# ───────────────────────────────────────── inizializzazione allineata al gradiente ─────────────────────────────────────────
def kmeans_coseno(X: torch.Tensor, k: int, giri: int = 20, seme: int = 0) -> tuple:
    """k-means sferico (coseno). Ritorna (centroidi unitari (k,H), etichette (N,))."""
    Xn = F.normalize(X.float(), dim=-1)
    g = torch.Generator(device="cpu").manual_seed(seme)
    C = Xn[torch.randperm(Xn.shape[0], generator=g)[:k].to(X.device)].clone()
    for _ in range(giri):
        lab = (Xn @ C.T).argmax(-1)
        for j in range(k):
            m = lab == j
            if m.any():
                C[j] = F.normalize(Xn[m].sum(0), dim=0)
            else:                                           # gruppo vuoto: riparte dal punto peggio servito
                C[j] = Xn[(Xn @ C.T).max(-1).values.argmin()]
    return C, (Xn @ C.T).argmax(-1)


@torch.no_grad()
def inizializza_da_gradiente(mod: EspertiRecupero, X: torch.Tensor, Gd: torch.Tensor, logit_spread: float = 4.0,
                             seme: int = 0) -> dict:
    """X (N,H) ingressi del blocco MoE, Gd (N,H) dL/dy del blocco (stessa riga = stesso token).
    Router <- centroidi (scalati: logit fra gruppi ~ logit_spread); U <- vettori singolari destri di C_e = Gd_e^T X_e,
    G <- centroide (porta ~ costante sul gruppo);
    D casuale ternario, gamma = 0 (partenza = R2B3m al bit). Ritorna diagnostica (energia catturata per gruppo)."""
    c = mod.cfg
    Cent, lab = kmeans_coseno(X, c.n_r, seme=seme)
    nx = X.float().norm(dim=-1).mean()
    mod.R.copy_(Cent * (logit_spread / max(float(nx), 1e-6)))
    diag = {"token_per_gruppo": [], "energia_topF": []}
    for e in range(c.n_r):
        m = lab == e
        Xe, Ge = X[m].float(), Gd[m].float()
        diag["token_per_gruppo"].append(int(m.sum()))
        if Xe.shape[0] < 2:
            continue
        Ce = Ge.T @ Xe                                       # (H, H): sum_t g_t x_t^T
        _, S, Vh = torch.linalg.svd(Ce, full_matrices=False)
        r = min(c.F, Vh.shape[0])
        V = Vh[:r]                                           # (r, H) direzioni d'ingresso
        z = Xe @ V.T
        sd = z.std(0)
        a = torch.where(sd > 0.05 * sd.max(), 1.0 / sd.clamp_min(1e-30), torch.zeros_like(sd))   # |U x| ~ 1; direzioni
        #   quasi vuote (gruppo con meno token di H) a ZERO: con 1/std puro la scala superava il massimo fp16 -> inf * trit 0 = NaN (prova driver)
        W = torch.zeros(c.F, mod.H, device=X.device)
        W[:r] = V * a.unsqueeze(-1)
        mod.U[e].copy_(W)
        # porta: centroide del gruppo, scalato perche' silu(G x) ~ silu(2) ~ costante positiva sui token del gruppo
        # => act ~ cost * (V x): l'esperto parte come mappa LINEARE di rango r sul suo gruppo (il «ritocco di rango
        # basso» di H2), che e' cio' che correla col gradiente; con G = U l'attivazione z*silu(z) e' quasi pari e la
        # correlazione con un gradiente lineare in x sparisce (prova test_init_gradiente_allinea, a pari norma delle feature: 1,83x sul caso).
        mu = (Xe @ Cent[e]).mean().clamp_min(1e-6)
        mod.G[e].copy_(Cent[e].expand(c.F, -1) * (2.0 / mu))
        diag["energia_topF"].append(float((S[:r] ** 2).sum() / (S ** 2).sum().clamp_min(1e-30)))
    if c.condiviso:                          # condiviso: U sulle direzioni dominanti di TUTTI i token, porta sul centroide globale
        Cg = Gd.float().T @ X.float()
        _, _, Vh = torch.linalg.svd(Cg, full_matrices=False)
        r = min(c.F, Vh.shape[0]); V = Vh[:r]
        sd = (X.float() @ V.T).std(0)
        a = torch.where(sd > 0.05 * sd.max(), 1.0 / sd.clamp_min(1e-30), torch.zeros_like(sd))
        W = torch.zeros(c.F, mod.H, device=X.device); W[:r] = V * a.unsqueeze(-1)
        mod.Us[0].copy_(W)
        cg = F.normalize(X.float().mean(0), dim=0)
        mu = (X.float() @ cg).mean().clamp_min(1e-6)
        mod.Gs[0].copy_(cg.expand(c.F, -1) * (2.0 / mu))
        mod.gamma_s.zero_()
    mod.gamma.zero_()                        # D resta casuale ternario: l'uscita parte nulla per gamma = 0
    return diag


# ───────────────────────────────────────── installazione nello studente ─────────────────────────────────────────
def installa(st, cfg: ConfigRexp, device=None, dtype_gemm=None) -> dict:
    """Aggiunge `rexp` ai MoEStudente degli strati scelti. Ritorna {L: EspertiRecupero}. Senza chiamata: studente al bit."""
    out = {}
    dev = device or st.device
    for L in cfg.strati:
        moe = st.moduli[L].mlp
        H = moe.router_w.shape[1]
        m = EspertiRecupero(H, cfg, device=dev, seme=1000 + L)
        if dtype_gemm is not None:
            m.dtype_gemm = dtype_gemm
        moe.rexp = m
        out[L] = m
    return out


def parametri(mods: dict) -> list:
    return [p for L in sorted(mods) for p in mods[L].parameters()]


# ───────────────────────────────────────── esportazione GGUF ─────────────────────────────────────────
def _byte_tq1(T: torch.Tensor, s: torch.Tensor) -> np.ndarray:
    """Piano ternario -> byte TQ1_0 (forma (..., righe, C/256*54))."""
    sys.path.insert(0, '@@ROOT:models@@/build/llamacpp-qat2/gguf-py')
    from gguf import quants, GGMLQuantizationType as QT
    W = ricostruisci(T, s).cpu().numpy().astype(np.float32)
    shape = W.shape
    b = quants.quantize(W.reshape(-1, shape[-1]), QT.TQ1_0)
    return b.reshape(*shape[:-1], b.shape[-1])


NOMI = {"R": "ffn_gate_inp_rexp", "G": "ffn_gate_rexps", "U": "ffn_up_rexps", "D": "ffn_down_rexps",
        "Gs": "ffn_gate_rshexp", "Us": "ffn_up_rshexp", "Ds": "ffn_down_rshexp"}


def tensori_gguf(mods: dict) -> dict:
    """{nome GGUF: byte uint8} per tutti gli strati. Piano p>1 -> suffisso p (es. ffn_gate_inp_rexp2). gamma fuso nelle
    scale fp16 di D/Ds (stessi valori dell'avanti). Condiviso: tensori 2D (forma degli esperti a esperto unico)."""
    out = {}
    for L, m in sorted(mods.items()):
        c = m.cfg
        voci = [("R", c.piani_router, None), ("G", c.piani_esperti, None), ("U", c.piani_esperti, None),
                ("D", c.piani_esperti, m.gamma)]
        if c.condiviso:
            voci += [("Gs", c.piani_esperti, None), ("Us", c.piani_esperti, None), ("Ds", c.piani_esperti, m.gamma_s)]
        for k, n, gm in voci:
            for p, (T, s) in enumerate(piani(getattr(m, k), n), 1):
                if gm is not None:
                    s = (s * gm.detach().float().view(-1, *([1] * (s.dim() - 1)))).half().float()
                suff = "" if p == 1 else str(p)
                b = _byte_tq1(T, s)
                out[f"blk.{L}.{NOMI[k]}{suff}.weight"] = b[0] if k.endswith("s") else b
    return out


def salva_latenti(mods: dict, p, meta: dict) -> None:
    torch.save({"meta": meta, "cfg": next(iter(mods.values())).cfg.__dict__,
                "latenti": {L: {k: v.detach().cpu() for k, v in m.state_dict().items()} for L, m in mods.items()}}, p)


def carica_latenti(mods: dict, p) -> dict:
    d = torch.load(p, map_location="cpu")
    for L, sd in d["latenti"].items():
        mods[int(L)].load_state_dict(sd)
    return d["meta"]


# ───────────────────────────────────────── KL col maestro top-k (cache) ─────────────────────────────────────────
@torch.no_grad()
def topk_maestro(lT: torch.Tensor, k: int = 64) -> tuple:
    """logit del maestro (P, V) fp32 -> (indici int32 (P,k), log p_T fp32 (P,k)) dei k token piu' probabili."""
    lp = torch.log_softmax(lT.float(), -1)
    v, i = torch.topk(lp, k, dim=-1)
    return i.to(torch.int32), v


class KLTop(torch.autograd.Function):
    """sum_t KL(p_T || p_S) con p_T ristretta ai k token in cache + un SECCHIO per la massa restante (correzione della massa:
    r_T log(r_T / r_S), r = 1 - massa dei k). Esatta sul secchio, nessun token fuori dai k trattato a zero.
    dL/dz_j = p_S,j - p_T,j (j nei k) ;  p_S,j (1 - r_T / r_S) (j fuori). Il gradiente si calcola nell'avanti, a pezzi."""

    @staticmethod
    def forward(ctx, hS, idx, lpT, logit_fn, per_fn, pezzo: int = 512):
        tot = torch.zeros((), dtype=torch.float64, device=hS.device)
        dh = torch.empty_like(hS, dtype=torch.float32)
        for a in range(0, hS.shape[0], pezzo):
            z = logit_fn(hS[a:a + pezzo].float())
            lpS = torch.log_softmax(z, -1); del z
            pS = lpS.exp()
            ix = idx[a:a + pezzo].long()
            lt = lpT[a:a + pezzo].float()
            pt = lt.exp()
            ls = lpS.gather(-1, ix); ps = ls.exp()
            rT = (1 - pt.sum(-1)).clamp_min(1e-12); rS = (1 - ps.sum(-1)).clamp_min(1e-12)
            tot += ((pt * (lt - ls)).sum(-1) + rT * (rT.log() - rS.log())).double().sum()
            dz = pS * (1 - rT / rS).unsqueeze(-1)
            dz.scatter_(-1, ix, ps - pt)
            dh[a:a + pezzo] = per_fn(dz)
            del lpS, pS, dz
        ctx.save_for_backward(dh)
        return tot

    @staticmethod
    def backward(ctx, g):
        (dh,) = ctx.saved_tensors
        return (dh * g.float()).to(dh.dtype), None, None, None, None, None
