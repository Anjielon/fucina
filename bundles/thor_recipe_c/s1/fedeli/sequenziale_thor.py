#!/usr/bin/env python3
"""Motore SEQUENZIALE strato per strato per le versioni FEDELI di SchurQuant (arXiv 2608.15567) e G²PTQ
(arXiv 2609.31009) sui 250 tensori densi di Thor-T. 10/10/2026 sera.

Due flussi sugli stessi token di calibrazione, tenuti in VRAM:
  * flusso PULITO: il modello in bf16 (safetensors HF di Ornith-1.5, @@ROOT:models@@/gguf/thor-src-hf), uno strato
    alla volta letto dal disco — e' il «clean teacher» dei due lavori (loro: FP16; noi bf16, l'originale);
  * flusso QUANTIZZATO (prefisso quantizzato): l'ospite ternario (PIENO-CARDAN) nello studente E2E corretto (cure P1:
    legge esperti a due piani e densi a 2/3/4 piani, `controlla_copertura` fail-closed), coi densi degli strati gia'
    trattati SOSTITUITI dai nuovi.
Per ogni strato L, in ordine (true-sequential dentro lo strato, come GPTQModel/QEP e `true_sequential_ratio 1.0` del
codice ufficiale di G²PTQ): stadio A ingressi del miscelatore (attn_qkv+attn_gate | attn_q/k/v), B uscita del miscelatore
(ssm_out | attn_output), C esperto condiviso gate+up, D esperto condiviso down.

SchurQuant (§3.5, App. D): X dal flusso quantizzato, bersaglio dal flusso pulito: per le proiezioni d'USCITA il
«residual-add target» (LoaQ): l'uscita pulita del sotto-strato col residuo (h + f(h)) meno il residuo del flusso
quantizzato (per ffn_down_shexp anche meno gli esperti instradati, diviso la porta sigmoide, peso porta²: forma esatta
del termine porta·W x); per le altre proiezioni l'uscita locale pulita W_ref X_ref. Pesi per token dalla decisione del
maestro (Eq. 18-19, ρ = 8, ε = 1e-8): prima di OGNI strato un passaggio completo del modello col prefisso quantizzato
(strati < L quantizzati, il resto pulito) confronta l'argmax con quello del modello pulito (calcolato una volta).
λ = 1, R = 16, g = 128 (TQ1_0R_1024: una scala ogni 128 colonne; TQ1_0: 256), smorzamento 1 %.
G²PTQ (codice ufficiale): per ogni stadio gradiente della perdita del blocco (MSE contro l'uscita pulita del blocco,
KL top-256 sull'ultimo) col blocco in precisione piena sul flusso quantizzato, salienza con bersaglio campionato,
Hessiana a 4 gruppi di righe, α per stadio su [0, 3] passo 0,1 con arresto (64 campioni), poi rinfresco del flusso.

Varianti: FEDELE = formato del lavoro (SchurQuant: affine asimmetrico 2 bit g = 128; G²PTQ: simmetrico 2 bit per
canale con clipping MSE, act_order), salvato come F16 per la misura (byte in piu', dichiarato); THOR = stesso metodo col
formato ternario a K piani dell'ospite (stessi byte: si riscrivono i piani AL POSTO con patch_gguf).
Ambito dichiarato: i 250 densi (gli esperti restano quelli dell'ospite); il maestro/flusso pulito e' bf16.
Ripresa: dopo ogni strato si salvano i pesi nuovi (`strato_LL.pt`); alla ripartenza i flussi si ricostruiscono
ripassando gli strati gia' fatti (nessun flusso salvato su disco)."""
from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import schurquant_paper as SQ  # noqa: E402
import g2ptq_paper as GP  # noqa: E402

HF_DIR = Path(os.getenv("THOR_HF_DIR", "@@ROOT:models@@/gguf/thor-src-hf"))
STUDENTE_DIR = Path("@@ROOT:models@@/gguf/tony-tern/CARDAN/codice/recupero_7dabd6f/thor_v2")
RAFFINA = Path("@@ROOT:models@@/build/leve/raffina")
CICLI = Path("@@ROOT:forge@@/stato/cicli.tsv")

HFMOD = {"attn_qkv": "linear_attn.in_proj_qkv", "attn_gate": "linear_attn.in_proj_z", "ssm_out": "linear_attn.out_proj",
         "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj",
         "attn_output": "self_attn.o_proj", "ffn_gate_shexp": "mlp.shared_expert.gate_proj",
         "ffn_up_shexp": "mlp.shared_expert.up_proj", "ffn_down_shexp": "mlp.shared_expert.down_proj"}
STADI = {"linear_attention": [["attn_qkv", "attn_gate"], ["ssm_out"], ["ffn_gate_shexp", "ffn_up_shexp"],
                              ["ffn_down_shexp"]],
         "full_attention": [["attn_q", "attn_k", "attn_v"], ["attn_output"], ["ffn_gate_shexp", "ffn_up_shexp"],
                            ["ffn_down_shexp"]]}
USCITA = {"ssm_out": "attn", "attn_output": "attn", "ffn_down_shexp": "mlp"}


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def riga_cicli(stato: str, testo: str) -> None:
    try:
        with open(CICLI, "a") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M')}\t{stato}\t{testo}\n")
    except OSError:
        pass


def modulo(strato, nome: str):
    m = strato
    for p in nome.split("."):
        m = getattr(m, p)
    return m


# ═══════════════════════════════ flussi (interfacce) ═══════════════════════════════
class Flusso:
    """Interfaccia comune: embed(ids) → h; avanti(L, h, ctx) → h; strato(L) → modulo HF (con hook); argmax(h) → id."""
    tipi: List[str]

    def contesto(self, B: int, T: int, dtype):
        raise NotImplementedError


# ─────────────────────────────── flusso PULITO bf16 (HF) ───────────────────────────────
class LettoreST:
    """Lettura a pezzi dei safetensors HF (pread esatto + fadvise DONTNEED)."""

    def __init__(self, cartella: Path):
        self.dir = cartella
        self.wmap = json.load(open(cartella / "model.safetensors.index.json"))["weight_map"]
        self._h = {}

    def _header(self, fname):
        if fname not in self._h:
            with open(self.dir / fname, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                self._h[fname] = (8 + n, json.loads(f.read(n)))
        return self._h[fname]

    def leggi(self, nome: str) -> torch.Tensor:
        fname = self.wmap[nome]
        base, h = self._header(fname)
        e = h[nome]
        a, b = e["data_offsets"]
        fd = os.open(self.dir / fname, os.O_RDONLY)
        try:
            buf = bytearray(b - a)
            mv = memoryview(buf)
            letti = 0
            while letti < len(buf):
                n = os.preadv(fd, [mv[letti:]], base + a + letti)
                assert n > 0
                letti += n
            os.posix_fadvise(fd, base + a, b - a, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
        dt = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16}[e["dtype"]]
        return torch.frombuffer(buf, dtype=dt).reshape(e["shape"]).clone()


class PulitoHF:
    """Modello pulito bf16 strato per strato. Uno scheletro per tipo di strato, riempito con copy_ (niente
    riallocazioni); embedding, norma finale e testa caricati una volta."""

    def __init__(self, cfg, device, cartella: Path = HF_DIR):
        from accelerate import init_empty_weights
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeDecoderLayer
        self.cfg = cfg
        self.dev = torch.device(device)
        self.st = LettoreST(cartella)
        self.pref = "model.language_model.layers."
        self.sch = {}
        for L, t in enumerate(cfg.layer_types):
            if t not in self.sch:
                with init_empty_weights():
                    s = Qwen3_5MoeDecoderLayer(cfg, layer_idx=L)
                self.sch[t] = (L, s)
        self.caricato = {t: None for t in self.sch}
        self.par = {}
        self.emb = self.st.leggi("model.language_model.embed_tokens.weight").to(self.dev)
        self.norm_w = self.st.leggi("model.language_model.norm.weight").to(self.dev).float()
        self.head = self.st.leggi("lm_head.weight").to(self.dev)
        self.eps = float(cfg.rms_norm_eps)
        self.tipi = list(cfg.layer_types)
        self.dt = torch.bfloat16

    def strato(self, L: int):
        t = self.tipi[L]
        L0, s = self.sch[t]
        if self.caricato[t] != L:
            sd = {}
            pre = f"{self.pref}{L}."
            for k in self.st.wmap:
                if k.startswith(pre):
                    sd[k[len(pre):]] = self.st.leggi(k).to(self.dev)
            if self.caricato[t] is None:
                s.load_state_dict(sd, strict=True, assign=True)
                s.eval()
                for p in s.parameters():
                    p.requires_grad_(False)
                self.par[t] = dict(s.named_parameters())
            else:
                with torch.no_grad():
                    for k, v in sd.items():
                        self.par[t][k].copy_(v)
            for m in s.modules():                     # indice di strato per la cache (non usata: cache None)
                if hasattr(m, "layer_idx"):
                    m.layer_idx = L
            self.caricato[t] = L
        return s

    def peso(self, L: int, fam: str) -> torch.Tensor:
        """W_ref HF (bf16 esatto, in fp64) di un denso."""
        return modulo(self.strato(L), HFMOD[fam]).weight.detach().to(torch.float64)

    def embed(self, ids):
        return self.emb[ids.to(self.dev)]

    def avanti(self, L, h, ctx):
        s = self.strato(L)
        t = self.tipi[L]
        o = s(h, position_embeddings=ctx["pe"], attention_mask=ctx["maschera"] if t == "full_attention" else None,
              position_ids=ctx["testo"])
        return o[0] if isinstance(o, tuple) else o

    def argmax(self, h, passo: int = 1024) -> torch.Tensor:
        x = h.reshape(-1, h.shape[-1]).float()
        out = []
        for i in range(0, x.shape[0], passo):
            xi = x[i:i + passo]
            xi = xi * torch.rsqrt(xi.pow(2).mean(-1, keepdim=True) + self.eps) * (1.0 + self.norm_w)
            out.append((xi.to(self.head.dtype) @ self.head.T).argmax(-1))
        return torch.cat(out)

    def logit(self, h) -> torch.Tensor:
        x = h.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * (1.0 + self.norm_w)
        return (x.to(self.head.dtype) @ self.head.T).float()


# ─────────────────────────────── flusso QUANTIZZATO (studente E2E) ───────────────────────────────
class QuantStudente:
    """Lo studente E2E corretto (cure P1) sull'ospite GGUF; i densi si sostituiscono nei moduli HF dello strato."""

    def __init__(self, gguf: Path, device, strati=range(40)):
        sys.path.insert(1, str(STUDENTE_DIR))
        import e2e_studente as S
        self.S = S
        self.st = S.Studente(gguf=Path(gguf), strati=strati, ricetta="B", motore_nome="triton", device=device)
        self.cfg = self.st.cfg
        self.tipi = list(self.cfg.layer_types)
        self.dt = self.st.ric.dtype_residuo

    def strato(self, L):
        return self.st.moduli[L]

    def embed(self, ids):
        return self.st.embedding(ids)

    def avanti(self, L, h, ctx):
        return self.st._strato(L, h, ctx)

    def contesto(self, B, T, dtype=None):
        return self.st._contesto(B, T, dtype or self.dt)

    def imposta(self, L: int, fam: str, W_hf: torch.Tensor) -> None:
        m = modulo(self.st.moduli[L], HFMOD[fam])
        with torch.no_grad():
            m.weight.copy_(W_hf.to(device=m.weight.device, dtype=m.weight.dtype))


# ═══════════════════════════════ basi: HF ↔ GGUF memorizzata ═══════════════════════════════
class Basi:
    """GGUF = HF[perm] (righe per attn_qkv/attn_gate, colonne per ssm_out), poi rotazione W' = W Rᵀ per TQ1_0R_1024."""

    def __init__(self, ruotato: Dict[str, bool]):
        sys.path.insert(1, str(STUDENTE_DIR))
        import carica_strato_gguf as CSG
        sys.path.insert(0, str(RAFFINA))
        import comune as CM
        self.CSG, self.CM = CSG, CM
        self.ruotato = ruotato                         # nome GGUF del tensore → ruotato?

    def voce(self, L, tipo, fam):
        return self.CSG.MAPPA["full" if tipo == "full_attention" else "linear"][fam + ".weight"]

    def a_memorizzata(self, L, tipo, fam, G, C, Wref, ruota: bool):
        v = self.voce(L, tipo, fam)
        if v["rel"] == "perm_righe":
            p = torch.from_numpy(self.CSG.PERM[v["perm_npz"]]).long().to(C.device)
            C, Wref = C[p], Wref[p]
        elif v["rel"] == "perm_colonne":
            p = torch.from_numpy(self.CSG.PERM[v["perm_npz"]]).long().to(C.device)
            G, C, Wref = G[p][:, p], C[:, p], Wref[:, p]
        else:
            assert v["rel"] == "identita", v
        if ruota:
            CM = self.CM
            G = CM.ruota_H(G.cpu()).to(G.device)
            C = CM.ruota(C.cpu()).to(G.device)
            Wref = CM.ruota(Wref.cpu()).to(G.device)
        return G, C, Wref

    def a_hf(self, L, tipo, fam, Wm, ruota: bool):
        W = self.CM.deruota(Wm.cpu()).to(Wm.device) if ruota else Wm
        gg = W.clone()                                  # ordine GGUF non ruotato (per la scrittura F16)
        v = self.voce(L, tipo, fam)
        if v["rel"] in ("perm_righe", "perm_colonne"):
            p = torch.from_numpy(self.CSG.PERM[v["perm_npz"]]).long().to(W.device)
            inv = torch.empty_like(p)
            inv[p] = torch.arange(len(p), device=W.device)
            W = W[inv] if v["rel"] == "perm_righe" else W[:, inv]
        return W, gg


# ═══════════════════════════════ cattura con hook ═══════════════════════════════
class Cattura:
    """Hook sullo strato: ingressi dei densi richiesti, ingresso di post_attention_layernorm (residuo dopo il
    miscelatore), uscita dell'esperto condiviso, logit della porta, uscita del blocco MoE."""

    def __init__(self, strato, fams: List[str]):
        self.d = {}
        self.h = []
        for f in fams:
            self.h.append(modulo(strato, HFMOD[f]).register_forward_hook(self._ing(f)))
        self.h.append(strato.post_attention_layernorm.register_forward_pre_hook(
            lambda m, a: self.d.__setitem__("_medio", a[0].detach())))
        mlp = strato.mlp
        self.h.append(mlp.shared_expert.register_forward_hook(lambda m, a, o: self.d.__setitem__("_cond", o.detach())))
        self.h.append(mlp.shared_expert_gate.register_forward_hook(
            lambda m, a, o: self.d.__setitem__("_porta", o.detach())))
        self.h.append(mlp.register_forward_hook(
            lambda m, a, o: self.d.__setitem__("_moe", (o[0] if isinstance(o, tuple) else o).detach())))

    def _ing(self, f):
        def h(m, a, o):
            self.d[f] = a[0].detach()
        return h

    def chiudi(self):
        for h in self.h:
            h.remove()


# ═══════════════════════════════ motore ═══════════════════════════════
class Motore:
    def __init__(self, pulito, quant, basi: Optional[Basi], ottimizza: Callable, ids: torch.Tensor, out: Path,
                 B: int = 4, device="cuda"):
        self.P, self.Q, self.basi, self.ott = pulito, quant, basi, ottimizza
        self.ids = ids
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.B = B
        self.dev = torch.device(device)
        self.N, self.T = ids.shape
        self.tipi = quant.tipi
        self.ctxq = quant.contesto(B, self.T, quant.dt)
        self.ctxp = quant.contesto(B, self.T, pulito.dt)

    def lotti(self):
        for i in range(0, self.N, self.B):
            yield slice(i, min(i + self.B, self.N))

    def _ctx(self, sl, q: bool):
        n = sl.stop - sl.start
        if n == self.B:
            return self.ctxq if q else self.ctxp
        return self.Q.contesto(n, self.T, self.Q.dt if q else self.P.dt)

    @torch.no_grad()
    def flussi_iniziali(self):
        hp = torch.empty(self.N, self.T, self.P.emb.shape[1], dtype=self.P.dt, device=self.dev)
        hq = torch.empty(self.N, self.T, self.P.emb.shape[1], dtype=self.Q.dt, device=self.dev)
        for sl in self.lotti():
            hp[sl] = self.P.embed(self.ids[sl])
            hq[sl] = self.Q.embed(self.ids[sl]).to(self.Q.dt)
        return hp, hq

    @torch.no_grad()
    def passa(self, flusso, L, h, q: bool):
        for sl in self.lotti():
            h[sl] = flusso.avanti(L, h[sl], self._ctx(sl, q)).to(h.dtype)

    @torch.no_grad()
    def argmax_da(self, h, L0: int) -> torch.Tensor:
        """Argmax del modello col prefisso dato (h all'ingresso dello strato L0) e gli strati L0..39 PULITI."""
        x = h.to(self.P.dt).clone()
        for L in range(L0, len(self.tipi)):
            self.passa(self.P, L, x, q=False)
        a = torch.empty(self.N * self.T, dtype=torch.long, device=self.dev)
        for sl in self.lotti():
            a[sl.start * self.T:sl.stop * self.T] = self.P.argmax(x[sl])
        del x
        return a

    def salva_strato(self, L, nuovi: Dict[str, torch.Tensor], info: dict):
        torch.save({"pesi_hf": {k: v.cpu() for k, v in nuovi.items()}, "info": info}, self.out / f"strato_{L:02d}.pt")

    def fatti(self) -> List[int]:
        return sorted(int(p.stem.split("_")[1]) for p in self.out.glob("strato_*.pt"))

    @torch.no_grad()
    def riprendi(self, hp, hq) -> int:
        """Ripassa gli strati gia' fatti (pesi nuovi riapplicati) e ritorna il primo da fare."""
        fatti = self.fatti()
        L = 0
        while L in fatti:
            d = torch.load(self.out / f"strato_{L:02d}.pt")
            for fam, W in d["pesi_hf"].items():
                self.Q.imposta(L, fam, W.to(self.dev))
            self.passa(self.P, L, hp, q=False)
            self.passa(self.Q, L, hq, q=True)
            log(f"ripresa: strato {L} riapplicato")
            L += 1
        return L

    @torch.no_grad()
    def cancello_imposta(self, hq):
        """Fail-closed: cambiare il peso di un denso deve cambiare l'uscita dello strato, e rimetterlo deve ridare
        l'uscita di prima AL BIT (nessuna cache/piega dello studente che scavalchi i moduli HF)."""
        L, f = 0, STADI[self.tipi[0]][0][0]
        sl = slice(0, min(self.B, self.N))
        m = modulo(self.Q.strato(L), HFMOD[f])
        W0 = m.weight.detach().clone()
        o0 = self.Q.avanti(L, hq[sl], self._ctx(sl, True)).clone()
        self.Q.imposta(L, f, W0 * 0.5)
        o1 = self.Q.avanti(L, hq[sl], self._ctx(sl, True)).clone()
        self.Q.imposta(L, f, W0)
        o2 = self.Q.avanti(L, hq[sl], self._ctx(sl, True))
        if torch.equal(o0, o1) or not torch.equal(o0, o2):
            raise RuntimeError("cancello_imposta: la sostituzione dei densi non agisce sul flusso quantizzato")
        log("cancello_imposta superato: la sostituzione dei densi agisce e si annulla al bit")

    def eta(self, metodo: str, L: int, sec: float, fatti_ora: int, totale: int):
        resto = (totale - fatti_ora) * sec
        testo = (f"FEDELI {metodo}: strato {L} in {sec / 60:.1f} min ({fatti_ora}/{totale}); fine stimata "
                 f"{time.strftime('%d/%m %H:%M', time.localtime(time.time() + resto))}")
        log(testo)
        if L in (0, 1) or L % 10 == 9:
            riga_cicli("ATTIVO", testo)

    # ─────────────────────────── SchurQuant ───────────────────────────
    def schurquant(self, lam: float = 1.0, rho: float = 8.0, eps: float = 1e-8, strati=None):
        hp, hq = self.flussi_iniziali()
        t0 = time.time()
        tau = self.argmax_da(hp, 0)
        log(f"argmax del modello pulito su {tau.numel()} token in {time.time() - t0:.0f} s")
        L0 = self.riprendi(hp, hq)
        self.cancello_imposta(hq)
        strati = range(L0, len(self.tipi)) if strati is None else [L for L in strati if L >= L0]
        for L in strati:
            tl = time.time()
            argq = self.argmax_da(hq, L)
            abar = SQ.pesi_decisione(argq, tau, rho).to(self.dev)
            girati = float((argq != tau).float().mean())
            tipo = self.tipi[L]
            nuovi, info = {}, {"girati": girati, "tensori": {}}
            for stadio in STADI[tipo]:
                stat = {}
                for f in stadio:
                    W = self.P.peso(L, f)
                    stat[f] = (SQ.Statistiche(W.shape[1], W.shape[0], self.dev), W)
                cp = Cattura(self.P.strato(L), stadio)
                cq = Cattura(self.Q.strato(L), stadio)
                try:
                    for sl in self.lotti():
                        cp.d.clear(); cq.d.clear()
                        outp = self.P.avanti(L, hp[sl], self._ctx(sl, False))
                        self.Q.avanti(L, hq[sl], self._ctx(sl, True))
                        ab = abar[sl.start * self.T:sl.stop * self.T]
                        for f in stadio:
                            S_, W = stat[f]
                            x = cq.d[f].reshape(-1, W.shape[1]).double()
                            if USCITA.get(f) == "attn":
                                y = (cp.d["_medio"].double() - hq[sl].double()).reshape(-1, W.shape[0])
                                om = SQ.pesi_proiezione(ab, W, x, y, eps)
                            elif USCITA.get(f) == "mlp":
                                sg = torch.sigmoid(cq.d["_porta"].double()).reshape(-1, 1)
                                medio = cq.d["_medio"].double().reshape(-1, W.shape[0])
                                instr = cq.d["_moe"].double().reshape(-1, W.shape[0]) - sg * cq.d["_cond"].double().reshape(-1, W.shape[0])
                                t = outp.double().reshape(-1, W.shape[0]) - medio - instr
                                y = t / sg
                                r = sg.squeeze(1) * torch.linalg.vector_norm(x @ W.T - y, dim=1)
                                om = ab.double() / r.clamp_min(eps) * sg.squeeze(1) ** 2
                            else:
                                xr = cp.d[f].reshape(-1, W.shape[1]).double()
                                y = xr @ W.T
                                om = SQ.pesi_proiezione(ab, W, x, y, eps)
                            S_.aggiungi(x, y, om)
                finally:
                    cp.chiudi(); cq.chiudi()
                for f in stadio:
                    S_, W = stat[f]
                    G, C = S_.finale(lam, W)
                    Wn_hf, rec = self.ott(L, tipo, f, G, C, W)
                    self.Q.imposta(L, f, Wn_hf)
                    nuovi[f] = Wn_hf.float()
                    info["tensori"][f] = rec
                    del G, C
            self.passa(self.P, L, hp, q=False)
            self.passa(self.Q, L, hq, q=True)
            info["secondi"] = time.time() - tl
            self.salva_strato(L, nuovi, info)
            log(f"SchurQuant strato {L} ({tipo}) in {info['secondi']:.0f} s, top-1 girati {100 * girati:.2f} %: "
                + ", ".join(f"{f} {r.get('rel', 0):.4f}" for f, r in info["tensori"].items()))
            self.eta("schurquant", L, info["secondi"], len(self.fatti()), len(self.tipi))
        return self.fatti()

    # ─────────────────────────── G²PTQ ───────────────────────────
    def _grad_uscita(self, out, ref, ultimo: bool, kl_topk: int, H: int, passo: int = 1024):
        """∂/∂uscita del blocco delle due perdite del codice ufficiale: salienza (bersaglio CAMPIONATO: rumore N(0, d)
        per la MSE, etichette dal maestro top-k per la NLL dell'ultimo blocco) e perdita vera (MSE contro l'uscita
        pulita del blocco; KL top-k sull'ultimo). Sull'ultimo i logit si fanno a pezzi di `passo` token."""
        o = out.reshape(-1, H)
        r = ref.reshape(-1, H).to(o.dtype)
        if not ultimo:
            rum = torch.distributions.Normal(torch.zeros_like(o), math.sqrt(H)).sample()
            # d/do [ Σ_t mean_j (o − (o+rum))² / 2 ] con bersaglio staccato = −rum / H ; idem per la MSE vera
            g_sal = (-rum) / H
            g_blk = (o - r) / H
            return g_sal.reshape(out.shape), g_blk.reshape(out.shape)
        g_sal = torch.empty_like(o)
        g_blk = torch.empty_like(o)
        for i in range(0, o.shape[0], passo):
            oi = o[i:i + passo].detach().float().requires_grad_(True)
            with torch.enable_grad():
                lq = self.P.logit(oi)
                with torch.no_grad():
                    lf = self.P.logit(r[i:i + passo].float())
                    lf, idx = lf.topk(kl_topk, dim=-1, sorted=False)
                    lab = torch.distributions.Categorical(logits=lf).sample()
                lqk = lq.gather(-1, idx)
                l_sal = F.cross_entropy(lqk, lab, reduction="sum")
                g_sal[i:i + passo] = torch.autograd.grad(l_sal, oi, retain_graph=True)[0].to(o.dtype)
                l_blk = F.kl_div(F.log_softmax(lqk, -1), F.softmax(lf, -1), reduction="none").sum()
                g_blk[i:i + passo] = torch.autograd.grad(l_blk, oi)[0].to(o.dtype)
        return g_sal.reshape(out.shape), g_blk.reshape(out.shape)

    def g2ptq(self, ottimizza_g2: Callable, num_groups: int = 4, autotune_n: int = 64, amin: float = 0.0,
              amax: float = 3.0, apasso: float = 0.1, kl_topk: int = 256, outlier: float = 1.0, strati=None,
              seme: int = 42):
        torch.manual_seed(seme)
        hp, hq = self.flussi_iniziali()
        L0 = self.riprendi(hp, hq)
        self.cancello_imposta(hq)
        n_strati = len(self.tipi)
        strati = range(L0, n_strati) if strati is None else [L for L in strati if L >= L0]
        H = hp.shape[-1]
        for L in strati:
            torch.manual_seed(seme * 1000 + L)          # campionamenti per strato: la ripresa resta identica al bit
            tl = time.time()
            tipo = self.tipi[L]
            ultimo = L == n_strati - 1
            # uscite pulite del blocco (fp_inps ← blocco_fp(fp_inps), come il codice ufficiale)
            self.passa(self.P, L, hp, q=False)
            nuovi, info = {}, {"tensori": {}}
            strato_q = self.Q.strato(L)
            for si, stadio in enumerate(STADI[tipo]):
                mods = {f: modulo(strato_q, HFMOD[f]) for f in stadio}
                Wref = {f: self.P.peso(L, f).float() for f in stadio}
                for f in stadio:                                   # il blocco parte in PRECISIONE PIENA (W_ref)
                    self.Q.imposta(L, f, Wref[f])
                sal = {f: [] for f in stadio}
                grd = {f: GP.GradienteMedio() for f in stadio}
                # (a) gradiente e salienza, lotto per lotto
                for sl in self.lotti():
                    for f in stadio:
                        mods[f].weight.requires_grad_(True)
                    usc = {}
                    hk = [mods[f].register_forward_hook((lambda ff: (lambda m, a, o: usc.__setitem__(ff, o)))(f))
                          for f in stadio]
                    try:
                        with torch.enable_grad():
                            out = self.Q.avanti(L, hq[sl], self._ctx(sl, True))
                            go_sal, go_blk = self._grad_uscita(out.detach(), hp[sl], ultimo, kl_topk, H)
                            gy = torch.autograd.grad(out, [usc[f] for f in stadio], grad_outputs=go_sal,
                                                     retain_graph=True)
                            for f, g in zip(stadio, gy):
                                sal[f].append(GP.salienza_da_grad_uscita(g, num_groups))
                            if not (L == 0 and si == 0):
                                gw = torch.autograd.grad(out, [mods[f].weight for f in stadio], grad_outputs=go_blk)
                                ntok = (sl.stop - sl.start) * self.T
                                for f, g in zip(stadio, gw):
                                    grd[f].aggiungi(g, ntok)
                            del out
                    finally:
                        for h in hk:
                            h.remove()
                        for f in stadio:
                            mods[f].weight.requires_grad_(False)
                # (b) Hessiana a gruppi (stessa sequenza di lotti, salienza consumata in ordine)
                hes = {f: GP.HessianaGruppi(mods[f].weight.shape[1], num_groups, self.dev) for f in stadio}
                xtx = {f: torch.zeros(mods[f].weight.shape[1], mods[f].weight.shape[1], device=self.dev)
                       for f in stadio}
                k = 0
                cq = Cattura(strato_q, stadio)
                try:
                    with torch.no_grad():
                        for sl in self.lotti():
                            cq.d.clear()
                            self.Q.avanti(L, hq[sl], self._ctx(sl, True))
                            for f in stadio:
                                x = cq.d[f].reshape(-1, cq.d[f].shape[-1]).float()
                                hes[f].aggiungi(x, sal[f][k])
                                xtx[f] += x.T @ x
                            k += 1
                finally:
                    cq.chiudi()
                del sal
                # (c) α per stadio (Alg. 3), non sul primo strato
                def perdita_blocco():
                    tot, n = 0.0, 0
                    with torch.no_grad():
                        for sl in self.lotti():
                            if sl.start >= autotune_n:
                                break
                            out = self.Q.avanti(L, hq[sl], self._ctx(sl, True)).float()
                            ref = hp[sl].float()
                            if ultimo:
                                lq = self.P.logit(out).reshape(-1, self.P.head.shape[0])
                                lf = self.P.logit(ref).reshape(-1, self.P.head.shape[0])
                                lf, idx = lf.topk(kl_topk, dim=-1, sorted=False)
                                lq = lq.gather(-1, idx)
                                v = F.kl_div(F.log_softmax(lq, -1), F.softmax(lf, -1), reduction="none").sum()
                            else:
                                o, r = out.clone(), ref.clone()
                                if outlier > 0:
                                    a = r.abs()
                                    msk = a > a.mean() + a.std() * outlier
                                    o[msk] = 0
                                    r[msk] = 0
                                v = F.mse_loss(o.reshape(-1, H), r.reshape(-1, H), reduction="none").mean(-1).sum() / 2
                            tot += float(v) / ((sl.stop - sl.start) * self.T)
                            n += 1
                    return tot / max(n, 1)

                def quantizza(alpha):
                    res = {}
                    for f in stadio:
                        g = grd[f].g if grd[f].g is not None else None
                        Wn_hf, rec = ottimizza_g2(L, tipo, f, Wref[f], hes[f].H, g, alpha, xtx[f] / max(1, hes[f].total_tokens),
                                                  enable_grad=not (L == 0 and si == 0))
                        self.Q.imposta(L, f, Wn_hf)
                        res[f] = (Wn_hf, rec)
                    return res
                alpha_best = 0.0
                prove = []
                if L > 0:
                    best = float("inf")
                    for a in np.arange(amin, amax + 1e-3, apasso):
                        a = float(f"{a:.2f}")
                        quantizza(a)
                        v = perdita_blocco()
                        prove.append([a, v])
                        if v < best:
                            best, alpha_best = v, a
                        for f in stadio:
                            self.Q.imposta(L, f, Wref[f])
                        if a - alpha_best >= apasso:
                            break
                res = quantizza(alpha_best)
                for f, (Wn_hf, rec) in res.items():
                    rec = dict(rec, alpha=alpha_best, prove_alpha=prove)
                    nuovi[f] = Wn_hf.float()
                    info["tensori"][f] = rec
            # (d) rinfresco del flusso quantizzato
            self.passa(self.Q, L, hq, q=True)
            info["secondi"] = time.time() - tl
            self.salva_strato(L, nuovi, info)
            log(f"G2PTQ strato {L} ({tipo}) in {info['secondi']:.0f} s: "
                + ", ".join(f"{f} α={r['alpha']}" for f, r in info["tensori"].items()))
            self.eta("g2ptq", L, info["secondi"], len(self.fatti()), n_strati)
        return self.fatti()


# ═══════════════════════════════ ottimizzatori per tensore (varianti) ═══════════════════════════════
class OttimizzaThor:
    """Variante THOR: formato dell'ospite (K piani, scale per gruppo), avvio dall'ospite; salva i byte dei piani per
    patch_gguf e restituisce W HF per il flusso quantizzato."""

    def __init__(self, gguf: str, basi: Basi, out: Path, R: int = 16, damp: float = 0.01, avvio: str = "ospite"):
        os.environ["LEVE_OSPITE"] = str(gguf)
        CM = basi.CM
        self.CM = CM
        self.src = CM.gp.GGUFPlaneSource(str(gguf))
        self.basi, self.R, self.damp, self.avvio = basi, R, damp, avvio
        self.td = Path(out) / "tensori"
        self.td.mkdir(parents=True, exist_ok=True)

    def __call__(self, L, tipo, fam, G, C, Wref):
        nome = f"blk.{L}.{fam}.weight"
        d = self.CM.Denso(self.src, nome)
        Gm, Cm, Wm = self.basi.a_memorizzata(L, tipo, fam, G, C, Wref, ruota=d.rotated)
        ng = d.S.shape[-1]
        g = d.in_ // ng
        T0 = d.T.to(Gm.device) if self.avvio == "ospite" else None
        S0 = d.S.to(Gm.device) if self.avvio == "ospite" else None
        Gd = SQ.smorza(Gm, self.damp)
        f_osp = float(SQ.obiettivo(SQ.ricostruisci(d.T.to(Gm.device), d.S.to(Gm.device)), Gd, Cm))
        r = SQ.schuropt_ternario(Gm, Cm, Wm, d.K, g=g, R=self.R, damp=self.damp, T0=T0, S0=S0)
        f_new = float(SQ.obiettivo(r["W"], Gd, Cm))
        T, S = r["T"].cpu(), r["S"].float().cpu()
        if f_new > f_osp:                                    # mai peggio dell'ospite sull'obiettivo del lavoro
            T, S = d.T.clone(), d.S.clone()
            rec_w = SQ.ricostruisci(T, S).to(Gm.device)
        else:
            rec_w = r["W"]
        for nm, bb in zip(d.nomi_piani, d.byte_piani(T, S)):
            np.save(self.td / f"{nm}.npy", bb)
        W_hf, _ = self.basi.a_hf(L, tipo, fam, rec_w, ruota=d.rotated)
        return W_hf, {"K": d.K, "g": g, "obiettivo_ospite": f_osp, "obiettivo_nuovo": f_new,
                      "rel": (f_osp - f_new) / max(abs(f_osp), 1e-300), "tenuto_ospite": f_new > f_osp}


class OttimizzaFedele:
    """Variante FEDELE di SchurQuant: affine asimmetrico 2 bit g = 128 nell'ordine GGUF NON ruotato (formato del lavoro);
    salva W (fp32, ordine GGUF) per lo scrittore F16."""

    def __init__(self, basi: Basi, out: Path, bits: int = 2, g: int = 128, R: int = 16, damp: float = 0.01):
        self.basi, self.bits, self.g, self.R, self.damp = basi, bits, g, R, damp
        self.td = Path(out) / "f16"
        self.td.mkdir(parents=True, exist_ok=True)

    def __call__(self, L, tipo, fam, G, C, Wref):
        Gm, Cm, Wm = self.basi.a_memorizzata(L, tipo, fam, G, C, Wref, ruota=False)
        r = SQ.schuropt_affine(Gm, Cm, Wm, self.bits, g=self.g, R=self.R, damp=self.damp)
        np.save(self.td / f"blk.{L}.{fam}.weight.npy", r["W"].float().cpu().numpy())
        W_hf, _ = self.basi.a_hf(L, tipo, fam, r["W"], ruota=False)
        Gd = SQ.smorza(Gm, self.damp)
        f = float(SQ.obiettivo(r["W"], Gd, Cm))
        return W_hf, {"bits": self.bits, "g": self.g, "obiettivo": f, "rel": f}


class OttimizzaG2:
    """G²PTQ per tensore: FEDELE (simmetrico 2 bit per canale, clipping MSE, act_order, ordine GGUF non ruotato) o THOR
    (formato dell'ospite nella base memorizzata: W', H' = R H Rᵀ per gruppo, gradiente g' = g Rᵀ, act_square' =
    diag(R XᵀX Rᵀ)/n)."""

    def __init__(self, variante: str, basi: Basi, out: Path, gguf: Optional[str] = None, bits: int = 2):
        self.v, self.basi, self.bits = variante, basi, bits
        self.out = Path(out)
        if variante == "thor":
            os.environ["LEVE_OSPITE"] = str(gguf)
            self.src = basi.CM.gp.GGUFPlaneSource(str(gguf))
            (self.out / "tensori").mkdir(parents=True, exist_ok=True)
        else:
            (self.out / "f16").mkdir(parents=True, exist_ok=True)

    def __call__(self, L, tipo, fam, Wref_hf, Hg, grad_hf, alpha, xtx_media, enable_grad=True):
        CM = self.basi.CM
        v = self.basi.voce(L, tipo, fam)
        dev = Wref_hf.device
        perm = None
        if v["rel"] in ("perm_righe", "perm_colonne"):
            perm = torch.from_numpy(self.basi.CSG.PERM[v["perm_npz"]]).long().to(dev)
        W = Wref_hf.float()
        g = grad_hf.float() if grad_hf is not None else None
        Hs, xa = Hg.float(), xtx_media.float()
        # righe: si resta nell'ordine HF (i 4 gruppi di righe della salienza sono contigui nell'uscita del modulo HF,
        # come nel codice ufficiale); la permutazione delle righe si applica DOPO a Q, codici e scale.
        if v["rel"] == "perm_colonne":
            W = W[:, perm]
            g = g[:, perm] if g is not None else None
            Hs = Hs[:, perm][:, :, perm]
            xa = xa[perm][:, perm]
        ruota = False
        if self.v == "thor":
            d = CM.Denso(self.src, f"blk.{L}.{fam}.weight")
            ruota = d.rotated
            if ruota:
                W = CM.ruota(W.double().cpu()).float().to(dev)
                g = CM.ruota(g.double().cpu()).float().to(dev) if g is not None else None
                Hs = torch.stack([CM.ruota_H(h.double().cpu()).float() for h in Hs]).to(dev)
                xa = CM.ruota_H(xa.double().cpu()).float().to(dev)
            gsz = d.in_ // d.S.shape[-1]
            r = GP.fasterquant(W, Hs, g, alpha, torch.diagonal(xa).clone(), quant="thor", K=d.K, groupsize=gsz,
                               enable_gradient_update=enable_grad)
            T, S, Wm = r["T"].cpu(), r["S"].float().cpu(), r["Q"].double()
            if v["rel"] == "perm_righe":
                pc = perm.cpu()
                T, S, Wm = T[:, pc], S[:, pc], Wm[perm]
            for nm, bb in zip(d.nomi_piani, d.byte_piani(T, S)):
                np.save(self.out / "tensori" / f"{nm}.npy", bb)
        else:
            r = GP.fasterquant(W, Hs, g, alpha, torch.diagonal(xa).clone(), quant="sym", bits=self.bits, groupsize=-1,
                               enable_gradient_update=enable_grad)
            Wm = r["Q"].double()
            if v["rel"] == "perm_righe":
                Wm = Wm[perm]
            np.save(self.out / "f16" / f"blk.{L}.{fam}.weight.npy", Wm.float().cpu().numpy())
        W_hf, _ = self.basi.a_hf(L, tipo, fam, Wm, ruota=ruota)
        return W_hf, {"beta_medio": float(r["beta"].mean()), "variante": self.v}


# ═══════════════════════════════ main ═══════════════════════════════
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("metodo", choices=["schurquant", "g2ptq"])
    ap.add_argument("variante", choices=["fedele", "thor"])
    ap.add_argument("--ospite", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lotto", type=int, default=4)
    ap.add_argument("--strati", default="")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    torch.backends.cuda.matmul.allow_tf32 = False
    dati = torch.load(a.corpus)
    ids = dati["ids"].long()
    log(f"corpus {a.corpus}: {tuple(ids.shape)} ({dati['meta']})")
    Q = QuantStudente(Path(a.ospite), a.device)
    P = PulitoHF(Q.cfg, a.device)
    basi = Basi({})
    out = Path(a.out)
    strati = [int(x) for x in a.strati.split(",")] if a.strati else None
    if a.metodo == "schurquant":
        ott = OttimizzaThor(a.ospite, basi, out) if a.variante == "thor" else OttimizzaFedele(basi, out)
        M = Motore(P, Q, basi, ott, ids, out, B=a.lotto, device=a.device)
        fatti = M.schurquant(strati=strati)
    else:
        ott = OttimizzaG2(a.variante, basi, out, gguf=a.ospite)
        M = Motore(P, Q, basi, None, ids, out, B=a.lotto, device=a.device)
        fatti = M.g2ptq(ott, strati=strati)
    log(f"strati completati: {fatti}")
    return 0 if len(fatti) == len(Q.tipi) or strati else 4


if __name__ == "__main__":
    sys.exit(main())
