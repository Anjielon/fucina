#!/usr/bin/env python3
"""S2+S3 tappa GPU: UNA passata SEQUENZIALE strato per strato sul flusso di C (piano casalingo §2 S2+S3, §1.2).

Due flussi sugli stessi token del corpus di S1 (meta' A costruisce, meta' B guarda; M manifesto, Kv wiki+edu, Kw WikiText-103):
  maestro  PulitoHF (sequenziale_thor.py: bf16 di Ornith-1.5 uno strato alla volta);
  C        lo studente E2E-C1/C3 (ac482f3c, densi ternari «tutte», ssm_alpha/beta ternari) sulla PARTENZA (uscita di S1b se
           accettata, altrimenti l'ospite di S1), coi tensori degli strati gia' trattati SOSTITUITI dai nuovi: gli ingressi di
           ogni strato sono RICATTURATI dopo aver quantizzato gli strati precedenti (differenza con la PTQ indipendente:
           GPTAQ 2504.02692 / QEP 2504.09629 «prefisso quantizzato»), e dentro lo strato dopo i tensori a monte gia' nuovi.
Per ogni strato L (ordine del piano §1.2):
  P1  ingresso del miscelatore (input_layernorm): attn_qkv, attn_gate, ssm_alpha, ssm_beta | attn_q, attn_k, attn_v -> S2
  P2  ingresso di ssm_out | attn_output (C ricalcolato coi tensori di P1 nuovi) -> S2 (+K1, +K2 piegata in attn_v)
  P3  ingresso del MoE (post_attention_layernorm): ffn_gate/up_shexp (S2) ed ESPERTI gate/up (S3, E1: Gq_e pesate dal router
      di C, termine incrociato col maestro)
  P4  ingresso VERO dei down (con gate/up nuovi): ffn_down_shexp (S2, +K1, +K2 piegata in up_shexp) ed esperti down (S3)
  P5  passaggio completo di C con lo strato nuovo -> ingresso dello strato L+1; deriva dello strato contro il maestro
Riduzione della GPU (senza togliere dati): (a) ogni passaggio parziale si FERMA al punto che serve (eccezione dal gancio:
niente MoE in P1-P3); (b) in modo «cache» il maestro passa UNA volta per strato (P1) e i suoi ingressi ai punti
att_out/ffn/down restano in VRAM bf16 per P2-P4 (~13 KB/token); senza cache il maestro si ferma anch'esso al punto;
(c) solutori sul dispositivo con le fattorizzazioni su CPU fp64 (s23_algebra); (d) statistiche degli esperti di una meta'
alla volta (scaricate su disco fp32, rilette a lotti).
Cancelli (fail-closed): (1) iniettare i byte della partenza come oggetti nuovi lascia l'uscita dello strato identica AL BIT,
cambiarli la cambia, rimetterli la ridà al bit; (2) top-8 ricalcolato = quello dello studente (P4); (3) byte scritti
riletti = (trit, scale) voluti (andata e ritorno per tensore e per esperto).
Ripresa: strato con FATTO saltato; i flussi si ricostruiscono ripassando gli strati fatti coi loro byte nuovi.
Uscite: <out>/strati/strato_LL/{tensori/<nome GGUF>.npy (byte del tensore intero, solo i cambiati), registro.json, FATTO},
<out>/passata.jsonl, <out>/FATTO_PASSATA.

CORRETTO 14/10/2026 (ricerche/C_S23_CORRETTO_1014.md; difetto in ricerche/C_S1B_AUDIT_1014.md §5): guardia indipendente
dall'obiettivo (errore d'uscita simmetrico verso il maestro sulla meta' B + energia d'uscita in [S23_ENERGIA_MIN,
S23_ENERGIA_MAX]) per densi ed esperti, candidati a in S23_ALFE (densi, default 0,0.5,1) e S23_ALFE_ESPERTI (esperti,
default 0,0.5) giudicati da essa, H corta S23_PESO_KW (variante Thor). Dettaglio nel docstring di s23_risolvi.py.

VELOCE2 14/10/2026 (ricerche/C_S23_VELOCE2_1014.md), stessi conti, stessi bit:
  V1 CODA (S23_CODA=1, default): P3 conserva in VRAM l'ingresso di post_attention_layernorm di C (residuo dopo il
     miscelatore, fp32, ~10,5 GiB); P4 e P5 ripartono da li' (post_attention_layernorm -> MoE -> residuo, la stessa
     sequenza di Qwen3_5MoeDecoderLayer.forward) invece di rifare input_layernorm + miscelatore + out_proj: niente a monte
     di quel punto cambia fra P3 e P5. Cancello (fail-safe) al primo lotto di P4: ganci della coda == ganci del passaggio
     completo AL BIT, altrimenti la coda si spegne e si torna al passaggio completo; al primo lotto di P5 uscita della coda
     == uscita di _strato AL BIT, altrimenti errore.
  V2 FLUSSI (S23_FLUSSI=1, default): a fine strato (dopo FATTO) i flussi hq (C, fp32) e hp (maestro, bf16) sono salvati in
     <out>/flussi (solo l'ultimo strato); la ripresa li ricarica invece di ripassare gli strati fatti, se il meta' combacia
     (strato, partenza, corpus, lotti, sha256 di registro.json e tensori/*.npy di ogni strato fatto); altrimenti ripassa.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

QUI = Path(__file__).resolve().parent
for p in (str(QUI), str(QUI.parent / "lib"), str(QUI.parent / "s1b"), str(QUI.parent / "rif")):
    if p not in sys.path:
        sys.path.insert(0, p)
import s23_algebra as AL  # noqa: E402
import s23_risolvi as RS  # noqa: E402
import s1b_comune as K  # noqa: E402
import s1_comune as SC  # noqa: E402

FAM_STADIO = {"linear_attention": {1: ("attn_qkv", "attn_gate", "ssm_alpha", "ssm_beta"), 2: ("ssm_out",),
                                   3: ("ffn_gate_shexp", "ffn_up_shexp"), 4: ("ffn_down_shexp",)},
              "full_attention": {1: ("attn_q", "attn_k", "attn_v"), 2: ("attn_output",),
                                 3: ("ffn_gate_shexp", "ffn_up_shexp"), 4: ("ffn_down_shexp",)}}
PUNTO = {1: "attn", 2: "att_out", 3: "ffn", 4: "down"}
TERNARI = (34, 45)
CICLI = None


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def riga_cicli(stato, testo):
    if CICLI:
        with open(CICLI, "a") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M')}\t{stato}\tC-S23: {testo}\n")


def elenco(s, n):
    if not s:
        return list(range(n))
    out = []
    for p in s.split(","):
        if "-" in p:
            a, b = p.split("-")
            out += list(range(int(a), int(b) + 1))
        elif p:
            out.append(int(p))
    return out


class Ferma(Exception):
    """lanciata da un gancio al punto che serve: il resto dello strato non si calcola."""


class Gancio:
    """Ganci ai punti d'ingresso (stessi moduli nel maestro HF e nello studente). `ferma` = punto dopo cui si interrompe.
    "a_ord" (solo studente) = ingresso del down degli esperti (toppa su FamigliaEsperti.apply, come s1_cattura)."""

    def __init__(self, strato, full, voglio, ferma=None, S=None, fam_down=None):
        self.d, self.h, self.ferma, self.S, self.orig = {}, [], ferma, S, None
        voglio = set(voglio) | ({ferma} if ferma else set())
        m_att = strato.self_attn.o_proj if full else strato.linear_attn.out_proj

        def metti(k, v):
            self.d[k] = v.detach()
            if self.ferma == k:
                raise Ferma(k)
        if "attn" in voglio:
            self.h.append(strato.input_layernorm.register_forward_hook(lambda m, i, o: metti("attn", o)))
        if "att_out" in voglio:
            self.h.append(m_att.register_forward_pre_hook(lambda m, i: metti("att_out", i[0])))
        if "hmid" in voglio:              # VELOCE2 V1: ingresso di post_attention_layernorm = residuo dopo il miscelatore
            self.h.append(strato.post_attention_layernorm.register_forward_pre_hook(lambda m, i: metti("hmid", i[0])))
        if "ffn" in voglio:
            self.h.append(strato.post_attention_layernorm.register_forward_hook(lambda m, i, o: metti("ffn", o)))
        if "down" in voglio:
            self.h.append(strato.mlp.shared_expert.down_proj.register_forward_pre_hook(lambda m, i: metti("down", i[0])))
        if "a_ord" in voglio:
            FE = S.FamigliaEsperti
            self.grezzo = FE.__dict__.get("apply")
            orig = FE.apply
            self.orig = orig
            d = self.d

            def apply(x, dd, fam, piano, *a, **k):
                if fam is fam_down:
                    d["piano"] = piano
                    metti("a_ord", x)
                return orig(x, dd, fam, piano, *a, **k)
            FE.apply = apply

    def chiudi(self):
        for h in self.h:
            h.remove()
        self.h = []
        if self.orig is not None:
            if self.grezzo is None:
                del self.S.FamigliaEsperti.apply
            else:
                self.S.FamigliaEsperti.apply = self.grezzo
            self.orig = None


def coda_strato(st, L, x):
    """VELOCE2 V1: la parte di Qwen3_5MoeDecoderLayer.forward (transformers 5.15) DOPO il miscelatore, dal residuo x
    (= ingresso di post_attention_layernorm): residual = x; h = post_attention_layernorm(x); h = mlp(h); residual + h.
    Stesso preambolo di Studente._strato (tensori f32 installati, mlp.instr None)."""
    modulo = st.moduli[L]
    for tf in st._f32_strato.get(L, ()):
        tf.installa()
    modulo.mlp.instr = None
    try:
        h = modulo.post_attention_layernorm(x)
        h = modulo.mlp(h)
        if isinstance(h, tuple):
            h, _ = h
        return x + h
    finally:
        modulo.mlp.instr = None


def esperti_senza_sync(self, hidden_states, top_k_index, top_k_weights):
    """VELOCE2 V3: Qwen3_5MoeExperts.forward (transformers 5.15, maestro bf16) con GLI STESSI prodotti, sugli stessi
    indici e nello stesso ordine, ma senza le due sincronizzazioni per esperto (torch.where + confronto di un tensore del
    dispositivo con un intero, 256 x 2 per lotto). nonzero() sulla maschera [E, k, T] da' le righe (e, pos, t) in ordine
    lessicografico = la concatenazione dei torch.where(expert_mask[e]) per e crescente; una sola lettura dei conteggi.
    Stessi ingressi per ogni F.linear e stesso ordine degli index_add_ -> stessi bit."""
    final_hidden_states = torch.zeros_like(hidden_states)
    with torch.no_grad():
        expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
        nz = expert_mask.nonzero()
        conta = torch.bincount(nz[:, 0], minlength=self.num_experts).tolist()
    o = 0
    for e in range(self.num_experts):
        c = conta[e]
        if c == 0:
            continue
        top_k_pos, token_idx = nz[o:o + c, 1], nz[o:o + c, 2]
        o += c
        current_state = hidden_states[token_idx]
        gate, up = F.linear(current_state, self.gate_up_proj[e]).chunk(2, dim=-1)
        current_hidden_states = self.act_fn(gate) * up
        current_hidden_states = F.linear(current_hidden_states, self.down_proj[e])
        current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
        final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
    return final_hidden_states


def sha_file(f) -> str:
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def impronta_strato(d: Path) -> dict:
    """VELOCE2 V2: sha256 di registro.json e di ogni tensori/*.npy di una cartella strato_LL chiusa."""
    return {"registro": sha_file(d / "registro.json"),
            "tensori": {f.name: sha_file(f) for f in sorted((d / "tensori").glob("*.npy"))}}


def scrivi_tensore(f, t, passo_byte=1 << 28):
    """VELOCE2 V2: tensore (anche del dispositivo) -> .npy a pezzi (bf16 come int16, bit per bit), fsync, page cache
    lasciata (la RAM host e' poca)."""
    if t.dtype == torch.bfloat16:
        t = t.view(torch.int16)
    mm = np.lib.format.open_memmap(f, mode="w+", dtype=str(t.dtype).replace("torch.", ""), shape=tuple(t.shape))
    riga = max(1, passo_byte // max(1, t[0].numel() * t.element_size()))
    for i in range(0, t.shape[0], riga):
        mm[i:i + riga] = t[i:i + riga].cpu().numpy()
    mm.flush()
    del mm
    fd = os.open(f, os.O_RDONLY)
    try:
        os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def leggi_tensore_in(f, t, passo_byte=1 << 28) -> bool:
    """VELOCE2 V2: .npy -> dentro t (stessa forma e tipo, bf16 come int16), a pezzi. False se forma/tipo diversi."""
    arr = np.load(f, mmap_mode="r")
    vista = t.view(torch.int16) if t.dtype == torch.bfloat16 else t
    if tuple(arr.shape) != tuple(t.shape) or str(arr.dtype) != str(vista.dtype).replace("torch.", ""):
        return False
    riga = max(1, passo_byte // max(1, arr[0].size * arr.itemsize))
    for i in range(0, arr.shape[0], riga):
        vista[i:i + riga] = torch.from_numpy(np.ascontiguousarray(arr[i:i + riga])).to(t.device)
    del arr
    return True


def router(mlp, x):
    """= s1_cattura.router (stessa matematica di MoEStudente.forward)."""
    if mlp.router_w32 is not None and x.dtype not in (torch.float32, torch.float64):
        logit = F.linear(x.float(), mlp.router_w32)
    else:
        logit = F.linear(x, mlp.router_w)
    prob = torch.softmax(logit, dim=-1, dtype=torch.float32 if x.dtype != torch.float64 else torch.float64)
    idx = torch.topk(prob, mlp.top_k, dim=-1).indices
    w = prob.gather(-1, idx)
    return idx, w / w.sum(-1, keepdim=True)


class AccDensi:
    """somme per (meta', sorgente): H = sum x_q x_q^T, Mx = sum x_f x_q^T (fp64, prodotti fp32 come s1_cattura), n."""

    def __init__(self, n, dev, perm=None):
        self.n, self.dev, self.perm = n, dev, perm
        self.H, self.M, self.N = {}, {}, {}

    def add(self, h, s, xq, xf):
        xq = xq.reshape(-1, self.n).float()
        xf = xf.reshape(-1, self.n).float()
        if self.perm is not None:
            xq, xf = xq[:, self.perm], xf[:, self.perm]
        k = (h, s)
        if k not in self.H:
            self.H[k] = torch.zeros(self.n, self.n, dtype=torch.float64, device=self.dev)
            self.M[k] = torch.zeros(self.n, self.n, dtype=torch.float64, device=self.dev)
            self.N[k] = 0
        self.H[k] += (xq.T @ xq).double()
        self.M[k] += (xf.T @ xq).double()
        self.N[k] += xq.shape[0]

    def meta(self, h):
        return {s: (self.H[(hh, s)], self.M[(hh, s)], self.N[(hh, s)]) for (hh, s) in self.H if hh == h}

    def pool(self, h, peso_kw: float = 1.0):
        """statistica di strato per gli esperti: tutte le sorgenti, pesate coi token (= sum_e sum_{t in e} w x x^T / N).
        14/10 G3: i token di Kw contano peso_kw volte (1 = identico a prima)."""
        ks = [k for k in self.H if k[0] == h]
        n = sum(self.N[k] for k in ks)
        if peso_kw == 1.0:
            return sum(self.H[k] for k in ks) / n, sum(self.M[k] for k in ks) / n, n
        c = {k: (peso_kw if k[1] == "Kw" else 1.0) for k in ks}
        ne = sum(c[k] * self.N[k] for k in ks)
        return sum(c[k] * self.H[k] for k in ks) / ne, sum(c[k] * self.M[k] for k in ks) / ne, n


class AccGateUp:
    """P3, UNA meta' alla volta: Gq_e = sum w x_q x_q^T, C_e = W_e[gate;up] sum w x_f x_q^T (= s1_cattura), conta, peso."""

    def __init__(self, E, I, H, dev):
        self.Gq = torch.zeros(E, H, H, dtype=torch.float32, device=dev)
        self.C = torch.zeros(E, 2 * I, H, dtype=torch.float32, device=dev)
        self.conta = torch.zeros(E, dtype=torch.int64, device=dev)
        self.peso = torch.zeros(E, dtype=torch.float64, device=dev)

    def azzera(self):
        for t in (self.Gq, self.C, self.conta, self.peso):
            t.zero_()

    def add(self, xq, xf, w, k, ordine, offs, Wgu32):
        tok = ordine // k
        wf = w.reshape(-1)[ordine].float()
        o = offs.tolist()
        for e in range(len(o) - 1):
            s0, s1 = o[e], o[e + 1]
            if s1 <= s0:
                continue
            t = tok[s0:s1]
            ww = wf[s0:s1, None]
            Xq = xq[t]
            self.Gq[e] += (Xq * ww).T @ Xq
            self.C[e] += (xf[t] @ Wgu32[e].T).mul_(ww).T @ Xq
            self.conta[e] += s1 - s0
            self.peso[e] += ww.double().sum()

    def scarica(self, d: Path):
        tmp = Path(str(d) + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        E, H = self.Gq.shape[0], self.Gq.shape[1]
        mm = np.lib.format.open_memmap(tmp / "Gq.npy", mode="w+", dtype=np.float32, shape=(E, H * (H + 1) // 2))
        for e0 in range(0, E, 16):
            mm[e0:e0 + 16] = SC.impacchetta(self.Gq[e0:e0 + 16]).cpu().numpy()
        mm.flush()
        del mm
        np.save(tmp / "C.npy", self.C.cpu().numpy())
        np.save(tmp / "conta.npy", self.conta.cpu().numpy())
        np.save(tmp / "peso.npy", self.peso.cpu().numpy())
        shutil.rmtree(d, ignore_errors=True)
        os.replace(tmp, d)


class AccDown:
    """P4, per meta': Hd_e = sum w a_q a_q^T (a_q = ingresso VERO del down di C), Md_e = sum w a_f a_q^T (a_f del maestro)."""

    def __init__(self, E, I, dev):
        self.Hd = {h: torch.zeros(E, I, I, dtype=torch.float32, device=dev) for h in "AB"}
        self.Md = {h: torch.zeros(E, I, I, dtype=torch.float32, device=dev) for h in "AB"}
        self.conta = {h: torch.zeros(E, dtype=torch.int64, device=dev) for h in "AB"}
        self.peso = {h: torch.zeros(E, dtype=torch.float64, device=dev) for h in "AB"}

    def add(self, h, a_ord, xf, w, k, ordine, offs, Wgu32):
        tok = ordine // k
        wf = w.reshape(-1)[ordine].float()
        o = offs.tolist()
        I = Wgu32.shape[1] // 2
        for e in range(len(o) - 1):
            s0, s1 = o[e], o[e + 1]
            if s1 <= s0:
                continue
            ww = wf[s0:s1, None]
            A = a_ord[s0:s1].float()
            gu = xf[tok[s0:s1]] @ Wgu32[e].T
            Af = F.silu(gu[:, :I]) * gu[:, I:]
            self.Hd[h][e] += (A * ww).T @ A
            self.Md[h][e] += (Af * ww).T @ A
            self.conta[h][e] += s1 - s0
            self.peso[h][e] += ww.double().sum()


def spacchetta_lotto(v, n, dev, dt):
    M = torch.zeros(v.shape[0], n, n, dtype=dt, device=dev)
    i, j = torch.triu_indices(n, n, device=dev)
    vv = v.to(dev, dt)
    M[:, i, j] = vv
    M[:, j, i] = vv
    return M


def U_Minv_lotto(Hn):
    """per ogni b: (Minv, U) = cholesky_smorzata(Hn_b) (percdamp 1 %), batch su CPU fp64; i falliti uno per uno."""
    dev, dt = Hn.device, Hn.dtype
    H = Hn.detach().to("cpu", torch.float64)
    n = H.shape[-1]
    dead = torch.diagonal(H, dim1=1, dim2=2) == 0
    if bool(dead.any()):
        H = H.clone()
        for b in range(H.shape[0]):
            if dead[b].any():
                H[b, dead[b], dead[b]] = 1
    dm = torch.diagonal(H, dim1=1, dim2=2).mean(1).view(-1, 1, 1)
    L, info = torch.linalg.cholesky_ex(H + 0.01 * dm * torch.eye(n, dtype=torch.float64))
    Minv = torch.cholesky_inverse(L)
    U, info2 = torch.linalg.cholesky_ex(Minv, upper=True)
    cattivi = ((info != 0) | (info2 != 0) | torch.isnan(U).flatten(1).any(1)).nonzero().flatten().tolist()
    for b in cattivi:
        Mi, Ub, _ = AL.cholesky_smorzata(Hn[b].detach().to("cpu", torch.float64))
        Minv[b], U[b] = Mi, Ub
    return Minv.to(dev, dt), U.to(dev, dt), len(cattivi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partenza", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--thor-v2", required=True)
    ap.add_argument("--fedeli", required=True)
    ap.add_argument("--rif", required=True, help="cartella con cardan_piega.py congelato")
    ap.add_argument("--grad-cal", default="")
    ap.add_argument("--grad-ctl", default="")
    ap.add_argument("--ricetta", default="B3_ind32")
    ap.add_argument("--motore", default="triton")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--strati", default="")
    ap.add_argument("--esperti", default="", help="prova: esperti risolti (vuoto = tutti); gli altri restano la partenza")
    ap.add_argument("--max-seq", type=int, default=0)
    ap.add_argument("--tronca", type=int, default=0)
    ap.add_argument("--lotto-token", type=int, default=8192)
    ap.add_argument("--lotto-esperti", type=int, default=16)
    ap.add_argument("--cache-maestro", type=int, default=1)
    ap.add_argument("--tetto", type=float, default=64.0)
    ap.add_argument("--cicli", default="")
    a = ap.parse_args()
    global CICLI
    CICLI = a.cicli or None
    opz = {"alfe": [float(x) for x in os.environ.get("S23_ALFE", "0,0.5,1").split(",")],      # 14/10 G2: + a = 0
           "w": float(os.environ.get("S23_W_CONOSCENZA", "1.0")),
           "damp_qep": float(os.environ.get("S23_DAMP_QEP", "0.01")),
           "drq": os.environ.get("S23_DRQ", "1") == "1",
           "k1": tuple(x for x in os.environ.get("S23_K1", ",".join(RS.FAM_K1_DEFAULT)).split(",") if x),
           "giri_k1": int(os.environ.get("S23_GIRI_K1", "2")), "k2": os.environ.get("S23_K2", "1") == "1",
           "giri_k2": int(os.environ.get("S23_GIRI_K2", "2")), "limite_c": 8.0,
           # 13/10 DRQ per famiglia (s23_risolvi «DRQ per famiglia»): griglia di 2610.11226 App. C, tarata sulla meta' B
           "drq_modo": os.environ.get("S23_DRQ_GRIGLIA", "paper+norm"),
           "drq_strati_tar": int(os.environ.get("S23_DRQ_STRATI_TAR", "2")),
           # 14/10 guardia indipendente + H corta (s23_risolvi G1-G3)
           "energia_min": float(os.environ.get("S23_ENERGIA_MIN", "0.9")),
           "energia_max": float(os.environ.get("S23_ENERGIA_MAX", "1.1")),
           "guardia": os.environ.get("S23_GUARDIA", "sim"),
           "peso_kw": float(os.environ.get("S23_PESO_KW", "1.0")),
           "energia_relativa": os.environ.get("S23_ENERGIA_RELATIVA", "1") == "1",
           # VELOCE 14/10 (L3): K1 a banda + ricalcolo esatto entro delta dalla scelta (s23_risolvi); 0 = fastkron_sd sempre
           "k1_banda": os.environ.get("S23_K1_BANDA", "1") == "1",
           "k1_banda_delta": float(os.environ.get("S23_K1_DELTA", "0.002"))}
    if opz["guardia"] not in ("sim", "kl_locale"):
        raise ValueError(f"S23_GUARDIA {opz['guardia']!r} non in (sim, kl_locale)")
    RS.griglia_drq(opz["drq_modo"])                       # modo sconosciuto -> errore subito, non allo strato 0
    alfa_e = [float(x) for x in os.environ.get("S23_ALFE_ESPERTI", os.environ.get("S23_ALFA_ESPERTI", "0,0.5")).split(",")]
    guardia_e = {"emin": opz["energia_min"], "emax": opz["energia_max"], "peso_kw": opz["peso_kw"],
                 "relativa": opz["energia_relativa"]}
    kk = float(os.environ.get("S23_K", "8000"))
    usa_coda = os.environ.get("S23_CODA", "1") == "1"                 # VELOCE2 V1
    usa_flussi = os.environ.get("S23_FLUSSI", "1") == "1"             # VELOCE2 V2
    quale_B = os.environ.get("S23_B", "B")
    sys.path.insert(0, a.thor_v2)
    sys.path.insert(0, a.fedeli)
    sys.path.insert(0, a.rif)
    import e2e_studente as S
    import sequenziale_thor as ST
    import cardan_piega as CP
    # S23_KBEST > 1: K-best di Klein (OJBKQ 2602.08376) nel passo di codici di JARQ; spento = CP.jarq_lotto al bit
    # VELOCE 14/10 (L2): spento = CP.jarq_lotto con le Cholesky dei blocchi di Babai tenute per lotto (al bit)
    JQL = AL.jarq_lotto_kbest_da(CP) if AL.KBEST > 1 else AL.jarq_lotto_cache_da(CP)
    print(f"K-best di Klein: S23_KBEST={AL.KBEST} ({'acceso' if AL.KBEST > 1 else 'spento'}), seme {AL.KBEST_SEME}", flush=True)
    from sorgente import SorgenteHF, bf16_a_f32
    gguf = K.gp._import_gguf(None)
    dev = torch.device(a.device)
    dt = AL.dt_lavoro(dev)
    torch.backends.cuda.matmul.allow_tf32 = False
    out = Path(a.out)
    (out / "strati").mkdir(parents=True, exist_ok=True)
    lav = out / "lavoro"
    t_avvio = time.time()
    log(f"opzioni {opz} | esperti: alfa {alfa_e}, k {kk} | B={quale_B} | cache maestro {a.cache_maestro} | dt {dt} | "
        f"VELOCE2 coda {usa_coda} flussi {usa_flussi} | fili torch {torch.get_num_threads()} OPENBLAS "
        f"{os.environ.get('OPENBLAS_NUM_THREADS', '-')} OMP {os.environ.get('OMP_NUM_THREADS', '-')}")
    rd = gguf.GGUFReader(a.partenza, "r")
    tens = {t.name: t for t in rd.tensors}
    perm_ssm = torch.from_numpy(dict(np.load(Path(a.thor_v2) / "mappa_completa_gguf_hf_perm.npz"))
                                ["linear.ssm_out.weight"].astype(np.int64)).to(dev)
    st = S.Studente(gguf=Path(a.partenza), strati=range(40), ricetta=a.ricetta, motore_nome=a.motore, device=str(dev),
                    densi_ternari="tutte", log=log)
    cfg = st.cfg
    nL = len(cfg.layer_types)
    strati = elenco(a.strati, nL)
    assert strati == list(range(strati[0], strati[-1] + 1)) and strati[0] == 0, "strati: prefisso contiguo da 0"
    if dev.type == "cuda":
        st.carica_embedding()
    P = ST.PulitoHF(cfg, str(dev))
    if os.environ.get("S23_MAESTRO_SENZA_SYNC", "1") == "1":         # VELOCE2 V3
        import types
        for _t, (_L, _s) in P.sch.items():
            _ex = _s.mlp.experts
            _ex.forward = types.MethodType(esperti_senza_sync, _ex)
        log(f"VELOCE2 V3: esperti del maestro senza sincronizzazioni per esperto ({sorted(P.sch)})")
    E, I, Hd = int(cfg.num_experts), int(cfg.moe_intermediate_size), int(cfg.hidden_size)
    P_gqa = AL.legame_gqa(cfg.num_attention_heads, cfg.num_key_value_heads,
                          getattr(cfg, "head_dim", None) or Hd // cfg.num_attention_heads, dev, dt)
    sh = SorgenteHF()
    es_lista = elenco(a.esperti, E) if a.esperti else list(range(E))
    dati = torch.load(a.corpus, map_location="cpu", weights_only=False)
    chiavi = [(h, s) for h in "AB" for s in SC.SORGENTI]
    ids, hp, hq, ctx = {}, {}, {}, {}
    for (h, s) in chiavi:
        v = dati["sorgenti"][s][dati["meta"][s][h]].long()
        if a.max_seq:
            v = v[: a.max_seq]
        if a.tronca:
            v = v[:, : a.tronca]
        ids[(h, s)] = v
        N, T = v.shape
        B = max(1, a.lotto_token // T)
        hp[(h, s)] = torch.empty(N, T, Hd, dtype=P.dt, device=dev)
        hq[(h, s)] = torch.empty(N, T, Hd, dtype=st.ric.dtype_residuo, device=dev)
        for i in range(0, N, B):
            hp[(h, s)][i:i + B] = P.embed(v[i:i + B])
            hq[(h, s)][i:i + B] = st.embedding(v[i:i + B]).to(hq[(h, s)].dtype)
        ctx[(h, s)] = {b: (st._contesto(b, T, st.ric.dtype_residuo), st._contesto(b, T, P.dt))
                       for b in sorted({B, N % B} - {0})}
        log(f"meta' {h} sorgente {s}: {N} x {T} = {N * T} token, lotti da {B}")
    tok_tot = sum(v.numel() for v in ids.values())
    cache = {}
    if a.cache_maestro:
        for k, v in ids.items():
            N, T = v.shape
            cache[k] = {p: torch.empty(N, T, n, dtype=P.dt, device=dev)
                        for p, n in (("att_out", 4096), ("ffn", Hd), ("down", int(cfg.shared_expert_intermediate_size)))}
        log(f"cache del maestro: {sum(x.numel() * 2 for c in cache.values() for x in c.values()) / 2**30:.1f} GiB")

    def lotti():
        for k in chiavi:
            N, T = ids[k].shape
            B = max(1, a.lotto_token // T)
            for i in range(0, N, B):
                sl = slice(i, min(i + B, N))
                yield k, sl, ctx[k][sl.stop - sl.start]

    def vram():
        if dev.type == "cuda":
            v = torch.cuda.memory_reserved() / 2**30
            if v > a.tetto:
                raise RuntimeError(f"VRAM {v:.1f} GiB > tetto {a.tetto}")
            return v
        return 0.0

    def coda(L, x):
        return coda_strato(st, L, x)

    def esegui(L, voglio_q=(), ferma_q=None, voglio_f=(), ferma_f=None, cb=None, avanza_q=False, avanza_f=False,
               fam_down=None, hmid=None):
        """un giro su tutti i lotti: maestro (o cache) e C, ognuno fermato al proprio punto o completo (avanza).
        VELOCE2 V1: hmid = {chiave: residuo dopo il miscelatore} -> C parte dalla coda invece che da _strato."""
        sp, sq = P.strato(L), st.moduli[L]
        full = cfg.layer_types[L] == "full_attention"
        for k, sl, (cq, cp) in lotti():
            xf, op = {}, None
            if voglio_f or avanza_f:
                if a.cache_maestro and not avanza_f and set(voglio_f) <= {"att_out", "ffn", "down"}:
                    xf = {p: cache[k][p][sl] for p in voglio_f}
                else:
                    g = Gancio(sp, full, voglio_f, None if avanza_f else ferma_f)
                    try:
                        with torch.no_grad():
                            op = P.avanti(L, hp[k][sl], cp)
                    except Ferma:
                        pass
                    finally:
                        g.chiudi()
                    xf = g.d
            g = Gancio(sq, full, voglio_q, None if avanza_q else ferma_q, S, fam_down)
            oq = None
            try:
                with torch.no_grad():
                    oq = coda(L, hmid[k][sl]) if hmid is not None else st._strato(L, hq[k][sl], cq)
            except Ferma:
                pass
            finally:
                g.chiudi()
            if cb is not None:
                cb(k, sl, g.d, xf, oq, op)
            if avanza_f and op is not None:
                hp[k][sl] = op.to(hp[k].dtype)
            if avanza_q:
                hq[k][sl] = oq.to(hq[k].dtype)
        vram()

    # ── iniezione dei byte nello studente (oggetti NUOVI: nessuna cache dello studente li scavalca) ──
    def inietta(L, fam, raw):
        es = st.esperti(L)
        old = es.fam[fam]
        Bb = torch.from_numpy(np.ascontiguousarray(raw))
        if Bb.dim() == 2:
            Bb = Bb[None]
        assert Bb.dtype == torch.uint8 and tuple(Bb.shape) == tuple(old.blocchi.shape), (L, fam, Bb.shape, old.blocchi.shape)
        d0 = S.CS.scale_da_blocchi(Bb.reshape(-1, old.nb), old.tipo).to(torch.float32).reshape(Bb.shape[0], Bb.shape[1], -1)
        es.fam[fam] = S.DatiFamiglia(old.nome, old.tipo, Bb.to(old.blocchi.device), d0.to(old.d0.device), old.strato)

    def raw_partenza(nome):
        return np.asarray(tens[nome].data)

    def codifica(T, Sc, tid, C):
        Tn = T.to("cpu").numpy().astype(np.int8)
        S16 = Sc.to("cpu", torch.float16).numpy()
        b = K.gp.encode_rows(Tn, S16, tid)
        tp, sp_ = K.gp.decode_rows(b, tid, C)
        assert np.array_equal(tp, Tn) and np.array_equal(sp_.view(np.uint16), S16.view(np.uint16)), "andata e ritorno"
        return b

    def decodifica(raw_rows, tid, C):
        t, d = K.gp.decode_rows(np.ascontiguousarray(raw_rows), tid, C)
        return torch.from_numpy(t.astype(np.int8)), torch.from_numpy(d.astype(np.float64))

    def fam_di(nome):
        return nome.split(".")[2]

    def ripassa(L):
        d = out / "strati" / f"strato_{L:02d}"
        for f in sorted((d / "tensori").glob("blk.*.npy")):
            nome = f.name[:-4]
            fam = fam_di(nome)
            chiave = {"ffn_gate_exps": "gate", "ffn_up_exps": "up", "ffn_down_exps": "down"}.get(fam, fam)
            inietta(L, chiave, np.load(f))
        esegui(L, avanza_q=True, avanza_f=True)

    def cancello_iniezione(L):
        """(1) byte della partenza come oggetti nuovi -> identico al bit; alterati -> diverso; rimessi -> identico."""
        k0 = chiavi[0]
        N, T = ids[k0].shape
        b = min(max(1, a.lotto_token // T), N)
        cq = ctx[k0][b][0] if b in ctx[k0] else st._contesto(b, T, st.ric.dtype_residuo)
        x = hq[k0][:b]
        with torch.no_grad():
            o0 = st._strato(L, x, cq).clone()
        es = st.esperti(L)
        nomi = {f: es.fam[f] for f in list(es.fam)}
        for f, fam in nomi.items():
            inietta(L, f, fam.blocchi.detach().cpu().numpy())
        with torch.no_grad():
            o1 = st._strato(L, x, cq).clone()
        f0 = next(f for f in nomi if f not in ("gate", "up", "down"))
        raw = es.fam[f0].blocchi.detach().cpu().numpy().copy()
        tid = {"TQ1_0": 34, "TQ1_0R_1024": 45}[es.fam[f0].tipo]
        C = es.fam[f0].C
        Tt, Ss = decodifica(raw[0], tid, C)
        inietta(L, f0, codifica(Tt, Ss * 0.5, tid, C))
        with torch.no_grad():
            o2 = st._strato(L, x, cq).clone()
        inietta(L, f0, raw)
        with torch.no_grad():
            o3 = st._strato(L, x, cq)
        ok = torch.equal(o0, o1) and not torch.equal(o0, o2) and torch.equal(o0, o3)
        log(f"cancello iniezione strato {L} ({f0} alterata): identico {torch.equal(o0, o1)}, alterato diverso "
            f"{not torch.equal(o0, o2)}, rimesso identico {torch.equal(o0, o3)}")
        if not ok:
            raise RuntimeError("cancello iniezione: la sostituzione dei byte non agisce (o non si annulla) sul flusso di C")

    coda_viva = {"si": usa_coda}

    def cancello_coda(L, hm, fam_down):
        """VELOCE2 V1, prima di P4: sul primo lotto i ganci di P4 (ffn, down, a_ord, piano) dalla coda == quelli del
        passaggio completo AL BIT. False -> la coda si spegne (P4/P5 tornano al passaggio completo)."""
        k, sl, (cq, _) = next(lotti())
        full = cfg.layer_types[L] == "full_attention"
        dd = []
        for da_coda in (False, True):
            g = Gancio(st.moduli[L], full, ("ffn", "down", "a_ord"), "a_ord", S, fam_down)
            try:
                with torch.no_grad():
                    if da_coda:
                        coda(L, hm[k][sl])
                    else:
                        st._strato(L, hq[k][sl], cq)
            except Ferma:
                pass
            finally:
                g.chiudi()
            dd.append(g.d)
        ok = all(torch.equal(dd[0][p], dd[1][p]) for p in ("ffn", "down", "a_ord")) \
            and torch.equal(dd[0]["piano"].ordine, dd[1]["piano"].ordine) and torch.equal(dd[0]["piano"].offs, dd[1]["piano"].offs)
        log(f"cancello coda P4 strato {L}: ganci ffn/down/a_ord/piano dalla coda == passaggio completo al bit: {ok}"
            + ("" if ok else " -> CODA SPENTA, P4/P5 col passaggio completo"))
        return ok

    def registri_chiusi(L):
        """13/10: [(strato, registro)] degli strati CHIUSI (FATTO) prima di L: la taratura DRQ per famiglia si ricalcola da
        qui a ogni tensore, cosi' una ripresa non conta due volte uno strato."""
        rr = []
        for x in range(L):
            dx = out / "strati" / f"strato_{x:02d}"
            if (dx / "FATTO").exists():
                rr.append((x, json.loads((dx / "registro.json").read_text())))
        return rr

    # ── VELOCE2 V2: flussi salvati a fine strato (solo l'ultimo), ricaricati alla ripresa se il meta' combacia ──
    FL = out / "flussi"

    def impronta(L):
        return impronta_strato(out / "strati" / f"strato_{L:02d}")

    def meta_base():
        pz = Path(a.partenza)
        cp = Path(a.corpus)
        return {"partenza": str(pz), "partenza_byte": pz.stat().st_size, "corpus": str(cp), "corpus_byte": cp.stat().st_size,
                "lotto_token": a.lotto_token, "max_seq": a.max_seq, "tronca": a.tronca, "ricetta": a.ricetta,
                "forme": {f"{h}_{s_}": list(ids[(h, s_)].shape) for (h, s_) in chiavi},
                "dtype_q": str(next(iter(hq.values())).dtype), "dtype_f": str(next(iter(hp.values())).dtype)}

    def salva_flussi(L, impronte):
        t0 = time.time()
        serve = sum(t.numel() * t.element_size() for d_ in (hq, hp) for t in d_.values())
        gia = sum(f.stat().st_size for f in FL.glob("*.npy")) if FL.exists() else 0
        libero = shutil.disk_usage(out).free + gia
        if libero - serve < 40 * 2**30:      # mai sotto 40 GiB liberi per i flussi (le uscite vere hanno la precedenza)
            log(f"flussi NON salvati dopo lo strato {L}: {libero / 2**30:.0f} GiB liberi, ne servono {serve / 2**30:.1f} + 40")
            return
        tmp = Path(str(FL) + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(FL, ignore_errors=True)     # prima: un solo insieme di flussi su disco (~16 GiB), mai due
        tmp.mkdir(parents=True)
        for (h, s_) in chiavi:
            scrivi_tensore(tmp / f"q_{h}_{s_}.npy", hq[(h, s_)])
            scrivi_tensore(tmp / f"f_{h}_{s_}.npy", hp[(h, s_)])
        meta = {**meta_base(), "strato": L, "impronte": impronte, "scritto": time.strftime("%FT%T"),
                "codice": str(QUI.parent)}
        SC.scrivi_json(tmp / "meta.json", meta)
        shutil.rmtree(FL, ignore_errors=True)
        os.replace(tmp, FL)
        log(f"flussi salvati dopo lo strato {L} in {time.time() - t0:.1f} s ({FL})")

    def carica_flussi(L0):
        """-> strato L dei flussi caricati (hq, hp = uscite dello strato L), o -1 se assenti/incoerenti."""
        mf = FL / "meta.json"
        if not usa_flussi or not mf.exists():
            return -1
        t0 = time.time()
        meta = json.loads(mf.read_text())
        Lc = int(meta.get("strato", -1))
        base = meta_base()
        diversi = [x for x in base if meta.get(x) != base[x]]
        if diversi or not 0 <= Lc < L0:
            log(f"ripresa: flussi di {FL} non usabili (strato {Lc}, fatti 0..{L0 - 1}, meta' diversi {diversi}): ripasso")
            return -1
        for L in range(Lc + 1):
            imp = impronta(L)
            if imp != meta["impronte"].get(str(L)):
                log(f"ripresa: impronta dello strato {L} diversa da quella dei flussi: ripasso")
                return -1
        for (h, s_) in chiavi:                # PRIMA di toccare hq/hp: tutte le forme e i tipi (i flussi restano intatti)
            for nome, dest in (("q", hq), ("f", hp)):
                f = FL / f"{nome}_{h}_{s_}.npy"
                t = dest[(h, s_)]
                vt = t.view(torch.int16) if t.dtype == torch.bfloat16 else t
                arr = np.load(f, mmap_mode="r") if f.exists() else None
                if arr is None or tuple(arr.shape) != tuple(t.shape) or str(arr.dtype) != str(vt.dtype).replace("torch.", ""):
                    log(f"ripresa: {f.name} assente o di forma/tipo diversi: ripasso")
                    return -1
                del arr
        for (h, s_) in chiavi:
            for nome, dest in (("q", hq), ("f", hp)):
                if not leggi_tensore_in(FL / f"{nome}_{h}_{s_}.npy", dest[(h, s_)]):
                    raise RuntimeError(f"ripresa: {nome}_{h}_{s_} cambiato durante la lettura")
        log(f"ripresa: flussi dello strato {Lc} ricaricati da {FL} in {time.time() - t0:.1f} s (impronte di 0..{Lc} ok)")
        return Lc

    impronte = {}
    fatti = [L for L in strati if (out / "strati" / f"strato_{L:02d}" / "FATTO").exists()]
    L0 = 0
    while L0 in fatti:
        L0 += 1
    Lc = carica_flussi(L0) if L0 else -1
    for L in range(L0):
        if L <= Lc:                          # flussi gia' oltre questo strato: solo i byte nello studente
            d = out / "strati" / f"strato_{L:02d}"
            for f in sorted((d / "tensori").glob("blk.*.npy")):
                nome = f.name[:-4]
                fam = fam_di(nome)
                inietta(L, {"ffn_gate_exps": "gate", "ffn_up_exps": "up", "ffn_down_exps": "down"}.get(fam, fam), np.load(f))
            log(f"ripresa: strato {L} iniettato (flussi salvati)")
        else:
            ripassa(L)
            log(f"ripresa: strato {L} riapplicato")
    if usa_flussi:
        impronte = {str(L): impronta(L) for L in range(L0)}
    resto = [L for L in strati if L >= L0]
    gradc = Path(a.grad_cal) if a.grad_cal and Path(a.grad_cal, "fatto.txt").exists() else None
    gradk = Path(a.grad_ctl) if a.grad_ctl and Path(a.grad_ctl, "fatto.txt").exists() else None
    log(f"fattori di Kronecker: cal {gradc} ctl {gradk} (K1 {'attivo' if gradc and gradk else 'SPENTO: fattori assenti'})")
    if resto:
        cancello_iniezione(resto[0])
    for L in resto:
        tl = time.time()
        tipo = cfg.layer_types[L]
        full = tipo == "full_attention"
        d = out / "strati" / f"strato_{L:02d}"
        tmpd = Path(str(d) + ".tmp")
        shutil.rmtree(tmpd, ignore_errors=True)
        (tmpd / "tensori").mkdir(parents=True)
        reg = {"strato": L, "tipo": tipo, "densi": {}, "esperti": {}, "tempi": {}, "partenza": a.partenza}
        nuovi = {}                       # nome -> byte del tensore intero
        scelti = {}                      # fam -> (T, S, tid) dei densi gia' decisi (per le pieghe K2)

        def risolvi_densi(stadio, acc):
            for fam in FAM_STADIO[tipo][stadio]:
                nome = f"blk.{L}.{fam}.weight"
                if nome not in tens or int(tens[nome].tensor_type) not in TERNARI:
                    continue
                t = tens[nome]
                tid = int(t.tensor_type)
                C, R = int(t.shape[0]), int(t.shape[1])
                T0, S0 = decodifica(raw_partenza(nome).reshape(R, -1), tid, C)
                u16, _ = sh.u16_ordine_gguf(nome)
                W = torch.from_numpy(bf16_a_f32(u16)).to(dev, dt)
                kron = RS.Kron(gradc, gradk, nome, quale_B, dev, dt) if fam in opz["k1"] else None
                monte = None
                if fam in RS.COPPIE_K2 and RS.COPPIE_K2[fam] in scelti:
                    Tm, Sm, _ = scelti[RS.COPPIE_K2[fam]]
                    monte = (Tm, Sm)
                opz_t = opz
                if opz["drq"]:                        # 13/10: taratura per famiglia dai registri degli strati CHIUSI
                    opz_t = {**opz, "drq_taratura": RS.taratura_famiglia(registri_chiusi(L), fam, opz["drq_strati_tar"],
                                                                           opz["drq_modo"])}
                r = RS.risolvi_denso(nome, fam, tid, W, acc.meta("A"), acc.meta("B"), T0, S0, opz_t, kron, P_gqa, monte)
                scelti[fam] = (r["T"], r["S"], tid)
                reg["densi"][nome] = r["registro"]
                if r["scelto"] != "partenza":
                    nuovi[nome] = codifica(r["T"], r["S"], tid, C)
                    inietta(L, fam, nuovi[nome])
                if r["S_monte"] is not None:
                    fm = RS.COPPIE_K2[fam]
                    nm = f"blk.{L}.{fm}.weight"
                    Tm, _, tidm = scelti[fm]
                    scelti[fm] = (Tm, r["S_monte"], tidm)
                    nuovi[nm] = codifica(Tm, r["S_monte"], tidm, int(tens[nm].shape[0]))
                    inietta(L, fm, nuovi[nm])
                    reg["densi"][nm]["piega_K2_da"] = fam
                rr = r["registro"]
                dg = rr.get("drq_griglia")
                if dg:
                    log(f"  {nome}: DRQ {'TARATO' if dg['tarato'] else 'fisso'} ({dg['modo']}) -> {dg['scelta']}"
                        + (f" (rel {dg['rel'][dg['scelta']]:+.3e}, coppia di prima {dg['rel_fissa']:+.3e}, "
                           f"somma {dg['somma_scelta']:+.3e}, {dg['secondi']} s)" if dg["tarato"] else ""))
                log(f"  {nome}: scelto {r['scelto']} (guardia {rr['metrica_guardia']}, trit cambiati "
                    f"{rr['trit_cambiati']:.3f}, {rr['secondi']} s) "
                    + " ".join(f"{k}:{v['dJB_rel']:+.4f}" for k, v in rr["candidati"].items()))
                del W

        # ═══ P1: maestro (completo in cache, altrimenti fermo all'attn) + C fermo all'attn ═══
        t = time.time()
        acc1 = AccDensi(Hd, dev)

        def cb1(k, sl, dq, df, oq, op):
            acc1.add(k[0], k[1], dq["attn"], df["attn"])
            if a.cache_maestro:
                for p in ("att_out", "ffn", "down"):
                    cache[k][p][sl] = df[p].reshape(cache[k][p][sl].shape)
        if a.cache_maestro:
            esegui(L, voglio_q=("attn",), ferma_q="attn", voglio_f=("attn", "att_out", "ffn", "down"), cb=cb1,
                   avanza_f=True)
        else:
            esegui(L, voglio_q=("attn",), ferma_q="attn", voglio_f=("attn",), ferma_f="attn", cb=cb1)
        reg["tempi"]["P1_passaggio"] = round(time.time() - t, 1)
        t = time.time()
        risolvi_densi(1, acc1)
        del acc1
        reg["tempi"]["P1_densi"] = round(time.time() - t, 1)

        # ═══ P2: ingresso di ssm_out | attn_output ═══
        t = time.time()
        acc2 = AccDensi(4096, dev, None if full else perm_ssm)
        esegui(L, voglio_q=("att_out",), ferma_q="att_out", voglio_f=("att_out",), ferma_f="att_out",
               cb=lambda k, sl, dq, df, oq, op: acc2.add(k[0], k[1], dq["att_out"], df["att_out"]))
        reg["tempi"]["P2_passaggio"] = round(time.time() - t, 1)
        t = time.time()
        risolvi_densi(2, acc2)
        del acc2
        reg["tempi"]["P2_densi"] = round(time.time() - t, 1)

        # ═══ P3: ingresso del MoE: shexp gate/up + esperti gate/up ═══
        t = time.time()
        sp = P.strato(L)
        Wgu = ST.modulo(sp, "mlp.experts").gate_up_proj.detach()
        Wd_m = ST.modulo(sp, "mlp.experts").down_proj.detach()
        assert tuple(Wgu.shape) == (E, 2 * I, Hd) and tuple(Wd_m.shape) == (E, Hd, I), (Wgu.shape, Wd_m.shape)
        Wgu32 = Wgu.float()
        acc3 = AccDensi(Hd, dev)
        agu = AccGateUp(E, I, Hd, dev)
        mlp = st.moduli[L].mlp
        dl = lav / f"strato_{L:02d}"
        stato3 = {"meta": None}
        hm = None
        if coda_viva["si"]:                  # VELOCE2 V1: residuo dopo il miscelatore di C, per P4 e P5
            hm = {k: torch.empty(ids[k].shape[0], ids[k].shape[1], Hd, dtype=hq[k].dtype, device=dev) for k in chiavi}

        def cb3(k, sl, dq, df, oq, op):
            if hm is not None:
                hm[k][sl] = dq["hmid"]
            h = k[0]
            if stato3["meta"] is not None and stato3["meta"] != h:
                agu.scarica(dl / f"gu_{stato3['meta']}")
                agu.azzera()
            stato3["meta"] = h
            acc3.add(h, k[1], dq["ffn"], df["ffn"])
            xq = dq["ffn"].reshape(-1, Hd)
            idx, w = router(mlp, xq)
            pz = S.Piano(idx, E)
            agu.add(xq.float(), df["ffn"].reshape(-1, Hd).float(), w, mlp.top_k, pz.ordine, pz.offs, Wgu32)
        esegui(L, voglio_q=("ffn", "hmid") if hm is not None else ("ffn",), ferma_q="ffn", voglio_f=("ffn",), ferma_f="ffn",
               cb=cb3)
        agu.scarica(dl / f"gu_{stato3['meta']}")
        del agu
        gc.collect()
        reg["tempi"]["P3_passaggio"] = round(time.time() - t, 1)
        t = time.time()
        risolvi_densi(3, acc3)
        reg["tempi"]["P3_densi"] = round(time.time() - t, 1)
        t = time.time()
        reg["esperti"]["gate_up"] = risolvi_esperti_gu(L, dl, acc3, Wgu32, nuovi, inietta, codifica, decodifica, raw_partenza,
                                                        es_lista, alfa_e, kk, a.lotto_esperti, JQL, dev, dt, E, I, Hd, guardia_e)
        del acc3
        reg["tempi"]["P3_esperti"] = round(time.time() - t, 1)

        # ═══ P4: ingressi VERI dei down (gate/up nuovi): ffn_down_shexp + esperti down ═══
        t = time.time()
        acc4 = AccDensi(int(cfg.shared_expert_intermediate_size), dev)
        adn = AccDown(E, I, dev)
        fam_down = st.esperti(L).fam["down"]
        cont = {"top8_diversi": 0}

        def cb4(k, sl, dq, df, oq, op):
            acc4.add(k[0], k[1], dq["down"], df["down"])
            xq = dq["ffn"].reshape(-1, Hd)
            idx, w = router(mlp, xq)
            if mlp.ultimo_idx is not None and not torch.equal(idx, mlp.ultimo_idx):
                cont["top8_diversi"] += int((idx != mlp.ultimo_idx).any(-1).sum())
            pz = dq["piano"]
            xf = df["ffn"].reshape(-1, Hd).float() if "ffn" in df else None
            adn.add(k[0], dq["a_ord"], xf, w, mlp.top_k, pz.ordine, pz.offs, Wgu32)
        if hm is not None and not cancello_coda(L, hm, fam_down):
            hm = None                        # fail-safe: coda diversa dal passaggio completo -> si torna a quello
            coda_viva["si"] = False
        esegui(L, voglio_q=("ffn", "down", "a_ord"), ferma_q="a_ord", voglio_f=("ffn", "down"), ferma_f="down",
               cb=cb4, fam_down=fam_down, hmid=hm)
        if cont["top8_diversi"]:
            raise RuntimeError(f"strato {L}: top-8 ricalcolato diverso da quello dello studente su {cont['top8_diversi']} token")
        reg["tempi"]["P4_passaggio"] = round(time.time() - t, 1)
        t = time.time()
        risolvi_densi(4, acc4)
        del acc4
        reg["tempi"]["P4_densi"] = round(time.time() - t, 1)
        t = time.time()
        reg["esperti"]["down"] = risolvi_esperti_down(L, adn, Wd_m, nuovi, inietta, codifica, decodifica, raw_partenza,
                                                      es_lista, alfa_e, kk, a.lotto_esperti, JQL, dev, dt, E, I, Hd, guardia_e)
        del adn, Wgu32, Wgu, Wd_m
        gc.collect()
        reg["tempi"]["P4_esperti"] = round(time.time() - t, 1)

        # ═══ P5: passaggio completo con lo strato nuovo; deriva contro il maestro ═══
        t = time.time()
        der = {h: [0.0, 0.0] for h in "AB"}

        prova5 = {"fatta": hm is None}

        def cb5(k, sl, dq, df, oq, op):
            if not prova5["fatta"]:          # VELOCE2 V1: primo lotto, uscita della coda == _strato AL BIT
                prova5["fatta"] = True
                with torch.no_grad():
                    o_full = st._strato(L, hq[k][sl], ctx[k][sl.stop - sl.start][0])
                ok5 = torch.equal(o_full, oq)
                log(f"cancello coda P5 strato {L}: uscita della coda == passaggio completo al bit: {ok5}")
                del o_full
                if not ok5:
                    raise RuntimeError(f"strato {L}: uscita della coda diversa dal passaggio completo (P5)")
            rif = op if op is not None else hp[k][sl]
            der[k[0]][0] += float((oq.float() - rif.float()).pow(2).sum())
            der[k[0]][1] += float(rif.float().pow(2).sum())
        esegui(L, cb=cb5, avanza_q=True, avanza_f=not a.cache_maestro, hmid=hm)
        del hm
        reg["deriva_strato"] = {h: v[0] / max(v[1], 1e-300) for h, v in der.items()}
        reg["tempi"]["P5_passaggio"] = round(time.time() - t, 1)
        for nome, b in nuovi.items():
            np.save(tmpd / "tensori" / f"{nome}.npy", b)
        reg["tensori_cambiati"] = sorted(nuovi)
        reg["secondi"] = round(time.time() - tl, 1)
        if dev.type == "cuda":
            reg["vram_picco_gib"] = round(torch.cuda.max_memory_reserved() / 2**30, 1)
        SC.scrivi_json(tmpd / "registro.json", reg)
        (tmpd / "FATTO").write_text(time.strftime("%FT%T") + "\n")
        shutil.rmtree(d, ignore_errors=True)
        os.replace(tmpd, d)
        if usa_flussi:                       # VELOCE2 V2 (dopo FATTO: i flussi sono le uscite di uno strato chiuso)
            impronte[str(L)] = impronta(L)
            try:
                salva_flussi(L, impronte)
            except OSError as ex:
                log(f"flussi NON salvati dopo lo strato {L}: {ex} (la ripresa ripassera')")
                shutil.rmtree(Path(str(FL) + ".tmp"), ignore_errors=True)
        if os.environ.get("S23_TIENI_LAVORO") != "1":
            shutil.rmtree(dl, ignore_errors=True)
        nd = sum(1 for v in reg["densi"].values() if v["scelto"] != "partenza")
        gu, dn = reg["esperti"]["gate_up"], reg["esperti"]["down"]
        msg = (f"S23 strato {L} ({'full' if full else 'GDN '}) {reg['secondi'] / 60:.1f} min: densi nuovi {nd}/{len(reg['densi'])}, "
               f"esperti gate {gu['accettati_gate']} up {gu['accettati_up']} down {dn['accettati']} su {len(es_lista)}; "
               f"deriva A {reg['deriva_strato']['A']:.4g} B {reg['deriva_strato']['B']:.4g}; tempi {reg['tempi']}")
        log(msg)
        with open(out / "passata.jsonl", "a") as f:
            f.write(json.dumps({k: reg[k] for k in ("strato", "tipo", "deriva_strato", "tempi", "secondi",
                                                     "tensori_cambiati")}) + "\n")
        fatti_ora = len([x for x in strati if (out / "strati" / f"strato_{x:02d}" / "FATTO").exists()])
        resto_s = (len(strati) - fatti_ora) * (time.time() - t_avvio) / max(1, fatti_ora - L0)
        if L in (0, 1) or L % 10 == 9:
            riga_cicli("ATTIVO", f"{msg[:300]}; fine stimata {time.strftime('%d/%m %H:%M', time.localtime(time.time() + resto_s))}")
        gc.collect()
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    if strati == list(range(nL)) and all((out / "strati" / f"strato_{L:02d}" / "FATTO").exists() for L in range(nL)):
        (out / "FATTO_PASSATA").write_text(f"{time.strftime('%FT%T')} {(time.time() - t_avvio) / 60:.1f} min\n")
    log(f"passata: strati {resto[0] if resto else '-'}..{strati[-1]} in {(time.time() - t_avvio) / 60:.1f} min; token {tok_tot}")
    return 0


def _lotti_esperti(es_lista, n):
    for i in range(0, len(es_lista), n):
        yield es_lista[i:i + n]


def risolvi_esperti_gu(L, dl, acc3, Wgu32, nuovi, inietta, codifica, decodifica, raw_partenza, es_lista, alfe, kk, lotto,
                       jarq_lotto, dev, dt, E, I, Hd, gd=None):
    """S3 gate/up (E1): gate e up hanno la STESSA H d'ingresso: si risolvono come un'unica matrice [2I, H] (GPTQ e JARQ sono
    indipendenti per riga: identico a due chiamate separate).
    14/10 G1/G2: un candidato per alfa in `alfe` (QEP, 0 = GPTQ simmetrico); per esperto e per tensore (gate, up) vince il
    minimo errore d'uscita SIMMETRICO sulla meta' B fra i candidati con energia ||Q x||/||W x|| in [emin, emax], ed entra
    solo se batte la partenza."""
    gd = gd or {"emin": 0.9, "emax": 1.1, "peso_kw": 1.0}
    emin, emax = gd["emin"], gd["emax"]
    alfe = list(alfe) if isinstance(alfe, (list, tuple)) else [float(alfe)]
    t0 = time.time()
    tid = 45
    ng = Hd // 128
    nomi = {f: f"blk.{L}.ffn_{f}_exps.weight" for f in ("gate", "up")}
    raw = {f: np.array(raw_partenza(n)) for f, n in nomi.items()}            # copia: si sostituiscono gli accettati
    st = {}
    for h in "AB":
        Hs, Mxs, n = acc3.pool(h, gd["peso_kw"])
        st[h] = {"Hs": Hs.to(dev, dt), "Mxs": Mxs.to(dev, dt), "Gq": np.load(dl / f"gu_{h}" / "Gq.npy", mmap_mode="r"),
                 "C": np.load(dl / f"gu_{h}" / "C.npy", mmap_mode="r"), "conta": np.load(dl / f"gu_{h}" / "conta.npy"),
                 "peso": np.load(dl / f"gu_{h}" / "peso.npy")}
    acc = {"gate": 0, "up": 0}
    dJ = {"gate": [], "up": []}
    en_scelti = {"gate": [], "up": []}
    en_part = {"gate": [], "up": []}
    alfa_scelti = {"gate": {}, "up": {}}
    rif_energia = {"gate": 0, "up": 0}
    ripieghi = 0
    for ee in _lotti_esperti(es_lista, lotto):
        b = len(ee)
        W = Wgu32[ee].to(dt)                                                      # [b, 2I, H]
        Hk, Ck = {}, {}
        for h in "AB":
            s = st[h]
            Gq = spacchetta_lotto(torch.from_numpy(np.asarray(s["Gq"][ee])), Hd, dev, dt)
            Cc = torch.from_numpy(np.asarray(s["C"][ee])).to(dev, dt)
            n = torch.from_numpy(s["conta"][ee].astype(np.float64)).to(dev, dt).view(b, 1, 1)
            p = torch.from_numpy(s["peso"][ee]).to(dev, dt).view(b, 1, 1)
            pz = torch.where(p > 0, p, torch.ones_like(p))
            He = torch.where(p > 0, Gq / pz, s["Hs"].expand(b, Hd, Hd))
            Cs = W @ s["Mxs"]
            Ce = torch.where(p > 0, Cc / pz, Cs)
            Hk[h] = (n * He + kk * s["Hs"]) / (n + kk)
            Ck[h] = (n * Ce + kk * Cs) / (n + kk)
            del Gq, Cc, He, Ce, Cs
        HmA = RS.ruota_H_lotto(Hk["A"])
        Hmn, dbar = RS.normalizza(HmA)
        Minv, U, nr = U_Minv_lotto(Hmn)
        ripieghi += nr
        Wm = K.ruota(W)
        Wstar = (torch.bmm(K.ruota(Ck["A"]), Minv) / dbar.view(b, 1, 1)            # QEP a=1 (damp 1 % = quello di GPTQ)
                 if any(x != 0 for x in alfe) else None)
        HmB = RS.ruota_H_lotto(Hk["B"])
        T0 = torch.empty(b, 2 * I, Hd, dtype=torch.int8)
        S0 = torch.empty(b, 2 * I, ng, dtype=torch.float64)
        for j, e in enumerate(ee):
            for q, f in enumerate(("gate", "up")):
                t_, s_ = decodifica(raw[f][e], tid, Hd)
                T0[j, q * I:(q + 1) * I], S0[j, q * I:(q + 1) * I] = t_, s_
        Q0 = AL.ricostruisci(T0.to(dev), S0.to(dev, dt))
        J0 = RS.J_sim_lotto(Q0, HmB, Wm)
        N0 = RS.N_lotto(Q0, HmB)
        den = RS.N_lotto(Wm, HmB)                                               # ||W x_q||^2 per riga
        RR = [slice(q * I, (q + 1) * I) for q in range(2)]
        dd = {f: den[:, RR[q]].sum(1).clamp_min(1e-300) for q, f in enumerate(("gate", "up"))}
        mig = {f: J0[:, RR[q]].sum(1).clone() for q, f in enumerate(("gate", "up"))}     # da battere: la partenza
        sel = {f: [None] * b for f in ("gate", "up")}
        ep_ = {}
        for q, f in enumerate(("gate", "up")):
            ep = (N0[:, RR[q]].sum(1) / dd[f]).sqrt()
            ep_[f] = ep
            en_part[f] += [float(x) for x in ep]
        kw_jq = {"cache": {}} if getattr(jarq_lotto, "usa_cache", False) else {}     # VELOCE L2: una cache per lotto (stessa Hmn)
        for alfa in alfe:
            Wa = Wm if alfa == 0 else Wm + alfa * (Wstar - Wm)
            T, Sx = AL.gptq_bil_lotto(Wa, U, ng)
            T2, S2, _ = jarq_lotto(Wa, Hmn, T.unsqueeze(1), Sx.unsqueeze(1), giri=3, nu2=1e-4, **kw_jq)
            T2, S2 = T2[:, 0], S2[:, 0]
            Q1 = AL.ricostruisci(T2, S2.to(dt))
            J1 = RS.J_sim_lotto(Q1, HmB, Wm)
            N1 = RS.N_lotto(Q1, HmB)
            for q, f in enumerate(("gate", "up")):
                r = RR[q]
                j1 = J1[:, r].sum(1)
                en = (N1[:, r].sum(1) / dd[f]).sqrt()
                ok_en = (en >= emin) & (en <= emax)
                if gd.get("relativa", True):                       # non piu' lontano da 1 della partenza (energia_ok)
                    ok_en = ok_en | ((en > 0) & (en.log().abs() <= ep_[f].log().abs()))
                rif_energia[f] += int((~ok_en & (j1 < mig[f])).sum())
                ok = (j1 < mig[f]) & torch.isfinite(j1) & ok_en
                for j in range(b):
                    if bool(ok[j]):
                        mig[f][j] = j1[j]
                        sel[f][j] = (alfa, T2[j, r].clone(), S2[j, r].clone(), float(en[j]))
            del Wa, T, Sx, T2, S2, Q1, J1, N1
        for q, f in enumerate(("gate", "up")):
            j0 = J0[:, RR[q]].sum(1)
            for j, e in enumerate(ee):
                dJ[f].append(float((mig[f][j] - j0[j]) / dd[f][j]))
                if sel[f][j] is not None:
                    alfa, Tj, Sj, enj = sel[f][j]
                    raw[f][e] = codifica(Tj, Sj, tid, Hd)
                    acc[f] += 1
                    en_scelti[f].append(enj)
                    alfa_scelti[f][f"{alfa:g}"] = alfa_scelti[f].get(f"{alfa:g}", 0) + 1
        del W, Hk, Ck, HmA, Hmn, Minv, U, Wm, Wstar, HmB, Q0, J0, N0, sel
    for f, n in nomi.items():
        if acc[f]:
            nuovi[n] = raw[f]
            inietta(L, f, raw[f])
    md = {f: (float(np.median(v)) if v else None) for f, v in dJ.items()}
    mde = {f: (float(np.median(v)) if v else None) for f, v in en_scelti.items()}
    mdp = {f: (float(np.median(v)) if v else None) for f, v in en_part.items()}
    log(f"  esperti gate/up strato {L}: accettati gate {acc['gate']} up {acc['up']} su {len(es_lista)}; dSim_B/|Wx|^2 mediana "
        f"{md}; alfa scelti {alfa_scelti}; energia mediana scelti {mde} partenza {mdp}; rifiutati per energia "
        f"{rif_energia}; ripieghi fattorizzazione {ripieghi}; {time.time() - t0:.0f} s")
    return {"accettati_gate": acc["gate"], "accettati_up": acc["up"], "dJB_rel_mediana": md, "metrica": "simmetrica_B+energia",
            "alfe": alfe, "alfa_scelti": alfa_scelti, "energia_mediana_scelti": mde, "energia_mediana_partenza": mdp,
            "rifiutati_energia": rif_energia, "k": kk,
            "ripieghi_fattorizzazione": ripieghi, "secondi": round(time.time() - t0, 1),
            "conta_A_mediana": float(np.median(st["A"]["conta"])), "conta_A_min": int(st["A"]["conta"].min())}


def risolvi_esperti_down(L, adn, Wd_m, nuovi, inietta, codifica, decodifica, raw_partenza, es_lista, alfe, kk, lotto,
                         jarq_lotto, dev, dt, E, I, Hd, gd=None):
    """S3 down (TQ1_0, gruppo 256, nessuna rotazione) sull'ingresso VERO del down di C, gate/up gia' nuovi.
    14/10 G1/G2: come gate/up (candidati per alfa, guardia simmetrica + energia, per esperto)."""
    gd = gd or {"emin": 0.9, "emax": 1.1, "peso_kw": 1.0}
    emin, emax = gd["emin"], gd["emax"]
    alfe = list(alfe) if isinstance(alfe, (list, tuple)) else [float(alfe)]
    t0 = time.time()
    tid = 34
    ng = I // 256
    nome = f"blk.{L}.ffn_down_exps.weight"
    raw = np.array(raw_partenza(nome))
    st = {}
    for h in "AB":
        pt = adn.peso[h].sum().clamp_min(1e-30)
        st[h] = {"Hs": (adn.Hd[h].double().sum(0) / pt).to(dt), "Ms": (adn.Md[h].double().sum(0) / pt).to(dt)}
    acc, dJ, ripieghi = 0, [], 0
    en_scelti, en_part, alfa_scelti, rif_energia = [], [], {}, 0
    for ee in _lotti_esperti(es_lista, lotto):
        b = len(ee)
        W = Wd_m[ee].to(dev, dt)                                                  # [b, H, I]
        Hk, Ck = {}, {}
        for h in "AB":
            n = adn.conta[h][ee].to(dt).view(b, 1, 1)
            p = adn.peso[h][ee].to(dt).view(b, 1, 1)
            pz = torch.where(p > 0, p, torch.ones_like(p))
            He = torch.where(p > 0, adn.Hd[h][ee].to(dt) / pz, st[h]["Hs"].expand(b, I, I))
            Me = torch.where(p > 0, adn.Md[h][ee].to(dt) / pz, st[h]["Ms"].expand(b, I, I))
            Hk[h] = (n * He + kk * st[h]["Hs"]) / (n + kk)
            Ck[h] = W @ ((n * Me + kk * st[h]["Ms"]) / (n + kk))
        Hn, dbar = RS.normalizza(Hk["A"])
        Minv, U, nr = U_Minv_lotto(Hn)
        ripieghi += nr
        Wstar = torch.bmm(Ck["A"], Minv) / dbar.view(b, 1, 1) if any(x != 0 for x in alfe) else None
        T0 = torch.empty(b, Hd, I, dtype=torch.int8)
        S0 = torch.empty(b, Hd, ng, dtype=torch.float64)
        for j, e in enumerate(ee):
            T0[j], S0[j] = decodifica(raw[e], tid, I)
        Q0 = AL.ricostruisci(T0.to(dev), S0.to(dev, dt))
        HB_ = Hk["B"]
        j0 = RS.J_sim_lotto(Q0, HB_, W).sum(1)
        den = RS.N_lotto(W, HB_).sum(1).clamp_min(1e-300)
        ep_d = (RS.N_lotto(Q0, HB_).sum(1) / den).sqrt()
        en_part += [float(x) for x in ep_d]
        mig = j0.clone()
        sel = [None] * b
        kw_jq = {"cache": {}} if getattr(jarq_lotto, "usa_cache", False) else {}     # VELOCE L2: una cache per lotto (stessa Hn)
        for alfa in alfe:
            Wa = W if alfa == 0 else W + alfa * (Wstar - W)
            T, Sx = AL.gptq_bil_lotto(Wa, U, ng)
            T2, S2, _ = jarq_lotto(Wa, Hn, T.unsqueeze(1), Sx.unsqueeze(1), giri=3, nu2=1e-4, **kw_jq)
            T2, S2 = T2[:, 0], S2[:, 0]
            Q1 = AL.ricostruisci(T2, S2.to(dt))
            j1 = RS.J_sim_lotto(Q1, HB_, W).sum(1)
            en = (RS.N_lotto(Q1, HB_).sum(1) / den).sqrt()
            ok_en = (en >= emin) & (en <= emax)
            if gd.get("relativa", True):
                ok_en = ok_en | ((en > 0) & (en.log().abs() <= ep_d.log().abs()))
            rif_energia += int((~ok_en & (j1 < mig)).sum())
            ok = (j1 < mig) & torch.isfinite(j1) & ok_en
            for j in range(b):
                if bool(ok[j]):
                    mig[j] = j1[j]
                    sel[j] = (alfa, T2[j].clone(), S2[j].clone(), float(en[j]))
            del Wa, T, Sx, T2, S2, Q1
        for j, e in enumerate(ee):
            dJ.append(float((mig[j] - j0[j]) / den[j]))
            if sel[j] is not None:
                alfa, Tj, Sj, enj = sel[j]
                raw[e] = codifica(Tj, Sj, tid, I)
                acc += 1
                en_scelti.append(enj)
                alfa_scelti[f"{alfa:g}"] = alfa_scelti.get(f"{alfa:g}", 0) + 1
        del W, Hk, Ck, Hn, Minv, U, Wstar, Q0, sel
    if acc:
        nuovi[nome] = raw
        inietta(L, "down", raw)
    md = float(np.median(dJ)) if dJ else None
    mde = float(np.median(en_scelti)) if en_scelti else None
    mdp = float(np.median(en_part)) if en_part else None
    log(f"  esperti down strato {L}: accettati {acc} su {len(es_lista)}; dSim_B/|Wx|^2 mediana {md}; alfa scelti "
        f"{alfa_scelti}; energia mediana scelti {mde} partenza {mdp}; rifiutati per energia {rif_energia}; "
        f"ripieghi {ripieghi}; {time.time() - t0:.0f} s")
    return {"accettati": acc, "dJB_rel_mediana": md, "metrica": "simmetrica_B+energia", "alfe": alfe,
            "alfa_scelti": alfa_scelti, "energia_mediana_scelti": mde, "energia_mediana_partenza": mdp,
            "rifiutati_energia": rif_energia, "k": kk, "ripieghi_fattorizzazione": ripieghi,
            "secondi": round(time.time() - t0, 1), "conta_A_mediana": float(adn.conta["A"].double().median())}


if __name__ == "__main__":
    sys.exit(main())
