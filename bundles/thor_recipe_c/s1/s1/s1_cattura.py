#!/usr/bin/env python3
"""S1 tappe 2-3 (GPU): statistiche del FLUSSO DI C, due flussi sugli stessi token, strato per strato.

  flusso del maestro  PulitoHF (sequenziale_thor.py: safetensors bf16 di Ornith-1.5, uno strato alla volta);
  flusso di C         lo studente E2E-C1/C3 (e2e_studente sha ac482f3c, densi ternari «tutte», ssm_alpha/beta ternari
                      con E2E_DENSI_SSM_AB=1) sull'OSPITE di C: lo stesso modello di cui si misura l'MMLU.
Il bug A_log di cattura_schur_esperti.py (P10, AUDIT_RIPRODUZIONE) NON c'e': qui il flusso quantizzato e' lo studente
che ha passato i cancelli d'identita' di E2E-C3/CARDAN, non il caricatore torch di quel file. Da lui si riprende solo
lo schema dell'accumulo per esperto (G_raw, M_raw, peso con i pesi VERI del router).

Per ogni strato L e meta' (A o B, una passata per meta': la memoria dei flussi si dimezza):
  densi    H_q per sorgente (M, Kv, Kw) ai 4 punti d'ingresso, H_f del maestro sulla sola M (controllo contro il
           manifesto), M = sum x_f x_q^T (termine incrociato di GPTAQ 2504.02692 / QEP 2504.09629), somme dei vettori;
  esperti  Gq_e = sum_{t in e} w_te x_q x_q^T (AGQ di MoEQuant 2505.03804), C_e = W_ref,e[gate;up] sum w x_f x_q^T
           (bersaglio asimmetrico), H del down per esperto sull'ingresso VERO del down di C (a = silu(g) * u, fp16 come
           il kernel), conteggi e somme dei pesi (per il restringimento k ~ 8000 verso la H di strato, F1 di T2).
Ordine (true-sequential non serve: si CATTURA, non si quantizza): per ogni lotto, maestro e C leggono lo stesso
ingresso dello strato e scrivono la propria uscita.
Cancelli (fail-closed): (1) l'uscita dello strato di C coi ganci e la toppa del down e' IDENTICA al bit a quella senza;
(2) il top-8 ricalcolato coincide con quello dello studente (ogni lotto); (3) sul maestro F.linear(ingresso catturato)
riproduce l'uscita del modulo (scarto massimo registrato, come FAIR-Calib).
Ripresa: strato con FATTO saltato; i flussi si ricostruiscono ripassando gli strati gia' fatti (nessun flusso su disco).
"""
from __future__ import annotations

import argparse
import gc
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
sys.path.insert(0, str(QUI))
import s1_comune as SC  # noqa: E402


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def elenco(s: str, n: int) -> list:
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


def libero_gib(p: Path) -> float:
    st = os.statvfs(p)
    return st.f_bavail * st.f_frsize / 2**30


class Accumulatori:
    def __init__(self, L, tipo, dims, E, I, H, dev, esperti: bool):
        self.L, self.tipo, self.dims, self.dev = L, tipo, dims, dev
        f64 = torch.float64
        self.Hq = {(p, s): torch.zeros(n, n, dtype=f64, device=dev) for p, n in dims.items() for s in SC.SORGENTI}
        self.Hf = {p: torch.zeros(n, n, dtype=f64, device=dev) for p, n in dims.items()}
        self.Mx = {p: torch.zeros(n, n, dtype=f64, device=dev) for p, n in dims.items()}
        self.sq = {(p, s): torch.zeros(n, dtype=f64, device=dev) for p, n in dims.items() for s in SC.SORGENTI}
        self.sf = {p: torch.zeros(n, dtype=f64, device=dev) for p, n in dims.items()}
        self.tok = {s: 0 for s in SC.SORGENTI}
        self.esperti = esperti
        if esperti:
            self.Gq = torch.zeros(E, H, H, dtype=torch.float32, device=dev)
            self.C = torch.zeros(E, 2 * I, H, dtype=torch.float32, device=dev)
            self.Hd = torch.zeros(E, I, I, dtype=torch.float32, device=dev)
            self.conta = torch.zeros(E, dtype=torch.int64, device=dev)
            self.peso = torch.zeros(E, dtype=torch.float64, device=dev)

    def densi(self, s, cp, cq):
        n = None
        for p in self.dims:
            xq = cq[p].reshape(-1, self.dims[p]).float()
            xf = cp[p].reshape(-1, self.dims[p]).float()
            n = xq.shape[0]
            self.Hq[(p, s)] += (xq.T @ xq).double()
            self.Mx[p] += (xf.T @ xq).double()
            self.sq[(p, s)] += xq.sum(0).double()
            self.sf[p] += xf.sum(0).double()
            if s == "M":
                self.Hf[p] += (xf.T @ xf).double()
        self.tok[s] += n

    def moe(self, xq, xf, w, k, a_ord, ordine, offs, Wgu):
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
            Xw = Xq * ww
            self.Gq[e] += Xw.T @ Xq
            Wg = Wgu[e] if Wgu.dtype == torch.float32 else Wgu[e].float()
            self.C[e] += (xf[t] @ Wg.T).mul_(ww).T @ Xq
            A = a_ord[s0:s1].float()
            self.Hd[e] += (A * ww).T @ A
            self.conta[e] += s1 - s0
            self.peso[e] += ww.double().sum()

    def salva(self, d: Path, perm_ssm, info: dict):
        tmp = Path(str(d) + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        nt = sum(self.tok.values())
        for p, n in self.dims.items():
            perm = None
            if p == "att_out" and self.tipo == "linear_attention":
                perm = torch.from_numpy(perm_ssm).long().to(self.dev)

            def ord_(M):
                if perm is None:
                    return M
                return M[perm][:, perm] if M.dim() == 2 else M[perm]
            for s in SC.SORGENTI:
                if self.tok[s]:
                    np.save(tmp / f"{p}_Hq_{s}.npy", SC.impacchetta(ord_(self.Hq[(p, s)] / self.tok[s])).float().cpu().numpy())
                    np.save(tmp / f"{p}_sq_{s}.npy", ord_(self.sq[(p, s)]).cpu().numpy())
            if self.tok["M"]:
                np.save(tmp / f"{p}_Hf_M.npy", SC.impacchetta(ord_(self.Hf[p] / self.tok["M"])).float().cpu().numpy())
            np.save(tmp / f"{p}_Mx.npy", ord_(self.Mx[p] / nt).float().cpu().numpy())
            np.save(tmp / f"{p}_sf.npy", ord_(self.sf[p]).cpu().numpy())
        if self.esperti:
            E = self.Gq.shape[0]
            for nome, X, tri in (("Gq", self.Gq, True), ("C", self.C, False), ("Hdown", self.Hd, True)):
                if tri:
                    sc = torch.diagonal(X, dim1=1, dim2=2).abs().amax(1)
                else:
                    sc = X.abs().flatten(1).amax(1)
                sc = torch.where(sc > 0, sc, torch.ones_like(sc))
                forma = (E, X.shape[-1] * (X.shape[-1] + 1) // 2) if tri else tuple(X.shape)
                mm = np.lib.format.open_memmap(tmp / f"esperti_{nome}.npy", mode="w+", dtype=np.float16, shape=forma)
                for e0 in range(0, E, 16):
                    Y = X[e0:e0 + 16] / sc[e0:e0 + 16].view(-1, 1, 1)
                    mm[e0:e0 + 16] = (SC.impacchetta(Y) if tri else Y).half().cpu().numpy()
                mm.flush()
                del mm
                np.save(tmp / f"esperti_{nome}_scala.npy", sc.cpu().numpy().astype(np.float32))
            np.save(tmp / "esperti_conta.npy", self.conta.cpu().numpy())
            np.save(tmp / "esperti_peso.npy", self.peso.cpu().numpy())
            Hds = self.Hd.sum(0).double() / self.peso.sum().clamp_min(1e-30)
            np.save(tmp / "Hdown_strato.npy", SC.impacchetta(Hds).float().cpu().numpy())
            c = self.conta.cpu().numpy()
            info["esperti"] = {"conta_min": int(c.min()), "conta_mediana": float(np.median(c)), "conta_max": int(c.max()),
                               "sotto_2000": int((c < 2000).sum()), "sotto_10000": int((c < 10000).sum()),
                               "peso_totale": float(self.peso.sum())}
        info["token"] = dict(self.tok)
        info["dims"] = dict(self.dims)
        info["ssm_out_in_ordine_gguf"] = self.tipo == "linear_attention"
        SC.scrivi_json(tmp / "info.json", info)
        (tmp / "FATTO").write_text(time.strftime("%FT%T") + "\n")
        shutil.rmtree(d, ignore_errors=True)
        os.replace(tmp, d)


class Ganci:
    """Ganci sui 4 punti d'ingresso (stessi moduli nel maestro HF e nello studente: lo studente sostituisce i Linear
    con LineareTernaria agli STESSI percorsi). Per C anche l'ingresso del MoE (pre-gancio) e l'ingresso del down
    degli esperti (toppa su FamigliaEsperti.apply, riconosciuta per identita' della famiglia)."""

    def __init__(self, strato, full: bool, moe=False, S=None, fam_down=None):
        self.d = {}
        self.h = []
        m_att = strato.self_attn.o_proj if full else strato.linear_attn.out_proj
        self.h.append(strato.input_layernorm.register_forward_hook(lambda m, i, o: self.d.__setitem__("attn", o.detach())))
        self.h.append(strato.post_attention_layernorm.register_forward_hook(
            lambda m, i, o: self.d.__setitem__("ffn", o.detach())))
        self.h.append(m_att.register_forward_pre_hook(lambda m, i: self.d.__setitem__("att_out", i[0].detach())))
        self.h.append(strato.mlp.shared_expert.down_proj.register_forward_pre_hook(
            lambda m, i: self.d.__setitem__("down", i[0].detach())))
        self.m_att, self.m_down = m_att, strato.mlp.shared_expert.down_proj
        self.S, self.orig = S, None
        if not moe:
            self.h.append(m_att.register_forward_hook(lambda m, i, o: self.d.__setitem__("att_out_o", o.detach())))
            self.h.append(self.m_down.register_forward_hook(lambda m, i, o: self.d.__setitem__("down_o", o.detach())))
            return
        self.h.append(strato.mlp.register_forward_pre_hook(lambda m, i: self.d.__setitem__("moe_x", i[0].detach())))
        FE = S.FamigliaEsperti
        self.grezzo = FE.__dict__.get("apply")          # di norma assente: apply e' il classmethod di Function
        self.orig = FE.apply
        orig = self.orig
        d = self.d

        def apply(x, dd, fam, piano, *a, **k):
            if fam is fam_down:
                d["a_ord"] = x.detach()
                d["piano"] = piano
            return orig(x, dd, fam, piano, *a, **k)
        FE.apply = apply

    def chiudi(self):
        for h in self.h:
            h.remove()
        if self.orig is not None:
            if self.grezzo is None:
                del self.S.FamigliaEsperti.apply
            else:
                self.S.FamigliaEsperti.apply = self.grezzo
            self.orig = None


def router(mlp, x):
    """Stessa matematica di MoEStudente.forward (router fp32, top-k, rinormalizzazione)."""
    if mlp.router_w32 is not None and x.dtype not in (torch.float32, torch.float64):
        logit = F.linear(x.float(), mlp.router_w32)
    else:
        logit = F.linear(x, mlp.router_w)
    prob = torch.softmax(logit, dim=-1, dtype=torch.float32 if x.dtype != torch.float64 else torch.float64)
    idx = torch.topk(prob, mlp.top_k, dim=-1).indices
    w = prob.gather(-1, idx)
    return idx, w / w.sum(-1, keepdim=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ospite", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta", choices=("A", "B"), required=True)
    ap.add_argument("--thor-v2", required=True)
    ap.add_argument("--fedeli", required=True)
    ap.add_argument("--ricetta", default="B3_ind32")
    ap.add_argument("--motore", default="triton")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--strati", default="")
    ap.add_argument("--strati-esperti", default="", help="strati con le statistiche per esperto (vuoto = tutti)")
    ap.add_argument("--lotto-token", type=int, default=8192)
    ap.add_argument("--max-seq", type=int, default=0, help="prova: sequenze per sorgente")
    ap.add_argument("--tronca", type=int, default=0, help="prova: token per sequenza")
    ap.add_argument("--tetto", type=float, default=44.0)
    ap.add_argument("--margine-disco", type=float, default=60.0)
    a = ap.parse_args()
    sys.path.insert(0, a.thor_v2)
    sys.path.insert(0, a.fedeli)
    import e2e_studente as S
    import sequenziale_thor as ST
    dev = torch.device(a.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    out = Path(a.out) / a.meta
    out.mkdir(parents=True, exist_ok=True)
    t_avvio = time.time()
    perm_ssm = dict(np.load(Path(a.thor_v2) / "mappa_completa_gguf_hf_perm.npz"))["linear.ssm_out.weight"]
    st = S.Studente(gguf=Path(a.ospite), strati=range(40), ricetta=a.ricetta, motore_nome=a.motore, device=str(dev),
                    densi_ternari="tutte", log=log)
    nL = len(st.cfg.layer_types)
    strati = elenco(a.strati, nL)
    assert strati == list(range(strati[0], strati[-1] + 1)) and strati[0] == 0, "strati: un prefisso contiguo da 0"
    con_esperti = set(elenco(a.strati_esperti, nL))
    if dev.type == "cuda":
        st.carica_embedding()
    P = ST.PulitoHF(st.cfg, str(dev))
    cfg = st.cfg
    E, I, Hd = int(cfg.num_experts), int(cfg.moe_intermediate_size), int(cfg.hidden_size)
    dati = torch.load(a.corpus, map_location="cpu", weights_only=False)
    ids = {}
    for s in SC.SORGENTI:
        v = dati["sorgenti"][s][dati["meta"][s][a.meta]].long()
        if a.max_seq:
            v = v[: a.max_seq]
        if a.tronca:
            v = v[:, : a.tronca]
        ids[s] = v
    hp, hq, ctx = {}, {}, {}
    for s, v in ids.items():
        N, T = v.shape
        B = max(1, a.lotto_token // T)
        hp[s] = torch.empty(N, T, Hd, dtype=P.dt, device=dev)
        hq[s] = torch.empty(N, T, Hd, dtype=st.ric.dtype_residuo, device=dev)
        for i in range(0, N, B):
            hp[s][i:i + B] = P.embed(v[i:i + B])
            hq[s][i:i + B] = st.embedding(v[i:i + B]).to(hq[s].dtype)
        ctx[s] = {}
        for b in sorted({B, N % B} - {0}):
            ctx[s][b] = (st._contesto(b, T, st.ric.dtype_residuo), st._contesto(b, T, P.dt))
        log(f"meta' {a.meta} sorgente {s}: {N} x {T} = {N * T} token, lotti da {B}")

    def lotti(s):
        N = ids[s].shape[0]
        B = max(1, a.lotto_token // ids[s].shape[1])
        for i in range(0, N, B):
            sl = slice(i, min(i + B, N))
            yield sl, ctx[s][sl.stop - sl.start]

    @torch.no_grad()
    def passa(L):
        for s in SC.SORGENTI:
            for sl, (cq, cp) in lotti(s):
                hp[s][sl] = P.avanti(L, hp[s][sl], cp).to(hp[s].dtype)
                hq[s][sl] = st._strato(L, hq[s][sl], cq).to(hq[s].dtype)

    fatti = [L for L in strati if (out / f"strato_{L:02d}" / "FATTO").exists()]
    L0 = 0
    while L0 in fatti:
        L0 += 1
    for L in range(L0):
        passa(L)
        log(f"ripresa: strato {L} ripassato")
    # spazio: stima per strato (densi + esperti) dalla dimensione nota
    dims_tipo = {"attn": Hd, "ffn": Hd, "att_out": 4096, "down": int(cfg.shared_expert_intermediate_size)}
    gib_densi = sum((3 * n * (n + 1) / 2 + n * (n + 1) / 2 + n * n) * 4 for n in dims_tipo.values()) / 2**30
    gib_esp = E * (Hd * (Hd + 1) / 2 + 2 * I * Hd + I * (I + 1) / 2) * 2 / 2**30
    resto = [L for L in strati if L >= L0]
    serve = sum(gib_densi + (gib_esp if L in con_esperti else 0) for L in resto)
    log(f"disco: servono ~{serve:.1f} GiB per {len(resto)} strati (densi {gib_densi:.2f}, esperti {gib_esp:.2f} per strato); "
        f"liberi {libero_gib(out):.0f}")
    if libero_gib(out) < serve + a.margine_disco:
        log(f"ERRORE: spazio insufficiente ({libero_gib(out):.0f} < {serve:.0f} + margine {a.margine_disco})")
        return 3
    controlli = {}
    for L in resto:
        tl = time.time()
        tipo = cfg.layer_types[L]
        full = tipo == "full_attention"
        sp, sq_ = P.strato(L), st.moduli[L]
        mlp = sq_.mlp
        esp = L in con_esperti
        dims = dict(dims_tipo)
        acc = Accumulatori(L, tipo, dims, E, I, Hd, dev, esp)
        fam_down = st.esperti(L).fam["down"]
        Wgu = ST.modulo(sp, "mlp.experts").gate_up_proj.detach()
        if dev.type == "cuda":
            Wgu = Wgu.float()
        c_L = {"cancello_bit": None, "top8_diversi": 0, "maestro_att_max_abs": 0.0, "maestro_down_max_abs": 0.0}
        primo = True
        for s in SC.SORGENTI:
            for sl, (cq, cp) in lotti(s):
                gp = Ganci(sp, full)
                gq = Ganci(sq_, full, moe=True, S=S, fam_down=fam_down)
                try:
                    with torch.no_grad():
                        op = P.avanti(L, hp[s][sl], cp)
                        oq = st._strato(L, hq[s][sl], cq)
                finally:
                    gp.chiudi()
                    gq.chiudi()
                if primo:                                            # cancello (1): ganci e toppa non cambiano nulla
                    with torch.no_grad():
                        oq2 = st._strato(L, hq[s][sl], cq)
                        oq3 = st._strato(L, hq[s][sl], cq)
                    c_L["cancello_bit"] = bool(torch.equal(oq, oq2))
                    rum = float((oq2.float() - oq3.float()).abs().max())       # non determinismo proprio del motore
                    dif = float((oq.float() - oq2.float()).abs().max())
                    c_L["cancello_scarto"], c_L["rumore_motore"] = dif, rum
                    if dif > max(rum, 0.0):
                        raise RuntimeError(f"strato {L}: i ganci cambiano l'uscita di C (scarto {dif:.3g}, rumore {rum:.3g})")
                    del oq2, oq3
                    primo = False
                for k, m, o in (("att_out", gp.m_att, "att_out_o"), ("down", gp.m_down, "down_o")):
                    r = (F.linear(gp.d[k], m.weight, m.bias).float() - gp.d[o].float()).abs().max()
                    c_L[f"maestro_{'att' if k == 'att_out' else 'down'}_max_abs"] = max(
                        c_L[f"maestro_{'att' if k == 'att_out' else 'down'}_max_abs"], float(r))
                acc.densi(s, gp.d, gq.d)
                xq = gq.d["moe_x"].reshape(-1, Hd)
                idx, w = router(mlp, xq)
                if mlp.ultimo_idx is not None and not torch.equal(idx, mlp.ultimo_idx):   # cancello (2)
                    c_L["top8_diversi"] += int((idx != mlp.ultimo_idx).any(-1).sum())
                if esp:
                    pz = gq.d["piano"]
                    acc.moe(xq.float(), gp.d["ffn"].reshape(-1, Hd).float(), w, mlp.top_k, gq.d["a_ord"],
                            pz.ordine, pz.offs, Wgu)
                hp[s][sl] = op.to(hp[s].dtype)
                hq[s][sl] = oq.to(hq[s].dtype)
                del gp, gq, op, oq
                if dev.type == "cuda":
                    v = torch.cuda.memory_reserved() / 2**30
                    if v > a.tetto:
                        raise RuntimeError(f"VRAM {v:.1f} GiB > tetto {a.tetto}")
        if c_L["top8_diversi"]:
            raise RuntimeError(f"strato {L}: top-8 ricalcolato diverso da quello dello studente su "
                               f"{c_L['top8_diversi']} token")
        info = {"strato": L, "tipo": tipo, "meta": a.meta, "ospite": a.ospite, "controlli": c_L,
                "esperti_catturati": esp, "secondi": None}
        info["secondi"] = round(time.time() - tl, 1)
        acc.salva(out / f"strato_{L:02d}", perm_ssm, info)
        controlli[L] = c_L
        msg = (f"S1 cattura meta' {a.meta} strato {L} ({'full' if full else 'GDN '}) {info['secondi']:.0f} s · token "
               f"{acc.tok} · esperti {info.get('esperti', {})} · cancelli {c_L}")
        log(msg)
        del acc, Wgu
        gc.collect()
        if dev.type == "cuda":
            torch.cuda.empty_cache()
            log(f"  VRAM picco {torch.cuda.max_memory_reserved() / 2**30:.1f} GiB")
    if strati == list(range(nL)) and all((out / f"strato_{L:02d}" / "FATTO").exists() for L in range(nL)):
        (out / "FATTO").write_text(f"{time.strftime('%FT%T')} {(time.time() - t_avvio) / 60:.1f} min\n")
    log(f"meta' {a.meta}: strati {resto[0] if resto else '-'}..{strati[-1]} in {(time.time() - t_avvio) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
