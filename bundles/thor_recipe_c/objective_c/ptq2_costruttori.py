#!/usr/bin/env python3
"""Obiettivo C, stadio 2 — seconda batteria PTQ sui 250 densi a UN piano (12/10/2026), SOLO CPU.
Azioni 2 e 3 di ~/tony-forgia/ricette/CENSIMENTO_LEVE_1012.md. Stesso bersaglio (bf16 puro), stessa H (manifesto R2B3,
N = 196.608), stesso formato e stesso registro di jarq_fedele.py / costruttori_ef.py; confronto PTQ contro PTQ con (c).

Varianti (cartella <out>/<nome>/tensori, manifesto come c0t_scrivi.py):
  p_bil1          CONTROLLO: GPTQ bilanciato K=1 SENZA JARQ (= avvio di (c), mai misurato da solo). Isola FOEM e JARQ.
  p_bil1foemR     FOEM (arXiv 2507.11017) CURATA: forma di riferimento degli autori (GPTQModel foem.py / mlx_foem.py),
                  beta 0,1 del lavoro. Diagnosi del 12/10: la forma "lavoro" di (f0) riportava verso W0 TUTTE le colonne
                  rimaste del blocco a ogni colonna -> (1-beta)^k, compensazione cancellata (errH x2, MMLU 25,30).
                  Il riferimento tocca solo la colonna successiva, una volta. FEDELE (griglia del formato: dichiarata).
  p_bil1foemRjarq (p_bil1foemR) + JARQ Alg. 1 fedele: variante Thor dichiarata (il lavoro non prevede JARQ dopo).
  p_bil1aos       GPTQ con ACT-ORDER A GRUPPI STATICI (IST-DASLab/gptq --act-order --static-groups, 2210.17323 repo) sulla
                  griglia bilanciata K=1 (D di ogni gruppo scelto dai pesi ORIGINALI come static_groups) + CRITERIO DEL
                  SEGNO da 2605.18933 (il lavoro non da' un algoritmo: variante Thor dichiarata): nella decisione di ogni
                  peso un livello che cambia il segno del bf16 costa pi/(pi-2) volte (Teorema 3: energia trasversale di un
                  cambio di segno / perturbazione di ampiezza a pari norma); sulle colonne con leva R_j = n a_j^2 (1 - 4 a_j
                  /(3 pi)) >= 10 (App. E; ~10x misurato a risposta lineare) il cambio di segno e' vietato (si preferisce 0).
                  a_j^2 = H_jj / tr H (ampiezza dell'ingresso normalizzato).  Si registra anche p_bil1ao (senza segno).
  p_bil1aosjarq   (p_bil1aos) + JARQ Alg. 1 fedele (variante; nessuno dei due lavori la vieta).
  p_bil1drq       DRQ (arXiv 2610.11226, Alg. 1 + App. C) sui codici di BIL K=1, scale fisse: obiettivo
                  L(W^) + alpha S_C(W^), C = (1-beta) H + beta I, H = X X^T / n (H GREZZA del manifesto, come nel lavoro),
                  (alpha, beta) = (1, 0,005) e tolleranza 1 % = impostazione del lavoro per QuaRot + GPTQ a W2 (App. D.2,
                  la piu' vicina: base ruotata Hadamard, sotto 3 bit); 8 sostituzioni per riga + coppie, frazioni
                  {1/32..1}, 3 giri. Scostamenti dichiarati: insieme di ricerca = H intera (196.608 token) invece di 512
                  vettori; insiemi di controllo = H di ALTRI corpora (-cot, -grande) per attn* e ffn_gate/up_shexp, nessuno
                  per attn_output/ssm_out/ffn_down_shexp (solo ricerca); (alpha, beta) presi dal lavoro, non tarati su Pile.
  p_bil1jarqdrq   DRQ sui codici di (c) = BIL K=1 + JARQ: e' l'ordine del lavoro (DRQ dopo il metodo PTQ di base, che qui e'
                  (c)). JARQ DOPO DRQ non si fa: rimetterebbe al minimo la ricostruzione sulla calibrazione, cioe' l'opposto
                  di cio' che DRQ fa (Fig. 1, Teorema 3.1).
Differenze ereditate (R1/R2 di jarq_fedele): X del bf16, gruppi del formato (128 ruotati / 256), scala fp16.
uso: HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=4 nice -n 10 ptq2_costruttori.py [--out DIR] [--solo nome,...] [--prova]
"""
import argparse, hashlib, json, math, os, sys, time
sys.path.insert(0, "@@ROOT:models@@/build/leve/raffina")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import comune as CM
import solutori as SO
import jarq_fedele as JF

F64 = torch.float64
FOEM_BETA = 0.1
KAPPA_SEGNO = math.pi / (math.pi - 2.0)       # 2605.18933 Teorema 3
R_SEGNO_VIETATO = 10.0                         # 2605.18933 App. E / Exp. E: leva ~10x a risposta lineare
DRQ_ALPHA, DRQ_BETA, DRQ_TOL, DRQ_GIRI, DRQ_K = 1.0, 0.005, 0.01, 3, 8
DRQ_FRAZ = (1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0)
DT = (-2.0, -1.0, 1.0, 2.0)
DTt = torch.tensor(DT, dtype=torch.float64)
TT = "@@ROOT:models@@/gguf/tony-tern"
CONTROLLO_H = {"attn": (f"{TT}/hessiane-attn-cot/H_{{L:02d}}.npy", f"{TT}/hessiane-attn-grande/H_{{L:02d}}.npy"),
               "ffn": (f"{TT}/hessiane-ffn-cot/H_{{L:02d}}.npy", f"{TT}/hessiane-ffn-grande/H_{{L:02d}}.npy")}
VARIANTI = ("p_bil1", "p_bil1foemR", "p_bil1foemRjarq", "p_bil1aos", "p_bil1aosjarq", "p_bil1drq", "p_bil1jarqdrq")


# ───────────────────────────── act-order a gruppi statici (+ criterio del segno) ─────────────────────────────
def gptq_ao_statico(W, H, ng, segno=False, W0=None, damp=0.01, blocksize=128):
    """GPTQ fasterquant con act-order (colonne per diag(H) decrescente) e static_groups (scala di ogni gruppo CONTIGUO
    fissata prima, dai pesi originali), sulla griglia bilanciata K=1 {-D, 0, +D}. Ritorna (T [1,R,C] int8, S [1,R,ng])."""
    R, C = W.shape
    G = C // ng
    tab = SO.tabella_trit(1)
    Wd = W.to(F64)
    S = torch.zeros(1, R, ng, dtype=F64)
    for g in range(ng):                                     # static_groups: find_params sui pesi originali del gruppo
        S[0, :, g] = SO.bil_scale(SO.bil_scegli_D(Wd[:, g * G:(g + 1) * G], 1, tab), 1)[:, 0]
    perm = torch.argsort(torch.diagonal(H), descending=True)
    Wk = Wd[:, perm].clone()
    Hp = H.to(F64)[perm][:, perm]
    _, U, _ = SO.cholesky_smorzata(Hp, percdamp=damp)
    T = torch.zeros(1, R, C, dtype=torch.int8)
    if segno:
        s0 = torch.sign((W0 if W0 is not None else Wd)[:, perm])
        dg = torch.diagonal(H).to(F64)
        a = (dg / dg.sum()).clamp_min(0).sqrt()
        Rlev = (C * a * a * (1 - 4 * a / (3 * math.pi)))[perm]
        vietato = Rlev >= R_SEGNO_VIETATO
    lv = torch.tensor([-1.0, 0.0, 1.0], dtype=F64)
    for i1 in range(0, C, blocksize):
        i2 = min(i1 + blocksize, C)
        Err = torch.zeros(R, i2 - i1, dtype=F64)
        for i in range(i1, i2):
            j = int(perm[i])
            s = S[0, :, j // G]
            w = Wk[:, i]
            costo = (w.unsqueeze(1) - s.unsqueeze(1) * lv.unsqueeze(0)) ** 2            # [R, 3]
            if segno:
                flip = (lv.unsqueeze(0) != 0) & (lv.unsqueeze(0) != s0[:, i:i + 1])
                costo = torch.where(flip, costo * (math.inf if bool(vietato[i]) else KAPPA_SEGNO), costo)
            t = costo.argmin(1) - 1
            T[0, :, j] = t.to(torch.int8)
            e = (w - s * t.to(F64)) / U[i, i]
            Wk[:, i:i2] -= e.unsqueeze(1) * U[i, i:i2].unsqueeze(0)
            Err[:, i - i1] = e
        if i2 < C:
            Wk[:, i2:] -= Err @ U[i1:i2, i2:]
    return T, S


# ───────────────────────────── DRQ (2610.11226) ─────────────────────────────
class _Peggiore:
    """S_C(W^) = max_{u^T C u = 1} ||E u||^2 = lambda_max(L^-1 E^T E L^-T), C = L L^T."""

    def __init__(self, H, beta):
        C = (1 - beta) * H + beta * torch.eye(H.shape[0], dtype=F64)
        self.L = torch.linalg.cholesky(0.5 * (C + C.T))
        self.v = None

    def __call__(self, E):
        A = torch.linalg.solve_triangular(self.L, E.T, upper=False)                  # L^-1 E^T  [C, R]
        M = A @ A.T
        M = 0.5 * (M + M.T)
        ok = False
        try:                                                    # Krylov (LOBPCG) + Rayleigh-Ritz, come App. E.3
            X0 = self.v.unsqueeze(1) if self.v is not None else None
            ev, V = torch.lobpcg(M, k=1, X=X0, largest=True, tol=1e-10, niter=600)
            lam, v = float(ev[0]), V[:, 0] / V[:, 0].norm()
            ok = math.isfinite(lam) and float((M @ v - lam * v).norm()) <= 1e-6 * max(lam, 1e-300)
        except Exception:
            ok = False
        if not ok:
            ev, V = torch.linalg.eigh(M)
            lam, v = float(ev[-1]), V[:, -1]
        self.v = v / v.norm()
        u = torch.linalg.solve_triangular(self.L.T, self.v.unsqueeze(1), upper=True).squeeze(1)
        return lam, u


def _L(E, H):
    return float(((E @ H) * E).sum())


def drq(W, Hs, Hcs, T0, S, alpha=DRQ_ALPHA, beta=DRQ_BETA, tol=DRQ_TOL, giri=DRQ_GIRI, kk=DRQ_K):
    """Alg. 1 di 2610.11226 sui codici ternari (scale fisse). Hs = H di ricerca (grezza), Hcs = H di controllo."""
    R, C = W.shape
    ng = S.shape[-1]
    G = C // ng
    Wd = W.to(F64)
    sc = S[0].to(F64).repeat_interleave(G, dim=1)                                    # [R, C] scala per elemento
    pegg = _Peggiore(Hs, beta)
    T = T0.clone()
    E0 = sc * T[0].to(F64) - Wd
    L0 = _L(E0, Hs)
    L0c = [_L(E0, h) for h in Hcs]
    lam, u = pegg(E0)
    J = L0 + alpha * lam
    cat = {"L0": L0, "S0": lam, "J0": J, "giri": []}
    dH = torch.diagonal(Hs)
    for r in range(giri):
        E = sc * T[0].to(F64) - Wd
        Gm = E @ Hs                                                                   # [R, C]
        p = E @ u                                                                     # [R]
        T64 = T[0].to(F64)
        dJs = []
        for dt in DT:                                                                 # salti verso gli altri due codici
            tn = T64 + dt
            d = sc * dt
            dJ = 2 * d * Gm + d * d * dH + alpha * (2 * d * p.unsqueeze(1) * u + d * d * u * u)
            dJs.append(torch.where((tn >= -1) & (tn <= 1), dJ, torch.full_like(dJ, math.inf)))
        DJ = torch.stack(dJs, -1).reshape(R, C * len(DT))                             # 2 alternative valide per peso
        k = min(kk, C)
        vals, idx = DJ.topk(k, dim=1, largest=False)                                  # le 8 sostituzioni singole migliori
        col, dsalto = idx // len(DT), DTt[idx % len(DT)]                              # [R, k]
        dv = dsalto * sc.gather(1, col)                                               # variazione di valore
        Hab = Hs[col.unsqueeze(2), col.unsqueeze(1)]                                  # [R, k, k]
        uab = u[col].unsqueeze(2) * u[col].unsqueeze(1)
        cop = vals.unsqueeze(2) + vals.unsqueeze(1) + 2 * dv.unsqueeze(2) * dv.unsqueeze(1) * (Hab + alpha * uab)
        valida = torch.triu(torch.ones(k, k, dtype=torch.bool), 1).unsqueeze(0) & (col.unsqueeze(2) != col.unsqueeze(1))
        cop = torch.where(valida & torch.isfinite(cop), cop, torch.full_like(cop, math.inf))
        cmin, carg = cop.reshape(R, -1).min(1)
        smin = vals[:, 0]
        usa_cop = cmin < smin
        dec = torch.where(usa_cop, cmin, smin)                                        # variazione (negativa = guadagno)
        righe = torch.nonzero(dec < 0).squeeze(1)
        if righe.numel() == 0:
            cat["giri"].append({"giro": r + 1, "righe": 0}); break
        ordine = righe[torch.argsort(dec[righe])]
        a_i = torch.where(usa_cop, carg // k, torch.zeros_like(carg))
        b_i = carg % k
        prop = []
        for f in DRQ_FRAZ:
            n = max(1, int(math.ceil(f * ordine.numel())))
            if prop and prop[-1][0] == n:
                continue
            rr = ordine[:n]
            Tn = T.clone()
            ca, da = col[rr, a_i[rr]], dsalto[rr, a_i[rr]]
            Tn[0, rr, ca] = (Tn[0, rr, ca].to(F64) + da).to(torch.int8)
            rc = rr[usa_cop[rr]]
            if rc.numel():
                cb, db = col[rc, b_i[rc]], dsalto[rc, b_i[rc]]
                Tn[0, rc, cb] = (Tn[0, rc, cb].to(F64) + db).to(torch.int8)
            assert int(Tn.abs().max()) <= 1
            En = sc * Tn[0].to(F64) - Wd
            Ln = _L(En, Hs)
            okc = Ln <= (1 + tol) * L0 and all(_L(En, h) <= (1 + tol) * l0 for h, l0 in zip(Hcs, L0c))
            if not okc:
                prop.append((n, None, None, None)); continue
            lam_n, u_n = pegg(En)
            Jn = Ln + alpha * lam_n
            prop.append((n, Jn, Tn, u_n) if Jn < J else (n, None, None, None))
        valide = [x for x in prop if x[1] is not None]
        cat["giri"].append({"giro": r + 1, "righe": int(righe.numel()), "provate": [x[0] for x in prop],
                            "valide": [x[0] for x in valide]})
        if not valide:
            break
        n, J, T, u = min(valide, key=lambda x: x[1])
        cat["giri"][-1]["scelta"] = n
        pegg(sc * T[0].to(F64) - Wd)                                                  # riallinea la direzione calda
    E = sc * T[0].to(F64) - Wd
    cat.update({"L": _L(E, Hs), "J": J, "Lc0": L0c, "Lc": [_L(E, h) for h in Hcs],
                "trit_cambiati": float((T != T0).double().mean())})
    return T, cat


def h_controllo(nome, rotated):
    L = int(nome.split(".")[1])
    fam = nome.split(".")[2]
    chiave = "attn" if fam in ("attn_gate", "attn_qkv", "attn_q", "attn_k", "attn_v") else (
        "ffn" if fam in ("ffn_gate_shexp", "ffn_up_shexp") else None)
    out = []
    for f in (CONTROLLO_H.get(chiave) or ()):
        H = torch.from_numpy(np.load(f.format(L=L))).double()
        H = 0.5 * (H + H.T)
        if rotated:
            H = CM.ruota_H(H)
        out.append(H)
    return out


def una(d, varianti, rif_c=None):
    W, H, c, prov = JF.bersaglio_bf16_e_H(d.nome, d.rotated)
    ng = d.S.shape[-1]
    t0 = time.time()
    m, out, tm = {}, {}, {}
    def e(T, S):
        return JF.errH(W, T, S, H)
    t = time.time()
    Tb, Sb, _ = SO.gptq_bilanciato(W, H, 1, ng)
    Sb = SO.fp16(Sb)
    m["errH_bil1"] = e(Tb, Sb); out["p_bil1"] = (Tb, Sb); tm["bil1"] = time.time() - t
    need_c = "p_bil1jarqdrq" in varianti or rif_c is not None
    if need_c:
        t = time.time()
        Tc, Sc, _ = JF.jarq_fedele(W, H, Tb, Sb, c)
        m["errH_bil1jarq"] = e(Tc, Sc); tm["c"] = time.time() - t
    if "p_bil1foemR" in varianti or "p_bil1foemRjarq" in varianti:
        t = time.time()
        Tf, Sf, _ = SO.gptq_bilanciato(W, H, 1, ng, foem_beta=FOEM_BETA, foem_forma="riferimento")
        Sf = SO.fp16(Sf)
        m["errH_bil1foemR"] = e(Tf, Sf); out["p_bil1foemR"] = (Tf, Sf)
        m["trit_cambiati_foemR_da_bil"] = float((Tf != Tb).double().mean())
        if "p_bil1foemRjarq" in varianti:
            Tfj, Sfj, _ = JF.jarq_fedele(W, H, Tf, Sf, c)
            m["errH_bil1foemRjarq"] = e(Tfj, Sfj); out["p_bil1foemRjarq"] = (Tfj, Sfj)
        tm["foemR"] = time.time() - t
    if "p_bil1aos" in varianti or "p_bil1aosjarq" in varianti:
        t = time.time()
        Ta, Sa = gptq_ao_statico(W, H, ng, segno=False)
        m["errH_bil1ao"] = e(Ta, Sa)
        Tas, Sas = gptq_ao_statico(W, H, ng, segno=True, W0=W)
        m["errH_bil1aos"] = e(Tas, Sas); out["p_bil1aos"] = (Tas, Sas)
        nz = Tas[0] != 0
        m["segni_cambiati_aos"] = float((nz & (Tas[0].to(F64) != torch.sign(W))).double().sum() / nz.sum().clamp_min(1))
        nz = Ta[0] != 0
        m["segni_cambiati_ao"] = float((nz & (Ta[0].to(F64) != torch.sign(W))).double().sum() / nz.sum().clamp_min(1))
        if "p_bil1aosjarq" in varianti:
            Taj, Saj, _ = JF.jarq_fedele(W, H, Tas, Sas, c)
            m["errH_bil1aosjarq"] = e(Taj, Saj); out["p_bil1aosjarq"] = (Taj, Saj)
        tm["ao"] = time.time() - t
    if "p_bil1drq" in varianti or "p_bil1jarqdrq" in varianti:
        t = time.time()
        Hs = H * prov["diag_media_grezza"]                                            # H = X X^T / n (grezza)
        Hcs = h_controllo(d.nome, d.rotated)
        if "p_bil1drq" in varianti:
            Td, cd = drq(W, Hs, Hcs, Tb, Sb)
            m["errH_bil1drq"] = e(Td, Sb); out["p_bil1drq"] = (Td, Sb); m["drq_bil1"] = cd
        if "p_bil1jarqdrq" in varianti:
            Tcd, ccd = drq(W, Hs, Hcs, Tc, Sc)
            m["errH_bil1jarqdrq"] = e(Tcd, Sc); out["p_bil1jarqdrq"] = (Tcd, Sc); m["drq_c"] = ccd
        m["n_H_controllo"] = len(Hcs)
        tm["drq"] = time.time() - t
    for v, (T, S) in out.items():
        assert T.dtype == torch.int8 and int(T.abs().max()) <= 1 and T.shape[0] == 1, f"{d.nome} {v}: codici fuori da {{-1,0,1}}"
        assert torch.isfinite(S).all() and torch.equal(S, SO.fp16(S)), f"{d.nome} {v}: scale non fp16 o non finite"
    m["tempi"] = {k: round(x, 1) for k, x in tm.items()}
    m["secondi"] = round(time.time() - t0, 1)
    return {v: out[v] for v in varianti if v in out}, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{TT}/OBIETTIVO_C/varianti_ptq2")
    ap.add_argument("--solo", default="")
    ap.add_argument("--varianti", default=",".join(VARIANTI))
    ap.add_argument("--registro", default="registro.jsonl")
    a = ap.parse_args()
    varianti = [v for v in a.varianti.split(",") if v]
    assert all(v in VARIANTI for v in varianti), varianti
    os.makedirs(a.out, exist_ok=True)
    reg = os.path.join(a.out, a.registro)
    fatti = {json.loads(l)["nome"] for l in open(reg)} if os.path.exists(reg) else set()
    rif_c = {}
    pc = f"{TT}/OBIETTIVO_C/varianti_fedeli/registro.jsonl"
    for l in open(pc):
        r = json.loads(l); rif_c[r["nome"]] = (r.get("errH_bil1"), r.get("errH_bil1jarq"))
    rif_f0 = {}
    for l in open(f"{TT}/OBIETTIVO_C/varianti_fedeli/registro_ef.jsonl"):
        r = json.loads(l); rif_f0[r["nome"]] = r.get("errH_bil1foem")
    src = CM.gp.GGUFPlaneSource(CM.R2B3M, layers=list(range(40)))
    nomi = CM.elenco_densi(src)
    if a.solo:
        nomi = [n for n in nomi if n in a.solo.split(",")]
    CM.log(f"ptq2: {len(nomi)} densi, gia' fatti {len(fatti)}, varianti {varianti}, threads {torch.get_num_threads()}")
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
        m["rif_errH_bil1_c"], m["rif_errH_bil1jarq_c"] = rif_c.get(nome, (None, None))
        m["rif_errH_bil1foem_f0"] = rif_f0.get(nome)
        with open(reg, "a") as f:
            f.write(json.dumps(m) + "\n")
        g = lambda k: f"{m[k]:.4f}" if m.get(k) is not None else "-"
        CM.log(f"{i+1}/{len(nomi)} {nome}: bil1 {g('errH_bil1')} (rif {g('rif_errH_bil1_c')}) c {g('errH_bil1jarq')} "
               f"(rif {g('rif_errH_bil1jarq_c')}) | foemR {g('errH_bil1foemR')} (f0 {g('rif_errH_bil1foem_f0')}) +jarq "
               f"{g('errH_bil1foemRjarq')} | ao {g('errH_bil1ao')} aos {g('errH_bil1aos')} +jarq {g('errH_bil1aosjarq')} | "
               f"drq {g('errH_bil1drq')} c+drq {g('errH_bil1jarqdrq')} | {m['tempi']} {m['secondi']} s")
    CM.log("fine ptq2")


if __name__ == "__main__":
    main()
