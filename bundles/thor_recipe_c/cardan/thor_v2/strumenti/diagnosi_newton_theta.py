#!/usr/bin/env python3
"""Diagnosi §19 del registro (TECNICHE_E2E_PV.md) prima di costruire il Newton diagonale stile Sophia sulle scale theta.
27/9/2026, ramo newton-theta. SOLO CPU, sola lettura del GGUF e dello stato di EB2 (mai la GPU).

Domanda: il passo di Newton diagonale |g / h| sulle scale theta e' ~ lr = 1e-3 (Adam e' gia' alla scala di Newton:
non vale la pena) o molto piu' grande (margine per passi piu' lunghi)?

  h_theta del gruppo (e, r, g) = d^2 * sum_s dY_{s,r}^2 u_{s,r,g}^2,  u_{s,r,g} = sum_{c in g} X_{s,c} t_{r,c}
  (Gauss-Newton campionata per token: dY dall'indietro di un vettore sqrt(c) (p - e_y), y ~ p dello studente, come
  la curvatura del passo V, e2e_pv_codici.py:17-21; registro §6 punto 36).

Due sottocomandi:
  strato   curvatura ESATTA su strati VERI (studente torch su CPU, token veri dell'held-out): h_tok (formula sopra),
           l'approssimazione disponibile dagli accumulatori del passo V (d^2 * nnz * acc2[cand], acc2 = sum dY^2 X^2
           sul solo codice candidato) e i fattori che la separano: kappa1 = h_tok / (d^2 sum_c t_c^2 acc2_c) (termini
           incrociati fra colonne del gruppo), kappa2 = sum_c t_c^2 acc2_c / (nnz acc2[cand]) (colonna candidata contro
           la media), kappa_tok = E_y[(sum_s dY u)^2] / E_y[h_tok] (termini incrociati fra TOKEN: Gauss-Newton piena,
           stimatore GNB di Sophia), kappa_riga = curvatura del moto coerente di tutti i gruppi di una riga / somma
           delle diagonali (accoppiamento fra gruppi della stessa uscita).
  stato    distribuzione di |m^ / h| sullo stato VERO di EB2 (campione estratto: m, v di Adam in unita' K, H della
           finestra del passo V, theta), con h = kappa * d^2 * nnz * H_passo (kappa dal sottocomando strato).

Terzo sottocomando, `estrai`: il campione dello stato vero (sola lettura, mmap) che `stato` legge.

Uso (limiti del flusso):
    HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 nice -n 15 @@ROOT:venv@@/bin/python3 \\
        strumenti/diagnosi_newton_theta.py strato --strati 0,3 --righe 4 --token 512 --campioni 8 --uscita X.json
    ... diagnosi_newton_theta.py estrai --ripresa @@ROOT:lavoro@@/e2e/allena/EB2/ripresa_000015.pt \\
            --accumulo @@ROOT:lavoro@@/e2e/allena/EB2/accumulo_pv_000015.pt --uscita campione_eb2_p15.npz
    ... diagnosi_newton_theta.py stato --campione campione_eb2_p15.npz --kappa X.json --uscita Y.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

QUI = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(QUI))
import e2e_studente as S  # noqa: E402  (sola lettura)

CORPUS_HELD = Path("@@ROOT:models@@/gguf/tony-tern/calibrazione/e2e/e2e_heldout_4096_tutte.pt")
QUANTILI = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)


def _q(x: np.ndarray) -> list:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return [float(np.quantile(x, p)) for p in QUANTILI] if x.size else []


# ══════════════════════════════════════ strato: curvatura esatta ══════════════════════════════════════
class MotoreMisura(S.MotoreTorch):
    """Motore torch dello studente; con `attivo` il `dd` NON restituisce il gradiente (zeri: theta.grad intatto) ma
    accumula, per gruppo (E, R, NG) in float64, le curvature della diagnosi (un campione y = una chiamata per famiglia).
    `d_attuale[(L, nome)]` = scala del forward (fp16 esatta) per il moto coerente di riga."""
    nome = "torch+misura"

    def __init__(self):
        self.attivo = False
        self.acc: dict = {}
        self.d_attuale: dict = {}

    def _buf(self, fam) -> dict:
        k = (int(fam.strato), fam.nome)
        if k not in self.acc:
            z = lambda: torch.zeros(fam.E, fam.R, fam.NG, dtype=torch.float64)  # noqa: E731
            self.acc[k] = {"htok": z(), "coldiag": z(), "a2cand": z(), "g2": z(), "nnz": z(),
                           "hriga": torch.zeros(fam.E, fam.R, dtype=torch.float64),
                           "hriga_diag": torch.zeros(fam.E, fam.R, dtype=torch.float64),
                           "tok": torch.zeros(fam.E, dtype=torch.float64), "campioni": 0}
        return self.acc[k]

    def dd(self, gy, x, fam, piano, d_dtype):
        if not self.attivo:
            return super().dd(gy, x, fam, piano, d_dtype)
        b = self._buf(fam)
        b["campioni"] += 1
        d = self.d_attuale[(int(fam.strato), fam.nome)].to(torch.float64)
        o = piano.offs.tolist()
        R, NG, gs = fam.R, fam.NG, fam.gs
        for e in range(fam.E):
            if o[e + 1] <= o[e]:
                continue
            g = gy[o[e]:o[e + 1]].to(torch.float64)                  # (s, R)
            xe = x[o[e]:o[e + 1]].to(torch.float64)                   # (s, C)
            t = S.trit_esperto(fam, e).to(torch.float64)              # (R, C)
            acc = g.T @ xe                                            # dL^/dW (R, C)
            a2 = (g * g).T @ (xe * xe)                                # sum_s dY^2 X^2 (R, C): acc2 del modo 3
            tg = t.view(R, NG, gs)
            b["g2"][e] += ((tg * acc.view(R, NG, gs)).sum(-1)) ** 2   # (dL^/dd)^2: GNB (con i termini fra token)
            b["coldiag"][e] += (tg * tg * a2.view(R, NG, gs)).sum(-1)
            # colonna candidata con la regola della proposta del passo V (e2e_pv_codici.proposta_da_dw)
            A = acc.view(R, NG, gs)
            pend = A * torch.sign(fam.d0[e].to(torch.float64)).unsqueeze(-1)
            dz = -torch.sign(pend)
            tn = tg + dz
            fatt = (dz != 0) & (tn >= -1) & (tn <= 1)
            ci = torch.where(fatt, A.abs(), torch.full_like(A, -1.0)).argmax(-1, keepdim=True)
            b["a2cand"][e] += a2.view(R, NG, gs).gather(-1, ci).squeeze(-1)
            u = torch.einsum("sgj,rgj->srg", xe.view(-1, NG, gs), tg)  # (s, R, NG)
            g2 = g * g
            b["htok"][e] += torch.einsum("sr,srg->rg", g2, u * u)
            ud = u * d[e].unsqueeze(0)                                # (s, R, NG): contributo del gruppo a Y
            b["hriga"][e] += (g2 * ud.sum(-1) ** 2).sum(0)
            b["hriga_diag"][e] += torch.einsum("sr,srg->r", g2, ud * ud)
            if b["campioni"] == 1:
                b["nnz"][e] = (tg != 0).sum(-1).to(torch.float64)
                b["tok"][e] = float(o[e + 1] - o[e])
        return torch.zeros(fam.E, fam.R, fam.NG, dtype=d_dtype, device=gy.device)


def _righe_held(n: int, T: int) -> list:
    d = torch.load(CORPUS_HELD, weights_only=True)
    out = []
    for dom in d["nomi_domini"]:
        X = d["domini"][dom]
        for r in range(X.shape[0]):
            out.append((dom, X[r, :T].to(torch.int64)))
    passo = max(1, len(out) // n)
    return [out[i * passo] for i in range(n)]


def testa_studente(st) -> torch.Tensor:
    from dequant_gguf_torch import _dequant_raw
    t = st._tens["output.weight"]
    return torch.from_numpy(_dequant_raw(np.asarray(t.data), t)).to(torch.float32)


def misura_strato(strati: list, n_righe: int, T: int, campioni: int, seme: int, log=print) -> dict:
    t0 = time.time()
    st = S.Studente(strati=strati, ricetta="B_ind32", motore_nome="torch", device="cpu", log=log)
    mm = MotoreMisura()
    for L in st.strati:
        st.esperti(L).mot = mm
        for n in S.FAMIGLIE:
            mm.d_attuale[(L, n)] = st.esperti(L).scala(n).detach()
    st.mot = mm
    WS = testa_studente(st)
    righe = _righe_held(n_righe, T)
    ids = torch.stack([r for _, r in righe])
    log(f"studente {strati}: {time.time() - t0:.1f} s; righe {[dm for dm, _ in righe]} x {T} token")
    gen = torch.Generator().manual_seed(seme)
    n_tok = ids.numel()
    rad_c = math.sqrt(1.0 / n_tok)                                    # perdita media per token del micro-lotto
    with torch.no_grad():
        stato = st.avanti(ids)
    for k in range(campioni):
        t1 = time.time()
        xL = stato["xL"].detach().requires_grad_(True)
        with torch.enable_grad():
            hS = st.esce(xL)
        gh = torch.empty(hS.shape, dtype=torch.float32)
        with torch.no_grad():
            for i in range(0, hS.shape[0], 256):
                p = torch.softmax(hS[i:i + 256].detach().float() @ WS.T, dim=-1)
                y = torch.multinomial(p, 1, generator=gen).squeeze(1)
                p[torch.arange(p.shape[0]), y] -= 1.0
                gh[i:i + 256] = (p @ WS) * rad_c
                del p
        with torch.enable_grad():
            hS.backward(gh.to(hS.dtype))
        mm.attivo = True
        st.indietro(stato, xL.grad)
        mm.attivo = False
        log(f"campione {k + 1}/{campioni}: {time.time() - t1:.1f} s")
    for p in st.parametri():
        assert p.grad is None or float(p.grad.abs().max()) == 0.0, "la misura ha toccato theta.grad"
    return riassumi(mm, st, time.time() - t0)


def riassumi(mm: MotoreMisura, st, secondi: float) -> dict:
    out = {"secondi": secondi, "famiglie": {}}
    for (L, n), b in sorted(mm.acc.items()):
        c = b["campioni"]
        f = st.esperti(L).fam[n]
        d = mm.d_attuale[(L, n)].to(torch.float64)
        vivi = (b["tok"] > 0).view(-1, 1, 1) & (f.d0 != 0) & (b["nnz"] > 0) & (b["coldiag"] > 0) & (b["a2cand"] > 0)
        k1 = (b["htok"] / b["coldiag"])[vivi]
        k2 = (b["coldiag"] / (b["nnz"] * b["a2cand"]))[vivi]
        ktok = (b["g2"] / b["htok"])[vivi]
        rv = (b["tok"] > 0).view(-1, 1) & (b["hriga_diag"] > 0)
        kriga = (b["hriga"] / b["hriga_diag"])[rv]
        htheta = (d * d * b["htok"] / c)[vivi]
        # rapporti di SOMME (meno rumorosi dei rapporti per gruppo, pesati per curvatura)
        out["famiglie"][f"{L}.{n}"] = {
            "campioni": c, "gruppi": int(vivi.sum()), "gs": f.gs, "NG": f.NG,
            "kappa1_q": _q(k1.numpy()), "kappa1_somme": float(b["htok"][vivi].sum() / b["coldiag"][vivi].sum()),
            "kappa2_q": _q(k2.numpy()),
            "kappa2_somme": float(b["coldiag"][vivi].sum() / (b["nnz"] * b["a2cand"])[vivi].sum()),
            "kappa_tok_q": _q(ktok.numpy()), "kappa_tok_somme": float(b["g2"][vivi].sum() / b["htok"][vivi].sum()),
            "kappa_riga_q": _q(kriga.numpy()), "kappa_riga_somme": float(b["hriga"][rv].sum() / b["hriga_diag"][rv].sum()),
            "kappa_tot_somme": float(b["htok"][vivi].sum() / (b["nnz"] * b["a2cand"])[vivi].sum()),
            "h_theta_per_campione_q": _q(htheta.numpy()),
        }
    return out


# ══════════════════════════════════════ stato: |m^/h| su EB2 ══════════════════════════════════════
def diagnosi_stato(campione: Path, kappa: dict | None, log=print) -> dict:
    from gguf import GGUFReader
    z = np.load(campione)
    meta = json.loads(Path(str(campione).replace(".npz", ".json")).read_text())
    NE, K, lr, eps = meta["NE"], meta["K"], meta["lr"], meta["eps_adam"]
    b1, b2 = meta["betas"]
    t = meta["steps"][0]
    n_micro = 16
    Hscala = n_micro / meta["n_curvatura"]                      # somma della finestra -> curvatura del PASSO
    tens = {x.name: x for x in GGUFReader(str(S.GGUF_THOR_T)).tensors}
    fam_k = {}
    if kappa:
        for n in S.FAMIGLIE:
            vs = [v["kappa_tot_somme"] for k_, v in kappa["famiglie"].items() if k_.endswith("." + n)]
            fam_k[n] = float(np.exp(np.mean(np.log(vs)))) if vs else 1.0
    tutti = {n: [] for n in S.FAMIGLIE}
    somma_h = {n: [0.0, 0] for n in S.FAMIGLIE}
    per_strato = []
    for L in range(40):
        for n in S.FAMIGLIE:
            k = f"{L}.{n}"
            f = S.famiglia_da_tensore(tens[f"blk.{L}.{S.FAMIGLIE[n]}.weight"], n, esperti=slice(0, NE))
            nnz = torch.stack([(S.trit_esperto(f, e) != 0).view(f.R, f.NG, f.gs).sum(-1) for e in range(NE)]).numpy()
            d0 = f.d0.numpy().astype(np.float64)
            d = (d0 * np.exp(z[f"{k}|theta"].astype(np.float64))).astype(np.float16).astype(np.float64)
            m = z[f"{k}|m"].astype(np.float64) / (1 - b1 ** t)
            v = z[f"{k}|v"].astype(np.float64) / (1 - b2 ** t)
            H = z[f"{k}|H"].astype(np.float64) * Hscala
            vive = (d0 != 0) & (z[f"{k}|prop"] != 0) & (H > 0) & (np.abs(m) > 0)
            h = fam_k.get(n, 1.0) * d * d * nnz * H                   # unita' K, come m
            somma_h[n][0] += float(h.sum()) / K
            somma_h[n][1] += h.size
            newton = np.abs(m[vive]) / h[vive]
            passo_adam = lr * np.abs(m[vive]) / (np.sqrt(v[vive]) + eps)
            snr = np.abs(m[vive]) / np.sqrt(v[vive])
            # gradiente di SEGNALE: m^2 - var(m) (EMA beta1 dopo t passi di rumore bianco di varianza ~ v): stima di
            # |g_vero| per gruppo (0 dove il rumore spiega tutto)
            w = (1 - b1) ** 2 * sum(b1 ** (2 * j) for j in range(int(t))) / (1 - b1 ** t) ** 2
            seg = np.sqrt(np.clip(m[vive] ** 2 - w * v[vive], 0, None))
            tutti[n].append(np.stack([newton, passo_adam, snr, seg / h[vive], np.abs(z[f"{k}|theta"][vive])]))
            per_strato.append({"L": L, "fam": n, "gruppi": int(vive.sum()), "newton_mediana": float(np.median(newton)),
                               "snr_mediana": float(np.median(snr)), "quota_segnale_zero": float((seg == 0).mean())})
        log(f"strato {L}: {per_strato[-1]}")
    tot = {}
    for n, xs in tutti.items():
        a = np.concatenate(xs, 1)
        tot[n] = {"gruppi": int(a.shape[1]), "kappa": fam_k.get(n, 1.0),
                  "newton_m_q": _q(a[0]), "newton_segnale_q": _q(a[3]), "passo_adam_q": _q(a[1]), "snr_q": _q(a[2]),
                  "theta_abs_q": _q(a[4]), "var_m_su_v": w,
                  "quota_newton_m_gt_lr": float((a[0] > lr).mean()), "quota_newton_m_gt_10lr": float((a[0] > 10 * lr).mean()),
                  "quota_newton_segnale_gt_lr": float((a[3] > lr).mean()),
                  "quota_segnale_zero": float((a[3] == 0).mean())}
    # accoppiamento: la ricerca lineare di §2 (theta = alpha theta_2 dopo 2 passi di Adam, |theta_2| ~ 1,5e-3 per
    # coordinata) misura la curvatura LUNGO la direzione di Adam: b = 0,00443 per alpha^2 (alpha* = 1,65). La somma delle
    # sole diagonali per lo stesso spostamento e' sum_i h_i delta^2 (popolazione intera: 40 strati x 256 esperti).
    gruppi = {"gate": 40 * 256 * 512 * 16, "up": 40 * 256 * 512 * 16, "down": 40 * 256 * 2048 * 2}
    b_diag = sum(somma_h[n][0] / max(somma_h[n][1], 1) * gruppi[n] for n in S.FAMIGLIE) * 1.5e-3 ** 2
    acc = {"h_media_vera": {n: somma_h[n][0] / max(somma_h[n][1], 1) for n in S.FAMIGLIE}, "b_diag": b_diag,
           "b_ricerca_lineare": 0.00443, "rapporto": 0.00443 / b_diag if b_diag > 0 else None,
           "nota": "b dalla ricerca lineare di §2 (passo 2, KL); h al passo 15 (CAKLD): ordine di grandezza"}
    log(f"accoppiamento: {json.dumps(acc)}")
    return {"meta": meta, "per_strato": per_strato, "totale": tot, "accoppiamento": acc}


def estrai(ripresa: Path, accumulo: Path, uscita: Path, NE: int = 16, log=print) -> None:
    """Campione (esperti 0..NE-1 di ogni famiglia di ogni strato) dallo stato VERO di una corsa, in sola lettura (mmap):
    theta, m e v di Adam (unita' K), EMA; buffer della finestra PV (cand, prop, S, Q, H) e d_finestra. 27/9: da
    ripresa_000015.pt + accumulo_pv_000015.pt di EB2 (finestra 2/4, 4 micro-lotti di curvatura)."""
    r = torch.load(ripresa, map_location="cpu", weights_only=False, mmap=True)
    a = torch.load(accumulo, map_location="cpu", weights_only=False, mmap=True)
    out, passi = {}, set()
    for i, k in enumerate(r["theta"]):
        s = r["opt"]["state"][i]
        passi.add(float(s["step"]))
        out[f"{k}|theta"] = r["theta"][k][:NE].clone().numpy()
        out[f"{k}|m"] = s["exp_avg"][:NE].clone().numpy()
        out[f"{k}|v"] = s["exp_avg_sq"][:NE].clone().numpy()
        if r.get("ema") is not None:
            out[f"{k}|ema"] = r["ema"][k][:NE].clone().numpy()
        for n in ("cand", "prop", "S", "Q", "H"):
            out[f"{k}|{n}"] = a["buffer"][k][n][:NE].clone().numpy()
    meta = {"steps": sorted(passi), "K": r["stato"]["g1"]["K"], "eps_adam": r["stato"]["g1"]["eps_adam"],
            "lr": r["opt"]["param_groups"][0]["lr"], "betas": list(r["opt"]["param_groups"][0]["betas"]),
            "n_verifica": a["n_verifica"], "n_propose": a["n_propose"], "n_curvatura": a["n_curvatura"],
            "passi_nella_finestra": a["passi_nella_finestra"], "NE": NE, "passo": r["stato"]["passo"],
            "ripresa": str(ripresa), "accumulo": str(accumulo)}
    np.savez(uscita, **out)
    Path(str(uscita).replace(".npz", ".json")).write_text(json.dumps(meta, indent=1, default=str))
    log(json.dumps(meta, default=str))


def main() -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("estrai")
    e.add_argument("--ripresa", required=True)
    e.add_argument("--accumulo", required=True)
    e.add_argument("--esperti", type=int, default=16)
    e.add_argument("--uscita", required=True)
    a = sp.add_parser("strato")
    a.add_argument("--strati", default="0,3")
    a.add_argument("--righe", type=int, default=4)
    a.add_argument("--token", type=int, default=512)
    a.add_argument("--campioni", type=int, default=8)
    a.add_argument("--seme", type=int, default=20260927)
    a.add_argument("--uscita", required=True)
    b = sp.add_parser("stato")
    b.add_argument("--campione", required=True)
    b.add_argument("--kappa", default=None)
    b.add_argument("--uscita", required=True)
    x = ap.parse_args()
    torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "4")))
    if x.cmd == "estrai":
        estrai(Path(x.ripresa), Path(x.accumulo), Path(x.uscita), x.esperti)
        return 0
    if x.cmd == "strato":
        r = misura_strato([int(v) for v in x.strati.split(",")], x.righe, x.token, x.campioni, x.seme,
                          log=lambda s: print(s, flush=True))
    else:
        kap = json.loads(Path(x.kappa).read_text()) if x.kappa else None
        r = diagnosi_stato(Path(x.campione), kap, log=lambda s: print(s, flush=True))
    Path(x.uscita).write_text(json.dumps(r, indent=1))
    print(json.dumps(r.get("famiglie", r.get("totale")), indent=1)[:6000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
