#!/usr/bin/env python3
"""e2e_pv_misure — i numeri del progetto `e2e_progetto_pv_codici.md` (passo V a sottospazio sui codici di Thor-T),
misurati sui BYTE VERI del GGUF. 26/9/2026 sera. Solo CPU (4 fili, nice), nessun file esistente modificato:
SOLA LETTURA di `e2e_codici_scale` (decode verificato), `e2e_esporta._deq_c` (codice C di llama.cpp, ctypes),
del codec di casa (S1024) e del GGUF.

Cosa misura:
  M1  codifica TQ1_0 del byte (5 cifre in qs, 4 in qh + cifra nulla): decode(encode(q)) == q per ogni q, e un
      salto di UNA cifra di +-1 cambia UN solo byte e UNA sola cifra (esaustivo, tutti i q e tutte le cifre).
  M2  per ogni famiglia e strato 0..39: codici, gruppi, trit -1/0/+1, gruppi a d = 0, gruppi senza trit non nulli,
      byte non canonici (valore del byte != codifica canonica delle sue cifre), e per OGNI matrice d'esperto il
      bilancio del trust ratio: tau(rho) = rho^2 * sum_g d_g^2 nnz_g / media_g(d_g^2) (salti di un livello).
  M3  salto di codici sui byte veri (strati 0 e 39, gate e down, esperti 0 e 255): posizioni a caso, salti fattibili,
      byte riscritti con l'aritmetica di q; decode verificato == trit attesi (tutti gli altri identici) e codice C
      di llama.cpp sul blocco riscritto == trit' x d (de-ruotato per TQ1_0R_1024), AL BIT.
  M4  memoria dei buffer del passo V (dalle forme vere).

Uso:
    export XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
    systemd-run --user --scope -p CPUQuota=400% -p MemoryMax=6G nice -n 10 \\
      env CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=4 \\
      @@ROOT:venv@@/bin/python3 e2e_pv_misure.py --uscita @@ROOT:lavoro@@/e2e/progetto_pv/misure_byte.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
sys.path.insert(0, "@@ROOT:models@@/build/llamacpp-tq1-k1024/gguf-py")
sys.path.insert(0, "@@ROOT:forge@@/kernel-rotazione")

from gguf import GGUFReader  # noqa: E402
import e2e_codici_scale as CS  # noqa: E402

GGUF = Path(os.getenv("E2E_GGUF", "@@ROOT:models@@/gguf/tony-tern/Thor-v2-ThorT-scor150.gguf"))
FAM = {"gate": "ffn_gate_exps", "up": "ffn_up_exps", "down": "ffn_down_exps"}
P3 = (1, 3, 9, 27, 81)


def cifra(v: int, n: int) -> int:
    """Cifra n (0 = la piu' significativa) del byte v: STESSA aritmetica di dequantize_row_tq1_0 (senza il -1)."""
    return (((v * P3[n]) & 255) * 3) >> 8


def codifica(q: int) -> int:
    return ((q * 256 + 242) // 243) & 255


# ─────────────────────────────── M1 ───────────────────────────────
def m1_codifica() -> dict:
    esito = {}
    for regione, nc in (("qs", 5), ("qh", 4)):
        biiez = salti = salti_ok = 0
        for cif in np.ndindex(*(3,) * nc):
            q = sum(c * 3 ** (4 - n) for n, c in enumerate(cif))
            v = codifica(q)
            if tuple(cifra(v, n) for n in range(nc)) == cif:
                biiez += 1
            for n in range(nc):
                for delta in (-1, 1):
                    c2 = cif[n] + delta
                    if not 0 <= c2 <= 2:
                        continue
                    salti += 1
                    v2 = codifica(q + delta * 3 ** (4 - n))
                    atteso = tuple(c2 if m == n else cif[m] for m in range(nc))
                    salti_ok += tuple(cifra(v2, m) for m in range(nc)) == atteso
        canon = [codifica(sum(cifra(v, n) * 3 ** (4 - n) for n in range(nc))) == v for v in range(256)]
        esito[regione] = {"codici": 3 ** nc, "biiezione_ok": biiez, "salti": salti, "salti_ok": salti_ok,
                          "byte_non_canonici": [v for v in range(256) if not canon[v]]}
    return esito


# ─────────────────────────────── M2 ───────────────────────────────
def regioni(tipo: str) -> tuple:
    if tipo == "TQ1_0":
        return [slice(0, 48)], [slice(48, 52)]
    return [slice(q * 48, (q + 1) * 48) for q in range(4)], [slice(192 + q * 4, 192 + (q + 1) * 4) for q in range(4)]


def m2_tensore(t, canon_qs: np.ndarray, canon_qh: np.ndarray, pezzo: int = 32) -> dict:
    tipo = t.tensor_type.name
    nb, qk, off, k = CS.LAYOUT[tipo]
    gs = qk // k
    raw = np.asarray(t.data)                               # (E, R, NBR)
    E, R, NBR = raw.shape
    C = NBR // nb * qk
    cont = dict(codici=0, gruppi=0, neg=0, zero=0, pos=0, gruppi_d0=0, codici_in_d0=0, gruppi_vuoti=0,
                gruppi_pieni=0, byte_qs=0, byte_qh=0, qs_non_canon=0, qh_non_canon=0)
    tau = []            # per esperto: tau(rho=0,01) = 1e-4 * S2 / media(d^2 sui gruppi vivi)
    nnz_e = []
    rqs, rqh = regioni(tipo)
    for e0 in range(0, E, pezzo):
        e1 = min(E, e0 + pezzo)
        blocchi_np = np.ascontiguousarray(raw[e0:e1]).reshape(-1, nb)
        for s in rqs:
            h = np.bincount(blocchi_np[:, s].ravel(), minlength=256)
            cont["byte_qs"] += int(h.sum())
            cont["qs_non_canon"] += int(h[~canon_qs].sum())
        for s in rqh:
            h = np.bincount(blocchi_np[:, s].ravel(), minlength=256)
            cont["byte_qh"] += int(h.sum())
            cont["qh_non_canon"] += int(h[~canon_qh].sum())
        B = torch.from_numpy(blocchi_np)
        T = CS.trit_da_blocchi(B, tipo)                    # (Nb, qk) int8
        d = CS.scale_da_blocchi(B, tipo).to(torch.float64)  # (Nb, k)
        Tg = T.view(-1, k, gs)
        nz = (Tg != 0).sum(-1).to(torch.float64)           # (Nb, k)
        cont["codici"] += T.numel()
        cont["gruppi"] += nz.numel()
        cont["neg"] += int((T == -1).sum())
        cont["pos"] += int((T == 1).sum())
        cont["zero"] += int((T == 0).sum())
        d0 = d == 0
        cont["gruppi_d0"] += int(d0.sum())
        cont["codici_in_d0"] += int(d0.sum()) * gs
        cont["gruppi_vuoti"] += int((nz == 0).sum())
        cont["gruppi_pieni"] += int((nz == gs).sum())
        ne = e1 - e0
        d2 = (d * d).view(ne, -1)
        nzv = nz.view(ne, -1)
        vivi = (d2 > 0)
        S2 = (d2 * nzv).sum(1)
        m2 = (d2 * vivi).sum(1) / vivi.sum(1).clamp(min=1)
        tau.append((1e-4 * S2 / m2.clamp(min=1e-300)).numpy())
        nnz_e.append(nzv.sum(1).numpy())
        del T, Tg, d, B, blocchi_np
    tau = np.concatenate(tau)
    nnz_e = np.concatenate(nnz_e)
    q = lambda a, p: float(np.quantile(a, p))  # noqa: E731
    return {"tipo": tipo, "E": E, "R": R, "C": C, "gs": gs, "gruppi_per_esperto": R * C // gs, **cont,
            "tau_rho001_per_esperto": {"min": q(tau, 0), "p10": q(tau, .1), "mediana": q(tau, .5), "p90": q(tau, .9),
                                       "max": q(tau, 1), "somma": float(tau.sum())},
            "tau_su_nnz_mediana": float(np.median(tau / np.maximum(nnz_e, 1))),
            "nnz_per_esperto": {"min": q(nnz_e, 0), "mediana": q(nnz_e, .5), "max": q(nnz_e, 1)}}


# ─────────────────────────────── M3 ───────────────────────────────
def posizione(c: int, tipo: str) -> tuple:
    """Colonna c di una riga -> (blocco nella riga, byte nel blocco, cifra n). Inverso di `_trit_da_byte`."""
    nb, qk, _, _ = CS.LAYOUT[tipo]
    blk, p = divmod(c, qk)
    if tipo == "TQ1_0R_1024":
        qu, j = divmod(p, 256)
        qs0, qh0 = qu * 48, 192 + qu * 4
    else:
        j, qs0, qh0 = p, 0, 48
    if j < 160:
        return blk, qs0 + j % 32, j // 32
    if j < 240:
        return blk, qs0 + 32 + (j - 160) % 16, (j - 160) // 16
    return blk, qh0 + (j - 240) % 4, (j - 240) // 4


def m3_salti(r: GGUFReader, L: int, fam: str, e: int, n_salti: int, seme: int) -> dict:
    from tq1_0r_1024_codec import S1024
    import e2e_esporta as EX  # sola lettura: _deq_c (codice C di llama.cpp)
    t = next(x for x in r.tensors if x.name == f"blk.{L}.{FAM[fam]}.weight")
    tipo = t.tensor_type.name
    nb, qk, off, k = CS.LAYOUT[tipo]
    riga_b = np.ascontiguousarray(np.asarray(t.data)[e])       # (R, NBR)
    R, NBR = riga_b.shape
    C = NBR // nb * qk
    nuovo = riga_b.copy()
    T0 = CS.trit_da_blocchi(torch.from_numpy(riga_b.reshape(-1, nb)), tipo).view(R, C)
    rng = np.random.default_rng(seme)
    rr = rng.integers(0, R, n_salti)
    cc = rng.integers(0, C, n_salti)
    atteso = T0.clone()
    delta_q = {}                   # (riga, byte assoluto nella riga) -> somma dei delta di q
    fatti = 0
    visti = set()
    for r_, c_ in zip(rr.tolist(), cc.tolist()):
        if (r_, c_) in visti:
            continue
        visti.add((r_, c_))
        t0 = int(atteso[r_, c_])
        dt = int(rng.choice([-1, 1])) if t0 == 0 else -t0          # un solo livello, dentro {-1,0,1}
        blk, byte, n = posizione(c_, tipo)
        chiave = (r_, blk * nb + byte)
        delta_q[chiave] = delta_q.get(chiave, 0) + dt * 3 ** (4 - n)
        atteso[r_, c_] = t0 + dt
        fatti += 1
    byte_in_comune = sum(1 for _ in delta_q) != fatti
    for (r_, ab), dq in delta_q.items():
        v = int(riga_b[r_, ab])
        in_qh = (ab % nb) >= (192 if tipo == "TQ1_0R_1024" else 48)
        nc = 4 if in_qh else 5
        q = sum(cifra(v, n) * 3 ** (4 - n) for n in range(nc))
        nuovo[r_, ab] = codifica(q + dq)
    T1 = CS.trit_da_blocchi(torch.from_numpy(nuovo.reshape(-1, nb)), tipo).view(R, C)
    decode_ok = bool(torch.equal(T1, atteso))
    byte_cambiati = int((nuovo != riga_b).sum())
    d = torch.from_numpy(nuovo.reshape(-1, nb)[:, off:off + 2 * k].copy().view(np.float16).astype(np.float32))
    W = CS.pesi_da_trit_scale(T1.reshape(-1, qk), d.reshape(-1, k), tipo,
                              torch.from_numpy(S1024) if tipo != "TQ1_0" else None).numpy()
    Wc = EX._deq_c(nuovo.reshape(-1, nb), tipo)
    return {"strato": L, "famiglia": fam, "esperto": e, "tipo": tipo, "salti": fatti,
            "byte_cambiati": byte_cambiati, "byte_con_piu_salti": fatti - len(delta_q),
            "trit_cambiati_decode": int((T1 != T0).sum()), "decode_al_bit": decode_ok,
            "codice_c_al_bit": bool(np.array_equal(W, Wc)), "_byte_in_comune": byte_in_comune}


# ─────────────────────────────── main ───────────────────────────────
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uscita", required=True)
    ap.add_argument("--strati", default="0-39")
    ap.add_argument("--fili", type=int, default=4)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.fili)
    t0 = time.time()
    out = {"gguf": str(GGUF), "byte": GGUF.stat().st_size, "inizio": time.strftime("%Y-%m-%dT%H:%M:%S")}
    out["M1"] = m1_codifica()
    print("M1", json.dumps(out["M1"]), flush=True)
    canon_qs = np.array([v not in out["M1"]["qs"]["byte_non_canonici"] for v in range(256)])
    canon_qh = np.array([v not in out["M1"]["qh"]["byte_non_canonici"] for v in range(256)])
    r = GGUFReader(str(GGUF))
    tens = {t.name: t for t in r.tensors}
    a0, a1 = (int(x) for x in a.strati.split("-"))
    per = {}
    for L in range(a0, a1 + 1):
        for f, nome in FAM.items():
            ts = time.time()
            per[f"{L}.{f}"] = m2_tensore(tens[f"blk.{L}.{nome}.weight"], canon_qs, canon_qh)
            x = per[f"{L}.{f}"]
            print(f"M2 {L}.{f} {x['tipo']} nnz {1 - x['zero'] / x['codici']:.4f} tau_med "
                  f"{x['tau_rho001_per_esperto']['mediana']:.1f} non_canon {x['qs_non_canon']}+{x['qh_non_canon']} "
                  f"d0 {x['gruppi_d0']} {time.time() - ts:.1f}s", flush=True)
    out["M2_per_tensore"] = per
    agg = {}
    for f in FAM:
        xs = [v for kk, v in per.items() if kk.endswith("." + f)]
        s = {k2: sum(x[k2] for x in xs) for k2 in ("codici", "gruppi", "neg", "zero", "pos", "gruppi_d0",
                                                   "codici_in_d0", "gruppi_vuoti", "gruppi_pieni", "byte_qs",
                                                   "byte_qh", "qs_non_canon", "qh_non_canon")}
        s["tipo"] = xs[0]["tipo"]
        s["gs"] = xs[0]["gs"]
        s["gruppi_per_esperto"] = xs[0]["gruppi_per_esperto"]
        s["quota_nnz"] = 1 - s["zero"] / s["codici"]
        s["quota_nnz_per_strato"] = [round(1 - x["zero"] / x["codici"], 5) for x in xs]
        s["tau_rho001_mediana_delle_mediane"] = float(np.median([x["tau_rho001_per_esperto"]["mediana"] for x in xs]))
        s["tau_rho001_min"] = float(min(x["tau_rho001_per_esperto"]["min"] for x in xs))
        s["tau_rho001_max"] = float(max(x["tau_rho001_per_esperto"]["max"] for x in xs))
        s["tau_rho001_somma"] = float(sum(x["tau_rho001_per_esperto"]["somma"] for x in xs))
        s["tau_su_nnz_mediana"] = float(np.median([x["tau_su_nnz_mediana"] for x in xs]))
        agg[f] = s
    out["M2_per_famiglia"] = agg
    print("M2", json.dumps(agg, indent=1), flush=True)
    out["M3"] = [m3_salti(r, L, f, e, 4000, 20260926 + 7 * L + e) for L in (0, 39) for f in ("gate", "down")
                 for e in (0, 255)]
    for x in out["M3"]:
        print("M3", x, flush=True)
    gruppi_strato = {f: per[f"{a0}.{f}"]["gruppi"] for f in FAM}
    g_tot = sum(agg[f]["gruppi"] for f in FAM)
    out["M4"] = {"gruppi_per_strato": gruppi_strato, "gruppi_totali": g_tot,
                 "candidati_idx_uint8_GiB": g_tot / 2 ** 30,
                 "somma_e_somma2_fp32_GiB": 8 * g_tot / 2 ** 30,
                 "punteggio_proposta_fp32_GiB": 4 * g_tot / 2 ** 30,
                 "per_strato_MiB_idx_somme": 9 * sum(gruppi_strato.values()) / 2 ** 20}
    print("M4", out["M4"], flush=True)
    out["secondi"] = round(time.time() - t0, 1)
    p = Path(a.uscita)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    print("scritto", p, out["secondi"], "s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
