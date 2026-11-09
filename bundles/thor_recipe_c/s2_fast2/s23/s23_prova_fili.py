#!/usr/bin/env python3
"""VELOCE2 14/10 (leva E0) — le fattorizzazioni CPU fp64 della passata danno gli STESSI BIT con meno fili OpenBLAS/OMP?

Perche' (ricerche/C_S23_VELOCE2_1014.md §1): la corsa gira senza OMP_NUM_THREADS/OPENBLAS_NUM_THREADS = 32 fili su 32 CPU
logiche, a nice 5. Con UN core occupato da un altro processo a nice 0 (jellyfin, scansione della libreria bloccata dalle
16:32) ogni regione parallela aspetta il filo sfrattato: U_Minv_lotto (Cholesky 2048 degli esperti gate/up) e l'eigh della
tridiagonale di Lanczos (DRQ) diventano 10-20 volte piu' lenti. Meno fili = margine per gli altri carichi.

Uso: python3 s23_prova_fili.py --uscita <cartella> --fili 32,16,8 (UN processo per configurazione, rilanciato da se'
con l'ambiente giusto). Ogni processo salva le uscite di: U_Minv_lotto (b 4, n 2048 e 512, come gate/up e down), chol_ex
a blocchi 128 (Babai), cholesky_inverse, eigh di tridiagonali (k 8..160, Lanczos), eigvalsh + cholesky_ex fp32 [256,16,16]
(scale di JARQ), cholesky_solve, linalg.solve (CPU fp64), e i tempi. Il padre confronta AL BIT con la prima configurazione
(quella della corsa) e scrive esito.json + una riga «SCELTA <fili>»: la configurazione piu' veloce fra quelle identiche al
bit su TUTTO; se nessuna lo e', la prima (= nessun cambio)."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

QUI = Path(__file__).resolve().parent


def figlio(uscita: Path, nome: str):
    import torch
    for p in (str(QUI), str(QUI.parent / "lib"), str(QUI.parent / "s1b"), str(QUI.parent / "rif")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import s23_passata as SP
    F64 = torch.float64
    g = torch.Generator().manual_seed(20261014)
    out, tempi = {}, {}

    def spd(b, n, N=None):
        X = torch.randn(b, N or n + 64, n, generator=g, dtype=F64) * torch.linspace(0.2, 3, n, dtype=F64)
        return X.transpose(1, 2) @ X / X.shape[1]

    def cron(k, f):
        t = time.time()
        r = f()
        tempi[k] = round(time.time() - t, 4)
        return r

    for n in (2048, 512):
        H = spd(4, n)
        H = H / torch.diagonal(H, dim1=1, dim2=2).mean(1).view(-1, 1, 1)
        Mi, U, nr = cron(f"U_Minv_{n}", lambda: SP.U_Minv_lotto(H))
        out[f"U_Minv_{n}"] = (Mi, U)
    Hb = spd(16, 128)
    out["chol_ex_128"] = cron("chol_ex_128", lambda: torch.linalg.cholesky_ex(Hb)[0])
    out["chol_inv_128"] = cron("chol_inv_128", lambda: torch.cholesky_inverse(out["chol_ex_128"]))
    Bv = torch.randn(16, 128, 64, generator=g, dtype=F64)
    out["chol_solve_128"] = cron("chol_solve_128", lambda: torch.cholesky_solve(Bv, out["chol_ex_128"]))
    for k in (8, 40, 80, 160):
        Tm = torch.diag(torch.rand(k, generator=g, dtype=F64)) + torch.diag(torch.rand(k - 1, generator=g, dtype=F64), 1)
        Tm = Tm + Tm.T
        out[f"eigh_{k}"] = tuple(cron(f"eigh_{k}", lambda: torch.linalg.eigh(Tm)))
    Hg = spd(1, 2048)[0]
    out["eigh_2048"] = tuple(cron("eigh_2048", lambda: torch.linalg.eigh(Hg)))
    Mp = spd(256, 16).float()
    out["eigvalsh_f32"] = cron("eigvalsh_f32", lambda: torch.linalg.eigvalsh(Mp))
    out["cholex_f32"] = cron("cholex_f32", lambda: torch.linalg.cholesky_ex(Mp)[0])
    out["solve_2048"] = cron("solve_2048", lambda: torch.linalg.solve(Hg, torch.randn(2048, 512, generator=g, dtype=F64)))
    out["chol_2048"] = cron("chol_2048", lambda: torch.linalg.cholesky(Hg))
    torch.save(out, uscita / f"uscite_{nome}.pt")
    (uscita / f"tempi_{nome}.json").write_text(json.dumps({"tempi": tempi, "torch_fili": torch.get_num_threads(),
                                                          "env": {k: os.environ.get(k) for k in
                                                                  ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}))


def uguali(a, b):
    import torch
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, (tuple, list)) or hasattr(a, "_fields"):
        return all(uguali(x, y) for x, y in zip(a, b))
    return a == b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uscita", required=True)
    ap.add_argument("--fili", default="nessuno,16,8", help="la PRIMA e' la configurazione della corsa (riferimento)")
    ap.add_argument("--figlio", default="")
    a = ap.parse_args()
    u = Path(a.uscita)
    u.mkdir(parents=True, exist_ok=True)
    if a.figlio:
        figlio(u, a.figlio)
        return 0
    import torch
    conf = [x for x in a.fili.split(",") if x]
    for c in conf:
        env = dict(os.environ)
        for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            if c == "nessuno":
                env.pop(k, None)
            else:
                env[k] = c
        env.update(HIP_VISIBLE_DEVICES="", CUDA_VISIBLE_DEVICES="", ROCR_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
        r = subprocess.run([sys.executable, __file__, "--uscita", str(u), "--figlio", c], env=env)
        if r.returncode:
            print(f"configurazione {c}: figlio fallito ({r.returncode})", flush=True)
            return 2
    rif = torch.load(u / f"uscite_{conf[0]}.pt")
    esito = {}
    for c in conf:
        x = torch.load(u / f"uscite_{c}.pt")
        t = json.loads((u / f"tempi_{c}.json").read_text())
        diversi = [k for k in rif if not uguali(rif[k], x[k])]
        esito[c] = {"identico_al_bit": not diversi, "diversi": diversi, "tempi": t["tempi"],
                    "totale_s": round(sum(t["tempi"].values()), 3), "torch_fili": t["torch_fili"]}
        print(f"fili {c}: identico al bit {not diversi} {diversi} | totale {esito[c]['totale_s']} s | {t['tempi']}",
              flush=True)
    ok = [c for c in conf if esito[c]["identico_al_bit"]]
    scelta = min(ok, key=lambda c: esito[c]["totale_s"]) if ok else conf[0]
    (u / "esito.json").write_text(json.dumps({"esito": esito, "scelta": scelta}, indent=1))
    print(f"SCELTA {scelta}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
