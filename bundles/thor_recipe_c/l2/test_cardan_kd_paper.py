#!/usr/bin/env python3
"""CPU test of cardan_kd_paper.py (10/10 night audit) on the same fake 2-layer student as test_cardan_kd.K6:
  P1  defaults (kl_piena) behave as cardan_kd.py: identity gate at the bit, held-out KL decreases;
  P2  --perdita paper --selezione paper: runs, the held-out PAPER loss (tau^2 KL top-k + lambda CE) decreases, KL24
      is logged at every evaluation, cardan_migliore.pt has a non-zero C."""
import json
import os
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_cardan_kd as TK  # noqa: E402
import test_rilq_thor_shexp as TS  # noqa: E402
import cardan_kd_paper as KDP  # noqa: E402


def _con_teste(fA):
    """the fake e2e_allena plus the three head helpers the paper loss uses."""
    fA._logit_studente = lambda h, W: h.float() @ W.float().T
    fA._logit_maestro = lambda h, W, dt: h.float() @ W.float().T
    fA._per_testa = lambda q, W: q.float() @ W.float()
    return fA


def _driver_paper(out, passi, extra=()):
    vecchio = sys.modules.get("cardan_kd")
    vecchia_fa = TS._allena_finto
    sys.modules["cardan_kd"] = KDP
    TS._allena_finto = lambda WS, WT, m: _con_teste(vecchia_fa(WS, WT, m))
    try:
        return TK._driver(out, passi, extra)
    finally:
        TS._allena_finto = vecchia_fa
        if vecchio is None:
            sys.modules.pop("cardan_kd", None)
        else:
            sys.modules["cardan_kd"] = vecchio


def test_paper():
    base = os.getenv("CARDAN_TEST_TMP", "@@ROOT:models@@/gguf/tony-tern/CARDAN")
    with tempfile.TemporaryDirectory(dir=base) as td:
        c = _driver_paper(td + "/a", 4)
        reg = open(td + "/a/registro.log").read()
        st = json.load(open(td + "/a/storia.json"))
        kl = [r["kl_held"] for r in st["storia"] if "kl_held" in r]
        assert c == 0 and "AL BIT" in reg and kl[-1] < kl[0], (c, kl)
        c = _driver_paper(td + "/b", 4, ["--perdita", "paper", "--selezione", "paper", "--tau", "2.0",
                                         "--lambda-ce", "0.3", "--topk", "16"])
        reg = open(td + "/b/registro.log").read()
        assert c == 0, reg[-800:]
        st = json.load(open(td + "/b/storia.json"))
        ph = [r["kl_held"] for r in st["storia"] if "kl_held" in r]
        assert all("kl24" in r for r in st["storia"] if "kl_held" in r), st["storia"]
        assert min(ph[1:]) < ph[0], ph
        d = torch.load(td + "/b/cardan_migliore.pt", weights_only=False)
        assert float(d["stato"][0]["C_gate"].abs().max()) > 0
        print(f"P1 defaults OK (KL {kl[0]:.4f} -> {kl[-1]:.4f}); P2 paper loss held-out {ph[0]:.4f} -> {min(ph):.4f}")


if __name__ == "__main__":
    test_paper()
    print("test_cardan_kd_paper: OK")
