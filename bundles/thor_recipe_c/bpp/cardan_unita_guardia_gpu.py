#!/usr/bin/env python3
"""Adattatore additivo: applica la guardia solo nei nuovi lavoratori."""
from __future__ import annotations


def crea(config, device):
    import os
    import sys
    from pathlib import Path
    import pianificatore_gpu as PG
    for nome, sha in config["codici_guardia"].items():
        if PG.sha_file(Path(__file__).with_name(nome)) != sha:
            raise RuntimeError(f"codice guardia cambiato dopo il piano: {nome}")
    import torch
    import cardan_guardia_gpu as G
    import cardan_unita_gpu as U
    torch.cuda.set_device(device)
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    os.environ["PIEGA_ALGEBRA"] = config["algebra"]
    os.environ["LEVE_OSPITE"] = config["gguf"]
    if config["algebra"] not in ("gpu_sicura", "cpu"):
        raise ValueError("rocSOLVER GPU vietato")
    CP = G.installa()
    if config["algebra"] == "cpu":
        # Fattorizzazioni esplicitamente CPU, non il backend device della guardia.
        def cpu_factor(M):
            L, info = torch.linalg.cholesky_ex(M.detach().to("cpu", torch.float64))
            return L.to(M.device, M.dtype), info.to(M.device)
        CP.chol_ex = cpu_factor

        def cpu_chol(M):
            return torch.linalg.cholesky(M.detach().to("cpu", torch.float64)).to(M.device, M.dtype)
        CP.chol = cpu_chol

        def cpu_solve(M, B):
            return torch.linalg.solve(M.detach().to("cpu", torch.float64),
                                      B.detach().to("cpu", torch.float64)).to(B.device, B.dtype)
        CP.risolvi = cpu_solve
    if config["fase"] == "esperti":
        import cardan_esperti_guardia_gpu as E
        sys.path.insert(0, CP.RAFF)
        import comune as CM
        handler = E.crea(config, torch.device(f"cuda:{device}"), CP, CM)
    else:
        handler = U.crea(config, device)

    def lavora(dati, out):
        with G.contesto(device=device, fase=config["fase"],
                        strato=dati.get("strato"), famiglia=dati.get("famiglia"), nomi=dati.get("nomi")):
            return handler(dati, out)
    return lavora
