#!/usr/bin/env python3
"""18/10: tappa del congiunto (residui_<p>.pt, latenti A,B,gamma del modo continuo) -> residui_delta.pt nel formato
dell'esportazione di cardan_congiunto.py. dW = (B * gamma) @ A, fp64: = residui_ternari.delta_file/pesi_q (continuo)."""
import sys, torch
src, dst, passo = sys.argv[1], sys.argv[2], int(sys.argv[3])
d = torch.load(src, map_location="cpu", weights_only=False); cfg = d["cfg"]; assert cfg["continuo"]
dW = {n: ((sd["B"].double() * sd["gamma"].double().view(-1, 1)) @ sd["A"].double()).float() for n, sd in d["latenti"].items()}
torch.save({"delta_file": dW, "rango": cfg["rango"], "famiglie": list(cfg["famiglie"]), "strati": sorted({int(n.split(".")[1]) for n in dW}),
            "passo": passo, "nome": f"C-S4b-CONGIUNTO-tappa{passo}",
            "nota": "dW = diag(gamma) B A, ordine del FILE, base NON ruotata (cardan_piega_densi.py)"}, dst)
print(f"{len(dW)} densi -> {dst}")
