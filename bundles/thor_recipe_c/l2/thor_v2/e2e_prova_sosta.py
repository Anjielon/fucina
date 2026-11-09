#!/usr/bin/env python3
"""e2e_prova_sosta — prova AL BIT di --scritture-sosta-dispositivo (2/10/2026, ramo tt5-fase-v). File NUOVO.

`salva_ripresa` (la funzione vera di e2e_allena) su uno studente finto con la forma di una ripresa vera (theta di piu'
dtype e forme, AdamW dopo due passi, stato del ciclo, EMA con una vista NON contigua e una trasposta) e un'istantanea
V-Bop (u8/f16/f32), per tre vie: (1) asincrona di oggi (copia CPU), (2) asincrona in SOSTA sul dispositivo,
(3) sincrona (solo la ripresa: il V-Bop sincrono si scrive altrove). I file ripresa_*.pt e bop_pv_*.pt devono avere lo
stesso sha256 nelle vie 1 e 2, e la ripresa anche nella 3.

    python3 e2e_prova_sosta.py cpu    # sulla CPU: la sosta e' forzata (copia = clone CPU), prova la struttura dei file
    python3 e2e_prova_sosta.py cuda   # sulla GPU (SOLO a corsa ferma, ~1 min, ~2 GB di VRAM): prova anche
                                      # l'etichetta «cpu» delle storage del dispositivo, e misura il picco di RAM host
Esito JSON su stdout; codice 0 = uguali.
"""
from __future__ import annotations

import hashlib
import json
import os
import resource
import sys
import tempfile
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2e_allena as A  # noqa: E402


def studente_finto(device: str, grande: int, seme: int = 7):
    g = torch.Generator().manual_seed(seme)
    forme = [(64, 33), (7, 5, 3), (129,)] + ([(grande,)] if grande else [])
    ps = [torch.nn.Parameter(torch.randn(*f, generator=g).to(device)) for f in forme]
    opt = torch.optim.AdamW(ps, lr=1e-3)
    for _ in range(2):
        for p in ps:
            p.grad = torch.randn(p.shape, generator=g).to(device)
        opt.step()
    st = types.SimpleNamespace(chiavi=lambda: [(0, f"t{i}") for i in range(len(ps))], parametri=lambda: ps)
    base = torch.randn(40, 16, generator=g).to(device)
    ema = [base.T[:, :3], base[:, ::2], torch.arange(10, dtype=torch.int64).to(device)] + \
        [p.detach() * 0.5 for p in ps[3:]]                # viste NON contigue e un int64, come una EMA vera
    bop = {"gamma": 0.5, "curvatura_ogni": 1, "passi_bop": 3,
           "stato": {"0.gate": {"cand": torch.randint(0, 128, (2, 8, 4), generator=g, dtype=torch.uint8).to(device),
                                "m": torch.randn(2, 8, 4, generator=g).half().to(device),
                                "sm": torch.rand(2, generator=g).to(device)}}}
    return st, opt, ema, bop


def stato_ciclo(passo: int) -> dict:
    return {"passo": passo, "migliore": {"dir": "migliore_000000"}, "pv": {"file": "r.npz", "sha": "abc"},
            "storia": [{"a": 1.5, "b": [1, 2]}]}


def via(d: Path, device: str, nome: str, grande: int) -> dict:
    d.mkdir(parents=True, exist_ok=True)
    st, opt, ema, bop = studente_finto(device, grande)
    hwm0 = _hwm()
    if nome == "sincrona":                               # come la via sincrona di allena: _salva_bop_pv, poi la ripresa
        stato = stato_ciclo(5)
        sha_b = A._torch_save_atomico(A._copia_cpu(bop), d / "bop_pv_000005.pt")
        stato["pv"]["bop"] = {"file": "bop_pv_000005.pt", "sha": sha_b}
        A.salva_ripresa(d, st, opt, stato, 2, ema=ema)
    else:
        sosta = nome == "sosta"
        vp = A.sosta_possibile
        if sosta and device == "cpu":
            A.sosta_possibile = lambda *a: True          # sulla CPU la sosta e' un clone CPU: si prova la struttura
        try:
            scr = A.ScrittoreAsincrono(log=lambda *_: None)
            b = bop if sosta else A._copia_cpu(bop)      # oggi: istantanea gia' su CPU (istantanea_bop)
            A.salva_ripresa(d, st, opt, stato_ciclo(5), 2, ema=ema, scrittore=scr, bop=("bop_pv_000005.pt", b, {}),
                            sosta=sosta)
            scr.attendi()
        finally:
            A.sosta_possibile = vp
    out = {"ripresa": hashlib.sha256((d / "ripresa_000005.pt").read_bytes()).hexdigest(),
           "picco_rss_mib_in_piu": (_hwm() - hwm0) / 1024}
    if (d / "bop_pv_000005.pt").exists():
        out["bop"] = hashlib.sha256((d / "bop_pv_000005.pt").read_bytes()).hexdigest()
    z = torch.load(d / "ripresa_000005.pt", map_location="cpu", weights_only=False)
    out["theta_uguali_al_vivo"] = all(torch.equal(z["theta"][f"0.t{i}"], p.detach().cpu())
                                      for i, p in enumerate(st.parametri()))
    out["etichette_cpu"] = _etichette(d / "ripresa_000005.pt")
    return out


def _etichette(p: Path) -> list:
    """Le etichette di posizione scritte nel pickle (zip data.pkl): devono essere tutte «cpu»."""
    import pickletools
    import zipfile
    with zipfile.ZipFile(p) as z:
        nome = [n for n in z.namelist() if n.endswith("data.pkl")][0]
        dati = z.read(nome)
    return sorted({a for op, a, _ in pickletools.genops(dati) if isinstance(a, str) and a in
                   ("cpu", "cuda", "cuda:0", "hip", "meta")})


def _hwm() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss          # KiB


def main() -> int:
    device = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    grande = int(os.getenv("E2E_SOSTA_GRANDE", "0" if device == "cpu" else str(64 << 20)))   # 256 MiB fp32 su GPU
    base = Path(tempfile.mkdtemp(prefix="sosta_", dir=os.getenv("E2E_SOSTA_DIR", "@@ROOT:depot@@/e2e/prove_fase_v")))
    ris = {}
    for nome in (("sosta", "cpu_oggi", "sincrona") if device != "cpu" else ("cpu_oggi", "sosta", "sincrona")):
        ris[nome] = via(base / nome, device, nome, grande)
    # la ripresa SINCRONA salva le viste con la loro storage intera (qui l'EMA ha viste di proposito): differisce gia' da
    # quella asincrona di oggi per costruzione (nota, non un difetto della sosta); il confronto che conta e' sosta == oggi
    ris["nota"] = {"sincrona_uguale_asincrona_oggi": ris["sincrona"]["ripresa"] == ris["cpu_oggi"]["ripresa"]}
    ok = (ris["sosta"]["ripresa"] == ris["cpu_oggi"]["ripresa"]
          and ris["sosta"]["bop"] == ris["cpu_oggi"]["bop"] == ris["sincrona"]["bop"]
          and all(r["theta_uguali_al_vivo"] and r["etichette_cpu"] == ["cpu"]
                  for k, r in ris.items() if k != "nota"))
    print(json.dumps({"device": device, "ok": ok, "vie": ris, "dir": str(base)}, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
