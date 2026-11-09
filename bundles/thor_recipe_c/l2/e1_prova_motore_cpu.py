#!/usr/bin/env python3
"""e1_prova_motore_cpu.py — collaudo CPU del GGUF E1 col motore vero (nessuna GPU), PRIMA della corsa.

Costruisce residui con le forme VERE di Thor (strati 12-27, attn su strati ad attenzione piena, gdn sugli altri, rango
32, dalle forme dei tensori del GGUF base) in due varianti: Z (gamma = 0, B = 0: ramo presente ma nullo) e R (gamma
casuale), scrive i due GGUF con e1_scrivi_residui_gguf e fa girare llama-perplexity su CPU (-ngl 0, -dev none) su un
testo corto. Attesi:  PPL(base) == PPL(Z) al bit;  PPL(R) != PPL(base);  PPL(R, THOR_SPEGNI=lr) == PPL(base).
Uso: e1_prova_motore_cpu.py --bin BUILD/bin --base R2B3m.gguf --dir CARTELLA_DI_LAVORO --testo FILE
"""
import argparse, json, os, re, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import residui_ternari as RT          # noqa: E402
import e1_scrivi_residui_gguf as SG   # noqa: E402


def mods_thor(base: str, strati, gamma_sd: float, seme: int) -> dict:
    t = {d["nome"]: d for d in SG.indice(base)["tensori"]}
    cfg = RT.ConfigResidui(rango=32, piani=1, strati=tuple(strati), famiglie=RT.famiglie_da("attn,gdn"))
    out = {}
    for L in strati:
        for fam in cfg.famiglie:
            n = f"blk.{L}.{fam}.weight"
            if n not in t:
                continue
            ne = t[n]["ne"]
            de, fu = int(ne[0]), int(ne[1])
            m = RT.ResiduoTernario(f"blk.{L}.{fam}", fu, de, cfg, seme=seme + 31 * L + len(out))
            with torch.no_grad():
                m.gamma.copy_(torch.randn(fu, generator=torch.Generator().manual_seed(seme + L)) * gamma_sd)
            out[f"blk.{L}.{fam}"] = m
    return out


def scrivi(base, out_gguf, mods, cart):
    tr, info = RT.tensori_gguf(mods)
    np.savez(cart / "r.npz", **tr)
    json.dump({"tensori": info}, open(cart / "r.json", "w"))
    r = subprocess.run([sys.executable, str(QUI / "e1_scrivi_residui_gguf.py"), "--src", base, "--out", str(out_gguf),
                        "--npz", str(cart / "r.npz"), "--info", str(cart / "r.json"), "--meta", "--min-free-gb", "20"],
                       capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(r.stdout[-2000:] + r.stderr[-2000:])
    return len(tr)


def ppl(bin_, gguf, testo, env_extra=None):
    env = dict(os.environ, LD_LIBRARY_PATH=str(bin_), GGML_VK_VISIBLE_DEVICES="", **(env_extra or {}))
    cmd = [f"{bin_}/llama-perplexity", "-m", str(gguf), "-ngl", "0", "-dev", "none", "-c", "128", "-b", "128",
           "--chunks", "2", "-t", "8", "-f", str(testo)]
    t0 = time.time()
    r = subprocess.run(["nice", "-n", "15"] + cmd, capture_output=True, text=True, env=env)
    log = r.stdout + r.stderr
    m = re.search(r"Final estimate: PPL = ([0-9.]+)", log)
    v = re.findall(r"\[\d+\]([0-9.]+)", log)
    return {"rc": r.returncode, "ppl": m.group(1) if m else None, "chunk": v, "s": round(time.time() - t0, 1),
            "coda": log[-600:] if r.returncode or not m else ""}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True); ap.add_argument("--base", required=True)
    ap.add_argument("--dir", required=True); ap.add_argument("--testo", required=True)
    ap.add_argument("--tieni", action="store_true")
    a = ap.parse_args()
    cart = Path(a.dir); cart.mkdir(parents=True, exist_ok=True)
    strati = range(12, 28)
    gz, gr = cart / "E1_Z.gguf", cart / "E1_R.gguf"
    nz = scrivi(a.base, gz, mods_thor(a.base, strati, 0.0, 1), cart)
    nr = scrivi(a.base, gr, mods_thor(a.base, strati, 0.05, 2), cart)
    ris = {"n_tensori": [nz, nr],
           "base": ppl(a.bin, a.base, a.testo),
           "Z": ppl(a.bin, gz, a.testo),
           "R": ppl(a.bin, gr, a.testo),
           "R_spento": ppl(a.bin, gr, a.testo, {"THOR_SPEGNI": "lr"})}
    print(json.dumps(ris, indent=1))
    ok = (all(ris[k]["rc"] == 0 and ris[k]["ppl"] for k in ("base", "Z", "R", "R_spento"))
          and ris["Z"]["chunk"] == ris["base"]["chunk"] and ris["R_spento"]["chunk"] == ris["base"]["chunk"]
          and ris["R"]["chunk"] != ris["base"]["chunk"])
    json.dump(ris, open(cart / "esito.json", "w"), indent=1)
    if not a.tieni:
        for f in (gz, gr):
            for g in (f, Path(str(f) + ".modifica.json")):
                g.unlink(missing_ok=True)
    print("PROVA MOTORE CPU", "OK" if ok else "FALLITA")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
