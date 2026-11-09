#!/usr/bin/env python3
"""1/10/2026 (ramo tt5-ottimizzato) — prova CPU della cache di embedding e testa dequantizzate (e2e_cache_dequant).

  C1  per token_embd e output di R2B3m (3 piani ciascuna, la configurazione di TT5): l'array di `_somma_piani` SENZA
      cache (la via di sempre), quello con la cache VUOTA (calcolo + scrittura) e quello con la cache PIENA (lettura)
      sono identici byte per byte (sha256 di dtype, forma e byte; e torch.equal sui tensori come li usa lo studente);
  C2  una cache rovinata (byte cambiati nel .npy) non si usa: si ricalcola e il risultato resta quello di sempre;
  C3  una chiave diversa (dequantizzatore diverso: chiave calcolata su un altro sorgente) non ritrova il file.

Uso (solo CPU, 1 filo come l'avvio vero): nice -n 19 env OMP_NUM_THREADS=1 python3 e2e_test_cache_dequant.py
Esito: $E2E_CD_ESITO (default accanto a questo file). La cartella della cache provata e' $E2E_CD_DIR (default
@@ROOT:depot@@/e2e/cache_dequant: la stessa dell'allenamento, che cosi' resta riempita)."""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

for _v in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
    os.environ[_v] = ""
import numpy as np  # noqa: E402
import torch  # noqa: E402

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import e2e_studente as S  # noqa: E402
import e2e_cache_dequant as CD  # noqa: E402

R1 = Path(os.getenv("E2E_R2B3M", "@@ROOT:models@@/gguf/tony-tern/R2B3m/Thor-T-R2B3m.gguf"))
CACHE = Path(os.getenv("E2E_CD_DIR", "@@ROOT:depot@@/e2e/cache_dequant"))
ESITO = {"data": time.strftime("%Y-%m-%d %H:%M:%S"), "gguf": str(R1), "cache": str(CACHE), "prove": {}}


def registra(nome, ok, **dati):
    ESITO["prove"][nome] = {"ok": bool(ok), **dati}
    print(f"[{nome}] {'OK' if ok else 'ROSSO'} {json.dumps(dati, default=str)[:2000]}", flush=True)
    return ok


def main() -> int:
    st = S.Studente(gguf=R1, strati=[0], ricetta="B3_ind32", motore_nome="torch", device="cpu", esperti=slice(0, 1),
                    log=lambda *_: None)
    ok = True
    prova_dir = CACHE.parent / (CACHE.name + "_prova_rovinata")
    for nome in ("output.weight", "token_embd.weight"):
        t0 = time.time()
        os.environ.pop("E2E_CACHE_DEQUANT", None)
        a = st._somma_piani(nome)                               # via di sempre
        t_a = time.time() - t0
        os.environ["E2E_CACHE_DEQUANT"] = str(CACHE)
        piani = S.piani_di(st._tens, nome)
        k = CD.chiave(st._tens, piani)
        p_npy = CACHE / f"{nome}-{k[:24]}.npy"
        gia = p_npy.exists()
        t0 = time.time()
        b = st._somma_piani(nome)                               # cache vuota (o gia' piena da un giro prima)
        t_b = time.time() - t0
        t0 = time.time()
        c = st._somma_piani(nome)                               # cache piena: lettura
        t_c = time.time() - t0
        sa, sb, sc = CD.sha_array(a), CD.sha_array(b), CD.sha_array(c)
        tt = torch.equal(torch.from_numpy(np.ascontiguousarray(a)), torch.from_numpy(c))
        r = {"piani": piani, "dtype": [a.dtype.str, c.dtype.str], "forma": list(a.shape), "sha_sempre": sa,
             "sha_cache_scritta": sb, "sha_cache_letta": sc, "torch_equal": tt, "file_gia_presente": gia,
             "secondi_sempre": round(t_a, 1), "secondi_prima_chiamata": round(t_b, 1), "secondi_lettura": round(t_c, 1)}
        ok &= registra(f"C1_{nome}", sa == sb == sc and tt and a.dtype == c.dtype and a.shape == c.shape, **r)
        if nome == "output.weight":
            # C2: copia rovinata in una cartella a parte
            if prova_dir.exists():
                shutil.rmtree(prova_dir)
            prova_dir.mkdir(parents=True)
            shutil.copy(p_npy, prova_dir / p_npy.name)
            shutil.copy(p_npy.with_name(p_npy.name[:-4] + ".json"), prova_dir / (p_npy.name[:-4] + ".json"))
            with open(prova_dir / p_npy.name, "r+b") as f:
                f.seek(-4096, 2)
                f.write(b"\x7f" * 64)
            righe = []
            d = CD.somma_piani(st._tens, nome, piani, prova_dir, lambda: a, log=righe.append)
            ok &= registra("C2_cache_rovinata_ricalcolata", CD.sha_array(d) == sa and any("sha diverso" in x for x in righe),
                           righe=righe)
            # C3: chiave con un altro dequantizzatore
            orig = CD._sha_file
            CD._sha_file = lambda p: "altro"
            try:
                k2 = CD.chiave(st._tens, piani)
            finally:
                CD._sha_file = orig
            ok &= registra("C3_chiave_dipende_dal_codice", k2 != k, k=k[:24], k_altro=k2[:24])
            shutil.rmtree(prova_dir)
        del a, b, c
    ESITO["ok"] = ok
    dest = Path(os.getenv("E2E_CD_ESITO", str(QUI / "e2e_test_cache_dequant_esito.json")))
    dest.write_text(json.dumps(ESITO, indent=1, default=str))
    print(("TUTTO VERDE " if ok else "ROSSO ") + str(dest))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
