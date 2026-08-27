#!/usr/bin/env python3
"""GUARIGIONE SICURA — il cancello che la v3.2 non aveva.

⛔ LEZIONE 27/8 (ODINO v3.2, PPL 5.81 → 6.88): una cura non si applica perche'
migliora una METRICA INTERNA. Il QESC alzava l'overlap dei router (0.5418 →
0.5565) su 35 strati e il modello ne usciva PEGGIORE: i router di un modello
compresso sono coerenti con le SUE attivazioni, non con quelle dell'originale.

REGOLA (obbligatoria per ogni cura della Fucina):
  1. si applica su una COPIA, mai sull'originale;
  2. si misura l'ESITO VERO (perplexity) prima e dopo;
  3. si tiene solo se l'esito migliora, altrimenti si butta la copia;
  4. se la cura ha piu' strati/pezzi, si prova a SCAGLIONI (25%, 50%, 100%):
     spesso una parte giova e il resto danneggia.
"""
from __future__ import annotations
import argparse, shutil, subprocess, sys, time
from pathlib import Path

def log(*a): print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def perplexity(gguf: str, corpus: str, motore: str, chunks: int = 3) -> float:
    """L'esito VERO. Nessuna cura si giudica senza questo numero."""
    B = Path(motore) / "bin"
    out = subprocess.run(
        [str(B / "llama-perplexity"), "-m", gguf, "-ngl", "999",
         "-c", "2048", "-b", "2048", "--chunks", str(chunks), "-f", corpus],
        capture_output=True, text=True,
        env={"LD_LIBRARY_PATH": str(B), "ODINO_NO_P2": "1", "HOME": str(Path.home())},
    )
    for riga in reversed((out.stdout + out.stderr).splitlines()):
        if "Final estimate: PPL" in riga:
            return float(riga.split("PPL =")[1].split("+/-")[0])
    raise RuntimeError("perplexity non misurabile:\n" + (out.stderr or "")[-600:])


def cura_con_cancello(originale: str, cura, corpus: str, motore: str,
                      etichetta: str = "cura") -> bool:
    """Applica `cura(copia)` su una COPIA e la tiene solo se la PPL migliora."""
    orig = Path(originale)
    copia = orig.with_suffix(f".{etichetta}.gguf")
    log(f"misuro l'esito PRIMA della cura…")
    prima = perplexity(str(orig), corpus, motore)
    log(f"  PPL prima: {prima:.4f}")
    log(f"copio il modello ({orig.stat().st_size / 2**30:.1f} GiB)…")
    shutil.copy2(orig, copia)
    try:
        cura(str(copia))
        dopo = perplexity(str(copia), corpus, motore)
        log(f"  PPL dopo:  {dopo:.4f}  ({'MEGLIO' if dopo < prima else 'PEGGIO'})")
        if dopo < prima:
            copia.replace(orig)
            log(f"✅ cura TENUTA: {prima:.4f} → {dopo:.4f}")
            return True
        copia.unlink()
        log(f"⛔ cura SCARTATA (originale intatto): {prima:.4f} → {dopo:.4f}")
        return False
    except Exception as e:
        copia.unlink(missing_ok=True)
        log(f"⛔ cura fallita, copia rimossa: {e}")
        raise


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="misura la PPL di un GGUF (esito vero)")
    ap.add_argument("gguf"); ap.add_argument("--corpus", required=True)
    ap.add_argument("--motore", default="/home/angelo/build-llamacpp-tq1/build")
    ap.add_argument("--chunks", type=int, default=3)
    a = ap.parse_args()
    print(f"PPL = {perplexity(a.gguf, a.corpus, a.motore, a.chunks):.4f}")
