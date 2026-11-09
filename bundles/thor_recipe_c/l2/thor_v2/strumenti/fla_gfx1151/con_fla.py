#!/usr/bin/env python3
"""Esegue uno script di thor_v2 (e2e_allena.py, ...) col GDN su fla (ricetta gfx1151), SENZA toccare lo script (27/9).

    cd ~/tony-forgia/thor_v2
    @@ROOT:venv@@/bin/python3 strumenti/fla_gfx1151/con_fla.py e2e_allena.py misura --teacher ... [argomenti]

Ordine obbligato: 1) fla bloccato mentre transformers importa il modeling (cosi' il decoratore
use_kernel_func_from_hub_with_fallback lega il ripiego PyTorch, come nella corsa); 2) abilita_sicuro(M): copia di fla
da pkg/ (mai dal venv), config fissate a scratch zero, guardia scratch, controllo fwd+bwd contro fp64 vero e
riscaldamento alle dimensioni della corsa; se un controllo non torna resta il ripiego e lo si dice; 3) runpy dello
script come __main__, nello stesso processo: il modulo modeling e' lo stesso oggetto, il forward di GatedDeltaNet
legge il globale sostituito a ogni chiamata.
THOR_GDN_FLA=0 -> nessuna sostituzione (stesso lanciatore, ripiego puro: per confronti a parita' di condizioni).
THOR_GDN_FLA_OBBLIGATORIO=1 -> se fla non si abilita, esce con errore invece di proseguire col ripiego."""
import os
import runpy
import sys
from pathlib import Path

QUI = Path(__file__).resolve().parent


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    script = Path(sys.argv[1]).resolve()
    sys.modules["fla"] = None
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as M
    if os.environ.get("THOR_GDN_FLA", "1") != "0":
        sys.path.insert(0, str(QUI))
        import gdn_fla_gfx1151 as G
        r = G.abilita_sicuro(M)
        print(f"[con_fla] GDN fla: {r}", flush=True)
        if not r["attivo"] and os.environ.get("THOR_GDN_FLA_OBBLIGATORIO") == "1":
            return 3
    else:
        print("[con_fla] THOR_GDN_FLA=0: ripiego PyTorch", flush=True)
    sys.argv = [str(script)] + sys.argv[2:]
    sys.path.insert(0, str(script.parent))
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as e:
        codice = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    else:
        codice = 0
    gdn = M.torch_chunk_gated_delta_rule
    if hasattr(gdn, "stato"):
        print(f"[con_fla] chiamate GDN: {gdn.stato}", flush=True)
    return codice


if __name__ == "__main__":
    raise SystemExit(main())
