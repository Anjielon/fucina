#!/usr/bin/env python3
"""Memoria di V-Bop sul Thor-T VERO (27/9, ramo vbop2), SOLO metadati del GGUF: nessun byte degli esperti letto,
nessun dispositivo. Conta i gruppi (E x R x NG) delle famiglie di esperti che `PV.installa` prende (le stesse
di `S.FAMIGLIE`), crea `StatoBop` sul dispositivo `meta` (forme e tipi veri, zero memoria) e somma i byte con
`StatoBop.byte()` (la funzione che la corsa scrive nel log). Confronta con cio' che V-Bop toglie: la finestra di
accumulo (`d_finestra` fp16 per gruppo, solo con pv_passi_accumulo > 1) e il file accumulo_pv_*.pt.

Uso: nice -n 15 env CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= python3 strumenti/memoria_vbop.py [GGUF]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

QUI = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(QUI))
import e2e_studente as S  # noqa: E402
import e2e_codici_scale as CS  # noqa: E402
import e2e_pv_codici as PV  # noqa: E402
from gguf import GGUFReader  # noqa: E402


def main() -> int:
    gguf = Path(sys.argv[1]) if len(sys.argv) > 1 else S.GGUF_THOR_T
    tens = {t.name: t for t in GGUFReader(str(gguf)).tensors}
    gruppi = esperti = fam_n = 0
    byte_bop = 0
    per_tipo: dict = {}
    for nome_t, t in tens.items():
        parti = nome_t.split(".")
        if len(parti) < 3 or parti[0] != "blk" or int(parti[1]) >= 40:   # la corsa: strati range(40) (blk.40 = MTP)
            continue
        fam_nome = next((n for n, suff in S.FAMIGLIE.items() if nome_t == f"blk.{parti[1]}.{suff}.weight"), None)
        if fam_nome is None:
            continue
        tipo = t.tensor_type.name
        nb, qk, _, sub = CS.LAYOUT[tipo]
        E, R, NBR = (int(x) for x in t.data.shape)       # memmap: solo la forma
        NG = (NBR // nb) * qk // (qk // sub)
        f = SimpleNamespace(E=E, R=R, NG=NG, blocchi=torch.empty(0, device="meta"))
        z = PV.StatoBop.crea(f, device="meta")
        byte_bop += z.byte()
        gruppi += E * R * NG
        esperti += E
        fam_n += 1
        per_tipo[tipo] = per_tipo.get(tipo, 0) + E * R * NG
    ris = {"gguf": str(gguf), "famiglie": fam_n, "esperti_x_famiglie": esperti, "gruppi": gruppi,
           "gruppi_per_tipo": per_tipo,
           "vbop_byte": byte_bop, "vbop_GB": byte_bop / 1e9, "vbop_GiB": byte_bop / 2 ** 30,
           "vbop_byte_attesi_8B_piu_12B": 8 * gruppi + 12 * esperti,
           "buffer_passo_17B_GB": 17 * gruppi / 1e9,
           "d_finestra_fp16_GB_tolta": 2 * gruppi / 1e9,
           "file_accumulo_GB_tolto_17B_piu_2B": 19 * gruppi / 1e9,
           "netto_VRAM_GB_da_accumulo4_a_vbop": (byte_bop - 2 * gruppi) / 1e9}
    print(json.dumps(ris, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
