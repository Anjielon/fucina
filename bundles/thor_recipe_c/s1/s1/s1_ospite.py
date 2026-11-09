#!/usr/bin/env python3
"""S1 tappa 0: scelta dell'OSPITE (il C migliore al lancio) e verifica di taglia al byte e tipi dei tensori.

Regola = quella di c_onpolicy_1012.sh: il miglior MMLU-2000 int8 Corsair (motore thor-c0t) fra i candidati il cui log ha
2000 righe MCLP e il cui GGUF, se ha un .sha256, lo rispetta. Unica differenza DICHIARATA: a parita' vince la linea PTQ
(CARDAN-esperti, poi i bracci PTQ2) prima delle linee E2E, perche' S2/S3 ripartono dai codici dell'ospite e la guardia
«mai peggio dell'ospite» li confronta coi codici PTQ (oggi (c)+DRQ 64,65 = E2E-C3 p3 64,65 -> (c)+DRQ).
S1_OSPITE=<gguf> forza la scelta (deve comunque passare i controlli). Una corsa gia' partita tiene il SUO ospite
(ospite.json): statistiche di due ospiti diversi non si mescolano.
Controlli: taglia = 8.631.199.776 B; gate/up esperti TQ1_0R_1024 a un piano, nessun *_exps2; densi a un piano (nessun
tensore densi *2/*3/*4); i 40 ffn_down_exps presenti.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

TT = Path("@@ROOT:models@@/gguf/tony-tern")
OC = TT / "OBIETTIVO_C"
CICLI = Path("@@ROOT:forge@@/stato/cicli.tsv")
TAGLIA = 8631199776


def candidati() -> list:
    c = [(OC / "CARDAN_ESPERTI_1013/kd/Thor-T-C-bil1jarqdrq-CARDAN.gguf", OC / "CARDAN_ESPERTI_1013/kd/mmlu_CARDAN.log",
          "C-CARDAN-esperti")]
    for g in sorted((OC / "PTQ2").glob("Thor-T-C-p_*.gguf")):
        n = g.stem.replace("Thor-T-", "")
        c.append((g, OC / "PTQ2" / f"mmlu_{n}.log", n))
    c.append((TT / "C_PTQ/Thor-T-C-f_bil1jarq.gguf", TT / "E2E_C2/mmlu_C-bil1jarq.log", "C-bil1jarq"))
    for p in ("p2", "p3"):
        c.append((TT / f"E2E_C2/Thor-T-E2E-C2-{p}.gguf", TT / f"E2E_C2/mmlu_E2E-C2-{p}.log", f"E2E-C2-{p}"))
    if CICLI.exists():
        g3 = None
        for r in CICLI.read_text(errors="replace").splitlines():
            if "\tCHIUSO\tE2E-C3" in r:
                m = re.findall(r"file (/\S+\.gguf)", r)
                g3 = m[-1] if m else g3
        if g3:
            n = Path(g3).stem.replace("Thor-T-", "")
            c.append((Path(g3), Path(g3).parent / f"mmlu_{n}.log", n))
    for d in sorted((OC / "ONPOLICY").glob("giro_*")):
        k = d.name.split("_")[-1]
        c.append((OC / f"ONPOLICY/Thor-T-C-ONPOLICY-g{k}.gguf", d / "mmlu.log", f"C-ONPOLICY-g{k}"))
    return c


def mmlu(log: Path):
    try:
        t = log.read_text(errors="replace")
    except OSError:
        return None
    if sum(1 for r in t.splitlines() if r.startswith("MCLP\t")) != 2000:
        return None
    m = re.findall(r"Final result: ([0-9.]+)", t)
    return float(m[-1]) if m else None


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    return h.hexdigest()


def tipi(g: Path) -> dict:
    sys.path.insert(0, "@@ROOT:models@@/build/thor-rexppiani2/gguf-py")
    from gguf import GGUFReader
    r = GGUFReader(str(g))
    t = {x.name: x for x in r.tensors}
    gu = sorted({t[f"blk.{L}.ffn_{f}_exps.weight"].tensor_type.name for L in range(40) for f in ("gate", "up")})
    dn = sorted({t[f"blk.{L}.ffn_down_exps.weight"].tensor_type.name for L in range(40)})
    ex2 = [n for n in t if "_exps2" in n]
    piani = [n for n in t if re.search(r"\.(attn_\w+|ssm_out|ssm_alpha|ssm_beta|ffn_\w+_shexp)[234]\.weight$", n)]
    ok = not ex2 and gu == ["TQ1_0R_1024"] and not piani and len(dn) >= 1
    return {"tensori": len(t), "gate_up_esperti": gu, "down_esperti": dn, "exps2": len(ex2),
            "densi_piani_extra": len(piani), "ok": ok}


def main():
    out = Path(sys.argv[1])
    if out.exists():
        d = json.loads(out.read_text())
        g = Path(d["gguf"])
        if g.exists() and g.stat().st_size == d["taglia"]:
            print(f"ospite gia' scelto per questa corsa: {d['nome']} ({d['mmlu']}) {g}")
            return 0
        print(f"ERRORE: ospite della corsa {g} sparito o cambiato di taglia")
        return 2
    forzato = os.environ.get("S1_OSPITE", "").strip()
    elenco, migliore = [], None
    for g, lg, n in candidati():
        a = mmlu(lg)
        stato = "ok"
        if not g.exists():
            stato = "assente"
        elif a is None:
            stato = "mmlu-incompleto"
        elif Path(str(g) + ".sha256").exists() and sha(g) != Path(str(g) + ".sha256").read_text().split()[0]:
            stato = "sha-diverso"
        elenco.append({"nome": n, "gguf": str(g), "mmlu": a, "log": str(lg), "stato": stato})
        if stato == "ok" and (migliore is None or a > migliore["mmlu"]):
            migliore = elenco[-1]
    if forzato:
        migliore = next((c for c in elenco if c["gguf"] == forzato and c["stato"] == "ok"), None) or \
            {"nome": Path(forzato).stem, "gguf": forzato, "mmlu": None, "log": None, "stato": "forzato"}
    if migliore is None:
        print("ERRORE: nessun ospite con MMLU-2000 completo")
        return 2
    g = Path(migliore["gguf"])
    sz = g.stat().st_size
    if sz != TAGLIA:
        print(f"ERRORE: ospite {g} di taglia {sz} != {TAGLIA}")
        return 2
    tp = tipi(g)
    if not tp["ok"]:
        print(f"ERRORE: ospite {g} non nel formato di C: {tp}")
        return 2
    d = dict(migliore, taglia=sz, sha256=sha(g), tipi=tp, candidati=elenco,
             regola="miglior MMLU-2000 int8 Corsair (c_onpolicy_1012.sh); a parita' prima la linea PTQ")
    out.write_text(json.dumps(d, indent=1))
    print(f"ospite: {d['nome']} MMLU {d['mmlu']} {g} ({sz} B, sha {d['sha256'][:16]}); tipi {tp}; candidati "
          + " ".join(f"{c['nome']}:{c['mmlu'] if c['stato'] == 'ok' else c['stato']}" for c in elenco))
    return 0


if __name__ == "__main__":
    sys.exit(main())
