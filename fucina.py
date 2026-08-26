#!/usr/bin/env python3
"""LA FUCINA — orchestratore. `fucina.py <fase> --modello ... --budget-gib N`

Fasi: analizza · misura · gara · pianifica · forgia · collauda
Ogni fase produce un file di stato in <lavoro>/fucina-stato.json cosi' il
processo e' RIPRENDIBILE e ispezionabile (la lezione del giornale di forgia).

⚠️ SCHELETRO ONESTO: le fasi delegano agli attrezzi gia' collaudati su ODINO
(vedi PROGETTO_FUCINA.md §"Cosa esiste gia'"). La generalizzazione ad altri
modelli richiede: mappa dei nomi tensore per architettura (oggi: qwen35moe),
e il forward del teacher (oggi: transformers Qwen3_5Moe). Il resto e' generico.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

FASI = ["analizza", "misura", "gara", "pianifica", "forgia", "collauda"]

def stato_carica(lavoro: Path) -> dict:
    f = lavoro / "fucina-stato.json"
    return json.loads(f.read_text()) if f.exists() else {"fasi_fatte": []}

def stato_salva(lavoro: Path, s: dict) -> None:
    (lavoro / "fucina-stato.json").write_text(json.dumps(s, indent=1))

def analizza(a, s):
    """carta d'identita': architettura, famiglie di tensori, aritmetica del budget"""
    from leve import REGISTRO
    # [delega: lettura config.json/GGUF header — il codice di ispezione usato
    #  su Ornith il 25-26/8, da fattorizzare qui]
    s["modello"] = a.modello; s["budget_gib"] = a.budget_gib
    print(f"analizza: {a.modello} → budget {a.budget_gib} GiB")
    print(f"registro: {len(REGISTRO)} leve candidate")

def misura(a, s):
    """Hessiane + counts + traccia + super weight (attrezzi: costruisci_hessiane,
    sw_scan, teacher_pass). Una tantum per modello."""
    print("misura: Hessiane → ../odino/costruisci_hessiane.py · SW → ../guarigione/sw_scan.sh")

def gara(a, s):
    """ogni leva applicabile si misura sul CAMPIONE (5 strati × 2 matrici ×
    caldi/medi/freddi) con la Hessiana vera; vince chi migliora."""
    from leve import REGISTRO
    print("gara: protocollo = verifica_diffusa.py generalizzato; leve in gara:")
    for l in REGISTRO:
        print(f"   {l.nome:<34} test: {l.test[:60]}")

def pianifica(a, s):
    """budget → mappa tipo-per-tensore + leve attive (profilo qualita/velocita/streaming)"""
    print(f"pianifica: profilo={a.profilo} → piano dei tensori (regole fisse + esiti gara)")

def forgia(a, s):
    """delega a forgia_odino.py generalizzata (giornale, 2 flussi, tetto systemd)"""
    print("forgia: ../odino/forgia_odino.py con il piano della fase precedente")

def collauda(a, s):
    """KLD 2-pass → eval reali → Div@32 → routing-flip+ρ → onesta' → NIAH pendenza"""
    print("collauda: ../collaudo/PIANO_COLLAUDO.md")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fase", choices=FASI + ["tutto"])
    ap.add_argument("--modello", required=True)
    ap.add_argument("--budget-gib", type=float, default=94.0)
    ap.add_argument("--profilo", choices=["qualita", "velocita", "streaming"], default="qualita")
    ap.add_argument("--lavoro", default="./fucina-lavoro")
    a = ap.parse_args()
    lavoro = Path(a.lavoro); lavoro.mkdir(parents=True, exist_ok=True)
    s = stato_carica(lavoro)
    fasi = FASI if a.fase == "tutto" else [a.fase]
    for f in fasi:
        print(f"═══ FASE {f.upper()} ═══")
        globals()[f](a, s)
        if f not in s["fasi_fatte"]: s["fasi_fatte"].append(f)
        stato_salva(lavoro, s)

if __name__ == "__main__":
    main()
