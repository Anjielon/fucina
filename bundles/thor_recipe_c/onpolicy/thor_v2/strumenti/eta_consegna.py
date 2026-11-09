#!/usr/bin/env python3
"""ETA di CONSEGNA di Thor-T dalla corsa E2E viva, in UNA riga per il ping (26/9/2026, Angelo: «tieni sempre sotto
controllo i tempi di consegna», «non vedo l'ETA»). Legge SOLO il registro.log della corsa.

CONSEGNA (Angelo): Thor funzionante (sonda parlata) che batte NETTAMENTE il rivale = MMLU >= 78,15 + 2 sigma (~0,95 con
~2000 domande) = ~80,05. Bersaglio in KL held-out dalla CURVA MISURATA (26/9 22:0x, taratura_kld.sh): llama-perplexity
--kl-divergence contro il teacher Q8 (base), wikitext-2 test, -c 4096, 20 chunk (40.940 token), strumento standard di
llama.cpp; la KL dal modello pieno come misura di fedelta' delle quantizzazioni: Dutta et al. 2024, "Accuracy is Not All
You Need" (arXiv 2407.09141). KLD media wikitext: rivale Q2_K 0,2201 (top-1 81,5%), B 0,3506 (77,3%), v20 0,4333 (73,9%).
Curva MMLU <-> KLD monotona su teacher (82,00 @ 0) · Q2_K (78,15 @ 0,2201) · B (71,95 @ 0,3506); v20 (76,50 @ 0,4333)
e' FUORI curva (altra famiglia: PPL alta a MMLU buono), quindi l'incertezza sistematica fra famiglie e' di punti MMLU.
MMLU 80,05 cade fra teacher e Q2_K -> KLD wikitext ~0,1115 (MMLU del Q2_K +-0,95 -> 0,089..0,148). In unita' held-out
(KL torch sulle nostre righe) si scala col rapporto misurato su B, 0,39907 / 0,3506 = 1,138 -> ~0,127 (0,10..0,17).
Prima (interpolazione lineare B-teacher) era 0,077: la curva misurata la allenta. Resta un'IPOTESI fra famiglie diverse:
la verita' e' il primo MMLU vero di un checkpoint della nostra famiglia.
Prima di due misure held-out l'ETA e' una FORCHETTA: bersaglio al passo ~13 (10% dell'epoca) .. passo 128 (epoca intera).
Dopo: proiezione lineare della discesa misurata (prudente all'inizio, quando la discesa e' piu' ripida).
"""
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

CORSA = Path(sys.argv[1] if len(sys.argv) > 1 else "@@ROOT:lavoro@@/e2e/allena/EB2")
MMLU_B, MMLU_T, MMLU_Q2K, KL_B, SIGMA = 71.95, 82.00, 78.15, 0.39907, 0.95
MS_TOKEN_STIMA = 28.3                    # B_ind32 a 40 strati da M0 (col teacher acceso): usato solo prima del passo 2
TOKEN_PASSO = 32 * 4096
HELDOUT_S = 23 * 60                      # misura held-out (col teacher acceso)


# curva misurata (MMLU, KLD wikitext) e scala held-out/wikitext misurata su B (vedi docstring)
CURVA_KLD_WIKI = [(82.00, 0.0), (78.15, 0.220075), (71.95, 0.350642)]
HELDOUT_SU_WIKI = KL_B / 0.350642


def kl_per_mmlu(m: float) -> float:
    for (m0, k0), (m1, k1) in zip(CURVA_KLD_WIKI, CURVA_KLD_WIKI[1:]):
        if m1 <= m <= m0:
            return HELDOUT_SU_WIKI * (k0 + (k1 - k0) * (m0 - m) / (m0 - m1))
    raise ValueError(f"MMLU {m} fuori dalla curva misurata")


def main() -> int:
    righe = (CORSA / "registro.log").read_text().splitlines() if (CORSA / "registro.log").exists() else []
    passi, held, tot, heldout_ogni = [], [], 128, 2
    for r in righe:
        t = datetime.fromisoformat(r[1:26]) if r.startswith("[") else None
        m = re.search(r"passo (\d+)/(\d+): KL train ([\d.]+)\s+([\d.]+) s", r)
        if m:
            passi.append((int(m[1]), float(m[4]), t))
            tot = int(m[2])
        m = re.search(r"held-out al passo (\d+): KL ([\d.]+)", r) or re.search(r"passo (0): held-out KL ([\d.]+)", r)
        if m:
            held.append((int(m[1]), float(m[2]), t))
    bersaglio = kl_per_mmlu(MMLU_Q2K + 2 * SIGMA)
    veri = [s for p, s, _ in passi if p > 1]                       # il passo 1 contiene la calibrazione G1
    s_passo = sum(veri[-3:]) / len(veri[-3:]) if veri else MS_TOKEN_STIMA * TOKEN_PASSO / 1000
    fonte = "misurato" if veri else "stimato"
    ultimo = passi[-1][0] if passi else 0
    t_rif = (passi[-1][2] if passi else (held[-1][2] if held else None)) or datetime.now().astimezone()
    kl_ora = held[-1][1] if held else KL_B
    stato = f"EB2 passo {ultimo}/{tot} ({s_passo / 60:.0f} min/passo {fonte}) · KL held-out {kl_ora:.3f} -> bersaglio {bersaglio:.3f}"

    def a_passo(p: int) -> datetime:
        n = max(p - ultimo, 0)
        return t_rif + timedelta(seconds=n * s_passo + (n // heldout_ogni + 1) * HELDOUT_S)

    if len(held) >= 2 and held[-1][1] < held[0][1]:
        (p0, k0, _), (p1, k1, _) = held[0], held[-1]
        if k1 <= bersaglio:
            print(f"⏱ ETA CONSEGNA: bersaglio KL RAGGIUNTO al passo {p1} -> verifica di consegna (MMLU+sonda) · {stato}")
            return 0
        rate = (k0 - k1) / (p1 - p0)
        p_b = p1 + (k1 - bersaglio) / rate
        tetto = KL_B * (1 - 0.33)                        # sole scale: -10..-33% della distanza (letteratura del progetto)
        if bersaglio < tetto:
            import json as _j
            pv = _j.loads(Path("@@ROOT:forge@@/stato/pv_tempi.json").read_text())
            if pv.get("verdetto"):
                print(f"⏱ ETA CONSEGNA: {pv['verdetto']} · bersaglio KL held-out {bersaglio:.3f} · {stato}")
                return 0
            t0 = datetime.fromisoformat(pv["prima_discesa_pv"]).astimezone()
            lo = t0 + timedelta(minutes=pv["passi_pv_min"] * pv["min_per_passo_pv"] + 75)
            hi = t0 + timedelta(minutes=pv["passi_pv_max"] * pv["min_per_passo_pv"] + 75)
            print(f"⏱ ETA CONSEGNA (via FASE 2 PV, stima): {lo:%d/%m %H:%M} .. {hi:%d/%m %H:%M} · tappe: prototipo "
                  f"{datetime.fromisoformat(pv['prototipo_verificato']):%d/%m %H:%M} -> GPU "
                  f"{datetime.fromisoformat(pv['integrazione_gpu_collaudata']):%d/%m %H:%M} -> 1a discesa PV {t0:%d/%m %H:%M}"
                  f" (da li' ETA misurato) · sole scale: tetto KL ~{tetto:.2f} > bersaglio {bersaglio:.3f} · {stato}")
        else:
            print(f"⏱ ETA CONSEGNA: {a_passo(round(p_b)):%d/%m %H:%M} (bersaglio al passo ~{p_b:.0f}, proiezione su "
                  f"{len(held)} misure) + ~75 min di verifica · {stato}")
        return 0
    fatte = {p for p, _, _ in held}
    prossima = ultimo if (ultimo % heldout_ogni == 0 and ultimo not in fatte) else heldout_ogni * (ultimo // heldout_ogni + 1)
    print(f"⏱ ETA CONSEGNA: {a_passo(13):%d/%m %H:%M} .. {a_passo(tot):%d/%m %H:%M} (forchetta finche' non misuro la "
          f"discesa: 1a misura vera ~{a_passo(prossima):%H:%M}) · {stato}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
