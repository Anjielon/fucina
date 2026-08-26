#!/usr/bin/env python3
"""IL REGISTRO DELLE LEVE — ogni tecnica conosciuta, con la sua storia.

Il cuore scientifico della Fucina: ogni leva porta
  - il PRIOR: cosa ha misurato chi l'ha provata (noi su Ornith, o il paper)
  - il TEST: come si misura in minuti sul modello CORRENTE (campione + Hessiana)
  - il COSTO: byte e ore
Nessuna leva si applica senza vincere il proprio test SUL MODELLO CORRENTE.
Le "escluse" restano nel registro: escluse PER ORNITH, non per il mondo.
"""
from dataclasses import dataclass, field

@dataclass
class Leva:
    nome: str
    fase: str              # scala | compensazione | trasformazione | allocazione | riparazione | runtime
    costo_gib: float       # byte aggiunti al modello (0 = gratis)
    costo_ore: float       # tempo di applicazione sulla classe 400B
    prior: dict = field(default_factory=dict)   # modello → esito misurato
    test: str = ""         # protocollo di misura sul campione
    fonte: str = ""

REGISTRO = [
 # ── SCALA (la leva piu' grande in assoluto) ────────────────────────────
 Leva("scala_ottima", "scala", 0, 0.1,
      {"ornith397": "81%→43,5% — LA piu' grande", "teoria": "d=max mette a zero l'87%"},
      "errore pesi su 1 tensore, scala max vs minimi-quadrati", "nostra + TWN"),
 Leva("scala_dal_blocco_compensato", "scala", 0, 0,
      {"ornith397": "28,43→28,13 (+0,3)"}, "dentro il ciclo GPTQ", "nostra"),
 Leva("bof4_scala_con_segno", "scala", 0, 0.1,
      {"paper": "meglio di absmax su NF4"}, "griglia (scala,soglia) analitica sul campione", "BOF4-S 2505.06653"),

 # ── COMPENSAZIONE ──────────────────────────────────────────────────────
 Leva("gptq_dedicato_per_piano", "compensazione", 0, 4,
      {"ornith397": "43,5→28,13 (10 punti!); ⛔ il piano-1 CONGIUNTO da solo perde 10"},
      "1 tensore: dedicato vs congiunto vs nudo", "nostra (co-adattazione)"),
 Leva("cd_ternaria", "compensazione", 0, 6,
      {"paper": "QuantEase −15% vs GPTQ; INEDITA sul ternario"},
      "gara CD/annealing/ADMM su 1 esperto", "QuantEase 2309.01885 + nostra"),
 Leva("gptaq_asimmetrico", "compensazione", 0, 4,
      {"ornith397": "⛔ 32,6 vs 28,1 del simmetrico — PEGGIORA", "paper": "405B su 1 GPU"},
      "1 tensore, asimmetrico vs simmetrico", "GPTAQ 2504.02692"),
 Leva("qep_propagazione_fra_strati", "compensazione", 0, 2,
      {"paper": "massimo ai bit bassi"}, "calibrare lo strato l sull'uscita QUANT di l-1", "QEP 2504.09629"),
 Leva("ordine_babai", "compensazione", 0, 0,
      {"paper": "GPTQ ultima→prima = Babai, gratis"}, "1 tensore, due ordini", "2507.18553"),

 # ── TRASFORMAZIONI (ripiegabili, costo zero a runtime) ────────────────
 Leva("rotazione_obiettivo_ternario", "trasformazione", 0, 8,
      {"ornith397": "−35% su tensore (rotazione LIBERA ottimizzata)", "spinquant": "fino a 13 punti fra rotazioni"},
      "ottimizza R su 1 tensore, misura con H vera; ⚠️ chirurgia QuaRot per ripiegarla",
      "nostra + SpinQuant"),
 Leva("rotazione_autovettori_H", "trasformazione", 0, 1,
      {"ornith397": "⛔ −14%: gaussianizza i blocchi. Vale per QuIP/dizionari, NON per scala scalare"},
      "1 tensore, 10 minuti", "nostra"),
 Leva("magr_riduzione_massimi", "trasformazione", 0, 2,
      {"ornith397": "⛔ 39→46: serve ad absmax, non ai minimi quadrati"},
      "1 tensore; applicare SOLO se il quantizzatore usa absmax", "MagR 2406.00800"),
 Leva("haar_locale", "trasformazione", 0, 0.5,
      {"ornith397": "0,1% — nulla"}, "1 tensore", "HBLLM 2512.00862"),
 Leva("permutazione_ssr", "trasformazione", 0, 0.5,
      {"ornith397": "0,6-0,7%", "pt2llm": "utile sui loro blocchi"},
      "1 tensore, ordina per |w| medio", "PT2-LLM"),
 Leva("permutazione_intra_esperto", "trasformazione", 0, 0.5,
      {"mai_misurata": "esatta e gratis (dim privata dell'esperto)"},
      "righe gate/up ↔ colonne down con la stessa P", "PermuQuant 2605.09503"),

 # ── ALLOCAZIONE (dove spendere i bit) ──────────────────────────────────
 Leva("secondo_piano_ai_caldi", "allocazione", 2.3, 1,
      {"ornith397": "37,4→32,0 col 3%; curva ripidissima all'inizio"},
      "curva errore-pesato vs frazione caldi (counts veri)", "nostra + DynaExq"),
 Leva("frequenza_non_sensibilita", "allocazione", 0, 0.5,
      {"ornith397": "32,6 vs 38,2 — la frequenza VINCE (correlazione +0,06)",
       "mopeq": "sostiene il contrario: RIMISURARE per modello"},
      "campione 64 esperti, 2 criteri", "nostra vs MoPEQ"),
 Leva("down_sopra_gateup", "allocazione", 1.5, 0,
      {"ds4": "si", "unsloth": "si (header letti)", "mxmoe": "-2,4 ppl"},
      "campione: down a 2 piani vs gate/up a 2 piani, stesso budget", "3 fonti"),
 Leva("protezione_primi_e_ultimi_strati", "allocazione", 1.0, 0,
      {"ornith397": "traccia 35×: ULTIMI 10", "unsloth": "primi ~7 e ultimi ~4"},
      "traccia delle Hessiane per strato", "nostra + Unsloth"),
 Leva("veto_rari_critici", "allocazione", 0.5, 0.5,
      {"paper": "rarita' ≠ sacrificabilita' (Δ‖router‖)"}, "checkpoint-only", "2604.06515"),
 Leva("attenzione_lineare_q8", "allocazione", 3.0, 0.5,
      {"ornith397": "⭐ 25,4→17,1 sul MODELLO (−32,6%) per +3 GiB — LA leva migliore",
       "quamba": "lo scan e' il punto piu' fragile; <4 bit MAI dimostrato"},
      "errore dei tensori ereditati vs Q8 (per famiglia)", "nostra + Quamba + DS4"),
 Leva("super_weight_restore", "allocazione", 0.001, 0.5,
      {"paper": "perderne 1 = collasso; coordinate note solo per Llama/Mistral/OLMo/Phi"},
      "sw_scan.sh: 2 forward + spike 8× mediana", "2411.07191"),

 # ── RIPARAZIONE (post-forgia, senza toccare i ternari) ────────────────
 Leva("qesc_router", "riparazione", 0, 8,
      {"paper": "il bias di selezione e' il fattore PRINCIPALE nei MoE low-bit"},
      "overlap top-10 prima/dopo su x-teacher vs x-quant", "EAC-MoE 2508.01625"),
 Leva("norm_tweaking", "riparazione", 0, 12,
      {"paper": "GLM-130B W2 quasi-fp; ⛔ Iters=1 TASSATIVO (5 = crollo)"},
      "1 strato di prova, loss mu/sigma2 per canale", "2309.02784"),
 Leva("qzo_scale", "riparazione", 0, 24,
      {"paper": "zeroth-order, SENZA teacher, -18x memoria"},
      "500 passi su 1 strato, KL su corpus", "QZO 2505.13430"),
 Leva("eora_basso_rango", "riparazione", 1.8, 6,
      {"ornith397": "+6,4% a rango 4; sui soli caldi -80% del costo"},
      "SVD sbiancata dell'errore su 1 strato", "EoRA 2410.21271"),
 Leva("romer_copie_dei_caldi", "riparazione", 0, 1,
      {"paper": "-58% ppl sotto rumore sui pesi"},
      "sostituire i sotto-attivati con copie dei caldi su 1 strato", "ROMER 2605.11800"),

 # ── RUNTIME (non toccano il file) ──────────────────────────────────────
 Leva("campionamento_lowbit", "runtime", 0, 0,
      {"unsloth+3paper": "temp 0.6, min_p 0.03, presence 1.5, DRY, thinking ON; ⛔ XTC"},
      "A/B sulle fixture", "doc §23"),
 Leva("mtp_speculazione", "runtime", 0, 2,
      {"ornith_famiglia": "+10% a n-max 1"}, "bench con/senza", "nostra"),
 Leva("self_draft_piano1", "runtime", 0, 24,
      {"letteratura": "ρ atteso 0,15-0,30 < soglia 0,5 → probabilmente no; QSpec 1,64× se ρ alto"},
      "misura ρ (overlap top-k fra token) — 1 ora", "doc §24-25"),
]

def per_fase(fase=None):
    return [l for l in REGISTRO if fase is None or l.fase == fase]

if __name__ == "__main__":
    import collections
    n = collections.Counter(l.fase for l in REGISTRO)
    print(f"{len(REGISTRO)} leve nel registro: {dict(n)}")
    for l in REGISTRO:
        segno = "⛔" if any("⛔" in str(v) for v in l.prior.values()) else "  "
        print(f" {segno} {l.nome:<34} {l.fase:<15} {l.costo_gib:>5.1f} GiB  {l.fonte}")
