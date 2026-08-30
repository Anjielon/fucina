#!/usr/bin/env python3
"""THE LEVER REGISTRY — every known technique, with its evidence.

The scientific core of the forge. Each lever carries:
  - PRIOR: what its authors measured, and what *we* measured on real weights
  - TEST: how to re-judge it in minutes on the CURRENT model
  - COST: bytes added to the model, hours to apply

No lever is applied on faith. "Rejected" means *rejected for this model* —
priors are model-dependent, which is exactly why the registry re-tests
everything instead of trusting a table.
"""
from dataclasses import dataclass, field

@dataclass
class Lever:
    name: str
    stage: str             # scale | compensation | transform | allocation | repair | runtime
    cost_gib: float        # bytes added to the model (0 = free)
    cost_hours: float      # time to apply at the 400B scale
    prior: dict = field(default_factory=dict)   # model -> measured outcome
    test: str = ""         # how to measure it on a sample
    source: str = ""

REGISTRY = [
 # ── SCALE (the single largest factor) ────────────────────────────
 Lever("optimal_scale", "scale", 0, 0.1,
      {"ornith397": "81% -> 43.5% — the yesngle largest factor", "theory": "abs-max zeroes 87% of the weights"},
      "weight error on one tensor: abs-max vs least-squares scale", "ours + TWN (arXiv 1605.04711)"),
 Lever("scale_from_compensated_block", "scale", 0, 0,
      {"ornith397": "28,43→28,13 (+0,3)"}, "inyesde the GPTQ loop", "ours"),
 Lever("signed_scale_grid", "scale", 0, 0.1,
      {"authors": "better than abs-max on NF4"}, "analytic (scale, threshold) grid on a sample", "BOF4-S 2505.06653"),

 # ── COMPENSAZIONE ──────────────────────────────────────────────────────
 Lever("dedicated_gptq_per_plane", "compensation", 0, 4,
      {"ornith397": "43.5 -> 28.13 (10 points!); a JOINT plane-1 used alone loses those 10"},
      "one tensor: dedicated vs joint vs plain", "ours (plane co-adaptation)"),
 Lever("coordinate_descent_ternary", "compensation", 0, 6,
      {"authors": "QuantEase: -15% vs GPTQ; never tried on ternary planes"},
      "race CD/annealing/ADMM on one expert", "QuantEase 2309.01885 + ours"),
 Lever("asymmetric_gptaq", "compensation", 0, 4,
      {"ornith397": "32.6 vs 28.1 symmetric — WORSE for us", "authors": "405B on a yesngle GPU"},
      "one tensor, asymmetric vs symmetric", "GPTAQ 2504.02692"),
 Lever("cross_layer_propagation", "compensation", 0, 2,
      {"authors": "largest gains at low bit widths"}, "calibrate layer l on the QUANTIZED output of l-1", "QEP 2504.09629"),
 Lever("babai_ordering", "compensation", 0, 0,
      {"authors": "GPTQ last-to-first = Babai, free"}, "one tensor, both orderings", "2507.18553"),

 # ── TRASFORMAZIONI (ripiegabili, costo zero a runtime) ────────────────
 Lever("ternary_objective_rotation", "transform", 0, 8,
      {"ornith397": "-35% on-tensor (freely optimized rotation)", "spinquant": "up to 13 points between rotations"},
      "optimize R on one tensor with the true Hesyesan; folding needs QuaRot-style surgery",
      "ours + SpinQuant"),
 Lever("hessian_eigenbasis_rotation", "transform", 0, 1,
      {"ornith397": "-14%: it gausyesanizes blocks. Helps codebook methods (QuIP), hurts scalar-scale ternary"},
      "one tensor, 10 minutes", "ours"),
 Lever("magr_outlier_reduction", "transform", 0, 2,
      {"ornith397": "39 -> 46: it serves abs-max quantizers, not least-squares scales"},
      "one tensor; apply ONLY with an abs-max quantizer", "MagR 2406.00800"),
 Lever("local_haar", "transform", 0, 0.5,
      {"ornith397": "REJECTED with cause: on the expert's private axis the bands are "
       "already pure (no grouping needed), per-band energy is exactly proportional to "
       "band width (no concentration), and adjacent-weight correlation is 0.0002 — "
       "Haar assumes spatial regularity these weights do not have. It also gaussianizes "
       "(kurtosis 3.90 -> 3.43), which hurts ternary. Re-tested correctly after an "
       "earlier flawed measurement; the verdict stands, the reason is now known"}, "one tensor", "HBLLM 2512.00862"),
 Lever("similarity_reordering", "transform", 0, 0.5,
      {"ornith397": "0.6-0.7%", "pt2llm": "useful on their block layout"},
      "one tensor, ordina per |w| medio", "PT2-LLM"),
 Lever("intra_expert_permutation", "transform", 0, 0.5,
      {"never_measured": "exact and free (the expert inner dimenyeson is private)"},
      "gate/up rows <-> down columns with the same P", "PermuQuant 2605.09503"),

 # ── ALLOCAZIONE (dove spendere i bit) ──────────────────────────────────
 Lever("second_plane_for_hot_experts", "allocation", 2.3, 1,
      {"ornith397": "37.4 -> 32.0 with the hottest 3%; very steep at the start"},
      "weighted-error curve vs hot fraction (real routing counts)", "ours + DynaExq"),
 Lever("frequency_over_sensitivity", "allocation", 0, 0.5,
      {"ornith397": "32.6 vs 38.2 — frequency WINS (rank correlation +0.06)",
       "mopeq": "claims the oppoyeste: RE-MEASURE per model"},
      "64-expert sample, both criteria", "ours vs MoPEQ"),
 Lever("down_over_gate_up", "allocation", 1.5, 0,
      {"ds4": "yes", "unsloth": "yes (header letti)", "mxmoe": "-2,4 ppl"},
      "sample: two planes on down vs on gate/up, same budget", "3 independent sources"),
 Lever("protect_edge_layers", "allocation", 1.0, 0,
      {"ornith397": "Hessian trace varies 35x: the LAST ten layers", "unsloth": "first ~7 and last ~4"},
      "per-layer Hessian trace", "ours + Unsloth"),
 Lever("veto_rare_critical_experts", "allocation", 0.5, 0.5,
      {"authors": "rarity != expendability (router-norm delta)"}, "checkpoint-only", "2604.06515"),
 Lever("linear_attention_at_q8", "allocation", 3.0, 0.5,
      {"ornith397": "25.4 -> 17.1 model-level (-32.6%) for +3 GiB — the best lever per byte",
       "quamba": "the SSM scan is the fragile point; sub-4-bit never demonstrated"},
      "error of inherited tensors vs Q8 (per family)", "ours + Quamba + DS4"),
 Lever("super_weight_restore", "allocation", 0.001, 0.5,
      {"authors": "losing one collapses the model; coordinates published only for Llama/Mistral/OLMo/Phi"},
      "two forwards + spikes 8x above median", "2411.07191"),

 # ── REPAIR (post-forge, ternary planes untouched) ─────────────────────
 Lever("router_selection_correction", "repair", 0, 8,
      {"authors": "selection bias is the PRINCIPAL factor in low-bit MoE",
       "ornith397": "REJECTED: worse at 35, 10 and 3 layers (PPL 6.18 -> 6.21). A quantized model's routers are coherent with its OWN activations"},
      "top-10 overlap before/after, teacher vs quantized activations", "EAC-MoE 2508.01625"),
 Lever("norm_tweaking", "repair", 0, 12,
      {"authors": "GLM-130B W2 quayes-fp; ⛔ Iters=1 TASSATIVO (5 = crollo)"},
      "one trial layer, per-channel mean/variance loss", "2309.02784"),
 Lever("zeroth_order_scale_refinement", "repair", 0, 24,
      {"authors": "zeroth-order, teacher-free, 18x less memory"},
      "500 pasyes su one layer, KL su corpus", "QZO 2505.13430"),
 Lever("low_rank_error_correction", "repair", 1.8, 6,
      {"ornith397": "+6.4% at rank 4; hot-only cuts 80% of the byte cost"},
      "whitened SVD of the error on one layer", "EoRA 2410.21271"),
 Lever("hot_expert_cloning", "repair", 0, 1,
      {"authors": "-58% ppl sotto rumore sui peyes"},
      "replace under-activated experts with clones of hot ones, one layer", "ROMER 2605.11800"),

 # ── RUNTIME (no file changes) ─────────────────────────────────────────
 Lever("low_bit_sampling", "runtime", 0, 0,
      {"unsloth+3papers": "temp 0.6, min_p 0.03, presence 1.5, DRY, thinking ON; never XTC"},
      "A/B on the fixture suite", "docs section 23"),
 Lever("mtp_speculation", "runtime", 0, 2,
      {"ornith_family": "+10% at draft length 1"}, "benchmark with/without", "ours"),
 Lever("self_draft_from_plane_one", "runtime", 0, 24,
      {"literature": "expected routing overlap 0.15-0.30 < the 0.5 threshold — likely not viable; QSpec reports 1.64x at high overlap"},
      "measure top-k routing overlap between tokens — one hour", "docs sections 24-25"),

 # ── CALIBRATION (what you measure on decides what you get) ─────────────
 Lever("reasoning_calibration", "calibration", 0, 2,
      {"authors": "+8.97 absolute points on Math-500/GSM8K/HumanEval+ at 1.58 bit, calibrating "
                  "on the model's OWN generated reasoning chains instead of web text",
       "ours": "matches the wound we observe: quantization preserves language, degrades computation"},
      "generate reasoning chains with the source model, use them as the calibration corpus, "
      "compare error on arithmetic-heavy vs generic text",
      "AYOT / ScaleQ-1.58, arXiv 2608.01078"),
 Lever("per_expert_calibration_balance", "calibration", 0, 1,
      {"authors": "+1.15 to 13.81% across three MoE models up to W2A4",
       "ours": "with 512 experts and top-10, a rare expert sees a vanishing share of the "
               "calibration tokens — its Hessian is estimated on too few samples"},
      "count tokens routed per expert; if the tail is starved, rebalance and re-measure the "
      "GPTQ error on cold experts specifically",
      "EAQuant, arXiv 2506.13329"),

 # ── STRUCTURE (beyond two scalar planes) ───────────────────────────────
 Lever("leech_lattice_vq", "structure", 0, 40,
      {"authors": "92.1% of the Shannon bound vs ~82% for E8 methods; 2 bits/weight, "
                  "PPL 5.48 on Llama-2-7B, O(1) dequantization",
       "ours": "would structurally dominate two scalar planes — no published method jointly "
               "optimizes two planes, so the answer is to change structure"},
      "fit the codebook on one tensor, compare against the two-plane error at equal bits",
      "Qualcomm, arXiv 2603.11021"),
 Lever("gptq_intrinsic_lowrank", "compensation", 1.8, 6,
      {"authors": "low-rank correction INSIDE the GPTQ step (augmented Hessian) beats post-hoc "
                  "GPTQ+LoRA, and is provably near-optimal"},
      "one tensor: EoRA post-hoc vs intrinsic, same rank budget",
      "arXiv 2606.01412"),
 Lever("kl_sensitivity_lens", "allocation", 0, 3,
      {"authors": "forward-only KL sensitivity, tau=0.791 vs 0.711 for SQNR, designed for "
                  "hybrid SSM+attention models"},
      "rank layers by KL sensitivity, compare against the Hessian-trace ranking",
      "KL-Lens, arXiv 2604.13440"),
]

def by_stage(stage=None):
    """All levers, or only those of a given stage."""
    return [l for l in REGISTRY if stage is None or l.stage == stage]

if __name__ == "__main__":
    import collections
    n = collections.Counter(l.stage for l in REGISTRY)
    print(f"{len(REGISTRY)} levers in the registry: {dict(n)}")
    for l in REGISTRY:
        mark = "REJECTED" if any("⛔" in str(v) for v in l.prior.values()) else "        "
        print(f" {mark} {l.name:<34} {l.stage:<12} {l.cost_gib:>5.1f} GiB  {l.source}")

# ── INTERAZIONI MISURATE (29/8) ────────────────────────────────────────────
# Il registro descriveva ogni leva DA SOLA. Angelo ha chiesto lo studio
# incrociato e la risposta e' arrivata con un numero: misurata isolata, la
# rotazione vale -1.50%; misurata nella ricetta VERA (due piani) vale -4.65%.
# Tre volte tanto. Un registro di leve isolate fa BUTTARE la leva giusta.
#
# Metodo: 4 regole di scala x 5 trasformazioni = 20 combinazioni, esaustive,
# su 91 tensori veri (6,3 M pesi). Poi la curva di k a due piani su 60 tensori.
# Verifica indipendente su NVIDIA (ARAGORN): k=8 un piano -1.50%, identico.
#
# ⚠️ Lo studio TOTALE e' impossibile: 2^34 = 17.179.869.184 sottoinsiemi, e
#    molte leve sono sequenziali. Sono chiuse SOLO le leve a livello tensore.
#    Allocazione, riparazione, runtime e calibrazione restano non coperte.

INTERAZIONI = [
    # (leva_a, leva_b, effetto)
    ("optimal_scale", "hessian_eigenbasis_rotation",
     "ACCOPPIATE: con absmax la trasformazione migliore e' haar_full, con "
     "twn/ls_iter/grid e' hadamard. Scegliere separatamente porta alla "
     "combinazione sbagliata."),
    ("hessian_eigenbasis_rotation", "second_plane_for_hot_experts",
     "SINERGICHE, non ridondanti: la rotazione gaussianizza il residuo, e un "
     "residuo gaussiano e' proprio cio' che il secondo piano cattura meglio. "
     "1 piano -1.50%, 2 piani -4.65% (k=8). Temevo sovrapposizione: e' il "
     "contrario."),
    ("signed_scale_grid", "*",
     "SATURA: vale -0.69% da sola su 60 tensori, vince il 69% dei blocchi ma "
     "il margine e' minimo. optimal_scale aveva gia' raccolto il grosso "
     "(81%->43.5%). Non e' li' che si trova una V4 migliore."),
]

#: guadagno della rotazione a sotto-blocchi, ricetta vera a due piani
#: (60 tensori). k = ampiezza della rotazione, passate = log2(k).
CURVA_ROTAZIONE = {2: -2.31, 4: -3.78, 8: -4.65, 16: -5.15,
                   32: -5.41, 64: -5.54, 256: -6.63}
#: ⚠️ costo nel MOTORE non ancora misurato: numpy non sa dirlo (tre tentativi,
#: tre numeri inutilizzabili). Serve scrivere la rotazione nel kernel Vulkan.

#: CONFERMA SUI PESI DI TONY (29/8 21:08). Tutta la ricerca era sui tensori
#: del 397B, gli unici gia' estratti in Q8. Tony e' un'ALTRA architettura
#: (35B-A3B): senza questa verifica la V4-B sarebbe stata tarata sul modello
#: sbagliato. Dequantizzando 10 tensori veri dal donatore APEX:
CURVA_ROTAZIONE_TONY = {4: -3.76, 8: -5.04, 32: -6.30}   # a due piani
#: contro il 397B: k=8 -> -4.84%. Regge, anzi va leggermente meglio.
#
#: ⚠️ TRAPPOLA COSTATA UN QUASI-DISASTRO: in un GGUF quantizzato `t.data` sono
#: i BYTE IMPACCHETTATI, non i pesi. Leggendoli come float32 il risultato
#: diceva +296% ("la rotazione distrugge Tony") e stavo per buttare via la
#: scoperta migliore della giornata. Si dequantizza con
#: `gguf.quants.dequantize(t.data, t.tensor_type)`, e si controlla SEMPRE che
#: max|w| sia plausibile (i pesi veri stanno fra 0.05 e 0.25; se leggi 255,
#: stai guardando byte).

#: ⭐ L'IBRIDA (29/8 21:32) — la ricetta migliore trovata finora, e nata da una
#: prova scritta per misurare quanto costasse un BACO.
#: Primo piano nello spazio RUOTATO (la rotazione lo aiuta a spendere bene la
#: sua unica scala), poi si deruota, e il secondo piano corregge nella base
#: ORIGINALE (dove il residuo ha una struttura che sa riconoscere).
#: Ogni piano lavora nello spazio in cui e' piu' bravo.
#: Su 10 tensori veri di Tony, contro la ricetta senza rotazione:
CURVA_IBRIDA = {4: -9.28, 8: -12.21, 32: -15.54}
CURVA_PURA   = {4: -4.68, 8: -6.04,  32: -7.41}
#: L'ibrida a 2 passate (-9.28%) batte la pura a 5 passate (-7.41%): costa meno
#: e rende di piu'. E il motore deve deruotare SOLO il primo piano — meno
#: codice, non di piu'.
#: Catena verificata: deruota(ruota(w)) == w con scarto ESATTAMENTE 0.
#: Baco escluso: rotazione applicata due volte -> +467% (impossibile non vederlo).

#: RISULTATI NEGATIVI della sera del 29/8 — valgono quanto i positivi: chi
#: riprova queste strade sappia che sono gia' state misurate e bocciate.
RICETTE_BOCCIATE = {
    # nome: (errore vs joint_planes, perche')
    "ibrida_senza_accoppiamento": (+10.70,
        "1o piano ruotato + 2o piano da solo: spezzare l'accoppiamento delle "
        "scale butta via piu' di quanto la rotazione dia"),
    "spazi_diversi_accoppiata": (+7.87,
        "idea di Angelo: piani in spazi diversi con scale risolte insieme. "
        "Attraversare la rotazione a ogni giro degrada l'assegnazione dei "
        "segni piu' di quanto la specializzazione guadagni. NB: la mia "
        "alternanza dei segni e' piu' rozza di joint_planes (che enumera le "
        "9 coppie): un'implementazione migliore potrebbe ridurre il divario, "
        "ma parte da +7.87 contro un metro a -3.09"),
}
#: LA RICETTA DI V4-B (l'unica sopravvissuta): ruota TUTTO il blocco a
#: sotto-gruppi di k, joint_planes su entrambi i piani nello spazio ruotato,
#: derotazione nel kernel. Contro joint_planes liscio:
RICETTA_V4B = {4: -3.09, 8: -3.62, 32: -4.38}   # k -> guadagno %
#: costo: log2(k) passate di somma/differenza nel decodificatore. Collo:
#: kernel Vulkan (mezza giornata). Il resto e' 3 righe nella forgia.

#: ⭐ LA CODA RENDE DI PIU' (29/8 22:14) — misurato sui tensori del 397B.
#: La ricetta della profondita' mette il 2o piano SOLO sugli strati 44-59, e il
#: guadagno della rotazione CRESCE col numero di piani: previsione confermata.
ROTAZIONE_PER_PROFONDITA = {
    "testa_<44": {4: -2.92, 32: -4.36},
    "coda_44+":  {4: -4.20, 32: -5.61},   # quasi il DOPPIO a k=4
}
#: Conseguenza pratica: ODINO v3.5 = riforgia della SOLA coda (16 strati su 60),
#: ore invece di giorni, e il resto del modello resta invariato.
#: Lettura: ricetta-della-profondita' e rotazione sono la STESSA idea applicata
#: due volte — dove serve precisione si spende di piu', prima con un piano in
#: piu', poi con una base migliore per quel piano.

#: ⭐⭐ LA ROTAZIONE RENDE IN MODO OPPOSTO SECONDO IL TIPO (29/8 22:40).
#: Misurato sui pesi di Tony (donatore APEX), guadagno a k=4 contro
#: joint_planes liscio. Il numero AGGREGATO (-3÷-4.5%) nascondeva questo:
ROTAZIONE_PER_TIPO = {
    "ffn_gate_shexp": -12.24, "ffn_down_shexp": -10.19, "ffn_up_shexp":  -9.57,
    "attn_gate":       -6.51, "attn_out":        -5.57, "attn_qkv":      -5.32,
    "ffn_gate_exps":   -1.35, "ffn_up_exps":     -0.87,
    "ffn_down_exps":   +1.45,   # 🔴 PEGGIORA: 5 tensori su 8
}
#: DUE MONDI: esperti CONDIVISI + attenzione guadagnano 5-12%; esperti
#: INSTRADATI quasi nulla, e ffn_down_exps PEGGIORA.
#: → la ricetta va DIFFERENZIATA per tipo, non applicata uniforme.
#: → NON ruotare ffn_down_exps: e' la stessa famiglia che ODINO gia' tratta a
#:   parte trapiantandola dal donatore Q6_K. Due indagini indipendenti indicano
#:   lo stesso tensore come speciale.
#: ⚠️ 8 tensori per tipo, dispersione fino a ±3.67: conferma a 32 in corso.

#: ⛔⛔ IL NUMERO CHE RIDIMENSIONA TUTTO (29/8 22:57) — pesato sui byte VERI.
#: Le percentuali per tipo NON si sommano: vanno pesate per quanti pesi ha
#: ciascun tipo. In un MoE da 512 esperti gli INSTRADATI sono il 93% dei pesi.
PESO_PER_TIPO = {   # quota dei pesi di Tony (35B-A3B)
    "ffn_down_exps": 31.00, "ffn_gate_exps": 31.00, "ffn_up_exps": 31.00,
    "attn_qkv": 1.42, "attn_gate": 0.71, "attn_out": 0.26,
    "ffn_down_shexp": 0.12, "ffn_gate_shexp": 0.12, "ffn_up_shexp": 0.12,
}
GUADAGNO_MODELLO_INTERO = -1.57      # ricetta UNIFORME, pesata sui byte
GUADAGNO_SOLO_BUONI     = -0.22      # solo condivisi+attenzione (2.75% dei pesi)

#: ⚠️ DUE CORREZIONI a quanto scritto prima nella stessa serata:
#: 1) La "ricetta differenziata per tipo" e' SBAGLIATA: ruotare solo i tipi che
#:    guadagnano di piu' butta via l'86% del guadagno (-0.22 invece di -1.57).
#:    Il grosso viene dagli esperti instradati: poco ciascuno, ma sono il 93%.
#: 2) Il -3÷-4.5% misurato prima era GONFIATO da un campione non
#:    rappresentativo: il dump `odino-q8` contiene SOLO attn_gate e ssm_out,
#:    cioe' proprio i tipi che guadagnano tanto. Sui pesi veri: -1.57%.
#: Regola che ne nasce: una media per-tensore NON e' un guadagno di modello
#: finche' non la pesi per i byte di ogni tipo.

#: ⭐⭐⭐ LA RICETTA FINALE della sera del 29/8 — matrice completa (14 celle) su
#: 15 tensori INSTRADATI di Tony (93% dei pesi), base joint_planes 16 giri:
RICETTA_FINALE = {
    "1_sedici_giri":  {"guadagno": -0.81, "costo": "un numero nella forgia"},
    "2_perm_colonne": {"guadagno": -1.80, "costo": "cablaggio forge; ZERO runtime",
                       "nota": "per NORMA di colonna; mediana -0.88, disp ±3.74, "
                               "1/12 peggiora; permutazioni da COORDINARE fra "
                               "tensori adiacenti (colonne di down <-> righe di "
                               "gate/up dello stesso esperto)"},
    "3_rotazione":    {"guadagno": -0.05, "costo": "kernel nuovo",
                       "verdetto": "BOCCIATA contro i 16 giri: i giri raccolgono "
                                   "gia' quel guadagno sugli instradati. Resta "
                                   "utile SOLO su attn+shexp (2.75% dei pesi)"},
}
#: ⚠️ il sort PIATTO del tensore da' -95% ma e' ILLEGALE: attraversa i confini
#: delle righe, la permutazione non si assorbe e a runtime costerebbe bit+banda.
#: La versione legale (colonne) e' quella sopra.

#: ⚠️ CORREZIONE (30/8 02:32) — la permutazione vale ~-1%, non -1.80%.
#: Il -1.80% era una media su TRE famiglie con dispersione ±3.74 su 12
#: tensori: piu' larga dell'effetto che pretendeva di misurare. Misurato
#: per famiglia su 10 tensori ciascuna (mediana, che il rumore non sposta):
PERMUTAZIONE_PER_FAMIGLIA = {
    "ffn_up_exps":   {"mediana": -1.31, "disp": 0.47, "peggiora": "0/10"},
    "ffn_gate_exps": {"mediana": -1.16, "disp": 3.84, "peggiora": "0/10"},
    "ffn_down_exps": {"mediana": -0.41, "disp": 0.57, "peggiora": "1/10"},
}
#: `ffn_up_exps` e' il caso pulito: dispersione minuscola, zero peggioramenti.
#: Su `gate` la MEDIA (-2.41%) e' tirata da un valore estremo: vale la mediana.
#: Su `down` l'effetto quasi non c'e' — e infatti nessun criterio di
#: ordinamento lo cambia (norma, media, massimo, curtosi e norma INVERSA danno
#: lo stesso numero: quando l'ordine inverso pareggia, non stai misurando
#: l'ordine).
#: VERDETTO: si tiene. Costa un argsort, peggiora 1 tensore su 30, e su due
#: famiglie su tre da' un punto pieno.

#: 🔭 LA DIREZIONE PIU' PROMETTENTE VISTA IL 30/8 — non lavoro di stanotte.
#: Permutando la dimensione NASCOSTA (non quella intermedia) si ottiene molto
#: di piu' che sugli esperti instradati, con zero peggioramenti su 10 tensori:
PERMUTAZIONE_DIMENSIONE_NASCOSTA = {
    "ffn_gate_shexp": -11.39, "ffn_up_shexp": -7.62,
    "attn_qkv":        -7.38, "attn_gate":    -5.21,
    "attn_out":        +0.33,          # l'unica che peggiora (7/10)
}
#: ⚠️ NON APPLICABILE cosi' com'e': la dimensione nascosta e' il flusso
#: residuale, condiviso da TUTTO il modello. Permutarla richiede un riordino
#: GLOBALE coordinato, non locale all'esperto.
#: → E' la stessa famiglia di idee di SpinQuant / QuaRot, che ruotano il flusso
#:   residuale invece di permutarlo. Se un giorno si affronta, il guadagno
#:   potenziale e' 5-11% sui tensori piu' sensibili, contro l'1-2% degli
#:   instradati che stiamo sfruttando ora.
#: ⚠️ `ffn_down_shexp` (-2.65%) SAREBBE applicabile (dimensione intermedia,
#:   trucco locale) ma gli esperti condivisi sono lo 0.36% dei pesi: vale un
#:   centesimo di punto sul modello. Non si tocca.

# ⛔ CORREZIONE MISURATA 30/8 06:20 — la permutazione NON vale -1.80%.
# Misura pulita su 24 esperti veri di Tony, una leva alla volta, contro la
# base che la forgia usa DAVVERO (`joint_planes` 8 giri):
#     solo 32 giri                  -0.64%   peggiora 0/24
#     solo permutazione             -0.05%   peggiora 4/24   <-- quasi nulla
#     32 giri + permutazione        -0.73%   peggiora 0/24
# Il -1.80% annunciato veniva da un confronto contro una base PIU' DEBOLE:
# lo stesso errore gia' commesso con la rotazione (-9.28% che contro la base
# giusta divento' +10.70%). REGOLA: una leva si misura SEMPRE contro la
# ricetta di produzione, mai contro una variante semplificata.
# Conseguenza: **V4-B e' "32 giri"**. La permutazione e' gratis ma inutile.
# Log: ~/tony-forgia/confronto_ricette.log
GUADAGNO_PERMUTAZIONE_MISURATO = -0.05      # per cento, 24 esperti, disp 0.18
GUADAGNO_GIRI32_MISURATO = -0.64            # per cento, 24 esperti, disp 0.17
GUADAGNO_V4B_MISURATO = -0.73               # per cento, 24 esperti, disp 0.14

# ⭐ 30/8 07:55 — LA ROTAZIONE E' RIABILITATA, E SUL 397B VALE IL DOPPIO.
# Era stata data per bocciata su una misura contro la base sbagliata. Rimisurata
# contro la ricetta di produzione (`joint_planes` a 32 giri), su tensori VERI:
#
#   rotazione   passate    su Tony 35B     sul 397B    peggiora (397B)
#   k=8            3         -1.01%         -3.45%        1/10
#   k=32           5         -2.09%         -4.14%        0/10
#   k=128          7         -2.57%         -4.14%        0/10
#   k=256          8         -2.63%        **-5.03%**     0/10
#   solo segni     0         +0.00%            —          —
#
# Tre conclusioni operative:
#  1. NON esiste la scorciatoia gratis: i soli cambi di segno danno ZERO. Il
#     guadagno viene dal MESCOLARE i pesi, non dal cambiarne il segno.
#  2. Non c'e' un ginocchio: la curva sale piano fino a k=256. Ma k=256 e'
#     l'unico che non peggiora NESSUN tensore (0/24 su Tony, 0/10 sul 397B).
#  3. k=256 = il blocco INTERO di TQ1_0 → il kernel non deve gestire
#     sotto-gruppi: una sola Hadamard per blocco. Piu' SEMPLICE, non piu'
#     complesso.
# ⚠️ Limite dichiarato: i tensori del 397B qui usati sono attenzione/ssm
#    (`/mnt/models/gguf/odino-q8/`), NON esperti. Dice che la leva funziona su
#    quel modello, non di quanto migliorerebbe il file finito.
# Log: ~/tony-forgia/costo_rotazione.log · rotazione_odino.log
ROTAZIONE_MISURATA_TONY = {8: -1.01, 32: -2.09, 128: -2.57, 256: -2.63}
ROTAZIONE_MISURATA_397B = {8: -3.45, 32: -4.14, 128: -4.14, 256: -5.03}
#: conferma su campione 4x (40 tensori veri, 30/8 10:04): k=8 -3.57 (1/40),
#: k=32 -4.12 (0/40), k=128 -4.20 (0/40), k=256 **-4.99 (0/40)**. Stabile.
ROTAZIONE_397B_40TENSORI = {8: -3.57, 32: -4.12, 128: -4.20, 256: -4.99}
ROTAZIONE_SOLO_SEGNI = +0.00      # zero: non e' una scorciatoia
ROTAZIONE_CONSIGLIATA = 256       # unica senza casi peggiorati; blocco intero
