<!-- Internal planning doc (Italian); English pass pending. 2026-08-28. -->

# ODINO v5 — MASTER SPEC (397B, TCQ1_7, budget ≤86 GiB)

Base di partenza: v3.2 = 84.00 GiB, ppl 6.1845±0.087; v3.3 = v3.2 + piano-2 su 44-59 (85.11 GiB, 6.1591, Δp +0.048%±0.014% a 3.4σ). HellaSwag 74.67 (±5, 300 task).

## 1. BUILD SPEC — assegnazione per classe di tensore

| classe | tensori | rate | GiB | err atteso (raw / GPTQ-wrapped) |
|---|---|---|---|---|
| esperti freddi standard | ~394/512 × 60L × 3 proj (~302 G pesi) | TCQ1_7 (2,2,1) = 1.6875 payload, 1.75 stored | 61.5 | 33% sim / **~21.5% [estrapol. ×0.65]** (oggi 43.5/28.13) |
| esperti freddissimi (coda impatto, ~90/512 per layer) | ~66 G pesi | TCQ (2,1) = 1.5625 payload, 1.625 stored | 13.4 | 37.2% sim — comunque < 43.5% odierno |
| esperti caldi, layer 44-59 | 28/512 × 16L × 3 (5.45 G pesi) | TCQ k=3 uniforme = 3.0625 payload, 3.125 stored | 1.98 | 14.1% sim vs 17.26% misurato two-plane — meglio E −0.375 bpw |
| down_exps | stessa classe del suo esperto; NESSUN piano additivo fuori 44-59 (regola profondità: solo ultimo quarto tollera correzione, 73% su entrambi i modelli) | — | — | — |
| percorso denso (attn/GDN/shexp/embed) | donor Q4_K → **saturazione Q8** su tutte le proiezioni dense censite | Q8_0 | 8.0 + **2.05** | 7.4-8.7% → **0.55%** (censimento fatto) |
| **TOTALE** | | | **~85.9 GiB** | serve a ≤86 sul target ✓ |

Conto lever 6: p2 attuale 16L×28×3 = 1.109 GiB; two-plane caldo (p1+p2, 3.5 bpw stored) = 2.22 GiB → k=3 trellis 1.98 GiB → **libera solo ~0.24 GiB**. I 2.05 GiB del Q8 si pagano con la fascia freddissima a 1.5625 (−0.95 GiB) + i 0.24: chiusura a 85.9.
Selezione caldi: **frequenza** (default robusto; impact ≈ freq nella banda buona, layer-44 top-4 identico). Impact solo se il ranking offline è già pronto al momento del forge.
Banding per-proiezione (leva 2, −0.5 ppl sul 35B): **NON validata sul 397B** — decisa dallo switch-test stanotte su v3.1. Ramo base = solo coda 44-59; ramo stretch (up/gate ovunque) solo se il test passa.
AYOT-lite (Hessiane rigenerate col 35B same-family, 1-2h) + EBSS-lite: **deviazione dichiarata** dal metodo pubblicato — dentro solo se il pre-test su Tony (sotto) non peggiora.

**Proiezione qualità composta vs v3.3 — ONESTÀ**: l'unico moltiplicatore misurato è il transfer GPTQ ×0.65 (28.13/43.54, per-tensore). Tutto il resto è estrapolazione: (a) −42% MSE esperti → ppl: NON lineare (il 27B denso dissocia ppl e generazione; il piano-2 migliorava fedeltà OVUNQUE e peggiorava il modello). (b) Attesa direzionale: ppl tra 6.16 (v3.3) e il bound donor; HellaSwag target ≥77 **[pura estrapolazione]**. (c) L'unica cifra difendibile ex-ante: errore per-tensore per classe (tabella). Il resto lo dice il Δp appaiato, non la moltiplicazione dei guadagni.

## 2. BUILD ORDER

| fase | cosa | wall-clock | gate |
|---|---|---|---|
| F0 (oggi) | switch-test v3.1 down-banding 397B (già in canna) | notte | decide ramo stretch |
| **F1 — Tony first (1 giorno)** | (a) `tcq_plane.py` sul tensore-riferimento 397B (quello del 43.54/28.13): atteso ~33 raw / ~21.5 H; (b) pack 56B + dequant CPU roundtrip bit-exact (disciplina B160); (c) Tony full-TCQ (~35 G pesi, Viterbi 1-3h) → ppl 12-chunk vs recipe 8.2501 + rep4 + Δp appaiato; (d) k=3 caldi Tony: coda vs all-layer (test regola-profondità sul trellis) | 1 g | se (a) manca 21.5±3 o (c) non batte 8.25 → STOP |
| F2 | kernel Vulkan TCQ (clone shader TQ1_0 + finestra 12-bit + 1MAD, ~12-15 ALU/peso) + variante k=3; microbench matvec vs TQ1_0 | 2-3 g | t/s ≥ 90% di TQ1_0 (QTIP: decode trellis 2-bit > fp16 su GPU — ma su NVIDIA, non gfx1151) |
| F3 | AYOT-lite Hessiane (1-2h) + EBSS-lite; pre-test 2 layer nuove-vs-vecchie su Tony | 0.5 g | err H non peggiora |
| F4 | **forge 397B v5**: 92,160 tensori esperti (~4.05 M pesi/tensore, ~16-32 K op/peso Viterbi vettorizzato → 65-130 GFLOP/tensore). Stima compute 3-26h + I/O NAS 50 MB/s ~4h (forge v3 attuale: 7.6h e2e). **Misurare s/tensore sul primo layer; soglia abort: proiezione >36h** | 0.5-1.5 g | /proc/pressure/io avg60 <40% (local I/O-pressure rule) |
| F5 | assembla + fetch Q8 denso (+2.05) + serve-check ≤86 + probe free-running PRIMA di ogni ppl (regola del 28/8: ppl e generazione dissociano) | 0.5 g | 5/5 behaviour probe |

## 3. RISK REGISTER — top 5

| # | rischio | pre-test economico |
|---|---|---|
| 1 | **Tempo Viterbi esplode** (range 3-26h è ×9 di incertezza; 92,160 tensori) | cronometrare 1 layer (1,536 tensori) prima del burn; abort >36h proiettate |
| 2 | **Interazione trellis×GPTQ**: il Viterbi è già il CVP-solver intra-blocco esatto — il transfer ×0.65 potrebbe non trasferirsi al segmented-BlockLDLQ | il tensore-riferimento in F1(a): atteso ~21.5%, misurato in ore |
| 3 | **Kernel Vulkan lento su gfx1151** (12-15 ALU vs 4 di TQ1_0; il claim bandwidth-bound è QTIP/NVIDIA) | microbench F2 prima del forge; fallback: k=2 sui soli caldi |
| 4 | **Regola-profondità vs trellis**: anche un quantizzatore migliore cambia gli output → routing (margine 8↔9 = 0.0003, churn 78.67%); il k=3 fuori coda potrebbe nuocere come il piano-2 | F1(d) su Tony: k=3 coda vs all-layer, 12-chunk + rep4, stesso giorno |
| 5 | **AYOT/EBSS-lite degrada** (generatore 35B ≠ routing a 512 esperti; variante mai pubblicata) | F3: 2 layer, confronto H-err + Δp; se neutro/peggio → Hessiane vecchie (16,640 campioni/layer, già buone) |

## 4. METRICHE DI SUCCESSO (criterio: battere la classe Q2 NETTAMENTE a pari taglia)

- **Appaiati sempre** (Δp con logit di riferimento, stessi chunk): v5 vs v3.3 e v5 vs **IQ1_M pari-taglia** (il vero avversario di classe).
- 73 fixture funzionali (mai solo ppl — regola v3b); rep4 = 0.000 richiesto.
- HellaSwag + Winogrande ≥300 task (risoluzione ±5: contano solo distacchi ≥10) — base 74.67.
- GSM8K (l'aritmetica è il punto debole misurato della famiglia senza thinking) + IFEval; sampling fisso temp 0.6/top_p 0.9/min_p 0.03/pres 1.5 (sotto i 2 bit il sampling è parte della misura).
- Vittoria dichiarabile: v5 > IQ1_M su fixtures E su ≥2 dei 4 bench con distacco fuori barra, a GiB ≤ pari; v5 ≥ v3.3 su Δp appaiato a ≥3σ.
