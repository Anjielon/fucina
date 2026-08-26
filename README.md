# FUCINA — forgia ternaria generale per modelli MoE

Ternarizza **qualsiasi** modello MoE in formato GGUF al 2-piani TQ1_0
(1,69-3,38 bit/peso), applicando automaticamente le leve di quantizzazione
validate sul campo — nata dalla costruzione di **ODINO** (Ornith-1.5-397B
intero, 512 esperti, ~88 GiB, residente in 96 GiB di VRAM UMA).

## Il metodo (misurato, non promesso)

- **2 piani ternari congiunti** `W ≈ d1·T1 + d2·T2` (blocchi da 256, scala f16)
  SOLO sugli esperti **caldi** (i più instradati, dai `counts` dell'imatrix):
  il 3% più caldo cattura ~18% del traffico.
- **GPTQ con Hessiane vere** (H=XXᵀ misurata sulle attivazioni) per il piano
  dedicato dei freddi: 37,95% → 28,13% d'errore (la co-adattazione del piano-1
  congiunto è la trappola n.1).
- **Riordino caldi-primi alla nascita** (esperti + righe del router): il motore
  assume che il piano-2 copra il prefisso 0..K-1.
- **Verifica al confine per-strato**: il caldo n.1 ricostruito dai 2 piani
  DEVE combaciare con la sorgente (assert, non speranza).
- Attenzione/embedding/router: copiati alla qualità della sorgente (Q6/Q8).

## Moduli

| file | cosa fa |
|---|---|
| `forgia_gguf.py` | la forgia: GGUF sorgente → GGUF TQ1_0 a 2 piani, giornale di ripresa |
| `ternario_gpu.py` | quantizzatore GPU: 2 piani congiunti + GPTQ a propagazione intera |
| `tq1_pack.py` | impacchettamento TQ1_0 su GPU + permutazioni (autotest vs decodificatore ufficiale) |
| `due_piani.py` | preparazione Hessiane (Cholesky, smorzamento) |
| `costruisci_hessiane.py` | dai dump delle attivazioni alle H per-strato |
| `leve.py` | registro delle 29 leve note (prior, costo, fonte scientifica) |
| `fucina.py` | orchestratore: applica e MISURA ogni leva sul modello corrente |

## Uso

```bash
python3 forgia_gguf.py --sorgente modello.gguf --uscita modello-tern.gguf \
    --caldi 28 --imatrix imatrix.gguf [--hessiane DIR]
```

Serve il motore llama.cpp con supporto TQ1_0 + 2° piano
(`*_exps2`, metadato `<arch>.expert_count2`) e i kernel Vulkan TQ1_0
(mul_mm **e** mul_mat_vec/(id) — collaudarli SEPARATAMENTE con
`test-backend-ops`, con offset non nulli e n=1: sono shader diversi).

## Le 7 regole del collaudo (imparate a caro prezzo, 26/8/2026)

1. Ogni convenzione condivisa fra due componenti → **assert al confine**.
2. La **prova di fumo** (4 domande verificabili) è parte della forgia.
3. Mai chirurgia sul file senza copia dei byte toccati.
4. Campionamento low-bit anche nei test (temp 0.6, min_p 0.03, presence 1.5).
5. Il **thinking** si collauda a parte (sotto 2 bit può rompersi da solo).
6. Un tipo quantizzato nuovo entra in `test-backend-ops` il giorno stesso.
7. Matrice e vettore sono shader diversi: il collaudo dell'uno non copre l'altro.

## Riferimenti scientifici

PTQTP (arXiv 2509.16989) · PT²-LLM · GPTQ (2210.17323) · QuantEase ·
Super Weight (2411.07191) · Norm Tweaking · QESC · HOBBIT (2411.01433) ·
Quamba — catalogo completo con le misure in `docs/RICERCA_TERNARIZZAZIONE.md`
del progetto MOGAVIS (~85 lavori letti, ogni leva misurata su pesi veri).
