# EfficientQAT ternary pilot — piano (nessun training eseguito)

> Scritto il 2026-08-28. Solo lettura del codice + verifiche locali (metadata
> GGUF, config HF via web, moduli transformers installati in
> `/home/angelo/venv-catq`). Nessun training, nessuna GPU usata per produrre
> questo documento.

## 0. Obiettivo

Il PTQ (post-training, il nostro `forge_dense.py`) rompe la generazione sul
27B denso ibrido quando si spinge la ternarizzazione oltre `ffn_{gate,up,down}`.
EfficientQAT (Block-AP + E2E-QP, ACL 2025) fa QAT block-wise con ricostruzione
MSE contro l'output fp — potenzialmente recupera dove il PTQ puro collassa.
Questo documento pianifica un **pilota** (2-3 blocchi, non tutto il modello)
per capire se vale la pena investire nel porting completo.

## 1. Il modello — chi è davvero (risolto)

Il nostro `qwen38-27b-bf16.gguf` (54.66 GB, `/mnt/models/gguf/qwen38-tern/`)
ha architettura GGUF `qwen35` con chiavi `qwen35.ssm.*` +
`qwen35.full_attention_interval` + `qwen35.nextn_predict_layers` → è il
backbone testuale di **`Qwen/Qwen3.8-27B`**, rilasciato ufficialmente su
HuggingFace (mirror anche `unsloth/Qwen3.8-27B`, origine dei GGUF UD-Q4_K_XL
già in flotta come `mogavis-qwen38-exec`).

**Config reale** (letta da `huggingface.co/Qwen/Qwen3.8-27B/raw/main/config.json`):
```
architectures: ["Qwen3_5ForConditionalGeneration"]
model_type: "qwen3_5"
hidden_size: 5120
num_hidden_layers: 64
layer_types: ibrido, full_attention_interval=4 (16 layer full-attention, 48 GatedDeltaNet)
```
64 layer combacia **esattamente** con i 64 file Hessiani già presenti
(`/mnt/models/gguf/qwen38-tern/hessiane/H_00.npy`…`H_63.npy`, shape
`(5120, 5120) float32` — combacia con `hidden_size=5120`). Le Hessiane sono
riusabili come calibrazione/validazione senza ricalcolarle.

### ⚠️ Trappola: il checkpoint HF è un VLM, non un CausalLM puro
`Qwen3_5ForConditionalGeneration` (classe dichiarata nel config) è il wrapper
multimodale: `Qwen3_5Model` ha `self.visual` (torre vision) +
`self.language_model` (il backbone testuale, un `Qwen3_5TextModel` generico
via `AutoModel.from_config`). Esiste **anche** una classe testo-puro
`Qwen3_5ForCausalLM` (`transformers/models/qwen3_5/modeling_qwen3_5.py:1613`)
con `self.model = Qwen3_5TextModel(config)` — quella con `embed_tokens`,
`layers`, `norm`, `rotary_emb` allo stile Llama che EfficientQAT si aspetta.

**Implicazione pratica**: scaricare il repo `Qwen/Qwen3.8-27B` porta anche la
torre vision (peso extra, non serve a noi — il nostro GGUF non la contiene
affatto: llama.cpp separa mmproj). Per usare EfficientQAT bisogna:
1. scaricare il checkpoint completo (bf16, ~54-58 GB incluso vision tower);
2. estrarre SOLO i pesi `model.language_model.*` dal state-dict, rinominarli
   `model.*`, e caricarli in `Qwen3_5ForCausalLM` con `config.text_config`
   (una `Qwen3_5TextConfig`) — EfficientQAT **non fa questo automaticamente**:
   `main_block_ap.py:128` chiama `AutoModelForCausalLM.from_pretrained(args.model)`
   col config crudo del repo (che dichiara `ForConditionalGeneration`) →
   fallisce o carica la classe sbagliata. Serve uno script di estrazione
   state-dict prima di invocare `main_block_ap.py` (mezza giornata di lavoro,
   pattern identico a quanto già fa `gguf_surgeon.py`/`forge_dense.py` per il
   lato GGUF, ma sul lato safetensors HF).

Non esiste un convertitore gguf→HF ufficiale per questa architettura ibrida
in llama.cpp: **l'unica via pulita è ri-scaricare l'originale HF**, non
convertire il nostro GGUF a ritroso.

## 2. Quantizzatore — dove sta il fake-quant, file:riga esatti

Catena di chiamata (dal punto in cui Block-AP sostituisce i Linear):
```
quantize/block_ap.py:166-169   → per ogni nn.Linear nel blocco:
                                  quantlinear = int_linear_fake.QuantLinear(module, args.wbits, args.group_size)
quantize/int_linear_fake.py:34 → self.weight_quantizer = UniformAffineQuantizer(wbits, group_size, weight=org_module.weight)
quantize/int_linear_fake.py:41 → weight = self.weight_quantizer(self.weight)   # nel forward
quantize/quantizer.py:23-86    → class UniformAffineQuantizer  ← IL FAKE-QUANT VERO E PROPRIO
  quantize/quantizer.py:40-50  →   init scale/zero_point per-gruppo da min-max del peso (asimmetrico)
  quantize/quantizer.py:58-74  →   fake_quant(): round-clamp-dequant INT affine con zero_point
  quantize/quantizer.py:35-36  →   group_size (assert weight.shape[-1] % group_size == 0)
```

**Sostituzione per la nostra semantica TQ1_0** (ternario {-1,0,+1}, scala
per-blocco 256, **simmetrico, niente zero_point**):
- Riscrivere `UniformAffineQuantizer.__init__` (righe 24-50): niente
  `qmin/qmax` da bit-width, niente `zero_point` (parametro da rimuovere, non
  solo azzerare — un `zero_point` allenabile romperebbe la simmetria
  ternaria); `group_size` fissato a 256 (il nostro BLOCK, coerente con
  `forge_dense.py:47` `BLOCK = 256`); scala iniziale = `absmax(gruppo) `
  (stessa euristica del nostro `ternary_gpu.py`, non min-max asimmetrico).
- Riscrivere `fake_quant()` (righe 58-74): `x_int = round_ste(x / scale).clamp(-1, 1)`
  (STE già presente in `quantizer.py:10-14`, riusabile) invece di
  `round-add(zero_point)-clamp(qmin,qmax)-sub(zero_point)`; dequant =
  `x_int * scale` (nessun offset).
- `int_linear_fake.py` non richiede modifiche strutturali: il forward è già
  agnostico al tipo di quantizzatore (`weight = self.weight_quantizer(self.weight)`),
  basta che la nuova classe esponga la stessa interfaccia (`forward`,
  `.scale` come `nn.Parameter` allenabile per E2E-QP).
- `quant_parameters`/`weight_parameters` in `quantize/utils.py` iterano per
  nome parametro (`scale`, `zero_point` esclusi esplicitamente da qualche
  parte?) — da verificare riga per riga quando si passa all'implementazione:
  se filtrano per nome `"zero_point"` va tolto il riferimento, altrimenti
  l'ottimizzatore proverà a allenare un parametro che non esiste più.

### ⚠️ Secondo problema, più importante del quantizzatore: QUALI Linear toccare
`block_ap.py:166-167` fa `for name, module in qlayer.named_modules(): if
isinstance(module, torch.nn.Linear)` — **generico, tocca OGNI Linear del
blocco**, incluse le proiezioni interne del GatedDeltaNet (`in_proj_qkvz`,
`in_proj_ba`, `out_proj` — nomi verificati in
`transformers/models/qwen3_5/modeling_qwen3_5.py`, classe
`Qwen3_5GatedDeltaNet`). La parte ricorsiva vera (state-space scan, `A_log`,
`dt_bias`, `conv1d`) NON è `nn.Linear` → resta intoccata di default, questo è
corretto. Ma le **proiezioni lineari del GDN lo sono**, e il nostro
`docs/LEVERS.md:92` misura già che le proiezioni GatedDeltaNet ternarizzate
sbagliano **47-51%**, mentre a Q8 l'errore scende dal 25.4% al 17.1%. Il
comportamento out-of-the-box di Block-AP le ternarizzerebbe insieme a
`mlp.{gate,up,down}_proj`, **contraddicendo la nostra scoperta più costosa**.
**Serve un allowlist per nome modulo** in `block_ap.py:166-169` (es. regex
`r"\.mlp\.(gate|up|down)_proj$"` oppure, per replicare esattamente
`forge_dense.py`, anche i `q/k/v/o_proj` delle 16 full-attention layer)
prima ancora di lanciare il pilota — altrimenti si misura la QAT sulla
combinazione sbagliata di tensori e si butta via il segnale.

## 3. Verdetto supporto modello

**Non gira out-of-the-box.** Due incompatibilità concrete trovate leggendo il
codice *installato* di transformers 5.15.1 contro le assunzioni di
EfficientQAT (scritto contro transformers==4.40.1, pre-refactor RoPE):

1. **Firma del decoder layer** — `Qwen3_5DecoderLayer.forward()`
   (`modeling_qwen3_5.py:756+`, stesso pattern di `Qwen3NextDecoderLayer`
   verificato a `modeling_qwen3_next.py:840-848`) richiede
   `position_embeddings: tuple[Tensor, Tensor]` come argomento posizionale
   **senza default**, calcolato una volta sola a monte da
   `Qwen3_5TextModel.forward()` (`modeling_qwen3_5.py:1198`,
   `self.rotary_emb(hidden_states, position_ids)`) e propagato a ogni layer.
   `quantize/block_ap.py` chiama i layer con `layer(inps,
   attention_mask=attention_mask, position_ids=position_ids)` in DUE punti
   (`update_dataset()` riga 22-30 e il `Catcher.forward` riga 79-95, usato
   anche per catturare gli input) — **mai** `position_embeddings` → `TypeError`
   immediato al primo forward del blocco. Fix: calcolare
   `position_embeddings = model.model.rotary_emb(hidden_states, position_ids)`
   una volta e infilarlo nella chiamata al layer in entrambi i punti (patch
   piccola ma non opzionale, e specifica per architetture "RoPE-once"
   post-refactor: Llama-2 non ne soffre, per questo il repo non l'ha mai
   avuta).
2. **Nesting VLM** — vedi §1: `model.model.layers` (usato da `block_ap.py:49`)
   esiste solo se si carica `Qwen3_5ForCausalLM`, non
   `Qwen3_5ForConditionalGeneration` (quello che il config del repo dichiara
   di default). Serve l'estrazione di state-dict descritta sopra.

Nessuna delle due è un muro invalicabile — sono ~1 giorno di patch in totale
— ma **"out of the box" è falso**: chi lancia `main_block_ap.py --model
Qwen/Qwen3.8-27B` senza questi due fix ottiene un crash al primo blocco, non
un training lento o degradato.

Il resto dell'iterazione è genuinamente generico: `layers = model.model.layers`
(no dispatch per-architettura), `qlayer.named_modules()` per raccogliere i
Linear (no lista hardcoded di nomi Llama), `update_dataset()` chiama i layer
come moduli neri — con i due fix sopra e l'allowlist del §2, il resto della
pipeline Block-AP (cattura input, MSE per blocco, ottimizzatore separato
weight/quant params, salvataggio periodico) non richiede altre modifiche
strutturali.

## 4. Verdetto dipendenze

| Voce | requirements.txt | venv-catq | Verdetto |
|---|---|---|---|
| torch | `2.2.2` (CUDA) | `2.11.0+rocm7.13.0a20260426` | Nessuna versione con `torch==2.2.2` supporta ROCm gfx1151 su questo stack: si usa **venv-catq**, non i requirements pinnati. |
| transformers | `4.40.1` | `5.15.1` | **Incompatibilità strutturale, non solo di versione**: `qwen3_5`/`qwen3_next` non esistono affatto in transformers 4.40.1 (i modelli Qwen3.8/Qwen3-Next sono molto più recenti). Non esiste un transformers che soddisfi contemporaneamente "requirements.txt come pubblicato" e "supporto Qwen3.8" — installare i requirements originali non è un'opzione, si usa **venv-catq così com'è**, che è la causa diretta del blocco §3.1 (il repo non è mai stato aggiornato al nuovo contratto dei decoder layer). |
| triton | `2.2.0` | `3.6.0+rocm7.13.0a20260426` (+ `pytorch-triton-rocm 3.5.1`) | Usato SOLO in `quantize/int_linear_real.py` (packing INT reale post-training, 2/3/4/8 bit) — **non serve al pilota** (Block-AP fake-quant non lo importa; l'export finale in TQ1_0 lo facciamo con la nostra toolchain GGUF già in `fucina/`, non con `model_transfer/`). Già presente e funzionante in venv-catq comunque. |
| flash-attn | non richiesto | non installato | **Zero hit** per `flash_attn` in tutto il repo (grep su `*.py`, `quantize/*.py`, `model_transfer/*.py`) — nessun problema ROCm, il modello usa sdpa/eager di default. |
| accelerate | `0.28.0` | `1.14.0` | API usate (`infer_auto_device_map`, `dispatch_model`, `init_empty_weights`, `load_checkpoint_in_model`) stabili tra le due versioni — rischio basso. |
| bitsandbytes | `0.41.0` | non verificato in venv-catq | Non referenziato nel path Block-AP/E2E-QP core (`main_block_ap.py`, `quantize/block_ap.py`) — verificare solo se si useranno script accessori. |

**Sintesi**: si lavora dentro **venv-catq** ignorando `requirements.txt` del
repo. Questo è anche l'unico modo per avere Qwen3.8 supportato — ma è
esattamente ciò che rompe la firma dei decoder layer (§3.1), perché
EfficientQAT non è mai stato toccato da quando transformers ha introdotto
`position_embeddings`.

## 5. Disegno del pilota (2-3 blocchi, NON tutto il modello)

Scopo: verificare se Block-AP recupera dove il PTQ collassa, PRIMA di
investire nel porting completo (patch §3 + allowlist §2 + loop di training
vero).

1. **Patch minime** (§3.1 position_embeddings, §3.2 estrazione text-only,
   §2 allowlist Linear) — necessarie anche solo per far girare 1 blocco.
2. **Selezione blocchi**: 2-3 layer full-attention (più semplici, nessuna
   proiezione GDN da escludere) oppure 2-3 layer GatedDeltaNet con
   l'allowlist attiva — meglio iniziare dai layer full-attention per isolare
   il problema del quantizzatore da quello della GDN.
3. **Calibrazione**: riuso delle Hessiane esistenti
   (`/mnt/models/gguf/qwen38-tern/hessiane/H_XX.npy`, 64 file, una per layer,
   `(5120,5120) float32`) come termine di controllo/validazione della
   direzione dell'errore, PIÙ un piccolo set wikitext-2 (già supportato
   nativamente da `datautils_block.py::get_loaders`) per l'input reale ai
   blocchi durante Block-AP (le Hessiane da sole non bastano a Block-AP, che
   ha bisogno di batch di hidden-state in ingresso/uscita, non solo della
   matrice di covarianza).
4. **Quantizzatore**: sostituzione ternaria come da §2, `group_size=256`.
5. **Training**: `--epochs` basso (2-4, non i default da centinaia usati per
   Llama-2 nel paper — è un pilota, non la ricetta finale), solo sui 2-3
   blocchi scelti, resto del modello invariato (fp16/bf16 o dal donor
   Q4_K_XL esistente per il resto).
6. **Patch dei blocchi nel modello reale**: dopo Block-AP sui 2-3 blocchi,
   sostituire SOLO quei layer nel modello caricato (fp16 pieno o dal GGUF via
   dequantizzazione) e far girare generazione libera (non teacher-forced).

### Criteri GO/NO-GO (dalla nostra ricerca precedente)
| Criterio | Soglia | Fonte |
|---|---|---|
| Convergenza ricostruzione blocco | MSE Block-AP scende in modo monotono e si stabilizza entro gli epoch previsti (no NaN, no oscillazione) | prassi standard EfficientQAT |
| Generazione libera con SOLO quei blocchi patchati | non collassa (no ripetizione degenere, no garbage) su almeno 3 prompt lunghi (>200 token) | requisito esplicito della task; il nostro `generation_health.py` già misura "teacher-forced perplexity non vede l'errore che si accumula in free-running" — usarlo come guardia MA non come unico giudice |
| Norma dello stato GDN stabile | nessuna crescita/esplosione della norma dello stato ricorrente lungo la generazione libera (confrontare col checkpoint fp16 sullo stesso prompt) | coerente con `docs/RESULTS.md:1100` ("GatedDeltaNet accumula errore FFN nello stato durante la generazione") — se lo stato diverge, il problema è nella ricorrenza, non nel quantizzatore |
| Confronto con PTQ puro sugli stessi blocchi | Block-AP deve battere il PTQ ternario diretto (stesso layer, stessa scala) sia in MSE di ricostruzione sia in salute della generazione, altrimenti il costo QAT non si giustifica | criterio di business del pilota |

Se anche solo il criterio "generazione libera non collassa" fallisce sui 2-3
blocchi pilota, **NO-GO**: il problema non è la mancanza di QAT ma qualcosa
di più strutturale nel ternario applicato al GDN (coerente con
`docs/LEVERS.md:92`), e il porting completo (patch + allowlist + loop E2E-QP)
non varrebbe l'investimento.

## 6. Stima wall-clock (pilota, 2-3 blocchi, non tutto il modello)

Stime, NON misure (nessun training eseguito):
- Patch §3.1 + §3.2 + allowlist §2: **0.5-1 giornata** di lavoro (codice, non
  compute).
- Download `Qwen/Qwen3.8-27B` bf16+vision (~54-58 GB stimati, coerente con i
  54.66 GB del nostro GGUF testuale + torre vision) via rete di casa: dipende
  dalla banda disponibile, ordine di **1-3 ore** a velocità tipiche HF hub.
- Estrazione state-dict testo-puro: minuti (I/O locale, nessuna GPU).
- Block-AP su 2-3 blocchi, poche epoch, batch piccolo: sul nostro hardware
  (Ryzen AI Max+ 395, 96 GiB VRAM Vulkan/ROCm) l'ordine di grandezza per
  blocco è **decine di minuti** per pochi epoch su un dense (nessun overhead
  MoE/router), quindi **1-2 ore totali** per 2-3 blocchi — cifra ottimistica,
  da ricalibrare al primo run reale (Block-AP fa forward+backward pieno per
  blocco, non solo forward come il PTQ).
- Validazione (generazione libera + norma stato + confronto PTQ): **30-60
  minuti**.
- **Totale pilota stimato: mezza giornata di calcolo + 1 giornata di porting
  codice**, escluso il tempo di download che dipende dalla rete.

## 7. Il blocco più grande

**Non è il quantizzatore** (sostituzione localizzata, ~50 righe in
`quantize/quantizer.py`, interfaccia già astratta). **È che il checkpoint HF
ufficiale è un modello VLM (`Qwen3_5ForConditionalGeneration`) e EfficientQAT
assume un CausalLM piatto stile Llama** — servono un'estrazione di
state-dict testo-puro (nessun tool esistente nel repo la fa) *e* una patch
alla firma dei decoder layer per il nuovo contratto `position_embeddings` di
transformers post-refactor, PRIMA che si possa anche solo caricare un blocco.
Nessuna delle due patch è nel repo upstream, e senza la seconda (allowlist
del §2 sui Linear della GatedDeltaNet) il pilota misurerebbe comunque la cosa
sbagliata, ripetendo l'errore già pagato in `docs/LEVERS.md`.
