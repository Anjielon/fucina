# LA FUCINA — forgia ternaria automatica per qualsiasi gigante

> Ideata da Angelo il 26/8/2026. Il metodo distillato da ~85 lavori letti e
> ~30 misure fatte in casa su Ornith-1.5-397B, generalizzato: dai a LA FUCINA
> un modello (safetensors HF o GGUF) e un budget, e lei produce la forgia
> ternaria ottima — misurando OGNI leva sul modello corrente prima di
> applicarla. Da pubblicare insieme a ODINO: il modello e' UNA uscita, il
> metodo e' il contributo.

## Il principio (pagato caro, due volte)
**Nessuna leva si applica per fede.** Su Ornith abbiamo misurato che:
- la rotazione negli autovettori di H DANNEGGIA (−14%) — ma su modelli con
  altre distribuzioni potrebbe valere il −35% che ha dato ottimizzata
- MagR peggiora (39→46%) — ma sui quantizzatori absmax e' oro
- MoPEQ (sensibilita') perde dalla frequenza — ma con top-k grandi puo' vincere
- il piano-1 congiunto perde 10 punti dal dedicato — scoperto SOLO misurando
Le «porte chiuse» sono chiuse PER ORNITH. La Fucina le riapre a ogni modello:
ogni leva ha un test da minuti su un CAMPIONE (pochi esperti/tensori con la
Hessiana vera) e si applica solo se vince li'.

## Le 6 fasi

### 1. ANALIZZA (minuti) — la carta d'identita' del modello
- architettura: MoE? quanti esperti, top-k, shared, strati lineari/attn,
  tensori fusi, teste MTP
- pesi: distribuzione per famiglia di tensori, bpw attuale se GGUF
- vincoli: budget VRAM target → bit/peso disponibili (l'aritmetica del budget)

### 2. MISURA (ore, una tantum) — gli ingredienti
- **Hessiane per strato** (cattura attivazioni con la patch cb_eval — o
  imatrix se il modello non gira): anisotropia α per strato
- **counts/frequenze** degli esperti (imatrix o telemetria router)
- **traccia di sensibilita'** per strato · **Super Weight scan** (2 forward)
- corpus di calibrazione: RAGIONAMENTO auto-generato (AYOT) con chat template

### 3. GARA (ore) — ogni leva si guadagna il posto
Su un CAMPIONE rappresentativo (nostro protocollo: 5 strati × 2 matrici ×
esperti caldi/medi/freddi), per ogni leva del registro:
  guadagno = errore_uscita(senza) − errore_uscita(con)  [Hessiana vera]
Le leve sono ORDINATE per guadagno/GiB e la selezione e' Pareto (alla Unsloth,
ma su campione invece che su 121 forgie complete).
Solver in gara dove esistono alternative: CD (QuantEase) vs annealing vs ADMM.

### 4. PIANIFICA — dal budget al piano dei tensori
Input: budget GiB + profilo. Output: la mappa tipo-per-tensore + leve attive.
| profilo | criterio |
|---|---|
| `qualita` | max qualita' dentro il budget (2° piano ai caldi, Q8 protetti) |
| `velocita` | tutto residente, bit minimi, MTP se c'e' |
| `streaming` | 2 piani pieni + partizione caldo/freddo per la cache |
Regole fisse (validate 3+ volte in letteratura + da noi):
- router/norm MAI quantizzati (F32)
- percorso ricorrente/attenzione mai sotto Q6-Q8 (⛔ scan <4 bit: MAI, nessuno
  al mondo l'ha fatto funzionare)
- down un gradino sopra gate/up · primi e ultimi strati un gradino sopra
- Super Weight ripristinati in half (costa byte)

### 5. FORGIA (ore) — con tutto quello che abbiamo gia'
- lettura strato-per-strato (2 flussi, misurare la banda della sorgente!)
- quantizzazione GPU (i nostri kernel torch: scala ottima → GPTQ dedicato per
  piano → CD se ha vinto la gara)
- scrittura sequenziale con GIORNALE (ripresa da crash = 1 tensore)
- tetto systemd su memoria (MAI soffocare la macchina)

### 6. COLLAUDA (ore) — il protocollo consolidato
KLD 2-pass (pre-filtro, minuti) → fixture/eval reali → Divergence@32 →
routing-flip + ρ → onesta' epistemica → NIAH differenziale (pendenza).
Campionamento raccomandato nel report finale (temp 0.6, min_p, presence, DRY).

## Cosa esiste gia' (da generalizzare, non da scrivere)
| pezzo | file oggi | fase |
|---|---|---|
| cattura Hessiane | debug.cpp patch + costruisci_hessiane.py | 2 |
| verifica diffusa a campione | verifica_diffusa.py | 3 |
| quantizzatore GPU 1/2 piani | ternario_gpu.py | 5 |
| CD ternaria | guarigione/cd_ternario.py | 3,5 |
| forgia con giornale | forgia_odino.py | 5 |
| chirurgo GGUF | guarigione/patch_gguf.py | 5,v3.2 |
| teacher pass | guarigione/teacher_pass.py | 2 |
| collaudo | collaudo/PIANO_COLLAUDO.md | 6 |
| sorveglianza pressione io | stato.sh/sorveglia.sh | 5 |


## Profilo CELLULARE (idea di Angelo, 26/8) — i MEDI in tasca
La Fucina non e' solo per i giganti: i 20-30B densi oggi NON ternarizzabili
(gli strumenti di serie li ammazzano: TQ1_0 stock = 73% di errore) diventano
modelli DA TELEFONO con la nostra pila.

**Caso di studio Qwen3.8-27B** (misurato dal GGUF di casa, 27,32 mld):
| componente | quota | tipo |
|---|---|---|
| FFN down/gate/up | 17,4 mld (64%) | TERNARIO (TQ2_0 su ARM: kernel gia' in mainline) |
| attn+GatedDeltaNet | 7,3 mld | Q4K-Q6 (⛔ regola Quamba: la ricorrenza mai sotto) |
| embed+testa | 2,5 mld | Q6 |
→ **9,5-11 GiB**: entra in un flagship da 16 GB (il Q4 attuale: 16,5, non entra).
Velocita': banda ~50-70 GB/s ÷ ~10 GiB/token = **5-7 t/s**, con la testa MTP
(che il 27B HA) → **7-10 t/s**.

**Adattamenti per i densi**: niente leve di frequenza-esperti → allocazione per
STRATO (traccia) e per famiglia; Super Weight nei down dei primi strati
(scan obbligatorio); 2° piano = semplice ggml_add di due matmul (piu' facile
del mul_mat_id mascherato); ⭐ sul telefono il ternario paga DOPPIO: memoria
E calcolo (FairyFuse: 29,6x sul kernel CPU; bitnet.cpp TL1/TL2 nati per ARM).
⚠️ Ridondanza minore del MoE → attesa piu' perdita relativa: la GARA delle
leve (fase 3) e' ancora piu' importante, non meno.

## La pubblicazione
1. repo GitHub `fucina` (fork llama.cpp linkato e pinnato) — MIT
2. ODINO come PRIMA uscita dimostrativa + il documento (2.000+ righe)
3. invito al mondo: «portaci il tuo gigante» — ogni modello nuovo arricchisce
   il registro con le sue misure (le leve vincono o perdono PER MODELLO)
