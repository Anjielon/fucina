# E1 lettera veloce (R007) — come lanciare

Ramo `cloud/lettera-e1-veloce` (da `cloud/lettera-e1`). Nessun cambio di ciò che E1 calcola: cambia quante volte lo ricalcola
e come si spende il tempo. Tutto collaudato su CPU con studente finto (`test_e1_veloce.py`, `test_residui_ternari.py`).

## Cosa c'è di nuovo

| file | cosa |
|---|---|
| `e1_veloce.py` | cache del prefisso congelato (strati < primo allenato), `avanti_da`, budget di tempo, profilo token→tempo |
| `e1_profila_passo.py` | misura il passo vero su GPU (~10-15 min, non allena) e dice se è legato ai pesi o ai token |
| `e1_recupero_lettera.py` | nuove opzioni `--cache-prefisso si/no`, `--token-cache`, `--cache-file`, `--tol-cache`, `--token-valuta`, `--budget-ore`, `--passi-misura`, `--passi-min` |
| `coda_lettera.sh` | E1_ARG di default = ricetta veloce (solo residui, budget 3 h); preflight conosce i file nuovi |
| `test_e1_veloce.py` | 7 prove: 2 piani trainer = motore, cache esatta, confine della cache, budget, profilo, driver |

## Passo 0 — misurare (10-15 min di GPU, una volta)

```bash
cd @@ROOT:forge@@/thor_v2/lettera
E2E_GDN=veloce E2E_GDN_MODULO=gdn_veloce_corsa_v5 E2E_MOTORE=triton2 E2E_CONV_DIRETTA=1 E2E_SENZA_COESISTENZA=1 \
E2E_CACHE_DEQUANT=@@ROOT:depot@@/e2e/cache_dequant E2E_DENSI_PIEGATI=x2 E2E_PIEGA_GEMM=triton PYTHONDONTWRITEBYTECODE=1 \
  @@ROOT:venv@@/bin/python3 e1_profila_passo.py --dati @@ROOT:models@@/gguf/tony-tern/lettera/corsa/items_teacher.jsonl \
  --uscita @@ROOT:models@@/gguf/tony-tern/lettera/corsa/profilo_e1 --token 1024,3072,8192,16384,32768 --budget-ore 3
```
Serve `items_teacher.jsonl` (almeno le prime 4000 righe col maestro) e la GPU col lucchetto. Restituire `profilo.json`
(in `cloud/scambio/dati/R007/`). Campi che decidono: `consiglio.legato_ai_pesi`, `consiglio.s_passo_ref`,
`consiglio.ricompila_per_forma_nuova`, `passo_da_L0.peso_fisso_a_xmax`.

## Passo 1 — la ricetta (≤ 3 h di allenamento, budget fissato dalla corsa)

`--budget-ore 3` conta dall'avvio del processo (carico dello studente, cache, init, valutazioni compresi): dopo 3 passi MISURATI
la corsa fissa il numero di passi in modo che il coseno del lr arrivi a ~zero dentro il budget (`--passi` è il massimo). Il tetto
duro `--ore-max` resta più largo (3,25 h). Scrittura del GGUF: +10-15 min su CPU, fuori budget.

**Se il profilo dice «legato ai token» (`legato_ai_pesi: false`)** — è il default di `coda_lettera.sh`:
```bash
E2E_GDN=veloce E2E_GDN_MODULO=gdn_veloce_corsa_v5 E2E_MOTORE=triton2 E2E_CONV_DIRETTA=1 E2E_SENZA_COESISTENZA=1 \
E2E_CACHE_DEQUANT=@@ROOT:depot@@/e2e/cache_dequant E2E_DENSI_PIEGATI=x2 E2E_PIEGA_GEMM=triton PYTHONDONTWRITEBYTECODE=1 \
  python e1_recupero_lettera.py --uscita OUT --dati items_teacher.jsonl --modo residui \
  --residui-strati 12-27 --residui-famiglie attn,gdn --residui-rango 32 --target raw --lam 0.5 --peso-pif 0.3 \
  --fonti-selezione logiqa,medmcqa --passi 150 --gruppi-passo 16 --token-lotto 8192 --token-valuta 16384 \
  --heldout-gruppi 100 --heldout-ogni 25 --budget-ore 3 --ore-max 3.25 \
  --cache-prefisso si --cache-file OUT/cache_prefisso.pt
```
**Se «legato ai pesi»** (tempo quasi piatto in funzione dei token): impacchettare molti più gruppi per passo e un solo lotto,
perché ogni lotto in più paga un'altra volta lo stesso costo di pesi:
`--gruppi-passo 64 --token-lotto 32768 --heldout-ogni 8` (≈ 24k token per passo; 32.768 token per lotto sono la taglia già
usata da P1, memoria nota). Il numero di passi scende (decide il budget), i token visti salgono.

**Se `ricompila_per_forma_nuova: true`**: il costo è la compilazione dei kernel a ogni lotto di taglia diversa, non il calcolo;
nessuna delle due ricette sopra lo cura. Aprire una richiesta per il padding a taglie fisse (buckets), non toccare i parametri.

## Cosa fa la cache, e cosa NO (importante)

- Stati d'ingresso del PRIMO strato allenato (12), calcolati UNA volta in lotti da `--token-cache` e riusati: passo, sonda di init
  e valutazione partono da lì. L'indietro troncato (già così in `Studente._indietro_troncato`) non li tocca.
- Esatta: gradienti uguali al bit nella prova CPU. La cache cade se cambiano ids dei token (chiave = id riga + impronta) o
  `--cache-file` di un'altra corsa (meta diversa → ignorata, ricalcolata). Memoria: 4 KiB/token a bf16 (H = 2048; 456k token ≈ 1,9 GiB di RAM).
- Cancello all'avvio: logit lettera di 2 gruppi held-out, cache contro avanti pieno, scarto massimo nel registro
  (`cancello cache`); oltre `--tol-cache` (5e-2) la corsa si ferma con rc 2. Sulla GPU i lotti di forma diversa possono dare
  differenze di arrotondamento: il numero va letto, non dato per zero.
- **Non si mette in cache nient'altro**: un confine dentro i blocchi allenati dà uno stato vecchio a ogni passo
  (prova CPU: scarto relativo dei logit 0,34). È lo stesso difetto che ha bocciato PB1.
- Il guadagno per passo è piccolo se il passo è già un lotto solo: vedi `risposte/R007-cloud-r007-1.md` §costo.

## Prove CPU
```bash
cd thor_v2/lettera && python3 test_residui_ternari.py && python3 test_e1_veloce.py      # ~3-5 min (torch CPU)
```
Dipendenze: `torch numpy gguf pyarrow transformers`. Nessuna rete, nessun peso.
