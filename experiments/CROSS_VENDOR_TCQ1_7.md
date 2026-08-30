# TCQ1_7 — riproduzione bit-esatta su due produttori (29/8/2026)

La domanda che un revisore fa per prima davanti a un formato di quantizzazione
nuovo non è «quanto è buono», è **«funziona uguale, o dipende dalla vostra
macchina?»**. Questa è la risposta, con la prova.

## Il risultato

Stesso tensore (`blk.29.attn_gate.weight`, Q8_0, 131 072 blocchi TCQ), stessi
64 blocchi campionati con `seed 0`, tre backend:

| backend | hardware | `rel_err` | stati diversi | `max abs(dscale)` |
|---|---|---|---|---|
| numpy / torch CPU | x86-64 | `0.3483299078` | — | — |
| torch **ROCm** | AMD Radeon 8060S (gfx1151), Fedora 43 | `0.3483299078` | 0/64 | `0.000e+00` |
| torch **CUDA** 12.4 | NVIDIA RTX 4060 Laptop, Windows | `0.3483299078` | 0/64 | `0.000e+00` |

Dieci cifre significative identiche. E non solo la metrica: il verificatore
riporta **`packed payload identical: True`** — i 56 byte per blocco sono gli
stessi byte, non un risultato "equivalente".

## Perché non era scontato

Il codificatore è un Viterbi esatto su un traliccio bitshift a 4096 stati, con
tail-biting: la scelta a ogni passo è un `argmin` su metriche in virgola
mobile. Ordini di riduzione diversi, FMA contratte diversamente, o un
`argmin` che rompe i pareggi in modo diverso avrebbero prodotto **stati
diversi a parità di costo** — e quindi payload diversi pur con un `rel_err`
quasi uguale. Non succede: l'ordine aritmetico interno al passo è fissato
(`(w_t - d*C)^2` in float32, somma del sopravvissuto, `argmin` al primo
indice) e tanto basta a renderlo deterministico ovunque.

## The 5 non-converging blocks are NOT an AMD defect

Sul batch da 2048 blocchi entrambe le GPU riportano `unconverged 5/2048`, con
lo **stesso conteggio**. Lo avevo annotato come possibile stranezza numerica
di ROCm: **non lo è**. È una proprietà del punto fisso del tail-biting — su
alcuni blocchi il vincolo sui 10 bit di chiusura non converge entro il numero
di passaggi concesso (`tb_passes = 3`) e si accetta l'ultimo stato. Va
dichiarato nel paper come caratteristica dell'algoritmo, non nascosto.

## Speed: portability is not performance

| | AMD Radeon 8060S | NVIDIA RTX 4060 Laptop |
|---|---|---|
| one Viterbi pass | 0.251 ms/block | **0.205 ms/block** |
| full forge | **0.530 ms/block** | 1.150 ms/block |
| 397B projection | 9.3 days | 20 days |

⚠️ **Il numero di velocità si misura solo con `torch.cuda.synchronize()`.**
Senza, si cronometrano code asincrone: un primo giro senza sincronizzazione
dava «0.008 ms/blocco, 62×», che è falso di trenta volte.

Nessuna delle due macchine rende la forgia v5 praticabile: il traliccio ha 256
passi in sequenza e Torch lancia 256 kernel piccoli, quindi la GPU aspetta più
di quanto calcoli. Cambiare produttore non lo risolve — serve **un kernel fuso**
che tenga il traliccio in registri.

## Come si rifà

```bash
# AMD (Corsair)
python3 tcq_plane.py  # (venv with torch+cuda) --verify --blocks 64 --device cuda
# NVIDIA (ARAGORN, vedi memoria project_dell_aragorn)
ssh aragorn 'cd C:\tcq && python tcq_plane.py --verify --tensor C:\tcq\tensore_full.bin --blocks 64 --device cuda'
```
Servono accanto al `.bin` il suo `.meta` (`{"tipo": 8, "dims": [ne0, ne1]}`) e
un taglio del binario **allineato a 272 byte** (8 righe Q8_0 = 1 blocco TCQ):
tagliarlo a caso rompe l'allineamento e il caricatore legge spazzatura.
