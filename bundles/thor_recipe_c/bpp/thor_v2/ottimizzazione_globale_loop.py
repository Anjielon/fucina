#!/usr/bin/env python3
"""Loop ESTERNO multi-finestra per l'ottimizzazione GLOBALE cross-strato —
compito ricevuto da Angelo il 24/9 03:2x: "finisci di scrivere tutto quello
che serviva per fare la correzione, non fermarti a meta'". `finestra_globale.py`
(03:11, gia' verificato, 4 test verdi) sa ottimizzare UNA finestra fissa di W
strati; qui manca il pezzo che FA SCORRERE quella finestra su tutta la rete:
parte dallo strato 0, avanza di stride (=W senza overlap, =W/2 con overlap),
e ALL'INIZIO di ogni nuova finestra RI-INSTRADA il MoE (mai fisso per tutta
la corsa — vedi PIANO.md §02:26, proposta scritta li' e cablata qui) loggando
il tasso di route-flip rispetto all'instradamento precedente.

⛔ VINCOLO DI SESSIONE (GPU condivisa, uguale agli altri file di stanotte):
qui SOLO strati GIOCATTOLO sintetici (stessa scala di finestra_globale.py:
hidden=8, intermedio=6). Mai il modello Thor intero, mai un training reale su
di esso. Questo file verifica il CABLAGGIO del loop esterno — che avanzi nel
modo giusto, copra ogni strato la volta giusta, ricalcoli il routing al bordo
di ogni finestra e produca il log di route-flip — non la qualita' della
ricostruzione su pesi veri (quello e' il passo SUCCESSIVO, fuori perimetro
qui, esattamente come gia' dichiarato in finestra_globale.py).

Importa (mai duplica) da `finestra_globale.py`: `StratoMoEGiocattolo`,
`congela_routing`, `crea_esperti_giocattolo`, `errore_lemma1_finestra`,
`parametri_ternari`, `passo_ottimizzazione`, `verifica_gpu_libera`,
`_chiama_quant`, `_perturba_z`. Importa da `quant_ternaria_globale.py`:
`TernarioDiff`. Nessuna di queste funzioni e' riscritta qui.

## Overlap 50% o finestre non sovrapposte — quale e' meglio, e perche'

⚠️ Onesta' su una citazione: il compito ricevuto attribuisce al "piano del
02:26" la raccomandazione esplicita di usare l'overlap "per continuita'
dell'errore fra finestre". Rileggendo `stato/PIANO.md` §02:26 per intero
("RICERCA: routing MoE nella via completa") quella sezione parla di
ri-instradamento periodico e monitoraggio del route-flip — non contiene
letteralmente la frase citata. La riporto qui perche' il compito la cita,
ma la RACCOMANDAZIONE SOTTO e' costruita sui fatti verificabili nel codice
e nel piano (tabella costi §~02:0x, semantica di `e0` in
`finestra_globale.errore_lemma1_finestra`), non su quella citazione.

**Costo, misurato/calcolato nel piano (tabella "Tempo totale per 1000
passi", W=4, rete a 40 strati)**:

| modalita' | finestre per coprire 40 strati | tempo totale |
|---|---|---|
| senza overlap (stride=W) | 10 | ~3h20min |
| overlap 50% (stride=W/2) | 19 | ~6h20min |

L'overlap **raddoppia** il costo — non e' un dettaglio, e' il prezzo pieno
(coerente col mandato "zero compromessi, mai scorciatoie": si dichiara il
costo, non si nasconde).

**Cosa compra quel raddoppio — la scoperta fatta scrivendo QUESTO file, non
nel piano**: `errore_lemma1_finestra` (finestra_globale.py) accetta `e0`
come "errore accumulato ENTRANTE (da fuori finestra); None => zeros_like(h0),
assunzione ESPLICITA 'nessun errore pregresso' (vera solo se la finestra
parte dal primo strato della rete, o se e0 e' stato seminato altrove dal
chiamante — MAI un default silenzioso spacciato per il caso generale)".
Questo loop risolve quel "seminato altrove" nel modo ESATTO, non con una
stima: dopo aver ottimizzato una finestra, un ultimo forward no_grad (con
i codici APPENA aggiornati) legge `h_l[offset]`/`e_l[offset]` allo stride
esatto di avanzamento e li passa come `(h0, e0)` alla finestra successiva —
`offset` e' la distanza fra l'inizio della finestra nuova e l'inizio di
quella vecchia, quindi il valore letto e' l'errore REALE accumulato sotto i
codici correnti, non una stima importata da un altro metodo (come invece
temeva la nota "Onesta' sui numeri" del piano, scritta PRIMA che
`finestra_globale.py` esistesse). **Questo vale ALLO STESSO MODO con o
senza overlap**: la continuita' di `e0`/`h0` fra finestre e' gia' esatta in
entrambi i casi, quindi NON e' li' che l'overlap guadagna qualcosa.

Il guadagno vero dell'overlap e' un altro, e si vede nel test W=2 gia'
verificato in `finestra_globale.py` (`_test_due_layer`,
"accoppiamento CROSS-STRATO dimostrato": il gradiente della sub-loss
dell'ULTIMO strato di una finestra raggiunge i PARAMETRI del PRIMO strato
della stessa finestra, `d(loss_strato2)/d(parametri_strato0) != 0`). Senza
overlap, ogni strato riceve questo accoppiamento incrociato **da una sola
direzione** (quella della finestra a cui appartiene, una volta sola: dopo,
la finestra si chiude e i suoi codici non vengono piu' toccati). Con overlap
50%, gli strati della meta' di coda di una finestra sono anche la meta' di
testa di quella successiva: gli STESSI oggetti `TernarioDiff` (stesso
`pesi_esperti`, mai ricreati — l'addestramento persiste, non riparte) vengono
riottimizzati una SECONDA volta, questa volta accoppiati con gli strati che
vengono DOPO invece che con quelli che vengono prima. Ricevono cosi'
accoppiamento incrociato **da entrambe le direzioni**, invece che da una
sola — coerente con l'obiettivo Eq.9 del paper (accoppiamento fra TUTTI gli
strati), che la finestra scorrevole approssima per costruzione tagliandolo a
pezzi di W. L'overlap non elimina l'approssimazione (il taglio a finestre
resta un'approssimazione della via globale letterale, dichiarata
impraticabile in memoria nel piano), ma la rende meno cieca esattamente nei
punti (i bordi) dove il taglio la introduce.

**Raccomandazione**: default `overlap=True` (stride=ceil(W/2)), coerente col
mandato "sempre la via completa" quando il costo (qui, tempo GPU: 2x) e'
dichiarato e accettabile; entrambe le modalita' restano configurabili e
testate qui sotto, la scelta finale sull'uso reale (Thor vero, non
giocattolo) resta di Angelo — non e' decisa in silenzio da questo file.

⛔⛔ **Aggiornamento 03:33, filone concorrente stessa notte — la tabella
costi sopra usa un W=4 che NON entra in VRAM su Thor vero**: un test
dedicato (`thor/verifica_memoria_finestra_reale.py`, shape VERE, pesi
random) ha misurato che la stima di PIANO.md 02:31 (~75 GiB per W=4) e'
**smentita** — il costo reale supera i ~95 GiB disponibili e va in OOM (misurato
~94.76 GiB), e W=3 fallisce **quasi allo stesso muro** (94.69 GiB, non un
terzo in meno). **La soglia reale misurata e' W=2** (65.32 GiB, margine
comodo). Causa: `errore_lemma1_finestra` (non toccato da nessuno dei due
filoni) trattiene nel grafo DUE forward "con grad" per famiglia per OGNI
strato della finestra fino a un unico `.backward()` finale — il costo cresce
piu' che linearmente con W, non "z + un forward" come stimato. Questo
loop resta CORRETTO qualunque sia W (la pianificazione delle finestre non
assume nulla sulla sua taglia oltre `1<=W<=n_strati_totale`); il DEFAULT di
`ConfigLoop.W` sotto e' stato aggiornato a **2** per riflettere il vincolo
REALE oggi noto, restando comunque un parametro — chi lo usa su Thor vero lo
sa gia' cambiare, ma un default onesto non deve promettere una taglia che
si sa gia' non entrare.

## Overlap con W dispari — arrotondamento dichiarato

`stride = max(1, ceil(W/2))`. Per W pari e' overlap ESATTO al 50% (W=4 ->
stride=2, un layer su due condiviso). Per W dispari e' un arrotondamento per
eccesso (W=3 -> stride=2, un solo strato condiviso su 3 = 33%, non un 50%
impossibile a interi dispari con questa convenzione) — dichiarato qui, non
nascosto in una tabella che promette "50%" senza dire come si arrotonda.

## Ri-instradamento — cosa fa QUESTO loop, letteralmente

All'inizio di ogni finestra (compresa la prima): si prende il flusso
QUANTIZZATO corrente in ingresso alla finestra (`h_hat = h0+e0`, exatto per
la finestra 0 dove e0=0 per costruzione — nessun errore ancora introdotto),
lo si propaga attraverso gli strati della finestra usando i codici CORRENTI
in forma DURA (`hard=True`, i codici ternari reali {-1,0,+1} — coerente con
"instrada sul flusso quantizzato corrente", non su un surrogato morbido), e
per ciascuno strato si ricalcola l'instradamento con `congela_routing` (STESSA
funzione di `finestra_globale.py`, mai riderivata). Il routing COSI'
ottenuto resta FISSO per tutti i passi di training di quella finestra
(nessuna macchina STE/Gumbel sul router — stessa disciplina di
`finestra_globale.py`, §PIANO.md 02:26 punto 1). Il tasso di route-flip si
logga per OGNI strato della finestra, confrontato con l'instradamento che
era in vigore per quello strato PRIMA di questo ricalcolo (dalla
costruzione iniziale della rete se e' la prima volta che lo strato viene
toccato, o dall'ultima finestra che lo conteneva se e' uno strato di
overlap).

⛔ Onesta' su cosa NON fa: se il tasso di route-flip cala verso zero dopo le
prime finestre, questo file lo LOGGA e basta — nessuna logica qui decide
"quindi il freeze e' sicuro, fermiamoci" o simile. La decisione (§PIANO.md
02:26 punto 3: "se crolla a zero -> freeze-a-convergenza sicuro; se resta
alto -> serve un termine esplicito nella loss") spetta a chi legge il log,
non a questo loop.

## ⛔ Scoperta scrivendo QUESTO file: un blocco MoE residuo senza normalizzazione
non e' stabile in profondita' — tolleranza algebrica resa RELATIVA

Incatenando 3+ finestre (8 strati giocattolo, W=3) il controllo algebrico
GIA' presente in `errore_lemma1_finestra` (`scarto < tol_algebrica`,
tol=1e-4 ASSOLUTA — verificato, mai un bug: e' li' apposta per scovare errori
di costruzione del grafo) ha iniziato a fallire alla terza finestra. Isolato
col debugger: la causa NON e' la quantizzazione (q_l/a_l), e' il TEACHER
stesso — `StratoMoEGiocattolo` e' `h_out = h + moe(h)` SENZA alcuna
normalizzazione (nessun RMSNorm, a differenza del vero `Qwen3_5MoeDecoderLayer`
che ce l'ha prima di ogni sotto-blocco), e con pesi random non calibrati per
la profondita' la scala di `h_l` (SOLO TEACHER, zero quantizzazione coinvolta)
e' misurata crescere da ~7 a ~787 in appena 3 strati non normalizzati in piu'
(misurato isolando `errore_lemma1_finestra` da questo loop, non stimato). E'
un fenomeno reale e generale delle reti residue non normalizzate (non un
artefatto giocattolo esotico), che il vero Thor NON ha (RMSNorm ad ogni
strato tiene la scala vicino a 1 per costruzione) — ma un blocco giocattolo
volutamente minimo (`finestra_globale.py`, "SwiGLU minimale... senza i
dettagli di packing") non ha motivo di replicarlo, e i test originali di
quel file restano apposta a 1-2 strati, dove il fenomeno non emerge.

**Cura, non scorciatoia**: la tolleranza ASSOLUTA (1e-4, calibrata per
scala~1) e' stata resa configurabile in `passo_ottimizzazione`
(`finestra_globale.py`, parametro nuovo, default INVARIATO — nessuna delle
sue 4 chiamate esistenti cambia comportamento). Questo loop, PRIMA di
allenare ciascuna finestra, misura la scala REALE attesa con un forward
usa-e-getta (`_scala_finestra`, sotto — no_grad, tolleranza disattivata) e
passa una tolleranza PROPORZIONALE a quella scala (stesso rapporto 1e-4 del
default, con un margine 10x per la crescita durante il training). Non e'
un allentamento arbitrario del controllo per farlo tacere: e' la stessa
tolleranza RELATIVA (non assoluta) che la letteratura numerica userebbe per
un controllo di uguaglianza floating-point quando la scala degli operandi
non e' fissa — il controllo resta capace di scovare un VERO bug (uno scarto
grande quanto la scala stessa), semplicemente non scatta piu' per il solo
crescere della rete.
"""
from __future__ import annotations

import argparse
import functools
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from finestra_globale import (  # noqa: E402 — riusate, mai duplicate
    GRUPPO_DOWN_REALE,
    GRUPPO_GATEUP_REALE,
    StratoMoEGiocattolo,
    _chiama_quant,
    _group_giocattolo,
    _perturba_z,
    congela_routing,
    crea_esperti_giocattolo,
    errore_lemma1_finestra,
    parametri_ternari,
    passo_ottimizzazione,
    verifica_gpu_libera,
)
from quant_ternaria_globale import TernarioDiff  # noqa: E402 — riusata, mai duplicata

# rapporto tol/scala del default di errore_lemma1_finestra (1e-4 a scala~1) —
# la tolleranza RELATIVA che questo loop applica alle finestre successive alla
# prima usa lo STESSO rapporto (vedi §docstring modulo, "Scoperta scrivendo
# QUESTO file"), con un margine 10x per la crescita di scala DURANTE
# l'ottimizzazione della finestra (misurata solo PRIMA di allenare).
_RAPPORTO_TOL_SCALA = 1e-4
_MARGINE_CRESCITA_TRAINING = 10.0


# ══════════════════════════ 1. pianificazione delle finestre (puro, no torch) ══════════════════════════
def stride_di(W: int, overlap: bool) -> int:
    """stride=W senza overlap (finestre consecutive, nessuna sovrapposizione);
    stride=ceil(W/2) con overlap (~50%, arrotondato per eccesso sui W dispari
    — §docstring modulo). Sempre >=1 (W=1 con overlap degenera a stride=1=W:
    un singolo strato non puo' sovrapporsi a se stesso, comportamento corretto
    e dichiarato, non un bug)."""
    if W < 1:
        raise ValueError(f"W deve essere >= 1, ricevuto {W}")
    return W if not overlap else max(1, math.ceil(W / 2))


def pianifica_finestre(n_strati_totale: int, W: int, overlap: bool) -> list[tuple[int, int]]:
    """Genera la sequenza di finestre (start, fine) semiaperte che copre
    [0, n_strati_totale) partendo dallo strato 0. Senza overlap: partizione
    ESATTA (nessun buco, nessuna duplicazione; l'ULTIMA finestra puo' essere
    piu' CORTA di W se n_strati_totale non e' multiplo di W — mai piu' lunga,
    mai overlappata: la coda si accorcia, non trabocca). Con overlap: le
    finestre avanzano di `stride_di(W, True)` e l'ultima e' agganciata
    esattamente a n_strati_totale (mai oltre)."""
    if W > n_strati_totale:
        raise ValueError(f"finestra W={W} piu' grande della rete ({n_strati_totale} strati)")
    stride = stride_di(W, overlap)
    finestre: list[tuple[int, int]] = []
    start = 0
    while True:
        fine = min(start + W, n_strati_totale)
        finestre.append((start, fine))
        if fine >= n_strati_totale:
            break
        start += stride
    _verifica_pianificazione(finestre, n_strati_totale, W, overlap)
    return finestre


def _verifica_pianificazione(finestre: list[tuple[int, int]], n_strati_totale: int,
                              W: int, overlap: bool) -> None:
    """Autocontrollo strutturale (SOLLEVA, mai un log muto — coerente con
    'un controllo che non puo' fallire non e' un controllo'): niente buchi,
    partenza a 0, arrivo esatto a n_strati_totale, nessuna finestra oltre W,
    e la copertura rispetta il regime dichiarato (esattamente 1x senza
    overlap; 1x o 2x, MAI 3x, con overlap — dimostrato in astratto nel
    docstring del modulo: 2*stride >= W per costruzione di stride_di)."""
    assert finestre[0][0] == 0, f"la prima finestra deve partire da 0, ottenuto {finestre[0]}"
    assert finestre[-1][1] == n_strati_totale, (
        f"l'ultima finestra deve arrivare a {n_strati_totale}, ottenuto {finestre[-1]}")
    for (s0, f0), (s1, f1) in zip(finestre, finestre[1:]):
        assert s1 > s0, f"gli inizi delle finestre devono crescere: {s0} -> {s1}"
        assert s1 <= f0, f"buco fra le finestre: {(s0, f0)} e {(s1, f1)}"
    for s, f in finestre:
        assert f - s <= W, f"finestra {(s, f)} piu' lunga di W={W}"
    conteggi = _conta_copertura(finestre, n_strati_totale)
    if overlap:
        assert all(1 <= c <= 2 for c in conteggi), f"copertura fuori da {{1,2}} con overlap: {conteggi}"
    else:
        assert all(c == 1 for c in conteggi), f"copertura non esattamente 1x senza overlap: {conteggi}"


def _conta_copertura(finestre: list[tuple[int, int]], n_strati_totale: int) -> list[int]:
    """Quante finestre includono ciascuno strato — il modo piu' diretto per
    verificare 'copre tutti gli strati esattamente una volta (o con
    l'overlap dichiarato)' richiesto dal compito."""
    conteggi = [0] * n_strati_totale
    for s, f in finestre:
        for l in range(s, f):
            conteggi[l] += 1
    return conteggi


# ══════════════════════════ 2. rete giocattolo COMPLETA (non solo una finestra) ══════════════════════════
def costruisci_rete_giocattolo(n_strati_totale: int, hidden: int = 8, intermedio: int = 6,
                                n_esperti: int = 4, top_k: int = 2, n_tok: int = 32,
                                seed: int = 0, device: str = "cpu",
                                fattore_stabilita_profondita: float = 0.4):
    """Come `finestra_globale.costruisci_finestra_giocattolo`, ma per l'INTERA
    rete (n_strati_totale strati) invece che per una finestra fissa — il loop
    esterno scorre finestre diverse sugli STESSI strati, quindi serve un'unica
    costruzione condivisa. Differenza dichiarata rispetto alla funzione
    gemella: qui il peso del gate (`gate_w`) di OGNI strato viene CONSERVATO
    (`gate_pesi_per_strato`), perche' il ri-instradamento periodico ne ha
    bisogno ad ogni bordo di finestra — la funzione gemella lo scarta subito
    dopo l'uso perche' per una finestra fissa basta un solo congelamento
    iniziale. Le primitive vere (creazione esperti, formula di routing,
    parametrizzazione ternaria, il wrapper MoE con skip-connection) sono
    IMPORTATE da `finestra_globale.py`, non riscritte.

    `fattore_stabilita_profondita` (nuovo qui, non esiste nella funzione
    gemella — quella resta a 1-2 strati, dove non serve): `StratoMoEGiocattolo`
    e' `h_out = h + moe(h)` SENZA normalizzazione (nessun RMSNorm, a
    differenza del vero decoder layer di Thor — vedi §docstring modulo,
    "Scoperta scrivendo QUESTO file"), e con `crea_esperti_giocattolo` alla
    sua scala originale (1/sqrt(fan_in), calibrata per un SINGOLO strato) la
    scala di h cresce ESPONENZIALMENTE con la profondita' (misurato: da ~7 a
    oltre 1e10 in 6-8 strati non normalizzati). Qui si compensa l'assenza di
    normalizzazione smorzando i pesi teacher di un fattore fisso PRIMA di
    costruire i `TernarioDiff` — misurato (24/9): a 0.4 la scala resta
    stabile (~3-3.5) anche su 24 strati consecutivi, contro l'esplosione a
    fattore=1.0. Non e' un modo di "far tornare i conti": e' la stessa
    funzione che un RMSNorm assente farebbe nel modello vero, resa esplicita
    invece che nascosta in una tolleranza allentata a dismisura.

    Il routing iniziale (uno per strato) e' calcolato sul flusso TEACHER
    (`h_rif = st_teacher(h_rif)`, come nella funzione gemella): e' l'unico
    punto di partenza sensato prima che qualunque ottimizzazione sia mai
    avvenuta (nessun flusso quantizzato esiste ancora). Dalla prima finestra
    del loop in poi, il routing si RICALCOLA sul flusso quantizzato corrente
    (vedi `ricongela_routing_finestra`), come da §PIANO.md 02:26.

    Ritorna (pesi_esperti_per_strato, gate_pesi_per_strato, strati_teacher, h0):
    pesi_esperti_per_strato[l] = lista di (TernarioDiff, TernarioDiff) per
    esperto (indicizzabile per slice di finestra, MUTA nel tempo: il loop la
    allena in-place); gate_pesi_per_strato[l] = Tensor (n_esperti, hidden)
    FISSO (il router non e' mai un bersaglio, per costruzione — stesso
    principio di `finestra_globale.py`); strati_teacher[l] = StratoMoEGiocattolo
    col routing INIZIALE gia' congelato (serve come valore di confronto per il
    PRIMO route-flip della finestra 0); h0 = ingresso della rete."""
    torch.manual_seed(seed)
    h0 = torch.randn(n_tok, hidden, device=device)
    pesi_esperti_per_strato: list[list] = []
    gate_pesi_per_strato: list[torch.Tensor] = []
    strati_teacher: list[StratoMoEGiocattolo] = []
    h_rif = h0
    for l in range(n_strati_totale):
        pesi_teacher = [(gu.to(device) * fattore_stabilita_profondita,
                         dw.to(device) * fattore_stabilita_profondita) for gu, dw in
                         crea_esperti_giocattolo(hidden, intermedio, n_esperti, seed=100 + l)]
        gate_w = (torch.randn(n_esperti, hidden, generator=torch.Generator().manual_seed(200 + l))
                  * 0.1).to(device)
        routing = congela_routing(h_rif, gate_w, top_k)
        # STESSO ripiego di finestra_globale.costruisci_finestra_giocattolo:
        # GRUPPO_GATEUP_REALE=128/GRUPPO_DOWN_REALE=256 non dividono mai le
        # dimensioni giocattolo (hidden/intermedio a una cifra) -> ng=1 per
        # riga (`_group_giocattolo` lo fa cadere su n_in). Import, non duplica.
        pesi_quant = [(TernarioDiff(gu, group=_group_giocattolo(gu.shape[1], GRUPPO_GATEUP_REALE)).to(device),
                       TernarioDiff(dw, group=_group_giocattolo(dw.shape[1], GRUPPO_DOWN_REALE)).to(device))
                      for gu, dw in pesi_teacher]
        st_teacher = StratoMoEGiocattolo(pesi_teacher, routing, ternario=False)
        strati_teacher.append(st_teacher)
        pesi_esperti_per_strato.append(pesi_quant)
        gate_pesi_per_strato.append(gate_w)
        with torch.no_grad():
            h_rif = st_teacher(h_rif)
    return pesi_esperti_per_strato, gate_pesi_per_strato, strati_teacher, h0


# ══════════════════════════ 3. route-flip fra due instradamenti dello stesso strato ══════════════════════════
def _token_a_esperti(routing: dict, n_tok: int) -> list[frozenset]:
    """Inverte {esperto: {token_idx, peso}} in n_tok insiemi {esperti che
    servono quel token}. Ogni insieme DEVE avere cardinalita' top_k (un
    token MoE e' sempre instradato a esattamente top_k esperti) — verificato
    da chi chiama (`misura_route_flip`), non assunto qui in silenzio."""
    per_token: list[set] = [set() for _ in range(n_tok)]
    for e, v in routing.items():
        for ti in v["token_idx"].tolist():
            per_token[ti].add(e)
    return [frozenset(s) for s in per_token]


def misura_route_flip(routing_vecchio: dict, routing_nuovo: dict, n_tok: int, top_k: int) -> dict:
    """Tasso di route-flip fra due instradamenti DELLO STESSO strato (stesso
    n_tok, stesso top_k) — stessa convenzione di `stato/route_flip.log`:
    uno 'slot' e' una delle top_k posizioni assegnate a un token; e' 'cambiato'
    se l'insieme di esperti di quel token e' diverso PRIMA/DOPO (la differenza
    simmetrica fra i due insiemi conta quanti slot sono usciti+entrati).
    Ritorna {'slot_totali', 'slot_cambiati', 'tasso', 'token_cambiati',
    'token_totali'} — tutti valori python semplici (si loggano/salvano,
    non entrano in un grafo).

    ⚠️ Nota tecnica su come si conta un flip: per un token la cui cardinalita'
    resta top_k su ENTRAMBI i lati (verificato sotto), la differenza
    simmetrica fra i due insiemi vale SEMPRE `2*(top_k - |intersezione|)` —
    un numero PARI (un esperto che esce e' sempre bilanciato da uno che
    entra, a cardinalita' fissa). Dividerla per 2 da' il numero VERO di
    sostituzioni per quel token (0..top_k), cosi' `slot_cambiati` resta
    confrontabile 1:1 con `slot_totali=n_tok*top_k` (tasso in [0,1] esatto:
    un turnover TOTALE, tutti gli esperti diversi per tutti i token, da'
    tasso=1.0, non 2.0 — verificato nel test 2 sotto, non solo assunto)."""
    vecchi = _token_a_esperti(routing_vecchio, n_tok)
    nuovi = _token_a_esperti(routing_nuovo, n_tok)
    for lista, nome in ((vecchi, "vecchio"), (nuovi, "nuovo")):
        for i, s in enumerate(lista):
            assert len(s) == top_k, (
                f"routing {nome}: token {i} assegnato a {len(s)} esperti, atteso "
                f"top_k={top_k} (un instradamento MoE valido ne assegna sempre esattamente top_k)")
    slot_cambiati = sum(len(v ^ n) // 2 for v, n in zip(vecchi, nuovi))
    token_cambiati = sum(1 for v, n in zip(vecchi, nuovi) if v != n)
    slot_totali = n_tok * top_k
    return {"slot_totali": slot_totali, "slot_cambiati": slot_cambiati,
            "tasso": slot_cambiati / slot_totali,
            "token_cambiati": token_cambiati, "token_totali": n_tok}


def ricongela_routing_finestra(pesi_quant_finestra: list, gate_pesi_finestra: list[torch.Tensor],
                                h_hat_iniziale: torch.Tensor, top_k: int, beta: float
                                ) -> tuple[list[dict], torch.Tensor]:
    """Ricalcola il routing per OGNI strato della finestra, propagando il
    flusso QUANTIZZATO CORRENTE in forma DURA (`hard=True`, i codici ternari
    reali — 'instrada sul flusso quantizzato corrente', non su un surrogato
    morbido ne' su una stima) a partire da `h_hat_iniziale` (per la finestra
    0 e' `h0` stesso: nessun errore ancora introdotto; per le successive e'
    `h0+e0` della finestra, fornito dal chiamante — vedi
    `ottimizza_rete_multi_finestra`). Ritorna (routing per strato della
    finestra, flusso quantizzato in uscita dall'ultimo strato — diagnostica,
    non serve al passo successivo: quello arriva da `errore_lemma1_finestra`
    dopo l'ottimizzazione, piu' preciso perche' calcolato con i codici
    APPENA aggiornati, non con quelli di ingresso alla finestra)."""
    routing_nuove: list[dict] = []
    h_hat = h_hat_iniziale
    for pesi_layer, gate_w in zip(pesi_quant_finestra, gate_pesi_finestra):
        routing = congela_routing(h_hat, gate_w, top_k)
        routing_nuove.append(routing)
        strato_tmp = StratoMoEGiocattolo(pesi_layer, routing, ternario=True)
        with torch.no_grad():
            h_hat = strato_tmp(h_hat, beta=beta, hard=True)
    return routing_nuove, h_hat


# ══════════════════════════ 3bis. tolleranza algebrica RELATIVA alla scala ══════════════════════════
def _scala_finestra(strati_teacher_fn: list, strati_quant_fn: list,
                     h0_win: torch.Tensor, e0_win: Optional[torch.Tensor]) -> float:
    """Misura preventiva (no_grad, `tol_algebrica=inf` — non deve MAI sollevare,
    e' solo diagnostica) della scala massima di h_l/e_l attesa in QUESTA
    finestra coi codici CORRENTI. Vedi §docstring modulo ("Scoperta scrivendo
    QUESTO file"): un blocco MoE residuo senza normalizzazione puo' far
    crescere la scala di ordini di grandezza gia' dopo poche finestre."""
    with torch.no_grad():
        ris = errore_lemma1_finestra(strati_teacher_fn, strati_quant_fn, h0_win, e0_win,
                                      tol_algebrica=float("inf"))
    scala = max(x.abs().max().item() for x in ris["h_l"] + ris["e_l"])
    return max(1.0, scala)


def _tol_algebrica_di(scala: float) -> float:
    """tol = rapporto_default * scala * margine — vedi le due costanti di
    modulo e la discussione nel docstring. A scala~1 (le finestre di
    `finestra_globale.py`, 1-2 strati) da' ~1e-3, comunque piu' larga del
    default 1e-4 di quel file: qui si usa SEMPRE questa versione relativa
    (mai il default nudo), perche' il loop non sa a priori se la rete che gli
    viene passata restera' a scala~1 o crescera' con la profondita'."""
    return _RAPPORTO_TOL_SCALA * scala * _MARGINE_CRESCITA_TRAINING


# ══════════════════════════ 4. il loop ESTERNO multi-finestra ══════════════════════════
@dataclass
class ConfigLoop:
    W: int = 2   # vincolo REALE misurato il 24/9 03:33 (VRAM Thor vero, filone concorrente
    # stesso PIANO.md): W=4/W=3 vanno in OOM, W=2 e' l'unica taglia verificata capiente.
    # Rimane un PARAMETRO — qui solo il default onesto, non un tetto imposto dal codice.
    overlap: bool = True
    n_passi_per_finestra: int = 20
    lr: float = 0.05
    beta: float = 16.0
    top_k: int = 2
    perturba_prima_finestra: bool = True   # stesso motivo di finestra_globale._perturba_z:
    # z parte GIA' vicino all'ottimo locale (inizializzato da W), senza disturbarlo
    # l'AdamW non avrebbe margine — ma SOLO alla finestra 0: le successive ripartono
    # da z GIA' parzialmente allenato dalle finestre precedenti (overlap), disturbarlo
    # di nuovo vanificherebbe quel lavoro invece di simulare un punto di partenza onesto.


def ottimizza_rete_multi_finestra(pesi_esperti_per_strato: list, gate_pesi_per_strato: list,
                                   strati_teacher: list, h0: torch.Tensor, cfg: ConfigLoop,
                                   verbose: bool = True) -> dict:
    """IL LOOP ESTERNO. Per ogni finestra pianificata: (1) ricongela il
    routing di tutti i suoi strati sul flusso quantizzato corrente, loggando
    il route-flip vs l'instradamento precedente di ciascuno; (2) ottimizza
    la finestra (AdamW su `parametri_ternari`, kernel Lemma-1 riusato da
    `finestra_globale.py`); (3) legge dal risultato (un ultimo forward
    no_grad coi codici APPENA aggiornati) il seme esatto (h0,e0) per la
    finestra successiva, all'offset preciso di avanzamento.

    Ritorna un diario completo: {'finestre', 'stride', 'route_flip' (lista
    di dict, uno per (finestra, strato)), 'loss_per_finestra' (lista di
    liste), 'copertura' (conteggio per strato)} — pensato per essere
    ispezionato dai test e, in futuro, salvato su disco per la corsa vera."""
    n_strati_totale = len(strati_teacher)
    finestre = pianifica_finestre(n_strati_totale, cfg.W, cfg.overlap)
    stride = stride_di(cfg.W, cfg.overlap)
    n_tok = h0.shape[0]

    if verbose:
        modo = f"CON overlap (stride={stride})" if cfg.overlap else f"SENZA overlap (stride={stride})"
        print(f"loop multi-finestra: {n_strati_totale} strati, W={cfg.W}, {modo}, "
              f"{len(finestre)} finestre: {finestre}")

    # routing corrente per strato ASSOLUTO — inizializzato dalla costruzione
    # (calcolato sul flusso teacher, l'unico disponibile prima di ogni training)
    routing_attuale: dict[int, dict] = {l: strati_teacher[l].routing_fisso for l in range(n_strati_totale)}

    diario: dict = {"finestre": finestre, "stride": stride, "route_flip": [],
                     "loss_per_finestra": [], "copertura": _conta_copertura(finestre, n_strati_totale)}

    h0_seme: Optional[torch.Tensor] = None
    e0_seme: Optional[torch.Tensor] = None

    for idx_finestra, (start, fine) in enumerate(finestre):
        W_reale = fine - start
        if verbose:
            print(f"\n── finestra {idx_finestra}: strati [{start},{fine}) (W_reale={W_reale}) ──")

        pesi_quant_finestra = pesi_esperti_per_strato[start:fine]
        gate_finestra = gate_pesi_per_strato[start:fine]

        # (1) ri-instradamento sul flusso quantizzato corrente + log route-flip
        h_hat_iniziale = h0 if idx_finestra == 0 else (h0_seme + e0_seme)
        routing_nuove, _ = ricongela_routing_finestra(
            pesi_quant_finestra, gate_finestra, h_hat_iniziale, cfg.top_k, cfg.beta)
        for i, L in enumerate(range(start, fine)):
            flip = misura_route_flip(routing_attuale[L], routing_nuove[i], n_tok, cfg.top_k)
            diario["route_flip"].append({"finestra": idx_finestra, "strato": L, **flip})
            if verbose:
                print(f"    strato {L}: route-flip {flip['slot_cambiati']}/{flip['slot_totali']} "
                      f"slot ({flip['tasso']:.1%})")
            routing_attuale[L] = routing_nuove[i]

        strati_quant_obj = [StratoMoEGiocattolo(pesi_quant_finestra[i], routing_nuove[i], ternario=True)
                             for i in range(W_reale)]
        strati_quant_fn = [functools.partial(_chiama_quant, strato=o, beta=cfg.beta) for o in strati_quant_obj]
        strati_teacher_fn = strati_teacher[start:fine]

        # (2) e0/h0 della finestra + ottimizzazione
        h0_win = h0 if idx_finestra == 0 else h0_seme
        e0_win = None if idx_finestra == 0 else e0_seme   # None => zeros: valido SOLO alla finestra 0

        if cfg.perturba_prima_finestra and idx_finestra == 0:
            _perturba_z(pesi_quant_finestra, scala=0.6, seed=10)

        # tolleranza algebrica RELATIVA alla scala di QUESTA finestra (vedi
        # §docstring modulo) — misurata PRIMA di ottimizzare, coi codici come
        # stanno adesso (che per gli strati di overlap sono gia' parzialmente
        # allenati dalla finestra precedente, non quelli di costruzione).
        scala = _scala_finestra(strati_teacher_fn, strati_quant_fn, h0_win, e0_win)
        tol = _tol_algebrica_di(scala)
        if verbose:
            print(f"    scala h/e attesa in questa finestra: {scala:.3e}  ->  "
                  f"tolleranza algebrica usata: {tol:.3e}")

        params: list[torch.nn.Parameter] = []
        for pq in pesi_quant_finestra:
            params += parametri_ternari(pq)
        opt = torch.optim.AdamW(params, lr=cfg.lr)
        perdite: list[float] = []
        for _ in range(cfg.n_passi_per_finestra):
            ris = passo_ottimizzazione(strati_teacher_fn, strati_quant_fn, h0_win, opt, e0_win,
                                        tol_algebrica=tol)
            perdite.append(ris["loss"].item())
        diario["loss_per_finestra"].append(perdite)
        if verbose and perdite:
            print(f"    ottimizzazione ({cfg.n_passi_per_finestra} passi AdamW): "
                  f"loss {perdite[0]:.4e} -> {perdite[-1]:.4e}")

        # (3) seme ESATTO per la finestra successiva — un ultimo forward
        # no_grad coi codici APPENA aggiornati (mai i valori del passo di
        # training precedente: quelli sono calcolati PRIMA dell'ultimo
        # opt.step(), sarebbero un seme di un passo indietro rispetto ai
        # parametri veri).
        if idx_finestra + 1 < len(finestre):
            offset = finestre[idx_finestra + 1][0] - start
            assert 0 < offset <= W_reale, f"offset di avanzamento {offset} fuori da (0,{W_reale}]"
            with torch.no_grad():
                # tolleranza generosa qui: e' una lettura DIAGNOSTICA dei
                # codici appena aggiornati, non serve a scovare un bug — quello
                # e' gia' stato verificato passo per passo dentro il training
                # sopra, con la tolleranza calibrata sulla scala pre-training.
                ris_finale = errore_lemma1_finestra(strati_teacher_fn, strati_quant_fn, h0_win, e0_win,
                                                     tol_algebrica=max(tol, _tol_algebrica_di(scala) * 10))
            h0_seme = ris_finale["h_l"][offset].detach()
            e0_seme = ris_finale["e_l"][offset].detach()

    return diario


# ══════════════════════════ TEST 1: pianificazione (puro, nessun torch pesante) ══════════════════════════
def _test_pianificazione() -> bool:
    print("\n── test 1: pianifica_finestre — copertura esattamente 1x (no overlap) o 1x/2x (overlap) ──")
    ok = True
    casi = [(8, 3), (8, 4), (9, 4), (12, 4), (7, 3), (5, 5), (10, 3)]
    for n, W in casi:
        for overlap in (False, True):
            finestre = pianifica_finestre(n, W, overlap)
            conteggi = _conta_copertura(finestre, n)
            attesa = "==1 ovunque" if not overlap else "in {1,2} ovunque"
            rispetta = (all(c == 1 for c in conteggi) if not overlap
                        else all(1 <= c <= 2 for c in conteggi))
            copre_tutto = all(c >= 1 for c in conteggi)
            arriva_a_n = finestre[-1][1] == n
            parte_da_0 = finestre[0][0] == 0
            stride = stride_di(W, overlap)
            esito = rispetta and copre_tutto and arriva_a_n and parte_da_0
            print(f"    n={n:2d} W={W} overlap={overlap!s:5} stride={stride}: "
                  f"{len(finestre)} finestre {finestre} copertura={conteggi} "
                  f"({'✅' if esito else '⛔ atteso ' + attesa})")
            ok = ok and esito
    # caso letterale richiesto dal compito: 8 strati, W=3
    f_no = pianifica_finestre(8, 3, overlap=False)
    f_si = pianifica_finestre(8, 3, overlap=True)
    print(f"    caso richiesto (8 strati, W=3): senza overlap={f_no} (stride={stride_di(3, False)}), "
          f"con overlap={f_si} (stride={stride_di(3, True)})")
    # W piu' grande della rete: deve sollevare, non fingere una finestra
    sollevato = False
    try:
        pianifica_finestre(3, 4, overlap=False)
    except ValueError:
        sollevato = True
    print(f"    W > n_strati_totale solleva ValueError: {'✅' if sollevato else '⛔'}")
    ok = ok and sollevato
    print(f"    {'✅ TEST 1 PASSA' if ok else '⛔ TEST 1 FALLISCE'}")
    return ok


# ══════════════════════════ TEST 2: route-flip — metrica isolata ══════════════════════════
def _test_route_flip_metrica() -> bool:
    print("\n── test 2: misura_route_flip — identita'=0, cambio totale=100% ──")
    n_tok, top_k = 4, 2
    routing_a = {0: {"token_idx": torch.tensor([0, 1]), "peso": torch.ones(2)},
                 1: {"token_idx": torch.tensor([2, 3]), "peso": torch.ones(2)},
                 2: {"token_idx": torch.tensor([0, 1, 2, 3]), "peso": torch.ones(4)}}
    # ogni token: {0,1}->esperti{0,2}; {2,3}->esperti{1,2}  (top_k=2 rispettato)
    flip_identico = misura_route_flip(routing_a, routing_a, n_tok, top_k)
    ok1 = flip_identico["slot_cambiati"] == 0 and flip_identico["tasso"] == 0.0
    print(f"    routing identico a se stesso: slot_cambiati={flip_identico['slot_cambiati']} "
          f"tasso={flip_identico['tasso']:.3f}  ({'✅' if ok1 else '⛔'})")

    routing_b = {3: {"token_idx": torch.tensor([0, 1]), "peso": torch.ones(2)},
                 4: {"token_idx": torch.tensor([2, 3]), "peso": torch.ones(2)},
                 5: {"token_idx": torch.tensor([0, 1, 2, 3]), "peso": torch.ones(4)}}
    # stessa struttura ma esperti TUTTI diversi (3,4,5 invece di 0,1,2): ogni
    # token sostituisce ENTRAMBI i suoi 2 esperti (0 in comune) -> turnover
    # TOTALE, tasso=1.0 esatto (non 2.0: vedi la nota tecnica nel docstring
    # di misura_route_flip sulla divisione per 2 della differenza simmetrica)
    flip_tot = misura_route_flip(routing_a, routing_b, n_tok, top_k)
    ok2 = flip_tot["slot_cambiati"] == n_tok * top_k and flip_tot["tasso"] == 1.0
    print(f"    routing completamente diverso: slot_cambiati={flip_tot['slot_cambiati']}/"
          f"{flip_tot['slot_totali']} tasso={flip_tot['tasso']:.3f}  ({'✅' if ok2 else '⛔'})")

    # cardinalita' sbagliata (bug di costruzione, non un caso reale) deve sollevare
    routing_rotto = {0: {"token_idx": torch.tensor([0]), "peso": torch.ones(1)}}  # token 1,2,3 mai instradati
    sollevato = False
    try:
        misura_route_flip(routing_a, routing_rotto, n_tok, top_k)
    except AssertionError:
        sollevato = True
    print(f"    routing con cardinalita' != top_k solleva AssertionError: {'✅' if sollevato else '⛔'}")

    ok = ok1 and ok2 and sollevato
    print(f"    {'✅ TEST 2 PASSA' if ok else '⛔ TEST 2 FALLISCE'}")
    return ok


# ══════════════════════════ TEST 3: il loop su rete giocattolo, SENZA overlap ══════════════════════════
def _test_loop_senza_overlap(device: str = "cpu") -> bool:
    print("\n── test 3: loop su 8 strati giocattolo, W=3, SENZA overlap ──")
    n_strati, W = 8, 3
    pesi, gate_pesi, teacher, h0 = costruisci_rete_giocattolo(
        n_strati_totale=n_strati, n_tok=16, seed=0, device=device)
    cfg = ConfigLoop(W=W, overlap=False, n_passi_per_finestra=3, top_k=2)
    diario = ottimizza_rete_multi_finestra(pesi, gate_pesi, teacher, h0, cfg, verbose=True)

    finestre_attese = pianifica_finestre(n_strati, W, overlap=False)
    avanza_ok = diario["finestre"] == finestre_attese
    print(f"    finestre generate == pianificate: {avanza_ok} ({diario['finestre']})")

    copertura_ok = all(c == 1 for c in diario["copertura"])
    print(f"    copertura ESATTAMENTE 1x per ogni strato: {diario['copertura']} "
          f"({'✅' if copertura_ok else '⛔'})")

    # senza overlap: un log di route-flip per OGNI strato, una volta sola —
    # nessuno strato ricompare in due finestre diverse.
    strati_loggati = [r["strato"] for r in diario["route_flip"]]
    log_ok = (sorted(strati_loggati) == list(range(n_strati))
              and len(strati_loggati) == len(set(strati_loggati)))
    print(f"    route-flip loggato per ogni strato ESATTAMENTE una volta: "
          f"{len(diario['route_flip'])} log per {n_strati} strati ({'✅' if log_ok else '⛔'})")

    tassi_validi = all(0.0 <= r["tasso"] <= 1.0 for r in diario["route_flip"])
    print(f"    tutti i tassi di route-flip in [0,1]: {'✅' if tassi_validi else '⛔'}")
    for r in diario["route_flip"]:
        print(f"      finestra={r['finestra']} strato={r['strato']}: "
              f"{r['slot_cambiati']}/{r['slot_totali']} slot ({r['tasso']:.1%})")

    loss_finite = all(math.isfinite(v) for perdite in diario["loss_per_finestra"] for v in perdite)
    print(f"    tutte le loss finite (nessun NaN/Inf): {'✅' if loss_finite else '⛔'}")

    ok = avanza_ok and copertura_ok and log_ok and tassi_validi and loss_finite
    print(f"    {'✅ TEST 3 PASSA' if ok else '⛔ TEST 3 FALLISCE'}")
    return ok


# ══════════════════════════ TEST 4: il loop su rete giocattolo, CON overlap ══════════════════════════
def _test_loop_con_overlap(device: str = "cpu") -> bool:
    print("\n── test 4: loop su 8 strati giocattolo, W=3, CON overlap ──")
    n_strati, W = 8, 3
    pesi, gate_pesi, teacher, h0 = costruisci_rete_giocattolo(
        n_strati_totale=n_strati, n_tok=16, seed=1, device=device)
    cfg = ConfigLoop(W=W, overlap=True, n_passi_per_finestra=3, top_k=2)
    diario = ottimizza_rete_multi_finestra(pesi, gate_pesi, teacher, h0, cfg, verbose=True)

    finestre_attese = pianifica_finestre(n_strati, W, overlap=True)
    avanza_ok = diario["finestre"] == finestre_attese
    print(f"    finestre generate == pianificate: {avanza_ok} ({diario['finestre']})")

    copertura_ok = all(1 <= c <= 2 for c in diario["copertura"]) and any(c == 2 for c in diario["copertura"])
    print(f"    copertura in {{1,2}} (overlap DICHIARATO, almeno uno strato doppio): "
          f"{diario['copertura']} ({'✅' if copertura_ok else '⛔'})")

    # con overlap: il numero di log di route-flip == somma delle lunghezze
    # delle finestre (uno per (finestra,strato), NON uno per strato assoluto:
    # uno strato di overlap viene loggato in DUE finestre diverse).
    attesi = sum(f - s for s, f in diario["finestre"])
    log_ok = len(diario["route_flip"]) == attesi
    print(f"    log di route-flip == somma lunghezze finestre: {len(diario['route_flip'])} == "
          f"{attesi} ({'✅' if log_ok else '⛔'})")

    # ogni strato che compare in DUE finestre deve avere DUE righe di log
    # (una per finestra) — verifica diretta della continuita' della catena.
    from collections import Counter
    conteggio_log = Counter(r["strato"] for r in diario["route_flip"])
    catena_ok = all(conteggio_log[l] == diario["copertura"][l] for l in range(n_strati))
    print(f"    ogni strato loggato tante volte quante finestre lo contengono: "
          f"{'✅' if catena_ok else '⛔'} ({dict(conteggio_log)})")

    tassi_validi = all(0.0 <= r["tasso"] <= 1.0 for r in diario["route_flip"])
    for r in diario["route_flip"]:
        print(f"      finestra={r['finestra']} strato={r['strato']}: "
              f"{r['slot_cambiati']}/{r['slot_totali']} slot ({r['tasso']:.1%})")

    loss_finite = all(math.isfinite(v) for perdite in diario["loss_per_finestra"] for v in perdite)
    print(f"    tutte le loss finite (nessun NaN/Inf): {'✅' if loss_finite else '⛔'}")

    ok = avanza_ok and copertura_ok and log_ok and catena_ok and tassi_validi and loss_finite
    print(f"    {'✅ TEST 4 PASSA' if ok else '⛔ TEST 4 FALLISCE'}")
    return ok


def _test(device: str = "cpu") -> bool:
    print(f"ottimizzazione_globale_loop.py — loop esterno multi-finestra, device={device}")
    r1 = _test_pianificazione()
    r2 = _test_route_flip_metrica()
    r3 = _test_loop_senza_overlap(device)
    r4 = _test_loop_con_overlap(device)
    ok = r1 and r2 and r3 and r4
    print(f"\n{'✅ TUTTI I TEST PASSANO' if ok else '⛔ ALMENO UN TEST FALLISCE'} "
          f"(pianificazione={r1}, route_flip_metrica={r2}, loop_senza_overlap={r3}, loop_con_overlap={r4})")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                     help="cuda = ROCm-as-cuda su venv-catq; strati GIOCATTOLO minuscoli, "
                          "mai il modello Thor. Verifica rocm-smi PRIMA di procedere.")
    args = ap.parse_args()
    if args.device == "cuda":
        if not torch.cuda.is_available():
            print("⛔ --device cuda richiesto ma torch non vede una GPU (serve venv-catq ROCm). Uso cpu.")
            args.device = "cpu"
        elif not verifica_gpu_libera():
            print("⛔ GPU non verificata libera — uso cpu per sicurezza (mandato: mai GPU alla cieca).")
            args.device = "cpu"
    esito = _test(device=args.device)
    if args.device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        print(f"VRAM allocata dopo pulizia: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
    raise SystemExit(0 if esito else 1)
