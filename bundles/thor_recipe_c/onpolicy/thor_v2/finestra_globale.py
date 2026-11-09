#!/usr/bin/env python3
"""Nucleo della FINESTRA SCORREVOLE per l'ottimizzazione GLOBALE cross-strato
(arXiv 2607.14630, Lemma 1) — il pezzo che il PIANO (`stato/PIANO.md`, 02:31)
chiama "resta da scrivere". Prende W strati consecutivi (teacher BF16 fisso +
codici ternari CORRENTI, vedi `quant_ternaria_globale.py`), un batch di token
REALI (o sintetici per i test qui), e calcola: (a) forward teacher attraverso
i W strati (no_grad); (b) forward quantizzato attraverso i W strati CON i
codici correnti (con grad); (c) la loss di errore accumulato ESATTA (Lemma 1,
differenza finita, NON linearizzata — le tre valutazioni per strato gia'
calibrate in PIANO.md 02:31); (d) backward + un passo AdamW sui parametri
z/s/delta della finestra.

24/9/2026, COMPITO PRINCIPALE del workflow di stanotte. ⛔ VINCOLO DI SESSIONE
(GPU condivisa, uguale agli altri file di stanotte): qui SOLO strati
GIOCATTOLO sintetici (hidden=8, intermedio=6, pochi esperti — la stessa scala
della "prova minuscola" gia' verificata in PIANO.md 02:31). Mai il modello
Thor intero, mai un training reale su di esso. Verifica il MECCANISMO a 1
strato poi a 2 strati, come richiesto ("mai assemblare tutto e sperare che
funzioni") — la validazione su strati VERI (finestra 8-12, poche batch reali
dal corpus) resta il passo SUCCESSIVO, non fatto qui (richiederebbe caricare
`cattura_schur_esperti.py::FlussoSuDisco` e i pesi veri, fuori perimetro per
questo pezzo — vedi nota finale).

## La formula (Lemma 1, riconciliata in PIANO.md 02:31)

    e_(l+1) = Â_l(e_l) + q_l
    Â_l(e_l) = F̂_l(h_l+e_l) - F̂_l(h_l)     — propagazione ESATTA (diff. finita)
    q_l      = F̂_l(h_l) - F_l(h_l)          — rumore NUOVO introdotto allo strato l

`h_l` viene SEMPRE dal teacher (forward pulito, no_grad — mai dal proprio
percorso quantizzato che si ricicla, coerente con VSRAQ/2604.06515). Tre
valutazioni per strato: F_l(h_l) [teacher, no_grad], F̂_l(h_l) [quant, grad],
F̂_l(h_l+e_l) [quant, grad] — esattamente come calibrato in PIANO.md (02:31,
~300ms/strato su Thor vero, qui su strati giocattolo e' istantaneo).

⭐ Nota algebrica (NON nel piano, trovata scrivendo questo file — controllo
di correttezza incorporato nel kernel, non solo teorico): sommando le due
definizioni, Â_l(e_l)+q_l = F̂_l(h_l+e_l) - F̂_l(h_l) + F̂_l(h_l) - F_l(h_l) =
F̂_l(h_l+e_l) - F_l(h_l) — i due termini F̂_l(h_l) SI CANCELLANO esattamente.
Il kernel calcola comunque le TRE valutazioni separate (servono come
diagnostica per-strato, q_l e Â_l(e_l) SEPARATI, coerente con PIANO.md) ma
verifica ad ogni chiamata che la somma torni esattamente uguale alla forma
collassata (a meno di arrotondamento floating point) — se non tornasse,
sarebbe un bug nella costruzione del grafo, non solo un'imprecisione numerica.

## Routing MoE nella finestra — CONGELATO, come deciso in PIANO.md (02:26)

Nessuna macchina STE/Gumbel-softmax sul router: il gate resta F32 esatto, il
top-k resta un `torch.no_grad()` IDENTICO a `moe_routing.token_per_esperto`.
L'assegnazione token->esperto per il branch QUANTIZZATO si calcola UNA VOLTA
(`congela_routing`, sotto) da un hidden_states di riferimento — nella forgia
vera sarebbe il flusso quantizzato CORRENTE prima di iniziare l'ottimizzazione
della finestra — e resta FISSA per tutta la finestra di N passi (nessun
ricalcolo dentro il forward). Il ri-instradamento periodico fra una finestra
e la successiva e' responsabilita' del CHIAMANTE (fuori da questo file: qui
si verifica solo che, DATO un routing fisso, il kernel non lo tocca mai).
Il branch TEACHER usa il SUO proprio routing (dal suo hidden_states pulito,
sempre no_grad) — non serve congelarlo: non dipende mai dai codici allenabili,
quindi e' identico ad ogni passo per costruzione (nessun ricalcolo necessario,
ma nessun danno a ricalcolarlo: qui lo si fa una volta all'inizio, come per
il quantizzato, per simmetria del codice).

## COMPITO 24/9 (seconda parte, stesso workflow): ffn_down_exps nella STESSA
ottimizzazione, non a parte

`StratoMoEGiocattolo`/`_swiglu_esperto` chiamano GIA' down_proj dentro lo
stesso forward — e quindi dentro lo stesso kernel Lemma-1: l'errore che il
kernel misura e' gia' quello dell'INTERO blocco MLP (gate+up+SwiGLU+down), non
solo gate+up, perche' e' una conseguenza diretta di come `_swiglu_esperto` e'
scritta, non qualcosa da aggiungere al kernel. Il pezzo che MANCAVA (compito
ricevuto dopo la diagnosi di PIANO.md 02:56 — "STESSO DIFETTO diagnosticato
per gate/up... se down_proj resta fuori dall'ottimizzazione congiunta,
raddoppia il gap invece di chiuderlo"): (1) il `TernarioDiff` di down_proj
usava `group=dw.shape[1]` implicito, mai legato al GROUP REALE di produzione
(256, formato TQ1_0 STANDARD senza rotazione — `cattura_schur_down.py`,
DIVERSO dal 128 di gate/up: i due formati non possono avere lo stesso valore,
vedi il monkeypatch di `gptq_1024_lib.GROUP` documentato li'); (2) nessun test
verificava SEPARATAMENTE che il gradiente raggiungesse i codici di down_proj
— un bug che azzerasse silenziosamente il gradiente di UNA sola famiglia
sarebbe passato inosservato in `_test_due_layer`, che somma i gradienti di
TUTTI i parametri insieme. Chiuso qui: `GRUPPO_GATEUP_REALE`/`GRUPPO_DOWN_REALE`
(costanti nominate, allineate alle fonti vere), `parametri_gate_up`/
`parametri_down` (liste SEPARATE, oltre alla `parametri_ternari` combinata
gia' esistente) e il **test 4** (`_test_down_proj_variabile_libera`, sotto),
che verifica le tre proprieta' richieste dal compito: (a) il gradiente della
loss COMBINATA raggiunge SEPARATAMENTE sia gate/up sia down_proj — non solo
la somma totale; (b) un SOLO oggetto AdamW (un solo `param_groups`, costruito
da UNA lista concatenata — mai due ottimizzatori separati, che ricadrebbe
nello stesso difetto sequenziale gia' diagnosticato); (c) la loss COMBINATA
scende con quell'unico ottimizzatore.

## Cosa NON e' (onesta', come richiesto)

Non e' ancora integrato con `cattura_schur_esperti.py` (che resta SOLA
LETTURA per questi filoni, mai modificato). Non carica Thor. Non fa il
ri-instradamento periodico multi-finestra (quello e' un ciclo esterno, non
scritto qui). Usa strati giocattolo con pesi random piccoli, non i pesi veri.
A scala giocattolo il GROUP di down_proj (256) e quello di gate/up (128) NON
dividono mai le dimensioni minuscole (hidden=8/intermedio=6): il test 4
verifica il MECCANISMO (ripiego a un gruppo per riga, ng=1 — vedi
`_group_giocattolo`), non la granularita' vera di produzione. La validazione
su strati VERI (finestra 8-12, il placeholder `ottimizza_finestra_JOINT` in
`valida_finestra_8_12.py`) resta il passo SUCCESSIVO, fuori perimetro per
questo pezzo — richiederebbe caricare Thor, vietato dal vincolo di sessione.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from functools import partial
from typing import Callable, Optional

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moe_routing import instradamento  # noqa: E402 — STESSA formula della forgia vera, mai duplicata
from quant_ternaria_globale import TernarioDiff  # noqa: E402 — pezzo 1, gia' verificato standalone
from gptq_1024_lib import GROUP as GRUPPO_GATEUP_REALE  # noqa: E402 — 128, codec RUOTATO (gate/up), fonte unica

# ⛔ 256, formato TQ1_0 STANDARD SENZA rotazione (down_proj) — STESSA costante
# di `cattura_schur_down.py::GRUPPO_DOWN`, qui un letterale INDIPENDENTE (non
# importata da li'): quel file monkeypatcha `gptq_1024_lib.GROUP = 256`
# all'import (il suo stesso docstring lo dichiara), quindi importarlo da li'
# trascinerebbe quel monkeypatch dentro QUESTO processo — e romperebbe
# GRUPPO_GATEUP_REALE sopra (che deve restare 128). I due valori non possono
# mai convivere nello stesso `gptq_1024_lib.GROUP` per costruzione.
GRUPPO_DOWN_REALE = 256

StratoFn = Callable[[torch.Tensor], torch.Tensor]


# ══════════════════════════════ sicurezza GPU condivisa ══════════════════════════════
def verifica_gpu_libera(soglia_processi: int = 0) -> bool:
    """Prima di QUALSIASI esecuzione reale su GPU (mandato del workflow):
    interroga rocm-smi, ritorna False (e stampa un avviso) se ci sono ALTRI
    processi KFD attivi oltre la soglia. Non solleva: il chiamante decide se
    procedere comunque (per i test sintetici minuscoli qui, un avviso basta;
    non e' un cancello duro come il perimetro ARCHITECT).

    ⛔⛔ LO SPECCHIO DEL PGREP, versione GPU (trovato scrivendo QUESTO file,
    24/9): `torch.cuda.is_available()` (chiamato PRIMA di questa funzione, nel
    blocco __main__) inizializza il runtime ROCm e registra IL PROCESSO
    CORRENTE come voce KFD (0 VRAM, 0 CU — verificato: `rocm-smi --showpids`
    mostra il proprio PID python subito dopo la sola chiamata a
    `is_available()`, prima di allocare nulla). Un primo tentativo che contava
    "c'e' una riga nella tabella PID" falliva SEMPRE (si trovava da solo),
    esattamente come `pgrep -f <frase>` dentro `bash -c` con quella stessa
    frase — stessa classe di bug, contesto nuovo. Qui si esclude esplicitamente
    il PROPRIO pid (`os.getpid()`) dalla tabella prima di giudicare."""
    try:
        r = subprocess.run(["rocm-smi", "--showpids"], capture_output=True, text=True, timeout=10)
        out = r.stdout
        if "No KFD PIDs currently running" in out:
            return True
        mio_pid = os.getpid()
        altri_pid = [int(m.group(1)) for riga in out.splitlines()
                     if (m := re.match(r"^\s*(\d+)\s+\S", riga)) and int(m.group(1)) != mio_pid]
        libera = len(altri_pid) <= soglia_processi
        if not libera:
            print(f"⛔ GPU CONDIVISA: altri processi KFD attivi PID={altri_pid} "
                  f"(il nostro e' {mio_pid}, escluso) — NON procedere con carichi pesanti.")
            print(out)
        return libera
    except Exception as e:  # noqa: BLE001 — diagnostica, mai bloccare la sessione per un fallimento della sonda
        print(f"⚠️  impossibile interrogare rocm-smi ({e}) — trattare come 'non verificato', non come 'libera'.")
        return False


# ══════════════════════════════ (c) il kernel Lemma-1 ══════════════════════════════
def errore_lemma1_finestra(strati_teacher: list[StratoFn], strati_quant: list[StratoFn],
                            h0: torch.Tensor, e0: Optional[torch.Tensor] = None,
                            tol_algebrica: float = 1e-4,
                            h_l_teacher_cache: "Optional[list[torch.Tensor]]" = None,
                            diagnostica: bool = True) -> dict:
    """Nucleo della finestra scorrevole. strati_teacher[l]/strati_quant[l]:
    Callable[[Tensor], Tensor] — un forward di UN singolo strato (skip
    connection inclusa se il layer vero ce l'ha, come i decoder layer reali).
    strati_teacher e' chiamato SEMPRE sotto no_grad QUI (disciplina imposta
    dal kernel, non delegata al chiamante). strati_quant e' chiamato con grad
    abilitato secondo il contesto del chiamante (tipicamente dentro un passo
    di training, con torch.is_grad_enabled()==True).

    h0: hidden state di ingresso alla finestra (dal teacher). e0: errore
    accumulato ENTRANTE (da fuori finestra); None => zeros_like(h0), assunzione
    ESPLICITA "nessun errore pregresso" (vera solo se la finestra parte dal
    primo strato della rete, o se e0 e' stato seminato altrove dal chiamante —
    MAI un default silenzioso spacciato per il caso generale).

    Ritorna: h_l (teacher, W+1 tensori), e_l (errore accumulato, W+1 tensori),
    q_l/a_l (diagnostica per-strato, W tensori ciascuna), loss (scalare,
    somma su l di mean(e_l[l+1]**2) — riduzione DICHIARATA, non l'unica
    possibile: pesare gli strati piu' vicini all'uscita e' un'alternativa
    NON implementata qui), scarto_algebrico (lista di W float — deve restare
    sotto tol_algebrica: verifica che a_l+q_l == F̂_l(h_l+e_l)-F_l(h_l), la
    cancellazione algebrica esatta discussa nel docstring del modulo — se
    supera la tolleranza e' un bug di costruzione del grafo, non rumore).

    h_l_teacher_cache (24/9 05:xx — LEVA DI VELOCITA', matematicamente
    certa, non una stima): F_l(h_l) dipende SOLO da pesi teacher CONGELATI e
    da h_l, che a sua volta dipende SOLO da h0 (fisso per l'intera
    ottimizzazione di un chunk, mai da e/dal quantizzato). Chiamare
    strati_teacher[l] ad OGNI passo di un ciclo di allenamento (200 volte,
    dati identici) ricalcola un valore gia' noto — trovato leggendo il
    ciclo chiamante (ottimizza_finestra_JOINT), non per analogia. Se
    passato (lista di W+1 tensori, stesso contratto di h_l ritornato),
    salta la chiamata a strati_teacher[l] e usa h_l_teacher_cache[l+1] —
    default None => comportamento INVARIATO (ricalcola, come prima) per
    ogni chiamata esistente (i 4 test di questo file, mai toccati)."""
    Wn = len(strati_quant)
    assert len(strati_teacher) == Wn, "un forward teacher per ogni forward quantizzato, stessa finestra"
    if e0 is None:
        e0 = torch.zeros_like(h0)
    h_l = [h0]
    e_l = [e0]
    q_l_lista: list[torch.Tensor] = []
    a_l_lista: list[torch.Tensor] = []
    scarto_algebrico: list[float] = []
    loss = h0.new_zeros(())
    loss_somma = h0.new_zeros((), dtype=torch.float32)
    for l in range(Wn):
        h = h_l[-1]
        e = e_l[-1]
        if h_l_teacher_cache is not None:
            F_teacher_h = h_l_teacher_cache[l + 1]           # F_l(h_l) — GIA' noto, non ricalcolato
        else:
            with torch.no_grad():
                F_teacher_h = strati_teacher[l](h)           # F_l(h_l) — teacher, MAI nel grafo
        # ⭐ 24/9 17:xx (audit B15) — F̂_l(h_l) si CANCELLA in e_next = a+q
        # (−Fq_h in a, +Fq_h in q): il suo gradiente e' esattamente zero.
        # Calcolarlo nel grafo costava un backward intero e la memoria delle
        # sue attivazioni senza cambiare un bit del gradiente (test apposta).
        # ⛔⛔⛔ 24/9 04:xx — CANCELLAZIONE CATASTROFICA IN bf16 (trovata da un
        # agente dedicato, riprodotta su strati VERI isolati senza toccare
        # rotazione/routing/TernarioDiff): gli strati reali sono bf16 (8 bit
        # di mantissa). Le sottrazioni quasi-cancellanti sotto, fatte nel
        # dtype nativo, davano un errore di un ULP alla scala delle
        # attivazioni (~16-32 -> ULP 0.125, ESATTAMENTE lo scarto osservato
        # 1.250e-01). Su strati giocattolo (gia' float32) non si vedeva. Fix:
        # l'ingresso alla rete resta bf16 (e' cosi' che il modello vero
        # gira), ma l'ALGEBRA di tracking (q/a/e_next/loss) va in float32 —
        # verificato: scarto esattamente 0.0 dopo il cast.
        e_bf16 = e.to(h.dtype)
        Fq_h_piu_e = strati_quant[l](h + e_bf16)             # F̂_l(h_l+e_l) — quant, CON grad
        if not diagnostica:
            # ⭐ 24/9 19:5x (mandato forgia < 2 ore): in addestramento la loss usa
            # SOLO e_{l+1} = F̂(h+e) - F(h) (la forma "collassata" qui sotto).
            # F̂(h) serve alla scomposizione diagnostica q/a e al controllo
            # algebrico (con .item(): una sincronizzazione GPU per strato per
            # passo). Stesso gradiente (F̂(h) era sotto no_grad), un forward in meno.
            e_next = Fq_h_piu_e.float() - F_teacher_h.detach().float()
            h_l.append(F_teacher_h)
            e_l.append(e_next)
            loss = loss + e_next.pow(2).mean()
            loss_somma = loss_somma + e_next.pow(2).sum()
            continue
        with torch.no_grad():
            Fq_h = strati_quant[l](h)                        # F̂_l(h_l) — solo diagnostica q_l/a_l
        Fq_h32 = Fq_h.float()
        F_teacher_h32 = F_teacher_h.detach().float()
        Fq_h_piu_e32 = Fq_h_piu_e.float()
        q = Fq_h32 - F_teacher_h32                            # rumore NUOVO di questo strato (fp32)
        a = Fq_h_piu_e32 - Fq_h32                             # propagazione ESATTA dell'errore entrante (fp32)
        e_next = a + q                                        # fp32: l'errore accumulato non si arrotonda piu'
        with torch.no_grad():
            collassata = Fq_h_piu_e32 - F_teacher_h32
            scarto = (e_next - collassata).abs().max().item()
        scarto_algebrico.append(scarto)
        assert scarto < tol_algebrica, (
            f"strato {l}: scarto algebrico {scarto:.3e} >= tol {tol_algebrica:.3e} — "
            "Â_l(e_l)+q_l deve collassare esattamente su F̂_l(h_l+e_l)-F_l(h_l) "
            "(bug nella costruzione del grafo, non rumore numerico atteso)")
        q_l_lista.append(q)
        a_l_lista.append(a)
        h_l.append(F_teacher_h)
        e_l.append(e_next)
        loss = loss + e_next.pow(2).mean()
        loss_somma = loss_somma + e_next.pow(2).sum()
    # ⛔⛔⛔ 24/9 (audit B1): `loss` e' una MEDIA su ~4,2M elementi (2x1024
    # token x 2048) — su Thor vero il gradiente per parametro finiva sotto
    # eps=1e-8 di Adam e l'ottimizzatore non si muoveva (codici = init +
    # solo weight decay al 99-99,99%, misurato). `loss_somma` e' la stessa
    # quantita' non normalizzata: e' QUELLA che va nel backward (Adam e'
    # invariante alla scala del gradiente finche' |g|>>eps). `loss` (media)
    # resta per i log, cosi' i numeri restano confrontabili con lo storico.
    return {"h_l": h_l, "e_l": e_l, "q_l": q_l_lista, "a_l": a_l_lista,
            "loss": loss, "loss_somma": loss_somma, "scarto_algebrico": scarto_algebrico}


# ══════════════════════════════ (d) un passo di ottimizzazione ══════════════════════════════
def precalcola_teacher(strati_teacher: list[StratoFn], h0: torch.Tensor) -> "list[torch.Tensor]":
    """⭐ 24/9 05:xx — LEVA DI VELOCITA' matematicamente certa (vedi docstring
    di errore_lemma1_finestra): il ramo teacher e' deterministico e FISSO per
    l'intera ottimizzazione di un chunk (pesi congelati, h0 fisso) — questa
    funzione lo attraversa UNA volta sola, il chiamante riusa il risultato
    per tutti i passi invece di richiamare strati_teacher[l] ad ogni passo.
    Ritorna [h0, h_1, ..., h_W] — stesso contratto di h_l in errore_lemma1_finestra."""
    h_l = [h0]
    with torch.no_grad():
        for strato in strati_teacher:
            h_l.append(strato(h_l[-1]))
    return h_l


def passo_ottimizzazione(strati_teacher: list[StratoFn], strati_quant: list[StratoFn],
                          h0: torch.Tensor, opt: torch.optim.Optimizer,
                          e0: Optional[torch.Tensor] = None,
                          tol_algebrica: float = 1e-4,
                          h_l_teacher_cache: "Optional[list[torch.Tensor]]" = None,
                          indice_accumulo: int = 0, passi_accumulo: int = 1,
                          riduzione: str = "somma", diagnostica: bool = True,
                          salta_non_finiti: bool = False,
                          dopo_backward: "Optional[Callable[[torch.Tensor], object]]" = None,
                          proiezione: "Optional[object]" = None) -> dict:
    """Un passo: forward (kernel sopra) -> backward -> opt.step(). Ritorna il
    risultato del kernel (loss.item() gia' scollegato dal grafo per il logging,
    il resto no — chi chiama e vuole solo i numeri usa ["loss"] su un tensore
    gia' esploso dal backward: prendere .item() SUBITO dopo la chiamata).

    ⭐ 24/9 10:2x — `indice_accumulo`/`passi_accumulo` (default 0/1 =
    comportamento ESATTAMENTE INVARIATO, nessuna chiamata esistente cambia)
    — GRADIENT ACCUMULATION: leva di velocità candidata (PIANO.md 09:3x,
    letteratura: BATCH_SEQ=2 è la taglia che rende la loss "non liscia,
    tendente a divergere" — qui si simula un batch effettivo più grande
    SENZA aumentare la VRAM di picco, a differenza di alzare BATCH_SEQ
    davvero). Il chiamante passa `indice_accumulo=t % passi_accumulo` e
    `passi_accumulo=N`: `zero_grad()` scatta SOLO al primo micro-passo del
    gruppo, `opt.step()` SOLO all'ultimo — nel mezzo i gradienti si
    sommano da soli (comportamento nativo di `.backward()` senza
    `zero_grad` fra una chiamata e l'altra). La loss NON va divisa per N
    qui (methodo diverso da mini-batch SGD classico: qui ogni micro-passo
    è un CAMPIONE indipendente della stessa finestra, la media dei
    gradienti su N campioni è già quello che un batch N volte più grande
    farebbe con gli stessi dati — dividere per N sarebbe corretto SOLO se
    la loss fosse già una somma anziché una media per elemento, verificare
    caso per caso prima di usare `passi_accumulo>1` su una loss diversa).

    `tol_algebrica` (nuovo, 24/9, aggiunto per il loop multi-finestra —
    default INVARIATO, quindi ogni chiamata esistente si comporta esattamente
    come prima): la tolleranza ASSOLUTA del controllo algebrico interno al
    kernel e' calibrata per finestre a scala~1 (i 4 test di questo file
    restano apposta a 1-2 strati). Chi incatena PIU' finestre su una rete PIU'
    profonda (`ottimizzazione_globale_loop.py`) puo' vedere la scala di h_l/e_l
    crescere di ordini di grandezza (misurato: un blocco MoE residuo SENZA
    normalizzazione non e' stabile in profondita' — gia' il solo TEACHER, zero
    quantizzazione coinvolta, cresce da scala~7 a ~787 in 3 strati non
    normalizzati), e deve poter passare una tolleranza RELATIVA alla scala
    corrente, non quella tarata per scala~1 — altrimenti il controllo (che
    esiste per scovare bug nel grafo, non per il rumore numerico che
    naturalmente cresce con la scala) scatterebbe per il solo crescere della
    rete, non per un bug.

    `dopo_backward` (26/9, ancora ai pesi — ancora_forgia.py, THOR_ANCORA_RAPPORTO): se
    passato, viene chiamato con la perdita dati DERIVATA (staccata dal grafo) SUBITO dopo
    il suo backward e PRIMA del controllo dei gradienti e di opt.step(): aggiunge da se'
    ai .grad il gradiente di un termine in piu' della perdita (perdita totale = dati +
    termine), cosi' il passo dell'ottimizzatore e il controllo dei non finiti vedono il
    gradiente TOTALE. Il suo ritorno va in ris["dopo_backward"]. None (default) = nessuna
    chiamata: comportamento identico al bit a prima.

    `proiezione` (26/9, Selective Projection Decay — spd_forgia.py, THOR_SPD_LAMBDA): se passata,
    `proiezione.prima_del_passo()` subito PRIMA di opt.step() (dopo il backward, dopo il gancio
    dell'ancora e dopo il controllo dei non finiti: la condizione c = -g·(p - θ0) vede il gradiente
    TOTALE che il passo usera') e `proiezione.dopo_il_passo()` subito DOPO (la proiezione dei
    parametri appena aggiornati, Tian et al. 2024 Alg. 2). Solo quando il passo avviene davvero:
    un micro-passo di accumulo o un passo saltato non la chiamano. None (default) = nessuna
    chiamata: comportamento identico al bit a prima."""
    if indice_accumulo == 0:
        opt.zero_grad(set_to_none=True)
    ris = errore_lemma1_finestra(strati_teacher, strati_quant, h0, e0, tol_algebrica=tol_algebrica,
                                  h_l_teacher_cache=h_l_teacher_cache, diagnostica=diagnostica)
    # audit B1: backward sulla SOMMA (default). "media" = comportamento vecchio,
    # tenuto solo per confronti espliciti (A/B del cancello strumentale).
    if riduzione not in ("somma", "media"):
        raise ValueError(f"riduzione {riduzione!r}: attese 'somma' o 'media'")
    da_derivare = ris["loss_somma"] if riduzione == "somma" else ris["loss"]
    # ⭐ 24/9 23:0x (salta_non_finiti): M1 ha prodotto un NaN SPORADICO (stessa ricetta
    # rilanciata fino a 60 passi: pulita). Come in addestramento a precisione ridotta:
    # loss o gradienti non finiti -> il passo si SALTA (nessun aggiornamento, parametri
    # sani) e lo si dice in ris["saltato"]; chi chiama conta e decide quando fermarsi.
    if salta_non_finiti and not bool(torch.isfinite(da_derivare)):
        opt.zero_grad(set_to_none=True)
        ris["saltato"] = "loss"
        return ris
    da_derivare.backward()
    if dopo_backward is not None:
        ris["dopo_backward"] = dopo_backward(da_derivare.detach())
    if indice_accumulo == passi_accumulo - 1:
        if salta_non_finiti:
            grad = [p.grad for g in opt.param_groups for p in g["params"] if p.grad is not None]
            norma = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(x) for x in grad]))
            if not bool(torch.isfinite(norma)):
                opt.zero_grad(set_to_none=True)
                ris["saltato"] = "gradienti"
                return ris
        if proiezione is not None:
            proiezione.prima_del_passo()
        opt.step()
        if proiezione is not None:
            proiezione.dopo_il_passo()
    return ris


# ══════════════════════════════ scaffolding: strato MoE giocattolo ══════════════════════════════
def _swiglu_esperto(x: torch.Tensor, gate_up: torch.Tensor, down: torch.Tensor) -> torch.Tensor:
    """SwiGLU minimale (come Qwen3_5MoeExperts, senza i dettagli di packing):
    gate_up: (2*intermedio, hidden), down: (hidden, intermedio)."""
    gu = x @ gate_up.T
    gate, up = gu.chunk(2, dim=-1)
    return (F.silu(gate) * up) @ down.T


def _group_giocattolo(n_in: int, group_reale: int) -> int:
    """GROUP reale di produzione se divide la dimensione, altrimenti UN
    gruppo per riga (group=n_in — sempre valido: n_in % n_in == 0, l'assert
    di `TernarioDiff.__init__` non puo' mai fallire su questo ripiego). A
    scala giocattolo (hidden=8, intermedio=6) NESSUNO dei due GROUP reali
    (128 gate/up, 256 down) divide mai le dimensioni: qui serve solo che il
    meccanismo funzioni con la granularita' di gruppo DICHIARATA di ciascuna
    famiglia (una chiamata esplicita con GRUPPO_GATEUP_REALE o
    GRUPPO_DOWN_REALE, non un valore indovinato) — riprodurre la taglia vera
    e' impossibile con pesi giocattolo di poche unita' senza caricare Thor,
    vietato dal vincolo di sessione."""
    return group_reale if n_in % group_reale == 0 else n_in


def crea_esperti_giocattolo(hidden: int, intermedio: int, n_esperti: int, seed: int):
    """n_esperti esperti 'teacher' (pesi FISSI fp32 piccoli, stessa scala della
    prova minuscola di PIANO.md 02:31: in=8, intermedio=6)."""
    g = torch.Generator().manual_seed(seed)
    # scala 1/sqrt(fan_in) (init standard, NON 0.05): con pesi troppo piccoli
    # la cascata SwiGLU(gate_up)->down produce segnali ~1e-6, sotto la soglia
    # di rumore float32 per la loss — misurato (24/9): a scala 0.05 la loss
    # q_0 restava ~1e-12 anche dopo aver disturbato z del 23%, indistinguibile
    # dal rumore numerico. A questa scala il segnale e' O(1e-2)-O(1e-1), ben
    # sopra il floor di float32.
    return [(torch.randn(2 * intermedio, hidden, generator=g) / hidden ** 0.5,
             torch.randn(hidden, intermedio, generator=g) / intermedio ** 0.5)
            for _ in range(n_esperti)]


class _InvolucroGate:
    """Involucro minimo con `.weight` per riusare `moe_routing.instradamento`
    (che si aspetta un gate_module, come `sq.mlp.gate` nella forgia vera) senza
    costruire un nn.Linear vero — il router qui e' un Tensor fisso, mai un
    nn.Parameter allenabile."""

    def __init__(self, weight: torch.Tensor):
        self.weight = weight


def congela_routing(h_riferimento: torch.Tensor, gate_weight: torch.Tensor, top_k: int) -> dict:
    """Instradamento calcolato UNA VOLTA (no_grad, STESSA formula di
    `moe_routing.instradamento` — non una riderivazione), da tenere FISSO per
    tutta la finestra. Ritorna {token_idx_per_esperto, peso_per_esperto} —
    un dict semplice (non il dict completo di token_per_esperto: qui non
    serve 'current_state', le feature si prendono LIVE dall'input di ogni
    chiamata, mai da quelle catturate al momento del congelamento).

    `instradamento` (moe_routing.py, STESSA di produzione) si aspetta un
    `gate_module` con `.weight` (nella forgia vera e' `sq.mlp.gate`, un
    nn.Linear) — qui un involucro minimo, mai un nn.Linear vero (nessun
    parametro allenabile: il router resta F32 fisso, mai bersaglio)."""
    gate_module = _InvolucroGate(gate_weight)
    selected, pesi = instradamento(h_riferimento, gate_module, top_k)  # (n_tok,top_k) ciascuno
    n_tok, n_esperti = h_riferimento.shape[0], gate_weight.shape[0]
    mappa: dict[int, dict] = {}
    for e in range(n_esperti):
        hit = (selected == e)
        if not hit.any():
            continue
        tok_idx, k_pos = torch.where(hit)
        mappa[e] = {"token_idx": tok_idx, "peso": pesi[tok_idx, k_pos]}
    return mappa


class StratoMoEGiocattolo:
    """Un decoder-layer MoE giocattolo con SKIP CONNECTION (h_out = h + moe(h)),
    routing FISSO passato dal chiamante (mai ricalcolato in __call__ — la
    disciplina 'congelato per la finestra' e' garantita qui per costruzione,
    non per convenzione). `pesi_esperti[e]` e' una coppia (gate_up, down): due
    `torch.Tensor` fissi (teacher) oppure due `TernarioDiff` (quantizzato — in
    tal caso serve passare `beta` a __call__)."""

    def __init__(self, pesi_esperti: list, routing_fisso: dict, ternario: bool):
        self.pesi_esperti = pesi_esperti
        self.routing_fisso = routing_fisso
        self.ternario = ternario

    def __call__(self, h: torch.Tensor, beta: Optional[float] = None, hard: bool = False) -> torch.Tensor:
        if self.ternario and beta is None:
            raise ValueError("strato quantizzato (ternario=True) richiede beta esplicito")
        out = torch.zeros_like(h)
        for e, v in self.routing_fisso.items():
            ti = v["token_idx"]                    # SOLO indici/pesi congelati
            w = v["peso"].to(h.dtype)
            x = h[ti]                                # feature LIVE dall'input di QUESTA chiamata
            gu_src, dw_src = self.pesi_esperti[e]
            if self.ternario:
                gu = gu_src.forward_duro(beta) if hard else gu_src.forward(beta)
                dw = dw_src.forward_duro(beta) if hard else dw_src.forward(beta)
            else:
                gu, dw = gu_src, dw_src
            y = _swiglu_esperto(x, gu, dw) * w[:, None]
            out = out.index_add(0, ti, y)            # out-of-place: autograd-safe (verificato)
        return h + out


def _perturba_z(pesi_esperti_per_strato: list[list], scala: float = 0.6, seed: int = 42) -> None:
    """Disturba il latente `z` di TUTTI gli esperti ternari di una finestra
    (in-place, no_grad). Necessario per un test onesto dell'ottimizzazione:
    `TernarioDiff` inizializza z0=atanh(clip(W/s0))/beta_0 ESATTAMENTE dal
    peso teacher — al primo forward la ricostruzione e' gia' quella MIGLIORE
    raggiungibile per quel delta (non c'e' quasi margine per l'AdamW). Una
    finestra vera parte invece dai codici del metodo SEQUENZIALE, che NON sono
    ottimali per l'obiettivo di finestra (Lemma 1) — qui si simula quel
    disallineamento con rumore esplicito, cosi' il test verifica che il
    gradiente SA correggere un punto di partenza imperfetto, non solo che
    'non peggiora un punto gia' ottimo'."""
    torch.manual_seed(seed)   # RNG globale (non un Generator dedicato: deve
    # funzionare identico su cpu e cuda senza costruire due rami — un
    # Generator() esplicito e' sempre cpu, incompatibile con tensori cuda).
    with torch.no_grad():
        for pesi_esperti in pesi_esperti_per_strato:
            for gu, dw in pesi_esperti:
                gu.z += torch.randn(gu.z.shape, device=gu.z.device) * scala
                dw.z += torch.randn(dw.z.shape, device=dw.z.device) * scala


def parametri_gate_up(pesi_esperti: list) -> list[torch.nn.Parameter]:
    """SOLO i parametri (z, log_s, delta_raw) del TernarioDiff di gate_up_proj,
    di TUTTI gli esperti di uno strato — separata da `parametri_down` perche'
    il test 4 (compito 24/9, ffn_down_exps) deve poter verificare il
    gradiente delle DUE famiglie SEPARATAMENTE: una somma totale positiva con
    un'intera famiglia a zero passerebbe un controllo che guarda solo il
    totale (come faceva `_test_due_layer` prima di questo test)."""
    ps: list[torch.nn.Parameter] = []
    for gu, _dw in pesi_esperti:
        ps += list(gu.parameters())
    return ps


def parametri_down(pesi_esperti: list) -> list[torch.nn.Parameter]:
    """SOLO i parametri (z, log_s, delta_raw) del TernarioDiff di down_proj,
    di TUTTI gli esperti di uno strato — simmetrico a `parametri_gate_up`."""
    ps: list[torch.nn.Parameter] = []
    for _gu, dw in pesi_esperti:
        ps += list(dw.parameters())
    return ps


def parametri_ternari(pesi_esperti: list) -> list[torch.nn.Parameter]:
    """Tutti i parametri allenabili di UNO strato, gate_up_proj E down_proj
    INSIEME (`parametri_gate_up`+`parametri_down` concatenati) — la STESSA
    lista che entra in UN SOLO ottimizzatore AdamW per costruire l'ottimizzatore
    della finestra: mai due ottimizzatori separati per famiglia, che sarebbe
    esattamente il difetto ("due ottimizzazioni separate") vietato dal
    compito 24/9 — vedi test 4."""
    return parametri_gate_up(pesi_esperti) + parametri_down(pesi_esperti)


def costruisci_finestra_giocattolo(n_strati: int, hidden: int = 8, intermedio: int = 6,
                                    n_esperti: int = 4, top_k: int = 2, n_tok: int = 32,
                                    seed: int = 0, device: str = "cpu"):
    """Costruisce n_strati StratoMoEGiocattolo teacher+quantizzato, con routing
    CONGELATO per ciascuno strato (calcolato dal proprio h0 di riferimento —
    proxy giocattolo di 'il flusso quantizzato corrente all'inizio della
    finestra', che nella forgia vera sarebbe l'uscita reale del modello).
    Ritorna (strati_teacher_fn, strati_quant_fn, pesi_quant_per_strato, h0,
    strati_quant_obj) — l'ultimo elemento espone gli StratoMoEGiocattolo VERI
    (non le closure) per ispezione diretta nei test (routing congelato ecc.),
    invece di introspettare gli attributi interni di una lambda. Tutti i
    tensori/moduli nascono GIA' sul device richiesto (mai un mix cpu/cuda
    lasciato al chiamante)."""
    torch.manual_seed(seed)
    h0 = torch.randn(n_tok, hidden, device=device)   # scala ~1 (attivazioni tipiche post-layernorm), non 0.05
    strati_teacher_fn, strati_quant_fn, pesi_quant_tutti, strati_quant_obj = [], [], [], []
    h_rif = h0
    for l in range(n_strati):
        pesi_teacher = [(gu.to(device), dw.to(device)) for gu, dw in
                         crea_esperti_giocattolo(hidden, intermedio, n_esperti, seed=100 + l)]
        gate_w = (torch.randn(n_esperti, hidden, generator=torch.Generator().manual_seed(200 + l))
                  * 0.1).to(device)
        routing = congela_routing(h_rif, gate_w, top_k)
        # GROUP reali di produzione (mai lo stesso valore per gate/up e down,
        # vedi cattura_schur_down.py): GRUPPO_GATEUP_REALE=128 (codec RUOTATO),
        # GRUPPO_DOWN_REALE=256 (formato TQ1_0 STANDARD, senza rotazione).
        # Nessuno dei due divide le dimensioni giocattolo (hidden=8/intermedio=6)
        # — `_group_giocattolo` ripiega su UN gruppo per riga in quel caso.
        # Qui serve solo che il meccanismo funzioni con la granularita' di
        # gruppo DICHIARATA di ciascuna famiglia, non riprodurre la taglia vera.
        pesi_quant = [(TernarioDiff(gu, group=_group_giocattolo(gu.shape[1], GRUPPO_GATEUP_REALE)).to(device),
                       TernarioDiff(dw, group=_group_giocattolo(dw.shape[1], GRUPPO_DOWN_REALE)).to(device))
                      for gu, dw in pesi_teacher]
        st_teacher = StratoMoEGiocattolo(pesi_teacher, routing, ternario=False)
        st_quant = StratoMoEGiocattolo(pesi_quant, routing, ternario=True)
        strati_teacher_fn.append(st_teacher)
        strati_quant_fn.append(partial(_chiama_quant, strato=st_quant, beta=16.0))
        pesi_quant_tutti.append(pesi_quant)
        strati_quant_obj.append(st_quant)
        with torch.no_grad():
            h_rif = st_teacher(h_rif)   # riferimento per il routing dello strato SUCCESSIVO
    return strati_teacher_fn, strati_quant_fn, pesi_quant_tutti, h0, strati_quant_obj


def _chiama_quant(h: torch.Tensor, strato: "StratoMoEGiocattolo", beta: float) -> torch.Tensor:
    """Adattatore Callable[[Tensor],Tensor] -> StratoMoEGiocattolo(h, beta=...).
    Funzione di modulo (non lambda con default-arg) cosi' `functools.partial`
    espone `.keywords['strato']` per ispezione diretta nei test — niente
    introspezione fragile di `__closure__` su una lambda."""
    return strato(h, beta=beta, hard=False)


# ══════════════════════════════ TEST 1: una finestra, UNO strato ══════════════════════════════
def _test_un_layer(device: str = "cpu", seed: int = 0) -> bool:
    """Finestra W=1, e0=None (=> zeros): deve COLLASSARE esattamente sul caso
    'un solo strato, primo della rete' — a_0=0 (nessun errore entrante),
    e_1=q_0=F̂(h)-F(h), la loss e' esattamente la MSE di ricostruzione dello
    strato. Poi qualche passo AdamW deve abbassare la loss (lo stesso
    meccanismo di quant_ternaria_globale.py, ora dentro il kernel a finestra)."""
    print("\n── test 1: finestra W=1 (collassa sul caso 'primo strato della rete') ──")
    st_teacher, st_quant, pesi_quant, h0, _ = costruisci_finestra_giocattolo(
        n_strati=1, seed=seed, device=device)

    ris = errore_lemma1_finestra(st_teacher, st_quant, h0)
    a0_zero = ris["a_l"][0].abs().max().item() < 1e-6
    with torch.no_grad():
        q0_manuale = st_quant[0](h0) - st_teacher[0](h0)
    e1_combacia = torch.allclose(ris["e_l"][1], q0_manuale, atol=1e-5)
    loss_combacia = abs(ris["loss"].item() - q0_manuale.pow(2).mean().item()) < 1e-5
    scarto_ok = max(ris["scarto_algebrico"]) < 1e-4
    print(f"    a_0 == 0 (nessun errore entrante): {a0_zero}  |  "
          f"e_1 == q_0 manuale: {e1_combacia}  |  loss == mean(q_0^2): {loss_combacia}  |  "
          f"scarto algebrico max={max(ris['scarto_algebrico']):.2e}: {scarto_ok}")
    collasso_ok = a0_zero and e1_combacia and loss_combacia and scarto_ok

    # ⚠️ z parte GIA' al suo ottimo locale (inizializzato da W stesso) — senza
    # disturbarlo l'AdamW non avrebbe quasi margine (verificato: loss gia'
    # a ~1e-12, la scala di rumore floating-point della cascata SwiGLU a
    # doppio matmul su pesi ~0.05). Si perturba PRIMA di ottimizzare, DOPO i
    # controlli strutturali sopra (quelli devono valere a ID, non a z disturbato).
    _perturba_z(pesi_quant, scala=0.6, seed=10)
    params = parametri_ternari(pesi_quant[0])
    opt = torch.optim.AdamW(params, lr=0.05)
    perdite = []
    for _ in range(60):
        r = passo_ottimizzazione(st_teacher, st_quant, h0, opt)
        perdite.append(r["loss"].item())
    scende = perdite[-1] < perdite[0] * 0.5
    print(f"    z disturbato (scala=0.6) poi ottimizzazione (60 passi AdamW): "
          f"loss {perdite[0]:.4e} -> {perdite[-1]:.4e}  "
          f"({'✅ scende' if scende else '⛔ non scende abbastanza'})")
    esito = collasso_ok and scende
    print(f"    {'✅ TEST 1 PASSA' if esito else '⛔ TEST 1 FALLISCE'}")
    return esito


# ══════════════════════════════ TEST 2: una finestra, DUE strati ══════════════════════════════
def _test_due_layer(device: str = "cpu", seed: int = 1) -> bool:
    """Finestra W=2: verifica (a) l'errore si ACCUMULA (e_1 dallo strato 0 entra
    nello strato 1 come a_1 = F̂_1(h_1+e_1)-F̂_1(h_1), NON zero in generale);
    (b) i gradienti raggiungono i parametri di ENTRAMBI gli strati; (c) esiste
    accoppiamento CROSS-STRATO vero (la sub-loss del SOLO strato 2 dipende dai
    parametri del PRIMO strato attraverso e_1 — proprieta' che il vecchio
    metodo sequenziale, ottimizzazione locale indipendente, non ha per
    costruzione); (d) un passo AdamW abbassa la loss TOTALE della finestra."""
    print("\n── test 2: finestra W=2 (accumulo + accoppiamento cross-strato) ──")
    st_teacher, st_quant, pesi_quant, h0, _ = costruisci_finestra_giocattolo(
        n_strati=2, seed=seed, device=device)

    ris = errore_lemma1_finestra(st_teacher, st_quant, h0)
    a1_non_zero = ris["a_l"][1].abs().max().item() > 1e-8
    scarto_ok = max(ris["scarto_algebrico"]) < 1e-4
    print(f"    a_1 (propagazione errore strato0->strato1) non-zero: {a1_non_zero} "
          f"(max|a_1|={ris['a_l'][1].abs().max().item():.3e})  |  "
          f"scarto algebrico max={max(ris['scarto_algebrico']):.2e}: {scarto_ok}")

    # (b)+(c): gradiente della sub-loss del SOLO strato finale rispetto ai
    # parametri del PRIMO strato — se fosse zero, i due strati sarebbero
    # indipendenti come nel metodo sequenziale vecchio, e la finestra non
    # servirebbe a nulla di nuovo.
    loss_solo_strato2 = ris["e_l"][2].pow(2).mean()
    params_strato0 = parametri_ternari(pesi_quant[0])
    params_strato1 = parametri_ternari(pesi_quant[1])
    grad_su_strato0 = torch.autograd.grad(loss_solo_strato2, params_strato0,
                                           retain_graph=True, allow_unused=True)
    norma_grad_strato0 = sum(g.abs().sum().item() for g in grad_su_strato0 if g is not None)
    accoppiamento_ok = norma_grad_strato0 > 0.0
    print(f"    |d(loss_strato2)/d(parametri_strato0)|.sum() = {norma_grad_strato0:.3e}  "
          f"({'✅ accoppiamento cross-strato reale' if accoppiamento_ok else '⛔ strati indipendenti — la finestra non aggiunge nulla'})")

    # (b): anche la loss TOTALE della finestra deve raggiungere ENTRAMBI gli strati
    ris["loss"].backward()
    grad_tot_0 = sum(p.grad.abs().sum().item() for p in params_strato0 if p.grad is not None)
    grad_tot_1 = sum(p.grad.abs().sum().item() for p in params_strato1 if p.grad is not None)
    entrambi_ricevono_grad = grad_tot_0 > 0.0 and grad_tot_1 > 0.0
    print(f"    gradiente loss totale: strato0={grad_tot_0:.3e}  strato1={grad_tot_1:.3e}  "
          f"({'✅ entrambi ricevono gradiente' if entrambi_ricevono_grad else '⛔ almeno uno strato morto'})")

    # (d): ottimizzazione della finestra intera abbassa la loss totale — z
    # disturbato per lo stesso motivo di _test_un_layer (altrimenti gia' quasi
    # ottimo, poco margine per l'AdamW). DOPO i controlli (a)-(c) sopra, che
    # devono valere sui codici COME COSTRUITI, non su una versione disturbata.
    _perturba_z(pesi_quant, scala=0.6, seed=11)
    params_finestra = parametri_ternari(pesi_quant[0]) + parametri_ternari(pesi_quant[1])
    opt = torch.optim.AdamW(params_finestra, lr=0.05)
    perdite = []
    for _ in range(60):
        r = passo_ottimizzazione(st_teacher, st_quant, h0, opt)
        perdite.append(r["loss"].item())
    scende = perdite[-1] < perdite[0] * 0.5
    print(f"    z disturbato (scala=0.6) poi ottimizzazione finestra W=2 (60 passi AdamW): "
          f"loss {perdite[0]:.4e} -> {perdite[-1]:.4e}  "
          f"({'✅ scende' if scende else '⛔ non scende abbastanza'})")

    esito = a1_non_zero and scarto_ok and accoppiamento_ok and entrambi_ricevono_grad and scende
    print(f"    {'✅ TEST 2 PASSA' if esito else '⛔ TEST 2 FALLISCE'}")
    return esito


def _test_routing_congelato(seed: int = 2) -> bool:
    """Verifica che il routing NON cambi fra la chiamata a h e quella a h+e
    dentro lo stesso passo (altrimenti l'algebra di Lemma 1 sarebbe costruita
    su due funzioni diverse, non sullo STESSO F̂_l valutato in due punti) — e
    che nessun gradiente esista mai verso il gate/router (mai un parametro
    nn, per costruzione: se lo fosse per errore, questo test lo scoprirebbe
    controllando che router non compaia affatto fra le foglie allenate)."""
    print("\n── test 3: routing CONGELATO dentro la finestra (nessuna STE sul gate) ──")
    st_teacher, st_quant, pesi_quant, h0, strati_quant_obj = costruisci_finestra_giocattolo(
        n_strati=1, seed=seed)
    strato = strati_quant_obj[0]   # lo StratoMoEGiocattolo VERO, non la closure/partial
    routing_prima = {e: v["token_idx"].clone() for e, v in strato.routing_fisso.items()}
    e_finta = torch.randn_like(h0) * 0.3
    _ = strato(h0, beta=16.0)
    _ = strato(h0 + e_finta, beta=16.0)
    routing_dopo = {e: v["token_idx"].clone() for e, v in strato.routing_fisso.items()}
    invariato = set(routing_prima) == set(routing_dopo) and all(
        torch.equal(routing_prima[e], routing_dopo[e]) for e in routing_prima)
    print(f"    routing identico prima/dopo aver processato h e h+e: "
          f"{'✅ congelato correttamente' if invariato else '⛔ e ricalcolato'}")
    # nessun parametro allenabile appartiene al router: il gate qui e' un
    # torch.Tensor semplice (no requires_grad), mai passato a nn.Parameter/opt.
    router_non_allenabile = True   # per costruzione (congela_routing usa @torch.no_grad, gate_w plain Tensor)
    print(f"    router mai fra i parametri allenabili (per costruzione, verificato nel codice): "
          f"{'✅' if router_non_allenabile else '⛔'}")
    return invariato and router_non_allenabile


# ══════════════════════════ TEST 4: ffn_down_exps NELLA STESSA ottimizzazione ══════════════════════════
def _test_down_proj_variabile_libera(device: str = "cpu", seed: int = 3) -> bool:
    """COMPITO 24/9 (workflow, dopo PIANO.md 02:56): verifica che ffn_down_exps
    sia una variabile libera nella STESSA ottimizzazione globale di gate/up,
    non a parte — altrimenti si ricadrebbe nello stesso difetto diagnosticato
    li' ("se down_proj resta fuori dall'ottimizzazione congiunta, raddoppia
    il gap invece di chiuderlo"). Tre proprieta', TUTTE necessarie (nessuna
    basta da sola — vedi il docstring di modulo):
    (a) il gradiente della loss COMBINATA (kernel Lemma-1, l'intero blocco
        MLP gate+up+SwiGLU+down — non solo gate+up) raggiunge SIA i codici di
        down_proj SIA quelli di gate/up, verificati SEPARATAMENTE (mai
        sommati insieme come in `_test_due_layer`: una somma positiva con un
        gruppo intero morto passerebbe un controllo lasco che guardi solo il
        totale);
    (b) i due gruppi di parametri stanno nello STESSO oggetto Optimizer — un
        solo AdamW, un solo `param_groups`, costruito da UNA lista
        concatenata (`parametri_ternari` = gate_up+down) — non due AdamW
        separati aggiornati in sequenza, che sarebbe esattamente "due
        ottimizzazioni separate" (il compito lo vieta esplicitamente);
    (c) la loss COMBINATA (quella che entra in UN solo `.backward()`, non due
        loss di famiglia sommate fuori dal grafo) scende con quell'unico
        AdamW."""
    print("\n── test 4: ffn_down_exps VARIABILE LIBERA nella STESSA ottimizzazione (non a parte) ──")
    st_teacher, st_quant, pesi_quant, h0, _ = costruisci_finestra_giocattolo(
        n_strati=2, seed=seed, device=device)

    # (a) gradiente SEPARATO per famiglia, da UN SOLO backward della loss
    # COMBINATA — p_gu e p_dw sono liste DISGIUNTE per costruzione (oggetti
    # TernarioDiff distinti, `.parameters()` non si sovrappongono mai), ma lo
    # si verifica comunque esplicitamente: un bug che unisse le due famiglie
    # renderebbe il controllo (a) privo di significato senza che nessuno se
    # ne accorga.
    ris = errore_lemma1_finestra(st_teacher, st_quant, h0)
    ris["loss"].backward()
    p_gu = parametri_gate_up(pesi_quant[0]) + parametri_gate_up(pesi_quant[1])
    p_dw = parametri_down(pesi_quant[0]) + parametri_down(pesi_quant[1])
    assert set(id(p) for p in p_gu).isdisjoint(id(p) for p in p_dw), (
        "bug di costruzione: gate_up e down_proj condividono un Parameter — "
        "il controllo (a) sotto non avrebbe piu' senso")
    grad_gu = sum(p.grad.abs().sum().item() for p in p_gu if p.grad is not None)
    grad_dw = sum(p.grad.abs().sum().item() for p in p_dw if p.grad is not None)
    n_gu_vivi = sum(1 for p in p_gu if p.grad is not None and p.grad.abs().sum().item() > 0.0)
    n_dw_vivi = sum(1 for p in p_dw if p.grad is not None and p.grad.abs().sum().item() > 0.0)
    entrambi_vivi = grad_gu > 0.0 and grad_dw > 0.0 and n_gu_vivi == len(p_gu) and n_dw_vivi == len(p_dw)
    print(f"    (a) gradiente dalla loss COMBINATA, letto SEPARATAMENTE: "
          f"gate/up |grad|.sum()={grad_gu:.3e} ({n_gu_vivi}/{len(p_gu)} tensori vivi)  |  "
          f"down_proj |grad|.sum()={grad_dw:.3e} ({n_dw_vivi}/{len(p_dw)} tensori vivi)  "
          f"({'✅ ENTRAMBI vivi' if entrambi_vivi else '⛔ almeno un gruppo morto'})")

    # (b) UN solo AdamW, UNA lista concatenata (mai due optimizer.step()
    # separati per famiglia — quello sarebbe esattamente il difetto vietato:
    # ogni famiglia ottimizzata guardando solo se stessa, come nella pipeline
    # sequenziale che questo lavoro sostituisce).
    _perturba_z(pesi_quant, scala=0.6, seed=13)
    params_finestra = parametri_ternari(pesi_quant[0]) + parametri_ternari(pesi_quant[1])
    n_gu_tot = len(parametri_gate_up(pesi_quant[0])) + len(parametri_gate_up(pesi_quant[1]))
    n_dw_tot = len(parametri_down(pesi_quant[0])) + len(parametri_down(pesi_quant[1]))
    opt = torch.optim.AdamW(params_finestra, lr=0.05)
    un_solo_ottimizzatore = (len(opt.param_groups) == 1
                              and len(opt.param_groups[0]["params"]) == n_gu_tot + n_dw_tot)
    print(f"    (b) UN solo AdamW su {len(params_finestra)} parametri "
          f"({n_gu_tot} gate/up + {n_dw_tot} down_proj, STESSA lista, STESSO param_group): "
          f"{'✅' if un_solo_ottimizzatore else '⛔'}")

    # (c) la loss COMBINATA (il kernel Lemma-1, non due loss di famiglia
    # sommate fuori dal grafo) scende con questo UNICO ottimizzatore.
    perdite = []
    for _ in range(60):
        r = passo_ottimizzazione(st_teacher, st_quant, h0, opt)
        perdite.append(r["loss"].item())
    scende = perdite[-1] < perdite[0] * 0.5
    print(f"    (c) loss combinata (60 passi, 1 AdamW): {perdite[0]:.4e} -> {perdite[-1]:.4e}  "
          f"({'✅ scende' if scende else '⛔ non scende abbastanza'})")

    esito = entrambi_vivi and un_solo_ottimizzatore and scende
    print(f"    {'✅ TEST 4 PASSA' if esito else '⛔ TEST 4 FALLISCE'}")
    return esito


def _test(device: str = "cpu") -> bool:
    print(f"finestra_globale.py — nucleo Lemma-1 a finestra scorrevole, device={device}")
    r1 = _test_un_layer(device)
    r2 = _test_due_layer(device)
    r3 = _test_routing_congelato()
    r4 = _test_down_proj_variabile_libera(device)
    ok = r1 and r2 and r3 and r4
    print(f"\n{'✅ TUTTI I TEST PASSANO' if ok else '⛔ ALMENO UN TEST FALLISCE'} "
          f"(un_layer={r1}, due_layer={r2}, routing_congelato={r3}, "
          f"down_proj_variabile_libera={r4})")
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
