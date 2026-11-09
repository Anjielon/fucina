#!/usr/bin/env python3
"""RACCORDO fra l'orchestratore della forgia scorrevole (compito B) e la
ripartenza di `ottimizza_finestra_JOINT` (compito A). Verificatore, 24/9 notte.

## Perche' esiste

I due compiti sono stati scritti in parallelo, e le due interfacce non combaciano
nel TIPO, non solo nel nome:

- B (`forgia_scorrevole.py`, blocco INTERFACCIA_A) chiama
  `ottimizza_finestra_JOINT(..., ternari_iniziali={L: {"gu": obj, "dw": obj}},
  ternari_uscita={})` e si aspetta di riavere GLI STESSI oggetti
  `TernarioDiffBatchFuso` per gli strati portati (controllo con `is`), piu' una
  `ternari_da_codici(codici, cfg, device)` per la modalita' (b).
- A (`strumenti/patch_ripartenza.py` + `ripartenza_codici.py`) ha realizzato
  l'aggancio come `ripartenza_per_strato={L: StatoStrato | dict di codici}` e
  `consegna_stato_per_strato={L: None}`: i parametri si COPIANO al bit in un
  modulo nuovo costruito dentro la finestra; i codici si ricostruiscono usando i
  pesi VERI del teacher (z dove lo metterebbe init_ls), che `ternari_da_codici`
  di B non riceve.

Nessuno dei due file e' stato cambiato: questo modulo traduce.

## Cosa fa

`ottimizza_finestra_JOINT_B` (interfaccia di B, sopra quella di A):
  1. oggetti portati -> `StatoStrato` (riferimenti, zero copie). Poi i parametri
     dell'oggetto si SVUOTANO (tensori vuoti): cosi' durante la finestra lo stato
     dello strato portato vive una volta sola (3 GiB su Thor), e si libera appena
     A lo ha copiato nel modulo nuovo. Se la finestra fallisce PRIMA della copia,
     l'oggetto viene ripristinato; dopo la copia resta vuoto (si riparte dal punto
     di ripresa su disco, che B scrive prima di ogni finestra).
  2. voci di `ternari_da_codici` (attributo `_codici_raccordo` sull'oggetto gate_up)
     -> il dizionario di CODICI passa ad A, che lo ricostruisce con i pesi del teacher.
  3. `consegna_stato_per_strato` su TUTTI gli strati della finestra (B vuole
     l'uscita completa) -> gli StatoStrato di fine finestra si RILEGANO negli
     stessi oggetti ricevuti (contratto `is` di B) o in oggetti nuovi, senza copie.
  Adam: THOR_SCORREVOLE_ADAM = "riparti" (default, INVARIATO: come il contratto di B
  e il codice di SliderQuant) oppure "conserva" (25/9, sotto).

## THOR_SCORREVOLE_ADAM=conserva (25/9 notte)

Perche': a pari calcolo per strato la scorrevole (2x75 passi) e' risultata PEGGIO
delle finestre separate (150) in ogni strato. Con "riparti" ogni finestra crea un
AdamW nuovo: il primo passo di un Adam azzerato vale lr*g/(|g|+eps) ~ lr*segno(g) su
OGNI parametro con gradiente, cioe' una scossa di ~lr su uno strato gia' ottimizzato.
Con "conserva" lo strato portato continua con lo STESSO stato di Adam:
  - a fine finestra `consegna_stato_per_strato` (A, con opt) da' anche i momenti
    exp_avg / exp_avg_sq e il contatore `step` di ogni parametro; il raccordo li
    attacca all'oggetto che esce (attributo CHIAVE_ADAM, su gu e su dw: riferimenti,
    zero copie) accanto ai parametri rilegati;
  - la finestra dopo li riprende (`adam_ripartenza="conserva"` di A ->
    `inietta_adam_`: momenti ADOTTATI, step clonato) per lo strato portato. Lo strato
    entrante resta com'e' oggi (teacher + LS o codici) con Adam NUOVO.
  - Cosa e' identico a una finestra sola: lo stato dell'ottimizzatore dello strato
    portato (momenti e step al bit, collaudo R8a). La TRAIETTORIA coincide con
    un'ottimizzazione unica solo se l'obiettivo non cambia fra le finestre (stessi
    strati, dati, beta, routing, lr: collaudo R8b); nella scorrevole vera l'obiettivo
    cambia (lo strato k e' commesso duro, entra k+2), quindi si conserva lo STATO, non
    si puo' conservare la traiettoria.
  - Ripresa: il punto di ripresa di B (forgia_scorrevole.Ripresa) salva l'attributo
    insieme ai parametri (+2x i parametri su disco: 6,08 GiB per strato portato su
    Thor). La variabile entra nella ricetta (tutte le THOR_*): una ripresa con modo
    diverso e' rifiutata da B. Se un oggetto portato arriva SENZA stato di Adam in
    modalita' conserva (punto scritto da un codice che non lo salvava, file toccato)
    il raccordo RIFIUTA prima di toccare l'oggetto: ripartire con Adam nuovo darebbe
    un'altra traiettoria sotto la stessa ricetta.
  Controlli rumorosi (prima di svuotare qualunque oggetto): stato mancante, chiavi
  diverse da {step, exp_avg, exp_avg_sq}, forma/tipo/device diversi dal parametro,
  step non scalare o < 1, stato presente in modalita' riparti.
  ⚠️ Costi e rischi (da misurare alla prima corsa su GPU): i momenti del portato
  (6,08 GiB su Thor) restano in VRAM anche FRA le finestre (transizione, verifica del
  commit, preparazione della finestra dopo) e dall'inizio della finestra, non dal primo
  passo; +6,08 GiB scritti in modo SINCRONO nel punto di ripresa (torch.save misurato
  0,63 GiB/s su @@ROOT:models@@: ~10 s a finestra); i momenti stimano il gradiente
  dell'obiettivo VECCHIO (vedi ripartenza_codici.py): conserva toglie la scossa, non
  garantisce un risultato migliore — lo decide il braccio S5. THOR_SCORREVOLE_ADAM=riparti
  scritta esplicitamente e' un'altra ricetta rispetto alla variabile assente (ripresa
  rifiutata): per riprendere una corsa vecchia non la si imposta.

`ternari_da_codici` (A2 di B): costruisce l'oggetto per il controllo di B
(`_da_codici_verificato`) con `imposta_da_codici_` di A e pesi di riferimento
D*Q (non c'e' il teacher qui): codici e scale esatti, ma z SATURO (|z| = atanh
0,95, lontano dalla soglia: con beta 16 il gradiente STE sarebbe ~0). Per questo
l'oggetto serve SOLO al controllo: porta con se' i codici (`_codici_raccordo`) e il
raccordo li passa ad A, che li rimette con il teacher. Gli oggetti del controllo
si liberano all'ingresso della finestra.

Una sola variabile d'ambiente nuova (25/9): THOR_SCORREVOLE_ADAM, letta a ogni
chiamata; assente = "riparti" = comportamento e file di prima, al bit. Collaudo:
test_raccordo_scorrevole.py (CPU, giocattolo; R8 = Adam conservato, `--adam` da solo).
"""
from __future__ import annotations

import inspect
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ripartenza_codici as rc  # noqa: E402
from quant_ternaria_globale_batch_v2 import TernarioDiffBatchFuso  # noqa: E402

PARAMETRI_A = ("ripartenza_per_strato", "consegna_stato_per_strato", "adam_ripartenza")
CHIAVE_CODICI = "_codici_raccordo"      # attributo dell'oggetto gate_up costruito da ternari_da_codici
# 25/9: stato di Adam dello strato portato, su OGNI oggetto (gu e dw): {nome parametro:
# {"step", "exp_avg", "exp_avg_sq"} | None (parametro mai aggiornato)}. Lo stesso nome lo
# usa forgia_scorrevole.Ripresa (ATTR_ADAM) per salvarlo nel punto di ripresa.
CHIAVE_ADAM = "_adam_raccordo"
VAR_ADAM = "THOR_SCORREVOLE_ADAM"
MODI_ADAM = ("riparti", "conserva")
CHIAVI_STATO_ADAM = frozenset({"step", "exp_avg", "exp_avg_sq"})   # AdamW senza amsgrad (quello di A)


def _valida():
    import valida_finestra_8_12 as V   # stesso modulo che usa forgia_scorrevole (import tardivo)
    return V


def verifica_patch_A(V=None) -> None:
    """Rifiuto rumoroso se la patch del compito A non e' applicata a valida_finestra_8_12."""
    V = V or _valida()
    firma = inspect.signature(V.ottimizza_finestra_JOINT).parameters
    mancano = [p for p in PARAMETRI_A if p not in firma]
    if mancano:
        raise RuntimeError(f"PATCH A ASSENTE: ottimizza_finestra_JOINT non ha {mancano}. "
                           "Applicare: python3 strumenti/patch_ripartenza.py")


# ═══════════════════════ oggetti <-> StatoStrato ═══════════════════════
def _controlla_oggetto(obj, nome: str) -> None:
    if not isinstance(obj, TernarioDiffBatchFuso):
        raise TypeError(f"{nome}: atteso TernarioDiffBatchFuso, ricevuto {type(obj).__name__}")
    if getattr(obj, "_svuotato_dal_raccordo", False):
        raise RuntimeError(f"{nome}: oggetto SVUOTATO da una finestra fallita dopo la copia: "
                           "lo stato vero e' solo nel punto di ripresa su disco")


def _svuota_(obj) -> None:
    """I parametri dell'oggetto diventano tensori vuoti (stesso device e dtype). Chi
    tiene ancora i tensori veri e' solo lo StatoStrato (riferimenti staccati)."""
    for n in rc.NOMI_PARAMETRI:
        p = getattr(obj, n)
        setattr(obj, n, nn.Parameter(torch.empty(0, device=p.device, dtype=p.dtype)))
    obj._svuotato_dal_raccordo = True


def _statici(obj) -> dict:
    return {"forma": tuple(obj.shape), "group": int(obj.group), "delta_max": float(obj.delta_max),
            "beta_0": float(obj.beta_0), "clip": float(obj.clip)}


def _rilega_(obj, parametri: dict, meta: dict) -> None:
    """Rimette nell'oggetto i tensori `parametri` (nessuna copia) dopo aver controllato
    che forma, gruppo e costanti del quantizzatore siano quelli dello stato."""
    st = _statici(obj)
    atteso = {k: meta[k] for k in st}
    if st != atteso:
        raise RuntimeError(f"rilegatura: oggetto {st} diverso dallo stato {atteso}")
    if tuple(parametri["z"].shape) != tuple(meta["gruppi"]):
        raise RuntimeError(f"rilegatura: z {tuple(parametri['z'].shape)} != {meta['gruppi']}")
    for n in rc.NOMI_PARAMETRI:
        setattr(obj, n, nn.Parameter(parametri[n]))
    obj._svuotato_dal_raccordo = False


def _nuovo_da_stato(parametri: dict, meta: dict) -> TernarioDiffBatchFuso:
    """Oggetto nuovo sui tensori dello stato, senza allocare il tensore dei pesi: il
    costruttore gira su device 'meta' (solo forme), poi i tre parametri si rilegano."""
    obj = TernarioDiffBatchFuso(torch.empty(meta["forma"], device="meta"), group=meta["group"])
    _rilega_(obj, parametri, meta)
    return obj


# ═══════════════════════ stato di Adam dello strato portato (25/9) ═══════════════════════
def modo_adam(valore: "str | None" = None) -> str:
    """THOR_SCORREVOLE_ADAM (o `valore` esplicito): 'riparti' (default) o 'conserva'."""
    m = os.getenv(VAR_ADAM, "riparti") if valore is None else valore
    if m not in MODI_ADAM:
        raise ValueError(f"{VAR_ADAM}={m!r}: attesi {MODI_ADAM}")
    return m


def _controlla_adam(obj, stato, nome: str) -> None:
    """Lo stato di Adam `stato` (formato CHIAVE_ADAM) deve appartenere ai parametri di `obj`:
    stesse chiavi, forma, tipo e device dei momenti; step scalare >= 1. Solo metadati e uno
    scalare: nessuna lettura dei momenti (6 GiB su Thor)."""
    if not isinstance(stato, dict) or set(stato) != set(rc.NOMI_PARAMETRI):
        raise RuntimeError(f"{nome}: stato di Adam con chiavi {sorted(stato) if isinstance(stato, dict) else type(stato)}"
                           f", attese {sorted(rc.NOMI_PARAMETRI)}")
    for n in rc.NOMI_PARAMETRI:
        st = stato[n]
        if st is None:
            continue                       # parametro mai aggiornato: nessuno stato, come in AdamW
        if not isinstance(st, dict) or set(st) != CHIAVI_STATO_ADAM:
            raise RuntimeError(f"{nome}.{n}: stato di Adam con chiavi "
                               f"{sorted(st) if isinstance(st, dict) else type(st)}, attese {sorted(CHIAVI_STATO_ADAM)}")
        p = getattr(obj, n)
        for k in ("exp_avg", "exp_avg_sq"):
            t = st[k]
            if not torch.is_tensor(t) or tuple(t.shape) != tuple(p.shape) or t.dtype != p.dtype or t.device != p.device:
                raise RuntimeError(
                    f"{nome}.{n}.{k}: forma/tipo/device "
                    f"{(tuple(t.shape), t.dtype, str(t.device)) if torch.is_tensor(t) else type(t)} != parametro "
                    f"{(tuple(p.shape), p.dtype, str(p.device))}")
        s = st["step"]
        if not torch.is_tensor(s) or s.numel() != 1 or not bool(torch.isfinite(s).all()) or float(s) < 1:
            raise RuntimeError(f"{nome}.{n}.step: atteso un contatore scalare >= 1, ricevuto "
                               f"{s if not torch.is_tensor(s) else (tuple(s.shape), s.tolist())}")


def _adam_da_oggetti(voce: dict, L: int) -> dict:
    """Stato di Adam portato dagli oggetti {'gu', 'dw'} (formato StatoStrato.adam), controllato."""
    uscita = {}
    for fam in ("gu", "dw"):
        obj = voce[fam]
        if obj is None:
            uscita[fam] = None
            continue
        stato = getattr(obj, CHIAVE_ADAM, None)
        if stato is None:
            raise RuntimeError(
                f"strato {L} {fam}: stato di Adam MANCANTE con {VAR_ADAM}=conserva (oggetto che non viene da una "
                "finestra con Adam conservato, o punto di ripresa scritto senza momenti): ripartire con Adam nuovo "
                "darebbe un'altra traiettoria sotto la stessa ricetta — rifiuto")
        _controlla_adam(obj, stato, f"strato {L} {fam}")
        uscita[fam] = stato
    return uscita


def _attacca_adam_(voce: dict, adam: dict, L: int) -> None:
    """Stato di Adam di fine finestra sugli oggetti che escono (riferimenti, zero copie)."""
    for fam in ("gu", "dw"):
        obj = voce[fam]
        if obj is None:
            if adam.get(fam) is not None:
                raise RuntimeError(f"strato {L}: stato di Adam per {fam} ma nessun oggetto {fam}")
            continue
        if adam.get(fam) is None:
            raise RuntimeError(f"strato {L} {fam}: la finestra non ha consegnato lo stato di Adam")
        _controlla_adam(obj, adam[fam], f"strato {L} {fam} (uscita)")
        setattr(obj, CHIAVE_ADAM, adam[fam])


# ═══════════════════════ A1 di B, sopra l'interfaccia di A ═══════════════════════
def ottimizza_finestra_JOINT_B(strati_correnti: dict, ids_batch: torch.Tensor, cfg, n_passi: int,
                               strati_teacher: dict, emb: torch.Tensor,
                               ternari_iniziali: "dict | None" = None,
                               ternari_uscita: "dict | None" = None, *, modo_adam_esplicito: "str | None" = None,
                               **kw) -> dict:
    """Contratto di B (forgia_scorrevole.py §INTERFACCIA ATTESA DAL COMPITO A, A1),
    realizzato con `ripartenza_per_strato`/`consegna_stato_per_strato` di A.
    `modo_adam_esplicito` (solo collaudo) sostituisce THOR_SCORREVOLE_ADAM."""
    V = _valida()
    verifica_patch_A(V)
    vietati = [k for k in PARAMETRI_A if k in kw]
    if vietati:
        raise TypeError(f"{vietati}: li gestisce il raccordo (usare l'interfaccia di A direttamente)")
    modo = modo_adam(modo_adam_esplicito)
    layers = sorted(strati_correnti)
    ripartenza: dict = {}
    portati: dict = {}
    adam_portati: dict = {}
    for L, voce in (ternari_iniziali or {}).items():
        if L not in layers:
            raise ValueError(f"ternari_iniziali: strato {L} fuori dalla finestra {layers}")
        if not isinstance(voce, dict) or "gu" not in voce or "dw" not in voce:
            raise TypeError(f"ternari_iniziali[{L}]: atteso {{'gu': ..., 'dw': ...}}")
        codici = getattr(voce["gu"], CHIAVE_CODICI, None)
        if codici is not None:
            # modalita' (b): i CODICI vanno ad A (ricostruzione col teacher); l'oggetto del
            # controllo di B ha z saturo e non deve allenare nulla: si libera qui
            ripartenza[L] = codici
            voce["gu"] = voce["dw"] = None
            continue
        _controlla_oggetto(voce["gu"], f"strato {L} gu")
        if voce["dw"] is not None:
            _controlla_oggetto(voce["dw"], f"strato {L} dw")
        if (voce["dw"] is not None) != bool(V.DOWN_TERNARIO):
            raise RuntimeError(f"strato {L}: down ternario {'presente' if voce['dw'] is not None else 'assente'} "
                               f"ma THOR_FORGIA_DOWN_TERNARIO={int(V.DOWN_TERNARIO)}")
        if modo == "conserva":
            adam_portati[L] = _adam_da_oggetti(voce, L)          # controlla TUTTO prima di toccare un oggetto
        elif any(o is not None and getattr(o, CHIAVE_ADAM, None) is not None for o in (voce["gu"], voce["dw"])):
            raise RuntimeError(f"strato {L}: l'oggetto portato ha uno stato di Adam ma {VAR_ADAM}=riparti: "
                               "ricetta incoerente (oggetto di una catena con Adam conservato)")
        ripartenza[L] = rc.estrai_stato_strato(voce["gu"], voce["dw"], origine="oggetto portato (raccordo B)")
        portati[L] = voce
    for L, voce in portati.items():
        if modo == "conserva":
            # lo stato passa allo StatoStrato (l'unico riferimento durante la finestra: A lo adotta)
            ripartenza[L].adam = adam_portati[L]
            for obj in (voce["gu"], voce["dw"]):
                if obj is not None:
                    delattr(obj, CHIAVE_ADAM)
        _svuota_(voce["gu"])
        if voce["dw"] is not None:
            _svuota_(voce["dw"])
    del adam_portati

    consegna = {L: None for L in layers}
    try:
        risultato = V.ottimizza_finestra_JOINT(
            strati_correnti, ids_batch, cfg, n_passi, strati_teacher=strati_teacher, emb=emb,
            ripartenza_per_strato=ripartenza or None, consegna_stato_per_strato=consegna,
            adam_ripartenza=modo, **kw)
    except BaseException:
        for L, voce in portati.items():
            st = ripartenza[L]
            if not st.consumato and st.parametri is not None:     # fallita PRIMA della copia: ripristino
                _rilega_(voce["gu"], st.parametri["gu"], st.meta["gu"])
                if voce["dw"] is not None:
                    _rilega_(voce["dw"], st.parametri["dw"], st.meta["dw"])
                if st.adam is not None:            # A inietta i momenti DOPO la copia: qui sono intatti
                    _attacca_adam_(voce, st.adam, L)
        raise
    non_applicati = [L for L, st in ripartenza.items() if isinstance(st, rc.StatoStrato) and not st.consumato]
    if non_applicati:
        raise RuntimeError(f"strati portati {non_applicati}: lo stato NON e' stato applicato dalla finestra")
    if modo == "conserva":
        non_iniettati = [L for L, st in ripartenza.items() if isinstance(st, rc.StatoStrato) and st.adam is not None]
        if non_iniettati:
            raise RuntimeError(f"strati portati {non_iniettati}: lo stato di Adam NON e' stato ripreso dalla finestra")

    uscita: dict = {}
    for L in layers:
        st = consegna[L]
        if not isinstance(st, rc.StatoStrato) or st.parametri is None:
            raise RuntimeError(f"strato {L}: la finestra non ha consegnato lo stato di fine finestra")
        if L in portati:
            obj = portati[L]
            _rilega_(obj["gu"], st.parametri["gu"], st.meta["gu"])
            if obj["dw"] is not None:
                _rilega_(obj["dw"], st.parametri["dw"], st.meta["dw"])
            uscita[L] = {"gu": obj["gu"], "dw": obj["dw"]}
        else:
            uscita[L] = {"gu": _nuovo_da_stato(st.parametri["gu"], st.meta["gu"]),
                         "dw": (None if st.parametri["dw"] is None
                                else _nuovo_da_stato(st.parametri["dw"], st.meta["dw"]))}
        if modo == "conserva":
            if st.adam is None:
                raise RuntimeError(f"strato {L}: la finestra non ha consegnato lo stato di Adam ({VAR_ADAM}=conserva)")
            _attacca_adam_(uscita[L], st.adam, L)
        st.parametri = None          # i tensori restano SOLO negli oggetti
        st.adam = None
        st.consumato = True
    if ternari_uscita is not None:
        ternari_uscita.clear()
        ternari_uscita.update(uscita)
    return risultato


# ═══════════════════════ A2 di B ═══════════════════════
def _oggetto_da_codici(D, Q, device, margine_z: float) -> TernarioDiffBatchFuso:
    D = torch.as_tensor(D)
    Q = torch.as_tensor(Q)
    E, n_out, n_in = Q.shape
    ng = D.shape[-1]
    G = n_in // ng
    W = (D.to(device, torch.float32).unsqueeze(-1)
         * Q.to(device).reshape(E, n_out, ng, G).to(torch.float32)).reshape(E, n_out, n_in)
    t = TernarioDiffBatchFuso(W, group=G).to(device)
    rc.imposta_da_codici_(t, D, Q, W, margine_z)      # solleva se codici o scale non tornano esatti
    return t


def ternari_da_codici(codici: dict, cfg=None, device="cpu") -> dict:
    """A2 di B. Oggetti SOLO per il controllo `_da_codici_verificato` (vedi docstring
    del modulo): la finestra riceve i codici, non questi oggetti."""
    V = _valida()
    margine = float(os.getenv("THOR_LR_Z", str(V.NUCLEO_LR)))
    D, Q = rc.codici_gate_up(codici)
    gu = _oggetto_da_codici(D, Q, device, margine)
    if V.DOWN_TERNARIO:
        if rc.FAMIGLIA_DW not in codici:
            raise RuntimeError("down ternario richiesto ma i codici non lo portano")
        dw = _oggetto_da_codici(*codici[rc.FAMIGLIA_DW], device, margine)
    else:
        if rc.FAMIGLIA_DW in codici:
            raise RuntimeError("THOR_FORGIA_DOWN_TERNARIO=0 ma i codici portano un down ternario: ricetta incoerente")
        dw = None
    # i codici viaggiano come attributo dell'oggetto gate_up: il dizionario resta {"gu", "dw"}
    # esattamente come lo vuole il contratto di B (e come lo leggono i suoi osservatori)
    setattr(gu, CHIAVE_CODICI, {**codici, "_origine": "codici su disco (raccordo B)"})
    return {"gu": gu, "dw": dw}
