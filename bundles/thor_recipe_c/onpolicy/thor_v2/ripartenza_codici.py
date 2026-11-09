#!/usr/bin/env python3
"""RIPARTENZA di uno strato ternario da PARAMETRI o da CODICI (24/9 sera).

## Perche' esiste

Decisione di Angelo (24/9): la ricetta e' la finestra SCORREVOLE a PASSO 1,
[0,1], [1,2], ..., [38,39], una sola catena (SliderQuant arXiv:2603.25284,
configurazione FSSW s=2 i=1; CBQ arXiv:2312.07950). Dopo la finestra [k,k+1]
lo strato k si commette; lo strato k+1 passa alla finestra [k+1,k+2] come
PRIMO strato e deve ripartire da dove la finestra precedente l'ha lasciato,
non dal teacher. Serve anche un secondo ingresso: far partire lo strato che
entra nuovo dai codici di una corsa PRECEDENTE (modalita' b, es. la forgia a
finestre separate `ThorT-veloce`), per confrontarla con l'init teacher + LS.

Due vie, stesso punto d'arrivo (i TernarioDiffBatchFuso della finestra nuova):

1. PARAMETRI (`estrai_stato_strato` -> `applica_ripartenza_`): z, log_s e
   delta_raw si copiano bit per bit. E' la via da usare per lo strato che
   scorre DENTRO la stessa corsa: esatta per costruzione (stessi numeri, stesso
   nucleo compilato), conserva lo stato FINE di z (quanto ogni peso e' vicino
   a cambiare codice: con beta=16 il gradiente STE vive solo entro ~0,1 dalla
   soglia, quindi e' quella distanza che decide chi puo' ancora cambiare), e
   non costa niente (nessuna ricostruzione, nessuna ricerca).
   E' anche cio' che fa il codice pubblico di SliderQuant: lo strato in
   sovrapposizione resta lo STESSO modulo, con i suoi parametri appresi
   (`quantize/sliderquant.py`, `replace_ori_layer` a fine giro).

2. CODICI (D, Q), in memoria (il `risultato` di ottimizza_finestra_JOINT) o da
   una cartella di corsa (`carica_codici_corsa`, con verifica della marca):
   D e Q definiscono i pesi duri, non il latente. Si ricostruisce un
   (z, log_s, delta_raw) che ridia ESATTAMENTE D*Q, con z messo dove lo
   metterebbe `init_ls.reinit_ls_`:

   - log_s: si cerca, fra i float32 vicini a log(D), quello per cui l'exp
     calcolata DAL NUCLEO di forward_duro da' esattamente D (l'oracolo e' il
     nucleo stesso: exp compilata e exp di torch non coincidono per forza su
     GPU, e sono i pesi del nucleo quelli che l'ottimizzazione vede e
     commette). Priorita': esatto per nucleo ed eager > esatto per il nucleo >
     il piu' vicino. Quanti gruppi restano non esatti si scrive nel log.
   - z "naturale" come in reinit_ls_: z_nat = atanh(clip(W/D)) / beta_0, con
     W i pesi di riferimento (teacher, ruotati per gate/up) e D la scala dei
     codici.
   - soglia per gruppo: il delta che rende la decisione |z_nat| > delta il
     piu' possibile uguale a [Q != 0] (minimo numero di disaccordi, fra due
     valori consecutivi di |z_nat| ordinati). Cosi' il massimo numero di pesi
     conserva la propria posizione naturale.
   - i pesi in disaccordo (il codice ottimizzato non e' l'arrotondamento
     naturale, o ha il segno opposto al peso) si PROIETTANO dal lato giusto a
     `margine_z` dalla soglia: il margine di default e' un passo di Adam
     (lr_z), cioe' il codice regge un passo di gradiente contrario ed e' li'
     dove il gradiente STE e' massimo. Tutti gli altri restano dove sono,
     spostati solo se cadono entro un margine NUMERICO relativo 1e-5 dalla
     soglia (la sigmoide del nucleo e quella di torch possono differire di un
     ulp: la decisione non deve dipendere da quale delle due si usa).

   Rifiuti rumorosi: Q deve tornare identico al 100% sia con la regola di
   export (`_estrai_famiglia`: z vs delta di torch) sia con la decisione del
   nucleo (segno di forward_duro); altrimenti RuntimeError.

## Stato di Adam: si riparte (default) o si conserva (`adam_ripartenza`)

Verificato nel codice pubblico di SliderQuant (github.com/deep-optimization/
SliderQuant, `quantize/sliderquant.py` righe ~620-690): per OGNI finestra si
crea un AdamW nuovo (`optimizer = torch.optim.AdamW(params)`) e un
programmatore di lr nuovo, e a fine finestra `del optimizer`; i parametri
dello strato in sovrapposizione invece restano. CBQ nel testo non lo dice
(verificato: il paper e' muto sul punto). Default = riparti, come l'unico
precedente pubblico della stessa ricetta; "conserva" resta disponibile per
l'A/B (`inietta_adam_`), perche' qui il caso non e' identico: SliderQuant
porta il lr a zero dentro ogni finestra (schedulatore lineare), la nostra
ricetta usa lr costante, e l'obiettivo dello strato k+1 cambia davvero fra le
due finestre (nella [k+1,k+2] il suo errore passa anche attraverso k+2, nella
[k,k+1] no): i momenti di Adam sono stime del gradiente dell'obiettivo VECCHIO.
Si decide misurando, non qui.

⛔ Questo file e' NUOVO: non tocca valida_finestra_8_12.py ne' altro. Lo
aggancio in ottimizza_finestra_JOINT e' la patch strumenti/patch_ripartenza.py.
Collaudo: test_ripartenza_codici.py (CPU, forme giocattolo e forme di gruppo vere).
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FAMIGLIE_GU = ("ffn_gate_exps", "ffn_up_exps")
FAMIGLIA_DW = "ffn_down_exps"
NOMI_PARAMETRI = ("log_s", "z", "delta_raw")          # = named_parameters() di TernarioDiffBatch
# nomi dei file su disco, STESSI di scrivi_codici_su_disco
FILE_PER_FAMIGLIA = {"ffn_gate_exps": ("Dg", "Qg"), "ffn_up_exps": ("Du", "Qu"), "ffn_down_exps": ("Dd", "Qd")}

MARGINE_REL_NUMERICO = 1e-5   # distanza relativa minima dalla soglia (decisione indipendente dall'ulp)
DELTA_MIN = 1e-6              # stesso pavimento di init_ls.reinit_ls_
BLOCCO_ESPERTI = 16           # come reinit_ls_: su Thor un blocco gate_up e' 16x1024x2048
RICERCA_ULP = 8               # candidati per log_s: log(D) +- 8 ulp (l'immagine di exp e' rada)
BETA_ORACOLO = 16.0           # la forma dura non dipende da beta (entra solo nel surrogato)


# ═══════════════════════════ codici e pesi di un modulo ═══════════════════════════
def _forma_gruppi(t) -> tuple:
    return tuple(t.z.shape)            # (E, n_out, ng, G)


@torch.no_grad()
def _codici_eager_blocco(t, b: slice) -> "tuple[torch.Tensor, torch.Tensor]":
    """STESSA regola di valida_finestra_8_12._estrai_famiglia (quella che va su
    disco e nel GGUF): D = exp(log_s) di torch, Q = z vs delta di torch."""
    s = t.log_s[b].exp().squeeze(-1).to(torch.float32)
    delta = (torch.sigmoid(t.delta_raw[b]) * t.delta_max)
    z = t.z[b]
    uno = torch.ones((), dtype=torch.int8, device=z.device)
    Q = torch.where(z > delta, uno, torch.where(z < -delta, -uno, torch.zeros_like(uno)))
    return s, Q


@torch.no_grad()
def _pesi_nucleo(t) -> torch.Tensor:
    """Pesi duri come li calcola il NUCLEO compilato di forward_duro (float32):
    sono quelli che la finestra vede nel forward e che committa negli strati."""
    return t.forward_duro(BETA_ORACOLO).detach().reshape(_forma_gruppi(t))


@torch.no_grad()
def verifica_contro_codici(t, D: torch.Tensor, Q: torch.Tensor) -> dict:
    """Confronta il modulo `t` con i codici (D (E,n_out,ng) float32, Q (E,n_out,ng,G)
    int8), sullo stesso device di t. Misura DUE letture del modulo:
      - eager  (regola di export, _estrai_famiglia): cio' che finirebbe su disco;
      - nucleo (forward_duro compilato): cio' che l'ottimizzazione usa e commette.
    Ritorna frazioni di codici identici e scarto massimo dei pesi (assoluto e
    relativo alla scala) per entrambe."""
    W_n = _pesi_nucleo(t)
    E = W_n.shape[0]
    tot = 0
    uguali_e = uguali_n = 0
    sc_e = sc_n = rel_n = rel_e = 0.0
    for e0 in range(0, E, BLOCCO_ESPERTI):
        b = slice(e0, min(E, e0 + BLOCCO_ESPERTI))
        Qb = Q[b]
        Db = D[b].unsqueeze(-1)
        atteso = Db * Qb.to(torch.float32)                 # prodotto esatto: q in {-1,0,1}
        s_e, Q_e = _codici_eager_blocco(t, b)
        W_e = s_e.unsqueeze(-1) * Q_e.to(torch.float32)
        Q_n = torch.sign(W_n[b]).to(torch.int8)            # s > 0 sempre: segno = codice
        tot += Qb.numel()
        uguali_e += (Q_e == Qb).sum().item()
        uguali_n += (Q_n == Qb).sum().item()
        d_e = (W_e - atteso).abs()
        sc_e = max(sc_e, d_e.max().item())
        rel_e = max(rel_e, (d_e / Db).max().item())
        d_n = (W_n[b] - atteso).abs()
        sc_n = max(sc_n, d_n.max().item())
        rel_n = max(rel_n, (d_n / Db).max().item())
        del atteso, W_e, Q_e, Q_n, d_n, d_e
    del W_n
    return {"codici_identici_eager": uguali_e / max(tot, 1), "codici_identici_nucleo": uguali_n / max(tot, 1),
            "scarto_max_pesi_eager": sc_e, "scarto_max_pesi_nucleo": sc_n,
            "scarto_max_rel_eager": rel_e, "scarto_max_rel_nucleo": rel_n, "n_pesi": tot}


# ═════════════════════════ via 2: ricostruzione dai CODICI ═════════════════════════
def _controlla_codici(nome: str, D, Q) -> None:
    """Forme, tipi e valori: rifiuto rumoroso di tutto cio' che non e' un codice valido."""
    if D.ndim != 3 or Q.ndim != 3:
        raise ValueError(f"{nome}: attesi D (E,n_out,ng) e Q (E,n_out,n_in), ricevuti {tuple(D.shape)} {tuple(Q.shape)}")
    if tuple(D.shape[:2]) != tuple(Q.shape[:2]) or Q.shape[2] % D.shape[2] != 0:
        raise ValueError(f"{nome}: D {tuple(D.shape)} e Q {tuple(Q.shape)} non compatibili")
    Dt, Qt = torch.as_tensor(D), torch.as_tensor(Q)
    if Dt.dtype != torch.float32 or Qt.dtype != torch.int8:
        raise ValueError(f"{nome}: tipi D {Dt.dtype} / Q {Qt.dtype}, attesi float32 / int8")
    if Qt.numel() and (Qt.min().item() < -1 or Qt.max().item() > 1):
        raise ValueError(f"{nome}: Q fuori da {{-1,0,1}} (min {Qt.min().item()}, max {Qt.max().item()})")
    if not torch.isfinite(Dt).all() or (Dt <= 0).any():
        raise ValueError(f"{nome}: scale D non finite o <= 0 (exp(log_s) e' sempre > 0: codici non prodotti "
                         f"da questa parametrizzazione)")


def _ulp_passo(y: torch.Tensor, k: int) -> torch.Tensor:
    verso = torch.full_like(y, float("inf") if k > 0 else float("-inf"))
    for _ in range(abs(k)):
        y = torch.nextafter(y, verso)
    return y


@torch.no_grad()
def _imposta_log_s_esatto_(t, D: torch.Tensor, ricerca_ulp: int = RICERCA_ULP) -> dict:
    """log_s tale che il NUCLEO di forward_duro dia esattamente D.

    L'immagine di exp sui float32 e' rada (un ulp di log_s sposta D di |log_s|
    ulp relativi: 4-8 per le scale di Thor), quindi log(D) arrotondato non
    basta. Oracolo = forward_duro stesso, con z = 4*delta_max (codice +1 su ogni
    peso: l'uscita e' la scala s come la calcola il nucleo). Si provano log(D)
    e i +-`ricerca_ulp` vicini, dal piu' vicino al piu' lontano; per gruppo vince
    il primo candidato esatto per nucleo ED eager, poi il primo esatto per il
    nucleo, poi il piu' vicino. `t.z` viene usato come area di lavoro: il
    chiamante lo riscrive subito dopo."""
    E, n_out, ng, G = _forma_gruppi(t)
    y0 = torch.log(D.to(torch.float64)).to(torch.float32)    # log correttamente arrotondato
    t.z.fill_(4.0 * float(t.delta_max))
    rango = torch.zeros_like(D, dtype=torch.int8)           # 0 niente, 1 vicino, 2 esatto nucleo, 3 esatto entrambi
    y_best = y0.clone()
    err_best = torch.full_like(D, float("inf"))
    passi = [0] + [s * k for k in range(1, ricerca_ulp + 1) for s in (-1, 1)]
    for k in passi:
        y = _ulp_passo(y0, k)
        t.log_s.copy_(y.unsqueeze(-1))
        s_n = _pesi_nucleo(t)[..., 0].clone()
        s_e = y.exp()
        err = (s_n - D).abs()
        r = torch.where(err == 0, torch.where(s_e == D, 3, 2), 1).to(torch.int8)
        meglio = (r > rango) | ((r == 1) & (rango == 1) & (err < err_best))
        y_best = torch.where(meglio, y, y_best)
        err_best = torch.where(meglio, err, err_best)
        rango = torch.where(meglio, r, rango)
        del s_n
    t.log_s.copy_(y_best.unsqueeze(-1))
    n = rango.numel()
    return {"scale_esatte_entrambi": (rango == 3).sum().item() / n,
            "scale_esatte_nucleo": (rango >= 2).sum().item() / n,
            "scala_max_rel_non_esatta": (err_best / D).max().item() if (rango < 2).any() else 0.0}


def _soglia_minimo_disaccordo(a: torch.Tensor, nz: torch.Tensor, margine_z: float) -> torch.Tensor:
    """Per gruppo: delta che rende [a > delta] il piu' possibile uguale a nz.
    a = |z_nat| (..., G), nz = codice != 0 (..., G). Mettendo la soglia dopo i
    primi i valori ordinati, i disaccordi sono c_i + (G - i) - (N - c_i), con c_i
    i non nulli fra i primi i. A parita' vince la soglia piu' bassa (argmin
    restituisce il primo minimo). delta = punto medio fra i due valori che la
    soglia separa (i=G: tutti nulli, delta = max|z_nat| + margine_z)."""
    G = a.shape[-1]
    a_s, idx = a.sort(dim=-1)
    nz_s = nz.gather(-1, idx).to(torch.int32)
    c = torch.cat([torch.zeros_like(nz_s[..., :1]), nz_s.cumsum(-1, dtype=torch.int32)], dim=-1)
    N = c[..., -1:]
    i = torch.arange(G + 1, device=a.device, dtype=torch.int32)
    i_best = (2 * c + G - i - N).argmin(dim=-1, keepdim=True)
    lo = torch.where(i_best > 0, a_s.gather(-1, (i_best - 1).clamp(min=0)), torch.zeros_like(a_s[..., :1]))
    hi = torch.where(i_best < G, a_s.gather(-1, i_best.clamp(max=G - 1)), lo + 2.0 * margine_z)
    return 0.5 * (lo + hi)


def _colloca_z(z_nat: torch.Tensor, q: torch.Tensor, delta: torch.Tensor, margine_z: float) -> "tuple":
    """z col codice q rispetto alla soglia delta (la stessa che il modulo calcola
    da delta_raw). Pesi d'accordo col proprio codice: z_nat, spostato solo se cade
    entro il margine NUMERICO dalla soglia. Pesi in disaccordo: proiettati a
    `margine_z` dalla soglia, dal lato del codice (q=0: dentro, col segno di z_nat)."""
    qf = q.to(z_nat.dtype)
    fuori = delta * (1.0 + MARGINE_REL_NUMERICO)
    dentro = delta * (1.0 - MARGINE_REL_NUMERICO)
    mag = z_nat.abs()
    segno = torch.sign(z_nat)
    ok_nz = (mag > delta) & (segno == qf)
    ok_zero = mag <= delta
    z_nz = qf * torch.where(ok_nz, torch.maximum(mag, fuori), torch.maximum(delta + margine_z, fuori))
    dentro_viol = torch.minimum(torch.maximum(delta - margine_z, 0.5 * delta), dentro)
    z_zero = segno * torch.where(ok_zero, torch.minimum(mag, dentro), dentro_viol)
    nz = q != 0
    z = torch.where(nz, z_nz, z_zero)
    proiettati = torch.where(nz, ~ok_nz, ~ok_zero)
    spinti = (~proiettati) & (z != z_nat)
    return z, proiettati.sum().item(), spinti.sum().item()


@torch.no_grad()
def imposta_da_codici_(t, D, Q, W_rif: torch.Tensor, margine_z: float,
                       ricerca_ulp: int = RICERCA_ULP) -> dict:
    """Sovrascrive IN-PLACE z, log_s, delta_raw di `t` (TernarioDiffBatch[Fuso])
    perche' i pesi duri siano esattamente D*Q (vedi docstring del modulo).
    D (E,n_out,ng) float32, Q (E,n_out,n_in) int8 — numpy o torch — nello
    STESSO dominio del modulo (gate/up: RUOTATO, come li scrive _estrai_famiglia);
    W_rif (E,n_out,n_in): pesi di riferimento nello stesso dominio (come per
    reinit_ls_). Solleva se il risultato non riproduce i codici al 100%."""
    E, n_out, ng, G = _forma_gruppi(t)
    _controlla_codici("codici", D, Q)
    dev = t.z.device
    D = torch.as_tensor(D).to(dev, torch.float32)
    Q = torch.as_tensor(Q).to(dev, torch.int8)
    if tuple(D.shape) != (E, n_out, ng) or tuple(Q.shape) != (E, n_out, ng * G):
        raise ValueError(f"codici D {tuple(D.shape)} / Q {tuple(Q.shape)} non per un modulo {(E, n_out, ng * G)} "
                         f"a gruppi di {G}")
    if tuple(W_rif.shape) != (E, n_out, ng * G):
        raise ValueError(f"W_rif {tuple(W_rif.shape)} != {(E, n_out, ng * G)}")
    Q = Q.reshape(E, n_out, ng, G)

    esito = _imposta_log_s_esatto_(t, D, ricerca_ulp)
    beta_0, clip, dmax = float(t.beta_0), float(t.clip), float(t.delta_max)
    proiettati = spinti = 0
    for e0 in range(0, E, BLOCCO_ESPERTI):
        b = slice(e0, min(E, e0 + BLOCCO_ESPERTI))
        Wg = W_rif[b].detach().to(dev, torch.float32).reshape(-1, n_out, ng, G)
        Db = D[b].unsqueeze(-1)
        z_nat = torch.atanh((Wg / Db).clamp(-clip, clip)) / beta_0
        delta = _soglia_minimo_disaccordo(z_nat.abs(), Q[b] != 0, margine_z)
        delta = delta.clamp(DELTA_MIN, dmax * (1 - 1e-6))
        p = delta / dmax
        t.delta_raw[b].copy_(torch.log(p) - torch.log1p(-p))
        delta_eff = torch.sigmoid(t.delta_raw[b]) * dmax        # = t._delta_norm()[b]
        z, n_p, n_s = _colloca_z(z_nat, Q[b], delta_eff, margine_z)
        t.z[b].copy_(z)
        proiettati += n_p
        spinti += n_s
        del Wg, z_nat, z
    v = verifica_contro_codici(t, D, Q)
    esito.update(v)
    esito.update({"frazione_proiettati": proiettati / Q.numel(), "frazione_spinti_numerici": spinti / Q.numel(),
                  "margine_z": margine_z})
    if v["codici_identici_eager"] < 1.0 or v["codici_identici_nucleo"] < 1.0:
        raise RuntimeError(f"ricostruzione dai codici FALLITA: codici identici eager "
                           f"{v['codici_identici_eager']*100:.6f}% / nucleo {v['codici_identici_nucleo']*100:.6f}%")
    # scala: esatta dove l'immagine di exp lo permette; altrove l'errore e' il
    # passo dell'immagine (<= ~|log D| ulp relativi). Oltre 64 ulp e' un difetto.
    # Vale per entrambe le letture (nucleo: forward e commit; eager: export).
    if max(v["scarto_max_rel_nucleo"], v["scarto_max_rel_eager"]) > 64 * 2.0 ** -23:
        raise RuntimeError(f"ricostruzione dai codici: scala oltre il passo dell'immagine di exp "
                           f"(scarto relativo nucleo {v['scarto_max_rel_nucleo']:.3e}, "
                           f"eager {v['scarto_max_rel_eager']:.3e})")
    return esito


def codici_gate_up(voce: dict) -> tuple:
    """(D, Q) della famiglia FUSA gate_up (righe 0:meta gate, meta:2*meta up —
    convenzione di _esporta_gate_up) dai due pezzi separati del risultato/disco."""
    (Dg, Qg), (Du, Qu) = voce["ffn_gate_exps"], voce["ffn_up_exps"]
    _controlla_codici("ffn_gate_exps", Dg, Qg)
    _controlla_codici("ffn_up_exps", Du, Qu)
    if tuple(Dg.shape) != tuple(Du.shape) or tuple(Qg.shape) != tuple(Qu.shape):
        raise ValueError(f"gate {tuple(Dg.shape)}/{tuple(Qg.shape)} e up {tuple(Du.shape)}/{tuple(Qu.shape)} diversi")
    cat = (lambda a, b: torch.cat([torch.as_tensor(a), torch.as_tensor(b)], dim=1))
    return cat(Dg, Du), cat(Qg, Qu)


# ═════════════════════════ via 1: passaggio dei PARAMETRI ═════════════════════════
@dataclass
class StatoStrato:
    """Stato FINE di uno strato a fine finestra, da passare alla finestra dopo.
    `parametri`: {"gu": {nome: tensore}, "dw": {nome: tensore} | None} — riferimenti
    staccati (detach) ai parametri, nessuna copia. `adam`: {"gu": {nome: {chiave:
    tensore}}, "dw": ...} o None. `codici_rif`: il risultato[L] della finestra che
    li ha prodotti (i codici scritti su disco): il riferimento del controllo."""
    parametri: "dict | None"
    meta: dict
    adam: "dict | None" = None
    codici_rif: "dict | None" = None
    origine: str = ""
    consumato: bool = field(default=False)


def _meta(t) -> dict:
    return {"forma": tuple(t.shape), "gruppi": tuple(t.z.shape), "group": int(t.group),
            "delta_max": float(t.delta_max), "beta_0": float(t.beta_0), "clip": float(t.clip)}


def _parametri(t) -> dict:
    d = {n: p.detach() for n, p in t.named_parameters()}
    if set(d) != set(NOMI_PARAMETRI):
        raise RuntimeError(f"parametri inattesi {sorted(d)} (attesi {sorted(NOMI_PARAMETRI)})")
    return d


def _stato_adam(opt, t) -> dict:
    uscita = {}
    for n, p in t.named_parameters():
        st = opt.state.get(p)
        uscita[n] = None if not st else {k: (v.detach() if torch.is_tensor(v) else v) for k, v in st.items()}
    return uscita


def estrai_stato_strato(ternari_gu, ternari_dw, opt=None, codici_rif: "dict | None" = None,
                        origine: str = "") -> StatoStrato:
    """Fotografa lo stato di uno strato a fine finestra (riferimenti, zero copie:
    i parametri non cambiano piu' dopo l'ottimizzazione). Con `opt` porta anche
    lo stato di Adam (6 GiB per strato su Thor: chiederlo solo se lo si conserva).

    ⚠️ Sono RIFERIMENTI agli stessi tensori del modulo e dell'ottimizzatore
    d'origine: vanno estratti quando quell'ottimizzatore ha FINITO (in
    ottimizza_finestra_JOINT: dopo l'ultimo passo della finestra). Se l'origine
    facesse altri passi, lo stato cambierebbe sotto i piedi; in collaudo (dove
    l'origine continua per confronto) i momenti vanno clonati: misurato, senza
    clone i due ottimizzatori aggiornano gli stessi tensori e divergono."""
    meta = {"gu": _meta(ternari_gu), "dw": None if ternari_dw is None else _meta(ternari_dw)}
    parametri = {"gu": _parametri(ternari_gu), "dw": None if ternari_dw is None else _parametri(ternari_dw)}
    adam = None
    if opt is not None:
        adam = {"gu": _stato_adam(opt, ternari_gu), "dw": None if ternari_dw is None else _stato_adam(opt, ternari_dw)}
    return StatoStrato(parametri=parametri, meta=meta, adam=adam, codici_rif=codici_rif, origine=origine)


@torch.no_grad()
def _applica_parametri_(t_gu, t_dw, voce: StatoStrato) -> dict:
    if voce.consumato or voce.parametri is None:
        raise RuntimeError("StatoStrato gia' consumato: ogni stato si applica una volta sola")
    if (t_dw is None) != (voce.parametri["dw"] is None):
        raise RuntimeError(f"down ternario: la finestra {'NON ' if t_dw is None else ''}lo ottimizza, lo stato "
                           f"{'lo porta' if voce.parametri['dw'] is not None else 'non lo porta'} — ricetta incoerente")
    esito = {}
    for fam, t in (("gu", t_gu), ("dw", t_dw)):
        if t is None:
            continue
        if _meta(t) != voce.meta[fam]:
            raise RuntimeError(f"{fam}: modulo {_meta(t)} diverso da quello dello stato {voce.meta[fam]}")
        sorgente = voce.parametri[fam]
        for n, p in t.named_parameters():
            s = sorgente[n]
            if s.shape != p.shape or s.dtype != p.dtype:
                raise RuntimeError(f"{fam}.{n}: forma/tipo {tuple(s.shape)}/{s.dtype} != {tuple(p.shape)}/{p.dtype}")
            p.copy_(s)
            if not torch.equal(p, s):
                raise RuntimeError(f"{fam}.{n}: copia non identica al bit")
        rif = None if voce.codici_rif is None else (
            codici_gate_up(voce.codici_rif) if fam == "gu" else voce.codici_rif[FAMIGLIA_DW])
        if rif is not None:
            E, n_out, ng, G = _forma_gruppi(t)
            D = torch.as_tensor(rif[0]).to(t.z.device, torch.float32)
            Q = torch.as_tensor(rif[1]).to(t.z.device, torch.int8).reshape(E, n_out, ng, G)
            v = verifica_contro_codici(t, D, Q)
            # la regola di export sugli STESSI numeri deve ridare gli stessi codici: al bit
            if v["codici_identici_eager"] < 1.0 or v["scarto_max_pesi_eager"] != 0.0:
                raise RuntimeError(f"{fam}: i parametri passati NON ridanno i codici della finestra precedente "
                                   f"(identici {v['codici_identici_eager']*100:.6f}%, scarto {v['scarto_max_pesi_eager']:.3e})")
            esito[fam] = v
        else:
            esito[fam] = {"codici_identici_eager": 1.0, "scarto_max_pesi_eager": 0.0, "riferimento": "parametri"}
    voce.parametri = None          # i tensori della finestra precedente si liberano (3 GiB/strato su Thor)
    voce.consumato = True
    return esito


def inietta_adam_(opt, t_gu, t_dw, voce: StatoStrato) -> int:
    """Rimette nello stato di `opt` (nuovo, appena creato) i momenti di Adam dello
    strato. Il passo resta per parametro (AdamW, anche fused, tiene `step` per
    tensore): lo strato che continua ha il suo contatore, quello nuovo parte da 0.
    I momenti si ADOTTANO (niente copia: +6 GiB evitati su Thor), il contatore
    `step` si clona sul device che l'ottimizzatore esige (fused: quello del
    parametro). Ritorna il numero di tensori iniettati. Consuma `voce.adam`."""
    if voce.adam is None:
        return 0
    presenti = {id(p): g for g in opt.param_groups for p in g["params"]}
    n = 0
    for fam, t in (("gu", t_gu), ("dw", t_dw)):
        if t is None:
            continue
        stati = voce.adam.get(fam)
        if stati is None:
            raise RuntimeError(f"stato di Adam senza la famiglia {fam}")
        for nome, p in t.named_parameters():
            st = stati[nome]
            if st is None:
                continue                  # parametro mai aggiornato nella finestra precedente
            g = presenti.get(id(p))
            if g is None:
                raise RuntimeError(f"{fam}.{nome}: il parametro non e' nell'ottimizzatore")
            if opt.state.get(p):
                raise RuntimeError(f"{fam}.{nome}: l'ottimizzatore ha gia' uno stato per questo parametro")
            for k in ("exp_avg", "exp_avg_sq"):
                if st[k].shape != p.shape or st[k].device != p.device:
                    raise RuntimeError(f"{fam}.{nome}.{k}: {tuple(st[k].shape)}@{st[k].device} != "
                                       f"{tuple(p.shape)}@{p.device}")
            nuovo = dict(st)
            dev_step = p.device if (g.get("fused") or g.get("capturable")) else torch.device("cpu")
            nuovo["step"] = st["step"].detach().to(dev_step, torch.float32).clone()
            opt.state[p] = nuovo
            n += 1
    voce.adam = None
    return n


# ═════════════════════════ ingresso unico per ottimizza_finestra_JOINT ═════════════════════════
def applica_ripartenza_(L: int, voce, ternari_gu, ternari_dw, W_gu_rif, W_dw_rif, margine_z: float,
                        conserva_adam: bool = False, log=print) -> dict:
    """Punto d'aggancio della patch: imposta i ternari dello strato L dalla `voce`.
    voce = StatoStrato (via 1, parametri) oppure dict di codici come risultato[L]
    / `carica_codici_corsa` (via 2). W_gu_rif / W_dw_rif: funzioni senza argomenti
    che danno i pesi di riferimento nel dominio del modulo (servono solo alla via 2:
    su Thor sono 2 GiB, si calcolano solo se servono). Scrive UNA riga di log per
    strato: codici identici ai forniti (atteso 100%) e scarto massimo dei pesi (atteso 0)."""
    if isinstance(voce, StatoStrato):
        if voce.adam is not None and not conserva_adam:
            voce.adam = None      # si riparte: i momenti (6 GiB/strato su Thor) si liberano subito
            log(f"  ripartenza strato {L}: stato di Adam presente ma SCARTATO (adam_ripartenza=riparti)")
        esito = _applica_parametri_(ternari_gu, ternari_dw, voce)
        via = f"PARAMETRI ({voce.origine or 'finestra precedente'})"
    elif isinstance(voce, dict) and all(f in voce for f in FAMIGLIE_GU):
        esito = {}
        D, Q = codici_gate_up(voce)
        esito["gu"] = imposta_da_codici_(ternari_gu, D, Q, W_gu_rif(), margine_z)
        if ternari_dw is None:
            if FAMIGLIA_DW in voce:
                raise RuntimeError(f"strato {L}: i codici portano un down ternario ma la finestra tiene il "
                                   f"down della base (THOR_FORGIA_DOWN_TERNARIO=0): ricetta incoerente")
        else:
            if FAMIGLIA_DW not in voce:
                raise RuntimeError(f"strato {L}: la finestra ottimizza il down ternario ma i codici non lo portano")
            Dd, Qd = voce[FAMIGLIA_DW]
            esito["dw"] = imposta_da_codici_(ternari_dw, Dd, Qd, W_dw_rif(), margine_z)
        via = f"CODICI ({voce.get('_origine', 'in memoria')})"
    else:
        raise TypeError(f"strato {L}: voce di ripartenza di tipo {type(voce).__name__} non riconosciuta")

    parti = []
    for fam, v in esito.items():
        pezzo = (f"{fam}: codici identici {v['codici_identici_eager']*100:.4f}% · "
                 f"scarto massimo pesi {v['scarto_max_pesi_eager']:.3e}")
        if "codici_identici_nucleo" in v:
            pezzo += (f" (nucleo: {v['codici_identici_nucleo']*100:.4f}%, "
                      f"{v['scarto_max_pesi_nucleo']:.3e})")
        if "frazione_proiettati" in v:
            pezzo += (f" · proiettati {v['frazione_proiettati']*100:.3f}% · scale esatte "
                      f"{v['scale_esatte_nucleo']*100:.3f}%")
        parti.append(pezzo)
    log(f"  ripartenza strato {L} da {via}: " + " | ".join(parti))
    return esito


# ═════════════════════════ codici da una cartella di corsa ═════════════════════════
def carica_codici_corsa(cartella, L: int, corsa_attesa: str, down_atteso: bool = True) -> dict:
    """Codici dello strato L scritti da scrivi_codici_su_disco in `cartella`, SOLO se
    la marca fatto_forgia_LL.ok dichiara `corsa=<corsa_attesa>` e il modo del down
    atteso, e nessun .npy e' piu' recente della marca (una corsa diversa li ha
    riscritti dopo). Ritorna {famiglia: (D, Q)} come risultato[L], piu' '_origine'."""
    cartella = Path(cartella)
    marca = cartella / f"fatto_forgia_{L:02d}.ok"
    if not marca.exists():
        raise FileNotFoundError(f"{marca}: marca assente (strato {L} non completato in questa corsa)")
    testo = marca.read_text()
    m_corsa = re.search(r"corsa=(\S+)", testo)
    m_down = re.search(r"down_ternario=(\d)", testo)
    if m_corsa is None or m_corsa[1] != corsa_attesa:
        raise RuntimeError(f"PROVENIENZA: {marca} dichiara corsa={m_corsa[1] if m_corsa else None}, "
                           f"attesa {corsa_attesa!r}")
    if m_down is None or (m_down[1] == "1") != down_atteso:
        raise RuntimeError(f"{marca}: down_ternario={m_down[1] if m_down else None}, atteso {int(down_atteso)}")
    famiglie = FAMIGLIE_GU + ((FAMIGLIA_DW,) if down_atteso else ())
    t_marca = marca.stat().st_mtime
    voce: dict = {}
    for fam in famiglie:
        coppia = []
        for pref in FILE_PER_FAMIGLIA[fam]:
            f = cartella / f"{pref}_{L:02d}.npy"
            if not f.exists():
                raise FileNotFoundError(f"{f}: manca, ma la marca dichiara lo strato completo")
            if f.stat().st_mtime > t_marca + 1.0:
                raise RuntimeError(f"PROVENIENZA: {f} e' piu' recente della marca {marca.name}: "
                                   f"riscritto da un'altra corsa")
            coppia.append(np.load(f))
        _controlla_codici(f"{cartella.name}/{fam}_{L:02d}", *coppia)
        voce[fam] = tuple(coppia)
    voce["_origine"] = f"{cartella}/corsa={corsa_attesa}"
    return voce
