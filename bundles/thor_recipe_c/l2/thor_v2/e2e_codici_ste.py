#!/usr/bin/env python3
"""Fase 2 (progetto `e2e_progetto_fase2_codici.md`): STE per i CODICI ternari, non solo le scale.

26/9/2026. File NUOVO, SOLO CPU (la GPU e' del teacher `e2e-teacher`, ~3h residue al momento in cui
questo file nasce: NON si tocca, NON si usa GPU per nessun motivo qui dentro). Nessun file esistente
e' modificato; si importa in SOLA LETTURA `e2e_codici_scale` (LAYOUT, lettura scale,
`pesi_da_trit_scale` — gia' generico nel dtype/requires_grad del trit, riusato al bit),
`e2e_campi_scala` (`trit_torch`/`scale_torch`, lo STESSO decoder device-agnostico che usa
`e2e_studente.py` per il motore vero) e `e2e_allena` (`regola_k`, in sola lettura, per confrontare
la scala K dei codici con quella delle scale sullo stesso gy giocattolo).

PERCHE' (Angelo, 26/9): la fase 1 (`e2e_allena.py`, gia' verde) allena SOLO le scale fp16 degli
esperti ternari, a codici FERMI. Se l'held-out si ferma presto, un'ipotesi e' che il collo di
bottiglia siano i CODICI, non le scale: questo file costruisce e collauda (solo CPU, oggi) il
motore per verificarlo, cosi' che sia GIA' pronto se/quando servira' — non da inventare dopo un
plateau che si scopre stasera.

Cosa c'e' qui:
1. `CodiceTernarioSTE` — straight-through estimator con dead-zone per il codice ternario: avanti
   round-clamp del latente a {-1,0,1} (il valore che finirebbe nel GGUF), indietro IDENTITA' dentro
   [-clip, clip] e ZERO fuori. E' la stessa clip-STE usata per pesi a pochi bit in Ternary Weight
   Networks (Li & Liu 2016, arXiv:1605.04711 — quantizzano UNA volta, qui si continua ad allenare
   coi codici gia' scelti dalla forgia) e in DoReFa-Net (Zhou et al. 2016, arXiv:1606.06160, §3.3:
   "we use the straight-through estimator... clip the gradient when the input is out of range").
   Letteratura riletta per intero in `e2e_progetto_fase2_codici.md` §3.
2. `pesi_da_latente_scale` — compone lo STE dei codici con `e2e_codici_scale.pesi_da_trit_scale`
   (gia' fa scala x trit + de/rotazione, verificato al bit in T1-T5 di quel file): ZERO
   duplicazione di rotazione/layout, come impone la lezione del 23/9 ("una sola fonte").
3. `_test()` — sui BYTE VERI di Thor-T, strati 0 e 3 (come fa oggi `e2e_test_studente.py`), poche
   esperte (CPU-leggero: un paio di esperti bastano, i byte sono mappati non caricati per intero):
   T1  nessuna regressione — interruttore spento (== usare il trit fisso, come oggi) e interruttore
       acceso ma PRIMA di ogni passo devono dare lo STESSO peso AL BIT;
   T2  gradiente dei codici contro un riferimento indipendente in float64 (la stessa costruzione
       del peso ma SENZA arrotondare, poi differenze finite su quella — come T3 di
       `e2e_codici_scale.py`, spostato dal gruppo (scala) al singolo elemento (codice));
   T3  ciclo minimo (pochi passi, bersaglio giocattolo): mostra che il meccanismo fa scendere la
       loss E misura la quota di codici che il clip azzera passo per passo — NON prova nulla su
       Thor-T vero, quello si misura solo su GPU, quando/se serve;
   T4  rapporto fra |dL/dcodice| e |dL/dscala| sullo STESSO dW: il numero che serve per decidere
       l'eps/K della perdita dei codici (§2.5 del progetto principale) — misurato, non supposto;
   T5  conteggio ESATTO dei codici totali (dai metadati GGUF veri, non da una percentuale scritta a
       mano) e costo di un ottimizzatore ingenuo per-codice: il numero che decide "tutti i codici o
       un sottoinsieme" in §1 del progetto di fase 2.

Uso (CPU confinata — vedi `e2e_progetto_fase2_codici.md` §7 per il comando systemd-run):
    OMP_NUM_THREADS=4 nice -n 10 python3 e2e_codici_ste.py --test
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
sys.path.insert(0, "@@ROOT:models@@/build/llamacpp-tq1-k1024/gguf-py")
sys.path.insert(0, "@@ROOT:forge@@/kernel-rotazione")

from gguf import GGUFReader  # noqa: E402
from tq1_0r_1024_codec import S1024  # noqa: E402  (sola lettura)
import e2e_allena as AL  # noqa: E402  (sola lettura: regola_k)
import e2e_campi_scala as CAMPI  # noqa: E402  (sola lettura: trit_torch, come e2e_studente.py)
import e2e_codici_scale as CS  # noqa: E402  (sola lettura: LAYOUT, scale_da_blocchi, pesi_da_trit_scale)

GGUF_THOR_T = Path(os.getenv("E2E_GGUF", "@@ROOT:models@@/gguf/tony-tern/Thor-v2-ThorT-scor150.gguf"))


# ═══════════════════════════ 1. STE per il codice ternario ═══════════════════════════
class CodiceTernarioSTE(torch.autograd.Function):
    """Avanti: round-clamp del latente w a {-1,0,1} (round-to-nearest-even di torch, poi clamp).
    Indietro: identita' dove |w| <= clip, zero fuori (clip-STE, vedi docstring del modulo). NON
    tocca la scala: si compone con `e2e_codici_scale.pesi_da_trit_scale`, che gia' fa scala x trit
    e l'eventuale de/rotazione, esattamente come oggi col trit fisso."""

    @staticmethod
    def forward(ctx, w: torch.Tensor, clip: float) -> torch.Tensor:
        trit = torch.clamp(torch.round(w), -1.0, 1.0)
        ctx.save_for_backward(w)
        ctx.clip = float(clip)
        return trit.to(w.dtype)

    @staticmethod
    def backward(ctx, g):
        (w,) = ctx.saved_tensors
        passa = (w.abs() <= ctx.clip).to(g.dtype)
        return g * passa, None


def latente_da_trit(trit0: torch.Tensor, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """trit0 (dai byte, valori esatti in {-1,0,1}) -> latente w, foglia con gradiente, inizializzato
    ESATTAMENTE al trit corrente: round(w) == trit0 AL BIT, prima di ogni passo (prova T1)."""
    return trit0.to(dtype).clone().detach().requires_grad_(True)


def pesi_da_latente_scale(w: torch.Tensor, scale: torch.Tensor, tipo: str,
                          S: "torch.Tensor | None" = None, clip: float = 1.0) -> torch.Tensor:
    """Fase 2, ricetta di PRODUZIONE (T1): passa da `CS.pesi_da_trit_scale`, che internamente
    lavora in fp32 (come fara' lo studente vero — e' apposta: T1 deve confrontarsi con la fase 1
    AL BIT, nella stessa precisione)."""
    trit = CodiceTernarioSTE.apply(w, clip)
    return CS.pesi_da_trit_scale(trit, scale, tipo, S)


def pesi_rilassato(w: torch.Tensor, scale: torch.Tensor, tipo: str,
                   S: "torch.Tensor | None" = None) -> torch.Tensor:
    """Come sopra ma SENZA arrotondare (w al posto del trit), stessa precisione fp32 di
    `CS.pesi_da_trit_scale`. Funzione liscia, coincide con la STE dove round(w) == w."""
    return CS.pesi_da_trit_scale(w, scale, tipo, S)


def _pesi_doppia(trit: torch.Tensor, scale: torch.Tensor, tipo: str,
                 S: "torch.Tensor | None" = None) -> torch.Tensor:
    """STESSA algebra di `CS.pesi_da_trit_scale`, ma senza il cast a fp32 interno (che quella
    funzione fa apposta per rispecchiare la produzione — vedi T1). Serve alle prove che hanno
    bisogno di precisione vera (T2 differenze finite, T3 ciclo giocattolo, T4 confronto di scala):
    con fp32 forzato le differenze finite a eps=1e-6 perderebbero cifre. Stesso schema di T3 in
    `e2e_codici_scale.py` (che per lo stesso motivo bypassa `pesi_da_trit_scale` e replica
    l'algebra in doppia)."""
    _, qk, _, k = CS.LAYOUT[tipo]
    W = (trit.reshape(trit.shape[0], k, qk // k) * scale.unsqueeze(-1)).reshape(trit.shape[0], qk)
    if tipo == "TQ1_0R_1024":
        from rotazione_gpu import _RT_1024_torch
        W = _RT_1024_torch(W, S)
    return W


def pesi_da_latente_scale_doppia(w: torch.Tensor, scale: torch.Tensor, tipo: str,
                                 S: "torch.Tensor | None" = None, clip: float = 1.0) -> torch.Tensor:
    return _pesi_doppia(CodiceTernarioSTE.apply(w, clip), scale, tipo, S)


def pesi_rilassato_doppia(w: torch.Tensor, scale: torch.Tensor, tipo: str,
                          S: "torch.Tensor | None" = None) -> torch.Tensor:
    return _pesi_doppia(w, scale, tipo, S)


# ═══════════════════════════ 2. lettura di un pezzo VERO del GGUF ═══════════════════════════
def carica_blocchi(percorso: Path, nome_tensore: str, esperti: tuple) -> tuple:
    """Blocchi FLAT (N, byte_blocco) uint8 di alcuni esperti di UN tensore vero — stessa
    convenzione di T3/T5 in `e2e_codici_scale.py` (l'origine riga/esperto dei blocchi non conta:
    ognuno e' un gruppo indipendente di `pesi_blocco` codici e `k` scale)."""
    r = GGUFReader(str(percorso))
    t = {x.name: x for x in r.tensors}[nome_tensore]
    tipo = t.tensor_type.name
    nb = CS.LAYOUT[tipo][0]
    raw = np.asarray(t.data)[list(esperti)]                       # (n_e, R, NBR), memmap
    blocchi = torch.from_numpy(np.ascontiguousarray(raw)).reshape(-1, nb)
    return blocchi, tipo


def _S(tipo: str) -> "torch.Tensor | None":
    return torch.from_numpy(S1024).to(torch.float64) if tipo == "TQ1_0R_1024" else None


# ═══════════════════════════ 3. T1 — nessuna regressione ═══════════════════════════
def t1_nessuna_regressione(blocchi: torch.Tensor, tipo: str) -> dict:
    S = _S(tipo)
    trit0 = CAMPI.trit_torch(blocchi, tipo).double()
    scale0 = CS.scale_da_blocchi(blocchi, tipo).double()
    rif = CS.pesi_da_trit_scale(trit0, scale0, tipo, S)                    # fase 1: trit FISSO
    w = latente_da_trit(trit0)
    nuovo = pesi_da_latente_scale(w, scale0, tipo, S, clip=1.0)            # fase 2: interruttore ON
    uguale = torch.equal(rif, nuovo.detach())
    # quota di codici gia' AL BORDO rappresentabile (|trit0| == 1): con clip=1,0 esatto, un
    # gradiente che li spinge oltre li blocca al PRIMO passo (§2.2 del progetto di fase 2) — non e'
    # un difetto di T1 (che resta al bit), e' il motivo per cui il clip di allenamento (§2.2, T3)
    # NON deve coincidere col bordo rappresentabile.
    quota_al_bordo = float((trit0.abs() == 1).double().mean())
    return {"N": int(trit0.shape[0]), "qk": int(trit0.shape[1]), "identico_al_bit": bool(uguale),
            "quota_al_bordo_clip1": quota_al_bordo}


# ═══════════════════════════ 4. T2 — gradiente contro un riferimento indipendente ═══════════════════════════
def t2_gradiente_indipendente(blocchi: torch.Tensor, tipo: str, seed: int = 0, n_fd: int = 12) -> dict:
    torch.manual_seed(seed)
    S = _S(tipo)
    trit0 = CAMPI.trit_torch(blocchi, tipo).double()
    scale0 = CS.scale_da_blocchi(blocchi, tipo).double()
    N, qk = trit0.shape
    x = torch.randn(qk, 6, dtype=torch.float64)

    # (a) via STE, al punto operativo trit0 (|trit0| <= 1 ovunque: round(trit0) == trit0)
    w_ste = trit0.clone().detach().requires_grad_(True)
    (pesi_da_latente_scale_doppia(w_ste, scale0, tipo, S, clip=1.0) @ x).pow(2).sum().backward()
    g_ste = w_ste.grad.clone()

    # (b) rilassato: stessa costruzione del peso, SENZA arrotondare — riferimento indipendente
    w_rel = trit0.clone().detach().requires_grad_(True)
    (pesi_rilassato_doppia(w_rel, scale0, tipo, S) @ x).pow(2).sum().backward()
    g_rel = w_rel.grad.clone()

    rel_a_b = float((g_ste - g_rel).abs().max() / g_rel.abs().max())

    # (c) differenze finite SUL RILASSATO (l'unica funzione liscia qui): controlla (b) da solo,
    # cosi' (a)==(b) non e' un pareggio fra due bug uguali. La funzione e' ESATTAMENTE quadratica
    # in ogni singola coordinata w[i,c] (y e' lineare in quella coordinata, L = somma di y^2): la
    # differenza centrata non ha errore di troncamento a NESSUN eps (f''' = 0 identicamente), quindi
    # eps si sceglie grande per allontanarsi dal rumore di cancellazione fp64, non piccolo come per
    # una funzione generica (con eps=1e-6 misurato 1,5-6,7e-6: e' rumore di cancellazione, non un
    # errore della formula — verificato qui allargando eps).
    idx = torch.randperm(N * qk)[:n_fd].tolist()
    eps = 1e-3
    num, rif = torch.zeros(n_fd, dtype=torch.float64), torch.zeros(n_fd, dtype=torch.float64)
    with torch.no_grad():
        for j, flat in enumerate(idx):
            i, c = divmod(flat, qk)
            wp, wm = trit0.clone(), trit0.clone()
            wp[i, c] += eps
            wm[i, c] -= eps
            fp = (pesi_rilassato_doppia(wp, scale0, tipo, S) @ x).pow(2).sum()
            fm = (pesi_rilassato_doppia(wm, scale0, tipo, S) @ x).pow(2).sum()
            num[j] = (fp - fm) / (2 * eps)
            rif[j] = g_rel[i, c]
    rel_b_c = float((rif - num).abs().max() / rif.abs().max().clamp_min(1e-300))
    return {"N": N, "qk": qk, "n_fd": n_fd,
            "err_rel_STE_contro_rilassato": rel_a_b,
            "err_rel_rilassato_contro_differenze_finite": rel_b_c}


# ═══════════════════════════ 5. T3 — ciclo minimo (mostra che scende, misura il clip) ═══════════════════════════
def _ciclo(w0: torch.Tensor, scale0: torch.Tensor, tipo: str, S, x: torch.Tensor, y_target: torch.Tensor,
          clip: float, lr: float, passi: int, ottimizzatore: str) -> dict:
    w = w0.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([w], lr=lr, betas=(0.9, 0.95)) if ottimizzatore == "adam" else None
    l0 = None
    for _ in range(passi):
        y = pesi_da_latente_scale_doppia(w, scale0, tipo, S, clip=clip) @ x
        L = (y - y_target).pow(2).sum()
        (opt.zero_grad() if opt else setattr(w, "grad", None))
        L.backward()
        if l0 is None:
            l0 = float(L)
        (opt.step() if opt else w.data.sub_(lr * w.grad))
    with torch.no_grad():
        n_flip = int((torch.clamp(torch.round(w), -1.0, 1.0) != w0).sum())
        quota_clip = float((w.abs() > clip).double().mean())
    return {"ottimizzatore": ottimizzatore, "lr": lr, "passi": passi, "loss_iniziale": l0,
            "loss_finale": float(L), "flip": n_flip, "quota_oltre_clip": quota_clip, "scende": float(L) < l0}


def t3_ciclo_minimo(blocchi: torch.Tensor, tipo: str, blocchi_bersaglio: torch.Tensor,
                    clip: float = 1.5, seed: int = 0) -> dict:
    """Bersaglio: STESSA scala del codice di partenza, ma i trit di un ALTRO pezzo VERO del GGUF
    (`blocchi_bersaglio`, stessa forma, un'altra esperta) — un miglioramento grande e raggiungibile
    con dei flip veri, non un rumore artificiale attorno al valore di partenza (che darebbe
    all'ottimizzatore quasi nulla da correggere, essendo il minimo gia' vicinissimo a trit0).
    Confronta SGD (senza normalizzazione per parametro) con Adam — lo stesso ottimizzatore/beta
    che la fase 1 usa per theta — sullo STESSO bersaglio: la differenza e' la scoperta di T3."""
    torch.manual_seed(seed)
    S = _S(tipo)
    trit0 = CAMPI.trit_torch(blocchi, tipo).double()
    trit_bersaglio = CAMPI.trit_torch(blocchi_bersaglio, tipo).double()
    scale0 = CS.scale_da_blocchi(blocchi, tipo).double()
    x = torch.randn(trit0.shape[1], 8, dtype=torch.float64)
    with torch.no_grad():
        y_target = pesi_rilassato_doppia(trit_bersaglio, scale0, tipo, S) @ x
    quota_gia_uguale = float((trit0 == trit_bersaglio).double().mean())
    sgd = _ciclo(trit0, scale0, tipo, S, x, y_target, clip, lr=0.5, passi=60, ottimizzatore="sgd")
    adam = _ciclo(trit0, scale0, tipo, S, x, y_target, clip, lr=0.03, passi=60, ottimizzatore="adam")
    return {"quota_gia_uguale_al_bersaglio": quota_gia_uguale, "n_codici": int(trit0.numel()),
            "sgd": sgd, "adam": adam}


# ═══════════════════════════ 6. T4 — quanto e' piu' piccolo il gradiente di un codice ═══════════════════════════
def t4_confronto_scala_codice(blocchi: torch.Tensor, tipo: str, seed: int = 0) -> dict:
    """dL/dcodice e dL/dtheta (scala, stessa parametrizzazione §2.2: d_eff = d0 exp(theta)) dallo
    STESSO passo indietro: la differenza e' SOLO che dL/dtheta somma su un gruppo di `gs` codici e
    dL/dcodice no. Misura, non suppone, se la K scelta per le scale basterebbe anche per i codici."""
    torch.manual_seed(seed)
    S = _S(tipo)
    trit0 = CAMPI.trit_torch(blocchi, tipo).double()
    d0 = CS.scale_da_blocchi(blocchi, tipo).double()
    N, qk = trit0.shape
    gs = qk // d0.shape[1]
    x = torch.randn(qk, 6, dtype=torch.float64)

    w = trit0.clone().detach().requires_grad_(True)
    theta = torch.zeros_like(d0, requires_grad=True)
    d_eff = d0 * torch.exp(theta)
    trit = CodiceTernarioSTE.apply(w, 1.0)
    y = _pesi_doppia(trit, d_eff, tipo, S) @ x
    (y ** 2).sum().backward()

    g_w = w.grad.abs().reshape(-1)
    g_t = theta.grad.abs().reshape(-1)
    g_w, g_t = g_w[g_w > 0], g_t[g_t > 0]

    def q01(v: torch.Tensor) -> float:
        return float(torch.kthvalue(v, max(1, math.ceil(0.01 * v.numel()))).values)

    r_codice, r_scala = AL.regola_k(q01(g_w)), AL.regola_k(q01(g_t))
    return {"gruppo_gs": gs, "n_codici_vivi": int(g_w.numel()), "n_scale_vive": int(g_t.numel()),
            "mediana_grad_codice": float(g_w.median()), "mediana_grad_scala": float(g_t.median()),
            "rapporto_mediane_scala_su_codice": float(g_t.median() / g_w.median()),
            "radice_gs_attesa": math.sqrt(gs),
            "q01_codice": q01(g_w), "q01_scala": q01(g_t),
            "K_codice": r_codice["K"], "K_scala": r_scala["K"], "stesso_K": r_codice["K"] == r_scala["K"]}


# ═══════════════════════════ 7. T5 — conto ESATTO e costo di "tutti i codici" ═══════════════════════════
def t5_costo_tutti_i_codici(percorso: Path) -> dict:
    """Conta i codici ternari VERI (tipo in `CS.LAYOUT`) su tutti gli strati. Uno strato puo'
    esistere (`blk.L.ffn_*_exps.weight` presente) senza essere ternario: `blk.40` (MTP) resta nel
    quant originale (§2.1 del progetto principale, "identico al byte") — si conta come strato
    presente ma non contribuisce a codici/scale, non e' un errore da sollevare."""
    r = GGUFReader(str(percorso))
    tens = {t.name: t for t in r.tensors}
    tot_codici = tot_scale = 0
    n_layer = n_layer_ternari = 0
    for L in range(64):
        famiglie_presenti = [f"blk.{L}.{fam}.weight" for fam in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps")
                             if f"blk.{L}.{fam}.weight" in tens]
        if not famiglie_presenti:
            if n_layer > 0:
                break
            continue
        n_layer += 1
        ternario_qui = False
        for nome in famiglie_presenti:
            t = tens[nome]
            tipo = t.tensor_type.name
            if tipo not in CS.LAYOUT:
                continue                                  # es. blk.40 MTP: quant originale, non ternario
            ternario_qui = True
            nb, qk, off, k = CS.LAYOUT[tipo]
            n_blocchi = int(t.n_bytes) // nb
            tot_codici += n_blocchi * qk
            tot_scale += n_blocchi * k
        n_layer_ternari += int(ternario_qui)
    b_adam, b_sgd = 16, 8    # fp32: (latente+grad+m+v) contro (latente persistente+grad transitorio)
    gib = 2 ** 30
    per_layer = tot_codici / n_layer_ternari
    return {"gguf": str(percorso), "n_layer": n_layer, "n_layer_ternari": n_layer_ternari,
            "codici_totali": tot_codici, "scale_totali": tot_scale, "gruppo_medio": tot_codici / tot_scale,
            "GiB_adam_fp32_tutti_i_codici": tot_codici * b_adam / gib,
            "GiB_sgd_fp32_tutti_i_codici": tot_codici * b_sgd / gib,
            "codici_per_layer_3_famiglie": per_layer,
            "GiB_adam_fp32_un_layer": per_layer * b_adam / gib,
            "GiB_sgd_fp32_un_layer": per_layer * b_sgd / gib,
            "GiB_sgd_fp32_due_layer_0_3": 2 * per_layer * b_sgd / gib}


# ═══════════════════════════════════════ test ═══════════════════════════════════════
def _test() -> bool:
    torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "4")))
    ok = True
    fonti = [("blk.0.ffn_gate_exps.weight", (0,), (191,)), ("blk.3.ffn_down_exps.weight", (0,), (191,))]
    print(f"GGUF: {GGUF_THOR_T}")
    for nome, esperti, esperti_bersaglio in fonti:
        blocchi, tipo = carica_blocchi(GGUF_THOR_T, nome, esperti)
        blocchi_b, _ = carica_blocchi(GGUF_THOR_T, nome, esperti_bersaglio)
        print(f"\n── {nome} ({tipo}, esperti {esperti}, {blocchi.shape[0]} blocchi; "
              f"bersaglio T3: esperti {esperti_bersaglio}) ──")

        r1 = t1_nessuna_regressione(blocchi, tipo)
        ok &= r1["identico_al_bit"]
        print(f"[T1] interruttore ON prima di ogni passo == fase 1 al bit: {r1['identico_al_bit']} "
              f"({r1['N']} blocchi x {r1['qk']} codici) · quota gia' al bordo ±1 (clip=1,0 li "
              f"bloccherebbe al primo passo): {r1['quota_al_bordo_clip1']:.4f}")

        r2 = t2_gradiente_indipendente(blocchi, tipo)
        soglia2 = 1e-9
        passa2 = r2["err_rel_STE_contro_rilassato"] < soglia2 and r2["err_rel_rilassato_contro_differenze_finite"] < 1e-6
        ok &= passa2
        print(f"[T2] STE contro rilassato: err_rel {r2['err_rel_STE_contro_rilassato']:.3e} · "
              f"rilassato contro differenze finite ({r2['n_fd']} codici): "
              f"err_rel {r2['err_rel_rilassato_contro_differenze_finite']:.3e} · passa: {passa2}")

        r3 = t3_ciclo_minimo(blocchi, tipo, blocchi_b)
        ok &= r3["sgd"]["scende"]
        print(f"[T3] bersaglio: un'altra esperta vera, {r3['quota_gia_uguale_al_bersaglio']:.3f} dei codici "
              f"gia' uguali all'inizio · {r3['n_codici']:,} codici")
        for tag in ("sgd", "adam"):
            e = r3[tag]
            print(f"  [{tag:4s}] lr {e['lr']}, {e['passi']} passi: loss {e['loss_iniziale']:.3f} -> "
                  f"{e['loss_finale']:.3f} (scende: {e['scende']}) · flip {e['flip']:,}/{r3['n_codici']:,} · "
                  f"oltre il clip: {e['quota_oltre_clip']:.4f}")
        print("  (NON prova nulla su Thor-T vero: solo GPU, con teacher e corpus reali — qui prova solo "
              "che il meccanismo SGD riduce una perdita giocattolo, e che Adam con le beta di theta no)")

        r4 = t4_confronto_scala_codice(blocchi, tipo)
        print(f"[T4] gruppo gs={r4['gruppo_gs']} (radice attesa {r4['radice_gs_attesa']:.2f}): "
              f"mediana |dL/dscala| / mediana |dL/dcodice| = {r4['rapporto_mediane_scala_su_codice']:.2f} · "
              f"K_scala=2^{math.log2(r4['K_scala']):.0f} K_codice=2^{math.log2(r4['K_codice']):.0f} "
              f"(stesso K: {r4['stesso_K']})")

    r5 = t5_costo_tutti_i_codici(GGUF_THOR_T)
    print(f"\n[T5] {r5['n_layer']} strati ({r5['n_layer_ternari']} ternari), "
          f"{r5['codici_totali']:,} codici totali "
          f"({r5['codici_totali'] / 1e9:.3f} G), {r5['scale_totali']:,} scale (gruppo medio "
          f"{r5['gruppo_medio']:.1f}) · Adam fp32 su TUTTI: {r5['GiB_adam_fp32_tutti_i_codici']:.1f} GiB · "
          f"SGD fp32 su TUTTI: {r5['GiB_sgd_fp32_tutti_i_codici']:.1f} GiB · "
          f"un solo strato (SGD): {r5['GiB_sgd_fp32_un_layer']:.2f} GiB · "
          f"strati 0+3 (SGD): {r5['GiB_sgd_fp32_due_layer_0_3']:.2f} GiB")

    print("\nESITO:", "TUTTO VERDE" if ok else "⛔ QUALCOSA NON TORNA")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    if a.test:
        sys.exit(0 if _test() else 1)
    ap.print_help()
