#!/usr/bin/env python3
"""FORGIA SchurQuant per-esperto — da W* (bersaglio float) a ternario reale.

⭐ Perche' (22/9 20:10, Angelo: «devi arrivare tutto in parallelo, tutto pronto
in parallelo»). La cattura produce W*: il bersaglio in virgola mobile, verificato
matematicamente identico al riferimento (2,3e-15) e vincente 4352/4352 esperti
sull'obiettivo J(W) — ma non e' ancora un peso che il GGUF puo' spedire. Questo
file chiude l'ultimo anello, riusando ESATTAMENTE la catena gia' verificata
16/16 in schurquant_esperto_reale.py (righe 141-153), mai cambiata di una riga:

    Gd  = smorza(G)                         (gia' fatto in cattura: G salvata)
    Wrot = ruota_righe_1024(W*)             ruota il BERSAGLIO
    Hrot = ruota_hessiana_1024(G)           ruota la STESSA G della cattura
    Q0,D0 = round-to-nearest (soglia 0.5*alfa0)   GPTQ-a-un-piano semplificato
    D,Q  = raffina_coorddesc(Wrot,Q0,D0,Hrot, n_round=4)   coordinate descent

⛔ Nessuna Hessiana nuova da catturare: Hrot viene da G, che la cattura ha
GIA' salvato per ogni strato. Zero forward pass qui — solo algebra e il
solver, che e' economico (16,4 min misurati per 80 tensori simili in v10).

Uscita per strato L: Dg_{L:02d}.npy/Qg_{L:02d}.npy (gate), Du_{L:02d}.npy/
Qu_{L:02d}.npy (up) — nel contratto di pack_da_d_q, pronti per l'esportatore.
Ripresa: fatto_forgia_{L:02d}.ok, marcata SOLO dopo scrittura completa.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getenv("GGUF_PY_DIR", "@@ROOT:models@@/build/llamacpp-tq1-k1024/gguf-py"))

from gptq_1024_lib import GROUP, QK, ruota_righe_1024, ruota_hessiana_1024, _RT_1024_np  # noqa: E402
from gptq_1024_coorddesc import raffina_coorddesc, raffina_trit_sweep_batch_blocchi  # noqa: E402
from gptq_1024_refine import raffina_scale_joint, raffina_scale_joint_batch  # noqa: E402
from kernel_blocco_fuso import raffina_trit_sweep_batch_fuso  # noqa: E402
from schurquant_riconduzione import SMORZ  # noqa: E402


@torch.no_grad()
def smorza_batch(G: torch.Tensor) -> torch.Tensor:
    """⛔⛔⛔ 23/9 14:20 — FIX del difetto trovato dalla revisione avversariale:
    forgia_batch usava G GREZZA (da Gq_L.npy) per rotazione+coordinate descent,
    MAI smorzata — a differenza del precedente PROVATO funzionante
    (forgia_thor_v11_rotexps.py, che chiama sempre prepara_hessiana(smorzamento=0.01)
    prima di GPTQ/refine). Misurato sui dati veri: 1497/10240 slot esperto-strato
    (14,6%) hanno meno di 2048 token instradati -> Hessiana strutturalmente
    singolare (caso peggiore: strato 38 esperto 159, rango 9/2048, autovalore
    NEGATIVO -3.47e-05). Con G quasi-singolare il coordinate descent ha liberta'
    totale nelle direzioni che G non vede: errore pesato-G "perfetto" (-3.1e-09)
    ma errore vero L2 85% — la metrica di verifica e' circolare e cieca proprio
    su questo guasto.

    Stessa IDENTICA idea di schurquant_riconduzione.smorza (SMORZ=0.01, uguale
    a gptq_1024_lib.prepara_hessiana), batchata sui 256 esperti: diagonali morte
    -> 1, poi + SMORZ*mean(diag)*I. Verificato (isola, prima di applicare qui):
    0 diff contro smorza() chiamata esperto-per-esperto su dati veri."""
    Gd = G.clone()
    E, n, _ = Gd.shape
    diag = torch.diagonal(Gd, dim1=-2, dim2=-1)
    morti = diag == 0
    if morti.any():
        diag[morti] = 1.0
    media = diag.mean(dim=-1)
    eye = torch.eye(n, dtype=Gd.dtype, device=Gd.device).unsqueeze(0)
    Gd = Gd + eye * (SMORZ * media).view(E, 1, 1)
    return Gd

OUT = Path(os.getenv("FINESTRA_SCHUR_OUT", "@@ROOT:models@@/gguf/tony-tern/schur-esperti"))
N_ROUND = int(os.getenv("THOR_FORGIA_ROUND", "4"))  # 4, come il riferimento verificato
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def q0_iniziale(Wrot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Round-to-nearest a soglia 0.5*alfa — IDENTICO a schurquant_esperto_reale.py
    righe 146-149. Non e' GPTQ vero (nessuna compensazione d'errore): e' il punto
    di partenza che raffina_coorddesc raffina; il riferimento verificato parte
    da qui, non da un GPTQ completo."""
    out_rows, in_feat = Wrot.shape
    ng = in_feat // GROUP
    alfa0 = np.abs(Wrot).reshape(out_rows, ng, GROUP).mean(-1)
    Q0 = np.sign(Wrot).reshape(out_rows, ng, GROUP)
    Q0 = np.where(np.abs(Wrot).reshape(out_rows, ng, GROUP) > 0.5 * alfa0[:, :, None], Q0, 0)
    return Q0.reshape(out_rows, in_feat), alfa0


def q0_iniziale_batch_gpu(Wrot_np: np.ndarray, device: str = "cuda") -> tuple[np.ndarray, np.ndarray]:
    """⭐ 23/9 09:00 — versione BATCHATA su GPU di q0_iniziale, per eliminare
    l'ultimo ciclo Python per-esperto rimasto in forgia_batch (256 chiamate
    numpy su CPU). Stessa formula, IDENTICA (mean/sign/where sono elementwise
    o riduzioni sull'ultimo asse, indifferenti a una dimensione batch in piu').
    Wrot_np: (E, out_rows, in_feat). Ritorna (Q0, alfa0) come q0_iniziale,
    con una dimensione E in testa."""
    E, out_rows, in_feat = Wrot_np.shape
    ng = in_feat // GROUP
    W = torch.from_numpy(Wrot_np).to(device, dtype=torch.float32)
    Wg = W.reshape(E, out_rows, ng, GROUP)
    alfa0 = Wg.abs().mean(-1)
    Q0 = torch.sign(Wg)
    Q0 = torch.where(Wg.abs() > 0.5 * alfa0.unsqueeze(-1), Q0, torch.zeros_like(Q0))
    return (Q0.reshape(E, out_rows, in_feat).cpu().numpy(),
            alfa0.cpu().numpy())


# ⭐⭐⭐⭐⭐⭐ 23/9 08:00 — CAUSA VERA TROVATA (non hardware): un pareggio a
# scala zero mai gestito. Quando la scala di un gruppo (`dj`) e' ESATTAMENTE
# 0.0, i tre candidati trit {-1,0,1} danno lo STESSO identico residuo/loss
# (0*qualsiasi=0): un pareggio matematico vero. Il tie-break (`<` stretto)
# teneva sempre il PRIMO candidato provato (-1.0) a prescindere da q_cur —
# un artefatto arbitrario, dipendente da rumore floating-point accumulato
# nei round precedenti in modo leggermente diverso fra il percorso
# sequenziale e quello batchato. Prova diretta: OGNI divergenza mai
# osservata (stanotte, in tutte le taglie B=16..256) cadeva ESATTAMENTE
# su posizioni con `alfa[riga,gruppo] == 0.0` esatto. Non era non-
# determinismo GPU (quello era gia' stato escluso con prove dirette) — era
# un caso limite mai scritto. Fix in `gptq_1024_coorddesc.py`: con dj==0
# non si tocca q_cur/e_j_old (nessun trit puo' cambiare la ricostruzione
# comunque). Verificato: 0/16.777.216 diversi su 16 esperti veri, poi
# confermato a scala piena (256 esperti, pipeline completa n_round=4).
#
# Rotazione su GPU (`rotazione_gpu.py`): via due applicazioni della FWHT
# (pura add/sub elementwise, MAI una riduzione) invece del matmul denso
# 2048x2048 — stesso risultato (diff ~1e-14 float64), determinismo
# verificato cross-processo (0.0 diff), 4.4x piu' veloce su 256 esperti.
def forgia_batch(W_star: np.ndarray, G: np.ndarray, n_round: int = N_ROUND) -> tuple[np.ndarray, np.ndarray]:
    """Versione BATCH vera: tutti gli esperti di UN tensore insieme, rotazione
    e coordinate descent entrambi su GPU. W_star: (E, out, in) · G: (E, in, in).
    Ritorna D (E, out, ng), Q (E, out, in)."""
    from rotazione_gpu import ruota_righe_batch_gpu, ruota_hessiana_batch_gpu

    E, out_rows, in_feat = W_star.shape
    ng = in_feat // GROUP
    G_t = smorza_batch(torch.from_numpy(G.astype(np.float32)).to(DEV)).cpu().numpy().astype(np.float32)
    Wrot = ruota_righe_batch_gpu(W_star.astype(np.float32), device=DEV).astype(np.float32)
    Hrot = ruota_hessiana_batch_gpu(G_t, device=DEV)
    del G_t  # ⛔ 23/9 16:45 — 4,29 GiB inutili per tutta la funzione, anti-swap-thrashing
    # ⭐ 23/9 16:1x — Angelo: "cura la pipeline, massimizza il parallelismo".
    # Misurato in produzione (cattura sequenziale): dopo la passata GPU
    # (97% busy), il processo cadeva a GPU 0%/CPU 99% per minuti — il ciclo
    # Python per-esperto qui sotto, mai sostituito dalla versione batchata
    # GPU gia' scritta e verificata stanotte (0 diff), nonostante il
    # commento originale la segnalasse come pronta ma non collegata.
    Q0_np, D0 = q0_iniziale_batch_gpu(Wrot, device=DEV)
    Q0 = Q0_np.astype(np.int8)
    D0 = D0.astype(np.float32)

    Wrot_t = torch.from_numpy(Wrot).to(DEV)
    Q_t = torch.from_numpy(Q0).to(DEV)
    D_t = torch.from_numpy(D0).to(DEV)
    Hrot_t = torch.from_numpy(Hrot).to(DEV, dtype=torch.float32)
    del Hrot  # ⛔ 23/9 16:45 — 8,59 GiB float64 su CPU, inutili dopo la conversione GPU
    for r in range(n_round):
        # ⭐⭐⭐⭐⭐⭐⭐ 23/9 11:20 — kernel Triton fuso: dopo il fix a blocchi
        # (sopra) la banda non era piu' il collo di bottiglia (~6 GB/s su un
        # picco di 210-260 — crollata da 4,4 TB/sweep a ~310 GB/sweep), il
        # vero limite era il NUMERO DI LANCI GPU (~41.000/sweep, overhead
        # fisso su tensori piccoli). Il kernel fa un blocco INTERO in un solo
        # lancio (da ~320 lanci/blocco a 1): misurato **6,99x** su 256 esperti
        # veri, verificato 0/268.435.456 diversi contro la versione a blocchi
        # (gia' verificata) e ripetibile (0 diff fra due lanci). Stessa
        # identica matematica, inclusa la guardia dj==0.
        Q_t = raffina_trit_sweep_batch_fuso(Wrot_t, Q_t, D_t, Hrot_t, device=DEV, blocksize=16).to(DEV)
        D_t = raffina_scale_joint_batch(Wrot_t, Q_t.float(), Hrot_t, device=DEV).to(DEV)

    return D_t.cpu().numpy(), Q_t.cpu().numpy()


def forgia_tensore(W_star_e: np.ndarray, G_e: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Un esperto, un tensore (gate O up, presi separatamente da W* fuso).
    Ritorna (D, Q) nel contratto di gptq_1024_lib.pack_da_d_q."""
    Wrot = ruota_righe_1024(W_star_e.astype(np.float32))
    Hrot = ruota_hessiana_1024(G_e.astype(np.float32))
    Q0, alfa0 = q0_iniziale(Wrot)
    D, Q = raffina_coorddesc(
        torch.from_numpy(Wrot).to(DEV),
        torch.from_numpy(Q0.astype(np.int8)).to(DEV),
        torch.from_numpy(alfa0.astype(np.float32)).to(DEV),
        torch.from_numpy(Hrot).to(DEV),
        n_round=N_ROUND, device=DEV, log_ogni=False,
    )
    return D.cpu().numpy(), Q.cpu().numpy()


def main():
    strati = sorted(int(p.stem.split("_")[1]) for p in OUT.glob("Wstar_*.npy"))
    log(f"strati da forgiare: {len(strati)} su {DEV}, n_round={N_ROUND} (forgia_batch: rotazione GPU + coordinate descent batchato)")
    for L in strati:
        marca = OUT / f"fatto_forgia_{L:02d}.ok"
        if marca.exists():
            log(f"strato {L}: gia' forgiato, salto")
            continue
        t0 = time.time()
        Ws = np.load(OUT / f"Wstar_{L:02d}.npy").astype(np.float32)   # (256, out, in) = gate+up fusi
        G = np.load(OUT / f"Gq_{L:02d}.npy").astype(np.float32)        # (256, in, in)
        n_esp, out_tot, in_feat = Ws.shape
        meta = out_tot // 2  # gate e up hanno la stessa taglia, fusi sull'asse out

        Dg, Qg = forgia_batch(Ws[:, :meta], G, n_round=N_ROUND)
        Du, Qu = forgia_batch(Ws[:, meta:], G, n_round=N_ROUND)

        np.save(OUT / f"Dg_{L:02d}.npy", Dg)
        np.save(OUT / f"Qg_{L:02d}.npy", Qg)
        np.save(OUT / f"Du_{L:02d}.npy", Du)
        np.save(OUT / f"Qu_{L:02d}.npy", Qu)
        marca.write_text(f"{time.strftime('%F %T')} {n_esp} esperti\n")
        log(f"strato {L}: forgiato ({n_esp} esperti, gate+up) in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
