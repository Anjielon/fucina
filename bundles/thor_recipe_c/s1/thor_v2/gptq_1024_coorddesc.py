"""Raffinamento congiunto scala+trit per TQ1_0R_1024, coordinate descent VERA
(21/9 sera) — al posto del trainer a gradiente (`gptq_1024_softern.py`, 3
tentativi tutti PEGGIO di GPTQ, vedi [[feedback_softern_trainer_non_funziona]]).

Perche' questa via e' piu' solida: ogni passo e' un OTTIMO ESATTO in forma
chiusa su UNA variabile alla volta (scala di gruppo, o trit di un peso), con
tutte le altre fissate — la loss pesata-H NON PUO' crescere ad ogni passo
(e' un minimo su un insieme che contiene lo stato precedente). Nessuna
sharpness da tarare, nessun ottimizzatore, nessuna approssimazione di
gradiente (STE) che puo' avere il segno sbagliato.

Due mosse, alternate a convergenza:
1. `raffina_scale_joint` (gia' esistente, 20/9, PROVATA: -2.7% PPL) — scala
   ottima per gruppo, trit fissi, minimi quadrati pesati da H su TUTTI i
   gruppi della riga insieme (non sequenziale come GPTQ).
2. `raffina_trit_sweep` (NUOVA, qui) — per ogni posizione j, col residuo
   corrente e la scala corrente, prova le 3 uscite {-1,0,1} e tiene quella
   che minimizza ESATTAMENTE il contributo locale alla forma quadratica
   (aggiornamento sequenziale del residuo, stesso trucco algebrico di
   `gptq_1024_lib.gptq_1024` ma ripetuto in PIU' sweep, non uno solo, e sui
   trit GIA' scelti — puo' correggerne le scelte greedy di GPTQ/RTN).
"""
from __future__ import annotations
import time

import torch

from gptq_1024_lib import GROUP
from gptq_1024_refine import raffina_scale_joint


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


@torch.no_grad()
def raffina_trit_sweep(Wrot: torch.Tensor, Q: torch.Tensor, D: torch.Tensor,
                        Hrot: torch.Tensor, device: str = "cuda") -> torch.Tensor:
    """Un passaggio SEQUENZIALE su tutte le posizioni: per ognuna, sceglie il
    trit ESATTAMENTE ottimo dati residuo corrente e scala corrente. Non puo'
    peggiorare la loss pesata-H (ogni scelta include l'opzione di NON cambiare).

    Wrot: (rows, n_in) target. Q: (rows, n_in) trit correnti {-1,0,1}.
    D: (rows, ng) scala corrente per gruppo. Hrot: (n_in, n_in) Hessiana
    ROTATA vera (non Cholesky). Ritorna Q_new (rows, n_in) int8.
    """
    rows, n_in = Wrot.shape
    ng = n_in // GROUP
    W = Wrot.to(device=device, dtype=torch.float64)
    Qc = Q.to(device=device, dtype=torch.float64).clone()
    Dg = D.to(device=device, dtype=torch.float64)  # (rows, ng)
    H = torch.as_tensor(Hrot, dtype=torch.float64, device=device)

    d_per_pos = Dg.repeat_interleave(GROUP, dim=1)  # (rows, n_in) — scala per posizione
    Wq = Qc * d_per_pos
    e = W - Wq  # residuo corrente (rows, n_in)
    g = e @ H  # (rows, n_in) — g[:,k] = (H e)_k, mantenuto incrementalmente sotto
    Hdiag = torch.diagonal(H).clamp(min=1e-12)

    for j in range(n_in):
        dj = d_per_pos[:, j]
        q_cur = Qc[:, j]
        e_j_old = e[:, j]
        g_j_old = g[:, j]
        w_j = e_j_old + dj * q_cur  # = W_j esatto (disfa il contributo corrente di j)
        # obiettivo LOCALE esatto (forma quadratica completa, non solo la diagonale):
        # L_rel(e'_j) = e'_j^2 * H_jj + 2*e'_j*(g_j_old - H_jj*e_j_old)
        lin_coeff = g_j_old - Hdiag[j] * e_j_old
        best_val = None
        best_q = q_cur.clone()
        best_e = e_j_old.clone()
        for cand in (-1.0, 0.0, 1.0):
            e_cand = w_j - dj * cand
            val = e_cand * e_cand * Hdiag[j] + 2.0 * e_cand * lin_coeff
            if best_val is None:
                best_val = val
                best_q = torch.full_like(q_cur, cand)
                best_e = e_cand
            else:
                migliora = val < best_val
                best_val = torch.where(migliora, val, best_val)
                best_q = torch.where(migliora, torch.full_like(q_cur, cand), best_q)
                best_e = torch.where(migliora, e_cand, best_e)
        # ⛔⛔⛔ 23/9 08:00 — dj==0 e' un pareggio VERO fra i 3 candidati (0*qualsiasi=0,
        # stesso e_cand, stesso val per costruzione): senza guardia, il tie-break
        # (`<` stretto) tiene sempre il PRIMO candidato provato (-1.0), a
        # prescindere da q_cur — un artefatto arbitrario che dipende da rumore
        # floating-point accumulato nei round precedenti, diverso fra il percorso
        # sequenziale e quello batchato (verificato: ogni divergenza trovata fra
        # raffina_trit_sweep e raffina_trit_sweep_batch cade ESATTAMENTE su
        # posizioni con scala di gruppo 0.0 esatta). Con dj=0 nessun trit cambia
        # la ricostruzione: la scelta giusta e' NON toccare q_cur/e_j_old.
        dj_zero = (dj == 0)
        if dj_zero.any():
            best_q = torch.where(dj_zero, q_cur, best_q)
            best_e = torch.where(dj_zero, e_j_old, best_e)
        delta_e = best_e - e_j_old
        if torch.count_nonzero(delta_e) == 0:
            continue
        # aggiornamento incrementale ESATTO (H e' lineare in e): g += delta_e * H[:,j]
        g += delta_e.unsqueeze(1) * H[j, :].unsqueeze(0)
        e[:, j] = best_e
        Qc[:, j] = best_q

    return Qc.to(torch.int8).cpu()


@torch.no_grad()
def raffina_trit_sweep_batch(Wrot: torch.Tensor, Q: torch.Tensor, D: torch.Tensor,
                              Hrot: torch.Tensor, device: str = "cuda") -> torch.Tensor:
    """⭐⭐⭐ 22/9 20:15 — STESSO algoritmo di raffina_trit_sweep, con una
    dimensione BATCH in piu' per una Hessiana DIVERSA per elemento del batch
    (un esperto ciascuno), invece di UNA condivisa da tutte le righe.

    Perche' esiste: raffina_trit_sweep fa un ciclo Python SEQUENZIALE su
    n_in=2048 posizioni (necessario: ogni scelta dipende dal residuo
    aggiornato dalla precedente). v10 lo fa girare UNA volta per 256 esperti
    insieme perche' condividono una H per-strato; qui ogni esperto ha la
    PROPRIA G/Hrot (misurato: 131s/esperto invece di 12s/256-esperti-insieme
    — 256x l'overhead, non il calcolo). Questa versione fa il MEDESIMO ciclo
    di 2048 iterazioni UNA sola volta, con tutti gli esperti come batch.

    Wrot,Q,D: (B, rows, n_in)/(B, rows, ng). Hrot: (B, n_in, n_in) — UNA
    Hessiana per elemento del batch. Ritorna Q_new (B, rows, n_in) int8.
    """
    B, rows, n_in = Wrot.shape
    ng = n_in // GROUP
    W = Wrot.to(device=device, dtype=torch.float64)
    Qc = Q.to(device=device, dtype=torch.float64).clone()
    Dg = D.to(device=device, dtype=torch.float64)
    H = torch.as_tensor(Hrot, dtype=torch.float64, device=device)
    assert H.shape == (B, n_in, n_in)

    d_per_pos = Dg.repeat_interleave(GROUP, dim=2)  # (B, rows, n_in)
    Wq = Qc * d_per_pos
    e = W - Wq
    # ⛔⛔⛔ 22/9 20:30 — ISOLATO IL PUNTO ESATTO della divergenza (Angelo:
    # «zero compromessi, deve funzionare»): con batch=1, `einsum` batchato e
    # matmul singolo danno 0.000e+00 di differenza; con batch>1 no — cuBLAS
    # sceglie un kernel di riduzione DIVERSO per un batch di matmul, che
    # somma i termini in un ordine diverso (float non associativo).
    # TUTTO il resto del ciclo (confronto elementwise dei 3 candidati,
    # aggiornamento incrementale g += delta_e ⊗ H[:,j,:]) NON fa riduzioni:
    # e' broadcast/elementwise puro, quindi deterministico e identico al
    # riferimento SE il punto di partenza g combacia esattamente. La cura:
    # questa UNICA matmul per-esperto (economica, 256 chiamate ma non nel
    # ciclo costoso da 2048 iterazioni), poi tutto il resto batchato.
    g = torch.stack([e[b] @ H[b] for b in range(H.shape[0])])  # (B, rows, n_in)
    Hdiag = torch.diagonal(H, dim1=-2, dim2=-1).clamp(min=1e-12)  # (B, n_in)

    for j in range(n_in):
        dj = d_per_pos[:, :, j]          # (B, rows)
        q_cur = Qc[:, :, j]
        e_j_old = e[:, :, j]
        g_j_old = g[:, :, j]
        w_j = e_j_old + dj * q_cur
        Hdiag_j = Hdiag[:, j].unsqueeze(1)  # (B,1) -> broadcast su rows
        lin_coeff = g_j_old - Hdiag_j * e_j_old
        best_val = None
        best_q = q_cur.clone()
        best_e = e_j_old.clone()
        for cand in (-1.0, 0.0, 1.0):
            e_cand = w_j - dj * cand
            val = e_cand * e_cand * Hdiag_j + 2.0 * e_cand * lin_coeff
            if best_val is None:
                best_val = val
                best_q = torch.full_like(q_cur, cand)
                best_e = e_cand
            else:
                migliora = val < best_val
                best_val = torch.where(migliora, val, best_val)
                best_q = torch.where(migliora, torch.full_like(q_cur, cand), best_q)
                best_e = torch.where(migliora, e_cand, best_e)
        # ⛔⛔⛔ 23/9 08:00 — STESSA guardia della versione non-batch: dj==0 e'
        # un pareggio VERO fra i 3 candidati, e senza guardia il tie-break tiene
        # sempre il primo candidato (-1.0) a prescindere da q_cur — causa VERA
        # (trovata stanotte, non non-determinismo GPU) di ogni divergenza mai
        # osservata fra questa funzione e raffina_trit_sweep: cadeva SEMPRE su
        # gruppi a scala 0.0 esatta.
        dj_zero = (dj == 0)
        if dj_zero.any():
            best_q = torch.where(dj_zero, q_cur, best_q)
            best_e = torch.where(dj_zero, e_j_old, best_e)
        delta_e = best_e - e_j_old
        # ⛔ NIENTE early-continue qui: con un batch di esperti, "tutti fermi
        # insieme" e' raro — l'aggiornamento su delta_e=0 e' comunque un no-op
        # matematico (g += 0), quindi ometterlo non serve e complicherebbe
        # la vettorizzazione senza guadagno reale.
        g += delta_e.unsqueeze(-1) * H[:, j, :].unsqueeze(1)  # (B,rows,1)*(B,1,n_in)
        e[:, :, j] = best_e
        Qc[:, :, j] = best_q

    return Qc.to(torch.int8).cpu()


@torch.no_grad()
def raffina_trit_sweep_batch_blocchi(Wrot: torch.Tensor, Q: torch.Tensor, D: torch.Tensor,
                                      Hrot: torch.Tensor, device: str = "cuda",
                                      blocksize: int = 128) -> torch.Tensor:
    """⭐⭐⭐⭐⭐ 23/9 09:20 — STESSA identica scelta del trit di
    raffina_trit_sweep_batch (verificata, mai una riga di matematica toccata
    nella logica di selezione), ma con l'aggiornamento a BLOCCHI del paper
    GPTQ originale ("lazy batch updates", Frantar et al. 2210.17323) invece
    dell'aggiornamento rank-1 a larghezza PIENA a ogni posizione.

    Misurato (Angelo: «voglio pochi minuti, non 500x inventati»): l'update
    `g += delta_e ⊗ H[:,j,:]` a larghezza 2048 per OGNI posizione sposta
    4,4 TB per sweep (256 esperti) — 36,4 GB/s effettivi, ~15% del picco
    della memoria unificata. Cercato online prima di inventare un kernel:
    GPTQ risolve ESATTAMENTE questo problema tenendo l'update INCREMENTALE
    solo DENTRO il blocco corrente (economico, larghezza <=blocksize) e
    facendo UN SOLO aggiornamento a matmul per il resto della matrice dopo
    ogni blocco — non 2048 piccoli aggiornamenti elementwise ma n_in/blocksize
    matmul batchati, che usano le routine BLAS ottimizzate della GPU invece
    del broadcast elementwise (molto meno efficiente in banda).

    Wrot,Q,D,Hrot: stesso contratto di raffina_trit_sweep_batch. Ritorna
    Q_new (B,rows,n_in) int8 — deve combaciare ESATTAMENTE con la versione
    non a blocchi (stessa aritmetica, solo riorganizzata)."""
    B, rows, n_in = Wrot.shape
    ng = n_in // GROUP
    W = Wrot.to(device=device, dtype=torch.float64)
    Qc = Q.to(device=device, dtype=torch.float64).clone()
    Dg = D.to(device=device, dtype=torch.float64)
    H = torch.as_tensor(Hrot, dtype=torch.float64, device=device)
    assert H.shape == (B, n_in, n_in)

    d_per_pos = Dg.repeat_interleave(GROUP, dim=2)
    Wq = Qc * d_per_pos
    e = W - Wq
    g = torch.stack([e[b] @ H[b] for b in range(H.shape[0])])  # (B, rows, n_in)
    Hdiag = torch.diagonal(H, dim1=-2, dim2=-1).clamp(min=1e-12)  # (B, n_in)

    for i1 in range(0, n_in, blocksize):
        i2 = min(i1 + blocksize, n_in)
        count = i2 - i1
        Err_blocco = torch.empty(B, rows, count, dtype=torch.float64, device=device)

        for jj in range(count):
            j = i1 + jj
            dj = d_per_pos[:, :, j]
            q_cur = Qc[:, :, j]
            e_j_old = e[:, :, j]
            g_j_old = g[:, :, j]
            w_j = e_j_old + dj * q_cur
            Hdiag_j = Hdiag[:, j].unsqueeze(1)
            lin_coeff = g_j_old - Hdiag_j * e_j_old
            best_val = None
            best_q = q_cur.clone()
            best_e = e_j_old.clone()
            for cand in (-1.0, 0.0, 1.0):
                e_cand = w_j - dj * cand
                val = e_cand * e_cand * Hdiag_j + 2.0 * e_cand * lin_coeff
                if best_val is None:
                    best_val = val
                    best_q = torch.full_like(q_cur, cand)
                    best_e = e_cand
                else:
                    migliora = val < best_val
                    best_val = torch.where(migliora, val, best_val)
                    best_q = torch.where(migliora, torch.full_like(q_cur, cand), best_q)
                    best_e = torch.where(migliora, e_cand, best_e)
            dj_zero = (dj == 0)
            if dj_zero.any():
                best_q = torch.where(dj_zero, q_cur, best_q)
                best_e = torch.where(dj_zero, e_j_old, best_e)
            delta_e = best_e - e_j_old
            # ⭐ aggiornamento LOCALE: solo le colonne [j:i2] restanti nel
            # blocco corrente (larghezza <= blocksize, MAI n_in intero).
            resto_blocco = i2 - j
            if resto_blocco > 0:
                g[:, :, j:i2] += delta_e.unsqueeze(-1) * H[:, j, j:i2].unsqueeze(1)
            e[:, :, j] = best_e
            Qc[:, :, j] = best_q
            Err_blocco[:, :, jj] = delta_e

        # ⭐ UN SOLO aggiornamento matmul per TUTTE le colonne oltre il blocco
        # (torch.matmul broadcasta sulla dimensione batch B nativamente).
        if i2 < n_in:
            g[:, :, i2:] += torch.matmul(Err_blocco, H[:, i1:i2, i2:])

    return Qc.to(torch.int8).cpu()


@torch.no_grad()
def raffina_coorddesc(Wrot: torch.Tensor, Q0: torch.Tensor, D0: torch.Tensor,
                       Hrot: torch.Tensor, n_round: int = 3, device: str = "cuda",
                       log_ogni: bool = False):
    """Alterna scala-joint e trit-sweep per n_round round. Ogni round non puo'
    peggiorare la loss pesata-H (entrambe le mosse sono ottimi esatti locali).
    Ritorna (D, Q) nello stesso contratto di gptq_1024."""
    from gptq_1024_softern import loss_quadratica  # riuso la stessa metrica

    Hrot_t = torch.as_tensor(Hrot, dtype=torch.float32)
    Q, D = Q0.clone(), D0.clone()
    if log_ogni:
        log(f"  round 0 (GPTQ base): loss={loss_quadratica(Wrot, Hrot_t, D, Q.float(), device=device):.6g}")
    for r in range(n_round):
        Q = raffina_trit_sweep(Wrot, Q, D, Hrot_t, device=device)
        D = raffina_scale_joint(Wrot, Q.float(), Hrot_t, device=device)
        if log_ogni:
            l = loss_quadratica(Wrot, Hrot_t, D, Q.float(), device=device)
            log(f"  round {r+1}: loss={l:.6g}")
    return D, Q
