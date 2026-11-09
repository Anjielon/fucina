"""⭐⭐⭐ 23/9 11:20 — KERNEL TRITON: un INTERO blocco del coordinate descent
in UN SOLO lancio GPU.

Perche' (diagnosi rifatta dopo il fix a blocchi, i conti di stanotte erano
superati): con l'aggiornamento a blocchi il traffico di memoria e' crollato
da 4,4 TB a ~310 GB per sweep — cioe' ~6 GB/s effettivi su un picco di
210-260. NON siamo piu' limitati dalla banda. Adesso domina il NUMERO DI
LANCI: 2048 posizioni x ~20 operazioni elementwise su tensori da appena
131k elementi (256x512) = ~41.000 lanci GPU, ognuno con un overhead fisso
di decine di microsecondi che nessun dato "riempie".

Questo kernel elimina esattamente quello: ogni programma tiene in REGISTRI
il riquadro (righe x colonne-del-blocco) e percorre le COUNT posizioni del
blocco INTERNAMENTE, senza mai tornare all'host. Lanci: 1 per blocco invece
di ~320 -> da ~41.000 a ~128 per sweep.

⛔ La matematica e' COPIATA RIGA PER RIGA da raffina_trit_sweep_batch_blocchi
(gia' verificata 0/268.435.456 diversi contro il riferimento sequenziale),
compresa la guardia dj==0 (il bug vero trovato stanotte). Va verificata con
lo stesso rigore PRIMA di usarla: se diverge di un solo trit, non si usa.

⚠️ L'aggiornamento FRA blocchi (matmul Err @ H[i1:i2, i2:]) resta in PyTorch:
li' il matmul e' gia' efficiente e non e' quello il collo di bottiglia.
"""
from __future__ import annotations
import torch
import triton
import triton.language as tl


@triton.jit
def _kernel_blocco(
    e_ptr, g_ptr, q_ptr, d_ptr, err_ptr,
    hloc_ptr, hdiag_ptr,
    ROWS, N_IN, I1,
    se_b, se_r, se_n,          # stride di e/g/q/d (B, rows, n_in)
    serr_b, serr_r, serr_c,    # stride di err (B, rows, COUNT)
    shl_b, shl_i, shl_j,       # stride di hloc (B, COUNT, COUNT)
    shd_b, shd_n,              # stride di hdiag (B, N_IN)
    COUNT: tl.constexpr, BLOCK_ROWS: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_r = tl.program_id(1)

    r_off = pid_r * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    r_mask = r_off < ROWS
    c_idx = tl.arange(0, COUNT)                 # 0..COUNT-1 (indice DENTRO il blocco)
    c_off = I1 + c_idx                          # colonna assoluta

    m2 = r_mask[:, None]

    base = pid_b * se_b + r_off[:, None] * se_r + c_off[None, :] * se_n
    e_t = tl.load(e_ptr + base, mask=m2, other=0.0)
    g_t = tl.load(g_ptr + base, mask=m2, other=0.0)
    q_t = tl.load(q_ptr + base, mask=m2, other=0.0)
    d_t = tl.load(d_ptr + base, mask=m2, other=0.0)

    hdiag = tl.load(hdiag_ptr + pid_b * shd_b + c_off * shd_n)          # (COUNT,)
    hloc = tl.load(hloc_ptr + pid_b * shl_b
                   + c_idx[:, None] * shl_i + c_idx[None, :] * shl_j)   # (COUNT, COUNT)

    err_t = tl.zeros((BLOCK_ROWS, COUNT), dtype=e_t.dtype)

    for jj in tl.static_range(COUNT):
        sel = (c_idx == jj)                     # (COUNT,) maschera della colonna jj
        selc = sel.to(e_t.dtype)
        # estrazione ESATTA della colonna jj (un solo elemento non nullo: nessun
        # arrotondamento introdotto dalla somma)
        dj = tl.sum(d_t * selc[None, :], axis=1)
        q_cur = tl.sum(q_t * selc[None, :], axis=1)
        e_old = tl.sum(e_t * selc[None, :], axis=1)
        g_old = tl.sum(g_t * selc[None, :], axis=1)
        hd = tl.sum(hdiag * selc)               # scalare
        hrow = tl.sum(hloc * selc[:, None], axis=0)   # riga jj di hloc, (COUNT,)

        w_j = e_old + dj * q_cur
        lin = g_old - hd * e_old

        # candidato -1.0 (imposta incondizionatamente, come il riferimento)
        e_m1 = w_j - dj * (-1.0)
        v_m1 = e_m1 * e_m1 * hd + 2.0 * e_m1 * lin
        best_val = v_m1
        best_e = e_m1
        best_q = tl.full(best_val.shape, -1.0, dtype=e_t.dtype)
        # candidato 0.0
        e_0 = w_j
        v_0 = e_0 * e_0 * hd + 2.0 * e_0 * lin
        migliora = v_0 < best_val
        best_val = tl.where(migliora, v_0, best_val)
        best_e = tl.where(migliora, e_0, best_e)
        best_q = tl.where(migliora, tl.zeros(best_q.shape, dtype=e_t.dtype), best_q)
        # candidato 1.0
        e_p1 = w_j - dj
        v_p1 = e_p1 * e_p1 * hd + 2.0 * e_p1 * lin
        migliora = v_p1 < best_val
        best_val = tl.where(migliora, v_p1, best_val)
        best_e = tl.where(migliora, e_p1, best_e)
        best_q = tl.where(migliora, tl.full(best_q.shape, 1.0, dtype=e_t.dtype), best_q)

        # ⛔ guardia dj==0 (il bug vero di stanotte): pareggio matematico fra i
        # tre candidati -> non si tocca il trit corrente
        dj_zero = dj == 0.0
        best_q = tl.where(dj_zero, q_cur, best_q)
        best_e = tl.where(dj_zero, e_old, best_e)

        delta = best_e - e_old

        # aggiornamento LOCALE dentro il blocco: solo colonne [jj, COUNT)
        avanti = (c_idx >= jj).to(e_t.dtype)
        g_t = g_t + delta[:, None] * (hrow * avanti)[None, :]

        e_t = tl.where(sel[None, :], best_e[:, None], e_t)
        q_t = tl.where(sel[None, :], best_q[:, None], q_t)
        err_t = tl.where(sel[None, :], delta[:, None], err_t)

    tl.store(e_ptr + base, e_t, mask=m2)
    tl.store(g_ptr + base, g_t, mask=m2)
    tl.store(q_ptr + base, q_t, mask=m2)
    err_base = pid_b * serr_b + r_off[:, None] * serr_r + c_idx[None, :] * serr_c
    tl.store(err_ptr + err_base, err_t, mask=m2)


@torch.no_grad()
def raffina_trit_sweep_batch_fuso(Wrot, Q, D, Hrot, device="cuda",
                                   blocksize=16, block_rows=64):
    """STESSO contratto e STESSA matematica di
    raffina_trit_sweep_batch_blocchi, ma il ciclo interno al blocco gira
    dentro UN kernel Triton invece che in Python. Ritorna Q_new int8 su CPU."""
    from gptq_1024_lib import GROUP

    B, rows, n_in = Wrot.shape
    W = Wrot.to(device=device, dtype=torch.float64)
    Qc = Q.to(device=device, dtype=torch.float64).clone()
    Dg = D.to(device=device, dtype=torch.float64)
    H = torch.as_tensor(Hrot, dtype=torch.float64, device=device)

    d_per_pos = Dg.repeat_interleave(GROUP, dim=2).contiguous()
    e = (W - Qc * d_per_pos).contiguous()
    g = torch.stack([e[b] @ H[b] for b in range(B)]).contiguous()
    Hdiag = torch.diagonal(H, dim1=-2, dim2=-1).clamp(min=1e-12).contiguous()
    Qc = Qc.contiguous()

    err = torch.empty(B, rows, blocksize, dtype=torch.float64, device=device)

    for i1 in range(0, n_in, blocksize):
        count = min(blocksize, n_in - i1)
        i2 = i1 + count
        err_b = err if count == blocksize else err[:, :, :count].contiguous()
        blocco_fuso(e, g, Qc, d_per_pos, H, Hdiag, i1, count, err_b,
                    block_rows=block_rows)
        if i2 < n_in:
            g[:, :, i2:] += torch.matmul(err_b, H[:, i1:i2, i2:])

    return Qc.to(torch.int8).cpu()


def blocco_fuso(e, g, Qc, d_per_pos, H, Hdiag, i1, count, err_out, block_rows=64):
    """Esegue le `count` posizioni del blocco [i1, i1+count) in UN lancio.
    Modifica e/g/Qc in place e riempie err_out (B, rows, count).
    e,g,Qc,d_per_pos: (B,rows,n_in) · H: (B,n_in,n_in) · Hdiag: (B,n_in)."""
    B, rows, n_in = e.shape
    hloc = H[:, i1:i1 + count, i1:i1 + count].contiguous()
    grid = (B, triton.cdiv(rows, block_rows))
    _kernel_blocco[grid](
        e, g, Qc, d_per_pos, err_out,
        hloc, Hdiag,
        rows, n_in, i1,
        e.stride(0), e.stride(1), e.stride(2),
        err_out.stride(0), err_out.stride(1), err_out.stride(2),
        hloc.stride(0), hloc.stride(1), hloc.stride(2),
        Hdiag.stride(0), Hdiag.stride(1),
        COUNT=count, BLOCK_ROWS=block_rows,
    )
