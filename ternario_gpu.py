"""Quantizzatore ternario a DUE PIANI + GPTQ, sulla GPU.

⛔ PERCHE' ESISTE. In numpy il conto e' cronometrato: 0,6 milioni di pesi al
   secondo → **192 ore** per i 386,5 miliardi di ODINO. Non e' un difetto di
   ciclo (ho vettorizzato e tolto la ricerca sui nove livelli: stesso ordine
   di grandezza). Servono ~50 operazioni per peso e numpy ne fa ~300 milioni
   al secondo: e' aritmetica, non pigrizia.
   La sola leva che cambia ordine di grandezza e' la GPU.

Stesso algoritmo di `gptq_veloce.py`, operatore per operatore:
  1. scale d1,d2 per blocco di 256, avviate col sequenziale e poi alternate
     in forma chiusa (sistema 2x2)
  2. ogni peso al migliore dei NOVE valori d1*t1 + d2*t2, t ∈ {-1,0,+1}
  3. GPTQ: un giro sulle 256 colonne, l'errore spinto sulle successive
     tramite il fattore di Cholesky di H⁻¹

⚠️ La VRAM vista da torch e' ~15 GiB (non i 96 di Vulkan), quindi si lavora
   a fette: `pezzo` righe per volta.
"""
from __future__ import annotations
import torch

BLOCCO = 256


def _scale(W, T1, T2, lam=1e-6):
    a11 = (T1 * T1).sum(-1) + lam
    a22 = (T2 * T2).sum(-1) + lam
    a12 = (T1 * T2).sum(-1)
    b1 = (W * T1).sum(-1); b2 = (W * T2).sum(-1)
    det = a11 * a22 - a12 * a12
    det = torch.where(det.abs() < 1e-12, torch.full_like(det, 1e-12), det)
    return (((b1 * a22 - b2 * a12) / det).unsqueeze(-1),
            ((b2 * a11 - b1 * a12) / det).unsqueeze(-1))


def _nove_rapido(w, d1, d2):
    """Assegnazione economica: t1 arrotondando w/d1, t2 sul residuo.
    4 operazioni invece di 36. Dentro il ciclo GPTQ (256 giri per blocco)
    e' li' che se ne va il tempo: con la ricerca esatta ODINO costa 7,7 ore,
    con questa ~2,3. La differenza sull'errore e' sotto il punto."""
    t1 = torch.clamp(torch.round(w / d1.clamp(min=1e-20)), -1, 1)
    t2 = torch.clamp(torch.round((w - d1 * t1) / d2.clamp(min=1e-20)), -1, 1)
    return t1, t2


def _nove(w, d1, d2):
    """il migliore dei nove livelli, esatto (9 combinazioni)"""
    best_e = None; best_c = None
    for a in (-1.0, 0.0, 1.0):
        for c in (-1.0, 0.0, 1.0):
            e = (w - (d1 * a + d2 * c)).square_()
            if best_e is None:
                best_e = e
                best_c = torch.full_like(w, a * 3 + c + 4)
            else:
                m = e < best_e
                best_e = torch.where(m, e, best_e)
                best_c = torch.where(m, torch.full_like(w, a * 3 + c + 4), best_c)
    return torch.div(best_c, 3, rounding_mode='floor') - 1.0, best_c.remainder(3) - 1.0


def _scala_un_piano(B, griglia=(0.55, 0.66, 0.76, 0.86, 1.00, 1.20)):
    """la scala ottima ai minimi quadrati (l'equivalente di ternario_ottimo)"""
    ab = B.abs(); media = ab.mean(-1, keepdim=True)
    en = (B * B).sum(-1)
    best_e = torch.full_like(en, float('inf')); best_d = torch.zeros_like(en)
    for m in griglia:
        vivi = ab > (m * media)
        k = vivi.sum(-1).to(B.dtype)
        somma = torch.where(vivi, ab, torch.zeros_like(ab)).sum(-1)
        e = en - torch.where(k > 0, somma * somma / k.clamp(min=1), torch.zeros_like(en))
        ok = e < best_e
        best_e = torch.where(ok, e, best_e)
        best_d = torch.where(ok, torch.where(k > 0, somma / k.clamp(min=1), torch.zeros_like(k)), best_d)
    d = best_d.unsqueeze(-1)
    q = torch.clamp(torch.round(B / d.clamp(min=1e-20)), -1, 1)
    return d, q


@torch.no_grad()
def quantizza(W, Hchol=None, giri: int = 4, pezzo: int = 2_000_000, device="cuda"):
    """W: (n_righe, n_in) su CPU → (d1, q1, d2, q2) su CPU, int8 per i segni."""
    n_righe, n_in = W.shape
    assert n_in % BLOCCO == 0
    nb = n_in // BLOCCO
    D1 = torch.empty(n_righe, nb, 1, dtype=torch.float32)
    D2 = torch.empty_like(D1)
    Q1 = torch.empty(n_righe, n_in, dtype=torch.int8)
    Q2 = torch.empty_like(Q1)
    H = Hchol.to(device) if Hchol is not None else None   # (n_in, n_in) INTERA

    # ⚡ Il ciclo GPTQ fa 4.096 lanci di kernel PER PEZZO (16 blocchi × 256
    #    colonne). Con pezzi piccoli i lanci si moltiplicano: a 4M erano 34
    #    pezzi per tensore = 139.000 lanci, e il tempo se ne andava li'.
    #    Pezzi grandi = stessi 4.096 lanci su piu' dati.
    # 🛟 Se la VRAM non regge si dimezza e si riprova, invece di morire.
    righe_per_pezzo = max(1, pezzo // nb)
    i0 = 0
    while i0 < n_righe:
        i1 = min(i0 + righe_per_pezzo, n_righe)
        try:
            B = W[i0:i1].to(device, torch.float32, non_blocking=True).reshape(-1, BLOCCO)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            righe_per_pezzo = max(1, righe_per_pezzo // 2)
            continue
        d1, t1 = _scala_un_piano(B)
        d2, t2 = _scala_un_piano(B - d1 * t1)
        for _ in range(giri):
            d1, d2 = _scale(B, t1, t2)
            n1, n2 = _nove_rapido(B, d1, d2)
            if torch.equal(n1, t1) and torch.equal(n2, t2):
                break
            t1, t2 = n1, n2
        d1, d2 = _scale(B, t1, t2)

        if H is None:
            q1, q2 = t1, t2
        else:
            # ── GPTQ vero: l'errore si spinge DENTRO il blocco (ciclo sulle
            #    256 colonne) E FRA i blocchi (un prodotto matriciale per
            #    blocco, che sulla GPU costa pochissimo).
            #    ⛔ Nella prima stesura mancava la seconda parte e usavo la
            #       MEDIA dei blocchi diagonali di H: misurato 15,67% invece
            #       dei ~6% che l'anisotropia vera (alfa 1,17) promette.
            righe = i1 - i0
            M = B.reshape(righe, nb, BLOCCO)          # (righe, blocchi, 256)
            S1 = d1.reshape(righe, nb, 1); S2 = d2.reshape(righe, nb, 1)
            Q1p = torch.empty_like(M); Q2p = torch.empty_like(M)
            for b in range(nb):
                Wb = M[:, b, :].clone()
                s1 = S1[:, b, 0]; s2 = S2[:, b, 0]
                j0 = b * BLOCCO
                err = torch.empty_like(Wb)
                for j in range(BLOCCO):
                    w = Wb[:, j]
                    a, c = _nove_rapido(w, s1, s2)
                    Q1p[:, b, j] = a; Q2p[:, b, j] = c
                    e = (w - (s1 * a + s2 * c)) / H[j0 + j, j0 + j].clamp(min=1e-8)
                    err[:, j] = e
                    if j + 1 < BLOCCO:
                        Wb[:, j+1:] -= e.unsqueeze(1) * H[j0 + j, j0+j+1 : j0+BLOCCO].unsqueeze(0)
                if b + 1 < nb:       # spinta sui blocchi successivi: UN matmul
                    M[:, b+1:, :] -= (err @ H[j0:j0+BLOCCO, j0+BLOCCO:]).reshape(righe, nb-b-1, BLOCCO)
            q1 = Q1p.reshape(-1, BLOCCO); q2 = Q2p.reshape(-1, BLOCCO)
        D1[i0:i1] = d1.reshape(i1 - i0, nb, 1).cpu()
        D2[i0:i1] = d2.reshape(i1 - i0, nb, 1).cpu()
        Q1[i0:i1] = q1.reshape(i1 - i0, n_in).to(torch.int8).cpu()
        Q2[i0:i1] = q2.reshape(i1 - i0, n_in).to(torch.int8).cpu()
        del B
        i0 = i1
    return D1.numpy(), Q1.numpy(), D2.numpy(), Q2.numpy()


@torch.no_grad()
def quantizza_un_piano(W, Hchol, pezzo: int = 3_000_000, device="cuda"):
    """UN SOLO piano ternario, con GPTQ e scala RICALCOLATA per blocco.

    ⛔⛔ PERCHE' ESISTE (misurato il 26/8 mattina, 10 punti di differenza):
       prendere il PIANO 1 di un'ottimizzazione a DUE piani congiunti da'
       **37,95%**; quantizzare a un piano solo, dedicato, da' **28,13%**.
       Il piano 1 congiunto e' CO-ADATTATO al secondo: da solo e' pessimo.
       (E' il rischio che AnyBCQ segnala e che avevo annotato senza verificare.)

    ⭐ E la scala si ricalcola DENTRO il ciclo, dal blocco GIA' COMPENSATO
       dai blocchi precedenti — non dal blocco originale (28,13 contro 28,43).
    """
    n_righe, n_in = W.shape
    assert n_in % BLOCCO == 0
    nb = n_in // BLOCCO
    D1 = torch.empty(n_righe, nb, 1, dtype=torch.float32)
    Q1 = torch.empty(n_righe, n_in, dtype=torch.int8)
    H = Hchol.to(device)
    righe_per_pezzo = max(1, pezzo // nb)
    i0 = 0
    while i0 < n_righe:
        i1 = min(i0 + righe_per_pezzo, n_righe)
        try:
            Wc = W[i0:i1].to(device, torch.float32, non_blocking=True)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache(); righe_per_pezzo = max(1, righe_per_pezzo // 2); continue
        righe = i1 - i0
        dloc = torch.empty(righe, nb, 1, device=device)
        qloc = torch.empty(righe, n_in, device=device)
        for b in range(nb):
            j0, j1 = b * BLOCCO, (b + 1) * BLOCCO
            blocco = Wc[:, j0:j1].clone()
            d, _ = _scala_un_piano(blocco)          # scala dal blocco COMPENSATO
            dloc[:, b] = d
            err = torch.empty_like(blocco)
            inv = 1.0 / d.clamp(min=1e-20)
            for j in range(BLOCCO):
                w = blocco[:, j]
                q = torch.clamp(torch.round(w * inv[:, 0]), -1, 1)
                qloc[:, j0 + j] = q
                e = (w - d[:, 0] * q) / H[j0 + j, j0 + j].clamp(min=1e-8)
                err[:, j] = e
                if j + 1 < BLOCCO:
                    blocco[:, j+1:] -= e.unsqueeze(1) * H[j0 + j, j0+j+1 : j1].unsqueeze(0)
            if j1 < n_in:
                Wc[:, j1:] -= err @ H[j0:j1, j1:]
        D1[i0:i1] = dloc.cpu(); Q1[i0:i1] = qloc.to(torch.int8).cpu()
        del Wc, dloc, qloc
        i0 = i1
    return D1.numpy(), Q1.numpy()
