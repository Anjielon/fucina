#!/usr/bin/env python3
"""HAAR + RAGGRUPPAMENTO PER BANDE — ri-processo di una leva bocciata male.

CONTESTO
--------
`levers.py` registra `local_haar` come "0.1% — nothing" (fonte HBLLM,
arXiv 2512.00862). Il riesame ha trovato DUE errori in quella misura:

  1. la trasformata era applicata alla dimensione 4096 (il flusso residuo,
     CONDIVISA fra tutti gli esperti) invece che alla dimensione PRIVATA
     dell'esperto (1024, l'asse su cui corrono davvero i blocchi da 256);
  2. mancava il raggruppamento per bande: la Haar multi-livello in ordine di
     Mallat mette le bande piccole tutte all'inizio, quindi il PRIMO blocco
     da 256 ne mescola 7-9 di magnitudine diversissima. Una sola scala f16
     per blocco non puo' servirle tutte, e il beneficio si annulla.

L'IDEA
------
Haar concentra l'energia nelle bande basse. Se ogni blocco da 256 contiene UNA
SOLA banda, i blocchi diventano omogenei per magnitudine e la scala per blocco
(un f16 ogni 256 pesi, il formato TQ1_0) diventa molto piu' efficace.

Su n_in = 1024 con blocchi da 256 l'allineamento e' ESATTO a L = 2 livelli:

    bande  = [ a2 (256) | d2 (256) | d1 (512) ]
    blocchi= [   b0     |    b1    |  b2 | b3 ]

cioe' a L=2 il "raggruppamento per bande" e' GRATIS: e' la permutazione
identita'. A L>=3 le bande diventano piu' corte di 256 e il formato a blocco
fisso non puo' piu' isolarle — quello e' il regime in cui la leva si autodistrugge.

COSA MISURA
-----------
Errore relativo di Frobenius ||W - W_ric|| / ||W|| della quantizzazione
ternaria con scala ai minimi quadrati per blocco di 256 (la stessa
`_scala_un_piano` di ternary_gpu.py), a UNO e a DUE piani congiunti.
La Haar usata e' ORTONORMALE, quindi la trasformata conserva la norma e
l'errore misurato nel dominio dei pesi e' confrontabile riga per riga.

Varianti a confronto:
    baseline        nessuna trasformata
    haar_L1..L10    Haar ortonormale sull'asse privato (1024), ordine di Mallat
                    (= bande contigue). A L<=2 i blocchi sono bande pure.
    haar_wrongdim   Haar sull'asse 4096 CONDIVISO — riproduce il nostro errore
    perm_magnitudine  sola permutazione per |w| medio di colonna (controllo
                    GRATIS: una permutazione si ripiega davvero a costo zero)

Vincoli: sola lettura, pochi esperti, < 4 GB di RAM, pochi minuti.

Uso:
    ~/venv-catq/bin/python haar_bands.py [--esperti 4] [--strato 10]
"""
from __future__ import annotations
import os

import argparse
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ternary_gpu as G  # noqa: E402  (solo lettura, non lo modifichiamo)

BLOCCO = G.BLOCCO  # 256

# I pesi originali in bf16. Il GGUF odino-v31 e' gia' TQ1_0: ri-quantizzare
# un tensore gia' ternario e' circolare (3 livelli per blocco -> errore finto
# vicino a zero), quindi la sorgente onesta e' il bf16 di partenza.
FP_DIR = Path(os.environ.get("FP_CHECKPOINT_DIR", "/mnt/checkpoints/Ornith-1.5-397B"))
GGUF_TQ1 = Path("/mnt/models/gguf/odino-v31/ODINO-397B-v31.gguf")


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


# ─────────────────────────────────────────────────────────────────────────
# Lettura: solo la fetta di byte degli esperti richiesti (niente shard intera)
# ─────────────────────────────────────────────────────────────────────────
def carica_down_exps(strato: int, n_esperti: int) -> torch.Tensor:
    """(n_esperti, 4096, 1024) float32 dal bf16, leggendo solo i byte serviti."""
    idx = json.loads((FP_DIR / "model.safetensors.index.json").read_text())
    chiave = f"model.language_model.layers.{strato}.mlp.experts.down_proj"
    shard = FP_DIR / idx["weight_map"][chiave]

    with open(shard, "rb") as f:
        n_hdr = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n_hdr))
        meta = hdr[chiave]
        assert meta["dtype"] == "BF16", meta["dtype"]
        E, n_out, n_in = meta["shape"]
        assert n_esperti <= E
        base = 8 + n_hdr + meta["data_offsets"][0]
        per_esperto = n_out * n_in * 2
        f.seek(base)
        crudo = f.read(per_esperto * n_esperti)

    bf16 = torch.frombuffer(bytearray(crudo), dtype=torch.bfloat16)
    W = bf16.reshape(n_esperti, n_out, n_in).to(torch.float32)
    log(f"caricato {chiave} · esperti 0..{n_esperti-1} · {tuple(W.shape)} "
        f"· {W.numel()*4/2**20:.0f} MiB")
    return W


# ─────────────────────────────────────────────────────────────────────────
# Haar ortonormale multi-livello, uscita in ordine di banda (Mallat)
# ─────────────────────────────────────────────────────────────────────────
_R2 = 2.0 ** -0.5


def haar_avanti(X: torch.Tensor, livelli: int) -> torch.Tensor:
    """(..., n) -> (..., n) coefficienti in ordine [a_L | d_L | ... | d_1].

    Ortonormale (fattori 1/sqrt(2)): la trasformata conserva la norma, quindi
    l'errore misurato nel dominio trasformato e' quello vero nei pesi.
    """
    a = X
    dettagli = []
    for _ in range(livelli):
        n = a.shape[-1]
        assert n % 2 == 0, f"lunghezza dispari a livello {n}"
        p = a.reshape(*a.shape[:-1], n // 2, 2)
        dettagli.append((p[..., 0] - p[..., 1]) * _R2)
        a = (p[..., 0] + p[..., 1]) * _R2
    return torch.cat([a] + dettagli[::-1], dim=-1)


def haar_indietro(C: torch.Tensor, livelli: int) -> torch.Tensor:
    n = C.shape[-1]
    tagli = [n >> livelli] + [n >> k for k in range(livelli, 0, -1)]
    pezzi = torch.split(C, tagli, dim=-1)
    a = pezzi[0]
    for i in range(1, livelli + 1):
        d = pezzi[i]
        p = torch.stack([(a + d) * _R2, (a - d) * _R2], dim=-1)
        a = p.reshape(*a.shape[:-1], a.shape[-1] * 2)
    return a


def bande(n: int, livelli: int) -> list[int]:
    return [n >> livelli] + [n >> k for k in range(livelli, 0, -1)]


def bande_per_blocco(n: int, livelli: int) -> list[int]:
    """quante bande distinte cadono dentro ciascun blocco da 256"""
    confini = np.cumsum([0] + bande(n, livelli))
    out = []
    for b in range(n // BLOCCO):
        i0, i1 = b * BLOCCO, (b + 1) * BLOCCO
        out.append(int(((confini[:-1] < i1) & (confini[1:] > i0)).sum()))
    return out


# ─────────────────────────────────────────────────────────────────────────
# Quantizzazione ternaria — la NOSTRA, importata, non riscritta
# ─────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def errore_un_piano(W: torch.Tensor) -> float:
    """UN piano ternario, scala ai minimi quadrati per blocco di 256."""
    n_in = W.shape[-1]
    assert n_in % BLOCCO == 0
    B = W.reshape(-1, BLOCCO)
    d, q = G._scala_un_piano(B)
    ric = d * q
    return float(torch.linalg.vector_norm(B - ric) / torch.linalg.vector_norm(B))


@torch.no_grad()
def errore_due_piani(W: torch.Tensor, device: str) -> float:
    """DUE piani congiunti (la strada dei 28 esperti caldi), senza Hessiana."""
    n_righe = int(np.prod(W.shape[:-1]))
    n_in = W.shape[-1]
    Wc = W.reshape(n_righe, n_in).cpu()
    D1, Q1, D2, Q2 = G.quantizza(Wc, Hchol=None, device=device)
    nb = n_in // BLOCCO
    ric = (torch.from_numpy(D1) * torch.from_numpy(Q1).float().reshape(n_righe, nb, BLOCCO)
           + torch.from_numpy(D2) * torch.from_numpy(Q2).float().reshape(n_righe, nb, BLOCCO)
           ).reshape(n_righe, n_in)
    return float(torch.linalg.vector_norm(Wc - ric) / torch.linalg.vector_norm(Wc))


@torch.no_grad()
def diagnosi(W: torch.Tensor, livelli: tuple = (1, 2, 9)) -> dict:
    """PERCHE' la leva funziona o no — le tre grandezze che la decidono.

    Haar paga solo se i pesi sono CORRELATI lungo l'asse trasformato: e' un
    filtro passa-basso locale, presuppone regolarita' spaziale. Se i pesi
    adiacenti sono scorrelati, la trasformata non concentra nulla e in piu'
    somma variabili indipendenti -> GAUSSIANIZZA, che al ternario fa male
    (il ternario vuole code pesanti/sparsita', non una gaussiana).
    """
    def curtosi(x):
        x = x.flatten().float()
        x = x - x.mean()
        return float((x ** 4).mean() / (x ** 2).mean() ** 2)

    c = W.reshape(-1, W.shape[-1])
    a, b = c[:, :-1].flatten(), c[:, 1:].flatten()
    a, b = a - a.mean(), b - b.mean()
    rho = float((a * b).mean() / (a.std() * b.std()))
    log(f"diagnosi · correlazione fra pesi adiacenti sull'asse privato: {rho:.5f} "
        f"({'CORRELATI: Haar puo' if abs(rho) > 0.05 else 'SCORRELATI: Haar non ha nulla da concentrare'})")
    log(f"diagnosi · curtosi pesi originali: {curtosi(W):.3f}")
    for L in livelli:
        C = haar_avanti(W, L)
        tg = bande(W.shape[-1], L)
        i, tot, quote = 0, float((C * C).sum()), []
        for k, t in enumerate(tg):
            e = float((C[..., i:i + t] ** 2).sum()) / tot * 100
            quote.append(f"{'a' if k == 0 else 'd'}{L if k == 0 else L-k+1}:{e:.1f}%")
            i += t
        log(f"diagnosi · L={L} curtosi {curtosi(C):.3f} (3.0=gaussiana) · "
            f"energia per banda {' '.join(quote)}")
    return {"rho_adiacenti": rho, "curtosi_originale": curtosi(W)}


@torch.no_grad()
def errore_un_piano_nei_pesi(W: torch.Tensor, avanti, indietro) -> float:
    """Quantizza nel dominio trasformato, ricostruisce e misura NEI PESI.

    Controllo di onesta': con una trasformata ortonormale deve coincidere con
    l'errore misurato nel dominio trasformato. Se non coincide, c'e' un bug.
    """
    C = avanti(W)
    n_in = C.shape[-1]
    B = C.reshape(-1, BLOCCO)
    d, q = G._scala_un_piano(B)
    Cric = (d * q).reshape(C.shape)
    Wric = indietro(Cric)
    return float(torch.linalg.vector_norm(W - Wric) / torch.linalg.vector_norm(W))


# ─────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strato", type=int, default=10)
    ap.add_argument("--esperti", type=int, default=4)
    ap.add_argument("--livelli-max", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    A = ap.parse_args()

    dev = A.device
    W = carica_down_exps(A.strato, A.esperti).to(dev)
    n_in = W.shape[-1]
    log(f"device={dev} · asse di quantizzazione = {n_in} (dimensione PRIVATA "
        f"dell'esperto) · blocchi da {BLOCCO}")

    # sanita': la Haar e' davvero ortonormale e invertibile?
    for L in (1, 2, 5, 10):
        c = haar_avanti(W[:1, :8], L)
        r = haar_indietro(c, L)
        assert torch.allclose(r, W[:1, :8], atol=1e-3), f"Haar non invertibile a L={L}"
        n0 = float(torch.linalg.vector_norm(W[:1, :8]))
        n1 = float(torch.linalg.vector_norm(c))
        assert abs(n0 - n1) / n0 < 1e-4, f"Haar non ortonormale a L={L}"
    log("controllo: Haar ortonormale e invertibile a L=1,2,5,10 ✓")

    ris = {}
    diagnosi(W)

    # ── baseline ────────────────────────────────────────────────────────
    b1 = errore_un_piano(W)
    ris["baseline"] = b1
    log(f"baseline (nessuna trasformata) 1 piano: {b1*100:.2f}%")

    # ── Haar sull'asse privato, ordine di banda, L = 1..max ─────────────
    for L in range(1, A.livelli_max + 1):
        if n_in >> L < 1:
            break
        e_tr = errore_un_piano(haar_avanti(W, L))
        e_pesi = errore_un_piano_nei_pesi(
            W, lambda X, L=L: haar_avanti(X, L), lambda X, L=L: haar_indietro(X, L))
        assert abs(e_tr - e_pesi) < 1e-4, (e_tr, e_pesi)
        bpb = bande_per_blocco(n_in, L)
        ris[f"haar_L{L}"] = e_pesi
        puro = "bande PURE" if max(bpb) == 1 else f"blocchi misti (bande/blocco: {bpb})"
        log(f"haar L={L:<2} 1 piano: {e_pesi*100:.2f}%  "
            f"({(e_pesi/b1-1)*100:+.2f}% vs baseline) · {puro}")

    # ── l'errore storico: Haar sull'asse 4096 CONDIVISO ─────────────────
    Wt = W.transpose(-1, -2).contiguous()          # (E, 1024, 4096)
    e_wd = errore_un_piano_nei_pesi(
        Wt, lambda X: haar_avanti(X, 2), lambda X: haar_indietro(X, 2))
    # attenzione: qui i blocchi corrono sull'asse 4096 -> non e' il nostro
    # formato. La misura utile e': trasformo su 4096, quantizzo su 1024.
    C = haar_avanti(W.transpose(-1, -2), 2).transpose(-1, -2).contiguous()
    e_wd2 = errore_un_piano(C)
    ris["haar_asse4096_condiviso"] = e_wd2
    log(f"haar sull'asse 4096 CONDIVISO (l'errore storico): {e_wd2*100:.2f}%  "
        f"({(e_wd2/b1-1)*100:+.2f}% vs baseline)")

    # ── controllo GRATIS: sola permutazione per magnitudine ─────────────
    ordine = W.abs().mean(dim=-2).mean(dim=0).argsort()   # colonne, |w| medio
    e_perm = errore_un_piano(W[..., ordine])
    ris["perm_magnitudine"] = e_perm
    log(f"sola permutazione per |w| medio (GRATIS, ripiegabile): "
        f"{e_perm*100:.2f}%  ({(e_perm/b1-1)*100:+.2f}% vs baseline)")

    # ── due piani congiunti: baseline vs la migliore Haar ───────────────
    migliore_L = min((L for L in range(1, A.livelli_max + 1) if f"haar_L{L}" in ris),
                     key=lambda L: ris[f"haar_L{L}"])
    log(f"— due piani congiunti (esperti caldi), baseline vs haar L={migliore_L} —")
    d1 = errore_due_piani(W, dev)
    d2 = errore_due_piani(haar_avanti(W, migliore_L), dev)
    ris["baseline_2piani"] = d1
    ris[f"haar_L{migliore_L}_2piani"] = d2
    log(f"baseline   2 piani: {d1*100:.2f}%")
    log(f"haar L={migliore_L} 2 piani: {d2*100:.2f}%  ({(d2/d1-1)*100:+.2f}%)")

    # ── verdetto ────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print(f"{'variante':<34}{'errore':>10}{'vs baseline':>16}")
    print("-" * 68)
    for k, v in ris.items():
        rif = ris["baseline_2piani"] if k.endswith("2piani") else ris["baseline"]
        print(f"{k:<34}{v*100:>9.2f}%{(v/rif-1)*100:>15.2f}%")
    print("=" * 68)
    g1 = (1 - ris[f"haar_L{migliore_L}"] / ris["baseline"]) * 100
    g2 = (1 - ris[f"haar_L{migliore_L}_2piani"] / ris["baseline_2piani"]) * 100
    print(f"\nGUADAGNO Haar+bande (L={migliore_L}): {g1:.2f}% a un piano, "
          f"{g2:.2f}% a due piani")
    print("(positivo = la leva VINCE, negativo = PERDE)")


if __name__ == "__main__":
    main()
