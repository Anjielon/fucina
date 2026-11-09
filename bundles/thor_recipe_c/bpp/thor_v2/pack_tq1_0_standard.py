"""Packer per il formato TQ1_0 STANDARD (enum 34, blocco 256, UNA scala per
blocco) che accetta valori GIA' OTTIMIZZATI (D, Q da forgia_batch_no_rot),
invece di ricalcolarli con round-to-nearest come fa gguf.quants.TQ1_0.quantize.

Perche' serve: chiudere il gap ffn_down_exps (Angelo, 23-24/9: "il 100% di
tutto deve essere chiuso, non dichiarato"). ffn_down_exps ha ingresso 512,
sotto il blocco 1024 richiesto dal codec ruotato TQ1_0R_1024 usato per gate/
up stanotte. Il verdetto cross-strato del 22/9 12:58 dice che il guadagno di
SchurQuant e' nell'obiettivo (W*/G), non nella rotazione (che peggiora
sempre) -> per down_proj basta il formato STANDARD (nessuna rotazione, nessun
nuovo codec/build llama.cpp), con gli stessi identici Q/D che l'obiettivo
Hessiano-contro-BF16 produce.

Riusa ESATTAMENTE la logica di bit-packing di gguf.quants.TQ1_0.quantize_blocks
(stesso schema base-243, stesso split qs0/qs1/qh) — sostituendo SOLO il primo
pezzo (calcolo di d/qs da float grezzi) con i valori GIA' pronti.

Verificato qui con un round-trip completo: pack(D,Q) -> dequantize_blocks
(la funzione VERA della libreria, non una mia reimplementazione) -> confronto
con D*Q atteso.
"""
import sys
sys.path.insert(0, "@@ROOT:models@@/build/llamacpp-tq1-k1024/gguf-py")
import numpy as np
from gguf.quants import np_roundf, TQ1_0

QK_STD = 256  # blocco standard TQ1_0 (non il 1024 del codec ruotato)


def pack_tq1_0_da_d_q(d: np.ndarray, trit: np.ndarray) -> np.ndarray:
    """d: (..., n_blocks) float32 (una scala ogni 256 pesi, GIA' calcolata da
    forgia_batch_no_rot). trit: (..., n_blocks*256) int8 in {-1,0,1}, GIA'
    ottimizzati (GPTQ coordinate descent), NON ricalcolati qui.
    Ritorna i byte impacchettati nel layout ESATTO di TQ1_0 (enum 34)."""
    orig_shape = trit.shape
    n_blocks_tot = int(np.prod(orig_shape[:-1])) * (orig_shape[-1] // QK_STD)
    blocks_trit = trit.reshape(n_blocks_tot, QK_STD)
    d_flat = d.reshape(n_blocks_tot, 1).astype(np.float32)

    # --- da qui, IDENTICO a TQ1_0.quantize_blocks, ma partendo da trit noto ---
    qs = (blocks_trit.astype(np.int8) + np.int8(1)).astype(np.uint8)  # {-1,0,1}->{0,1,2}

    qs0, qs1, qh = qs[..., :(32 * 5)], qs[..., (32 * 5):(48 * 5)], qs[..., (48 * 5):]
    qs0 = qs0.reshape((n_blocks_tot, -1, 5, 32)) * np.array([81, 27, 9, 3, 1], dtype=np.uint8).reshape((1, 1, 5, 1))
    qs0 = np.sum(qs0, axis=-2).reshape((n_blocks_tot, -1))
    qs1 = qs1.reshape((n_blocks_tot, -1, 5, 16)) * np.array([81, 27, 9, 3, 1], dtype=np.uint8).reshape((1, 1, 5, 1))
    qs1 = np.sum(qs1, axis=-2).reshape((n_blocks_tot, -1))
    qh = qh.reshape((n_blocks_tot, -1, 4, 4)) * np.array([81, 27, 9, 3], dtype=np.uint8).reshape((1, 1, 4, 1))
    qh = np.sum(qh, axis=-2).reshape((n_blocks_tot, -1))
    qs_packed = np.concatenate([qs0, qs1, qh], axis=-1)
    qs_packed = (qs_packed.astype(np.uint16) * 256 + (243 - 1)) // 243
    qs_packed = qs_packed.astype(np.uint8)

    d_bytes = d_flat.astype(np.float16).view(np.uint8)
    blocks_out = np.concatenate([qs_packed, d_bytes], axis=-1)
    return blocks_out.reshape(*orig_shape[:-1], (orig_shape[-1] // QK_STD), blocks_out.shape[-1])


def _test():
    rng = np.random.default_rng(42)
    E, OUT, IN = 4, 8, 512  # piccolo, come i test precedenti di stanotte
    ng = IN // QK_STD  # 2

    # simula l'uscita di forgia_batch_no_rot: D e trit GIA' ottimizzati
    D = rng.uniform(0.01, 0.5, size=(E, OUT, ng)).astype(np.float32)
    trit = rng.integers(-1, 2, size=(E, OUT, IN)).astype(np.int8)

    packed = pack_tq1_0_da_d_q(D, trit)
    print(f"packed shape: {packed.shape} (atteso: {E},{OUT},{ng},54)")

    # round-trip con la dequantize VERA della libreria (non una mia copia)
    flat_blocks = packed.reshape(-1, packed.shape[-1])
    dequant = TQ1_0.dequantize_blocks(flat_blocks)  # (n_blocks, 256) float32

    atteso = (trit.astype(np.float32) * D.repeat(QK_STD, axis=-1)).reshape(-1, QK_STD)
    diff = np.abs(dequant - atteso)
    print(f"round-trip vs (D*trit) atteso: max_diff={diff.max():.6e} mean_diff={diff.mean():.6e}")
    # d in fp16 -> piccolo errore di arrotondamento atteso, NON zero esatto
    ok = diff.max() < 1e-3
    print("✅ PACKER CORRETTO (round-trip via libreria vera)" if ok else "⛔⛔⛔ PACKER ROTTO")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if _test() else 1)
