"""
Correttezza END-TO-END, solo CPU/numpy, PRIMA di toccare la GPU (regola del
progetto: misura, non supporre).

Pipeline per un tensore denso vero di Tony (Ornith-1.5-35B-A3B):
  1. legge blk.10.attn_gate (4096x2048, Q6_K nel file) con gguf-py, dequantizzato
  2. per ogni riga, per ogni blocco di 256 lungo le colonne (l'ingresso):
       BASE:    ternarizza direttamente (quantize_block_tq1_0) -> dequant
       RUOTATO: FWHT(w) -> ternarizza -> dequant -> FWHT (de-rotazione)
  3. genera un vettore x casuale (proxy dell'attivazione — qui NON serve la
     Hessiana: quella leva e' gia' misurata nel documento di ricerca; questo
     test verifica solo che IL CODICE (encoder/decoder/rotazione) sia corretto)
  4. confronta full-precision W*x  vs  W_hat_base*x  vs  W_hat_rot*x
"""
import sys
import numpy as np

sys.path.insert(0, "@@ROOT:home@@/build-llamacpp-tq1/gguf-py")
import gguf  # noqa: E402

from tq1_codec import quantize_block_tq1_0, dequantize_block_tq1_0  # noqa: E402
from fwht import fwht_shader_convention  # noqa: E402

GGUF_PATH = "@@ROOT:models@@/gguf/ornith-1.5/Ornith-1.5-35B-A3B-APEX-MTP-I-Quality.gguf"
TENSOR_NAME = "blk.10.attn_gate.weight"
N_ROWS = 256          # sottocampione di righe (il tensore ne ha 4096)
SEED = 20260919


def carica_tensore(nome: str) -> np.ndarray:
    reader = gguf.GGUFReader(GGUF_PATH)
    for t in reader.tensors:
        if t.name == nome:
            print(f"[verifica] {t.name}  quant={t.tensor_type.name}  shape={t.shape}")
            w = gguf.quants.dequantize(t.data, t.tensor_type)
            return np.asarray(w, dtype=np.float32)
    raise SystemExit(f"tensore {nome} non trovato")


def pipeline_riga(row: np.ndarray, ruota: bool) -> np.ndarray:
    """row: (C,) float32, C multiplo di 256 -> W_hat ricostruita, stessa forma."""
    C = row.shape[0]
    assert C % 256 == 0
    out = np.zeros_like(row)
    for b in range(0, C, 256):
        blk = row[b:b + 256]
        if ruota:
            blk_r = fwht_shader_convention(blk)
            enc = quantize_block_tq1_0(blk_r)
            dec = dequantize_block_tq1_0(enc)
            out[b:b + 256] = fwht_shader_convention(dec)   # de-rotazione (involutoria)
        else:
            enc = quantize_block_tq1_0(blk)
            out[b:b + 256] = dequantize_block_tq1_0(enc)
    return out


def main():
    W = carica_tensore(TENSOR_NAME)
    R, C = W.shape
    assert C % 256 == 0, f"colonne={C} non multiplo di 256"
    rng = np.random.default_rng(SEED)
    rows_idx = rng.choice(R, size=min(N_ROWS, R), replace=False)
    x = rng.standard_normal(C).astype(np.float32)

    y_true = np.zeros(len(rows_idx), dtype=np.float64)
    y_base = np.zeros(len(rows_idx), dtype=np.float64)
    y_rot = np.zeros(len(rows_idx), dtype=np.float64)
    mse_w_base = []
    mse_w_rot = []

    for k, r in enumerate(rows_idx):
        row = W[r]
        w_base = pipeline_riga(row, ruota=False)
        w_rot = pipeline_riga(row, ruota=True)
        y_true[k] = float(np.dot(row.astype(np.float64), x.astype(np.float64)))
        y_base[k] = float(np.dot(w_base.astype(np.float64), x.astype(np.float64)))
        y_rot[k] = float(np.dot(w_rot.astype(np.float64), x.astype(np.float64)))
        mse_w_base.append(float(np.mean((row - w_base) ** 2)))
        mse_w_rot.append(float(np.mean((row - w_rot) ** 2)))

    err_base = np.abs(y_base - y_true)
    err_rot = np.abs(y_rot - y_true)
    rel_base = err_base / (np.abs(y_true) + 1e-8)
    rel_rot = err_rot / (np.abs(y_true) + 1e-8)

    print(f"\n=== {TENSOR_NAME}  righe campionate={len(rows_idx)}  C={C} ({C//256} blocchi/riga) ===")
    print(f"MSE pesi   BASE   : mediana={np.median(mse_w_base):.6f}  media={np.mean(mse_w_base):.6f}")
    print(f"MSE pesi   RUOTATO: mediana={np.median(mse_w_rot):.6f}  media={np.mean(mse_w_rot):.6f}")
    delta_mse = (np.median(mse_w_rot) - np.median(mse_w_base)) / np.median(mse_w_base) * 100
    print(f"  -> delta MSE pesi (ruotato vs base): {delta_mse:+.2f}%")
    print()
    print(f"errore matvec |y_hat - y_true|   BASE   : mediana={np.median(err_base):.4f}  media={np.mean(err_base):.4f}")
    print(f"errore matvec |y_hat - y_true|   RUOTATO: mediana={np.median(err_rot):.4f}  media={np.mean(err_rot):.4f}")
    delta_out = (np.median(err_rot) - np.median(err_base)) / np.median(err_base) * 100
    print(f"  -> delta errore d'uscita (ruotato vs base): {delta_out:+.2f}%")
    print()
    print(f"errore RELATIVO matvec   BASE   : mediana={np.median(rel_base):.4f} ({np.median(rel_base)*100:.1f}%)")
    print(f"errore RELATIVO matvec   RUOTATO: mediana={np.median(rel_rot):.4f} ({np.median(rel_rot)*100:.1f}%)")
    print()
    ok = np.isfinite(y_rot).all() and np.median(rel_rot) < 1.0  # sanity: non deve esplodere
    print(f"SANITA' (nessuna NaN/inf, errore relativo < 100%): {'OK' if ok else 'FALLITA'}")


if __name__ == "__main__":
    main()
