"""Patch TEMPO «beta senza sincronizzazione» (leva L4 del piano) per quant_ternaria_globale_batch_v2.py — 25/9.

`TernarioDiffBatchFuso.forward_duro` (e `forward`) costruiscono a OGNI chiamata
`beta_t = torch.as_tensor(beta, dtype, device)`: su GPU e' una copia host->device da memoria
paginabile, che PyTorch fa seguire da una sincronizzazione dello stream. Quattro chiamate per
passo (gate_up e down dei due strati): py-spy della corsa conf6b-S1 vede il thread principale
fermo su quella riga 42-56 ms per passo (attese «a», rapporto costo_passo.md §2.2) — tempo in
cui la GPU lavora comunque, ma dopo ogni sincronizzazione la CPU non ha nulla in coda e la GPU
resta un attimo scarica. Qui il tensore 0-dim si crea UNA volta per (valore, dtype, device) e
si riusa: stesso valore, stesso tipo, stesso device → il nucleo compilato riceve lo stesso
ingresso e scrive gli stessi bit. Un beta gia' tensore (mai in produzione) passa come oggi.

Guadagno NON MISURATO (stima 2-15 ms/passo: bolle dopo 4 sincronizzazioni su ~20 per passo).
NUMERI: IDENTICI AL BIT. Nessun interruttore: il valore e' lo stesso per costruzione.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from patch_tempo_comune import esegui_patch  # noqa: E402

MARCA = "patch_tempo_beta_costante"

QUANT = [
    ('''class TernarioDiffBatchFuso(TernarioDiffBatch):''',
     '''# ⭐ 25/9 (patch_tempo_beta_costante): beta 0-dim creato UNA volta per (valore, dtype, device) —
# torch.as_tensor(float, device=cuda) e' una copia host->device con sincronizzazione dello stream
import math as _math  # noqa: E402

_BETA_TENSORI: dict = {}


def _beta_tensore(beta, dtype: torch.dtype, device) -> torch.Tensor:
    if isinstance(beta, torch.Tensor) or not _math.isfinite(float(beta)):
        return torch.as_tensor(beta, dtype=dtype, device=device)          # come prima, nessuna cache
    chiave = (float(beta), dtype, torch.device(device))
    t = _BETA_TENSORI.get(chiave)
    if t is None:
        t = torch.as_tensor(beta, dtype=dtype, device=device)
        _BETA_TENSORI[chiave] = t
    return t


class TernarioDiffBatchFuso(TernarioDiffBatch):'''),
    ('''    def forward(self, beta: float) -> torch.Tensor:
        beta_t = torch.as_tensor(beta, dtype=self.z.dtype, device=self.z.device)''',
     '''    def forward(self, beta: float) -> torch.Tensor:
        beta_t = _beta_tensore(beta, self.z.dtype, self.z.device)'''),
    ('''    def forward_duro(self, beta_surrogato: float, uscita_bf16: bool = False) -> torch.Tensor:
        beta_t = torch.as_tensor(beta_surrogato, dtype=self.z.dtype, device=self.z.device)''',
     '''    def forward_duro(self, beta_surrogato: float, uscita_bf16: bool = False) -> torch.Tensor:
        beta_t = _beta_tensore(beta_surrogato, self.z.dtype, self.z.device)'''),
]

PIANO = {"quant_ternaria_globale_batch_v2.py": QUANT}

if __name__ == "__main__":
    raise SystemExit(esegui_patch(__doc__.splitlines()[0], PIANO, MARCA))
