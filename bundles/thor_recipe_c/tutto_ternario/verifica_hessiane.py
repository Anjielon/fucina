"""Prova analitica (20/9) che una Hessiana e' sul punto giusto: dopo una RMSNorm con moltiplicatore m,
sum_i diag(H)_i / m_i^2 = N * n (N campioni, n = 2048). Negativo: col moltiplicatore di un altro strato il conto sbaglia."""
import sys, numpy as np
sys.path.insert(0, '.')
from sorgente import SorgenteHF, bf16_a_f32
src = SorgenteHF()
HT = '@@ROOT:models@@/gguf/tony-tern/'
def m(nome_hf):
    u, _ = src.leggi_hf(nome_hf); return 1.0 + bf16_a_f32(u).astype(np.float64)
P = 'model.language_model.'
for L in (0, 3, 20, 39):
    for cart, norma in (('hessiane-attn', 'input_layernorm'), ('hessiane', 'post_attention_layernorm')):
        H = np.load(f'{HT}{cart}/H_{L:02d}.npy', mmap_mode='r'); dg = np.diag(H).astype(np.float64)
        giusto = (dg / m(f'{P}layers.{L}.{norma}.weight') ** 2).sum() / 2048
        altro = (dg / m(f'{P}layers.{(L + 1) % 40}.{norma}.weight') ** 2).sum() / 2048
        print(f'strato {L:2d} {cart:14s} N stimato {giusto:10.6f}   (col moltiplicatore dello strato {(L+1)%40}: {altro:10.6f})')
H = np.load(HT + 'hessiane-output/H.npy', mmap_mode='r'); dg = np.diag(H).astype(np.float64)
print('output          N stimato', (dg / m(P + 'norm.weight') ** 2).sum() / 2048, '  (con input_layernorm 39:', (dg / m(P + 'layers.39.input_layernorm.weight') ** 2).sum() / 2048, ')')
