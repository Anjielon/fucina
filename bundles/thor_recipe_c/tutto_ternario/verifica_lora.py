"""Il Delta W della LoRA (ricostruito da lora.npz e portato nell'ordine del GGUF) e' orientato giusto? Nel GGUF di EB3 il
tensore e' Q8_0(W_file + Delta W): la differenza D = deq(EB3) - deq(Q8_0(W)) deve essere ALLINEATA a Delta W
(correlazione > 0 e |D| cresce dove Delta W e' grande). Negativo: Delta W senza la permutazione (o trasposta) non correla."""
import sys, numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, '@@ROOT:models@@/build/llamacpp-tq1-k1024/gguf-py')
import gguf
from gguf import GGUFReader, GGMLQuantizationType as T
from sorgente import SorgenteHF, bf16_a_f32
src = SorgenteHF(); eb = {t.name: t for t in GGUFReader('@@ROOT:models@@/gguf/tony-tern/Thor-T-EB3-passo_000020.gguf').tensors}
for nome in ['blk.0.attn_gate.weight', 'blk.0.ssm_out.weight', 'blk.21.attn_gate.weight', 'blk.21.ssm_out.weight',
             'blk.30.ssm_out.weight', 'blk.39.ffn_down_shexp.weight', 'blk.36.ffn_gate_shexp.weight']:
    u, _ = src.u16_ordine_gguf(nome); W = bf16_a_f32(u)
    dW = src.delta_lora(nome)
    q = gguf.quants.dequantize(gguf.quants.quantize(W, T.Q8_0), T.Q8_0).reshape(W.shape)
    D = gguf.quants.dequantize(np.asarray(eb[nome].data), T.Q8_0).reshape(W.shape) - q
    c = lambda a, b: float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))
    v = src.voce(nome); rel = v['rel']
    if rel == 'perm_righe':
        inv = np.argsort(src.perm[v['perm_npz']]); neg = dW[inv]
    elif rel == 'perm_colonne':
        inv = np.argsort(src.perm[v['perm_npz']]); neg = dW[:, inv]
    else:
        neg = dW[::-1]
    # proiezione: quanta parte di Delta W e' nel file (regressione D ~ a * dW)
    a = float((D * dW).sum() / (dW * dW).sum())
    print(f'{nome:32s} rel={rel:12s} corr(D,dW)={c(D, dW):+.4f}  negativo={c(D, neg):+.4f}  quota di dW nel file a={a:+.3f}  frazione elementi cambiati={float((D != 0).mean()):.4f}')
