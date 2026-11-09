"""Mini-prova del percorso di scrittura GGUF di assembla.py: 3 tensori (TQ1_0R_1024 e TQ1_0 dal manifesto, un F32
copiato da EB3) in un GGUF giocattolo, riletti: tipo, forma in elementi, byte identici."""
import sys, hashlib, numpy as np, json, os
sys.path.insert(0, '@@ROOT:models@@/build/llamacpp-tq1-k1024/gguf-py')
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType as T, GGML_QUANT_SIZES
print('GGML_QUANT_SIZES TQ1_0R_1024', GGML_QUANT_SIZES[T.TQ1_0R_1024], 'TQ1_0', GGML_QUANT_SIZES[T.TQ1_0])
man = json.load(open('@@ROOT:depot@@/tutto_ternario/tensori/manifesto.json'))
eb = {t.name: t for t in GGUFReader('@@ROOT:models@@/gguf/tony-tern/Thor-T-EB3-passo_000020.gguf').tensors}
w = GGUFWriter(path=None, arch='qwen35moe')
voci = []
for nome in ['blk.3.attn_k.weight', 'blk.0.ffn_down_shexp.weight']:
    m = man[nome]; p = np.load(f'@@ROOT:depot@@/tutto_ternario/tensori/{nome}.npy')
    tipo = T.TQ1_0R_1024 if m['misure']['formato'] == 'TQ1_0R_1024' else T.TQ1_0
    ne0, ne1 = m['forma_ggml']; qk, nb = GGML_QUANT_SIZES[tipo]
    fb = [ne1, ne0 // qk * nb]
    w.add_tensor_info(nome, fb, np.dtype(np.uint8), int(p.size), raw_dtype=tipo); voci.append((nome, p.reshape(fb), tipo, [ne0, ne1]))
t = eb['blk.0.attn_norm.weight']; raw = np.asarray(t.data)
w.add_tensor_info(t.name, list(raw.shape), raw.dtype, int(raw.nbytes), raw_dtype=t.tensor_type); voci.append((t.name, raw, t.tensor_type, [int(x) for x in t.shape]))
out = '@@ROOT:depot@@/tutto_ternario/prova_scrittura.gguf'
w.open_output_file(out); w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
for _, d, _, _ in voci: w.write_tensor_data(np.ascontiguousarray(d))
w.close()
r = {t.name: t for t in GGUFReader(out).tensors}
ok = True
for nome, d, tipo, forma in voci:
    t = r[nome]
    same = hashlib.sha256(np.ascontiguousarray(np.asarray(t.data)).tobytes()).hexdigest() == hashlib.sha256(np.ascontiguousarray(d).tobytes()).hexdigest()
    print(nome, t.tensor_type.name, [int(x) for x in t.shape], 'atteso', tipo.name, forma, 'byte uguali', same)
    ok &= same and t.tensor_type == tipo and [int(x) for x in t.shape] == forma
os.remove(out); print('OK' if ok else 'ROSSO')
