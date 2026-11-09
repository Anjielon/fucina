"""Compiti MMLU del banco (mc_v2/mmlu_00000.bin, zero-shot) impacchettati in sequenze da 1024 token con --segmenti:
scelti fra quelli che EMA16 azzecca e K01 sbaglia sul Dell (e alcuni che entrambi azzeccano)."""
import struct, json, numpy as np
from transformers import AutoTokenizer
f = open('@@ROOT:forge@@/mc_v2/mmlu_00000.bin', 'rb').read()
n = struct.unpack_from('<I', f, 0)[0]
off = struct.unpack_from(f'<{n}I', f, 4)
def task(i):
    p = off[i]
    l = struct.unpack_from('<I', f, p)[0]; ctx = f[p+4:p+4+l].decode(); p += 4 + l
    k = struct.unpack_from('<I', f, p)[0]; p += 4
    seq = []
    for _ in range(k):
        l = struct.unpack_from('<I', f, p)[0]; seq.append(f[p+4:p+4+l].decode()); p += 4 + l
    lab = struct.unpack_from(f'<{k}i', f, p)
    return ctx, seq, lab
def mclp(fn):
    R = {}
    for r in open(fn, errors='ignore'):
        if r.startswith('MCLP\t'):
            q = r.rstrip('\n').split('\t'); R[int(q[1])] = (int(q[2]), int(q[3]))
    return R
A = '@@ROOT:models@@/gguf/tony-tern/mmlu_logprob/aragorn/'
E, K = mclp(A + 'EB2-migliore_ema_000016.log'), mclp(A + 'rifint-R2039d4v-K01.log')
tk = AutoTokenizer.from_pretrained('@@ROOT:models@@/gguf/thor-src-hf')
ctx0, seq0, lab0 = task(0)
print(repr(ctx0[-30:]), seq0, lab0, [tk.encode(s, add_special_tokens=False) for s in seq0])
perse = [i for i in range(n) if E[i][0] == E[i][1] and K[i][0] != K[i][1]]
tenute = [i for i in range(n) if E[i][0] == E[i][1] and K[i][0] == K[i][1]]
rng = np.random.default_rng(20260928)
scelte = list(rng.permutation(perse)) + list(rng.permutation(tenute))
files, cur, seg, meta, fm = [], [], [], [], []
q_p = q_t = 0
for i in scelte:
    i = int(i)
    tipo = 'persa' if i in perse else 'tenuta'
    if (tipo == 'persa' and q_p >= 16) or (tipo == 'tenuta' and q_t >= 8):
        continue
    ctx, seq, lab = task(i)
    ids = tk.encode(ctx, add_special_tokens=False)
    if len(ids) > 400:
        continue
    if len(cur) + len(ids) > 1024:
        if len(files) >= 5: break
        files.append((cur, seg, fm)); cur, seg, fm = [], [], []
    seg += [1] + [0] * (len(ids) - 1); fm.append({"compito": i, "tipo": tipo, "ultima": len(cur) + len(ids) - 1,
                                                  "giusta": lab.index(1), "dell_ema16": E[i][1], "dell_k01": K[i][1]})
    cur += ids
    q_p += tipo == 'persa'; q_t += tipo == 'tenuta'
if cur: files.append((cur, seg, fm))
out = []
for j, (c, s, fm) in enumerate(files):
    pad = 1024 - len(c)
    # riempitivo: un segmento finale di token di a-capo (non valutato) per arrivare a 1024
    c = c + [tk.encode('\n', add_special_tokens=False)[0]] * pad; s = s + ([1] + [0] * (pad - 1) if pad else [])
    np.array(c, dtype=np.int32).tofile(f'token/zs{j}.i32'); np.array(s, dtype=np.int32).tofile(f'token/zs{j}_seg.i32')
    out.append({"file": f"zs{j}", "compiti": fm})
json.dump(out, open('token/zero_shot.json', 'w'), indent=1)
print([(o['file'], len(o['compiti']), sum(c['tipo']=='persa' for c in o['compiti'])) for o in out])
