"""Usage: cmp_gguf_tensors.py PRODUCED.gguf RECORDED.gguf — tensor-by-tensor byte comparison and metadata field diff
(used to explain a sha256 mismatch). Needs numpy and the gguf-py of thor-rexppiani2."""
import sys, numpy as np
sys.path.insert(0,'/mnt/models/build/thor-rexppiani2/gguf-py')
from gguf import GGUFReader
a=GGUFReader(sys.argv[1]); b=GGUFReader(sys.argv[2])
ta={t.name:t for t in a.tensors}; tb={t.name:t for t in b.tensors}
print(len(ta),len(tb), set(ta)^set(tb) and list(set(ta)^set(tb))[:3])
bad=[n for n in ta if n in tb and (ta[n].data_offset-tb[n].data_offset or not np.array_equal(np.asarray(ta[n].data).view(np.uint8),np.asarray(tb[n].data).view(np.uint8)))]
print('tensors differing',len(bad),bad[:5])
for k in set(a.fields)|set(b.fields):
    fa=a.fields.get(k); fb=b.fields.get(k)
    va=None if fa is None else bytes(fa.parts[-1]) if len(fa.parts) else None
    vb=None if fb is None else bytes(fb.parts[-1]) if len(fb.parts) else None
    if va!=vb: print('field',k,(va or b'')[:200],(vb or b'')[:200])
for n in bad:
    x=np.asarray(ta[n].data).view(np.uint8).ravel(); y=np.asarray(tb[n].data).view(np.uint8).ravel()
    d=np.nonzero(x!=y)[0]
    print(n, x.size, 'differing bytes', d.size, f'{d.size/x.size:.2e}', 'first', d[:3])
