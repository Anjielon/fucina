"""1/10 sera — QAT2: R2B3m + parametri della Prova 2 (QAT leggero, pod, checkpoint step 15) in un GGUF.

Sorgente parametri: ~/claude-pod-lavoro/ricerca/prova2-qat/misure/parametri_selezionati.npz (251 vettori fp32,
267.128 parametri) e la loro semantica letta in qat_leggero.py (attach_student / routed_expert_math / final_logits):

  famiglia (pod)              semantica nel pod                                   -> GGUF
  norm_logscale__lL_*  (130)  uscita RMSNorm * exp(s).to(bf16)  [uscita bf16!]    -> NIENTE (vedi sotto)
  final_norm_logscale  (1)    normalized*(1+w)*exp(s) in fp32                     -> output_norm.weight *= exp(s)
  router_bias__L       (40)   logit router + b PRIMA del softmax (fp32)           -> NUOVO blk.L.ffn_gate_inp.bias {256}
  expert_bias__L       (40)   somma pesata esperti + b (fp32)                     -> NUOVO blk.L.ffn_down_exps.bias {2048,256}
                                                                                     (stessa riga per esperto: pesi normalizzati)
  log_group_scale__L   (40)   gate*exp(s0), up*exp(s1) prima di SiLU; y*exp(s2)  -> NUOVI blk.L.ffn_{gate,up,down}_exps.scale {256}
                                                                                     (caricati dal passo generico ".scale" di llama.cpp)

Norme di blocco: nel pod l'uscita delle RMSNorm HF e' bf16 (type_as(x)), e exp(s) viene convertito a bf16 prima del
prodotto; con |s| <= 0,0015 exp(s) in bf16 vale ESATTAMENTE 1,0 (passo bf16 vicino a 1 = 2^-7): il modello MISURATO
(+1,50 sul 2000 disgiunto) non le applicava. Lo script lo VERIFICA su tutti i 130 vettori e lascia quei tensori al byte;
se anche un solo elemento desse != 1 in bf16, si ferma.

Nessuna permutazione di teste coinvolta: i soli tensori densi toccati sono vettori per-esperto o output_norm
(asse n_embd, stesso ordine HF/GGUF: lo script lo verifica confrontando output_norm GGUF con 1+w HF).

--neutro: stessi tensori NUOVI ma neutri (bias 0, scale 1, output_norm al byte) -> controllo che la patch llama.cpp
sia l'identita' (logit al bit con R2B3m).
Tutto il resto: identico al byte a R2B3m (sha256 per tensore confrontato con la provenienza di R2B3m e coi byte letti).
"""
import argparse, hashlib, json, os, struct, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, '@@ROOT:models@@/build/llamacpp-qat2/gguf-py')
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType as QT

BASE = '@@ROOT:models@@/gguf/tony-tern/R2B3m/Thor-T-R2B3m.gguf'
SHA_BASE = '56cb5bcdd9813a1015a7cad5364b5a80226f990fb8bd561b7cae4727d70efc43'
NPZ = '@@ROOT:home@@/claude-pod-lavoro/ricerca/prova2-qat/misure/parametri_selezionati.npz'
HF = Path('@@ROOT:models@@/gguf/thor-src-hf')
N_LAYER, N_EXP, N_EMBD = 40, 256, 2048


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def bf16_round(x32):
    u = np.ascontiguousarray(x32, dtype=np.float32).view(np.uint32).astype(np.uint64)
    u = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) << 16          # round-to-nearest-even come torch
    return u.astype(np.uint32).view(np.float32)


def hf_tensor(key):
    idx = json.load(open(HF / 'model.safetensors.index.json'))['weight_map']
    fn = HF / idx[key]
    with open(fn, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]; h = json.loads(f.read(n)); base = 8 + n
        e = h[key]; a, b = e['data_offsets']; f.seek(base + a); buf = f.read(b - a)
    assert e['dtype'] == 'BF16'
    return (np.frombuffer(buf, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(e['shape'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--neutro', action='store_true')
    a = ap.parse_args()
    OUT = a.out
    assert not OUT.startswith('/tmp'), 'mai file grandi in /tmp'
    s = sha_file(BASE); assert s == SHA_BASE, s; log('base sha ok')
    P = dict(np.load(NPZ))
    assert len(P) == 251 and sum(v.size for v in P.values()) == 267128, (len(P), sum(v.size for v in P.values()))

    # 1) norme di blocco: nel modello misurato exp(s) in bf16 == 1 ovunque?
    nk = [k for k in P if k.startswith('norm_logscale__')]
    assert len(nk) == 130
    non_uno = {k: int((bf16_round(np.exp(P[k].astype(np.float32))) != 1.0).sum()) for k in nk}
    tot_non_uno = sum(non_uno.values())
    log('norme di blocco: elementi con bf16(exp(s)) != 1 :', tot_non_uno, '| max|s| =',
        max(float(np.abs(P[k]).max()) for k in nk))
    assert tot_non_uno == 0, {k: v for k, v in non_uno.items() if v}

    r = GGUFReader(BASE)
    tb = {t.name: t for t in r.tensors}
    prov_base = json.load(open(BASE + '.provenienza.json'))

    # 2) output_norm: convenzione GGUF = 1 + w_HF, stesso ordine (niente permutazione)?
    on = np.asarray(tb['output_norm.weight'].data, dtype=np.float32).copy()
    hf_on = hf_tensor('model.language_model.norm.weight')
    # R2B3m ha output_norm ritoccata dalla forgia (rel <= 4,4e-4 misurato 1/10): si esige stesso ordine e convenzione 1+w
    # (rel < 2e-3, contro ~1 per w senza +1); exp(s) e' un fattore MOLTIPLICATIVO sull'uscita, si applica al valore GGUF.
    d1 = float(np.abs(on / (1.0 + hf_on) - 1).max()); d0 = float(np.abs(on - hf_on).max())
    log(f'output_norm: max rel |gguf/(1+w_hf)-1| = {d1:.3e}   max|gguf-w_hf| = {d0:.3e}')
    assert d1 < 2e-3 and d0 > 0.5, 'output_norm GGUF non e\' 1+w_HF nello stesso ordine: convenzione da capire'
    fs = P['final_norm_logscale'].astype(np.float32); assert fs.shape == (N_EMBD,)

    nuovi = {}   # nome -> array numpy (forma numpy = ggml rovesciata)
    for L in range(N_LAYER):
        rb = P[f'router_bias__{L}'].astype(np.float32); eb = P[f'expert_bias__{L}'].astype(np.float32)
        gs = P[f'log_group_scale__{L}'].astype(np.float32)
        assert rb.shape == (N_EXP,) and eb.shape == (N_EMBD,) and gs.shape == (3,)
        assert tb[f'blk.{L}.ffn_gate_inp.weight'].shape.tolist() == [N_EMBD, N_EXP]
        assert tb[f'blk.{L}.ffn_down_exps.weight'].shape.tolist()[1:] == [N_EMBD, N_EXP]
        if a.neutro:
            rb = np.zeros_like(rb); eb = np.zeros_like(eb); gs = np.zeros_like(gs)
        nuovi[f'blk.{L}.ffn_gate_inp.bias'] = rb
        nuovi[f'blk.{L}.ffn_down_exps.bias'] = np.ascontiguousarray(np.broadcast_to(eb, (N_EXP, N_EMBD)))
        for j, fam in enumerate(('gate', 'up', 'down')):
            nuovi[f'blk.{L}.ffn_{fam}_exps.scale'] = np.full((N_EXP,), np.exp(gs[j]), dtype=np.float32)
    for n in nuovi:
        assert n not in tb, n

    sostituiti = {}
    if not a.neutro:
        sostituiti['output_norm.weight'] = (on * np.exp(fs)).astype(np.float32)

    w = GGUFWriter(path=None, arch='qwen35moe')
    for f in r.fields.values():
        if f.name.startswith(('GGUF.', 'split.')):
            continue
        w.add_key_value(f.name, f.contents(), f.types[0], sub_type=f.types[-1] if len(f.types) > 1 else None)
    w.add_string('thor.QAT2', ('QAT2' + (' NEUTRO (controllo: bias 0, scale 1)' if a.neutro else '') +
                               f': R2B3m (sha256 {SHA_BASE}) al byte + Prova 2 QAT leggero (pod 1/10, step 15): '
                               'output_norm*=exp(final_norm_logscale); nuovi ffn_gate_inp.bias, ffn_down_exps.bias, '
                               'ffn_{gate,up,down}_exps.scale per strato 0-39; norme di blocco invariate '
                               '(nel modello misurato exp(s) in bf16 = 1). Serve llama.cpp ramo qat2-bias.'))
    piano, prov = [], {}
    for t in r.tensors:
        raw = np.asarray(t.data)
        if t.name in sostituiti:
            new = sostituiti[t.name]
            assert new.shape == raw.shape and raw.dtype == np.float32 and t.tensor_type == QT.F32, t.name
            w.add_tensor_info(t.name, list(raw.shape), np.dtype(np.float32), int(new.nbytes), raw_dtype=QT.F32)
            piano.append((t.name, new))
            prov[t.name] = {'azione': 'sostituito', 'tipo': 'F32', 'sha256_byte': sha(new), 'sha256_vecchio': sha(raw),
                            'max_rel': float(np.abs(new / raw - 1).max())}
        else:
            hb = sha(raw)
            assert hb == prov_base[t.name]['sha256_byte'], ('byte base diversi dalla provenienza', t.name)
            w.add_tensor_info(t.name, list(raw.shape), raw.dtype, int(raw.nbytes), raw_dtype=t.tensor_type)
            piano.append((t.name, raw)); prov[t.name] = {'azione': 'copiato al byte da R2B3m', 'tipo': t.tensor_type.name,
                                                         'sha256_byte': hb}
    for n, arr in nuovi.items():
        w.add_tensor_info(n, list(arr.shape), np.dtype(np.float32), int(arr.nbytes), raw_dtype=QT.F32)
        piano.append((n, arr)); prov[n] = {'azione': 'nuovo', 'tipo': 'F32', 'sha256_byte': sha(arr),
                                           'forma_numpy': list(arr.shape)}
    Path(OUT).parent.mkdir(parents=True, exist_ok=True); tmp = OUT + '.parziale'
    w.open_output_file(Path(tmp)); w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
    for _, dati in piano:
        w.write_tensor_data(np.ascontiguousarray(dati))
    w.close(); os.replace(tmp, OUT); log('scritto', OUT, os.path.getsize(OUT))

    # 3) rilettura e verifica
    r2 = GGUFReader(OUT); err = []
    nomi2 = [t.name for t in r2.tensors]
    if nomi2[:len(r.tensors)] != [t.name for t in r.tensors]: err.append('nomi/ordine dei tensori base')
    if set(nomi2[len(r.tensors):]) != set(nuovi): err.append('tensori nuovi')
    for t2 in r2.tensors:
        p = prov[t2.name]
        if sha(np.asarray(t2.data)) != p['sha256_byte']: err.append('byte ' + t2.name)
        if t2.tensor_type.name != p['tipo']: err.append('tipo ' + t2.name)
        if t2.name in tb and t2.shape.tolist() != tb[t2.name].shape.tolist(): err.append('forma ' + t2.name)
    toccati = [n for n, p in prov.items() if p['azione'] == 'sostituito']
    if sorted(toccati) != sorted(sostituiti): err.append(f'sostituiti inattesi {toccati}')
    ka = {f.name for f in r.fields.values()}; kb = {f.name for f in r2.fields.values()}
    if ka - kb: err.append(f'chiavi mancanti {ka - kb}')
    ver = {'errori': err, 'ok': not err, 'neutro': a.neutro, 'n_tensori': len(r2.tensors),
           'n_copiati_al_byte': sum(p['azione'].startswith('copiato') for p in prov.values()),
           'sostituiti': {n: {k: prov[n][k] for k in ('sha256_vecchio', 'sha256_byte', 'max_rel')} for n in toccati},
           'n_nuovi': len(nuovi), 'byte_nuovi': int(sum(v.nbytes for v in nuovi.values())),
           'norme_blocco_bf16_non_uno': tot_non_uno, 'output_norm_conv_1piuw_maxdiff': d1,
           'chiavi_aggiunte': sorted(kb - ka), 'taglia_file': os.path.getsize(OUT), 'taglia_base': os.path.getsize(BASE),
           'sha256_file': sha_file(OUT)}
    json.dump(prov, open(OUT + '.provenienza.json', 'w'), indent=1)
    json.dump(ver, open(OUT + '.verifica.json', 'w'), indent=1)
    log('VERIFICA', 'OK' if ver['ok'] else err, json.dumps({k: v for k, v in ver.items() if k != 'sostituiti'}))
    sys.exit(0 if ver['ok'] else 1)


if __name__ == '__main__':
    main()
