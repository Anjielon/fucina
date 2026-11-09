#!/usr/bin/env python3
"""3/10 — GGUF con gli ESPERTI DI RECUPERO: R2B3m AL BYTE + tensori blk.L.ffn_{gate_inp_rexp*,gate_rexps*,up_rexps*,
down_rexps*}.weight (tutti TQ1_0) da `rexp_tq1.npz` di e2e_recupero, + chiave qwen35moe.expert_recovery_used_count = k_r.
Rilettura e verifica (sha per tensore, tetto 9 GiB). Motore: ramo esperti-recupero di thor-velocita (THOR_SPEGNI=rexp).

  python costruisci_recupero.py --npz DIR/rexp_tq1.npz --esportazione DIR/esportazione.json --out @@ROOT:models@@/.../X.gguf
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, '@@ROOT:models@@/build/llamacpp-qat2/gguf-py')
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType as QT, GGUFValueType as VT

BASE = '@@ROOT:models@@/gguf/tony-tern/R2B3m/Thor-T-R2B3m.gguf'
SHA_BASE = '56cb5bcdd9813a1015a7cad5364b5a80226f990fb8bd561b7cae4727d70efc43'
TETTO = 9 * 2**30
CHIAVE_K = 'qwen35moe.expert_recovery_used_count'
CHIAVE_N = 'qwen35moe.expert_recovery_count'


def sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def costruisci(base: str, nuovi: dict, k_r: int, out: str, sha_base=SHA_BASE, nota: str = '') -> dict:
    assert not out.startswith('/tmp'), 'mai file grandi in /tmp'
    if sha_base:
        assert sha_file(base) == sha_base, 'sha base'
    r = GGUFReader(base)
    tb = {t.name: t for t in r.tensors}
    for n, arr in nuovi.items():
        assert n not in tb and arr.dtype == np.uint8 and arr.shape[-1] % 54 == 0, n
    w = GGUFWriter(path=None, arch='qwen35moe')
    for f in r.fields.values():
        if f.name.startswith(('GGUF.', 'split.')):
            continue
        w.add_key_value(f.name, f.contents(), f.types[0], sub_type=f.types[-1] if len(f.types) > 1 else None)
    n_r = {int(a.shape[0]) for n, a in nuovi.items() if '_rexps' in n}
    assert len(n_r) == 1, n_r
    n_r = n_r.pop()
    w.add_key_value(CHIAVE_N, n_r, VT.UINT32)
    w.add_key_value(CHIAVE_K, int(k_r), VT.UINT32)
    w.add_string('thor.esperti_recupero', f'R2B3m (sha256 {sha_base}) al byte + esperti di recupero (3/10): '
                 f'{len(nuovi)} tensori TQ1_0, k_r={k_r}. {nota}')
    piano, prov = [], {}
    for t in r.tensors:
        raw = np.asarray(t.data)
        w.add_tensor_info(t.name, list(raw.shape), raw.dtype, int(raw.nbytes), raw_dtype=t.tensor_type)
        piano.append(raw); prov[t.name] = {'azione': 'copiato al byte dalla base', 'tipo': t.tensor_type.name,
                                           'sha256_byte': sha(raw)}
    for n, arr in sorted(nuovi.items()):
        w.add_tensor_info(n, list(arr.shape), arr.dtype, int(arr.nbytes), raw_dtype=QT.TQ1_0)
        piano.append(arr); prov[n] = {'azione': 'nuovo (esperti di recupero)', 'tipo': 'TQ1_0', 'sha256_byte': sha(arr)}
    tmp = out + '.parziale'; Path(out).parent.mkdir(parents=True, exist_ok=True)
    w.open_output_file(Path(tmp)); w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
    for dati in piano:
        w.write_tensor_data(np.ascontiguousarray(dati))
    w.close(); os.replace(tmp, out)
    r2 = GGUFReader(out); err = []
    for t2 in r2.tensors:
        p = prov.get(t2.name)
        if p is None or sha(np.asarray(t2.data)) != p['sha256_byte'] or t2.tensor_type.name != p['tipo']:
            err.append(t2.name)
    if len(r2.tensors) != len(prov): err.append('numero tensori')
    if int(r2.fields[CHIAVE_K].contents()) != k_r or int(r2.fields[CHIAVE_N].contents()) != n_r: err.append('chiavi')
    taglia = os.path.getsize(out)
    if taglia > TETTO: err.append(f'oltre 9 GiB: {taglia}')
    ver = {'errori': err, 'ok': not err, 'n_tensori': len(r2.tensors), 'n_nuovi': len(nuovi),
           'byte_nuovi': int(sum(v.nbytes for v in nuovi.values())), 'taglia_file': taglia, 'margine': TETTO - taglia,
           'k_r': k_r, 'argv': sys.argv}
    if sha_base:
        ver['sha256_file'] = sha_file(out)
    json.dump(prov, open(out + '.provenienza.json', 'w'), indent=1)
    json.dump(ver, open(out + '.verifica.json', 'w'), indent=1)
    return ver


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True); ap.add_argument('--esportazione', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--base', default=BASE)
    a = ap.parse_args()
    e = json.load(open(a.esportazione))
    z = np.load(a.npz)
    ver = costruisci(a.base, {k: z[k] for k in z.files}, int(e['k_r']), a.out,
                     nota=f"passo {e['passo']}, KL held {e['kl_held']:.5f} (da {e['kl0']:.5f})")
    log('VERIFICA', 'OK' if ver['ok'] else ver['errori'], json.dumps({k: v for k, v in ver.items() if k != 'argv'}))
    sys.exit(0 if ver['ok'] else 1)


if __name__ == '__main__':
    main()
