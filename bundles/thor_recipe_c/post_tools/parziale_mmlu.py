#!/usr/bin/env python3
# parziale MMLU-2000: accuratezza sulle domande gia' fatte + T400 sulle STESSE domande + vinte/perse.
# uso: parziale_mmlu.py LOG [RIF] [--indice FILE.indice.tsv] [--rimappa USCITA.log]
# 26/10: --indice per i banchi mescolati di stato/banco_mescolato_1026.py (mc_v2/mmlu2000_mescolato.bin,
# mc_v2/mmlu400_sonda.bin): l'indice MCLP del LOG e' la posizione nel banco mescolato e viene rimappato
# sull'indice originale di mmlu_00000.bin, cioe' sulle stesse domande di T400 (RIF, sempre nell'ordine originale).
# --rimappa scrive le righe MCLP con indice originale e in ordine originale (+ "Final estimate"): con le 2000
# complete il file e' accettato da thor_v2/strumenti/mcnemar_mclp.py al posto del log mescolato.
import argparse, collections, sys
ap = argparse.ArgumentParser()
ap.add_argument('log'); ap.add_argument('rif', nargs='?',
    default='@@ROOT:models@@/gguf/tony-tern/OBIETTIVO_C/CARDAN_CONGIUNTO_S7T550_1020/tappa_400/mmlu_CONGIUNTO.log')
ap.add_argument('--indice'); ap.add_argument('--rimappa'); x = ap.parse_args()

def righe(p):
    d = {}
    for r in open(p, errors='ignore'):
        if r.startswith('MCLP\t'):
            d[int(r.split('\t')[1])] = r
    return d
def leggi(d):
    return {i: (int(r.split('\t')[2]), int(r.split('\t')[3])) for i, r in d.items()}

RA = righe(x.log); R = leggi(righe(x.rif)); tot, mat = 2000, {}
if x.indice:
    mappa = {}
    for r in open(x.indice):
        if r.startswith('#') or not r.strip():
            continue
        c = r.rstrip('\n').split('\t'); mappa[int(c[0])] = int(c[1]); mat[int(c[1])] = c[2]
    tot = len(mappa)
    fuori = [j for j in RA if j not in mappa]
    if fuori:
        sys.exit(f'⛔ {len(fuori)} indici MCLP del log non sono nell\'indice {x.indice} (banco sbagliato?)')
    RA = {mappa[j]: r for j, r in RA.items()}
elif x.rimappa:
    sys.exit('⛔ --rimappa richiede --indice')
A = leggi(RA)
k = sorted(i for i in A if i in R)
if not k:
    sys.exit('nessuna domanda in comune con il riferimento')
dis = sum(A[i][0] != R[i][0] for i in k)
if dis:
    sys.exit(f'⛔ {dis}/{len(k)} domande con risposta giusta diversa dal riferimento: indice/banco non coerenti')
ga = sum(A[i][0] == A[i][1] for i in k); gr = sum(R[i][0] == R[i][1] for i in k)
v = sum(A[i][0] == A[i][1] and R[i][0] != R[i][1] for i in k); p = sum(A[i][0] != A[i][1] and R[i][0] == R[i][1] for i in k)
print(f"{len(k)}/{tot} domande: {100*ga/len(k):.2f} vs T400 {100*gr/len(k):.2f} sulle stesse ({100*(ga-gr)/len(k):+.2f}), vinte/perse {v}/{p}")
if mat:
    c = collections.Counter(mat[i] for i in k)
    print(f"  materie coperte {len(c)}/{len(set(mat.values()))}: " + ', '.join(f'{s} {c[s]}' for s in sorted(c)))
if x.rimappa:
    with open(x.rimappa, 'w') as f:
        for i in sorted(RA):
            c = RA[i].split('\t'); c[1] = str(i); f.write('\t'.join(c))
        if len(RA) == tot:
            f.write(f'Final estimate: {100*sum(a == b for a, b in A.values())/len(A):.4f} (rimappato da {x.log})\n')
    print(f"  rimappato: {x.rimappa} ({len(RA)} righe MCLP in ordine originale)")
