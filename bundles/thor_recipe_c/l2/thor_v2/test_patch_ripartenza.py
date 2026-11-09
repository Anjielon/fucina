#!/usr/bin/env python3
"""Collaudo della patch strumenti/patch_ripartenza.py (24/9 sera) — CPU, scala giocattolo.

Si esegue su una COPIA patchata della cartella (la catena di stanotte importa gli
originali: non si toccano):

    cp thor_v2/*.py thor_v2/*.json thor_v2/*.npz COPIA/ ; mkdir COPIA/strumenti
    cp thor_v2/strumenti/patch_ripartenza.py COPIA/strumenti/
    python3 COPIA/strumenti/patch_ripartenza.py
    env CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 19 \\
        @@ROOT:venv@@/bin/python3 test_patch_ripartenza.py --patchata COPIA --originale thor_v2

Prove:
 A. DEFAULT INVARIATI: la stessa finestra, stesso seme, senza i nuovi argomenti,
    con l'originale e con la copia patchata (processi separati): codici, perdite e
    pesi commessi identici al bit.
 B. CATENA A PASSO 1 sulla copia: [0,1] -> [1,2] -> [2,3], una sola catena,
    ricetta della forgia veloce (solo fase dura, rotazione sull'ingresso, routing
    aggiornato, diagnostica spenta). Lo strato k+1 passa per PARAMETRI (log:
    codici identici 100%, scarto 0); nella finestra [1,2] lo strato 2 entra dai
    CODICI di una corsa precedente a finestre separate (modalita' b), letti da
    disco con verifica della marca. Seme della finestra [k+1,k+2] = seme della
    [k,k+1] passato per il SOLO strato k commesso (calcola_confine_finestra sul
    sotto-dizionario {k}); invariante: coincide col forward diretto di tutti gli
    strati 0..k commessi. Due varianti: Adam ripartito e Adam conservato.
 C. argomenti invalidi rifiutati.
"""
import argparse
import contextlib
import io
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

RICETTA = {   # come forgia_veloce_thor_t.sh, per quanto ha senso a scala giocattolo
    "THOR_VALIDA_FRAZIONE_DURA": "1.0", "THOR_ROTAZIONE_INGRESSO": "1", "THOR_NUCLEO_BF16": "1",
    "THOR_ROUTING_AGGIORNATO": "1", "THOR_ROUTING_VIVO": "0", "THOR_DIAGNOSTICA_PASSO": "0",
    "THOR_HELDOUT_OGNI": "0", "THOR_RICONGELA_OGNI": "0", "THOR_FORGIA_DOWN_TERNARIO": "1",
    "THOR_LR_Z": "0.01", "THOR_LR_LOGS": "0.01", "THOR_LR_DELTA": "0.01",
}
N_PASSI = 20
LR = 0.01


def _carica(cartella: str):
    os.environ.update(RICETTA)
    sys.path.insert(0, os.path.abspath(cartella))
    import valida_finestra_8_12 as V
    if os.path.dirname(os.path.abspath(V.__file__)) != os.path.abspath(cartella):
        raise RuntimeError(f"importato {V.__file__}, atteso da {cartella}")
    return V


def _strati(V, n: int):
    """Come _test_ottimizza_finestra_JOINT_giocattolo: teacher + corrente rumoroso, semi fissi."""
    import torch
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeDecoderLayer
    torch.manual_seed(0)
    cfg = V._costruisci_config_giocattolo(n)
    cfg._attn_implementation = "sdpa"
    P, Q = {}, {}
    for L in range(n):
        p = Qwen3_5MoeDecoderLayer(cfg, layer_idx=L)
        V._init_pesi_giocattolo(p, seed=100 + L)
        q = Qwen3_5MoeDecoderLayer(cfg, layer_idx=L)
        with torch.no_grad():
            g = torch.Generator().manual_seed(900 + L)
            for (_, a), (_, b) in zip(p.named_parameters(), q.named_parameters()):
                b.copy_(a + torch.randn(a.shape, generator=g) * 0.15)
        for x in (p, q):
            for par in x.parameters():
                par.requires_grad_(False)
        P[L], Q[L] = p.eval(), q.eval()
    emb = torch.randn(cfg.vocab_size, cfg.hidden_size)
    ids = torch.randint(0, cfg.vocab_size, (3, 2, 6))        # 2 di addestramento + 1 held-out
    return cfg, P, Q, emb, ids


def _sub(d, ks):
    return {k: d[k] for k in ks}


# ═══════════════════════ A. default invariati (un processo per cartella) ═══════════════════════
def dump(cartella: str, uscita: str) -> None:
    import numpy as np
    V = _carica(cartella)
    cfg, P, Q, emb, ids = _strati(V, 3)
    diag: list = []
    r = V.ottimizza_finestra_JOINT(_sub(Q, [0, 1]), ids, cfg, n_passi=N_PASSI, strati_teacher=_sub(P, [0, 1]),
                                   emb=emb, lr=LR, n_batch_addestramento=2, raccogli_diagnostica=diag)
    h, e = V.calcola_confine_finestra(_sub(P, [0, 1]), _sub(Q, [0, 1]), ids, cfg, emb)
    r2 = V.ottimizza_finestra_JOINT(_sub(Q, [2]), ids, cfg, n_passi=N_PASSI, strati_teacher=_sub(P, [2]),
                                    emb=emb, lr=LR, n_batch_addestramento=2, h0_esterno_per_b=h, e0_esterno_per_b=e)
    arr = {}
    for L, fam in {**r, **r2}.items():
        for f, (D, Qc) in fam.items():
            arr[f"{L}.{f}.D"], arr[f"{L}.{f}.Q"] = D, Qc
    for L in range(3):
        for n, p in Q[L].named_parameters():
            arr[f"commesso.{L}.{n}"] = p.detach().numpy()
    arr["perdite"] = np.array(diag[0]["perdite"])
    np.savez(uscita, **arr)


def prova_default(originale: str, patchata: str) -> bool:
    import numpy as np
    print("\n══ A. default invariati: originale vs copia patchata, stessa finestra, stesso seme ══", flush=True)
    with tempfile.TemporaryDirectory(prefix="test_patch_rip_") as tmp:
        uscite = []
        for c in (originale, patchata):
            u = os.path.join(tmp, f"{Path(c).name}.npz")
            subprocess.run([sys.executable, __file__, "--dump", c, u], check=True, stdout=subprocess.DEVNULL)
            uscite.append(dict(np.load(u)))
        a, b = uscite
        diversi = [k for k in a if k not in b or not np.array_equal(a[k], b[k])]
        ok = not diversi and set(a) == set(b)
        print(f"  {'✅' if ok else '⛔'} {len(a)} tensori (codici, pesi commessi, curva di perdita) identici al bit"
              + (f" — DIVERSI: {diversi[:5]}" if diversi else ""))
        return ok


# ═══════════════════════ B. catena a passo 1 sulla copia patchata ═══════════════════════
class _Tee(io.TextIOBase):
    def __init__(self, *flussi):
        self.flussi = flussi

    def write(self, s):
        for f in self.flussi:
            f.write(s)
        return len(s)

    def flush(self):
        for f in self.flussi:
            f.flush()


def _corsa_b(V, cartella_codici: str) -> None:
    """'Corsa precedente' a finestre SEPARATE [0,1],[2,3] (come ThorT-veloce), codici su disco."""
    cfg, P, Q, emb, ids = _strati(V, 4)
    h = e = None
    V.OUT_DIR, V.ID_CORSA = Path(cartella_codici), "corsa-b"
    for fin in ([0, 1], [2, 3]):
        r = V.ottimizza_finestra_JOINT(_sub(Q, fin), ids, cfg, n_passi=N_PASSI, strati_teacher=_sub(P, fin),
                                       emb=emb, lr=LR, n_batch_addestramento=2, h0_esterno_per_b=h, e0_esterno_per_b=e)
        V.scrivi_codici_su_disco(r)
        h, e = V.calcola_confine_finestra(_sub(P, fin), _sub(Q, fin), ids, cfg, emb,
                                          h0_ingresso_per_b=h, e0_ingresso_per_b=e)


def prova_catena(V, adam: str, cartella_codici: str) -> bool:
    import torch
    from ripartenza_codici import StatoStrato, carica_codici_corsa
    print(f"\n══ B. catena a passo 1 [0,1]→[1,2]→[2,3], Adam {adam} ══", flush=True)
    cfg, P, Q, emb, ids = _strati(V, 4)
    buf = io.StringIO()
    ok = True
    with contextlib.redirect_stdout(_Tee(sys.stdout, buf)):
        h = e = None            # finestra 0: dall'embedding (a scala giocattolo niente GGUF)
        stato = None
        for k in range(3):
            fin = [k, k + 1]
            rip = {} if stato is None else {k: stato}
            if k == 1:
                rip[2] = carica_codici_corsa(cartella_codici, 2, "corsa-b")       # modalita' b
            consegna = {k + 1: None} if k + 1 < 3 else None
            V.ottimizza_finestra_JOINT(_sub(Q, fin), ids, cfg, n_passi=N_PASSI, strati_teacher=_sub(P, fin),
                                       emb=emb, lr=LR, n_batch_addestramento=2,
                                       h0_esterno_per_b=h, e0_esterno_per_b=e,
                                       ripartenza_per_strato=rip or None, consegna_stato_per_strato=consegna,
                                       adam_ripartenza=adam)
            if consegna is not None:
                stato = consegna[k + 1]
                ok &= isinstance(stato, StatoStrato) and (stato.adam is not None) == (adam == "conserva")
            # seme della finestra dopo: il seme di QUESTA passato per il SOLO strato k commesso
            h, e = V.calcola_confine_finestra(_sub(P, [k]), _sub(Q, [k]), ids, cfg, emb,
                                              h0_ingresso_per_b=h, e0_ingresso_per_b=e)
            # invariante: = forward diretto degli strati 0..k commessi, dall'embedding
            hd, ed = V.calcola_confine_finestra(_sub(P, range(k + 1)), _sub(Q, range(k + 1)), ids, cfg, emb)
            for b in hd:
                rt = ((h[b] - hd[b]).norm() / hd[b].norm()).item()
                rq = (((h[b] + e[b]) - (hd[b] + ed[b])).norm() / (hd[b] + ed[b]).norm()).item()
                bene = rt < 1e-6 and rq < 1e-5
                ok &= bene
                print(f"  {'✅' if bene else '⛔'} seme dopo lo strato {k} (batch {b}): catena vs diretto "
                      f"teacher {rt:.1e} · quant {rq:.1e}")
    righe = [r for r in buf.getvalue().splitlines() if "ripartenza strato" in r and " da " in r]
    attese = {1: "PARAMETRI", 2: "PARAMETRI"}
    for L, via in attese.items():
        trovate = [r for r in righe if f"ripartenza strato {L} da {via}" in r]
        bene = len(trovate) == 1
        ok &= bene
        print(f"  {'✅' if bene else '⛔'} strato {L} ripartito da {via} ({len(trovate)} righe)")
    codici = [r for r in righe if "da CODICI" in r]
    bene = len(codici) == 1 and "ripartenza strato 2" in codici[0]
    ok &= bene
    print(f"  {'✅' if bene else '⛔'} strato 2 nella finestra [1,2] dai CODICI della corsa b (modalita' b)")
    percentuali = [float(x) for r in righe for x in re.findall(r"codici identici ([\d.]+)%", r)]
    scarti = [float(x) for r in righe for x in re.findall(r"scarto massimo pesi ([\d.e+-]+)", r)]
    bene = percentuali and all(p == 100.0 for p in percentuali) and scarti and all(s == 0.0 for s in scarti)
    ok &= bool(bene)
    print(f"  {'✅' if bene else '⛔'} in ogni riga: codici identici 100% ({len(percentuali)} letture) e "
          f"scarto massimo pesi 0 ({len(scarti)} letture)")
    if adam == "conserva":
        n = len(re.findall(r"stato di Adam CONSERVATO dalla finestra precedente \(6 tensori\)", buf.getvalue()))
        ok &= n == 2
        print(f"  {'✅' if n == 2 else '⛔'} stato di Adam conservato nelle finestre [1,2] e [2,3] ({n} volte)")
    return bool(ok)


def prova_rifiuti(V) -> bool:
    print("\n══ C. argomenti invalidi ══", flush=True)
    cfg, P, Q, emb, ids = _strati(V, 2)
    ok = True
    for nome, kw in (("strato fuori finestra in ripartenza", {"ripartenza_per_strato": {5: {}}}),
                     ("strato fuori finestra in consegna", {"consegna_stato_per_strato": {5: None}}),
                     ("adam_ripartenza sconosciuto", {"adam_ripartenza": "forse"}),
                     ("voce di tipo sconosciuto", {"ripartenza_per_strato": {1: 42}})):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                V.ottimizza_finestra_JOINT(_sub(Q, [0, 1]), ids, cfg, n_passi=2, strati_teacher=_sub(P, [0, 1]),
                                           emb=emb, lr=LR, n_batch_addestramento=2, **kw)
            print(f"  ⛔ {nome}: NON rifiutato")
            ok = False
        except (ValueError, TypeError, RuntimeError) as err:
            print(f"  ✅ {nome}: {type(err).__name__}: {str(err)[:80]}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patchata", help="cartella con valida_finestra_8_12.py PATCHATO")
    ap.add_argument("--originale", help="cartella con valida_finestra_8_12.py originale (prova A)")
    ap.add_argument("--dump", nargs=2, metavar=("CARTELLA", "USCITA"), help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.dump:
        dump(*args.dump)
        return 0
    esiti = {}
    if args.originale:
        esiti["A default invariati"] = prova_default(args.originale, args.patchata)
    V = _carica(args.patchata)
    with tempfile.TemporaryDirectory(prefix="test_patch_rip_codici_") as tmp:
        with contextlib.redirect_stdout(io.StringIO()):
            _corsa_b(V, tmp)
        for adam in ("riparti", "conserva"):
            esiti[f"B catena, Adam {adam}"] = prova_catena(V, adam, tmp)
    esiti["C rifiuti"] = prova_rifiuti(V)
    print("\n" + " · ".join(f"{'✅' if v else '⛔'} {k}" for k, v in esiti.items()))
    return 0 if all(esiti.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
