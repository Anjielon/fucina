#!/usr/bin/env python3
"""R026 voce 12 — KL REALIZZATA di uno o piu' GGUF contro la cache del maestro (held-out), stessi token.

Perche' non llama-perplexity: la cache `@@ROOT:depot@@/e2e_teacher/q8-maestro` (e2e_teacher.py) NON contiene logit
ma gli stati nascosti dopo output_norm (h_nextn, bf16 [L, 2048]) del Q8 del maestro, in modo `segmenti` (stato GDN
azzerato a ogni confine di documento: llama-perplexity non sa farlo). I logit del maestro si ricalcolano come fa
la perdita di allenamento (e2e_teacher.logit_maestro: h fp32 @ testa bf16 ORIGINALE fp32). I logit dello studente
si prendono dal motore vero (e2e_dump_logit, Vulkan, stesso GGUF, stessi segmenti). KL(maestro || studente) PIENA
(tutto il vocabolario) per posizione, media su tutte le posizioni del held-out = la stessa grandezza che
e2e_allena stampa come «KL held-out» (R2B3m: ~0,3966 sulle 16 righe di P1).

Uso:
  kl_realizzata.py --thor-v2 DIR --cache DIR --dump-bin BIN --out DIR [--righe N] [--gruppo 3] [--thread 8]
                   [--controllo-q8 PATH] [--device cpu|cuda] nome=percorso.gguf [nome=percorso.gguf ...]
Scrive OUT/kl_<nome>.json (KL per riga, riprendibile: un json esistente completo si salta), OUT/kl_realizzata.tsv
e stampa una riga `nome KL ±se` per GGUF (se = errore standard fra le RIGHE; delta appaiato vs il primo GGUF).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def kl_piena_righe(lp_t, lp_s):
    """KL(T||S) per posizione dai log-softmax [n, V] (float64 o float32)."""
    return (lp_t.exp() * (lp_t - lp_s)).sum(-1)


def kl_riga(torch, h, W, logit_s, pezzo=64, dev="cpu", dt=None):
    """KL per posizione di una riga. h [L, d] (bf16/float), W [V, d] fp32 (testa del maestro), logit_s [L, V]
    (memmap float32 dello studente: la riga r e' la previsione dopo la posizione r). Ritorna (kl [L] float64,
    top1_uguale [L] bool). Il maestro: UNICA definizione e2e_teacher.logit_maestro = h.float() @ W.float().T."""
    L = h.shape[0]
    if logit_s.shape[0] != L:
        raise ValueError(f"logit studente {logit_s.shape[0]} righe, h {L}: righe non allineate")
    dt = dt or torch.float64
    out = np.empty(L, np.float64)
    top = np.empty(L, bool)
    for a in range(0, L, pezzo):
        b = min(a + pezzo, L)
        z = h[a:b].to(dev).float() @ W.float().T              # W gia' sul dispositivo (main la sposta UNA volta)
        d = torch.from_numpy(np.array(logit_s[a:b], np.float32)).to(dev)
        if z.shape != d.shape:
            raise ValueError(f"vocabolario diverso: maestro {tuple(z.shape)} studente {tuple(d.shape)}")
        lt = torch.log_softmax(z.to(dt), -1)
        ls = torch.log_softmax(d.to(dt), -1)
        out[a:b] = kl_piena_righe(lt, ls).double().cpu().numpy()
        top[a:b] = (z.argmax(-1) == d.argmax(-1)).cpu().numpy()
    return out, top


def righe_scelte(cm, n):
    tutte = cm.righe("heldout")
    if not tutte:
        raise SystemExit("la cache non ha righe 'heldout' sigillate")
    if n is None or n >= len(tutte):
        return tutte
    idx = np.linspace(0, len(tutte) - 1, n).round().astype(int)
    return [tutte[i] for i in sorted(set(idx.tolist()))]


def dump_gruppo(a, cm, righe, gguf, cartella):
    """Un giro di e2e_dump_logit su un gruppo di righe (modo segmenti, tutte le posizioni)."""
    L = cm.piano["corpus"]["heldout"]["L"]
    if cartella.exists():
        shutil.rmtree(cartella)
    cartella.mkdir(parents=True)
    ids = np.concatenate([cm.ids("heldout", i) for i in righe]).astype(np.int32)
    inizi = np.concatenate([np.isin(np.arange(L), cm.cu_seqlens("heldout", i)[:-1]).astype(np.int32) for i in righe])
    ids.tofile(cartella / "ids.i32")
    cmd = [a.dump_bin, "logit", "--vulkan", "-m", str(gguf), "-o", str(cartella / "dump"), "-t", str(a.thread),
           "-c", str(L), "-b", str(L), "-ub", str(a.ub), "--lunghezza", str(L), "--uscite", "tutte"]
    if cm.piano["modo"] == "segmenti":
        inizi.tofile(cartella / "inizi.i32")
        cmd += ["--segmenti", str(cartella / "inizi.i32")]
    cmd.append(str(cartella / "ids.i32"))
    with open(cartella / "strumento.log", "w") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ)).returncode
    if rc != 0:
        raise SystemExit(f"e2e_dump_logit rc={rc} su {gguf}: vedi {cartella / 'strumento.log'}\n"
                         + (cartella / "strumento.log").read_text()[-1500:])
    return cartella / "dump"


def misura(a, torch, cm, W, nome, gguf, righe, ctrl=False):
    from e2e_dump_logit_leggi import Uscita
    t0 = time.time()
    per_riga = {}
    lav = Path(a.out) / f"_lavoro_{nome}"
    for g in range(0, len(righe), a.gruppo):
        gr = righe[g:g + a.gruppo]
        dump = dump_gruppo(a, cm, gr, gguf, lav)
        try:
            u = Uscita(dump)
            for k, i in enumerate(gr):
                pos, Ld = u.logit(k)
                if len(pos) != len(Ld) or pos[0] != 0:
                    raise SystemExit(f"{nome}: le uscite non partono da 0 ({pos[:3]}): manca --uscite tutte?")
                kl, top = kl_riga(torch, cm.h("heldout", i), W, Ld, pezzo=a.pezzo, dev=a.device, dt=a.dt)
                L = len(kl)
                per_riga[str(i)] = {"kl_media": float(kl.mean()), "kl_meta2": float(kl[L // 2:].mean()),
                                    "top1_uguale": float(top.mean()), "kl_max": float(kl.max()),
                                    "dominio": cm.info("heldout", i).get("dominio")}
                print(f"  {nome} riga {i}: KL {kl.mean():.5f} (meta2 {kl[L // 2:].mean():.5f}) top1 {top.mean():.4f}", flush=True)
        finally:
            shutil.rmtree(lav, ignore_errors=True)         # 4 GB di logit f32 a riga: via subito
    kk = np.array([v["kl_media"] for v in per_riga.values()])
    se = float(kk.std(ddof=1) / np.sqrt(len(kk))) if len(kk) > 1 else float("nan")
    ris = {"nome": nome, "gguf": str(gguf), "n_righe": len(kk), "righe": [int(x) for x in per_riga],
           "kl_media": float(kk.mean()), "se_righe": se,
           "kl_meta2_media": float(np.mean([v["kl_meta2"] for v in per_riga.values()])),
           "top1_uguale_media": float(np.mean([v["top1_uguale"] for v in per_riga.values()])),
           "per_riga": per_riga, "secondi": round(time.time() - t0, 1), "modo": cm.piano["modo"],
           "controllo": ctrl}
    return ris


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--thor-v2", required=True, help="cartella thor_v2 della forgia (e2e_teacher.py, e2e_dump_logit_leggi.py)")
    ap.add_argument("--cache", required=True, help="corsa del teacher (piano.json), es. @@ROOT:depot@@/e2e_teacher/q8-maestro")
    ap.add_argument("--dump-bin", required=True, help="e2e_dump_logit compilato contro il motore che carica i GGUF")
    ap.add_argument("--out", required=True)
    ap.add_argument("--righe", type=int, default=None, help="quante righe del held-out (default tutte)")
    ap.add_argument("--gruppo", type=int, default=3, help="righe per chiamata dello strumento (4 GB di logit f32 a riga)")
    ap.add_argument("--thread", type=int, default=8)
    ap.add_argument("--ub", type=int, default=1024)
    ap.add_argument("--device", default="auto", help="matmul del maestro e log-softmax: cpu | cuda | auto (cuda se c'e')")
    ap.add_argument("--pezzo", type=int, default=None, help="posizioni per pezzo (default 64 cpu, 128 cuda)")
    ap.add_argument("--fp32", action="store_true", help="log-softmax in float32 (default su cuda: la GPU di casa ha fp64 lento; "
                    "errore ~1e-6 assoluto su una KL di ~0,4)")
    ap.add_argument("--controllo-q8", default=None, help="GGUF Q8 del maestro: KL ~0 attesa su 2 righe (allineamento)")
    ap.add_argument("--soglia-controllo", type=float, default=0.05)
    ap.add_argument("--solo-controllo", action="store_true")
    ap.add_argument("gguf", nargs="*", help="nome=percorso.gguf")
    a = ap.parse_args(argv)
    sys.path.insert(0, a.thor_v2)
    import torch
    import e2e_teacher as T
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(a.thread)
    if a.device == "auto":
        a.device = "cuda" if torch.cuda.is_available() else "cpu"
    a.dt = torch.float32 if (a.fp32 or a.device == "cuda") else torch.float64
    a.pezzo = a.pezzo or (128 if a.device == "cuda" else 64)
    print(f"device {a.device}, log-softmax {a.dt}", flush=True)
    cm = T.CacheMaestro(Path(a.cache))
    righe = righe_scelte(cm, a.righe)
    print(f"cache {a.cache}: modo {cm.piano['modo']}, held-out {len(cm.righe('heldout'))} righe, uso {len(righe)}; "
          f"L={cm.piano['corpus']['heldout']['L']}", flush=True)
    print("testa bf16 del maestro...", flush=True)
    W = T.TestaMaestro("bf16").W.to(a.device)
    if a.controllo_q8 and not (out / "controllo_q8.json").exists():
        r = misura(a, torch, cm, W, "controllo_q8", a.controllo_q8, righe[:2], ctrl=True)
        (out / "controllo_q8.json").write_text(json.dumps(r, indent=1))
        if r["kl_media"] > a.soglia_controllo:
            raise SystemExit(f"CONTROLLO FALLITO: il Q8 del maestro da' KL {r['kl_media']:.4f} > {a.soglia_controllo} contro la "
                             f"propria cache: righe/segmenti/testa non allineati. Non lancio i GGUF veri.")
        print(f"controllo Q8: KL {r['kl_media']:.5f} (soglia {a.soglia_controllo}) ok", flush=True)
    if a.solo_controllo:
        return 0
    rif = None
    righe_tsv = []
    for spec in a.gguf:
        nome, _, p = spec.partition("=")
        f = out / f"kl_{nome}.json"
        if f.exists():
            r = json.loads(f.read_text())
            if r.get("righe") == [int(x) for x in righe]:
                print(f"{nome}: gia' misurato ({f})", flush=True)
            else:
                raise SystemExit(f"{f} esiste con altre righe: cancellalo o usa --out nuovo")
        else:
            if not Path(p).exists():
                raise SystemExit(f"{nome}: {p} non esiste")
            r = misura(a, torch, cm, W, nome, p, righe)
            f.write_text(json.dumps(r, indent=1))
        if rif is None:
            rif = r
        d = np.array([r["per_riga"][str(i)]["kl_media"] - rif["per_riga"][str(i)]["kl_media"] for i in righe])
        r["delta_vs_primo"] = float(d.mean())
        r["delta_vs_primo_se"] = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
        riga = (f"{nome} KL {r['kl_media']:.5f} ±{r['se_righe']:.5f} (meta2 {r['kl_meta2_media']:.5f}; top1 {r['top1_uguale_media']:.4f}; "
                f"vs {rif['nome']} {r['delta_vs_primo']:+.5f} ±{r['delta_vs_primo_se']:.5f}; {r['n_righe']} righe)")
        print(riga, flush=True)
        righe_tsv.append(riga)
    (out / "kl_realizzata.tsv").write_text("\n".join(righe_tsv) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
