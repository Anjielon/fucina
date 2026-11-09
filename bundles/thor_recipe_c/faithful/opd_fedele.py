#!/usr/bin/env python3
"""opd_fedele — rollout, premio e cancello del braccio FEDELE dell'OPD (arXiv 2609.26708, codice MingZwhy/QAOPD), 13/10/2026.

Nato dall'audit ricerche/AUDIT_1013/onpolicy.md: il primo giro (c_onpolicy_1012.sh) era fedele nello stimatore ma NON nel
protocollo. Qui le parti di protocollo che il codice ufficiale fissa e che il driver c_onpolicy_fedele_1013.sh usa; la
matematica dell'allenamento resta quella di e2e_onpolicy/e2e_allena (albero ONPOLICY/codice, solo LETTO, mai toccato).

  PREMIO STRETTO   = opd/rewards/math_mixed_reward.py ufficiale (copia identica accanto, PROVENIENZA.txt) con gli argomenti
                     di scripts/opd/run_math.sh: require_hash=True, correct_without_hash_score=0, hash_format_score=0,
                     overlong_penalty=0,5, overlong_chars=3000 -> 1 solo se giusta E con '#### N' o \\boxed{}; -0,5 se
                     sbagliata, senza delimitatore e > 3000 caratteri; altrimenti 0. Oro GSM8K = la stringa dopo '####'.
  VANTAGGI GRPO    = verl compute_grpo_outcome_advantage (norm_adv_by_std_in_grpo): (r - media) / (std + 1e-6) con la std
                     NON distorta (torch.std); gruppo di 1 -> media 0, std 1. (Prima: std di popolazione, x1,15 con G=4.)
  ROLLOUT NUOVI    = `genera` produce SOLO il lato train del giro con lo studente di ADESSO (il driver lo chiama a ogni giro
                     sul GGUF esportato dal giro prima); tutte le completioni di un turno sono sottomesse INSIEME al server
                     (stessi semi per indice, stesso campionamento: e' solo la coda, non la statistica).
  CANCELLO         = `cancello`: GSM8K held-out a DECODIFICA GREEDY (come la valutazione del lavoro: lm-eval do_sample
                     false), N prompt distinti (default 256), 1 risposta ciascuno, premio stretto e flessibile per prompt
                     -> McNemar esatto appaiato fra due modelli (`mcnemar`).
  PROMPT           : perm (seme 2609 di e2e_onpolicy_rollout) [0, N_CANCELLO) = cancello, [N_CANCELLO, +16) = held-out di
                     KL per e2e_allena (G=4, generato UNA volta dall'ospite), poi i prompt di train, mai ripetuti.
Restano DICHIARATE (non cambiate qui): template di chat + istruzione '####' invece del prompt «plain» 5-shot con stop
ufficiale (lo studente gira sul cammino di deployment), solo GSM8K (il lavoro: GSM8K+MATH+DAPO), clamp k1 +-10 e
logp >= -10 assenti in e2e_onpolicy, maestro Q8 invece di bf16.

Sottocomandi:
  genera    --server --stato --giro K --out DIR --righe R [--max-token 1536]   lato train del giro
  held-kl   --server --out DIR [--righe-held 2]                               held-out di KL (una volta, dall'ospite)
  cancello  --server --out DIR [--n 256]                                      GSM8K held-out greedy, premio stretto
  mcnemar   A/riassunto.json B/riassunto.json                                 McNemar esatto appaiato (stretto)
  prova     --out DIR [--rollout-giro1 FILE]                                  SOLO CPU: premio contro i casi ufficiali,
            vantaggi contro la formula di verl, genera/held-kl/cancello col tokenizzatore finto, rilettura del teacher
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import os  # noqa: E402
sys.path.insert(1, os.environ.get("OPD_THOR_V2", "@@ROOT:models@@/gguf/tony-tern/OBIETTIVO_C/ONPOLICY/codice/thor_v2"))
import math_mixed_reward_ufficiale as UFF  # noqa: E402
import e2e_onpolicy as O  # noqa: E402            (albero ONPOLICY/codice/thor_v2, OPD_THOR_V2)
import e2e_onpolicy_rollout as R  # noqa: E402

KW_UFFICIALI = {"require_hash": True, "correct_without_hash_score": 0.0, "hash_format_score": 0.0,
                "overlong_penalty": 0.5, "overlong_chars": 3000}          # scripts/opd/run_math.sh
EPS_VERL = 1e-6
N_KL = 16                                                                 # prompt del held-out di KL (x G=4)
GREEDY = {**R.CAMPIONAMENTO, "temperature": 0.0, "top_k": 1}


def oro_str(answer: str) -> str:
    return answer.split("####")[-1].strip().replace(",", "")


def premio_stretto(testo: str, answer: str) -> float:
    return float(UFF.compute_score("openai/gsm8k", testo, oro_str(answer), **KW_UFFICIALI))


def premio_flessibile(testo: str, answer: str) -> float:
    return O.ricompensa(testo, O.oro_gsm8k(answer))


def vantaggi_verl(r: list) -> list:
    a = np.asarray(r, dtype=np.float64)
    if len(a) == 1:
        m, s = 0.0, 1.0
    else:
        m, s = float(a.mean()), float(a.std(ddof=1))
    return ((a - m) / (s + EPS_VERL)).tolist()


class ServerGreedy(R.Server):
    def completa(self, ids, seme, max_token):
        d = R._post(self.url + "/completion", {"prompt": ids, "n_predict": max_token, "seed": seme, "stream": False,
                                               "return_tokens": True, "cache_prompt": False, **GREEDY})
        return {"gen": list(d.get("tokens") or []), "testo": d.get("content", ""), "stop": d.get("stop_type", "?")}


def genera_gruppi(srv, dati, indici, G, max_token, paralleli, seme0) -> list:
    """come e2e_onpolicy_rollout.genera_gruppi (stessi semi seme0 + 1000 i + g) ma: premio STRETTO ufficiale, vantaggi di
    verl, e tutte le completioni sottomesse insieme (il server resta pieno: niente attesa della piu' lunga per lotto)."""
    pr = {i: srv.prompt_ids(dati[i]["question"]) for i in indici}
    with cf.ThreadPoolExecutor(paralleli) as ex:
        fut = {(i, g): ex.submit(srv.completa, pr[i], seme0 + 1000 * i + g, max_token) for i in indici for g in range(G)}
        out = []
        for i in indici:
            gr = [(g, fut[(i, g)].result()) for g in range(G)]
            rs = [premio_stretto(c["testo"], dati[i]["answer"]) for _, c in gr]
            rf = [premio_flessibile(c["testo"], dati[i]["answer"]) for _, c in gr]
            for (g, c), r_, f_, a_ in zip(gr, rs, rf, vantaggi_verl(rs)):
                out.append({"indice": i, "g": g, "prompt": pr[i], "gen": c["gen"], "testo": c["testo"], "stop": c["stop"],
                            "ricompensa": r_, "ricompensa_flessibile": f_, "adv": a_, "rip8": R.rip8(c["gen"])})
    return out


def riassunto(seqs: list) -> dict:
    if not seqs:
        return {}
    r = R.riassunto(seqs)
    r["ricompensa_flessibile_media"] = float(np.mean([s["ricompensa_flessibile"] for s in seqs]))
    r["penalita_lunghe"] = int(sum(1 for s in seqs if s["ricompensa"] < 0))
    return r


def _id_pad(srv, a) -> int:
    if a.id_pad is not None:
        return a.id_pad
    f = srv.testo_ids("<|im_end|>")
    if len(f) != 1:
        raise RuntimeError(f"<|im_end|> non e' un token solo: {f}")
    return f[0]


def cmd_genera(a, srv) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dati = R.carica_gsm8k(Path(a.gsm8k))
    perm = R.ordine(len(dati))
    id_pad = _id_pad(srv, a)
    t0 = time.time()
    stato = Path(a.stato)
    stato.mkdir(parents=True, exist_ok=True)
    fg = stato / f"giro_{a.giro}.json"
    inizio = a.n_cancello + N_KL
    if fg.exists():                                       # idempotente: un giro rifatto usa GLI STESSI prompt
        cur0 = json.loads(fg.read_text())["cursore_inizio"]
    else:
        fc = stato / "cursore.json"
        cur0 = json.loads(fc.read_text())["cursore"] if fc.exists() else inizio
    if cur0 < inizio:
        raise RuntimeError(f"cursore {cur0} dentro i prompt di cancello/KL (< {inizio})")
    cur, seqs, turni = cur0, [], 0
    n = a.prompt_primo
    while True:
        if cur + n > len(perm):
            raise RuntimeError("prompt di GSM8K esauriti")
        seqs += genera_gruppi(srv, dati, perm[cur:cur + n], a.G, a.max_token, a.paralleli, a.seme)
        cur += n
        turni += 1
        if len(O.impacchetta(seqs, a.L, id_pad)["ids"]) > a.righe:      # le prime `righe` righe sono PIENE
            break
        n = a.prompt_poi
    pk = O.impacchetta(seqs, a.L, id_pad, righe_max=a.righe)
    if len(pk["ids"]) != a.righe:
        raise RuntimeError(f"{len(pk['ids'])} righe invece di {a.righe}")
    R._scrivi_jsonl(out / "rollout_train.jsonl", seqs)
    w = O.scrivi_lato(out, "train", a.variante, pk, {"lato": "train", "cursore": [cur0, cur], "G": a.G,
                                                       "premio": "stretto ufficiale", "vantaggi": "verl"})
    usate = [seqs[k] for k in pk["usate"]]
    rs = {"versione": O.VERSIONE, "giro": a.giro, "campionamento": R.CAMPIONAMENTO, "max_token": a.max_token, "G": a.G,
          "id_pad": id_pad, "L": a.L, "premio": KW_UFFICIALI, "turni": turni, "generate": len(seqs),
          "train": {**w, **riassunto(usate), "scartate": len(pk["scartate"])}}
    fg.write_text(json.dumps({"cursore_inizio": cur0, "cursore_fine": cur}))
    (stato / "cursore.json").write_text(json.dumps({"cursore": cur, "giro": a.giro}))
    rs["secondi"] = round(time.time() - t0, 1)
    (out / "riassunto.json").write_text(json.dumps(rs, indent=1))
    print(json.dumps({k: rs[k] for k in ("train", "generate", "turni", "secondi")}))
    return 0


def cmd_held_kl(a, srv) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dati = R.carica_gsm8k(Path(a.gsm8k))
    perm = R.ordine(len(dati))
    id_pad = _id_pad(srv, a)
    t0 = time.time()
    idx = perm[a.n_cancello:a.n_cancello + N_KL]
    seqs = genera_gruppi(srv, dati, idx, a.G, a.max_token, a.paralleli, a.seme + 7)
    pk = O.impacchetta(seqs, a.L, id_pad, righe_max=a.righe_held)
    R._scrivi_jsonl(out / "rollout_heldout.jsonl", seqs)
    w = O.scrivi_lato(out, "heldout", a.variante, pk, {"lato": "heldout", "prompt": idx, "G": a.G})
    rs = {"heldout": {**w, **riassunto(seqs), "scartate": len(pk["scartate"])}, "secondi": round(time.time() - t0, 1)}
    (out / "riassunto_heldout.json").write_text(json.dumps(rs, indent=1))
    print(json.dumps(rs))
    return 0


def cmd_cancello(a, srv) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dati = R.carica_gsm8k(Path(a.gsm8k))
    perm = R.ordine(len(dati))
    t0 = time.time()
    idx = perm[:a.n_cancello]
    pr = {i: srv.prompt_ids(dati[i]["question"]) for i in idx}
    with cf.ThreadPoolExecutor(a.paralleli) as ex:
        fut = {i: ex.submit(srv.completa, pr[i], a.seme + 1000 * i, a.max_token) for i in idx}
        righe = []
        for i in idx:
            c = fut[i].result()
            righe.append({"indice": i, "testo": c["testo"], "n_gen": len(c["gen"]), "stop": c["stop"],
                          "stretto": premio_stretto(c["testo"], dati[i]["answer"]),
                          "flessibile": premio_flessibile(c["testo"], dati[i]["answer"])})
    with open(out / "cancello.jsonl", "w") as f:
        for r_ in righe:
            f.write(json.dumps(r_) + "\n")
    n = len(righe)
    rs = {"n": n, "decodifica": "greedy", "max_token": a.max_token, "prompt": f"perm[0:{n}] seme {R.SEME_PERM}",
          "giuste_stretto": int(sum(r_["stretto"] == 1.0 for r_ in righe)),
          "giuste_flessibile": int(sum(r_["flessibile"] == 1.0 for r_ in righe)),
          "troncate": int(sum(r_["stop"] == "limit" for r_ in righe)),
          "lunghezza_media": float(np.mean([r_["n_gen"] for r_ in righe])),
          "per_prompt": {str(r_["indice"]): [int(r_["stretto"] == 1.0), int(r_["flessibile"] == 1.0)] for r_ in righe},
          "secondi": round(time.time() - t0, 1)}
    rs["acc_stretto"] = 100.0 * rs["giuste_stretto"] / n
    rs["acc_flessibile"] = 100.0 * rs["giuste_flessibile"] / n
    (out / "riassunto.json").write_text(json.dumps(rs, indent=1))
    print(f"GSM8K held-out {n} greedy: stretto {rs['acc_stretto']:.2f} % flessibile {rs['acc_flessibile']:.2f} % "
          f"troncate {rs['troncate']} lunghezza {rs['lunghezza_media']:.0f} ({rs['secondi'] / 60:.1f} min)")
    return 0


def mcnemar(fa: str, fb: str, quale: int = 0) -> dict:
    A, B = json.loads(Path(fa).read_text())["per_prompt"], json.loads(Path(fb).read_text())["per_prompt"]
    if set(A) != set(B):
        raise ValueError("i due cancelli non hanno gli stessi prompt")
    b = sum(1 for k in A if B[k][quale] and not A[k][quale])      # B giusta, A sbagliata
    c = sum(1 for k in A if A[k][quale] and not B[k][quale])
    n = b + c
    p = 1.0 if n == 0 else min(1.0, 2.0 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)
    return {"n": len(A), "acc_A": 100.0 * sum(v[quale] for v in A.values()) / len(A),
            "acc_B": 100.0 * sum(v[quale] for v in B.values()) / len(B), "B_giusta_A_sbagliata": b,
            "A_giusta_B_sbagliata": c, "p_mcnemar_esatto": p}



def cmd_prova(a) -> int:
    """SOLO CPU, nessun server."""
    esiti = []

    def prova(nome, cond, info=""):
        esiti.append(bool(cond))
        print(f"{'OK' if cond else 'KO'} {nome} {info}", flush=True)

    prov = (QUI / "PROVENIENZA.txt").read_text()
    sha = hashlib.sha256((QUI / "math_mixed_reward_ufficiale.py").read_bytes()).hexdigest()
    prova("premio: copia identica dell'ufficiale (sha256 in PROVENIENZA.txt)", sha in prov, sha[:16])
    lungo = "x " * 1600
    casi = [("so 3+15 = 18.\n#### 18", "... #### 18", 1.0), ("the answer is 18", "#### 18", 0.0),
            ("\\boxed{18}", "#### 18", 1.0), ("#### 17", "#### 18", 0.0), (lungo + " 17", "#### 18", -0.5),
            (lungo + " 18", "#### 18", 0.0), ("#### 1,234", "#### 1,234", 1.0), ("#### $18", "#### 18", 1.0)]
    ok = all(premio_stretto(t, g) == v for t, g, v in casi)
    prova("premio stretto: delimitatore obbligatorio, -0,5 alle lunghe senza risposta, virgole e $",
          ok, str([premio_stretto(t, g) for t, g, _ in casi]))
    rr = [[1, 0, 0, 1], [1, 1, 1, 1], [0, 0, 0, 1], [0.0, -0.5, 1.0, 0.0], [1]]
    ok = all(np.allclose(vantaggi_verl(r), (np.array(r) - np.mean(r)) / (np.std(r, ddof=1) + EPS_VERL))
             for r in rr[:4] if np.std(r) > 0)
    ok &= vantaggi_verl(rr[1]) == [0.0] * 4 and abs(vantaggi_verl(rr[4])[0] - 1 / (1 + EPS_VERL)) < 1e-12
    try:
        import torch
        t = torch.tensor(rr[0], dtype=torch.float64)
        ok &= np.allclose(vantaggi_verl(rr[0]), ((t - t.mean()) / (t.std() + EPS_VERL)).numpy())
    except ImportError:
        pass
    prova("vantaggi = verl (std non distorta, eps 1e-6, gruppo costante 0, gruppo di 1 -> media 0 std 1)", ok)
    if a.rollout_giro1 and Path(a.rollout_giro1).exists():
        dati = {i: d for i, d in enumerate(R.carica_gsm8k(Path(a.gsm8k)))}
        rl = [json.loads(x) for x in open(a.rollout_giro1)]
        fl = [premio_flessibile(s["testo"], dati[s["indice"]]["answer"]) for s in rl]
        st = [premio_stretto(s["testo"], dati[s["indice"]]["answer"]) for s in rl]
        gruppi = {}
        for s, v in zip(rl, st):
            gruppi.setdefault(s["indice"], []).append((s, v))
        segni = sum(1 for g in gruppi.values() if any(np.sign(a_) != np.sign(s["adv"]) for (s, _), a_ in
                                                       zip(g, vantaggi_verl([v for _, v in g]))))
        prova("giro 1 rigiudicato col premio ufficiale (audit: 0,419 -> 0,267, segno cambiato in 39/80 gruppi)",
              all(f == s["ricompensa"] for f, s in zip(fl, rl)),
              f"{len(rl)} rollout: flessibile {np.mean(fl):.3f} (= quello del giro), stretto {np.mean(st):.3f}, "
              f"penalita' {sum(v < 0 for v in st)}, gruppi col segno cambiato {segni}/{len(gruppi)}")
    out = Path(a.out)
    import shutil
    shutil.rmtree(out, ignore_errors=True)
    srv = R.TokFinto()
    a.righe, a.L, a.max_token, a.id_pad, a.prompt_primo, a.prompt_poi, a.righe_held, a.n_cancello = 4, 1024, 200, 2, 4, 2, 1, 8
    a.stato, a.giro = str(out / "stato"), 1
    cmd_held_kl(a, srv)
    cmd_genera(a, srv)
    import e2e_teacher as T
    for lato in ("train", "heldout"):
        c = T.righe_lato(lato, out, a.variante)
        f = out / (f"e2e_corpus_4096_{a.variante}.pt" if lato == "train" else f"e2e_heldout_4096_{a.variante}.pt")
        dati_, firma = O.leggi_onpolicy(out / f"onpolicy_{lato}.pt", f, c["L"], len(c["righe"]), lambda r: c["inizi"][r])
        ok = all(not c["inizi"][r][t + 1].any() for r, (t, adv, seg, anc) in dati_.items())
        prova(f"lato {lato}: rilettura col lettore del teacher, token generati mai inizio di segmento", ok,
              f"{len(c['righe'])} righe, posizioni {firma['n']}")
    g1 = json.loads((out / "stato" / "giro_1.json").read_text())
    prova("prompt di train dopo cancello + KL (mai consumati)", g1["cursore_inizio"] == a.n_cancello + N_KL, str(g1))
    a.giro = 2
    cmd_genera(a, srv)
    g2 = json.loads((out / "stato" / "giro_2.json").read_text())
    prova("giro 2 riparte dal cursore del giro 1 (rollout NUOVI, prompt mai ripetuti)", g2["cursore_inizio"] == g1["cursore_fine"])
    gg = Path(a.modello)
    T.prepara("prova-opd-fedele", ["heldout", "train"], variante=a.variante, modo="segmenti", righe_blocco=2, modello=gg,
              radice=out / "teacher", corpus_dir=out, sha_modello=Path(str(gg) + ".sha256").read_text().split()[0])
    p = json.loads((out / "teacher" / "prova-opd-fedele" / "piano.json").read_text())
    prova("piano del teacher sui due lati", list(p["corpus"]) == ["heldout", "train"], f"blocchi {len(p['blocchi'])}")
    for nome in ("A", "B"):
        a.out, a.seme = str(out / f"cancello_{nome}"), (12 if nome == "A" else 99)
        cmd_cancello(a, srv)
    m = mcnemar(out / "cancello_A" / "riassunto.json", out / "cancello_B" / "riassunto.json")
    m0 = mcnemar(out / "cancello_A" / "riassunto.json", out / "cancello_A" / "riassunto.json")
    prova("cancello: N prompt distinti, McNemar esatto (A contro A: p 1, nessuna discordanza)",
          m["n"] == 8 and m0["p_mcnemar_esatto"] == 1.0 and m0["B_giusta_A_sbagliata"] == 0, json.dumps(m))
    print(f"{sum(esiti)}/{len(esiti)} verdi")
    return 0 if all(esiti) else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("comando", choices=["genera", "held-kl", "cancello", "mcnemar", "prova"])
    ap.add_argument("file", nargs="*")
    ap.add_argument("--server", default="http://127.0.0.1:11465")
    ap.add_argument("--gsm8k", default=str(R.GSM8K))
    ap.add_argument("--stato", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--giro", type=int, default=1)
    ap.add_argument("--variante", default="onpolicy")
    ap.add_argument("--righe", type=int, default=8)
    ap.add_argument("--righe-held", type=int, default=2)
    ap.add_argument("--n-cancello", type=int, default=256)
    ap.add_argument("--prompt-primo", type=int, default=16, help="prompt del primo turno del giro (x G completioni)")
    ap.add_argument("--prompt-poi", type=int, default=4, help="prompt dei turni dopo, finche' le righe sono piene")
    ap.add_argument("--G", type=int, default=4, help="ROLLOUT_N di run_math.sh")
    ap.add_argument("--max-token", type=int, default=1536, help="MAX_RESPONSE_LENGTH di run_math.sh")
    ap.add_argument("--paralleli", type=int, default=8)
    ap.add_argument("--seme", type=int, default=12)
    ap.add_argument("--L", type=int, default=4096)
    ap.add_argument("--id-pad", type=int, default=None)
    ap.add_argument("--rollout-giro1", default=None)
    ap.add_argument("--modello", default="@@ROOT:models@@/gguf/tony-tern/teacher-q8/Ornith-1.5-35B-A3B-Q8_0.gguf")
    a = ap.parse_args(argv)
    if a.comando == "mcnemar":
        print(json.dumps(mcnemar(a.file[0], a.file[1])))
        return 0
    if not a.out:
        ap.error("--out obbligatorio")
    if a.comando == "prova":
        return cmd_prova(a)
    if a.comando == "cancello":
        return cmd_cancello(a, ServerGreedy(a.server))
    srv = R.Server(a.server)
    if a.comando == "held-kl":
        return cmd_held_kl(a, srv)
    if not a.stato:
        ap.error("genera vuole --stato")
    return cmd_genera(a, srv)


if __name__ == "__main__":
    sys.exit(main())
