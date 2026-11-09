#!/usr/bin/env python3
"""Test di `e2e_corpus_4096_eb.py` (26/9/2026, file NUOVO). Tre parti:

1. **Unitari** (senza file, secondi): quote del corpus intero (`quote_domini_estese`, `quote_fonti_nuove`), ordine
   (`ordine_pieno`), scelta degli esempi dal flusso (`scegli_per_flusso`, id ripetuti come in ReasonXL), flusso edu in
   un colpo (`pool_edu_lista`), `assembla` e `identita_base` su tensori sintetici; piu' gli unitari del costruttore
   (`e2e_test_corpus_4096.unitari`).
2. **Fonti** (senza --solo-unitari; tokenizzatore vero): `nemotron_da_righe` = `V2.nemotron_v2`, conteggio per sito
   + `mfaq_da_pagine` = `FontiV2.mfaq`, `pool_edu_lista` = `C.pool_edu`, tracce Cascade, sorgenti E-B integre.
3. **Sui file** (`--dir`): i controlli INDIPENDENTI di `e2e_test_corpus_4096.sui_file` (forma, sha256, cu_seqlens,
   ordine, disgiunzione train/held-out, contro wiki.test.raw / mmlu_test.bin / pannelli, held-out contro la forgia),
   piu':
   - la base (`--base`): ogni riga di addestramento ritrovata IDENTICA per contenuto (sha256 degli id int32, la chiave
     del riuso del teacher) e in testa al suo dominio; held-out e held-out dei pannelli identici al byte;
   - il banco MMLU `mc_v2/mmlu_*.bin` (domanda + scelte, col template del banco): finestre di 32 token (hash del test,
     diverso da quello del costruttore) in comune con le righe intere e coi messaggi = 0;
   - quote U dichiarate/effettive dal manifesto, 4096 righe, ordine: le righe della base in testa a ogni dominio.

    env CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 10 ionice -c3 \\
        @@ROOT:venv@@/bin/python3 e2e_test_corpus_4096_eb.py [--dir DIR --base DIR_BASE] [--solo-unitari]
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HIP_VISIBLE_DEVICES"] = ""
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_k, "2")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import e2e_test_corpus_4096 as T  # noqa: E402  (i suoi controlli sui file, riusati)
import e2e_corpus_4096_eb as EB  # noqa: E402

E, C = EB.E, EB.C
prova = T.prova
ESITI = T.ESITI


# ═══════════════════════════ 1. unitari ═══════════════════════════
class TkOffset(T.TkFinto):
    """Tokenizzatore finto con `tok(testo, return_offsets_mapping=True)`: un token per carattere."""

    def __init__(self):
        self.tok = self

    def __call__(self, testo, add_special_tokens=False, return_offsets_mapping=False):
        out = {"input_ids": [ord(c) for c in testo]}
        if return_offsets_mapping:
            out["offset_mapping"] = [(i, i + 1) for i in range(len(testo))]
        return out


def unitari_eb() -> None:
    pesi = E.quote_u("tutte")
    base = {"edu": 490, "cot": 385, "it_web": 343, "it_wiki": 166, "it_chat": 196, "it_pensiero": 315, "strumenti": 217}
    grande = {d: 10**6 for d in pesi}
    q, rap = EB.quote_domini_estese(4096, pesi, base, grande)
    dich = C.ripartisci(4096, tuple(pesi.items()))
    prova("quote_domini_estese senza tetti = C.ripartisci(4096) e mai sotto la base",
          q == dich and all(q[d] >= base[d] for d in pesi) and rap["mancano_sequenze"] == 0, q)
    cap = dict(grande, cot=100, it_chat=300)
    q, rap = EB.quote_domini_estese(4096, pesi, base, cap)
    prova("quote_domini_estese: cot e it_chat al tetto (base + nuova), il resto a edu e it_web, somma 4096",
          q["cot"] == 485 and q["it_chat"] == 496 and sum(q.values()) == 4096 and rap["nuove"]["cot"] == 100
          and all(q[d] == dich[d] for d in ("it_wiki", "it_pensiero", "strumenti"))
          and q["edu"] + q["it_web"] == dich["edu"] + dich["it_web"] + (dich["cot"] - 485) + (dich["it_chat"] - 496), q)
    cap = dict(cap, edu=150, it_web=300)
    q, rap = EB.quote_domini_estese(4096, pesi, base, cap)
    prova("quote_domini_estese: assorbitori pieni -> mancano dichiarate", rap["mancano_sequenze"] == 4096 - sum(q.values())
          and rap["mancano_sequenze"] > 0, rap["mancano_sequenze"])
    piccola = {"edu": 79, "cot": 79, "it_web": 79, "it_wiki": 39, "it_chat": 118, "it_pensiero": 200, "strumenti": 79}
    q, rap = EB.quote_domini_estese(1000, pesi, piccola, grande)
    prova("quote_domini_estese: base oltre la quota (it_pensiero 200, U ne vuole 71) -> nuove tolte altrove, somma esatta",
          sum(q.values()) == 1000 and all(q[d] >= piccola[d] for d in pesi) and q["it_pensiero"] == 200, q)
    # quote per fonte: dominio intero coi pesi, meno la base
    comp = (("aya", 60), ("oasst2", 30), ("mfaq", 35), ("smoltalk_nothink", 100))
    bf = {"aya": 12, "oasst2": 9, "mfaq": 130, "smoltalk_nothink": 45}
    nuove, info = EB.quote_fonti_nuove(750, comp, bf, {"aya": 0, "oasst2": 0, "mfaq": 400, "smoltalk_nothink": 800})
    t = info["quote_dominio_intero"]
    prova("quote_fonti_nuove: somma esatta, fonti vuote a 0, intero = base + nuove",
          sum(nuove.values()) == 750 and nuove["aya"] == 0 and nuove["oasst2"] == 0
          and all(bf[f] + nuove[f] == t[f] for f in ("mfaq", "smoltalk_nothink")), nuove)
    nuove, _ = EB.quote_fonti_nuove(10, comp, bf, {"aya": 0, "oasst2": 0, "mfaq": 400, "smoltalk_nothink": 800})
    prova("quote_fonti_nuove: base gia' sopra il peso (mfaq) -> le nuove vanno alle altre", nuove["mfaq"] == 0
          and sum(nuove.values()) == 10, nuove)
    try:
        EB.quote_fonti_nuove(900, comp, bf, {"aya": 0, "oasst2": 0, "mfaq": 100, "smoltalk_nothink": 100})
        prova("quote_fonti_nuove: capienza insufficiente -> errore", False)
    except RuntimeError:
        prova("quote_fonti_nuove: capienza insufficiente -> errore", True)
    # ordine: composizione U fra i domini e fonti intercalate dentro il dominio
    et = {d: ["a"] * 4 for d in EB.DOMINI}
    et["it_chat"] = ["base"] * 6 + ["nuova"] * 6
    o = EB.ordine_pieno(et)
    visti = {tuple(x) for x in o}
    n = sum(len(v) for v in et.values())
    chat = [i for k, i in o if EB.DOMINI[k] == "it_chat"]
    prova("ordine_pieno: ogni riga una volta; dentro it_chat base e nuove alternate",
          len(o) == n and len(visti) == n and sorted(chat) == list(range(12))
          and {et["it_chat"][i] for i in chat[:4]} == {"base", "nuova"}, chat)
    # scelta degli esempi dal flusso con id ripetuti (ReasonXL: stesso ds_uid, testi diversi)
    tk = T.TkFinto()
    L = 16
    es = [C.Esempio("f", "x:1", "AAAAA", []), C.Esempio("f", "x:1", "BBBBBBB", []), C.Esempio("f", "x:2", "CCCCCCCCCC", []),
          C.Esempio("f", "x:3", "DDDDDDDDDDDDDDDD", [])]
    mappa = {}
    for e in es:
        mappa.setdefault(e.id, []).append(("d", "f", e))
    flusso = [ord(c) for c in "BBBBBBB" + "CCCCCCCCCC" + "DDDDDDDDDDDDDDDD"][:2 * L]
    righe = torch.tensor(flusso).reshape(2, L)
    s = EB.scegli_per_flusso(["x:1", "x:2", "x:3"], mappa, righe, ["f", "f"], tk)
    prova("scegli_per_flusso: fra due testi dello stesso id sceglie quello del flusso; ultimo troncato ok",
          [e.testo for e in s["f"]] == ["BBBBBBB", "CCCCCCCCCC", "DDDDDDDDDDDDDDDD"], [e.testo[:3] for e in s["f"]])
    try:
        EB.scegli_per_flusso(["x:2", "x:1", "x:3"], mappa, righe, ["f", "f"], tk)
        prova("scegli_per_flusso: ordine sbagliato -> errore", False)
    except RuntimeError:
        prova("scegli_per_flusso: ordine sbagliato -> errore", True)
    # flusso edu in un colpo
    tko = TkOffset()
    docs = [C.Esempio("fineweb_edu", f"edu:{i}", "x" * (5 + i), ["x"], extra={"doc": 2000 + i}) for i in range(6)]
    s, info, u = EB.pool_edu_lista(tko, 2, docs, 8)
    testo = "\n\n".join(d.testo for d in docs)
    prova("pool_edu_lista = prefisso del flusso unito con \\n\\n; usati fino all'ultimo carattere",
          [c for r in s for c in r] == [ord(c) for c in testo[:16]] and [e.extra["doc"] for e in u] == [2000, 2001, 2002],
          [e.extra["doc"] for e in u])
    # assembla + identita_base su tensori sintetici (L = 4096)
    g = torch.Generator().manual_seed(3)
    base_t = {"domini": {}, "fonti": {}, "esempi": {}}
    base_l = E.Lato("train")
    nuova = {}
    for k, d in enumerate(EB.DOMINI):
        nb, nn = 2 + k % 2, 1 + k % 3
        base_t["domini"][d] = torch.randint(0, 1000, (nb, EB.L4), generator=g)
        base_t["fonti"][d] = ["b"] * nb
        base_t["esempi"][d] = [f"{d}:b{i}" for i in range(nb)]
        base_l.metti(d, None, ["b"] * nb, {}, [], [[0, EB.L4]] * nb, [False] * nb)
        nuova[d] = {"seq": torch.randint(0, 1000, (nn, EB.L4), generator=g).tolist(), "et": ["n"] * nn,
                    "usati": [C.Esempio("n", f"{d}:n{i}", "", []) for i in range(nn)], "cu": [[0, EB.L4]] * nn,
                    "ini": [False] * nn, "info": {}}
    t, lato = EB.assembla(base_t, base_l, nuova)
    idb = EB.identita_base(t, base_t)
    prova("assembla: base in testa, nuove dopo, pezzi coerenti; identita_base = tutte",
          idb["tutte"] and all(v["tenute_identiche_in_testa"] for v in idb["per_dominio"].values())
          and all(t["domini"][d].shape[0] == len(t["fonti"][d]) == len(t["esempi"][d]) for d in EB.DOMINI)
          and sorted(map(tuple, t["ordine"].tolist())) == sorted((k, i) for k, d in enumerate(EB.DOMINI)
                                                                  for i in range(t["domini"][d].shape[0])),
          idb["ritrovate_per_contenuto"])
    rotta = {"domini": {d: x.clone() for d, x in base_t["domini"].items()}}
    rotta["domini"]["cot"][0, 5] += 1
    prova("identita_base: una riga della base alterata -> non tutte", not EB.identita_base(t, rotta)["tutte"])
    k = hashlib.sha256(np.ascontiguousarray(t["domini"]["edu"][0].numpy().astype(np.int32)).tobytes()).hexdigest()
    prova("sha_righe = chiave di riga del teacher (sha256 degli id int32)", EB.sha_righe(t["domini"]["edu"][:1]) == [k])
    # proprietari dei token: dominio_multi VERO a L = 16, poi ogni token torna al suo esempio
    es = {f: [C.Esempio(f, f"{f}:{i}", chr(65 + i % 26) * (3 + (7 * i) % 11), []) for i in range(40)]
          for f in ("x", "y", "z")}
    with E.lunghezza(16):
        comp = (("x", 3), ("y", 2), ("z", 4))
        sq, et2, info, u = C.dominio_multi("d", 9, es, comp, tk)
    pr = EB.proprietari_multi(sq, et2, info, u, comp, 16)
    flusso = [x for r in sq for x in r]
    prova("proprietari_multi: un indice per token, ogni token e' del suo esempio (dominio_multi vero)",
          pr.size == len(flusso) and all(u[int(i)].testo[0] == chr(tok) for i, tok in zip(pr, flusso)), pr.size)
    sq2 = [list(r) for r in sq]
    sq2[2][0] += 1
    try:
        EB.proprietari_multi(sq2, et2, info, u, comp, 16)
        prova("proprietari_multi rifiuta una riga alterata", False)
    except RuntimeError:
        prova("proprietari_multi rifiuta una riga alterata", True)
    pr_edu: list = []
    s_e, _, u_e = EB.pool_edu_lista(tko, 2, docs, 8, pr_edu)
    fl = "\n\n".join(d.testo for d in docs)[:16]
    atteso = []
    for k, dd in enumerate(docs):
        atteso += [k] * (len(dd.testo) + 2)
    prova("pool_edu_lista: proprietari = documento di ogni carattere (separatore al documento prima)",
          pr_edu[0].tolist() == atteso[:16] and len(fl) == 16, pr_edu[0].tolist())
    t_chat = ("<|im_start|>system\nIstruzioni lunghe di sistema che non contano<|im_end|>\n<|im_start|>user\nChe ore sono a"
              " Roma adesso, per favore?<|im_end|>\n<|im_start|>assistant\n<think>\nDevo chiamare lo strumento dell'ora"
              " subito.\n</think>\n\n<tool_call>\n<function=orologio>\n</function>\n</tool_call><|im_end|>\n<|im_start|>"
              "user\n<tool_response>\nSono le 10:15 del mattino a Roma\n</tool_response><|im_end|>\n")
    # base tenuta: righe tolte, esempi e proprietari rimappati
    es2 = [C.Esempio("f", f"q:{i}", "", [], extra={"pos": f"rg0:{i}"}) for i in range(6)]
    for i, e in enumerate(es2):
        e.ids = [100 + i] * (10 + 3 * i)
    righe2 = np.concatenate([e.ids for e in es2])[:4 * 16].reshape(4, 16)
    et4 = ["f"] * 4
    pr2 = EB.proprietari_per_fonte(righe2, et4, es2, ["f"] * 6)
    prova("proprietari_per_fonte: ogni token al suo esempio", all(int(righe2.reshape(-1)[k]) == 100 + int(pr2[k])
                                                                  for k in range(pr2.size)), np.unique(pr2).tolist())
    bl2 = E.Lato("train")
    for d in EB.DOMINI:
        bl2.metti(d, None, [], {}, [], [], [])
    bt2 = {"domini": {d: torch.empty((0, EB.L4), dtype=torch.int64) for d in EB.DOMINI},
           "fonti": {d: [] for d in EB.DOMINI}, "esempi": {d: [] for d in EB.DOMINI}}
    # un dominio con 3 righe da L4 e 4 esempi: 0 in riga 0, 1 fra riga 0 e 1, 2 in riga 1-2, 3 in riga 2
    es3 = [C.Esempio("f", f"z:{i}", "", []) for i in range(4)]
    lung = [3000, 3000, 4000, 2288]
    for i, e in enumerate(es3):
        e.ids = [7 + i] * lung[i]
    bt2["domini"]["cot"] = torch.tensor(np.concatenate([e.ids for e in es3])).reshape(3, EB.L4)
    bt2["fonti"]["cot"] = ["f"] * 3
    bt2["esempi"]["cot"] = [e.id for e in es3]
    bl2.metti("cot", None, ["f"] * 3, {}, es3, [[0, EB.L4]] * 3, [True, False, False])
    prop3 = {d: np.empty(0, dtype=np.int64) for d in EB.DOMINI}
    prop3["cot"] = EB.proprietari_per_fonte(bt2["domini"]["cot"].numpy(), ["f"] * 3, es3, ["f"] * 4)
    bten = EB.BaseTenuta(bt2, bl2, prop3)
    bten.togli("cot", [1], "prova")
    v_bt, v_bl, v_own, v_idx = bten.vista()
    prova("BaseTenuta: riga 1 tolta; esempi = quelli con token nelle righe tenute; proprietari rimappati",
          v_idx["cot"] == [0, 2] and v_bt["esempi"]["cot"] == ["z:0", "z:1", "z:2", "z:3"]
          and torch.equal(v_bt["domini"]["cot"], bt2["domini"]["cot"][[0, 2]]) and v_own["cot"].size == 2 * EB.L4
          and v_bl.inizio["cot"] == [True, False], v_bt["esempi"]["cot"])
    bten.togli("cot", [0], "prova")
    v_bt, _, v_own, _ = bten.vista()
    prova("BaseTenuta: tolte 0 e 1 -> restano gli esempi della riga 2", v_bt["esempi"]["cot"] == ["z:2", "z:3"]
          and set(np.unique(v_own["cot"]).tolist()) == {0, 1}, v_bt["esempi"]["cot"])
    pe = EB.per_esempio(np.array([-1, 0, 0, 1, -1, 1, 2]))
    prova("per_esempio: posizioni per esempio, -1 ignorato", {k: v.tolist() for k, v in pe.items()}
          == {0: [1, 2], 1: [3, 5], 2: [6]})
    es4 = {d: [] for d in EB.DOMINI}
    us4 = {d: [] for d in EB.DOMINI}
    es4["it_pensiero"] = ["reasonxl:1", "reasonxl:2", "reasonxl:1"]
    us4["it_pensiero"] = [C.Esempio("r", "reasonxl:1", "", [], extra={"pos": "rg0:26"}),
                          C.Esempio("r", "reasonxl:2", "", [], extra={"pos": "rg0:30"}),
                          C.Esempio("r", "reasonxl:1", "", [], extra={"pos": "rg0:27"})]
    q4, m4 = EB.id_univoci(es4, us4)
    prova("id_univoci: solo gli id ripetuti diventano <id>@<pos>", q4["it_pensiero"] ==
          ["reasonxl:1@rg0:26", "reasonxl:2", "reasonxl:1@rg0:27"] and len(m4) == 2, q4["it_pensiero"])
    t5 = {"domini": {d: torch.empty((0, EB.L4), dtype=torch.int64) for d in EB.DOMINI}}
    t5["domini"]["cot"] = bt2["domini"]["cot"][[0, 2]]
    ib = EB.identita_base(t5, bt2, {d: ({1: "x"} if d == "cot" else {}) for d in EB.DOMINI})
    prova("identita_base con righe tolte: tenute in testa, tolte assenti", ib["tutte"] and ib["righe_tolte"] == 1
          and ib["tolte_assenti"] == 1, ib["per_dominio"]["cot"])
    t5["domini"]["cot"] = bt2["domini"]["cot"]
    prova("identita_base: una riga tolta ancora presente -> non tutte",
          not EB.identita_base(t5, bt2, {d: ({1: "x"} if d == "cot" else {}) for d in EB.DOMINI})["tutte"])
    prova("messaggi_riga = messaggi del controllo indipendente (T.messaggi)", EB.messaggi_riga(t_chat) == set(T.messaggi(t_chat))
          and len(EB.messaggi_riga(t_chat)) == 3, sorted(EB.messaggi_riga(t_chat)))


# ═══════════════════════════ 2. sui file ═══════════════════════════
def sui_file_eb(d: Path, base: Path) -> None:
    t0 = time.time()
    m = json.loads((d / "e2e_corpus_4096_tutte.json").read_text())
    mb = json.loads((base / "e2e_corpus_4096_tutte.json").read_text())
    tr = torch.load(d / m["file"]["train"]["file"], weights_only=True)
    trb = torch.load(base / mb["file"]["train"]["file"], weights_only=True)
    n = sum(x.shape[0] for x in tr["domini"].values())
    prova("E-B: righe di addestramento = --sequenze del manifesto (4096 per E-B)",
          n == sum(m["quote_domini"]["train"]["effettive"].values()) == m["file"]["train"]["righe"], n)
    q = m["quote_domini"]["train"]
    prova("E-B: quote dichiarate = C.ripartisci(righe, U senza wiki)",
          q["dichiarate"] == C.ripartisci(n, tuple(E.quote_u("tutte").items())), q["dichiarate"])
    prova("E-B: righe per dominio = quote effettive",
          all(tr["domini"][k].shape[0] == q["effettive"][k] for k in EB.DOMINI), q["effettive"])
    # righe della base: TENUTE identiche, in testa e nell'ordine, per contenuto (chiave del teacher); TOLTE assenti
    tolte = {k: {int(r) for r in m.get("righe_base_tolte", {}).get(k, {})} for k in EB.DOMINI}
    chiavi = set()
    for k in EB.DOMINI:
        chiavi |= set(EB.sha_righe(tr["domini"][k]))
    tot = ok = n_tolte = assenti = 0
    in_testa = True
    for k in EB.DOMINI:
        b = trb["domini"][k]
        cb = EB.sha_righe(b)
        ten = [r for r in range(b.shape[0]) if r not in tolte[k]]
        tot += len(ten)
        ok += sum(1 for r in ten if cb[r] in chiavi)
        n_tolte += len(tolte[k])
        assenti += sum(1 for r in tolte[k] if cb[r] not in chiavi)
        in_testa &= bool(torch.equal(tr["domini"][k][:len(ten)], b[ten])) and \
            list(tr["fonti"][k][:len(ten)]) == [trb["fonti"][k][r] for r in ten]
        pb, pt = trb["cu_ptr"][k], tr["cu_ptr"][k]
        cu_b = [trb["cu"][k][int(pb[r]):int(pb[r + 1])].tolist() for r in ten]
        cu_t = [tr["cu"][k][int(pt[i]):int(pt[i + 1])].tolist() for i in range(len(ten))]
        in_testa &= cu_b == cu_t and [bool(trb["inizio_esempio"][k][r]) for r in ten] == \
            tr["inizio_esempio"][k][:len(ten)].tolist()
        if tolte[k]:
            prova(f"base {k}: righe tolte dichiarate col motivo e lo sha256 giusto",
                  all(m["righe_base_tolte"][k][str(r)]["sha256_ids"] == cb[r] for r in tolte[k]), sorted(tolte[k]))
    prova("base: righe TENUTE ritrovate identiche per contenuto (sha256 id int32)", ok == tot, f"{ok}/{tot}")
    prova("base: righe TOLTE assenti dal corpus", assenti == n_tolte, f"{assenti}/{n_tolte}")
    prova("base: righe tenute, fonti, cu_seqlens e inizi identici, in testa e nell'ordine", in_testa)
    prova("base: tutte le righe del corpus sono uniche anche fra base e nuove", len(chiavi) == n, f"{len(chiavi)}/{n}")
    q_id = {x["id_qualificato"]: x["id"] for x in m.get("id_qualificati", [])}
    tutti_id = [i for k in EB.DOMINI for i in tr["esempi"][k]]
    presenti = set(tutti_id)
    prova("id qualificati: <id>@<pos>, presenti nel corpus al posto dell'id ripetuto, nessun id ripetuto",
          all(k.startswith(v + "@") and k in presenti and v not in presenti for k, v in q_id.items())
          and len(tutti_id) == len(presenti), len(q_id))
    prova("riuso del teacher: righe da calcolare = righe - tenute della base", m["riuso_teacher"]["righe_da_calcolare"]
          == n - tot and sorted(m["riuso_teacher"]["righe_della_base_NON_riusabili_sha256"])
          == sorted(v["sha256_ids"] for dd in m.get("righe_base_tolte", {}).values() for v in dd.values()),
          m["riuso_teacher"]["righe_da_calcolare"])
    for chiave in ("heldout", "heldout_pannelli"):
        a, b = d / m["file"][chiave]["file"], base / mb["file"][chiave]["file"]
        prova(f"base: {chiave} identico al byte a quello della base", C.sha256_file(a) == C.sha256_file(b)
              == mb["file"][chiave]["sha256"])
    prova("manifesto: righe_della_base = tutte", m["righe_della_base"]["tutte"]
          and m["righe_della_base"]["ritrovate_per_contenuto"] == tot and m["righe_della_base"]["righe_tolte"] == n_tolte)
    # banco MMLU mc_v2: hash del test (seme 777), righe intere e messaggi
    tk = T.Tk()
    testi = []
    for p in sorted(EB.MC_V2.glob("mmlu_*.bin")):
        testi += C.leggi_mmlu(p)
    fm = np.unique(np.concatenate([T.finestre(x) for x in tk.enc(testi)]))
    ho = torch.load(d / m["file"]["heldout"]["file"], weights_only=True)
    righe_int = np.unique(np.concatenate([T.finestre(x.reshape(-1).numpy()) for lato in (tr, ho)
                                          for x in lato["domini"].values()]))
    c1 = T.comuni(righe_int, fm)
    imp = T.impronta(tk, {k: v for k, v in tr["domini"].items() if k not in E.TESTO_CONTINUO}, tr["fonti"], set())
    c2 = T.comuni(imp["fin_messaggi"], fm)
    prova(f"banco mc_v2 ({len(testi)} domande MMLU, domanda + scelte): finestre 32 in comune con le righe intere / "
          "coi messaggi = 0", c1 == 0 and c2 == 0, f"{c1} / {c2}")
    prova("manifesto: banco mc_v2 = 0", m["banco_mmlu_mc_v2"]["finestre_in_comune_col_corpus_intero"] == 0)
    # testo continuo: finestre (hash del test) delle righe NUOVE, come flusso che prosegue dalla base, contro pannelli,
    # wiki.test.raw, domande MMLU e held-out; separate da quelle tutte dentro la base (che non si possono cambiare)
    rif = [T.finestre(tk.enc([C.FILE_WIKI_TEST.read_text(encoding="utf-8")])[0])]
    rif += [T.finestre(x) for x in tk.enc(EB.S.domande_mmlu(C.FILE_MMLU))]
    for pp in (C.DIR_PAN / "pannello_ids.pt", C.DIR_PAN / "pannello_nuovi_domini_v2.pt"):
        rif += [T.finestre(v.reshape(-1).numpy()) for v in torch.load(pp, weights_only=False)["domini"].values()]
    rif += [T.finestre(ho["domini"][k].reshape(-1).numpy()) for k in E.TESTO_CONTINUO]
    rif = np.unique(np.concatenate(rif))
    colpi = {}
    for k in E.TESTO_CONTINUO:
        a_ = tr["domini"][k].reshape(-1).numpy().astype(np.int64)
        w = np.lib.stride_tricks.sliding_window_view(a_.astype(np.uint64), 32)
        with np.errstate(over="ignore"):
            h = (w * T._P).sum(axis=1, dtype=np.uint64)
            h = h ^ (h >> np.uint64(29))
        colpi[k] = int(np.isin(h, rif).sum())
    prova("testo continuo, TUTTE le righe (base tenuta e nuove): finestre 32 contro pannelli / wiki.test / MMLU / "
          "held-out = 0", not any(colpi.values()), colpi)
    print(f"   controlli E-B in {time.time() - t0:.0f} s", flush=True)


# ═══════════════════════════ 3. fonti (tokenizzatore vero, sorgenti su disco) ═══════════════════════════
def fonti_eb() -> None:
    """Le funzioni che rifanno le regole del costruttore sulle righe NUOVE danno, sulle righe VECCHIE, esattamente
    cio' che danno le funzioni del costruttore; sorgenti E-B integre e alle revisioni fissate."""
    import random
    t0 = time.time()
    V2 = EB.V2
    m = EB.man_eb()
    prova("sorgenti E-B integre (sha256 = manifesto)", True, sorted(m["estratti"]))
    revs = {k: v["revisione"] for k, v in m["estratti"].items()}
    man_v2 = json.loads(V2.MAN_SORG_V2.read_text())
    stesse = {k: revs[k] == man_v2["revisioni"][m["estratti"][k]["repo"]] for k in revs
              if m["estratti"][k]["repo"] in man_v2["revisioni"]}
    prova("sorgenti E-B: stessa revisione del manifesto v2 per ogni repo gia' usato", all(stesse.values()) and
          len(stesse) == 5, stesse)
    edu_rev = json.loads((C.DIR_PAN / "manifesto.json").read_text())["fineweb_edu"]["revisione"]
    prova("sorgenti E-B: FineWeb-Edu alla revisione del pannello", revs["fineweb_edu_rg2-3"] == edu_rev, edu_rev)
    tk = C.Tok()
    F = V2.FontiV2(tk, 0.30)
    vietati = {"__nessuno__"}
    rif, _ = V2.nemotron_v2(F, vietati, set(), tk)
    mio, _ = EB.nemotron_da_righe(C.jsonl(F.uscita("nemotron_tool_cal")), vietati, set(), tk,
                                  f"{C.SEME}:nemotron_tool_cal")
    prova("nemotron_da_righe(righe v2) == V2.nemotron_v2 (id e ragionamento)",
          [(e.id, e.extra["pensiero"]) for e in rif] == [(e.id, e.extra["pensiero"]) for e in mio], len(mio))
    siti = {x["domain"] for x in C.jsonl(F.uscita("mfaq_it_valid"))}
    rif = F.mfaq("mfaq_it_train", siti)
    pag = [x for x in C.jsonl(F.uscita("mfaq_it_train")) if x["domain"] not in siti]
    random.Random(f"{C.SEME}:mfaq:mfaq_it_train").shuffle(pag)
    mio = [e for e in EB.mfaq_da_pagine(pag, {}, tk) if not V2.mfaq_scarto(e)]
    prova("mfaq_da_pagine(testa) + filtro v2 == FontiV2.mfaq", [e.id for e in mio] == [e.id for e in rif], len(mio))
    tutte = C.jsonl(EB.DIR_EB / m["estratti"]["mfaq_it_train_intero"]["uscita"])
    testa = C.jsonl(F.uscita("mfaq_it_train"))
    prova("mfaq intero: la testa gia' usata e' il suo prefisso", tutte[:len(testa)] == testa, len(testa))
    docs = [e for e in E.docs_edu() if 300 <= e.extra["doc"] < 420]
    tutti = {e.extra["doc"] for e in E.docs_edu()}
    with E.lunghezza(EB.L4):
        s_rif, _, u_rif = C.pool_edu(tk, 20, tutti - {e.extra["doc"] for e in docs})
    s_mio, _, u_mio = EB.pool_edu_lista(tk, 20, docs, EB.L4)
    prova("pool_edu_lista == C.pool_edu (stessi documenti, 20 righe da 4096, tokenizzatore vero)", s_mio == s_rif
          and [e.id for e in u_mio] == [e.id for e in u_rif], len(u_mio))
    nuovi = EB.docs_edu_nuovi(m)
    prova("edu nuovi: documenti 2000-3999, disgiunti da rg 0-1", min(e.extra["doc"] for e in nuovi) == 2000
          and max(e.extra["doc"] for e in nuovi) == 3999 and not ({e.id for e in nuovi} & {e.id for e in E.docs_edu()}))
    for ramo in ("math", "code"):
        es, info = EB.cascade_nuovi(m, tk, ramo)
        ok = all("<think>" in e.testo and e.prompt.strip() not in ("", "-")
                 and e.extra["indice_mescolato"] >= EB.INDICE_ESTERNO for e in es)
        prova(f"cascade {ramo}: template di Thor col ragionamento, domande vere, indice fuori da Thor",
              ok and len(es) > 500, info)
    print(f"   fonti verificate in {time.time() - t0:.0f} s", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="")
    ap.add_argument("--base", default="")
    ap.add_argument("--solo-unitari", action="store_true")
    ap.add_argument("--senza-fonti", action="store_true", help="salta le prove sulle sorgenti (tokenizzatore vero)")
    ap.add_argument("--esito", default="")
    a = ap.parse_args()
    t0 = time.time()
    T.unitari()
    unitari_eb()
    if not a.solo_unitari and not a.senza_fonti:
        fonti_eb()
    if not a.solo_unitari:
        if a.dir and a.base:
            T.sui_file(Path(a.dir))
            sui_file_eb(Path(a.dir), Path(a.base))
        elif a.dir or a.base:
            raise SystemExit("--dir e --base insieme")
    ko = [e for e in ESITI if not e["ok"]]
    print(f"\n{'TUTTO VERDE' if not ko else f'{len(ko)} ROSSI'}: {len(ESITI)} prove in {time.time() - t0:.0f} s")
    if a.esito:
        Path(a.esito).write_text(json.dumps({"prove": ESITI, "rossi": len(ko), "secondi": round(time.time() - t0, 1)},
                                            ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return 1 if ko else 0


if __name__ == "__main__":
    sys.exit(main())
