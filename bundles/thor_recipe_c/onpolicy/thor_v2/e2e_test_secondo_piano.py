#!/usr/bin/env python3
"""Prove del SECONDO PIANO dei densi nello studente (29/9/2026, ramo secondo-piano-e2e da 7d92c83).

Il GGUF R1 (@@ROOT:models@@/gguf/tony-tern/R1/Thor-T-R1.gguf = A2 + 250 tensori blk.N.<nome>2.weight, stesso tipo ternario
del primo piano; motore: fork llamacpp-tq1-k1024 ramo secondo-piano-densi-tutti 8d2506e, uscita W1 x + W2 x). Con
--densi-ternari lo studente legge i *2 come famiglie a esperto unico PROPRIE (theta, codici del passo V, export) e la
LineareTernaria somma i due prodotti; nel cammino dequantizzato il Linear HF riceve W1 + W2. SOLO CPU (la GPU e' di TT2).

GGUF giocattolo: strati 0 (GDN) e 3 (attenzione piena) di R1 coi byte VERI, in tre varianti:
  senza  = senza i *2 (cioe' A2 su quegli strati);   con = coi *2 veri;   zero = coi *2 a campi d azzerati (W2 = 0).

  I   senza *2 == 7d92c83 AL BIT (albero da `git archive 7d92c83`, processo figlio): chiavi, avanti, perdita,
      theta.grad, scale per l'export (densi spenti e accesi), e l'export completo (scale + codici) byte per byte.
      Negativo: lo stesso confronto su `con` (dove 7d92c83 ignora i *2) deve risultare DIVERSO.
  II  *2 a zero == senza *2: avanti, perdita, theta.grad delle famiglie comuni AL BIT (densi accesi e spenti); le chiavi
      di `zero` sono quelle di `senza` piu' i *2 (dopo i primi piani di ogni strato).
  III contro llama.cpp CPU (e2e_dump_logit linkato statico sulla build-cpu-p2 del fork): R1 intero, strati 0-3,
      l_out-0..3 contro gli ingressi/uscita dello studente (ricetta B3_ind32 dell'allenamento), errore relativo per token
      (e2e_fedelta.errore_per_token). Tolleranza DICHIARATA: mediana <= 3 % per strato (SOGLIA_ERR_STRATO di F0b) e
      mediana R1 <= 1,5 x quella di A2 sullo stesso cammino (A2 = studente di 7d92c83 contro il motore, stessa misura).
      Negativo: studente coi *2 ignorati (mutante) -> mediana allo strato 3 >= 3 x quella corretta.
  IV  export -> rilettura -> avanti AL BIT sul giocattolo `con`: theta casuali su TUTTE le famiglie (anche *2), registri
      di salti anche sui *2, istantanea, e2e_esporta.esporta, e2e_pv_codici.esporta_codici, ricarica_codici; studente dal
      file finale: codici, scale e codice C di ogni *2 == allenati al bit, uscita al bit.
  V   mutazioni che devono far diventare rossa una verifica: (a) avanti che ignora UN *2 (per ciascuno dei 13 nel
      giocattolo) -> uscita diversa; (b) cammino dequantizzato che ignora i *2 -> lontano dal cammino ternario (B64,
      corretto entro 1e-6); (c) export senza le scale dei *2 -> la ricarica solleva o le scale del file non sono le
      allenate; (d) export dei codici senza i registri dei *2 -> byte dei codici dei *2 diversi dagli allenati.

Uso (mai la GPU; al massimo 8 fili):
    governa_export.sh systemd-run --user --scope --quiet -p CPUQuota=800% -p MemoryMax=12G nice -n 19 ionice -c3 \\
      env OMP_NUM_THREADS=8 @@ROOT:venv@@/bin/python3 e2e_test_secondo_piano.py [--solo I,II,...]
    esito: $E2E_P2_DIR/esito_test_secondo_piano.json (default @@ROOT:lavoro@@/secondo_piano_prove)
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("TRITON_INTERPRET", "1")
for _v in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
    os.environ[_v] = ""
import numpy as np  # noqa: E402
import torch  # noqa: E402

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import e2e_studente as S  # noqa: E402
import e2e_codici_scale as CS  # noqa: E402
import e2e_pv_codici as PV  # noqa: E402
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType as GT  # noqa: E402

DIR = Path(os.getenv("E2E_P2_DIR", "@@ROOT:lavoro@@/secondo_piano_prove"))
R1 = Path(os.getenv("E2E_R1", "@@ROOT:models@@/gguf/tony-tern/R1/Thor-T-R1.gguf"))
A2 = Path(os.getenv("E2E_A2", "@@ROOT:depot@@/tutto_ternario/gguf/Thor-T-A2.gguf"))
LLAMA = Path(os.getenv("E2E_LLAMA_P2", "@@ROOT:models@@/build/llamacpp-tq1-k1024"))
FILI = int(os.getenv("OMP_NUM_THREADS", "8"))
torch.set_num_threads(FILI)
BASE = "7d92c83"
ESITO: dict = {"data": time.strftime("%Y-%m-%d %H:%M:%S"), "r1": str(R1), "base": BASE, "prove": {}}
MUTO = lambda *_: None  # noqa: E731
TUTTI = tuple(S.FAMIGLIE_DENSE)
STRATI = (0, 3)


def registra(nome: str, ok: bool, **dati) -> bool:
    ESITO["prove"][nome] = {"ok": bool(ok), **dati}
    print(f"[{nome}] {'OK' if ok else 'ROSSO'} {json.dumps(dati, ensure_ascii=False, default=str)[:3000]}", flush=True)
    return ok


def rel(a, b) -> float:
    a, b = a.double(), b.double()
    return float((a - b).abs().max() / b.abs().max().clamp_min(1e-300))


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for pezzo in iter(lambda: f.read(1 << 24), b""):
            h.update(pezzo)
    return h.hexdigest()


# ═══════════════════════════════════ GGUF giocattolo da R1 ═══════════════════════════════════
def e_p2(nome: str) -> bool:
    parti = nome.split(".")
    return len(parti) == 4 and parti[0] == "blk" and parti[3] == "weight" and parti[2] in S.FAMIGLIE_DENSE2.values()


def giocattolo(variante: str) -> Path:
    """Strati STRATI di R1 coi byte veri + output_norm. `senza`: senza *2; `con`: coi *2; `zero`: *2 coi campi d a 0."""
    assert variante in ("senza", "con", "zero")
    p = DIR / f"giocattolo_R1_{variante}.gguf"
    firma = {"sorgente": str(R1), "dimensione": R1.stat().st_size, "strati": list(STRATI), "variante": variante,
             "versione": 1}
    js = p.with_name(p.name + ".json")
    if p.exists() and js.exists() and json.loads(js.read_text()).get("firma") == firma:
        return p
    r = GGUFReader(str(R1))
    tmp = p.with_name(p.name + ".parziale")
    w = GGUFWriter(str(tmp), arch="qwen35moe")
    n_p2 = 0
    for t in r.tensors:
        nome = t.name
        if not (any(nome.startswith(f"blk.{L}.") for L in STRATI) or nome == "output_norm.weight"):
            continue
        p2 = e_p2(nome)
        if p2 and variante == "senza":
            continue
        dati = np.array(np.asarray(t.data), copy=True)
        if dati.dtype != np.uint8:
            dati = dati.view(np.uint8)
        if p2 and variante == "zero":
            nb, _, off, k = CS.LAYOUT[t.tensor_type.name]
            v = dati.reshape(-1, nb)
            v[:, off:off + 2 * k] = 0
        n_p2 += int(p2)
        w.add_tensor(nome, dati, raw_dtype=GT[t.tensor_type.name])
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    os.replace(tmp, p)
    js.write_text(json.dumps({"firma": firma, "tensori_p2": n_p2, "sha256": sha(p)}, indent=1))
    return p


def p2_presenti(st, L: int) -> list:
    return [n for n in st.esperti(L).fam if n in S.FAMIGLIE_DENSE2]


def lineari_ternarie(st) -> list:
    return [m for L in st.strati for m in st.moduli[L].modules() if isinstance(m, S.LineareTernaria)]


# ═══════════════════════════════════ I: senza *2 == 7d92c83 al bit ═══════════════════════════════════
CALCOLO = r'''
def calcola(S, gguf, densi=(), ricetta="B_ind32"):
    import torch
    kw = {"densi_ternari": densi} if densi else {}
    st = S.Studente(gguf=gguf, strati=[0, 3], ricetta=ricetta, motore_nome="torch", device="cpu",
                    esperti=slice(0, 64), log=lambda *_: None, **kw)
    with torch.no_grad():
        for (L, n), p in zip(st.chiavi(), st.parametri()):
            g = torch.Generator().manual_seed(1000 * L + sum(ord(c) for c in n))
            p.copy_(0.01 * torch.randn(p.shape, generator=g))
    g = torch.Generator().manual_seed(2909)
    T = 48
    x0 = torch.randn(1, T, 2048, generator=g) * 0.5
    G = torch.randn(T, 2048, generator=g)
    ids = torch.zeros(1, T, dtype=torch.long)
    with torch.no_grad():
        xL = st.avanti(ids, x0=x0)["xL"].clone()
    val = st.gradiente(ids, lambda h: (h * G).sum() / T, x0=x0)
    return {"chiavi": [list(k) for k in st.chiavi()], "xL": xL, "val": val,
            "grad": {f"{L}.{n}": p.grad.detach().clone() for (L, n), p in zip(st.chiavi(), st.parametri())},
            "scale": {k: torch.from_numpy(v.astype("float32")) for k, v in st.scale_per_export().items()}}
'''

# export completo (scale + codici) con registri deterministici su tutte le famiglie dense presenti (anche *2) e sul
# gate dello strato 0; ritorna gli sha dei file e i registri (per le prove del nuovo codice).
EXPORT = r'''
def esporta_tutto(S, A, X, PV, gguf, d, famiglie_salti=None, senza_scale=(), senza_registri=(), ricarica=True):
    import numpy as np, torch, shutil, hashlib
    from pathlib import Path
    d = Path(d)
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    TUTTI = tuple(S.FAMIGLIE_DENSE)
    st = S.Studente(gguf=gguf, strati=[0, 3], ricetta="B", motore_nome="torch", device="cpu", log=lambda *_: None,
                    densi_ternari=TUTTI)
    g = torch.Generator().manual_seed(2912)
    with torch.no_grad():
        for p in st.parametri():
            p.copy_(0.05 * torch.randn(p.shape, generator=g))
    fams = {(L, n): PV.famiglia_scrivibile(f) for L in (0, 3) for n, f in st.esperti(L).fam.items()}
    densa = lambda n: n in S.FAMIGLIE_DENSE or n in getattr(S, "FAMIGLIE_DENSE2", {})
    regs = []
    for passo in range(2):
        parti = []
        for k in [k for k in fams if densa(k[1]) or k == (0, "gate")]:
            f = fams[k]
            T_ = PV.CAMPI.trit_torch(f.blocchi.reshape(-1, f.nb), f.tipo).view(f.E, f.R, f.C)
            m = 60
            e_ = torch.randint(0, f.E, (m,), generator=g)
            r_ = torch.randint(0, f.R, (m,), generator=g)
            c_ = torch.randint(0, f.C, (m,), generator=g)
            if passo == 1:
                prima = regs[0].righe[(regs[0].righe["L"] == k[0]) & (regs[0].righe["f"] == PV.FAMIGLIE_IDX[k[1]])]
                e_ = torch.cat([torch.from_numpy(prima["e"][:10].astype(np.int64)), e_])
                r_ = torch.cat([torch.from_numpy(prima["r"][:10].astype(np.int64)), r_])
                c_ = torch.cat([torch.from_numpy(prima["c"][:10].astype(np.int64)), c_])
            chiave_ = (e_ * f.R + r_) * f.C + c_
            _, pr = np.unique(chiave_.numpy(), return_index=True)
            pr = np.sort(pr)
            e_, r_, c_ = e_[pr], r_[pr], c_[pr]
            t_ = T_[e_, r_, c_].to(torch.int32)
            dr = torch.where(t_ == 0, torch.randint(0, 2, t_.shape, generator=g, dtype=torch.int32) * 2 - 1, -t_)
            z = torch.zeros(0)
            parti.append(PV.applica_salti(f, PV.Selezione(e_, r_, c_ // f.gs, c_ % f.gs, dr.to(torch.int8), z, z, z,
                                                           z, z, z, z)))
        regs.append(PV.Registro.unisci(parti))
    nome = A.salva_istantanea(d, st, 7, {"prova": "secondo_piano"})
    ist = A.carica_istantanea(d / nome)
    theta = torch.load(d / nome / "theta.pt", weights_only=True)
    scale = {k: v for k, v in ist["scale"].items() if not any(s in k for s in senza_scale)}
    scalato = d / "scalato.gguf"
    prov = X.esporta(gguf, scalato, scale, ist["meta"], theta=theta, strati_ricarica=(0, 3) if ricarica else (),
                     device="cpu", esperti_ricarica=slice(0, 32), log=lambda *_: None)
    fi_via = {PV.FAMIGLIE_IDX[n] for n in PV.FAMIGLIE_IDX if n in senza_registri}
    regs_x = [PV.Registro(r.righe[~np.isin(r.righe["f"], list(fi_via))]) for r in regs] if fi_via else regs
    finale = d / "finale.gguf"
    rep = PV.esporta_codici(scalato, finale, regs_x)
    ric = PV.ricarica_codici(scalato, finale, regs_x, [0, 3]) if ricarica else None
    h = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    return {"st": st, "regs": regs, "ist": ist, "prov": prov, "rep": rep, "ric": ric, "scalato": scalato,
            "finale": finale, "sha_scalato": h(scalato), "sha_finale": h(finale)}
'''

FIGLIO = r'''
import os, sys, json, torch
albero, modo, gguf, uscita = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
sys.path.insert(0, albero)
torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "8")))
import e2e_studente as S
assert os.path.dirname(os.path.abspath(S.__file__)) == albero, S.__file__
assert not hasattr(S, "FAMIGLIE_DENSE2"), "l'albero vecchio ha gia' il secondo piano?"
ns = {}
exec(open(sys.argv[5]).read(), ns)
if modo == "calcola":
    densi = tuple(x for x in sys.argv[6].split(",") if x)
    torch.save(ns["calcola"](S, gguf, densi), uscita)
else:
    import e2e_allena as A, e2e_esporta as X, e2e_pv_codici as PV
    r = ns["esporta_tutto"](S, A, X, PV, gguf, uscita)
    print(json.dumps({"sha_scalato": r["sha_scalato"], "sha_finale": r["sha_finale"]}))
'''


def albero_base() -> Path:
    d = DIR / f"albero_{BASE}"
    if not (d / "thor_v2" / "e2e_studente.py").exists():
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        tar = subprocess.run(["git", "-C", str(QUI.parent), "archive", BASE, "thor_v2"], capture_output=True,
                             check=True).stdout
        subprocess.run(["tar", "-x", "-C", str(d)], input=tar, check=True)
        shutil.copyfile(QUI / "mappa_completa_gguf_hf_perm.npz", d / "thor_v2" / "mappa_completa_gguf_hf_perm.npz")
    return d / "thor_v2"


def _figlio(modo: str, gguf: Path, uscita: Path, codice: str, *extra) -> str:
    albero = albero_base()
    fc = DIR / f"codice_{modo}.py"
    fc.write_text(codice)
    ff = DIR / "figlio.py"
    ff.write_text(FIGLIO)
    env = dict(os.environ, TRITON_INTERPRET="1", HIP_VISIBLE_DEVICES="", CUDA_VISIBLE_DEVICES="",
               ROCR_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, str(ff), str(albero), modo, str(gguf), str(uscita), str(fc), *extra],
                       check=True, env=env, capture_output=True, text=True, cwd=str(albero))
    return r.stdout


def uguali(a: dict, b: dict, solo_comuni: bool = False) -> dict:
    kg = set(a["grad"]) & set(b["grad"]) if solo_comuni else set(a["grad"])
    return {"chiavi": a["chiavi"] == b["chiavi"],
            "avanti": bool(torch.equal(a["xL"], b["xL"])),
            "perdita": a["val"] == b["val"],
            "theta_grad": (solo_comuni or set(a["grad"]) == set(b["grad"])) and
                          all(torch.equal(a["grad"][k], b["grad"][k]) for k in kg),
            "scale": (solo_comuni or set(a["scale"]) == set(b["scale"])) and
                     all(torch.equal(a["scale"][k], b["scale"][k]) for k in set(a["scale"]) & set(b["scale"]))}


def prova_i() -> bool:
    t0 = time.time()
    ns: dict = {}
    exec(CALCOLO, ns)
    esiti, ok = {}, True
    for densi in ((), TUTTI):
        tag = "accesi" if densi else "spenti"
        for var in ("senza", "con"):
            out = DIR / f"i_base_{var}_{tag}.pt"
            _figlio("calcola", giocattolo(var), out, CALCOLO, ",".join(densi))
            vecchio = torch.load(out, weights_only=False)
            nuovo = ns["calcola"](S, giocattolo(var), densi)
            u = uguali(vecchio, nuovo)
            if var == "senza":
                ok &= all(u.values())
            else:   # negativo: sui *2 veri il codice di prima (che li ignora) DEVE dare un avanti diverso
                ok &= not u["avanti"]
            esiti[f"{var}_{tag}"] = u
            del vecchio, nuovo
            gc.collect()
    # export completo al byte (scale + codici) sul giocattolo senza *2
    exec(EXPORT, ns)
    import e2e_allena as A
    import e2e_esporta as X
    vecchio = json.loads(_figlio("esporta", giocattolo("senza"), DIR / "i_export_base", EXPORT).strip().splitlines()[-1])
    nuovo = ns["esporta_tutto"](S, A, X, PV, giocattolo("senza"), DIR / "i_export_nuovo")
    exp = {"scalato_al_byte": vecchio["sha_scalato"] == nuovo["sha_scalato"],
           "finale_al_byte": vecchio["sha_finale"] == nuovo["sha_finale"], "sha_finale": nuovo["sha_finale"][:16]}
    ok &= exp["scalato_al_byte"] and exp["finale_al_byte"]
    del nuovo
    gc.collect()
    return registra("I_senza_p2_identico_a_7d92c83", ok, casi=esiti, export=exp, secondi=round(time.time() - t0, 1))


# ═══════════════════════════════════ II: *2 a zero == senza *2 ═══════════════════════════════════
def prova_ii() -> bool:
    t0 = time.time()
    ns: dict = {}
    exec(CALCOLO, ns)
    esiti, ok = {}, True
    for densi in ((), TUTTI):
        tag = "accesi" if densi else "spenti"
        a = ns["calcola"](S, giocattolo("senza"), densi)
        b = ns["calcola"](S, giocattolo("zero"), densi)
        u = uguali(a, b, solo_comuni=True)
        u.pop("chiavi")
        if densi:
            senza_p2 = [k for k in b["chiavi"] if k[1] not in S.FAMIGLIE_DENSE2]
            extra = [k[1] for k in b["chiavi"] if k[1] in S.FAMIGLIE_DENSE2]
            u["chiavi_comuni_nello_stesso_ordine"] = senza_p2 == a["chiavi"]
            u["p2_in_piu"] = len(extra) == 13 and set(extra) == set(S.FAMIGLIE_DENSE2)
            u["p2_dopo_i_primi_piani"] = all(
                max(i for i, k in enumerate(b["chiavi"]) if k[0] == L and k[1] not in S.FAMIGLIE_DENSE2) <
                min(i for i, k in enumerate(b["chiavi"]) if k[0] == L and k[1] in S.FAMIGLIE_DENSE2) for L in STRATI)
        else:
            u["chiavi_identiche"] = a["chiavi"] == b["chiavi"]
        ok &= all(u.values())
        esiti[tag] = u
    return registra("II_p2_a_zero_uguale_a_senza", ok, casi=esiti, secondi=round(time.time() - t0, 1))


# ═══════════════════════════════════ III: contro llama.cpp CPU ═══════════════════════════════════
def dump_bin() -> Path:
    b = DIR / "e2e_dump_logit_p2"
    if not b.exists():
        bb = LLAMA / "build-cpu-p2"
        subprocess.run(["g++", "-std=c++17", "-O3", "-DNDEBUG", "-fopenmp", f"-I{LLAMA}/include",
                        f"-I{LLAMA}/ggml/include", f"-I{LLAMA}/vendor", str(QUI / "e2e_dump_logit.cpp"), "-o", str(b),
                        str(bb / "src/libllama.a"), str(bb / "ggml/src/libggml.a"), str(bb / "ggml/src/libggml-cpu.a"),
                        str(bb / "ggml/src/libggml-base.a"), "-lgomp", "-lpthread"], check=True, capture_output=True)
    return b


def ids_prova() -> np.ndarray:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(S.SRC_HF))
    testo = ("La cucina italiana e' famosa nel mondo per la pasta, la pizza e il gelato. "
             "def somma(a, b):\n    return a + b\n")
    return np.asarray(tok(testo)["input_ids"][:32], dtype=np.int32)


def dump_llama(gguf: Path, nome: str, ids: np.ndarray):
    from e2e_dump_logit_leggi import Uscita
    d = DIR / f"iii_{nome}"
    if (d / "dump" / "manifesto.json").exists():
        return Uscita(d / "dump")
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    ids.tofile(d / "ids.i32")
    n = len(ids)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="", GGML_VK_VISIBLE_DEVICES="")
    with open(d / "strumento.log", "w") as f:
        subprocess.run([str(dump_bin()), "logit", "-m", str(gguf), "-o", str(d / "dump"), "-t", str(FILI), "-c", str(n),
                        "-b", str(n), "-ub", str(n), "--lunghezza", str(n), "--uscite", "ultima", "--no-logit",
                        "--l-out", "0,1,2,3", str(d / "ids.i32")], check=True, stdout=f, stderr=subprocess.STDOUT,
                       env=env)
    return Uscita(d / "dump")


def _errori(st, ids: np.ndarray, u) -> dict:
    from e2e_fedelta import errore_per_token
    with torch.no_grad():
        out = st.avanti(torch.from_numpy(ids.astype(np.int64))[None])
    stud = out["ingressi"][1:] + [out["xL"]]
    res = {}
    for L in range(4):
        rif = torch.from_numpy(np.array(u.tensore(0, f"l_out-{L}"), dtype=np.float32))
        e = errore_per_token(stud[L][0].float(), rif)
        res[L] = {"mediana": float(e.median()), "p90": float(e.quantile(0.9)), "max": float(e.max())}
    return res


def prova_iii() -> bool:
    t0 = time.time()
    ids = ids_prova()
    u_r1 = dump_llama(R1, "R1", ids)
    u_a2 = dump_llama(A2, "A2", ids)
    kw = dict(strati=[0, 1, 2, 3], ricetta="B3_ind32", motore_nome="torch", device="cpu", log=MUTO,
              densi_ternari=TUTTI)
    st = S.Studente(gguf=R1, **kw)
    n_p2 = sum(len(p2_presenti(st, L)) for L in st.strati)
    e_r1 = _errori(st, ids, u_r1)
    lin = [m for m in lineari_ternarie(st) if m.nome2 is not None]
    salvati = [m.nome2 for m in lin]
    for m in lin:                                   # mutante: il forward ignora i *2
        m.nome2 = None
    e_mut = _errori(st, ids, u_r1)
    for m, n2 in zip(lin, salvati):
        m.nome2 = n2
    del st
    gc.collect()
    sa = S.Studente(gguf=A2, **kw)
    e_a2 = _errori(sa, ids, u_a2)
    del sa
    gc.collect()
    soglia = 0.03
    crit = {"R1_mediana_<=3%": all(e_r1[L]["mediana"] <= soglia for L in range(4)),
            "R1_<=_1.5x_A2": all(e_r1[L]["mediana"] <= 1.5 * e_a2[L]["mediana"] for L in range(4)),
            "mutante_>=_3x": e_mut[3]["mediana"] >= 3 * e_r1[3]["mediana"],
            "p2_nello_studente": n_p2 == 25}
    return registra("III_contro_llama_cpu", all(crit.values()), token=len(ids), p2_caricati=n_p2, R1=e_r1, A2=e_a2,
                    mutante_senza_p2=e_mut, criteri=crit, tolleranza="mediana errore per token <= 3% (F0b); "
                    "ricetta B3_ind32 (ingressi fp16 ruotati, residuo fp32) contro llama.cpp CPU (attivazioni q8_K)",
                    secondi=round(time.time() - t0, 1))


# ═══════════════════════════════════ IV + V(c,d): export -> rilettura -> avanti ═══════════════════════════════════
def _controlla_file(st, finale: Path, solo=None) -> dict:
    """Per ogni famiglia densa (anche *2) dello studente allenato `st`: codici, scale (campi d) == allenati al bit."""
    import e2e_esporta as X
    from tq1_0r_1024_codec import S1024
    sf = S.Studente(gguf=finale, strati=[0, 3], ricetta="B", motore_nome="torch", device="cpu", log=MUTO,
                    densi_ternari=TUTTI)
    per = {}
    for L in STRATI:
        for n in [m for m in st.esperti(L).fam if S.e_densa(m)]:
            if solo is not None and n not in solo:
                continue
            fa, fb = st.esperti(L).fam[n], sf.esperti(L).fam[n]
            nb_, _, off_, k_ = CS.LAYOUT[fa.tipo]
            m_ = torch.ones(nb_, dtype=torch.bool)
            m_[off_:off_ + 2 * k_] = False
            byte_ok = bool(torch.equal(fa.blocchi.reshape(-1, nb_)[:, m_], fb.blocchi.reshape(-1, nb_)[:, m_]))
            sc_ok = bool(torch.equal(st.esperti(L).scala(n).detach().to(torch.float16), fb.d0.to(torch.float16)))
            Wc = X._deq_c(fb.blocchi[0].numpy(), fb.tipo)
            W = CS.pesi_da_trit_scale(CS.trit_da_blocchi(fa.blocchi.reshape(-1, fa.nb), fa.tipo),
                                      st.esperti(L).scala(n).detach().reshape(-1, CS.LAYOUT[fa.tipo][3]), fa.tipo,
                                      torch.from_numpy(S1024) if fa.ruotata else None).numpy()
            per[f"{L}.{n}"] = {"tipo": fa.tipo, "codici": byte_ok, "scale": sc_ok, "codice_C": bool(np.array_equal(W, Wc))}
    g = torch.Generator().manual_seed(2914)
    x = torch.randn(1, 64, 2048, generator=g) * 0.5
    ids = torch.zeros(1, 64, dtype=torch.long)
    with torch.no_grad():
        usc = bool(torch.equal(st.avanti(ids, x0=x)["xL"], sf.avanti(ids, x0=x)["xL"]))
    del sf
    return {"per": per, "uscita_al_bit": usc}


def prova_iv_v() -> bool:
    import e2e_allena as A
    import e2e_esporta as X
    t0 = time.time()
    ns: dict = {}
    exec(EXPORT, ns)
    src = giocattolo("con")
    r = ns["esporta_tutto"](S, A, X, PV, src, DIR / "iv_export")
    st = r["st"]
    n_reg_p2 = sum(int(np.isin(g.righe["f"], [PV.FAMIGLIE_IDX[n] for n in S.FAMIGLIE_DENSE2]).sum()) for g in r["regs"])
    scale_p2 = [k for k in r["ist"]["scale"] if e_p2(k)]
    c = _controlla_file(st, r["finale"])
    p2_ok = [v for k, v in c["per"].items() if k.split(".")[1] in S.FAMIGLIE_DENSE2]
    ok_iv = c["uscita_al_bit"] and len(p2_ok) == 13 and len(c["per"]) == 26 and \
        all(v["codici"] and v["scale"] and v["codice_C"] for v in c["per"].values()) and len(scale_p2) == 13 and \
        n_reg_p2 > 0 and all(v["uscita_al_bit"] and v["byte_codici_uguali"] for v in r["ric"].values()) and \
        all(r["prov"]["ricarica"][str(L)]["scale_al_bit"] and r["prov"]["ricarica"][str(L)]["uscita_al_bit"]
            for L in STRATI)
    registra("IV_export_rilettura_al_bit", ok_iv, famiglie=len(c["per"]), p2_verificati=len(p2_ok),
             scale_p2_nell_istantanea=len(scale_p2), salti_sui_p2=n_reg_p2, uscita_al_bit=c["uscita_al_bit"],
             ricarica_codici=r["ric"], ricarica_scale=r["prov"]["ricarica"],
             per_famiglia=c["per"], secondi=round(time.time() - t0, 1))
    # V(c): export SENZA le scale dei *2 -> la ricarica deve sollevare (o le scale del file non sono le allenate)
    try:
        rc = ns["esporta_tutto"](S, A, X, PV, src, DIR / "v_senza_scale", senza_scale=tuple(
            f"{v}.weight" for v in S.FAMIGLIE_DENSE2.values()), ricarica=False)
        cc = _controlla_file(rc["st"], rc["finale"], solo=set(S.FAMIGLIE_DENSE2))
        vc = {"sollevato": False, "scale_p2_diverse": sum(not v["scale"] for v in cc["per"].values())}
        try:
            X.ricarica(src, rc["scalato"], {k: v for k, v in rc["ist"]["scale"].items() if not e_p2(k)}, None, [0],
                       log=MUTO)
            vc["ricarica_solleva"] = False
        except RuntimeError:
            vc["ricarica_solleva"] = True
        neg_c = vc["scale_p2_diverse"] == 13 and vc["ricarica_solleva"]
    except RuntimeError as e:
        vc, neg_c = {"sollevato": True, "errore": str(e)[:300]}, True
    # V(d): codici senza i registri dei *2 -> i codici dei *2 nel file non sono gli allenati
    rd = ns["esporta_tutto"](S, A, X, PV, src, DIR / "v_senza_registri", senza_registri=tuple(S.FAMIGLIE_DENSE2),
                             ricarica=False)
    cd = _controlla_file(rd["st"], rd["finale"], solo=set(S.FAMIGLIE_DENSE2))
    vd = {"codici_p2_diversi": sum(not v["codici"] for v in cd["per"].values()), "uscita_al_bit": cd["uscita_al_bit"]}
    neg_d = vd["codici_p2_diversi"] == 13 and not vd["uscita_al_bit"]
    gc.collect()
    return registra("V_cd_mutazioni_export", neg_c and neg_d, senza_scale_p2=vc, senza_registri_p2=vd) and ok_iv


# ═══════════════════════════════════ V(a,b): mutazioni dell'avanti ═══════════════════════════════════
def prova_v_ab() -> bool:
    t0 = time.time()
    g = torch.Generator().manual_seed(2915)
    x0 = torch.randn(1, 32, 2048, generator=g) * 0.5
    ids = torch.zeros(1, 32, dtype=torch.long)
    st = S.Studente(gguf=giocattolo("con"), strati=[0, 3], ricetta="B", motore_nome="torch", device="cpu",
                    esperti=slice(0, 16), log=MUTO, densi_ternari=TUTTI)
    with torch.no_grad():
        y = st.avanti(ids, x0=x0)["xL"]
        va = {}
        for L, m in [(L, m) for L in st.strati for m in st.moduli[L].modules()
                     if isinstance(m, S.LineareTernaria) and m.nome2 is not None]:
            n2, m.nome2 = m.nome2, None
            va[f"{L}.{m.nome}->{n2}"] = not torch.equal(st.avanti(ids, x0=x0)["xL"], y)
            m.nome2 = n2
        va_ritorno = bool(torch.equal(st.avanti(ids, x0=x0)["xL"], y))
    del st
    ok_a = len(va) == 13 and all(va.values()) and va_ritorno
    # V(b): cammino dequantizzato (densi spenti) == ternario in B64 entro 1e-6; mutante che ignora i *2 -> lontano
    x64 = x0.to(torch.float64)
    kw = dict(gguf=giocattolo("con"), strati=[0, 3], ricetta="B64", motore_nome="torch", device="cpu",
              esperti=slice(0, 16), log=MUTO)
    with torch.no_grad():
        y_on = S.Studente(**kw, densi_ternari=TUTTI).avanti(ids, x0=x64)["xL"]
        y_off = S.Studente(**kw).avanti(ids, x0=x64)["xL"]
        vero = S.nome_p2
        try:
            S.nome_p2 = lambda n: n + ".assente"
            y_mut = S.Studente(**kw).avanti(ids, x0=x64)["xL"]
        finally:
            S.nome_p2 = vero
    vb = {"deq_vs_ternario": rel(y_off, y_on), "mutante_deq_senza_p2": rel(y_mut, y_on)}
    ok_b = vb["deq_vs_ternario"] < 1e-6 and vb["mutante_deq_senza_p2"] > 1e-2
    gc.collect()
    return registra("V_ab_mutazioni_avanti", ok_a and ok_b, ignora_un_p2=va, ritorno=va_ritorno, dequant=vb,
                    secondi=round(time.time() - t0, 1))


# ═══════════════════════════════════ VI: passo V vero sui *2 ═══════════════════════════════════
def prova_vi() -> bool:
    """Le prove C (selezione ricalcolata a parte, byte/trit/campi d, annullamento al bit, negativi) e C2 (V-Bop come
    TT2) di e2e_test_densi_ternari, sul giocattolo `con` e con l'elenco dei densi esteso ai *2: ogni *2 e' una famiglia
    del passo V come le altre, con salti propri."""
    import types
    import e2e_test_densi_ternari as TD
    vero_S, vero_A = TD.S, TD.gguf_A
    finto = types.ModuleType("e2e_studente_con_p2")
    finto.__dict__.update(S.__dict__)
    finto.FAMIGLIE_DENSE = {**S.FAMIGLIE_DENSE, **S.FAMIGLIE_DENSE2}
    TD.S, TD.gguf_A = finto, (lambda: giocattolo("con"))
    try:
        ok_c, ok_c2 = TD.prova_c(), TD.prova_c2()
    finally:
        TD.S, TD.gguf_A = vero_S, vero_A
    per = TD.ESITO["prove"]["C_passo_V_densi"]["per_famiglia"]
    p2 = {k: v for k, v in per.items() if k.split(".")[1] in S.FAMIGLIE_DENSE2}
    con_salti = sum(v["salti"] > 0 for v in p2.values())
    ok = ok_c and ok_c2 and len(p2) == 13 and con_salti > 0
    return registra("VI_passo_V_sui_p2", ok, p2_nel_passo_V=len(p2), p2_con_salti=con_salti,
                    salti_p2=sum(v["salti"] for v in p2.values()),
                    c2=TD.ESITO["prove"]["C2_vbop_densi"].get("famiglie_dense"))


PROVE = {"I": prova_i, "II": prova_ii, "III": prova_iii, "IV": prova_iv_v, "V": prova_v_ab, "VI": prova_vi}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo", default="")
    a = ap.parse_args()
    DIR.mkdir(parents=True, exist_ok=True)
    scelte = [x for x in a.solo.split(",") if x] or list(PROVE)
    ok = True
    for k in scelte:
        ok &= PROVE[k]()
        (DIR / "esito_test_secondo_piano.json").write_text(json.dumps(ESITO, indent=1, default=str))
    print("TUTTE VERDI" if ok else "QUALCOSA E' ROSSO", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
