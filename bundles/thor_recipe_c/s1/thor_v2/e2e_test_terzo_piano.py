#!/usr/bin/env python3
"""Prove del TERZO PIANO dei densi e dei piani in piu' delle tabelle nello studente (30/9/2026, ramo tt5-terzo-piano
da cfb718f = codice di TT3c/TT4).

GGUF R2B3m (@@ROOT:models@@/gguf/tony-tern/R2B3m/Thor-T-R2B3m.gguf = R2B + 250 tensori blk.N.<nome>3.weight + testa MTP
ternaria; tabelle token_embd/output a 3 piani; motore: llamacpp-terzo-densi 0ed1faa, uscita W1 x + W2 x + W3 x). Con
--densi-ternari lo studente legge i *3 come famiglie a esperto unico PROPRIE (theta, codici del passo V, export), la
LineareTernaria somma i tre prodotti; nel cammino dequantizzato il Linear HF riceve W1 + W2 + W3; embedding e testa sono la
somma dei piani (get_rows / mm del motore). SOLO CPU, al massimo 4 fili (la GPU e' di R2B4).

GGUF giocattolo: strati 0 (GDN) e 3 (attenzione piena) di R2B3m coi byte VERI (i *2 sempre), in tre varianti:
  senza = senza i *3 (R2B su quegli strati);   con = coi *3 veri;   zero = coi *3 a campi d azzerati (W3 = 0).

  I   senza *3 == cfb718f AL BIT (albero da `git archive cfb718f`, processo figlio): chiavi, avanti, perdita, theta.grad,
      scale (densi spenti e accesi) ed export completo byte per byte. Negativo: su `con` (cfb718f ignora i *3) DIVERSO.
  II  *3 a zero == senza *3: avanti, perdita, theta.grad delle famiglie comuni AL BIT; chiavi = quelle di `senza` piu' i
      *3, DOPO tutti i primi e secondi piani di ogni strato.
  III contro llama.cpp CPU (e2e_dump_logit linkato statico su llamacpp-terzo-densi/build-cpu): R2B3m intero dai token
      (embedding a 3 piani compresa), strati 0-3, l_out-0..3. Tolleranza DICHIARATA prima: mediana <= 3 % per strato e
      mediana R2B3m <= 1,5 x quella di R2B (stesso motore, stessa misura). Negativi: studente che ignora i *3 -> mediana
      allo strato 3 >= 1,5 x la corretta; studente con embedding a UN piano -> mediana allo strato 0 >= 3 x la corretta.
  IV  export -> rilettura -> avanti AL BIT sul giocattolo `con` (theta casuali su tutte le famiglie, registri anche sui *3).
  V   mutazioni: (a) avanti che ignora UN *3 (13 nel giocattolo) -> uscita diversa; (b) cammino dequantizzato che ignora i
      *3 -> lontano dal ternario (B64, corretto entro 1e-6); (c) export senza le scale dei *3; (d) senza i loro registri.
  VI  passo V vero sui *3 (prove C e C2 di e2e_test_densi_ternari con l'elenco dei densi esteso ai *2 e *3).
  VII testa e embedding: `testa()` == somma fp32 di output, output2, output3 dequantizzati (al bit); righe
      dell'embedding (cammino a righe e tabella intera) == somma dei tre piani (al bit).

Uso (mai la GPU; al massimo 4 fili):
    systemd-run --user --scope --quiet -p CPUQuota=400% -p MemoryMax=12G nice -n 19 ionice -c3 \\
      env OMP_NUM_THREADS=4 @@ROOT:venv@@/bin/python3 e2e_test_terzo_piano.py [--solo I,II,...]
    esito: $E2E_P3_DIR/esito_test_terzo_piano.json (default @@ROOT:depot@@/e2e/prove_tt5/terzo_piano)
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

DIR = Path(os.getenv("E2E_P3_DIR", "@@ROOT:depot@@/e2e/prove_tt5/terzo_piano"))
R1 = Path(os.getenv("E2E_R2B3M", "@@ROOT:models@@/gguf/tony-tern/R2B3m/Thor-T-R2B3m.gguf"))
A2 = Path(os.getenv("E2E_R2B", "@@ROOT:models@@/gguf/tony-tern/R2B/Thor-T-R2B.gguf"))
LLAMA = Path(os.getenv("E2E_LLAMA_P3", "@@ROOT:models@@/build/llamacpp-terzo-densi"))
FILI = int(os.getenv("OMP_NUM_THREADS", "4"))
torch.set_num_threads(FILI)
BASE = "cfb718f"
ESITO: dict = {"data": time.strftime("%Y-%m-%d %H:%M:%S"), "r2b3m": str(R1), "base": BASE, "prove": {}}
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
def e_p3(nome: str) -> bool:
    parti = nome.split(".")
    return len(parti) == 4 and parti[0] == "blk" and parti[3] == "weight" and parti[2] in S.FAMIGLIE_DENSE3.values()


def giocattolo(variante: str) -> Path:
    """Strati STRATI di R2B3m coi byte veri (anche i *2) + output_norm. `senza`: senza *3; `con`: coi *3; `zero`: *3 coi
    campi d a 0."""
    assert variante in ("senza", "con", "zero")
    p = DIR / f"giocattolo_R2B3m_{variante}.gguf"
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
        p2 = e_p3(nome)
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
    js.write_text(json.dumps({"firma": firma, "tensori_p3": n_p2, "sha256": sha(p)}, indent=1))
    return p


def p3_presenti(st, L: int) -> list:
    return [n for n in st.esperti(L).fam if n in S.FAMIGLIE_DENSE3]


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
    densa = lambda n: n in S.FAMIGLIE_DENSE or n in getattr(S, "FAMIGLIE_DENSE2", {}) or n in getattr(S, "FAMIGLIE_DENSE3", {})
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
    nome = A.salva_istantanea(d, st, 7, {"prova": "terzo_piano"})
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
torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "4")))
import e2e_studente as S
assert os.path.dirname(os.path.abspath(S.__file__)) == albero, S.__file__
assert hasattr(S, "FAMIGLIE_DENSE2") and not hasattr(S, "FAMIGLIE_DENSE3"), "l'albero vecchio ha gia' il terzo piano?"
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
            else:   # negativo: sui *3 veri il codice di prima (che li ignora) DEVE dare un avanti diverso
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
    return registra("I_senza_p3_identico_a_cfb718f", ok, casi=esiti, export=exp, secondi=round(time.time() - t0, 1))


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
            senza_p3 = [k for k in b["chiavi"] if k[1] not in S.FAMIGLIE_DENSE3]
            extra = [k[1] for k in b["chiavi"] if k[1] in S.FAMIGLIE_DENSE3]
            u["chiavi_comuni_nello_stesso_ordine"] = senza_p3 == a["chiavi"]
            u["p3_in_piu"] = len(extra) == 13 and set(extra) == set(S.FAMIGLIE_DENSE3)
            u["p3_dopo_primi_e_secondi_piani"] = all(
                max(i for i, k in enumerate(b["chiavi"]) if k[0] == L and k[1] not in S.FAMIGLIE_DENSE3) <
                min(i for i, k in enumerate(b["chiavi"]) if k[0] == L and k[1] in S.FAMIGLIE_DENSE3) for L in STRATI)
        else:
            u["chiavi_identiche"] = a["chiavi"] == b["chiavi"]
        ok &= all(u.values())
        esiti[tag] = u
    return registra("II_p3_a_zero_uguale_a_senza", ok, casi=esiti, secondi=round(time.time() - t0, 1))


# ═══════════════════════════════════ III: contro llama.cpp CPU ═══════════════════════════════════
def dump_bin() -> Path:
    b = DIR / "e2e_dump_logit_p3"
    if not b.exists():
        bb = LLAMA / "build-cpu"
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
    u_r1 = dump_llama(R1, "R2B3m", ids)
    u_a2 = dump_llama(A2, "R2B", ids)
    kw = dict(strati=[0, 1, 2, 3], ricetta="B3_ind32", motore_nome="torch", device="cpu", log=MUTO,
              densi_ternari=TUTTI)
    st = S.Studente(gguf=R1, **kw)
    n_p2 = sum(len(p3_presenti(st, L)) for L in st.strati)
    e_r1 = _errori(st, ids, u_r1)
    lin = [m for m in lineari_ternarie(st) if m.nome3 is not None]
    salvati = [m.nome3 for m in lin]
    for m in lin:                                   # mutante: il forward ignora i *3
        m.nome3 = None
    e_mut = _errori(st, ids, u_r1)
    for m, n3 in zip(lin, salvati):
        m.nome3 = n3
    vero_piani = S.piani_di                         # mutante: embedding a UN piano (token_embd2/3 ignorati)
    try:
        S.piani_di = lambda tens, nome: [nome]
        e_emb = _errori(st, ids, u_r1)
    finally:
        S.piani_di = vero_piani
    del st
    gc.collect()
    sa = S.Studente(gguf=A2, **kw)
    e_a2 = _errori(sa, ids, u_a2)
    del sa
    gc.collect()
    soglia = 0.03
    crit = {"R2B3m_mediana_<=3%": all(e_r1[L]["mediana"] <= soglia for L in range(4)),
            "R2B3m_<=_1.5x_R2B": all(e_r1[L]["mediana"] <= 1.5 * e_a2[L]["mediana"] for L in range(4)),
            "mutante_senza_p3_>=_1.5x": e_mut[3]["mediana"] >= 1.5 * e_r1[3]["mediana"],
            "mutante_embedding_1_piano_>=_3x": e_emb[0]["mediana"] >= 3 * e_r1[0]["mediana"],
            "p3_nello_studente": n_p2 == 25}
    return registra("III_contro_llama_cpu", all(crit.values()), token=len(ids), p3_caricati=n_p2, R2B3m=e_r1, R2B=e_a2,
                    mutante_senza_p3=e_mut, mutante_embedding_un_piano=e_emb, criteri=crit, tolleranza="mediana errore per token <= 3% (F0b); "
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
    n_reg_p2 = sum(int(np.isin(g.righe["f"], [PV.FAMIGLIE_IDX[n] for n in S.FAMIGLIE_DENSE3]).sum()) for g in r["regs"])
    scale_p2 = [k for k in r["ist"]["scale"] if e_p3(k)]
    c = _controlla_file(st, r["finale"])
    p2_ok = [v for k, v in c["per"].items() if k.split(".")[1] in S.FAMIGLIE_DENSE3]
    ok_iv = c["uscita_al_bit"] and len(p2_ok) == 13 and len(c["per"]) == 39 and \
        all(v["codici"] and v["scale"] and v["codice_C"] for v in c["per"].values()) and len(scale_p2) == 13 and \
        n_reg_p2 > 0 and all(v["uscita_al_bit"] and v["byte_codici_uguali"] for v in r["ric"].values()) and \
        all(r["prov"]["ricarica"][str(L)]["scale_al_bit"] and r["prov"]["ricarica"][str(L)]["uscita_al_bit"]
            for L in STRATI)
    registra("IV_export_rilettura_al_bit", ok_iv, famiglie=len(c["per"]), p3_verificati=len(p2_ok),
             scale_p3_nell_istantanea=len(scale_p2), salti_sui_p3=n_reg_p2, uscita_al_bit=c["uscita_al_bit"],
             ricarica_codici=r["ric"], ricarica_scale=r["prov"]["ricarica"],
             per_famiglia=c["per"], secondi=round(time.time() - t0, 1))
    # V(c): export SENZA le scale dei *2 -> la ricarica deve sollevare (o le scale del file non sono le allenate)
    try:
        rc = ns["esporta_tutto"](S, A, X, PV, src, DIR / "v_senza_scale", senza_scale=tuple(
            f"{v}.weight" for v in S.FAMIGLIE_DENSE3.values()), ricarica=False)
        cc = _controlla_file(rc["st"], rc["finale"], solo=set(S.FAMIGLIE_DENSE3))
        vc = {"sollevato": False, "scale_p2_diverse": sum(not v["scale"] for v in cc["per"].values())}
        try:
            X.ricarica(src, rc["scalato"], {k: v for k, v in rc["ist"]["scale"].items() if not e_p3(k)}, None, [0],
                       log=MUTO)
            vc["ricarica_solleva"] = False
        except RuntimeError:
            vc["ricarica_solleva"] = True
        neg_c = vc["scale_p2_diverse"] == 13 and vc["ricarica_solleva"]
    except RuntimeError as e:
        vc, neg_c = {"sollevato": True, "errore": str(e)[:300]}, True
    # V(d): codici senza i registri dei *2 -> i codici dei *2 nel file non sono gli allenati
    rd = ns["esporta_tutto"](S, A, X, PV, src, DIR / "v_senza_registri", senza_registri=tuple(S.FAMIGLIE_DENSE3),
                             ricarica=False)
    cd = _controlla_file(rd["st"], rd["finale"], solo=set(S.FAMIGLIE_DENSE3))
    vd = {"codici_p2_diversi": sum(not v["codici"] for v in cd["per"].values()), "uscita_al_bit": cd["uscita_al_bit"]}
    neg_d = vd["codici_p2_diversi"] == 13 and not vd["uscita_al_bit"]
    gc.collect()
    return registra("V_cd_mutazioni_export", neg_c and neg_d, senza_scale_p3=vc, senza_registri_p3=vd) and ok_iv


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
                     if isinstance(m, S.LineareTernaria) and m.nome3 is not None]:
            n3, m.nome3 = m.nome3, None
            va[f"{L}.{m.nome}->{n3}"] = not torch.equal(st.avanti(ids, x0=x0)["xL"], y)
            m.nome3 = n3
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
        vero = S.nome_p3
        try:
            S.nome_p3 = lambda n: n + ".assente"
            y_mut = S.Studente(**kw).avanti(ids, x0=x64)["xL"]
        finally:
            S.nome_p3 = vero
    vb = {"deq_vs_ternario": rel(y_off, y_on), "mutante_deq_senza_p3": rel(y_mut, y_on)}
    ok_b = vb["deq_vs_ternario"] < 1e-6 and vb["mutante_deq_senza_p3"] > 1e-3
    gc.collect()
    return registra("V_ab_mutazioni_avanti", ok_a and ok_b, ignora_un_p3=va, ritorno=va_ritorno, dequant=vb,
                    secondi=round(time.time() - t0, 1))


# ═══════════════════════════════════ VI: passo V vero sui *2 ═══════════════════════════════════
def prova_vi() -> bool:
    """Le prove C (selezione ricalcolata a parte, byte/trit/campi d, annullamento al bit, negativi) e C2 (V-Bop come
    TT2) di e2e_test_densi_ternari, sul giocattolo `con` e con l'elenco dei densi esteso ai *2: ogni *2 e' una famiglia
    del passo V come le altre, con salti propri."""
    import types
    import e2e_test_densi_ternari as TD
    vero_S, vero_A = TD.S, TD.gguf_A
    finto = types.ModuleType("e2e_studente_con_p3")
    finto.__dict__.update(S.__dict__)
    finto.FAMIGLIE_DENSE = {**S.FAMIGLIE_DENSE, **S.FAMIGLIE_DENSE2, **S.FAMIGLIE_DENSE3}
    TD.S, TD.gguf_A = finto, (lambda: giocattolo("con"))
    try:
        ok_c, ok_c2 = TD.prova_c(), TD.prova_c2()
    finally:
        TD.S, TD.gguf_A = vero_S, vero_A
    per = TD.ESITO["prove"]["C_passo_V_densi"]["per_famiglia"]
    p2 = {k: v for k, v in per.items() if k.split(".")[1] in S.FAMIGLIE_DENSE3}
    con_salti = sum(v["salti"] > 0 for v in p2.values())
    ok = ok_c and ok_c2 and len(p2) == 13 and con_salti > 0
    return registra("VI_passo_V_sui_p3", ok, p3_nel_passo_V=len(p2), p3_con_salti=con_salti,
                    salti_p3=sum(v["salti"] for v in p2.values()),
                    c2=TD.ESITO["prove"]["C2_vbop_densi"].get("famiglie_dense"))


# ═══════════════════════════════════ VII: testa ed embedding a tre piani ═══════════════════════════════════
def prova_vii() -> bool:
    from dequant_gguf_torch import dequant_np, _dequant_raw
    t0 = time.time()
    st = S.Studente(gguf=R1, strati=[0], ricetta="B", motore_nome="torch", device="cpu", esperti=slice(0, 8), log=MUTO)
    tens = st._tens
    piani_o = S.piani_di(tens, "output.weight")
    piani_e = S.piani_di(tens, "token_embd.weight")
    Wo = np.asarray(dequant_np(tens["output.weight"]), dtype=np.float32)
    for p in ("output2.weight", "output3.weight"):
        Wo = Wo + np.asarray(dequant_np(tens[p]), dtype=np.float32)
    testa_ok = bool(torch.equal(st.testa(), torch.from_numpy(Wo)))
    ids = torch.tensor([[0, 17, 9999, 17, 248000, 123456, 5]])
    unici = np.unique(ids.numpy())
    E = None
    for p in ("token_embd.weight", "token_embd2.weight", "token_embd3.weight"):
        t = tens[p]
        r = np.asarray(_dequant_raw(np.asarray(t.data)[unici], t), dtype=np.float32)
        E = r if E is None else E + r
    rif = torch.from_numpy(E)[torch.searchsorted(torch.from_numpy(unici), ids[0])].to(st.ric.dtype_residuo)
    righe_ok = bool(torch.equal(st.embedding(ids)[0], rif))
    st.carica_embedding()
    tab_ok = bool(torch.equal(st.embedding(ids)[0], rif))
    diff_un_piano = float((torch.from_numpy(np.asarray(dequant_np(tens["output.weight"]), dtype=np.float32)) -
                           torch.from_numpy(Wo)).abs().max())
    del st
    gc.collect()
    ok = testa_ok and righe_ok and tab_ok and len(piani_o) == 3 and len(piani_e) == 3 and diff_un_piano > 0
    return registra("VII_testa_embedding_tre_piani", ok, piani_testa=piani_o, piani_embedding=piani_e,
                    testa_al_bit=testa_ok, embedding_righe_al_bit=righe_ok, embedding_tabella_al_bit=tab_ok,
                    diff_testa_un_piano=diff_un_piano, secondi=round(time.time() - t0, 1))


PROVE = {"I": prova_i, "II": prova_ii, "III": prova_iii, "IV": prova_iv_v, "V": prova_v_ab, "VI": prova_vi,
         "VII": prova_vii}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo", default="")
    a = ap.parse_args()
    DIR.mkdir(parents=True, exist_ok=True)
    vecchio = DIR / "esito_test_terzo_piano.json"          # 30/9: un giro --solo non cancella gli esiti degli altri
    if vecchio.exists():
        prima = json.loads(vecchio.read_text()).get("prove", {})
        ESITO["prove"] = {**prima, **ESITO["prove"]}
    scelte = [x for x in a.solo.split(",") if x] or list(PROVE)
    ok = True
    for k in scelte:
        ok &= PROVE[k]()
        (DIR / "esito_test_terzo_piano.json").write_text(json.dumps(ESITO, indent=1, default=str))
    print("TUTTE VERDI" if ok else "QUALCOSA E' ROSSO", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
