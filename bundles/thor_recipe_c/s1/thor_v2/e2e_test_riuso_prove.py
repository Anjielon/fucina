#!/usr/bin/env python3
"""Prove del ramo tt3-cure (29/9/2026, da df753ee): --pv-riuso (e2e_riuso_prove) e --pv-ricentra-obiettivo. SOLO CPU.

Giocattolo VERO: strati 0-3 di R1 (3 Gated DeltaNet + 1 attenzione piena) coi byte veri, token_embd e output_norm,
16 esperti, densi ternari TUTTI col secondo piano (*2), motore torch. Una SEQUENZA scritta come la fase V di TT3
(sonda A e lettere M1 misurate, prezzi al punto medio: gradiente in q0 + uno per primitiva, dosi golose con primitive
tenute e annullate, sonda B, ricentro: gradiente + annullamento, conferma M2), con primitive su strati e famiglie
diverse (densi di primo e secondo piano, esperti).

  R0  spento == df753ee AL BIT (albero da `git archive df753ee`, processo figlio): la sequenza intera (misure, S dei
      gradienti, byte finali) col codice congelato di TT3 e col ramo, riuso spento.
  R1  acceso == spento AL BIT, modo riga e modo segmenti: ogni misura (==), ogni S dei gradienti (torch.equal), byte
      finali; i contatori dicono che strati e indietro sono stati davvero saltati.
  R2  CICLO: `allena` (studente sintetico di e2e_test_lora, fase V a dosi + punto medio + golosa + ricentro) con e senza
      --pv-riuso: theta, F32, LoRA, byte, stato dell'ottimizzatore, storia (tolta la voce "riuso"), sha dei registri.
  R3  MUTAZIONI che devono far diventare rossa R1: (a) firma di strato costante (il prefisso non si invalida mai);
      (b) indietro troncato uno strato troppo in alto; (c) scala riusata senza la versione di theta (con theta cambiato
      a meta' sequenza).
  R4  MISURA su CPU: operazioni aten (TorchDispatchMode: ~ lanci di kernel sulla GPU), strati percorsi e secondi della
      sequenza, spento contro acceso; per strato: operazioni di un avanti MoE+densi con e senza piani/scale riusati.
  R5  --pv-ricentra-obiettivo sul «mosaico» di e2e_test_pm_golosa (G buona in entrambi i funzionali, K buona sulle
      lettere e cattiva sulla generica): spento = come df753ee al bit (modulo); spento, il ricentro toglie K (lettere
      M2 e C peggiori della dose); acceso, K resta (lettere M2 e C migliori, B <= partenza). Mutazione: accettazione
      che ignora l'obiettivo -> torna come spento (rosso).
  R7  --riprendi-da-ema: theta == theta_EMA al bit, F32 grezzi, Adam di theta azzerato (F32 tenuto), una volta sola;
      senza opzione ripresa e fine corsa == df753ee al bit; mutazioni (theta non copiato, Adam non azzerato) rosse.
  R6  etichetta del registro: il caso di TT3 (B confermata, ricentro, lettere M2 NON confermate) scrive «confermata» per
      B e dice che a bocciare e' l'obiettivo; il vecchio testo scriveva «NON confermata».

Uso (mai la GPU; al massimo 8 fili):
    governa_export.sh systemd-run --user --scope --quiet -p CPUQuota=800% -p MemoryMax=12G nice -n 19 ionice -c3 \\
      env OMP_NUM_THREADS=8 @@ROOT:venv@@/bin/python3 e2e_test_riuso_prove.py [R0 R1 ...]
    esito: $E2E_CURE_DIR/esito_test_riuso_prove.json (default @@ROOT:lavoro@@/tt3_cure_prove)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

os.environ.setdefault("TRITON_INTERPRET", "1")
for _v in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
    os.environ[_v] = ""
_D = os.environ.setdefault("E2E_CURE_DIR", "@@ROOT:lavoro@@/tt3_cure_prove")
os.environ.setdefault("E2E_PM_DIR", str(Path(_D) / "pm"))
os.environ.setdefault("E2E_VI_DIR", str(Path(_D) / "vi"))
os.environ.setdefault("E2E_PV_DIR", str(Path(_D) / "pv"))
os.environ.setdefault("E2E_ALLENA_PROVE", str(Path(_D) / "allena"))
os.environ.setdefault("E2E_LORA_DIR", str(Path(_D) / "lora"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import e2e_studente as S  # noqa: E402
import e2e_pv_codici as PV  # noqa: E402
import e2e_riuso_prove as RP  # noqa: E402

DIR = Path(_D)
R1_GGUF = Path(os.getenv("E2E_R1", "@@ROOT:models@@/gguf/tony-tern/R1/Thor-T-R1.gguf"))
BASE = os.getenv("E2E_CURE_BASE", "df753ee")
STRATI = (0, 1, 2, 3)
FILI = int(os.getenv("OMP_NUM_THREADS", "8"))
torch.set_num_threads(FILI)
ESITO: dict = {"data": time.strftime("%Y-%m-%d %H:%M:%S"), "base": BASE, "prove": {}}


def registra(nome: str, ok: bool, **dati) -> bool:
    ESITO["prove"][nome] = {"ok": bool(ok), **dati}
    print(f"[{nome}] {'OK' if ok else 'ROSSO'} {json.dumps(dati, ensure_ascii=False, default=str)[:3000]}", flush=True)
    return bool(ok)


# ═══════════════════════════════════ giocattolo da R1 ═══════════════════════════════════
def giocattolo() -> Path:
    from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType as GT
    p = DIR / "giocattolo_R1_0123.gguf"
    firma = {"sorgente": str(R1_GGUF), "dimensione": R1_GGUF.stat().st_size, "strati": list(STRATI), "versione": 1}
    js = p.with_name(p.name + ".json")
    if p.exists() and js.exists() and json.loads(js.read_text()).get("firma") == firma:
        return p
    DIR.mkdir(parents=True, exist_ok=True)
    r = GGUFReader(str(R1_GGUF))
    tmp = p.with_name(p.name + ".parziale")
    w = GGUFWriter(str(tmp), arch="qwen35moe")
    for t in r.tensors:
        nome = t.name
        if not (any(nome.startswith(f"blk.{L}.") for L in STRATI) or nome in ("output_norm.weight", "token_embd.weight")):
            continue
        dati = np.array(np.asarray(t.data), copy=True)
        if dati.dtype != np.uint8:
            dati = dati.view(np.uint8)
        w.add_tensor(nome, dati, raw_dtype=GT[t.tensor_type.name])
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    os.replace(tmp, p)
    js.write_text(json.dumps({"firma": firma}, indent=1))
    return p


# La sequenza e' testo: la stessa funzione gira col modulo del ramo e, in un processo figlio, con quello di df753ee.
SEQUENZA = r'''
def sequenza(S, PV, gguf, riuso=None, segmenti=False, cambia_theta=False, conta=None):
    """Fase V scritta: ritorna (misure, gradienti S, byte finali, contatori). `riuso`: None (spento) o una classe
    RiusoProve (acceso)."""
    import numpy as np, torch
    from contextlib import nullcontext
    st = S.Studente(gguf=gguf, strati=[0, 1, 2, 3], ricetta="B3_ind32", motore_nome="torch", device="cpu",
                    esperti=slice(0, 16), log=lambda *_: None, densi_ternari=tuple(S.FAMIGLIE_DENSE))
    with torch.no_grad():
        for (L, n), p in zip(st.chiavi(), st.parametri()):
            g = torch.Generator().manual_seed(1000 * L + sum(ord(c) for c in n))
            p.copy_(0.01 * torch.randn(p.shape, generator=g))
    for L in st.strati:
        for f in st.esperti(L).fam.values():
            PV.famiglia_scrivibile(f)
    stato = PV.StatoV()
    fams = PV.installa(st, stato, "torch")
    stato.inizia_passo()
    g = torch.Generator().manual_seed(2909)
    B, T, V = 2, 12, 248320
    lotti = {}
    for nome in ("A", "B", "M1", "M2"):
        ids = torch.randint(0, V, (B, T), generator=g)
        iz = None
        if segmenti:
            iz = torch.zeros(B, T, dtype=torch.long)
            iz[:, 0] = 1
            iz[0, 5] = 1
            iz[1, 3] = 1
            iz[1, 8] = 1
        lotti[nome] = (ids, iz, torch.randn(B * T, 2048, generator=g))

    def misura(nome):
        ids, iz, G = lotti[nome]
        with torch.no_grad():
            h = st.nascosto(ids, inizi=iz)
        return float((h.double() * G.double()).sum())

    def primitiva(k, n, seme):
        L, nome = k
        f = fams[k]
        gg = torch.Generator().manual_seed(seme)
        righe = np.zeros(n, dtype=PV.REG_DTYPE)
        usati = set()
        i = 0
        while i < n:
            e = int(torch.randint(0, f.E, (1,), generator=gg))
            r = int(torch.randint(0, f.R, (1,), generator=gg))
            c = int(torch.randint(0, f.C, (1,), generator=gg))
            if (e, r, c) in usati:
                continue
            t = int(S.trit_esperto(f, e)[r, c])
            tn = t - 1 if t == 1 else (t + 1 if t == -1 else (1 if int(torch.randint(0, 2, (1,), generator=gg)) else -1))
            usati.add((e, r, c))
            righe[i] = (L, PV.FAMIGLIE_IDX[nome], e, r, c, t, tn)
            i += 1
        return PV.Registro(righe)

    def gradiente(nome, regs):
        ids, iz, G = lotti[nome]
        salti = {}
        for reg in regs:
            for (L, fi) in reg.chiavi():
                k = (L, PV.NOMI_IDX[fi])
                rr = reg.di(L, fi)
                gs = fams[k].gs
                c = torch.from_numpy(rr["c"].astype(np.int64))
                salti[k] = (torch.from_numpy(rr["e"].astype(np.int64)), torch.from_numpy(rr["r"].astype(np.int64)),
                            c // gs, (c % gs).to(torch.uint8))
        for p in st.parametri():
            p.grad = None
        stato.prepara_ricentro(salti)
        try:
            s = st.avanti(ids, inizi=iz)
            xL = s["xL"].detach().requires_grad_(True)
            with torch.enable_grad():
                (st.esce(xL) * G).sum().backward()
            st.indietro(s, xL.grad)
        finally:
            stato.fine_raccolta()
        return {f"{k[0]}.{k[1]}": stato.buffer(fams[k]).S[salti[k][0], salti[k][1], salti[k][2]].clone() for k in salti}

    P = [((3, "attn_q2"), 6), ((1, "ssm_out"), 6), ((2, "down"), 12), ((0, "attn_qkv2"), 6), ((3, "gate"), 12)]
    regs = [primitiva(k, n, 77 + i) for i, (k, n) in enumerate(P)]
    mis, grads = [], {}
    ctx = riuso(st, stato, max_lotti=2, conta=conta) if riuso is not None else nullcontext()
    with ctx:
        mis += [misura("A"), misura("M1")]
        grads["g0_A"] = gradiente("A", regs)
        for i, reg in enumerate(regs):
            PV.rigioca(fams, [reg])
            grads[f"g1_A_{i}"] = gradiente("A", [reg])
            PV.annulla(fams, reg)
        grads["g0_M1"] = gradiente("M1", regs)
        for i, reg in enumerate(regs):
            PV.rigioca(fams, [reg])
            grads[f"g1_M1_{i}"] = gradiente("M1", [reg])
            PV.annulla(fams, reg)
        mis.append(misura("M2"))
        mis.append(misura("B"))
        mis.append(misura("M1"))
        tenuti = []
        for i, reg in enumerate(regs):
            PV.rigioca(fams, [reg])
            mis += [misura("A"), misura("M1")]
            if i in (0, 2, 3):
                tenuti.append(reg)
            else:
                PV.annulla(fams, reg)
            if cambia_theta and i == 3:
                with torch.no_grad():
                    st.esperti(2).theta["down"].add_(0.05)
        mis.append(misura("B"))
        grads["ricentro"] = gradiente("A", tenuti)
        PV.annulla(fams, tenuti[1])
        mis.append(misura("B"))
        mis.append(misura("M2"))
    byte = {f"{k[0]}.{k[1]}": f.blocchi.clone() for k, f in fams.items()}
    return {"misure": mis, "grad": grads, "byte": byte}
'''


def _sequenza_modulo():
    ns: dict = {}
    exec(SEQUENZA, ns)
    return ns["sequenza"]


def confronta(a: dict, b: dict) -> dict:
    return {"misure": a["misure"] == b["misure"], "n_misure": len(a["misure"]),
            "grad": set(a["grad"]) == set(b["grad"]) and all(
                set(a["grad"][k]) == set(b["grad"][k]) and all(torch.equal(a["grad"][k][j], b["grad"][k][j])
                                                                 for j in a["grad"][k]) for k in a["grad"]),
            "grad_non_nulli": all(any(bool((v != 0).any()) for v in d.values()) for d in a["grad"].values()),
            "byte": all(torch.equal(a["byte"][k], b["byte"][k]) for k in a["byte"])}


# ═══════════════════════════════════ R0 ═══════════════════════════════════
def albero_base() -> Path:
    d = DIR / f"albero_{BASE}"
    if not (d / "thor_v2" / "e2e_studente.py").exists():
        d.mkdir(parents=True, exist_ok=True)
        arch = subprocess.run(["git", "-C", str(QUI.parent), "archive", BASE, "thor_v2"], check=True, capture_output=True).stdout
        subprocess.run(["tar", "-x", "-C", str(d)], input=arch, check=True)
    npz = d / "thor_v2" / "mappa_completa_gguf_hf_perm.npz"     # ignorato da git: lo stesso file del ramo
    if not npz.exists():
        npz.symlink_to((QUI / "mappa_completa_gguf_hf_perm.npz").resolve())
    return d / "thor_v2"


FIGLIO = r'''
import os, sys, torch
sys.path.insert(0, sys.argv[1])
os.environ.setdefault("TRITON_INTERPRET", "1")
torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "8")))
import e2e_studente as S, e2e_pv_codici as PV
ns = {}
exec(open(sys.argv[2]).read(), ns)
seg = sys.argv[5] == "1"
torch.save(ns["sequenza"](S, PV, sys.argv[3], segmenti=seg), sys.argv[4])
'''


def prova_r0() -> bool:
    t0 = time.time()
    gg = giocattolo()
    base = albero_base()
    (DIR / "sequenza.py").write_text(SEQUENZA)
    (DIR / "figlio.py").write_text(FIGLIO)
    out = {}
    seq = _sequenza_modulo()
    for seg in (False, True):
        u = DIR / f"r0_base_{int(seg)}.pt"
        subprocess.run([sys.executable, str(DIR / "figlio.py"), str(base), str(DIR / "sequenza.py"), str(gg), str(u),
                        "1" if seg else "0"], check=True, cwd=str(base))
        a = torch.load(u, weights_only=False)
        b = seq(S, PV, gg, segmenti=seg)
        out["segmenti" if seg else "riga"] = confronta(a, b)
        u.unlink()
    ok = all(v["misure"] and v["grad"] and v["byte"] and v["grad_non_nulli"] for v in out.values())
    return registra("R0_spento_uguale_df753ee", ok, base=BASE, **out, secondi=round(time.time() - t0, 1))


# ═══════════════════════════════════ R1 / R3 ═══════════════════════════════════
def prova_r1() -> bool:
    t0 = time.time()
    gg = giocattolo()
    seq = _sequenza_modulo()
    out = {}
    for seg in (False, True):
        a = seq(S, PV, gg, segmenti=seg)
        conta: dict = {}
        b = seq(S, PV, gg, riuso=RP.RiusoProve, segmenti=seg, conta=conta)
        out["segmenti" if seg else "riga"] = {**confronta(a, b), "contatori": conta,
                                              "riuso_spento_dopo": not S.RIUSO["attivo"]}
    ok = all(v["misure"] and v["grad"] and v["byte"] and v["grad_non_nulli"] and v["riuso_spento_dopo"]
             and v["contatori"]["strati_saltati"] > 0 and v["contatori"]["strati_indietro_saltati"] > 0
             for v in out.values())
    return registra("R1_acceso_uguale_spento", ok, **out, secondi=round(time.time() - t0, 1))


class _TroncaAlto(RP.RiusoProve):
    def _prepara_ricentro(self, salti):
        r = super()._prepara_ricentro(salti)
        self._strati_ricentro = {L + 1 for L in self._strati_ricentro}
        return r


def prova_r3() -> bool:
    t0 = time.time()
    gg = giocattolo()
    seq = _sequenza_modulo()
    esiti = {}
    # (a) firma costante
    rif = seq(S, PV, gg)
    orig = RP.RiusoProve._firma
    RP.RiusoProve._firma = lambda self, L: 0
    try:
        esiti["a_firma_costante"] = confronta(rif, seq(S, PV, gg, riuso=RP.RiusoProve))
    finally:
        RP.RiusoProve._firma = orig
    # (b) indietro troncato uno strato piu' in alto
    esiti["b_tronca_alto"] = confronta(rif, seq(S, PV, gg, riuso=_TroncaAlto))
    # (c) scala riusata senza la versione di theta (theta cambiato a meta')
    rif_c = seq(S, PV, gg, cambia_theta=True)
    giusta = seq(S, PV, gg, riuso=RP.RiusoProve, cambia_theta=True)
    orig_s = S.StratoEsperti.scala

    def scala_mutata(self, nome):
        if S.RIUSO["attivo"] and not torch.is_grad_enabled():
            k = (id(self), nome)
            v = S.RIUSO["scale"].get(k)
            if v is None:
                v = S.RIUSO["scale"][k] = (None, self._scala(nome))
            return v[1]
        return self._scala(nome)
    S.StratoEsperti.scala = scala_mutata
    try:
        esiti["c_scala_senza_versione"] = confronta(rif_c, seq(S, PV, gg, riuso=RP.RiusoProve, cambia_theta=True))
    finally:
        S.StratoEsperti.scala = orig_s
    corretta_c = confronta(rif_c, giusta)
    rossi = {k: not (v["misure"] and v["grad"] and v["byte"]) for k, v in esiti.items()}
    ok = all(rossi.values()) and corretta_c["misure"] and corretta_c["grad"] and corretta_c["byte"]
    return registra("R3_mutazioni_rosse", ok, rossi=rossi, dettaglio=esiti, theta_cambiato_col_riuso_giusto=corretta_c,
                    secondi=round(time.time() - t0, 1))


# ═══════════════════════════════════ R4: misura ═══════════════════════════════════
class _ContaOp(torch.utils._python_dispatch.TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.n = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.n[str(func.overloadpacket.__name__)] += 1
        return func(*args, **(kwargs or {}))


def prova_r4() -> bool:
    t0 = time.time()
    gg = giocattolo()
    seq = _sequenza_modulo()
    ris = {}
    for tag, riuso in (("spento", None), ("acceso", RP.RiusoProve)):
        seq(S, PV, gg, riuso=riuso)                        # riscaldamento (caricamento, prime allocazioni)
        conta: dict = {}
        c = _ContaOp()
        t1 = time.time()
        with c:
            seq(S, PV, gg, riuso=riuso, conta=conta)
        sec_op = time.time() - t1
        t1 = time.time()
        seq(S, PV, gg, riuso=riuso)
        ris[tag] = {"operazioni_aten": sum(c.n.values()), "secondi": round(time.time() - t1, 2),
                    "secondi_con_conteggio": round(sec_op, 2), "contatori": conta,
                    "prime_op": c.n.most_common(12)}
    # per strato: UN avanti di ogni strato, piani/scale riusati o no (prefisso escluso: stesso strato, stesso ingresso)
    st = S.Studente(gguf=gg, strati=list(STRATI), ricetta="B3_ind32", motore_nome="torch", device="cpu",
                    esperti=slice(0, 16), log=lambda *_: None, densi_ternari=tuple(S.FAMIGLIE_DENSE))
    ids = torch.randint(0, 248320, (2, 12), generator=torch.Generator().manual_seed(3))
    per_strato = {}
    for tag, on in (("spento", False), ("acceso", True)):
        S.riuso_attivo(on)
        try:
            with torch.no_grad():
                x = st.embedding(ids)
                ctx = st._contesto(2, 12, x.dtype)
                st._strato(0, x, ctx)                     # riempie le cache (acceso) / riscalda (spento)
                per = {}
                for L in STRATI:
                    c = _ContaOp()
                    with c:
                        st._strato(L, x, ctx)
                    per[L] = sum(c.n.values())
        finally:
            S.riuso_attivo(False)
        per_strato[tag] = per
    ris["operazioni_per_strato_avanti"] = per_strato
    a, b = ris["spento"], ris["acceso"]
    ris["rapporto_operazioni"] = round(b["operazioni_aten"] / a["operazioni_aten"], 4)
    ris["rapporto_secondi"] = round(b["secondi"] / a["secondi"], 4)
    ok = b["operazioni_aten"] < a["operazioni_aten"]
    return registra("R4_misura_cpu", ok, **ris, secondi=round(time.time() - t0, 1))


# ═══════════════════════════════════ R2: ciclo allena ═══════════════════════════════════
def prova_r2() -> bool:
    import e2e_allena as A
    import e2e_test_integrata as TI
    import e2e_test_lora as TL
    t0 = time.time()
    src, WS = TL.sorgente_lora(n_train=12, n_held=4)
    src = TI.con_domini(src)
    kw = dict(pv_sonda_dosi=True, pv_ricentra=True, pv_dosi_frazioni=(1 / 16, 1 / 4, 1.0), pv_punto_medio=True,
              pv_pm_max=4, pv_pm_golosa=True)
    ris = {}
    for nome, on in (("spento", False), ("acceso", True)):
        d = DIR / f"r2_{nome}"
        TI._pulisci(d)
        st = TI._studente_int()
        z = A.allena(st, src, d, TI._ca_integrata(pv_riuso=on, **kw), WS, WS, log=TI.muto)
        ris[nome] = (st, z, A.carica_ripresa(d, "cpu"))
    (sa, za, ra), (sb, zb, rb) = ris["spento"], ris["acceso"]
    a_, b_ = TI._stato_confrontabile(za), TI._stato_confrontabile(zb)
    riusi = [x["pv"].pop("riuso", None) for x in b_.get("storia", []) if x.get("pv")]
    for s_ in (a_, b_):
        s_["config"].pop("pv_riuso", None)
    u = {"theta": TI._uguali_lista(sb.parametri(), sa.parametri()),
         "f32": TI._uguali_lista(sb.parametri_f32(), sa.parametri_f32()),
         "lora": TI._uguali_lora_ga(sb, sa), "byte": TI._uguali_lista(TI._byte(sb), TI._byte(sa)),
         "opt": TI._uguali_stato_opt(rb["opt"], ra["opt"]),
         "stato": json.dumps(a_, sort_keys=True, default=str) == json.dumps(b_, sort_keys=True, default=str),
         "registri_sha": zb["pv"]["sha"] == za["pv"]["sha"]}
    pv = [(x["pv"]["salti"], x["pv"]["tenuto"], x["pv"].get("salti_tenuti")) for x in za["storia"] if x.get("pv")]
    ok = all(u.values()) and any(r for r in riusi)
    return registra("R2_ciclo_acceso_uguale_spento", ok, uguali=u, pv=pv, riuso=riusi,
                    secondi=round(time.time() - t0, 1))


# ═══════════════════════════════════ R5 / R6: ricentro con l'obiettivo ═══════════════════════════════════
SPEC_R5 = (("G", "g", 8, 1.0, 1.0), ("K", "k", 4, 0.0, 1.0))


def _r5(M, on: bool, gio=None):
    import e2e_test_pm_golosa as TG
    gio = gio or TG.Mosaico(M, spec=SPEC_R5)
    cfg = M.ConfigDosi(punto_medio=True, pm_golosa=True, ricentra=True,
                       **({"ricentra_obiettivo": True} if on else {}))
    gio, fams, pv, reg, diag, es, L = TG.passo_v_mosaico(M, cfg, gio=gio, ricentra_grad=True)
    return {"tenuti": es["salti_tenuti"], "per_famiglia": gio.conta(fams), "ricentro": es["ricentro"],
            "M2": gio.misura(fams, L["M2"], "obj"), "C": gio.misura(fams, L["C"], "obj"),
            "B": gio.misura(fams, L["B"]), "B_prima": es["sonda_prima_b"], "esito": es,
            "M2_prima": (es.get("punto_medio") or {}).get("obiettivo", {}).get("prima")}


def prova_r5() -> bool:
    import e2e_test_pm_golosa as TG
    t0 = time.time()
    # spento == df753ee (modulo e2e_pv_codici di BASE)
    import importlib.util
    testo = subprocess.run(["git", "-C", str(QUI), "show", f"{BASE}:thor_v2/e2e_pv_codici.py"], check=True,
                           capture_output=True, text=True).stdout
    VEC = TG._modulo_da_testo(testo, f"e2e_pv_codici_{BASE}")
    del importlib
    v, n = _r5(VEC, False), _r5(PV, False)
    spento_al_bit = (json.dumps(v["esito"], sort_keys=True, default=str) == json.dumps(n["esito"], sort_keys=True,
                                                                                      default=str)
                     and v["per_famiglia"] == n["per_famiglia"])
    acc = _r5(PV, True)
    # mutazione: l'accettazione del giro ignora l'obiettivo
    orig = PV.PassoV._ricentra

    def mutato(self, *a, **k):
        k.pop("misura_obj", None)
        k.pop("obiettivo", None)
        return orig(self, *a, **k)
    PV.PassoV._ricentra = mutato
    try:
        mut = _r5(PV, True)
    finally:
        PV.PassoV._ricentra = orig
    k_tolta_spento = n["per_famiglia"]["K"] < acc["per_famiglia"]["K"]
    meglio = acc["M2"] < n["M2"] and acc["C"] < n["C"] and acc["B"] <= acc["B_prima"]
    mut_rossa = mut["per_famiglia"] == n["per_famiglia"]
    ok = spento_al_bit and k_tolta_spento and meglio and mut_rossa
    riass = lambda r: {k: r[k] for k in ("tenuti", "per_famiglia", "M2", "M2_prima", "C", "B", "B_prima")}  # noqa: E731
    return registra("R5_ricentro_con_obiettivo", ok, spento_al_bit=spento_al_bit, spento=riass(n), acceso=riass(acc),
                    mutazione=riass(mut), giri_spento=n["ricentro"]["giri"] if n["ricentro"] else None,
                    giri_acceso=acc["ricentro"]["giri"] if acc["ricentro"] else None, secondi=round(time.time() - t0, 1))


def prova_r6() -> bool:
    import e2e_allena as A
    ev = {"dosi": [{"salti": 1540, "tenuta": True, "sonda_a": 0.28378, "obiettivo": 0.32557}],
          "dose": {"salti": 64290, "frazione": 0.004}, "salti_tenuti": 0, "salti": 16070289,
          "conferma": {"prima": 0.39762, "dose": 0.39691, "dopo": 0.39369, "confermata": False},
          "ricentro": {"valido": True, "dannosi": 29914, "annullati": 14957, "tenuti": 49333, "giri": [{}, {}]}}
    t = A.testo_dosi(ev)
    ev2 = json.loads(json.dumps(ev))
    ev2["conferma"]["dose"] = 0.39800                     # B davvero non confermata: resta «NON confermata»
    t2 = A.testo_dosi(ev2)
    ok = ("sonda B 0.39762 -> 0.39691 confermata" in t and "obiettivo" in t
          and "NON confermata" in t2 and "obiettivo" not in t2)
    return registra("R6_etichetta_registro", ok, testo_tt3=t, testo_b_non_confermata=t2)


# ═══════════════════════════════════ R7: --riprendi-da-ema ═══════════════════════════════════
def prova_r7() -> bool:
    """`allena` (studente sintetico, AdamW, ema_theta 0,9, F32 accesi, passo V a dosi) fermato al passo 2; poi:
    (a) ripresa --riprendi-da-ema senza passi: theta == theta_EMA della ripresa AL BIT, F32 == grezzi della ripresa, Adam
        di theta vuoto, Adam dei F32 == ripresa, EMA == ripresa, voce nello stato; (b) ripresa di nuovo con l'opzione
        (voce gia' nello stato): IGNORATA, theta resta; (c) SENZA opzione, ripresa e fine della corsa: == df753ee al bit
        (theta, F32, byte, sha dei registri); (d) mutazioni: theta non copiato / Adam non azzerato -> (a) rossa."""
    import copy
    import e2e_allena as A
    import e2e_test_allena as TA
    import e2e_test_integrata as TI
    import e2e_test_lora as TL
    t0 = time.time()
    src, WS = TL.sorgente_lora(n_train=12, n_held=4)
    src = TI.con_domini(src)
    kw = dict(ottimizzatore_theta="adamw", ema_theta=0.9, pv_sonda_dosi=True, pv_ricentra=True,
              pv_dosi_frazioni=(1 / 16, 1 / 4, 1.0))
    ca = lambda: TI._ca_integrata(**kw)  # noqa: E731
    d = DIR / "r7_base"
    TI._pulisci(d)
    A.allena(TI._studente_int(), src, d, ca(), WS, WS, ferma_a=2, log=TI.muto)
    rip = A.carica_ripresa(d, "cpu")

    def riprendi(cartella, da_ema, mod=A):
        TI._pulisci(cartella)
        import shutil
        shutil.copytree(d, cartella, dirs_exist_ok=True)
        st = TI._studente_int()
        lg: list = []
        cattura = {}
        orig = mod.nuovo_ottimizzatore

        def spia(*a, **k):
            o = orig(*a, **k)
            cattura["opt"] = o
            return o
        mod.nuovo_ottimizzatore = spia
        try:
            z = mod.allena(st, src, cartella, ca(), WS, WS, riprendi=True, ferma_a=2, log=lg.append,
                           **({"da_ema": True} if da_ema else {}))
        finally:
            mod.nuovo_ottimizzatore = orig
        return st, z, cattura["opt"], lg

    def controlla_a(st, z, opt):
        th = all(torch.equal(p.detach(), rip["ema"][A._chiave(k)]) for k, p in zip(st.chiavi(), st.parametri()))
        f32 = all(torch.equal(a_, b_) for a_, b_ in zip(st.valori_f32().values(), rip["f32"].values())) \
            if isinstance(rip["f32"], dict) else True
        adam_theta = all(p not in opt.state for p in st.parametri())
        adam_f32 = all(p in opt.state for p in st.parametri_f32())
        voce = bool(z.get("ripresa_da_ema"))
        return {"theta_uguale_ema": th, "f32_grezzi": f32, "adam_theta_vuoto": adam_theta, "adam_f32_tenuto": adam_f32,
                "voce": voce}
    st_a, z_a, opt_a, lg_a = riprendi(DIR / "r7_a", True)
    a = controlla_a(st_a, z_a, opt_a)
    a["theta_diverso_dal_grezzo"] = any(not torch.equal(p.detach(), rip["theta"][A._chiave(k)])
                                        for k, p in zip(st_a.chiavi(), st_a.parametri()))
    # (b) una volta sola: ripresa (salvata da (a)? no: ferma_a=2 non salva) -> stato con la voce scritto a mano
    d_b = DIR / "r7_b"
    TI._pulisci(d_b)
    import shutil
    shutil.copytree(d, d_b, dirs_exist_ok=True)
    pr = json.loads((d_b / "ripresa_ultima.json").read_text())
    ogg = torch.load(d_b / pr["file"], weights_only=False)
    ogg["stato"]["ripresa_da_ema"] = [{"passo": 1}]
    sha = A._torch_save_atomico(ogg, d_b / pr["file"])
    pr["sha256"] = sha
    (d_b / "ripresa_ultima.json").write_text(json.dumps(pr))
    idx = json.loads((d_b / "riprese.json").read_text())
    for v in idx:
        if v["file"] == pr["file"]:
            v["sha256"] = sha
    (d_b / "riprese.json").write_text(json.dumps(idx))
    st_b = TI._studente_int()
    lg_b: list = []
    A.allena(st_b, src, d_b, ca(), WS, WS, riprendi=True, ferma_a=2, log=lg_b.append, da_ema=True)
    b = {"theta_grezzo": all(torch.equal(p.detach(), rip["theta"][A._chiave(k)])
                             for k, p in zip(st_b.chiavi(), st_b.parametri())),
         "log_ignorato": any("IGNORATO" in str(m) for m in lg_b)}
    # (c) ripresa scritta da df753ee (come quella vera di TT3): il ramo SENZA opzioni la continua come df753ee, al bit;
    # (e) il ramo la riprende con la riga del rilancio (--pv-riuso, lr cambiato, --riprendi-da-ema)
    VEC = TA._modulo_prima(BASE)
    ca_v = lambda: VEC.ConfigAllena(**{k: v for k, v in ca().__dict__.items()  # noqa: E731
                                       if k in VEC.ConfigAllena.__dataclass_fields__})
    d_v = DIR / "r7_base_df753ee"
    TI._pulisci(d_v)
    VEC.allena(TI._studente_int(), src, d_v, ca_v(), WS, WS, ferma_a=2, log=TI.muto)
    rip_v = A.carica_ripresa(d_v, "cpu")
    fine = {}
    for nome, mod, cfg in (("nuovo", A, ca), ("base", VEC, ca_v)):
        dd = DIR / f"r7_c_{nome}"
        TI._pulisci(dd)
        shutil.copytree(d_v, dd, dirs_exist_ok=True)
        st = TI._studente_int()
        z = mod.allena(st, src, dd, cfg(), WS, WS, riprendi=True, log=TI.muto)
        fine[nome] = (st, z)
    (sn, zn), (sv, zv) = fine["nuovo"], fine["base"]
    c = {"theta": TI._uguali_lista(sn.parametri(), sv.parametri()),
         "f32": TI._uguali_lista(sn.parametri_f32(), sv.parametri_f32()),
         "byte": TI._uguali_lista(TI._byte(sn), TI._byte(sv)), "registri_sha": zn["pv"]["sha"] == zv["pv"]["sha"],
         "passi": zn["passo"] == zv["passo"] and zn["passo"] > 2}
    de = DIR / "r7_e"
    TI._pulisci(de)
    shutil.copytree(d_v, de, dirs_exist_ok=True)
    st_e = TI._studente_int()
    lg_e: list = []
    ca_e = TI._ca_integrata(**{**kw, "pv_riuso": True, "lr": ca().lr / 5})
    z_e = A.allena(st_e, src, de, ca_e, WS, WS, riprendi=True, ferma_a=3, log=lg_e.append, da_ema=True)
    e = {"voce_da_ema": bool(z_e.get("ripresa_da_ema")) and z_e["ripresa_da_ema"][0]["passo"] == 2,
         "un_passo_fatto": z_e["passo"] == 3,
         "campi_cambiati_nel_registro": any("ripresa con campi cambiati" in str(m) for m in lg_e),
         "riuso_nella_storia": any((x.get("pv") or {}).get("riuso") for x in z_e["storia"])}
    # (d) mutazioni
    mut = {}
    orig = A._riprendi_da_ema

    def m_theta(st, opt, rip_, stato, log):
        rip2 = dict(rip_)
        rip2["ema"] = rip_["theta"]
        return orig(st, opt, rip2, stato, log)

    def m_adam(st, opt, rip_, stato, log):
        salvato = {p: copy.deepcopy(v) for p, v in opt.state.items()}   # chiavi = gli STESSI parametri
        orig(st, opt, rip_, stato, log)
        opt.state.update(salvato)
    for nome, f in (("theta_non_copiato", m_theta), ("adam_non_azzerato", m_adam)):
        A._riprendi_da_ema = f
        try:
            s_, z_, o_, _ = riprendi(DIR / f"r7_m_{nome}", True)
            r_ = controlla_a(s_, z_, o_)
            mut[nome] = {"rossa": not all(r_.values()), **r_}
        finally:
            A._riprendi_da_ema = orig
    ok = (all(a.values()) and all(b.values()) and all(c.values()) and all(e.values())
          and all(v["rossa"] for v in mut.values()))
    return registra("R7_riprendi_da_ema", ok, a_ripresa_da_ema=a, b_una_volta=b, c_spento_uguale_base=c,
                    e_riga_del_rilancio_su_ripresa_df753ee=e,
                    mutazioni=mut, log=[str(m) for m in lg_a if "EMA" in str(m)][:3], secondi=round(time.time() - t0, 1))


# R2 cambia la configurazione globale dello studente (giocattolo sintetico): va lanciata da sola, o per ultima
PROVE = {"R0": prova_r0, "R1": prova_r1, "R2": prova_r2, "R3": prova_r3, "R4": prova_r4, "R5": prova_r5,
         "R6": prova_r6, "R7": prova_r7}


def main() -> int:
    DIR.mkdir(parents=True, exist_ok=True)
    for v in ("E2E_PM_DIR", "E2E_PV_DIR", "E2E_VI_DIR", "E2E_ALLENA_PROVE", "E2E_LORA_DIR"):
        Path(os.environ[v]).mkdir(parents=True, exist_ok=True)
    scelte = [a for a in sys.argv[1:] if a in PROVE] or list(PROVE)
    esito_p = DIR / "esito_test_riuso_prove.json"
    if esito_p.exists():
        try:
            ESITO["prove"].update(json.loads(esito_p.read_text()).get("prove", {}))
        except Exception:  # noqa: BLE001
            pass
    ok = True
    for nome in scelte:
        t0 = time.time()
        try:
            ok &= PROVE[nome]()
        except Exception as ex:  # noqa: BLE001
            traceback.print_exc()
            ok &= registra(f"{nome}_eccezione", False, errore=repr(ex))
        print(f"  ({nome}: {time.time() - t0:.1f} s)", flush=True)
        esito_p.write_text(json.dumps(ESITO, indent=1, default=str))
    print("ESITO:", "TUTTO VERDE" if ok else "QUALCOSA NON TORNA", "->", esito_p, flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
