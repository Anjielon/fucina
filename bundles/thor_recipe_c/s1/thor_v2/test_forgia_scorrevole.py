#!/usr/bin/env python3
"""Collaudo a scala GIOCATTOLO di `forgia_scorrevole.py` (CPU, zero Thor, zero GPU).

Uso (regola di sessione: la GPU e' della forgia):
    env CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= OMP_NUM_THREADS=2 nice -n 19 \\
        @@ROOT:venv@@/bin/python3 test_forgia_scorrevole.py [--cartella DIR]

Cosa dimostra (ognuna delle proprieta' chieste per la ricetta scorrevole):
  T1 pianificazione: 39 finestre per 40 strati, passo 1, = pianifica_finestre(40,2,overlap).
  T2 catena completa (strati Qwen3_5MoeDecoderLayer VERI a dimensioni giocattolo):
     - seme della finestra k = forward DIRETTO degli strati 0..k-1 commessi (moduli
       ricostruiti dai codici su disco) e = ricostruisci_seme (strato per strato);
     - ogni strato commesso UNA volta, coi codici della SUA ULTIMA finestra;
     - lo strato portato entra nella finestra dopo coi codici appena ottimizzati
       (non dal teacher); lo strato entrante parte da teacher + LS;
     - ogni strato caricato una volta sola; in memoria solo la finestra corrente;
     - i codici su disco ridanno i pesi commessi nel modulo.
  T3 W=3 (generalita' del ciclo).
  T4 ripresa: interruzioni in 5 punti diversi -> ripresa -> codici IDENTICI al bit
     alla corsa senza interruzioni; ricetta cambiata e file corrotto RIFIUTATI.
  T5 modalita' (b): strati entranti inizializzati dai codici di una corsa precedente.
  T6 riuso del teacher catturato = teacher ricalcolato (codici identici).
  T7 contratto dell'interfaccia A: oggetto portato usato TALE E QUALE (lr=0 ->
     codici invariati) — sul sostituto oggi, sull'interfaccia vera quando c'e'.
  T8 bf16: scarto fra catena e forward diretto misurato (informativo).

⛔ SOSTITUTO DELL'INTERFACCIA A: finche' la patch del compito A non e' applicata,
`ottimizza_finestra_JOINT` non ha `ternari_iniziali`/`ternari_uscita`. Per
collaudare l'orchestratore ADESSO, `SostitutoA` li realizza intercettando (solo
dentro questo test, ripristinati sempre) `_costruisci_ternari_layer` e `reinit_ls_`
di valida: lo strato portato viene restituito tale e quale al posto della
costruzione dal teacher, e reinit_ls_ lo salta. Non e' il compito A e non va in
produzione. Quando l'interfaccia vera c'e', T2/T4/T7 girano ANCHE su di essa."""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import valida_finestra_8_12 as V  # noqa: E402
import forgia_scorrevole as FS  # noqa: E402
from ottimizzazione_globale_loop import pianifica_finestre  # noqa: E402
from quant_ternaria_globale_batch_v2 import TernarioDiffBatchFuso  # noqa: E402

ESITI: list = []


def verifica(nome: str, ok: bool, dettaglio: str = "") -> bool:
    ESITI.append((nome, bool(ok)))
    print(f"    {'✅' if ok else '⛔'} {nome}" + (f" — {dettaglio}" if dettaglio else ""), flush=True)
    return bool(ok)


# ═══════════════════════ strati giocattolo (infrastruttura di valida, riusata) ═══════════════════════
class Giocattolo:
    """Fabbrica deterministica: teacher = seme 100+L, corrente = teacher + rumore 900+L
    (come `_test_ottimizza_finestra_JOINT_giocattolo`). Registra ogni caricamento e
    tiene un riferimento a ogni modulo creato (per i controlli dopo la corsa)."""

    def __init__(self, n: int, dtype=torch.float32):
        self.n, self.dtype = n, dtype
        self.cfg = V._costruisci_config_giocattolo(n)
        self.cfg._attn_implementation = "sdpa"
        g = torch.Generator().manual_seed(7)
        self.emb = (torch.randn(self.cfg.vocab_size, self.cfg.hidden_size, generator=g)).to(dtype)
        self.ids = torch.randint(0, self.cfg.vocab_size, (3, 2, 6), generator=g)   # 2 addestramento + 1 held-out
        self.caricati: list = []
        self.quant_creati: dict = {}

    def strato(self, L: int, quant: bool):
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeDecoderLayer
        p = Qwen3_5MoeDecoderLayer(self.cfg, layer_idx=L)
        V._init_pesi_giocattolo(p, seed=100 + L)
        if quant:
            q = Qwen3_5MoeDecoderLayer(self.cfg, layer_idx=L)
            with torch.no_grad():
                g = torch.Generator().manual_seed(900 + L)
                for (_, a), (_, b) in zip(p.named_parameters(), q.named_parameters()):
                    b.copy_(a + torch.randn(a.shape, generator=g) * 0.15)
            p = q
        p = p.to(self.dtype).eval()
        for par in p.parameters():
            par.requires_grad_(False)
        return p

    def carica(self, layers):
        self.caricati.append(list(layers))
        pul = {L: self.strato(L, False) for L in layers}
        qua = {L: self.strato(L, True) for L in layers}
        for L in layers:
            self.quant_creati.setdefault(L, []).append(qua[L])
        return self.cfg, self.emb.clone(), pul, qua

    def batch(self):
        return self.ids.clone(), 2

    def seme0(self, ids_batch, emb):
        h, e = {}, {}
        for b in range(ids_batch.shape[0]):
            h[b] = torch.nn.functional.embedding(ids_batch[b], emb).to(emb.dtype)
            g = torch.Generator().manual_seed(500 + b)
            e[b] = (torch.randn(h[b].shape, generator=g) * 0.05).float()
        return h, e


# ═══════════════════════ SOSTITUTO dell'interfaccia A (solo test) ═══════════════════════
class SostitutoA:
    """Realizza A1 (ternari_iniziali / ternari_uscita) intercettando la costruzione dei
    ternari dentro `ottimizza_finestra_JOINT` vera — nessuna riga di valida cambiata."""

    def __init__(self):
        self.init_codici: dict = {}     # (chiamata, L) -> codici all'init (dopo LS o portati)
        self.chiamata = 0

    def ottimizza(self, strati_correnti, ids_batch, cfg, n_passi, strati_teacher, emb,
                  ternari_iniziali=None, ternari_uscita=None, **kw):
        self.chiamata += 1
        mappa = {}
        for sorg in (strati_teacher, strati_correnti):
            for L, s in sorg.items():
                mappa[id(s.mlp.experts.gate_up_proj)] = (L, "gu")
                mappa[id(s.mlp.experts.down_proj)] = (L, "dw")
        costruiti: dict = {}
        chi: dict = {}                  # id(oggetto) -> (L, famiglia, portato?)
        orig_costr, orig_reinit = V._costruisci_ternari_layer, V.reinit_ls_

        def costr(tensore, S, group, ruota):
            L, fam = mappa[id(tensore)]
            dato = (ternari_iniziali or {}).get(L, {}).get(fam)
            if dato is not None:
                assert dato.group == group and tuple(dato.shape) == tuple(tensore.shape)
                obj = dato
            else:
                obj = orig_costr(tensore, S, group, ruota)
            chi[id(obj)] = (L, fam, dato is not None)
            costruiti.setdefault(L, {})[fam] = obj
            self.init_codici[(self.chiamata, L, fam)] = (dato is not None, V._estrai_famiglia(obj))
            return obj

        def reinit(obj, W, **k):
            L, fam, portato = chi[id(obj)]
            if portato:
                return {"frazione_disaccordo_ls": 0.0, "frazione_gruppi_morti": 0.0}
            r = orig_reinit(obj, W, **k)
            self.init_codici[(self.chiamata, L, fam)] = (False, V._estrai_famiglia(obj))
            return r

        V._costruisci_ternari_layer, V.reinit_ls_ = costr, reinit
        try:
            ris = V.ottimizza_finestra_JOINT(strati_correnti, ids_batch, cfg, n_passi,
                                            strati_teacher=strati_teacher, emb=emb, **kw)
        finally:
            V._costruisci_ternari_layer, V.reinit_ls_ = orig_costr, orig_reinit
        if ternari_uscita is not None:
            ternari_uscita.clear()
            ternari_uscita.update({L: {"gu": d["gu"], "dw": d.get("dw")} for L, d in costruiti.items()})
        return ris


def sostituto_da_codici(codici, cfg, device):
    """A2 sostituto: z = Q (|z|=1 sopra la soglia 0,5), log_s = log D."""
    def _uno(D, Q):
        E, n_out, n_in = Q.shape
        ng = D.shape[-1]
        obj = TernarioDiffBatchFuso(torch.zeros(E, n_out, n_in, device=device), group=n_in // ng)
        with torch.no_grad():
            obj.log_s.copy_(torch.from_numpy(D).log().unsqueeze(-1))
            obj.z.copy_(torch.from_numpy(Q).float().reshape(E, n_out, ng, n_in // ng))
            p = 0.5 / obj.delta_max
            obj.delta_raw.fill_(math.log(p) - math.log1p(-p))
        return obj
    D, Q = FS._codici_gate_up_fusi(codici)
    return {"gu": _uno(D, Q),
            "dw": _uno(*codici["ffn_down_exps"]) if "ffn_down_exps" in codici else None}


def interfaccia_vera_presente() -> bool:
    try:
        FS.verifica_interfaccia_A(V.ottimizza_finestra_JOINT, serve_da_codici=False)
        return True
    except RuntimeError:
        return False


# ═══════════════════════ esecuzione di una corsa giocattolo ═══════════════════════
class Osservatore:
    def __init__(self):
        self.eventi: list = []
        self.semi: dict = {}
        self.init_portati: dict = {}
        self.risultati: dict = {}
        self.attivi: dict = {}
        self.entranti: dict = {}

    def __call__(self, evento, **d):
        self.eventi.append((evento, d.get("finestra"), d.get("strato")))
        k = d.get("finestra")
        if evento == "carica":
            self.attivi[k] = d["attivi"]
        elif evento == "finestra_inizio":
            cl = (lambda x: None if x is None else {b: t.detach().clone() for b, t in x.items()})
            self.semi[k] = (cl(d["h_seme"]), cl(d["e_seme"]))
            self.entranti[k] = list(d["entranti"])
            self.init_portati[k] = {L: {f: V._estrai_famiglia(o) for f, o in fam.items() if o is not None}
                                    for L, fam in d["iniziali"].items()}
        elif evento == "finestra_fine":
            self.risultati[k] = d["risultato"]


def corsa(cartella: Path, gioc: Giocattolo, n_passi: int = 3, w: int = 2, corsa_id: str = "prova",
          asincrona: bool = True, interrompi=None, ottimizza_fn=None, **kw):
    V.OUT_DIR = cartella / "out"
    V.ID_CORSA = corsa_id
    oss = Osservatore()
    FS.esegui(gioc.n, w, n_passi, kw.pop("modo_init", "teacher"), kw.pop("init_dir", ""),
              kw.pop("init_corsa", ""), salta_export=True, carica_strati=gioc.carica,
              carica_batch=gioc.batch, seme_iniziale=gioc.seme0,
              ottimizza_fn=ottimizza_fn or SostitutoA().ottimizza,
              da_codici_fn=sostituto_da_codici, ripresa_dir=cartella / "ripresa",
              scrittura_asincrona=asincrona, osservatore=oss, interrompi=interrompi, **kw)
    return oss


def codici_su_disco(cartella: Path, n: int, corsa_id: str) -> dict:
    return {L: FS.leggi_codici_strato(cartella / "out", L, corsa_id, V.DOWN_TERNARIO) for L in range(n)}


def uguali(c1: dict, c2: dict) -> bool:
    return all(np.array_equal(c1[f][0], c2[f][0]) and np.array_equal(c1[f][1], c2[f][1]) for f in c1) \
        and sorted(c1) == sorted(c2)


def seme_diretto(gioc: Giocattolo, k: int, cartella: Path, corsa_id: str):
    """Forward DIRETTO (una sola chiamata di confine su tutti gli strati 0..k-1), moduli
    correnti ricostruiti dai codici su disco."""
    pul, qua = {}, {}
    S = torch.from_numpy(V._S1024_NP)
    for L in range(k):
        pul[L] = gioc.strato(L, False)
        qua[L] = gioc.strato(L, True)
        FS.inietta_codici_nel_modulo(qua[L], FS.leggi_codici_strato(cartella / "out", L, corsa_id, True), S)
    h0, e0 = gioc.seme0(gioc.ids, gioc.emb)
    return V.calcola_confine_finestra(pul, qua, gioc.ids, gioc.cfg, gioc.emb, h0_ingresso_per_b=h0,
                                      e0_ingresso_per_b=e0)


# ═══════════════════════ i test ═══════════════════════
def t0_percorso_ruotato():
    """A scala giocattolo n_in=8 la rotazione e' l'identita': qui gate_up ha la larghezza
    VERA di Thor (hidden 2048 -> rotazione a blocchi di 1024 attiva, gruppo 128) e down
    512 (gruppo 256), con E=2 e poche righe. Il commit e' fatto ESATTAMENTE come in
    ottimizza_finestra_JOINT; i codici estratti come li estrae lei."""
    print("\n── T0 percorso ruotato (hidden 2048, rotazione attiva): codici -> pesi = commit ──")
    torch.manual_seed(3)
    S = torch.from_numpy(V._S1024_NP)
    E, meta, hid, inter = 2, 4, 2048, 512
    W_gu = torch.randn(E, 2 * meta, hid) * 0.02
    W_dw = torch.randn(E, hid, inter) * 0.02
    Wr = V._ruota_se_possibile(W_gu, S)
    tg = TernarioDiffBatchFuso(Wr, group=V.GRUPPO_GATEUP_REALE)
    V.reinit_ls_(tg, Wr)
    td = TernarioDiffBatchFuso(W_dw, group=V.GRUPPO_DOWN_REALE)
    V.reinit_ls_(td, W_dw)
    with torch.no_grad():
        tg.z.add_(torch.randn_like(tg.z) * 0.3)
        tg.log_s.add_(torch.randn_like(tg.log_s) * 0.2)

    class _M(torch.nn.Module):
        pass
    mod = _M()
    mod.mlp = _M()
    mod.mlp.experts = _M()
    mod.mlp.experts.gate_up_proj = torch.nn.Parameter(torch.zeros(E, 2 * meta, hid, dtype=torch.bfloat16))
    mod.mlp.experts.down_proj = torch.nn.Parameter(torch.zeros(E, hid, inter, dtype=torch.bfloat16))
    codici = V.committa_strato(mod, tg, td, S)     # = commit di ottimizza_finestra_JOINT (stessa funzione)
    # il commit dai codici, a scale fp32, coincide al bit col vecchio commit da forward_duro (su CPU
    # exp compilato = eager): l'unica differenza voluta e' l'arrotondamento fp16 delle scale
    with torch.no_grad():
        vecchio_gu = V._deruota_se_possibile(tg.forward_duro(16.0), S).to(torch.bfloat16)
        vecchio_dw = td.forward_duro(16.0).to(torch.bfloat16)
    gu32, dw32 = V.pesi_duri_da_codici(codici, S, "cpu", torch.bfloat16, scala_fp16=False)
    verifica("T0: commit dai codici a scale fp32 = vecchio commit da forward_duro, al bit",
             torch.equal(gu32, vecchio_gu) and torch.equal(dw32, vecchio_dw))
    gu16, dw16 = V.pesi_duri_da_codici(codici, S, "cpu", torch.bfloat16, scala_fp16=True)
    verifica(f"T0: il modulo commesso porta le scale fp16 del GGUF (COMMIT_SCALA_FP16={V.COMMIT_SCALA_FP16})",
             torch.equal(mod.mlp.experts.gate_up_proj.data, gu16 if V.COMMIT_SCALA_FP16 else gu32)
             and torch.equal(mod.mlp.experts.down_proj.data, dw16 if V.COMMIT_SCALA_FP16 else dw32))
    esito = FS.verifica_commit(mod, codici, S, 99)
    verifica("T0: codici (gate/up separati, dominio ruotato) + de-rotazione = pesi commessi, identici al bit",
             esito["gate_up"]["diversi"] == 0 and esito["down"]["diversi"] == 0,
             f"gate_up {esito['gate_up']['diversi']}, down {esito['down']['diversi']} elementi diversi")
    for fam, idx in (("ffn_up_exps", (1, 2, 700)), ("ffn_down_exps", (0, 5, 3))):
        rotti = {k: (v[0].copy(), v[1].copy()) for k, v in codici.items()}
        q = rotti[fam][1]
        q[idx] = 1 if q[idx] != 1 else -1
        try:
            FS.verifica_commit(mod, rotti, S, 99)
            verifica(f"T0: un codice sbagliato in {fam} viene SCOPERTO", False)
        except RuntimeError as e:
            verifica(f"T0: un codice sbagliato in {fam} viene SCOPERTO", "NON ridanno" in str(e))
    ob = sostituto_da_codici(codici, None, "cpu")
    ok = True
    try:
        FS._da_codici_verificato(codici, None, "cpu", sostituto_da_codici, "T0")
    except RuntimeError:
        ok = False
    verifica("T0: A2 (sostituto) ridà Q identico e D entro 1e-6 — controllo di _da_codici_verificato", ok)
    D_male = {k: (v[0].copy(), v[1].copy()) for k, v in codici.items()}
    D_male["ffn_gate_exps"][0][0, 0, 0] *= 1.001
    try:
        FS._da_codici_verificato(D_male, None, "cpu", lambda c, cfg, device: ob, "T0")
        verifica("T0: oggetto che NON ridà le scale dei codici RIFIUTATO", False)
    except RuntimeError as e:
        verifica("T0: oggetto che NON ridà le scale dei codici RIFIUTATO", "scale diverse" in str(e))


def t9_quaranta(base: Path):
    print("\n── T9 catena completa a 40 strati giocattolo (39 finestre) ──")
    gioc = Giocattolo(40)
    oss = corsa(base / "t9", gioc, n_passi=2)
    controlli_catena("T9", base / "t9", gioc, oss, 2, "prova")


def t1_pianificazione():
    print("\n── T1 pianificazione ──")
    f = FS.pianifica_scorrevole(40, 2)
    verifica("40 strati, W=2 -> 39 finestre [0,1]..[38,39]",
             len(f) == 39 and f[0] == (0, 2) and f[-1] == (38, 40) and f == pianifica_finestre(40, 2, True))
    commessi = [L for fin in f for L in FS.strati_commessi_da(fin, 40)]
    verifica("ogni strato commesso da esattamente una finestra", commessi == list(range(40)))
    f3 = FS.pianifica_scorrevole(40, 3)
    verifica("W=3 -> 38 finestre, passo 1", len(f3) == 38 and f3[-1] == (37, 40))
    verifica("n == W -> una finestra che commette tutto",
             FS.pianifica_scorrevole(2, 2) == [(0, 2)] and FS.strati_commessi_da((0, 2), 2) == [0, 1])
    for n, w in ((1, 2), (5, 1)):
        try:
            FS.pianifica_scorrevole(n, w)
            verifica(f"n={n} W={w} rifiutato", False)
        except ValueError:
            verifica(f"n={n} W={w} rifiutato", True)


def controlli_catena(nome: str, cartella: Path, gioc: Giocattolo, oss: Osservatore, w: int, corsa_id: str,
                     sostituto: "SostitutoA | None" = None, toll_q: float = 1e-5):
    n = gioc.n
    fin = FS.pianifica_scorrevole(n, w)
    eseguite = [k for (e, k, _) in oss.eventi if e == "finestra_inizio"]
    verifica(f"{nome}: finestre eseguite = pianificate ({len(fin)})", eseguite == [k for k, _ in fin])
    caricati = [L for c in gioc.caricati for L in c]
    verifica(f"{nome}: ogni strato caricato una sola volta (lo strato che resta NON si ricarica)",
             sorted(caricati) == list(range(n)), f"caricamenti {gioc.caricati}")
    verifica(f"{nome}: in memoria solo la finestra corrente",
             all(oss.attivi[k] == list(range(k, k + w)) for k, _ in fin))
    # commit una volta, coi codici dell'ultima finestra
    commit = [(k, L) for (e, k, L) in oss.eventi if e == "commit"]
    verifica(f"{nome}: ogni strato commesso UNA volta", sorted(L for _, L in commit) == list(range(n)))
    disco = codici_su_disco(cartella, n, corsa_id)
    ultima_finestra = {L: min(L, n - w) for L in range(n)}
    verifica(f"{nome}: codici su disco = codici della ULTIMA finestra di ogni strato",
             all(uguali(disco[L], oss.risultati[ultima_finestra[L]][L]) for L in range(n)))
    giornale = [json.loads(r) for r in (cartella / "out" / "giornale_scorrevole.jsonl").read_text().splitlines()]
    verifica(f"{nome}: giornale: un commit per strato, finestra giusta",
             sorted(g["strato"] for g in giornale) == list(range(n))
             and all(g["finestra"][0] == ultima_finestra[g["strato"]] for g in giornale))
    # strato portato: entra coi codici appena ottimizzati
    ok_port = True
    for k, _ in fin[1:]:
        for L in range(k, k + w - 1):
            ini = oss.init_portati[k].get(L)
            prec = oss.risultati[k - 1][L]
            D_att, Q_att = FS._codici_gate_up_fusi(prec)
            ok_port &= ini is not None and np.array_equal(ini["gu"][1], Q_att) and np.array_equal(ini["gu"][0], D_att)
            ok_port &= np.array_equal(ini["dw"][1], prec["ffn_down_exps"][1])
    verifica(f"{nome}: strato portato inizializzato dai codici della finestra precedente (identici)", ok_port)
    verifica(f"{nome}: strato entrante unico, gli altri portati",
             all(oss.entranti[k] == ([k + w - 1] if k > 0 else list(range(w))) for k, _ in fin))
    if sostituto is not None:
        # entrante = teacher + LS (ricalcolato qui indipendentemente)
        ok_ls, n_entranti = True, 0
        S = torch.from_numpy(V._S1024_NP)
        for (ch, L, fam), (portato, (D, Q)) in sostituto.init_codici.items():
            k = ch - 1                      # chiamata 1 = finestra 0 (corsa da capo)
            atteso_portato = k > 0 and L < k + w - 1
            ok_ls &= portato == atteso_portato
            if portato:
                continue
            n_entranti += 1
            t = gioc.strato(L, False).mlp.experts
            Wt = (V._ruota_se_possibile(t.gate_up_proj.detach().float(), S) if fam == "gu"
                  else t.down_proj.detach().float())
            g_ = (V._group_giocattolo(gioc.cfg.hidden_size, V.GRUPPO_GATEUP_REALE) if fam == "gu"
                  else V._group_giocattolo(gioc.cfg.moe_intermediate_size, V.GRUPPO_DOWN_REALE))
            rif = TernarioDiffBatchFuso(Wt, group=g_)
            V.reinit_ls_(rif, Wt)
            Dr, Qr = V._estrai_famiglia(rif)
            ok_ls &= np.array_equal(Q, Qr) and np.array_equal(D, Dr)
        # ogni strato entra UNA volta (W alla finestra 0, uno per finestra dopo): 2 famiglie x n strati
        verifica(f"{nome}: strato entrante inizializzato da teacher + LS (ricalcolato a parte), portati mai",
                 ok_ls and n_entranti == 2 * n, f"{n_entranti} famiglie inizializzate dal teacher (attese {2 * n})")
    # codici su disco -> pesi del modulo commesso (l'unico modulo corrente creato per lo strato)
    S = torch.from_numpy(V._S1024_NP)
    ok_mod = True
    for L in range(n):
        mod = gioc.quant_creati[L][-1].mlp.experts
        gu, dw = FS.pesi_duri_da_codici(disco[L], S, "cpu", mod.gate_up_proj.dtype)
        ok_mod &= torch.equal(gu, mod.gate_up_proj.data) and torch.equal(dw, mod.down_proj.data)
    verifica(f"{nome}: i codici su disco ridanno ESATTAMENTE i pesi commessi nei moduli", ok_mod)
    # seme della finestra k = forward diretto dei 0..k-1 commessi
    peggio_t = peggio_q = 0.0
    for k, _ in fin[1:]:
        h_d, e_d = seme_diretto(gioc, k, cartella, corsa_id)
        h_c, e_c = oss.semi[k]
        rt, rq = FS._confronta_semi(h_d, e_d, h_c, e_c, {b: b for b in h_d})
        peggio_t, peggio_q = max(peggio_t, rt), max(peggio_q, rq)
    verifica(f"{nome}: seme(k) = forward DIRETTO degli strati 0..k-1 commessi, per ogni k",
             peggio_t < 1e-6 and peggio_q < toll_q, f"peggior scarto teacher {peggio_t:.1e} · quantizzato {peggio_q:.1e}")
    kk = fin[len(fin) // 2][0]
    g2 = Giocattolo(n, gioc.dtype)
    h_r, e_r = FS.ricostruisci_seme(kk, gioc.ids, gioc.cfg, gioc.emb, gioc.seme0(gioc.ids, gioc.emb), g2.carica,
                                    cartella / "out", corsa_id)
    rt, rq = FS._confronta_semi(oss.semi[kk][0], oss.semi[kk][1], h_r, e_r, {b: b for b in h_r})
    verifica(f"{nome}: ricostruisci_seme({kk}) dai codici su disco = seme della catena",
             rt == 0.0 and rq == 0.0, f"teacher {rt:.1e} · quantizzato {rq:.1e}")
    verifica(f"{nome}: punti di ripresa tutti rimossi a fine corsa",
             not any((cartella / "ripresa").glob("*finestra_*")))
    return disco


def t2_catena(base: Path, ottimizza_fn=None, etichetta="sostituto"):
    print(f"\n── T2 catena scorrevole completa (W=2, 10 strati, interfaccia A: {etichetta}) ──")
    gioc = Giocattolo(10)
    sost = SostitutoA() if ottimizza_fn is None else None
    oss = corsa(base / f"t2_{etichetta}", gioc, ottimizza_fn=ottimizza_fn or sost.ottimizza)
    return controlli_catena("T2", base / f"t2_{etichetta}", gioc, oss, 2, "prova", sostituto=sost), gioc


def t3_w3(base: Path):
    print("\n── T3 W=3 (7 strati) ──")
    gioc = Giocattolo(7)
    oss = corsa(base / "t3", gioc, w=3)
    controlli_catena("T3", base / "t3", gioc, oss, 3, "prova")


def t4_ripresa(base: Path, ottimizza_fn=None, etichetta="sostituto"):
    print(f"\n── T4 ripresa (6 strati, scrittura sincrona, interfaccia A: {etichetta}) ──")
    n = 6
    rif_dir = base / f"t4_{etichetta}_rif"
    corsa(rif_dir, Giocattolo(n), asincrona=False, ottimizza_fn=ottimizza_fn)
    rif = codici_su_disco(rif_dir, n, "prova")
    casi = [("prima_transizione", 2), ("dopo_ripresa", 2), ("dopo_commit", 2),
            ("dopo_commit_strato", 4), ("dopo_ripresa", 0)]
    for punto_nome, k_int in casi:
        d = base / f"t4_{etichetta}_{punto_nome}_{k_int}"

        def interrompi(p, k, _pn=punto_nome, _k=k_int):
            if p == _pn and k == _k:
                raise FS.InterruzioneSimulata(f"{p}@{k}")
        try:
            corsa(d, Giocattolo(n), asincrona=False, interrompi=interrompi, ottimizza_fn=ottimizza_fn)
            verifica(f"interruzione {punto_nome}@{k_int} avvenuta", False)
            continue
        except FS.InterruzioneSimulata:
            pass
        c = FS.strati_commessi(d / "out", n, "prova", True)
        gioc = Giocattolo(n)
        oss = corsa(d, gioc, asincrona=False, ottimizza_fn=ottimizza_fn)
        fatte = [k for (e, k, _) in oss.eventi if e == "finestra_inizio"]
        uguali_tutti = all(uguali(rif[L], v) for L, v in codici_su_disco(d, n, "prova").items())
        verifica(f"interruzione {punto_nome}@{k_int}: {c} strati commessi -> ripresa dalla finestra "
                 f"{fatte[0]} -> codici IDENTICI alla corsa senza interruzioni", uguali_tutti,
                 f"finestre rifatte {fatte}, caricati {gioc.caricati}")
    # ricetta cambiata / punto corrotto -> rifiuto
    d = base / f"t4_{etichetta}_rifiuti"

    def interrompi(p, k):
        if p == "prima_transizione" and k == 3:
            raise FS.InterruzioneSimulata("stop")
    try:
        corsa(d, Giocattolo(n), asincrona=False, interrompi=interrompi, ottimizza_fn=ottimizza_fn)
    except FS.InterruzioneSimulata:
        pass
    os.environ["THOR_PROVA_RICETTA_CAMBIATA"] = "1"
    try:
        corsa(d, Giocattolo(n), asincrona=False, ottimizza_fn=ottimizza_fn)
        verifica("ripresa con ricetta cambiata RIFIUTATA", False)
    except RuntimeError as e:
        verifica("ripresa con ricetta cambiata RIFIUTATA", "ricetta diversa" in str(e), str(e)[:120])
    finally:
        del os.environ["THOR_PROVA_RICETTA_CAMBIATA"]
    seme = sorted((d / "ripresa").glob("finestra_*/seme_b000.pt"))[0]
    originale = seme.read_bytes()
    seme.write_bytes(originale[:-8] + b"\x00" * 8)
    try:
        corsa(d, Giocattolo(n), asincrona=False, ottimizza_fn=ottimizza_fn)
        verifica("punto di ripresa corrotto RIFIUTATO", False)
    except RuntimeError as e:
        verifica("punto di ripresa corrotto RIFIUTATO", "corrotto" in str(e), str(e)[:100])
    seme.write_bytes(originale)
    corsa(d, Giocattolo(n), asincrona=False, ottimizza_fn=ottimizza_fn)
    verifica("dopo il ripristino del file la ripresa completa e combacia",
             all(uguali(rif[L], v) for L, v in codici_su_disco(d, n, "prova").items()))
    # marche di un'altra corsa nella stessa cartella -> rifiuto
    try:
        corsa(d, Giocattolo(n), asincrona=False, corsa_id="altra", ottimizza_fn=ottimizza_fn)
        verifica("marche di un'altra corsa RIFIUTATE", False)
    except RuntimeError as e:
        verifica("marche di un'altra corsa RIFIUTATE", "corsa" in str(e), str(e)[:100])


def t5_init_da_codici(base: Path, cartella_prec: Path):
    print("\n── T5 modalita' (b): strato entrante dai codici di una corsa precedente ──")
    n = 10
    gioc = Giocattolo(n)
    d = base / "t5"
    oss = corsa(d, gioc, modo_init="codici", init_dir=str(cartella_prec / "out"), init_corsa="prova",
                corsa_id="prova_b")
    prec = codici_su_disco(cartella_prec, n, "prova")
    ok = True
    for k, _ in FS.pianifica_scorrevole(n, 2):
        for L in oss.entranti[k]:
            ini = oss.init_portati[k][L]
            D_att, Q_att = FS._codici_gate_up_fusi(prec[L])
            ok &= np.array_equal(ini["gu"][1], Q_att) and np.allclose(ini["gu"][0], D_att, rtol=1e-6, atol=0)
            ok &= np.array_equal(ini["dw"][1], prec[L]["ffn_down_exps"][1])
    verifica("T5: ogni strato entrante parte ESATTAMENTE dai codici della corsa precedente", ok)
    controlli_catena("T5", d, gioc, oss, 2, "prova_b")
    try:
        corsa(base / "t5_rifiuto", Giocattolo(n), modo_init="codici", init_dir=str(cartella_prec / "out"),
              init_corsa="un_altra_corsa")
        verifica("T5: corsa dichiarata diversa da quella delle marche RIFIUTATA", False)
    except RuntimeError as e:
        verifica("T5: corsa dichiarata diversa da quella delle marche RIFIUTATA", "corsa attesa" in str(e), str(e)[:90])
    # cartella sorgente incompleta (come v2-ThorT-veloce fermata a meta'): rifiuto PRIMA di ogni finestra
    monca = base / "t5_sorgente_monca"
    shutil.copytree(cartella_prec / "out", monca)
    (monca / f"fatto_forgia_{n-1:02d}.ok").unlink()
    (monca / f"Qd_{n-2:02d}.npy").unlink()
    gioc_m = Giocattolo(n)
    try:
        corsa(base / "t5_monca", gioc_m, modo_init="codici", init_dir=str(monca), init_corsa="prova")
        verifica("T5: sorgente incompleta RIFIUTATA prima di partire", False)
    except RuntimeError as e:
        verifica("T5: sorgente incompleta RIFIUTATA prima di partire",
                 "incompleta" in str(e) and not gioc_m.caricati, str(e)[-110:])


def t6_riuso_teacher(base: Path, codici_rif: dict):
    print("\n── T6 teacher catturato vs ricalcolato ──")
    gioc = Giocattolo(10)
    corsa(base / "t6", gioc, riusa_teacher=False)
    verifica("T6: con il teacher ricalcolato i codici sono IDENTICI a quelli col teacher catturato",
             all(uguali(codici_rif[L], v) for L, v in codici_su_disco(base / "t6", 10, "prova").items()))


def t7_contratto_A(ottimizza_fn, etichetta: str):
    print(f"\n── T7 contratto dell'interfaccia A ({etichetta}) ──")
    gioc = Giocattolo(3)
    cfg, emb, pul, qua = gioc.carica([0, 1, 2])
    ids, n_tr = gioc.batch()
    usc: dict = {}
    ottimizza_fn({L: qua[L] for L in (0, 1)}, ids, cfg, 2, strati_teacher={L: pul[L] for L in (0, 1)}, emb=emb,
                 **{FS.PARAM_INIZIALI: None, FS.PARAM_USCITA: usc}, n_batch_addestramento=n_tr)
    verifica("uscita contiene gli oggetti di tutti gli strati", sorted(usc) == [0, 1])
    portato = usc[1]
    prima = {f: V._estrai_famiglia(o) for f, o in portato.items()}
    salva = {k: os.environ.get(k) for k in ("THOR_LR_Z", "THOR_LR_LOGS", "THOR_LR_DELTA")}
    os.environ.update({k: "0" for k in salva})
    try:
        usc2: dict = {}
        ris = ottimizza_fn({L: qua[L] for L in (1, 2)}, ids, cfg, 2, strati_teacher={L: pul[L] for L in (1, 2)},
                           emb=emb, **{FS.PARAM_INIZIALI: {1: portato}, FS.PARAM_USCITA: usc2},
                           n_batch_addestramento=n_tr)
    finally:
        for k, v in salva.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    verifica("lo strato portato e' LO STESSO oggetto in uscita", usc2[1]["gu"] is portato["gu"]
             and usc2[1]["dw"] is portato["dw"])
    D, Q = FS._codici_gate_up_fusi(ris[1])
    verifica("lr=0: i codici dello strato portato escono IDENTICI a quelli passati (nessuna re-init)",
             np.array_equal(Q, prima["gu"][1]) and np.array_equal(D, prima["gu"][0])
             and np.array_equal(ris[1]["ffn_down_exps"][1], prima["dw"][1]))


def t8_bf16(base: Path):
    print("\n── T8 bf16 (informativo: arrotondamenti ai confini della catena) ──")
    gioc = Giocattolo(8, dtype=torch.bfloat16)
    oss = corsa(base / "t8", gioc)
    controlli_catena("T8-bf16", base / "t8", gioc, oss, 2, "prova", toll_q=5e-2)


def t10_gara_verifica_seme(base: Path):
    """25/9 00:58 (S2 fermata in corsa vera): la verifica del seme LEGGE da disco i codici appena
    commessi; con la scrittura asincrona in ritardo leggeva prima della marca ('nessuna marca
    fatto_forgia_01.ok'). Qui la scrittura e' rallentata apposta: la verifica deve aspettarla."""
    print("\n── T10 verifica del seme con scrittura asincrona LENTA (gara) ──")
    orig = V.scrivi_codici_su_disco

    def lenta(r):
        time.sleep(1.5)
        return orig(r)
    V.scrivi_codici_su_disco = lenta
    nome = "T10: verifica del seme alle finestre 1, 2, 3 con scrittura lenta: nessuna gara"
    try:
        oss = corsa(base / "t10", Giocattolo(5), n_passi=2, asincrona=True, corsa_id="t10", verifica_seme_a=[1, 2, 3])
        verifica(nome, True, f"{len(oss.eventi) if hasattr(oss, 'eventi') else '?'} eventi")
    except RuntimeError as e:
        verifica(nome, False, str(e)[:200])
    finally:
        V.scrivi_codici_su_disco = orig


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cartella", default=None, help="cartella di lavoro (default: temporanea, poi cancellata)")
    ap.add_argument("--senza-quaranta", action="store_true", help="salta T9 (catena a 40 strati, ~2 min)")
    args = ap.parse_args()
    assert not torch.cuda.is_available(), "questo collaudo gira SOLO su CPU (CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES=)"
    torch.manual_seed(0)
    base = Path(args.cartella) if args.cartella else Path(tempfile.mkdtemp(prefix="test_scorrevole_"))
    if base.exists() and args.cartella:
        shutil.rmtree(base)
    base.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    vera = interfaccia_vera_presente()
    print(f"interfaccia A vera in valida_finestra_8_12: {'PRESENTE' if vera else 'ASSENTE (uso il SOSTITUTO del test)'}")
    t0_percorso_ruotato()
    t1_pianificazione()
    disco_t2, _ = t2_catena(base)
    t3_w3(base)
    t4_ripresa(base)
    t5_init_da_codici(base, base / "t2_sostituto")
    t6_riuso_teacher(base, disco_t2)
    t7_contratto_A(SostitutoA().ottimizza, "sostituto")
    t8_bf16(base)
    t10_gara_verifica_seme(base)
    if not args.senza_quaranta:
        t9_quaranta(base)
    if vera:
        t7_contratto_A(V.ottimizza_finestra_JOINT, "VERA")
        t2_catena(base, ottimizza_fn=V.ottimizza_finestra_JOINT, etichetta="vera")
        t4_ripresa(base, ottimizza_fn=V.ottimizza_finestra_JOINT, etichetta="vera")
    else:
        try:
            FS.verifica_interfaccia_A(V.ottimizza_finestra_JOINT, serve_da_codici=False)
        except RuntimeError as e:
            verifica("senza interfaccia A la forgia RIFIUTA di partire", "INTERFACCIA A ASSENTE" in str(e))
    falliti = [n for n, ok in ESITI if not ok]
    print(f"\n{'✅ TUTTI' if not falliti else '⛔ FALLITI ' + str(len(falliti)) + ' su'} {len(ESITI)} controlli "
          f"in {time.time()-t0:.0f}s" + ("" if not falliti else ": " + "; ".join(falliti)))
    if not args.cartella:
        shutil.rmtree(base, ignore_errors=True)
    return 0 if not falliti else 1


if __name__ == "__main__":
    raise SystemExit(main())
