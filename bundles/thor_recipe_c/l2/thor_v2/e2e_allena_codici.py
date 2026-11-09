#!/usr/bin/env python3
"""Lanciatore fase 2 (punto 3 del compito del 26/9 sera): allena i CODICI ternari (STE, `e2e_codici_motore.py`)
di uno o piu' strati, SOPRA la fase 1 gia' verde (scale, `e2e_allena.py`) -- NON un sostituto.

26/9/2026 sera. **Nessuna riga di `e2e_studente.py` ne' di `e2e_allena.py` cambia** (entrambi importati in
sola lettura, `S`/`AL`): l'aggancio ai codici e' un monkeypatch MIRATO, per istanza, del `forward` di
`MoEStudente` (`aggancia_codici`, sotto) -- non una modifica di classe. Conseguenza VERIFICATA (`--test`,
smoke test sul `Studente` VERO): con `strati_codici=[]` il comportamento e' IDENTICO AL BIT a
`e2e_allena.py`/`e2e_studente.py` di oggi, perche' letteralmente nessun metodo viene sostituito quando la
lista e' vuota.

⛔ NON lanciare su GPU oggi (26/9): la GPU e' del teacher `e2e-teacher`, ~3h residue quando questo file
nasce. Pronto quando serve, con UN comando:

  export XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus
  python3 e2e_allena_codici.py g1-codici --corsa <nome> --teacher <dir_cache_teacher> \\
      --strati-codici 0,3 --clip-codici 1.5 --device cuda
  python3 e2e_allena_codici.py allena --corsa <nome> --teacher <dir_cache_teacher> \\
      --strati-codici 0,3 --clip-codici 1.5 --lr-codici 0.5 --passi 50 --device cuda

Cosa fa `allena` (v0, DICHIARATAMENTE minimale -- NON e' `e2e_allena.allena()`: niente epoche/checkpoint/
ripresa/arresto automatico. E' un ciclo DIAGNOSTICO: bastano poche decine di passi per vedere se la KL
held-out si muove quando anche i codici, non solo le scale, sono liberi di cambiare -- la domanda di oggi di
Angelo. Il ciclo completo e' lavoro futuro, DOPO che questo dice se vale la pena, stesso ordine con cui la
fase 1 ha usato M0/F0a/F0b prima di scrivere `allena()`):

  1. `Studente(motore_nome="torch", ...)` -- portabile CPU/GPU (`torch.matmul`, non serve Triton: la
     versione fusa resta un passo successivo, §7 del progetto di fase 2 -- qui la correttezza viene prima
     della velocita', stesso ordine gia' seguito dalla fase 1).
  2. `aggancia_codici(st, strati_codici, clip_codici)`: per ogni strato in scope, un latente w (E,R,C) per
     famiglia, inizializzato AL TRIT CORRENTE (`e2e_codici_motore.w_da_famiglia`), e il `forward` di quel
     `MoEStudente` sostituito (SOLO per quell'istanza) con `_forward_con_codici` (sotto): stessa logica di
     `S.MoEStudente.forward`, i tre `S.FamigliaEsperti.apply` diventano
     `motore.FamigliaEspertiConCodici.apply` SOLO per le famiglie con un `w` assegnato.
  3. Ottimizzatori SEPARATI (§3, §6 del progetto -- Adam con le beta di theta e' instabile sui codici,
     misurato oggi): theta con `AL.nuovo_ottimizzatore` (INVARIATO, fase 1), w con `torch.optim.SGD`.
  4. `AL.gradiente_passo` (INVARIATO) fa gia' tutto il lavoro: il backward attraverso il forward patchato
     popola `w.grad` esattamente come popola `theta.grad`, senza NESSUNA modifica a quella funzione --
     l'autograd non sa (ne' deve sapere) che il grafo e' piu' grande.

Uso (CPU, solo per il `--test` di oggi):
    systemd-run --user --scope -p CPUQuota=800% -p MemoryMax=10G nice -n 10 \\
      env OMP_NUM_THREADS=4 python3 e2e_allena_codici.py --test
"""
from __future__ import annotations

import argparse
import math
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))

import e2e_allena as AL  # noqa: E402  (sola lettura)
import e2e_studente as S  # noqa: E402  (sola lettura)
import e2e_codici_motore as CM  # noqa: E402  (FamigliaEspertiConCodici, w_da_famiglia)


# ═══════════════════════════ configurazione (separata da ConfigAllena: e2e_allena.py non cambia) ═══════════════════════════
@dataclass(frozen=True)
class ConfigCodici:
    strati_codici: tuple = ()          # strati in cui i CODICI (non solo le scale) sono allenabili
    clip_codici: float = 1.5           # §2.2 del progetto: NON 1.0 (blocca al primo passo i codici gia' al bordo)
    lr_codici: float = 0.5             # SGD (misurato oggi: Adam con le beta di theta e' instabile sui codici)
    passi: int = 50                    # v0: numero fisso di passi, niente arresto automatico
    heldout_ogni: int = 10
    righe_heldout: int = 8


# ═══════════════════════════ 1. aggancio: monkeypatch mirato, per istanza ═══════════════════════════
def _forward_con_codici(self, hidden_states):
    """COPIA di `S.MoEStudente.forward` (letta oggi, non modificata li'): le UNICHE tre righe cambiate sono
    le chiamate a `FamigliaEsperti.apply` per le famiglie con un `w` assegnato (`self.esperti.w`), che
    diventano `CM.FamigliaEspertiConCodici.apply`. Tutto il resto -- router, esperto condiviso, Piano,
    CombinaEsperti/_Disperdi -- e' IDENTICO: se cambia la' (un domani), questa funzione va risincronizzata
    a mano (nessun test automatico lo garantisce oggi -- limite dichiarato, non nascosto)."""
    B, T, H = hidden_states.shape
    x = hidden_states.reshape(-1, H)
    st = self.esperti
    ric = st.ric
    w = getattr(st, "w", None) or {}
    clip = getattr(st, "clip_codici", 1.5)
    if self.router_w32 is not None and x.dtype not in (torch.float32, torch.float64):
        logit = torch.nn.functional.linear(x.float(), self.router_w32)
    else:
        logit = torch.nn.functional.linear(x, self.router_w)
    prob = torch.softmax(logit, dim=-1, dtype=torch.float32 if x.dtype != torch.float64 else torch.float64)
    with torch.no_grad():
        vivo = torch.topk(prob, self.top_k, dim=-1).indices
    if self.instr is None:
        idx = vivo
    else:
        idx = self.instr
        with torch.no_grad():
            diversi = (torch.sort(vivo, -1).values != torch.sort(idx, -1).values).any(-1).sum()
            self.cambi = diversi if self.cambi is None else self.cambi + diversi
    self.ultimo_idx = idx.detach()
    wr = prob.gather(-1, idx)
    wr = wr / wr.sum(-1, keepdim=True)
    condiviso = torch.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)

    piano = S.Piano(idx, st.E)
    if ric.ruota:
        xg = S.ruota(x.to(torch.float64 if ric.dtype_gemm == torch.float64 else torch.float32)).to(ric.dtype_gemm)
    else:
        xg = x.to(ric.dtype_gemm)
    x_ord = S._Raccogli.apply(xg, piano.ordine, piano.inversa, self.top_k)
    dg, du, dd = st.scala("gate"), st.scala("up"), st.scala("down")

    def famiglia(nome, xin, dscala):
        if nome in w:
            return CM.FamigliaEspertiConCodici.apply(xin, dscala, w[nome], st.fam[nome], piano, ric, clip)
        return S.FamigliaEsperti.apply(xin, dscala, st.fam[nome], piano, ric, st.mot, st.oss)

    g = famiglia("gate", x_ord, dg)
    u = famiglia("up", x_ord, du)
    a = (torch.nn.functional.silu(g) * u).to(ric.dtype_gemm)
    o = famiglia("down", a, dd)
    wo = wr.reshape(-1)[piano.ordine].to(o.dtype if o.dtype != torch.bfloat16 else torch.float32)
    o = o.to(wo.dtype) * wo.unsqueeze(-1)
    y = S._Disperdi.apply(o, piano.ordine, piano.inversa, self.top_k)
    return (y.to(x.dtype) + condiviso).reshape(B, T, H)


def aggancia_codici(st: "S.Studente", strati_codici, clip_codici: float = 1.5,
                    dtype: torch.dtype = torch.float32) -> list:
    """Per ogni strato in `strati_codici`: un latente w (E,R,C) per famiglia (inizializzato al trit
    CORRENTE, `CM.w_da_famiglia`), il `forward` del SUO `MoEStudente` sostituito -- SOLO per quell'istanza,
    `types.MethodType`, mai la classe -- con `_forward_con_codici`. `strati_codici=[]` -> nessuna chiamata,
    nessun accesso a `st.moduli`: l'interruttore spento e' l'ASSENZA di lavoro, non un ramo controllato a
    runtime. Ritorna la lista piatta dei `w` (per l'ottimizzatore SGD)."""
    w_list = []
    for L in strati_codici:
        mlp = st.moduli[L].mlp
        esp = mlp.esperti
        esp.w = {n: CM.w_da_famiglia(esp.fam[n], dtype).to(st.device).requires_grad_(True) for n in S.FAMIGLIE}
        esp.clip_codici = clip_codici
        mlp.forward = types.MethodType(_forward_con_codici, mlp)
        w_list.extend(esp.w.values())
    return w_list


# ═══════════════════════════ 2. G1 per i codici, sul gradiente VERO (riusa AL.gradiente_passo INVARIATO) ═══════════════════════════
def _azzera_extra(tensori: list) -> None:
    for t in tensori:
        t.grad = None


def _copia_extra(tensori: list) -> list:
    return [t.grad.detach().clone() for t in tensori]


def cancello_g1_codici_vivo(st, car, lato: str, righe: list, ca: "AL.ConfigAllena", WS, WT, w_list: list,
                            K0: float, log) -> dict:
    """G1 per i codici sul gradiente VERO del teacher (non sintetico come `CM.cancello_g1_codici`): riusa
    `AL.gradiente_passo` (INVARIATO) -- il monkeypatch fa si' che il SUO backward popoli anche `w.grad`, non
    solo `theta.grad`. Confronto fp16-vs-fp32 con lo STESSO meccanismo di `AL.cancello_g1`
    (`AL.ricetta_indietro_fp32`/`AL.ricetta_temporanea`, RIUSATI, non riscritti). NON eseguito oggi (serve
    teacher+GPU): sintassi e logica lette a mano, non collaudate contro un teacher vero. Soglie di
    accettazione NON dichiarate qui apposta: `e2e_progetto_fase2_codici.md` §5 dice che l'eps dei codici va
    MISURATO sul passo vero, non assunto uguale a quello delle scale -- una soglia scritta oggi sarebbe un
    numero inventato."""
    if not w_list:
        return {"nota": "nessuno strato in strati_codici: nulla da misurare"}
    K = K0
    esito: dict = {"K0": K0, "giri": []}
    g16 = None
    for giro in range(3):
        AL._azzera(st)
        _azzera_extra(w_list)
        AL.gradiente_passo(st, car, lato, righe, ca, WS, WT, K)
        g16 = _copia_extra(w_list)
        AL._azzera(st)
        _azzera_extra(w_list)
        with AL.ricetta_temporanea(st, AL.ricetta_indietro_fp32(st.ric)):
            AL.gradiente_passo(st, car, lato, righe, ca, WS, WT, K)
        g32 = _copia_extra(w_list)
        vivo = nulli = rotti = 0
        rel = []
        for a, b in zip(g16, g32):
            v = b != 0
            vivo += int(v.sum())
            nulli += int((v & (a == 0)).sum())
            rotti += int((v & (a != 0) & (torch.sign(a) != torch.sign(b))).sum())
            rr = ((a - b).abs() / b.abs())[v]
            if rr.numel():
                rel.append(rr.float())
        rel_t = torch.cat(rel) if rel else torch.zeros(1)
        g_abs = torch.cat([g.abs().reshape(-1) / K for g in g32])
        g_abs = g_abs[g_abs > 0]
        q01 = float(torch.kthvalue(g_abs, max(1, math.ceil(0.01 * g_abs.numel()))).values) if g_abs.numel() else 0.0
        r = AL.regola_k(q01)
        quota_rotti = (nulli + rotti) / max(vivo, 1)
        giro_esito = {"K_usata": K, "vivo": vivo, "nulli_in_fp16": nulli, "segno_sbagliato": rotti,
                     "quota_rotti": quota_rotti, "q01": q01,
                     "err_rel_mediano": float(rel_t.median()) if rel_t.numel() else 0.0, **r}
        esito["giri"].append(giro_esito)
        log(f"G1-codici giro {giro}: K usata 2^{math.log2(K):.0f} -> quota rotti/nulli {quota_rotti:.2e}, "
            f"K proposta 2^{math.log2(r['K']):.0f}")
        if r["K"] == K:
            break
        K = r["K"]
    for w, g in zip(w_list, g16):
        w.grad = g
    esito.update(esito["giri"][-1])
    return esito


# ═══════════════════════════ 3. ciclo diagnostico v0 (NON e' AL.allena(): niente checkpoint/ripresa/arresto) ═══════════════════════════
def allena_diagnostico(st, src, ca: "AL.ConfigAllena", cc: ConfigCodici, w_list: list, log) -> dict:
    car = AL.Caricatore(src, st.device)
    if st.device.type == "cuda":
        st.carica_embedding()
    WS = st.testa(torch.float32)
    WT, sha_t = AL.testa_maestro(st.device)
    log(f"teste: studente {tuple(WS.shape)}, teacher sha256 {sha_t[:16]} · codici su strati "
        f"{cc.strati_codici} ({sum(w.numel() for w in w_list):,} parametri, clip {cc.clip_codici})")

    righe_g1 = src.righe_previste("train")[:ca.righe_per_passo]
    AL.attendi_righe(src, "train", righe_g1, log)
    g1 = AL.cancello_g1(st, car, "train", righe_g1, ca, WS, WT, log)     # INVARIATO: gate di fase 1 su theta
    K, eps = g1["K"], g1["eps_adam"]
    g1c = cancello_g1_codici_vivo(st, car, "train", righe_g1, ca, WS, WT, w_list, K, log) if w_list else {}

    opt_theta = AL.nuovo_ottimizzatore(st, ca, eps)
    opt_w = torch.optim.SGD(w_list, lr=cc.lr_codici) if w_list else None

    righe_tot = src.righe_previste("train")
    righe_held = src.righe_previste("heldout")[:cc.righe_heldout]
    ordine = AL.ordine_epoca(righe_tot, 0, ca.seme)
    storia_train, storia_held = [], []
    fatti = 0
    for passo in range(1, cc.passi + 1):
        righe = ordine[(passo - 1) * ca.righe_per_passo: passo * ca.righe_per_passo]
        if len(righe) < ca.righe_per_passo:
            log(f"passo {passo}: righe esaurite ({len(righe)} < {ca.righe_per_passo}), fermo (v0: niente epoca 2)")
            break
        AL.attendi_righe(src, "train", righe, log)
        AL._azzera(st)
        if opt_w:
            opt_w.zero_grad()
        kl = AL.gradiente_passo(st, car, "train", righe, ca, WS, WT, K)
        opt_theta.step()
        if opt_w:
            opt_w.step()
        st.controlla(tollera_trabocco=False)
        fatti = passo
        storia_train.append(float(kl))
        log(f"passo {passo}: KL_treno_somma={float(kl):.4f}")
        if passo % cc.heldout_ogni == 0:
            AL.attendi_righe(src, "heldout", righe_held, log)
            v = AL.valuta(st, car, righe_held, ca, WS, WT)
            storia_held.append(v["metrica"])
            log(f"  held-out passo {passo}: KL {v['kl']:.5f} metrica {v['metrica']:.5f}")
    car.chiudi()
    return {"K": K, "eps_adam": eps, "g1_theta": g1, "g1_codici": g1c, "passi_fatti": fatti,
            "storia_train": storia_train, "storia_heldout": storia_held,
            "heldout_scende": (len(storia_held) >= 2 and storia_held[-1] < storia_held[0]) if len(storia_held) >= 2
            else None}


# ═══════════════════════════ smoke test (CPU, Studente VERO, senza teacher) ═══════════════════════════
def _smoke_test() -> dict:
    """NON e' un test di qualita' (nessuna KL vera, nessun teacher: quello e' `allena_diagnostico`, che
    serve GPU+teacher e oggi non gira). Prova SOLO il CABLAGGIO, sul `Studente` VERO (attenzione/GDN reali,
    non un motore isolato come in `e2e_codici_motore.py`):
    1. `strati_codici=[]` -> `Studente.gradiente` IDENTICO AL BIT a uno `Studente` mai toccato da
       `aggancia_codici` (stesso seme, stessi id, stessa perdita giocattolo).
    2. `strati_codici=[0]` -> lo stesso passo popola `w.grad` (non None, non tutto zero) per lo strato in
       scope, attraverso IL VERO `Studente.avanti`/`indietro` (checkpoint per strato, Piano vero)."""
    muto = lambda *_: None
    torch.manual_seed(55)
    ids = torch.randint(0, 2000, (1, 8))

    def perdita(h):
        return (h.float() ** 2).sum()

    st_off = S.Studente(gguf=S.GGUF_THOR_T, strati=[0], ricetta="B", motore_nome="torch", device="cpu", log=muto)
    val_off = st_off.gradiente(ids, perdita)
    grad_off = [p.grad.clone() for p in st_off.parametri()]

    st_nop = S.Studente(gguf=S.GGUF_THOR_T, strati=[0], ricetta="B", motore_nome="torch", device="cpu", log=muto)
    w_vuoto = aggancia_codici(st_nop, [])
    val_nop = st_nop.gradiente(ids, perdita)
    grad_nop = [p.grad.clone() for p in st_nop.parametri()]
    interruttore_spento_ok = (val_off == val_nop and len(w_vuoto) == 0 and
                              all(torch.equal(a, b) for a, b in zip(grad_off, grad_nop)) and
                              not hasattr(st_nop.esperti(0), "w"))

    st_on = S.Studente(gguf=S.GGUF_THOR_T, strati=[0], ricetta="B", motore_nome="torch", device="cpu", log=muto)
    w_list = aggancia_codici(st_on, [0], clip_codici=1.5)
    val_on = st_on.gradiente(ids, perdita)
    w_popolato = all(w.grad is not None for w in w_list) and any(bool((w.grad != 0).any()) for w in w_list)
    theta_ancora_vivo = any(p.grad is not None and bool((p.grad != 0).any()) for p in st_on.parametri())

    return {"interruttore_spento_identico_al_bit": interruttore_spento_ok,
            "n_w_con_strati_codici_0": len(w_list), "w_grad_popolato": w_popolato,
            "theta_grad_ancora_popolato_insieme_a_w": theta_ancora_vivo,
            "loss_off": val_off, "loss_on_con_w_al_trit": val_on,
            "loss_identica_al_trit": val_off == val_on}


def _test() -> bool:
    r = _smoke_test()
    for k, v in r.items():
        print(f"{k}: {v}")
    ok = (r["interruttore_spento_identico_al_bit"] and r["w_grad_popolato"] and
         r["theta_grad_ancora_popolato_insieme_a_w"] and r["loss_identica_al_trit"])
    print("\nESITO:", "TUTTO VERDE" if ok else "⛔ QUALCOSA NON TORNA")
    return ok


# ═══════════════════════════ riga di comando ═══════════════════════════
def _strati_da_arg(s: str) -> tuple:
    return tuple(int(x) for x in s.split(",") if x != "") if s else ()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--test", action="store_true", help="smoke test CPU (nessun teacher, nessuna GPU)")
    sub = ap.add_subparsers(dest="cmd")
    for nome in ("allena", "g1-codici"):
        p = sub.add_parser(nome)
        p.add_argument("--corsa", required=True)
        p.add_argument("--teacher", required=True)
        p.add_argument("--gguf", default=None)
        p.add_argument("--strati-codici", default="", help="es. 0,3 (vuoto = solo fase 1, nessun aggancio)")
        p.add_argument("--clip-codici", type=float, default=ConfigCodici.clip_codici)
        p.add_argument("--lr-codici", type=float, default=ConfigCodici.lr_codici)
        p.add_argument("--passi", type=int, default=ConfigCodici.passi)
        p.add_argument("--heldout-ogni", type=int, default=ConfigCodici.heldout_ogni)
        p.add_argument("--righe-heldout", type=int, default=ConfigCodici.righe_heldout)
        p.add_argument("--righe-passo", type=int, default=AL.ConfigAllena.righe_per_passo)
        p.add_argument("--righe-micro", type=int, default=AL.ConfigAllena.righe_per_micro)
        p.add_argument("--ricetta", default="B")
        p.add_argument("--device", default="cuda")
        p.add_argument("--attendi-gpu", type=int, default=600)
    a = ap.parse_args(argv)

    if a.test or a.cmd is None:
        return 0 if _test() else 1

    gguf = Path(a.gguf) if a.gguf else S.GGUF_THOR_T
    strati_codici = _strati_da_arg(a.strati_codici)
    dir_c = AL.RADICE / a.corsa
    dir_c.mkdir(parents=True, exist_ok=True)
    log = AL._registro(dir_c)
    if a.device == "cuda":
        AL._guardia_gpu(a.attendi_gpu)
    ca = AL.ConfigAllena(righe_per_passo=a.righe_passo, righe_per_micro=a.righe_micro)
    cc = ConfigCodici(strati_codici=strati_codici, clip_codici=a.clip_codici, lr_codici=a.lr_codici,
                      passi=a.passi, heldout_ogni=a.heldout_ogni, righe_heldout=a.righe_heldout)
    src = AL.SorgenteCache(a.teacher)
    dev = torch.device(a.device)
    st = S.Studente(gguf=gguf, strati=range(40), ricetta=a.ricetta,
                    motore_nome="torch", device=a.device, log=log)      # "torch": vedi docstring del modulo
    w_list = aggancia_codici(st, strati_codici, cc.clip_codici)
    log(f"aggancio codici: strati {strati_codici}, {sum(w.numel() for w in w_list):,} parametri w")

    if a.cmd == "g1-codici":
        car = AL.Caricatore(src, dev)
        righe = src.righe_previste("train")[:ca.righe_per_passo]
        AL.attendi_righe(src, "train", righe, log)
        WS = st.testa(torch.float32)
        WT, sha_t = AL.testa_maestro(dev)
        g1 = AL.cancello_g1(st, car, "train", righe, ca, WS, WT, log)
        g1c = cancello_g1_codici_vivo(st, car, "train", righe, ca, WS, WT, w_list, g1["K"], log)
        car.chiudi()
        AL._scrivi_json(dir_c / "g1_codici.json", {"theta": g1, "codici": g1c})
        print(f"theta: {g1.get('K')}, {g1.get('eps_adam')}")
        print(f"codici: {g1c}")
        return 0

    stato = allena_diagnostico(st, src, ca, cc, w_list, log)
    AL._scrivi_json(dir_c / "esito_codici.json", stato)
    log(f"FINE (v0 diagnostico): {stato['passi_fatti']} passi, held-out scende: {stato['heldout_scende']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
