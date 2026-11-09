#!/usr/bin/env python3
"""prova_strumenti_f32 — quale leva porta la KL del dominio «strumenti» di EB3 da 0,65 a 1,04 (28/9/2026,
thor_v2/analisi_strumenti.md §8.1). Sola andata, nessun parametro salvato, nessun file della corsa toccato.

Dall'istantanea di un passo (default @@ROOT:depot@@/e2e_istantanee/EB3/passo_000020; `--ema` = la sua sottocartella ema/,
cioe' theta_EMA con F32 e LoRA correnti) ricostruisce lo studente ESATTO della corsa: GGUF di partenza + registri PV
rigiocati sui byte + theta + i 361 F32 + la LoRA-GA (A, B, A0, B0), stessa ricetta/teste/GDN della corsa (config.json).
Poi valuta la KL col maestro (per riga: KL in avanti, KL inversa, CAKLD, top-1) sulle 10 righe STRUMENTI dell'held-out e
su 2 righe di CONOSCENZA (default: riga 8 edu e riga 44 it_wiki, entrambe nell'held-out rapido), in 6 casi:
  tutto         lo studente del passo, come lo ha misurato la corsa
  f32_5_gguf    ffn_gate_inp, ffn_gate_inp_shexp, ssm_alpha, ssm_beta, ssm_conv1d riportati al GGUF (tutti gli strati)
  router_gguf   solo ffn_gate_inp riportato al GGUF
  ssm_ab_gguf   solo ssm_alpha + ssm_beta riportati al GGUF
  lora_spenta   Delta W della LoRA = 0 (A = A0, B = B0: la forma non fusa della LoRA-GA da' zero esatto)
  theta_p12     theta del passo 12 di EB2 (migliore_000012: prima di CAKLD/F32/LoRA/Sophia), codici e resto del passo
Ogni caso riporta indietro cio' che ha cambiato e lo VERIFICA al bit (theta, master F32, A/B della LoRA) prima del caso
dopo. Controllo di fedelta': «tutto» (theta grezzo) sulle 10 righe strumenti deve ridare la KL strumenti dell'held-out
pieno che la corsa ha scritto per quel passo in registro_componenti.jsonl (entro 1e-3 relativo): se non torna, l'esito
porta «⛔ NON FEDELE» (campo "fedelta") e la tabella lo dice in testa: i numeri dei casi non valgono.

Uscite: <uscita>.json (scritto dopo OGNI caso: un'interruzione lascia i casi fatti) e <uscita>.md (tabella).
Costo atteso [S]: caricamento ~2-3 min; 12 righe x 6 casi = 72 righe da 4.096 token a ~6,5 s (held-out pieno del
passo 20: 128 righe in ~14 min) = ~8 min; totale ~11-13 min. Stima del resto stampata dopo il primo caso.

⛔ Un solo processo GPU. Guardia all'avvio (prima di toccare la GPU): `gpu_occupata.sh` deve dare rc 0 e nessuna unita'
e2e-allena-* / riforgia-* dev'essere attiva; altrimenti esce 3. Lancio (unita' systemd --user: la shell interattiva ha
HSA_OVERRIDE_GFX_VERSION sbagliato; stesse variabili di EB3):
  XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus \\
  systemd-run --user --unit=prova-strumenti-f32 --collect -p Nice=5 -p MemoryMax=20G \\
    --setenv=E2E_GDN=veloce --setenv=E2E_GDN_MODULO=gdn_veloce_corsa_v5 --setenv=E2E_MOTORE=triton2 \\
    --setenv=PYTHONDONTWRITEBYTECODE=1 --working-directory @@ROOT:home@@/claude-pod-lavoro/integrata/thor_v2 \\
    @@ROOT:venv@@/bin/python3 strumenti/prova_strumenti_f32.py
  journalctl --user -u prova-strumenti-f32 -f      (esito: @@ROOT:lavoro@@/e2e/allena/prova_strumenti_f32_p20.{json,md})
Consigliato: dal codice CONGELATO del commit (come EB3: nessun file cambia sotto la prova mentre gira), cioe'
  C=<commit>; D=@@ROOT:depot@@/e2e/codice/prova-strumenti-$C; mkdir -p $D && git -C @@ROOT:home@@/claude-pod-lavoro/integrata \\
    archive $C thor_v2 | tar -x -C $D && cp -p @@ROOT:forge@@/thor_v2/mappa_completa_gguf_hf_perm.npz $D/thor_v2/
  e --working-directory $D/thor_v2 nel comando sopra. Usa solo API gia' presenti al commit a014d42 (quello di EB3).
Prova su CPU (modello minuscolo, ogni caso contro uno studente costruito a parte, al bit): e2e_test_prova_strumenti.py P1.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

QUI = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(QUI))
import e2e_allena as A  # noqa: E402
import e2e_pv_codici as PV  # noqa: E402
import e2e_studente as S  # noqa: E402

CORSA = Path("@@ROOT:depot@@/e2e/allena/EB3")
ISTANTANEA = Path("@@ROOT:depot@@/e2e_istantanee/EB3/passo_000020")
THETA_P12 = Path("@@ROOT:lavoro@@/e2e/allena/EB2/migliore_000012/theta.pt")
USCITA = Path("@@ROOT:lavoro@@/e2e/allena/prova_strumenti_f32")
F32_5 = ("ffn_gate_inp", "ffn_gate_inp_shexp", "ssm_alpha", "ssm_beta", "ssm_conv1d")
CASI = ("tutto", "f32_5_gguf", "router_gguf", "ssm_ab_gguf", "lora_spenta", "theta_p12")
TUTTE_F32 = tuple(S.FAMIGLIE_F32) + tuple(S.F32_GLOBALI)


def log(*a) -> None:
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def guardia_gpu() -> list:
    """Motivi per NON partire (vuota = GPU libera): la guardia unica di thor_v2 e le unita' di allenamento/riforgia."""
    motivi = []
    g = subprocess.run(["bash", str(QUI / "strumenti" / "gpu_occupata.sh")], capture_output=True, text=True)
    if g.returncode != 0:
        motivi.append("gpu_occupata.sh rc %d: %s" % (g.returncode, g.stdout.strip()[:400]))
    env = dict(os.environ, XDG_RUNTIME_DIR="/run/user/1000", DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/1000/bus")
    u = subprocess.run(["systemctl", "--user", "list-units", "--state=active", "--no-legend", "--plain",
                        "e2e-allena-*", "riforgia-*"], capture_output=True, text=True, env=env)
    attive = [r.split()[0] for r in u.stdout.splitlines() if r.strip()]
    if u.returncode != 0:
        motivi.append(f"systemctl --user non risponde (rc {u.returncode}): {u.stderr.strip()[:200]}")
    if attive:
        motivi.append(f"unita' attive: {attive}")
    return motivi


def config_corsa(corsa: Path) -> tuple:
    """(ConfigAllena per la valutazione, config.json) dalla configurazione della corsa: perdita/gamma, teste, LoRA,
    pezzo KL; allena_f32 = TUTTE le famiglie (i master servono per riportarne alcune al GGUF); una riga per valutazione e
    nessun dominio escluso (la «metrica» macro della corsa non entra: si leggono le KL per riga)."""
    cj = json.loads((corsa / "config.json").read_text())
    c = cj["config"]
    ca = A.ConfigAllena(perdita=c["perdita"], gamma=c["gamma"], testa_kl=c.get("testa_kl", "fp32"),
                        logit_maestro=c.get("logit_maestro", "fp32"), allena_f32=S.famiglie_f32(TUTTE_F32),
                        lora_rango=int(c.get("lora_rango", 0)), lora_alpha=c.get("lora_alpha"),
                        lora_famiglie=tuple(c.get("lora_famiglie") or ()), lora_tipi=tuple(c.get("lora_tipi") or ()),
                        lora_init=c.get("lora_init", "zero"), seme=int(c["seme"]), pezzo_kl=int(c["pezzo_kl"]),
                        righe_per_micro_valuta=1, escludi_domini=())   # per riga: la «metrica» macro non serve
    return ca, cj


def riferimento_corsa(corsa: Path, passo: int) -> dict:
    """KL per dominio dell'held-out PIENO del passo (theta grezzo) da registro_componenti.jsonl ({} se assente)."""
    p = corsa / "registro_componenti.jsonl"
    if not p.exists():
        return {}
    for riga in p.read_text().splitlines():
        r = json.loads(riga)
        if r.get("passo") == passo and "heldout" in r:
            return r["heldout"].get("kl_per_dominio", {})
    return {}


# ─────────────────────────────── stato dello studente: fotografie e confronti al bit ───────────────────────────────
def foto_theta(st) -> list:
    return [p.detach().to("cpu", copy=True) for p in st.parametri()]


def imposta_theta(st, valori) -> None:
    with torch.no_grad():
        if isinstance(valori, dict):
            for k, p in zip(st.chiavi(), st.parametri()):
                p.copy_(valori[A._chiave(k)].to(device=p.device, dtype=p.dtype))
        else:
            for v, p in zip(valori, st.parametri()):
                p.copy_(v.to(device=p.device, dtype=p.dtype))


def foto_lora(st) -> dict:
    return st.valori_lora() if getattr(st, "lora", None) else {}


def uguale_stato(st, th: list, f32: dict, lo: dict) -> dict:
    """Theta, master F32 e A/B/A0/B0 della LoRA uguali al bit alle fotografie del passo."""
    t = all(torch.equal(p.detach().cpu(), v) for p, v in zip(st.parametri(), th))
    f = all(torch.equal(st.f32[n].master.detach().cpu(), v) for n, v in f32.items())
    ora = foto_lora(st)
    lo_ok = ora.keys() == lo.keys() and all(torch.equal(ora[n][k], lo[n][k]) for n in lo for k in lo[n])
    return {"theta": t, "f32": f, "lora": lo_ok}


def famiglie(st, fams) -> list:
    return [n for n, t in st.f32.items() if t.famiglia in set(fams)]


# ─────────────────────────────── misura ───────────────────────────────
def valuta_righe(st, car, righe: list, ca, WS, WT) -> dict:
    """Una `A.valuta` per riga (stesso calcolo dell'held-out della corsa): {riga: {dominio, kl, kl_inv, perdita, top1}}."""
    out = {}
    for i in righe:
        v = A.valuta(st, car, [i], ca, WS, WT)
        (d, kl), = v["kl_per_dominio"].items()
        out[i] = {"dominio": d, "kl": kl, "kl_inv": v.get("kl_inv_per_dominio", {}).get(d),
                  "perdita": v.get("perdita_per_dominio", {}).get(d), "top1": v["top1_uguale_per_dominio"][d]}
    return out


def riassunto(per_riga: dict) -> dict:
    per_dom: dict = {}
    for r in per_riga.values():
        per_dom.setdefault(r["dominio"], []).append(r)
    return {d: {k: float(np.mean([x[k] for x in rr])) for k in ("kl", "kl_inv", "perdita", "top1") if rr[0][k] is not None}
            | {"righe": len(rr)} for d, rr in sorted(per_dom.items())}


def tabella(esito: dict) -> str:
    casi = [c for c in CASI if c in esito["casi"]]
    base = esito["casi"].get("tutto", {}).get("per_dominio", {})
    dom = list(base) or sorted({d for c in casi for d in esito["casi"][c]["per_dominio"]})
    t = [f"# prova_strumenti_f32 — {esito.get('istantanea', '?')} ({'theta EMA' if esito.get('ema') else 'theta grezzo'})",
         "",
         f"Fedelta' («tutto» contro l'held-out pieno della corsa, strumenti): "
         f"{esito.get('fedelta', {}).get('esito', 'non verificabile')} "
         f"(misurata {esito.get('fedelta', {}).get('misurata')}, corsa {esito.get('fedelta', {}).get('corsa')})", "",
         "| caso | " + " | ".join(f"KL {d}" for d in dom) + " | Δ strumenti su «tutto» | KL inv strumenti | top-1 strumenti |",
         "|---|" + "---|" * (len(dom) + 3)]
    for c in casi:
        pd = esito["casi"][c]["per_dominio"]
        s, s0 = pd.get("strumenti", {}), base.get("strumenti", {})
        delta = (f"{s['kl'] - s0['kl']:+.4f}" if s and s0 else "—")
        t.append(f"| {c} | " + " | ".join(f"{pd[d]['kl']:.4f}" if d in pd else "—" for d in dom) +
                 f" | {delta} | {s.get('kl_inv', float('nan')):.4f} | {s.get('top1', float('nan')):.4f} |")
    t += ["", "Per riga (KL in avanti):", "", "| riga | dominio | " + " | ".join(casi) + " |", "|---|---|" + "---|" * len(casi)]
    righe = list(esito["casi"][casi[0]]["per_riga"])
    for i in righe:
        d = esito["casi"][casi[0]]["per_riga"][i]["dominio"]
        t.append(f"| {i} | {d} | " + " | ".join(f"{esito['casi'][c]['per_riga'][i]['kl']:.4f}" for c in casi) + " |")
    t += ["", f"Secondi: caricamento {esito.get('secondi_caricamento')}, casi "
          + ", ".join(f"{c} {esito['casi'][c]['secondi']}" for c in casi)]
    return "\n".join(t) + "\n"


def scrivi(esito: dict, uscita: Path) -> None:
    uscita.parent.mkdir(parents=True, exist_ok=True)
    for suf, testo in ((".json", json.dumps(esito, indent=1, default=str)), (".md", tabella(esito) if esito["casi"] else "")):
        tmp = uscita.with_suffix(suf + ".tmp")
        tmp.write_text(testo)
        tmp.rename(uscita.with_suffix(suf))


def carica_studente(cj: dict, ca, ist_dir: Path, ema: bool, device: str, log=log) -> tuple:
    """Lo studente ESATTO dell'istantanea: GGUF di partenza (cj["gguf"], ricetta cj["ricetta"], tutti gli strati), F32 e
    LoRA dell'istantanea, theta (grezzo o ema/theta.pt), registri PV rigiocati sui byte. Ritorna (studente, master F32 nati
    dal GGUF, registri, passo)."""
    dev = torch.device(device)
    meta = json.loads((ist_dir / "meta.json").read_text())
    st = S.Studente(gguf=cj["gguf"], strati=range(S.config_testo().num_hidden_layers), ricetta=cj["ricetta"],
                    motore_nome=os.environ.get("E2E_MOTORE", "triton") if dev.type == "cuda" else "torch",
                    device=device, log=log, allena_f32=ca.allena_f32, lora=A.config_lora(ca))
    gg_f32 = st.valori_f32()                                      # i master nati dal GGUF (valori del file)
    if dev.type == "cuda":
        st.carica_embedding()
    ist = A.carica_istantanea(ist_dir)
    mancano = sorted(set(st.f32) - set(ist.get("f32", {})))
    if mancano:
        raise RuntimeError(f"istantanea senza {len(mancano)} F32 dello studente ({mancano[:3]}...)")
    st.imposta_f32({n: v for n, v in ist["f32"].items() if n in st.f32})
    if ca.lora_rango:
        lm = ist["lora_meta"]
        if (int(lm["rango"]), tuple(lm["famiglie"])) != (ca.lora_rango, tuple(st.lora_cfg.famiglie)) or \
                sorted(ist["lora"]) != sorted(st.lora):
            raise RuntimeError(f"LoRA dell'istantanea ({lm['rango']}, {lm['famiglie']}, {len(ist['lora'])}) diversa dallo "
                               f"studente ({ca.lora_rango}, {st.lora_cfg.famiglie}, {len(st.lora)})")
        st.imposta_lora(ist["lora"])
    imposta_theta(st, torch.load((ist_dir / "ema" if ema else ist_dir) / "theta.pt", map_location="cpu"))
    fams = {(L, n): f for L in st.strati for n, f in st.esperti(L).fam.items()}
    if dev.type == "cpu":
        for f in fams.values():
            PV.famiglia_scrivibile(f)
    regs = PV.Registro.carica_lista(ist_dir / "registri_pv.npz") if (ist_dir / "registri_pv.npz").exists() else []
    PV.rigioca(fams, regs)
    return st, gg_f32, regs, int(meta["passo"])


def esegui(st, gg_f32: dict, car, righe: list, ca, WS, WT, casi: list, theta_p12, esito: dict,
           uscita=None, rif: dict = None, log=log) -> dict:
    """I casi su `righe` (una `A.valuta` per riga). Dopo ogni caso lo stato torna AL BIT a quello dell'istantanea
    (verificato: altrimenti solleva). `theta_p12` = dict theta (chiavi `A._chiave`) o percorso di theta.pt.
    `rif` = KL per dominio dell'held-out pieno della corsa (fedelta' di «tutto»; solo theta grezzo)."""
    th0, f0, lo0 = foto_theta(st), {n: t.master.detach().cpu().clone() for n, t in st.f32.items()}, foto_lora(st)

    def riporta_f32(fs):
        st.imposta_f32({n: gg_f32[n] for n in famiglie(st, fs)})

    def rimetti_f32(fs):
        st.imposta_f32({n: f0[n] for n in famiglie(st, fs)})

    def theta_12():
        return theta_p12 if isinstance(theta_p12, dict) else torch.load(theta_p12, map_location="cpu")

    lora_zero = {n: {"A": v["A0"], "B": v["B0"], "A0": v["A0"], "B0": v["B0"]} for n, v in lo0.items() if "A0" in v}
    azioni = {
        "tutto": (lambda: None, lambda: None),
        "f32_5_gguf": (lambda: riporta_f32(F32_5), lambda: rimetti_f32(F32_5)),
        "router_gguf": (lambda: riporta_f32(("ffn_gate_inp",)), lambda: rimetti_f32(("ffn_gate_inp",))),
        "ssm_ab_gguf": (lambda: riporta_f32(("ssm_alpha", "ssm_beta")), lambda: rimetti_f32(("ssm_alpha", "ssm_beta"))),
        "lora_spenta": (lambda: st.imposta_lora(lora_zero), lambda: st.imposta_lora(lo0)),
        "theta_p12": (lambda: imposta_theta(st, theta_12()), lambda: imposta_theta(st, th0)),
    }
    if "lora_spenta" in casi and (not lo0 or len(lora_zero) != len(lo0)):
        raise RuntimeError("lora_spenta: nessuna LoRA in forma GA (A0/B0) nell'istantanea")
    esito.setdefault("casi", {})
    for c in casi:
        t1 = time.time()
        applica, ripristina = azioni[c]
        applica()
        cambiato = uguale_stato(st, th0, f0, lo0)
        per_riga = valuta_righe(st, car, righe, ca, WS, WT)
        ripristina()
        dopo = uguale_stato(st, th0, f0, lo0)
        if not all(dopo.values()):
            raise RuntimeError(f"caso {c}: lo stato del passo non e' tornato al bit ({dopo}): mi fermo")
        esito["casi"][c] = {"per_riga": per_riga, "per_dominio": riassunto(per_riga), "secondi": round(time.time() - t1, 1),
                            "stato_cambiato": {k: not v for k, v in cambiato.items()}}
        if c == "tutto" and rif and rif.get("strumenti") is not None and "strumenti" in esito["casi"][c]["per_dominio"]:
            m, r = esito["casi"][c]["per_dominio"]["strumenti"]["kl"], rif["strumenti"]
            esito["fedelta"] = {"misurata": m, "corsa": r, "scarto_relativo": abs(m - r) / r,
                                "esito": "FEDELE" if abs(m - r) / r <= 1e-3 else "⛔ NON FEDELE"}
            log(f"fedelta': KL strumenti {m:.5f} contro {r:.5f} della corsa -> {esito['fedelta']['esito']}")
        s = esito["casi"][c]["per_dominio"].get("strumenti", {})
        resto = (len(casi) - len(esito["casi"])) * (time.time() - t1)
        log(f"caso {c}: KL strumenti {s.get('kl', float('nan')):.5f} (inversa {s.get('kl_inv', float('nan')):.5f}), " +
            ", ".join(f"{d} {x['kl']:.4f}" for d, x in esito["casi"][c]["per_dominio"].items() if d != "strumenti") +
            f"; {esito['casi'][c]['secondi']} s, resto stimato {resto / 60:.1f} min")
        if uscita is not None:
            scrivi(esito, uscita)
    return esito


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--istantanea", default=str(ISTANTANEA))
    ap.add_argument("--ema", action="store_true", help="theta della sottocartella ema/ (F32 e LoRA sono gli stessi)")
    ap.add_argument("--corsa", default=str(CORSA), help="cartella della corsa (config.json, registro_componenti.jsonl)")
    ap.add_argument("--theta-p12", default=str(THETA_P12))
    ap.add_argument("--righe-conoscenza", default="8,44", help="righe dell'held-out di conoscenza (default edu 8, it_wiki 44)")
    ap.add_argument("--casi", default=",".join(CASI))
    ap.add_argument("--uscita", default=None, help=f"prefisso delle uscite (default {USCITA}_p<passo>[_ema])")
    a = ap.parse_args(argv)
    casi = [c.strip() for c in a.casi.split(",") if c.strip()]
    ignoti = [c for c in casi if c not in CASI]
    if ignoti:
        raise SystemExit(f"casi sconosciuti {ignoti}: {CASI}")
    motivi = guardia_gpu()
    if motivi:
        log("⛔ GPU NON libera, non parto: " + " · ".join(motivi))
        return 3
    t0 = time.time()
    ist_dir, corsa = Path(a.istantanea), Path(a.corsa)
    ca, cj = config_corsa(corsa)
    if ca.gamma is None:
        raise RuntimeError("perdita della corsa senza gamma deciso: questa prova confronta la CAKLD col gamma della corsa")
    dev = torch.device("cuda")
    gdn = A.scegli_gdn(log)
    A.controlla_gdn_f32(gdn, ca.allena_f32)
    st, gg_f32, regs, passo = carica_studente(cj, ca, ist_dir, a.ema, "cuda")
    uscita = Path(a.uscita) if a.uscita else Path(f"{USCITA}_p{passo}{'_ema' if a.ema else ''}")
    log(f"studente del passo {passo} ({'theta EMA' if a.ema else 'theta grezzo'}): {len(st.f32)} F32, "
        f"{len(getattr(st, 'lora', {}))} adattatori LoRA, {len(regs)} registri PV ({sum(len(r) for r in regs)} salti) "
        f"rigiocati; ricetta {cj['ricetta']}, teste {ca.testa_kl}/{ca.logit_maestro}, GDN {gdn}")
    WS, WT, _ = A.teste(st, dev, ca)
    src = A.SorgenteCache(cj["teacher"])
    car = A.Caricatore(src, dev)
    held = src.righe_previste("heldout")
    strum = [i for i in held if src.dominio("heldout", i) == "strumenti"]
    cono = [int(x) for x in a.righe_conoscenza.split(",") if x.strip()]
    righe = strum + [i for i in cono if i not in strum]
    esito = {"istantanea": str(ist_dir), "ema": a.ema, "passo": passo, "corsa": str(corsa), "righe": righe,
             "domini_righe": {i: src.dominio("heldout", i) for i in righe}, "casi": {},
             "secondi_caricamento": round(time.time() - t0, 1), "config": {k: str(v) for k, v in vars(a).items()}}
    log(f"righe: {len(strum)} strumenti {strum} + conoscenza {cono}; caricamento {esito['secondi_caricamento']} s")
    esegui(st, gg_f32, car, righe, ca, WS, WT, casi, a.theta_p12, esito, uscita=uscita,
           rif=None if a.ema else riferimento_corsa(corsa, passo))
    car.chiudi()
    esito["secondi"] = round(time.time() - t0, 1)
    scrivi(esito, uscita)
    log(f"FINE in {esito['secondi']} s: {uscita}.json, {uscita}.md")
    print(tabella(esito))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
