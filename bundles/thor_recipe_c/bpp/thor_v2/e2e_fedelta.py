#!/usr/bin/env python3
"""e2e_fedelta — i cancelli di fedelta' della fase E2E (progetto `e2e_progetto_scelto.md` §2.6, §9 A8). 26/9/2026.
File NUOVO: nessun file esistente e' modificato; SOLA LETTURA di `e2e_studente`, `e2e_teacher`, `e2e_allena`,
`e2e_dump_logit_leggi` e dello strumento `e2e_dump_logit` (llama.cpp, il motore del verdetto).

## F0a — teacher (§2.6), prima di allenare
Le misure le fa `e2e_teacher` (`f0a-cache`, `f0a-ppl`); qui si applicano i criteri:
  - PPL wikitext del teacher Q8 (llama-perplexity) entro 0,5 sigma dal bf16 7,7772 +- 0,0558 [C];
  - logit ricalcolati dalla CACHE (h bf16 + testa bf16 originale, cioe' quelli della perdita) contro i logit
    DIRETTI del motore: KL media <= 1/20 della KL(teacher || studente) al passo 0;
  - G0b: cache + testa Q8 contro i diretti (lo stesso motore: resta solo l'arrotondamento bf16 di h).

## F0b — studente torch contro llama.cpp Vulkan, sullo STESSO GGUF (§2.6)
Sequenze del held-out (default 2 per dominio), ricette A e B dello studente (`e2e_studente.RICETTE`):
  - per strato: errore relativo per token ||x_torch - x_llama|| / ||x_llama|| sulle uscite `l_out-<il>`
    (mediana, p90, massimo) — criterio: mediana <= 3% su OGNI strato;
  - logit: KL(llama || torch) per token — criterio: media <= 1/10 della KL(teacher || llama);
  - instradamento (diagnostica, A8): quota di token con insieme top-k identico fra torch e llama
    (`ffn_moe_topk-<il>`), per strato.
Si tiene la ricetta che passa con la KL(llama || torch) minore; se nessuna passa si indica il primo strato del
salto (come B9). Lo strumento gira UNA sequenza alla volta (logit f32 di 4096 x 248.320 = 4 GB a sequenza):
dump, confronto, cancellazione.

## F0c — trasferimento, dopo l'export (§2.6)
rho = (KL_llama(prima) - KL_llama(dopo)) / (KL_torch(prima) - KL_torch(dopo)), KL(teacher || .) per token sulle
stesse sequenze: llama sul GGUF sorgente e su quello esportato, torch con theta = 0 e con theta del migliore.
Criterio rho >= 0,8 (rho < 0,5: il guadagno e' un artefatto di torch). In piu', per strato: accordo top-k
torch <-> llama sul GGUF esportato, e (se c'e' il dump del top-k del teacher) studente <-> teacher prima e dopo.

Uso (GPU solo libera: la guardia `strumenti/gpu_occupata.sh` esclude il processo stesso):
    python3 e2e_fedelta.py f0a --teacher @@ROOT:lavoro@@/e2e/teacher/q8-tutte-riga --kl-ts 0.12
    python3 e2e_fedelta.py f0b --teacher ... --gguf @@ROOT:models@@/gguf/tony-tern/Thor-v2-ThorT-scor150.gguf \\
        --ricette B,A [--per-dominio 2] --uscita @@ROOT:lavoro@@/e2e/fedelta/f0b_scor150.json
    python3 e2e_fedelta.py f0c --teacher ... --sorgente SRC.gguf --esportato DST.gguf \\
        --corsa @@ROOT:lavoro@@/e2e/allena/NOME [--ricetta B] --uscita .../f0c.json
Prove (CPU, modello minuscolo per lo strumento): `e2e_test_allena.py` (A11-A12).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
sys.path.insert(0, "@@ROOT:models@@/build/llamacpp-tq1-k1024/gguf-py")

from e2e_dump_logit_leggi import Uscita, bf16_a_f32, leggi_tensore  # noqa: E402

VERSIONE = "e2e_fedelta 1 (26/9/2026)"
BIN = QUI / "e2e_dump_logit"
GUARDIA = QUI / "strumenti" / "gpu_occupata.sh"
LAVORO = Path(os.getenv("E2E_FEDELTA_DIR", "@@ROOT:lavoro@@/e2e/fedelta"))
PPL_BF16, SIGMA_BF16 = 7.7772, 0.05581
SOGLIA_ERR_STRATO = 0.03
FRAZ_KL_F0B = 0.10
FRAZ_KL_F0A = 0.05
RHO_MIN = 0.8


def adesso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _scrivi(p: Path, d) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".parziale")
    with open(tmp, "w") as f:
        json.dump(d, f, indent=1, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)


# ═══════════════════════════════════════ F0a: criteri ═══════════════════════════════════════
def f0a_verdetto(dir_teacher: Path, kl_teacher_studente: float) -> dict:
    """Criteri di §2.6 F0a sulle misure di `e2e_teacher f0a-cache / f0a-ppl` (<corsa>/f0a/{cache,ppl}.json)."""
    dir_teacher = Path(dir_teacher)
    out: dict = {"kl_teacher_studente": kl_teacher_studente, "problemi": []}
    pc, pp = dir_teacher / "f0a" / "cache.json", dir_teacher / "f0a" / "ppl.json"
    if pp.exists():
        p = json.loads(pp.read_text())
        s = p.get("q8_meno_bf16_in_sigma")
        out["ppl"] = {"q8_meno_bf16_in_sigma": s, "identita_strumento_4_decimali": p.get("identita_4_decimali"),
                      "passa": s is not None and abs(s) <= 0.5}
        if not out["ppl"]["passa"]:
            out["problemi"].append(f"PPL del teacher a {s} sigma dal bf16 (> 0,5)")
    else:
        out["problemi"].append(f"{pp} manca: f0a-ppl non eseguito")
    if pc.exists():
        c = json.loads(pc.read_text())
        kl_bf16 = [r["h_cache|testa_bf16"]["kl_diretti_vs_ricalcolati_media"] for r in c["per_riga"]]
        kl_q8 = [r["h_cache|testa_q8"]["kl_diretti_vs_ricalcolati_media"] for r in c["per_riga"]
                 if "h_cache|testa_q8" in r]
        media = float(np.mean(kl_bf16))
        out["cache"] = {"kl_diretti_vs_cache_testa_bf16": media, "per_riga": kl_bf16,
                        "kl_diretti_vs_cache_testa_q8": float(np.mean(kl_q8)) if kl_q8 else None,
                        "soglia": FRAZ_KL_F0A * kl_teacher_studente,
                        "passa": media <= FRAZ_KL_F0A * kl_teacher_studente}
        if not out["cache"]["passa"]:
            out["problemi"].append(f"KL(diretti || cache+testa bf16) {media:.3e} > 1/20 di {kl_teacher_studente:.3e}")
    else:
        out["problemi"].append(f"{pc} manca: f0a-cache non eseguito")
    out["passa"] = not out["problemi"]
    return out


# ═══════════════════════════════════════ sequenze e strumento ═══════════════════════════════════════
def scegli_sequenze(src, lato: str = "heldout", per_dominio: int = 2) -> list:
    """Le prime `per_dominio` righe di ogni dominio, nell'ordine del lato (deterministico)."""
    visti: dict = {}
    scelte = []
    for i in src.righe_previste(lato):
        d = src.dominio(lato, i)
        if visti.get(d, 0) < per_dominio:
            visti[d] = visti.get(d, 0) + 1
            scelte.append(i)
    return scelte


def esegui_strumento(gguf: Path, uscita: Path, ids: np.ndarray, inizi: Optional[np.ndarray], vulkan: bool = True,
                     l_out: bool = True, topk: bool = True, logit: bool = True, ub: int = 1024, thread: int = 4,
                     salva_tipo: str = "f32", attendi_gpu_s: int = 600) -> Uscita:
    """UNA sequenza (ids int32 (L,)) in `e2e_dump_logit logit`. Logit f32 di TUTTE le posizioni; l_out di tutti
    gli strati e top-k se chiesti. Guardia GPU prima (esclude questo processo, che puo' tenere lo studente)."""
    if uscita.exists():
        shutil.rmtree(uscita)
    uscita.mkdir(parents=True)
    L = int(ids.shape[0])
    f_ids = uscita / "ids.i32"
    np.ascontiguousarray(ids, dtype=np.int32).tofile(f_ids)
    cmd = [str(BIN), "logit", "-m", str(gguf), "-o", str(uscita / "dump"), "-t", str(thread), "-c", str(L),
           "-b", str(L), "-ub", str(min(ub, L)), "--lunghezza", str(L), "--uscite", "tutte", "--salva-tipo", salva_tipo]
    if vulkan:
        cmd.append("--vulkan")
    if l_out:
        cmd += ["--l-out", "tutti"]
    if topk:
        cmd.append("--topk")
    if not logit:
        cmd.append("--no-logit")
    if inizi is not None:
        f_iz = uscita / "inizi.i32"
        np.ascontiguousarray(inizi, dtype=np.int32).tofile(f_iz)
        cmd += ["--segmenti", str(f_iz)]
    cmd.append(str(f_ids))
    env = dict(os.environ)
    if vulkan:
        # 26/9 sera: si AGGIUNGE ai PID gia' esclusi dal chiamante (la verifica di consegna gira accanto al training)
        env["GPU_ESCLUDI_PID"] = " ".join([str(os.getpid())] + env.get("GPU_ESCLUDI_PID", "").split())
        r = subprocess.run(["bash", str(GUARDIA), "--attendi", str(attendi_gpu_s)], capture_output=True, text=True,
                           env=env)
        if r.returncode != 0:
            raise RuntimeError(f"GPU occupata: {r.stdout.strip()[:400]}")
    else:
        env.update(CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="", GGML_VK_VISIBLE_DEVICES="")
    t0 = time.time()
    with open(uscita / "strumento.log", "w") as f:
        subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT, env=env)
    u = Uscita(uscita / "dump")
    u.secondi = time.time() - t0
    return u


# ═══════════════════════════════════════ metriche ═══════════════════════════════════════
def _stat(v: torch.Tensor) -> dict:
    v = v.double().cpu()
    s = torch.sort(v).values
    n = s.numel()
    return {"mediana": float(s[(n - 1) // 2]), "p90": float(s[min(n - 1, int(0.9 * n))]), "max": float(s[-1]),
            "media": float(v.mean())}


def errore_per_token(x: torch.Tensor, rif: torch.Tensor) -> torch.Tensor:
    """||x - rif|| / ||rif|| per riga (token), in fp64."""
    x, rif = x.double(), rif.double()
    return (x - rif).norm(dim=-1) / rif.norm(dim=-1).clamp_min(1e-30)


def accordo_topk(a: torch.Tensor, b: torch.Tensor) -> float:
    """Quota di token con lo stesso INSIEME top-k (ordine ignorato)."""
    a = torch.sort(a.to(torch.int64).cpu(), -1).values
    b = torch.sort(b.to(torch.int64).cpu(), -1).values
    return float((a == b).all(-1).double().mean())


def kl_righe(lp_a: torch.Tensor, lp_b: torch.Tensor) -> torch.Tensor:
    """KL(a || b) per riga dai log-softmax (fp64)."""
    return (lp_a.exp() * (lp_a - lp_b)).sum(-1, dtype=torch.float64)


def kl_logit(z_llama: np.ndarray, hS: torch.Tensor, WS: torch.Tensor, hT: Optional[torch.Tensor],
             WT: Optional[torch.Tensor], pezzo: int = 256) -> dict:
    """Per token: KL(llama || torch), e se c'e' il teacher KL(teacher || llama), KL(teacher || torch).
    z_llama memmap f32 (T, V); logit torch = hS @ WS^T (fp32); teacher = `e2e_teacher.logit_maestro`."""
    import e2e_teacher as T
    dev = hS.device
    out = {"llama_torch": [], "teacher_llama": [], "teacher_torch": []}
    for i in range(0, hS.shape[0], pezzo):
        zl = torch.from_numpy(np.array(z_llama[i:i + pezzo], dtype=np.float32)).to(dev)
        ll = torch.log_softmax(zl.double(), -1)
        ls = torch.log_softmax((hS[i:i + pezzo].float() @ WS.T).double(), -1)
        out["llama_torch"].append(kl_righe(ll, ls))
        if hT is not None:
            lt = torch.log_softmax(T.logit_maestro(hT[i:i + pezzo], WT).double(), -1)
            out["teacher_llama"].append(kl_righe(lt, ll))
            out["teacher_torch"].append(kl_righe(lt, ls))
            del lt
        del zl, ll, ls
    return {k: torch.cat(v).cpu() for k, v in out.items() if v}


@torch.no_grad()
def confronta_sequenza(st, u: Uscita, k: int, ids: np.ndarray, inizi: Optional[np.ndarray], WS, hT, WT,
                       pezzo: int = 256) -> dict:
    """Studente torch (tutti i suoi strati) contro l'uscita k dello strumento sullo stesso GGUF."""
    dev = st.device
    x_ids = torch.from_numpy(np.asarray(ids, dtype=np.int64))[None].to(dev)
    iz = None if inizi is None else torch.from_numpy(np.asarray(inizi, dtype=np.int32))[None]
    stato = st.avanti(x_ids, inizi=iz)
    uscite = stato["ingressi"][1:] + [stato["xL"]]
    strati = {}
    for j, L in enumerate(st.strati):
        rif = torch.from_numpy(np.array(u.tensore(k, f"l_out-{L}"), dtype=np.float32)).to(dev)
        xt = uscite[j].reshape(-1, uscite[j].shape[-1])
        e = errore_per_token(xt, rif)
        r = {"errore": _stat(e), "_err": e.float().cpu()}
        try:
            tk = torch.from_numpy(np.array(u.tensore(k, f"ffn_moe_topk-{L}"), dtype=np.int32))
            r["accordo_topk"] = accordo_topk(stato["instr"][j], tk)
        except KeyError:
            pass
        strati[L] = r
        del rif, e
    hS = st.esce(stato["xL"])
    del stato, uscite
    _, zl = u.logit(k)
    kl = kl_logit(zl, hS, WS, hT, WT, pezzo)
    return {"strati": strati, "kl": {n: _stat(v) for n, v in kl.items()}, "_kl_grezze": kl, "token": int(hS.shape[0])}


def verdetto_f0b(per_seq: list) -> dict:
    """Per strato: errore per token su TUTTI i token delle sequenze insieme (mediana = il criterio; p90, max) e,
    per diagnosi, il massimo delle mediane per sequenza; KL medie per token."""
    strati = sorted({L for s in per_seq for L in s["strati"]})
    per_strato = {}
    for L in strati:
        tutti = torch.cat([s["strati"][L]["_err"] for s in per_seq])
        med = [s["strati"][L]["errore"]["mediana"] for s in per_seq]
        acc = [s["strati"][L].get("accordo_topk") for s in per_seq if s["strati"][L].get("accordo_topk") is not None]
        per_strato[L] = {**_stat(tutti), "max_delle_mediane_per_sequenza": float(np.max(med)),
                         "accordo_topk_medio": float(np.mean(acc)) if acc else None}
    kl_lt = torch.cat([s["_kl_grezze"]["llama_torch"] for s in per_seq])
    out = {"per_strato": per_strato, "kl_llama_torch_media": float(kl_lt.mean())}
    if all("teacher_llama" in s["_kl_grezze"] for s in per_seq):
        out["kl_teacher_llama_media"] = float(torch.cat([s["_kl_grezze"]["teacher_llama"] for s in per_seq]).mean())
        out["kl_teacher_torch_media"] = float(torch.cat([s["_kl_grezze"]["teacher_torch"] for s in per_seq]).mean())
    fuori = [L for L in strati if per_strato[L]["mediana"] > SOGLIA_ERR_STRATO]
    out["strati_oltre_soglia"] = fuori
    out["primo_strato_del_salto"] = fuori[0] if fuori else None
    kl_ok = "kl_teacher_llama_media" in out and out["kl_llama_torch_media"] <= FRAZ_KL_F0B * out["kl_teacher_llama_media"]
    out["criteri"] = {"errore_mediano_per_strato_<=3%": not fuori, "kl_llama_torch_<=_1/10_kl_teacher_llama": kl_ok}
    out["passa"] = (not fuori) and kl_ok
    return out


# ═══════════════════════════════════════ F0b ═══════════════════════════════════════
def f0b(gguf: Path, dir_teacher: Path, ricette=("B", "A"), per_dominio: int = 2, device: str = "cuda",
        lavoro: Path = LAVORO / "f0b", log: Callable = print) -> dict:
    import e2e_allena as A
    import e2e_studente as S
    src = A.SorgenteCache(dir_teacher)
    righe = scegli_sequenze(src, "heldout", per_dominio)
    dev = torch.device(device)
    WT, sha_t = A.testa_maestro(dev)
    ris = {"versione": VERSIONE, "gguf": str(gguf), "teacher": str(dir_teacher), "modo": src.modo, "righe": righe,
           "domini": [src.dominio("heldout", i) for i in righe], "ricette": {}, "inizio": adesso()}
    for ric in ricette:
        st = S.Studente(gguf=gguf, strati=range(40), ricetta=ric, motore_nome="triton" if dev.type == "cuda"
                        else "torch", device=device, log=log)
        if dev.type == "cuda":
            st.carica_embedding()
        WS = st.testa(torch.float32)
        per_seq = []
        for i in righe:
            ids, iz = np.asarray(src.ids("heldout", i)), src.inizi("heldout", i)
            u = esegui_strumento(gguf, lavoro / f"seq_{i:04d}", ids, iz)
            hT = src.h("heldout", i).to(dev)
            r = confronta_sequenza(st, u, 0, ids, iz, WS, hT, WT)
            r["riga"], r["secondi_strumento"] = i, round(u.secondi, 1)
            per_seq.append(r)
            shutil.rmtree(lavoro / f"seq_{i:04d}")
            log(f"F0b ricetta {ric} riga {i} ({src.dominio('heldout', i)}): KL(llama||torch) "
                f"{r['kl']['llama_torch']['media']:.3e}, errore mediano peggiore "
                f"{max(v['errore']['mediana'] for v in r['strati'].values()):.4f}")
        v = verdetto_f0b(per_seq)
        v["per_sequenza"] = [{"riga": s["riga"], "token": s["token"], "kl": s["kl"],
                              "secondi_strumento": s["secondi_strumento"],
                              "strati": {L: {k: x for k, x in d.items() if k != "_err"} for L, d in s["strati"].items()}}
                             for s in per_seq]
        ris["ricette"][ric] = v
        del st, WS
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    passano = [r for r, v in ris["ricette"].items() if v["passa"]]
    ris["scelta"] = min(passano, key=lambda r: ris["ricette"][r]["kl_llama_torch_media"]) if passano else None
    ris["passa"] = bool(passano)
    ris["fine"] = adesso()
    return ris


# ═══════════════════════════════════════ F0c ═══════════════════════════════════════
def rho(kl: dict) -> dict:
    """kl: medie per token di KL(teacher || .) per llama/torch, prima/dopo."""
    d_l = kl["llama_prima"] - kl["llama_dopo"]
    d_t = kl["torch_prima"] - kl["torch_dopo"]
    r = d_l / d_t if d_t != 0 else float("nan")
    return {"delta_llama": d_l, "delta_torch": d_t, "rho": r, "passa": bool(d_t > 0 and r >= RHO_MIN),
            "artefatto_di_torch": bool(d_t > 0 and r < 0.5)}


def f0c(sorgente: Path, esportato: Path, dir_teacher: Path, theta: dict, ricetta: str = "B", per_dominio: int = 2,
        device: str = "cuda", topk_teacher: Optional[Path] = None, lavoro: Path = LAVORO / "f0c",
        log: Callable = print) -> dict:
    """ρ sulle sequenze del held-out; accordo top-k torch<->llama sul GGUF esportato e (con `topk_teacher`,
    cartella di un'uscita dello strumento col --topk del teacher sulle stesse righe, una sequenza per riga
    nell'ordine di `righe`) studente<->teacher prima e dopo."""
    import e2e_allena as A
    import e2e_studente as S
    src = A.SorgenteCache(dir_teacher)
    righe = scegli_sequenze(src, "heldout", per_dominio)
    dev = torch.device(device)
    WT, _ = A.testa_maestro(dev)
    st = S.Studente(gguf=sorgente, strati=range(40), ricetta=ricetta, motore_nome="triton" if dev.type == "cuda"
                    else "torch", device=device, log=log)
    if dev.type == "cuda":
        st.carica_embedding()
    WS = st.testa(torch.float32)
    somme = {"llama_prima": 0.0, "llama_dopo": 0.0, "torch_prima": 0.0, "torch_dopo": 0.0}
    n = 0
    strati_acc: dict = {}
    ut = Uscita(topk_teacher) if topk_teacher else None
    per_riga = []
    for kk, i in enumerate(righe):
        ids, iz = np.asarray(src.ids("heldout", i)), src.inizi("heldout", i)
        hT = src.h("heldout", i).to(dev)
        rr = {"riga": i, "dominio": src.dominio("heldout", i)}
        for fase, gguf in (("prima", sorgente), ("dopo", esportato)):
            with torch.no_grad():
                for (L, f), p in zip(st.chiavi(), st.parametri()):
                    p.copy_(theta[f"{L}.{f}"] if fase == "dopo" else torch.zeros_like(p))
            u = esegui_strumento(gguf, lavoro / f"seq_{i:04d}_{fase}", ids, iz, l_out=False, topk=True)
            with torch.no_grad():
                stato = st.avanti(torch.from_numpy(ids.astype(np.int64))[None].to(dev),
                                  inizi=None if iz is None else torch.from_numpy(iz)[None])
                hS = st.esce(stato["xL"])
                _, zl = u.logit(0)
                kl = kl_logit(zl, hS, WS, hT, WT)
            somme[f"llama_{fase}"] += float(kl["teacher_llama"].sum())
            somme[f"torch_{fase}"] += float(kl["teacher_torch"].sum())
            for j, L in enumerate(st.strati):
                tl = torch.from_numpy(np.array(u.tensore(0, f"ffn_moe_topk-{L}"), dtype=np.int32))
                a = strati_acc.setdefault(L, {})
                a.setdefault(f"torch_llama_{fase}", []).append(accordo_topk(stato["instr"][j], tl))
                if ut is not None:
                    tt = torch.from_numpy(np.array(ut.tensore(kk, f"ffn_moe_topk-{L}"), dtype=np.int32))
                    a.setdefault(f"studente_teacher_torch_{fase}", []).append(accordo_topk(stato["instr"][j], tt))
                    a.setdefault(f"studente_teacher_llama_{fase}", []).append(accordo_topk(tl, tt))
            rr[fase] = {"kl_teacher_llama": float(kl["teacher_llama"].mean()),
                        "kl_teacher_torch": float(kl["teacher_torch"].mean()), "secondi_strumento": round(u.secondi, 1)}
            del stato, hS, kl
            shutil.rmtree(lavoro / f"seq_{i:04d}_{fase}")
        n += len(ids)
        per_riga.append(rr)
        log(f"F0c riga {i}: {rr}")
    medie = {k: v / n for k, v in somme.items()}
    ris = {"versione": VERSIONE, "sorgente": str(sorgente), "esportato": str(esportato), "ricetta": ricetta,
           "righe": righe, "token": n, "kl_medie": medie, **rho(medie), "per_riga": per_riga,
           "instradamento_per_strato": {L: {k: float(np.mean(v)) for k, v in a.items()} for L, a in strati_acc.items()},
           "fine": adesso()}
    return ris


# ═══════════════════════════════════════ riga di comando ═══════════════════════════════════════
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("f0a")
    p.add_argument("--teacher", required=True)
    p.add_argument("--kl-ts", type=float, required=True, help="KL(teacher || studente) per token al passo 0")
    p.add_argument("--uscita", default=None)
    for nome in ("f0b", "f0c"):
        p = sub.add_parser(nome)
        p.add_argument("--teacher", required=True)
        p.add_argument("--per-dominio", type=int, default=2)
        p.add_argument("--device", default="cuda")
        p.add_argument("--uscita", required=True)
        if nome == "f0b":
            p.add_argument("--gguf", required=True)
            p.add_argument("--ricette", default="B,A")
        else:
            p.add_argument("--sorgente", required=True)
            p.add_argument("--esportato", required=True)
            p.add_argument("--corsa", required=True)
            p.add_argument("--istantanea", default=None)
            p.add_argument("--ricetta", default="B")
            p.add_argument("--topk-teacher", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "f0a":
        r = f0a_verdetto(Path(a.teacher), a.kl_ts)
        if a.uscita:
            _scrivi(Path(a.uscita), r)
        print(json.dumps(r, indent=1, default=str))
        return 0 if r["passa"] else 1
    if a.cmd == "f0b":
        r = f0b(Path(a.gguf), Path(a.teacher), tuple(a.ricette.split(",")), a.per_dominio, a.device)
    else:
        corsa = Path(a.corsa)
        nome = a.istantanea or json.loads((corsa / "ripresa_ultima.json").read_text())["migliore"]
        theta = torch.load(corsa / nome / "theta.pt", map_location=a.device, weights_only=True)
        r = f0c(Path(a.sorgente), Path(a.esportato), Path(a.teacher), theta, a.ricetta, a.per_dominio, a.device,
                Path(a.topk_teacher) if a.topk_teacher else None)
    _scrivi(Path(a.uscita), r)
    print(json.dumps({k: v for k, v in r.items() if k not in ("per_sequenza",)}, indent=1, default=str)[:6000])
    return 0 if r.get("passa") else 1


if __name__ == "__main__":
    raise SystemExit(main())
