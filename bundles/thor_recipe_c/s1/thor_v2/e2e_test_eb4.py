#!/usr/bin/env python3
"""Prove su CPU di EB4 (28/9/2026, TECNICHE_E2E_PV.md §41): corsa E2E NUOVA dal GGUF riforgiato, LoRA a init zero,
teacher CONDIVISO coi primi blocchi consumati da un'altra corsa (--righe-base-da). Studente VERO minuscolo (2 strati: GDN
+ attenzione piena, esperti TQ1_0, non-esperti k-quant, F32, LoRA, passo V vero), niente GPU, niente /mnt vivo.

  E1  «EB4 in piccolo». Maestro = lo studente minuscolo del file A (righe e righe mc fatte UNA volta, sottoprocesso di
      I10/I11); studente = un GGUF B DIVERSO (strato 1 «riforgiato»: esperti ffn_up in ordine rovesciato, attn_norm,
      post_attention_norm e router cambiati), strato allenato 1, LoRA init zero (niente LoRA-GA), AdamW, lr F32, righe mc
      permutate, peso risposta + lettere + dominio, V a dosi + ricentro, testa bf16x3, logit del maestro bf16, EMA,
      attribuzione, --non-consumare. Teacher con 4 blocchi da 3 righe e il PRIMO consumato altrove: la sorgente SOLLEVA se
      la corsa legge o ASPETTA una riga consumata (una corsa vera aspetterebbe per sempre). Con --righe-base-da 3:
        - la corsa gira fino in fondo: passi (12 - 3) / (4 - 1) = 3, primo segmento {passo 0, cursore_base 3};
        - nessuna riga base < 3 letta; tutte le 3..11 lette;
        - istantanea del passo 0: theta = 0 esatto, F32 == i valori del FILE B (`valori_f32_gguf`, al bit) e diversi da
          quelli del file A dove B li ha cambiati; LoRA: B = 0 esatto, nessun A0/B0 (init zero), nessun lora_ga_*.json;
        - indipendenza del maestro dallo studente: la sorgente (h di ogni riga e firma) e' identica al bit prima e dopo;
          una corsa con lo studente A (stessa sorgente) produce logit del maestro IDENTICI per ogni h del maestro che le
          due corse hanno in comune (`_logit_maestro` osservato: stessa h -> stessi logit al bit, qualunque studente);
          stato["sorgente"] uguale nelle due corse, stato["studente"] diverso (GGUF B contro A).
      MUTAZIONE: --righe-base-da 0 (il piano di sempre) -> la corsa aspetta le righe consumate -> ROSSA.
  E2  consumo dei blocchi con --righe-base-da e consumo ACCESO (--rotazione senza --non-consumare): i blocchi prima di N
      NON si consumano (la corsa non li ha letti), quelli letti si'.
      MUTAZIONE: `PianoRighe.base_iniziale` -> 0 (il consumo di prima: train[:cursore]) -> blocco 0 consumato -> ROSSA.
  E3  logica pura sui numeri veri di EB4 (4096 righe, 32 a passo, 8 mc, N = 512): primo segmento, righe del passo 0 da 512,
      passi totali 149, base_iniziale 512; segmento di tipo EB3 (mc accese al passo 19): base_iniziale 0 (consumo di
      sempre); N = 0: segmento identico a quello di sempre; N a una ripresa (passo > 0) o N >= righe: rifiutati;
      ConfigAllena(righe_base_da=-1) rifiutata.
      MUTAZIONE: `_segmenti_mc` del codice congelato di EB3 (a014d42, senza --righe-base-da) -> cursore 0 -> ROSSA.
Uso (CPU, nice, niente GPU):
    systemd-run --user --scope -p CPUQuota=200% nice -n 19 env CUDA_VISIBLE_DEVICES= HIP_VISIBLE_DEVICES= \\
        OMP_NUM_THREADS=2 E2E_ALLENA_PROVE=@@ROOT:depot@@/<cartella> @@ROOT:venv@@/bin/python3 e2e_test_eb4.py \\
        [E1 E2 E3] [--mutazioni]
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("TRITON_INTERPRET", "1")
os.environ.setdefault("E2E_ALLENA_PROVE", "@@ROOT:depot@@/e2e_prove_eb4")
import numpy as np  # noqa: E402
import torch  # noqa: E402

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import e2e_allena as A  # noqa: E402
import e2e_pv_codici as PV  # noqa: E402
import e2e_test_integrata as TI  # noqa: E402
import e2e_test_lora as TL  # noqa: E402

torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "2")))
DIR = Path(os.environ["E2E_ALLENA_PROVE"]) / "eb4"
ESITO: dict = {"data": time.strftime("%Y-%m-%d %H:%M:%S"), "prove": {}}
muto = lambda *a, **k: None  # noqa: E731
CODICE_EB3 = Path(os.getenv("E2E_CODICE_EB3", "@@ROOT:depot@@/e2e/codice/EB3-a014d42/thor_v2"))
BLOCCO = 3                                            # righe per blocco del teacher minuscolo (12 righe = 4 blocchi)
N0 = 3                                                # --righe-base-da delle prove: il primo blocco e' consumato


def registra(nome: str, ok: bool, **dati) -> bool:
    ESITO["prove"][nome] = {"ok": bool(ok), **dati}
    print(f"[{nome}] {'✅' if ok else '❌'} {json.dumps(dati, default=str)[:3000]}", flush=True)
    return bool(ok)


class RigaConsumata(RuntimeError):
    pass


class SorgenteConsumata(A.SorgenteMemoria):
    """Teacher CONDIVISO: i blocchi in `altrove` sono stati consumati da un'altra corsa. Come `CacheMaestro`, le loro righe
    non sono disponibili e non tornano (il teacher non rifa' un blocco con .consumato): se la corsa ne LEGGE una o ne
    ASPETTA una (`attendi_righe` -> `aggiorna`) si solleva SUBITO, invece di aspettare per sempre come farebbe la corsa
    vera. `lette` = righe train lette (h)."""

    def __init__(self, righe, blocchi, modo, altrove=()):
        super().__init__(righe, blocchi, modo)
        self.altrove = set(altrove)
        self.lette: set = set()
        # come CacheMaestro: la lunghezza delle righe viene dal PIANO (`_lunghezza_lato`), non dalla riga 0 (consumata)
        self.piano = {"corpus": {lato: {"L": int(np.asarray(v[0][0]).shape[0])} for lato, v in righe.items() if v}}

    def _di_altri(self, lato, i) -> bool:
        return lato == "train" and self.blocco_di(lato, i) in self.altrove

    def disponibile(self, lato, i) -> bool:
        return not self._di_altri(lato, i) and super().disponibile(lato, i)

    def aggiorna(self) -> None:
        raise RigaConsumata("la corsa ASPETTA righe che il teacher non rifara' (blocchi consumati da un'altra corsa)")

    def h(self, lato, i):
        if self._di_altri(lato, i):
            raise RigaConsumata(f"letta la riga consumata {lato}[{i}]")
        if lato == "train":
            self.lette.add(i)
        return super().h(lato, i)

    def ids(self, lato, i):
        if self._di_altri(lato, i):
            raise RigaConsumata(f"letti gli id della riga consumata {lato}[{i}]")
        return super().ids(lato, i)


def _dati() -> dict:
    """Righe del maestro (studente minuscolo del file A con theta*, B*) + righe mc e held-out mc: lo stesso script di
    I10/I11 (sottoprocesso, codice di OGGI), UNA volta."""
    uscita = DIR / "sorgente.pt"
    if not uscita.exists():
        f1 = subprocess.run([sys.executable, "-c", TI._FASE1_I10, str(QUI), "-", "-", str(uscita), "solo_dati"],
                            cwd=str(DIR), env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), capture_output=True, text=True)
        if f1.returncode != 0:
            raise RuntimeError(f"fase 1 (righe del maestro) fallita: {f1.stderr[-2000:]}")
    TL._GGUF = TL.DIR / "sintetico_lora.gguf"
    return torch.load(uscita, map_location="cpu", weights_only=False)


def sorgente(dati: dict, d: Path, altrove=(), consumati_ora=None):
    """Come `e2e_test_integrata._sorgente_mc_i10`, con il teacher base a BLOCCHI (4 da 3 righe) e i blocchi `altrove`
    consumati da un'altra corsa."""
    n = len(dati["r"]["train"])
    blocchi = {"train": [(f"train/blocco_{k:04d}", a, min(a + BLOCCO, n)) for k, a in enumerate(range(0, n, BLOCCO))]}
    base = SorgenteConsumata(dati["r"], blocchi, dati["modo"], altrove)
    d.mkdir(parents=True, exist_ok=True)
    firma = {"teacher_mc": "memoria", "train": None, "heldout": None}
    parti = []
    for chi, (righe, risp) in (("train", dati["mc"]), ("heldout", dati["held"])):
        s = A.SorgenteMemoria({"train": righe}, {})
        p = d / f"risposte_{chi}.pt"
        torch.save(risp, p)
        r = A.leggi_risposte(p, lambda i, s=s: s.ids("train", i), s.righe_previste("train"), 16)
        firma[chi] = {"sha256": s.firma()["sha256"], "risposte_sha256": r.sha256}
        parti.append((s, r))
    return base, A.SorgenteConMC(base, parti[0], (parti[1][0], "train", parti[1][1]), firma)


def gguf_riforgiato() -> Path:
    """Il GGUF B: copia del file A con lo strato 1 «riforgiato» (esperti ffn_up in ordine rovesciato: codici TQ1_0 validi
    e diversi; attn_norm + 0,05; post_attention_norm x 1,1; router x 0,9). Stessa struttura (nomi, tipi, forme)."""
    sys.path.insert(0, "@@ROOT:models@@/build/llamacpp-tq1-k1024/gguf-py")
    from gguf import GGUFReader
    a = TL.gguf_lora() if TL._GGUF is None else TL._GGUF
    b = DIR / "sintetico_riforgiato.gguf"
    shutil.copyfile(a, b)
    r = GGUFReader(str(b))
    t = {x.name: x for x in r.tensors}
    del r
    mm = np.memmap(b, dtype=np.uint8, mode="r+")
    for nome, fare in (("blk.1.ffn_up_exps.weight", "rovescia"), ("blk.1.attn_norm.weight", ("+", 0.05)),
                       ("blk.1.post_attention_norm.weight", ("*", 1.1)), ("blk.1.ffn_gate_inp.weight", ("*", 0.9))):
        x = t[nome]
        o, nb = int(x.data_offset), int(x.n_bytes)
        if fare == "rovescia":
            v = np.array(mm[o:o + nb]).reshape(8, -1)[::-1]
            mm[o:o + nb] = v.reshape(-1)
        else:
            v = np.array(mm[o:o + nb]).view(np.float32)
            v = (v + np.float32(fare[1])) if fare[0] == "+" else (v * np.float32(fare[1]))
            mm[o:o + nb] = v.astype(np.float32).view(np.uint8)
    mm.flush()
    del mm
    return b


_PD = TI._PD


def _ca_eb4(**kw):
    """La configurazione di EB4 in piccolo (TECNICHE §41): la riga integrata di EB3 v3 con LoRA init zero, AdamW, righe mc
    permutate, peso risposta/lettere/dominio, strato allenato 1, V a dosi + ricentro, bf16x3 + logit bf16, --non-consumare,
    --righe-base-da N0."""
    base = dict(lora_init="zero", lora_ga_gamma=16.0, ottimizzatore_theta="adamw", ema_theta=0.9, righe_mc=1, righe_mc_permutate=True,
                peso_token="risposta", beta_risposta=0.1, gamma_lettere=1.0, peso_dominio=_PD, strati_allenati=(1,),
                testa_kl="bf16x3", logit_maestro="bf16", rotazione=True, consuma_blocchi=False, righe_base_da=N0,
                **TI._DOSI)
    base.update(kw)
    return TI._ca_integrata(**base)


def _teste(st, WT):
    import e2e_bf16x3 as X3
    return X3.TestaX3(st.testa(torch.float32)), WT.to(torch.bfloat16)


class Spia:
    """`_logit_maestro` osservato: per ogni chiamata sha256(hT, WT) -> sha256(logit)."""

    def __init__(self):
        self.mappa: dict = {}
        self.funzione = True
        self._orig = A._logit_maestro

    def __enter__(self):
        def spia(hT, WT, dtype):
            out = self._orig(hT, WT, dtype)
            k = hashlib.sha256(hT.detach().contiguous().view(torch.int16).numpy().tobytes() +
                               WT.detach().contiguous().view(torch.int16).numpy().tobytes()).hexdigest()
            v = hashlib.sha256(out.detach().float().contiguous().numpy().tobytes()).hexdigest()
            if self.mappa.setdefault(k, v) != v:
                self.funzione = False
            return out
        A._logit_maestro = spia
        return self

    def __exit__(self, *e):
        A._logit_maestro = self._orig


def _firma_h(src) -> str:
    h = hashlib.sha256()
    for lato in sorted(src.r):
        for ids, hh, dom, _ in src.r[lato]:
            h.update(np.ascontiguousarray(ids).tobytes())
            h.update(hh.contiguous().view(torch.int16).numpy().tobytes())
    return h.hexdigest()


def prova_e1(mutazione: bool = False) -> bool:
    nome = "E1_eb4_in_piccolo" + ("_MUTAZIONE_righe_base_da_0" if mutazione else "")
    t0 = time.time()
    dati = _dati()
    gb = gguf_riforgiato()
    ga = TL._GGUF
    WT = dati["WS"]
    c: dict = {}
    base, src = sorgente(dati, DIR / "e1_mc", altrove=("train/blocco_0000",))
    firma0, h0 = src.firma(), _firma_h(base)
    d = DIR / ("e1_corsa_mut" if mutazione else "e1_corsa")
    TI._pulisci(d)
    ca = _ca_eb4(righe_base_da=0 if mutazione else N0)
    st = TL.studente("A32", TL.cfg_lora(4), allena_f32=TI.FAM, gguf=gb)
    sa = TL.studente("A32", TL.cfg_lora(4), allena_f32=TI.FAM, gguf=ga)
    fb, fa = st.valori_f32(), sa.valori_f32()
    cambiati = [n for n in fb if not torch.equal(fb[n], fa[n])]
    c["f32_del_file_B_diversi_da_A"] = sorted(cambiati) == ["blk.1.attn_norm.weight", "blk.1.ffn_gate_inp.weight",
                                                            "blk.1.post_attention_norm.weight"]
    WS, WTb = _teste(st, WT)
    o_esito = PV.RegioneFiducia.esito

    def tiene_sempre(self, rho, prima, dopo, n, **kw):
        o_esito(self, rho, prima, dopo, n, **kw)
        return True
    PV.RegioneFiducia.esito = tiene_sempre
    reg: list = []
    try:
        with Spia() as spia_b:
            z = A.allena(st, src, d, ca, WS, WTb, log=reg.append)
        c["corsa_finita"] = True
    except RigaConsumata as e:
        z = None
        c["corsa_finita"] = False
        reg.append(f"RigaConsumata: {e}")
    finally:
        PV.RegioneFiducia.esito = o_esito
    if z is not None:
        seg = (z.get("mc") or {}).get("segmenti") or [{}]
        c["passi"] = z["passo"] == (12 - N0) // (4 - 1) == z["passi_totali"]
        c["primo_segmento"] = seg[0].get("passo") == 0 and seg[0].get("cursore_base") == N0
        c["nessuna_riga_prima_di_N_letta"] = bool(base.lette) and min(base.lette) >= N0
        c["righe_da_N_tutte_lette"] = set(range(N0, 12)) <= base.lette
        c["log_flusso_base"] = any(f"flusso base dalla riga {N0}" in m for m in reg)
        i0 = d / "migliore_000000"
        th = torch.load(i0 / "theta.pt", map_location="cpu")
        c["theta_zero_al_passo_0"] = all(float(v.abs().max()) == 0.0 for v in th.values())
        f32 = np.load(i0 / "f32.npz")
        dal_file = st.valori_f32_gguf()
        c["f32_passo_0_dal_file_B"] = sorted(f32.files) == sorted(dal_file) and all(
            np.array_equal(f32[n], np.asarray(dal_file[n], np.float32).reshape(f32[n].shape)) for n in f32.files)
        lo = np.load(i0 / "lora.npz")
        c["lora_zero_esatta"] = (any(k.endswith("|B") for k in lo.files) and
                                 all(not np.any(lo[k]) for k in lo.files if k.endswith("|B")) and
                                 not any(k.endswith(("|A0", "|B0")) for k in lo.files))
        c["niente_lora_ga"] = not list(d.glob("lora_ga_*.json")) and "lora_ga" not in z
        c["sorgente_intatta"] = src.firma() == firma0 and _firma_h(base) == h0
        c["logit_maestro_funzione_di_h"] = spia_b.funzione and len(spia_b.mappa) > 0
        # la STESSA sorgente con lo studente A (file del maestro): logit del maestro identici per le h in comune
        base_a, src_a = sorgente(dati, DIR / "e1_mc_a", altrove=("train/blocco_0000",))
        d_a = DIR / "e1_corsa_studente_a"
        TI._pulisci(d_a)
        WS_a, _ = _teste(sa, WT)
        PV.RegioneFiducia.esito = tiene_sempre
        try:
            with Spia() as spia_a:
                z_a = A.allena(sa, src_a, d_a, ca, WS_a, WTb, ferma_a=1, log=muto)
        finally:
            PV.RegioneFiducia.esito = o_esito
        comuni = set(spia_a.mappa) & set(spia_b.mappa)
        c["logit_maestro_uguali_fra_studenti"] = len(comuni) > 0 and all(spia_a.mappa[k] == spia_b.mappa[k] for k in comuni)
        c["stato_sorgente_uguale_studente_diverso"] = (z_a["sorgente"] == z["sorgente"] and
                                                       z_a["studente"]["gguf"] != z["studente"]["gguf"] and
                                                       z["studente"]["gguf"] == str(gb))
        c["righe_comuni_osservate"] = len(comuni)
    ok = all(v for v in c.values() if isinstance(v, bool))
    return registra(nome, ok, **c, registro=[m for m in reg if "flusso base" in m or "RigaConsumata" in m][:3],
                    secondi=round(time.time() - t0, 1))


def prova_e2(mutazione: bool = False) -> bool:
    nome = "E2_consumo_da_N" + ("_MUTAZIONE_base_iniziale_0" if mutazione else "")
    t0 = time.time()
    dati = _dati()
    base, src = sorgente(dati, DIR / "e2_mc")                  # niente consumato altrove: le righe 0-2 CI SONO
    d = DIR / ("e2_corsa_mut" if mutazione else "e2_corsa")
    TI._pulisci(d)
    ca = _ca_eb4(consuma_blocchi=True, attribuzione=False, pv_sonda_dosi=False, pv_ricentra=False, pv_dosi_frazioni=())
    st = TL.studente("A32", TL.cfg_lora(4), allena_f32=TI.FAM)
    WS, WTb = _teste(st, dati["WS"])
    orig = A.PianoRighe.base_iniziale
    if mutazione:
        A.PianoRighe.base_iniziale = lambda self: 0
    try:
        z = A.allena(st, src, d, ca, WS, WTb, log=muto)
    finally:
        A.PianoRighe.base_iniziale = orig
    cons = list(base.consumati)
    c = {"blocco_prima_di_N_intatto": "train/blocco_0000" not in cons,
         "blocchi_letti_consumati": set(cons) == {"train/blocco_0001", "train/blocco_0002", "train/blocco_0003"},
         "righe_prima_di_N_non_lette": bool(base.lette) and min(base.lette) >= N0,
         "stato_consumati_uguale": sorted(z["consumati"]) == sorted(set(cons))}
    return registra(nome, all(c.values()), **c, consumati=cons, secondi=round(time.time() - t0, 1))


def _segmenti_vecchi():
    """`_segmenti_mc` del codice congelato di EB3 (senza --righe-base-da): modulo caricato a parte."""
    import importlib.util
    sp = importlib.util.spec_from_file_location("e2e_allena_eb3_vecchio", CODICE_EB3 / "e2e_allena.py")
    m = importlib.util.module_from_spec(sp)
    sys.modules[sp.name] = m                     # @dataclass cerca il modulo in sys.modules
    try:
        sp.loader.exec_module(m)
    finally:
        sys.modules.pop(sp.name, None)
    return m._segmenti_mc, m.PianoRighe


def prova_e3(mutazione: bool = False) -> bool:
    nome = "E3_logica_numeri_di_EB4" + ("_MUTAZIONE_segmenti_di_EB3" if mutazione else "")
    t0 = time.time()
    seg_f, Piano = (_segmenti_vecchi() if mutazione else (A._segmenti_mc, A.PianoRighe))
    train = list(range(4096))
    mc = [A.OFFSET_MC + j for j in range(768)]

    class SrcMC:
        def firma_mc(self):
            return {"teacher_mc": "memoria"}

        def righe_mc(self):
            return mc
    ca = A.ConfigAllena(righe_mc=8, righe_mc_permutate=True, righe_base_da=512)
    c: dict = {}
    stato = {"passo": 0}
    seg = seg_f(stato, ca, SrcMC(), train, muto)
    p = Piano(train, ca, mc, seg)
    r0 = p.righe(0)[0]
    base0 = [i for i in r0 if i < A.OFFSET_MC]
    c["primo_segmento_512"] = seg[0]["passo"] == 0 and seg[0]["cursore_base"] == 512 and seg[0]["righe_mc"] == 8
    c["passo_0_da_512"] = base0 == list(range(512, 536)) and len(r0) == 32
    c["passi_totali_149"] = p.totale() == (4096 - 512) // 24 == 149
    c["base_iniziale_512"] = getattr(p, "base_iniziale", lambda: None)() == 512
    c["base_consumate_dopo_2_passi"] = p.base_consumate(2) == 512 + 48
    if not mutazione:
        st_eb3 = {"passo": 19, "mc": {"firme": [{}], "segmenti": [{"passo": 19, "cursore_base": 608, "cursore_mc": 0,
                                                                    "righe_mc": 8, "firma": 0, "permutate": True}]}}
        p3 = A.PianoRighe(train, A.ConfigAllena(righe_mc=8, righe_mc_permutate=True), mc, st_eb3["mc"]["segmenti"])
        c["tipo_EB3_base_iniziale_0"] = p3.base_iniziale() == 0
        s0 = A._segmenti_mc({"passo": 0}, A.ConfigAllena(righe_mc=8, righe_mc_permutate=True), SrcMC(), train, muto)
        c["N0_zero_come_sempre"] = s0 == [{"passo": 0, "cursore_base": 0, "cursore_mc": 0, "righe_mc": 8, "firma": 0,
                                           "permutate": True}]
        c["senza_mc_e_N0_zero_classico"] = A._segmenti_mc({"passo": 0}, A.ConfigAllena(), None, train, muto) is None

        def solleva(f) -> bool:
            try:
                f()
                return False
            except (ValueError, RuntimeError):
                return True
        c["N_alla_ripresa_rifiutato"] = solleva(lambda: A._segmenti_mc({"passo": 5}, ca, SrcMC(), train, muto))
        c["N_oltre_le_righe_rifiutato"] = solleva(lambda: A._segmenti_mc(
            {"passo": 0}, A.ConfigAllena(righe_mc=8, righe_base_da=4096), SrcMC(), train, muto))
        c["N_negativo_rifiutato"] = solleva(lambda: A.ConfigAllena(righe_base_da=-1))
        c["N_classico_senza_mc"] = (lambda s: s is not None and s[0]["cursore_base"] == 512 and s[0]["righe_mc"] == 0)(
            A._segmenti_mc({"passo": 0}, A.ConfigAllena(righe_base_da=512), None, train, muto))
    return registra(nome, all(c.values()), **c, secondi=round(time.time() - t0, 1))


PROVE = {"E1": prova_e1, "E2": prova_e2, "E3": prova_e3}


def main() -> int:
    DIR.mkdir(parents=True, exist_ok=True)
    arg = [a for a in sys.argv[1:] if not a.startswith("--")]
    mut = "--mutazioni" in sys.argv
    scelte = [a.upper() for a in arg] or list(PROVE)
    ok, t0 = True, time.time()
    for k in scelte:
        try:
            ok &= PROVE[k]()
        except Exception as e:
            ok = False
            registra(f"{k}_eccezione", False, errore=repr(e), traccia=traceback.format_exc()[-3000:])
    rosse = {}
    if mut:                                   # ogni prova con UN difetto iniettato deve diventare ROSSA
        for k in scelte:
            try:
                verde = PROVE[k](mutazione=True)
            except Exception as e:           # un'eccezione NON e' la rossa giusta: la mutazione non e' provata
                verde = True
                registra(f"{k}_mutazione_eccezione", False, errore=repr(e), traccia=traceback.format_exc()[-1500:])
            rosse[k] = not verde
        ESITO["mutazioni_rosse"] = rosse
        ok &= all(rosse.values())
    ESITO["tutto_verde"] = ok
    ESITO["secondi"] = round(time.time() - t0, 1)
    (DIR / "esito_test_eb4.json").write_text(json.dumps(ESITO, indent=1, default=str))
    print(f"ESITO: {'TUTTO VERDE' if ok else 'ROSSO'}" + (f" (mutazioni rosse: {rosse})" if mut else "") +
          f" ({ESITO['secondi']} s)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
