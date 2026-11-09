#!/usr/bin/env python3
"""e2e_attribuzione — di quale componente e' il peggioramento? (27/9/2026 16:15, richiesta di Angelo per EB3). File NUOVO.

## Perche'
EB3 accende insieme sette tecniche (CAKLD, F32, LoRA-GA, Sophia sulle scale, pesi per token, V-Bop senza BH, ...). Se
la discesa si ferma o l'held-out sale, serve sapere QUALE parte del passo ha peggiorato il modello. Il registro di
sempre dice solo il totale.

## Cosa misura (con `--attribuzione`; spento = nessun calcolo, nessun campo, allenamento identico al bit)
Una SONDA FISSA: le stesse `--attribuzione-righe` righe del held-out a ogni passo (equispaziate, mai allenate).
  L0      = perdita della corsa (kl o CAKLD, uniforme) sulla sonda PRIMA del passo, e la KL in avanti a parte
  L_tutto = dopo il passo, con TUTTI gli aggiornamenti (P su theta/F32/LoRA, salti V sui codici)
  per ogni componente i in {theta, f32, lora, codici} (solo quelle che il passo ha davvero cambiato):
    L_senza_i = sonda con la SOLA componente i riportata allo stato d'inizio passo (le altre restano nuove)
    Delta_i   = L_tutto - L_senza_i      > 0: la componente i ha PEGGIORATO la sonda (toglierla la migliora)
    L_solo_i  = sonda sullo stato d'inizio passo con la SOLA componente i applicata   (ogni `attribuzione_solo_ogni`)
    dsolo_i   = L_solo_i - L0            > 0: da sola la componente peggiora
Come si riporta indietro una componente, AL BIT:
  theta    i parametri puntano per la durata della misura ai tensori d'inizio passo (`theta_scambiato`, come la EMA:
           nessuna copia, all'uscita tornano GLI STESSI tensori)
  f32      copy_ nei master (il valore HF con la cache per versione, neg_exp di ssm_a, si ricalcola da solo)
  lora     copy_ in A e B (A0, B0 della LoRA-GA sono fermi)
  codici   il registro dei salti TENUTI di questo passo: `e2e_pv_codici.annulla` (inverso, controlla i byte) e
           `rigioca` per rimetterli; salti annullati dalla sonda V o nessun salto = componente assente (Delta 0)
Alla fine lo stato e' verificato uguale a quello dopo il passo: stessi tensori theta, F32 e LoRA torch.equal alle copie,
impronte dei byte uguali. Se qualcosa non torna solleva (meglio fermare la corsa che allenare un modello diverso).
Nessuno stato dell'OTTIMIZZATORE, del passo V (V-Bop), della EMA o del caricatore cambia: si toccano solo i valori.

## Costo (sonda di R righe x 4096 token, sola andata)
valutazioni per passo = 2 + n + n_solo (n = componenti presenti, 4 in EB3; n_solo = n ogni `attribuzione_solo_ogni`
passi, 0 negli altri). EB2 misurato: held-out rapido 16 righe in 279 s = 4,26 ms/token di andata. R = 2: 34,9 s per
valutazione; 10 valutazioni = 5,8 min per passo, ~12 % di un passo da 2.857 s (passo 16 di EB2), ~10 % di uno da 3.500 s.
Memoria: theta d'inizio passo sulla CPU durante il passo (4 B/gruppo: 0,83 GB su Thor-T), sul dispositivo solo durante la
misura (una copia); F32 e LoRA copie CPU (0,1 GB e 0,03 GB).

Prove: e2e_test_integrata.py (I3: identita' esatte e allenamento identico al bit con e senza attribuzione).
"""
from __future__ import annotations

import math
import time
from contextlib import contextmanager
from typing import Callable, Optional

import torch

COMPONENTI = ("theta", "f32", "lora", "codici")


def righe_sonda(held: list, n: int) -> list:
    """Le n righe fisse della sonda: equispaziate nel held-out (i domini sono a blocchi), stesse a ogni passo."""
    if n <= 0:
        raise ValueError("attribuzione_righe >= 1")
    if n >= len(held):
        return list(held)
    return [held[(i * len(held)) // n] for i in range(n)]


def _sync(dev) -> None:
    if getattr(dev, "type", "cpu") == "cuda":
        torch.cuda.synchronize(dev)


@torch.no_grad()
def perdita_sonda(st, car, righe: list, ca, WS, WT, kl_pezzo: Callable, gamma: Optional[float]) -> dict:
    """Perdita della corsa media per token sulla sonda (lato held-out), SOLA andata: "perdita" (kl o CAKLD, uniforme,
    stesso calcolo della sonda del passo V) e "kl" (in avanti) a parte; con CAKLD anche "kl_inv". Somme in fp64 sul
    dispositivo, una sincronizzazione alla fine."""
    tot = {}
    n = 0
    for j in range(0, len(righe), ca.righe_per_micro):
        lot = car.prendi("heldout", righe[j:j + ca.righe_per_micro])
        hS = st.nascosto(lot.ids, inizi=lot.inizi)
        for i in range(0, hS.shape[0], ca.pezzo_kl):
            o = kl_pezzo(hS[i:i + ca.pezzo_kl], WS, lot.hT[i:i + ca.pezzo_kl], WT, False, gamma=gamma)
            for k in ("kl", "kl_inv", "perdita"):
                if k in o:
                    s = o[k].sum()
                    tot[k] = s if k not in tot else tot[k] + s
        n += hS.shape[0]
        del hS
    out = {k: float(v) / max(n, 1) for k, v in tot.items()}
    if "perdita" not in out:
        out["perdita"] = out["kl"]
    out["token"] = n
    return out


class InizioPasso:
    """Lo stato dei VALORI allenabili all'inizio del passo (copie sulla CPU): theta, master F32, A e B della LoRA."""

    def __init__(self, st, parametri_f32: list, parametri_lora: list):
        self.theta = [p.detach().to("cpu", copy=True) for p in st.parametri()]
        self.f32 = [p.detach().to("cpu", copy=True) for p in parametri_f32]
        self.lora = [p.detach().to("cpu", copy=True) for p in parametri_lora]


class Commutatore:
    """Porta ogni componente allo stato «vecchio» (inizio passo) o «nuovo» (dopo il passo), al bit, e alla fine
    verifica di aver lasciato lo stato nuovo."""

    def __init__(self, st, ini: InizioPasso, parametri_f32: list, parametri_lora: list, pv, reg_passo, impronte: Callable):
        self.st = st
        dev = st.device
        self.p_theta = list(st.parametri())
        self.theta_nuovo = [p.data for p in self.p_theta]
        self.theta_vecchio = [t.to(dev) for t in ini.theta]
        self.p_f32 = list(parametri_f32)
        self.f32_nuovo = [p.detach().to("cpu", copy=True) for p in self.p_f32]
        self.f32_vecchio = ini.f32
        self.p_lora = list(parametri_lora)
        self.lora_nuovo = [p.detach().to("cpu", copy=True) for p in self.p_lora]
        self.lora_vecchio = ini.lora
        self.pv = pv
        self.reg = reg_passo if (reg_passo is not None and len(reg_passo) > 0) else None
        self.impronte = impronte
        self.impronte_nuove = impronte(st) if self.reg is not None else None
        self.stato = {c: "nuovo" for c in COMPONENTI}

    def presenti(self) -> list:
        """Le componenti che il passo ha cambiato davvero (theta c'e' sempre)."""
        c = ["theta"]
        if self.p_f32 and any(not torch.equal(a, b) for a, b in zip(self.f32_nuovo, self.f32_vecchio)):
            c.append("f32")
        if self.p_lora and any(not torch.equal(a, b) for a, b in zip(self.lora_nuovo, self.lora_vecchio)):
            c.append("lora")
        if self.reg is not None:
            c.append("codici")
        return c

    @torch.no_grad()
    def metti(self, comp: str, quale: str) -> None:
        if self.stato[comp] == quale:
            return
        if comp == "theta":
            for p, x in zip(self.p_theta, self.theta_vecchio if quale == "vecchio" else self.theta_nuovo):
                p.data = x
        elif comp == "f32":
            for p, x in zip(self.p_f32, self.f32_vecchio if quale == "vecchio" else self.f32_nuovo):
                p.copy_(x.to(device=p.device, dtype=p.dtype))
        elif comp == "lora":
            for p, x in zip(self.p_lora, self.lora_vecchio if quale == "vecchio" else self.lora_nuovo):
                p.copy_(x.to(device=p.device, dtype=p.dtype))
        elif comp == "codici":
            import e2e_pv_codici as PV
            if quale == "vecchio":
                PV.annulla(self.pv.fam, self.reg)
            else:
                PV.rigioca(self.pv.fam, [self.reg])
        else:
            raise KeyError(comp)
        self.stato[comp] = quale

    def theta_alternativo(self, alt: list, misura: Callable) -> dict:
        """Misura con theta = `alt` (tensori sul dispositivo), tutto il resto nuovo; theta torna quello nuovo."""
        if any(v != "nuovo" for v in self.stato.values()):
            raise RuntimeError("theta alternativo con componenti non nuove")
        try:
            for p, x in zip(self.p_theta, alt):
                p.data = x
            return misura()
        finally:
            for p, x in zip(self.p_theta, self.theta_nuovo):
                p.data = x

    def tutto(self, quale: str) -> None:
        for c in COMPONENTI:
            if c == "codici" and self.reg is None:
                continue
            self.metti(c, quale)

    def verifica_nuovo(self) -> None:
        """Dopo la misura: lo stato del modello e' ESATTAMENTE quello dopo il passo, altrimenti solleva."""
        rotti = []
        if any(p.data_ptr() != x.data_ptr() or p.shape != x.shape for p, x in zip(self.p_theta, self.theta_nuovo)):
            rotti.append("theta")
        if any(not torch.equal(p.detach().cpu(), x) for p, x in zip(self.p_f32, self.f32_nuovo)):
            rotti.append("f32")
        if any(not torch.equal(p.detach().cpu(), x) for p, x in zip(self.p_lora, self.lora_nuovo)):
            rotti.append("lora")
        if self.reg is not None and self.impronte(self.st) != self.impronte_nuove:
            rotti.append("codici")
        if rotti or any(v != "nuovo" for v in self.stato.values()):
            raise RuntimeError(f"attribuzione: stato NON ripristinato al bit ({rotti}, {self.stato}): corsa fermata")


def attribuisci(st, car, righe: list, ca, WS, WT, kl_pezzo: Callable, gamma: Optional[float], ini: InizioPasso,
                L0: dict, parametri_f32: list, parametri_lora: list, pv, reg_passo, impronte: Callable,
                con_solo: bool, theta_adam: Optional[Callable] = None) -> dict:
    """Lascia-uno-fuori (e solo-uno) per componente. Ritorna la voce del registro (numeri gia' sincronizzati).
    `theta_adam(p, theta0) -> tensore | None` (Sophia con l'ombra di AdamW): UNA valutazione in piu' con theta = il passo
    che AdamW avrebbe fatto dallo stesso stato (resto nuovo): "theta_adam" nel registro, delta = L_tutto - L_adam (> 0 =
    il passo di Sophia e' PEGGIO di quello di AdamW sulla sonda)."""
    t0 = time.time()
    com = Commutatore(st, ini, parametri_f32, parametri_lora, pv, reg_passo, impronte)
    pres = com.presenti()

    def misura() -> dict:
        return perdita_sonda(st, car, righe, ca, WS, WT, kl_pezzo, gamma)
    out = {"righe": list(righe), "L0": L0, "componenti": pres, "senza": {}, "solo": {}, "delta": {}, "delta_kl": {},
           "dsolo": {}, "dsolo_kl": {}}
    try:
        tutto = misura()
        out["L_tutto"] = tutto
        for c in pres:
            com.metti(c, "vecchio")
            s = misura()
            com.metti(c, "nuovo")
            out["senza"][c] = s
            out["delta"][c] = tutto["perdita"] - s["perdita"]
            out["delta_kl"][c] = tutto["kl"] - s["kl"]
        if con_solo and len(pres) > 1:
            com.tutto("vecchio")
            for c in pres:
                com.metti(c, "nuovo")
                s = misura()
                com.metti(c, "vecchio")
                out["solo"][c] = s
                out["dsolo"][c] = s["perdita"] - L0["perdita"]
                out["dsolo_kl"][c] = s["kl"] - L0["kl"]
        elif con_solo:                                   # una sola componente: solo = tutto (nessuna valutazione in piu')
            c = pres[0]
            out["solo"][c] = tutto
            out["dsolo"][c] = tutto["perdita"] - L0["perdita"]
            out["dsolo_kl"][c] = tutto["kl"] - L0["kl"]
        com.tutto("nuovo")
        if theta_adam is not None:                       # controfattuale: il passo di AdamW sulle scale
            alt = [theta_adam(p, v) for p, v in zip(com.p_theta, com.theta_vecchio)]
            if all(x is not None for x in alt):
                ta = com.theta_alternativo(alt, misura)
                # 27/9 dominio-strati: `theta_adam` ritorna lo STESSO tensore d'inizio passo per le scale degli strati
                # congelati (AdamW non le avrebbe mosse): fuori dalla mediana del passo
                d_ad = [(x.double() - v.double()).abs().flatten()[::max(1, v.numel() // 4096)]
                        for x, v in zip(alt, com.theta_vecchio) if x is not v]
                del alt
                out["theta_adam"] = {"L": ta, "delta": tutto["perdita"] - ta["perdita"],
                                     "delta_kl": tutto["kl"] - ta["kl"],
                                     "mediana_passo_adam_su_lr": float(torch.cat(d_ad).float().median()) /
                                     float(ca.lr) if d_ad else float("nan")}
    finally:
        com.tutto("nuovo")
    com.verifica_nuovo()
    out["delta_totale"] = out["L_tutto"]["perdita"] - L0["perdita"]
    out["delta_totale_kl"] = out["L_tutto"]["kl"] - L0["kl"]
    pos = {c: d for c, d in out["delta"].items() if d > 0}
    out["peggiore"] = max(pos, key=pos.get) if pos else None
    out["valutazioni"] = 1 + len(pres) + (len(pres) if con_solo and len(pres) > 1 else 0) + ("theta_adam" in out)
    out["secondi"] = round(time.time() - t0, 2)
    return out


def testo_attribuzione(a: dict) -> str:
    s = (f"sonda {len(a['righe'])} righe: L0 {a['L0']['perdita']:.5f} -> tutto {a['L_tutto']['perdita']:.5f} "
         f"({a['delta_totale']:+.5f}); Delta_i = tutto - senza_i: " +
         ", ".join(f"{c} {a['delta'][c]:+.5f}" for c in a["componenti"]))
    if a["dsolo"]:
        s += "; solo_i - L0: " + ", ".join(f"{c} {a['dsolo'][c]:+.5f}" for c in a["componenti"] if c in a["dsolo"])
    if a.get("theta_adam"):
        s += (f"; theta col passo di AdamW {a['theta_adam']['L']['perdita']:.5f} (tutto - adam "
              f"{a['theta_adam']['delta']:+.5f}: > 0 = Sophia peggio)")
    if a["peggiore"]:
        s += f"; PEGGIORA: {a['peggiore']}"
    return s + f" ({a['valutazioni']} valutazioni, {a['secondi']:.1f} s)"


# ─────────────────────────── diagnostica per componente (numeri del passo) ───────────────────────────
@torch.no_grad()
def statistiche_theta(st, ini: InizioPasso, lr: float, campione: int = 1 << 22) -> dict:
    """|Delta theta| del passo in unita' di lr sulle scale VIVE (d0 != 0): quota a passo pieno (>= 0,99 lr: con Sophia
    = quota clippata), mediana e q90 su un campione a passo fisso (<= `campione` valori, deterministico). Stessa misura
    con AdamW e con Sophia: e' il confronto «passo di Sophia contro passo di Adam» (Adam, §36: mediana ~0,22 lr)."""
    allenato = getattr(st, "allenato", lambda L: True)      # 27/9 dominio-strati: solo gli strati allenati
    tot = sum(int(p.numel()) for (L, _), p in zip(st.chiavi(), st.parametri()) if allenato(L))
    passo = max(1, tot // campione)
    vivi = 0
    pieni = 0
    zeri = 0
    parti = []
    for (L, n), p, p0 in zip(st.chiavi(), st.parametri(), ini.theta):
        if not allenato(L):
            continue
        d0 = st.esperti(L).fam[n].d0.reshape(p.shape)
        v = (p.detach().double() - p0.to(p.device).double()).abs() / lr
        m = d0 != 0
        vv = v[m]
        vivi += int(vv.numel())
        pieni += int((vv >= 0.99).sum())
        zeri += int((vv == 0).sum())
        parti.append(vv.flatten()[::passo].float().cpu())
    s = torch.cat(parti) if parti else torch.zeros(0)
    q = torch.quantile(s, torch.tensor([0.5, 0.9])).tolist() if s.numel() else [float("nan")] * 2
    return {"vivi": vivi, "quota_passo_pieno": pieni / max(vivi, 1), "quota_ferme": zeri / max(vivi, 1),
            "mediana_su_lr": q[0], "q90_su_lr": q[1], "campione": int(s.numel())}


@torch.no_grad()
def statistiche_f32(st, ini: InizioPasso) -> dict:
    """|Delta master| del passo per famiglia F32 (norma 2) e relativa alla norma d'inizio passo."""
    per: dict = {}
    allenato = getattr(st, "allenato", lambda L: True)      # 27/9 dominio-strati: solo gli strati allenati
    for (n, tf), p0 in zip(st.f32.items(), ini.f32):
        if not allenato(tf.L):
            continue
        a = tf.master.detach().double().cpu()
        b = p0.double()
        d2, n2 = float(((a - b) ** 2).sum()), float((b ** 2).sum())
        x = per.setdefault(tf.famiglia, [0.0, 0.0])
        x[0] += d2
        x[1] += n2
    return {f: {"norma_passo": math.sqrt(d2), "relativa": math.sqrt(d2 / n2) if n2 > 0 else None}
            for f, (d2, n2) in sorted(per.items())}


def _n2_prodotto(B, A) -> float:
    return float(torch.trace((B.T @ B) @ (A @ A.T)))


def _n2_misto(B, A, B2, A2) -> float:
    return float(torch.trace((B.T @ B2) @ (A2 @ A.T)))


def _n2_diff(B, A, B2, A2) -> float:
    """|B A - B2 A2|_F^2 con sole matrici r x r (fp64)."""
    return _n2_prodotto(B, A) - 2.0 * _n2_misto(B, A, B2, A2) + _n2_prodotto(B2, A2)


@torch.no_grad()
def statistiche_lora(st, ini: InizioPasso) -> dict:
    """Per famiglia: |Delta W| totale (s (B A - B0 A0) con la LoRA-GA, s B A con init zero) e |Delta W| del passo
    (s (B A - B_i A_i), i = inizio passo), norma di Frobenius, fp64 con matrici r x r."""
    per: dict = {}
    vecchi = iter(ini.lora)
    allenato = getattr(st, "allenato", lambda L: True)      # 27/9 dominio-strati: solo gli strati allenati
    for a in st.lora.values():
        A_, B_ = a.A.detach().double().cpu(), a.B.detach().double().cpu()
        Ai, Bi = next(vecchi).double(), next(vecchi).double()
        if not allenato(a.L):
            continue
        s2 = a.scala ** 2
        if a.A0 is not None:
            A0, B0 = a.A0.detach().double().cpu(), a.B0.detach().double().cpu()
            tot = _n2_diff(B_, A_, B0, A0) * s2
        else:
            tot = _n2_prodotto(B_, A_) * s2
        pas = _n2_diff(B_, A_, Bi, Ai) * s2
        x = per.setdefault(a.famiglia, [0.0, 0.0])
        x[0] += max(tot, 0.0)
        x[1] += max(pas, 0.0)
    return {f: {"norma_totale": math.sqrt(t), "norma_passo": math.sqrt(p)} for f, (t, p) in sorted(per.items())}


@contextmanager
def cronometra(st, tempi: dict):
    """Per la durata del blocco `st.avanti` e `st.indietro` sommano in `tempi` i secondi di andata e ritorno degli strati
    (sincronizzati: sulla GPU l'orologio altrimenti misura solo il lancio). Il resto del gradiente (testa KL, caricatore)
    = gradiente - avanti - indietro. Nessun numero cambia."""
    dev = st.device
    av, ind = st.avanti, st.indietro
    propri = {k: st.__dict__[k] for k in ("avanti", "indietro") if k in st.__dict__}   # attributi d'istanza gia' presenti

    def avanti(*a, **k):
        _sync(dev)
        t = time.time()
        r = av(*a, **k)
        _sync(dev)
        tempi["avanti"] = tempi.get("avanti", 0.0) + time.time() - t
        return r

    def indietro(*a, **k):
        _sync(dev)
        t = time.time()
        r = ind(*a, **k)
        _sync(dev)
        tempi["indietro"] = tempi.get("indietro", 0.0) + time.time() - t
        return r
    st.avanti, st.indietro = avanti, indietro
    try:
        yield
    finally:
        for k in ("avanti", "indietro"):
            if k in propri:
                setattr(st, k, propri[k])
            else:
                delattr(st, k)                               # torna visibile il metodo della classe
