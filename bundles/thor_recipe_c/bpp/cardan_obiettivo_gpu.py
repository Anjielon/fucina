"""Modello misurato della somma pesata dei completamenti, senza torch.

Le alternative sono piani a ondate: una barriera dopo ogni ondata permette
di ritirare lavoratori senza interrompere unita'. Un contesto non misurato
NON viene stimato come privo di contesa e NON autorizza parallelismo.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
import os


@dataclass(frozen=True)
class Lavoro:
    chiave: str
    classe: str
    gruppo: str
    peso: float = 1.0

    def __post_init__(self):
        if any(not isinstance(x, str) or not x for x in (self.chiave, self.classe, self.gruppo)):
            raise ValueError("chiave/classe/gruppo devono essere stringhe non vuote")
        if not math.isfinite(self.peso) or self.peso <= 0:
            raise ValueError("priorita' positiva e finita obbligatoria")


def contesto(assegnazioni):
    return json.dumps(sorted((int(d), str(c), n) for (d, c), n in
                             Counter(assegnazioni).items()), separators=(",", ":"))


class Tempi:
    def __init__(self, campioni=()):
        self.dati = defaultdict(list)
        for c in campioni:
            self.aggiungi(c["contesto"], c["device"], c["classe"], c["secondi"])

    def aggiungi(self, ctx, device, classe, secondi):
        if not math.isfinite(secondi) or secondi <= 0:
            raise ValueError("tempo misurato non positivo/finito")
        parsed = json.loads(ctx)
        if (not parsed or any(not isinstance(r, list) or len(r) != 3 or
                              type(r[0]) is not int or not isinstance(r[1], str) or
                              type(r[2]) is not int or r[2] <= 0 for r in parsed)):
            raise ValueError("contesto misurato non valido")
        espanso = [(d, c) for d, c, n in parsed for _ in range(n)]
        if ctx != contesto(espanso) or (device, classe) not in espanso:
            raise ValueError("contesto non canonico o campione estraneo")
        self.dati[(ctx, int(device), classe)].append(secondi)

    def leggi(self, ctx, device, classe):
        valori = self.dati.get((ctx, device, classe))
        if not valori:
            raise LookupError(f"tempo NON misurato: {ctx}/{device}/{classe}")
        # Ultimi campioni, senza perpetuare condizioni di carico ormai cessate.
        return sum(valori[-8:]) / len(valori[-8:])

    def esporta(self):
        return [{"contesto": ctx, "device": d, "classe": c, "secondi": s}
                for (ctx, d, c), vv in sorted(self.dati.items()) for s in vv[-8:]]


def ordina(lavori, tempi, devices):
    gruppi = defaultdict(list)
    for u in lavori:
        gruppi[u.gruppo].append(u)
    pesi, durate = {}, {}
    for g, uu in gruppi.items():
        if len({u.peso for u in uu}) != 1:
            raise ValueError(f"pesi incoerenti per il lavoro {g}")
        pesi[g] = uu[0].peso
        durate[g] = 0.0
        for u in uu:
            tt = []
            for d in devices:
                try:
                    tt.append(tempi.leggi(contesto([(d, u.classe)]), d, u.classe))
                except LookupError:
                    pass
            if not tt:
                raise LookupError(f"tempo isolato mancante: {u.classe}")
            durate[g] += min(tt)
    ordine = sorted(gruppi, key=lambda g: (durate[g] / pesi[g], g))
    return [u for g in ordine for u in sorted(gruppi[g], key=lambda u: u.chiave)]


def simula(lavori, posti, tempi, avvio=None, residenti=None):
    """Somma peso * fine del GRUPPO, non delle sue singole unita'.

    I tempi sono completamenti WALL misurati dall'avvio dell'ondata, NON
    tempi kernel o velocita' istantanee. Comprendono lancio sfalsato,
    pubblicazione e la minore contesa dopo la fine dei compagni d'ondata.
    Non estrapolare frazioni di lavoro lineari da quei completamenti.
    """
    residenti = residenti or {}
    slots = [d for d, n in sorted(posti.items()) for _ in range(n)]
    if not slots:
        raise ValueError("nessun posto")
    coda = ordina(lavori, tempi, sorted(d for d, n in posti.items() if n > 0))
    ora = 0.0
    for d, n in posti.items():
        nuovi = max(0, n - residenti.get(d, 0))
        if not nuovi:
            continue
        if avvio is not None and str(d) not in avvio:
            raise LookupError(f"tempo di caricamento worker NON misurato: device {d}")
        ora += nuovi * float((avvio or {}).get(str(d), 0))
    if not math.isfinite(ora) or ora < 0:
        raise ValueError("tempo avvio non valido")
    fine, pesi, ondate = {}, {}, []
    while coda:
        onda = list(zip(coda[:len(slots)], slots))
        del coda[:len(slots)]
        ondate.append([(u.chiave, d) for u, d in onda])
        ctx = contesto([(d, u.classe) for u, d in onda])
        durate = [tempi.leggi(ctx, d, u.classe) for u, d in onda]
        for (u, _), durata in zip(onda, durate):
            pesi[u.gruppo] = u.peso
            fine[u.gruppo] = max(fine.get(u.gruppo, 0), ora + durata)
        ora += max(durate)
    return {"somma_pesata": sum(pesi[g] * t for g, t in fine.items()),
            "completamenti": fine, "ondate": ondate}


def scegli(lavori, limiti, tempi, avvio=None, residenti=None, max_lavoratori=None, guadagno_minimo=None):
    """Confronta serie su ogni device e incrementi di un lavoratore.

    Ricerca locale esplicita, NON un'affermazione di ottimo combinatorio.
    Senza baseline misurata ammette solo una calibrazione isolata.
    Regola di contesa (Angelo 21/10 13:2x, c): un lavoratore in piu' entra solo se il rendimento complessivo
    cresce almeno di guadagno_minimo (FORGIA_GUADAGNO_MINIMO, 0,10): somma pesata stimata con N+1 <=
    somma con N / (1 + guadagno_minimo) (a parita' di lavoro, tempo ~ 1/rendimento).
    """
    if guadagno_minimo is None:
        guadagno_minimo = float(os.environ.get("FORGIA_GUADAGNO_MINIMO", "0.10"))
    tetto = min(len(lavori), sum(limiti.values()) if max_lavoratori is None else max_lavoratori)
    if tetto < 1:
        raise ValueError("nessun lavoratore entra nel tetto globale")
    alternative = []

    def valuta(posti):
        try:
            r = simula(lavori, posti, tempi, avvio, residenti)
            rec = {"posti": posti.copy(), **r, "ammessa": True}
        except LookupError as e:
            rec = {"posti": posti.copy(), "somma_pesata": None,
                   "ammessa": False, "motivo": str(e)}
        alternative.append(rec)
        return rec

    candidati = [valuta({d: int(d == dd) for d in limiti})
                 for dd, n in limiti.items() if n > 0]
    misurati = [c for c in candidati if c["ammessa"]]
    if not misurati:
        # Prima le coppie classe/device senza baseline, anche sulle altre GPU.
        coppie = [(u, d) for u in lavori for d, n in limiti.items() if n > 0]
        u, d = min(coppie, key=lambda ud: (
            bool(tempi.dati.get((contesto([(ud[1], ud[0].classe)]), ud[1], ud[0].classe))),
            -ud[0].peso, ud[0].gruppo, ud[0].chiave, ud[1]))
        scelta = {"posti": {dd: int(dd == d) for dd in limiti},
                  "somma_pesata": None, "ondate": [[(u.chiave, d)]],
                  "calibrazione": True}
    else:
        scelta = min(misurati, key=lambda c: c["somma_pesata"])
        while sum(scelta["posti"].values()) < tetto:
            nuove = []
            for d, n in limiti.items():
                if scelta["posti"][d] < n:
                    p = scelta["posti"].copy()
                    p[d] += 1
                    nuove.append(valuta(p))
            migliori = [c for c in nuove if c["ammessa"] and
                        c["somma_pesata"] < scelta["somma_pesata"] / (1 + guadagno_minimo) - 1e-9]
            for c in nuove:
                c["guadagno_minimo"] = guadagno_minimo
                c["soglia_somma_pesata"] = scelta["somma_pesata"] / (1 + guadagno_minimo)
            if not migliori:
                break
            scelta = min(migliori, key=lambda c: c["somma_pesata"])
    return {"criterio": "sum(peso_lavoro * fine_lavoro)",
            "alternative": alternative, "scelta": scelta}
