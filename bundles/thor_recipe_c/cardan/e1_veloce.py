#!/usr/bin/env python3
"""e1_veloce.py — il passo di E1 piu' economico SENZA cambiare cio' che calcola (R007, 4/10).

Tre pezzi, tutti esatti o dichiarati; nessuno tocca la KL del modello intero:

1. CACHE DEL PREFISSO CONGELATO. Gli strati prima del PRIMO strato allenato (residui 12-27 => strato 12) non hanno
   parametri che cambiano: la loro uscita su un prompt e' una funzione fissa del prompt. Si calcola UNA volta, in
   lotti grandi (il costo di uno strato e' dominato dai pesi, non dai token), e ogni passo parte dallo strato 12 con
   x0 = stato in cache. Il resto dell'avanti, il ricalcolo e l'indietro troncato (`Studente._indietro_troncato`,
   che si ferma al primo strato allenato) sono quelli di sempre: stessi gradienti, stesso risultato.
   NON si mette in cache nient'altro: un confine DENTRO i blocchi allenati (es. l'uscita dello strato 19 mentre i
   residui 12-19 cambiano) darebbe a monte uno stato vecchio e al gradiente un bersaglio sbagliato: e' esattamente il
   difetto locale che ha bocciato PB1 (KL held-out 0,397 -> 0,527). `test_e1_veloce.py` lo dimostra (cache dentro il
   blocco = gradiente diverso; cache sul prefisso = gradiente uguale al bit).
2. BUDGET DI TEMPO. `passi_da_budget` fissa il numero di passi dopo aver MISURATO i primi, cosi' la pianificazione
   del lr (riscaldamento + coseno) arriva a zero dentro il budget invece di essere troncata dal tetto `--ore-max`.
3. PROFILO. `profila` misura avanti/indietro per numero di token: dice se il passo e' legato ai pesi (tempo quasi
   piatto in funzione dei token: conviene IMPACCHETTARE piu' gruppi per passo) o ai token (conviene il contrario).

Dipendenze: solo torch. Duck typing sullo studente: `strati`, `moduli`, `embedding`, `_contesto`, `_strato`,
`strati_allenati` (come `e2e_studente.Studente`, copia congelata in recupero_7dabd6f).
"""
from __future__ import annotations

import hashlib
import math
import time
from typing import Optional, Sequence

import torch


# ───────────────────────────────────────── avanti spezzato al confine del prefisso ─────────────────────────────────────────
def primo_strato_allenato(st) -> int:
    """Indice dello strato da cui si allena (minimo di `strati_allenati`); ValueError se tutto e' allenato/vuoto."""
    s = getattr(st, "strati_allenati", None)
    if not s:
        raise ValueError("nessuno strato congelato in testa: niente prefisso da mettere in cache")
    L0 = min(int(L) for L in s)
    if L0 <= min(st.strati):
        raise ValueError(f"il primo strato allenato ({L0}) e' il primo del modello: prefisso vuoto")
    return L0


@torch.no_grad()
def prefisso(st, ids: torch.Tensor, inizi: Optional[torch.Tensor], L0: int) -> torch.Tensor:
    """Uscita degli strati < L0 (ingresso dello strato L0), stessa sequenza di `Studente.avanti`."""
    x = st.embedding(ids)
    B, T = (x.shape[0], x.shape[1]) if x.dim() == 3 else (1, x.shape[0])
    ctx = st._contesto(B, T, x.dtype, inizi)
    for L in st.strati:
        if L >= L0:
            break
        x = st._strato(L, x, ctx)
    return x


@torch.no_grad()
def avanti_da(st, ids: torch.Tensor, inizi: Optional[torch.Tensor], x0: torch.Tensor, L0: int) -> dict:
    """Come `Studente.avanti` ma dallo strato L0 con ingresso x0. `ingressi`/`instr` hanno la stessa lunghezza e lo
    stesso indice di `st.strati` (None sotto L0: l'indietro troncato non li legge mai)."""
    B, T = (x0.shape[0], x0.shape[1]) if x0.dim() == 3 else (1, x0.shape[0])
    ctx = st._contesto(B, T, x0.dtype, inizi)
    x, ingressi, instr = x0, [], []
    for L in st.strati:
        if L < L0:
            ingressi.append(None)
            instr.append(None)
            continue
        ingressi.append(x)
        x = st._strato(L, x, ctx)
        instr.append(st.moduli[L].mlp.ultimo_idx)
    return {"xL": x, "ingressi": ingressi, "instr": instr, "ctx": ctx}


def impronta_ids(ids: Sequence[int]) -> str:
    return hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()[:16]


class CachePrefisso:
    """{id riga: stato (n_token, H) su CPU} all'ingresso dello strato L0. La chiave porta l'impronta dei token: una
    riga ritokenizzata in modo diverso NON trova lo stato vecchio (si ricalcola o, con `solo_cache`, si rifiuta)."""

    def __init__(self, st, L0: int):
        self.st, self.L0 = st, int(L0)
        self.stati: dict = {}
        self.secondi = 0.0
        self._tre = True

    @staticmethod
    def _chiave(r: dict) -> tuple:
        return (r["id"], impronta_ids(r["ids"]))

    def mancanti(self, rows: Sequence[dict]) -> list:
        return [r for r in rows if self._chiave(r) not in self.stati]

    def riempi(self, rows: Sequence[dict], token_max: int, device, log=print) -> int:
        """Calcola le righe che mancano, in lotti impacchettati (modo segmenti: ogni riga e' un segmento) fino a
        `token_max` token. Ritorna quante righe ha calcolato."""
        da_fare = self.mancanti(rows)
        if not da_fare:
            return 0
        t0, n, lotto, tok = time.time(), 0, [], 0
        for r in da_fare + [None]:
            if r is None or (lotto and tok + len(r["ids"]) > token_max):
                if lotto:
                    self._lotto(lotto, device)
                    n += len(lotto)
                    log(f"cache prefisso (strati < {self.L0}): {n}/{len(da_fare)} righe, {time.time() - t0:.0f} s")
                lotto, tok = [], 0
            if r is not None:
                lotto.append(r)
                tok += len(r["ids"])
        self.secondi += time.time() - t0
        return n

    def _lotto(self, rows: list, device) -> None:
        ids, inizi = [], []
        for r in rows:
            ids += r["ids"]
            inizi += [1] + [0] * (len(r["ids"]) - 1)
        x = prefisso(self.st, torch.tensor([ids], device=device), torch.tensor([inizi], device=device), self.L0)
        self._tre = x.dim() == 3                           # lo studente vero ha (B, T, H); un finto puo' avere (T, H)
        x = x[0] if self._tre else x
        i = 0
        for r in rows:
            k = len(r["ids"])
            self.stati[self._chiave(r)] = x[i:i + k].detach().cpu().clone()
            i += k

    def x0(self, rows: Sequence[dict], device) -> torch.Tensor:
        """(1, T, H) nell'ordine delle righe; KeyError se una riga non e' in cache (nessun ripiego silenzioso)."""
        x = torch.cat([self.stati[self._chiave(r)] for r in rows]).to(device)
        return x.unsqueeze(0) if self._tre else x

    def avanti(self, lot: dict) -> dict:
        """`lot` = uscita di `impacchetta` (ha "ids", "inizi", "rows"): lo stesso dizionario di `Studente.avanti`."""
        return avanti_da(self.st, lot["ids"], lot["inizi"], self.x0(lot["rows"], lot["ids"].device), self.L0)

    def byte(self) -> int:
        return sum(v.numel() * v.element_size() for v in self.stati.values())

    def salva(self, path, meta: Optional[dict] = None) -> None:
        torch.save({"L0": self.L0, "meta": meta or {}, "stati": self.stati}, path)

    def carica(self, path, meta: Optional[dict] = None) -> int:
        d = torch.load(path, map_location="cpu")
        if d["L0"] != self.L0 or (meta is not None and d.get("meta") != meta):
            return 0                                       # cache di un'altra corsa: si ignora, si ricalcola
        self.stati.update(d["stati"])
        return len(d["stati"])


# ───────────────────────────────────────── budget di tempo ─────────────────────────────────────────
def passi_da_budget(budget_s: float, speso_s: float, s_passo: float, s_eval: float, ogni: int, fatti: int,
                    massimo: int, minimo: int = 20, riserva_s: float = 120.0) -> int:
    """Numero TOTALE di passi (compresi i `fatti`) che sta nel budget.
    costo di un passo = s_passo + s_eval/ogni (la valutazione held-out ogni `ogni` passi); a fine corsa una valutazione
    e l'esportazione (`riserva_s`). Mai oltre `massimo`; sotto `minimo` ritorna `minimo` (il chiamante lo segnala:
    il budget non basta, meglio dirlo che accorciare in silenzio il lavoro)."""
    if min(s_passo, ogni) <= 0:
        raise ValueError("s_passo e ogni devono essere positivi")
    residuo = budget_s - speso_s - s_eval - riserva_s
    n = fatti + int(max(0.0, residuo) // (s_passo + s_eval / ogni))
    return max(minimo, min(massimo, n))


def mediana(v: Sequence[float]) -> float:
    s = sorted(v)
    return s[len(s) // 2] if s else float("nan")


# ───────────────────────────────────────── profilo token -> tempo ─────────────────────────────────────────
def _sync(device) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def regressione(xs: Sequence[float], ys: Sequence[float]) -> dict:
    """t = a + b x ai minimi quadrati. 'peso_fisso' = a / (a + b x_max): vicino a 1 = legato ai pesi."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx > 0 else 0.0
    a = my - b * mx
    a = max(a, 0.0)
    return {"a_s": a, "b_s_per_ktoken": b * 1000.0, "peso_fisso_a_xmax": a / max(a + b * max(xs), 1e-12)}


def profila(st, righe: list, token_list: Sequence[int], device, impacchetta, L0: int,
            ripetizioni: int = 2, perdita=None) -> dict:
    """Per ogni N token: tempo di (prefisso L<L0), avanti da L0 e indietro troncato, su righe vere impacchettate
    fino a ~N token. `impacchetta(gruppo_di_righe, device)` e' quella del driver. `perdita(lot, stato) -> (xL, loss)`
    opzionale: senza, l'indietro usa g = d/dx (xL^2).mean(). Ritorna le misure e la regressione tempo = a + b*token."""
    out = []
    for N in token_list:
        gr, tok = [], 0
        for r in righe:
            if tok >= N:
                break
            gr.append(r)
            tok += len(r["ids"])
        lot = impacchetta([gr], device)
        t_pre, t_av, t_in, prima = [], [], [], 0.0
        for rip in range(ripetizioni + 1):
            _sync(device); t0 = time.time()
            x0 = prefisso(st, lot["ids"], lot["inizi"], L0)
            _sync(device); t1 = time.time()
            stato = avanti_da(st, lot["ids"], lot["inizi"], x0, L0)
            _sync(device); t2 = time.time()
            xL = stato["xL"].detach().requires_grad_(True)
            with torch.enable_grad():
                (st.esce(xL).float().pow(2).mean()).backward()
            st.indietro(stato, xL.grad)
            _sync(device); t3 = time.time()
            if rip:                                        # la prima compila i kernel: si scarta dal conto...
                t_pre.append(t1 - t0); t_av.append(t2 - t1); t_in.append(t3 - t2)
            else:                                          # ...ma si registra: una forma NUOVA che costa >> una nota
                prima = t3 - t0                            # vuol dire ricompilazione per ogni lotto di taglia diversa
            for p in st.parametri():
                p.grad = None
        out.append({"token": int(lot["ids"].numel()), "prefisso_s": min(t_pre), "avanti_s": min(t_av),
                    "indietro_s": min(t_in), "prima_volta_s": prima})
    toks = [o["token"] for o in out]
    passo = [o["avanti_s"] + o["indietro_s"] for o in out]
    return {"L0": L0, "misure": out,
            "passo_da_L0": regressione(toks, passo),
            "prefisso": regressione(toks, [o["prefisso_s"] for o in out])}


def consiglia(profilo: dict, budget_s: float, token_per_passo_ref: int = 3040, passi_ref: int = 150) -> dict:
    """Dal profilo: secondi/passo alla taglia di riferimento e al ginocchio, e come spendere il budget.
    Legato ai pesi (peso_fisso > 0,6): i token extra costano poco -> meno passi, piu' gruppi per passo.
    Legato ai token: si resta alla taglia di riferimento e si accorcia solo il numero di passi."""
    f = profilo["passo_da_L0"]
    t_ref = f["a_s"] + f["b_s_per_ktoken"] * token_per_passo_ref / 1000.0
    legato_pesi = f["peso_fisso_a_xmax"] > 0.6
    n_ref = int(budget_s // max(t_ref, 1e-9))
    m = profilo.get("misure") or []
    ricompila = any(x["prima_volta_s"] > max(20.0, 3.0 * (x["prefisso_s"] + x["avanti_s"] + x["indietro_s"])) for x in m)
    out = {"s_passo_ref": t_ref, "passi_nel_budget_ref": n_ref, "legato_ai_pesi": legato_pesi,
           "ricompila_per_forma_nuova": ricompila,
           "passi_ref_richiesti": passi_ref, "ci_sta": n_ref >= passi_ref}
    if legato_pesi and f["b_s_per_ktoken"] > 0:
        # token per passo tali che il tempo cresca al piu' del 50 %: ginocchio della retta
        k = f["a_s"] * 0.5 / f["b_s_per_ktoken"] * 1000.0
        out["token_ginocchio"] = int(max(token_per_passo_ref, k))
    return out
