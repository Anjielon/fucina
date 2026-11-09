#!/usr/bin/env python3
"""Riuso dentro il passo V (29/9/2026, ramo tt3-cure, `--pv-riuso` di e2e_allena). File NUOVO.

Perche'. In TT3 il passo 2 e' durato 12.231 s, di cui ~1 h 30 nella fase V (sonda a dosi golose + punto medio +
ricentro) con la GPU all'1-5 %: il tempo va nel lancio di operazioni piccole (py-spy: tabelle d'instradamento, controllo
fp16 della scala, sentinella), e col secondo piano dei densi le primitive sono raddoppiate. Nella fase V lo studente
fa ~32 avanti e ~27 avanti+indietro sugli STESSI 3-4 micro-lotti, e fra una prova e l'altra cambiano soltanto i byte dei
codici di UNA primitiva (una famiglia di uno strato): theta, F32 e LoRA sono fermi.

Cosa riusa (tutto acceso da `RiusoProve`, dentro un `with`; fuori, e con l'opzione spenta, niente cambia):
  1. PREFISSO PER STRATO. Per ogni micro-lotto (al massimo `max_lotti`, i piu' recenti) si tengono gli ingressi di
     ogni strato, il top-k salvato e il contesto posizionale. L'ingresso dello strato i dipende solo dagli strati < i:
     se la firma di uno strato (identita' e versione `_version` dei suoi byte, theta, F32, LoRA, parametri e buffer del
     modulo; ricetta) e' cambiata, si ricalcola da quello strato in giu' e non prima. Un salto annullato rialza la
     versione (i byte tornano uguali ma la firma no): si ricalcola, per prudenza, anche se non servirebbe.
  2. INDIETRO TRONCATO. Negli indietro della fase V (StatoV in modo 2 dopo `prepara_ricentro`) servono solo dL/dW_rot
     sulle colonne dei salti preparati: l'indietro si ferma al piu' basso strato che ne ha (come `_indietro_troncato`
     di --strati-allenati, prova D3: gradienti degli strati percorsi al bit). I gradienti di theta di questi indietro
     si buttano gia' (gradiente_ricentro/gradiente_lettere li rimettono): nulla si perde.
  3. PIANI E SCALE (e2e_studente.RIUSO): Piano a esperto unico dei densi per forma, Piano dell'avanti MoE riusato nel
     ricalcolo dell'indietro, scala fp16 di ogni famiglia fuori dal grafo riusata finche' theta non cambia.

Identita'. Con avanti deterministico, tutto quello che si legge (sonde, lettere, S del ricentro e dei prezzi al punto
medio) e' IDENTICO AL BIT a senza riuso (prove R1-R3 di e2e_test_riuso_prove.py). Una sola differenza DICHIARATA: il
controllo di trabocco fp16 dei gradienti (`valido` di gradiente_ricentro) guarda solo gli strati percorsi; un trabocco
negli strati sotto il salto piu' basso, che non tocca nessun numero letto, non invalida piu' il gradiente.
Memoria: ingressi fp32 (B, T, 2048) x 40 strati per lotto = 2,7 GB a 8192 token; `max_lotti` 2 = ~5,4 GB di VRAM
(le dosi golose alternano sonda A e lettere M1).
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Optional

import torch

import e2e_studente as S


class RiusoProve:
    """Context manager sul singolo `S.Studente` (metodi d'istanza sostituiti e rimessi). `stato_v`: lo StatoV del
    passo V (per l'indietro troncato: None = indietro intero). `conta`: dizionario di contatori (prove)."""

    def __init__(self, st, stato_v=None, max_lotti: int = 2, tronca: bool = True, conta: Optional[dict] = None):
        self.st = st
        self.stato_v = stato_v
        self.max_lotti = max(1, int(max_lotti))
        self.tronca = bool(tronca)
        self.conta = conta if conta is not None else {}
        for k in ("strati_avanti", "strati_saltati", "strati_indietro", "strati_indietro_saltati", "contesti_riusati"):
            self.conta.setdefault(k, 0)
        self._lotti: "OrderedDict" = OrderedDict()
        self._strati_ricentro: Optional[set] = None
        self._orig: dict = {}

    # ── firma di uno strato: tutto cio' da cui dipende la sua uscita dato l'ingresso
    def _tensori_strato(self, L: int) -> list:
        st = self.st
        es = st.esperti(L)
        ts = [f.blocchi for f in es.fam.values()] + [f.d0 for f in es.fam.values()] + list(es.theta.values())
        ts += [tf.master for tf in st._f32_strato.get(L, ())]
        mod = st.moduli[L]
        ts += list(mod.parameters()) + [b for b in mod.buffers() if b is not None]
        return ts

    def _firma(self, L: int) -> tuple:
        es = self.st.esperti(L)
        return (id(es.ric), id(self.st.ric)) + tuple((id(t), t._version) for t in self._tensori_strato(L))

    # ── avanti con prefisso riusato
    def _chiave(self, ids: torch.Tensor, inizi: Optional[torch.Tensor]) -> tuple:
        ki = None if inizi is None else (inizi.data_ptr(), tuple(inizi.shape), inizi._version, str(inizi.device))
        return (ids.data_ptr(), tuple(ids.shape), ids._version, str(ids.device), ki)

    @torch.no_grad()
    def _avanti(self, ids: torch.Tensor, inizi: Optional[torch.Tensor]) -> dict:
        st = self.st
        k = self._chiave(ids, inizi)
        v = self._lotti.get(k)
        n = len(st.strati)
        firme = [self._firma(L) for L in st.strati]
        if v is None:
            x = st.embedding(ids)
            B, T = x.shape[:2]
            v = {"ids": ids, "inizi": inizi, "ctx": st._contesto(B, T, x.dtype, inizi), "ingressi": [x] + [None] * n,
                 "instr": [None] * n, "firme": [None] * n}
            i0 = 0
        else:
            self._lotti.move_to_end(k)
            self.conta["contesti_riusati"] += 1
            i0 = next((i for i in range(n) if v["firme"][i] != firme[i]), n)
        x = v["ingressi"][i0]
        for i in range(i0, n):
            L = st.strati[i]
            x = st._strato(L, x, v["ctx"])
            v["instr"][i] = st.moduli[L].mlp.ultimo_idx
            v["firme"][i] = firme[i]
            v["ingressi"][i + 1] = x
        self.conta["strati_avanti"] += n - i0
        self.conta["strati_saltati"] += i0
        self._lotti[k] = v
        while len(self._lotti) > self.max_lotti:
            self._lotti.popitem(last=False)
        return {"xL": v["ingressi"][n], "ingressi": v["ingressi"][:n], "instr": list(v["instr"]), "ctx": v["ctx"]}

    def avanti(self, ids, x0=None, inizi=None):
        if x0 is not None:
            return self._orig["avanti"](ids, x0, inizi)
        return self._avanti(ids, inizi)

    def nascosto(self, ids, x0=None, inizi=None):
        if x0 is not None or torch.is_grad_enabled():   # nascosto col grafo (solo prove): quello di sempre
            return self._orig["nascosto"](ids, x0, inizi)
        return self.st.esce(self._avanti(ids, inizi)["xL"])

    # ── indietro troncato al salto piu' basso (solo nei gradienti della fase V: StatoV in modo 2 da prepara_ricentro)
    def _prepara_ricentro(self, salti: dict) -> None:
        self._strati_ricentro = {int(k[0]) for k, t in salti.items() if t[0].numel()}
        return self._orig["prepara_ricentro"](salti)

    def _fine_raccolta(self) -> None:
        self._strati_ricentro = None
        return self._orig["fine_raccolta"]()

    def indietro(self, stato: dict, g: torch.Tensor):
        st = self.st
        sv = self.stato_v
        if not (self.tronca and sv is not None and sv.modo == 2 and self._strati_ricentro is not None):
            n = len(st.strati)
            self.conta["strati_indietro"] += n
            return self._orig["indietro"](stato, g)
        n = len(st.strati)
        pos = {L: i for i, L in enumerate(st.strati)}
        cand = [pos[L] for L in self._strati_ricentro if L in pos]
        if st.strati_allenati is not None:          # --strati-allenati: non si scende mai sotto il primo allenato
            on = set(st.strati_allenati)
            ia = min(i for i, L in enumerate(st.strati) if L in on)
            cand = [max(i, ia) for i in cand]
        i0 = min(cand) if cand else n
        for i in reversed(range(i0, n)):
            L = st.strati[i]
            xin = stato["ingressi"][i].detach()
            if i > i0:
                xin.requires_grad_(True)
            with torch.enable_grad():
                y = st._strato(L, xin, stato["ctx"], instr=stato["instr"][i])
                if y.requires_grad:
                    y.backward(g)
            if i > i0:
                g = xin.grad
                if g is None:
                    raise RuntimeError(f"strato {L}: dL/dx assente nell'indietro troncato (riuso)")
        self.conta["strati_indietro"] += n - i0
        self.conta["strati_indietro_saltati"] += i0
        return None

    # ── accensione
    def __enter__(self) -> "RiusoProve":
        st = self.st
        self._orig = {"avanti": st.avanti, "nascosto": st.nascosto, "indietro": st.indietro}
        st.avanti, st.nascosto, st.indietro = self.avanti, self.nascosto, self.indietro
        if self.stato_v is not None:
            sv = self.stato_v
            self._orig["prepara_ricentro"], self._orig["fine_raccolta"] = sv.prepara_ricentro, sv.fine_raccolta
            sv.prepara_ricentro, sv.fine_raccolta = self._prepara_ricentro, self._fine_raccolta
        S.riuso_attivo(True)
        return self

    def __exit__(self, *exc) -> None:
        st = self.st
        for n in ("avanti", "nascosto", "indietro"):
            if st.__dict__.get(n) is not None:
                del st.__dict__[n]                  # torna il metodo della classe
        if self.stato_v is not None:
            for n in ("prepara_ricentro", "fine_raccolta"):
                if self.stato_v.__dict__.get(n) is not None:
                    del self.stato_v.__dict__[n]
        S.riuso_attivo(False)
        self._lotti.clear()
        self._orig = {}
