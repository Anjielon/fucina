#!/usr/bin/env python3
"""Giudice UNICO della sonda parlata (leva L0, 13/10/2026).

Perche' esiste: fino al 13/10 ogni copia della sonda giudicava con
`re.search(ATTESA, contenuto)` su TUTTO il testo. Per D2 la regex era `\\b3\\b`:
un modello che riscrive il problema («Alice ha 3 fratelli…») e risponde 2
risultava GIUSTO. Il giudice premiava la verbosita' (diagnosi:
ricerche/C_SONDA_DIAGNOSI_1013.md §2.1). Inoltre «MUTO» non distingueva il
troncamento a max_tokens (finish_reason=length) dal silenzio vero.

Regola nuova: si giudica la RISPOSTA FINALE DICHIARATA, non un numero isolato.
  1. si toglie tutto fino all'ultimo </think> (pensiero trapelato nel contenuto);
  2. si tolgono Markdown e LaTeX ({,} -> ,  \\times -> x  $ ** ` #);
  3. la risposta dichiarata, in ordine di precedenza:
     a) dopo l'ULTIMO marcatore esplicito («risposta (finale)», «answer», «la risposta e'»),
     b) l'ULTIMO \\boxed{...},
     c) l'ULTIMO segmento che contiene SOLO la risposta (un numero/una parola),
     d) l'ULTIMO segmento con un connettivo conclusivo («quindi», «pertanto», «in totale», «so», «therefore»),
     e) il PRIMO segmento che non e' un passaggio di calcolo (la domanda chiede «il solo numero»:
        cio' che viene dopo e' spiegazione),
     f) l'ultimo segmento;
  4. dal segmento scelto: dominio numerico -> l'ULTIMO numero; dominio parola -> la parola.
Esiti: OK · SBAGLIA · TRONCATA (finish_reason=length) · MUTA (fine regolare, contenuto vuoto) · GUASTO.

Uso dalla sonda bash (JSON della risposta su stdin):
  curl ... | python3 sonda_giudice.py giudica --dominio D2 --seed 7 --grezzo FILE.jsonl
  -> stampa «ESITO|car_pensiero|tok_s|estratta|anteprima»
Uso in libreria:  from sonda_giudice import giudica, DOMINI
"""
import argparse
import json
import re
import sys

# ── i domini: domanda, tipo, attesa (una sola fonte per tutte le sonde) ──────────────────────────────────
DOMINI = {
    "D1": {"domanda": "Un prodotto costa 23 euro. Sconto del 15 per cento, poi meno altri 2 euro. "
                      "Quanto costa alla fine? Rispondi col solo numero.",
           "tipo": "numero", "attesa": 17.55},
    "D2": {"domanda": "Alice ha 3 fratelli e 2 sorelle. Quante sorelle ha il fratello di Alice? "
                      "Rispondi col solo numero.",
           "tipo": "numero", "attesa": 3},
    "D3": {"domanda": "Scrivi la parola CANE al contrario. Rispondi con la sola parola.",
           "tipo": "parola", "attesa": "ENAC"},
}

NUM = r"-?\d+(?:[.,]\d+)?"
_MARCATORI = re.compile(r"(?:risposta(?:\s+finale)?|final\s+answer|answer|la\s+risposta\s+(?:è|e'|e))\s*[:=è]?\s*",
                        re.I)
_CONCLUSIVI = re.compile(r"\b(?:quindi|pertanto|dunque|in\s+totale|alla\s+fine|therefore|so\s+the|thus|"
                         r"il\s+risultato|risultato\s+finale)\b", re.I)
_PASSO = re.compile(r"=|\d\s*[x×*/+\-−]\s*\d|\bpasso\b|\bstep\b", re.I)


def _pulisci(t: str) -> str:
    """Toglie pensiero trapelato, LaTeX e Markdown; conserva gli a capo."""
    if "</think>" in t:
        t = t.rsplit("</think>", 1)[1]
    elif "<think>" in t:          # pensiero aperto e mai chiuso: niente risposta
        return ""
    t = t.replace("{,}", ",").replace("{.}", ".")
    t = re.sub(r"\\(?:times|cdot)", " x ", t)
    t = re.sub(r"\\(?:text|mathrm|mathbf)\{([^}]*)\}", r"\1", t)
    t = re.sub(r"\\[,;!: ]", "", t)                   # spazi LaTeX
    t = t.replace("\\(", " ").replace("\\)", " ").replace("\\[", " ").replace("\\]", " ")
    t = re.sub(r"[$*`_]", "", t)
    return t


def _segmenti(t: str) -> list:
    """Spezza in segmenti: a capo, poi frasi. Le intestazioni senza cifre si scartano."""
    out = []
    for riga in re.split(r"\n+", t):
        r = riga.strip()
        if not r:
            continue
        if r.startswith("#"):
            r = r.lstrip("#").strip()
            if not re.search(r"\d", r):
                continue
        r = re.sub(r"^(?:[-•>]\s+|\d+[.)]\s+(?=\D))", "", r).strip()
        for frase in re.split(r"(?<=[.!?;])\s+(?=[A-ZÀ-Ý(\-•])", r):
            if frase.strip():
                out.append(frase.strip())
    return out


def _nudo(seg: str, tipo: str) -> bool:
    s = seg.strip(" .:;!()[]{}\"'")
    s = re.sub(r"\s*(?:€|euro|eur|sorelle?)\s*$", "", s, flags=re.I)
    if tipo == "numero":
        return re.fullmatch(NUM, s) is not None
    return re.fullmatch(r"[A-Za-zÀ-ÿ]{2,}", s) is not None


def _valore(seg: str, tipo: str, domanda: str):
    if tipo == "numero":
        nums = re.findall(NUM, seg)
        return float(nums[-1].replace(",", ".")) if nums else None
    parole = re.findall(r"[A-Za-zÀ-ÿ]{2,}", seg)
    if not parole:
        return None
    if len(parole) == 1:
        return parole[0].upper()
    nella_domanda = {p.upper() for p in re.findall(r"[A-Za-zÀ-ÿ]{2,}", domanda)}
    maiuscole = [p for p in parole if p.isupper() and p.upper() not in nella_domanda]
    return maiuscole[-1].upper() if maiuscole else None   # ambigua: nessuna parola dichiarata


def estrai(contenuto: str, tipo: str, domanda: str = "", anteprima: bool = False):
    """-> (valore o None, fonte, segmento). Fonte: marcatore|boxed|nudo|conclusivo|primo|ultimo|vuoto.
    anteprima=True: testo dai log storici, dove ogni a capo e' diventato uno spazio (due spazi = riga vuota)."""
    if anteprima:
        contenuto = re.sub(r" {2,}", "\n", contenuto)
    boxed = re.findall(r"\\boxed\{([^{}]*)\}", contenuto.replace("{,}", ","))
    t = _pulisci(contenuto)
    if not t.strip():
        return None, "vuoto", ""
    m = list(_MARCATORI.finditer(t))
    if m:
        dopo = t[m[-1].end():].strip()
        seg = (_segmenti(dopo) or [dopo])[0]
        v = _valore(seg, tipo, domanda)
        if v is not None:
            return v, "marcatore", seg
    if boxed:
        v = _valore(_pulisci(boxed[-1]), tipo, domanda)
        if v is not None:
            return v, "boxed", boxed[-1]
    segs = _segmenti(t)
    if not segs:
        return None, "vuoto", ""
    nudi = [s for s in segs if _nudo(s, tipo)]
    if nudi:
        return _valore(nudi[-1], tipo, domanda), "nudo", nudi[-1]
    concl = [s for s in segs if _CONCLUSIVI.search(s)]
    if concl:
        seg = concl[-1]
        v = _valore(seg[_CONCLUSIVI.search(seg).end():], tipo, domanda)   # solo cio' che segue il connettivo
        if v is not None:
            return v, "conclusivo", seg
    if not _PASSO.search(segs[0]):
        v = _valore(segs[0], tipo, domanda)
        if v is not None:
            return v, "primo", segs[0]
    return _valore(segs[-1], tipo, domanda), "ultimo", segs[-1]


def confronta(valore, dominio: str) -> bool:
    d = DOMINI[dominio]
    if valore is None:
        return False
    if d["tipo"] == "numero":
        return abs(float(valore) - float(d["attesa"])) < 1e-6
    return str(valore).upper() == str(d["attesa"]).upper()


def giudica(contenuto: str, dominio: str, finish_reason=None, anteprima: bool = False) -> dict:
    """Esito della risposta. TRONCATA ha la precedenza: un testo tagliato non e' una risposta dichiarata."""
    c = (contenuto or "").strip()
    if finish_reason == "length":
        return {"esito": "TRONCATA", "valore": None, "fonte": "length", "segmento": ""}
    if not c:
        return {"esito": "MUTA", "valore": None, "fonte": "vuoto", "segmento": ""}
    v, fonte, seg = estrai(c, DOMINI[dominio]["tipo"], DOMINI[dominio]["domanda"], anteprima)
    if fonte == "vuoto":                      # solo pensiero trapelato, mai chiuso
        return {"esito": "MUTA", "valore": None, "fonte": "solo-pensiero", "segmento": ""}
    return {"esito": "OK" if confronta(v, dominio) else "SBAGLIA", "valore": v, "fonte": fonte, "segmento": seg}


def dominio_da_domanda(domanda: str):
    for k, d in DOMINI.items():
        if d["domanda"][:40] == (domanda or "")[:40]:
            return k
    return None


def _cli_giudica(a):
    grezzo = sys.stdin.read()
    try:
        d = json.loads(grezzo)
    except Exception as e:  # noqa: BLE001 — la sonda deve vedere il guasto, non morire
        print(f"GUASTO|0|0||risposta illeggibile: {str(e)[:60]}")
        if a.grezzo:
            with open(a.grezzo, "a", encoding="utf-8") as f:
                f.write(json.dumps({"seed": a.seed, "dominio": a.dominio, "esito": "GUASTO",
                                    "grezzo": grezzo[:4000]}, ensure_ascii=False) + "\n")
        return
    ch = (d.get("choices") or [{}])[0]
    m = ch.get("message", {}) or {}
    rc = (m.get("reasoning_content") or "").strip()
    c = (m.get("content") or "").strip()
    fr = ch.get("finish_reason")
    g = giudica(c, a.dominio, fr)
    tps = round((d.get("timings") or {}).get("predicted_per_second", 0), 1)
    if a.grezzo:
        with open(a.grezzo, "a", encoding="utf-8") as f:
            f.write(json.dumps({"seed": a.seed, "dominio": a.dominio, "domanda": DOMINI[a.dominio]["domanda"],
                                "esito": g["esito"], "valore": g["valore"], "fonte": g["fonte"],
                                "finish_reason": fr, "content": m.get("content"),
                                "reasoning_content": m.get("reasoning_content"),
                                "timings": d.get("timings"), "usage": d.get("usage")}, ensure_ascii=False) + "\n")
    val = "" if g["valore"] is None else str(g["valore"])
    print(f"{g['esito']}|{len(rc)}|{tps}|{val}|{c[:110].replace(chr(10), ' ').replace('|', '/')}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    s = p.add_subparsers(dest="cmd", required=True)
    g = s.add_parser("giudica", help="giudica una risposta JSON di /v1/chat/completions letta da stdin")
    g.add_argument("--dominio", required=True, choices=sorted(DOMINI))
    g.add_argument("--seed", default="")
    g.add_argument("--grezzo", default="", help="jsonl in cui accodare la risposta intera")
    q = s.add_parser("domanda", help="stampa la domanda del dominio")
    q.add_argument("dominio", choices=sorted(DOMINI))
    a = p.parse_args()
    if a.cmd == "giudica":
        _cli_giudica(a)
    else:
        print(DOMINI[a.dominio]["domanda"])


if __name__ == "__main__":
    main()
