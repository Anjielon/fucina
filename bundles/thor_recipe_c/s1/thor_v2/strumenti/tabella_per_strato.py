"""Tabella PER STRATO di un pannello in profondita' (25/9 01:2x).

Perche': la tabella del rapporto di pannello_heldout.py e' la SOMMA degli errori sugli strati
0..N-1; letta come "uscita dello strato N-1" ha fatto dichiarare una vittoria su tutti i domini
che allo strato 5 era un pareggio sul testo generale (wiki 0,955 contro 0,954). Qui, dal JSON
grezzo del pannello: rapporto appaiato contro la v20 per ogni strato e dominio (media ± errore
standard sulle batch) e differenze appaiate all'ULTIMO strato fra coppie di ricette.

L'errore standard misura solo la variabilita' fra batch dello STESSO modello forgiato. La
variabilita' fra due forge della stessa ricetta (rumore di fine addestramento) e' molto piu'
grande: si stima confrontando due corse della stessa ricetta (es. S0 e S0vecchio) e va usata
come soglia, non i sigma appaiati.

Uso: python3 strumenti/tabella_per_strato.py PANNELLO.json [--coppie A-B,C-D] [--domini math_code,wiki,edu]
"""
import argparse
import json
import sys


def media_es(x: list) -> tuple:
    n = len(x)
    mu = sum(x) / n
    es = (sum((a - mu) ** 2 for a in x) / (n - 1)) ** 0.5 / n ** 0.5 if n > 1 else float("nan")
    return mu, es


def tabella(grezzi: dict, ricette: list, dominio: str, rif: str = "v20") -> list:
    base = [m["per_strato"] for m in grezzi[rif][dominio]]
    n_strati = len(base[0])
    righe = []
    for L in range(n_strati):
        riga = {"strato": L}
        for r in ricette:
            x = [m["per_strato"][L] / b[L] for m, b in zip(grezzi[r][dominio], base)]
            riga[r] = media_es(x)
        righe.append(riga)
    return righe


def differenza_ultimo(grezzi: dict, a: str, b: str, dominio: str, rif: str = "v20") -> tuple:
    base = [m["per_strato"][-1] for m in grezzi[rif][dominio]]
    x = [(ma["per_strato"][-1] - mb["per_strato"][-1]) / v
         for ma, mb, v in zip(grezzi[a][dominio], grezzi[b][dominio], base)]
    return media_es(x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    ap.add_argument("--coppie", default="", help="coppie A-B separate da virgola per le differenze all'ultimo strato")
    ap.add_argument("--domini", default="math_code,wiki,edu")
    args = ap.parse_args()
    d = json.load(open(args.json))
    g = d["grezzi"]
    ricette = [r for r in g if r not in ("v20", "init_LS")]
    for dom in args.domini.split(","):
        if dom not in g["v20"]:
            print(f"⛔ dominio {dom} assente nel pannello", file=sys.stderr)
            return 2
        print(f"\n### {dom} — rapporto appaiato vs v20, per strato (media ± errore standard fra batch)\n")
        print("| strato | " + " | ".join(ricette) + " |")
        print("|---|" + "---|" * len(ricette))
        for riga in tabella(g, ricette, dom):
            print(f"| {riga['strato']} | " + " | ".join(f"{riga[r][0]:.3f} ± {riga[r][1]:.3f}" for r in ricette) + " |")
        for coppia in filter(None, args.coppie.split(",")):
            a, b = coppia.split("-")
            mu, es = differenza_ultimo(g, a, b, dom)
            print(f"\nultimo strato, {a} − {b}: {mu:+.4f} ± {es:.4f} (in punti di rapporto: {mu*100:+.1f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
