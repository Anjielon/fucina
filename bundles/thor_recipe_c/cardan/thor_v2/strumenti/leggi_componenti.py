#!/usr/bin/env python3
"""leggi_componenti — tabella passo x componente dell'attribuzione (27/9 integrata, `--attribuzione` di e2e_allena).

Legge <corsa>/registro_componenti.jsonl (una riga JSON per passo; un passo rifatto dopo una ripresa ha piu' righe:
vale l'ULTIMA) e stampa per ogni passo:
  L0 -> L_tutto sulla sonda fissa, Delta_i = L_tutto - L_senza_i per theta/f32/lora/codici (> 0 = quella componente ha
  PEGGIORATO la sonda: in ROSSO la piu' grande del passo), solo_i - L0, S-AdamW = L_tutto - L(theta col passo che AdamW
  avrebbe fatto dallo stesso stato: > 0 in ROSSO = Sophia peggio), held-out rapido/pieno (metrica), KL di train, minuti.
In fondo: somma dei Delta_i sui passi (quale componente ha peggiorato di piu' in tutto) e quante volte ciascuna e' stata
la peggiore.

Uso:  python3 strumenti/leggi_componenti.py @@ROOT:lavoro@@/e2e/allena/EB3 [--kl] [--ultimi N] [--senza-colori]
      --kl: Delta sulla KL in avanti invece che sulla perdita della corsa (CAKLD)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

COMP = ("theta", "f32", "lora", "codici")
ROSSO, GIALLO, FINE = "\033[31;1m", "\033[33m", "\033[0m"


def leggi(p: Path) -> list:
    """Righe per passo, l'ULTIMA vince (registro in coda: un passo rifatto dopo una ripresa si ripete)."""
    per = {}
    for n, riga in enumerate(p.read_text().splitlines(), 1):
        if not riga.strip():
            continue
        try:
            r = json.loads(riga)
        except json.JSONDecodeError:
            print(f"riga {n} illeggibile (troncata?): saltata", file=sys.stderr)
            continue
        per[int(r["passo"])] = r
    return [per[k] for k in sorted(per)]


def _f(x, w=9, p=5, segno=True) -> str:
    if x is None:
        return " " * (w - 1) + "-"
    return f"{x:+{w}.{p}f}" if segno else f"{x:{w}.{p}f}"


def tabella(righe: list, kl: bool = False, colori: bool = True) -> str:
    chiave_d, chiave_s = ("delta_kl", "dsolo_kl") if kl else ("delta", "dsolo")
    m = "perdita" if not kl else "kl"
    out = [f"{'passo':>5} {'L0':>9} {'L_tutto':>9} " + " ".join(f"{'D_' + c:>10}" for c in COMP) + "  " +
           " ".join(f"{'solo_' + c:>10}" for c in COMP) + f" {'S-AdamW':>10} {'rapido':>8} {'pieno':>8} {'train':>8} "
           f"{'min':>6}  peggiore"]
    somma = {c: 0.0 for c in COMP}
    peggiore_n = {c: 0 for c in COMP}
    for r in righe:
        a = r.get("attribuzione")
        if not a:
            continue
        d = a.get(chiave_d, {})
        pos = {c: v for c, v in d.items() if v > 0}
        pegg = max(pos, key=pos.get) if pos else None
        if pegg:
            peggiore_n[pegg] += 1
        celle = []
        for c in COMP:
            v = d.get(c)
            if v is not None:
                somma[c] += v
            s = _f(v, 10)
            if colori and c == pegg:
                s = ROSSO + s + FINE
            elif colori and v is not None and v > 0:
                s = GIALLO + s + FINE
            celle.append(s)
        solo = [_f(a.get(chiave_s, {}).get(c), 10) for c in COMP]
        ta = (a.get("theta_adam") or {}).get("delta_kl" if kl else "delta")   # > 0: il passo di Sophia peggio di AdamW
        s_ta = _f(ta, 10)
        if colori and ta is not None and ta > 0:
            s_ta = ROSSO + s_ta + FINE
        hr = (r.get("heldout_rapido") or {}).get("metrica")
        hp = (r.get("heldout") or {}).get("metrica")
        sec = r.get("secondi_passo")
        out.append(f"{r['passo']:>5} {_f(a['L0'][m], 9, 5, False)} {_f(a['L_tutto'][m], 9, 5, False)} " + " ".join(celle)
                   + "  " + " ".join(solo) + f" {s_ta} {_f(hr, 8, 5, False)} {_f(hp, 8, 5, False)} "
                   f"{_f(r.get('kl_train'), 8, 5, False)} {(sec or 0) / 60:6.1f}  {pegg or '-'}")
    out.append("")
    out.append("somma dei Delta_i sui passi (> 0 = ha peggiorato la sonda nel complesso): " +
               ", ".join(f"{c} {somma[c]:+.5f}" for c in COMP))
    out.append("passi in cui e' stata la PEGGIORE: " + ", ".join(f"{c} {peggiore_n[c]}" for c in COMP))
    return "\n".join(out)


def dettaglio(r: dict) -> str:
    """Diagnostica per componente e tempi dell'ultimo passo (la riga piu' recente)."""
    a = r.get("attribuzione") or {}
    dg = a.get("diagnostica") or {}
    s = [f"passo {r['passo']} ({r.get('data')}): tempi " + ", ".join(f"{k} {v:.0f} s" for k, v in
                                                                    (r.get("tempi") or {}).items())]
    for chiave, nome in (("heldout", "held-out PIENO"), ("heldout_rapido", "held-out rapido")):
        h = r.get(chiave)
        if h and h.get("kl_per_dominio"):
            s.append(f"  {nome}: metrica {h.get('metrica', float('nan')):.5f}; per dominio " +
                     ", ".join(f"{d} {x:.4f}" for d, x in h["kl_per_dominio"].items()))
    if r.get("per_origine"):                              # righe mc: perdita di allenamento per origine
        po = r["per_origine"]
        s.append("  perdita di allenamento per origine: " + ", ".join(
            f"{o} {po[o]:.5f} ({po[o + '_token']} token)" for o in ("base", "mc") if po.get(o) is not None))
    for chiave in ("heldout", "heldout_rapido"):
        mc = (r.get(chiave) or {}).get("mc")
        if mc:
            s.append(f"  held-out mc ({chiave}): " + ", ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}"
                                                                 for k, v in mc.items() if not isinstance(v, (dict, list))))
    if a.get("theta_adam"):
        ta = a["theta_adam"]
        s.append(f"  theta col passo di AdamW: sonda {ta['L']['perdita']:.5f} (tutto - adam {ta['delta']:+.5f}; > 0 = Sophia "
                 f"peggio), |passo AdamW|/lr mediana {ta['mediana_passo_adam_su_lr']:.4f}")
    if "theta" in dg:
        t = dg["theta"]
        s.append(f"  theta: |passo|/lr mediana {t['mediana_su_lr']:.4f}, q90 {t['q90_su_lr']:.4f}, a passo pieno "
                 f"{100 * t['quota_passo_pieno']:.2f}%, ferme {100 * t['quota_ferme']:.2f}%")
    if r.get("sophia"):
        so = r["sophia"]
        s.append(f"  sophia: non clippati {100 * so['quota_non_clippati']:.2f}% ({so['non_clippati']}/{so['vivi']})")
    if r.get("pesi"):
        p = r["pesi"]
        s.append("  pesi per token: " + ", ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}"
                                                  for k, v in p.items()))
    if r.get("pv"):
        v = r["pv"]
        s.append(f"  V: salti {v.get('salti')} ({'tenuti' if v.get('tenuto') else 'ANNULLATI'}), sonda "
                 f"{v.get('sonda_prima'):.5f} -> {v.get('sonda_dopo'):.5f}, rho {v.get('rho')}, rho dopo "
                 f"{v.get('rho_dopo')}, convenienti {v.get('convenienti')}" +
                 "".join(f", {k} {v[k]}" for k in v if k.startswith(("dose", "curva")) and k != "dosi"))
        if v.get("dosi"):                                 # --pv-sonda-dosi: curva dose -> sonda A (salti:sonda)
            s.append("  V dosi (salti:sonda A): " + ", ".join(
                f"{d.get('salti')}:{d.get('sonda_a', float('nan')):.5f}" for d in v["dosi"] if isinstance(d, dict)) +
                (f"; tenuti {v.get('salti_tenuti')}" if "salti_tenuti" in v else ""))
        cf = v.get("conferma")
        if isinstance(cf, dict):
            s.append("  V conferma sonda B: " + ", ".join(f"{k} {x:.5f}" if isinstance(x, float) else f"{k} {x}"
                                                        for k, x in cf.items()))
        rc = v.get("ricentro")
        if isinstance(rc, dict):
            s.append("  V ricentro: " + ", ".join(f"{k} {x:.4g}" if isinstance(x, float) else f"{k} {x}"
                                                  for k, x in rc.items() if not isinstance(x, (list, dict))) +
                     f", giri {len(rc.get('giri') or [])} (accettati {sum(bool(g.get('accettato')) for g in rc.get('giri') or [])})")
    if r.get("peso_dominio"):                             # --peso-dominio: quota della perdita per dominio
        pd = r["peso_dominio"]
        s.append(f"  peso per dominio: c {pd.get('c', float('nan')):.3f}; quota della perdita " +
                 ", ".join(f"{d} {q:.3f}" for d, q in (pd.get("quota_perdita") or {}).items()))
    for fam in ("f32", "lora"):
        if fam in dg:
            s.append(f"  {fam}: " + ", ".join(f"{f} " + "/".join(f"{x:.3e}" for x in d.values() if x is not None)
                                             for f, d in dg[fam].items()))
    return "\n".join(s)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("corsa", help="cartella della corsa (o il file registro_componenti.jsonl)")
    ap.add_argument("--kl", action="store_true", help="Delta sulla KL in avanti invece che sulla perdita della corsa")
    ap.add_argument("--ultimi", type=int, default=0, help="solo gli ultimi N passi")
    ap.add_argument("--senza-colori", action="store_true")
    a = ap.parse_args(argv)
    p = Path(a.corsa)
    if p.is_dir():
        p = p / "registro_componenti.jsonl"
    if not p.exists():
        print(f"{p}: non esiste (la corsa gira con --attribuzione?)", file=sys.stderr)
        return 1
    righe = leggi(p)
    if a.ultimi > 0:
        righe = righe[-a.ultimi:]
    colori = not a.senza_colori                      # il rosso resta anche in un file o in una pipe (less -R)
    print(tabella(righe, a.kl, colori))
    if righe:
        print()
        print(dettaglio(righe[-1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
