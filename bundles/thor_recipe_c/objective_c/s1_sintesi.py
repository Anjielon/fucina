#!/usr/bin/env python3
"""Sintesi di s1_fedele.py contro varianti_fedeli (stessi 250 densi, stesso bf16 puro, stessa H, errH nella base
ORIGINALE). uso: s1_sintesi.py <dir_s1> [--json file]"""
import json, math, sys, collections
VF = "@@ROOT:models@@/gguf/tony-tern/OBIETTIVO_C/varianti_fedeli/registro.jsonl"
d = sys.argv[1]
vf = {json.loads(l)["nome"]: json.loads(l) for l in open(VF)}
s1 = {json.loads(l)["nome"]: json.loads(l) for l in open(d + "/registro.jsonl")}
geo = lambda xs: math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")
com = sorted(set(vf) & set(s1))
mappa = {"a": "errH_gptq1", "b": "errH_gptq1jarq", "c": "errH_bil1jarq"}
out = {"dir": d, "comuni": len(com), "ricette": {}, "famiglie": {}}
for v, k in mappa.items():
    prima = [vf[n][k] for n in com]; dopo = [s1[n]["errH_" + v] for n in com]
    out["ricette"][v] = {"fedeli": geo(prima), "s1": geo(dopo), "migliorano": sum(b < a for a, b in zip(prima, dopo)),
                         "tt2": geo([vf[n]["errH_tt2"] for n in com]),
                         "sotto_tt2": sum(s1[n]["errH_" + v] < vf[n]["errH_tt2"] for n in com)}
fam = collections.defaultdict(list)
for n in com:
    fam[n.split(".")[2]].append(n)
for f, ns in sorted(fam.items()):
    out["famiglie"][f] = {v: [round(geo([vf[n][k] for n in ns]), 4), round(geo([s1[n]["errH_" + v] for n in ns]), 4),
                              sum(s1[n]["errH_" + v] < vf[n][k] for n in ns), len(ns)] for v, k in mappa.items()}
ab = [n for n in s1 if n.endswith((".ssm_alpha.weight", ".ssm_beta.weight"))]
if ab:
    out["alpha_beta"] = {v: geo([s1[n]["errH_" + v] for n in ab]) for v in "abc"}
    out["alpha_beta"]["n"] = len(ab)
out["osaq_solo_errH_geo"] = geo([s1[n]["errH_osaq_solo"] for n in com])
if "--json" in sys.argv:
    json.dump(out, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)
r = out["ricette"]
print(f"{len(com)} densi; errH geo (fedeli -> S1, migliorano/{len(com)}): "
      + " | ".join(f"{v} {r[v]['fedeli']:.4f}->{r[v]['s1']:.4f} ({r[v]['migliorano']})" for v in "abc")
      + (f"; alpha/beta {out['alpha_beta']['n']}: a {out['alpha_beta']['a']:.4f} b {out['alpha_beta']['b']:.4f} c {out['alpha_beta']['c']:.4f}" if ab else ""))
