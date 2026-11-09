import json, sys, glob
for f in sorted(glob.glob("esito_torch_seq*.json")):
    e = json.load(open(f))
    for L, var in e["isolato"].items():
        for v, seqs in var.items():
            for s, x in seqs.items():
                print(f"{s:6s} L{L} {v:16s} l_out med {x['l_out']['mediana']:.4f} p99 {x['l_out']['p99']:.4f} glob {x['l_out_globale']:.4f} | apn med {x['attn_post_norm']['mediana']:.4f} | moe med {x['ffn_moe_out']['mediana']:.4f} glob {x['ffn_moe_out_globale']:.4f} | ffn med {x['ffn_out']['mediana']:.4f} | rout uguale {x['routing']['insieme_uguale']:.3f} sovr {x['routing']['sovrapposizione_media']:.4f}")
    for L, var in e["catena"].items():
        for v, seqs in var.items():
            for s, x in seqs.items():
                print(f"{s:6s} catena L{L} {v:12s} l_out med {x['l_out']['mediana']:.4f} p99 {x['l_out']['p99']:.4f} glob {x['l_out_globale']:.4f} rout {x['routing']['insieme_uguale']:.3f}")
    for s, x in e["seme"].items():
        print(s, "seme", {k: (round(v, 4) if isinstance(v, float) else {kk: round(vv, 4) for kk, vv in v.items()}) for k, v in x.items()})
