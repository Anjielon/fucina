"""P5: TQ1_0 pesato (ricetta 20set a gruppi da 256, niente rotazione): catena non crescente, round-trip dai byte (gguf-py)
esatto, pesato-H migliore dell'ottimo non pesato; negativo: GROUP ripristinato a 128 dopo la chiamata."""
import os, sys, numpy as np
os.environ.setdefault("TT_DEVICE", "cpu"); os.environ["TT_RICETTA"] = "20set"
sys.path.insert(0, "."); sys.path.insert(0, "@@ROOT:models@@/build/llamacpp-tq1-k1024/gguf-py")
import tt_quant as TQ, gptq_1024_lib as G, gptq_1024_refine as RF
rng = np.random.default_rng(5)
C, R = 512, 256
A = rng.standard_normal((C, 3 * C)) * rng.uniform(0.2, 3, (C, 1)); H = A @ A.T / (3 * C)
W = (rng.standard_normal((R, C)) * 0.02).astype(np.float32)
p, m = TQ.quantizza_tq1_0(W, H, "prova")
c = m["catena_pesata_rel"]
ok = {"catena_non_crescente": c["scale_congiunte"] <= c["gptq"] + 1e-12,
      "roundtrip_zero": m["roundtrip_max_abs"] == 0.0,
      "pesato_meglio": m["err_rel_pesato_H"] < m["rif_nonpesato_err_rel_pesato_H"],
      "GROUP_ripristinato": G.GROUP == 128 and RF.GROUP == 128,
      "blocchi_54B": p.shape == (R * C // 256, 54)}
print(m); print(ok); print("TUTTO VERDE" if all(ok.values()) else "⛔ ROSSO")
