"""MMLU 2000 del Dell: confronto APPAIATO di una variante con EB3 p20 (stessi compiti, stesso banco).
Δ accuratezza, z appaiato (media delle differenze / errore standard), discordanti, bootstrap stratificato per materia
(10000, seme fisso), Δ logp_scelte (log p della giusta normalizzata sulle 4) appaiato."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "@@ROOT:forge@@/thor_v2/strumenti")
import mmlu_depurato as MD  # noqa: E402  (sola lettura)

reg, rif = Path(sys.argv[1]), Path(sys.argv[2] if len(sys.argv) > 2 else
                                    "@@ROOT:models@@/gguf/tony-tern/mmlu_logprob/aragorn/EB3-passo_000020.log")
mappa = json.loads(MD.MAPPA.read_text(encoding="utf-8"))
assert mappa["sha256_compiti"] == MD.SHA_BANCO
oro = np.array([MD.LETTERE.index(x) for x in mappa["lettera"]])
materia = np.array(mappa["materia"])
out = {}
acc, lsc = {}, {}
for k, p in (("var", reg), ("rif", rif)):
    r = MD.leggi_registro(p)
    assert (r["giusta"] == oro).all(), p
    lp = r["lp32"].astype(np.float64)
    acc[k] = (lp.argmax(1) == oro).astype(np.float64)
    S = lp - np.log(np.exp(lp - lp.max(1, keepdims=True)).sum(1, keepdims=True)) - lp.max(1, keepdims=True)
    lsc[k] = S[np.arange(len(oro)), oro]
    out[f"acc_{k}"] = 100 * acc[k].mean()
    out[f"finale_{k}"] = r["finale"]
d = acc["var"] - acc["rif"]
n = len(d)
out["delta_acc"] = 100 * d.mean()
out["z_appaiato"] = float(d.mean() / (d.std(ddof=1) / np.sqrt(n)))
out["giusta_qui_sbagliata_rif"] = int(((acc["var"] == 1) & (acc["rif"] == 0)).sum())
out["sbagliata_qui_giusta_rif"] = int(((acc["var"] == 0) & (acc["rif"] == 1)).sum())
dl = lsc["var"] - lsc["rif"]
out["delta_logp_scelte"] = float(dl.mean())
out["z_logp_scelte"] = float(dl.mean() / (dl.std(ddof=1) / np.sqrt(n)))
rng = np.random.default_rng(20260928)
strati = [np.nonzero(materia == s)[0] for s in sorted(set(materia))]
bs = np.empty(10000)
for b in range(10000):
    idx = np.concatenate([s[rng.integers(0, len(s), len(s))] for s in strati])
    bs[b] = 100 * d[idx].mean()
out["ic95_delta_acc_bootstrap_stratificato"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
print(json.dumps(out, indent=1))
