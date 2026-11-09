"""Stima dei costi (FLOP, memoria, lanci di kernel, tempo) del menu completo.

I conti seguono esattamente le operazioni di menu.py:
  preparazione : Cholesky(A) + inversa + Cholesky(A^-1) ~ (1/3 + 1 + 1/3) i^3
                 rifit delle scale correnti 2 K o i^2, distorsione 2 (o i^2 + o^2 i) x2
  "+" (pieno o sparso): GPTQ o i^2 + rifit 2 (K+1) o i^2 + una distorsione (riusata dal menu)
  "-1"         : K volte (rifit 2 (K-1) o i^2 + distorsione)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class TensorSpec:
    kind: str
    out: int
    in_: int
    n_planes: int
    count_per_layer: int        # 256 per gli esperti, 1 per i densi
    shares_input_with: str = "" # es. up condivide H_I (e quindi Cholesky) con gate


@dataclass
class CostReport:
    flops: Dict[str, float] = field(default_factory=dict)
    seconds: Dict[str, float] = field(default_factory=dict)
    launches: float = 0.0
    peak_batch_bytes: float = 0.0

    @property
    def total_seconds(self) -> float:
        return sum(self.seconds.values())


def estimate(specs: List[TensorSpec], layers: int, n_plus: int, allow_drop_min: int = 2,
             tflops_fp32: float = 4.0, tflops_mm: float = 12.0, launch_us: float = 25.0,
             batch: int = 64, share_cholesky: bool = False) -> CostReport:
    """tflops_fp32: throughput EFFETTIVO per Cholesky e passo colonna GPTQ;
    tflops_mm: throughput effettivo dei grandi prodotti (fp16/bf16 WMMA se mm_dtype ridotto)."""
    r = CostReport()
    f = {"cholesky": 0.0, "gptq": 0.0, "refit": 0.0, "distortion": 0.0}
    for s in specs:
        n = s.count_per_layer * layers
        o, i, K = s.out, s.in_, s.n_planes
        if not (share_cholesky and s.shares_input_with):
            f["cholesky"] += n * (5.0 / 3.0) * i ** 3
        dist = 2.0 * (o * i * i + o * o * i)
        f["refit"] += n * 2 * K * o * i * i
        f["distortion"] += n * 2 * dist
        f["gptq"] += n * n_plus * o * i * i
        f["refit"] += n * n_plus * 2 * (K + 1) * o * i * i
        f["distortion"] += n * n_plus * dist
        if K >= allow_drop_min:
            f["refit"] += n * K * 2 * (K - 1) * o * i * i
            f["distortion"] += n * K * dist
        # lanci: ogni chiamata GPTQ su un lotto ha ~12 kernel per colonna; i lotti
        # raggruppano tensori della stessa forma ANCHE fra layer diversi
        n_batches = -(-n // batch)
        r.launches += n_batches * n_plus * i * 12
        # memoria di un lotto: W, R, T, D, A, G, U, Y del rifit (a blocchi di 128 righe)
        mem = batch * (6 * o * i * 4 + 3 * i * i * 4 + o * o * 4 + (K + 1) * 128 * (i // 256) * i * 4)
        r.peak_batch_bytes = max(r.peak_batch_bytes, mem)
    r.flops = f
    r.seconds = {"cholesky": f["cholesky"] / (tflops_fp32 * 1e12),
                 "gptq": f["gptq"] / (tflops_fp32 * 1e12),
                 "refit": f["refit"] / (tflops_mm * 1e12),
                 "distortion": f["distortion"] / (tflops_mm * 1e12),
                 "launch_overhead": r.launches * launch_us * 1e-6}
    return r


def qwen_moe_specs(hidden: int = 2048, inter: int = 512, experts: int = 256,
                   dense: Tuple[Tuple[str, int, int], ...] = ()) -> List[TensorSpec]:
    """Esperti a 1 piano (gate/up [inter,hidden], down [hidden,inter]) + densi a 3 piani."""
    s = [TensorSpec("gate", inter, hidden, 1, experts),
         TensorSpec("up", inter, hidden, 1, experts, shares_input_with="gate"),
         TensorSpec("down", hidden, inter, 1, experts)]
    s += [TensorSpec(k, o, i, 3, 1) for k, o, i in dense]
    return s


if __name__ == "__main__":
    dense = (("q", 4096, 2048), ("k", 512, 2048), ("v", 512, 2048), ("o", 2048, 4096),
             ("shared_gate", 512, 2048), ("shared_up", 512, 2048), ("shared_down", 2048, 512))
    for label, kw in [("fp32 ovunque (4 TF eff.)", dict(tflops_fp32=4, tflops_mm=4)),
                      ("fp16 WMMA sui prodotti (12 TF eff.)", dict(tflops_fp32=4, tflops_mm=12))]:
        rep = estimate(qwen_moe_specs(dense=dense), layers=40, n_plus=4, **kw)
        print(label)
        for k, v in rep.flops.items():
            print(f"  {k:12s} {v / 1e12:9.1f} TFLOP  {rep.seconds[k] / 60:7.1f} min")
        print(f"  lanci {rep.launches / 1e6:.2f} M -> {rep.seconds['launch_overhead'] / 60:.1f} min")
        print(f"  totale {rep.total_seconds / 60:.1f} min, picco lotto {rep.peak_batch_bytes / 2**30:.1f} GiB")
