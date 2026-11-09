"""grouped_mm_thor (esperti_thor.py) contro grouped_mm di transformers: stessa uscita e
stessi gradienti (ingresso, pesi degli esperti, pesi di instradamento), sul VERO modulo
Qwen3_5MoeExperts costruito piccolo. Uso: python test_esperti_thor.py [cpu|cuda]
Su cuda misura anche i tempi a forme vere (256 esperti, 8192 token, top-8)."""
import sys
import time

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from esperti_thor import grouped_mm_experts_forward_thor  # noqa: E402
from transformers import AutoConfig  # noqa: E402
from transformers.integrations.moe import grouped_mm_experts_forward  # noqa: E402
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeExperts  # noqa: E402


def _prova(dev, E, H, I, T, k, dtype, seme=0, tempi=False):
    cfg = AutoConfig.from_pretrained("@@ROOT:models@@/gguf/thor-src-hf").text_config
    cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size = E, H, I
    torch.manual_seed(seme)
    mod = Qwen3_5MoeExperts(cfg).to(dev, dtype)
    with torch.no_grad():
        for p in mod.parameters():
            p.normal_(0, 0.02)
    h0 = torch.randn(T, H, device=dev, dtype=dtype)
    logit = torch.randn(T, E, device=dev) + torch.randn(E, device=dev) * 1.5   # carichi disuguali
    w0, idx = torch.topk(torch.softmax(logit, -1), k, dim=-1)
    go = torch.randn(T, H, device=dev, dtype=dtype)
    ris = {}
    for nome, fn in (("orig", grouped_mm_experts_forward), ("thor", grouped_mm_experts_forward_thor)):
        h = h0.clone().requires_grad_(True)
        w = w0.clone().to(dtype).requires_grad_(True)
        for p in mod.parameters():
            p.grad = None
        n = 3 if tempi else 1
        for _ in range(n):
            h.grad = None
            w.grad = None
            for p in mod.parameters():
                p.grad = None
            if dev == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            y = fn(mod, h, idx, w)
            (y.float() * go.float()).sum().backward()
            if dev == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
        ris[nome] = [y.detach().float(), h.grad.float(), w.grad.float()] + \
                    [p.grad.float() for p in mod.parameters()] + [dt]
    rel = lambda a, b: ((a - b).norm() / b.norm().clamp_min(1e-30)).item()
    diffs = [rel(a, b) for a, b in zip(ris["thor"][:-1], ris["orig"][:-1])]
    return diffs, ris["orig"][-1], ris["thor"][-1]


def main() -> int:
    dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    if dev == "cpu":
        torch.set_num_threads(2)
        casi = [(8, 64, 32, 50, 2, torch.float32, 1e-5), (16, 64, 32, 37, 4, torch.float32, 1e-5)]
    else:
        casi = [(256, 2048, 512, 8192, 8, torch.bfloat16, 2e-2)]
    guasti = 0
    for (E, H, I, T, k, dt, tol) in casi:
        diffs, t_o, t_t = _prova(dev, E, H, I, T, k, dt, tempi=(dev == "cuda"))
        ok = max(diffs) <= tol
        guasti += not ok
        nomi = ["uscita", "grad_h", "grad_pesi_instr", "grad_gate_up", "grad_down"]
        print(f"E={E} H={H} I={I} T={T} k={k} {dt}: " + " · ".join(f"{n} {d:.2e}" for n, d in zip(nomi, diffs))
              + f" · tempo fwd+bwd orig {t_o*1e3:.1f} ms thor {t_t*1e3:.1f} ms · {'OK' if ok else 'DIVERSO'}")
    print("OK" if not guasti else f"FALLITO ({guasti})")
    return 1 if guasti else 0


if __name__ == "__main__":
    sys.exit(main())
