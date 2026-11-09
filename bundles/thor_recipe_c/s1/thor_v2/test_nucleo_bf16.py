"""24/9 20:2x: _nucleo_duro_compilato_bf16 == _nucleo_duro_compilato(...).to(bf16), al bit,
in uscita e nei gradienti di z, log_s, delta_raw (scala piccola, CPU)."""
import sys

import torch

torch.set_num_threads(2)
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from quant_ternaria_globale_batch_v2 import _nucleo_duro_compilato, _nucleo_duro_compilato_bf16  # noqa: E402

guasti = 0
for seme in (0, 1):
    g = torch.Generator().manual_seed(seme)
    z0 = torch.randn(4, 16, 8, 128, generator=g)
    ls0 = torch.randn(4, 16, 8, 1, generator=g) * 0.1 - 3
    dr0 = torch.randn(4, 16, 8, 1, generator=g)
    up = torch.randn(4, 16, 8, 128, generator=g).to(torch.bfloat16)
    beta = torch.tensor(16.0)
    ris = []
    for f, cast in ((_nucleo_duro_compilato, True), (_nucleo_duro_compilato_bf16, False)):
        z, ls, dr = (t.clone().requires_grad_(True) for t in (z0, ls0, dr0))
        y = f(z, ls, dr, 2.0, beta)
        y = y.to(torch.bfloat16) if cast else y
        (y.float() * up.float()).sum().backward()
        ris.append((y.detach(), z.grad, ls.grad, dr.grad))
    uguali = [torch.equal(a, b) for a, b in zip(*ris)]
    print(f"seme {seme}: uscita {uguali[0]} · grad z {uguali[1]} · grad log_s {uguali[2]} · grad delta {uguali[3]}")
    if not all(uguali):
        diff = [((a.float() - b.float()).abs().max().item()) for a, b in zip(*ris)]
        print("   differenze massime:", diff)
        guasti += 1
print("OK: identico al bit" if not guasti else f"DIVERSO ({guasti})")
sys.exit(1 if guasti else 0)
