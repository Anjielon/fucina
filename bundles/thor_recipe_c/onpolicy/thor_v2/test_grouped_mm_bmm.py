"""grouped_mm_bmm (un solo bmm) contro il riferimento: stessa uscita, stessi gradienti.

Riferimento = il `_grouped_mm` originale di transformers (su GPU: F.grouped_mm).
Uso:  python test_grouped_mm_bmm.py cpu    (scala piccola, riferimento = ciclo di mm in fp32)
      python test_grouped_mm_bmm.py cuda   (forme VERE di Thor: 256 esperti, gate_up e down,
                                            2 e 8 sequenze da 1024 token x top-8; misura anche i tempi)
Tolleranza bf16: errore relativo (norma) <= 2e-2 sull'uscita e sui gradienti, e il
riferimento stesso contro un calcolo fp32 fa da scala (i due devono stare li' vicino).
"""
import sys
import time

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from grouped_mm_bmm import grouped_mm_bmm  # noqa: E402


def _ciclo_fp32(x, w, offs):
    out = torch.zeros(x.shape[0], w.shape[2], dtype=torch.float32, device=x.device)
    inizio = 0
    for e, fine in enumerate(offs.tolist()):
        if fine > inizio:
            out[inizio:fine] = x[inizio:fine].float() @ w[e].float()
        inizio = fine
    return out


def _caso(dev, E, S, n_in, n_out, trasposto, dtype, riferimento, seme=0):
    g = torch.Generator(device="cpu").manual_seed(seme)
    logit = torch.randn(E, generator=g) * 1.2                     # carichi disuguali, come nel modello vero
    ids = torch.multinomial(torch.softmax(logit, 0), S, replacement=True, generator=g).sort().values
    conteggi = torch.bincount(ids, minlength=E)
    offs = torch.cumsum(conteggi, 0).to(torch.int32).to(dev)
    x = (torch.randn(S, n_in, generator=g) * 0.5).to(dev, dtype).requires_grad_(True)
    if trasposto:   # come down_proj: parametro (E, out, in), passato come vista (E, in, out)
        base = (torch.randn(E, n_out, n_in, generator=g) * 0.02).to(dev, dtype).requires_grad_(True)
        w = base.transpose(-2, -1)
    else:
        base = (torch.randn(E, n_in, n_out, generator=g) * 0.02).to(dev, dtype).requires_grad_(True)
        w = base
    go = torch.randn(S, n_out, generator=g).to(dev, dtype)
    ris = {}
    for nome, fn in (("rif", riferimento), ("bmm", grouped_mm_bmm)):
        x.grad = None
        base.grad = None
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        y = fn(x, w, offs)
        (y.float() * go.float()).sum().backward()
        if dev == "cuda":
            torch.cuda.synchronize()
        ris[nome] = (y.detach().float(), x.grad.float().clone(), base.grad.float().clone(),
                     time.perf_counter() - t0)
    esatto = _ciclo_fp32(x.detach(), w.detach(), offs)
    rel = lambda a, b: ((a - b).norm() / b.norm().clamp_min(1e-30)).item()
    r = ris["rif"]
    b = ris["bmm"]
    return {"C": int(conteggi.max()), "media": S / E,
            "uscita_bmm_vs_rif": rel(b[0], r[0]), "gin": rel(b[1], r[1]), "gw": rel(b[2], r[2]),
            "rif_vs_fp32": rel(r[0], esatto), "bmm_vs_fp32": rel(b[0], esatto),
            "t_rif": r[3], "t_bmm": b[3]}


def main() -> int:
    dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    if dev == "cpu":
        torch.set_num_threads(1)
        casi = [(6, 40, 16, 12, False), (6, 40, 12, 16, True), (5, 3, 8, 8, False)]
        riferimento = lambda x, w, o: _ciclo_fp32(x, w, o).to(x.dtype)
        dtype, tol = torch.float32, 1e-5
    else:
        import transformers.integrations.moe as moe
        riferimento = moe._grouped_mm
        casi = [(256, 2048 * 8, 2048, 1024, False), (256, 2048 * 8, 512, 2048, True),
                (256, 8192 * 8, 2048, 1024, False), (256, 8192 * 8, 512, 2048, True)]
        dtype, tol = torch.bfloat16, 2e-2
    guasti = 0
    for (E, S, n_in, n_out, tr) in casi:
        for _ in range(2 if dev == "cuda" else 1):     # il primo giro scalda i kernel
            m = _caso(dev, E, S, n_in, n_out, tr, dtype, riferimento)
        ok = max(m["uscita_bmm_vs_rif"], m["gin"], m["gw"]) <= tol
        guasti += not ok
        print(f"E={E} S={S} {n_in}->{n_out}{' (trasposto)' if tr else ''}: C={m['C']} (media {m['media']:.0f}) · "
              f"uscita {m['uscita_bmm_vs_rif']:.2e} · grad_in {m['gin']:.2e} · grad_w {m['gw']:.2e} · "
              f"vs fp32: rif {m['rif_vs_fp32']:.2e} bmm {m['bmm_vs_fp32']:.2e} · "
              f"tempo fwd+bwd rif {m['t_rif']*1e3:.1f} ms · bmm {m['t_bmm']*1e3:.1f} ms · {'OK' if ok else 'DIVERSO'}")
    print("OK" if not guasti else f"FALLITO ({guasti})")
    return 1 if guasti else 0


if __name__ == "__main__":
    sys.exit(main())
