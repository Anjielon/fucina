"""2/10 TT6-veloce: censimento delle COPIE (aten copy_/_to_copy/clone/contiguous/index) di avanti senza grafo +
ricalcolo + indietro di pochi strati veri, con la riga Python che le lancia, la contiguita' della sorgente e i byte.
Gira su CPU (motore torch, GDN veloce v3): le operazioni aten chiamate dal codice Python sono le stesse della GPU,
quindi dice QUALI copie esistono e quanto sono grandi; quanto costano sulla GPU lo dice il profilo GPU.
Uso: python strumenti/censisci_copie.py --strati 0,3 --T 512 --uscita FILE.json"""
import argparse
import json
import os
import sys
import traceback
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch  # noqa: E402
from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402

G = "@@ROOT:models@@/gguf/tony-tern/R2B3m/Thor-T-R2B3m.gguf"
COPIE = {"aten.copy_.default", "aten._to_copy.default", "aten.clone.default", "aten.index.Tensor",
         "aten.cat.default", "aten.stack.default", "aten.contiguous.default", "aten._unsafe_view.default",
         "aten.expand.default", "aten.index_put_.default", "aten.scatter_add.default", "aten.gather.default",
         "aten.constant_pad_nd.default", "aten.slice_scatter.default", "aten.new_zeros.default",
         "aten.zeros_like.default", "aten.fill_.Scalar", "aten.zero_.default", "aten.index_add.default",
         "aten.scatter_reduce.two"}


class Censo(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.righe = defaultdict(lambda: [0, 0, 0, ""])   # (op, riga) -> [chiamate, byte, non_contigui, esempio]
        self.fase = "?"

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        nome = str(func)
        if nome in COPIE:
            srg = next((a for a in args if isinstance(a, torch.Tensor)), None)
            o = out[0] if isinstance(out, (tuple, list)) else out
            nb = (o.numel() * o.element_size()) if isinstance(o, torch.Tensor) else 0
            nc = int(srg is not None and not srg.is_contiguous())
            pila = [f for f in traceback.extract_stack()[:-1] if "thor_v2" in f.filename or "transformers" in
                    f.filename]
            riga = " < ".join(f"{Path(f.filename).name}:{f.lineno}:{f.name}" for f in reversed(pila[-3:]))
            k = (self.fase, nome, riga)
            v = self.righe[k]
            v[0] += 1
            v[1] += nb
            v[2] += nc
            if not v[3] and srg is not None:
                v[3] = f"src {tuple(srg.shape)} str {srg.stride()} {srg.dtype} -> {tuple(o.shape) if isinstance(o, torch.Tensor) else ''}"
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strati", default="0,3")
    ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--ricetta", default="B3_ind2")
    ap.add_argument("--motore", default="torch")
    ap.add_argument("--uscita", required=True)
    a = ap.parse_args()
    torch.set_num_threads(4)
    import e2e_studente as S
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import importlib
    gm = importlib.import_module(os.environ.get("E2E_GDN_MODULO", "gdn_veloce_corsa_v5"))
    gm.installa()
    strati = [int(s) for s in a.strati.split(",")]
    st = S.Studente(gguf=G, strati=strati, ricetta=a.ricetta, motore_nome=a.motore, device="cpu",
                    densi_ternari="tutte")
    ids = S._ids_reali(1, a.T)
    c = Censo()
    with c:
        c.fase = "avanti"
        stato = st.avanti(ids)
        xL = stato["xL"].detach().requires_grad_(True)
        with torch.enable_grad():
            (st.esce(xL).float().pow(2).mean() * 1e3).backward()
        g = xL.grad
        for i in reversed(range(len(st.strati))):
            L = st.strati[i]
            xin = stato["ingressi"][i].detach().requires_grad_(True)
            c.fase = f"ricalcolo+indietro L{L} ({st.cfg.layer_types[L]})"
            with torch.enable_grad():
                y = st._strato(L, xin, stato["ctx"], instr=stato["instr"][i])
                y.backward(g)
            g = xin.grad
    righe = sorted(([k[0], k[1], k[2], v[0], v[1], v[2], v[3]] for k, v in c.righe.items()), key=lambda r: -r[4])
    tot = sum(r[4] for r in righe)
    out = {"T": a.T, "strati": strati, "byte_totali": tot,
           "righe": [{"fase": r[0], "op": r[1], "dove": r[2], "chiamate": r[3], "MB": r[4] / 2 ** 20,
                      "byte_per_token": r[4] / a.T, "non_contigue": r[5], "esempio": r[6]} for r in righe]}
    Path(a.uscita).write_text(json.dumps(out, indent=1))
    for r in out["righe"][:45]:
        print(f"{r['MB']:9.1f}MB {r['chiamate']:4d} nc{r['non_contigue']:3d} {r['fase'][:28]:28s} {r['op'][5:25]:20s} "
              f"{r['dove'][:120]} | {r['esempio'][:110]}")


if __name__ == "__main__":
    main()
