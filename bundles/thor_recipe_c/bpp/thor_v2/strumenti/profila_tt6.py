"""2/10 TT6-veloce: profilo per fase e per kernel di un passo (avanti senza grafo + ricalcolo + indietro) su pochi
strati veri di R2B3m, configurazione di TT6 (B3_ind2, triton2, densi tutte, piegati se E2E_DENSI_PIEGATI).
Per i kernel piu' pesanti risale all'operatore CPU che li lancia (con le forme) e alla riga Python.
Uso: python strumenti/profila_tt6.py --strati 0,1,2,3 --righe 2 --uscita FILE.json [--senza-f32-grad]
Solo misura: nessuna scrittura fuori da --uscita."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch  # noqa: E402

import e2e_studente as S  # noqa: E402

G = "@@ROOT:models@@/gguf/tony-tern/R2B3m/Thor-T-R2B3m.gguf"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strati", default="0,1,2,3")
    ap.add_argument("--righe", type=int, default=2)
    ap.add_argument("--T", type=int, default=4096)
    ap.add_argument("--ricetta", default="B3_ind2")
    ap.add_argument("--uscita", required=True)
    ap.add_argument("--ripetizioni", type=int, default=2)
    ap.add_argument("--niente-profilo", action="store_true")
    a = ap.parse_args()
    strati = [int(s) for s in a.strati.split(",")]
    import e2e_allena as A
    A.scegli_gdn()                    # E2E_GDN come la corsa (senza: ripiego di transformers)
    t0 = time.time()
    st = S.Studente(gguf=G, strati=strati, ricetta=a.ricetta, motore_nome=os.environ.get("E2E_MOTORE", "triton2"),
                    device="cuda", densi_ternari="tutte")
    print(f"studente in {time.time() - t0:.1f} s", flush=True)
    ids = S._ids_reali(a.righe, a.T).cuda()
    out = {"strati": strati, "righe": a.righe, "T": a.T, "ricetta": a.ricetta,
           "env": {k: v for k, v in os.environ.items() if k.startswith("E2E_")}}
    out["misura"] = S._misura_passo(st, ids, a.ripetizioni)
    ntok = ids.numel()
    pl = out["misura"]["per_strato_s"]
    out["ms_token_per_strato"] = {L: {k: v * 1e3 / ntok for k, v in d.items()} for L, d in pl.items()}
    print(json.dumps(out["ms_token_per_strato"]), flush=True)
    out["picco_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
    if not a.niente_profilo:
        from torch.profiler import ProfilerActivity, profile
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True,
                     with_stack=True) as prof:
            st.gradiente(ids, lambda h: h.float().pow(2).mean() * 1e3)
            torch.cuda.synchronize()
        for p in st.parametri():
            p.grad = None
        # operatori CPU (aten) col tempo PROPRIO di dispositivo (i kernel che lanciano), forme e pila
        cpu = []
        for ev in prof.key_averages(group_by_input_shape=True, group_by_stack_n=6):
            t = float(getattr(ev, "self_device_time_total", 0) or 0)
            if t <= 0:
                continue
            pila = [x for x in (ev.stack or []) if "thor_v2" in x or "transformers" in x][:4]
            cpu.append((t, ev.key, ev.count, str(ev.input_shapes)[:160], pila))
        cpu.sort(key=lambda r: -r[0])
        tot = sum(r[0] for r in cpu) or 1.0
        out["profilo_ms_tot"] = tot / 1e3
        out["profilo_top"] = [{"ms": t / 1e3, "quota": t / tot, "n": n, "op": k, "forme": f, "pila": p}
                              for t, k, n, f, p in cpu[:60]]
        kern = {}
        for ev in prof.key_averages():
            t = float(getattr(ev, "self_device_time_total", 0) or 0)
            if t > 0 and not ev.key.startswith("aten::"):
                kern[ev.key[:120]] = (t / 1e3, ev.count)
        out["kernel_top"] = sorted(([k, v[0], v[1]] for k, v in kern.items()), key=lambda r: -r[1])[:30]
    Path(a.uscita).write_text(json.dumps(out, indent=1, default=str))
    print("scritto", a.uscita, flush=True)


if __name__ == "__main__":
    main()
