"""Prove di e2e_hilo_fuso (2/10 TT6-veloce): indietro fp16x2 con hi e lo nello stesso kernel == due lanci di oggi, AL BIT.
CPU (TRITON_INTERPRET=1): HF1 famiglie vere dello strato 0 di R2B3m (gate, up, down; 4 esperti, instradamento
casuale con esperti vuoti) + un denso a esperto unico (ffn_up_shexp), gradiente con dinamica larga (fino a 2^-30:
parti basse subnormali in fp16) -> dX e dL/dd identici al bit, fuso contro due lanci.
HF2: spento (variabile assente) = nessun lancio fuso; passo V (modo != 0) = via di oggi.
GPU (--gpu): HF3 le stesse famiglie con TUTTI gli esperti e 8192 token sulla GPU (al bit) + tempi;
HF4 strato 0 e 3 veri, avanti + ricalcolo + indietro con e senza E2E_HILO_FUSO: h e dL/dtheta al bit."""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch  # noqa: E402

import e2e_studente as S  # noqa: E402
import e2e_indietro_hilo as HL  # noqa: E402
import e2e_hilo_fuso as HF  # noqa: E402

G = "@@ROOT:models@@/gguf/tony-tern/R2B3m/Thor-T-R2B3m.gguf"


def _confronta(st, L, nomi, T, k, dev, ripeti=0):
    es = st.esperti(L)
    torch.manual_seed(7)
    ric = S.RICETTE["B3_ind2"]
    out = []
    for nome in nomi:
        fam = es.fam[nome]
        d = es.scala(nome).detach()
        if fam.E == 1:
            piano = S.piano_unico(T, dev)
        else:
            idx = torch.randint(0, max(fam.E - 1, 1), (T, k), device=dev)      # l'ultimo esperto resta vuoto
            piano = S.Piano(idx, fam.E)
        Sn = piano.S
        x = (torch.randn(Sn, fam.C, device=dev) * 0.5).to(torch.float16)
        esp = torch.randint(-30, 4, (Sn, fam.R), device=dev).float()
        gy = torch.randn(Sn, fam.R, device=dev) * torch.pow(2.0, esp)
        gy[: max(1, Sn // 50)] = 0.0                                             # righe nulle (scala 1)
        ris = {}
        for acceso in (False, True):
            if acceso:
                os.environ["E2E_HILO_FUSO"] = "1"
            else:
                os.environ.pop("E2E_HILO_FUSO", None)
            if ripeti:                                                           # riscaldamento (compilazione)
                HL.indietro_famiglia(gy, x, d, fam, piano, ric, st.mot, None, True, True)
            n0 = HF.CONTI["fuso"]
            if dev == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(1 + ripeti):
                dx, dd = HL.indietro_famiglia(gy, x, d, fam, piano, ric, st.mot, None, True, True)
            if dev == "cuda":
                torch.cuda.synchronize()
            ris[acceso] = (dx, dd, (time.time() - t0) / (1 + ripeti), HF.CONTI["fuso"] - n0)
        os.environ.pop("E2E_HILO_FUSO", None)
        assert ris[False][3] == 0 and ris[True][3] == 2 * (1 + ripeti), (ris[False][3], ris[True][3])
        ok_x = torch.equal(ris[False][0], ris[True][0])
        ok_d = torch.equal(ris[False][1], ris[True][1])
        rel = float(((ris[False][1] - ris[True][1]).abs().max() / ris[False][1].abs().max()))
        print(f"    dd: max |diff| / max |dd| = {rel:.3e}")
        out.append((nome, fam.E, fam.R, fam.C, fam.gs, ok_x, ok_d, ris[False][2] * 1e3, ris[True][2] * 1e3,
                    int((ris[False][0] != ris[True][0]).sum()), int((ris[False][1] != ris[True][1]).sum())))
        print(f"  {nome:16s} E{fam.E:3d} R{fam.R:5d} C{fam.C:5d} gs{fam.gs} dX {'=' if ok_x else 'DIVERSO'} "
              f"dd {'=' if ok_d else 'DIVERSO'}  oggi {ris[False][2] * 1e3:8.1f} ms  fuso {ris[True][2] * 1e3:8.1f} ms"
              f"  (diversi {out[-1][9]}, {out[-1][10]})", flush=True)
    assert all(r[5] and r[6] for r in out), "fuso diverso da oggi"
    return out


def hf1():
    st = S.Studente(gguf=G, strati=[0], ricetta="B3_ind2", motore_nome="triton2", device="cpu",
                    esperti=slice(0, 4), densi_ternari="shexp")
    _confronta(st, 0, ["gate", "up", "down", "ffn_up_shexp", "ffn_down_shexp"], 40, 2, "cpu")
    print("HF1 ok: fuso == due lanci al bit (CPU interprete, famiglie vere)")


def hf2():
    st = S.Studente(gguf=G, strati=[0], ricetta="B3_ind2", motore_nome="triton2", device="cpu",
                    esperti=slice(0, 4))
    os.environ.pop("E2E_HILO_FUSO", None)
    fam = st.esperti(0).fam["gate"]
    assert not HF.ammesso(st.mot, fam, torch.zeros(1, 1, dtype=torch.float16))
    os.environ["E2E_HILO_FUSO"] = "1"
    try:
        class Finto:
            modo = 1
        st.mot.stato = Finto()
        assert not HF.ammesso(st.mot, fam, torch.zeros(1, 1, dtype=torch.float16))
        del st.mot.stato
        assert HF.ammesso(st.mot, fam, torch.zeros(1, 1, dtype=torch.float16))
    finally:
        os.environ.pop("E2E_HILO_FUSO", None)
    print("HF2 ok: spento e passo V -> via di oggi")


def hf3():
    st = S.Studente(gguf=G, strati=[0], ricetta="B3_ind2", motore_nome="triton2", device="cuda",
                    densi_ternari="shexp")
    _confronta(st, 0, ["gate", "up", "down", "ffn_up_shexp", "ffn_down_shexp"], 8192, 8, "cuda", ripeti=3)
    print("HF3 ok: GPU, 256 esperti, 8192 token: al bit")


def hf4():
    import e2e_allena as A
    A.scegli_gdn()
    st = S.Studente(gguf=G, strati=[0, 3], ricetta="B3_ind2", motore_nome="triton2", device="cuda",
                    densi_ternari="tutte")
    ids = S._ids_reali(2, 4096).cuda()
    ris = {}
    for acceso in (False, True):
        if acceso:
            os.environ["E2E_HILO_FUSO"] = "1"
        else:
            os.environ.pop("E2E_HILO_FUSO", None)
        for p in st.parametri():
            p.grad = None
        h = {}
        st.gradiente(ids, lambda x: (h.setdefault("h", x.detach().clone()), x.float().pow(2).mean() * 1e3)[1])
        torch.cuda.synchronize()
        ris[acceso] = (h["h"], [p.grad.clone() for p in st.parametri()])
    os.environ.pop("E2E_HILO_FUSO", None)
    assert torch.equal(ris[False][0], ris[True][0])
    diversi = [i for i, (a, b) in enumerate(zip(ris[False][1], ris[True][1])) if not torch.equal(a, b)]
    assert not diversi, diversi
    print(f"HF4 ok: strati 0 e 3 veri, h e {len(ris[True][1])} gradienti al bit (lanci fusi {HF.CONTI['fuso']})")


if __name__ == "__main__":
    torch.set_num_threads(4)
    if "--gpu" in sys.argv:
        hf3()
        hf4()
    else:
        hf1()
        hf2()
