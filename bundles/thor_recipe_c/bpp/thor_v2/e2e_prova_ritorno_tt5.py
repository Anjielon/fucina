#!/usr/bin/env python3
"""Prova di ANDATA E RITORNO del codice TT5 sul GGUF vero (30/9/2026, ramo tt5-terzo-piano). SOLO CPU.

Lo studente TT5 costruito da R2B3m (tutti i 40 strati, uno alla volta per stare in memoria; densi ternari TUTTI coi
secondi e terzi piani; F32 allenabili come in TT5) al passo 0 (theta = 0, F32 = valori del file) scrive un'istantanea
nello STESSO formato di `e2e_allena.salva_istantanea` (scale.npz, theta.pt, f32.npz, meta.json), poi l'export VERO
(`e2e_esporta.py` da riga di comando, come catena_tt5.sh, con la ricarica degli strati 0, 3, 39). Criteri scritti prima:
  (1) il GGUF esportato e' IDENTICO al byte a R2B3m (sha256): nessuna scala ne' F32 cambia al passo 0;
  (2) la ricarica degli strati 0, 3, 39 dal file esportato da' scale e uscita AL BIT (controllo interno di e2e_esporta);
  (3) i logit del grafo principale (dump_logit.p3d CPU, testo_regressione.txt, 204 token) del file esportato sono
      identici al bit a r2b3_p3.bin (R2B3, motore 0ed1faa) — confronta_logit.py.
Uso: nice -n 19 env OMP_NUM_THREADS=4 python3 e2e_prova_ritorno_tt5.py [--uscita DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("TRITON_INTERPRET", "1")
for _v in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
    os.environ[_v] = ""
import numpy as np  # noqa: E402
import torch  # noqa: E402

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import e2e_studente as S  # noqa: E402
import e2e_allena as A  # noqa: E402

R2B3M = Path("@@ROOT:models@@/gguf/tony-tern/R2B3m/Thor-T-R2B3m.gguf")
SHA_R2B3M = "56cb5bcdd9813a1015a7cad5364b5a80226f990fb8bd561b7cae4727d70efc43"
F32 = ("attn_norm,post_attention_norm,attn_q_norm,attn_k_norm,ssm_norm,ffn_gate_inp,ffn_gate_inp_shexp,ssm_a,ssm_alpha,"
       "ssm_beta,ssm_conv1d,ssm_dt,output_norm")
REG = Path("@@ROOT:models@@/gguf/tony-tern/R1/regressione")
DUMP = Path("@@ROOT:models@@/build/llamacpp-terzo-densi/build-cpu/dump_logit.p3d")
CONFRONTA = Path("@@ROOT:forge@@/thor_v2/matematica/secondo_piano_bilancio/confronta_logit.py")
TXT = Path("@@ROOT:forge@@/thor_v2/matematica/secondo_piano_bilancio/testo_regressione.txt")
torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "4")))


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for pezzo in iter(lambda: f.read(1 << 24), b""):
            h.update(pezzo)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uscita", default="@@ROOT:depot@@/e2e/prove_tt5/ritorno")
    a = ap.parse_args()
    d = Path(a.uscita)
    corsa = d / "corsa"
    ist = corsa / "migliore_000000"
    ist.mkdir(parents=True, exist_ok=True)
    esito = {"data": time.strftime("%Y-%m-%d %H:%M:%S"), "codice": str(QUI)}
    t0 = time.time()
    if sha(R2B3M) != SHA_R2B3M:
        raise SystemExit("sha di R2B3m diverso")
    scale, theta, f32, conta = {}, {}, {}, {"densi": 0, "p2": 0, "p3": 0}
    for L in range(40):
        st = S.Studente(gguf=R2B3M, strati=[L], ricetta="B3_ind32", motore_nome="torch", device="cpu",
                        log=lambda *_: None, densi_ternari=tuple(S.FAMIGLIE_DENSE), allena_f32=F32)
        scale.update(st.scale_per_export())
        theta.update({A._chiave(k): p.detach().cpu() for k, p in zip(st.chiavi(), st.parametri())})
        if A._parametri_f32(st):
            f32.update(st.f32_per_export())
        fam = st.esperti(L).fam
        conta["densi"] += sum(n in S.FAMIGLIE_DENSE for n in fam)
        conta["p2"] += sum(n in S.FAMIGLIE_DENSE2 for n in fam)
        conta["p3"] += sum(n in S.FAMIGLIE_DENSE3 for n in fam)
        del st
        print(f"strato {L}: {len(scale)} tensori con scale, {conta}", flush=True)
    # output_norm (F32 globale) e' dello studente intero: con strati=[L] c'e' comunque (F32_GLOBALI)
    esito["famiglie"] = conta
    esito["tensori_scale"] = len(scale)
    esito["tensori_f32"] = len(f32)
    esito["theta_tutti_zero"] = all(float(v.abs().max()) == 0.0 for v in theta.values())
    np.savez(ist / "scale.npz", **{n: np.ascontiguousarray(v).view(np.uint16) for n, v in scale.items()})
    torch.save(theta, ist / "theta.pt")
    meta = {"passo": 0, "valutazione": {"prova": "ritorno_tt5"}, "scritto": A.adesso(),
            "sha256_scale": A.sha256_file(ist / "scale.npz"), "sha256_theta": A.sha256_file(ist / "theta.pt"),
            "tensori": sorted(scale)}
    if f32:
        np.savez(ist / "f32.npz", **f32)
        meta["sha256_f32"] = A.sha256_file(ist / "f32.npz")
        meta["tensori_f32"] = sorted(f32)
    (ist / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    esito["secondi_istantanea"] = round(time.time() - t0, 1)
    out = d / "Thor-T-R2B3m-ritorno.gguf"
    for p in (out, Path(str(out) + ".provenienza.json")):
        p.unlink(missing_ok=True)
    r = subprocess.run([sys.executable, str(QUI / "e2e_esporta.py"), "--sorgente", str(R2B3M), "--corsa", str(corsa),
                        "--istantanea", ist.name, "--uscita", str(out), "--device", "cpu"],
                       capture_output=True, text=True, cwd=str(QUI))
    (d / "export.log").write_text(r.stdout + r.stderr)
    esito["export_rc"] = r.returncode
    if r.returncode == 0:
        prov = json.loads(Path(str(out) + ".provenienza.json").read_text())
        esito["ricarica"] = prov["ricarica"]
        esito["scale_cambiate"] = sum(prov["scale_cambiate_per_tensore"].values())
        esito["f32_cambiati"] = sum(prov.get("tensori_f32", {}).get("valori_cambiati_per_tensore", {}).values())
        esito["sha_esportato"] = prov["esportato"]["sha256"]
        esito["crit1_identico_al_byte"] = prov["esportato"]["sha256"] == SHA_R2B3M
        esito["crit2_ricarica_al_bit"] = all(v.get("scale_al_bit") and v.get("uscita_al_bit")
                                             for v in prov["ricarica"].values()) and len(prov["ricarica"]) == 3
        b = d / "ritorno_p3.bin"
        env = dict(os.environ, OMP_NUM_THREADS=os.getenv("OMP_NUM_THREADS", "4"))
        with open(d / "dump.log", "w") as f:
            rd = subprocess.run([str(DUMP), str(out), str(TXT), str(b), os.getenv("OMP_NUM_THREADS", "4")], stdout=f,
                                stderr=subprocess.STDOUT, env=env)
        esito["dump_rc"] = rd.returncode
        if rd.returncode == 0:
            rc = subprocess.run([sys.executable, str(CONFRONTA), str(REG / "r2b3_p3.bin"), str(b)], capture_output=True,
                                text=True)
            c = json.loads(rc.stdout.strip().splitlines()[-1])
            esito["confronto_logit"] = c
            esito["crit3_logit_identici"] = bool(c.get("identici"))
    esito["secondi"] = round(time.time() - t0, 1)
    esito["TUTTO_VERDE"] = all(esito.get(k) for k in ("crit1_identico_al_byte", "crit2_ricarica_al_bit",
                                                       "crit3_logit_identici")) and esito["theta_tutti_zero"] \
        and conta["p3"] == 250 and conta["p2"] == 250 and conta["densi"] == 250
    (d / "esito_ritorno_tt5.json").write_text(json.dumps(esito, indent=1, default=str))
    print(json.dumps(esito, default=str)[:3000], flush=True)
    return 0 if esito["TUTTO_VERDE"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
