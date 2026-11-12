#!/usr/bin/env python3
"""Re-run, step by step, the recipe of the released Thor-T-35B-A3B-M (lineage A, 12.19 GiB):

    /mnt/usb-ssd/lancio_1109/Thor-T-35B-A3B-M.gguf   sha256 6f56858975fc3bc847f7fcf1a7bab3ac51764e2ec9c26adb82b4e7884bdd93f8
    13,088,098,592 B (MMLU full test 80.06 on the launch engine; see RECIPE_M.md section 1)

Same format and modes as run_recipe_c_d2.py: every step runs the ORIGINAL script with the ORIGINAL arguments (documented
in RECIPE_M.md with the file that proves each link), writes into $FORGIA_OUT/tree/<name>, and is verified against the
recorded output (sha256 for files, per-file manifests in recipe_m/recorded/, declared tolerances for GPU steps).

Differences from run_recipe_c_d2.py (stated in RECIPE_M.md section 3): no mount-namespace virtualisation (the M chain
embeds no input path in its outputs as far as verified: PIENO, splice, pack and publish are checked bitwise); the frozen
code is the live tree, checked against recipe_m/recorded/code_*.manifest (`--dry-run` reports drift).

Modes
  --mode chained   (default) each step consumes the outputs of the re-run steps before it (from scratch).
  --mode anchored  each step consumes the RECORDED inputs (every step verified in isolation; CPU steps can run while
                   the GPU is busy).
Usage
  run_recipe_m.py --dry-run                      plan, resolved paths, recorded hashes and their current status
  run_recipe_m.py --mode anchored --cpu-only     every CPU step against the recorded inputs
  run_recipe_m.py --mode anchored --steps 13,14,15
  FORGIA_GPU=all|0,1  FORGIA_OUT=<dir>  FORGIA_HOUSE_LOCK=0  FORGIA_DISCARD_VERIFIED_GGUF=1  FORGIA_THREADS=8
GPU steps take the house lock (lucchetto.sh) unless FORGIA_HOUSE_LOCK=0; the lock enforces the machine-wide limit of
4 GPU workers.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
AUX = HERE / "recipe_m"
REC = AUX / "recorded"
HASHTOOL = AUX / "hashtool.py"
COMPARE = AUX / "compare_tol.py"

TT = "/mnt/models/gguf/tony-tern"
OC = f"{TT}/OBIETTIVO_C"
FORGE = "/home/angelo/tony-forgia"
NAS = "/mnt/nas/CACHEDEV1_DATA/modelli-fp/obiettivo_c_1022"
BUILD = "/mnt/models/build"
PY = os.environ.get("FORGIA_PY", "/home/angelo/venv-catq/bin/python3")
RAFF = f"{BUILD}/leve/raffina"
RD = f"{BUILD}/wt-r022b/thor_v2/rd_ternary"            # rd_ternary package (unisci, scrivi_piano_gguf)
RDP = f"{TT}/rd_ternary"                                # per-part RD plans of the root phase
CARDAN = f"{OC}/CARDAN_C0B/codice"
ENG = f"{BUILD}/thor-rexppiani2/build-rexp/bin"
ENG_LAUNCH = os.environ.get("FORGIA_LAUNCH_ENGINE", f"{BUILD}/motore-lancio-1107/bin")   # loads the packed release types (46-48)         # thor-rexppiani2 05589d47, llama-perplexity 6ab99a99e34ee172
PACK = f"{BUILD}/thor-pub/tools/ternary/pack_planes.py"  # thor-pub 5d2e382a
GGUFPY_PUB = f"{BUILD}/thor-pub/gguf-py"
GGUFPY_Q8 = f"{BUILD}/thor-pub-q8/gguf-py"
MTPC = f"{TT}/MTP_1101/codice"
SPLICE = f"{TT}/MTP_1101/prova_M_testaXS_1106/splice.py"
BANCO_M = f"{FORGE}/mc_v2/mmlu2000_mescolato.bin"
CAP = 13085779616
NOGPU = "HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES="
THREADS = os.environ.get("FORGIA_THREADS", "8")

# artifact key -> (record path, recorded sha256 or manifest file)
A = {
    "R2B3m": (f"{TT}/R2B3m/Thor-T-R2B3m.gguf", "56cb5bcdd9813a1015a7cad5364b5a80226f990fb8bd561b7cae4727d70efc43"),
    "dkl1": (f"{TT}/dkl_completo/dkl.tsv", "dc9ac5ca0bccd12387c9e1b1b9a4547b69a9447ce959ca5bbade04df9de1a7d2"),
    "dkl2": (f"{TT}/dkl_preciso/dkl.tsv", "5ce2e65374a2dde185e0647574b0203f5cff04d47904971c503226cc717a52ea"),
    "dkl_unito": (f"{TT}/PRECISO/dkl_unito.tsv", "4377cd3c9869f77adf8928d8457a8e13f0170885654969e8716d292273a29ad5"),
    "piano_pv12p": (f"{TT}/PV12P/piano.json", "e5d847cdc4d063cf0fe3c2ca3802be2522d4bfd2235c000bbac7ccb3bc31678d"),
    "PV12P": (f"{TT}/PV12P/Thor-T-pv12p.gguf", "9af6813534f8b004f62e32e95e29234e2e43017cb93c35dbd7da6ce71074f6ce"),
    "piano_pieno": (f"{TT}/PIENO/piano_su_pv12p.json", "5ee8bd38694ddf9056e508adfd84bee3760859a87af2d6f8c08b93aa2b541fe7"),
    "PIENO": (f"{TT}/PIENO/Thor-T-pv12p-pieno.gguf", "edbfbfec2cb2248ceede6927c72e7b6da3bfb8143cd94a7b135b0fe36e807998"),
    "cardan_pt": (f"{TT}/CARDAN_FEDELE/kd/allena/cardan_migliore.pt", None),
    "A": (f"{TT}/CARDAN_FEDELE/kd/Thor-T-PIENO-FED-CARDAN.gguf",
          "8b1e07ae867858466f1866963fe1f076f3d30becd55a7c2cddf2e50136aef621"),
    "T50": (f"{TT}/A_POST_1101/tappa_50/Thor-T-A-POST-T50.gguf",
            "18d379f97757396bd3311b9154ff6628fc75563b317dd3eed5e789ea9ab452bc"),
    "blk40_B": (f"{TT}/MTP_1101/kaggle_c2/risultati/blk40_B", "mtp_blk40_B.manifest"),
    "PACKED": (f"{TT}/MTP_1101/prova_M_testaXS_1106/Thor-T-M-packed-testaXS.gguf",
               "49183e40a2480c56322d5dd999273a143b707854745eefa843d89e20e6d135e3"),
    "M": ("/mnt/usb-ssd/lancio_1109/Thor-T-35B-A3B-M.gguf",
          "6f56858975fc3bc847f7fcf1a7bab3ac51764e2ec9c26adb82b4e7884bdd93f8"),
}
ROOT_INPUTS = [   # (label, path, sha256 | manifest file | None)
    ("bf16 source (HF safetensors), see ../recipe_c_d2/recorded/bf16_source.manifest", f"{TT}/../thor-src-hf", None),
    ("R2B3m (frozen root, RICETTE_ABC_FATTI A.1 #0-7)", A["R2B3m"][0], A["R2B3m"][1]),
    ("RD plan parts 0-2", RDP, "plans_and_dkl.sha256"),
    ("bench MMLU-2000 shuffled (M measurement)", BANCO_M, "benches.sha256"),
    ("engine llama-perplexity (thor-rexppiani2 05589d47)", f"{ENG}/llama-perplexity", "engine.sha256"),
    ("pack_planes.py / gguf-py (thor-pub 5d2e382a, thor-pub-q8)", PACK, "gguf_py_and_pack_tools.manifest"),
    ("A-POST / CARDAN code", f"{OC}/SINTESI_1025/BPP/codice", "code_a_post.manifest"),
    ("MTP code + splice.py", MTPC, "code_mtp_head.manifest"),
]


@dataclass
class Step:
    id: str
    title: str
    kind: str                       # cpu | gpu | gpu-measure | input
    cmd: str = ""                   # bash; {key} -> input path, {out:key} -> output path, {T} -> $FORGIA_OUT/tree
    inputs: list = field(default_factory=list)     # artifact keys consumed
    outputs: list = field(default_factory=list)    # (artifact key, how) how: sha | manifest | none
    determinism: str = "bitwise"    # bitwise | tolerance | measure | recorded
    tolerance: str = ""
    vram_gib: int = 0
    measured: str = ""
    discard: list = field(default_factory=list)    # artifact keys deleted after the next consumer is done (disk)


ENVE = ("E2E_GDN=veloce E2E_GDN_MODULO=gdn_veloce_corsa_v5 E2E_MOTORE=triton2 E2E_CONV_DIRETTA=1 "
        "E2E_SENZA_COESISTENZA=1 E2E_DENSI_PIEGATI=x2 E2E_PIEGA_GEMM=triton E2E_KERNEL_NUOVI=1")
CK = (f"cardan_kd_paper.py --perdita paper --tau 1.0 --lambda-ce 0.5 --topk 256 --init-c a1 --selezione paper "
      "--kl-partenza-motore 0.23700450785334518 --soglia-coerenza 0.03 --strati 0-39 --rango 16 --passi 300 --righe-passo 1 "
      "--heldout-righe 24 --heldout-ogni 50 --pazienza 3 --ore-max 3")
APOST = ("--scale-down-lr 1e-4 --strati 0-39 --rango-esperti 0 --famiglie attn,gdn --rango-densi 16 --lr-densi 1e-5 "
         "--lr-gamma 1e-4 --tau 1.0 --lambda-ce 0.5 --topk 256 --kl4-peso 1 --kl-peso 0 --kl4-n 896 --modo congiunto "
         "--passi 150 --heldout-ogni 50 --pazienza 1000 --selezione kl24_paper --guardia-kl24 0.02 --fuso --riprendi "
         "--kl4-rotazioni 4 --kl4-rotazioni-modo ciclo --allinea-peso 0.3 --allinea-modo rapporto --allinea-metrica cka "
         "--allinea-strati 0-39 --scale-gateup-lr 1e-4 --ricetta B3 --vram-tetto 60 --ore-max 18 --allena-f32 tutte")

DKL_MERGE = r'''env %(NOGPU)s %(PY)s - {dkl1} {dkl2} {out:dkl_unito} <<'Q'
import sys, csv, math, os
a, b, o = sys.argv[1:4]
def leggi(p):
    d = {}
    for r in csv.DictReader(open(p), delimiter='\t'):
        if r['option'] == '+1' and '.E' in r['group']: d[r['group']] = (float(r['delta_kl']), float(r['se']))
    return d
A, B = leggi(a), leggi(b)
with open(o, 'w') as f:
    f.write('group\toption\tdelta_kl\tse\tprimo\tsecondo\n')
    for r0 in csv.DictReader(open(a), delimiter='\t'):
        g, op = r0['group'], r0['option']
        if op == '+1' and g in B:
            da, sa = A[g]; db, sb = B[g]
            f.write(f"{g}\t{op}\t{(da+db)/2:.10e}\t{math.sqrt(sa*sa+sb*sb)/2:.4e}\t{r0['primo']}\t{r0['secondo']}\n")
        else:
            f.write('\t'.join(r0[c] for c in ('group', 'option', 'delta_kl', 'se', 'primo', 'secondo')) + '\n')
Q
''' % {"NOGPU": NOGPU, "PY": PY}

STEPS = [
    Step("01", "dKL of the +1 expert-plane options on R2B3m, 32 rows seed 1 (dkl_completo) and 32 rows seed 2 (dkl_preciso)",
         "gpu", f"cd {RD} && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True E2E_GGUF={{R2B3m}} {ENVE.split(' E2E_DENSI')[0]} "
         f"E2E_CACHE_DEQUANT=/mnt/deposito/e2e/cache_dequant PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/mnt/deposito/e2e/codice/RECUPERO/thor_v2 "
         f"{{GPU_ENV}} {PY} dkl_raccogli.py --thor-v2 /mnt/deposito/e2e/codice/RECUPERO/thor_v2 --teacher-cache "
         f"/mnt/deposito/e2e_teacher_tt6/q8-TT6b --out {{T}}/dkl_preciso --strati 0-39 --blocco 3 --righe-kl 32 --righe-fisher 8 "
         "--gruppi esperto --solo-esperti --opzioni +1 --seme 2",
         inputs=["R2B3m"], outputs=[("dkl2", "none")], determinism="tolerance", vram_gib=80,
         tolerance="seed 2 command from dkl_preciso/comando.sh; the seed-1 run (dkl_completo) has no archived launcher, "
                   "recorded as an input. delta_kl per group: ranking correlation (Spearman) >= 0.9 against the recorded "
                   "table; the ranking, not the digits, is what the plan uses",
         measured="~6 h, gfx1151 (stato/dkl_preciso.sh, 6-7/10)"),
    Step("02", "merge seeds 1+2 into dkl_unito.tsv (mean of the two measurements, se = sqrt(se1^2+se2^2)/2)", "cpu",
         DKL_MERGE, inputs=["dkl1", "dkl2"], outputs=[("dkl_unito", "sha")],
         measured="seconds (snippet of stato/dkl_preciso_dopo.sh)"),
    Step("03", "PV12P plan: RD plan parts 0-2 + dKL, loadable-only, variable-planes engine, se-k 1, cap 13,085,779,616 B",
         "cpu", f"cd {RD} && env {NOGPU} {PY} -m rd_ternary.unisci {RDP}/piano_parte0.json {RDP}/piano_parte1.json "
         f"{RDP}/piano_parte2.json --expect-layers 0-40 --delta-kl {{dkl_unito}} --unmeasured cur --loadable-only "
         f"--motore piani-variabili --gguf {{R2B3m}} --max-file-bytes {CAP} --scale-cur sorgente --se-k 1 --out {{out:piano_pv12p}}",
         inputs=["R2B3m", "dkl_unito"], outputs=[("piano_pv12p", "json")],
         determinism="bitwise", measured="minutes, CPU (stato/pv12_preciso.sh 7/10)"),
    Step("04", "PV12P write in 3 rounds (layers 0-13, 14-27, 28-40), device auto", "gpu",
         f"cd {RD} && for r in 'R2B3m 0-13 g1' 'g1 14-27 g2' 'g2 28-40 PV12P'; do set -- $r; "
         f"RD_TOLL_DIST=0.15 env {{GPU_ENV}} {PY} -m rd_ternary.scrivi_piano_gguf --gguf $([ $1 = R2B3m ] && echo {{R2B3m}} || echo {{T}}/$1.gguf) "
         f"--piano {{piano_pv12p}} --motore piani-variabili --max-bytes {CAP} --device auto --strati $2 --out {{T}}/$3.gguf || exit 1; done; "
         "mv {T}/PV12P.gguf {out:PV12P}; rm -f {T}/g1.gguf {T}/g2.gguf",
         inputs=["R2B3m", "piano_pv12p"], outputs=[("PV12P", "sha")], determinism="tolerance", vram_gib=40,
         tolerance="bitwise expected where the device kernels are the recorded ones (gfx1151); otherwise: same tensor "
                   "names/types/sizes, share of differing packed bytes reported",
         measured="~hours, device auto (PV12P/scrivi_{1,2,3}.log)"),
    Step("05", "PIENO: incremental plan on PV12P (se-k 0.5, +4138 additions, 957 MB) written on CPU", "cpu",
         f"cd {RD} && env {NOGPU} RD_TOLL_DIST=0.15 OMP_NUM_THREADS={THREADS} MKL_NUM_THREADS={THREADS} nice -n 10 {PY} -m "
         f"rd_ternary.scrivi_piano_gguf --gguf {{PV12P}} --piano {{piano_pieno}} --motore piani-variabili --max-bytes {CAP} "
         "--device cpu --out {out:PIENO}",
         inputs=["PV12P", "piano_pieno"], outputs=[("PIENO", "sha")], determinism="bitwise", discard=["PV12P"],
         measured="2838 s CPU, 16 threads (PIENO/scrivi.out 10/10)"),
    Step("06", "CARDAN A.1 base (rank 16) on PIENO", "cpu",
         f"cd {CARDAN} && env {NOGPU} OMP_NUM_THREADS=4 nice -n 19 {PY} cardan_costruisci.py --gguf {{PIENO}} "
         "--out {T}/base_A1_r16_PIENO --strati 0-39 --rango 16 --thread 4",
         inputs=["PIENO"], outputs=[], determinism="bitwise",
         measured="~25 min CPU (CARDAN_FEDELE/base_A1_r16_PIENO); consumed by step 07, no separate hash recorded"),
    Step("07", "A train: faithful CARDAN K=1 on the 80 expert gate/up tensors, rank 16, 300 steps (stopped at 200 by the 3 h cap)",
         "gpu", f"cd {CARDAN} && env {ENVE} E2E_CACHE_DEQUANT={{T}}/cache_dequant {{GPU_ENV}} nice -n 5 {PY} {CK} "
         f"--uscita {{T}}/kd/allena --gguf {{PIENO}} --nome PIENO-FED --teacher {TT}/CARDAN_FEDELE/teacher/q8-maestro-wt103 "
         "--teacher-kl24 /mnt/deposito/e2e_teacher/q8-maestro --base-a1 {T}/base_A1_r16_PIENO",
         inputs=["PIENO"], outputs=[("cardan_pt", "none")], determinism="tolerance", vram_gib=60,
         tolerance="registry kl_held/kl24/paper_held at steps 50..200 within 1 % relative; selection value 1.40967 at "
                   "step 200 (step 0: 1.57358); bitwise not claimed",
         measured="189 min to step 200, gfx1151 (CARDAN_FEDELE/kd/allena)"),
    Step("08", "A fold: CARDAN fold into the TQ1_0R planes of the 80 gate/up tensors (GPU, fp32, batched)", "gpu",
         f"cd {CARDAN} && env PIEGA_ALGEBRA=cpu OMP_NUM_THREADS=6 LEVE_DT=float32 LEVE_THREADS=6 {{GPU_ENV}} nice -n 5 {PY} -X faulthandler "
         f"cardan_piega.py --pt {{cardan_pt}} --gguf {{PIENO}} --out {{T}}/kd/piega --device cuda --lotto 16",
         inputs=["PIENO", "cardan_pt"], outputs=[], determinism="tolerance", vram_gib=24,
         tolerance="fold is a function of cardan_migliore.pt; per-tensor identical on the same device class "
                   "(a_fold_tensors.manifest, 160 tensors, checked after step 09)",
         measured="95 min, gfx1151, 78.5 % of the correction energy captured"),
    Step("09", "A patch: PIENO with the folded tensors = A", "cpu",
         f"env {NOGPU} LEVE_OSPITE={{PIENO}} nice -n 10 {PY} {RAFF}/patch_gguf.py --dir {{T}}/kd/piega/tensori --out {{out:A}} "
         "--base {PIENO}", inputs=["PIENO"], outputs=[("A", "sha")], determinism="bitwise",
         measured="~20 s CPU (CARDAN_FEDELE/kd/piega/patch.out)"),
    Step("10", "A-POST train: joint expert scale + dense rank-16 + F32 tensors, 150 steps, checkpoints at 50/100/150", "gpu",
         f"cd {OC}/SINTESI_1025/BPP/codice && env E2E_GDN=veloce E2E_GDN_MODULO=gdn_veloce_corsa_v5 E2E_MOTORE=triton2 "
         "E2E_CONV_DIRETTA=1 E2E_SENZA_COESISTENZA=1 E2E_CACHE_DEQUANT={T}/cache_dequant_apost E2E_DENSI_PIEGATI=x2 "
         "E2E_PIEGA_GEMM=triton E2E_KQAT=cache_densi E2E_SCALE_FISSE=1 HSA_ENABLE_SDMA=0 TORCHINDUCTOR_COMPILE_THREADS=1 "
         f"{{GPU_ENV}} {PY} cardan_congiunto_scale.py --uscita {{T}}/A_POST_1101/allena --gguf {{A}} --nome A-POST {APOST} "
         f"--tappe-dir {{T}}/A_POST_1101/allena/tappe --kl4-rotazioni-dir {OC}/SINTESI_1025/L2/kl4_rotazioni "
         f"--allinea-cache {OC}/SINTESI_1025/L2/STATI_MAESTRO_wt103_1024tok",
         inputs=["A"], outputs=[], determinism="tolerance", vram_gib=60,
         tolerance="registry (KL24, paper selection) at steps 50/100/150 within 1 % relative; checkpoint tensors relative "
                   "error reported; the .pt of the record are hashed in recorded/a_post_nas_files.sha256 (NAS)",
         measured="~196 s/step, 4.4 min with held-out evals (gfx1151); the record was restarted once (16 steps re-run)"),
    Step("11", "T50 build: export scales/F32 of step 50, fold 130 dense residuals, patch = A-POST step 50", "gpu",
         f"env PROVA=1 APOST_FINO=gguf APOST_TAPPE=50 APOST_CICLI=/dev/null bash {FORGE}/stato/apost_a_tappe_1101.sh",
         inputs=["A"], outputs=[("T50", "sha")], determinism="tolerance", vram_gib=30,
         tolerance="original driver (stato/apost_a_tappe_1101.sh, paths of the record: it writes under A_POST_1101/). Verified "
                   "per tensor against t50_changed_tensors.manifest (422 tensors): `hashtool.py verify-gguf-tensors`. "
                   "Not run by this script in a mirror tree: use it only on the recording machine layout",
         measured="~1 h per tap (A_POST_1101/tappa_50/percorso.txt, route h1)"),
    Step("12", "MTP head blk40_B (20 tensors, trained on C on Kaggle 2xT4, 600 steps): recorded input", "input",
         inputs=["blk40_B"], outputs=[("blk40_B", "manifest")], determinism="recorded",
         tolerance="not re-runnable here (Kaggle launcher not archived); verified by per-file sha256",
         measured="allena_B 3676 s + export 440 s (kaggle_c2 run 2/11)"),
    Step("13", "splice: graft the MTP head onto the UNPACKED T50 (full rewrite, 1580 tensors checked, blk40 20 tensors)", "cpu",
         f"env {NOGPU} {PY} -I {SPLICE} {MTPC} {{T50}} {{blk40_B}} {{T}}/M-testaXS-unpacked.gguf > {{T}}/splice.log 2>&1 && "
         "grep -q 'blk40_uguali.*true' {T}/splice.log && grep -q 'tronco_uguale.*true' {T}/splice.log",
         inputs=["T50", "blk40_B"], outputs=[], determinism="bitwise",
         tolerance="no recorded hash of the unpacked file; checked by the splice self-test (blk40 identical, trunk identical) "
                   "and by the packed result of step 14",
         measured="35.7 s CPU (MTP_1101/prova_S/splice.log layout)"),
    Step("14", "pack: separate-plane TQ1_0R/TQ2_0R -> release types (pack_planes.py --check) = M-packed-testaXS", "cpu",
         f"env {NOGPU} PYTHONPATH={GGUFPY_PUB} nice -n 10 {PY} -I {PACK} {{T}}/M-testaXS-unpacked.gguf {{out:PACKED}} --check "
         "> {T}/pack.log 2>&1 && rm -f {T}/M-testaXS-unpacked.gguf",
         inputs=[], outputs=[("PACKED", "sha")], determinism="bitwise",
         measured="minutes CPU (13,088,098,688 B; the 6/11 creation command was not logged, see RECIPE_M.md section 8)"),
    Step("15", "publish: general.name, remove mtp1101.nota, file_type 36 = the released M", "cpu",
         f"env {NOGPU} G={GGUFPY_Q8} PYTHONPATH={GGUFPY_Q8} {PY} {GGUFPY_Q8}/gguf/scripts/gguf_new_metadata.py --force "
         "--general-name 'Thor-T 35B-A3B M' --remove-metadata mtp1101.nota {PACKED} {out:M} && "
         f"PYTHONPATH={GGUFPY_Q8} {PY} {GGUFPY_Q8}/gguf/scripts/gguf_set_metadata.py --force {{out:M}} general.file_type 36",
         inputs=["PACKED"], outputs=[("M", "sha")], determinism="bitwise", discard=["PACKED"],
         measured="~2.5 min CPU (/mnt/usb-ssd/lancio_1109/M.log)"),
    Step("16", "measure: MMLU-2000 (shuffled bench) of M on the launch engine, Vulkan", "gpu-measure",
         f"env LD_LIBRARY_PATH={ENG_LAUNCH} GGML_VK_TQ1R_Q8=1 GGML_VK_DISABLE_GRAPH_OPTIMIZE=1 {{GPU_ENV}} nice -n 5 {ENG_LAUNCH}/llama-perplexity "
         f"-m {{M}} -ngl 999 -c 4096 -np 4 -ub 512 -dev {{VK_DEV}} --multiple-choice --multiple-choice-tasks 100000 -f {BANCO_M} "
         "> {T}/mmlu_M.log 2>&1",
         inputs=["M"], outputs=[], determinism="measure", vram_gib=24,
         tolerance="MMLU-2000 shuffled bench: T50 81.05 (rexppiani2 Vulkan, recorded mmlu_APOST50.log); the released M adds "
                   "only the MTP head (blk40) and the pack, so the trunk score is expected within |delta| <= 1.0",
         measured="~17 min, gfx1151"),
]
BY_ID = {s.id: s for s in STEPS}


def out_root() -> Path:
    return Path(os.environ.get("FORGIA_OUT", f"{TT}/REPRO_M"))


@dataclass
class Ctx:
    out: Path
    mode: str
    dry: bool
    cpu_only: bool
    produced: set


def tdir(ctx):
    return ctx.out / "tree"


def art_out(ctx, key):
    name = Path(A[key][0]).name
    return str(tdir(ctx) / name)


def resolve_in(ctx, key):
    if ctx.mode == "chained" and key in ctx.produced_keys():
        return art_out(ctx, key)
    return A[key][0]


Ctx.produced_keys = lambda self: {k for s in STEPS if s.id in self.produced for k, _ in s.outputs}


def gpus():
    sel = os.environ.get("FORGIA_GPU", "all")
    if sel != "all":
        return [x for x in sel.split(",") if x]
    topo = Path("/sys/class/kfd/kfd/topology/nodes")
    n = 0
    if topo.exists():
        for node in sorted(topo.iterdir()):
            try:
                props = dict(l.split()[:2] for l in (node / "properties").read_text().splitlines() if l.strip())
                n += int(props.get("simd_count", "0")) > 0
            except (OSError, ValueError):
                pass
    return [str(i) for i in range(max(n, 1))]


def hashtool(*args):
    r = subprocess.run(["nice", "-n", "19", "ionice", "-c3", sys.executable, str(HASHTOOL), *args], capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).strip()


def render(ctx, s, gpu=""):
    c = s.cmd
    for k in A:
        c = c.replace("{out:%s}" % k, art_out(ctx, k))
    for k in A:
        c = c.replace("{%s}" % k, resolve_in(ctx, k))
    return (c.replace("{T}", str(tdir(ctx))).replace("{GPU_ENV}", f"HIP_VISIBLE_DEVICES={gpu} HSA_ENABLE_SDMA=0 "
            "TORCHINDUCTOR_COMPILE_THREADS=1 PYTHONDONTWRITEBYTECODE=1").replace("{VK_DEV}", f"Vulkan{gpu}"))


def wrap(ctx, s, body, gpu):
    head = f"set -o pipefail\nmkdir -p {tdir(ctx)}\nexport PYTHONDONTWRITEBYTECODE=1 TMPDIR={tdir(ctx)}/_tmp\nmkdir -p $TMPDIR\n"
    lock = ""
    if s.kind in ("gpu", "gpu-measure") and os.environ.get("FORGIA_HOUSE_LOCK", "1") == "1" and Path(f"{FORGE}/lucchetto.sh").exists():
        lock = f". {FORGE}/lucchetto.sh; prendi_gpu \"RECIPE-M-{s.id} $$\" {s.vram_gib}; trap molla_gpu EXIT\n"
    return ["bash", "-c", head + lock + body]


def verify(ctx, s):
    res = {}
    for key, how in s.outputs:
        rel = art_out(ctx, key)
        rec = A[key][1]
        if how == "sha":
            rc, msg = hashtool("check", rel, rec)
            res[key] = {"ok": rc == 0, "detail": msg}
            if rc != 0 and rel.endswith(".gguf") and Path(rel).exists() and Path(A[key][0]).exists():
                r = subprocess.run(["nice", "-n", "19", PY, str(AUX / "cmp_gguf_tensors.py"), rel, A[key][0]], capture_output=True, text=True)
                res[key]["detail"] += " | tensor diff: " + " ; ".join(r.stdout.strip().splitlines())[:600]
            if rc == 0 and os.environ.get("FORGIA_DISCARD_VERIFIED_GGUF") == "1" and ctx.mode == "anchored" and key != "M" and rel.endswith(".gguf"):
                Path(rel).unlink()
                res[key]["detail"] += " (discarded after verification)"
        elif how == "json":   # identical content except the "created" timestamp written by the tool
            def canon(q):
                d = json.loads(Path(q).read_text())
                d.pop("created", None)
                return d
            try:
                ok = canon(rel) == canon(A[key][0])
            except OSError:
                ok = False
            res[key] = {"ok": ok, "detail": f"{'MATCH' if ok else 'DIFFER'} {rel} (JSON content equal to the record, 'created' ignored)"}
        elif how.startswith("sha-file:"):
            p = tdir(ctx) / how[9:]
            rc, msg = hashtool("check", str(p), rec)
            res[key] = {"ok": rc == 0, "detail": msg}
        elif how == "manifest":
            rc, msg = hashtool("check", A[key][0], str(REC / rec))
            res[key] = {"ok": rc == 0, "detail": msg}
        else:
            res[key] = {"ok": True, "detail": "no recorded hash; see tolerance"}
    return res


def status(s):
    rows = []
    for key, how in s.outputs:
        p, rec = A[key]
        if how == "manifest":
            rows.append(f"      recorded output {key}: manifest {rec} ({len((REC / rec).read_text().splitlines())} files)")
        elif rec:
            rc, msg = hashtool("check", p, rec)
            rows.append(f"      recorded output {key}: {'MATCH' if rc == 0 else 'DIFFER/MISSING'} {rec[:16]}  ({p})")
        else:
            rows.append(f"      recorded output {key}: no hash ({'present' if Path(p).exists() else 'MISSING'}) {p}")
    return rows


def plan(ctx, sel):
    gl = gpus()
    print(f"# recipe M — mode {ctx.mode}, out {ctx.out}, GPUs {gl}")
    print("# root inputs")
    for label, path, spec in ROOT_INPUTS:
        if spec is None:
            st = "present" if Path(path).exists() else "MISSING"
        elif spec.endswith((".manifest", ".sha256")):
            want = (REC / spec).read_text().splitlines()
            st = f"manifest {spec} ({len(want)} rows); " + ("path present" if Path(path).exists() else "path MISSING")
            bad = [l for l in want if len(l) > 66 and l[64:66] == "  " and l[66:].startswith("/") and Path(l[66:].split()[0]).is_file()
                   and False]
        else:
            rc, st = hashtool("check", path, spec)
        print(f"  [{label}] {path}\n      {st}")
    for s in sel:
        g = gl[int(s.id) % len(gl)] if s.kind in ("gpu", "gpu-measure") else ""
        print(f"\n## step {s.id} [{s.kind}, {s.determinism}{', VRAM ' + str(s.vram_gib) + ' GiB' if s.vram_gib else ''}] {s.title}")
        print(f"   recorded: {s.measured}")
        for k in s.inputs:
            src = "re-run" if ctx.mode == "chained" and any(k in [o for o, _ in t.outputs] and t.id < s.id for t in STEPS) else "record (read-only)"
            print(f"   input  {k}: {A[k][0]}  [{src}]  {'present' if Path(A[k][0]).exists() else 'MISSING'}")
        for r in status(s):
            print(r)
        if s.tolerance:
            print(f"   tolerance: {s.tolerance}")
        if s.cmd:
            print("   command:")
            for line in render(ctx, s, g).splitlines():
                print(f"     {line[:400]}")


def run_step(ctx, s):
    state = ctx.out / "state"
    state.mkdir(parents=True, exist_ok=True)
    done = state / f"{s.id}.json"
    if done.exists() and json.loads(done.read_text()).get("ok"):
        print(f"[{s.id}] already verified: skip")
        ctx.produced.add(s.id)
        return True
    if s.kind in ("gpu", "gpu-measure") and ctx.cpu_only:
        print(f"[{s.id}] GPU step skipped (--cpu-only)")
        return True
    t0 = time.time()
    if s.kind == "input":
        rc = 0
    else:
        if s.id == "11" and ctx.mode != "anchored":
            pass
        log = ctx.out / "logs" / f"{s.id}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        gl = gpus()
        g = gl[int(s.id) % len(gl)] if s.kind in ("gpu", "gpu-measure") else ""
        with open(log, "a") as f:
            rc = subprocess.run(wrap(ctx, s, render(ctx, s, g), g), stdout=f, stderr=subprocess.STDOUT).returncode
    res = verify(ctx, s) if rc == 0 else {}
    ok = rc == 0 and all(v["ok"] for v in res.values())
    rec = {"step": s.id, "ok": ok, "rc": rc, "minutes": round((time.time() - t0) / 60, 1), "mode": ctx.mode,
           "determinism": s.determinism, "verification": res, "time": time.strftime("%FT%T")}
    (done if ok else state / f"{s.id}.failed.json").write_text(json.dumps(rec, indent=1))
    print(f"[{s.id}] rc {rc} in {rec['minutes']} min: " + "; ".join(f"{k}: {v['detail'][:200]}" for k, v in res.items()))
    if ok:
        ctx.produced.add(s.id)
        for t in STEPS:   # free disk: delete a discardable input once its consumer step is verified
            if t.id == s.id:
                for k in s.discard:
                    p = Path(art_out(ctx, k))
                    if ctx.mode == "chained" and p.exists():
                        p.unlink()
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--mode", choices=("chained", "anchored"), default="chained")
    ap.add_argument("--steps", default="")
    ap.add_argument("--from", dest="frm", default="01")
    ap.add_argument("--to", default="16")
    ap.add_argument("--cpu-only", action="store_true")
    a = ap.parse_args()
    ctx = Ctx(out_root(), a.mode, a.dry_run, a.cpu_only, set())
    ids = a.steps.split(",") if a.steps else [s.id for s in STEPS if a.frm <= s.id <= a.to]
    sel = [BY_ID[i] for i in ids]
    if a.dry_run:
        plan(ctx, sel)
        return 0
    ctx.out.mkdir(parents=True, exist_ok=True)
    if a.mode == "chained" and (ctx.out / "state").exists():
        ctx.produced |= {p.stem for p in (ctx.out / "state").glob("*.json") if not p.stem.endswith("failed")
                         and json.loads(p.read_text()).get("ok")}
    for s in sel:
        if not run_step(ctx, s):
            print(f"step {s.id} failed: see {ctx.out}/logs/{s.id}.log")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
