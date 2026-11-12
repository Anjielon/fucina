#!/usr/bin/env python3
"""Regenerates recorded/ (hashes and manifests of the M recipe) from the files of the record. Read-only on the record.
Run on the recording machine; sizes are small except the GGUF/.pt hashes (cached by hashtool, nice 19)."""
import glob, hashlib, json, os, re, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REC = HERE / "recorded"
REC.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE))
import hashtool as H  # noqa: E402

TT = "/mnt/models/gguf/tony-tern"
OC = f"{TT}/OBIETTIVO_C"
NAS = "/mnt/nas/CACHEDEV1_DATA/modelli-fp/obiettivo_c_1022"
FORGE = "/home/angelo/tony-forgia"
cache = H._load_cache()


def sha(p):
    return H.sha_file(Path(p), cache)


def write(name, rows):
    (REC / name).write_text("\n".join(rows) + "\n")
    print(f"{name}: {len(rows)} rows")


def files_manifest(name, paths):
    write(name, [f"{sha(p)}  {p}" for p in sorted(set(paths)) if os.path.isfile(p)])


def pyfiles(d, pats=("*.py", "*.sh", "*.cpp", "*.patch")):
    out = []
    for pat in pats:
        out += [f for f in glob.glob(f"{d}/{pat}") if ".bak" not in f and ".prima" not in f]
    return out


# 1. code trees used by the M chain (frozen files, no *.bak)
files_manifest("code_a_post.manifest",
               pyfiles(f"{OC}/SINTESI_1025/BPP/codice") + [f"{OC}/SINTESI_1025/L3/codice/scale_gateup_esporta.py",
                                                            f"{OC}/SINTESI_1025/F32/codice/f32_esporta.py"] +
               [f"{FORGE}/stato/{n}" for n in ("apost_a_1101b.sh", "apost_a_tappe_1101.sh", "apost_nas_1101.sh",
                                              "bpost_scale_down_fp32_1026.py", "tappa_a_delta_1018.py",
                                              "bpost_verifica_1026.py", "parziale_mmlu.py", "pieno_scrivi_1010.sh",
                                              "pieno_1010.sh", "cardan_fedele_1010.sh", "m_testa_xs_pub_1106.sh",
                                              "gguf_pubblicazione_1109.sh")])
files_manifest("code_mtp_head.manifest", pyfiles(f"{TT}/MTP_1101/codice") + [f"{TT}/MTP_1101/prova_M_testaXS_1106/splice.py"])
files_manifest("code_pieno_cardan.manifest",
               pyfiles(f"{TT}/CARDAN/codice", ("cardan_kd*.py", "cardan_piega*.py", "cardan_costruisci*.py")) +
               [f"{TT}/CARDAN/cardan_ospite.sh", f"{TT}/CARDAN/cardan_corsa.sh", "/mnt/models/build/leve/raffina/patch_gguf.py"] +
               pyfiles("/mnt/models/build/wt-r022b/thor_v2/rd_ternary", ("*.py",)))
tool = ["/mnt/models/build/thor-pub/tools/ternary/pack_planes.py", f"{FORGE}/pubblicazioni/LANCIO_FINALE_1102/strumenti/pulisci_intestazione.py"]
for g in ("thor-pub", "thor-pub-q8", "thor-rexppiani2"):
    tool += glob.glob(f"/mnt/models/build/{g}/gguf-py/gguf/*.py") + glob.glob(f"/mnt/models/build/{g}/gguf-py/gguf/scripts/*.py")
files_manifest("gguf_py_and_pack_tools.manifest", tool)

# 2. engines
eng = []
for d in ("/mnt/models/build/thor-rexppiani2/build-rexp/bin", "/mnt/models/build/motore-lancio-1107/bin"):
    for n in ("llama-perplexity", "libggml-vulkan.so", "libllama.so"):
        if os.path.exists(f"{d}/{n}"):
            eng.append(f"{d}/{n}")
files_manifest("engine.sha256", eng)

# 3. benches and teacher
files_manifest("benches.sha256", [f"{FORGE}/mc_v2/mmlu_00000.bin", f"{FORGE}/mc_v2/mmlu2000_mescolato.bin",
                                  f"{FORGE}/mc_v2/mmlu2000_mescolato.indice.tsv"])

# 4. chain GGUFs measured today
rows = []
for p in (f"{TT}/R2B3m/Thor-T-R2B3m.gguf", f"{TT}/PIENO/Thor-T-pv12p-pieno.gguf",
          f"{TT}/CARDAN_FEDELE/kd/Thor-T-PIENO-FED-CARDAN.gguf", f"{TT}/A_POST_1101/tappa_50/Thor-T-A-POST-T50.gguf",
          f"{TT}/MTP_1101/prova_M_testaXS_1106/Thor-T-M-packed-testaXS.gguf", "/mnt/usb-ssd/lancio_1109/Thor-T-35B-A3B-M.gguf"):
    rows.append(f"{sha(p)}  {p}  {os.path.getsize(p)}")
write("gguf_chain_measured_1112.sha256", rows)

# 5. tensors changed by A -> T50 (patch_gguf provenance: per-tensor sha256 of the bytes)
for name, p in (("t50_changed_tensors.manifest", f"{TT}/A_POST_1101/tappa_50/Thor-T-A-POST-T50.gguf.provenienza.json"),
                ("a_fold_tensors.manifest", f"{TT}/CARDAN_FEDELE/kd/Thor-T-PIENO-FED-CARDAN.gguf.provenienza.json")):
    t = json.load(open(p))["tensori"]
    write(name, [f"{e['sha256_byte']}  {n}" for n, e in sorted(t.items())])

# 6. MTP head (blk40_B): per-file hashes as written by the Kaggle run + our own
kb = f"{TT}/MTP_1101/kaggle_c2/risultati/blk40_B"
write("mtp_blk40_B.manifest", H.manifest(Path(kb), cache))

# 7. A-POST checkpoints of step 50/100/150 on the NAS (sha256 files written by the record) + verification today
rows = []
for f in sorted(glob.glob(f"{NAS}/A_POST_1101/tappe/*.pt")):
    rows.append(f"{sha(f)}  tappe/{os.path.basename(f)}")
for f in sorted(glob.glob(f"{NAS}/A_POST_1101/allena/*.pt")):
    rows.append(f"{sha(f)}  allena/{os.path.basename(f)}")
write("a_post_nas_files.sha256", rows)

# 8. recorded MMLU logs: score and per-question digest
rows = []
for tag, p in (("PIENO", f"{TT}/PIENO/mmlu_int8.log"), ("A", f"{TT}/CARDAN_FEDELE/kd/mmlu_CARDAN.log"),
               ("A-base-shuffled-bench", f"{TT}/A_POST_1101/base/mmlu_A_base.log"),
               ("A-POST-T50", f"{TT}/A_POST_1101/tappa_50/mmlu_APOST50.log")):
    t = open(p, errors="ignore").read()
    m = re.findall(r"Final result: ([0-9.]+)", t)
    n = t.count("\nMCLP\t") + (1 if t.startswith("MCLP\t") else 0)
    rows.append(f"{tag}\t{m[-1] if m else 'n/a'}\t{n}\t{sha(p)}\t{p}")
write("mmlu_logs_recorded.tsv", rows)
H._save_cache(cache)
