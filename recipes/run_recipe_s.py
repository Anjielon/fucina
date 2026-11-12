#!/usr/bin/env python3
"""Reproduce (verify, or re-run) the released Thor-T-35B-A3B-S model, lineage B.

    release file Thor-T-35B-A3B-S.gguf   sha256 9d5afbf85afd0613dc67940c8c56e774790cc580f919a716445bd479cc6d65f9
    = pack_planes(B-POST-T100) + metadata clean-up,  B-POST-T100 sha256 60e2d7da44e96b56...

The chain, every step with its input -> output sha256, command and determinism class, is in RECIPE_S.md.
Recorded hashes and manifests: recipe_s/recorded/.  Same format and helpers as run_recipe_c_d2.py (hashtool.py).

Modes
  --dry-run   print the plan: steps, resolved commands, recorded hashes, which artifacts exist on this machine.
  --verify    (1) re-hash every recorded artifact that exists locally (GGUFs, tools, tensor directories, checkpoints);
              (2) re-run the deterministic CPU steps against the recorded inputs and compare sha256:
                  patch SEL_pred_CARDAN, patch CARDAN fold, pack planes, clean header (--hashes-only skips (2);
                  --deep also compares every tensor of the local B-POST-T100 with recorded/t100_tensors.manifest).
              No GPU is used. Outputs go to $FORGIA_OUT/verify and are deleted once they match (--keep to retain).
  --run       re-run the chain (--steps / --from / --to). CPU steps run directly; GPU steps are the ORIGINAL launchers,
              which take the house GPU lock (lucchetto.sh) themselves; the runner is strictly serial (1 GPU worker).
              Every output path of a launcher is rewritten into $FORGIA_OUT/tree so the record is never written.
Exit status 0 only if every executed check matched.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
AUX = HERE / "recipe_c_d2"          # hashtool.py is shared with recipe C
REC = HERE / "recipe_s" / "recorded"
sys.path.insert(0, str(AUX))
import hashtool  # noqa: E402

TT = os.environ.get("FORGIA_TT", "/mnt/models/gguf/tony-tern")
FORGE = os.environ.get("FORGIA_ROOT", "/home/angelo/tony-forgia")
NAS = os.environ.get("FORGIA_NAS", "/mnt/nas/CACHEDEV1_DATA/modelli-fp")
PY = os.environ.get("FORGIA_PY", "/home/angelo/venv-catq/bin/python3")
RAFF = os.environ.get("FORGIA_RAFFINA", "/mnt/models/build/leve/raffina")
PUB = os.environ.get("FORGIA_PUB_BUILD", "/mnt/models/build/thor-pub-q8")
PATCH = f"{RAFF}/patch_gguf.py"
PACK = f"{PUB}/tools/ternary/pack_planes.py"
GGUFPY = f"{PUB}/gguf-py"
CLEAN = f"{FORGE}/pubblicazioni/LANCIO_FINALE_1102/strumenti/pulisci_intestazione.py"
LOCK = f"{FORGE}/lucchetto.sh"
OUT = Path(os.environ.get("FORGIA_OUT", "/mnt/models/gguf/tony-tern/REPRO_S"))   # 12/11: never the home disk (a re-run writes ~27 GB)
S_NAME = "Thor-T 35B-A3B S"


def sha_of(path: str | Path) -> str:
    return hashtool.sha_file(Path(path))


# recorded GGUFs: key -> (path on the record machine, sha256 | None when the file no longer exists, size)
G = {
    "R2B3m": (f"{TT}/R2B3m/Thor-T-R2B3m.gguf", "56cb5bcdd9813a1015a7cad5364b5a80226f990fb8bd561b7cae4727d70efc43", 8846968608),
    "eb882": (None, "247395e18124889dcd2147f2c0af0edbd231e8e1c91426a28cd4b2aeff7397b7", 9470184864),
    "eb882K": (f"{TT}/TUTTE/eb882K.gguf", "099f1ececa3381aaf808af303d767d0b09839c70c96100a98d67241b6ff63262", 9470184864),
    "CARDAN": (f"{TT}/CARDAN/kd_r16_eb882K/Thor-T-eb882K-CARDAN.gguf",
               "dae59b0f2c2dfe9092e64608c122193dbd3c86bfb5f1c1b2fc980606e6015142", 9470184864),
    "SEL_pred": (None, "6bdceb310b5113f2dc160a2b2d50b541f6ab9449a894ae83ddcd06939f1390c3", 9470184864),
    "B": (f"{TT}/BANCO_1031/Thor-T-eb882K-SEL_pred_CARDAN.gguf",
          "4e4f94087067858f0437de7cfc6d12fc7a9657563cfa83ab40f840734dc9c867", 9470184864),
    "T100": (f"{TT}/B_POST_1101/tappa_100/Thor-T-B-POST-T100.gguf",
             "60e2d7da44e96b5627ea80990e9a681a9f49fb35e98ef32db82285cb9bcfe69b", 9470184864),
    "packed": (None, "39a0f0a3caae71d0910bf063f58eac5e412212d94b69e111e66d954eaf7be6d9", 9470145184),
    "S": (None, "9d5afbf85afd0613dc67940c8c56e774790cc580f919a716445bd479cc6d65f9", 9470119968),
}
# alternative locations (same bytes) tried when the primary path is missing
ALT = {
    "eb882K": [f"{NAS}/archivio_thor_2026-10/R029/eb882_K1.045.gguf"],
    "T100": [f"{NAS}/obiettivo_c_1022/B_POST_1101/tappa_100/Thor-T-B-POST-T100.gguf"],
    "B": [f"{TT}/COMBINAZIONE_eb882K/SEL_pred_CARDAN/Thor-T-eb882K-SEL_pred_CARDAN.gguf",
          f"{NAS}/thor-versioni/Thor-T-eb882K-SEL_pred_CARDAN.gguf"],
}
TOOLS = REC / "tools.sha256"
DIRS = {   # tensor directories (inputs of the CPU patch steps): path, recorded manifest
    "sel_pred_cardan_tensori": (f"{TT}/COMBINAZIONE_eb882K/SEL_pred_CARDAN/tensori", "sel_pred_cardan_tensori.manifest"),
    "cardan_fold_tensori": (f"{TT}/CARDAN/kd_r16_eb882K/piega/tensori", "cardan_fold_tensori.manifest"),
    "sel_pred_tensori": (f"{TT}/COMBINAZIONE_eb882K/SEL_pred/tensori", "sel_pred_tensori.manifest"),
    "cardan_base_a1": (f"{TT}/CARDAN/base_A1_r16_eb882K", "cardan_base_a1_eb882K.manifest"),
}
CKPT100 = f"{NAS}/obiettivo_c_1022/B_POST_1101/tappe_100"   # step-100 training state, sha256 in recorded/

# output roots that launchers write to: rewritten into $OUT/tree/ so that the record is never touched
MIRRORED = [
    f"{TT}/B_POST_1101", f"{TT}/CARDAN/kd_r16_eb882K", f"{TT}/CARDAN/cache_dequant", f"{TT}/CARDAN/_tmp",
    f"{TT}/COMBINAZIONE_eb882K", f"{TT}/AUDIT/prova_leve/grad_eb882K_kl4", f"{TT}/AUDIT/prova_leve/_tmp",
    f"{TT}/AUDIT/prova_leve/cache_dequant", f"{TT}/TUTTE", f"{TT}/BANCO_1031", f"{FORGE}/stato/cicli.tsv",
]


@dataclasses.dataclass
class Step:
    id: str
    title: str
    kind: str                 # frozen | archived | gap | cpu | gpu | measure
    determinism: str          # bitwise-verified | bitwise-expected | tolerance | measure | frozen | gap
    out: str = ""             # key of G produced
    cmd: str = ""             # bash, run through render(); {PY} {PATCH} ... resolved by fmt()
    rerun: bool = False       # re-executed by --verify
    note: str = ""


STEPS = [
    Step("00", "legacy root R2B3m (EB1->EB3 experts, A2 dense, TT2 scales, R2*): frozen input", "frozen", "frozen", "R2B3m"),
    Step("01", "eb882: second plane on 806 experts + fourth plane on 103 dense (code r030 @58aedd416ca1, empirical-Bayes dKL)",
         "gap", "gap", "eb882", note="writer code and measured-dKL inputs are not archived in this forge; only the hash is recorded"),
    Step("02", "eb882K: scale x1.045 on the single-plane experts (lever K)", "archived", "frozen", "eb882K",
         note="archived GGUF (NAS R029/eb882_K1.045.gguf = TUTTE/eb882K.gguf); the producing command was not archived"),
    Step("03", "CARDAN-KD (rank 16, gate/up experts, step 173) on eb882K + fold + patch -> eb882K-CARDAN", "gpu", "tolerance",
         "CARDAN", "env CK_OSPITE={TT}/TUTTE/eb882K.gguf CK_NOME=eb882K CK_BASE={TT}/CARDAN/base_A1_r16_eb882K "
                   "CK_BASE_UNITA=none CK_OUT={TT}/CARDAN/kd_r16_eb882K CK_LIMITE=9492293427 CK_MMLU=1 CK_ORE=3 "
                   "CK_PASSI=300 CK_RANGO=16 bash {REC}/launchers/cardan_kd_comando_20261010_0053.sh"),
    Step("04", "CARDAN fold patch (152 expert tensors) -> eb882K-CARDAN", "cpu", "bitwise-verified", "CARDAN",
         "LEVE_THREADS=6 LEVE_OSPITE={eb882K} {PY} {PATCH} --dir {fold_dir} --out {O04} --base {eb882K}",
         rerun=True),
    Step("05", "K-FAC gradients of the KL4 answer loss + Kronecker factors on eb882K (250 dense tensors, 896 questions)", "gpu",
         "tolerance", "", "bash {REC}/launchers/grad_eb882K_kl4.sh"),
    Step("06", "nine generic levers on eb882K, K-FAC predictor, per-family selection (SEL_pred_CARDAN)", "gpu", "tolerance",
         "", "bash {REC}/launchers/comando_combinazione.sh",
         note="the launcher also measures every candidate (MMLU-2000); the run takes ~1 day"),
    Step("07", "SEL_pred_CARDAN patch (829 tensors) on eb882K -> B", "cpu", "bitwise-verified", "B",
         "LEVE_THREADS=6 LEVE_OSPITE={eb882K} {PY} {PATCH} --dir {sel_dir} --out {O07} --base {eb882K}",
         rerun=True),
    Step("08", "POST training, joint, 150 steps (50 + resume to 150): scales, F32, rank-16 dense residuals; checkpoint at step 100",
         "gpu", "tolerance", "",
         "BPOST_PASSI=50 bash {REC}/launchers/bpost_b_1101.sh ; BPOST_PASSI=150 bash {REC}/launchers/bpost_b_cont_1101.sh",
         note="the launchers also export and measure T50/T100/T150"),
    Step("09", "T100 export: scales + F32 patch (host H1), dense fold of 130 residuals, patch -> B-POST-T100", "gpu",
         "bitwise-expected", "T100",
         "BPOST_TAPPE=100 BPOST_TAPPE_DIR={CKPT100} BPOST_NON_ATTENDERE=1 BPOST_FINO=gguf bash {REC}/launchers/bpost_b_tappe_1101.sh"),
    Step("10", "MMLU-2000 of T100 (shuffled bench, Q8 activations, Vulkan)", "measure", "measure", "",
         "LD_LIBRARY_PATH={ENGINE} GGML_VK_TQ1R_Q8=1 GGML_VK_DISABLE_GRAPH_OPTIMIZE=1 {ENGINE}/llama-perplexity -m {T100} "
         "-ngl 999 -c 4096 -np 4 -ub 512 -dev Vulkan0 --multiple-choice --multiple-choice-tasks 100000 -f {BENCH}"),
    Step("11", "pack planes (607 plane tensors absorbed into 252 packed families; 1576 -> 969 tensors)", "cpu",
         "bitwise-verified", "packed", "{PY} {PACK} {T100} {WORK}/packed.gguf --check", rerun=True),
    Step("12", "clean metadata (17 internal keys removed, file_type 36, general.name) -> Thor-T-35B-A3B-S.gguf", "cpu",
         "bitwise-verified", "S",
         "{PY} {CLEAN} {WORK}/packed.gguf {WORK}/release.gguf --name '{NAME}' --gguf-py {GGUFPY}", rerun=True),
]
ENGINE = os.environ.get("FORGIA_ENGINE", "/mnt/models/build/thor-rexppiani2/build-rexp/bin")
BENCH = f"{FORGE}/mc_v2/mmlu2000_mescolato.bin"


def located(key: str) -> str | None:
    p, _, _ = G[key]
    for c in ([p] if p else []) + ALT.get(key, []):
        if os.path.exists(c):
            return c
    return None


def chained(key: str) -> str | None:
    """output of a previous --run step, if the mirror holds it"""
    p = OUT / "tree" / G[key][0].lstrip("/")
    return str(p) if p.exists() else None


def fmt(cmd: str, variant: str = "verify") -> str:
    """variant 'verify': recorded inputs, outputs in $OUT/verify.  'run': chained view inside the mirror namespace."""
    run = variant == "run"
    work = OUT / ("release" if run else "verify")
    v = dict(TT=TT, PY=PY, PATCH=PATCH, PACK=PACK, CLEAN=CLEAN, GGUFPY=GGUFPY, OUT=str(OUT), REC=str(REC),
             NAME=S_NAME, ENGINE=ENGINE, BENCH=BENCH, WORK=str(work),
             eb882K=located("eb882K") or G["eb882K"][0], T100=chained("T100") or located("T100") or G["T100"][0],
             CKPT100=CKPT100, O04=str(work / "cardan.gguf"), O07=str(work / "sel_cardan.gguf"),
             fold_dir=DIRS["cardan_fold_tensori"][0], sel_dir=DIRS["sel_pred_cardan_tensori"][0])
    return cmd.format(**v)


def ns_prologue() -> str:
    """Bash that hides every output root behind an empty mirror in $OUT/tree (the record stays reachable read-only in
    $OUT/_rec), then seeds the archived host of the chain. Runs inside `unshare --user --map-root-user --mount`."""
    L = ["set -e"]
    for root in MIRRORED:
        rel = root.lstrip("/")
        mir, rec = OUT / "tree" / rel, OUT / "_rec" / rel
        if root.endswith(".tsv"):
            L += [f"mkdir -p {shlex.quote(str(mir.parent))} {shlex.quote(str(rec.parent))}",
                  f"touch {shlex.quote(str(mir))}",
                  f"if [ -e {shlex.quote(root)} ]; then touch {shlex.quote(str(rec))}; "
                  f"mount --bind {shlex.quote(root)} {shlex.quote(str(rec))}; fi",
                  f"mount --bind {shlex.quote(str(mir))} {shlex.quote(root)}"]
        else:
            L += [f"mkdir -p {shlex.quote(str(mir))} {shlex.quote(str(rec))}",
                  f"if [ -d {shlex.quote(root)} ]; then mount --bind {shlex.quote(root)} {shlex.quote(str(rec))}; fi",
                  f"mkdir -p {shlex.quote(root)}", f"mount --bind {shlex.quote(str(mir))} {shlex.quote(root)}"]
    host = f"{TT}/TUTTE/eb882K.gguf"
    alt = ALT["eb882K"][0]
    L.append(f"ln -sfn \"$( [ -e {shlex.quote(str(OUT / '_rec' / TT.lstrip('/') / 'TUTTE/eb882K.gguf'))} ] && "
             f"echo {shlex.quote(str(OUT / '_rec' / TT.lstrip('/') / 'TUTTE/eb882K.gguf'))} || echo {shlex.quote(alt)} )\" "
             f"{shlex.quote(host)}")
    return "; ".join(L)


def ns_wrap(body: str) -> str:
    return "unshare --user --map-root-user --mount bash -c " + shlex.quote(ns_prologue() + "; " + body)


def lock_wrapped(body: str, who: str, gib: int = 30) -> str:
    return f". {shlex.quote(LOCK)}; prendi_gpu {shlex.quote(who + ' $$')} {gib}; trap molla_gpu EXIT; {body}"


def render(step: Step, variant: str = "verify") -> str:
    c = fmt(step.cmd, variant)
    if step.kind == "measure" or step.id == "09":   # commands the runner owns take the lock; the other launchers take it themselves
        c = lock_wrapped(c, f"RECIPE-S-{step.id}")
    if variant == "run" and step.kind in ("gpu", "measure"):
        c = ns_wrap(c)
    return c


def run_bash(cmd: str, log: Path | None = None) -> int:
    env = dict(os.environ, HIP_VISIBLE_DEVICES="", CUDA_VISIBLE_DEVICES="", ROCR_VISIBLE_DEVICES="") \
        if not re.search(r"lucchetto|launchers/", cmd) else dict(os.environ)
    with open(log, "ab") if log else open(os.devnull, "wb") as f:
        return subprocess.run(["bash", "-c", cmd], stdout=f, stderr=subprocess.STDOUT, env=env).returncode


# ───────────────────────────── verification ─────────────────────────────
Result = list


def check_file(label: str, path: str | None, want: str | None, res: Result) -> None:
    if want is None:
        res.append((label, "RECORDED-ONLY", ""))
    elif not path or not os.path.exists(path):
        res.append((label, "ABSENT", f"recorded {want[:16]}"))
    else:
        got = sha_of(path)
        res.append((label, "MATCH" if got == want else "DIFFER", f"{path} {got[:16]}"))


def verify_hashes(res: Result, nas: bool = True) -> None:
    for k, (p, sha, _) in G.items():
        check_file(f"gguf {k}", located(k), sha, res)
    rel = (REC / "release_s.sha256").read_text().split()[0]
    res.append(("release sha256 equals recorded/release_s.sha256", "MATCH" if rel == G["S"][1] else "DIFFER", rel[:16]))
    for line in TOOLS.read_text().splitlines():
        sha, path = line.split(None, 1)
        check_file(f"tool {Path(path).name}", path.strip(), sha, res)
    for k, (d, man) in DIRS.items():
        if not os.path.isdir(d):
            res.append((f"dir {k}", "ABSENT", d))
            continue
        want = (REC / man).read_text().splitlines()
        got = hashtool.manifest(Path(d), hashtool._load_cache())
        res.append((f"dir {k}", "MATCH" if got == want else "DIFFER", f"{len(got)} files vs {len(want)} recorded"))
    ck = REC / "post_step100_checkpoints.sha256"
    for line in ck.read_text().splitlines():
        sha, name = line.split()
        p = Path(CKPT100) / name
        res.append((f"step-100 checkpoint {name}", "MATCH" if p.exists() and sha_of(p) == sha else
                    ("ABSENT" if not p.exists() else "DIFFER"), CKPT100))


def verify_deep(res: Result) -> None:
    p = located("T100")
    if not p:
        res.append(("tensors of T100", "ABSENT", ""))
        return
    sys.path.insert(0, str(HERE.parent))
    from fucina.replay import gguf_hash
    want = dict(reversed(x.split("  ", 1)) for x in (REC / "t100_tensors.manifest").read_text().splitlines())
    got = gguf_hash.tensor_hashes(p)
    bad = [k for k in want if got.get(k) != want[k]]
    res.append(("tensors of T100 (1576)", "MATCH" if not bad and len(got) == len(want) else "DIFFER", f"{len(bad)} differ"))


def rerun_step(step: Step, res: Result, keep: bool) -> None:
    vd = OUT / "verify"
    vd.mkdir(parents=True, exist_ok=True)
    srcs = {"04": located("eb882K"), "07": located("eb882K"), "11": located("T100"), "12": None}
    need = [srcs[step.id]] if step.id != "12" else []
    if step.id == "12" and not (vd / "packed.gguf").exists():
        res.append((f"step {step.id} {step.title[:40]}", "SKIPPED", "needs step 11 output"))
        return
    if step.id in ("04", "07"):
        d = DIRS["cardan_fold_tensori" if step.id == "04" else "sel_pred_cardan_tensori"][0]
        need.append(d if os.path.isdir(d) else None)
    if any(n is None for n in need):
        res.append((f"step {step.id} {step.title[:40]}", "ABSENT", "recorded inputs missing on this machine"))
        return
    t0 = time.time()
    rc = run_bash(render(step), vd / f"{step.id}.log")
    outs = {"04": "cardan.gguf", "07": "sel_cardan.gguf", "11": "packed.gguf", "12": "release.gguf"}
    o = vd / outs[step.id]
    want = G[step.out][1]
    got = sha_of(o) if rc == 0 and o.exists() else None
    ok = got == want
    res.append((f"step {step.id} rerun {step.title[:40]}", "MATCH" if ok else "DIFFER",
                f"rc={rc} {int(time.time() - t0)} s sha {(got or '-')[:16]} (recorded {want[:16]})"))
    if ok and not keep and step.id != "11":
        o.unlink()
    if step.id == "12" and ok and not keep:
        (vd / "packed.gguf").unlink(missing_ok=True)


def do_verify(args) -> int:
    res: Result = []
    verify_hashes(res)
    if args.deep:
        verify_deep(res)
    if not args.hashes_only:
        for s in STEPS:
            if s.rerun and (not args.steps or s.id in args.steps):
                rerun_step(s, res, args.keep)
    w = max(len(r[0]) for r in res)
    for label, st, info in res:
        print(f"{st:14s} {label:{w}s}  {info}")
    bad = [r for r in res if r[1] == "DIFFER"]
    (OUT / "state").mkdir(parents=True, exist_ok=True)
    (OUT / "state" / "verify.json").write_text(json.dumps(res, indent=1))
    print(f"\n{sum(r[1] == 'MATCH' for r in res)} match, {len(bad)} differ, "
          f"{sum(r[1] in ('ABSENT', 'SKIPPED') for r in res)} absent/skipped, "
          f"{sum(r[1] == 'RECORDED-ONLY' for r in res)} recorded-only")
    return 1 if bad else 0


# ─────────────────────────────── plan / run ───────────────────────────────
def select(args) -> list[Step]:
    ids = [s.id for s in STEPS]
    sel = [s for s in STEPS if (not args.steps or s.id in args.steps)
           and (not args.from_ or ids.index(s.id) >= ids.index(args.from_))
           and (not args.to or ids.index(s.id) <= ids.index(args.to))]
    return sel


def do_plan(args) -> int:
    for s in select(args):
        out = G.get(s.out) if s.out else None
        print(f"[{s.id}] {s.title}\n     kind={s.kind} determinism={s.determinism}"
              + (f" output={s.out} sha256={out[1]}" if out else ""))
        if s.cmd:
            print(f"     cmd: {render(s)}")
        if s.note:
            print(f"     note: {s.note}")
    print("\nlocal artifacts:")
    for k in G:
        print(f"  {k:8s} {located(k) or '-'}")
    return 0


def _unused() -> None:
    """Seed the mirror with the archived host so the first launcher finds its input; nothing is copied."""
    for key, rel in (("eb882K", "TUTTE/eb882K.gguf"), ("B", "BANCO_1031/Thor-T-eb882K-SEL_pred_CARDAN.gguf")):
        dst = OUT / "tree" / TT.lstrip("/") / rel
        src = located(key)
        if src and not dst.exists() and key == "eb882K":
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.symlink_to(src)


def do_run(args) -> int:
    if not os.path.exists(LOCK):
        print(f"house GPU lock {LOCK} not found: set FORGIA_ROOT", file=sys.stderr)
        return 2
    (OUT / "state").mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    if subprocess.run(['unshare', '--user', '--map-root-user', '--mount', 'true']).returncode:
        print('unprivileged user namespaces are required (record protection)', file=sys.stderr)
        return 2
    before = {} if args.dry_run else {k: located(k) and sha_of(located(k)) for k in ("eb882K", "B", "T100")}
    rc_all = 0
    for s in select(args):
        stf = OUT / "state" / f"{s.id}.json"
        if stf.exists() and json.loads(stf.read_text()).get("ok"):
            print(f"[{s.id}] already done, skipped")
            continue
        if s.kind in ("frozen", "archived", "gap"):
            p = located(s.out)
            ok = bool(p) and sha_of(p) == G[s.out][1]
            print(f"[{s.id}] {s.kind}: {'input present and verified' if ok else 'NOT AVAILABLE (see RECIPE_S.md gaps)'}")
            if not ok and not args.allow_gaps:
                return 3
            continue
        t0 = time.time()
        cmd = render(s, 'run')
        (OUT / 'release').mkdir(parents=True, exist_ok=True)
        print(f"[{s.id}] {s.title}\n   $ {cmd}", flush=True)
        rc = 0 if args.dry_run else run_bash(cmd, OUT / "logs" / f"{s.id}.log")
        ok = rc == 0
        if ok and s.determinism.startswith("bitwise") and s.rerun and not args.dry_run:
            o = OUT / "release" / {"04": "cardan.gguf", "07": "sel_cardan.gguf", "11": "packed.gguf", "12": "release.gguf"}[s.id]
            ok = o.exists() and sha_of(o) == G[s.out][1]
        if not args.dry_run:
            stf.write_text(json.dumps({"ok": ok, "rc": rc, "seconds": int(time.time() - t0),
                                   "determinism": s.determinism}))
        print(f"[{s.id}] {'OK' if ok else 'FAILED'} ({s.determinism})")
        if not ok:
            rc_all = 1
            break
    after = {k: located(k) and sha_of(located(k)) for k in before}
    if after != before:
        print("ERROR: a recorded artifact changed during the run", file=sys.stderr)
        return 4
    return rc_all


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    m = ap.add_mutually_exclusive_group()
    m.add_argument("--verify", action="store_true")
    m.add_argument("--run", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="plan only; with --run, render the namespaced commands without executing")
    ap.add_argument("--steps", type=lambda x: x.split(","), default=None)
    ap.add_argument("--from", dest="from_")
    ap.add_argument("--to")
    ap.add_argument("--hashes-only", action="store_true")
    ap.add_argument("--deep", action="store_true")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--allow-gaps", action="store_true")
    a = ap.parse_args(argv)
    if a.verify:
        return do_verify(a)
    if a.run:
        return do_run(a)
    return do_plan(a)


if __name__ == "__main__":
    sys.exit(main())
