"""Recipe C, step 31 (second half): turn the POST checkpoint at step 50 into the deployable GGUF C and measure it.

The POST training (50 joint steps on the folded T400) leaves torch checkpoints, not a model. This module runs the
recorded export sequence (``t400post_tappe_v2_1027.sh``, as derived by the internal re-run of 31 Oct) with every
location passed on the command line:

1. check the step-50 checkpoint is deployable: no expert low-rank state, dense residuals only on attention/GDN
   families, all 301 F32 tensors present;
2. H1 = the folded T400 with the trained gate/up scales (exported with the student's exact fp16 arithmetic, checked
   sha by sha), the trained down scales (fp32 product, as the student computes it) and the 301 trained F32 tensors;
   a byte check allows only scale fields and F32 tensors to change;
3. the dense residuals of step 50 are folded into H1's dense tensors (zero bytes added) -> C; a byte check allows
   only the folded dense tensors to change;
4. MMLU-2000 of C on the mixed-order bench, remapped to the original order, and an exact McNemar test against the
   replay's own T400 log.

Nothing is deleted. Usage::

    python -m fucina.replay.thor.post_export --base T400.gguf --work CARDAN_T400POST_DIR --code BPP_CODE_DIR \\
        --scripts STATO_DIR --tools TOOLS_DIR --patch patch_gguf.py --engine ENGINE_BIN \\
        --bench-mixed mmlu2000_mescolato.bin --index mmlu2000_mescolato.indice.tsv --reference T400_MMLU.log \\
        --python PYTHON [--dry-run]

``--tools`` holds ``scale_gateup_esporta.py``, ``f32_esporta.py`` and ``mcnemar_mclp.py``; ``--scripts`` holds
``bpost_scale_down_fp32_1026.py``, ``bpost_verifica_1026.py``, ``tappa_a_delta_1018.py`` and ``parziale_mmlu.py``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path

STEP = 50
FAMILIES = {"attn_q", "attn_k", "attn_v", "attn_output", "attn_qkv", "attn_gate", "ssm_out"}
F32_TENSORS = 301
CPU = "env HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= OMP_NUM_THREADS=4 nice -n 19 env"
GPU = "env HSA_ENABLE_SDMA=0 TORCHINDUCTOR_COMPILE_THREADS=1 nice -n 19 env"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def commands(a) -> list[str]:
    q = lambda p: shlex.quote(str(p))  # noqa: E731
    t, o = a.work / "allena" / "tappe", a.work / f"tappa_{STEP}"
    h1, g = o / "ospite_H1.gguf", o / f"Thor-T-C-T400POST-T{STEP}.gguf"
    delta, pd = o / "residui_delta.pt", o / "piega_d"
    log = o / f"mmlu_T400POST{STEP}.log"
    py = q(a.python)
    rx = r"'blk\.[0-9]+\.ffn_(gate|up|down)_exps\.weight'"
    ver = q(a.scripts / "bpost_verifica_1026.py")
    return [
        f"mkdir -p {q(o)}/scale_gu {q(o)}/scale_down {q(o)}/f32 {q(o)}/h1_tensori {q(pd)}",
        # gate/up scales with the student's exact arithmetic, then a sha-by-sha check
        f"{GPU} {py} {q(a.tools / 'scale_gateup_esporta.py')} {q(a.base)} {q(t)}/scale_gateup_{STEP}.pt "
        f"{q(o)}/scale_gu --device cuda",
        f"{CPU} {py} -c \"import json; t = json.load(open('{o}/scale_gu/scale_gateup_esporta.json'))['tensori']; "
        "assert t and all(v['sha_come_studente'] is True for v in t.values()), 'gate/up export differs from the student'\"",
        # down scales: fp32 product as in the student (not fp64)
        f"{GPU} {py} {q(a.scripts / 'bpost_scale_down_fp32_1026.py')} {q(a.base)} {q(t)}/scale_{STEP}.pt "
        f"{q(o)}/scale_down --device cuda",
        # the 301 F32 tensors (exit 3 = nothing to change for some tensors, accepted as recorded)
        f"rc=0; {CPU} {py} {q(a.tools / 'f32_esporta.py')} --base {q(a.base)} --pacco {q(t)}/f32_{STEP}.pt "
        f"--uscita {q(o)}/f32 || rc=$?; [ \"$rc\" = 0 ] || [ \"$rc\" = 3 ]",
        f"ln {q(o)}/scale_gu/*.npy {q(o)}/scale_down/*.npy {q(o)}/h1_tensori/; "
        f"for f in {q(o)}/f32/tensori/*.npy; do [ ! -e \"$f\" ] || ln \"$f\" {q(o)}/h1_tensori/; done",
        f"env LEVE_OSPITE={q(a.base)} {CPU} {py} {q(a.patch)} --dir {q(o)}/h1_tensori --out {q(h1)} --base {q(a.base)}",
        f"{CPU} {py} {ver} {q(a.base)} {q(h1)} --ammessi {rx} --f32 --codici {rx} --almeno 1 "
        f"--json {q(o)}/verifica_h1.json",
        # dense residuals -> zero-byte fold into H1
        f"{CPU} {py} {q(a.scripts / 'tappa_a_delta_1018.py')} {q(t)}/residui_{STEP}.pt {q(delta)} {STEP}",
        f"{GPU} OMP_NUM_THREADS=6 LEVE_THREADS=6 {py} -X faulthandler {q(a.code / 'cardan_piega_densi_gpu_veloce.py')} "
        f"--delta {q(delta)} --gguf {q(h1)} --out {q(pd)} --device cuda --dtype float32 --lotto 8",
        f"{CPU} {py} -m fucina.replay.thor.post_export --fold-inventory {q(delta)} {q(pd)}",
        f"env LEVE_OSPITE={q(h1)} {CPU} {py} {q(a.patch)} --dir {q(pd)}/tensori --out {q(g)} --base {q(h1)}",
        f"{CPU} {py} {ver} {q(h1)} {q(g)} --ammessi-file {q(pd)}/nomi.txt --almeno 1 --json {q(o)}/verifica_g.json",
        f"nice -n 19 sha256sum {q(g)} > {q(g)}.sha256",
        # MMLU-2000 on the mixed bench, remapped to the original order, then McNemar against the replay's T400
        f"env LD_LIBRARY_PATH={q(a.engine)} GGML_VK_TQ1R_Q8=1 GGML_VK_DISABLE_GRAPH_OPTIMIZE=1 nice -n 19 "
        f"{q(a.engine / 'llama-perplexity')} -m {q(g)} -ngl 999 -c 4096 -np 4 -ub 512 -dev {a.device} "
        f"--multiple-choice --multiple-choice-tasks 100000 -f {q(a.bench_mixed)} > {q(log)}.mesc 2>&1",
        f"{CPU} {py} {q(a.scripts / 'parziale_mmlu.py')} --indice {q(a.index)} --rimappa {q(log)} {q(log)}.mesc "
        f"{q(a.reference)}",
        f"[ \"$(grep -c '^MCLP\t' {q(log)})\" = 2000 ]",
        f"{CPU} {py} {q(a.tools / 'mcnemar_mclp.py')} {q(a.reference)} {q(log)} --json {q(o)}/mcnemar_vs_T400.json",
    ]


def check_checkpoint(work: Path) -> None:
    import torch
    t = work / "allena" / "tappe"
    for name in ("cardan", "residui", "scale", "scale_gateup", "f32", "ottimizzatori"):
        if not (t / f"{name}_{STEP}.pt").is_file():
            raise ValueError(f"missing step-{STEP} checkpoint: {name}")
    load = lambda n: torch.load(t / f"{n}_{STEP}.pt", map_location="cpu", weights_only=False)  # noqa: E731
    c = load("cardan")
    if c.get("rango", 0) != 0 or c.get("stato"):
        raise ValueError("expert low-rank state present: not deployable")
    if (t / f"cardan_down_{STEP}.pt").exists() and load("cardan_down").get("stato"):
        raise ValueError("down low-rank state present: not deployable")
    r = load("residui")
    if not r["cfg"].get("continuo") or not set(r["cfg"]["famiglie"]) <= FAMILIES:
        raise ValueError("residuals outside the attention/GDN dense families")
    n = len(load("f32")["valori"])
    if n != F32_TENSORS:
        raise ValueError(f"expected all {F32_TENSORS} F32 tensors, got {n}")


def fold_inventory(delta: Path, fold_dir: Path) -> int:
    """The folded tensors must be exactly the dense tensors of the delta; writes nomi.txt for the byte check."""
    import torch
    d = torch.load(delta, map_location="cpu", weights_only=False, mmap=True)["delta_file"]
    names = sorted(x.stem for x in (fold_dir / "tensori").glob("*.npy"))
    want = {x if x.endswith(".weight") else x + ".weight" for x in d}
    if not names or set(names) != want:
        raise ValueError("dense fold inventory differs from the delta")
    (fold_dir / "nomi.txt").write_text("\n".join(names) + "\n")
    return len(names)


def finalize(work: Path) -> dict:
    o = work / f"tappa_{STEP}"
    g = o / f"Thor-T-C-T400POST-T{STEP}.gguf"
    stats = json.loads((o / "mcnemar_vs_T400.json").read_text())
    if stats.get("n") != 2000:
        raise ValueError("incomplete aligned McNemar sample")
    for report in ("verifica_h1.json", "verifica_g.json"):
        if json.loads((o / report).read_text()).get("esito") is not True:
            raise ValueError(f"byte check failed: {report}")
    rows = [json.loads(x) for x in (o / "piega_d" / "registro.jsonl").read_text().splitlines() if x.strip()]
    latest = {r["nome"]: r for r in rows}
    energy = sum(r["energia_correzione_H"] for r in latest.values())
    residual = sum(r["residuo_dopo_piega_H"] for r in latest.values())
    if energy <= 0:
        raise ValueError("empty dense fold")
    result = {"ok": True, "steps": STEP, "mmlu2000": stats["acc_B"], "t400_mmlu2000": stats["acc_A"],
              "p_mcnemar_exact": stats["p_mcnemar_esatto"], "n": stats["n"],
              "wins": stats["B_giusta_A_sbagliata"], "losses": stats["A_giusta_B_sbagliata"],
              "dense_capture_percent": 100 * (1 - residual / energy), "gguf": str(g), "sha256": sha256(g)}
    (work / "result.json").write_text(json.dumps(result, indent=1) + "\n")
    return result


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["--fold-inventory"]:
        print(fold_inventory(Path(argv[1]), Path(argv[2])), "folded dense tensors")
        return 0
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for k in ("base", "work", "code", "scripts", "tools", "patch", "engine", "bench_mixed", "index", "reference"):
        ap.add_argument("--" + k.replace("_", "-"), type=Path, required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--device", default="Vulkan0")
    ap.add_argument("--dry-run", action="store_true", help="print the commands and stop")
    a = ap.parse_args(argv)
    cmds = commands(a)
    if a.dry_run:
        print("\n".join(cmds))
        return 0
    check_checkpoint(a.work)
    if (a.work / f"tappa_{STEP}").exists():
        raise SystemExit(f"{a.work}/tappa_{STEP} exists: a partial export is kept for diagnosis; use a fresh work dir")
    subprocess.run(["bash", "-c", "set -euo pipefail\n" + "\n".join(cmds)], check=True)
    print(json.dumps(finalize(a.work)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
