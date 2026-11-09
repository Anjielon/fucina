#!/usr/bin/env python3
"""Profilo di tre passi REALI del trainer, escluso il caricamento iniziale.

Ingresso sorvegliato, non ottimizzazione certificata. Argomenti dopo "--" inoltrati
al trainer originale (o --variante veloce), salvo --passi che DEVE essere 3.
Richiede --uscita nuova, fuori da /tmp. Non riprende o copia checkpoint vivi.
Una sonda Kineto minima separata deve riuscire prima di caricare il trainer.
TORCHINDUCTOR_COMPILE_THREADS=1 prima di importare Torch.
Cache dequant esistente sempre mmap verificata in sola lettura: il confronto e'
del PASSO, non del caricamento originale. Cache mancante/corrotta = errore.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import shlex
import sys
import time

os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"


def opzioni_trainer():
    """Legge i nomi dal parser originale senza importare Torch o il trainer."""
    sorgente = Path(__file__).with_name("cardan_congiunto.py")
    modulo = ast.parse(sorgente.read_text(), filename=str(sorgente))
    funzione = next(n for n in modulo.body
                    if isinstance(n, ast.FunctionDef) and n.name == "argomenti")
    opzioni = set()
    for n in ast.walk(funzione):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "a_":
            if not n.args or not all(isinstance(a, ast.Constant) and isinstance(a.value, str)
                                     for a in n.args):
                raise RuntimeError("parser trainer cambiato: opzioni non statiche, profilo negato")
            opzioni.update(a.value for a in n.args)
    richieste = {"--uscita", "--passi", "--device", "--modo", "--riprendi",
                 "--prova-secco", "--prova-fuso"}
    if not richieste <= opzioni:
        raise RuntimeError("parser trainer cambiato: controlli incompleti, profilo negato")
    return opzioni


def argomenti(argv):
    ap = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    ap.add_argument("--variante", choices=("originale", "veloce"), default="originale")
    ap.add_argument("trainer", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    args = a.trainer[1:] if a.trainer[:1] == ["--"] else a.trainer
    if any(any(c in s for c in "\t\r\n") for s in args):
        ap.error("argomenti con tab/newline non rappresentabili nel piano di prove")
    opzioni = opzioni_trainer()
    for token in args:
        if token.startswith("--") and token.split("=", 1)[0] not in opzioni:
            ap.error(f"opzione trainer non esatta o sconosciuta: {token}")
        if token == "-h":
            ap.error("help del trainer non ammesso in una prova sorvegliata")
    p = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    p.add_argument("--uscita", required=True)
    p.add_argument("--passi", required=True, type=int)
    p.add_argument("--device", default="cuda")
    p.add_argument("--modo", default="congiunto")
    p.add_argument("--riprendi", action="store_true")
    p.add_argument("--prova-secco", type=int, default=0)
    p.add_argument("--prova-fuso", type=int, default=0)
    b, _ = p.parse_known_args(args)
    if b.passi != 3 or b.riprendi or b.prova_secco or b.prova_fuso:
        ap.error("servono --passi 3, senza ripresa, prova-secco o prova-fuso")
    if b.modo != "congiunto":
        ap.error("profilo a due update richiede --modo congiunto")
    if b.device.split(":")[0] != "cuda":
        ap.error("profilazione fisica richiede --device cuda[:N]")
    # finestra_gpu usa code e busy del device 0; non autorizzare altri device.
    if b.device not in ("cuda", "cuda:0"):
        ap.error("questo ingresso sorvegliato supporta solo cuda:0")
    if any(os.environ.get(k, "") not in ("", "0")
           for k in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")):
        ap.error("rimappatura device non supportata dal gate fisico")
    out = Path(b.uscita).resolve()
    if out == Path("/tmp") or Path("/tmp") in out.parents:
        ap.error("--uscita sotto /tmp non ammessa")
    return a.variante, args, out


class TrePassi:
    """Due Adam originali, un confine dopo ENTRAMBI gli update del passo."""

    def __init__(self, torch, out):
        self.torch, self.out = torch, out
        self.optimizzatori = []
        self.aggiornati = set()
        self.durate = []
        self.profilo = None
        self.inizio = None
        self.zero = torch.optim.Adam.zero_grad
        self.step = torch.optim.Adam.step

    def __enter__(self):
        controllo = self

        def zero(opt, *args, **kwargs):
            if controllo.profilo is None and not controllo.durate:
                controllo.profilo = controllo.torch.profiler.profile(
                    activities=[controllo.torch.profiler.ProfilerActivity.CPU,
                                controllo.torch.profiler.ProfilerActivity.CUDA],
                    record_shapes=True, profile_memory=True, with_stack=False)
                controllo.profilo.__enter__()
            if not any(opt is x for x in controllo.optimizzatori):
                if len(controllo.optimizzatori) == 2:
                    raise RuntimeError("trainer cambiato: piu' di due Adam")
                controllo.optimizzatori.append(opt)
            if opt is controllo.optimizzatori[0] and len(controllo.durate) < 3:
                controllo.torch.cuda.synchronize()
                controllo.inizio = time.perf_counter()
            return controllo.zero(opt, *args, **kwargs)

        def step(opt, *args, **kwargs):
            risultato = controllo.step(opt, *args, **kwargs)
            if len(controllo.durate) >= 3:
                return risultato
            if len(controllo.optimizzatori) != 2:
                raise RuntimeError("profilo incompleto: attesi due Adam")
            if not any(opt is x for x in controllo.optimizzatori):
                raise RuntimeError("Adam non osservato durante zero_grad")
            controllo.aggiornati.add(id(opt))
            if len(controllo.aggiornati) == 2:
                controllo.torch.cuda.synchronize()
                controllo.durate.append(time.perf_counter() - controllo.inizio)
                controllo.aggiornati.clear()
                controllo.profilo.step()
                if len(controllo.durate) == 3:
                    controllo.chiudi()
            return risultato

        self.torch.optim.Adam.zero_grad = zero
        self.torch.optim.Adam.step = step
        return self

    def chiudi(self):
        if self.profilo is not None:
            profilo, self.profilo = self.profilo, None
            profilo.__exit__(None, None, None)
            temporaneo = self.out / "traccia.parziale.json"
            profilo.export_chrome_trace(str(temporaneo))
            os.replace(temporaneo, self.out / "traccia.json")
            operatori = self.out / "operatori.parziale.txt"
            operatori.write_text(
                profilo.key_averages().table(sort_by="self_device_time_total", row_limit=100))
            os.replace(operatori, self.out / "operatori.txt")

    def __exit__(self, *exc):
        self.torch.optim.Adam.zero_grad = self.zero
        self.torch.optim.Adam.step = self.step
        self.chiudi()


def lavoratore(variante, args, out):
    with (out / "_profilo" / "lavoratore.json").open("x") as f:
        json.dump({"pid": os.getpid(), "variante": variante, "argv": args,
                   "compile_threads": os.environ["TORCHINDUCTOR_COMPILE_THREADS"]}, f)
    import cardan_congiunto as CG
    import torch
    # Il log globale delle corse storiche non appartiene a questo collaudo.
    CG.KDP.CICLI = str(out / "cicli_profilo.tsv")
    if variante == "veloce":
        from cardan_congiunto_veloce_gpu import prepara
        prepara()
    if os.environ.get("E2E_CACHE_DEQUANT"):
        import e2e_cache_dequant as CD
        from cardan_cache_mmap_gpu import lettore
        CD.somma_piani = lettore(CD)
    with TrePassi(torch, out / "_profilo") as misura:
        rc = CG.main(args)
    if rc:
        return rc
    if len(misura.durate) != 3:
        raise RuntimeError(f"profilo incompleto: {len(misura.durate)}/3 passi aggiornati")
    storia = json.loads((out / "storia.json").read_text())
    passi = [r for r in storia["storia"] if r.get("passo", 0) > 0]
    if len(passi) != 3 or any("saltato" in r for r in passi):
        raise RuntimeError("profilo incompleto: passi saltati o storia diversa da tre passi")
    (out / "_profilo" / "misure.json").write_text(json.dumps({
        "variante": variante, "passi": passi, "secondi_sincronizzati": misura.durate,
        "nota": "Tempi CON profiler e sincronizzazioni: NON benchmark isolato, "
                "NON confronto A/B o certificato dello stato iniziale."
    }, indent=2))
    return 0


def sonda_profiler(out):
    meta = out / "_profilo"
    with (meta / "sonda_lavoratore.json").open("x") as f:
        json.dump({"pid": os.getpid(), "compile_threads":
                   os.environ["TORCHINDUCTOR_COMPILE_THREADS"]}, f)
    import torch
    inizio = time.perf_counter()
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True, profile_memory=True, with_stack=False) as profilo:
        x = torch.ones((32, 32), device="cuda", requires_grad=True)
        y = (x @ x).sum()
        y.backward()
        torch.cuda.synchronize()
    if not any(e.device_type == torch.autograd.DeviceType.CUDA for e in profilo.events()):
        raise RuntimeError("AVVISO: sonda profiler senza eventi GPU; trainer non autorizzato")
    traccia = meta / "sonda_traccia.parziale.json"
    profilo.export_chrome_trace(str(traccia))
    os.replace(traccia, meta / "sonda_traccia.json")
    from pianificatore_gpu import scrivi_json
    scrivi_json(meta / "sonda_profiler.json", {
        "ok": True, "secondi": time.perf_counter() - inizio,
        "nota": "Solo sonda minima Kineto, NON certificazione di stabilita' del profilo completo."
    })
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    interno = argv[:1] in (["--lavoratore"], ["--sonda-profiler"])
    variante, args, out = argomenti(argv[1:] if interno else argv)
    if interno:
        if os.environ.get("FORGIA_PROFILO_SORVEGLIATO") != str(out):
            raise RuntimeError("lavoratore avviabile solo dal supervisore")
        if argv[0] == "--sonda-profiler":
            return sonda_profiler(out)
        sonda = json.loads((out / "_profilo" / "sonda_profiler.json").read_text())
        if sonda.get("ok") is not True:
            raise RuntimeError("AVVISO: sonda profiler non riuscita; trainer non autorizzato")
        return lavoratore(variante, args, out)
    if out.exists():
        raise FileExistsError(f"uscita deve essere nuova: {out}")
    import shutil
    import finestra_gpu as F
    os.nice(max(0, 19 - os.getpriority(os.PRIO_PROCESS, 0)))
    parent = out.parent
    while not parent.exists():
        parent = parent.parent
    if shutil.disk_usage(parent).free < 24 * (1 << 30):
        raise RuntimeError("meno di 24 GiB liberi: nessun profilo avviato")
    a = argparse.Namespace(corsa="cardan_congiunto.py", ram_gib=14,
                           compute=2, sdma=0, busy_max=10)
    motivo = F.finestra(a)
    out.mkdir(parents=True)
    meta = out / "_profilo"
    meta.mkdir()
    if motivo:
        (meta / "AVVISO.json").write_text(json.dumps({
            "motivo": motivo, "gpu_lanciata": False, "torch_importato": "torch" in sys.modules
        }, indent=2))
        print(f"AVVISO: {motivo}; nessun profilo GPU avviato", file=sys.stderr)
        return 5
    os.environ["FORGIA_PROFILO_SORVEGLIATO"] = str(out)
    # Cache e temporanei del solo figlio; mai cambiare gli ambienti delle corse.
    for nome, sub in (("TMPDIR", "tmp"), ("TRITON_CACHE_DIR", "triton"),
                       ("TORCHINDUCTOR_CACHE_DIR", "inductor")):
        d = meta / sub
        d.mkdir()
        os.environ[nome] = str(d)
    cmd = [sys.executable, str(Path(__file__).resolve()), "--lavoratore",
           "--variante", variante, "--", *args]
    smoke = ["timeout", "--foreground", "--kill-after=5s", "60s",
             sys.executable, str(Path(__file__).resolve()), "--sonda-profiler",
             "--variante", variante, "--", *args]
    prove = meta / "prove.tsv"
    prove.write_text("sonda_profiler\t" + shlex.join(smoke) + "\n"
                     "tre_passi\t" + shlex.join(cmd) + "\n")
    return F.main(["--prove", str(prove), "--registro", str(meta / "finestra.jsonl"),
                   "--ram-gib", "14", "--attesa-max-s", "1"])


if __name__ == "__main__":
    raise SystemExit(main())
