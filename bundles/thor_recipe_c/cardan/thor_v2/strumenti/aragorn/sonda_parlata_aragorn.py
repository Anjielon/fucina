# 27/9 — SONDA PARLATA su ARAGORN (RTX 4060 8 GB, Vulkan), gemella di @@ROOT:forge@@/sonda_parlata_k1024.sh
# del Corsair: stesse 3 domande, stesse regex, stessi 7 seed, stesso corpo della richiesta, stessi argomenti del
# llama-server. Uniche differenze, dichiarate:
#   * motore: albero del banco MMLU di Aragorn (llama.cpp 521a64cd + TQ1_0R_1024 + patch MCLP), llama-server compilato
#     il 27/9 dallo stesso build;
#   * posizione: -dev Vulkan0 e gli esperti degli strati 0-15 in RAM (-ot come banco_ot.bat), perche' 8 GB di VRAM;
#   * il client e' Python (urllib) invece di curl, e l'attesa per richiesta e' 900 s invece di 300 (piu' lento del
#     Corsair: un taglio per tempo sarebbe un difetto della macchina, non del modello). La durata di ogni richiesta e'
#     registrata: se una supera 300 s lo si vede.
# Uso (lanciato da schtasks, MAI Start-Process da ssh):  python sonda_parlata_aragorn.py NOME FILE.gguf [HH:MM limite]
import datetime, json, os, re, subprocess, sys, time, urllib.request

NOME, GGUF = sys.argv[1], sys.argv[2]
LIMITE = sys.argv[3] if len(sys.argv) > 3 else "14:35"
B = r"C:\tcq\thor_mmlu\sorgente\build\bin\Release"
L = r"C:\tcq\thor_mmlu\log"
PORTA = 11458
THINK = True
SEEDS = [1, 2, 3, 7, 42, 100, 999]
DOMANDE = [
    ("Un prodotto costa 23 euro. Sconto del 15 per cento, poi meno altri 2 euro. Quanto costa alla fine? "
     "Rispondi col solo numero.", r"17[.,]55"),
    ("Alice ha 3 fratelli e 2 sorelle. Quante sorelle ha il fratello di Alice? Rispondi col solo numero.", r"\b3\b"),
    ("Scrivi la parola CANE al contrario. Rispondi con la sola parola.", r"enac"),
]
ESITO = os.path.join(L, f"sonda_{NOME}.txt")
GREZZE = os.path.join(L, f"sonda_{NOME}_grezze.jsonl")

hh, mm = (int(x) for x in LIMITE.split(":"))
oggi = datetime.datetime.now()
SCADENZA = oggi.replace(hour=hh, minute=mm, second=0, microsecond=0)
if SCADENZA <= oggi:   # 1/10: limite gia' passato oggi = limite di DOMANI (una sonda partita dopo le 23:59 moriva subito)
    SCADENZA += datetime.timedelta(days=1)


def log(s=""):
    print(s, flush=True)


def scaduto():
    return datetime.datetime.now() >= SCADENZA


ARGS = [os.path.join(B, "llama-server.exe"), "--model", GGUF, "--alias", "sonda", "-ngl", "999",
        "-dev", "Vulkan0", "-ot", r"blk\.([0-9]|1[0-5])\.ffn_.*_exps=CPU",
        "--ctx-size", "8192", "--parallel", "1", "--flash-attn", "on",
        "--cache-type-k", "q8_0", "--cache-type-v", "q8_0", "--host", "127.0.0.1", "--port", str(PORTA),
        "--threads", "8", "--temp", "0.6", "--min-p", "0.03", "--top-p", "0.9", "--jinja",
        "--reasoning-budget", "4096"]

log("=" * 60)
log(f"SONDA PARLATA (ARAGORN) · {NOME} · thinking={str(THINK).lower()} · {oggi:%d/%m %H:%M:%S} · limite {LIMITE}")
log(f"modello: {GGUF} ({os.path.getsize(GGUF)} byte)")
log("server: " + " ".join(ARGS))
log("=" * 60)

srv_log = open(os.path.join(L, f"sonda_{NOME}_server.log"), "w")
srv = subprocess.Popen(ARGS, stdout=srv_log, stderr=subprocess.STDOUT)
t0 = time.time()
rc = 3
try:
    acceso = False
    while time.time() - t0 < 900 and not scaduto():
        if srv.poll() is not None:
            log(f"⛔ llama-server uscito con rc={srv.returncode} prima di accendersi")
            raise SystemExit(4)
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORTA}/health", timeout=2) as r:
                if b"ok" in r.read():
                    acceso = True
                    break
        except Exception:
            pass
        time.sleep(5)
    if not acceso:
        log("⛔ il modello non si accende (15 min o limite orario)")
        raise SystemExit(4)
    try:
        vram = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e:
        vram = f"? ({e})"
    log(f"acceso dopo ~{int(time.time() - t0)}s · VRAM {vram}")
    log()

    def chiedi(dom, att, seed):
        corpo = {"model": "s", "stream": False, "max_tokens": 2000, "temperature": 0.6, "seed": seed,
                 "messages": [{"role": "user", "content": dom}],
                 "chat_template_kwargs": {"enable_thinking": THINK}}
        req = urllib.request.Request(f"http://127.0.0.1:{PORTA}/v1/chat/completions",
                                     data=json.dumps(corpo).encode(), headers={"Content-Type": "application/json"})
        ti = time.time()
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                d = json.loads(r.read())
        except Exception as e:
            dt = time.time() - ti
            log(f"  seed {seed:>4}  GUASTO   {dt:6.0f} s · risposta illeggibile: {str(e)[:60]}")
            return "GUASTO"
        dt = time.time() - ti
        m = (d.get("choices") or [{}])[0].get("message", {}) or {}
        rcont = (m.get("reasoning_content") or "").strip()
        c = (m.get("content") or "").strip()
        tps = round((d.get("timings") or {}).get("predicted_per_second", 0), 1)
        esito = "MUTO" if not c else ("OK" if re.search(att, c, re.I) else "SBAGLIA")
        with open(GREZZE, "a", encoding="utf-8") as f:
            f.write(json.dumps({"seed": seed, "domanda": dom, "esito": esito, "secondi": round(dt, 1),
                                "content": c, "reasoning_content": rcont, "timings": d.get("timings"),
                                "usage": d.get("usage"), "finish_reason": (d.get("choices") or [{}])[0]
                                .get("finish_reason")}, ensure_ascii=False) + "\n")
        testo = c[:110].replace("\n", " ")
        log(f"  seed {seed:>4}  {esito:<8} pensiero {len(rcont):>5} car · {tps:>5} tok/s · {dt:5.0f} s · {testo}")
        return esito

    muti = 0
    giusti_tot = 0
    tassi = ""
    fatte = 0
    interrotta = False
    for n, (dom, att) in enumerate(DOMANDE, 1):
        log(f"── domanda {n} · {len(SEEDS)} seed ({' '.join(map(str, SEEDS))}) ────────────────────────────")
        g = 0
        for seed in SEEDS:
            if scaduto():
                interrotta = True
                break
            e = chiedi(dom, att, seed)
            fatte += 1
            muti += e == "MUTO"
            g += e == "OK"
        log(f"  → domanda {n}: giuste {g}/{len(SEEDS)}")
        tassi += f" D{n}={g}/{len(SEEDS)}"
        giusti_tot += g
        if interrotta:
            break
    TOT = 3 * len(SEEDS)
    log()
    log("── verdetto ───────────────────────────────────────────────")
    if interrotta:
        V = f"⛔ INTERROTTA al limite orario {LIMITE}: fatte {fatte}/{TOT} richieste — esito PARZIALE, non un verdetto"
        rc = 5
    elif muti > 0:
        V = f"⛔ MUTO su {muti} risposte su {TOT} — NON portarlo al banco: il banco non se ne accorgerebbe"
        rc = 2
    elif giusti_tot == TOT:
        V = f"✅ PARLA e risponde giusto {TOT}/{TOT} su tutti i seed — via libera al banco"
        rc = 0
    else:
        V = (f"⚠️ parla: giuste {giusti_tot}/{TOT} ({tassi} ) — il tasso va letto ACCANTO a quello del modello base "
             f"sugli stessi seed (riferimento base v20-downaggr3 22/9: D1=3/7); da solo non e' un verdetto")
        rc = 1
    log(V)
    with open(ESITO, "w", encoding="utf-8") as f:
        f.write(f"{NOME}  thinking={str(THINK).lower()}  seed={' '.join(map(str, SEEDS))}  "
                f"{datetime.datetime.now():%d/%m %H:%M}\n{V}\ngiuste {giusti_tot}/{TOT} ·{tassi} · mute {muti}/{TOT}\n")
    log(f"esito in {ESITO}")
finally:
    srv.terminate()
    try:
        srv.wait(timeout=30)
    except Exception:
        srv.kill()
    srv_log.close()
sys.exit(rc)
