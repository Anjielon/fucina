"""Sonda DIAGNOSTICA (24/9 22:1x): stessi server, parametri e domande di lancia_sonda, ma salva
i TESTI completi (contenuto + coda del ragionamento), per capire PERCHE' una risposta e' sbagliata.
Uso: sonda_testi.py GGUF INDICE_DOMANDA SEME [SEME...]   -> stampa e salva in stato/sonda_testi/"""
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import valida_finestra_8_12 as V  # noqa: E402

gguf, i_dom, semi = Path(sys.argv[1]), int(sys.argv[2]), [int(x) for x in sys.argv[3:]]
domanda, attesa = V.DOMANDE_SONDA[i_dom]
os.environ.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
os.environ.setdefault("DBUS_SESSION_BUS_ADDRESS", f"unix:path=/run/user/{os.getuid()}/bus")
subprocess.run(["systemctl", "--user", "stop", V.UNITA_SONDA], capture_output=True)
cmd = ["systemd-run", "--user", f"--unit={V.UNITA_SONDA}", "--collect", str(V.LLAMA_BIN_K1024), "--model", str(gguf),
       "--alias", "sonda", "-ngl", "999", "--ctx-size", "8192", "--parallel", "1", "--flash-attn", "on",
       "--cache-type-k", "q8_0", "--cache-type-v", "q8_0", "--host", "127.0.0.1", "--port", str(V.PORTA_SONDA),
       "--threads", "8", "--temp", "0.6", "--min-p", "0.03", "--top-p", "0.9", "--jinja", "--reasoning-budget", "4096"]
subprocess.run(cmd, env=dict(os.environ, LD_LIBRARY_PATH=str(V.LLAMA_BIN_K1024.parent)), check=True)
out = Path("@@ROOT:forge@@/stato/sonda_testi")
out.mkdir(parents=True, exist_ok=True)
try:
    for _ in range(90):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{V.PORTA_SONDA}/health", timeout=2) as r:
                if json.loads(r.read()).get("status") == "ok":
                    break
        except Exception:
            pass
        time.sleep(5)
    for seme in semi:
        corpo = json.dumps({"model": "s", "stream": False, "max_tokens": 6000, "temperature": 0.6, "seed": seme,
                            "messages": [{"role": "user", "content": domanda}],
                            "chat_template_kwargs": {"enable_thinking": True}}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{V.PORTA_SONDA}/v1/chat/completions", data=corpo,
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=300) as r:
            d = json.loads(r.read())
        msg = d["choices"][0]["message"]
        esito = V._chiedi.__wrapped__ if hasattr(V._chiedi, "__wrapped__") else None
        rec = {"gguf": gguf.name, "domanda": domanda, "attesa": attesa, "seme": seme,
               "contenuto": msg.get("content"), "ragionamento": msg.get("reasoning_content"),
               "token": d.get("usage", {})}
        (out / f"{gguf.stem}_d{i_dom}_s{seme}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1))
        rag = (msg.get("reasoning_content") or "")
        print(f"=== seme {seme} · risposta: {msg.get('content')!r} · token {d.get('usage', {}).get('completion_tokens')}")
        print("   ragionamento (ultimi 600 caratteri):", rag[-600:].replace("\n", " "))
finally:
    subprocess.run(["systemctl", "--user", "stop", V.UNITA_SONDA], capture_output=True)
    subprocess.run(["systemctl", "--user", "reset-failed", V.UNITA_SONDA], capture_output=True)
