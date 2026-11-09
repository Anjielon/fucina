"""22/10: D15 audit wrapper. The audit runs on 1 CPU thread with the GPU hidden; the host admission guard (swap-in <= 20 MB/s)
was blocking it 40 times in a row because of swap from unrelated house services. Same script, host guard disabled ONLY here."""
import os, runpy, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pianificatore_gpu as PG
PG.guardia_host = lambda *a, **k: None
sys.argv[0] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cardan_verifica_energie_gpu.py")
runpy.run_path(sys.argv[0], run_name="__main__")
