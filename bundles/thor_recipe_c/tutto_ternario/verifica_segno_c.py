# Confronta S1024 del codec numpy col letterale C (albero k1024 e sorgente del banco MMLU del Dell).
import re, sys
import numpy as np
sys.path.insert(0, ".")
from tq1_0r_1024_codec import S1024
for f in sys.argv[1:]:
    t = open(f).read()
    m = re.search(r"ggml_tq1_0r_1024_sign1024\s*\[\s*1024\s*\]\s*=\s*\{([^}]*)\}", t)
    if not m:
        print(f, "letterale NON trovato"); continue
    v = np.array([float(x.strip().rstrip("f")) for x in m.group(1).split(",") if x.strip()])
    print(f, "n", v.size, "diff vs codec", int((v != S1024).sum()))
