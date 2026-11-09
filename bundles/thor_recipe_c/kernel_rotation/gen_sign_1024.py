"""
Vettore di segni FISSO S (±1, 1024 elementi) per il tipo TQ1_0R_1024
(rotazione Hadamard a blocco 1024 + 8 scale da 128, ricetta Bonsai 2 27B:
"Weight basis: block 1024" / "Weight format: ternary g128", Table 1 del
whitepaper PrismML). Stesso generatore di gen_sign.py (N=256): seed
deterministico 20260920, np.random.default_rng, rng.choice([-1,1]) — chi
rigenera con lo stesso seed ottiene BIT PER BIT lo stesso S. E' un vettore
NUOVO per un blocco NUOVO (non un piastrellamento di S256: un blocco 1024 ha
la sua S). Fonte di verita' UNICA: questo script scrive la lista JSON (letta
dal forge forgia_thor_v9_k1024.py) e i due letterali (C per ggml-quants.c,
GLSL per gli shader) che vanno copiati IDENTICI — mai trascritti a mano.
"""
import json
import numpy as np

SEED = 20260920
N = 1024


def genera() -> np.ndarray:
    rng = np.random.default_rng(SEED)
    return rng.choice([-1.0, 1.0], size=N).astype(np.float64)


if __name__ == "__main__":
    S = genera()
    assert S.shape == (N,)
    assert set(np.unique(S).tolist()) <= {-1.0, 1.0}
    print(f"seed={SEED}  n={N}  n_meno1={int((S<0).sum())}  n_piu1={int((S>0).sum())}")

    with open("sign_s1024.json", "w") as f:
        json.dump({"seed": SEED, "n": N, "s": S.astype(int).tolist()}, f)
    print("scritto sign_s1024.json")

    ivals = S.astype(int).tolist()
    with open("sign_s1024.c.txt", "w") as f:
        f.write("static const float ggml_tq1_0r_1024_sign1024[1024] = { "
                + ", ".join(f"{v}.0f" for v in ivals) + " };\n")
    print("scritto sign_s1024.c.txt")

    with open("sign_s1024.glsl.txt", "w") as f:
        f.write("const float S1024[1024] = float[1024]("
                + ", ".join(f"{v}.0" for v in ivals) + ");\n")
    print("scritto sign_s1024.glsl.txt")

    # Forma IMPACCHETTATA per gli shader (32 uint, bit=1 <=> segno -1): stessi
    # valori, letti con ((S1024_BITS[e>>5] >> (e&31)) & 1) — 128 byte invece di
    # 4 KiB di float, e indicizzazione dinamica su 32 elementi (economica).
    bits = [0] * 32
    for e, v in enumerate(ivals):
        if v < 0:
            bits[e >> 5] |= (1 << (e & 31))
    with open("sign_s1024.bits.glsl.txt", "w") as f:
        f.write("const uint S1024_BITS[32] = uint[32]("
                + ", ".join(f"0x{b:08x}u" for b in bits) + ");\n")
    # autoverifica: decodifica i bit e confronta con la lista
    for e, v in enumerate(ivals):
        assert ((bits[e >> 5] >> (e & 31)) & 1) == (1 if v < 0 else 0), e
    print("scritto sign_s1024.bits.glsl.txt (autoverificato contro la lista)")

    # Forma a SWITCH (lezione di A, 20/9 13:30, r2 da 27 a 1.7 tok/s): un array
    # costante indicizzato dinamicamente viene materializzato da glslang come
    # variabile Function-scope per invocazione (scratch memory). Uno switch su
    # letterali e' solo ALU. Stesse 32 parole, stesso bit: 1 <=> S[e] == -1.
    righe = ["uint tq1_0r_1024_sign_word(const uint w) {", "    switch (w) {"]
    for i, b in enumerate(bits[:-1]):
        righe.append(f"        case {i}u: return 0x{b:08x}u;")
    righe.append(f"        default: return 0x{bits[-1]:08x}u;   // w == 31")
    righe += ["    }", "}", ""]
    with open("sign_s1024.switch.glsl.txt", "w") as f:
        f.write("\n".join(righe))
    print("scritto sign_s1024.switch.glsl.txt")
