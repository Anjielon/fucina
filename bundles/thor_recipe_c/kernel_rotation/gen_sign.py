"""
Vettore di segni FISSO S (±1, 256 elementi) per la Randomized Hadamard
Transform (SpinQuant, arXiv:2405.16406; Bonsai 2 27B whitepaper sez. 2.4:
R = (1/sqrt(n)) H_n S). Seed deterministico e documentato: 20260920 (data di
oggi, AAAAMMGG) — chiunque rigeneri questo file con lo stesso seed ottiene
BIT PER BIT lo stesso S. Fonte di verita' UNICA: questo script genera sia
la lista JSON (letta dal forge Python) sia i due letterali (C e GLSL) che
vanno copiati IDENTICI in ggml-quants.c e mul_mat_vec_tq1_0r2.comp — mai
trascritti a mano (rischio di errore di trascrizione su 256 valori).
"""
import json
import numpy as np

SEED = 20260920
N = 256


def genera() -> np.ndarray:
    rng = np.random.default_rng(SEED)
    return rng.choice([-1.0, 1.0], size=N).astype(np.float64)


if __name__ == "__main__":
    S = genera()
    assert S.shape == (N,)
    assert set(np.unique(S).tolist()) <= {-1.0, 1.0}
    print(f"seed={SEED}  n_meno1={int((S<0).sum())}  n_piu1={int((S>0).sum())}")

    out_json = {"seed": SEED, "n": N, "s": S.astype(int).tolist()}
    with open("sign_s256.json", "w") as f:
        json.dump(out_json, f)
    print("scritto sign_s256.json")

    ivals = S.astype(int).tolist()
    c_lit = ", ".join(f"{v}.0f" for v in ivals)
    with open("sign_s256.c.txt", "w") as f:
        f.write(f"static const float ggml_tq1_0r_sign256[256] = {{ {c_lit} }};\n")
    print("scritto sign_s256.c.txt")

    glsl_lit = ", ".join(f"{v}.0" for v in ivals)
    with open("sign_s256.glsl.txt", "w") as f:
        f.write(f"const float S256[256] = float[256]({glsl_lit});\n")
    print("scritto sign_s256.glsl.txt")
