"""
FIXED sign vector S (±1, 1024 elements) for the TQ1_0R_1024 type
(1024-block Hadamard rotation + 8 scales of 128, recipe of Bonsai 2 27B:
"Weight basis: block 1024" / "Weight format: ternary g128", Table 1 of the
PrismML whitepaper). Same generator as gen_sign.py (N=256): deterministic
seed 20260920, np.random.default_rng, rng.choice([-1,1]) — anyone who
regenerates with the same seed gets the same S BIT FOR BIT. It is a NEW
vector for a NEW block (not a tiling of S256: a 1024 block has its own S).
SINGLE source of truth: this script writes the JSON list (read by the k1024
forge) and the two literals (C for ggml-quants.c, GLSL for the shaders) that
must be copied IDENTICALLY — never transcribed by hand.
"""
import json
import numpy as np

SEED = 20260920
N = 1024


def generate() -> np.ndarray:
    rng = np.random.default_rng(SEED)
    return rng.choice([-1.0, 1.0], size=N).astype(np.float64)


if __name__ == "__main__":
    S = generate()
    assert S.shape == (N,)
    assert set(np.unique(S).tolist()) <= {-1.0, 1.0}
    print(f"seed={SEED}  n={N}  n_minus1={int((S<0).sum())}  n_plus1={int((S>0).sum())}")

    with open("sign_s1024.json", "w") as f:
        json.dump({"seed": SEED, "n": N, "s": S.astype(int).tolist()}, f)
    print("written sign_s1024.json")

    ivals = S.astype(int).tolist()
    with open("sign_s1024.c.txt", "w") as f:
        f.write("static const float ggml_tq1_0r_1024_sign1024[1024] = { "
                + ", ".join(f"{v}.0f" for v in ivals) + " };\n")
    print("written sign_s1024.c.txt")

    with open("sign_s1024.glsl.txt", "w") as f:
        f.write("const float S1024[1024] = float[1024]("
                + ", ".join(f"{v}.0" for v in ivals) + ");\n")
    print("written sign_s1024.glsl.txt")

    # PACKED form for the shaders (32 uints, bit=1 <=> sign -1): same values,
    # read with ((S1024_BITS[e>>5] >> (e&31)) & 1) — 128 bytes instead of
    # 4 KiB of floats, and dynamic indexing over 32 elements (cheap).
    bits = [0] * 32
    for e, v in enumerate(ivals):
        if v < 0:
            bits[e >> 5] |= (1 << (e & 31))
    with open("sign_s1024.bits.glsl.txt", "w") as f:
        f.write("const uint S1024_BITS[32] = uint[32]("
                + ", ".join(f"0x{b:08x}u" for b in bits) + ");\n")
    # self-check: decode the bits and compare with the list
    for e, v in enumerate(ivals):
        assert ((bits[e >> 5] >> (e & 31)) & 1) == (1 if v < 0 else 0), e
    print("written sign_s1024.bits.glsl.txt (self-checked against the list)")

    # SWITCH form (lesson from variant A, 2026-09-20, r2 from 27 to 1.7 tok/s): a dynamically
    # indexed constant array is materialized by glslang as a Function-scope variable
    # per invocation (scratch memory). A switch over literals is ALU only.
    # Same 32 words, same bits: 1 <=> S[e] == -1.
    rows = ["uint tq1_0r_1024_sign_word(const uint w) {", "    switch (w) {"]
    for i, b in enumerate(bits[:-1]):
        rows.append(f"        case {i}u: return 0x{b:08x}u;")
    rows.append(f"        default: return 0x{bits[-1]:08x}u;   // w == 31")
    rows += ["    }", "}", ""]
    with open("sign_s1024.switch.glsl.txt", "w") as f:
        f.write("\n".join(rows))
    print("written sign_s1024.switch.glsl.txt")
