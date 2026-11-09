"""Ternary {-s,0,+s} with ONE fp16 scale per group (TQ1_0 rule: group 256) for VECTORS (norm gains, ssm_a, dt_bias,
k-gain): exact optimum per group (sort |v|, try every support size, s = mean |v| on the support, rounded to fp16, pick the
min squared error, optionally weighted). Used by c0t_* scripts; pure numpy fp64."""
import numpy as np


def ternario_gruppo(v, w=None):
    v = np.asarray(v, np.float64); w = np.ones_like(v) if w is None else np.asarray(w, np.float64)
    a = np.abs(v); o = np.argsort(-a)
    best = (np.sum(w * v * v), 0.0, np.zeros_like(v))
    for k in range(1, len(v) + 1):
        idx = o[:k]
        s = float(np.sum(w[idx] * a[idx]) / np.sum(w[idx]))
        s = float(np.float16(s))
        q = np.zeros_like(v); q[idx] = np.sign(v[idx])
        e = float(np.sum(w * (v - s * q) ** 2))
        if e < best[0]:
            best = (e, s, q)
    return best[1], best[2]


def ternario(v, gruppo=256, w=None):
    v = np.asarray(v, np.float64); out = np.zeros_like(v); S = []; Q = np.zeros(len(v), np.int8)
    for i in range(0, len(v), gruppo):
        s, q = ternario_gruppo(v[i:i + gruppo], None if w is None else w[i:i + gruppo])
        out[i:i + gruppo] = s * q; S.append(s); Q[i:i + gruppo] = q
    return out, np.array(S), Q
