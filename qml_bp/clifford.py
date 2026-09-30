"""Gradient variance at any qubit count by Clifford (Pauli-propagation) sampling.

For the circuits of :mod:`qml_bp.ansatz` every rotation angle appears exactly
once and every generator is a Pauli, so the cost C(theta) is a trigonometric
polynomial of degree one in each angle and the squared gradient has degree at
most two.  A four-point quadrature is exact for such polynomials:

    E_{theta ~ U[0,2pi)^m} [ (dC/dtheta_k)^2 ]
        = E_{theta in {0, pi/2, pi, 3pi/2}^m} [ (dC/dtheta_k)^2 ].

On that grid every rotation is a Clifford gate, so in the Heisenberg picture
the observable stays a single Pauli string, <0|P|0> is 0 or +-1, and
dC/dtheta_k is 0 or +-1.  Because E[dC/dtheta_k] = 0, the variance of the
gradient is simply the probability that it is non-zero:

    Var[dC/dtheta_k] = P_grid( dC/dtheta_k != 0 ).

Signs never matter, so only the Pauli *type* is tracked: two bits (x, z) per
qubit.  Samples are packed 64 per machine word and all parameters are handled
in one backward sweep, so the cost is O(gates * parameters * samples / 64) bit
operations and is independent of 2^n.

What this estimator cannot do is resolve a variance much smaller than
1 / (samples * parameters): exponentially small variances need exponentially
many samples.  ``estimate`` therefore samples adaptively and reports whether
the result is resolved.
"""

import numpy as np

ONES = np.uint64(0xFFFFFFFFFFFFFFFF)


def gate_list(spec):
    """Forward gate list: ('r', qubit, axis, param) and ('cx'|'cz', a, b)."""
    gates = []
    p = 0
    for layer in range(spec.n_layers):
        for q in range(spec.n_qubits):
            gates.append(("r", q, spec._axes[layer][q][1], p))  # 'rx' -> 'x'
            p += 1
        for (a, b) in spec.pairs_for_layer(layer):
            gates.append((spec.entangler_gate, a, b))
    return gates


def _sweep(spec, gates, W, rng):
    """One batch of 64*W grid samples.  Returns hits[k] = number of samples in
    which dC/dtheta_k is non-zero, for every parameter k."""
    n, m = spec.n_qubits, spec.n_params
    Xm = np.zeros((n, W), dtype=np.uint64)          # observable, per sample
    Zm = np.zeros((n, W), dtype=np.uint64)
    for q in spec.cost_qubits:
        Zm[q] = ONES
    XD = np.zeros((n, m, W), dtype=np.uint64)       # derivative branch of row k
    ZD = np.zeros((n, m, W), dtype=np.uint64)
    alive = np.zeros((m, W), dtype=np.uint64)
    odd = rng.integers(0, 2 ** 64, size=(m, W), dtype=np.uint64)  # angle parity
    lo = m                                           # rows [lo, m) are active
    for g in reversed(gates):
        if g[0] == "cx":
            c, t = g[1], g[2]
            Xm[t] ^= Xm[c]; Zm[c] ^= Zm[t]
            if lo < m:
                XD[t, lo:] ^= XD[c, lo:]; ZD[c, lo:] ^= ZD[t, lo:]
        elif g[0] == "cz":
            a, b = g[1], g[2]
            Zm[a] ^= Xm[b]; Zm[b] ^= Xm[a]
            if lo < m:
                ZD[a, lo:] ^= XD[b, lo:]; ZD[b, lo:] ^= XD[a, lo:]
        else:
            _, q, axis, k = g
            par = odd[k]
            # rows already active: conjugate by this rotation
            if lo < m:
                if axis == "x":
                    XD[q, lo:] ^= ZD[q, lo:] & par
                elif axis == "z":
                    ZD[q, lo:] ^= XD[q, lo:] & par
                else:
                    f = (XD[q, lo:] ^ ZD[q, lo:]) & par
                    XD[q, lo:] ^= f; ZD[q, lo:] ^= f
            # new row k: the derivative is the branch the rotation did NOT take
            if axis == "x":
                ac = Zm[q].copy()
            elif axis == "z":
                ac = Xm[q].copy()
            else:
                ac = Xm[q] ^ Zm[q]
            XD[:, k, :] = Xm; ZD[:, k, :] = Zm
            fd = ac & ~par                           # even angle: derivative = sigma*P
            fm = ac & par                            # odd angle: main takes sigma*P
            if axis in ("x", "y"):
                XD[q, k] ^= fd; Xm[q] ^= fm
            if axis in ("z", "y"):
                ZD[q, k] ^= fd; Zm[q] ^= fm
            alive[k] = ac
            lo = min(lo, k)
    anyx = np.bitwise_or.reduce(XD, axis=0)          # (m, W)
    hit = alive & ~anyx
    return np.bitwise_count(hit).sum(axis=1).astype(np.int64)


def estimate(spec, rng, batch_words=64, min_samples=4096, max_samples=2 ** 20,
             target_hits=2000, resolved_hits=100):
    """Adaptive estimate of the mean gradient variance over ALL parameters.

    Samples in batches of ``64 * batch_words`` until the total number of
    non-zero gradient events reaches ``target_hits`` or ``max_samples`` is
    reached.  Returns a dict with the per-circuit summary.
    """
    gates = gate_list(spec)
    m = spec.n_params
    hits = np.zeros(m, dtype=np.int64)
    S = 0
    batch_means = []
    while True:
        h = _sweep(spec, gates, batch_words, rng)
        hits += h
        S += 64 * batch_words
        batch_means.append(h.sum() / (64.0 * batch_words * m))
        if S >= min_samples and (hits.sum() >= target_hits or S >= max_samples):
            break
    total = int(hits.sum())
    mean_all = total / (S * float(m))
    bm = np.array(batch_means)
    se = float(bm.std(ddof=1) / np.sqrt(len(bm))) if len(bm) > 1 else float("nan")
    return {
        "samples": int(S), "total_hits": total,
        "var_mean_all": mean_all,
        "log_var_all": float(np.log10(mean_all)) if total > 0 else float("nan"),
        "rel_se": float(se / mean_all) if total > 0 and se == se else float("nan"),
        "resolved": bool(total >= resolved_hits),
        "n_zero_hit": int((hits == 0).sum()),
        "frac_zero_hit": float((hits == 0).mean()),
        "resolution_floor": 1.0 / (S * float(m)),
        "per_param": hits / float(S),
    }
