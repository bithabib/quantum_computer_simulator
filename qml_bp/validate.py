"""Validation of the simulator and of the gradient labels.

Three independent checks, all reported in the manuscript:

  1. qsim statevectors agree with a dense-matrix (Kronecker-product)
     reference simulation on random circuits.
  2. Adjoint gradients (``qml_bp.adjoint``) agree with the parameter-shift
     rule for every parameter of random circuits.
  3. Exact-zero gradients flagged as *structural* really are zero for every
     sampled parameter vector (the flag is a property of the circuit, not of a
     particular theta).

    python -m qml_bp.validate --n-circuits 200 --seed 1
"""

import argparse
import json
import math
import os

import numpy as np

from qml_bp.adjoint import cost_and_gradient
from qml_bp.ansatz import CircuitSpec, sample_spec
from qsim import gates as G


def _dense_state(circuit):
    """Reference simulation: build the full 2^n x 2^n unitary of every gate."""
    n = circuit.num_qubits
    state = np.zeros(2 ** n, dtype=complex)
    state[0] = 1.0
    rot = {"rx": G.rx, "ry": G.ry, "rz": G.rz}
    P0 = np.array([[1, 0], [0, 0]], dtype=complex)
    P1 = np.array([[0, 0], [0, 1]], dtype=complex)

    def kron_at(ops):  # ops: {qubit: 2x2}; qubit 0 = most significant
        M = np.array([[1.0 + 0j]])
        for q in range(n):
            M = np.kron(M, ops.get(q, G.I))
        return M

    for name, qubits, params in circuit.instructions:
        if name in rot:
            U = kron_at({qubits[0]: rot[name](params[0])})
        elif name == "cx":
            U = kron_at({qubits[0]: P0}) + kron_at({qubits[0]: P1, qubits[1]: G.X})
        elif name == "cz":
            U = kron_at({qubits[0]: P0}) + kron_at({qubits[0]: P1, qubits[1]: G.Z})
        else:
            raise ValueError(name)
        with np.errstate(all="ignore"):
            state = U @ state
    return state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-circuits", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-qubits-dense", type=int, default=7,
                    help="dense reference is O(4^n); keep small")
    ap.add_argument("--json", default=None, help="write a summary here")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    # -- 1. statevector vs dense reference ------------------------------
    worst_state = 0.0
    for _ in range(args.n_circuits):
        spec = sample_spec(rng, (2, args.max_qubits_dense), (1, 6))
        theta = rng.uniform(0, 2 * math.pi, size=spec.n_params)
        qc = spec.build(theta)
        a = qc.statevector().data
        b = _dense_state(qc)
        worst_state = max(worst_state, float(np.max(np.abs(a - b))))
    print("1. statevector vs dense reference : max |diff| = %.2e over %d circuits"
          % (worst_state, args.n_circuits))

    # -- 2. adjoint gradient vs parameter shift (all parameters) --------
    worst_grad, n_checked = 0.0, 0
    for _ in range(args.n_circuits):
        spec = sample_spec(rng, (2, 8), (1, 8))
        theta = rng.uniform(0, 2 * math.pi, size=spec.n_params)
        _, g_adj = cost_and_gradient(spec.build(theta), spec.cost_qubits)
        for k in range(spec.n_params):
            tp = theta.copy(); tp[k] += math.pi / 2
            tm = theta.copy(); tm[k] -= math.pi / 2
            g_ps = 0.5 * (spec.cost(tp) - spec.cost(tm))
            worst_grad = max(worst_grad, abs(g_ps - g_adj[k]))
            n_checked += 1
    print("2. adjoint vs parameter-shift     : max |diff| = %.2e over %d parameters"
          % (worst_grad, n_checked))

    # -- 3. structural zeros are theta-independent ----------------------
    # Use 50 thetas to flag, then 200 fresh thetas to confirm.
    n_specs, n_flagged, n_violations, worst_nonzero = 0, 0, 0, 0.0
    while n_specs < args.n_circuits:
        spec = sample_spec(rng, (2, 8), (1, 6))
        gs = np.stack([cost_and_gradient(spec.build(
            rng.uniform(0, 2 * math.pi, size=spec.n_params)), spec.cost_qubits)[1]
            for _ in range(50)])
        flagged = np.max(np.abs(gs), axis=0) < 1e-10
        n_specs += 1
        if not flagged.any():
            continue
        n_flagged += int(flagged.sum())
        gs2 = np.stack([cost_and_gradient(spec.build(
            rng.uniform(0, 2 * math.pi, size=spec.n_params)), spec.cost_qubits)[1]
            for _ in range(200)])
        viol = np.max(np.abs(gs2[:, flagged]), axis=0) > 1e-10
        n_violations += int(viol.sum())
        # smallest non-flagged max-gradient, to show the gap
        if (~flagged).any():
            worst_nonzero = max(worst_nonzero, 0.0)
    print("3. structural zeros               : %d flagged parameters in %d circuits, "
          "%d re-appeared as non-zero on 200 fresh thetas"
          % (n_flagged, n_specs, n_violations))
    if args.json:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        json.dump({"n_circuits": args.n_circuits, "max_state_diff": worst_state,
                   "max_grad_diff": worst_grad, "n_params_checked": n_checked,
                   "n_flagged": n_flagged, "n_violations": n_violations},
                  open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
