"""Exact gradients of a Pauli-string cost w.r.t. *every* rotation angle.

Reverse-mode (adjoint) differentiation for the hardware-efficient circuits in
:mod:`qml_bp.ansatz`.  For a circuit ``|psi> = U_N ... U_1 |0>`` and cost
``C = <psi|O|psi>`` with ``O`` a Pauli-Z string, the derivative w.r.t. the
angle of a rotation ``U_i = exp(-i theta_i G_i / 2)`` is

    dC/dtheta_i = Im <lambda_i | G_i | psi_i>,

where ``psi_i = U_i ... U_1 |0>`` and ``lambda_i = U_{i+1}^dag ... U_N^dag O psi_N``.
Walking the circuit backwards once therefore yields all ``nL`` derivatives at the
cost of roughly three forward simulations, instead of ``2 nL`` simulations with
the parameter-shift rule.  The two agree to floating-point precision; see
``qml_bp/validate.py``.
"""

from functools import lru_cache

import numpy as np

from qsim import gates
from qsim.simulator import _apply_1q

_PAULI = {"rx": gates.X, "ry": gates.Y, "rz": gates.Z}
_ROT = {"rx": gates.rx, "ry": gates.ry, "rz": gates.rz}


@lru_cache(maxsize=None)
def _ctrl_masks(n, c, t):
    """Basis-state index sets for a controlled gate (qubit 0 = MSB).

    Returns (i11, i10): indices with control=1,target=1 and control=1,target=0.
    """
    idx = np.arange(2 ** n)
    cbit = (idx >> (n - 1 - c)) & 1
    tbit = (idx >> (n - 1 - t)) & 1
    return idx[(cbit == 1) & (tbit == 1)], idx[(cbit == 1) & (tbit == 0)]


def _apply_cz(state, c, t, n):
    i11, _ = _ctrl_masks(n, c, t)
    state[i11] *= -1.0
    return state


def _apply_cx(state, c, t, n):
    i11, i10 = _ctrl_masks(n, c, t)
    tmp = state[i10].copy()
    state[i10] = state[i11]
    state[i11] = tmp
    return state


def _apply(state, name, qubits, params, n, dagger=False):
    """Apply one qsim instruction (or its inverse) to ``state``.

    Rotations return a new array; CX/CZ act in place on ``state`` (which is
    always an array owned by the caller here) and are self-inverse.
    """
    if name in _ROT:
        th = -params[0] if dagger else params[0]
        return _apply_1q(state, _ROT[name](th), qubits[0], n)
    if name == "cx":
        return _apply_cx(state, qubits[0], qubits[1], n)
    if name == "cz":
        return _apply_cz(state, qubits[0], qubits[1], n)
    raise ValueError("adjoint gradient supports rx/ry/rz/cx/cz only, got %r" % name)


def _apply_z_string(state, n, qubits):
    """Return O|state> for O = product of Z on ``qubits`` (qubit 0 = MSB)."""
    idx = np.arange(state.shape[0])
    sign = np.ones(state.shape[0])
    for q in qubits:
        bit = (idx >> (n - 1 - q)) & 1
        sign = sign * np.where(bit == 1, -1.0, 1.0)
    return state * sign


def cost_and_gradient(circuit, cost_qubits):
    """Return ``(C, grad)`` with ``grad[k]`` = dC/dtheta_k for the k-th rotation
    in instruction order.  ``circuit`` is a :class:`qsim.QuantumCircuit` built
    from rx/ry/rz/cx/cz instructions."""
    n = circuit.num_qubits
    instr = circuit.instructions
    psi = np.zeros(2 ** n, dtype=complex)
    psi[0] = 1.0
    for name, qubits, params in instr:
        psi = _apply(psi, name, qubits, params, n)

    lam = _apply_z_string(psi, n, cost_qubits)
    cost = float(np.real(np.vdot(psi, lam)))

    n_rot = sum(1 for name, _, _ in instr if name in _ROT)
    grad = np.empty(n_rot)
    k = n_rot - 1
    for name, qubits, params in reversed(instr):
        if name in _ROT:
            g_psi = _apply_1q(psi, _PAULI[name], qubits[0], n)
            grad[k] = float(np.imag(np.vdot(lam, g_psi)))
            k -= 1
        psi = _apply(psi, name, qubits, params, n, dagger=True)
        lam = _apply(lam, name, qubits, params, n, dagger=True)
    return cost, grad
