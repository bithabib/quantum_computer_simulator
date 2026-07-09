"""Statevector simulation for qsim circuits.

The simulator walks a circuit's recorded instruction list and applies each gate
to the statevector using tensor contractions (O(2^n) per gate), so it stays
fast for the small circuits used in the Quantum Lab challenges.
"""

import numpy as np

from . import gates


def _apply_1q(state, U, target, n):
    """Apply single-qubit unitary ``U`` to ``target`` on an n-qubit state."""
    t = state.reshape([2] * n)
    t = np.tensordot(U, t, axes=([1], [target]))  # new target axis lands at 0
    t = np.moveaxis(t, 0, target)
    return t.reshape(-1)


def _apply_controlled_1q(state, U, controls, target, n):
    """Apply ``U`` to ``target`` only where every control qubit is |1>."""
    t = state.reshape([2] * n)
    out = t.copy()

    sl = [slice(None)] * n
    for c in controls:
        sl[c] = 1
    sl = tuple(sl)

    block = t[sl]  # integer-indexing the controls drops those axes
    # target axis shifts left by the number of control axes that precede it
    tpos = target - sum(1 for c in controls if c < target)
    block = np.tensordot(U, block, axes=([1], [tpos]))
    block = np.moveaxis(block, 0, tpos)
    out[sl] = block
    return out.reshape(-1)


def _apply_swap(state, a, b, n):
    t = state.reshape([2] * n)
    t = np.swapaxes(t, a, b)
    return t.reshape(-1).copy()


# Map single-qubit gate names to a matrix (or a factory taking params).
_ONE_Q = {
    "i": lambda p: gates.I,
    "x": lambda p: gates.X,
    "y": lambda p: gates.Y,
    "z": lambda p: gates.Z,
    "h": lambda p: gates.H,
    "s": lambda p: gates.S,
    "sdg": lambda p: gates.SDG,
    "t": lambda p: gates.T,
    "tdg": lambda p: gates.TDG,
    "rx": lambda p: gates.rx(p[0]),
    "ry": lambda p: gates.ry(p[0]),
    "rz": lambda p: gates.rz(p[0]),
    "p": lambda p: gates.phase(p[0]),
}


def simulate(circuit):
    """Return the final :class:`Statevector` for a ``QuantumCircuit``."""
    n = circuit.num_qubits
    state = np.zeros(2 ** n, dtype=complex)
    state[0] = 1.0  # |00...0>

    for name, qubits, params in circuit.instructions:
        if name in _ONE_Q:
            state = _apply_1q(state, _ONE_Q[name](params), qubits[0], n)
        elif name == "cx":
            state = _apply_controlled_1q(state, gates.X, [qubits[0]], qubits[1], n)
        elif name == "cz":
            state = _apply_controlled_1q(state, gates.Z, [qubits[0]], qubits[1], n)
        elif name == "cp":
            state = _apply_controlled_1q(
                state, gates.phase(params[0]), [qubits[0]], qubits[1], n
            )
        elif name == "ccx":
            state = _apply_controlled_1q(
                state, gates.X, [qubits[0], qubits[1]], qubits[2], n
            )
        elif name == "swap":
            state = _apply_swap(state, qubits[0], qubits[1], n)
        else:  # pragma: no cover - guarded by QuantumCircuit
            raise ValueError("Unknown instruction: %r" % (name,))

    return Statevector(state, n)


class Statevector:
    """An immutable snapshot of an n-qubit statevector."""

    def __init__(self, data, num_qubits):
        self.data = np.asarray(data, dtype=complex)
        self.num_qubits = num_qubits

    def probabilities(self):
        """Measurement probabilities for each computational basis state."""
        return np.abs(self.data) ** 2

    def prob_dict(self):
        """Non-negligible outcomes as ``{"01": prob, ...}``."""
        probs = self.probabilities()
        out = {}
        for i, p in enumerate(probs):
            if p > 1e-9:
                out[format(i, "0%db" % self.num_qubits)] = float(p)
        return out

    def amplitudes(self):
        """Amplitudes as a list of (real, imag) pairs, ordered by basis state."""
        return [(float(a.real), float(a.imag)) for a in self.data]

    def __repr__(self):
        parts = []
        for i, a in enumerate(self.data):
            if abs(a) > 1e-9:
                label = format(i, "0%db" % self.num_qubits)
                parts.append("(%+.3f%+.3fj)|%s>" % (a.real, a.imag, label))
        return " + ".join(parts) if parts else "0"
