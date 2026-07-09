"""qsim - a small, self-contained quantum computing library.

A pure-NumPy statevector simulator with a Qiskit-inspired API, built for the
Eshyana Quantum Lab. Users write real Python against ``QuantumCircuit`` and the
grader simulates the same circuit to check answers exactly.

Qubit convention
----------------
Qubit 0 is the *most significant* (left-most) bit, so the basis state of a
3-qubit register is written ``|q0 q1 q2>``. This "big-endian" ordering matches
how tensor products are written in most textbooks, which is friendlier for
learners. (Note: Qiskit uses the opposite, little-endian, ordering.)

Example
-------
    from qsim import QuantumCircuit

    qc = QuantumCircuit(2)
    qc.h(0)         # superposition on qubit 0
    qc.cx(0, 1)     # entangle -> Bell pair
    print(qc.statevector())
    print(qc.probabilities())
"""

from .circuit import QuantumCircuit
from .simulator import Statevector, simulate

__all__ = ["QuantumCircuit", "Statevector", "simulate"]
