"""The user-facing QuantumCircuit class.

Gate methods record instructions; simulation happens lazily when the user asks
for a statevector or probabilities. Methods return ``self`` so calls can be
chained: ``qc.h(0).cx(0, 1)``.
"""

MAX_QUBITS = 12  # keep statevectors small enough to simulate instantly


class QuantumCircuit:
    """A quantum circuit over ``num_qubits`` qubits, all starting in |0>."""

    def __init__(self, num_qubits):
        if not isinstance(num_qubits, int):
            raise TypeError("num_qubits must be an int")
        if num_qubits < 1 or num_qubits > MAX_QUBITS:
            raise ValueError("num_qubits must be between 1 and %d" % MAX_QUBITS)
        self.num_qubits = num_qubits
        self.instructions = []  # list of (name, (qubits...), (params...))

    # -- helpers ---------------------------------------------------------
    def _q(self, *qubits):
        for q in qubits:
            if not isinstance(q, int) or q < 0 or q >= self.num_qubits:
                raise ValueError(
                    "qubit index %r out of range 0..%d" % (q, self.num_qubits - 1)
                )
        if len(set(qubits)) != len(qubits):
            raise ValueError("gate qubits must be distinct: %r" % (qubits,))
        return qubits

    def _add(self, name, qubits, params=()):
        self.instructions.append((name, tuple(qubits), tuple(params)))
        return self

    # -- single-qubit gates ---------------------------------------------
    def i(self, q):
        return self._add("i", self._q(q))

    def x(self, q):
        return self._add("x", self._q(q))

    def y(self, q):
        return self._add("y", self._q(q))

    def z(self, q):
        return self._add("z", self._q(q))

    def h(self, q):
        return self._add("h", self._q(q))

    def s(self, q):
        return self._add("s", self._q(q))

    def sdg(self, q):
        return self._add("sdg", self._q(q))

    def t(self, q):
        return self._add("t", self._q(q))

    def tdg(self, q):
        return self._add("tdg", self._q(q))

    def rx(self, theta, q):
        return self._add("rx", self._q(q), (float(theta),))

    def ry(self, theta, q):
        return self._add("ry", self._q(q), (float(theta),))

    def rz(self, theta, q):
        return self._add("rz", self._q(q), (float(theta),))

    def p(self, lam, q):
        return self._add("p", self._q(q), (float(lam),))

    # -- multi-qubit gates ----------------------------------------------
    def cx(self, control, target):
        return self._add("cx", self._q(control, target))

    cnot = cx

    def cz(self, control, target):
        return self._add("cz", self._q(control, target))

    def cp(self, lam, control, target):
        return self._add("cp", self._q(control, target), (float(lam),))

    def swap(self, a, b):
        return self._add("swap", self._q(a, b))

    def ccx(self, c1, c2, target):
        return self._add("ccx", self._q(c1, c2, target))

    toffoli = ccx

    # -- results ---------------------------------------------------------
    def statevector(self):
        from .simulator import simulate

        return simulate(self)

    def probabilities(self):
        return self.statevector().probabilities()

    def draw(self):
        """A tiny text description of the circuit (for print())."""
        lines = ["QuantumCircuit(%d qubits)" % self.num_qubits]
        for name, qubits, params in self.instructions:
            arg = ", ".join(str(q) for q in qubits)
            if params:
                arg = ", ".join("%.4g" % p for p in params) + " -> " + arg
            lines.append("  %-5s %s" % (name, arg))
        return "\n".join(lines)

    def __repr__(self):
        return self.draw()
