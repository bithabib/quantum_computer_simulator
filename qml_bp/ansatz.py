"""Random variational circuits and their gradient-variance (trainability) label.

A *spec* fully describes a circuit family (qubits, layers, ansatz, entanglement,
cost locality). From a spec we build a hardware-efficient ansatz whose structure
is fixed but whose rotation angles are free parameters. The trainability label is

    Var_theta[ dC/dtheta_k ]

estimated over many random parameter vectors theta, where C(theta) = <psi|O|psi>
is the cost (expectation of a Pauli-Z string), and the gradient is computed with
the parameter-shift rule. A vanishing variance that shrinks with qubit count is
the signature of a barren plateau.
"""

import math

import numpy as np

from qsim import QuantumCircuit

# ---- spec space -----------------------------------------------------------

ANSATZ_TYPES = ["ry", "random_pauli"]          # single-qubit rotation scheme
ENTANGLE_PATTERNS = ["linear", "circular", "all_to_all"]
ENTANGLER_GATES = ["cz", "cx"]

# Feature columns handed to the ML model (order matters for downstream code).
FEATURE_COLUMNS = [
    "n_qubits",
    "n_layers",
    "n_params",
    "ansatz_type",       # index into ANSATZ_TYPES
    "entangle_pattern",  # index into ENTANGLE_PATTERNS
    "entangler_gate",    # index into ENTANGLER_GATES
    "cost_global",       # 0 = local Z_0, 1 = global Z...Z
    "n_entanglers",
    "depth_ratio",       # n_layers / n_qubits
]


class CircuitSpec:
    """A hardware-efficient ansatz family + a cost observable."""

    def __init__(self, n_qubits, n_layers, ansatz_type, entangle_pattern,
                 entangler_gate, cost_global, seed):
        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)
        self.ansatz_type = ansatz_type
        self.entangle_pattern = entangle_pattern
        self.entangler_gate = entangler_gate
        self.cost_global = bool(cost_global)
        self.seed = int(seed)

        rng = np.random.default_rng(seed)
        # Fixed rotation axis per (layer, qubit) position for this spec.
        if ansatz_type == "ry":
            self._axes = [["ry"] * self.n_qubits for _ in range(self.n_layers)]
        else:  # random_pauli
            self._axes = [
                [rng.choice(["rx", "ry", "rz"]) for _ in range(self.n_qubits)]
                for _ in range(self.n_layers)
            ]
        self.n_params = self.n_qubits * self.n_layers
        self._pairs = self._entangling_pairs()
        # cost = Z on qubit 0 (local) or Z on every qubit (global)
        self.cost_qubits = list(range(self.n_qubits)) if self.cost_global else [0]
        # gradient measured w.r.t. a rotation in the middle of the circuit
        self.target_param = (self.n_layers // 2) * self.n_qubits + 0

    def _entangling_pairs(self):
        n = self.n_qubits
        if n < 2:
            return []
        if self.entangle_pattern == "linear":
            return [(q, q + 1) for q in range(n - 1)]
        if self.entangle_pattern == "circular":
            return [(q, (q + 1) % n) for q in range(n)]
        # all_to_all
        return [(a, b) for a in range(n) for b in range(a + 1, n)]

    @property
    def n_entanglers(self):
        return len(self._pairs) * self.n_layers

    # -- circuit construction / evaluation ------------------------------
    def build(self, theta):
        """Instantiate the qsim circuit for a given parameter vector theta."""
        qc = QuantumCircuit(self.n_qubits)
        p = 0
        for layer in range(self.n_layers):
            for q in range(self.n_qubits):
                getattr(qc, self._axes[layer][q])(float(theta[p]), q)
                p += 1
            for (a, b) in self._pairs:
                getattr(qc, self.entangler_gate)(a, b)
        return qc

    def cost(self, theta):
        sv = self.build(theta).statevector()
        return _z_string_expectation(sv.data, self.n_qubits, self.cost_qubits)

    def grad_target(self, theta):
        """Parameter-shift gradient of the cost w.r.t. the target parameter."""
        k = self.target_param
        tp = np.array(theta, dtype=float); tp[k] += math.pi / 2
        tm = np.array(theta, dtype=float); tm[k] -= math.pi / 2
        return 0.5 * (self.cost(tp) - self.cost(tm))

    def features(self):
        return {
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "n_params": self.n_params,
            "ansatz_type": ANSATZ_TYPES.index(self.ansatz_type),
            "entangle_pattern": ENTANGLE_PATTERNS.index(self.entangle_pattern),
            "entangler_gate": ENTANGLER_GATES.index(self.entangler_gate),
            "cost_global": int(self.cost_global),
            "n_entanglers": self.n_entanglers,
            "depth_ratio": self.n_layers / self.n_qubits,
        }


def _z_string_expectation(data, n, qubits):
    """<Z_{q1} Z_{q2} ...> for a statevector (qubit 0 = most significant bit)."""
    probs = np.abs(data) ** 2
    idx = np.arange(data.shape[0])
    sign = np.ones(data.shape[0])
    for q in qubits:
        bit = (idx >> (n - 1 - q)) & 1
        sign = sign * np.where(bit == 1, -1.0, 1.0)
    return float(np.sum(probs * sign))


# ---- sampling + labelling -------------------------------------------------

def sample_spec(rng, qubit_range, layer_range):
    """Draw a random CircuitSpec within the given ranges."""
    n_qubits = int(rng.integers(qubit_range[0], qubit_range[1] + 1))
    n_layers = int(rng.integers(layer_range[0], layer_range[1] + 1))
    return CircuitSpec(
        n_qubits=n_qubits,
        n_layers=n_layers,
        ansatz_type=rng.choice(ANSATZ_TYPES),
        entangle_pattern=rng.choice(ENTANGLE_PATTERNS),
        entangler_gate=rng.choice(ENTANGLER_GATES),
        cost_global=bool(rng.integers(0, 2)),
        seed=int(rng.integers(0, 2 ** 31)),
    )


def compute_datapoint(spec, samples, rng):
    """Return a feature+label dict for one spec.

    Label = variance over `samples` random parameter vectors of the
    parameter-shift gradient of the cost w.r.t. the target parameter.
    """
    grads = np.empty(samples)
    for i in range(samples):
        theta = rng.uniform(0.0, 2.0 * math.pi, size=spec.n_params)
        grads[i] = spec.grad_target(theta)

    var = float(np.var(grads))
    row = spec.features()
    row["grad_mean"] = float(np.mean(grads))
    row["grad_var"] = var
    # log10 of the variance is the natural regression target (clip the floor so
    # exact-zero variances from tiny samples stay finite).
    row["log_grad_var"] = math.log10(max(var, 1e-18))
    row["samples"] = samples
    return row
