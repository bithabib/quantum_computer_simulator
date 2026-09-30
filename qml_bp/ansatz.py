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
from qml_bp.adjoint import cost_and_gradient

# ---- spec space -----------------------------------------------------------

ANSATZ_TYPES = ["ry", "random_pauli"]          # single-qubit rotation scheme
# The first three patterns form the main dataset.  "brickwork" (even pairs on
# even layers, odd pairs on odd layers) and "star" (qubit 0 to every other
# qubit) are used for the unseen-pattern transfer experiment (qml_bp.lopo).
ENTANGLE_PATTERNS = ["linear", "circular", "all_to_all", "brickwork", "star"]
DEFAULT_PATTERNS = ENTANGLE_PATTERNS[:3]
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
    "cost_weight_eff",   # weight of the observable after the last entangling layer
    "cone_qubits",       # qubits in the backward causal cone of the observable
    "cone_frac",         # fraction of parameters inside the causal cone
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
        if self.entangle_pattern == "all_to_all":
            return [(a, b) for a in range(n) for b in range(a + 1, n)]
        if self.entangle_pattern == "star":
            return [(0, q) for q in range(1, n)]
        if self.entangle_pattern == "brickwork":
            return [(q, q + 1) for q in range(0, n - 1)]  # union over layers
        raise ValueError("unknown entangle_pattern %r" % self.entangle_pattern)

    def effective_weight(self):
        """Weight of the measured Z-string once the last entangling layer is
        absorbed into it.  The last layer of CX/CZ gates sits between the final
        rotations and the measurement and is Clifford, so it maps the nominal
        observable O to another Pauli-Z string O_eff = U_ent^dag O U_ent.  CZ is
        diagonal and leaves O unchanged; CX maps Z_target -> Z_control Z_target.
        A nominally local cost can therefore be effectively global, and vice
        versa."""
        Z = set(self.cost_qubits)
        if self.entangler_gate == "cx":
            for (c, t) in reversed(self.pairs_for_layer(self.n_layers - 1)):
                if t in Z:
                    Z ^= {c}
        return len(Z)

    # Pauli types as (x, z) bit pairs: I=0, X=1, Z=2, Y=3.  The causal cone is
    # computed by propagating the SET of possible types of every qubit backward
    # through the circuit (a per-qubit relaxation: correlations between qubits
    # are dropped, so the cone is an outer bound).  A rotation about axis a
    # mixes the two types that anticommute with sigma_a; for the random-Pauli
    # scheme any axis may occur, so any non-identity type may become any other.
    _MIX = {"ry": {0: {0}, 3: {3}, 1: {1, 2}, 2: {1, 2}},
            "any": {0: {0}, 1: {1, 2, 3}, 2: {1, 2, 3}, 3: {1, 2, 3}}}
    _ANTI = {"ry": {1, 2}, "any": {1, 2, 3}}

    def causal_cone(self):
        """(cone_qubits, cone_params): the number of qubits whose Pauli type can
        be non-identity when the observable is propagated back to the input,
        and the number of rotations that can have a non-zero gradient.  These
        are closed-form functions of the design parameters (for random-Pauli
        axes the generic case is assumed)."""
        axis = "ry" if self.ansatz_type == "ry" else "any"
        T = [{2} if q in self.cost_qubits else {0} for q in range(self.n_qubits)]
        n_par = 0
        for layer in range(self.n_layers - 1, -1, -1):
            for (a, b) in reversed(self.pairs_for_layer(layer)):
                na, nb = set(), set()
                for pa in T[a]:
                    for pb in T[b]:
                        xa, za, xb, zb = pa & 1, (pa >> 1) & 1, pb & 1, (pb >> 1) & 1
                        if self.entangler_gate == "cx":
                            xb ^= xa; za ^= zb
                        else:
                            za ^= xb; zb ^= xa
                        na.add(xa | (za << 1)); nb.add(xb | (zb << 1))
                T[a], T[b] = na, nb
            for q in range(self.n_qubits):
                if T[q] & self._ANTI[axis]:
                    n_par += 1
                T[q] = set().union(*[self._MIX[axis][t] for t in T[q]])
        return sum(1 for q in range(self.n_qubits) if T[q] != {0}), n_par

    def pairs_for_layer(self, layer):
        """Ordered (control, target) pairs applied in ``layer``."""
        if self.entangle_pattern == "brickwork":
            start = 0 if layer % 2 == 0 else 1
            return [(q, q + 1) for q in range(start, self.n_qubits - 1, 2)]
        return self._pairs

    @property
    def n_entanglers(self):
        return sum(len(self.pairs_for_layer(l)) for l in range(self.n_layers))

    # -- circuit construction / evaluation ------------------------------
    def build(self, theta):
        """Instantiate the qsim circuit for a given parameter vector theta."""
        qc = QuantumCircuit(self.n_qubits)
        p = 0
        for layer in range(self.n_layers):
            for q in range(self.n_qubits):
                getattr(qc, self._axes[layer][q])(float(theta[p]), q)
                p += 1
            for (a, b) in self.pairs_for_layer(layer):
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
            "cost_weight_eff": self.effective_weight(),
            "cone_qubits": self.causal_cone()[0],
            "cone_frac": self.causal_cone()[1] / self.n_params,
        }


def causal_cone(n_qubits, n_layers, ansatz_type, entangle_pattern, entangler_gate, cost_global):
    """Closed-form (cone_qubits, cone_frac) from the design parameters."""
    pat = entangle_pattern if isinstance(entangle_pattern, str) else ENTANGLE_PATTERNS[int(entangle_pattern)]
    gate = entangler_gate if isinstance(entangler_gate, str) else ENTANGLER_GATES[int(entangler_gate)]
    ans = ansatz_type if isinstance(ansatz_type, str) else ANSATZ_TYPES[int(ansatz_type)]
    s = CircuitSpec(int(n_qubits), int(n_layers), ans, pat, gate, bool(cost_global), 0)
    cq, cp = s.causal_cone()
    return cq, cp / s.n_params


def effective_weight(n_qubits, n_layers, entangle_pattern, entangler_gate, cost_global):
    """Closed-form effective observable weight from the design parameters
    (pattern and gate given by name or by index)."""
    pat = entangle_pattern if isinstance(entangle_pattern, str) else ENTANGLE_PATTERNS[int(entangle_pattern)]
    gate = entangler_gate if isinstance(entangler_gate, str) else ENTANGLER_GATES[int(entangler_gate)]
    return CircuitSpec(int(n_qubits), int(n_layers), "ry", pat, gate, bool(cost_global), 0).effective_weight()


# ---- pattern-agnostic descriptors of the entangling graph ----------------

GRAPH_FEATURE_COLUMNS = [
    "ent_per_layer",   # mean number of entangling gates per layer
    "ent_density",     # edges of the union graph / n(n-1)/2
    "ent_max_degree",  # max degree of the union graph
    "ent_diameter",    # diameter of the union graph (n if disconnected)
]


def entangling_graph_features(n_qubits, n_layers, entangle_pattern):
    """Closed-form descriptors of the entangling graph.  They are defined for
    any pattern, so a model trained on them can be asked about a pattern it
    has never seen (unlike the categorical ``entangle_pattern`` index)."""
    n = int(n_qubits)
    spec = CircuitSpec(n, int(n_layers), "ry", entangle_pattern, "cz", False, 0)
    edges = set()
    for l in range(spec.n_layers):
        for a, b in spec.pairs_for_layer(l):
            edges.add((min(a, b), max(a, b)))
    adj = {q: set() for q in range(n)}
    for a, b in edges:
        adj[a].add(b); adj[b].add(a)
    max_deg = max(len(v) for v in adj.values()) if n else 0
    # BFS diameter
    diam = 0
    for s0 in range(n):
        dist = {s0: 0}; frontier = [s0]
        while frontier:
            nxt = []
            for u in frontier:
                for v in adj[u]:
                    if v not in dist:
                        dist[v] = dist[u] + 1; nxt.append(v)
            frontier = nxt
        if len(dist) < n:
            diam = n; break
        diam = max(diam, max(dist.values()))
    n_possible = n * (n - 1) / 2 if n > 1 else 1.0
    return {
        "ent_per_layer": spec.n_entanglers / spec.n_layers,
        "ent_density": len(edges) / n_possible,
        "ent_max_degree": max_deg,
        "ent_diameter": diam,
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

def sample_spec(rng, qubit_range, layer_range, patterns=None):
    """Draw a random CircuitSpec within the given ranges."""
    patterns = list(patterns) if patterns else DEFAULT_PATTERNS
    n_qubits = int(rng.integers(qubit_range[0], qubit_range[1] + 1))
    n_layers = int(rng.integers(layer_range[0], layer_range[1] + 1))
    return CircuitSpec(
        n_qubits=n_qubits,
        n_layers=n_layers,
        ansatz_type=rng.choice(ANSATZ_TYPES),
        entangle_pattern=rng.choice(patterns),
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


# Parameters whose gradient never exceeds this over all sampled theta are
# *structural zeros*: the rotation commutes with everything between it and the
# observable (light cone / symmetry), so its gradient is identically zero.  A
# genuine barren-plateau variance at n <= 12 is many orders of magnitude larger.
STRUCTURAL_TOL = 1e-10

# Extra columns written by compute_datapoint_all (in addition to the legacy ones).
ALL_PARAM_COLUMNS = [
    "n_structural",    # number of structural-zero parameters
    "frac_structural", # n_structural / n_params
    "var_mean_all",    # mean over ALL parameters of Var[dC/dtheta_k]
    "var_mean_nz",     # mean over non-structural parameters (NaN if none)
    "var_median_nz",   # median over non-structural parameters
    "var_min_nz",      # smallest non-structural variance
    "var_max",         # largest variance over all parameters
    "log_var",         # regression label: log10(var_mean_nz)
]


def compute_datapoint_all(spec, samples, rng):
    """Feature+label dict for one spec using the gradient of *every* parameter.

    ``samples`` random parameter vectors are drawn; for each, the exact gradient
    w.r.t. all nL angles is obtained by adjoint differentiation.  The legacy
    single-parameter columns (grad_mean, grad_var, log_grad_var) are filled from
    the same gradient matrix so the two labels can be compared row by row.
    """
    grads = np.empty((samples, spec.n_params))
    for i in range(samples):
        theta = rng.uniform(0.0, 2.0 * math.pi, size=spec.n_params)
        _, grads[i] = cost_and_gradient(spec.build(theta), spec.cost_qubits)

    var = grads.var(axis=0)                      # population variance (ddof=0)
    structural = np.max(np.abs(grads), axis=0) < STRUCTURAL_TOL
    nz = ~structural

    row = spec.features()
    k = spec.target_param
    row["grad_mean"] = float(grads[:, k].mean())
    row["grad_var"] = float(var[k])
    row["log_grad_var"] = math.log10(max(float(var[k]), 1e-18))
    row["samples"] = samples

    row["n_structural"] = int(structural.sum())
    row["frac_structural"] = float(structural.mean())
    row["var_mean_all"] = float(var.mean())
    if nz.any():
        v = var[nz]
        row["var_mean_nz"] = float(v.mean())
        row["var_median_nz"] = float(np.median(v))
        row["var_min_nz"] = float(v.min())
        row["log_var"] = math.log10(float(v.mean()))
    else:
        row["var_mean_nz"] = row["var_median_nz"] = row["var_min_nz"] = float("nan")
        row["log_var"] = float("nan")
    row["var_max"] = float(var.max())
    return row
