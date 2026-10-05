"""Two further circuit families, used to test how general the predictor is.

* **Shared parameters** (``TiedSpec``): the circuits of :mod:`qml_bp.ansatz`
  with ONE angle per layer shared by all qubits (as in QAOA-type and
  Hamiltonian-variational ansaetze).  Because an angle now appears n times,
  the cost is no longer of degree one in each angle and the Clifford
  four-point quadrature of :mod:`qml_bp.clifford` is not exact: for this family
  there is no cheap exact label beyond statevector reach.

* **Random entangling graphs** (``GraphSpec``): every pair of qubits carries an
  entangling gate with probability p (random orientation and order), with the
  same graph in every layer or a fresh one per layer.  There is no finite list
  of patterns, so the architecture is described by closed-form graph
  descriptors plus the effective observable weight and the causal cone.

Labels are computed as in the main dataset: exact adjoint gradients for 200
uniformly random parameter vectors, mean variance over the parameters that are
not structural zeros.
"""

import math

import numpy as np

from qsim import QuantumCircuit
from qml_bp.adjoint import cost_and_gradient
from qml_bp.ansatz import (ANSATZ_TYPES, ENTANGLER_GATES, FEATURE_COLUMNS,
                           STRUCTURAL_TOL, CircuitSpec, sample_spec)

LABEL_COLUMNS = ["n_structural", "frac_structural", "log_var", "samples"]

# ---- shared parameters ------------------------------------------------------
TIED_COLUMNS = list(FEATURE_COLUMNS) + ["n_shared_params"] + LABEL_COLUMNS


class TiedSpec:
    """A :class:`CircuitSpec` whose rotations share one angle per layer."""

    def __init__(self, base):
        self.base = base
        self.n_params = base.n_layers

    def build(self, phi):
        b = self.base
        qc = QuantumCircuit(b.n_qubits)
        for layer in range(b.n_layers):
            for q in range(b.n_qubits):
                getattr(qc, b._axes[layer][q])(float(phi[layer]), q)
            for (a, c) in b.pairs_for_layer(layer):
                getattr(qc, b.entangler_gate)(a, c)
        return qc

    def cost_and_gradient(self, phi):
        """Cost and its gradient w.r.t. the L shared angles (the gradient of a
        shared angle is the sum of the gradients of the gates that carry it)."""
        b = self.base
        c, g = cost_and_gradient(self.build(phi), b.cost_qubits)
        return c, g.reshape(b.n_layers, b.n_qubits).sum(axis=1)


def label_from_gradients(G):
    var = G.var(axis=0)
    st = np.max(np.abs(G), axis=0) < STRUCTURAL_TOL
    return {"n_structural": int(st.sum()), "frac_structural": float(st.mean()),
            "log_var": math.log10(float(var[~st].mean())) if (~st).any() else float("nan"),
            "samples": int(G.shape[0])}


def tied_datapoint(rng, qubit_range, layer_range, samples):
    spec = TiedSpec(sample_spec(rng, qubit_range, layer_range))
    G = np.empty((samples, spec.n_params))
    for i in range(samples):
        _, G[i] = spec.cost_and_gradient(rng.uniform(0.0, 2.0 * math.pi, spec.n_params))
    row = spec.base.features()
    row["n_shared_params"] = spec.n_params
    row.update(label_from_gradients(G))
    return row


# ---- random entangling graphs ---------------------------------------------
GRAPH_DESIGN = ["n_qubits", "n_layers", "ansatz_type", "entangler_gate", "cost_global",
                "resample", "p_edge"]
GRAPH_DESCRIPTORS = ["n_params", "n_entanglers", "ent_per_layer", "ent_density",
                     "ent_max_degree", "ent_mean_degree", "deg_q0", "ent_diameter",
                     "depth_ratio", "cost_weight_eff", "cone_qubits", "cone_frac"]
GRAPH_FEATURES = GRAPH_DESIGN + GRAPH_DESCRIPTORS
GRAPH_COLUMNS = GRAPH_FEATURES + LABEL_COLUMNS


class GraphSpec(CircuitSpec):
    """Hardware-efficient layers whose entangling pairs form a random graph."""

    def __init__(self, n, L, ansatz_type, entangler_gate, cost_global, p, resample, seed):
        self.n_qubits, self.n_layers = int(n), int(L)
        self.ansatz_type, self.entangler_gate = ansatz_type, entangler_gate
        self.cost_global = bool(cost_global)
        self.entangle_pattern = "random"
        self.seed, self.p, self.resample = int(seed), float(p), bool(resample)
        rng = np.random.default_rng(seed)
        if ansatz_type == "ry":
            self._axes = [["ry"] * self.n_qubits for _ in range(self.n_layers)]
        else:
            self._axes = [[rng.choice(["rx", "ry", "rz"]) for _ in range(self.n_qubits)]
                          for _ in range(self.n_layers)]

        def draw():
            pairs = [(a, b) if rng.random() < 0.5 else (b, a)
                     for a in range(self.n_qubits) for b in range(a + 1, self.n_qubits)
                     if rng.random() < self.p]
            rng.shuffle(pairs)
            return [tuple(int(x) for x in pr) for pr in pairs]

        g0 = draw()
        self._layer_pairs = [draw() if self.resample else g0 for _ in range(self.n_layers)]
        self._pairs = sorted({(min(a, b), max(a, b)) for lp in self._layer_pairs for a, b in lp})
        self.n_params = self.n_qubits * self.n_layers
        self.cost_qubits = list(range(self.n_qubits)) if self.cost_global else [0]
        self.target_param = (self.n_layers // 2) * self.n_qubits

    def pairs_for_layer(self, layer):
        return self._layer_pairs[layer]

    def features(self):
        n = self.n_qubits
        edges = set(self._pairs)
        adj = {q: set() for q in range(n)}
        for a, b in edges:
            adj[a].add(b); adj[b].add(a)
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
        cq, cp = self.causal_cone()
        return {
            "n_qubits": n, "n_layers": self.n_layers,
            "ansatz_type": ANSATZ_TYPES.index(self.ansatz_type),
            "entangler_gate": ENTANGLER_GATES.index(self.entangler_gate),
            "cost_global": int(self.cost_global), "resample": int(self.resample), "p_edge": self.p,
            "n_params": self.n_params, "n_entanglers": self.n_entanglers,
            "ent_per_layer": self.n_entanglers / self.n_layers,
            "ent_density": len(edges) / max(1.0, n * (n - 1) / 2),
            "ent_max_degree": max(len(v) for v in adj.values()),
            "ent_mean_degree": 2 * len(edges) / n, "deg_q0": len(adj[0]), "ent_diameter": diam,
            "depth_ratio": self.n_layers / n, "cost_weight_eff": self.effective_weight(),
            "cone_qubits": cq, "cone_frac": cp / self.n_params,
        }


def graph_datapoint(rng, qubit_range, layer_range, samples):
    n = int(rng.integers(max(3, qubit_range[0]), qubit_range[1] + 1))
    L = int(rng.integers(layer_range[0], layer_range[1] + 1))
    spec = GraphSpec(n, L, rng.choice(ANSATZ_TYPES), rng.choice(ENTANGLER_GATES),
                     bool(rng.integers(0, 2)), float(rng.uniform(0.1, 1.0)),
                     bool(rng.integers(0, 2)), int(rng.integers(0, 2 ** 31)))
    G = np.empty((samples, spec.n_params))
    for i in range(samples):
        _, G[i] = cost_and_gradient(spec.build(rng.uniform(0.0, 2.0 * math.pi, spec.n_params)),
                                    spec.cost_qubits)
    row = spec.features()
    row.update(label_from_gradients(G))
    return row
