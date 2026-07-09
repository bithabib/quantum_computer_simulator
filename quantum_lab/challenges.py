"""Challenge library for the Quantum Lab.

Each challenge carries a reference ``solution`` that builds the target circuit.
The expected answer (statevector or measurement distribution) is computed from
that reference at import time and baked into a JSON-serialisable *grading spec*
handed to the sandbox. Grading therefore never trusts a stored vector by hand -
it is always derived from a known-correct circuit.
"""

import qsim

_STARTER = (
    "# Build your circuit in a variable named `qc`.\n"
    "# The grader reads `qc` and checks the quantum state it produces.\n"
    "qc = QuantumCircuit({n})\n"
)


def _sol_superposition(qc):
    qc.h(0)


def _sol_bitflip(qc):
    qc.x(0)


def _sol_minus(qc):
    qc.x(0)
    qc.h(0)


def _sol_plus_i(qc):
    qc.h(0)
    qc.s(0)


def _sol_bell(qc):
    qc.h(0)
    qc.cx(0, 1)


def _sol_ghz(qc):
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(0, 2)


def _sol_swap(qc):
    qc.x(0)          # prepare |10>
    qc.swap(0, 1)    # -> |01>


def _sol_grover2(qc):
    qc.h(0); qc.h(1)          # uniform superposition
    qc.cz(0, 1)               # oracle: phase-flip |11>
    qc.h(0); qc.h(1)          # diffuser
    qc.x(0); qc.x(1)
    qc.cz(0, 1)
    qc.x(0); qc.x(1)
    qc.h(0); qc.h(1)


# id, title, difficulty, num_qubits, check-mode, solution, task, hints
_DEFS = [
    dict(
        id="superposition",
        title="Equal superposition",
        difficulty="Beginner",
        num_qubits=1,
        check="statevector",
        solution=_sol_superposition,
        task="Put the single qubit into an equal superposition of |0> and |1>, "
        "so measuring it gives 0 or 1 with 50/50 probability.",
        hints=["The Hadamard gate `h` creates an equal superposition.",
               "One line: qc.h(0)"],
    ),
    dict(
        id="bit-flip",
        title="Flip the bit",
        difficulty="Beginner",
        num_qubits=1,
        check="statevector",
        solution=_sol_bitflip,
        task="Turn the qubit from |0> into |1>.",
        hints=["The Pauli-X gate `x` is the quantum NOT gate.", "qc.x(0)"],
    ),
    dict(
        id="minus-state",
        title="The minus state",
        difficulty="Beginner",
        num_qubits=1,
        check="statevector",
        solution=_sol_minus,
        task="Prepare the state |-> = (|0> - |1>) / sqrt(2).",
        hints=["|-> is what you get by applying H to |1>.",
               "First flip to |1> with x, then apply h."],
    ),
    dict(
        id="plus-i-state",
        title="The +i state",
        difficulty="Intermediate",
        num_qubits=1,
        check="statevector",
        solution=_sol_plus_i,
        task="Prepare the state (|0> + i|1>) / sqrt(2), which points along +Y "
        "on the Bloch sphere.",
        hints=["Start with a Hadamard to reach (|0>+|1>)/sqrt(2).",
               "The S gate adds a 90-degree phase to |1>: qc.s(0)"],
    ),
    dict(
        id="bell-pair",
        title="Bell pair (entanglement)",
        difficulty="Intermediate",
        num_qubits=2,
        check="statevector",
        solution=_sol_bell,
        task="Create the entangled Bell state (|00> + |11>) / sqrt(2).",
        hints=["Put qubit 0 in superposition with h.",
               "Then entangle with a CNOT: qc.cx(0, 1)"],
    ),
    dict(
        id="swap",
        title="Swap two qubits",
        difficulty="Intermediate",
        num_qubits=2,
        check="statevector",
        solution=_sol_swap,
        task="Produce the state |01> by first preparing |10> and then swapping "
        "the two qubits.",
        hints=["Flip qubit 0 to get |10> with qc.x(0).",
               "Exchange the qubits with qc.swap(0, 1)."],
    ),
    dict(
        id="ghz",
        title="GHZ state (3-way entanglement)",
        difficulty="Advanced",
        num_qubits=3,
        check="statevector",
        solution=_sol_ghz,
        task="Create the 3-qubit GHZ state (|000> + |111>) / sqrt(2).",
        hints=["Superpose qubit 0, then fan the entanglement out.",
               "h(0), then cx(0,1) and cx(0,2)."],
    ),
    dict(
        id="grover-2q",
        title="Grover search (2 qubits)",
        difficulty="Advanced",
        num_qubits=2,
        check="probabilities",
        solution=_sol_grover2,
        task="Use one Grover iteration so that measuring both qubits returns "
        "|11> with certainty. (Oracle marks |11>, then apply the diffuser.)",
        hints=["Start in uniform superposition: h(0), h(1).",
               "Oracle = cz(0,1). Diffuser = H, X, CZ, X, H on both qubits."],
    ),
]


def _build_spec(d):
    """Compute the expected result from the reference solution."""
    qc = qsim.QuantumCircuit(d["num_qubits"])
    d["solution"](qc)
    sv = qc.statevector()
    spec = {
        "id": d["id"],
        "target_var": "qc",
        "num_qubits": d["num_qubits"],
        "check": d["check"],
        "tolerance": 1e-6,
    }
    if d["check"] == "statevector":
        spec["expected_statevector"] = sv.amplitudes()
    else:
        spec["expected_probabilities"] = sv.prob_dict()
    return spec


_SPECS = {d["id"]: _build_spec(d) for d in _DEFS}


def list_challenges():
    """Public metadata for every challenge (no answers leaked)."""
    out = []
    for d in _DEFS:
        out.append(
            {
                "id": d["id"],
                "title": d["title"],
                "difficulty": d["difficulty"],
                "num_qubits": d["num_qubits"],
                "task": d["task"],
                "hints": d["hints"],
                "starter_code": _STARTER.format(n=d["num_qubits"]),
            }
        )
    return out


def get_challenge(challenge_id):
    for d in _DEFS:
        if d["id"] == challenge_id:
            return d
    return None


def get_spec(challenge_id):
    """The grading spec for a challenge (JSON-safe), or None if unknown."""
    return _SPECS.get(challenge_id)
