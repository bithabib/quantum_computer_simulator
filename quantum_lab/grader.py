"""Auto-grading: compare a user's circuit to a challenge's expected result.

A challenge spec is a plain (JSON-serialisable) dict so it can be handed to the
sandbox child. It carries the expected answer as raw numbers computed ahead of
time from a reference solution (see :mod:`quantum_lab.challenges`).

Statevectors are compared by *fidelity* ``|<expected|actual>|^2`` which is 1.0
for a perfect match and automatically ignores global phase (an unobservable
degree of freedom), so a valid answer is not marked wrong for a phase offset.
"""

import numpy as np

import qsim


def _fail(msg):
    return {"passed": False, "message": msg}


def grade(namespace, spec):
    target_var = spec.get("target_var", "qc")
    circuit = namespace.get(target_var)

    if circuit is None:
        return _fail("Define a QuantumCircuit named '%s'." % target_var)
    if not isinstance(circuit, qsim.QuantumCircuit):
        return _fail("'%s' must be a QuantumCircuit." % target_var)

    expected_n = spec.get("num_qubits")
    if expected_n is not None and circuit.num_qubits != expected_n:
        return _fail(
            "Expected a %d-qubit circuit, but '%s' has %d qubit(s)."
            % (expected_n, target_var, circuit.num_qubits)
        )

    try:
        sv = circuit.statevector()
    except Exception as exc:  # noqa: BLE001
        return _fail("Could not simulate your circuit: %s" % exc)

    check = spec.get("check", "statevector")
    tol = float(spec.get("tolerance", 1e-6))

    if check == "statevector":
        expected = np.array(
            [complex(re, im) for re, im in spec["expected_statevector"]],
            dtype=complex,
        )
        actual = sv.data
        if actual.shape != expected.shape:
            return _fail("Statevector has the wrong dimension.")
        fidelity = float(np.abs(np.vdot(expected, actual)) ** 2)
        if abs(fidelity - 1.0) <= max(tol, 1e-9) * 10 or fidelity >= 1.0 - 1e-6:
            return {
                "passed": True,
                "message": "Correct! Fidelity with the target state is %.4f." % fidelity,
                "fidelity": fidelity,
            }
        return {
            "passed": False,
            "message": "Not quite - your state overlaps the target with fidelity %.4f "
            "(need ~1.0)." % fidelity,
            "fidelity": fidelity,
        }

    if check == "probabilities":
        expected = spec["expected_probabilities"]
        actual = sv.prob_dict()
        keys = set(expected) | set(actual)
        for k in keys:
            if abs(expected.get(k, 0.0) - actual.get(k, 0.0)) > max(tol, 1e-3):
                return {
                    "passed": False,
                    "message": "Measurement distribution does not match the target yet.",
                }
        return {"passed": True, "message": "Correct! Measurement distribution matches."}

    return _fail("Unknown grading check '%s'." % check)
