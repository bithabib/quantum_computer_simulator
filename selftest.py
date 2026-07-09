"""Quick self-test for the Quantum Lab (run: python selftest.py).

Verifies the qsim library, the sandbox isolation, and the auto-grader without
needing the web server. Exits non-zero on any failure.
"""

import math

from qsim import QuantumCircuit
from quantum_lab import challenges, sandbox

ISQRT2 = 1 / math.sqrt(2)
_failures = []


def check(name, cond):
    print(("  PASS " if cond else "  FAIL ") + name)
    if not cond:
        _failures.append(name)


def approx(a, b, tol=1e-9):
    return abs(a - b) <= tol


print("qsim library")
bell = QuantumCircuit(2)
bell.h(0).cx(0, 1)
p = bell.statevector().prob_dict()
check("Bell pair is 50/50 over |00> and |11>",
      approx(p.get("00", 0), 0.5) and approx(p.get("11", 0), 0.5) and len(p) == 2)

flip = QuantumCircuit(1)
flip.x(0)
check("X gate: |0> -> |1>", approx(flip.statevector().prob_dict().get("1", 0), 1.0))

print("\nsandbox: safe code runs")
r = sandbox.execute("qc = QuantumCircuit(1)\nqc.h(0)\nprint('hello')")
check("runs and captures stdout", r.get("ok") and "hello" in r.get("stdout", ""))
check("returns probabilities", "probabilities" in r)

print("\nsandbox: malicious code is blocked")
r = sandbox.execute("import os\nos.system('echo pwned')")
check("import os rejected", not r.get("ok") and r.get("stage") == "validation")
r = sandbox.execute("open('secret.txt').read()")
check("open() rejected", not r.get("ok"))
r = sandbox.execute("().__class__.__bases__")
check("dunder escape rejected", not r.get("ok"))
r = sandbox.execute("while True:\n    pass")
check("infinite loop times out", not r.get("ok") and r.get("stage") == "timeout")

print("\nauto-grader")
spec = challenges.get_spec("bell-pair")
r = sandbox.execute("qc = QuantumCircuit(2)\nqc.h(0)\nqc.cx(0, 1)", challenge=spec)
check("correct Bell solution passes", r.get("grade", {}).get("passed") is True)
r = sandbox.execute("qc = QuantumCircuit(2)\nqc.h(0)", challenge=spec)
check("wrong Bell solution fails", r.get("grade", {}).get("passed") is False)

grover = challenges.get_spec("grover-2q")
sol = (
    "qc = QuantumCircuit(2)\n"
    "qc.h(0); qc.h(1)\n"
    "qc.cz(0, 1)\n"
    "qc.h(0); qc.h(1)\n"
    "qc.x(0); qc.x(1)\n"
    "qc.cz(0, 1)\n"
    "qc.x(0); qc.x(1)\n"
    "qc.h(0); qc.h(1)\n"
)
r = sandbox.execute(sol, challenge=grover)
check("Grover solution passes", r.get("grade", {}).get("passed") is True)

print("\n%d failure(s)." % len(_failures))
raise SystemExit(1 if _failures else 0)
