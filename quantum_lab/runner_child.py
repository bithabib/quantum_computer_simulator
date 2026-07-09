"""Child process that actually executes user code, in isolation.

Never import this into the Flask process. It is launched as a separate Python
interpreter by :mod:`quantum_lab.sandbox`, reads a JSON job on stdin, and writes
a single JSON result line to stdout. Running out-of-process means a crash,
hang, or resource blow-up in user code cannot take down the web server, and the
parent can hard-kill it on timeout.

Layers of protection applied here:
  1. POSIX resource limits (CPU seconds, address space) where available.
  2. AST allowlist validation (quantum_lab.validator).
  3. A restricted ``__builtins__`` with a guarded importer.
"""

import io
import json
import sys

# Make sibling packages importable regardless of CWD.
import os as _os  # noqa: used only for path setup, not exposed to user code

_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from quantum_lab.validator import validate, ValidationError, ALLOWED_IMPORTS  # noqa: E402
from quantum_lab import grader as grader_mod  # noqa: E402


def _apply_resource_limits():
    """Best-effort CPU/memory caps. No-op on platforms without ``resource``."""
    try:
        import resource
    except ImportError:
        return
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (5, 6))
    except (ValueError, OSError):
        pass
    try:
        # 512 MB address space cap.
        cap = 512 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    except (ValueError, OSError):
        pass


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".")[0]
    if root not in ALLOWED_IMPORTS:
        raise ImportError("import of %r is not allowed in the Quantum Lab" % name)
    return __import__(name, globals, locals, fromlist, level)


# A conservative allowlist of safe builtins. Notably excludes eval/exec/open/
# getattr/setattr/__import__ (a guarded importer is injected separately).
_SAFE_BUILTIN_NAMES = [
    "abs", "all", "any", "bool", "bytes", "callable", "chr", "complex",
    "dict", "divmod", "enumerate", "filter", "float", "format", "frozenset",
    "hash", "hex", "int", "isinstance", "issubclass", "iter", "len", "list",
    "map", "max", "min", "next", "oct", "ord", "pow", "print", "range",
    "repr", "reversed", "round", "set", "slice", "sorted", "str", "sum",
    "tuple", "type", "zip", "True", "False", "None",
    "Exception", "ValueError", "TypeError", "IndexError", "KeyError",
    "ZeroDivisionError", "ArithmeticError", "RuntimeError", "StopIteration",
    "AttributeError", "NotImplementedError", "AssertionError",
]


def _build_safe_builtins():
    import builtins

    safe = {}
    for nm in _SAFE_BUILTIN_NAMES:
        if hasattr(builtins, nm):
            safe[nm] = getattr(builtins, nm)
    safe["__import__"] = _safe_import
    return safe


def run(job):
    code = job.get("code", "")
    challenge = job.get("challenge")  # grading spec or None
    target_var = (challenge or {}).get("target_var", "qc")

    # 1) Static validation.
    try:
        validate(code)
    except ValidationError as exc:
        return {"ok": False, "error": str(exc), "stage": "validation"}

    # 2) Prepare a restricted namespace with qsim available.
    import qsim

    namespace = {
        "__builtins__": _build_safe_builtins(),
        "qsim": qsim,
        "QuantumCircuit": qsim.QuantumCircuit,
    }

    # 3) Execute user code with stdout captured.
    captured = io.StringIO()
    real_stdout = sys.stdout
    sys.stdout = captured
    try:
        compiled = compile(code, "<user_code>", "exec")
        exec(compiled, namespace)  # noqa: S102 - sandboxed on purpose
    except Exception as exc:  # noqa: BLE001 - report any user error
        sys.stdout = real_stdout
        return {
            "ok": False,
            "error": "%s: %s" % (type(exc).__name__, exc),
            "stage": "execution",
            "stdout": captured.getvalue(),
        }
    finally:
        sys.stdout = real_stdout

    result = {"ok": True, "stdout": captured.getvalue()}

    # 4) Inspect the resulting circuit / statevector for output + grading.
    circuit = namespace.get(target_var)
    if isinstance(circuit, qsim.QuantumCircuit):
        try:
            sv = circuit.statevector()
            result["num_qubits"] = circuit.num_qubits
            result["probabilities"] = sv.prob_dict()
            result["statevector"] = sv.amplitudes()
        except Exception as exc:  # noqa: BLE001
            result["sim_error"] = "%s: %s" % (type(exc).__name__, exc)

    # 5) Grade against the challenge, if any.
    if challenge is not None:
        result["grade"] = grader_mod.grade(namespace, challenge)

    return result


def main():
    _apply_resource_limits()
    try:
        job = json.loads(sys.stdin.read() or "{}")
        out = run(job)
    except Exception as exc:  # noqa: BLE001 - never crash silently
        out = {"ok": False, "error": "runner error: %s" % exc, "stage": "runner"}
    sys.stdout.write(json.dumps(out))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
