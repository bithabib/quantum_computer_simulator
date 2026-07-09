"""Launch user code in an isolated subprocess with a hard timeout.

The parent (Flask) never execs user code itself. It spawns a fresh Python
interpreter running :mod:`quantum_lab.runner_child`, hands it the job as JSON on
stdin, and reads a JSON result from stdout. If the child overruns the timeout it
is killed, so an infinite loop in user code cannot hang a request forever.
"""

import json
import os
import subprocess
import sys

_CHILD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runner_child.py")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_TIMEOUT = 8  # wall-clock seconds
MAX_CODE_LEN = 20_000  # characters


def execute(code, challenge=None, timeout=DEFAULT_TIMEOUT):
    """Run ``code`` (optionally graded against ``challenge``) and return a dict.

    The returned dict always has a boolean ``ok`` key. On failure it carries an
    ``error`` string and a ``stage`` ("validation" | "execution" | "timeout" |
    "runner" | "internal").
    """
    if not isinstance(code, str) or not code.strip():
        return {"ok": False, "error": "No code provided.", "stage": "validation"}
    if len(code) > MAX_CODE_LEN:
        return {
            "ok": False,
            "error": "Code too long (max %d characters)." % MAX_CODE_LEN,
            "stage": "validation",
        }

    job = json.dumps({"code": code, "challenge": challenge})

    # Isolate the child: minimal env, no inherited PYTHONPATH surprises.
    env = {
        "PYTHONPATH": _ROOT,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PATH": os.environ.get("PATH", ""),
    }
    if os.name == "nt" and "SYSTEMROOT" in os.environ:
        env["SYSTEMROOT"] = os.environ["SYSTEMROOT"]

    try:
        proc = subprocess.run(
            [sys.executable, _CHILD],
            input=job,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=_ROOT,
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "error": "Execution timed out after %ds (possible infinite loop)." % timeout,
            "stage": "timeout",
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": "Sandbox error: %s" % exc, "stage": "internal"}

    # On POSIX a negative return code means the child was killed by a signal -
    # e.g. SIGXCPU/SIGKILL from the CPU or memory resource limits, which is how
    # a busy infinite loop gets stopped before the wall-clock timeout.
    if proc.returncode is not None and proc.returncode < 0:
        return {
            "ok": False,
            "error": "Your code was terminated for exceeding resource limits "
            "(CPU/memory) - likely an infinite loop or too much computation.",
            "stage": "timeout",
        }

    if not proc.stdout.strip():
        detail = proc.stderr.strip()[-500:] or "no output"
        return {
            "ok": False,
            "error": "Sandbox produced no result (%s)." % detail,
            "stage": "internal",
        }

    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {
            "ok": False,
            "error": "Malformed sandbox output.",
            "stage": "internal",
        }
