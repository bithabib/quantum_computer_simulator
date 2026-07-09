"""Deprecated & disabled: the old arbitrary-code-execution endpoint.

The previous implementation ran ``exec()`` on unsanitised POST input, a remote
code execution hole. It has been replaced by the sandboxed Quantum Lab API in
:mod:`views.public.api.quantum_lab` (``/api/quantum/run`` and
``/api/quantum/check``).

The legacy route is kept only so old clients get a clear error instead of a
silent 404, and it no longer executes anything.
"""

from flask import jsonify

from app import app


@app.route("/run_python_code", methods=["POST"])
def run_python_code():
    return (
        jsonify(
            {
                "ok": False,
                "error": "This endpoint has been removed for security. "
                "Use /api/quantum/run or /api/quantum/check instead.",
            }
        ),
        410,  # Gone
    )
