"""Quantum Lab API: list challenges, run code, and check solutions.

All user code is executed by :mod:`quantum_lab.sandbox` in an isolated
subprocess with static validation and a timeout - the Flask process itself
never runs user-supplied code.
"""

from flask import jsonify, request

from app import app
from quantum_lab import challenges, sandbox


@app.route("/api/quantum/challenges", methods=["GET"])
def quantum_challenges():
    """Return the list of available challenges (metadata only, no answers)."""
    return jsonify(challenges.list_challenges())


@app.route("/api/quantum/run", methods=["POST"])
def quantum_run():
    """Run code in the sandbox (freeform, no grading)."""
    data = request.get_json(silent=True) or {}
    result = sandbox.execute(data.get("code", ""))
    return jsonify(result)


@app.route("/api/quantum/check", methods=["POST"])
def quantum_check():
    """Run code and grade it against a challenge."""
    data = request.get_json(silent=True) or {}
    challenge_id = data.get("challenge_id")
    spec = challenges.get_spec(challenge_id)
    if spec is None:
        return jsonify({"ok": False, "error": "Unknown challenge.", "stage": "validation"})
    result = sandbox.execute(data.get("code", ""), challenge=spec)
    return jsonify(result)
