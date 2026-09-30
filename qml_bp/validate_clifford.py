"""Validate the Clifford-sampling estimator against exact statevector results.

For random circuits with n <= 12, compare, parameter by parameter, the
gradient variance from Clifford sampling with the variance of exact adjoint
gradients over uniformly random angles.  Both estimate the same quantity.

    python -m qml_bp.validate_clifford --n-circuits 200 --workers 9 \
        --json paper/qmi/results/validation_clifford.json
"""

import argparse
import json
import math
import os
from multiprocessing import Pool

import numpy as np

from qml_bp.adjoint import cost_and_gradient
from qml_bp.ansatz import STRUCTURAL_TOL, sample_spec
from qml_bp.clifford import estimate

_CFG = {}


def _init(cfg):
    _CFG.update(cfg)


def _work(seed):
    rng = np.random.default_rng(seed)
    spec = sample_spec(rng, (2, _CFG["qmax"]), (1, 20))
    S = _CFG["sv_samples"]
    grads = np.empty((S, spec.n_params))
    for i in range(S):
        theta = rng.uniform(0, 2 * math.pi, size=spec.n_params)
        _, grads[i] = cost_and_gradient(spec.build(theta), spec.cost_qubits)
    var_sv = grads.var(axis=0)
    struct_sv = np.max(np.abs(grads), axis=0) < STRUCTURAL_TOL
    cl = estimate(spec, rng, max_samples=_CFG["cl_samples"], min_samples=_CFG["cl_samples"],
                  target_hits=10 ** 12)
    var_cl = cl["per_param"]
    return {
        "n": spec.n_qubits, "L": spec.n_layers, "m": spec.n_params,
        "mean_sv": float(var_sv.mean()), "mean_cl": float(var_cl.mean()),
        "struct_sv": int(struct_sv.sum()), "zero_cl": int((var_cl == 0).sum()),
        # structural zeros must have zero Clifford hits
        "struct_with_hits": int((struct_sv & (var_cl > 0)).sum()),
        # per-parameter agreement where the variance is large enough to resolve
        "max_abs_diff": float(np.max(np.abs(var_sv - var_cl))),
        "corr": float(np.corrcoef(var_sv, var_cl)[0, 1]) if var_sv.std() > 0 and var_cl.std() > 0 else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-circuits", type=int, default=200)
    ap.add_argument("--qmax", type=int, default=12)
    ap.add_argument("--sv-samples", type=int, default=400)
    ap.add_argument("--cl-samples", type=int, default=65536)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    base = np.random.SeedSequence(args.seed)
    seeds = [int(s.generate_state(1)[0]) for s in base.spawn(args.n_circuits)]
    cfg = {"qmax": args.qmax, "sv_samples": args.sv_samples, "cl_samples": args.cl_samples}
    with Pool(args.workers, initializer=_init, initargs=(cfg,)) as pool:
        rows = pool.map(_work, seeds, chunksize=2)
    ok = [r for r in rows if r["mean_sv"] > 0 and r["mean_cl"] > 0]
    d = np.array([math.log10(r["mean_cl"]) - math.log10(r["mean_sv"]) for r in ok])
    out = {
        "n_circuits": len(rows), "n_params": int(sum(r["m"] for r in rows)),
        "sv_samples": args.sv_samples, "cl_samples": args.cl_samples,
        "log10_diff_mean": float(d.mean()), "log10_diff_mad": float(np.mean(np.abs(d))),
        "log10_diff_max": float(np.max(np.abs(d))),
        "struct_sv_total": int(sum(r["struct_sv"] for r in rows)),
        "struct_with_clifford_hits": int(sum(r["struct_with_hits"] for r in rows)),
        "both_zero_circuits": int(sum(1 for r in rows if r["mean_sv"] == 0 and r["mean_cl"] == 0)),
        "mismatch_zero_circuits": int(sum(1 for r in rows if (r["mean_sv"] == 0) != (r["mean_cl"] == 0))),
        "median_per_param_corr": float(np.nanmedian([r["corr"] for r in rows])),
    }
    print(json.dumps(out, indent=1))
    if args.json:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        json.dump(out, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
