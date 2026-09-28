"""Label repeatability: how much does the label move if we redraw the samples?

Draws fresh random specs from the same design space as the dataset, labels
each one twice with two independent sets of ``--samples`` parameter vectors,
and reports the mean and maximum absolute difference of ``log_var`` (the
all-parameter label).  This is the direct, empirical sampling uncertainty of
the label quoted in the manuscript.

    python -m qml_bp.repeat_labels --n-specs 200 --samples 200 --workers 9 \
        --json paper/qmi/results/repeat.json
"""

import argparse
import json
import os
from multiprocessing import Pool

import numpy as np

from qml_bp.ansatz import compute_datapoint_all, sample_spec

_CFG = {}


def _init(cfg):
    _CFG.update(cfg)


def _work(seed):
    rng = np.random.default_rng(seed)
    spec = sample_spec(rng, (2, 12), (1, 20))
    a = compute_datapoint_all(spec, _CFG["samples"], np.random.default_rng(seed + 1))
    b = compute_datapoint_all(spec, _CFG["samples"], np.random.default_rng(seed + 2))
    return spec.n_qubits, a["log_var"], b["log_var"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-specs", type=int, default=200)
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    base = np.random.SeedSequence(args.seed)
    seeds = [int(s.generate_state(1)[0]) for s in base.spawn(args.n_specs)]
    with Pool(args.workers, initializer=_init, initargs=({"samples": args.samples},)) as pool:
        rows = pool.map(_work, seeds)
    d = np.array([[a, b] for _, a, b in rows if np.isfinite(a) and np.isfinite(b)])
    diff = np.abs(d[:, 0] - d[:, 1])
    out = {"n_specs": int(len(d)), "samples": args.samples,
           "mad": float(diff.mean()), "median_abs": float(np.median(diff)),
           "max_abs": float(diff.max()), "p95_abs": float(np.percentile(diff, 95))}
    print(json.dumps(out, indent=1))
    if args.json:
        os.makedirs(os.path.dirname(args.json), exist_ok=True)
        json.dump(out, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
