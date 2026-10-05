"""Generate a dataset labeled by Clifford sampling (any qubit count).

The label is the mean gradient variance over ALL parameters (structural zeros
included), which the Clifford estimator measures without bias.  Sampling is
adaptive; circuits whose variance is below the resolution reached at
``--max-samples`` are written with ``resolved = 0``.

    python -m qml_bp.generate_clifford --n-specs 30000 --qubit-min 2 --qubit-max 32 \
        --workers 9 --seed 2026 --out data_bp/bp_clifford.csv
"""

import argparse
import csv
import os
import time
from multiprocessing import Pool

import numpy as np

from qml_bp.ansatz import FEATURE_COLUMNS, sample_spec
from qml_bp.clifford import estimate

_COLUMNS = list(FEATURE_COLUMNS) + [
    "samples", "total_hits", "var_mean_all", "log_var_all", "rel_se",
    "resolved", "n_zero_hit", "frac_zero_hit", "resolution_floor",
]
_CFG = {}


def _init(cfg):
    _CFG.update(cfg)


def _work(seed):
    rng = np.random.default_rng(seed)
    spec = sample_spec(rng, (_CFG["qmin"], _CFG["qmax"]), (_CFG["lmin"], _CFG["lmax"]),
                       patterns=_CFG["patterns"])
    e = estimate(spec, rng, min_samples=_CFG["min_samples"], max_samples=_CFG["max_samples"],
                 target_hits=_CFG["target_hits"])
    row = spec.features()
    for k in ("samples", "total_hits", "var_mean_all", "log_var_all", "rel_se",
              "n_zero_hit", "frac_zero_hit", "resolution_floor"):
        row[k] = e[k]
    row["resolved"] = int(e["resolved"])
    return {k: row[k] for k in _COLUMNS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-specs", type=int, default=1000)
    ap.add_argument("--qubit-min", type=int, default=2)
    ap.add_argument("--qubit-max", type=int, default=32)
    ap.add_argument("--layer-min", type=int, default=1)
    ap.add_argument("--layer-max", type=int, default=20)
    ap.add_argument("--min-samples", type=int, default=8192)
    ap.add_argument("--max-samples", type=int, default=2 ** 20)
    ap.add_argument("--target-hits", type=int, default=2000)
    ap.add_argument("--patterns", default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="data_bp/bp_clifford.csv")
    args = ap.parse_args()
    cfg = {"qmin": args.qubit_min, "qmax": args.qubit_max, "lmin": args.layer_min,
           "lmax": args.layer_max, "min_samples": args.min_samples,
           "max_samples": args.max_samples, "target_hits": args.target_hits,
           "patterns": args.patterns.split(",") if args.patterns else None}
    base = np.random.SeedSequence(args.seed)
    seeds = [int(s.generate_state(1)[0]) for s in base.spawn(args.n_specs)]
    t0 = time.time(); done = 0
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_COLUMNS); w.writeheader()
        with Pool(args.workers, initializer=_init, initargs=(cfg,)) as pool:
            for row in pool.imap_unordered(_work, seeds, chunksize=4):
                w.writerow(row); done += 1
                if done % 1000 == 0 or done == args.n_specs:
                    fh.flush()
                    print("  %6d/%d  (%.1f rows/s)" % (done, args.n_specs, done / (time.time() - t0)), flush=True)
    print("Done in %.1f min -> %s" % ((time.time() - t0) / 60.0, args.out))


if __name__ == "__main__":
    main()
