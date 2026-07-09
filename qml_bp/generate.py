"""Generate a barren-plateau trainability dataset in parallel.

Each row = one random circuit spec + its measured gradient variance (the label).
Work is spread across CPU cores; rows are streamed to a CSV as they complete, so
a long run can be interrupted and resumed-by-appending without losing progress.

Example (big run on an M4, ~10 cores):

    python -m qml_bp.generate --n-specs 100000 --samples 200 \
        --qubit-min 2 --qubit-max 12 --layer-min 1 --layer-max 20 \
        --workers 10 --out data_bp/bp_dataset.csv

Start small first to gauge speed:

    python -m qml_bp.generate --n-specs 500 --samples 100 --out data_bp/sample.csv
"""

import argparse
import csv
import os
import time
from multiprocessing import Pool

import numpy as np

from qml_bp.ansatz import compute_datapoint, sample_spec

# Column order written to CSV.
_COLUMNS = [
    "n_qubits", "n_layers", "n_params", "ansatz_type", "entangle_pattern",
    "entangler_gate", "cost_global", "n_entanglers", "depth_ratio",
    "grad_mean", "grad_var", "log_grad_var", "samples",
]

# Config shared with worker processes (set in the initializer to avoid pickling
# it on every task).
_CFG = {}


def _init_worker(cfg):
    _CFG.update(cfg)


def _work(task_seed):
    """Compute one datapoint. `task_seed` seeds both the spec and the sampling."""
    rng = np.random.default_rng(task_seed)
    spec = sample_spec(
        rng,
        qubit_range=(_CFG["qubit_min"], _CFG["qubit_max"]),
        layer_range=(_CFG["layer_min"], _CFG["layer_max"]),
    )
    return compute_datapoint(spec, _CFG["samples"], rng)


def main():
    ap = argparse.ArgumentParser(description="Generate a barren-plateau dataset.")
    ap.add_argument("--n-specs", type=int, default=1000, help="number of circuits")
    ap.add_argument("--samples", type=int, default=150,
                    help="random parameter vectors per circuit (variance estimate)")
    ap.add_argument("--qubit-min", type=int, default=2)
    ap.add_argument("--qubit-max", type=int, default=12)
    ap.add_argument("--layer-min", type=int, default=1)
    ap.add_argument("--layer-max", type=int, default=20)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", type=str, default="data_bp/bp_dataset.csv")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    cfg = {
        "samples": args.samples,
        "qubit_min": args.qubit_min, "qubit_max": args.qubit_max,
        "layer_min": args.layer_min, "layer_max": args.layer_max,
    }
    # Distinct, reproducible seed per task.
    base = np.random.SeedSequence(args.seed)
    seeds = [int(s.generate_state(1)[0]) for s in base.spawn(args.n_specs)]

    print("Generating %d circuits x %d samples on %d workers -> %s"
          % (args.n_specs, args.samples, args.workers, args.out))
    t0 = time.time()
    done = 0

    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_COLUMNS)
        writer.writeheader()
        with Pool(args.workers, initializer=_init_worker, initargs=(cfg,)) as pool:
            for row in pool.imap_unordered(_work, seeds, chunksize=8):
                writer.writerow(row)
                done += 1
                if done % 200 == 0 or done == args.n_specs:
                    rate = done / max(1e-9, time.time() - t0)
                    eta = (args.n_specs - done) / max(1e-9, rate)
                    fh.flush()
                    print("  %6d/%d  (%.1f rows/s, ETA %5.1f min)"
                          % (done, args.n_specs, rate, eta / 60.0), flush=True)

    print("Done in %.1f min -> %s" % ((time.time() - t0) / 60.0, args.out))


if __name__ == "__main__":
    main()
