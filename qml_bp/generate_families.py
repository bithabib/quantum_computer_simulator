"""Generate the shared-parameter and random-graph datasets.

    python -m qml_bp.generate_families --family tied  --n-specs 10000 --seed 31337 \
        --workers 9 --out data_bp/bp_tied_v2.csv
    python -m qml_bp.generate_families --family graph --n-specs 10000 --seed 27182 \
        --workers 9 --out data_bp/bp_graph_v2.csv
"""

import argparse
import csv
import os
import time
from multiprocessing import Pool

import numpy as np

from qml_bp.families import (GRAPH_COLUMNS, TIED_COLUMNS, graph_datapoint,
                             tied_datapoint)

_CFG = {}


def _init(cfg):
    _CFG.update(cfg)


def _work(seed):
    rng = np.random.default_rng(seed)
    fn = tied_datapoint if _CFG["family"] == "tied" else graph_datapoint
    return fn(rng, (_CFG["qmin"], _CFG["qmax"]), (_CFG["lmin"], _CFG["lmax"]), _CFG["samples"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["tied", "graph"], required=True)
    ap.add_argument("--n-specs", type=int, default=1000)
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--qubit-min", type=int, default=2)
    ap.add_argument("--qubit-max", type=int, default=12)
    ap.add_argument("--layer-min", type=int, default=1)
    ap.add_argument("--layer-max", type=int, default=20)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = {"family": args.family, "samples": args.samples, "qmin": args.qubit_min,
           "qmax": args.qubit_max, "lmin": args.layer_min, "lmax": args.layer_max}
    cols = TIED_COLUMNS if args.family == "tied" else GRAPH_COLUMNS
    seeds = [int(s.generate_state(1)[0]) for s in np.random.SeedSequence(args.seed).spawn(args.n_specs)]
    t0 = time.time(); done = 0
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        with Pool(args.workers, initializer=_init, initargs=(cfg,)) as pool:
            for row in pool.imap_unordered(_work, seeds, chunksize=4):
                w.writerow({k: row[k] for k in cols}); done += 1
                if done % 1000 == 0 or done == args.n_specs:
                    fh.flush(); print("  %6d/%d (%.1f rows/s)" % (done, args.n_specs, done / (time.time() - t0)), flush=True)
    print("Done in %.1f min -> %s" % ((time.time() - t0) / 60.0, args.out))


if __name__ == "__main__":
    main()
