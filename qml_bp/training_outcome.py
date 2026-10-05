"""Does the (predicted) gradient-variance label predict actual training?

Fresh circuits are drawn from the dataset's design space.  For each one we
(i) compute the true all-parameter label as in the dataset, (ii) predict the
label from architecture alone with an HGB model fitted on the released
dataset, and (iii) actually train the circuit: minimise C(theta) with Adam
from ``--restarts`` random initialisations for a fixed budget of ``--steps``
steps, using exact adjoint gradients, and record the loss decrease
``dC = C(theta_0) - C(theta_final)`` averaged over restarts.  Because C is the
expectation of a Pauli-Z string, C lies in [-1, 1] and the best possible
decrease from a typical start is about 1.

Reported: Spearman correlation of dC with the true and with the predicted
label; mean dC in predicted-barren versus predicted-trainable circuits; and a
scatter figure.

    python -m qml_bp.training_outcome --data data_bp/bp_dataset_v2.csv \
        --n-circuits 300 --workers 7 --outdir paper/qmi/results --figdir paper/qmi/figs
"""

import argparse
import json
import math
import os
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from qml_bp.adjoint import cost_and_gradient
from qml_bp.analyze import CUTOFF, hgb_reg
from qml_bp.ansatz import FEATURE_COLUMNS, compute_datapoint_all, sample_spec

warnings.filterwarnings("ignore", category=RuntimeWarning)
_CFG = {}


def _init(cfg):
    _CFG.update(cfg)


def _noisy_gradient(spec, theta, shots, rng):
    """Exact gradient plus the shot noise a parameter-shift estimate would
    carry: each shifted cost is a mean of ``shots`` +/-1 outcomes, so the
    estimate of g_k = (C+ - C-)/2 has variance ~ (1 - C^2) / (2 shots)."""
    c, g = cost_and_gradient(spec.build(theta), spec.cost_qubits)
    if shots > 0:
        sd = math.sqrt(max(1.0 - c * c, 0.05) / (2.0 * shots))
        g = g + rng.normal(0.0, sd, size=g.shape)
    return c, g


def _optimise(spec, theta, steps, lr, optimizer, shots, rng):
    """Minimise C(theta) for a fixed step budget.  Returns (C_0, best C).

    'adam'  : Adam with the given learning rate (scale-invariant steps).
    'gd'    : plain gradient descent, theta <- theta - lr * g (step size
              proportional to the gradient, so barren circuits stall).
    The final cost is evaluated exactly (no shot noise)."""
    m = np.zeros_like(theta); v = np.zeros_like(theta)
    b1, b2, eps = 0.9, 0.999, 1e-8
    c0, _ = cost_and_gradient(spec.build(theta), spec.cost_qubits)
    best = c0
    for t in range(1, steps + 1):
        c, g = _noisy_gradient(spec, theta, shots, rng)
        best = min(best, c)
        if optimizer == "adam":
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g * g
            mh = m / (1 - b1 ** t); vh = v / (1 - b2 ** t)
            theta = theta - lr * mh / (np.sqrt(vh) + eps)
        else:
            theta = theta - lr * g
    c, _ = cost_and_gradient(spec.build(theta), spec.cost_qubits)
    best = min(best, c)
    return c0, best


def _work(seed):
    rng = np.random.default_rng(seed)
    spec = sample_spec(rng, (2, 12), (1, 20))
    row = compute_datapoint_all(spec, _CFG["samples"], rng)
    decs = []
    for r in range(_CFG["restarts"]):
        theta = rng.uniform(0, 2 * math.pi, size=spec.n_params)
        c0, cbest = _optimise(spec, theta, _CFG["steps"], _CFG["lr"], _CFG["optimizer"],
                              _CFG["shots"], rng)
        decs.append(c0 - cbest)
    row["dC_mean"] = float(np.mean(decs)); row["dC_max"] = float(np.max(decs))
    row["dC_min"] = float(np.min(decs))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--n-circuits", type=int, default=300)
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--optimizer", choices=["adam", "gd"], default="adam")
    ap.add_argument("--shots", type=int, default=1000,
                    help="shots per cost evaluation for the gradient estimate (0 = exact)")
    ap.add_argument("--tag", default="", help="suffix for output file names")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--seed", type=int, default=31415)
    ap.add_argument("--outdir", default="paper/qmi/results")
    ap.add_argument("--figdir", default="paper/qmi/figs")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True); os.makedirs(args.figdir, exist_ok=True)

    # predictor fitted on the released dataset
    df = pd.read_csv(args.data).dropna(subset=["log_var"])
    model = hgb_reg().fit(df[FEATURE_COLUMNS].to_numpy(float), df.log_var.to_numpy(float))

    base = np.random.SeedSequence(args.seed)
    seeds = [int(s.generate_state(1)[0]) for s in base.spawn(args.n_circuits)]
    cfg = {"samples": args.samples, "steps": args.steps, "restarts": args.restarts, "lr": args.lr,
           "optimizer": args.optimizer, "shots": args.shots}
    tag = args.tag
    with Pool(args.workers, initializer=_init, initargs=(cfg,)) as pool:
        rows = pool.map(_work, seeds, chunksize=2)
    out = pd.DataFrame(rows).dropna(subset=["log_var"]).reset_index(drop=True)
    out["pred_log_var"] = model.predict(out[FEATURE_COLUMNS].to_numpy(float))
    out.to_csv(os.path.join(args.outdir, "training_outcome%s.csv" % tag), index=False)

    y, p, d = out.log_var.to_numpy(), out.pred_log_var.to_numpy(), out.dC_mean.to_numpy()
    pb = p < CUTOFF; tb = y < CUTOFF
    R = {
        "n_circuits": int(len(out)), "steps": args.steps, "restarts": args.restarts, "lr": args.lr,
        "optimizer": args.optimizer, "shots": args.shots,
        "spearman_true_vs_dC": float(spearmanr(y, d).correlation),
        "spearman_pred_vs_dC": float(spearmanr(p, d).correlation),
        "spearman_true_vs_pred": float(spearmanr(y, p).correlation),
        "dC_mean_pred_barren": float(d[pb].mean()), "dC_mean_pred_trainable": float(d[~pb].mean()),
        "dC_mean_true_barren": float(d[tb].mean()), "dC_mean_true_trainable": float(d[~tb].mean()),
        "frac_progress_pred_barren": float((d[pb] > 0.5).mean()),
        "frac_progress_pred_trainable": float((d[~pb] > 0.5).mean()),
        "n_pred_barren": int(pb.sum()), "n_pred_trainable": int((~pb).sum()),
    }
    # dC by predicted-label bins
    bins = [-6, -3, -2.5, -2, -1.5, -1, 0]
    out["bin"] = pd.cut(out.pred_log_var, bins)
    R["dC_by_pred_bin"] = {str(k): [float(v.mean()), int(len(v))] for k, v in out.groupby("bin", observed=True).dC_mean}
    print(json.dumps(R, indent=1))
    json.dump(R, open(os.path.join(args.outdir, "training_outcome%s.json" % tag), "w"), indent=1)

    lines = ["\\begin{tabular}{lrcc}", "\\toprule",
             "Predicted label bin & Circuits & Mean loss decrease $\\Delta C$ & Fraction with $\\Delta C>0.5$ \\\\",
             "\\midrule"]
    for k, g in out.groupby("bin", observed=True):
        lines.append("$%s$ & %d & %.2f & %.0f\\%% \\\\" % (
            str(k).replace("(", "(").replace("]", "]"), len(g), g.dC_mean.mean(), 100 * (g.dC_mean > 0.5).mean()))
    lines += ["\\botrule", "\\end{tabular}"]
    open(os.path.join(args.outdir, "table_training%s.tex" % tag), "w").write("\n".join(lines) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.8))
    for a, xv, lab in [(ax[0], y, "true label $\\log_{10}\\overline{\\mathrm{Var}}$"),
                       (ax[1], p, "predicted label (architecture only)")]:
        a.scatter(xv, d, s=9, alpha=0.5, color="#3b6ea5", edgecolors="none")
        a.axvline(CUTOFF, color="#c0392b", lw=0.8, ls="--")
        a.set_xlabel(lab, fontsize=8); a.set_ylabel("loss decrease $\\Delta C$ (%d %s steps)" % (args.steps, "Adam" if args.optimizer == "adam" else "GD"), fontsize=8)
        a.tick_params(labelsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(args.figdir, "training_outcome%s.pdf" % tag)); plt.close()
    print("wrote training_outcome.json / .csv, table_training.tex, figs/training_outcome.pdf")


if __name__ == "__main__":
    main()
