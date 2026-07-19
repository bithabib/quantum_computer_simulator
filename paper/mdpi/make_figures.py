"""Generate the paper's figures from the trainability dataset.

Produces two vector PDFs in paper/mdpi/figs/:
  * importance.pdf     -- permutation feature importance (regression)
  * predicted.pdf      -- predicted vs. true log10 gradient variance (hold-out)

    python paper/mdpi/make_figures.py --data data_bp/bp_dataset.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
from qml_bp.ansatz import FEATURE_COLUMNS  # noqa: E402

FIGDIR = os.path.join(HERE, "figs")
NICE = {
    "n_qubits": "qubits", "n_layers": "layers", "n_params": "params",
    "ansatz_type": "ansatz", "entangle_pattern": "entangle pattern",
    "entangler_gate": "entangler gate", "cost_global": "cost locality",
    "n_entanglers": "entanglers", "depth_ratio": "depth ratio",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(FIGDIR, exist_ok=True)

    df = pd.read_csv(os.path.join(ROOT, args.data))
    X = df[FEATURE_COLUMNS].to_numpy(float)
    y = df["log_grad_var"].to_numpy(float)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(df))
    cut = int(len(df) * 0.8)
    tr, te = perm[:cut], perm[cut:]

    reg = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.08)
    reg.fit(X[tr], y[tr])
    pred = reg.predict(X[te])
    r2 = r2_score(y[te], pred)

    # --- Figure 1: permutation importance -------------------------------
    imp = permutation_importance(reg, X[te], y[te], n_repeats=8,
                                 random_state=args.seed)
    order = np.argsort(imp.importances_mean)
    labels = [NICE[FEATURE_COLUMNS[i]] for i in order]
    plt.figure(figsize=(3.4, 2.6))
    plt.barh(range(len(order)), imp.importances_mean[order],
             xerr=imp.importances_std[order], color="#3b6ea5", height=0.7)
    plt.yticks(range(len(order)), labels, fontsize=8)
    plt.xlabel("permutation importance ($\\Delta R^2$)", fontsize=8)
    plt.xticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "importance.pdf"))
    plt.close()

    # --- Figure 2: predicted vs true ------------------------------------
    plt.figure(figsize=(3.4, 2.9))
    plt.scatter(y[te], pred, s=6, alpha=0.35, color="#3b6ea5",
                edgecolors="none")
    lo, hi = min(y[te].min(), pred.min()), max(y[te].max(), pred.max())
    plt.plot([lo, hi], [lo, hi], "k--", lw=1)
    plt.xlabel("true $\\log_{10}\\mathrm{Var}[\\partial C/\\partial\\theta]$",
               fontsize=8)
    plt.ylabel("predicted", fontsize=8)
    plt.title("hold-out  $R^2=%.2f$" % r2, fontsize=9)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "predicted.pdf"))
    plt.close()

    # --- Figure 3: trainability vs qubit count (barren-plateau signature) --
    plt.figure(figsize=(3.6, 2.8))
    for cost, label, color, marker in [
        (0, "local $Z_0$", "#3b6ea5", "o"),
        (1, "global $Z^{\\otimes n}$", "#c0392b", "s"),
    ]:
        sub = df[df.cost_global == cost]
        g = sub.groupby("n_qubits")["log_grad_var"]
        xs = g.mean().index.to_numpy()
        means = g.mean().to_numpy()
        stds = g.std().to_numpy()
        plt.plot(xs, means, marker=marker, color=color, label=label, ms=4, lw=1.5)
        plt.fill_between(xs, means - stds, means + stds, color=color, alpha=0.12)
    plt.xlabel("number of qubits $n$", fontsize=8)
    plt.ylabel("mean $\\log_{10}\\mathrm{Var}[\\partial C/\\partial\\theta]$",
               fontsize=8)
    plt.legend(fontsize=8, frameon=False)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "qubit_trend.pdf"))
    plt.close()

    print("Wrote figs/importance.pdf, figs/predicted.pdf, figs/qubit_trend.pdf"
          "  (hold-out R2=%.3f)" % r2)


if __name__ == "__main__":
    main()
