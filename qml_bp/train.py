"""Train models to predict circuit trainability from architecture.

Two tasks on the generated dataset:
  * regression  -> predict log10(Var[dC/dtheta])   (how trainable)
  * classification -> predict barren vs trainable  (thresholded variance)

Reports both a random hold-out split and the headline **extrapolation** split
(train on small qubit counts, test on larger ones), plus permutation feature
importance so we can say *which* design choices drive trainability.

    python -m qml_bp.train --data data_bp/bp_dataset.csv --barren-threshold -2
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)

from qml_bp.ansatz import FEATURE_COLUMNS


def _report_regression(name, y_true, y_pred):
    print("  [%s] R2 = %.3f   MAE = %.3f (in log10 units)"
          % (name, r2_score(y_true, y_pred), mean_absolute_error(y_true, y_pred)))


def _report_classification(name, y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    line = "  [%s] accuracy = %.3f" % (name, acc)
    if len(np.unique(y_true)) > 1:
        line += "   AUC = %.3f" % roc_auc_score(y_true, y_prob)
    print(line)


def main():
    ap = argparse.ArgumentParser(description="Train trainability predictors.")
    ap.add_argument("--data", required=True, help="CSV from qml_bp.generate")
    ap.add_argument("--barren-threshold", type=float, default=-2.0,
                    help="log10(var) below this = 'barren' (default -2 => var<0.01)")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--extrap-qubit", type=int, default=None,
                    help="train on n_qubits <= this, test on the rest "
                         "(default: max_qubits - 2)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    print("Loaded %d rows from %s" % (len(df), args.data))
    print("qubit range %d..%d, layers %d..%d, %.1f%% global-cost"
          % (df.n_qubits.min(), df.n_qubits.max(),
             df.n_layers.min(), df.n_layers.max(), 100 * df.cost_global.mean()))

    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_reg = df["log_grad_var"].to_numpy(dtype=float)
    y_cls = (y_reg < args.barren_threshold).astype(int)  # 1 = barren
    print("barren fraction (log_var < %.1f): %.1f%%"
          % (args.barren_threshold, 100 * y_cls.mean()))

    rng = np.random.default_rng(args.seed)
    n = len(df)
    perm = rng.permutation(n)
    cut = int(n * (1 - args.test_frac))
    tr, te = perm[:cut], perm[cut:]

    print("\n=== Random hold-out split ===")
    reg = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.08)
    reg.fit(X[tr], y_reg[tr])
    _report_regression("regression", y_reg[te], reg.predict(X[te]))

    if len(np.unique(y_cls[tr])) > 1:
        clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08)
        clf.fit(X[tr], y_cls[tr])
        prob = clf.predict_proba(X[te])[:, 1]
        _report_classification("classification", y_cls[te], clf.predict(X[te]), prob)
    else:
        print("  [classification] skipped (only one class present)")

    # -- headline: extrapolate to larger, unseen qubit counts ----------
    kmax = int(df.n_qubits.max())
    kcut = args.extrap_qubit if args.extrap_qubit is not None else kmax - 2
    tr2 = np.where(df.n_qubits.to_numpy() <= kcut)[0]
    te2 = np.where(df.n_qubits.to_numpy() > kcut)[0]
    print("\n=== Extrapolation split (train n<=%d, test n>%d) ===" % (kcut, kcut))
    if len(tr2) and len(te2):
        reg2 = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.08)
        reg2.fit(X[tr2], y_reg[tr2])
        _report_regression("regression", y_reg[te2], reg2.predict(X[te2]))
        print("  (train rows %d, test rows %d)" % (len(tr2), len(te2)))
    else:
        print("  not enough qubit spread to form an extrapolation split")

    # -- interpretability: which features matter ------------------------
    print("\n=== Permutation feature importance (regression) ===")
    imp = permutation_importance(reg, X[te], y_reg[te], n_repeats=8,
                                 random_state=args.seed)
    order = np.argsort(imp.importances_mean)[::-1]
    for i in order:
        print("  %-18s %.4f" % (FEATURE_COLUMNS[i], imp.importances_mean[i]))


if __name__ == "__main__":
    main()
