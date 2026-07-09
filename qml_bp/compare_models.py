"""Compare several ML models for trainability prediction and pick the best.

Evaluates a zoo of scikit-learn models on the same two protocols as train.py:
  * random hold-out split       -> generalization to unseen architectures
  * extrapolation split         -> train small n_qubits, test larger

Regression target  : log10 Var[dC/dtheta]        (R^2, MAE)
Classification tgt : barren vs trainable          (accuracy, AUC)

Prints a ranked comparison table and the winning model for each task, and can
emit a LaTeX table fragment for the paper.

    python -m qml_bp.compare_models --data data_bp/bp_dataset.csv
    python -m qml_bp.compare_models --data data_bp/bp_dataset.csv --latex paper/latex/table_models.tex
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from qml_bp.ansatz import FEATURE_COLUMNS

SEED = 0


def regressors():
    """name -> fresh regressor (scaled where the model needs it)."""
    return {
        "Linear baseline": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "k-NN": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=10)),
        "MLP": make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=800,
                         random_state=SEED),
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=400, n_jobs=-1, random_state=SEED),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=400, n_jobs=-1, random_state=SEED),
        "Hist Gradient Boosting": HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.08, random_state=SEED),
    }


def classifiers():
    return {
        "Linear baseline": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000)),
        "k-NN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=10)),
        "MLP": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=800,
                          random_state=SEED),
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=400, n_jobs=-1, random_state=SEED),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=400, n_jobs=-1, random_state=SEED),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.08, random_state=SEED),
    }


def splits(df):
    """Return (tr, te) for the random hold-out and (tr2, te2) for extrapolation."""
    rng = np.random.default_rng(SEED)
    n = len(df)
    perm = rng.permutation(n)
    cut = int(n * 0.8)
    hold = (perm[:cut], perm[cut:])

    kcut = int(df.n_qubits.max()) - 2
    nq = df.n_qubits.to_numpy()
    extrap = (np.where(nq <= kcut)[0], np.where(nq > kcut)[0])
    return hold, extrap, kcut


def eval_regression(df, X, y):
    (tr, te), (tr2, te2), kcut = splits(df)
    rows = []
    for name, model in regressors().items():
        m = model
        m.fit(X[tr], y[tr])
        p = m.predict(X[te])
        r2_h, mae_h = r2_score(y[te], p), mean_absolute_error(y[te], p)
        m2 = regressors()[name]
        m2.fit(X[tr2], y[tr2])
        p2 = m2.predict(X[te2])
        r2_e, mae_e = r2_score(y[te2], p2), mean_absolute_error(y[te2], p2)
        rows.append((name, r2_h, mae_h, r2_e, mae_e))
        print("  %-24s holdout R2=%.3f MAE=%.3f | extrap R2=%.3f MAE=%.3f"
              % (name, r2_h, mae_h, r2_e, mae_e))
    rows.sort(key=lambda r: r[1], reverse=True)
    return rows, kcut


def eval_classification(df, X, y):
    (tr, te), (tr2, te2), kcut = splits(df)
    rows = []
    for name, model in classifiers().items():
        m = model
        m.fit(X[tr], y[tr])
        pred = m.predict(X[te])
        prob = m.predict_proba(X[te])[:, 1]
        acc_h = accuracy_score(y[te], pred)
        auc_h = roc_auc_score(y[te], prob) if len(np.unique(y[te])) > 1 else float("nan")
        m2 = classifiers()[name]
        m2.fit(X[tr2], y[tr2])
        pred2 = m2.predict(X[te2])
        prob2 = m2.predict_proba(X[te2])[:, 1]
        acc_e = accuracy_score(y[te2], pred2)
        auc_e = roc_auc_score(y[te2], prob2) if len(np.unique(y[te2])) > 1 else float("nan")
        rows.append((name, acc_h, auc_h, acc_e, auc_e))
        print("  %-24s holdout acc=%.3f AUC=%.3f | extrap acc=%.3f AUC=%.3f"
              % (name, acc_h, auc_h, acc_e, auc_e))
    rows.sort(key=lambda r: (np.nan_to_num(r[2]), r[1]), reverse=True)
    return rows


def write_latex(path, reg_rows, cls_rows, kcut):
    """Emit a booktabs LaTeX table fragment comparing all models."""
    # merge on model name
    cls_map = {r[0]: r for r in cls_rows}
    lines = [
        "% Auto-generated by qml_bp/compare_models.py -- do not edit by hand.",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        " & \\multicolumn{2}{c}{Regression $R^2$} "
        "& \\multicolumn{2}{c}{Classification AUC} \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
        "Model & Hold-out & Extrap. & Hold-out & Extrap. \\\\",
        "\\midrule",
    ]
    best = reg_rows[0][0]
    for name, r2_h, _mae_h, r2_e, _mae_e in reg_rows:
        c = cls_map.get(name)
        auc_h = "%.3f" % c[2] if c else "--"
        auc_e = "%.3f" % c[4] if c else "--"
        label = "\\textbf{%s}" % name if name == best else name
        lines.append("%s & %.3f & %.3f & %s & %s \\\\"
                     % (label, r2_h, r2_e, auc_h, auc_e))
    lines += ["\\bottomrule", "\\end{tabular}"]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("Wrote LaTeX table -> %s" % path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--barren-threshold", type=float, default=-2.0)
    ap.add_argument("--latex", default=None, help="write a LaTeX table fragment here")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    print("Loaded %d rows | qubits %d..%d | %.1f%% global-cost"
          % (len(df), df.n_qubits.min(), df.n_qubits.max(),
             100 * df.cost_global.mean()))
    X = df[FEATURE_COLUMNS].to_numpy(float)
    y_reg = df["log_grad_var"].to_numpy(float)
    y_cls = (y_reg < args.barren_threshold).astype(int)

    print("\n=== Regression (predict log10 gradient variance) ===")
    reg_rows, kcut = eval_regression(df, X, y_reg)
    print("\n=== Classification (barren vs trainable) ===")
    cls_rows = eval_classification(df, X, y_cls)

    print("\n" + "=" * 60)
    print("BEST regressor  : %s  (hold-out R2=%.3f, extrap R2=%.3f)"
          % (reg_rows[0][0], reg_rows[0][1], reg_rows[0][3]))
    print("BEST classifier : %s  (hold-out AUC=%.3f, extrap AUC=%.3f)"
          % (cls_rows[0][0], cls_rows[0][2], cls_rows[0][4]))
    print("=" * 60)

    if args.latex:
        write_latex(args.latex, reg_rows, cls_rows, kcut)


if __name__ == "__main__":
    main()
