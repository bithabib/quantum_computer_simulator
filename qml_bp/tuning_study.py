"""Does the conclusion depend on model choice or tuning?

Nine learner configurations (the paper's settings, a nested randomized
hyperparameter search, larger and ensembled networks, kernel methods) against
the fitted linear rule, on unseen architectures, with the noise ceiling.

    python -m qml_bp.tuning_study --main data_bp/bp_dataset_v2.csv \
        --tied data_bp/bp_tied_v2.csv --outdir paper/qmi/results
"""

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from qml_bp.analyze import StructuredLinear, arch_id, hgb_reg, load_dataset
from qml_bp.ansatz import FEATURE_COLUMNS

warnings.filterwarnings("ignore")
N_ITER = 20
SPACE = {"learning_rate": [0.02, 0.04, 0.08, 0.15], "max_iter": [400, 800, 1500],
         "max_leaf_nodes": [15, 31, 63, 127], "min_samples_leaf": [5, 10, 20, 40],
         "l2_regularization": [0, 0.1, 1.0], "max_bins": [128, 255]}
ORDER = ["Fitted linear rule (structured linear)", "HGB, paper settings",
         "HGB, nested randomized search (%d configurations)" % N_ITER, "Extra trees, 1000 trees",
         "MLP (256, 256, 128), early stopping", "Ensemble of 5 MLPs (128, 64)",
         "Blend of HGB and the MLP ensemble", "SVR, RBF kernel", "Kernel ridge, RBF kernel"]


class Mean:
    def __init__(self, ms): self.ms = ms
    def predict(self, Z): return np.mean([m.predict(Z) for m in self.ms], axis=0)


def study(X, y, groups):
    w = pd.Series(y).groupby(groups).transform(lambda s: s.var(ddof=1) if len(s) > 1 else np.nan)
    out = {"rows": int(len(y)), "r2_ceiling": float(1 - np.nanmean(w) / np.var(y)), "models": {}}
    outer = list(GroupKFold(5).split(X, y, groups))
    sub = lambda tr: tr if len(tr) <= 6000 else np.random.default_rng(0).choice(tr, 6000, replace=False)

    def ens(tr):
        return Mean([make_pipeline(StandardScaler(), MLPRegressor((128, 64), max_iter=800, random_state=s)).fit(X[tr], y[tr])
                     for s in range(5)])

    def tuned(tr):
        rs = RandomizedSearchCV(HistGradientBoostingRegressor(random_state=0, early_stopping=False), SPACE,
                                n_iter=N_ITER, cv=GroupKFold(3), scoring="r2", random_state=0, n_jobs=4)
        rs.fit(X[tr], y[tr], groups=groups[tr]); return rs.best_estimator_

    makers = [
        lambda tr: StructuredLinear("reg").fit(X[tr], y[tr]),
        lambda tr: hgb_reg().fit(X[tr], y[tr]),
        tuned,
        lambda tr: ExtraTreesRegressor(1000, n_jobs=-1, random_state=0).fit(X[tr], y[tr]),
        lambda tr: make_pipeline(StandardScaler(), MLPRegressor((256, 256, 128), max_iter=1500, early_stopping=True,
                                                               n_iter_no_change=40, random_state=0)).fit(X[tr], y[tr]),
        ens,
        lambda tr: Mean([hgb_reg().fit(X[tr], y[tr]), ens(tr)]),
        lambda tr: make_pipeline(StandardScaler(), SVR(C=30, epsilon=0.02)).fit(X[sub(tr)], y[sub(tr)]),
        lambda tr: make_pipeline(StandardScaler(), KernelRidge(alpha=0.01, kernel="rbf", gamma=0.05)).fit(X[sub(tr)], y[sub(tr)]),
    ]
    for name, mk in zip(ORDER, makers):
        r = [r2_score(y[te], mk(tr).predict(X[te])) for tr, te in outer]
        out["models"][name] = [float(np.mean(r)), float(np.std(r, ddof=1))]
        print("  %-52s %.4f +- %.4f" % (name, np.mean(r), np.std(r, ddof=1)), flush=True)
    learners = {k: v[0] for k, v in out["models"].items() if k != ORDER[0]}
    out["best_learner"] = max(learners, key=learners.get); out["best_r2"] = max(learners.values())
    out["spread_learners"] = float(max(learners.values()) - min(learners.values()))
    out["gap_to_rule"] = float(out["best_r2"] - out["models"][ORDER[0]][0])
    out["headroom"] = float(out["r2_ceiling"] - out["best_r2"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", required=True)
    ap.add_argument("--tied", required=True)
    ap.add_argument("--outdir", default="paper/qmi/results")
    args = ap.parse_args()
    R = {}
    for key, path in [("main", args.main), ("tied", args.tied)]:
        d = load_dataset(path).dropna(subset=["log_var"]).reset_index(drop=True)
        print("== %s (%d rows) ==" % (key, len(d)), flush=True)
        R[key] = study(d[FEATURE_COLUMNS].to_numpy(float), d.log_var.to_numpy(float), arch_id(d).to_numpy())
    json.dump(R, open(os.path.join(args.outdir, "tuning.json"), "w"), indent=1)
    lines = ["\\begin{tabular}{lcc}", "\\toprule", "Model & Named patterns & Shared parameters \\\\", "\\midrule"]
    for i, name in enumerate(ORDER):
        if i == 1:
            lines.append("\\midrule")
        lines.append("%s & $%.3f \\pm %.3f$ & $%.3f \\pm %.3f$ \\\\" % (name, *R["main"]["models"][name], *R["tied"]["models"][name]))
    lines += ["\\midrule", "Noise ceiling & %.3f & %.3f \\\\" % (R["main"]["r2_ceiling"], R["tied"]["r2_ceiling"]),
              "\\botrule", "\\end{tabular}"]
    open(os.path.join(args.outdir, "table_tuning.tex"), "w").write("\n".join(lines) + "\n")
    print("wrote tuning.json, table_tuning.tex")


if __name__ == "__main__":
    main()
