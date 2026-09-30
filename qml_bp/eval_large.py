"""Evaluate predictors trained on n <= 12 on a held-out 13- and 14-qubit set.

    python -m qml_bp.eval_large --train data_bp/bp_dataset_v2.csv \
        --test data_bp/bp_large_v2.csv --outdir paper/qmi/results --figdir paper/qmi/figs
"""

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score

from qml_bp.analyze import (CUTOFF, PhysicsLinear, StructuredLinear, SubgroupMean,
                            bootstrap_ci, hgb_cls, hgb_reg, regressors, classifiers,
                            load_dataset)
from qml_bp.ansatz import FEATURE_COLUMNS

warnings.filterwarnings("ignore", category=RuntimeWarning)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--outdir", default="paper/qmi/results")
    ap.add_argument("--figdir", default="paper/qmi/figs")
    args = ap.parse_args()
    tr = load_dataset(args.train).dropna(subset=["log_var"])
    te = load_dataset(args.test).dropna(subset=["log_var"]).reset_index(drop=True)
    Xtr, ytr = tr[FEATURE_COLUMNS].to_numpy(float), tr.log_var.to_numpy(float)
    Xte, yte = te[FEATURE_COLUMNS].to_numpy(float), te.log_var.to_numpy(float)
    ctr, cte = (ytr < CUTOFF).astype(int), (yte < CUTOFF).astype(int)
    nq = te.n_qubits.to_numpy(int); glob = te.cost_global.to_numpy(int)
    R = {"train_rows": int(len(tr)), "test_rows": int(len(te)),
         "test_rows_by_n": {int(n): int((nq == n).sum()) for n in sorted(set(nq))},
         "label_mean": float(yte.mean()), "label_std": float(yte.std()),
         "barren_frac": float(cte.mean()), "all_structural_rows": int(pd.read_csv(args.test).log_var.isna().sum()),
         "models": {}}
    preds = {}
    models = {"Hist Gradient Boosting": hgb_reg(), "MLP": regressors()["MLP"],
              "Random Forest": regressors()["Random Forest"], "k-NN": regressors()["k-NN"],
              "Linear baseline": regressors()["Linear baseline"],
              "Structured linear": StructuredLinear("reg"),
              "Physics-informed linear (effective weight)": PhysicsLinear("reg", effective=True),
              "Physics-informed linear": PhysicsLinear("reg"),
              "Cost-locality subgroup mean": SubgroupMean(FEATURE_COLUMNS.index("cost_global"))}
    for name, m in models.items():
        p = m.fit(Xtr, ytr).predict(Xte); preds[name] = p
        entry = {"r2": float(r2_score(yte, p)), "mae": float(mean_absolute_error(yte, p)),
                 "r2_ci": bootstrap_ci(r2_score, yte, p), "mae_ci": bootstrap_ci(mean_absolute_error, yte, p),
                 "per_n": {}, "per_cost": {}}
        for n in sorted(set(nq)):
            mk = nq == n
            entry["per_n"][int(n)] = {"r2": float(r2_score(yte[mk], p[mk])), "mae": float(mean_absolute_error(yte[mk], p[mk]))}
        for g, lab in [(0, "local"), (1, "global")]:
            mk = glob == g
            entry["per_cost"][lab] = {"r2": float(r2_score(yte[mk], p[mk])), "mae": float(mean_absolute_error(yte[mk], p[mk]))}
        R["models"][name] = entry
        print("%-28s R2 %.3f [%.3f, %.3f]  MAE %.3f | n=13 %.3f  n=14 %.3f | local %.3f global %.3f" % (
            name, entry["r2"], *entry["r2_ci"], entry["mae"], entry["per_n"][13]["r2"], entry["per_n"][14]["r2"],
            entry["per_cost"]["local"]["r2"], entry["per_cost"]["global"]["r2"]))
    # classification
    for name, m in {"Hist Gradient Boosting": hgb_cls(), "MLP": classifiers()["MLP"]}.items():
        pr = m.fit(Xtr, ctr).predict_proba(Xte)[:, 1]
        R["models"][name]["auc"] = float(roc_auc_score(cte, pr)) if len(np.unique(cte)) > 1 else float("nan")
    # seed spread for the MLP (10 seeds)
    sv = [r2_score(yte, regressors(s)["MLP"].fit(Xtr, ytr).predict(Xte)) for s in range(10)]
    R["mlp_seeds"] = {"mean": float(np.mean(sv)), "sd": float(np.std(sv, ddof=1)),
                      "min": float(np.min(sv)), "max": float(np.max(sv)), "n": 10}
    json.dump(R, open(os.path.join(args.outdir, "large.json"), "w"), indent=1)

    lines = ["\\begin{tabular}{lcccccc}", "\\toprule",
             "Model & $R^2$ [95\\% CI] & MAE & $R^2$, $n=13$ & $R^2$, $n=14$ & $R^2$, local & $R^2$, global \\\\", "\\midrule"]
    for name in ["Hist Gradient Boosting", "MLP", "Random Forest", "k-NN", "Linear baseline",
                 "Structured linear", "Physics-informed linear (effective weight)",
                 "Physics-informed linear", "Cost-locality subgroup mean"]:
        e = R["models"][name]
        if name == "Structured linear":
            lines.append("\\midrule")
        lines.append("%s & %.3f [%.3f, %.3f] & %.3f & %.3f & %.3f & %.3f & %.3f \\\\" % (
            name, e["r2"], e["r2_ci"][0], e["r2_ci"][1], e["mae"], e["per_n"][13]["r2"], e["per_n"][14]["r2"],
            e["per_cost"]["local"]["r2"], e["per_cost"]["global"]["r2"]))
    lines += ["\\botrule", "\\end{tabular}"]
    open(os.path.join(args.outdir, "table_large.tex"), "w").write("\n".join(lines) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.8), sharey=True)
    for a, name, c in [(ax[0], "Hist Gradient Boosting", "#3b6ea5"), (ax[1], "MLP", "#e67e22")]:
        a.scatter(yte, preds[name], s=7, alpha=0.4, color=c, edgecolors="none")
        lo, hi = yte.min() - 0.2, yte.max() + 0.2
        a.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        a.set_title("%s: $R^2=%.2f$" % ("HGB" if name.startswith("Hist") else name, R["models"][name]["r2"]), fontsize=9)
        a.set_xlabel("true label, $n\\in\\{13,14\\}$", fontsize=8); a.tick_params(labelsize=8)
    ax[0].set_ylabel("predicted (trained on $n \\leq 12$)", fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(args.figdir, "large.pdf")); plt.close()
    print("wrote large.json, table_large.tex, figs/large.pdf")


if __name__ == "__main__":
    main()
