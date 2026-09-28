"""Collect the training-outcome runs (one JSON per optimizer/shot configuration)
into a single table and JSON for the manuscript.

    python -m qml_bp.training_sweep_table --outdir paper/qmi/results
"""

import argparse
import glob
import json
import os

CONFIGS = [  # (file tag, label in the table)
    ("_gd_exact", "GD, exact gradients"),
    ("_gd_100shots", "GD, 100 shots"),
    ("", "Adam, 1000 shots"),
    ("_adam_100shots", "Adam, 100 shots"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="paper/qmi/results")
    args = ap.parse_args()
    rows = []
    for tag, label in CONFIGS:
        path = os.path.join(args.outdir, "training_outcome%s.json" % tag)
        if not os.path.exists(path):
            continue
        T = json.load(open(path))
        rows.append({"tag": tag, "label": label, "n": T["n_circuits"], "steps": T["steps"],
                     "lr": T["lr"], "shots": T["shots"], "optimizer": T["optimizer"],
                     "spearman_true": T["spearman_true_vs_dC"], "spearman_pred": T["spearman_pred_vs_dC"],
                     "dC_pred_barren": T["dC_mean_pred_barren"], "dC_pred_trainable": T["dC_mean_pred_trainable"],
                     "frac_pred_barren": T["frac_progress_pred_barren"],
                     "frac_pred_trainable": T["frac_progress_pred_trainable"]})
    json.dump(rows, open(os.path.join(args.outdir, "training_sweep.json"), "w"), indent=1)
    lines = ["\\begin{tabular}{lrrcccc}", "\\toprule",
             " & & & \\multicolumn{2}{c}{Spearman $\\rho(\\Delta C,\\cdot)$} & \\multicolumn{2}{c}{Mean $\\Delta C$ (\\% with $\\Delta C>0.5$)} \\\\",
             "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
             "Optimizer, shots & Circuits & Step size & true label & predicted label & predicted barren & predicted trainable \\\\",
             "\\midrule"]
    for r in rows:
        lines.append("%s & %d & %g & %.2f & %.2f & %.2f (%.0f\\%%) & %.2f (%.0f\\%%) \\\\" % (
            r["label"], r["n"], r["lr"], r["spearman_true"], r["spearman_pred"],
            r["dC_pred_barren"], 100 * r["frac_pred_barren"], r["dC_pred_trainable"], 100 * r["frac_pred_trainable"]))
    lines += ["\\botrule", "\\end{tabular}"]
    open(os.path.join(args.outdir, "table_training_sweep.tex"), "w").write("\n".join(lines) + "\n")
    for r in rows:
        print("%-22s n=%3d  rho(true)=%.2f rho(pred)=%.2f  dC barren %.2f (%.0f%%)  trainable %.2f (%.0f%%)" % (
            r["label"], r["n"], r["spearman_true"], r["spearman_pred"], r["dC_pred_barren"],
            100 * r["frac_pred_barren"], r["dC_pred_trainable"], 100 * r["frac_pred_trainable"]))


if __name__ == "__main__":
    main()
