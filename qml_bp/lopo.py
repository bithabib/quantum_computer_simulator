"""Leave-one-pattern-out (LOPO): can the model transfer to an entanglement
pattern it has never seen?

A categorical pattern index cannot support this question, so the model is
given closed-form descriptors of the entangling graph instead (gates per
layer, edge density, maximum degree, diameter; see
``qml_bp.ansatz.entangling_graph_features``).  For each of the five patterns
we train on the other four and test on the held-out one, and compare with
(i) the same model when the pattern *is* seen (architecture-grouped CV on the
pooled data, scored on that pattern's rows) and (ii) the physics-informed
linear control.

    python -m qml_bp.lopo --main data_bp/bp_dataset_v2.csv \
        --extra data_bp/bp_patterns_v2.csv \
        --outdir paper/qmi/results --figdir paper/qmi/figs
"""

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score

from qml_bp.analyze import (CUTOFF, PhysicsLinear, SEED, SubgroupMean, arch_id,
                            grouped_folds, hgb_cls, hgb_reg, load_dataset)
from qml_bp.ansatz import (ENTANGLE_PATTERNS, FEATURE_COLUMNS,
                           GRAPH_FEATURE_COLUMNS, entangling_graph_features)

warnings.filterwarnings("ignore", category=RuntimeWarning)

# pattern-agnostic feature set: everything except the categorical index,
# plus the four graph descriptors
DESC_COLUMNS = [c for c in FEATURE_COLUMNS if c != "entangle_pattern"] + GRAPH_FEATURE_COLUMNS
NICE = {"linear": "linear", "circular": "circular", "all_to_all": "all-to-all",
        "brickwork": "brickwork", "star": "star"}


def add_graph_features(df):
    cache = {}
    rows = []
    for n, L, p in zip(df.n_qubits, df.n_layers, df.entangle_pattern):
        key = (int(n), int(L), int(p))
        if key not in cache:
            cache[key] = entangling_graph_features(n, L, ENTANGLE_PATTERNS[int(p)])
        rows.append(cache[key])
    g = pd.DataFrame(rows, index=df.index)
    return pd.concat([df, g], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", required=True)
    ap.add_argument("--extra", required=True)
    ap.add_argument("--outdir", default="paper/qmi/results")
    ap.add_argument("--figdir", default="paper/qmi/figs")
    ap.add_argument("--cutoff", type=float, default=CUTOFF)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True); os.makedirs(args.figdir, exist_ok=True)

    df = pd.concat([load_dataset(args.main), load_dataset(args.extra)], ignore_index=True)
    df = df.dropna(subset=["log_var"]).reset_index(drop=True)
    df = add_graph_features(df)
    y = df.log_var.to_numpy(float); yc = (y < args.cutoff).astype(int)
    pat = df.entangle_pattern.to_numpy(int)
    Xd = df[DESC_COLUMNS].to_numpy(float)        # descriptor features
    Xi = df[FEATURE_COLUMNS].to_numpy(float)     # original index features
    Xd0 = df[[c for c in DESC_COLUMNS if c not in ("cost_weight_eff", "cone_qubits", "cone_frac")]].to_numpy(float)  # graph descriptors only, no physics features
    # PhysicsLinear indexes FEATURE_COLUMNS positions; build a view with the
    # same column order for the control.
    groups = arch_id(df).to_numpy()
    R = {"n_rows": int(len(df)),
         "rows_per_pattern": {ENTANGLE_PATTERNS[k]: int((pat == k).sum()) for k in range(5)}}

    # -- seen-pattern reference: grouped CV on pooled data, both feature sets --
    oof_d = np.full(len(df), np.nan); oof_i = np.full(len(df), np.nan)
    oof_dc = np.full(len(df), np.nan)
    for r, tr, te in grouped_folds(groups, 5, 1, SEED):
        oof_d[te] = hgb_reg().fit(Xd[tr], y[tr]).predict(Xd[te])
        oof_i[te] = hgb_reg().fit(Xi[tr], y[tr]).predict(Xi[te])
        oof_dc[te] = hgb_cls().fit(Xd[tr], yc[tr]).predict_proba(Xd[te])[:, 1]
    R["pooled_grouped_cv"] = {
        "descriptor_r2": float(r2_score(y, oof_d)), "index_r2": float(r2_score(y, oof_i)),
        "descriptor_auc": float(roc_auc_score(yc, oof_dc))}

    # -- leave-one-pattern-out ---------------------------------------------
    lopo = {}
    for k, name in enumerate(ENTANGLE_PATTERNS):
        tr = np.where(pat != k)[0]; te = np.where(pat == k)[0]
        m = hgb_reg().fit(Xd[tr], y[tr]); p = m.predict(Xd[te])
        c = hgb_cls().fit(Xd[tr], yc[tr]); pr = c.predict_proba(Xd[te])[:, 1]
        phys = PhysicsLinear("reg", effective=True).fit(Xi[tr], y[tr]).predict(Xi[te])
        m_noeff = hgb_reg().fit(Xd0[tr], y[tr]); p_noeff = m_noeff.predict(Xd0[te])
        sub = SubgroupMean(FEATURE_COLUMNS.index("cost_global")).fit(Xi[tr], y[tr]).predict(Xi[te])
        lopo[name] = {
            "rows": int(len(te)), "label_mean": float(y[te].mean()), "label_std": float(y[te].std()),
            "unseen_r2": float(r2_score(y[te], p)), "unseen_mae": float(mean_absolute_error(y[te], p)),
            "unseen_auc": float(roc_auc_score(yc[te], pr)) if len(np.unique(yc[te])) > 1 else float("nan"),
            "seen_r2": float(r2_score(y[te], oof_d[te])),
            "seen_mae": float(mean_absolute_error(y[te], oof_d[te])),
            "seen_auc": float(roc_auc_score(yc[te], oof_dc[te])) if len(np.unique(yc[te])) > 1 else float("nan"),
            "phys_r2": float(r2_score(y[te], phys)), "phys_mae": float(mean_absolute_error(y[te], phys)),
            "sub_mae": float(mean_absolute_error(y[te], sub)),
            "bias": float(np.mean(p - y[te])),
            "unseen_r2_no_physics": float(r2_score(y[te], p_noeff)),
        }
        print("%-11s unseen R2=%.3f (no physics features %.3f) MAE=%.3f | seen R2=%.3f | phys R2=%.3f | bias %+.2f"
              % (name, lopo[name]["unseen_r2"], lopo[name]["unseen_r2_no_physics"], lopo[name]["unseen_mae"],
                 lopo[name]["seen_r2"], lopo[name]["phys_r2"], lopo[name]["bias"]))
    R["lopo"] = lopo
    json.dump(R, open(os.path.join(args.outdir, "lopo.json"), "w"), indent=1)

    lines = ["\\begin{tabular}{lrccccccc}", "\\toprule",
             " & & \\multicolumn{3}{c}{Pattern unseen} & \\multicolumn{2}{c}{Pattern seen} & \\multicolumn{2}{c}{Physics-linear} \\\\",
             "\\cmidrule(lr){3-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}",
             "Held-out pattern & Rows & $R^2$ & $R^2$, graph only & MAE & $R^2$ & MAE & $R^2$ & MAE \\\\", "\\midrule"]
    for name in ENTANGLE_PATTERNS:
        d = lopo[name]
        lines.append("%s & %d & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f & %.3f \\\\" % (
            NICE[name], d["rows"], d["unseen_r2"], d["unseen_r2_no_physics"], d["unseen_mae"], d["seen_r2"], d["seen_mae"],
            d["phys_r2"], d["phys_mae"]))
    lines += ["\\botrule", "\\end{tabular}"]
    open(os.path.join(args.outdir, "table_lopo.tex"), "w").write("\n".join(lines) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [NICE[p] for p in ENTANGLE_PATTERNS]
    x = np.arange(5); w = 0.27
    plt.figure(figsize=(4.2, 2.7))
    plt.bar(x - w, [lopo[p]["seen_r2"] for p in ENTANGLE_PATTERNS], w, color="#3b6ea5", label="pattern seen (grouped CV)")
    plt.bar(x, [lopo[p]["unseen_r2"] for p in ENTANGLE_PATTERNS], w, color="#e67e22", label="pattern unseen (LOPO)")
    plt.bar(x + w, [lopo[p]["phys_r2"] for p in ENTANGLE_PATTERNS], w, color="#7f8c8d", label="physics-informed linear")
    plt.xticks(x, names, fontsize=8); plt.yticks(fontsize=8)
    plt.ylabel("$R^2$ on held-out pattern", fontsize=8)
    plt.axhline(0, color="k", lw=0.6)
    plt.legend(fontsize=6.5, frameon=False, loc="lower left")
    plt.tight_layout(); plt.savefig(os.path.join(args.figdir, "lopo.pdf")); plt.close()
    print("wrote", os.path.join(args.outdir, "lopo.json"), "and figs/lopo.pdf")


if __name__ == "__main__":
    main()
