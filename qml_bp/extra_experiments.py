"""Additional experiments for the revised manuscript (analysis only).

  B. Screening precision: rank circuits by out-of-fold predicted label and
     measure how many of the top-k are truly trainable (and how many truly
     barren circuits slip through).  This tests the proposed workflow.
  C. Extrapolation gap: train on n <= k, test on n > k, for k = 6..10, for
     HGB, the MLP and the physics-informed linear control; plus per-n scores
     for the k = 8 split.
  D. Learning curve: architecture-grouped CV R^2 as a function of the number
     of training rows (architectures are sampled whole).

    python -m qml_bp.extra_experiments --data data_bp/bp_dataset_v2.csv \
        --outdir paper/qmi/results --figdir paper/qmi/figs
"""

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

from qml_bp.analyze import (CUTOFF, PhysicsLinear, SEED, StructuredLinear, arch_id,
                            hgb_reg, load_dataset, regressors)
from qml_bp.ansatz import FEATURE_COLUMNS

warnings.filterwarnings("ignore", category=RuntimeWarning)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--outdir", default="paper/qmi/results")
    ap.add_argument("--figdir", default="paper/qmi/figs")
    ap.add_argument("--cutoff", type=float, default=CUTOFF)
    ap.add_argument("--figures-only", action="store_true", help="re-draw figures from extra.json")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True); os.makedirs(args.figdir, exist_ok=True)
    if args.figures_only:
        R = json.load(open(os.path.join(args.outdir, "extra.json")))
        gap = R["gap"]; sizes = [int(s) for s in R["learning_curve"]]
        lc = {s: {m: [R["learning_curve"][str(s)][m][0]] for m in ("HGB", "MLP")} for s in sizes}
        lc_sd = {s: {m: R["learning_curve"][str(s)][m][1] for m in ("HGB", "MLP")} for s in sizes}
        draw_figures(args, gap, sizes, lc, lc_sd)
        return

    df = load_dataset(args.data).dropna(subset=["log_var"]).reset_index(drop=True)
    X = df[FEATURE_COLUMNS].to_numpy(float); y = df.log_var.to_numpy(float)
    nq = df.n_qubits.to_numpy(int); groups = arch_id(df).to_numpy()
    trainable = y >= args.cutoff
    R = {}

    # ---- B. screening precision (out-of-fold HGB) --------------------
    oof = np.full(len(df), np.nan); oof_phys = oof.copy(); oof_struct = oof.copy()
    folds = list(GroupKFold(n_splits=5).split(X, y, groups=groups))
    for tr, te in folds:
        oof[te] = hgb_reg().fit(X[tr], y[tr]).predict(X[te])
        oof_phys[te] = PhysicsLinear("reg").fit(X[tr], y[tr]).predict(X[te])
        oof_struct[te] = StructuredLinear("reg").fit(X[tr], y[tr]).predict(X[te])
    order = np.argsort(-oof)                     # most trainable first
    scr = {"spearman": float(spearmanr(oof, y).correlation),
           "trainable_frac": float(trainable.mean()), "top": {}, "bottom": {}}
    for pct in [1, 5, 10, 20, 50]:
        k = int(len(df) * pct / 100)
        top = order[:k]; bot = order[-k:]
        scr["top"][str(pct)] = {
            "k": k,
            "precision_trainable": float(trainable[top].mean()),
            "barren_leak": float((~trainable[top]).mean()),
            "recall_trainable": float(trainable[top].sum() / trainable.sum()),
            "min_true_label": float(y[top].min()),
            "mean_true_label": float(y[top].mean()),
        }
        scr["bottom"][str(pct)] = {
            "k": k, "precision_barren": float((~trainable[bot]).mean()),
            "mean_true_label": float(y[bot].mean()),
        }
    R["screening"] = scr
    # Ranking by qubit count alone reproduces the global table (every circuit
    # with n <= 5 is above the cutoff), so the informative test is WITHIN a
    # fixed qubit count.
    rng0 = np.random.default_rng(SEED)
    by_n = np.argsort(nq + 1e-6 * rng0.random(len(nq)))
    k20 = int(len(df) * 0.2)
    scr["by_n_only_top20"] = {"precision": float(trainable[by_n[:k20]].mean()),
                              "recall": float(trainable[by_n[:k20]].sum() / trainable.sum())}
    within = {}
    for n in sorted(set(nq)):
        mk = nq == n
        if trainable[mk].all() or not trainable[mk].any():
            continue
        k = max(1, int(mk.sum() * 0.2)); yy = y[mk]; tt = trainable[mk]
        def prec(score):
            idx = np.argsort(-score)[:k]; return float(tt[idx].mean())
        def rec(score):
            idx = np.argsort(-score)[:k]; return float(tt[idx].sum() / tt.sum())
        within[int(n)] = {
            "rows": int(mk.sum()), "above_cutoff_frac": float(tt.mean()),
            "spearman_hgb": float(spearmanr(oof[mk], yy).correlation),
            "spearman_struct": float(spearmanr(oof_struct[mk], yy).correlation),
            "spearman_phys": float(spearmanr(oof_phys[mk], yy).correlation),
            "prec_hgb": prec(oof[mk]), "prec_struct": prec(oof_struct[mk]), "prec_phys": prec(oof_phys[mk]),
            "recall_hgb": rec(oof[mk]),
            "r2_hgb": float(r2_score(yy, oof[mk])), "r2_struct": float(r2_score(yy, oof_struct[mk])),
            "r2_phys": float(r2_score(yy, oof_phys[mk])),
        }
    # per-n R2 for every n (including those entirely above the cutoff)
    R["per_n_r2"] = {int(n): {"hgb": float(r2_score(y[nq == n], oof[nq == n])),
                              "struct": float(r2_score(y[nq == n], oof_struct[nq == n])),
                              "phys": float(r2_score(y[nq == n], oof_phys[nq == n]))} for n in sorted(set(nq))}
    R["screening_within_n"] = within

    # ---- C. extrapolation gap ------------------------------------------
    gap = {}
    for k in range(6, 11):
        tr = np.where(nq <= k)[0]; te = np.where(nq > k)[0]
        entry = {"train_rows": int(len(tr)), "test_rows": int(len(te))}
        models = {"HGB": hgb_reg(), "MLP": regressors()["MLP"], "Structured linear": StructuredLinear("reg"),
                  "Physics-informed linear": PhysicsLinear("reg")}
        for name, m in models.items():
            p = m.fit(X[tr], y[tr]).predict(X[te])
            entry[name] = {"r2": float(r2_score(y[te], p)), "mae": float(mean_absolute_error(y[te], p))}
            if k == 8:  # per-n breakdown
                entry[name]["per_n"] = {int(n): {"r2": float(r2_score(y[te][nq[te] == n], p[nq[te] == n])),
                                                 "mae": float(mean_absolute_error(y[te][nq[te] == n], p[nq[te] == n]))}
                                        for n in sorted(set(nq[te]))}
        sv = [r2_score(y[te], regressors(s)["MLP"].fit(X[tr], y[tr]).predict(X[te])) for s in range(10)]
        entry["MLP_seeds"] = {"mean": float(np.mean(sv)), "sd": float(np.std(sv, ddof=1)),
                              "min": float(np.min(sv)), "max": float(np.max(sv)), "n": 10}
        gap[str(k)] = entry
        print("gap k=%d  HGB %.3f  MLP %.3f  phys %.3f" % (k, entry["HGB"]["r2"], entry["MLP"]["r2"],
                                                          entry["Physics-informed linear"]["r2"]))
    R["gap"] = gap

    # ---- D. learning curve ---------------------------------------------
    sizes = [250, 500, 1000, 2000, 4000, 8000, 16000]
    rng = np.random.default_rng(SEED)
    lc = {s: {"HGB": [], "MLP": []} for s in sizes}
    for tr, te in folds:
        g_tr = groups[tr]; uniq = np.unique(g_tr); perm = rng.permutation(uniq)
        for s in sizes:
            chosen = set(); n_rows = 0
            for g in perm:
                chosen.add(g); n_rows += int((g_tr == g).sum())
                if n_rows >= s:
                    break
            sub = tr[np.isin(g_tr, list(chosen))]
            lc[s]["HGB"].append(float(r2_score(y[te], hgb_reg().fit(X[sub], y[sub]).predict(X[te]))))
            lc[s]["MLP"].append(float(r2_score(y[te], regressors()["MLP"].fit(X[sub], y[sub]).predict(X[te]))))
    R["learning_curve"] = {str(s): {m: [float(np.mean(v)), float(np.std(v, ddof=1))] for m, v in d.items()}
                           for s, d in lc.items()}
    for s in sizes:
        print("learning curve %5d rows: HGB %.3f  MLP %.3f" % (s, np.mean(lc[s]["HGB"]), np.mean(lc[s]["MLP"])))

    json.dump(R, open(os.path.join(args.outdir, "extra.json"), "w"), indent=1)
    lc_sd = {s: {m: float(np.std(v, ddof=1)) for m, v in d.items()} for s, d in lc.items()}

    # ---- tables --------------------------------------------------------
    lines = ["\\begin{tabular}{rrcccc}", "\\toprule",
             "Top $k$ & Circuits & Truly trainable (\\%) & Truly barren (\\%) & Recall of trainable (\\%) & Mean true label \\\\",
             "\\midrule"]
    for pct in ["1", "5", "10", "20", "50"]:
        t = scr["top"][pct]
        lines.append("%s\\%% & %d & %.1f & %.1f & %.1f & $%.2f$ \\\\" % (
            pct, t["k"], 100 * t["precision_trainable"], 100 * t["barren_leak"],
            100 * t["recall_trainable"], t["mean_true_label"]))
    lines += ["\\botrule", "\\end{tabular}"]
    open(os.path.join(args.outdir, "table_screening.tex"), "w").write("\n".join(lines) + "\n")

    w = R["screening_within_n"]
    lines = ["\\begin{tabular}{rrccccccc}", "\\toprule",
             " & & & \\multicolumn{3}{c}{Spearman $\\rho$ with true label} & \\multicolumn{3}{c}{Above cutoff in top 20\\% (\\%)} \\\\",
             "\\cmidrule(lr){4-6}\\cmidrule(lr){7-9}",
             "$n$ & Circuits & Above cutoff (\\%) & HGB & Struct.~linear & Physics-linear & HGB & Struct.~linear & Physics-linear \\\\",
             "\\midrule"]
    for n in sorted(w):
        d = w[n]
        lines.append("%d & %d & %.0f & %.3f & %.3f & %.3f & %.1f & %.1f & %.1f \\\\" % (
            n, d["rows"], 100 * d["above_cutoff_frac"], d["spearman_hgb"], d["spearman_struct"], d["spearman_phys"],
            100 * d["prec_hgb"], 100 * d["prec_struct"], 100 * d["prec_phys"]))
    lines += ["\\botrule", "\\end{tabular}"]
    open(os.path.join(args.outdir, "table_screening_within.tex"), "w").write("\n".join(lines) + "\n")

    lines = ["\\begin{tabular}{crrcccc}", "\\toprule",
             "Train $n\\le k$ & Train rows & Test rows & HGB & MLP (10 seeds, mean $\\pm$ sd [min, max]) & Structured linear & Physics-linear \\\\", "\\midrule"]
    for k in range(6, 11):
        e = gap[str(k)]; s = e["MLP_seeds"]
        lines.append("$k=%d$ & %d & %d & %.3f & $%.3f \\pm %.3f$ [%.2f, %.2f] & %.3f & %.3f \\\\" % (
            k, e["train_rows"], e["test_rows"], e["HGB"]["r2"], s["mean"], s["sd"], s["min"], s["max"],
            e["Structured linear"]["r2"], e["Physics-informed linear"]["r2"]))
    lines += ["\\botrule", "\\end{tabular}"]
    open(os.path.join(args.outdir, "table_gap.tex"), "w").write("\n".join(lines) + "\n")

    draw_figures(args, gap, sizes, lc, lc_sd)


def draw_figures(args, gap, sizes, lc, lc_sd):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, RED, GREY, ORANGE = "#3b6ea5", "#c0392b", "#7f8c8d", "#e67e22"

    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.3))
    ks = list(range(6, 11))
    GREEN = "#27ae60"
    for name, c, mk in [("HGB", BLUE, "o"), ("Structured linear", GREEN, "D"), ("Physics-informed linear", GREY, "^")]:
        ax[0].plot(ks, [gap[str(k)][name]["r2"] for k in ks], marker=mk, color=c, label=name, ms=4, lw=1.4)
    mu = np.array([gap[str(k)]["MLP_seeds"]["mean"] for k in ks])
    lo_ = np.array([gap[str(k)]["MLP_seeds"]["min"] for k in ks]); hi_ = np.array([gap[str(k)]["MLP_seeds"]["max"] for k in ks])
    ax[0].plot(ks, mu, marker="s", color=ORANGE, label="MLP (10 seeds)", ms=4, lw=1.4)
    ax[0].fill_between(ks, lo_, hi_, color=ORANGE, alpha=0.18)
    ax[0].set_xlabel("largest training qubit count $k$ (test: $n>k$)", fontsize=10)
    ax[0].set_ylabel("$R^2$ on all $n>k$", fontsize=10); ax[0].tick_params(labelsize=9)
    ax[0].axhline(0, color="k", lw=0.5)
    ns = sorted(int(n) for n in gap["8"]["HGB"]["per_n"])
    for name, c, mk in [("HGB", BLUE, "o"), ("MLP", ORANGE, "s"), ("Structured linear", "#27ae60", "D"),
                        ("Physics-informed linear", GREY, "^")]:
        pn = gap["8"][name]["per_n"]
        ax[1].plot(ns, [pn[n]["r2"] if n in pn else pn[str(n)]["r2"] for n in ns],
                   marker=mk, color=c, label=name, ms=4, lw=1.4)
    ax[1].set_xlabel("test qubit count $n$ (trained on $n \\leq 8$)", fontsize=10)
    ax[1].set_ylabel("$R^2$ at that $n$", fontsize=10); ax[1].tick_params(labelsize=9)
    ax[1].set_xticks(ns)
    ax[1].axhline(0, color="k", lw=0.5)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.0))
    plt.tight_layout(rect=(0, 0, 1, 0.9)); plt.savefig(os.path.join(args.figdir, "gap.pdf"), bbox_inches="tight"); plt.close()

    plt.figure(figsize=(3.6, 2.7))
    for name, c, mk in [("HGB", BLUE, "o"), ("MLP", ORANGE, "s")]:
        mu = [np.mean(lc[s][name]) for s in sizes]; sd = [lc_sd[s][name] for s in sizes]
        plt.errorbar(sizes, mu, yerr=sd, marker=mk, color=c, label=name, ms=4, lw=1.4, capsize=2)
    plt.xscale("log"); plt.xlabel("training rows (whole architectures)", fontsize=8)
    plt.ylabel("grouped-CV $R^2$", fontsize=8); plt.xticks(fontsize=8); plt.yticks(fontsize=8)
    plt.legend(fontsize=7, frameon=False); plt.tight_layout()
    plt.savefig(os.path.join(args.figdir, "learning_curve.pdf")); plt.close()
    print("wrote figs/gap.pdf, figs/learning_curve.pdf")


if __name__ == "__main__":
    main()
