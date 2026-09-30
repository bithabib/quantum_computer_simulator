"""Extrapolation far beyond statevector reach, with Clifford-sampled labels.

Train on Clifford-labeled circuits with n <= 12, test on circuits with
13 <= n <= 32.  The label is log10 of the mean gradient variance over ALL
parameters (structural zeros included), which Clifford sampling estimates
without bias.  Circuits whose variance is below the sampling resolution are
censored: only an upper bound on their label is known, and they are scored
by whether the prediction respects that bound.

    python -m qml_bp.clifford_study --train data_bp/bp_clifford_train.csv \
        --test data_bp/bp_clifford_test.csv --outdir paper/qmi/results --figdir paper/qmi/figs
"""

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from qml_bp.analyze import (PhysicsLinear, StructuredLinear, hgb_reg, load_dataset,
                            regressors)
from qml_bp.ansatz import FEATURE_COLUMNS

warnings.filterwarnings("ignore", category=RuntimeWarning)
BINS = [(13, 16), (17, 20), (21, 24), (25, 28), (29, 32)]
N_SEEDS = 10


def upper_bound(d):
    """log10 of a 95% upper bound on the mean variance of a censored circuit
    (Poisson: observed h events in S*m trials -> bound (h + 3) / (S*m))."""
    return np.log10((d.total_hits.to_numpy() + 3.0) / (d.samples.to_numpy() * d.n_params.to_numpy()))


def fit_predict(models, Xtr, ytr, Xte):
    return {name: make().fit(Xtr, ytr).predict(Xte) for name, make in models.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--outdir", default="paper/qmi/results")
    ap.add_argument("--figdir", default="paper/qmi/figs")
    args = ap.parse_args()
    tr_all = load_dataset(args.train); te_all = load_dataset(args.test)
    tr = tr_all[tr_all.resolved == 1].reset_index(drop=True)
    te = te_all[te_all.resolved == 1].reset_index(drop=True)
    ce = te_all[te_all.resolved == 0].reset_index(drop=True)
    Xtr, ytr = tr[FEATURE_COLUMNS].to_numpy(float), tr.log_var_all.to_numpy(float)
    Xte, yte = te[FEATURE_COLUMNS].to_numpy(float), te.log_var_all.to_numpy(float)
    Xce = ce[FEATURE_COLUMNS].to_numpy(float); ub = upper_bound(ce) if len(ce) else np.array([])
    nq = te.n_qubits.to_numpy(int)

    models = {
        "Structured linear": lambda: StructuredLinear("reg", dilution=True),
        "MLP": lambda: regressors(0)["MLP"],
        "Hist Gradient Boosting": lambda: hgb_reg(),
        "Physics-informed linear (effective weight)": lambda: PhysicsLinear("reg", effective=True),
        "Physics-informed linear": lambda: PhysicsLinear("reg"),
    }
    pred = fit_predict(models, Xtr, ytr, Xte)
    pred_ce = fit_predict(models, Xtr, ytr, Xce) if len(ce) else {}
    # MLP over seeds
    mlp_seed = [regressors(s)["MLP"].fit(Xtr, ytr).predict(Xte) for s in range(N_SEEDS)]

    R = {"train_rows": int(len(tr)), "train_unresolved": int((tr_all.resolved == 0).sum()),
         "test_rows": int(len(te_all)), "test_resolved": int(len(te)), "test_censored": int(len(ce)),
         "median_rel_se_test": float(np.nanmedian(te.rel_se)),
         "median_samples_test": int(te_all.samples.median()),
         "label_min_resolved": float(yte.min()), "bins": {}, "overall": {}}
    for name, p in pred.items():
        R["overall"][name] = {"r2": float(r2_score(yte, p)), "mae": float(mean_absolute_error(yte, p)),
                              "bias": float(np.mean(p - yte))}
    sv = [r2_score(yte, p) for p in mlp_seed]; sm = [mean_absolute_error(yte, p) for p in mlp_seed]
    R["overall"]["MLP_seeds"] = {"r2_mean": float(np.mean(sv)), "r2_sd": float(np.std(sv, ddof=1)),
                                 "r2_min": float(np.min(sv)), "r2_max": float(np.max(sv)),
                                 "mae_mean": float(np.mean(sm)), "mae_sd": float(np.std(sm, ddof=1))}
    for lo, hi in BINS:
        mk = (nq >= lo) & (nq <= hi)
        allb = te_all[(te_all.n_qubits >= lo) & (te_all.n_qubits <= hi)]
        b = {"rows": int(mk.sum()), "total": int(len(allb)),
             "resolved_frac": float(allb.resolved.mean()),
             "resolved_frac_local": float(allb[allb.cost_global == 0].resolved.mean()),
             "resolved_frac_global": float(allb[allb.cost_global == 1].resolved.mean()),
             "label_mean": float(yte[mk].mean()), "label_std": float(yte[mk].std()),
             "label_min": float(yte[mk].min()), "models": {}}
        for name, p in pred.items():
            b["models"][name] = {"r2": float(r2_score(yte[mk], p[mk])),
                                 "mae": float(mean_absolute_error(yte[mk], p[mk])),
                                 "bias": float(np.mean(p[mk] - yte[mk]))}
        ss = [r2_score(yte[mk], p[mk]) for p in mlp_seed]; mm = [mean_absolute_error(yte[mk], p[mk]) for p in mlp_seed]
        b["models"]["MLP_seeds"] = {"r2_mean": float(np.mean(ss)), "r2_sd": float(np.std(ss, ddof=1)),
                                    "r2_min": float(np.min(ss)), "r2_max": float(np.max(ss)),
                                    "mae_mean": float(np.mean(mm)), "mae_sd": float(np.std(mm, ddof=1))}
        if len(ce):
            cm = (ce.n_qubits.to_numpy() >= lo) & (ce.n_qubits.to_numpy() <= hi)
            b["censored"] = int(cm.sum())
            for name, p in pred_ce.items():
                b["models"][name]["censored_consistent"] = float(np.mean(p[cm] <= ub[cm] + 0.3)) if cm.any() else float("nan")
        R["bins"]["%d-%d" % (lo, hi)] = b
    if len(ce):
        for name, p in pred_ce.items():
            R["overall"][name]["censored_consistent"] = float(np.mean(p <= ub + 0.3))
            R["overall"][name]["censored_pred_mean"] = float(np.mean(p))
        R["censored_ub_mean"] = float(np.mean(ub))

    # second regime: larger base, train n <= 20 (Clifford), test 21..32
    both = pd.concat([tr, te], ignore_index=True)
    m_tr = both.n_qubits <= 20; m_te = both.n_qubits > 20
    X2, y2 = both[FEATURE_COLUMNS].to_numpy(float), both.log_var_all.to_numpy(float)
    R["train_le20"] = {"train_rows": int(m_tr.sum()), "test_rows": int(m_te.sum())}
    for name, make in models.items():
        p = make().fit(X2[m_tr], y2[m_tr]).predict(X2[m_te])
        R["train_le20"][name] = {"r2": float(r2_score(y2[m_te], p)), "mae": float(mean_absolute_error(y2[m_te], p))}
    sv = [r2_score(y2[m_te], regressors(s)["MLP"].fit(X2[m_tr], y2[m_tr]).predict(X2[m_te])) for s in range(5)]
    R["train_le20"]["MLP_seeds"] = {"r2_mean": float(np.mean(sv)), "r2_sd": float(np.std(sv, ddof=1))}

    for name in models:
        o = R["overall"][name]
        print("%-44s R2 %.3f MAE %.3f bias %+.2f | censored ok %.2f | " % (
            name, o["r2"], o["mae"], o["bias"], o.get("censored_consistent", float("nan"))) +
            "  ".join("%s: %.2f/%.2f" % (k, b["models"][name]["r2"], b["models"][name]["mae"]) for k, b in R["bins"].items()))
    print("MLP seeds overall:", R["overall"]["MLP_seeds"])
    print("resolved fraction by bin:", {k: round(b["resolved_frac"], 2) for k, b in R["bins"].items()})
    print("train<=20 -> 21..32:", {k: v for k, v in R["train_le20"].items()})
    R["macros"] = {}
    json.dump(R, open(os.path.join(args.outdir, "clifford_study.json"), "w"), indent=1)

    # ---- tables ----
    lines = ["\\begin{tabular}{lrrcccccc}", "\\toprule",
             " & & & \\multicolumn{2}{c}{Structured linear} & \\multicolumn{2}{c}{MLP (10 seeds)} & \\multicolumn{2}{c}{HGB} \\\\",
             "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}",
             "Qubits & Resolved & Censored & $R^2$ & MAE & $R^2$ & MAE & $R^2$ & MAE \\\\", "\\midrule"]
    for k, b in R["bins"].items():
        m = b["models"]
        lines.append("%s & %d & %d & %.3f & %.3f & $%.3f \\pm %.3f$ & $%.3f \\pm %.3f$ & %.3f & %.3f \\\\" % (
            k.replace("-", "--"), b["rows"], b.get("censored", 0),
            m["Structured linear"]["r2"], m["Structured linear"]["mae"],
            m["MLP_seeds"]["r2_mean"], m["MLP_seeds"]["r2_sd"], m["MLP_seeds"]["mae_mean"], m["MLP_seeds"]["mae_sd"],
            m["Hist Gradient Boosting"]["r2"], m["Hist Gradient Boosting"]["mae"]))
    o = R["overall"]
    lines.append("\\midrule")
    lines.append("13--32 & %d & %d & %.3f & %.3f & $%.3f \\pm %.3f$ & $%.3f \\pm %.3f$ & %.3f & %.3f \\\\" % (
        R["test_resolved"], R["test_censored"], o["Structured linear"]["r2"], o["Structured linear"]["mae"],
        o["MLP_seeds"]["r2_mean"], o["MLP_seeds"]["r2_sd"], o["MLP_seeds"]["mae_mean"], o["MLP_seeds"]["mae_sd"],
        o["Hist Gradient Boosting"]["r2"], o["Hist Gradient Boosting"]["mae"]))
    lines += ["\\botrule", "\\end{tabular}"]
    open(os.path.join(args.outdir, "table_clifford.tex"), "w").write("\n".join(lines) + "\n")

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, ORANGE, GREEN, GREY = "#3b6ea5", "#e67e22", "#27ae60", "#7f8c8d"
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.7))
    ns = sorted(set(nq))
    def per_n(p, fn):
        return [fn(yte[nq == n], p[nq == n]) for n in ns]
    for name, c, mkr in [("Structured linear", GREEN, "D"), ("Hist Gradient Boosting", BLUE, "o"),
                         ("Physics-informed linear (effective weight)", GREY, "^")]:
        ax[0].plot(ns, per_n(pred[name], mean_absolute_error), marker=mkr, color=c, ms=3, lw=1.2,
                   label={"Hist Gradient Boosting": "HGB", "Physics-informed linear (effective weight)": "physics-linear"}.get(name, name))
    mm = np.array([per_n(p, mean_absolute_error) for p in mlp_seed])
    ax[0].plot(ns, mm.mean(0), marker="s", color=ORANGE, ms=3, lw=1.2, label="MLP (10 seeds)")
    ax[0].fill_between(ns, mm.min(0), mm.max(0), color=ORANGE, alpha=0.18)
    ax[0].set_xlabel("test qubit count $n$ (trained on $n \\leq 12$)", fontsize=8)
    ax[0].set_ylabel("MAE ($\\log_{10}$ units)", fontsize=8); ax[0].legend(fontsize=6, frameon=False)
    ax[0].tick_params(labelsize=7)
    for a, name, c in [(ax[1], "Structured linear", GREEN), (ax[2], "MLP", ORANGE)]:
        sc = a.scatter(yte, pred[name], s=4, c=nq, cmap="viridis", alpha=0.5, edgecolors="none")
        lo_, hi_ = yte.min() - 0.3, 0.2
        a.plot([lo_, hi_], [lo_, hi_], "k--", lw=0.7)
        a.set_title("%s: $R^2=%.2f$" % (name, R["overall"][name]["r2"]), fontsize=8)
        a.set_xlabel("true label, $13 \\leq n \\leq 32$", fontsize=8); a.tick_params(labelsize=7)
    ax[1].set_ylabel("predicted", fontsize=8)
    cb = fig.colorbar(sc, ax=ax[2], fraction=0.046, pad=0.04); cb.set_label("$n$", fontsize=7); cb.ax.tick_params(labelsize=6)
    plt.tight_layout(); plt.savefig(os.path.join(args.figdir, "clifford.pdf")); plt.close()
    print("wrote clifford_study.json, table_clifford.tex, figs/clifford.pdf")


if __name__ == "__main__":
    main()
