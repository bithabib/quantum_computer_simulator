"""Full analysis for the revised manuscript (architecture-grouped evaluation).

Reads the all-parameter dataset written by ``qml_bp.generate --label-mode all``
and produces every number, table and figure quoted in the paper:

  * dataset facts: unique architectures, multiplicities, row-split leakage,
    structural-zero statistics, legacy-vs-all-parameter label agreement
  * architecture-grouped K-fold CV (repeated) for six learners + three controls
    (training mean, cost-locality subgroup mean, physics-informed linear model)
  * the old row-wise random split, for comparison only
  * extrapolation (train n<=10, test n=11,12) with per-n / per-cost breakdown,
    bootstrap confidence intervals and model-seed spread
  * barren-cutoff sensitivity
  * feature-group ablation under grouped CV (replaces permutation importance)
  * figures: predicted-vs-true (out-of-fold), qubit trend, ablation,
    structural-zero fraction, legacy-vs-all label comparison

    python -m qml_bp.analyze --data data_bp/bp_dataset_v2.csv \
        --outdir paper/qmi/results --figdir paper/qmi/figs
"""

import argparse
import json
import os
import platform
import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import (
    ExtraTreesClassifier, ExtraTreesRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, mean_absolute_error, r2_score,
                             roc_auc_score)
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from qml_bp.ansatz import FEATURE_COLUMNS

# Apple Accelerate emits spurious matmul RuntimeWarnings; results are unaffected.
warnings.filterwarnings("ignore", category=RuntimeWarning)

PRIMITIVE = ["n_qubits", "n_layers", "ansatz_type", "entangle_pattern",
             "entangler_gate", "cost_global"]
DERIVED = ["n_params", "n_entanglers", "depth_ratio"]
GROUPS = {                       # feature groups for the ablation study
    "size (n, L, nL, L/n)": ["n_qubits", "n_layers", "n_params", "depth_ratio"],
    "qubit count n": ["n_qubits"],
    "layers L": ["n_layers"],
    "entanglement (pattern, gate, count)": ["entangle_pattern", "entangler_gate",
                                           "n_entanglers"],
    "entangle pattern": ["entangle_pattern"],
    "entangler gate": ["entangler_gate"],
    "cost locality": ["cost_global"],
    "rotation scheme": ["ansatz_type"],
}
SEED = 0
CUTOFF = -2.0
EXTRAP_CUT = 10


# ---------------------------------------------------------------- models ---
def regressors(seed=SEED):
    return {
        "Linear baseline": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "k-NN": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=10)),
        "MLP": make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=(128, 64), max_iter=800, random_state=seed)),
        "Random Forest": RandomForestRegressor(n_estimators=400, n_jobs=-1,
                                               random_state=seed),
        "Extra Trees": ExtraTreesRegressor(n_estimators=400, n_jobs=-1,
                                           random_state=seed),
        "Hist Gradient Boosting": HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.08, random_state=seed),
    }


def classifiers(seed=SEED):
    return {
        "Linear baseline": make_pipeline(StandardScaler(),
                                         LogisticRegression(max_iter=2000)),
        "k-NN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=10)),
        "MLP": make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(128, 64), max_iter=800, random_state=seed)),
        "Random Forest": RandomForestClassifier(n_estimators=400, n_jobs=-1,
                                                random_state=seed),
        "Extra Trees": ExtraTreesClassifier(n_estimators=400, n_jobs=-1,
                                            random_state=seed),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.08, random_state=seed),
    }


def hgb_reg(seed=SEED):
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.08,
                                         random_state=seed)


def hgb_cls(seed=SEED):
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08,
                                          random_state=seed)


# Controls -------------------------------------------------------------------
class MeanBaseline:
    """Predict the training-set mean label (regression) / barren rate (cls)."""
    def fit(self, X, y):
        self.m = float(np.mean(y)); return self
    def predict(self, X):
        return np.full(len(X), self.m)
    def predict_proba(self, X):
        p = np.full(len(X), self.m); return np.c_[1 - p, p]


class SubgroupMean:
    """Mean label within each cost-locality subgroup of the training set."""
    def __init__(self, col):
        self.col = col
    def fit(self, X, y):
        g = X[:, self.col]
        self.m = {v: float(np.mean(y[g == v])) for v in np.unique(g)}
        self.default = float(np.mean(y)); return self
    def predict(self, X):
        return np.array([self.m.get(v, self.default) for v in X[:, self.col]])
    def predict_proba(self, X):
        p = np.clip(self.predict(X), 0, 1); return np.c_[1 - p, p]


def physics_features(X):
    """(n, L, 1_global, n*1_global) -- the reviewer's physics-informed model."""
    n = X[:, FEATURE_COLUMNS.index("n_qubits")]
    L = X[:, FEATURE_COLUMNS.index("n_layers")]
    g = X[:, FEATURE_COLUMNS.index("cost_global")]
    return np.c_[n, L, g, n * g]


class PhysicsLinear:
    def __init__(self, task):
        self.task = task
        self.m = (LinearRegression() if task == "reg"
                  else LogisticRegression(max_iter=2000))
    def fit(self, X, y):
        self.m.fit(physics_features(X), y); return self
    def predict(self, X):
        return self.m.predict(physics_features(X))
    def predict_proba(self, X):
        return self.m.predict_proba(physics_features(X))


def controls_reg():
    return {
        "Training mean": MeanBaseline(),
        "Cost-locality subgroup mean": SubgroupMean(FEATURE_COLUMNS.index("cost_global")),
        "Physics-informed linear": PhysicsLinear("reg"),
    }


def controls_cls():
    return {
        "Training mean": MeanBaseline(),
        "Cost-locality subgroup mean": SubgroupMean(FEATURE_COLUMNS.index("cost_global")),
        "Physics-informed linear": PhysicsLinear("cls"),
    }


# ------------------------------------------------------------- utilities ---
def arch_id(df):
    return (df[PRIMITIVE].astype(int).astype(str).agg("|".join, axis=1))


def grouped_folds(groups, n_splits, n_repeats, seed=SEED):
    """Repeated GroupKFold: shuffle group labels between repeats."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    for r in range(n_repeats):
        perm = rng.permutation(len(uniq))
        remap = dict(zip(uniq, perm))
        g2 = np.array([remap[g] for g in groups])
        for tr, te in GroupKFold(n_splits=n_splits).split(g2, groups=g2):
            yield r, tr, te


def bootstrap_ci(fn, y, p, n_boot=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    vals = []
    idx = np.arange(len(y))
    for _ in range(n_boot):
        b = rng.choice(idx, size=len(idx), replace=True)
        try:
            vals.append(fn(y[b], p[b]))
        except ValueError:
            continue
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def ms(vals):
    return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0


def fmt_ms(vals, nd=3):
    m, s = ms(vals)
    return "$%.*f \\pm %.*f$" % (nd, m, nd, s)


def write(path, lines):
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote", path)


# ------------------------------------------------------------- analysis ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--outdir", default="paper/qmi/results")
    ap.add_argument("--figdir", default="paper/qmi/figs")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--cutoff", type=float, default=CUTOFF)
    ap.add_argument("--quick", action="store_true", help="skip slow models")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.figdir, exist_ok=True)
    R = {"software": {"python": platform.python_version(),
                      "numpy": np.__version__, "sklearn": sklearn.__version__,
                      "pandas": pd.__version__}}

    raw = pd.read_csv(args.data)
    R["n_rows_raw"] = int(len(raw))
    R["n_all_structural"] = int(raw.log_var.isna().sum())
    df = raw.dropna(subset=["log_var"]).reset_index(drop=True)
    R["n_rows"] = int(len(df))
    X = df[FEATURE_COLUMNS].to_numpy(float)
    y = df["log_var"].to_numpy(float)
    yc = (y < args.cutoff).astype(int)
    groups = arch_id(df).to_numpy()
    nq = df.n_qubits.to_numpy(int)

    # ---- 1. dataset facts ---------------------------------------------
    mult = pd.Series(groups).value_counts()
    R["n_unique_arch"] = int(len(mult))
    R["mult_median"] = float(mult.median()); R["mult_max"] = int(mult.max())
    R["design_space_size"] = 11 * 20 * 2 * 3 * 2 * 2
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(df)); cut = int(0.8 * len(df))
    tr_row, te_row = perm[:cut], perm[cut:]
    seen = set(groups[tr_row])
    R["row_split_leak_frac"] = float(np.mean([g in seen for g in groups[te_row]]))
    R["barren_frac"] = float(yc.mean())
    R["global_frac"] = float(df.cost_global.mean())
    R["label_mean"] = float(y.mean()); R["label_std"] = float(y.std())
    R["label_min"] = float(y.min()); R["label_max"] = float(y.max())

    # noise ceiling: rows sharing a feature vector differ only in their random
    # Pauli axes and in sampling noise, so within-architecture label variance
    # is irreducible for any model that sees only the feature vector.
    gv = pd.Series(y).groupby(groups)
    within = gv.transform(lambda s: s.var(ddof=0) if len(s) > 1 else np.nan)
    R["within_arch_var"] = float(np.nanmean(within))
    R["total_var"] = float(np.var(y))
    R["r2_ceiling"] = float(1 - np.nanmean(within) / np.var(y))
    # same ceiling restricted to fixed-ry circuits (no hidden axis pattern)
    m_ry = (df.ansatz_type == 0).to_numpy()
    R["r2_ceiling_fixed_ry"] = float(1 - np.nanmean(within[m_ry]) / np.var(y[m_ry]))
    R["r2_ceiling_random_pauli"] = float(1 - np.nanmean(within[~m_ry]) / np.var(y[~m_ry]))

    # structural zeros
    st = raw.frac_structural
    R["struct_frac_params_overall"] = float((raw.n_structural.sum()) / raw.n_params.sum())
    R["struct_rows_any"] = float((raw.n_structural > 0).mean())
    R["struct_by_layers"] = {int(k): float(v) for k, v in
                             raw.groupby("n_layers").frac_structural.mean().items()}
    R["struct_by_cost"] = {int(k): float(v) for k, v in
                           raw.groupby("cost_global").frac_structural.mean().items()}
    R["struct_by_gate"] = {int(k): float(v) for k, v in
                           raw.groupby("entangler_gate").frac_structural.mean().items()}
    R["struct_by_ansatz"] = {int(k): float(v) for k, v in
                             raw.groupby("ansatz_type").frac_structural.mean().items()}
    # legacy (single middle-layer, qubit-0 parameter) label vs new label
    leg = raw.log_grad_var.to_numpy(); legacy_floor = leg <= -17.9
    R["legacy_floor_frac"] = float(legacy_floor.mean())
    R["legacy_floor_min_nonfloor"] = float(leg[~legacy_floor].min())
    ok = ~legacy_floor & raw.log_var.notna().to_numpy()
    R["legacy_vs_new_corr"] = float(np.corrcoef(leg[ok], raw.log_var.to_numpy()[ok])[0, 1])
    R["legacy_vs_new_mad"] = float(np.mean(np.abs(leg[ok] - raw.log_var.to_numpy()[ok])))
    R["legacy_vs_new_barren_disagree"] = float(np.mean(
        (leg[ok] < args.cutoff) != (raw.log_var.to_numpy()[ok] < args.cutoff)))
    R["legacy_floor_barren_but_new_trainable"] = float(np.mean(
        raw.log_var.to_numpy()[legacy_floor & raw.log_var.notna().to_numpy()] >= args.cutoff))
    # spread of per-parameter variance within a circuit
    spread = np.log10(raw.var_max / raw.var_min_nz)
    R["within_circuit_spread_median"] = float(np.nanmedian(spread))
    R["within_circuit_spread_p90"] = float(np.nanpercentile(spread, 90))

    # ---- 2. grouped CV: learners + controls ---------------------------
    folds = list(grouped_folds(groups, args.folds, args.repeats))
    R["n_folds"] = len(folds); R["n_folds_k"] = args.folds; R["n_repeats"] = args.repeats
    R["cost_gap"] = float(df[df.cost_global == 1].log_var.mean() - df[df.cost_global == 0].log_var.mean())
    reg_models = regressors(); cls_models = classifiers()
    if args.quick:
        for k in ["MLP", "Random Forest", "Extra Trees"]:
            reg_models.pop(k); cls_models.pop(k)
    res_reg = {k: {"r2": [], "mae": []} for k in list(reg_models) + list(controls_reg())}
    res_cls = {k: {"auc": [], "acc": []} for k in list(cls_models) + list(controls_cls())}
    oof_pred = np.full(len(df), np.nan)   # out-of-fold HGB predictions (repeat 0)
    for r, tr, te in folds:
        for name, m in list(reg_models.items()) + list(controls_reg().items()):
            m = regressors()[name] if name in reg_models else m
            m.fit(X[tr], y[tr]); p = m.predict(X[te])
            res_reg[name]["r2"].append(r2_score(y[te], p))
            res_reg[name]["mae"].append(mean_absolute_error(y[te], p))
            if name == "Hist Gradient Boosting" and r == 0:
                oof_pred[te] = p
        for name, m in list(cls_models.items()) + list(controls_cls().items()):
            m = classifiers()[name] if name in cls_models else m
            m.fit(X[tr], yc[tr]); pr = m.predict_proba(X[te])[:, 1]
            res_cls[name]["auc"].append(roc_auc_score(yc[te], pr))
            res_cls[name]["acc"].append(accuracy_score(yc[te], (pr >= 0.5).astype(int)))
    R["grouped_cv_reg"] = {k: {m: ms(v) for m, v in d.items()} for k, d in res_reg.items()}
    R["grouped_cv_cls"] = {k: {m: ms(v) for m, v in d.items()} for k, d in res_cls.items()}
    R["oof_r2"] = float(r2_score(y, oof_pred))

    # row-wise random split for comparison (HGB only)
    m = hgb_reg().fit(X[tr_row], y[tr_row])
    R["row_split_r2"] = float(r2_score(y[te_row], m.predict(X[te_row])))
    R["row_split_mae"] = float(mean_absolute_error(y[te_row], m.predict(X[te_row])))
    c = hgb_cls().fit(X[tr_row], yc[tr_row])
    R["row_split_auc"] = float(roc_auc_score(yc[te_row], c.predict_proba(X[te_row])[:, 1]))

    # ---- 3. extrapolation ---------------------------------------------
    tr2 = np.where(nq <= EXTRAP_CUT)[0]; te2 = np.where(nq > EXTRAP_CUT)[0]
    R["extrap_train_rows"] = int(len(tr2)); R["extrap_test_rows"] = int(len(te2))
    ext = {}
    for name, m in list(regressors().items()) + list(controls_reg().items()):
        if args.quick and name in ("MLP", "Random Forest", "Extra Trees"):
            continue
        m.fit(X[tr2], y[tr2]); p = m.predict(X[te2])
        ext[name] = {"r2": float(r2_score(y[te2], p)),
                     "mae": float(mean_absolute_error(y[te2], p))}
        if name == "Hist Gradient Boosting":
            p_hgb = p
    R["extrap_reg"] = ext
    ext_c = {}
    for name, m in list(classifiers().items()) + list(controls_cls().items()):
        if args.quick and name in ("MLP", "Random Forest", "Extra Trees"):
            continue
        m.fit(X[tr2], yc[tr2]); pr = m.predict_proba(X[te2])[:, 1]
        ext_c[name] = {"auc": float(roc_auc_score(yc[te2], pr)),
                       "acc": float(accuracy_score(yc[te2], (pr >= .5).astype(int)))}
    R["extrap_cls"] = ext_c
    # bootstrap CIs and seed spread for HGB
    R["extrap_hgb_r2_ci"] = bootstrap_ci(r2_score, y[te2], p_hgb)
    R["extrap_hgb_mae_ci"] = bootstrap_ci(mean_absolute_error, y[te2], p_hgb)
    seeds = [hgb_reg(s).fit(X[tr2], y[tr2]).predict(X[te2]) for s in range(5)]
    R["extrap_hgb_r2_seeds"] = ms([r2_score(y[te2], p) for p in seeds])
    # breakdown by n and cost, HGB vs physics-informed vs subgroup mean
    phys = PhysicsLinear("reg").fit(X[tr2], y[tr2]).predict(X[te2])
    sub = SubgroupMean(FEATURE_COLUMNS.index("cost_global")).fit(X[tr2], y[tr2]).predict(X[te2])
    mlp = regressors()["MLP"].fit(X[tr2], y[tr2]).predict(X[te2]) if not args.quick else phys
    R["extrap_mlp_r2_ci"] = bootstrap_ci(r2_score, y[te2], mlp)
    brk = {}
    for label, mask in [("n=11", nq[te2] == 11), ("n=12", nq[te2] == 12),
                        ("local", X[te2, 6] == 0), ("global", X[te2, 6] == 1),
                        ("n=11 local", (nq[te2] == 11) & (X[te2, 6] == 0)),
                        ("n=11 global", (nq[te2] == 11) & (X[te2, 6] == 1)),
                        ("n=12 local", (nq[te2] == 12) & (X[te2, 6] == 0)),
                        ("n=12 global", (nq[te2] == 12) & (X[te2, 6] == 1)),
                        ("all", np.ones(len(te2), bool))]:
        yt = y[te2][mask]
        brk[label] = {
            "rows": int(mask.sum()), "label_mean": float(yt.mean()),
            "label_std": float(yt.std()),
            "hgb_r2": float(r2_score(yt, p_hgb[mask])),
            "hgb_mae": float(mean_absolute_error(yt, p_hgb[mask])),
            "hgb_mae_ci": bootstrap_ci(mean_absolute_error, yt, p_hgb[mask], 500),
            "phys_r2": float(r2_score(yt, phys[mask])),
            "phys_mae": float(mean_absolute_error(yt, phys[mask])),
            "mlp_r2": float(r2_score(yt, mlp[mask])),
            "mlp_mae": float(mean_absolute_error(yt, mlp[mask])),
            "mlp_mae_ci": bootstrap_ci(mean_absolute_error, yt, mlp[mask], 500),
            "sub_r2": float(r2_score(yt, sub[mask])),
            "sub_mae": float(mean_absolute_error(yt, sub[mask])),
        }
    R["extrap_breakdown"] = brk
    # hold-out (grouped) bootstrap CI for HGB, using OOF predictions
    R["oof_r2_ci"] = bootstrap_ci(r2_score, y, oof_pred)
    R["oof_mae"] = float(mean_absolute_error(y, oof_pred))

    # ---- 4. cutoff sensitivity ---------------------------------------
    sens = {}
    for cut in [-1.5, -2.0, -2.5, -3.0]:
        ycc = (y < cut).astype(int); aucs = []
        for r, tr, te in folds[:args.folds]:
            if len(np.unique(ycc[tr])) < 2 or len(np.unique(ycc[te])) < 2:
                continue
            pr = hgb_cls().fit(X[tr], ycc[tr]).predict_proba(X[te])[:, 1]
            aucs.append(roc_auc_score(ycc[te], pr))
        sens[str(cut)] = {"barren_frac": float(ycc.mean()), "auc": ms(aucs)}
    R["cutoff_sensitivity"] = sens

    # ---- 5. ablation (grouped CV, HGB) --------------------------------
    def cv_r2(cols):
        idx = [FEATURE_COLUMNS.index(c) for c in cols]; out = []
        for r, tr, te in folds:
            m = hgb_reg().fit(X[tr][:, idx], y[tr])
            out.append(r2_score(y[te], m.predict(X[te][:, idx])))
        return out
    full = cv_r2(FEATURE_COLUMNS); prim = cv_r2(PRIMITIVE)
    abl = {"full (9 features)": ms(full), "primitive only (6 features)": ms(prim)}
    drops = {}
    for gname, cols in GROUPS.items():
        keep = [c for c in FEATURE_COLUMNS if c not in cols]
        v = cv_r2(keep)
        drops[gname] = {"r2": ms(v), "delta": ms(np.array(full) - np.array(v))}
    R["ablation"] = {"reference": abl, "drop": drops}

    # ---- write JSON ----------------------------------------------------
    with open(os.path.join(args.outdir, "results.json"), "w") as fh:
        json.dump(R, fh, indent=1)
    print(json.dumps({k: v for k, v in R.items() if not isinstance(v, dict)}, indent=1))

    # ---- LaTeX tables --------------------------------------------------
    order = ["Hist Gradient Boosting", "MLP", "Random Forest", "Extra Trees",
             "k-NN", "Linear baseline"]
    order = [o for o in order if o in res_reg]
    ctrl = ["Physics-informed linear", "Cost-locality subgroup mean", "Training mean"]
    lines = ["\\begin{tabular}{lcccc}", "\\toprule",
             " & \\multicolumn{2}{c}{Regression} & \\multicolumn{2}{c}{Classification} \\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
             "Model & $R^2$ & MAE & AUC & Accuracy \\\\", "\\midrule"]
    for name in order + ["MID"] + ctrl:
        if name == "MID":
            lines.append("\\midrule"); continue
        lab = "\\textbf{%s}" % name if name == "Hist Gradient Boosting" else name
        lines.append("%s & %s & %s & %s & %s \\\\" % (
            lab, fmt_ms(res_reg[name]["r2"]), fmt_ms(res_reg[name]["mae"]),
            fmt_ms(res_cls[name]["auc"]), fmt_ms(res_cls[name]["acc"])))
    lines += ["\\botrule", "\\end{tabular}"]
    write(os.path.join(args.outdir, "table_grouped_cv.tex"), lines)

    lines = ["\\begin{tabular}{lcccc}", "\\toprule",
             "Model & $R^2$ & MAE & AUC & Accuracy \\\\", "\\midrule"]
    for name in order + ["MID"] + ctrl:
        if name == "MID":
            lines.append("\\midrule"); continue
        lab = "\\textbf{%s}" % name if name == "Hist Gradient Boosting" else name
        lines.append("%s & %.3f & %.3f & %.3f & %.3f \\\\" % (
            lab, ext[name]["r2"], ext[name]["mae"], ext_c[name]["auc"], ext_c[name]["acc"]))
    lines += ["\\botrule", "\\end{tabular}"]
    write(os.path.join(args.outdir, "table_extrap.tex"), lines)

    lines = ["\\begin{tabular}{lrrcccc}", "\\toprule",
             "Subset & Rows & Label mean (std) & HGB MAE [95\\% CI] & MLP MAE [95\\% CI] & Physics-linear MAE & Subgroup-mean MAE \\\\",
             "\\midrule"]
    for label in ["n=11 local", "n=11 global", "n=12 local", "n=12 global", "all"]:
        b = brk[label]
        lines.append("%s & %d & $%.2f$ ($%.2f$) & %.3f [%.3f, %.3f] & %.3f [%.3f, %.3f] & %.3f & %.3f \\\\" % (
            label.replace("n=", "$n=$"), b["rows"], b["label_mean"], b["label_std"],
            b["hgb_mae"], b["hgb_mae_ci"][0], b["hgb_mae_ci"][1],
            b["mlp_mae"], b["mlp_mae_ci"][0], b["mlp_mae_ci"][1], b["phys_mae"], b["sub_mae"]))
    lines += ["\\botrule", "\\end{tabular}"]
    write(os.path.join(args.outdir, "table_extrap_breakdown.tex"), lines)

    lines = ["\\begin{tabular}{lcc}", "\\toprule",
             "Features removed & $R^2$ (grouped CV) & $\\Delta R^2$ \\\\", "\\midrule",
             "none (all 9 features) & %s & -- \\\\" % fmt_ms(full),
             "derived features (keep 6 primitives) & %s & %s \\\\" % (
                 fmt_ms(prim), fmt_ms(np.array(full) - np.array(prim)))]
    for gname, d in drops.items():
        lines.append("%s & $%.3f \\pm %.3f$ & $%.3f \\pm %.3f$ \\\\" % (
            gname, d["r2"][0], d["r2"][1], d["delta"][0], d["delta"][1]))
    lines += ["\\botrule", "\\end{tabular}"]
    write(os.path.join(args.outdir, "table_ablation.tex"), lines)

    lines = ["\\begin{tabular}{lcc}", "\\toprule",
             "Cutoff $y<c$ & Barren fraction & AUC (grouped CV) \\\\", "\\midrule"]
    for cut, d in sens.items():
        lines.append("$c=%s$ & %.1f\\%% & $%.3f \\pm %.3f$ \\\\" % (
            cut, 100 * d["barren_frac"], d["auc"][0], d["auc"][1]))
    lines += ["\\botrule", "\\end{tabular}"]
    write(os.path.join(args.outdir, "table_cutoff.tex"), lines)

    # label-by-design and descriptive stats (new label)
    dfb = df.assign(barren=yc)
    lines = ["\\begin{tabular}{lrrrr}", "\\toprule",
             "\\textbf{Design choice} & \\textbf{Count} & \\textbf{Mean} $\\boldsymbol{\\log_{10}\\overline{\\mathrm{Var}}}$ & \\textbf{Barren \\%} & \\textbf{Structural-zero \\%}\\\\",
             "\\midrule"]
    for gi, (title, col, labels) in enumerate([
            ("Cost-observable locality", "cost_global", ["local $Z_0$", "global $Z^{\\otimes n}$"]),
            ("Entanglement pattern", "entangle_pattern", ["linear", "circular", "all-to-all"]),
            ("Entangler gate", "entangler_gate", ["CZ", "CX"]),
            ("Single-qubit rotation scheme", "ansatz_type", ["fixed-\\texttt{ry}", "random-Pauli"])]):
        if gi:
            lines.append("\\midrule")
        lines.append("\\multicolumn{5}{l}{\\emph{%s}}\\\\" % title)
        for i, lab in enumerate(labels):
            s = dfb[dfb[col] == i]
            lines.append("\\quad %s & %d & $%.2f$ & %.1f & %.1f\\\\" % (
                lab, len(s), s.log_var.mean(), 100 * s.barren.mean(),
                100 * s.frac_structural.mean()))
    lines += ["\\botrule", "\\end{tabular}"]
    write(os.path.join(args.outdir, "table_labelbydesign.tex"), lines)

    lines = ["\\begin{tabular}{lrrrrrr}", "\\toprule",
             "\\textbf{Variable} & \\textbf{Mean} & \\textbf{Std} & \\textbf{Min} & \\textbf{25\\%} & \\textbf{Median} & \\textbf{Max}\\\\",
             "\\midrule"]
    nice = {"n_qubits": "\\texttt{n\\_qubits}", "n_layers": "\\texttt{n\\_layers}",
            "n_params": "\\texttt{n\\_params}", "n_entanglers": "\\texttt{n\\_entanglers}",
            "depth_ratio": "\\texttt{depth\\_ratio}",
            "frac_structural": "structural-zero fraction",
            "log_var": "$\\log_{10}\\overline{\\mathrm{Var}}$ (label)"}
    for c in ["n_qubits", "n_layers", "n_params", "n_entanglers", "depth_ratio",
              "frac_structural", "log_var"]:
        s = df[c]
        lines.append("%s & %s\\\\" % (nice[c], " & ".join(
            ("$%.2f$" % v) for v in [s.mean(), s.std(), s.min(), s.quantile(.25),
                                     s.median(), s.max()])))
    lines += ["\\botrule", "\\end{tabular}"]
    write(os.path.join(args.outdir, "table_datastats.tex"), lines)

    # ---- figures -------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    BLUE, RED, GREY = "#3b6ea5", "#c0392b", "#7f8c8d"

    plt.figure(figsize=(3.4, 2.9))
    plt.scatter(y, oof_pred, s=5, alpha=0.3, color=BLUE, edgecolors="none")
    lo, hi = min(y.min(), np.nanmin(oof_pred)), max(y.max(), np.nanmax(oof_pred))
    plt.plot([lo, hi], [lo, hi], "k--", lw=1)
    plt.xlabel("true $\\log_{10}\\overline{\\mathrm{Var}}$", fontsize=8)
    plt.ylabel("predicted (out-of-fold)", fontsize=8)
    plt.title("architecture-grouped CV  $R^2=%.2f$" % R["oof_r2"], fontsize=9)
    plt.xticks(fontsize=8); plt.yticks(fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(args.figdir, "predicted.pdf")); plt.close()

    plt.figure(figsize=(3.6, 2.8))
    for cost, label, color, marker in [(0, "local $Z_0$", BLUE, "o"),
                                       (1, "global $Z^{\\otimes n}$", RED, "s")]:
        sub = df[df.cost_global == cost]; g = sub.groupby("n_qubits")["log_var"]
        xs = g.mean().index.to_numpy(); mu = g.mean().to_numpy(); sd = g.std().to_numpy()
        plt.plot(xs, mu, marker=marker, color=color, label=label, ms=4, lw=1.5)
        plt.fill_between(xs, mu - sd, mu + sd, color=color, alpha=0.12)
    plt.xlabel("number of qubits $n$", fontsize=8)
    plt.ylabel("mean $\\log_{10}\\overline{\\mathrm{Var}}$", fontsize=8)
    plt.legend(fontsize=8, frameon=False); plt.xticks(fontsize=8); plt.yticks(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(args.figdir, "qubit_trend.pdf")); plt.close()

    names = list(drops); deltas = [drops[g]["delta"][0] for g in names]
    errs = [drops[g]["delta"][1] for g in names]
    order_i = np.argsort(deltas)
    plt.figure(figsize=(3.8, 2.8))
    plt.barh(range(len(names)), np.array(deltas)[order_i], xerr=np.array(errs)[order_i],
             color=BLUE, height=0.7)
    plt.yticks(range(len(names)), [names[i] for i in order_i], fontsize=7)
    plt.xlabel("$\\Delta R^2$ when feature group is removed", fontsize=8)
    plt.xticks(fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(args.figdir, "ablation.pdf")); plt.close()

    plt.figure(figsize=(3.6, 2.8))
    for cost, label, color, marker in [(0, "local $Z_0$", BLUE, "o"),
                                       (1, "global $Z^{\\otimes n}$", RED, "s")]:
        sub = raw[raw.cost_global == cost]; g = sub.groupby("n_layers")["frac_structural"]
        plt.plot(g.mean().index, 100 * g.mean().to_numpy(), marker=marker, color=color,
                 label=label, ms=3.5, lw=1.4)
    plt.xlabel("number of layers $L$", fontsize=8)
    plt.ylabel("structural-zero parameters (%)", fontsize=8)
    plt.legend(fontsize=8, frameon=False); plt.xticks(fontsize=8); plt.yticks(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(args.figdir, "structural.pdf")); plt.close()

    plt.figure(figsize=(3.4, 2.9))
    plt.scatter(raw.log_var[ok], leg[ok], s=4, alpha=0.25, color=BLUE, edgecolors="none",
                label="non-zero legacy gradient")
    fl = legacy_floor & raw.log_var.notna().to_numpy()
    plt.scatter(raw.log_var[fl], np.full(fl.sum(), -6.5), s=4, alpha=0.4, color=RED,
                edgecolors="none", marker="v", label="legacy structural zero")
    plt.plot([-5, 0], [-5, 0], "k--", lw=1)
    plt.xlabel("all-parameter label $\\log_{10}\\overline{\\mathrm{Var}}$", fontsize=8)
    plt.ylabel("single-parameter label", fontsize=8)
    plt.ylim(-7.2, 0.3); plt.legend(fontsize=6.5, frameon=False, loc="lower right")
    plt.xticks(fontsize=8); plt.yticks(fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(args.figdir, "label_compare.pdf")); plt.close()
    print("figures written to", args.figdir)


if __name__ == "__main__":
    main()
