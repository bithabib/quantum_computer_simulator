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

from qml_bp.ansatz import FEATURE_COLUMNS, causal_cone, effective_weight

# Apple Accelerate emits spurious matmul RuntimeWarnings; results are unaffected.
warnings.filterwarnings("ignore", category=RuntimeWarning)

PRIMITIVE = ["n_qubits", "n_layers", "ansatz_type", "entangle_pattern",
             "entangler_gate", "cost_global"]
DERIVED = ["n_params", "n_entanglers", "depth_ratio", "cost_weight_eff", "cone_qubits", "cone_frac"]
GROUPS = {                       # feature groups for the ablation study
    "size (n, L, nL, L/n)": ["n_qubits", "n_layers", "n_params", "depth_ratio"],
    "qubit count n": ["n_qubits"],
    "layers L": ["n_layers"],
    "entanglement (pattern, gate, count)": ["entangle_pattern", "entangler_gate",
                                           "n_entanglers"],
    "entangle pattern": ["entangle_pattern"],
    "entangler gate": ["entangler_gate"],
    "cost locality": ["cost_global"],
    "effective observable weight": ["cost_weight_eff"],
    "cost observable (locality + effective weight)": ["cost_global", "cost_weight_eff"],
    "entangler gate + effective weight": ["entangler_gate", "cost_weight_eff"],
    "causal cone (qubits, fraction)": ["cone_qubits", "cone_frac"],
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


def load_dataset(path):
    """Read a dataset CSV and add closed-form feature columns it may predate."""
    df = pd.read_csv(path)
    if "cost_weight_eff" not in df.columns:
        cache = {}
        vals = []
        for key in zip(df.n_qubits, df.n_layers, df.entangle_pattern, df.entangler_gate, df.cost_global):
            key = tuple(int(k) for k in key)
            if key not in cache:
                cache[key] = effective_weight(*key)
            vals.append(cache[key])
        df["cost_weight_eff"] = vals
    if "cone_qubits" not in df.columns:
        cache = {}; cq = []; cf = []
        for key in zip(df.n_qubits, df.n_layers, df.ansatz_type, df.entangle_pattern, df.entangler_gate, df.cost_global):
            key = tuple(int(k) for k in key)
            if key not in cache:
                cache[key] = causal_cone(*key)
            cq.append(cache[key][0]); cf.append(cache[key][1])
        df["cone_qubits"] = cq; df["cone_frac"] = cf
    return df


def physics_features(X, effective=False):
    """(n, L, g, n*g).  g is the nominal global-cost indicator, or, with
    ``effective=True``, the effective observable weight divided by n."""
    n = X[:, FEATURE_COLUMNS.index("n_qubits")]
    L = X[:, FEATURE_COLUMNS.index("n_layers")]
    if effective:
        g = X[:, FEATURE_COLUMNS.index("cost_weight_eff")] / n
    else:
        g = X[:, FEATURE_COLUMNS.index("cost_global")]
    return np.c_[n, L, g, n * g]


class PhysicsLinear:
    def __init__(self, task, effective=False):
        self.task = task; self.effective = effective
        self.m = (LinearRegression() if task == "reg"
                  else LogisticRegression(max_iter=2000))
    def fit(self, X, y):
        self.m.fit(physics_features(X, self.effective), y); return self
    def predict(self, X):
        return self.m.predict(physics_features(X, self.effective))
    def predict_proba(self, X):
        return self.m.predict_proba(physics_features(X, self.effective))


class StructuredLinear:
    """Interpretable control: one linear model per categorical cell.

    A cell is a combination of (rotation scheme, entanglement pattern,
    entangler gate, cost observable).  Within each cell the label is modeled
    as an intercept plus slopes on four closed-form size terms,
        c,  min(L, c),  exp(-L/c),  1/L,
    where c is the number of qubits in the causal cone of the observable
    (c = n whenever the cone covers the circuit), i.e. a linear decay in the
    relevant qubit count and a saturating dependence on depth.  With
    ``dilution=True`` (labels that average over ALL parameters, including
    structural zeros) the model predicts the per-in-cone-parameter value and
    adds log10 of the in-cone fraction.  Fitted by ridge regression (logistic
    regression for classification)."""
    CELL = ["ansatz_type", "entangle_pattern", "entangler_gate", "cost_global"]

    def __init__(self, task, alpha=1e-3, dilution=False):
        self.task = task; self.dilution = dilution
        self.m = (Ridge(alpha=alpha) if task == "reg"
                  else make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=10.0)))

    def _offset(self, X):
        if not self.dilution:
            return 0.0
        return np.log10(np.clip(X[:, FEATURE_COLUMNS.index("cone_frac")], 1e-6, None))

    def _design(self, X):
        n = X[:, FEATURE_COLUMNS.index("cone_qubits")]
        L = X[:, FEATURE_COLUMNS.index("n_layers")]
        base = np.c_[np.ones(len(X)), n, np.minimum(L, n), np.exp(-L / n), 1.0 / L]
        keys = [tuple(int(v) for v in row) for row in X[:, [FEATURE_COLUMNS.index(c) for c in self.CELL]]]
        oh = np.zeros((len(X), len(self.cells)))
        for i, k in enumerate(keys):
            j = self.cells.get(k)
            if j is not None:
                oh[i, j] = 1.0
        return np.hstack([oh * base[:, [j]] for j in range(base.shape[1])] + [base[:, 1:]])

    def fit(self, X, y):
        keys = sorted({tuple(int(v) for v in row)
                       for row in X[:, [FEATURE_COLUMNS.index(c) for c in self.CELL]]})
        self.cells = {k: i for i, k in enumerate(keys)}
        self.n_coef = len(keys) * 5 + 4
        self.m.fit(self._design(X), y - self._offset(X)); return self
    def predict(self, X):
        return self.m.predict(self._design(X)) + self._offset(X)
    def predict_proba(self, X):
        return self.m.predict_proba(self._design(X))


def controls_reg():
    return {
        "Training mean": MeanBaseline(),
        "Cost-locality subgroup mean": SubgroupMean(FEATURE_COLUMNS.index("cost_global")),
        "Physics-informed linear": PhysicsLinear("reg"),
        "Physics-informed linear (effective weight)": PhysicsLinear("reg", effective=True),
        "Structured linear": StructuredLinear("reg"),
    }


def controls_cls():
    return {
        "Training mean": MeanBaseline(),
        "Cost-locality subgroup mean": SubgroupMean(FEATURE_COLUMNS.index("cost_global")),
        "Physics-informed linear": PhysicsLinear("cls"),
        "Physics-informed linear (effective weight)": PhysicsLinear("cls", effective=True),
        "Structured linear": StructuredLinear("cls"),
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


def dataset_facts(df, raw, cutoff):
    """Small facts quoted verbatim in the text."""
    mean_by_n = df.groupby("n_qubits").log_var.mean()
    above = [int(n) for n in mean_by_n.index if mean_by_n[n] >= cutoff]
    below_small = {int(n): int((g.log_var < cutoff).sum()) for n, g in df[df.n_qubits <= 5].groupby("n_qubits")}
    all_above = [n for n, c in below_small.items() if c == 0]
    return {"cutoff_cross_lo": max(above), "cutoff_cross_hi": max(above) + 1,
            "all_above_max_n": max(all_above),
            "below_cutoff_at_n5": below_small.get(5, 0), "rows_at_n5": int((df.n_qubits == 5).sum()),
            "min_nonstructural_param_var": float(raw.var_min_nz.min())}


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

    raw = load_dataset(args.data)
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
    # unbiased within-group variance (groups have a median of only 4 rows)
    within = gv.transform(lambda s: s.var(ddof=1) if len(s) > 1 else np.nan)
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
    stl = StructuredLinear("reg").fit(X[tr2], y[tr2]).predict(X[te2])
    R["extrap_struct_r2_ci"] = bootstrap_ci(r2_score, y[te2], stl)
    R["struct_n_coef"] = int(StructuredLinear("reg").fit(X, y).n_coef)
    # the MLP is sensitive to its random initialisation out of range: 10 seeds
    n_seeds = 2 if args.quick else 10
    sv = [r2_score(y[te2], regressors(s)["MLP"].fit(X[tr2], y[tr2]).predict(X[te2])) for s in range(n_seeds)]
    R["extrap_mlp_r2_seeds"] = {"mean": float(np.mean(sv)), "sd": float(np.std(sv, ddof=1)),
                                "min": float(np.min(sv)), "max": float(np.max(sv)), "n": n_seeds}
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
            "struct_r2": float(r2_score(yt, stl[mask])),
            "struct_mae": float(mean_absolute_error(yt, stl[mask])),
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
    R["cone_vs_structural"] = {
        "corr": float(np.corrcoef(1 - df.frac_structural, df.cone_frac)[0, 1]),
        "mad": float(np.mean(np.abs((1 - df.frac_structural) - df.cone_frac))),
        "mad_fixed_ry": float(np.mean(np.abs((1 - df.frac_structural) - df.cone_frac)[df.ansatz_type == 0])),
        "mad_random_pauli": float(np.mean(np.abs((1 - df.frac_structural) - df.cone_frac)[df.ansatz_type == 1])),
        "exact_frac_fixed_ry": float(np.mean(np.abs((1 - df.frac_structural) - df.cone_frac)[df.ansatz_type == 0] < 1e-9)),
    }
    R["n_features"] = len(FEATURE_COLUMNS)
    drops = {}
    for gname, cols in GROUPS.items():
        keep = [c for c in FEATURE_COLUMNS if c not in cols]
        v = cv_r2(keep)
        drops[gname] = {"r2": ms(v), "delta": ms(np.array(full) - np.array(v))}
    R["ablation"] = {"reference": abl, "drop": drops}
    # primitive-feature ablation: what the derived features add to the six
    # design parameters, and what removing each design parameter costs
    NICE_P = {"n_qubits": "qubit count $n$", "n_layers": "layers $L$", "ansatz_type": "rotation scheme",
              "entangle_pattern": "entanglement pattern", "entangler_gate": "entangler gate",
              "cost_global": "cost locality"}
    ap_add = {"+ effective weight": cv_r2(PRIMITIVE + ["cost_weight_eff"]),
              "+ causal cone": cv_r2(PRIMITIVE + ["cone_qubits", "cone_frac"]),
              "+ effective weight + causal cone": cv_r2(PRIMITIVE + ["cost_weight_eff", "cone_qubits", "cone_frac"])}
    ap_drop = {c: cv_r2([q for q in PRIMITIVE if q != c]) for c in PRIMITIVE}
    R["ablation_primitive"] = {
        "full": ms(full), "primitive": ms(prim),
        "add": {k: {"r2": ms(v), "delta": ms(np.array(v) - np.array(prim))} for k, v in ap_add.items()},
        "drop": {NICE_P[c]: {"r2": ms(v), "delta": ms(np.array(prim) - np.array(v))} for c, v in ap_drop.items()}}

    R.update(dataset_facts(df, raw, args.cutoff))

    # ---- 6. nominal vs effective observable weight (n >= 10) -----------
    R["gate_gap"] = float(abs(df[df.entangler_gate == 1].log_var.mean() - df[df.entangler_gate == 0].log_var.mean()))
    big_n = df[df.n_qubits >= 10]
    eff_rows = []
    for gate, gname in [(0, "CZ"), (1, "CX")]:
        for pat, pname in [(0, "linear"), (1, "circular"), (2, "all-to-all")]:
            for cg, cname in [(0, "local"), (1, "global")]:
                d = big_n[(big_n.entangler_gate == gate) & (big_n.entangle_pattern == pat) & (big_n.cost_global == cg)]
                if len(d) == 0:
                    continue
                n_ = d.n_qubits.to_numpy(); w_ = d.cost_weight_eff.to_numpy()
                if (w_ == 1).all(): wdesc = "$1$"
                elif (w_ == n_).all(): wdesc = "$n$"
                elif (w_ == n_ - 1).all(): wdesc = "$n-1$"
                elif (w_ == n_ // 2).all() or (np.abs(w_ - n_ / 2) <= 0.5).all(): wdesc = "$\\approx n/2$"
                else: wdesc = "%d--%d" % (w_.min(), w_.max())
                eff_rows.append({"gate": gname, "pattern": pname, "cost": cname, "w": wdesc,
                                 "w_min": int(w_.min()), "w_max": int(w_.max()),
                                 "mean_label": float(d.log_var.mean()), "rows": int(len(d))})
    R["effweight_table"] = eff_rows

    # ---- write JSON ----------------------------------------------------
    with open(os.path.join(args.outdir, "results.json"), "w") as fh:
        json.dump(R, fh, indent=1)
    print(json.dumps({k: v for k, v in R.items() if not isinstance(v, dict)}, indent=1))

    # ---- LaTeX tables --------------------------------------------------
    order = ["Hist Gradient Boosting", "MLP", "Random Forest", "Extra Trees",
             "k-NN", "Linear baseline"]
    order = [o for o in order if o in res_reg]
    ctrl = ["Structured linear", "Physics-informed linear (effective weight)",
            "Physics-informed linear", "Cost-locality subgroup mean", "Training mean"]
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
             "Subset & Rows & Label mean (std) & HGB MAE [95\\% CI] & MLP MAE [95\\% CI] & Structured linear MAE & Physics-linear MAE \\\\",
             "\\midrule"]
    for label in ["n=11 local", "n=11 global", "n=12 local", "n=12 global", "all"]:
        b = brk[label]
        lines.append("%s & %d & $%.2f$ ($%.2f$) & %.3f [%.3f, %.3f] & %.3f [%.3f, %.3f] & %.3f & %.3f \\\\" % (
            label.replace("n=", "$n=$"), b["rows"], b["label_mean"], b["label_std"],
            b["hgb_mae"], b["hgb_mae_ci"][0], b["hgb_mae_ci"][1],
            b["mlp_mae"], b["mlp_mae_ci"][0], b["mlp_mae_ci"][1], b["struct_mae"], b["phys_mae"]))
    lines += ["\\botrule", "\\end{tabular}"]
    write(os.path.join(args.outdir, "table_extrap_breakdown.tex"), lines)

    lines = ["\\begin{tabular}{lcc}", "\\toprule", "Feature set & $R^2$ (grouped CV) & $\\Delta R^2$ \\\\", "\\midrule",
             "six primitive features & %s & -- \\\\" % fmt_ms(prim)]
    for k, v in ap_add.items():
        lines.append("\\quad %s & %s & %s \\\\" % (k, fmt_ms(v), fmt_ms(np.array(v) - np.array(prim))))
    lines.append("all %d features & %s & %s \\\\" % (len(FEATURE_COLUMNS), fmt_ms(full), fmt_ms(np.array(full) - np.array(prim))))
    lines.append("\\midrule")
    for c in PRIMITIVE:
        lines.append("primitive minus %s & %s & %s \\\\" % (NICE_P[c], fmt_ms(ap_drop[c]), fmt_ms(np.array(prim) - np.array(ap_drop[c]))))
    lines += ["\\botrule", "\\end{tabular}"]
    write(os.path.join(args.outdir, "table_ablation.tex"), lines)

    lines = ["\\begin{tabular}{lcc}", "\\toprule",
             "Cutoff $y<c$ & Barren fraction & AUC (grouped CV) \\\\", "\\midrule"]
    for cut, d in sens.items():
        lines.append("$c=%s$ & %.1f\\%% & $%.3f \\pm %.3f$ \\\\" % (
            cut, 100 * d["barren_frac"], d["auc"][0], d["auc"][1]))
    lines += ["\\botrule", "\\end{tabular}"]
    write(os.path.join(args.outdir, "table_cutoff.tex"), lines)

    lines = ["\\begin{tabular}{lllcrr}", "\\toprule",
             "Gate & Pattern & Nominal cost & Effective weight & Mean $\\log_{10}\\overline{\\mathrm{Var}}$ & Circuits \\\\",
             "\\midrule"]
    for i, r_ in enumerate(eff_rows):
        if i == 6:
            lines.append("\\midrule")
        lines.append("%s & %s & %s & %s & $%.2f$ & %d \\\\" % (
            r_["gate"], r_["pattern"], r_["cost"], r_["w"], r_["mean_label"], r_["rows"]))
    lines += ["\\botrule", "\\end{tabular}"]
    write(os.path.join(args.outdir, "table_effweight.tex"), lines)

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

    names = [NICE_P[c] for c in PRIMITIVE]
    deltas = [ms(np.array(prim) - np.array(ap_drop[c]))[0] for c in PRIMITIVE]
    errs = [ms(np.array(prim) - np.array(ap_drop[c]))[1] for c in PRIMITIVE]
    order_i = np.argsort(deltas)
    plt.figure(figsize=(4.2, 2.6))
    plt.barh(range(len(names)), np.array(deltas)[order_i], xerr=np.array(errs)[order_i],
             color=BLUE, height=0.7)
    plt.yticks(range(len(names)), [names[i] for i in order_i], fontsize=8)
    plt.xlabel("$\\Delta R^2$ when the primitive feature is removed", fontsize=8)
    plt.xticks(fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(args.figdir, "ablation.pdf"), bbox_inches="tight"); plt.close()

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
                label="single parameter: non-zero gradient")
    fl = legacy_floor & raw.log_var.notna().to_numpy()
    plt.scatter(raw.log_var[fl], np.full(fl.sum(), -6.5), s=4, alpha=0.4, color=RED,
                edgecolors="none", marker="v", label="single parameter: structural zero")
    plt.plot([-5, 0], [-5, 0], "k--", lw=1)
    plt.xlabel("all-parameter label $\\log_{10}\\overline{\\mathrm{Var}}$", fontsize=8)
    plt.ylabel("single-parameter label", fontsize=8)
    plt.ylim(-7.2, 0.3); plt.legend(fontsize=6.5, frameon=False, loc="lower right")
    plt.xticks(fontsize=8); plt.yticks(fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(args.figdir, "label_compare.pdf")); plt.close()
    print("figures written to", args.figdir)


if __name__ == "__main__":
    main()
