"""Generality study: shared-parameter circuits and random entangling graphs.

For each family: accuracy on unseen architectures, extrapolation from
n <= 10 to n = 11, 12, comparison with fitted linear rules, and family-specific
checks (failure of the Clifford shortcut for shared parameters; accuracy of the
causal-cone bound on random graphs; transfer from the main family).

    python -m qml_bp.families_study --main data_bp/bp_dataset_v2.csv \
        --tied data_bp/bp_tied_v2.csv --graph data_bp/bp_graph_v2.csv \
        --outdir paper/qmi/results
"""

import argparse
import json
import math
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import RepeatedKFold

from qml_bp.analyze import (PhysicsLinear, SEED, StructuredLinear, arch_id,
                            grouped_folds, hgb_reg, load_dataset, ms, regressors)
from qml_bp.ansatz import FEATURE_COLUMNS, sample_spec
from qml_bp.families import GRAPH_DESIGN, GRAPH_FEATURES, TiedSpec

warnings.filterwarnings("ignore", category=RuntimeWarning)
N_SEEDS = 10


class GraphRule:
    """Hand-built rule for random graphs: within each (rotation scheme, gate,
    cost) cell, linear in the cone-based size terms of the structured model and
    in closed-form graph and physics descriptors."""

    def __init__(self, alpha=1e-3):
        self.m = Ridge(alpha=alpha)

    @staticmethod
    def design(d):
        c = d["cone_qubits"].to_numpy(float); L = d["n_layers"].to_numpy(float); n = d["n_qubits"].to_numpy(float)
        dens = np.clip(d["ent_density"].to_numpy(float), 1e-3, None)
        base = np.c_[np.ones(len(d)), c, np.minimum(L, c), np.exp(-L / c), 1.0 / L,
                     dens, np.log10(dens), d["ent_mean_degree"].to_numpy(float),
                     d["cost_weight_eff"].to_numpy(float) / n, d["cone_frac"].to_numpy(float),
                     d["resample"].to_numpy(float), d["deg_q0"].to_numpy(float) / n]
        cell = (d["ansatz_type"].astype(int) * 4 + d["entangler_gate"].astype(int) * 2
                + d["cost_global"].astype(int)).to_numpy()
        oh = np.eye(8)[cell]
        return np.hstack([oh * base[:, [j]] for j in range(base.shape[1])])

    def fit(self, d, y):
        self.n_coef = self.design(d.iloc[:1]).shape[1]
        self.m.fit(self.design(d), y); return self

    def predict(self, d):
        return self.m.predict(self.design(d))


class Cols:
    """Wrap a scikit-learn model so that it reads named columns of a DataFrame."""
    def __init__(self, make, cols):
        self.m = make(); self.cols = cols
    def fit(self, d, y):
        self.m.fit(d[self.cols].to_numpy(float), y); return self
    def predict(self, d):
        return self.m.predict(d[self.cols].to_numpy(float))


def seeds_extrap(make_seeded, dtr, ytr, dte, yte):
    v = [r2_score(yte, make_seeded(s).fit(dtr, ytr).predict(dte)) for s in range(N_SEEDS)]
    return {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)), "min": float(np.min(v)), "max": float(np.max(v))}


def shortcut_check(n_circuits=100, seed=7):
    """Shared angles: compare the true gradient variance (uniform angles) with
    what the four-point Clifford grid would give."""
    rng = np.random.default_rng(seed)
    diffs = []; zero_but_nonzero = 0; total = 0
    for _ in range(n_circuits):
        spec = TiedSpec(sample_spec(rng, (3, 8), (2, 6)))
        U = np.array([spec.cost_and_gradient(rng.uniform(0, 2 * math.pi, spec.n_params))[1] for _ in range(600)])
        Gd = np.array([spec.cost_and_gradient(rng.integers(0, 4, spec.n_params) * math.pi / 2)[1] for _ in range(600)])
        vu, vg = U.var(axis=0), Gd.var(axis=0)
        ok = vu > 1e-6
        total += int(ok.sum()); zero_but_nonzero += int((vg[ok] < 1e-12).sum())
        if ok.any() and vg[ok].mean() > 0:
            diffs.append(abs(math.log10(vg[ok].mean()) - math.log10(vu[ok].mean())))
    return {"n_circuits": n_circuits, "median_abs_log10_diff": float(np.median(diffs)),
            "frac_circuits_off_by_factor_two": float(np.mean(np.array(diffs) > 0.3)),
            "frac_params_grid_zero_but_true_nonzero": float(zero_but_nonzero / total)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", required=True)
    ap.add_argument("--tied", required=True)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--outdir", default="paper/qmi/results")
    args = ap.parse_args()
    R = {}

    # ---------------- shared parameters ----------------
    t = load_dataset(args.tied)
    R_t = {"rows_generated": int(len(t))}
    t = t.dropna(subset=["log_var"]).reset_index(drop=True)
    y = t.log_var.to_numpy(float); g = arch_id(t).to_numpy(); nq = t.n_qubits.to_numpy(int)
    R_t.update({"rows": int(len(t)), "label_mean": float(y.mean()), "label_std": float(y.std()),
                "below_cutoff_frac": float((y < -2).mean()), "n_unique_arch": int(len(set(g)))})
    w = pd.Series(y).groupby(g).transform(lambda s: s.var(ddof=1) if len(s) > 1 else np.nan)
    R_t["r2_ceiling"] = float(1 - np.nanmean(w) / np.var(y))
    F = FEATURE_COLUMNS
    models = {"Hist Gradient Boosting": lambda s=0: Cols(lambda: hgb_reg(s), F),
              "MLP": lambda s=0: Cols(lambda: regressors(s)["MLP"], F),
              "Structured linear": lambda s=0: Cols(lambda: StructuredLinear("reg"), F),
              "Physics-informed linear (effective weight)": lambda s=0: Cols(lambda: PhysicsLinear("reg", effective=True), F),
              "Physics-informed linear": lambda s=0: Cols(lambda: PhysicsLinear("reg"), F)}
    folds = list(grouped_folds(g, 5, 2))
    R_t["cv"] = {}
    for name, mk in models.items():
        r = [r2_score(y[te], mk().fit(t.iloc[tr], y[tr]).predict(t.iloc[te])) for _, tr, te in folds]
        R_t["cv"][name] = ms(r)
        print("tied  CV   %-44s %.3f +- %.3f" % (name, *ms(r)), flush=True)
    tr = np.where(nq <= 10)[0]; te = np.where(nq > 10)[0]
    R_t["extrap"] = {"train_rows": int(len(tr)), "test_rows": int(len(te))}
    for name, mk in models.items():
        if name == "MLP":
            R_t["extrap"][name] = seeds_extrap(mk, t.iloc[tr], y[tr], t.iloc[te], y[te])
        else:
            R_t["extrap"][name] = {"mean": float(r2_score(y[te], mk().fit(t.iloc[tr], y[tr]).predict(t.iloc[te])))}
        print("tied  11-12 %-44s %.3f" % (name, R_t["extrap"][name]["mean"]), flush=True)
    # transfer from the main (independent-parameter) family
    u = load_dataset(args.main).dropna(subset=["log_var"]).reset_index(drop=True)
    yu = u.log_var.to_numpy(float)
    p = Cols(lambda: hgb_reg(), F).fit(u, yu).predict(t)
    R_t["transfer_from_main"] = {"r2": float(r2_score(y, p)), "bias": float(np.mean(p - y))}
    ku = pd.Series(yu).groupby(arch_id(u).to_numpy()).mean(); kt = pd.Series(y).groupby(g).mean()
    d = (kt - ku).dropna()
    R_t["offset_vs_main"] = {"mean": float(d.mean()), "sd": float(d.std()), "corr": float(np.corrcoef(kt[d.index], ku[d.index])[0, 1]), "n_arch": int(len(d))}
    R_t["clifford_shortcut"] = shortcut_check()
    print("tied  shortcut check:", R_t["clifford_shortcut"], flush=True)
    R["tied"] = R_t

    # ---------------- random graphs ----------------
    gdf = pd.read_csv(args.graph)
    R_g = {"rows_generated": int(len(gdf))}
    gdf = gdf.dropna(subset=["log_var"]).reset_index(drop=True)
    yg = gdf.log_var.to_numpy(float); nqg = gdf.n_qubits.to_numpy(int)
    R_g.update({"rows": int(len(gdf)), "label_mean": float(yg.mean()), "label_std": float(yg.std()),
                "below_cutoff_frac": float((yg < -2).mean())})
    fr = 1 - gdf.frac_structural - gdf.cone_frac
    R_g["cone_vs_structural"] = {"mad": float(np.mean(np.abs(fr))),
                                 "mad_fixed_ry": float(np.mean(np.abs(fr)[gdf.ansatz_type == 0])),
                                 "exact_frac_fixed_ry": float(np.mean(np.abs(fr)[gdf.ansatz_type == 0] < 1e-9))}
    gm = {"Hist Gradient Boosting": lambda s=0: Cols(lambda: hgb_reg(s), GRAPH_FEATURES),
          "MLP": lambda s=0: Cols(lambda: regressors(s)["MLP"], GRAPH_FEATURES),
          "Descriptor rule (linear)": lambda s=0: GraphRule(),
          "HGB, design parameters only": lambda s=0: Cols(lambda: hgb_reg(s), GRAPH_DESIGN)}
    kf = list(RepeatedKFold(n_splits=5, n_repeats=2, random_state=SEED).split(gdf))
    R_g["cv"] = {}
    for name, mk in gm.items():
        r = [r2_score(yg[te], mk().fit(gdf.iloc[tr], yg[tr]).predict(gdf.iloc[te])) for tr, te in kf]
        R_g["cv"][name] = ms(r)
        print("graph CV   %-44s %.3f +- %.3f" % (name, *ms(r)), flush=True)
    R_g["rule_n_coef"] = int(GraphRule().fit(gdf, yg).n_coef)
    tr = np.where(nqg <= 10)[0]; te = np.where(nqg > 10)[0]
    R_g["extrap"] = {"train_rows": int(len(tr)), "test_rows": int(len(te))}
    for name, mk in gm.items():
        if name == "MLP":
            R_g["extrap"][name] = seeds_extrap(mk, gdf.iloc[tr], yg[tr], gdf.iloc[te], yg[te])
        else:
            R_g["extrap"][name] = {"mean": float(r2_score(yg[te], mk().fit(gdf.iloc[tr], yg[tr]).predict(gdf.iloc[te])))}
        print("graph 11-12 %-44s %.3f" % (name, R_g["extrap"][name]["mean"]), flush=True)
    R["graph"] = R_g
    json.dump(R, open(os.path.join(args.outdir, "families.json"), "w"), indent=1)

    # ---------------- summary table (with the main family) ----------------
    M = json.load(open(os.path.join(args.outdir, "results.json")))
    mcv = M["grouped_cv_reg"]; mex = M["extrap_reg"]
    def pm(v):
        return "$%.3f \\pm %.3f$" % (v[0], v[1])
    lines = ["\\begin{tabular}{lccc}", "\\toprule",
             " & Named patterns & Shared parameters & Random graphs \\\\", "\\midrule",
             "Circuits & %d & %d & %d \\\\" % (M["n_rows"], R_t["rows"], R_g["rows"]),
             "Exact Clifford shortcut & yes & no & yes \\\\",
             "Noise ceiling on $R^2$ & %.3f & %.3f & -- \\\\" % (M["r2_ceiling"], R_t["r2_ceiling"]),
             "\\midrule", "\\multicolumn{4}{l}{\\emph{Unseen architectures, $R^2$}}\\\\",
             "\\quad HGB & %s & %s & %s \\\\" % (pm(mcv["Hist Gradient Boosting"]["r2"]), pm(R_t["cv"]["Hist Gradient Boosting"]), pm(R_g["cv"]["Hist Gradient Boosting"])),
             "\\quad MLP & %s & %s & %s \\\\" % (pm(mcv["MLP"]["r2"]), pm(R_t["cv"]["MLP"]), pm(R_g["cv"]["MLP"])),
             "\\quad fitted linear rule & %s & %s & %s \\\\" % (pm(mcv["Structured linear"]["r2"]), pm(R_t["cv"]["Structured linear"]), pm(R_g["cv"]["Descriptor rule (linear)"])),
             "\\quad five-coefficient rule & %s & %s & -- \\\\" % (pm(mcv["Physics-informed linear"]["r2"]), pm(R_t["cv"]["Physics-informed linear"])),
             "\\midrule", "\\multicolumn{4}{l}{\\emph{Trained on $n\\le10$, tested on $n=11,12$, $R^2$}}\\\\",
             "\\quad MLP (10 seeds) & $%.3f \\pm %.3f$ & $%.3f \\pm %.3f$ & $%.3f \\pm %.3f$ \\\\" % (
                 M["extrap_mlp_r2_seeds"]["mean"], M["extrap_mlp_r2_seeds"]["sd"], R_t["extrap"]["MLP"]["mean"], R_t["extrap"]["MLP"]["sd"],
                 R_g["extrap"]["MLP"]["mean"], R_g["extrap"]["MLP"]["sd"]),
             "\\quad HGB & %.3f & %.3f & %.3f \\\\" % (mex["Hist Gradient Boosting"]["r2"], R_t["extrap"]["Hist Gradient Boosting"]["mean"], R_g["extrap"]["Hist Gradient Boosting"]["mean"]),
             "\\quad fitted linear rule & %.3f & %.3f & %.3f \\\\" % (mex["Structured linear"]["r2"], R_t["extrap"]["Structured linear"]["mean"], R_g["extrap"]["Descriptor rule (linear)"]["mean"]),
             "\\botrule", "\\end{tabular}"]
    open(os.path.join(args.outdir, "table_families.tex"), "w").write("\n".join(lines) + "\n")
    print("wrote families.json, table_families.tex")


if __name__ == "__main__":
    main()
