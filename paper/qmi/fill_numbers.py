"""Inline every number and table of the manuscript from the analysis outputs.

Reads  results/results.json   (from  python -m qml_bp.analyze)
       results/validation.json (from  python -m qml_bp.validate --json ...)
       results/repeat.json     (from  python -m qml_bp.repeat_labels --json ...)
and rewrites the %%BEGIN-NUMBERS / %%END-NUMBERS macro block and every
%%BEGIN-TABLE:<name> / %%END-TABLE:<name> block in main.tex and in
response_to_reviewers.tex, so the single-file manuscript stays self-contained.

    python paper/qmi/fill_numbers.py
"""

import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def f(x, nd=3):
    return "%.*f" % (nd, x)


def pct(x, nd=1):
    return "%.*f" % (nd, 100 * x)


def big(n):
    return "{:,}".format(int(n)).replace(",", "\\,")


def main():
    R = json.load(open(os.path.join(RES, "results.json")))
    V = json.load(open(os.path.join(RES, "validation.json")))
    P = json.load(open(os.path.join(RES, "repeat.json")))
    cv = R["grouped_cv_reg"]; cc = R["grouped_cv_cls"]
    hgb, phys, sub = "Hist Gradient Boosting", "Physics-informed linear", "Cost-locality subgroup mean"
    drops = R["ablation"]["drop"]
    # rank primitive groups (exclude the composite 'size' and 'entanglement' groups)
    prim = {k: v for k, v in drops.items() if k in
            ("qubit count n", "layers L", "entangle pattern", "entangler gate",
             "cost locality", "rotation scheme")}
    ranked = sorted(prim.items(), key=lambda kv: -kv[1]["delta"][0])
    full = R["ablation"]["reference"]["full (9 features)"][0]
    primonly = R["ablation"]["reference"]["primitive only (6 features)"][0]
    gain = cv[hgb]["r2"][0] - cv[phys]["r2"][0]
    brk = R["extrap_breakdown"]

    M = {
        "Nrows": big(R["n_rows_raw"]), "Nspecs": big(R["n_rows_raw"]),
        "Nused": big(R["n_rows"]), "NallStruct": str(R["n_all_structural"]),
        "Samples": "200", "QubitLo": "2", "QubitHi": "12", "LayerLo": "1", "LayerHi": "20",
        "Narch": big(R["n_unique_arch"]), "DesignSpace": big(R["design_space_size"]),
        "MultMedian": "%g" % R["mult_median"], "MultMax": str(R["mult_max"]),
        "RowLeak": pct(R["row_split_leak_frac"]),
        "BarrenFrac": pct(R["barren_frac"]), "GlobalFrac": pct(R["global_frac"]),
        "LabelMean": f(R["label_mean"], 2), "LabelStd": f(R["label_std"], 2),
        "LabelMin": f(R["label_min"], 2), "LabelMax": f(R["label_max"], 2),
        "WithinVar": f(R["within_arch_var"], 3), "TotalVar": f(R["total_var"], 3),
        "RsqCeiling": f(R["r2_ceiling"], 3), "RsqCeilingRy": f(R["r2_ceiling_fixed_ry"], 3),
        "RsqCeilingPauli": f(R["r2_ceiling_random_pauli"], 3),
        "StructFracParams": pct(R["struct_frac_params_overall"]),
        "StructRowsAny": pct(R["struct_rows_any"]),
        "LegacyFloorFrac": pct(R["legacy_floor_frac"]),
        "LegacyCorr": f(R["legacy_vs_new_corr"], 3), "LegacyMAD": f(R["legacy_vs_new_mad"], 2),
        "LegacyFloorTrainable": pct(R["legacy_floor_barren_but_new_trainable"], 0),
        "SpreadMedian": f(R["within_circuit_spread_median"], 1),
        "SpreadP90": f(R["within_circuit_spread_p90"], 1),
        "CostGap": f(abs(R["cost_gap"]), 2) if "cost_gap" in R else "--",
        "NFolds": str(R["n_folds"]), "NFoldsK": str(R["n_folds_k"]), "NRepeats": str(R["n_repeats"]),
        "RsqCV": f(cv[hgb]["r2"][0]), "RsqCVsd": f(cv[hgb]["r2"][1]),
        "MaeCV": f(cv[hgb]["mae"][0]), "MaeCVsd": f(cv[hgb]["mae"][1]),
        "AucCV": f(cc[hgb]["auc"][0]), "AucCVsd": f(cc[hgb]["auc"][1]),
        "AccCV": f(cc[hgb]["acc"][0]), "AccCVsd": f(cc[hgb]["acc"][1]),
        "RsqPhys": f(cv[phys]["r2"][0]), "RsqSub": f(cv[sub]["r2"][0]),
        "AucPhys": f(cc[phys]["auc"][0]),
        "GainOverPhys": f(gain, 2),
        "FracOfCeiling": pct(cv[hgb]["r2"][0] / R["r2_ceiling"], 0),
        "OofRsq": f(R["oof_r2"]), "OofRsqLo": f(R["oof_r2_ci"][0]), "OofRsqHi": f(R["oof_r2_ci"][1]),
        "RowRsq": f(R["row_split_r2"]), "RowMae": f(R["row_split_mae"]), "RowAuc": f(R["row_split_auc"]),
        "ExtrapCut": "10", "ExtrapTrainRows": big(R["extrap_train_rows"]),
        "ExtrapTestRows": big(R["extrap_test_rows"]),
        "RsqExtrap": f(R["extrap_reg"][hgb]["r2"]), "MaeExtrap": f(R["extrap_reg"][hgb]["mae"]),
        "RsqExtrapLo": f(R["extrap_hgb_r2_ci"][0]), "RsqExtrapHi": f(R["extrap_hgb_r2_ci"][1]),
        "MaeExtrapLo": f(R["extrap_hgb_mae_ci"][0]), "MaeExtrapHi": f(R["extrap_hgb_mae_ci"][1]),
        "RsqExtrapSeedSd": f(R["extrap_hgb_r2_seeds"][1]),
        "AucExtrap": f(R["extrap_cls"][hgb]["auc"]),
        "RsqExtrapPhys": f(R["extrap_reg"][phys]["r2"]), "MaeExtrapPhys": f(R["extrap_reg"][phys]["mae"]),
        "RsqExtrapSub": f(R["extrap_reg"][sub]["r2"]),
        "RsqFull": f(full), "RsqPrim": f(primonly), "DeltaDerived": f(full - primonly, 3),
        "AblTopOne": ranked[0][0], "AblTopOneDelta": f(ranked[0][1]["delta"][0]),
        "AblTopTwo": ranked[1][0], "AblTopTwoDelta": f(ranked[1][1]["delta"][0]),
        "AblTopThree": ranked[2][0], "AblTopThreeDelta": f(ranked[2][1]["delta"][0]),
        "DeltaSize": f(drops["size (n, L, nL, L/n)"]["delta"][0]),
        "PythonVersion": R["software"]["python"], "NumpyVersion": R["software"]["numpy"],
        "SklearnVersion": R["software"]["sklearn"],
        "ValN": str(V["n_circuits"]), "ValState": "%.1e" % V["max_state_diff"],
        "ValGrad": "%.1e" % V["max_grad_diff"], "ValGradN": big(V["n_params_checked"]),
        "ValStructN": str(V["n_flagged"]), "ValStructViol": str(V["n_violations"]),
        "RepeatN": str(P["n_specs"]), "RepeatMAD": f(P["mad"], 3), "RepeatMax": f(P["max_abs"], 2),
    }
    for k in ["n=11 local", "n=11 global", "n=12 local", "n=12 global", "n=11", "n=12", "local", "global"]:
        key = k.replace("n=", "N").replace(" ", "").capitalize()
        M["MaeE" + key] = f(brk[k]["hgb_mae"]); M["MaeEP" + key] = f(brk[k]["phys_mae"])
        M["RsqE" + key] = f(brk[k]["hgb_r2"]); M["RsqEP" + key] = f(brk[k]["phys_r2"])
        if "mlp_r2" in brk[k]:
            M["RsqEM" + key] = f(brk[k]["mlp_r2"]); M["MaeEM" + key] = f(brk[k]["mlp_mae"])
    if "extrap_mlp_r2_ci" in R:
        M["RsqExtrapMLPLo"] = f(R["extrap_mlp_r2_ci"][0]); M["RsqExtrapMLPHi"] = f(R["extrap_mlp_r2_ci"][1])
    for cut, d in R["cutoff_sensitivity"].items():
        key = cut.replace("-", "m").replace(".", "p")
        M["AucCut" + key] = f(d["auc"][0]); M["BarrenCut" + key] = pct(d["barren_frac"], 0)
    for name in ["MLP", "Random Forest", "Extra Trees", "k-NN", "Linear baseline"]:
        if name in cv:
            key = name.replace(" ", "").replace("-", "")
            M["RsqCV" + key] = f(cv[name]["r2"][0]); M["AucCV" + key] = f(cc[name]["auc"][0])
            M["RsqExtrap" + key] = f(R["extrap_reg"][name]["r2"])
    for k in ["N11local", "N11global", "N12local", "N12global", "N11", "N12", "Local", "Global"]:
        M.setdefault("RsqEM" + k, "??"); M.setdefault("MaeEM" + k, "??")
    M.setdefault("RsqExtrapMLPLo", "??"); M.setdefault("RsqExtrapMLPHi", "??")
    extra_path = os.path.join(RES, "extra.json")
    if os.path.exists(extra_path):
        E = json.load(open(extra_path)); sc = E["screening"]
        M["ScreenSpearman"] = f(sc["spearman"]); M["ScreenTrainableFrac"] = pct(sc["trainable_frac"])
        M["ScreenPrecFifty"] = pct(sc["top"]["50"]["precision_trainable"])
        M["ScreenRecallFifty"] = pct(sc["top"]["50"]["recall_trainable"])
        M["ScreenBottomTen"] = pct(sc["bottom"]["10"]["precision_barren"])
        # largest top-k% with zero barren leakage
        clean = [int(k) for k, v in sc["top"].items() if v["barren_leak"] == 0.0]
        top = max(clean) if clean else 1
        M["ScreenTopPct"] = str(top); M["ScreenTopK"] = big(sc["top"][str(top)]["k"])
        words = {"6": "Six", "7": "Seven", "8": "Eight", "9": "Nine", "10": "Ten"}
        for k, e in E["gap"].items():
            M["GapHGBk" + words[k]] = f(e["HGB"]["r2"]); M["GapMLPk" + words[k]] = f(e["MLP"]["r2"])
            M["GapPhysk" + words[k]] = f(e["Physics-informed linear"]["r2"])
        lcw = {"250": "TwoFifty", "500": "FiveHundred", "1000": "OneThousand", "2000": "TwoThousand",
               "4000": "FourThousand", "8000": "EightThousand", "16000": "SixteenThousand"}
        lc = E["learning_curve"]
        for k, w in lcw.items():
            M["LCHGB" + w] = f(lc[k]["HGB"][0]); M["LCMLP" + w] = f(lc[k]["MLP"][0])
        final = lc["16000"]["HGB"][0]
        M["LCRowsNinety"] = big(min(int(k) for k in lc if lc[k]["HGB"][0] > 0.9 and lc[k]["MLP"][0] > 0.9))
        M["LCRowsSaturate"] = big(min(int(k) for k in lc if final - lc[k]["HGB"][0] < 0.02))
    lopo_path = os.path.join(RES, "lopo.json")
    if os.path.exists(lopo_path):
        Lp = json.load(open(lopo_path)); lo = Lp["lopo"]
        M["LopoExtraRows"] = big(lo["brickwork"]["rows"] + lo["star"]["rows"])
        M["LopoDescRsq"] = f(Lp["pooled_grouped_cv"]["descriptor_r2"])
        M["LopoIndexRsq"] = f(Lp["pooled_grouped_cv"]["index_r2"])
        for name, key in [("linear", "Linear"), ("circular", "Circular"), ("all_to_all", "All"),
                          ("brickwork", "Brick"), ("star", "Star")]:
            M["Lopo%sUnseen" % key] = f(lo[name]["unseen_r2"])
            M["Lopo%sSeen" % key] = f(lo[name]["seen_r2"])
            M["Lopo%sPhys" % key] = f(lo[name]["phys_r2"])
            M["Lopo%sUnseenMae" % key] = f(lo[name]["unseen_mae"])
            M["Lopo%sUnseenAuc" % key] = f(lo[name]["unseen_auc"])
    else:
        for k in ["LopoExtraRows", "LopoDescRsq", "LopoIndexRsq"] + [
                "Lopo%s%s" % (a, b) for a in ["Linear", "Circular", "All", "Brick", "Star"]
                for b in ["Unseen", "Seen", "Phys", "UnseenMae", "UnseenAuc"]]:
            M[k] = "??"
    digits = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four",
              "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}
    def latex_name(k):  # macro names may not contain digits
        return "".join(digits.get(ch, ch) for ch in k)
    M = {latex_name(k): v for k, v in M.items()}
    lines = ["%%BEGIN-NUMBERS"]
    lines += ["\\newcommand{\\%s}{%s}" % (k, v) for k, v in M.items()]
    lines.append("%%END-NUMBERS")
    block = "\n".join(lines)

    tables = {}
    for t in ["datastats", "labelbydesign", "grouped_cv", "extrap", "extrap_breakdown",
              "ablation", "cutoff", "lopo", "gap", "screening"]:
        tp = os.path.join(RES, "table_%s.tex" % t)
        tables[t] = open(tp).read().strip() if os.path.exists(tp) else "% (pending)"

    for fname in ["main.tex", "response_to_reviewers.tex", "cover_letter.tex", "supplement.tex"]:
        path = os.path.join(HERE, fname)
        if not os.path.exists(path):
            continue
        s = open(path).read()
        s = re.sub(r"^%%BEGIN-NUMBERS\n.*?^%%END-NUMBERS", lambda m: block, s, flags=re.S | re.M)
        for t, body in tables.items():
            s = re.sub(r"(^%%%%BEGIN-TABLE:%s\n).*?(^%%?%%END-TABLE:%s)" % (t, t),
                       lambda m: m.group(1) + body + "\n" + m.group(2), s, flags=re.S | re.M)
        open(path, "w").write(s)
        print("filled", fname, "(%d macros)" % len(M))


if __name__ == "__main__":
    main()
