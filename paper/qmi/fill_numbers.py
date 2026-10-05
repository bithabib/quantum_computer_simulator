"""Inline every number and table of the manuscript from the analysis outputs.

Reads  results/results.json   (from  python -m qml_bp.analyze)
       results/validation.json (from  python -m qml_bp.validate --json ...)
       results/repeat.json     (from  python -m qml_bp.repeat_labels --json ...)
and rewrites the %%BEGIN-NUMBERS / %%END-NUMBERS macro block and every
%%BEGIN-TABLE:<name> / %%END-TABLE:<name> block in main.tex, supplement.tex
and (when present) the letters, so the single-file manuscript stays self-contained.

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
    stl, phe = "Structured linear", "Physics-informed linear (effective weight)"
    drops = R["ablation"]["drop"]
    AP = R["ablation_primitive"]
    ranked = sorted(AP["drop"].items(), key=lambda kv: -kv[1]["delta"][0])
    full = AP["full"][0]; primonly = AP["primitive"][0]
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
        "DeltaAddEff": f(AP["add"]["+ effective weight"]["delta"][0], 3),
        "DeltaGatePrim": f(AP["drop"]["entangler gate"]["delta"][0], 3),
        "DeltaAddCone": f(AP["add"]["+ causal cone"]["delta"][0], 3),
        "DeltaAddBoth": f(max(abs(v["delta"][0]) for v in AP["add"].values()), 3),
        "PythonVersion": R["software"]["python"], "NumpyVersion": R["software"]["numpy"],
        "SklearnVersion": R["software"]["sklearn"],
        "ValN": str(V["n_circuits"]), "ValState": "%.1e" % V["max_state_diff"],
        "ValGrad": "%.1e" % V["max_grad_diff"], "ValGradN": big(V["n_params_checked"]),
        "ValStructN": str(V["n_flagged"]), "ValStructViol": str(V["n_violations"]),
        "RepeatN": str(P["n_specs"]), "RepeatMAD": f(P["mad"], 3), "RepeatMax": f(P["max_abs"], 2),
        "RsqStruct": f(cv[stl]["r2"][0]), "RsqStructsd": f(cv[stl]["r2"][1]),
        "AucStruct": f(cc[stl]["auc"][0]), "StructNcoef": str(R["struct_n_coef"]),
        "RsqPhysEff": f(cv[phe]["r2"][0]),
        "GainOverStruct": f(cv[hgb]["r2"][0] - cv[stl]["r2"][0], 3),
        "RsqExtrapStruct": f(R["extrap_reg"][stl]["r2"]), "MaeExtrapStruct": f(R["extrap_reg"][stl]["mae"]),
        "RsqExtrapMLPmean": f(R["extrap_mlp_r2_seeds"]["mean"]), "RsqExtrapMLPsd": f(R["extrap_mlp_r2_seeds"]["sd"]),
        "DeltaCostObs": f(drops["cost observable (locality + effective weight)"]["delta"][0]),
        "DeltaGateEff": f(drops["entangler gate + effective weight"]["delta"][0]),
        "DeltaEffWeight": f(drops["effective observable weight"]["delta"][0]),
        "GateGap": f(R["gate_gap"], 2), "NFeatures": str(R["n_features"]),
        "ConeExactRy": pct(R["cone_vs_structural"]["exact_frac_fixed_ry"], 0),
        "ConeMadRy": f(R["cone_vs_structural"]["mad_fixed_ry"], 3),
        "ConeMadPauli": f(R["cone_vs_structural"]["mad_random_pauli"], 3),
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
    M["CrossLo"] = {6: "six", 7: "seven", 8: "eight"}.get(R["cutoff_cross_lo"], str(R["cutoff_cross_lo"]))
    M["CrossHi"] = {7: "seven", 8: "eight", 9: "nine"}.get(R["cutoff_cross_hi"], str(R["cutoff_cross_hi"]))
    M["AllAboveMaxN"] = str(R["all_above_max_n"]); M["BelowAtFive"] = {3: "three"}.get(R["below_cutoff_at_n5"], str(R["below_cutoff_at_n5"]))
    M["RowsAtFive"] = big(R["rows_at_n5"])
    import math as _m
    v_ = R["min_nonstructural_param_var"]; e_ = _m.floor(_m.log10(v_))
    M["MinParamVar"] = "%.1f\\times10^{%d}" % (v_ / 10 ** e_, e_)
    M["MinParamGradExp"] = "%d" % round(_m.log10(_m.sqrt(v_)))
    sw_path = os.path.join(RES, "training_sweep.json")
    if os.path.exists(sw_path):
        SW = json.load(open(sw_path))
        fr = [min(r["frac_pred_barren"], r["frac_pred_trainable"]) for r in SW] + [max(r["frac_pred_barren"], r["frac_pred_trainable"]) for r in SW]
        rho = [r["spearman_true"] for r in SW] + [r["spearman_pred"] for r in SW]
        M["TOFracMin"] = pct(min(fr), 0); M["TOFracMax"] = pct(max(fr), 0)
        M["TORhoMin"] = f(min(rho), 2); M["TORhoMax"] = f(max(rho), 2)
        rt = [r["spearman_true"] for r in SW]; rp = [r["spearman_pred"] for r in SW]
        M["TORhoTrueMin"] = f(min(rt), 2); M["TORhoTrueMax"] = f(max(rt), 2)
        M["TORhoPredMin"] = f(min(rp), 2); M["TORhoPredMax"] = f(max(rp), 2)
        M["TOCircuitsMin"] = str(min(r["n"] for r in SW)); M["TOCircuitsMax"] = str(max(r["n"] for r in SW))
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
            M["GapStructk" + words[k]] = f(e["Structured linear"]["r2"])
            sd_ = e["MLP_seeds"]
            M["GapMLPmeank" + words[k]] = f(sd_["mean"]); M["GapMLPsdk" + words[k]] = f(sd_["sd"])
            M["GapMLPmink" + words[k]] = f(sd_["min"], 2); M["GapMLPmaxk" + words[k]] = f(sd_["max"], 2)
        M["ScreenByNPrec"] = pct(sc["by_n_only_top20"]["precision"], 0)
        M["ScreenByNRecall"] = pct(sc["by_n_only_top20"]["recall"], 0)
        wn = E["screening_within_n"]
        for key, col in [("HGB", "spearman_hgb"), ("Phys", "spearman_phys"), ("Struct", "spearman_struct")]:
            vals = [d[col] for d in wn.values()]
            M["ScrSp%smin" % key] = f(min(vals), 2); M["ScrSp%smax" % key] = f(max(vals), 2)
        n12 = wn.get("12") or wn.get(12)
        M["ScrAboveTwelve"] = pct(n12["above_cutoff_frac"], 0); M["ScrPrecTwelve"] = pct(n12["prec_hgb"], 1)
        lcw = {"250": "TwoFifty", "500": "FiveHundred", "1000": "OneThousand", "2000": "TwoThousand",
               "4000": "FourThousand", "8000": "EightThousand", "16000": "SixteenThousand"}
        lc = E["learning_curve"]
        for k, w in lcw.items():
            M["LCHGB" + w] = f(lc[k]["HGB"][0]); M["LCMLP" + w] = f(lc[k]["MLP"][0])
        final = lc["16000"]["HGB"][0]
        M["LCRowsNinety"] = big(min(int(k) for k in lc if lc[k]["HGB"][0] > 0.9 and lc[k]["MLP"][0] > 0.9))
        M["LCRowsSaturate"] = big(min(int(k) for k in lc if final - lc[k]["HGB"][0] < 0.02))
    large_path = os.path.join(RES, "large.json")
    if os.path.exists(large_path):
        Lg = json.load(open(large_path)); mm = Lg["models"]
        M["LargeN"] = big(Lg["test_rows"])
        M["LargeRsqMLPmean"] = f(Lg["mlp_seeds"]["mean"]); M["LargeRsqMLPsd"] = f(Lg["mlp_seeds"]["sd"])
        for name, key in [("Hist Gradient Boosting", "HGB"), ("MLP", "MLP"), ("Physics-informed linear", "Phys"),
                          ("Structured linear", "Struct")]:
            e = mm[name]
            M["LargeRsq" + key] = f(e["r2"]); M["LargeMae" + key] = f(e["mae"])
            M["LargeRsq%sLo" % key] = f(e["r2_ci"][0]); M["LargeRsq%sHi" % key] = f(e["r2_ci"][1])
            pn = e["per_n"]
            M["LargeRsq%sThirteen" % key] = f((pn.get(13) or pn.get("13"))["r2"])
            M["LargeRsq%sFourteen" % key] = f((pn.get(14) or pn.get("14"))["r2"])
    else:
        for k in ["LargeN"] + ["LargeRsq%s%s" % (a, b) for a in ["HGB", "MLP", "Phys"]
                               for b in ["", "Lo", "Hi", "Thirteen", "Fourteen"]] + \
                 ["LargeMae%s" % a for a in ["HGB", "MLP", "Phys"]]:
            M[k] = "??"
    # Clifford-sampling study (validation + extrapolation), if present
    cv_path = os.path.join(RES, "validation_clifford.json")
    if os.path.exists(cv_path):
        C = json.load(open(cv_path))
        M["ClValN"] = str(C["n_circuits"]); M["ClValParams"] = big(C["n_params"])
        M["ClValMAD"] = f(C["log10_diff_mad"], 3); M["ClValMax"] = f(C["log10_diff_max"], 2)
        M["ClValStruct"] = big(C["struct_sv_total"]); M["ClValStructHits"] = str(C["struct_with_clifford_hits"])
        M["ClValSamples"] = big(C["cl_samples"]); M["ClValSvSamples"] = str(C["sv_samples"])
        M["ClValBelow"] = {1: "one"}.get(C["mismatch_zero_circuits"], str(C["mismatch_zero_circuits"]))
    cs_path = os.path.join(RES, "clifford_study.json")
    if os.path.exists(cs_path):
        CS = json.load(open(cs_path)); ov = CS["overall"]
        M["ClTrainRows"] = big(CS["train_rows"]); M["ClTestRows"] = big(CS["test_rows"])
        M["ClResolved"] = big(CS["test_resolved"]); M["ClCensored"] = big(CS["test_censored"])
        M["ClStructCensOK"] = pct(ov["Structured linear"].get("censored_consistent", float("nan")), 0)
        letters = {"13-16": "A", "17-20": "B", "21-24": "C", "25-28": "D", "29-32": "E"}
        for k, b in CS["bins"].items():
            L_ = letters[k]; m = b["models"]
            M["ClStructRsq" + L_] = f(m["Structured linear"]["r2"], 2); M["ClStructMae" + L_] = f(m["Structured linear"]["mae"], 2)
            M["ClMLPRsq" + L_] = "%.2f \\pm %.2f" % (m["MLP_seeds"]["r2_mean"], m["MLP_seeds"]["r2_sd"])
            M["ClHGBRsq" + L_] = f(m["Hist Gradient Boosting"]["r2"], 2)
            M["ClResolvedFrac" + L_] = pct(b["resolved_frac"], 0)
        bc = CS["bins_by_cost"]
        for k, L_ in letters.items():
            l_, g_ = bc["local " + k], bc["global " + k]
            M["ClLocStruct" + L_] = f(l_["struct_r2"], 2); M["ClLocBias" + L_] = "%+.2f" % l_["struct_bias"]
            M["ClGloStruct" + L_] = f(g_["struct_r2"], 2)
            M["ClGloCens" + L_] = pct(g_.get("censored_consistent", float("nan")), 0)
        sh = CS["shallow_local"]
        M["ClShallowTrueA"] = f(sh["13-16"]["true"], 2); M["ClShallowTrueE"] = f(sh["29-32"]["true"], 2)
        M["ClShallowStructA"] = f(sh["13-16"]["struct"], 2); M["ClShallowStructE"] = f(sh["29-32"]["struct"], 2)
        import math
        M["ClFloorMinExp"] = "%d" % round(math.log10(CS["floor_min"])); M["ClFloorMaxExp"] = "%d" % round(math.log10(CS["floor_max"]))
        M["ClTrainGenerated"] = big(CS["train_generated"]); M["ClTrainUnresolved"] = str(CS["train_generated"] - CS["train_rows"])
        M["ClSlack"] = "%g" % CS["censor_slack_log10"]
        M["ClLeTwentyResolved"] = big(CS["train_le20"]["test_rows"]); M["ClLeTwentyTotal"] = big(CS["train_le20"]["test_total_incl_censored"])
        t20 = CS["train_le20"]
        M["ClLeTwentyStruct"] = f(t20["Structured linear"]["r2"], 2)
        M["ClLeTwentyMLP"] = "%.2f \\pm %.2f" % (t20["MLP_seeds"]["r2_mean"], t20["MLP_seeds"]["r2_sd"])
        M["ClLeTwentyHGB"] = f(t20["Hist Gradient Boosting"]["r2"], 2)
        M["ClAbstractSentence"] = ("With unbiased Clifford-sampled labels up to 32 qubits we measure a prediction "
            "horizon: an ensemble of neural networks stays on average within a factor of two of the true variance for "
            "five to six qubits beyond a training range of 12 or more qubits, for local costs. Prediction beyond that horizon is an open problem, "
            "for which we release a benchmark.")
        if "horizon" in CS:
            HZ = CS["horizon"]; wk = {"8": "Eight", "12": "Twelve", "16": "Sixteen", "20": "Twenty", "24": "Twentyfour"}
            M["HzEnsemble"] = {10: "ten"}.get(HZ["n_ensemble"], str(HZ["n_ensemble"]))
            for k, w_ in wk.items():
                cut_ = HZ["cuts"][k]; mm_ = cut_["models"]
                M["HzResGlobk%s" % w_] = pct(cut_["resolved_global_within6"], 0)
                M["HzResLock%s" % w_] = pct(cut_["resolved_local_within6"], 0)
                for name, key in [("MLP ensemble", "MLP"), ("Structured linear", "Table"), ("Hist Gradient Boosting", "HGB")]:
                    for sub_, pre_ in [("all", ""), ("local", "Loc")]:
                        h_ = mm_[name][sub_]["horizon"]["0.3"]
                        ge_ = "$\\ge$" if h_.get("censored_by_data") else ""
                        M["Hz%s%sk%s" % (pre_, key, w_)] = ge_ + str(h_["h"])
                        M["Hz%s%sIntk%s" % (pre_, key, w_)] = "%d--%d" % (h_["lo"], h_["hi"])
                        M["Hz%s%sWidek%s" % (pre_, key, w_)] = str(mm_[name][sub_]["horizon"]["0.5"]["h"])
            m12 = HZ["cuts"]["12"]["models"]["MLP ensemble"]["local"]["mae_by_distance"]
            M["HzMaeTwelveDOne"] = f(m12["1"], 2); M["HzMaeTwelveDFour"] = f(m12["4"], 2); M["HzMaeTwelveDSix"] = f(m12["6"], 2)
    M["ClTargetHits"] = "2\\,000"; M["ClMaxSamples"] = "$2^{20}$"; M["ClResolvedHits"] = "100"
    M.setdefault("ClQmax", "32"); M.setdefault("ClAbstractSentence", "")
    fam_path = os.path.join(RES, "families.json")
    if os.path.exists(fam_path):
        FA = json.load(open(fam_path)); T_, G_ = FA["tied"], FA["graph"]
        M.update({
            "TiedRows": big(T_["rows_generated"]), "TiedCeil": f(T_["r2_ceiling"]),
            "TiedHGB": f(T_["cv"]["Hist Gradient Boosting"][0]), "TiedMLP": f(T_["cv"]["MLP"][0]),
            "TiedRule": f(T_["cv"]["Structured linear"][0]), "TiedPhys": f(T_["cv"]["Physics-informed linear"][0]),
            "TiedExMLP": f(T_["extrap"]["MLP"]["mean"]), "TiedExHGB": f(T_["extrap"]["Hist Gradient Boosting"]["mean"]),
            "TiedExRule": f(T_["extrap"]["Structured linear"]["mean"]),
            "TiedTransfer": f(T_["transfer_from_main"]["r2"], 2), "TiedOffset": f(T_["offset_vs_main"]["mean"], 2),
            "TiedCorr": f(T_["offset_vs_main"]["corr"], 2),
            "TiedBelowCut": pct(T_["below_cutoff_frac"], 0),
            "TiedLabelMin": f(T_.get("label_min", float("nan")), 2),
            "TiedShortN": str(T_["clifford_shortcut"]["n_circuits"]),
            "TiedShortZero": pct(T_["clifford_shortcut"]["frac_params_grid_zero_but_true_nonzero"], 0),
            "TiedShortOff": pct(T_["clifford_shortcut"]["frac_circuits_off_by_factor_two"], 0),
            "GraphRows": big(G_["rows_generated"]), "GraphHGB": f(G_["cv"]["Hist Gradient Boosting"][0]),
            "GraphMLP": f(G_["cv"]["MLP"][0]), "GraphRule": f(G_["cv"]["Descriptor rule (linear)"][0]),
            "GraphDesignOnly": f(G_["cv"]["HGB, design parameters only"][0]), "GraphRuleCoef": str(G_["rule_n_coef"]),
            "GraphExMLP": f(G_["extrap"]["MLP"]["mean"]), "GraphExHGB": f(G_["extrap"]["Hist Gradient Boosting"]["mean"]),
            "GraphExRule": f(G_["extrap"]["Descriptor rule (linear)"]["mean"]),
            "GraphConeMadRy": f(G_["cone_vs_structural"]["mad_fixed_ry"], 3),
            "GraphConeExactRy": pct(G_["cone_vs_structural"]["exact_frac_fixed_ry"], 0), "GraphNFeat": "19",
        })
        if "large" in T_:
            TL = T_["large"]
            M.update({"TiedLargeN": big(TL["rows_generated"]),
                      "TiedLargeMLP": "%.3f \\pm %.3f" % (TL["MLP"]["mean"], TL["MLP"]["sd"]),
                      "TiedLargeRule": f(TL["Structured linear"]["mean"]),
                      "TiedLargeHGB": f(TL["Hist Gradient Boosting"]["mean"]),
                      "TiedLargeBelowCut": pct(TL["below_cutoff_frac"], 0),
                      "TiedLargeInRange": pct(TL["frac_within_train_range"], 0),
                      "TiedLargeLabelMin": f(TL["label_min"], 2)})
    for k in ["TiedLargeN", "TiedLargeMLP", "TiedLargeRule", "TiedLargeHGB", "TiedLargeBelowCut", "TiedLargeLabelMin", "TiedLargeInRange"]:
        M.setdefault(k, "??")
    tune_path = os.path.join(RES, "tuning.json")
    if os.path.exists(tune_path):
        TU = json.load(open(tune_path))
        M.update({"TuneBest": f(TU["main"]["best_r2"]), "TuneSpread": f(TU["main"]["spread_learners"]),
                  "TuneGap": f(TU["main"]["gap_to_rule"]), "TuneHeadroom": f(TU["main"]["headroom"]),
                  "TuneBestTied": f(TU["tied"]["best_r2"]), "TuneGapTied": f(TU["tied"]["gap_to_rule"])})
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
            M["Lopo%sNoPhys" % key] = f(lo[name]["unseen_r2_no_physics"])
        M["LopoMinAuc"] = f(min(v["unseen_auc"] for v in lo.values()), 2)
    else:
        for k in ["LopoExtraRows", "LopoDescRsq", "LopoIndexRsq"] + [
                "Lopo%s%s" % (a, b) for a in ["Linear", "Circular", "All", "Brick", "Star"]
                for b in ["Unseen", "Seen", "Phys", "UnseenMae", "UnseenAuc", "NoPhys"]]:
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
              "ablation", "cutoff", "lopo", "gap", "screening_within", "large", "effweight",
              "clifford", "clifford_cost", "horizon", "families", "tuning", "large_short"]:
        tp = os.path.join(RES, "table_%s.tex" % t)
        body = open(tp).read().strip() if os.path.exists(tp) else "% (pending)"
        # never let a generated table exceed the text width
        tables[t] = "\\fittable{%\n" + body + "%\n}"

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
