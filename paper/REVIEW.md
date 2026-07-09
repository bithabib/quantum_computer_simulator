# Paper Review & Fix Checklist

> Review of `paper/mdpi/main.tex` (MDPI/Entropy build) and shared tables/refs.
> Work through these on the lab machine, then rebuild with `./build.sh`.
> Most items apply equally to `paper/ieee-access/` since it shares text/tables.
>
> **Overall verdict:** scientifically sound, honestly framed, numbers verified
> consistent across macros/tables/prose. Submittable after the fixes below.
> Priority: do **A** and **C9–C10** before submitting; **B**/**D** are polish.

---

## 0. Figures (you said you'll do these on the lab machine)
- [ ] Figures don't render: `main.tex` includes `figs/*.pdf` (guarded by
      `\IfFileExists`, so they silently vanish). No `figs/` dir; `build.sh` doesn't
      make one; `make_figures.py` (in `paper/latex/`) writes to `paper/latex/figs/`
      and only makes 3 of 4 (no `pipeline.pdf`); `figs/*.pdf` is git-ignored.
- [ ] Two are fully guarded, so their `\ref` breaks to "??": `fig:predicted`
      (main.tex ~L373) and `fig:importance` (~L407).
- [ ] Fix option (robust): point the 4 `\includegraphics` at the committed PNGs
      `../../static/research/{pipeline,predicted,qubit_trend,importance}.png` and
      drop the `\IfFileExists` guards. Or: generate `figs/*.pdf` into each paper dir
      in `build.sh` (and add a `pipeline.pdf`).

## A. Real errors / inconsistencies (fix before submitting)
- [ ] **A1 — Orphaned duplicate results table.** `main.tex` hard-codes the model
      table inline (Sec. 4.1) instead of `\input{table_models}`; the auto-generated
      `table_models.tex` is never used → will drift if results are regenerated.
      Fix: replace the inline `tabularx` with `\input{table_models}` (move the
      bold-best styling into `qml_bp/compare_models.py`).
- [ ] **A2 — Self-contradicting linear-baseline sentence** (~L339):
      "linear baseline is far behind (regression R²=0.27; classifier AUC 0.95)."
      AUC 0.95 is NOT far behind. Reword so "far behind"/nonlinearity rests on
      regression, e.g.: *"far behind on regression (R²=0.27 vs 0.69); classification
      is easy for all models (AUC ≥ 0.95), so the architecture→trainability map is
      nonlinear mainly in its fine-grained (regression) structure."*
- [ ] **A3 — Extrapolation oversold.** Split is train n≤10 → test n=11,12 (+2
      qubits, both simulable). Soften "progressively harder to simulate" /
      "transferable across scale" → "transfers to the two largest, unseen sizes,"
      and state the gap explicitly. (Abstract ~L78 and Sec. 4.3.)
- [ ] **A4 — Explain why extrapolation R² (0.719) > hold-out R² (0.685).** Add one
      sentence: the n=11–12 test set is dominated by saturated global-cost labels
      that are easy to predict, so higher R² there ≠ an easier task in general.
- [ ] **A5 — `\conflictsofinterest{The author declares…}` → "authors"** (two
      authors). main.tex ~L470.

## B. Copyediting / consistency
- [ ] **B6 — Unify British vs American spelling.** You mix British
      *emphasise, de-emphasise, labelled, labelling* with American
      *generalize, visualization, standardized, summarizes*. MDPI defaults to
      American: emphasize, de-emphasize, labeled, labeling.
- [ ] **B7 — `eshyana2026` defined in refs.bib but never cited.** Cite it (Methods,
      where `qsim` is mentioned, or Data Availability) or remove it.
- [ ] **B8 — Bib nits:** add DOI to `cerezo2025simulability` (arXiv note only);
      change `ke2017lightgbm` to `@inproceedings` (NeurIPS is a conference); note
      that `HistGradientBoosting` is scikit-learn's own impl *inspired by* LightGBM.

## C. MDPI submission requirements (editors bounce papers missing these)
- [ ] **C9 — Add the required N/A statements:**
      `\institutionalreview{Not applicable.}` and
      `\informedconsent{Not applicable.}`
- [ ] **C10 — Data Availability: add the real code repo** (GitHub URL) and ideally a
      tagged release / Zenodo DOI. Biggest credibility win for the "reproducible
      pipeline" claim. Currently only the simulator URL is given (main.tex ~L464).
- [ ] **C11 — Fill the real ORCID** (placeholder `0000-0000-0000-0000`, L50).

## D. Reviewer-proofing (optional, strengthens acceptance)
- [ ] **D12 — Single fixed gradient parameter.** You measure variance w.r.t. one
      middle-layer rotation; note as a limitation (variance can depend on which
      parameter). One line in Discussion.
- [ ] **D13 — Arbitrary barren threshold (log₁₀Var < −2).** Add a one-line
      sensitivity note, or a small appendix sweep at −1.5 / −2.5.
- [ ] **D14 — Soften the absolute novelty claim** ("to our knowledge…has not been
      reported") → "we are not aware of…", so a missed citation isn't fatal.

## Consistency checks already PASSED (no action needed)
- Numbers agree across macros ↔ tables ↔ prose (R²=0.685, AUC=0.991, extrap 0.719,
  barren 46.7%, global 50.1%, train/test 16,395/3,605).
- Label-by-design recovers theory (local −2.33 vs global −4.49; matches our early
  validation run).
- All `\cite` keys resolve to `refs.bib` entries; references are accurate
  (venues/years/DOIs spot-checked).

## Process note
- You have 3 builds: `paper/latex/`, `paper/ieee-access/`, `paper/mdpi/`. Pick ONE
  target venue; keep the others only if you'll maintain them (tables/text drift).
  Header says MDPI **Entropy**, a good fit.
- Repo hygiene (not paper, but noticed): stray `template.tex` / `template.pdf` and
  an `ACCESS_latex_template_20260513/` dump at the repo root — consider removing or
  moving under `paper/`. Also still-open security items (committed `.pem`, weak
  `secret_key`) from earlier.
