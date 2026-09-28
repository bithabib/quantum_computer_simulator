# Quantum Machine Intelligence manuscript (revised)

Single-file Springer `sn-jnl` manuscript plus the point-by-point response.

- `main.tex` — the manuscript. The `%%BEGIN-NUMBERS` block and every
  `%%BEGIN-TABLE:<name>` block are generated; do not edit them by hand.
- `response_to_reviewers.tex` — response letter (same generated macros).
- `fill_numbers.py` — reads `results/*.json` and `results/table_*.tex`
  (written by `python -m qml_bp.analyze`, `qml_bp.validate --json`,
  `qml_bp.repeat_labels --json`) and rewrites the generated blocks.
- `fig_pipeline.tex` — standalone TikZ source of `figs/pipeline.pdf`.
- `figs/` — all figures (`predicted`, `qubit_trend`, `ablation`,
  `structural`, `label_compare` come from `qml_bp.analyze`).
- `reviews/` — the referee reports as received.
- `refs.bib`, `sn-jnl.cls`, `sn-mathphys-num.bst` — bibliography and class.

Build (after the analysis has been run; see `qml_bp/README.md`):
```bash
python paper/qmi/fill_numbers.py
cd paper/qmi
tectonic fig_pipeline.tex && mv fig_pipeline.pdf figs/pipeline.pdf
tectonic --keep-intermediates main.tex      # produces main.pdf and main.bbl
tectonic response_to_reviewers.tex
```
