# Zenodo and repository release checklist (author action required)

The manuscript's data availability statement cites
https://doi.org/10.5281/zenodo.21446194 and the GitHub repository. The Zenodo
record currently holds the version-1 dataset (single-parameter label). Before
resubmission the record must be updated so the DOI resolves to the data that
the revised paper actually uses.

## 1. Files to upload as a NEW VERSION of the Zenodo record
(Zenodo → the record → "New version". Keep the concept DOI; a new version DOI
is minted. Quote the new version DOI in the paper if you prefer it to the
concept DOI.)

| File | Rows | What it is |
|---|---|---|
| `data_bp/bp_dataset_v2.csv` | 20,000 | main dataset, all-parameter label (`log_var`), legacy label kept (`log_grad_var`) |
| `data_bp/bp_patterns_v2.csv` | 10,000 | brickwork + star circuits for the unseen-pattern experiment |
| `data_bp/bp_large_v2.csv` | ~2,000 | 13- and 14-qubit held-out test set |
| `data_bp/bp_dataset.csv` | 20,000 | version-1 dataset (keep for the comparison in Sec. 3.2) |
| `paper/qmi/results/*.json`, `training_outcome*.csv` | | every number in the paper |
| `CITATION.cff` or a README | | column descriptions (copy from `qml_bp/README.md` "Conventions") |

Suggested version description: "Version 2: labels recomputed as the mean
gradient variance over all parameters with structural zeros excluded (adjoint
differentiation); adds unseen-pattern and 13–14-qubit sets. Version-1 file
retained for comparison."

## 2. Repository
- Commit the revision on a branch (`qmi-revision`), merge to `main`, then tag:
  `git tag -a v2.0-qmi-revision -m "QMI major revision"` and `git push --tags`.
- Do NOT push: `paper/qmi/reviews/`, `paper/qmi/response_to_reviewers.*`,
  `paper/qmi/cover_letter.*`, the preprint PDF (all gitignored).
- Update the Data availability statement in `main.tex` with the tag name
  (search for "root seed 12345").

## 3. Journal submission package
- `main.tex` (single file), `main.bbl`, `sn-jnl.cls`, `sn-mathphys-num.bst`, `refs.bib`
- `figs/*.pdf`
- `supplement.pdf` (if the supplement is used)
- `response_to_reviewers.pdf`, `cover_letter.pdf`
- The system may ask for a marked-up version; `latexdiff` between the
  submitted v1 `main.tex` (git history, commit 1064857) and the new one
  produces it: `latexdiff old.tex main.tex > diff.tex`.
