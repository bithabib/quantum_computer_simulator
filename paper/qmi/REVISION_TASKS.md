# QMI revision: task list (deadline ~2026-10-10)

Status legend: [x] done · [~] in progress · [ ] pending

## Baseline revision (submittable version)
- [x] Corrected label (all-parameter adjoint gradients, structural zeros removed), 20k dataset regenerated
- [x] Validation (dense reference, parameter shift, structural-zero stability) and label repeatability
- [x] Architecture-grouped CV, controls, noise ceiling, extrapolation breakdown (HGB + MLP), cutoff sensitivity, feature ablation
- [x] Experiment 1: two new patterns (brickwork, star), graph descriptors, leave-one-pattern-out
- [x] A. Fill transfer numbers, write outcome sentences, compile manuscript + response letter (main.pdf 23 pp, response 9 pp)

## Additional experiments (in order)
- [x] B. Screening precision (Sec. 4.8, Table 13): Spearman 0.988; top 20% contains zero barren circuits
- [x] C. Wider extrapolation gap (Sec. 4.2, Fig. gap, Table gap): MLP 0.95 -> 0.67 as k goes 10 -> 6; HGB collapses
- [x] D. Learning curve (Sec. 4.7): R^2 > 0.9 by ~1,000-2,000 rows, saturates by ~4,000
- [~] E. Held-out 13- and 14-qubit test set: generating (qsim cap lifted via QSIM_MAX_QUBITS); then evaluate models trained on n<=12
- [ ] F. Training-outcome validation: gradient-descent runs on a few hundred circuits; loss decrease vs predicted label

## Packaging
- [~] G. Supplementary material (supplement.tex, 3 pp): hyperparameters, descriptive stats, cutoff, full extrapolation table, gap table, learning curve moved; training-outcome and 13-14-qubit tables to be added when E/F finish
- [x] H. Cover letter to the editor (cover_letter.tex, filled from results, gitignored)
- [ ] I. Git branch + commit of the revision; release tag; Zenodo v2 upload checklist (upload itself needs the author's account)
- [ ] J. Final read-through: numbering, consistency, page count, abstract, response letter cross-references
