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
- [x] E. Held-out 13- and 14-qubit test set (1,991 circuits): trained on n<=12, MLP R^2 0.964 (0.966 at 13, 0.962 at 14), HGB 0.906, physics-linear 0.350. INCLUDED (Sec. 4.2 paragraph, Table large, Fig. large; abstract and conclusion updated)
- [x] F. Training-outcome validation: DONE and HELD OUT. Four budgets tried (GD exact, GD 100 shots, Adam 1000 shots, Adam 100 shots): 94-99% of circuits reach the ground state within 200 steps regardless of label; Spearman(predicted label, loss decrease) 0.08-0.17. At n<=12 the Z-string cost is easy to minimize, so the label does not predict optimization outcome in this regime. Not in the paper (paper claims gradient-variance prediction only). Code kept (qml_bp/training_outcome.py); outputs gitignored.

## Packaging
- [~] G. Supplementary material (supplement.tex, 3 pp): hyperparameters, descriptive stats, cutoff, full extrapolation table, gap table, learning curve moved; training-outcome and 13-14-qubit tables to be added when E/F finish
- [x] H. Cover letter to the editor (cover_letter.tex, filled from results, gitignored)
- [~] I. Git: branch `qmi-revision`, checkpoint commits made; release tag after final commit; Zenodo checklist written (ZENODO_CHECKLIST.md; upload needs the author's account)
- [~] J. Final read-through: response letter now uses \ref via xr (check_refs.py); final pass after E/F land
