# QMI revision: task list (deadline ~2026-10-10)

Status legend: [x] done · [~] in progress · [ ] pending

## Round 1: baseline revision and experiments (complete, tag v2.0-qmi-revision)
- [x] Corrected label, 20k dataset, validation, grouped CV, controls, ablation
- [x] Unseen-pattern transfer (brickwork, star), widening-gap extrapolation, learning curve
- [x] Held-out 13- and 14-qubit statevector test set
- [x] Training-outcome experiment: run, uninformative at n<=12, held out of the paper
- [x] Supplement, cover letter, response letter with label-driven references

## Round 2: internal third-referee review (2026-09-30), all claims verified
### K. Clifford-sampling experiment (labels beyond statevector reach)
- [x] K1. Estimator `qml_bp/clifford.py` (bit-packed Pauli propagation, all parameters per sweep)
- [x] K2. Validation against exact gradients: 200 circuits, 15,182 parameters; label MAD 0.009, max 0.12; 0 of 2,804 structural zeros had Clifford hits; same-architecture agreement with the statevector dataset: mean +0.002
- [x] K3. Datasets: 8,000 training circuits n=2..12 (99.3% resolved); 6,000 test circuits n=13..32 (73% resolved; censoring reported)
- [x] K4. Extrapolation study: from n<=12, reliable to ~20 qubits (structured linear R^2 0.92/0.90 at 13-16/17-20), degrades at 21-24, fails beyond; trained on n<=20 predicts 21-32 with R^2 0.90-0.94
- [x] K5. INCLUDED as Sec. 4.3 (a horizon result, honestly framed) + Methods 3.6 + validation item (iv)

### L. Review concerns
- [x] L1. Effective observable weight (CX conjugation) as a closed-form feature; physics insight paragraph
- [x] L2. Stronger control: structured linear model (per-cell slopes); reframe "beyond known rules"
- [x] L3. Screening: replace the global table by within-n screening; fix the wrong "discards no good architecture" sentence
- [x] L4. MLP extrapolation averaged over 10 seeds everywhere; soften "degrades gracefully"
- [x] L5. Noise ceiling with unbiased within-group variance (0.985)
- [x] L6. Related work: efficient gradient-variance methods (Letcher et al. 2024; Uvarov & Biamonte 2021; Napp 2022); discuss what Clifford sampling means for the cost motivation
- [x] L7. Text: move revision history out of the body; ablation wording; "seven qubits"; "statistically indistinguishable"; abstract to journal length; unused bib entry; strip unused macros; table headers "above/below cutoff"
- [ ] L8. AUTHOR DECISION PENDING: one-paragraph limitation reporting that gradient variance did not predict optimization outcome at n<=12 (drafted in limitation_training_outcome.tex.hold, not inserted)

### M. Packaging
- [x] M1. Response letter and cover letter updated to the changed claims
- [x] M2. Refilled, recompiled (main 29 pp, supplement 3, response 9, cover 2), references checked, committed, retagged v2.1-qmi-revision
- [x] N. Causal-cone feature (closed-form, validated against exact structural-zero fraction) added after the Clifford study exposed cone saturation; transfer to unseen patterns improved from 0.41-0.74 to 0.75-0.95
