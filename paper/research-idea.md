# Research Idea (finalized)

**Title (working):** *Predicting the Trainability of Variational Quantum Circuits:
A Data-Driven Model for Barren Plateaus*

**Target:** MDPI Quantum Reports (open access). Template: MDPI `quantumrep` class.

> The browser/simulator is just the data-generation tool, NOT the contribution.
> The contribution is the **prediction task**: a classical ML model that predicts
> a quantum circuit's trainability from its architecture, trained on data the
> `qsim` simulator generates.

## The output (core of the paper)
- **Input X** = a variational quantum circuit "spec": n_qubits, depth/#layers,
  ansatz type, entanglement pattern (linear/circular/all-to-all), #entangling
  gates, rotation-gate mix, and **cost/observable locality (local vs global)**.
- **Output Y** = the circuit's **trainability**, as either
  - regression: `log10 Var[∂C/∂θ]` (gradient-variance magnitude), or
  - classification: `trainable` vs `barren`.

## Why it matters
Barren plateaus (gradients vanishing exponentially with qubit count) are the
central obstacle to scaling QML (Larocca et al., *Nat. Rev. Phys.* 2025). Today
they're diagnosed by expensive gradient sampling. We **predict** them from the
design alone — a cheap ansatz-screening tool + insight into which design choices
cause them.

## Data generation (via qsim)
For each of many random circuit specs: build the circuit, sample many random
parameter vectors, compute gradients (parameter-shift), estimate `Var[∂C/∂θ]`
→ that variance is the label. Sweep n_qubits (small, simulable range), depth,
ansatz, entanglement pattern, and cost locality.

## Model
- Baseline: gradient-boosted trees / random forest on the tabular spec features.
- Ablation / stretch: represent the circuit as a graph or gate sequence and use a
  GNN / small transformer.

## Experiments
1. **Generalization:** train/test split over held-out architectures. Report R²/MAE
   (regression) or accuracy/AUC (classification).
2. **Extrapolation (headline):** train on small n_qubits, test on larger n_qubits
   too costly to simulate directly — does the predictor still work?
3. **Baselines:** known theoretical scaling (global cost → exponential decay),
   simple heuristics.
4. **Interpretability:** feature importance — which design knobs drive trainability
   (expected: cost locality and entanglement density dominate).

## Novelty & honest framing
- Fresh **data-driven prediction** angle on the field's #1 open problem (vs. yet
  another theoretical bound).
- **No quantum-advantage claim** — this is classical ML *about* quantum circuits,
  sidestepping the skeptical benchmarking literature (Bowles/Schuld 2024; Cerezo
  *Nat. Commun.* 2025).
- **Limitation:** ground-truth labels limited to simulable qubit counts;
  extrapolation validated carefully and scoped honestly.

## Key references (from the literature review)
- McClean et al. 2018 (original barren plateaus) — add to bib.
- Cerezo et al. 2021, *Cost-function-dependent barren plateaus* (local vs global cost).
- Larocca et al., *Barren plateaus in variational quantum computing*, Nat. Rev. Phys. 7:174 (2025).
- Cerezo et al., *Nat. Commun.* 16:7907 (2025) — BP ↔ classical simulability.
- (Foundational VQC/QNN: Havlíček 2019; Farhi & Neven 2018; Schuld et al. 2020.)

See `paper/literature-review.md` for the full survey and `paper/outline.md` for the
section structure (to be updated to this framing).
