# qml_bp — Barren-Plateau Trainability Dataset & Predictor

Predict a hardware-efficient variational circuit's **trainability** (the
variance of its cost gradient over random parameters) from its **architecture
alone**. The in-house `qsim` NumPy statevector simulator generates the
ground-truth labels; a classical model learns to predict them.

- **Input (X):** circuit spec — qubits, layers, rotation scheme, entanglement
  pattern, entangler gate, cost locality, and three derived counts
  (see `FEATURE_COLUMNS` in `ansatz.py`).
- **Label (y):** `log_var = log10` of the **mean gradient variance over all
  `nL` parameters**, excluding *structural zeros* (parameters whose gradient is
  identically zero by symmetry). Barren/trainable classification thresholds it
  at `-2`.

## What changed in version 2 (journal revision)

| | v1 (preprint) | v2 (revised manuscript) |
|---|---|---|
| Gradient | parameter shift, one fixed parameter | adjoint differentiation, **every** parameter |
| Label | `log10 max(Var, 1e-18)` of that one parameter | `log10` mean Var over non-structural parameters, no floor |
| Structural zeros | clipped to floor and called "barren" | detected (`max|g| < 1e-10`), counted, excluded |
| Evaluation | row-wise random split (leaks architectures) | architecture-grouped K-fold CV + controls |
| Files | `data_bp/bp_dataset.csv` | `data_bp/bp_dataset_v2.csv` (v1 columns kept for comparison) |

`v2` keeps the v1 columns (`grad_var`, `log_grad_var` = the single middle-layer,
qubit-0 parameter) so the two labels can be compared row by row.

## Setup
```bash
cd quantum_computer_simulator
python3 -m venv .venv && source .venv/bin/activate
pip install -r qml_bp/requirements.txt   # numpy, pandas, scikit-learn, matplotlib
```
`qsim` is imported from the repo root, so run all commands from the repo root.

## Reproduce the paper end to end
```bash
# 0. validate simulator + gradients (≈1 min)
python -m qml_bp.validate --n-circuits 200 --seed 1 --json paper/qmi/results/validation.json

# 1. dataset: 20,000 circuits x 200 samples, all-parameter labels (~1.5 h on 9 cores)
python -m qml_bp.generate --n-specs 20000 --samples 200 \
    --qubit-min 2 --qubit-max 12 --layer-min 1 --layer-max 20 \
    --workers 9 --seed 12345 --label-mode all --out data_bp/bp_dataset_v2.csv

# 2. label repeatability (sampling uncertainty of the label)
python -m qml_bp.repeat_labels --n-specs 200 --samples 200 --workers 9 \
    --json paper/qmi/results/repeat.json

# 3. every number, table and figure in the paper
python -m qml_bp.analyze --data data_bp/bp_dataset_v2.csv \
    --outdir paper/qmi/results --figdir paper/qmi/figs

# 3b. unseen-pattern, 13-14-qubit and Clifford-sampling experiments
python -m qml_bp.generate --n-specs 10000 --patterns brickwork,star --workers 9 --seed 54321 --out data_bp/bp_patterns_v2.csv
python -m qml_bp.lopo --main data_bp/bp_dataset_v2.csv --extra data_bp/bp_patterns_v2.csv
QSIM_MAX_QUBITS=14 python -m qml_bp.generate --n-specs 2000 --qubit-min 13 --qubit-max 14 --workers 7 --seed 777 --out data_bp/bp_large_v2.csv
QSIM_MAX_QUBITS=14 python -m qml_bp.eval_large --train data_bp/bp_dataset_v2.csv --test data_bp/bp_large_v2.csv
python -m qml_bp.extra_experiments --data data_bp/bp_dataset_v2.csv
python -m qml_bp.validate_clifford --n-circuits 200 --workers 9 --json paper/qmi/results/validation_clifford.json
python -m qml_bp.generate_clifford --n-specs 8000 --qubit-min 2 --qubit-max 12 --workers 9 --seed 2026 --out data_bp/bp_clifford_train.csv
python -m qml_bp.generate_clifford --n-specs 6000 --qubit-min 13 --qubit-max 32 --workers 9 --seed 2027 --out data_bp/bp_clifford_test.csv
python -m qml_bp.clifford_study --train data_bp/bp_clifford_train.csv --test data_bp/bp_clifford_test.csv
# two further families and the model-robustness study
python -m qml_bp.generate_families --family tied  --n-specs 10000 --seed 31337 --workers 9 --out data_bp/bp_tied_v2.csv
python -m qml_bp.generate_families --family graph --n-specs 10000 --seed 27182 --workers 9 --out data_bp/bp_graph_v2.csv
QSIM_MAX_QUBITS=14 python -m qml_bp.generate_families --family tied --n-specs 1000 --qubit-min 13 --qubit-max 14 --seed 4242 --workers 6 --out data_bp/bp_tied_large_v2.csv
python -m qml_bp.families_study --main data_bp/bp_dataset_v2.csv --tied data_bp/bp_tied_v2.csv --graph data_bp/bp_graph_v2.csv --tied-large data_bp/bp_tied_large_v2.csv
python -m qml_bp.tuning_study --main data_bp/bp_dataset_v2.csv --tied data_bp/bp_tied_v2.csv
# optimization check quoted in the limitations (four budgets)
python -m qml_bp.training_outcome --data data_bp/bp_dataset_v2.csv --n-circuits 300 --optimizer adam --shots 1000 --workers 9
python -m qml_bp.training_outcome --data data_bp/bp_dataset_v2.csv --n-circuits 200 --optimizer gd --lr 0.5 --shots 0 --tag _gd_exact --workers 9
python -m qml_bp.training_outcome --data data_bp/bp_dataset_v2.csv --n-circuits 200 --optimizer gd --lr 0.5 --shots 100 --tag _gd_100shots --workers 9
python -m qml_bp.training_outcome --data data_bp/bp_dataset_v2.csv --n-circuits 200 --optimizer adam --shots 100 --tag _adam_100shots --workers 9
python -m qml_bp.training_sweep_table

# 4. inline the numbers into the single-file manuscript and the response letter
python paper/qmi/fill_numbers.py
cd paper/qmi && tectonic main.tex && tectonic response_to_reviewers.tex
```

## Modules
- `ansatz.py` — `CircuitSpec` (the circuit family), feature extraction,
  `compute_datapoint_all` (v2 label) and `compute_datapoint` (v1 label).
- `adjoint.py` — exact reverse-mode gradient of a Pauli-Z-string cost with
  respect to every rotation angle, ≈3 forward passes per parameter vector.
- `generate.py` — parallel dataset generator (multiprocessing; rows are written as
  they finish; the output file is overwritten at start, runs are not resumable).
  `--label-mode all` (default) or `legacy`.
- `validate.py` — statevector vs dense-matrix reference; adjoint vs
  parameter-shift; structural zeros are sample-independent.
- `repeat_labels.py` — relabel fresh circuits twice; reports label noise.
- `analyze.py` — architecture-grouped CV, controls (training mean,
  cost-locality subgroup mean, physics-informed linear model), extrapolation
  split with bootstrap CIs and per-subset breakdown, cutoff sensitivity,
  feature-group ablation, figures and LaTeX table fragments.
- `clifford.py` — exact gradient variance at any qubit count by Clifford
  (Pauli-propagation) sampling; `validate_clifford.py` checks it against
  adjoint gradients; `generate_clifford.py` builds datasets with it;
  `clifford_study.py` is the 13-32-qubit extrapolation study.
- `families.py`, `generate_families.py`, `families_study.py` — circuits with
  layer-shared parameters and with random entangling graphs; `tuning_study.py`
  — robustness of the results to model choice and tuning.
- `lopo.py` (unseen entanglement patterns), `eval_large.py` (13-14-qubit
  statevector test set), `extra_experiments.py` (widening gap, learning curve,
  ranking at fixed n), `training_outcome.py` and `training_sweep_table.py`
  (the optimization check reported as a limitation in the paper).
- `train.py`, `compare_models.py`, `describe_data.py` — the v1 scripts
  (row-wise split, permutation importance). Kept for the record; they still run
  on either CSV but are no longer used for the paper.

## Conventions
- Qubit 0 is the most significant bit of the basis-state index.
- Entanglement patterns per layer: linear `(q, q+1)` for `q=0..n-2`; circular
  adds `(n-1, 0)`; all-to-all every `(a, b)` with `a<b`. First index is the CX
  control. Gates are applied in list order after the rotation layer.
- Random-Pauli axes are drawn once per spec, per `(layer, qubit)`, from the
  spec's seed; they are fixed across the 200 parameter samples.
- Variance is the population variance (`ddof=0`) over the samples.
- Closed-form physics features (computed on load for older CSVs):
  `cost_weight_eff` = weight of the observable after the last entangling layer
  (CX conjugation); `cone_qubits`, `cone_frac` = backward causal cone of the
  observable (per-qubit Pauli-type relaxation).
- Seeds: root `12345` for generation (`SeedSequence.spawn`), `0` for models.
