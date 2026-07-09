# qml_bp — Barren-Plateau Trainability Dataset & Predictor

Predict a variational quantum circuit's **trainability** (the variance of its cost
gradient over random parameters — the barren-plateau signature) from its
**architecture alone**. The `qsim` statevector simulator generates the ground-truth
labels; a classical model learns to predict them.

- **Input (X):** circuit spec — qubits, layers, ansatz, entanglement pattern,
  entangler gate, cost locality, and derived counts (see `FEATURE_COLUMNS`).
- **Label (Y):** `log10 Var[dC/dtheta]` (regression) and `barren` vs `trainable`
  (classification, thresholded).

## Setup (macOS, Apple Silicon / M4)
```bash
cd quantum_computer_simulator
python3 -m venv .venv && source .venv/bin/activate
pip install -r qml_bp/requirements.txt   # numpy, pandas, scikit-learn
```
`qsim` is imported from the repo root, so run all commands from the repo root.

## 1. Quick sanity run (seconds)
```bash
python -m qml_bp.generate --n-specs 300 --samples 80 --out data_bp/sample.csv
python -m qml_bp.train    --data data_bp/sample.csv
```
Sanity check: **global-cost** circuits and **more qubits/layers** should show much
lower `grad_var` (barren); **local-cost, shallow** circuits should stay trainable.

## 2. Big dataset (overnight on an M4)
The dataset is tiny on disk (~a few MB for 100k rows); the cost is CPU time.
```bash
python -m qml_bp.generate \
  --n-specs 100000 --samples 200 \
  --qubit-min 2 --qubit-max 12 \
  --layer-min 1 --layer-max 20 \
  --workers 10 \
  --out data_bp/bp_dataset.csv
```
Tips for scale:
- **Cost ≈ n_specs × samples × 2 × (gates) statevector sims.** Raising `--samples`
  improves label quality (needed for tiny variances at high qubit counts) but costs
  linearly. 150–300 is a good range.
- **Qubit ceiling:** a statevector is `2^n` complex numbers. `n<=12` is fast and
  light; `n=16` (~1 MB/state) is fine on 16 GB; go higher only if you raise
  `--samples` too, since variances get tiny.
- `--workers` ≈ physical cores (M4: try 8–10). Rows stream to CSV as they finish,
  so you can stop and inspect partial output any time.

## 3. Train + evaluate
```bash
python -m qml_bp.train --data data_bp/bp_dataset.csv --barren-threshold -2
```
Reports:
- **Random split** R²/MAE (regression) and accuracy/AUC (classification).
- **Extrapolation split** — trains on small qubit counts, tests on larger unseen
  ones (the headline result for the paper).
- **Permutation feature importance** — which design choices drive trainability
  (expect cost locality + entanglement density to dominate).

## Notes for the paper
- Ground-truth labels are limited to *simulable* qubit counts; extrapolation claims
  must be scoped to that. This is the honest limitation to state.
- Everything is seeded (`--seed`) and reproducible.
- To scale beyond a laptop later, the generator is embarrassingly parallel and can
  be sharded across machines by splitting `--n-specs` with different `--seed`s.
