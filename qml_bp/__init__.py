"""qml_bp - barren-plateau trainability dataset generation and modelling.

Generates a labelled dataset that maps a variational quantum circuit's
*architecture* to its *trainability* (the variance of the cost gradient over
random parameters), using the project's qsim statevector simulator to compute
the ground-truth labels. A classical ML model is then trained to predict
trainability from the architecture alone.

See qml_bp/README.md for how to generate a large dataset and train on it.
"""

from .ansatz import CircuitSpec, sample_spec, compute_datapoint, FEATURE_COLUMNS

__all__ = ["CircuitSpec", "sample_spec", "compute_datapoint", "FEATURE_COLUMNS"]
