"""Standard single-qubit gate matrices (as 2x2 complex NumPy arrays)."""

import cmath
import math

import numpy as np

_ISQRT2 = 1.0 / math.sqrt(2.0)

I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = _ISQRT2 * np.array([[1, 1], [1, -1]], dtype=complex)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
SDG = np.array([[1, 0], [0, -1j]], dtype=complex)
T = np.array([[1, 0], [0, cmath.exp(1j * math.pi / 4)]], dtype=complex)
TDG = np.array([[1, 0], [0, cmath.exp(-1j * math.pi / 4)]], dtype=complex)


def rx(theta):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def ry(theta):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rz(theta):
    return np.array(
        [[cmath.exp(-1j * theta / 2), 0], [0, cmath.exp(1j * theta / 2)]],
        dtype=complex,
    )


def phase(lam):
    """Phase gate P(lambda) = diag(1, e^{i*lambda})."""
    return np.array([[1, 0], [0, cmath.exp(1j * lam)]], dtype=complex)
