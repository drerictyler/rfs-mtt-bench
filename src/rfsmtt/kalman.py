"""Vectorised Kalman filter primitives shared by every tracker in the package.

All functions operate on stacks of Gaussian components so that a mixture or a
track table can be predicted and updated in one call:

* ``means`` has shape ``(n, dx)``
* ``covs`` has shape ``(n, dx, dx)``
* measurements ``Z`` have shape ``(m, dz)``
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2

_LOG2PI = float(np.log(2.0 * np.pi))


def symmetrize(A: np.ndarray) -> np.ndarray:
    """Return the symmetric part of a (stack of) square matrices."""
    return 0.5 * (A + np.swapaxes(A, -1, -2))


def gate_threshold(probability: float, dz: int) -> float:
    """Squared Mahalanobis distance that contains ``probability`` of the innovation mass."""
    return float(chi2.ppf(probability, dz))


def predict(means, covs, F: np.ndarray, Q: np.ndarray):
    """Kalman prediction for a stack of Gaussian components."""
    dx = F.shape[0]
    means = np.asarray(means, dtype=float).reshape(-1, dx)
    covs = np.asarray(covs, dtype=float).reshape(-1, dx, dx)
    return means @ F.T, symmetrize(F @ covs @ F.T + Q)


def update(means, covs, H: np.ndarray, R: np.ndarray, Z, gate: float | None = None):
    """Kalman update of every component with every measurement.

    Returns
    -------
    q : (n, m) array
        Measurement likelihoods N(z; H m, S). Entries outside the gate are zero.
    mu : (n, m, dx) array
        Updated means, one per (component, measurement) pair.
    Pu : (n, dx, dx) array
        Updated covariances (independent of the measurement value).
    d2 : (n, m) array
        Squared Mahalanobis distances of the innovations.
    """
    dz, dx = H.shape
    means = np.asarray(means, dtype=float).reshape(-1, dx)
    covs = np.asarray(covs, dtype=float).reshape(-1, dx, dx)
    Z = np.asarray(Z, dtype=float).reshape(-1, dz)

    eta = means @ H.T
    S = symmetrize(H @ covs @ H.T + R)
    S_inv = np.linalg.inv(S)
    _, logdet = np.linalg.slogdet(S)
    K = covs @ H.T @ S_inv

    # Joseph form keeps the covariance positive definite under round-off.
    IKH = np.eye(dx) - K @ H
    Pu = symmetrize(IKH @ covs @ np.swapaxes(IKH, -1, -2) + K @ R @ np.swapaxes(K, -1, -2))

    nu = Z[None, :, :] - eta[:, None, :]
    d2 = np.einsum("nmi,nij,nmj->nm", nu, S_inv, nu)
    q = np.exp(-0.5 * (d2 + logdet[:, None] + dz * _LOG2PI))
    if gate is not None:
        q = np.where(d2 <= gate, q, 0.0)
    mu = means[:, None, :] + np.einsum("nij,nmj->nmi", K, nu)
    return q, mu, Pu, d2
