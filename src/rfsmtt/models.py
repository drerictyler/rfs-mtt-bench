"""Motion, measurement, clutter and birth models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import block_diag

from .kalman import predict


@dataclass
class LinearGaussianModel:
    """Linear-Gaussian single-object model with constant survival and detection probabilities."""

    F: np.ndarray
    Q: np.ndarray
    H: np.ndarray
    R: np.ndarray
    p_s: float = 0.99
    p_d: float = 0.98
    pos_idx: tuple = (0, 2)

    @property
    def dx(self) -> int:
        return self.F.shape[0]

    @property
    def dz(self) -> int:
        return self.H.shape[0]

    def positions(self, states: np.ndarray) -> np.ndarray:
        """Position sub-vector of each state, shape (n, len(pos_idx))."""
        return np.asarray(states, dtype=float).reshape(-1, self.dx)[:, list(self.pos_idx)]


def constant_velocity(
    ndim: int = 2,
    dt: float = 1.0,
    q: float = 1.0,
    sigma_pos: float = 10.0,
    sigma_vel: float | None = None,
    p_s: float = 0.99,
    p_d: float = 0.98,
) -> LinearGaussianModel:
    """Nearly-constant-velocity model with state ``[x, vx, y, vy, ...]``.

    ``q`` is the power spectral density of the white-noise acceleration (m^2/s^3).
    Positions are always measured; velocities are measured too when ``sigma_vel`` is given
    (e.g. range and range-rate from a Doppler radar).
    """
    F1 = np.array([[1.0, dt], [0.0, 1.0]])
    Q1 = q * np.array([[dt**3 / 3.0, dt**2 / 2.0], [dt**2 / 2.0, dt]])
    F = block_diag(*([F1] * ndim))
    Q = block_diag(*([Q1] * ndim))
    if sigma_vel is None:
        H = np.kron(np.eye(ndim), np.array([[1.0, 0.0]]))
        R = sigma_pos**2 * np.eye(ndim)
    else:
        H = np.eye(2 * ndim)
        R = np.diag(np.tile([sigma_pos**2, sigma_vel**2], ndim))
    return LinearGaussianModel(F, Q, H, R, p_s=p_s, p_d=p_d, pos_idx=tuple(range(0, 2 * ndim, 2)))


@dataclass
class PoissonClutter:
    """Poisson clutter, uniform over an axis-aligned box in measurement space."""

    rate: float
    bounds: np.ndarray  # shape (dz, 2): [low, high] per measurement dimension

    def __post_init__(self):
        self.bounds = np.asarray(self.bounds, dtype=float)

    @property
    def volume(self) -> float:
        return float(np.prod(self.bounds[:, 1] - self.bounds[:, 0]))

    @property
    def density(self) -> float:
        """Spatial density of a single clutter point."""
        return 1.0 / self.volume

    @property
    def intensity(self) -> float:
        """Clutter intensity kappa(z) = rate / volume."""
        return self.rate / self.volume

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        n = rng.poisson(self.rate)
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        return lo + (hi - lo) * rng.random((n, len(lo)))


def inverse_measurement(model: LinearGaussianModel, Z: np.ndarray, unmeasured_std):
    """Gaussian state densities implied by single measurements.

    Measured directions take the measurement value and noise; unmeasured directions
    (e.g. velocity when only position is observed) get zero mean and ``unmeasured_std``.
    """
    Z = np.asarray(Z, dtype=float).reshape(-1, model.dz)
    H_pinv = np.linalg.pinv(model.H)
    N = np.eye(model.dx) - H_pinv @ model.H
    prior = np.diag(np.broadcast_to(np.asarray(unmeasured_std, dtype=float), (model.dx,)) ** 2)
    P0 = H_pinv @ model.R @ H_pinv.T + N @ prior @ N.T
    means = Z @ H_pinv.T
    covs = np.repeat(P0[None], len(Z), axis=0)
    return means, covs


@dataclass
class GaussianBirth:
    """Fixed birth model: a Gaussian per birth location with a birth probability (or PHD weight)."""

    weights: np.ndarray
    means: np.ndarray
    covs: np.ndarray

    def __post_init__(self):
        self.weights = np.asarray(self.weights, dtype=float).reshape(-1)
        self.means = np.asarray(self.means, dtype=float).reshape(len(self.weights), -1)
        dx = self.means.shape[1]
        self.covs = np.asarray(self.covs, dtype=float).reshape(len(self.weights), dx, dx)

    def components(self, k, Z_prev=None, r_used=None):
        return self.weights.copy(), self.means.copy(), self.covs.copy()


@dataclass
class AdaptiveBirth:
    """Measurement-driven birth (Ristic et al. 2012; Reuter et al. 2014).

    Objects are born at step k near measurements of step k-1 that existing tracks did not
    explain. ``r_used`` is the probability that each previous measurement was associated with
    a track; the birth probability of measurement z is
    ``min(r_max, (1 - r_used(z)) / sum(1 - r_used) * expected_births)``.
    """

    model: LinearGaussianModel
    expected_births: float = 0.1
    r_max: float = 0.05
    unmeasured_std: object = 50.0
    max_components: int = 100
    min_weight: float = 1e-5

    def components(self, k, Z_prev, r_used):
        dx = self.model.dx
        empty = (np.zeros(0), np.zeros((0, dx)), np.zeros((0, dx, dx)))
        Z_prev = np.asarray(Z_prev, dtype=float).reshape(-1, self.model.dz)
        if len(Z_prev) == 0:
            return empty
        r_used = np.zeros(len(Z_prev)) if r_used is None else np.clip(np.asarray(r_used), 0.0, 1.0)
        free = 1.0 - r_used
        if free.sum() <= 0:
            return empty
        r = np.minimum(self.r_max, free / free.sum() * self.expected_births)
        keep = np.flatnonzero(r > self.min_weight)
        keep = keep[np.argsort(-r[keep])][: self.max_components]
        if len(keep) == 0:
            return empty
        m0, P0 = inverse_measurement(self.model, Z_prev[keep], self.unmeasured_std)
        m, P = predict(m0, P0, self.model.F, self.model.Q)
        return r[keep], m, P


@dataclass
class Estimate:
    """Output of one tracker step."""

    states: np.ndarray
    labels: list
    cardinality_pmf: np.ndarray | None = None

    def __len__(self):
        return len(self.labels)


def unique_labels(tags) -> list:
    """Make a list of tags unique by suffixing repeats, preserving order."""
    seen: dict = {}
    out = []
    for t in tags:
        t = int(t) if isinstance(t, (np.integer, int)) else t
        n = seen.get(t, 0)
        out.append(t if n == 0 else (t, n))
        seen[t] = n + 1
    return out
