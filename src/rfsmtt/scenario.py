"""Simulated multi-object scenarios: ground truth and cluttered measurements."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .models import GaussianBirth, LinearGaussianModel, PoissonClutter, constant_velocity


@dataclass
class Scenario:
    model: LinearGaussianModel
    clutter: PoissonClutter
    n_scans: int
    truth: dict = field(default_factory=dict)  # label -> (n_scans, dx) array, NaN when absent
    birth: object = None

    def truth_at(self, k: int):
        labels, states = [], []
        for lab, traj in self.truth.items():
            if np.isfinite(traj[k, 0]):
                labels.append(lab)
                states.append(traj[k])
        return labels, np.array(states).reshape(-1, self.model.dx)

    def cardinality(self) -> np.ndarray:
        return np.array([len(self.truth_at(k)[0]) for k in range(self.n_scans)])

    def true_position_tracks(self) -> dict:
        idx = list(self.model.pos_idx)
        return {lab: traj[:, idx] for lab, traj in self.truth.items()}

    def measurements(self, rng) -> list[np.ndarray]:
        """Detections with probability p_d, Gaussian noise, Poisson clutter, shuffled order."""
        rng = np.random.default_rng(rng)
        mdl = self.model
        L = np.linalg.cholesky(mdl.R)
        out = []
        for k in range(self.n_scans):
            _, X = self.truth_at(k)
            det = rng.random(len(X)) < mdl.p_d
            Zt = X[det] @ mdl.H.T + rng.standard_normal((det.sum(), mdl.dz)) @ L.T
            Z = np.vstack([Zt, self.clutter.sample(rng)])
            out.append(Z[rng.permutation(len(Z))])
        return out


def propagate_truth(
    model: LinearGaussianModel, specs, n_scans: int, process_noise: bool = False, rng=None
) -> dict:
    """Ground-truth trajectories from ``(label, initial_state, k_birth, k_death)`` specs.

    ``k_death`` is exclusive; ``None`` means the object survives to the end.
    """
    rng = np.random.default_rng(rng)
    truth = {}
    L = np.linalg.cholesky(model.Q + 1e-12 * np.eye(model.dx))
    for label, x0, k0, k1 in specs:
        k1 = n_scans if k1 is None else min(k1, n_scans)
        traj = np.full((n_scans, model.dx), np.nan)
        x = np.asarray(x0, dtype=float)
        for k in range(k0, k1):
            traj[k] = x
            x = model.F @ x + (L @ rng.standard_normal(model.dx) if process_noise else 0.0)
        truth[label] = traj
    return truth


BIRTH_LOCATIONS = np.array([[0.0, 0.0], [400.0, -600.0], [-800.0, -200.0], [-200.0, 800.0]])


def default_scenario(
    p_d: float = 0.95, clutter_rate: float = 10.0, sigma: float = 10.0, n_scans: int = 100
) -> Scenario:
    """Ten targets in a 2 km x 2 km surveillance region with staggered births,
    deaths and crossing trajectories (in the style of the benchmark of Vo & Ma 2006).
    """
    model = constant_velocity(ndim=2, dt=1.0, q=1.0, sigma_pos=sigma, p_s=0.99, p_d=p_d)
    clutter = PoissonClutter(clutter_rate, np.array([[-1000.0, 1000.0], [-1000.0, 1000.0]]))
    L1, L2, L3, L4 = BIRTH_LOCATIONS
    specs = [
        ("T01", [L1[0], 0.0, L1[1], -10.0], 0, 70),
        ("T02", [L2[0], -10.0, L2[1], 5.0], 0, None),
        ("T03", [L3[0], 20.0, L3[1], -5.0], 0, 70),
        ("T04", [L3[0], 7.0, L3[1], -8.0], 20, None),
        ("T05", [L3[0], 12.0, L3[1], 7.0], 20, None),
        ("T06", [L4[0], 12.0, L4[1], -10.0], 20, None),
        ("T07", [L1[0], -12.0, L1[1], 15.0], 40, None),
        ("T08", [L4[0], 3.0, L4[1], -12.0], 40, None),
        ("T09", [L2[0], -3.0, L2[1], 15.0], 60, None),
        ("T10", [L1[0], 10.0, L1[1], 10.0], 60, None),
    ]
    truth = propagate_truth(model, specs, n_scans)
    birth_means = np.array([[x, 0.0, y, 0.0] for x, y in BIRTH_LOCATIONS])
    birth_cov = np.diag([10.0, 20.0, 10.0, 20.0]) ** 2
    birth = GaussianBirth(np.full(4, 0.03), birth_means, np.repeat(birth_cov[None], 4, axis=0))
    return Scenario(model, clutter, n_scans, truth, birth)
