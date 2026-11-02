"""Baseline: global nearest neighbour tracker with M-of-N track management."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .kalman import gate_threshold, predict, update
from .models import Estimate, LinearGaussianModel, inverse_measurement


@dataclass
class _Track:
    label: int
    mean: np.ndarray
    cov: np.ndarray
    hits: list = field(default_factory=list)
    confirmed: bool = False
    misses: int = 0


class GNNTracker:
    """Single-hypothesis tracker: optimal 2-D assignment per scan, Kalman filter per track.

    Tracks start from unassigned measurements, are confirmed after ``confirm_m`` hits in
    their first ``confirm_n`` scans and are deleted after ``delete_misses`` consecutive misses.
    """

    name = "GNN"

    def __init__(
        self,
        model: LinearGaussianModel,
        *,
        gate_prob: float = 0.999,
        confirm_m: int = 3,
        confirm_n: int = 4,
        delete_misses: int = 3,
        init_unmeasured_std=50.0,
    ):
        self.model = model
        self.gate = gate_threshold(gate_prob, model.dz)
        self.confirm_m, self.confirm_n = confirm_m, confirm_n
        self.delete_misses = delete_misses
        self.init_unmeasured_std = init_unmeasured_std
        self.tracks: list[_Track] = []
        self._next_label = 0

    def step(self, Z, k: int | None = None) -> Estimate:
        mdl = self.model
        Z = np.asarray(Z, dtype=float).reshape(-1, mdl.dz)
        assigned_meas = np.zeros(len(Z), dtype=bool)

        if self.tracks:
            m, P = predict(
                np.array([t.mean for t in self.tracks]), np.array([t.cov for t in self.tracks]), mdl.F, mdl.Q
            )
            for t, mi, Pi in zip(self.tracks, m, P):
                t.mean, t.cov = mi, Pi
            hit = np.zeros(len(self.tracks), dtype=bool)
            if len(Z):
                q, mu, Pu, _ = update(m, P, mdl.H, mdl.R, Z, self.gate)
                big = 1e12
                with np.errstate(divide="ignore"):
                    cost = np.where(q > 0, -np.log(q), big)
                rows, cols = linear_sum_assignment(cost)
                for r, c in zip(rows, cols):
                    if cost[r, c] < big:
                        t = self.tracks[r]
                        t.mean, t.cov = mu[r, c], Pu[r]
                        hit[r] = True
                        assigned_meas[c] = True
            for t, h in zip(self.tracks, hit):
                t.hits.append(bool(h))
                t.misses = 0 if h else t.misses + 1
                if not t.confirmed and sum(t.hits[: self.confirm_n]) >= self.confirm_m:
                    t.confirmed = True
            self.tracks = [t for t in self.tracks if self._keep(t)]

        new_Z = Z[~assigned_meas]
        if len(new_Z):
            means, covs = inverse_measurement(mdl, new_Z, self.init_unmeasured_std)
            for mi, Pi in zip(means, covs):
                self.tracks.append(_Track(self._next_label, mi, Pi, hits=[True]))
                self._next_label += 1

        confirmed = [t for t in self.tracks if t.confirmed]
        states = np.array([t.mean for t in confirmed]).reshape(-1, mdl.dx)
        return Estimate(states, [t.label for t in confirmed])

    def _keep(self, t: _Track) -> bool:
        if t.confirmed:
            return t.misses < self.delete_misses
        return len(t.hits) < self.confirm_n or sum(t.hits) >= self.confirm_m
