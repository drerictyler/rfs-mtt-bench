"""Gaussian-mixture PHD filter (Vo & Ma, IEEE TSP 2006) with tag propagation."""

from __future__ import annotations

import numpy as np

from .kalman import gate_threshold, predict, update
from .mixture import cap, merge, prune
from .models import Estimate, LinearGaussianModel, PoissonClutter, unique_labels


class GMPHD:
    """Gaussian-mixture probability hypothesis density filter.

    The PHD filter propagates the first moment (intensity) of the multi-object
    posterior. Identities are not part of the model; they are recovered by
    propagating a tag with every Gaussian component.
    """

    name = "GM-PHD"

    def __init__(
        self,
        model: LinearGaussianModel,
        clutter: PoissonClutter,
        birth,
        *,
        prune_threshold: float = 1e-5,
        merge_threshold: float = 4.0,
        max_components: int = 100,
        extract_threshold: float = 0.5,
        gate_prob: float | None = 0.9999,
    ):
        self.model = model
        self.clutter = clutter
        self.birth = birth
        self.prune_threshold = prune_threshold
        self.merge_threshold = merge_threshold
        self.max_components = max_components
        self.extract_threshold = extract_threshold
        self.gate = gate_threshold(gate_prob, model.dz) if gate_prob else None
        dx = model.dx
        self.w = np.zeros(0)
        self.m = np.zeros((0, dx))
        self.P = np.zeros((0, dx, dx))
        self.tags = np.zeros(0, dtype=int)
        self._next_tag = 0
        self._Z_prev = np.zeros((0, model.dz))
        self._r_used_prev = np.zeros(0)

    # ------------------------------------------------------------------ helpers
    def _new_tags(self, n: int) -> np.ndarray:
        tags = np.arange(self._next_tag, self._next_tag + n)
        self._next_tag += n
        return tags

    def _predict(self, k):
        mdl = self.model
        w = mdl.p_s * self.w
        if len(w):
            m, P = predict(self.m, self.P, mdl.F, mdl.Q)
        else:
            m, P = self.m, self.P
        wb, mb, Pb = self.birth.components(k, self._Z_prev, self._r_used_prev)
        return (
            np.concatenate([wb, w]),
            np.concatenate([mb.reshape(-1, mdl.dx), m]),
            np.concatenate([Pb.reshape(-1, mdl.dx, mdl.dx), P]),
            np.concatenate([self._new_tags(len(wb)), self.tags]),
        )

    def _assemble(self, w_miss, w_det, m, P, mu, Pu, tags):
        """Stack missed-detection and detected components into one mixture."""
        n, mz = w_det.shape
        w = np.concatenate([w_miss, w_det.T.ravel()])
        means = np.concatenate([m, mu.transpose(1, 0, 2).reshape(-1, self.model.dx)])
        covs = np.concatenate([P, np.tile(Pu, (mz, 1, 1))])
        t = np.concatenate([tags, np.tile(tags, mz)])
        return w, means, covs, t

    def _reduce(self, w, m, P, tags):
        w, m, P, tags = prune(w, m, P, tags, self.prune_threshold)
        w, m, P, tags = merge(w, m, P, tags, self.merge_threshold)
        w, m, P, tags = cap(w, m, P, tags, self.max_components)
        return w, m, P, self._split_tags(w, tags)

    def _split_tags(self, w, tags):
        """Give a fresh tag to every significant component that shares its tag with a heavier one.

        One component (typically a birth component) can spawn several objects in the same
        scan; without this step they would all inherit the same tag.
        """
        tags = tags.copy()
        seen = set()
        for i in np.argsort(-w):
            if w[i] <= self.extract_threshold:
                break
            if tags[i] in seen:
                tags[i] = self._new_tags(1)[0]
            seen.add(tags[i])
        return tags

    # --------------------------------------------------------------------- step
    def step(self, Z, k: int | None = None) -> Estimate:
        mdl = self.model
        Z = np.asarray(Z, dtype=float).reshape(-1, mdl.dz)
        w, m, P, tags = self._predict(k)

        if len(w):
            q, mu, Pu, _ = update(m, P, mdl.H, mdl.R, Z, self.gate)
            num = mdl.p_d * w[:, None] * q
            w_det = num / (self.clutter.intensity + num.sum(axis=0))
            r_used = w_det.sum(axis=0)
            w, m, P, tags = self._assemble((1.0 - mdl.p_d) * w, w_det, m, P, mu, Pu, tags)
        else:
            r_used = np.zeros(len(Z))

        self.w, self.m, self.P, self.tags = self._reduce(w, m, P, tags)
        self._Z_prev, self._r_used_prev = Z, r_used
        return self._extract()

    def _extract(self) -> Estimate:
        idx = np.flatnonzero(self.w > self.extract_threshold)
        reps = np.maximum(1, np.round(self.w[idx]).astype(int))
        sel = np.repeat(idx, reps)
        return Estimate(self.m[sel].copy(), unique_labels(self.tags[sel]))

    @property
    def expected_count(self) -> float:
        return float(self.w.sum())
