"""Labeled multi-object tracking with the delta-GLMB filter.

Implements the joint prediction-update with Gibbs sampling of
B.-N. Vo, B.-T. Vo and H. G. Hoang, "An efficient implementation of the
generalized labeled multi-Bernoulli filter", IEEE TSP 65(8), 2017.

The posterior is a weighted set of hypotheses. Each hypothesis is a set of
labeled tracks (indices into a shared track table); each track is a Gaussian.
Labels are ``(k, i)``: the time step and index of the birth component.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from .kalman import gate_threshold, predict, update
from .models import Estimate, LinearGaussianModel, PoissonClutter


def gibbs_sample(eta_death, eta_miss, eta_det, n_samples: int, rng: np.random.Generator):
    """Draw distinct joint association hypotheses with a Gibbs sampler.

    Parameters
    ----------
    eta_death : (n,) weight of track i not existing (death, or not born).
    eta_miss : (n,) weight of track i existing but undetected.
    eta_det : (n, m) weight of track i generating measurement j.
    n_samples : number of Gibbs iterations (the first sample is "all missed").

    Returns
    -------
    codes : (S, n) int array, -1 = not existing, 0 = missed, j >= 1 = measurement j - 1.
    log_weights : (S,) log of the product of the chosen weights.
    """
    eta_death = np.asarray(eta_death, dtype=float)
    eta_miss = np.asarray(eta_miss, dtype=float)
    if len(eta_death) == 0:
        return np.zeros((1, 0), dtype=int), np.zeros(1)
    eta_det = np.asarray(eta_det, dtype=float).reshape(len(eta_death), -1)
    n, m = eta_det.shape

    rows = np.concatenate([eta_death[:, None], eta_miss[:, None], eta_det], axis=1)
    current = np.zeros(n, dtype=int)
    used = np.zeros(m, dtype=bool)
    samples = [current.copy()]
    for _ in range(1, max(1, n_samples)):
        for i in range(n):
            if current[i] > 0:
                used[current[i] - 1] = False
            p = rows[i].copy()
            p[2:][used] = 0.0
            cdf = np.cumsum(p)
            choice = int(np.searchsorted(cdf, rng.random() * cdf[-1], side="right"))
            current[i] = min(choice, len(p) - 1) - 1
            if current[i] > 0:
                used[current[i] - 1] = True
        samples.append(current.copy())

    codes = np.unique(np.array(samples), axis=0)
    with np.errstate(divide="ignore"):
        log_rows = np.log(rows)
    log_weights = log_rows[np.arange(n)[None, :], codes + 1].sum(axis=1)
    return codes, log_weights


class GLMB:
    """delta-GLMB filter with joint prediction and update."""

    name = "GLMB"

    def __init__(
        self,
        model: LinearGaussianModel,
        clutter: PoissonClutter,
        birth,
        *,
        h_upd: int = 1000,
        h_max: int = 1000,
        h_prune: float = 1e-15,
        gate_prob: float | None = 0.9999,
        rng=None,
    ):
        self.model = model
        self.clutter = clutter
        self.birth = birth
        self.h_upd = h_upd
        self.h_max = h_max
        self.h_prune = h_prune
        self.gate = gate_threshold(gate_prob, model.dz) if gate_prob else None
        self.rng = np.random.default_rng(rng)
        dx = model.dx
        self.labels: list = []
        self.means = np.zeros((0, dx))
        self.covs = np.zeros((0, dx, dx))
        self.hyp_w = np.array([1.0])
        self.hyp_tracks: list[np.ndarray] = [np.zeros(0, dtype=int)]
        self.cardinality_pmf = np.array([1.0])
        self._Z_prev = np.zeros((0, model.dz))
        self._r_used_prev = np.zeros(0)

    def step(self, Z, k: int | None = None) -> Estimate:
        mdl = self.model
        dx = mdl.dx
        Z = np.asarray(Z, dtype=float).reshape(-1, mdl.dz)
        mz = len(Z)
        k = 0 if k is None else k

        # --- predicted track table: births first, then survivors
        r_b, m_b, P_b = self.birth.components(k, self._Z_prev, self._r_used_prev)
        nb = len(r_b)
        if len(self.means):
            m_s, P_s = predict(self.means, self.covs, mdl.F, mdl.Q)
        else:
            m_s, P_s = self.means, self.covs
        pm = np.concatenate([np.reshape(m_b, (-1, dx)), m_s])
        pP = np.concatenate([np.reshape(P_b, (-1, dx, dx)), P_s])
        plabels = [(k, i) for i in range(nb)] + self.labels
        cp = len(pm)

        p_exist = np.concatenate([r_b, np.full(len(self.means), mdl.p_s)])
        eta_death = 1.0 - p_exist
        eta_miss = p_exist * (1.0 - mdl.p_d)
        if cp:
            q, mu, Pu, _ = update(pm, pP, mdl.H, mdl.R, Z, self.gate)
        else:
            q, mu, Pu = np.zeros((0, mz)), np.zeros((0, mz, dx)), np.zeros((0, dx, dx))
        eta_det = (p_exist * mdl.p_d)[:, None] * q / self.clutter.intensity

        # --- joint prediction-update, hypothesis by hypothesis
        sqrt_w = np.sqrt(self.hyp_w)
        n_samples = np.maximum(1, np.round(self.h_upd * sqrt_w / sqrt_w.sum())).astype(int)
        new: dict[tuple, float] = {}
        for w_h, tracks_h, s_h in zip(self.hyp_w, self.hyp_tracks, n_samples):
            tind = np.concatenate([np.arange(nb), nb + tracks_h]).astype(int)
            sub = eta_det[tind]
            mcols = np.flatnonzero(sub.any(axis=0))
            codes, log_c = gibbs_sample(eta_death[tind], eta_miss[tind], sub[:, mcols], s_h, self.rng)
            for code, lc in zip(codes, log_c):
                if not np.isfinite(lc):
                    continue
                alive = code >= 0
                meas = np.zeros(len(code), dtype=int)
                det = code > 0
                meas[det] = mcols[code[det] - 1] + 1
                key = tuple(sorted((cp * meas[alive] + tind[alive]).tolist()))
                lw = np.log(w_h) + lc
                new[key] = np.logaddexp(new[key], lw) if key in new else lw

        keys = list(new)
        log_w = np.array([new[key] for key in keys])
        w = np.exp(log_w - logsumexp(log_w))
        order = np.argsort(-w)
        order = order[w[order] > self.h_prune][: self.h_max]
        keys = [keys[i] for i in order]
        w = w[order] / w[order].sum()

        # --- new track table holding only referenced tracks
        all_idx = np.unique(np.fromiter((i for key in keys for i in key), dtype=int))
        base = max(cp, 1)
        pred_i, meas_i = all_idx % base, all_idx // base
        detected = meas_i > 0
        means = pm[pred_i].copy()
        covs = pP[pred_i].copy()
        means[detected] = mu[pred_i[detected], meas_i[detected] - 1]
        covs[detected] = Pu[pred_i[detected]]
        remap = {int(old): new_i for new_i, old in enumerate(all_idx)}

        self.labels = [plabels[i] for i in pred_i]
        self.means, self.covs = means, covs
        self.hyp_w = w
        self.hyp_tracks = [np.array([remap[i] for i in key], dtype=int) for key in keys]

        # probability that each measurement was used by some track (for adaptive birth)
        r_used = np.zeros(mz)
        for w_h, key in zip(w, keys):
            used = [i // base - 1 for i in key if i // base > 0]
            r_used[used] += w_h
        self._Z_prev, self._r_used_prev = Z, np.minimum(r_used, 1.0)

        card = np.array([len(t) for t in self.hyp_tracks])
        self.cardinality_pmf = np.bincount(card, weights=w)
        return self._extract(card)

    def _extract(self, card) -> Estimate:
        n_hat = int(np.argmax(self.cardinality_pmf))
        candidates = np.flatnonzero(card == n_hat)
        best = candidates[np.argmax(self.hyp_w[candidates])]
        tracks = self.hyp_tracks[best]
        return Estimate(
            self.means[tracks].copy(),
            [self.labels[i] for i in tracks],
            cardinality_pmf=self.cardinality_pmf.copy(),
        )
