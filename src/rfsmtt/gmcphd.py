"""Gaussian-mixture cardinalized PHD filter (Vo, Vo & Cantoni, IEEE TSP 2007)."""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln, logsumexp, xlogy
from scipy.stats import binom, poisson

from .gmphd import GMPHD
from .kalman import update
from .models import Estimate, unique_labels


def esf(values) -> np.ndarray:
    """Elementary symmetric functions e_0, ..., e_n of ``values``."""
    values = np.asarray(values, dtype=float)
    e = np.zeros(len(values) + 1)
    e[0] = 1.0
    for v in values:
        e[1:] = e[1:] + v * e[:-1]
    return e


def log_esf(values) -> np.ndarray:
    """log of :func:`esf`, scaled to avoid overflow for many measurements."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.zeros(1)
    c = max(float(values.max()), np.finfo(float).tiny)
    with np.errstate(divide="ignore"):
        return np.log(esf(values / c)) + np.arange(len(values) + 1) * np.log(c)


class GMCPHD(GMPHD):
    """GM-CPHD filter: propagates the intensity and the full cardinality distribution.

    Assumes Poisson clutter, constant survival and detection probabilities and a
    Poisson birth process, as in the original derivation.
    """

    name = "GM-CPHD"

    def __init__(self, model, clutter, birth, *, n_max: int = 60, **kwargs):
        super().__init__(model, clutter, birth, **kwargs)
        self.n_max = n_max
        self.cdn = np.zeros(n_max + 1)
        self.cdn[0] = 1.0

    def _predict_cardinality(self, birth_mass: float) -> np.ndarray:
        n = np.arange(self.n_max + 1)
        survive = binom.pmf(n[:, None], n[None, :], self.model.p_s) @ self.cdn
        born = poisson.pmf(n, birth_mass)
        cdn = np.convolve(survive, born)[: self.n_max + 1]
        return cdn / cdn.sum()

    def _log_upsilon(self, log_e: np.ndarray, n_meas: int, u: int, log_W: float) -> np.ndarray:
        """log Upsilon^u[w, Z](n) for n = 0..n_max (Vo, Vo & Cantoni, eq. 21)."""
        lam, q_d = self.clutter.rate, 1.0 - self.model.p_d
        n = np.arange(self.n_max + 1)[:, None]
        j = np.arange(n_meas + 1)[None, :]
        valid = n - j - u >= 0
        rest = np.where(valid, n - j - u, 0)
        terms = (
            -lam
            + xlogy(n_meas - j, lam)
            + gammaln(n + 1)
            - gammaln(rest + 1)
            + xlogy(rest, q_d)
            - (j + u) * log_W
            + log_e[j]
        )
        terms = np.where(valid, terms, -np.inf)
        return logsumexp(terms, axis=1)

    def step(self, Z, k: int | None = None) -> Estimate:
        mdl = self.model
        Z = np.asarray(Z, dtype=float).reshape(-1, mdl.dz)
        n_prev = len(self.w)
        w, m, P, tags = self._predict(k)
        birth_mass = float(w[: len(w) - n_prev].sum())
        cdn_pred = self._predict_cardinality(birth_mass)

        if len(w) == 0 or w.sum() <= 0:
            self.cdn = cdn_pred
            self.w, self.m, self.P, self.tags = w, m, P, tags
            self._Z_prev, self._r_used_prev = Z, np.zeros(len(Z))
            return self._extract()

        q, mu, Pu, _ = update(m, P, mdl.H, mdl.R, Z, self.gate)
        mz = len(Z)
        log_W = np.log(w.sum())
        xi = mdl.p_d * (w @ q) / self.clutter.density
        log_e = log_esf(xi)
        with np.errstate(divide="ignore"):
            log_cdn = np.log(cdn_pred)

        log_u0 = self._log_upsilon(log_e, mz, 0, log_W)
        log_u1 = self._log_upsilon(log_e, mz, 1, log_W)
        log_den = logsumexp(log_u0 + log_cdn)

        w_miss = (1.0 - mdl.p_d) * w * np.exp(logsumexp(log_u1 + log_cdn) - log_den)
        w_det = np.zeros((len(w), mz))
        for ell in range(mz):
            log_e_minus = log_esf(np.delete(xi, ell))
            log_u1_d = self._log_upsilon(log_e_minus, mz - 1, 1, log_W)
            ratio = np.exp(logsumexp(log_u1_d + log_cdn) - log_den)
            w_det[:, ell] = mdl.p_d * w * q[:, ell] * ratio / self.clutter.density

        cdn = np.exp(log_u0 + log_cdn - log_den)
        self.cdn = cdn / cdn.sum()
        r_used = np.minimum(w_det.sum(axis=0), 1.0)
        w, m, P, tags = self._assemble(w_miss, w_det, m, P, mu, Pu, tags)
        self.w, self.m, self.P, self.tags = self._reduce(w, m, P, tags)
        self._Z_prev, self._r_used_prev = Z, r_used
        return self._extract()

    def _extract(self) -> Estimate:
        n_hat = min(int(np.argmax(self.cdn)), len(self.w))
        idx = np.argsort(-self.w)[:n_hat]
        return Estimate(self.m[idx].copy(), unique_labels(self.tags[idx]), cardinality_pmf=self.cdn.copy())
