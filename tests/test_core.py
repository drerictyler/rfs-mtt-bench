import itertools

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from rfsmtt import (
    AdaptiveBirth,
    PoissonClutter,
    constant_velocity,
    default_scenario,
    esf,
    gibbs_sample,
    inverse_measurement,
    log_esf,
)
from rfsmtt.kalman import predict, update
from rfsmtt.mixture import cap, merge


def test_kalman_matches_closed_form():
    model = constant_velocity(ndim=1, dt=1.0, q=0.5, sigma_pos=2.0)
    m0, P0 = np.array([[0.0, 1.0]]), np.array([np.diag([4.0, 1.0])])
    m, P = predict(m0, P0, model.F, model.Q)
    assert np.allclose(m, [[1.0, 1.0]])
    assert np.allclose(P[0], model.F @ P0[0] @ model.F.T + model.Q)

    z = np.array([[1.7]])
    q, mu, Pu, d2 = update(m, P, model.H, model.R, z)
    S = model.H @ P[0] @ model.H.T + model.R
    K = P[0] @ model.H.T @ np.linalg.inv(S)
    assert np.allclose(mu[0, 0], m[0] + K @ (z[0] - model.H @ m[0]))
    assert np.allclose(Pu[0], (np.eye(2) - K @ model.H) @ P[0])
    assert np.isclose(q[0, 0], multivariate_normal(model.H @ m[0], S).pdf(z[0]))


def test_gate_zeroes_distant_measurements():
    model = constant_velocity(sigma_pos=1.0)
    q, *_ = update(np.zeros((1, 4)), np.eye(4)[None], model.H, model.R, [[0.5, 0.0], [50.0, 0.0]], gate=9.21)
    assert q[0, 0] > 0 and q[0, 1] == 0


def test_esf_matches_brute_force():
    x = np.array([0.3, 2.0, 1.5, 0.7])
    expected = [sum(np.prod(c) for c in itertools.combinations(x, j)) for j in range(len(x) + 1)]
    assert np.allclose(esf(x), expected)
    assert np.allclose(np.exp(log_esf(x)), expected)


def test_log_esf_handles_large_values():
    x = np.full(200, 1e4)
    assert np.all(np.isfinite(log_esf(x)))


def test_merge_preserves_mass_and_moments():
    w = np.array([0.6, 0.4])
    m = np.array([[0.0, 0.0], [1.0, 0.0]])
    P = np.repeat(np.eye(2)[None], 2, axis=0)
    w2, m2, P2, t2 = merge(w, m, P, np.array([1, 2]), threshold=4.0)
    assert len(w2) == 1 and np.isclose(w2[0], 1.0)
    assert np.allclose(m2[0], [0.4, 0.0])
    assert t2[0] == 1
    assert P2[0][0, 0] > 1.0


def test_cap_rescales_total_weight():
    w = np.array([0.5, 0.3, 0.2])
    w2, *_ = cap(w, np.zeros((3, 1)), np.ones((3, 1, 1)), np.arange(3), 2)
    assert len(w2) == 2 and np.isclose(w2.sum(), 1.0)


def test_gibbs_samples_are_unique_and_exclusive():
    rng = np.random.default_rng(0)
    eta_det = rng.random((5, 4))
    codes, logw = gibbs_sample(np.full(5, 0.05), np.full(5, 0.1), eta_det, 200, rng)
    assert np.all(codes[0] >= -1)
    assert len(np.unique(codes, axis=0)) == len(codes)
    for c in codes:
        used = c[c > 0]
        assert len(used) == len(set(used.tolist()))
    assert any(np.all(c == 0) for c in codes)  # the all-missed initial sample
    assert np.all(np.isfinite(logw))


def test_inverse_measurement_and_adaptive_birth():
    model = constant_velocity(sigma_pos=5.0)
    m, P = inverse_measurement(model, [[10.0, -20.0]], 30.0)
    assert np.allclose(m[0], [10.0, 0.0, -20.0, 0.0])
    assert np.isclose(P[0][0, 0], 25.0) and np.isclose(P[0][1, 1], 900.0)
    birth = AdaptiveBirth(model, expected_births=0.2, r_max=0.1)
    r, mb, Pb = birth.components(1, np.array([[0.0, 0.0], [100.0, 0.0]]), np.array([1.0, 0.0]))
    assert len(r) == 1 and np.isclose(r[0], 0.1)


def test_clutter_density():
    c = PoissonClutter(10.0, [[0, 100], [0, 50]])
    assert np.isclose(c.intensity, 10.0 / 5000.0)
    pts = c.sample(np.random.default_rng(1))
    assert np.all((pts >= 0) & (pts[:, :1] <= 100) & (pts[:, 1:] <= 50))


def test_scenario_is_reproducible():
    sc = default_scenario()
    a, b = sc.measurements(3), sc.measurements(3)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert sc.cardinality().max() == 10


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_constant_velocity_shapes(ndim):
    model = constant_velocity(ndim=ndim, sigma_vel=1.0)
    assert model.F.shape == (2 * ndim, 2 * ndim)
    assert model.H.shape == (2 * ndim, 2 * ndim)
    assert model.pos_idx == tuple(range(0, 2 * ndim, 2))
