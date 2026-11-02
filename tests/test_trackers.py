import numpy as np
import pytest

from rfsmtt import (
    GLMB,
    GMCPHD,
    GMPHD,
    AdaptiveBirth,
    GaussianBirth,
    GNNTracker,
    PoissonClutter,
    Scenario,
    constant_velocity,
    ospa,
    propagate_truth,
    tracks_from_estimates,
)


def two_target_scenario(clutter_rate=2.0, p_d=0.98, n=30):
    model = constant_velocity(sigma_pos=5.0, q=0.5, p_d=p_d)
    clutter = PoissonClutter(clutter_rate, [[-500, 500], [-500, 500]])
    specs = [("A", [-300, 10, -200, 0], 0, None), ("B", [300, -10, 200, 0], 0, None)]
    truth = propagate_truth(model, specs, n)
    birth = GaussianBirth(
        [0.05, 0.05],
        [[-300, 0, -200, 0], [300, 0, 200, 0]],
        np.repeat(np.diag([10.0, 15.0, 10.0, 15.0])[None] ** 2, 2, axis=0),
    )
    return Scenario(model, clutter, n, truth, birth)


def run(tracker, sc, seed=0):
    Z = sc.measurements(seed)
    return [tracker.step(z, k) for k, z in enumerate(Z)]


def mean_ospa(sc, est, start=5):
    return np.mean(
        [
            ospa(sc.model.positions(e.states), sc.model.positions(sc.truth_at(k)[1]))[0]
            for k, e in enumerate(est)
            if k >= start
        ]
    )


FACTORIES = {
    "phd": lambda sc: GMPHD(sc.model, sc.clutter, sc.birth),
    "cphd": lambda sc: GMCPHD(sc.model, sc.clutter, sc.birth),
    "glmb": lambda sc: GLMB(sc.model, sc.clutter, sc.birth, h_upd=200, h_max=200, rng=1),
    "gnn": lambda sc: GNNTracker(sc.model, confirm_m=4, confirm_n=5, init_unmeasured_std=20.0),
}


@pytest.mark.parametrize("name", list(FACTORIES))
def test_tracks_two_targets(name):
    sc = two_target_scenario()
    est = run(FACTORIES[name](sc), sc)
    assert mean_ospa(sc, est) < 15.0
    counts = np.array([len(e) for e in est[5:]])
    assert np.mean(counts == 2) > 0.8


@pytest.mark.parametrize("name", ["glmb", "gnn"])
def test_labelled_trackers_keep_identities(name):
    sc = two_target_scenario(clutter_rate=1.0)
    est = run(FACTORIES[name](sc), sc)
    tracks = tracks_from_estimates(
        [e.labels for e in est], [sc.model.positions(e.states) for e in est], sc.n_scans
    )
    long = [lab for lab, t in tracks.items() if np.isfinite(t[:, 0]).sum() > 20]
    assert len(long) == 2


def test_glmb_weights_and_cardinality_are_normalised():
    sc = two_target_scenario()
    tracker = GLMB(sc.model, sc.clutter, sc.birth, h_upd=200, h_max=200, rng=2)
    for k, z in enumerate(sc.measurements(1)[:10]):
        est = tracker.step(z, k)
        assert np.isclose(tracker.hyp_w.sum(), 1.0)
        assert np.isclose(est.cardinality_pmf.sum(), 1.0)
        assert len(set(est.labels)) == len(est.labels)


def test_cphd_cardinality_distribution():
    sc = two_target_scenario()
    tracker = GMCPHD(sc.model, sc.clutter, sc.birth)
    for k, z in enumerate(sc.measurements(1)):
        est = tracker.step(z, k)
    assert np.isclose(tracker.cdn.sum(), 1.0)
    assert np.argmax(est.cardinality_pmf) == 2


def test_adaptive_birth_initiates_tracks_anywhere():
    model = constant_velocity(sigma_pos=5.0, q=0.5)
    clutter = PoissonClutter(1.0, [[-500, 500], [-500, 500]])
    truth = propagate_truth(model, [("A", [120, 5, -340, 5], 0, None)], 25)
    sc = Scenario(
        model, clutter, 25, truth, AdaptiveBirth(model, expected_births=0.1, r_max=0.1, unmeasured_std=15)
    )
    for f in (GMPHD(model, clutter, sc.birth), GLMB(model, clutter, sc.birth, h_upd=100, h_max=100, rng=0)):
        est = run(f, sc)
        assert np.mean([len(e) == 1 for e in est[8:]]) > 0.8
