import numpy as np

from rfsmtt import label_switches, ospa, ospa2, tracks_from_estimates


def test_ospa_basic_properties():
    X = np.array([[0.0, 0.0], [10.0, 0.0]])
    assert ospa(X, X)[0] == 0.0
    assert ospa(X, np.zeros((0, 2)), c=50)[0] == 50.0
    assert ospa(np.zeros((0, 2)), np.zeros((0, 2)))[0] == 0.0
    Y = np.array([[0.0, 3.0]])
    assert np.isclose(ospa(X, Y)[0], ospa(Y, X)[0])
    total, loc, card = ospa(X, Y, c=100, p=1)
    assert np.isclose(total, (3.0 + 100.0) / 2)
    assert np.isclose(loc + card, total)


def test_ospa_cutoff():
    assert np.isclose(ospa([[0.0, 0.0]], [[1000.0, 0.0]], c=100)[0], 100.0)


def _tracks(n, pts):
    out = {}
    for lab, k, x in pts:
        out.setdefault(lab, np.full((n, 2), np.nan))[k] = x
    return out


def test_ospa2_zero_for_identical_tracks():
    true = _tracks(5, [("a", k, (k, 0)) for k in range(5)])
    assert ospa2(true, true) == 0.0


def test_ospa2_penalises_label_switch():
    true = _tracks(6, [("a", k, (k, 0)) for k in range(6)])
    est_same = _tracks(6, [(1, k, (k, 0)) for k in range(6)])
    est_switch = _tracks(6, [(1 if k < 3 else 2, k, (k, 0)) for k in range(6)])
    assert ospa2(est_same, true) == 0.0
    assert ospa2(est_switch, true, c=10) > 0.0
    assert label_switches(est_switch, true, gate=5) == 1
    assert label_switches(est_same, true, gate=5) == 0


def test_tracks_from_estimates():
    labels = [[1], [1, 2], []]
    pos = [np.array([[0, 0]]), np.array([[1, 0], [5, 5]]), np.zeros((0, 2))]
    tr = tracks_from_estimates(labels, pos, 3)
    assert set(tr) == {1, 2}
    assert np.isnan(tr[2][0, 0]) and np.allclose(tr[2][1], [5, 5])
