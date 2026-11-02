"""Check that the tracker integration tests do not pass by luck of the seed.

Re-runs the assertions of tests/test_trackers.py with measurement seeds 0-19 (and, for GLMB,
tracker RNG seeds 0-19) and prints the margins and the number of failing seeds.

    python tools/seed_sweep.py          # from the repository root, ~1 min
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import test_trackers as T  # noqa: E402

from rfsmtt import GLMB, tracks_from_estimates  # noqa: E402

SEEDS = range(20)
OSPA_LIMIT, FRACTION_LIMIT = 15.0, 0.8  # the limits used in test_tracks_two_targets


def two_target_metrics(est, sc):
    mean_ospa = T.mean_ospa(sc, est)
    frac_two = np.mean(np.array([len(e) for e in est[5:]]) == 2)
    return mean_ospa, frac_two


def long_track_count(est, sc):
    positions = [sc.model.positions(e.states) for e in est]
    tracks = tracks_from_estimates([e.labels for e in est], positions, sc.n_scans)
    return int(sum(np.isfinite(t[:, 0]).sum() > 20 for t in tracks.values()))


def main():
    print(f"test_tracks_two_targets: mean OSPA (limit {OSPA_LIMIT} m), "
          f"fraction of scans with 2 tracks (limit {FRACTION_LIMIT})")
    for name, factory in T.FACTORIES.items():
        ospas, fractions, fails = [], [], 0
        for seed in SEEDS:
            sc = T.two_target_scenario()
            mean_ospa, frac_two = two_target_metrics(T.run(factory(sc), sc, seed=seed), sc)
            ospas.append(mean_ospa)
            fractions.append(frac_two)
            fails += not (mean_ospa < OSPA_LIMIT and frac_two > FRACTION_LIMIT)
        print(f"  {name:5s} OSPA min/median/max {min(ospas):5.2f}/{np.median(ospas):5.2f}/{max(ospas):5.2f}"
              f"  worst fraction {min(fractions):.2f}  failing seeds {fails}/{len(SEEDS)}")

    print("test_labelled_trackers_keep_identities: tracks longer than 20 scans (must be 2)")
    for name in ("glmb", "gnn"):
        counts = []
        for seed in SEEDS:
            sc = T.two_target_scenario(clutter_rate=1.0)
            counts.append(long_track_count(T.run(T.FACTORIES[name](sc), sc, seed=seed), sc))
        print(f"  {name:5s} counts {counts}  failing {sum(c != 2 for c in counts)}/{len(counts)}")

    print("GLMB tracker RNG sweep (measurement seed 0), two-target test")
    fails = 0
    for rng in SEEDS:
        sc = T.two_target_scenario()
        est = T.run(GLMB(sc.model, sc.clutter, sc.birth, h_upd=200, h_max=200, rng=rng), sc)
        mean_ospa, frac_two = two_target_metrics(est, sc)
        fails += not (mean_ospa < OSPA_LIMIT and frac_two > FRACTION_LIMIT)
    print(f"  failing {fails}/{len(SEEDS)}")


if __name__ == "__main__":
    main()
