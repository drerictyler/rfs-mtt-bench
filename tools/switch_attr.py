"""Attribute GLMB and GNN ID switches to the true targets that suffer them.

Uses the same counting rule as rfsmtt.metrics.label_switches, so the per-run totals match the
benchmark tables. Runs both conditions with 20 Monte Carlo runs each (seeds 0-19).

    python tools/switch_attr.py         # ~10 min
"""

from collections import Counter

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from rfsmtt.benchmark import CONDITIONS, make_tracker
from rfsmtt.metrics import tracks_from_estimates
from rfsmtt.scenario import default_scenario

RUNS = 20
GATE = 50.0  # the benchmark uses cutoff / 2


def switches_by_target(est_tracks: dict, true_tracks: dict, gate: float) -> Counter:
    """label_switches, but returning the count per true target instead of the total."""
    e_labels, E = list(est_tracks), np.stack(list(est_tracks.values()))
    t_labels, T = list(true_tracks), np.stack(list(true_tracks.values()))
    e_index = {lab: i for i, lab in enumerate(e_labels)}
    last: dict = {}
    out: Counter = Counter()
    for k in range(T.shape[1]):
        ei = set(np.flatnonzero(np.isfinite(E[:, k, 0])).tolist())
        ti = np.flatnonzero(np.isfinite(T[:, k, 0]))
        if not ei or len(ti) == 0:
            continue
        free_t = []
        for t in ti:
            j = e_index.get(last.get(t_labels[t]))
            if j in ei and np.linalg.norm(T[t, k] - E[j, k]) <= gate:
                ei.discard(j)
            else:
                free_t.append(t)
        if not free_t or not ei:
            continue
        free_e = sorted(ei)
        D = cdist(T[free_t, k], E[free_e, k])
        for r, c in zip(*linear_sum_assignment(np.minimum(D, gate * 10))):
            if D[r, c] > gate:
                continue
            tl, el = t_labels[free_t[r]], e_labels[free_e[c]]
            if tl in last and last[tl] != el:
                out[tl] += 1
            last[tl] = el
    return out


def main():
    for cond in CONDITIONS:
        sc = default_scenario(**CONDITIONS[cond])
        totals = {"GLMB": Counter(), "GNN": Counter()}
        for r in range(RUNS):
            Z = sc.measurements(r)
            for name in totals:
                tracker = make_tracker(name, sc, r)
                est = [tracker.step(z, k) for k, z in enumerate(Z)]
                positions = [sc.model.positions(e.states) for e in est]
                tracks = tracks_from_estimates([e.labels for e in est], positions, sc.n_scans)
                totals[name] += switches_by_target(tracks, sc.true_position_tracks(), GATE)
        for name, c in totals.items():
            n = sum(c.values())
            share = (c["T04"] + c["T05"]) / max(n, 1)
            print(f"{cond:7s} {name}: mean {n / RUNS:.2f} per run; by target {dict(sorted(c.items()))}; "
                  f"T04 + T05 share {share:.0%}", flush=True)


if __name__ == "__main__":
    main()
