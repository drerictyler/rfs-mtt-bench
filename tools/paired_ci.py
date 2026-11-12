"""Paired 95% confidence intervals for the benchmark's headline comparisons.

Runs every tracker on the same Monte Carlo runs as the benchmark (seeds 0-19), keeps the
per-run OSPA(2), cardinality error and ID switches, and reports each tracker's mean with a
95% t-interval and the paired differences behind the README's table. Runs, not scans, are
the independent units.

    python tools/paired_ci.py           # ~10 min
"""

import numpy as np
from scipy import stats

from rfsmtt.benchmark import CONDITIONS, evaluate, make_tracker
from rfsmtt.scenario import default_scenario

RUNS = 20
TRACKERS = ("GM-PHD", "GM-CPHD", "GLMB", "GNN")
PAIRS = [  # (a, b, metric): the README's table reports a - b
    ("GNN", "GLMB", "ospa2"),
    ("GLMB", "GNN", "switches"),
    ("GM-PHD", "GM-CPHD", "ospa2"),
    ("GLMB", "GM-CPHD", "ospa2"),
    ("GM-CPHD", "GM-PHD", "card"),
]


def ci95(x) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    return float(x.mean()), float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def main():
    for cond, cfg in CONDITIONS.items():
        sc = default_scenario(**cfg)
        true_card = sc.cardinality()
        per_run = {}
        for name in TRACKERS:
            rows = {"ospa2": [], "card": [], "switches": []}
            for r in range(RUNS):
                Z = sc.measurements(r)
                tracker = make_tracker(name, sc, r)
                ev = evaluate(sc, [tracker.step(z, k) for k, z in enumerate(Z)])
                rows["ospa2"].append(ev["ospa2"].mean())
                rows["card"].append(np.abs(ev["card"] - true_card).mean())
                rows["switches"].append(ev["switches"])
            per_run[name] = rows
        print(f"== {cond}: mean ± 95% half-width over {RUNS} runs")
        for name, rows in per_run.items():
            summary = ((k, *ci95(v)) for k, v in rows.items())
            cells = "  ".join(f"{k} {m:6.2f} ± {h:4.2f}" for k, m, h in summary)
            print(f"  {name:8s} {cells}")
        print("  paired differences (a - b), mean and 95% confidence interval of the mean:")
        for a, b, metric in PAIRS:
            m, h = ci95(np.array(per_run[a][metric]) - np.array(per_run[b][metric]))
            verdict = "excludes 0" if abs(m) > h else "includes 0: no winner established"
            print(f"    {a} - {b} {metric}: {m:+.2f}  [{m - h:+.2f}, {m + h:+.2f}]  ({verdict})")


if __name__ == "__main__":
    main()
