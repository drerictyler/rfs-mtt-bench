"""How much do GLMB's results depend on its sampler's random stream?

Re-runs GLMB on the benchmark's 20 Monte Carlo runs per condition (the same measurements every
time) with independent Gibbs-sampler streams: stream s of run r is
``np.random.SeedSequence(r, spawn_key=(s,))``. Stream 0 is the benchmark's own
(``rfsmtt.benchmark.sampler_seed``). Prints each stream's mean OSPA(2), cardinality error and ID
switches per condition, and the spread across streams.

    python tools/stream_sweep.py                 # streams 0-4, ~20 min
    python tools/stream_sweep.py --streams 1 2   # selected streams
"""

import argparse

import numpy as np

from rfsmtt.benchmark import CONDITIONS, evaluate
from rfsmtt.glmb import GLMB
from rfsmtt.scenario import default_scenario

RUNS = 20


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--streams", type=int, nargs="+", default=list(range(5)))
    args = ap.parse_args()
    for cond, cfg in CONDITIONS.items():
        sc = default_scenario(**cfg)
        true_card = sc.cardinality()
        rows = []
        for s in args.streams:
            ospa2, card, switches = [], [], []
            for r in range(RUNS):
                rng = np.random.SeedSequence(r, spawn_key=(s,))
                tracker = GLMB(sc.model, sc.clutter, sc.birth, h_upd=1000, h_max=1000, rng=rng)
                ev = evaluate(sc, [tracker.step(z, k) for k, z in enumerate(sc.measurements(r))])
                ospa2.append(ev["ospa2"].mean())
                card.append(np.abs(ev["card"] - true_card).mean())
                switches.append(ev["switches"])
            rows.append((s, np.mean(ospa2), np.mean(card), np.mean(switches)))
            print(f"{cond:7s} stream {s}: OSPA(2) {rows[-1][1]:6.2f} m  card MAE {rows[-1][2]:.3f}  "
                  f"ID switches {rows[-1][3]:5.2f} per run", flush=True)
        a = np.array([r[1:] for r in rows])
        print(f"{cond:7s} across streams: OSPA(2) {a[:, 0].min():.2f}-{a[:, 0].max():.2f} m, card MAE "
              f"{a[:, 1].min():.3f}-{a[:, 1].max():.3f}, ID switches {a[:, 2].min():.2f}-{a[:, 2].max():.2f}")


if __name__ == "__main__":
    main()
