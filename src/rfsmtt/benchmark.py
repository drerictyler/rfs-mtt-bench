"""Monte Carlo benchmark of GM-PHD, GM-CPHD, delta-GLMB and GNN on the default scenario.

Usage::

    rfsmtt-bench --runs 20 --out results
    python -m rfsmtt.benchmark --runs 5 --conditions nominal
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .glmb import GLMB  # noqa: E402
from .gmcphd import GMCPHD  # noqa: E402
from .gmphd import GMPHD  # noqa: E402
from .gnn import GNNTracker  # noqa: E402
from .metrics import label_switches, ospa, ospa2_curve, tracks_from_estimates  # noqa: E402
from .scenario import Scenario, default_scenario  # noqa: E402

CONDITIONS = {
    "nominal": {"p_d": 0.95, "clutter_rate": 10.0},
    "hard": {"p_d": 0.80, "clutter_rate": 30.0},
}
FILTERS = ["GM-PHD", "GM-CPHD", "GLMB", "GNN"]
COLORS = {"GM-PHD": "#4C72B0", "GM-CPHD": "#55A868", "GLMB": "#C44E52", "GNN": "#8172B2"}


def make_tracker(name: str, sc: Scenario, seed: int):
    if name == "GM-PHD":
        return GMPHD(sc.model, sc.clutter, sc.birth)
    if name == "GM-CPHD":
        return GMCPHD(sc.model, sc.clutter, sc.birth)
    if name == "GLMB":
        return GLMB(sc.model, sc.clutter, sc.birth, h_upd=1000, h_max=1000, rng=seed)
    if name == "GNN":
        return GNNTracker(sc.model, confirm_m=4, confirm_n=5, init_unmeasured_std=25.0)
    raise ValueError(f"unknown tracker {name!r}")


def evaluate(sc: Scenario, estimates, cutoff: float = 100.0, window: int = 10) -> dict:
    true_pos = [sc.model.positions(sc.truth_at(k)[1]) for k in range(sc.n_scans)]
    est_pos = [sc.model.positions(e.states) for e in estimates]
    per_scan = np.array([ospa(e, t, c=cutoff)[0] for e, t in zip(est_pos, true_pos)])
    est_tracks = tracks_from_estimates([e.labels for e in estimates], est_pos, sc.n_scans)
    true_tracks = sc.true_position_tracks()
    return {
        "ospa": per_scan,
        "card": np.array([len(e) for e in estimates]),
        "ospa2": ospa2_curve(est_tracks, true_tracks, sc.n_scans, window=window, c=cutoff),
        "switches": label_switches(est_tracks, true_tracks, gate=cutoff / 2),
        "tracks": len(est_tracks),
    }


def run_condition(cfg: dict, runs: int, seed: int, filters, cutoff: float, window: int):
    sc = default_scenario(**cfg)
    keys = ["ospa", "card", "ospa2", "switches", "tracks", "ms_per_scan"]
    results = {f: {k: [] for k in keys} for f in filters}
    examples, example_meas = {}, None
    for r in range(runs):
        Z = sc.measurements(seed + r)
        if r == 0:
            example_meas = Z
        for name in filters:
            tracker = make_tracker(name, sc, seed + r)
            t0 = perf_counter()
            est = [tracker.step(z, k) for k, z in enumerate(Z)]
            elapsed = perf_counter() - t0
            ev = evaluate(sc, est, cutoff, window)
            ev["ms_per_scan"] = 1000 * elapsed / sc.n_scans
            for k in keys:
                results[name][k].append(ev[k])
            if r == 0:
                examples[name] = est
    for name in filters:
        for k in keys:
            results[name][k] = np.array(results[name][k])
    return sc, results, examples, example_meas


def summarize(sc: Scenario, results: dict) -> dict:
    true_card = sc.cardinality()
    out = {}
    for name, r in results.items():
        out[name] = {
            "mean_ospa": float(r["ospa"].mean()),
            "mean_ospa2": float(r["ospa2"].mean()),
            "card_mae": float(np.abs(r["card"] - true_card).mean()),
            "id_switches": float(r["switches"].mean()),
            "tracks": float(r["tracks"].mean()),
            "ms_per_scan": float(r["ms_per_scan"].mean()),
        }
    return out


# ------------------------------------------------------------------------ plots
def plot_scenario(sc: Scenario, meas, path: Path):
    fig, ax = plt.subplots(figsize=(6.2, 6))
    allz = np.vstack(meas)
    ax.scatter(allz[:, 0], allz[:, 1], s=2, c="0.75", label="measurements (incl. clutter)")
    for lab, traj in sc.true_position_tracks().items():
        ok = np.isfinite(traj[:, 0])
        ax.plot(traj[ok, 0], traj[ok, 1], "k-", lw=1.4)
        ax.plot(traj[ok, 0][0], traj[ok, 1][0], "ko", ms=4)
        ax.annotate(lab, traj[ok][-1], fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set(
        xlim=(-1000, 1000),
        ylim=(-1000, 1000),
        xlabel="x (m)",
        ylabel="y (m)",
        aspect="equal",
        title="Scenario: 10 targets, births, deaths and crossings",
    )
    ax.legend(loc="lower right", fontsize=8, markerscale=4)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_tracks(sc: Scenario, estimates, path: Path, title: str):
    fig, ax = plt.subplots(figsize=(6.2, 6))
    for traj in sc.true_position_tracks().values():
        ok = np.isfinite(traj[:, 0])
        ax.plot(traj[ok, 0], traj[ok, 1], "-", color="0.8", lw=4)
    est_pos = [sc.model.positions(e.states) for e in estimates]
    tracks = tracks_from_estimates([e.labels for e in estimates], est_pos, sc.n_scans)
    cmap = plt.get_cmap("tab20")
    for i, traj in enumerate(tracks.values()):
        ok = np.isfinite(traj[:, 0])
        ax.plot(traj[ok, 0], traj[ok, 1], ".", ms=3, color=cmap(i % 20))
    ax.set(
        xlim=(-1000, 1000), ylim=(-1000, 1000), xlabel="x (m)", ylabel="y (m)", aspect="equal", title=title
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_metric(results: dict, key: str, ylabel: str, title: str, path: Path, truth=None):
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    k = np.arange(next(iter(results.values()))[key].shape[1])
    if truth is not None:
        ax.step(k, truth, "k-", where="mid", lw=1.5, label="truth")
    for name, r in results.items():
        mean = r[key].mean(axis=0)
        ax.plot(k, mean, color=COLORS.get(name), lw=1.6, label=name)
        if key == "card":
            sd = r[key].std(axis=0)
            ax.fill_between(k, mean - sd, mean + sd, color=COLORS.get(name), alpha=0.15)
    ax.set(xlabel="scan", ylabel=ylabel, title=title, xlim=(0, k[-1]))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=5)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def markdown_table(summary: dict) -> str:
    lines = []
    for cond, rows in summary.items():
        cfg = CONDITIONS[cond]
        lines.append(f"**{cond}** (P_D = {cfg['p_d']}, clutter = {cfg['clutter_rate']:.0f} per scan)\n")
        lines.append(
            "| Tracker | Mean OSPA (m) | Mean OSPA(2) (m) | Cardinality MAE | ID switches | "
            "Tracks reported | ms / scan |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for name, s in rows.items():
            lines.append(
                f"| {name} | {s['mean_ospa']:.1f} | {s['mean_ospa2']:.1f} | {s['card_mae']:.2f} | "
                f"{s['id_switches']:.1f} | {s['tracks']:.1f} | {s['ms_per_scan']:.1f} |"
            )
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=10, help="Monte Carlo runs per condition")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    ap.add_argument("--filters", nargs="+", default=FILTERS, choices=FILTERS)
    ap.add_argument("--cutoff", type=float, default=100.0, help="OSPA cut-off c (m)")
    ap.add_argument("--window", type=int, default=10, help="OSPA(2) window length (scans)")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    summary = {}
    for cond in args.conditions:
        print(f"[{cond}] {args.runs} runs ...", flush=True)
        sc, results, examples, meas = run_condition(
            CONDITIONS[cond], args.runs, args.seed, args.filters, args.cutoff, args.window
        )
        summary[cond] = summarize(sc, results)
        if cond == args.conditions[0]:
            plot_scenario(sc, meas, args.out / "scenario.png")
        plot_metric(
            results, "ospa", "OSPA (m)", f"Mean OSPA per scan ({cond})", args.out / f"ospa_{cond}.png"
        )
        plot_metric(
            results,
            "ospa2",
            f"OSPA(2) (m), window {args.window}",
            f"Mean OSPA(2) over a sliding window ({cond})",
            args.out / f"ospa2_{cond}.png",
        )
        plot_metric(
            results,
            "card",
            "number of objects",
            f"Cardinality, mean ± 1 sd ({cond})",
            args.out / f"cardinality_{cond}.png",
            truth=sc.cardinality(),
        )
        for name in ("GLMB", "GM-PHD"):
            if name in examples:
                plot_tracks(
                    sc,
                    examples[name],
                    args.out / f"tracks_{name.lower()}_{cond}.png",
                    f"{name} estimates, one run ({cond}); colour = label",
                )

    table = markdown_table(summary)
    (args.out / "summary.md").write_text(table)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(table)


if __name__ == "__main__":
    main()
