"""Multi-object performance metrics.

* OSPA (Schuhmacher, Vo & Vo, IEEE TSP 2008): per-scan distance between sets of states.
* OSPA(2) (Beard, Vo & Vo, IEEE TSP 2020): distance between sets of *tracks* over a time
  window. It penalises track fragmentation, identity switches and false or missed tracks,
  which per-scan OSPA cannot see.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def ospa(X, Y, c: float = 100.0, p: float = 1.0):
    """OSPA distance between two point sets.

    Returns ``(total, localisation, cardinality)`` components.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    X = X.reshape(len(X), -1) if X.size else np.zeros((0, 1))
    Y = Y.reshape(len(Y), -1) if Y.size else np.zeros((0, 1))
    n, m = len(X), len(Y)
    if n == 0 and m == 0:
        return 0.0, 0.0, 0.0
    if n == 0 or m == 0:
        return float(c), 0.0, float(c)
    D = np.minimum(cdist(X, Y), c) ** p
    rows, cols = linear_sum_assignment(D)
    cost = D[rows, cols].sum()
    N = max(n, m)
    card = c**p * abs(n - m)
    return (
        float(((cost + card) / N) ** (1 / p)),
        float((cost / N) ** (1 / p)),
        float((card / N) ** (1 / p)),
    )


def tracks_from_estimates(labels_per_scan: Sequence[Sequence[Hashable]], positions_per_scan, n_scans: int):
    """Convert per-scan labelled estimates into ``{label: (n_scans, d) array}`` with NaN gaps."""
    tracks: dict = {}
    dim = None
    for pos in positions_per_scan:
        pos = np.asarray(pos)
        if pos.size:
            dim = pos.reshape(len(pos), -1).shape[1]
            break
    if dim is None:
        return tracks
    for k, (labels, pos) in enumerate(zip(labels_per_scan, positions_per_scan)):
        pos = np.asarray(pos, dtype=float).reshape(-1, dim)
        for lab, x in zip(labels, pos):
            if lab not in tracks:
                tracks[lab] = np.full((n_scans, dim), np.nan)
            tracks[lab][k] = x
    return tracks


def _stack(tracks: dict, t0: int, t1: int) -> np.ndarray:
    arrs = [a[t0:t1] for a in tracks.values()]
    arrs = [a for a in arrs if np.isfinite(a[:, 0]).any()]
    if not arrs:
        return np.zeros((0, t1 - t0, 1))
    return np.stack(arrs)


def ospa2(
    est_tracks: dict,
    true_tracks: dict,
    t0: int = 0,
    t1: int | None = None,
    c: float = 100.0,
    p: float = 1.0,
    q: float = 1.0,
) -> float:
    """OSPA(2) distance between estimated and true tracks over scans ``[t0, t1)``."""
    if t1 is None:
        lengths = [len(a) for a in list(est_tracks.values()) + list(true_tracks.values())]
        t1 = max(lengths) if lengths else 0
    E = _stack(est_tracks, t0, t1)
    T = _stack(true_tracks, t0, t1)
    ne, nt = len(E), len(T)
    if ne == 0 and nt == 0:
        return 0.0
    if ne == 0 or nt == 0:
        return float(c)

    e_exist = np.isfinite(E[:, :, 0])[:, None, :]
    t_exist = np.isfinite(T[:, :, 0])[None, :, :]
    both = e_exist & t_exist
    either = e_exist | t_exist
    with np.errstate(invalid="ignore"):
        d = np.linalg.norm(E[:, None] - T[None, :], axis=-1)
    per_time = np.where(both, np.minimum(np.nan_to_num(d, nan=c), c) ** q, np.where(either, c**q, 0.0))
    base = (per_time.sum(axis=-1) / either.sum(axis=-1)) ** (1 / q)

    D = base**p
    rows, cols = linear_sum_assignment(D)
    cost = D[rows, cols].sum()
    N = max(ne, nt)
    return float(((cost + c**p * abs(ne - nt)) / N) ** (1 / p))


def ospa2_curve(est_tracks: dict, true_tracks: dict, n_scans: int, window: int = 10, **kwargs) -> np.ndarray:
    """OSPA(2) over a sliding window ending at each scan."""
    return np.array(
        [ospa2(est_tracks, true_tracks, max(0, k - window + 1), k + 1, **kwargs) for k in range(n_scans)]
    )


def label_switches(est_tracks: dict, true_tracks: dict, gate: float) -> int:
    """Identity switches counted as in the CLEAR MOT metrics (Bernardin & Stiefelhagen, 2008).

    At each scan, a true track keeps its previous estimated label if that label is still
    within ``gate``; the remaining true and estimated tracks are matched optimally within
    ``gate``. A switch is counted whenever a true track is matched to a different label from
    the one it was last matched to. Keeping existing correspondences first avoids counting
    spurious switches when objects pass through the same point.
    """
    if not est_tracks or not true_tracks:
        return 0
    e_labels, E = list(est_tracks), np.stack(list(est_tracks.values()))
    t_labels, T = list(true_tracks), np.stack(list(true_tracks.values()))
    e_index = {lab: i for i, lab in enumerate(e_labels)}
    last: dict = {}
    switches = 0
    for k in range(T.shape[1]):
        ei = set(np.flatnonzero(np.isfinite(E[:, k, 0])).tolist())
        ti = np.flatnonzero(np.isfinite(T[:, k, 0]))
        if not ei or len(ti) == 0:
            continue
        free_t = []
        for t in ti:
            prev = last.get(t_labels[t])
            j = e_index.get(prev) if prev is not None else None
            if j in ei and np.linalg.norm(T[t, k] - E[j, k]) <= gate:
                ei.discard(j)
            else:
                free_t.append(t)
        if not free_t or not ei:
            continue
        free_e = sorted(ei)
        D = cdist(T[free_t, k], E[free_e, k])
        rows, cols = linear_sum_assignment(np.minimum(D, gate * 10))
        for r, c_ in zip(rows, cols):
            if D[r, c_] > gate:
                continue
            tl, el = t_labels[free_t[r]], e_labels[free_e[c_]]
            if tl in last and last[tl] != el:
                switches += 1
            last[tl] = el
    return switches
