"""Pruning, merging and capping of Gaussian mixtures (Vo & Ma, 2006).

Each component carries an integer tag so that PHD-type filters can report
track identities (tag propagation, Clark et al. 2006).
"""

from __future__ import annotations

import numpy as np


def prune(w, m, P, tags, threshold: float):
    keep = w > threshold
    return w[keep], m[keep], P[keep], tags[keep]


def merge(w, m, P, tags, threshold: float):
    """Greedy moment-matching merge of components within ``threshold`` squared Mahalanobis distance."""
    if len(w) == 0:
        return w, m, P, tags
    alive = np.ones(len(w), dtype=bool)
    out_w, out_m, out_P, out_t = [], [], [], []
    while alive.any():
        cand = np.flatnonzero(alive)
        j = cand[np.argmax(w[cand])]
        d = m[cand] - m[j]
        d2 = np.einsum("ni,ij,nj->n", d, np.linalg.inv(P[j]), d)
        group = cand[d2 <= threshold]
        wg = w[group]
        W = wg.sum()
        mm = (wg[:, None] * m[group]).sum(axis=0) / W
        dm = m[group] - mm
        PP = (wg[:, None, None] * (P[group] + dm[:, :, None] * dm[:, None, :])).sum(axis=0) / W
        out_w.append(W)
        out_m.append(mm)
        out_P.append(0.5 * (PP + PP.T))
        out_t.append(tags[j])
        alive[group] = False
    return np.array(out_w), np.array(out_m), np.array(out_P), np.array(out_t)


def cap(w, m, P, tags, max_components: int):
    """Keep the heaviest components, rescaling so the total weight is preserved."""
    if len(w) <= max_components:
        return w, m, P, tags
    idx = np.argsort(-w)[:max_components]
    scale = w.sum() / w[idx].sum()
    return w[idx] * scale, m[idx], P[idx], tags[idx]
