# Design notes: rfs-mtt-bench

This document records the background, the design decisions and the reasons behind them, the
parameter values used, how the code was validated, and the sources it builds on.

## 1. Purpose and scope

The repository provides three random finite set (RFS) multi-target filters and a classical
baseline behind one interface, plus a benchmark that compares them fairly. The aim is clarity
and correctness rather than speed: everything is NumPy/SciPy, and every tracker is small enough
to read in one sitting.

Out of scope: non-linear measurement models (EKF/UKF variants), manoeuvring-target models (IMM),
multi-sensor fusion and compiled implementations.

## 2. Background

### 2.1 Multi-target state as a random finite set

At scan $k$ the multi-target state is a finite set $X_k = \{x_1, \dots, x_n\}$ whose cardinality
$n$ is itself random, and the scan's measurements form another finite set $Z_k$ that mixes
target-originated detections (each target detected with probability $p_D$) and clutter. The
multi-target Bayes filter propagates the full posterior over such sets (Mahler 2007). It is
intractable in general; the filters here are principled approximations of it.

### 2.2 PHD filter

The probability hypothesis density (PHD) is the first moment (intensity) $D(x)$ of the
multi-target posterior: its integral over a region is the expected number of targets there.
Propagating only the PHD under Poisson assumptions gives the PHD recursion (Mahler 2003). With
linear-Gaussian models and Gaussian-mixture birth, the PHD stays a Gaussian mixture (Vo & Ma 2006):

$$
D_{k|k}(x) = (1-p_D)\,D_{k|k-1}(x) + \sum_{z\in Z_k}\sum_i
\frac{p_D\, w_i\, q_i(z)}{\kappa(z) + p_D \sum_j w_j\, q_j(z)}\;\mathcal N\big(x;\, m_i(z), P_i\big),
$$

where $q_i(z)=\mathcal N(z; H m_i, H P_i H^\top + R)$ and $\kappa(z)$ is the clutter intensity.

**Weakness:** the PHD carries one number of cardinality information (the mean), so its count
estimate has high variance and drops sharply after missed detections. It also has no notion of
identity.

### 2.3 CPHD filter

The cardinalized PHD (Mahler 2007) propagates the intensity *and* the full cardinality
distribution $\rho(n)$. In the Gaussian-mixture form (Vo, Vo & Cantoni 2007) the update uses
elementary symmetric functions $e_j$ of the per-measurement terms
$\Xi_z = \langle 1,\kappa\rangle\, p_D \langle w, q_z\rangle / \kappa(z)$:

$$
\Upsilon^u[w,Z](n) = \sum_{j=0}^{\min(|Z|,\,n-u)} (|Z|-j)!\,p_K(|Z|-j)\;
\frac{n!}{(n-j-u)!}\;\frac{(1-p_D)^{\,n-j-u}}{\langle 1, w\rangle^{j+u}}\; e_j(\Xi),
\qquad \rho_{k}(n) \propto \Upsilon^0(n)\,\rho_{k|k-1}(n).
$$

Missed-detection components are scaled by
$\langle \Upsilon^1, \rho_{k|k-1}\rangle / \langle \Upsilon^0, \rho_{k|k-1}\rangle$, and the
component for measurement $z$ uses $\Upsilon^1$ computed on $Z \setminus \{z\}$.

### 2.4 Labeled RFS and the δ-GLMB filter

Labeled RFS (Vo & Vo 2013) attach a unique label $\ell$ to each target, so identity is part of
the state and tracks come out of the filter rather than from post-processing. The
generalized labeled multi-Bernoulli (GLMB) family is closed under the Bayes recursion; the
δ-GLMB form represents the posterior as weighted hypotheses, each being a set of labels with a
single-target density per label (Vo, Vo & Phung 2014).

The efficient implementation (Vo, Vo & Hoang 2017) performs prediction and update in one step.
For each hypothesis, every candidate track (surviving or newly born) takes one of three roles,
with weights:

| Role | Surviving track | Birth candidate |
|---|---|---|
| does not exist | $1-p_S$ | $1-r_B$ |
| exists, missed | $p_S(1-p_D)$ | $r_B(1-p_D)$ |
| exists, detected by $z_j$ | $p_S\, p_D\, q(z_j)/\kappa(z_j)$ | $r_B\, p_D\, q(z_j)/\kappa(z_j)$ |

A new hypothesis is a joint choice of roles in which no measurement is used twice; its weight is
the parent weight times the product of the chosen entries. Rather than enumerating the best
assignments with Murty's algorithm, a Gibbs sampler draws them, which scales linearly in the
number of tracks and measurements per sweep.

### 2.5 Metrics

- **OSPA** (Schuhmacher, Vo & Vo 2008) combines localisation and cardinality error between two
  point sets, with cut-off $c$ and order $p$. It is computed per scan and is blind to identity.
- **OSPA(2)** (Beard, Vo & Vo 2017, 2020) applies OSPA to *sets of tracks*, using a time-averaged
  distance between tracks over a window. Fragmentation, identity switches and false or missed
  tracks all raise it.
- **Identity switches** follow the CLEAR MOT convention (Bernardin & Stiefelhagen 2008).

## 3. Architecture

```
scenario.py  --> truth trajectories, cluttered detections per scan
models.py    --> F, Q, H, R, p_S, p_D; Poisson clutter; fixed or adaptive birth
kalman.py    --> vectorised predict / update / gating, shared by every tracker
gmphd.py, gmcphd.py, glmb.py, gnn.py --> step(Z, k) -> Estimate(states, labels)
metrics.py   --> OSPA, OSPA(2), identity switches
benchmark.py --> Monte Carlo runs, tables, figures
```

Every tracker implements `step(Z, k) -> Estimate`, so scenario generation, metrics and plotting
never depend on the tracker type.

## 4. Design decisions

### 4.1 Shared, vectorised Kalman algebra
All trackers call the same `predict`/`update`, which operate on stacks of components
(`(n, dx, dx)` covariances) and every measurement at once. One correct implementation is
reused everywhere, and the GLMB track table and the PHD mixtures update in a single call.
The covariance update uses the Joseph form, $(I-KH)P(I-KH)^\top + KRK^\top$, which keeps
covariances symmetric positive definite under round-off.

### 4.2 Gating
Likelihoods outside a 99.99% $\chi^2$ gate are set to zero. This makes the GLMB cost matrices
sparse (columns with no gated track are dropped before sampling) and prevents distant clutter
from creating negligible components in the PHD filters. The gate is set wide enough that it
very rarely removes a true association.

### 4.3 PHD identities by tag propagation
PHD filters do not model identity. To score them with OSPA(2), each Gaussian component carries
a tag (Panta, Clark & Vo 2009): updated components inherit their parent's tag, and a merge keeps
the tag of the heaviest component.

*Problem found during development:* one birth component can spawn two targets in the same scan
(targets T04 and T05 are born at the same point at scan 20). Both children inherited the same
tag, so two targets shared one identity. The fix: after pruning and merging, any significant
component (weight above the extraction threshold) that shares a tag with a heavier one gets a
fresh tag. With the original switch counter, GM-PHD switches fell from 61 to about 13 per run.

### 4.4 CPHD in the log domain
$e_j(\Xi)$ grows like $\Xi^j$ and overflows for many measurements or large $\Xi$. The values are
divided by their maximum before the recursion and the scale is restored in log space;
$\Upsilon$ terms are combined with `logsumexp`, and `xlogy` handles $p_D = 1$ and zero clutter
without producing `0 * log 0` NaNs. A first version looped over $n$ and ran at about 114 ms per
scan; vectorising $\Upsilon$ over $(n, j)$ brought it to about 8 ms with identical output.

### 4.5 GLMB: joint prediction-update with Gibbs sampling
- **Sample allocation:** each parent hypothesis receives a number of Gibbs sweeps proportional
  to the square root of its weight (at least one), as in Vo, Vo & Hoang (2017). This spreads
  effort beyond the single best hypothesis without wasting it on negligible ones.
- **Initial sample:** "all tracks exist and are missed", so a sampler that never moves still
  returns a valid hypothesis.
- **Duplicates:** the same child hypothesis can arise from different parents; their weights are
  summed (`logaddexp`) under a canonical key (sorted track-table indices).
- **Track table:** updated tracks are indexed as `(measurement index) * (number of predicted
  tracks) + (predicted track index)`, which encodes parent and association in one integer.
  After pruning, the table is rebuilt from referenced indices only, so it stays proportional to
  the hypotheses kept.
- **Estimation:** the maximum a posteriori cardinality is found from the hypothesis weights, and
  the highest-weight hypothesis with that cardinality supplies the tracks. This is the standard
  estimator; it can report short-lived tracks that appear in the best hypothesis for one scan.

### 4.6 Adaptive (measurement-driven) birth
Fixed birth locations suit a benchmark but not real data. `AdaptiveBirth` places a candidate at
each measurement of the previous scan with existence probability
$r_B(z) = \min\big(r_{\max},\, \lambda_B\,(1-r_U(z)) / \sum_{z'} (1-r_U(z'))\big)$, where $r_U(z)$
is the probability that $z$ was associated with an existing track (Reuter et al. 2014; Ristic et
al. 2012). Unmeasured state components (velocity when only position is measured) get a broad
prior via the null space of $H$. All three RFS filters supply $r_U$, so they share one birth
model.

### 4.7 GNN baseline
GNN solves one optimal 2-D assignment per scan (Hungarian algorithm on negative
log-likelihoods within a 99.9% gate) and runs a Kalman filter per track, with M-of-N
confirmation and deletion after consecutive misses. *Tuning:* the textbook 2-of-3 logic with a
50 m/s initial velocity spread confirmed clutter tracks constantly (about 350 reported tracks per
run for 10 targets). 4-of-5 confirmation with a 25 m/s spread made it a fair baseline (about 12 tracks). The
benchmark uses those values; the class defaults are 3-of-4.

### 4.8 Metric choices
- **OSPA(2) over a sliding 10-scan window, averaged over time,** rather than one value over the
  whole run. A single-window OSPA(2) over 100 scans lets a one-scan false track weigh as much as
  a 100-scan true track; the sliding window reflects what an operator sees.
- **CLEAR MOT identity switches.** The first switch counter re-matched truth to estimates from
  scratch every scan and counted switches when targets passed through the same point (the
  three-way crossing at scan 40), even when the tracker kept identities. Keeping an existing
  correspondence while it stays within the gate, before solving the remaining assignment,
  removed those artefacts.
- **Cut-off $c$ = 100 m, order $p$ = 1**, following the original GM-PHD benchmark; the switch gate
  is $c/2$.

### 4.9 Scenario design
Ten targets in a 2 km × 2 km region over 100 scans; four birth locations; births at scans 0,
20, 40 and 60; two deaths at scan 70. It deliberately includes a three-way crossing at scan 40
(T01, T02 and T03 pass through (0, −400) m) and two targets born at the same point in the same
scan (T04, T05). Two conditions: nominal ($p_D$ = 0.95, 10 false alarms per scan) and hard
($p_D$ = 0.80, 30 per scan). Truth is fixed; only measurements change between Monte Carlo runs.

## 5. Parameters

| Item | Value |
|---|---|
| Motion model | nearly constant velocity, $q$ = 1 m²/s³, $T$ = 1 s |
| Measurement | position, σ = 10 m |
| $p_S$ | 0.99 |
| Birth (benchmark) | 4 Gaussians, weight 0.03 each, σ = (10 m, 20 m/s) per axis |
| GM-PHD / GM-CPHD | prune 1e-5, merge 4 (squared Mahalanobis), cap 100, extract weight > 0.5, gate 99.99% |
| GM-CPHD | $N_{\max}$ = 60 |
| GLMB | 1000 Gibbs sweeps per scan in total, at most 1000 hypotheses, prune 1e-15, gate 99.99% |
| GNN (benchmark) | gate 99.9%, confirm 4 of 5, delete after 3 misses, initial velocity σ 25 m/s |
| Metrics | OSPA $c$ = 100 m, $p$ = 1; OSPA(2) window 10 scans; switch gate 50 m |

## 6. Validation

Unit tests check each building block against an independent reference:

- Kalman update against the closed form and `scipy.stats.multivariate_normal`; gating.
- Elementary symmetric functions against brute-force sums over combinations; the log-domain
  version at extreme values.
- Mixture merging preserves mass and moments; capping preserves total weight.
- Gibbs samples are unique, never reuse a measurement and include the initial solution.
- OSPA is zero for identical sets, equals $c$ against an empty set, and is symmetric; OSPA(2)
  is zero for identical tracks and positive after an ID switch.

Integration tests run every tracker on a two-target scenario (OSPA below 15 m; correct count
more than 80% of the time), check that GLMB and GNN keep two identities, that GLMB weights and
cardinality distributions stay normalised, and that adaptive birth initiates a target anywhere.
The benchmark itself (20 runs per condition) is the system-level check.

## 7. Known limitations and next steps

- Linear-Gaussian models only; adding EKF/UKF updates would allow range-bearing measurements.
- Pure Python: GLMB runs at 75–91 ms per scan on the machine described in the README
  (100–180 times GNN). A C++ core for the Gibbs sampler and Kalman updates, or Numba, is the
  obvious speed-up.
- Fixed $p_S$ and $p_D$; state-dependent detection (sensor footprint) and survival (region
  boundary) would make the models more realistic for real sensor data.
- The GLMB estimator reports the best hypothesis each scan; track-based estimators (for example,
  reporting tracks by existence probability over a window) would suppress one-scan false tracks.
- Only the fixed-birth benchmark is reported; an adaptive-birth version of the benchmark would be
  more representative of real data.

## 8. Sources

| Source | Used for |
|---|---|
| R. Mahler, "Multitarget Bayes filtering via first-order multitarget moments," *IEEE TAES* 39(4), 1152–1178, 2003 ([doi](https://doi.org/10.1109/TAES.2003.1261119)) | PHD recursion |
| R. Mahler, "PHD filters of higher order in target number," *IEEE TAES* 43(4), 1523–1543, 2007 ([doi](https://doi.org/10.1109/TAES.2007.4441756)) | CPHD recursion |
| R. Mahler, *Statistical Multisource-Multitarget Information Fusion*, Artech House, 2007 | RFS background |
| B.-N. Vo and W.-K. Ma, "The Gaussian mixture probability hypothesis density filter," *IEEE TSP* 54(11), 4091–4104, 2006 ([doi](https://doi.org/10.1109/TSP.2006.881190)) | GM-PHD, pruning and merging, benchmark style |
| B.-T. Vo, B.-N. Vo and A. Cantoni, "Analytic implementations of the cardinalized probability hypothesis density filter," *IEEE TSP* 55(7), 3553–3567, 2007 ([doi](https://doi.org/10.1109/TSP.2007.894241)) | GM-CPHD, elementary symmetric functions |
| D. Clark, K. Panta and B.-N. Vo, "The GM-PHD filter multiple target tracker," Proc. FUSION, 2006 ([doi](https://doi.org/10.1109/ICIF.2006.301809)) | Tag propagation for GM-PHD (cited in `mixture.py`) |
| K. Panta, D. Clark and B.-N. Vo, "Data association and track management for the Gaussian mixture probability hypothesis density filter," *IEEE TAES* 45(3), 1003–1016, 2009 ([doi](https://doi.org/10.1109/TAES.2009.5259179)) | Tag-based identities for GM-PHD |
| B.-T. Vo and B.-N. Vo, "Labeled random finite sets and multi-object conjugate priors," *IEEE TSP* 61(13), 3460–3475, 2013 ([doi](https://doi.org/10.1109/TSP.2013.2259822)) | Labeled RFS, GLMB |
| B.-N. Vo, B.-T. Vo and D. Phung, "Labeled random finite sets and the Bayes multi-target tracking filter," *IEEE TSP* 62(24), 6554–6567, 2014 ([doi](https://doi.org/10.1109/TSP.2014.2364014)) | δ-GLMB filter |
| B.-N. Vo, B.-T. Vo and H. G. Hoang, "An efficient implementation of the generalized labeled multi-Bernoulli filter," *IEEE TSP* 65(8), 1975–1987, 2017 ([doi](https://doi.org/10.1109/TSP.2016.2641392)) | Joint prediction-update, Gibbs sampler, sample allocation |
| S. Reuter, B.-T. Vo, B.-N. Vo and K. Dietmayer, "The labeled multi-Bernoulli filter," *IEEE TSP* 62(12), 3246–3260, 2014 ([doi](https://doi.org/10.1109/TSP.2014.2323064)) | Adaptive birth from unassociated measurements |
| B. Ristic, D. Clark, B.-N. Vo and B.-T. Vo, "Adaptive target birth intensity for PHD and CPHD filters," *IEEE TAES* 48(2), 1656–1668, 2012 ([doi](https://doi.org/10.1109/TAES.2012.6178085)) | Adaptive birth for PHD-type filters |
| D. Schuhmacher, B.-T. Vo and B.-N. Vo, "A consistent metric for performance evaluation of multi-object filters," *IEEE TSP* 56(8), 3447–3457, 2008 ([doi](https://doi.org/10.1109/TSP.2008.920469)) | OSPA |
| M. Beard, B.-T. Vo and B.-N. Vo, "OSPA(2): Using the OSPA metric to evaluate multi-target tracking performance," ICCAIS, 86–91, 2017 ([doi](https://doi.org/10.1109/ICCAIS.2017.8217598)) | OSPA(2) |
| M. Beard, B.-T. Vo and B.-N. Vo, "A solution for large-scale multi-object tracking," *IEEE TSP* 68, 2754–2769, 2020 ([doi](https://doi.org/10.1109/TSP.2020.2986136)) | OSPA(2) usage, windowed evaluation |
| K. Bernardin and R. Stiefelhagen, "Evaluating multiple object tracking performance: the CLEAR MOT metrics," *EURASIP JIVP* 2008, article 246309 ([doi](https://doi.org/10.1155/2008/246309)) | Identity-switch counting |
| S. Blackman and R. Popoli, *Design and Analysis of Modern Tracking Systems*, Artech House, 1999 | GNN, M-of-N track logic |
| Y. Bar-Shalom, X. R. Li and T. Kirubarajan, *Estimation with Applications to Tracking and Navigation*, Wiley, 2001 | Kalman filtering, Joseph form, gating, white-noise acceleration model |
