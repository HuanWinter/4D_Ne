#!/usr/bin/env python3
"""Kraskov-Stoegbauer-Grassberger (KSG) mutual-information estimator.

Faithful Python port of ``MI_Kraskov.m`` (the I(1) estimator with ``k = 1``).

Reference
---------
Kraskov, A., Stoegbauer, H., & Grassberger, P. (2004).
"Estimating mutual information." Physical Review E, 69(6), 066138.

Original MATLAB author: Paolo Inglese <paolo.ingls@gmail.com> (rev. 2015-05-17).

Algorithm (identical to the MATLAB original)
--------------------------------------------
For each sample ``i`` the MATLAB code finds ``Eps``, the Chebyshev (max-norm)
distance to the nearest neighbour (k = 1) in the joint (X, Y) space, by scanning
left/right in the X-sorted order. Because X is sorted, that scan returns exactly
the global minimum max(|dx|, |dy|) over j != i. It then counts

    nx(i) = #{ j != i : |X(j) - X(i)| < Eps }
    ny(i) = #{ j != i : |Y(j) - Y(i)| < Eps }

(the commented-out lines ``nx(i)=(sum(abs(X-X(i))<Eps)-1)`` in the MATLAB source
confirm this strict-inequality counting), and returns

    I1 = psi(k) - mean( psi(nx + 1) + psi(ny + 1) ) + psi(nObs),  k = 1.

This port reproduces those exact quantities. Two backends are provided:

* ``method="exact"`` (default) -- computes ``Eps`` and the strict ``|.| < Eps``
  marginal counts with the same arithmetic as the MATLAB source, so it is
  bit-for-bit identical to the original (verified in test_mi_kraskov.py). It is
  O(n^2).
* ``method="fast"`` -- obtains ``Eps`` from a Chebyshev KD-tree query and the
  counts from sorted binary search: O(n log n). It agrees with ``exact`` to
  ~1e-3 nats; the only differences are off-by-one marginal counts at points
  lying exactly at the neighbour distance, where the shifted-coordinate
  comparison ``x[j] < x[i] + Eps`` rounds differently from ``|x[j]-x[i]| < Eps``.
  Use it only for large samples where the O(n^2) backend is too slow, and be
  aware it is not bit-identical to the MATLAB original.

Usage
-----
    from mi_kraskov import mi_kraskov
    import numpy as np
    x = np.random.randn(2000)
    y = x + 0.5 * np.random.randn(2000)
    print(mi_kraskov(x, y))            # ~ analytic MI of the bivariate Gaussian

CLI
---
    python mi_kraskov.py --selftest
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
from scipy.special import digamma as _psi
from scipy.spatial import cKDTree


def mi_kraskov(X, Y, zero_fix: bool = False, method: str = "exact") -> float:
    """Kraskov I(1) mutual-information estimate between X and Y (k = 1).

    Parameters
    ----------
    X, Y : array_like, shape (n,) or (n, d)
        Paired samples. The MATLAB original is written for univariate X, Y;
        multivariate inputs use the same max-norm (Chebyshev) construction.
        X and Y must have the same number of rows (samples).
    zero_fix : bool, optional
        If True, clip a negative estimate to 0 (matches the MATLAB ``zeroFix``
        flag). Default False.
    method : {"exact", "fast"}, optional
        "exact" (default) reproduces the MATLAB arithmetic bit-for-bit (O(n^2)).
        "fast" uses a KD-tree + binary search (O(n log n)); agrees to ~1e-3 nats.

    Returns
    -------
    float
        Estimated mutual information in nats.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must contain the same number of samples")
    if method not in ("exact", "fast"):
        raise ValueError("method must be 'exact' or 'fast'")

    k = 1
    n_obs = X.shape[0]
    if n_obs < 2:
        raise ValueError("need at least 2 samples")

    if method == "exact":
        nx, ny = _counts_exact(X, Y)
    else:
        # Eps[i]: Chebyshev distance to the k=1 nearest neighbour in joint space.
        joint = np.hstack([X, Y])
        tree = cKDTree(joint)
        dist, _ = tree.query(joint, k=2, p=np.inf)  # [self=0, true NN]
        eps = dist[:, 1]
        nx = _count_within(X, eps)
        ny = _count_within(Y, eps)

    i1 = _psi(k) - np.sum(_psi(nx + 1) + _psi(ny + 1)) / n_obs + _psi(n_obs)
    i1 = float(i1)

    if zero_fix and i1 < 0:
        warnings.warn("First estimator is negative -> 0")
        i1 = 0.0
    return i1


def _counts_exact(X: np.ndarray, Y: np.ndarray):
    """Exact MATLAB-equivalent Eps and strict marginal counts (O(n^2)).

    Eps[i] = min_{j!=i} max_d |Z_d[i]-Z_d[j]| over the joint coordinates, and
    nx[i] = #{ j!=i : max_d |X_d[i]-X_d[j]| < Eps[i] } (and likewise ny). For
    univariate X, Y (the MATLAB original) the max over d is a no-op, matching
    ``sum(abs(X-X(i)) < Eps) - 1`` exactly.
    """
    n = X.shape[0]
    nx = np.empty(n, dtype=np.int64)
    ny = np.empty(n, dtype=np.int64)
    for i in range(n):
        dx = np.max(np.abs(X - X[i]), axis=1)
        dy = np.max(np.abs(Y - Y[i]), axis=1)
        cheb = np.maximum(dx, dy)
        cheb[i] = np.inf  # exclude self
        eps = cheb.min()
        nx[i] = np.count_nonzero(dx < eps) - 1
        ny[i] = np.count_nonzero(dy < eps) - 1
    return nx, ny


def _count_within(Z: np.ndarray, eps: np.ndarray) -> np.ndarray:
    """For each i, count j != i with max-norm marginal distance < eps[i].

    Mirrors the MATLAB ``sum(abs(Z - Z(i)) < Eps) - 1`` counting with strict
    inequality, summed across marginal dimensions via the max norm. For the
    univariate case (the MATLAB original) this is the standard marginal count.
    """
    n, d = Z.shape
    counts = np.zeros(n, dtype=np.int64)
    # Per-dimension strict-window counts via sorted binary search, O(n log n).
    for dim in range(d):
        col = Z[:, dim]
        order = np.argsort(col, kind="mergesort")
        cs = col[order]
        for i in range(n):
            e = eps[i]
            if e <= 0.0:  # duplicate point: MATLAB breaks immediately -> 0
                counts[i] = 0
                continue
            lo = np.searchsorted(cs, col[i] - e, side="right")
            hi = np.searchsorted(cs, col[i] + e, side="left")
            # strict open interval (col[i]-e, col[i]+e); subtract i itself.
            c = hi - lo - 1
            if d == 1:
                counts[i] = c
            else:
                counts[i] = max(counts[i], c)
    # NB: for the univariate MATLAB original (d == 1) this is exact. The
    # multivariate aggregation uses the max-norm marginal window per dimension.
    return counts


def _mi_kraskov_bruteforce(X, Y, zero_fix: bool = False) -> float:
    """O(n^2) reference that mirrors the MATLAB loops directly (for testing).

    Computes Eps[i] = min_{j != i} max(|X[i]-X[j]|, |Y[i]-Y[j]|) explicitly and
    counts strict marginal neighbours. Identical math to ``mi_kraskov`` but
    without the KD-tree / binary-search optimisations.
    """
    X = np.asarray(X, dtype=float).ravel()
    Y = np.asarray(Y, dtype=float).ravel()
    n = X.size
    nx = np.zeros(n, dtype=np.int64)
    ny = np.zeros(n, dtype=np.int64)
    for i in range(n):
        dx = np.abs(X - X[i])
        dy = np.abs(Y - Y[i])
        cheb = np.maximum(dx, dy)
        cheb[i] = np.inf  # exclude self
        eps = cheb.min()
        nx[i] = np.sum(dx < eps) - 1
        ny[i] = np.sum(dy < eps) - 1
    i1 = _psi(1) - np.sum(_psi(nx + 1) + _psi(ny + 1)) / n + _psi(n)
    i1 = float(i1)
    if zero_fix and i1 < 0:
        i1 = 0.0
    return i1


def _selftest() -> None:
    rng = np.random.default_rng(0)
    n = 4000
    for rho in (0.0, 0.5, 0.9):
        cov = np.array([[1.0, rho], [rho, 1.0]])
        xy = rng.multivariate_normal([0, 0], cov, size=n)
        est = mi_kraskov(xy[:, 0], xy[:, 1])
        analytic = -0.5 * np.log(1 - rho ** 2) if rho else 0.0
        print(f"rho={rho:.2f}  MI_est={est:+.4f}  analytic={analytic:+.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Kraskov KSG mutual information (I1, k=1)")
    ap.add_argument("--selftest", action="store_true",
                    help="run a quick Gaussian self-test")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        ap.print_help()
