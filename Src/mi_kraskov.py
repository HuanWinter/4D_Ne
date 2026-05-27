# Kraskov (KSG) mutual information estimator, I1 with k=1.
# Python version of MI_Kraskov.m
# Ref: Kraskov, Stoegbauer, Grassberger, PRE 69, 066138 (2004).
# method='exact' matches the MATLAB code; method='fast' uses a KD-tree (O(n log n),
# ~1e-3 nats off, for large samples).

import argparse
import warnings
import numpy as np
from scipy.special import digamma as psi
from scipy.spatial import cKDTree


def mi_kraskov(X, Y, zero_fix=False, method="exact"):
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
        joint = np.hstack([X, Y])
        tree = cKDTree(joint)
        dist, _ = tree.query(joint, k=2, p=np.inf)   # self + nearest neighbour
        eps = dist[:, 1]
        nx = _count_within(X, eps)
        ny = _count_within(Y, eps)

    i1 = float(psi(k) - np.sum(psi(nx + 1) + psi(ny + 1)) / n_obs + psi(n_obs))

    if zero_fix and i1 < 0:
        warnings.warn("First estimator is negative -> 0")
        i1 = 0.0
    return i1


# Eps[i] = nearest-neighbour Chebyshev distance, nx/ny = strict marginal counts.
# O(n^2) but matches MI_Kraskov.m exactly.
def _counts_exact(X, Y):
    n = X.shape[0]
    nx = np.empty(n, dtype=np.int64)
    ny = np.empty(n, dtype=np.int64)
    for i in range(n):
        dx = np.max(np.abs(X - X[i]), axis=1)
        dy = np.max(np.abs(Y - Y[i]), axis=1)
        cheb = np.maximum(dx, dy)
        cheb[i] = np.inf
        eps = cheb.min()
        nx[i] = np.count_nonzero(dx < eps) - 1
        ny[i] = np.count_nonzero(dy < eps) - 1
    return nx, ny


# count j!=i with marginal distance < eps[i], via sorted search (used by 'fast')
def _count_within(Z, eps):
    n, d = Z.shape
    counts = np.zeros(n, dtype=np.int64)
    for dim in range(d):
        col = Z[:, dim]
        cs = np.sort(col, kind="mergesort")
        for i in range(n):
            e = eps[i]
            if e <= 0.0:
                counts[i] = 0
                continue
            lo = np.searchsorted(cs, col[i] - e, side="right")
            hi = np.searchsorted(cs, col[i] + e, side="left")
            c = hi - lo - 1
            counts[i] = c if d == 1 else max(counts[i], c)
    return counts


# brute-force reference (mirrors the MATLAB loops), used by the tests
def _mi_kraskov_bruteforce(X, Y, zero_fix=False):
    X = np.asarray(X, dtype=float).ravel()
    Y = np.asarray(Y, dtype=float).ravel()
    n = X.size
    nx = np.zeros(n, dtype=np.int64)
    ny = np.zeros(n, dtype=np.int64)
    for i in range(n):
        dx = np.abs(X - X[i])
        dy = np.abs(Y - Y[i])
        cheb = np.maximum(dx, dy)
        cheb[i] = np.inf
        eps = cheb.min()
        nx[i] = np.sum(dx < eps) - 1
        ny[i] = np.sum(dy < eps) - 1
    i1 = float(psi(1) - np.sum(psi(nx + 1) + psi(ny + 1)) / n + psi(n))
    if zero_fix and i1 < 0:
        i1 = 0.0
    return i1


def _selftest():
    rng = np.random.default_rng(0)
    n = 4000
    for rho in (0.0, 0.5, 0.9):
        cov = np.array([[1.0, rho], [rho, 1.0]])
        xy = rng.multivariate_normal([0, 0], cov, size=n)
        est = mi_kraskov(xy[:, 0], xy[:, 1])
        analytic = -0.5 * np.log(1 - rho ** 2) if rho else 0.0
        print("rho=%.2f  MI_est=%+.4f  analytic=%+.4f" % (rho, est, analytic))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Kraskov KSG mutual information (I1, k=1)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
    else:
        ap.print_help()
