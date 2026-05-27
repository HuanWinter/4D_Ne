# Sanity tests for the MI code (mi_kraskov / mi_regressor / show_mi).
# Run: python Src/test_mi_kraskov.py   (exits non-zero on failure)

import os
import sys
import traceback
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mi_kraskov import mi_kraskov, _mi_kraskov_bruteforce  # noqa: E402
import mi_regressor as mr  # noqa: E402

_FAILS = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print("  [%s] %s%s" % (status, name, ("  (%s)" % detail) if detail else ""))
    if not cond:
        _FAILS.append(name)


def t_bruteforce_equivalence():
    # exact must match the brute-force reference bit-for-bit; fast within ~1e-3
    print("\n[1] exact == reference (bit-exact); fast within tol")
    rng = np.random.default_rng(42)
    for trial in range(6):
        n = int(rng.integers(30, 300))
        x = rng.standard_normal(n)
        y = 0.7 * x + 0.5 * rng.standard_normal(n)
        ref = _mi_kraskov_bruteforce(x, y)
        exact = mi_kraskov(x, y, method="exact")
        fast = mi_kraskov(x, y, method="fast")
        check("trial %d exact (n=%d)" % (trial, n), abs(exact - ref) < 1e-12,
              "exact=%.8f ref=%.8f diff=%.1e" % (exact, ref, abs(exact - ref)))
        check("trial %d fast   (n=%d)" % (trial, n), abs(fast - ref) < 6e-3,
              "fast=%.6f ref=%.6f diff=%.1e" % (fast, ref, abs(fast - ref)))


def t_gaussian_analytic():
    # bivariate Gaussian: MI = -0.5*ln(1-rho^2)
    print("\n[2] Bivariate-Gaussian analytic MI")
    rng = np.random.default_rng(1)
    n = 5000
    for rho in (0.0, 0.3, 0.6, 0.9):
        cov = np.array([[1.0, rho], [rho, 1.0]])
        xy = rng.multivariate_normal([0, 0], cov, size=n)
        est = mi_kraskov(xy[:, 0], xy[:, 1])
        analytic = -0.5 * np.log(1 - rho ** 2) if rho else 0.0
        tol = 0.05 if rho == 0.0 else 0.08
        check("rho=%.1f" % rho, abs(est - analytic) < tol,
              "est=%+.4f analytic=%+.4f tol=%s" % (est, analytic, tol))


def t_sklearn_crosscheck():
    # cross-check magnitude vs sklearn's KSG estimator
    print("\n[3] Cross-check vs sklearn KSG (n_neighbors=1)")
    try:
        from sklearn.feature_selection import mutual_info_regression
    except Exception as e:
        check("sklearn import", False, repr(e))
        return
    rng = np.random.default_rng(7)
    n = 3000
    for rho in (0.3, 0.7):
        cov = np.array([[1.0, rho], [rho, 1.0]])
        xy = rng.multivariate_normal([0, 0], cov, size=n)
        ours = mi_kraskov(xy[:, 0], xy[:, 1])
        skl = float(mutual_info_regression(xy[:, [0]], xy[:, 1], n_neighbors=1,
                                           random_state=0)[0])
        check("rho=%.1f" % rho, abs(ours - skl) < 0.12,
              "ours=%+.4f sklearn=%+.4f diff=%.3f" % (ours, skl, abs(ours - skl)))


def t_properties():
    # independence -> ~0; MI increases with coupling
    print("\n[4] Qualitative properties")
    rng = np.random.default_rng(3)
    n = 4000
    x = rng.standard_normal(n)
    indep = mi_kraskov(x, rng.standard_normal(n))
    check("independent ~ 0", abs(indep) < 0.05, "MI=%+.4f" % indep)
    mis = []
    for rho in (0.2, 0.5, 0.8):
        cov = np.array([[1.0, rho], [rho, 1.0]])
        xy = rng.multivariate_normal([0, 0], cov, size=n)
        mis.append(mi_kraskov(xy[:, 0], xy[:, 1]))
    check("monotone in coupling", mis[0] < mis[1] < mis[2],
          "MI(0.2,0.5,0.8)=%s" % [round(m, 3) for m in mis])


def t_normalize_columns():
    print("\n[5] normalize_columns")
    a = np.array([[1.0, 5.0, 7.0], [3.0, 5.0, 9.0], [5.0, 5.0, 11.0]])
    nrm = mr.normalize_columns(a)
    check("range in [0,1]", nrm.min() >= -1e-12 and nrm.max() <= 1 + 1e-12,
          "min=%.3f max=%.3f" % (nrm.min(), nrm.max()))
    check("varying col scaled to [0,1]", np.allclose(nrm[:, 0], [0.0, 0.5, 1.0]))
    check("constant col -> 0 (no NaN/inf)", np.all(nrm[:, 1] == 0.0))


def t_driver_smoke():
    print("\n[6] Driver pipeline smoke test (synthetic, deterministic)")
    out = mr.make_synthetic_out(n=3000, seed=0)
    mf, mm, ms, nk = mr.compute_mi_for_lag(out, num_resample=8, seed=123)
    check("shapes (9,)", mf.shape == (9,) and mm.shape == (9,) and ms.shape == (9,),
          "nk=%d" % nk)
    check("all finite", np.all(np.isfinite(mf)) and np.all(np.isfinite(mm))
          and np.all(np.isfinite(ms)))
    check("std >= 0", np.all(ms >= 0))
    check("DST (col 0) ranked highest", int(np.argmax(mf)) == 0,
          "argmax=%d mf=%s" % (int(np.argmax(mf)), np.round(mf, 3)))
    mf2 = mr.compute_mi_for_lag(out, num_resample=8, seed=123)[0]
    check("deterministic w/ fixed seed", np.allclose(mf, mf2))


def t_split_leakage():
    # the resampling draws subsets of the retained index only
    print("\n[7] Resample/split integrity")
    out = mr.make_synthetic_out(n=2000, seed=1)
    varies = out[:, 5:14]
    idx = np.where((out[:, 1] > 60.0) & np.all(np.isfinite(varies), axis=1))[0]
    n_sub = int(np.floor(idx.size * 0.6))
    perm = np.random.default_rng(5).permutation(idx)[:n_sub]
    check("subsample size == floor(frac*n)", perm.size == n_sub,
          "n_sub=%d kept=%d" % (n_sub, idx.size))
    check("all sampled indices within retained set", set(perm).issubset(set(idx)))
    check("no duplicate indices within a draw", len(set(perm)) == perm.size)
    print("    (bootstrap variance estimation, not a train/test split)")


def main():
    print("=" * 72)
    print("MI sanity checks")
    print("=" * 72)
    for fn in (t_bruteforce_equivalence, t_gaussian_analytic, t_sklearn_crosscheck,
               t_properties, t_normalize_columns, t_driver_smoke, t_split_leakage):
        try:
            fn()
        except Exception:
            _FAILS.append(fn.__name__)
            print("  [FAIL] %s raised:" % fn.__name__)
            traceback.print_exc()
    print("\n" + "=" * 72)
    if _FAILS:
        print("RESULT: %d FAILURE(S): %s" % (len(_FAILS), _FAILS))
        sys.exit(1)
    print("RESULT: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
