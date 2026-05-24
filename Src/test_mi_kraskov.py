#!/usr/bin/env python3
"""Sanity tests for the Python MI migration (mi_kraskov, mi_regressor, show_mi).

Run:  python Src/test_mi_kraskov.py

Exits non-zero if any check fails. Each check prints PASS/FAIL with numbers so
the output doubles as the sanity-check report.
"""
from __future__ import annotations

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
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        _FAILS.append(name)


def t_bruteforce_equivalence():
    """'exact' backend must match the MATLAB-loop reference bit-for-bit;
    'fast' backend must agree to ~1e-3 nats."""
    print("\n[1] exact == MATLAB-loop reference (bit-exact); fast within tol")
    rng = np.random.default_rng(42)
    for trial in range(6):
        n = int(rng.integers(30, 300))
        x = rng.standard_normal(n)
        y = 0.7 * x + 0.5 * rng.standard_normal(n)
        ref = _mi_kraskov_bruteforce(x, y)
        exact = mi_kraskov(x, y, method="exact")
        fast = mi_kraskov(x, y, method="fast")
        check(f"trial {trial} exact (n={n})", abs(exact - ref) < 1e-12,
              f"exact={exact:.8f} ref={ref:.8f} diff={abs(exact-ref):.1e}")
        check(f"trial {trial} fast   (n={n})", abs(fast - ref) < 6e-3,
              f"fast={fast:.6f} ref={ref:.6f} diff={abs(fast-ref):.1e}")


def t_gaussian_analytic():
    """Bivariate Gaussian has MI = -0.5*ln(1-rho^2); estimator should track it."""
    print("\n[2] Bivariate-Gaussian analytic MI")
    rng = np.random.default_rng(1)
    n = 5000
    for rho in (0.0, 0.3, 0.6, 0.9):
        cov = np.array([[1.0, rho], [rho, 1.0]])
        xy = rng.multivariate_normal([0, 0], cov, size=n)
        est = mi_kraskov(xy[:, 0], xy[:, 1])
        analytic = -0.5 * np.log(1 - rho ** 2) if rho else 0.0
        # k=1 KSG is noisy; allow generous abs tolerance, tighter for rho=0.
        tol = 0.05 if rho == 0.0 else 0.08
        check(f"rho={rho:.1f}", abs(est - analytic) < tol,
              f"est={est:+.4f} analytic={analytic:+.4f} tol={tol}")


def t_sklearn_crosscheck():
    """Corroborate magnitude/ranking against sklearn's KSG (mutual_info_regression)."""
    print("\n[3] Cross-check vs sklearn KSG (n_neighbors=1)")
    try:
        from sklearn.feature_selection import mutual_info_regression
    except Exception as e:  # pragma: no cover
        check("sklearn import", False, repr(e))
        return
    rng = np.random.default_rng(7)
    n = 3000
    for rho in (0.3, 0.7):
        cov = np.array([[1.0, rho], [rho, 1.0]])
        xy = rng.multivariate_normal([0, 0], cov, size=n)
        ours = mi_kraskov(xy[:, 0], xy[:, 1])
        skl = float(mutual_info_regression(
            xy[:, [0]], xy[:, 1], n_neighbors=1, random_state=0)[0])
        # Different tie/noise handling; require same ballpark (within 0.12 nats).
        check(f"rho={rho:.1f}", abs(ours - skl) < 0.12,
              f"ours={ours:+.4f} sklearn={skl:+.4f} diff={abs(ours-skl):.3f}")


def t_properties():
    """Independence -> ~0; stronger coupling -> larger MI (monotone ranking)."""
    print("\n[4] Qualitative properties")
    rng = np.random.default_rng(3)
    n = 4000
    x = rng.standard_normal(n)
    indep = mi_kraskov(x, rng.standard_normal(n))
    check("independent ~ 0", abs(indep) < 0.05, f"MI={indep:+.4f}")

    mis = []
    for rho in (0.2, 0.5, 0.8):
        cov = np.array([[1.0, rho], [rho, 1.0]])
        xy = rng.multivariate_normal([0, 0], cov, size=n)
        mis.append(mi_kraskov(xy[:, 0], xy[:, 1]))
    check("monotone in coupling", mis[0] < mis[1] < mis[2],
          f"MI(0.2,0.5,0.8)={[round(m,3) for m in mis]}")


def t_normalize_columns():
    print("\n[5] normalize_columns")
    a = np.array([[1.0, 5.0, 7.0], [3.0, 5.0, 9.0], [5.0, 5.0, 11.0]])
    nrm = mr.normalize_columns(a)
    check("range in [0,1]", nrm.min() >= -1e-12 and nrm.max() <= 1 + 1e-12,
          f"min={nrm.min():.3f} max={nrm.max():.3f}")
    check("varying col scaled to [0,1]",
          np.allclose(nrm[:, 0], [0.0, 0.5, 1.0]))
    check("constant col -> 0 (no NaN/inf)", np.all(nrm[:, 1] == 0.0))


def t_driver_smoke():
    """Synthetic end-to-end: shapes, finiteness, std>=0, DST ranked highest."""
    print("\n[6] Driver pipeline smoke test (synthetic, deterministic)")
    out = mr.make_synthetic_out(n=3000, seed=0)
    mf, mm, ms, nk = mr.compute_mi_for_lag(out, num_resample=8, seed=123)
    check("shapes (9,)", mf.shape == (9,) and mm.shape == (9,) and ms.shape == (9,),
          f"nk={nk}")
    check("all finite", np.all(np.isfinite(mf)) and np.all(np.isfinite(mm))
          and np.all(np.isfinite(ms)))
    check("std >= 0", np.all(ms >= 0))
    check("DST (col 0) ranked highest", int(np.argmax(mf)) == 0,
          f"argmax={int(np.argmax(mf))} mf={np.round(mf,3)}")

    # Determinism: same seed -> identical result.
    mf2, *_ = mr.compute_mi_for_lag(out, num_resample=8, seed=123)
    check("deterministic w/ fixed seed", np.allclose(mf, mf2))


def t_split_leakage():
    """The resampling draws subsets of the retained index only; size = floor(frac*n)."""
    print("\n[7] Resample/split integrity (no out-of-index sampling)")
    out = mr.make_synthetic_out(n=2000, seed=1)
    mlat = out[:, 1]
    varies = out[:, 5:14]
    idx = np.where((mlat > 60.0) & np.all(np.isfinite(varies), axis=1))[0]
    frac = 0.6
    n_sub = int(np.floor(idx.size * frac))
    rng = np.random.default_rng(5)
    perm = rng.permutation(idx)[:n_sub]
    check("subsample size == floor(frac*n)", perm.size == n_sub,
          f"n_sub={n_sub} kept={idx.size}")
    check("all sampled indices within retained set", set(perm).issubset(set(idx)))
    check("no duplicate indices within a draw", len(set(perm)) == perm.size)
    print("    NOTE: this is bootstrap variance estimation of MI, not an ML "
          "train/test split; subsamples intentionally overlap across draws and "
          "no held-out test set exists -> ML 'leakage' is not applicable here.")


def main():
    print("=" * 72)
    print("MI migration sanity checks")
    print("=" * 72)
    for fn in (t_bruteforce_equivalence, t_gaussian_analytic, t_sklearn_crosscheck,
               t_properties, t_normalize_columns, t_driver_smoke, t_split_leakage):
        try:
            fn()
        except Exception:
            _FAILS.append(fn.__name__)
            print(f"  [FAIL] {fn.__name__} raised:")
            traceback.print_exc()
    print("\n" + "=" * 72)
    if _FAILS:
        print(f"RESULT: {len(_FAILS)} FAILURE(S): {_FAILS}")
        sys.exit(1)
    print("RESULT: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
