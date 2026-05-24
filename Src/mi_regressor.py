#!/usr/bin/env python3
"""Mutual-information feature/lag analysis driver.

Faithful Python port of ``MI_regressor.m``. For each time lag (default -8..+7
hours) it loads the corresponding delay dataset, keeps high-latitude
NaN-free samples, min-max normalises each candidate driver, and estimates the
Kraskov mutual information (k = 1) between each driver and log(NmF2). A
bootstrap-style resampling (default 16 draws of 60% of the data) yields the
mean and standard deviation of the MI; the full-sample MI is also recorded.

This reproduces the *continuous* Kraskov estimator used in the MATLAB pipeline,
NOT the discretised ``normalized_mutual_info_score`` / pyitlib approach in
``NmF2-MI.ipynb`` (which rounds variables into 0-100 bins). See MIGRATION_REPORT.md.

Data layout (0-based, matching the MATLAB 1-based columns)
----------------------------------------------------------
    out[:, 0]      Altitude          (MATLAB col 1)
    out[:, 1]      magnetic latitude (MATLAB col 2)
    out[:, 5:14]   9 driver variables: DST, AE, ap, F10.7, Kp, Ve, Bx, By, Bz
                                       (MATLAB cols 6:14)
    out[:, 16]     NmF2              (MATLAB col 17)

Note: ``MI_regressor.m`` filters on ``mLat > 60`` and the 9 driver NaN checks.
The notebook additionally used ``abs(mLat) > 60`` and ``VTEC1 > 0``; this port
follows the MATLAB script. Use ``--abs-mlat`` to opt into the |mLat| variant.

Usage
-----
    # real data (expects <data-dir>/all_<lag>.mat with an `out` array)
    # NOTE: pass negative lag ranges with '=' so argparse does not read them as
    #       flags, e.g. --lags=-8:7
    python mi_regressor.py --data-dir Data/Delay --lags=-8:7 --out mi_results.npz

    # deterministic smoke test on synthetic data (no external files needed)
    python mi_regressor.py --synthetic --lags=-2:2 --out /tmp/mi_smoke.npz
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from scipy import io as sio

# Allow running both as `python Src/mi_regressor.py` and as a module.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mi_kraskov import mi_kraskov  # noqa: E402

VARI_SET = ["DST", "AE Index", "ap", "f10.7 index", "Kp", "Ve", "Bx", "By", "Bz"]


def normalize_columns(arr: np.ndarray) -> np.ndarray:
    """Per-column min-max scaling to [0, 1] (matches MATLAB normA/max(normA)).

    Constant columns (max == min) map to 0 to avoid divide-by-zero.
    """
    arr = np.asarray(arr, dtype=float)
    # nanmin/nanmax: match MATLAB's min/max, which ignore NaN. Using plain
    # min/max would let a single NaN in a column poison the whole normalized
    # column (and later break the MI estimator). Finite entries normalize to
    # [0,1]; NaN entries stay NaN and are dropped by the idx filter upstream.
    col_min = np.nanmin(arr, axis=0)
    span = np.nanmax(arr, axis=0) - col_min
    span_safe = np.where((span == 0) | ~np.isfinite(span), 1.0, span)
    out = (arr - col_min) / span_safe
    out[:, (span == 0) | ~np.isfinite(span)] = 0.0
    return out


def compute_mi_for_lag(out: np.ndarray, *, num_resample: int = 16,
                       frac: float = 0.6, mlat_min: float = 60.0,
                       abs_mlat: bool = False, seed: int | None = None,
                       method: str = "exact"):
    """Compute MI of each driver vs log(NmF2) for one lag's data array.

    Returns
    -------
    mi_full : (9,) MI on all retained samples.
    mi_mean : (9,) mean MI over resamples.
    mi_std  : (9,) std  MI over resamples.
    n_kept  : int   number of retained samples.
    """
    alt = out[:, 0]            # noqa: F841 (kept for parity/clarity)
    mlat = out[:, 1]
    varies = out[:, 5:14].copy()
    nmf2 = out[:, 16]

    lat_test = np.abs(mlat) > mlat_min if abs_mlat else mlat > mlat_min
    finite = np.all(np.isfinite(varies), axis=1)
    idx = np.where(lat_test & finite)[0]
    n_kept = idx.size
    if n_kept < 5:
        nan9 = np.full(len(VARI_SET), np.nan)
        return nan9, nan9.copy(), nan9.copy(), n_kept

    varies = normalize_columns(varies)
    log_nmf2 = np.log(nmf2)

    rng = np.random.default_rng(seed)
    n_sub = int(np.floor(n_kept * frac))

    mi_full = np.zeros(len(VARI_SET))
    mi_mean = np.zeros(len(VARI_SET))
    mi_std = np.zeros(len(VARI_SET))

    for j in range(len(VARI_SET)):
        mi_t = np.zeros(num_resample)
        for k in range(num_resample):
            perm = rng.permutation(idx)[:n_sub]
            x_t = varies[perm, j]
            y_t = log_nmf2[perm]
            mi_t[k] = mi_kraskov(x_t, y_t, method=method)
        mi_mean[j] = mi_t.mean()
        mi_std[j] = mi_t.std(ddof=1)  # sample std, matching MATLAB std (see report)
        mi_full[j] = mi_kraskov(varies[idx, j], log_nmf2[idx], method=method)
    return mi_full, mi_mean, mi_std, n_kept


def make_synthetic_out(n: int = 3000, seed: int = 0) -> np.ndarray:
    """Deterministic synthetic `out` array (>=17 cols) for smoke testing.

    Driver 0 (DST) is strongly coupled to log(NmF2); others are progressively
    weaker / independent, so a correct pipeline should rank DST highest.
    """
    rng = np.random.default_rng(seed)
    out = np.full((n, 17), np.nan)
    out[:, 0] = rng.uniform(200, 600, n)          # Altitude
    out[:, 1] = rng.uniform(-90, 90, n)           # mLat (some pass >60)
    drivers = rng.standard_normal((n, 9))
    out[:, 5:14] = drivers
    base = 12.0 + 0.9 * drivers[:, 0] + 0.3 * drivers[:, 1]
    out[:, 16] = np.exp(base + 0.2 * rng.standard_normal(n))  # NmF2 (positive)
    return out


def run(data_dir: str, lags, *, synthetic: bool, num_resample: int, frac: float,
        mlat_min: float, abs_mlat: bool, seed: int, out_path: str | None,
        method: str = "exact"):
    n_lags = len(lags)
    nv = len(VARI_SET)
    MI = np.full((n_lags, nv), np.nan)
    MI_mean = np.full((n_lags, nv), np.nan)
    MI_std = np.full((n_lags, nv), np.nan)
    missing = []

    for i, lag in enumerate(lags):
        lag_seed = seed + i  # non-negative, deterministic per lag (MATLAB used global rng)
        if synthetic:
            out = make_synthetic_out(seed=lag_seed)
        else:
            path = os.path.join(data_dir, f"all_{lag}.mat")
            if not os.path.isfile(path):
                missing.append(path)
                print(f"  [lag {lag:+d}] MISSING: {path}", file=sys.stderr)
                continue
            data = sio.loadmat(path)
            if "out" not in data:
                print(f"  [lag {lag:+d}] no 'out' var in {path}", file=sys.stderr)
                continue
            out = np.asarray(data["out"], dtype=float)
            if out.shape[1] < 17:
                print(f"  [lag {lag:+d}] {path} has {out.shape[1]} cols (<17); skipping",
                      file=sys.stderr)
                continue

        t0 = time.time()
        mf, mm, ms, nk = compute_mi_for_lag(
            out, num_resample=num_resample, frac=frac, mlat_min=mlat_min,
            abs_mlat=abs_mlat, seed=lag_seed, method=method)
        MI[i], MI_mean[i], MI_std[i] = mf, mm, ms
        print(f"  [lag {lag:+d}] n_kept={nk}  ({time.time()-t0:.1f}s)")

    if missing:
        print(f"\n{len(missing)} of {n_lags} lag files missing; "
              f"those rows are NaN.", file=sys.stderr)

    if out_path:
        np.savez(out_path, MI=MI, MI_mean=MI_mean, MI_std=MI_std,
                 lags=np.asarray(lags), vari_set=np.array(VARI_SET, dtype=object))
        print(f"Saved results to {out_path}")
    return MI, MI_mean, MI_std


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="Data/Delay",
                    help="directory containing all_<lag>.mat files")
    ap.add_argument("--lags", default="-8:7",
                    help="lag range 'start:stop' inclusive (default -8:7)")
    ap.add_argument("--num-resample", type=int, default=16)
    ap.add_argument("--frac", type=float, default=0.6)
    ap.add_argument("--mlat-min", type=float, default=60.0)
    ap.add_argument("--abs-mlat", action="store_true",
                    help="use |mLat| > mlat-min (notebook variant) instead of mLat >")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--method", choices=["exact", "fast"], default="exact",
                    help="'exact' matches MATLAB bit-for-bit (O(n^2)); "
                         "'fast' is O(n log n), ~1e-3 nats different")
    ap.add_argument("--synthetic", action="store_true",
                    help="ignore data-dir; run on deterministic synthetic data")
    ap.add_argument("--out", dest="out_path", default=None,
                    help="path to save an .npz of results")
    args = ap.parse_args(argv)

    start, stop = (int(x) for x in args.lags.split(":"))
    lags = list(range(start, stop + 1))
    print(f"Lags: {lags}")
    run(args.data_dir, lags, synthetic=args.synthetic,
        num_resample=args.num_resample, frac=args.frac, mlat_min=args.mlat_min,
        abs_mlat=args.abs_mlat, seed=args.seed, out_path=args.out_path,
        method=args.method)


if __name__ == "__main__":
    main()
