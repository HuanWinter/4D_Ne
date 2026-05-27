# MI feature/lag analysis driver (Python version of MI_regressor.m).
# For each lag: load Delay/all_<lag>.mat, keep high-lat NaN-free samples,
# min-max normalise each driver, and compute Kraskov MI vs log(NmF2), with a
# bootstrap (16 draws of 60%) for mean/std plus the full-sample MI.
# out columns (0-based): 0 Alt, 1 mLat, 5:14 = DST,AE,ap,F10.7,Kp,Ve,Bx,By,Bz, 16 NmF2.
# Pass negative lag ranges with '=', e.g. --lags=-8:7

import argparse
import os
import sys
import time
import numpy as np
from scipy import io as sio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mi_kraskov import mi_kraskov  # noqa: E402

VARI_SET = ["DST", "AE Index", "ap", "f10.7 index", "Kp", "Ve", "Bx", "By", "Bz"]


def normalize_columns(arr):
    # per-column min-max to [0,1]; nanmin/nanmax so a stray NaN doesn't poison
    # the whole column (MATLAB min/max ignore NaN). Constant cols -> 0.
    arr = np.asarray(arr, dtype=float)
    col_min = np.nanmin(arr, axis=0)
    span = np.nanmax(arr, axis=0) - col_min
    bad = (span == 0) | ~np.isfinite(span)
    out = (arr - col_min) / np.where(bad, 1.0, span)
    out[:, bad] = 0.0
    return out


def compute_mi_for_lag(out, num_resample=16, frac=0.6, mlat_min=60.0,
                       abs_mlat=False, seed=None, method="exact"):
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
            mi_t[k] = mi_kraskov(varies[perm, j], log_nmf2[perm], method=method)
        mi_mean[j] = mi_t.mean()
        mi_std[j] = mi_t.std(ddof=1)   # sample std, like MATLAB std
        mi_full[j] = mi_kraskov(varies[idx, j], log_nmf2[idx], method=method)
    return mi_full, mi_mean, mi_std, n_kept


# deterministic synthetic 'out' (>=17 cols) for the smoke test; DST coupled to NmF2
def make_synthetic_out(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    out = np.full((n, 17), np.nan)
    out[:, 0] = rng.uniform(200, 600, n)
    out[:, 1] = rng.uniform(-90, 90, n)
    drivers = rng.standard_normal((n, 9))
    out[:, 5:14] = drivers
    base = 12.0 + 0.9 * drivers[:, 0] + 0.3 * drivers[:, 1]
    out[:, 16] = np.exp(base + 0.2 * rng.standard_normal(n))
    return out


def run(data_dir, lags, synthetic, num_resample, frac, mlat_min, abs_mlat,
        seed, out_path, method="exact"):
    n_lags = len(lags)
    nv = len(VARI_SET)
    MI = np.full((n_lags, nv), np.nan)
    MI_mean = np.full((n_lags, nv), np.nan)
    MI_std = np.full((n_lags, nv), np.nan)
    missing = []

    for i, lag in enumerate(lags):
        lag_seed = seed + i        # non-negative, deterministic per lag
        if synthetic:
            out = make_synthetic_out(seed=lag_seed)
        else:
            path = os.path.join(data_dir, "all_%d.mat" % lag)
            if not os.path.isfile(path):
                missing.append(path)
                print("  [lag %+d] MISSING: %s" % (lag, path), file=sys.stderr)
                continue
            data = sio.loadmat(path)
            if "out" not in data:
                print("  [lag %+d] no 'out' in %s" % (lag, path), file=sys.stderr)
                continue
            out = np.asarray(data["out"], dtype=float)
            if out.shape[1] < 17:
                print("  [lag %+d] %s has %d cols (<17); skipping"
                      % (lag, path, out.shape[1]), file=sys.stderr)
                continue

        t0 = time.time()
        mf, mm, ms, nk = compute_mi_for_lag(
            out, num_resample=num_resample, frac=frac, mlat_min=mlat_min,
            abs_mlat=abs_mlat, seed=lag_seed, method=method)
        MI[i], MI_mean[i], MI_std[i] = mf, mm, ms
        print("  [lag %+d] n_kept=%d  (%.1fs)" % (lag, nk, time.time() - t0))

    if missing:
        print("\n%d of %d lag files missing; those rows are NaN."
              % (len(missing), n_lags), file=sys.stderr)

    if out_path:
        np.savez(out_path, MI=MI, MI_mean=MI_mean, MI_std=MI_std,
                 lags=np.asarray(lags), vari_set=np.array(VARI_SET, dtype=object))
        print("Saved results to", out_path)
    return MI, MI_mean, MI_std


def main(argv=None):
    ap = argparse.ArgumentParser(description="MI of drivers vs log(NmF2) per lag")
    ap.add_argument("--data-dir", default="Data/Delay")
    ap.add_argument("--lags", default="-8:7", help="lag range start:stop (pass as --lags=-8:7)")
    ap.add_argument("--num-resample", type=int, default=16)
    ap.add_argument("--frac", type=float, default=0.6)
    ap.add_argument("--mlat-min", type=float, default=60.0)
    ap.add_argument("--abs-mlat", action="store_true", help="use |mLat| > mlat-min")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--method", choices=["exact", "fast"], default="exact")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", dest="out_path", default=None)
    args = ap.parse_args(argv)

    start, stop = (int(x) for x in args.lags.split(":"))
    lags = list(range(start, stop + 1))
    print("Lags:", lags)
    run(args.data_dir, lags, args.synthetic, args.num_resample, args.frac,
        args.mlat_min, args.abs_mlat, args.seed, args.out_path, args.method)


if __name__ == "__main__":
    main()
