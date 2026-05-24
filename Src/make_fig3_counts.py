#!/usr/bin/env python3
"""Reproduce manuscript Fig. 3: sample counts in the three datasets per variable.

"Numbers of samples in three datasets (training / cross-validation / testing,
all from COSMIC-1) shown as a function of each input variable." Uses the prepared
`Data/data_4d_ne/XY_*.mat` (12 primary features) with the paper's 70/15/15
mutually-exclusive split.

NOTE: the prepared XY files carry only the 12 primary features
(Alt, Lat, Lon, Azi, DST, AE, AP, F10.7, Kp, Vf, DoY, UT); the paper's Fig. 3
also shows hmF2, NmF2 and VSH, which are the sub-model features absent from
these files (see MIGRATION_REPORT.md). Counts are model-independent, so this is
a faithful reproduction for the 12 available variables.

Usage:
  python Src/make_fig3_counts.py --xy-dir Data/data_4d_ne --save fig3_counts.png
"""
from __future__ import annotations
import argparse, glob, json, os
import numpy as np
import scipy.io as sio

FEATURES = ["Altitude", "Latitude", "Longitude", "Azi", "DST", "AE",
            "AP", "F10.7", "Kp", "Vf", "DoY", "UT"]
UNITS = ["km", "deg", "deg", "deg", "nT", "nT", "nT", "sfu", "", "km/s", "month", "hr"]


def apply_filters(X, Y, cfg):
    pr = cfg["Para_range"]
    Lat, Lon, F107, Kp, DoY, UT = X[:, 1], X[:, 2], X[:, 7], X[:, 8]/10, X[:, 10], X[:, 11]
    Ne = Y[:, 1]
    LT = UT + Lon/15; LT = np.where(LT >= 12, LT-12, LT); LT = np.where(LT < 0, LT+12, LT)
    m = ((DoY > pr["DoY"][0]) & (DoY < pr["DoY"][1]) & (LT > pr["LT"][0]) & (LT < pr["LT"][1]) &
         (Kp > pr["Kp"][0]) & (Kp < pr["Kp"][1]) & (F107 > pr["F107"][0]) & (F107 < pr["F107"][1]) &
         (Ne > pr["Ne"][0]) & (Ne < pr["Ne"][1]) &
         (Lat > pr["Latitude"][0]) & (Lat < pr["Latitude"][1]) &
         (Lon > pr["Longitude"][0]) & (Lon < pr["Longitude"][1]) & np.all(np.isfinite(X), 1))
    return X[m]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xy-dir", default="Data/data_4d_ne")
    ap.add_argument("--config", default="Config.json")
    ap.add_argument("--n-files", type=int, default=None)
    ap.add_argument("--bins", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--save", default="fig3_counts.png")
    args = ap.parse_args(argv)

    cfg = json.load(open(args.config))
    files = sorted(glob.glob(os.path.join(args.xy_dir, "XY_*.mat")),
                   key=lambda p: int(p.split("_")[-1].split(".")[0]))
    if args.n_files:
        files = files[:args.n_files]
    Xs, Ys = [], []
    for f in files:
        d = sio.loadmat(f); Xs.append(np.asarray(d["X"], float)); Ys.append(np.asarray(d["Y"], float))
    X = apply_filters(np.vstack(Xs), np.vstack(Ys), cfg)
    X[:, 8] /= 10.0  # Kp stored x10
    print(f"loaded {len(files)} files -> {X.shape[0]} samples after filter")

    # paper's 70/15/15 mutually-exclusive split
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(X.shape[0]); n = X.shape[0]
    n_tr, n_cv = int(0.70*n), int(0.15*n)
    splits = {"train": perm[:n_tr], "cv": perm[n_tr:n_tr+n_cv], "test": perm[n_tr+n_cv:]}
    print({k: len(v) for k, v in splits.items()})

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"train": "tab:blue", "cv": "tab:orange", "test": "tab:green"}
    fig, axs = plt.subplots(3, 4, figsize=(20, 12))
    for i, ax in enumerate(axs.flat):
        if i >= len(FEATURES):
            ax.axis("off"); continue
        lo, hi = X[:, i].min(), X[:, i].max()
        edges = np.linspace(lo, hi, args.bins + 1)
        for k, idx in splits.items():
            ax.hist(X[idx, i], bins=edges, histtype="step", lw=1.6,
                    color=colors[k], label=k)
        u = f" ({UNITS[i]})" if UNITS[i] else ""
        ax.set_title(FEATURES[i]); ax.set_xlabel(FEATURES[i] + u)
        ax.set_ylabel("number of samples"); ax.grid(alpha=0.3)
        if i == 0:
            ax.legend()
    fig.suptitle("Fig. 3 — sample counts in train/cv/test (70/15/15) per input variable "
                 "(12 of 15 features available)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(args.save, dpi=130)
    print(f"saved -> {args.save}")


if __name__ == "__main__":
    main()
