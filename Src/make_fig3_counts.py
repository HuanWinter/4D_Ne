# Manuscript Fig. 3: sample counts in train/cv/test per input variable.
# Built straight from the COSMIC ionPrf profiles so it includes ALL 15 features,
# including hmF2 (=edmaxalt), NmF2 (=edmax) and VSH (1/e scale height) -- these
# are NOT in the prepared Data/data_4d_ne/XY_*.mat (12 features only), so they
# have to be read from the profiles (same build as reproduce_ne_15feat.py).
# 70/15/15 split.
#   python Src/make_fig3_counts.py --list <cosmic_list_local.txt> --n-profiles 40000 \
#       --save fig3_counts.png

import argparse
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_delay_files import load_omni                      # noqa: E402
from reproduce_ne_15feat import build_samples, FEAT15       # noqa: E402

# units per FEAT15 column (Alt,Lat,Lon,Azi,DST,AE,AP,F107,Kp,Vf,DoY,UT,hmF2,NmF2,VSH)
UNITS = ["km", "deg", "deg", "deg", "nT", "nT", "nT", "sfu", "", "km/s",
         "month", "hr", "km", "el cm$^{-3}$", "km"]


def main(argv=None):
    default = "/glade/derecho/scratch/%s/cosmic/cosmic_list_local.txt" % os.environ.get("USER", "andonghu")
    ap = argparse.ArgumentParser(description="manuscript Fig. 3 (15 variables)")
    ap.add_argument("--list", default=default)
    ap.add_argument("--n-profiles", type=int, default=40000)
    ap.add_argument("--omni-cache", default="Data/omni2_cache")
    ap.add_argument("--config", default="Config.json")
    ap.add_argument("--bins", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--save", default="fig3_counts.png")
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    import linecache as lc
    paths = [l for l in lc.getlines(args.list) if l.strip()]
    sel = rng.choice(len(paths), min(args.n_profiles, len(paths)), replace=False)
    paths = [paths[i] for i in sel]
    print("sampling %d profiles; loading OMNI2 2006-2015 ..." % len(paths))
    omni = load_omni(range(2006, 2016), cache_dir=args.omni_cache, source="omni2")

    X, Ne, yr = build_samples(paths, omni)   # X has the 15 features incl. hmF2/NmF2/VSH
    cfg = json.load(open(args.config))
    pr = cfg["Para_range"]
    m = ((Ne > pr["Ne"][0]) & (Ne < pr["Ne"][1]) &
         (X[:, 13] > pr["NmF2"][0]) & (X[:, 13] < pr["NmF2"][1]) &
         (X[:, 12] > pr["hmF2"][0]) & (X[:, 12] < pr["hmF2"][1]) &
         (X[:, 14] > pr["VSH"][0]) & (X[:, 14] < pr["VSH"][1]) & np.all(np.isfinite(X), 1))
    X = X[m]
    print("built %d topside samples, %d features (incl. hmF2/NmF2/VSH)" % (X.shape[0], X.shape[1]))

    n = X.shape[0]
    perm = rng.permutation(n)
    n_tr, n_cv = int(0.70 * n), int(0.15 * n)
    splits = {"train": perm[:n_tr], "cv": perm[n_tr:n_tr + n_cv], "test": perm[n_tr + n_cv:]}
    print({k: len(v) for k, v in splits.items()})

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"train": "tab:blue", "cv": "tab:orange", "test": "tab:green"}
    fig, axs = plt.subplots(3, 5, figsize=(24, 12))
    for i, ax in enumerate(axs.flat):
        if i >= len(FEAT15):
            ax.axis("off")
            continue
        # plot NmF2 in ln, like the manuscript
        xcol = np.log(X[:, i]) if FEAT15[i] == "NmF2" else X[:, i]
        label = "ln(NmF2)" if FEAT15[i] == "NmF2" else FEAT15[i]
        edges = np.linspace(np.percentile(xcol, 0.5), np.percentile(xcol, 99.5), args.bins + 1)
        for k, idx in splits.items():
            ax.hist(xcol[idx], bins=edges, histtype="step", lw=1.6, color=colors[k], label=k)
        u = " (%s)" % UNITS[i] if UNITS[i] else ""
        ax.set_title(label)
        ax.set_xlabel(label + u)
        ax.set_ylabel("number of samples")
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend()
    fig.suptitle("Fig. 3 - sample counts in train/cv/test (70/15/15) per input variable",
                 fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(args.save, dpi=130)
    print("saved ->", args.save)


if __name__ == "__main__":
    main()
