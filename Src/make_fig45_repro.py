#!/usr/bin/env python3
"""Manuscript-style Figs. 4 (Relative Residual) & 5 (RMSE) from the reproduced
15-feature topside-Ne model (Projects/ne15/test_arrays.npz).

Unlike make_fig45_errmaps.py (the 12-feature model), this uses the full
15-feature model (with measured NmF2/hmF2/VSH), so all 9 manuscript panels --
including ln(NmF2) and hmF2 -- can be drawn, and the magnitudes should land in
the paper's range. Evaluated on the held-out test years (2009 & 2013).

2x5 layout matching the manuscript: latitude, longitude, altitude, month, UT,
F10.7, Kp, ln(NmF2), hmF2.

Usage:
  python Src/make_fig45_repro.py --npz Projects/ne15/test_arrays.npz \
      --fig4 fig4_repro.png --fig5 fig5_repro.png
"""
from __future__ import annotations
import argparse
import numpy as np

# X 15-feature columns: 0 Alt,1 Lat,2 Lon,3 Azi,4 DST,5 AE,6 AP,7 F107,8 Kp,
#                        9 Vf,10 DoY,11 UT,12 hmF2,13 NmF2,14 VSH
PANELS = [
    ("latitude", lambda X: X[:, 1]),
    ("longitude", lambda X: X[:, 2]),
    ("altitude (km)", lambda X: X[:, 0]),
    ("month", lambda X: X[:, 10]),
    ("UT (hour)", lambda X: X[:, 11]),
    ("$F_{107}$ (sfu)", lambda X: X[:, 7]),
    ("Kp", lambda X: X[:, 8]),
    ("ln(NmF2 (cm$^{-3}$))", lambda X: np.log(np.clip(X[:, 13], 1e-30, None))),
    ("hmF2 (km)", lambda X: X[:, 12]),
]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", default="Projects/ne15/test_arrays.npz")
    ap.add_argument("--bins", type=int, default=100)
    ap.add_argument("--fig4", default="fig4_repro.png")
    ap.add_argument("--fig5", default="fig5_repro.png")
    args = ap.parse_args(argv)

    d = np.load(args.npz, allow_pickle=True)
    X, pred, ref = d["X"].astype(float), d["pred"].astype(float), d["ref"].astype(float)
    re = np.abs(pred - ref) / ref
    print(f"n_test={len(ref)}  mean rel-err={re.mean()*100:.2f}%  median={np.median(re)*100:.2f}%  "
          f"RMSE={np.sqrt(np.mean((pred-ref)**2))/1e5:.3f}x1e5")

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def make(metric, ylabel, save):
        fig, axs = plt.subplots(2, 5, figsize=(22, 8))
        axf = axs.flat
        for pi, (label, getx) in enumerate(PANELS):
            ax = axf[pi]; xv = getx(X)
            edges = np.linspace(np.percentile(xv, 0.5), np.percentile(xv, 99.5), args.bins + 1)
            cen = 0.5 * (edges[:-1] + edges[1:]); vals = np.full(args.bins, np.nan)
            for b in range(args.bins):
                m = (xv >= edges[b]) & (xv < edges[b+1])
                if m.sum() >= 20:
                    vals[b] = (np.mean(re[m]) if metric == "relerr"
                               else np.sqrt(np.mean((pred[m]-ref[m])**2)))
            ax.plot(cen, vals, "-", lw=0.9, color="tab:blue")
            ax.set_xlabel(label); ax.grid(alpha=0.25)
            if pi == 0:
                ax.set_ylabel(ylabel)
        axf[9].axis("off")
        axf[9].text(0.5, 0.5, "reproduced 15-feature model\n(test years 2009 & 2013)\n"
                    f"median rel-err = {np.median(re)*100:.1f}%",
                    ha="center", va="center", fontsize=11)
        fig.suptitle(f"{ylabel} per input variable — reproduced 15-feature topside-Ne model "
                     f"(measured NmF2/hmF2/VSH; cf. manuscript Figs. 4-5)", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(save, dpi=140); print(f"saved -> {save}")

    make("relerr", "Relative Residual", args.fig4)
    make("rmse", "RMSE (el cm$^{-3}$)", args.fig5)


if __name__ == "__main__":
    main()
