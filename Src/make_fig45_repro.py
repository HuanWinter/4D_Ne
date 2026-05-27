# Manuscript Figs. 4 (Relative Residual) & 5 (RMSE) per input variable, from the
# reproduced 15-feature model (Projects/ne15/test_arrays.npz, test yrs 2009 & 2013).
# 2x5 layout: latitude, longitude, altitude, month, UT, F10.7, Kp, ln(NmF2), hmF2.
#   python Src/make_fig45_repro.py --npz Projects/ne15/test_arrays.npz \
#       --fig4 fig4_repro.png --fig5 fig5_repro.png

import argparse
import numpy as np

# 15-feature columns: 0 Alt,1 Lat,2 Lon,3 Azi,4 DST,5 AE,6 AP,7 F107,8 Kp,
#                      9 Vf,10 DoY,11 UT,12 hmF2,13 NmF2,14 VSH
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
    ap = argparse.ArgumentParser(description="manuscript Figs. 4-5 (15-feature model)")
    ap.add_argument("--npz", default="Projects/ne15/test_arrays.npz")
    ap.add_argument("--bins", type=int, default=100)
    ap.add_argument("--fig4", default="fig4_repro.png")
    ap.add_argument("--fig5", default="fig5_repro.png")
    args = ap.parse_args(argv)

    d = np.load(args.npz, allow_pickle=True)
    X, pred, ref = d["X"].astype(float), d["pred"].astype(float), d["ref"].astype(float)
    re = np.abs(pred - ref) / ref
    print("n_test=%d  mean rel-err=%.2f%%  median=%.2f%%  RMSE=%.3fx1e5"
          % (len(ref), re.mean() * 100, np.median(re) * 100,
             np.sqrt(np.mean((pred - ref) ** 2)) / 1e5))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def make(metric, ylabel, save):
        fig, axs = plt.subplots(2, 5, figsize=(22, 8))
        axf = axs.flat
        for pi, (label, getx) in enumerate(PANELS):
            ax = axf[pi]
            xv = getx(X)
            edges = np.linspace(np.percentile(xv, 0.5), np.percentile(xv, 99.5), args.bins + 1)
            cen = 0.5 * (edges[:-1] + edges[1:])
            vals = np.full(args.bins, np.nan)
            for b in range(args.bins):
                m = (xv >= edges[b]) & (xv < edges[b + 1])
                if m.sum() >= 20:
                    vals[b] = (np.mean(re[m]) if metric == "relerr"
                               else np.sqrt(np.mean((pred[m] - ref[m]) ** 2)))
            ax.plot(cen, vals, "-", lw=0.9, color="tab:blue")
            ax.set_xlabel(label)
            ax.grid(alpha=0.25)
            if pi == 0:
                ax.set_ylabel(ylabel)
        axf[9].axis("off")
        axf[9].text(0.5, 0.5, "15-feature model\n(test years 2009 & 2013)\n"
                    "median rel-err = %.1f%%" % (np.median(re) * 100),
                    ha="center", va="center", fontsize=11)
        fig.suptitle("%s per input variable - 15-feature topside-Ne model "
                     "(cf. manuscript Figs. 4-5)" % ylabel, fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(save, dpi=140)
        print("saved ->", save)

    make("relerr", "Relative Residual", args.fig4)
    make("rmse", "RMSE (el cm$^{-3}$)", args.fig5)


if __name__ == "__main__":
    main()
