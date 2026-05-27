# Contrast version of Figs. 4-5: the 12-feature model's Relative Residual / RMSE
# per variable, on the SAME samples as make_fig45_repro.py (the ne15 test set).
# The 12-feature model (Projects/ne_ann) is run on the first 12 features; hmF2,
# NmF2 and VSH are used as binning axes (they aren't model inputs here), so all
# panels are filled and the figure is directly comparable to the 15-feature one.
# Magnitudes are ~10x the paper's because this model lacks the peak features.
#   python Src/make_fig45_errmaps.py --npz Projects/ne15/test_arrays.npz \
#       --model Projects/ne_ann --fig4 fig4_relerr.png --fig5 fig5_rmse.png

import argparse
import json
import os
import sys
import numpy as np
import scipy.io as sio
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_ne_ann import build_net   # noqa: E402

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
    ("VSH (km)", lambda X: X[:, 14]),
]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Figs 4-5 contrast (12-feature model)")
    ap.add_argument("--npz", default="Projects/ne15/test_arrays.npz")
    ap.add_argument("--model", default="Projects/ne_ann")
    ap.add_argument("--config", default="Config.json")
    ap.add_argument("--bins", type=int, default=80)
    ap.add_argument("--fig4", default="fig4_relerr.png")
    ap.add_argument("--fig5", default="fig5_rmse.png")
    args = ap.parse_args(argv)

    d = np.load(args.npz, allow_pickle=True)
    X = d["X"].astype(float)        # 15 features (incl. hmF2/NmF2/VSH)
    ref = d["ref"].astype(float)

    # run the 12-feature model on the first 12 columns
    cfg = json.load(open(args.config))
    ns = sio.loadmat(os.path.join(args.model, "norm_stats.mat"))
    meanX, stdX = ns["meanX"].ravel(), ns["stdX"].ravel()
    meanY, stdY = float(ns["meanY"]), float(ns["stdY"])
    net = build_net(12, cfg["NN_parameters"]["Hidden_layer"],
                    cfg["NN_parameters"]["Activation_finction"])
    net.load_state_dict(torch.load(os.path.join(args.model, "ne_ann.pt"), map_location="cpu"))
    net.eval()
    Xn = ((X[:, :12] - meanX) / stdX).astype(np.float32)
    with torch.no_grad():
        out = [net(torch.from_numpy(Xn[i:i + 200000])).numpy().squeeze()
               for i in range(0, len(Xn), 200000)]
    pred = np.concatenate(out) * stdY + meanY
    re = np.abs(pred - ref) / ref
    print("n=%d  12-feature model: mean rel-err=%.1f%%  median=%.1f%%"
          % (len(ref), re.mean() * 100, np.median(re) * 100))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def make(metric, ylabel, save):
        fig, axs = plt.subplots(3, 4, figsize=(22, 12))
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
            ax.plot(cen, vals, "-", lw=0.9, color="tab:red")
            ax.set_xlabel(label)
            ax.grid(alpha=0.25)
            if pi == 0:
                ax.set_ylabel(ylabel)
        for j in range(len(PANELS), len(axf)):
            axf[j].axis("off")
        axf[len(PANELS)].text(0.5, 0.5, "12-feature model (contrast)\n"
                              "median rel-err = %.0f%%\n(hmF2/NmF2/VSH are binning\n"
                              "axes, not model inputs)" % (np.median(re) * 100),
                              ha="center", va="center", fontsize=11)
        fig.suptitle("%s per input variable - 12-feature model (contrast; cf. "
                     "make_fig45_repro.py for the 15-feature model)" % ylabel, fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(save, dpi=140)
        print("saved ->", save)

    make("relerr", "Relative Residual", args.fig4)
    make("rmse", "RMSE (el cm$^{-3}$)", args.fig5)


if __name__ == "__main__":
    main()
