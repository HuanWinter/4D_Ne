#!/usr/bin/env python3
"""Reproduce manuscript Figs. 4 (Relative Residual) & 5 (RMSE) per input variable,
in the manuscript's 2x5 panel layout (training / cross-validation / test curves).

*** THIS USES THIS REPRODUCTION'S 12-FEATURE L2-ANN (`Projects/ne_ann`), whose
median rel-err is ~49% because it lacks the paper's hmF2/NmF2/VSH sub-model
features. The LAYOUT matches the manuscript; the MAGNITUDES are this model's
(~0.5 relative residual), about 10x the paper's ~0.02-0.05, and the latitude
panel will NOT show the paper's clean high-latitude spike. See MIGRATION_REPORT
sec.9. Two manuscript panels -- ln(NmF2) and hmF2 -- cannot be drawn because
those features are absent from the available data; they are shown blanked. ***

The 3 curves use a fresh 70/15/15 partition for visual parity with the
manuscript; the model itself was trained on a separate 80/20 split, so the
"training" curve here is only illustrative (the curves overlap regardless
because the model underfits -> no overfitting, consistent with the manuscript).

Usage:
  python Src/make_fig45_errmaps.py --fig4 fig4_relerr.png --fig5 fig5_rmse.png
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import scipy.io as sio
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_ne_ann import load_xy, apply_filters, build_net

# manuscript panel layout: (label, X-column or None if unavailable, scale-by)
PANELS = [
    ("latitude", 1, 1.0), ("longitude", 2, 1.0), ("altitude (km)", 0, 1.0),
    ("month", 10, 1.0), ("UT (hour)", 11, 1.0),
    ("$F_{107}$ (sfu)", 7, 1.0), ("Kp", 8, 0.1),
    ("ln(NmF2 (cm$^{-3}$))", None, 1.0), ("hmF2 (km)", None, 1.0),
]
COLORS = {"training": "tab:blue", "cross-validation": "peru", "test": "orange"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xy-dir", default="Data/data_4d_ne")
    ap.add_argument("--config", default="Config.json")
    ap.add_argument("--model", default="Projects/ne_ann")
    ap.add_argument("--bins", type=int, default=100)
    ap.add_argument("--max-eval", type=int, default=700000)
    ap.add_argument("--fig4", default="fig4_relerr.png")
    ap.add_argument("--fig5", default="fig5_rmse.png")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)

    cfg = json.load(open(args.config))
    X, Y, _ = load_xy(args.xy_dir)
    X, Ne = apply_filters(X, Y, cfg)
    n = X.shape[0]
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_tr, n_cv = int(0.70 * n), int(0.15 * n)
    split = {"training": perm[:n_tr], "cross-validation": perm[n_tr:n_tr+n_cv],
             "test": perm[n_tr+n_cv:]}

    ns = sio.loadmat(os.path.join(args.model, "norm_stats.mat"))
    meanX, stdX = ns["meanX"].ravel(), ns["stdX"].ravel()
    meanY, stdY = float(ns["meanY"]), float(ns["stdY"])
    net = build_net(X.shape[1], cfg["NN_parameters"]["Hidden_layer"],
                    cfg["NN_parameters"]["Activation_finction"])
    net.load_state_dict(torch.load(os.path.join(args.model, "ne_ann.pt"), map_location="cpu"))
    net.eval()

    data = {}
    for k, idx in split.items():
        if len(idx) > args.max_eval:
            idx = rng.choice(idx, args.max_eval, replace=False)
        Xn = ((X[idx] - meanX) / stdX).astype(np.float32)
        with torch.no_grad():
            out = [net(torch.from_numpy(Xn[i:i+200000])).numpy().squeeze()
                   for i in range(0, len(idx), 200000)]
        pred = np.concatenate(out) * stdY + meanY
        data[k] = (X[idx], pred, Ne[idx])
        print(f"{k}: n={len(idx)} median rel-err={np.median(np.abs(pred-Ne[idx])/Ne[idx])*100:.1f}%")

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def make(metric, ylabel, save, scale=1.0):
        fig, axs = plt.subplots(2, 5, figsize=(22, 8))
        axf = axs.flat
        for pi, (label, col, mul) in enumerate(PANELS):
            ax = axf[pi]
            if col is None:
                ax.text(0.5, 0.5, f"{label}\n(feature not in\navailable data)",
                        ha="center", va="center", fontsize=11, color="gray")
                ax.set_xlabel(label); ax.set_xticks([]); ax.set_yticks([]); continue
            allx = np.concatenate([data[k][0][:, col] * mul for k in data])
            edges = np.linspace(np.percentile(allx, 0.5), np.percentile(allx, 99.5), args.bins + 1)
            cen = 0.5 * (edges[:-1] + edges[1:])
            for k in ("training", "cross-validation", "test"):
                Xk, pred, ref = data[k]
                xv = Xk[:, col] * mul
                vals = np.full(args.bins, np.nan)
                for b in range(args.bins):
                    m = (xv >= edges[b]) & (xv < edges[b+1])
                    if m.sum() >= 20:
                        if metric == "relerr":
                            vals[b] = np.mean(np.abs(pred[m]-ref[m])/ref[m]) * scale
                        else:
                            vals[b] = np.sqrt(np.mean((pred[m]-ref[m])**2)) * scale
                ax.plot(cen, vals, "-", lw=0.8, color=COLORS[k], label=k)
            ax.set_xlabel(label); ax.grid(alpha=0.25)
            if pi == 0:
                ax.set_ylabel(ylabel)
        # legend in the 10th (empty) cell
        axf[9].axis("off")
        h, l = axf[0].get_legend_handles_labels()
        axf[9].legend(h, l, loc="center", fontsize=12, frameon=True)
        fig.suptitle(f"{ylabel} per input variable -- THIS reproduction's 12-feature "
                     f"L2-ANN (NOT the paper's model; magnitudes ~10x the paper's)",
                     fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(save, dpi=140); print(f"saved -> {save}")

    make("relerr", "Relative Residual", args.fig4)
    make("rmse", "RMSE (el cm$^{-3}$)", args.fig5)


if __name__ == "__main__":
    main()
