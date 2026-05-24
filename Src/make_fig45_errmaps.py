#!/usr/bin/env python3
"""Reproduce manuscript Figs. 4 & 5: relative-error and RMSE per input variable.

For each input variable, bin the samples and plot the model's relative error
(Fig. 4) and RMSE (Fig. 5) within each bin, for the model's train vs held-out
test split.

*** IMPORTANT: this uses THIS reproduction's 12-feature L2-ANN
(`Projects/ne_ann/ne_ann.pt`), whose overall test rel-err is ~86% because it
lacks the paper's hmF2/NmF2/VSH sub-model features. So these maps show the
SHAPE of the error-vs-variable dependence for our model, NOT the paper's
numbers. See MIGRATION_REPORT.md sec. 9. ***

Evaluation is on the model's own held-out TEST split (reproduced exactly from
`train_ne_ann.py`: torch.randperm(seed=1), 80/20), so test errors are on data
the model never saw.

Usage:
  python Src/make_fig45_errmaps.py --model Projects/ne_ann \
      --fig4 fig4_relerr.png --fig5 fig5_rmse.png
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import scipy.io as sio
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_ne_ann import load_xy, apply_filters, build_net, FEATURES12

UNITS = ["km", "deg", "deg", "deg", "nT", "nT", "nT", "sfu", "", "km/s", "month", "hr"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xy-dir", default="Data/data_4d_ne")
    ap.add_argument("--config", default="Config.json")
    ap.add_argument("--model", default="Projects/ne_ann")
    ap.add_argument("--bins", type=int, default=25)
    ap.add_argument("--max-eval", type=int, default=800000, help="subsample per split for speed")
    ap.add_argument("--fig4", default="fig4_relerr.png")
    ap.add_argument("--fig5", default="fig5_rmse.png")
    args = ap.parse_args(argv)

    cfg = json.load(open(args.config))
    seed = cfg["NN_parameters"]["Seed_num"]
    X, Y, _ = load_xy(args.xy_dir)
    X, Ne = apply_filters(X, Y, cfg)
    n = X.shape[0]

    # reproduce train_ne_ann.py split exactly
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).numpy()
    n_tr = int(cfg["NN_parameters"]["Train_percent"] * n)
    split = {"train": perm[:n_tr], "test": perm[n_tr:]}
    print({k: len(v) for k, v in split.items()})

    ns = sio.loadmat(os.path.join(args.model, "norm_stats.mat"))
    meanX, stdX = ns["meanX"].ravel(), ns["stdX"].ravel()
    meanY, stdY = float(ns["meanY"]), float(ns["stdY"])
    net = build_net(X.shape[1], cfg["NN_parameters"]["Hidden_layer"],
                    cfg["NN_parameters"]["Activation_finction"])
    net.load_state_dict(torch.load(os.path.join(args.model, "ne_ann.pt"), map_location="cpu"))
    net.eval()

    def predict(idx):
        Xn = ((X[idx] - meanX) / stdX).astype(np.float32)
        out = []
        with torch.no_grad():
            for i in range(0, len(idx), 200000):
                out.append(net(torch.from_numpy(Xn[i:i+200000])).numpy().squeeze())
        pred = np.concatenate(out) * stdY + meanY
        return pred, Ne[idx]

    rng = np.random.default_rng(0)
    data = {}
    for k, idx in split.items():
        if len(idx) > args.max_eval:
            idx = rng.choice(idx, args.max_eval, replace=False)
        pred, ref = predict(idx)
        data[k] = (X[idx], pred, ref)
        print(f"{k}: median rel-err = {np.median(np.abs(pred-ref)/ref)*100:.1f}%")

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"train": "tab:blue", "test": "tab:green"}

    def make(metric, title, save):
        fig, axs = plt.subplots(3, 4, figsize=(20, 12))
        for vi, ax in enumerate(axs.flat):
            if vi >= len(FEATURES12):
                ax.axis("off"); continue
            allx = np.concatenate([data[k][0][:, vi] for k in data])
            edges = np.linspace(np.percentile(allx, 0.5), np.percentile(allx, 99.5), args.bins + 1)
            cen = 0.5 * (edges[:-1] + edges[1:])
            for k in ("train", "test"):
                Xk, pred, ref = data[k]
                xv = Xk[:, vi]
                vals = np.full(args.bins, np.nan)
                for b in range(args.bins):
                    m = (xv >= edges[b]) & (xv < edges[b+1])
                    if m.sum() >= 50:
                        if metric == "relerr":
                            vals[b] = np.median(np.abs(pred[m]-ref[m])/ref[m]) * 100
                        else:
                            vals[b] = np.sqrt(np.mean((pred[m]-ref[m])**2)) / 1e5
                ax.plot(cen, vals, "o-", ms=3, color=colors[k], label=k)
            u = f" ({UNITS[vi]})" if UNITS[vi] else ""
            ax.set_title(FEATURES12[vi]); ax.set_xlabel(FEATURES12[vi]+u)
            ax.set_ylabel("median rel-err (%)" if metric == "relerr" else "RMSE ($\\times10^5$)")
            ax.grid(alpha=0.3)
            if vi == 0:
                ax.legend()
        fig.suptitle(title, fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        fig.savefig(save, dpi=130); print(f"saved -> {save}")

    make("relerr", "Fig. 4 — median relative error per variable "
         "(THIS reproduction's 12-feature L2-ANN; not the paper's numbers)", args.fig4)
    make("rmse", "Fig. 5 — RMSE per variable "
         "(THIS reproduction's 12-feature L2-ANN; not the paper's numbers)", args.fig5)


if __name__ == "__main__":
    main()
