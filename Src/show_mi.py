# Plot MI vs time gap for each driver (3x3), Python version of show_MI.m.
# Each panel: full-sample MI, resample mean, and mean +/- 3*std.
#   python show_mi.py mi_results.npz --save mi_plot.png

import argparse
import matplotlib
import numpy as np


def show_mi(MI, MI_mean, MI_std, X, vari_set, save=None, show=True):
    import matplotlib.pyplot as plt

    MI = np.asarray(MI)
    MI_mean = np.asarray(MI_mean)
    MI_std = np.asarray(MI_std)
    X = np.asarray(X)
    n_vars = MI.shape[1]

    ncol = 3
    nrow = int(np.ceil(n_vars / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 12))
    axes = np.atleast_1d(axes).ravel()

    for i in range(n_vars):
        ax = axes[i]
        ax.plot(X, MI[:, i], "yo-", linewidth=5, label="all")
        ax.plot(X, MI_mean[:, i], "g*-", label="mean")
        ax.plot(X, MI_mean[:, i] - 3 * MI_std[:, i], "r--", label="lower")
        ax.plot(X, MI_mean[:, i] + 3 * MI_std[:, i], "b--", label="upper")
        ax.set_xlabel("Time gap (h)")
        ax.set_ylabel("MI (nats)")
        ax.legend(fontsize=7)
        ax.set_title("MI between NmF2 and %s" % vari_set[i])
    for j in range(n_vars, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=120)
        print("Saved figure to", save)
    if show:
        plt.show()
    return fig


def main(argv=None):
    ap = argparse.ArgumentParser(description="Plot mi_regressor .npz results")
    ap.add_argument("npz")
    ap.add_argument("--save", default=None)
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args(argv)

    if args.no_show or args.save:
        matplotlib.use("Agg")

    d = np.load(args.npz, allow_pickle=True)
    show_mi(d["MI"], d["MI_mean"], d["MI_std"], d["lags"],
            list(d["vari_set"]), save=args.save, show=not args.no_show)


if __name__ == "__main__":
    main()
