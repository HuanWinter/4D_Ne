# NmF2 sub-model as a regional deep-kernel KISS-GP (from NmF2_NN_GP-HRO.ipynb).
# A separate GP per (latitude x local-time) region: DNN feature extractor
# (10->100->100->50->2) feeding a gpytorch ExactGP with GridInterpolationKernel
# (SKI), trained on the marginal log-likelihood with AdamW. Data: HRO_iono table;
# test = years 2009 & 2013; metric = pooled median |pred-ref|/ref.
# (Like train_nmf2_submodel.py this only reaches ~44% on the HRO table; the
# working sub-models are in train_submodels.py, rebuilt from raw profiles.)
#   python Src/train_nmf2_dkgp.py --device cuda --cap 40000 --iters 100 --out Projects/nmf2_dkgp

import argparse
import json
import os
import time
import numpy as np
import scipy.io as sio
import torch
import gpytorch

FEAT = list(range(10))
LAT_COL, LT_COL = 0, 2


class FeatureExtractor(torch.nn.Sequential):
    def __init__(self, d):
        super().__init__()
        self.add_module("l1", torch.nn.Linear(d, 100)); self.add_module("r1", torch.nn.ReLU())
        self.add_module("l2", torch.nn.Linear(100, 100)); self.add_module("r2", torch.nn.ReLU())
        self.add_module("l3", torch.nn.Linear(100, 50)); self.add_module("r3", torch.nn.ReLU())
        self.add_module("l4", torch.nn.Linear(50, 2))


class DKGP(gpytorch.models.ExactGP):
    def __init__(self, tx, ty, lik, d):
        super().__init__(tx, ty, lik)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.GridInterpolationKernel(
            gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=2)),
            num_dims=2, grid_size=100)
        self.feature_extractor = FeatureExtractor(d)

    def forward(self, x):
        z = self.feature_extractor(x)
        z = z - z.min(0)[0]
        z = 2 * (z / z.max(0)[0]) - 1          # scale features to [-1,1]
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(z), self.covar_module(z))


def main(argv=None):
    ap = argparse.ArgumentParser(description="NmF2 sub-model (regional KISS-GP)")
    ap.add_argument("--data", default="Data/Delay/HRO_iono_height0.mat")
    ap.add_argument("--target", choices=["log", "linear"], default="log")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--cap", type=int, default=40000, help="max train pts per region")
    ap.add_argument("--lat-interval", type=int, default=30)
    ap.add_argument("--lt-bins", type=int, default=2)
    ap.add_argument("--buffer", type=float, default=5.0)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=1029)
    ap.add_argument("--out", default="Projects/nmf2_dkgp")
    args = ap.parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device(args.device)
    t0 = time.time()

    d = sio.loadmat(args.data)
    X = np.asarray(d["X"], float)
    Y = np.asarray(d["Y"], float).ravel()
    year = np.asarray(d["Ref"], float)[:, 0]
    F107, PF107 = X[:, 8], X[:, 9]
    keep = ((Y > np.exp(11)) & (Y < np.exp(14.5)) & (F107 > 50) & (F107 < 200) &
            (PF107 > 50) & (PF107 < 200) & np.all(np.isfinite(X), 1) & np.isfinite(Y))
    X, Y, year = X[keep][:, FEAT], Y[keep], year[keep]
    lo = X.min(0)
    span = np.where(X.max(0) - lo == 0, 1.0, X.max(0) - lo)
    Xn = (X - lo) / span
    lat_n, lt_n = Xn[:, LAT_COL], Xn[:, LT_COL]
    tgt = np.log(Y) if args.target == "log" else Y.copy()
    ty_mean, ty_std = tgt.mean(), tgt.std()

    test_mask = np.isin(year.astype(int), [2009, 2013])
    print("samples=%d train=%d test=%d target=%s"
          % (X.shape[0], (~test_mask).sum(), test_mask.sum(), args.target))

    n_lat = 180 // args.lat_interval
    preds_all, refs_all = [], []
    for li in range(n_lat):
        lat0 = -90 + li * args.lat_interval
        lo_b = (lat0 - args.buffer + 90) / 180.0
        hi_b = (lat0 + args.lat_interval + args.buffer + 90) / 180.0
        lo_t = (lat0 + 90) / 180.0
        hi_t = (lat0 + args.lat_interval + 90) / 180.0
        for ti in range(args.lt_bins):
            t_lo, t_hi = ti / args.lt_bins, (ti + 1) / args.lt_bins
            tr = (~test_mask) & (lat_n > lo_b) & (lat_n <= hi_b) & (lt_n > t_lo) & (lt_n <= t_hi)
            teb = test_mask & (lat_n > lo_t) & (lat_n <= hi_t) & (lt_n > t_lo) & (lt_n <= t_hi)
            ntr, nte = int(tr.sum()), int(teb.sum())
            if ntr < 100 or nte < 10:
                continue
            tr_idx = np.where(tr)[0]
            if ntr > args.cap:
                tr_idx = np.random.choice(tr_idx, args.cap, replace=False)
            txn = torch.tensor((tgt[tr_idx] - ty_mean) / ty_std, dtype=torch.float32, device=dev)
            tx = torch.tensor(Xn[tr_idx], dtype=torch.float32, device=dev)
            te_i = np.where(teb)[0]
            tex = torch.tensor(Xn[te_i], dtype=torch.float32, device=dev)

            lik = gpytorch.likelihoods.GaussianLikelihood().to(dev)
            model = DKGP(tx, txn, lik, X.shape[1]).to(dev)
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)
            model.train(); lik.train()
            for _ in range(args.iters):
                opt.zero_grad(); loss = -mll(model(tx), txn); loss.backward(); opt.step()
            model.eval(); lik.eval()
            with torch.no_grad(), gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var():
                pn = lik(model(tex)).mean.cpu().numpy()
            pred_tgt = pn * ty_std + ty_mean
            pred = np.exp(pred_tgt) if args.target == "log" else pred_tgt
            preds_all.append(pred)
            refs_all.append(Y[te_i])
            re = np.median(np.abs(pred - Y[te_i]) / Y[te_i]) * 100
            print("  lat[%+4d,%+4d] LT-bin%d: ntr=%d nte=%d median_relerr=%.1f%%"
                  % (lat0, lat0 + args.lat_interval, ti, ntr, nte, re))

    pred = np.concatenate(preds_all)
    ref = np.concatenate(refs_all)
    re = np.abs(pred - ref) / ref
    med, mean = np.median(re) * 100, np.mean(re) * 100
    rmse = np.sqrt(np.mean((pred - ref) ** 2))
    print("\nFINAL NmF2 DKL-GP  pooled MEDIAN rel_err=%.2f%% (mean=%.1f%%) RMSE=%.0f"
          % (med, mean, rmse))
    print("      paper NmF2=22.5%%  IRI-2016=33.5%%   [%.0fs]" % (time.time() - t0))
    json.dump({"median_relerr_pct": float(med), "mean_relerr_pct": float(mean),
               "rmse_el_cm3": float(rmse), "target": args.target, "cap": args.cap,
               "paper_nmf2": 22.5, "paper_iri": 33.5, "n_test_used": int(ref.size)},
              open(os.path.join(args.out, "metrics.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
