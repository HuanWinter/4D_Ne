#!/usr/bin/env python3
"""NmF2 sub-model as a regional Deep-Kernel KISS-GP (faithful to NmF2_NN_GP-HRO.ipynb).

Reproduces the paper's actual NmF2 sub-model design (cells 6-8):
  * NOT one global model -- a separate model per (latitude x local-time) region:
    lat bins of 30 deg (with +/-5 deg buffer), 2 local-time bins.
  * Deep Kernel Learning GP (KISS-GP / SKI): a DNN feature extractor
    (10->100->100->50->2) feeding gpytorch ExactGP with
    GridInterpolationKernel(ScaleKernel(RBF(ard=2)), grid_size=100).
  * AdamW(lr=0.02), 100 iterations on the exact marginal log-likelihood.
  * metric: MEDIAN relative error |pred-ref|/ref, pooled over test points,
    compared to the paper (NmF2 22.5%, IRI-2016 33.5%).

Data: Data/Delay/HRO_iono_height0.mat  (X[:,0:10], Y=NmF2, Ref[:,0]=year).
Split: train = years not in {2009,2013}; test = {2009 (low solar), 2013 (high)}.

Ambiguities in the original notebook handled explicitly (documented choices):
  * target transform: --target {log,linear} (default log; standard for NmF2).
  * per-region training subsample (--cap) to keep ExactGP/SKI tractable.
These are best-effort faithful choices; see MIGRATION_REPORT.md.
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np, scipy.io as sio, torch, gpytorch

FEAT = list(range(10))   # HRO X columns 0..9
LAT_COL, LT_COL = 0, 2   # latitude, local-time columns in X


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
        z = 2 * (z / z.max(0)[0]) - 1          # scale projected features to [-1,1]
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(z), self.covar_module(z))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device(args.device); t0 = time.time()

    d = sio.loadmat(args.data)
    X, Y, Ref = np.asarray(d["X"], float), np.asarray(d["Y"], float).ravel(), np.asarray(d["Ref"], float)
    year, F107, PF107 = Ref[:, 0], X[:, 8], X[:, 9]
    keep = ((Y > np.exp(11)) & (Y < np.exp(14.5)) & (F107 > 50) & (F107 < 200) &
            (PF107 > 50) & (PF107 < 200) & np.all(np.isfinite(X), 1) & np.isfinite(Y))
    X, Y, year = X[keep][:, FEAT], Y[keep], year[keep]
    # min-max normalise features (norm_ah) using full-data range
    lo = X.min(0); span = np.where(X.max(0) - lo == 0, 1.0, X.max(0) - lo)
    Xn = (X - lo) / span
    lat_n, lt_n = Xn[:, LAT_COL], Xn[:, LT_COL]   # normalised lat/LT for binning
    tgt = np.log(Y) if args.target == "log" else Y.copy()
    ty_mean, ty_std = tgt.mean(), tgt.std()       # standardise target for stable GP

    test_mask = np.isin(year.astype(int), [2009, 2013])
    print(f"samples={X.shape[0]} train={(~test_mask).sum()} test={test_mask.sum()} target={args.target}")

    n_lat = 180 // args.lat_interval
    preds_all, refs_all = [], []
    for li in range(n_lat):
        lat0 = -90 + li * args.lat_interval
        lo_b = (lat0 - args.buffer + 90) / 180.0; hi_b = (lat0 + args.lat_interval + args.buffer + 90) / 180.0
        lo_t = (lat0 + 90) / 180.0; hi_t = (lat0 + args.lat_interval + 90) / 180.0
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
            preds_all.append(pred); refs_all.append(Y[te_i])
            re = np.median(np.abs(pred - Y[te_i]) / Y[te_i]) * 100
            print(f"  lat[{lat0:+4d},{lat0+args.lat_interval:+4d}] LT-bin{ti}: "
                  f"ntr={ntr} nte={nte} median_relerr={re:.1f}%")

    pred = np.concatenate(preds_all); ref = np.concatenate(refs_all)
    re = np.abs(pred - ref) / ref
    med, mean = np.median(re) * 100, np.mean(re) * 100
    rmse = np.sqrt(np.mean((pred - ref) ** 2))
    print(f"\nFINAL NmF2 DKL-GP  pooled MEDIAN rel_err={med:.2f}%  (mean={mean:.1f}%)  RMSE={rmse:.0f}")
    print(f"      paper NmF2={22.5}%  IRI-2016={33.5}%   [{time.time()-t0:.0f}s]")
    json.dump({"median_relerr_pct": float(med), "mean_relerr_pct": float(mean),
               "rmse_el_cm3": float(rmse), "target": args.target, "cap": args.cap,
               "paper_nmf2": 22.5, "paper_iri": 33.5, "n_test_used": int(ref.size)},
              open(os.path.join(args.out, "metrics.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
