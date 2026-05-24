#!/usr/bin/env python3
"""Train the NmF2 sub-model (stage 1 of the paper's two-stage / multi-fidelity build).

Reproduces the NmF2 ANN from NmF2_NN_GP-HRO.ipynb using the prepared data
`Data/Delay/HRO_iono_height0.mat`:
  X   : (N, 10) features = [mLat, mLon, LT, drv1, drv2, drv3, drv4, DoY, F10.7, PF10.7]
  Y   : (N, 1) NmF2 (el/cm^3)
  Ref : (N, 6) = [year, month, day, LT, lat, lon]

Recipe (from the notebook):
  - filter: exp(11) < NmF2 < exp(14.5), 50 < F10.7,PF10.7 < 200, finite
  - TEMPORAL holdout: test = years {2009 (low solar), 2013 (high solar)},
    train = all other years. (No random split -> no profile leakage.)
  - model log(NmF2); features min-max normalised to [0,1] (norm_ah)
  - net: Linear(10->32)->ReLU->Linear(32->16)->ReLU->Linear(16->8)->ReLU->Linear(8->1)
  - L2 regularisation via weight_decay.

Relative error (linear NmF2 space) is reported and compared to the paper
(NmF2 sub-model 22.5%, IRI-2016 33.5%).

NOTE: the notebook used skorch defaults; the exact optimizer/epochs are not
fully pinned down there, so Adam(lr, weight_decay) is used here. This is a faithful
reproduction of the *architecture, data, split, and target*, with the optimizer
chosen to converge; treat the optimizer/epoch count as the reproduction's choice.

Usage:
  python Src/train_nmf2_submodel.py --device cuda --epochs 60 --out Projects/nmf2_submodel
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import scipy.io as sio
import torch

FEATURES = ["mLat", "mLon", "LT", "drv1", "drv2", "drv3", "drv4", "DoY", "F107", "PF107"]
TEST_YEARS = [2009, 2013]


def minmax_fit(X):
    lo = X.min(0)
    span = X.max(0) - lo
    span = np.where(span == 0, 1.0, span)
    return lo, span


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="Data/Delay/HRO_iono_height0.mat")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=1029)
    ap.add_argument("--out", default="Projects/nmf2_submodel")
    ap.add_argument("--n-sub", type=int, default=None, help="subsample N (debug)")
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    d = sio.loadmat(args.data)
    X, Y, Ref = np.asarray(d["X"], float), np.asarray(d["Y"], float).ravel(), np.asarray(d["Ref"], float)
    year = Ref[:, 0]
    F107, PF107 = X[:, 8], X[:, 9]
    keep = ((Y > np.exp(11)) & (Y < np.exp(14.5)) &
            (F107 > 50) & (F107 < 200) & (PF107 > 50) & (PF107 < 200) &
            np.all(np.isfinite(X), axis=1) & np.isfinite(Y))
    X, Y, year = X[keep], Y[keep], year[keep]
    print(f"after filter: {X.shape[0]} samples")
    if args.n_sub:
        sel = np.random.choice(X.shape[0], args.n_sub, replace=False)
        X, Y, year = X[sel], Y[sel], year[sel]

    logY = np.log(Y)
    test_mask = np.isin(year.astype(int), TEST_YEARS)
    tr, te = ~test_mask, test_mask
    print(f"train={tr.sum()} (years != {TEST_YEARS})  test={te.sum()} (years {TEST_YEARS})")

    # min-max normalise features on TRAIN only (no test leakage)
    lo, span = minmax_fit(X[tr])
    Xn = (X - lo) / span
    # standardise log-target on train
    my, sy = logY[tr].mean(), logY[tr].std()
    Yn = (logY - my) / sy

    dev = torch.device(args.device)
    Xtr = torch.tensor(Xn[tr], dtype=torch.float32)
    Ytr = torch.tensor(Yn[tr], dtype=torch.float32).unsqueeze(1)
    Xte = torch.tensor(Xn[te], dtype=torch.float32).to(dev)

    net = torch.nn.Sequential(
        torch.nn.Linear(10, 32), torch.nn.ReLU(),
        torch.nn.Linear(32, 16), torch.nn.ReLU(),
        torch.nn.Linear(16, 8), torch.nn.ReLU(),
        torch.nn.Linear(8, 1)).to(dev)
    print(net)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lossf = torch.nn.MSELoss()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xtr, Ytr), batch_size=args.batch,
        shuffle=True, num_workers=0)

    Y_te_lin = Y[te]

    def rel_err():
        net.eval()
        with torch.no_grad():
            pn = net(Xte).cpu().numpy().squeeze()
        net.train()
        pred = np.exp(pn * sy + my)             # back to linear NmF2
        return np.mean(np.abs(pred - Y_te_lin) / Y_te_lin) * 100, \
               np.sqrt(np.mean((pred - Y_te_lin) ** 2))

    print("epoch   rel_err(%)   RMSE(el/cm^3)")
    for ep in range(args.epochs):
        for bx, by in loader:
            bx, by = bx.to(dev), by.to(dev)
            loss = lossf(net(bx), by)
            optim.zero_grad(); loss.backward(); optim.step()
        if (ep + 1) % 5 == 0 or ep == 0:
            re, rm = rel_err()
            print(f"{ep+1:5d}   {re:9.2f}   {rm:12.0f}")

    re, rm = rel_err()
    torch.save(net.state_dict(), os.path.join(args.out, "nmf2_submodel.pt"))
    sio.savemat(os.path.join(args.out, "nmf2_norm.mat"),
                {"lo": lo, "span": span, "meanY": my, "stdY": sy,
                 "features": np.array(FEATURES, dtype=object)})
    json.dump({"rel_err_pct": float(re), "rmse_el_cm3": float(rm),
               "paper_nmf2_relerr": 22.5, "paper_iri_relerr": 33.5,
               "n_train": int(tr.sum()), "n_test": int(te.sum()),
               "test_years": TEST_YEARS, "epochs": args.epochs},
              open(os.path.join(args.out, "metrics.json"), "w"), indent=2)
    print(f"\nFINAL  NmF2 sub-model rel_err={re:.2f}%  RMSE={rm:.0f} el/cm^3")
    print(f"       (paper: NmF2 model 22.5%, IRI-2016 33.5%)   [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
