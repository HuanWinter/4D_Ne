# NmF2 sub-model (plain ANN) on the prepared Data/Delay/HRO_iono_height0.mat.
# X (N,10) = [mLat, mLon, LT, drv1-4, DoY, F10.7, PF10.7], Y = NmF2, Ref[:,0] = year.
# Filter exp(11) < NmF2 < exp(14.5), 50 < F10.7,PF10.7 < 200. Model log(NmF2),
# features min-max normalised. Temporal holdout: test = years 2009 & 2013.
# (This was the early attempt -- it only reaches ~44% because the HRO table's
# local time is uninformative; see train_submodels.py for the working version.)
#   python Src/train_nmf2_submodel.py --device cuda --epochs 60 --out Projects/nmf2_submodel

import argparse
import json
import os
import time
import numpy as np
import scipy.io as sio
import torch

FEATURES = ["mLat", "mLon", "LT", "drv1", "drv2", "drv3", "drv4", "DoY", "F107", "PF107"]
TEST_YEARS = [2009, 2013]


def main(argv=None):
    ap = argparse.ArgumentParser(description="NmF2 sub-model (ANN, HRO table)")
    ap.add_argument("--data", default="Data/Delay/HRO_iono_height0.mat")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=1029)
    ap.add_argument("--out", default="Projects/nmf2_submodel")
    ap.add_argument("--n-sub", type=int, default=None)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    d = sio.loadmat(args.data)
    X = np.asarray(d["X"], float)
    Y = np.asarray(d["Y"], float).ravel()
    year = np.asarray(d["Ref"], float)[:, 0]
    F107, PF107 = X[:, 8], X[:, 9]
    keep = ((Y > np.exp(11)) & (Y < np.exp(14.5)) &
            (F107 > 50) & (F107 < 200) & (PF107 > 50) & (PF107 < 200) &
            np.all(np.isfinite(X), axis=1) & np.isfinite(Y))
    X, Y, year = X[keep], Y[keep], year[keep]
    print("after filter: %d samples" % X.shape[0])
    if args.n_sub:
        sel = np.random.choice(X.shape[0], args.n_sub, replace=False)
        X, Y, year = X[sel], Y[sel], year[sel]

    logY = np.log(Y)
    te = np.isin(year.astype(int), TEST_YEARS)
    tr = ~te
    print("train=%d test=%d (test yrs %s)" % (tr.sum(), te.sum(), TEST_YEARS))

    lo = X[tr].min(0)
    span = np.where(X[tr].max(0) - lo == 0, 1.0, X[tr].max(0) - lo)
    Xn = (X - lo) / span
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
    optim = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lossf = torch.nn.MSELoss()
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xtr, Ytr),
                                         batch_size=args.batch, shuffle=True)
    Y_te_lin = Y[te]

    def rel_err():
        net.eval()
        with torch.no_grad():
            pn = net(Xte).cpu().numpy().squeeze()
        net.train()
        pred = np.exp(pn * sy + my)
        return (np.mean(np.abs(pred - Y_te_lin) / Y_te_lin) * 100,
                np.sqrt(np.mean((pred - Y_te_lin) ** 2)))

    print("epoch   rel_err(%)   RMSE(el/cm^3)")
    for ep in range(args.epochs):
        for bx, by in loader:
            loss = lossf(net(bx.to(dev)), by.to(dev))
            optim.zero_grad(); loss.backward(); optim.step()
        if (ep + 1) % 5 == 0 or ep == 0:
            re, rm = rel_err()
            print("%5d   %9.2f   %12.0f" % (ep + 1, re, rm))

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
    print("\nFINAL  NmF2 sub-model rel_err=%.2f%%  RMSE=%.0f el/cm^3" % (re, rm))
    print("       (paper: NmF2 model 22.5%%, IRI-2016 33.5%%)   [%.0fs]" % (time.time() - t0))


if __name__ == "__main__":
    main()
