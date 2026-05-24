#!/usr/bin/env python3
"""Train the L2-regularized ANN topside-Ne model (architecture reproduction).

Faithful to the architecture/optimizer in Config.json + Py_Fun.Modelling:
  Linear(in->16)->Tanh->Linear(16->16)->Tanh->Linear(16->8)->Sigmoid->Linear(8->1)
  SGD(lr=5e-5, momentum=0.8, weight_decay=0.001)   # weight_decay == L2 reg
  MSELoss, batch=256, 40 epochs, seed=1, standardized inputs/target.

IMPORTANT SCOPE NOTE
--------------------
This trains on the prepared `Data/data_4d_ne/XY_*.mat`, which contain only the
**12 primary features** (Alt, Lat, Lon, Azi, DST, AE, AP, F10.7, Kp, Vf, DoY,
UT) and Y=[TEC, Ne]. The paper's full model uses **15 features** — these 12
plus the two-stage sub-model outputs hmF2, NmF2 and the scale height VSH — which
are NOT in these files. So this reproduces the ANN *architecture and training on
COSMIC-1*, NOT the paper's headline numbers (which require the 15-feature,
sub-model-augmented inputs). See MIGRATION_REPORT.md.

Usage
-----
  python Src/train_ne_ann.py --xy-dir Data/data_4d_ne --epochs 40 --device cuda \
      --out Projects/ne_ann
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import scipy.io as sio
import torch

# 12-feature column layout of the prepared XY files (inferred + matches
# Py_Fun.Preprocess ordering for the first 12 of its 15 features).
FEATURES12 = ["Altitude", "Latitude", "Longitude", "Azi", "DST", "AE",
              "AP", "F107", "Kp", "Vf", "DoY", "UT"]


def load_xy(xy_dir, n_files=None):
    files = sorted(glob.glob(os.path.join(xy_dir, "XY_*.mat")),
                   key=lambda p: int(p.split("_")[-1].split(".")[0]))
    if n_files:
        files = files[:n_files]
    Xs, Ys = [], []
    for f in files:
        d = sio.loadmat(f)
        Xs.append(np.asarray(d["X"], dtype=np.float64))
        Ys.append(np.asarray(d["Y"], dtype=np.float64))
    X = np.vstack(Xs)
    Y = np.vstack(Ys)
    return X, Y, len(files)


def apply_filters(X, Y, cfg):
    """Apply the Config Para_range filters that the 12 features support.

    Py_Fun.Preprocess also filters on NmF2/hmF2/VSH, which are absent here, so
    those criteria are omitted (documented deviation)."""
    pr = cfg["Para_range"]
    Alt, Lat, Lon = X[:, 0], X[:, 1], X[:, 2]
    F107, Kp, DoY, UT = X[:, 7], X[:, 8] / 10.0, X[:, 10], X[:, 11]
    Ne = Y[:, 1]
    LT = UT + Lon / 15.0
    LT = np.where(LT >= 12, LT - 12, LT)
    LT = np.where(LT < 0, LT + 12, LT)
    m = ((DoY > pr["DoY"][0]) & (DoY < pr["DoY"][1]) &
         (LT > pr["LT"][0]) & (LT < pr["LT"][1]) &
         (Kp > pr["Kp"][0]) & (Kp < pr["Kp"][1]) &
         (F107 > pr["F107"][0]) & (F107 < pr["F107"][1]) &
         (Ne > pr["Ne"][0]) & (Ne < pr["Ne"][1]) &
         (Lat > pr["Latitude"][0]) & (Lat < pr["Latitude"][1]) &
         (Lon > pr["Longitude"][0]) & (Lon < pr["Longitude"][1]) &
         np.all(np.isfinite(X), axis=1))
    return X[m], Ne[m]


def build_net(n_in, hidden, acts, n_out=1):
    layers, sta = [], n_in
    act_map = {"Tanh": torch.nn.Tanh, "Sigmoid": torch.nn.Sigmoid, "ReLU": torch.nn.ReLU}
    for h, a in zip(hidden, acts):
        layers += [torch.nn.Linear(sta, h), act_map[a]()]
        sta = h
    layers += [torch.nn.Linear(sta, n_out)]
    return torch.nn.Sequential(*layers)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xy-dir", default="Data/data_4d_ne")
    ap.add_argument("--config", default="Config.json")
    ap.add_argument("--n-files", type=int, default=None, help="limit XY files (debug)")
    ap.add_argument("--epochs", type=int, default=None, help="override Config epoch_num")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="Projects/ne_ann")
    args = ap.parse_args(argv)

    cfg = json.load(open(args.config))
    nn_p = cfg["NN_parameters"]
    seed = nn_p["Seed_num"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs(args.out, exist_ok=True)

    t0 = time.time()
    X, Y, nf = load_xy(args.xy_dir, args.n_files)
    print(f"Loaded {nf} XY files: X{X.shape} Y{Y.shape}  ({time.time()-t0:.1f}s)")
    X, Ne = apply_filters(X, Y, cfg)
    print(f"After Para_range filter: {X.shape[0]} samples, {X.shape[1]} features")

    # Standardize (matches Py_Fun.Normlise: per-feature for X, scalar for Y).
    meanX, stdX = X.mean(0), X.std(0)
    meanY, stdY = Ne.mean(), Ne.std()
    Xn = (X - meanX) / stdX
    Yn = (Ne - meanY) / stdY

    g = torch.Generator().manual_seed(seed)
    n = Xn.shape[0]
    perm = torch.randperm(n, generator=g).numpy()
    n_tr = int(nn_p["Train_percent"] * n)
    tr, te = perm[:n_tr], perm[n_tr:]
    Xtr = torch.tensor(Xn[tr], dtype=torch.float32)
    Ytr = torch.tensor(Yn[tr], dtype=torch.float32).unsqueeze(1)
    Xte = torch.tensor(Xn[te], dtype=torch.float32)
    Yte = torch.tensor(Yn[te], dtype=torch.float32).unsqueeze(1)
    print(f"train={n_tr} test={n-n_tr}  (Train_percent={nn_p['Train_percent']})")

    dev = torch.device(args.device)
    net = build_net(X.shape[1], nn_p["Hidden_layer"],
                    nn_p["Activation_finction"]).to(dev)
    print(net)
    opt_c = nn_p["Optim"][nn_p["Optim_ind"]]
    optim = torch.optim.SGD(net.parameters(), lr=opt_c["Learning_rate"],
                            momentum=opt_c["momentum"],
                            weight_decay=opt_c["weight_decay"])  # weight_decay = L2
    lossf = getattr(torch.nn, nn_p["Loss"])()
    bs = nn_p["BATCH_SIZE"]
    epochs = args.epochs if args.epochs is not None else nn_p["epoch_num"]

    ds = torch.utils.data.TensorDataset(Xtr, Ytr)
    loader = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True,
                                         num_workers=0, generator=g)
    Xte_d, Yte_d = Xte.to(dev), Yte.to(dev)

    def test_metrics():
        net.eval()
        with torch.no_grad():
            pred_n = net(Xte_d).cpu().numpy().squeeze()
        net.train()
        pred = pred_n * stdY + meanY          # de-standardize to el/cm^3
        ref = Ne[te]
        rmse = np.sqrt(np.mean((pred - ref) ** 2))
        rel = np.mean(np.abs(pred - ref) / np.abs(ref)) * 100
        return rmse, rel

    print("epoch  RMSE(el/cm^3)  RMSE(x1e5)  rel_err(%)")
    for ep in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(dev), by.to(dev)
            loss = lossf(net(bx), by)
            optim.zero_grad(); loss.backward(); optim.step()
        if (ep + 1) % 5 == 0 or ep == 0 or ep == epochs - 1:
            rmse, rel = test_metrics()
            print(f"{ep+1:5d}  {rmse:12.1f}  {rmse/1e5:9.3f}  {rel:9.2f}")

    rmse, rel = test_metrics()
    torch.save(net.state_dict(), os.path.join(args.out, "ne_ann.pt"))
    sio.savemat(os.path.join(args.out, "norm_stats.mat"),
                {"meanX": meanX, "stdX": stdX, "meanY": meanY, "stdY": stdY,
                 "features": np.array(FEATURES12, dtype=object)})
    json.dump({"rmse_el_cm3": float(rmse), "rmse_x1e5": float(rmse/1e5),
               "rel_err_pct": float(rel), "n_train": int(n_tr),
               "n_test": int(n - n_tr), "epochs": epochs, "features": FEATURES12},
              open(os.path.join(args.out, "metrics.json"), "w"), indent=2)
    print(f"\nFINAL  test RMSE={rmse:.1f} el/cm^3 ({rmse/1e5:.3f} x1e5)  "
          f"rel_err={rel:.2f}%   ({time.time()-t0:.0f}s total)")
    print(f"saved -> {args.out}/ (ne_ann.pt, norm_stats.mat, metrics.json)")


if __name__ == "__main__":
    main()
