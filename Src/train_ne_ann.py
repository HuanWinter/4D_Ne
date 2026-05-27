# Train the L2-ANN topside-Ne model on the prepared 12-feature data.
# Architecture/optimizer from Config.json (Py_Fun.Modelling):
#   Linear(in->16)-Tanh-Linear(16->16)-Tanh-Linear(16->8)-Sigmoid-Linear(8->1)
#   SGD(lr=5e-5, momentum=0.8, weight_decay=1e-3), MSELoss, batch 256, seed 1.
# XY_*.mat hold 12 features (Alt,Lat,Lon,Azi,DST,AE,AP,F107,Kp,Vf,DoY,UT); the
# paper's model adds hmF2/NmF2/VSH (see reproduce_ne_15feat.py).
#   python Src/train_ne_ann.py --xy-dir Data/data_4d_ne --device cuda --out Projects/ne_ann

import argparse
import glob
import json
import os
import time
import numpy as np
import scipy.io as sio
import torch

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
    return np.vstack(Xs), np.vstack(Ys), len(files)


# Para_range filters supported by the 12 features (NmF2/hmF2/VSH filters omitted)
def apply_filters(X, Y, cfg):
    pr = cfg["Para_range"]
    Lat, Lon = X[:, 1], X[:, 2]
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
    act_map = {"Tanh": torch.nn.Tanh, "Sigmoid": torch.nn.Sigmoid, "ReLU": torch.nn.ReLU}
    layers, sta = [], n_in
    for h, a in zip(hidden, acts):
        layers += [torch.nn.Linear(sta, h), act_map[a]()]
        sta = h
    layers += [torch.nn.Linear(sta, n_out)]
    return torch.nn.Sequential(*layers)


def main(argv=None):
    ap = argparse.ArgumentParser(description="train the 12-feature L2-ANN")
    ap.add_argument("--xy-dir", default="Data/data_4d_ne")
    ap.add_argument("--config", default="Config.json")
    ap.add_argument("--n-files", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
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
    print("Loaded %d XY files: X%s Y%s  (%.1fs)" % (nf, X.shape, Y.shape, time.time() - t0))
    X, Ne = apply_filters(X, Y, cfg)
    print("After Para_range filter: %d samples, %d features" % (X.shape[0], X.shape[1]))

    # standardise X per-feature, Y scalar (matches Py_Fun.Normlise)
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
    print("train=%d test=%d (Train_percent=%s)" % (n_tr, n - n_tr, nn_p["Train_percent"]))

    dev = torch.device(args.device)
    net = build_net(X.shape[1], nn_p["Hidden_layer"], nn_p["Activation_finction"]).to(dev)
    print(net)
    opt_c = nn_p["Optim"][nn_p["Optim_ind"]]
    optim = torch.optim.SGD(net.parameters(), lr=opt_c["Learning_rate"],
                            momentum=opt_c["momentum"], weight_decay=opt_c["weight_decay"])
    lossf = getattr(torch.nn, nn_p["Loss"])()
    bs = nn_p["BATCH_SIZE"]
    epochs = args.epochs if args.epochs is not None else nn_p["epoch_num"]

    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xtr, Ytr),
                                         batch_size=bs, shuffle=True, num_workers=0, generator=g)
    Xte_d = Xte.to(dev)

    def test_metrics():
        net.eval()
        with torch.no_grad():
            pred = net(Xte_d).cpu().numpy().squeeze() * stdY + meanY
        net.train()
        ref = Ne[te]
        rmse = np.sqrt(np.mean((pred - ref) ** 2))
        rel = np.mean(np.abs(pred - ref) / np.abs(ref)) * 100
        return rmse, rel

    print("epoch  RMSE(el/cm^3)  RMSE(x1e5)  rel_err(%)")
    for ep in range(epochs):
        for bx, by in loader:
            loss = lossf(net(bx.to(dev)), by.to(dev))
            optim.zero_grad(); loss.backward(); optim.step()
        if (ep + 1) % 5 == 0 or ep == 0 or ep == epochs - 1:
            rmse, rel = test_metrics()
            print("%5d  %12.1f  %9.3f  %9.2f" % (ep + 1, rmse, rmse / 1e5, rel))

    rmse, rel = test_metrics()
    torch.save(net.state_dict(), os.path.join(args.out, "ne_ann.pt"))
    sio.savemat(os.path.join(args.out, "norm_stats.mat"),
                {"meanX": meanX, "stdX": stdX, "meanY": meanY, "stdY": stdY,
                 "features": np.array(FEATURES12, dtype=object)})
    json.dump({"rmse_el_cm3": float(rmse), "rmse_x1e5": float(rmse / 1e5),
               "rel_err_pct": float(rel), "n_train": int(n_tr),
               "n_test": int(n - n_tr), "epochs": epochs, "features": FEATURES12},
              open(os.path.join(args.out, "metrics.json"), "w"), indent=2)
    print("\nFINAL  test RMSE=%.1f el/cm^3 (%.3f x1e5)  rel_err=%.2f%%  (%.0fs)"
          % (rmse, rmse / 1e5, rel, time.time() - t0))


if __name__ == "__main__":
    main()
