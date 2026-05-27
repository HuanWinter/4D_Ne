# Reproduce the topside-Ne model with the full 15 features, built straight from
# the COSMIC-1 ionPrf profiles (NmF2/hmF2/VSH measured per profile -- no sub-models).
# Per profile: Lat=edmaxlat, Lon=edmaxlon, Azi=median OCC_azi, NmF2=edmax,
# hmF2=edmaxalt, VSH=1/e scale height, drivers=OMNI2; each topside point -> (Alt, Ne).
# Feature order = Py_Fun.Preprocess:
#   [Alt, Lat, Lon, Azi, DST, AE, AP, F107, Kp, Vf, DoY, UT, hmF2, NmF2, VSH] -> Ne
# Out-of-sample test = years 2009 & 2013.
#   python Src/reproduce_ne_15feat.py --list <cosmic_list_local.txt> --n-profiles 40000 \
#       --device cuda --out Projects/ne15

import argparse
import datetime as dt
import json
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_fig1_profile import compute_vsh                  # noqa: E402
from make_delay_files import load_omni, omni_at            # noqa: E402
from train_ne_ann import build_net                         # noqa: E402

# OMNI2 order is [DST, AE, ap, F107, Kp, FlowSpeed, Bx, By, Bz]; we use the first 6
OMNI_PICK = [0, 1, 2, 3, 4, 5]
FEAT15 = ["Altitude", "Latitude", "Longitude", "Azi", "DST", "AE", "AP",
          "F107", "Kp", "Vf", "DoY", "UT", "hmF2", "NmF2", "VSH"]


def build_samples(paths, omni_index, alt_lo=200, alt_hi=700):
    import netCDF4 as nc
    rows, tgt, yr = [], [], []
    for k, p in enumerate(paths):
        if k % 2000 == 0:
            print("  read %d/%d" % (k, len(paths)), end="\r", file=sys.stderr)
        try:
            f = nc.Dataset(p.strip())
            alt = np.asarray(f.variables["MSL_alt"][:], float)
            ne = np.asarray(f.variables["ELEC_dens"][:], float)
            azi = np.asarray(f.variables["OCC_azi"][:], float)
            year = int(f.getncattr("year")); month = int(f.getncattr("month"))
            day = int(f.getncattr("day"))
            ut = f.getncattr("hour") + f.getncattr("minute") / 60 + f.getncattr("second") / 3600
            nmf2 = float(f.getncattr("edmax")); hmf2 = float(f.getncattr("edmaxalt"))
            lat = float(f.getncattr("edmaxlat")); lon = float(f.getncattr("edmaxlon"))
            f.close()
        except Exception:
            continue
        vsh = compute_vsh(alt.copy(), ne.copy(), hmf2)[0]
        if not np.isfinite(vsh):
            continue
        rt = dt.datetime(year, month, day, int(ut))
        drv = omni_at(omni_index, rt)[OMNI_PICK]    # DST,AE,AP,F107,Kp,Vf
        if not np.all(np.isfinite(drv)):
            continue
        doy = month + day / 35.0 - 1.0
        azi_pk = np.nanmedian(azi)
        m = (alt > max(hmf2, alt_lo)) & (alt < alt_hi) & (ne > 0)   # topside points
        if m.sum() < 3:
            continue
        for h, nev in zip(alt[m], ne[m]):
            rows.append([h, lat, lon, azi_pk, drv[0], drv[1], drv[2], drv[3],
                         drv[4], drv[5], doy, ut, hmf2, nmf2, vsh])
            tgt.append(nev); yr.append(year)
    return np.asarray(rows), np.asarray(tgt), np.asarray(yr)


def main(argv=None):
    default = "/glade/derecho/scratch/%s/cosmic/cosmic_list_local.txt" % os.environ.get("USER", "andonghu")
    ap = argparse.ArgumentParser(description="reproduce the 15-feature topside-Ne model")
    ap.add_argument("--list", default=default)
    ap.add_argument("--n-profiles", type=int, default=20000)
    ap.add_argument("--omni-cache", default="Data/omni2_cache")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="Projects/ne15")
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    import linecache as lc
    paths = [l for l in lc.getlines(args.list) if l.strip()]
    sel = rng.choice(len(paths), min(args.n_profiles, len(paths)), replace=False)
    paths = [paths[i] for i in sel]
    print("sampling %d profiles; loading OMNI2 2006-2015 ..." % len(paths))
    omni = load_omni(range(2006, 2016), cache_dir=args.omni_cache, source="omni2")

    X, Ne, yr = build_samples(paths, omni)
    print("\nbuilt %d topside samples, %d features" % (X.shape[0], X.shape[1]))
    cfg = json.load(open("Config.json"))
    pr = cfg["Para_range"]
    m = ((Ne > pr["Ne"][0]) & (Ne < pr["Ne"][1]) &
         (X[:, 13] > pr["NmF2"][0]) & (X[:, 13] < pr["NmF2"][1]) &
         (X[:, 12] > pr["hmF2"][0]) & (X[:, 12] < pr["hmF2"][1]) &
         (X[:, 14] > pr["VSH"][0]) & (X[:, 14] < pr["VSH"][1]) & np.all(np.isfinite(X), 1))
    X, Ne, yr = X[m], Ne[m], yr[m]
    print("after filter: %d samples" % X.shape[0])

    te = np.isin(yr.astype(int), [2009, 2013])
    tr = ~te
    if te.sum() < 100:    # sample missed those years -> random 80/20
        idx = rng.permutation(X.shape[0]); ntr = int(0.8 * len(idx))
        tr = np.zeros(X.shape[0], bool); tr[idx[:ntr]] = True; te = ~tr
        print("(year holdout too small -> random 80/20 split)")
    print("train=%d test=%d" % (tr.sum(), te.sum()))

    meanX, stdX = X[tr].mean(0), X[tr].std(0) + 1e-9
    meanY, stdY = Ne[tr].mean(), Ne[tr].std()
    Xn = (X - meanX) / stdX
    Yn = (Ne - meanY) / stdY
    dev = torch.device(args.device)
    net = build_net(15, cfg["NN_parameters"]["Hidden_layer"],
                    cfg["NN_parameters"]["Activation_finction"]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    lossf = torch.nn.MSELoss()
    Xtr = torch.tensor(Xn[tr], dtype=torch.float32)
    Ytr = torch.tensor(Yn[tr], dtype=torch.float32).unsqueeze(1)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xtr, Ytr),
                                         batch_size=512, shuffle=True)
    Xte = torch.tensor(Xn[te], dtype=torch.float32).to(dev)
    ref = Ne[te]

    def metrics():
        net.eval()
        with torch.no_grad():
            p = net(Xte).cpu().numpy().squeeze() * stdY + meanY
        net.train()
        re = np.abs(p - ref) / ref
        return np.mean(re) * 100, np.median(re) * 100, np.sqrt(np.mean((p - ref) ** 2)) / 1e5

    print("epoch  mean_relerr%  median%  RMSE(x1e5)")
    for ep in range(args.epochs):
        for bx, by in loader:
            l = lossf(net(bx.to(dev)), by.to(dev)); opt.zero_grad(); l.backward(); opt.step()
        if (ep + 1) % 5 == 0 or ep == 0:
            me, md, rm = metrics()
            print("%5d  %11.2f  %7.2f  %8.3f" % (ep + 1, me, md, rm))
    me, md, rm = metrics()

    net.eval()
    with torch.no_grad():
        pred_te = net(Xte).cpu().numpy().squeeze() * stdY + meanY
    torch.save(net.state_dict(), os.path.join(args.out, "ne15.pt"))
    np.savez(os.path.join(args.out, "test_arrays.npz"),
             X=X[te].astype(np.float32), pred=pred_te.astype(np.float32),
             ref=Ne[te].astype(np.float32), features=np.array(FEAT15, dtype=object))
    import scipy.io as sio
    sio.savemat(os.path.join(args.out, "norm.mat"),
                {"meanX": meanX, "stdX": stdX, "meanY": meanY, "stdY": stdY})
    json.dump({"mean_relerr_pct": float(me), "median_relerr_pct": float(md),
               "rmse_x1e5": float(rm), "n_train": int(tr.sum()), "n_test": int(te.sum()),
               "features": FEAT15}, open(os.path.join(args.out, "metrics.json"), "w"), indent=2)
    print("\nFINAL 15-feature model: mean rel-err=%.2f%%  median=%.2f%%  RMSE=%.3fx1e5  (%.0fs)"
          % (me, md, rm, time.time() - t0))


if __name__ == "__main__":
    main()
