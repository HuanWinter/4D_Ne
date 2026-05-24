#!/usr/bin/env python3
"""Reproduce the paper's topside-Ne model with the FULL 15 features, built from
the COSMIC-1 ionPrf profiles (NmF2/hmF2/VSH measured per profile, no sub-models).

Builds per-(topside-altitude-point) samples with the 15 features in the exact
order Py_Fun.Preprocess expects:
  [Alt, Lat, Lon, Azi, DST, AE, AP, F10.7, Kp, Vf, DoY, UT, hmF2, NmF2, VSH] -> Ne
where per profile: Lat=edmaxlat, Lon=edmaxlon, Azi=OCC_azi(peak), NmF2=edmax,
hmF2=edmaxalt, VSH=1/e-drop scale height (compute_vsh), drivers=OMNI2 at RO_time,
and each topside altitude point gives (Alt, Ne).

Trains the paper's L2-ANN (Config.json architecture) and reports relative error.
Year-based holdout (test = 2009 & 2013) for an honest out-of-sample estimate.

Usage:
  python Src/reproduce_ne_15feat.py --list /glade/derecho/scratch/$USER/cosmic/cosmic_list_local.txt \
      --n-profiles 20000 --device cuda --out Projects/ne15
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_fig1_profile import compute_vsh
from make_delay_files import load_omni, omni_at          # OMNI2 loader + lookup
from train_ne_ann import build_net
import datetime as dt

# OMNI2 FEATURES order from make_delay_files:
#   [DST, AE, ap, F10.7, Kp, FlowSpeed, Bx, By, Bz]
# main-model drivers needed: DST, AE, AP, F10.7, Kp, Vf(=FlowSpeed)
OMNI_PICK = [0, 1, 2, 3, 4, 5]
FEAT15 = ["Altitude", "Latitude", "Longitude", "Azi", "DST", "AE", "AP",
          "F107", "Kp", "Vf", "DoY", "UT", "hmF2", "NmF2", "VSH"]


def build_samples(paths, omni_index, alt_lo=200, alt_hi=700):
    import netCDF4 as nc
    rows, tgt, yr = [], [], []
    for k, p in enumerate(paths):
        if k % 2000 == 0:
            print(f"  read {k}/{len(paths)}", end="\r", file=sys.stderr)
        try:
            f = nc.Dataset(p.strip())
            alt = np.asarray(f.variables["MSL_alt"][:], float)
            ne = np.asarray(f.variables["ELEC_dens"][:], float)
            azi = np.asarray(f.variables["OCC_azi"][:], float)
            year = int(f.getncattr("year")); month = int(f.getncattr("month"))
            day = int(f.getncattr("day"))
            ut = f.getncattr("hour") + f.getncattr("minute")/60 + f.getncattr("second")/3600
            nmf2 = float(f.getncattr("edmax")); hmf2 = float(f.getncattr("edmaxalt"))
            lat = float(f.getncattr("edmaxlat")); lon = float(f.getncattr("edmaxlon"))
            f.close()
        except Exception:
            continue
        vsh, *_ = compute_vsh(alt.copy(), ne.copy(), hmf2)
        if not np.isfinite(vsh):
            continue
        rt = dt.datetime(year, month, day, int(ut))
        drv = omni_at(omni_index, rt)[OMNI_PICK]       # DST,AE,AP,F107,Kp,Vf
        if not np.all(np.isfinite(drv)):
            continue
        doy = month + day/35.0 - 1.0
        azi_pk = np.nanmedian(azi)
        # topside points: above the peak, in altitude range, positive Ne
        m = (alt > max(hmf2, alt_lo)) & (alt < alt_hi) & (ne > 0)
        if m.sum() < 3:
            continue
        for h, nev in zip(alt[m], ne[m]):
            rows.append([h, lat, lon, azi_pk, drv[0], drv[1], drv[2], drv[3],
                         drv[4], drv[5], doy, ut, hmf2, nmf2, vsh])
            tgt.append(nev); yr.append(year)
    return np.asarray(rows), np.asarray(tgt), np.asarray(yr)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", default="/glade/derecho/scratch/{}/cosmic/cosmic_list_local.txt".format(os.environ.get("USER", "andonghu")))
    ap.add_argument("--n-profiles", type=int, default=20000)
    ap.add_argument("--omni-cache", default="Data/omni2_cache")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="Projects/ne15")
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed); torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True); t0 = time.time()

    import linecache as lc
    paths = [l for l in lc.getlines(args.list) if l.strip()]
    paths = [paths[i] for i in rng.choice(len(paths), min(args.n_profiles, len(paths)), replace=False)]
    print(f"sampling {len(paths)} profiles")
    print("loading OMNI2 2006-2015 ...")
    omni = load_omni(range(2006, 2016), cache_dir=args.omni_cache, source="omni2")

    X, Ne, yr = build_samples(paths, omni)
    print(f"\nbuilt {X.shape[0]} topside samples, {X.shape[1]} features")
    # Para_range-style filter on Ne and peak
    cfg = json.load(open("Config.json")); pr = cfg["Para_range"]
    m = ((Ne > pr["Ne"][0]) & (Ne < pr["Ne"][1]) &
         (X[:, 13] > pr["NmF2"][0]) & (X[:, 13] < pr["NmF2"][1]) &
         (X[:, 12] > pr["hmF2"][0]) & (X[:, 12] < pr["hmF2"][1]) &
         (X[:, 14] > pr["VSH"][0]) & (X[:, 14] < pr["VSH"][1]) & np.all(np.isfinite(X), 1))
    X, Ne, yr = X[m], Ne[m], yr[m]
    print(f"after filter: {X.shape[0]} samples")

    te = np.isin(yr.astype(int), [2009, 2013]); tr = ~te
    if te.sum() < 100:      # fallback to random split if the sample missed those years
        idx = rng.permutation(X.shape[0]); ntr = int(0.8*len(idx))
        tr = np.zeros(X.shape[0], bool); tr[idx[:ntr]] = True; te = ~tr
        print("(year holdout too small -> random 80/20 split)")
    print(f"train={tr.sum()} test={te.sum()}")

    meanX, stdX = X[tr].mean(0), X[tr].std(0) + 1e-9
    meanY, stdY = Ne[tr].mean(), Ne[tr].std()
    Xn = (X - meanX) / stdX; Yn = (Ne - meanY) / stdY
    dev = torch.device(args.device)
    net = build_net(15, cfg["NN_parameters"]["Hidden_layer"],
                    cfg["NN_parameters"]["Activation_finction"]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    lossf = torch.nn.MSELoss()
    Xtr = torch.tensor(Xn[tr], dtype=torch.float32); Ytr = torch.tensor(Yn[tr], dtype=torch.float32).unsqueeze(1)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xtr, Ytr),
                                         batch_size=512, shuffle=True)
    Xte = torch.tensor(Xn[te], dtype=torch.float32).to(dev); ref = Ne[te]

    def metrics():
        net.eval()
        with torch.no_grad():
            p = net(Xte).cpu().numpy().squeeze() * stdY + meanY
        net.train()
        re = np.abs(p - ref) / ref
        return np.mean(re)*100, np.median(re)*100, np.sqrt(np.mean((p-ref)**2))/1e5

    print("epoch  mean_relerr%  median%  RMSE(x1e5)")
    for ep in range(args.epochs):
        for bx, by in loader:
            l = lossf(net(bx.to(dev)), by.to(dev)); opt.zero_grad(); l.backward(); opt.step()
        if (ep+1) % 5 == 0 or ep == 0:
            me, md, rm = metrics(); print(f"{ep+1:5d}  {me:11.2f}  {md:7.2f}  {rm:8.3f}")
    me, md, rm = metrics()
    # save model, norm stats, and test arrays (for the manuscript-style figures)
    net.eval()
    with torch.no_grad():
        pred_te = net(Xte).cpu().numpy().squeeze() * stdY + meanY
    torch.save(net.state_dict(), os.path.join(args.out, "ne15.pt"))
    np.savez(os.path.join(args.out, "test_arrays.npz"),
             X=X[te].astype(np.float32), pred=pred_te.astype(np.float32),
             ref=Ne[te].astype(np.float32), features=np.array(FEAT15, dtype=object))
    import scipy.io as _sio
    _sio.savemat(os.path.join(args.out, "norm.mat"),
                 {"meanX": meanX, "stdX": stdX, "meanY": meanY, "stdY": stdY})
    json.dump({"mean_relerr_pct": float(me), "median_relerr_pct": float(md),
               "rmse_x1e5": float(rm), "n_train": int(tr.sum()), "n_test": int(te.sum()),
               "features": FEAT15}, open(os.path.join(args.out, "metrics.json"), "w"), indent=2)
    print(f"\nFINAL 15-feature model: mean rel-err={me:.2f}%  median={md:.2f}%  "
          f"RMSE={rm:.3f}x1e5  (paper full model ~2%, 0.43x1e5)  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
