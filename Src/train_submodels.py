#!/usr/bin/env python3
"""NmF2 and hmF2 sub-models: predict the F2 peak from geophysics alone.

These are the "sub-models" used by the main topside-Ne model when a measured F2
peak is unavailable (the "sub-models" mode, and GRACE/ISR). Unlike
train_nmf2_submodel.py (which used the prepared HRO_iono_height0.mat, whose
local-time feature carries ~no information — see diagnose_nmf2.py), this rebuilds
the training table DIRECTLY from the COSMIC-1 ionPrf profiles so the features are
clean: NmF2=edmax, hmF2=edmaxalt, lat/lon from the peak, local time from the
profile UT, plus OMNI2 drivers.

One row per profile. Features (geophysics, NO altitude — the peak is a property
of a profile, not of an altitude point):
  [Lat, Lon, LT, sinLT, cosLT, DoY, sinDoY, cosDoY, F10.7, PF10.7, Kp, DST, AE, ap]
Targets: log(NmF2) and hmF2 (trained jointly, standardised).

Honest out-of-sample split: test = years 2009 (low solar) & 2013 (high solar).
Reports median relative error to compare with the paper (NmF2 22.5% vs IRI 33.5%;
hmF2 5.8% vs IRI 10.3%).

Usage:
  python Src/train_submodels.py --list <cosmic_list_local.txt> --n-profiles 200000 \
      --device cuda --out Projects/submodels
"""
from __future__ import annotations
import argparse, json, os, sys, time
import datetime as dt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_delay_files import load_omni, omni_at      # OMNI2 (DST,AE,ap,F107,Kp,Vf,Bx,By,Bz)

FEATS = ["Lat", "Lon", "LT", "sinLT", "cosLT", "DoY", "sinDoY", "cosDoY",
         "F107", "PF107", "Kp", "DST", "AE", "ap"]


def build(paths, omni, progress=True):
    import netCDF4 as nc
    rows, nmf2, hmf2, yr = [], [], [], []
    for k, p in enumerate(paths):
        if progress and k % 5000 == 0:
            print(f"  read {k}/{len(paths)}", end="\r", file=sys.stderr)
        try:
            f = nc.Dataset(p.strip())
            nm = float(f.getncattr("edmax")); hm = float(f.getncattr("edmaxalt"))
            lat = float(f.getncattr("edmaxlat")); lon = float(f.getncattr("edmaxlon"))
            year = int(f.getncattr("year")); mon = int(f.getncattr("month")); day = int(f.getncattr("day"))
            ut = f.getncattr("hour") + f.getncattr("minute")/60 + f.getncattr("second")/3600
            f.close()
        except Exception:
            continue
        rt = dt.datetime(year, mon, day, int(ut))
        drv = omni_at(omni, rt)                  # [DST,AE,ap,F107,Kp,Vf,Bx,By,Bz]
        pf = omni_at(omni, rt - dt.timedelta(days=1))  # previous-day F10.7 proxy
        if not np.isfinite(drv[3]) or not np.isfinite(pf[3]):
            continue
        lt = (ut + lon/15.0) % 24.0
        doy = rt.timetuple().tm_yday
        rows.append([lat, lon, lt,
                     np.sin(2*np.pi*lt/24), np.cos(2*np.pi*lt/24),
                     doy, np.sin(2*np.pi*doy/365.25), np.cos(2*np.pi*doy/365.25),
                     drv[3], pf[3], drv[4], drv[0], drv[1], drv[2]])
        nmf2.append(nm); hmf2.append(hm); yr.append(year)
    return (np.asarray(rows), np.asarray(nmf2), np.asarray(hmf2), np.asarray(yr))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", default="/glade/derecho/scratch/{}/cosmic/cosmic_list_local.txt".format(os.environ.get("USER","andonghu")))
    ap.add_argument("--n-profiles", type=int, default=200000)
    ap.add_argument("--omni-cache", default="Data/omni2_cache")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="Projects/submodels")
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed); torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True); t0 = time.time()

    import linecache as lc
    paths = [l for l in lc.getlines(args.list) if l.strip()]
    paths = [paths[i] for i in rng.choice(len(paths), min(args.n_profiles, len(paths)), replace=False)]
    print(f"profiles: {len(paths)}; loading OMNI2 2006-2015 ...")
    omni = load_omni(range(2006, 2016), cache_dir=args.omni_cache, source="omni2")

    X, NmF2, hmF2, yr = build(paths, omni)
    print(f"\nbuilt {X.shape[0]} profiles, {X.shape[1]} features")
    m = ((NmF2 > np.exp(11)) & (NmF2 < np.exp(14.5)) & (hmF2 > 150) & (hmF2 < 600) &
         (X[:, 8] > 60) & (X[:, 8] < 250) & np.all(np.isfinite(X), 1))
    X, NmF2, hmF2, yr = X[m], NmF2[m], hmF2[m], yr[m]
    print(f"after filter: {X.shape[0]}")

    te = np.isin(yr.astype(int), [2009, 2013]); tr = ~te
    print(f"train={tr.sum()} test={te.sum()} (test yrs 2009 & 2013)")

    # targets: log NmF2, hmF2 (standardised on train)
    Y = np.column_stack([np.log(NmF2), hmF2])
    mx, sx = X[tr].mean(0), X[tr].std(0) + 1e-9
    my, sy = Y[tr].mean(0), Y[tr].std(0) + 1e-9
    Xn = (X - mx)/sx; Yn = (Y - my)/sy
    dev = torch.device(args.device)
    net = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], 64), torch.nn.ReLU(),
        torch.nn.Linear(64, 32), torch.nn.ReLU(),
        torch.nn.Linear(32, 16), torch.nn.ReLU(),
        torch.nn.Linear(16, 2)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    lossf = torch.nn.MSELoss()
    Xtr = torch.tensor(Xn[tr], dtype=torch.float32); Ytr = torch.tensor(Yn[tr], dtype=torch.float32)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(Xtr, Ytr),
                                         batch_size=1024, shuffle=True)
    Xte = torch.tensor(Xn[te], dtype=torch.float32).to(dev)

    def metrics():
        net.eval()
        with torch.no_grad():
            pn = net(Xte).cpu().numpy()
        net.train()
        p = pn*sy + my
        nm_pred = np.exp(p[:, 0]); hm_pred = p[:, 1]
        nm_re = np.abs(nm_pred - NmF2[te]) / NmF2[te]
        hm_re = np.abs(hm_pred - hmF2[te]) / hmF2[te]
        return np.median(nm_re)*100, np.median(hm_re)*100

    print("epoch  NmF2_med%  hmF2_med%")
    for ep in range(args.epochs):
        for bx, by in loader:
            l = lossf(net(bx.to(dev)), by.to(dev)); opt.zero_grad(); l.backward(); opt.step()
        if (ep+1) % 10 == 0 or ep == 0:
            nm, hm = metrics(); print(f"{ep+1:5d}  {nm:8.2f}  {hm:8.2f}")
    nm, hm = metrics()
    torch.save(net.state_dict(), os.path.join(args.out, "submodels.pt"))
    json.dump({"nmf2_median_relerr_pct": float(nm), "hmf2_median_relerr_pct": float(hm),
               "paper_nmf2": 22.5, "paper_nmf2_iri": 33.5, "paper_hmf2": 5.8, "paper_hmf2_iri": 10.3,
               "n_train": int(tr.sum()), "n_test": int(te.sum()), "features": FEATS},
              open(os.path.join(args.out, "metrics.json"), "w"), indent=2)
    print(f"\nFINAL  NmF2 median rel-err={nm:.2f}% (paper 22.5%, IRI 33.5%)")
    print(f"       hmF2 median rel-err={hm:.2f}% (paper 5.8%, IRI 10.3%)   [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
