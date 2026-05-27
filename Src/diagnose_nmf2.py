# Diagnose the NmF2 sub-model gap on Data/Delay/HRO_iono_height0.mat.
# Checks: data sanity, error-convention sweep (per-region/global median baselines),
# achievable ceiling (gradient boosting), feature-target association, and an
# optional PyIRI baseline (--iri). The key finding is that local time carries
# ~no information in this prepared table, so no model reaches the paper's 22.5%
# from it -- the gap is in the feature table, not the model.
#   python Src/diagnose_nmf2.py            # core checks
#   python Src/diagnose_nmf2.py --iri      # + PyIRI baseline

import argparse
import sys
import time
import numpy as np
import scipy.io as sio

TEST_YEARS = [2009, 2013]
COL = dict(mLat=0, mLon=1, LT=2, drv1=3, drv2=4, drv3=5, drv4=6, DoY=7, F107=8, PF107=9)


def hr(t):
    print("\n" + "=" * 78 + "\n " + t + "\n" + "=" * 78)


def relerr(pred, ref, denom="ref"):
    d = ref if denom == "ref" else pred
    with np.errstate(divide="ignore", invalid="ignore"):
        e = np.abs(pred - ref) / np.abs(d)
    return e[np.isfinite(e)]


def main(argv=None):
    ap = argparse.ArgumentParser(description="diagnose the NmF2 sub-model gap")
    ap.add_argument("--data", default="Data/Delay/HRO_iono_height0.mat")
    ap.add_argument("--iri", action="store_true", help="run PyIRI baseline (slow)")
    ap.add_argument("--iri-n", type=int, default=1500)
    ap.add_argument("--gbm-n", type=int, default=300000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    d = sio.loadmat(args.data)
    Xall = np.asarray(d["X"], float)
    Yall = np.asarray(d["Y"], float).ravel()
    Refall = np.asarray(d["Ref"], float)

    hr("1. DATA SANITY")
    print("X %s  Y %s  Ref %s" % (Xall.shape, Yall.shape, Refall.shape))
    print("NaN: X=%d Y=%d Ref=%d"
          % (np.isnan(Xall).sum(), np.isnan(Yall).sum(), np.isnan(Refall).sum()))
    print("X feature ranges:")
    for nm, c in COL.items():
        col = Xall[:, c][np.isfinite(Xall[:, c])]
        print("  col%2d %-6s: [%9.3g, %9.3g]  mean=%.3g" % (c, nm, col.min(), col.max(), col.mean()))
    print("Y (NmF2): [%.3g, %.3g]  -- exp(11)=%.0f exp(14.5)=%.0f"
          % (Yall.min(), Yall.max(), np.exp(11), np.exp(14.5)))
    for c in range(Refall.shape[1]):
        col = Refall[:, c][np.isfinite(Refall[:, c])]
        print("  Ref%d: [%.4g, %.4g]" % (c, col.min(), col.max()))
    yrs, cnts = np.unique(Refall[:, 0].astype(int), return_counts=True)
    print("year histogram:", dict(zip(yrs.tolist(), cnts.tolist())))

    F107, PF107 = Xall[:, 8], Xall[:, 9]
    keep = ((Yall > np.exp(11)) & (Yall < np.exp(14.5)) & (F107 > 50) & (F107 < 200) &
            (PF107 > 50) & (PF107 < 200) & np.all(np.isfinite(Xall), 1) & np.isfinite(Yall))
    X, Y, year = Xall[keep], Yall[keep], Refall[keep, 0]
    tr = ~np.isin(year.astype(int), TEST_YEARS)
    te = ~tr
    print("\nafter filter: %d (%.1f%% kept)" % (X.shape[0], keep.mean() * 100))
    print("train=%d test=%d (years %s)" % (tr.sum(), te.sum(), TEST_YEARS))
    print("leakage check: test years in train?",
          bool(set(year[te].astype(int)) & set(year[tr].astype(int))))
    s = rng.choice(np.where(tr)[0], min(50000, tr.sum()), replace=False)
    tr_keys = set(map(tuple, np.round(X[s, :3], 3)))
    s2 = rng.choice(np.where(te)[0], min(50000, te.sum()), replace=False)
    dup = np.mean([tuple(r) in tr_keys for r in np.round(X[s2, :3], 3)])
    print("test (lat,lon,LT) keys also in train sample: %.2f%%" % (dup * 100))

    logY = np.log(Y)

    hr("2. ERROR-CONVENTION SWEEP (per-region & global median baselines)")
    lo = X.min(0)
    span = np.where(X.max(0) - lo == 0, 1, X.max(0) - lo)
    Xn = (X - lo) / span
    lat_b = np.clip((Xn[:, COL["mLat"]] * 6).astype(int), 0, 5)
    lt_b = np.clip((Xn[:, COL["LT"]] * 2).astype(int), 0, 1)
    reg = lat_b * 2 + lt_b

    def region_median_pred(in_log):
        src = logY if in_log else Y
        refidx = np.where(te)[0]
        pred = np.empty(refidx.size)
        tmed = {r: np.median(src[tr & (reg == r)]) for r in np.unique(reg[tr])}
        gm = np.median(src[tr])
        for k, i in enumerate(refidx):
            pred[k] = tmed.get(reg[i], gm)
        return np.exp(pred) if in_log else pred

    Yte = Y[te]
    print("%-22s %-7s %-5s %7s %8s" % ("predictor", "target", "denom", "mean%", "median%"))
    for in_log in (True, False):
        p = region_median_pred(in_log)
        for den in ("ref", "pred"):
            e = relerr(p, Yte, den)
            print("%-22s %-7s %-5s %7.1f %8.1f"
                  % ("per-region median", "log" if in_log else "linear", den,
                     e.mean() * 100, np.median(e) * 100))
    e = relerr(np.full_like(Yte, np.median(Y[tr])), Yte, "ref")
    print("%-22s %-7s %-5s %7.1f %8.1f"
          % ("global median", "linear", "ref", e.mean() * 100, np.median(e) * 100))

    hr("3. ACHIEVABLE CEILING -- HistGradientBoosting on the 10 features")
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        itr = np.where(tr)[0]
        ite = np.where(te)[0]
        if itr.size > args.gbm_n:
            itr = rng.choice(itr, args.gbm_n, replace=False)
        if ite.size > args.gbm_n:
            ite = rng.choice(ite, args.gbm_n, replace=False)
        gb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, random_state=0)
        gb.fit(X[itr], logY[itr])
        pred = np.exp(gb.predict(X[ite]))
        ref = Y[ite]
        for den in ("ref", "pred"):
            e = relerr(pred, ref, den)
            print("  GBM rel-err (/%s):  mean=%5.1f%%  median=%5.1f%%"
                  % (den, e.mean() * 100, np.median(e) * 100))
        rmse = np.sqrt(np.mean((pred - ref) ** 2))
        print("  GBM RMSE = %.0f el/cm^3 (%.3f x1e5)" % (rmse, rmse / 1e5))
        print("  -> ~the best these features + year-split + this eval can do.")
    except Exception as e:
        print("  sklearn GBM unavailable:", e)

    hr("4. FEATURE-TARGET ASSOCIATION (vs log NmF2)")
    sidx = rng.choice(X.shape[0], min(200000, X.shape[0]), replace=False)
    try:
        from sklearn.feature_selection import mutual_info_regression
        mi = mutual_info_regression(X[sidx], logY[sidx], random_state=0)
    except Exception:
        mi = [np.nan] * X.shape[1]
    print("%-8s %7s %9s" % ("feature", "corr", "MI(nats)"))
    for nm, c in COL.items():
        cc = np.corrcoef(X[sidx, c], logY[sidx])[0, 1]
        print("%-8s %7.3f %9.3f" % (nm, cc, mi[c]))

    if args.iri:
        hr("5. PyIRI NmF2 BASELINE (PyIRI ~ IRI-2020, not IRI-2016)")
        try:
            run_iri_baseline(X, Y, Refall[keep], te, rng, args.iri_n)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("  PyIRI baseline failed:", e)
    else:
        hr("5. PyIRI baseline skipped (pass --iri; paper reports IRI 33.5%)")

    hr("VERDICT")
    print("Compare the section-3 GBM median to the paper's 22.5%:")
    print("  GBM median ~ 22.5% -> features+protocol CAN reach it (our NN/GP was weak).")
    print("  GBM median >> 22.5% -> these features/split/eval CANNOT reach it; the gap")
    print("    is in the data/eval pipeline, not the model.")
    print("\n[%.0fs]" % (time.time() - t0))


def run_iri_baseline(X, Y, Ref, te, rng, n):
    import os
    import PyIRI
    import PyIRI.main_library as ml
    from collections import defaultdict
    coeff = os.path.join(os.path.dirname(PyIRI.__file__), "coefficients")
    ite = np.where(te)[0]
    sel = rng.choice(ite, min(n, ite.size), replace=False)
    yr, mo, day = Ref[sel, 0].astype(int), Ref[sel, 1].astype(int), Ref[sel, 2].astype(int)
    ut = Ref[sel, 3].astype(float)
    lat, lon = Ref[sel, 4], Ref[sel, 5]
    f107 = X[sel, 8]
    nm_iri = np.full(sel.size, np.nan)
    groups = defaultdict(list)
    for k in range(sel.size):
        groups[(yr[k], mo[k], day[k])].append(k)
    done = 0
    for (Yr, Mo, Dy), ks in groups.items():
        ks = np.array(ks)
        aUT = np.unique(np.clip(ut[ks], 0, 23.999))
        try:
            f2 = ml.IRI_density_1day(Yr, Mo, Dy, aUT, lon[ks], lat[ks],
                                     np.array([300.0]), float(np.median(f107[ks])), coeff, 0)[0]
            Nm = np.asarray(f2["Nm"])
            for j, kk in enumerate(ks):
                iu = int(np.argmin(np.abs(aUT - min(max(ut[kk], 0), 23.999))))
                nm_iri[kk] = Nm[iu, j]
        except Exception:
            continue
        done += 1
        if done % 50 == 0:
            print("  ...%d/%d days" % (done, len(groups)))
    ok = np.isfinite(nm_iri) & (nm_iri > 0)
    if ok.sum() < 10:
        print("  too few PyIRI points; skipping")
        return
    e = relerr(nm_iri[ok] / 1e6, Y[sel][ok], "ref")   # IRI Nm m^-3 -> cm^-3
    print("  PyIRI points used: %d" % ok.sum())
    print("  IRI rel-err (/ref): mean=%.1f%%  median=%.1f%%  (paper IRI-2016: 33.5%%)"
          % (e.mean() * 100, np.median(e) * 100))


if __name__ == "__main__":
    main()
