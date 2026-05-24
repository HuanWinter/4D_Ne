#!/usr/bin/env python3
"""Diagnose & sanity-check the NmF2 sub-model reproduction gap.

Our faithful reconstructions of the paper's NmF2 sub-model (global ANN and the
regional KISS-GP deep-kernel model) both land at ~44% MEDIAN relative error,
~2x the paper's reported 22.5% and worse than IRI's 33.5%. A data-driven model
trained on COSMIC NmF2 should beat IRI on COSMIC, so the gap most likely lives
in the data/evaluation pipeline, not the model. This script tests that
systematically and prints a verdict.

Checks
------
1. DATA SANITY      shapes, NaNs, per-feature ranges + inferred identity, year
                    histogram, train/test sizes, duplicate rows, NmF2 units.
2. CONVENTION SWEEP how much the *error definition* moves the number: target
                    {log,linear} x denominator {ref,pred} x {mean,median}, using
                    a per-(lat,LT)-region median predictor (a no-skill-but-local
                    baseline) and a global-median predictor.
3. ACHIEVABLE CEIL  a strong, hyperparameter-light model (HistGradientBoosting)
                    on the 10 features with the same year split -> the realistic
                    best these features+protocol can give. If this is >> 22.5%,
                    the features/eval cannot reach the paper's number (so the
                    paper used something we don't have); if it's ~22%, our
                    NN/GP training was just weak.
4. FEATURE ASSOC    correlation + mutual information of each feature vs log(NmF2).
5. IRI BASELINE     (optional, --iri) PyIRI NmF2 at a sample of test points;
                    rel-err should land near the paper's 33.5% if our test set /
                    units / eval match the paper. PyIRI ~= IRI-2020 (not 2016).

Usage
-----
  python Src/diagnose_nmf2.py                 # core checks (CPU, ~1-2 min)
  python Src/diagnose_nmf2.py --iri --iri-n 2000   # also run PyIRI baseline
"""
from __future__ import annotations
import argparse, sys, time
import numpy as np
import scipy.io as sio

TEST_YEARS = [2009, 2013]
# HRO_iono_height0.mat X columns (inferred): see ranges printed in section 1.
COL = dict(mLat=0, mLon=1, LT=2, drv1=3, drv2=4, drv3=5, drv4=6, DoY=7, F107=8, PF107=9)


def hr(t):
    print("\n" + "=" * 78 + f"\n {t}\n" + "=" * 78)


def relerr(pred, ref, denom="ref"):
    d = ref if denom == "ref" else pred
    with np.errstate(divide="ignore", invalid="ignore"):
        e = np.abs(pred - ref) / np.abs(d)
    return e[np.isfinite(e)]


def load(path):
    d = sio.loadmat(path)
    X = np.asarray(d["X"], float)
    Y = np.asarray(d["Y"], float).ravel()
    Ref = np.asarray(d["Ref"], float)
    return X, Y, Ref


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="Data/Delay/HRO_iono_height0.mat")
    ap.add_argument("--iri", action="store_true", help="run PyIRI baseline (slow)")
    ap.add_argument("--iri-n", type=int, default=1500, help="PyIRI sample size")
    ap.add_argument("--gbm-n", type=int, default=300000, help="subsample for GBM/MI")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    Xall, Yall, Refall = load(args.data)

    # ---- 1. DATA SANITY -----------------------------------------------------
    hr("1. DATA SANITY")
    print(f"X {Xall.shape}  Y {Yall.shape}  Ref {Refall.shape}")
    print(f"NaN: X={np.isnan(Xall).sum()}  Y={np.isnan(Yall).sum()}  Ref={np.isnan(Refall).sum()}")
    names = list(COL)
    print("X feature ranges (inferred identity):")
    for nm, c in COL.items():
        col = Xall[:, c][np.isfinite(Xall[:, c])]
        print(f"  col{c:2d} {nm:6s}: [{col.min():9.3g}, {col.max():9.3g}]  mean={col.mean():.3g}")
    print(f"Y (NmF2): [{Yall.min():.3g}, {Yall.max():.3g}]  "
          f"-- exp(11)={np.exp(11):.0f}, exp(14.5)={np.exp(14.5):.0f}")
    print("  units check: topside F2-peak NmF2 ~1e5-1e6 el/cm^3 (not el/m^3=1e11-1e12)")
    print(f"Ref cols [year,month,day,LT,lat,lon] ranges:")
    for c in range(Refall.shape[1]):
        col = Refall[:, c][np.isfinite(Refall[:, c])]
        print(f"  Ref{c}: [{col.min():.4g}, {col.max():.4g}]")
    yrs, cnts = np.unique(Refall[:, 0].astype(int), return_counts=True)
    print("year histogram:", dict(zip(yrs.tolist(), cnts.tolist())))

    # filter (same as training) + split
    F107, PF107 = Xall[:, 8], Xall[:, 9]
    keep = ((Yall > np.exp(11)) & (Yall < np.exp(14.5)) & (F107 > 50) & (F107 < 200) &
            (PF107 > 50) & (PF107 < 200) & np.all(np.isfinite(Xall), 1) & np.isfinite(Yall))
    X, Y, year = Xall[keep], Yall[keep], Refall[keep, 0]
    tr = ~np.isin(year.astype(int), TEST_YEARS)
    te = ~tr
    print(f"\nafter filter: {X.shape[0]} ({keep.mean()*100:.1f}% kept)")
    print(f"train={tr.sum()} (years not in {TEST_YEARS})  test={te.sum()} (years {TEST_YEARS})")
    # leakage: any test year present in train? duplicate rows across split?
    print("leakage check: test years in train?",
          bool(set(year[te].astype(int)) & set(year[tr].astype(int))))
    # duplicate feature rows shared train/test (sampled for speed)
    s = rng.choice(np.where(tr)[0], min(50000, tr.sum()), replace=False)
    tr_keys = set(map(tuple, np.round(X[s, :3], 3)))
    s2 = rng.choice(np.where(te)[0], min(50000, te.sum()), replace=False)
    dup = np.mean([tuple(r) in tr_keys for r in np.round(X[s2, :3], 3)])
    print(f"approx fraction of test (lat,lon,LT)-keys also in train sample: {dup*100:.2f}%")

    logY = np.log(Y)

    # ---- 2. CONVENTION SWEEP -----------------------------------------------
    hr("2. ERROR-CONVENTION SWEEP (per-region median predictor & global median)")
    # region bins matching the GP script (normalised lat/LT into 6x2)
    lo = X.min(0); span = np.where(X.max(0) - lo == 0, 1, X.max(0) - lo)
    Xn = (X - lo) / span
    lat_b = np.clip((Xn[:, COL["mLat"]] * 6).astype(int), 0, 5)
    lt_b = np.clip((Xn[:, COL["LT"]] * 2).astype(int), 0, 1)
    reg = lat_b * 2 + lt_b

    def region_median_pred(in_log):
        src = logY if in_log else Y
        pred = np.empty(te.sum())
        refidx = np.where(te)[0]
        tmed = {}
        for r in np.unique(reg[tr]):
            tmed[r] = np.median(src[tr & (reg == r)])
        gm = np.median(src[tr])
        for k, i in enumerate(refidx):
            pred[k] = tmed.get(reg[i], gm)
        return np.exp(pred) if in_log else pred

    Yte = Y[te]
    print(f"{'predictor':28s} {'target':7s} {'denom':5s} {'mean%':>7s} {'median%':>8s}")
    for in_log in (True, False):
        p = region_median_pred(in_log)
        for den in ("ref", "pred"):
            e = relerr(p, Yte, den)
            print(f"{'per-region median':28s} {('log' if in_log else 'linear'):7s} "
                  f"{den:5s} {e.mean()*100:7.1f} {np.median(e)*100:8.1f}")
    gmed = np.median(Y[tr])
    e = relerr(np.full_like(Yte, gmed), Yte, "ref")
    print(f"{'global median (no-skill)':28s} {'linear':7s} {'ref':5s} "
          f"{e.mean()*100:7.1f} {np.median(e)*100:8.1f}")

    # ---- 3. ACHIEVABLE CEILING (gradient boosting) -------------------------
    hr("3. ACHIEVABLE CEILING -- HistGradientBoosting on the 10 features")
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        itr = np.where(tr)[0]; ite = np.where(te)[0]
        if itr.size > args.gbm_n:
            itr = rng.choice(itr, args.gbm_n, replace=False)
        if ite.size > args.gbm_n:
            ite = rng.choice(ite, args.gbm_n, replace=False)
        gb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1,
                                           max_depth=None, random_state=0)
        gb.fit(X[itr], logY[itr])
        pred = np.exp(gb.predict(X[ite]))
        ref = Y[ite]
        for den in ("ref", "pred"):
            e = relerr(pred, ref, den)
            print(f"  GBM rel-err (/{den}):  mean={e.mean()*100:5.1f}%  median={np.median(e)*100:5.1f}%")
        rmse = np.sqrt(np.mean((pred - ref) ** 2))
        print(f"  GBM RMSE = {rmse:.0f} el/cm^3 ({rmse/1e5:.3f} x1e5)")
        print("  -> this is ~the best these 10 features + year-split + this eval can do.")
    except Exception as e:
        print("  sklearn GBM unavailable:", e)

    # ---- 4. FEATURE ASSOCIATION --------------------------------------------
    hr("4. FEATURE-TARGET ASSOCIATION (vs log NmF2)")
    sidx = rng.choice(X.shape[0], min(200000, X.shape[0]), replace=False)
    try:
        from sklearn.feature_selection import mutual_info_regression
        mi = mutual_info_regression(X[sidx], logY[sidx], random_state=0)
    except Exception:
        mi = [np.nan] * X.shape[1]
    print(f"{'feature':8s} {'corr':>7s} {'MI(nats)':>9s}")
    for nm, c in COL.items():
        cc = np.corrcoef(X[sidx, c], logY[sidx])[0, 1]
        print(f"{nm:8s} {cc:7.3f} {mi[c]:9.3f}")

    # ---- 5. IRI BASELINE (optional) ----------------------------------------
    if args.iri:
        hr("5. PyIRI NmF2 BASELINE (PyIRI ~= IRI-2020, not IRI-2016)")
        try:
            run_iri_baseline(X, Y, Refall[keep], te, rng, args.iri_n)
        except Exception as e:
            import traceback; traceback.print_exc()
            print("  PyIRI baseline failed:", e)
    else:
        hr("5. PyIRI baseline skipped (pass --iri to run; paper reports IRI 33.5%)")

    # ---- VERDICT ------------------------------------------------------------
    hr("VERDICT")
    print("Compare section-3 GBM median to the paper's 22.5%:")
    print("  * GBM median ~= 22.5%  -> features+protocol CAN reach it; our NN/GP")
    print("    training was underpowered (tune/retrain).")
    print("  * GBM median >> 22.5% (e.g. ~40%) -> these features + this year-split +")
    print("    this error definition CANNOT reach 22.5%; the paper used a different")
    print("    target transform / test set / feature set we don't have. Reproduction")
    print("    gap is in the data/eval pipeline, not the model.")
    print(f"\n[{time.time()-t0:.0f}s]")


def run_iri_baseline(X, Y, Ref, te, rng, n):
    """PyIRI NmF2 at a sample of test points vs COSMIC NmF2 (best-effort)."""
    import PyIRI, PyIRI.main_library as ml
    import os
    coeff = os.path.join(os.path.dirname(PyIRI.__file__), "coefficients")
    ite = np.where(te)[0]
    sel = rng.choice(ite, min(n, ite.size), replace=False)
    # Ref cols: [year, month, day, LT/hour, lat, lon]; F107 = X[:,8]
    yr, mo, day = Ref[sel, 0].astype(int), Ref[sel, 1].astype(int), Ref[sel, 2].astype(int)
    ut = Ref[sel, 3].astype(float); lat = Ref[sel, 4]; lon = Ref[sel, 5]
    f107 = X[sel, 8]
    nm_iri = np.full(sel.size, np.nan)
    # group by (year,month,day) and call PyIRI per day on that day's points
    from collections import defaultdict
    groups = defaultdict(list)
    for k in range(sel.size):
        groups[(yr[k], mo[k], day[k])].append(k)
    done = 0
    for (Yr, Mo, Dy), ks in groups.items():
        ks = np.array(ks)
        aUT = np.unique(np.clip(ut[ks], 0, 23.999))
        alon = lon[ks]; alat = lat[ks]
        try:
            f2, f1, epeak, es_peak, sun, mag, ne = ml.IRI_density_1day(
                Yr, Mo, Dy, aUT, alon, alat, np.array([300.0]),
                float(np.median(f107[ks])), coeff, 0)
            # f2['Nm'] shape (nUT, nloc); pick nearest UT per point
            Nm = np.asarray(f2["Nm"])
            for j, kk in enumerate(ks):
                iu = int(np.argmin(np.abs(aUT - min(max(ut[kk], 0), 23.999))))
                nm_iri[kk] = Nm[iu, j]
        except Exception:
            continue
        done += 1
        if done % 50 == 0:
            print(f"  ...{done}/{len(groups)} days")
    ok = np.isfinite(nm_iri) & (nm_iri > 0)
    if ok.sum() < 10:
        print("  too few PyIRI points; skipping"); return
    # PyIRI Nm is in m^-3; COSMIC NmF2 here is el/cm^3 -> convert IRI to cm^-3
    nm_iri_cm3 = nm_iri[ok] / 1e6
    ref = Y[sel][ok]
    e = relerr(nm_iri_cm3, ref, "ref")
    print(f"  PyIRI points used: {ok.sum()}")
    print(f"  IRI rel-err (/ref): mean={e.mean()*100:.1f}%  median={np.median(e)*100:.1f}%"
          f"   (paper IRI-2016: 33.5%)")
    print("  NOTE: if this is far from 33.5%, our test-set/units/eval differ from the paper.")


if __name__ == "__main__":
    main()
