# Regenerate Data/Delay/all_<lag>.mat from the COSMIC ionPrf profiles.
# Port of NmF2-MI.ipynb readRO/Read_RO_delay: per profile derive the F2-peak
# quantities and sample the 9 OMNI drivers at RO_time + lag_hours.
#
# out columns (0-based): 0 Alt_F2(edmaxalt), 1 mLat, 2 mLon, 3 mLT (apex),
#   4 VTEC0(tec0), 5:14 = DST,AE,ap,F10.7,Kp,FlowSpeed,Bx,By,Bz, 14 doy(=month+day/35-1),
#   15 VTEC1(tec1), 16 NmF2(edmax), 17 year.
#
# OMNI comes from NASA SPDF omni2_<year>.dat (the original used aidapy, which now
# depends on the dead heliopy; --omni-source aidapy is a legacy fallback).
# The MI driver min-max normalises each driver, so OMNI unit/scale differences
# (e.g. Kp*10) don't change the result.
# mlon2mlt is called with geographic lon, as in the original (--fix-mlt for magnetic).
#
#   python Src/make_delay_files.py --check       # deps + data readiness
#   python Src/make_delay_files.py --selftest    # column logic, no data
#   python Src/make_delay_files.py --cosmic-list cosmic_list.txt --data-dir Data/Delay \
#       --lags=-8:7 --workers 16

import argparse
import datetime as dt
import os
import sys
import numpy as np

FEATURES = ["DST Index", "AE Index", "ap index", "f10.7 index", "Kp",
            "Plasma Flow Speed", "Bx GSE, GSM", "By GSM", "Bz GSM"]
N_FEATURES = len(FEATURES)
N_COLS = 5 + N_FEATURES + 4   # 18

# OMNI2 word positions (1-indexed) and fill values, in FEATURES order.
# https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2.text
OMNI2_WORDS = [41, 42, 50, 51, 39, 25, 13, 16, 17]
OMNI2_FILLS = [99999, 9999, 999, 999.9, 99, 9999.0, 999.9, 999.9, 999.9]
OMNI2_URL = "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_{year}.dat"


# --- pure helpers ---------------------------------------------------------
def assemble_row(alt_f2, mlat, mlon, mlt, vtec0, x_omni, doy, vtec1, nmf2, year):
    x_omni = np.asarray(x_omni, dtype=float).ravel()
    if x_omni.size != N_FEATURES:
        raise ValueError("x_omni must have %d values, got %d" % (N_FEATURES, x_omni.size))
    row = np.empty(N_COLS, dtype=float)
    row[0] = alt_f2; row[1] = mlat; row[2] = mlon; row[3] = mlt; row[4] = vtec0
    row[5:14] = x_omni
    row[14] = doy; row[15] = vtec1; row[16] = nmf2; row[17] = year
    return row


def keep_row(row):
    # acceptance filter: 0 < VTEC0 < 50, mLat finite and != 0
    return (row[4] > 0) and (row[4] < 50) and (not np.isnan(row[1])) and (row[1] != 0)


def doy_encoding(month, day):
    return month + day / 35.0 - 1.0   # the original's lossy month.day proxy


# --- OMNI -----------------------------------------------------------------
def _omni2_parse_line(parts):
    year = int(parts[0]); doy = int(parts[1]); hour = int(parts[2])
    vals = np.empty(N_FEATURES)
    for i, (w, fill) in enumerate(zip(OMNI2_WORDS, OMNI2_FILLS)):
        raw = float(parts[w - 1])
        vals[i] = np.nan if abs(raw) >= 0.99 * fill else raw   # near-fill -> missing
    return year, doy, hour, vals


def load_omni(years, cache_dir="Data/omni2_cache", source="omni2", start=None, end=None):
    # returns dict {(year, doy, hour): 9-feature array}
    if source == "aidapy":
        return _load_omni_aidapy(start, end)
    import urllib.request
    os.makedirs(cache_dir, exist_ok=True)
    index = {}
    for year in years:
        path = os.path.join(cache_dir, "omni2_%d.dat" % year)
        if not os.path.isfile(path):
            url = OMNI2_URL.format(year=year)
            try:
                urllib.request.urlretrieve(url, path)
            except Exception as e:
                print("  [omni2] could not fetch %s: %s" % (url, e), file=sys.stderr)
                continue
        with open(path) as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < max(OMNI2_WORDS):
                    continue
                y, d, h, vals = _omni2_parse_line(parts)
                index[(y, d, h)] = vals
    return index


def _load_omni_aidapy(start, end):   # legacy, needs heliopy<1
    from aidapy import load_data
    import aidapy.aidaxr  # noqa: F401
    xr_omni = load_data(mission="omni", start_time=start, end_time=end)
    all1 = xr_omni["all1"]
    times = np.asarray(xr_omni["time1"].values)
    index = {}
    for j, t in enumerate(times):
        ts = (t - np.datetime64("1970-01-01T00:00:00Z")) / np.timedelta64(1, "s")
        a = dt.datetime.utcfromtimestamp(float(ts))
        vals = np.asarray(all1[j].sel(products=FEATURES).values, float).ravel()
        index[(a.year, a.timetuple().tm_yday, a.hour)] = vals
    return index


def omni_at(omni_index, when):
    key = (when.year, when.timetuple().tm_yday, when.hour)
    vals = omni_index.get(key)
    return np.full(N_FEATURES, np.nan) if vals is None else vals


# --- RO profile reading ---------------------------------------------------
# apexpy's Fortran backend has a single global epoch, so we never cache Apex
# objects across days; read_profiles processes profiles grouped by day.
def _get_apex(when):
    from apexpy import Apex
    return Apex(date=when)


def read_ro_profile(path, delay_hour, omni_index, fix_mlt=False):
    import netCDF4 as nc
    f = nc.Dataset(path.strip())
    try:
        vtec0 = float(f.getncattr("tec0")); vtec1 = float(f.getncattr("tec1"))
        year = int(f.getncattr("year")); month = int(f.getncattr("month"))
        day = int(f.getncattr("day")); nmf2 = float(f.getncattr("edmax"))
        lat_f2 = float(f.getncattr("edmaxlat")); lon_f2 = float(f.getncattr("edmaxlon"))
        alt_f2 = float(f.getncattr("edmaxalt"))
        ut = (f.getncattr("hour") + f.getncattr("minute") / 60.0
              + f.getncattr("second") / 3600.0)
    finally:
        f.close()
    ro_time = dt.datetime(year, month, day, int(ut))
    x_omni = omni_at(omni_index, ro_time + dt.timedelta(hours=delay_hour))
    apex = _get_apex(ro_time)
    mlat, mlon = apex.convert(lat_f2, lon_f2, "geo", "apex", height=alt_f2)
    mlt = apex.mlon2mlt(mlon if fix_mlt else lon_f2, ro_time)
    return assemble_row(alt_f2, mlat, mlon, mlt, vtec0, x_omni,
                        doy_encoding(month, day), vtec1, nmf2, year)


# Read every profile once (RO-derived fields are lag-independent; only the OMNI
# columns vary per lag). Apex coords computed grouped by day.
#   apex_epoch='day' (fast, ~3e-4 deg) or 'profile' (notebook-exact, slower)
def read_profiles(paths, fix_mlt=False, apex_epoch="day", progress=True):
    import netCDF4 as nc
    from collections import defaultdict

    meta = []   # (alt, lat, lon, vtec0, vtec1, nmf2, year, month, day, ro_time)
    n = len(paths)
    for k, p in enumerate(paths):
        if progress and (k % 5000 == 0):
            print("    read %d/%d" % (k, n), end="\r", file=sys.stderr)
        p = p.strip()
        try:
            f = nc.Dataset(p)
            try:
                vtec0 = float(f.getncattr("tec0")); vtec1 = float(f.getncattr("tec1"))
                year = int(f.getncattr("year")); month = int(f.getncattr("month"))
                day = int(f.getncattr("day")); nmf2 = float(f.getncattr("edmax"))
                lat = float(f.getncattr("edmaxlat")); lon = float(f.getncattr("edmaxlon"))
                alt = float(f.getncattr("edmaxalt"))
                ut = (f.getncattr("hour") + f.getncattr("minute") / 60.0
                      + f.getncattr("second") / 3600.0)
            finally:
                f.close()
            ro_time = dt.datetime(year, month, day, int(ut))
        except Exception as e:
            if progress and k < 5:
                print("    skip %s: %s" % (p, e), file=sys.stderr)
            continue
        meta.append((alt, lat, lon, vtec0, vtec1, nmf2, year, month, day, ro_time))

    base = np.full((len(meta), N_COLS), np.nan)
    ro_times = [None] * len(meta)
    nan9 = np.full(N_FEATURES, np.nan)
    groups = defaultdict(list)
    for i, r in enumerate(meta):
        groups[(r[6], r[7], r[8])].append(i)

    for idxs in groups.values():
        if apex_epoch == "day":
            apex = _get_apex(meta[idxs[0]][9])   # fresh, used contiguously
            lats = np.array([meta[i][1] for i in idxs])
            lons = np.array([meta[i][2] for i in idxs])
            alts = np.array([meta[i][0] for i in idxs])
            mlat, mlon = apex.convert(lats, lons, "geo", "apex", height=alts)
            for j, i in enumerate(idxs):
                r = meta[i]
                mlt = apex.mlon2mlt(mlon[j] if fix_mlt else r[2], r[9])
                base[i] = assemble_row(r[0], mlat[j], mlon[j], mlt, r[3], nan9,
                                       doy_encoding(r[7], r[8]), r[4], r[5], r[6])
                ro_times[i] = r[9]
        else:
            for i in idxs:
                r = meta[i]
                apex = _get_apex(r[9])
                mlat, mlon = apex.convert(r[1], r[2], "geo", "apex", height=r[0])
                mlt = apex.mlon2mlt(mlon if fix_mlt else r[2], r[9])
                base[i] = assemble_row(r[0], mlat, mlon, mlt, r[3], nan9,
                                       doy_encoding(r[7], r[8]), r[4], r[5], r[6])
                ro_times[i] = r[9]

    keep = np.array([keep_row(base[i]) for i in range(len(meta))], dtype=bool)
    return base[keep], [ro_times[i] for i in range(len(meta)) if keep[i]]


def _read_chunk_worker(args):
    chunk, fix_mlt, apex_epoch = args
    return read_profiles(chunk, fix_mlt=fix_mlt, apex_epoch=apex_epoch, progress=False)


# parallel read: split paths across processes (each process has its own apexpy
# global state). Row order is not preserved, which doesn't matter here.
def read_profiles_parallel(paths, fix_mlt=False, apex_epoch="day", workers=1):
    if workers <= 1 or len(paths) < 2 * workers:
        return read_profiles(paths, fix_mlt=fix_mlt, apex_epoch=apex_epoch)
    from multiprocessing import Pool
    # contiguous chunks (list is sorted by year/doy) -> few Apex inits per worker
    q, r = divmod(len(paths), workers)
    chunks, start = [], 0
    for w in range(workers):
        size = q + (1 if w < r else 0)
        chunks.append(paths[start:start + size])
        start += size
    args = [(c, fix_mlt, apex_epoch) for c in chunks]
    bases, times = [], []
    with Pool(workers) as pool:
        for b, t in pool.map(_read_chunk_worker, args):
            if b.size:
                bases.append(b)
                times.extend(t)
    base = np.vstack(bases) if bases else np.empty((0, N_COLS))
    return base, times


def fill_lag(base, ro_times, omni_index, delay):
    out = base.copy()
    td = dt.timedelta(hours=delay)
    for i, t in enumerate(ro_times):
        out[i, 5:14] = omni_at(omni_index, t + td)
    return out


# serial per-lag generation (re-reads profiles each lag); cmd_run uses the
# read-once path instead.
def generate_lag(delay, paths, omni_index, fix_mlt=False, workers=1, progress=True):
    rows = []
    n = len(paths)
    for k, p in enumerate(paths):
        if progress and (k % 5000 == 0):
            print("    [lag %+d] %d/%d" % (delay, k, n), end="\r", file=sys.stderr)
        try:
            row = read_ro_profile(p, delay, omni_index, fix_mlt=fix_mlt)
        except Exception as e:
            if progress and k < 5:
                print("    [lag %+d] skip %s: %s" % (delay, p, e), file=sys.stderr)
            continue
        if keep_row(row):
            rows.append(row)
    out = np.asarray(rows, dtype=float) if rows else np.empty((0, N_COLS))
    assert out.ndim == 2 and out.shape[1] == N_COLS, out.shape
    return out


# --- readiness check / self-test / main -----------------------------------
def cmd_check(args):
    print("=== make_delay_files readiness check ===")
    ok = True
    for mod in ("numpy", "scipy", "netCDF4", "apexpy"):
        try:
            __import__(mod)
            print("  [ok]   import %s" % mod)
        except Exception as e:
            ok = False
            print("  [MISS] import %s: %s" % (mod, e))

    if args.omni_source == "omni2":
        try:
            import urllib.request
            req = urllib.request.Request(OMNI2_URL.format(year=2010), method="HEAD")
            urllib.request.urlopen(req, timeout=30)
            print("  [ok]   OMNI2 SPDF reachable")
        except Exception as e:
            print("  [warn] OMNI2 SPDF HEAD failed (may still GET): %s" % e)
    else:
        print("  [note] omni-source=aidapy is legacy and needs heliopy<1")

    if os.path.isfile(args.cosmic_list):
        import linecache as lc
        lines = [l for l in lc.getlines(args.cosmic_list) if l.strip()]
        print("  [ok]   cosmic_list: %s (%d paths)" % (args.cosmic_list, len(lines)))
        n_exist = sum(os.path.isfile(p.strip()) for p in lines[:20])
        if not n_exist:
            ok = False
        print("  [%s] %d/20 sampled ionPrf paths exist on disk"
              % ("ok" if n_exist else "MISS", n_exist))
        if not n_exist and lines:
            print("         e.g. %s" % lines[0].strip())
    else:
        ok = False
        print("  [MISS] cosmic_list not found: %s" % args.cosmic_list)

    print("  output dir: %s (%s)"
          % (args.data_dir, "exists" if os.path.isdir(args.data_dir) else "will be created"))
    print("\nREADY" if ok else "\nNOT READY -- resolve the [MISS] items above.")
    return 0 if ok else 2


def cmd_selftest(args):
    print("=== make_delay_files self-test (no external data) ===")
    fails = []

    def chk(name, cond, detail=""):
        print("  [%s] %s%s" % ("PASS" if cond else "FAIL", name,
                                ("  (%s)" % detail) if detail else ""))
        if not cond:
            fails.append(name)

    x = np.arange(10, 19, dtype=float)
    row = assemble_row(300.0, 65.0, 120.0, 11.5, 12.0, x, 5.4, 3.0, 5.5e5, 2010)
    chk("row length == 18", row.shape == (N_COLS,), str(row.shape))
    chk("Alt at col 0", row[0] == 300.0)
    chk("mLat at col 1", row[1] == 65.0)
    chk("VTEC0 at col 4", row[4] == 12.0)
    chk("9 drivers at cols 5:14", np.array_equal(row[5:14], x))
    chk("doy at col 14", row[14] == 5.4)
    chk("NmF2 at col 16", row[16] == 5.5e5)
    chk("year at col 17", row[17] == 2010)
    chk("doy_encoding(7,1)", abs(doy_encoding(7, 1) - (7 + 1 / 35 - 1)) < 1e-12)

    chk("keep_row accepts valid", keep_row(assemble_row(300, 65, 0, 0, 10.0, x, 0, 0, 1, 2010)))
    chk("keep_row rejects VTEC0>=50", not keep_row(assemble_row(300, 65, 0, 0, 80.0, x, 0, 0, 1, 2010)))
    chk("keep_row rejects mLat==0", not keep_row(assemble_row(300, 0.0, 0, 0, 10.0, x, 0, 0, 1, 2010)))

    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import mi_regressor as mr
        out = np.tile(row, (8, 1))
        mf, mm, ms, nk = mr.compute_mi_for_lag(out, num_resample=2, mlat_min=60.0)
        chk("mi_regressor consumes the schema", mf.shape == (9,), "n_kept=%d" % nk)
    except Exception as e:
        chk("mi_regressor consumes the schema", False, repr(e)[:120])

    if fails:
        print("\nRESULT: %d FAILURE(S): %s" % (len(fails), fails))
        return 1
    print("\nRESULT: ALL SELF-TESTS PASSED")
    return 0


def cmd_run(args):
    import linecache as lc
    start, stop = (int(v) for v in args.lags.split(":"))
    lags = list(range(start, stop + 1))
    paths = [l for l in lc.getlines(args.cosmic_list) if l.strip()]
    if args.n_profiles:
        paths = paths[:args.n_profiles]
    print("Lags: %s  |  profiles: %d" % (lags, len(paths)))

    years = range(args.omni_year_start, args.omni_year_end + 1)
    print("Loading OMNI (%s) for years %d-%d..."
          % (args.omni_source, args.omni_year_start, args.omni_year_end))
    omni_index = load_omni(years, cache_dir=args.omni_cache, source=args.omni_source)
    print("OMNI hourly records indexed: %d" % len(omni_index))
    if not omni_index:
        print("ERROR: no OMNI data loaded; cannot fill driver columns.", file=sys.stderr)
        return 3

    print("Reading %d profiles once (workers=%d)..." % (len(paths), args.workers))
    base, ro_times = read_profiles_parallel(paths, fix_mlt=args.fix_mlt,
                                            apex_epoch=args.apex_epoch, workers=args.workers)
    print("Retained %d profiles after acceptance filter." % base.shape[0])

    os.makedirs(args.data_dir, exist_ok=True)
    from scipy import io as sio
    for lag in lags:
        out = fill_lag(base, ro_times, omni_index, lag)
        dest = os.path.join(args.data_dir, "%s%d.mat" % (args.out_prefix, lag))
        sio.savemat(dest, {"out": out})
        print("  wrote %s  shape=%s" % (dest, out.shape))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="regenerate Data/Delay/all_<lag>.mat")
    ap.add_argument("--cosmic-list", default="cosmic_list.txt")
    ap.add_argument("--data-dir", default="Data/Delay")
    ap.add_argument("--lags", default="-8:7", help="lag range start:stop (pass as --lags=-8:7)")
    ap.add_argument("--out-prefix", default="all_")
    ap.add_argument("--n-profiles", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--fix-mlt", action="store_true")
    ap.add_argument("--apex-epoch", choices=["day", "profile"], default="day")
    ap.add_argument("--omni-source", choices=["omni2", "aidapy"], default="omni2")
    ap.add_argument("--omni-cache", default="Data/omni2_cache")
    ap.add_argument("--omni-year-start", type=int, default=2006)
    ap.add_argument("--omni-year-end", type=int, default=2018)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        return cmd_selftest(args)
    if args.check:
        return cmd_check(args)
    return cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
