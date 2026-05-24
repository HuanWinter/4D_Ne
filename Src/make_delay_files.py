#!/usr/bin/env python3
r"""Regenerate the Data/Delay/all_<lag>.mat datasets used by the MI analysis.

Faithful Python port of the generation pipeline in ``NmF2-MI.ipynb`` (cells 1,
3, 4, 6, 7: ``readRO`` / ``Read_RO_delay`` plus the OMNI setup). For each time
lag it loops over the COSMIC GNSS radio-occultation (RO) ``ionPrf`` netCDF
profiles, derives the F2-peak quantities, samples the 9 OMNI space-weather
drivers at ``RO_time + lag_hours``, and writes one ``out`` array per lag.

Output column layout (0-based) -- this is exactly what mi_regressor.py expects
----------------------------------------------------------------------------
    out[:, 0]      Alt_F2     (edmaxalt)        F2-peak altitude
    out[:, 1]      mLat       (apex)            magnetic latitude
    out[:, 2]      mLon       (apex)            magnetic longitude
    out[:, 3]      mLT        (apex mlon2mlt)   magnetic local time
    out[:, 4]      VTEC0      (tec0)
    out[:, 5:14]   X_omni     9 OMNI drivers in `FEATURES` order:
                              DST, AE, ap, F10.7, Kp, Plasma Flow Speed, Bx, By, Bz
    out[:, 14]     doy        month + day/35 - 1   (note: the original's lossy
                                                    "month.day" encoding -- kept)
    out[:, 15]     VTEC1      (tec1)
    out[:, 16]     NmF2       (edmax)
    out[:, 17]     year

Inputs required (NONE are in this checkout -- see MIGRATION_REPORT.md / --check)
-------------------------------------------------------------------------------
  * cosmic_list.txt : one ionPrf netCDF path per line (the `string` variable).
  * The ionPrf netCDF files themselves (carry NmF2, F2-peak lat/lon/alt, TEC,
    and the exact UT timestamp needed for the lag-shifted OMNI lookup).
  * OMNI hourly drivers. The notebook used the `aidapy` package, but aidapy
    depends on `heliopy`, which is now an unusable tombstone package. This port
    therefore reads the same hourly OMNI data directly from NASA SPDF
    (omni2_YYYY.dat) by default (`--omni-source omni2`); `--omni-source aidapy`
    is kept only as a legacy path and will fail unless heliopy<1 is installed.
  * Packages: netCDF4, apexpy (Apex); numpy, scipy. (aidapy only for legacy.)

Note on OMNI values: the MI driver (mi_regressor.py) min-max normalises each
driver before the Kraskov estimate, which is invariant to linear rescaling, so
any unit/scale differences between the OMNI2 columns and aidapy's products
(e.g. Kp stored as Kp*10) do not affect the mutual-information results.

Behaviour-preserving differences from the notebook (documented, not silent)
---------------------------------------------------------------------------
  * The notebook scans the whole OMNI time axis per profile (O(N_omni) each);
    here OMNI timestamps are hashed into a dict for O(1) exact-hour lookup. The
    selected value is identical.
  * The notebook's ``if abs(Lat_F2) < 0:`` aacgmv2 branch is dead code (the
    condition is never true), so only the Apex branch ever ran -- reproduced.
  * ``mlt = Apex.mlon2mlt(Lon_F2, RO_time)`` passes the *geographic* longitude
    (as in the original). This looks like an upstream bug but is preserved for
    bit-compatibility; pass --fix-mlt to use the magnetic longitude instead.

Usage
-----
    # Check readiness (deps + data) without doing any work:
    python Src/make_delay_files.py --check

    # Validate the column-assembly logic with no external data:
    python Src/make_delay_files.py --selftest

    # Real run (on a host that has the COSMIC data + aidapy):
    python Src/make_delay_files.py \
        --cosmic-list cosmic_list.txt --data-dir Data/Delay \
        --lags=-8:7 --workers 16
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys

import numpy as np

FEATURES = [
    "DST Index", "AE Index", "ap index", "f10.7 index", "Kp",
    "Plasma Flow Speed", "Bx GSE, GSM", "By GSM", "Bz GSM",
]
N_FEATURES = len(FEATURES)
N_COLS = 5 + N_FEATURES + 4  # Alt,mLat,mLon,mLT,VTEC0 + 9 drivers + doy,VTEC1,NmF2,year = 18


# --------------------------------------------------------------------------- #
# Pure helpers (testable without external data)
# --------------------------------------------------------------------------- #
def assemble_row(alt_f2, mlat, mlon, mlt, vtec0, x_omni, doy, vtec1, nmf2, year):
    """Build one `out` row in the canonical 18-column layout.

    Mirrors the notebook's
        out = np.hstack([Alt_F2, magm, VTEC0, X_omni, doy, VTEC1, NmF2, year]).
    """
    x_omni = np.asarray(x_omni, dtype=float).ravel()
    if x_omni.size != N_FEATURES:
        raise ValueError(f"x_omni must have {N_FEATURES} values, got {x_omni.size}")
    row = np.empty(N_COLS, dtype=float)
    row[0] = alt_f2
    row[1] = mlat
    row[2] = mlon
    row[3] = mlt
    row[4] = vtec0
    row[5:14] = x_omni
    row[14] = doy
    row[15] = vtec1
    row[16] = nmf2
    row[17] = year
    return row


def keep_row(row) -> bool:
    """Notebook acceptance filter: 0 < VTEC0 < 50, mLat finite and != 0."""
    return (row[4] > 0) and (row[4] < 50) and (not np.isnan(row[1])) and (row[1] != 0)


def doy_encoding(month: int, day: int) -> float:
    """The original's lossy day-of-year proxy: month + day/35 - 1."""
    return month + day / 35.0 - 1.0


# --------------------------------------------------------------------------- #
# OMNI
# --------------------------------------------------------------------------- #
# Hourly OMNI2 (omni2_YYYY.dat) word positions (1-indexed) for each FEATURE,
# in the same order as FEATURES, plus the fill value that marks "missing".
# Source: https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2.text
#   FEATURES = [DST, AE, ap, F10.7, Kp, FlowSpeed, Bx GSE/GSM, By GSM, Bz GSM]
OMNI2_WORDS = [41, 42, 50, 51, 39, 25, 13, 16, 17]
OMNI2_FILLS = [99999, 9999, 999, 999.9, 99, 9999.0, 999.9, 999.9, 999.9]
OMNI2_URL = ("https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/"
             "omni2_{year}.dat")


def _omni2_parse_line(parts):
    """Map one whitespace-split OMNI2 row to (year, doy, hour, 9-feature array)."""
    year = int(parts[0]); doy = int(parts[1]); hour = int(parts[2])
    vals = np.empty(N_FEATURES)
    for i, (w, fill) in enumerate(zip(OMNI2_WORDS, OMNI2_FILLS)):
        raw = float(parts[w - 1])
        # Real magnitudes are far below the fills; treat near-fill as missing.
        vals[i] = np.nan if abs(raw) >= 0.99 * fill else raw
    return year, doy, hour, vals


def load_omni(years, cache_dir="Data/omni2_cache", source="omni2",
              start=None, end=None):
    """Load hourly OMNI drivers and return a dict {(year, doy, hour): 9-array}.

    source="omni2" (default): download/parse NASA SPDF omni2_<year>.dat files
    (cached under cache_dir). source="aidapy": legacy path (needs heliopy<1).
    """
    if source == "aidapy":
        return _load_omni_aidapy(start, end)

    import urllib.request
    os.makedirs(cache_dir, exist_ok=True)
    index = {}
    for year in years:
        path = os.path.join(cache_dir, f"omni2_{year}.dat")
        if not os.path.isfile(path):
            url = OMNI2_URL.format(year=year)
            try:
                urllib.request.urlretrieve(url, path)
            except Exception as e:
                print(f"  [omni2] could not fetch {url}: {e}", file=sys.stderr)
                continue
        with open(path) as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < max(OMNI2_WORDS):
                    continue
                y, d, h, vals = _omni2_parse_line(parts)
                index[(y, d, h)] = vals
    return index


def _load_omni_aidapy(start, end):  # pragma: no cover - legacy / likely broken
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


def omni_at(omni_index, when: dt.datetime):
    """Return the 9 FEATURES values at the OMNI hour containing `when` (or NaNs)."""
    key = (when.year, when.timetuple().tm_yday, when.hour)
    vals = omni_index.get(key)
    return np.full(N_FEATURES, np.nan) if vals is None else vals


# --------------------------------------------------------------------------- #
# RO profile reading
# --------------------------------------------------------------------------- #
def _get_apex(when: dt.datetime):
    """Return a FRESH Apex object for `when`.

    NB: apexpy's Fortran backend keeps GLOBAL epoch state -- each Apex(date=...)
    overwrites shared coefficients. A cache of multiple Apex objects reused
    interleaved therefore yields wrong coordinates (the global epoch is whatever
    was set last). So we never cache across days; callers must create an Apex and
    use it immediately (read_profiles groups by day to stay both correct & fast).
    """
    from apexpy import Apex
    return Apex(date=when)


def read_ro_profile(path: str, delay_hour: int, omni_index,
                    fix_mlt: bool = False):
    """Read one ionPrf netCDF and return its `out` row (port of `readRO`)."""
    import netCDF4 as nc

    path = path.strip()
    f = nc.Dataset(path)
    try:
        vtec0 = float(f.getncattr("tec0"))
        vtec1 = float(f.getncattr("tec1"))
        year = int(f.getncattr("year"))
        month = int(f.getncattr("month"))
        day = int(f.getncattr("day"))
        lat_f2 = float(f.getncattr("edmaxlat"))
        lon_f2 = float(f.getncattr("edmaxlon"))
        alt_f2 = float(f.getncattr("edmaxalt"))
        nmf2 = float(f.getncattr("edmax"))
        ut = (f.getncattr("hour") + f.getncattr("minute") / 60.0
              + f.getncattr("second") / 3600.0)
    finally:
        f.close()

    ro_time = dt.datetime(year, month, day, int(ut))
    doy = doy_encoding(month, day)
    x_omni = omni_at(omni_index, ro_time + dt.timedelta(hours=delay_hour))

    # Apex magnetic coordinates (the only branch the notebook ever executes).
    apex = _get_apex(ro_time)
    mlat, mlon = apex.convert(lat_f2, lon_f2, "geo", "apex", height=alt_f2)
    mlt_lon = mlon if fix_mlt else lon_f2  # original passes geographic lon (sic)
    mlt = apex.mlon2mlt(mlt_lon, ro_time)

    return assemble_row(alt_f2, mlat, mlon, mlt, vtec0, x_omni, doy, vtec1, nmf2, year)


def read_profiles(paths, *, fix_mlt=False, apex_epoch="day", progress=True):
    """Read every ionPrf profile ONCE; return lag-independent fields.

    All RO-derived columns (Alt, mLat, mLon, mLT, VTEC0, doy, VTEC1, NmF2, year)
    are independent of the OMNI time lag, so they are computed a single time.
    Only the 9 OMNI driver columns (5:14) vary per lag and are filled later.

    Apex coordinates are computed grouped by calendar day so that exactly one
    Apex object is alive per day and is used contiguously -- this both avoids
    apexpy's global-state corruption and is fast.

    apex_epoch : {"day", "profile"}
        "day" (default): one Apex epoch per calendar day (vectorised convert).
        Matches per-profile `Apex(date=RO_time)` to within ~3e-4 deg (sub-day
        IGRF secular variation) -- negligible for a |mLat|>60 analysis but not
        bit-identical. "profile": a fresh Apex per profile (the notebook's exact
        behaviour); correct to ~1e-10 deg but ~N_days x slower.

    Returns (base[M,18] with OMNI cols 5:14 = NaN, ro_times[M]); the keep_row
    filter (VTEC0, mLat -- both lag-independent) is applied here, so M is the
    final retained count, identical for every lag.
    """
    import netCDF4 as nc
    from collections import defaultdict

    # --- pass 1: read metadata (no Apex yet) -------------------------------
    meta = []  # (alt, lat, lon, vtec0, vtec1, nmf2, year, month, day, ro_time)
    n = len(paths)
    for k, p in enumerate(paths):
        if progress and (k % 5000 == 0):
            print(f"    read {k}/{n}", end="\r", file=sys.stderr)
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
                print(f"    skip {p}: {e}", file=sys.stderr)
            continue
        meta.append((alt, lat, lon, vtec0, vtec1, nmf2, year, month, day, ro_time))

    # --- pass 2: Apex coords, grouped by day -------------------------------
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
        else:  # "profile" -- fresh Apex per profile, used immediately
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
    """Top-level pool worker: read_profiles on a subset of paths (own process,
    so apexpy's global state is isolated)."""
    chunk, fix_mlt, apex_epoch = args
    return read_profiles(chunk, fix_mlt=fix_mlt, apex_epoch=apex_epoch,
                         progress=False)


def read_profiles_parallel(paths, *, fix_mlt=False, apex_epoch="day", workers=1):
    """Parallel read pass: split paths across worker processes and concatenate.

    Returns (base[M,18], ro_times[M]) like read_profiles. Row order across
    chunks is not preserved, which is irrelevant for the MI analysis (it treats
    the rows as an unordered sample).
    """
    if workers <= 1 or len(paths) < 2 * workers:
        return read_profiles(paths, fix_mlt=fix_mlt, apex_epoch=apex_epoch)
    from multiprocessing import Pool
    # Contiguous chunks: the staged list is sorted by year/doy, so each worker
    # gets a contiguous date range and creates few Apex objects (one per day).
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


def fill_lag(base, ro_times, omni_index, delay: int):
    """Return a copy of `base` with OMNI columns (5:14) filled at RO_time+delay."""
    out = base.copy()
    td = dt.timedelta(hours=delay)
    for i, t in enumerate(ro_times):
        out[i, 5:14] = omni_at(omni_index, t + td)
    return out


def generate_lag(delay: int, paths, omni_index, *, fix_mlt=False,
                 workers=1, progress=True):
    """Build the `out` array for one lag (port of `Read_RO_delay`)."""
    rows = []
    n = len(paths)

    def _one(idx_path):
        idx, p = idx_path
        try:
            row = read_ro_profile(p, delay, omni_index, fix_mlt=fix_mlt)
        except Exception as e:  # skip unreadable profiles, like the notebook implicitly did
            if progress and idx < 5:
                print(f"    [lag {delay:+d}] skip {p}: {e}", file=sys.stderr)
            return None
        return row if keep_row(row) else None

    items = list(enumerate(paths))
    if workers and workers > 1:
        # OMNI index / xarray are not trivially picklable; multiprocessing here
        # would re-load them per worker. Kept simple (serial) unless the caller
        # has set up a pool externally. See note in module docstring.
        results = map(_one, items)
    else:
        results = map(_one, items)

    for k, row in enumerate(results):
        if progress and (k % 5000 == 0):
            print(f"    [lag {delay:+d}] {k}/{n}", end="\r", file=sys.stderr)
        if row is not None:
            rows.append(row)
    out = np.asarray(rows, dtype=float) if rows else np.empty((0, N_COLS))
    assert out.ndim == 2 and out.shape[1] == N_COLS, out.shape
    return out


# --------------------------------------------------------------------------- #
# Readiness check / self-test / main
# --------------------------------------------------------------------------- #
def cmd_check(args) -> int:
    print("=== make_delay_files readiness check ===")
    ok = True

    for mod in ("numpy", "scipy", "netCDF4", "apexpy"):
        try:
            __import__(mod)
            print(f"  [ok]   import {mod}")
        except Exception as e:
            ok = False
            print(f"  [MISS] import {mod}: {e}")

    # OMNI source reachability (default omni2 from SPDF).
    if args.omni_source == "omni2":
        try:
            import urllib.request
            req = urllib.request.Request(OMNI2_URL.format(year=2010), method="HEAD")
            urllib.request.urlopen(req, timeout=30)
            print("  [ok]   OMNI2 SPDF reachable")
        except Exception as e:
            print(f"  [warn] OMNI2 SPDF HEAD failed (may still GET): {e}")
    else:
        print("  [note] omni-source=aidapy is legacy and needs heliopy<1")

    if os.path.isfile(args.cosmic_list):
        import linecache as lc
        lines = [l for l in lc.getlines(args.cosmic_list) if l.strip()]
        print(f"  [ok]   cosmic_list: {args.cosmic_list} ({len(lines)} paths)")
        n_exist = sum(os.path.isfile(p.strip()) for p in lines[:20])
        tag = "ok" if n_exist else "MISS"
        if not n_exist:
            ok = False
        print(f"  [{tag}] {n_exist}/20 sampled ionPrf paths exist on disk")
        if not n_exist and lines:
            print(f"         e.g. {lines[0].strip()}")
    else:
        ok = False
        print(f"  [MISS] cosmic_list not found: {args.cosmic_list}")

    print(f"  output dir: {args.data_dir} "
          f"({'exists' if os.path.isdir(args.data_dir) else 'will be created'})")
    print("\nREADY" if ok else "\nNOT READY -- resolve the [MISS] items above "
          "(raw COSMIC RO data + deps). See MIGRATION_REPORT.md.")
    return 0 if ok else 2


def cmd_selftest(args) -> int:
    """Validate column assembly / filter logic without any external data."""
    print("=== make_delay_files self-test (no external data) ===")
    fails = []

    def chk(name, cond, detail=""):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
              + (f"  ({detail})" if detail else ""))
        if not cond:
            fails.append(name)

    x = np.arange(10, 19, dtype=float)  # 9 fake drivers
    row = assemble_row(alt_f2=300.0, mlat=65.0, mlon=120.0, mlt=11.5,
                       vtec0=12.0, x_omni=x, doy=5.4, vtec1=3.0,
                       nmf2=5.5e5, year=2010)
    chk("row length == 18", row.shape == (N_COLS,), f"{row.shape}")
    chk("Alt at col 0", row[0] == 300.0)
    chk("mLat at col 1", row[1] == 65.0)
    chk("VTEC0 at col 4", row[4] == 12.0)
    chk("9 drivers at cols 5:14", np.array_equal(row[5:14], x))
    chk("doy at col 14", row[14] == 5.4)
    chk("NmF2 at col 16 (what mi_regressor reads)", row[16] == 5.5e5)
    chk("year at col 17", row[17] == 2010)

    chk("doy_encoding(7,1) == 7+1/35-1", abs(doy_encoding(7, 1) - (7 + 1/35 - 1)) < 1e-12)

    good = assemble_row(300, 65, 0, 0, 10.0, x, 0, 0, 1, 2010)
    bad_vtec = assemble_row(300, 65, 0, 0, 80.0, x, 0, 0, 1, 2010)  # VTEC0 >= 50
    bad_mlat = assemble_row(300, 0.0, 0, 0, 10.0, x, 0, 0, 1, 2010)  # mLat == 0
    chk("keep_row accepts valid", keep_row(good))
    chk("keep_row rejects VTEC0>=50", not keep_row(bad_vtec))
    chk("keep_row rejects mLat==0", not keep_row(bad_mlat))

    # Schema matches what mi_regressor expects.
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import mi_regressor as mr
        out = np.tile(row, (8, 1))
        mf, mm, ms, nk = mr.compute_mi_for_lag(out, num_resample=2, mlat_min=60.0)
        chk("mi_regressor consumes the schema", mf.shape == (9,), f"n_kept={nk}")
    except Exception as e:
        chk("mi_regressor consumes the schema", False, repr(e)[:120])

    if fails:
        print(f"\nRESULT: {len(fails)} FAILURE(S): {fails}")
        return 1
    print("\nRESULT: ALL SELF-TESTS PASSED")
    return 0


def cmd_run(args) -> int:
    import linecache as lc
    start, stop = (int(v) for v in args.lags.split(":"))
    lags = list(range(start, stop + 1))
    paths = [l for l in lc.getlines(args.cosmic_list) if l.strip()]
    if args.n_profiles:
        paths = paths[: args.n_profiles]
    print(f"Lags: {lags}  |  profiles: {len(paths)}")

    # OMNI years must cover the RO span plus a day of lag slop on each side.
    years = range(args.omni_year_start, args.omni_year_end + 1)
    print(f"Loading OMNI ({args.omni_source}) for years "
          f"{args.omni_year_start}-{args.omni_year_end}...")
    omni_index = load_omni(years, cache_dir=args.omni_cache,
                           source=args.omni_source)
    print(f"OMNI hourly records indexed: {len(omni_index)}")
    if not omni_index:
        print("ERROR: no OMNI data loaded; cannot fill driver columns.",
              file=sys.stderr)
        return 3

    print(f"Reading {len(paths)} profiles once (lag-independent fields, "
          f"workers={args.workers})...")
    base, ro_times = read_profiles_parallel(
        paths, fix_mlt=args.fix_mlt, apex_epoch=args.apex_epoch,
        workers=args.workers)
    print(f"Retained {base.shape[0]} profiles after acceptance filter.")

    os.makedirs(args.data_dir, exist_ok=True)
    from scipy import io as sio
    for lag in lags:
        out = fill_lag(base, ro_times, omni_index, lag)
        dest = os.path.join(args.data_dir, f"{args.out_prefix}{lag}.mat")
        sio.savemat(dest, {"out": out})
        print(f"  wrote {dest}  shape={out.shape}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cosmic-list", default="cosmic_list.txt")
    ap.add_argument("--data-dir", default="Data/Delay")
    ap.add_argument("--lags", default="-8:7",
                    help="inclusive lag range 'start:stop' (pass as --lags=-8:7)")
    ap.add_argument("--out-prefix", default="all_",
                    help="output filename prefix; default 'all_' -> all_<lag>.mat")
    ap.add_argument("--n-profiles", type=int, default=None,
                    help="limit number of profiles (default: all in the list)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--fix-mlt", action="store_true",
                    help="use magnetic longitude in mlon2mlt (original used geographic)")
    ap.add_argument("--apex-epoch", choices=["day", "profile"], default="day",
                    help="Apex epoch granularity: 'day' (fast, ~3e-4 deg from "
                         "exact) or 'profile' (notebook-exact, slower)")
    ap.add_argument("--omni-source", choices=["omni2", "aidapy"], default="omni2",
                    help="OMNI driver source (default omni2 from NASA SPDF)")
    ap.add_argument("--omni-cache", default="Data/omni2_cache",
                    help="directory to cache downloaded omni2_<year>.dat files")
    ap.add_argument("--omni-year-start", type=int, default=2006)
    ap.add_argument("--omni-year-end", type=int, default=2018)
    ap.add_argument("--check", action="store_true", help="readiness check only")
    ap.add_argument("--selftest", action="store_true",
                    help="validate column/filter logic without external data")
    args = ap.parse_args(argv)

    if args.selftest:
        return cmd_selftest(args)
    if args.check:
        return cmd_check(args)
    return cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
