# Data catalog — 4D_Ne

The data files themselves are **not** in git (too large — tens of GB; `Data/` and
`Projects/` are git-ignored). This catalog documents every dataset: where it
lives, what it contains, its schema, size, how it was produced, and which scripts
use it. This README *is* tracked so the catalog is visible without the data.

Data lives in three places:
- **`Data/`** (this folder) — derived/prepared tables and caches (~17 GB).
- **`/glade/derecho/scratch/$USER/cosmic/`** — raw COSMIC-1 downloads (~49 GB, re-downloadable).
- **`Projects/`** — trained models and evaluation outputs.

DoY note: throughout, the column called `DoY` in the `Delay` tables is the
lossy encoding `month + day/35 − 1` used by the original pipeline (≈ month),
not the true day-of-year.

---

## `Data/Delay/` — lag-shifted driver tables (~8.1 GB)

| File | Var | Shape | Columns | Source → Used by |
|---|---|---|---|---|
| `all_<lag>.mat` (lags −8…+7, 16 files, ~482 MB ea.) | `out` | N×18 | `Alt, mLat, mLon, mLT, VTEC0, DST, AE, ap, F10.7, Kp, Ve, Bx, By, Bz, DoY, VTEC1, NmF2, year` | `Src/make_delay_files.py` → `Src/mi_regressor.py` |
| `330w_0_minutes_geom_height0.mat` (3.3 M rows) | `out` | N×15 | `Alt, mLat, mLon, mLT, VTEC0, drv5, drv6, drv7, drv8, DoY, VTEC1, NmF2, year, F10.7, PF10.7` | original notebook; NmF2 model input |
| `HRO_iono_height0.mat` (3.54 M rows) | `X` | N×10 | `mLat, mLon, LT, drv1, drv2, drv3, drv4, DoY, F10.7, PF10.7` | NmF2 sub-model (`train_nmf2_*.py`, `diagnose_nmf2.py`) |
| ″ | `Y` | N×1 | `NmF2` (el cm⁻³) | |
| ″ | `Ref` | N×6 | `year, month, day, LT, lat, lon` | |

(NmF2 = peak electron density `edmax`; hmF2 = peak height `edmaxalt`; VSH =
vertical scale height, the 1/e-drop altitude span — see `Src/make_fig1_profile.py`.)

## `Data/data_4d_ne/` — 12-feature ANN training data (~8.6 GB)

| File | Var | Shape | Columns |
|---|---|---|---|
| `XY_<i>.mat` (i=0…18, ~104 MB ea.) | `X` | ~1 M×12 | `Alt, Lat, Lon, Azi, DST, AE, AP, F10.7, Kp, Vf, DoY, UT` |
| ″ | `Y` | ~1 M×2 | `TEC, Ne` (Ne = target, el cm⁻³) |
| `RO_output.mat` | `out` | 330 k×11 | RO-derived output table |
| `res_variables.mat` | — | — | (empty/placeholder) |

Used by `Src/train_ne_ann.py`, `Src/make_fig3_counts.py`, `Src/make_fig45_errmaps.py`.
These files lack the three measured peak features (NmF2/hmF2/VSH); the
15-feature model rebuilds those from the raw profiles (see below).

## `Data/omni2_cache/` — OMNI hourly drivers (~36 MB)

`omni2_<year>.dat` — NASA SPDF OMNI2 hourly files, auto-downloaded by
`make_delay_files.py` / `reproduce_ne_15feat.py`. Columns used (word index):
DST(41), AE(42), ap(50), F10.7(51), Kp(39), flow speed(25), Bx(13), By(16), Bz(17).
Source: `https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/`.

## Other `Data/` files

- `HRO/` (~108 MB) — `*.lst` ionosonde/OMNI lists and `*.gif` plots.
- `coastline.mat` — coastline coordinates for global maps.
- `CMI_5million.mat` — `CMI` (14×14) conditional-mutual-information matrix.

---

## Raw COSMIC-1 downloads — `/glade/derecho/scratch/$USER/cosmic/` (~49 GB)

- `<year>/<doy>/ionPrf_*` — COSMIC-1 `repro2013` Level-2 ionospheric profiles
  (netCDF; files end `_nc`). Variables: `MSL_alt, GEO_lat, GEO_lon, OCC_azi,
  TEC_cal, ELEC_dens`; attributes: `edmax`(NmF2), `edmaxalt`(hmF2),
  `edmaxlat/lon`, `year/month/day/hour/minute/second`, `tec0/tec1`.
- `cosmic_list_local.txt` — list of staged profile paths (use as `--cosmic-list`).
- Obtain with `bash Src/download_cdaac.sh` (open portal, no login):
  `https://data.cosmic.ucar.edu/gnss-ro/cosmic1/repro2013/level2/<year>/<doy>/`.

`../cosmic_list.txt` (345 MB, repo root) is the *original* path list pointing at a
decommissioned host — superseded by `cosmic_list_local.txt`. **Keep it out of git.**

## Model outputs — `Projects/` (git-ignored)

| Path | Contents |
|---|---|
| `ne15/` | `ne15.pt` (15-feature L2-ANN), `norm.mat`, `test_arrays.npz` (`X` 2.13 M×15, `pred`, `ref`, `features`), `metrics.json` (median rel-err ≈ 5.6 %) |
| `ne_ann/` | 12-feature model + `norm_stats.mat` + `metrics.json` |
| `nmf2_submodel/`, `nmf2_dkgp/` | NmF2 sub-model (ANN / deep-kernel KISS-GP) artifacts |

Small committed inputs that regenerate the figures without the bulk data live in
`../figures/data/` (one ionPrf profile, a 250 k-row `ne15_test_sample.npz`,
`mi_results.npz`).

---

## Regeneration quick reference

| Dataset | Command |
|---|---|
| Raw COSMIC profiles | `bash Src/download_cdaac.sh` |
| `Delay/all_<lag>.mat` | `python Src/make_delay_files.py --cosmic-list <list> --data-dir Data/Delay --lags=-8:7` |
| 15-feature model + test set | `python Src/reproduce_ne_15feat.py --list <cosmic_list_local.txt> --n-profiles 40000 --device cuda --out Projects/ne15` |
| MI results | `python Src/mi_regressor.py --data-dir Data/Delay --lags=-8:7 --method fast --out mi_results.npz` |
