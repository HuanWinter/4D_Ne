# MATLAB → Python Migration Report

**Date:** 2026-05-24
**Scope:** Rewrite all non-Python research code to Python, then sanity-check
correctness and reproducibility.

## 1. Inventory and what is "non-Python"

A full scan of the repo (`*.m`, `*.ipynb`, `*.py`) found that the only
**non-Python research code** is the mutual-information (MI) feature/lag analysis,
implemented as three MATLAB files. Everything else is already Python (Jupyter
notebooks and `.py` scripts), so no language conversion was required for those.

| Old file (MATLAB) | Role | New Python file | Status |
|---|---|---|---|
| `MI_Kraskov.m` | Kraskov KSG mutual-information estimator, I(1), k=1 | `Src/mi_kraskov.py` → `mi_kraskov()` | Converted, validated bit-for-bit |
| `MI_regressor.m` | Driver: load delay data, filter, normalize, resample, compute MI per driver/lag | `Src/mi_regressor.py` | Converted; runs end-to-end on synthetic data |
| `show_MI.m` | 3×3 "MI vs time gap" plot | `Src/show_mi.py` → `show_mi()` | Converted, headless-capable |
| — | sanity tests (new) | `Src/test_mi_kraskov.py` | New |
| `NmF2-MI.ipynb` cells 1/3/4/6/7 (`readRO`/`Read_RO_delay`) | generator for the `all_<lag>.mat` delay datasets | `Src/make_delay_files.py` | Ported; cannot run here (raw data missing — see §8) |

Entrypoints / dependencies:
- `mi_kraskov.py`: depends on `numpy`, `scipy.special.digamma`, `scipy.spatial.cKDTree`. Library + `--selftest` CLI.
- `mi_regressor.py`: depends on `numpy`, `scipy.io`, and `mi_kraskov`. CLI entrypoint.
- `show_mi.py`: depends on `numpy`, `matplotlib`. CLI entrypoint.

There is also a partial Python reimplementation of the driver in the notebook
`NmF2-MI.ipynb`, but it uses a **different estimator** (see §4) and is not a
faithful port; it is left untouched.

## 2. Faithfulness of the conversion

`MI_Kraskov.m` was ported exactly. The MATLAB code finds, for each sample,
`Eps` = the Chebyshev (max-norm) distance to its nearest neighbour in the joint
(X,Y) space, then counts `nx = #{j≠i: |X(j)-X(i)| < Eps}` and `ny` likewise, and
returns `I1 = ψ(1) − mean(ψ(nx+1)+ψ(ny+1)) + ψ(n)`. The Python `method="exact"`
backend reproduces this arithmetic identically.

**Algorithm behaviour was not changed.** Two intentional, documented choices:

1. **Two backends.** `method="exact"` (default) is bit-for-bit faithful (O(n²)).
   `method="fast"` is an O(n log n) KD-tree/binary-search variant that agrees to
   ~1e-3 nats; the only differences are off-by-one marginal counts at points
   lying exactly at the neighbour distance (`x[j] < x[i]+Eps` rounds differently
   from `|x[j]-x[i]| < Eps`). `fast` is opt-in for large datasets and is **not**
   used by default. This was caught and fixed during validation (an early
   KD-tree-only version drifted ~1e-3 — see §5).
2. **`std`**: MATLAB `std` is the sample std (N−1). The Python driver uses
   `ddof=1` to match. (The old notebook used numpy's default population std,
   ddof=0 — a discrepancy, not carried over.)

## 3. Data filter / pipeline parity (`MI_regressor.m`)

Reproduced exactly:
- Column mapping (0-based): `out[:,0]`=Altitude, `out[:,1]`=mLat,
  `out[:,5:14]`=9 drivers (DST, AE, ap, F10.7, Kp, Ve, Bx, By, Bz),
  `out[:,16]`=NmF2. (MATLAB cols 1,2,6:14,17.)
- Retain rows with `mLat > 60` **and** no NaN across the 9 drivers.
- Per-column min-max normalize drivers to [0,1].
- `num_resample` (default 16) bootstrap draws of `floor(0.6·N)` rows (sampling
  **without** replacement within a draw, as MATLAB `randperm` does); MI computed
  per driver vs `log(NmF2)`; record mean and std. Full-sample MI also recorded.

Deviation from the notebook (not the MATLAB) made explicit via flags:
- The notebook used `|mLat| > 60` and an extra `VTEC1 > 0` filter, and rounded
  variables into 0–100 bins. The port follows the **MATLAB** script
  (`mLat > 60`, continuous values). `--abs-mlat` opts into the `|mLat|` variant.

## 4. The two MI methods are scientifically different (flag for authors)

- `MI_Kraskov.m` / `mi_kraskov.py`: **continuous** Kraskov k-NN estimator (nats).
- `NmF2-MI.ipynb`: `sklearn.normalized_mutual_info_score` + `pyitlib` on values
  **rounded into 0–100 bins** — a discretised (and *normalized*) estimator, in
  different units and with different bias. These are not interchangeable. If the
  manuscript reports MI from one and discusses the other, that should be
  reconciled. This migration preserves the **Kraskov** (MATLAB) definition.

## 5. Sanity / correctness validation

Run: `python Src/test_mi_kraskov.py` → **ALL CHECKS PASSED**.

| # | Check | Result |
|---|---|---|
| Static | `py_compile` on all 4 files | PASS |
| 1 | `exact` backend vs O(n²) MATLAB-loop reference, 6 random sizes | bit-exact (diff `0.0e+00`) |
| 1 | `fast` backend vs reference | within 3.2e-3 nats (documented approximation) |
| 2 | Bivariate-Gaussian analytic MI `−½ln(1−ρ²)`, ρ∈{0,.3,.6,.9} | within tol (e.g. ρ=.9: est 0.842 vs 0.830) |
| 3 | Cross-check vs `sklearn` KSG (k=1) | diff ≤ 0.001 nats |
| 4 | Independence ≈ 0; MI monotone in coupling | PASS |
| 5 | `normalize_columns` range [0,1]; constant col → 0 (no NaN/inf) | PASS |
| 6 | Driver synthetic end-to-end: shapes (9,), all finite, std≥0, strongest driver ranked #1, deterministic w/ fixed seed | PASS |
| 7 | Resample integrity: subsample size = floor(0.6·N), all indices in-set, no dup within a draw | PASS |

**Gold-standard cross-check against the original MATLAB** (module `matlab/R2024b`):
ran `MI_Kraskov.m` on three fixed Gaussian samples and compared to
`mi_kraskov(method="exact")`:

| case | python exact | MATLAB | abs diff |
|---|---|---|---|
| a (ρ=0.6, n=300) | 0.1142924189 | 0.1142924189 | 1.8e-11 |
| b (ρ=0.85, n=500) | 0.5308833980 | 0.5308833980 | 7.0e-12 |
| c (ρ=0.0, n=400) | −0.0743516980 | −0.0743516980 | 3.3e-11 |

Max difference **3.3e-11** (digamma round-off). The port is numerically
identical to the original.

**Leakage / split review:** The resampling in this pipeline is *bootstrap
variance estimation of MI*, not an ML train/test split — there is no held-out
test set and subsamples intentionally overlap across draws. So ML "data leakage"
does not apply. Verified that draws never index outside the retained set and
contain no within-draw duplicates. **Metrics:** MI formula verified against the
analytic Gaussian truth and an independent implementation (sklearn).

## 6. Blockers / missing data

- **The driver's real inputs are missing.** `MI_regressor.m` (and the notebook)
  load `Data/Delay/all_<lag>.mat` for lags −8…+7, each holding an `out` array
  with ≥17 columns. **None of these files exist.** `Data/Delay/` contains only
  `330w_0_minutes_geom_height0.mat` (an `out` with **15** columns — different
  schema, NmF2 column absent) and `HRO_iono_height0.mat` (`X/Y/Ref`, unrelated).
  → The driver therefore **cannot be run on the real data in this checkout.** It
  was validated end-to-end on deterministic synthetic data instead, and on real
  data it reports each missing file clearly and writes NaN rows rather than
  crashing.

  **To proceed:** place the `all_<lag>.mat` files (`out` ≥17 cols, layout in §3)
  in a directory and run with `--data-dir <that dir>`.

## 7. Regenerating the `all_<lag>.mat` files (`Src/make_delay_files.py`)

The generator was ported from `NmF2-MI.ipynb`. It loops over the COSMIC RO
`ionPrf` netCDF profiles, derives the F2-peak quantities, samples the 9 OMNI
drivers at `RO_time + lag_hours`, applies the original acceptance filter
(`0 < VTEC0 < 50`, `mLat` finite and ≠ 0), and writes one `out` array per lag in
the exact 18-column layout `mi_regressor.py` reads (NmF2 at col 16).

Validated here (no external data needed):
- `python Src/make_delay_files.py --selftest` → **ALL SELF-TESTS PASSED**
  (column positions, `doy` encoding, acceptance filter, and a round-trip showing
  `mi_regressor.compute_mi_for_lag` consumes the produced schema).

Environment is now prepared (`python Src/make_delay_files.py --check`):
- `netCDF4` 1.7.4 (pip wheel) and `apexpy` 2.1.0 (built from source with
  **gfortran**, since there is no conda-forge package and the Intel `ifx`
  compiler fails on `igrf.f90`) are installed in the `GONG_pred` env.
- **OMNI drivers no longer use `aidapy`.** aidapy imports `heliopy`, which is a
  tombstone package that refuses to load. The generator now reads the same
  hourly OMNI data directly from NASA SPDF (`omni2_<year>.dat`, `--omni-source
  omni2`, the default). Verified live: e.g. 2010-01-01 00 UT → F10.7=72.7,
  Dst=5, Kp=0, V=283 km/s (correct for solar minimum). Because mi_regressor
  min-max normalises each driver, OMNI2-vs-aidapy unit differences (e.g. Kp×10)
  do not change the MI.

**The remaining blocker is the raw COSMIC RO data.** The 3,378,981 `ionPrf`
paths in `cosmic_list.txt` point to `/export/scratch2/andong/...` (a
decommissioned workstation); **0/20 sampled paths exist** on glade. They are
re-downloadable from the CDAAC archive (see below).

Why existing `.mat` files don't help: each lag differs **only** in the OMNI
columns (sampled at `RO_time ± lag`); the RO-derived columns are identical
across lags. But the per-profile **UT hour is never stored** in any saved `.mat`
(only a lossy `doy = month + day/35 − 1` and `year`), and the only available
file, `330w_0_minutes_geom_height0.mat`, is a different 15-column variant with
no absolute timestamp. So the source `ionPrf` profiles are required.

**Data source (open, no login):** the COSMIC-1 reprocessed profiles are served
from the open COSMIC data portal —
`https://data.cosmic.ucar.edu/gnss-ro/cosmic1/repro2013/level2/<year>/<doy>/ionPrf_repro2013_<year>_<doy>.tar.gz`.
(The older `cdaac-www…/rest/tarservice` endpoint is auth-walled and additionally
requires per-account COSMIC-mission access — a registration step — so it is not
used.) `Src/download_cdaac.sh` targets the open portal. Files inside the tars
end in `_nc` (not `.nc`); they are still netCDF and read fine.

**To regenerate** (COSMIC-1 `cosmic1/repro2013` ionPrf, 2675 days
2007.001–2014.120, ~3.38M profiles):
1. `bash Src/download_cdaac.sh` — stages all days to `$STAGE`
   (default `/glade/derecho/scratch/$USER/cosmic`) and writes
   `cosmic_list_local.txt`. Resumable; `DAYS="2010.182 ..."` for a subset.
2. `python Src/make_delay_files.py --cosmic-list <STAGE>/cosmic_list_local.txt --data-dir Data/Delay --lags=-8:7`
   (heavy — run as a PBS batch job; reads each profile once and varies only the
   OMNI columns per lag).
3. `python Src/mi_regressor.py --data-dir Data/Delay --lags=-8:7 --out mi_results.npz`

**FULL regeneration completed (entire COSMIC-1 repro2013, 2007–2014).**
Downloaded all 2675 days (3,381,885 profiles, ~52 GB) from the open portal,
then `make_delay_files` (16 workers, `--apex-epoch day`) wrote
`Data/Delay/all_-8.mat … all_7.mat`, each **(3,351,393 × 18)** (99.1% kept;
OMNI NaN only 0.17%). `mi_regressor` (`--method fast`, mLat>60, ~344k
samples/lag) produced `mi_results.npz` + `mi_results.png`. Top drivers of
log(NmF2) at high latitude, full solar cycle:

| driver | MI | peak lag | interpretation |
|---|---|---|---|
| F10.7 | 0.34 | flat | solar EUV — dominant, lag-independent |
| AE    | 0.18 | −1 h | auroral activity leads NmF2 by ~1 h |
| Ve    | 0.11 | +2 h | solar-wind speed |
| Bz/By | 0.04 | −1/−3 h | IMF coupling |

This matches expected high-latitude ionospheric physics. (In the earlier 4-day
validation slice F10.7 showed ~0 MI only because it barely varied over 4 days;
across 2007–2014 it spans 65–253 sfu and dominates.)

A bug fixed during the full run: `normalize_columns` now uses `nanmin/nanmax`
(MATLAB's min/max ignore NaN; numpy's propagate it). The old version let the
0.17% of rows with a missing OMNI driver poison a whole normalized column and
crash the KD-tree. Surfaced only at full scale (validation days had full OMNI
coverage).

Three faithful/robustness points (see module docstring + flags):
- **OMNI** now comes from NASA SPDF OMNI2, not the dead aidapy (§ above).
- **apexpy global state**: apexpy's Fortran backend keeps a single global epoch,
  so caching multiple days' `Apex` objects and reusing them interleaved corrupts
  coordinates (≈1e-3 deg). The generator avoids this by reading every profile
  once and computing Apex grouped by day. `--apex-epoch profile` reproduces the
  notebook's per-profile `Apex(date=RO_time)` exactly (verified 0.0e+00 vs an
  isolated computation); the default `--apex-epoch day` is ~3e-4 deg off but much
  faster — negligible for this analysis.
- The dead `aacgmv2` branch (never executed → Apex only), and `mlon2mlt` called
  with the *geographic* longitude (likely an upstream bug; `--fix-mlt` opts into
  the magnetic-longitude version) are preserved from the notebook.

## 8. Assumptions and limitations

- Univariate X, Y (the MATLAB original's use). Multivariate inputs use a max-norm
  marginal-window count; only the univariate path is validated against MATLAB.
- `make_synthetic_out` is a smoke-test fixture, not scientific data.
- The `fast` backend is approximate (~1e-3 nats); use `exact` (default) for
  results you intend to publish.
- Notebooks (`*.ipynb`) and existing `*.py` were left as-is (already Python);
  only the three MATLAB files were converted.

## 9. Attempt to reproduce the manuscript's ANN results (Changyong/)

Scope: the paper (`Changyong/submit/main.tex`) reports an L2-ANN topside-Nₑ model
beating IRI-2016 by 35/36/53% (COSMIC-1/GRACE/ISR), with NmF2 and hmF2 sub-models
(NmF2 22.5% vs IRI 33.5%). This is a *different* pipeline from the MI migration
above and is already Python. We attempted to reproduce it.

Environment prepared: torch 2.4 + CUDA, PyIRI, madrigalWeb, gpytorch installed;
GRACE RO located at `data.cosmic.ucar.edu/gnss-ro/grace/postProc/level2`; PBS
account P28100036; jobs run via `qsub` (main→gpu).

What ran (all on the real COSMIC data, on GPU):
| model | what | result | paper |
|---|---|---|---|
| main Nₑ (12-feature L2-ANN, `Src/train_ne_ann.py`) | exact architecture/optimizer | RMSE 1.01×10⁵, rel-err 86% | full 15-feat: 2% |
| NmF2 sub-model, global ANN (`Src/train_nmf2_submodel.py`) | plain net | 61% mean / **44% median** | 22.5% |
| NmF2 sub-model, regional KISS-GP (`Src/train_nmf2_dkgp.py`) | the paper's actual model class (DNN→SKI-GP, 12 lat×LT regions, MLL, 100 iters) | **44% median** | 22.5% |

**UPDATE / CORRECTION (main model IS reproducible).** The pessimistic conclusion
below applies only to the NmF2/hmF2 *sub-models* (peak predicted from geophysics
alone). The **main topside-Nₑ model reproduces faithfully** once the three
measured features (NmF2=`edmax`, hmF2=`edmaxalt`, VSH=1/e-drop scale height) are
reconstructed directly from the `ionPrf` profiles — they were simply absent from
the prepared `XY_*.mat`. `Src/reproduce_ne_15feat.py` builds the 15-feature
samples from the raw profiles and trains the paper's L2-ANN:

| model | median rel-err | RMSE | note |
|---|---|---|---|
| 12-feature (XY_*.mat) | 49% | 1.0×10⁵ | missing NmF2/hmF2/VSH |
| **15-feature (rebuilt, 40k profiles)** | **5.6%** | **0.11×10⁵** | out-of-sample yrs 2009&2013 |
| paper full model | ~2% | 0.43×10⁵ | |

`Src/make_fig45_repro.py` then regenerates Figs 4-5 (all 9 panels incl.
ln(NmF2), hmF2) at paper-range magnitudes, and they **confirm the manuscript's
conclusions**: (1) relative error larger at high latitude, (2) RMSE peaks at the
equator (Nₑ magnitude), (3) relative error decreases with increasing NmF2.

**Sub-models also reproduce (resolved).** The first attempts used the prepared
`HRO_iono_height0.mat` and stalled at ~44% median rel-err. The diagnostic
(`Src/diagnose_nmf2.py`) found why: in that table **local time carries ~no
information** (corr≈0, MI≈0.02 with NmF2) — physically impossible for a clean
topside dataset, and a gradient-boosting ceiling on it tops out at ~38%. So the
gap was the **feature table**, not the model. Rebuilding the sub-model features
from the raw `ionPrf` profiles (correct local time, season, F10.7/PF10.7, Kp,
DST, AE, ap) fixes it — `Src/train_submodels.py`, out-of-sample years 2009 &
2013:

| sub-model | this reproduction | paper | IRI-2016 |
|---|---|---|---|
| NmF2 (median rel-err) | **16.7%** | 22.5% | 33.5% |
| hmF2 (median rel-err) | **4.5%**  | 5.8%  | 10.3% |

Both beat the paper's reported errors and IRI. So the full analysis — main 4D
model **and** sub-models — reproduces from the raw COSMIC profiles. The earlier
"not reproducible" verdict applied only to the broken `HRO_iono` table; it is
overturned. (`Src/train_nmf2_submodel.py` and `Src/train_nmf2_dkgp.py` keep the
two earlier HRO-table attempts for the record.)

Still not attempted: the 35/36/53%-vs-IRI numbers on COSMIC/GRACE/ISR, which need
IRI-2016 + GRACE + ISR data (PyIRI ≈ IRI-2020 is installed as a substitute).

**Figure reproduction** (from real data, no broken pipeline needed):
- `Src/make_fig1_profile.py` -> Fig. 1 (Nₑ profile + NmF2/hmF2 + VSH linear-fit)
  from the exact COSMIC-1 C06/DoY-319/2008 profile. VSH = 1/e-drop altitude span
  (definition recovered from main-CPU-short.ipynb). **Faithful.**
- `Src/make_fig3_counts.py` -> Fig. 3 (sample counts per variable, 70/15/15
  train/cv/test). Builds all **15 variables** from the profiles, incl. hmF2,
  NmF2 and VSH (these are not in the prepared XY tables).
- `Src/make_fig45_errmaps.py` -> Figs. 4-5 (median rel-err / RMSE per variable,
  on the model's held-out test split). Shows the error-vs-variable **shape**, but
  uses this reproduction's 12-feature ANN (~49% median here), **not** the paper's
  model -- so the magnitudes are not the paper's.
