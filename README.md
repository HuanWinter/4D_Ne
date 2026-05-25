# 4D_Ne — Four-dimensional topside ionospheric electron-density model

Machine-learning model of the topside ionospheric electron density
(N_e, above the F2 peak) in latitude, longitude, altitude, and time,
built from GNSS radio-occultation (GNSS-RO) data. This repository contains the
data pipeline, the model code, the mutual-information feature analysis, and the
scripts that produce the manuscript figures.

> Manuscript: *A four-dimensional topside ionospheric electron-density model
> from GNSS radio occultation* (see `Changyong/`). The model is an
> L2-regularized artificial neural network (L2-ANN) with NmF2 and hmF2
> sub-models, evaluated against IRI-2016, GRACE, and incoherent-scatter-radar
> (ISR) data.

## Repository layout

```
Src/                     all executable code (see below)
figures/                 generated figures + small inputs + README (plotting)
Changyong/               manuscript (LaTeX/PDF) and reviewer material
MIGRATION_REPORT.md      MATLAB→Python migration + reproduction report
Config.json              model / preprocessing configuration
Data/  Projects/         data and run outputs (git-ignored; see "Data" below)
*.ipynb                  original research notebooks (exploratory)
```

### `Src/` scripts

| Script | Purpose |
|---|---|
| `mi_kraskov.py` | Kraskov KSG mutual-information estimator (Python port of `MI_Kraskov.m`) |
| `mi_regressor.py` | MI of each space-weather driver vs NmF2 across time lags (port of `MI_regressor.m`) |
| `show_mi.py` | plot MI vs time lag (port of `show_MI.m`) |
| `test_mi_kraskov.py` | numerical/sanity tests for the MI estimator |
| `download_cdaac.sh` | stage COSMIC-1 `repro2013` ionPrf profiles from the open COSMIC data portal |
| `make_delay_files.py` | build the lag-shifted `Delay/all_<lag>.mat` datasets (ionPrf + OMNI2 + Apex) |
| `reproduce_ne_15feat.py` | build the 15-feature topside-Ne dataset from profiles and train the L2-ANN |
| `train_ne_ann.py` | train the L2-ANN on the 12-feature prepared data |
| `train_nmf2_submodel.py`, `train_nmf2_dkgp.py` | NmF2 sub-model (ANN / deep-kernel KISS-GP) |
| `diagnose_nmf2.py` | data sanity + diagnostics for the NmF2 sub-model |
| `make_fig1_profile.py` | Fig. 1 — N_e profile with NmF2/hmF2 and VSH fit |
| `make_fig3_counts.py` | Fig. 3 — sample counts per variable (train/cv/test) |
| `make_fig45_repro.py`, `make_fig45_errmaps.py` | Figs. 4–5 — relative error / RMSE per variable |
| `run_*.pbs` | PBS batch jobs (Derecho GPU/CPU) |

## Environment

Python 3.10+ with `numpy`, `scipy`, `pandas`, `scikit-learn`, `torch`,
`netCDF4`, `apexpy`, `gpytorch`, `matplotlib` (and `PyIRI` / `madrigalWeb` for
the IRI / ISR comparisons). On NCAR Derecho the project env used here is
`/glade/work/$USER/conda-envs/GONG_pred`.

## Workflows

**1. Mutual-information driver analysis** (which space-weather drivers, at which
time lag, most inform NmF2):
```bash
python Src/make_delay_files.py --cosmic-list <list> --data-dir Data/Delay --lags=-8:7
python Src/mi_regressor.py --data-dir Data/Delay --lags=-8:7 --method fast --out mi_results.npz
python Src/show_mi.py mi_results.npz --save figures/mi_results.png --no-show
```
The `exact` MI backend reproduces `MI_Kraskov.m` to ~1e-11 (verified against
MATLAB R2024b). See `Src/test_mi_kraskov.py`.

**2. Download COSMIC-1 data** (open portal, no login):
```bash
bash Src/download_cdaac.sh                 # all days, or DAYS="2010.182 ..." for a subset
```

**3. Reproduce the topside-Ne model and figures:**
```bash
python Src/reproduce_ne_15feat.py --list <cosmic_list_local.txt> --n-profiles 40000 \
    --device cuda --out Projects/ne15
python Src/make_fig45_repro.py --npz Projects/ne15/test_arrays.npz \
    --fig4 figures/fig4_repro.png --fig5 figures/fig5_repro.png
```
See `figures/README.md` for the exact command behind each figure and the small
committed inputs that regenerate them without the multi-GB raw data.

## Reproduction status

The main 4D topside-Ne model reproduces the manuscript: out-of-sample
(years 2009 & 2013) **median relative error ≈ 5.6 %**, and the regenerated
Figs. 4–5 confirm the paper's conclusions (relative error larger at high
latitude, RMSE peaks at the equator, error decreases with increasing NmF2).
The stand-alone NmF2/hmF2 sub-models (peak predicted from geophysics alone, for
the "sub-models" mode / GRACE / ISR) are not yet fully reproduced. Full details,
including the gap analysis, are in `MIGRATION_REPORT.md` (§9).

## Data availability

Large datasets are **not** stored in git:
- COSMIC-1 `repro2013` ionPrf profiles — re-downloadable via `Src/download_cdaac.sh`
  from `https://data.cosmic.ucar.edu/gnss-ro/cosmic1/repro2013/level2/`.
- OMNI hourly drivers — fetched from NASA SPDF (OMNI2) by `make_delay_files.py`.
- `Data/`, `Projects/` are git-ignored. `figures/data/` holds small inputs so the
  figures can be regenerated without the full archives.
