# Figures — plotting code, outputs, and inputs

Self-contained bundle to (re)produce the figures. Generator scripts live in
`../Src/`; small input artifacts are in `data/`. Use the project Python env
(numpy, scipy, torch, netCDF4, matplotlib), e.g.
`/glade/work/andonghu/conda-envs/GONG_pred/bin/python`.

## Figures and how to regenerate

| Output | Generator | Command |
|---|---|---|
| `fig1_Ne_profile.png` — Nₑ profile + NmF2/hmF2 + VSH fit (manuscript Fig 1) | `Src/make_fig1_profile.py` | `python Src/make_fig1_profile.py --file figures/data/ionPrf_C006.2008.319.07.40.G28.nc --save figures/fig1_Ne_profile.png` |
| `fig3_counts.png` — sample counts per variable (all **15** incl. hmF2/NmF2/VSH), train/cv/test (Fig 3) | `Src/make_fig3_counts.py` | builds from the ionPrf profiles: `python Src/make_fig3_counts.py --list <cosmic_list_local.txt> --n-profiles 40000 --save figures/fig3_counts.png` |
| `fig4_repro.png`, `fig5_repro.png` — Relative-Residual / RMSE per variable, **reproduced 15-feature model** (manuscript Figs 4–5) | `Src/make_fig45_repro.py` | `python Src/make_fig45_repro.py --npz figures/data/ne15_test_sample.npz --fig4 figures/fig4_repro.png --fig5 figures/fig5_repro.png` |
| `fig4_relerr.png`, `fig5_rmse.png` — same maps from the **12-feature** model (contrast; all panels incl. hmF2/NmF2/VSH as binning axes; magnitudes ~10× higher) | `Src/make_fig45_errmaps.py` | `python Src/make_fig45_errmaps.py --npz Projects/ne15/test_arrays.npz --model Projects/ne_ann` |
| `mi_results.png` — mutual information vs time lag | `Src/show_mi.py` | `python Src/show_mi.py figures/data/mi_results.npz --save figures/mi_results.png --no-show` |

## `data/` (committed, small)

- `ionPrf_C006.2008.319.07.40.G28.nc` (12 KB) — the COSMIC-1 profile for Fig 1.
- `ne15_test_sample.npz` (12 MB) — 250k-row subsample of the 15-feature model's
  held-out test set (features + prediction + reference). Reproduces Figs 4–5 to
  median rel-err 5.59% (full set: 5.58%).
- `mi_results.npz` (5 KB) — MI-vs-lag results for `show_mi.py`.

## Inputs NOT in git (too large) — for full regeneration

- `Data/data_4d_ne/XY_*.mat` (8.6 GB) — 12-feature training data (Figs 3, 4_relerr, 5_rmse).
- COSMIC-1 `ionPrf` archive on scratch (`download_cdaac.sh`) and
  `Projects/ne15/test_arrays.npz` (138 MB) — full 15-feature test set.
- Rebuild the 15-feature model + full test set with:
  `python Src/reproduce_ne_15feat.py --list <cosmic_list_local.txt> --n-profiles 40000 --device cuda --out Projects/ne15`

## Notes

- Figs 4–5 (`*_repro`) reproduce the manuscript's conclusions: relative error
  larger at high latitude, RMSE peaks at the equator (Nₑ magnitude), error
  decreases with increasing NmF2. The 12-feature `fig4_relerr/fig5_rmse` are a
  weaker model (no NmF2/hmF2/VSH features) shown only for contrast.
- See `../MIGRATION_REPORT.md` §9 for the full reproduction account.
