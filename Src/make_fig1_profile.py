#!/usr/bin/env python3
"""Reproduce manuscript Fig. 1: the topside-ionosphere variable definitions.

Upper panel : a sample COSMIC-1 electron-density profile Ne(alt), marking the
              peak density NmF2 and its height hmF2.
Lower panel : the vertical scale height (VSH) from the topside, where ln(Ne) is
              ~linear in altitude (slope = -1/H); VSH is the altitude span over
              which Ne falls by 1/e, starting 20 km above hmF2.

VSH definition reproduced from main-CPU-short.ipynb (cell 8):
    ind_sta = first altitude > hmF2 + 20 km
    ind_top = first topside altitude where ln(Ne) - ln(Ne[ind_sta]) < -1
    VSH     = alt[ind_top] - alt[ind_sta]

Default profile is the one in the paper's Fig. 1 caption:
  COSMIC-1 C06, 07:40 UT, 16 Nov 2008 (DoY 319), GPS-28.

Usage
-----
  python Src/make_fig1_profile.py \
      --file /glade/derecho/scratch/$USER/cosmic/2008/319/ionPrf_C006.2008.319.07.40.G28_2013.3520_nc \
      --save fig1_Ne_profile.png
"""
from __future__ import annotations
import argparse, os
import numpy as np


def compute_vsh(alt, ne, hmf2):
    """VSH = altitude span for Ne to drop by 1/e, starting 20 km above hmF2.

    Returns (vsh_km, i_sta, i_top, fit_slope, fit_intercept) where the fit is
    ln(Ne) = slope*alt + intercept over [alt_sta, alt_top] (slope = -1/H)."""
    order = np.argsort(alt)
    alt, ne = alt[order], ne[order]
    top = (alt > hmf2 + 20) & (ne > 0)
    idx = np.where(top)[0]
    if idx.size < 3:
        return np.nan, None, None, np.nan, np.nan
    i_sta = idx[0]
    ln0 = np.log(ne[i_sta])
    drop = np.where((np.log(np.clip(ne, 1e-30, None)) - ln0 < -1) &
                    (np.arange(ne.size) > i_sta))[0]
    if drop.size == 0:
        return np.nan, alt, ne, np.nan, np.nan
    i_top = drop[0]
    vsh = alt[i_top] - alt[i_sta]
    seg = slice(i_sta, i_top + 1)
    if (i_top - i_sta) >= 2:
        slope, intercept = np.polyfit(alt[seg], np.log(ne[seg]), 1)
    else:
        slope = intercept = np.nan
    return vsh, (alt, ne, i_sta, i_top), (alt[seg], ne[seg]), slope, intercept


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", default="/glade/derecho/scratch/{user}/cosmic/2008/319/"
                    "ionPrf_C006.2008.319.07.40.G28_2013.3520_nc".format(user=os.environ.get("USER", "andonghu")))
    ap.add_argument("--save", default="fig1_Ne_profile.png")
    args = ap.parse_args(argv)

    import netCDF4 as nc
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = nc.Dataset(args.file)
    alt = np.asarray(f.variables["MSL_alt"][:], float)
    ne = np.asarray(f.variables["ELEC_dens"][:], float)
    nmf2 = float(f.getncattr("edmax")); hmf2 = float(f.getncattr("edmaxalt"))
    lat = float(f.getncattr("edmaxlat")); lon = float(f.getncattr("edmaxlon"))
    f.close()
    good = np.isfinite(alt) & np.isfinite(ne) & (ne > 0)
    alt, ne = alt[good], ne[good]

    vsh, prof, seg, slope, intercept = compute_vsh(alt, ne, hmf2)
    print(f"NmF2={nmf2:.3g} el/cm^3  hmF2={hmf2:.1f} km  VSH={vsh:.1f} km  "
          f"(lat={lat:.1f}, lon={lon:.1f})")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 10))

    # Upper: Ne profile with NmF2 / hmF2
    ax1.plot(ne / 1e5, alt, "b-", lw=1.5)
    ax1.plot(nmf2 / 1e5, hmf2, "ro", ms=9)
    ax1.axhline(hmf2, color="r", ls="--", lw=0.8)
    ax1.axvline(nmf2 / 1e5, color="r", ls="--", lw=0.8)
    ax1.annotate(f"NmF2 = {nmf2/1e5:.2f}$\\times10^5$", (nmf2/1e5, hmf2),
                 xytext=(nmf2/1e5*0.45, hmf2+90),
                 arrowprops=dict(arrowstyle="->"), fontsize=11)
    ax1.annotate(f"hmF2 = {hmf2:.0f} km", (nmf2/1e5*0.05, hmf2),
                 xytext=(nmf2/1e5*0.05, hmf2-70), fontsize=11)
    ax1.set_xlabel("$N_e$  ($\\times10^5$ el cm$^{-3}$)")
    ax1.set_ylabel("Altitude (km)")
    ax1.set_title("Upper: $N_e$ profile (COSMIC-1 C06, 07:40 UT, DoY 319, 2008, GPS-28)")
    ax1.grid(alpha=0.3)

    # Lower: topside ln(Ne) linear fit + VSH span
    if prof is not None:
        a, n, i_sta, i_top = prof
        ax2.plot(np.log(n), a, "b.", ms=3, alpha=0.5, label="ln($N_e$)")
        if np.isfinite(slope):
            yfit = np.array([a[i_sta], a[i_top]])
            ax2.plot(intercept + slope * yfit, yfit, "g-", lw=2,
                     label=f"linear fit (slope=-1/H)")
        for ai in (a[i_sta], a[i_top]):
            ax2.axhline(ai, color="r", ls="--", lw=0.8)
        ax2.annotate("", xy=(np.log(n[i_top]), a[i_top]),
                     xytext=(np.log(n[i_top]), a[i_sta]),
                     arrowprops=dict(arrowstyle="<->", color="k"))
        ax2.text(np.log(n[i_top]) + 0.1, (a[i_sta] + a[i_top]) / 2,
                 f"VSH = {vsh:.0f} km\n(1/e drop)", fontsize=11, va="center")
        ax2.set_ylim(hmf2, a[i_top] + 60)
        ax2.legend(loc="upper right")
    ax2.set_xlabel("ln($N_e$)")
    ax2.set_ylabel("Altitude (km)")
    ax2.set_title("Lower: vertical scale height (VSH) from topside ln($N_e$) fit")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.save, dpi=150)
    print(f"saved -> {args.save}")


if __name__ == "__main__":
    main()
