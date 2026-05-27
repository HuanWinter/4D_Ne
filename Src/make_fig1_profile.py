# Manuscript Fig. 1: topside variable definitions.
# Upper: a sample COSMIC-1 Ne profile with NmF2 and hmF2 marked.
# Lower: vertical scale height (VSH) = altitude span for Ne to drop by 1/e
#        starting 20 km above hmF2 (ln(Ne) ~ linear topside, slope = -1/H).
# Default profile: C06, 07:40 UT, 16 Nov 2008 (DoY 319), GPS-28.
#   python Src/make_fig1_profile.py --file <ionPrf_...nc> --save fig1_Ne_profile.png

import argparse
import os
import numpy as np


def compute_vsh(alt, ne, hmf2):
    order = np.argsort(alt)
    alt, ne = alt[order], ne[order]
    idx = np.where((alt > hmf2 + 20) & (ne > 0))[0]
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
    slope = intercept = np.nan
    if (i_top - i_sta) >= 2:
        aseg, lseg = alt[seg], np.log(ne[seg])
        ok = np.isfinite(aseg) & np.isfinite(lseg)
        if ok.sum() >= 2 and np.ptp(aseg[ok]) > 0:
            try:
                slope, intercept = np.polyfit(aseg[ok], lseg[ok], 1)
            except np.linalg.LinAlgError:
                pass
    return vsh, (alt, ne, i_sta, i_top), (alt[seg], ne[seg]), slope, intercept


def main(argv=None):
    default = ("/glade/derecho/scratch/%s/cosmic/2008/319/"
               "ionPrf_C006.2008.319.07.40.G28_2013.3520_nc"
               % os.environ.get("USER", "andonghu"))
    ap = argparse.ArgumentParser(description="manuscript Fig. 1")
    ap.add_argument("--file", default=default)
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
    print("NmF2=%.3g el/cm^3  hmF2=%.1f km  VSH=%.1f km  (lat=%.1f, lon=%.1f)"
          % (nmf2, hmf2, vsh, lat, lon))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 10))

    ax1.plot(ne / 1e5, alt, "b-", lw=1.5)
    ax1.plot(nmf2 / 1e5, hmf2, "ro", ms=9)
    ax1.axhline(hmf2, color="r", ls="--", lw=0.8)
    ax1.axvline(nmf2 / 1e5, color="r", ls="--", lw=0.8)
    ax1.annotate("NmF2 = %.2f$\\times10^5$" % (nmf2 / 1e5), (nmf2 / 1e5, hmf2),
                 xytext=(nmf2 / 1e5 * 0.45, hmf2 + 90),
                 arrowprops=dict(arrowstyle="->"), fontsize=11)
    ax1.annotate("hmF2 = %.0f km" % hmf2, (nmf2 / 1e5 * 0.05, hmf2),
                 xytext=(nmf2 / 1e5 * 0.05, hmf2 - 70), fontsize=11)
    ax1.set_xlabel("$N_e$  ($\\times10^5$ el cm$^{-3}$)")
    ax1.set_ylabel("Altitude (km)")
    ax1.set_title("Upper: $N_e$ profile (COSMIC-1 C06, 07:40 UT, DoY 319, 2008, GPS-28)")
    ax1.grid(alpha=0.3)

    if prof is not None:
        a, n, i_sta, i_top = prof
        ax2.plot(np.log(n), a, "b.", ms=3, alpha=0.5, label="ln($N_e$)")
        if np.isfinite(slope):
            yfit = np.array([a[i_sta], a[i_top]])
            ax2.plot(intercept + slope * yfit, yfit, "g-", lw=2, label="linear fit (slope=-1/H)")
        for ai in (a[i_sta], a[i_top]):
            ax2.axhline(ai, color="r", ls="--", lw=0.8)
        ax2.annotate("", xy=(np.log(n[i_top]), a[i_top]),
                     xytext=(np.log(n[i_top]), a[i_sta]),
                     arrowprops=dict(arrowstyle="<->", color="k"))
        ax2.text(np.log(n[i_top]) + 0.1, (a[i_sta] + a[i_top]) / 2,
                 "VSH = %.0f km\n(1/e drop)" % vsh, fontsize=11, va="center")
        ax2.set_ylim(hmf2, a[i_top] + 60)
        ax2.legend(loc="upper right")
    ax2.set_xlabel("ln($N_e$)")
    ax2.set_ylabel("Altitude (km)")
    ax2.set_title("Lower: vertical scale height (VSH) from topside ln($N_e$) fit")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.save, dpi=150)
    print("saved ->", args.save)


if __name__ == "__main__":
    main()
