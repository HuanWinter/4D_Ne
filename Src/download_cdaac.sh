#!/usr/bin/env bash
# Download COSMIC-1 (repro2013) Level-2 ionPrf profiles from the COSMIC open
# data portal and stage them for Src/make_delay_files.py.
#
# Source (no login required):
#   https://data.cosmic.ucar.edu/gnss-ro/cosmic1/repro2013/level2/<year>/<doy>/
#       ionPrf_repro2013_<year>_<doy>.tar.gz
#
# Usage:
#   bash Src/download_cdaac.sh                              # full set from cosmic_list.txt
#   DAYS="2010.182 2010.183" bash Src/download_cdaac.sh     # just these year.doy days
#   PARALLEL=8 STAGE=/glade/derecho/scratch/$USER/cosmic bash Src/download_cdaac.sh
#
# It downloads one daily tar per year.doy, extracts it into
# <STAGE>/<year>/<doy>/, and (on completion) writes
# <STAGE>/cosmic_list_local.txt listing the staged profile files for use as the
# --cosmic-list argument to make_delay_files.py.
#
# NOTE: the repro2013 tars store files with a trailing "_nc" (not ".nc"); they
# are still netCDF and read fine. The local list matches "ionPrf_*".
set -euo pipefail

MISSION="${MISSION:-cosmic1}"
REPRO="${REPRO:-repro2013}"
LEVEL="${LEVEL:-level2}"
FILETYPE="${FILETYPE:-ionPrf}"
LIST="${LIST:-cosmic_list.txt}"
STAGE="${STAGE:-/glade/derecho/scratch/$USER/cosmic}"
PARALLEL="${PARALLEL:-6}"
BASE="https://data.cosmic.ucar.edu/gnss-ro/${MISSION}/${REPRO}/${LEVEL}"

mkdir -p "$STAGE"

# --- Build the list of year.doy days to fetch ------------------------------
if [[ -n "${DAYS:-}" ]]; then
    printf '%s\n' $DAYS > "$STAGE/_days.txt"
elif [[ -f "$LIST" ]]; then
    # cosmic_list.txt paths contain a ".../<year>.<doy>/..." component.
    awk -F/ '{print $(NF-1)}' "$LIST" | sort -u > "$STAGE/_days.txt"
else
    echo "ERROR: no DAYS set and list '$LIST' not found" >&2
    exit 1
fi
NDAYS=$(wc -l < "$STAGE/_days.txt")
echo "Mission=$MISSION/$REPRO/$LEVEL Filetype=$FILETYPE  days=$NDAYS  stage=$STAGE  parallel=$PARALLEL"

# --- Per-day fetch + extract (idempotent / resumable) ----------------------
fetch_day() {
    local day="$1"                 # e.g. 2010.182
    local year="${day%.*}"
    local doy="${day#*.}"
    local dest="$STAGE/$year/$doy"
    local marker="$dest/.done"
    [[ -f "$marker" ]] && return 0  # already extracted

    local tar="$dest/${FILETYPE}_${REPRO}_${year}_${doy}.tar.gz"
    local url="$BASE/$year/$doy/${FILETYPE}_${REPRO}_${year}_${doy}.tar.gz"
    mkdir -p "$dest"
    if ! curl -fsS -C - -m 600 "$url" -o "$tar"; then
        echo "FAIL download $day ($url)" >&2
        return 1
    fi
    if ! tar tzf "$tar" >/dev/null 2>&1; then
        echo "FAIL not-a-tar $day" >&2
        rm -f "$tar"
        return 1
    fi
    tar xzf "$tar" -C "$dest"
    rm -f "$tar"        # save space; comment out to keep tarballs
    touch "$marker"
    echo "ok $day"
}
export -f fetch_day
export MISSION REPRO LEVEL FILETYPE STAGE BASE

set +e
cat "$STAGE/_days.txt" | xargs -P "$PARALLEL" -I{} bash -c 'fetch_day "$@"' _ {} \
    | tee "$STAGE/_download.log"
set -e

# --- Rebuild a local cosmic_list pointing at the staged files --------------
LOCAL_LIST="$STAGE/cosmic_list_local.txt"
find "$STAGE" -name "${FILETYPE}_*" -type f ! -name "*.tar.gz" | sort > "$LOCAL_LIST"
echo "Staged $(wc -l < "$LOCAL_LIST") $FILETYPE files."
echo "Next:"
echo "  python Src/make_delay_files.py --cosmic-list $LOCAL_LIST --data-dir Data/Delay --lags=-8:7"
