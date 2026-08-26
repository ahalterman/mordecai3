#!/usr/bin/env bash
# Fetch the geoBoundaries CGAZ global composites (~1.25 GB) that
# build_boundaries.py turns into the console's 85 MB SQLite store.
#
# CGAZ ("Comprehensive Global Administrative Zones") is the composite release:
# one file per admin level covering every country, clipped to a consistent
# international boundary set. That is why this fetches three files rather than
# walking the per-country API -- and why the console needs no network at all
# once the store is built.
#
# Source:  https://www.geoboundaries.org/
# Licence: CC-BY 4.0 (gbOpen). Attribution is required; the console carries it
#          in the RESOLVE panel and in the GeoJSON export properties.
set -euo pipefail

DEST="${1:-$(cd "$(dirname "$0")/.." && pwd)/data/geoboundaries}"
BASE="https://github.com/wmgeolab/geoBoundaries/raw/main/releaseData/CGAZ"

mkdir -p "$DEST"
for LEVEL in ADM0 ADM1 ADM2; do
  OUT="$DEST/geoBoundariesCGAZ_${LEVEL}.geojson"
  if [ -s "$OUT" ]; then
    echo "have  $LEVEL  ($(du -h "$OUT" | cut -f1))"
    continue
  fi
  echo "fetch $LEVEL ..."
  curl -fL --retry 3 --progress-bar -o "$OUT" "$BASE/geoBoundariesCGAZ_${LEVEL}.geojson"
  echo "done  $LEVEL  ($(du -h "$OUT" | cut -f1))"
done

echo
echo "Now build the store:"
echo "  python console/build_boundaries.py"
