"""Turn the geoBoundaries CGAZ bulk GeoJSON into a queryable SQLite store.

The three CGAZ composites (`geoBoundariesCGAZ_ADM0/1/2.geojson`, ~1.25 GB
together) are single JSON objects with one giant `features` array, so they are
streamed with ijson rather than loaded -- ADM2 alone would be several GB as
Python objects.

Two things happen on the way in:

1. **Simplification.** CGAZ is already a simplified composite, but individual
   ADM2 polygons still run to thousands of vertices, which is far more than a
   web map at province scale can draw. Douglas-Peucker at the per-level
   tolerances below cuts the store to a few percent of the source size with no
   visible difference above ~1 km.
2. **Bounding boxes.** Stored per shape, from the *unsimplified* geometry, so
   the query path can reject almost every candidate polygon with an index
   lookup before it ever parses geometry.

The lookup key is deliberately not a name. geoBoundaries carries only
`shapeGroup` (ISO3) and `shapeName`; GeoNames carries admin codes and a
coordinate. Joining on names means fighting transliteration and "Province of
X" forever, so `boundaries.py` joins on the coordinate instead -- which is why
`name_norm` here is a fallback column, not the primary index.

Usage:
    python console/build_boundaries.py            # data/geoboundaries -> boundaries.sqlite
    python console/build_boundaries.py --levels 0 1
"""

import argparse
import json
import sqlite3
import sys
import time
import unicodedata
from pathlib import Path

import ijson
from shapely.geometry import shape
from shapely.ops import transform

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = REPO_ROOT / "data" / "geoboundaries"
DEFAULT_DB = REPO_ROOT / "data" / "geoboundaries" / "boundaries.sqlite"

# Douglas-Peucker tolerance in degrees, per admin level. Countries are drawn at
# the smallest scale so they can lose the most detail; ADM2 units can be a few
# km across, where 0.01 deg would collapse them into slivers.
TOLERANCE_DEG = {0: 0.012, 1: 0.008, 2: 0.004}

# Coordinates are stored at this many decimal places. Six is ~10 cm, which is
# pointless for a polygon simplified to ~1 km; four is ~11 m and roughly halves
# the JSON.
COORD_PRECISION = 4

SCHEMA = """
CREATE TABLE IF NOT EXISTS shapes (
    id        INTEGER PRIMARY KEY,
    level     INTEGER NOT NULL,
    iso3      TEXT    NOT NULL,
    name      TEXT    NOT NULL,
    name_norm TEXT    NOT NULL,
    shape_id  TEXT,
    minx REAL, miny REAL, maxx REAL, maxy REAL,
    geom      TEXT    NOT NULL
);
-- The bbox prefilter runs as (level, iso3) then a range scan, so the composite
-- index below is the one that matters; the name index only backs the fallback.
CREATE INDEX IF NOT EXISTS ix_shapes_level_iso  ON shapes(level, iso3);
CREATE INDEX IF NOT EXISTS ix_shapes_level_name ON shapes(level, name_norm);

CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


def normalize_name(name: str) -> str:
    """Casefold and strip diacritics, for the name-match fallback.

    "Tillabéri" and "Tillaberi" are the same province; GeoNames and
    geoBoundaries do not agree on which spelling to use.
    """
    decomposed = unicodedata.normalize("NFKD", name)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return " ".join(stripped.casefold().split())


def _round_coords(geom):
    """Round every coordinate to COORD_PRECISION.

    shapely's `transform` walks the coordinate sequences for us, so this works
    uniformly over Polygon and MultiPolygon without unpacking the rings by
    hand.
    """
    return transform(
        lambda x, y, z=None: (round(x, COORD_PRECISION), round(y, COORD_PRECISION)),
        geom,
    )


def build_level(conn, source: Path, level: int) -> int:
    """Stream one CGAZ file into the store. Returns the number of shapes written."""
    path = source / f"geoBoundariesCGAZ_ADM{level}.geojson"
    if not path.exists():
        raise SystemExit(f"missing {path} -- run console/fetch_boundaries.sh first")

    tolerance = TOLERANCE_DEG[level]
    written = 0
    skipped = 0
    started = time.time()
    rows = []

    with open(path, "rb") as fh:
        # use_float=True: ijson hands back Decimal otherwise, which shapely
        # accepts but json.dumps does not.
        for feat in ijson.items(fh, "features.item", use_float=True):
            props = feat.get("properties") or {}
            iso3 = (props.get("shapeGroup") or "").strip().upper()
            name = (props.get("shapeName") or "").strip()
            geom_json = feat.get("geometry")
            if not iso3 or not geom_json:
                skipped += 1
                continue

            geom = shape(geom_json)
            # The bbox comes off the full-detail geometry: it is the query
            # prefilter, so it must never be tighter than the real extent.
            minx, miny, maxx, maxy = geom.bounds

            simplified = geom.simplify(tolerance, preserve_topology=True)
            # preserve_topology can still hand back an empty geometry for a
            # sliver smaller than the tolerance. Keep the original in that case
            # -- a tiny polygon costs nothing and a missing one is a hole in
            # the map.
            if simplified.is_empty:
                simplified = geom
            simplified = _round_coords(simplified)

            rows.append((
                level, iso3, name, normalize_name(name),
                props.get("shapeID"),
                minx, miny, maxx, maxy,
                json.dumps(simplified.__geo_interface__, separators=(",", ":")),
            ))
            written += 1

            if len(rows) >= 500:
                _flush(conn, rows)
                rows.clear()
                print(f"  ADM{level}: {written:>6} shapes "
                      f"({time.time() - started:.0f}s)", end="\r", flush=True)

    if rows:
        _flush(conn, rows)
    print(f"  ADM{level}: {written} shapes, {skipped} skipped "
          f"({time.time() - started:.0f}s)                  ")
    return written


def _flush(conn, rows):
    conn.executemany(
        "INSERT INTO shapes (level, iso3, name, name_norm, shape_id,"
        "                    minx, miny, maxx, maxy, geom)"
        " VALUES (?,?,?,?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                    help="directory holding the geoBoundariesCGAZ_ADM*.geojson files")
    ap.add_argument("--db", type=Path, default=DEFAULT_DB,
                    help="SQLite file to write")
    ap.add_argument("--levels", type=int, nargs="+", default=[0, 1, 2],
                    choices=[0, 1, 2])
    args = ap.parse_args(argv)

    args.db.parent.mkdir(parents=True, exist_ok=True)
    if args.db.exists():
        args.db.unlink()

    conn = sqlite3.connect(args.db)
    conn.executescript(SCHEMA)

    total = 0
    for level in args.levels:
        total += build_level(conn, args.source, level)

    conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?,?)",
                 ("source", "geoBoundaries CGAZ (gbOpen, CC-BY 4.0)"))
    conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?,?)",
                 ("levels", ",".join(str(l) for l in args.levels)))
    conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?,?)",
                 ("tolerance_deg", json.dumps(
                     {str(l): TOLERANCE_DEG[l] for l in args.levels})))
    conn.commit()
    conn.execute("VACUUM")
    conn.close()

    size_mb = args.db.stat().st_size / 1e6
    print(f"\n{total} shapes -> {args.db} ({size_mb:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
