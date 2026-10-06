"""Calibrate and report the GeoNames -> geoBoundaries join.

`boundaries.py` joins the two gazetteers geometrically -- which polygon
contains the coordinate -- and then confirms the hit by comparing names, so
that a structural disagreement between the sources produces "no boundary"
rather than a confidently drawn wrong one. That confirmation needs a
similarity threshold, and a threshold picked from two examples is a guess.

This script measures it. It walks every ADM1 and ADM2 record in the GeoNames
index, runs the point-in-polygon step, and reports the distribution of name
similarity between what GeoNames calls the unit and what geoBoundaries calls
the polygon the unit's coordinate lands in.

The distribution is strongly bimodal, and that is the whole argument: pairs
that are the same unit spelled differently sit near 1.0, pairs that are a unit
and its *parent or neighbour* sit far below, and there is a wide empty valley
between them. The threshold goes in the valley, which is why its exact value
does not matter much -- a fact worth knowing before trusting it.

Usage:
    python console/validate_boundaries.py            # both levels
    python console/validate_boundaries.py --level 1 --examples 25
"""

import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan

from boundaries import (NAME_AGREE_THRESHOLD, BoundaryStore, _name_core,
                        name_similarity)

CANDIDATE_THRESHOLDS = [0.0, 0.70, 0.80, 0.84, 0.86, 0.88, 0.90, 0.95, 1.0]


def gazetteer_units(es, level, limit=None):
    """Every GeoNames record for administrative units at `level`."""
    code = f"ADM{level}"
    query = {"query": {"term": {"feature_code": code}},
             "_source": ["geonameid", "name", "asciiname", "country_code3",
                         "admin1_name", "admin2_name", "coordinates",
                         "feature_code"]}
    n = 0
    for hit in scan(es, index="geonames", query=query, size=1000,
                    preserve_order=False):
        src = hit["_source"]
        lat, lon = src["coordinates"].split(",")
        yield {"geonameid": src["geonameid"],
               "name": src["name"],
               "country_code3": src.get("country_code3") or "",
               "admin1_name": src.get("admin1_name") or "",
               "admin2_name": src.get("admin2_name") or "",
               "feature_code": src["feature_code"],
               "lat": float(lat), "lon": float(lon)}
        n += 1
        if limit and n >= limit:
            return


def run_level(store, es, level, limit, n_examples):
    """Score every unit at one level. Returns (rows, counters)."""
    rows = []
    counts = collections.Counter()
    for unit in gazetteer_units(es, level, limit):
        counts["units"] += 1
        iso3 = unit["country_code3"].upper()
        if not iso3:
            counts["no_country"] += 1
            continue
        hit = store._containing(level, iso3, unit["lon"], unit["lat"])
        if hit is None:
            counts["no_containing_polygon"] += 1
            continue
        counts["pip_hit"] += 1
        shape_core = _name_core(hit[0])
        # The best of the record's own name and the admin name GeoNames files
        # it under -- exactly what `_names_agree` compares.
        own = [unit["name"],
               unit["admin2_name"] if level == 2 else unit["admin1_name"]]
        sim = max((name_similarity(_name_core(c), shape_core)
                   for c in own if c), default=0.0)
        rows.append({"level": level, "sim": sim, "geonames": unit["name"],
                     "cgaz": hit[0], "iso3": iso3})
    return rows, counts


def histogram(sims, width=54):
    """A coarse text histogram, which is all the shape of this needs."""
    edges = [0.0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.84, 0.88, 0.92, 0.96, 1.0001]
    buckets = [0] * (len(edges) - 1)
    for s in sims:
        for i in range(len(edges) - 1):
            if edges[i] <= s < edges[i + 1]:
                buckets[i] += 1
                break
    peak = max(buckets) or 1
    print(f"\n  name similarity, n={len(sims)}")
    for i, count in enumerate(buckets):
        bar = "#" * round(width * count / peak)
        label = f"{edges[i]:.2f}-{edges[i+1] if i < len(edges)-2 else 1.0:.2f}"
        print(f"    {label:>11}  {count:>6}  {bar}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2],
                    choices=[1, 2])
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after this many units per level (for a quick look)")
    ap.add_argument("--examples", type=int, default=15,
                    help="how many below-threshold pairs to print")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=9200)
    args = ap.parse_args(argv)

    store = BoundaryStore()
    if not store.available:
        raise SystemExit("no boundary store -- run console/build_boundaries.py")
    es = Elasticsearch(hosts=[args.host], port=args.port)

    all_rows = []
    for level in args.levels:
        print(f"\n=== ADM{level} " + "=" * 58)
        rows, counts = run_level(store, es, level, args.limit, args.examples)
        all_rows.extend(rows)
        total = counts["units"]
        print(f"  GeoNames ADM{level} records      {total:>7}")
        print(f"  no country code               {counts['no_country']:>7}")
        print(f"  no containing CGAZ polygon    {counts['no_containing_polygon']:>7}"
              f"   ({counts['no_containing_polygon']/max(total,1):.1%})")
        print(f"  point-in-polygon hit          {counts['pip_hit']:>7}"
              f"   ({counts['pip_hit']/max(total,1):.1%})")
        histogram([r["sim"] for r in rows])

        print(f"\n  coverage by threshold (share of all {total} records that"
              f" would get a boundary):")
        for t in CANDIDATE_THRESHOLDS:
            kept = sum(1 for r in rows if r["sim"] >= t)
            mark = "  <- current" if abs(t - NAME_AGREE_THRESHOLD) < 1e-9 else ""
            print(f"    >= {t:.2f}   {kept:>7}   {kept/max(total,1):>6.1%}{mark}")

        rejected = sorted((r for r in rows if r["sim"] < NAME_AGREE_THRESHOLD),
                          key=lambda r: -r["sim"])
        print(f"\n  rejected at {NAME_AGREE_THRESHOLD} "
              f"({len(rejected)} of {counts['pip_hit']} PIP hits) -- "
              f"the {min(args.examples, len(rejected))} closest calls:")
        for r in rejected[:args.examples]:
            print(f"    {r['sim']:.3f}  {r['iso3']}  "
                  f"GeoNames {r['geonames']!r:<34} vs CGAZ {r['cgaz']!r}")

        accepted_low = sorted((r for r in rows
                               if NAME_AGREE_THRESHOLD <= r["sim"] < 1.0),
                              key=lambda r: r["sim"])
        print(f"\n  accepted with a non-identical name -- the "
              f"{min(args.examples, len(accepted_low))} weakest:")
        for r in accepted_low[:args.examples]:
            print(f"    {r['sim']:.3f}  {r['iso3']}  "
                  f"GeoNames {r['geonames']!r:<34} vs CGAZ {r['cgaz']!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
