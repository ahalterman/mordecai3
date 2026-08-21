"""Ad-hoc read-only probes of the geonames index for the census write-up.

    python probe.py ids 4140963 4138106
    python probe.py name "D.C." --size 12
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/home/andy/projects/mordecai3")
import es_util  # noqa: E402
from mordecai3.geonames import _clean_search_name  # noqa: E402


def row(s, extra=""):
    return "%-10s %-32s %-6s %-4s %-12s pop=%-10s alt=%s %s" % (
        s.get("geonameid"), s.get("name")[:32], s.get("feature_code"),
        s.get("country_code3"), s.get("coordinates"), s.get("population"),
        s.get("alt_name_length"), extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["ids", "name"])
    ap.add_argument("args", nargs="+")
    ap.add_argument("--size", type=int, default=100)
    ap.add_argument("--show", type=int, default=15)
    a = ap.parse_args()
    if a.mode == "ids":
        got = es_util.mget(a.args)
        for g in a.args:
            s = got.get(str(g))
            print(row(s) if s else "%-10s NOT IN INDEX" % g)
    else:
        for name in a.args:
            cleaned = _clean_search_name(name)
            hits = es_util.phrase_search(cleaned, size=a.size)
            print("== %r -> cleaned %r : %d hits (window %d)"
                  % (name, cleaned, len(hits), a.size))
            for i, h in enumerate(hits[:a.show]):
                print("  %3d %s" % (i, row(h)))


if __name__ == "__main__":
    main()
