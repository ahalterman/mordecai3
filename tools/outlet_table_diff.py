"""Compare the curated outlet table against the independently researched one.

e50's one uncloseable risk was that 55.3% of its newsroom homes came from a
language model's knowledge of newspaper mastheads, which cannot be proven free
of the LGL corpus itself. The fix is to rebuild the table from public sources
that are demonstrably not this corpus -- Wikipedia, the Library of Congress
newspaper catalogue, archive.org snapshots of the outlets' own contact pages --
and check whether the arm's gain survives the swap.

This script does the comparison and weights it by what it costs: a domain that
never appears in LGL's held-out half cannot move the held-out numbers no matter
how wrong it is, so disagreements are reported with their held-out entity
counts.

    uv run python tools/outlet_table_diff.py
"""

import csv
import os
import pickle
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from outlet_align import entity_domains  # noqa: E402
from outlet_home_table import HOME  # noqa: E402

TSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                   "outlet_homes_researched.tsv")
PICKLES = os.path.join("raw_data", "pickled_es")


def load_researched(path=TSV):
    """domain -> (place|None, admin1|None, iso3|None, scope, confidence, source)."""
    out = {}
    with open(path, encoding="utf8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            row = line.rstrip("\n").split("\t")
            domain, place, admin1, iso3, scope, conf, source = (row + [""] * 7)[:7]
            place = None if place.strip().upper() in ("", "NONE") else place.strip()
            out[domain.strip()] = (place, admin1.strip() or None,
                                   iso3.strip() or None, scope.strip(),
                                   conf.strip(), source.strip())
    return out


def as_curated(entry):
    """The curated table's row in the researched table's shape."""
    if entry is None:
        return (None, None, None, "unresolved")
    place, admin1, iso3, scope = entry
    if scope == "national":
        place = None                     # curated convention: country-level only
    return (place, admin1, iso3, scope)


def held_out_counts():
    counts = Counter()
    for source in ("lgl", "tr"):
        path = os.path.join(
            PICKLES, "es_formatted_{}_500_all_loc_types_fuzzy_0.pkl".format(source))
        with open(path, "rb") as f:
            data = pickle.load(f)
        data = [i for i in data if len(i["tensor"]) > 1]
        domains = entity_domains(data, source, "raw_data")
        split = round(0.7 * len(data))
        for i in range(split, len(data)):
            if domains.get(i):
                counts[domains[i]] += 1
        del data
    return counts


def main():
    researched = load_researched()
    counts = held_out_counts()

    agree = []
    rows = []
    for domain in sorted(HOME):
        cur = as_curated(HOME.get(domain))
        res = researched.get(domain)
        if res is None:
            rows.append((domain, cur, ("MISSING", None, None, ""), "", "", 0))
            continue
        r_place, r_a1, r_iso, r_scope, r_conf, r_src = res
        same_place = (cur[0] or "").lower() == (r_place or "").lower()
        same_a1 = (cur[1] or "").lower() == (r_a1 or "").lower()
        n = counts.get(domain, 0)
        if same_place and (same_a1 or not cur[0]):
            agree.append(domain)
        else:
            rows.append((domain, cur, (r_place, r_a1, r_iso, r_scope), r_conf,
                         r_src, n))

    n_total = len(HOME)
    print("curated rows: {}   researched rows: {}".format(n_total, len(researched)))
    print("AGREE on place+admin1: {} ({:.1%})".format(
        len(agree), len(agree) / n_total))
    print("DISAGREE or newly resolved: {}\n".format(len(rows)))

    moved = sum(n for *_x, n in rows)
    print("held-out entities behind a disagreeing domain: {}\n".format(moved))

    print("{:<40} {:<26} {:<26} {:>5} {}".format(
        "domain", "curated", "researched", "n_ho", "conf"))
    print("-" * 118)
    for domain, cur, res, conf, _src, n in sorted(rows, key=lambda r: -r[5]):
        cs = "{}, {}".format(cur[0], cur[1]) if cur[0] else \
             ("UNRESOLVED" if cur[3] == "unresolved" else "country-only " + str(cur[2]))
        rs = "{}, {}".format(res[0], res[1]) if res[0] else \
             ("MISSING" if res[0] == "MISSING" else "country-only " + str(res[2]))
        print("{:<40} {:<26} {:<26} {:>5} {}".format(
            domain[:40], cs[:26], rs[:26], n, conf))

    print("\nsources: every researched row carries a URL; see {}".format(TSV))


if __name__ == "__main__":
    main()
