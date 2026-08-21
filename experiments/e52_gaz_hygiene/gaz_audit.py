"""Model-free audit of the gazetteer itself, over the held-out candidate sets.

Four questions the census raised, answered by counting rather than by anecdote:

  A. How many held-out golds point at a geonameid the live index does not have?
     (`Ireland` 2646052, `Slavonia` 3205300 -- a broken gold, or a row the
     index build dropped.)
  B. How much defunct-row (*H / PPLQ) mass is there, how often is a defunct row
     the top-ranked candidate, and -- the guard that matters -- how often is
     the GOLD itself a defunct row?  A *H demotion rule that deletes golds is
     worse than no rule.
  C. How many entities carry a co-located same-name duplicate pair, and how
     often does the gold sit on each side?  That is the A/P convention split,
     and it decides whether dedupe can be anything but a coin flip.
  D. Unretrievable golds: how many, and how many are state abbreviations.

    python gaz_audit.py --sources TR,LGL,GWN
"""
import argparse
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/home/andy/projects/mordecai3"
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import es_util  # noqa: E402
import hygiene  # noqa: E402
import rebuild  # noqa: E402
from twin_credit_eval import SOURCES, load_val  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="TR,LGL,GWN")
    a = ap.parse_args()
    want = a.sources.split(",")

    tot = Counter()
    missing_golds = Counter()
    hist_gold_examples = []
    dup_examples = []
    unretr_examples = []
    for source, stems in SOURCES:
        if source not in want:
            continue
        es_data = load_val(source, stems, os.path.join(ROOT, "raw_data"),
                           "_enriched", 500, "all_loc_types", 0)
        golds = {str(e.get("correct_geonamesid")) for e in es_data}
        found = es_util.mget(sorted(g for g in golds if g and g != "None"))
        for e in es_data:
            g = str(e.get("correct_geonamesid"))
            real = rebuild.real_choices(e)
            ids = [str(c["geonameid"]) for c in real]
            gi = ids.index(g) if g in ids else None
            tot["entities"] += 1

            # A. gold row absent from the live index
            if g not in found:
                tot["gold_not_in_index"] += 1
                missing_golds[(source, e.get("search_name"), g)] += 1

            # B. defunct rows
            hist = [i for i, c in enumerate(real)
                    if hygiene.HIST_CODES.match(str(c.get("feature_code", "")))]
            if hist:
                tot["has_hist_candidate"] += 1
            if hist and min(hist) < 5:
                tot["hist_in_top5"] += 1
            if gi is not None and gi in hist:
                tot["gold_is_hist"] += 1
                if len(hist_gold_examples) < 25:
                    hist_gold_examples.append(
                        (source, e.get("search_name"), g,
                         real[gi].get("feature_code")))
            drops_h = hygiene.hygiene_drops(e, ["demote_h"])
            if drops_h:
                tot["demote_h_fires"] += 1
                tot["demote_h_rows"] += len(drops_h)
                if gi is not None and gi in drops_h:
                    tot["demote_h_DELETES_GOLD"] += 1

            # C. duplicate pairs
            drops_d = hygiene.dedupe_rows(real, keep="p")
            if drops_d:
                tot["dedupe_fires"] += 1
                tot["dedupe_rows"] += len(drops_d)
                if gi is not None and gi in drops_d:
                    tot["dedupe_p_DELETES_GOLD"] += 1
                    if len(dup_examples) < 25:
                        dup_examples.append(
                            (source, e.get("search_name"), g,
                             real[gi].get("feature_code")))
            drops_da = hygiene.dedupe_rows(real, keep="a")
            if drops_da and gi is not None and gi in drops_da:
                tot["dedupe_a_DELETES_GOLD"] += 1

            # R4
            drops_j = hygiene.drop_junk_exact(real, e.get("search_name", ""))
            if drops_j:
                tot["junk_fires"] += 1
                if gi is not None and gi in drops_j:
                    tot["junk_DELETES_GOLD"] += 1

            # D. unretrievable
            if gi is None:
                tot["unretrievable"] += 1
                if hygiene.alias_query(e.get("search_name", "")):
                    tot["unretrievable_abbrev"] += 1
                elif len(unretr_examples) < 40:
                    unretr_examples.append((source, e.get("search_name"), g))
            elif gi >= 100:
                tot["gold_beyond_window100"] += 1
        del es_data

    print("sources:", want)
    for k in ["entities", "unretrievable", "unretrievable_abbrev",
              "gold_beyond_window100", "gold_not_in_index",
              "has_hist_candidate", "hist_in_top5", "gold_is_hist",
              "demote_h_fires", "demote_h_rows", "demote_h_DELETES_GOLD",
              "dedupe_fires", "dedupe_rows", "dedupe_p_DELETES_GOLD",
              "dedupe_a_DELETES_GOLD", "junk_fires", "junk_DELETES_GOLD"]:
        print("  %-26s %6d" % (k, tot[k]))

    print("\ngolds whose geonameid is NOT in the index:")
    for (s, n, g), c in missing_golds.most_common(20):
        print("   %-9s %-24s %-10s x%d" % (s, str(n)[:24], g, c))
    print("\nentities whose GOLD is a defunct row (a *H rule must not touch):")
    for r in hist_gold_examples[:15]:
        print("   %-9s %-24s %-10s %s" % r)
    print("\nentities whose gold is the row `dedupe --keep p` would delete:")
    for r in dup_examples[:15]:
        print("   %-9s %-24s %-10s %s" % r)
    print("\nunretrievable golds that are NOT abbreviations:")
    for r in unretr_examples[:40]:
        print("   %-9s %-28s %s" % r)


if __name__ == "__main__":
    main()
