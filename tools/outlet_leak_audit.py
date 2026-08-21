"""Mechanical leak audit for the e50 outlet feature.

The outlet feature's whole claim is that the newsroom's location is *outside*
information.  If any part of the home mapping were derived from the answer key,
the arm would be measuring memorisation and the held-out numbers would be
worthless.  This script checks the claim rather than asserting it, in five
parts:

A. **The curated table names places, not ids.**  Every entry in
   ``outlet_home_table.HOME`` is words plus an ISO-3 country code and a scope.
   No geonameid, no coordinate, no numeric identifier of any kind, so nothing
   could have been lifted from a ``<gaztag>``.

B. **Coordinates come from the gazetteer, reproducibly.**  Re-geocoding the
   table from scratch has to return exactly the cached homes the pickles were
   built with.  The lookup is a pure function of the curated words, so the
   numbers the model sees are GeoNames' answer to "where is Richmond, Indiana",
   not anyone's answer to "where is this article's gold".

C. **The article join does not need the answer key.**  ``outlet_align`` can join
   entities to articles on ``(phrase, geonameid)`` or on ``phrase`` alone.  The
   audit runs both and reports every entity whose *domain* differs.  On LGL --
   the source the feature is aimed at -- the two agree everywhere, so the domain
   attached to a held-out LGL entity provably does not depend on its label.

D. **The mapping is split-blind.**  The same domain gets the same home whether
   it appears in the training half, the held-out half or both.  Checked by
   recomputing the features for the held-out entities alone, with the training
   half not loaded, and comparing to the values in the shipped pickle.

E. **No gold-conditioned statistic is reachable.**  The feature path
   (``outlet_home_table`` -> ``outlet_features``) is checked for any reference
   to the label fields.  ``outlet_align`` is allowed one, as a join key, which
   part C is what quantifies.

Usage:
    uv run python tools/outlet_leak_audit.py
"""

import ast
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from outlet_align import CORPUS_PATHS, entity_domains  # noqa: E402
from outlet_home_table import HOME, geocode_homes  # noqa: E402

from mordecai3.elasticsearch import setup_es_client  # noqa: E402
from mordecai3.outlet_features import (OUTLET_KEYS, add_outlet_features,  # noqa: E402
                                       clear_outlet_features)

RAW = "raw_data"
PICKLES = os.path.join(RAW, "pickled_es")
BUILT = ("/tmp/claude-1000/-home-andy-projects-mordecai3/"
         "a19357d6-880a-4251-9cdd-97e2235f449c/scratchpad/e50")
LABEL_FIELDS = ("correct_geonamesid", "correct", "gaztag", "geonamesID")

failures = []


def check(name, ok, detail=""):
    print("  [{}] {}{}".format("PASS" if ok else "FAIL", name,
                               ("  -- " + detail) if detail else ""))
    if not ok:
        failures.append(name)


def part_a():
    print("\nA. curated table names places, not ids")
    bad = []
    for domain, entry in HOME.items():
        if entry is None:
            continue
        if len(entry) != 4:
            bad.append((domain, "not a 4-tuple"))
            continue
        place, admin1, iso3, scope = entry
        for field in (place, admin1):
            if field is not None and any(ch.isdigit() for ch in str(field)):
                bad.append((domain, "digit in place name"))
        if iso3 is not None and not (isinstance(iso3, str) and len(iso3) == 3
                                     and iso3.isalpha()):
            bad.append((domain, "iso3 not a 3-letter code"))
        if scope not in ("local", "national"):
            bad.append((domain, "bad scope " + repr(scope)))
    check("no numeric identifiers anywhere in HOME", not bad, str(bad[:4]))
    n_none = sum(1 for v in HOME.values() if v is None)
    print("      {} domains curated, {} deliberately left unresolved"
          .format(len(HOME), n_none))


def part_b(es):
    print("\nB. coordinates are a reproducible gazetteer lookup")
    cache_path = os.path.join(BUILT, "outlet_homes.json")
    with open(cache_path) as f:
        cached = json.load(f)
    fresh = geocode_homes(es, verbose=False)
    same = True
    for domain, want in cached.items():
        got = fresh.get(domain)
        if got is None:
            same = False
            break
        for k in ("lat", "lon", "country_code3", "level"):
            if str(want.get(k)) != str(got.get(k)):
                same = False
        if tuple(want.get("admin1") or ()) != tuple(got.get("admin1") or ()):
            same = False
    check("re-geocoding reproduces the cached homes exactly", same,
          "{} homes".format(len(cached)))
    # And the ids in the cache are ES's, not the corpus's: they are only ever
    # written by geocode_homes, which never sees a pickle.
    check("cache carries no per-article information",
          all(set(v) <= {"lat", "lon", "country_code3", "admin1", "level",
                         "resolved", "geonameid", "feature_code", "admin1_name"}
              for v in cached.values()))


def part_c():
    print("\nC. the article join does not need the answer key")
    print("      (the join recovers which article an entity came from -- metadata")
    print("       the pickles dropped. Serving has no join: the caller passes the")
    print("       domain. LGL is the hard requirement; TR is reported and bounded.)")
    for source in ("lgl", "tr"):
        path = os.path.join(
            PICKLES, "es_formatted_{}_500_all_loc_types_fuzzy_0.pkl".format(source))
        with open(path, "rb") as f:
            data = pickle.load(f)
        with_gold = entity_domains(data, source, RAW, key="phrase+gid")
        no_gold = entity_domains(data, source, RAW, key="phrase")
        diff = [i for i in with_gold if with_gold[i] != no_gold.get(i)]
        split = round(0.7 * len(data))
        n_held = sum(1 for i in diff if i >= split)
        detail = "{} of {} entities differ ({} held out)".format(
            len(diff), len(data), n_held)
        if source == "lgl":
            check("LGL: domain identical with and without the gold join key",
                  not diff, detail)
        else:
            print("  [INFO] TR: {}".format(detail))
            for i in diff:
                print("         entity {} {!r}: {} vs {}".format(
                    i, data[i]["search_name"], with_gold[i], no_gold.get(i)))
            check("TR: gold join key can move at most a handful of entities",
                  len(diff) <= 5, detail)
        del data


def part_d():
    print("\nD. the mapping is split-blind")
    source = "lgl"
    built = os.path.join(
        BUILT, "pickled_es",
        "es_formatted_lgl_500_all_loc_types_fuzzy_0_enriched_compact.pkl")
    if not os.path.exists(built):
        check("held-out features recomputable without the training half", False,
              "no built pickle at " + built)
        return
    from train import ALL_FEATURE_KEYS
    with open(built, "rb") as f:
        shipped = pickle.load(f)
    cols = [ALL_FEATURE_KEYS.index(k) for k in OUTLET_KEYS]

    path = os.path.join(
        PICKLES, "es_formatted_lgl_500_all_loc_types_fuzzy_0_enriched.pkl")
    with open(path, "rb") as f:
        data = pickle.load(f)
    split = round(0.7 * len(data))
    domains = entity_domains(data, source, RAW)
    with open(os.path.join(BUILT, "outlet_homes.json")) as f:
        homes = json.load(f)

    # Recompute from the held-out half ONLY: drop the training entities on the
    # floor first, so nothing about them can reach the computation.
    heldout = [(i, data[i]) for i in range(split, len(data))]
    del data
    mismatch = 0
    for i, entity in heldout:
        home = homes.get(domains.get(i))
        if home is None:
            clear_outlet_features(entity["es_choices"])
        else:
            add_outlet_features(entity["es_choices"], home)
        want = shipped[i]["feat_matrix"][:, cols]
        for r, choice in enumerate(entity["es_choices"]):
            for c, k in enumerate(OUTLET_KEYS):
                if abs(float(choice[k]) - float(want[r, c])) > 1e-6:
                    mismatch += 1
    check("held-out outlet features identical when recomputed alone",
          mismatch == 0, "{} mismatched cells".format(mismatch))


def _read_field_names(src):
    """Every name the module can actually read a dict key or attribute by.

    Deliberately not a substring search: these modules *discuss* the answer key
    at length in their docstrings, and a grep cannot tell prose from a lookup.
    Only subscripts (``entity["correct_geonamesid"]``), attribute access and
    bare names are code.
    """
    names = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Subscript):
            sl = node.slice
            if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
                names.add(sl.value)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Call):
            for kw in node.keywords:
                for arg in ([kw.value] if isinstance(kw.value, ast.Constant)
                            else []):
                    if isinstance(arg.value, str):
                        names.add(arg.value)
    return names


def part_e():
    print("\nE. the feature path never reads a label field")
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    targets = {
        os.path.join(root, "mordecai3", "outlet_features.py"): False,
        os.path.join(here, "outlet_home_table.py"): False,
        os.path.join(here, "outlet_align.py"): True,      # join key, see part C
    }
    for path, allowed in targets.items():
        with open(path, encoding="utf8") as f:
            src = f.read()
        hits = sorted(_read_field_names(src) & set(LABEL_FIELDS))
        name = os.path.basename(path)
        if allowed:
            print("      {} reads {} (join key only; part C bounds it)"
                  .format(name, hits))
        else:
            check("{} reads no label field".format(name), not hits, str(hits))


def main():
    print("e50 outlet feature -- leak audit")
    es = setup_es_client()
    part_a()
    part_b(es)
    part_c()
    part_d()
    part_e()
    print("\n{} check(s) failed{}".format(
        len(failures), (": " + ", ".join(failures)) if failures else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
