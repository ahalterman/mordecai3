"""Rebuild one entity's candidate list from live Elasticsearch rows.

Why this exists: several of the model's 26+13 features are normalised WITHIN
the candidate set (`min_dist`/`max_dist`/`avg_dist`/`ascii_dist` through
`geoparse.normalize`, and every `WITHIN_SET_KEY` of the enrichment).  Dropping
a candidate therefore changes the features of the candidates that remain, and a
hygiene arm that edited the list without recomputing would be measuring a
different, broken model rather than the transform.

So: `rebuild_choices` reproduces `geoparse.res_formatter(..., extra_features=
True)` exactly, from the pickle's structural fields plus the `name` /
`asciiname` / `alternativenames` / `population` the live index still holds.
`verify_parity` re-runs it with NO rule enabled and diffs against the pickled
values -- that must read 0.0 before any before/after number is trusted.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/home/andy/projects/mordecai3"
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import jellyfish  # noqa: E402

from mordecai3.candidate_features import (add_entity_features,  # noqa: E402
                                          is_null_choice)
from mordecai3.geoparse import _null_choice, normalize  # noqa: E402

STRUCT_KEYS = ["feature_code", "feature_class", "country_code3", "lat", "lon",
               "name", "admin1_code", "admin1_name", "admin2_code",
               "admin2_name", "geonameid", "admin1_parent_match",
               "country_code_parent_match"]
DIST_KEYS = ["alt_name_length", "min_dist", "max_dist", "avg_dist", "ascii_dist"]


def source_from_hit(hit):
    """An ES `_source` reduced to what the formatter reads."""
    return {"name": hit["name"], "asciiname": hit.get("asciiname", ""),
            "alternativenames": hit.get("alternativenames") or [],
            "population": hit.get("population", 0)}


def struct_from_hit(hit):
    lat, lon = hit["coordinates"].split(",")
    d = {k: hit.get(k, "") for k in
         ["feature_code", "feature_class", "country_code3", "name",
          "admin1_code", "admin1_name", "admin2_code", "admin2_name",
          "geonameid"]}
    d["lat"] = float(lat)
    d["lon"] = float(lon)
    d["admin1_parent_match"] = 0
    d["country_code_parent_match"] = 0
    return d


def rebuild_choices(search_name, structs, sources):
    """res_formatter's body, over rows we already have in hand.

    `structs` are candidate dicts carrying STRUCT_KEYS (the parent-match values
    are carried through from the pickle, since the parent lookup does not
    depend on the candidate set).  `sources` are the matching ES rows.
    """
    choices = []
    alt_lengths, min_d, max_d, avg_d, asc_d = [], [], [], [], []
    for st, src in zip(structs, sources):
        names = [src["name"]] + list(src["alternativenames"])
        dists = [jellyfish.levenshtein_distance(search_name, j) for j in names]
        choices.append({k: st[k] for k in STRUCT_KEYS})
        alt_lengths.append(len(src["alternativenames"]) + 1)
        min_d.append(np.min(dists))
        max_d.append(np.max(dists))
        avg_d.append(np.mean(dists))
        asc_d.append(jellyfish.levenshtein_distance(search_name, src["asciiname"]))
    alt_lengths = np.log(alt_lengths) if alt_lengths else np.zeros(0)
    min_d, max_d = normalize(min_d), normalize(max_d)
    avg_d, asc_d = normalize(avg_d), normalize(asc_d)
    for n, c in enumerate(choices):
        c["alt_name_length"] = alt_lengths[n]
        c["min_dist"] = min_d[n]
        c["max_dist"] = max_d[n]
        c["avg_dist"] = avg_d[n]
        c["ascii_dist"] = asc_d[n]
    add_entity_features(search_name, choices, sources)
    return choices


def real_choices(entity):
    return [c for c in entity["es_choices"] if not is_null_choice(c)]


def finish(entity, choices):
    """Attach a rebuilt list, restore the NULL sentinel and the label vector."""
    null = _null_choice(entity.get("search_name"))
    entity["es_choices"] = choices + [null]
    gold = str(entity.get("correct_geonamesid"))
    entity["correct"] = [str(c.get("geonameid")) == gold
                         for c in entity["es_choices"]]
    return entity


def parity_report(entity, rebuilt, keys):
    """Max abs difference per feature key between pickled and rebuilt rows."""
    old = real_choices(entity)
    if len(old) != len(rebuilt):
        return {"__len__": abs(len(old) - len(rebuilt))}
    out = {}
    for k in keys:
        d = 0.0
        for a, b in zip(old, rebuilt):
            va, vb = a.get(k), b.get(k)
            if isinstance(va, (int, float, np.floating)) and \
               isinstance(vb, (int, float, np.floating)):
                d = max(d, abs(float(va) - float(vb)))
            elif str(va) != str(vb):
                d = max(d, 1.0)
        out[k] = d
    return out
