"""Train/serve parity gate for the e52 harness.

Before any before/after number means anything, three things must reproduce the
frozen pickles bit-for-bit on UNCHANGED input:

  1. `rebuild.rebuild_choices` vs the pickled per-candidate features
     (res_formatter's distance block + the entity-level enrichment).
  2. `mordecai3.geoparse._add_cross_entity_counts` vs the pickled
     `adm1_count` / `country_count`.
  3. `mordecai3.candidate_features.add_document_features` vs the pickled
     sibling and geometry block.

Anything that does not read ~0 here is a confound that would show up as a fake
hygiene delta on every touched document.

    python parity_check.py --source TR --n 200
"""
import argparse
import copy
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/home/andy/projects/mordecai3"
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import es_util  # noqa: E402
import rebuild  # noqa: E402
from mordecai3.candidate_features import (CONTEXT_KEYS,  # noqa: E402
                                          GEOMETRY_KEYS, ENTITY_KEYS,
                                          add_document_features)
from mordecai3.geoparse import _add_cross_entity_counts  # noqa: E402
from twin_credit_eval import SOURCES, load_val  # noqa: E402


def maxdiff(old, new, keys):
    out = {}
    for k in keys:
        d = 0.0
        for a, b in zip(old, new):
            va, vb = a.get(k), b.get(k)
            try:
                d = max(d, abs(float(va) - float(vb)))
            except (TypeError, ValueError):
                d = max(d, 0.0 if str(va) == str(vb) else 1.0)
        out[k] = d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="TR")
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    stems = dict(SOURCES)[a.source]
    es_data = load_val(a.source, stems, os.path.join(ROOT, "raw_data"),
                       "_enriched", 500, "all_loc_types", 0)[:a.n]
    es_data = copy.deepcopy(es_data)
    before = copy.deepcopy(es_data)

    # ---- 1. per-candidate rebuild
    gids = [str(c["geonameid"]) for e in es_data
            for c in rebuild.real_choices(e)]
    src = es_util.mget(sorted(set(gids)))
    worst, n_ok, n_skip = {}, 0, 0
    for e in es_data:
        real = rebuild.real_choices(e)
        srcs, ok = [], True
        for c in real:
            s = src.get(str(c["geonameid"]))
            if s is None:
                ok = False
                break
            srcs.append(rebuild.source_from_hit(s))
        if not ok or not real:
            n_skip += 1
            continue
        rb = rebuild.rebuild_choices(e["search_name"], copy.deepcopy(real), srcs)
        for k, v in maxdiff(real, rb, rebuild.DIST_KEYS + ENTITY_KEYS).items():
            worst[k] = max(worst.get(k, 0.0), v)
        n_ok += 1
    print("1. candidate rebuild: %d entities checked, %d skipped (row gone "
          "from the index)" % (n_ok, n_skip))
    bad = {k: round(v, 8) for k, v in sorted(worst.items()) if v > 1e-6}
    print("   max |diff|:", bad or "0.0 on all %d keys" % len(worst))

    # ---- 2/3. document-level recompute
    by_doc = defaultdict(list)
    for e in es_data:
        by_doc[e["doc_key"]].append(e)
    for doc in by_doc.values():
        _add_cross_entity_counts(doc)
        add_document_features(doc)
    worst2 = {}
    keys = ["adm1_count", "country_count"] + CONTEXT_KEYS + GEOMETRY_KEYS
    for old_e, new_e in zip(before, es_data):
        for k, v in maxdiff(old_e["es_choices"], new_e["es_choices"], keys).items():
            worst2[k] = max(worst2.get(k, 0.0), v)
    print("2/3. document recompute over %d docs" % len(by_doc))
    bad2 = {k: round(v, 8) for k, v in sorted(worst2.items()) if v > 1e-6}
    print("   max |diff|:", bad2 or "0.0 on all %d keys" % len(worst2))


if __name__ == "__main__":
    main()
