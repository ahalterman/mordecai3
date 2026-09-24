"""Do the inference-time enrichment features match the ones the model trained on?

The ranking features in ``mordecai3/candidate_features.py`` were originally
computed offline by ``tools/enrich_pickles.py``, which rewrote the cached
candidate pickles the trainer reads.  A model trained on those numbers is only
servable if the live pipeline produces the same numbers, and "the same" has to
mean bit-for-bit-ish, not "similar": ``is_max_pop_exact_match`` flipping on one
candidate changes which place the model picks.

This module takes documents out of the enriched pickles, feeds their candidate
lists back through the *inference* code path -- ``geoparse.res_formatter``,
``geoparse._null_choice``, ``candidate_features.add_document_features``, the
same three calls ``add_es_data_batch`` makes -- and asserts every feature comes
back equal to what the pickle holds.  Because it drives the real functions
rather than a copy of them, it also covers the wiring in ``res_formatter``.

The Geonames ``_source`` records that the features are computed from come from
Elasticsearch by geonameid, which is where ``enrich_pickles.py`` got them too,
so the test needs a live ES with the full index (it skips otherwise) and the
enriched pickles in ``raw_data/pickled_es`` (it skips otherwise).

Run it as a script for the full report -- per-feature max deviation, plus the
window-sensitivity study of inference's 100-candidate window against training's
500-candidate one::

    uv run python tests/test_feature_parity.py
    uv run python tests/test_feature_parity.py --sources prodigy tr --per-source 40
"""

import argparse
import math
import os
import pickle
import sys
import time
from collections import defaultdict

import numpy as np
import pytest

import mordecai3.geoparse as gp
from mordecai3 import candidate_features as cfeat
from mordecai3.elasticsearch import setup_es_client
from mordecai3.torch_model import expand_feature_blocks

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "raw_data", "pickled_es")
FILE_TEMPLATE = "es_formatted_{source}_500_all_loc_types_fuzzy_0_enriched.pkl"

# wiki_docs is left out by default only because its pickle is 2.3 GB; pass it
# with --sources to include it.
DEFAULT_SOURCES = ["prodigy", "tr", "lgl", "gwn", "syn_cities", "syn_caps"]

# The blocks the trained models actually ship with: everything except the
# case-folded edit distances.
SHIPPED_BLOCKS = "prom,name,cue,sib,geo,shape"
SHIPPED_KEYS = expand_feature_blocks(SHIPPED_BLOCKS)
ALL_KEYS = cfeat.ALL_KEYS

# res_formatter's original features, which the reconstruction also has to
# reproduce. The two parent-match features are left out: they depend on the
# "in" relation guessed from the text, which the pickle does not record.
CLASSIC_KEYS = ["alt_name_length", "min_dist", "max_dist", "avg_dist", "ascii_dist"]

TOL = 1e-9

# What a real inference call retrieves, versus what the training pickles hold.
INFERENCE_WINDOW = 100
TRAINING_WINDOW = 500


#
#   Pulling documents out of the enriched pickles
#


def source_path(source, data_dir=DATA_DIR):
    return os.path.join(data_dir, FILE_TEMPLATE.format(source=source))


def sample_documents(source, per_source=40, data_dir=DATA_DIR):
    """Return whole documents from one enriched pickle.

    Whole documents, because the sibling and anchor features are defined against
    the other mentions in the same document; a random sample of entities would
    have nothing to be a sibling of. Documents are taken at an even stride
    through the file so the sample spans it rather than sitting at the front.
    """
    with open(source_path(source, data_dir), "rb") as f:
        data = pickle.load(f)

    by_doc = defaultdict(list)
    for entity in data:
        by_doc[entity["doc_key"]].append(entity)
    doc_keys = list(by_doc)
    stride = max(1, len(doc_keys) // 200)

    docs = []
    n_entities = 0
    for key in doc_keys[::stride]:
        if n_entities >= per_source:
            break
        doc = [
            {
                "source": source,
                "search_name": e["search_name"],
                "doc_key": e["doc_key"],
                "correct": list(e["correct"]),
                "es_choices": e["es_choices"],
            }
            for e in by_doc[key]
        ]
        docs.append(doc)
        n_entities += len(doc)

    del data, by_doc
    return docs


def fetch_sources(es, geonameids, chunk_size=1000):
    """geonameid -> Geonames _source, the same fields enrich_pickles read."""
    ids = sorted(geonameids)
    out = {}
    for start in range(0, len(ids), chunk_size):
        chunk = ids[start:start + chunk_size]
        resp = es.mget(
            index="geonames",
            body={"ids": chunk},
            _source_includes=["population", "name", "asciiname", "alternativenames"],
            request_timeout=120,
        )
        for doc in resp["docs"]:
            if doc.get("found"):
                out[doc["_id"]] = doc["_source"]
    return out


def all_geonameids(docs):
    ids = set()
    for doc in docs:
        for entity in doc:
            for choice in entity["es_choices"]:
                if not cfeat.is_null_choice(choice):
                    ids.add(str(choice["geonameid"]))
    return ids


#
#   Rebuilding a document's candidate lists through the inference code path
#


def _fake_es_response(entity, es_sources, window=None):
    """The ES response res_formatter would have seen for this mention.

    The candidate order in the pickle is the order ES returned, so replaying the
    sources in that order reproduces the response exactly. ``window`` truncates
    to the first N candidates, which is what a smaller ``max_choices`` would
    have retrieved.
    """
    real = [c for c in entity["es_choices"] if not cfeat.is_null_choice(c)]
    if window is not None:
        real = real[:window]
    hits = []
    kept = []
    for choice in real:
        src = es_sources.get(str(choice["geonameid"]))
        if src is None:
            # Not in the index any more; enrich_pickles would have scored it
            # with a population of 0, but res_formatter never sees such a hit.
            continue
        hit = dict(src)
        hit.setdefault("alternativenames", [])
        hit["asciiname"] = src.get("asciiname") or ""
        hit["coordinates"] = "{},{}".format(choice["lat"], choice["lon"])
        for key in ("feature_code", "feature_class", "country_code3", "admin1_code",
                    "admin1_name", "admin2_code", "admin2_name", "geonameid"):
            hit[key] = choice[key]
        hits.append({"_source": hit})
        kept.append(choice)
    return {"hits": {"hits": hits}}, kept


def rebuild_document(doc, es_sources, window=None):
    """Recompute a document's features with the live inference functions.

    Returns ``(rebuilt, gold)``: a list of entity dicts carrying freshly
    computed candidates, and the matching list of the pickle's own candidate
    dicts, aligned candidate by candidate.
    """
    rebuilt = []
    gold = []
    for entity in doc:
        res, kept = _fake_es_response(entity, es_sources, window)
        choices = gp.res_formatter(res, entity["search_name"], None,
                                   extra_features=True)
        choices.append(gp._null_choice(entity["search_name"]))
        rebuilt.append({"search_name": entity["search_name"],
                        "es_choices": choices,
                        "source": entity["source"]})
        null_gold = [c for c in entity["es_choices"] if cfeat.is_null_choice(c)]
        gold.append(kept + null_gold)

    cfeat.add_document_features(rebuilt)
    return rebuilt, gold


def compare(rebuilt, gold, keys):
    """Per-key max absolute deviation and a count of the candidates compared."""
    diffs = {k: 0.0 for k in keys}
    worst = {k: None for k in keys}
    n = 0
    for ent_new, ent_gold in zip(rebuilt, gold):
        for new, old in zip(ent_new["es_choices"], ent_gold):
            assert str(new["geonameid"]) == str(old["geonameid"])
            n += 1
            for key in keys:
                d = abs(float(new[key]) - float(old[key]))
                if d > diffs[key]:
                    diffs[key] = d
                    worst[key] = (ent_new["search_name"], old["geonameid"],
                                  float(old[key]), float(new[key]))
    return diffs, worst, n


#
#   The test
#


@pytest.fixture(scope="module")
def parity_sample(geonames_service_all_data):
    sources = [s for s in DEFAULT_SOURCES if os.path.exists(source_path(s))]
    if not sources:
        pytest.skip(f"no enriched pickles in {DATA_DIR}")
    docs = []
    for source in sources:
        docs.extend(sample_documents(source, per_source=40))
    n_entities = sum(len(d) for d in docs)
    if n_entities < 200:
        pytest.skip(f"only {n_entities} entities available; need 200")
    es_sources = fetch_sources(geonames_service_all_data.conn, all_geonameids(docs))
    return docs, es_sources


def test_enrichment_features_match_training_pickles(parity_sample):
    """Every shipped feature, recomputed live, equals the trained-on value."""
    docs, es_sources = parity_sample
    diffs = {k: 0.0 for k in ALL_KEYS}
    n = 0
    for doc in docs:
        rebuilt, gold = rebuild_document(doc, es_sources)
        d, _, count = compare(rebuilt, gold, ALL_KEYS)
        n += count
        for k, v in d.items():
            diffs[k] = max(diffs[k], v)
    bad = {k: v for k, v in diffs.items() if v > TOL}
    assert not bad, f"features differ from the training pickles: {bad}"
    assert n > 1000


def test_res_formatter_reproduces_the_original_features(parity_sample):
    """The rebuild is faithful: the pre-enrichment features match too.

    If this fails, the parity test above is comparing against candidate lists
    that inference would never have produced, so it proves nothing.
    """
    docs, es_sources = parity_sample
    diffs = {k: 0.0 for k in CLASSIC_KEYS}
    for doc in docs:
        rebuilt, gold = rebuild_document(doc, es_sources)
        d, _, _ = compare(rebuilt, gold, CLASSIC_KEYS)
        for k, v in d.items():
            diffs[k] = max(diffs[k], v)
    bad = {k: v for k, v in diffs.items() if v > TOL}
    assert not bad, f"res_formatter's own features differ: {bad}"


def test_null_row_is_not_the_best_supported_candidate():
    """The placeholder row must never look well-supported on a distance."""
    null = gp._null_choice("Aleppo")
    assert null["log_min_km_anchor"] == pytest.approx(math.log10(20001))
    assert null["log_mean_sibmin"] == pytest.approx(math.log10(20001))
    assert null["min_dist_cf"] == 1.0
    assert null["mention_admin_cue"] == 0.0
    assert gp._null_choice("Fairfax County")["mention_admin_cue"] == 1.0
    # ...and legacy callers get the plain row they always got.
    assert not set(ALL_KEYS) & set(gp._null_choice())


def test_feature_blocks_are_off_by_default():
    """A candidate dict from the default path carries no enrichment keys."""
    res = {"hits": {"hits": []}}
    assert gp.res_formatter(res, "Aleppo") == []
    assert len(SHIPPED_KEYS) == 26


def test_live_lookup_produces_every_feature(geoparser_all_data):
    """The real lookup path, against the real index, fills in all 26 keys."""
    geo = geoparser_all_data
    texts = ["Fighting continued in Aleppo, Syria for a third day.",
             "Trains from Utrecht reach the Netherlands' Lafayette County twin."]
    docs = list(geo.nlp.pipe(texts, batch_size=2))
    all_doc_ex = [gp.doc_to_ex_expanded(d) for d in docs]

    geo.geonames.clear_cache()
    plain = gp.add_es_data_batch([[dict(e) for e in d] for d in all_doc_ex],
                                 geo.geonames, max_results=100)
    for doc_es in plain:
        for ent in doc_es:
            for choice in ent["es_choices"]:
                assert not set(ALL_KEYS) & set(choice)

    geo.geonames.clear_cache()
    rich = gp.add_es_data_batch([[dict(e) for e in d] for d in all_doc_ex],
                                geo.geonames, max_results=100, extra_features=True)
    for doc_es in rich:
        for ent in doc_es:
            for choice in ent["es_choices"]:
                missing = [k for k in ALL_KEYS if k not in choice]
                assert not missing, f"{ent['search_name']}: missing {missing}"
                # The raw ES fields the features are computed from must not be
                # riding along: alternativenames lists are the reason a
                # candidate list is expensive to hold.
                assert "alternativenames" not in choice
                assert "population" not in choice

    # ...and the document-level features actually fire. "Syria" is a sibling
    # mention of "Aleppo", so Syrian candidates should be near its anchor.
    # (sib_country wants the ISO short name -- "syrian arab republic" -- so it
    # is checked on the Utrecht/Netherlands document instead.)
    aleppo = rich[0][0]
    assert aleppo["search_name"] == "Aleppo"
    assert any(c["frac_anchors_150km"] == 1.0 for c in aleppo["es_choices"])
    assert any(c["log_min_km_anchor"] < 3 for c in aleppo["es_choices"])
    utrecht = [e for e in rich[1] if e["search_name"] == "Utrecht"]
    assert utrecht and any(c["sib_country"] == 1.0
                           for c in utrecht[0]["es_choices"])
    # The mention "Lafayette County" announces an administrative unit.
    lafayette = [e for e in rich[1] if e["search_name"] == "Lafayette County"]
    if lafayette:
        assert all(c["mention_admin_cue"] == 1.0
                   for c in lafayette[0]["es_choices"])


def test_feature_blocks_must_match_the_checkpoint():
    """Asking for features a checkpoint wasn't trained on is an error, not a crash."""
    from importlib import resources
    from mordecai3.geoparse import load_model
    legacy = resources.files("mordecai3") / "assets/mordecai_2025-08-27.pt"
    with pytest.raises(ValueError, match="extra"):
        load_model(legacy, device="cpu", n_extra_features=26)
    assert load_model(legacy, device="cpu", n_extra_features=0) is not None


#
#   Script mode: the full report
#


def _print_table(title, rows, headers):
    print()
    print(title)
    widths = [max(len(str(r[i])) for r in [headers] + rows) for i in range(len(headers))]
    line = "  ".join(str(h).ljust(w) for h, w in zip(headers, widths))
    print(line)
    print("-" * len(line))
    for row in rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(row, widths)))


def report_parity(docs, es_sources):
    diffs = {k: 0.0 for k in ALL_KEYS + CLASSIC_KEYS}
    worst = {}
    n = 0
    t0 = time.time()
    for doc in docs:
        rebuilt, gold = rebuild_document(doc, es_sources)
        d, w, count = compare(rebuilt, gold, ALL_KEYS + CLASSIC_KEYS)
        n += count
        for k, v in d.items():
            if v > diffs[k]:
                diffs[k] = v
                worst[k] = w[k]
    secs = time.time() - t0
    rows = []
    for key in ALL_KEYS + CLASSIC_KEYS:
        tag = "shipped" if key in SHIPPED_KEYS else (
            "classic" if key in CLASSIC_KEYS else "cf")
        rows.append([key, tag, "{:.3e}".format(diffs[key]),
                     "OK" if diffs[key] <= TOL else "MISMATCH"])
    _print_table(
        f"== parity: {n:,} candidate rows, recomputed in {secs:.1f}s ==",
        rows, ["feature", "block", "max abs diff", ""])
    bad = {k: v for k, v in diffs.items() if v > TOL}
    if bad:
        print()
        print("MISMATCHES:")
        for k in bad:
            print("  {}: {}".format(k, worst[k]))
    else:
        print()
        print(f"all {len(ALL_KEYS)} enrichment features + {len(CLASSIC_KEYS)} "
              f"original features match to {TOL:g}")
    return not bad


def report_window_sensitivity(docs, es_sources):
    """What changes when inference retrieves 100 candidates instead of 500?"""
    shifts = {k: [] for k in SHIPPED_KEYS}
    flips = {k: 0 for k in ("is_max_pop_exact_match", "is_unique_exact_match",
                            "is_max_pop", "exact_altname_match")}
    gold_seen = 0
    gold_lost = 0
    n_entities = 0
    n_rows = 0

    for doc in docs:
        wide, _ = rebuild_document(doc, es_sources, window=TRAINING_WINDOW)
        narrow, _ = rebuild_document(doc, es_sources, window=INFERENCE_WINDOW)
        for entity, ent_wide, ent_narrow in zip(doc, wide, narrow):
            n_entities += 1
            # Where is the gold candidate in the retrieved order?
            real_gold = [
                i for i, (c, ok) in enumerate(
                    zip([c for c in entity["es_choices"]
                         if not cfeat.is_null_choice(c)], entity["correct"]))
                if ok
            ]
            if real_gold:
                gold_seen += 1
                if min(real_gold) >= INFERENCE_WINDOW:
                    gold_lost += 1

            by_id_narrow = {}
            for c in ent_narrow["es_choices"]:
                by_id_narrow.setdefault(str(c["geonameid"]), c)
            for c_wide in ent_wide["es_choices"]:
                c_narrow = by_id_narrow.get(str(c_wide["geonameid"]))
                if c_narrow is None:
                    continue          # dropped by the smaller window
                n_rows += 1
                for key in SHIPPED_KEYS:
                    shifts[key].append(abs(float(c_wide[key]) - float(c_narrow[key])))

            # Argmax-relevant flips, on the gold candidate only.
            for idx in real_gold[:1]:
                real = [c for c in entity["es_choices"]
                        if not cfeat.is_null_choice(c)]
                gid = str(real[idx]["geonameid"])
                c_narrow = by_id_narrow.get(gid)
                c_wide = next((c for c in ent_wide["es_choices"]
                               if str(c["geonameid"]) == gid), None)
                if c_narrow is None or c_wide is None:
                    continue
                for key in flips:
                    if float(c_wide[key]) != float(c_narrow[key]):
                        flips[key] += 1

    rows = []
    for key in SHIPPED_KEYS:
        arr = np.array(shifts[key]) if shifts[key] else np.zeros(1)
        rows.append([key,
                     "{:.5f}".format(arr.mean()),
                     "{:.5f}".format(arr.max()),
                     "{:.2f}%".format(100.0 * (arr > 1e-9).mean())])
    _print_table(
        f"== window sensitivity: top-{INFERENCE_WINDOW} vs top-{TRAINING_WINDOW}, "
        f"{n_rows:,} shared candidate rows ==",
        rows, ["feature", "mean abs shift", "max abs shift", "% rows changed"])

    print()
    print("argmax-relevant flips on the gold candidate "
          f"({gold_seen:,} entities with a gold candidate in the top {TRAINING_WINDOW}):")
    for key, count in flips.items():
        print("  {:<24} {:>6} ({:.2f}%)".format(
            key, count, 100.0 * count / max(gold_seen, 1)))
    print()
    print("gold candidate falls outside the top {}: {:,} of {:,} ({:.2f}%)".format(
        INFERENCE_WINDOW, gold_lost, gold_seen, 100.0 * gold_lost / max(gold_seen, 1)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs="+", default=DEFAULT_SOURCES)
    parser.add_argument("--per-source", type=int, default=40)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--skip-window", action="store_true")
    args = parser.parse_args()

    es = setup_es_client()
    if not es.ping():
        sys.exit("cannot reach Elasticsearch at localhost:9200")

    docs = []
    for source in args.sources:
        path = source_path(source, args.data_dir)
        if not os.path.exists(path):
            print(f"skipping {source}: {path} not found")
            continue
        t0 = time.time()
        sampled = sample_documents(source, args.per_source, args.data_dir)
        docs.extend(sampled)
        print("{:<12} {:>4} docs, {:>5} entities ({:.1f}s)".format(
            source, len(sampled), sum(len(d) for d in sampled), time.time() - t0))

    n_entities = sum(len(d) for d in docs)
    print("\n{:,} entities from {:,} documents across {} sources".format(
        n_entities, len(docs), len(args.sources)))

    ids = all_geonameids(docs)
    t0 = time.time()
    es_sources = fetch_sources(es, ids)
    print("fetched {:,} of {:,} geonameids from ES ({:.1f}s)".format(
        len(es_sources), len(ids), time.time() - t0))

    ok = report_parity(docs, es_sources)
    if not args.skip_window:
        report_window_sensitivity(docs, es_sources)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
