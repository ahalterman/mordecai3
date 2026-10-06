"""Add extra ranking features to the cached Elasticsearch candidate pickles.

The pickles in ``raw_data/pickled_es`` hold, for every location mention, the list
of candidate Geonames entries that Elasticsearch returned for it.  Those
candidate dicts carry the features the ranking model currently uses, but they are
missing a few obvious signals: how big the place is, whether the mention string
is literally one of the place's names, and whether the mention itself announces
an administrative unit.

This script writes ``<original>_enriched.pkl`` next to every source pickle.  The
enriched copies are byte-for-byte identical to the originals except that each
candidate dict gains:

    log_population      log10(population + 1), population pulled from ES by
                        geonameid (0.0 when the id is not in the index)
    exact_name_match    1.0 if the mention equals the candidate's name
    exact_altname_match 1.0 if the mention equals the candidate's asciiname or
                        any of its Geonames alternate names
    mention_admin_cue   1.0 if the mention contains a word like "county" or
                        "province" (a per-entity value, repeated on every
                        candidate so the scorer can let it interact with
                        feature_class)
    is_admin_class      1.0 if feature_class == "A"

Four more features are computed *within* each entity's candidate set, so they say
how a candidate stacks up against its rivals rather than describing it alone:

    has_population      1.0 if the ES population is > 0.  Roughly 92% of Geonames
                        rows carry population 0, which mixes "genuinely empty"
                        with "nobody filled it in"; this lets the scorer tell a
                        log_population of 0 from a real, small number.
    is_max_pop          1.0 if this candidate ties the largest population in the
                        set and that maximum is > 0 (every tied candidate gets
                        1.0)
    log_pop_rel         log10(pop + 1) minus the set's maximum of the same, so
                        0.0 for the leader and negative below it (0.0 for every
                        candidate when the whole set has no population)
    is_max_pop_exact_match
                        1.0 if the candidate matches the mention by name
                        (exact_altname_match == 1.0) and has the largest
                        population among the candidates that do, that maximum
                        being > 0 -- "of the places actually called Denver, the
                        big one"

The placeholder "no correct answer" row that is appended to every candidate list
(geonameid == "NULL") gets the same keys with neutral values so the per-entity
feature arrays stay rectangular; it is never looked up in Elasticsearch.

A last group of features looks outside the entity altogether, at the other
mentions in the same document.  Entities from one document carry a byte-identical
``doc_tensor``, so hashing those bytes recovers document membership; every entity
dict gains a ``doc_key`` holding that hash (score-pooling experiments will want
it).  The sibling features below ask whether some *other* mention in the document
names the candidate's parent unit -- "Springfield" is much more likely to be the
Illinois one when "Illinois" appears elsewhere in the article:

    sib_adm1            1.0 if the candidate's admin1_name is one of the other
                        mention strings in the document
    sib_adm2            same for admin2_name
    sib_country         same for the candidate's country, matched through the
                        ISO short names in mordecai3/assets
    ap_twin             1.0 if another candidate in the set shares this
                        candidate's name, sits within 0.15 degrees of it in both
                        lat and lon, and the two straddle feature classes A and P
                        -- the city-inside-its-own-district pattern

Following tools/an.py, the sibling name set excludes the entity's own mention
string, so a repeated mention of the same name is not evidence for itself.

A fourth group, screened separately (see the feature-screening report), places
the candidate against the *anchors* of its sibling mentions.  Each sibling
mention is resolved provisionally by the population prior -- exact name match
first, then largest population -- and the resulting point is an anchor.  A
candidate that sits near where the rest of the document points is usually the
right one:

    log_min_km_anchor       log10(km + 1) to the nearest sibling anchor
    log_mean_sibmin         the "soft" form: for each sibling, the distance to
                            that sibling's NEAREST plausible candidate (top 20
                            by the same prior), averaged over siblings, so one
                            badly-resolved sibling cannot dominate
    frac_anchors_50km       fraction of anchors within 50 km
    frac_anchors_150km      fraction of anchors within 150 km
    frac_sibs_50km          the soft counterpart of frac_anchors_50km
    anchor_same_adm1_frac   fraction of anchors in this candidate's country+adm1
    anchor_same_country_frac    fraction of anchors in this candidate's country
    is_parent_of_anchor     1.0 if the candidate is the country/adm1/adm2 that
                            contains at least one anchor
    frac_anchors_inside     the fraction of anchors it contains

With no usable sibling (a one-mention document), the distance features take the
sentinel log10(20001) and the fractions are 0.0.  The NULL placeholder row takes
the same sentinels, so that "there is no right answer" never scores as the
best-supported option in the set.

Three set-shape gates say how ambiguous the mention is, which is what lets a
scorer decide when to trust the geometry over the priors, plus one prominence
flag:

    log_n_same_name, log_n_exact_matches, is_unique_exact_match, is_seat_any

Finally, four case-folded edit distances.  The originals in the pickles come from
geoparse.res_formatter, which calls jellyfish on the raw strings, so an all-caps
mention such as "SYRIAN ARAB REPUBLIC" scores a large distance against every
candidate.  These recompute the same statistics with both sides lowercased and
the same normalisation, and are added as NEW keys so an ablation can separate
them from the originals:

    min_dist_cf, max_dist_cf, avg_dist_cf, ascii_dist_cf

Two more, from the error analysis of the best Wave 2 model:

    ap_twin_stripped    ap_twin again, but the two names only have to match
                        after de-accenting and dropping administrative words,
                        so "Yangon Region" pairs with "Yangon" and "Kathmandu
                        District" with "Kathmandu".  13% of that model's errors
                        are twin pairs the exact-name ap_twin cannot see.
    is_historical       1.0 for a defunct GeoNames row (ADM1H, PPLH, PCLH, ...)
    ap_twin_stripped_wide
                        the same stripped-name A/P rule at 0.5 degrees.  The
                        0.15-degree gate is tuned to a city sitting inside its
                        own district; a large first-order unit's centroid is
                        further out than that (Yangon Region is 0.195 degrees
                        from Yangon), so the narrow flag misses exactly the
                        province-versus-city pairs.  Kept as a separate flag so
                        the shared AP_TWIN_DEGREES, and tools/rewrite_labels.py
                        with it, stay untouched.

The name key comes from tools/rewrite_labels.strip_key, imported rather than
copied so the two scripts cannot drift apart.

Work is done in three passes so that Elasticsearch traffic stays sane and no two
pickles are ever in memory at once:

    1. scan every pickle for the unique geonameids and unique mention strings
    2. one mget sweep over that id set
    3. re-read each pickle, attach features, write the enriched copy, verify it

Usage (from the repo root):

    uv run python tools/enrich_pickles.py
    uv run python tools/enrich_pickles.py --sources prodigy --limit 200 --out-dir /tmp

``--outlet-only`` is a separate, much cheaper mode that does not re-enrich
anything: it reads the *already enriched* pickles, attaches the ``outlet``
feature block (mordecai3/outlet_features.py) and writes the compacted cache the
trainer reads.  Keeping it separate is deliberate -- the outlet arm has to be
contrasted against a baseline on pickles whose other 33 features are provably
the frozen ones, and re-deriving those from Elasticsearch would put a second,
uncontrolled difference into the comparison.

    uv run python tools/enrich_pickles.py --outlet-only \
        --data-dir raw_data/pickled_es --out-dir /path/to/e50/pickled_es
"""

import argparse
import csv
import hashlib
import math
import os
import pickle
import re
import sys
import time
from collections import Counter, defaultdict

import jellyfish
import numpy as np

from mordecai3.elasticsearch import setup_es_client

# The stripped name key is shared with tools/rewrite_labels.py, which is where
# the e12 twin analysis defined it.  Import it rather than restate it: these two
# scripts have to agree on what counts as the same name.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rewrite_labels import strip_key  # noqa: E402

DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "raw_data",
    "pickled_es",
)

SOURCES = [
    "prodigy",
    "tr",
    "lgl",
    "gwn",
    "syn_cities",
    "syn_caps",
    "wiki_docs",
]

FILE_TEMPLATE = "es_formatted_{source}_500_all_loc_types_fuzzy_0.pkl"

# The sentinel row appended to every candidate list uses this geonameid.
NULL_GEONAMEID = "NULL"

# Per-candidate features that depend only on the candidate itself.
BASE_KEYS = [
    "log_population",
    "exact_name_match",
    "exact_altname_match",
    "mention_admin_cue",
    "is_admin_class",
]

# Features that read the rest of the document, or pair candidates up.
CONTEXT_KEYS = [
    "sib_adm1",
    "sib_adm2",
    "sib_country",
    "ap_twin",
]

# Sibling-anchor geometry, copied from the screen's extract_feats.py, plus the
# "soft" per-sibling variant from its anchor_variants.py.
GEOMETRY_KEYS = [
    "log_min_km_anchor",
    "anchor_same_adm1_frac",
    "frac_anchors_150km",
    "frac_anchors_50km",
    "log_mean_sibmin",
    "frac_sibs_50km",
    "anchor_same_country_frac",
    "is_parent_of_anchor",
    "frac_anchors_inside",
]

# How ambiguous is this mention, and is the candidate an administrative seat?
SETSHAPE_KEYS = [
    "log_n_same_name",
    "log_n_exact_matches",
    "is_unique_exact_match",
    "is_seat_any",
]

# Case-folded reruns of the res_formatter edit distances.
CASEFOLD_KEYS = ["min_dist_cf", "max_dist_cf", "avg_dist_cf", "ascii_dist_cf"]

# From the error analysis of the best Wave 2 model.
ROUND5_KEYS = ["ap_twin_stripped", "is_historical"]

# The wide-radius companion to ap_twin_stripped.
ROUND6_KEYS = ["ap_twin_stripped_wide"]

PCL_CODES = {"PCLI", "PCL", "PCLD", "PCLS", "PCLF", "PCLIX", "TERR"}

# GeoNames marks a defunct entry by suffixing "H" to the code, but only for the
# administrative / populated / political families.  Plain endswith("H") is a
# trap: it also catches SCH (school), CH (church), MRSH (marsh), RNCH (ranch),
# BCH (beach) and a dozen more, which between them are 97% of the *H rows in
# this data and have nothing to do with history.  So: an explicit list of the
# historical codes whose base is a place in the hierarchy.
HISTORICAL_BASE_CODES = ["ADM1", "ADM2", "ADM3", "ADM4", "ADM5", "ADMD",
                         "PPL", "PPLC", "PCL", "RGN", "LTER", "TERR"]
HISTORICAL_CODES = {base + "H" for base in HISTORICAL_BASE_CODES}
SEAT_CODES = {"PPLC", "PPLA", "PPLA2", "PPLA3", "PPLA4", "PPLA5", "PPLG"}

# How many candidates of a sibling count as "plausible" for the soft variant.
ANCHOR_TOPK = 20

# Stand-in distance when the document offers no usable sibling anchor.
NO_ANCHOR_KM = 20000.0

# What the NULL placeholder row gets.  0.0 is the right filler for a fraction or
# an indicator -- it reads as "no evidence" -- but it is the *best* possible
# value for a distance, and the placeholder competes with the real candidates in
# the softmax.  Distances therefore get the same sentinel a real candidate gets
# when the document offers nothing (the no-anchor distance, and the worst
# normalised edit distance), so the placeholder never looks well-supported.
NULL_SENTINELS = {
    "log_min_km_anchor": math.log10(NO_ANCHOR_KM + 1),
    "log_mean_sibmin": math.log10(NO_ANCHOR_KM + 1),
    "min_dist_cf": 1.0,
    "max_dist_cf": 1.0,
    "avg_dist_cf": 1.0,
    "ascii_dist_cf": 1.0,
}


def null_value(key):
    """The value the NULL placeholder row carries for one enrichment feature."""
    return NULL_SENTINELS.get(key, 0.0)

# Wider gate for the stripped-name twin flag only; see ap_twin_stripped_wide.
AP_TWIN_WIDE_DEGREES = 0.5

# Degrees of latitude/longitude within which an A and a P candidate of the same
# name are treated as the same place seen twice.
AP_TWIN_DEGREES = 0.15

COUNTRY_CODES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "mordecai3",
    "assets",
    "wikipedia-iso-country-codes.txt",
)

# Features that compare a candidate against the rest of its entity's set.
WITHIN_SET_KEYS = [
    "has_population",
    "is_max_pop",
    "log_pop_rel",
    "is_max_pop_exact_match",
]

NEW_KEYS = (
    BASE_KEYS + WITHIN_SET_KEYS + CONTEXT_KEYS
    + GEOMETRY_KEYS + SETSHAPE_KEYS + CASEFOLD_KEYS + ROUND5_KEYS + ROUND6_KEYS
)

ADMIN_CUE_TOKENS = [
    "county",
    "district",
    "province",
    "governorate",
    "state",
    "region",
    "prefecture",
    "department",
    "municipality",
    "oblast",
    "canton",
    "parish",
    "division",
    "territory",
]

ADMIN_CUE_RE = re.compile(r"\b(?:{})\b".format("|".join(ADMIN_CUE_TOKENS)), re.IGNORECASE)

# Sanity anchors for the population join: (geonameid, label, low, high).
SPOT_CHECKS = [
    ("5419384", "Denver, CO", 400_000, 1_200_000),
    ("2643743", "London, GBR", 6_000_000, 12_000_000),
    ("5128581", "New York City", 6_000_000, 12_000_000),
    ("1850147", "Tokyo", 5_000_000, 15_000_000),
    ("2988507", "Paris", 1_500_000, 3_000_000),
    ("1275339", "Mumbai", 8_000_000, 16_000_000),
    ("360630", "Cairo", 5_000_000, 12_000_000),
    ("2147714", "Sydney", 3_000_000, 6_000_000),
    ("1642911", "Jakarta", 5_000_000, 12_000_000),
    ("4887398", "Chicago", 2_000_000, 4_000_000),
]


def load_country_names(path=COUNTRY_CODES_PATH):
    """Map alpha-3 country code -> lowercased English short name."""
    with open(path, encoding="utf8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames
        name_col = [c for c in cols if "name" in c.lower() or "Country" in c][0]
        alpha3_col = [c for c in cols if "Alpha-3" in c][0]
        return {row[alpha3_col]: norm(row[name_col]) for row in reader}


def clean_code(v):
    """Admin codes as the screen treats them: placeholders become empty."""
    v = str(v).strip()
    return "" if v in ("", "NULL", "00", "None") else v


def haversine(lat1, lon1, lat2, lon2):
    """lat1/lon1 arrays, lat2/lon2 arrays -> [len1, len2] km."""
    r1 = np.radians(lat1)[:, None]
    r2 = np.radians(lat2)[None, :]
    dlat = r2 - r1
    dlon = np.radians(lon2)[None, :] - np.radians(lon1)[:, None]
    a = np.sin(dlat / 2) ** 2 + np.cos(r1) * np.cos(r2) * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def normalize_distances(values):
    """geoparse.normalize: shift by the min, divide by the max of the raw array."""
    arr = np.array(values, dtype=np.float64)
    if len(arr) > 0:
        max_arr = np.max(arr)
        if max_arr == 0:
            max_arr = 0.001
        arr = (arr - np.min(arr)) / max_arr
    return arr


def prep_entity(entity):
    """The per-candidate arrays the geometry needs (screen's prep_entity)."""
    real = [c for c in entity["es_choices"] if not is_null_choice(c)]
    n = len(real)
    return {
        "cands": real,
        "n": n,
        "lat": np.array([float(c["lat"]) for c in real], dtype=np.float64)
        if n
        else np.zeros(0),
        "lon": np.array([float(c["lon"]) for c in real], dtype=np.float64)
        if n
        else np.zeros(0),
        "pop": np.array([float(c.get("log_population", 0.0)) for c in real]),
        "exact": np.array([float(c.get("exact_name_match", 0.0)) for c in real]),
        "name": [norm(c["name"]) for c in real],
        "cc": [str(c.get("country_code3", "")) for c in real],
        "a1": [clean_code(c.get("admin1_code")) for c in real],
        "a2": [clean_code(c.get("admin2_code")) for c in real],
        "fc": [str(c.get("feature_code", "")) for c in real],
    }


def document_key(entity):
    """Entities from the same document share a byte-identical doc_tensor."""
    return hashlib.sha1(entity["doc_tensor"].tobytes()).hexdigest()


def source_path(data_dir, source):
    return os.path.join(data_dir, FILE_TEMPLATE.format(source=source))


def enriched_path(out_dir, source, suffix):
    base = FILE_TEMPLATE.format(source=source)
    return os.path.join(out_dir, base.replace(".pkl", suffix + ".pkl"))


def load_pickle(path, limit=None):
    with open(path, "rb") as f:
        data = pickle.load(f)
    if limit:
        data = data[:limit]
    return data


def norm(text):
    """Lowercase/strip a name for exact-match comparisons."""
    return str(text).strip().lower()


def is_null_choice(choice):
    return str(choice.get("geonameid", "")).strip().upper() == NULL_GEONAMEID


#
#   Pass 1: what do we need from Elasticsearch?
#


def scan_source(path, limit=None):
    """Collect the unique geonameids and mention strings in one pickle."""
    data = load_pickle(path, limit)
    ids = set()
    names = set()
    n_slots = 0
    for entity in data:
        names.add(norm(entity["search_name"]))
        choices = entity["es_choices"]
        n_slots += len(choices)
        for choice in choices:
            if not is_null_choice(choice):
                ids.add(str(choice["geonameid"]))
    del data
    return ids, names, n_slots


#
#   Pass 2: one sweep over Elasticsearch
#


def fetch_es_features(es, geonameids, mention_names, chunk_size=1000, verbose=True):
    """Fetch population and names for every geonameid we care about.

    Returns ``(populations, altname_hits, n_missing)`` where ``populations`` maps
    geonameid -> the raw population (kept raw because the within-set features
    compare populations to each other) and ``altname_hits`` maps geonameid -> a
    set of mention strings that match one of that place's alternate names.  Only
    alternate names that some mention actually uses are kept, which is what keeps
    this table small enough to hold in memory.
    """
    ids = sorted(geonameids)
    populations = {}
    altname_hits = {}
    # gid -> (lowercased [name] + alternativenames, lowercased asciiname), which
    # is what res_formatter measures edit distance against.
    es_names = {}
    missing = 0
    t0 = time.time()
    for start in range(0, len(ids), chunk_size):
        chunk = ids[start : start + chunk_size]
        resp = es.mget(
            index="geonames",
            body={"ids": chunk},
            _source_includes=["population", "name", "asciiname", "alternativenames"],
            request_timeout=120,
        )
        for doc in resp["docs"]:
            if not doc.get("found"):
                missing += 1
                continue
            src = doc["_source"]
            gid = doc["_id"]
            try:
                pop = int(src.get("population") or 0)
            except (TypeError, ValueError):
                pop = 0
            populations[gid] = pop

            candidates = []
            ascii_name = src.get("asciiname")
            if ascii_name:
                candidates.append(ascii_name)
            alt = src.get("alternativenames") or []
            if isinstance(alt, str):
                alt = [alt]
            candidates.extend(alt)
            hits = {n for n in (norm(c) for c in candidates) if n in mention_names}
            if hits:
                altname_hits[gid] = hits

            name = src.get("name") or ""
            es_names[gid] = (
                tuple(str(x).lower() for x in [name] + list(alt)),
                str(ascii_name or "").lower(),
            )
        if verbose and (start // chunk_size) % 25 == 0:
            done = min(start + chunk_size, len(ids))
            rate = done / max(time.time() - t0, 1e-6)
            print(
                "    fetched {:,}/{:,} ids ({:.0f}/s)".format(done, len(ids), rate),
                flush=True,
            )
    if verbose:
        print(
            "    fetched {:,} ids in {:.1f}s, {:,} not in index".format(
                len(ids), time.time() - t0, missing
            ),
            flush=True,
        )
    return populations, altname_hits, es_names, missing


#
#   Pass 3: attach the features and write the enriched pickle
#


def fingerprint(data):
    """Hash the parts of a pickle that enrichment must not touch."""
    h = hashlib.sha1()
    for entity in data:
        h.update(str(entity["search_name"]).encode("utf8", "replace"))
        h.update(b"\x00")
        h.update(str(entity["correct_geonamesid"]).encode("utf8", "replace"))
        h.update(b"\x00")
        for choice in entity["es_choices"]:
            h.update(str(choice["geonameid"]).encode("utf8", "replace"))
            h.update(b",")
        h.update(b"\x00")
        for flag in entity["correct"]:
            h.update(b"1" if flag else b"0")
        h.update(b"\n")
    return h.hexdigest()


def enrich_source(data, pop_map, altname_hits, es_names, country_names):
    """Attach the new keys in place and return summary stats for this source."""
    # Recover document membership, and tag every entity with its document.
    doc_names = defaultdict(set)
    doc_members = defaultdict(list)
    for i, entity in enumerate(data):
        key = document_key(entity)
        entity["doc_key"] = key
        doc_names[key].add(norm(entity["search_name"]))
        doc_members[key].append(i)

    stats = {
        "documents": len(doc_names),
        "entities_with_siblings": 0,
        "entities_with_ap_twin": 0,
        "entities_with_ap_twin_at_mention": 0,
        "entities_with_ap_twin_stripped": 0,
        "entities_with_ap_twin_stripped_wide": 0,
        "entities": len(data),
        "real_candidates": 0,
        "null_rows": 0,
        "pop_gt_zero": 0,
        "in_es": 0,
        "exact_name_match": 0,
        "exact_altname_match": 0,
        "admin_class": 0,
        "entities_with_admin_cue": 0,
        "is_max_pop": 0,
        "is_max_pop_exact_match": 0,
        "correct_candidates": 0,
        "incorrect_candidates": 0,
        "is_max_pop_correct": 0,
        "is_max_pop_incorrect": 0,
        "is_max_alt_correct": 0,
        "is_max_alt_incorrect": 0,
        "sets_all_zero_pop": 0,
        "sets_with_pop_tie": 0,
        # firing counts and gold hits, overall and on an.py's "hard set"
        # (candidates whose name equals the mention, in docs that have siblings)
        "hard_candidates": 0,
        "hard_gold": 0,
        "entities_with_candidates": 0,
        "entities_with_anchor": 0,
        "anchor_total": 0,
    }
    for key in CONTEXT_KEYS + ROUND5_KEYS + ROUND6_KEYS:
        stats[key] = 0
        stats[key + "_gold"] = 0
        stats[key + "_hard"] = 0
        stats[key + "_hard_gold"] = 0
    spot_seen = defaultdict(set)
    spot_ids = {gid for gid, _, _, _ in SPOT_CHECKS}

    for entity in data:
        search_name = norm(entity["search_name"])
        # Other mention strings in this document.  Following an.py, the entity's
        # own string is removed, so a repeated mention is not its own sibling.
        siblings = doc_names[entity["doc_key"]] - {search_name}
        if siblings:
            stats["entities_with_siblings"] += 1
        cue = 1.0 if ADMIN_CUE_RE.search(str(entity["search_name"])) else 0.0
        if cue:
            stats["entities_with_admin_cue"] += 1

        # (choice, population, exact_altname_match, is_correct) for the real rows
        real = []
        if len(entity["es_choices"]) != len(entity["correct"]):
            raise ValueError(
                "candidate list and correct list disagree for {!r}: {} vs {}".format(
                    entity["search_name"], len(entity["es_choices"]), len(entity["correct"])
                )
            )

        for choice, is_correct in zip(entity["es_choices"], entity["correct"]):
            choice["mention_admin_cue"] = cue
            if is_null_choice(choice):
                stats["null_rows"] += 1
                for key in NEW_KEYS:
                    if key != "mention_admin_cue":
                        choice[key] = null_value(key)
                continue

            stats["real_candidates"] += 1
            gid = str(choice["geonameid"])

            pop = pop_map.get(gid)
            if pop is None:
                pop = 0
                choice["log_population"] = 0.0
            else:
                stats["in_es"] += 1
                lp = math.log10(pop + 1)
                choice["log_population"] = lp
                if pop > 0:
                    stats["pop_gt_zero"] += 1
                if gid in spot_ids:
                    spot_seen[gid].add(round(lp, 6))

            name_match = 1.0 if norm(choice["name"]) == search_name else 0.0
            choice["exact_name_match"] = name_match
            stats["exact_name_match"] += name_match

            alt_match = 1.0 if search_name in altname_hits.get(gid, ()) else 0.0
            choice["exact_altname_match"] = alt_match
            stats["exact_altname_match"] += alt_match

            admin = 1.0 if choice.get("feature_class") == "A" else 0.0
            choice["is_admin_class"] = admin
            stats["admin_class"] += admin

            real.append((choice, pop, alt_match, bool(is_correct)))

        # Within-set features: how does this candidate compare to its rivals?
        max_pop = max((p for _, p, _, _ in real), default=0)
        max_log_pop = math.log10(max_pop + 1)
        alt_pops = [p for _, p, a, _ in real if a]
        max_alt_pop = max(alt_pops, default=0)

        if real and max_pop == 0:
            stats["sets_all_zero_pop"] += 1
        if max_pop > 0 and sum(1 for _, p, _, _ in real if p == max_pop) > 1:
            stats["sets_with_pop_tie"] += 1

        # Same-name candidates that sit on top of each other but differ in
        # feature class: an administrative unit and the settlement inside it.
        for choice, _, _, _ in real:
            choice["ap_twin"] = 0.0
            choice["ap_twin_stripped"] = 0.0
            choice["ap_twin_stripped_wide"] = 0.0
        by_name = defaultdict(list)
        by_strip = defaultdict(list)
        for choice, _, _, _ in real:
            by_name[norm(choice["name"])].append(choice)
            by_strip[strip_key(choice["name"])].append(choice)
        twin_here = False
        twin_at_mention = False
        for cand_name, group in by_name.items():
            if len(group) < 2:
                continue
            a_side = [c for c in group if c.get("feature_class") == "A"]
            p_side = [c for c in group if c.get("feature_class") == "P"]
            if not a_side or not p_side:
                continue
            for a in a_side:
                for b in p_side:
                    if (
                        abs(float(a["lat"]) - float(b["lat"])) < AP_TWIN_DEGREES
                        and abs(float(a["lon"]) - float(b["lon"])) < AP_TWIN_DEGREES
                    ):
                        a["ap_twin"] = 1.0
                        b["ap_twin"] = 1.0
                        twin_here = True
                        if cand_name == search_name:
                            twin_at_mention = True
        if twin_here:
            stats["entities_with_ap_twin"] += 1
        if twin_at_mention:
            stats["entities_with_ap_twin_at_mention"] += 1

        # Same geometry and A/P rule, looser name test.  Note this keeps
        # ap_twin's convention of not requiring a shared country, so the two
        # flags differ only in how names are compared.
        twin_stripped_here = False
        twin_wide_here = False
        for group in by_strip.values():
            if len(group) < 2:
                continue
            a_side = [c for c in group if c.get("feature_class") == "A"]
            p_side = [c for c in group if c.get("feature_class") == "P"]
            if not a_side or not p_side:
                continue
            for a in a_side:
                for b in p_side:
                    dlat = abs(float(a["lat"]) - float(b["lat"]))
                    dlon = abs(float(a["lon"]) - float(b["lon"]))
                    if dlat >= AP_TWIN_WIDE_DEGREES or dlon >= AP_TWIN_WIDE_DEGREES:
                        continue
                    a["ap_twin_stripped_wide"] = 1.0
                    b["ap_twin_stripped_wide"] = 1.0
                    twin_wide_here = True
                    if dlat < AP_TWIN_DEGREES and dlon < AP_TWIN_DEGREES:
                        a["ap_twin_stripped"] = 1.0
                        b["ap_twin_stripped"] = 1.0
                        twin_stripped_here = True
        if twin_stripped_here:
            stats["entities_with_ap_twin_stripped"] += 1
        if twin_wide_here:
            stats["entities_with_ap_twin_stripped_wide"] += 1

        for choice, pop, alt_match, is_correct in real:
            choice["has_population"] = 1.0 if pop > 0 else 0.0
            is_max = 1.0 if (max_pop > 0 and pop == max_pop) else 0.0
            choice["is_max_pop"] = is_max
            choice["log_pop_rel"] = choice["log_population"] - max_log_pop
            is_max_alt = (
                1.0 if (alt_match and max_alt_pop > 0 and pop == max_alt_pop) else 0.0
            )
            choice["is_max_pop_exact_match"] = is_max_alt

            # Does another mention in the document name this candidate's parent?
            adm1 = choice.get("admin1_name")
            adm2 = choice.get("admin2_name")
            sib_adm1 = (
                1.0 if (norm(adm1) in siblings and adm1 not in ("", "NULL")) else 0.0
            )
            sib_adm2 = (
                1.0 if (norm(adm2) in siblings and adm2 not in ("", "NULL")) else 0.0
            )
            country = country_names.get(choice.get("country_code3"), "~~")
            sib_country = 1.0 if country in siblings else 0.0
            choice["sib_adm1"] = sib_adm1
            choice["sib_adm2"] = sib_adm2
            choice["sib_country"] = sib_country

            choice["is_historical"] = (
                1.0 if str(choice.get("feature_code")) in HISTORICAL_CODES else 0.0
            )
            # an.py's "hard set": name-matching candidates in docs with siblings.
            hard = bool(siblings) and choice["exact_name_match"] == 1.0
            if hard:
                stats["hard_candidates"] += 1
                if is_correct:
                    stats["hard_gold"] += 1
            for key in CONTEXT_KEYS + ROUND5_KEYS + ROUND6_KEYS:
                val = choice[key]
                if not val:
                    continue
                stats[key] += 1
                if is_correct:
                    stats[key + "_gold"] += 1
                if hard:
                    stats[key + "_hard"] += 1
                    if is_correct:
                        stats[key + "_hard_gold"] += 1

            stats["is_max_pop"] += is_max
            stats["is_max_pop_exact_match"] += is_max_alt
            if is_correct:
                stats["correct_candidates"] += 1
                stats["is_max_pop_correct"] += is_max
                stats["is_max_alt_correct"] += is_max_alt
            else:
                stats["incorrect_candidates"] += 1
                stats["is_max_pop_incorrect"] += is_max
                stats["is_max_alt_incorrect"] += is_max_alt

    add_cross_entity_features(data, doc_members, es_names, stats)
    return stats, spot_seen


def add_cross_entity_features(data, doc_members, es_names, stats):
    """Sibling-anchor geometry, set-shape gates and case-folded distances.

    Definitions are copied from the screen's extract_feats.py (geometry, set
    shape) and anchor_variants.py (the soft per-sibling variant) so the trained
    features are the screened ones.
    """
    # Resolve every mention provisionally with the population prior.  This has to
    # happen for the whole source before any entity can look at its siblings.
    prepped = [None] * len(data)
    anchors = [None] * len(data)
    tops = [None] * len(data)
    for i, entity in enumerate(data):
        d = prep_entity(entity)
        prepped[i] = d
        if d["n"] == 0:
            continue
        score = d["exact"] * 100.0 + d["pop"]
        k = int(np.argmax(score))
        anchors[i] = (d["lat"][k], d["lon"][k], d["cc"][k], d["a1"][k], d["a2"][k])
        # Stable: the score ties constantly (every exact-name match with no
        # population scores exactly 100.0), and an unstable sort would let the
        # numpy version decide which tied candidates make the cut.
        idx = np.argsort(-score, kind="stable")[:ANCHOR_TOPK]
        tops[i] = (d["lat"][idx], d["lon"][idx])

    for i, entity in enumerate(data):
        d = prepped[i]
        n = d["n"]
        if n == 0:
            for choice in entity["es_choices"]:
                for key in GEOMETRY_KEYS + SETSHAPE_KEYS + CASEFOLD_KEYS:
                    choice[key] = null_value(key)
            continue
        stats["entities_with_candidates"] += 1
        m_low = norm(entity["search_name"])

        # Siblings: other mentions in the document with a different string that
        # the prior could resolve.
        sib_idx = [
            j
            for j in doc_members[entity["doc_key"]]
            if j != i
            and anchors[j] is not None
            and norm(data[j]["search_name"]) != m_low
        ]
        na = len(sib_idx)
        stats["anchor_total"] += na
        if na:
            stats["entities_with_anchor"] += 1
            anc = [anchors[j] for j in sib_idx]
            alat = np.array([a[0] for a in anc])
            alon = np.array([a[1] for a in anc])
            km = haversine(d["lat"], d["lon"], alat, alon)
            min_km = km.min(axis=1)
            f50 = (km < 50).mean(axis=1)
            f150 = (km < 150).mean(axis=1)
            cc_ct = Counter(a[2] for a in anc)
            a1_ct = Counter((a[2], a[3]) for a in anc if a[3])
            a2_ct = Counter((a[2], a[3], a[4]) for a in anc if a[3] and a[4])
            same_cc = np.array([cc_ct.get(c, 0) / na for c in d["cc"]])
            same_a1 = np.array(
                [
                    a1_ct.get((c, a), 0) / na if a else 0.0
                    for c, a in zip(d["cc"], d["a1"])
                ]
            )
            par = np.zeros(n)
            inside = np.zeros(n)
            for k in range(n):
                fc = d["fc"][k]
                if fc in PCL_CODES:
                    cnt = cc_ct.get(d["cc"][k], 0)
                elif fc.startswith("ADM1") and d["a1"][k]:
                    cnt = a1_ct.get((d["cc"][k], d["a1"][k]), 0)
                elif fc.startswith("ADM2") and d["a2"][k]:
                    cnt = a2_ct.get((d["cc"][k], d["a1"][k], d["a2"][k]), 0)
                else:
                    cnt = 0
                par[k] = 1.0 if cnt else 0.0
                inside[k] = cnt / na
            # soft variant: distance to each sibling's nearest plausible candidate
            per_sib = np.zeros((n, na))
            for si, j in enumerate(sib_idx):
                sib_lat, sib_lon = tops[j]
                per_sib[:, si] = haversine(d["lat"], d["lon"], sib_lat, sib_lon).min(
                    axis=1
                )
            mean_sibmin = per_sib.mean(axis=1)
            fsib50 = (per_sib < 50).mean(axis=1)
        else:
            min_km = np.full(n, NO_ANCHOR_KM)
            f50 = np.zeros(n)
            f150 = np.zeros(n)
            same_cc = np.zeros(n)
            same_a1 = np.zeros(n)
            par = np.zeros(n)
            inside = np.zeros(n)
            mean_sibmin = np.full(n, NO_ANCHOR_KM)
            fsib50 = np.zeros(n)

        # ---- set shape ----
        name_ct = Counter(d["name"])
        n_exact = int(d["exact"].sum())
        log_n_exact = math.log10(n_exact + 1)

        # ---- case-folded edit distances, res_formatter's recipe on lowered text
        search_low = str(entity["search_name"]).lower()
        min_cf, max_cf, avg_cf, ascii_cf = [], [], [], []
        for choice in d["cands"]:
            names_lc, ascii_lc = es_names.get(
                str(choice["geonameid"]), ((norm(choice["name"]),), norm(choice["name"]))
            )
            dists = [jellyfish.levenshtein_distance(search_low, x) for x in names_lc]
            min_cf.append(min(dists))
            max_cf.append(max(dists))
            avg_cf.append(sum(dists) / len(dists))
            ascii_cf.append(jellyfish.levenshtein_distance(search_low, ascii_lc))
        min_cf = normalize_distances(min_cf)
        max_cf = normalize_distances(max_cf)
        avg_cf = normalize_distances(avg_cf)
        ascii_cf = normalize_distances(ascii_cf)

        for k, choice in enumerate(d["cands"]):
            choice["log_min_km_anchor"] = math.log10(min_km[k] + 1)
            choice["anchor_same_adm1_frac"] = same_a1[k]
            choice["frac_anchors_150km"] = f150[k]
            choice["frac_anchors_50km"] = f50[k]
            choice["log_mean_sibmin"] = math.log10(mean_sibmin[k] + 1)
            choice["frac_sibs_50km"] = fsib50[k]
            choice["anchor_same_country_frac"] = same_cc[k]
            choice["is_parent_of_anchor"] = par[k]
            choice["frac_anchors_inside"] = inside[k]

            choice["log_n_same_name"] = math.log10(name_ct[d["name"][k]])
            choice["log_n_exact_matches"] = log_n_exact
            choice["is_unique_exact_match"] = (
                1.0 if (d["exact"][k] > 0 and n_exact == 1) else 0.0
            )
            choice["is_seat_any"] = 1.0 if d["fc"][k] in SEAT_CODES else 0.0

            choice["min_dist_cf"] = min_cf[k]
            choice["max_dist_cf"] = max_cf[k]
            choice["avg_dist_cf"] = avg_cf[k]
            choice["ascii_dist_cf"] = ascii_cf[k]

        for choice in entity["es_choices"]:
            if is_null_choice(choice):
                for key in GEOMETRY_KEYS + SETSHAPE_KEYS + CASEFOLD_KEYS:
                    choice[key] = null_value(key)


#
#   Validation
#


def validate(original_path, enriched_path_, limit, sample=400):
    """Re-read both files and confirm the enriched copy only added the new keys."""
    original = load_pickle(original_path, limit)
    enriched = load_pickle(enriched_path_, limit)
    problems = []

    if len(original) != len(enriched):
        problems.append(
            "length differs: {} vs {}".format(len(original), len(enriched))
        )
        return problems

    if fingerprint(original) != fingerprint(enriched):
        problems.append("fingerprint (geonameid sequence / correct arrays) differs")

    for i, (orig_ent, new_ent) in enumerate(zip(original, enriched)):
        if new_ent.get("doc_key") != document_key(orig_ent):
            problems.append("entity {}: doc_key missing or wrong".format(i))
            break
        if len(orig_ent["es_choices"]) != len(new_ent["es_choices"]):
            problems.append("entity {}: candidate count differs".format(i))
            break
        if list(orig_ent["correct"]) != list(new_ent["correct"]):
            problems.append("entity {}: correct array differs".format(i))
            break
        for choice in new_ent["es_choices"]:
            for key in NEW_KEYS:
                if key not in choice:
                    problems.append(
                        "entity {}: candidate missing key {}".format(i, key)
                    )
                    break
        if problems:
            break

    # Deep-compare a sample: every pre-existing key/value must be untouched.
    step = max(1, len(original) // sample)
    for i in range(0, len(original), step):
        orig_ent, new_ent = original[i], enriched[i]
        for key in orig_ent:
            if key == "es_choices":
                continue
            ov, nv = orig_ent[key], new_ent[key]
            if hasattr(ov, "shape"):
                if ov.shape != nv.shape or not (ov == nv).all():
                    problems.append("entity {}: field {} changed".format(i, key))
            elif ov != nv:
                problems.append("entity {}: field {} changed".format(i, key))
        for j, (oc, nc) in enumerate(zip(orig_ent["es_choices"], new_ent["es_choices"])):
            for key, val in oc.items():
                if nc.get(key) != val:
                    problems.append(
                        "entity {} candidate {}: field {} changed".format(i, j, key)
                    )
            extra = set(nc) - set(oc) - set(NEW_KEYS)
            if extra:
                problems.append(
                    "entity {} candidate {}: unexpected keys {}".format(i, j, sorted(extra))
                )
        if problems:
            break

    del original, enriched
    return problems


def spot_check_report(es):
    """Fetch a handful of well-known places and check the populations look sane."""
    ids = [gid for gid, _, _, _ in SPOT_CHECKS]
    resp = es.mget(
        index="geonames",
        body={"ids": ids},
        _source_includes=["population", "name"],
        request_timeout=60,
    )
    found = {d["_id"]: d for d in resp["docs"]}
    rows = []
    for gid, label, low, high in SPOT_CHECKS:
        doc = found.get(gid, {})
        if not doc.get("found"):
            rows.append((gid, label, None, None, "MISSING FROM INDEX"))
            continue
        pop = int(doc["_source"].get("population") or 0)
        lp = math.log10(pop + 1)
        ok = "ok" if low <= pop <= high else "OUT OF RANGE ({:,}-{:,})".format(low, high)
        rows.append((gid, label, pop, lp, ok))
    return rows


#
#   The outlet block: a separate pass over the already-enriched pickles
#


def permute_homes(homes, seed):
    """Give every outlet a *different* outlet's home, keeping the masks fixed.

    This is the control that separates the two things the outlet block could be
    doing.  It is a locality prior -- "the gold is near this newsroom" -- but it
    is also, unavoidably, a corpus indicator: ``has_outlet_home`` is 1 for LGL
    and TR and 0 for the four sources that have no outlet metadata, and the model
    is already known to exploit corpus identity (it recovers the corpus 93.3% of
    the time from ``doc_tensor`` alone) to switch annotation conventions.

    Permuting *within* the point-home group and *within* the country-home group
    leaves ``has_outlet_home`` and ``has_outlet_country`` bit-identical for every
    entity in the corpus, and leaves the marginal distribution of distances and
    same-adm1 rates almost unchanged.  The only thing destroyed is the
    correspondence between an article and its own newsroom.  Whatever survives
    this permutation is not locality.
    """
    import random as _random

    rng = _random.Random(seed)
    out = dict(homes)
    for level in ("point", "country"):
        keys = sorted(k for k, v in homes.items() if v.get("level") == level)
        vals = [homes[k] for k in keys]
        # A derangement: no outlet may keep its own home.
        for _attempt in range(1000):
            order = list(range(len(vals)))
            rng.shuffle(order)
            if all(i != j for i, j in enumerate(order)) or len(vals) < 2:
                break
        for k, j in zip(keys, order):
            out[k] = vals[j]
    return out


def outlet_pass(data_dir, out_dir, sources, suffix, es, cache_path, limit=None,
                permute_seed=None, table_name="researched"):
    """Add the ``outlet`` block to enriched pickles and write compacted caches.

    Reads ``{data_dir}/es_formatted_{source}..._enriched.pkl``, attaches the five
    outlet columns to every candidate (including the NULL placeholder row), and
    writes ``..._enriched_compact.pkl`` into ``out_dir`` -- which is what
    ``tools/train.py --data-dir`` picks up.  The other 33 features are copied
    through untouched, which is what makes the baseline-vs-outlet contrast a
    one-variable comparison.

    Only ``lgl`` and ``tr`` have outlet metadata; every other source is given the
    well-defined null.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from outlet_align import entity_domains
    from outlet_home_table import load_or_build, researched_table
    from train import compact_candidates

    from mordecai3.outlet_features import add_outlet_features, clear_outlet_features

    # The researched table is canonical (e53 §10): it was rebuilt from public
    # sources with a URL per row, resolves more domains than the curated one,
    # and a model trained on either scores the same on the other to within
    # 0.0007 LGL EM. `curated` is kept only so the e50 runs stay reproducible.
    if os.environ.get("OUTLET_TABLE"):
        table_name = os.environ["OUTLET_TABLE"]
    if table_name == "curated":
        table = None
        print("using the CURATED outlet table (provenance only)", flush=True)
    elif table_name == "researched":
        table = researched_table()
        print("using the INDEPENDENTLY RESEARCHED outlet table", flush=True)
    else:
        sys.exit("unknown --outlet-table {!r}".format(table_name))
    homes = load_or_build(es, cache_path, table=table)
    print("resolved {} outlet homes ({} with a point)".format(
        len(homes), sum(1 for h in homes.values() if h.get("level") == "point")),
        flush=True)
    if permute_seed is not None:
        homes = permute_homes(homes, permute_seed)
        print("CONTROL: homes permuted within level (seed {}) -- masks unchanged, "
              "locality destroyed".format(permute_seed), flush=True)

    # The corpus XMLs sit one level above pickled_es.
    corpus_root = os.path.dirname(os.path.abspath(data_dir))
    os.makedirs(out_dir, exist_ok=True)

    for source in sources:
        t0 = time.time()
        in_path = enriched_path(data_dir, source, suffix)
        if not os.path.exists(in_path):
            sys.exit("missing enriched pickle: {}".format(in_path))
        data = load_pickle(in_path, limit)

        domains = entity_domains(data, source, corpus_root)
        n_point = n_country = n_none = 0
        for i, entity in enumerate(data):
            home = homes.get(domains.get(i)) if domains else None
            if home is None:
                clear_outlet_features(entity["es_choices"])
                n_none += 1
            else:
                add_outlet_features(entity["es_choices"], home)
                if home.get("level") == "point":
                    n_point += 1
                else:
                    n_country += 1

        compact_candidates(data)
        out_path = enriched_path(out_dir, source, suffix + "_compact")
        with open(out_path, "wb") as f:
            pickle.dump(data, f, protocol=4)
        print("  {:<12} {:>6,} entities  point {:>6,}  country {:>5,}  none {:>6,}"
              "  -> {} ({:.2f} GB, {:.0f}s)".format(
                  source, len(data), n_point, n_country, n_none,
                  os.path.basename(out_path),
                  os.path.getsize(out_path) / 1e9, time.time() - t0), flush=True)
        del data
    print("outlet pass complete", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", default=None, help="defaults to --data-dir")
    parser.add_argument("--sources", nargs="+", default=SOURCES)
    parser.add_argument("--suffix", default="_enriched")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument(
        "--limit", type=int, default=None, help="only process the first N entities (smoke test)"
    )
    parser.add_argument("--skip-validate", action="store_true")
    parser.add_argument(
        "--outlet-only", action="store_true",
        help="add only the outlet block to already-enriched pickles, and write "
             "the compacted cache (see the module docstring)")
    parser.add_argument(
        "--permute-homes", type=int, default=None, metavar="SEED",
        help="control arm: give each outlet another outlet's home, keeping both "
             "mask channels bit-identical (see permute_homes)")
    parser.add_argument(
        "--outlet-home-cache",
        default=os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "outlet_homes.json"),
        help="where the geocoded outlet homes are cached")
    parser.add_argument(
        "--outlet-table", choices=["researched", "curated"], default="researched",
        help="which domain->home table to geocode; 'researched' (the default, "
             "tools/data/outlet_homes_researched.tsv) is canonical, 'curated' "
             "is the e50 table kept for provenance")
    args = parser.parse_args()

    out_dir = args.out_dir or args.data_dir
    os.makedirs(out_dir, exist_ok=True)

    if args.outlet_only:
        es = setup_es_client()
        if not es.ping():
            sys.exit("cannot reach Elasticsearch at localhost:9200")
        outlet_pass(args.data_dir, out_dir, args.sources, args.suffix, es,
                    args.outlet_home_cache, args.limit, args.permute_homes,
                    table_name=args.outlet_table)
        return

    paths = {}
    for source in args.sources:
        path = source_path(args.data_dir, source)
        if not os.path.exists(path):
            sys.exit("missing input pickle: {}".format(path))
        paths[source] = path

    country_names = load_country_names()
    print("loaded {} country names".format(len(country_names)), flush=True)

    es = setup_es_client()
    if not es.ping():
        sys.exit("cannot reach Elasticsearch at localhost:9200")

    t_start = time.time()

    print("== pass 1: scanning pickles for unique geonameids ==", flush=True)
    all_ids = set()
    all_names = set()
    slot_counts = {}
    for source in args.sources:
        t0 = time.time()
        ids, names, slots = scan_source(paths[source], args.limit)
        all_ids |= ids
        all_names |= names
        slot_counts[source] = slots
        print(
            "  {:<12} {:>10,} candidate slots, {:>9,} unique ids ({:.1f}s)".format(
                source, slots, len(ids), time.time() - t0
            ),
            flush=True,
        )
    print(
        "  union: {:,} unique geonameids, {:,} unique mention strings".format(
            len(all_ids), len(all_names)
        ),
        flush=True,
    )

    print("== pass 2: fetching population and names from Elasticsearch ==", flush=True)
    t_fetch = time.time()
    pop_map, altname_hits, es_names, n_missing = fetch_es_features(
        es, all_ids, all_names, chunk_size=args.chunk_size
    )
    fetch_secs = time.time() - t_fetch
    del all_ids, all_names

    print("== pass 3: enriching and writing ==", flush=True)
    all_stats = {}
    all_problems = {}
    spot_observed = defaultdict(set)
    for source in args.sources:
        t0 = time.time()
        data = load_pickle(paths[source], args.limit)
        stats, spot_seen = enrich_source(
            data, pop_map, altname_hits, es_names, country_names
        )
        for gid, vals in spot_seen.items():
            spot_observed[gid] |= vals
        out_path = enriched_path(out_dir, source, args.suffix)
        with open(out_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        del data
        stats["seconds"] = time.time() - t0
        stats["out_path"] = out_path
        stats["size_gb"] = os.path.getsize(out_path) / 1e9
        all_stats[source] = stats
        print(
            "  {:<12} -> {} ({:.2f} GB, {:.1f}s)".format(
                source, os.path.basename(out_path), stats["size_gb"], stats["seconds"]
            ),
            flush=True,
        )

        if not args.skip_validate:
            t1 = time.time()
            problems = validate(paths[source], out_path, args.limit)
            all_problems[source] = problems
            print(
                "     validate: {} ({:.1f}s)".format(
                    "OK" if not problems else "FAILED: " + "; ".join(problems[:3]),
                    time.time() - t1,
                ),
                flush=True,
            )

    print()
    print("== spot checks (population straight from ES) ==", flush=True)
    for gid, label, pop, lp, ok in spot_check_report(es):
        pop_s = "{:,}".format(pop) if pop is not None else "-"
        lp_s = "{:.3f}".format(lp) if lp is not None else "-"
        seen = spot_observed.get(gid)
        in_pkl = ""
        if seen:
            in_pkl = " | in pickles as log_pop={}".format(
                ", ".join("{:.3f}".format(v) for v in sorted(seen))
            )
        print(
            "  {:<10} {:<16} pop={:>12} log10={:<7} {}{}".format(
                gid, label, pop_s, lp_s, ok, in_pkl
            ),
            flush=True,
        )

    print()
    print("== per-source stats ==", flush=True)
    header = "{:<12} {:>8} {:>12} {:>9} {:>9} {:>9} {:>9} {:>9}".format(
        "source", "entities", "real_cands", "%pop>0", "%in_es", "mean_exact", "mean_alt",
        "%A_class",
    )
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for source in args.sources:
        s = all_stats[source]
        rc = max(s["real_candidates"], 1)
        print(
            "{:<12} {:>8,} {:>12,} {:>8.1f}% {:>8.1f}% {:>9.4f} {:>9.4f} {:>8.1f}%".format(
                source,
                s["entities"],
                s["real_candidates"],
                100.0 * s["pop_gt_zero"] / rc,
                100.0 * s["in_es"] / rc,
                s["exact_name_match"] / rc,
                s["exact_altname_match"] / rc,
                100.0 * s["admin_class"] / rc,
            ),
            flush=True,
        )
    print(flush=True)
    print("{:<12} {:>10} {:>18}".format("source", "null_rows", "%ents admin_cue"), flush=True)
    for source in args.sources:
        s = all_stats[source]
        print(
            "{:<12} {:>10,} {:>17.1f}%".format(
                source,
                s["null_rows"],
                100.0 * s["entities_with_admin_cue"] / max(s["entities"], 1),
            ),
            flush=True,
        )

    print(flush=True)
    print("== within-set features: correct vs incorrect candidates ==", flush=True)
    header2 = "{:<12} {:>10} {:>10} {:>10} {:>10} {:>9} {:>9}".format(
        "source", "maxpop_c", "maxpop_i", "maxpopnm_c", "maxpopnm_i", "0pop_sets", "tie_sets"
    )
    print(header2, flush=True)
    print("-" * len(header2), flush=True)
    for source in args.sources:
        s = all_stats[source]
        nc = max(s["correct_candidates"], 1)
        ni = max(s["incorrect_candidates"], 1)
        ne = max(s["entities"], 1)
        print(
            "{:<12} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f} {:>8.1f}% {:>8.1f}%".format(
                source,
                s["is_max_pop_correct"] / nc,
                s["is_max_pop_incorrect"] / ni,
                s["is_max_alt_correct"] / nc,
                s["is_max_alt_incorrect"] / ni,
                100.0 * s["sets_all_zero_pop"] / ne,
                100.0 * s["sets_with_pop_tie"] / ne,
            ),
            flush=True,
        )

    print(flush=True)
    print("== document context features ==", flush=True)
    header3 = "{:<12} {:>7} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}".format(
        "source", "docs", "ents/doc", "%sib_ents", "fire_a1", "fire_a2", "fire_cty", "fire_ap"
    )
    print(header3, flush=True)
    print("-" * len(header3), flush=True)
    for source in args.sources:
        st = all_stats[source]
        rc = max(st["real_candidates"], 1)
        print(
            "{:<12} {:>7,} {:>9.1f} {:>8.1f}% {:>8.3f}% {:>8.3f}% {:>8.2f}% {:>8.2f}%".format(
                source,
                st["documents"],
                st["entities"] / max(st["documents"], 1),
                100.0 * st["entities_with_siblings"] / max(st["entities"], 1),
                100.0 * st["sib_adm1"] / rc,
                100.0 * st["sib_adm2"] / rc,
                100.0 * st["sib_country"] / rc,
                100.0 * st["ap_twin"] / rc,
            ),
            flush=True,
        )

    print(flush=True)
    print("== P(gold | feature = 1), hard set = name-matching candidates in docs with siblings ==", flush=True)
    header4 = "{:<12} {:>10}".format("source", "hard_base") + "".join(
        " {:>15}".format(k) for k in CONTEXT_KEYS + ROUND5_KEYS
    )
    print(header4, flush=True)
    print("-" * len(header4), flush=True)
    totals = defaultdict(int)
    for source in args.sources:
        st = all_stats[source]
        cells = []
        for key in CONTEXT_KEYS + ROUND5_KEYS:
            n = st[key + "_hard"]
            cells.append(
                "{:.3f} (n={:,})".format(st[key + "_hard_gold"] / n, n) if n else "-"
            )
            totals[key] += n
            totals[key + "_gold"] += st[key + "_hard_gold"]
        totals["hard"] += st["hard_candidates"]
        totals["hard_gold"] += st["hard_gold"]
        base = st["hard_gold"] / max(st["hard_candidates"], 1)
        print(
            "{:<12} {:>10.3f}".format(source, base)
            + "".join(" {:>15}".format(c) for c in cells),
            flush=True,
        )
    print(
        "{:<12} {:>10.3f}".format("POOLED", totals["hard_gold"] / max(totals["hard"], 1))
        + "".join(
            " {:>15}".format(
                "{:.3f} (n={:,})".format(totals[k + "_gold"] / totals[k], totals[k])
                if totals[k]
                else "-"
            )
            for k in CONTEXT_KEYS + ROUND5_KEYS
        ),
        flush=True,
    )
    print(
        "  entities with an A/P twin pair: "
        + ", ".join(
            "{} {:.1f}%".format(
                src,
                100.0
                * all_stats[src]["entities_with_ap_twin"]
                / max(all_stats[src]["entities"], 1),
            )
            for src in args.sources
        ),
        flush=True,
    )
    print(
        "  entities with an A/P twin at the mention name (an2.py convention): "
        + ", ".join(
            "{} {:.1f}%".format(
                src,
                100.0
                * all_stats[src]["entities_with_ap_twin_at_mention"]
                / max(all_stats[src]["entities"], 1),
            )
            for src in args.sources
        ),
        flush=True,
    )

    print(flush=True)
    print("== round 5: stripped-name twins and historical rows ==", flush=True)
    header6 = "{:<12} {:>11} {:>11} {:>11} {:>10} {:>10} {:>11} {:>11} {:>10}".format(
        "source", "fire_twin", "fire_strip", "fire_wide", "ents_strip", "ents_wide",
        "strip_golds", "wide_golds", "hist_fire",
    )
    print(header6, flush=True)
    print("-" * len(header6), flush=True)
    for source in args.sources:
        st = all_stats[source]
        rc = max(st["real_candidates"], 1)
        ng = max(st["correct_candidates"], 1)
        print(
            "{:<12} {:>10.2f}% {:>10.2f}% {:>10.2f}% {:>9.1f}% {:>9.1f}% {:>10.2f}% "
            "{:>10.2f}% {:>9.3f}%".format(
                source,
                100.0 * st["ap_twin"] / rc,
                100.0 * st["ap_twin_stripped"] / rc,
                100.0 * st["ap_twin_stripped_wide"] / rc,
                100.0 * st["entities_with_ap_twin_stripped"] / max(st["entities"], 1),
                100.0 * st["entities_with_ap_twin_stripped_wide"] / max(st["entities"], 1),
                100.0 * st["ap_twin_stripped_gold"] / ng,
                100.0 * st["ap_twin_stripped_wide_gold"] / ng,
                100.0 * st["is_historical"] / rc,
            ),
            flush=True,
        )
    tot_h = sum(all_stats[s]["is_historical_gold"] for s in args.sources)
    tot_g = sum(all_stats[s]["correct_candidates"] for s in args.sources)
    print(
        "  is_historical fires on {:,} of {:,} gold candidates ({:.3f}%)".format(
            tot_h, tot_g, 100.0 * tot_h / max(tot_g, 1)
        ),
        flush=True,
    )

    print(flush=True)
    print("== sibling-anchor coverage ==", flush=True)
    header5 = "{:<12} {:>10} {:>14} {:>14}".format(
        "source", "entities", "%>=1 anchor", "mean anchors"
    )
    print(header5, flush=True)
    print("-" * len(header5), flush=True)
    for source in args.sources:
        st = all_stats[source]
        ne = max(st["entities_with_candidates"], 1)
        print(
            "{:<12} {:>10,} {:>13.1f}% {:>14.1f}".format(
                source,
                st["entities_with_candidates"],
                100.0 * st["entities_with_anchor"] / ne,
                st["anchor_total"] / ne,
            ),
            flush=True,
        )

    print(flush=True)
    failed = {k: v for k, v in all_problems.items() if v}
    if failed:
        print("VALIDATION FAILURES:", flush=True)
        for source, problems in failed.items():
            for problem in problems[:10]:
                print("  {}: {}".format(source, problem), flush=True)
    else:
        print("validation: all sources OK", flush=True)
    print(
        "geonameids not found in ES: {:,} of {:,} unique".format(
            n_missing, n_missing + len(pop_map)
        ),
        flush=True,
    )
    print(
        "wall clock: {:.1f}s total ({:.1f}s of it Elasticsearch)".format(
            time.time() - t_start, fetch_secs
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
