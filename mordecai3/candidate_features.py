"""Extra ranking features for Geonames candidates, computed at inference time.

The ranking model can be trained with a set of features that the plain
Elasticsearch lookup does not produce: how big a place is, whether the mention
string is literally one of its names, and how it sits relative to the other
place names in the same document.  Those features were originally added offline
by ``tools/enrich_pickles.py``, which rewrote the cached candidate pickles the
trainer reads.  A model trained on them can only be served if the live pipeline
computes the same numbers, so the definitions live here and both paths use them:

* ``geoparse.res_formatter`` calls :func:`add_entity_features` on each fresh
  candidate list (everything that depends only on the mention string and its own
  candidate set), and
* ``geoparse.add_es_data_batch`` calls :func:`add_document_features` once every
  entity in a document has its candidates back (everything that reads the other
  mentions in the document).

``tests/test_feature_parity.py`` drives these two functions with the candidate
lists out of the enriched training pickles and asserts the values match what
``enrich_pickles.py`` wrote, to 1e-9.  That test is the contract: the definitions
below are deliberately literal transcriptions of that script -- same sentinels,
same stable sort, same "no anchor" distance -- and should not be "improved"
without retraining.

The feature groups, in the order ``torch_model.FEATURE_BLOCKS`` names them:

prom
    ``log_population``, ``has_population``, ``is_max_pop``, ``log_pop_rel``,
    ``is_max_pop_exact_match`` -- how prominent this candidate is, absolutely
    and relative to the rest of its set.
name
    ``exact_name_match``, ``exact_altname_match`` -- is the mention literally
    one of this place's names?
cue
    ``mention_admin_cue``, ``is_admin_class`` -- does the mention announce an
    administrative unit ("Fairfax County"), and is the candidate one?
sib
    ``sib_adm1``, ``sib_adm2``, ``sib_country``, ``ap_twin`` -- does another
    mention in the document name this candidate's parent unit?
geo
    ``log_min_km_anchor`` and friends -- where the candidate sits relative to
    the *anchors* of the document's other mentions, each anchor being that
    mention's population-prior pick.
shape
    ``log_n_same_name``, ``log_n_exact_matches``, ``is_unique_exact_match``,
    ``is_seat_any`` -- how ambiguous the mention is.
cf
    ``min_dist_cf`` and friends -- the ``res_formatter`` edit distances redone
    with both sides lowercased.

The placeholder "no correct answer" row that is appended to every candidate list
gets the same keys with neutral values (see :data:`NULL_SENTINELS`) so the
per-entity feature arrays stay rectangular and the placeholder never looks like
the best-supported option.
"""

import csv
import functools
import math
import os
import re
from collections import Counter, defaultdict

import jellyfish
import numpy as np

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

# Features that compare a candidate against the rest of its entity's set.
WITHIN_SET_KEYS = [
    "has_population",
    "is_max_pop",
    "log_pop_rel",
    "is_max_pop_exact_match",
]

# Features that read the rest of the document, or pair candidates up.
CONTEXT_KEYS = [
    "sib_adm1",
    "sib_adm2",
    "sib_country",
    "ap_twin",
]

# Sibling-anchor geometry.
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

# Everything add_entity_features can compute on its own: the candidate, its
# rivals, and the mention string.
ENTITY_KEYS = BASE_KEYS + WITHIN_SET_KEYS + ["ap_twin"] + SETSHAPE_KEYS + CASEFOLD_KEYS

# Everything that needs the other mentions in the document.
DOCUMENT_KEYS = ["sib_adm1", "sib_adm2", "sib_country"] + GEOMETRY_KEYS

ALL_KEYS = (
    BASE_KEYS + WITHIN_SET_KEYS + CONTEXT_KEYS
    + GEOMETRY_KEYS + SETSHAPE_KEYS + CASEFOLD_KEYS
)

PCL_CODES = {"PCLI", "PCL", "PCLD", "PCLS", "PCLF", "PCLIX", "TERR"}
SEAT_CODES = {"PPLC", "PPLA", "PPLA2", "PPLA3", "PPLA4", "PPLA5", "PPLG"}

# How many candidates of a sibling count as "plausible" for the soft variant.
ANCHOR_TOPK = 20

# Stand-in distance when the document offers no usable sibling anchor.
NO_ANCHOR_KM = 20000.0

# Degrees of latitude/longitude within which an A and a P candidate of the same
# name are treated as the same place seen twice.
AP_TWIN_DEGREES = 0.15

# What the NULL placeholder row gets.  0.0 is the right filler for a fraction or
# an indicator -- it reads as "no evidence" -- but it is the *best* possible
# value for a distance, and the placeholder competes with the real candidates in
# the softmax.  Distances therefore get the same sentinel a real candidate gets
# when the document offers nothing.
NULL_SENTINELS = {
    "log_min_km_anchor": math.log10(NO_ANCHOR_KM + 1),
    "log_mean_sibmin": math.log10(NO_ANCHOR_KM + 1),
    "min_dist_cf": 1.0,
    "max_dist_cf": 1.0,
    "avg_dist_cf": 1.0,
    "ascii_dist_cf": 1.0,
}

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

COUNTRY_CODES_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "assets",
    "wikipedia-iso-country-codes.txt",
)


def null_value(key):
    """The value the NULL placeholder row carries for one enrichment feature."""
    return NULL_SENTINELS.get(key, 0.0)


def norm(text):
    """Lowercase/strip a name for exact-match comparisons."""
    return str(text).strip().lower()


def is_null_choice(choice):
    return str(choice.get("geonameid", "")).strip().upper() == NULL_GEONAMEID


def clean_code(v):
    """Admin codes as the geometry treats them: placeholders become empty."""
    v = str(v).strip()
    return "" if v in ("", "NULL", "00", "None") else v


@functools.lru_cache(maxsize=1)
def load_country_names(path=COUNTRY_CODES_PATH):
    """Map alpha-3 country code -> lowercased English short name."""
    with open(path, encoding="utf8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames
        name_col = [c for c in cols if "name" in c.lower() or "Country" in c][0]
        alpha3_col = [c for c in cols if "Alpha-3" in c][0]
        return {row[alpha3_col]: norm(row[name_col]) for row in reader}


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


def fill_null_features(choice, mention_admin_cue=0.0):
    """Give the "no correct answer" placeholder row its neutral feature values.

    ``mention_admin_cue`` is a property of the mention, not of the candidate, so
    the placeholder carries the real value like every other row.
    """
    for key in ALL_KEYS:
        choice[key] = null_value(key)
    choice["mention_admin_cue"] = mention_admin_cue
    return choice


def mention_admin_cue(search_name):
    """1.0 if the mention string contains a word like "county" or "province"."""
    return 1.0 if ADMIN_CUE_RE.search(str(search_name)) else 0.0


def _source_names(source):
    """(all lowercased names, lowercased asciiname, alternate/ascii name set).

    ``source`` is a Geonames ``_source`` dict as it comes back from
    Elasticsearch.  The third element is what ``exact_altname_match`` tests
    against, and deliberately excludes ``name`` -- that is what
    ``exact_name_match`` is for.
    """
    name = source.get("name") or ""
    ascii_name = source.get("asciiname") or ""
    alt = source.get("alternativenames") or []
    if isinstance(alt, str):
        alt = [alt]
    alt_or_ascii = set()
    if ascii_name:
        alt_or_ascii.add(norm(ascii_name))
    alt_or_ascii.update(norm(a) for a in alt)
    names_lc = tuple(str(x).lower() for x in [name] + list(alt))
    return names_lc, str(ascii_name).lower(), alt_or_ascii


def _population(source):
    try:
        return int(source.get("population") or 0)
    except (TypeError, ValueError):
        return 0


def add_entity_features(search_name, choices, sources):
    """Attach every feature that only needs this mention and its own candidates.

    Parameters
    ----------
    search_name : str
        The raw mention string from the document (not the cleaned query string).
    choices : list of dict
        Formatted candidates, as ``res_formatter`` builds them.  The NULL
        placeholder row must NOT be in this list; use :func:`fill_null_features`
        for it.
    sources : list of dict
        The Geonames ``_source`` dict behind each candidate, same order and
        length as ``choices``.  Only ``population``, ``name``, ``asciiname`` and
        ``alternativenames`` are read, and nothing from them is retained on the
        candidate dicts.

    Returns
    -------
    list of dict
        ``choices``, mutated in place.
    """
    if len(choices) != len(sources):
        raise ValueError(
            "add_entity_features got {} choices and {} sources".format(
                len(choices), len(sources)))

    cue = mention_admin_cue(search_name)
    search_norm = norm(search_name)
    search_low = str(search_name).lower()

    pops = []
    alt_matches = []
    names_lc_all = []
    ascii_lc_all = []
    for choice, source in zip(choices, sources):
        pop = _population(source)
        names_lc, ascii_lc, alt_or_ascii = _source_names(source)
        names_lc_all.append(names_lc)
        ascii_lc_all.append(ascii_lc)

        choice["mention_admin_cue"] = cue
        choice["log_population"] = math.log10(pop + 1)
        choice["exact_name_match"] = 1.0 if norm(choice["name"]) == search_norm else 0.0
        alt_match = 1.0 if search_norm in alt_or_ascii else 0.0
        choice["exact_altname_match"] = alt_match
        choice["is_admin_class"] = 1.0 if choice.get("feature_class") == "A" else 0.0
        pops.append(pop)
        alt_matches.append(alt_match)

    #
    #   Within-set prominence
    #
    max_pop = max(pops, default=0)
    max_log_pop = math.log10(max_pop + 1)
    max_alt_pop = max([p for p, a in zip(pops, alt_matches) if a], default=0)
    for choice, pop, alt_match in zip(choices, pops, alt_matches):
        choice["has_population"] = 1.0 if pop > 0 else 0.0
        choice["is_max_pop"] = 1.0 if (max_pop > 0 and pop == max_pop) else 0.0
        choice["log_pop_rel"] = choice["log_population"] - max_log_pop
        choice["is_max_pop_exact_match"] = (
            1.0 if (alt_match and max_alt_pop > 0 and pop == max_alt_pop) else 0.0
        )

    #
    #   A/P twins: same-name candidates that sit on top of each other but differ
    #   in feature class -- an administrative unit and the settlement inside it.
    #
    for choice in choices:
        choice["ap_twin"] = 0.0
    by_name = defaultdict(list)
    for choice in choices:
        by_name[norm(choice["name"])].append(choice)
    for group in by_name.values():
        if len(group) < 2:
            continue
        a_side = [c for c in group if c.get("feature_class") == "A"]
        p_side = [c for c in group if c.get("feature_class") == "P"]
        if not a_side or not p_side:
            continue
        for a in a_side:
            for b in p_side:
                if (abs(float(a["lat"]) - float(b["lat"])) < AP_TWIN_DEGREES
                        and abs(float(a["lon"]) - float(b["lon"])) < AP_TWIN_DEGREES):
                    a["ap_twin"] = 1.0
                    b["ap_twin"] = 1.0

    #
    #   Set shape: how ambiguous is this mention?
    #
    cand_names = [norm(c["name"]) for c in choices]
    name_ct = Counter(cand_names)
    n_exact = int(sum(c["exact_name_match"] for c in choices))
    log_n_exact = math.log10(n_exact + 1)

    #
    #   Case-folded edit distances: res_formatter's recipe on lowered text.
    #
    min_cf, max_cf, avg_cf, ascii_cf = [], [], [], []
    for names_lc, ascii_lc in zip(names_lc_all, ascii_lc_all):
        dists = [jellyfish.levenshtein_distance(search_low, x) for x in names_lc]
        min_cf.append(min(dists))
        max_cf.append(max(dists))
        avg_cf.append(sum(dists) / len(dists))
        ascii_cf.append(jellyfish.levenshtein_distance(search_low, ascii_lc))
    min_cf = normalize_distances(min_cf)
    max_cf = normalize_distances(max_cf)
    avg_cf = normalize_distances(avg_cf)
    ascii_cf = normalize_distances(ascii_cf)

    for k, choice in enumerate(choices):
        choice["log_n_same_name"] = math.log10(name_ct[cand_names[k]])
        choice["log_n_exact_matches"] = log_n_exact
        choice["is_unique_exact_match"] = (
            1.0 if (choice["exact_name_match"] > 0 and n_exact == 1) else 0.0
        )
        choice["is_seat_any"] = 1.0 if choice.get("feature_code") in SEAT_CODES else 0.0
        choice["min_dist_cf"] = min_cf[k]
        choice["max_dist_cf"] = max_cf[k]
        choice["avg_dist_cf"] = avg_cf[k]
        choice["ascii_dist_cf"] = ascii_cf[k]

    return choices


def _prep_entity(entity):
    """The per-candidate arrays the document geometry needs."""
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


def add_document_features(doc_es):
    """Attach the features that read the other mentions in the same document.

    Every entity in ``doc_es`` must already have been through
    :func:`add_entity_features` -- the anchors are picked with the population
    prior, which reads ``log_population`` and ``exact_name_match`` back off the
    candidate dicts.

    Parameters
    ----------
    doc_es : list of dict
        One document's worth of entity dicts, each with ``search_name`` and
        ``es_choices``.  Mutated in place.
    """
    if not doc_es:
        return doc_es

    country_names = load_country_names()
    mention_norms = [norm(e["search_name"]) for e in doc_es]
    doc_names = set(mention_norms)

    #
    #   Resolve every mention provisionally with the population prior: exact
    #   name match first, then largest population.  The resulting point is that
    #   mention's anchor.
    #
    prepped = [None] * len(doc_es)
    anchors = [None] * len(doc_es)
    tops = [None] * len(doc_es)
    for i, entity in enumerate(doc_es):
        d = _prep_entity(entity)
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

    for i, entity in enumerate(doc_es):
        d = prepped[i]
        n = d["n"]
        m_low = mention_norms[i]
        if n == 0:
            for choice in entity["es_choices"]:
                for key in DOCUMENT_KEYS:
                    choice[key] = null_value(key)
            continue

        # Other mention strings in this document.  The entity's own string is
        # removed, so a repeated mention is not evidence for itself.
        siblings = doc_names - {m_low}

        #
        #   Does another mention in the document name this candidate's parent?
        #
        for choice in d["cands"]:
            adm1 = choice.get("admin1_name")
            adm2 = choice.get("admin2_name")
            choice["sib_adm1"] = (
                1.0 if (norm(adm1) in siblings and adm1 not in ("", "NULL")) else 0.0
            )
            choice["sib_adm2"] = (
                1.0 if (norm(adm2) in siblings and adm2 not in ("", "NULL")) else 0.0
            )
            country = country_names.get(choice.get("country_code3"), "~~")
            choice["sib_country"] = 1.0 if country in siblings else 0.0

        #
        #   Sibling-anchor geometry
        #
        sib_idx = [
            j
            for j in range(len(doc_es))
            if j != i and anchors[j] is not None and mention_norms[j] != m_low
        ]
        na = len(sib_idx)
        if na:
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

        for choice in entity["es_choices"]:
            if is_null_choice(choice):
                for key in DOCUMENT_KEYS:
                    choice[key] = null_value(key)

    return doc_es
