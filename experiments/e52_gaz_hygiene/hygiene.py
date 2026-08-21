"""e52: candidate-set hygiene as a post-retrieval / pre-query serving transform.

Nothing here touches the Elasticsearch index.  Every rule is a transform on the
candidate list a query already returned (or on the query string before it is
sent), so it can ship in `mordecai3/geoparse.py` without a reindex.  Each rule
is keyed to a specific finding in census.json -- see the report; there are no
speculative rules.

RULES
-----

R1  `abbrev`     Mention normalisation BEFORE the ES query.  A mention that is
                 a US-state / Canadian-province abbreviation ("Ind.", "W.Va.",
                 "WA") is phrase-matched against `alternativenames` and sorted
                 by `alt_name_length`, which returns countries: "WA" -> DR
                 Congo, "Ky." -> the United Kingdom, "N.M." -> a Santa Fe
                 hotel.  61% of ALL unretrievable golds in held-out TR/LGL/GWN
                 are this one defect.  The rule REPLACES the query: the
                 expanded name's hits take the head of the candidate list and
                 any duplicate is removed from the tail.  Prepending instead
                 is wrong -- at a serving window of 100 it evicts the tail, and
                 "Ky." (Kentucky already at rank 6) then loses its own gold.
                 Alias table lifted from experiments/e51_state_abbrev/sizing.py
                 (51 USPS + 13 Canadian + 62 AP dotted forms) with its case
                 guards.  The merge itself lives in eval_hygiene.apply_rules;
                 this module supplies the table and `alias_query`.

R2  `demote_h`   Drop a defunct GeoNames row (ADM*H / PPLH / PCLH / PPLQ ...)
                 when a LIVE row for the same place is also in the candidate
                 list -- same country, same stripped name key, within 25 km.
                 Census: `Kathmandu` -> `1283241 Kathmandu District ADM3H`
                 (population 1.26M, so it wins the population features) over
                 the live `1283240 Kathmandu PPLC`.  The "live twin exists"
                 guard is what stops the rule from deleting a gold whose only
                 gazetteer row is historical (`Varzaqan` PPLQ, TR #246).

R3  `dedupe`     Collapse candidates that are the same real place: same
                 country, same stripped name key, within 1 km.  Keeps ONE
                 representative and drops the rest.  `--dedupe-keep` chooses
                 the convention (p = populated place, a = admin unit, alt =
                 the row with the most alternate names).  Census: `Solferino`
                 ADM3 vs PPLA3 (0.4 km), `Paris` ADM2 vs PPLC (0.01 km),
                 `Bishkek` ADM1 vs PPLC (0.9 km).  NOTE: this rule cannot be
                 evaluated as "hygiene" alone -- picking the representative IS
                 picking an annotation convention, which e24 showed is worth
                 0.0000 under twin credit.  Measured anyway, both ways.

R4  `demote_junk` Drop a candidate that is an exact string match for the
                 mention but is a bare, unreferenced row (<= 2 alternate names,
                 feature class not A/P) when another candidate in the same
                 country has the same WHITESPACE-STRIPPED name and >= 10
                 alternate names.  Census: `Mauna Kea` -- gold `5850911
                 Maunakea MT` (43 alt names) loses 19 times to `6326699 Mauna
                 Kea MT` on Maui (1 alt name, pop 0) purely because the gold's
                 canonical spelling is one word and so `exact_name_match` = 0.
                 Also reaches `Miami Beach` BCH and `McKee School (historical)`
                 SCH.  Thresholds are in ALTERNATE-NAME COUNTS, recovered from
                 the pickled `alt_name_length` (which is log(n + 1)) by
                 `n_altnames`; a threshold written against the stored value
                 directly never fires.
"""
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/home/andy/projects/mordecai3"
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mordecai3.candidate_features import is_null_choice  # noqa: E402

# ------------------------------------------------------------------ R1 alias

STATES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut",
    "DE": "Delaware", "FL": "Florida", "GA": "Georgia", "HI": "Hawaii",
    "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
    "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine",
    "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan",
    "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri",
    "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico",
    "NY": "New York", "NC": "North Carolina", "ND": "North Dakota",
    "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania",
    "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota",
    "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont",
    "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming", "DC": "District of Columbia",
    "PR": "Puerto Rico",
}
CA_PROV = {
    "AB": "Alberta", "BC": "British Columbia", "MB": "Manitoba",
    "NB": "New Brunswick", "NL": "Newfoundland and Labrador",
    "NS": "Nova Scotia", "ON": "Ontario", "PE": "Prince Edward Island",
    "QC": "Quebec", "SK": "Saskatchewan", "NT": "Northwest Territories",
    "YT": "Yukon", "NU": "Nunavut",
}
AP = {
    "ala": "Alabama", "ariz": "Arizona", "ark": "Arkansas",
    "calif": "California", "colo": "Colorado", "conn": "Connecticut",
    "del": "Delaware", "fla": "Florida", "ga": "Georgia", "ill": "Illinois",
    "ind": "Indiana", "kan": "Kansas", "kans": "Kansas", "ky": "Kentucky",
    "la": "Louisiana", "md": "Maryland", "mass": "Massachusetts",
    "mich": "Michigan", "minn": "Minnesota", "miss": "Mississippi",
    "mo": "Missouri", "mont": "Montana", "neb": "Nebraska",
    "nebr": "Nebraska", "nev": "Nevada", "n.h": "New Hampshire",
    "n.j": "New Jersey", "n.m": "New Mexico", "n.mex": "New Mexico",
    "n.y": "New York", "n.c": "North Carolina", "n.d": "North Dakota",
    "n.dak": "North Dakota", "okla": "Oklahoma", "ore": "Oregon",
    "oreg": "Oregon", "pa": "Pennsylvania", "penn": "Pennsylvania",
    "penna": "Pennsylvania", "r.i": "Rhode Island", "s.c": "South Carolina",
    "s.d": "South Dakota", "s.dak": "South Dakota", "tenn": "Tennessee",
    "tex": "Texas", "vt": "Vermont", "va": "Virginia", "wash": "Washington",
    "w.va": "West Virginia", "wis": "Wisconsin", "wisc": "Wisconsin",
    "wyo": "Wyoming", "d.c": "District of Columbia", "p.r": "Puerto Rico",
    "alta": "Alberta", "b.c": "British Columbia", "man": "Manitoba",
    "n.b": "New Brunswick", "n.l": "Newfoundland and Labrador",
    "n.s": "Nova Scotia", "ont": "Ontario", "p.e.i": "Prince Edward Island",
    "que": "Quebec", "sask": "Saskatchewan",
}
UPPER = dict(STATES)
UPPER.update(CA_PROV)

# `_clean_search_name` in mordecai3/geonames.py strips the tokens "City",
# "District", "Region", "Province", "County", "Territory" out of any query.
# Three expansions are destroyed by that: "District of Columbia" -> "of
# Columbia", "Northwest Territories" -> "Northwest Territories" is safe but
# "District of Columbia" is not.  Send a query string the cleaner leaves alone.
QUERY_OVERRIDE = {
    "District of Columbia": "Washington, D.C.",
    "Northwest Territories": "Northwest Territories",
}


def alias_targets(raw):
    """Full ADM1 name(s) this raw mention string abbreviates, or empty.

    Two guards, both load-bearing:

    * a bare two-letter code expands only when the WHOLE mention is that code,
      in caps, with no dots -- "WA", "NC" yes, "in LA" or "Wa" no (e51's rule);
    * an AP form expands only when the mention actually ENDS IN A PERIOD.
      Without that, `AP["la"]` fires on the bare word "La" and `AP["miss"]`,
      `AP["man"]`, `AP["del"]`, `AP["ore"]`, `AP["ind"]` on ordinary words.
      Every one of the 62 AP firings in the six held-out sources is written
      with the period, so the guard costs nothing measured and closes the whole
      false-positive class.
    """
    s = str(raw).strip()
    out = []
    core = s.rstrip(".").replace(".", "").replace(" ", "")
    if len(core) == 2 and core.isalpha() and core.isupper() and s == core:
        if core in UPPER:
            out.append(UPPER[core])
    if s.endswith("."):
        key = s.rstrip(".").lower().replace(" ", "")
        if key in AP:
            v = AP[key]
            if v not in out:
                out.append(v)
    return out


def alias_query(raw):
    """The ES query string R1 sends instead of `raw`, or None."""
    t = alias_targets(raw)
    if not t:
        return None
    return QUERY_OVERRIDE.get(t[0], t[0])


# ------------------------------------------------------- shared name handling

_PUNCT = re.compile(r"[^a-z0-9 ]+")
_WS = re.compile(r"\s+")
# Same list rewrite_labels.strip_key uses, trimmed to the admin words that turn
# a settlement name into its unit name.
STRIP_TOKENS = {
    "county", "counties", "province", "provincia", "provincie", "district",
    "districts", "governorate", "muhafazat", "state", "states", "region",
    "regione", "prefecture", "department", "departement", "municipality",
    "municipio", "oblast", "krai", "canton", "parish", "division", "city",
    "territory", "metropolitan", "borough", "township", "regency", "shire",
    "voivodeship", "raion", "rayon", "commune", "arrondissement", "kreis",
    "landkreis", "stadtkreis", "kreisfreie", "gorod", "horad", "of", "the",
    "and", "shi", "ken", "si", "gun", "do", "urban", "rural", "greater",
}
HIST_CODES = re.compile(
    r"^(ADM[1-5]H|ADMDH|PPLH|PCLH|RGNH|PPLQ|ADMF|LCTY|PPLW)$")


def strip_key(name):
    s = _PUNCT.sub(" ", str(name).lower())
    toks = [t for t in _WS.split(s) if t and t not in STRIP_TOKENS]
    return " ".join(toks)


def tight_key(name):
    """Whitespace- and punctuation-free name key ("Maunakea" == "Mauna Kea")."""
    return _PUNCT.sub("", str(name).lower()).replace(" ", "")


def _km(a, b):
    import math
    lat1, lon1 = a
    lat2, lon2 = b
    p1, p2 = math.radians(lat1), math.radians(lat2)
    h = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0 * math.asin(min(1.0, math.sqrt(h)))


def _pt(c):
    return (float(c["lat"]), float(c["lon"]))


# ---------------------------------------------------------------- the rules

def drop_historical(choices, radius_km=25.0):
    """R2. Indices to drop: defunct rows superseded by a live row in the set."""
    live = {}
    for i, c in enumerate(choices):
        if not HIST_CODES.match(str(c.get("feature_code", ""))):
            live.setdefault((c.get("country_code3"), strip_key(c.get("name"))),
                            []).append(i)
    out = set()
    for i, c in enumerate(choices):
        if not HIST_CODES.match(str(c.get("feature_code", ""))):
            continue
        for j in live.get((c.get("country_code3"), strip_key(c.get("name"))), ()):
            if _km(_pt(c), _pt(choices[j])) <= radius_km:
                out.add(i)
                break
    return out


def dedupe_rows(choices, keep="p", radius_km=1.0):
    """R3. Indices to drop: co-located same-name rows beyond one representative."""
    n = len(choices)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    keys = [strip_key(c.get("name")) for c in choices]
    # Bucket on (country, name key) first: the pairwise scan is O(n^2) and n is
    # 500, which is 1.6 billion haversines over WikiDocs' held-out half.
    buckets = {}
    for i in range(n):
        if keys[i]:
            buckets.setdefault((choices[i].get("country_code3"), keys[i]),
                               []).append(i)
    for members in buckets.values():
        for x in range(len(members)):
            for y in range(x + 1, len(members)):
                i, j = members[x], members[y]
                if _km(_pt(choices[i]), _pt(choices[j])) <= radius_km:
                    parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    drop = set()
    for members in groups.values():
        if len(members) < 2:
            continue
        if keep == "p":
            rank = lambda i: (choices[i].get("feature_class") != "P",
                              -float(choices[i].get("alt_name_length", 0)), i)
        elif keep == "a":
            rank = lambda i: (choices[i].get("feature_class") != "A",
                              -float(choices[i].get("alt_name_length", 0)), i)
        else:
            rank = lambda i: (-float(choices[i].get("alt_name_length", 0)), i)
        winner = sorted(members, key=rank)[0]
        drop.update(m for m in members if m != winner)
    return drop


def n_altnames(choice):
    """Alternate-name count behind a candidate's `alt_name_length`.

    res_formatter stores log(len(alternativenames) + 1), not the raw count, so
    a threshold written against raw counts silently never fires.
    """
    return math.exp(float(choice.get("alt_name_length", 0) or 0)) - 1.0


def drop_junk_exact(choices, search_name, max_junk_alts=2.0, min_twin_alts=10.0):
    """R4. Indices to drop: bare unreferenced rows shadowing a notable twin."""
    tight_mention = tight_key(search_name)
    if not tight_mention:
        return set()
    out = set()
    for i, c in enumerate(choices):
        if n_altnames(c) > max_junk_alts:
            continue
        if str(c.get("feature_class", "")) in ("A", "P"):
            continue
        if tight_key(c.get("name")) != tight_mention:
            continue
        for j, d in enumerate(choices):
            if j == i:
                continue
            if (d.get("country_code3") == c.get("country_code3")
                    and tight_key(d.get("name")) == tight_mention
                    and n_altnames(d) >= min_twin_alts):
                out.add(i)
                break
    return out


def hygiene_drops(entity, rules, dedupe_keep="p"):
    """Union of the indices every enabled dropping rule removes.

    Operates on the REAL candidate rows only; the NULL sentinel (last) is never
    touched, because dropping it would change what "no answer" means.
    """
    choices = [c for c in entity["es_choices"] if not is_null_choice(c)]
    drop = set()
    if "demote_h" in rules:
        drop |= drop_historical(choices)
    if "dedupe" in rules:
        drop |= dedupe_rows(choices, keep=dedupe_keep)
    if "demote_junk" in rules:
        drop |= drop_junk_exact(choices, entity.get("search_name", ""))
    return drop
