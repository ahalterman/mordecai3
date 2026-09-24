"""Abbreviation aliases for the Elasticsearch name query (e52 rule R1).

`GeonamesService.build_name_search` phrase-matches the mention against
`name`/`asciiname`/`alternativenames` and sorts the hits by `alt_name_length`
descending. That sort is a pure fame prior with no relevance term, so a mention
that is a two-letter or AP-dateline abbreviation retrieves the most-aliased
document that happens to contain those letters -- which is almost always a
country. Verified live against the shipped index:

    "Ky."  -> United Kingdom, United States, Turkey    (no Kentucky at all)
    "WA"   -> DR Congo, UAE, Central African Republic  (no Washington)
    "Ind." -> Indus River, Indianapolis, Indore        (no Indiana)
    "N.M." -> a railway station in Mexico, a Santa Fe hotel

The census in `experiments/campaign2/gazetteer_hygiene_report.md` found that
**61% of every unretrievable gold in held-out TR/LGL/GWN is this one defect**:
the gold is not mis-ranked, it is never returned. Expanding the mention to the
full ADM1 name *before* the query fixes it for free -- one query is replaced by
another, none is added.

Guards
------
Both guards are load-bearing and both are measured, not assumed.

* **bare two-letter codes** expand only when the whole mention is that code, in
  capitals, with no dots ("WA", "NC"). "Wa", "wa" and the word "in" inside a
  longer mention never expand. (From `experiments/e51_state_abbrev/sizing.py`.)
* **AP dotted forms** expand only when the mention actually **ends in a
  period**. Without that, `AP["la"]` fires on the bare word "La" and
  `AP["miss"]`, `AP["man"]`, `AP["del"]`, `AP["ore"]`, `AP["ind"]` on ordinary
  English words. All 62 AP firings across the six held-out sources are written
  with the period, so the guard costs exactly nothing measured and closes the
  whole false-positive class.

These tables are for **place spans only**. Do not lift them into a general text
normaliser: "Miss.", "Del.", "Man." and "Ore." are ordinary English outside a
span the tagger has already called a place.

Residual risk, stated because held-out cannot price it: "LA" in capitals still
expands to Louisiana, and in US newswire "LA" usually means Los Angeles. See
`experiments/campaign2/r1_retrieval_report.md` for the held-out scan; if it
bites in production, drop "LA" (and arguably "IN", "OR", "ME", "OK") from
`BARE_CODES` and keep only their dotted AP forms.
"""

__all__ = ["STATES", "CA_PROV", "AP", "BARE_CODES", "QUERY_OVERRIDE",
           "alias_targets", "alias_query"]

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

#: The bare (dotless, capitalised) codes. USPS plus the Canada Post codes.
BARE_CODES = dict(STATES)
BARE_CODES.update(CA_PROV)

# `_clean_search_name` deletes the tokens "City", "District", "Region",
# "Province", "County", "Territory" from any query, and R1 runs BEFORE it (it
# has to: the cleaner would otherwise be handed the abbreviation). So an
# expansion containing one of those words has to be written as a string the
# cleaner leaves usable -- "District of Columbia" would be sent as
# "of Columbia".  "Washington, D.C." retrieves 4140963 at rank 0.
QUERY_OVERRIDE = {
    "District of Columbia": "Washington, D.C.",
}


def alias_targets(raw):
    """Full ADM1 name(s) the mention string `raw` abbreviates, or ``[]``.

    Examples
    --------
    >>> alias_targets("Ind.")
    ['Indiana']
    >>> alias_targets("WA")
    ['Washington']
    >>> alias_targets("Wa")          # not capitals
    []
    >>> alias_targets("Ind")         # no trailing period
    []
    >>> alias_targets("La")          # the word "La", not Louisiana
    []
    """
    s = str(raw).strip()
    out = []
    core = s.rstrip(".").replace(".", "").replace(" ", "")
    if len(core) == 2 and core.isalpha() and core.isupper() and s == core:
        if core in BARE_CODES:
            out.append(BARE_CODES[core])
    if s.endswith("."):
        key = s.rstrip(".").lower().replace(" ", "")
        if key in AP:
            v = AP[key]
            if v not in out:
                out.append(v)
    return out


def alias_query(raw):
    """The Elasticsearch query string R1 sends instead of `raw`, or ``None``.

    ``None`` means "this mention is not an abbreviation" and the caller must
    leave the query untouched -- it is not the same as an empty string.
    """
    t = alias_targets(raw)
    if not t:
        return None
    return QUERY_OVERRIDE.get(t[0], t[0])
