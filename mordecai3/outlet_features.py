"""The ``outlet`` candidate-feature block: where the newspaper is.

A local paper writes about its own town.  Across every linked toponym in LGL,
80.8% sit in their outlet's home admin1, and on the *ambiguous* phrases -- the
ones the ranker actually gets wrong -- the concentration is higher still.  That
is a prior no other feature in the model can see: the sibling and anchor
geometry in ``candidate_features`` needs at least one other place name in the
document to triangulate from, and the errors this is aimed at are precisely the
ones where the document offers nothing (``parispi.net`` writing "Paris" with no
other toponym in the article, resolved to France three times over).

So the block gives the ranker one external anchor per document: the newsroom's
own location, looked up from the article's outlet domain.  Five columns:

``has_outlet_home``
    1.0 if this document's outlet has a known home *point*.  A mask, not
    evidence: it tells the model whether the next two columns mean anything.
``log_km_to_outlet_home``
    ``log10(km + 1)`` from the candidate to that point.  When there is no home
    point this takes the same ``NO_ANCHOR_KM`` sentinel the sibling geometry
    uses when a document offers no anchor, so "no outlet" and "outlet very far
    away" are not confusable with "outlet next door".
``outlet_same_adm1``
    1.0 if the candidate is in the home's ``(country, admin1)``.
``outlet_same_country``
    1.0 if the candidate is in the home's country.  Set for country-level homes
    too, which is the only signal a national paper carries.
``has_outlet_country``
    1.0 if the outlet resolves to a country at all.  The second mask; it is 1.0
    whenever ``has_outlet_home`` is, and also for the national outlets.

Both masks are properties of the *document*, not of the candidate, so they are
constant down an entity's candidate list and the placeholder "no correct answer"
row carries the real value -- exactly how ``mention_admin_cue`` is handled in
``candidate_features``.  The placeholder's *evidence* columns get the neutral /
worst values, so it never looks like the best-supported option.

Documents from a source with no outlet metadata at all -- GWN, Prodigy, Synth,
WikiDocs, which is 81% of held-out entities -- get the full null: both masks
0.0, the distance at its sentinel, both indicators 0.0.  That is a well-defined
"no evidence", and it is also, unavoidably, a corpus indicator.  The guardrail
for that is in the report: if a source that structurally cannot have an outlet
moves, the model is reading the mask rather than the feature.

The home table and its leak protocol live in ``tools/outlet_home_table.py``.
Nothing here ever sees a gold label.
"""

import math

import numpy as np

from .candidate_features import NO_ANCHOR_KM, haversine, is_null_choice

OUTLET_KEYS = [
    "has_outlet_home",
    "log_km_to_outlet_home",
    "outlet_same_adm1",
    "outlet_same_country",
    "has_outlet_country",
]

# Same sentinel the sibling geometry uses for "this document offers no anchor".
NO_OUTLET_KM = NO_ANCHOR_KM

OUTLET_NULL_SENTINELS = {
    "log_km_to_outlet_home": math.log10(NO_OUTLET_KM + 1),
}


def outlet_null_value(key):
    """What a candidate with no outlet information carries for one key."""
    return OUTLET_NULL_SENTINELS.get(key, 0.0)


def clear_outlet_features(choices):
    """Give every candidate the no-outlet values."""
    for choice in choices:
        for key in OUTLET_KEYS:
            choice[key] = outlet_null_value(key)
    return choices


def add_outlet_features(choices, home):
    """Attach the outlet block to one entity's candidate list.

    Parameters
    ----------
    choices : list of dict
        Formatted candidates, including the NULL placeholder row if there is
        one.  Mutated in place.
    home : dict or None
        The resolved home for this document's outlet, as
        ``tools.outlet_home_table.geocode_homes`` returns it: ``lat``/``lon``
        (``None`` for a country-level home), ``country_code3``, ``admin1`` as a
        ``(country_code3, admin1_code)`` pair or ``None``, and ``level``.
        ``None`` means this document has no outlet, or an outlet with no home.
    """
    if not home:
        return clear_outlet_features(choices)

    has_point = 1.0 if (home.get("lat") is not None
                        and home.get("level") == "point") else 0.0
    has_country = 1.0 if home.get("country_code3") else 0.0
    home_cc = str(home.get("country_code3") or "")
    home_a1 = home.get("admin1")
    if isinstance(home_a1, list):        # survives a JSON round-trip as a list
        home_a1 = tuple(home_a1)

    real = [c for c in choices if not is_null_choice(c)]

    if has_point and real:
        lat = np.array([float(c["lat"]) for c in real], dtype=np.float64)
        lon = np.array([float(c["lon"]) for c in real], dtype=np.float64)
        km = haversine(lat, lon,
                       np.array([float(home["lat"])]),
                       np.array([float(home["lon"])]))[:, 0]
    else:
        km = np.full(len(real), NO_OUTLET_KM)

    for k, choice in enumerate(real):
        cc = str(choice.get("country_code3") or "")
        a1 = str(choice.get("admin1_code") or "").strip()
        choice["has_outlet_home"] = has_point
        choice["has_outlet_country"] = has_country
        choice["log_km_to_outlet_home"] = math.log10(km[k] + 1)
        choice["outlet_same_adm1"] = (
            1.0 if (home_a1 and a1 and (cc, a1) == tuple(home_a1)) else 0.0
        )
        choice["outlet_same_country"] = 1.0 if (home_cc and cc == home_cc) else 0.0

    # The placeholder row: the two masks are document properties and carry the
    # real value; the evidence columns take the neutral / worst end.
    for choice in choices:
        if is_null_choice(choice):
            choice["has_outlet_home"] = has_point
            choice["has_outlet_country"] = has_country
            choice["log_km_to_outlet_home"] = outlet_null_value(
                "log_km_to_outlet_home")
            choice["outlet_same_adm1"] = 0.0
            choice["outlet_same_country"] = 0.0

    return choices
