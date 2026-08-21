"""e52 rule R1: abbreviation normalisation before the Elasticsearch query.

Report: experiments/campaign2/r1_retrieval_report.md. Ledger:
experiments/e57_r1_retrieval/NOTES.md.

The unit cases pin the two guards and the query-construction order; they need
no Elasticsearch. The integration cases at the bottom need the full geonames
index and are skipped without it.
"""
import pytest

from mordecai3.geonames import GeonamesService, _clean_search_name
from mordecai3.place_aliases import (AP, BARE_CODES, QUERY_OVERRIDE,
                                     alias_query, alias_targets)


# --------------------------------------------------------------- the table

def test_table_covers_states_and_provinces():
    assert len(BARE_CODES) == 65        # 50 states + DC + PR, 13 Canadian
    assert BARE_CODES["KY"] == "Kentucky"
    assert BARE_CODES["BC"] == "British Columbia"
    assert BARE_CODES["DC"] == "District of Columbia"


def test_every_ap_form_is_dotless_in_the_table():
    # The lookup key is the mention with its TRAILING periods stripped, so a
    # table key must never carry one -- "n.h" is right, "n.h." would be dead.
    assert not any(k.endswith(".") for k in AP)
    assert all(k == k.lower() for k in AP)


# ------------------------------------------------------------ period guard

@pytest.mark.parametrize("mention,expect", [
    ("Ind.", "Indiana"),
    ("Ky.", "Kentucky"),
    ("W.Va.", "West Virginia"),
    ("N.M.", "New Mexico"),
    ("Tenn.", "Tennessee"),
    ("B.C.", "British Columbia"),
    ("Sask.", "Saskatchewan"),
])
def test_ap_forms_with_a_period_expand(mention, expect):
    assert alias_targets(mention) == [expect]


@pytest.mark.parametrize("mention", [
    "La", "Miss", "Man", "Del", "Ore", "Ind", "Mo", "Va", "Wash", "Penn",
])
def test_ap_forms_without_a_period_never_expand(mention):
    """Without the trailing-period guard these are ordinary English words.

    `AP["la"]`, `AP["miss"]`, `AP["man"]`, `AP["del"]`, `AP["ore"]` and
    `AP["ind"]` would otherwise fire on the bare words. All 62 AP firings in
    the six held-out sources are written with the period, so the guard costs
    nothing measured and closes the whole false-positive class.
    """
    assert alias_targets(mention) == []
    assert alias_query(mention) is None


# -------------------------------------------------------------- case guard

@pytest.mark.parametrize("mention,expect", [
    ("WA", "Washington"),
    ("NC", "North Carolina"),
    ("SC", "South Carolina"),
])
def test_bare_codes_in_capitals_expand(mention, expect):
    assert alias_targets(mention) == [expect]


@pytest.mark.parametrize("mention", [
    "Wa", "wa", "Nc", "nc", "Ky", "ky",         # not capitals
    "in LA", "LA County", "W.A.", "L.A.",       # not the whole mention
    " WA x", "WAS", "W",                        # wrong shape
])
def test_bare_codes_expand_only_as_the_whole_capitalised_mention(mention):
    assert alias_targets(mention) == []
    assert alias_query(mention) is None


def test_la_in_capitals_still_expands_and_that_is_the_known_risk():
    """Documented residual risk, not an accident.

    In US newswire bare "LA" usually means Los Angeles. No held-out mention in
    any of the six sources is "LA", and the held-out scan in
    experiments/e57_r1_retrieval/la_risk.py finds zero standalone capitalised
    "LA" spans reaching the gazetteer, so the campaign has no evidence either
    way. If it bites in production, delete "LA" from `STATES`; the dotted form
    "La." keeps working through `AP`.
    """
    assert alias_targets("LA") == ["Louisiana"]
    assert alias_targets("La.") == ["Louisiana"]


# ------------------------------------------------------- ordering vs. clean

def test_dc_expansion_survives_clean_search_name():
    """`_clean_search_name` deletes the token "District".

    R1 has to run BEFORE it, so the expansion of "D.C." must be a string the
    cleaner leaves usable. "District of Columbia" would be sent as
    "of Columbia"; QUERY_OVERRIDE sends "Washington, D.C." instead.
    """
    assert alias_targets("D.C.") == ["District of Columbia"]
    assert alias_query("D.C.") == "Washington, D.C."
    assert QUERY_OVERRIDE["District of Columbia"] == "Washington, D.C."
    # The trap this override exists to avoid:
    assert "of Columbia" == _clean_search_name("District of Columbia")
    # ...and what actually gets sent:
    assert _clean_search_name(alias_query("D.C.")) == "Washington, D.C."


def test_no_expansion_is_destroyed_by_the_cleaner():
    """Every reachable expansion must survive `_clean_search_name` intact."""
    for mention in list(BARE_CODES) + [k + "." for k in AP]:
        q = alias_query(mention)
        if q is None:
            continue
        assert _clean_search_name(q) == q, (mention, q)


def test_non_places_and_non_abbreviations_are_left_alone():
    for mention in ["Springfield", "New York", "U.S.", "UK", "UAE", "EU",
                    "St. Paul", "Phila.", "", "   ", "Washington"]:
        assert alias_query(mention) is None


# ------------------------------------------- the query, replace not prepend

class _FakeConn:
    """Enough of an Elasticsearch client for Search() to be constructed."""


def _service(**kw):
    return GeonamesService(es_client=_FakeConn(), **kw)


def _query_string(search):
    return search.to_dict()["query"]["multi_match"]["query"]


def test_flag_on_replaces_the_query_string():
    """R1 REPLACES the query; it does not add a second one.

    e52 measured the alternative: prepending the expanded query's hits to the
    original list costs 9 entities, because at a serving window of 100 the
    100 new rows evict the tail -- "Ky." already had Kentucky at rank 6 and
    lost it. Replacement is also why the rule is free: one query in, one query
    out, no extra Elasticsearch round trip.
    """
    svc = _service(normalize_place_abbrevs=True)
    s = svc.build_name_search("Ind.", max_results=100)
    assert _query_string(s) == "Indiana"
    body = s.to_dict()
    # exactly one query, and the window is unchanged
    assert set(body["query"]) == {"multi_match"}
    assert body["from"] == 0 and body["size"] == 100


def test_flag_off_is_the_pre_e57_query():
    svc = _service(normalize_place_abbrevs=False)
    for mention in ["Ind.", "WA", "D.C.", "Ky."]:
        off = svc.build_name_search(mention).to_dict()
        assert _query_string(svc.build_name_search(mention)) == \
            _clean_search_name(mention)
        assert off["sort"] == [{"alt_name_length": {"order": "desc"}}]


def test_flag_defaults_to_on():
    assert _service().normalize_place_abbrevs is True
    assert _query_string(_service().build_name_search("Ky.")) == "Kentucky"


def test_flag_does_not_disturb_other_query_parameters():
    on = _service(normalize_place_abbrevs=True)
    off = _service(normalize_place_abbrevs=False)
    a = on.build_name_search("Ind.", max_results=37, fuzzy=1,
                             known_country="USA").to_dict()
    b = off.build_name_search("Indiana", max_results=37, fuzzy=1,
                              known_country="USA").to_dict()
    assert a == b        # the ONLY difference the flag makes is the string


def test_flag_is_inert_on_an_ordinary_mention():
    on = _service(normalize_place_abbrevs=True)
    off = _service(normalize_place_abbrevs=False)
    for mention in ["Springfield", "New York City", "the Hague", "US"]:
        assert (on.build_name_search(mention).to_dict()
                == off.build_name_search(mention).to_dict())


# --------------------------------------------------------------- live index

@pytest.mark.parametrize("mention,gold_id,name,baseline_rank", [
    ("Ky.", "6254925", "Kentucky", 6),      # present, but behind the UK and US
    ("Ind.", "4921868", "Indiana", None),   # absent from all 40 hits
    ("N.M.", "5481136", "New Mexico", None),
    ("Minn.", "5037779", "Minnesota", None),
    ("WA", "5815135", "Washington", 39),
])
def test_abbreviation_retrieves_its_state(geonames_service_all_data, es_client,
                                          mention, gold_id, name,
                                          baseline_rank):
    """The integration case: a "Ky." style mention now resolves to Kentucky.

    Under the `alt_name_length` sort the state is either absent ("Ind." brings
    back the Indus River, Indianapolis and Indore in 40 hits, no Indiana) or
    buried behind countries ("Ky." returns the United Kingdom, the United
    States and Turkey first). R1 puts it in the top two.
    """
    off = GeonamesService(es_client=es_client, normalize_place_abbrevs=False)
    on = GeonamesService(es_client=es_client, normalize_place_abbrevs=True)
    before = [h["geonameid"] for h in off.search_by_name(mention, 100)]
    after = [h["geonameid"] for h in on.search_by_name(mention, 100)]
    assert (before.index(gold_id) if gold_id in before else None) \
        == baseline_rank, f"{mention}: baseline retrieval of {name} moved"
    assert gold_id in after, f"{mention}: R1 did not retrieve {name}"
    assert after.index(gold_id) < 2


def test_dc_query_still_finds_washington(geonames_service_all_data, es_client):
    """The ordering trap, end to end.

    "D.C." is the one string where the expansion could make things worse: the
    baseline already returns 4140963 Washington at rank 0, and sending
    "of Columbia" (what `_clean_search_name` would do to "District of
    Columbia") would lose it.
    """
    on = GeonamesService(es_client=es_client, normalize_place_abbrevs=True)
    hits = [h["geonameid"] for h in on.search_by_name("D.C.", 20)]
    assert hits[0] == "4140963"


# ------------------------------------------------- through the whole parser

IND_TEXT = ("Officials in Evansville, Ind., said the flooding along the Ohio "
            "River had crested overnight.")


def _abbrev_pick(geo, mention, text):
    res = geo.geoparse_doc(text)
    return next((e for e in res["geolocated_ents"]
                 if e.get("search_name", "").rstrip(",") == mention), None)


def test_geoparser_resolves_an_ap_dateline_abbreviation(geoparser_all_data):
    """The integration case, through `Geoparser.geoparse_doc`.

    "Ind." is the right string to test with, not "Ky.": the baseline query for
    "Ky." does return 6254925 Kentucky, at rank 6, so the ranker can find it
    anyway and the test would pass for the wrong reason. "Ind." returns the
    Indus River, Indianapolis and Indore -- 40 hits with no Indiana in them --
    so the state is unreachable until R1 rewrites the query.
    """
    geo = geoparser_all_data
    assert geo.geonames.normalize_place_abbrevs is True
    ent = _abbrev_pick(geo, "Ind.", IND_TEXT)
    assert ent is not None, "the tagger did not emit an 'Ind.' span"
    assert ent.get("geonameid") == "4921868", ent.get("name")

    try:
        geo.geonames.normalize_place_abbrevs = False
        geo.geonames.clear_cache()
        before = _abbrev_pick(geo, "Ind.", IND_TEXT)
        assert before is None or before.get("geonameid") != "4921868", \
            "the baseline query already resolves Ind. -- the test is inert"
    finally:
        geo.geonames.normalize_place_abbrevs = True
        geo.geonames.clear_cache()


def test_geoparser_flag_passthrough(geonames_service_all_data):
    """`Geoparser(normalize_place_abbrevs=...)` reaches the service."""
    from mordecai3.geoparse import Geoparser
    svc = geonames_service_all_data
    original = svc.normalize_place_abbrevs
    try:
        Geoparser(geonames=svc, check_es=False, normalize_place_abbrevs=False)
        assert svc.normalize_place_abbrevs is False
        Geoparser(geonames=svc, check_es=False, normalize_place_abbrevs=True)
        assert svc.normalize_place_abbrevs is True
        # None means "leave the caller's service exactly as configured".
        svc.normalize_place_abbrevs = False
        Geoparser(geonames=svc, check_es=False)
        assert svc.normalize_place_abbrevs is False
    finally:
        svc.normalize_place_abbrevs = original


def test_flipping_the_flag_clears_the_candidate_cache():
    """`_es_cache` is keyed on the mention, not on the query it produces.

    Flipping the flag on a live service would otherwise serve candidate lists
    built under the other setting. Anything measuring both arms in one process
    depends on this.
    """
    svc = _service(normalize_place_abbrevs=True)
    svc._es_cache[("Ind.",)] = ["stale"]
    svc._parent_cache[("country_by_name", "x")] = None
    svc.normalize_place_abbrevs = True          # no change, no clear
    assert svc._es_cache
    svc.normalize_place_abbrevs = False         # changed -> cleared
    assert not svc._es_cache and not svc._parent_cache
