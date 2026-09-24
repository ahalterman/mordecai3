"""The `outlet` feature block, from its null encoding up to the serving path.

The block (mordecai3/outlet_features.py) gives the ranker one external anchor
per document: where the newspaper that published it is. Two properties have to
hold for it to be safe to ship, and both are tested here.

1. **The null encoding is exact.** A document with no outlet -- no argument
   passed, an unknown domain, a checkpoint without the block -- has to produce
   the numbers the four no-outlet training sources carried, not zeros. In
   particular the distance takes the sibling geometry's `NO_ANCHOR_KM`
   sentinel, because 0 km is the *best* value a distance can have and the
   reserved row competes in the softmax.
2. **Nothing changes for a caller who does not use it.** The argument is
   optional and inert on a checkpoint trained without the block, so the merge
   cannot move any pre-existing behaviour.

The serving tests need the e54 ship-candidate checkpoint
(`assets/mordecai_2026-08-20_e54_seed42.pt`) and the full GeoNames index; they
skip when either is missing. The rest are pure unit tests.
"""

import math
import os

import pytest

from mordecai3 import Geoparser
from mordecai3.candidate_features import NO_ANCHOR_KM
from mordecai3.geoparse import (lookup_outlet_home, normalize_outlet,
                                read_outlet_homes)
from mordecai3.outlet_features import (OUTLET_KEYS, add_outlet_features,
                                       clear_outlet_features,
                                       outlet_null_value)
from mordecai3.torch_model import FEATURE_BLOCKS, expand_feature_blocks

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHIP_CANDIDATE = os.path.join(REPO_ROOT, "mordecai3", "assets",
                              "mordecai_2026-08-20_e54_seed42.pt")

# The marquee case of experiments/campaign2/outlet_feature_report.md: the Paris
# Post-Intelligencer is published in Paris, *Tennessee*, and its articles say
# "Paris" with no other toponym to triangulate from. Without the outlet the
# population prior wins and the answer is France.
PARIS_TN = "4647963"
PARIS_FR = "2988507"
PARIS_TEXT = "The city council will meet Tuesday in Paris to discuss the new budget."


#
#   The block itself
#


def test_outlet_block_is_registered_last():
    """Every pre-existing column keeps its index, which is what makes a
    baseline recipe read byte-identical features out of an outlet pickle."""
    assert list(FEATURE_BLOCKS)[-1] == "outlet"
    assert FEATURE_BLOCKS["outlet"] == OUTLET_KEYS
    assert len(OUTLET_KEYS) == 5
    base = expand_feature_blocks("prom,name,cue,sib,geo,shape")
    with_outlet = expand_feature_blocks("prom,name,cue,sib,geo,shape,outlet")
    assert with_outlet[:len(base)] == base


def test_the_no_outlet_encoding_is_the_sibling_sentinel_not_zero():
    assert outlet_null_value("log_km_to_outlet_home") == math.log10(NO_ANCHOR_KM + 1)
    assert outlet_null_value("log_km_to_outlet_home") == pytest.approx(4.301051709845226)
    for key in OUTLET_KEYS:
        if key != "log_km_to_outlet_home":
            assert outlet_null_value(key) == 0.0


def _choices():
    return [
        {"geonameid": "4647963", "lat": 36.302, "lon": -88.326,
         "country_code3": "USA", "admin1_code": "TN"},
        {"geonameid": "2988507", "lat": 48.853, "lon": 2.349,
         "country_code3": "FRA", "admin1_code": "11"},
    ]


PARIS_TN_HOME = {"lat": 36.302, "lon": -88.326, "country_code3": "USA",
                 "admin1": ("USA", "TN"), "level": "point"}


def test_a_missing_home_is_exactly_the_cleared_block():
    """`add_outlet_features(..., None)` and `clear_outlet_features` agree.

    This is the fallback every unknown domain takes, so the two paths must not
    be allowed to drift apart.
    """
    a, b = _choices(), _choices()
    add_outlet_features(a, None)
    clear_outlet_features(b)
    assert a == b
    for choice in a:
        for key in OUTLET_KEYS:
            assert choice[key] == outlet_null_value(key)


def test_a_point_home_marks_its_own_admin1_and_country():
    choices = _choices()
    add_outlet_features(choices, PARIS_TN_HOME)
    tn, fr = choices
    assert tn["has_outlet_home"] == 1.0
    assert tn["has_outlet_country"] == 1.0
    assert tn["outlet_same_adm1"] == 1.0
    assert tn["outlet_same_country"] == 1.0
    assert tn["log_km_to_outlet_home"] < 1.0          # ~0 km from its own home
    assert fr["outlet_same_adm1"] == 0.0
    assert fr["outlet_same_country"] == 0.0
    assert fr["log_km_to_outlet_home"] > 3.0          # ~7,000 km away


def test_a_country_home_fires_only_the_country_channel():
    """A national paper has no newsroom point, so the geometry stays null."""
    choices = _choices()
    add_outlet_features(choices, {"lat": None, "lon": None,
                                  "country_code3": "FRA", "admin1": None,
                                  "level": "country"})
    for choice in choices:
        assert choice["has_outlet_home"] == 0.0
        assert choice["has_outlet_country"] == 1.0
        assert choice["outlet_same_adm1"] == 0.0
        assert choice["log_km_to_outlet_home"] == outlet_null_value(
            "log_km_to_outlet_home")
    assert choices[1]["outlet_same_country"] == 1.0
    assert choices[0]["outlet_same_country"] == 0.0


#
#   Looking an outlet up
#


@pytest.mark.parametrize("given,want", [
    ("parispi.net", "parispi.net"),
    ("  ParisPI.net ", "parispi.net"),
    ("https://www.parispi.net/news/local/story.html", "www.parispi.net"),
    ("http://parispi.net:8080/", "parispi.net"),
    ("", ""),
    (None, ""),
])
def test_normalize_outlet(given, want):
    assert normalize_outlet(given) == want


def test_lookup_tolerates_the_www_prefix_either_way():
    """LGL writes `ajc.com`, TR-News writes `www.cbc.ca`; callers write both."""
    homes = {"ajc.com": {"level": "point"}, "www.cbc.ca": {"level": "country"}}
    assert lookup_outlet_home(homes, "www.ajc.com")["level"] == "point"
    assert lookup_outlet_home(homes, "cbc.ca")["level"] == "country"
    assert lookup_outlet_home(homes, "https://www.ajc.com/news")["level"] == "point"
    assert lookup_outlet_home(homes, "example.invalid") is None
    assert lookup_outlet_home(homes, None) is None
    assert lookup_outlet_home({}, "ajc.com") is None


def test_the_packaged_home_table_loads():
    homes = read_outlet_homes(None)
    assert len(homes) >= 100
    assert lookup_outlet_home(homes, "parispi.net")["country_code3"] == "USA"


#
#   Serving
#


@pytest.fixture(scope="module")
def outlet_geoparser(geonames_service_all_data):
    if not os.path.exists(SHIP_CANDIDATE):
        pytest.skip(f"no outlet checkpoint at {SHIP_CANDIDATE}")
    return Geoparser(model_path=SHIP_CANDIDATE,
                     geonames=geonames_service_all_data)


def _ids(result):
    return [e.get("geonameid") for e in result["geolocated_ents"]]


def test_the_outlet_checkpoint_loads_its_block_and_its_table(outlet_geoparser):
    geo = outlet_geoparser
    assert "outlet" in geo.feature_blocks
    assert geo.uses_outlet
    assert len(geo.outlet_homes) >= 100


def test_a_local_papers_own_town_beats_the_population_prior(outlet_geoparser):
    """The case the whole arm was aimed at, end to end through geoparse_doc."""
    geo = outlet_geoparser
    assert _ids(geo.geoparse_doc(PARIS_TEXT)) == [PARIS_FR]
    assert _ids(geo.geoparse_doc(PARIS_TEXT, outlet="parispi.net")) == [PARIS_TN]
    # ...and a URL is accepted wherever a bare domain is.
    assert _ids(geo.geoparse_doc(
        PARIS_TEXT, outlet="https://www.parispi.net/news/x")) == [PARIS_TN]


def test_a_different_papers_town_does_not_move_it(outlet_geoparser):
    """The feature is the article-to-newsroom correspondence, not a US prior.

    The Pittsburgh Post-Gazette is as American as the Paris Post-Intelligencer,
    and it leaves Paris in France -- which is the serving-side version of the
    home-permutation control in outlet_feature_report.md §6.4.
    """
    geo = outlet_geoparser
    assert _ids(geo.geoparse_doc(PARIS_TEXT, outlet="post-gazette.com")) == [PARIS_FR]


def test_an_unknown_outlet_is_the_no_outlet_path(outlet_geoparser):
    """A domain that is not in the table has to degrade, not guess."""
    geo = outlet_geoparser
    plain = geo.geoparse_doc(PARIS_TEXT)
    unknown = geo.geoparse_doc(PARIS_TEXT, outlet="not-a-real-paper.invalid")
    assert _ids(unknown) == _ids(plain)
    assert [e["score"] for e in unknown["geolocated_ents"]] == pytest.approx(
        [e["score"] for e in plain["geolocated_ents"]])


def test_outlet_features_actually_reach_the_candidates(outlet_geoparser):
    """Not just the answer: the five columns themselves change, and only them.

    `debug=True` keeps the internal keys, so this reads the numbers the model
    was handed rather than inferring them from the prediction.
    """
    geo = outlet_geoparser
    plain = geo.geoparse_doc(PARIS_TEXT, trim=False, debug=True)
    with_outlet = geo.geoparse_doc(PARIS_TEXT, trim=False, debug=True,
                                   outlet="parispi.net")
    by_id = {e["geonameid"]: e for e in plain["geolocated_ents"]}
    for ent in with_outlet["geolocated_ents"]:
        before = by_id.get(ent["geonameid"])
        if before is None:
            continue
        assert before["has_outlet_home"] == 0.0
        assert before["log_km_to_outlet_home"] == pytest.approx(
            outlet_null_value("log_km_to_outlet_home"))
        assert ent["has_outlet_home"] == 1.0
        assert ent["has_outlet_country"] == 1.0
        # everything outside the block is untouched by the outlet argument
        for key in ("alt_name_length", "min_dist", "adm1_count",
                    "log_min_km_anchor", "is_country"):
            if key in before and key in ent:
                assert before[key] == pytest.approx(ent[key]), key
    tn = next(e for e in with_outlet["geolocated_ents"]
              if e["geonameid"] == PARIS_TN)
    assert tn["outlet_same_adm1"] == 1.0


def test_batch_takes_one_outlet_per_document(outlet_geoparser):
    geo = outlet_geoparser
    texts = [PARIS_TEXT, PARIS_TEXT, PARIS_TEXT]
    res = geo.geoparse_batch(texts, outlets=["parispi.net", None,
                                             "theparisnews.com"])
    assert _ids(res[0]) == [PARIS_TN]
    assert _ids(res[1]) == [PARIS_FR]
    assert _ids(res[2]) == ["4717560"]        # Paris, Texas
    with pytest.raises(ValueError):
        geo.geoparse_batch(texts, outlets=["parispi.net"])


#
#   ...and none of the above may touch a caller who is not using it
#


def test_the_default_checkpoint_ignores_the_outlet_argument(geoparser_all_data):
    """The packaged model has no outlet block, so `outlet=` is a no-op.

    This is the pre-merge-behaviour guarantee: same answer, same score, same
    keys, whatever the caller passes.
    """
    geo = geoparser_all_data
    assert not geo.uses_outlet
    assert geo.outlet_homes == {}
    plain = geo.geoparse_doc(PARIS_TEXT, trim=False, debug=True)
    passed = geo.geoparse_doc(PARIS_TEXT, trim=False, debug=True,
                              outlet="parispi.net")
    assert plain == passed
    for ent in plain["geolocated_ents"]:
        assert not set(OUTLET_KEYS) & set(ent)


def test_batch_without_outlets_is_unchanged(geoparser_all_data):
    geo = geoparser_all_data
    texts = ["I visited Aleppo in Syria.", PARIS_TEXT]
    assert (geo.geoparse_batch(texts)
            == geo.geoparse_batch(texts, outlets=[None, "parispi.net"]))
