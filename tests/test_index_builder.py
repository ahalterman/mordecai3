"""The GeoNames row -> ES document conversion behind `mordecai3 index build`.

No Elasticsearch needed. The document shape must stay field-for-field what
es-geonames produced, or a rebuilt index stops matching the prebuilt one.
"""

from mordecai3.index_builder import ISO3, documents, remove_accents

# allCountries.txt columns: id, name, asciiname, altnames, lat, lon, fclass,
# fcode, cc, cc2, admin1, admin2, admin3, admin4, population, ...
ROW = ["6252001", "United States", "United States", "USA,Estados Unidos",
       "39.76", "-98.5", "A", "PCLI", "US", "", "00", "", "", "", "327167434",
       "", "", "America/Chicago", "2024-01-01"]
SYRIA_ROW = ["170063", "Aleppo", "Aleppo", "Ḩalab,حلب", "36.2", "37.16", "P",
             "PPLA", "SY", "", "09", "", "", "", "2098210", "", "", "", ""]


def test_remove_accents():
    assert remove_accents("Ḩadīqat ash Shahbā") == "Hadiqat ash Shahba"
    assert remove_accents("北京") == "北京"


def test_document_fields():
    (action,) = documents([ROW], {"US.00": "Nowhere"}, {})
    doc = action["_source"]
    assert action["_id"] == "6252001" and action["_index"] == "geonames"
    assert doc["country_code3"] == "USA"
    assert doc["coordinates"] == "39.76,-98.5"
    # the hand-added aliases for the US
    assert {"US", "U.S.", "USA", "Estados Unidos"} <= set(doc["alternativenames"])
    # counted after the US aliases, before the accent-stripped copies (as in
    # es-geonames; the ranker's features were trained on this definition)
    assert doc["alt_name_length"] == 4


def test_accent_stripped_alternatives_and_admin1():
    (action,) = documents([SYRIA_ROW], {"SY.09": "Aleppo"}, {})
    doc = action["_source"]
    assert {"Ḩalab", "Halab", "حلب"} <= set(doc["alternativenames"])
    assert doc["admin1_name"] == "Aleppo"


def test_unknown_country_code():
    bad = set()
    row = list(SYRIA_ROW)
    row[8] = "ZZ"
    (action,) = documents([row], {}, {}, bad_codes=bad)
    assert action["_source"]["country_code3"] == "NA" and bad == {"ZZ"}
    assert ISO3["SY"] == "SYR"
