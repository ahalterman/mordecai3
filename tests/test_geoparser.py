
import pytest

from mordecai3 import Geoparser
from mordecai3.elasticsearch import setup_es_client


def test_geoparser_can_be_initialized():
    geo = Geoparser(hosts=["localhost"])
    assert isinstance(geo, Geoparser)


# issue #17
def test_geoparser_arbitrary_es_connection():
    client = setup_es_client(hosts=["localhost"], port=9200, request_timeout=1)
    geo = Geoparser(es_client=client)
    assert isinstance(geo, Geoparser)


def test_geoparse_doc(geo):
    res = geo.geoparse_doc("I visited The Hague in the Netherlands.")
    names = [e.get("name", "") for e in res["geolocated_ents"]]
    assert any("Hague" in n for n in names), f"Expected 'Hague' in results, got {names}"


def test_geoparse_doc_output_structure(geo):
    res = geo.geoparse_doc("Fighting continued in Aleppo, Syria.")
    assert "doc_text" in res
    assert "event_location_raw" in res
    assert "geolocated_ents" in res
    assert isinstance(res["geolocated_ents"], list)
    for ent in res["geolocated_ents"]:
        assert "search_name" in ent
        assert "start_char" in ent
        assert "end_char" in ent


def test_geoparse_doc_no_entities(geo):
    res = geo.geoparse_doc("This sentence has no place names at all.")
    assert res["geolocated_ents"] == []


def test_geoparse_doc_accepts_spacy_doc(geo):
    doc = geo.nlp("I visited Berlin, Germany.")
    res = geo.geoparse_doc(doc)
    assert len(res["geolocated_ents"]) > 0


# ---------- Batch tests ----------


def test_geoparse_batch_basic(geo):
    texts = [
        "I visited Berlin, Germany.",
        "Fighting broke out in Aleppo, Syria.",
        "The president spoke from Washington, D.C.",
    ]
    results = geo.geoparse_batch(texts)
    assert len(results) == len(texts)
    for r in results:
        assert "doc_text" in r
        assert "geolocated_ents" in r


def test_geoparse_batch_empty_input(geo):
    results = geo.geoparse_batch([])
    assert results == []


def test_geoparse_batch_no_entities(geo):
    texts = [
        "There are no place names here.",
        "Just a regular sentence about nothing geographic.",
    ]
    results = geo.geoparse_batch(texts)
    assert len(results) == 2
    for r in results:
        assert r["geolocated_ents"] == []


def test_geoparse_batch_mixed(geo):
    """Mix of documents with and without location entities."""
    texts = [
        "No locations here.",
        "I visited The Hague in the Netherlands.",
        "Another sentence with no places.",
    ]
    results = geo.geoparse_batch(texts)
    assert len(results) == 3
    assert results[0]["geolocated_ents"] == []
    assert len(results[1]["geolocated_ents"]) > 0
    assert results[2]["geolocated_ents"] == []


def test_geoparse_batch_parity(geo):
    """Batch results should match individual geoparse_doc calls."""
    texts = [
        "I visited Berlin, Germany.",
        "Fighting broke out in Aleppo, Syria.",
        "The earthquake struck Christchurch, New Zealand.",
    ]
    batch_results = geo.geoparse_batch(texts)
    for text, batch_result in zip(texts, batch_results):
        single_result = geo.geoparse_doc(text)
        batch_ids = {e.get("geonameid") for e in batch_result["geolocated_ents"]}
        single_ids = {e.get("geonameid") for e in single_result["geolocated_ents"]}
        assert batch_ids == single_ids, (
            f"Parity mismatch for '{text[:40]}...': "
            f"batch={batch_ids}, single={single_ids}"
        )


def test_geoparse_batch_preserves_order(geo):
    """Output order matches input order."""
    texts = [
        "Events in Tokyo, Japan.",
        "The meeting was held in Paris, France.",
    ]
    results = geo.geoparse_batch(texts)
    assert "Tokyo" in results[0]["doc_text"]
    assert "Paris" in results[1]["doc_text"]


def test_geoparse_batch_with_progress(geo):
    """show_progress=True should not change results."""
    texts = ["I visited Berlin, Germany."]
    results = geo.geoparse_batch(texts, show_progress=True)
    assert len(results) == 1
    assert len(results[0]["geolocated_ents"]) > 0


def test_geoparse_batch_chunk_boundary(geo):
    """Processing works correctly across chunk boundaries."""
    texts = ["I visited Berlin, Germany."] * 5
    results = geo.geoparse_batch(texts, chunk_size=2)
    assert len(results) == 5
    for r in results:
        assert len(r["geolocated_ents"]) > 0
