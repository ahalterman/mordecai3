"""Tests for the batched (_msearch) Elasticsearch lookup path."""

from unittest.mock import patch

import pytest

import mordecai3.geoparse as gp
from mordecai3.exceptions import GeonamesQueryError


TEXTS = [
    "Fighting continued in Aleppo, Syria for a third day.",
    "I visited The Hague in the Netherlands.",
    "The summit was held in Geneva, Switzerland, with delegates from Berlin.",
    "This sentence has no place names at all.",
    "Talks resumed in Aleppo, Syria.",  # repeats names from doc 0
]


def _entity_data(geo, texts):
    docs = list(geo.nlp.pipe(texts, batch_size=8))
    return [gp.doc_to_ex_expanded(d) for d in docs]


def _signature(doc_es):
    """Everything the model actually consumes, so parity is checked on features
    and ordering, not just on which geonames ids came back."""
    return [[(c['geonameid'],
              round(float(c.get('min_dist', 0)), 9),
              round(float(c.get('max_dist', 0)), 9),
              round(float(c.get('ascii_dist', 0)), 9),
              round(float(c.get('adm1_count', 0)), 9),
              round(float(c.get('country_count', 0)), 9),
              c.get('admin1_parent_match'),
              c.get('country_code_parent_match'))
             for c in ent['es_choices']]
            for ent in doc_es]


def test_msearch_matches_sequential(geoparser_all_data):
    """The batched path must return exactly what looping over add_es_data returns.

    This is the core guarantee of using _msearch: it is a transport change, not
    a query change.
    """
    geo = geoparser_all_data
    gs = geo.geonames
    all_doc_ex = _entity_data(geo, TEXTS)

    gs.clear_cache()
    sequential = []
    for doc_ex in all_doc_ex:
        doc_es = [gp.add_es_data(dict(ex), gs, max_results=100) for ex in doc_ex]
        if doc_es:
            gp._add_cross_entity_counts(doc_es)
        sequential.append(doc_es)

    gs.clear_cache()
    batched = gp.add_es_data_batch([[dict(e) for e in d] for d in all_doc_ex],
                                   gs, max_results=100)

    assert len(sequential) == len(batched)
    for seq_doc, bat_doc in zip(sequential, batched):
        assert _signature(seq_doc) == _signature(bat_doc)


def test_batch_dedups_repeated_names(geoparser_all_data):
    """Entities sharing a name should produce one cache entry, not one per mention."""
    geo = geoparser_all_data
    gs = geo.geonames
    gs.clear_cache()
    # Same two place names in every document
    texts = ["Fighting broke out in Aleppo, Syria."] * 6
    gp.add_es_data_batch(_entity_data(geo, texts), gs, max_results=100)
    assert len(gs._es_cache) == 2, f"expected 2 cached lookups, got {len(gs._es_cache)}"


def test_batch_entities_get_independent_copies(geoparser_all_data):
    """Entities sharing a cache key must not share candidate dicts.

    Cross-entity counts are written into the candidates per document, so shared
    references would corrupt the model's features.
    """
    geo = geoparser_all_data
    gs = geo.geonames
    gs.clear_cache()
    texts = ["Fighting broke out in Aleppo, Syria."] * 3
    res = gp.add_es_data_batch(_entity_data(geo, texts), gs, max_results=100)
    first = res[0][0]['es_choices'][0]
    second = res[1][0]['es_choices'][0]
    assert first is not second
    first['min_dist'] = 12345.0
    assert second['min_dist'] != 12345.0
    # and the cached copy must stay pristine
    cached = next(iter(gs._es_cache.values()))
    assert cached[0]['min_dist'] != 12345.0
    # The cache deliberately holds candidates from *before* the cross-entity
    # counts are applied, since those are per-document. If they ever start
    # being cached, entries would leak between documents.
    assert 'adm1_count' not in cached[0]
    assert 'adm1_count' in first


def test_failed_subquery_raises_rather_than_returning_empty(geoparser_all_data):
    """A failed sub-search must not look like 'this place isn't in geonames'.

    _msearch reports per-query failures as a response with a status and no
    'hits'. Silently treating that as zero candidates would make an
    infrastructure error indistinguishable from a genuine miss.
    """
    geo = geoparser_all_data
    gs = geo.geonames
    gs.clear_cache()
    all_doc_ex = _entity_data(geo, TEXTS[:2])

    bad = {"responses": [{"status": 500,
                          "error": {"type": "search_phase_execution_exception"}}]}
    with patch.object(gs.conn, "msearch", return_value=bad):
        with pytest.raises(GeonamesQueryError):
            gp.add_es_data_batch(all_doc_ex, gs, max_results=100)


def test_truncated_msearch_response_raises(geoparser_all_data):
    """A response count that doesn't match the request must be an error.

    A parse-time error rejects the whole _msearch rather than returning
    per-query errors, so zipping responses back onto queries would silently
    misalign results with entities.
    """
    geo = geoparser_all_data
    gs = geo.geonames
    gs.clear_cache()
    all_doc_ex = _entity_data(geo, TEXTS[:3])

    with patch.object(gs.conn, "msearch", return_value={"responses": []}):
        with pytest.raises(GeonamesQueryError):
            gp.add_es_data_batch(all_doc_ex, gs, max_results=100)


def test_add_es_data_doc_matches_batch(geoparser_all_data):
    """The single-document wrapper should agree with the batch path."""
    geo = geoparser_all_data
    gs = geo.geonames
    doc_ex = _entity_data(geo, [TEXTS[2]])[0]

    gs.clear_cache()
    via_doc = gp.add_es_data_doc([dict(e) for e in doc_ex], gs, max_results=100)
    gs.clear_cache()
    via_batch = gp.add_es_data_batch([[dict(e) for e in doc_ex]], gs, max_results=100)[0]
    assert _signature(via_doc) == _signature(via_batch)


def test_geoparse_batch_still_matches_geoparse_doc(geoparser_all_data):
    """End-to-end parity through the full pipeline, on the msearch path."""
    geo = geoparser_all_data
    batch = geo.geoparse_batch(TEXTS)
    for text, b in zip(TEXTS, batch):
        single = geo.geoparse_doc(text)
        assert ({e.get("geonameid") for e in b["geolocated_ents"]}
                == {e.get("geonameid") for e in single["geolocated_ents"]})
