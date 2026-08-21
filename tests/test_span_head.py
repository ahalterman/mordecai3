"""The place-span detection head, from the checkpoint up to the serving path.

The head (mordecai3/span_head.py, experiments/campaign2/ner_head_scaling_
report.md) replaces three things in one call when `Geoparser(span_detector=...)`
is set: the GEO_LABELS filter over spaCy's entities, `trim_span_tokens`, and
the opt-in `nested_gazetteer_spans` pass. Four properties have to hold.

1. **The packaged checkpoints are the measured ones.** The detection threshold
   travels inside the checkpoint; a caller who overrides it is off the
   operating point every number in the report was measured at.
2. **The dicts it emits are the dicts the rest of the pipeline expects.** Every
   field except which spans are chosen is computed exactly as
   `doc_to_ex_expanded` computes it, so `add_es_data_batch`, `ProductionData`
   and the ranker are untouched. This is asserted numerically, on a span both
   paths emit.
3. **The behaviour it is for actually happens**: a toponym nested inside an ORG
   comes out beside its host with no gazetteer pass, and a demonym does not
   come out at all.
4. **Nothing changes for a caller who does not use it.** `span_detector=None`
   is the default and leaves the old path in place, tagger unloaded.

The corpus-level parity check -- that this module reproduces the training
harness's detection row exactly on the 260 held-out documents -- is not here,
because it needs the corpora and the cached DocBins:
`experiments/e56_span_head_serving/parity_span_head.py`.
"""

import numpy as np
import pytest

from mordecai3 import Geoparser
from mordecai3.geoparse import CONTEXT_LABELS, doc_to_ex_expanded
from mordecai3.span_head import (SPAN_HEAD_ASSETS, SpanTagger,
                                 load_span_tagger, resolve_span_head)

# The two shipped heads and the operating point each was measured at
# (experiments/e55_ner_head/ship/INTEGRATION.md).
EXPECTED = {"gold": 0.5, "all": 0.3}

# "Pittsburgh" is a toponym nested inside an ORG span that the label filter can
# never reach; "Turkish" is a demonym, which decision D2 puts out of the task.
NESTED_TEXT = ("Police in Pittsburgh said the University of Pittsburgh had "
               "warned Turkish students about travel to New Mexico.")


@pytest.fixture(scope="module")
def nlp():
    import spacy
    from mordecai3.mordecai_utilities import spacy_doc_setup
    spacy_doc_setup()
    nlp = spacy.load("en_core_web_trf")
    nlp.add_pipe("token_tensors")
    return nlp


@pytest.fixture(scope="module")
def doc(nlp):
    return nlp(NESTED_TEXT)


# --------------------------------------------------------------------- 1.


@pytest.mark.parametrize("name,threshold", sorted(EXPECTED.items()))
def test_packaged_head_loads_at_its_measured_threshold(name, threshold):
    tagger = load_span_tagger(name, device="cpu")
    assert isinstance(tagger, SpanTagger)
    assert tagger.threshold == pytest.approx(threshold)
    assert tagger.head.max_span == 8


def test_span_detector_names_resolve_to_packaged_assets():
    for name in SPAN_HEAD_ASSETS:
        assert str(resolve_span_head(name)).endswith(SPAN_HEAD_ASSETS[name][7:])


def test_unknown_span_detector_fails_loudly():
    with pytest.raises(ValueError):
        resolve_span_head("not-a-head")
    with pytest.raises(ValueError):
        resolve_span_head(None)


def test_threshold_override_is_available_for_sweeps():
    tagger = load_span_tagger("gold", device="cpu", threshold=0.9)
    assert tagger.threshold == pytest.approx(0.9)


# --------------------------------------------------------------------- 2.


def test_doc_to_ex_fields_match_the_reference_implementation(doc):
    """Same keys, and the same values on a span both paths emit."""
    tagger = load_span_tagger("gold", device="cpu")
    head = tagger.doc_to_ex(doc, context_labels=CONTEXT_LABELS)
    ref = doc_to_ex_expanded(doc)
    assert head, "the head found nothing in a document full of toponyms"
    assert {k for e in head for k in e} == {k for e in ref for k in e}

    by_span = {(e["start_char"], e["end_char"]): e for e in ref}
    shared = [e for e in head if (e["start_char"], e["end_char"]) in by_span]
    assert shared, "no span in common with the label-filter path to compare"
    for e in shared:
        r = by_span[(e["start_char"], e["end_char"])]
        assert e["search_name"] == r["search_name"]
        assert e["sent"] == r["sent"]
        assert np.allclose(e["tensor"], r["tensor"])
        assert np.allclose(e["doc_tensor"], r["doc_tensor"])
        assert np.allclose(e["locs_tensor"], r["locs_tensor"])


def test_emitted_spans_are_sorted_and_unique(doc):
    tagger = load_span_tagger("all", device="cpu")
    spans = [(e["start_char"], e["end_char"])
             for e in tagger.doc_to_ex(doc, context_labels=CONTEXT_LABELS)]
    assert spans == sorted(spans)
    assert len(spans) == len(set(spans))


# --------------------------------------------------------------------- 3.


def test_nested_toponym_comes_out_beside_its_host(doc):
    """"Pittsburgh" inside "the University of Pittsburgh", no gazetteer pass.

    This is the "all" head's job, and this one sentence is the two heads'
    measured difference in miniature: at their own thresholds "all" emits the
    nested mention and "gold" does not, which is D2 nested detection recall
    76.5 against 67.9.
    """
    spans = load_span_tagger("all", device="cpu").spans(doc)
    org = next(e for e in doc.ents if e.label_ == "ORG")
    nested = [(a, b) for a, b in spans
              if a >= org.start_char and b <= org.end_char]
    assert nested, f"nothing found inside {org.text!r}, spans={spans}"
    assert any(doc.text[a:b] == "Pittsburgh" for a, b in nested)
    # ...and the flat mention of the same name is still there.
    assert sum(1 for a, b in spans if doc.text[a:b] == "Pittsburgh") == 2
    assert sum(1 for a, b in load_span_tagger("gold", device="cpu").spans(doc)
               if doc.text[a:b] == "Pittsburgh") == 1


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_demonyms_are_not_emitted(doc, name):
    tagger = load_span_tagger(name, device="cpu")
    texts = {doc.text[a:b] for a, b in tagger.spans(doc)}
    assert "Turkish" not in texts, texts


def test_spans_carry_no_leading_determiner(doc):
    """What `trim_span_tokens` used to do, learned instead of patched."""
    tagger = load_span_tagger("all", device="cpu")
    for a, b in tagger.spans(doc):
        first = doc.text[a:b].split()[0].lower()
        assert first not in ("the", "a", "an"), doc.text[a:b]


def test_empty_document_is_handled(nlp):
    tagger = load_span_tagger("gold", device="cpu")
    empty = nlp("")
    assert tagger.spans(empty) == []
    assert tagger.doc_to_ex(empty) == []


# --------------------------------------------------------------------- 4.


def test_default_geoparser_has_no_span_head(geoparser_all_data):
    """The flip is the owner's decision; the default path is unchanged."""
    assert geoparser_all_data.span_detector is None
    assert geoparser_all_data.span_tagger is None


def test_geoparser_with_span_head_resolves_a_nested_toponym(
        geonames_service_all_data, geoparser_all_data):
    geo = Geoparser(geonames=geonames_service_all_data,
                    nlp=geoparser_all_data.nlp, span_detector="all")
    assert geo.span_tagger is not None
    res = geo.geoparse_doc(NESTED_TEXT)
    ents = res["geolocated_ents"]
    names = [e["search_name"] for e in ents]
    assert names.count("Pittsburgh") == 2, names
    assert "Turkish" not in names, names
    placed = {e["search_name"] for e in ents if not e.get("no_match")}
    assert "New Mexico" in placed, ents


def test_nested_gazetteer_pass_is_inert_under_the_head(
        geonames_service_all_data, geoparser_all_data):
    """Both on is a caller mistake, not a crash: the head wins, loudly."""
    geo = Geoparser(geonames=geonames_service_all_data,
                    nlp=geoparser_all_data.nlp, span_detector="all",
                    nested_gazetteer_pass=True)
    res = geo.geoparse_doc(NESTED_TEXT)
    assert [e["search_name"] for e in res["geolocated_ents"]].count(
        "Pittsburgh") == 2
