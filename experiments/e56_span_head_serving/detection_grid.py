"""Detection-only rows for the three span sources, on e55's scorer.

`tools/end_to_end_eval.py` reports detection precision/recall, but it has no
demonym-false-positive column: under D2 a demonym gold row is out of the
denominator, and a prediction landing on one is pooled with the corpora's
unlinked gold rows. e55's scorer separates them, and the demonym FP count is a
first-class metric of the head's report, so it is computed here with e55's
scorer -- which is why this file also re-prints P/R/F1 on that convention.

Three span sources over the same 260 held-out documents, from the cached
DocBins:

  serving   spaCy entities filtered to GEO_LABELS and trimmed by
            `trim_span_tokens` -- `Geoparser(span_detector=None)` today
  gold      the packaged "gold" head at its own threshold
  all       the packaged "all" (C_all) head at its own threshold

Ranker-independent: detection does not depend on which checkpoint resolves.

    uv run python experiments/e56_span_head_serving/detection_grid.py
"""
import json
import os
import sys

import spacy
from spacy.tokens import DocBin

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
os.chdir(REPO)

from parity_span_head import SOURCES, add, load_docs, prf, score  # noqa: E402

from mordecai3.geoparse import GEO_LABELS, trim_span_tokens  # noqa: E402
from mordecai3.mordecai_utilities import spacy_doc_setup  # noqa: E402
from mordecai3.span_head import load_span_tagger  # noqa: E402


def serving_spans(doc):
    """`doc_to_ex_expanded`'s spans, without needing the token tensors."""
    out = []
    for ent in doc.ents:
        if ent.label_ not in GEO_LABELS:
            continue
        own = trim_span_tokens(list(ent))
        if not own:
            continue
        out.append((own[0].idx, own[-1].idx + len(own[-1].text)))
    return out


def main():
    spacy_doc_setup()
    sources = {"serving": serving_spans}
    for name in ("gold", "all"):
        tagger = load_span_tagger(name)
        sources[name] = tagger.spans

    vocab = spacy.blank("en").vocab
    pooled = {k: {} for k in sources}
    per_corpus = {k: {} for k in sources}
    for src in SOURCES:
        docs = [d for d in load_docs(src) if d["heldout"]]
        keep = {d["doc_idx"] for d in docs}
        cached = {i: doc for i, doc in
                  enumerate(DocBin().from_disk(
                      f"raw_data/spacyed/source_{src}.spacy").get_docs(vocab))
                  if i in keep}
        for name, fn in sources.items():
            r = score(docs, {i: fn(doc) for i, doc in cached.items()})
            pooled[name] = add(pooled[name], r)
            per_corpus[name][src] = {k: float(v) for k, v in prf(r).items()}

    print(f"{'span source':<10}{'n_pred':>8}{'P':>8}{'R':>8}{'F1':>8}"
          f"{'R_flat':>9}{'R_nested':>10}{'demFP':>7}{'unlFP':>7}{'othFP':>7}")
    for name in ("serving", "gold", "all"):
        p = prf(pooled[name])
        print(f"{name:<10}{p['n_pred']:>8}{p['P']:>8.2f}{p['R']:>8.2f}"
              f"{p['F1']:>8.2f}{p['R_flat']:>9.1f}{p['R_nested']:>10.1f}"
              f"{p['fp_on_demonym']:>7}{p['fp_on_unlinked']:>7}"
              f"{p['fp_other']:>7}")
    print(f"n_gold {prf(pooled['serving'])['n_gold']}  "
          f"nested golds {prf(pooled['serving'])['n_gold_nested']}")

    print(f"\nper corpus, detection F1:\n{'span source':<10}" +
          "".join(f"{s:>10}" for s in SOURCES))
    for name in ("serving", "gold", "all"):
        print(f"{name:<10}" +
              "".join(f"{per_corpus[name][s]['F1']:>10.2f}" for s in SOURCES))
    # loco.py reads the `serving` row: an out-of-family head has to beat the
    # spaCy label filter on the corpus it never saw.
    with open(os.path.join(HERE, "detection_per_corpus.json"), "w") as f:
        json.dump(per_corpus["serving"], f, indent=1)
    with open(os.path.join(HERE, "detection_per_corpus_all.json"), "w") as f:
        json.dump(per_corpus, f, indent=1)
    with open(os.path.join(HERE, "detection_pooled.json"), "w") as f:
        json.dump({k: {a: float(b) for a, b in prf(v).items()}
                   for k, v in pooled.items()}, f, indent=1)


if __name__ == "__main__":
    main()
