"""Serving cost of the span head, on the held-out documents.

Measures only the extra work the head adds: the spaCy pass and its `._.tensor`
values are already paid for by the ranker, so the head's cost is the candidate
enumeration plus one batched forward. Compared against
`nested_gazetteer_spans`, the pass it replaces.
"""
import os
import statistics
import sys
import time

import spacy
from spacy.tokens import DocBin

HERE = os.path.dirname(os.path.abspath(__file__))
SHIP = os.path.join(HERE, "ship")
PILOT = os.path.join(os.path.dirname(HERE), "ner")
REPO = "/home/andy/projects/mordecai3"
for p in (REPO, SHIP, PILOT):
    sys.path.insert(0, p)
os.chdir(REPO)

from mordecai3.mordecai_utilities import spacy_doc_setup  # noqa: E402
spacy_doc_setup()
from mordecai3.geoparse import (CONTEXT_LABELS, doc_to_ex_expanded,  # noqa: E402
                                geoparse_labels, nested_gazetteer_spans)
from mordecai3.geonames import GeonamesService  # noqa: E402
from mordecai3.elasticsearch import setup_es_client  # noqa: E402
from evalcore import SOURCES, load_docs  # noqa: E402
from span_head import SpanTagger  # noqa: E402


def main(ckpt, n=120):
    tagger = SpanTagger.load(ckpt)
    gs = GeonamesService(es_client=setup_es_client())
    nlp = spacy.blank("en")
    docs = []
    for src in SOURCES:
        meta = {d["doc_idx"] for d in load_docs(src) if d["heldout"]}
        db = DocBin().from_disk(f"raw_data/spacyed/source_{src}.spacy")
        for i, doc in enumerate(db.get_docs(nlp.vocab)):
            if i in meta:
                docs.append(doc)
        if len(docs) >= n:
            break
    docs = docs[:n]
    print(f"{len(docs)} documents, mean {sum(len(d) for d in docs)/len(docs):.0f} tokens")

    for _ in range(3):                       # warm up
        tagger.doc_to_ex(docs[0])

    t_head, t_base, t_nested = [], [], []
    for d in docs:
        t0 = time.perf_counter(); tagger.doc_to_ex(d); t_head.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        ex = doc_to_ex_expanded(d, geo_labels=geoparse_labels(),
                                context_labels=CONTEXT_LABELS, trim_spans=True)
        t_base.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); nested_gazetteer_spans(d, ex, gs); t_nested.append(time.perf_counter() - t0)

    for name, ts in (("span head (doc_to_ex)", t_head),
                     ("doc_to_ex_expanded (today)", t_base),
                     ("nested_gazetteer_spans (replaced)", t_nested)):
        ms = [1000 * t for t in ts]
        print(f"{name:36s} mean {statistics.mean(ms):6.2f} ms  "
              f"median {statistics.median(ms):6.2f} ms")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1
         else f"{SHIP}/span_head_gold_42.pt")
