"""Parity gate: the packaged head reproduces its training-harness row exactly.

This is `experiments/e55_ner_head/ship/test_span_head.py` ported to the library
module and the packaged assets, with e55's scorer vendored in (it lived in a
session scratchpad). It runs `mordecai3.span_head.SpanTagger` over the 260
held-out TR/LGL/GWN documents, straight from the cached DocBins the training
harness read, and scores the spans on the D2 denominator.

The numbers must land on the arm's reported row -- the module and the harness
enumerate the same candidates over the same `._.tensor` values, so any gap is a
bug in one of them:

    span_head_2026-08-20_gold.pt   P 85.83  R 89.49  F1 87.62  nested 67.9  demFP 81  preds 2173
    span_head_2026-08-20_all.pt    P 83.95  R 88.87  F1 86.34  nested 76.5  demFP 56  preds 2206

Usage
-----
    uv run python experiments/e56_span_head_serving/parity_span_head.py gold
    uv run python experiments/e56_span_head_serving/parity_span_head.py all

`data/{tr,lgl,gwn}_docs.json` are e55's gold-span dumps (gold offsets, spaCy's
label for each, and whether the gold is nested inside a larger entity), copied
here so the gate stays runnable.
"""
import json
import os
import sys

import spacy
from spacy.tokens import DocBin

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(HERE, "data")
SOURCES = ["tr", "lgl", "gwn"]

sys.path.insert(0, REPO)
os.chdir(REPO)

from mordecai3.mordecai_utilities import spacy_doc_setup  # noqa: E402
from mordecai3.span_head import load_span_tagger  # noqa: E402


# --------------------------------------------------------------------------
# e55's scorer, verbatim in behaviour (scratchpad `ner/evalcore.py`)
# --------------------------------------------------------------------------

def load_docs(src):
    with open(os.path.join(DATA, f"{src}_docs.json")) as f:
        return json.load(f)["docs"]


def is_demonym(g):
    labs = set(g["spacy_labels"] or [])
    norp = ("NORP" in labs) and not (labs & {"GPE", "LOC", "EVENT_LOC"})
    return norp or g.get("gtype") == "Non_Literal_Modifier"


def gold_spans(doc):
    return [g for g in doc["golds"]
            if g["geonameid"] and not is_demonym(g)]


def score(docs, preds_by_doc):
    r = {k: 0 for k in ["n_gold", "n_pred", "tp_exact", "tp_overlap",
                        "n_gold_nested", "tp_nested", "n_gold_flat", "tp_flat",
                        "fp_on_demonym", "fp_on_unlinked", "fp_other"]}
    for d in docs:
        golds = gold_spans(d)
        r["n_gold"] += len(golds)
        gset = {}
        for i, g in enumerate(golds):
            gset.setdefault((g["start"], g["end"]), i)
        preds = sorted(set(preds_by_doc.get(d["doc_idx"], [])))
        r["n_pred"] += len(preds)

        matched = set()
        for p in preds:
            gi = gset.get(p)
            if gi is not None and gi not in matched:
                matched.add(gi)
        r["tp_exact"] += len(matched)
        for i, g in enumerate(golds):
            nested = bool(g["nested_in"])
            r["n_gold_nested" if nested else "n_gold_flat"] += 1
            if i in matched:
                r["tp_nested" if nested else "tp_flat"] += 1

        pairs = [(min(p[1], g["end"]) - max(p[0], g["start"]), pi, gi)
                 for pi, p in enumerate(preds) for gi, g in enumerate(golds)]
        pairs = sorted((x for x in pairs if x[0] > 0), reverse=True)
        up, ug = set(), set()
        for ov, pi, gi in pairs:
            if pi in up or gi in ug:
                continue
            up.add(pi)
            ug.add(gi)
        r["tp_overlap"] += len(up)

        dem = [(g["start"], g["end"]) for g in d["golds"]
               if g["geonameid"] and is_demonym(g)]
        unl = [(g["start"], g["end"]) for g in d["golds"] if not g["geonameid"]]
        for pi, p in enumerate(preds):
            if pi in up:
                continue
            if any(min(p[1], b) - max(p[0], a) > 0 for a, b in dem):
                r["fp_on_demonym"] += 1
            elif any(min(p[1], b) - max(p[0], a) > 0 for a, b in unl):
                r["fp_on_unlinked"] += 1
            else:
                r["fp_other"] += 1
    return r


def prf(r):
    def f(tp, npred, ngold):
        p = tp / npred if npred else 0.0
        rc = tp / ngold if ngold else 0.0
        return p, rc, (2 * p * rc / (p + rc) if p + rc else 0.0)
    pe, re_, fe = f(r["tp_exact"], r["n_pred"], r["n_gold"])
    out = {"P": 100 * pe, "R": 100 * re_, "F1": 100 * fe,
           "R_nested": 100 * r["tp_nested"] / max(r["n_gold_nested"], 1),
           "R_flat": 100 * r["tp_flat"] / max(r["n_gold_flat"], 1)}
    out.update(r)
    return out


def add(a, b):
    return {k: a.get(k, 0) + b.get(k, 0) for k in set(a) | set(b)}


def main(name):
    spacy_doc_setup()
    tagger = load_span_tagger(name)
    print(f"{name}: threshold {tagger.threshold}  "
          f"max_span {tagger.head.max_span}")
    nlp = spacy.blank("en")
    pooled = {}
    for src in SOURCES:
        docs = [d for d in load_docs(src) if d["heldout"]]
        keep = {d["doc_idx"] for d in docs}
        db = DocBin().from_disk(f"raw_data/spacyed/source_{src}.spacy")
        preds = {i: tagger.spans(doc)
                 for i, doc in enumerate(db.get_docs(nlp.vocab)) if i in keep}
        p = prf(score(docs, preds))
        print(f"  {src}: P {p['P']:.2f} R {p['R']:.2f} F1 {p['F1']:.2f} "
              f"nested {p['R_nested']:.1f}")
        pooled = add(pooled, score(docs, preds))
    p = prf(pooled)
    print(f"POOLED n_gold {p['n_gold']} P {p['P']:.2f} R {p['R']:.2f} "
          f"F1 {p['F1']:.2f} nested {p['R_nested']:.1f} "
          f"demonymFP {p['fp_on_demonym']} preds {p['n_pred']}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "gold")
