"""End-to-end EM on the D2 denominator, for an arbitrary set of detected spans.

Feeds a span set (spaCy's, or the pilot tagger's) through the real retrieval +
ranker path -- tools/end_to_end_eval.score_examples, imported read-only -- and
scores every non-demonym linked gold toponym as detected-and-resolved or not.

Usage:
  e2e_tagger.py spacy_serving
  e2e_tagger.py preds_frozen_42.json
"""
import json
import os
import sys

import numpy as np
import spacy
from spacy.tokens import DocBin

REPO = "/home/andy/projects/mordecai3"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tools"))
os.chdir(REPO)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from mordecai3.mordecai_utilities import spacy_doc_setup  # noqa: E402
spacy_doc_setup()
from mordecai3 import Geoparser  # noqa: E402
from mordecai3.geoparse import (CONTEXT_LABELS, doc_to_ex_expanded,  # noqa: E402
                                geoparse_labels, guess_in_rel,
                                nested_gazetteer_spans)
from end_to_end_eval import score_examples  # noqa: E402
from evalcore import SOURCES, add, gold_spans, load_docs, prf, score  # noqa: E402

MODEL = "experiments/e29_swa_ep15/seed42.pt"


def ex_for_spans(doc, spans):
    """Entity dicts in doc_to_ex_expanded's shape, for arbitrary char spans."""
    doc_tensor = np.mean(np.vstack([t._.tensor for t in doc]), axis=0)
    ctx = [t for e in doc.ents if e.label_ in CONTEXT_LABELS for t in e]
    out = []
    for lo, hi in sorted(set(spans)):
        toks = [t for t in doc if t.idx >= lo and t.idx + len(t.text) <= hi]
        if not toks:
            continue
        own = {t.i for t in toks}
        other = [t for t in ctx if t.i not in own]
        tensor = np.mean(np.vstack([t._.tensor for t in toks]), axis=0)
        out.append({
            "search_name": doc.text[lo:hi],
            "tensor": tensor,
            "doc_tensor": doc_tensor,
            "locs_tensor": (np.mean(np.vstack([t._.tensor for t in other]), axis=0)
                            if other else np.zeros(len(tensor))),
            "sent": toks[0].sent.text,
            "in_rel": guess_in_rel(doc[toks[0].i:toks[-1].i + 1]),
            "start_char": lo, "end_char": hi})
    return out


def main(which):
    geo = Geoparser(model_path=MODEL, feature_blocks="prom,name,cue,sib,geo,shape",
                    oov_bucket_fix=True,
                    model_options={"return_logits": True, "mask_padding": True,
                                   "modern_mlp": True})
    tagger = None
    if which.endswith(".json"):
        tagger = json.load(open(os.path.join(HERE, which)))
    nlp = spacy.blank("en")
    pooled = {}
    per = {}
    for src in SOURCES:
        meta = {d["doc_idx"]: d for d in load_docs(src) if d["heldout"]}
        db = DocBin().from_disk(f"raw_data/spacyed/source_{src}.spacy")
        docs = [(i, d) for i, d in enumerate(db.get_docs(nlp.vocab)) if i in meta]
        all_ex = []
        for i, doc in docs:
            if which == "oracle":
                ex = ex_for_spans(doc, [(g["start"], g["end"])
                                        for g in gold_spans(meta[i])])
            elif tagger is None:
                ex = doc_to_ex_expanded(doc, geo_labels=geoparse_labels(),
                                        context_labels=CONTEXT_LABELS,
                                        trim_spans=True)
                if which == "serving_nested":
                    ex = ex + nested_gazetteer_spans(doc, ex, geo.geonames)
            else:
                ex = ex_for_spans(doc, [tuple(s) for s in
                                        tagger[src].get(str(i), [])])
            all_ex.append(ex)
        _, picks, timing = score_examples(geo, all_ex, max_choices=100)

        r = {k: 0 for k in ["n_gold", "n_pred", "em", "em_nested", "n_nested",
                            "n_emitted", "n_emitted_correct", "det_tp"]}
        for (i, doc), ex, pk in zip(docs, all_ex, picks):
            d = meta[i]
            golds = gold_spans(d)
            r["n_gold"] += len(golds)
            r["n_pred"] += len(ex)
            by_span = {(e["start_char"], e["end_char"]): p
                       for e, p in zip(ex, pk)}
            r["n_emitted"] += sum(1 for p in pk if p is not None)
            for g in golds:
                if g["nested_in"]:
                    r["n_nested"] += 1
                k = (g["start"], g["end"])
                if k not in by_span:
                    continue
                r["det_tp"] += 1
                p = by_span[k]
                if p is not None and str(p.get("geonameid")) == str(g["geonameid"]):
                    r["em"] += 1
                    r["n_emitted_correct"] += 1
                    if g["nested_in"]:
                        r["em_nested"] += 1
        per[src] = dict(r)
        pooled = add(pooled, r)
        print(f"{src}: n_gold {r['n_gold']} detR {100*r['det_tp']/r['n_gold']:.1f} "
              f"e2e EM {100*r['em']/r['n_gold']:.1f} "
              f"emitted {r['n_emitted']} correct {r['n_emitted_correct']} "
              f"({100*r['n_emitted_correct']/max(r['n_emitted'],1):.1f}% output precision)")
    p = pooled
    print(f"\nPOOLED  n_gold {p['n_gold']}  detR {100*p['det_tp']/p['n_gold']:.2f}"
          f"  E2E EM {100*p['em']/p['n_gold']:.2f}"
          f"  nested EM {100*p['em_nested']/max(p['n_nested'],1):.2f}"
          f"  output precision {100*p['n_emitted_correct']/max(p['n_emitted'],1):.2f}"
          f"  ({p['n_emitted_correct']} correct of {p['n_emitted']} emitted)")
    out = f"{HERE}/e2e_{which.replace('.json','')}.json"
    json.dump({"pooled": p, "per_source": per}, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
