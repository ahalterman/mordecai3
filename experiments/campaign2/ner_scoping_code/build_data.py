"""Build the span-detection dataset for the NER-retrain pilot.

Reads the TR/LGL/GWN corpora and the already-spaCy'd DocBins (which carry the
en_core_web_trf token tensors), and writes one npz+json per source with

  * per-document token offsets, texts, spaCy entity spans, and the 768-d
    frozen trf tensor per token
  * gold toponym spans mapped to token index ranges
  * the D2 flag: whether a gold row is a demonym (excluded from the task)

Nothing in the shared tree is touched; tools/end_to_end_eval.py is imported
read-only for its corpus reader and held-out split.
"""
import json
import os
import pickle
import sys

import numpy as np
import spacy
from spacy.tokens import DocBin

REPO = "/home/andy/projects/mordecai3"
sys.path.insert(0, os.path.join(REPO, "tools"))
os.chdir(REPO)

from end_to_end_eval import read_corpus, heldout_doc_indices  # noqa: E402
from mordecai3.mordecai_utilities import spacy_doc_setup  # noqa: E402

spacy_doc_setup()

OUT = ("/tmp/claude-1000/-home-andy-projects-mordecai3/"
       "a19357d6-880a-4251-9cdd-97e2235f449c/scratchpad/ner/data")

# Entity labels that routinely swallow a toponym (mordecai3.geoparse.NESTED_LABELS)
NESTED_LABELS = ("ORG", "FAC", "EVENT", "WORK_OF_ART", "LAW", "PRODUCT")


def tokens_in(doc, start, end):
    """Token indices fully inside [start, end) -- train.py's own rule."""
    return [t.i for t in doc if t.idx >= start and t.idx + len(t) <= end]


def build(source):
    articles = read_corpus(source, "raw_data")
    heldout, info = heldout_doc_indices(source, articles, "raw_data")
    nlp = spacy.blank("en")
    db = DocBin().from_disk(f"raw_data/spacyed/source_{source}.spacy")
    docs = list(db.get_docs(nlp.vocab))
    assert len(docs) == len(articles), (len(docs), len(articles))

    docs_out = []
    tensors = []
    tok_offset = 0
    for i, (doc, art) in enumerate(zip(docs, articles)):
        assert doc.text == art["text"], f"{source} doc {i} text mismatch"
        ents = [{"start": e.start, "end": e.end, "label": e.label_,
                 "start_char": e.start_char, "end_char": e.end_char,
                 "text": e.text} for e in doc.ents]
        ent_label_by_tok = {}
        for e in doc.ents:
            for k in range(e.start, e.end):
                ent_label_by_tok[k] = e.label_

        golds = []
        for t in art["toponyms"]:
            toks = tokens_in(doc, t["start"], t["end"])
            labels = sorted({ent_label_by_tok.get(k, "") for k in toks}) if toks else None
            # is the gold span strictly inside a larger spaCy entity?
            nested_in = None
            if toks:
                for e in doc.ents:
                    if e.start <= toks[0] and toks[-1] < e.end and \
                       (e.end - e.start) > (toks[-1] - toks[0] + 1):
                        nested_in = e.label_
                        break
            golds.append({
                "start": t["start"], "end": t["end"], "phrase": t["phrase"],
                "geonameid": t["geonameid"], "gtype": t["gtype"],
                "lat": t["lat"], "lon": t["lon"],
                "tok_start": toks[0] if toks else None,
                "tok_end": (toks[-1] + 1) if toks else None,
                "spacy_labels": labels,
                "nested_in": nested_in,
            })

        arr = np.vstack([tok._.tensor for tok in doc]).astype("float32")
        tensors.append(arr)
        docs_out.append({
            "doc_idx": i,
            "source": source,
            "heldout": i in heldout,
            "text": doc.text,
            "n_tokens": len(doc),
            "tok_offset": tok_offset,
            "tok_idx": [t.idx for t in doc],
            "tok_len": [len(t.text) for t in doc],
            "tok_text": [t.text for t in doc],
            "sent_starts": [t.i for t in doc if t.is_sent_start],
            "ents": ents,
            "golds": golds,
        })
        tok_offset += len(doc)

    os.makedirs(OUT, exist_ok=True)
    np.save(f"{OUT}/{source}_tensors.npy", np.vstack(tensors))
    with open(f"{OUT}/{source}_docs.json", "w") as f:
        json.dump({"info": info, "docs": docs_out}, f)
    return docs_out, info


if __name__ == "__main__":
    for src in ["tr", "lgl", "gwn"]:
        docs, info = build(src)
        n_h = sum(d["heldout"] for d in docs)
        ng = sum(len([g for g in d["golds"] if g["geonameid"]]) for d in docs)
        ngh = sum(len([g for g in d["golds"] if g["geonameid"]])
                  for d in docs if d["heldout"])
        print(f"{src}: {len(docs)} docs ({n_h} heldout), linked golds {ng} "
              f"({ngh} heldout); {info}")
