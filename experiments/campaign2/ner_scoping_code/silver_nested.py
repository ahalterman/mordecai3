"""How good is the gazetteer rule as a *silver* annotator of nested toponyms?

Gold nested spans do not exist in OntoNotes/CoNLL/WikiANN, so scaling the
tagger beyond TR/LGL/GWN needs silver nested labels. The obvious source is the
rule mordecai3.geoparse.nested_gazetteer_spans already implements: sub-spans of
1-3 tokens inside an ORG/FAC/EVENT/WORK_OF_ART/LAW/PRODUCT entity whose exact
string is a geonames A/P entry.

TR/LGL/GWN annotate those nested toponyms in gold, so the rule's silver
precision and recall can be measured directly, and the precision-raising
filters can be tuned on real data.
"""
import json
import os
import sys
from collections import Counter

import spacy
from spacy.tokens import DocBin

REPO = "/home/andy/projects/mordecai3"
sys.path.insert(0, REPO)
os.chdir(REPO)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from mordecai3.mordecai_utilities import spacy_doc_setup  # noqa: E402
spacy_doc_setup()
from mordecai3.geoparse import (CONTEXT_LABELS, NESTED_LABELS,  # noqa: E402
                               doc_to_ex_expanded, geoparse_labels,
                               nested_gazetteer_spans)
from mordecai3.geonames import GeonamesService  # noqa: E402
from mordecai3.elasticsearch import setup_es_client  # noqa: E402
from evalcore import SOURCES, is_demonym, load_docs  # noqa: E402


def main(split="train"):
    gs = GeonamesService(es_client=setup_es_client())
    nlp = spacy.blank("en")
    tot = Counter()
    by_pop = Counter()
    examples = {"tp": [], "fp": []}
    for src in SOURCES:
        meta = {d["doc_idx"]: d for d in load_docs(src)
                if d["heldout"] == (split == "heldout")}
        db = DocBin().from_disk(f"raw_data/spacyed/source_{src}.spacy")
        for i, doc in enumerate(db.get_docs(nlp.vocab)):
            d = meta.get(i)
            if d is None:
                continue
            gold = {(g["start"], g["end"]) for g in d["golds"]
                    if g["geonameid"] and not is_demonym(g)}
            gold_all = {(g["start"], g["end"]) for g in d["golds"]}
            dem = {(g["start"], g["end"]) for g in d["golds"]
                   if g["geonameid"] and is_demonym(g)}
            # gold toponyms that sit inside one of the swallowing labels
            nested_gold = set()
            for g in d["golds"]:
                if not g["geonameid"] or is_demonym(g):
                    continue
                for e in doc.ents:
                    if e.label_ in NESTED_LABELS and \
                       e.start_char <= g["start"] and g["end"] <= e.end_char and \
                       (e.end_char - e.start_char) > (g["end"] - g["start"]):
                        nested_gold.add((g["start"], g["end"]))
                        break
            base = doc_to_ex_expanded(doc, geo_labels=geoparse_labels(),
                                      context_labels=CONTEXT_LABELS)
            extra = nested_gazetteer_spans(doc, base, gs)
            tot["nested_gold"] += len(nested_gold)
            tot["proposed"] += len(extra)
            for e in extra:
                k = (e["start_char"], e["end_char"])
                if k in gold:
                    tot["silver_tp"] += 1
                    if len(examples["tp"]) < 15:
                        examples["tp"].append(e["search_name"])
                elif k in dem:
                    tot["hits_demonym"] += 1
                elif k in gold_all:
                    tot["hits_unlinked_gold"] += 1
                else:
                    tot["silver_fp"] += 1
                    if len(examples["fp"]) < 40:
                        examples["fp"].append(e["search_name"])
            tot["nested_gold_found"] += len(
                nested_gold & {(e["start_char"], e["end_char"]) for e in extra})
    p = 100 * tot["silver_tp"] / max(tot["proposed"], 1)
    p_lenient = 100 * (tot["silver_tp"] + tot["hits_unlinked_gold"]) / \
        max(tot["proposed"], 1)
    r = 100 * tot["nested_gold_found"] / max(tot["nested_gold"], 1)
    print(f"split={split}", dict(tot))
    print(f"silver precision (strict gold) {p:.1f}%   "
          f"(+ unlinked gold rows) {p_lenient:.1f}%   "
          f"recall of nested gold {r:.1f}%")
    print("TP examples:", examples["tp"])
    print("FP examples:", examples["fp"])
    with open(f"{HERE}/silver_nested_{split}.json", "w") as f:
        json.dump({"counts": dict(tot), "precision": p,
                   "precision_lenient": p_lenient, "recall": r,
                   "examples": examples}, f, indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "train")
