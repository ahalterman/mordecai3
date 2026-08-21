"""Dump every nested-gazetteer proposal with the features a filter could use,
then score candidate filters offline.

The unfiltered rule is only ~39% precise as a silver annotator (silver_nested.py),
which is the same 9 points of detection precision it costs at serving. This
measures which cheap filters buy the precision back and what recall they cost.
"""
import json
import os
import sys
from collections import Counter, defaultdict

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
from mordecai3.geonames import GeonamesService, hit_sources  # noqa: E402
from mordecai3.elasticsearch import setup_es_client  # noqa: E402
from evalcore import SOURCES, is_demonym, load_docs  # noqa: E402

# The nouns that a bare gazetteer match on a capitalised word inside an
# organisation name is almost never really about.
INSTITUTION_HEADS = {
    "police", "department", "university", "college", "school", "county",
    "sheriff", "office", "hospital", "court", "district", "council", "state",
    "city", "township", "board", "authority", "commission", "airport",
    "fire", "bureau", "agency", "government", "ministry", "parish"}


def main(split="train"):
    gs = GeonamesService(es_client=setup_es_client())
    nlp = spacy.blank("en")
    rows = []
    # corpus-level: how often does each string appear as a standalone GPE/LOC?
    standalone = Counter()
    docs_cache = {}
    for src in SOURCES:
        meta = {d["doc_idx"]: d for d in load_docs(src)
                if d["heldout"] == (split == "heldout")}
        db = DocBin().from_disk(f"raw_data/spacyed/source_{src}.spacy")
        keep = []
        for i, doc in enumerate(db.get_docs(nlp.vocab)):
            if i in meta:
                keep.append((i, doc))
                for e in doc.ents:
                    if e.label_ in ("GPE", "LOC"):
                        standalone[e.text] += 1
        docs_cache[src] = keep

    for src, keep in docs_cache.items():
        meta = {d["doc_idx"]: d for d in load_docs(src)
                if d["heldout"] == (split == "heldout")}
        for i, doc in keep:
            d = meta[i]
            gold = {(g["start"], g["end"]) for g in d["golds"]
                    if g["geonameid"] and not is_demonym(g)}
            doc_standalone = Counter(e.text for e in doc.ents
                                     if e.label_ in ("GPE", "LOC"))
            base = doc_to_ex_expanded(doc, geo_labels=geoparse_labels(),
                                      context_labels=CONTEXT_LABELS)
            extra = nested_gazetteer_spans(doc, base, gs)
            for e in extra:
                lo, hi = e["start_char"], e["end_char"]
                host = None
                for ent in doc.ents:
                    if ent.label_ in NESTED_LABELS and \
                       ent.start_char <= lo and hi <= ent.end_char:
                        host = ent
                        break
                nm = e["search_name"]
                rest = ""
                if host is not None:
                    rest = (host.text[:lo - host.start_char] + " " +
                            host.text[hi - host.start_char:]).lower()
                res = gs.search_by_names([(nm, 5, 0, False, None)])[0]
                pop = 0
                for hit in hit_sources(res):
                    if str(hit.get("name", "")).lower() == nm.lower() or \
                       str(hit.get("asciiname", "")).lower() == nm.lower():
                        try:
                            pop = max(pop, int(hit.get("population") or 0))
                        except (TypeError, ValueError):
                            pass
                rows.append({
                    "src": src, "doc": i, "name": nm,
                    "gold": (lo, hi) in gold,
                    "host_label": host.label_ if host is not None else None,
                    "host_text": host.text if host is not None else None,
                    "rest": rest,
                    "inst_head": any(w in INSTITUTION_HEADS for w in rest.split()),
                    "doc_standalone": doc_standalone.get(nm, 0),
                    "corpus_standalone": standalone.get(nm, 0),
                    "population": pop,
                    "ntok": len(nm.split()),
                })
    with open(f"{HERE}/silver_rows_{split}.json", "w") as f:
        json.dump(rows, f)
    report(rows)


FILTERS = {
    "none": lambda r: True,
    "doc_standalone>=1": lambda r: r["doc_standalone"] >= 1,
    "corpus_standalone>=1": lambda r: r["corpus_standalone"] >= 1,
    "institution head": lambda r: r["inst_head"],
    "pop>=5000": lambda r: r["population"] >= 5000,
    "pop>=50000": lambda r: r["population"] >= 50000,
    "inst OR doc_standalone": lambda r: r["inst_head"] or r["doc_standalone"] >= 1,
    "inst AND pop>=5000": lambda r: r["inst_head"] and r["population"] >= 5000,
    "inst OR (pop>=50k AND doc_standalone)":
        lambda r: r["inst_head"] or (r["population"] >= 50000 and r["doc_standalone"] >= 1),
}


def report(rows):
    tot_gold = sum(r["gold"] for r in rows)
    print(f"{len(rows)} proposals, {tot_gold} of them gold "
          f"({100*tot_gold/max(len(rows),1):.1f}% raw precision)\n")
    print(f"{'filter':38s} {'kept':>6s} {'prec':>6s} {'gold kept':>10s} "
          f"{'% of rule recall':>16s}")
    for name, fn in FILTERS.items():
        kept = [r for r in rows if fn(r)]
        g = sum(r["gold"] for r in kept)
        print(f"{name:38s} {len(kept):6d} "
              f"{100*g/max(len(kept),1):6.1f} {g:10d} "
              f"{100*g/max(tot_gold,1):16.1f}")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[2] == "--report":
        report(json.load(open(f"{HERE}/silver_rows_{sys.argv[1]}.json")))
    else:
        main(sys.argv[1] if len(sys.argv) > 1 else "train")
