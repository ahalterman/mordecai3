"""Detection-only baselines on the D2 denominator, from the cached DocBins.

No Elasticsearch and no ranker: this scores which character spans the serving
front end proposes, which is all a detection number needs. Variants:

  spacy_ship     GPE/LOC/EVENT_LOC/FAC, untrimmed  (pre-fix path)
  spacy_serving  same labels, trimmed              (what ships today)
  spacy_nofac    trimmed, FAC dropped
  spacy_norp     trimmed + NORP (the D2-forbidden config, for reference)
  spacy_nested   trimmed + the nested gazetteer pass (needs ES)
"""
import json
import os
import sys

import spacy
from spacy.tokens import DocBin

REPO = "/home/andy/projects/mordecai3"
sys.path.insert(0, REPO)
os.chdir(REPO)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from mordecai3.mordecai_utilities import spacy_doc_setup  # noqa: E402
spacy_doc_setup()
from mordecai3.geoparse import (doc_to_ex_expanded, geoparse_labels,  # noqa: E402
                               nested_gazetteer_spans, CONTEXT_LABELS)
from mordecai3.geonames import GeonamesService  # noqa: E402
from evalcore import SOURCES, add, load_docs, prf, score  # noqa: E402

VARIANTS = {
    "spacy_ship":    dict(labels=geoparse_labels(), trim=False, gaz=False),
    "spacy_serving": dict(labels=geoparse_labels(), trim=True,  gaz=False),
    "spacy_nofac":   dict(labels=geoparse_labels(include_fac=False), trim=True, gaz=False),
    "spacy_norp":    dict(labels=geoparse_labels(accept_norp=True), trim=True, gaz=False),
    "spacy_nested":  dict(labels=geoparse_labels(), trim=True,  gaz=True),
}


def run(variant_names):
    gs = None
    if any(VARIANTS[v]["gaz"] for v in variant_names):
        from mordecai3.elasticsearch import setup_es_client
        gs = GeonamesService(es_client=setup_es_client())
    nlp = spacy.blank("en")
    per_source = {v: {} for v in variant_names}
    for src in SOURCES:
        meta = load_docs(src)
        heldout = [d for d in meta if d["heldout"]]
        keep = {d["doc_idx"] for d in heldout}
        db = DocBin().from_disk(f"raw_data/spacyed/source_{src}.spacy")
        docs = {i: d for i, d in enumerate(db.get_docs(nlp.vocab)) if i in keep}
        for v in variant_names:
            cfg = VARIANTS[v]
            preds = {}
            for i, doc in docs.items():
                ex = doc_to_ex_expanded(doc, geo_labels=cfg["labels"],
                                        context_labels=CONTEXT_LABELS,
                                        trim_spans=cfg["trim"])
                if cfg["gaz"]:
                    ex = ex + nested_gazetteer_spans(doc, ex, gs)
                preds[i] = [(e["start_char"], e["end_char"]) for e in ex]
            per_source[v][src] = score(heldout, preds)
    out = {}
    for v in variant_names:
        pooled = {}
        for src in SOURCES:
            pooled = add(pooled, per_source[v][src])
        out[v] = {"pooled": prf(pooled),
                  **{s: prf(per_source[v][s]) for s in SOURCES}}
    return out


if __name__ == "__main__":
    names = sys.argv[1].split(",") if len(sys.argv) > 1 else list(VARIANTS)
    res = run(names)
    with open(f"{HERE}/baseline_detect.json", "w") as f:
        json.dump(res, f, indent=1)
    hdr = f"{'variant':16s} {'P':>6s} {'R':>6s} {'F1':>6s} {'R_ov':>6s} " \
          f"{'R_nest':>7s} {'R_flat':>7s} {'npred':>6s} {'fp_dem':>7s}"
    print(hdr)
    for v in names:
        p = res[v]["pooled"]
        print(f"{v:16s} {p['P']:6.1f} {p['R']:6.1f} {p['F1']:6.1f} "
              f"{p['R_ov']:6.1f} {p['R_nested']:7.1f} {p['R_flat']:7.1f} "
              f"{p['n_pred']:6d} {p['fp_on_demonym']:7d}")
    print("\nper source (recall):")
    for v in names:
        print(f"{v:16s} " + "  ".join(
            f"{s}: P {res[v][s]['P']:.1f} R {res[v][s]['R']:.1f}" for s in SOURCES))
    print("\nD2 gold pooled:", res[names[0]]["pooled"]["n_gold"],
          "of which nested:", res[names[0]]["pooled"]["n_gold_nested"],
          "unalignable:", res[names[0]]["pooled"]["n_gold_unalignable"])
