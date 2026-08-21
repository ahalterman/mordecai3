"""Text-window ceiling for e51_state_abbrev.

The mention-only sizing asks what changes if abbreviations that are ALREADY
annotated toponyms are aliased.  This asks the bigger question: if we scanned
the raw document text for state abbreviations (not just annotated mentions),
how many currently-wrong held-out entities have their gold's ADM1 named by an
abbreviation in the document?
"""
import hashlib
import os
import pickle
import re
import sys

import numpy as np
import pandas as pd
import spacy
from spacy.tokens import DocBin

ROOT = "/home/andy/projects/mordecai3"
sys.path.insert(0, ROOT)
from mordecai3.mordecai_utilities import spacy_doc_setup  # noqa: E402
from mordecai3.candidate_features import norm, is_null_choice  # noqa: E402

sys.path.insert(0, "/tmp/claude-1000/-home-andy-projects-mordecai3/"
                   "a19357d6-880a-4251-9cdd-97e2235f449c/scratchpad")
from sizing import STATES, CA_PROV, AP, TRAIN_FRAC, split_val, gold_idx  # noqa: E402

DATA = os.path.join(ROOT, "raw_data", "pickled_es")
TEMPLATE = "es_formatted_{s}_500_all_loc_types_fuzzy_0_enriched.pkl"

UPPER = dict(STATES)
UPPER.update(CA_PROV)

# AP dotted forms, capitalised, as they appear in copy: "Ill.", "S.C.",
# "W. Va.".  Build a regex per canonical name.
AP_PATTERNS = []
for key, full in AP.items():
    # key is lowercase with internal dots, e.g. "s.c", "w.va", "calif"
    parts = key.split(".")
    body = r"\.\s?".join(re.escape(p.capitalize()) for p in parts)
    AP_PATTERNS.append((re.compile(r"\b" + body + r"\."), full.lower()))

USPS_RE = re.compile(r"\b(" + "|".join(sorted(UPPER)) + r")\b")
USPS_COMMA_RE = re.compile(r",\s*(" + "|".join(sorted(UPPER)) + r")\b")


def text_aliases(text):
    """(loose set, comma-guarded set) of full ADM1 names named by abbreviation."""
    loose, guarded = set(), set()
    for rx, full in AP_PATTERNS:
        if rx.search(text):
            loose.add(full)
            guarded.add(full)
    for m in USPS_RE.finditer(text):
        loose.add(UPPER[m.group(1)].lower())
    for m in USPS_COMMA_RE.finditer(text):
        guarded.add(UPPER[m.group(1)].lower())
    return loose, guarded


def doc_key(doc):
    dt = np.mean(np.vstack([t._.tensor for t in doc]), axis=0)
    return hashlib.sha1(dt.tobytes()).hexdigest()


def main():
    spacy_doc_setup()
    nlp = spacy.blank("en")
    pq = pd.read_parquet(
        os.path.join(ROOT, "experiments/campaign2/preds/e29_seed42_w100.parquet"))

    out = []
    for label, stem in [("TR", "tr"), ("LGL", "lgl"), ("GWN", "gwn")]:
        db = DocBin().from_disk(os.path.join(ROOT, "raw_data", "spacyed",
                                             f"source_{stem}.spacy"))
        key2text = {}
        for doc in db.get_docs(nlp.vocab):
            key2text[doc_key(doc)] = doc.text
        del db
        with open(os.path.join(DATA, TEMPLATE.format(s=stem)), "rb") as f:
            data = pickle.load(f)
        val = split_val(data)
        pqs = pq[pq.source == label].set_index("idx")
        miss = 0
        for i, e in enumerate(val):
            text = key2text.get(e["doc_key"])
            if text is None:
                miss += 1
                continue
            loose, guarded = text_aliases(text)
            gi = gold_idx(e)
            gold_a1 = ""
            gold_fc = ""
            gold_cc = ""
            already = False
            if gi is not None and gi < len(e["es_choices"]):
                ch = e["es_choices"][gi]
                if not is_null_choice(ch):
                    gold_a1 = norm(ch.get("admin1_name"))
                    gold_fc = str(ch.get("feature_code", ""))
                    gold_cc = str(ch.get("country_code3", ""))
                    already = float(ch.get("sib_adm1", 0.0)) > 0.0
            out.append(dict(
                source=label, idx=i, name=str(e["search_name"]),
                gold_a1=gold_a1, gold_fc=gold_fc, gold_cc=gold_cc,
                already_sib_adm1=already,
                text_loose=gold_a1 in loose and gold_a1 != "",
                text_guarded=gold_a1 in guarded and gold_a1 != "",
                n_loose=len(loose), n_guarded=len(guarded),
                correct42=bool(pqs.loc[i, "correct"]) if i in pqs.index else None,
                in_win42=bool(pqs.loc[i, "gold_in_window"]) if i in pqs.index else None,
            ))
        print(f"{label}: {len(val)} held-out, {miss} docs unmatched")
        del data, val, key2text

    df = pd.DataFrame(out)
    df.to_parquet("/tmp/claude-1000/-home-andy-projects-mordecai3/"
                  "a19357d6-880a-4251-9cdd-97e2235f449c/scratchpad/sizing_text.parquet")

    t = df[(df.in_win42 == True)]  # noqa: E712
    t = t[~t.gold_fc.str.startswith("PCL")]
    base = t.groupby("source").correct42.mean()
    print("\nTLG-hard baseline (seed42 w100):", round(base.mean(), 4))
    print(base)
    for col in ("text_loose", "text_guarded"):
        new = t[col] & ~t.already_sib_adm1
        flip = new & (t.correct42 == False)  # noqa: E712
        ceil = (t.correct42 | flip).groupby(t.source).mean()
        print(f"\n-- {col}: gold ADM1 named by abbrev in text, not already a "
              f"sibling")
        print("   entities:", int(new.sum()),
              " of which currently wrong:", int(flip.sum()),
              f" (total wrong {int((t.correct42 == False).sum())})")
        print("   per source new:", dict(new.groupby(t.source).sum()))
        print("   per source flip:", dict(flip.groupby(t.source).sum()))
        print("   ceiling TLG-hard macro:", round(ceil.mean(), 4),
              " delta:", round(ceil.mean() - base.mean(), 4))
    # US-only view
    us = t[t.gold_cc == "USA"]
    print("\nUS golds in TLG-hard:", len(us), " wrong:",
          int((us.correct42 == False).sum()))  # noqa: E712


if __name__ == "__main__":
    main()
