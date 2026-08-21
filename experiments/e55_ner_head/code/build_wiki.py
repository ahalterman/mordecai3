"""Cache WikiDocsFull as token tensors + span metadata for the span head.

Reads the 9 already-spaCy'd DocBin shards (43 GB, tensors on the tokens) and
`raw_data/wiki/wiki_docs_full.jsonl` (20,302 docs / 71,475 anchor-linked
toponyms), and writes one compact npz + one json per shard:

  npz   tensors  (T, 768) float16   -- the en_core_web_trf `._.tensor`
        tok_idx  (T,) int32         -- character offset of each token
        tok_len  (T,) int16
        sent_start (T,) bool
        ent_label (T,) int8         -- index into LABELS, 0 = no entity
        ent_start (T,) bool         -- token begins a spaCy entity
        doc_off  (D+1,) int32       -- token slice of each document
  json  per-document text, global index, and the anchor spans

Anchors are PARTIAL annotations: an anchored toponym is a true positive, an
un-anchored one in the same article is unlabelled, not negative. Everything a
partial-annotation loss needs to tell those apart is derivable from the arrays
above (capitalisation from the text, spaCy GPE/LOC from `ent_label`).

Read-only w.r.t. raw_data.
"""
import glob
import json
import os
import sys
import time

import numpy as np
import spacy
from spacy.tokens import DocBin

REPO = "/home/andy/projects/mordecai3"
sys.path.insert(0, REPO)
os.chdir(REPO)
from mordecai3.mordecai_utilities import spacy_doc_setup  # noqa: E402

spacy_doc_setup()

OUT = ("/tmp/claude-1000/-home-andy-projects-mordecai3/"
       "a19357d6-880a-4251-9cdd-97e2235f449c/scratchpad/ner_scale/wiki")

LABELS = ["", "GPE", "LOC", "FAC", "NORP", "ORG", "EVENT", "WORK_OF_ART",
          "LAW", "PRODUCT", "PERSON", "DATE", "CARDINAL", "ORDINAL", "TIME",
          "MONEY", "PERCENT", "QUANTITY", "LANGUAGE", "OTHER"]
LAB_ID = {l: i for i, l in enumerate(LABELS)}


def main():
    os.makedirs(OUT, exist_ok=True)
    records = [json.loads(l) for l in
               open("raw_data/wiki/wiki_docs_full.jsonl", encoding="utf-8")]
    print(f"{len(records)} jsonl records")
    shards = sorted(glob.glob(
        "raw_data/spacyed/source_wiki_docs_full.[0-9][0-9][0-9].spacy"))
    print(f"{len(shards)} shards")
    nlp = spacy.blank("en")
    gi = 0
    total_tok = 0
    total_anchor = 0
    for sh in shards:
        stem = os.path.basename(sh).split(".")[1]
        if os.path.exists(f"{OUT}/wiki_{stem}.json"):
            n = len(json.load(open(f"{OUT}/wiki_{stem}.json"))["docs"])
            gi += n
            print(f"  skip shard {stem} ({n} docs)")
            continue
        t0 = time.time()
        db = DocBin().from_disk(sh)
        tens, tok_idx, tok_len, sent_start = [], [], [], []
        ent_label, ent_start = [], []
        doc_off = [0]
        docs_out = []
        off = 0
        for doc in db.get_docs(nlp.vocab):
            rec = records[gi]
            if doc.text != rec["text"]:
                # keep the index in step: the shard order is the jsonl order
                raise SystemExit(f"text mismatch at global doc {gi}")
            n = len(doc)
            tens.append(np.vstack([t._.tensor for t in doc]).astype("float16"))
            tok_idx.append(np.array([t.idx for t in doc], dtype="int32"))
            tok_len.append(np.array([len(t.text) for t in doc], dtype="int16"))
            ss = np.zeros(n, dtype=bool)
            for t in doc:
                if t.is_sent_start:
                    ss[t.i] = True
            sent_start.append(ss)
            el = np.zeros(n, dtype="int8")
            es = np.zeros(n, dtype=bool)
            for e in doc.ents:
                lid = LAB_ID.get(e.label_, LAB_ID["OTHER"])
                el[e.start:e.end] = lid
                es[e.start] = True
            ent_label.append(el)
            ent_start.append(es)
            anchors = [{"start": t["start"], "end": t["end"],
                        "phrase": t["phrase"], "geonameid": t["geonamesid"]}
                       for t in rec["toponyms"]]
            docs_out.append({"gi": gi, "text": doc.text, "n_tokens": n,
                             "anchors": anchors})
            total_anchor += len(anchors)
            off += n
            doc_off.append(off)
            gi += 1
        # tensors get their own .npy so training can memory-map them
        np.save(f"{OUT}/wiki_{stem}_tensors.npy", np.vstack(tens))
        np.savez(f"{OUT}/wiki_{stem}.npz",
                 tok_idx=np.concatenate(tok_idx),
                 tok_len=np.concatenate(tok_len),
                 sent_start=np.concatenate(sent_start),
                 ent_label=np.concatenate(ent_label),
                 ent_start=np.concatenate(ent_start),
                 doc_off=np.array(doc_off, dtype="int32"))
        with open(f"{OUT}/wiki_{stem}.json", "w") as f:
            json.dump({"docs": docs_out}, f)
        total_tok += off
        print(f"  shard {stem}: {len(docs_out)} docs, {off} tokens "
              f"({time.time() - t0:.0f}s)")
        del db, tens, docs_out
    print(f"done: {gi} docs, {total_tok} tokens, {total_anchor} anchors")


if __name__ == "__main__":
    main()
