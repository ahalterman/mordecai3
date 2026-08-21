"""Learning curve for the frozen span head: is it data-limited?

If detection F1 is still climbing at 100% of the 646 TR/LGL/GWN training
documents, the lever is more labelled spans (WikiDocs anchors, silver nested
labels), not more encoder parameters. Same head, same splits, same protocol as
spanpilot.py; only the number of training documents changes.
"""
import json
import os
import random
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir("/home/andy/projects/mordecai3")
sys.path.insert(0, "/home/andy/projects/mordecai3")

import spanpilot as sp  # noqa: E402
from evalcore import SOURCES, load_docs  # noqa: E402


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("roberta-base")
    chunks, tensors = sp.build_chunks(tok)
    offs = {s: {d["doc_idx"]: d["tok_offset"] for d in load_docs(s)}
            for s in SOURCES}
    for ch in chunks:
        b = offs[ch["src"]][ch["doc_idx"]]
        ch["g0"] = b + ch["a"]
        ch["g1"] = b + ch["b"]

    rng = random.Random(0)
    dev_docs = set()
    for src in SOURCES:
        tr = [d["doc_idx"] for d in load_docs(src) if not d["heldout"]]
        rng.shuffle(tr)
        dev_docs |= {(src, i) for i in tr[:max(1, int(0.15 * len(tr)))]}
    for ch in chunks:
        ch["split"] = ("test" if ch["heldout"] else
                       "dev" if (ch["src"], ch["doc_idx"]) in dev_docs
                       else "train")

    train_docs = sorted({(c["src"], c["doc_idx"]) for c in chunks
                         if c["split"] == "train"})
    out = {}
    path = f"{HERE}/curve.json"
    if os.path.exists(path):
        out = json.load(open(path))
    for frac in [0.125, 0.25, 0.5, 1.0]:
        for seed in [42, 101, 202]:
            key = f"{frac}|{seed}"
            if key in out:
                continue
            r = random.Random(1000 + seed)
            keep = set(r.sample(train_docs, max(1, int(frac * len(train_docs)))))
            sub = [c for c in chunks
                   if c["split"] != "train" or (c["src"], c["doc_idx"]) in keep]
            npos = sum(sum(c["labels"]) for c in sub if c["split"] == "train")
            preds, info = sp.run_arm("frozen", seed, sub, tensors, 20, "cuda",
                                     log=lambda *a: None)
            res = sp.full_score(preds)
            p = res["pooled"]
            print(f"frac {frac:5.3f} seed {seed}: docs {len(keep):3d} "
                  f"spans {int(npos):5d}  P {p['P']:5.1f} R {p['R']:5.1f} "
                  f"F1 {p['F1']:5.1f} R_nested {p['R_nested']:5.1f}")
            out[key] = {"frac": frac, "seed": seed, "n_docs": len(keep),
                        "n_spans": npos, "scores": res, "info": info}
            json.dump(out, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
