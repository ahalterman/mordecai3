"""Miss / false-positive breakdown for one or two prediction files.

Same categories as the scoping report's §5a residual analysis, so the numbers
are directly comparable.
"""
import json
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.join(os.path.dirname(HERE), "ner")
sys.path.insert(0, PILOT)
os.chdir("/home/andy/projects/mordecai3")

from evalcore import SOURCES, gold_spans, is_demonym, load_docs  # noqa: E402


def analyse(path):
    preds = json.load(open(path))
    miss, fp = Counter(), Counter()
    for src in SOURCES:
        for d in load_docs(src):
            if not d["heldout"]:
                continue
            P = sorted({tuple(x) for x in preds[src].get(str(d["doc_idx"]), [])})
            G = gold_spans(d)
            gset = {(g["start"], g["end"]) for g in G}
            dem = [(g["start"], g["end"]) for g in d["golds"]
                   if g["geonameid"] and is_demonym(g)]
            unl = [(g["start"], g["end"]) for g in d["golds"]
                   if not g["geonameid"]]
            for g in G:
                k = (g["start"], g["end"])
                if k in P:
                    continue
                if g["tok_start"] is None:
                    miss["gold offsets align to no token"] += 1
                elif any(min(p[1], k[1]) - max(p[0], k[0]) > 0 for p in P):
                    miss["boundary error (overlapping span emitted)"] += 1
                elif g["nested_in"] == "ORG":
                    miss["nested in ORG, not found"] += 1
                elif g["nested_in"]:
                    miss[f"nested in {g['nested_in']}, not found"] += 1
                else:
                    miss["flat toponym, not found at all"] += 1
            for p in P:
                if p in gset:
                    continue
                if any(min(p[1], b) - max(p[0], a) > 0 for a, b in dem):
                    fp["lands on a demonym gold row"] += 1
                elif any(min(p[1], b) - max(p[0], a) > 0 for a, b in unl):
                    fp["lands on an unlinked gold row"] += 1
                elif any(min(p[1], b) - max(p[0], a) > 0 for a, b in gset):
                    fp["overlaps a gold span, wrong boundary"] += 1
                else:
                    fp["nothing annotated at that span"] += 1
    return miss, fp


if __name__ == "__main__":
    for path in sys.argv[1:]:
        m, f = analyse(path)
        print(f"\n=== {os.path.basename(path)}  "
              f"misses {sum(m.values())}  false positives {sum(f.values())}")
        for k, v in m.most_common():
            print(f"  miss  {k:48s} {v}")
        for k, v in f.most_common():
            print(f"  FP    {k:48s} {v}")
