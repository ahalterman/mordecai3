"""Per-entity diff between two eval_hygiene runs: who flipped, and why.

Splits flips into DIRECT (the entity's own candidate list was edited) and
KNOCK-ON (a different mention in the same document was edited, so this entity's
document features moved).  The knock-on column is the honest cost of a
retrieval fix: an unretrievable abbreviation currently injects a garbage anchor
into its document's sibling geometry, and repairing it moves its neighbours.

    python diff_arms.py base.json arm.json [--show 40]
"""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/andy/projects/mordecai3")
import hygiene  # noqa: E402


def load(p):
    with open(p) as f:
        d = json.load(f)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("arm")
    ap.add_argument("--show", type=int, default=40)
    a = ap.parse_args()
    b, r = load(a.base), load(a.arm)

    print("base rules %s / arm rules %s" % (b["rules"], r["rules"]))
    gained, lost = [], []
    for src in b["per_entity"]:
        bb, rr = b["per_entity"][src], r["per_entity"][src]
        assert len(bb) == len(rr), src
        touched_docs = set()
        for x, y in zip(bb, rr):
            if x["pred"] != y["pred"] or x["retrievable"] != y["retrievable"]:
                touched_docs.add(x.get("doc_key"))
        for x, y in zip(bb, rr):
            if x["correct"] == y["correct"]:
                continue
            rec = dict(source=src, name=x["name"], gold=x["gold"],
                       base_pred=x["pred"], arm_pred=y["pred"],
                       was_retrievable=x["retrievable"],
                       now_retrievable=y["retrievable"],
                       n_before=x.get("n_choices"), n_after=y.get("n_choices"))
            (gained if y["correct"] else lost).append(rec)

    def kind(rec):
        """DIRECT = this mention's own list was edited; else a doc knock-on."""
        if rec.get("n_before") is not None and rec.get("n_after") is not None \
                and rec["n_before"] != rec["n_after"]:
            return "direct"
        if hygiene.alias_query(rec["name"] or ""):
            return "direct"
        if rec["was_retrievable"] != rec["now_retrievable"]:
            return "direct"
        return "knock-on"

    print("\nGAINED %d, LOST %d, net %+d"
          % (len(gained), len(lost), len(gained) - len(lost)))
    print("  direct  : +%d / -%d"
          % (sum(kind(x) == "direct" for x in gained),
             sum(kind(x) == "direct" for x in lost)))
    print("  knock-on: +%d / -%d"
          % (sum(kind(x) == "knock-on" for x in gained),
             sum(kind(x) == "knock-on" for x in lost)))
    for tag, rows in (("GAINED", gained), ("LOST", lost)):
        print("\n== %s ==" % tag)
        c = Counter(x["name"] for x in rows)
        for name, n in c.most_common(a.show):
            ex = [x for x in rows if x["name"] == name][0]
            print("  %-24s x%-3d %-6s gold=%-10s base=%-10s arm=%-10s ret %s->%s"
                  % (name[:24], n, ex["source"], ex["gold"], ex["base_pred"],
                     ex["arm_pred"], ex["was_retrievable"],
                     ex["now_retrievable"]))


if __name__ == "__main__":
    main()
