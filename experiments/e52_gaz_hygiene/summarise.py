"""Collect every res_*.json arm into one table."""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = ["TR", "LGL", "GWN", "WikiDocs", "Prodigy", "Synth"]
TLG = ["TR", "LGL", "GWN"]


def main():
    pat = sys.argv[1] if len(sys.argv) > 1 else "res_*.json"
    rows = []
    for p in sorted(glob.glob(os.path.join(HERE, pat))):
        d = json.load(open(p))
        s = d["summary"]
        name = os.path.basename(p)[4:-5]
        have = [x for x in TLG if x in s]
        tlg_c = sum(s[x]["em_hard_cond"] for x in have) / len(have) if have else float("nan")
        tlg_a = sum(s[x]["em_hard_all"] for x in have) / len(have) if have else float("nan")
        six = [x for x in SRC if x in s]
        macro = sum(s[x]["em_cond"] for x in six) / len(six)
        macro_a = sum(s[x]["em_all"] for x in six) / len(six)
        rows.append((name, tlg_c, tlg_a, macro, macro_a,
                     {x: (s[x]["em_cond"], s[x]["em_all"]) for x in six}))
    print("%-26s %9s %9s %9s %9s" %
          ("arm", "TLGh_c", "TLGh_all", "macro_c", "macro_all"))
    for r in rows:
        print("%-26s %9.4f %9.4f %9.4f %9.4f" % r[:5])
    print()
    print("%-26s %s" % ("arm", "  ".join("%-14s" % x for x in SRC)))
    for r in rows:
        print("%-26s %s" % (r[0], "  ".join(
            "%.4f/%.4f" % r[5][x] if x in r[5] else "     -/-      "
            for x in SRC)))


if __name__ == "__main__":
    main()
