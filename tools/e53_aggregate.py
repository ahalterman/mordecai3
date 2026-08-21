"""e53: paired deltas for the outlet-dropout arms against both references.

Condition (a) -- outlet present at eval -- comes from the campaign-2 scoreboard
each run already writes. Condition (b) -- outlet withheld -- comes from
`tools/outlet_conditions_eval.py`. Condition (c) is the guardrail block.

Two reference arms, because the two questions are different:
  vs e29 (baseline, no outlet block)  -- did we keep the win?
  vs e50 (outlet, no dropout)         -- what did dropout cost?

    uv run python tools/e53_aggregate.py
"""

import json
import math
import os
import sys

SEEDS = [42, 101, 202, 617, 1848]
EXP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "experiments")
# The two reference arms are e50's; the dropout arms are e53's.
ARM_DIR = {"baseline": os.path.join(EXP, "e50_outlet", "baseline"),
           "arm": os.path.join(EXP, "e50_outlet", "arm"),
           "d50": os.path.join(EXP, "e53_outlet_dropout", "d50"),
           "d30": os.path.join(EXP, "e53_outlet_dropout", "d30")}
T_CRIT = 2.776
ARMS = ["baseline", "arm", "d50", "d30"]


def load(arm, seed):
    with open(os.path.join(ARM_DIR[arm], "seed{}.json".format(seed))) as f:
        frozen = json.load(f)
    with open(os.path.join(ARM_DIR[arm], "seed{}.metrics2.json".format(seed))) as f:
        c2 = json.load(f)
    return frozen, c2


def paired(base, arm):
    d = [a - b for a, b in zip(arm, base)]
    n = len(d)
    mean = sum(d) / n
    var = sum((x - mean) ** 2 for x in d) / (n - 1)
    se = math.sqrt(var / n)
    t = mean / se if se > 0 else (float("inf") if mean else 0.0)
    return mean, se, t


def main():
    have = {}
    for arm in ARMS:
        if all(os.path.exists(os.path.join(ARM_DIR[arm],
                                           "seed{}.metrics2.json".format(s)))
               for s in SEEDS):
            have[arm] = {s: load(arm, s) for s in SEEDS}
    print("arms with all five seeds: {}\n".format(", ".join(have)))

    def c2(arm, path):
        out = []
        for s in SEEDS:
            v = have[arm][s][1]
            for k in path:
                v = v[k]
            out.append(v)
        return out

    def fr(arm, key):
        return [have[arm][s][0][key] for s in SEEDS]

    def block(title, getter, refs=("baseline", "arm")):
        print(title)
        print("{:<34} {:>8}".format("arm", "mean") +
              "".join("   {:>26}".format("vs " + r) for r in refs if r in have))
        for arm in ARMS:
            if arm not in have:
                continue
            vals = getter(arm)
            line = "{:<34} {:>8.4f}".format(arm, sum(vals) / len(vals))
            for r in refs:
                if r not in have or r == arm:
                    line += "   {:>26}".format("--" if r == arm else "")
                    continue
                m, se, t = paired(getter(r), vals)
                line += "   {:+8.4f} +/- {:.4f} t={:5.2f}{}".format(
                    m, se, t, "*" if abs(t) > T_CRIT else " ")
            print(line)
        print()

    print("=" * 100)
    print("CONDITION (a) -- outlet PRESENT at eval")
    print("=" * 100)
    block("TLG-hard (primary)", lambda a: c2(a, ["tlg_hard"]))
    block("LGL non-country EM",
          lambda a: c2(a, ["per_source", "LGL", "em_noncountry"]))
    block("LGL novel-pair EM",
          lambda a: c2(a, ["per_source", "LGL", "novel_pair_em"]))
    block("novel-pair EM, all sources", lambda a: c2(a, ["novel_pair_em"]))

    print("=" * 100)
    print("CONDITION (c) -- guardrails")
    print("=" * 100)
    for src in ("Prodigy", "TR", "GWN", "Synth", "WikiDocs"):
        block("{} exact match".format(src), lambda a, s=src: fr(a, s + "_exact_match"))

    print("=" * 100)
    print("continuity")
    print("=" * 100)
    block("macro EM, 6 sources", lambda a: fr(a, "exact_match_avg"))
    block("twin-credit macro (no Synth)", lambda a: c2(a, ["twin_credit_macro"]))

    print("PER-SEED TLG-hard")
    print("{:<8}".format("seed") + "".join("{:>12}".format(a) for a in ARMS if a in have))
    for i, s in enumerate(SEEDS):
        print("{:<8}".format(s) + "".join(
            "{:>12.4f}".format(c2(a, ["tlg_hard"])[i]) for a in ARMS if a in have))


if __name__ == "__main__":
    sys.exit(main())
