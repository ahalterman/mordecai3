"""Paired per-seed deltas for e50_outlet, on the campaign-2 scoreboard.

Same statistics as the rest of the campaign: five seeds, the two arms differ
only in `--feature-blocks`, so a same-seed difference is the feature and nothing
else.  Significance is the campaign-2 rule, `t(4) > 2.776` (two-sided, 0.05),
not the old `|mean| > 2 SE`.

    uv run python tools/outlet_aggregate.py
"""

import json
import math
import os
import sys

SEEDS = [42, 101, 202, 617, 1848]
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "experiments", "e50_outlet")
T_CRIT = 2.776           # t(4), two-sided, alpha=0.05


def load(arm, seed):
    with open(os.path.join(ROOT, arm, "seed{}.json".format(seed))) as f:
        frozen = json.load(f)
    with open(os.path.join(ROOT, arm, "seed{}.metrics2.json".format(seed))) as f:
        c2 = json.load(f)
    return frozen, c2


def paired(base, arm):
    """mean delta, SE, t for one metric across the seeds."""
    d = [a - b for a, b in zip(arm, base)]
    n = len(d)
    mean = sum(d) / n
    if n < 2:
        return mean, float("nan"), float("nan")
    var = sum((x - mean) ** 2 for x in d) / (n - 1)
    se = math.sqrt(var / n)
    t = mean / se if se > 0 else float("inf") if mean else 0.0
    return mean, se, t


def row(label, base_vals, arm_vals, note=""):
    mean, se, t = paired(base_vals, arm_vals)
    sig = "*" if abs(t) > T_CRIT else " "
    print("{:<44} {:>8.4f} {:>8.4f}  {:+8.4f} +/- {:.4f}  t={:6.2f} {}{}".format(
        label, sum(base_vals) / len(base_vals), sum(arm_vals) / len(arm_vals),
        mean, se, t, sig, note))
    return {"baseline": sum(base_vals) / len(base_vals),
            "arm": sum(arm_vals) / len(arm_vals),
            "delta": mean, "se": se, "t": t, "sig": abs(t) > T_CRIT}


def main():
    data = {}
    for arm in ("baseline", "arm"):
        data[arm] = {s: load(arm, s) for s in SEEDS}

    def c2(arm, path):
        out = []
        for s in SEEDS:
            v = data[arm][s][1]
            for key in path:
                v = v[key]
            out.append(v)
        return out

    def frozen(arm, key):
        return [data[arm][s][0][key] for s in SEEDS]

    print("e50_outlet -- 5 seeds {}, paired, t(4) > {} is significant\n"
          .format(SEEDS, T_CRIT))
    print("{:<44} {:>8} {:>8}  {:>19}  {:>8}".format(
        "metric", "baseline", "outlet", "paired delta", "t"))
    print("-" * 104)

    results = {}
    print("PRIMARY")
    results["tlg_hard"] = row("TLG-hard (TR/LGL/GWN macro, non-country)",
                              c2("baseline", ["tlg_hard"]), c2("arm", ["tlg_hard"]))
    print("\nTHE SLICE THIS FEATURE TARGETS")
    results["lgl_noncountry"] = row("LGL exact match, non-country golds",
                                    c2("baseline", ["per_source", "LGL", "em_noncountry"]),
                                    c2("arm", ["per_source", "LGL", "em_noncountry"]))
    results["lgl_em"] = row("LGL exact match (frozen metric)",
                            frozen("baseline", "LGL_exact_match"),
                            frozen("arm", "LGL_exact_match"))
    results["lgl_novel"] = row("LGL novel-pair exact match",
                               c2("baseline", ["per_source", "LGL", "novel_pair_em"]),
                               c2("arm", ["per_source", "LGL", "novel_pair_em"]))
    results["lgl_twin"] = row("LGL twin-credit",
                              c2("baseline", ["per_source", "LGL", "twin_credit"]),
                              c2("arm", ["per_source", "LGL", "twin_credit"]))

    print("\nTLG-hard's three ingredients (EM on non-country golds)")
    for src in ("TR", "LGL", "GWN"):
        results["nc_" + src] = row(
            "  {} non-country EM".format(src),
            c2("baseline", ["per_source", src, "em_noncountry"]),
            c2("arm", ["per_source", src, "em_noncountry"]))

    print("\nGUARDRAILS")
    results["novel_pair"] = row("novel-pair EM (all sources)",
                                c2("baseline", ["novel_pair_em"]),
                                c2("arm", ["novel_pair_em"]))
    for src in ("Prodigy", "TR", "GWN", "Synth", "WikiDocs"):
        results["src_" + src] = row(
            "  {} exact match (must not regress)".format(src),
            frozen("baseline", src + "_exact_match"),
            frozen("arm", src + "_exact_match"),
            note="  <- structurally has no outlet"
            if src in ("Prodigy", "Synth", "WikiDocs", "GWN") else "")

    print("\nSECONDARY / CONTINUITY")
    results["twin"] = row("twin-credit EM (macro, no Synth)",
                          c2("baseline", ["twin_credit_macro"]),
                          c2("arm", ["twin_credit_macro"]))
    results["macro6"] = row("macro EM, 6 sources (ledger continuity)",
                            frozen("baseline", "exact_match_avg"),
                            frozen("arm", "exact_match_avg"))
    results["macro5"] = row("macro EM, 5 sources (no Synth)",
                            c2("baseline", ["em_conditioned_macro"]),
                            c2("arm", ["em_conditioned_macro"]))
    results["acc161"] = row("acc@161km (macro)",
                            frozen("baseline", "acc_at_161"),
                            frozen("arm", "acc_at_161"))

    print("\nPER-SEED TLG-hard and LGL non-country")
    print("{:<8} {:>10} {:>10} {:>10}   {:>10} {:>10} {:>10}".format(
        "seed", "TLG base", "TLG arm", "delta", "LGLnc base", "LGLnc arm", "delta"))
    for i, s in enumerate(SEEDS):
        tb = c2("baseline", ["tlg_hard"])[i]
        ta = c2("arm", ["tlg_hard"])[i]
        lb = c2("baseline", ["per_source", "LGL", "em_noncountry"])[i]
        la = c2("arm", ["per_source", "LGL", "em_noncountry"])[i]
        print("{:<8} {:>10.4f} {:>10.4f} {:>+10.4f}   {:>10.4f} {:>10.4f} {:>+10.4f}"
              .format(s, tb, ta, ta - tb, lb, la, la - lb))

    # ---- the permuted-home control -------------------------------------
    # Same five columns, same two mask channels bit-identical, but every outlet
    # wears another outlet's home. Anything that survives here is not locality.
    if all(os.path.exists(os.path.join(ROOT, "perm",
                                       "seed{}.metrics2.json".format(s)))
           for s in SEEDS):
        data["perm"] = {s: load("perm", s) for s in SEEDS}
        print("\n" + "=" * 104)
        print("CONTROL: outlet homes permuted within level (masks bit-identical)")
        print("=" * 104)
        print("{:<44} {:>8} {:>8}  {:>19}  {:>8}".format(
            "metric", "baseline", "permuted", "paired delta", "t"))
        results["perm_tlg"] = row("TLG-hard, permuted vs baseline",
                                  c2("baseline", ["tlg_hard"]),
                                  c2("perm", ["tlg_hard"]))
        results["perm_lgl_nc"] = row("LGL non-country, permuted vs baseline",
                                     c2("baseline", ["per_source", "LGL", "em_noncountry"]),
                                     c2("perm", ["per_source", "LGL", "em_noncountry"]))
        results["perm_lgl_em"] = row("LGL exact match, permuted vs baseline",
                                     frozen("baseline", "LGL_exact_match"),
                                     frozen("perm", "LGL_exact_match"))
        results["perm_macro6"] = row("macro EM 6-source, permuted vs baseline",
                                     frozen("baseline", "exact_match_avg"),
                                     frozen("perm", "exact_match_avg"))
        print("\n  real outlet vs permuted outlet (the locality signal itself):")
        results["real_vs_perm_tlg"] = row("  TLG-hard, real vs permuted",
                                          c2("perm", ["tlg_hard"]),
                                          c2("arm", ["tlg_hard"]))
        results["real_vs_perm_lgl"] = row("  LGL non-country, real vs permuted",
                                          c2("perm", ["per_source", "LGL", "em_noncountry"]),
                                          c2("arm", ["per_source", "LGL", "em_noncountry"]))

    out = os.path.join(ROOT, "aggregate.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print("\nwrote {}".format(out))


if __name__ == "__main__":
    sys.exit(main())
