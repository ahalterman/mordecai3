"""e57 gate 2: reproduce e52's R1 row with the MAINLINE alias table.

e52 measured R1 from a worktree, against its own copy of the table in
`experiments/e52_gaz_hygiene/hygiene.py`. e57 canonicalises that table as
`mordecai3/place_aliases.py`, so the gate is: swap the mainline table into
e52's own harness and get e52's own numbers back.

    seed101, window 100, TLG-hard em_all   0.8244 -> 0.8444   (+0.0200)
    entity flips over all six sources      +28 / -0

Nothing here runs Elasticsearch writes; `es_util` issues only _mget/_search.
The harness lives in the e52 worktree and is imported read-only.

    uv run python experiments/e57_r1_retrieval/frozen_gate.py \
        --checkpoint experiments/e29_swa_ep15/seed101.pt --window 100
"""
import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import torch

ROOT = "/home/andy/projects/mordecai3"
E52 = os.path.join(ROOT, ".claude/worktrees/agent-a8c4f7c56da33eaf9/"
                         "experiments/e52_gaz_hygiene")
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (E52, ROOT, os.path.join(ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

import es_util            # noqa: E402  (e52 worktree, read-only ES)
import eval_hygiene       # noqa: E402
import hygiene            # noqa: E402
from twin_credit_eval import SOURCES, build_model  # noqa: E402

from mordecai3 import place_aliases  # noqa: E402

TLG = ["TR", "LGL", "GWN"]


def table_parity():
    """The mainline table must agree with e52's on every string that matters."""
    assert hygiene.STATES == place_aliases.STATES, "STATES drifted"
    assert hygiene.CA_PROV == place_aliases.CA_PROV, "CA_PROV drifted"
    assert hygiene.AP == place_aliases.AP, "AP drifted"
    assert hygiene.UPPER == place_aliases.BARE_CODES, "bare-code table drifted"

    # QUERY_OVERRIDE differs by one identity entry ("Northwest Territories" ->
    # itself), which cannot change any output. Prove that rather than assert
    # the dicts are equal: probe every table key in every casing/punctuation
    # form the guards care about, plus the false-positive words.
    probes = []
    for k in set(hygiene.UPPER) | set(hygiene.AP):
        probes += [k, k.upper(), k.lower(), k.title(),
                   k + ".", k.upper() + ".", k.title() + ".",
                   f"in {k}", f"{k} County", f" {k} ", k + ".."]
    probes += ["", "La", "LA", "L.A.", "Miss.", "Man.", "Del.", "Ore.",
               "Ind.", "D.C.", "U.S.", "UK", "Springfield", "New York"]
    bad = [s for s in probes
           if hygiene.alias_query(s) != place_aliases.alias_query(s)
           or hygiene.alias_targets(s) != place_aliases.alias_targets(s)]
    assert not bad, f"alias output drifted on {bad[:10]}"
    print(f"table parity: OK ({len(set(probes))} probe strings, "
          f"{len(hygiene.UPPER)} bare codes, {len(hygiene.AP)} AP forms)")


def summarise_arm(per_entity, fcmap):
    return {s: eval_hygiene.summarise(rows, fcmap)
            for s, rows in per_entity.items()}


def flips(base_pe, arm_pe):
    gained, lost = [], []
    for src in base_pe:
        for x, y in zip(base_pe[src], arm_pe[src]):
            if x["correct"] == y["correct"]:
                continue
            rec = dict(source=src, name=x["name"], gold=x["gold"],
                       base_pred=x["pred"], arm_pred=y["pred"],
                       was_retrievable=x["retrievable"],
                       now_retrievable=y["retrievable"],
                       n_before=x.get("n_choices"), n_after=y.get("n_choices"))
            (gained if y["correct"] else lost).append(rec)

    def kind(rec):
        if (rec.get("n_before") is not None and rec.get("n_after") is not None
                and rec["n_before"] != rec["n_after"]):
            return "direct"
        if place_aliases.alias_query(rec["name"] or ""):
            return "direct"
        if rec["was_retrievable"] != rec["now_retrievable"]:
            return "direct"
        return "knock-on"

    return gained, lost, kind


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                    default=os.path.join(ROOT,
                                         "experiments/e29_swa_ep15/seed101.pt"))
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--sources", default="TR,LGL,GWN,WikiDocs,Prodigy,Synth")
    ap.add_argument("--out", default=os.path.join(HERE, "frozen_gate.json"))
    a = ap.parse_args()

    table_parity()
    # The gate measures the MAINLINE table through e52's harness.
    hygiene.alias_query = place_aliases.alias_query
    hygiene.alias_targets = place_aliases.alias_targets

    want = a.sources.split(",")
    srcs = [(s, st) for s, st in SOURCES if s in want]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, cfg = build_model(a.checkpoint, a.checkpoint + ".json", device)

    t0 = time.time()
    per_entity, stats = {}, {}
    alias_cache, es_cache = {}, {}
    for arm, rules in (("base", []), ("abbrev", ["abbrev"])):
        _, pe, st = eval_hygiene.run_arm(model, cfg, a.window, rules, "p", 100,
                                         srcs, False, alias_cache, es_cache)
        per_entity[arm], stats[arm] = pe, st
        print(f"  {arm}: {st}")

    golds = {r["gold"] for pe in per_entity.values()
             for rows in pe.values() for r in rows}
    fcmap = {g: (s or {}).get("feature_code", "")
             for g, s in es_util.mget(list(golds)).items()}
    for g in golds:
        fcmap.setdefault(g, "")

    summ = {arm: summarise_arm(pe, fcmap) for arm, pe in per_entity.items()}
    print("\n%-9s %7s %7s %7s %7s" %
          ("source", "cond_b", "cond_a", "all_b", "all_a"))
    for s in want:
        b, r = summ["base"].get(s), summ["abbrev"].get(s)
        if not b:
            continue
        print("%-9s %7.4f %7.4f %7.4f %7.4f"
              % (s, b["em_hard_cond"], r["em_hard_cond"],
                 b["em_hard_all"], r["em_hard_all"]))
    tlg = [s for s in TLG if s in summ["base"]]
    head = {}
    if len(tlg) == 3:
        for arm in ("base", "abbrev"):
            head[arm + "_tlg_hard_cond"] = float(
                np.mean([summ[arm][s]["em_hard_cond"] for s in tlg]))
            head[arm + "_tlg_hard_all"] = float(
                np.mean([summ[arm][s]["em_hard_all"] for s in tlg]))
        print("\nTLG-hard em_all  %.4f -> %.4f  (%+.4f)"
              % (head["base_tlg_hard_all"], head["abbrev_tlg_hard_all"],
                 head["abbrev_tlg_hard_all"] - head["base_tlg_hard_all"]))
        print("TLG-hard em_cond %.4f -> %.4f  (%+.4f)"
              % (head["base_tlg_hard_cond"], head["abbrev_tlg_hard_cond"],
                 head["abbrev_tlg_hard_cond"] - head["base_tlg_hard_cond"]))
    if len(summ["base"]) == 6:
        for arm in ("base", "abbrev"):
            head[arm + "_macro6_cond"] = float(
                np.mean([summ[arm][s]["em_cond"] for s in want]))
            head[arm + "_macro6_all"] = float(
                np.mean([summ[arm][s]["em_all"] for s in want]))
        print("macro-of-six em_cond %.4f -> %.4f"
              % (head["base_macro6_cond"], head["abbrev_macro6_cond"]))
        print("macro-of-six em_all  %.4f -> %.4f"
              % (head["base_macro6_all"], head["abbrev_macro6_all"]))

    gained, lost, kind = flips(per_entity["base"], per_entity["abbrev"])
    print("\nGAINED %d, LOST %d, net %+d"
          % (len(gained), len(lost), len(gained) - len(lost)))
    print("  direct  : +%d / -%d"
          % (sum(kind(x) == "direct" for x in gained),
             sum(kind(x) == "direct" for x in lost)))
    print("  knock-on: +%d / -%d"
          % (sum(kind(x) == "knock-on" for x in gained),
             sum(kind(x) == "knock-on" for x in lost)))
    print("  gained by string:",
          Counter(x["name"] for x in gained).most_common())
    if lost:
        print("  LOST:", [(x["source"], x["name"]) for x in lost])

    n_total = sum(len(rows) for rows in per_entity["base"].values())
    n_alias = sum(v for k, v in stats["abbrev"].items()
                  if k.endswith(":aliased"))
    print(f"\nalias firings: {n_alias} in {n_total} held-out mentions "
          f"({100 * n_alias / n_total:.2f}%)")

    with open(a.out, "w") as f:
        json.dump(dict(checkpoint=a.checkpoint, window=a.window,
                       headline=head, summary=summ, stats=stats,
                       n_mentions=n_total, n_alias=n_alias,
                       gained=gained, lost=lost), f, indent=1)
    print("wrote", a.out, "in %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
