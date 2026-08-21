"""e57: gold-by-gold diff of an R1-off / R1-on pair, and the retrieval residual.

The aggregate table says "+1.44 EM". This says which golds moved, in which
direction, and -- for the ones still lost to retrieval after R1 -- why.

The retrieval residual is bucketed by cause, live from Elasticsearch:

  abbrev_other   the query is an abbreviation R1's table does not cover
  gold_row_gone  the gold geonameid is not in the index at all
  past_window    the gold IS returned by the query, but past `max_choices`
  nested_span    the query is a toponym the head found inside an ORG/FAC name
  name_mismatch  the gold row's names simply do not phrase-match the mention

    uv run python experiments/e57_r1_retrieval/flips.py e54_seed42_outlet head_gold
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

ROOT = "/home/andy/projects/mordecai3"
HERE = os.path.dirname(os.path.abspath(__file__))
E2E = os.path.join(HERE, "e2e")
for p in (ROOT, os.path.join(ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

from mordecai3 import place_aliases                       # noqa: E402
from mordecai3.elasticsearch import setup_es_client       # noqa: E402
from mordecai3.geonames import GeonamesService, _clean_search_name  # noqa: E402

SOURCES = ["tr", "lgl", "gwn"]


def outcomes(run, flag, var):
    with open(os.path.join(E2E, f"{run}_{flag}.json")) as f:
        res = json.load(f)
    out = {}
    for src in SOURCES:
        for r in res["corpora"][src]["variants"][var]["gold_outcomes"]:
            out[(src, r["doc"], r["start"], r["end"])] = r
    return out


def retrieval_rows(run, flag, var):
    with open(os.path.join(E2E, f"{run}_{flag}.json")) as f:
        res = json.load(f)
    rows = []
    for src in SOURCES:
        for e in res["corpora"][src]["variants"][var].get(
                "retrieval_examples", []):
            e = dict(e)
            e["src"] = src
            rows.append(e)
    return rows


def classify_residual(rows, svc, max_choices=100):
    """Why is each of these golds still unretrievable with R1 on?"""
    out = []
    for r in rows:
        q = str(r["query"])
        gid = str(r["gold_id"])
        rec = dict(r)
        rec["alias_fired"] = place_aliases.alias_query(q) is not None
        # Does the gold row still exist?
        entry = svc.get_entry_by_id(gid)
        if entry is None:
            rec["cause"] = "gold_row_gone"
            out.append(rec)
            continue
        rec["gold_name"] = entry.get("name")
        rec["gold_fc"] = entry.get("feature_code")
        rec["gold_cc"] = entry.get("country_code3")
        # Is it returned at all, at a deep window?
        deep = [str(h["geonameid"]) for h in svc.search_by_name(q, 1000)]
        if gid in deep:
            rec["cause"] = "past_window"
            rec["deep_rank"] = deep.index(gid)
        elif r.get("cover"):
            rec["cause"] = "nested_span"
        else:
            core = q.rstrip(".").replace(".", "").replace(" ", "")
            if (len(core) <= 5 and (q.endswith(".") or q.isupper())
                    and not place_aliases.alias_query(q)):
                rec["cause"] = "abbrev_other"
            else:
                rec["cause"] = "name_mismatch"
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", default="e54_seed42_outlet", nargs="?")
    ap.add_argument("var", default="head_gold", nargs="?")
    ap.add_argument("--no-es", action="store_true")
    a = ap.parse_args()

    off, on = outcomes(a.run, "off", a.var), outcomes(a.run, "on", a.var)
    assert set(off) == set(on), "the two runs scored different gold sets"

    moves = Counter()
    gained, lost = [], []
    for k in off:
        b, r = off[k]["outcome"], on[k]["outcome"]
        if b == r:
            continue
        moves[(b, r)] += 1
        rec = {"src": k[0], "doc": k[1], "phrase": off[k]["phrase"],
               "gold_id": off[k]["gold_id"], "from": b, "to": r}
        if r == "correct":
            gained.append(rec)
        elif b == "correct":
            lost.append(rec)
    print(f"=== {a.run} / {a.var}: R1 off -> on, gold by gold ===")
    print(f"golds scored: {len(off)}")
    print(f"GAINED {len(gained)}, LOST {len(lost)}, net {len(gained)-len(lost):+d}")
    print("transitions:")
    for (b, r), n in moves.most_common():
        print(f"   {b:>15} -> {r:<15} {n}")
    print("\ngained, by mention string:")
    for s, n in Counter(g["phrase"] for g in gained).most_common():
        print(f"   {s!r:<24} x{n}")
    if lost:
        print("\nLOST:")
        for l in lost:
            print(f"   {l['src']} doc {l['doc']} {l['phrase']!r} "
                  f"gold {l['gold_id']} -> {l['to']}")

    resid = {}
    if not a.no_es:
        svc = GeonamesService(es_client=setup_es_client(),
                              normalize_place_abbrevs=True)
        rows_off = retrieval_rows(a.run, "off", a.var)
        rows_on = retrieval_rows(a.run, "on", a.var)
        print(f"\n=== retrieval_miss: {len(rows_off)} off -> {len(rows_on)} on ===")
        fixed = Counter(r["query"] for r in rows_off) - \
            Counter(r["query"] for r in rows_on)
        print("queries that stopped missing:", dict(fixed.most_common()))
        cls = classify_residual(rows_on, svc)
        c = Counter(r["cause"] for r in cls)
        print("\nresidual retrieval misses by cause:")
        for k, n in c.most_common():
            print(f"   {k:<16} {n:>4}")
            for r in [x for x in cls if x["cause"] == k][:8]:
                print(f"        {r['src']:<4} {r['query']!r:<28} gold "
                      f"{r['gold_id']:<10} {r.get('gold_name','?')!r} "
                      f"{r.get('gold_fc','?')} "
                      f"{'cover=' + r['cover'] if r.get('cover') else ''}"
                      f"{' rank=' + str(r.get('deep_rank')) if r.get('deep_rank') is not None else ''}")
        resid = {"by_cause": dict(c), "rows": cls}

    with open(os.path.join(HERE, f"flips_{a.run}_{a.var}.json"), "w") as f:
        json.dump({"run": a.run, "variant": a.var, "n_golds": len(off),
                   "gained": gained, "lost": lost,
                   "transitions": {f"{b}->{r}": n for (b, r), n in moves.items()},
                   "residual": resid}, f, indent=1)


if __name__ == "__main__":
    main()
