"""e52 step 1: the repeated-error census on held-out TR/LGL/GWN.

Weighted the way the campaign now scores: TLG-hard relevance, i.e. non-country
golds only (PCLI/PCL/PCLD/PCLS/PCLF/PCLIX/TERR excluded).  For every repeated
mention string we pull the gold row and the predicted row out of Elasticsearch
and try to name a ROOT CAUSE:

  dup_row       two gazetteer rows for the same real place (<1 km, same
                country, same stripped name) -- either answer names it
  historical    the gold or the prediction is a defunct *H / PPLQ row
  retrieval     the gold is not in the candidate list at all
  altname       the gold IS the right place but the mention string is not one
                of its names (abbreviations: "D.C.", "W.Va.")
  granularity   A/P (or admin-level) twin, 1-30 km, same name
  wrong_place   genuinely different place (>30 km, not a twin)

`dup_row`, `historical`, `retrieval` and `altname` are gazetteer/serving
problems.  `granularity` is a convention dispute and `wrong_place` is a real
ranker failure; neither is fixable by hygiene, and they are reported so the
census is honest about how much of the residual hygiene can actually reach.

    python census.py --preds <parquet> --out census.json
"""
import argparse
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/home/andy/projects/mordecai3"
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import es_util  # noqa: E402

PCL_CODES = {"PCLI", "PCL", "PCLD", "PCLS", "PCLF", "PCLIX", "TERR"}
HIST_CODES = re.compile(r"^(ADM[1-5]H|PPLH|PCLH|PPLQ|ADMDH|RGNH|LCTY)$")
STRIP = re.compile(r"[^a-z0-9]+")


def haversine(a, b):
    lat1, lon1 = a
    lat2, lon2 = b
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(h)))


def coords(src):
    lat, lon = src["coordinates"].split(",")
    return float(lat), float(lon)


def key(name):
    return STRIP.sub("", str(name).lower())


def classify(gold, pred, err_km, retrievable, mention=None, in_index=True):
    """Root cause for one (gold, prediction) pair."""
    if not in_index:
        # The gold geonameid is not in the gazetteer at all -- a stale label
        # pointing at a GeoNames row that has since been deleted or merged.
        return "gold_row_gone"
    if not retrievable:
        import hygiene
        if mention is not None and hygiene.alias_query(mention):
            return "retr_abbrev"
        return "retr_other"
    if gold is None or pred is None:
        return "unknown"
    gfc = gold.get("feature_code", "")
    pfc = pred.get("feature_code", "")
    same_country = gold.get("country_code3") == pred.get("country_code3")
    same_name = key(gold.get("name")) == key(pred.get("name"))
    if HIST_CODES.match(gfc) or HIST_CODES.match(pfc):
        return "historical"
    if err_km is not None and err_km < 1.0 and same_country and same_name:
        return "dup_row"
    if err_km is not None and err_km < 1.0 and same_country:
        return "dup_row"
    if (err_km is not None and err_km <= 30.0 and same_country
            and (same_name or {gold.get("feature_class"),
                              pred.get("feature_class")} == {"A", "P"})):
        return "granularity"
    return "wrong_place"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default=os.path.join(
        ROOT, "experiments/campaign2/preds/e29_seed42_w100.parquet"))
    ap.add_argument("--out", default=os.path.join(HERE, "census.json"))
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--sources", default="TR,LGL,GWN")
    a = ap.parse_args()
    SRCS = a.sources.split(",")

    df = pd.read_parquet(a.preds)
    tlg = df[df.source.isin(SRCS)].copy()

    gids = list(tlg.gold_gid.dropna().astype(str)) + \
        list(tlg.pred_gid.dropna().astype(str))
    src = es_util.mget(gids)

    def fc(g):
        s = src.get(str(g))
        return s.get("feature_code", "") if s else ""

    tlg["gold_fc"] = [fc(g) for g in tlg.gold_gid]
    tlg["pred_fc"] = [fc(g) for g in tlg.pred_gid]
    tlg["is_country_gold"] = tlg.gold_fc.isin(PCL_CODES)

    hard = tlg[~tlg.is_country_gold].copy()
    # TLG-hard EM, for reference: the reported metric conditions on a
    # retrievable gold, so report both.
    scored = hard[hard.gold_retrievable]
    print("TLG entities %d, non-country %d, retrievable %d"
          % (len(tlg), len(hard), len(scored)))
    for s in SRCS:
        sub = scored[scored.source == s]
        print("  %-4s n=%4d EM=%.4f" % (s, len(sub), sub.correct.mean()))
    print("  TLG-hard macro EM = %.4f"
          % (sum(scored[scored.source == s].correct.mean()
                 for s in SRCS) / len(SRCS)))

    errs = hard[(~hard.correct)].copy()
    print("non-country TLG errors (incl. unretrievable): %d" % len(errs))

    causes = []
    for _, r in errs.iterrows():
        g = src.get(str(r.gold_gid))
        p = src.get(str(r.pred_gid))
        km = r.err_km if pd.notna(r.err_km) else None
        causes.append(classify(g, p, km, bool(r.gold_retrievable),
                               mention=str(r["name"]),
                               in_index=str(r.gold_gid) in src))
    errs["cause"] = causes
    print("\nroot-cause mix over all %d non-country TLG errors:" % len(errs))
    for c, n in Counter(causes).most_common():
        print("   %-12s %4d  %5.1f%%" % (c, n, 100 * n / len(errs)))

    # ---- repeated mention strings
    by_name = defaultdict(list)
    for _, r in errs.iterrows():
        by_name[str(r["name"])].append(r)
    rows = []
    for name, group in by_name.items():
        cs = Counter(g["cause"] for g in group)
        pairs = Counter((str(g.gold_gid), str(g.pred_gid)) for g in group)
        (gg, pp), npair = pairs.most_common(1)[0]
        gs, ps = src.get(gg), src.get(pp)
        rows.append(dict(
            mention=name, n_err=len(group),
            sources=dict(Counter(g["source"] for g in group)),
            cause=cs.most_common(1)[0][0], causes=dict(cs),
            gold_gid=gg, pred_gid=pp, n_this_pair=npair,
            gold_name=gs.get("name") if gs else None,
            gold_fc=gs.get("feature_code") if gs else None,
            gold_cc=gs.get("country_code3") if gs else None,
            gold_pop=int(gs.get("population", 0)) if gs else None,
            pred_name=ps.get("name") if ps else None,
            pred_fc=ps.get("feature_code") if ps else None,
            pred_cc=ps.get("country_code3") if ps else None,
            pred_pop=int(ps.get("population", 0)) if ps else None,
            median_km=float(pd.Series([g.err_km for g in group]).median()),
            n_unretrievable=int(sum(not g.gold_retrievable for g in group)),
        ))
    rows.sort(key=lambda r: (-r["n_err"], r["mention"]))
    print("\ntop %d repeated error strings (of %d distinct):"
          % (a.top, len(rows)))
    print("%-22s %3s %-12s %-30s %-30s %8s"
          % ("mention", "n", "cause", "gold", "pred", "km"))
    for r in rows[:a.top]:
        print("%-22s %3d %-12s %-30s %-30s %8.1f" % (
            r["mention"][:22], r["n_err"], r["cause"],
            ("%s %s %s" % (r["gold_name"], r["gold_fc"], r["gold_cc"]))[:30],
            ("%s %s %s" % (r["pred_name"], r["pred_fc"], r["pred_cc"]))[:30],
            r["median_km"] if r["median_km"] == r["median_km"] else -1))

    head = sum(r["n_err"] for r in rows[:a.top])
    print("\ntop %d strings = %d/%d = %.1f%% of the non-country TLG error mass"
          % (a.top, head, len(errs), 100 * head / len(errs)))

    with open(a.out, "w") as f:
        json.dump(dict(rows=rows,
                       n_errors=len(errs),
                       cause_mix=dict(Counter(causes))), f, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
