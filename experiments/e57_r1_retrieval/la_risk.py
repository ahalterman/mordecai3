"""e57 work item 4: price the "LA" risk on held-out data.

R1's bare-code guard expands a two-letter code only when the whole mention is
that code, in capitals, with no dots. That still leaves capitalised codes whose
common English or newswire meaning is not a state: **LA** (Los Angeles), **IN**,
**OR**, **OK**, **NO**, **SO**, **DE**, **ME**, **AS**, **BE**... The e52 report
could not price this ("no held-out mention is `LA`"). This script prices it
three ways, from the strongest evidence to the weakest.

1. **What the serving pipeline actually queries.** Wraps
   `GeonamesService.build_name_search` and records every `search_name` that
   reaches it during a real end-to-end run, for both span sources. Every
   recorded string is classified: fired (and correctly?), or blocked by which
   guard. This is the only frame where a misfire could actually cost a number.

2. **What the corpora annotate.** Every gold toponym phrase in the held-out
   documents that is a bare two-letter code, with the gold id -- so a firing
   can be scored right or wrong against the annotation.

3. **What the text contains.** A regex sweep of the held-out document text for
   standalone capitalised two-letter tokens in the alias table, with context,
   split into the "…, IL" dateline shape (where the state reading is right) and
   everything else. This is the upper bound on exposure: it counts strings the
   tagger does not currently hand to the gazetteer at all.

    uv run python experiments/e57_r1_retrieval/la_risk.py
"""
import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

ROOT = "/home/andy/projects/mordecai3"
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (ROOT, os.path.join(ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

import end_to_end_eval as e2e  # noqa: E402
from mordecai3 import place_aliases  # noqa: E402
from mordecai3.geonames import GeonamesService  # noqa: E402

# Codes whose ordinary English / newswire reading is usually NOT the state.
# `LA` is the one e52 flagged; the rest are the English words and the common
# non-state abbreviations that share a USPS code.
AMBIGUOUS = {
    "LA": "Los Angeles / the article 'La'",
    "IN": "the preposition 'in'",
    "OR": "the conjunction 'or'",
    "OK": "'OK'",
    "ME": "the pronoun 'me'",
    "DE": "'de' in names",
    "AL": "the name 'Al'",
    "PA": "'pa' / Pennsylvania",
    "MO": "'Mo'",
    "AS": "the conjunction 'as'",
    "BC": "'BC' the era",
    "ID": "'ID' the document",
    "MS": "'MS' the honorific / manuscript",
    "MD": "'MD' the physician",
    "OH": "the interjection 'oh'",
    "HI": "the greeting 'hi'",
    "NE": "'ne'",
    "SC": "'SC'",
    "VA": "'VA' the department",
    "AK": "'AK'",
    "AR": "'AR'",
    "CA": "'ca.' circa",
    "NM": "'NM'",
    "PE": "'PE'",
    "NT": "'NT'",
    "NB": "'NB' nota bene",
}


def classify(mention):
    """Why did / didn't R1 fire on this mention string?"""
    s = str(mention).strip()
    q = place_aliases.alias_query(s)
    if q:
        core = s.rstrip(".").replace(".", "").replace(" ", "")
        if len(core) == 2 and s == core and core.isupper():
            return ("fired_bare", core)
        return ("fired_ap", s)
    core = s.rstrip(".").replace(".", "").replace(" ", "")
    if s.endswith(".") and s.rstrip(".").lower().replace(" ", "") in place_aliases.AP:
        return ("would_fire_ap", s)          # unreachable; the guard is `endswith`
    if core.upper() in place_aliases.BARE_CODES and len(core) == 2:
        if not s.isupper():
            return ("blocked_case", s)       # "Wa", "wa", "La"
        return ("blocked_shape", s)          # "W.A.", "WA County", "in WA"
    key = s.rstrip(".").lower().replace(" ", "")
    if key in place_aliases.AP and not s.endswith("."):
        return ("blocked_period", s)         # "Miss", "Man", "Ore", "Ind", "La"
    return ("no_match", s)


# --------------------------------------------------------------------- (1)

def serving_frame(model_path, feature_blocks, outlet_sources, variants,
                  sources, out_path):
    """Record every query the serving path sends, for each span source."""
    seen = defaultdict(Counter)
    current = {"var": "?"}
    orig = GeonamesService.build_name_search

    def wrapped(self, search_name, *a, **kw):
        seen[current["var"]][str(search_name)] += 1
        return orig(self, search_name, *a, **kw)

    GeonamesService.build_name_search = wrapped
    # One evaluate() call per variant so the recorder can attribute strings.
    try:
        for var in variants:
            current["var"] = var
            e2e.evaluate(model_path=model_path, sources=sources,
                         variants=var, feature_blocks=feature_blocks,
                         outlet_sources=outlet_sources, oracle=False,
                         normalize_place_abbrevs=True, out="")
    finally:
        GeonamesService.build_name_search = orig

    report = {}
    for var, counts in seen.items():
        buckets = defaultdict(Counter)
        for m, n in counts.items():
            kind, key = classify(m)
            buckets[kind][m] += n
        report[var] = {
            "n_distinct_queries": len(counts),
            "n_queries": sum(counts.values()),
            "buckets": {k: dict(v.most_common()) for k, v in buckets.items()},
        }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=1)
    return report


# --------------------------------------------------------------------- (2,3)

TOKEN = re.compile(r"(?<![A-Za-z0-9.'-])([A-Z]{2})(?![A-Za-z0-9.'-])")


def corpus_scan(sources):
    gold_hits = Counter()
    gold_rows = []
    text_hits = Counter()
    text_rows = []
    n_docs = n_gold = 0
    for src in sources:
        articles = e2e.read_corpus(src)
        heldout, _ = e2e.heldout_doc_indices(src, articles)
        for i in sorted(heldout):
            art = articles[i]
            n_docs += 1
            for t in art["toponyms"]:
                n_gold += 1
                ph = (t["phrase"] or "").strip()
                if len(ph) == 2 and ph.isupper() and ph.isalpha() \
                        and ph in place_aliases.BARE_CODES:
                    gold_hits[ph] += 1
                    gold_rows.append({"src": src, "doc": art["doc_idx"],
                                      "phrase": ph,
                                      "gold_id": t.get("geonameid"),
                                      "context": art["text"][
                                          max(0, t["start"] - 50):
                                          t["end"] + 50].replace("\n", " ")})
            for m in TOKEN.finditer(art["text"]):
                code = m.group(1)
                if code not in place_aliases.BARE_CODES:
                    continue
                lo, hi = m.span(1)
                before = art["text"][max(0, lo - 30):lo].replace("\n", " ")
                after = art["text"][hi:hi + 20].replace("\n", " ")
                # "Springfield, IL" -- the dateline shape, where the state
                # reading is the right one.
                dateline = bool(re.search(r"[A-Za-zÀ-ɏ]\s*,\s*$",
                                          before))
                text_hits[(code, dateline)] += 1
                if not dateline:
                    text_rows.append({"src": src, "doc": art["doc_idx"],
                                      "code": code,
                                      "context": (before + "[[" + code + "]]"
                                                  + after)})
    return dict(n_docs=n_docs, n_gold_rows=n_gold,
                gold_bare_codes=dict(gold_hits.most_common()),
                gold_rows=gold_rows,
                text_dateline={c: n for (c, d), n in
                               sorted(text_hits.items()) if d},
                text_bare={c: n for (c, d), n in
                           sorted(text_hits.items()) if not d},
                text_rows=text_rows[:200])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=os.path.join(
        ROOT, "mordecai3/assets/mordecai_2026-08-20_e54_seed42.pt"))
    ap.add_argument("--feature-blocks",
                    default="prom,name,cue,sib,geo,shape,outlet")
    ap.add_argument("--outlet-sources", default="lgl,tr")
    ap.add_argument("--variants", default="serving,head_gold")
    ap.add_argument("--sources", default="tr,lgl,gwn")
    ap.add_argument("--skip-serving", action="store_true")
    a = ap.parse_args()

    srcs = [s.strip() for s in a.sources.split(",") if s.strip()]
    scan = corpus_scan(srcs)
    print(f"held-out: {scan['n_docs']} documents, {scan['n_gold_rows']} gold "
          f"toponym rows")
    print("gold phrases that are a bare alias code:",
          scan["gold_bare_codes"] or "(none)")
    for r in scan["gold_rows"][:40]:
        print(f"   {r['src']} {r['phrase']} -> {r['gold_id']}  ...{r['context']}...")
    print("\nraw-text standalone capitalised codes, ',CODE' dateline shape:",
          scan["text_dateline"])
    print("raw-text standalone capitalised codes, NOT a dateline:",
          scan["text_bare"])
    amb = {c: n for c, n in scan["text_bare"].items() if c in AMBIGUOUS}
    print("  ...of which ambiguous by prior:", amb)
    for r in scan["text_rows"][:60]:
        if r["code"] in AMBIGUOUS:
            print(f"   {r['src']} {r['code']}: {r['context']}")

    with open(os.path.join(HERE, "la_risk_corpus.json"), "w") as f:
        json.dump(scan, f, indent=1)

    if not a.skip_serving:
        rep = serving_frame(a.model_path, a.feature_blocks, a.outlet_sources,
                            [v.strip() for v in a.variants.split(",")],
                            a.sources,
                            os.path.join(HERE, "la_risk_serving.json"))
        print("\n=== queries the serving path actually sent ===")
        for var, b in rep.items():
            print(f"\n-- {var}: {b['n_queries']} queries, "
                  f"{b['n_distinct_queries']} distinct")
            for kind in ("fired_bare", "fired_ap", "blocked_case",
                         "blocked_shape", "blocked_period"):
                d = b["buckets"].get(kind)
                if d:
                    print(f"   {kind:<15} n={sum(d.values()):<4} {d}")


if __name__ == "__main__":
    main()
