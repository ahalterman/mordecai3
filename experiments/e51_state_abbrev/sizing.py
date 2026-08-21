"""Sizing pass for e51_state_abbrev (campaign-2 Phase 2).

Q: if US state (and CA-province) abbreviations appearing as sibling MENTIONS
were aliased to full ADM1 names, how many held-out entities would gain a new
sib_adm1 / sib_adm2 firing, how many of those are currently wrong, and what is
the maximum EM / TLG-hard gain?
"""
import os
import pickle
import sys
from collections import Counter, defaultdict

import pandas as pd

ROOT = "/home/andy/projects/mordecai3"
DATA = os.path.join(ROOT, "raw_data", "pickled_es")
TEMPLATE = "es_formatted_{s}_500_all_loc_types_fuzzy_0_enriched.pkl"
SOURCES = [("Prodigy", ["prodigy"]), ("TR", ["tr"]), ("LGL", ["lgl"]),
           ("GWN", ["gwn"]), ("WikiDocs", ["wiki_docs"])]
TRAIN_FRAC = 0.7

sys.path.insert(0, ROOT)
from mordecai3.candidate_features import norm, is_null_choice  # noqa: E402

STATES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut",
    "DE": "Delaware", "FL": "Florida", "GA": "Georgia", "HI": "Hawaii",
    "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
    "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine",
    "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan",
    "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri",
    "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico",
    "NY": "New York", "NC": "North Carolina", "ND": "North Dakota",
    "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania",
    "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota",
    "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont",
    "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming", "DC": "District of Columbia",
    "PR": "Puerto Rico",
}
CA_PROV = {
    "AB": "Alberta", "BC": "British Columbia", "MB": "Manitoba",
    "NB": "New Brunswick", "NL": "Newfoundland and Labrador",
    "NS": "Nova Scotia", "ON": "Ontario", "PE": "Prince Edward Island",
    "QC": "Quebec", "SK": "Saskatchewan", "NT": "Northwest Territories",
    "YT": "Yukon", "NU": "Nunavut",
}
AP = {
    "ala": "Alabama", "ariz": "Arizona", "ark": "Arkansas",
    "calif": "California", "colo": "Colorado", "conn": "Connecticut",
    "del": "Delaware", "fla": "Florida", "ga": "Georgia", "ill": "Illinois",
    "ind": "Indiana", "kan": "Kansas", "kans": "Kansas", "ky": "Kentucky",
    "la": "Louisiana", "md": "Maryland", "mass": "Massachusetts",
    "mich": "Michigan", "minn": "Minnesota", "miss": "Mississippi",
    "mo": "Missouri", "mont": "Montana", "neb": "Nebraska",
    "nebr": "Nebraska", "nev": "Nevada", "n.h": "New Hampshire",
    "n.j": "New Jersey", "n.m": "New Mexico", "n.mex": "New Mexico",
    "n.y": "New York", "n.c": "North Carolina", "n.d": "North Dakota",
    "n.dak": "North Dakota", "okla": "Oklahoma", "ore": "Oregon",
    "oreg": "Oregon", "pa": "Pennsylvania", "penn": "Pennsylvania",
    "penna": "Pennsylvania", "r.i": "Rhode Island", "s.c": "South Carolina",
    "s.d": "South Dakota", "s.dak": "South Dakota", "tenn": "Tennessee",
    "tex": "Texas", "vt": "Vermont", "va": "Virginia", "wash": "Washington",
    "w.va": "West Virginia", "wis": "Wisconsin", "wisc": "Wisconsin",
    "wyo": "Wyoming", "d.c": "District of Columbia", "p.r": "Puerto Rico",
    "alta": "Alberta", "b.c": "British Columbia", "man": "Manitoba",
    "n.b": "New Brunswick", "n.l": "Newfoundland and Labrador",
    "n.s": "Nova Scotia", "ont": "Ontario", "p.e.i": "Prince Edward Island",
    "que": "Quebec", "sask": "Saskatchewan",
}
UPPER = dict(STATES)
UPPER.update(CA_PROV)


def alias_targets(raw):
    """Lowercased full ADM1 names this raw mention string could abbreviate."""
    s = str(raw).strip()
    out = set()
    core = s.rstrip(".").replace(".", "").replace(" ", "")
    if len(core) == 2 and core.isalpha() and core.isupper() and s == core:
        # bare uppercase two-letter code, no dots: "IL"
        if core in UPPER:
            out.add(UPPER[core].lower())
    key = s.rstrip(".").lower().replace(" ", "")
    if key in AP:
        out.add(AP[key].lower())
    return out


def split_val(data):
    data = [i for i in data if len(i["tensor"]) > 1]
    return data[round(TRAIN_FRAC * len(data)):]


def gold_idx(entity):
    for i, c in enumerate(entity["correct"]):
        if c:
            return i
    return None


def main():
    pq = pd.read_parquet(
        os.path.join(ROOT, "experiments/campaign2/preds/e29_seed42_w100.parquet"))
    ens = pd.read_parquet(
        os.path.join(ROOT, "experiments/campaign2/preds/e29_ens5_w100.parquet"))

    rows = []
    abbrev_ct = Counter()
    for label, stems in SOURCES:
        with open(os.path.join(DATA, TEMPLATE.format(s=stems[0])), "rb") as f:
            data = pickle.load(f)
        doc_raw = defaultdict(list)
        for e in data:
            doc_raw[e["doc_key"]].append(str(e["search_name"]))
        val = split_val(data)
        pqs = pq[pq.source == label].set_index("idx")
        ens_s = ens[ens.source == label].set_index("idx")
        for i, e in enumerate(val):
            own = norm(e["search_name"])
            alias = set()
            fired_by = []
            for raw in doc_raw[e["doc_key"]]:
                if norm(raw) == own:
                    continue
                t = alias_targets(raw)
                if t:
                    alias |= t
                    fired_by.append(raw)
            existing_sibs = {norm(r) for r in doc_raw[e["doc_key"]]} - {own}
            new_alias = alias - existing_sibs
            gi = gold_idx(e)
            n_new_a1 = 0
            n_new_a2 = 0
            gold_new_a1 = False
            gold_fc = ""
            pred_gid = pqs.loc[i, "pred_gid"] if i in pqs.index else None
            pred_new_a1 = False
            for k, ch in enumerate(e["es_choices"]):
                if is_null_choice(ch):
                    continue
                a1 = ch.get("admin1_name")
                a2 = ch.get("admin2_name")
                hit_a1 = (norm(a1) in new_alias and a1 not in ("", "NULL")
                          and float(ch.get("sib_adm1", 0.0)) == 0.0)
                hit_a2 = (norm(a2) in new_alias and a2 not in ("", "NULL")
                          and float(ch.get("sib_adm2", 0.0)) == 0.0)
                n_new_a1 += hit_a1
                n_new_a2 += hit_a2
                if k == gi:
                    gold_fc = str(ch.get("feature_code", ""))
                    gold_new_a1 = hit_a1
                if pred_gid is not None and str(ch.get("geonameid")) == str(pred_gid):
                    pred_new_a1 = pred_new_a1 or hit_a1
            for r in fired_by:
                abbrev_ct[r] += 1
            rows.append(dict(
                source=label, idx=i, name=str(e["search_name"]),
                has_abbrev_sib=bool(alias), has_new_alias=bool(new_alias),
                n_new_a1=int(n_new_a1), n_new_a2=int(n_new_a2),
                gold_new_a1=bool(gold_new_a1), gold_fc=gold_fc,
                correct42=bool(pqs.loc[i, "correct"]) if i in pqs.index else None,
                in_win42=bool(pqs.loc[i, "gold_in_window"]) if i in pqs.index else None,
                correct_ens=bool(ens_s.loc[i, "correct"]) if i in ens_s.index else None,
                pred_new_a1=bool(pred_new_a1),
            ))
    df = pd.DataFrame(rows)
    df.to_parquet("/tmp/claude-1000/-home-andy-projects-mordecai3/"
                  "a19357d6-880a-4251-9cdd-97e2235f449c/scratchpad/sizing.parquet")

    print("=== held-out sizes ===")
    print(df.groupby("source").size())
    print("\n=== abbreviation mentions that would newly alias (top 40) ===")
    for k, v in abbrev_ct.most_common(40):
        print(f"  {k!r}: {v}")

    print("\n=== per-source counts ===")
    hdr = ("source", "n", "abbrev_sib", "new_alias", "new_a1_ent",
           "new_a1_wrong42", "gold_new_a1", "gold_new_a1_wrong42",
           "pred_new_a1_correct42")
    print(" | ".join(hdr))
    for src, g in df.groupby("source"):
        w = g[g.correct42 == False]  # noqa: E712
        na1 = g[g.n_new_a1 > 0]
        print(" | ".join(str(x) for x in (
            src, len(g), int(g.has_abbrev_sib.sum()), int(g.has_new_alias.sum()),
            len(na1), int((na1.correct42 == False).sum()),  # noqa: E712
            int(g.gold_new_a1.sum()),
            int(g[g.gold_new_a1].correct42.eq(False).sum()),
            int(g[g.pred_new_a1].correct42.eq(True).sum()),
        )))

    print("\n=== TLG-hard ceiling (TR/LGL/GWN, non-PCL gold, answerable) ===")
    t = df[df.source.isin(["TR", "LGL", "GWN"])]
    t = t[t.in_win42 == True]  # noqa: E712
    t = t[~t.gold_fc.str.startswith("PCL")]
    base = t.groupby("source").correct42.mean()
    print("baseline TLG-hard macro (seed42, w100):", round(base.mean(), 4))
    print(base)
    # ceiling: every currently-wrong entity whose GOLD newly gains sib_adm1
    flip = t.gold_new_a1 & (t.correct42 == False)  # noqa: E712
    ceil = (t.correct42 | flip).groupby(t.source).mean()
    print("\nflippable (gold gains sib_adm1 & currently wrong):",
          int(flip.sum()), "of", int((t.correct42 == False).sum()), "wrong")
    print(flip.groupby(t.source).sum())
    print("ceiling TLG-hard macro:", round(ceil.mean(), 4),
          " delta:", round(ceil.mean() - base.mean(), 4))
    # downside: currently-correct entities where a NON-gold candidate gains it
    risk = t[(t.correct42 == True) & (t.n_new_a1 > 0) & (~t.gold_new_a1)]  # noqa: E712
    print("at-risk (correct now, only non-gold candidates gain sib_adm1):",
          len(risk))
    print(risk.groupby("source").size())

    print("\n=== macro-of-six-style EM ceiling on all five sources ===")
    a = df[df.in_win42 == True]  # noqa: E712
    b = a.groupby("source").correct42.mean()
    f = a.gold_new_a1 & (a.correct42 == False)  # noqa: E712
    c = (a.correct42 | f).groupby(a.source).mean()
    for s in b.index:
        print(f"  {s}: {b[s]:.4f} -> {c[s]:.4f}  (+{c[s]-b[s]:.4f})")


if __name__ == "__main__":
    main()
