"""Follow-ups: generous ceilings, seed robustness, and WHY the ceiling is small."""
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = "/home/andy/projects/mordecai3"
sys.path.insert(0, ROOT)
SCR = ("/tmp/claude-1000/-home-andy-projects-mordecai3/"
       "a19357d6-880a-4251-9cdd-97e2235f449c/scratchpad")
sys.path.insert(0, SCR)
from mordecai3.candidate_features import norm, is_null_choice  # noqa: E402
from sizing import alias_targets, split_val, gold_idx  # noqa: E402

DATA = os.path.join(ROOT, "raw_data", "pickled_es")
TEMPLATE = "es_formatted_{s}_500_all_loc_types_fuzzy_0_enriched.pkl"

men = pd.read_parquet(os.path.join(SCR, "sizing.parquet"))
txt = pd.read_parquet(os.path.join(SCR, "sizing_text.parquet"))
ens = pd.read_parquet(os.path.join(
    ROOT, "experiments/campaign2/preds/e29_ens5_w100.parquet"))

m = men.merge(txt[["source", "idx", "gold_a1", "gold_fc", "gold_cc",
                   "already_sib_adm1", "text_loose", "text_guarded"]],
              on=["source", "idx"], how="inner", suffixes=("", "_t"))
m = m.merge(ens[["source", "idx", "correct"]].rename(
    columns={"correct": "correct_ens5"}), on=["source", "idx"], how="left")

t = m[(m.in_win42 == True) & (~m.gold_fc_t.str.startswith("PCL"))]  # noqa: E712
print("TLG-hard n:", len(t), dict(t.groupby("source").size()))

for lbl, col in [("seed42", "correct42"), ("ens5", "correct_ens5")]:
    base = t.groupby("source")[col].mean().mean()
    wrong = t[col] == False  # noqa: E712
    print(f"\n### {lbl}: TLG-hard macro {base:.4f}, wrong {int(wrong.sum())}")
    variants = {
        "A. gold gains sib_adm1 (mention aliasing)": t.gold_new_a1,
        "B. ANY candidate gains sib_adm1 (mention aliasing)": t.n_new_a1 > 0,
        "C. gold ADM1 named by abbrev in TEXT, not already sib": (
            t.text_loose & ~t.already_sib_adm1),
        "D. doc contains any abbrev mention (loosest)": t.has_abbrev_sib,
    }
    for name, sel in variants.items():
        flip = sel & wrong
        ceil = (t[col] | flip).groupby(t.source).mean().mean()
        print(f"  {name}: n={int(sel.sum())} wrong={int(flip.sum())} "
              f"ceiling={ceil:.4f} delta=+{ceil - base:.4f}")

# ---- WHY: is the signal already carried by the geometry block?
print("\n### geometry already carries it")
rows = []
for label, stem in [("TR", "tr"), ("LGL", "lgl"), ("GWN", "gwn")]:
    with open(os.path.join(DATA, TEMPLATE.format(s=stem)), "rb") as f:
        data = pickle.load(f)
    from collections import defaultdict
    doc_raw = defaultdict(list)
    for e in data:
        doc_raw[e["doc_key"]].append(str(e["search_name"]))
    val = split_val(data)
    idxs = set(m[(m.source == label) & m.gold_new_a1].idx)
    for i in idxs:
        e = val[i]
        gi = gold_idx(e)
        ch = e["es_choices"][gi]
        rows.append(dict(source=label, idx=i, name=str(e["search_name"]),
                         a1=ch.get("admin1_name"),
                         anchor_a1=float(ch.get("anchor_same_adm1_frac", 0)),
                         min_km=float(ch.get("log_min_km_anchor", 0)),
                         f150=float(ch.get("frac_anchors_150km", 0)),
                         sib_adm2=float(ch.get("sib_adm2", 0)),
                         sib_country=float(ch.get("sib_country", 0))))
    del data, val
g = pd.DataFrame(rows)
print("entities where gold would gain sib_adm1:", len(g))
print("  gold already has anchor_same_adm1_frac > 0:",
      int((g.anchor_a1 > 0).sum()), f"({(g.anchor_a1 > 0).mean():.0%})")
print("  gold already within 150km of some anchor (frac>0):",
      int((g.f150 > 0).sum()), f"({(g.f150 > 0).mean():.0%})")
print("  mean anchor_same_adm1_frac on gold:", round(g.anchor_a1.mean(), 3))

# how often the full ADM1 name is ALSO a mention somewhere in the doc
print("\n### redundancy: full state name also present as a mention")
red = []
for label, stem in [("TR", "tr"), ("LGL", "lgl"), ("GWN", "gwn")]:
    with open(os.path.join(DATA, TEMPLATE.format(s=stem)), "rb") as f:
        data = pickle.load(f)
    from collections import defaultdict
    doc_raw = defaultdict(list)
    for e in data:
        doc_raw[e["doc_key"]].append(str(e["search_name"]))
    val = split_val(data)
    for i, e in enumerate(val):
        own = norm(e["search_name"])
        names = {norm(r) for r in doc_raw[e["doc_key"]]} - {own}
        al = set()
        for r in doc_raw[e["doc_key"]]:
            if norm(r) == own:
                continue
            al |= alias_targets(r)
        if not al:
            continue
        red.append(dict(source=label, idx=i,
                        n_alias=len(al), n_redundant=len(al & names)))
    del data, val
r = pd.DataFrame(red)
print("held-out entities with >=1 abbrev sibling:", len(r))
print("  of their aliased state names, share already present as a full-name "
      "mention in the same doc:",
      f"{r.n_redundant.sum() / max(r.n_alias.sum(), 1):.0%}")
