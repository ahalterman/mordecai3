"""The abbreviation mention ITSELF: how well does e29 resolve "Ky." / "W.Va."?"""
import os
import pickle
import sys

import pandas as pd

ROOT = "/home/andy/projects/mordecai3"
sys.path.insert(0, ROOT)
SCR = ("/tmp/claude-1000/-home-andy-projects-mordecai3/"
       "a19357d6-880a-4251-9cdd-97e2235f449c/scratchpad")
sys.path.insert(0, SCR)
from mordecai3.candidate_features import norm, is_null_choice  # noqa: E402
from sizing import alias_targets, split_val, gold_idx  # noqa: E402

DATA = os.path.join(ROOT, "raw_data", "pickled_es")
T = "es_formatted_{s}_500_all_loc_types_fuzzy_0_enriched.pkl"
pq = pd.read_parquet(os.path.join(ROOT, "experiments/campaign2/preds/e29_seed42_w100.parquet"))
ens = pd.read_parquet(os.path.join(ROOT, "experiments/campaign2/preds/e29_ens5_w100.parquet"))

rows = []
for label, stem in [("TR", "tr"), ("LGL", "lgl"), ("GWN", "gwn")]:
    with open(os.path.join(DATA, T.format(s=stem)), "rb") as f:
        d = pickle.load(f)
    val = split_val(d)
    p = pq[pq.source == label].set_index("idx")
    e5 = ens[ens.source == label].set_index("idx")
    for i, ent in enumerate(val):
        name = str(ent["search_name"])
        tgt = alias_targets(name)
        gi = gold_idx(ent)
        gold_fc = gold_name = ""
        gold_alias_match = False
        if gi is not None and gi < len(ent["es_choices"]):
            ch = ent["es_choices"][gi]
            if not is_null_choice(ch):
                gold_fc = str(ch.get("feature_code", ""))
                gold_name = norm(ch.get("name"))
                gold_alias_match = gold_name in tgt
        r = dict(source=label, idx=i, name=name, is_abbrev=bool(tgt),
                 gold_fc=gold_fc, gold_name=gold_name,
                 gold_is_alias_target=gold_alias_match)
        if i in p.index:
            r.update(correct42=bool(p.loc[i, "correct"]),
                     in_win=bool(p.loc[i, "gold_in_window"]),
                     retriev=bool(p.loc[i, "gold_retrievable"]),
                     correct_ens=bool(e5.loc[i, "correct"]),
                     pred_gid=str(p.loc[i, "pred_gid"]),
                     gold_gid=str(p.loc[i, "gold_gid"]),
                     err_km=float(p.loc[i, "err_km"]),
                     n_choices=int(p.loc[i, "n_choices"]))
        rows.append(r)
    del d, val
df = pd.DataFrame(rows)
df.to_parquet(os.path.join(SCR, "sizing_selfabbrev.parquet"))

a = df[df.is_abbrev]
print("abbreviation mentions in held-out TR/LGL/GWN:", len(a))
print(a.groupby("source").size().to_dict())
print("gold retrievable:", int(a.retriev.sum()), " in window(100):", int(a.in_win.sum()))
print("gold row IS the aliased full-name place:", int(a.gold_is_alias_target.sum()))
print("gold feature codes:", a.gold_fc.value_counts().to_dict())
print("seed42 EM:", round(a.correct42.mean(), 4), " ens5:", round(a.correct_ens.mean(), 4))
print("median err_km on the wrong ones:",
      round(a[~a.correct42].err_km.median(), 1))
print("\nsample of errors:")
print(a[~a.correct42][["source", "name", "gold_name", "gold_fc", "err_km",
                       "n_choices", "in_win"]].head(25).to_string())

# TLG-hard ceiling if every abbreviation mention resolved correctly
t = df[(df.in_win == True) & (~df.gold_fc.str.startswith("PCL"))]  # noqa: E712
base = t.groupby("source").correct42.mean()
print("\nTLG-hard macro baseline (seed42,w100):", round(base.mean(), 4), base.to_dict())
for col in ("correct42", "correct_ens"):
    b = t.groupby("source")[col].mean().mean()
    c = (t[col] | t.is_abbrev).groupby(t.source).mean().mean()
    print(f"  {col}: {b:.4f} -> {c:.4f}  (+{c - b:.4f}) if all abbrev mentions correct")
    print("    abbrev in TLG-hard slice:", int(t.is_abbrev.sum()),
          " wrong:", int((t.is_abbrev & (t[col] == False)).sum()))  # noqa: E712
