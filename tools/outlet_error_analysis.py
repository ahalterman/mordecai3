"""What the outlet feature could fix on LGL, and what it did fix.

Two questions, both answered against the same seed-42 pair of checkpoints:

*The ceiling.*  Cross-tab every held-out LGL error of the **baseline** against
the outlet's home admin1, reproducing the table in
`experiments/campaign2/encoder_scoping_report.md` §5b.  An error is *fixable* by
this feature when the gold sits in the outlet's home admin1 and the prediction
does not; it *would hurt* in the mirror case.

*The realisation.*  Diff the baseline's predictions against the **arm**'s and
report how many of those fixable errors actually flipped to correct, how many
new errors the arm introduced, and where they sit relative to the home admin1.

Everything here reads gold labels.  It is post-hoc analysis of a frozen mapping,
run after `tools/outlet_home_table.py` was written and after the pickles were
built -- no number produced here can flow back into the feature.

    uv run python tools/outlet_error_analysis.py
"""

import json
import os
import sys
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mordecai3.torch_model import TrainData  # noqa: E402
from outlet_align import entity_domains  # noqa: E402
from rewrite_labels import gold_index  # noqa: E402
from twin_credit_eval import build_model  # noqa: E402

BUILT = ("/tmp/claude-1000/-home-andy-projects-mordecai3/"
         "a19357d6-880a-4251-9cdd-97e2235f449c/scratchpad/e50")
CKPT = os.path.join(BUILT, "ckpt")
BASE_BLOCKS = "prom,name,cue,sib,geo,shape"
ARM_BLOCKS = "prom,name,cue,sib,geo,shape,outlet"


def predictions(es_data, checkpoint, blocks):
    """argmax candidate index per entity for one checkpoint."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, _cfg = build_model(checkpoint, checkpoint + ".json", device)
    ds = TrainData(es_data, max_choices=500, oov_bucket_fix=True,
                   feature_blocks=[b for b in blocks.split(",")])
    loader = DataLoader(dataset=ds, batch_size=64, shuffle=False)
    out = []
    with torch.no_grad():
        model.eval()
        for _label, _country, inp in loader:
            inp = {k: v.to(device) for k, v in inp.items()}
            pred = model(inp)
            if model.country_pred:
                pred = pred[0]
            out.append(pred.float().cpu().numpy())
    scores = np.vstack(out)
    picks = []
    for ent, row in zip(es_data, scores):
        n_live = min(len(ent["es_choices"]), len(row))
        picks.append(int(np.argmax(row[:n_live])) if n_live else None)
    return picks


def home_of(choice, home):
    """(in home admin1, in home country) for one candidate."""
    if not home:
        return (False, False)
    cc = str(choice.get("country_code3") or "")
    a1 = str(choice.get("admin1_code") or "").strip()
    ha1 = home.get("admin1")
    if isinstance(ha1, list):
        ha1 = tuple(ha1)
    in_a1 = bool(ha1 and a1 and (cc, a1) == tuple(ha1))
    in_cc = bool(home.get("country_code3") and cc == home["country_code3"])
    return in_a1, in_cc


def main():
    import pickle
    from train import split_list

    path = os.path.join(
        BUILT, "pickled_es",
        "es_formatted_lgl_500_all_loc_types_fuzzy_0_enriched_compact.pkl")
    with open(path, "rb") as f:
        data = pickle.load(f)
    data = [i for i in data if len(i["tensor"]) > 1]
    domains_all = entity_domains(data, "lgl", "raw_data")
    split = round(0.7 * len(data))
    val = data[split:]
    val_domains = {n: domains_all.get(split + n) for n in range(len(val))}
    with open(os.path.join(BUILT, "outlet_homes.json")) as f:
        homes = json.load(f)

    print("held-out LGL entities: {}".format(len(val)))

    base = predictions(val, os.path.join(CKPT, "baseline42.pt"), BASE_BLOCKS)
    arm = predictions(val, os.path.join(CKPT, "arm42.pt"), ARM_BLOCKS)

    tab = Counter()
    fixable_idx = []
    hurtable_idx = []
    n_scored = 0
    base_wrong = []
    for n, ent in enumerate(val):
        gi = gold_index(ent)
        if gi is None or base[n] is None:
            continue
        n_scored += 1
        if base[n] == gi:
            continue
        base_wrong.append(n)
        home = homes.get(val_domains.get(n))
        gold_a1, _ = home_of(ent["es_choices"][gi], home)
        pred_a1, _ = home_of(ent["es_choices"][base[n]], home)
        if gold_a1 and not pred_a1:
            tab["gold in home adm1, prediction elsewhere (FIXABLE)"] += 1
            fixable_idx.append(n)
        elif pred_a1 and not gold_a1:
            tab["prediction in home adm1, gold elsewhere (would hurt)"] += 1
            hurtable_idx.append(n)
        elif gold_a1 and pred_a1:
            tab["both in home adm1 (feature inert)"] += 1
        else:
            tab["neither in home adm1"] += 1

    print("\n--- ceiling: baseline seed42 held-out LGL errors ---")
    print("scored entities {}, errors {}".format(n_scored, len(base_wrong)))
    for k, v in tab.most_common():
        print("  {:<52} {:>4}  {:5.1%}".format(k, v, v / max(len(base_wrong), 1)))

    fixed = sum(1 for n in fixable_idx if arm[n] == gold_index(val[n]))
    hurt_kept = sum(1 for n in hurtable_idx if arm[n] == gold_index(val[n]))
    print("\n--- realisation: arm seed42 vs baseline seed42 ---")
    b_correct = {n for n in range(len(val))
                 if gold_index(val[n]) is not None and base[n] == gold_index(val[n])}
    a_correct = {n for n in range(len(val))
                 if gold_index(val[n]) is not None and arm[n] == gold_index(val[n])}
    print("  baseline correct {}  arm correct {}  net {:+d}".format(
        len(b_correct), len(a_correct), len(a_correct) - len(b_correct)))
    print("  arm fixed {} of the {} fixable errors ({:.1%})".format(
        fixed, len(fixable_idx), fixed / max(len(fixable_idx), 1)))
    print("  arm also recovered {} of the {} 'would hurt' errors".format(
        hurt_kept, len(hurtable_idx)))
    newly_broken = sorted(b_correct - a_correct)
    newly_fixed = sorted(a_correct - b_correct)
    print("  newly fixed {}, newly broken {}".format(
        len(newly_fixed), len(newly_broken)))

    nb_home = Counter()
    for n in newly_broken:
        home = homes.get(val_domains.get(n))
        gi = gold_index(val[n])
        gold_a1, _ = home_of(val[n]["es_choices"][gi], home)
        pred_a1, _ = home_of(val[n]["es_choices"][arm[n]], home)
        nb_home[("gold in home adm1" if gold_a1 else "gold elsewhere",
                 "pred in home adm1" if pred_a1 else "pred elsewhere")] += 1
    print("  newly-broken breakdown:")
    for k, v in nb_home.most_common():
        print("    {} / {}: {}".format(k[0], k[1], v))

    print("\n  examples of newly fixed:")
    for n in newly_fixed[:8]:
        gi = gold_index(val[n])
        print("    {!r} ({}) base->{} arm->{} gold {}".format(
            val[n]["search_name"], val_domains.get(n),
            val[n]["es_choices"][base[n]]["geonameid"],
            val[n]["es_choices"][arm[n]]["geonameid"],
            val[n]["es_choices"][gi]["geonameid"]))


if __name__ == "__main__":
    main()
