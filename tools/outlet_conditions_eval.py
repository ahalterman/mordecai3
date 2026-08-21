"""Score every e50/e53 checkpoint under the three e53 eval conditions.

The e53 question is not "is the outlet feature good" -- e50 settled that -- but
"does the model still work when the outlet is missing". So each checkpoint is
scored twice on the same held-out entities:

(a) **outlet present** -- the features as the pickles carry them.
(b) **outlet withheld** -- the five outlet columns overwritten with the exact
    null encoding a source with no outlet metadata gets. This is the serving
    condition where the caller has no domain for the document.

The `e29` baseline has no outlet block at all, so it has a single number, and
that number is the bar condition (b) has to clear.

One scorer for all arms and both conditions, so the numbers are on one
denominator: an entity counts if it has a reachable gold, and the prediction is
the argmax over its live candidate rows.

    uv run python tools/outlet_conditions_eval.py
"""

import json
import os
import pickle
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mordecai3.outlet_features import OUTLET_KEYS, outlet_null_value  # noqa: E402
from mordecai3.torch_model import TrainData  # noqa: E402
from rewrite_labels import gold_index  # noqa: E402
from twin_credit_eval import build_model  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRATCH = ("/tmp/claude-1000/-home-andy-projects-mordecai3/"
           "a19357d6-880a-4251-9cdd-97e2235f449c/scratchpad/e50")
CKPT = os.path.join(SCRATCH, "ckpt")
SEEDS = [42, 101, 202, 617, 1848]
BASE_BLOCKS = "prom,name,cue,sib,geo,shape"
OUT_BLOCKS = "prom,name,cue,sib,geo,shape,outlet"
# The e50/e53 default: four arms, all in one scratch checkpoint directory named
# `<arm><seed>.pt`. `--arm name:blocks:pattern` overrides it, which is how e54
# points the same scorer at mainline checkpoints living beside their metrics.
ARMS = [("baseline", BASE_BLOCKS), ("arm", OUT_BLOCKS),
        ("d50", OUT_BLOCKS), ("d30", OUT_BLOCKS)]
T_CRIT = 2.776
DEFAULT_DATA_DIR = os.path.join(SCRATCH, "pickled_es")


def held_out(source, data_dir=None):
    path = os.path.join(
        data_dir or DEFAULT_DATA_DIR,
        "es_formatted_{}_500_all_loc_types_fuzzy_0_enriched_compact.pkl".format(source))
    with open(path, "rb") as f:
        data = pickle.load(f)
    data = [i for i in data if len(i["tensor"]) > 1]
    return data[round(0.7 * len(data)):]


def blank_outlet(es_data):
    """Overwrite the outlet columns with the no-outlet encoding, in place."""
    from train import ALL_FEATURE_KEYS
    for key in OUTLET_KEYS:
        col = ALL_FEATURE_KEYS.index(key)
        val = outlet_null_value(key)
        for ent in es_data:
            ent["feat_matrix"][:, col] = val
    return es_data


def exact_match(es_data, checkpoint, blocks):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, _ = build_model(checkpoint, checkpoint + ".json", device)
    ds = TrainData(es_data, max_choices=500, oov_bucket_fix=True,
                   feature_blocks=[b for b in blocks.split(",")])
    loader = DataLoader(dataset=ds, batch_size=64, shuffle=False)
    out = []
    with torch.no_grad():
        model.eval()
        for _l, _c, inp in loader:
            inp = {k: v.to(device) for k, v in inp.items()}
            pred = model(inp)
            if model.country_pred:
                pred = pred[0]
            out.append(pred.float().cpu().numpy())
    scores = np.vstack(out)
    right = tot = 0
    for ent, row in zip(es_data, scores):
        gi = gold_index(ent)
        if gi is None or not ent["es_choices"]:
            continue
        tot += 1
        n_live = min(len(ent["es_choices"]), len(row))
        if int(np.argmax(row[:n_live])) == gi:
            right += 1
    del model
    return right / tot, tot


def paired(base, arm):
    d = [a - b for a, b in zip(arm, base)]
    n = len(d)
    mean = sum(d) / n
    var = sum((x - mean) ** 2 for x in d) / (n - 1)
    se = (var / n) ** 0.5
    t = mean / se if se > 0 else float("inf") if mean else 0.0
    return mean, se, t


def main(source="lgl", data_dir=None, tag="", arms=None, out_path=None,
         baseline_name="baseline"):
    present = held_out(source, data_dir)
    withheld = blank_outlet(held_out(source, data_dir))
    n_ent = None
    results = {}
    arms = arms or [(name, blocks, os.path.join(CKPT, name + "{seed}.pt"))
                    for name, blocks in ARMS]

    for arm, blocks, pattern in arms:
        for seed in SEEDS:
            ck = pattern.format(seed=seed)
            if not os.path.exists(ck):
                continue
            p, n_ent = exact_match(present, ck, blocks)
            results.setdefault(arm, {}).setdefault("present", []).append(p)
            if arm == baseline_name:
                # no outlet block -> the two conditions are the same model input
                results[arm].setdefault("withheld", []).append(p)
            else:
                w, _ = exact_match(withheld, ck, blocks)
                results[arm].setdefault("withheld", []).append(w)

    print("\n=== {} held-out, {} scored entities {}===".format(
        source.upper(), n_ent, tag))
    print("{:<10} {:>10} {:>10}   {:>26}".format(
        "arm", "present", "withheld", "withheld - baseline"))
    base_w = results.get(baseline_name, {}).get("withheld")
    for arm, _, _ in arms:
        r = results.get(arm)
        if not r or len(r["present"]) < len(SEEDS):
            continue
        pm = sum(r["present"]) / len(r["present"])
        wm = sum(r["withheld"]) / len(r["withheld"])
        if base_w and arm != baseline_name:
            m, se, t = paired(base_w, r["withheld"])
            tail = "{:+.4f} +/- {:.4f}  t={:6.2f} {}".format(
                m, se, t, "*" if abs(t) > T_CRIT else " ")
        else:
            tail = "(reference)"
        print("{:<10} {:>10.4f} {:>10.4f}   {}".format(arm, pm, wm, tail))

    print("\nper-seed")
    hdr = "{:<8}".format("seed")
    for arm, _, _ in arms:
        if arm in results and len(results[arm]["present"]) == len(SEEDS):
            hdr += " {:>18}".format(arm)
    print(hdr + "     (present / withheld)")
    for i, s in enumerate(SEEDS):
        line = "{:<8}".format(s)
        for arm, _, _ in arms:
            r = results.get(arm)
            if r and len(r["present"]) == len(SEEDS):
                line += "  {:.4f}/{:.4f}".format(r["present"][i], r["withheld"][i])
        print(line)

    out = out_path or os.path.join(
        REPO, "experiments", "e53_outlet_dropout",
        "conditions_{}{}.json".format(source, tag.strip() and "_v2"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print("\nwrote {}".format(out))
    return results


def _parse_arm(spec):
    """`name:blocks:pattern` -> tuple. `pattern` takes a `{seed}` placeholder."""
    name, _, rest = spec.partition(":")
    blocks, _, pattern = rest.partition(":")
    if not name or not blocks or not pattern:
        raise SystemExit("--arm wants name:blocks:pattern, got {!r}".format(spec))
    return (name, blocks, pattern if os.path.isabs(pattern)
            else os.path.join(REPO, pattern))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="lgl")
    ap.add_argument("--data-dir", default=None,
                    help="directory of *_enriched_compact.pkl (default: the e50 scratch build)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=None, help="where to write the results json")
    ap.add_argument("--arm", action="append", default=None, metavar="NAME:BLOCKS:PATTERN",
                    help="repeatable; PATTERN takes {seed}, e.g. "
                         "'e54:prom,name,cue,sib,geo,shape,outlet:"
                         "experiments/e54_outlet_ship/seed{seed}.pt'")
    ap.add_argument("--baseline", default="baseline",
                    help="which arm is the no-outlet reference")
    a = ap.parse_args()
    main(a.source, a.data_dir, a.tag,
         arms=[_parse_arm(s) for s in a.arm] if a.arm else None,
         out_path=a.out, baseline_name=a.baseline)
