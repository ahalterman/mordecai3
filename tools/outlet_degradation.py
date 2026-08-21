"""Does the outlet-trained model still work when no outlet is supplied?

The serving story depends on this.  A deployment will have an outlet domain for
some documents and not others -- an unknown masthead, a bare text field, a
non-news source -- and if the outlet-trained checkpoint were worse than the
baseline on documents with no outlet, adoption would mean shipping two models
and routing between them.

The test: score the held-out LGL set with the **arm** checkpoint but with the
five outlet columns set to their no-outlet values, exactly as
`clear_outlet_features` writes them for GWN, Prodigy, Synth and WikiDocs, and
compare to the **baseline** checkpoint on the same entities.  Training already
showed the model 21,521 WikiDocs + 1,580 GWN + 1,668 Prodigy + 5,701 Synth
entities in exactly that state, so this is in-distribution for it, not an
extrapolation.

    uv run python tools/outlet_degradation.py
"""

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

BUILT = ("/tmp/claude-1000/-home-andy-projects-mordecai3/"
         "a19357d6-880a-4251-9cdd-97e2235f449c/scratchpad/e50")
CKPT = os.path.join(BUILT, "ckpt")


def em(es_data, checkpoint, blocks):
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
        n_live = min(len(ent["es_choices"]), len(row))
        tot += 1
        if int(np.argmax(row[:n_live])) == gi:
            right += 1
    return right / tot, tot


def main():
    from train import ALL_FEATURE_KEYS

    path = os.path.join(
        BUILT, "pickled_es",
        "es_formatted_lgl_500_all_loc_types_fuzzy_0_enriched_compact.pkl")
    with open(path, "rb") as f:
        data = pickle.load(f)
    data = [i for i in data if len(i["tensor"]) > 1]
    val = data[round(0.7 * len(data)):]

    base_em, n = em(val, os.path.join(CKPT, "baseline42.pt"),
                    "prom,name,cue,sib,geo,shape")
    arm_em, _ = em(val, os.path.join(CKPT, "arm42.pt"),
                   "prom,name,cue,sib,geo,shape,outlet")

    # Blank the outlet columns in place, the way clear_outlet_features would.
    cols = {k: ALL_FEATURE_KEYS.index(k) for k in OUTLET_KEYS}
    for ent in val:
        for k, c in cols.items():
            ent["feat_matrix"][:, c] = outlet_null_value(k)
    blank_em, _ = em(val, os.path.join(CKPT, "arm42.pt"),
                     "prom,name,cue,sib,geo,shape,outlet")

    print("held-out LGL, {} scored entities, seed 42\n".format(n))
    print("  baseline checkpoint                              {:.4f}".format(base_em))
    print("  outlet checkpoint, outlet supplied               {:.4f}  ({:+.4f})"
          .format(arm_em, arm_em - base_em))
    print("  outlet checkpoint, NO outlet supplied (nulled)   {:.4f}  ({:+.4f})"
          .format(blank_em, blank_em - base_em))
    print("\n  -> the cost of serving the outlet checkpoint on a document whose")
    print("     outlet is unknown is {:+.4f} EM against the baseline checkpoint."
          .format(blank_em - base_em))


if __name__ == "__main__":
    main()
