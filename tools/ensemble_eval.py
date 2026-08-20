"""Score a K-checkpoint ensemble on the held-out split, strict and twin-credit.

The error analysis found run-to-run variance to be as large as the remaining
feature headroom: two same-recipe seeds share only 70% of their errors, and a
2-seed probability average measured 92.1% against 91.7% expected for a single
model.  This script measures what a "train K seeds, ship the ensemble"
deployment actually buys, on the frozen metric.

Candidates are averaged in *probability* space: each model's scores are
softmaxed over the live candidate rows and the means are argmaxed.  Averaging
logits instead would let one over-confident model dominate, and the checkpoints
this campaign produces are trained with `--logits`, so their raw scales are not
comparable across seeds.

Everything about which entities are scored, and how twin credit is defined,
comes from tools/twin_credit_eval.py, so the numbers sit on the same
denominator as every other table in the campaign.

    uv run python tools/ensemble_eval.py --checkpoints experiments/e14_no_cf/seed{42,101}.pt
    uv run python tools/ensemble_eval.py --checkpoints "experiments/e14_no_cf/seed*.pt" --k 3
"""
import argparse
import glob
import itertools
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mordecai3.torch_model import TrainData                     # noqa: E402
from twin_credit_eval import (EXPECTED_VAL, SOURCES, build_model,  # noqa: E402
                              gold_twin_gids, load_val)
from rewrite_labels import gold_index                           # noqa: E402


def model_probs(model, loader, n_rows):
    """Softmax scores for every entity, (N, max_choices), padded rows included.

    A model trained with --mask-padding has already driven padded rows to -1e9,
    so they contribute nothing to the softmax; models trained without it are
    softmaxed the same way, which is what evaluate_results effectively compares.
    """
    device = next(model.parameters()).device
    out = []
    with torch.no_grad():
        model.eval()
        for label, country, inp in loader:
            inp = {k: v.to(device, non_blocking=True) for k, v in inp.items()}
            pred = model(inp)
            if model.country_pred:
                pred = pred[0]
            out.append(torch.softmax(pred.float(), dim=1).cpu().numpy())
    return np.vstack(out)


def score(es_data, twins, probs):
    """Strict and twin-credit exact match from averaged probabilities.

    The entity filter is twin_credit_eval's, which is evaluate_results': skip
    entities with no candidates, give credit for a correct "not present" call,
    and drop entities with no reachable gold.
    """
    strict, credited = [], []
    swaps = 0
    for ent, tw, pred in zip(es_data, twins, probs):
        if not ent["es_choices"]:
            continue
        n_live = min(len(ent["es_choices"]), len(pred))
        correct_position = gold_index(ent)
        predicted_position = int(np.argmax(pred[:n_live]))
        last = len(ent["es_choices"]) - 1
        if correct_position == last and predicted_position == last:
            continue
        if correct_position is None:
            continue
        pred_gid = ent["es_choices"][predicted_position]["geonameid"]
        hit = ent["correct_geonamesid"] == pred_gid
        in_twin = str(pred_gid) in tw
        if in_twin and not hit:
            swaps += 1
        strict.append(hit)
        credited.append(bool(hit or in_twin))
    return len(strict), float(np.mean(strict)), float(np.mean(credited)), swaps


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints", nargs="+", required=True,
                    help="checkpoint paths, or a glob in quotes")
    ap.add_argument("--k", type=int, nargs="*", default=None,
                    help="ensemble sizes to report (default: every size up to "
                         "the number of checkpoints given)")
    ap.add_argument("--combinations", type=int, default=0,
                    help="for each K, average over this many random subsets "
                         "(0 = every subset, which is what makes the K=1 and "
                         "K=2 rows unbiased)")
    ap.add_argument("--data-dir", default="raw_data")
    ap.add_argument("--pickle-suffix", default="")
    ap.add_argument("--limit-types", default="all_loc_types")
    ap.add_argument("--fuzzy", type=int, default=0)
    ap.add_argument("--test-batch-size", type=int, default=64)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    paths = []
    for c in args.checkpoints:
        paths.extend(sorted(glob.glob(c)) if any(ch in c for ch in "*?[") else [c])
    if not paths:
        sys.exit("no checkpoints matched")
    ks = args.k or list(range(1, len(paths) + 1))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    models, cfgs = [], []
    for p in paths:
        sidecar = p + ".json"
        if not os.path.exists(sidecar):
            sidecar = os.path.splitext(p)[0] + ".json"
        m, cfg = build_model(p, sidecar, device)
        models.append(m)
        cfgs.append(cfg)
    for c in cfgs[1:]:
        if c.get("feature_blocks") != cfgs[0].get("feature_blocks"):
            sys.exit("checkpoints disagree on feature_blocks; they would be "
                     "scored on different candidate features")
    cfg = cfgs[0]
    max_results = cfg.get("max_choices", 500)
    blocks = cfg.get("feature_blocks") or None
    suffix = f"_enriched{args.pickle_suffix}" if cfg.get("enriched") else args.pickle_suffix
    print(f"{len(paths)} checkpoints: {', '.join(os.path.basename(p) for p in paths)}")
    print(f"features: {blocks}\n")

    # Per-source probabilities for every model, computed once.
    per_source = {}
    for source, stems in SOURCES:
        es_data = load_val(source, stems, args.data_dir, suffix, max_results,
                           args.limit_types, args.fuzzy)
        if EXPECTED_VAL.get(source) not in (None, len(es_data)):
            sys.exit(f"{source}: held-out size {len(es_data)} != "
                     f"{EXPECTED_VAL[source]}; the split drifted")
        twins = [gold_twin_gids(e) for e in es_data]
        dataset = TrainData(es_data, max_choices=max_results,
                            oov_bucket_fix=cfg.get("oov_bucket_fix", False),
                            feature_blocks=blocks,
                            full_null_row=cfg.get("full_null_row", False))
        loader = DataLoader(dataset=dataset, batch_size=args.test_batch_size,
                            shuffle=False)
        probs = [model_probs(m, loader, max_results) for m in models]
        per_source[source] = (es_data, twins, probs)
        del dataset, loader

    rows = {}
    for k in ks:
        subsets = list(itertools.combinations(range(len(models)), k))
        if args.combinations and len(subsets) > args.combinations:
            rng = np.random.default_rng(617)
            pick = rng.choice(len(subsets), args.combinations, replace=False)
            subsets = [subsets[i] for i in pick]
        acc = {s: {"em": [], "tw": []} for s, _ in SOURCES}
        for sub in subsets:
            for source, _ in SOURCES:
                es_data, twins, probs = per_source[source]
                mean = np.mean([probs[i] for i in sub], axis=0)
                n, em, tw, _ = score(es_data, twins, mean)
                acc[source]["em"].append(em)
                acc[source]["tw"].append(tw)
        rows[k] = {s: (float(np.mean(acc[s]["em"])), float(np.mean(acc[s]["tw"])))
                   for s, _ in SOURCES}
        rows[k]["_n_subsets"] = len(subsets)

    names = [s for s, _ in SOURCES]
    print("| K | " + " | ".join(names) + " | macro EM | macro twin | subsets |")
    print("|" + "---|" * (len(names) + 4))
    for k in ks:
        r = rows[k]
        cells = " | ".join(f"{100 * r[s][0]:.2f}%" for s in names)
        macro_em = np.mean([r[s][0] for s in names])
        macro_tw = np.mean([r[s][1] for s in names])
        print(f"| {k} | {cells} | **{100 * macro_em:.2f}%** | "
              f"{100 * macro_tw:.2f}% | {r['_n_subsets']} |")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"checkpoints": paths,
                       "rows": {str(k): rows[k] for k in ks}}, f, indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
