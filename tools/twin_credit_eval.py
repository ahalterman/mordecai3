"""Score a checkpoint on the held-out split with strict and twin-credit exact match.

Strict exact match comes from `error_utils.evaluate_results`, unchanged -- that
function defines the campaign's frozen headline metric and nothing here touches
it.  Twin credit is the secondary metric from the e12 analysis: a prediction
also counts when it is a member of the gold answer's A/P twin class, i.e. when
the model named the same real-world place at a different granularity
(`Genève` PPLA vs `Genève` ADM3, 0.5 km apart).  The class definition is
`nstrip_cc` with the cue exemption -- see tools/rewrite_labels.py, whose twin
machinery this imports:

  * same de-accented, admin-word-stripped name, within 0.15 degrees, same
    country, A/P edges, connected components;
  * no credit when the mention itself carries an admin word ("Aleppo
    Governorate"), because there the granularity is what was asked for.

`--pickle-suffix` selects which label set to score against, so the same
checkpoint can be scored on the original labels and on the R** rewrite; that
difference is the "metric got easier" component of an arm's delta.

    uv run python tools/twin_credit_eval.py --checkpoint experiments/e24_rstar/seed42.pt
    uv run python tools/twin_credit_eval.py --checkpoint ... --pickle-suffix _r2
"""
import argparse
import json
import os
import pickle
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from error_utils import evaluate_results                       # noqa: E402
from mordecai3.torch_model import TrainData, geoparse_model    # noqa: E402
from rewrite_labels import (ADMIN_CUE_RE, gold_index,          # noqa: E402
                            pickle_path, twin_classes, visible_candidates)

TRAIN_FRAC = 0.7
SOURCES = [
    ("Prodigy", ["prodigy"]),
    ("TR", ["tr"]),
    ("LGL", ["lgl"]),
    ("GWN", ["gwn"]),
    ("Synth", ["syn_cities", "syn_caps"]),
    ("WikiDocs", ["wiki_docs"]),
]
# The held-out sizes every run in this campaign reports. A mismatch means the
# split drifted and none of the numbers below would be comparable.
EXPECTED_VAL = {"Prodigy": 500, "TR": 274, "LGL": 973, "GWN": 474,
                "Synth": 300, "WikiDocs": 6456}


def split_list(data, frac=TRAIN_FRAC):
    split = round(frac * len(data))
    return data[0:split], data[split:]


def load_val(source, stems, data_dir, suffix, max_results, limit_types, fuzzy,
             with_train=False):
    """The held-out half of one source, exactly as tools/train.py splits it.

    `with_train=True` returns (held_out, train_half) instead.
    """
    def load(stem):
        fn = pickle_path(data_dir, stem, suffix, max_results, limit_types, fuzzy)
        if not os.path.exists(fn):
            sys.exit(f"missing pickle: {fn}")
        with open(fn, "rb") as f:
            return pickle.load(f)

    if source == "Synth":
        syn1, syncaps = load("syn_cities"), load("syn_caps")
        random.seed(617)
        random.shuffle(syn1)
        random.shuffle(syncaps)
        es_data = syn1[0:500] + syncaps[0:500]
    else:
        es_data = load(stems[0])
    es_data = [i for i in es_data if len(i["tensor"]) > 1]
    train, val = split_list(es_data)
    return (val, train) if with_train else val


def gold_twin_gids(entity):
    """Geonameids in the gold answer's twin class; empty when there is none.

    Cue-exempt: a mention that names the administrative unit gets no credit for
    answering with the city.
    """
    if ADMIN_CUE_RE.search(str(entity.get("search_name") or "")):
        return frozenset()
    gi = gold_index(entity)
    if gi is None:
        return frozenset()
    cands = visible_candidates(entity)
    by_idx = dict(cands)
    if gi not in by_idx:
        return frozenset()
    klass = twin_classes(cands).get(gi)
    if not klass:
        return frozenset()
    return frozenset(str(by_idx[i].get("geonameid")) for i in klass)


def twin_credit(es_data, twins, loader, model):
    """Twin-credit exact match, over the same entities evaluate_results scores.

    evaluate_results skips entities with no reachable gold and handles the
    reserved "not present" row; the filter here is the same one, so the two
    exact-match numbers share a denominator.
    """
    device = next(model.parameters()).device
    preds = []
    with torch.no_grad():
        model.eval()
        for label, country, inp in loader:
            inp = {k: v.to(device, non_blocking=True) for k, v in inp.items()}
            out = model(inp)
            if model.country_pred:
                out = out[0]
            preds.append(out.detach().cpu().numpy())
    pred_array = np.vstack(preds)

    strict = []
    credited = []
    n_twin_involved = 0
    n_twin_swaps = 0
    for ent, tw, pred in zip(es_data, twins, pred_array):
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
        strict.append(hit)
        if tw:
            n_twin_involved += 1
        in_twin = str(pred_gid) in tw
        if in_twin and not hit:
            n_twin_swaps += 1
        credited.append(bool(hit or in_twin))
    return dict(n=len(strict), strict=float(np.mean(strict)),
                twin=float(np.mean(credited)),
                twin_involved=n_twin_involved, twin_swaps=n_twin_swaps)


def build_model(checkpoint, sidecar, device):
    with open(sidecar) as f:
        cfg = json.load(f)
    model = geoparse_model(
        device=device,
        bert_size=cfg.get("bert_size", 768),
        num_feature_codes=cfg.get("num_feature_codes", 54),
        country_size=cfg.get("country_size", 24),
        code_size=cfg.get("code_size", 8),
        dropout=cfg.get("dropout", 0.2),
        mix_dim=cfg.get("mix_dim", 24),
        country_pred=cfg.get("country_pred", False),
        n_extra_features=cfg.get("n_extra_features", 0),
        return_logits=cfg.get("return_logits", False),
        mask_padding=cfg.get("mask_padding", False),
        modern_mlp=cfg.get("modern_mlp", False),
        mix_depth=cfg.get("mix_depth", 2),
        residual=cfg.get("residual", False),
        listwise=cfg.get("listwise", False),
        listwise_heads=cfg.get("listwise_heads", 4),
        aux_country=cfg.get("aux_country", False),
        aux_class=cfg.get("aux_class", False),
    )
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()
    model.to(device)
    return model, cfg


def write_twin_cache(path, data_dir, suffix, max_results, limit_types, fuzzy):
    """Precompute the gold twin class of every held-out entity, once.

    The twin class depends only on the pickles and the labels, never on the
    model, and computing it needs the *uncompacted* enriched pickles (the
    compact ones drop candidate `name`, which the name-stripping twin rule
    reads). Caching it is what lets `tools/train.py` print twin-credit exact
    match at the end of every run without loading 3.7 GB of dicts.
    """
    out = {"data_dir": data_dir, "suffix": suffix, "max_results": max_results,
           "limit_types": limit_types, "fuzzy": fuzzy, "sources": {},
           "train_pairs": []}
    pairs = set()
    for source, stems in SOURCES:
        es_data, train_half = load_val(source, stems, data_dir, suffix,
                                       max_results, limit_types, fuzzy,
                                       with_train=True)
        # The (mention, gold id) pairs the model was trained on: the novel-pair
        # guardrail needs them, and recovering them from the compacted pickles
        # inside a training run costs nothing but is not available to the
        # standalone evaluators, so they are cached here too.
        pairs.update((str(e.get("search_name")), str(e.get("correct_geonamesid")))
                     for e in train_half)
        del train_half
        if EXPECTED_VAL.get(source) not in (None, len(es_data)):
            sys.exit(f"{source}: held-out size {len(es_data)} != "
                     f"{EXPECTED_VAL[source]}; the split drifted")
        out["sources"][source] = [sorted(gold_twin_gids(e)) for e in es_data]
        n_tw = sum(1 for r in out["sources"][source] if r)
        print(f"{source}: {len(es_data)} held-out entities, {n_tw} with a twin class")
        del es_data
    out["train_pairs"] = sorted(pairs)
    print(f"{len(pairs)} distinct (mention, gold id) training pairs")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--sidecar", default="",
                    help="model config json (default: <checkpoint>.json, then "
                         "the checkpoint path with its extension replaced)")
    ap.add_argument("--data-dir", default="raw_data")
    ap.add_argument("--pickle-suffix", default="",
                    help='label set to score against: "" (frozen) or "_r2"')
    ap.add_argument("--max-results", type=int, default=0,
                    help="default: the checkpoint's max_choices")
    ap.add_argument("--limit-types", default="all_loc_types")
    ap.add_argument("--fuzzy", type=int, default=0)
    ap.add_argument("--test-batch-size", type=int, default=64)
    ap.add_argument("--json-out", default="")
    ap.add_argument("--cache-out", default="",
                    help="write the per-entity gold twin classes here and "
                         "exit; no checkpoint needed. tools/train.py reads "
                         "this cache to print twin-credit EM every run.")
    args = ap.parse_args()

    if args.cache_out:
        write_twin_cache(args.cache_out, args.data_dir,
                         f"_enriched{args.pickle_suffix}",
                         args.max_results or 500, args.limit_types, args.fuzzy)
        return
    if not args.checkpoint:
        ap.error("--checkpoint is required (or use --cache-out)")

    # `<checkpoint>.json` is the model config train.py writes; the file with the
    # extension swapped is usually the *metrics* sidecar, which is a different
    # thing, so only fall back to it.
    sidecar = args.sidecar
    if not sidecar:
        sidecar = args.checkpoint + ".json"
        if not os.path.exists(sidecar):
            sidecar = os.path.splitext(args.checkpoint)[0] + ".json"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, cfg = build_model(args.checkpoint, sidecar, device)
    max_results = args.max_results or cfg.get("max_choices", 500)
    blocks = cfg.get("feature_blocks") or None
    suffix = f"_enriched{args.pickle_suffix}" if cfg.get("enriched") else args.pickle_suffix

    print(f"checkpoint: {args.checkpoint}")
    print(f"labels:     {suffix or '(unenriched)'}")
    print(f"features:   {blocks} ({cfg.get('n_extra_features', 0)} extra columns)\n")

    results = {}
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
        frozen = evaluate_results(es_data, loader, model)
        tc = twin_credit(es_data, twins, loader, model)
        results[source] = dict(
            n=tc["n"], exact_match=float(frozen["exact_match"]),
            exact_match_recomputed=tc["strict"], twin_credit=tc["twin"],
            acc_at_161=float(frozen["acc_at_161"]),
            twin_involved=tc["twin_involved"], twin_swaps=tc["twin_swaps"])
        del es_data, dataset, loader

    print("| source | N | strict EM | twin-credit EM | delta | twin-involved | "
          "twin swaps | acc@161 |")
    print("|---|---|---|---|---|---|---|---|")
    for source, _ in SOURCES:
        r = results[source]
        print(f"| {source} | {r['n']} | {100 * r['exact_match']:.2f}% | "
              f"{100 * r['twin_credit']:.2f}% | "
              f"+{100 * (r['twin_credit'] - r['exact_match_recomputed']):.2f} | "
              f"{r['twin_involved']} | {r['twin_swaps']} | "
              f"{100 * r['acc_at_161']:.2f}% |")

    def pooled(names):
        n = sum(results[s]["n"] for s in names)
        em = sum(results[s]["exact_match_recomputed"] * results[s]["n"] for s in names) / n
        tw = sum(results[s]["twin_credit"] * results[s]["n"] for s in names) / n
        return n, em, tw

    for label, names in [("TR+LGL+GWN (pooled)", ["TR", "LGL", "GWN"]),
                         ("All held-out (pooled)", [s for s, _ in SOURCES])]:
        n, em, tw = pooled(names)
        print(f"| *{label}* | {n} | {100 * em:.2f}% | {100 * tw:.2f}% | "
              f"+{100 * (tw - em):.2f} |  |  |  |")

    macro_em = np.mean([results[s]["exact_match"] for s, _ in SOURCES])
    macro_tw = np.mean([results[s]["twin_credit"] for s, _ in SOURCES])
    macro_161 = np.mean([results[s]["acc_at_161"] for s, _ in SOURCES])
    print(f"| **macro avg (headline)** |  | **{100 * macro_em:.2f}%** | "
          f"**{100 * macro_tw:.2f}%** | +{100 * (macro_tw - macro_em):.2f} |  |  | "
          f"{100 * macro_161:.2f}% |")

    # evaluate_results and the twin-credit loop must agree on strict EM; if they
    # do not, the twin numbers are not on the frozen metric's denominator.
    drift = max(abs(results[s]["exact_match"] - results[s]["exact_match_recomputed"])
                for s, _ in SOURCES)
    print(f"\nmax |frozen EM - recomputed EM| across sources: {drift:.6f}")

    if args.json_out:
        payload = dict(checkpoint=args.checkpoint, pickle_suffix=args.pickle_suffix,
                       per_source=results, exact_match_avg=float(macro_em),
                       twin_credit_avg=float(macro_tw), acc_at_161=float(macro_161))
        with open(args.json_out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
