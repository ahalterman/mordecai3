"""Run several label-scaling arms in one process (the wiki chunk cache is
expensive to rebuild, so arms that share a label set share it).

Config file is a json list of dicts, each with `name`, `seeds` and any
scaletrain argument override.
"""
import argparse
import json
import os
import sys
import types

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir("/home/andy/projects/mordecai3")

import scaledata as SD  # noqa: E402
import scaletrain as ST  # noqa: E402

DEFAULTS = dict(wiki=False, wiki_loss="mask", wiki_weight=1.0, wiki_frac=1.0,
                demonym_weight=1.0, silver="", epochs=20, gold_repeat=1,
                batch_tokens=1200, save_preds=False, save_head="",
                stage2_epochs=0, stage2_lr=3e-4,
                teacher="", pseudo_hi=0.9, pseudo_lo=0.05)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=f"{HERE}/results.json")
    args = ap.parse_args()
    cfgs = json.load(open(args.config))

    print("building gold chunks...")
    gc, gt = SD.gold_chunks()
    cache = {"gold_split": ST.gold_split(gc), "gold_tensors": gt,
             "wiki": {}, "gold_dem": {}}
    silver_cache = {}

    results = json.load(open(args.out)) if os.path.exists(args.out) else {}
    for cfg in cfgs:
        a = dict(DEFAULTS)
        a.update(cfg)
        name = a.pop("name")
        seeds = a.pop("seeds", [42, 101, 202])
        if a["silver"]:
            if a["silver"] not in silver_cache:
                sm = json.load(open(a["silver"]))
                silver_cache[a["silver"]] = {
                    int(k): [tuple(x) for x in v] for k, v in sm.items()}
                print(f"  silver {a['silver']}: "
                      f"{sum(len(v) for v in silver_cache[a['silver']].values())}"
                      f" spans")
            cache["silver_map"] = silver_cache[a["silver"]]
        ST.BATCH_TOKENS = a["batch_tokens"]
        ns = types.SimpleNamespace(**a)
        for seed in seeds:
            key = f"{name}|{seed}"
            if key in results:
                print("skip", key)
                continue
            print(f"== {key}  {a}")
            model, info, res, preds = ST.run(ns, seed, cache)
            p = res["pooled"]
            print(f"== {key}: P {p['P']:.2f} R {p['R']:.2f} F1 {p['F1']:.2f} "
                  f"nestedR {p['R_nested']:.1f} demFP {p['fp_on_demonym']} "
                  f"({info['seconds']:.0f}s)")
            results[key] = {"info": info, "scores": res, "args": a}
            with open(args.out, "w") as f:
                json.dump(results, f, indent=1)
            if a["save_preds"]:
                with open(f"{HERE}/preds_{name}_{seed}.json", "w") as f:
                    json.dump({s: {str(k): v for k, v in preds[s].items()}
                               for s in preds}, f)
            if a["save_head"]:
                torch.save({"state_dict": model.state_dict(),
                            "threshold": info["threshold"],
                            "max_span": SD.MAX_SPAN, "arm": name,
                            "seed": seed},
                           a["save_head"].replace("SEED", str(seed)))


if __name__ == "__main__":
    main()
