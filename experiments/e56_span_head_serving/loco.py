"""Leave-one-corpus-out generalisation for the place-span head (the N1 gate).

Every detection number in e55 and in §3 of the serving report is on held-out
*documents* of the three corpora the head trained on, so the head has seen each
corpus's span conventions. This runs the honest cross-family read that a TEST
corpus would give, using the only three families on disk: for each of
{TR-News, LGL, GeoWebNews}, train the head on the D2 gold spans of the other
two and score detection on the excluded corpus's held-out documents.

Two arms, both gold-only recipes (no Wikipedia, no silver — those were measured
n.s. in e55 and are not in the ship recipe):

  gold        the packaged "gold" recipe: 20 epochs, lr 1e-3
  gold_dem10  the same with demonym negatives weighted 10x, which is the
              plausible ship variant if demonym false positives are weighted

3 seeds per fold. The threshold is picked on the *training* families' dev
split, never on the excluded corpus — that is what makes this a gate and not a
tuning exercise.

The question is not whether it drops (it will). It is whether an out-of-family
head still beats the spaCy label-filter path on a corpus it has never seen.

    uv run python experiments/e56_span_head_serving/loco.py
    uv run python experiments/e56_span_head_serving/loco.py --show

e55's training harness (`scaledata.py`, `scaletrain.py`) and its cached token
tensors are imported read-only from a copy under the session scratchpad; the
paths are arguments so the run is reproducible from a different copy.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SCRATCH = ("/tmp/claude-1000/-home-andy-projects-mordecai3/"
           "a19357d6-880a-4251-9cdd-97e2235f449c/scratchpad/e56")
OUT = os.path.join(HERE, "loco.json")

ARMS = [("gold", 1.0), ("gold_dem10", 10.0)]
FOLDS = ["tr", "lgl", "gwn"]
SEEDS = [42, 101, 202]

# In-family references, all from this ledger:
#   packaged gold head, per corpus (parity_span_head.py)
IN_FAMILY = {"tr": 83.87, "lgl": 90.31, "gwn": 83.71}
#   the spaCy label-filter path the head has to beat (detection_grid.py --per-corpus)
SPACY_PATH = {}


def setup(scratch):
    sys.path.insert(0, os.path.join(scratch, "loco"))
    sys.path.insert(0, os.path.join(scratch, "ner"))
    os.chdir(REPO)


def run(scratch=SCRATCH, out=OUT, save_dir="", only=""):
    setup(scratch)
    import scaledata as SD
    import scaletrain as ST
    from evalcore import add, load_docs, prf, score

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = json.load(open(out)) if os.path.exists(out) else {}
    chunk_cache = {}
    for arm, dem_w in ARMS:
        if only and not only.startswith(arm):
            continue
        if dem_w not in chunk_cache:
            gc, gt = SD.gold_chunks(demonym_weight=dem_w)
            chunk_cache[dem_w] = (ST.gold_split(gc), gt)
        (tr_all, dev_all, te_all), tensors = chunk_cache[dem_w]
        for fold in FOLDS:
            if only and only != f"{arm}|{fold}":
                continue
            train = [c for c in tr_all if c["src"] != fold]
            dev = [c for c in dev_all if c["src"] != fold]
            test = [c for c in te_all if c["src"] == fold]
            docs = [d for d in load_docs(fold) if d["heldout"]]
            for seed in SEEDS:
                key = f"{arm}|{fold}|{seed}"
                if key in results:
                    print("skip", key)
                    continue
                model, info = ST.train_head([(train, 20, 1e-3, True)], dev,
                                            tensors, seed, device,
                                            log=lambda *a: None)
                preds = ST.predict_test(model, test, tensors, device,
                                        info["threshold"])
                p = prf(score(docs, preds[fold]))
                results[key] = {
                    "threshold": info["threshold"], "dev_f1": info["dev_f1"],
                    "seconds": info["seconds"],
                    "n_train_chunks": len(train), "n_dev_chunks": len(dev),
                    "scores": {k: (float(v) if isinstance(v, (int, float))
                                   else v) for k, v in p.items()}}
                print(f"{key}: P {p['P']:.2f} R {p['R']:.2f} F1 {p['F1']:.2f} "
                      f"nested {p['R_nested']:.1f} demFP {p['fp_on_demonym']} "
                      f"thr {info['threshold']:.2f} ({info['seconds']:.0f}s)")
                with open(out, "w") as f:
                    json.dump(results, f, indent=1)
                if save_dir:
                    # Runs are deterministic in (config, seed), so a head saved
                    # by a re-run is the head that produced the row above.
                    os.makedirs(save_dir, exist_ok=True)
                    torch.save({"state_dict": model.state_dict(),
                                "threshold": info["threshold"], "max_span": 8,
                                "arm": f"loco_{arm}_not_{fold}", "seed": seed},
                               os.path.join(save_dir,
                                            f"loco_{arm}_not_{fold}_{seed}.pt"))
                del model
    return results


def show(out=OUT):
    results = json.load(open(out))
    spacy_path = json.load(open(os.path.join(HERE, "detection_per_corpus.json"))) \
        if os.path.exists(os.path.join(HERE, "detection_per_corpus.json")) else {}
    print(f"{'arm':<12}{'held-out corpus':<17}{'F1 (3 seeds)':>16}"
          f"{'P':>8}{'R':>8}{'nested R':>10}{'demFP':>8}"
          f"{'spaCy F1':>10}{'in-family':>11}")
    for arm, _ in ARMS:
        for fold in FOLDS:
            rows = [results[f"{arm}|{fold}|{s}"]["scores"] for s in SEEDS
                    if f"{arm}|{fold}|{s}" in results]
            if not rows:
                continue

            def m(k):
                return float(np.mean([r[k] for r in rows]))

            sd = float(np.std([r["F1"] for r in rows], ddof=1)) \
                if len(rows) > 1 else 0.0
            sp = (spacy_path.get(fold) or {}).get("F1")
            print(f"{arm:<12}{fold:<17}{m('F1'):>9.2f} ±{sd:<5.2f}"
                  f"{m('P'):>8.2f}{m('R'):>8.2f}{m('R_nested'):>10.1f}"
                  f"{m('fp_on_demonym'):>8.1f}"
                  f"{(sp if sp else float('nan')):>10.2f}"
                  f"{IN_FAMILY[fold]:>11.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", default=SCRATCH)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--save-dir", default="")
    ap.add_argument("--only", default="", help="arm|fold, e.g. gold|gwn")
    a = ap.parse_args()
    if not a.show:
        run(a.scratch, a.out, a.save_dir, a.only)
    show(a.out)
