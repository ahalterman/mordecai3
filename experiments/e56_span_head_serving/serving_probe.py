"""Abstention and latency through the real `Geoparser`, not the harness.

Two questions the e2e harness cannot answer, because it decodes predictions
itself and never builds a `Geoparser`:

`calib`  Does the span head move the abstention story? The head hands the
         ranker mentions the label filter never produced -- nested toponyms,
         and whatever else it finds -- so `p_no_match`, the reserved row's
         probability, is being asked about a different population. This is a
         distribution check, not a recalibration.

`lat`    What does a document cost end to end with the head on, against the
         default path and against the `nested_gazetteer_pass` the head
         replaces? Wall clock of `geoparse_batch`, which is spaCy + extraction
         + Elasticsearch + ranker, i.e. what a caller actually pays.

    uv run python experiments/e56_span_head_serving/serving_probe.py calib
    uv run python experiments/e56_span_head_serving/serving_probe.py lat
    uv run python experiments/e56_span_head_serving/serving_probe.py lat --cpu
"""
import json
import os
import statistics
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
os.chdir(REPO)

from mordecai3 import Geoparser  # noqa: E402
from tools.end_to_end_eval import heldout_doc_indices, read_corpus  # noqa: E402

BASE = "prom,name,cue,sib,geo,shape"
E29 = "mordecai3/assets/mordecai_2026-08-20_seed101.pt"
E54 = "mordecai3/assets/mordecai_2026-08-20_e54_seed42.pt"
OPTS = dict(feature_blocks=None, oov_bucket_fix=None)   # read the sidecar


def heldout(source):
    articles = read_corpus(source)
    keep, _ = heldout_doc_indices(source, articles)
    return [articles[i] for i in sorted(keep)]


def build(model_path, span_detector, device=None, **kw):
    return Geoparser(model_path=model_path, span_detector=span_detector,
                     device=device, **OPTS, **kw)


# ------------------------------------------------------------------ calib

def calib():
    docs = {s: heldout(s) for s in ("tr", "lgl", "gwn")}
    rows = {}
    for ranker, name in ((E29, "e29_seed101"), (E54, "e54_seed42")):
        for det in (None, "gold", "all"):
            geo = build(ranker, det)
            supply = geo.uses_outlet
            ps, absts = [], 0
            for src, arts in docs.items():
                outlets = ([a.get("domain") for a in arts]
                           if supply and src in ("lgl", "tr") else None)
                res = geo.geoparse_batch([a["text"] for a in arts],
                                         outlets=outlets)
                for r in res:
                    for e in r["geolocated_ents"]:
                        ps.append(float(e["p_no_match"]))
                        absts += bool(e.get("no_match"))
            a = np.asarray(ps)
            key = f"{name}/{det or 'none'}"
            rows[key] = {
                "n_mentions": len(a),
                "abstained": absts,
                "abstain_rate": round(100 * absts / len(a), 2),
                "p_no_match_mean": round(float(a.mean()), 4),
                "p_no_match_median": round(float(np.median(a)), 4),
                "p_gt_0.5": round(100 * float((a > 0.5).mean()), 2),
                "p_gt_0.1": round(100 * float((a > 0.1).mean()), 2),
                "p_lt_0.01": round(100 * float((a < 0.01).mean()), 2),
                "deciles": [round(float(x), 4)
                            for x in np.percentile(a, range(10, 100, 10))],
            }
            print(key, json.dumps(rows[key]))
            del geo
    with open(os.path.join(HERE, "calibration_shift.json"), "w") as f:
        json.dump(rows, f, indent=1)


# -------------------------------------------------------------------- lat

def lat(cpu=False, n_docs=50, reps=3):
    arts = heldout("lgl")[:n_docs]
    texts = [a["text"] for a in arts]
    device = "cpu" if cpu else None
    configs = [("default (spaCy filter)", dict(span_detector=None)),
               ("nested_gazetteer_pass", dict(span_detector=None,
                                              nested_gazetteer_pass=True)),
               ("span head: gold", dict(span_detector="gold")),
               ("span head: all", dict(span_detector="all"))]
    print(f"LGL held-out, {len(texts)} documents, {reps} reps, "
          f"{'CPU' if cpu else 'GPU'}; median of reps")
    print(f"  {'configuration':<24}{'ms/doc':>10}{'docs/s':>9}{'mentions':>10}")
    out = {}
    for label, kw in configs:
        geo = build(E29, kw.pop("span_detector"), device=device, **kw)
        geo.geoparse_batch(texts[:3])          # warm up
        times, n_ents = [], 0
        for _ in range(reps):
            geo.geonames.clear_cache()
            t0 = time.perf_counter()
            res = geo.geoparse_batch(texts)
            times.append(time.perf_counter() - t0)
            n_ents = sum(len(r["geolocated_ents"]) for r in res)
        med = statistics.median(times)
        out[label] = {"ms_per_doc": round(1000 * med / len(texts), 2),
                      "docs_per_s": round(len(texts) / med, 2),
                      "mentions": n_ents}
        print(f"  {label:<24}{out[label]['ms_per_doc']:>10.2f}"
              f"{out[label]['docs_per_s']:>9.2f}{n_ents:>10}")
        del geo
    fn = os.path.join(HERE, f"latency_{'cpu' if cpu else 'gpu'}.json")
    with open(fn, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "calib"
    if cmd == "calib":
        calib()
    else:
        lat(cpu="--cpu" in sys.argv)
