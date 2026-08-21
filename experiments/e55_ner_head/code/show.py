"""Pooled table over one or more results.json files, mean +- sd across seeds."""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load(paths):
    rows = defaultdict(list)
    for p in paths:
        if not os.path.exists(p):
            continue
        for k, v in json.load(open(p)).items():
            arm, seed = k.rsplit("|", 1)
            rows[arm].append((int(seed), v))
    return rows


def main(paths, per_source=False):
    rows = load(paths)
    hdr = (f"{'arm':24s} {'n':>2s} {'det P':>12s} {'det R':>12s} "
           f"{'det F1':>13s} {'nestedR':>12s} {'demFP':>10s} {'preds':>6s} {'s':>5s}")
    print(hdr)
    print("-" * len(hdr))
    for arm in sorted(rows):
        vs = rows[arm]
        g = lambda f: np.array([f(v) for _, v in vs])  # noqa: E731
        P = g(lambda v: v["scores"]["pooled"]["P"])
        R = g(lambda v: v["scores"]["pooled"]["R"])
        F = g(lambda v: v["scores"]["pooled"]["F1"])
        N = g(lambda v: v["scores"]["pooled"]["R_nested"])
        D = g(lambda v: v["scores"]["pooled"]["fp_on_demonym"])
        Q = g(lambda v: v["scores"]["pooled"]["n_pred"])
        T = g(lambda v: v["info"]["seconds"])
        f = lambda a, d=2: f"{a.mean():.{d}f}+-{a.std(ddof=1) if len(a) > 1 else 0:.{d}f}"  # noqa: E731
        print(f"{arm:24s} {len(vs):2d} {f(P,1):>12s} {f(R,1):>12s} "
              f"{f(F,2):>13s} {f(N,1):>12s} {f(D,1):>10s} "
              f"{Q.mean():6.0f} {T.mean():5.0f}")
        if per_source:
            for s in ("tr", "lgl", "gwn"):
                sf = np.array([v["scores"][s]["F1"] for _, v in vs])
                sr = np.array([v["scores"][s]["R"] for _, v in vs])
                print(f"    {s:6s} F1 {sf.mean():.2f}  R {sr.mean():.2f}")


def paired(paths, base="gold_ship"):
    """Paired per-seed deltas against a reference arm.

    Runs are deterministic given (config, seed) -- verified -- so the same seed
    in two arms is a matched pair and a paired t-test is legitimate.
    """
    rows = load(paths)
    if base not in rows:
        print("no base arm", base)
        return
    b = {s: v for s, v in rows[base]}
    keys = [("F1", lambda v: v["scores"]["pooled"]["F1"]),
            ("nestedR", lambda v: v["scores"]["pooled"]["R_nested"]),
            ("demFP", lambda v: v["scores"]["pooled"]["fp_on_demonym"])]
    print(f"paired vs {base} (n = shared seeds)")
    print(f"{'arm':22s} {'n':>2s} " +
          " ".join(f"{k:>22s}" for k, _ in keys))
    for arm in sorted(rows):
        if arm == base:
            continue
        shared = sorted(s for s, _ in rows[arm] if s in b)
        if len(shared) < 2:
            continue
        av = {s: v for s, v in rows[arm]}
        cells = []
        for k, f in keys:
            d = np.array([f(av[s]) - f(b[s]) for s in shared])
            sd = d.std(ddof=1)
            t = d.mean() / (sd / np.sqrt(len(d))) if sd > 0 else float("inf")
            star = "*" if abs(t) > 2.776 and len(d) >= 5 else (
                "~" if abs(t) > 4.303 and len(d) >= 3 else "")
            cells.append(f"{d.mean():+7.2f}+-{sd:5.2f} t{t:+6.2f}{star:1s}")
        print(f"{arm:22s} {len(shared):2d} " + " ".join(cells))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    base = next((a for a in args if not a.endswith(".json")), "gold_ship")
    paths = [a for a in args if a.endswith(".json")] or \
        sorted(glob.glob(f"{HERE}/results*.json"))
    if "--paired" in sys.argv:
        paired(paths, base)
    else:
        main(paths, per_source="--per-source" in sys.argv)
