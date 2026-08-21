"""Pooled and per-source summary of the pilot arms, mean +/- SE over seeds."""
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
KEYS = ["P", "R", "F1", "R_ov", "F1_ov", "R_nested", "R_flat"]


def main(path=None, base=None):
    res = json.load(open(path or f"{HERE}/pilot_results.json"))
    base = json.load(open(base or f"{HERE}/baseline_detect.json"))
    rows = []
    for name, r in base.items():
        p = r["pooled"]
        rows.append((name, 1, {k: (p[k], 0.0) for k in KEYS},
                     p["n_pred"], p["fp_on_demonym"], p["fp_on_unlinked"], r))
    by_arm = defaultdict(list)
    for key, v in res.items():
        if key == "meta":
            continue
        by_arm[key.split("|")[0]].append(v)
    for arm, vs in by_arm.items():
        agg = {}
        for k in KEYS:
            a = np.array([v["scores"]["pooled"][k] for v in vs])
            agg[k] = (a.mean(), a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0)
        npred = np.mean([v["scores"]["pooled"]["n_pred"] for v in vs])
        fpd = np.mean([v["scores"]["pooled"]["fp_on_demonym"] for v in vs])
        fpu = np.mean([v["scores"]["pooled"]["fp_on_unlinked"] for v in vs])
        rows.append((arm, len(vs), agg, npred, fpd, fpu,
                     {s: {k: np.mean([v["scores"][s][k] for v in vs]) for k in KEYS}
                      for s in ["tr", "lgl", "gwn"]}))

    print(f"{'arm':16s} {'n':>2s} {'P':>12s} {'R':>12s} {'F1':>12s} "
          f"{'R_nested':>12s} {'R_flat':>7s} {'npred':>6s} {'fp_dem':>6s} {'fp_unl':>6s}")
    for name, n, agg, npred, fpd, fpu, _ in rows:
        def f(k):
            m, s = agg[k]
            return f"{m:6.1f}+-{s:4.1f}" if s else f"{m:6.1f}      "
        print(f"{name:16s} {n:2d} {f('P')} {f('R')} {f('F1')} {f('R_nested')} "
              f"{agg['R_flat'][0]:7.1f} {npred:6.0f} {fpd:6.0f} {fpu:6.0f}")

    print("\nper source F1 (exact):")
    for name, n, agg, npred, fpd, fpu, per in rows:
        if isinstance(per, dict) and "tr" in per:
            print(f"{name:16s} " + "  ".join(
                f"{s}: P {per[s]['P']:.1f} R {per[s]['R']:.1f} F1 {per[s]['F1']:.1f}"
                for s in ["tr", "lgl", "gwn"]))

    print("\nper-seed detail:")
    for key in sorted(k for k in res if k != "meta"):
        p = res[key]["scores"]["pooled"]
        i = res[key]["info"]
        print(f"  {key:18s} P {p['P']:5.1f} R {p['R']:5.1f} F1 {p['F1']:5.1f} "
              f"R_nest {p['R_nested']:5.1f} thr {i['threshold']:.2f} "
              f"devF1 {i['dev_f1']:5.1f} {i['seconds']:.0f}s")
    print("\nmeta:", res.get("meta"))


if __name__ == "__main__":
    main(*sys.argv[1:])
