"""Paired 5-seed aggregation for e54_outlet_ship against its e29 reference.

Both arms are read from `<dir>/seed*.metrics2.json`, the standard metric suite
Phase 0 made every training run print, so the two sides are computed by one
implementation over one convention.  The reference is
`experiments/e54_outlet_ship/e29_ref/`, which is the e29 recipe rerun in *this*
tree on the outlet-carrying pickles -- its `seed*.json` are md5-identical to the
frozen `experiments/e29_swa_ep15/seed*.json`, so it is the frozen baseline, just
scored by the current metric code.

Criterion: five paired seeds, t(4) > 2.776.
"""

import argparse
import json
import math
import os

SEEDS = [42, 101, 202, 617, 1848]


def load(directory, seeds=SEEDS):
    out = {}
    for seed in seeds:
        path = os.path.join(directory, f"seed{seed}.metrics2.json")
        with open(path) as f:
            out[seed] = json.load(f)
    return out


def pull(metrics, key):
    """One scalar out of a metrics2 dict. `Source:field` reads per_source."""
    if ":" in key:
        source, field = key.split(":", 1)
        return metrics["per_source"][source][field]
    return metrics[key]


def paired(base, arm, key, seeds=SEEDS):
    b = [pull(base[s], key) for s in seeds]
    a = [pull(arm[s], key) for s in seeds]
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    mean = sum(d) / n
    var = sum((x - mean) ** 2 for x in d) / (n - 1) if n > 1 else 0.0
    sd = math.sqrt(var)
    se = sd / math.sqrt(n) if n > 1 else 0.0
    t = mean / se if se else float("inf") if mean else 0.0
    return {"base": sum(b) / n, "arm": sum(a) / n, "delta": mean, "sd": sd,
            "t": t, "sig": abs(t) > 2.776, "per_seed_base": b, "per_seed_arm": a}


ROWS = [
    ("TLG-hard (primary)", "tlg_hard"),
    ("LGL non-country EM", "LGL:em_noncountry"),
    ("LGL EM", "LGL:em_conditioned"),
    ("LGL novel-pair EM", "LGL:novel_pair_em"),
    ("LGL twin-credit", "LGL:twin_credit"),
    ("novel-pair EM, all sources", "novel_pair_em"),
    ("twin-credit macro (no Synth)", "twin_credit_macro"),
    ("macro EM, 5 sources (Synth excluded)", "em_conditioned_macro"),
    ("macro EM, 6 sources (continuity)", "em_conditioned_macro_legacy6"),
]

GUARDRAILS = ["Prodigy", "TR", "GWN", "Synth", "WikiDocs"]
TLG_PARTS = [("LGL", "LGL:em_noncountry"), ("TR", "TR:em_noncountry"),
             ("GWN", "GWN:em_noncountry")]


def table(base, arm, rows):
    lines = ["| metric | e29 | e54 | paired Δ | t | |",
             "|---|---|---|---|---|---|"]
    for label, key in rows:
        r = paired(base, arm, key)
        lines.append(
            f"| {label} | {r['base']:.4f} | {r['arm']:.4f} | "
            f"{r['delta']:+.4f} ± {r['sd']:.4f} | {r['t']:.2f} | "
            f"{'*' if r['sig'] else 'n.s.'} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", default="experiments/e54_outlet_ship")
    ap.add_argument("--base", default="experiments/e54_outlet_ship/e29_ref")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    base = load(args.base)
    arm = load(args.arm)

    print("## Headline (5 paired seeds, criterion t(4) > 2.776)\n")
    print(table(base, arm, ROWS))
    print("\n## TLG-hard's three ingredients\n")
    print(table(base, arm, TLG_PARTS))
    print("\n## Guardrails (per-source EM)\n")
    print(table(base, arm, [(s, f"{s}:em_conditioned") for s in GUARDRAILS]))

    print("\n## Per seed\n")
    print("| seed | TLG e29 | TLG e54 | Δ | LGL-nc e29 | LGL-nc e54 | Δ |")
    print("|---|---|---|---|---|---|---|")
    for s in SEEDS:
        tb, ta = base[s]["tlg_hard"], arm[s]["tlg_hard"]
        lb = base[s]["per_source"]["LGL"]["em_noncountry"]
        la = arm[s]["per_source"]["LGL"]["em_noncountry"]
        print(f"| {s} | {tb:.4f} | {ta:.4f} | {ta - tb:+.4f} | "
              f"{lb:.4f} | {la:.4f} | {la - lb:+.4f} |")

    if args.json_out:
        blob = {label: paired(base, arm, key)
                for label, key in ROWS + TLG_PARTS
                + [(s, f"{s}:em_conditioned") for s in GUARDRAILS]}
        with open(args.json_out, "w") as f:
            json.dump(blob, f, indent=1)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
