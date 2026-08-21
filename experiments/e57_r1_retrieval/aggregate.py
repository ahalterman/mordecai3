"""Pool the e57 grid (R1 off vs on) over TR + LGL + GWN and print the tables.

`tools/end_to_end_eval.py` writes one summary per (corpus, variant); the
campaign's headline numbers are pooled over the three corpora. Counts are
recovered from the rates times each corpus's own denominator, which is exact:
rates carry four decimals and no denominator exceeds 1,230. Same arithmetic as
`experiments/e56_span_head_serving/aggregate.py`, so the OFF rows are directly
comparable to that report's.

    uv run python experiments/e57_r1_retrieval/aggregate.py
"""
import json
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
E2E = os.path.join(HERE, "e2e")
SOURCES = ["tr", "lgl", "gwn"]
RUNS = [("e29_seed42", ["serving"]),
        ("e29_seed101", ["serving", "head_gold"]),
        ("e54_seed42_noout", ["head_gold"]),
        ("e54_seed42_outlet", ["serving", "head_gold"])]
DECOMP = ["correct", "ner_miss", "boundary_wrong", "boundary_ok",
          "retrieval_miss", "ranker_error", "null_answer"]


def load(run, flag):
    p = os.path.join(E2E, f"{run}_{flag}.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def pooled(res, var):
    c = Counter()
    per_src = {}
    decomp = Counter()
    for src in SOURCES:
        s = res["corpora"][src]["variants"][var]
        n = s["n_gold"]
        c["n_gold"] += n
        c["n_pred"] += s["n_pred"]
        c["em"] += round(s["e2e_em"] * n)
        c["a161"] += round(s["e2e_161"] * n)
        c["tp"] += round(s["det_exact_r"] * n)
        c["n_out"] += s["n_out"]
        c["out_ok"] += round(s["out_precision"] * s["n_out"])
        o = s.get("oracle") or {}
        if o:
            c["or_n"] += o["n"]
            c["or_em"] += round(o["em_all_gold"] * o["n"])
            c["or_ret"] += round(o["retrieval_recall"] * o["n"])
        for k, v in s["decomp_counts"].items():
            decomp[k] += v
        per_src[src] = 100 * s["e2e_em"]
    p = 100 * c["tp"] / c["n_pred"]
    r = 100 * c["tp"] / c["n_gold"]
    return {"n_gold": c["n_gold"], "n_pred": c["n_pred"],
            "e2e_em": 100 * c["em"] / c["n_gold"],
            "n_em": c["em"],
            "acc161": 100 * c["a161"] / c["n_gold"],
            "det_p": p, "det_r": r, "det_f1": 2 * p * r / (p + r),
            "out_prec": 100 * c["out_ok"] / c["n_out"],
            "oracle_em": 100 * c["or_em"] / c["or_n"] if c["or_n"] else None,
            "oracle_ret": 100 * c["or_ret"] / c["or_n"] if c["or_n"] else None,
            "decomp": dict(decomp), "per_src": per_src}


def retrieval_rows(res, var):
    rows = []
    for src in SOURCES:
        for e in res["corpora"][src]["variants"][var].get(
                "retrieval_examples", []):
            e = dict(e)
            e["src"] = src
            rows.append(e)
    return rows


def main():
    cells = []
    for run, variants in RUNS:
        off, on = load(run, "off"), load(run, "on")
        if not off or not on:
            continue
        for var in variants:
            cells.append((run, var, pooled(off, var), pooled(on, var)))

    print(f"{'ranker / outlets':<22}{'span det':<10}"
          f"{'EM off':>9}{'EM on':>9}{'Δ':>8}"
          f"{'@161 off':>10}{'@161 on':>9}{'ret off':>9}{'ret on':>8}"
          f"{'orc off':>9}{'orc on':>8}")
    for run, var, a, b in cells:
        print(f"{run:<22}{var:<10}{a['e2e_em']:>9.2f}{b['e2e_em']:>9.2f}"
              f"{b['e2e_em'] - a['e2e_em']:>+8.2f}"
              f"{a['acc161']:>10.2f}{b['acc161']:>9.2f}"
              f"{a['decomp'].get('retrieval_miss', 0):>9}"
              f"{b['decomp'].get('retrieval_miss', 0):>8}"
              f"{a['oracle_em']:>9.2f}{b['oracle_em']:>8.2f}")

    print("\nper corpus e2e EM (off -> on)")
    print(f"{'ranker / outlets':<22}{'span det':<10}" +
          "".join(f"{s:>18}" for s in SOURCES))
    for run, var, a, b in cells:
        print(f"{run:<22}{var:<10}" +
              "".join(f"{a['per_src'][s]:>10.2f}->{b['per_src'][s]:<8.2f}"
                      for s in SOURCES))

    print("\nloss decomposition, counts (off -> on)")
    hdr = f"{'ranker / outlets':<22}{'span det':<10}"
    print(hdr + "".join(f"{k[:13]:>16}" for k in DECOMP))
    for run, var, a, b in cells:
        print(f"{run:<22}{var:<10}" +
              "".join(f"{a['decomp'].get(k, 0):>8}->{b['decomp'].get(k, 0):<8}"
                      for k in DECOMP))

    print("\ndetection (identical by construction; a sanity check on the flag)")
    for run, var, a, b in cells:
        print(f"  {run:<22}{var:<10} n_pred {a['n_pred']}->{b['n_pred']}  "
              f"det F1 {a['det_f1']:.2f}->{b['det_f1']:.2f}  "
              f"emitted-loc P {a['out_prec']:.2f}->{b['out_prec']:.2f}")


if __name__ == "__main__":
    main()
