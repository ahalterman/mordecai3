"""Pool the e56 grid runs over the three corpora and print the report table.

`tools/end_to_end_eval.py` writes one summary per (corpus, variant); the
campaign's headline numbers are pooled over TR + LGL + GWN. Counts are
recovered from the rates and each corpus's own denominator, which is exact --
the rates are rounded to four decimals and no denominator exceeds 1,230.

    uv run python experiments/e56_span_head_serving/aggregate.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
E2E = os.path.join(HERE, "e2e")
SOURCES = ["tr", "lgl", "gwn"]
VARIANTS = ["serving", "head_gold", "head_all"]
RUNS = ["e29_seed42", "e29_seed101", "e54_seed42_noout", "e54_seed42_outlet"]


def pooled(run):
    with open(os.path.join(E2E, f"{run}.json")) as f:
        res = json.load(f)
    out = {}
    for var in VARIANTS:
        c = {k: 0 for k in ("n_gold", "n_pred", "em", "a161", "tp", "n_out",
                            "out_ok", "or_n", "or_em", "or_ret")}
        per_src = {}
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
            per_src[src] = 100 * s["e2e_em"]
        p = 100 * c["tp"] / c["n_pred"]
        r = 100 * c["tp"] / c["n_gold"]
        out[var] = {
            "n_gold": c["n_gold"], "n_pred": c["n_pred"],
            "e2e_em": 100 * c["em"] / c["n_gold"],
            "acc161": 100 * c["a161"] / c["n_gold"],
            "det_p": p, "det_r": r, "det_f1": 2 * p * r / (p + r),
            "n_out": c["n_out"],
            "out_prec": 100 * c["out_ok"] / c["n_out"],
            "oracle_em": 100 * c["or_em"] / c["or_n"] if c["or_n"] else None,
            "oracle_ret": 100 * c["or_ret"] / c["or_n"] if c["or_n"] else None,
            "per_src": per_src,
            "timing": res["corpora"]["lgl"]["variants"][var]["timing"],
        }
    return out


def main():
    runs = {r: pooled(r) for r in RUNS if
            os.path.exists(os.path.join(E2E, f"{r}.json"))}
    print(f"{'ranker / outlets':<26}{'span det':<11}{'e2e EM':>8}{'@161':>8}"
          f"{'det P':>8}{'det R':>8}{'det F1':>8}{'emit P':>8}{'n_pred':>8}"
          f"{'oracle':>8}")
    for run, block in runs.items():
        for var in VARIANTS:
            b = block[var]
            print(f"{run:<26}{var:<11}{b['e2e_em']:>8.2f}{b['acc161']:>8.2f}"
                  f"{b['det_p']:>8.2f}{b['det_r']:>8.2f}{b['det_f1']:>8.2f}"
                  f"{b['out_prec']:>8.2f}{b['n_pred']:>8}"
                  f"{b['oracle_em']:>8.2f}")
    print()
    print(f"{'run':<26}{'span det':<11}" +
          "".join(f"{s:>10}" for s in SOURCES))
    for run, block in runs.items():
        for var in VARIANTS:
            print(f"{run:<26}{var:<11}" +
                  "".join(f"{block[var]['per_src'][s]:>10.2f}"
                          for s in SOURCES))
    print()
    print("LGL timing (175 docs, s):")
    for run, block in runs.items():
        for var in VARIANTS:
            print(f"  {run:<26}{var:<11}{block[var]['timing']}")
    json.dump({r: {v: {k: b[k] for k in b if k != "timing"}
                   for v, b in block.items()} for r, block in runs.items()},
              open(os.path.join(HERE, "grid_pooled.json"), "w"), indent=1)


def decomp(run, var):
    """Where the gold toponyms of one cell go, pooled over the three corpora.

    The seven buckets partition the D2 gold set: `ner_miss` / `boundary_*` are
    detection losses, `retrieval_miss` / `null_answer` / `ranker_error` are
    resolution losses on a span that WAS detected exactly, and `correct` is the
    e2e exact match. The oracle-span row is the same ranker on gold spans, so
    the difference between its miss rate and the resolution losses here is what
    detection is still costing indirectly (worse context, worse candidates).
    """
    with open(os.path.join(E2E, f"{run}.json")) as f:
        res = json.load(f)
    c = {}
    n_gold = 0
    for src in SOURCES:
        s = res["corpora"][src]["variants"][var]
        n_gold += s["n_gold"]
        for k, v in s["decomp_counts"].items():
            c[k] = c.get(k, 0) + v
    o = res["corpora"]["lgl"]["variants"][var]["oracle"]
    oracle = {k: 0 for k in ("n", "em", "ret")}
    for src in SOURCES:
        o = res["corpora"][src]["variants"][var]["oracle"]
        oracle["n"] += o["n"]
        oracle["em"] += round(o["em_all_gold"] * o["n"])
        oracle["ret"] += round(o["retrieval_recall"] * o["n"])
    det_loss = sum(c.get(k, 0) for k in ("ner_miss", "boundary_ok",
                                         "boundary_wrong"))
    res_loss = sum(c.get(k, 0) for k in ("retrieval_miss", "null_answer",
                                         "ranker_error"))
    print(f"\n{run} / {var}: {n_gold} gold toponyms")
    for k in sorted(c):
        print(f"  {k:<16}{c[k]:>6}{100 * c[k] / n_gold:>8.2f}%")
    print(f"  {'-- detection':<16}{det_loss:>6}{100 * det_loss / n_gold:>8.2f}%")
    print(f"  {'-- resolution':<16}{res_loss:>6}{100 * res_loss / n_gold:>8.2f}%")
    print(f"  oracle spans: n {oracle['n']} EM "
          f"{100 * oracle['em'] / oracle['n']:.2f} "
          f"retrieval recall {100 * oracle['ret'] / oracle['n']:.2f}")
    g = {}
    for src in SOURCES:
        for k, v in (res["corpora"][src]["variants"][var].get("gap") or
                     {}).items():
            g[k] = g.get(k, 0) + v
    if g:
        # The gap to the ceiling, gold by gold: of the toponyms this same
        # ranker resolves correctly from the GOLD span, which does the pipeline
        # lose, and did it lose them by not finding the span or by resolving
        # the span it found differently?
        n = g["n"]
        lost = g.get("lost_to_detection", 0) + g.get("lost_to_resolution", 0)
        print(f"  gap: n {n} oracle_correct {g['oracle_correct']} "
              f"pipeline_correct {g['pipeline_correct']}")
        print(f"       lost_to_detection {g.get('lost_to_detection', 0)}"
              f" ({100 * g.get('lost_to_detection', 0) / n:.2f}%)"
              f"  lost_to_resolution {g.get('lost_to_resolution', 0)}"
              f" ({100 * g.get('lost_to_resolution', 0) / n:.2f}%)"
              f"  won_without_oracle {g.get('won_without_oracle', 0)}")
        if lost:
            print(f"       detection share of the gap "
                  f"{100 * g.get('lost_to_detection', 0) / lost:.1f}%")


if __name__ == "__main__":
    main()
    # Extra arguments are `<run>:<variant>` cells to decompose.
    for arg in sys.argv[1:]:
        if ":" in arg:
            decomp(*arg.split(":", 1))
