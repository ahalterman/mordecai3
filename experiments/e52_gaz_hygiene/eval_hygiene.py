"""e52: measure the gazetteer-hygiene transform against a frozen checkpoint.

No retraining.  The checkpoint is e29_swa_ep15 (ship = seed101); the only thing
that changes between arms is the candidate list the frozen ranker sees.

Honest-measurement notes, because this arm is unusually easy to fake:

  * The campaign metric (`error_utils.evaluate_results`) CONDITIONS on the gold
    being retrievable, so a rule that makes a previously-unretrievable gold
    retrievable ADDS hard entities to the denominator and can lower the
    reported EM while strictly helping.  Every table here therefore carries
    both `em_cond` (campaign convention, moving denominator) and `em_all`
    (fixed denominator = every held-out mention, an unretrievable gold scored
    wrong).  `em_all` is the number to read.
  * The model was trained on features computed from UN-hygienic candidate sets,
    so any rule that changes the set changes feature distributions the model
    never saw (`is_max_pop_exact_match`, `log_n_exact_matches`, the within-set
    distance normalisation).  That skew is part of the measurement, not an
    excuse; collateral damage is reported per source.
  * Dropping a candidate also changes the DOCUMENT features of every other
    mention in the same document (anchors, `sib_*`, `adm1_count`), so the
    document features of any touched document are recomputed and the knock-on
    is reported separately from the direct effect.

    python eval_hygiene.py --checkpoint .../seed101.pt --rules alias,demote_h
"""
import argparse
import copy
import json
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/home/andy/projects/mordecai3"
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import es_util  # noqa: E402
import hygiene  # noqa: E402
import rebuild  # noqa: E402
from error_utils import evaluate_results  # noqa: E402
from mordecai3.candidate_features import add_document_features  # noqa: E402
from mordecai3.geoparse import _add_cross_entity_counts  # noqa: E402
from mordecai3.torch_model import ProductionData  # noqa: E402
from twin_credit_eval import (EXPECTED_VAL, SOURCES, build_model,  # noqa: E402
                              gold_twin_gids, load_val)

PCL_CODES = {"PCLI", "PCL", "PCLD", "PCLS", "PCLF", "PCLIX", "TERR"}
TLG = ["TR", "LGL", "GWN"]


#
#   ---------------------------------------------------------------- planning
#

def plan(es_data, rules, dedupe_keep, alias_cache):
    """What each rule would do, without touching anything yet.

    Returns (drops, alias_hits, touched) where `drops` maps entity index ->
    set of candidate positions (within the REAL rows), `alias_hits` maps entity
    index -> list of extra ES `_source` rows to prepend, and `touched` is the
    set of entity indices whose candidate list changes.
    """
    drops, alias_hits = {}, {}
    for i, e in enumerate(es_data):
        d = hygiene.hygiene_drops(e, rules, dedupe_keep)
        if d:
            drops[i] = d
        if "abbrev" in rules:
            q = hygiene.alias_query(e.get("search_name", ""))
            if q:
                hits = alias_cache.get(q)
                if hits is None:
                    hits = es_util.phrase_search(q, size=100)
                    alias_cache[q] = hits
                if hits:
                    alias_hits[i] = hits
    touched = set(drops) | set(alias_hits)
    return drops, alias_hits, touched


#
#   ------------------------------------------------------------------ apply
#

def apply_rules(es_data, drops, alias_hits, es_src, alias_prepend_max):
    """Rebuild every touched entity, then recompute its document's features.

    Returns the set of document keys whose features were recomputed.
    """
    by_doc = defaultdict(list)
    for i, e in enumerate(es_data):
        by_doc[e.get("doc_key")].append(i)

    for i in sorted(set(drops) | set(alias_hits)):
        e = es_data[i]
        name = e.get("search_name", "")
        real = rebuild.real_choices(e)
        keep = [c for k, c in enumerate(real) if k not in drops.get(i, ())]
        structs = list(keep)
        sources = []
        missing = False
        for c in keep:
            s = es_src.get(str(c["geonameid"]))
            if s is None:
                missing = True
                break
            sources.append(rebuild.source_from_hit(s))
        if missing:
            # A candidate whose row has left the index since the pickles were
            # built: leave the entity completely alone rather than guess.
            continue
        hits = alias_hits.get(i, [])[:alias_prepend_max]
        if hits:
            # Mention-normalisation semantics: the expanded query REPLACES the
            # abbreviation's own query, so its hits take the head of the list
            # and any duplicate is removed from the tail.  (Naively prepending
            # instead pushes the original list down by len(hits); at a serving
            # window of 100 that evicted the golds the abbreviation query had
            # already found -- "Ky." had Kentucky at rank 6 and lost it.)
            new_ids = {str(h["geonameid"]) for h in hits}
            tail_s, tail_src = [], []
            for st, sr in zip(structs, sources):
                if str(st["geonameid"]) not in new_ids:
                    tail_s.append(st)
                    tail_src.append(sr)
            head_s = [rebuild.struct_from_hit(h) for h in hits]
            head_src = [rebuild.source_from_hit(h) for h in hits]
            structs = head_s + tail_s
            sources = head_src + tail_src
        if not structs:
            continue
        rebuilt = rebuild.rebuild_choices(name, structs, sources)
        rebuild.finish(e, rebuilt)

    touched_docs = {es_data[i].get("doc_key")
                    for i in set(drops) | set(alias_hits)}
    for dk in touched_docs:
        doc = [es_data[i] for i in by_doc[dk]]
        _add_cross_entity_counts(doc)
        add_document_features(doc)
    return touched_docs, by_doc


#
#   ------------------------------------------------------------------ score
#

def score(es_data, model, cfg, window, batch_size=64):
    # ProductionData, not TrainData: at a serving window smaller than the
    # pickle's 500 candidates, TrainData.create_labels indexes the label vector
    # with an out-of-window gold position and crashes.  `evaluate_results`
    # itself only needs the score array, and it handles an out-of-window gold
    # correctly (the gold position stays >= window, so nothing can select it
    # and the entity is scored wrong) -- which is what serving does too.
    ds = ProductionData(es_data, max_choices=window,
                        oov_bucket_fix=cfg.get("oov_bucket_fix", False),
                        feature_blocks=cfg.get("feature_blocks") or None,
                        full_null_row=cfg.get("full_null_row", False))
    loader = DataLoader(dataset=ds, batch_size=batch_size, shuffle=False)
    device = next(model.parameters()).device
    preds = []
    with torch.no_grad():
        model.eval()
        for inp in loader:
            inp = {k: v.to(device, non_blocking=True) for k, v in inp.items()}
            out = model(inp)
            if model.country_pred:
                out = out[0]
            preds.append(out.detach().cpu().numpy())
    pred_array = np.vstack(preds)
    res = evaluate_results(es_data, None, model, pred_array=pred_array)
    # per-entity outcome, on a FIXED denominator
    rows = []
    for e in es_data:
        ch = e["es_choices"]
        if not ch:
            rows.append(dict(name=e.get("search_name"), pred=None,
                             gold=str(e.get("correct_geonamesid")),
                             retrievable=False, correct=False, scored=False))
            continue
        scores = [c.get("score", -1e30) for c in ch]
        p = int(np.argmax(scores))
        gold = str(e.get("correct_geonamesid"))
        try:
            gi = int(np.where(e["correct"])[0][0])
        except Exception:
            gi = None
        pred_gid = str(ch[p].get("geonameid"))
        rows.append(dict(name=e.get("search_name"), pred=pred_gid, gold=gold,
                         retrievable=gi is not None and gi != len(ch) - 1,
                         correct=pred_gid == gold,
                         scored=gi is not None,
                         # `n_choices` is how diff_arms separates a DIRECT flip
                         # (this mention's own candidate list was edited) from a
                         # knock-on (a neighbour was edited and the document
                         # features moved).
                         n_choices=len(ch),
                         doc_key=e.get("doc_key")))
    return res, rows


def summarise(rows, gold_fc):
    n = len(rows)
    cond = [r for r in rows if r["scored"]]
    hard = [r for r in rows if gold_fc.get(r["gold"], "") not in PCL_CODES]
    hard_cond = [r for r in hard if r["scored"]]
    f = lambda xs: float(np.mean([x["correct"] for x in xs])) if xs else float("nan")
    return dict(n=n, n_cond=len(cond), em_cond=f(cond), em_all=f(rows),
                n_hard=len(hard), n_hard_cond=len(hard_cond),
                em_hard_cond=f(hard_cond), em_hard_all=f(hard),
                n_unretrievable=sum(1 for r in rows if not r["retrievable"]))


#
#   ------------------------------------------------------------------- main
#

def run_arm(model, cfg, window, rules, dedupe_keep, alias_prepend_max,
            sources, verify, alias_cache, es_cache):
    out = {}
    per_entity = {}
    stats = Counter()
    for source, stems in sources:
        es_data = load_val(source, stems, os.path.join(ROOT, "raw_data"),
                           "_enriched" if cfg.get("enriched") else "",
                           cfg.get("max_choices", 500), "all_loc_types", 0)
        if EXPECTED_VAL.get(source) not in (None, len(es_data)):
            sys.exit(f"{source}: held-out size drifted")
        es_data = copy.deepcopy(es_data)
        if rules:
            drops, alias_hits, touched = plan(es_data, rules, dedupe_keep,
                                              alias_cache)
            gids = []
            for i in touched:
                gids += [str(c["geonameid"])
                         for c in rebuild.real_choices(es_data[i])]
            if verify:
                # the parity check needs rows for untouched entities too
                for i in [j for j in range(len(es_data)) if j not in touched][:25]:
                    gids += [str(c["geonameid"])
                             for c in rebuild.real_choices(es_data[i])]
            need = [g for g in set(gids) if g not in es_cache]
            for k, v in es_util.mget(need).items():
                es_cache[k] = v
            for g in need:
                es_cache.setdefault(g, None)
            src = {g: es_cache[g] for g in set(gids) if es_cache.get(g)}
            if verify:
                _verify(es_data, touched, src)
            tdocs, _ = apply_rules(es_data, drops, alias_hits, src,
                                   alias_prepend_max)
            stats[source + ":touched"] = len(touched)
            stats[source + ":touched_docs"] = len(tdocs)
            stats[source + ":dropped"] = sum(len(v) for v in drops.values())
            stats[source + ":aliased"] = len(alias_hits)
        res, rows = score(es_data, model, cfg, window)
        gold_fc = {}
        out[source] = dict(raw=res)
        per_entity[source] = rows
        del es_data
    return out, per_entity, dict(stats)


def _verify(es_data, touched, src):
    """Rebuild a sample of untouched entities with no rules and diff."""
    keys = rebuild.DIST_KEYS + ["exact_name_match", "exact_altname_match",
                                "is_max_pop_exact_match", "log_population",
                                "is_max_pop", "log_n_exact_matches"]
    worst = {}
    n = 0
    for i, e in enumerate(es_data):
        if i in touched or n >= 25:
            continue
        real = rebuild.real_choices(e)
        srcs = []
        ok = True
        for c in real:
            s = src.get(str(c["geonameid"]))
            if s is None:
                ok = False
                break
            srcs.append(rebuild.source_from_hit(s))
        if not ok or not real:
            continue
        rb = rebuild.rebuild_choices(e["search_name"], list(real), srcs)
        for k, v in rebuild.parity_report(e, rb, keys).items():
            worst[k] = max(worst.get(k, 0.0), v)
        n += 1
    if n:
        print("  parity on %d untouched entities: max |diff| = %s"
              % (n, {k: round(v, 6) for k, v in sorted(worst.items())
                     if v > 1e-9} or "0.0 on every key"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                    default=os.path.join(ROOT, "experiments/e29_swa_ep15/seed101.pt"))
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--rules", default="",
                    help="comma list of abbrev,demote_h,dedupe,demote_junk "
                         "(empty = baseline)")
    ap.add_argument("--dedupe-keep", default="p", choices=["p", "a", "alt"])
    ap.add_argument("--alias-prepend-max", type=int, default=100)
    ap.add_argument("--sources", default="TR,LGL,GWN,WikiDocs,Prodigy,Synth")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    rules = [r for r in a.rules.split(",") if r]
    want = a.sources.split(",")
    srcs = [(s, st) for s, st in SOURCES if s in want]

    sidecar = a.checkpoint + ".json"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, cfg = build_model(a.checkpoint, sidecar, device)
    print("checkpoint %s  window %d  rules %s"
          % (os.path.basename(a.checkpoint), a.window, rules or ["(baseline)"]))

    t0 = time.time()
    _, per_entity, stats = run_arm(model, cfg, a.window, rules, a.dedupe_keep,
                                   a.alias_prepend_max, srcs, a.verify, {}, {})
    # gold feature codes, for the non-country (TLG-hard) split
    golds = {r["gold"] for rows in per_entity.values() for r in rows}
    fcmap = {g: (s or {}).get("feature_code", "")
             for g, s in es_util.mget(list(golds)).items()}
    for g in golds:
        fcmap.setdefault(g, "")

    summ = {s: summarise(rows, fcmap) for s, rows in per_entity.items()}
    print("\n%-9s %5s %7s %7s  %5s %8s %8s" %
          ("source", "n", "em_cond", "em_all", "hard", "hard_c", "hard_all"))
    for s in want:
        if s not in summ:
            continue
        d = summ[s]
        print("%-9s %5d %7.4f %7.4f  %5d %8.4f %8.4f" %
              (s, d["n"], d["em_cond"], d["em_all"], d["n_hard"],
               d["em_hard_cond"], d["em_hard_all"]))
    tlg = [s for s in TLG if s in summ]
    if len(tlg) == 3:
        print("TLG-hard (cond)  = %.4f"
              % np.mean([summ[s]["em_hard_cond"] for s in tlg]))
        print("TLG-hard (all)   = %.4f"
              % np.mean([summ[s]["em_hard_all"] for s in tlg]))
    if len(summ) == 6:
        print("macro-of-six     = %.4f"
              % np.mean([summ[s]["em_cond"] for s in want]))
    print("stats:", stats)
    print("%.1fs" % (time.time() - t0))

    if a.out:
        with open(a.out, "w") as f:
            json.dump(dict(checkpoint=a.checkpoint, window=a.window,
                           rules=rules, dedupe_keep=a.dedupe_keep,
                           summary=summ, stats=stats,
                           per_entity=per_entity), f)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
