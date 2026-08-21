"""Shared detection scoring on the D2 (demonym-excluded) denominator.

A gold toponym counts if it has a geonames id and is NOT a demonym. Demonym
golds and unlinked gold rows are removed from the denominator; a prediction
landing on one is still a false positive, because under D2 the detector is not
supposed to emit them.
"""
import json

D = ("/tmp/claude-1000/-home-andy-projects-mordecai3/"
     "a19357d6-880a-4251-9cdd-97e2235f449c/scratchpad/ner/data")
SOURCES = ["tr", "lgl", "gwn"]


def is_demonym(g):
    labs = set(g["spacy_labels"] or [])
    norp = ("NORP" in labs) and not (labs & {"GPE", "LOC", "EVENT_LOC"})
    return norp or g.get("gtype") == "Non_Literal_Modifier"


def load_docs(src):
    with open(f"{D}/{src}_docs.json") as f:
        return json.load(f)["docs"]


def gold_spans(doc):
    """(start_char, end_char) of every D2-countable gold toponym."""
    out = []
    for g in doc["golds"]:
        if not g["geonameid"] or is_demonym(g):
            continue
        out.append(g)
    return out


def score(docs, preds_by_doc):
    """preds_by_doc: doc_idx -> list of (start_char, end_char).

    Returns a dict of counters. `preds` are deduplicated per document.
    """
    r = {k: 0 for k in ["n_gold", "n_pred", "tp_exact", "tp_overlap",
                        "n_gold_nested", "tp_nested", "n_gold_flat", "tp_flat",
                        "fp_on_demonym", "fp_on_unlinked", "fp_other",
                        "n_gold_unalignable"]}
    for d in docs:
        golds = gold_spans(d)
        r["n_gold"] += len(golds)
        r["n_gold_unalignable"] += sum(1 for g in golds if g["tok_start"] is None)
        gset = {}
        for i, g in enumerate(golds):
            gset.setdefault((g["start"], g["end"]), i)
        preds = sorted(set(preds_by_doc.get(d["doc_idx"], [])))
        r["n_pred"] += len(preds)

        matched = set()
        for p in preds:
            gi = gset.get(p)
            if gi is not None and gi not in matched:
                matched.add(gi)
        r["tp_exact"] += len(matched)
        for i, g in enumerate(golds):
            nested = bool(g["nested_in"])
            r["n_gold_nested" if nested else "n_gold_flat"] += 1
            if i in matched:
                r["tp_nested" if nested else "tp_flat"] += 1

        # one-to-one greedy overlap match
        pairs = []
        for pi, p in enumerate(preds):
            for gi, g in enumerate(golds):
                ov = min(p[1], g["end"]) - max(p[0], g["start"])
                if ov > 0:
                    pairs.append((ov, pi, gi))
        pairs.sort(reverse=True)
        up, ug = set(), set()
        for ov, pi, gi in pairs:
            if pi in up or gi in ug:
                continue
            up.add(pi)
            ug.add(gi)
        r["tp_overlap"] += len(up)

        # what the false positives sit on
        dem = [(g["start"], g["end"]) for g in d["golds"]
               if g["geonameid"] and is_demonym(g)]
        unl = [(g["start"], g["end"]) for g in d["golds"] if not g["geonameid"]]
        for pi, p in enumerate(preds):
            if pi in up:
                continue
            if any(min(p[1], b) - max(p[0], a) > 0 for a, b in dem):
                r["fp_on_demonym"] += 1
            elif any(min(p[1], b) - max(p[0], a) > 0 for a, b in unl):
                r["fp_on_unlinked"] += 1
            else:
                r["fp_other"] += 1
    return r


def prf(r):
    def f(tp, np_, ng):
        p = tp / np_ if np_ else 0.0
        rc = tp / ng if ng else 0.0
        return p, rc, (2 * p * rc / (p + rc) if p + rc else 0.0)
    pe, re_, fe = f(r["tp_exact"], r["n_pred"], r["n_gold"])
    po, ro, fo = f(r["tp_overlap"], r["n_pred"], r["n_gold"])
    out = {"P": 100 * pe, "R": 100 * re_, "F1": 100 * fe,
           "P_ov": 100 * po, "R_ov": 100 * ro, "F1_ov": 100 * fo,
           "R_nested": 100 * r["tp_nested"] / max(r["n_gold_nested"], 1),
           "R_flat": 100 * r["tp_flat"] / max(r["n_gold_flat"], 1)}
    out.update({k: r[k] for k in r})
    return out


def add(a, b):
    return {k: a.get(k, 0) + b.get(k, 0) for k in set(a) | set(b)}
