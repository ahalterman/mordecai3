"""Calibration and selective prediction on the SERVING mention population.

`tools/calibration_eval.py` calibrates the ranker on the *oracle-span* frame:
every mention it scores is a gold toponym from the training pickles, and the
question `p_no_match` is asked is "is the gold id in this window". T = 0.874 and
the recommended `p >= 0.7` policy were fitted there.

The span head changes the population, not the model: it hands the ranker a
different set of mentions (§7a of the serving report — abstentions fall from
14.2% to 5.3%). So the operating points have to be re-derived on the frame the
head actually produces. This script builds that frame from raw text:

  spans      `Geoparser(span_detector=...)`'s own extraction, over the 260
             held-out documents
  answerable a mention whose span EXACTLY matches a D2 gold toponym and whose
             gold geonameid is a selectable candidate in the window
  correct    the pipeline answers, and the answer is that gold id -- an answer
             on an unanswerable mention is wrong, which is what a user sees

Both detectors are run on the same frame, so "did abstention quality degrade"
is a paired question and not a change of denominator.

The metric implementations are imported from `tools/calibration_eval.py` --
same `ece`, `auroc`, `risk_coverage`, `masked_softmax`, same reserved-row
convention, same temperature grid — so this stays on that lineage.

    uv run python experiments/e56_span_head_serving/calibration_refit.py
"""
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tools"))
os.chdir(REPO)

from mordecai3 import Geoparser  # noqa: E402
from mordecai3.geoparse import (CONTEXT_LABELS, add_es_data_batch,  # noqa: E402
                                candidate_row_count, doc_to_ex_expanded)
from mordecai3.torch_model import ProductionData  # noqa: E402
from calibration_eval import (auroc, ece, masked_softmax,  # noqa: E402
                              risk_coverage)
from end_to_end_eval import (demonym_gold_spans, doc_to_ex_gold,  # noqa: E402
                             heldout_doc_indices, read_corpus)

E54 = "mordecai3/assets/mordecai_2026-08-20_e54_seed42.pt"
WINDOW = 100
NEG = -1e30
SOURCES = ("tr", "lgl", "gwn")
OUTLET_SOURCES = ("tr", "lgl")
GRID = np.exp(np.linspace(np.log(0.2), np.log(5.0), 97))


# --------------------------------------------------------------- the frame

def build_frame(geo, detector):
    """One row per emitted mention: logits, masks, and whether it is answerable.

    `geo` must already carry the detector (`span_detector=`); this only uses it
    for extraction, so the two detectors share the ranker and the ES cache
    policy exactly.
    """
    rows = {k: [] for k in ("logits", "n_choices", "gold_idx", "source")}
    for src in SOURCES:
        articles = read_corpus(src)
        keep, _ = heldout_doc_indices(src, articles)
        meta = [articles[i] for i in sorted(keep)]
        texts = [a["text"] for a in meta]
        docs = list(geo.nlp.pipe(texts, batch_size=8))

        # D2: demonym gold rows are out of the task, so a mention landing on
        # one is unanswerable rather than a hallucination or a recall failure.
        gold_ex = [doc_to_ex_gold(d, a["toponyms"]) for d, a in zip(docs, meta)]
        demonyms = demonym_gold_spans(meta, gold_ex)
        gold_by_span = {}
        for a in meta:
            for t in a["toponyms"]:
                key = (a["doc_idx"], t["start"], t["end"])
                if t["geonameid"] and key not in demonyms:
                    gold_by_span[key] = str(t["geonameid"])

        all_ex = []
        for doc in docs:
            if geo.span_tagger is not None:
                all_ex.append(geo.span_tagger.doc_to_ex(
                    doc, context_labels=CONTEXT_LABELS))
            else:
                all_ex.append(doc_to_ex_expanded(
                    doc, geo_labels=geo.geo_labels, trim_spans=geo.trim_spans))
        outlets = ([a.get("domain") for a in meta]
                   if geo.uses_outlet and src in OUTLET_SOURCES else None)
        geo.geonames.clear_cache()
        all_es = add_es_data_batch(
            all_ex, geo.geonames, max_results=WINDOW,
            extra_features=bool(geo.extra_feature_keys),
            outlet_homes=geo._outlet_homes_for(len(all_ex), outlets))

        pooled = [e for d in all_es for e in d]
        if not pooled:
            continue
        ds = ProductionData(pooled, max_choices=WINDOW,
                            oov_bucket_fix=geo.oov_bucket_fix,
                            feature_blocks=geo.feature_blocks)
        loader = DataLoader(dataset=ds, batch_size=64, shuffle=False)
        out = []
        with torch.no_grad():
            geo.model.eval()
            for b in loader:
                out.append(geo.model({k: v.to(geo.model.device)
                                      for k, v in b.items()}).float().cpu()
                           .numpy())
        logits = np.vstack(out)

        i = 0
        for art, es_doc in zip(meta, all_es):
            for ent in es_doc:
                gid = gold_by_span.get((art["doc_idx"], ent["start_char"],
                                        ent["end_char"]))
                n_ch = len(ent["es_choices"])
                gi = -1
                if gid is not None:
                    n_sel = candidate_row_count(n_ch, WINDOW)
                    for j in range(min(n_sel, n_ch)):
                        if str(ent["es_choices"][j].get("geonameid")) == gid:
                            gi = j
                            break
                rows["logits"].append(logits[i])
                rows["n_choices"].append(n_ch)
                rows["gold_idx"].append(gi)
                rows["source"].append(src)
                i += 1
        del docs, all_es, pooled, ds, loader
    return {k: (np.asarray(v) if k != "logits" else np.vstack(v))
            for k, v in rows.items()}


def masks(frame):
    """The reserved-row convention, as `mordecai3.geoparse` states it."""
    n_ch = frame["n_choices"]
    cols = np.arange(WINDOW)[None, :]
    n_live = np.minimum(n_ch, WINDOW)
    sel = cols < n_live[:, None]
    full = sel.copy()
    full[:, WINDOW - 1] = True
    cand = sel.copy()
    cand[n_ch > WINDOW, WINDOW - 1] = False
    return sel, full, cand


def score_frame(frame, T):
    """Everything the operating-point tables need, at temperature T."""
    sel, full, cand = masks(frame)
    logits = np.where(full, frame["logits"], NEG)
    p = masked_softmax(logits, full, T)
    pc = np.where(cand, p, -1.0)
    pred = pc.argmax(1)
    n_ch = frame["n_choices"]
    gi = frame["gold_idx"]
    # The gazetteer NULL row is always the last candidate; picking it, or the
    # reserved row winning the raw argmax, are both refusals (`_decode`).
    raw = np.where(full, frame["logits"], NEG)
    reserved_argmax = raw[:, WINDOW - 1] >= raw.max(1)
    null_row = pred == (n_ch - 1)
    refused = reserved_argmax | null_row
    answerable = gi >= 0
    correct = answerable & (pred == gi) & ~refused
    return {"p_pred": pc[np.arange(len(pred)), pred],
            "p_no_match": p[:, WINDOW - 1],
            "pred": pred, "refused": refused, "answerable": answerable,
            "correct": correct}


def fit_T(frame):
    """T minimising the NLL of the target `create_labels` trains on.

    The `full` convention of `calibration_eval.fit_temperature`: every mention
    counts, an answerable one targets its gold slot and an unanswerable one
    targets the reserved row.
    """
    _, full, _ = masks(frame)
    logits = np.where(full, frame["logits"], NEG)
    gi = frame["gold_idx"]
    tgt = np.where(gi >= 0, gi, WINDOW - 1)
    idx = np.arange(len(tgt))
    best = (None, np.inf)
    for T in GRID:
        p = masked_softmax(logits, full, T)
        nll = -np.log(np.maximum(p[idx, np.clip(tgt, 0, WINDOW - 1)],
                                 1e-30)).mean()
        if nll < best[1]:
            best = (float(T), float(nll))
    return best


def report(name, frame, T_old=0.874):
    T_new, nll = fit_T(frame)
    out = {"n_mentions": int(len(frame["gold_idx"])),
           "n_answerable": int((frame["gold_idx"] >= 0).sum()),
           "T_refit": round(T_new, 3), "nll": round(nll, 4)}
    for label, T in (("T_old", T_old), ("T_refit", T_new)):
        s = score_frame(frame, T)
        answered = ~s["refused"]
        # ECE of the reported confidence, over answered mentions.
        e, _ = ece(s["p_pred"][answered], s["correct"][answered])
        conf = np.where(s["refused"], -1.0, s["p_pred"])
        rc, aurc = risk_coverage(conf, s["correct"].astype(float),
                                 grid=(1.0, .95, .9, .8, .7))
        wrong = (~s["correct"]).astype(int)
        blk = {
            "T": round(float(T), 3),
            "answer_everything_EM": round(100 * s["correct"].mean(), 2),
            "reserved_flag_coverage": round(100 * answered.mean(), 2),
            "reserved_flag_selective_EM": round(
                100 * s["correct"][answered].mean(), 2),
            "ECE_answered": round(e, 4),
            "AURC": round(aurc, 4),
            "auroc_wrong_p_pred": round(auroc(-s["p_pred"], wrong), 4),
            "auroc_wrong_p_no_match": round(auroc(s["p_no_match"], wrong), 4),
            "auroc_unanswerable_p_no_match": round(
                auroc(s["p_no_match"], (~s["answerable"]).astype(int)), 4),
            "risk_coverage": [(g, round(100 * a, 2), round(c, 4))
                              for g, a, c in rc],
            "policies": [],
        }
        for thr in (0.0, 0.5, 0.6, 0.7, 0.8, 0.9):
            keep = answered & (s["p_pred"] >= thr)
            if keep.sum() == 0:
                continue
            n_unans_caught = int((~s["answerable"] & ~keep).sum())
            blk["policies"].append({
                "policy": ("reserved flag only" if thr == 0.0
                           else f"p >= {thr} or flag"),
                "coverage": round(100 * keep.mean(), 2),
                "selective_EM": round(100 * s["correct"][keep].mean(), 2),
                "unanswerable_caught": n_unans_caught,
                "unanswerable_total": int((~s["answerable"]).sum())})
        out[label] = blk
    print(f"\n=== {name} ===")
    print(json.dumps(out, indent=1))
    return out


def main():
    res = {}
    for det in (None, "gold"):
        geo = Geoparser(model_path=E54, span_detector=det)
        frame = build_frame(geo, det)
        res[det or "none"] = report(f"e54 seed42 / span_detector={det!r}",
                                    frame)
        del geo, frame
    with open(os.path.join(HERE, "calibration_refit.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
