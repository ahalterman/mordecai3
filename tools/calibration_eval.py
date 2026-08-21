"""Calibration and abstention report for a trained ranker checkpoint.

The campaign's headline metric answers "when the gold is reachable, does the
model pick it".  This script answers the two questions a downstream user asks
instead: *how much should I trust this answer*, and *when should the model
refuse to answer at all*.

What the model actually emits, and why the slot layout matters here:

  * `es_choices` is the ES candidate list with a "none of the above" NULL row
    appended last (mordecai3/geoparse.py `_null_choice`), so a mention with
    `k` hits has `k+1` entries.
  * `ProductionData` lays those out in `max_choices` rows and overwrites the
    LAST row with a reserved sentinel (feature code 53, country NULL, every
    gazetteer feature -1).  `TrainData.create_labels` points at that reserved
    row whenever nothing in the list is correct, so **the reserved row is a
    trained "no correct answer" class**, not dead space.
  * The reserved row is live in the mask but sits at index `max_choices-1`, so
    on a mention with fewer candidates than the window `evaluate_results`
    never ranks it -- the campaign's exact match is blind to it.  Serving is
    not: `geoparse.py` tests `pred[-1] == pred.max()` separately and returns a
    blank result, so the shipped model already abstains.  On a mention whose
    candidate list FILLS the window the sentinel sits at an index the scorer
    does read, and its score is then attributed to the real candidate it
    overwrote -- an answer the model never scored.  All of this is measured
    below.

Scores reported for every entity: top-1 probability, margin over the runner-up
(probability and logit), entropy, the reserved row's probability, the NULL
row's probability, and -- with several checkpoints -- seed disagreement.

    uv run python tools/calibration_eval.py --checkpoints experiments/e29_swa_ep15/seed42.pt
    uv run python tools/calibration_eval.py --checkpoints "experiments/e29_swa_ep15/seed*.pt" \
        --window 100 --preds-out experiments/campaign2/preds/e29_w100.parquet

`--window` is the scoring window: the checkpoints are trained at 500 but
`Geoparser` serves at `max_choices=100`, and the two differ in what is
reachable (retrieval recall) and in whether the reserved row is selectable.
Report the serving window when the question is "what should we ship".

NO DEV SET EXISTS in this project: the six sources are split 70/30 and the 30%
is what every campaign number is measured on.  Temperature is therefore fit
leave-one-source-out -- T comes from the other five sources' held-out halves
and is scored on the sixth, which never sees its own data.  The pooled "global
T" row is the in-sample fit and is optimistic by construction; it is printed
for reference, not as a result.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mordecai3.torch_model import ProductionData                 # noqa: E402
from twin_credit_eval import (EXPECTED_VAL, SOURCES, build_model,  # noqa: E402
                              load_val)
from rewrite_labels import gold_index                            # noqa: E402

NEG = -1e30


# ---------------------------------------------------------------- extraction

def raw_logits(model, loader):
    """Model output for every entity, (N, window). Padded rows are already
    -1e9 for a `--mask-padding` checkpoint; the mask is reapplied here anyway
    so checkpoints trained without it are treated identically."""
    device = next(model.parameters()).device
    out = []
    with torch.no_grad():
        model.eval()
        for inp in loader:
            inp = {k: v.to(device, non_blocking=True) for k, v in inp.items()}
            pred = model(inp)
            if model.country_pred:
                pred = pred[0]
            out.append(pred.float().cpu().numpy())
    return np.vstack(out)


def masked_softmax(logits, mask, temperature=1.0):
    """Softmax over the True entries of `mask`, at temperature T."""
    z = np.where(mask, logits / temperature, NEG)
    z = z - z.max(axis=1, keepdims=True)
    e = np.where(mask, np.exp(z), 0.0)
    return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-300)


class SourceBatch:
    """Everything one held-out source contributes, model output included.

    `sel_mask` is what the scorer sees (`evaluate_results` reads
    `es_choices[:len(pred)]`); `full_mask` adds the reserved row, which is what
    the model's own softmax ran over during training.
    """

    def __init__(self, source, es_data, logits, window):
        self.source = source
        self.window = window
        n = len(es_data)
        self.n_choices = np.array([len(e["es_choices"]) for e in es_data], np.int32)
        self.n_live = np.minimum(self.n_choices, window)
        self.full_list = self.n_choices > window
        cols = np.arange(window)[None, :]
        self.sel_mask = cols < self.n_live[:, None]
        self.full_mask = self.sel_mask.copy()
        self.full_mask[:, window - 1] = True
        # The unified reserved-row convention (mordecai3.geoparse; §2a of
        # experiments/campaign2/calibration_report.md): the reserved row is
        # never a candidate, so on a list that fills the window the row it
        # overwrote is not selectable either. `sel_mask` is kept as it was --
        # it is what `evaluate_results` sees, and the frozen numbers are
        # reproduced from it.
        self.cand_mask = self.sel_mask.copy()
        self.cand_mask[self.full_list, window - 1] = False
        self.logits = [np.where(self.full_mask, lg, NEG) for lg in logits]

        gi = np.array([gold_index(e) if gold_index(e) is not None else -1
                       for e in es_data], np.int32)
        self.gold_idx = gi
        self.gold_retrievable = gi >= 0
        # A gold the scorer could actually pick: inside the window, and not the
        # in-window row the reserved sentinel overwrites.
        self.gold_in_window = (gi >= 0) & (gi < self.n_live) & \
            ~(self.full_list & (gi == window - 1))
        self.gold_gid = np.array([str(e["correct_geonamesid"]) for e in es_data])
        self.gids = [np.array([str(c["geonameid"]) for c in e["es_choices"]])
                     for e in es_data]
        self.gold_code = np.array([
            str(e["es_choices"][gi[i]].get("feature_code", ""))
            if gi[i] >= 0 else "" for i, e in enumerate(es_data)])
        self.lat = [np.array([float(c["lat"]) for c in e["es_choices"]]) for e in es_data]
        self.lon = [np.array([float(c["lon"]) for c in e["es_choices"]]) for e in es_data]
        self.names = np.array([str(e.get("search_name")) for e in es_data])
        self.gold_lat = np.array([self.lat[i][gi[i]] if gi[i] >= 0 else np.nan
                                  for i in range(n)])
        self.gold_lon = np.array([self.lon[i][gi[i]] if gi[i] >= 0 else np.nan
                                  for i in range(n)])

    def probs(self, mask, temperature=1.0):
        """Ensemble-mean probabilities at temperature T (mean of members'
        softmaxes, as tools/ensemble_eval.py averages them)."""
        ps = [masked_softmax(lg, mask, temperature) for lg in self.logits]
        return ps[0] if len(ps) == 1 else np.mean(ps, axis=0)


def load_batches(models, cfg, window, data_dir, limit_types, fuzzy, batch_size):
    pickle_max = cfg.get("max_choices", 500)
    blocks = cfg.get("feature_blocks") or None
    suffix = "_enriched_compact" if cfg.get("enriched") else ""
    out = []
    for source, stems in SOURCES:
        es_data = load_val(source, stems, data_dir, suffix, pickle_max,
                           limit_types, fuzzy)
        if EXPECTED_VAL.get(source) not in (None, len(es_data)):
            sys.exit(f"{source}: held-out size {len(es_data)} != "
                     f"{EXPECTED_VAL[source]}; the split drifted")
        ds = ProductionData(es_data, max_choices=window,
                            oov_bucket_fix=cfg.get("oov_bucket_fix", False),
                            feature_blocks=blocks,
                            full_null_row=cfg.get("full_null_row", False))
        loader = DataLoader(dataset=ds, batch_size=batch_size, shuffle=False)
        logits = [raw_logits(m, loader) for m in models]
        out.append(SourceBatch(source, es_data, logits, window))
        del es_data, ds, loader
    return out


# ------------------------------------------------------------------ scoring

def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def entity_frame(batches, temperature=1.0):
    """One row per held-out entity: prediction, correctness, confidence scores."""
    import pandas as pd
    rows = []
    for b in batches:
        p_sel = b.probs(b.sel_mask, temperature)
        p_full = b.probs(b.full_mask, temperature)
        lg = np.mean([np.where(b.sel_mask, l, np.nan) for l in b.logits], axis=0)
        order = np.argsort(-p_sel, axis=1)
        pred = order[:, 0]
        top1 = p_sel[np.arange(len(pred)), pred]
        top2 = p_sel[np.arange(len(pred)), order[:, 1]]
        ent = -np.nansum(np.where(p_sel > 0, p_sel * np.log(np.maximum(p_sel, 1e-30)), 0), 1)
        lgs = np.sort(np.where(b.sel_mask, lg, -np.inf), axis=1)[:, ::-1]
        res_p = p_full[:, b.window - 1]
        raw_arg = np.argmax(p_full, axis=1)
        # Unified convention: argmax over the candidate rows decides which
        # place, p(reserved) decides whether to answer at all.
        p_cand = b.probs(b.cand_mask, temperature)
        pred_u = np.argmax(p_cand, axis=1)
        for i in range(len(pred)):
            pi = int(pred[i])
            gid = b.gids[i][pi] if pi < len(b.gids[i]) else "OUT_OF_LIST"
            ok = bool(b.gold_in_window[i] and gid == b.gold_gid[i])
            km = (haversine_km(b.gold_lat[i], b.gold_lon[i], b.lat[i][pi], b.lon[i][pi])
                  if b.gold_retrievable[i] and pi < len(b.lat[i]) else np.nan)
            nullrow = (float(p_sel[i, b.n_choices[i] - 1])
                       if b.n_choices[i] <= b.window else np.nan)
            pu = int(pred_u[i])
            gid_u = b.gids[i][pu] if pu < len(b.gids[i]) else "OUT_OF_LIST"
            abstained = bool(raw_arg[i] == b.window - 1)
            rows.append(dict(
                source=b.source, idx=i, name=b.names[i],
                n_choices=int(b.n_choices[i]), n_live=int(b.n_live[i]),
                full_list=bool(b.full_list[i]),
                gold_idx=int(b.gold_idx[i]),
                gold_retrievable=bool(b.gold_retrievable[i]),
                gold_in_window=bool(b.gold_in_window[i]),
                pred_idx=pi, pred_gid=gid, gold_gid=b.gold_gid[i],
                gold_feature_code=b.gold_code[i],
                correct=ok, err_km=km,
                p_top1=float(top1[i]), margin=float(top1[i] - top2[i]),
                logit_margin=float(lgs[i, 0] - lgs[i, 1]) if b.n_live[i] > 1 else 0.0,
                entropy=float(ent[i]),
                p_pred_full=float(p_full[i, pi]),
                p_reserved=float(res_p[i]), p_nullrow=nullrow,
                reserved_argmax=bool(raw_arg[i] == b.window - 1),
                pred_is_reserved=bool(pi == b.window - 1 and b.full_list[i]),
                # geoparse.py drops a mention when the last *scored* row wins:
                # the gazetteer NULL row on a short list, the reserved sentinel
                # on a full one.
                pred_is_last_row=bool(pi == b.n_live[i] - 1),
                # --- unified reserved-row convention ---------------------
                pred_idx_unified=pu, pred_gid_unified=gid_u,
                # gold_in_window already excludes the overwritten row, so it
                # is the unified convention's "answerable" too.
                picked_unified=bool(b.gold_in_window[i] and
                                    gid_u == b.gold_gid[i]),
                abstained=abstained,
                correct_unified=bool(b.gold_in_window[i] and not abstained and
                                     gid_u == b.gold_gid[i]),
            ))
    df = pd.DataFrame(rows)
    df["group"] = np.where(df.gold_in_window, "a_answerable",
                           np.where(df.gold_retrievable, "b_out_of_window",
                                    "c_unretrievable"))
    return df


def per_seed_disagreement(batches):
    """Vote disagreement and top-1 spread across the checkpoints, per entity."""
    import pandas as pd
    rows = []
    for b in batches:
        preds = [np.argmax(masked_softmax(lg, b.sel_mask), axis=1) for lg in b.logits]
        tops = np.stack([masked_softmax(lg, b.sel_mask).max(axis=1) for lg in b.logits])
        preds = np.stack(preds)
        for i in range(preds.shape[1]):
            gids = [b.gids[i][p] if p < len(b.gids[i]) else "?" for p in preds[:, i]]
            vals, cnt = np.unique(gids, return_counts=True)
            rows.append(dict(source=b.source, idx=i,
                             vote_n_distinct=int(len(vals)),
                             vote_top_frac=float(cnt.max() / len(gids)),
                             p_top1_std=float(tops[:, i].std())))
    return pd.DataFrame(rows)


# -------------------------------------------------------------- calibration

def ece(conf, correct, bins=15):
    """Expected calibration error with equal-MASS bins.

    Equal-width bins are useless here: 80% of the mass sits above 0.9, so the
    usual 10 equal-width bins put four fifths of the data in one bucket.
    """
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    n = len(conf)
    if n == 0:
        return float("nan"), []
    order = np.argsort(conf)
    conf, correct = conf[order], correct[order]
    e, table = 0.0, []
    for ix in np.array_split(np.arange(n), bins):
        if len(ix) == 0:
            continue
        c, a = conf[ix].mean(), correct[ix].mean()
        e += len(ix) / n * abs(c - a)
        table.append((len(ix), float(conf[ix].min()), float(conf[ix].max()),
                      float(c), float(a)))
    return float(e), table


def fit_temperature(batches, sources, mask_kind="sel", grid=None):
    """T minimising NLL of the gold slot, over the named sources.

    `sel`: the distribution the scorer argmaxes, fit on answerable entities.
    `full`: adds the reserved row and fits every entity, with the unanswerable
    ones targeted at the reserved slot -- exactly the target `create_labels`
    trains on, so this is calibration of the model's own objective.
    """
    grid = grid if grid is not None else np.exp(np.linspace(np.log(0.2), np.log(5.0), 97))
    picks = [b for b in batches if b.source in sources]

    def nll(T):
        tot, n = 0.0, 0
        for b in picks:
            mask = b.sel_mask if mask_kind == "sel" else b.full_mask
            p = b.probs(mask, T)
            if mask_kind == "sel":
                keep = b.gold_in_window
                tgt = b.gold_idx.copy()
            else:
                keep = np.ones(len(b.gold_idx), bool)
                tgt = np.where(b.gold_in_window, b.gold_idx, b.window - 1)
            if keep.sum() == 0:
                continue
            pr = p[np.arange(len(tgt)), np.clip(tgt, 0, b.window - 1)][keep]
            tot += -np.log(np.maximum(pr, 1e-30)).sum()
            n += keep.sum()
        return tot / max(n, 1)

    vals = [nll(T) for T in grid]
    return float(grid[int(np.argmin(vals))]), float(np.min(vals))


# ---------------------------------------------------- abstention / selection

def auroc(score, label):
    """AUROC by rank statistic; `label` 1 = the thing the score should flag."""
    score, label = np.asarray(score, float), np.asarray(label, int)
    m = ~np.isnan(score)
    score, label = score[m], label[m]
    if label.sum() == 0 or label.sum() == len(label):
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty(len(score), float)
    s = score[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    npos, nneg = label.sum(), len(label) - label.sum()
    return float((ranks[label == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def risk_coverage(conf, correct, grid=(1.0, .95, .9, .8, .7, .6, .5)):
    """Selective accuracy at each coverage level, plus AURC over all cutoffs."""
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    order = np.argsort(-conf)
    c = correct[order]
    cum = np.cumsum(c) / np.arange(1, len(c) + 1)
    aurc = float(np.mean(1 - cum))
    out = []
    for g in grid:
        k = max(1, int(round(g * len(c))))
        out.append((g, float(cum[k - 1]), float(conf[order][k - 1])))
    return out, aurc


def logistic_score(train_X, train_y, X, iters=400, lr=0.5):
    """Tiny logistic regression (no sklearn dependency) on standardised scores."""
    mu, sd = train_X.mean(0), train_X.std(0) + 1e-9
    Xt = (train_X - mu) / sd
    Xe = np.hstack([Xt, np.ones((len(Xt), 1))])
    w = np.zeros(Xe.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Xe @ w))
        g = Xe.T @ (p - train_y) / len(Xe)
        h = (Xe * (p * (1 - p))[:, None]).T @ Xe / len(Xe) + 1e-4 * np.eye(Xe.shape[1])
        w -= np.linalg.solve(h, g)
    Xs = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    return 1 / (1 + np.exp(-Xs @ w)), w


# ------------------------------------------------- campaign-2 scoreboard

def add_scoreboard_columns(df, cache_path):
    """Attach the novel-pair and twin-credit columns from the metric cache.

    The cache (tools/twin_credit_eval.py --cache-out) holds, per held-out
    entity, the geonameids in its gold answer's A/P twin class, plus every
    (mention, gold id) pair that occurs in the training half. Neither depends
    on the model, so both are computed once and read here.
    """
    import numpy as _np
    df["novel_pair"] = _np.nan
    df["twin_ok"] = _np.nan
    df["twin_ok_unified"] = _np.nan
    if not cache_path or not os.path.exists(cache_path):
        print(f"(no metric cache at {cache_path}: novel-pair and twin-credit "
              f"exact match are not reported)\n")
        return df
    with open(cache_path) as f:
        cache = json.load(f)
    pairs = {(str(a), str(b)) for a, b in cache.get("train_pairs", [])}
    if pairs:
        df["novel_pair"] = [(str(n), str(g)) not in pairs
                            for n, g in zip(df.name, df.gold_gid)]
    twins = cache.get("sources", {})
    tw_ok, tw_ok_u = [], []
    sizes = df.groupby("source").size().to_dict()
    usable = {s: rows for s, rows in twins.items()
              if len(rows) == sizes.get(s, -1)}
    for s in sizes:
        if s not in usable and s in twins:
            print(f"(twin cache for {s} has {len(twins[s])} entities, this run "
                  f"has {sizes[s]}: twin credit skipped for it)")
    for r in df.itertuples():
        rows = usable.get(r.source)
        if rows is None:
            tw_ok.append(np.nan)
            tw_ok_u.append(np.nan)
            continue
        tw = set(rows[r.idx])
        tw_ok.append(bool(r.correct or (r.gold_in_window and
                                        str(r.pred_gid) in tw)))
        tw_ok_u.append(bool(r.correct_unified or
                            (r.gold_in_window and not r.abstained and
                             str(r.pred_gid_unified) in tw)))
    df["twin_ok"] = tw_ok
    df["twin_ok_unified"] = tw_ok_u
    return df


def scoreboard(df, headline_drop=("Synth",)):
    """TLG-hard, novel-pair EM, twin credit, and the three accuracies."""
    ans = df[df.gold_in_window]
    keep = ans[~ans.source.isin(headline_drop)]
    tlg = ans[ans.source.isin(["TR", "LGL", "GWN"])]
    tlg = tlg[~tlg.gold_feature_code.str.startswith("PCL")]

    def macro(frame, col):
        return float(frame.groupby("source")[col].mean().mean()) if len(frame) \
            else float("nan")

    out = {
        "tlg_hard": macro(tlg, "correct_unified"),
        "tlg_hard_legacy": macro(tlg, "correct"),
        "tlg_hard_n": int(len(tlg)),
        "macro_em_headline": macro(keep, "correct_unified"),
        "macro_em_headline_legacy": macro(keep, "correct"),
        "macro_em_six_legacy": macro(ans, "correct"),
        "macro_em_six_unified": macro(ans, "correct_unified"),
        "em_conditioned": float(ans.correct_unified.mean()),
        "em_all_mentions": float(df.correct_unified.mean()),
        "abstain_rate": float(df.abstained.mean()),
        "abstain_precision": float((~df[df.abstained].gold_in_window).mean())
        if df.abstained.any() else float("nan"),
        "unanswerable_rate": float((~df.gold_in_window).mean()),
        # what the unification costs on the frozen key, decomposed
        "delta_selection": float(ans.picked_unified.mean() - ans.correct.mean()),
        "delta_abstention": float(ans.correct_unified.mean() -
                                  ans.picked_unified.mean()),
    }
    if df.novel_pair.notna().any():
        nov = ans[ans.novel_pair == True]                        # noqa: E712
        seen = ans[ans.novel_pair == False]                      # noqa: E712
        out.update(novel_pair_em=float(nov.correct_unified.mean()),
                   novel_pair_em_legacy=float(nov.correct.mean()),
                   novel_pair_n=int(len(nov)),
                   seen_pair_em=float(seen.correct_unified.mean()),
                   seen_pair_em_legacy=float(seen.correct.mean()))
    if df.twin_ok.notna().any():
        t = ans[ans.twin_ok.notna()]
        out.update(twin_credit_macro=macro(t[~t.source.isin(headline_drop)],
                                           "twin_ok_unified"),
                   twin_credit_macro_legacy=macro(
                       t[~t.source.isin(headline_drop)], "twin_ok"),
                   twin_credit_six_legacy=macro(t, "twin_ok"))
    return out


def print_scoreboard(sb):
    print("\n## Campaign-2 scoreboard and the reserved-row convention\n")
    print("`legacy` is the campaign's convention (`error_utils.evaluate_results`: "
          "argmax over `es_choices`, the reserved row invisible); `unified` is "
          "the one adopted in Phase 0 -- the reserved row is never a candidate, "
          "and when it wins the mention is an explicit abstention rather than "
          "an answer.\n")
    print("| metric | legacy | unified |")
    print("|---|---|---|")
    rows = [("**TLG-hard** (TR/LGL/GWN macro EM, non-country golds) -- PRIMARY",
             "tlg_hard_legacy", "tlg_hard"),
            ("macro EM, 5 sources (headline, Synth dropped -- D4)",
             "macro_em_headline_legacy", "macro_em_headline"),
            ("macro EM, 6 sources (ledger continuity)",
             "macro_em_six_legacy", "macro_em_six_unified"),
            ("novel-pair EM (guardrail)", "novel_pair_em_legacy",
             "novel_pair_em"),
            ("seen-pair EM", "seen_pair_em_legacy", "seen_pair_em"),
            ("twin-credit EM (macro, no Synth)", "twin_credit_macro_legacy",
             "twin_credit_macro")]
    for label, a, b in rows:
        if a not in sb and b not in sb:
            continue
        print(f"| {label} | {pct(sb.get(a, float('nan')))} | "
              f"{pct(sb.get(b, float('nan')))} |")
    print(f"\nThree accuracies (S1): conditioned on an answerable gold "
          f"{pct(sb['em_conditioned'])}, over every held-out mention "
          f"{pct(sb['em_all_mentions'])}, abstaining on "
          f"{pct(sb['abstain_rate'])} of mentions of which "
          f"{pct(sb['abstain_precision'])} really had no answer "
          f"(base rate {pct(sb['unanswerable_rate'])}).")
    print(f"\nWhere the legacy/unified gap comes from, on answerable mentions: "
          f"{sb['delta_selection']:+.4f} from no longer crediting the sentinel's "
          f"score to the candidate it overwrote, {sb['delta_abstention']:+.4f} "
          f"from charging abstentions. TLG-hard n = {sb['tlg_hard_n']}.")


# ------------------------------------------------------------------ report

def pct(x):
    return "n/a" if x != x else f"{100 * x:.2f}%"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints", nargs="+", required=True,
                    help="checkpoint path(s), or a glob in quotes; several are "
                         "averaged in probability space (the ensemble recipe)")
    ap.add_argument("--window", type=int, default=0,
                    help="scoring window (default: the checkpoint's max_choices; "
                         "100 is what Geoparser serves)")
    ap.add_argument("--data-dir", default="raw_data")
    ap.add_argument("--limit-types", default="all_loc_types")
    ap.add_argument("--fuzzy", type=int, default=0)
    ap.add_argument("--test-batch-size", type=int, default=64)
    ap.add_argument("--bins", type=int, default=15)
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="divide the logits by this before the softmax, so the "
                         "probabilities, ECE and thresholds below are the ones "
                         "serving would expose. AUROC and risk-coverage are "
                         "invariant to it (a temperature is monotone); only the "
                         "numbers on the probability axis move.")
    ap.add_argument("--metric-cache",
                    default="experiments/campaign2/twin_gold.json",
                    help="gold A/P twin classes and the training "
                         "(mention, gold id) pairs, from "
                         "`tools/twin_credit_eval.py --cache-out`. Without it "
                         "twin-credit and novel-pair EM are skipped.")
    ap.add_argument("--preds-out", default="",
                    help="per-entity table (.parquet or .csv)")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    paths = []
    for c in args.checkpoints:
        paths.extend(sorted(glob.glob(c)) if any(ch in c for ch in "*?[") else [c])
    if not paths:
        sys.exit("no checkpoints matched")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    models, cfgs = [], []
    for p in paths:
        sidecar = p + ".json"
        if not os.path.exists(sidecar):
            sidecar = os.path.splitext(p)[0] + ".json"
        m, cfg = build_model(p, sidecar, device)
        models.append(m)
        cfgs.append(cfg)
    for c in cfgs[1:]:
        if c.get("feature_blocks") != cfgs[0].get("feature_blocks"):
            sys.exit("checkpoints disagree on feature_blocks")
    cfg = cfgs[0]
    window = args.window or cfg.get("max_choices", 500)

    print(f"# Calibration and abstention report\n")
    print(f"checkpoints: {', '.join(paths)}")
    print(f"features:    {cfg.get('feature_blocks')} "
          f"({cfg.get('n_extra_features', 0)} extra columns)")
    print(f"window:      {window}"
          f"{'  (serving window)' if window == 100 else ''}")
    print(f"temperature: {args.temperature}"
          f"{'  (raw model output)' if args.temperature == 1.0 else ''}\n")

    batches = load_batches(models, cfg, window, args.data_dir, args.limit_types,
                           args.fuzzy, args.test_batch_size)
    df = entity_frame(batches, args.temperature)
    if len(models) > 1:
        df = df.merge(per_seed_disagreement(batches), on=["source", "idx"], how="left")
    else:
        df["vote_n_distinct"] = np.nan
        df["vote_top_frac"] = np.nan
        df["p_top1_std"] = np.nan
    # "answer is usable": right, and there was something right to pick.
    df["answer_ok"] = df.correct
    df["wrong"] = (~df.answer_ok).astype(int)
    df = add_scoreboard_columns(df, args.metric_cache)

    # ---- 1. what the held-out set is made of
    print("## Entity groups\n")
    print("| group | what it is | N | share | EM | mean top-1 p | share p>0.9 |")
    print("|---|---|---|---|---|---|---|")
    labels = {"a_answerable": "gold is in the scored window",
              "b_out_of_window": f"gold retrievable but past row {window}",
              "c_unretrievable": "gold not in the candidate list at all"}
    for g in ["a_answerable", "b_out_of_window", "c_unretrievable"]:
        s = df[df.group == g]
        if not len(s):
            continue
        print(f"| {g} | {labels[g]} | {len(s)} | {pct(len(s) / len(df))} | "
              f"{pct(s.correct.mean())} | {s.p_top1.mean():.3f} | "
              f"{pct((s.p_top1 > 0.9).mean())} |")
    macro = df[df.gold_in_window].groupby("source").correct.mean().mean()
    print(f"\nExact match on group (a), pooled over entities, "
          f"{pct(df[df.gold_in_window].correct.mean())}; unweighted mean over "
          f"the six sources -- the campaign's headline metric -- {pct(macro)}; "
          f"over every held-out entity, answerable or not, "
          f"{pct(df.correct.mean())}.\n")

    sb = scoreboard(df)
    print_scoreboard(sb)

    # ---- 2. calibration
    print("## Calibration of the top-1 probability (group (a) only)\n")
    ans = df[df.gold_in_window]
    print("| source | N | accuracy | mean confidence | ECE | ECE (p_pred incl. reserved) |")
    print("|---|---|---|---|---|---|")
    for src, s in ans.groupby("source"):
        e1, _ = ece(s.p_top1, s.correct, args.bins)
        e2, _ = ece(s.p_pred_full, s.correct, args.bins)
        print(f"| {src} | {len(s)} | {pct(s.correct.mean())} | {s.p_top1.mean():.3f} "
              f"| {e1:.4f} | {e2:.4f} |")
    e_pool, table = ece(ans.p_top1, ans.correct, args.bins)
    e_pool_full, _ = ece(ans.p_pred_full, ans.correct, args.bins)
    print(f"| **pooled** | {len(ans)} | {pct(ans.correct.mean())} | "
          f"{ans.p_top1.mean():.3f} | **{e_pool:.4f}** | {e_pool_full:.4f} |")
    print("\nReliability (equal-mass bins, pooled):\n")
    print("| bin | N | p range | mean confidence | accuracy | gap |")
    print("|---|---|---|---|---|---|")
    for n, (cnt, lo, hi, c, a) in enumerate(table, 1):
        print(f"| {n} | {cnt} | {lo:.3f}-{hi:.3f} | {c:.3f} | {a:.3f} | {a - c:+.3f} |")

    # ---- 3. temperature scaling, leave-one-source-out
    print("\n## Temperature scaling\n")
    names = [s for s, _ in SOURCES]
    t_glob, _ = fit_temperature(batches, names, "sel")
    t_glob_full, _ = fit_temperature(batches, names, "full")
    print(f"Global T (in-sample, all six sources): **{t_glob:.3f}** on the "
          f"selectable-rows softmax, {t_glob_full:.3f} on the "
          f"reserved-row-inclusive one.\n")
    print("| held-out source | T fit on the other five | ECE before | ECE after | "
          "accuracy | conf before | conf after |")
    print("|---|---|---|---|---|---|---|")
    loso_before, loso_after, w_before, w_after, tot = [], [], 0.0, 0.0, 0
    for src in names:
        T, _ = fit_temperature(batches, [s for s in names if s != src], "sel")
        b = [x for x in batches if x.source == src][0]
        p = b.probs(b.sel_mask, T)
        pred = np.argmax(b.probs(b.sel_mask), axis=1)
        conf_after = p[np.arange(len(pred)), pred]
        s = df[df.source == src]
        keep = s.gold_in_window.values
        e_b, _ = ece(s.p_top1.values[keep], s.correct.values[keep], args.bins)
        e_a, _ = ece(conf_after[keep], s.correct.values[keep], args.bins)
        loso_before.append(e_b)
        loso_after.append(e_a)
        w_before += e_b * keep.sum()
        w_after += e_a * keep.sum()
        tot += keep.sum()
        print(f"| {src} | {T:.3f} | {e_b:.4f} | {e_a:.4f} | "
              f"{pct(s.correct.values[keep].mean())} | "
              f"{s.p_top1.values[keep].mean():.3f} | {conf_after[keep].mean():.3f} |")
    print(f"| **mean** | | {np.mean(loso_before):.4f} | {np.mean(loso_after):.4f} | "
          f"| | |")
    print(f"| **N-weighted** | | {w_before / tot:.4f} | {w_after / tot:.4f} | | | |")

    # ---- 4. abstention
    print("\n## Abstention scores\n")
    print("AUROC for flagging an answer that is wrong -- including the entities "
          "where nothing in the window could have been right.\n")
    lab_all = df.wrong.values
    lab_a = df[df.gold_in_window].wrong.values
    lab_c = (~df.gold_in_window).astype(int).values
    scores = [("p_top1", -1), ("margin", -1), ("logit_margin", -1),
              ("entropy", +1), ("p_pred_full", -1), ("p_reserved", +1),
              ("p_nullrow", +1), ("vote_top_frac", -1), ("vote_n_distinct", +1),
              ("p_top1_std", +1)]
    print("| score | all entities | group (a) only | detects unanswerable |")
    print("|---|---|---|---|")
    rows = []
    for col, sign in scores:
        v = df[col].values.astype(float)
        if np.isnan(v).all():
            continue
        a1 = auroc(sign * v, lab_all)
        a2 = auroc(sign * v[df.gold_in_window.values], lab_a)
        a3 = auroc(sign * v, lab_c)
        rows.append((col, a1, a2, a3))
        print(f"| {col} | {a1:.4f} | {a2:.4f} | {a3:.4f} |")

    # combination, leave-one-source-out so the weights never see the source
    feats = ["p_top1", "margin", "logit_margin", "entropy", "p_reserved"]
    if df.vote_top_frac.notna().any():
        feats += ["vote_top_frac", "p_top1_std"]
    X = df[feats].fillna(0.0).values
    comb = np.zeros(len(df))
    for src in names:
        te = (df.source == src).values
        p, _ = logistic_score(X[~te], df.wrong.values[~te].astype(float), X[te])
        comb[te] = p
    df["combo_wrong"] = comb        # higher = more likely wrong
    df["combo"] = 1 - comb          # higher = more confident
    a1 = auroc(df.combo_wrong.values, lab_all)
    a2 = auroc(df.combo_wrong.values[df.gold_in_window.values], lab_a)
    a3 = auroc(df.combo_wrong.values, lab_c)
    print(f"| **combo (logistic, LOSO)** | {a1:.4f} | {a2:.4f} | {a3:.4f} |")
    print(f"\nCombination features: {', '.join(feats)}.\n")

    print("### The reserved row as an explicit flag\n")
    fire = df.reserved_argmax
    print(f"- The reserved 'no correct answer' row wins the raw argmax on "
          f"{pct(fire.mean())} of entities ({int(fire.sum())} of {len(df)}).")
    print(f"- Of those, {pct((~df[fire].gold_in_window).mean())} have no "
          f"answerable gold at all (base rate {pct((~df.gold_in_window).mean())}).")
    print(f"- Answers where it fires are right {pct(df[fire].correct.mean())} "
          f"of the time, vs {pct(df[~fire].correct.mean())} where it does not.")
    sel = df.pred_is_reserved
    if sel.any():
        print(f"- On the {int(df.full_list.sum())} entities whose candidate list "
              f"fills the window, the reserved row is *selectable*: it won "
              f"{int(sel.sum())} times, and the geoparser then returns the real "
              f"candidate the sentinel overwrote -- right "
              f"{pct(df[sel].correct.mean())} of the time.")

    print("\n`p_reserved` read as 'this mention has no right answer here':\n")
    print("| p_reserved | N | share unanswerable | share wrong |")
    print("|---|---|---|---|")
    cuts = [(0.0, .01), (.01, .05), (.05, .2), (.2, .5), (.5, 1.01)]
    for lo, hi in cuts:
        s = df[(df.p_reserved >= lo) & (df.p_reserved < hi)]
        if not len(s):
            continue
        print(f"| {lo:.2f}-{hi:.2f} | {len(s)} | "
              f"{pct((~s.gold_in_window).mean())} | {pct(s.wrong.mean())} |")

    # ---- 4b. is a confident error a *wild* error?
    print("\n### How wrong the errors are, by confidence\n")
    print("Distance from the gold place to the predicted one, over the errors "
          "in group (a). A granularity slip (city vs the identically named "
          "county) is ~0 km; picking the wrong Denver is ~1,000 km.\n")
    err = df[df.gold_in_window & ~df.correct]
    print("| top-1 p | errors | median km | share <=161 km | share >1000 km |")
    print("|---|---|---|---|---|")
    edges = [(0.0, 0.5), (0.5, 0.8), (0.8, 0.9), (0.9, 0.99), (0.99, 1.01)]
    for lo, hi in edges:
        s = err[(err.p_top1 >= lo) & (err.p_top1 < hi)]
        if not len(s):
            continue
        print(f"| {lo:.2f}-{hi:.2f} | {len(s)} | {s.err_km.median():.1f} | "
              f"{pct((s.err_km <= 161).mean())} | {pct((s.err_km > 1000).mean())} |")
    print(f"| **all** | {len(err)} | {err.err_km.median():.1f} | "
          f"{pct((err.err_km <= 161).mean())} | {pct((err.err_km > 1000).mean())} |")

    # ---- 5. risk-coverage
    print("\n## Risk-coverage (selective prediction)\n")
    print("Coverage = share of mentions answered, after abstaining on the "
          "least confident. Selective EM counts an answer on an unanswerable "
          "mention as wrong, which is what a user experiences.\n")
    cands = [("p_top1", df.p_top1.values), ("margin", df.margin.values),
             ("p_pred_full", df.p_pred_full.values), ("combo", df.combo.values)]
    grid = (1.0, .95, .9, .8, .7, .6, .5)
    print("| score | " + " | ".join(f"{int(100 * g)}%" for g in grid) + " | AURC |")
    print("|" + "---|" * (len(grid) + 2))
    for name, v in cands:
        rc, aurc = risk_coverage(np.nan_to_num(v, nan=-1), df.correct.values.astype(float), grid)
        print(f"| {name} | " + " | ".join(pct(x[1]) for x in rc) + f" | {aurc:.4f} |")
    print("\nThresholds on `p_pred_full`, the recommended score:\n")
    rc, _ = risk_coverage(df.p_pred_full.values, df.correct.values.astype(float), grid)
    print("| coverage | p_pred_full cutoff | selective EM | unanswerable mentions caught |")
    print("|---|---|---|---|")
    for g, acc, cut in rc:
        caught = (~df[df.p_pred_full < cut].gold_in_window).sum()
        print(f"| {pct(g)} | {cut:.3f} | {pct(acc)} | {caught} of "
              f"{int((~df.gold_in_window).sum())} |")

    # ---- 6. concrete serving policies
    print("\n## Serving policies\n")
    print("`flagged` = the mention the geoparser would mark low-confidence or "
          "refuse. Precision is the share of flagged mentions that really were "
          "wrong; recall is the share of all wrong answers that got flagged.\n")
    print("| policy | flagged | coverage | selective EM | flag precision | "
          "flag recall | unanswerable caught |")
    print("|---|---|---|---|---|---|---|")
    n_un = int((~df.gold_in_window).sum())
    # What mordecai3/geoparse.py already does: `pred[-1] == pred.max()` drops
    # the mention, and so does `argmax(scores) == len(scores)-1` (the last
    # scored row -- the gazetteer NULL row, or the sentinel on a full list).
    live_rule = df.reserved_argmax.values | df.pred_is_last_row.values
    policies = [("geoparse.py today: reserved argmax or last-scored-row argmax",
                 live_rule),
                ("reserved row wins the raw argmax", df.reserved_argmax.values)]
    for t in (0.5, 0.7, 0.8, 0.9):
        policies.append((f"p_pred_full < {t}", df.p_pred_full.values < t))
    policies.append(("p_pred_full < 0.7 or reserved argmax",
                     (df.p_pred_full.values < 0.7) | df.reserved_argmax.values))
    for name, flag in policies:
        kept = ~flag
        prec = df.wrong.values[flag].mean() if flag.sum() else float("nan")
        rec = df.wrong.values[flag].sum() / max(df.wrong.sum(), 1)
        print(f"| {name} | {int(flag.sum())} | {pct(kept.mean())} | "
              f"{pct(df.correct.values[kept].mean())} | {pct(prec)} | {pct(rec)} | "
              f"{int((~df.gold_in_window.values[flag]).sum())} of {n_un} |")

    if args.preds_out:
        os.makedirs(os.path.dirname(args.preds_out) or ".", exist_ok=True)
        if args.preds_out.endswith(".csv"):
            df.to_csv(args.preds_out, index=False)
        else:
            df.to_parquet(args.preds_out, index=False)
        print(f"\nwrote {args.preds_out} ({len(df)} rows)")
    if args.json_out:
        ans_ = df[df.gold_in_window]
        per_source = {src: dict(n=int(len(g)), em=float(g.correct.mean()))
                      for src, g in ans_.groupby("source")}
        # `TLG-hard` (data-quality track's proposed primary metric): macro over
        # the three human news corpora, on entities whose gold is not a country
        # (feature codes PCLI/PCLD/PCLIX/...).
        tlg = ans_[ans_.source.isin(["TR", "LGL", "GWN"])]
        tlg = tlg[~tlg.gold_feature_code.str.startswith("PCL")]
        rc90, aurc90 = risk_coverage(df.p_pred_full.values,
                                     df.correct.values.astype(float), (0.9,))
        flag = df.reserved_argmax.values
        payload = dict(
            checkpoints=paths, window=window,
            n=len(df), em_group_a=float(df[df.gold_in_window].correct.mean()),
            em_all=float(df.correct.mean()),
            macro_em_group_a=float(np.mean([v["em"] for v in per_source.values()])),
            per_source=per_source,
            tlg_hard=dict(n=int(len(tlg)),
                          macro=float(tlg.groupby("source").correct.mean().mean()),
                          pooled=float(tlg.correct.mean())),
            scoreboard=sb,
            groups={k: int(v) for k, v in df.group.value_counts().items()},
            n_unanswerable=int((~df.gold_in_window).sum()),
            sel_em_at_90=float(rc90[0][1]), aurc=float(aurc90),
            flag=dict(rate=float(flag.mean()),
                      precision=float(df.wrong.values[flag].mean()),
                      recall=float(df.wrong.values[flag].sum() / max(df.wrong.sum(), 1)),
                      unanswerable_caught=int((~df.gold_in_window.values[flag]).sum())),
            ece_pooled=e_pool, ece_pooled_full=e_pool_full,
            global_T=t_glob, global_T_full=t_glob_full,
            loso_ece_before=float(np.mean(loso_before)),
            loso_ece_after=float(np.mean(loso_after)),
            auroc={c: dict(all=a, group_a=b2, unanswerable=c2)
                   for c, a, b2, c2 in rows},
            auroc_combo=dict(all=a1, group_a=a2, unanswerable=a3),
            reserved_argmax_rate=float(fire.mean()))
        with open(args.json_out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
