"""N2: scale the span head's labels, one source at a time.

Same head, same decode, same D2 held-out denominator as the pilot
(`ner/spanpilot.py`); the arms differ only in which labels the head is trained
on and how the O-class is weighted on partially-annotated text.

Micro-batched over chunks so the wiki-scale mixes stay cheap on a 4090.
"""
import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.join(os.path.dirname(HERE), "ner")
sys.path.insert(0, HERE)
sys.path.insert(0, PILOT)

from evalcore import SOURCES, add, load_docs, prf, score  # noqa: E402
import scaledata as SD  # noqa: E402

MAX_SPAN = SD.MAX_SPAN


# ---------------------------------------------------------------- model
class SpanHead(nn.Module):
    """Identical to the pilot's head (0.5 M params)."""

    def __init__(self, dim=768, hid=256, width=32, dropout=0.2):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(dim, hid), nn.GELU(),
                                  nn.LayerNorm(hid), nn.Dropout(dropout))
        self.width = nn.Embedding(MAX_SPAN + 1, width)
        self.out = nn.Sequential(nn.Linear(3 * hid + width, hid), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hid, 1))

    def forward(self, h, spans):
        p = self.proj(h)
        cs = torch.cat([torch.zeros(1, p.shape[1], device=p.device,
                                    dtype=p.dtype), p.cumsum(0)], 0)
        s, e = spans[:, 0], spans[:, 1]
        mean = (cs[e] - cs[s]) / (e - s).unsqueeze(1).to(p.dtype)
        rep = torch.cat([p[s], p[e - 1], mean, self.width(e - s)], -1)
        return self.out(rep).squeeze(-1)


# ---------------------------------------------------------------- batching
BATCH_TOKENS = 1200


def make_batches(chunks, tensors, max_tokens=None, max_spans=200_000):
    max_tokens = max_tokens or BATCH_TOKENS
    """Group chunks into micro-batches; spans are offset into the group."""
    batches = []
    cur, ntok, nsp = [], 0, 0
    for c in chunks:
        s = c["g1"] - c["g0"]
        if cur and (ntok + s > max_tokens or nsp + len(c["spans"]) > max_spans):
            batches.append(cur)
            cur, ntok, nsp = [], 0, 0
        cur.append(c)
        ntok += s
        nsp += len(c["spans"])
    if cur:
        batches.append(cur)
    return batches


def materialise(batch, tensors, device):
    rows = []
    spans = []
    off = 0
    for c in batch:
        rows.append(np.asarray(tensors[c["src"]][c["g0"]:c["g1"]],
                               dtype="float32"))
        spans.append(c["spans"] + off)
        off += c["g1"] - c["g0"]
    h = torch.from_numpy(np.concatenate(rows)).to(device, non_blocking=True)
    sp = torch.from_numpy(np.concatenate(spans).astype("int64")).to(device)
    return h, sp


# ---------------------------------------------------------------- train
def train_head(stages, dev_chunks, tensors, seed, device, log=print):
    """stages: [(chunks, epochs, lr, select)] run in order on one model.

    `select=False` stages (a wiki pre-train) do not contribute checkpoints to
    the dev-F1 selection; only the final gold stage does. A single-stage run
    with select=True is the pilot's recipe.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    model = SpanHead().to(device)
    lossfn = nn.BCEWithLogitsLoss(reduction="none")
    dev_batches = make_batches(dev_chunks, tensors)
    best = (-1.0, None, 0.5)
    t0 = time.time()
    nb_total = 0
    for si, (chunks, epochs, lr, select) in enumerate(stages):
        # shuffle before batching so every micro-batch mixes the label sources
        # in proportion; otherwise a 33:1 wiki:gold chunk count gives gold
        # 1/33 of the optimiser steps regardless of its loss weight.
        cs = list(chunks)
        random.Random(seed + si).shuffle(cs)
        batches = make_batches(cs, tensors)
        nb_total += len(batches)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        steps = max(1, epochs * len(batches))
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=lr, total_steps=steps, pct_start=0.1,
            anneal_strategy="linear")
        done = 0
        order = list(range(len(batches)))
        for ep in range(epochs):
            model.train()
            random.shuffle(order)
            tot, nb = 0.0, 0
            for bi in order:
                b = batches[bi]
                h, sp = materialise(b, tensors, device)
                y = torch.from_numpy(np.concatenate(
                    [c["labels"] for c in b])).to(device)
                w = torch.from_numpy(np.concatenate(
                    [c["weights"] for c in b])).to(device)
                logits = model(h, sp)
                loss = (lossfn(logits, y) * w).sum() / w.sum().clamp(min=1e-6)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                if done < steps - 1:
                    sched.step()
                    done += 1
                tot += float(loss.detach())
                nb += 1
            f1, thr = dev_eval(model, dev_batches, tensors, device)
            log(f"    s{si} ep{ep + 1:2d} loss {tot / max(nb, 1):.5f} "
                f"devF1 {f1:.2f} thr {thr:.2f} ({time.time() - t0:.0f}s)")
            if select and f1 > best[0]:
                best = (f1, {k: v.detach().clone()
                             for k, v in model.state_dict().items()}, thr)
    if best[1] is not None:
        model.load_state_dict(best[1])
    return model, {"dev_f1": best[0], "threshold": best[2],
                   "stages": [(len(s[0]), s[1], s[2], s[3]) for s in stages],
                   "seconds": time.time() - t0, "batches": nb_total}


@torch.no_grad()
def raw_scores(model, batches, tensors, device):
    model.eval()
    out = []
    for b in batches:
        h, sp = materialise(b, tensors, device)
        out.append(torch.sigmoid(model(h, sp)).float().cpu().numpy())
    return out


def dev_eval(model, dev_batches, tensors, device):
    ps = raw_scores(model, dev_batches, tensors, device)
    p = np.concatenate(ps)
    y = np.concatenate([c["labels"] for b in dev_batches for c in b])
    best = (-1.0, 0.5)
    for thr in np.arange(0.05, 0.96, 0.05):
        pred = p >= thr
        tp = float((pred & (y > 0)).sum())
        if tp == 0:
            continue
        prec = tp / pred.sum()
        rec = tp / (y > 0).sum()
        f1 = 200 * prec * rec / (prec + rec)
        if f1 > best[0]:
            best = (f1, float(thr))
    return best


def predict_test(model, test_chunks, tensors, device, thr):
    batches = make_batches(test_chunks, tensors)
    ps = raw_scores(model, batches, tensors, device)
    by_src = {s: {} for s in SOURCES}
    i = 0
    for b in batches:
        p = ps[i]
        i += 1
        o = 0
        for c in b:
            q = p[o:o + len(c["spans"])]
            o += len(c["spans"])
            keep = q >= thr
            ti, tl = c["tok_idx"], c["tok_len"]
            spans = c["spans"][keep]
            by_src[c["src"]].setdefault(c["doc_idx"], []).extend(
                (int(ti[s]), int(ti[e - 1] + tl[e - 1])) for s, e in spans)
    return by_src


def full_score(by_src):
    per, pooled = {}, {}
    for src in SOURCES:
        docs = [d for d in load_docs(src) if d["heldout"]]
        r = score(docs, by_src[src])
        per[src] = prf(r)
        pooled = add(pooled, r)
    return {"pooled": prf(pooled), **per}


# ---------------------------------------------------------------- splits
@torch.no_grad()
def relabel_pseudo(model, chunks, tensors, device, hi=0.9, lo=0.05):
    """Self-training round: a teacher head resolves the unlabelled candidates.

    Wikipedia anchors are partial, so the ambiguous candidates (`risky`: inside
    a spaCy place span, a repeat of an anchor string, or a capitalised sub-span
    of a nested-label host) carry no usable O-label. Instead of masking them,
    the teacher labels the confident ones in both directions and only the
    middle band is dropped. Anchors stay positive whatever the teacher says.
    """
    model.eval()
    todo = [c for c in chunks if "risky" in c]
    batches = make_batches(todo, tensors)
    n_pos = n_neg = n_drop = 0
    for b in batches:
        h, sp = materialise(b, tensors, device)
        p = torch.sigmoid(model(h, sp)).float().cpu().numpy()
        o = 0
        for c in b:
            q = p[o:o + len(c["spans"])]
            o += len(c["spans"])
            r = c["risky"] & (c["labels"] == 0)
            c["labels"] = np.where(r & (q >= hi), 1.0,
                                   c["labels"]).astype("float32")
            drop = r & (q > lo) & (q < hi)
            c["weights"] = np.where(drop, 0.0, c["weights"]).astype("float32")
            n_pos += int((r & (q >= hi)).sum())
            n_neg += int((r & (q <= lo)).sum())
            n_drop += int(drop.sum())
    return {"pseudo_pos": n_pos, "pseudo_neg": n_neg, "pseudo_dropped": n_drop}


def gold_split(chunks):
    """The pilot's split: 15% of training documents as dev, rng seed 0."""
    rng = random.Random(0)
    dev_docs = set()
    for src in SOURCES:
        tr = [d["doc_idx"] for d in load_docs(src) if not d["heldout"]]
        rng.shuffle(tr)
        dev_docs |= {(src, i) for i in tr[:max(1, int(0.15 * len(tr)))]}
    train, dev, test = [], [], []
    for c in chunks:
        if c["heldout"]:
            test.append(c)
        elif (c["src"], c["doc_idx"]) in dev_docs:
            dev.append(c)
        else:
            train.append(c)
    return train, dev, test


# ---------------------------------------------------------------- driver
def run(args, seed, cache):
    gold_tr, gold_dev, gold_test = cache["gold_split"]
    tensors = dict(cache["gold_tensors"])
    train = list(gold_tr)
    stats = {}
    if args.wiki:
        key = (args.wiki_loss, args.wiki_weight, args.wiki_frac,
               args.demonym_weight, args.silver)
        if key not in cache["wiki"]:
            silver = cache.get("silver_map") if args.silver else None
            wc, wt, st = SD.wiki_chunks(
                loss=args.wiki_loss, weight=args.wiki_weight,
                doc_frac=args.wiki_frac, seed=0,
                demonym_weight=args.demonym_weight, silver=silver)
            cache["wiki"][key] = (wc, wt, st)
            print("  wiki:", st)
        wc, wt, st = cache["wiki"][key]
        train = train + wc
        tensors.update(wt)
        stats["wiki"] = st
    if args.demonym_weight != 1.0:
        # rebuild gold chunks with upweighted demonym negatives
        gk = args.demonym_weight
        if gk not in cache["gold_dem"]:
            gc, gt = SD.gold_chunks(demonym_weight=gk)
            cache["gold_dem"][gk] = gold_split(gc)
        gtr, gdv, gte = cache["gold_dem"][gk]
        train = [c for c in train if c["kind"] != "gold"] + list(gtr)
        gold_dev, gold_test = gdv, gte
    if args.gold_repeat == 0:          # diagnostic: wiki labels alone
        train = [c for c in train if c["kind"] != "gold"]
    elif args.gold_repeat > 1:
        g = [c for c in train if c["kind"] == "gold"]
        train = train + g * (args.gold_repeat - 1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if getattr(args, "teacher", ""):
        ck = torch.load(args.teacher, map_location="cpu", weights_only=False)
        t = SpanHead().to(device)
        t.load_state_dict(ck["state_dict"])
        # relabelling mutates the chunks, so work on private copies
        fresh = []
        for c in train:
            if c["kind"] == "wiki":
                c = dict(c, labels=c["labels"].copy(),
                         weights=c["weights"].copy())
            fresh.append(c)
        train = fresh
        wiki_chunks_ = [c for c in train if c["kind"] == "wiki"]
        st = relabel_pseudo(t, wiki_chunks_, tensors, device,
                            hi=getattr(args, "pseudo_hi", 0.9),
                            lo=getattr(args, "pseudo_lo", 0.05))
        print("  pseudo:", st)
        stats["pseudo"] = st
        del t
    if getattr(args, "stage2_epochs", 0):
        # wiki pre-train, then fine-tune on gold alone: the scaled labels shape
        # the representation, the gold corpora keep the span convention.
        gold_only = [c for c in train if c["kind"] == "gold"]
        stages = [(train, args.epochs, 1e-3, False),
                  (gold_only, args.stage2_epochs,
                   getattr(args, "stage2_lr", 3e-4), True)]
    else:
        stages = [(train, args.epochs, 1e-3, True)]
    model, info = train_head(stages, gold_dev, tensors, seed, device)
    preds = predict_test(model, gold_test, tensors, device, info["threshold"])
    res = full_score(preds)
    info.update(stats)
    return model, info, res, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--seeds", default="42,101,202")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--wiki", action="store_true")
    ap.add_argument("--wiki-loss", default="mask", choices=SD.WIKI_LOSSES)
    ap.add_argument("--wiki-weight", type=float, default=1.0)
    ap.add_argument("--wiki-frac", type=float, default=1.0)
    ap.add_argument("--demonym-weight", type=float, default=1.0)
    ap.add_argument("--silver", default="")
    ap.add_argument("--out", default=f"{HERE}/results.json")
    ap.add_argument("--save-preds", action="store_true")
    ap.add_argument("--save-head", default="")
    ap.add_argument("--batch-tokens", type=int, default=1200)
    ap.add_argument("--gold-repeat", type=int, default=1)
    ap.add_argument("--stage2-epochs", type=int, default=0)
    ap.add_argument("--stage2-lr", type=float, default=3e-4)
    args = ap.parse_args()
    global BATCH_TOKENS
    BATCH_TOKENS = args.batch_tokens

    os.chdir("/home/andy/projects/mordecai3")
    print("building gold chunks...")
    gc, gt = SD.gold_chunks()
    cache = {"gold_split": gold_split(gc), "gold_tensors": gt,
             "wiki": {}, "gold_dem": {}}
    if args.silver:
        sm = json.load(open(args.silver))
        cache["silver_map"] = {int(k): [tuple(x) for x in v]
                               for k, v in sm.items()}
        print(f"silver: {sum(len(v) for v in cache['silver_map'].values())} "
              f"spans over {len(cache['silver_map'])} docs")
    n = {k: len(v) for k, v in zip(["train", "dev", "test"],
                                   cache["gold_split"])}
    print("gold chunks", n)

    results = json.load(open(args.out)) if os.path.exists(args.out) else {}
    for seed in [int(s) for s in args.seeds.split(",")]:
        key = f"{args.name}|{seed}"
        if key in results:
            print("skip", key)
            continue
        print(f"== {key}")
        model, info, res, preds = run(args, seed, cache)
        p = res["pooled"]
        print(f"== {key}: P {p['P']:.2f} R {p['R']:.2f} F1 {p['F1']:.2f} "
              f"nestedR {p['R_nested']:.1f} demFP {p['fp_on_demonym']} "
              f"({info['seconds']:.0f}s)")
        results[key] = {"info": info, "scores": res,
                        "args": {k: v for k, v in vars(args).items()}}
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)
        if args.save_preds:
            with open(f"{HERE}/preds_{args.name}_{seed}.json", "w") as f:
                json.dump({s: {str(k): v for k, v in preds[s].items()}
                           for s in preds}, f)
        if args.save_head:
            torch.save({"state_dict": model.state_dict(),
                        "threshold": info["threshold"],
                        "max_span": MAX_SPAN, "arm": args.name, "seed": seed},
                       args.save_head.replace("SEED", str(seed)))


if __name__ == "__main__":
    main()
