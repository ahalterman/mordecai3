"""Pilot: does a full NER retrain beat a head over the frozen trf tensors?

One span-classification objective, one head architecture, one data set, one
decode. The arms differ only in where the per-token 768-d vector comes from:

  frozen        the en_core_web_trf tensors already in the DocBins (D5's plan)
  frozen_ctx    same, plus a 2-layer transformer over the document window
  ft_onto       en_core_web_trf's OWN roberta, unfrozen (the deep retrain)
  ft_raw        stock roberta-base, unfrozen (does the OntoNotes init matter?)
  hf_onto       the extracted roberta, frozen, under this script's windowing
                (bridge control: isolates windowing from fine-tuning)

Task: every gold toponym with a geonames id that is not a demonym (decision
D2). Positives therefore include the toponyms nested inside ORG/FAC spans, and
demonyms are negatives by construction -- the retrained detector is asked to
learn the D2 policy instead of having it filtered downstream.
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
import torch.nn.functional as F

REPO = "/home/andy/projects/mordecai3"
os.chdir(REPO)
sys.path.insert(0, REPO)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from evalcore import SOURCES, add, is_demonym, load_docs, prf, score  # noqa: E402

D = f"{HERE}/data"
ONTO = f"{HERE}/roberta_onto"
MAX_SPAN = 8
CHUNK_PIECES = 384
SAVE_ENCODER = False
SAVE_PREDS = False


# ---------------------------------------------------------------- data
def build_chunks(tokenizer):
    """Sentence-aligned windows with piece alignment, gold labels and offsets."""
    chunks = []
    tensors = {}
    for src in SOURCES:
        docs = load_docs(src)
        tensors[src] = np.load(f"{D}/{src}_tensors.npy", mmap_mode="r")
        for d in docs:
            starts = d["sent_starts"] or [0]
            bounds = starts + [d["n_tokens"]]
            gold = {(g["tok_start"], g["tok_end"])
                    for g in d["golds"]
                    if g["geonameid"] and not is_demonym(g)
                    and g["tok_start"] is not None}
            i = 0
            while i < len(bounds) - 1:
                j = i + 1
                # grow the window until the piece budget is spent
                while j < len(bounds) - 1:
                    ntok = bounds[j + 1] - bounds[i]
                    if ntok * 1.6 + 2 > CHUNK_PIECES:
                        break
                    j += 1
                a, b = bounds[i], bounds[j]
                chunks.append(make_chunk(d, src, a, b, gold, tokenizer))
                i = j
    return chunks, tensors


def make_chunk(d, src, a, b, gold, tokenizer):
    tok_idx = d["tok_idx"][a:b]
    tok_len = d["tok_len"][a:b]
    c0 = tok_idx[0]
    c1 = tok_idx[-1] + tok_len[-1]
    text = d["text"][c0:c1]
    enc = tokenizer(text, return_offsets_mapping=True, truncation=True,
                    max_length=512)
    om = enc["offset_mapping"]
    n = b - a
    piece_of = [[] for _ in range(n)]
    k = 0
    for pi, (s, e) in enumerate(om):
        if e <= s:
            continue
        s += c0
        e += c0
        while k < n and tok_idx[k] + tok_len[k] <= s:
            k += 1
        kk = k
        while kk < n and tok_idx[kk] < e:
            if s < tok_idx[kk] + tok_len[kk] and e > tok_idx[kk]:
                piece_of[kk].append(pi)
            kk += 1
    # candidate spans, inside sentences, up to MAX_SPAN tokens
    sent_bounds = [s - a for s in d["sent_starts"] if a <= s < b] or [0]
    sent_bounds = sorted(set(sent_bounds + [0]))
    sb = sent_bounds + [n]
    spans, labels = [], []
    for si in range(len(sb) - 1):
        lo, hi = sb[si], sb[si + 1]
        for s in range(lo, hi):
            for e in range(s + 1, min(s + MAX_SPAN, hi) + 1):
                spans.append((s, e))
                labels.append(1.0 if (s + a, e + a) in gold else 0.0)
    return {"src": src, "doc_idx": d["doc_idx"], "heldout": d["heldout"],
            "a": a, "b": b, "c0": c0,
            "input_ids": enc["input_ids"], "piece_of": piece_of,
            "tok_idx": tok_idx, "tok_len": tok_len,
            "spans": spans, "labels": labels}


# ---------------------------------------------------------------- model
class SpanHead(nn.Module):
    def __init__(self, dim=768, hid=256, width=32, ctx_layers=0, dropout=0.2):
        super().__init__()
        self.ctx = None
        if ctx_layers:
            layer = nn.TransformerEncoderLayer(dim, 8, 1024, dropout,
                                               batch_first=True,
                                               norm_first=True)
            self.ctx = nn.TransformerEncoder(layer, ctx_layers)
        self.proj = nn.Sequential(nn.Linear(dim, hid), nn.GELU(),
                                  nn.LayerNorm(hid), nn.Dropout(dropout))
        self.width = nn.Embedding(MAX_SPAN + 1, width)
        self.out = nn.Sequential(nn.Linear(3 * hid + width, hid), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hid, 1))

    def forward(self, h, spans):
        """h: (T, dim) token vectors for one window; spans: (S, 2) long."""
        if self.ctx is not None:
            h = self.ctx(h.unsqueeze(0)).squeeze(0)
        p = self.proj(h)
        cs = torch.cat([torch.zeros(1, p.shape[1], device=p.device),
                        p.cumsum(0)], 0)
        s, e = spans[:, 0], spans[:, 1]
        mean = (cs[e] - cs[s]) / (e - s).unsqueeze(1).float()
        rep = torch.cat([p[s], p[e - 1], mean, self.width(e - s)], -1)
        return self.out(rep).squeeze(-1)


class Arm(nn.Module):
    def __init__(self, kind, seed):
        super().__init__()
        self.kind = kind
        self.encoder = None
        if kind in ("ft_onto", "ft_raw", "hf_onto"):
            from transformers import RobertaModel
            path = ONTO if kind in ("ft_onto", "hf_onto") else "roberta-base"
            self.encoder = RobertaModel.from_pretrained(path,
                                                        add_pooling_layer=False)
            if kind == "hf_onto":
                for p in self.encoder.parameters():
                    p.requires_grad = False
        self.head = SpanHead(ctx_layers=2 if kind == "frozen_ctx" else 0)

    def token_vecs(self, ch, frozen_rows, device):
        if self.encoder is None:
            return torch.as_tensor(frozen_rows, device=device)
        ids = torch.as_tensor(ch["input_ids"], device=device).unsqueeze(0)
        if self.kind == "hf_onto":
            with torch.no_grad():
                h = self.encoder(ids).last_hidden_state[0]
        else:
            h = self.encoder(ids).last_hidden_state[0]
        n = len(ch["piece_of"])
        out = torch.zeros(n, h.shape[1], device=device, dtype=h.dtype)
        for k, ps in enumerate(ch["piece_of"]):
            if ps:
                out[k] = h[ps].mean(0)
        return out


# ---------------------------------------------------------------- train/eval
def run_arm(kind, seed, chunks, tensors, epochs, device, lr_enc=2e-5,
            lr_head=1e-3, log=print):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    train = [c for c in chunks if c["split"] == "train"]
    dev = [c for c in chunks if c["split"] == "dev"]
    test = [c for c in chunks if c["split"] == "test"]

    model = Arm(kind, seed).to(device)
    groups = [{"params": model.head.parameters(), "lr": lr_head}]
    if model.encoder is not None and kind != "hf_onto":
        groups.append({"params": model.encoder.parameters(), "lr": lr_enc})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    steps = max(1, epochs * math.ceil(len(train) / 4))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[g["lr"] for g in groups], total_steps=steps,
        pct_start=0.1, anneal_strategy="linear")

    best = (-1, None, 0.5)
    t0 = time.time()
    done = 0
    for ep in range(epochs):
        model.train()
        random.shuffle(train)
        tot, nb = 0.0, 0
        opt.zero_grad()
        for i, ch in enumerate(train):
            rows = tensors[ch["src"]][ch["g0"]:ch["g1"]]
            h = model.token_vecs(ch, np.ascontiguousarray(rows), device)
            spans = torch.as_tensor(ch["spans"], device=device)
            y = torch.as_tensor(ch["labels"], device=device)
            logits = model.head(h, spans)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            (loss / 4).backward()
            tot += float(loss)
            nb += 1
            if (i + 1) % 4 == 0 or i == len(train) - 1:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()
                if done < steps - 1:
                    sched.step()
                    done += 1
        dscore, dthr = eval_split(model, dev, tensors, device)
        log(f"  {kind} s{seed} ep{ep+1:2d} loss {tot/max(nb,1):.5f} "
            f"devF1 {dscore:.2f} thr {dthr:.2f} ({time.time()-t0:.0f}s)")
        if dscore > best[0]:
            best = (dscore, {k: v.detach().clone() for k, v in
                             model.state_dict().items()}, dthr)
    model.load_state_dict(best[1])
    res = predict(model, test, tensors, device, best[2])
    if SAVE_ENCODER and model.encoder is not None:
        out = f"{HERE}/enc_{kind}_{seed}"
        model.encoder.save_pretrained(out)
        log(f"  saved encoder to {out}")
    return res, {"dev_f1": best[0], "threshold": best[2],
                 "epochs": epochs, "seconds": time.time() - t0}


@torch.no_grad()
def _scores(model, chunks, tensors, device):
    model.eval()
    out = []
    for ch in chunks:
        rows = tensors[ch["src"]][ch["g0"]:ch["g1"]]
        h = model.token_vecs(ch, np.ascontiguousarray(rows), device)
        spans = torch.as_tensor(ch["spans"], device=device)
        p = torch.sigmoid(model.head(h, spans)).cpu().numpy()
        out.append(p)
    return out


def _spans_at(ch, p, thr):
    res = []
    for (s, e), pi in zip(ch["spans"], p):
        if pi >= thr:
            res.append((ch["tok_idx"][s],
                        ch["tok_idx"][e - 1] + ch["tok_len"][e - 1]))
    return res


def eval_split(model, chunks, tensors, device):
    """Best micro-F1 over thresholds, on chunk-local gold (fast, for dev)."""
    ps = _scores(model, chunks, tensors, device)
    y = np.concatenate([c["labels"] for c in chunks])
    p = np.concatenate(ps)
    best = (-1, 0.5)
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


def predict(model, chunks, tensors, device, thr):
    ps = _scores(model, chunks, tensors, device)
    by_src = {s: {} for s in SOURCES}
    for ch, p in zip(chunks, ps):
        by_src[ch["src"]].setdefault(ch["doc_idx"], []).extend(
            _spans_at(ch, p, thr))
    return by_src


def full_score(by_src):
    per, pooled = {}, {}
    for src in SOURCES:
        docs = [d for d in load_docs(src) if d["heldout"]]
        r = score(docs, by_src[src])
        per[src] = prf(r)
        pooled = add(pooled, r)
    return {"pooled": prf(pooled), **per}


# ---------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="frozen,ft_onto")
    ap.add_argument("--seeds", default="42,101,202")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--frozen-epochs", type=int, default=30)
    ap.add_argument("--out", default=f"{HERE}/pilot_results.json")
    ap.add_argument("--save-encoder", action="store_true")
    ap.add_argument("--save-preds", action="store_true")
    args = ap.parse_args()
    global SAVE_ENCODER, SAVE_PREDS
    SAVE_ENCODER = args.save_encoder
    SAVE_PREDS = args.save_preds

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    print("building chunks...")
    chunks, tensors = build_chunks(tokenizer)

    # global row offsets into each source's tensor matrix
    offs = {}
    for src in SOURCES:
        offs[src] = {d["doc_idx"]: d["tok_offset"] for d in load_docs(src)}
    for ch in chunks:
        base = offs[ch["src"]][ch["doc_idx"]]
        ch["g0"] = base + ch["a"]
        ch["g1"] = base + ch["b"]

    # dev = 15% of the training documents, fixed across arms and seeds
    rng = random.Random(0)
    dev_docs = set()
    for src in SOURCES:
        tr = [d["doc_idx"] for d in load_docs(src) if not d["heldout"]]
        rng.shuffle(tr)
        dev_docs |= {(src, i) for i in tr[:max(1, int(0.15 * len(tr)))]}
    for ch in chunks:
        if ch["heldout"]:
            ch["split"] = "test"
        elif (ch["src"], ch["doc_idx"]) in dev_docs:
            ch["split"] = "dev"
        else:
            ch["split"] = "train"
    n = {s: sum(c["split"] == s for c in chunks) for s in ["train", "dev", "test"]}
    npos = {s: sum(sum(c["labels"]) for c in chunks if c["split"] == s)
            for s in ["train", "dev", "test"]}
    ncand = {s: sum(len(c["labels"]) for c in chunks if c["split"] == s)
             for s in ["train", "dev", "test"]}
    print("chunks", n, "positives", npos, "candidates", ncand)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}
    if os.path.exists(args.out):
        results = json.load(open(args.out))
    results.setdefault("meta", {}).update(
        {"chunks": n, "positives": npos, "candidates": ncand,
         "max_span": MAX_SPAN, "chunk_pieces": CHUNK_PIECES})
    for kind in args.arms.split(","):
        for seed in [int(s) for s in args.seeds.split(",")]:
            key = f"{kind}|{seed}"
            if key in results:
                print("skip", key)
                continue
            ep = args.frozen_epochs if kind in ("frozen", "frozen_ctx",
                                                "hf_onto") else args.epochs
            preds, info = run_arm(kind, seed, chunks, tensors, ep, device)
            res = full_score(preds)
            p = res["pooled"]
            print(f"== {key}: P {p['P']:.1f} R {p['R']:.1f} F1 {p['F1']:.1f} "
                  f"R_nested {p['R_nested']:.1f} ({info['seconds']:.0f}s)")
            results[key] = {"info": info, "scores": res}
            if SAVE_PREDS:
                with open(f"{HERE}/preds_{kind}_{seed}.json", "w") as f:
                    json.dump({s: {str(k): v for k, v in preds[s].items()}
                               for s in preds}, f)
            with open(args.out, "w") as f:
                json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
