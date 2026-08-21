"""Chunk builders for the label-scaling arms (N2).

Two label sources, one candidate-span objective, per-candidate weights.

  gold   TR/LGL/GWN D2 gold toponyms, from the pilot's cache. Complete
         annotation: every unlabelled candidate is a true negative.
  wiki   WikiDocsFull anchors, from `build_wiki.py`'s cache. PARTIAL
         annotation: an anchored toponym is a true positive, an un-anchored
         one is unlabelled -- not negative. The `loss` argument selects how
         the O-class is treated on those candidates (see WIKI_LOSSES).

Every chunk carries `spans` (S,2), `labels` (S,), `weights` (S,). Weight 0
candidates are dropped at build time, so masking is also a compute saving.
Wiki negatives are subsampled (NEG_KEEP) with a compensating weight so the
loss stays an unbiased estimate of the full-enumeration loss.
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.join(os.path.dirname(HERE), "ner")
sys.path.insert(0, PILOT)

from evalcore import SOURCES, is_demonym, load_docs  # noqa: E402

WIKI = f"{HERE}/wiki"
GOLD_D = f"{PILOT}/data"
MAX_SPAN = 8
CHUNK_PIECES = 384
NEG_KEEP = 0.12          # wiki plain-negative subsample rate

LABELS = ["", "GPE", "LOC", "FAC", "NORP", "ORG", "EVENT", "WORK_OF_ART",
          "LAW", "PRODUCT", "PERSON", "DATE", "CARDINAL", "ORDINAL", "TIME",
          "MONEY", "PERCENT", "QUANTITY", "LANGUAGE", "OTHER"]
LAB_ID = {l: i for i, l in enumerate(LABELS)}
LOC_IDS = {LAB_ID[x] for x in ("GPE", "LOC", "FAC")}
NORP_ID = LAB_ID["NORP"]
# hosts a nested toponym hides inside (mordecai3.geoparse.NESTED_LABELS)
HOST_IDS = {LAB_ID[x] for x in
            ("ORG", "FAC", "EVENT", "WORK_OF_ART", "LAW", "PRODUCT")}
NOPROP_IDS = {LAB_ID[x] for x in ("ORG", "PERSON", "WORK_OF_ART", "PRODUCT")}

WIKI_LOSSES = ("naive", "mask", "pu", "mask_prop", "mask_cap",
               "mask2", "mask2_prop", "pu2", "pseudo")


# ------------------------------------------------------------------ helpers
def enum_spans(n, sent_starts):
    """(s, e) for every span of <= MAX_SPAN tokens inside one sentence."""
    sb = sorted(set([0] + [s for s in sent_starts if 0 < s < n])) + [n]
    out = []
    for i in range(len(sb) - 1):
        lo, hi = sb[i], sb[i + 1]
        for s in range(lo, hi):
            for e in range(s + 1, min(s + MAX_SPAN, hi) + 1):
                out.append((s, e))
    return np.asarray(out, dtype="int32").reshape(-1, 2)


def window_bounds(n, sent_starts):
    """Sentence-aligned windows, same rule as the pilot's build_chunks."""
    bounds = sorted(set([0] + [s for s in sent_starts if 0 < s < n])) + [n]
    out = []
    i = 0
    while i < len(bounds) - 1:
        j = i + 1
        while j < len(bounds) - 1:
            if (bounds[j + 1] - bounds[i]) * 1.6 + 2 > CHUNK_PIECES:
                break
            j += 1
        out.append((bounds[i], bounds[j]))
        i = j
    return out


# ------------------------------------------------------------------ gold
def gold_chunks(demonym_weight=1.0):
    """TR/LGL/GWN chunks. Complete annotation, so every negative is real."""
    chunks = []
    tensors = {}
    offs = {}
    for src in SOURCES:
        docs = load_docs(src)
        tensors[src] = np.load(f"{GOLD_D}/{src}_tensors.npy", mmap_mode="r")
        offs[src] = {d["doc_idx"]: d["tok_offset"] for d in docs}
        for d in docs:
            n = d["n_tokens"]
            gold = {(g["tok_start"], g["tok_end"]) for g in d["golds"]
                    if g["geonameid"] and not is_demonym(g)
                    and g["tok_start"] is not None}
            # hard demonym negatives: the D2 demonym gold rows (spaCy NORP or
            # GeoWebNews `Non_Literal_Modifier`) plus every spaCy NORP span.
            dem = {(g["tok_start"], g["tok_end"]) for g in d["golds"]
                   if is_demonym(g) and g["tok_start"] is not None}
            dem |= {(e["start"], e["end"]) for e in d["ents"]
                    if e["label"] == "NORP"}
            dem -= gold
            base = offs[src][d["doc_idx"]]
            for a, b in window_bounds(n, d["sent_starts"]):
                sp = enum_spans(b - a, [s - a for s in d["sent_starts"]
                                        if a <= s < b])
                gsp = sp + a
                keys = [(int(x), int(y)) for x, y in gsp]
                lab = np.array([1.0 if k in gold else 0.0 for k in keys],
                               dtype="float32")
                w = np.ones(len(sp), dtype="float32")
                if demonym_weight != 1.0 and dem:
                    for i, k in enumerate(keys):
                        if k in dem:
                            w[i] = demonym_weight
                chunks.append({
                    "src": src, "doc_idx": d["doc_idx"], "a": a, "b": b,
                    "g0": base + a, "g1": base + b,
                    "spans": sp, "labels": lab, "weights": w,
                    "tok_idx": np.asarray(d["tok_idx"][a:b], dtype="int32"),
                    "tok_len": np.asarray(d["tok_len"][a:b], dtype="int32"),
                    "heldout": d["heldout"], "kind": "gold"})
    return chunks, tensors


# ------------------------------------------------------------------ wiki
def _wiki_shards():
    return sorted(glob.glob(f"{WIKI}/wiki_*.npz"))


def wiki_chunks(loss="mask", weight=1.0, doc_frac=1.0, seed=0,
                demonym_weight=1.0, silver=None, max_docs=None):
    """WikiDocsFull anchor chunks under one partial-annotation loss design.

    silver: optional {global_doc_index: [(char_start, char_end), ...]} of
            filtered silver nested spans to add as positives.
    """
    assert loss in WIKI_LOSSES, loss
    rng = np.random.default_rng(1234 + seed)
    chunks = []
    tensors = {}
    n_pos = n_prop = n_silver = n_masked = n_neg = 0
    kept_docs = 0
    for npz_path in _wiki_shards():
        stem = os.path.basename(npz_path)[len("wiki_"):-len(".npz")]
        z = np.load(npz_path)
        meta = json.load(open(f"{WIKI}/wiki_{stem}.json"))["docs"]
        tname = f"wiki_{stem}"
        tensors[tname] = np.load(f"{WIKI}/wiki_{stem}_tensors.npy",
                                 mmap_mode="r")
        tok_idx = z["tok_idx"]
        tok_len = z["tok_len"].astype("int32")
        sent_start = z["sent_start"]
        ent_label = z["ent_label"]
        doc_off = z["doc_off"]
        for di, d in enumerate(meta):
            if max_docs is not None and kept_docs >= max_docs:
                break
            if doc_frac < 1.0 and rng.random() >= doc_frac:
                continue
            kept_docs += 1
            lo, hi = int(doc_off[di]), int(doc_off[di + 1])
            n = hi - lo
            if n == 0:
                continue
            text = d["text"]
            ti = tok_idx[lo:hi]
            tl = tok_len[lo:hi]
            el = ent_label[lo:hi]
            ss = np.nonzero(sent_start[lo:hi])[0].tolist()

            # anchors -> token spans
            anchor_tok = set()
            anchor_str = set()
            for anc in d["anchors"]:
                s, e = anc["start"], anc["end"]
                m = np.nonzero((ti >= s) & (ti + tl <= e))[0]
                if len(m):
                    anchor_tok.add((int(m[0]), int(m[-1]) + 1))
                anchor_str.add(anc["phrase"])
            sil = set()
            if silver is not None:
                for s, e in silver.get(d["gi"], ()):
                    m = np.nonzero((ti >= s) & (ti + tl <= e))[0]
                    if len(m):
                        sil.add((int(m[0]), int(m[-1]) + 1))

            is_loc = np.isin(el, list(LOC_IDS))
            is_host = np.isin(el, list(HOST_IDS))
            is_norp = el == NORP_ID
            is_noprop = np.isin(el, list(NOPROP_IDS))
            toks = [text[int(a):int(a) + int(b)] for a, b in zip(ti, tl)]
            is_cap = np.array([t[:1].isupper() and t.isalpha() for t in toks])

            for a, b in window_bounds(n, ss):
                sp = enum_spans(b - a, [s - a for s in ss if a <= s < b])
                if not len(sp):
                    continue
                s0 = sp[:, 0] + a
                s1 = sp[:, 1] + a
                keys = list(zip(s0.tolist(), s1.tolist()))
                lab = np.zeros(len(sp), dtype="float32")
                w = np.full(len(sp), weight, dtype="float32")
                pos = np.array([k in anchor_tok for k in keys])
                lab[pos] = 1.0
                if sil:
                    ps = np.array([k in sil for k in keys])
                    lab[ps & ~pos] = 1.0
                    n_silver += int((ps & ~pos).sum())
                # span-level flags
                cum_loc = np.concatenate([[0], np.cumsum(is_loc)])
                any_loc = (cum_loc[s1] - cum_loc[s0]) > 0
                cum_cap = np.concatenate([[0], np.cumsum(is_cap)])
                all_cap = (cum_cap[s1] - cum_cap[s0]) == (s1 - s0)
                cum_norp = np.concatenate([[0], np.cumsum(is_norp)])
                exact_norp = (cum_norp[s1] - cum_norp[s0]) == (s1 - s0)
                cum_np_ = np.concatenate([[0], np.cumsum(is_noprop)])
                any_noprop = (cum_np_[s1] - cum_np_[s0]) > 0
                cum_host = np.concatenate([[0], np.cumsum(is_host)])
                in_host = (cum_host[s1] - cum_host[s0]) == (s1 - s0)
                surf = [text[int(ti[x]):int(ti[y - 1] + tl[y - 1])]
                        for x, y in zip(s0, s1)]
                str_hit = np.array([u in anchor_str for u in surf])

                if loss.endswith("_prop"):
                    prop = str_hit & all_cap & ~any_noprop & (lab == 0)
                    lab[prop] = 1.0
                    n_prop += int(prop.sum())

                unl = lab == 0
                if loss == "naive":
                    risky = np.zeros(len(sp), dtype=bool)
                elif loss == "mask_cap":
                    risky = all_cap
                elif loss in ("mask", "pu", "mask_prop"):
                    risky = any_loc | (str_hit & all_cap)
                else:   # mask2, mask2_prop, pu2: also the nested-host interior
                    risky = (any_loc | (str_hit & all_cap)
                             | (all_cap & in_host))
                if loss == "pseudo":
                    # keep the ambiguous candidates in full (no subsample) and
                    # flag them; a teacher head relabels them at train time.
                    risky = (any_loc | (str_hit & all_cap)
                             | (all_cap & in_host))
                risky = risky & unl
                if loss in ("pu", "pu2"):
                    w[risky] *= 0.1
                elif loss != "pseudo":
                    w[risky] = 0.0
                n_masked += int((risky & (w == 0)).sum())

                if demonym_weight != 1.0:
                    w[exact_norp & unl] *= demonym_weight

                # subsample plain negatives (positives and masked kept as is)
                no_sub = risky if loss == "pseudo" else np.zeros(len(sp), bool)
                plain = unl & (w > 0) & ~no_sub
                keep = np.ones(len(sp), dtype=bool)
                draw = rng.random(len(sp))
                keep[plain] = draw[plain] < NEG_KEEP
                w[plain] /= NEG_KEEP
                sel = keep & (w > 0)
                if not sel.any():
                    continue
                n_pos += int((lab[sel] > 0).sum())
                n_neg += int((lab[sel] == 0).sum())
                ch = {"src": tname, "doc_idx": d["gi"], "a": a, "b": b,
                      "g0": lo + a, "g1": lo + b,
                      "spans": sp[sel], "labels": lab[sel],
                      "weights": w[sel], "heldout": False, "kind": "wiki"}
                if loss == "pseudo":
                    ch["risky"] = risky[sel]
                chunks.append(ch)
        del z
    stats = {"docs": kept_docs, "chunks": len(chunks), "pos": n_pos,
             "neg_kept": n_neg, "masked": n_masked, "propagated": n_prop,
             "silver": n_silver, "loss": loss, "weight": weight,
             "doc_frac": doc_frac}
    return chunks, tensors, stats
