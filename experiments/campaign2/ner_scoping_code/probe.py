"""Ranker-interaction proxy: what does each encoder's *mention* vector know?

encoder_scoping_report.md found that the ranker's mention slot lives or dies on
GeoNames feature class (city vs county vs river), that en_core_web_trf's
OntoNotes fine-tuning supplies it, and that generic encoders do not. If a
place-specialised NER retrain is going to replace that encoder, the question is
whether its fine-tuning preserves -- or improves -- that signal.

Linear probe on the mention vector alone (the slot in question), predicting the
gold candidate's feature class / country / admin1, trained on the same
positional train split and scored on the same held-out documents.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

REPO = "/home/andy/projects/mordecai3"
os.chdir(REPO)
sys.path.insert(0, REPO)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from evalcore import SOURCES, is_demonym, load_docs  # noqa: E402

D = f"{HERE}/data"
CHUNK_PIECES = 384


def mentions():
    out = []
    for src in SOURCES:
        for d in load_docs(src):
            for g in d["golds"]:
                if not g["geonameid"] or is_demonym(g) or g["tok_start"] is None:
                    continue
                out.append({"src": src, "doc_idx": d["doc_idx"],
                            "heldout": d["heldout"],
                            "a": g["tok_start"], "b": g["tok_end"],
                            "gid": str(g["geonameid"]), "name": g["phrase"]})
    return out


def labels_for(ms):
    from mordecai3.elasticsearch import setup_es_client
    from mordecai3.geonames import GeonamesService
    gs = GeonamesService(es_client=setup_es_client())
    cache = {}
    for m in ms:
        if m["gid"] not in cache:
            e = gs.get_entry_by_id(m["gid"])
            cache[m["gid"]] = ({"fc": e.get("feature_class"),
                                "fcode": e.get("feature_code"),
                                "cc": e.get("country_code3"),
                                "a1": f"{e.get('country_code3')}|{e.get('admin1_name')}"}
                               if e else None)
        m["labels"] = cache[m["gid"]]
    return ms


def spacy_vecs(ms):
    ten = {s: np.load(f"{D}/{s}_tensors.npy", mmap_mode="r") for s in SOURCES}
    offs = {s: {d["doc_idx"]: d["tok_offset"] for d in load_docs(s)}
            for s in SOURCES}
    return np.vstack([
        np.asarray(ten[m["src"]][offs[m["src"]][m["doc_idx"]] + m["a"]:
                                 offs[m["src"]][m["doc_idx"]] + m["b"]]).mean(0)
        for m in ms]).astype("float32")


def hf_vecs(ms, path, device="cuda"):
    from transformers import AutoTokenizer, RobertaModel
    tok = AutoTokenizer.from_pretrained("roberta-base")
    model = RobertaModel.from_pretrained(path, add_pooling_layer=False)
    model.to(device).eval()
    docs = {s: {d["doc_idx"]: d for d in load_docs(s)} for s in SOURCES}
    want = {}
    for i, m in enumerate(ms):
        want.setdefault((m["src"], m["doc_idx"]), []).append(i)
    out = np.zeros((len(ms), 768), dtype="float32")
    for (src, di), idxs in want.items():
        d = docs[src][di]
        bounds = (d["sent_starts"] or [0]) + [d["n_tokens"]]
        # same windowing as the pilot
        wins = []
        i = 0
        while i < len(bounds) - 1:
            j = i + 1
            while j < len(bounds) - 1 and \
                    (bounds[j + 1] - bounds[i]) * 1.6 + 2 <= CHUNK_PIECES:
                j += 1
            wins.append((bounds[i], bounds[j]))
            i = j
        h_doc = np.zeros((d["n_tokens"], 768), dtype="float32")
        for a, b in wins:
            c0 = d["tok_idx"][a]
            c1 = d["tok_idx"][b - 1] + d["tok_len"][b - 1]
            enc = tok(d["text"][c0:c1], return_offsets_mapping=True,
                      truncation=True, max_length=512, return_tensors="pt")
            om = enc.pop("offset_mapping")[0].tolist()
            with torch.no_grad():
                h = model(**{k: v.to(device) for k, v in enc.items()}
                          ).last_hidden_state[0].float().cpu().numpy()
            n = b - a
            piece_of = [[] for _ in range(n)]
            k = 0
            for pi, (s, e) in enumerate(om):
                if e <= s:
                    continue
                s += c0
                e += c0
                while k < n and d["tok_idx"][a + k] + d["tok_len"][a + k] <= s:
                    k += 1
                kk = k
                while kk < n and d["tok_idx"][a + kk] < e:
                    if s < d["tok_idx"][a + kk] + d["tok_len"][a + kk] and \
                       e > d["tok_idx"][a + kk]:
                        piece_of[kk].append(pi)
                    kk += 1
            for k in range(n):
                if piece_of[k]:
                    h_doc[a + k] = h[piece_of[k]].mean(0)
        for i in idxs:
            out[i] = h_doc[ms[i]["a"]:ms[i]["b"]].mean(0)
    del model
    torch.cuda.empty_cache()
    return out


def probe(X, ms, key, seeds=(42, 101), steps=2000, device="cuda"):
    ys = [m["labels"][key] for m in ms]
    vocab = sorted({y for y in ys if y is not None})
    idx = {v: i for i, v in enumerate(vocab)}
    keep = [i for i, y in enumerate(ys) if y is not None]
    X = X[keep]
    y = np.array([idx[ys[i]] for i in keep])
    ho = np.array([ms[i]["heldout"] for i in keep])
    Xt = torch.tensor(X[~ho], device=device)
    yt = torch.tensor(y[~ho], device=device)
    Xv = torch.tensor(X[ho], device=device)
    yv = torch.tensor(y[ho], device=device)
    mu, sd = Xt.mean(0), Xt.std(0) + 1e-6
    Xt, Xv = (Xt - mu) / sd, (Xv - mu) / sd
    accs = []
    for s in seeds:
        torch.manual_seed(s)
        lin = torch.nn.Linear(X.shape[1], len(vocab)).to(device)
        opt = torch.optim.AdamW(lin.parameters(), lr=1e-2, weight_decay=1e-3)
        for _ in range(steps):
            opt.zero_grad()
            torch.nn.functional.cross_entropy(lin(Xt), yt).backward()
            opt.step()
        with torch.no_grad():
            accs.append(float((lin(Xv).argmax(1) == yv).float().mean()))
    maj = float((yv == torch.mode(yt).values).float().mean())
    return float(np.mean(accs)), maj, int(ho.sum())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", default="spacy")
    ap.add_argument("--out", default=f"{HERE}/probe_results.json")
    args = ap.parse_args()

    cache = f"{HERE}/probe_mentions.json"
    if os.path.exists(cache):
        ms = json.load(open(cache))
    else:
        ms = labels_for(mentions())
        json.dump(ms, open(cache, "w"))
    ms = [m for m in ms if m["labels"]]
    print(len(ms), "mentions with gazetteer labels")

    res = {}
    if os.path.exists(args.out):
        res = json.load(open(args.out))
    for name in args.encoders.split(","):
        X = spacy_vecs(ms) if name == "spacy" else hf_vecs(ms, f"{HERE}/{name}")
        row = {}
        for key in ["fc", "fcode", "cc", "a1"]:
            acc, maj, n = probe(X, ms, key)
            row[key] = {"acc": acc, "majority": maj, "n_heldout": n}
            print(f"{name:22s} {key}: {acc:.3f} (majority {maj:.3f}, n={n})")
        res[name] = row
        json.dump(res, open(args.out, "w"), indent=1)
