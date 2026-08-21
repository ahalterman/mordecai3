"""Filtered silver nested toponym labels over WikiDocsFull.

Replicates `mordecai3.geoparse.nested_gazetteer_spans` from the cached spaCy
arrays (no spaCy re-run), then applies the scoping report's filter: keep a
proposal only if the same surface string is also emitted as a standalone
GPE/LOC entity somewhere in the corpus, or is an anchor phrase somewhere in the
corpus. On TR/LGL/GWN that filter took the rule from 38.9% to 74-80% precision
at 77-88% of its yield (ner_retrain_scoping_report.md §1d).

Writes {global_doc_index: [(start_char, end_char), ...]}.
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "/home/andy/projects/mordecai3"
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
os.chdir(REPO)

from mordecai3.geoparse import GAZ_STOP  # noqa: E402
from mordecai3.geonames import GeonamesService, hit_sources  # noqa: E402
from mordecai3.elasticsearch import setup_es_client  # noqa: E402
import scaledata as SD  # noqa: E402

GEO_IDS = {SD.LAB_ID[x] for x in ("GPE", "LOC", "FAC")}
STANDALONE_IDS = {SD.LAB_ID[x] for x in ("GPE", "LOC")}


def ent_spans(el, es):
    """(tok_start, tok_end, label_id) from the per-token label/start arrays."""
    out = []
    i = 0
    n = len(el)
    while i < n:
        if el[i] and es[i]:
            j = i + 1
            while j < n and el[j] == el[i] and not es[j]:
                j += 1
            out.append((i, j, int(el[i])))
            i = j
        else:
            i += 1
    return out


def scan(max_docs=None):
    """Pass 1: candidate proposals, corpus standalone counts, anchor strings."""
    props = {}           # gi -> [(lo, hi, name, tok_a, tok_b)]
    standalone = Counter()
    anchor_names = Counter()
    seen = 0
    for npz_path in SD._wiki_shards():
        stem = os.path.basename(npz_path)[len("wiki_"):-len(".npz")]
        z = np.load(npz_path)
        meta = json.load(open(f"{SD.WIKI}/wiki_{stem}.json"))["docs"]
        tok_idx, tok_len = z["tok_idx"], z["tok_len"].astype("int32")
        ent_label, ent_start, doc_off = z["ent_label"], z["ent_start"], z["doc_off"]
        for di, d in enumerate(meta):
            if max_docs is not None and seen >= max_docs:
                break
            seen += 1
            lo_t, hi_t = int(doc_off[di]), int(doc_off[di + 1])
            if hi_t <= lo_t:
                continue
            text = d["text"]
            ti = tok_idx[lo_t:hi_t]
            tl = tok_len[lo_t:hi_t]
            el = ent_label[lo_t:hi_t]
            es = ent_start[lo_t:hi_t]
            toks = [text[int(a):int(a) + int(b)] for a, b in zip(ti, tl)]
            spans = ent_spans(el, es)
            for a, b, lid in spans:
                if lid in STANDALONE_IDS:
                    standalone[text[int(ti[a]):int(ti[b - 1] + tl[b - 1])]] += 1
            for anc in d["anchors"]:
                anchor_names[anc["phrase"]] += 1

            taken = set()
            for a, b, lid in spans:
                if lid in GEO_IDS:
                    taken.update(range(int(ti[a]), int(ti[b - 1] + tl[b - 1])))
            for anc in d["anchors"]:
                taken.update(range(anc["start"], anc["end"]))

            out = []
            for a, b, lid in spans:
                if lid not in SD.HOST_IDS:
                    continue
                for k in range(a, b):
                    for length in range(3, 0, -1):
                        if k + length > b:
                            continue
                        sub = toks[k:k + length]
                        if not all(t[:1].isupper() and t.isalpha()
                                   and t.lower() not in GAZ_STOP for t in sub):
                            continue
                        clo = int(ti[k])
                        chi = int(ti[k + length - 1] + tl[k + length - 1])
                        if any(c in taken for c in range(clo, chi)):
                            continue
                        out.append((clo, chi, text[clo:chi], k, k + length))
            if out:
                props[d["gi"]] = out
        del z
        if max_docs is not None and seen >= max_docs:
            break
    return props, standalone, anchor_names


def gazetteer_ok(names, batch=400):
    gs = GeonamesService(es_client=setup_es_client())
    ok = set()
    names = list(names)
    for i in range(0, len(names), batch):
        chunk = names[i:i + batch]
        res = gs.search_by_names([(nm, 5, 0, False, None) for nm in chunk])
        for nm, r in zip(chunk, res):
            for hit in hit_sources(r):
                if str(hit.get("name", "")).lower() != nm.lower() and \
                   str(hit.get("asciiname", "")).lower() != nm.lower():
                    continue
                if hit.get("feature_class") in ("A", "P"):
                    ok.add(nm)
                    break
        if (i // batch) % 20 == 0:
            print(f"  gazetteer {i}/{len(names)} -> {len(ok)} exact A/P")
    return ok


CACHE = f"{HERE}/silver_wiki_cache.json"

FILTERS = {
    # the scoping report's filter: the string is also emitted standalone
    "standalone": lambda nm, sa, an: sa.get(nm, 0) >= 1 or an.get(nm, 0) >= 1,
    # tighter: the string is an anchor-linked toponym somewhere in the corpus
    "anchor1": lambda nm, sa, an: an.get(nm, 0) >= 1,
    "anchor5": lambda nm, sa, an: an.get(nm, 0) >= 5,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{HERE}/silver_wiki.json")
    ap.add_argument("--filter", default="standalone", choices=list(FILTERS))
    ap.add_argument("--max-docs", type=int, default=None)
    args = ap.parse_args()

    if os.path.exists(CACHE):
        print("reusing", CACHE)
        c = json.load(open(CACHE))
        props = {int(k): [tuple(x) for x in v] for k, v in c["props"].items()}
        standalone = Counter(c["standalone"])
        anchor_names = Counter(c["anchor_names"])
        ok = set(c["ok"])
        n_raw = sum(len(v) for v in props.values())
        names = {p[2] for v in props.values() for p in v}
    else:
        print("scanning wiki for nested-gazetteer proposals...")
        props, standalone, anchor_names = scan(args.max_docs)
        n_raw = sum(len(v) for v in props.values())
        names = {p[2] for v in props.values() for p in v}
        print(f"{n_raw} raw proposals over {len(props)} docs, "
              f"{len(names)} distinct strings")
        ok = gazetteer_ok(names)
        # only the strings that cleared the gazetteer are ever needed again
        props = {k: [p for p in v if p[2] in ok] for k, v in props.items()}
        props = {k: v for k, v in props.items() if v}
        with open(CACHE, "w") as f:
            json.dump({"props": {str(k): v for k, v in props.items()},
                       "standalone": {k: v for k, v in standalone.items()
                                      if k in ok},
                       "anchor_names": {k: v for k, v in anchor_names.items()
                                        if k in ok},
                       "ok": sorted(ok)}, f)
    print(f"{len(ok)} strings are an exact geonames A/P name")

    fn = FILTERS[args.filter]
    kept_names = {nm for nm in ok if fn(nm, standalone, anchor_names)}
    print(f"{len(kept_names)} survive filter '{args.filter}'")

    out = {}
    n_kept = 0
    for gi, v in props.items():
        # leftmost-longest
        placed = sorted(v, key=lambda x: (x[0], -(x[1] - x[0])))
        taken = set()
        keep = []
        for lo, hi, nm, a, b in placed:
            if nm not in kept_names:
                continue
            if any(c in taken for c in range(lo, hi)):
                continue
            keep.append((lo, hi))
            taken.update(range(lo, hi))
        if keep:
            out[gi] = keep
            n_kept += len(keep)
    print(f"{n_kept} silver nested spans over {len(out)} documents")
    with open(args.out, "w") as f:
        json.dump({str(k): v for k, v in out.items()}, f)
    with open(args.out.replace(".json", "_stats.json"), "w") as f:
        json.dump({"raw_proposals": n_raw, "distinct": len(names),
                   "gazetteer_ok": len(ok), "kept_names": len(kept_names),
                   "kept_spans": n_kept, "docs": len(out),
                   "top_kept": Counter(
                       nm for v in props.values() for _, _, nm, _, _ in v
                       if nm in kept_names).most_common(40),
                   "top_dropped": Counter(
                       nm for v in props.values() for _, _, nm, _, _ in v
                       if nm in ok and nm not in kept_names).most_common(40)},
                  f, indent=1)


if __name__ == "__main__":
    main()
