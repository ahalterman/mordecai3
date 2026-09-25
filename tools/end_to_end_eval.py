"""End-to-end (NER + resolution) evaluation for the mordecai3 pipeline.

The accuracy campaign scored resolution *given* a gold mention span: every
number in ACCURACY_CAMPAIGN.md is computed over the entities in the pickles,
and those entities only exist because spaCy already tagged the gold span
GPE/LOC (tools/train.py `data_formatter`) and because the gold id came back in
the candidate list (tools/error_utils.py skips the rest). In deployment the
spans come from spaCy NER over raw text, so misses, boundary errors and
spurious spans are invisible in those numbers.

This script runs the real serving path (mordecai3.geoparse) over the raw text
of the held-out TR-News / LGL / GeoWebNews documents, aligns predicted spans to
the gold ones, and reports:

  * span detection precision/recall/F1 (exact and overlap matching)
  * end-to-end accuracy: gold toponym detected AND resolved to the right
    geonameid (plus acc@161km)
  * a decomposition of every lost gold toponym into NER miss / NER boundary
    error / retrieval miss / ranker error
  * spurious spans and what the pipeline resolves them to
  * the resolution-only numbers on the *same* documents, so the NER gap is
    measured apples-to-apples

Denominator (decision D2, experiments/campaign2/SYNTHESIS.md): **demonyms are
out of the task**. A gold toponym spaCy sees as NORP -- "Turkish",
"Palestinian", which all three corpora annotate with the country's geonameid --
is excluded from every denominator here, exactly as the corpora's unlinked gold
rows already were, and a predicted span landing on one is not counted as a
hallucination. Every summary also carries `legacy_incl_demonym`, the same
numbers on the pre-D2 gold set, so the tables of
experiments/campaign2/end_to_end_report.md stay checkable. The `*norp*` and
`*_best`/`combo_best` variants add NORP spans to the *predictions*: under D2
they can only cost precision, and they are kept for continuity, not as
candidates to ship (`Geoparser` no longer has an `accept_norp` flag).

Variants (`--variants`) re-run the same documents under a different NER
configuration -- extra entity labels, a different spaCy model, case
normalisation, a gazetteer second pass -- and are scored identically. The
`serving*` variants are what mordecai3.Geoparser does today; the `ship`
family is the pre-fix path (untrimmed spans, NORP in the context tensor),
kept so the baseline rows of experiments/campaign2/end_to_end_report.md
still reproduce.

Usage
-----
    uv run python tools/end_to_end_eval.py evaluate \\
        --model-path experiments/e29_swa_ep15/seed42.pt \\
        --variants serving,serving_norp,serving_best,ship \\
        --out experiments/campaign2/e2e_heldout.json

    uv run python tools/end_to_end_eval.py latency --n-docs 50
"""
import json
import logging
import os
import pickle
import re
import time
from collections import Counter, defaultdict

import haversine as hs
import numpy as np
import torch
import typer
import xmltodict
from torch.utils.data import DataLoader

from mordecai3 import Geoparser
from mordecai3.geonames import hit_sources
from mordecai3.geoparse import (CONTEXT_LABELS, GEO_LABELS, add_es_data_batch,
                                candidate_row_count, doc_to_ex_expanded,
                                guess_in_rel, lookup_outlet_home,
                                trim_span_tokens)
from mordecai3.span_head import load_span_tagger
from mordecai3.torch_model import ProductionData

logger = logging.getLogger("end_to_end_eval")
logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")

# The label sets the serving pipeline uses (mordecai3/geoparse.py
# doc_to_ex_expanded): GEO_LABELS are geoparsed, the context labels only
# contribute to the context tensor. Both are imported from the serving module
# so this harness cannot drift away from what ships.
# LEGACY_CTX_LABELS is the pre-fix context set: serving used to pool
# EVENT_LOC/NORP into locs_tensor while every training formatter but one used
# GPE/LOC. The baseline rows of experiments/campaign2/end_to_end_report.md
# were measured with it, so the variants that reproduce them keep it.
LEGACY_CTX_LABELS = ("GPE", "LOC", "EVENT_LOC", "NORP")

SOURCES = {
    "tr": "Pragmatic-Guide-to-Geoparsing-Evaluation/data/Corpora/TR-News.xml",
    "lgl": "Pragmatic-Guide-to-Geoparsing-Evaluation/data/Corpora/lgl.xml",
    "gwn": "Pragmatic-Guide-to-Geoparsing-Evaluation/data/GWN.xml",
}

app = typer.Typer(add_completion=False)


# --------------------------------------------------------------------------
# Corpora
# --------------------------------------------------------------------------

def _as_list(x):
    """xmltodict collapses a single repeated child into a bare dict."""
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


def read_corpus(source, base_dir="raw_data"):
    """Articles with their gold toponyms, in the order tools/train.py reads them.

    Every toponym row in the XML is kept. `geonameid` is None for the rows that
    have no gazetteer link -- GeoWebNews annotates a lot of those ("church",
    "the building", literal expressions), and TR/LGL a few. They are excluded
    from the gold set that recall is measured against, but a predicted span
    that lands on one is not counted as a hallucination either.
    """
    fn = os.path.join(base_dir, SOURCES[source])
    with open(fn, encoding="utf-8") as f:
        data = xmltodict.parse(f.read())
    articles = data["articles"]["article"]
    out = []
    for n, art in enumerate(articles):
        tops = []
        for t in _as_list((art.get("toponyms") or {}).get("toponym")):
            gtype = t.get("type")
            if source == "gwn":
                gid = t.get("geonamesID") or None
                phrase = t.get("extractedName")
                lat, lon = t.get("latitude"), t.get("longitude")
            else:
                gaz = t.get("gaztag")
                gid = (gaz.get("@geonameid") if gaz else None) or None
                phrase = t.get("phrase")
                lat = gaz.get("lat") if gaz else None
                lon = gaz.get("lon") if gaz else None
            try:
                start, end = int(t["start"]), int(t["end"])
            except (KeyError, TypeError, ValueError):
                continue
            tops.append({"start": start, "end": end, "phrase": phrase,
                         "geonameid": gid, "gtype": gtype,
                         "lat": float(lat) if lat else None,
                         "lon": float(lon) if lon else None})
        # TR-News and LGL carry the outlet the article was published by; GWN
        # does not. It is metadata about the document, never about a toponym,
        # and it is what the `outlet` feature block reads at serve time.
        out.append({"doc_idx": n, "text": art["text"], "toponyms": tops,
                    "domain": (str(art.get("domain")).strip().lower()
                               if art.get("domain") else None)})
    return out


def heldout_doc_indices(source, articles, base_dir="raw_data", train_frac=0.7):
    """Which articles are in the held-out 30% the campaign scored on.

    tools/train.py splits the *flat entity list* of each pickle positionally at
    `train_frac`, and the entities are in document order, so the split is a
    document boundary up to the one article that straddles it. Documents are
    recovered from the pickle by grouping consecutive entities with an
    identical `doc_tensor` (the trick used for the Wave-2b sibling features),
    and each group is mapped back to its article by matching the group's
    `search_name` sequence against the article's gold phrases.

    Returns (heldout_idx, info). A document counts as held out only if *all*
    its entities are on the held-out side, so the straddling article is
    dropped rather than half-scored.
    """
    fn = (f"{base_dir}/pickled_es/es_formatted_{source}"
          f"_500_all_loc_types_fuzzy_0.pkl")
    with open(fn, "rb") as f:
        entities = pickle.load(f)
    split = round(train_frac * len(entities))

    groups = []  # [first_entity_index, [search_name, ...]]
    last = None
    for i, ex in enumerate(entities):
        key = ex["doc_tensor"].tobytes()
        if key != last:
            groups.append([i, []])
            last = key
        groups[-1][1].append(ex["search_name"])

    # Greedy in-order alignment: a group's names are a subsequence of its
    # article's gold phrases (data_formatter drops toponyms spaCy did not tag).
    mapping = []
    art_i = 0
    for first_i, names in groups:
        while art_i < len(articles):
            phrases = [t["phrase"] for t in articles[art_i]["toponyms"]
                       if t["geonameid"]]
            it = iter(phrases)
            if names and all(any(p == n for p in it) for n in names):
                mapping.append((art_i, first_i, len(names)))
                art_i += 1
                break
            art_i += 1
        else:
            raise RuntimeError(f"{source}: could not align pickle document "
                               f"groups to articles")

    heldout = [a for a, first_i, n in mapping if first_i >= split]
    straddle = [a for a, first_i, n in mapping
                if first_i < split <= first_i + n]
    info = {"n_entities": len(entities), "split_index": split,
            "n_doc_groups": len(groups), "n_articles": len(articles),
            "n_heldout_docs": len(heldout), "straddling_doc": straddle}
    return set(heldout), info


# --------------------------------------------------------------------------
# Entity extraction: the serving path, and the gold-span (oracle NER) path
# --------------------------------------------------------------------------

def _trim_span(toks):
    """Drop a leading determiner and a trailing possessive/punctuation.

    spaCy's GPE/LOC spans routinely include them -- "the United States",
    "New Mexico's", "the Thames Valley" -- and every one of those is scored as
    a boundary error against corpora that annotate the bare toponym, even when
    the pipeline resolves it correctly.

    This now delegates to the serving implementation (which also carries the
    "The Hague" guard), so a variant measured here is the thing that ships.
    """
    return trim_span_tokens(list(toks))


def doc_to_ex_labels(doc, geo_labels=GEO_LABELS, ctx_labels=CONTEXT_LABELS,
                     ner_doc=None, trim_spans=True):
    """doc_to_ex_expanded with the entity label sets exposed.

    Identical to mordecai3.geoparse.doc_to_ex_expanded at the default
    arguments (asserted in `check_parity`); the point of the parameters is to
    measure what accepting more labels buys. Pass `trim_spans=False` and
    `ctx_labels=LEGACY_CTX_LABELS` to get the pre-fix serving path, which is
    what the `ship` variant and the baseline rows of
    experiments/campaign2/end_to_end_report.md are.

    `ner_doc` lets the spans come from a *different* spaCy model while the token
    tensors still come from `doc`. The ranker consumes en_core_web_trf's 768-d
    token vectors, so any other NER model can only ever be an addition to the
    transformer, never a replacement -- this is how that configuration is
    measured.
    """
    data = []
    doc_tensor = np.mean(np.vstack([i._.tensor for i in doc]), axis=0)
    src = ner_doc if ner_doc is not None else doc
    loc_ents = [e for e in src.ents if e.label_ in ctx_labels]

    def tokens_for(lo, hi):
        toks = [t for t in doc if t.idx >= lo and t.idx + len(t) <= hi]
        if not toks:  # tokenisations disagree; fall back to any overlap
            toks = [t for t in doc if t.idx < hi and t.idx + len(t) > lo]
        return toks

    ctx_tokens = [t for e in loc_ents
                  for t in tokens_for(e.start_char, e.end_char)]
    for ent in src.ents:
        if ent.label_ not in geo_labels:
            continue
        own = tokens_for(ent.start_char, ent.end_char)
        if trim_spans:
            own = _trim_span(own)
        if not own:
            continue
        lo = own[0].idx
        hi = own[-1].idx + len(own[-1].text)
        tensor = np.mean(np.vstack([t._.tensor for t in own]), axis=0)
        own_i = {t.i for t in own}
        other_locs = [t for t in ctx_tokens if t.i not in own_i]
        if other_locs:
            locs_tensor = np.mean(np.vstack([t._.tensor for t in other_locs]),
                                  axis=0)
        else:
            locs_tensor = np.zeros(len(tensor))
        data.append({"search_name": doc.text[lo:hi] if trim_spans else ent.text,
                     "tensor": tensor,
                     "doc_tensor": doc_tensor,
                     "locs_tensor": locs_tensor,
                     "sent": own[0].sent.text,
                     "in_rel": guess_in_rel(ent),
                     "start_char": lo if trim_spans else ent.start_char,
                     "end_char": hi if trim_spans else ent.end_char,
                     "ent_label": ent.label_,
                     "source": "ner"})
    return data


def doc_to_ex_gold(doc, toponyms):
    """One example per gold toponym, mirroring tools/train.py `data_formatter`.

    The gold span's own tokens are used (not whichever spaCy entity overlaps
    it) and `search_name` is the gold phrase, exactly as the training pickles
    were built -- so this is the "oracle NER" condition the campaign's numbers
    live in. `spacy_tagged` records whether spaCy put a GPE/LOC on any token of
    the span, which is the filter that decided whether the toponym made it into
    the pickles at all.
    """
    data = []
    doc_tensor = np.mean(np.vstack([i._.tensor for i in doc]), axis=0)
    loc_ents = [e for e in doc.ents if e.label_ in ("GPE", "LOC")]
    for topo in toponyms:
        place_tokens = [i for i in doc
                        if i.idx >= topo["start"]
                        and i.idx + len(i) <= topo["end"]]
        if not place_tokens:
            continue
        tensor = np.mean(np.vstack([i._.tensor for i in place_tokens]), axis=0)
        other_locs = [i for e in loc_ents for i in e if i not in place_tokens]
        if other_locs:
            locs_tensor = np.mean(np.vstack([i._.tensor for i in other_locs]),
                                  axis=0)
        else:
            locs_tensor = np.zeros(len(tensor))
        labels = {i.ent_type_ for i in place_tokens}
        # Did spaCy find *an* entity here, and if so, is it the same span?
        exact_ent = next((e for e in doc.ents
                          if e.start_char == topo["start"]
                          and e.end_char == topo["end"]), None)
        cover_ent = next((e for e in doc.ents
                          if e.start_char <= topo["start"]
                          and e.end_char >= topo["end"]
                          and (e.start_char, e.end_char) !=
                          (topo["start"], topo["end"])), None)
        data.append({"search_name": topo["phrase"],
                     "tensor": tensor,
                     "doc_tensor": doc_tensor,
                     "locs_tensor": locs_tensor,
                     "sent": place_tokens[0].sent.text,
                     "in_rel": guess_in_rel(place_tokens),
                     "start_char": topo["start"],
                     "end_char": topo["end"],
                     "gold_geonameid": topo["geonameid"],
                     "spacy_labels": sorted(labels),
                     "spacy_tagged": bool(labels & {"GPE", "LOC"}),
                     "exact_ent_label": exact_ent.label_ if exact_ent else None,
                     "cover_ent_label": cover_ent.label_ if cover_ent else None,
                     "cover_ent_text": cover_ent.text if cover_ent else None,
                     "source": "gold"})
    return data


# --------------------------------------------------------------------------
# Candidate lookup + model scoring
# --------------------------------------------------------------------------

def score_examples(geo, all_doc_ex, max_choices=100, outlets=None):
    """ES lookup + model forward for a batch of documents.

    Mirrors steps 2-4 of Geoparser._geoparse_docs, but keeps the candidate
    lists so the decomposition can ask whether the gold id was ever retrieved.
    Returns (all_es_data, picks, timing), where `picks[d][e]` is the chosen
    candidate dict or None ("no answer"), decoded exactly as
    Geoparser._resolve_results decodes it.

    `outlets` is one outlet domain per document (None where unknown), the same
    argument `Geoparser.geoparse_batch` takes. It is inert unless the
    checkpoint carries the `outlet` feature block.
    """
    t0 = time.perf_counter()
    all_es = add_es_data_batch(all_doc_ex, geo.geonames, max_results=max_choices,
                               extra_features=bool(geo.extra_feature_keys),
                               outlet_homes=geo._outlet_homes_for(
                                   len(all_doc_ex), outlets))
    t_es = time.perf_counter() - t0

    pooled = [e for doc in all_es for e in doc]
    picks = [[None] * len(doc) for doc in all_es]
    t_model = 0.0
    if pooled:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        dataset = ProductionData(pooled, max_choices=max_choices,
                                 oov_bucket_fix=geo.oov_bucket_fix,
                                 feature_blocks=geo.feature_blocks)
        loader = DataLoader(dataset=dataset, batch_size=64, shuffle=False)
        with torch.no_grad():
            geo.model.eval()
            out = [geo.model({k: v.to(geo.model.device) for k, v in b.items()})
                   for b in loader]
            preds = torch.cat(out, dim=0).cpu()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_model = time.perf_counter() - t0

        i = 0
        for d, doc in enumerate(all_es):
            for e in range(len(doc)):
                picks[d][e] = _decode(doc[e], preds[i])
                i += 1
    return all_es, picks, {"es": t_es, "model": t_model}


def _decode(ent, pred):
    """The candidate Geoparser._resolve_results would return, or None.

    The reserved-row convention (mordecai3.geoparse): the reserved last row of
    the window decides *whether* to answer, the candidate rows decide *which*
    place, and the appended gazetteer NULL row winning is also a refusal.
    Numerically this is the same decode as before the convention was written
    down -- the row the sentinel overwrites could never win the argmax without
    the reserved-row test firing first -- and the `ship`/`serving` rows of
    experiments/campaign2/end_to_end_report.md reproduce unchanged.
    """
    if pred[-1] == pred.max():
        return None
    n = candidate_row_count(len(ent["es_choices"]), len(pred))
    scores = np.array([pred[i].item() for i in range(n)])
    if len(scores) == 0:
        return None
    best = int(np.argmax(scores))
    if best == len(ent["es_choices"]) - 1:  # the NULL candidate is always last
        return None
    choice = dict(ent["es_choices"][best])
    choice["score"] = scores[best]
    return choice


# --------------------------------------------------------------------------
# Alignment and scoring
# --------------------------------------------------------------------------

def align(pred_spans, gold_spans):
    """Map predicted spans to gold spans, exactly and by character overlap.

    Returns (exact, overlap): dicts pred_index -> gold_index. `overlap` is a
    one-to-one greedy assignment, largest overlap first, so two predictions
    cannot both claim the same gold toponym.
    """
    exact = {}
    by_span = {}
    for gi, g in enumerate(gold_spans):
        by_span.setdefault((g["start"], g["end"]), gi)
    for pi, p in enumerate(pred_spans):
        gi = by_span.get((p["start_char"], p["end_char"]))
        if gi is not None:
            exact[pi] = gi

    pairs = []
    for pi, p in enumerate(pred_spans):
        for gi, g in enumerate(gold_spans):
            ov = min(p["end_char"], g["end"]) - max(p["start_char"], g["start"])
            if ov > 0:
                pairs.append((ov, pi, gi))
    pairs.sort(reverse=True)
    overlap, used_p, used_g = {}, set(), set()
    for ov, pi, gi in pairs:
        if pi in used_p or gi in used_g:
            continue
        overlap[pi] = gi
        used_p.add(pi)
        used_g.add(gi)
    return exact, overlap


def _same_place(pred, gold_topo, cutoff=161):
    """(exact geonameid match, within `cutoff` km of the gold coordinates)."""
    if pred is None:
        return False, False
    em = str(pred.get("geonameid")) == str(gold_topo["geonameid"])
    near = False
    if gold_topo["lat"] is not None and pred.get("lat") is not None:
        try:
            d = hs.haversine((gold_topo["lat"], gold_topo["lon"]),
                             (float(pred["lat"]), float(pred["lon"])))
            near = d <= cutoff
        except (TypeError, ValueError):
            near = em
    else:
        near = em
    return em, near


def _gold_class(labels, phrase=None):
    """How the gold toponym looks to spaCy, for the miss taxonomy.

    NORP is called out separately: `doc_to_ex_expanded` deliberately refuses to
    geoparse demonyms ("Turkish", "Chinese"), while TR/LGL/GWN annotate many of
    them with a country geonameid. That is a policy disagreement, not a
    detection failure, and it has to be counted apart from the rest.
    """
    if labels is None:
        # No token fell entirely inside the gold span, so the annotation's
        # character offsets do not line up with spaCy's tokens at all.
        return "no aligned token"
    labels = set(labels)
    if labels & {"GPE", "LOC", "EVENT_LOC"}:
        return "GPE/LOC"
    if "NORP" in labels:
        return "NORP (demonym)"
    if "FAC" in labels:
        return "FAC"
    if "ORG" in labels:
        return "ORG"
    if labels - {""}:
        return "other label: " + "/".join(sorted(l for l in labels if l))
    return "untagged"


def demonym_gold_spans(articles, gold_ex):
    """The gold toponyms that are demonyms, by (doc_idx, start, end).

    Decision D2 takes demonyms out of the task entirely, so these rows leave
    every end-to-end denominator: they are not counted as recall failures, and
    a predicted span landing on one is not counted as a hallucination either --
    exactly how the corpora's unlinked gold rows are already treated.

    A gold toponym is a demonym if EITHER

      * spaCy sees the span as NORP and nothing more place-like (no
        GPE/LOC/EVENT_LOC on any of its tokens) -- the only signal TR-News and
        LGL offer, since neither types its toponyms; or
      * GeoWebNews, which does type them, calls it `Non_Literal_Modifier` --
        its own name for this category ("the Turkish president"). 93 of GWN's
        153 such rows are spans spaCy tags GPE, so the label test alone misses
        them.

    Both are fixed properties of the gold set, computed once per corpus from
    the en_core_web_trf pass, so every variant is scored on the same
    denominator. Pooled over the 260 held-out documents of TR/LGL/GWN this
    removes 308 of 2,392 linked gold toponyms, leaving 2,084 -- the same
    denominator as experiments/campaign2/ner_retrain_scoping_report.md §2.
    """
    labels = {(a["doc_idx"], e["start_char"], e["end_char"]): e["spacy_labels"]
              for a, doc in zip(articles, gold_ex) for e in doc}
    out = set()
    for a in articles:
        for t in a["toponyms"]:
            key = (a["doc_idx"], t["start"], t["end"])
            lab = labels.get(key)
            if (lab is not None and _gold_class(lab) == "NORP (demonym)") or \
                    (t.get("gtype") or "") == "Non_Literal_Modifier":
                out.add(key)
    return out


def evaluate_docs(articles, all_es, picks, gold_es=None, gold_picks=None,
                  gold_labels=None, demonyms=None):
    """Score one corpus: detection, end-to-end, decomposition, spurious spans.

    `all_es`/`picks` are the NER-path results per document; `gold_es`/
    `gold_picks` the oracle-span results for the same documents (optional).
    `gold_labels` maps (doc_idx, start, end) to spaCy's labels for that gold
    span, so every variant can be broken down by the same taxonomy.
    `demonyms` is the set of gold spans decision D2 removes from the task; pass
    None to score on the pre-D2 denominator (what
    experiments/campaign2/end_to_end_report.md reports).
    """
    demonyms = demonyms or set()
    res = {
        "n_docs": len(articles),
        "n_gold": 0, "n_gold_rows": 0, "n_pred": 0, "n_demonym_excluded": 0,
        "n_out": 0, "n_out_correct": 0, "n_out_correct_overlap": 0,
        "det_exact_tp": 0, "det_overlap_tp": 0,
        "e2e_em": 0, "e2e_161": 0,
        "e2e_overlap_em": 0, "e2e_overlap_161": 0,
        "decomp": Counter(),
        "miss_labels": Counter(),
        "miss_kind": Counter(),
        "miss_examples": [],
        "boundary_examples": [],
        # Every gold whose span was found exactly but whose id never came back
        # from Elasticsearch. Dumped per gold (not just counted) because the
        # residual after e57's alias fix has to be attributed by cause.
        "retrieval_examples": [],
        "spurious": Counter(),
        "spurious_examples": [],
        "oracle": Counter(),
        "oracle_examples": [],
        # Gold-by-gold attribution of the distance to the oracle-span ceiling:
        # of the toponyms the SAME ranker gets right when handed the gold span,
        # which ones does the pipeline lose, and to what?
        "gap": Counter(),
        # One row per gold toponym in the denominator: which decomposition
        # bucket it landed in, keyed by span. Two runs of this harness can then
        # be diffed gold by gold instead of only in aggregate, which is what
        # turns "+1.44 EM" into "+31 golds gained, 1 lost".
        "gold_outcomes": [],
        "by_class": defaultdict(Counter),
        "by_gtype": defaultdict(Counter),
    }
    # spaCy's own view of each gold span, from the oracle-span pass.
    gold_labels = dict(gold_labels or {})
    if gold_es is not None:
        for art, es_doc in zip(articles, gold_es):
            for ent in es_doc:
                gold_labels[(art["doc_idx"], ent["start_char"],
                             ent["end_char"])] = ent
    # Which gold toponyms the oracle-span pass resolves correctly, keyed by
    # span. This is the ceiling row, gold by gold rather than in aggregate.
    oracle_ok = {}
    if gold_es is not None and gold_picks is not None:
        for art, es_doc, pick_doc in zip(articles, gold_es, gold_picks):
            for ent, pick in zip(es_doc, pick_doc):
                if not ent.get("gold_geonameid"):
                    continue
                oracle_ok[(art["doc_idx"], ent["start_char"],
                           ent["end_char"])] = bool(
                    pick is not None and str(pick.get("geonameid")) ==
                    str(ent["gold_geonameid"]))

    for art, es_doc, pick_doc in zip(articles, all_es, picks):
        def is_demonym(t):
            return (art["doc_idx"], t["start"], t["end"]) in demonyms

        gold = [t for t in art["toponyms"]
                if t["geonameid"] and not is_demonym(t)]
        # Demonym rows join the unlinked ones: out of the denominator, and a
        # prediction that lands on one is reported separately rather than as a
        # hallucination.
        other_gold = [t for t in art["toponyms"]
                      if not t["geonameid"] or is_demonym(t)]
        res["n_demonym_excluded"] += sum(1 for t in art["toponyms"]
                                         if t["geonameid"] and is_demonym(t))
        res["n_gold"] += len(gold)
        res["n_gold_rows"] += len(art["toponyms"])
        res["n_pred"] += len(es_doc)
        exact, overlap = align(es_doc, gold)
        res["det_exact_tp"] += len(exact)
        res["det_overlap_tp"] += len(overlap)
        gold_to_pred_exact = {gi: pi for pi, gi in exact.items()}
        gold_to_pred_ov = {gi: pi for pi, gi in overlap.items()}

        for gi, g in enumerate(gold):
            pi = gold_to_pred_exact.get(gi)
            pi_ov = gold_to_pred_ov.get(gi)
            ginfo = gold_labels.get((art["doc_idx"], g["start"], g["end"]))
            cls = _gold_class(ginfo["spacy_labels"] if ginfo else None)
            bucket = res["by_class"][cls]
            gbucket = res["by_gtype"][g.get("gtype") or "(untyped)"]
            for b in (bucket, gbucket):
                b["n"] += 1
                b["det_exact"] += pi is not None
                b["det_overlap"] += pi_ov is not None
            if pi is not None:
                pick = pick_doc[pi]
                em, near = _same_place(pick, g)
                res["e2e_em"] += em
                res["e2e_161"] += near
                bucket["em"] += em
                gbucket["em"] += em
                retrieved = any(str(c.get("geonameid")) == str(g["geonameid"])
                                for c in es_doc[pi]["es_choices"])
                if em:
                    outcome = "correct"
                    res["decomp"]["correct"] += 1
                elif not retrieved:
                    outcome = "retrieval_miss"
                    res["decomp"]["retrieval_miss"] += 1
                    res["retrieval_examples"].append(
                        {"query": es_doc[pi]["search_name"],
                         "gold_phrase": g["phrase"],
                         "gold_id": str(g["geonameid"]),
                         "doc": art["doc_idx"],
                         "start": g["start"],
                         "n_choices": len(es_doc[pi]["es_choices"]),
                         "class": cls,
                         "cover": (f"{ginfo['cover_ent_label']}:"
                                   f"{ginfo['cover_ent_text']}"
                                   if ginfo and ginfo.get("cover_ent_label")
                                   else None),
                         "picked": (None if pick is None else
                                    f"{pick.get('name')} "
                                    f"({pick.get('country_code3')}/"
                                    f"{pick.get('feature_code')})")})
                elif pick is None:
                    outcome = "null_answer"
                    res["decomp"]["null_answer"] += 1
                else:
                    outcome = "ranker_error"
                    res["decomp"]["ranker_error"] += 1
            elif pi_ov is not None:
                pick = pick_doc[pi_ov]
                em, near = _same_place(pick, g)
                outcome = "boundary_ok" if em else "boundary_wrong"
                res["decomp"][outcome] += 1
                res["boundary_examples"].append(
                    {"gold": g["phrase"], "pred": es_doc[pi_ov]["search_name"],
                     "label": es_doc[pi_ov].get("ent_label"),
                     "resolved_ok": bool(em), "class": cls})
            else:
                outcome = "ner_miss"
                res["decomp"]["ner_miss"] += 1
                res["miss_labels"][cls] += 1
                # A miss is one of three very different things: spaCy found the
                # same span under a label the pipeline discards (a config fix),
                # spaCy swallowed the toponym inside a bigger entity such as
                # "Paris Police Department" (needs nested extraction), or spaCy
                # found nothing here at all (a real NER failure).
                if ginfo and ginfo["exact_ent_label"]:
                    res["miss_kind"][f"same span, label "
                                     f"{ginfo['exact_ent_label']}"] += 1
                elif ginfo and ginfo["cover_ent_label"]:
                    res["miss_kind"][f"nested in {ginfo['cover_ent_label']}"] += 1
                else:
                    res["miss_kind"]["no entity here"] += 1
                res["miss_examples"].append(
                    {"phrase": g["phrase"], "start": g["start"],
                     "doc": art["doc_idx"], "class": cls,
                     "gtype": g.get("gtype"),
                     "exact_ent_label": ginfo["exact_ent_label"] if ginfo else None,
                     "cover": (f"{ginfo['cover_ent_label']}:{ginfo['cover_ent_text']}"
                               if ginfo and ginfo["cover_ent_label"] else None),
                     "context": art["text"][max(0, g["start"] - 40):
                                            g["end"] + 40].replace("\n", " ")})
            res["gold_outcomes"].append(
                {"doc": art["doc_idx"], "start": g["start"], "end": g["end"],
                 "phrase": g["phrase"], "gold_id": str(g["geonameid"]),
                 "outcome": outcome})

            if pi_ov is not None:
                pick = pick_doc[pi_ov]
                em, near = _same_place(pick, g)
                res["e2e_overlap_em"] += em
                res["e2e_overlap_161"] += near

            # Where the distance to the oracle-span ceiling goes, gold by gold.
            ok = oracle_ok.get((art["doc_idx"], g["start"], g["end"]))
            if ok is not None:
                got = bool(pi is not None and
                           _same_place(pick_doc[pi], g)[0])
                res["gap"]["n"] += 1
                res["gap"]["oracle_correct"] += ok
                res["gap"]["pipeline_correct"] += got
                if ok and not got:
                    res["gap"]["lost_to_detection" if pi is None
                               else "lost_to_resolution"] += 1
                elif got and not ok:
                    res["gap"]["won_without_oracle"] += 1

        # Deployment-facing precision: of the locations the pipeline actually
        # emits (a span it resolved to a geonames id), how many are a real gold
        # toponym resolved to the right place?
        for pi, p in enumerate(es_doc):
            if pick_doc[pi] is None:
                continue
            res["n_out"] += 1
            gi = exact.get(pi)
            if gi is not None and _same_place(pick_doc[pi], gold[gi])[0]:
                res["n_out_correct"] += 1
            gi_ov = overlap.get(pi)
            if gi_ov is not None and _same_place(pick_doc[pi], gold[gi_ov])[0]:
                res["n_out_correct_overlap"] += 1

        # Spurious predictions: no overlap with any gold toponym row at all.
        for pi, p in enumerate(es_doc):
            if pi in overlap:
                continue
            hit_unlinked = any(min(p["end_char"], t["end"]) -
                               max(p["start_char"], t["start"]) > 0
                               for t in other_gold)
            hit_linked = any(min(p["end_char"], t["end"]) -
                             max(p["start_char"], t["start"]) > 0 for t in gold)
            kind = ("unlinked_gold" if hit_unlinked else
                    "double_count" if hit_linked else "no_gold")
            pick = pick_doc[pi]
            res["spurious"][kind + ("_resolved" if pick else "_null")] += 1
            if kind == "no_gold":
                res["spurious_examples"].append(
                    {"text": p["search_name"], "label": p.get("ent_label"),
                     "resolved": None if pick is None else
                     f"{pick.get('name')} ({pick.get('country_code3')}/"
                     f"{pick.get('feature_code')})"})

    # Oracle-span (resolution-only) numbers on the same documents.
    if gold_es is not None:
        for art, es_doc, pick_doc in zip(articles, gold_es, gold_picks):
            for ent, pick in zip(es_doc, pick_doc):
                if not ent.get("gold_geonameid"):
                    continue
                if (art["doc_idx"], ent["start_char"],
                        ent["end_char"]) in demonyms:
                    continue        # D2: demonyms are not part of the task
                topo = {"geonameid": ent["gold_geonameid"], "lat": None,
                        "lon": None}
                gold_row = next((t for t in art["toponyms"]
                                 if t["start"] == ent["start_char"]
                                 and t["end"] == ent["end_char"]), None)
                if gold_row:
                    topo = gold_row
                em, near = _same_place(pick, topo)
                retrieved = any(str(c.get("geonameid")) == str(topo["geonameid"])
                                for c in ent["es_choices"])
                tagged = ent["spacy_tagged"]
                res["oracle"]["n"] += 1
                res["oracle"]["em"] += em
                res["oracle"]["at161"] += near
                res["oracle"]["retrieved"] += retrieved
                if tagged:
                    res["oracle"]["n_tagged"] += 1
                    res["oracle"]["em_tagged"] += em
                    res["oracle"]["at161_tagged"] += near
                    if retrieved:
                        res["oracle"]["n_tagged_retrieved"] += 1
                        res["oracle"]["em_tagged_retrieved"] += em
                        res["oracle"]["at161_tagged_retrieved"] += near
                if not tagged:
                    res["oracle_examples"].append(
                        {"phrase": ent["search_name"],
                         "labels": ent["spacy_labels"],
                         "resolved_ok": bool(em)})
    return res


def summarize(res):
    """Headline rates for one corpus/variant."""
    n_gold, n_pred = res["n_gold"], max(res["n_pred"], 1)
    d = res["decomp"]
    o = res["oracle"]

    def rate(a, b):
        return round(a / b, 4) if b else None

    out = {
        "n_docs": res["n_docs"], "n_gold": n_gold, "n_pred": res["n_pred"],
        "n_demonym_excluded": res["n_demonym_excluded"],
        "n_out": res["n_out"],
        "out_precision": rate(res["n_out_correct"], res["n_out"]),
        "out_precision_overlap": rate(res["n_out_correct_overlap"],
                                      res["n_out"]),
        "det_exact_p": rate(res["det_exact_tp"], n_pred),
        "det_exact_r": rate(res["det_exact_tp"], n_gold),
        "det_overlap_p": rate(res["det_overlap_tp"], n_pred),
        "det_overlap_r": rate(res["det_overlap_tp"], n_gold),
        "e2e_em": rate(res["e2e_em"], n_gold),
        "e2e_161": rate(res["e2e_161"], n_gold),
        "e2e_overlap_em": rate(res["e2e_overlap_em"], n_gold),
        "e2e_overlap_161": rate(res["e2e_overlap_161"], n_gold),
        "decomp": {k: rate(v, n_gold) for k, v in sorted(d.items())},
        "decomp_counts": dict(sorted(d.items())),
        "spurious": dict(sorted(res["spurious"].items())),
        "gap": dict(sorted(res["gap"].items())),
        "miss_by_class": dict(sorted(res["miss_labels"].items(),
                                     key=lambda kv: -kv[1])),
        "miss_by_kind": dict(sorted(res["miss_kind"].items(),
                                    key=lambda kv: -kv[1])),
        "by_class": {k: {"n": v["n"],
                         "det_exact_r": rate(v["det_exact"], v["n"]),
                         "det_overlap_r": rate(v["det_overlap"], v["n"]),
                         "e2e_em": rate(v["em"], v["n"])}
                     for k, v in sorted(res["by_class"].items(),
                                        key=lambda kv: -kv[1]["n"])},
        "by_gtype": {k: {"n": v["n"],
                         "det_exact_r": rate(v["det_exact"], v["n"]),
                         "det_overlap_r": rate(v["det_overlap"], v["n"]),
                         "e2e_em": rate(v["em"], v["n"])}
                     for k, v in sorted(res["by_gtype"].items(),
                                        key=lambda kv: -kv[1]["n"])},
    }
    for k in ("det_exact_f1", "det_overlap_f1"):
        p, r = out[k.replace("f1", "p")], out[k.replace("f1", "r")]
        out[k] = round(2 * p * r / (p + r), 4) if p and r else None
    if o:
        out["oracle"] = {
            "n": o["n"],
            "em_all_gold": rate(o["em"], o["n"]),
            "at161_all_gold": rate(o["at161"], o["n"]),
            "retrieval_recall": rate(o["retrieved"], o["n"]),
            "n_spacy_tagged": o["n_tagged"],
            "em_spacy_tagged": rate(o["em_tagged"], o["n_tagged"]),
            "retrieval_recall_tagged": rate(o["n_tagged_retrieved"],
                                            o["n_tagged"]),
            "em_campaign_parity": rate(o["em_tagged_retrieved"],
                                       o["n_tagged_retrieved"]),
            "at161_campaign_parity": rate(o["at161_tagged_retrieved"],
                                          o["n_tagged_retrieved"]),
        }
    return out


# --------------------------------------------------------------------------
# NER variants
# --------------------------------------------------------------------------

def truecase_text(text):
    """Down-case ALL-CAPS runs, which spaCy's NER handles badly.

    Only lines/sentences that are essentially all upper case are touched, and
    offsets are preserved character for character so gold spans still line up.
    """
    out = []
    for line in text.split("\n"):
        letters = [c for c in line if c.isalpha()]
        upper = sum(c.isupper() for c in letters)
        if len(letters) >= 8 and upper / len(letters) > 0.9:
            out.append(_titlecase_preserving(line))
        else:
            out.append(line)
    return "\n".join(out)


def _titlecase_preserving(line):
    small = {"of", "the", "in", "and", "for", "on", "at", "to", "a", "an",
             "as", "by", "from", "with"}
    parts = re.split(r"(\W+)", line)
    out = []
    for i, p in enumerate(parts):
        if not p or not p[0].isalpha():
            out.append(p)
        elif len(p) <= 3 and p.lower() in small and i > 0:
            out.append(p.lower())
        else:
            out.append(p[0].upper() + p[1:].lower())
    return "".join(out)


GAZ_STOP = {"the", "a", "an", "of", "and", "in", "on", "at", "for", "to",
            "he", "she", "it", "they", "we", "you", "i", "is", "was", "said",
            "mr", "mrs", "ms", "dr"}


# Entity labels the pipeline never geoparses but that routinely swallow a
# toponym: "Paris Police Department", "University of Pennsylvania", "Montana
# Department of Corrections".
NESTED_LABELS = ("ORG", "FAC", "EVENT", "WORK_OF_ART", "LAW", "PRODUCT")


def _gaz_span_ok(span):
    return all(t.text[:1].isupper() and t.text.isalpha()
               and t.text.lower() not in GAZ_STOP for t in span)


def gazetteer_second_pass(doc, existing, geonames, outside=True, nested=True,
                          min_tokens=2, max_tokens=4, ner_doc=None):
    """Add spans the NER path missed but the gazetteer knows exactly.

    Two bounded rules, both requiring a geonames entry whose own `name` equals
    the candidate string:

    `outside`: runs of `min_tokens`..`max_tokens` capitalised alphabetic tokens
        that no entity covers. Multi-token only -- single capitalised words are
        where an unbounded gazetteer match turns into a precision disaster.
    `nested`: inside an ORG/FAC/... entity the pipeline discards, sub-spans of
        1-3 tokens, leftmost-longest, restricted to populated places and admin
        units (feature class A/P). This is the rule aimed at the dominant miss
        class, and it is the one that can cost precision.
    """
    taken = set()
    for ex in existing:
        taken.update(range(ex["start_char"], ex["end_char"]))

    cands = {}   # string -> [(start_token, end_token, require_ap)]
    if outside:
        n = len(doc)
        for i in range(n):
            for length in range(min_tokens, max_tokens + 1):
                if i + length > n:
                    break
                span = doc[i:i + length]
                if not _gaz_span_ok(span):
                    break
                if any(c in taken for c in range(span[0].idx,
                                                 span[-1].idx + len(span[-1]))):
                    continue
                cands.setdefault(span.text, []).append((i, i + length, False))
    if nested:
        src = ner_doc if ner_doc is not None else doc
        ranges = [(e.start_char, e.end_char) for e in src.ents
                  if e.label_ in NESTED_LABELS]
        for lo, hi in ranges:
            inside = [t.i for t in doc if t.idx >= lo and t.idx + len(t) <= hi]
            for k, i in enumerate(inside):
                for length in (3, 2, 1):
                    if k + length > len(inside):
                        continue
                    if inside[k + length - 1] != i + length - 1:
                        continue
                    span = doc[i:i + length]
                    if not _gaz_span_ok(span):
                        continue
                    if any(c in taken for c in range(span[0].idx,
                                                     span[-1].idx + len(span[-1]))):
                        continue
                    cands.setdefault(span.text, []).append((i, i + length, True))

    if not cands:
        return []
    names = list(cands)
    responses = geonames.search_by_names([(nm, 5, 0, False, None)
                                          for nm in names])
    ok_any, ok_ap = set(), set()
    for nm, res in zip(names, responses):
        for s in hit_sources(res):
            if str(s.get("name", "")).lower() != nm.lower() and \
               str(s.get("asciiname", "")).lower() != nm.lower():
                continue
            ok_any.add(nm)
            if s.get("feature_class") in ("A", "P"):
                ok_ap.add(nm)

    # Leftmost-longest, so "Santa Barbara" beats "Santa" inside the same entity.
    placed = sorted(((a, b, req, nm) for nm, spans in cands.items()
                     for (a, b, req) in spans),
                    key=lambda x: (x[0], -(x[1] - x[0])))
    added = []
    doc_tensor = np.mean(np.vstack([i._.tensor for i in doc]), axis=0)
    for a, b, require_ap, nm in placed:
        if nm not in (ok_ap if require_ap else ok_any):
            continue
        span = doc[a:b]
        lo, hi = span[0].idx, span[-1].idx + len(span[-1].text)
        if any(c in taken for c in range(lo, hi)):
            continue
        tensor = np.mean(np.vstack([t._.tensor for t in span]), axis=0)
        added.append({"search_name": span.text,
                      "tensor": tensor,
                      "doc_tensor": doc_tensor,
                      "locs_tensor": np.zeros(len(tensor)),
                      "sent": span.sent.text,
                      "in_rel": "",
                      "start_char": lo,
                      "end_char": hi,
                      "ent_label": "GAZ",
                      "source": "gaz"})
        taken.update(range(lo, hi))
    return added


def _variant(ner="en_core_web_trf", geo_labels=GEO_LABELS,
             ctx_labels=LEGACY_CTX_LABELS, transform=None, gaz=None,
             trim=False, span_head=None):
    """One NER configuration to score. gaz is None, "outside", "nested" or "both".

    The defaults are the *pre-fix* serving path, so every row of
    experiments/campaign2/end_to_end_report.md keeps reproducing after
    mordecai3/geoparse.py adopted trimming and the GPE/LOC context set. The
    `serving*` variants below are the current serving path.

    `span_head` names a packaged place-span head ("gold" / "all"), which
    replaces the label filter, the trimmer and the gazetteer pass all at once;
    it is what `Geoparser(span_detector=...)` runs.
    """
    return {"ner": ner, "geo_labels": geo_labels, "ctx_labels": ctx_labels,
            "transform": transform, "gaz": gaz, "trim": trim,
            "span_head": span_head}


def _serving(geo_labels=GEO_LABELS, gaz=None, **kw):
    """A variant that matches what mordecai3.Geoparser does today."""
    return _variant(geo_labels=geo_labels, ctx_labels=CONTEXT_LABELS,
                    trim=True, gaz=gaz, **kw)


def doc_to_ex_head(doc, tagger, ctx_labels=CONTEXT_LABELS):
    """The serving path under `Geoparser(span_detector=...)`.

    Delegates to the shipped `SpanTagger.doc_to_ex` -- the same call
    `Geoparser._geoparse_docs` makes -- and only adds the two bookkeeping keys
    the rest of this harness reads off an entity.
    """
    ex = tagger.doc_to_ex(doc, context_labels=ctx_labels)
    for e in ex:
        e["ent_label"] = "HEAD"
        e["source"] = "head"
    return ex


VARIANTS = {
    # --- what ships now: trimmed spans, GPE/LOC context tensor -------------
    "serving":      _serving(),
    "serving_norp": _serving(geo_labels=GEO_LABELS + ("NORP",)),
    "serving_gaz":  _serving(gaz="nested"),
    "serving_best": _serving(geo_labels=GEO_LABELS + ("NORP",), gaz="nested"),
    "serving_nofac": _serving(geo_labels=("GPE", "LOC", "EVENT_LOC")),
    # --- the place-span head (e55/e56): replaces the label filter, the span
    # trimmer and the nested-gazetteer pass in one call ---------------------
    "head_gold":   _serving(span_head="gold"),
    "head_all":    _serving(span_head="all"),
    # --- the pre-fix path, kept so the report's baseline rows reproduce ----
    "ship":        _variant(),
    "no_fac":      _variant(geo_labels=("GPE", "LOC", "EVENT_LOC")),
    "labels_org":  _variant(geo_labels=GEO_LABELS + ("ORG",)),
    "labels_norp": _variant(geo_labels=GEO_LABELS + ("NORP",)),
    "labels_all":  _variant(geo_labels=GEO_LABELS +
                            ("ORG", "NORP", "EVENT", "WORK_OF_ART", "LAW")),
    "truecase":    _variant(transform="truecase"),
    "gaz_out":     _variant(gaz="outside"),
    "gaz_nested":  _variant(gaz="nested"),
    "gaz_both":    _variant(gaz="both"),
    "lg":          _variant(ner="en_core_web_lg"),
    "lg_gaz":      _variant(ner="en_core_web_lg", gaz="both"),
    "trim":        _variant(trim=True),
    "trim_nofac":  _variant(trim=True, geo_labels=("GPE", "LOC", "EVENT_LOC")),
    "trim_gaz":    _variant(trim=True, gaz="both"),
    "trim_norp":   _variant(trim=True, geo_labels=GEO_LABELS + ("NORP",)),
    "combo":       _variant(trim=True, gaz="both",
                            geo_labels=("GPE", "LOC", "EVENT_LOC")),
    # The three levers that paid, stacked: trimmed spans, demonyms accepted,
    # and the nested gazetteer pass.
    "combo_best":  _variant(trim=True, gaz="nested",
                            geo_labels=GEO_LABELS + ("NORP",)),
}


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def load_nlp_for(model_name, tensors=True):
    """`tensors=False` for a model used only for spans (no trf_data to read)."""
    import spacy
    from mordecai3.mordecai_utilities import spacy_doc_setup
    spacy_doc_setup()
    spacy.prefer_gpu()
    nlp = spacy.load(model_name)
    if tensors:
        nlp.add_pipe("token_tensors")
    return nlp


def check_parity(doc):
    """doc_to_ex_labels at its defaults == the shipped doc_to_ex_expanded.

    Both sides now trim spans and pool GPE/LOC into the context tensor, so
    this asserts the harness is measuring the code that ships. The pre-fix
    path is still reachable, and still what the `ship`-family variants use.
    """
    a = doc_to_ex_expanded(doc)
    b = doc_to_ex_labels(doc)
    assert len(a) == len(b), (len(a), len(b))
    for x, y in zip(a, b):
        assert x["search_name"] == y["search_name"]
        assert x["start_char"] == y["start_char"]
        assert x["end_char"] == y["end_char"]
        assert x["in_rel"] == y["in_rel"]
        assert np.allclose(x["tensor"], y["tensor"])
        assert np.allclose(x["locs_tensor"], y["locs_tensor"])


@app.command()
def evaluate(model_path: str = "experiments/e29_swa_ep15/seed42.pt",
             base_dir: str = "raw_data",
             sources: str = "tr,lgl,gwn",
             variants: str = "serving",
             all_docs: bool = False,
             max_choices: int = 100,
             limit_docs: int = 0,
             oracle: bool = True,
             feature_blocks: str = "prom,name,cue,sib,geo,shape",
             outlet_sources: str = "",
             span_head_paths: str = "",
             normalize_place_abbrevs: bool = True,
             out: str = ""):
    """Run the end-to-end evaluation.

    By default this scores the held-out 30% of each corpus -- the same
    documents the campaign's resolution numbers come from -- and also reports
    the oracle-span (resolution-only) numbers on exactly those documents.

    `--span-head-paths` overrides which checkpoint a `head_*` variant loads,
    as `gold=/path/to/head.pt[,all=...]`.

    `--feature-blocks` must match the checkpoint (add `,outlet` for an e54
    checkpoint). `--outlet-sources` names the corpora whose documents hand
    their `<domain>` to the geoparser -- "lgl", or "lgl,tr", or "" (the
    default) for the pessimistic serving condition where no outlet is known.
    Only LGL and TR-News have the metadata at all.

    `--no-normalize-place-abbrevs` turns off e52 rule R1 (the abbreviation
    expansion in `GeonamesService.build_name_search`), which is what reproduces
    every pre-e57 row of this harness byte for byte.
    """
    src_list = [s.strip() for s in sources.split(",") if s.strip()]
    var_list = [v.strip() for v in variants.split(",") if v.strip()]
    outlet_src = {s.strip() for s in outlet_sources.split(",") if s.strip()}
    if outlet_src - set(SOURCES):
        raise ValueError(f"unknown --outlet-sources {sorted(outlet_src - set(SOURCES))}")

    geo = Geoparser(model_path=model_path,
                    feature_blocks=feature_blocks,
                    oov_bucket_fix=True,
                    normalize_place_abbrevs=normalize_place_abbrevs,
                    model_options={"return_logits": True, "mask_padding": True,
                                   "modern_mlp": True})
    if outlet_src and not geo.uses_outlet:
        raise ValueError("--outlet-sources needs a checkpoint with the "
                         "'outlet' feature block; add it to --feature-blocks")
    # One tagger per head named by a variant, on the ranker's device -- exactly
    # what Geoparser(span_detector=...) loads.
    # `--span-head-paths gold=/path/to.pt` swaps the checkpoint a head variant
    # loads without inventing a variant, which is how a leave-one-corpus-out
    # head (experiments/e56_span_head_serving/loco.py) is scored end to end.
    head_override = dict(kv.split("=", 1)
                         for kv in span_head_paths.split(",") if "=" in kv)
    span_taggers = {}
    for var in var_list:
        head = VARIANTS[var]["span_head"]
        if head and head not in span_taggers:
            span_taggers[head] = load_span_tagger(
                head_override.get(head, head), device=geo.model.device)
    results = {"model_path": model_path, "max_choices": max_choices,
               "all_docs": all_docs, "feature_blocks": feature_blocks,
               "outlet_sources": sorted(outlet_src),
               "normalize_place_abbrevs": bool(normalize_place_abbrevs),
               "span_heads": {h: [head_override.get(h, h), t.threshold]
                              for h, t in span_taggers.items()},
               "corpora": {}}
    nlp_cache = {}
    parity_checked = False

    for src in src_list:
        articles = read_corpus(src, base_dir)
        if all_docs:
            keep = list(range(len(articles)))
            split_info = {"note": "all documents"}
        else:
            heldout, split_info = heldout_doc_indices(src, articles, base_dir)
            keep = sorted(heldout)
        if limit_docs:
            keep = keep[:limit_docs]
        docs_meta = [articles[i] for i in keep]
        # One outlet per document, or None everywhere for a corpus we are not
        # supplying outlets for. `None` is a real value here, not "unset": it
        # is the null encoding a no-outlet document carries in training.
        outlets = ([a.get("domain") for a in docs_meta]
                   if src in outlet_src else None)
        outlet_info = None
        if outlets is not None:
            n_known = sum(1 for o in outlets
                          if lookup_outlet_home(geo.outlet_homes, o))
            outlet_info = {"n_docs": len(outlets),
                           "n_with_domain": sum(1 for o in outlets if o),
                           "n_resolved": n_known}
            logger.info(f"{src}: outlet supplied for "
                        f"{outlet_info['n_with_domain']}/{len(outlets)} "
                        f"documents, {n_known} of them in the home table")
        logger.info(f"{src}: {len(docs_meta)} documents, "
                    f"{sum(len([t for t in a['toponyms'] if t['geonameid']]) for a in docs_meta)} gold toponyms")
        results["corpora"].setdefault(src, {"split": split_info,
                                            "variants": {}})
        if outlet_info is not None:
            results["corpora"][src]["outlets"] = outlet_info

        # The gold-span pass, once per corpus, always with the shipped spaCy
        # model on untransformed text. It supplies spaCy's label for every gold
        # span -- which the miss taxonomy and the D2 demonym exclusion are both
        # built on -- and, with `--oracle`, the resolution-only numbers the
        # campaign's metric lives in.
        gold_es = gold_picks = gold_labels = None
        if "en_core_web_trf" not in nlp_cache:
            nlp_cache["en_core_web_trf"] = load_nlp_for("en_core_web_trf")
        gold_docs = list(nlp_cache["en_core_web_trf"].pipe(
            [a["text"] for a in docs_meta], batch_size=8))
        gold_ex = [doc_to_ex_gold(d, a["toponyms"])
                   for d, a in zip(gold_docs, docs_meta)]
        del gold_docs
        demonyms = demonym_gold_spans(docs_meta, gold_ex)
        n_linked = sum(len([t for t in a["toponyms"] if t["geonameid"]])
                       for a in docs_meta)
        n_dem = sum(1 for a in docs_meta for t in a["toponyms"]
                    if t["geonameid"] and
                    (a["doc_idx"], t["start"], t["end"]) in demonyms)
        logger.info(f"{src}: D2 excludes {n_dem} demonym gold toponyms of "
                    f"{n_linked} linked ({100 * n_dem / max(n_linked, 1):.1f}%); "
                    f"denominator is {n_linked - n_dem}")
        results["corpora"][src]["d2"] = {"n_linked_gold": n_linked,
                                         "n_demonym_excluded": n_dem,
                                         "n_gold_after_d2": n_linked - n_dem}
        if oracle:
            geo.geonames.clear_cache()
            gold_es, gold_picks, _ = score_examples(geo, gold_ex, max_choices,
                                                    outlets=outlets)
            gold_labels = {(a["doc_idx"], e["start_char"], e["end_char"]): e
                           for a, doc in zip(docs_meta, gold_es) for e in doc}
        else:
            gold_labels = {(a["doc_idx"], e["start_char"], e["end_char"]): e
                           for a, doc in zip(docs_meta, gold_ex) for e in doc}

        # The transformer pass is the same for every variant that reads the
        # same text, and it is the most expensive thing in the loop, so it is
        # run once per (text transform) and reused. `t_spacy` is the measured
        # cost of that one pass; every variant sharing it reports it, which is
        # what it would cost if the variant were run alone.
        doc_cache = {}

        for var in var_list:
            spec = VARIANTS[var]
            model_name, geo_labels = spec["ner"], spec["geo_labels"]
            ctx_labels, transform = spec["ctx_labels"], spec["transform"]
            gaz, trim = spec["gaz"], spec["trim"]
            texts = [a["text"] for a in docs_meta]
            if transform == "truecase":
                texts = [truecase_text(t) for t in texts]

            # Tensors always come from en_core_web_trf -- the ranker's frozen
            # inputs are its 768-d token vectors -- so a different NER model is
            # an extra pass, not a replacement.
            if "en_core_web_trf" not in nlp_cache:
                nlp_cache["en_core_web_trf"] = load_nlp_for("en_core_web_trf")
            cache_key = transform or "raw"
            if cache_key not in doc_cache:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                cached = list(nlp_cache["en_core_web_trf"].pipe(texts,
                                                                batch_size=8))
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                doc_cache[cache_key] = (cached, time.perf_counter() - t0)
            docs, t_spacy = doc_cache[cache_key]

            ner_docs = [None] * len(docs)
            t_ner_extra = 0.0
            if model_name != "en_core_web_trf":
                if model_name not in nlp_cache:
                    nlp_cache[model_name] = load_nlp_for(model_name,
                                                         tensors=False)
                t0 = time.perf_counter()
                ner_docs = list(nlp_cache[model_name].pipe(texts, batch_size=32))
                t_ner_extra = time.perf_counter() - t0

            if not parity_checked and model_name == "en_core_web_trf":
                check_parity(docs[0])
                parity_checked = True

            t_gaz = t_head = 0.0
            tagger = (span_taggers[spec["span_head"]] if spec["span_head"]
                      else None)
            all_doc_ex = []
            for doc, ndoc in zip(docs, ner_docs):
                if tagger is not None:
                    th = time.perf_counter()
                    ex = doc_to_ex_head(doc, tagger, ctx_labels)
                    t_head += time.perf_counter() - th
                    all_doc_ex.append(ex)
                    continue
                ex = doc_to_ex_labels(doc, geo_labels, ctx_labels, ner_doc=ndoc,
                                      trim_spans=trim)
                if gaz:
                    tg = time.perf_counter()
                    ex = ex + gazetteer_second_pass(
                        doc, ex, geo.geonames, ner_doc=ndoc,
                        outside=gaz in ("outside", "both"),
                        nested=gaz in ("nested", "both"))
                    ex.sort(key=lambda e: e["start_char"])
                    t_gaz += time.perf_counter() - tg
                all_doc_ex.append(ex)

            geo.geonames.clear_cache()
            all_es, picks, timing = score_examples(geo, all_doc_ex, max_choices,
                                                   outlets=outlets)

            res = evaluate_docs(docs_meta, all_es, picks, gold_es, gold_picks,
                                gold_labels, demonyms=demonyms)
            summary = summarize(res)
            # The pre-D2 denominator, so the rows of
            # experiments/campaign2/end_to_end_report.md stay checkable. Same
            # predictions, same model output -- only the gold set differs.
            old = summarize(evaluate_docs(docs_meta, all_es, picks, gold_es,
                                          gold_picks, gold_labels))
            summary["legacy_incl_demonym"] = {
                k: old[k] for k in ("n_gold", "det_exact_p", "det_exact_r",
                                    "det_exact_f1", "det_overlap_r", "e2e_em",
                                    "e2e_161", "e2e_overlap_em",
                                    "out_precision", "oracle")
                if k in old}
            summary["timing"] = {"spacy_s": round(t_spacy, 2),
                                 "es_s": round(timing["es"], 2),
                                 "model_s": round(timing["model"], 2),
                                 "gaz_s": round(t_gaz, 2),
                                 "head_s": round(t_head, 2),
                                 "ner_extra_s": round(t_ner_extra, 2),
                                 "docs": len(docs),
                                 "docs_per_s": round(len(docs) /
                                                     (t_spacy + timing["es"] +
                                                      timing["model"] + t_gaz +
                                                      t_head + t_ner_extra), 2)}
            summary["miss_examples"] = res["miss_examples"][:400]
            summary["boundary_examples"] = res["boundary_examples"][:200]
            summary["retrieval_examples"] = res["retrieval_examples"][:400]
            summary["gold_outcomes"] = res["gold_outcomes"]
            summary["spurious_examples"] = res["spurious_examples"][:200]
            summary["oracle_examples"] = res["oracle_examples"][:200]
            results["corpora"][src]["variants"][var] = summary
            logger.info(f"  {var}: det R(exact)={summary['det_exact_r']} "
                        f"P={summary['det_exact_p']} e2e EM={summary['e2e_em']} "
                        f"(pre-D2 denominator: {old['e2e_em']})")
        doc_cache.clear()
        del docs

    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=1, default=str)
        logger.info(f"wrote {out}")
    return results


@app.command()
def ner_errors(results_path: str, top: int = 40):
    """Categorise the NER misses recorded by `evaluate`."""
    with open(results_path) as f:
        results = json.load(f)
    for src, block in results["corpora"].items():
        for var, summary in block["variants"].items():
            miss = summary.get("miss_examples", [])
            if not miss:
                continue
            print(f"\n=== {src} / {var}: {len(miss)} recorded misses ===")
            c = Counter(m["phrase"] for m in miss)
            print("  most common:", c.most_common(top))
            shape = Counter()
            for m in miss:
                p = m["phrase"] or ""
                if p.isupper() and len(p) > 2:
                    shape["ALL CAPS"] += 1
                elif p.islower():
                    shape["lowercase"] += 1
                elif len(p.split()) > 1:
                    shape["multiword"] += 1
                else:
                    shape["single Titlecase"] += 1
            print("  shape:", dict(shape))


@app.command()
def latency(model_path: str = "experiments/e29_swa_ep15/seed42.pt",
            base_dir: str = "raw_data", source: str = "lgl",
            n_docs: int = 50, reps: int = 3, variants: str = "serving"):
    """Steady-state docs/sec and the share of time in NER / ES / model.

    The candidate cache is cleared before every repetition, so this is the
    cold-cache cost of a fresh batch of documents, not the flattering number a
    repeated corpus gives.
    """
    geo = Geoparser(model_path=model_path,
                    feature_blocks="prom,name,cue,sib,geo,shape",
                    oov_bucket_fix=True,
                    model_options={"return_logits": True, "mask_padding": True,
                                   "modern_mlp": True})
    articles = read_corpus(source, base_dir)
    heldout, _ = heldout_doc_indices(source, articles, base_dir)
    texts = [articles[i]["text"] for i in sorted(heldout)][:n_docs]
    nlp_cache = {"en_core_web_trf": load_nlp_for("en_core_web_trf")}
    list(nlp_cache["en_core_web_trf"].pipe(texts[:4]))  # warm up

    print(f"{source}: {len(texts)} documents")
    print(f"  {'variant':<12}{'spaCy':>8}{'extra NER':>10}{'extract':>11}"
          f"{'ES':>8}{'model':>8}{'total':>8}{'docs/s':>9}{'entities':>10}")
    for var in [v.strip() for v in variants.split(",") if v.strip()]:
        spec = VARIANTS[var]
        if spec["ner"] not in nlp_cache:
            nlp_cache[spec["ner"]] = load_nlp_for(spec["ner"], tensors=False)
        tagger = (load_span_tagger(spec["span_head"], device=geo.model.device)
                  if spec["span_head"] else None)
        rows = []
        for _ in range(reps):
            geo.geonames.clear_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            docs = list(nlp_cache["en_core_web_trf"].pipe(texts, batch_size=8))
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_spacy = time.perf_counter() - t0

            t_extra = 0.0
            ner_docs = [None] * len(docs)
            if spec["ner"] != "en_core_web_trf":
                t0 = time.perf_counter()
                ner_docs = list(nlp_cache[spec["ner"]].pipe(texts, batch_size=32))
                t_extra = time.perf_counter() - t0

            # The extraction column is the whole span-finding stage: the head
            # when there is one, otherwise the gazetteer pass it replaces.
            # doc_to_ex_labels itself is measured either way.
            t_gaz = 0.0
            all_doc_ex = []
            for doc, ndoc in zip(docs, ner_docs):
                if tagger is not None:
                    th = time.perf_counter()
                    ex = doc_to_ex_head(doc, tagger, spec["ctx_labels"])
                    t_gaz += time.perf_counter() - th
                    all_doc_ex.append(ex)
                    continue
                tg0 = time.perf_counter()
                ex = doc_to_ex_labels(doc, spec["geo_labels"], spec["ctx_labels"],
                                      ner_doc=ndoc, trim_spans=spec["trim"])
                t_gaz += time.perf_counter() - tg0
                if spec["gaz"]:
                    tg = time.perf_counter()
                    ex = ex + gazetteer_second_pass(
                        doc, ex, geo.geonames, ner_doc=ndoc,
                        outside=spec["gaz"] in ("outside", "both"),
                        nested=spec["gaz"] in ("nested", "both"))
                    t_gaz += time.perf_counter() - tg
                all_doc_ex.append(ex)
            _, _, timing = score_examples(geo, all_doc_ex, 100)
            rows.append((t_spacy, t_extra, t_gaz, timing["es"], timing["model"]))
        med = [float(np.median([r[i] for r in rows])) for i in range(5)]
        total = sum(med)
        n_ents = sum(len(x) for x in all_doc_ex)
        print(f"  {var:<12}" + "".join(f"{t:>8.2f}" if i not in (1, 2)
                                       else f"{t:>10.2f}" if i == 1
                                       else f"{t:>11.2f}"
                                       for i, t in enumerate(med)) +
              f"{total:>8.2f}{len(texts)/total:>9.2f}{n_ents:>10}")


if __name__ == "__main__":
    app()
