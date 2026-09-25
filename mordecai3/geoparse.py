
from collections import Counter
import json
import logging
import numpy as np
import os
import spacy
import torch
import re
import warnings

from elasticsearch import Elasticsearch
try: 
    from elasticsearch.dsl import Search             # type: ignore
except ImportError:
    # elasticsearch < 8.18.0
    from elasticsearch_dsl import Search
from importlib import resources
try: 
    from importlib.resources.abc import Traversable  # type: ignore[import-untyped]
except ImportError:
    # Python < 3.13
    from importlib.abc import Traversable
from torch.utils.data import DataLoader
import jellyfish
import numpy as np
import numpy.typing as npt

from tqdm import tqdm

from .elasticsearch import (
    setup_es_client,
    es_is_accepting_connection,
    es_has_geonames_index,
)
from .exceptions import (
    SpacyModelError,
    ElasticsearchConnectionError,
    GeonamesIndexError,
)
from .citation import maybe_show_citation_notice
from .candidate_features import (
    ALL_KEYS as EXTRA_FEATURE_KEYS,
    add_document_features,
    add_entity_features,
    fill_null_features,
    mention_admin_cue,
)
from .outlet_features import (
    OUTLET_KEYS,
    add_outlet_features,
    clear_outlet_features,
)
from .geonames import GeonamesService, hit_sources
from .mordecai_utilities import spacy_doc_setup
from .span_head import SPAN_HEAD_ASSETS, load_span_tagger
from .torch_model import ProductionData, expand_feature_blocks, geoparse_model


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


spacy_doc_setup()

def load_nlp(use_gpu=False):
    if use_gpu:
        activated = spacy.prefer_gpu()
        if activated:
            logger.info("spaCy: GPU activated")
        else:
            logger.info("spaCy: GPU requested but not available, using CPU")
    try:
        nlp = spacy.load("en_core_web_trf")
    except OSError:
        raise SpacyModelError()
    nlp.add_pipe("token_tensors")
    return nlp

def load_model(model_path, device=None, n_extra_features=None, **model_kwargs):
    """Rebuild a geoparse_model from a saved state dict and load the weights.

    Parameters
    ----------
    model_path : path-like
    device : torch.device or None
    n_extra_features : int or None
        Number of enrichment feature columns the checkpoint was trained with.
        None (the default) reads it off the checkpoint's mix_linear layer, which
        is 13 wide (4 cosine similarities + 9 gazetteer features) plus one column
        per enrichment feature. Pass a number to assert what you expect.
    **model_kwargs
        Passed to geoparse_model. Behavioral training flags that leave no trace
        in the layer shapes -- ``modern_mlp``, ``mask_padding``, ``return_logits``
        -- must be repeated here if the checkpoint was trained with them.
    """
    if not device:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    state = torch.load(model_path, map_location=device)
    # Read the layer sizes off the checkpoint instead of hardcoding them. The
    # shipped model happens to use the defaults, but tools/train.py exposes
    # --mix-dim, --country-size and --code-size, and a model trained with any
    # of those changed used to fail here with a shape mismatch.
    ckpt_extra = state['mix_linear.weight'].shape[1] - 13
    if n_extra_features is None:
        n_extra_features = ckpt_extra
    elif n_extra_features != ckpt_extra:
        raise ValueError(
            f"checkpoint {model_path} was trained with {ckpt_extra} extra "
            f"gazetteer features, but {n_extra_features} were requested. The "
            f"feature_blocks passed to Geoparser must match the ones the model "
            f"was trained with.")
    model = geoparse_model(device=device,
                           bert_size=state['text_to_country.weight'].shape[1],
                           num_feature_codes=state['code_emb.weight'].shape[0],
                           country_size=state['text_to_country.weight'].shape[0],
                           code_size=state['code_emb.weight'].shape[1],
                           mix_dim=state['mix_linear.weight'].shape[0],
                           n_extra_features=n_extra_features,
                           **model_kwargs)
    model.load_state_dict(state)
    model.eval()
    return model



# The packaged ship artifact: `experiments/e29_swa_ep15/seed101.pt`, the
# campaign's best single checkpoint (macro exact match 0.9300), promoted here
# by decision D3. `mordecai_2025-08-27.pt` is the pre-campaign asset and is
# left in place for anyone who needs the old behavior.
#
# STAGED, NOT DEFAULT: `assets/mordecai_2026-08-20_e54_seed42.pt` is the e54
# outlet ship candidate (+0.0368 TLG-hard, +3.4 e2e EM with outlets supplied,
# no regression without them -- experiments/campaign2/outlet_integration_report.md).
# It is packaged and its sidecar turns on the `outlet` block by itself, so
# promoting it is this one line and nothing else. The flip is the owner's call.
DEFAULT_MODEL_ASSET = "assets/mordecai_2026-08-20_seed101.pt"

# The behavioral flags a checkpoint's config sidecar can supply. They leave no
# trace in the layer shapes (or, for mix_depth/listwise, they do but silently),
# so a checkpoint loaded without them mis-runs. tools/train.py writes them to
# `<checkpoint>.json`; anything the caller passes explicitly still wins.
_SIDECAR_MODEL_KEYS = ("return_logits", "mask_padding", "modern_mlp",
                       "country_pred", "mix_depth", "residual", "listwise",
                       "listwise_heads", "aux_country", "aux_class")


# The geocoded domain -> newsroom-home table the `outlet` feature block reads,
# built from tools/data/outlet_homes_researched.tsv (see
# experiments/campaign2/outlet_integration_report.md). Loaded only when the
# checkpoint actually carries the block, so a model without it pays nothing.
DEFAULT_OUTLET_HOMES_ASSET = "assets/outlet_homes.json"


def normalize_outlet(outlet):
    """A caller's outlet string -> a key the home table can be looked up with.

    Accepts a bare domain (`parispi.net`), a host with a `www.` prefix, or a
    whole URL, because all three are what a document's provenance field
    actually looks like in the wild.
    """
    if not outlet:
        return ""
    text = str(outlet).strip().lower()
    if "//" in text:
        text = text.split("//", 1)[1]
    text = text.split("/", 1)[0].split("?", 1)[0]
    if "@" in text:                      # user:pass@host
        text = text.rsplit("@", 1)[1]
    return text.split(":", 1)[0].strip(".")


def lookup_outlet_home(homes, outlet):
    """The resolved home for one outlet string, or None.

    The table's keys are inconsistent about `www.` (LGL writes `ajc.com`,
    TR-News writes `www.cbc.ca`), so both forms are tried. An unknown domain
    returns None, which the feature code turns into the exact null encoding a
    document with no outlet at all carries -- that is the fallback the whole
    outlet-dropout arm exists to make safe.
    """
    if not homes:
        return None
    key = normalize_outlet(outlet)
    if not key:
        return None
    if key in homes:
        return homes[key]
    alt = key[4:] if key.startswith("www.") else "www." + key
    return homes.get(alt)


def read_outlet_homes(source):
    """Resolved outlet homes from a dict, a JSON path, or None for the asset."""
    if source is None:
        path = resources.files("mordecai3") / DEFAULT_OUTLET_HOMES_ASSET
        try:
            if not path.is_file():
                logger.info("No packaged outlet home table; the outlet block "
                            "will serve its null encoding for every document.")
                return {}
        except (AttributeError, OSError):
            return {}
    elif isinstance(source, dict):
        return source
    else:
        path = source
    try:
        with open(path, encoding="utf8") as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        logger.warning(f"Could not read outlet homes from {path}: {e}")
        return {}


def read_model_sidecar(model_path):
    """The training config tools/train.py wrote next to a checkpoint, or {}.

    `<checkpoint>.json` is the model config; `<checkpoint stem>.json` is
    usually the *metrics* dump, which is a different thing, so it is only
    accepted when it looks like a model config.
    """
    try:
        base = os.fspath(model_path)
    except TypeError:
        return {}
    for cand in (base + ".json", os.path.splitext(base)[0] + ".json"):
        if not os.path.exists(cand):
            continue
        try:
            with open(cand) as f:
                cfg = json.load(f)
        except (OSError, ValueError):
            continue
        if isinstance(cfg, dict) and "return_logits" in cfg:
            logger.info(f"Reading model config from {cand}")
            return cfg
    return {}


def guess_in_rel(ent):
    """
    A quick rule-based system to detect common "in" relations, such as 
    "Berlin, Germany" or "Aleppo in Syria".

    It tries to skip series of places and respects sentence boundaries.

    This uses some slightly clunky notation to handle the case in training data
    where we don't have a real span, just tokens. 
    """
    if type(ent) is list:
        ent = ent[0].doc[ent[0].i:ent[-1].i+1]
    try:
        next_ent = [e for e in ent.doc.ents if e.start > ent.end]
    except:
        return ""
    # if it's the last ent in the DOC, assume no "in" relation:
    if not next_ent:
        return ""
    next_ent = next_ent[0]
    # if it's the last ent in the SENT, assume no "in" relation:
    if ent.sent != next_ent.sent:
        return ""
    # If the next entity isn't a place, assume no "in" relation
    if next_ent.label_ not in ['GPE', "LOC", 'EVENT_LOC', 'FAC']:
        return ""
    # there's a following entity, separeted only by "in"
    diff = ent.doc[ent.end:next_ent.start]
    diff_text = [i.text for i in diff]
    if len(diff) <= 2 and "in" in diff_text and "and" not in diff_text:
        return next_ent.text
    # There's a comma relation
    if "," in diff_text:
        # skip if there's a ", and":
        if "and" in diff_text:
            return ""
        # skip if the following ent is followed by a comma
        try:
            if ent.doc[next_ent.end].text in [",", "and"]:
                return ""
        except IndexError:
            # next_ent is the last token in the doc; nothing follows it to check.
            logger.debug("No token after next_ent; treating as no \"in\" relation.")
            return ""
        return next_ent.text
    else:
        return ""


# The spaCy entity labels the geoparser resolves. FAC is in it by default for
# backwards compatibility (see Geoparser(include_fac=...)).
#
# NORP is NOT here and cannot be added: demonyms are out of the task
# (decision D2, experiments/campaign2/SYNTHESIS.md). Resolving "Turkish" to
# Turkey means reporting a location for "the Turkish president", the three
# geoparsing corpora annotate those spans with a country id, and the campaign
# now excludes them from every denominator instead of chasing them. The
# `accept_norp` constructor flag and its code path were deleted with D2; NORP
# is never emitted.
GEO_LABELS = ("GPE", "LOC", "EVENT_LOC", "FAC")
# The labels that feed `locs_tensor`, the document-context vector. This is
# GPE/LOC because that is the set tools/train.py's formatters pool over: a
# wider set at serving time hands the model a context vector it never saw in
# training (experiments/campaign2/encoder_scoping_report.md).
CONTEXT_LABELS = ("GPE", "LOC")


# --------------------------------------------------------------------------
# The reserved-row convention
# --------------------------------------------------------------------------
# Training, evaluation and serving used to treat the last row of the scored
# window three different ways, which is why a working abstention mechanism was
# invisible for the whole first campaign
# (experiments/campaign2/calibration_report.md §2a). The single convention,
# adopted in campaign 2 Phase 0 and implemented here, in
# tools/error_utils.py's `evaluate_results` and in tools/calibration_eval.py:
#
#   1. The scored window is `W = max_choices` rows. Row `W - 1` is the
#      RESERVED "no correct answer" row: `ProductionData` overwrites it with a
#      sentinel (feature code 53, country NULL, every gazetteer feature -1),
#      `TrainData.create_labels` targets it when nothing in the candidate list
#      is correct, and the mask keeps it live for every mention. It is a
#      trained class and is never identified with a candidate.
#   2. The CANDIDATE rows are `0 .. min(n_choices, W) - 1`, *minus* row
#      `W - 1` when the candidate list fills the window: there the sentinel
#      overwrote a real candidate, so that candidate was never scored and must
#      never be reported. `argmax` over the candidate rows decides *which*
#      place.
#   3. The reserved row is always in the softmax denominator, so
#      `p(reserved)` -- exposed as `p_no_match` -- decides *whether* to answer.
#   4. Abstention is an explicit outcome, not a dropped mention: see
#      `Geoparser._no_match_result`.

def candidate_row_count(n_choices, window):
    """How many rows of a mention's window hold a rankable candidate.

    `n_choices` is `len(es_choices)` (the ES hits plus the appended gazetteer
    NULL row); `window` is the model's `max_choices`. See the convention above:
    a list that fills the window loses its last in-window candidate to the
    reserved sentinel.
    """
    n_choices, window = int(n_choices), int(window)
    n = window - 1 if n_choices >= window else min(n_choices, window)
    return max(n, 0)


# Span-boundary junk that spaCy routinely includes and the gazetteer never has:
# "the United States", "New Mexico's".
LEADING_DET = frozenset({"the", "a", "an"})
TRAILING_POSSESSIVE = frozenset({"'s", "’s"})

# ...except where the article *is* the name. Derived from the index rather than
# guessed, and from the primary `name` field rather than the alternate names:
# "The Hague" and "The Bronx" are geonames names, while "The Gambia" and "The
# Netherlands" are only alternate names of places whose own name is bare -- and
# English prose ("the Gambia", "the Netherlands", "the United States") wants
# exactly those trimmed. Regenerate with:
#
#   es.search(index="geonames", size=1000, sort=[{"population": "desc"}],
#             query={"bool": {"must": [{"match": {"name": "the"}}],
#                    "filter": [{"terms": {"feature_class": ["A", "P"]}},
#                               {"range": {"population": {"gte": 10000}}}]}})
#   # then keep the hits whose name.lower() starts with "the "
KEEP_LEADING_THE = frozenset({
    "the acreage", "the beaches", "the big 5 false bay", "the boldons",
    "the bronx", "the colony", "the crossings", "the dalles", "the dāngs",
    "the gap", "the hague", "the hammocks", "the hills shire", "the msunduzi",
    "the peak", "the pinery", "the ponds", "the scottish borders",
    "the villages", "the woodlands"})


def _keeps_leading_article(tokens):
    """Is the leading article part of this place's own gazetteer name?

    Only a capitalised "The" counts: lower-case prose ("the villages of Kent")
    is never the name, and the whole span has to match, so "The Hague" is
    protected while "the Thames Valley" is not.
    """
    if not tokens or tokens[0].text != "The":
        return False
    text = tokens[0].doc.text[tokens[0].idx:
                              tokens[-1].idx + len(tokens[-1].text)]
    return text.lower() in KEEP_LEADING_THE


def geoparse_labels(include_fac=True):
    """The entity labels to geoparse under a given Geoparser configuration.

    NORP is never in the result: see GEO_LABELS and decision D2.
    """
    labels = ["GPE", "LOC", "EVENT_LOC"]
    if include_fac:
        labels.append("FAC")
    return tuple(labels)


def trim_span_tokens(tokens):
    """Drop a leading determiner and a trailing possessive/punctuation.

    spaCy's GPE/LOC spans routinely carry them -- "the United States",
    "New Mexico's", "the Thames Valley" -- and they are pure cost: the corpora
    annotate the bare toponym, so every one of them is a boundary error even
    when the pipeline resolves it correctly, and the determiner also goes into
    the Elasticsearch query. Measured on the held-out TR-News/LGL/GWN
    documents this is +2.0 end-to-end exact match, +2.6 detection F1, at
    slightly *better* precision and slightly lower latency
    (experiments/campaign2/end_to_end_report.md §5e).

    Places whose gazetteer name begins with the article keep it: see
    `KEEP_LEADING_THE`. The tail is trimmed first so the guard sees the name
    itself ("The Hague's" -> "The Hague", not "Hague").

    Returns a (possibly empty) list of tokens.
    """
    while tokens and (tokens[-1].is_punct
                      or tokens[-1].text.lower() in TRAILING_POSSESSIVE):
        tokens = tokens[:-1]
    while tokens and (tokens[0].text.lower() in LEADING_DET or tokens[0].is_punct):
        if _keeps_leading_article(tokens):
            break
        tokens = tokens[1:]
    return tokens


def doc_to_ex_expanded(doc, geo_labels=GEO_LABELS, context_labels=CONTEXT_LABELS,
                       trim_spans=True):
    """
    Take in a spaCy doc with a custom ._.tensor attribute on each token and create a list
    of dictionaries with information on each place name entity.

    In the broader pipeline, this is called after nlp() and the results are passed to the
    Elasticsearch step.

    Parameters
    ---------
    doc: spacy.Doc
      Needs custom ._.tensor attribute.
    geo_labels: tuple of str
      Entity labels to geoparse. See `geoparse_labels`.
    context_labels: tuple of str
      Entity labels whose tokens are pooled into `locs_tensor`. These entities
      are context only; they are not geoparsed unless they are also in
      `geo_labels`.
    trim_spans: bool
      Strip a leading determiner and a trailing possessive from each span (see
      `trim_span_tokens`). On by default.

    Returns
    -------
    data: list of dicts
    """
    data = []
    doc_tensor = np.mean(np.vstack([i._.tensor for i in doc]), axis=0)
    # the "loc_ents" are the ones we use for context. Anecdotally, FACs aren't
    # so useful for context, but we do want to geoparse them.
    loc_ents = [ent for ent in doc.ents if ent.label_ in context_labels]
    context_tokens = [tok for e in loc_ents for tok in e]
    for ent in doc.ents:
        if ent.label_ in geo_labels:
            own = trim_span_tokens(list(ent)) if trim_spans else list(ent)
            if not own:
                # The whole span was a determiner or punctuation.
                continue
            start_char = own[0].idx
            end_char = own[-1].idx + len(own[-1].text)
            tensor = np.mean(np.vstack([i._.tensor for i in own]), axis=0)
            own_i = {i.i for i in own}
            other_locs = [i for i in context_tokens if i.i not in own_i]
            # The "in" relation is read off the untrimmed entity: it looks at
            # what follows the span, which trimming does not change.
            in_rel = guess_in_rel(ent)
            #print("detected relation: ", ent.text, "-->", in_rel)
            if other_locs:
                locs_tensor = np.mean(np.vstack([i._.tensor for i in other_locs]), axis=0)
            else:
                locs_tensor = np.zeros(len(tensor))
            d = {"search_name": doc.text[start_char:end_char],
                 # The detector's own verdict on what kind of span this is.
                 # Not used by the ranker; carried so a caller can report why
                 # the mention was picked up at all (GPE/LOC/FAC from spaCy,
                 # "NESTED" from the gazetteer pass, "SPAN" from the learned
                 # head).
                 "label": ent.label_,
                 "tensor": tensor,
                 "doc_tensor": doc_tensor,
                 "locs_tensor": locs_tensor,
                 "sent": own[0].sent.text,
                 "in_rel": in_rel,
                "start_char": start_char,
                "end_char": end_char}
            data.append(d)
    return data


# Entity labels the pipeline never geoparses but that routinely swallow a
# toponym: "Paris Police Department", "University of Pennsylvania", "Montana
# Department of Corrections".
NESTED_LABELS = ("ORG", "FAC", "EVENT", "WORK_OF_ART", "LAW", "PRODUCT")

# Words a gazetteer match on its own is never worth making.
GAZ_STOP = frozenset({"the", "a", "an", "of", "and", "in", "on", "at", "for",
                      "to", "he", "she", "it", "they", "we", "you", "i", "is",
                      "was", "said", "mr", "mrs", "ms", "dr"})


def _gaz_span_ok(tokens):
    """Could this run of tokens plausibly be a place name?"""
    return all(t.text[:1].isupper() and t.text.isalpha()
               and t.text.lower() not in GAZ_STOP for t in tokens)


def nested_gazetteer_spans(doc, existing, geonames_service: GeonamesService,
                           max_tokens=3):
    """Toponyms buried inside an entity the pipeline discards.

    Half of the toponyms the pipeline misses are nested in a larger entity --
    "Paris" inside "Paris Police Department", "Pennsylvania" inside "University
    of Pennsylvania" -- where spaCy emits one ORG span and the toponym is never
    looked up. This pass walks the sub-spans of those entities, leftmost-longest,
    and keeps the ones the gazetteer knows exactly: a geonames entry whose own
    `name` (or `asciiname`) equals the candidate string and whose feature class
    is A or P (an administrative unit or a populated place).

    Measured at +7.9 end-to-end exact match, for a detection precision cost of
    75.5 -> 67.0 and ~19% of throughput
    (experiments/campaign2/end_to_end_report.md §5d), which is why it is
    opt-in: Geoparser(nested_gazetteer_pass=True).

    Parameters
    ----------
    doc : spacy.Doc
    existing : list of dicts
        What doc_to_ex_expanded already found; their character ranges are not
        re-extracted.
    geonames_service : GeonamesService
    max_tokens : int
        Longest sub-span to consider.

    Returns
    -------
    list of dicts
        Extra entity dicts in the same shape doc_to_ex_expanded returns.
    """
    taken = set()
    for ex in existing:
        taken.update(range(ex["start_char"], ex["end_char"]))

    def is_free(lo, hi):
        return not any(c in taken for c in range(lo, hi))

    cands = {}   # candidate string -> [(first token index, last token index + 1)]
    for e in doc.ents:
        if e.label_ not in NESTED_LABELS:
            continue
        inside = [t for t in doc
                  if t.idx >= e.start_char and t.idx + len(t) <= e.end_char]
        for k in range(len(inside)):
            for length in range(max_tokens, 0, -1):
                if k + length > len(inside):
                    continue
                span = inside[k:k + length]
                if not _gaz_span_ok(span):
                    continue
                lo, hi = span[0].idx, span[-1].idx + len(span[-1].text)
                if not is_free(lo, hi):
                    continue
                cands.setdefault(doc.text[lo:hi], []).append(
                    (span[0].i, span[-1].i + 1))

    if not cands:
        return []

    names = list(cands)
    responses = geonames_service.search_by_names([(nm, 5, 0, False, None)
                                                  for nm in names])
    exact_admin_or_place = set()
    for nm, res in zip(names, responses):
        for hit in hit_sources(res):
            if str(hit.get("name", "")).lower() != nm.lower() and \
               str(hit.get("asciiname", "")).lower() != nm.lower():
                continue
            if hit.get("feature_class") in ("A", "P"):
                exact_admin_or_place.add(nm)
                break

    # Leftmost-longest, so "Santa Barbara" beats "Santa" inside the same entity.
    placed = sorted(((a, b, nm) for nm, spans in cands.items() for a, b in spans),
                    key=lambda x: (x[0], -(x[1] - x[0])))
    added = []
    doc_tensor = np.mean(np.vstack([i._.tensor for i in doc]), axis=0)
    for a, b, nm in placed:
        if nm not in exact_admin_or_place:
            continue
        span = doc[a:b]
        lo, hi = span[0].idx, span[-1].idx + len(span[-1].text)
        if not is_free(lo, hi):
            continue
        tensor = np.mean(np.vstack([t._.tensor for t in span]), axis=0)
        added.append({"search_name": doc.text[lo:hi],
                      "label": "NESTED",
                      "tensor": tensor,
                      "doc_tensor": doc_tensor,
                      # These mentions sit inside an organisation name, so the
                      # document's place-name context is not theirs to claim.
                      "locs_tensor": np.zeros(len(tensor)),
                      "sent": span.sent.text,
                      "in_rel": "",
                      "start_char": lo,
                      "end_char": hi})
        taken.update(range(lo, hi))
    return added


def load_hierarchy(asset_path):
    fn = os.path.join(asset_path, "hierarchy.txt")
    with open(fn, "r", encoding="utf-8") as f:
        hierarchy = f.read()
    hierarchy = hierarchy.split("\n")
    hier_dict = {}
    for h in hierarchy:
        h_split = h.split("\t")
        try:
            hier_dict.update({h_split[1]: h_split[0]})
        except IndexError:
            continue
    return hier_dict
            

class Geoparser:
    conn: Search

    def __init__(self, 
                 model_path: str | Traversable | None=None, 
                 geo_asset_path: str | Traversable | None=None,
                 geonames: GeonamesService | None=None,
                 nlp=None,
                 debug: bool=False,
                 trim=None,
                 check_es: bool=True,
                 hosts: list[str] | None = None,
                 port: int = 9200,
                 device=None,
                 use_ssl: bool=False,
                 es_client: Elasticsearch | None=None,
                 feature_blocks: str | list[str] | None=None,
                 outlet_homes: str | dict | None=None,
                 oov_bucket_fix: bool | None=None,
                 model_options: dict | None=None,
                 trim_spans: bool=True,
                 nested_gazetteer_pass: bool=False,
                 include_fac: bool=True,
                 span_detector: str | None=None,
                 span_threshold: float | None=None,
                 normalize_place_abbrevs: bool | None=None,
                 temperature: float=0.874):
        """
        feature_blocks : str, list of str, or None
            Enrichment feature blocks the loaded checkpoint was trained with,
            e.g. "prom,name,cue,sib,geo,shape" (see
            torch_model.FEATURE_BLOCKS). None (the default) reads them from the
            checkpoint's config sidecar `<model_path>.json` if there is one --
            which is how the packaged model gets its six blocks -- and falls
            back to the legacy behavior when there is not: the extra features
            are neither computed nor fed to the model, so pre-enrichment
            checkpoints keep working unchanged. When set, the lookup path
            computes those features for every candidate and the model is built
            with the matching input width, which is checked against the
            checkpoint.
        outlet_homes : str, dict, or None
            The geocoded outlet-domain -> newsroom-home table the `outlet`
            feature block reads, as a path to the JSON `tools/enrich_pickles.py
            --outlet-only` writes, or the loaded dict itself. None (the
            default) reads the packaged `assets/outlet_homes.json` when the
            checkpoint carries the `outlet` block, and is ignored entirely when
            it does not. A document whose outlet is unknown, or not supplied at
            all, gets the block's null encoding -- the ship checkpoint is
            trained with `--outlet-dropout 0.5` precisely so that costs nothing
            (experiments/e53_outlet_dropout/NOTES.md).
        oov_bucket_fix : bool or None
            Must match tools/train.py's --oov-bucket-fix for the checkpoint:
            it decides whether an out-of-vocabulary feature code shares the
            "NULL" embedding, and what country the reserved last row gets. It
            leaves no trace in the layer shapes. None (the default) reads it
            from the config sidecar, falling back to False.
        model_options : dict or None
            Extra keyword arguments for geoparse_model, for training flags that
            leave no trace in the checkpoint's layer shapes: modern_mlp,
            mask_padding, return_logits. Anything not given here is taken from
            the config sidecar; anything given here wins over it.
        trim_spans : bool
            Strip a leading determiner and a trailing possessive from spaCy's
            entity spans ("the United States" -> "United States"). On by
            default: it is worth +2.0 end-to-end exact match at better
            precision and slightly lower latency. See `trim_span_tokens`.
        nested_gazetteer_pass : bool
            Also extract toponyms nested inside organisation and facility names
            ("Paris" in "Paris Police Department"). Off by default: +7.9
            end-to-end exact match, but detection precision 75.5 -> 67.0 and
            ~19% of throughput. See `nested_gazetteer_spans`.
        include_fac : bool
            Resolve FAC (facility) spans. True is the historical behavior and
            the default. Dropping it buys 11.6 points of detection precision
            and 43% more throughput for 1.1 points of detection recall, so a
            deployment that does not want stadiums and airports should set it
            to False.
        span_detector : str or None
            Which place-span detector finds the mentions to geoparse. None (the
            default) is the historical path: spaCy's own entities, filtered to
            `geo_labels`, trimmed by `trim_span_tokens`, optionally extended by
            `nested_gazetteer_spans`.

            "gold" or "all" replace all three with the 0.5 M-parameter span
            head of experiments/campaign2/ner_head_scaling_report.md, which
            scores every span of <= 8 tokens over the token vectors the
            pipeline has already computed for the ranker -- no second
            transformer pass, no Elasticsearch round trip, and nested toponyms
            ("Pittsburgh" inside "University of Pittsburgh") come out beside
            their host. A filesystem path to any head checkpoint also works.

            "gold" is the higher flat-detection head (D2 det F1 87.62); "all"
            trades 1.3 F1 for +8.6 nested detection recall and 25 fewer demonym
            false positives. End to end, "all" wins
            (experiments/campaign2/span_head_serving_report.md).

            Two behaviours change for callers when this is on: the output can
            contain OVERLAPPING spans, and demonyms are suppressed by training
            rather than guaranteed away by the label filter, so a caller who
            must never see one needs a NORP post-filter of their own.
            `trim_spans` and `nested_gazetteer_pass` are inert while it is set.
        span_threshold : float or None
            Override the detection threshold that travels inside the head
            checkpoint (0.5 for "gold", 0.3 for "all"). Those are the operating
            points every measured number is at; a caller who changes this is
            off the measured curve. For sweeps.
        normalize_place_abbrevs : bool or None
            Expand US-state / Canadian-province abbreviations to the full ADM1
            name before the Elasticsearch query (e52 rule R1; see
            `GeonamesService` and `mordecai3.place_aliases`). None, the
            default, means "use the GeonamesService's own setting" -- which is
            True for a service this constructor builds, and whatever the caller
            chose for a service passed in as `geonames=`. Setting it True or
            False here overrides that, including on a supplied service.

            It is on by default because it fixes a retrieval bug rather than
            expressing a preference: worth +0.020 TLG-hard exact match on a
            fixed denominator with 28 entities gained and 0 lost, and +1.44 EM
            end to end in the best serving cell (80.85 -> 82.29)
            (`experiments/campaign2/r1_retrieval_report.md`). Turn it off to
            reproduce a pre-e57 candidate list byte for byte.
        temperature : float
            Temperature the reported probabilities are scaled by. The default,
            0.874, is the leave-one-source-out fit for the campaign's ship
            recipe (experiments/campaign2/calibration_report.md §5), which
            takes its pooled expected calibration error from 0.019 to 0.012;
            use 0.79 for a 5-seed ensemble and 1.0 for the model's own
            probabilities. A checkpoint trained before the double-softmax fix
            is saturated, and no temperature makes its scores calibrated.
        """
        # device=None (the default) auto-detects CUDA. Pass device='cpu' to force CPU.
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)
        use_gpu = (device.type != "cpu")
        logger.info(f"Using device: {device}")
        self.debug = debug
        self.trim = trim
        self.trim_spans = trim_spans
        self.nested_gazetteer_pass = nested_gazetteer_pass
        self.include_fac = include_fac
        self.geo_labels = geoparse_labels(include_fac=include_fac)
        # The span head replaces the label filter, the span trimmer and the
        # nested-gazetteer pass in one call (see `span_detector` above and
        # experiments/e55_ner_head/ship/INTEGRATION.md). Loaded eagerly so a
        # bad name fails at construction rather than on the first document.
        self.span_detector = span_detector
        self.span_tagger = None
        if span_detector is not None:
            self.span_tagger = load_span_tagger(span_detector, device=device,
                                                threshold=span_threshold)
            if nested_gazetteer_pass:
                logger.warning("nested_gazetteer_pass is ignored when "
                               "span_detector is set: the head emits nested "
                               "spans itself, without the ES round trip.")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature
        if not nlp:
            self.nlp = load_nlp(use_gpu=use_gpu)
        else:
            if 'token_tensors' not in nlp.pipe_names:
                try:
                    nlp.add_pipe("token_tensors")
                except Exception as e:
                    # TODO: this is currently catching the error that the pipe already exists,
                    # but it shouldn't catch the error that it doesn't know what
                    # token_tensors is.
                    logger.info(f"Error loading token_tensors pipe: {e}")
                    pass
            self.nlp = nlp
        
        # Handle ES and GeonamesService connection

        if es_client is None:
            if geonames is not None:
                # Reuse the client the caller's GeonamesService already holds, so we
                # don't open a second connection (and so check_es validates the same
                # client that lookups will actually go through).
                es_client = geonames.conn
            else:
                es_client = setup_es_client(hosts=hosts, port=port, use_ssl=use_ssl)
        self.conn = Search(using=es_client, index="geonames")

        if geonames is not None:
            self.geonames = geonames
            # An explicit flag overrides a supplied service's own setting; None
            # leaves the caller's object exactly as they configured it.
            if normalize_place_abbrevs is not None:
                self.geonames.normalize_place_abbrevs = normalize_place_abbrevs
        else:
            self.geonames = GeonamesService(
                es_client=es_client,
                normalize_place_abbrevs=(True if normalize_place_abbrevs is None
                                         else normalize_place_abbrevs))

        if check_es:
            logger.info("Checking Elasticsearch connection...")
            if not es_is_accepting_connection(es_client):
                raise ElasticsearchConnectionError()
            if not es_has_geonames_index(es_client):
                raise GeonamesIndexError()
            logger.info("Successfully connected to Elasticsearch.")

        
        if not model_path:
            model_path = resources.files("mordecai3") / DEFAULT_MODEL_ASSET
        # Kept so a caller can report which checkpoint is answering. Guessing
        # it from the assets directory gets the wrong file: the packaged
        # default is not the alphabetically last mordecai_*.pt there.
        self.model_path = model_path
        # A checkpoint's config sidecar carries the training flags that the
        # layer shapes do not reveal. Reading it is what lets `Geoparser()`
        # with no arguments load the campaign's model correctly instead of
        # silently running it as a default-configured one.
        sidecar = read_model_sidecar(model_path)
        # The enrichment features are opt-in: computing them costs a little CPU
        # per candidate, and a checkpoint that wasn't trained on them can't use
        # them anyway.
        if feature_blocks is None:
            feature_blocks = sidecar.get("feature_blocks") or None
        if oov_bucket_fix is None:
            oov_bucket_fix = bool(sidecar.get("oov_bucket_fix", False))
        opts = {k: sidecar[k] for k in _SIDECAR_MODEL_KEYS if k in sidecar}
        opts.update(model_options or {})
        self.model_options = opts
        self.feature_blocks = feature_blocks
        self.extra_feature_keys = expand_feature_blocks(feature_blocks)
        self.oov_bucket_fix = oov_bucket_fix
        # The outlet block is per-document, so unlike every other block it
        # needs a lookup table at serve time. Load it only when the checkpoint
        # was trained with the block; otherwise the argument is inert and the
        # keys are never written onto a candidate.
        self.uses_outlet = any(k in self.extra_feature_keys for k in OUTLET_KEYS)
        self.outlet_homes = read_outlet_homes(outlet_homes) if self.uses_outlet else {}
        if self.uses_outlet:
            logger.info(f"Outlet feature block active; "
                        f"{len(self.outlet_homes)} outlet homes loaded.")

        self.model = load_model(model_path, device=device,
                                n_extra_features=len(self.extra_feature_keys),
                                **opts)
        if not geo_asset_path:
            geo_asset_path = resources.files("mordecai3") / "assets/"
        self.hierarchy = load_hierarchy(geo_asset_path)
        self.model.to(device)
        maybe_show_citation_notice()

    def lookup_city(self, entry):
        """
        
        """
        city_id = ""
        city_name = ""
        if entry['feature_code'] == 'PPLX':
            try:
                parent_id = self.hierarchy[entry['geonameid']]
                parent_res = self.geonames.get_entry_by_id(parent_id)
                if parent_res['feature_class'] == "P":
                    city_id = parent_id
                    city_name = parent_res['name']
            except KeyError:
                city_id = entry['name']
                city_name = entry['geonameid']
        elif entry['feature_class'] == 'S':
            try:
                parent_id = self.hierarchy[entry['geonameid']]
                parent_res = self.geonames.get_entry_by_id(parent_id)
                if parent_res['feature_class'] == "P":
                    city_id = parent_id
                    city_name = parent_res['name']
            except KeyError:
                city_id = ""
                city_name = ""
        elif re.search("PPL", entry['feature_code']):
            # all other cities, just return self
            city_name = entry['name']
            city_id = entry['geonameid']
        else:
            # if it's something else, there is no city
            city_id = ""
            city_name = ""
        return city_id, city_name


    def _candidate_probs(self, pred, n_choices):
        """Temperature-scaled probabilities over one mention's scored rows.

        The model scores a window of `len(pred)` rows. `candidate_row_count`
        says how many hold a rankable candidate; the last row is the reserved
        "no correct answer" class, which is live even when the candidate list
        is shorter than the window. Both are in the softmax denominator,
        padding rows are not -- the `p_pred_full` convention of
        tools/calibration_eval.py, which is the best single wrong-answer score
        the calibration study found (AUROC 0.899) and the one the temperature
        was fit for (calibration_report.md §2a, §6, §8).
        """
        logits = pred.detach().float().cpu().numpy()
        if not getattr(self.model, "return_logits", False):
            # A model built without return_logits has already softmaxed its
            # scores; the log recovers logits that reproduce them exactly.
            logits = np.log(np.clip(logits, 1e-30, None))
        window = len(logits)
        live = np.zeros(window, dtype=bool)
        live[:candidate_row_count(n_choices, window)] = True
        live[-1] = True   # the reserved row is always scored
        z = np.where(live, logits / self.temperature, -np.inf)
        e = np.where(live, np.exp(z - z.max()), 0.0)
        return e / max(e.sum(), 1e-300)

    @staticmethod
    def _no_match_result(ent, p_no_match, candidates=None):
        """The result for a mention the model declines to place.

        Both abstention branches return this, so a caller can tell "no location
        here" from "a location, but a doubtful one" with a single key
        (calibration_report.md §2a: the two branches used to disagree, one
        returning a bare dict and the other dropping the mention entirely).

        `candidates` is the ranked list the model rejected, attached only when
        the caller asked for one with `top_k`. An abstention with its runners-up
        visible is a far more useful thing to show a human than an empty
        result: it is the difference between "this is not a place" and "this is
        a place I cannot choose between".
        """
        out = {"search_name": ent['search_name'],
               "label": ent.get('label'),
               "start_char": ent['start_char'],
               "end_char": ent['end_char'],
               "no_match": True,
               "p_no_match": p_no_match}
        if candidates is not None:
            out["candidates"] = candidates
        return out

    def _resolve_results(self, es_data, pred_val, debug=False, top_k=None):
        """Select the best geonames candidates based on model predictions.

        Parameters
        ----------
        es_data : list of dicts
            ES-enriched entity data for a single document.
        pred_val : torch.Tensor
            Model predictions, shape (num_entities, max_choices).
        debug : bool
            If True, return the top 4 candidates per entity instead of just the best.
        top_k : int or None
            When set, attach the mention's `top_k` best-scoring candidates to
            its result under a `candidates` key. This is orthogonal to `debug`:
            `debug` changes how many *rows* a mention contributes to the output
            list, while `top_k` leaves the one-row-per-mention shape alone and
            hangs the runners-up off it. A UI that has to explain why "Gao"
            beat "Gao Region" wants the latter.

        Returns
        -------
        best_list : list of dicts
            Each dict carries `no_match` and `p_no_match`; the ones that placed
            the mention also carry the geonames fields and `score`, the
            probability of the chosen candidate.
        """
        best_list = []
        for (ent, pred) in zip(es_data, pred_val):
            logger.debug("**Place name**: {}".format(ent['search_name']))
            probs = self._candidate_probs(pred, len(ent['es_choices']))
            p_no_match = float(probs[-1])
            # if the last one is the argmax, then the model thinks that no answer is
            # correct, so return blank
            if pred[-1] == pred.max():
                logger.debug("Model predicts no answer")
                best_list.append(self._no_match_result(
                    ent, p_no_match, self._ranked_candidates(ent, probs, top_k)))
                continue

            # The probabilities are a monotone transform of the scores the
            # model emits, so this scores and ranks exactly what the raw
            # outputs did -- it just reports a number a user can act on.
            # Only the candidate rows get a score: when the list fills the
            # window, its last in-window candidate was overwritten by the
            # reserved sentinel and was never scored, so attributing the
            # sentinel's probability to it would report a place the model
            # never ranked (the reserved-row convention above).
            for n in range(candidate_row_count(len(ent['es_choices']), len(probs))):
                ent['es_choices'][n]['score'] = float(probs[n])
            results = [e for e in ent['es_choices'] if 'score' in e.keys()]

            # this is what the elements of "results" look like
             #  {'feature_code': 'PPL',
             #  'feature_class': 'P',
             #  'country_code3': 'BRA',
             #  'lat': -22.99835,
             #  'lon': -43.36545,
             #  'name': 'Barra da Tijuca',
             #  'admin1_code': '21',
             #  'admin1_name': 'Rio de Janeiro',
             #  'admin2_code': '3304557',
             #  'admin2_name': 'Rio de Janeiro',
             #  'geonameid': '7290718',
             #  'score': 1.0,
             #  'search_name': 'Barra da Tijuca',
             #  'start_char': 557,
             #  'end_char': 581
             #  }
            scores = np.array([r['score'] for r in results])
            if len(scores) == 0:
                logger.debug("No scores found.")
                best_list.append(self._no_match_result(ent, p_no_match, 
                                                       [] if top_k else None))
                continue
            # The gazetteer "none of the above" row is always the last element
            # of es_choices -- which is inside the scored window only when the
            # candidate list is shorter than it. Testing its own index rather
            # than "the last scored row" keeps a real candidate at the end of a
            # truncated window from being mistaken for it.
            if int(np.argmax(scores)) == len(ent['es_choices']) - 1:
                logger.debug("Picking final ''null'' result.")
                # print the next best result:
                if len(scores) == 1:
                    logger.debug(f"Only one score found: {results[0]}")
                if len(scores) > 1:
                    second_best_idx = np.argsort(scores)[-2]
                    second_best = results[second_best_idx]
                    logger.debug(f"Second best result: {second_best.get('name', 'N/A')} (score: {second_best.get('score', 'N/A')})")
                best_list.append(self._no_match_result(
                    ent, p_no_match, self._ranked_candidates(ent, probs, top_k)))
                continue
            results = sorted(results, key=lambda k: -k['score'])
            if not debug:
                logger.debug("Picking top predicted result")
                results = results[0:1]
            else:
                logger.debug("Returning top 4 predicted results for each location")
                results = results[0:4]
            # Built before the loop below mutates `results`, and from a copy,
            # so the runners-up a caller sees are never the same dicts the
            # chosen result is about to have `search_name`/offsets stamped onto.
            ranked = self._ranked_candidates(ent, probs, top_k)
            for best in results:
                best["search_name"] = ent['search_name']
                best["label"] = ent.get('label')
                best["start_char"] = ent['start_char']
                best["end_char"] = ent['end_char']
                best["no_match"] = False
                best["p_no_match"] = p_no_match
                ## Add in city info here
                best['city_id'], best['city_name'] = self.lookup_city(best)
                if ranked is not None:
                    best["candidates"] = ranked
                best_list.append(best)
        return best_list

    @staticmethod
    def _ranked_candidates(ent, probs, top_k):
        """The mention's `top_k` best-scoring candidates, or None if not asked for.

        Scores come from the same temperature-scaled `probs` the choice was made
        from, so the list a caller displays is the ranking the model actually
        produced. The gazetteer's own "none of the above" row is dropped: it is
        an abstention signal, already reported as `p_no_match`, and showing it
        as a candidate named "NULL" helps nobody.
        """
        if not top_k:
            return None
        out = []
        for n in range(candidate_row_count(len(ent['es_choices']), len(probs))):
            choice = ent['es_choices'][n]
            if choice.get('geonameid') == 'NULL':
                continue
            row = dict(choice)
            row['score'] = float(probs[n])
            out.append(row)
        out.sort(key=lambda c: -c['score'])
        return out[:top_k]

    @staticmethod
    def _trim_results(best_list):
        """Remove the internal-only keys that are used to pick the best result."""
        trim_keys = ['admin1_parent_match', 'country_code_parent_match', 'alt_name_length',
                    'min_dist', 'max_dist', 'avg_dist', 'ascii_dist', 'adm1_count',
                    'country_count'] + EXTRA_FEATURE_KEYS + OUTLET_KEYS
        for entry in best_list:
            for key in trim_keys:
                entry.pop(key, None)
            # The runners-up carry the same internal keys and are the same
            # thing to a caller, so trimming has to reach them as well -- a
            # `trim=True` result with the ranker's private features hanging off
            # `candidates` would be the worst of both.
            for cand in entry.get('candidates') or ():
                for key in trim_keys:
                    cand.pop(key, None)

    def _outlet_homes_for(self, n_docs, outlets):
        """One resolved home (or None) per document, or None if inert.

        Returns None when the checkpoint has no outlet block, which is what
        tells `add_es_data_batch` to leave the candidates alone. When the block
        *is* there every document gets an entry, so a document with no outlet
        is written to the null encoding rather than left with the keys missing.
        """
        if not self.uses_outlet:
            if outlets is not None and any(outlets):
                logger.warning("outlet= was supplied but this checkpoint was "
                               "not trained with the 'outlet' feature block; "
                               "ignoring it.")
            return None
        if outlets is None:
            return [None] * n_docs
        outlets = list(outlets)
        if len(outlets) != n_docs:
            raise ValueError(f"got {len(outlets)} outlets for {n_docs} "
                             "documents; pass one per document (None is fine)")
        return [lookup_outlet_home(self.outlet_homes, o) for o in outlets]

    def _geoparse_docs(self, docs, max_choices=100, known_country=None,
                       trim=True, debug=False, es_workers=4, outlets=None,
                       top_k=None):
        """Core geoparsing pipeline for a list of spaCy docs.

        Handles entity extraction, ES lookups (threaded across all documents),
        cross-document model batching, and result resolution. This is the shared
        implementation behind both geoparse_doc() and geoparse_batch().

        Parameters
        ----------
        docs : list of spacy.tokens.doc.Doc
        max_choices : int
            Maximum ES candidates per entity.
        known_country : str or None
            Restrict results to a single country (ISO 3166-1 alpha-3).
        trim : bool
            Remove internal keys from output.
        debug : bool
            Return the top 4 candidates per entity.
        es_workers : int
            Thread pool size for ES lookups.
        top_k : int or None
            Attach each mention's top `top_k` candidates under `candidates`.
            See `_resolve_results`.

        Returns
        -------
        list of dicts
            One result dict per input document.
        """
        # 1. Entity extraction
        all_doc_ex = []
        for doc in docs:
            try:
                if self.span_tagger is not None:
                    # One call replaces the label filter, the span trimmer and
                    # the nested-gazetteer pass. Spans come out sorted and may
                    # overlap.
                    doc_ex = self.span_tagger.doc_to_ex(
                        doc, context_labels=CONTEXT_LABELS)
                else:
                    doc_ex = doc_to_ex_expanded(doc, geo_labels=self.geo_labels,
                                                trim_spans=self.trim_spans)
                    if self.nested_gazetteer_pass:
                        doc_ex = doc_ex + nested_gazetteer_spans(doc, doc_ex,
                                                                 self.geonames)
                        doc_ex.sort(key=lambda e: e["start_char"])
            except Exception as e:
                logger.warning(f"Entity extraction failed for document: {e}")
                doc_ex = []
            all_doc_ex.append(doc_ex)

        # 2. ES lookups across all documents via a shared thread pool
        all_es_data = add_es_data_batch(
            all_doc_ex, self.geonames, max_results=max_choices,
            known_country=known_country, es_workers=es_workers,
            extra_features=bool(self.extra_feature_keys),
            outlet_homes=self._outlet_homes_for(len(docs), outlets))

        # 3. Cross-document model batching: pool all entities into one inference pass
        pooled_es_data = []
        entity_counts = []
        for es_data in all_es_data:
            entity_counts.append(len(es_data))
            pooled_es_data.extend(es_data)

        all_preds = None
        if pooled_es_data:
            dataset = ProductionData(pooled_es_data, max_choices=max_choices,
                                     oov_bucket_fix=self.oov_bucket_fix,
                                     feature_blocks=self.feature_blocks)
            data_loader = DataLoader(dataset=dataset, batch_size=64, shuffle=False)
            with torch.no_grad():
                self.model.eval()
                pred_val_list = []
                for input_batch in data_loader:
                    # Move the entire input batch to the model's device
                    input_batch_on_device = {k: v.to(self.model.device) for k, v in input_batch.items()}
                    pred_val_list.append(self.model(input_batch_on_device))
                all_preds = torch.cat(pred_val_list, dim=0)

        # 4. Split the predictions back out by document and resolve results
        results = []
        pred_offset = 0
        for doc, es_data, n_ents in zip(docs, all_es_data, entity_counts):
            output = {"doc_text": doc.text,
                     "event_location_raw": "",
                     "geolocated_ents": []}

            if n_ents == 0 or not es_data:
                results.append(output)
                continue

            pred_val = all_preds[pred_offset:pred_offset + n_ents]
            pred_offset += n_ents

            best_list = self._resolve_results(es_data, pred_val, debug,
                                              top_k=top_k)
            if (self.trim or trim) and best_list:
                self._trim_results(best_list)
            output["geolocated_ents"] = best_list
            results.append(output)

        return results

    def geoparse_doc(self,
                     text,
                     debug=False,
                     trim=True,
                     known_country=None,
                     max_choices=100,
                     outlet=None,
                     top_k=None):
        """
        Geoparse a single document.

        This is a convenience wrapper around the same pipeline that backs
        geoparse_batch(). For multiple documents, use geoparse_batch() instead:
        it batches the spaCy and model forward passes and shares ES lookups.

        Parameters
        ----------
        text : str or spacy Doc (with ._.tensor attributes)
            The text to geoparse.
        debug : bool
            If True, returns the top 4 results for each geoparsed location, rather than the single best.
            This is useful for debugging or collecting new annotations.
        trim : bool
            If True (default: True), removes some of the keys from the output dictionary that are only used
            internally for selecting the best geoparsed location. Including these keys is
            useful for debugging.
        known_country : str
            If provided, the geoparser will only consider locations in the given country.
        outlet : str or None
            The news outlet this document came from, as a domain
            ("parispi.net") or a URL. A local paper writes about its own patch,
            so knowing the masthead is worth +7.2 exact match on held-out LGL
            (experiments/campaign2/outlet_integration_report.md). Only used by
            a checkpoint trained with the `outlet` feature block; an unknown or
            absent outlet falls back to the block's null encoding at no cost.
        top_k : int or None
            When set, every entry in "geolocated_ents" also carries a
            "candidates" key: that mention's `top_k` best-scoring gazetteer
            candidates, each with its own calibrated "score", in rank order.
            Unlike `debug` this does not change the number of entries -- it is
            still one per mention -- so it is the key to reach for when
            something downstream has to show or explain the ranking. Mentions
            the model declined to place carry the list too.

        Returns
        -------
        output : dict
            Includes the following keys:
            - "doc_text": a string of the input text
            - "event_location_raw": str, always empty. Retained for backwards
              compatibility; event geolocation was removed in favor of the
              standalone event geolocation models.
            - "geolocated_ents": list of dicts, each dict is a geoparsed
              location. Every entry has "search_name", "start_char",
              "end_char", "no_match" and "p_no_match" (the model's probability
              that no candidate is right). Entries with "no_match": False also
              carry the geonames fields and "score", the calibrated
              probability of the chosen candidate; entries with
              "no_match": True carry nothing else, because the model declined
              to place the mention.

        Example
        -------
        >>> text = "The earthquake struck in the city of Christchurch, New Zealand."
        >>> geoparser.geoparse_doc(text)
        """
        if isinstance(text, spacy.tokens.doc.Doc):
            doc = text
        elif isinstance(text, str):
            doc = self.nlp(text)
        else:
            raise ValueError("Text must be either of type 'str' or 'spacy.tokens.doc.Doc'.")

        return self._geoparse_docs(
            [doc], max_choices=max_choices, known_country=known_country,
            trim=trim, debug=debug, outlets=[outlet], top_k=top_k)[0]

    def geoparse_batch(self, texts, batch_size=32, chunk_size=200,
                       es_workers=4, max_choices=100, known_country=None,
                       trim=True, debug=False, show_progress=False,
                       outlets=None, top_k=None):
        """
        Geoparse multiple documents with optimized batching.

        Uses three layers of optimization:
        1. spaCy batching via nlp.pipe() for transformer forward passes
        2. Cross-document threaded ES lookups via a shared thread pool
        3. Cross-document model batching (all entities from a chunk in one inference pass)

        Parameters
        ----------
        texts : list of str
            Documents to geoparse.
        batch_size : int
            Batch size for spaCy's nlp.pipe() transformer inference. Default: 32.
        chunk_size : int
            Number of documents per processing chunk (bounds memory). Default: 200.
        es_workers : int
            Thread pool size for parallel ES lookups. Default: 4.
        max_choices : int
            Maximum ES candidates per entity. Default: 100.
        known_country : str or None
            Restrict results to a single country (ISO 3166-1 alpha-3).
        trim : bool
            Remove internal keys from output. Default: True.
        debug : bool
            Return the top 4 candidates per entity. Default: False.
        show_progress : bool
            Show tqdm progress bar. Default: False.
        outlets : list of (str or None), or None
            One outlet domain per document, in the same order as `texts`. See
            geoparse_doc's `outlet`. None (the default) means no document has a
            known outlet.

        Returns
        -------
        list of dicts
            One result dict per input document. Each dict has the same structure
            as the output of geoparse_doc(): keys "doc_text", "event_location_raw",
            and "geolocated_ents".
        """
        all_results = []
        if outlets is not None:
            outlets = list(outlets)
            if len(outlets) != len(texts):
                raise ValueError(f"got {len(outlets)} outlets for {len(texts)} "
                                 "texts; pass one per document (None is fine)")
        self.geonames.clear_cache()  # fresh cache per geoparse_batch() run
        progress = tqdm(total=len(texts), desc="Geoparsing",
                        disable=not show_progress)

        for chunk_start in range(0, len(texts), chunk_size):
            chunk_texts = texts[chunk_start:chunk_start + chunk_size]
            chunk_outlets = (None if outlets is None
                             else outlets[chunk_start:chunk_start + chunk_size])

            # Layer 1: spaCy batching
            docs = []
            for doc in self.nlp.pipe(chunk_texts, batch_size=batch_size):
                docs.append(doc)
                progress.update(1)

            # Layers 2-3: ES lookups, model inference, result resolution
            try:
                chunk_results = self._geoparse_docs(
                    docs, max_choices=max_choices, known_country=known_country,
                    trim=trim, debug=debug, es_workers=es_workers,
                    outlets=chunk_outlets, top_k=top_k)
            except Exception as e:
                logger.error(f"Chunk processing failed: {e}")
                chunk_results = [
                    {"doc_text": doc.text, "event_location_raw": "",
                     "geolocated_ents": [], "error": str(e)}
                    for doc in docs
                ]

            all_results.extend(chunk_results)

        progress.close()
        return all_results


def add_es_data(ex, 
                geonames_service: GeonamesService, 
                max_results=50, 
                fuzzy=0, 
                limit_types=False,
                remove_correct=False,
                known_country=None,
                extra_features=False):
    """
    Run an Elasticsearch/geonames query for a single example and add the results
    to the object.

    Parameters
    ---------
    ex: dict
      output of doc_to_ex_expanded
    conn: elasticsearch connection
    max_results: int
      Maximum results to bring back from ES
    fuzzy: int
      Allow fuzzy results? 0=exact matches. Higher numbers will
      increase the fuzziness of the search. 
    remove_correct: bool
        If True, remove the correct result from the list of results.
        This is useful for training a model to handle "none of the above"
        cases.
    extra_features: bool
        If True, add the enrichment features (see mordecai3.candidate_features)
        that depend on the mention and its own candidate set. The document-level
        features need every entity in the document, so they are only added by
        add_es_data_batch/add_es_data_doc.

    Examples
    --------
    ex = {"search_name": ent.text,
         "tensor": tensor,
         "doc_tensor": doc_tensor,
         "locs_tensor": locs_tensor,
         "sent": ent.sent.text,
         "in_rel": in_rel,    # this comes from the heuristic `guess_in_rel` fuction defined in geoparse.py
         "start_char": ent[0].idx,
         "end_char": ent[-1].idx + len(ent.text)}
    d_es = add_es_data(d)
    # d_es now has a "es_choices" key and a "correct" key that indicates which geonames 
    # entry was the correct one.
    """
    max_results = int(max_results)
    fuzzy = int(fuzzy)
    search_name = ex['search_name']

    cache_key = _es_cache_key(ex, max_results, fuzzy, limit_types, known_country,
                              extra_features)
    cache = geonames_service._es_cache
    if cache_key in cache:
        # Deep-copy because downstream code mutates choices (adm1_count, country_count)
        logger.debug(f"ES cache hit for '{search_name}'")
        return _finish_es_example(ex, _copy_choices(cache[cache_key]), remove_correct)

    # if we detect a parent location using our heuristic (see `guess_in_rel` in geoparse.py),
    # check to see if that's a country or admin1.
    if 'in_rel' in ex.keys():
        if ex['in_rel']:
            parent_place = geonames_service.get_country_by_name(ex['in_rel'])
            if not parent_place:
                parent_place = geonames_service.get_adm1_country_entry(ex['in_rel'], None)
        else:
            parent_place = None
    else:
        parent_place = None

    search_res = geonames_service.search_by_name(search_name, max_results, fuzzy, limit_types, known_country)
    choices = res_formatter(search_res, search_name, parent_place, extra_features)

    # Always try a fuzzy search if no results from previous search, to avoid
    # having no candidates for the ML model to choose from.
    if not choices:
        search_res = geonames_service.search_by_name(search_name, max_results, fuzzy+1, limit_types, known_country)
        choices = res_formatter(search_res, ex['search_name'], parent_place, extra_features)

    logger.debug("Adding NULL choice")
    choices.append(_null_choice(search_name if extra_features else None))

    # Cache a copy (downstream code mutates dicts via adm1_count/country_count)
    cache[cache_key] = _copy_choices(choices)
    logger.debug(f"ES cache miss for '{search_name}', cached {len(choices)} choices")

    return _finish_es_example(ex, choices, remove_correct)


def _es_cache_key(ex, max_results, fuzzy, limit_types, known_country,
                  extra_features=False):
    """The inputs that determine an entity's ES candidate list.

    Repeated place names are extremely common within and across documents, so
    this is what lets both the cache and the batched path collapse duplicate
    work. The value it keys is the candidate list *before* `remove_correct` is
    applied, so it stays reusable either way.

    `extra_features` is part of the key because the entity-level enrichment
    features are computed once and cached with the candidate list: two callers
    that disagree about whether they want them must not share an entry.
    """
    return (ex['search_name'], ex.get('in_rel', '') or '', known_country or '',
            int(max_results), int(fuzzy), limit_types, bool(extra_features))


def _copy_choices(choices):
    """Copy a candidate list so callers can mutate it independently.

    Candidate dicts are flat -- every value is a str/int/float -- so a per-dict
    shallow copy is equivalent to a deepcopy here and much cheaper. This is on
    the hot path: every entity gets its own copy of ~100 candidates, which was
    the single largest cost in the lookup stage once queries were batched.
    """
    return [dict(c) for c in choices]


def _null_choice(search_name=None):
    """A fresh "none of the above" candidate, always appended last.

    Pass `search_name` to also give it the enrichment features, at the neutral
    values that keep the per-entity feature arrays rectangular without making
    "no answer" look well-supported.
    """
    choice = {'feature_code': 'NULL',
              'feature_class': 'NULL',
              'country_code3': 'NULL',
              'lat': 0,
              'lon': 0,
              'name': 'NULL',
              'admin1_code': 'NULL',
              'admin1_name': 'NULL',
              'admin2_code': 'NULL',
              'admin2_name': 'NULL',
              'geonameid': 'NULL',
              'population': 0,
              'admin1_parent_match': -1,
              'country_code_parent_match': -1,
              'alt_name_length': 0,
              'min_dist': 99.0,
              'max_dist': 99.0,
              'avg_dist': 99.0,
              'ascii_dist': 99.0,
              'adm1_count': 0.0,
              'country_count': 0.0}
    if search_name is not None:
        fill_null_features(choice, mention_admin_cue(search_name))
    return choice


def _finish_es_example(ex, choices, remove_correct):
    """Attach candidate choices to an example and mark which one is correct.

    Split out of add_es_data so the cache-hit and cache-miss paths stay identical.
    """
    if remove_correct:
        choices = [c for c in choices if c['geonameid'] != ex['correct_geonamesid']]

    ex['es_choices'] = choices

    if remove_correct:
        ex['correct'] = [False for c in choices]
    else:
        if 'correct_geonamesid' in ex.keys():
            ex['correct'] = [c['geonameid'] == ex['correct_geonamesid'] for c in choices]
    return ex


def _add_cross_entity_counts(doc_es):
    """Add the within-document admin1/country co-occurrence counts to each candidate.

    These are features for the model, so they can only be computed once every entity
    in the document has its ES results back.
    """
    admin1_count = make_admin1_counts(doc_es)
    country_count = make_country_counts(doc_es)
    for i in doc_es:
        for e in i['es_choices']:
            e['adm1_count'] = admin1_count[e['admin1_name']]
            e['country_count'] = country_count[e['country_code3']]


def _resolve_parents_batch(in_rel_names, geonames_service):
    """Resolve "in" relations to a parent place, batching the lookups.

    Mirrors the sequential logic in add_es_data: try country first, fall back to
    ADM1 with no country filter. Returns {in_rel value: parent entry or None}.
    """
    names = [n for n in in_rel_names if n]
    if not names:
        return {}
    countries = geonames_service.get_country_by_name_batch(names)
    missing = [n for n, v in countries.items() if not v]
    adm1s = geonames_service.get_adm1_country_entry_batch(missing) if missing else {}
    return {n: (countries[n] or adm1s.get(n)) for n in countries}


def add_es_data_doc(doc_ex, geonames_service: GeonamesService, max_results=50, fuzzy=0,
                    limit_types=False, remove_correct=False, known_country=None,
                    es_workers=None, extra_features=False):
    """Add ES candidates for every entity in one document.

    Thin wrapper over add_es_data_batch so there is a single lookup path.

    es_workers is accepted for backwards compatibility and ignored: lookups are
    now batched into _msearch requests, so concurrency is handled by ES rather
    than by a client-side thread pool.
    """
    if not doc_ex:
        return []
    return add_es_data_batch([doc_ex], geonames_service, max_results, fuzzy,
                             limit_types, remove_correct, known_country,
                             extra_features=extra_features)[0]


def add_es_data_batch(all_doc_ex, geonames_service: GeonamesService, max_results=50,
                      fuzzy=0, limit_types=False, remove_correct=False,
                      known_country=None, es_workers=None, extra_features=False,
                      outlet_homes=None):
    """Look up ES candidates for every entity across many documents.

    All lookups for the batch are bundled into a small number of _msearch
    requests. Each sub-query is executed by ES exactly as it would have been on
    its own, so results are identical to looping over add_es_data -- the win is
    that per-request overhead (HTTP, JSON parsing, client object construction)
    is paid once per chunk rather than once per entity.

    Work is deduplicated by cache key before anything is sent. The sequential
    path got that for free, because the cache filled in as it went; here every
    lookup is planned up front, so duplicates have to be collapsed explicitly.

    Parameters
    ----------
    all_doc_ex : list of list of dicts
        Entity dicts per document (output of doc_to_ex_expanded for each doc).
    geonames_service : GeonamesService
    es_workers : ignored
        Accepted for backwards compatibility. See add_es_data_doc.
    extra_features : bool
        If True, every candidate also gets the enrichment features that the
        `--feature-blocks` models are trained on (see
        mordecai3.candidate_features). Off by default, so a checkpoint that
        predates them sees exactly the candidate dicts it always did.
    outlet_homes : list or None
        One resolved outlet home per document (see
        mordecai3.outlet_features.add_outlet_features), or None to leave the
        `outlet` block off entirely. A `None` entry means "this document has no
        outlet" and is written to the block's null encoding, which is a
        different thing from passing None here: a checkpoint trained with the
        block needs those five keys on every candidate.

    Returns
    -------
    list of list of dicts
        ES-enriched entity data, one list per document.
    """
    # Callers reach this from a CLI as often as from library code, so coerce
    # rather than trusting the caller: `fuzzy + 1` in the retry below turns a
    # stringy "0" into a TypeError several hundred lookups deep.
    max_results = int(max_results)
    fuzzy = int(fuzzy)

    if outlet_homes is not None and len(outlet_homes) != len(all_doc_ex):
        raise ValueError(f"got {len(outlet_homes)} outlet homes for "
                         f"{len(all_doc_ex)} documents")

    tasks = [(doc_idx, ent_idx, ex)
             for doc_idx, doc_ex in enumerate(all_doc_ex)
             for ent_idx, ex in enumerate(doc_ex)]
    if not tasks:
        return [[] for _ in all_doc_ex]

    cache = geonames_service._es_cache

    # 1. Collapse duplicate lookups. Many entities across a batch share a name.
    by_key = {}
    for doc_idx, ent_idx, ex in tasks:
        key = _es_cache_key(ex, max_results, fuzzy, limit_types, known_country,
                            extra_features)
        by_key.setdefault(key, []).append((doc_idx, ent_idx, ex))

    todo = [k for k in by_key if k not in cache]
    logger.debug(f"{len(tasks)} entities -> {len(by_key)} unique lookups, "
                 f"{len(todo)} not cached")

    if todo:
        # 2. Parent lookups. These only feed res_formatter, never the name query
        #    itself, so they are independent of the searches below.
        reps = [by_key[k][0][2] for k in todo]
        parents = _resolve_parents_batch([ex.get('in_rel') for ex in reps],
                                         geonames_service)

        def parent_for(ex):
            return parents.get(ex.get('in_rel') or '')

        # 3. One batched round of name searches.
        specs = [(k[0], max_results, fuzzy, limit_types, known_country) for k in todo]
        responses = geonames_service.search_by_names(specs)

        fresh = {}
        retry = []
        for k, ex, res in zip(todo, reps, responses):
            choices = res_formatter(res, k[0], parent_for(ex), extra_features)
            if choices:
                fresh[k] = choices
            else:
                retry.append((k, ex))

        # 4. Fuzzy retry for names that came back empty, so the model always has
        #    something to choose from. In practice this is ~1% of entities.
        if retry:
            logger.debug(f"fuzzy retry for {len(retry)} names")
            specs = [(k[0], max_results, fuzzy + 1, limit_types, known_country)
                     for k, _ in retry]
            for (k, ex), res in zip(retry, geonames_service.search_by_names(specs)):
                fresh[k] = res_formatter(res, k[0], parent_for(ex), extra_features)

        for k, choices in fresh.items():
            choices.append(_null_choice(k[0] if extra_features else None))
            # Stored pristine; every consumer takes its own copy below, since
            # downstream code mutates adm1_count/country_count in place.
            cache[k] = choices

    # 5. Hand each entity its own copy and reassemble per document.
    results_by_doc = {i: [] for i in range(len(all_doc_ex))}
    for key, members in by_key.items():
        pristine = cache.get(key)
        if pristine is None:
            logger.warning(f"no ES candidates resolved for '{key[0]}', skipping")
            continue
        for doc_idx, ent_idx, ex in members:
            done = _finish_es_example(ex, _copy_choices(pristine), remove_correct)
            results_by_doc[doc_idx].append((ent_idx, done))

    all_doc_es = []
    for doc_idx in range(len(all_doc_ex)):
        entries = sorted(results_by_doc[doc_idx], key=lambda x: x[0])
        doc_es = [r for _, r in entries]
        if doc_es:
            _add_cross_entity_counts(doc_es)
            if extra_features:
                # The sibling and anchor-geometry features read the other
                # mentions in the document, so like the co-occurrence counts
                # above they can only be computed now that every entity in the
                # document has its candidates back.
                add_document_features(doc_es)
            if outlet_homes is not None:
                # Also a document-level feature, and the same functions the
                # enrichment calls, so train/serve parity is by construction
                # (tests/test_feature_parity.py polices it). A document with no
                # outlet gets the null encoding rather than missing keys.
                home = outlet_homes[doc_idx]
                for ent in doc_es:
                    if home is None:
                        clear_outlet_features(ent["es_choices"])
                    else:
                        add_outlet_features(ent["es_choices"], home)
        all_doc_es.append(doc_es)

    return all_doc_es


def res_formatter(res, search_name, parent=None, extra_features=False):
    """
    Helper function to format the ES/Geonames results into a format for the ML model, including
    edit distance statistics and parent matches.

    Parameters
    ----------
    res: Elasticsearch/Geonames output
    search_name: str
      The original search term from the document
    parent: dict
      Geonames/ES entry for the inferred parent
    extra_features: bool
      Also compute the enrichment features that depend on the mention and its
      own candidate set (see mordecai3.candidate_features). The population and
      name lists they need are already in the ES hits, so this costs no extra
      queries -- and none of those raw fields are kept on the returned dicts.

    Returns
    -------
    choices: list
      List of formatted Geonames results, including edit distance statistics
    """
    # choices is our eventual output, a list of dicts, each of which is a formatted Geonames result
    choices = []
    sources = []
    alt_lengths = []
    min_dist = []
    max_dist = []
    avg_dist = []
    ascii_dist = []
    # iterate through the docs returned by ES. hit_sources handles both the
    # raw dicts from the batched _msearch path and the elasticsearch_dsl
    # objects from a single .execute().
    for i in hit_sources(res):
        names = [i['name']] + i['alternativenames'] 
        dists = [jellyfish.levenshtein_distance(search_name, j) for j in names]
        lat, lon = i['coordinates'].split(",")
        d = {"feature_code": i['feature_code'],
            "feature_class": i['feature_class'],
            "country_code3": i['country_code3'],
            "lat": float(lat),
            "lon": float(lon),
            "name": i['name'],
            "admin1_code": i['admin1_code'],
            "admin1_name": i['admin1_name'],
            "admin2_code": i['admin2_code'],
            "admin2_name": i['admin2_name'],
            "geonameid": i['geonameid'],
            # Not a model feature in its raw form -- `candidate_features`
            # derives log_population/is_max_pop from the same ES field -- but
            # the single most useful number for a human deciding between two
            # candidates with the same name, so it is kept on the dict.
            "population": int(i.get('population') or 0)}
        # if we detect a parent country or ADM1, add the parent match features
        if parent: 
            if parent['admin1_name'] == "":
                d['admin1_parent_match'] = 0
            elif parent['admin1_name'] == i['admin1_name']:
                d['admin1_parent_match'] = 1
            else:
                d['admin1_parent_match'] = -1

            if parent['country_code3'] == "":
                d['country_code_parent_match'] = 0
            elif parent['country_code3'] == i['country_code3']:
                d['country_code_parent_match'] = 1
            else:
                d['country_code_parent_match'] = -1
        else:
            d['admin1_parent_match'] = 0
            d['country_code_parent_match'] = 0

        choices.append(d)
        if extra_features:
            sources.append(i)
        alt_lengths.append(len(i['alternativenames'])+1)
        min_dist.append(np.min(dists))
        max_dist.append(np.max(dists))
        avg_dist.append(np.mean(dists))
        ascii_dist.append(jellyfish.levenshtein_distance(search_name, i['asciiname']))
    alt_lengths = np.log(alt_lengths)
    min_dist = normalize(min_dist)
    max_dist = normalize(max_dist)
    avg_dist = normalize(avg_dist)
    ascii_dist = normalize(ascii_dist)

    for n, i in enumerate(choices):
        i['alt_name_length'] = alt_lengths[n]
        i['min_dist'] = min_dist[n]
        i['max_dist'] = max_dist[n]
        i['avg_dist'] = avg_dist[n]
        i['ascii_dist'] = ascii_dist[n]
    if extra_features:
        add_entity_features(search_name, choices, sources)
    return choices


def make_admin1_counts(out):
    """
    Get the ADM1s from all candidate results for all locations in a document and return
    the count of each ADM1. This allows us to prefer candidates that share an ADM1 with other
    locations in a document.
    
    This is getting at roughly the same info as previous (slow) approaches that tried to minimize
    the distance between the returned geolocations.

    Parameters
    ---------
    out: list of dicts
      List of place names from the document with candidate geolocations
      from ES/Geonames

    Returns
    -------
    admin1_count: dict
      A dictionary {adm1: count}, where count is the proportion of place names in the
      document that have at least one candidate entry from this adm1.
    """
    admin1s = []

    # for each entity, get the unique ADM1s from the search results 
    for es in out:
        other_adm1 = set([i['admin1_name'] for i in es['es_choices']])
        admin1s.extend(list(other_adm1))
    
    # TODO: handle the "" admins here.
    admin1_count = dict(Counter(admin1s))
    for k, v in admin1_count.items():
        admin1_count[k] = v / len(out)
    return admin1_count

def make_country_counts(out):
    """Take in a document's worth of examples and return the count of countries"""
    all_countries = []
    for es in out:
        countries = set([i['country_code3'] for i in es['es_choices']])
        all_countries.extend(list(countries))
    
    country_count = dict(Counter(all_countries))
    for k, v in country_count.items():
        country_count[k] = v / len(out)
        
    return country_count


def normalize(ll: list[float]) -> npt.NDArray[np.float64]:    
    """Normalize an array to [0, 1]"""
    arr = np.array(ll)
    if len(arr) > 0:
        max_arr = np.max(arr)
        if max_arr == 0:
            max_arr = 0.001
        arr = (arr - np.min(arr)) / max_arr
    return arr

