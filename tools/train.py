import glob
import hashlib
import json
import os
import pickle
import random
import re
from collections import Counter

import jsonlines

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import datetime
import logging

import mordecai3.elasticsearch as es_util
from mordecai3.geonames import GeonamesService
import numpy as np
import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import typer
import wandb
import xmltodict
from error_utils import make_wandb_dict
from mordecai3.geoparse import guess_in_rel, add_es_data_batch
from mordecai3.outlet_features import OUTLET_KEYS, outlet_null_value

from mordecai3.torch_model import geoparse_model

from mordecai3.mordecai_utilities import fast_docbin_io, spacy_doc_setup
from spacy.tokens import DocBin
from torch.utils.data import DataLoader
from mordecai3.torch_model import (ALL_FEATURE_KEYS, FEATURE_BLOCKS, TrainData,
                                   expand_feature_blocks, geoparse_model)
from tqdm import tqdm

logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter(
        '%(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for i in loggers:
    if re.search(r"NGEC\.", i.name):
        i.addHandler(handler) 
        i.setLevel(logging.INFO)
        i.propagate = False
    if re.search("elasticsearch", i.name):
        i.addHandler(handler) 
        i.setLevel(logging.WARNING)
    if re.search("urllib3", i.name):
        i.addHandler(handler) 
        i.setLevel(logging.WARNING)

spacy_doc_setup()
# later, after loading the nlp object, don't forget to run this:
# nlp.add_pipe("token_tensors")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def masked_smoothed_ce(pred, label, mask, eps, weights=None):
    """Cross entropy with the smoothing mass spread over live candidates only.

    `nn.CrossEntropyLoss(label_smoothing=eps)` puts eps/K on every one of the K
    classes. With --mask-padding the padded classes sit at -1e9, so each of them
    contributes about eps/K * 1e9 to the loss: the objective blows up to ~1e7,
    the gradient chases the padding, and accuracy collapses (measured: 0.71-0.78
    exact match). The smoothing is meant to spread doubt over the *candidates*,
    so it is spread over the rows the mask says are real.

    `weights` (--abstain-weight) reweights individual examples. It is None on
    every run in the campaign, and the unweighted branch is byte-for-byte the
    original reduction, so the default stays bit-identical.
    """
    logp = F.log_softmax(pred, dim=1)
    nll = -logp.gather(1, label.unsqueeze(1)).squeeze(1)
    # mask is 0.0 on padded rows, so their -1e9 log-prob drops out here.
    smooth = -(logp * mask).sum(1) / mask.sum(1)
    per_example = (1 - eps) * nll + eps * smooth
    if weights is None:
        return per_example.mean()
    return (per_example * weights).sum() / weights.sum()


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.argmax(y_pred, axis=1)
    #y_test_cat = torch.argmax(y_test, axis=1) 
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_pred.shape[0]
    acc = torch.round(acc * 100)
    return acc

def read_file(fn):
    if re.search("xml", fn):
        with open(fn, "r", encoding='utf-8') as f:
            xml = f.read()
            data = xmltodict.parse(xml)
    elif re.search("jsonl", fn):
        with jsonlines.open(fn, "r") as f:
            data = list(f.iter())
    else:
        raise NotImplementedError("Don't know how to handle this filetype")
    return data 

def _as_list(x):
    """xmltodict collapses a single repeated child into a bare dict.

    An article with exactly one <toponym> therefore came back as a dict, and
    iterating it yielded its *keys*; every such toponym was lost to the broad
    `except` below with "string indices must be integers".
    """
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


def split_list(data, frac=0.7):
    split = round(frac*len(data))
    return data[0:split], data[split:]


def doc_key(ex):
    """The document an entity belongs to.

    The pickles carry no document id, but every entity of a document shares the
    same `doc_tensor` (the mean of its token vectors), so hashing it recovers
    document membership exactly -- the trick the Wave-2b sibling features and
    tools/end_to_end_eval.py's `heldout_doc_indices` already rely on.
    """
    t = ex.get('doc_tensor')
    if t is None:
        return None
    return hashlib.md5(np.asarray(t).tobytes()).hexdigest()


def split_by_doc(data, frac=0.7):
    """Assign whole documents to train/held-out, deterministically.

    The default split cuts the flat entity list at 70%, which is a document
    boundary only because the entities happen to be in document order -- and
    for Synth, which is shuffled before the split, it is not one at all: 14% of
    its held-out documents also appear in training
    (experiments/campaign2/data_quality_report.md). Keying on the document
    itself makes the split independent of order, so no document can be on both
    sides, and it is stable across sources, seeds and reruns.

    This is NOT the default: every frozen number in `experiments/` is on the
    positional split, and the two are not comparable (decision D1 freezes the
    current held-out sets as DEV).
    """
    train, val = [], []
    for ex in data:
        k = doc_key(ex)
        # A hash bucket, not a shuffle: the same document lands on the same
        # side in every source, every run.
        h = int(k[:8], 16) / float(1 << 32) if k is not None else 1.0
        (train if h < frac else val).append(ex)
    return train, val


# Everything the training path ever reads off a candidate dict: the model
# features (torch_model.ProductionData), the enrichment features, and the keys
# error_utils.evaluate_results scores with. The enriched pickles carry ~50 keys
# per candidate, including long name strings, and a run holds ~29,000 entities x
# 500 candidates of them: keeping only these takes peak RSS from ~40 GB to
# something that fits five runs on one box.
# What still has to be a dict on each candidate after compaction: the fields
# the model looks up by name and the ones error_utils.evaluate_results reports.
CANDIDATE_KEYS_KEPT = ("feature_code", "feature_class", "country_code3",
                       "admin1_code", "lat", "lon", "geonameid")


def compact_candidates(es_data):
    """Collapse each entity's candidate features into one float32 matrix.

    An enriched candidate is a ~50-key dict, and a run holds ~14 million of
    them: the dict headers alone are ~25 GB, which is what kept these runs to
    one at a time. The numeric features become `ex['feat_matrix']`, a
    (candidates, len(ALL_FEATURE_KEYS)) float32 array in a fixed column order,
    and the dicts keep only the keys that are still looked up by name.
    ProductionData._gaz_from_matrix reads the matrix and produces exactly the
    array the dict path produced.
    """
    n_feat = len(ALL_FEATURE_KEYS)
    # The `outlet` block is attached by a separate pass over the *already*
    # enriched pickles (`enrich_pickles.py --outlet-only`), because it needs
    # the corpus XML's <domain> and not Elasticsearch. So a candidate straight
    # out of `es_formatted_*_enriched.pkl` has no outlet keys, and compacting
    # one has to mean "this document has no outlet" -- the block's well-defined
    # null -- rather than KeyError. A cache built this way trains the outlet
    # block on nothing but nulls, which is why the outlet pass has to be rerun
    # whenever the compact caches are rebuilt; the log line below says so.
    missing_outlet = [ex for ex in es_data if ex['es_choices']
                      and OUTLET_KEYS[0] not in ex['es_choices'][0]]
    if missing_outlet:
        logger.warning(
            f"{len(missing_outlet)} entities carry no outlet features; "
            f"compacting them to the no-outlet null encoding. Rerun "
            f"`tools/enrich_pickles.py --outlet-only` if you meant to train "
            f"the outlet block on this data.")
    outlet_null = {k: outlet_null_value(k) for k in OUTLET_KEYS}
    for ex in es_data:
        choices = ex['es_choices']
        fm = np.empty((len(choices), n_feat), dtype=np.float32)
        for n, c in enumerate(choices):
            fm[n] = [c[k] if k in c else outlet_null[k] for k in ALL_FEATURE_KEYS]
        ex['feat_matrix'] = fm
        ex['es_choices'] = [{k: c[k] for k in CANDIDATE_KEYS_KEPT if k in c}
                            for c in choices]
    return es_data


def load_es_data(data_dir, 
             max_results, 
             limit_types, 
             fuzzy,
             batch_size,
             test_batch_size,
             train_frac=0.7,
             data_sources=["Prodigy", "TR", "LGL", "GWN", "Synth", "Wiki"],
             source_limits=None,
             oov_bucket_fix=False,
             enriched=False,
             feature_blocks=None,
             full_null_row=False,
             pickle_suffix="",
             window=None,
             split_mode="entity"):
    """
    Load formatted training data with Elasticsearch results

    Parameters
    --------
    data_dir: Path
      location of pickled data with Elasticsearch info
    max_results: int
      read in the data with the right number of results
    limit_types: str 
      Either 'all_loc_types' to allow all types of features or 'pa_only' to limit to cities
      and administrative units (that is, excluding geographic features)
    fuzzy: int
      Fuzzy ES search? 0=none, 1=some, etc
    pickle_suffix: str
      Variant of the enriched pickles to train on, appended after `_enriched`.
      "" (the default) is the frozen enrichment; "_r2" is the A/P label rewrite
      written by tools/rewrite_labels.py. Only the labels differ.
    window: int or None
      Number of candidate rows the model scores. `max_results` names the pickle
      files (they were built with 500 hits per mention) and used to double as
      the window; `window` separates the two so a 500-candidate pickle can be
      trained at the 100-row window `Geoparser` actually serves. None keeps the
      old behavior exactly: window == max_results.
    split_mode: str
      "entity" (the default, and every frozen number in `experiments/`) cuts
      each source's flat entity list at `train_frac`. "doc" assigns whole
      documents by a hash of their `doc_tensor`, so no document straddles the
      split -- see `split_by_doc`. The two splits are not comparable.

    Returns
    -------
    list
      a list of formatted, shuffled training data
    """
    # The `_enriched` pickles hold the same entities, candidates and labels as
    # the originals, with 30 extra keys per candidate (tools/enrich_pickles.py).
    # `pickle_suffix` selects a label variant of those same pickles (see
    # tools/rewrite_labels.py); "" keeps the frozen enrichment.
    sfx = f"_enriched{pickle_suffix}" if enriched else ""
    if enriched and os.path.isdir(f'{data_dir}/pickled_es'):
        # A compacted cache (see the `compact-cache` command) holds the same
        # entities with their candidate features already in one float32 matrix.
        # It is what makes several runs fit on the box at once. The cache key
        # includes the suffix, so a label variant never reads the base cache.
        probe = (f'{data_dir}/pickled_es/es_formatted_tr_{max_results}_{limit_types}'
                 f'_fuzzy_{fuzzy}_enriched{pickle_suffix}_compact.pkl')
        if os.path.exists(probe):
            sfx = f"_enriched{pickle_suffix}_compact"
            logger.info(f"Using the compacted enriched pickles ({sfx})")
    es_train_data = [] 
    data_loaders = []
    val_datasets = []

    for source in data_sources:
        logger.info(f"Loading data for {source}")
        if source == 'Prodigy':
            with open(f'{data_dir}/pickled_es/es_formatted_prodigy_{max_results}_{limit_types}_fuzzy_{fuzzy}{sfx}.pkl', 'rb') as f:
                es_data = pickle.load(f)
        elif source == "TR":
            with open(f'{data_dir}/pickled_es/es_formatted_tr_{max_results}_{limit_types}_fuzzy_{fuzzy}{sfx}.pkl', 'rb') as f:
                es_data = pickle.load(f)
        elif source == "LGL":
            with open(f'{data_dir}/pickled_es/es_formatted_lgl_{max_results}_{limit_types}_fuzzy_{fuzzy}{sfx}.pkl', 'rb') as f:
                es_data = pickle.load(f)
        elif source == "GWN":
            with open(f'{data_dir}/pickled_es/es_formatted_gwn_{max_results}_{limit_types}_fuzzy_{fuzzy}{sfx}.pkl', 'rb') as f:
                es_data = pickle.load(f)
        elif source == "Synth":
            # this one's a little different bc there are two files
            with open(f'{data_dir}/pickled_es/es_formatted_syn_cities_{max_results}_{limit_types}_fuzzy_{fuzzy}{sfx}.pkl', 'rb') as f:
                es_data_syn1 = pickle.load(f)
            with open(f'{data_dir}/pickled_es/es_formatted_syn_caps_{max_results}_{limit_types}_fuzzy_{fuzzy}{sfx}.pkl', 'rb') as f:
                es_data_syn_caps= pickle.load(f)
            random.seed(617)
            random.shuffle(es_data_syn1)
            random.shuffle(es_data_syn_caps)
            # combine both syn datasets and split
            es_data = es_data_syn1[0:500] + es_data_syn_caps[0:500]
        elif source == "Wiki":
            with open(f'{data_dir}/pickled_es/es_formatted_wiki_{max_results}_{limit_types}_fuzzy_{fuzzy}{sfx}.pkl', 'rb') as f:
                es_data = pickle.load(f)
                logger.debug(f"Total wiki results: {len(es_data)}")
        elif source in ("WikiDocs", "WikiDocsFull"):
            stem = "wiki_docs" if source == "WikiDocs" else "wiki_docs_full"
            # Written as one pickle per spaCy shard; concatenating them in
            # filename order restores the document order they were read in, so
            # the positional train/test split still splits by article.
            # Two explicit patterns rather than one `{stem}*` glob: the
            # sharded names for `wiki_docs` are `wiki_docs.000_...`, and a
            # trailing wildcard would also swallow every `wiki_docs_full_...`.
            tail = f'_{max_results}_{limit_types}_fuzzy_{fuzzy}{sfx}.pkl'
            base = f'{data_dir}/pickled_es/es_formatted_{stem}'
            files = sorted(glob.glob(f'{base}.[0-9][0-9][0-9]{tail}')) or \
                sorted(glob.glob(f'{base}{tail}'))
            if not files:
                raise FileNotFoundError(f"No pickled {stem} data at {base}*{tail}")
            es_data = []
            for fn in files:
                with open(fn, 'rb') as f:
                    es_data.extend(pickle.load(f))
            logger.info(f"Total {stem} results: {len(es_data)} from {len(files)} shard(s)")
            # mean of 'correct' key
            #np.mean([np.mean(i['correct']) for i in es_data])

        # Guard against malformed tensors. This used to drop real examples: the
        # old token_tensors fell back to a scalar/0-d value on some tokens, so
        # "some sort of bug in the spacy step" was this pipeline's own. It now
        # drops nothing across all 15,208 entities -- kept as a cheap assertion.
        es_data = [i for i in es_data if len(i['tensor']) > 1]
        if 'feat_matrix' not in es_data[0]:
            es_data = compact_candidates(es_data)
        if split_mode == "doc":
            es_data, es_data_val = split_by_doc(es_data, train_frac)
        else:
            es_data, es_data_val = split_list(es_data, train_frac)
        # Capping happens after the split, so the held-out set for a capped
        # source is the same one an uncapped run is scored on.
        limit = (source_limits or {}).get(source)
        if limit is not None and len(es_data) > limit:
            logger.info(f"Capping {source} training examples at {limit} "
                        f"(from {len(es_data)})")
            # Sampled, not truncated. The examples are in document order, so a
            # prefix of a wiki corpus is a few hundred articles about a few
            # events -- which measures the effect of *less* data confounded
            # with the effect of *narrower* data.
            es_data = random.Random(4242).sample(es_data, limit)
        logger.info(f"Training examples from {source}: {len(es_data)}, "
                    f"held out: {len(es_data_val)}")
        es_train_data.extend(es_data)
        val_datasets.append(es_data_val)
        dataset = TrainData(es_data_val, max_choices=(window or max_results),
                            oov_bucket_fix=oov_bucket_fix,
                            feature_blocks=feature_blocks,
                            full_null_row=full_null_row)
        loader = DataLoader(dataset=dataset, batch_size=test_batch_size, shuffle=False)
        data_loaders.append(loader)

    # now make one loader for all training data
    random.seed(617)
    random.shuffle(es_train_data)
    train_data = TrainData(es_train_data, max_choices=(window or max_results),
                           oov_bucket_fix=oov_bucket_fix,
                           feature_blocks=feature_blocks,
                           full_null_row=full_null_row)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    # The training candidates have been turned into tensors and are never read
    # again -- only `len()` and the tensor width of the first entity are. The
    # held-out sets keep theirs, because evaluate_results scores through them.
    # This is the single biggest live allocation in a run (~11M dicts).
    for ex in es_train_data:
        ex['es_choices'] = []

    #debugging
    #[len(i['tensor']) for i in es_train_data]
    ######

    return train_loader, es_train_data, data_loaders, val_datasets


def data_formatter_prodigy(docs, data):
    """
    Format the annotated documents from the Prodigy training round into a format for training.
    
    Returns a list of lists, with one list for each document, consisting of each entity
    within the document. This round of training only annotated one location per
    "document"/sentence, so non-annotated entities have None as their value for
    correct_geonamesid. These will be discarded later.

    Parameters
    ---------
    docs: list of spaCy docs
    data: list of dicts
    source: the short name of the source used

    Returns
    -------
    all_formatted: list of lists
      The list is of length docs, which each element a list of all the place 
      names within the document.
    """
    all_formatted = []
    doc_num = 0
    for doc, ex in tqdm(zip(docs, data), total=len(docs), leave=False): 
        doc_formatted = []
        # Check if the example is good
        if ex['answer'] in ['reject', 'ignore']:
            continue
        # get the correct geonames ID
        if 'accept' not in ex.keys():
            continue
        correct_id = [i['text'] for i in ex['options'] if i['id'] == ex['accept'][0]][0]
        try:
            correct_id = re.findall(r"\d+$", correct_id)[0]
        except IndexError:
            # this means it's a None/other example. Drop those for now
            continue
        # get the tokens matching what the annotator saw
        places = [i for i in doc if i.idx >= ex['spans'][0]['start'] and i.idx + len(i) <= ex['spans'][0]['end']]
        search_name = ''.join([i.text_with_ws for i in places]).strip()
        # get the tensor for those tokens
        if places:
            loc_ents = [ent for ent in doc.ents if ent.label_ in ['GPE', 'LOC']]
            tensor = np.mean(np.vstack([i._.tensor for i in places]), axis=0)
            doc_tensor = np.mean(np.vstack([i._.tensor for i in doc]), axis=0)
            other_locs = [i for e in loc_ents for i in e if i not in places]
            if other_locs:
                locs_tensor = np.mean(np.vstack([i._.tensor for i in other_locs]), axis=0)
            else:
                locs_tensor = np.zeros(len(tensor))
            in_rel = guess_in_rel(places)
            d = {"search_name": search_name,
               "tensor": tensor,
                "locs_tensor": locs_tensor,
                "doc_tensor": doc_tensor,
                "in_rel": in_rel,
               "correct_geonamesid": correct_id}
            doc_formatted.append(d)
            # Only one place name is annotated in each example, but we still want to know
            # the other place names that were extracted to calculate Geonames overlap
            # features. We'll throw these away later, so we can set the other values
            # to None. 
            for loc in other_locs:
                d = {"search_name": loc.text,
                     "tensor": None,
                     "locs_tensor": None,
                     "doc_tensor": None,
                     "in_rel": None,
                     "correct_geonamesid": None}
                doc_formatted.append(d)
            all_formatted.append(doc_formatted)
        doc_num += 1
    return all_formatted


def data_formatter_wiki(docs, data):
    """
    Format scraped Wikipedia location data into a format for training.
    
    Returns a list of lists, with one list for each document, consisting of each entity
    within the document. This round of training only annotated one location per
    "document"/sentence, so non-annotated entities have None as their value for
    correct_geonamesid. These will be discarded later.

    Parameters
    ---------
    docs: list of spaCy docs
    data: list of dicts

    Returns
    -------
    all_formatted: list of lists
      The list is of length docs, which each element a list of all the place 
      names within the document.
    """
    all_formatted = []
    doc_num = 0
    for doc, ex in tqdm(zip(docs, data), total=len(docs), leave=False): 
        doc_formatted = []
        correct_id = ex['correct_geonamesid']
        # we might lose some examples here if the tokenization ever changes, but
        # it should be extremely rare
        # These are the keys for the old 'sent' format. Replace with the document-level ones.
        #if 'start_char_sent' in ex.keys():
        #    orig_places = [ent for ent in doc.ents if ent.start_char >= ex['start_char_sent'] and ent.start_char < ex['end_char_sent']]
        #else:
        #    orig_places = [ent for ent in doc.ents if ent.start_char >= ex['start_char'] and ent.start_char < ex['end_char']]
        ## NEW:
        if 'start_char_sent' in ex.keys():
            orig_places = [ent for ent in doc.ents if ent.start_char >= ex['start_char_sent'] and ent.start_char < ex['end_char_sent']]
        else:
            orig_places = [ent for ent in doc.ents if ent.start_char >= ex['start_char_doc'] and ent.start_char < ex['end_char_doc']]
        # get the tensor for those tokens
        if orig_places:
            places = [i for i in orig_places[0]]
            search_name = ''.join([i.text_with_ws for i in places]).strip()
            #try:
            tensor = np.mean(np.vstack([i._.tensor for i in places]), axis=0)
            loc_ents = [ent for ent in doc.ents if ent.label_ in ['GPE', 'LOC']]
            doc_tensor = np.mean(np.vstack([i._.tensor for i in doc]), axis=0)
            orig_place_tokens = [i for i in orig_places[0]]
            other_locs = [i for e in loc_ents for i in e if i not in orig_place_tokens]
            if other_locs:
                locs_tensor = np.mean(np.vstack([i._.tensor for i in other_locs]), axis=0)
            else:
                locs_tensor = np.zeros(len(tensor))
            in_rel = guess_in_rel(places)
            d = {"search_name": search_name,
               "tensor": tensor,
                "locs_tensor": locs_tensor,
                "doc_tensor": doc_tensor,
                "in_rel": in_rel,
               "correct_geonamesid": correct_id}
            doc_formatted.append(d)
            #except Exception as e:
            #    logger.info(f"Exception {e}: {ex}")
            # Only one place name is annotated in each example, but we still want to know
            # the other place names that were extracted to calculate Geonames overlap
            # features. We'll throw these away later, so we can set the other values
            # to None. 
            for loc in other_locs:
                d = {"search_name": loc.text,
                     "tensor": None,
                     "locs_tensor": None,
                     "doc_tensor": None,
                     "in_rel": None,
                     "correct_geonamesid": None}
                doc_formatted.append(d)
            all_formatted.append(doc_formatted)
        doc_num += 1
    return all_formatted

def data_formatter_wiki_docs(docs, data, source="wiki_docs"):
    """
    Format the document-level Wikipedia data written by `prepare_wiki.py`.

    The older `data_formatter_wiki` reads the one-mention-per-row scrape, where
    each row is its own single-sentence document with exactly one label. This
    one takes a whole document and every mention linked inside it, which is what
    the geoparser actually sees at inference: `_add_cross_entity_counts` builds
    the adm1/country overlap features from the other place names in the same
    example, and in a one-sentence example there mostly aren't any.

    Entity selection mirrors `doc_to_ex_expanded` so training and inference
    agree on what counts as a place name:

      * the gold span's own tokens are used, as in `data_formatter`, rather than
        whichever spaCy entity happens to start inside it. spaCy's boundaries
        disagree with the wiki links often enough to matter -- "Aleppo" vs
        "Aleppo Governorate", "Raqqa" vs "Raqqa Governorate" -- and taking
        spaCy's span there would attach the gold id to a different place.
      * every remaining GPE/LOC/FAC entity is emitted unlabelled, so it still
        contributes to the document's overlap features before being discarded.
      * `locs_tensor` averages GPE/LOC/NORP tokens, which is the context set
        `doc_to_ex_expanded` uses.
    """
    all_formatted = []
    skipped = Counter()
    for doc, ex in tqdm(zip(docs, data), total=len(docs), leave=False):
        if doc.text != ex['text']:
            skipped["document text did not match its record"] += 1
            all_formatted.append([])
            continue
        doc_formatted = []
        doc_tensor = np.mean(np.vstack([i._.tensor for i in doc]), axis=0)
        context_ents = [ent for ent in doc.ents
                        if ent.label_ in ['GPE', 'LOC', 'EVENT_LOC', 'NORP']]
        context_tokens = [i for e in context_ents for i in e]

        def entry(place_tokens, search_name, correct_id):
            tensor = np.mean(np.vstack([i._.tensor for i in place_tokens]), axis=0)
            own = {i.i for i in place_tokens}
            other_locs = [i for i in context_tokens if i.i not in own]
            if other_locs:
                locs_tensor = np.mean(np.vstack([i._.tensor for i in other_locs]), axis=0)
            else:
                locs_tensor = np.zeros(len(tensor))
            return {"search_name": search_name,
                    "tensor": tensor,
                    "locs_tensor": locs_tensor,
                    "doc_tensor": doc_tensor,
                    "in_rel": guess_in_rel(place_tokens),
                    "correct_geonamesid": correct_id}

        labelled_spans = []
        for topo in ex['toponyms']:
            start, end = topo['start'], topo['end']
            place_tokens = [i for i in doc
                            if i.idx >= start and i.idx + len(i) <= end]
            if not place_tokens:
                skipped["no token fell inside the linked span"] += 1
                continue
            # Only entities the geoparser would extract are useful to train on.
            if not [i for i in place_tokens
                    if i.ent_type_ in ['GPE', 'LOC', 'EVENT_LOC', 'FAC']]:
                skipped["linked span is not a place name to spaCy"] += 1
                continue
            labelled_spans.append((start, end))
            if topo['geonamesid']:
                doc_formatted.append(entry(place_tokens, topo['phrase'],
                                           topo['geonamesid']))
            else:
                # Linked to Wikidata but never resolved to geonames. Useful as
                # context, useless as a label.
                skipped["linked mention has no geonames id"] += 1
                doc_formatted.append(entry(place_tokens, topo['phrase'], None))

        for ent in doc.ents:
            if ent.label_ not in ['GPE', 'LOC', 'EVENT_LOC', 'FAC']:
                continue
            lo = ent[0].idx
            hi = ent[-1].idx + len(ent[-1])
            if any(lo < e and s < hi for s, e in labelled_spans):
                continue
            doc_formatted.append(entry(list(ent), ent.text, None))

        all_formatted.append(doc_formatted)
    for reason, count in skipped.most_common():
        logger.warning(f"{source}: {count} mentions -- {reason}")
    return all_formatted


def shard_paths(base_dir, source):
    """Every .spacy file for a source, in the order its documents were read.

    A source is either one `source_x.spacy` or a numbered set of
    `source_x.000.spacy` shards; callers do not need to care which.
    """
    single = os.path.join(base_dir, "spacyed", f"source_{source}.spacy")
    sharded = sorted(glob.glob(os.path.join(base_dir, "spacyed",
                                            f"source_{source}.[0-9][0-9][0-9].spacy")))
    if sharded:
        return sharded
    return [single] if os.path.exists(single) else []


def texts_for(data, source):
    if source in ["prodigy", "syn_cities", "syn_caps", "wiki", "wiki_docs", "wiki_docs_full"]:
        return [i['text'] for i in data]
    return [i['text'] for i in data['articles']['article']]


def data_to_docs(data, source, base_dir, nlp, shard_size=0):
    """
    Run spaCy over a source and write the docs out for `add_es` to pick up.

    Each doc carries a 768-float tensor per token, so a corpus of any size has
    to be written in pieces -- the wiki corpus is ~1.2 KB of DocBin per
    character of text, and a single DocBin for all of it neither fits in RAM
    here nor in `format_source` when it reads it back. `shard_size` documents
    are held at a time and flushed to `source_{source}.{n:03d}.spacy`;
    shard_size=0 keeps the old single-file layout.
    """
    print("NLPing docs...")
    print("spaCy batch size: ", nlp.batch_size)
    texts = texts_for(data, source)

    # Empty and whitespace-only texts crash the tagger on GPU, inside cupy,
    # before any of our code runs.
    blank = [n for n, t in enumerate(texts) if not t.strip()]
    if blank:
        raise ValueError(f"{source}: {len(blank)} documents are empty or "
                         f"whitespace-only (first at index {blank[0]}); "
                         f"spaCy cannot run on them")

    written = []

    def flush(doc_bin, shard):
        if shard is None:
            fn = f"{base_dir}/spacyed/source_{source}.spacy"
        else:
            fn = f"{base_dir}/spacyed/source_{source}.{shard:03d}.spacy"
        with fast_docbin_io():
            doc_bin.to_disk(fn)
        written.append(fn)
        print(f"Wrote NLPed docs out to {fn}")

    doc_bin = DocBin(store_user_data=True)
    shard = 0
    n_in_bin = 0
    for doc in tqdm(nlp.pipe(texts, batch_size=100), total=len(texts)):
        doc_bin.add(doc)
        n_in_bin += 1
        if shard_size and n_in_bin >= shard_size:
            flush(doc_bin, shard)
            doc_bin = DocBin(store_user_data=True)
            n_in_bin = 0
            shard += 1
    if n_in_bin or not written:
        flush(doc_bin, shard if shard_size else None)

    # A shard count that shrank between runs would otherwise leave the tail of
    # the previous run on disk, and `format_source` would read it back as if it
    # belonged to this one.
    for stale in shard_paths(base_dir, source):
        if stale not in written:
            print(f"Removing stale shard {stale}")
            os.remove(stale)


def data_formatter(docs, data, source):
    """
    Calculate named entity and tensor info for training from the data provided by Gritta et al.

    Returns a list of lists, with one list for each document, consisting of each entity
    within the document.

    Parameters
    ---------
    docs: list of spaCy docs
    data: list of dicts
      Data from Gritta et al, converted from XML to dict
    source: the short name of the source used

    Returns
    -------
    all_formatted: list of lists
      The list is of length docs, which each element a list of all the place 
      names within the document.
    """
    all_formatted = []
    doc_num = 0
    skipped = Counter()
    if source in ["syn_cities", "syn_caps", "wiki"]:
        articles = data
    else:
        articles = data['articles']['article']
    for doc, ex in tqdm(zip(docs, articles), total=len(docs), leave=False):
        doc_formatted = []
        doc_tensor = np.mean(np.vstack([i._.tensor for i in doc]), axis=0)
        loc_ents = [ent for ent in doc.ents if ent.label_ in ['GPE', 'LOC']]
        for n, topo in enumerate(_as_list(ex['toponyms']['toponym'])):
            #print(topo['phrase'])
            if source == "gwn" and 'geonamesID' not in topo.keys():
                continue
            if source == "gwn" and not topo['geonamesID']:
                continue
            try:
                place_tokens = [i for i in doc if i.idx >= int(topo['start']) and i.idx + len(i) <= int(topo['end'])]
                # remove NORPs?
                gpes = [i for i in place_tokens if i.ent_type_ in ['GPE', 'LOC']]
                if not gpes:
                    continue
                tensor = np.mean(np.vstack([i._.tensor for i in place_tokens]), axis=0)
                other_locs = [i for e in loc_ents for i in e if i not in place_tokens]
                if other_locs:
                    locs_tensor = np.mean(np.vstack([i._.tensor for i in other_locs]), axis=0)
                else:
                    locs_tensor = np.zeros(len(tensor))
                if source == "gwn":
                    correct_geonamesid = topo['geonamesID']
                    search_name = topo['extractedName']
                elif source in ["syn_cities", "syn_caps"]:
                    correct_geonamesid = topo['geonamesID']
                    search_name = topo['placename']
                else:
                    correct_geonamesid = topo['gaztag']['@geonameid']
                    search_name = topo['phrase']
                in_rel = guess_in_rel(place_tokens)
                doc_formatted.append({"search_name": search_name,
                                  "tensor": tensor,
                                  "locs_tensor": locs_tensor,
                                  "doc_tensor": doc_tensor,
                                  "in_rel": in_rel,
                                  "correct_geonamesid": correct_geonamesid})
            except Exception as e:
                skipped[f"{type(e).__name__}: {e}"] += 1
                logger.debug(f"{e}: {doc_num}_{n}")
        all_formatted.append(doc_formatted)
        doc_num += 1
    for reason, count in skipped.most_common():
        logger.warning(f"{source}: dropped {count} toponyms -- {reason}")
    return all_formatted

#base_dir = "../raw_data/"
#source = "wiki"
#limit_types = "all_loc_types"
#max_results = 500
#fuzzy = 0
# !!!

def format_source(base_dir, source, geonames, max_results, fuzzy,
                 limit_types, source_dict, nlp, remove_correct=False,
                 es_chunk_size=250):
    print(f"limit types: {limit_types}")
    print(f"===== {source} =====")
    shards = shard_paths(base_dir, source)
    if not shards:
        raise FileNotFoundError(f"No spaCy output for {source}; run nlp-docs first")

    data = read_file(source_dict[source])
    records = (data if source in ["prodigy", "syn_cities", "syn_caps", "wiki", "wiki_docs", "wiki_docs_full"]
               else data['articles']['article'])

    if limit_types == True:
        limit_type_str = "pa_only"
    else:
        limit_type_str = "all_loc_types"

    total = 0
    read_so_far = 0
    for shard_n, fn in enumerate(shards):
        print(f"Converting {fn} back to spaCy docs...")
        doc_bin = DocBin().from_disk(fn)
        docs = list(doc_bin.get_docs(nlp.vocab))
        # Shards are written in reading order, so a shard's docs line up with
        # the records that follow the ones already consumed.
        shard_records = records[read_so_far:read_so_far + len(docs)]
        read_so_far += len(docs)

        if source == "prodigy":
            formatted = data_formatter_prodigy(docs, shard_records)
        elif source in ("wiki_docs", "wiki_docs_full"):
            formatted = data_formatter_wiki_docs(docs, shard_records, source)
        elif source == "wiki":
            formatted = data_formatter_wiki(docs, shard_records)
        else:
            formatted = data_formatter(docs, shard_records, source)
        del docs, doc_bin

        # formatted is a list of lists. We want the final data to be a flat list.
        # At the same time, we can exclude examples with missing geonames info
        esed_data = []
        print("Adding Elasticsearch data...")
        # Look up a chunk of documents at a time rather than one document at a time.
        # Each add_es_data_batch call collapses duplicate place names across the
        # whole chunk and sends what's left as a handful of _msearch requests, so
        # per-document calls were paying the per-request overhead ~2,000 times over.
        # Chunking (rather than one call for the corpus) just bounds peak memory
        # while planning; the candidate cache is per-service and carries across.
        for start in tqdm(range(0, len(formatted), es_chunk_size), leave=False):
            chunk = formatted[start:start + es_chunk_size]
            for esd in add_es_data_batch(chunk, geonames, max_results, fuzzy,
                                         limit_types, remove_correct):
                for e in esd:
                    if e['correct_geonamesid'] != None:
                        esed_data.append(e)

        suffix = "" if len(shards) == 1 else f".{shard_n:03d}"
        out_fn = (f"es_formatted_{source}{suffix}_{max_results}"
                  f"_{limit_type_str}_fuzzy_{fuzzy}.pkl")
        out_file = os.path.join(base_dir, "pickled_es", out_fn)
        print(f"Place names in shard: {len(esed_data)}. Writing to {out_file}...")
        with open(out_file, 'wb') as f:
            pickle.dump(esed_data, f)
        total += len(esed_data)
        del esed_data

    print(f"Total place names in {source}: {total}")

def training_pairs(es_train_data):
    """Every (mention string, gold geonameid) pair the model was trained on.

    81% of held-out entities have their exact pair somewhere in training and
    score 0.960 there against 0.771 on the rest
    (experiments/campaign2/data_quality_report.md), so "novel-pair exact
    match" is the guardrail metric: the one number answer-key memorisation
    cannot move.
    """
    return {(str(ex.get('search_name')), str(ex.get('correct_geonamesid')))
            for ex in es_train_data}


def load_twin_cache(path, names, datasets):
    """Gold A/P twin classes per held-out entity, from the cache on disk.

    Written by `tools/twin_credit_eval.py twin-cache`; it depends only on the
    pickles and the labels, not on the model, so it is computed once. Sources
    whose held-out size does not match the cache are skipped rather than
    silently mis-scored.
    """
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        cache = json.load(f)
    out = {}
    for name, data in zip(names, datasets):
        rows = cache.get("sources", {}).get(name)
        if rows is None:
            continue
        if len(rows) != len(data):
            logger.warning(f"twin cache for {name} has {len(rows)} entities, "
                           f"held-out set has {len(data)}; skipping twin credit")
            continue
        out[name] = [frozenset(r) for r in rows]
    return out


def campaign2_scoreboard(names, datasets, data_loaders, model, es_train_data,
                         twin_cache_path):
    """TLG-hard, novel-pair EM, twin credit and the three accuracies."""
    from error_utils import make_campaign2_dict
    twins = load_twin_cache(twin_cache_path, names, datasets)
    return make_campaign2_dict(names, datasets, data_loaders, model,
                               train_pairs=training_pairs(es_train_data),
                               twins=twins)


def print_campaign2(c):
    def f(x):
        return "  n/a" if x != x else f"{x:.4f}"
    print("---- campaign-2 scoreboard "
          "(primary: TLG-hard; headline macro excludes Synth, D4) ----")
    print(f"  TLG-hard (TR/LGL/GWN macro EM, non-country golds, "
          f"n={c['tlg_hard_n']}):  {f(c['tlg_hard'])}"
          f"   [campaign convention, abstentions not charged: "
          f"{f(c['tlg_hard_noabstain'])}]")
    print(f"  novel-pair EM (guardrail, n={c['novel_pair_n']}):            "
          f"{f(c['novel_pair_em'])}   [seen pairs {f(c['seen_pair_em'])}]")
    print(f"  twin-credit EM (macro, no Synth):                {f(c['twin_credit_macro'])}")
    print(f"  macro EM, 5 sources (headline, no Synth):        "
          f"{f(c['em_conditioned_macro'])}   [no abstention charge: "
          f"{f(c['em_conditioned_macro_noabstain'])}]")
    print(f"  macro EM, 6 sources (legacy, ledger continuity): "
          f"{f(c['em_conditioned_macro_legacy6'])}   [no abstention charge: "
          f"{f(c['em_conditioned_macro_legacy6_noabstain'])}]")
    print(f"  EM over every held-out mention (macro/pooled):   "
          f"{f(c['em_all_macro'])} / {f(c['em_all_pooled'])}")
    print(f"  abstained on {100 * c['abstain_rate']:.2f}% of mentions, "
          f"{100 * c['abstain_precision']:.1f}% of those unanswerable "
          f"(base rate {100 * c['unanswerable_rate']:.2f}%)")
    for s, v in c['per_source'].items():
        print(f"    {s:<10} EM {f(v['em_conditioned'])}  non-country "
              f"{f(v['em_noncountry'])} (n={v['n_noncountry']})  novel "
              f"{f(v['novel_pair_em'])} (n={v['n_novel']})  twin "
              f"{f(v['twin_credit'])}  abstain {100 * v['abstain_rate']:.2f}%")


##################################

app = typer.Typer(add_completion=True)


@app.command()
def nlp_docs(base_dir: str,
            sources: str = "tr, lgl, gwn, prodigy, syn_cities, syn_caps, wiki_docs",
            shard_size: int = 0):
    """
    Run spaCy over a list of training data sources and save the output.

    Parameters
    ---------
    base_dir: Path
      path to the directory with training data
    sources: list
      
    """
    print("Loading NLP stuff...")
    spacy.prefer_gpu()
    nlp = spacy.load("en_core_web_trf")
    nlp.add_pipe("token_tensors")

    # check the spaCy model is on the GPU
    source_dict = {"tr":"Pragmatic-Guide-to-Geoparsing-Evaluation/data/Corpora/TR-News.xml",
                  "lgl":"Pragmatic-Guide-to-Geoparsing-Evaluation/data/Corpora/lgl.xml",
                  "gwn": "Pragmatic-Guide-to-Geoparsing-Evaluation/data/GWN.xml",
                  "prodigy": "orig_mordecai/loc_rank_db.jsonl",
                  "syn_cities": "synth_raw/synthetic_cities_short.jsonl",
                  "syn_caps": "synth_raw/synth_caps.jsonl",
                  "wiki": "wiki/wiki_training_data_sents.jsonl",
                  "wiki_docs": "wiki/wiki_docs.jsonl",
                  "wiki_docs_full": "wiki/wiki_docs_full.jsonl"}
    for k, v in source_dict.items():
        source_dict[k] = os.path.join(base_dir, v)

    print("Reading in data...")
    print("sources: ", sources, type(sources))
    if type(sources) is str:
        sources = [i.strip() for i in sources.split(",")]
    for source in sources:
        print(source_dict[source])
        data = read_file(source_dict[source])
        data_to_docs(data, source, base_dir, nlp, shard_size=shard_size)

@app.command()
def add_es(base_dir: str,
          max_results: int = 500,
          fuzzy: int = 0,
          limit_types: bool = False,
          sources: str = "tr, lgl, gwn, prodigy, syn_cities, syn_caps, wiki_docs"):
    """
    Process spaCy outputs to add candidate entity data from Geonames/Elasticsearch.

    Note: You must run `nlp_docs` before you can run `add_es`.

    Parameters
    ----------
    base_dir: Path
      Path to the saved .pkl files produced by the command `nlp_docs`
    max_results: int
      How many results to get back from ES/Geonames?
    limit_types: bool
      restrict ES/Geonames results to only places/areas, excluding geographic
      features, facilities, etc?
    source: list
      Which sources to process?
    """
    geonames = GeonamesService(es_client=es_util.setup_es_client())
    print("Loading spacy model...")
    spacy.prefer_gpu()
    nlp = spacy.load("en_core_web_trf")
    nlp.add_pipe("token_tensors")
    source_dict = {"tr":"Pragmatic-Guide-to-Geoparsing-Evaluation/data/Corpora/TR-News.xml",
                  "lgl":"Pragmatic-Guide-to-Geoparsing-Evaluation/data/Corpora/lgl.xml",
                  "gwn": "Pragmatic-Guide-to-Geoparsing-Evaluation/data/GWN.xml",
                  "prodigy": "orig_mordecai/loc_rank_db.jsonl",
                  "syn_cities": "synth_raw/synthetic_cities_short.jsonl",
                  "syn_caps": "synth_raw/synth_caps.jsonl",
                  "wiki": "wiki/wiki_training_data_sents.jsonl",
                  "wiki_docs": "wiki/wiki_docs.jsonl",
                  "wiki_docs_full": "wiki/wiki_docs_full.jsonl"}
    for k, v in source_dict.items():
        source_dict[k] = os.path.join(base_dir, v)
    if type(sources) is str:
        sources = [i.strip() for i in sources.split(",")]
    for source in sources:
        remove_correct = source == "wiki_incorrect"
        format_source(base_dir, 
                      source, 
                      geonames,
                      max_results=max_results, 
                      limit_types=limit_types, 
                      fuzzy=fuzzy,
                      source_dict=source_dict, 
                      nlp=nlp,
                      remove_correct=remove_correct)
    print("Complete")


@app.command()
def compact_cache(data_dir: str = "raw_data",
                  max_results: int = 500,
                  fuzzy: int = 0,
                  limit_types: str = "all_loc_types",
                  pickle_suffix: str = ""):
    """
    Write `_enriched{suffix}_compact` copies of the enriched pickles.

    Same entities, same labels, same feature values -- the per-candidate numeric
    features move into one float32 matrix per entity (`feat_matrix`) and the
    candidate dicts keep only the seven keys that are still looked up by name.
    Training auto-detects these and uses them; a run then peaks at ~6 GB instead
    of ~22 GB, which is the difference between one run at a time and five.
    """
    import glob as _glob
    # The pattern is anchored on the full suffix, so `--pickle-suffix ""` does
    # not also pick up `_enriched_r2.pkl` and vice versa.
    tail = f'_enriched{pickle_suffix}.pkl'
    pat = (f'{data_dir}/pickled_es/es_formatted_*_{max_results}_{limit_types}'
           f'_fuzzy_{fuzzy}{tail}')
    for fn in sorted(_glob.glob(pat)):
        out = fn[:-len(tail)] + f'_enriched{pickle_suffix}_compact.pkl'
        if os.path.exists(out):
            print(f"exists, skipping: {out}")
            continue
        with open(fn, 'rb') as f:
            data = pickle.load(f)
        compact_candidates(data)
        with open(out, 'wb') as f:
            pickle.dump(data, f, protocol=4)
        print(f"{os.path.basename(fn)} -> {os.path.basename(out)} "
              f"({len(data)} entities, {os.path.getsize(out)/1e9:.2f} GB)")
        del data
    print("Complete")


@app.command()
def train(data_dir: str = "raw_data",
          batch_size: int = 32,
          test_batch_size: int = 64,
          epochs: int = 20,
          lr: float = 0.001,
          max_choices: int = 500,
          dropout: float = 0.3,
          avg_params: bool = False,
          limit_es_results: str = "all_loc_types",
          country_size: int = 24,
          code_size: int = 8,
          country_pred: bool = False,
          mix_dim: int = 24,
          fuzzy: int = 0,
          dataset_names: str = "Prodigy, TR, LGL, GWN, Synth, WikiDocs",
          device: str = "",
          seed: int = 42,
          source_limits: str = "",
          metrics_out: str = "",
          run_name: str = "",
          logits: bool = False,
          mask_padding: bool = False,
          oov_bucket_fix: bool = False,
          modern_mlp: bool = False,
          label_smoothing: float = 0.0,
          weight_decay: float = 0.0,
          enriched: bool = False,
          feature_blocks: str = "",
          outlet_dropout: float = 0.0,
          pickle_suffix: str = "",
          mix_depth: int = 2,
          residual: bool = False,
          listwise: bool = False,
          listwise_heads: int = 4,
          aux_country_weight: float = 0.0,
          aux_class_weight: float = 0.0,
          full_null_row: bool = False,
          window: int = 0,
          abstain_weight: float = 1.0,
          avg_mode: str = "swa",
          avg_start: int = 0,
          ema_decay: float = 0.9,
          train_after_eval: bool = False,
          lr_schedule: bool = False,
          checkpoint_out: str = "",
          overwrite_checkpoint: bool = False,
          split_mode: str = "entity",
          twin_cache: str = "experiments/campaign2/twin_gold.json"
):
    """
    Train the pytorch model from formatted training data.

    `--checkpoint-out` refuses to overwrite an existing file unless
    `--overwrite-checkpoint` is passed: on 2026-08-20 a rejected experiment arm
    silently clobbered the repo-root checkpoint, and the replacement was only
    noticed by checksumming it against the arm's own seed file.
    """
    if split_mode not in ("entity", "doc"):
        raise typer.BadParameter("--split-mode must be 'entity' or 'doc'")
    # Checked before any data is loaded: a 50-second run is cheap, but finding
    # out at the end that the destination was taken is not.
    for path, what in ((checkpoint_out, "--checkpoint-out"),
                       (checkpoint_out + ".json" if checkpoint_out else "",
                        "--checkpoint-out's config sidecar")):
        if path and os.path.exists(path) and not overwrite_checkpoint:
            raise typer.BadParameter(
                f"{what} {path} already exists. Pass --overwrite-checkpoint to "
                f"replace it, or write somewhere else -- a rejected arm "
                f"overwriting a shipped checkpoint is how the 2026-08-20 "
                f"clobber happened.")
    # The 'seed' in the config was never applied to anything, so two runs with
    # identical data and hyperparameters differed by more than most of the
    # effects we want to measure. Seed everything that moves.
    set_seed(seed)
    wandb.init(project="mordecai3", entity="ahalt", allow_val_change=True,
               name=run_name if run_name else None)

    dataset_names_list = [str(name).strip() for name in str(dataset_names).split(",")]
    limits = {}
    for pair in str(source_limits).split(","):
        if pair.strip():
            name, _, count = pair.partition("=")
            limits[name.strip()] = int(count)
    blocks = [b.strip() for b in str(feature_blocks).split(",") if b.strip()]
    if blocks and not enriched:
        raise typer.BadParameter("--feature-blocks needs --enriched: the extra "
                                 "features only exist in the enriched pickles")
    if outlet_dropout > 0 and "outlet" not in blocks:
        raise typer.BadParameter("--outlet-dropout needs the 'outlet' feature "
                                 "block; there is nothing to drop without it")
    if not 0.0 <= outlet_dropout < 1.0:
        raise typer.BadParameter("--outlet-dropout must be in [0, 1)")
    if pickle_suffix and not enriched:
        raise typer.BadParameter("--pickle-suffix needs --enriched: it selects a "
                                 "label variant of the enriched pickles")
    # --window scores fewer rows than the pickle holds candidates. Then some
    # golds fall outside the window, and TrainData.create_labels indexes a
    # labels array of length `window` with the gold's position in the full list
    # -> IndexError. --full-null-row is the flag that maps those golds onto the
    # reserved "no correct answer" row, which is the point of the arm.
    if window and window < max_choices and not full_null_row:
        raise typer.BadParameter(
            f"--window {window} < --max-choices {max_choices} needs "
            "--full-null-row: without it, a gold outside the window has no "
            "label to point at")
    if abstain_weight != 1.0 and not (label_smoothing > 0 and mask_padding):
        raise typer.BadParameter(
            "--abstain-weight only reaches the loss through masked_smoothed_ce, "
            "which needs --label-smoothing and --mask-padding")
    config = wandb.config          # Initialize config
    config.update({
        'batch_size': batch_size,
        'test_batch_size': test_batch_size,
        'epochs': epochs,
        'lr': lr,
        'seed': seed,
        'log_interval': 10,
        'max_choices': max_choices,
        'dropout': dropout,
        'avg_params': avg_params,
        'limit_es_results': limit_es_results,
        'country_size': country_size,
        'code_size': code_size,
        'country_pred': country_pred,
        'mix_dim': mix_dim,
        'dataset_names': dataset_names_list,
        'source_limits': limits,
        'fuzzy': fuzzy,
        'logits': logits,
        'mask_padding': mask_padding,
        'oov_bucket_fix': oov_bucket_fix,
        'modern_mlp': modern_mlp,
        'label_smoothing': label_smoothing,
        'weight_decay': weight_decay,
        'enriched': enriched,
        'feature_blocks': blocks,
        'outlet_dropout': outlet_dropout,
        'pickle_suffix': pickle_suffix,
        'mix_depth': mix_depth,
        'residual': residual,
        'listwise': listwise,
        'listwise_heads': listwise_heads,
        'aux_country_weight': aux_country_weight,
        'aux_class_weight': aux_class_weight,
        'full_null_row': full_null_row,
        'window': window,
        'abstain_weight': abstain_weight,
        'avg_mode': avg_mode,
        'avg_start': avg_start,
        'ema_decay': ema_decay,
        'train_after_eval': train_after_eval,
        'lr_schedule': lr_schedule,
        'split_mode': split_mode
    },
    allow_val_change=True)

    print(config.__dict__)

    train_loader, es_train_data, data_loaders, datasets = load_es_data(data_dir, 
                                                  config.max_choices, 
                                                  config.limit_es_results,
                                                  config.fuzzy,
                                                  config.batch_size,
                                                  config.test_batch_size,
                                                  data_sources=dataset_names_list,
                                                  source_limits=limits,
                                                  oov_bucket_fix=config.oov_bucket_fix,
                                                  enriched=config.enriched,
                                                  feature_blocks=blocks,
                                                  pickle_suffix=config.pickle_suffix,
                                                  full_null_row=config.full_null_row,
                                                  window=config.window or None,
                                                  split_mode=config.split_mode)
    logger.info(f"Total training examples: {len(es_train_data)}")

    device = torch.device(device if device else
                          ("cuda:0" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Training on {device}")
    model = geoparse_model(device = device,
                              bert_size = es_train_data[0]['tensor'].shape[0],
                              num_feature_codes=53+1,
                              dropout = config.dropout,
                              country_size=config.country_size,
                              code_size=config.code_size,
                              mix_dim=config.mix_dim,
                              country_pred=config.country_pred,
                              n_extra_features=len(expand_feature_blocks(blocks)),
                              mix_depth=config.mix_depth,
                              residual=config.residual,
                              listwise=config.listwise,
                              listwise_heads=config.listwise_heads,
                              aux_country=config.aux_country_weight > 0,
                              aux_class=config.aux_class_weight > 0,
                              return_logits=config.logits,
                              mask_padding=config.mask_padding,
                              modern_mlp=config.modern_mlp)
    model.to(device)
    # Future work: Can add  an "ignore_index" argument so that some inputs don't have losses calculated
    # label_smoothing only means anything when the model returns logits: on a
    # softmaxed output CrossEntropyLoss is already smoothing by accident.
    loss_func=nn.CrossEntropyLoss(label_smoothing=config.label_smoothing) # single label, multi-class
    aux_w = config.aux_country_weight + config.aux_class_weight
    if aux_w >= 1:
        raise typer.BadParameter("auxiliary weights must leave room for the "
                                 "main loss (they sum to >= 1)")
    use_aux = aux_w > 0
    aux_loss_func = nn.CrossEntropyLoss()

    smooth_over_live = config.label_smoothing > 0 and config.mask_padding

    # --abstain-weight upweights the examples whose target is the reserved
    # "no correct answer" row (2.4% of the training set). At the default 1.0 the
    # weights are never built and the loss takes its original code path.
    reweight_abstain = config.abstain_weight != 1.0

    def label_loss(pred, label, input):
        if smooth_over_live:
            weights = None
            if reweight_abstain:
                is_abstain = (label == pred.shape[1] - 1).float()
                weights = 1.0 + (config.abstain_weight - 1.0) * is_abstain
            return masked_smoothed_ce(pred, label, input['mask'],
                                      config.label_smoothing, weights)
        return loss_func(pred, label)

    if config.weight_decay > 0:
        # AdamW, not Adam(weight_decay=): the latter folds L2 into the moment
        # estimates, which is not the decay we're asking for.
        optimizer = optim.AdamW(model.parameters(), lr=config.lr,
                                weight_decay=config.weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs+1)

    # Weight averaging, rewritten. What was here before had three faults: the
    # averaged copy was never evaluated and never saved (every reported number
    # came from the raw model), SWALR pinned the learning rate at 0.05 -- fifty
    # times the base rate -- for every epoch after the fifth, and the cosine
    # scheduler it fought with was only stepped when averaging was on. There is
    # no BatchNorm anywhere in the model, so update_bn is genuinely unnecessary.
    #
    # `swa`: equal-weight average of the epoch-end weights from avg_start on,
    # keeping the ordinary schedule (no SWALR: this recipe's accuracy is tuned
    # around a constant 1e-3, and a 0.05 phase destroys it).
    # `ema`: exponential moving average with `ema_decay` per epoch.
    avg_state = None
    n_avg = 0
    avg_start = config.avg_start or (config.epochs // 2 + 1)
    if config.avg_params and config.avg_mode not in ("swa", "ema"):
        raise typer.BadParameter("--avg-mode must be 'swa' or 'ema'")

    def update_average():
        nonlocal avg_state, n_avg
        n_avg += 1
        sd = model.state_dict()
        if avg_state is None:
            avg_state = {k: v.detach().clone().float() for k, v in sd.items()}
            return
        for k, v in sd.items():
            v = v.detach().float()
            if config.avg_mode == "ema":
                avg_state[k].mul_(config.ema_decay).add_(v, alpha=1 - config.ema_decay)
            else:
                avg_state[k].add_((v - avg_state[k]) / n_avg)

    def load_average():
        """Swap the averaged weights in; returns the raw weights to restore."""
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict({k: avg_state[k].to(v.dtype)
                               for k, v in backup.items()})
        return backup

    wandb.watch(model, log='all')

    model.train()
    history = []
    for epoch in range(1, config.epochs+1):
        epoch_loss = 0
        epoch_acc = 0

        # e53: blank the outlet block for a random half (or `p`) of the
        # documents that have one, redrawn every epoch so the model sees the
        # same article both ways over training. Deterministic in
        # (document, epoch, seed); a no-op at the default 0.0.
        if config.outlet_dropout > 0:
            n_dropped = train_loader.dataset.set_outlet_dropout(
                config.outlet_dropout, epoch, config.seed)
            if epoch == 1:
                logger.info(f"Outlet dropout p={config.outlet_dropout}: "
                            f"{n_dropped} documents blanked in epoch 1")

        for label, country, input in train_loader:
            label = label.type(torch.LongTensor).to(device)
            country = country.type(torch.LongTensor).to(device)
            input = {k: v.to(device, non_blocking=True) for k, v in input.items()}
            optimizer.zero_grad()
            if config.country_pred:
                label_pred, country_pred = model(input)
                #label_pred = label_pred.type(torch.LongTensor)
                #country_pred = label_pred.type(torch.LongTensor)
                loss_1 = label_loss(label_pred, label, input)
                loss_country = loss_func(country_pred, country)
                loss = 0.8*loss_1 + 0.2*loss_country
            elif use_aux:
                label_pred, aux = model(input, return_aux=True)
                loss = (1 - aux_w) * label_loss(label_pred, label, input)
                if 'country' in aux:
                    loss = loss + config.aux_country_weight * aux_loss_func(
                        aux['country'], country)
                if 'fclass' in aux:
                    loss = loss + config.aux_class_weight * aux_loss_func(
                        aux['fclass'], input['class_label'].type(torch.LongTensor).to(device))
                loss = loss.squeeze() if loss.dim() else loss
            else:
                label_pred = model(input)
                #label_pred = label_pred.type(torch.LongTensor)
                loss = label_loss(label_pred, label, input)

            #logger.debug(country_pred[1])
            #loss_country = loss_func(country_pred, country)
            #loss = loss_label + loss_country
            acc = binary_acc(label_pred, label)
            #country_acc = binary_acc(country_pred, country)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

        if config.lr_schedule:
            # Never stepped before: the only call site sat inside the
            # avg_params branch, so every run in this campaign trained at a
            # flat 1e-3 despite constructing a cosine schedule.
            scheduler.step()

        if config.avg_params and epoch >= avg_start:
            update_average()

        if avg_state is not None:
            # Report the weights we would ship, not the raw ones.
            backup = load_average()
            wandb_dict = make_wandb_dict(config.dataset_names, datasets,
                                         data_loaders, model)
            model.load_state_dict(backup)
        else:
            wandb_dict = make_wandb_dict(config.dataset_names, datasets, data_loaders, model)
        if config.train_after_eval:
            # evaluate_results leaves the model in eval mode and nothing put it
            # back, so dropout has been inert from epoch 2 onward in every run.
            model.train()
        wandb_dict['loss'] = epoch_loss/len(train_loader)
        history.append({k: float(v) for k, v in wandb_dict.items()})

        print(f"Epoch {epoch+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Exact Match: {wandb_dict['exact_match_avg']:.3f} | Country Match: {wandb_dict['country_avg']:.3f}")  # | Prodigy Acc: {epoch_acc_prod/len(prod_loader):.3f} | TR Acc: {epoch_acc_tr/len(tr_loader):.3f} | LGL Acc: {epoch_acc_lgl/len(lgl_loader):.3f} | GWN Acc: {epoch_acc_gwn/len(gwn_loader):.3f} | Syn Acc: {epoch_acc_syn/len(syn_loader):.3f}')
        wandb.log(wandb_dict)

    if metrics_out:
        final = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                 for k, v in wandb_dict.items()}
        # A single final-epoch number moves by ~0.01 between seeds; the mean of
        # the last five epochs is what these runs are actually compared on.
        tail = history[-5:]
        final['_last5'] = {k: sum(h[k] for h in tail)/len(tail) for k in tail[0]}
        final['_history'] = history
        final['_config'] = {'seed': seed, 'epochs': config.epochs,
                            'mix_dim': config.mix_dim, 'lr': config.lr,
                            'dropout': config.dropout,
                            'max_choices': config.max_choices,
                            'logits': config.logits,
                            'mask_padding': config.mask_padding,
                            'oov_bucket_fix': config.oov_bucket_fix,
                            'modern_mlp': config.modern_mlp,
                            'label_smoothing': config.label_smoothing,
                            'weight_decay': config.weight_decay,
                            'enriched': config.enriched,
                            'pickle_suffix': config.pickle_suffix,
                            'feature_blocks': list(config.feature_blocks),
                            'n_extra_features': len(expand_feature_blocks(blocks)),
                            'mix_depth': config.mix_depth,
                            'residual': config.residual,
                            'listwise': config.listwise,
                            'listwise_heads': config.listwise_heads,
                            'aux_country_weight': config.aux_country_weight,
                            'aux_class_weight': config.aux_class_weight,
                            'full_null_row': config.full_null_row,
                            'avg_params': config.avg_params,
                            'avg_mode': config.avg_mode,
                            'avg_start': avg_start if config.avg_params else None,
                            'ema_decay': config.ema_decay,
                            'train_after_eval': config.train_after_eval,
                            'lr_schedule': config.lr_schedule,
                            'country_size': config.country_size,
                            'code_size': config.code_size,
                            'dataset_names': list(config.dataset_names),
                            'source_limits': dict(config.source_limits),
                            'n_train': len(es_train_data),
                            'n_val': {n: len(d) for n, d in
                                      zip(config.dataset_names, datasets)}}
        # Only non-default values are recorded: a default run's json has to stay
        # byte-identical to the frozen ones in experiments/ (verified by rerun).
        if config.window:
            final['_config']['window'] = config.window
        if config.abstain_weight != 1.0:
            final['_config']['abstain_weight'] = config.abstain_weight
        if config.split_mode != "entity":
            final['_config']['split_mode'] = config.split_mode
        with open(metrics_out, 'w') as f:
            json.dump(final, f, indent=2)
        logger.info(f"Wrote metrics to {metrics_out}")

    if avg_state is not None:
        # Ship the average, not the last raw step.
        load_average()
    today = datetime.datetime.today().strftime('%Y-%m-%d')
    ckpt = checkpoint_out if checkpoint_out else f"mordecai_{today}.pt"
    cfg_path = (ckpt + ".json" if checkpoint_out
                else f"mordecai_{today}.json")
    if os.path.dirname(ckpt):
        os.makedirs(os.path.dirname(ckpt), exist_ok=True)
    logger.info(f"Saving model to {ckpt}")
    torch.save(model.state_dict(), ckpt)
    # Layer shapes reveal the dimensions but not the behaviour flags: a
    # checkpoint trained with --logits/--mask-padding/--modern-mlp loads
    # silently into a default model and mis-runs. Write them down next to it.
    model_config = {
        'bert_size': int(es_train_data[0]['tensor'].shape[0]),
        'num_feature_codes': 53 + 1,
        'dropout': config.dropout,
        'country_size': config.country_size,
        'code_size': config.code_size,
        'mix_dim': config.mix_dim,
        'country_pred': config.country_pred,
        'max_choices': config.max_choices,
        'return_logits': config.logits,
        'mask_padding': config.mask_padding,
        'modern_mlp': config.modern_mlp,
        'oov_bucket_fix': config.oov_bucket_fix,
        'full_null_row': config.full_null_row,
        'feature_blocks': list(config.feature_blocks),
        'n_extra_features': len(expand_feature_blocks(blocks)),
        'enriched': config.enriched,
        'pickle_suffix': config.pickle_suffix,
        'mix_depth': config.mix_depth,
        'residual': config.residual,
        'listwise': config.listwise,
        'listwise_heads': config.listwise_heads,
        'aux_country': config.aux_country_weight > 0,
        'aux_class': config.aux_class_weight > 0,
        'weight_avg': (config.avg_mode if avg_state is not None else None),
        **({'outlet_dropout': config.outlet_dropout}
           if config.outlet_dropout else {}),
        # `max_choices` names the pickles and is the default window; a run that
        # overrode the window records it so the checkpoint is not silently
        # served at a width it never saw.
        **({'train_window': config.window} if config.window else {}),
        **({'abstain_weight': config.abstain_weight}
           if config.abstain_weight != 1.0 else {}),
    }
    with open(cfg_path, 'w') as f:
        json.dump(model_config, f, indent=2)
    logger.info(f"Wrote model config to {cfg_path}")

    # ---- campaign-2 scoreboard (decisions D1/D4) --------------------------
    # One extra forward pass over the held-out sets, on the weights that were
    # just saved. It is deliberately NOT written into `metrics_out`: that file
    # is the ledger and a rerun of any frozen arm has to reproduce it byte for
    # byte. It goes to `<metrics_out>.metrics2.json` instead, and to the log.
    try:
        campaign2 = campaign2_scoreboard(dataset_names_list, datasets,
                                         data_loaders, model, es_train_data,
                                         twin_cache if split_mode == "entity"
                                         else "")
    except Exception as e:                       # never lose a run over a metric
        logger.warning(f"campaign-2 metrics failed: {type(e).__name__}: {e}")
        campaign2 = None
    if campaign2:
        campaign2['_config'] = {'seed': seed, 'split_mode': split_mode,
                                'checkpoint': ckpt}
        print_campaign2(campaign2)
        if metrics_out:
            out2 = os.path.splitext(metrics_out)[0] + ".metrics2.json"
            with open(out2, 'w') as f:
                json.dump(campaign2, f, indent=2)
            logger.info(f"Wrote campaign-2 metrics to {out2}")
    logger.info("Run complete.")



if __name__ == "__main__":
    app()
