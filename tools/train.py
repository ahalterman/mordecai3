import glob
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
import torch.optim as optim
import typer
import wandb
import xmltodict
from error_utils import make_wandb_dict
from mordecai3.geoparse import guess_in_rel, add_es_data_batch

from mordecai3.torch_model import geoparse_model

from mordecai3.mordecai_utilities import fast_docbin_io, spacy_doc_setup
from spacy.tokens import DocBin
from torch.utils.data import DataLoader
from mordecai3.torch_model import TrainData, geoparse_model
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


def load_es_data(data_dir, 
             max_results, 
             limit_types, 
             fuzzy,
             batch_size,
             test_batch_size,
             train_frac=0.7,
             data_sources=["Prodigy", "TR", "LGL", "GWN", "Synth", "Wiki"],
             source_limits=None):
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

    Returns
    -------
    list
      a list of formatted, shuffled training data
    """
    es_train_data = [] 
    data_loaders = []
    val_datasets = []

    for source in data_sources:
        logger.info(f"Loading data for {source}")
        if source == 'Prodigy':
            with open(f'{data_dir}/pickled_es/es_formatted_prodigy_{max_results}_{limit_types}_fuzzy_{fuzzy}.pkl', 'rb') as f:
                es_data = pickle.load(f)
        elif source == "TR":
            with open(f'{data_dir}/pickled_es/es_formatted_tr_{max_results}_{limit_types}_fuzzy_{fuzzy}.pkl', 'rb') as f:
                es_data = pickle.load(f)
        elif source == "LGL":
            with open(f'{data_dir}/pickled_es/es_formatted_lgl_{max_results}_{limit_types}_fuzzy_{fuzzy}.pkl', 'rb') as f:
                es_data = pickle.load(f)
        elif source == "GWN":
            with open(f'{data_dir}/pickled_es/es_formatted_gwn_{max_results}_{limit_types}_fuzzy_{fuzzy}.pkl', 'rb') as f:
                es_data = pickle.load(f)
        elif source == "Synth":
            # this one's a little different bc there are two files
            with open(f'{data_dir}/pickled_es/es_formatted_syn_cities_{max_results}_{limit_types}_fuzzy_{fuzzy}.pkl', 'rb') as f:
                es_data_syn1 = pickle.load(f)
            with open(f'{data_dir}/pickled_es/es_formatted_syn_caps_{max_results}_{limit_types}_fuzzy_{fuzzy}.pkl', 'rb') as f:
                es_data_syn_caps= pickle.load(f)
            random.seed(617)
            random.shuffle(es_data_syn1)
            random.shuffle(es_data_syn_caps)
            # combine both syn datasets and split
            es_data = es_data_syn1[0:500] + es_data_syn_caps[0:500]
        elif source == "Wiki":
            with open(f'{data_dir}/pickled_es/es_formatted_wiki_{max_results}_{limit_types}_fuzzy_{fuzzy}.pkl', 'rb') as f:
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
            tail = f'_{max_results}_{limit_types}_fuzzy_{fuzzy}.pkl'
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
        dataset = TrainData(es_data_val, max_choices=max_results)
        loader = DataLoader(dataset=dataset, batch_size=test_batch_size, shuffle=False)
        data_loaders.append(loader)

    # now make one loader for all training data
    random.seed(617)
    random.shuffle(es_train_data)
    train_data = TrainData(es_train_data, max_choices=max_results)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

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
          run_name: str = ""
):
    """
    Train the pytorch model from formatted training data.
    """
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
        'fuzzy': fuzzy
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
                                                  source_limits=limits) 
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
                              country_pred=config.country_pred)
    model.to(device)
    # Future work: Can add  an "ignore_index" argument so that some inputs don't have losses calculated
    loss_func=nn.CrossEntropyLoss() # single label, multi-class
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    if config.avg_params:
        from torch.optim.lr_scheduler import CosineAnnealingLR
        from torch.optim.swa_utils import SWALR, AveragedModel

        swa_model = AveragedModel(model)
        scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs+1)
        swa_start = 5
        swa_scheduler = SWALR(optimizer, swa_lr=0.05)
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs+1)

    wandb.watch(model, log='all')

    model.train()
    history = []
    for epoch in range(1, config.epochs+1):
        epoch_loss = 0
        epoch_acc = 0

        for label, country, input in train_loader:
            label = label.type(torch.LongTensor).to(device)
            country = country.type(torch.LongTensor).to(device)
            input = {k: v.to(device, non_blocking=True) for k, v in input.items()}
            optimizer.zero_grad()
            if config.country_pred:
                label_pred, country_pred = model(input)
                #label_pred = label_pred.type(torch.LongTensor)
                #country_pred = label_pred.type(torch.LongTensor)
                loss_1 = loss_func(label_pred, label)
                loss_country = loss_func(country_pred, country)
                loss = 0.8*loss_1 + 0.2*loss_country
            else:
                label_pred = model(input)
                #label_pred = label_pred.type(torch.LongTensor)
                loss = loss_func(label_pred, label)

            #logger.debug(country_pred[1])
            #loss_country = loss_func(country_pred, country)
            #loss = loss_label + loss_country
            acc = binary_acc(label_pred, label)
            #country_acc = binary_acc(country_pred, country)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

        if config.avg_params:
            if epoch > swa_start:
                swa_model.update_parameters(model)
                swa_scheduler.step()
            else:
                scheduler.step()

        wandb_dict = make_wandb_dict(config.dataset_names, datasets, data_loaders, model)
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
                            'dataset_names': list(config.dataset_names),
                            'source_limits': dict(config.source_limits),
                            'n_train': len(es_train_data),
                            'n_val': {n: len(d) for n, d in
                                      zip(config.dataset_names, datasets)}}
        with open(metrics_out, 'w') as f:
            json.dump(final, f, indent=2)
        logger.info(f"Wrote metrics to {metrics_out}")

    today = datetime.datetime.today().strftime('%Y-%m-%d')
    logger.info(f"Saving model to mordecai_{today}.pt")
    torch.save(model.state_dict(), f"mordecai_{today}.pt")
    logger.info("Run complete.")



if __name__ == "__main__":
    app()
