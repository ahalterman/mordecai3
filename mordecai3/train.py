import os
import pickle
import random
import re

import jsonlines

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import datetime
import logging

import elastic_utilities as es_util
import numpy as np
import spacy
import torch
import torch.nn as nn
import torch.optim as optim
import typer
import wandb
import xmltodict
from error_utils import make_wandb_dict
from geoparse import guess_in_rel

from torch_model import geoparse_model
import elastic_utilities as es_util

from mordecai_utilities import spacy_doc_setup
from spacy.tokens import DocBin
from torch.utils.data import DataLoader
from torch_model import TrainData, geoparse_model
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
    if re.search("NGEC\.", i.name):
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

def split_list(data, frac=0.7):
    split = round(frac*len(data))
    return data[0:split], data[split:]

def load_data(data_dir, 
             max_results, 
             limit_types, 
             fuzzy,
             batch_size,
             test_batch_size,
             train_frac=0.7,
             data_sources=["Prodigy", "TR", "LGL", "GWN", "Synth", "Wiki"]):
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
            # mean of 'correct' key
            #np.mean([np.mean(i['correct']) for i in es_data])

        es_data = [i for i in es_data if len(i['tensor']) > 1] # This is really weird!! Some sort of bug in the spacy step
        es_data, es_data_val = split_list(es_data, train_frac)
        logger.debug(f"Training examples from {source}: {len(es_data)}")
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

def data_to_docs(data, source, base_dir, nlp):
    """
    search data is involved yet.
    """
    # NOTE: doing more than 5000 at a time maxes out RAM. 
    # To get the full set of Wiki stories, we'll need to batch and
    # save each batch to disk.
    print("NLPing docs...")
    doc_bin = DocBin(store_user_data=True)
    print("spaCy batch size: ", nlp.batch_size)
    if source in ["prodigy", "syn_cities", "syn_caps", "wiki"]:
        for doc in tqdm(nlp.pipe([i['text'] for i in data], 
                                 batch_size=100),
                                   total=len(data)):
            doc_bin.add(doc)
    else:
        for doc in tqdm(nlp.pipe([i['text'] for i in data['articles']['article']]), total=len(data['articles']['article'])):
            doc_bin.add(doc)
    fn = f"{base_dir}/spacyed/source_{source}.spacy"
    print(f"Writing NLPed docs out to {fn}...")
    with open(fn, "wb") as f:
        doc_bin.to_disk(fn)
    print(f"Wrote NLPed docs out to {fn}")


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
    if source in ["syn_cities", "syn_caps", "wiki"]:
        articles = data
    else:
        articles = data['articles']['article']
    for doc, ex in tqdm(zip(docs, articles), total=len(docs), leave=False):
        doc_formatted = []
        doc_tensor = np.mean(np.vstack([i._.tensor for i in doc]), axis=0)
        loc_ents = [ent for ent in doc.ents if ent.label_ in ['GPE', 'LOC']]
        for n, topo in enumerate(ex['toponyms']['toponym']):
            #print(topo['phrase'])
            if source == "gwn" and 'geonamesID' not in topo.keys():
                continue
            if source == "gwn" and not topo['geonamesID']:
                continue
            try:
                place_tokens = [i for i in doc if i.idx >= int(topo['start']) and i.idx + len(i) <= int(topo['end'])]
                other_locs = [i for e in loc_ents for i in e if i not in place_tokens]
                if other_locs:
                    locs_tensor = np.mean(np.vstack([i._.tensor for i in other_locs]), axis=0)
                else:
                    locs_tensor = np.zeros(len(tensor))
                # remove NORPs?
                gpes = [i for i in place_tokens if i.ent_type_ in ['GPE', 'LOC']]
                if not gpes:
                    continue
                tensor = np.mean(np.vstack([i._.tensor for i in place_tokens]), axis=0)
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
                print(f"{e}: {doc_num}_{n}")
        all_formatted.append(doc_formatted)
        doc_num += 1
    return all_formatted

#base_dir = "../raw_data/"
#source = "wiki"
#limit_types = "all_loc_types"
#max_results = 500
#fuzzy = 0
# !!!

def format_source(base_dir, source, conn, max_results, fuzzy, 
                 limit_types, source_dict, nlp, remove_correct=False):
    print(f"limit types: {limit_types}")
    fn = f"source_{source}.spacy"
    fn = os.path.join(base_dir, "spacyed", fn)
    print(f"===== {source} =====")

    with open(fn, "rb") as f:
        doc_bin = DocBin().from_disk(fn)
    print(f"Converting back to spaCy docs...")

    docs = list(doc_bin.get_docs(nlp.vocab))
    data = read_file(source_dict[source])

    if source == "prodigy":
        formatted = data_formatter_prodigy(docs, data)
    elif source == "wiki":
        formatted = data_formatter_wiki(docs, data)
    else:
        formatted = data_formatter(docs, data, source)
    # formatted is a list of lists. We want the final data to be a flat list.
    # At the same time, we can exclude examples with missing geonames info 
    esed_data = []
    print("Adding Elasticsearch data...")
    #with multiprocessing.Pool(8) as p:
    #    esed_data = p.starmap(es_util.add_es_data_doc, zip(formatted, repeat(conn), repeat(max_results), 
    #                                                       repeat(fuzzy), repeat(limit_types), 
    #                                                       repeat(remove_correct)))
    for ff in tqdm(formatted, leave=False):
        esd = es_util.add_es_data_doc(ff, conn, max_results, fuzzy, limit_types, remove_correct)
        for e in esd:
            if e['correct_geonamesid'] != None:
                esed_data.append(e)

    if limit_types == True:
        limit_type_str = "pa_only"
    else:
        limit_type_str = "all_loc_types"
    print(f"Total place names in {source}: {len(esed_data)}")
    fn = f"es_formatted_{source}_{max_results}_{limit_type_str}_fuzzy_{fuzzy}.pkl"
    out_file = os.path.join(base_dir, "pickled_es", fn)
    print(f"Writing to {out_file}...")
    with open(out_file, 'wb') as f:
        pickle.dump(esed_data, f)

##################################

app = typer.Typer(add_completion=True)


@app.command()
def nlp_docs(base_dir, 
            sources = "tr, lgl, gwn, prodigy, syn_cities, syn_caps, wiki"):
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
                  "wiki": "wiki/wiki_training_data_sents.jsonl"}
    for k, v in source_dict.items():
        source_dict[k] = os.path.join(base_dir, v)

    print("Reading in data...")
    print("sources: ", sources, type(sources))
    if type(sources) is str:
        sources = [i.strip() for i in sources.split(",")]
    for source in sources:
        print(source_dict[source])
        data = read_file(source_dict[source])
        data_to_docs(data, source, base_dir, nlp)

@app.command()
def add_es(base_dir, 
          max_results=500,
          fuzzy=0, 
          limit_types = False,
          sources= "tr, lgl, gwn, prodigy, syn_cities, syn_caps, wiki"):
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
    conn = es_util.make_conn()
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
                  "wiki": "wiki/wiki_training_data_sents.jsonl"}
    for k, v in source_dict.items():
        source_dict[k] = os.path.join(base_dir, v)
    if type(sources) is str:
        sources = [i.strip() for i in sources.split(",")]
    for source in sources:
        remove_correct = source == "wiki_incorrect"
        format_source(base_dir, 
                      source, 
                      conn, 
                      max_results=max_results, 
                      limit_types=limit_types, 
                      fuzzy=fuzzy,
                      source_dict=source_dict, 
                      nlp=nlp,
                      remove_correct=remove_correct)
    print("Complete")


@app.command()
def train(batch_size: int = typer.Option(32, "--batch_size"),         # input batch size for training 
          test_batch_size: int= typer.Option(64, "--test_batch_size"),    # input batch size for testing 
          epochs: int = typer.Option(20, "--epochs"),         # number of epochs to train 
          lr: float = typer.Option(0.001, "--lr"),            # learning rate 
          max_choices: int = typer.Option(500, "--max_choices"),
          dropout: float = typer.Option(0.3, "--dropout"),
          avg_params: str = typer.Option("False", "--avg_params"),
          limit_es_results: str = typer.Option("all_loc_types", "--limit_es_results"),
          country_size: int = typer.Option(24, "--country_size"),
          code_size: int = typer.Option(8, "--code_size"),
          country_pred: str = typer.Option("False", "--country_pred"),
          mix_dim: int = typer.Option(24, "--mix_dim"),
          fuzzy: int = typer.Option(0, "--fuzzy"),
          dataset_names: str = typer.Option("Prodigy, TR, LGL, GWN, Synth, Wiki", "--datasets")
):
    """
    Train the pytorch model from formatted training data.
    """
    wandb.init(project="mordecai3", entity="ahalt", allow_val_change=True)

    config = wandb.config          # Initialize config
    config.batch_size = batch_size
    config.test_batch_size = test_batch_size 
    config.epochs = epochs 
    config.lr = lr
    config.seed = 42          
    config.log_interval = 10
    config.max_choices = max_choices
    config.dropout = dropout 
    config.avg_params = avg_params=="True"
    config.limit_es_results = limit_es_results 
    config.country_size = country_size
    config.code_size = code_size
    config.country_pred = country_pred=="True"
    config.mix_dim = mix_dim
    config.fuzzy = fuzzy
    dataset_names = [i.strip() for i in dataset_names.split(",")]
    config.names = dataset_names 
    #dataset_names = ['Prodigy', 'TR', 'LGL', 'Synth', 'Wiki'] 

    data_dir = "../raw_data"
    print(config.__dict__)

    train_loader, es_train_data, data_loaders, datasets = load_data(data_dir, 
                                                  config.max_choices, 
                                                  config.limit_es_results,
                                                  config.fuzzy,
                                                  config.batch_size,
                                                  config.test_batch_size,
                                                  data_sources=dataset_names) 
    logger.info(f"Total training examples: {len(es_train_data)}")

    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    model = geoparse_model(device = device,
                              bert_size = es_train_data[0]['tensor'].shape[0],
                              num_feature_codes=53+1,
                              dropout = config.dropout,
                              country_size=config.country_size,
                              code_size=config.code_size, 
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
    for epoch in range(1, config.epochs+1):
        epoch_loss = 0
        epoch_acc = 0

        for label, country, input in train_loader:
            label = label.type(torch.LongTensor) #.to(device)
            # input is a dict. It should be moved to device
            #for k, v in input.items():
            #    input[k] = v.to(device)
            optimizer.zero_grad()
            if config.country_pred:
                label = label.type(torch.LongTensor)
                label_pred, country_pred = model(input)
                #label_pred = label_pred.type(torch.LongTensor)
                #country_pred = label_pred.type(torch.LongTensor)
                loss_1 = loss_func(label_pred, label)
                loss_country = loss_func(country_pred, country)
                loss = 0.8*loss_1 + 0.2*loss_country
            else:
                label = label.type(torch.LongTensor)
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

        wandb_dict = make_wandb_dict(config.names, datasets, data_loaders, model)
        wandb_dict['loss'] = epoch_loss/len(train_loader)

        print(f"Epoch {epoch+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Exact Match: {wandb_dict['exact_match_avg']:.3f} | Country Match: {wandb_dict['country_avg']:.3f}")  # | Prodigy Acc: {epoch_acc_prod/len(prod_loader):.3f} | TR Acc: {epoch_acc_tr/len(tr_loader):.3f} | LGL Acc: {epoch_acc_lgl/len(lgl_loader):.3f} | GWN Acc: {epoch_acc_gwn/len(gwn_loader):.3f} | Syn Acc: {epoch_acc_syn/len(syn_loader):.3f}')
        wandb.log(wandb_dict)

    logger.info("Saving model...")
    today = datetime.datetime.today().strftime('%Y-%m-%d')
    torch.save(model.state_dict(), f"mordecai_{today}.pt")
    logger.info("Run complete.")

if __name__ == "__main__":
    app()
