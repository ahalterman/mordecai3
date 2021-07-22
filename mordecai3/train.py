import random
import pickle
import re
import os
import jsonlines

import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import xmltodict
import wandb
import typer
import click_spinner
import spacy
from spacy.tokens import DocBin
from spacy.pipeline import Pipe

from torch_model import geoparse_model
import elastic_utilities as es_util
from utilities import spacy_doc_setup
from torch_model import TrainData, ProductionData
from error_utils import make_wandb_dict, evaluate_results

import logging
logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter(
        '%(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.WARN)

spacy_doc_setup()

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.argmax(y_pred, axis=1)
    #y_test_cat = torch.argmax(y_test, axis=1) 
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_pred.shape[0]
    acc = torch.round(acc * 100)
    return acc

def read_file(fn):
    if re.search("xml", fn):
        with open(fn, "r") as f:
            xml = f.read()
            data = xmltodict.parse(xml)
    elif re.search("jsonl", fn):
        with jsonlines.open(fn, "r") as f:
            data = list(f.iter())
    else:
        raise NotImplementedError("Don't know how to handle this filetype")
    return data 


def load_data(data_dir, limit_es_results=False):
    """
    Load formatted training data with Elasticsearch results

    Parameters
    --------
    limit_es_results: bool
      If True, use the training that only has populated place and area features,
      not other geographic features. Defaults to False

    Returns
    -------
    list
      a list of formatted, shuffled training data
    """
    if limit_es_results:
        path_mod = "pa_only"
    else:
        path_mod = "all_loc_types"
    with open(f'{data_dir}/{path_mod}/es_formatted_prodigy.pkl', 'rb') as f:
        es_data_prod = pickle.load(f)
    
    with open(f'{data_dir}/{path_mod}/es_formatted_tr.pkl', 'rb') as f:
        es_data_tr = pickle.load(f)
    
    with open(f'{data_dir}/{path_mod}/es_formatted_lgl.pkl', 'rb') as f:
        es_data_lgl = pickle.load(f)

    with open(f'{data_dir}/{path_mod}/es_formatted_gwn.pkl', 'rb') as f:
        es_data_gwn = pickle.load(f)
    
    with open(f'{data_dir}/{path_mod}/es_formatted_syn_cities.pkl', 'rb') as f:
        es_data_syn = pickle.load(f)

    with open(f'{data_dir}/{path_mod}/es_formatted_syn_caps.pkl', 'rb') as f:
        es_data_syn_caps = pickle.load(f)

    def split_list(data, frac=0.7):
        split = round(frac*len(data))
        return data[0:split], data[split:]

    es_data_prod, es_data_prod_val = split_list(es_data_prod)
    es_data_tr, es_data_tr_val = split_list(es_data_tr)
    es_data_lgl, es_data_lgl_val = split_list(es_data_lgl)
    es_data_gwn, es_data_gwn_val = split_list(es_data_gwn)
    es_data_syn, es_data_syn_val = split_list(es_data_syn)
    train_data = es_data_prod + es_data_tr + es_data_lgl + es_data_gwn + es_data_syn
    random.seed(617)
    random.shuffle(train_data)
    return train_data, es_data_prod_val, es_data_tr_val, es_data_lgl_val, es_data_gwn_val, es_data_syn_val


def data_formatter_prodigy(docs, data):
    """
    Format the annotated documents from the Prodigy training round into a format for training.
    
    Returns a list of lists, with one list for each document, consisting of each entity
    within the document. This round of training only annotated one location per
    "document"/sentence, so non-annotated entitys have None as their value for
    correct_geonamesid. These will be discarded later.

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
        placename = ''.join([i.text_with_ws for i in places])
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
            d = {"placename": placename,
               "tensor": tensor,
                "locs_tensor": locs_tensor,
                "doc_tensor": doc_tensor,
               "correct_geonamesid": correct_id}
            doc_formatted.append(d)
            # Only one place name is annotated in each example, but we still want to know
            # the other place names that were extracted to calculate Geonames overlap
            # features. We'll throw these away later, so we can set the other values
            # to None. 
            for loc in other_locs:
                d = {"placename": loc.text,
                     "tensor": None,
                     "locs_tensor": None,
                     "doc_tensor": None,
                     "correct_geonamesid": None}
                doc_formatted.append(d)
            all_formatted.append(doc_formatted)
        doc_num += 1
    return all_formatted

def data_to_docs(data, source, nlp):
    """
    NLP the training data and save the docs to disk
    """
    print("NLPing docs...")
    doc_bin = DocBin(store_user_data=True)
    if source in ["prodigy", "syn_cities", "syn_caps"]:
        for doc in tqdm(nlp.pipe([i['text'] for i in data]), total=len(data)):
            doc_bin.add(doc)
    else:
        for doc in tqdm(nlp.pipe([i['text'] for i in data['articles']['article']]), total=len(data['articles']['article'])):
            doc_bin.add(doc)
    fn = f"source_{source}.spacy"
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
    if source in ["syn_cities", "syn_caps"]:
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
                    placename = topo['extractedName']
                elif source in ["syn_cities", "syn_caps"]:
                    correct_geonamesid = topo['geonamesID']
                    placename = topo['placename']
                else:
                    correct_geonamesid = topo['gaztag']['@geonameid']
                    placename = topo['phrase']

                doc_formatted.append({"placename": placename,
                                  "tensor": tensor,
                                  "locs_tensor": locs_tensor,
                                  "doc_tensor": doc_tensor,
                                  "correct_geonamesid": correct_geonamesid})
            except Exception as e:
                #pass
                print(e)
                #print(f"{doc_num}_{n}")
        all_formatted.append(doc_formatted)
        doc_num += 1
    return all_formatted



def format_source(base_dir, source, conn, max_results, limit_types, source_dict, nlp):
    print(f"limit types: {limit_types}")
    fn = f"source_{source}.pkl"
    fn = os.path.join(base_dir, fn)
    print(f"===== {source} =====")
    # did it two different ways...
    try:
        with open(fn, "rb") as f:
            doc_bin = pickle.load(f)

    except FileNotFoundError:
        fn = f"source_{source}.spacy"
        with open(fn, "rb") as f:
            doc_bin = DocBin().from_disk(fn)
    print(f"Converting back to spaCy docs...")

    docs = list(doc_bin.get_docs(nlp.vocab))
    data = read_file(source_dict[source])

    if source == "prodigy":
        formatted = data_formatter_prodigy(docs, data)
    else:
        formatted = data_formatter(docs, data, source)
    # formatted is a list of lists. We want the final data to be a flat list.
    # At the same time, we can exclude examples with missing geonames info 
    esed_data = []
    print("Adding Elasticsearch data...")
    for ff in tqdm(formatted, leave=False):
        esd = es_util.add_es_data_doc(ff, conn, max_results, limit_types)
        for e in esd:
            if e['correct_geonamesid']:
                esed_data.append(e)

    print(f"Total place names in {source}: {len(esed_data)}")
    fn = f"es_formatted_{source}.pkl"
    print(f"Writing to {fn}...")
    with open(fn, 'wb') as f:
        pickle.dump(esed_data, f)

##################################

app = typer.Typer()

@app.command()
def nlp_docs(base_dir, sources=['tr', 'lgl', 'gwn', 'prodigy', 'syn_cities', 'syn_caps']):
    """
    Run spaCy over a list of training data sources and save the output.

    Parameters
    ---------
    base_dir: Path
      path to the directory with training data
    sources: list
      
    """
    print("Loading NLP stuff...")
    with click_spinner.spinner():
        nlp = spacy.load("en_core_web_trf")
    nlp.add_pipe("token_tensors")
    source_dict = {"tr":"Pragmatic-Guide-to-Geoparsing-Evaluation/data/Corpora/TR-News.xml",
                  "lgl":"Pragmatic-Guide-to-Geoparsing-Evaluation/data/corpora/lgl.xml",
                  "gwn": "Pragmatic-Guide-to-Geoparsing-Evaluation/data/GWN.xml",
                  "prodigy": "orig_mordecai/loc_rank_db.jsonl",
                  "syn_cities": "synth_raw/synthetic_cities_short.jsonl",
                  "syn_caps": "synth_raw/synth_caps.jsonl"}
    for k, v in source_dict.items():
        source_dict[k] = os.path.join(base_dir, v)

    print("Reading in data...")
    for s in sources:
        data = read_file(source_dict[s])
        data_to_docs(data, s, nlp)

@app.command()
def add_es(base_dir, 
          max_results=500, 
          limit_types = False,
          sources=['tr', 'lgl', 'gwn', 'prodigy', 'syn_cities', 'syn_caps']):
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
    with click_spinner.spinner():
        nlp = spacy.load("en_core_web_trf")
    nlp.add_pipe("token_tensors")
    source_dict = {"tr":"Pragmatic-Guide-to-Geoparsing-Evaluation/data/Corpora/TR-News.xml",
                  "lgl":"Pragmatic-Guide-to-Geoparsing-Evaluation/data/corpora/lgl.xml",
                  "gwn": "Pragmatic-Guide-to-Geoparsing-Evaluation/data/GWN.xml",
                  "prodigy": "orig_mordecai/geo_annotated/loc_rank_db.jsonl",
                  "syn_cities": "synth_raw/synthetic_cities_short.jsonl",
                  "syn_caps": "synth_raw/synth_caps.jsonl"}
    for k, v in source_dict.items():
        source_dict[k] = os.path.join(base_dir, v)
    for source in sources:            
        format_source(base_dir, source, conn, max_results=max_results, limit_types=limit_types, source_dict=source_dict, nlp=nlp)
    print("Complete")

@app.command()
def train():
    """
    Train the pytorch model from formatted training data.
    """
    names = ["mixed training", "prodigy", "TR", "LGL", "GWN", "Synth"]
    wandb.init(project="mordecai3", entity="ahalt")

    config = wandb.config          # Initialize config
    config.batch_size = 32         # input batch size for training 
    config.test_batch_size = 64    # input batch size for testing 
    config.epochs = 12          # number of epochs to train 
    config.lr = 0.01               # learning rate 
    config.seed = 42               # random seed (default: 42)
    config.log_interval = 10     # how many batches to wait before logging training status
    config.max_choices = 500
    config.avg_params = True
    config.limit_es_results = False
    data_dir = "../raw_data"

    es_train_data, es_data_prod_val, es_data_tr_val, es_data_lgl_val, es_data_gwn_val, es_data_syn_val  = load_data(data_dir, config.limit_es_results)
    logger.info(f"Total training examples: {len(es_train_data)}")
    
    train_data = TrainData(es_train_data, max_choices=config.max_choices)
    tr_data = TrainData(es_data_tr_val, max_choices=config.max_choices)
    prod_data = TrainData(es_data_prod_val, max_choices=config.max_choices)
    lgl_data = TrainData(es_data_lgl_val, max_choices=config.max_choices)
    gwn_data = TrainData(es_data_gwn_val, max_choices=config.max_choices)
    syn_data = TrainData(es_data_syn_val, max_choices=config.max_choices)

    train_loader = DataLoader(dataset=train_data, batch_size=config.batch_size, shuffle=True)
    train_val_loader = DataLoader(dataset=train_data, batch_size=config.batch_size, shuffle=False)
    prod_loader = DataLoader(dataset=prod_data, batch_size=config.test_batch_size, shuffle=False)
    tr_loader = DataLoader(dataset=tr_data, batch_size=config.test_batch_size, shuffle=False)
    lgl_loader = DataLoader(dataset=lgl_data, batch_size=config.test_batch_size, shuffle=False)
    gwn_loader = DataLoader(dataset=gwn_data, batch_size=config.test_batch_size, shuffle=False)
    syn_loader = DataLoader(dataset=syn_data, batch_size=config.test_batch_size, shuffle=False)

    datasets = [es_train_data, es_data_prod_val, es_data_tr_val, es_data_lgl_val, es_data_gwn_val, es_data_syn_val]
    data_loaders = [train_val_loader, prod_loader, tr_loader, lgl_loader, gwn_loader, syn_loader]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = geoparse_model(device = device,
                              bert_size = train_data.placename_tensor.shape[1],
                              num_feature_codes=53+1)
    #model = torch.nn.DataParallel(model)
    model.to(device)
    # Future work: Can add  an "ignore_index" argument so that some inputs don't have losses calculated
    loss_func=nn.CrossEntropyLoss() # single label, multi-class
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    if config.avg_params:
        from torch.optim.swa_utils import AveragedModel, SWALR
        from torch.optim.lr_scheduler import CosineAnnealingLR

        swa_model = AveragedModel(model)
        scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs+1)
        swa_start = 5
        swa_scheduler = SWALR(optimizer, swa_lr=0.05)
    #else:
    #    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs+1)

    wandb.watch(model, log='all')

    model.train()
    for epoch in range(1, config.epochs+1):
        epoch_loss = 0
        epoch_acc = 0

        for label, country, input in train_loader:
            optimizer.zero_grad()
            label_pred = model(input)

            #logger.debug(country_pred[1])
            loss = loss_func(label_pred, label)
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

        wandb_dict = make_wandb_dict(names, datasets, data_loaders, model)
        wandb_dict['loss'] = epoch_loss/len(train_loader)

        print(f"Epoch {epoch+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | Exact Match: {wandb_dict['exact_match_avg']:.3f} | Country Match: {wandb_dict['country_avg']:.3f}")  # | Prodigy Acc: {epoch_acc_prod/len(prod_loader):.3f} | TR Acc: {epoch_acc_tr/len(tr_loader):.3f} | LGL Acc: {epoch_acc_lgl/len(lgl_loader):.3f} | GWN Acc: {epoch_acc_gwn/len(gwn_loader):.3f} | Syn Acc: {epoch_acc_syn/len(syn_loader):.3f}')
        wandb.log(wandb_dict)

    logger.info("Saving model...")
    torch.save(model.state_dict(), "mordecai_new.pt")
    logger.info("Run complete.")

if __name__ == "__main__":
    app()
