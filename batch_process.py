import pandas as pd 
from collections import Counter
import typer
from pathlib import Path

import spacy
from spacy.language import Language
from spacy.tokens import Token, Doc
from spacy.pipeline import Pipe
import numpy as np
import jsonlines
from tqdm import tqdm
import re
import torch
import pandas as pd
import time
import jsonlines

from torch.utils.data import Dataset, DataLoader
import elastic_utilities as es_util
from format_geoparsing_data import doc_to_ex_expanded
from torch_bert_placename_compare import ProductionData, embedding_compare
from roberta_qa import setup_qa, add_event_loc

from elasticsearch import Elasticsearch, helpers
from elasticsearch_dsl import Search, Q

import logging
logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter(
        '%(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

logging.getLogger("elasticsearch").setLevel(logging.WARN)

try:
    Token.set_extension('tensor', default=False)
except ValueError:
    pass

try:
    @Language.component("token_tensors")
    def token_tensors(doc):
        chunk_len = len(doc._.trf_data.tensors[0][0])
        token_tensors = [[]]*len(doc)
        
        for n, i in enumerate(doc):
            wordpiece_num = doc._.trf_data.align[n]
            for d in wordpiece_num.dataXd:
                which_chunk = int(np.floor(d[0] / chunk_len))
                which_token = d[0] % chunk_len
                ## You can uncomment this to see that spaCy tokens are being aligned with the correct 
                ## wordpieces.
                #wordpiece = doc._.trf_data.wordpieces.strings[which_chunk][which_token]
                #print(n, i, wordpiece)
                token_tensors[n] = token_tensors[n] + [doc._.trf_data.tensors[0][which_chunk][which_token]]
        for n, d in enumerate(doc):
            if token_tensors[n]:
                d._.set('tensor', np.mean(np.vstack(token_tensors[n]), axis=0))
            else:
                d._.set('tensor',  np.zeros(doc._.trf_data.tensors[0].shape[-1]))
        return doc
except ValueError:
    pass

country_info = pd.read_csv("data/countryInfo.txt", sep="\t")
country_dict = dict(zip(country_info['ISO3'], country_info['Country']))

def setup_es():
    kwargs = dict(
        hosts=['localhost'],
        port=9200,
        use_ssl=False,
    )
    CLIENT = Elasticsearch(**kwargs)
    try:
        CLIENT.ping()
        logger.info("Successfully connected to Elasticsearch.")
    except:
        ConnectionError("Could not locate Elasticsearch. Are you sure it's running?")
    conn = Search(using=CLIENT, index="geonames")
    return conn

def load_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = embedding_compare(device = device,
                                bert_size = 768,
                                num_feature_codes=54) 
    model.load_state_dict(torch.load("data/mordecai_prod.pt"))
    model.eval()
    return model



def pick_event_loc(d):
    """
    Heuristic rules for picking the best event location when multiple are present with
    explanations for why each was selected.
    """
    if not d['geo']:
        d['event_loc'] = None
        d['event_loc_reason'] = "No locations found"
        return d
    loc_start = d['qa_output']['start']
    loc_end = d['qa_output']['end']
    geo = [i for i in d['geo'] if i]
    loc_ents = [i for i in geo if i['start_char'] in range(loc_start, loc_end+1)]
    if not loc_ents:
        if len(d['geo']) == 0:
            d['event_loc'] = None
            d['event_loc_reason'] = "No locations found"
        elif len(d['geo']) > 1:
            if len(set([i['placename'] for i in d['geo']])) == 1:
                d['event_loc'] = d['geo'][0]
                d['event_loc_reason'] = "No location identified in event location model, but only one (unique) location in text"
            else:
                d['event_loc'] = None
                d['event_loc_reason'] = "Multiple locations, but none identified as the event location"
        else:
            d['event_loc'] = d['geo'][0]
            d['event_loc_reason'] = "No location identified in event location model, but only one location in text"
    else:
        if len(loc_ents) == 1:
            d['event_loc'] = loc_ents[0]
            d['event_loc_reason'] = "Match: single geo result overlapping with event location"
        else:
            places = [i for i in loc_ents if i['feature_code'][0] == "P"]
            if places:
                d['event_loc'] = places[0]
                d['event_loc_reason'] = "Multiple locations are within the event location: picking the first non-admin unit"
            else:
                d['event_loc'] = None
                d['event_loc_reason'] = "Multiple locations within event location. None are feature type places."
    return d


def read_icews(filename):
    """Read in final ICEWS formatted data. (Useful for comparision with existing geoparser)."""
    events = pd.read_csv(filename, sep="\t", header=None)
    events.columns = ['Event ID', 'Event Date', 'Source Name', 'Source Sectors',
       'Source Country', 'Event Text', 'CAMEO Code', 'Intensity',
       'Target Name', 'Target Sectors', 'Target Country', 'Story ID',
       'Sentence Number', 'Event Sentence', 'Publisher', 'Source', 'Headline',
       'City', 'District', 'Province', 'Country', 'Latitude', 'Longitude',
       'productID', 'holdingID', 'Language']

    event_dict = events.to_dict('records')
    return event_dict

def read_production(filename):
    """
    Read in a production event file. Use this one in production.
    """
    with jsonlines.open(filename, "r") as f:
        stories = list(f.iter())
    expanded_events = []
    for story in stories:
        for e in story['events']:
            d = {"Event ID": e['event_id'],
                "Event Sentence": e['text'],
                "Event Text": e['name'],
                "Sentence Number": e['sentence_num'],
                "Headline": story['Headline'],
                "Story ID": story['storyid']}
            expanded_events.append(d)
    return expanded_events

def main(filename: Path,
        qa_batch_size: int=50,
        input_format: str="production",
        limit: int=-1):
    logger.info("Loading spaCy model...")
    nlp = spacy.load("en_core_web_trf", exclude=["tagger", "lemmatizer"])
    nlp.add_pipe("token_tensors")

    conn = setup_es()
    logger.info("Loading geoparsing model...")
    model = load_model()
    logger.info("Loading event-location linking model...")
    trf = setup_qa()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    start = time.time()

    if input_format == "icews":
        event_dict = read_icews(filename)
    elif input_format == "production":
        event_dict = read_production(filename)
    else:
        raise ValueError("Input format must be one of ['icews', 'production']. 'icews' should be used when coding final, formatted ICEWS event records. 'production' should be used in the actual production pipeline.")
    
    event_dict = event_dict[0:limit]
    logger.info(f"Beginning geoparse of {len(event_dict)} events...")

    doc_info = []
    text_list = []
    for i in event_dict:
        d = {"headline": i['Headline'],
            "sentence": i['Event Sentence'],
            "story_id": i['Story ID'],
            "sentence_num": i['Sentence Number'],
            "event_text": i['Event Text']}
        d['text'] =  d['headline'] + ". " + d['sentence'] 
        doc_info.append(d)
        text_list.append(d['text'])

    # Around ~25% of the sentences are duplicates.
    # Dedup these for processing
    text_list = list(set(text_list))
    # Configure as a tuple for spacy processing
    text_list = [(i, i) for i in text_list]

    logger.info("Beginning spaCy parse...")
    docs = list(tqdm(
            nlp.pipe(text_list, batch_size=1000,
                        as_tuples=True), 
        total=len(text_list)))

    # Use the original text as a key
    doc_dict = {i[1]: i[0] for i in docs}
    del docs

    for n, i in enumerate(doc_info):
        # merge the spaCy docs in using the text as a key
        i['doc'] = doc_dict[i['text']]

    def resolve_entities(d):
        doc_ex = doc_to_ex_expanded(d['doc'])
        if not doc_ex:
            d['geo'] = None
            return d
        ## Format for the NN model
        es_data = es_util.add_es_data_doc(doc_ex, conn, max_results=500)
        dataset = ProductionData(es_data, max_choices=500)
        data_loader = DataLoader(dataset=dataset, batch_size=256, shuffle=False)
        ## run the NN model on all entities within a single document
        with torch.no_grad():
            model.eval()
            for input in data_loader:
                pred_val = model(input)
        # save the results for each document
        resolved_entities = [] 
        for ent, pred in zip(es_data, pred_val):
            for n, score in enumerate(pred):
                if n < len(ent['es_choices']):
                    ent['es_choices'][n]['score'] = score
            results = [e for e in ent['es_choices'] if 'score' in e.keys()]
            if len(results) == 0:
                logger.debug(f"no locations found for {ent['placename']}")
            results = sorted(results, key=lambda k: -k['score'])
            results = [i for i in results if i['score'] > 0.01]
            try:
                results = results[0]
                r = {"extracted_name": ent['placename'],
                    "placename": results['name'],
                    "lat": results['lat'],
                    "admin1_name": results['admin1_name'],
                    "admin2_name": results['admin2_name'],
                    "country_code3": results['country_code3'],
                    "feature_code": results['feature_code'],
                    "lon": results['lon'],
                    "start_char": ent['start_char'],
                    "end_char": ent['end_char']}
            except Exception as e:
                r = None
            resolved_entities.append(r)
        d['geo'] = resolved_entities
        return d

    for d in tqdm(doc_info):
        d = resolve_entities(d)

    chunks = (len(doc_info) - 1) // qa_batch_size + 1
    
    logger.info("Running event-location linking model...")
    for c in tqdm(range(chunks)):
        batch = doc_info[c*qa_batch_size:(c+1)*qa_batch_size]
        batch_QA_input = []
        for d in batch:
            q = {"question": f"Where did {d['event_text'].lower()} happen?",
                   "context": d['text']}
            batch_QA_input.append(q)
        batch_res = trf(batch_QA_input)
        for n, d in enumerate(batch):
            d['qa_output'] = batch_res[n] 


    for d in doc_info:
        d = pick_event_loc(d)

    for n, d in enumerate(doc_info):
        if d['event_loc']:
            event_dict[n]['mordecai_extracted_place'] = d['event_loc']['extracted_name']
            event_dict[n]['mordecai_resolved_place'] = d['event_loc']['placename']
            try:
                country_name = country_dict[d['event_loc']['country_code3']]
            except KeyError:
                country_name = d['event_loc']['country_code3']
            event_dict[n]['mordecai_country'] = country_name
            event_dict[n]['mordecai_adm1'] = d['event_loc']['admin1_name']
            event_dict[n]['mordecai_lat'] = d['event_loc']['lat']
            event_dict[n]['mordecai_lon'] = d['event_loc']['lon']
            event_dict[n]['mordecai_reason'] = d['event_loc_reason']
        else:
            event_dict[n]['mordecai_reason'] = d['event_loc_reason']

    fn = f"mordecai_geo_results_{filename}.csv"
    #if input_format == "icews":
    mordecai_df = pd.DataFrame(event_dict)
    mordecai_df.to_csv(fn)
    logger.info(f"Successfully wrote events to {fn}")
    #else:
    #    event_dict = read_production(filename) 

    end = time.time()
    logger.info(f"Total time elapsed (s): {str(round(end - start, 2))}")

if __name__ == "__main__":
    typer.run(main)