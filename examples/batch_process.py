import logging
import time
from pathlib import Path

import jsonlines
import numpy as np
import pandas as pd
import spacy
import torch
import typer
from torch.utils.data import DataLoader
from tqdm import tqdm
from utilities import spacy_doc_setup

import mordecai3.elastic_utilities as es_util
from mordecai3.geoparse import doc_to_ex_expanded
from mordecai3.roberta_qa import setup_qa
from mordecai3.torch_model import ProductionData, geoparse_model

logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter(
        '%(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

logging.getLogger("elasticsearch").setLevel(logging.WARN)

spacy_doc_setup()

country_info = pd.read_csv("../mordecai3/assets/countryInfo.txt", sep="\t")
country_dict = dict(zip(country_info['ISO3'], country_info['Country']))



def load_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = geoparse_model(device = device,
                                bert_size = 768,
                                num_feature_codes=54) 
    model.load_state_dict(torch.load("../mordecai3/assets/mordecai_prod.pt"))
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
        if len(geo) == 0:
            d['event_loc'] = None
            d['event_loc_reason'] = "No locations found"
        elif len(geo) > 1:
            if len(set([i['name'] for i in geo])) == 1:
                d['event_loc'] = geo[0]
                d['event_loc_reason'] = "No location identified in event location model, but only one (unique) location in text"
            else:
                d['event_loc'] = None
                d['event_loc_reason'] = "Multiple locations, but none identified as the event location"
        else:
            d['event_loc'] = geo[0]
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
    """
    Geoparse a set of news stories with event information. 

    This is a batch process script for (1) identifying locations in text
    (2) resolving them to their geographic coordinates, and (3) assigning
    events in the text to their reported locations.

    For additional speedup, if you're processing ICEWS formatted data (not production)
    you can parallelize the process over chunks of an 
    input file using GNU parallell. e.g.:

    sed 1d events.20210301073501.Release507.csv | parallel -j4 --pipe --block 1M "cat > {#}; python batch_process.py {#} --input_format=icews"

    Inputs
    ------
    filename: Path
      The files of text and events to read in and process
    qa_batch_size: int
      Parameter for how many documents the event-location linking model
      will process in a single step. If running on a GPU, this may need to be
      smaller. Conversely, if running on CPU, this could be larger if the available
      RAM will support it.
    input_format: str
      One of "production" or "icews". If using the script in a production setting,
      set to "production". The "icews" option will read in a final formatted ICEWS
      CSV and perform a second round of geolocation. This is only used for comparisions
      with the existing geoparsing system.
    limit: int
      For testing use only: limit the number of events to be processed. Defaults
      to all events in the file.

    Returns
    -------
    None
      Writes an output file to "mordecai_geo_results_{filename}.csv" where
      `filename` is the input file.

    Example
    ------
    python batch_process.py storiesWithEvents-3.json
    """
    logger.setLevel(logging.INFO)
    logger.info("Loading spaCy model...")
    nlp = spacy.load("en_core_web_trf", exclude=["tagger", "lemmatizer"])
    nlp.add_pipe("token_tensors")

    conn = es_util.setup_es()
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
    # free up memory
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
                logger.debug(f"no locations found for {ent['search_name']}")
            results = sorted(results, key=lambda k: -k['score'])
            results = [i for i in results if i['score'] > 0.01]
            try:
                results = results[0]
                r = {"extracted_name": ent['search_name'],
                    "name": results['name'],
                    "lat": results['lat'],
                    "admin1_name": results['admin1_name'],
                    "admin2_name": results['admin2_name'],
                    "country_code3": results['country_code3'],
                    "feature_code": results['feature_code'],
                    "lon": results['lon'],
                    "geonameid": results['geonameid'],
                    "start_char": ent['start_char'],
                    "end_char": ent['end_char']}
            except Exception as e:
                logger.debug(e, d)
                r = None
            resolved_entities.append(r)
        d['geo'] = resolved_entities
        return d

    for d in tqdm(doc_info):
        d = resolve_entities(d)

    logger.debug(f"qa_batch_size: {qa_batch_size}")
    #chunks = (len(doc_info) - 1) // qa_batch_size + 1

    # it doesn't do well with single document batches. If one is going
    # to be produced, increase the batch size by 1 to avoid
    if len(doc_info) % qa_batch_size == 1:
        qa_batch_size = qa_batch_size + 1
    chunks = int(np.ceil(float(len(doc_info)) / qa_batch_size))
    
    
    logger.info("Running event-location linking model...")
    logger.debug(f"Doc info length: {len(doc_info)}")
    logger.debug(f"Batch size: {qa_batch_size}")
    logger.debug(f"Chunk number: {chunks}")
    for c in tqdm(range(chunks)):
        start = c*qa_batch_size
        end = (c+1)*qa_batch_size
        if end > len(doc_info):
            end = len(doc_info) + 1
        logger.debug(f"Batch start: {start}. Batch end: {end}")
        batch = doc_info[start:end]
        logger.debug(f"Actual batch length: {len(batch)}")
        if len(batch) == 0:
            continue
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
            event_dict[n]['mordecai_resolved_place'] = d['event_loc']['name']
            try:
                country_name = country_dict[d['event_loc']['country_code3']]
            except KeyError:
                country_name = d['event_loc']['country_code3']
            event_dict[n]['mordecai_district'] = d['event_loc']['admin2_name']
            event_dict[n]['mordecai_province'] = d['event_loc']['admin1_name']
            event_dict[n]['mordecai_country'] = country_name
            event_dict[n]['mordecai_lat'] = d['event_loc']['lat']
            event_dict[n]['mordecai_lon'] = d['event_loc']['lon']
            event_dict[n]['mordecai_geonameid'] = d['event_loc']['geonameid']
            event_dict[n]['mordecai_event_loc_reason'] = d['event_loc_reason']
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