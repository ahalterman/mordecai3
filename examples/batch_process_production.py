import logging
import re
import time
from configparser import ConfigParser
from pathlib import Path

import jsonlines
import numpy as np
import pandas as pd
import spacy
import torch
import typer
from torch.utils.data import DataLoader
from tqdm import tqdm

import mordecai3.elastic_utilities as es_util
from mordecai3.geoparse import Geoparser, doc_to_ex_expanded
from mordecai3.torch_model import ProductionData
from mordecai3.utilities import spacy_doc_setup

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter(
        ' %(levelname)-8s %(name)-8s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False

logging.getLogger("elasticsearch").setLevel(logging.WARN)

spacy_doc_setup()

def read_production(filename):
    """
    Read in a production event file. Use this one in production.
    """
    with jsonlines.open(filename, "r") as f:
        stories = list(f.iter())
    return stories


def main(input_file: Path, config_file: Path):
    """
    Geoparse a set of news stories with event information. 

    This is a batch process script for (1) identifying locations in text
    (2) resolving them to their geographic coordinates, and (3) assigning
    events in the text to their reported locations.

    For additional speedup, if you're processing ICEWS final formatted data (NOT production data)
    you can parallelize the process over chunks of an 
    input file using GNU parallell. e.g.:

    sed 1d events.20210301073501.Release507.csv | parallel -j4 --pipe --block 1M "cat > {#}; python batch_process.py {#} --input_format=icews"

    Inputs
    ------
    input_file: Path
      The files of text and events to read in and process
    config: Path
      Config file specifying the locations of the model, other required files, and other parameters.

    Returns
    -------
    None
      Writes an output file to "mordecai_geo_results_{input_file}.csv" where
      `filename` is the input file.

    Example
    ------
    python batch_process_production.py storiesWithEvents-3.json config.ini
    """
    # Read values from the config file
    parser = ConfigParser()
    parser.read(config_file)
    logging_level = parser.get('setup', 'logging_level')
    logger.setLevel(logging_level)
    limit = parser.getint('setup', 'limit')
    if limit >=0:
        logger.info(f"CAUTION! A story limit is set. Only processing the first {limit} stories.")
    qa_batch_size = parser.getint('setup', 'qa_batch_size')
    spacy_batch_size = parser.getint('setup', 'spacy_batch_size')
    mordecai_model = parser.get('locations', 'mordecai_model')
    mordecai_countryinfo = parser.get('locations', 'mordecai_countryinfo')

    logger.info("Loading spaCy model...")
    nlp = spacy.load("en_core_web_trf", exclude=["tagger", "lemmatizer"])
    nlp.add_pipe("token_tensors")

    logger.info("Loading geoparsing model...")
    geo = Geoparser(nlp=nlp, model_path=mordecai_model)

    country_info = pd.read_csv(mordecai_countryinfo, sep="\t")
    country_dict = dict(zip(country_info['ISO3'], country_info['Country']))

    start_time = time.time()
    logger.debug(start_time)

    story_list = read_production(input_file)

    # add text to events: everything up to and including the event sentence
    # This gets used later in the event location step 
    for i in story_list:
        full_text = '\n'.join([i['Headline'], i['Rawtext']])
        for e in i['events']:
            patt = f"(.*{re.escape(e['text'])})"
            e['partial_doc'] = re.findall(patt, full_text, re.DOTALL)[0]


    story_list = story_list[0:limit]
    logger.info(f"Beginning geoparse of {len(story_list)} documents.")

    text_list = ['\n'.join([i['Headline'], i['Rawtext']]) for i in story_list]

    logger.info("Running spaCy parse...")
    docs = list(tqdm(
            geo.nlp.pipe(text_list, batch_size=spacy_batch_size),
        total=len(text_list)))


    for n, i in enumerate(story_list):
        i['doc'] = docs[n]

    def resolve_entities(d):
        doc_ex = doc_to_ex_expanded(d['doc'])
        if not doc_ex:
            d['geo'] = None
            return d
        ## Format for the NN model
        es_data = es_util.add_es_data_doc(doc_ex, geo.conn, max_results=500)
        dataset = ProductionData(es_data, max_choices=500)
        data_loader = DataLoader(dataset=dataset, batch_size=256, shuffle=False)
        ## run the NN model on all entities within a single document
        with torch.no_grad():
            geo.model.eval()
            for input in data_loader:
                pred_val = geo.model(input)
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

    logger.info("Running geolocation model...")
    for d in tqdm(story_list):
        d = resolve_entities(d)

    #### Switch from stories to events  ######
    event_list = []
    for i in story_list:
        for e in i['events']:
            e['Headline'] = i['Headline']
            e['storyid'] = i['storyid']
            if i['geo']:
                clean_geo = [g for g in i['geo'] if g]
                e['geo'] = [g for g in clean_geo if g['start_char'] < len(e['partial_doc'])]
            else:
                e['geo'] = []
            event_list.append(e)

    logger.debug(f"qa_batch_size: {qa_batch_size}")
    #chunks = (len(doc_info) - 1) // qa_batch_size + 1

    # it doesn't do well with single document batches. If one is going
    # to be produced, increase the batch size by 1 to avoid
    if len(event_list) % qa_batch_size == 1:
        qa_batch_size = qa_batch_size + 1
    chunks = int(np.ceil(float(len(event_list)) / qa_batch_size))

    logger.info("Running event-location linking model...")
    logger.debug(f"Doc info length: {len(event_list)}")
    logger.debug(f"Batch size: {qa_batch_size}")
    logger.debug(f"Chunk number: {chunks}")
    for c in tqdm(range(chunks)):
        start = c*qa_batch_size
        end = (c+1)*qa_batch_size
        if end > len(event_list):
            end = len(event_list) + 1
        logger.debug(f"Batch start: {start}. Batch end: {end}")
        batch = event_list[start:end]
        logger.debug(f"Actual batch length: {len(batch)}")
        if len(batch) == 0:
            continue
        batch_QA_input = []
        for d in batch:
            q = {"question": f"Where did {d['name'].lower()} happen?", # name = CAMEO code
            #q = {"question": f"Which place did {d['name'].lower()} happen?", # name = CAMEO code
                   "context": d['partial_doc']} # doc up to and including event sentence
            batch_QA_input.append(q)
        batch_res = geo.trf(batch_QA_input)
        for n, d in enumerate(batch):
            d['qa_output'] = batch_res[n] 


    for d in event_list:
        d = geo.pick_event_loc(d)

    clean_event_list = []
    for n, d in enumerate(event_list):
        c = {'event_id': d['event_id'],
                 'storyid': d['storyid'],
                 'text': d['text'],
                 'name': d['name'],
                 'sentence_num': d['sentence_num'],
                 'Headline': d['Headline'],
                 'tmp_partial_doc': d['partial_doc'],
                 'tmp_qa_answer': d['qa_output']['answer']}

        if d['event_loc']:
            try:
                country_name = country_dict[d['event_loc']['country_code3']]
            except KeyError:
                country_name = d['event_loc']['country_code3']
            c['extracted_name'] = d['event_loc']['extracted_name']
            c['mordecai_resolved_place'] = d['event_loc']['name']
            c['mordecai_district'] = d['event_loc']['admin2_name']
            c['mordecai_province'] = d['event_loc']['admin1_name']
            c['mordecai_country'] = country_name
            c['mordecai_lat'] = d['event_loc']['lat']
            c['mordecai_lon'] = d['event_loc']['lon']
            c['mordecai_geonameid'] = d['event_loc']['geonameid']
            c['mordecai_event_loc_reason'] = d['event_loc_reason']
        else:
            c['mordecai_event_loc_reason'] = d['event_loc_reason']
        clean_event_list.append(c)

    fn = f"mordecai_geo_results_{input_file}.csv"
    mordecai_df = pd.DataFrame(clean_event_list)
    mordecai_df.to_csv(fn)
    logger.info(f"Successfully wrote events to {fn}")

    diff = time.time() - start_time
    logger.info(f"Total time elapsed (s): {str(round(diff, 2))}")

if __name__ == "__main__":
    typer.run(main)