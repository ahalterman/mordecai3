import pandas as pd 
from collections import Counter

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

from torch.utils.data import Dataset, DataLoader
import elastic_utilities as es_util
from format_geoparsing_data import doc_to_ex_expanded
from torch_bert_placename_compare import ProductionData, embedding_compare
from roberta_qa import setup_qa, add_event_loc

from elasticsearch import Elasticsearch, helpers
from elasticsearch_dsl import Search, Q

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

def setup_es():
    kwargs = dict(
        hosts=['localhost'],
        port=9200,
        use_ssl=False,
    )
    CLIENT = Elasticsearch(**kwargs)
    conn = Search(using=CLIENT, index="geonames")
    return conn

def load_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = embedding_compare(device = device,
                                bert_size = 768,
                                num_feature_codes=54) 
    model.load_state_dict(torch.load("mordecai2.pt"))
    model.eval()
    return model

def load_trf():
    trf = setup_qa()
    return trf

nlp = spacy.load("en_core_web_trf")
nlp.add_pipe("token_tensors")

conn = setup_es()
model = load_model()
trf = load_trf()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

events = pd.read_csv("events.20210301073501.Release507.csv", sep="\t")

event_dict = events.to_dict('records')

short_events = event_dict[0:200]

doc_info = []
text_list = []
for i in short_events:
    d = {"headline": i['Headline'],
        "sentence": i['Event Sentence'],
        "story_id": i['Story ID'],
        "sentence_num": i['Sentence Number'],
        "event_text": i['Event Text']}
    d['text'] =  d['headline'] + ". " + d['sentence'] 
    doc_info.append(d)
    text_list.append(d['text'])

docs = list(tqdm(nlp.pipe(text_list), total=len(text_list)))

for n, i in enumerate(doc_info):
    i['doc'] = docs[n]



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
            print(ent['placename'], i)
        results = sorted(results, key=lambda k: -k['score'])
        results = [i for i in results if i['score'] > 0.01]
        try:
            results = results[0]
        except:
            d['geo'] = None
            return d
        r = {"placename": ent['placename'],
            "lat": results['lat'],
            "admin1_name": results['admin1_name'],
            "admin2_name": results['admin2_name'],
            "countrycode": results['country_code3'],
            "lon": results['lon'],
            "start_char": ent['start_char'],
            "end_char": ent['end_char']}
        resolved_entities.append(r)
    d['geo'] = resolved_entities
    return d

    
resolved = []
for d in tqdm(doc_info):
    dr = resolve_entities(d)
    resolved.append(dr)



