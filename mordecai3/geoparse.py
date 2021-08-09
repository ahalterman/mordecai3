
import jsonlines
from tqdm import tqdm
import re

import torch
import pandas as pd
import spacy
from spacy.language import Language
from spacy.tokens import Token, Doc
from spacy.pipeline import Pipe
import numpy as np
from torch.utils.data import Dataset, DataLoader

import elastic_utilities as es_util
from torch_model import ProductionData, geoparse_model
from roberta_qa import setup_qa, add_event_loc
from utilities import spacy_doc_setup

spacy_doc_setup()

def load_nlp():
    nlp = spacy.load("en_core_web_trf")
    nlp.add_pipe("token_tensors")
    return nlp

def load_model(model_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = geoparse_model(device = device,
                                bert_size = 768,
                                num_feature_codes=54) 
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def load_trf():
    trf = setup_qa()
    return trf


def doc_to_ex_expanded(doc):
    """
    Take in a spaCy doc with a custom ._.tensor attribute on each token and create a list
    of dictionaries with information on each place name entity.

    In the broader pipeline, this is called after nlp() and the results are passed to the 
    Elasticsearch step.

    Parameters
    ---------
    doc: spacy.Doc 
      Needs custom ._.tensor attribute.

    Returns
    -------
    data: list of dicts
    """
    data = []
    doc_tensor = np.mean(np.vstack([i._.tensor for i in doc]), axis=0)
    loc_ents = [ent for ent in doc.ents if ent.label_ in ['GPE', 'LOC', 'EVENT_LOC', 'NORP']]
    for ent in doc.ents:
        if ent.label_ in ['GPE', 'LOC', 'EVENT_LOC', 'FAC']:
            tensor = np.mean(np.vstack([i._.tensor for i in ent]), axis=0)
            other_locs = [i for e in loc_ents for i in e if i not in ent]
            if other_locs:
                locs_tensor = np.mean(np.vstack([i._.tensor for i in other_locs if i not in ent]), axis=0)
            else:
                locs_tensor = np.zeros(len(tensor))
            d = {"placename": ent.text,
                 "tensor": tensor,
                 "doc_tensor": doc_tensor,
                 "locs_tensor": locs_tensor,
                 "sent": ent.sent.text,
                "start_char": ent[0].idx,
                "end_char": ent[-1].idx + len(ent.text)}
            data.append(d)
    return data

class Geoparser:
    def __init__(self, 
                 model_path="assets/mordecai2.pt", 
                 nlp=None):
        if not nlp:
            self.nlp = load_nlp()
        else:
            self.nlp = nlp
        self.conn = es_util.make_conn()
        self.model = load_model(model_path)
        self.trf = load_trf()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

    def geoparse_doc(self, text, icews_cat=None):
        """
        text = "Speaking from Berlin, President Obama expressed his hope for a peaceful resolution to the fighting in Homs and Aleppo."
        icews_cat = "Make statement"
        """
        doc = self.nlp(text)

        print("Doc ents: ", doc.ents)
        doc_ex = doc_to_ex_expanded(doc)
        if doc_ex:
            es_data = es_util.add_es_data_doc(doc_ex, self.conn, max_results=500)

            dataset = ProductionData(es_data, max_choices=500)
            data_loader = DataLoader(dataset=dataset, batch_size=64, shuffle=False)
            with torch.no_grad():
                self.model.eval()
                for input in data_loader:
                    pred_val = self.model(input)

        if icews_cat:
            question = f"Where did {icews_cat.lower()} happen?"
            QA_input = {
                    'question': question,
                    'context':text
                }
            res = self.trf(QA_input)
            event_doc = add_event_loc(doc, res)
        else:
            event_doc = doc

        #labels = ["GPE", "LOC", "FAC", "EVENT_LOC"]

        best_list = []
        output = {"doc_text": doc.text,
                 "event_location": '',
                 "geolocated_ents": []}
        if len(doc_ex) == 0:
            return output
        elif len(es_data) == 0:
            return output
        else:
            for (ent, pred) in zip(es_data, pred_val):
                #print("**Place name**: {}".format(ent['placename']))
                #print(len(ent['es_choices']))
                for n, score in enumerate(pred):
                    if n < len(ent['es_choices']):
                        ent['es_choices'][n]['score'] = score.item() # torch tensor --> float
                results = [e for e in ent['es_choices'] if 'score' in e.keys()]
                if not results:
                    print("(no results)")
                best = {"placename": ent['placename'],
                        "start_char": ent['start_char'],
                        "end_char": ent['end_char']}

                if results:
                    results = sorted(results, key=lambda k: -k['score'])
                    results = [i for i in results if i['score'] > 0.01]
                    best = results[0]
                    best["placename"] = ent['placename']
                    best["start_char"] = ent['start_char']
                    best["end_char"] = ent['end_char']
                best_list.append(best)
        output = {"doc_text": doc.text,
                 "event_location_raw": ''.join([i.text_with_ws for i in event_doc.ents if i.label_ == "EVENT_LOC"]),
                 "geolocated_ents": best_list}
        return output

    def pick_event_loc(self, d):
        """
        Heuristic rules for picking the best event location after the QA geolocation step.
        Provides explanations for why each was selected.

        This takes an event+story dictionary as input, with the important keys being:
        'qa_output': the output from the QA model.
           Example: {'score': 0.6123082637786865, 'start': 188, 'end':
                     214, 'answer': 'Bethel Baptist High School'}
        'geo': a list of geolocated place names from the document.
        'partial_doc': the text leading up through the event sentence

        To this dictionary, it adds two keys:
        - event_loc: the entry from the `geo` list that is the best event location
        - event_loc_reason: a short string describing why that particular location
          was selected as the event location.
        """
        if not d['geo']:
            d['event_loc'] = None
            d['event_loc_reason'] = "No locations found"
            return d
        loc_start = d['qa_output']['start']
        loc_end = d['qa_output']['end']
        geo = [i for i in d['geo'] if i]
        loc_ents = [i for i in geo if i['start_char'] in range(loc_start, loc_end+1)]
        # if there are no loc ents...
        if not loc_ents:
            if len(geo) == 0:
                d['event_loc'] = None
                d['event_loc_reason'] = "No locations found"
            elif len(geo) > 1:
                countries = list(set([i['country_code3'] for i in geo]))
                adm1s = list(set(['_'.join([i['country_code3'], i['admin1_name']]) for i in geo]))
                soft_locs = [i for i in geo if i['start_char'] in range(loc_start, loc_end+6)]
                if soft_locs:
                    soft_loc = soft_locs[0]
                else:
                    soft_loc = None

                if soft_loc and re.search(",|in", d['partial_doc'][loc_end:soft_loc['start_char']]):
                        d['event_loc'] = soft_loc
                        d['event_loc_reason'] = "No event location identified, picking the following comma sep'ed location"
                elif len(set([i['placename'] for i in geo])) == 1:
                    # note: cities and states could have the same place name
                    d['event_loc'] = geo[0]
                    d['event_loc_reason'] = "No event location identified, but only one (unique) location in text"
                ## TODO: see if the placenames all fit in a hierarchy (P --> ADM2 --> ADM1) etc.
                elif len(adm1s) == 1:
                    iso3c, adm1 = adm1s[0].split("_")
                    d['event_loc'] = es_util.get_adm1_country_entry(adm1, iso3c, self.conn)
                    d['event_loc_reason'] = "No event location identified, using common ADM1"
                elif len(countries) == 1:
                    ## TODO: This needs to get the actual country entry
                    common_country = geo[0]['country_code3']
                    d['event_loc'] = es_util.get_country_entry(common_country, self.conn)
                    d['event_loc_reason'] = "No event location identified, using common country"
                #elif:
                #   # see if there's a nearby loc in the sentence here
                else:
                    d['event_loc'] = None
                    d['event_loc_reason'] = "Multiple locations, but none identified as the event location"
            else:
                d['event_loc'] = geo[0]
                d['event_loc_reason'] = "No event location identified, but only one location in text"
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
                    d['event_loc'] = loc_ents[0]
                    d['event_loc_reason'] = "Multiple locations within event location. Picking the first location."
        if 'event_loc' not in d.keys():
            d['event_loc'] = None
            d['event_loc_reason'] = "CHECK THIS! Missed the conditionals."
        return d

            
if __name__ == "__main__":
    geo = Geoparser()
    text = "Speaking from Berlin, President Obama expressed his hope for a peaceful resolution to the fighting in Homs and Aleppo."
    icews_cat = "fight"
    out = geo.geoparse_doc(text, icews_cat) 
    print(out)