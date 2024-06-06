import logging
import os
import re

import numpy as np
import pkg_resources
import spacy
import torch
from torch.utils.data import DataLoader

from mordecai3.elastic_utilities import (
    add_es_data_doc,
    get_adm1_country_entry,
    get_country_entry,
    get_entry_by_id,
    make_conn,
)
from mordecai3.mordecai_utilities import spacy_doc_setup
from mordecai3.roberta_qa import add_event_loc, setup_qa
from mordecai3.torch_model import ProductionData, geoparse_model

import logging
logger = logging.getLogger()
handler = logging.StreamHandler() 
formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

spacy_doc_setup()

def load_nlp():
    nlp = spacy.load("en_core_web_trf")
    nlp.add_pipe("token_tensors")
    return nlp

def load_model(model_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = geoparse_model(device=device,
                           bert_size=768,
                           num_feature_codes=54) 
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def load_trf():
    trf = setup_qa()
    return trf


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
            logger.warning("Error getting 'next_ent'.")
            return ""
        return next_ent.text
    else:
        return ""


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
    doc_tensor = np.mean(np.vstack([i._.tensor.data for i in doc]), axis=0)
    # the "loc_ents" are the ones we use for context. NORPs are useful for context,
    # but we don't want to geoparse them. Anecdotally, FACs aren't so useful for context,
    # but we do want to geoparse them.
    loc_ents = [ent for ent in doc.ents if ent.label_ in ['GPE', 'LOC', 'EVENT_LOC', 'NORP']]
    for ent in doc.ents:
        if ent.label_ in ['GPE', 'LOC', 'EVENT_LOC', 'FAC']:
            tensor = np.mean(np.vstack([i._.tensor.data for i in ent]), axis=0)
            other_locs = [i for e in loc_ents for i in e if i not in ent]
            in_rel = guess_in_rel(ent)
            #print("detected relation: ", ent.text, "-->", in_rel)
            if other_locs:
                locs_tensor = np.mean(np.vstack([i._.tensor.data for i in other_locs if i not in ent]), axis=0)
            else:
                locs_tensor = np.zeros(len(tensor))
            d = {"search_name": ent.text,
                 "tensor": tensor,
                 "doc_tensor": doc_tensor,
                 "locs_tensor": locs_tensor,
                 "sent": ent.sent.text,
                 "in_rel": in_rel,
                "start_char": ent[0].idx,
                "end_char": ent[-1].idx + len(ent[-1].text)}
            data.append(d)
    return data

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
    def __init__(self, 
                 model_path=None, 
                 geo_asset_path=None,
                 nlp=None,
                 event_geoparse=False,
                 debug=False,
                 trim=None,
                 check_es=True):
        self.debug = debug
        self.trim = trim
        if not nlp:
            self.nlp = load_nlp()
        else:
            try:
                nlp.add_pipe("token_tensors")
            except Exception as e:
                # TODO: this is currently catching the error that the pipe already exists,
                # but it shouldn't catch the error that it doesn't know what
                # token_tensors is.
                logger.info(f"Error loading token_tensors pipe: {e}")
                pass
            self.nlp = nlp
        self.conn = make_conn()
        if check_es:
            try:
                assert len(list(geo.conn[1])) > 0
                logger.info("Successfully connected to Elasticsearch.")
            except:
                ConnectionError("Could not locate Elasticsearch. Are you sure it's running?")
        if not model_path:
            model_path = pkg_resources.resource_filename("mordecai3", "assets/mordecai_2024-06-04.pt")
        self.model = load_model(model_path)
        if not geo_asset_path:
            geo_asset_path = pkg_resources.resource_filename("mordecai3", "assets/")
        self.hierarchy = load_hierarchy(geo_asset_path)
        self.event_geoparse = event_geoparse
        if event_geoparse:
            self.trf = load_trf()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

    def lookup_city(self, entry):
        """
        
        """
        city_id = ""
        city_name = ""
        if entry['feature_code'] == 'PPLX':
            try:
                parent_id = self.hierarchy[entry['geonameid']]
                parent_res = get_entry_by_id(parent_id, self.conn)
                if parent_res['feature_class'] == "P":
                    city_id = parent_id
                    city_name = parent_res['name']
            except KeyError:
                city_id = entry['name']
                city_name = entry['geonameid']
        elif entry['feature_class'] == 'S':
            try:
                parent_id = self.hierarchy[entry['geonameid']]
                parent_res = get_entry_by_id(parent_id, self.conn)
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
                elif len(set([i['search_name'] for i in geo])) == 1:
                    # note: cities and states could have the same place name
                    d['event_loc'] = geo[0]
                    d['event_loc_reason'] = "No event location identified, but only one (unique) location in text"
                ## TODO: see if the placenames all fit in a hierarchy (P --> ADM2 --> ADM1) etc.
                elif len(adm1s) == 1:
                    iso3c, adm1 = adm1s[0].split("_")
                    d['event_loc'] = get_adm1_country_entry(adm1, iso3c, self.conn)
                    d['event_loc_reason'] = "No event location identified, using common ADM1"
                elif len(countries) == 1:
                    ## TODO: This needs to get the actual country entry
                    common_country = geo[0]['country_code3']
                    d['event_loc'] = get_country_entry(common_country, self.conn)
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


    def geoparse_doc(self, 
                     text, 
                     plover_cat=None, 
                     debug=False, 
                     trim=True, 
                     known_country=None,
                     max_choices=50):
        """
        Geoparse a document.

        Parameters
        ----------
        text : str or spacy Doc (with ._.tensor attributes)
            The text to geoparse.
        plover_cat : str
            The PLOVER category of the event you'd like to geolocated. If provided, the event geoparsing
            will identify the place name in the document that is most likely to be the event location.
            If not provided, the event geoparsing will be skipped.
        debug : bool
            If True, returns the top 4 results for each geoparsed location, rather than the single best.
            This is useful for debugging or collecting new annotations.
        trim : bool
            If True, removes some of the keys from the output dictionary that are only used
            internally for selecting the best geoparsed location. Including these keys is
            useful for debugging.
        known_country : str
            If provided, the geoparser will only consider locations in the given country.

        Returns
        -------
        output : dict
            Includes the following keys:
            - "doc_text": a string of the input text
            - "event_location_raw": str, the place name of the 'event location' (if provided)
            - "geolocated_ents": list of dicts, each dict is a geoparsed location

        Example
        -------
        >>> text = "The earthquake struck in the city of Christchurch, New Zealand."
        >>> geoparser.geoparse_doc(text)
        """
        if type(text) is str:   
            doc = self.nlp(text)
        elif type(text) is spacy.tokens.doc.Doc:
            doc = text
        else:
            raise ValueError("Text must be either of type 'str' or 'spacy.tokens.doc.Doc'.")

        if plover_cat and not self.event_geoparse:
            raise Warning("A PLOVER category was provided but event geoparsing is disabled. Skipping event geolocation!")

        doc_ex = doc_to_ex_expanded(doc)
        if doc_ex:
            es_data = add_es_data_doc(doc_ex, self.conn, max_results=100,
                                              known_country=known_country)

            dataset = ProductionData(es_data, max_choices=100)

            data_loader = DataLoader(dataset=dataset, batch_size=64, shuffle=False)
            with torch.no_grad():
                self.model.eval()
                for input in data_loader:
                    pred_val = self.model(input)

        if plover_cat and self.event_geoparse:
            question = f"Where did {plover_cat.lower()} happen?"
            QA_input = {
                    'question': question,
                    'context':text
                }
            res = self.trf(QA_input)
            event_doc = add_event_loc(doc, res)
        else:
            event_doc = doc

        best_list = []
        output = {"doc_text": doc.text,
                 "event_location": '',
                 "geolocated_ents": []}
        if len(doc_ex) == 0:
            return output
        elif len(es_data) == 0:
            return output
        else:
            # Iterate over all the entities in the document
            for (ent, pred) in zip(es_data, pred_val):
                logger.debug("**Place name**: {}".format(ent['search_name']))
                # if the last one is the argmax, then the model thinks that no answer is correct
                # so return blank
                if pred[-1] == pred.max():
                    logger.debug("Model predicts no answer")
                    best = {"search_name": ent['search_name'],
                        "start_char": ent['start_char'],
                        "end_char": ent['end_char']}
                    best_list.append(best)
                    continue

                for n, score in enumerate(pred):
                    if n < len(ent['es_choices']):
                        ent['es_choices'][n]['score'] = score.item() # torch tensor --> float
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
                if not results:
                    logger.debug("(no results)")
                best = {"search_name": ent['search_name'],
                        "start_char": ent['start_char'],
                        "end_char": ent['end_char']}
                scores = np.array([r['score'] for r in results])
                if len(scores) == 0:
                    logger.debug("No scores found.")
                    continue
                if np.argmax(scores) == len(scores) - 1:
                    logger.debug("Picking final ''null'' result.")
                    continue
                results = sorted(results, key=lambda k: -k['score'])
                if results and debug==False:
                    logger.debug("Picking top predicted result")
                    best = results[0]
                    best["search_name"] = ent['search_name']
                    best["start_char"] = ent['start_char']
                    best["end_char"] = ent['end_char']
                    ## Add in city info here
                    best['city_id'], best['city_name'] = self.lookup_city(best)
                    best_list.append(best)
                if results and debug==True:
                    logger.debug("Returning top 4 predicted results for each location")
                    best = results[0:4]
                    for b in best:
                        b["search_name"] = ent['search_name']
                        b["start_char"] = ent['start_char']
                        b["end_char"] = ent['end_char']
                        b['city_id'], b['city_name'] = self.lookup_city(b)
                        best_list.append(best)

        if (self.trim or trim) and best_list:
            trim_keys = ['admin1_parent_match', 'country_code_parent_match', 'alt_name_length',
                        'min_dist', 'max_dist', 'avg_dist', 'ascii_dist', 'adm1_count', 'country_count']
            for i in best_list:
                i = [i.pop(key) for key in trim_keys if key in i.keys()]
            output = {"doc_text": doc.text,
                 "event_location_raw": ''.join([i.text_with_ws for i in event_doc.ents if i.label_ == "EVENT_LOC"]).strip(),
                 "geolocated_ents": best_list} 
        else:
            output = {"doc_text": doc.text,
                 "event_location_raw": ''.join([i.text_with_ws for i in event_doc.ents if i.label_ == "EVENT_LOC"]).strip(),
                 "geolocated_ents": best_list}
        return output

            
if __name__ == "__main__":
    geo = Geoparser("/home/andy/projects/mordecai3/mordecai_2024-06-04.pt",
                    event_geoparse=False, trim=False)
    text = "Speaking from Berlin, President Obama expressed his hope for a peaceful resolution to the fighting in Homs and Aleppo."
    text = "Speaking from the city of Xinwonsnos, President Obama expressed his hope for a peaceful resolution to the fighting in Janskan and Alanenesla."
    geo.geoparse_doc(text)
    plover_cat = "fight"
    out = geo.geoparse_doc(text, plover_cat) 
    print(out)

    geo = Geoparser("mordecai_new.pt", event_geoparse=False)
    doc = geo.nlp("Speaking from Berlin, President Obama expressed his hope for a peaceful resolution to the fighting in Homs and Aleppo.")
    out = geo.geoparse_doc(doc, trim=True)
