
from collections import Counter
import logging
import numpy as np
import os
import spacy
import torch
import re
import warnings

from elasticsearch import Elasticsearch
try: 
    from elasticsearch.dsl import Search             # type: ignore
except ImportError:
    # elasticsearch < 8.18.0
    from elasticsearch_dsl import Search
from importlib import resources
try: 
    from importlib.resources.abc import Traversable  # type: ignore[import-untyped]
except ImportError:
    # Python < 3.13
    from importlib.abc import Traversable
from torch.utils.data import DataLoader
import jellyfish
import numpy as np
import numpy.typing as npt

from .elasticsearch import setup_es_client
from .geonames import GeonamesService
from .mordecai_utilities import spacy_doc_setup
from .torch_model import ProductionData, geoparse_model


logger = logging.getLogger(__name__)


spacy_doc_setup()

def load_nlp():
    nlp = spacy.load("en_core_web_trf")
    nlp.add_pipe("token_tensors")
    return nlp

def load_model(model_path, device=None):
    if not device:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = geoparse_model(device=device,
                           bert_size=768,
                           num_feature_codes=54) 
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model



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
    doc_tensor = np.mean(np.vstack([i._.tensor for i in doc]), axis=0)
    # the "loc_ents" are the ones we use for context. NORPs are useful for context,
    # but we don't want to geoparse them. Anecdotally, FACs aren't so useful for context,
    # but we do want to geoparse them.
    loc_ents = [ent for ent in doc.ents if ent.label_ in ['GPE', 'LOC', 'EVENT_LOC', 'NORP']]
    for ent in doc.ents:
        if ent.label_ in ['GPE', 'LOC', 'EVENT_LOC', 'FAC']:
            tensor = np.mean(np.vstack([i._.tensor for i in ent]), axis=0)
            other_locs = [i for e in loc_ents for i in e if i not in ent]
            in_rel = guess_in_rel(ent)
            #print("detected relation: ", ent.text, "-->", in_rel)
            if other_locs:
                locs_tensor = np.mean(np.vstack([i._.tensor for i in other_locs if i not in ent]), axis=0)
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
    conn: Search

    def __init__(self, 
                 model_path: str | Traversable | None=None, 
                 geo_asset_path: str | Traversable | None=None,
                 geonames: GeonamesService | None=None,
                 nlp=None,
                 debug: bool=False,
                 trim=None,
                 check_es: bool=True,
                 hosts: list[str] | None = None,
                 port: int = 9200,
                 device='cpu',
                 use_ssl: bool=False,
                 es_client: Elasticsearch | None=None):
        if device != "cpu":
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.debug = debug
        self.trim = trim
        if not nlp:
            self.nlp = load_nlp()
        else:
            if 'token_tensors' not in nlp.pipe_names:
                try:
                    nlp.add_pipe("token_tensors")
                except Exception as e:
                    # TODO: this is currently catching the error that the pipe already exists,
                    # but it shouldn't catch the error that it doesn't know what
                    # token_tensors is.
                    logger.info(f"Error loading token_tensors pipe: {e}")
                    pass
            self.nlp = nlp
        
        # Handle ES and GeonamesService connection

        if es_client is not None:
            self.conn = Search(using=es_client, index="geonames")
        else:
            es_client = setup_es_client(hosts=hosts, port=port, use_ssl=use_ssl)
            self.conn = Search(using=es_client, index="geonames")
        
        if geonames is not None:
            self.geonames = geonames
        else:
            self.geonames = GeonamesService(es_client=es_client)

        if check_es:
            logger.info("Checking Elasticsearch connection...")
            try:
                assert len(list(self.conn[1])) > 0
                logger.info("Successfully connected to Elasticsearch.")
            except:
                logger.warning("Could not connect to Elasticsearch, but the logic of this code path may be wrong...")
                ConnectionError("Could not locate Elasticsearch. Are you sure it's running?")
        
        
        if not model_path:
            model_path =  resources.files("mordecai3") / "assets/mordecai_2025-08-27.pt"
        self.model = load_model(model_path, device=device)
        if not geo_asset_path:
            geo_asset_path = resources.files("mordecai3") / "assets/"
        self.hierarchy = load_hierarchy(geo_asset_path)
        self.model.to(device)

    def lookup_city(self, entry):
        """
        
        """
        city_id = ""
        city_name = ""
        if entry['feature_code'] == 'PPLX':
            try:
                parent_id = self.hierarchy[entry['geonameid']]
                parent_res = self.geonames.get_entry_by_id(parent_id)
                if parent_res['feature_class'] == "P":
                    city_id = parent_id
                    city_name = parent_res['name']
            except KeyError:
                city_id = entry['name']
                city_name = entry['geonameid']
        elif entry['feature_class'] == 'S':
            try:
                parent_id = self.hierarchy[entry['geonameid']]
                parent_res = self.geonames.get_entry_by_id(parent_id)
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


    def geoparse_doc(self, 
                     text, 
                     debug=False, 
                     trim=True, 
                     known_country=None,
                     max_choices=100):
        """
        Geoparse a document.

        Parameters
        ----------
        text : str or spacy Doc (with ._.tensor attributes)
            The text to geoparse.
        debug : bool
            If True, returns the top 4 results for each geoparsed location, rather than the single best.
            This is useful for debugging or collecting new annotations.
        trim : bool
            If True (default: True), removes some of the keys from the output dictionary that are only used
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

        doc_ex = doc_to_ex_expanded(doc)
        if doc_ex:
            es_data = add_es_data_doc(doc_ex, self.geonames, max_results=max_choices,
                                              known_country=known_country)

            dataset = ProductionData(es_data, max_choices=max_choices)

            data_loader = DataLoader(dataset=dataset, batch_size=64, shuffle=False)
            with torch.no_grad():
                self.model.eval()
                pred_val_list = []
                for input_batch in data_loader:
                    # Move the entire input batch to the model's device
                    input_batch_on_device = {k: v.to(self.model.device) for k, v in input_batch.items()}
                    pred_val_list.append(self.model(input_batch_on_device))
                pred_val = torch.cat(pred_val_list, dim=0)


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
                    # print the next best result:
                    if len(scores) == 1:
                        logger.debug(f"Only one score found: {results[0]}")
                    if len(scores) > 1:
                        second_best_idx = np.argsort(scores)[-2]
                        second_best = results[second_best_idx]
                        logger.debug(f"Second best result: {second_best.get('name', 'N/A')} (score: {second_best.get('score', 'N/A')})")
                    continue
                results = sorted(results, key=lambda k: -k['score'])
                if results and (not debug):
                    logger.debug("Picking top predicted result")
                    best = results[0]
                    best["search_name"] = ent['search_name']
                    best["start_char"] = ent['start_char']
                    best["end_char"] = ent['end_char']
                    ## Add in city info here
                    best['city_id'], best['city_name'] = self.lookup_city(best)
                    best_list.append(best)
                if results and debug:
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

            

def add_es_data(ex, 
                geonames_service: GeonamesService, 
                max_results=50, 
                fuzzy=0, 
                limit_types=False,
                remove_correct=False, 
                known_country=None):
    """
    Run an Elasticsearch/geonames query for a single example and add the results
    to the object.

    Parameters
    ---------
    ex: dict
      output of doc_to_ex_expanded
    conn: elasticsearch connection
    max_results: int
      Maximum results to bring back from ES
    fuzzy: int
      Allow fuzzy results? 0=exact matches. Higher numbers will
      increase the fuzziness of the search. 
    remove_correct: bool
        If True, remove the correct result from the list of results.
        This is useful for training a model to handle "none of the above"
        cases.

    Examples
    --------
    ex = {"search_name": ent.text,
         "tensor": tensor,
         "doc_tensor": doc_tensor,
         "locs_tensor": locs_tensor,
         "sent": ent.sent.text,
         "in_rel": in_rel,    # this comes from the heuristic `guess_in_rel` fuction defined in geoparse.py
         "start_char": ent[0].idx,
         "end_char": ent[-1].idx + len(ent.text)}
    d_es = add_es_data(d)
    # d_es now has a "es_choices" key and a "correct" key that indicates which geonames 
    # entry was the correct one.
    """
    max_results = int(max_results)
    fuzzy = int(fuzzy)
    search_name = ex['search_name']
    # if we detect a parent location using our heuristic (see `guess_in_rel` in geoparse.py),
    # check to see if that's a country or admin1. 
    if 'in_rel' in ex.keys():
        if ex['in_rel']:
            parent_place = geonames_service.get_country_by_name(ex['in_rel'])
            if not parent_place:
                parent_place = geonames_service.get_adm1_country_entry(ex['in_rel'], None)
        else:
            parent_place = None 
    else:
        parent_place = None 
    
    search_res = geonames_service.search_by_name(search_name, max_results, fuzzy, limit_types, known_country)
    choices = res_formatter(search_res, search_name, parent_place)

    # Always try a fuzzy search if no results from previous search, to avoid 
    # having no candidates for the ML model to choose from.
    if not choices:
        search_res = geonames_service.search_by_name(search_name, max_results, fuzzy+1, limit_types, known_country)
        choices = res_formatter(search_res, ex['search_name'], parent_place)

    if remove_correct:
        choices = [c for c in choices if c['geonameid'] != ex['correct_geonamesid']]

    # Always add a final "NULL" choice at the end
    logger.debug("Adding NULL choice")
    null_choice = {'feature_code': 'NULL', 
            'feature_class': 'NULL', 
            'country_code3': 'NULL', 
            'lat': 0, 
            'lon': 0, 
            'name': 'NULL', 
            'admin1_code': 'NULL', 
            'admin1_name': 'NULL', 
            'admin2_code': 'NULL', 
            'admin2_name': 'NULL', 
            'geonameid': 'NULL', 
            'admin1_parent_match': -1, 
            'country_code_parent_match': -1, 
            'alt_name_length': 0, 
            'min_dist': 99.0, 
            'max_dist': 99.0, 
            'avg_dist': 99.0, 
            'ascii_dist': 99.0, 
            'adm1_count': 0.0, 
            'country_count': 0.0}
    choices.append(null_choice)
    ex['es_choices'] = choices

    if remove_correct:
        ex['correct'] = [False for c in choices]
    else:
        if 'correct_geonamesid' in ex.keys():
            ex['correct'] = [c['geonameid'] == ex['correct_geonamesid'] for c in choices]
    return ex



def add_es_data_doc(doc_ex, conn, max_results=50, fuzzy=0, limit_types=False,
                    remove_correct=False, known_country=None):
    doc_es = []
    for ex in doc_ex:
        with warnings.catch_warnings():
            try:
                es = add_es_data(ex, conn, max_results, fuzzy, limit_types, remove_correct, known_country)
                doc_es.append(es)
            except Warning:
                continue
    if not doc_es:
        return []
    admin1_count = make_admin1_counts(doc_es)
    country_count = make_country_counts(doc_es)

    for i in doc_es:
        for e in i['es_choices']:
            e['adm1_count'] = admin1_count[e['admin1_name']]
            e['country_count'] = country_count[e['country_code3']]
    return doc_es


def res_formatter(res, search_name, parent=None):
    """
    Helper function to format the ES/Geonames results into a format for the ML model, including
    edit distance statistics and parent matches.

    Parameters
    ----------
    res: Elasticsearch/Geonames output
    search_name: str
      The original search term from the document
    parent: dict
      Geonames/ES entry for the inferred parent 

    Returns
    -------
    choices: list
      List of formatted Geonames results, including edit distance statistics
    """
    # choices is our eventual output, a list of dicts, each of which is a formatted Geonames result 
    choices = []
    alt_lengths = []
    min_dist = []
    max_dist = []
    avg_dist = []
    ascii_dist = []
    # iterate through the docs returned by ES
    for i in res['hits']['hits']:
        i = i.to_dict()['_source']
        names = [i['name']] + i['alternativenames'] 
        dists = [jellyfish.levenshtein_distance(search_name, j) for j in names]
        lat, lon = i['coordinates'].split(",")
        d = {"feature_code": i['feature_code'],
            "feature_class": i['feature_class'],
            "country_code3": i['country_code3'],
            "lat": float(lat),
            "lon": float(lon),
            "name": i['name'],
            "admin1_code": i['admin1_code'],
            "admin1_name": i['admin1_name'],
            "admin2_code": i['admin2_code'],
            "admin2_name": i['admin2_name'],
            "geonameid": i['geonameid']}
        # if we detect a parent country or ADM1, add the parent match features
        if parent: 
            if parent['admin1_name'] == "":
                d['admin1_parent_match'] = 0
            elif parent['admin1_name'] == i['admin1_name']:
                d['admin1_parent_match'] = 1
            else:
                d['admin1_parent_match'] = -1

            if parent['country_code3'] == "":
                d['country_code_parent_match'] = 0
            elif parent['country_code3'] == i['country_code3']:
                d['country_code_parent_match'] = 1
            else:
                d['country_code_parent_match'] = -1
        else:
            d['admin1_parent_match'] = 0
            d['country_code_parent_match'] = 0

        choices.append(d)
        alt_lengths.append(len(i['alternativenames'])+1)
        min_dist.append(np.min(dists))
        max_dist.append(np.max(dists))
        avg_dist.append(np.mean(dists))
        ascii_dist.append(jellyfish.levenshtein_distance(search_name, i['asciiname']))
    alt_lengths = np.log(alt_lengths)
    min_dist = normalize(min_dist)
    max_dist = normalize(max_dist)
    avg_dist = normalize(avg_dist)
    ascii_dist = normalize(ascii_dist)

    for n, i in enumerate(choices):
        i['alt_name_length'] = alt_lengths[n]
        i['min_dist'] = min_dist[n]
        i['max_dist'] = max_dist[n]
        i['avg_dist'] = avg_dist[n]
        i['ascii_dist'] = ascii_dist[n]
    return choices


def make_admin1_counts(out):
    """
    Get the ADM1s from all candidate results for all locations in a document and return
    the count of each ADM1. This allows us to prefer candidates that share an ADM1 with other
    locations in a document.
    
    This is getting at roughly the same info as previous (slow) approaches that tried to minimize
    the distance between the returned geolocations.

    Parameters
    ---------
    out: list of dicts
      List of place names from the document with candidate geolocations
      from ES/Geonames

    Returns
    -------
    admin1_count: dict
      A dictionary {adm1: count}, where count is the proportion of place names in the
      document that have at least one candidate entry from this adm1.
    """
    admin1s = []

    # for each entity, get the unique ADM1s from the search results 
    for es in out:
        other_adm1 = set([i['admin1_name'] for i in es['es_choices']])
        admin1s.extend(list(other_adm1))
    
    # TODO: handle the "" admins here.
    admin1_count = dict(Counter(admin1s))
    for k, v in admin1_count.items():
        admin1_count[k] = v / len(out)
    return admin1_count

def make_country_counts(out):
    """Take in a document's worth of examples and return the count of countries"""
    all_countries = []
    for es in out:
        countries = set([i['country_code3'] for i in es['es_choices']])
        all_countries.extend(list(countries))
    
    country_count = dict(Counter(all_countries))
    for k, v in country_count.items():
        country_count[k] = v / len(out)
        
    return country_count


def normalize(ll: list[float]) -> npt.NDArray[np.float64]:    
    """Normalize an array to [0, 1]"""
    arr = np.array(ll)
    if len(arr) > 0:
        max_arr = np.max(arr)
        if max_arr == 0:
            max_arr = 0.001
        arr = (arr - np.min(arr)) / max_arr
    return arr

