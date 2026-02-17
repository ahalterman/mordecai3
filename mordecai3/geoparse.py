
import logging
import numpy as np
import os
import spacy
import torch
import re

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

from .elasticsearch import setup_es_client, make_conn, es_is_accepting_connection, es_has_geonames_index
from .exceptions import SpacyModelError, ElasticsearchConnectionError, GeonamesIndexError
from .geonames import (
    add_es_data_batch,
    add_es_data_doc,
    get_adm1_country_entry,
    get_country_entry,
    get_entry_by_id,
)
from tqdm import tqdm
from .mordecai_utilities import spacy_doc_setup
from .roberta_qa import add_event_loc, setup_qa
from .torch_model import ProductionData, geoparse_model


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

spacy_doc_setup()

def load_nlp(use_gpu=False):
    if use_gpu:
        activated = spacy.prefer_gpu()
        if activated:
            logger.info("spaCy: GPU activated")
        else:
            logger.info("spaCy: GPU requested but not available, using CPU")
    try:
        nlp = spacy.load("en_core_web_trf")
    except OSError:
        raise SpacyModelError()
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
                 nlp=None,
                 event_geoparse: bool=False,
                 debug: bool=False,
                 trim=None,
                 check_es: bool=True,
                 hosts: list[str] | None = None,
                 port: int = 9200,
                 device=None,
                 use_ssl: bool=False,
                 es_client: Elasticsearch | None=None):
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)
        use_gpu = (device.type != "cpu")
        logger.info(f"Using device: {device}")
        self.debug = debug
        self.trim = trim
        if not nlp:
            self.nlp = load_nlp(use_gpu=use_gpu)
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
        
        if check_es:
            logger.info("Checking Elasticsearch connection...")
            if not es_is_accepting_connection(es_client):
                raise ElasticsearchConnectionError()
            if not es_has_geonames_index(es_client):
                raise GeonamesIndexError()
            logger.info("Successfully connected to Elasticsearch.")
        
        
        if not model_path:
            model_path =  resources.files("mordecai3") / "assets/mordecai_2025-08-27.pt"
        self.model = load_model(model_path, device=device)
        if not geo_asset_path:
            geo_asset_path = resources.files("mordecai3") / "assets/"
        self.hierarchy = load_hierarchy(geo_asset_path)
        self.event_geoparse = event_geoparse
        if event_geoparse:
            self.trf = load_trf()
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


    def _resolve_results(self, es_data, pred_val, debug=False):
        """Select best geonames candidates based on model predictions.

        Parameters
        ----------
        es_data : list of dicts
            ES-enriched entity data for a single document.
        pred_val : torch.Tensor
            Model predictions, shape (num_entities, max_choices).
        debug : bool
            If True, return top 4 candidates per entity instead of just the best.

        Returns
        -------
        best_list : list of dicts
        """
        best_list = []
        for (ent, pred) in zip(es_data, pred_val):
            logger.debug("**Place name**: {}".format(ent['search_name']))
            # if the last one is the argmax, the model thinks no answer is correct
            if pred[-1] == pred.max():
                logger.debug("Model predicts no answer")
                best = {"search_name": ent['search_name'],
                    "start_char": ent['start_char'],
                    "end_char": ent['end_char']}
                best_list.append(best)
                continue

            for n, score in enumerate(pred):
                if n < len(ent['es_choices']):
                    ent['es_choices'][n]['score'] = score.item()
            results = [e for e in ent['es_choices'] if 'score' in e.keys()]

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
                best['city_id'], best['city_name'] = self.lookup_city(best)
                best_list.append(best)
            if results and debug:
                logger.debug("Returning top 4 predicted results for each location")
                for b in results[0:4]:
                    b["search_name"] = ent['search_name']
                    b["start_char"] = ent['start_char']
                    b["end_char"] = ent['end_char']
                    b['city_id'], b['city_name'] = self.lookup_city(b)
                    best_list.append(b)
        return best_list

    @staticmethod
    def _trim_results(best_list):
        """Remove internal-only keys from result dicts."""
        trim_keys = ['admin1_parent_match', 'country_code_parent_match', 'alt_name_length',
                    'min_dist', 'max_dist', 'avg_dist', 'ascii_dist', 'adm1_count', 'country_count']
        for entry in best_list:
            for key in trim_keys:
                entry.pop(key, None)

    def _geoparse_docs(self, docs, max_choices=100, known_country=None,
                       trim=True, debug=False, es_workers=4):
        """Core geoparsing pipeline for a list of spaCy docs.

        Handles entity extraction, ES lookups (cross-doc threaded), cross-document
        model batching, and result resolution.

        Parameters
        ----------
        docs : list of spacy.tokens.doc.Doc
        max_choices : int
            Maximum ES candidates per entity.
        known_country : str or None
            Restrict results to a single country (ISO 3166-1 alpha-3).
        trim : bool
            Remove internal keys from output.
        debug : bool
            Return top 4 candidates per entity.
        es_workers : int
            Thread pool size for ES lookups.

        Returns
        -------
        list of dicts
            One result dict per input document.
        """
        # 1. Entity extraction
        all_doc_ex = []
        for doc in docs:
            try:
                doc_ex = doc_to_ex_expanded(doc)
            except Exception as e:
                logger.warning(f"Entity extraction failed for document: {e}")
                doc_ex = []
            all_doc_ex.append(doc_ex)

        # 2. ES lookups across all documents via shared thread pool
        all_es_data = add_es_data_batch(
            all_doc_ex, self.conn, max_results=max_choices,
            known_country=known_country, es_workers=es_workers)

        # 3. Cross-document model batching: pool all entities into one inference pass
        pooled_es_data = []
        entity_counts = []
        for es_data in all_es_data:
            entity_counts.append(len(es_data))
            pooled_es_data.extend(es_data)

        all_preds = None
        if pooled_es_data:
            dataset = ProductionData(pooled_es_data, max_choices=max_choices)
            data_loader = DataLoader(dataset=dataset, batch_size=64, shuffle=False)
            with torch.no_grad():
                self.model.eval()
                pred_val_list = []
                for input_batch in data_loader:
                    input_batch_on_device = {k: v.to(self.model.device) for k, v in input_batch.items()}
                    pred_val_list.append(self.model(input_batch_on_device))
                all_preds = torch.cat(pred_val_list, dim=0)

        # 4. Split predictions by document and resolve results
        results = []
        pred_offset = 0
        for doc, doc_ex, es_data, n_ents in zip(docs, all_doc_ex, all_es_data, entity_counts):
            output = {"doc_text": doc.text,
                     "event_location_raw": "",
                     "geolocated_ents": []}

            if n_ents == 0 or not es_data:
                results.append(output)
                continue

            pred_val = all_preds[pred_offset:pred_offset + n_ents]
            pred_offset += n_ents

            best_list = self._resolve_results(es_data, pred_val, debug)
            if (self.trim or trim) and best_list:
                self._trim_results(best_list)
            output["geolocated_ents"] = best_list
            results.append(output)

        return results

    def geoparse_doc(self,
                     text,
                     plover_cat=None,
                     debug=False,
                     trim=True,
                     known_country=None,
                     max_choices=100):
        """
        Geoparse a single document.

        This is a convenience wrapper around geoparse_batch() for processing
        a single text. For multiple documents, use geoparse_batch() directly.

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
        if isinstance(text, spacy.tokens.doc.Doc):
            doc = text
        elif isinstance(text, str):
            doc = self.nlp(text)
        else:
            raise ValueError("Text must be either of type 'str' or 'spacy.tokens.doc.Doc'.")

        result = self._geoparse_docs(
            [doc], max_choices=max_choices, known_country=known_country,
            trim=trim, debug=debug)[0]

        # Event geoparsing (only available in single-doc API)
        if plover_cat and self.event_geoparse:
            question = f"Where did {plover_cat.lower()} happen?"
            QA_input = {'question': question, 'context': doc.text}
            res = self.trf(QA_input)
            event_doc = add_event_loc(doc, res)
            result['event_location_raw'] = ''.join(
                [i.text_with_ws for i in event_doc.ents if i.label_ == "EVENT_LOC"]
            ).strip()
        elif plover_cat and not self.event_geoparse:
            logger.warning("A PLOVER category was provided but event geoparsing is disabled. Skipping event geolocation!")

        return result

    def geoparse_batch(self, texts, batch_size=32, chunk_size=200,
                       es_workers=4, max_choices=100, known_country=None,
                       trim=True, debug=False, show_progress=False):
        """
        Geoparse multiple documents with optimized batching.

        Uses three layers of optimization:
        1. spaCy batching via nlp.pipe() for transformer forward passes
        2. Cross-document threaded ES lookups via a shared thread pool
        3. Cross-document model batching (all entities from a chunk in one inference pass)

        Parameters
        ----------
        texts : list of str
            Documents to geoparse.
        batch_size : int
            Batch size for spaCy's nlp.pipe() transformer inference. Default: 32.
        chunk_size : int
            Number of documents per processing chunk (bounds memory). Default: 200.
        es_workers : int
            Thread pool size for parallel ES lookups. Default: 8.
        max_choices : int
            Maximum ES candidates per entity. Default: 100.
        known_country : str or None
            Restrict results to a single country (ISO 3166-1 alpha-3).
        trim : bool
            Remove internal keys from output. Default: True.
        debug : bool
            Return top 4 candidates per entity. Default: False.
        show_progress : bool
            Show tqdm progress bar. Default: False.

        Returns
        -------
        list of dicts
            One result dict per input document. Each dict has the same structure
            as the output of geoparse_doc(): keys "doc_text", "event_location_raw",
            and "geolocated_ents".
        """
        all_results = []
        progress = tqdm(total=len(texts), desc="Geoparsing",
                        disable=not show_progress)

        for chunk_start in range(0, len(texts), chunk_size):
            chunk_texts = texts[chunk_start:chunk_start + chunk_size]

            # Layer 1: spaCy batching
            docs = []
            for doc in self.nlp.pipe(chunk_texts, batch_size=batch_size):
                docs.append(doc)
                progress.update(1)

            # Layers 2-4: ES lookups, model inference, result resolution
            try:
                chunk_results = self._geoparse_docs(
                    docs, max_choices=max_choices, known_country=known_country,
                    trim=trim, debug=debug, es_workers=es_workers)
            except Exception as e:
                logger.error(f"Chunk processing failed: {e}")
                chunk_results = [
                    {"doc_text": doc.text, "event_location_raw": "",
                     "geolocated_ents": [], "error": str(e)}
                    for doc in docs
                ]

            all_results.extend(chunk_results)

        progress.close()
        return all_results

            
if __name__ == "__main__":
    geo = Geoparser("/home/andy/projects/mordecai3/mordecai_2025-08-27.pt",
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
