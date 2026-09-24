
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

from tqdm import tqdm

from .elasticsearch import (
    setup_es_client,
    es_is_accepting_connection,
    es_has_geonames_index,
)
from .exceptions import (
    SpacyModelError,
    ElasticsearchConnectionError,
    GeonamesIndexError,
)
from .candidate_features import (
    ALL_KEYS as EXTRA_FEATURE_KEYS,
    add_document_features,
    add_entity_features,
    fill_null_features,
    mention_admin_cue,
)
from .geonames import GeonamesService, hit_sources
from .mordecai_utilities import spacy_doc_setup
from .torch_model import ProductionData, expand_feature_blocks, geoparse_model


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

def load_model(model_path, device=None, n_extra_features=None, **model_kwargs):
    """Rebuild a geoparse_model from a saved state dict and load the weights.

    Parameters
    ----------
    model_path : path-like
    device : torch.device or None
    n_extra_features : int or None
        Number of enrichment feature columns the checkpoint was trained with.
        None (the default) reads it off the checkpoint's mix_linear layer, which
        is 13 wide (4 cosine similarities + 9 gazetteer features) plus one column
        per enrichment feature. Pass a number to assert what you expect.
    **model_kwargs
        Passed to geoparse_model. Behavioral training flags that leave no trace
        in the layer shapes -- ``modern_mlp``, ``mask_padding``, ``return_logits``
        -- must be repeated here if the checkpoint was trained with them.
    """
    if not device:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    state = torch.load(model_path, map_location=device)
    # Read the layer sizes off the checkpoint instead of hardcoding them. The
    # shipped model happens to use the defaults, but tools/train.py exposes
    # --mix-dim, --country-size and --code-size, and a model trained with any
    # of those changed used to fail here with a shape mismatch.
    ckpt_extra = state['mix_linear.weight'].shape[1] - 13
    if n_extra_features is None:
        n_extra_features = ckpt_extra
    elif n_extra_features != ckpt_extra:
        raise ValueError(
            f"checkpoint {model_path} was trained with {ckpt_extra} extra "
            f"gazetteer features, but {n_extra_features} were requested. The "
            f"feature_blocks passed to Geoparser must match the ones the model "
            f"was trained with.")
    model = geoparse_model(device=device,
                           bert_size=state['text_to_country.weight'].shape[1],
                           num_feature_codes=state['code_emb.weight'].shape[0],
                           country_size=state['text_to_country.weight'].shape[0],
                           code_size=state['code_emb.weight'].shape[1],
                           mix_dim=state['mix_linear.weight'].shape[0],
                           n_extra_features=n_extra_features,
                           **model_kwargs)
    model.load_state_dict(state)
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
            # next_ent is the last token in the doc; nothing follows it to check.
            logger.debug("No token after next_ent; treating as no \"in\" relation.")
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
                 device=None,
                 use_ssl: bool=False,
                 es_client: Elasticsearch | None=None,
                 feature_blocks: str | list[str] | None=None,
                 oov_bucket_fix: bool=False,
                 model_options: dict | None=None):
        """
        feature_blocks : str, list of str, or None
            Enrichment feature blocks the loaded checkpoint was trained with,
            e.g. "prom,name,cue,sib,geo,shape" (see
            torch_model.FEATURE_BLOCKS). None (the default) is the legacy
            behavior: the extra features are neither computed nor fed to the
            model, so pre-enrichment checkpoints keep working unchanged.
            When set, the lookup path computes those features for every
            candidate and the model is built with the matching input width,
            which is checked against the checkpoint.
        oov_bucket_fix : bool
            Must match tools/train.py's --oov-bucket-fix for the checkpoint:
            it decides whether an out-of-vocabulary feature code shares the
            "NULL" embedding, and what country the reserved last row gets. It
            leaves no trace in the layer shapes, so it cannot be auto-detected.
        model_options : dict or None
            Extra keyword arguments for geoparse_model, for training flags that
            leave no trace in the checkpoint's layer shapes: modern_mlp,
            mask_padding, return_logits.
        """
        # device=None (the default) auto-detects CUDA. Pass device='cpu' to force CPU.
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

        if es_client is None:
            if geonames is not None:
                # Reuse the client the caller's GeonamesService already holds, so we
                # don't open a second connection (and so check_es validates the same
                # client that lookups will actually go through).
                es_client = geonames.conn
            else:
                es_client = setup_es_client(hosts=hosts, port=port, use_ssl=use_ssl)
        self.conn = Search(using=es_client, index="geonames")

        if geonames is not None:
            self.geonames = geonames
        else:
            self.geonames = GeonamesService(es_client=es_client)

        if check_es:
            logger.info("Checking Elasticsearch connection...")
            if not es_is_accepting_connection(es_client):
                raise ElasticsearchConnectionError()
            if not es_has_geonames_index(es_client):
                raise GeonamesIndexError()
            logger.info("Successfully connected to Elasticsearch.")

        
        # The enrichment features are opt-in: computing them costs a little CPU
        # per candidate, and a checkpoint that wasn't trained on them can't use
        # them anyway.
        self.feature_blocks = feature_blocks
        self.extra_feature_keys = expand_feature_blocks(feature_blocks)
        self.oov_bucket_fix = oov_bucket_fix

        if not model_path:
            model_path =  resources.files("mordecai3") / "assets/mordecai_2025-08-27.pt"
        self.model = load_model(model_path, device=device,
                                n_extra_features=len(self.extra_feature_keys),
                                **(model_options or {}))
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


    def _resolve_results(self, es_data, pred_val, debug=False):
        """Select the best geonames candidates based on model predictions.

        Parameters
        ----------
        es_data : list of dicts
            ES-enriched entity data for a single document.
        pred_val : torch.Tensor
            Model predictions, shape (num_entities, max_choices).
        debug : bool
            If True, return the top 4 candidates per entity instead of just the best.

        Returns
        -------
        best_list : list of dicts
        """
        best_list = []
        for (ent, pred) in zip(es_data, pred_val):
            logger.debug("**Place name**: {}".format(ent['search_name']))
            # if the last one is the argmax, then the model thinks that no answer is
            # correct, so return blank
            if pred[-1] == pred.max():
                logger.debug("Model predicts no answer")
                best = {"search_name": ent['search_name'],
                    "start_char": ent['start_char'],
                    "end_char": ent['end_char']}
                best_list.append(best)
                continue

            for n, score in enumerate(pred):
                if n < len(ent['es_choices']):
                    ent['es_choices'][n]['score'] = score.item()  # torch tensor --> float
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
                for b in results[0:4]:
                    b["search_name"] = ent['search_name']
                    b["start_char"] = ent['start_char']
                    b["end_char"] = ent['end_char']
                    b['city_id'], b['city_name'] = self.lookup_city(b)
                    best_list.append(b)
        return best_list

    @staticmethod
    def _trim_results(best_list):
        """Remove the internal-only keys that are used to pick the best result."""
        trim_keys = ['admin1_parent_match', 'country_code_parent_match', 'alt_name_length',
                    'min_dist', 'max_dist', 'avg_dist', 'ascii_dist', 'adm1_count',
                    'country_count'] + EXTRA_FEATURE_KEYS
        for entry in best_list:
            for key in trim_keys:
                entry.pop(key, None)

    def _geoparse_docs(self, docs, max_choices=100, known_country=None,
                       trim=True, debug=False, es_workers=4):
        """Core geoparsing pipeline for a list of spaCy docs.

        Handles entity extraction, ES lookups (threaded across all documents),
        cross-document model batching, and result resolution. This is the shared
        implementation behind both geoparse_doc() and geoparse_batch().

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
            Return the top 4 candidates per entity.
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

        # 2. ES lookups across all documents via a shared thread pool
        all_es_data = add_es_data_batch(
            all_doc_ex, self.geonames, max_results=max_choices,
            known_country=known_country, es_workers=es_workers,
            extra_features=bool(self.extra_feature_keys))

        # 3. Cross-document model batching: pool all entities into one inference pass
        pooled_es_data = []
        entity_counts = []
        for es_data in all_es_data:
            entity_counts.append(len(es_data))
            pooled_es_data.extend(es_data)

        all_preds = None
        if pooled_es_data:
            dataset = ProductionData(pooled_es_data, max_choices=max_choices,
                                     oov_bucket_fix=self.oov_bucket_fix,
                                     feature_blocks=self.feature_blocks)
            data_loader = DataLoader(dataset=dataset, batch_size=64, shuffle=False)
            with torch.no_grad():
                self.model.eval()
                pred_val_list = []
                for input_batch in data_loader:
                    # Move the entire input batch to the model's device
                    input_batch_on_device = {k: v.to(self.model.device) for k, v in input_batch.items()}
                    pred_val_list.append(self.model(input_batch_on_device))
                all_preds = torch.cat(pred_val_list, dim=0)

        # 4. Split the predictions back out by document and resolve results
        results = []
        pred_offset = 0
        for doc, es_data, n_ents in zip(docs, all_es_data, entity_counts):
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
                     debug=False,
                     trim=True,
                     known_country=None,
                     max_choices=100):
        """
        Geoparse a single document.

        This is a convenience wrapper around the same pipeline that backs
        geoparse_batch(). For multiple documents, use geoparse_batch() instead:
        it batches the spaCy and model forward passes and shares ES lookups.

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
            - "event_location_raw": str, always empty. Retained for backwards
              compatibility; event geolocation was removed in favor of the
              standalone event geolocation models.
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

        return self._geoparse_docs(
            [doc], max_choices=max_choices, known_country=known_country,
            trim=trim, debug=debug)[0]

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
            Thread pool size for parallel ES lookups. Default: 4.
        max_choices : int
            Maximum ES candidates per entity. Default: 100.
        known_country : str or None
            Restrict results to a single country (ISO 3166-1 alpha-3).
        trim : bool
            Remove internal keys from output. Default: True.
        debug : bool
            Return the top 4 candidates per entity. Default: False.
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
        self.geonames.clear_cache()  # fresh cache per geoparse_batch() run
        progress = tqdm(total=len(texts), desc="Geoparsing",
                        disable=not show_progress)

        for chunk_start in range(0, len(texts), chunk_size):
            chunk_texts = texts[chunk_start:chunk_start + chunk_size]

            # Layer 1: spaCy batching
            docs = []
            for doc in self.nlp.pipe(chunk_texts, batch_size=batch_size):
                docs.append(doc)
                progress.update(1)

            # Layers 2-3: ES lookups, model inference, result resolution
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


def add_es_data(ex, 
                geonames_service: GeonamesService, 
                max_results=50, 
                fuzzy=0, 
                limit_types=False,
                remove_correct=False,
                known_country=None,
                extra_features=False):
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
    extra_features: bool
        If True, add the enrichment features (see mordecai3.candidate_features)
        that depend on the mention and its own candidate set. The document-level
        features need every entity in the document, so they are only added by
        add_es_data_batch/add_es_data_doc.

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

    cache_key = _es_cache_key(ex, max_results, fuzzy, limit_types, known_country,
                              extra_features)
    cache = geonames_service._es_cache
    if cache_key in cache:
        # Deep-copy because downstream code mutates choices (adm1_count, country_count)
        logger.debug(f"ES cache hit for '{search_name}'")
        return _finish_es_example(ex, _copy_choices(cache[cache_key]), remove_correct)

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
    choices = res_formatter(search_res, search_name, parent_place, extra_features)

    # Always try a fuzzy search if no results from previous search, to avoid
    # having no candidates for the ML model to choose from.
    if not choices:
        search_res = geonames_service.search_by_name(search_name, max_results, fuzzy+1, limit_types, known_country)
        choices = res_formatter(search_res, ex['search_name'], parent_place, extra_features)

    logger.debug("Adding NULL choice")
    choices.append(_null_choice(search_name if extra_features else None))

    # Cache a copy (downstream code mutates dicts via adm1_count/country_count)
    cache[cache_key] = _copy_choices(choices)
    logger.debug(f"ES cache miss for '{search_name}', cached {len(choices)} choices")

    return _finish_es_example(ex, choices, remove_correct)


def _es_cache_key(ex, max_results, fuzzy, limit_types, known_country,
                  extra_features=False):
    """The inputs that determine an entity's ES candidate list.

    Repeated place names are extremely common within and across documents, so
    this is what lets both the cache and the batched path collapse duplicate
    work. The value it keys is the candidate list *before* `remove_correct` is
    applied, so it stays reusable either way.

    `extra_features` is part of the key because the entity-level enrichment
    features are computed once and cached with the candidate list: two callers
    that disagree about whether they want them must not share an entry.
    """
    return (ex['search_name'], ex.get('in_rel', '') or '', known_country or '',
            int(max_results), int(fuzzy), limit_types, bool(extra_features))


def _copy_choices(choices):
    """Copy a candidate list so callers can mutate it independently.

    Candidate dicts are flat -- every value is a str/int/float -- so a per-dict
    shallow copy is equivalent to a deepcopy here and much cheaper. This is on
    the hot path: every entity gets its own copy of ~100 candidates, which was
    the single largest cost in the lookup stage once queries were batched.
    """
    return [dict(c) for c in choices]


def _null_choice(search_name=None):
    """A fresh "none of the above" candidate, always appended last.

    Pass `search_name` to also give it the enrichment features, at the neutral
    values that keep the per-entity feature arrays rectangular without making
    "no answer" look well-supported.
    """
    choice = {'feature_code': 'NULL',
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
    if search_name is not None:
        fill_null_features(choice, mention_admin_cue(search_name))
    return choice


def _finish_es_example(ex, choices, remove_correct):
    """Attach candidate choices to an example and mark which one is correct.

    Split out of add_es_data so the cache-hit and cache-miss paths stay identical.
    """
    if remove_correct:
        choices = [c for c in choices if c['geonameid'] != ex['correct_geonamesid']]

    ex['es_choices'] = choices

    if remove_correct:
        ex['correct'] = [False for c in choices]
    else:
        if 'correct_geonamesid' in ex.keys():
            ex['correct'] = [c['geonameid'] == ex['correct_geonamesid'] for c in choices]
    return ex


def _add_cross_entity_counts(doc_es):
    """Add the within-document admin1/country co-occurrence counts to each candidate.

    These are features for the model, so they can only be computed once every entity
    in the document has its ES results back.
    """
    admin1_count = make_admin1_counts(doc_es)
    country_count = make_country_counts(doc_es)
    for i in doc_es:
        for e in i['es_choices']:
            e['adm1_count'] = admin1_count[e['admin1_name']]
            e['country_count'] = country_count[e['country_code3']]


def _resolve_parents_batch(in_rel_names, geonames_service):
    """Resolve "in" relations to a parent place, batching the lookups.

    Mirrors the sequential logic in add_es_data: try country first, fall back to
    ADM1 with no country filter. Returns {in_rel value: parent entry or None}.
    """
    names = [n for n in in_rel_names if n]
    if not names:
        return {}
    countries = geonames_service.get_country_by_name_batch(names)
    missing = [n for n, v in countries.items() if not v]
    adm1s = geonames_service.get_adm1_country_entry_batch(missing) if missing else {}
    return {n: (countries[n] or adm1s.get(n)) for n in countries}


def add_es_data_doc(doc_ex, geonames_service: GeonamesService, max_results=50, fuzzy=0,
                    limit_types=False, remove_correct=False, known_country=None,
                    es_workers=None, extra_features=False):
    """Add ES candidates for every entity in one document.

    Thin wrapper over add_es_data_batch so there is a single lookup path.

    es_workers is accepted for backwards compatibility and ignored: lookups are
    now batched into _msearch requests, so concurrency is handled by ES rather
    than by a client-side thread pool.
    """
    if not doc_ex:
        return []
    return add_es_data_batch([doc_ex], geonames_service, max_results, fuzzy,
                             limit_types, remove_correct, known_country,
                             extra_features=extra_features)[0]


def add_es_data_batch(all_doc_ex, geonames_service: GeonamesService, max_results=50,
                      fuzzy=0, limit_types=False, remove_correct=False,
                      known_country=None, es_workers=None, extra_features=False):
    """Look up ES candidates for every entity across many documents.

    All lookups for the batch are bundled into a small number of _msearch
    requests. Each sub-query is executed by ES exactly as it would have been on
    its own, so results are identical to looping over add_es_data -- the win is
    that per-request overhead (HTTP, JSON parsing, client object construction)
    is paid once per chunk rather than once per entity.

    Work is deduplicated by cache key before anything is sent. The sequential
    path got that for free, because the cache filled in as it went; here every
    lookup is planned up front, so duplicates have to be collapsed explicitly.

    Parameters
    ----------
    all_doc_ex : list of list of dicts
        Entity dicts per document (output of doc_to_ex_expanded for each doc).
    geonames_service : GeonamesService
    es_workers : ignored
        Accepted for backwards compatibility. See add_es_data_doc.
    extra_features : bool
        If True, every candidate also gets the enrichment features that the
        `--feature-blocks` models are trained on (see
        mordecai3.candidate_features). Off by default, so a checkpoint that
        predates them sees exactly the candidate dicts it always did.

    Returns
    -------
    list of list of dicts
        ES-enriched entity data, one list per document.
    """
    # Callers reach this from a CLI as often as from library code, so coerce
    # rather than trusting the caller: `fuzzy + 1` in the retry below turns a
    # stringy "0" into a TypeError several hundred lookups deep.
    max_results = int(max_results)
    fuzzy = int(fuzzy)

    tasks = [(doc_idx, ent_idx, ex)
             for doc_idx, doc_ex in enumerate(all_doc_ex)
             for ent_idx, ex in enumerate(doc_ex)]
    if not tasks:
        return [[] for _ in all_doc_ex]

    cache = geonames_service._es_cache

    # 1. Collapse duplicate lookups. Many entities across a batch share a name.
    by_key = {}
    for doc_idx, ent_idx, ex in tasks:
        key = _es_cache_key(ex, max_results, fuzzy, limit_types, known_country,
                            extra_features)
        by_key.setdefault(key, []).append((doc_idx, ent_idx, ex))

    todo = [k for k in by_key if k not in cache]
    logger.debug(f"{len(tasks)} entities -> {len(by_key)} unique lookups, "
                 f"{len(todo)} not cached")

    if todo:
        # 2. Parent lookups. These only feed res_formatter, never the name query
        #    itself, so they are independent of the searches below.
        reps = [by_key[k][0][2] for k in todo]
        parents = _resolve_parents_batch([ex.get('in_rel') for ex in reps],
                                         geonames_service)

        def parent_for(ex):
            return parents.get(ex.get('in_rel') or '')

        # 3. One batched round of name searches.
        specs = [(k[0], max_results, fuzzy, limit_types, known_country) for k in todo]
        responses = geonames_service.search_by_names(specs)

        fresh = {}
        retry = []
        for k, ex, res in zip(todo, reps, responses):
            choices = res_formatter(res, k[0], parent_for(ex), extra_features)
            if choices:
                fresh[k] = choices
            else:
                retry.append((k, ex))

        # 4. Fuzzy retry for names that came back empty, so the model always has
        #    something to choose from. In practice this is ~1% of entities.
        if retry:
            logger.debug(f"fuzzy retry for {len(retry)} names")
            specs = [(k[0], max_results, fuzzy + 1, limit_types, known_country)
                     for k, _ in retry]
            for (k, ex), res in zip(retry, geonames_service.search_by_names(specs)):
                fresh[k] = res_formatter(res, k[0], parent_for(ex), extra_features)

        for k, choices in fresh.items():
            choices.append(_null_choice(k[0] if extra_features else None))
            # Stored pristine; every consumer takes its own copy below, since
            # downstream code mutates adm1_count/country_count in place.
            cache[k] = choices

    # 5. Hand each entity its own copy and reassemble per document.
    results_by_doc = {i: [] for i in range(len(all_doc_ex))}
    for key, members in by_key.items():
        pristine = cache.get(key)
        if pristine is None:
            logger.warning(f"no ES candidates resolved for '{key[0]}', skipping")
            continue
        for doc_idx, ent_idx, ex in members:
            done = _finish_es_example(ex, _copy_choices(pristine), remove_correct)
            results_by_doc[doc_idx].append((ent_idx, done))

    all_doc_es = []
    for doc_idx in range(len(all_doc_ex)):
        entries = sorted(results_by_doc[doc_idx], key=lambda x: x[0])
        doc_es = [r for _, r in entries]
        if doc_es:
            _add_cross_entity_counts(doc_es)
            if extra_features:
                # The sibling and anchor-geometry features read the other
                # mentions in the document, so like the co-occurrence counts
                # above they can only be computed now that every entity in the
                # document has its candidates back.
                add_document_features(doc_es)
        all_doc_es.append(doc_es)

    return all_doc_es


def res_formatter(res, search_name, parent=None, extra_features=False):
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
    extra_features: bool
      Also compute the enrichment features that depend on the mention and its
      own candidate set (see mordecai3.candidate_features). The population and
      name lists they need are already in the ES hits, so this costs no extra
      queries -- and none of those raw fields are kept on the returned dicts.

    Returns
    -------
    choices: list
      List of formatted Geonames results, including edit distance statistics
    """
    # choices is our eventual output, a list of dicts, each of which is a formatted Geonames result
    choices = []
    sources = []
    alt_lengths = []
    min_dist = []
    max_dist = []
    avg_dist = []
    ascii_dist = []
    # iterate through the docs returned by ES. hit_sources handles both the
    # raw dicts from the batched _msearch path and the elasticsearch_dsl
    # objects from a single .execute().
    for i in hit_sources(res):
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
        if extra_features:
            sources.append(i)
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
    if extra_features:
        add_entity_features(search_name, choices, sources)
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

