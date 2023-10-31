import logging
import re
import warnings
from collections import Counter

import jellyfish
import numpy as np
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Q, Search

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

def make_conn():
    kwargs = dict(
        hosts=['localhost'],
        port=9200,
        use_ssl=False,
    )
    CLIENT = Elasticsearch(**kwargs)
    conn = Search(using=CLIENT, index="geonames")
    return conn

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

def normalize(ll: list) -> np.array:    
    """Normalize an array to [0, 1]"""
    ll = np.array(ll)
    if len(ll) > 0:
        max_ll = np.max(ll)
        if max_ll == 0:
            max_ll = 0.001
        ll = (ll - np.min(ll)) / max_ll
    return ll


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

#search_name = "Marat"
#q = {"multi_match": {"query": search_name,
#                             "fields": ['name', 'asciiname', 'alternativenames'],
#                             "type" : "phrase"}}
#res = conn.query(q).execute()
#parent = get_country_by_name("Morocco", conn)
#res_formatter(res, search_name, parent)

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

def _clean_search_name(search_name):
    """
    Strip out place names that might be preventing the right results
    """
    search_name = re.sub("^the", "", search_name).strip()
    search_name = re.sub("tribal district", "", search_name).strip()
    search_name = re.sub("[Cc]ity", "", search_name).strip()
    search_name = re.sub("[Dd]istrict", "", search_name).strip()
    search_name = re.sub("[Mm]etropolis", "", search_name).strip()
    search_name = re.sub("[Cc]ounty", "", search_name).strip()
    search_name = re.sub("[Rr]egion", "", search_name).strip()
    search_name = re.sub("[Pp]rovince", "", search_name).strip()
    search_name = re.sub("[Tt]territory", "", search_name).strip()
    search_name = re.sub("[Bb]ranch", "", search_name).strip()
    search_name = re.sub("'s$", "", search_name).strip()
    # super hacky!! This one is the most egregious 
    if search_name == "US":
        search_name = "United States"
    return search_name

def add_es_data(ex, conn, max_results=50, fuzzy=0, limit_types=False,
                remove_correct=False, known_country=None):
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
    search_name = _clean_search_name(search_name)
    # if we detect a parent location using our heuristic (see `guess_in_rel` in geoparse.py),
    # check to see if that's a country or admin1. 
    if 'in_rel' in ex.keys():
        if ex['in_rel']:
            parent_place = get_country_by_name(ex['in_rel'], conn)
            if not parent_place:
                parent_place = get_adm1_country_entry(ex['in_rel'], None, conn)
        else:
            parent_place = None 
    else:
        parent_place = None 
    if fuzzy:
        q = {"multi_match": {"query": search_name,
                             "fields": ['name', 'alternativenames', 'asciiname'],
                             "fuzziness" : fuzzy,
                            }}
    else:
        q = {"multi_match": {"query": search_name,
                                 "fields": ['name', 'asciiname', 'alternativenames'],
                                "type" : "phrase"}}

    if limit_types:
        p_filter = Q("term", feature_class="P")
        a_filter = Q("term", feature_class="A")
        combined_filter = p_filter | a_filter
        if known_country:
            country_filter = Q("term", country_code3=known_country)
            combined_filter = combined_filter & country_filter
        res = conn.query(q).filter(combined_filter).sort({"alt_name_length": {'order': "desc"}})[0:max_results].execute()
    elif known_country:
        country_filter = Q("term", country_code3=known_country)
        res = conn.query(q).filter(country_filter).sort({"alt_name_length": {'order': "desc"}})[0:max_results].execute()
    else:
        res = conn.query(q).sort({"alt_name_length": {'order': "desc"}})[0:max_results].execute()
    
    choices = res_formatter(res, search_name, parent_place)

    if not choices:
        # always do a fuzzy step if nothing came up the first time
        q = {"multi_match": {"query": search_name,
                             "fields": ['name', 'alternativenames', 'asciiname'],
                             "fuzziness" : fuzzy+1,
                            }}
        if limit_types:
            p_filter = Q("term", feature_class="P")
            a_filter = Q("term", feature_class="A")
            combined_filter = p_filter | a_filter
            res = conn.query(q).filter(combined_filter).sort({"alt_name_length": {'order': "desc"}})[0:max_results].execute()
        if known_country:
            country_filter = Q("term", country_code3=known_country)
            res = conn.query(q).filter(country_filter).sort({"alt_name_length": {'order': "desc"}})[0:max_results].execute()
        else:
            res = conn.query(q).sort({"alt_name_length": {'order': "desc"}})[0:max_results].execute()
    
        choices = res_formatter(res, ex['search_name'], parent_place)

    if remove_correct:
        choices = [c for c in choices if c['geonameid'] != ex['correct_geonamesid']]

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

def _format_country_results(res):
    if not res:
        return None
    results = res['hits']['hits'][0].to_dict()['_source']
    lat, lon = results['coordinates'].split(",") 
    results['lon'] = float(lon)
    results['lat'] = float(lat)
    r = {"extracted_name": "",
         "name": results['name'],
         "lat": lat,
         "lon": lon,
         "admin1_name": results['admin1_name'],
         "admin2_name": results['admin2_name'],
         "country_code3": results['country_code3'],
         "feature_code": results['feature_code'],
         "feature_class": results['feature_class'],
         "geonameid": results['geonameid'],
         "start_char": "",
         "end_char": ""}
    return r

def get_country_entry(iso3c: str, conn):
    """Return the Geonames result for a country given its three letter country code"""
    name_filter = Q("term", country_code3=iso3c) 
    type_filter = Q("term", feature_code="PCLI") 
    res = conn.filter(type_filter).filter(name_filter).execute()
    r = _format_country_results(res)
    return r

def get_country_by_name(country_name: str, conn):
    """Return the Geonames result for a country given its three letter country code"""
    type_filter = Q("term", feature_code="PCLI") 
    q = {"multi_match": {"query": country_name,
                         "fields": ['name', 'asciiname', 'alternativenames'],
                         "type" : "phrase"}}
    res = conn.query(q).filter(type_filter).execute()
    r = _format_country_results(res)
    return r

def get_entry_by_id(geonameid: str, conn):
    """Return the Geonames result for a country given its three letter country code"""
    id_filter = Q("term", geonameid=geonameid) 
    res = conn.filter(id_filter).execute()
    r = _format_country_results(res)
    return r

def get_adm1_country_entry(adm1: str, iso3c: str, conn):
    """
    Return the Geonames result for an ADM1 code.
    
    iso3c can be None if the country isn't known.
    """
    type_filter = Q("term", feature_code="ADM1") 
    q = {"multi_match": {"query": adm1,
                             "fields": ['name', 'asciiname', 'alternativenames'],
                             "type" : "phrase"}}
    if iso3c:
        country_filter = Q("term", country_code3=iso3c) 
        res = conn.query(q).filter(type_filter).filter(country_filter).execute()
    else:
        res = conn.query(q).filter(type_filter).execute()
    r = _format_country_results(res)
    return r

#get_adm1_country_entry("Kaduna", "NGA", conn)