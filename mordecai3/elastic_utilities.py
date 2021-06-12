from elasticsearch import Elasticsearch, helpers
from elasticsearch_dsl import Search, Q
import numpy as np
import jellyfish
from collections import Counter
import warnings
import re

def make_conn():
    kwargs = dict(
        hosts=['localhost'],
        port=9200,
        use_ssl=False,
    )
    CLIENT = Elasticsearch(**kwargs)
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
    """Take in a document's worth of examples and return the count of adm1s"""
    admin1s = []
    
    for n, es in enumerate(out):
        other_adm1 = set([i['admin1_name'] for i in es['es_choices']])
        admin1s.extend(list(other_adm1))
    
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

def res_formatter(res, placename):
    """
    Helper function to format the ES/Geonames results into a format for the ML model, including
    edit distance statistics.

    Parameters
    ----------
    res: Elasticsearch/Geonames output
    placename: str
      The original search term

    Returns
    -------
    choices: list
      List of formatted Geonames results, including edit distance statistics
    """
    choices = []
    alt_lengths = []
    min_dist = []
    max_dist = []
    avg_dist = []
    ascii_dist = []
    for i in res['hits']['hits']:
        i = i.to_dict()['_source']
        names = [i['name']] + i['alternativenames'] 
        dists = [jellyfish.levenshtein_distance(placename, j) for j in names]
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
        choices.append(d)
        alt_lengths.append(len(i['alternativenames']))
        dists = [jellyfish.levenshtein_distance(placename, j) for j in names]
        min_dist.append(np.min(dists))
        max_dist.append(np.max(dists))
        avg_dist.append(np.mean(dists))
        ascii_dist.append(jellyfish.levenshtein_distance(placename, i['asciiname']))
    alt_lengths = normalize(alt_lengths)
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

def _clean_placename(placename):
    """
    Strip out place names that might be preventing the right results
    """
    placename = re.sub("tribal district", "", placename).strip()
    placename = re.sub("[Cc]ity", "", placename).strip()
    placename = re.sub("[Dd]istrict", "", placename).strip()
    placename = re.sub("[Mm]etropolis", "", placename).strip()
    placename = re.sub("[Cc]ounty", "", placename).strip()
    placename = re.sub("[Rr]egion", "", placename).strip()
    return placename

def add_es_data(ex, conn, max_results=50, fuzzy=True, limit_types=False):
    """
    Run an Elasticsearch/geonames query for a single example and add the results

    Parameters
    ---------
    ex: dict
      output of doc_to_ex_expanded
    conn: elasticsearch connection

    Examples
    --------
    d = {"placename": ent.text,
         "tensor": tensor,
         "doc_tensor": doc_tensor,
         "locs_tensor": locs_tensor,
         "sent": ent.sent.text,
         "start_char": ent[0].idx,
         "end_char": ent[-1].idx + len(ent.text)}
    """
    placename = ex['placename']
    placename = re.sub("^the", "", placename).strip()
    if limit_types:
        q = {"multi_match": {"query": placename,
                             "fields": ['name', 'asciiname', 'alternativenames'],
                             "type" : "phrase"}}
        p_filter = Q("term", feature_class="P")
        a_filter = Q("term", feature_class="A")
        combined_filter = p_filter | a_filter
        res = conn.query(q).filter(combined_filter).sort({"alt_name_length": {'order': "desc"}})[0:max_results].execute()
    else:
        q = {"multi_match": {"query": placename,
                                 "fields": ['name', 'asciiname', 'alternativenames'],
                                "type" : "phrase"}}
        res = conn.query(q).sort({"alt_name_length": {'order': "desc"}})[0:max_results].execute()
    
    choices = res_formatter(res, ex['placename'])
    if fuzzy and not choices:
        placename = _clean_placename(placename)
        q = {"multi_match": {"query": placename,
                             "fields": ['name', 'alternativenames', 'asciiname'],
                             "fuzziness" : 1,
                            }}
        res = conn.query(q).sort({"alt_name_length": {'order': "desc"}})[0:max_results].execute()
        choices = res_formatter(res, ex['placename'])

    ex['es_choices'] = choices
    if 'correct_geonamesid' in ex.keys():
        ex['correct'] = [c['geonameid'] == ex['correct_geonamesid'] for c in choices]
    return ex

def add_es_data_doc(doc_ex, conn, max_results=50, limit_types=False):
    doc_es = []
    for ex in doc_ex:
        with warnings.catch_warnings():
            try:
                es = add_es_data(ex, conn, max_results, limit_types)
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


