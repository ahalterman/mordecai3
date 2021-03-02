from elasticsearch import Elasticsearch, helpers
from elasticsearch_dsl import Search, Q
import numpy as np

def res_formatter(res):
    """
    Helper function to format the ES results into a format for the ML model
    """
    choices = []
    alt_lengths = []
    for i in res['hits']['hits']:
        i = i.to_dict()['_source']
        d = {"feature_code": i['feature_code'],
            "feature_class": i['feature_class'],
            "country_code3": i['country_code3'],
            "name": i['name'],
            "geonameid": i['geonameid']}
        choices.append(d)
        alt_lengths.append(len(i['alternativenames']))
    alt_lengths = np.array(alt_lengths)
    if len(alt_lengths) > 0:
        alt_lengths = (alt_lengths - np.min(alt_lengths)) / np.max(alt_lengths)

    for n, i in enumerate(choices):
        i['alt_name_length'] = alt_lengths[n]
    return choices


def add_es_data(ex, conn, fuzzy=True):
    q = {"multi_match": {"query": ex['placename'],
                                 "fields": ['name', 'asciiname', 'alternativenames'],
                                "type" : "phrase"}}
    res = conn.query(q).sort({"alt_name_length": {'order': "desc"}})[0:50].execute()
    choices = res_formatter(res)
    if not choices and fuzzy:
        q = {"multi_match": {"query": ex['placename'],
                             "fields": ['name', 'alternativenames', 'asciiname'],
                             "fuzziness" : 1,
                            }}
        res = conn.query(q)[0:10].execute()
        choices = res_formatter(res)
    ex['es_choices'] = choices
    if 'correct_geonamesid' in ex.keys():
        ex['correct'] = [c['geonameid'] == ex['correct_geonamesid'] for c in choices]
    return ex


def doc_to_ex_expanded(doc):
    data = []
    doc_tensor = np.max(np.vstack([i._.tensor for i in doc]), axis=0)
    loc_ents = [ent for ent in doc.ents if ent.label_ in ['GPE', 'LOC']]
    for ent in doc.ents:
        if ent.label_ in ['GPE', 'LOC']:
            print(ent.text)
            tensor = np.mean(np.vstack([i._.tensor for i in ent]), axis=0)
            other_locs = [i for e in loc_ents for i in e if i not in ent]
            if other_locs:
                locs_tensor = np.max(np.vstack([i._.tensor for i in other_locs]), axis=0)
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