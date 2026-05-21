

import logging
import re

from elasticsearch import Elasticsearch
from elasticsearch_dsl import Q, Search
from enum import IntEnum

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


#
#   Helpers for data extent checking, used in elasticsearch.py
#   ================================
#
#   But depend on GeonamesService functionality, so leave it here
#

# Using an IntEnum here so that we can test whether sufficient data for a test
# is present, since if we have "all" data, we also have "test" data.
class DataExtent(IntEnum):
    NA   = 0  # Fallback for ES client connection problems or missing index   
    NONE = 1
    TEST = 2
    ALL  = 3



#
#   Geonames service class: how to actually use Geonames in mordecai3
#   =========================================
#


class GeonamesService:
    """Class to encapsulate Geonames functionality needed for mordecai3"""
    def __init__(self, es_client: Elasticsearch):
        self.conn = es_client
        self.search = Search(using=self.conn, index="geonames")

    def determine_data_extent(self) -> DataExtent:
        # TODO: this is a bit hacky, but it works for now. 
        """Check what extent of data we have in the ES/Geonames index
        
        Returns
        -------
        DataExtent
        Either "ALL" if the full Geonames dataset is present, "TEST" if only
        the reduced test set is present, or "NONE" if no data appears to be present.
        """
        usa = self.get_adm1_country_entry("New York", "USA")
        nld = self.get_adm1_country_entry("North Holland", "NLD")
        if usa and nld:
            return DataExtent.ALL
        elif nld:   
            return DataExtent.TEST
        else:
            return DataExtent.NONE

    def get_entry_by_id(self, geonameid: str) -> dict | None:
        """Return the Geonames result for a country given its three letter country code"""
        id_filter = Q("term", geonameid=geonameid) 
        res = self.search.filter(id_filter).execute()
        r = _format_country_results(res)
        return r

    def get_adm1_country_entry(self, 
                               adm1: str, 
                               iso3c: str | None=None, 
                               ) -> dict | None:
        """
        Return the Geonames entity for an ADM1 code.
        
        Parameters
        ----------
        adm1: str
        Name of the ADM1 (state/province)
        iso3c: str or None
        Optional three letter country code to limit the search
        conn: elasticsearch connection
        An elasticsearch connection object, as returned by setup_es()

        Examples
        --------
        >>> conn = setup_es()
        >>> get_adm1_country_entry("North Holland", "NLD", conn)
        {'extracted_name': '', 'name': 'Provincie Noord-Holland', 'lat': '52.58333', 'lon': '4.91667', 'admin1_name': 'North Holland', 'admin2_name': '', 'country_code3': 'NLD', 'feature_code': 'ADM1', 'feature_class': 'A', 'geonameid': '2749879', 'start_char': '', 'end_char': ''}
        """
        type_filter = Q("term", feature_code="ADM1") 
        q = {"multi_match": {"query": adm1,
                                "fields": ['name', 'asciiname', 'alternativenames'],
                                "type" : "phrase"}}
        if iso3c:
            country_filter = Q("term", country_code3=iso3c) 
            res = self.search.query(q).filter(type_filter).filter(country_filter).execute()
        else:
            res = self.search.query(q).filter(type_filter).execute()
        r = _format_country_results(res)
        return r

    def get_country_entry(self, iso3c: str):
        """Return the Geonames result for a country given its three letter country code"""
        name_filter = Q("term", country_code3=iso3c) 
        type_filter = Q("term", feature_code="PCLI") 
        res = self.search.filter(type_filter).filter(name_filter).execute()
        r = _format_country_results(res)
        return r

    def get_country_by_name(self, country_name: str) -> dict | None:
        """Return the Geonames result for a country given its three letter country code"""
        type_filter = Q("term", feature_code="PCLI") 
        q = {"multi_match": {"query": country_name,
                            "fields": ['name', 'asciiname', 'alternativenames'],
                            "type" : "phrase"}}
        res = self.search.query(q).filter(type_filter).execute()
        r = _format_country_results(res)
        return r

    def search_by_name(self, 
                       search_name: str, 
                       max_results: int=50, 
                       fuzzy: int=0,
                       limit_types: bool=False,
                       known_country: str | None=None) -> list[dict]:
        """
        Run an Elasticsearch/geonames query for a single example and add the results
        to the object.

        Parameters
        ---------
        search_name: str
            search string
        max_results: int
            Maximum results to bring back from ES
        fuzzy: int
            Allow fuzzy results? 0=exact matches. Higher numbers will increase 
            the fuzziness of the search. 
        limit_types: bool
            Limit types to Q and A types
        known_country: str
            ISO 3 letter country code to restrict results by

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
        search_name = _clean_search_name(search_name)
        
        # Construct query
        if fuzzy:
            q = {"multi_match": {"query": search_name,
                                "fields": ['name', 'alternativenames', 'asciiname'],
                                "fuzziness" : fuzzy,
                                }}
        else:
            q = {"multi_match": {"query": search_name,
                                    "fields": ['name', 'asciiname', 'alternativenames'],
                                    "type" : "phrase"}}

        # Add optional filters
        if limit_types:
            p_filter = Q("term", feature_class="P")
            a_filter = Q("term", feature_class="A")
            combined_filter = p_filter | a_filter
            res = self.search.query(q).filter(combined_filter).sort({"alt_name_length": {'order': "desc"}})[0:max_results].execute()
        if known_country:
            country_filter = Q("term", country_code3=known_country)
            res = self.search.query(q).filter(country_filter).sort({"alt_name_length": {'order': "desc"}})[0:max_results].execute()
        else:
            res = self.search.query(q).sort({"alt_name_length": {'order': "desc"}})[0:max_results].execute()
        
        return res




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



