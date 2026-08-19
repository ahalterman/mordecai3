

import logging
import re

from elasticsearch import Elasticsearch
from elasticsearch_dsl import Q, Search
from enum import IntEnum

from .exceptions import GeonamesQueryError

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
        self.index = "geonames"
        self.search = Search(using=self.conn, index=self.index)
        # Cap on sub-queries per _msearch request. Responses are the constraint,
        # not the request body: 100 queries x 100 results is roughly 8MB back.
        self.msearch_chunk_size = 100
        # Caches to avoid redundant ES queries when the same place names recur,
        # which happens a lot within and across documents in a batch.
        # `_es_cache` holds full candidate lists from add_es_data (see geoparse.py);
        # `_parent_cache` holds the country/ADM1 lookups used to resolve "in" relations.
        # Both are per-instance rather than module-level so that separate Geoparsers
        # (or parallel workers) don't share state, and so the cache is garbage
        # collected with the service rather than growing for the life of the process.
        self._es_cache: dict[tuple, list] = {}
        self._parent_cache: dict[tuple, dict | None] = {}

    def clear_cache(self):
        """Clear the ES result and parent lookup caches."""
        self._es_cache.clear()
        self._parent_cache.clear()
        logger.debug("Geonames caches cleared")

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
        cache_key = ("adm1_country", adm1, iso3c)
        if cache_key in self._parent_cache:
            return self._parent_cache[cache_key]
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
        self._parent_cache[cache_key] = r
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
        cache_key = ("country_by_name", country_name)
        if cache_key in self._parent_cache:
            return self._parent_cache[cache_key]
        type_filter = Q("term", feature_code="PCLI")
        q = {"multi_match": {"query": country_name,
                            "fields": ['name', 'asciiname', 'alternativenames'],
                            "type" : "phrase"}}
        res = self.search.query(q).filter(type_filter).execute()
        r = _format_country_results(res)
        self._parent_cache[cache_key] = r
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
        return self.build_name_search(search_name, max_results, fuzzy,
                                      limit_types, known_country).execute()

    def build_name_search(self,
                          search_name: str,
                          max_results: int=50,
                          fuzzy: int=0,
                          limit_types: bool=False,
                          known_country: str | None=None) -> Search:
        """
        Build (but don't run) the Search for a single name lookup.

        This is the single source of truth for the name query, so that the
        one-at-a-time path (search_by_name) and the batched _msearch path
        send byte-identical queries.

        NOTE: `limit_types` currently has no effect on the results. In the
        original code the feature-class filter was built and executed, but the
        unfiltered `else` branch below then overwrote the result whenever
        `known_country` was unset, so the filtered query was always discarded.
        That behavior is preserved here deliberately -- tools/train.py passes
        limit_types when building training data, so "fixing" it would silently
        change the candidate sets the model is trained on. The only change is
        that we no longer pay for the throwaway query.
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

        s = self.search.query(q)
        if known_country:
            s = s.filter(Q("term", country_code3=known_country))
        return s.sort({"alt_name_length": {'order': "desc"}})[0:max_results]

    #
    #   Batched (_msearch) lookups
    #   ==========================
    #

    def _msearch(self, searches: list[Search]) -> list[dict]:
        """
        Run many Searches in one (or a few) _msearch requests.

        Each sub-search is executed by ES exactly as if it had been sent on its
        own -- same analyzer, scoring, and sort -- so results are identical to
        looping over .execute(). The win is purely that the per-request costs
        (HTTP round trip, urllib3, JSON parse, elasticsearch_dsl object
        construction) are paid once per chunk instead of once per query.

        Returns raw response dicts, one per input Search, in the same order.
        """
        if not searches:
            return []

        out: list[dict] = []
        for start in range(0, len(searches), self.msearch_chunk_size):
            chunk = searches[start:start + self.msearch_chunk_size]
            body = []
            for s in chunk:
                body.append({"index": self.index})
                body.append(s.to_dict())
            resp = self.conn.msearch(body=body)
            responses = resp.get("responses")
            if responses is None or len(responses) != len(chunk):
                # A parse-time error rejects the whole request rather than
                # returning a per-query error, so there is nothing to salvage.
                raise GeonamesQueryError(
                    f"_msearch returned {0 if responses is None else len(responses)} "
                    f"responses for {len(chunk)} queries")
            for s, r in zip(chunk, responses):
                # Check explicitly: a failed sub-search has no 'hits' key, and
                # silently treating that as "no candidates" would look exactly
                # like "this place isn't in geonames".
                if "error" in r or r.get("status", 200) >= 400:
                    raise GeonamesQueryError(
                        f"_msearch sub-query failed (status "
                        f"{r.get('status')}): {r.get('error')}")
                out.append(r)
        return out

    def search_by_names(self, specs: list[tuple]) -> list[dict]:
        """
        Batched form of search_by_name.

        Parameters
        ----------
        specs: list of tuples
          Each is (search_name, max_results, fuzzy, limit_types, known_country).

        Returns
        -------
        list of raw ES response dicts, one per spec, in order.
        """
        return self._msearch([self.build_name_search(*spec) for spec in specs])

    def get_country_by_name_batch(self, names: list[str]) -> dict:
        """Batched get_country_by_name. Returns {name: entry or None}."""
        todo = [n for n in dict.fromkeys(names)
                if ("country_by_name", n) not in self._parent_cache]
        if todo:
            type_filter = Q("term", feature_code="PCLI")
            searches = [
                self.search.query({"multi_match": {
                    "query": n,
                    "fields": ['name', 'asciiname', 'alternativenames'],
                    "type": "phrase"}}).filter(type_filter)
                for n in todo]
            for n, res in zip(todo, self._msearch(searches)):
                self._parent_cache[("country_by_name", n)] = _format_country_results(res)
        return {n: self._parent_cache[("country_by_name", n)]
                for n in dict.fromkeys(names)}

    def get_adm1_country_entry_batch(self, adm1_names: list[str]) -> dict:
        """Batched get_adm1_country_entry (no country filter). Returns {name: entry or None}."""
        todo = [n for n in dict.fromkeys(adm1_names)
                if ("adm1_country", n, None) not in self._parent_cache]
        if todo:
            type_filter = Q("term", feature_code="ADM1")
            searches = [
                self.search.query({"multi_match": {
                    "query": n,
                    "fields": ['name', 'asciiname', 'alternativenames'],
                    "type": "phrase"}}).filter(type_filter)
                for n in todo]
            for n, res in zip(todo, self._msearch(searches)):
                self._parent_cache[("adm1_country", n, None)] = _format_country_results(res)
        return {n: self._parent_cache[("adm1_country", n, None)]
                for n in dict.fromkeys(adm1_names)}




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


def hit_sources(res) -> list[dict]:
    """
    Return the `_source` dict for each hit in an ES response.

    Accepts either a raw response dict (what _msearch returns) or an
    elasticsearch_dsl Response (what .execute() returns), so the batched and
    one-at-a-time paths can share the same result formatting code.
    """
    if res is None:
        return []
    try:
        hits = res['hits']['hits']
    except (KeyError, TypeError):
        return []
    return [h['_source'] if isinstance(h, dict) else h.to_dict()['_source']
            for h in hits]


def _format_country_results(res):
    sources = hit_sources(res)
    if not sources:
        return None
    results = sources[0]
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



