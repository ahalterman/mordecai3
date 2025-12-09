

from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search
from typing import Any
from urllib3.exceptions import NewConnectionError


from .geonames import DataExtent, determine_geonames_data_extent


def setup_es_client(hosts: str | list[str] | None | Any = "localhost",
                    port: int | None = 9200,
                    **kwargs) -> Elasticsearch:
    """Helper class to setup a Elasticsearch client connection
    
    All arguments are passed on to elasticsearch.Elasticsearch() as is. 

    If you are using the default package setup instructions with a local 
    Elasticsearch instance, you likely do not need to change anything here.

    Otherwise, the keys here correspond to arguments that are passed to 
    elasticsearch.Elasticsearch(), as is. See the official documentation for 
    more details on the different ways setup and connect to ES: 
    
    https://www.elastic.co/docs/reference/elasticsearch/clients/python/connecting
    """
    es = Elasticsearch(hosts=hosts, port=port, **kwargs)
    return es


def make_conn(**kwargs) -> Search:
    """Helper function to setup an Elasticsearch DSL Search connection"""
    es_client = setup_es_client(**kwargs)
    conn = Search(using=es_client, index="geonames")
    return conn


# 
#   Connection and data extent checking
#   ===================================
#
#   The next set of function are utilities to comprehensively check whether 
#   Elasticsearch is running and accessible, whether the Geonames index is
#   present, and whether the Geonames index has data in it.
#


def check_es_and_geonames(es_client: Elasticsearch) -> tuple[DataExtent, str]:
    """Utility function to check the status of ES and the Geonames index"""
    if not es_is_accepting_connection(es_client):
        return DataExtent.NA, "ES connection failed"
    
    if not es_has_geonames_index(es_client):
        return DataExtent.NA, "Can connect to ES, but geonames index missing"
    
    extent = determine_geonames_data_extent(es_client)
    if extent == DataExtent.NONE:
        return DataExtent.NONE, "Can connect to ES, geonames index present, but empty"
    elif extent == DataExtent.TEST:
        return DataExtent.TEST, "Geonames test data present"
    elif extent == DataExtent.ALL:
        return DataExtent.ALL, "Full Geonames data present"
    
    return DataExtent.NA, "Unknown status"


def es_is_accepting_connection(conn: Elasticsearch) -> bool:
    """Check if the Elasticsearch service is running and accessible"""
    try:
        res = conn.ping()
    except (NewConnectionError, ConnectionRefusedError):
        return False
    return res  


def es_has_geonames_index(conn: Elasticsearch) -> bool:
    """Check if the ES GeoNames index is accessible"""
    return conn.indices.exists(index="geonames")


