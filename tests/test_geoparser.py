
import pytest

from mordecai3 import Geoparser
from mordecai3.elasticsearch import setup_es_client


def test_geoparser_can_be_initialized():
    geo = Geoparser(hosts=["localhost"])
    assert isinstance(geo, Geoparser)


# issue #17
def test_geoparser_arbitrary_es_connection():
    client = setup_es_client(hosts=["localhost"], port=9200, request_timeout=1)
    geo = Geoparser(es_client=client)
    assert isinstance(geo, Geoparser)


def test_geoparse_doc(all_data_required):
    geo = Geoparser(hosts=["localhost"])
    res = geo.geoparse_doc("I visited The Hague in the Netherlands.")
    assert res["geolocated_ents"][0]["name"] == "The Hague"