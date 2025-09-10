
import pytest

from mordecai3 import Geoparser
from mordecai3.elastic_utilities import (setup_es, es_check_geonames_index, 
                                         es_determine_data_extent, DataExtent)

if not es_check_geonames_index():
    pytest.skip("Elasticsearch isn't available", allow_module_level=True)

conn = setup_es()

if es_determine_data_extent(conn) == DataExtent.NONE:
    pytest.skip("Elasticsearch Geonames index is empty", allow_module_level=True)


def test_geoparse_doc():
    geo = Geoparser(hosts=["localhost"])
    res = geo.geoparse_doc("I visited The Hague in the Netherlands.")
    assert res["geolocated_ents"][0]["name"] == "The Hague"