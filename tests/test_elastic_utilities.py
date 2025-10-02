
import pytest

from mordecai3.elastic_utilities import (setup_es, get_adm1_country_entry, 
                                         es_check_geonames_index, es_determine_data_extent,
                                         DataExtent)


if not es_check_geonames_index():
    pytest.skip("Elasticsearch isn't available", allow_module_level=True)

conn = setup_es()

if es_determine_data_extent(conn) == DataExtent.NONE:
    pytest.skip("Elasticsearch Geonames index is empty", allow_module_level=True)


def test_get_adm1_country_entry():
    nld = get_adm1_country_entry("North Holland", "NLD", conn)
    assert nld is not None
    assert nld["geonameid"] == "2749879"
    
    # Non-existent entry
    xyz = get_adm1_country_entry("NonExistent", None, conn)
    assert xyz is None