
from mordecai3.geonames import DataExtent
from mordecai3.elasticsearch import (setup_es_client, check_es_and_geonames)



def test_setup_es_client_with_nonsense():
    """Test that setup_es_client fails gracefully with bad host."""
    client = setup_es_client(hosts=["nonsense_host_12345"])
    assert client is not None


def test_check_es_and_geonames_no_es():
    """Test that check_es_and_geonames returns NA when ES is not available."""
    client = setup_es_client(hosts=["nonsense_host_12345"])
    res = check_es_and_geonames(client)
    assert res[0] == DataExtent.NA
