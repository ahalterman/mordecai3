import os
import pytest

from mordecai3.geonames import DataExtent
from mordecai3.elasticsearch import setup_es_client, check_es_and_geonames
from mordecai3.logging import setup_logging

# Set default ES settings
ES_HOST = os.getenv("ES_HOST", "localhost")
ES_PORT = int(os.getenv("ES_PORT", 9200))


# Quieten 3rd party loggers
setup_logging()


@pytest.fixture(scope="session")
def es_client():
    client = setup_es_client(hosts=[ES_HOST], port=ES_PORT)
    return client


@pytest.fixture(scope="session")
def geonames_data_extent(es_client):
    """Print startup info about external services."""
    
    res = check_es_and_geonames(es_client)
    return res


@pytest.fixture(scope="session")
def all_data_required(geonames_data_extent):
    extent = geonames_data_extent[0]
    if extent < DataExtent.ALL:
        pytest.skip(
            f"Geonames data not available (extent: {extent})",
        )


@pytest.fixture(scope="module")
def test_data_required(geonames_data_extent):
    extent = geonames_data_extent[0]
    if extent < DataExtent.TEST:
        pytest.skip(
            f"Geonames test data not available (extent: {extent})",
        )


@pytest.fixture(scope="session", autouse=True)
def log_data_extent(geonames_data_extent):
    res = geonames_data_extent
    if res[0] == DataExtent.NA:
        print("\n⚠️  WARNING: Elasticsearch is not available - skipping ES-dependent tests\n")
    elif res[0] == DataExtent.NONE:
        print("\n⚠️  WARNING: Elasticsearch Geonames index is empty - skipping ES-dependent tests\n")
    elif res[0] == DataExtent.TEST:
        print("\nℹ️ Elasticsearch is available with test data\n")
    else:
        print("\n✅ Elasticsearch is available with full data\n")
