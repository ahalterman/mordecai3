import os
import pytest

from mordecai3.geonames import DataExtent, GeonamesService
from mordecai3.elasticsearch import (
    setup_es_client,
    check_es_and_geonames,
    es_is_accepting_connection,
    es_has_geonames_index,
)
from mordecai3.logging import setup_logging

ES_HOST = os.getenv("ES_HOST", "localhost")
ES_PORT = int(os.getenv("ES_PORT", 9200))

setup_logging()


@pytest.fixture(scope="session")
def es_client():
    return setup_es_client(hosts=[ES_HOST], port=ES_PORT)


@pytest.fixture(scope="session")
def geonames_service(es_client):
    if not es_is_accepting_connection(es_client):
        pytest.skip("Elasticsearch not available")
    if not es_has_geonames_index(es_client):
        pytest.skip("Geonames index not found")
    return GeonamesService(es_client=es_client)


@pytest.fixture(scope="session")
def geonames_service_test_data(geonames_service):
    if geonames_service.determine_data_extent() < DataExtent.TEST:
        pytest.skip("Geonames test data not available")
    return geonames_service


@pytest.fixture(scope="session")
def geonames_service_all_data(geonames_service):
    if geonames_service.determine_data_extent() < DataExtent.ALL:
        pytest.skip("Full geonames data not available")
    return geonames_service


@pytest.fixture(scope="session")
def geoparser_all_data(geonames_service_all_data):
    from mordecai3.geoparse import Geoparser
    return Geoparser(geonames=geonames_service_all_data)


@pytest.fixture(scope="session")
def geoparser_test_data(geonames_service_test_data):
    from mordecai3.geoparse import Geoparser
    return Geoparser(geonames=geonames_service_test_data)


@pytest.fixture(scope="session", autouse=True)
def log_data_extent(es_client):
    extent, msg = check_es_and_geonames(es_client)
    if extent == DataExtent.NA:
        print(f"\n⚠️  WARNING: {msg} - skipping ES-dependent tests\n")
    elif extent == DataExtent.NONE:
        print(f"\n⚠️  WARNING: {msg} - skipping ES-dependent tests\n")
    elif extent == DataExtent.TEST:
        print(f"\nℹ️  {msg}\n")
    else:
        print(f"\n✅ {msg}\n")
