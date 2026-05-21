

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


def test_geoparse_doc(geonames_service_all_data):
    geo = Geoparser(es_client=geonames_service_all_data.conn)
    res = geo.geoparse_doc("I visited The Hague in the Netherlands.")
    assert res["geolocated_ents"][0]["name"] == "Hague"


def test_geoparse_doc_with_spacy_doc(geonames_service_all_data):
    geo = Geoparser(es_client=geonames_service_all_data.conn)
    import spacy
    from mordecai3.mordecai_utilities import spacy_doc_setup
    def load_nlp():
        nlp = spacy.load("en_core_web_trf")
        nlp.add_pipe("token_tensors")
        return nlp

    spacy_doc_setup()
    nlp = load_nlp()
    doc = nlp("I visited The Hague in the Netherlands.")
    res = geo.geoparse_doc(doc)
    assert res["geolocated_ents"][0]["name"] == "Hague" # It's getting the US