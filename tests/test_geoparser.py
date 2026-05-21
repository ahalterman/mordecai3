



def test_geoparse_doc(geoparser_all_data):
    geo = geoparser_all_data
    res = geo.geoparse_doc("I visited The Hague in the Netherlands.")
    assert res["geolocated_ents"][0]["name"] == "Hague"


def test_geoparse_doc_with_spacy_doc(geoparser_all_data):
    geo = geoparser_all_data
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