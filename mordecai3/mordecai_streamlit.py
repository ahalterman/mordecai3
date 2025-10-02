import spacy
import streamlit as st
import torch

from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search
from importlib import resources
from spacy.language import Language
from spacy.tokens import Token

from mordecai3  import Geoparser 
from torch_model import geoparse_model
from mordecai_utilities import spacy_doc_setup

HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""

try:
    Token.set_extension('tensor', default=False)
except ValueError:
    pass


# define and register "token_tensors" component with spaCy
spacy_doc_setup()



@st.cache_resource
def load_nlp():
    nlp = spacy.load("en_core_web_trf")
    nlp.add_pipe("token_tensors")
    return nlp



@st.cache_resource
def setup_es():
    kwargs = dict(
        hosts=['localhost'],
        port=9200,
        use_ssl=False,
    )
    CLIENT = Elasticsearch(**kwargs)
    conn = Search(using=CLIENT, index="geonames")
    return conn

@st.cache_resource
def load_model():
    model = geoparse_model(device=-1,
                           bert_size = 768,
                           num_feature_codes=54)
    model_path = str(resources.files("mordecai3") / "assets/mordecai_2025-08-27.pt")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

@st.cache_resource
def load_geo():
    geo = Geoparser(model_path=resources.files("mordecai3") / "assets/mordecai_2025-08-27.pt", 
                 geo_asset_path=resources.files("mordecai3") / "assets",
                 hosts=["localhost"],
                 nlp=None,
                 event_geoparse=True,
                 debug=False,
                 trim=None)
    return geo


st.title('Mordecai geoparsing (v3)')
nlp = load_nlp()
conn = setup_es()
#model = load_model()
geo = load_geo()

#= "Afghanistan's major population centers are all government-held, with capital city Kabul especially well-fortified, though none are immune to occasional attacks by Taliban operatives. And though the conflict sometimes seems to engulf the whole country, the provinces of Panjshir, Bamyan, and Nimroz stand out as being mostly free of Taliban influence."
#default_text = 'A "scorched earth"-type policy was used in the city of New York City and the north-western governorate of Idleb.'
default_text = """COTABATO CITY (MindaNews/03 March) – A provincial board member is proposing the declaration of a state of calamity in the entire province of Maguindanao as more residents are fleeing in at least nine towns due to armed conflict."""
text = st.text_area("Text to geoparse", default_text)

print(text)

doc = nlp(text)

output = geo.geoparse_doc(doc)

st.write(output)

#labels = ["GPE", "LOC"]
#html = spacy.displacy.render(doc, style="ent", options={"ents": labels})
#html = html.replace("\n", " ")
#st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)
#
#try:
#    for (ent, pred) in zip(es_data, pred_val):
#        st.markdown("**Place name**: {}".format(ent['placename']))
#        print(len(ent['es_choices']))
#        print(len(pred))
#        for n, i in enumerate(pred):
#            if n < len(ent['es_choices']):
#                ent['es_choices'][n]['score'] = i
#        results = [e for e in ent['es_choices'] if 'score' in e.keys()]
#        results = sorted(results, key=lambda k: -k['score'])
#        results = results[:3]
#        print(ent)
#        for i in results:
#            st.text(f"{i['name']} ({i['feature_code']}), {i['country_code3']}: {i['score']}")
#except NameError:
#    st.text("No entities found.")
