import numpy as np
import spacy
import streamlit as st
import torch
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search
from spacy.language import Language
from spacy.tokens import Token
from torch_model import geoparse_model

HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""

try:
    Token.set_extension('tensor', default=False)
except ValueError:
    pass


@Language.component("token_tensors")
def token_tensors(doc):
    chunk_len = len(doc._.trf_data.tensors[0][0])
    token_tensors = [[]]*len(doc)

    for n, i in enumerate(doc):
        wordpiece_num = doc._.trf_data.align[n]
        for d in wordpiece_num.dataXd:
            which_chunk = int(np.floor(d[0] / chunk_len))
            which_token = d[0] % chunk_len
            ## You can uncomment this to see that spaCy tokens are being aligned with the correct
            ## wordpieces.
            #wordpiece = doc._.trf_data.wordpieces.strings[which_chunk][which_token]
            #print(n, i, wordpiece)
            token_tensors[n] = token_tensors[n] + [doc._.trf_data.tensors[0][which_chunk][which_token]]
    for n, d in enumerate(doc):
        if token_tensors[n]:
            d._.set('tensor', np.mean(np.vstack(token_tensors[n]), axis=0))
        else:
            d._.set('tensor',  np.zeros(doc._.trf_data.tensors[0].shape[-1]))
    return doc



@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_nlp():
    nlp = spacy.load("en_core_web_trf")
    nlp.add_pipe("token_tensors")
    return nlp



@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def setup_es():
    kwargs = dict(
        hosts=['localhost'],
        port=9200,
        use_ssl=False,
    )
    CLIENT = Elasticsearch(**kwargs)
    conn = Search(using=CLIENT, index="geonames")
    return conn

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_model():
    model = geoparse_model(device=-1,
                           bert_size = 768,
                           num_feature_codes=54)
    model.load_state_dict(torch.load("mordecai_2023-02-07.pt"))
    model.eval()
    return model

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_geo():
    geo = Geoparser(model_path="mordecai_2023-02-07_good.pt", 
                 geo_asset_path="assets",
                 nlp=None,
                 event_geoparse=True,
                 debug=None,
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
