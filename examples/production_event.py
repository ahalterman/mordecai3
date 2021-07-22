
import jsonlines
from tqdm import tqdm
import re

import streamlit as st
import torch
import pandas as pd
from elasticsearch import Elasticsearch, helpers
from elasticsearch_dsl import Search, Q
import spacy
from spacy.language import Language
from spacy.tokens import Token, Doc
from spacy.pipeline import Pipe
import numpy as np
from torch.utils.data import Dataset, DataLoader

import mordecai3.elastic_utilities as es_util
from mordecai3.format_geoparsing_data import doc_to_ex_expanded
from mordecai3.torch_bert_placename_compare import ProductionData, embedding_compare
from mordecai3.roberta_qa import setup_qa, add_event_loc



HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""

try:
    Token.set_extension('tensor', default=False)
except ValueError:
    pass

try:
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
except ValueError:
    pass



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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = embedding_compare(device = device,
                                bert_size = 768,
                                num_feature_codes=54) 
    model.load_state_dict(torch.load("../mordecai3/assets/mordecai2.pt"))
    model.eval()
    return model

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_trf():
    trf = setup_qa()
    return trf



st.title('Mordecai geoparsing (v3)')
nlp = load_nlp()
conn = setup_es()
model = load_model()
trf = load_trf()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

icews_cat = st.sidebar.text_input(label='CAMEO/ICEWS event description (e.g. "Use conventional military force")',
value="")

#= "Afghanistan's major population centers are all government-held, with capital city Kabul especially well-fortified, though none are immune to occasional attacks by Taliban operatives. And though the conflict sometimes seems to engulf the whole country, the provinces of Panjshir, Bamyan, and Nimroz stand out as being mostly free of Taliban influence."
#default_text = 'A "scorched earth"-type policy was used in the city of New York City and the north-western governorate of Idleb.'
default_text = """Speaking from Berlin, President Obama expressed his hope for a peaceful resolution to the fighting in Homs and Aleppo."""
text = st.text_area("Text to geoparse", default_text)    
doc = nlp(text)



print("Doc ents: ", doc.ents)
doc_ex = doc_to_ex_expanded(doc)
if doc_ex:
    es_data = es_util.add_es_data_doc(doc_ex, conn, max_results=500)

    dataset = ProductionData(es_data, max_choices=500)
    data_loader = DataLoader(dataset=dataset, batch_size=64, shuffle=False)
    with torch.no_grad():
        model.eval()
        for input in data_loader:
            pred_val = model(input)

if icews_cat:
    question = f"Where did {icews_cat.lower()} happen?"
    QA_input = {
            'question': question,
            'context':text
        }
    res = trf(QA_input)
    event_doc = add_event_loc(doc, res)
else:
    event_doc = doc

labels = ["GPE", "LOC", "FAC", "EVENT_LOC"]
html = spacy.displacy.render(event_doc, style="ent", options={"ents": labels})
html = html.replace("\n", " ")
st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)


#try:
if len(doc_ex) == 0:
    st.text("No entities found.")
elif len(es_data) == 0:
    st.text("No entities found.")
else:
    pretty = []
    for (ent, pred) in zip(es_data, pred_val):
        st.markdown("**Place name**: {}".format(ent['placename']))
        print(len(ent['es_choices']))
        for n, score in enumerate(pred):
            if n < len(ent['es_choices']):
                ent['es_choices'][n]['score'] = score
        results = [e for e in ent['es_choices'] if 'score' in e.keys()]
        if not results:
            st.text("(no results)")
        if results:
            results = sorted(results, key=lambda k: -k['score'])
            results = [i for i in results if i['score'] > 0.01]
            results = results[:3]
            best = results[0]
            pretty.append({"lat": float(best['lat']), "lon": float(best['lon']), "name": best['name']})

            for n, i in enumerate(results):
                if n == 0:
                    st.text(f"✔️ {i['name']} ({i['feature_code']}), {i['country_code3']}: {i['score']}")
                    if len(results) > 1:
                        st.text("Other choices: ")
                else:
                    st.text(f"* {i['name']} ({i['feature_code']}), {i['country_code3']}: {i['score']}")

    map = st.checkbox("Show map", value = False) 
    if map:
        st.subheader("Map")
        df = pd.DataFrame(pretty)
        st.map(df)
    #st.subheader("Raw JSON result")
    #st.json(output)
#except NameError:
#    st.text("No entities found.")
