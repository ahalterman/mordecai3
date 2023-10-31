import numpy as np
import pandas as pd
import spacy
import streamlit as st
import torch
from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search
from torch.utils.data import DataLoader

import mordecai3.elastic_utilities as es_util
from mordecai3.geoparse import doc_to_ex_expanded
from mordecai3.roberta_qa import add_event_loc, setup_qa
from mordecai3.torch_model import ProductionData, geoparse_model
from mordecai3.utilities import spacy_doc_setup


# for dumping raw output to JSON
# https://stackoverflow.com/a/52604722
def default(obj):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    raise TypeError('Unknown type:', type(obj))


spacy_doc_setup()

HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""


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
    model = geoparse_model(device = device,
                                bert_size = 768,
                                num_feature_codes=54)
    model.load_state_dict(torch.load("../mordecai3/mordecai_new.pt"))
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
freeform_qa = st.sidebar.text_input(label='Advanced option: write a complete question here.")',
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

if freeform_qa:
    QA_input = {
            'question': freeform_qa,
            'context':text
        }
    res = trf(QA_input)
    event_doc = add_event_loc(doc, res)
elif icews_cat:
    #question = f"Where did {icews_cat.lower()} happen?"
    question = f"Which place did {icews_cat.lower()} happen?"
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
        st.markdown("**Place name**: {}".format(ent['search_name']))
        print(len(ent['es_choices']))
        if len(ent['es_choices']) < 10:
            print(ent['es_choices'])
        for n, score in enumerate(pred):
            if n < len(ent['es_choices']):
                ent['es_choices'][n]['score'] = score.item() # torch tensor --> float
        results = [e for e in ent['es_choices'] if 'score' in e.keys()]
        if not results:
            st.text("(no results)")
        if results:
            results = sorted(results, key=lambda k: -k['score'])
            results = [i for i in results if i['score'] > 0.01]
            print(results)
            results = results[:3]
            best = results[0]
            pretty.append({"lat": float(best['lat']), "lon": float(best['lon']), "name": best['name']})

            for n, i in enumerate(results):
                if n == 0:
                    st.text(f"✔️ {i['name']} ({i['feature_code']}), {i['admin1_name']}, {i['country_code3']} ({i['geonameid']}): {i['score']}")
                    if len(results) > 1:
                        st.text("Other choices: ")
                else:
                    st.text(f"* {i['name']} ({i['feature_code']}), {i['admin1_name']}, {i['country_code3']} ({i['geonameid']}): {i['score']}")

    map = st.sidebar.checkbox("Show map", value = False)
    if map:
        st.subheader("Map")
        df = pd.DataFrame(pretty)
        st.map(df)
    show_raw = st.sidebar.checkbox("Show raw output", value = False)
    if show_raw:
        st.subheader("Raw JSON result")
        #dumped = json.dumps(es_data, default=default)
        st.json(es_data)
#except NameError:
#    st.text("No entities found.")
