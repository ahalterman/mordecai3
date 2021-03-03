import spacy
from spacy.language import Language
from spacy.tokens import Token, Doc
from spacy.pipeline import Pipe
import numpy as np
import jsonlines
from tqdm import tqdm
import re
import streamlit as st
import torch
from torch.utils.data import Dataset, DataLoader

from elastic_utilities import res_formatter, doc_to_ex_expanded, add_es_data
from torch_bert_placename_compare import ProductionData, embedding_compare

from elasticsearch import Elasticsearch, helpers
from elasticsearch_dsl import Search, Q

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
    model = embedding_compare(bert_size = 768,
                                num_feature_codes=54, 
                                max_choices=25) 
    model.load_state_dict(torch.load("mordecai2.pt"))
    model.eval()
    return model



st.title('Mordecai geoparsing (v3)')
nlp = load_nlp()
conn = setup_es()
model = load_model()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

#= "Afghanistan's major population centers are all government-held, with capital city Kabul especially well-fortified, though none are immune to occasional attacks by Taliban operatives. And though the conflict sometimes seems to engulf the whole country, the provinces of Panjshir, Bamyan, and Nimroz stand out as being mostly free of Taliban influence."
#default_text = 'A "scorched earth"-type policy was used in the city of New York City and the north-western governorate of Idleb.'
default_text = """COTABATO CITY (MindaNews/03 March) – A provincial board member is proposing the declaration of a state of calamity in the entire province of Maguindanao as more residents are fleeing in at least nine towns due to armed conflict.

The latest wave of evacuations occurred immediately after Armed Forces Chief of Staff Gen. Gregorio Pio Catapang launched an “all-out offensive” against the Bangsamoro Islamic Freedom Fighters (BIFF) on February 25.

Maguindanao board member Bobby Katambak said the Sangguniang Panglalawigan will hold an emergency session Tuesday to pass a resolution declaring the entire province of Maguindanao under a state of calamity. Maguindanao has 36 towns.

“This will give the provincial government an authority to use its calamity fund to provide relief assistance to internally displaced persons,” Katambak, a lawyer, told reporters.

“We call on the attention of (Maguindanao Vice Governor and SP presiding officer) Datu Lester Sinsuat for the conduct of emergency session to act on the resolution because the affected towns swelled to 11,” Katambak said.

The Autonomous Region in Muslim Mindanao’s Humanitarian Emergency Action and Response Team (ARMM-HEART) in its Situation 03-2A report released Monday night said a total of 41,720 residents were displaced in five towns with two towns serving as “host towns.”

No need for declaration

Under RA 10121 or the Disaster Risk Reduction and Management Act of 2010, a declaration of a state of calamity is no longer necessary to access and utilize the DRRM Fund or what used to be referred to as “calamity fund.”

Under the law, the local DRRM Fund “shall be sourced from not less than 5% of the estimated revenue from regular sources.”

Thirty per cent of the DRRM fund shall be set aside as a “quick response fund” for relief and recovery programs.

On Monday, ARMM Governor Mujiv Hataman met with all relief agencies of the region, including security officials, local government executives and representatives from international non-government organizations.

The meeting aimed to centralize all relief operations and avoid duplication of relief distribution and missing other beneficiaries.

Hataman is concerned over health and sanitation issues in evacuation sites.

Hataman will lead Tuesday the ARMM relief and medical missions to various towns in Maguindanao.

The Humanitarian Emergency Action and Response Team (HEART), the region’s relief arm, reported that as of evening of March 2, 8,139 families or 41,720 individuals had been displaced by the “all-out offensive” against the BIFF.

The evacuees of Pagalungan (1,900 families) and Datu Montawal towns (400) – or about 12,650 persons who fled their homes during the skirmishes between the BIFF and the Moro Islamic Liberation Front (MILF) were supposed to return home but opted to stay on following Catapang’s “all-out offensive.”   (Ferdinandh B. Cabrera / MindaNews)"""
text = st.text_area("Text to geoparse", default_text)    
doc = nlp(text)
ex = doc_to_ex_expanded(doc)
es_data = []
for e in ex:
    d = add_es_data(e, conn)
    es_data.append(d)

dataset = ProductionData(es_data)
data_loader = DataLoader(dataset=dataset, batch_size=64)
with torch.no_grad():
    model.eval()
    for X_val, code_val, doc_val, loc_val, country_val in data_loader:
        X_val = X_val.to(device)
        code_val = code_val.to(device)
        loc_val = loc_val.to(device)
        country_val = country_val.to(device)

        pred_val = model(X_val, code_val, country_val, loc_val)

labels = ["GPE", "LOC"]
html = spacy.displacy.render(doc, style="ent", options={"ents": labels})
html = html.replace("\n", " ")
st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)

try:
    for (ent, pred) in zip(es_data, pred_val):
        st.markdown("**Place name**: {}".format(ent['placename']))
        print(len(ent['es_choices']))
        print(len(pred))
        for n, i in enumerate(pred):
            if n < len(ent['es_choices']):
                ent['es_choices'][n]['score'] = i
        results = [e for e in ent['es_choices'] if 'score' in e.keys()]
        results = sorted(results, key=lambda k: -k['score']) 
        results = results[:3]
        print(ent)
        for i in results:
            st.text(f"{i['name']} ({i['feature_code']}), {i['country_code3']}: {i['score']}")
except NameError:
    st.text("No entities found.")