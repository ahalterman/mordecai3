import os
import sys

# `streamlit run` puts this file's own directory -- the mordecai3 package -- at
# the front of sys.path, where `elasticsearch.py` and `logging.py` shadow the
# top-level modules of the same name and the app dies on the first import. Drop
# it: everything below imports the package by name.
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _PKG_DIR]

import spacy
import streamlit as st

from importlib import resources
from spacy.tokens import Token

from mordecai3 import Geoparser
from mordecai3.mordecai_utilities import spacy_doc_setup

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
def load_geo(_nlp):
    # No model_path: use the packaged default checkpoint and its config sidecar.
    # The spaCy pipeline is handed in so the app and the geoparser share one
    # transformer instead of loading en_core_web_trf twice.
    geo = Geoparser(geo_asset_path=resources.files("mordecai3") / "assets",
                 hosts=["localhost"],
                 nlp=_nlp,
                 debug=False,
                 trim=None)
    return geo


st.title('Mordecai geoparsing (v3)')
nlp = load_nlp()
geo = load_geo(nlp)

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
