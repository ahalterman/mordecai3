
import streamlit as st
import spacy
import base64
import pandas as pd
import jsonlines

HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_model():
    """Load a spaCy model."""
    model_load_state = st.info(f"Loading model...(may take up to 60 seconds)")
    from mordecai import Geoparser
    geo = Geoparser(country_threshold=0.2)
    model_load_state.empty()
    return geo

def visualize(geo):
    #default_text = "During 24 hours of fighting in the deserts of Aleppo, Hama, and Raqqa provinces, 21 IS fighters and 13 government soldiers were killed.\n\nAirstrikes on the outskirts of Abu-Kamal, near the Syrian border with Iraq, left seven Iran-backed Iraqi militiamen dead. Israel was thought to be behind the strikes. Meanwhile, a failed attack by Turkish-backed Free Syrian Army (TFSA) rebels against the Kurdish-led Syrian Democratic Forces (SDF) west of Tel Abyad left 14 TFSA fighters dead."
    default_text = "Afghanistan's major population centers are all government-held, with capital city Kabul especially well-fortified, though none are immune to occasional attacks by Taliban operatives. And though the conflict sometimes seems to engulf the whole country, the provinces of Panjshir, Bamyan, and Nimroz stand out as being mostly free of Taliban influence."
    text = st.text_area("Text to geoparse", default_text)        
    with jsonlines.open("all_input.jsonl", "a") as f:
        f.write({"text": text})
    doc = geo.nlp(text)
    output = geo.geoparse(doc)

    st.subheader("Geolocation results")
    html = spacy.displacy.render(doc, style="ent", options={"ents": labels})
    html = html.replace("\n", " ")
    st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)
    resolved_places = []
    for i in output:
        if 'geo' in i.keys():
            f_code = i['geo']['feature_class']
            f_class = i['geo']['feature_code']
            if f_class == "PCLI":
                fc_txt = "a country: "
            elif f_class == "ADM1":
                fc_txt = "a province/state/governorate in"
            elif f_class == "ADM2":
                fc_txt = "a district/county in"
            elif f_code == "A":
                fc_txt = "a region in "
            elif f_class == "PPLC":
                fc_txt = "the capital of"
            elif f_class == "PPLX":
                fc_txt = "a neighborhood in"
            elif f_code == "P":
                fc_txt = "a city/town in"
            else:
                fc_txt = "an other type in"
            resolved = f"• \"{i['word']}\" --> \"{i['geo']['place_name']}\", {fc_txt} {i['country_predicted']}"
        else:
            print(i)
            resolved = f"• \"{i['word']}\" --> [none] in {i['country_predicted']}"

        resolved_places.append(resolved)

        #except KeyError as e:
        #    with jsonlines.open("error_text.jsonl", "a") as f:
        #        f.write({"text": text,
        #        "exception": str(e)})

    buttons = []
    for n, i in enumerate(resolved_places):
        st.markdown(i)
        buttons.append(st.button("Incorrect?", key=n))
        
    for n, button in enumerate(buttons):
        if button:
            st.info(f"Thanks for your feedback! {resolved_places[n]} has been marked as incorrect.")
            incorrect = output[n]
            incorrect['text'] = text
            incorrect['country_conf'] = float(incorrect['country_conf'])
            print(incorrect)
            with jsonlines.open("resolution_error.jsonl", "a") as f:
                f.write(incorrect)

    st.subheader("Named Entity Recognition")
    st.markdown("Mordecai geolocates placenames identified with spaCy's NER. These results are from spaCy 2's `en_core_web_lg`.")
    labels = ["GPE", "LOC"]
    html = spacy.displacy.render(doc, style="ent", options={"ents": labels})
    # Newlines seem to mess with the rendering
    html = html.replace("\n", " ")
    st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)
    ner = st.button("Incorrect NER?", key="ner")
    if ner:
        st.info(f"Thanks for your feedback!")
        with jsonlines.open("ner_error.jsonl", "a") as f:
            to_log = {"text": text,
                      "ents": [{"start": i.start, 
                                "end": i.end, 
                                "start_char": i.start_char, 
                                "text": i.text, 
                                "label": i.label_} for i in doc.ents]
            }
            f.write(to_log)
    #st.text(output)
    st.subheader("Map")
    pretty_out = [{"lat": float(i['geo']['lat']), "lon": float(i['geo']['lon']), "name": i['geo']['place_name']} for i in output if 'geo' in i.keys()]
    df = pd.DataFrame(pretty_out)
    st.map(df)
    st.subheader("Raw JSON result")
    st.json(output)


st.title('Mordecai geoparsing (v2)')
st.markdown("Mordecai (https://github.com/openeventdata/mordecai) is a text geoparser that extracts place names from text and resolves them to their geographic coordinates.")
st.markdown("This page lets you view the output from Mordecai's v2 model. I'm planning new work on a rewritten v3 and your feedback here will be very helpful. If any of the results below seem incorrect, please use the button below to mark them! You can also reach me at ahalterman0@gmail.com or open an issue on the Mordecai Github page.")
st.markdown("Note: The app will log the results you mark as incorrect but does not collect any identifying information.") 
geo = load_model()
visualize(geo)