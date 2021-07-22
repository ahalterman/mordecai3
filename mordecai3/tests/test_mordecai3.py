import pytest
from .. import elastic_utilities as es_utils

def test_statement_event_loc(geo):
    text = "Speaking from Berlin, President Obama expressed his hope for a peaceful resolution to the fighting in Homs and Aleppo."
    #text = "President Obama expressed his hope for a peaceful resolution to the fighting."
    icews_cat = "Make statement"
    out = geo.geoparse_doc(text, icews_cat) 
    assert out['event_location'] == 'Berlin'

def test_fight_event_loc(geo):
    text = "Speaking from Berlin, President Obama expressed his hope for a peaceful resolution to the fighting in Homs and Aleppo."
    icews_cat = "Use conventional military force"
    out = geo.geoparse_doc(text, icews_cat) 
    assert out['event_location'] == 'Homs and Aleppo'

def test_no_event_given(geo):
    text = "Speaking from Berlin, President Obama expressed his hope for a peaceful resolution to the fighting in Homs and Aleppo."
    out = geo.geoparse_doc(text) 
    assert out['event_location'] == ''

def test_no_locs(geo):
    text = "President Obama expressed his hope for a peaceful resolution to the fighting."
    icews_cat = "Make statement"
    out = geo.geoparse_doc(text, icews_cat) 
    assert out['geolocated_ents'] == []

def test_berlin(geo):
    text = "Speaking from Berlin, President Obama expressed his hope for a peaceful resolution to the fighting in Homs and Aleppo Governorates."
    out = geo.geoparse_doc(text) 
    assert out['geolocated_ents'][0]['geonameid'] == "2950159" # Berlin

def test_governorates(geo):
    text = "Speaking from Berlin, President Obama expressed his hope for a peaceful resolution to the fighting in Homs and Aleppo Governorates."
    out = geo.geoparse_doc(text) 
    assert out['geolocated_ents'][1]['geonameid'] == "169575" # Homs (governorate)
    assert out['geolocated_ents'][2]['geonameid'] == "170062" # Aleppo  (governorate)
    

def test_syria_cities(geo):
    text = "Speaking from Berlin, President Obama expressed his hope for a peaceful resolution to the fighting in the cities of Homs and Aleppo."
    out = geo.geoparse_doc(text) 
    assert out['geolocated_ents'][1]['geonameid'] == "169577" # Homs (city)
    assert out['geolocated_ents'][2]['geonameid'] == "170063" # Aleppo (city)

@pytest.mark.skip(reason="messes up on capital-D District")
def test_district_upper_term(geo):
    text = "Afghanistan: Southern Radio, Television Highlights 22 February 2021. He added: 'Ten Taliban, including four Pakistani Nationals, were killed in clashes between the commandos and Taliban in Arghistan District on the night of 21 February."
    out = geo.geoparse_doc(text)
    print(out)
    assert out['geolocated_ents'][0]['placename'] == "Afghanistan" 
    assert out['geolocated_ents'][1]['feature_code'] == "ADM2"
    assert out['geolocated_ents'][1]['geonameid'] == "7053299" 

def test_district_lower_term(geo):
    text = "Afghanistan: Southern Radio, Television Highlights 22 February 2021. He added: 'Ten Taliban, including four Pakistani Nationals, were killed in clashes between the commandos and Taliban in Arghistan district on the night of 21 February."
    out = geo.geoparse_doc(text) 
    assert out['geolocated_ents'][0]['placename'] == "Afghanistan" 
    assert out['geolocated_ents'][1]['feature_code'] == "ADM2"
    assert out['geolocated_ents'][1]['geonameid'] == "7053299" 

def test_miss_oxford(geo):
    text = "Ole Miss is located in Oxford."
    out = geo.geoparse_doc(text) 
    assert out['geolocated_ents'][0]['geonameid'] == "4449414" 
    assert out['geolocated_ents'][1]['geonameid'] == "4440076" 

def test_uk_oxford(geo):
    text = "Oxford University, in the town of Oxford, is the best British university."
    out = geo.geoparse_doc(text) 
    assert out['geolocated_ents'][0]['geonameid'] == "2640729" 

######

def test_adm1_count(geo):
    out = [{"es_choices":[
                 {"admin1_name": "MA"},
                 {"admin1_name": "England"}]},
            {"es_choices":[
                {"admin1_name": "MA"}]},
            {"es_choices":[{"admin1_name": "MA"}]}]
    adm1_counts = es_utils.make_admin1_counts(out) 
    assert adm1_counts['MA'] == 1.0
    assert adm1_counts['England'] == float(1/3)
