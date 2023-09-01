import pytest

from .. import elastic_utilities as es_utils
from .. import geoparse


def test_statement_event_loc(geo):
    text = "Speaking from Berlin, President Obama expressed his hope for a peaceful resolution to the fighting in Homs and Aleppo."
    #text = "President Obama expressed his hope for a peaceful resolution to the fighting."
    icews_cat = "Make statement"
    out = geo.geoparse_doc(text, icews_cat) 
    assert out['event_location_raw'] == 'Berlin'

def test_fight_event_loc(geo):
    text = "Speaking from Berlin, President Obama expressed his hope for a peaceful resolution to the fighting in Homs and Aleppo."
    icews_cat = "Use conventional military force"
    out = geo.geoparse_doc(text, icews_cat) 
    assert out['event_location_raw'] == 'Homs and Aleppo'

def test_no_event_given(geo):
    text = "Speaking from Berlin, President Obama expressed his hope for a peaceful resolution to the fighting in Homs and Aleppo."
    out = geo.geoparse_doc(text) 
    assert out['event_location_raw'] == ''

def test_no_locs(geo):
    text = "President Obama expressed his hope for a peaceful resolution to the fighting."
    icews_cat = "Make statement"
    out = geo.geoparse_doc(text, icews_cat) 
    assert out['geolocated_ents'] == []

def test_three_locs(geo):
    text = "Speaking from Berlin, President Obama expressed his hope for a peaceful resolution to the fighting in Homs and Aleppo Governorates."
    out = geo.geoparse_doc(text) 
    assert out['geolocated_ents'][0]['geonameid'] == "2950159" # Berlin
    assert out['geolocated_ents'][1]['geonameid'] == "169577" # Homs (city)
    assert out['geolocated_ents'][2]['geonameid'] == "170063" # Aleppo (city)

def test_governorates(geo):
    text = "Speaking from Berlin, President Obama expressed his hope for a peaceful resolution to the fighting in Homs and Aleppo Governorates."
    out = geo.geoparse_doc(text) 
    assert out['geolocated_ents'][1]['geonameid'] == "169575" # Homs (governorate)
    assert out['geolocated_ents'][2]['geonameid'] == "170062" # Aleppo  (governorate)
    

@pytest.mark.skip(reason="messes up on capital-D District")
def test_district_upper_term(geo):
    text = "Afghanistan: Southern Radio, Television Highlights 22 February 2021. He added: 'Ten Taliban, including four Pakistani Nationals, were killed in clashes between the commandos and Taliban in Arghistan District on the night of 21 February."
    out = geo.geoparse_doc(text)
    print(out)
    assert out['geolocated_ents'][0]['search_name'] == "Afghanistan" 
    assert out['geolocated_ents'][1]['feature_code'] == "ADM2"
    assert out['geolocated_ents'][1]['geonameid'] == "7053299"  # Arghistan district

def test_district_lower_term(geo):
    """Same as above, but lower case district"""
    text = "Afghanistan: Southern Radio, Television Highlights 22 February 2021. He added: 'Ten Taliban, including four Pakistani Nationals, were killed in clashes between the commandos and Taliban in Arghistan district on the night of 21 February."
    out = geo.geoparse_doc(text) 
    assert out['geolocated_ents'][0]['search_name'] == "Afghanistan" 
    assert out['geolocated_ents'][1]['feature_code'] == "ADM2"
    assert out['geolocated_ents'][1]['geonameid'] == "7053299" 

def test_miss_oxford(geo):
    text = "Ole Miss is located in Oxford."
    out = geo.geoparse_doc(text) 
    assert out['geolocated_ents'][0]['geonameid'] == "4449414" 
    assert out['geolocated_ents'][0]['admin1_name'] == "Mississippi" 
    assert out['geolocated_ents'][1]['geonameid'] == "4440076" 

def test_uk_oxford(geo):
    text = "Oxford University, in the town of Oxford, is the best British university."
    out = geo.geoparse_doc(text) 
    assert out['geolocated_ents'][0]['geonameid'] == "2640729" 

def test_uk_oxford2(geo):
    text = "Oxford is home to Oxford University, one of the best universities in the world."
    out = geo.geoparse_doc(text) 
    assert out['geolocated_ents'][0]['geonameid'] == "2640729" 

def test_multi_sent(geo):
    text = """Gangster Kulveer Singh and his accomplice, Chamkaur Singh, were shot dead at Naruana village in Bathinda district on Wednesday morning.  Police said the two were shot dead at Singh's house at his native village by another accomplice, Manpreet Singh Manna, who also sustained a bullet injury and was undergoing treatment at the Bathinda Civil Hospital."""
    out = geo.geoparse_doc(text) 

def test_prague(geo):
    text = "A group of Czech settlers in Oklahoma named their new town Prague."
    out = geo.geoparse_doc(text) 
    assert out['geolocated_ents'][0]['feature_code'] == "ADM1"
    assert out['geolocated_ents'][1]['admin1_name'] == "Oklahoma"

    text = "Barack Obama gave a speech on nuclear weapons in Prague."
    out = geo.geoparse_doc(text) 
    assert out['geolocated_ents'][0]['feature_code'] == "PPLC"
    assert out['geolocated_ents'][0]['country_code3'] == "CZE"


def test_pragues(geo):
    out = geo.geoparse_doc("I visted family in Prague.")
    assert out['geolocated_ents'][0]['geonameid'] == "3067696"
    assert out['geolocated_ents'][0]['country_code3'] == "CZE"
    out = geo.geoparse_doc("I visted family in Prague, Oklahoma.")
    assert out['geolocated_ents'][0]['geonameid'] == "4548393"
    assert out['geolocated_ents'][0]['admin1_name'] == "Oklahoma"

def test_double_event(geo):
    """
    Make sure it's picking the latter location when two of the same event type are
    reported in different sentences.
    """
    text = """A group of bandits were arrested in Nigeria earlier this month."""
    out = geo.geoparse_doc(text, "arrest")
    assert out['event_location_raw'] == "Nigeria" 
    text = """A group of bandits were arrested in Nigeria earlier this month. "We are working together to see what we can do. We don't know what the bandits will say. "They may call for a ransom and will go for negotiation. We have agreed with parents to supervise in prayers and look up to God in this matter."
Jangadi also said the management of the school had informed the government about the incident adding that the government had assured that everything will be done to rescue the students. Later on, police arrested a group of rebels in Liberia."""
    out = geo.geoparse_doc(text, "arrest")
    assert out['event_location_raw'] == "Liberia" 


def test_event_loc2(geo):
    text = """'Many Injured As Urchins Engage Security Men in Shootout in Lagos.

Many persons were said to have been injured yesterday in a shootout between urchins popularly known as "Area Boys" and Joint Security Task Force Team at Tin Can Island, Lagos.

The crisis started when the security team led by the police was deployed to dislodge the miscreants in the area."""
    out = geo.geoparse_doc(text, "make statement")
    assert out['event_location_raw'] == "Tin Can Island"
    assert out['geolocated_ents'][0]['geonameid'] == "2332459"
    assert out['geolocated_ents'][1]['geonameid'] == "2566635"
    assert out['geolocated_ents'][2]['geonameid'] == "2332459"


def test_index_error(geo):
    # adding the event category induced an index error
    text = """Ukraine Reform Conference opens in Vilnius

VILNIUS, Jul 07, BNS – Lithuanian and Ukrainian President Gitanas Nauseda and Volodymyr Zelensky will open the Ukraine Reform Conference in Vilnius on Wednesday.\nInternational partners and Ukraine's representatives will discuss the country's reform achievements and challenges, as well as confirm the international community's support for Ukraine's sovereignty, territorial integrity and the reform process.\nPlans for Ukraine's European integrations up to 2030 should also be defined.\nDuring the event, the presidents will turn to the international community, seeking its attention and support for Ukraine's reforms on its path towards the European Union and NATO.\nThis year's conference will also discuss ways to bolster democratic institutions, the rule of law, fight against corruption, social and economic development issues.\nThe conference will take place two days and will be attended by Ukrainian Prime Minister Denys Shmyhal, other politicians and officials, experts, European Commissioner for Neighborhood and Enlargement Oliver Varhelyi, Matti Maasik, head of the EU Delegation to Ukraine, representatives of the US administration, NATO, etc."""
    out = geo.geoparse_doc(text)
    out = geo.geoparse_doc(text, "express intent to meet")


def test_geneva(geo):
    text = "On June 16, Russian President Vladimir Putin and his counterpart Joe Biden held talks in Geneva."
    out = geo.geoparse_doc(text)
    assert out['geolocated_ents'][-1]['country_code3'] == "CHE" 

def test_geneva_il(geo):
    text = "On June 16, Russian President Vladimir Putin and his counterpart Joe Biden held talks in Geneva, Illinois."
    out = geo.geoparse_doc(text)
    assert out['geolocated_ents'][0]['geonameid'] == "4893591"


 


###### Testing specific components #####

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


def check_for_accent_stripping(geo):
    """
    This will only work with Geonames indicies built after 2021-08-08 with
    the new option `expand_ascii` enabled (which it is by default). 
    """
    q = {"multi_match": {"query": "Qārat Muşāri‘",
                             "fields": ['name', 'asciiname', 'alternativenames'],
                             "type" : "phrase"}}
    res = geo.conn.query(q).execute()
    out = res['hits']['hits'][0].to_dict()['_source']
    stripped = "Qarat Musari"
    assert stripped in out['alternativenames']


def test_rel(geo):
    doc = geo.nlp("I visited Paris, France.")
    assert geoparse.guess_in_rel(doc.ents[0]) == "France"
    assert geoparse.guess_in_rel([doc.ents[0][0]]) == "France"

    doc = geo.nlp("I visited Paris, Berlin, and Munich.")
    assert geoparse.guess_in_rel(doc.ents[0]) == ""
    assert geoparse.guess_in_rel([doc.ents[0][0]]) == ""

def test_adm1_country_lookup(geo):
    res = es_utils.get_adm1_country_entry("Maine", None, geo.conn)
    assert res['geonameid'] == '4971068'
    res = es_utils.get_adm1_country_entry("Maine", "USA", geo.conn)
    assert res['geonameid'] == '4971068'
    res = es_utils.get_country_by_name("Cuba", geo.conn)
    assert res['feature_code'] == 'PCLI'
    assert res['geonameid'] == '3562981'
    res = es_utils.get_country_by_name("Atlantis", geo.conn)
    assert res is None
    res = es_utils.get_country_by_name("Syria", geo.conn)
    assert res['country_code3'] == "SYR"
    assert res['feature_code'] == "PCLI"
