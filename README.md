# Mordecai v3

Mordecai3 is a new geoparser that replaces the earlier [Mordecai](https://github.com/openeventdata/mordecai) geoparser. 

## Install and Requirements

To install the libraries required for Mordecai, `pip install` the list of required libraries:

```
pip install -r requirements.txt
```

Then run the following command to download the NLP model used to identify place names:

```
python -m spacy download en_core_web_trf
```

The event-location linking step requires other models to be downloaded from https://huggingface.co/. These will be automatically downloaded the first time the program is run (if it's run on an internet-connected machine) or can be downloaded first by running `python roberta_qa.py`.

Finally, Mordecai3 requires a local instance of Elasticsearch with a Geonames index. Instructions for setting up the index are available here: https://github.com/openeventdata/es-geonames

Once built, the index can be started like this:

```
docker run -d -p 127.0.0.1:9200:9200 -e "discovery.type=single-node" -v $PWD/geonames_index/:/usr/share/elasticsearch/data elasticsearch:7.10.1
```

## Usage

```
>>> from mordecai3 import Geoparser
>>> geo = Geoparser(model_path="mordecai3/mordecai_2023-03-23.pt", 
                 geo_asset_path="mordecai3/assets/")
>>> out = geo.geoparse_doc("I visited Alexanderplatz in Berlin.")
{'doc_text': 'I visited Alexanderplatz in Berlin.',
 'event_location_raw': '',
 'geolocated_ents': [{'admin1_code': '16',
                      'admin1_name': 'Berlin',
                      'admin2_code': '00',
                      'admin2_name': '',
                      'city_id': '',
                      'city_name': '',
                      'country_code3': 'DEU',
                      'end_char': 24,
                      'feature_class': 'S',
                      'feature_code': 'SQR',
                      'geonameid': '6944049',
                      'lat': 52.5225,
                      'lon': 13.415,
                      'name': 'Alexanderplatz',
                      'score': 1.0,
                      'search_name': 'Alexanderplatz',
                      'start_char': 10},
                     {'admin1_code': '16',
                      'admin1_name': 'Berlin',
                      'admin2_code': '00',
                      'admin2_name': '',
                      'city_id': '2950159',
                      'city_name': 'Berlin',
                      'country_code3': 'DEU',
                      'end_char': 34,
                      'feature_class': 'P',
                      'feature_code': 'PPLC',
                      'geonameid': '2950159',
                      'lat': 52.52437,
                      'lon': 13.41053,
                      'name': 'Berlin',
                      'score': 1.0,
                      'search_name': 'Berlin',
                      'start_char': 28}]}
```

## Acknowledgements

This work was sponsored by the Political Instability Task Force (PITF). The PITF is funded by the Central Intelligence Agency. The views expressed in this here are the authors' alone and do not represent the views of the US Government.
