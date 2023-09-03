# Mordecai v3

Mordecai3 is a new geoparser that replaces the earlier [Mordecai](https://github.com/openeventdata/mordecai) geoparser. It uses spaCy to identify place names in text, retrieves candidate geolocations from the Geonames gazetteer running in a local Elasticsearch index, and ranks the candidate results using a neural model trained on around 6,000 gold standard training examples.

## Usage

```pycon
>>> from mordecai3 import Geoparser
>>> geo = Geoparser()
>>> geo.geoparse_doc("I visited Alexanderplatz in Berlin.")
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

## Installation and Requirements

To install Mordecai3, run

```bash
pip install mordecai3
```

The library has two external dependencies that you'll need to set up.

First, run following command to download the spaCy model used to identify place names and to compute the tensors used in the ranking model.

```bash
python -m spacy download en_core_web_trf
```

Second, Mordecai3 requires a local instance of Elasticsearch with a Geonames index. Instructions for setting up the index are available here: https://github.com/openeventdata/es-geonames

Once built, the index can be started like this:

```bash
docker run -d -p 127.0.0.1:9200:9200 -e "discovery.type=single-node" -v $PWD/geonames_index/:/usr/share/elasticsearch/data elasticsearch:7.10.1
```

If you're doing event geoparsing, that step requires other models to be downloaded from https://huggingface.co/. These will be automatically downloaded the first time the program is run (if it's 

## Details and Citation

More details on the model and its accuracy are available here: https://arxiv.org/abs/2303.13675

If you use Mordecai 3, please cite:

```bibtex
@article{halterman2023mordecai,
      title={Mordecai 3: A Neural Geoparser and Event Geocoder}, 
      author={Andrew Halterman},
      year={2023},
      journal={arXiv preprint arXiv:2303.13675}
}
```

## Acknowledgements

This work was sponsored by the Political Instability Task Force (PITF). The PITF is funded by the Central Intelligence Agency. The views expressed in this here are the authors' alone and do not represent the views of the US Government.
