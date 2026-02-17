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

In real production use, you'll want to use the batched processing function:

```
results = geo.geoparse_batch(
    texts,               # list of strings
    batch_size=32,       # spaCy transformer batch size
    chunk_size=200,      # docs per processing chunk (bounds memory)
    es_workers=4,        # thread pool size for ES lookups
    show_progress=True,  # tqdm progress bar
)
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

The current version of Mordecai3 includes a retrained model that slightly improves on the results reported in the paper.

```
┏━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━┳
┃            ┃        ┃             ┃            ┃     Correct ┃            ┃
┃            ┃        ┃             ┃    Correct ┃     Feature ┃    Correct ┃
┃ Dataset    ┃ Eval N ┃ Exact match ┃    Country ┃        Code ┃       ADM1 ┃
┡━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━╇
│ training   │   7337 │       90.9% │      99.1% │       95.4% │      93.7% │
│ set        │        │             │            │             │            │
│ prodigy    │    500 │       87.8% │      96.8% │       87.8% │      95.1% │
│ TR         │    274 │       84.3% │      97.8% │       89.6% │      88.1% │
│ LGL        │    967 │       79.4% │      97.9% │       87.3% │      82.5% │
│ GWN        │    474 │       90.1% │      97.4% │       91.0% │      95.6% │
│ GWN_compl… │   1564 │       92.0% │      98.5% │       93.4% │      97.2% │
│ Synth      │    300 │       93.3% │      96.9% │       96.1% │      94.9% │
│ Wiki       │    630 │       86.0% │      98.2% │       86.3% │      96.7% │
└────────────┴────────┴─────────────┴────────────┴─────────────┴────────────┴
```

```
┏━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃            ┃        ┃            ┃             ┃             ┃            ┃            ┃
┃            ┃        ┃ Mean Error ┃      Median ┃     Missing ┃      Total ┃            ┃
┃ Dataset    ┃ Eval N ┃       (km) ┃  Error (km) ┃     correct ┃    missing ┃ Acc @161km ┃
┡━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ training   │   7337 │      119.8 │         0.0 │       71.9% │      10.2% │       93.6 │
│ set        │        │            │             │             │            │            │
│ prodigy    │    500 │      273.5 │         0.0 │      100.0% │       6.8% │       95.9 │
│ TR         │    274 │      294.7 │         0.0 │       66.7% │       2.2% │       87.3 │
│ LGL        │    967 │      303.9 │         0.0 │       37.7% │       5.5% │       82.7 │
│ GWN        │    474 │      249.4 │         0.0 │       31.6% │       4.0% │       94.1 │
│ GWN_compl… │   1564 │      178.3 │         0.0 │       57.5% │       5.6% │       95.7 │
│ Synth      │    300 │      215.3 │         0.0 │       97.8% │      15.0% │       95.7 │
│ Wiki       │    630 │       23.8 │         0.0 │       54.5% │       3.5% │       98.0 │
└────────────┴────────┴────────────┴─────────────┴─────────────┴────────────┴────────────┘
```

## Acknowledgements

This work was sponsored by the Political Instability Task Force (PITF). The PITF is funded by the Central Intelligence Agency. The views expressed in this here are the authors' alone and do not represent the views of the US Government.
