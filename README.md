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
    show_progress=True,  # tqdm progress bar
)
```

## Demos

`mordecai3-app` starts a small Streamlit page for poking at single documents.

`console/` is a fuller one: a single-screen analyst UI showing the document,
a map, the ranked gazetteer candidates behind each decision, and administrative
boundary polygons where the resolved place has one. `console/README.md` is the
writeup; `console/DEPLOY.md` stands it up on a fresh server end to end.

## Installation and Requirements

```bash
pip install mordecai3
python -m spacy download en_core_web_trf
```

Mordecai also needs Elasticsearch with a GeoNames index. The fastest route is
the prebuilt index that the model was trained and tested against (GeoNames dump
of 2026-09-24, 1.5 GB download, 2.2 GB unpacked):

```bash
mordecai3 index fetch            # download, verify checksum, unpack to ./geonames_index
docker run -d -p 127.0.0.1:9200:9200 -e "discovery.type=single-node" \
    -v $PWD/geonames_index/:/usr/share/elasticsearch/data elasticsearch:7.10.1
mordecai3 check                  # spaCy model, Elasticsearch, index size and age
```

If the download fails (or its checksum doesn't match), `fetch` offers to build
the index from GeoNames instead. To get another copy yourself, set
`MORDECAI_INDEX_URL` to a mirror or a `file://` path and run `fetch` again; the
checksum is still verified.

To build from the current GeoNames dump (about 30 minutes; start an empty
Elasticsearch with the same `docker run` first):

```bash
mordecai3 index build            # download GeoNames, create and load the index
mordecai3 index status           # document count and which dump it was built from
```

A fresh build uses today's GeoNames, not the dump the model was tested on, so a
few answers can differ from the prebuilt index. `index build` deletes and
recreates only the `geonames` index, so it is safe on a node that holds other
indices. Use `--es-url` (or `MORDECAI_ES_URL`) for a node that is not on
`localhost:9200`.

## Accuracy and speed

On six held-out evaluation sets (LGL, TR-News, GeoWebNews, the Prodigy news
annotations, Wikipedia, and synthetic sentences), given the place name, the
3.5 model picks the correct GeoNames entry more often than the previous one:

| | exact GeoNames match | within 161 km |
|---|---|---|
| 3.4 training recipe | 88.1% | 92.6% |
| **3.5 model** | **92.5%** | **96.7%** |

Macro average over the six sets, mean of five training seeds. The 3.5 model is
trained and evaluated against the GeoNames dump of 2026-09-24 (the prebuilt
index); the 3.4 recipe against the January 2024 dump.

**These numbers are conditional on the place name being found correctly.**
They measure the step that picks a GeoNames entry for a place name, starting
from the annotated place name. In real use, spaCy's named entity recognizer
finds place names first, and its misses (nested names like "Aleppo" inside
"Aleppo University", unusual spellings) are the main source of errors from raw
text. Better place-name detection is the focus of ongoing work; the opt-in
`span_detector="gold"` below is an early version of it.

Speed: `geoparse_batch` handles 60–110 documents/second on one RTX 4090.
Elasticsearch lookups, not the model, take most of that time.

### Confidence scores and thresholds

Each place's `score` is a calibrated probability that the chosen GeoNames entry
is correct: of the answers scored around 0.8, roughly 80% are right. (The model
is slightly under-confident, so it's usually a bit better than that.) Thresholds
tuned for 3.4's scores will throw away many correct answers in 3.5, so re-check
any cutoff you use.

Filtering the answers the 3.5 model returns on the held-out sets (pooled over
all six; the news sets alone lose a few points more correct answers at each
cutoff):

| keep `score >=` | answers kept | accuracy of kept | correct answers dropped | wrong answers dropped |
|---|---|---|---|---|
| (none) | 100% | 92.3% | 0% | 0% |
| 0.6 | 95% | 94.6% | 3% | 34% |
| **0.7** | **92%** | **95.9%** | **5%** | **51%** |
| 0.8 | 88% | 96.9% | 8% | 65% |
| 0.85 | 86% | 97.2% | 10% | 69% |
| 0.9 | 82% | 97.8% | 14% | 77% |

**0.7 is a sensible default.** Use a higher cutoff only if a wrong location
costs you much more than a missing one.

When the model decides none of its candidates is right, the result carries
`"no_match": True` and a `p_no_match` probability, with no `score` or GeoNames
fields, so check for that key before reading `score`. About 1% of mentions
come back this way, and the model's best guess on them would have been wrong
nearly 90% of the time. The accuracy numbers above don't apply a threshold. They
count every answer the model gives.

### What changed in 3.5

- A fresh GeoNames index (dump of 2026-09-24, 13.5M places), with the model
  retrained against it.
- A retrained ranker with 26 new candidate features (prominence, name match,
  context cues, sibling places, geography, name shape).
- `score` is now a calibrated probability, so re-check any cutoff you used
  with 3.4 (see "Confidence scores and thresholds" above).
- Abbreviated place names ("Calif.", "N.Y.") are normalized before lookup.
- Demonyms are no longer returned as places; the `accept_norp` argument is gone.
- `geoparse_batch()`, batched Elasticsearch queries, and automatic GPU use.
- Opt-in: a learned place-span detector (`span_detector="gold"`) and an
  outlet-aware ranker that uses where a story was published
  (`geoparse_doc(text, outlet="nytimes.com")`).
- The `mordecai3` command: `index fetch`, `index build`, `index status`, `check`, `cite`.

## Details and Citation

More details on the model and its accuracy are available here: https://arxiv.org/abs/2303.13675

If you use Mordecai 3, please cite (`mordecai3 cite` prints this). The first `Geoparser()` on a machine prints a one-time reminder to stderr; set `MORDECAI_NO_CITATION_NOTICE=1` to skip it:

```bibtex
@article{halterman2023mordecai,
  title={Mordecai 3: A neural geoparser and event geocoder},
  author={Halterman, Andrew},
  journal={arXiv preprint arXiv:2303.13675},
  year={2023}
}
```

The current version of Mordecai3 includes a retrained model that improves on the results reported in the paper (see "Accuracy and speed" above).

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
