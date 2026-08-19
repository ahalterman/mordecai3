# Development notes

- This assumes you are using `uv`. 

Running `pytest` without a reachable ES instance will run only the most basic tests. 



## Fixtures for writing tests

The big thing here is data availability. Rather than asking for a data extent and
then building your own objects, request the fixture for the object you need at the
level of data you need, and the test is skipped automatically if that data isn't
loaded:

| Fixture | Gives you | Skips unless |
|---|---|---|
| `es_client` | a raw `Elasticsearch` client | (never skips) |
| `geonames_service` | a `GeonamesService` | ES is up and the geonames index exists |
| `geonames_service_test_data` | a `GeonamesService` | the reduced test data is loaded |
| `geonames_service_all_data` | a `GeonamesService` | the full geonames data is loaded |
| `geoparser_test_data` | a `Geoparser` | the reduced test data is loaded |
| `geoparser_all_data` | a `Geoparser` | the full geonames data is loaded |

```python
def test_big(geoparser_all_data):
    res = geoparser_all_data.geoparse_doc("I visited Berlin, Germany.")
    ...

def test_small(geonames_service_test_data):
    res = geonames_service_test_data.get_country_by_name("Netherlands")
    ...
```

The `Geoparser` fixtures are session-scoped and share a single `GeonamesService`,
so the spaCy and torch models are only loaded once for the whole run.

If you need the data extent directly, `GeonamesService.determine_data_extent()`
returns a `DataExtent`, which is an `IntEnum` and so can be compared with `<`:

```python
from mordecai3.geonames import DataExtent

@pytest.fixture(scope='module', autouse=True)
def check_data_extent(geonames_service):
    if geonames_service.determine_data_extent() < DataExtent.ALL:
        pytest.skip("Geonames data not available", allow_module_level=True)
```

## Batch processing

`Geoparser.geoparse_batch()` is the entry point for processing many documents. It
batches the spaCy transformer pass, bundles the Elasticsearch lookups for a whole
chunk into a small number of `_msearch` requests, and pools all the entities in a
chunk into a single model forward pass. `geoparse_doc()` is a thin wrapper over
the same code path, so the two return identical results.

Repeated place names are cached per `GeonamesService` instance
(`_es_cache`/`_parent_cache`). `geoparse_batch()` clears the cache at the start of
each run; call `geo.geonames.clear_cache()` yourself if you need to drop it
between `geoparse_doc()` calls.

### A note on the ES lookups

Each `_msearch` sub-query is executed by Elasticsearch exactly as if it had been
sent on its own, so batching them changes throughput and nothing else --
`tests/test_msearch.py` asserts the batched path returns the same candidates,
in the same order, with the same features, as looping over `add_es_data`.

Two things to be aware of if you touch this code:

- A sub-query that fails comes back as a response with a status and *no* `hits`.
  Treating that as "no candidates" would be indistinguishable from a genuine
  miss, so `GeonamesService._msearch` raises `GeonamesQueryError` instead.
- Lookups are deduplicated by cache key *before* anything is sent. The old
  one-at-a-time path got this for free because the cache filled in as it went;
  when every lookup is planned up front, duplicates have to be collapsed
  explicitly or a batch re-queries every repeated name.

The `es_workers` argument is still accepted but ignored -- concurrency now
happens inside Elasticsearch rather than in a client-side thread pool.

## Training the model

Three commands, run in order, each caching its output to disk so the next one
can be re-run on its own. `raw_data/` holds both the corpora and the caches:

```bash
cd tools
python train.py nlp-docs  ../raw_data     # spaCy  -> raw_data/spacyed/*.spacy
python train.py add-es    ../raw_data     # ES     -> raw_data/pickled_es/*.pkl
python train.py train --data-dir ../raw_data --epochs 30 --mix-dim 512
```

Timings for the full 7,674-document corpus on a 4090 with local Elasticsearch:

| stage | time | output |
|---|---|---|
| `nlp-docs` | 44s | 3.4 GB |
| `add-es` | 47s | 536 MB |
| `train` (30 epochs) | 34s | one `.pt` |

Only `train` normally needs re-running. Re-run `add-es` when the candidate
query or the gazetteer features change, and `nlp-docs` when the spaCy pipeline
changes or a corpus is added -- neither checks whether its output is already
current, so it always redoes the whole corpus.

`train` requires wandb. Set `WANDB_MODE=offline` for runs you don't want logged.

### Things that make this fast, and are easy to lose

- **`cupy` must be installed or spaCy silently runs the transformer on CPU.**
  `spacy.prefer_gpu()` returns False rather than raising, and the only sign is
  an INFO line. This is worth ~5x on `nlp-docs` and ~2x on production
  geoparsing. It has to be `cupy-cuda12x<14`: thinc 8.3's `xp2torch` goes
  through the deprecated `cupy.ndarray.toDlpack()`, which cupy 14 no longer
  hands over in a form torch will accept ("invalid capsule").
- **`token_tensors` averages wordpiece vectors per token in one pass**
  (`_segment_means`), not with a slice-and-mean per token. On GPU the old
  version launched a kernel per token and cost more than the transformer
  forward pass it was reading from.
- **`nlp-docs` writes DocBins uncompressed** (`fast_docbin_io`). Docs carry a
  768-float tensor per token, so zlib spent 93s to save 10% of the file size.
  Files stay valid `.spacy` and load with an unmodified `DocBin.from_disk`.
- **`add-es` looks up a chunk of documents at a time**, so duplicate place
  names collapse across documents and the ES round trips are `_msearch`-batched.
- **`train` uses the GPU** and moves each batch there. The model is tiny, but
  it is ~50x the difference over 30 epochs.

### Known rough edges

- `limit_types` still has no effect on the candidate set (see MERGE_NOTES.md).
- A whitespace-only document crashes the spaCy pipeline on GPU, inside the
  tagger, before any of our code runs. Filter empty texts before `nlp.pipe`.
- `tests/test_miss_oxford` and `tests/test_prague` fail, and did before this
  work -- they are geoparsing accuracy problems, not pipeline problems.
