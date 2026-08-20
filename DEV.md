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
python train.py train --data-dir ../raw_data --epochs 30 --mix-dim 512 \
    --source-limits "WikiDocs=6000"
```

The default source list uses `wiki_docs`, the document-level wiki corpus built
by `tools/prepare_wiki.py`; the older sentence-level `wiki` source is still
selectable but costs ~0.005 exact match (`WIKI_TRAINING_DATA.md`). Build the
corpus once with:

```bash
python prepare_wiki.py ../raw_data/wiki ../raw_data/wiki/wiki_docs.jsonl
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
`--seed` seeds torch, numpy and python (the `seed` in the wandb config used to
be recorded and never applied), and `--metrics-out FILE` dumps the final
metrics, a mean over the last five epochs, and the full per-epoch history as
JSON. Both exist so two runs can be compared; see "Comparing runs" below.

`--source-limits "WikiDocs=6000"` caps how many training examples a source
contributes. The cap is applied *after* the train/test split, so a capped run
and an uncapped one are still scored on the same held-out set.

### Comparing runs

A single run is not evidence. Two runs of identical data and hyperparameters
differ by ~0.01 exact match between seeds, which is larger than most data
changes are worth; the numbers below are five seeds each, compared pairwise by
seed. Note also that the models are still improving at epoch 30 -- the default
is a budget, not convergence.

The held-out sets for Prodigy, TR, LGL, GWN and Synth are the last 30% of each
source and do not move when the wiki corpus changes, so they are what a change
to the wiki data should be judged on. The wiki set's own held-out accuracy is
not a useful target: it mostly measures how wiki-like the model has become.

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

### Sharding

A corpus is ~1.2 KB of DocBin per character of text, because every token
carries a 768-float tensor. `nlp-docs --shard-size N` flushes every N documents
to `source_x.000.spacy`, `source_x.001.spacy`, ...; `add-es` then processes one
shard at a time and writes a pickle per shard, and `load_es_data` concatenates
them in filename order. Nothing else changes: a source with no shards keeps the
single-file names, and the concatenated order is the order the documents were
read, so the positional train/test split still cuts where it used to.

Use it for anything over a few thousand documents. Two limits, both real:

- **`DocBin.to_disk` peaks at ~4x the shard's size in RAM** while it builds the
  serialized blob. A 2,500-document wiki shard is 5.7 GB on disk and spikes to
  21 GB resident. Shards much bigger than that will not fit in 64 GB.
- **`format_source` used to load every doc in the corpus at once** to hand them
  to a formatter. That is what the "doing more than 5000 at a time maxes out
  RAM" note in `data_to_docs` was about; it now holds one shard.

Shards are a cache, not an input, and a big corpus's shards are worth deleting
once `add-es` has written its pickles -- the pickles are ~5% of the size and are
what `train` actually reads. The full wiki corpus is the case in point:

```bash
rm raw_data/spacyed/source_wiki_docs_full.*.spacy   # frees 46 GB
```

Nothing in the training path needs those back. `wiki_docs_full.jsonl` is 49 MB,
and re-running `nlp-docs --sources wiki_docs_full --shard-size 2500` rebuilds
them in ~18 minutes if a future experiment wants them -- but per
`WIKI_TRAINING_DATA.md` there is no reason to expect one to.

### The wiki corpus

`tools/prepare_wiki.py` builds the document-level wiki training data that the
`wiki_docs` source reads. `WIKI_TRAINING_DATA.md` is the writeup: what the old
sentence-level file was doing wrong, and why adding more Wikipedia than this
does not improve accuracy. Read it before proposing more of it.

### Known rough edges

- `limit_types` still has no effect on the candidate set (see MERGE_NOTES.md).
- A whitespace-only document crashes the spaCy pipeline on GPU, inside the
  tagger, before any of our code runs. `data_to_docs` now raises on empty texts
  rather than letting cupy fail several minutes in, but `geoparse_batch` still
  needs the same guard.
- `tests/test_miss_oxford` and `tests/test_prague` fail, and did before this
  work -- they are geoparsing accuracy problems, not pipeline problems.
