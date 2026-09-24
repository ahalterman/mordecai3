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
