# Development notes

- This assumes you are using `uv`. 

Running `pytest` without a reachable ES instance will run only the most basic tests. 



## Fixtures for writing tests

The big thing here is data availability. Two helper fixtures are automatically setup: `all_data_required`, `test_data_required`. 

To require a specific level of data availability at the test level, do like this:

```python
def test_big(all_data_required):
    pass

def test_small(test_data_required):
    pass
```

To do the same at the module level, which makes more sense in some cases, you can directly use another fixture that returns the data extent, `geonames_data_extent`.

```python
@pytest.fixture(scope='module', autouse=True)
def check_data_extent(geonames_data_extent):
    extent = geonames_data_extent[0]
    
    if extent < DataExtent.ALL:
        pytest.skip(
            f"Geonames data not available (extent: {extent})",
            allow_module_level=True
        )
```