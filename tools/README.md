# Setup ES test data

The `data/` folder here contains a small amount of data that is meant to be 
loaded into a running Elasticsearch instance for testing purposes. But it can
also be used for a simple hello work check.

Assuming you cloned this repo, start docker:

```
docker compose up -d
```

Install `textacy`, which is required for the data loading:

```
uv sync --group es-data

# or, with pip:
# pip install textacy
```

Then, on linux or macOS:

```bash
bash tools/load-es-test-data.sh
```

If on Windows 11, use the powershell version of the loader script:

```powershell
.\tools\load-es-test-data.ps1
```

This only needs to be done once locally---the ES container uses a mapped volume
which will keep the data at `test/es_data/geonames_index` (which is why that 
folder is on git). Thus stopping and restarting the container will pick up the 
already existing index. 

`download-raw-est-test-data.sh` downloads the raw test data `data/` folder here. 
There shouldn't be a need to run this unless we want to change the ES test data. 