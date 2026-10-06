# AGENTS.md

A briefing for coding agents (and people) new to Mordecai 3. `README.md` is the
user guide; `DEV.md` covers training, tests, and the environment in depth. This
file is the map and the list of traps.

## What it does

Mordecai 3 turns text into GeoNames entries:

1. **Spans.** spaCy `en_core_web_trf` finds place names (GPE/LOC/FAC entities),
   and a custom `token_tensors` pipe keeps the transformer's token vectors. An
   optional learned span head (`span_detector="gold"`, `mordecai3/span_head.py`)
   replaces spaCy's NER on the same tensors.
2. **Candidates.** Each span is looked up in a local Elasticsearch `geonames`
   index (`mordecai3/geonames.py`), batched through `_msearch`. Abbreviations
   are normalized first (`place_aliases.py`: "Calif." -> "California").
3. **Features.** Each candidate gets the original hand-built features plus 26
   enriched ones (`candidate_features.py`) and, optionally, outlet features
   (`outlet_features.py`: where the story was published).
4. **Ranking.** A small PyTorch model (`torch_model.py`) scores the candidates
   against the context tensors; the top one, if it clears the threshold, is the
   answer. `geoparse.py` orchestrates all of it: `Geoparser.geoparse_doc()` and
   the faster `geoparse_batch()`.

Demonyms ("Syrian", "Turkish") are deliberately **not** returned as places.

## Layout

| path | what |
|---|---|
| `mordecai3/` | the library; `geoparse.py` is the entry point |
| `mordecai3/assets/` | checkpoints (`*.pt`) with config sidecars (`*.pt.json`), GeoNames lookup tables, the ES mapping |
| `mordecai3/cli.py`, `index_builder.py` | the `mordecai3` command: `index fetch/build/status`, `check`, `cite` |
| `tests/` | pytest; most tests need a running ES with the full index |
| `tools/` | training (`train.py`) and the data prep and evaluation it needs (see `DEV.md`) |
| `console/` | a FastAPI + d3 analyst demo; not part of the wheel (`console/README.md`) |
| `examples/` | batch and parallel processing scripts |

## Setting up

```bash
uv sync --extra console --extra gpu --group train --group dev   # everything
docker compose up -d        # ES on :9200, data in tests/es_data
bash tools/load-es-test-data.sh   # small test index (tools/README.md), or use the full one
mordecai3 check             # spaCy model, torch/CUDA, ES, index size and age
```

For real results you need the **full** index (13M+ documents): download the
prebuilt one with `mordecai3 index fetch`, or run `mordecai3 index build`.

## Diagnosing problems

Start with `mordecai3 check`. Then, by symptom:

- **`uv sync` removed something and a module is now missing.** `uv sync` installs
  exactly the extras and groups you name and prunes everything else. Always pass
  the full set (above). Same for `uv run --group ...`.
- **`uv sync` fails compiling `curated_tokenizers` (Cython error).** No cp313
  wheels exist, and the sdist doesn't build under Cython >= 3.1.
  `pyproject.toml` pins the build to `cython<3.1`; a stale clone predates that.
  Or use Python 3.12. `console/DEPLOY.md` has the details.
- **`mordecai3 index fetch` fails.** It lists each mirror it tried and why
  (network, HTTP status, checksum mismatch). A checksum mismatch usually means a
  truncated download or a re-uploaded archive: the SHA-256 is pinned in
  `index_builder.PREBUILT_SHA256` and must change together with the filename
  whenever a new index is published. `MORDECAI_INDEX_URL` adds a mirror.
- **Results are poor or strange, with no errors.** Check the index size with
  `mordecai3 index status`. The test index from `tools/load-es-test-data.sh` has a few
  thousand rows, so everything outside the test fixtures resolves badly or not
  at all. `check` flags any index under 10M documents.
- **An index load dies with `429 ... flood-stage watermark`.** The disk is over
  95% full and ES made the index read-only. `mordecai3 index build` relaxes the
  watermarks before it creates the index. For a hand-built index, set
  `cluster.routing.allocation.disk.watermark.*` first.
- **A checkpoint loads but predicts nonsense, or you get "trained with N extra
  features".** The checkpoint's `.pt.json` sidecar tells `Geoparser` which
  feature blocks and model flags to use. A checkpoint copied without its sidecar
  silently mis-runs. Keep them together, or pass `feature_blocks=` explicitly.
- **It runs on the GPU when you didn't ask.** CUDA is used automatically when
  available. Pass `device="cpu"` to `Geoparser`.
- **Tests skip en masse.** `tests/conftest.py` skips whatever it can't reach:
  no ES, no `geonames` index, test-size vs full index, missing training pickles
  (`raw_data/`, which is not in git). Read the skip reasons with `pytest -rs`.

## Tests

```bash
uv run pytest -rs                      # library tests
uv run pytest console/test_console_ui.py   # console UI (needs playwright chromium)
```

Known model quirks are `xfail`, not bugs to fix by editing tests: Homs as city
vs governorate, Oxford UK vs Mississippi, Geneva IL, Prague OK. The ranker has
a strong prior for the most prominent place of a name, and in-text cues don't
always override it.

## Conventions

- Measure accuracy changes, never judge them from a single run: training seed
  noise is about 0.01 exact match. Compare five paired seeds (see `DEV.md`,
  "Comparing runs").
- Training must stay byte-identical at fixed seed, and inference features must
  match training features exactly (`tests/test_feature_parity.py`).
- Comments in `mordecai3/` and `tests/` sometimes cite `experiments/...`
  reports. Those are the research record from the 2026 accuracy campaigns, kept
  outside this repo; the citation tells you where a number or decision came from.
- A checkpoint's candidate features (e.g. `alt_name_length`) come from the
  GeoNames index, so serve a checkpoint with the dump it was trained on. A new
  dump means rebuilding the training pickles (`train.py add-es`, then
  `enrich_pickles.py`) and retraining; see `DEV.md`.
- Commit messages: short, lowercase, imperative subject; terse bullets if needed.
