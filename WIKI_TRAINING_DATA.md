# The Wikipedia training data: what was wrong, and what more of it buys

Written 2026-08-19, first experiment of the accuracy campaign. Short version:

- **The Wikipedia data in the training set was making the model worse**, by about
  half a point of exact match. That is the largest single effect measured here,
  and it was a data-preparation bug, not a property of Wikipedia.
- **Rebuilding it at the document level fixes that**, and turns wiki from a
  significant loss into a small, mostly-not-significant gain.
- **Adding more Wikipedia does not help.** Going from 2k to 67k linked mentions,
  and from 647 articles to 4,037, moves nothing outside seed noise -- and using
  all of it is the *worst* of the rebuilt options. There is no accuracy left in
  scraping more of it, so don't.

Everything below is five seeds per configuration, compared pairwise by seed, at
the standard recipe (`--epochs 30 --mix-dim 512`). Raw metrics are in
`_history` inside each `--metrics-out` JSON.

## How runs were compared

Two runs with identical data and hyperparameters differ by ~0.01 exact match
between seeds, which is larger than every effect in this document except the
first one. Anything measured from a single run here would have been noise.

- `train.py --seed` now seeds torch/numpy/python. The `seed` in the wandb config
  was recorded and never applied to anything.
- `train.py --metrics-out FILE` writes the final metrics, the mean over the last
  five epochs, and the full per-epoch history.
- Scores are the mean of the last five epochs, and the reported delta is the
  per-seed paired difference against the no-wiki baseline. `*` marks a delta
  larger than 2 standard errors.
- The held-out sets are the last 30% of Prodigy, TR, LGL, GWN and Synth. They do
  not change when the wiki corpus changes, which is what makes the runs
  comparable. The wiki corpus's *own* held-out accuracy is not a target: it
  measures how wiki-like the model has become, not whether it geolocates better.

## Finding 1: the existing wiki data was hurting

| | exact match | delta | acc@161km | delta |
|---|---|---|---|---|
| no wiki | 0.8795 | -- | 0.9148 | -- |
| `wiki_training_data_sents.jsonl` | 0.8742 | **-0.0053 ± 0.0009** * | 0.9115 | **-0.0033 ± 0.0008** * |

On the three published corpora alone (TR + LGL + GWN) it is worse: -0.0096 on
exact match, -0.0080 on acc@161. Every seed, every metric, same direction.

**Why.** `wiki_training_data_sents.jsonl` holds 2,160 rows -- 9% of the 23,668
linked mentions in those dumps -- and each row is a single *sentence* treated as
its own document with exactly one labelled place name in it. The model's
`adm1_count` and `country_count` features are built by
`_add_cross_entity_counts` from the other place names *in the same example*. In
a one-sentence example there are hardly any, so those features are near-zero on
every wiki example while being informative on every real article at inference
time. A tenth of the training set was teaching the model to distrust a feature
it depends on.

The scrape itself is fine: all offsets in all four raw files line up with their
text exactly, and 96% of the mentions already resolved to a spaCy entity. The
loss was in the framing, not the annotations.

## Finding 2: rebuilding at the document level fixes it

`tools/prepare_wiki.py` regroups the scrape into documents:

- merges the raw files (three different schemas, one per scrape) and groups
  every linked mention back onto the document it came from;
- strips the wiki markup still in the text -- `'''bold'''`, `''italic''`,
  `==headings==` -- remapping every offset, and dropping (rather than shifting)
  any span that overlapped deleted characters;
- splits documents over 4,000 characters on paragraph boundaries, which puts
  them in the same length range as the TR-News and LGL articles (median ~1.5k
  chars, p90 ~4k);
- shuffles by article title, so the positional train/test split in
  `load_es_data` neither straddles an article nor puts one whole category on one
  side.

`data_formatter_wiki_docs` in `train.py` then reads that. It differs from the
old `data_formatter_wiki` in two ways that matter:

- it uses the **gold span's own tokens**, not whichever spaCy entity happens to
  start inside the span. spaCy's boundaries disagree with the wiki links often
  enough to matter -- gold `Aleppo` against spaCy's `Aleppo Governorate`, gold
  `Raqqa Governorate` against spaCy's `Raqqa` -- and taking spaCy's span there
  attaches the gold id to a different place.
- it emits **every other GPE/LOC/FAC entity in the document unlabelled**, so
  they still feed the document's overlap features before being discarded, which
  is what `doc_to_ex_expanded` does at inference.

That takes the same three raw files from 2,100 usable training entities to
21,521, and the harm disappears:

| | n wiki train | exact match | delta | acc@161km | delta |
|---|---|---|---|---|---|
| no wiki | 0 | 0.8795 | -- | 0.9148 | -- |
| old sentence-level | 1,470 | 0.8742 | -0.0053 ± 0.0009 * | 0.9115 | -0.0033 ± 0.0008 * |
| doc-level, capped 2k | 2,000 | 0.8826 | +0.0031 ± 0.0020 | 0.9181 | +0.0033 ± 0.0014 * |
| doc-level, all | 15,065 | 0.8788 | -0.0006 ± 0.0025 | 0.9189 | +0.0041 ± 0.0013 * |

## Finding 3: more Wikipedia buys nothing

`geo_wiki/geo_wiki_parse.py` is the scraper. It resolves a linked page title
through the Wikipedia API to a Wikidata id, then reads property P1566 for a
geonames id, and it is heavily rate-limited (a 2s sleep per title). Re-running
it is not something to do casually -- but it does not need re-running, because
its output is already on disk: `wiki_big_sample.jsonl`, 57,161 mentions over
3,993 articles, every one already linked to geonames.

Merged and deduplicated with the three category files, that is **71,475 linked
mentions across 20,302 documents from 4,037 articles** (`wiki_docs_full`),
67,335 of which survive to training. 32x the mentions and 6x the articles of the
data the model was training on.

It does not help.

| wiki examples in training | exact match | delta vs no wiki | acc@161km | delta |
|---|---|---|---|---|
| 0 | 0.8795 | -- | 0.9148 | -- |
| 2,000 (of 67k, sampled) | 0.8817 | +0.0023 ± 0.0021 | 0.9185 | +0.0037 ± 0.0016 * |
| 6,000 (of 67k, sampled) | 0.8832 | +0.0037 ± 0.0025 | 0.9192 | +0.0044 ± 0.0021 * |
| 47,134 (all of it) | 0.8754 | -0.0040 ± 0.0031 | 0.9186 | +0.0038 ± 0.0017 * |

Three things to read off this:

1. **Volume is not the lever.** 2k, 6k, 21k and 47k are all within noise of each
   other and of the baseline on exact match. The best configurations are the
   *small* ones.
2. **Diversity is not the lever either.** 6k mentions drawn from across 4,037
   articles scores the same as 2k from 647 articles (+0.0037 vs +0.0031). This
   was the most plausible remaining reason to want more data, and it is not
   there. (Sampling method does matter a little for how the comparison reads --
   `--source-limits` originally truncated, which in document order means a few
   hundred articles about a few events, so it confounded "less data" with
   "narrower data". It now samples. The conclusion is the same either way.)
3. **Using all of it is actively worse**: -0.0040 exact match, -0.0059 on
   TR+LGL+GWN. At 47k of 53k training examples, wiki is 89% of the training set
   and the human-annotated corpora stop mattering.

The one consistent gain anywhere in this table is **acc@161km, and only on the
five-set average** (+0.004, significant for every doc-level configuration). It
vanishes when restricted to TR+LGL+GWN, so it is coming from Prodigy and Synth.
The plausible story is that Wikipedia teaches country- and region-level priors
-- get to roughly the right part of the world -- while discriminating between
two same-named places 50km apart comes from the human-annotated data, and
diluting that with wiki does not help.

## What to do

**Point the wiki source at the rebuilt data and cap it at a few thousand
examples.** It is free, it is at worst neutral, and it removes a measured
half-point loss that is in the model today:

```bash
cd tools
python prepare_wiki.py ../raw_data/wiki ../raw_data/wiki/wiki_docs.jsonl
python train.py nlp-docs ../raw_data --sources wiki_docs
python train.py add-es  ../raw_data --sources wiki_docs
python train.py train --data-dir ../raw_data --epochs 30 --mix-dim 512 \
    --dataset-names "Prodigy, TR, LGL, GWN, Synth, WikiDocs" \
    --source-limits "WikiDocs=6000"
```

**Do not scrape more Wikipedia.** The next accuracy work should go somewhere
else. The candidates this experiment surfaced, in rough order of promise:

- **The gap between acc@161 and exact match is where the errors are.** The model
  lands in the right region and picks the wrong entry: 0.919 vs 0.883. That gap,
  not more training documents, is the accuracy problem.
- **`limit_types` has never had any effect** on the candidate set (see
  `MERGE_NOTES.md`), so nothing has ever been tested with the feature-class
  filter actually applied.
- **1-3% of held-out entities do not have the right answer among their 500
  candidates at all** (TR 1.1%, LGL 2.8%, GWN 3.0%) -- an Elasticsearch recall
  ceiling no amount of model work can cross.
- **The models are still improving at epoch 30.** Every number here is at that
  budget; a longer-schedule comparison was not run. If a config is going to be
  re-tested, test it there first.

## Caveats

- All of this is at `--epochs 30 --mix-dim 512`. The training curves were still
  rising at epoch 30 for the baseline, so these are comparisons at a fixed
  budget, not at convergence.
- `tests/test_miss_oxford` and `tests/test_prague` still fail, as they did
  before this work and before the pipeline work.
- **The full corpus's spaCy shards can be deleted.** Given the conclusion,
  nothing needs them again:

  ```bash
  rm raw_data/spacyed/source_wiki_docs_full.*.spacy   # frees 46 GB
  ```

  Keep `raw_data/wiki/wiki_docs_full.jsonl` (49 MB) and the nine
  `es_formatted_wiki_docs_full.*.pkl` (2.4 GB) -- together they are enough to
  re-run any training comparison in this document without touching spaCy. To
  rebuild the shards anyway, `nlp-docs --sources wiki_docs_full --shard-size
  2500` takes ~18 minutes.
