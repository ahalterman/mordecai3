# Serving fixes: shipping the measured NER-side and calibration wins

Second campaign, Phase 1. Written 2026-08-20, revised the same day with the
trim guard (§1a) and the harness alignment (§2e). Implements in
`mordecai3/geoparse.py` the changes measured in
`experiments/campaign2/end_to_end_report.md` (§5) and
`experiments/campaign2/calibration_report.md` (§2a, §8), and re-measures them
through the *serving* path rather than through the research harness.

Nothing here retrains anything. The §2b validation drives
`Geoparser.geoparse_batch` and scores its output with
`tools/end_to_end_eval.py`'s own corpus reader and scorer, so it does not
depend on the harness's own extraction code; `tools/end_to_end_eval.py` was
separately updated to share the serving label sets and trimmer, and its
`check_parity` assertion is green again (§2e).

## 1. What changed

**Default behavior changes in exactly one way: spans are trimmed.** Everything
else is either opt-in, or additive output fields.

| # | change | default | measured effect |
|---|---|---|---|
| 1 | **span trimming** — a leading `the/a/an` (or punctuation) and a trailing `'s` are dropped from every spaCy span, with `search_name`, `start_char`, `end_char` and the tensor all recomputed on the trimmed tokens, **except where the article is part of the gazetteer name** (§1a) | **ON** | +2.0 e2e EM, +2.6 det F1, *better* precision, slightly faster |
| 2a | `accept_norp` — resolve demonym spans ("Turkish", "Palestinian") | off | +6.3 e2e EM at unchanged det precision, −7% throughput |
| 2b | `nested_gazetteer_pass` — extract gazetteer-exact A/P toponyms nested inside ORG/FAC/EVENT/WORK_OF_ART/LAW/PRODUCT entities | off | +7.9 e2e EM, det precision 75.5 → 67.0, −19% throughput |
| 2c | `include_fac` — resolve FAC spans | **True** (current behavior) | setting False: +11.6 det precision, +43% throughput, −1.1 det recall |
| 3 | **calibrated confidence** — `score` is now the temperature-scaled probability of the chosen candidate with the reserved row in the denominator (`p_pred_full`), and every result carries `p_no_match` (`p_reserved`) and `no_match` | `temperature=0.874` | pooled ECE 0.019 → 0.012; `p_pred_full` is the best single wrong-answer score (AUROC 0.899) |
| 4 | **train/serve context fix** — `locs_tensor` is pooled over GPE/LOC, not GPE/LOC/EVENT_LOC/NORP | — | no measurable e2e effect (≤0.12 EM, see §2) |

New public surface in `mordecai3/geoparse.py`:

* `Geoparser(..., trim_spans=True, accept_norp=False, nested_gazetteer_pass=False, include_fac=True, temperature=0.874)`, all documented in the constructor docstring.
* `trim_span_tokens(tokens)`, `KEEP_LEADING_THE`, `geoparse_labels(accept_norp, include_fac)`,
  `nested_gazetteer_spans(doc, existing, geonames_service)`,
  and module constants `GEO_LABELS` / `CONTEXT_LABELS`.
* `doc_to_ex_expanded(doc, geo_labels=..., context_labels=..., trim_spans=True)`
  — same function, parameters added. The old call `doc_to_ex_expanded(doc)`
  still works and now trims.

### 1a. The trim guard ("The Hague")

`trim_span_tokens` trims the tail first and then walks the head, stopping if
the remaining span is one of `KEEP_LEADING_THE` — so `"The Hague's"` becomes
`"The Hague"`, not `"Hague"`.

**Why a static set, and why this one.** The set is derived from the index, not
guessed, and from the **primary `name` field only**:

```python
es.search(index="geonames", size=1000, sort=[{"population": "desc"}],
          query={"bool": {"must": [{"match": {"name": "the"}}],
                 "filter": [{"terms": {"feature_class": ["A", "P"]}},
                            {"range": {"population": {"gte": 10000}}}]}})
# keep the hits whose name.lower() starts with "the "  -> 20 names
```

Alternate names were deliberately excluded, and that is the crux: *"The
Gambia", "The Netherlands", "The Philippines" and "The United States" are all
alternate names of places whose own name is bare*, and English prose writes
them with a lower-case article that the corpora never annotate. Guarding on
alternate names would have destroyed the single biggest source of the +2.0
("the United States" → "United States"). Guarding on primary names picks up
exactly the places where the article is the name: The Hague, The Bronx, The
Dalles, The Woodlands, The Villages, The Colony, ... (20 entries). The guard
additionally requires a capitalised `The`, so `"the Hague"` in prose still
trims.

Retrieval was left alone (option (b) in the brief) because it would have meant
a second query, or a second name per entity through the shared
`add_es_data_batch` cache keys, for a case a static 20-name set covers at zero
cost.

**What it buys, measured two ways.**

*On the held-out corpora: nothing, and nothing is lost.* Re-running the 260
documents at defaults with the guard gives **e2e EM 60.58, det P 78.18, det R
70.57 — identical to the pre-guard numbers, corpus by corpus.** That is
expected rather than disappointing: across all 906 documents of TR/LGL/GWN
only 14 gold toponym spans keep a leading article, all of them organisation-ish
(`The Washington Post`, `The Boise Fire Department`), and there is no
Hague/Gambia/Bahamas-class mention in the gold at all. The guard is therefore
**strictly non-regressive here** (EM ≥ 60.58 ✓, precision unchanged ✓) and its
value has to be shown off-corpus.

*At retrieval, where it does show.* For each of the 20 guarded names, the rank
of the correct gazetteer row in its own 100-candidate list, queried with and
without the article:

| | with `The` | without |
|---|---|---|
| target ranked worse without the article | — | **10 of 20** |
| target unchanged | 7 | 7 |
| target falls out of the 100-candidate window entirely | 0 | **3** (The Gap, The Peak, The Ponds) |
| examples | The Beaches 1, The Colony 1, The Hammocks 1, The Bronx 1 | 55, 24, 21, 3 |

Dropping the article never improves the rank and sometimes makes the place
unresolvable at `max_choices=100`. That is the guard's job.

**Correction to the previous version of this report.** It said "The Hague"
trimmed to "Hague" comes back `no_match`, and attributed that to the trim.
That attribution was wrong. Measured on the same sentences with
`experiments/e29_swa_ep15/seed42.pt`:

| span fed to the ranker | candidate list | e29 result |
|---|---|---|
| `Hague` (trimmed) | The Hague NLD (2747373) at **rank 0** of 84 | `no_match`, p_no_match 0.56 |
| `The Hague` (untrimmed, and what the guard now produces) | The Hague NLD at **rank 0** of 12 | `no_match`, p_no_match 0.90 |

Retrieval is correct either way and the pre-fix serving path abstains on this
mention *identically*: it is a ranker behaviour on `The Hague`, not a trim
regression. The guard fixes the span, not the abstention. (The packaged legacy
checkpoint answers this mention, which is why `tests/test_geoparser.py`'s
Hague tests pass on both sides.) **A ranker that abstains at p = 0.9 on a
mention whose gold is rank 0 of 12 is worth a look in Phase 2** — the same
mention with the same candidate list is answered by the older checkpoint.

### Output shape

Every mention now comes back exactly once, and every result dict carries
`no_match` and `p_no_match`:

```python
{"search_name": "Aleppo", "start_char": 21, "end_char": 27,
 "no_match": False, "p_no_match": 0.0028, "score": 0.9752,
 "name": "Aleppo", "geonameid": "170063", "lat": ..., ...}      # placed

{"search_name": "Hague", "start_char": 10, "end_char": 15,
 "no_match": True,  "p_no_match": 0.5608}                       # abstained
```

`score` used to be a raw logit (or, on a pre-fix checkpoint, a raw softmax
output); it is now a probability in (0, 1). It is a strictly monotone
transform of what the model emitted, so **which** candidate wins and the
ordering of the `debug=True` top-4 are unchanged.

This repairs the §2a inconsistency: the two abstention branches
(`pred[-1] == pred.max()`, and "the last scored row wins") used to return a
bare dict and *silently drop the mention* respectively. Both now return the
same `no_match` row, as does the (unreachable-in-practice) empty-candidate
branch. Callers that iterated `geolocated_ents` expecting geo fields must now
check `no_match` — the first branch already handed them keyless dicts, so this
makes an existing hazard explicit rather than creating a new one.

## 2. Validation

### (a) Test suite

`uv run pytest tests/` → **59 passed, 1 skipped, 2 failed**. The two failures
are `tests/test_mordecai3.py::test_miss_oxford` and `::test_prague`, both
failing identically on the unmodified file (verified by stashing the change).
`tests/test_feature_parity.py` passes, so train/inference feature parity is
intact. Seven new tests were added to `tests/test_geoparser.py`: span trimming
as a unit, the trim guard (`The Hague`/`The Hague's`/`The Bronx` keep the
article; `the city`, `the United States`, `The Gambia` and lower-case
`the Hague` do not), trimmed spans and offsets through `doc_to_ex_expanded`,
`The Hague` surviving the serving path, NORP off by default and on when asked,
the label helper, and the confidence fields.

### (b) End to end, through the serving path

`Geoparser.geoparse_batch` over the held-out TR-News / LGL / GWN documents
(260 docs, 2,392 gold toponyms), model `experiments/e29_swa_ep15/seed42.pt`,
`max_choices=100`, scored by `tools/end_to_end_eval.py`'s `evaluate_docs` /
`summarize`. Targets are the corresponding rows of `end_to_end_report.md` §5,
which were produced by the harness's own reimplementation of extraction.

| config | pooled e2e EM | target | det P | target | det R | target | correct locations emitted | target |
|---|---|---|---|---|---|---|---|---|
| **default** (trim) | **60.58** | 60.7 | 78.18 | 78.2 | 70.57 | 70.6 | 1,449 / 1,811 (80.0%) | 1,452 / 1,817 (79.9%) |
| **default + trim guard** (shipping) | **60.58** | ≥ 60.58 | 78.18 | no drop | 70.57 | — | 1,449 / 1,811 (80.0%) | — |
| `accept_norp=True` | **64.88** | 64.9 | 75.04 | 75.0 | 79.18 | 79.2 | 1,552 / 2,015 (77.0%) | 1,552 / 2,018 (76.9%) |
| `accept_norp=True, nested_gazetteer_pass=True` | **72.83** | 72.9 | 67.62 | 67.6 | 88.96 | 89.0 | 1,742 / 2,513 (69.3%) | 1,742 / 2,518 (69.2%) |

Per corpus at the default: TR 61.3 / LGL 59.4 / GWN 62.6 e2e EM (the shipped
baseline was TR 61.3 / LGL 57.2 / GWN 60.0, pooled 58.6). With
`accept_norp`: TR 61.5 / LGL 64.9 / GWN 66.5. With both flags: TR 66.1 /
LGL 75.7 / GWN 70.6, against the report's `combo_best` of 66.4 / 75.6 / 70.9.

Every arm lands within 0.12 EM of its target, so **the serving code reproduces
the measured numbers**, and the port of the nested-gazetteer rule out of the
harness is faithful. The residual −0.12 on the default arm (three mentions) is
the only place the train/serve `locs_tensor` fix (change 4) can be hiding: the
harness's `trim` row pooled EVENT_LOC and NORP into `locs_tensor`, serving now
pools GPE/LOC. Its effect is therefore at most ~0.1 EM on these corpora,
i.e. nothing measurable either way — this change is for train/serve
consistency, not for accuracy.

The guard row is identical to the row above it in every per-corpus cell, not
just pooled: it changes no span in these corpora (§1a).

Two smoke arms on 30 LGL documents confirm the other flags behave as the
report says: `include_fac=False` moves det precision 63.1 → 79.5 and
throughput 11.8 → 14.6 docs/s at det recall 76.7 → 74.3;
`nested_gazetteer_pass=True` alone moves det recall 76.7 → 88.6 and e2e EM
65.9 → 73.1 at det precision 63.1 → 54.0.

### (c) Calibrated confidence

Five documents (16 mentions) through the serving path, with the same logits
re-scored by `tools/calibration_eval.py`'s own `masked_softmax` and its
`sel_mask` / `full_mask` definitions at T = 0.874:

* `max |serving probability − calibration_eval p_full| = 0.0` (exactly zero,
  every candidate row of every mention),
* the reported `score` equals `p_pred_full` at the chosen index for all 16
  mentions, and the chosen index equals `argmax(p_sel)`,
* `p_no_match` equals `p_reserved` to 1e-6 for all 16,
* the one mention where `calibration_eval` reports `reserved_argmax=True` is
  exactly the one mention serving returns as `no_match`,
* all scores and `p_no_match` lie strictly in (0, 1), and each mention's
  probabilities sum to 1.

So the number a user sees is, formula for formula, the score
`calibration_report.md` §6–§8 characterised: AUROC 0.899 for "this answer is
wrong", 0.86 for "this mention has no answer here", and the §8 default filter
(`score >= 0.7` and not `no_match`) is now expressible by a caller in one line.


### (d) Latency

50 LGL held-out documents are the report's latency setting; this check used
the first 30, three repetitions each, cache cleared per run (median docs/s of
the whole `geoparse_batch` call, RTX 4090):

| | docs/s |
|---|---|
| before the change | 11.33 (11.17 / 11.53 / 11.33) |
| after (default path) | **11.82** (11.71 / 11.92 / 11.82) |

**+4.3%**, i.e. within the ±5% budget and on the right side of it: trimming
removes "the" from the Elasticsearch query and occasionally collapses two
lookups into one. Nothing on the default path calls the gazetteer pass.

### (e) The evaluation harness agrees again

`tools/end_to_end_eval.py` now imports `GEO_LABELS`, `CONTEXT_LABELS` and
`trim_span_tokens` from `mordecai3.geoparse`, `doc_to_ex_labels` defaults to
`ctx_labels=CONTEXT_LABELS, trim_spans=True`, and `_trim_span` delegates to the
serving trimmer (guard included), so a variant measured there is the thing that
ships. **`check_parity` is green**: it was run over 24 held-out documents (8
per corpus), not just the one document the driver checks.

The pre-fix path is still reachable — `_variant()`'s defaults are unchanged
(`trim=False`, `ctx_labels=LEGACY_CTX_LABELS`), so the whole `ship` family of
variants still reproduces the baseline rows of `end_to_end_report.md`. Five new
`serving*` variants describe the current serving path, and `evaluate`/`latency`
now default to `--variants serving`. Confirmed on the 260 held-out documents,
one run, both variants:

| harness variant | pooled e2e EM | det R | det P | reproduces |
|---|---|---|---|---|
| `serving` | **60.58** | 70.57 | 78.18 | this report's default row (60.58) and the `trim` row of end_to_end_report.md (60.7) |
| `ship` | **58.61** | 68.14 | 75.46 | the report's baseline `ship` row (58.6 / 68.1 / 75.5), per corpus TR 61.25 / LGL 57.20 / GWN 60.03 vs 61.3 / 57.2 / 60.0 |

```
uv run python tools/end_to_end_eval.py evaluate \
  --model-path experiments/e29_swa_ep15/seed42.pt \
  --variants serving,ship --no-oracle
```

## 3. Deviations and things deliberately not done

1. **The abstention *decision* rule is untouched.** §8 keeps
   `pred[-1] == pred.max()`, and so does this: the masked argmax that
   `calibration_eval` uses is equivalent for a `mask_padding` checkpoint but
   *not* for the packaged pre-fix asset, where padding rows still carry
   probability mass. Changing the rule would have silently changed how often
   the legacy default checkpoint abstains, which is out of scope here.
2. **Probabilities are recovered by logs for a non-`return_logits` model.** A
   checkpoint built without `return_logits` has already softmaxed; serving
   takes `log(p)` and re-softmaxes over the live rows at T. That reproduces
   the model's own distribution exactly at T = 1 and is what makes the field
   meaningful for both checkpoint kinds — but the packaged
   `assets/mordecai_2025-08-27.pt` predates the double-softmax fix and is
   saturated (scores of 1.0 to float precision). Its `score` is a probability,
   not a calibrated one. The docstring says so; the fix is to ship an
   `e29_swa_ep15`-recipe checkpoint as the default asset.
3. **`tools/end_to_end_eval.py` is aligned; `tools/calibration_eval.py` is
   untouched.** The harness now shares the serving label sets and trimmer, and
   `check_parity` passes again (§2e). `calibration_eval.py` needed no change:
   serving reproduces its `p_pred_full` exactly (§2c), so it stays the
   reference implementation. §2b still drives `Geoparser` directly rather than
   the harness (appendix), so the two are independent checks, not one check
   run twice.
4. **The trim guard covers primary gazetteer names only** (§1a): "The Hague"
   and "The Bronx" keep their article; "The Gambia" and "the Netherlands" do
   not, because those are alternate names of bare-named places and trimming
   them is where most of the +2.0 comes from. The set is the 20 A/P names with
   population ≥ 10,000 whose own name starts with "The "; a deployment that
   cares about smaller article-led places extends `KEEP_LEADING_THE` with the
   same query. Not done: a runtime retrieval fallback (query both forms, keep
   whichever matches a gazetteer name exactly), which generalises to every
   such place but costs either a second lookup or a restructured cache key in
   the shared `add_es_data_batch` path, for a case the static set covers for
   free.
5. **The "The Hague" abstention is not fixed, because it is not a serving
   bug** (§1a). `experiments/e29_swa_ep15/seed42.pt` abstains on that mention
   at p_no_match 0.90 whether the span is trimmed or not, with the gold row at
   rank 0 of the candidate list; the older packaged checkpoint answers it.
   Flagged for Phase 2 as a ranker question, not patched here.
6. **`locs_tensor` alignment is with three of the four training formatters.**
   `data_formatter`, `data_formatter_prodigy` and `data_formatter_wiki` use
   GPE/LOC (`tools/train.py:376, 452, 684`), but
   `data_formatter_wiki_docs:520` uses GPE/LOC/EVENT_LOC/NORP — and WikiDocs
   is the bulk of the training mass. Serving now matches the three, not the
   one. Since `encoder_scoping_report.md` also finds the slot earns nothing
   (arm `nolocs`), the honest fix is E45; this change just removes a
   train/serve difference nobody chose.
7. **No gazetteer "outside" pass** (`gaz_out`, +1.7 EM): the report recommends
   nested only, because combining them is worse on LGL. Not implemented.
8. **No default abstention filter.** §8's `p >= 0.7` recommendation is left to
   the caller; serving exposes the number and does not drop answers on it.

## Appendix: reproducing §2b

The check drives the real `Geoparser` and re-uses the harness only for corpus
reading and scoring:

```python
import sys; sys.path.insert(0, "tools")
from end_to_end_eval import evaluate_docs, heldout_doc_indices, read_corpus, summarize
from mordecai3 import Geoparser

geo = Geoparser(model_path="experiments/e29_swa_ep15/seed42.pt",
                feature_blocks="prom,name,cue,sib,geo,shape", oov_bucket_fix=True,
                model_options={"return_logits": True, "mask_padding": True,
                               "modern_mlp": True})          # + accept_norp=True, etc.
articles = read_corpus("lgl", "raw_data")
keep = sorted(heldout_doc_indices("lgl", articles, "raw_data")[0])
docs_meta = [articles[i] for i in keep]
outs = geo.geoparse_batch([a["text"] for a in docs_meta], batch_size=8)
all_es = [[{"search_name": e["search_name"], "start_char": e["start_char"],
            "end_char": e["end_char"], "es_choices": []}
           for e in o["geolocated_ents"]] for o in outs]
picks = [[None if e.get("no_match") else e for e in o["geolocated_ents"]] for o in outs]
print(summarize(evaluate_docs(docs_meta, all_es, picks)))
```

(`es_choices` is left empty because only the retrieval/ranker decomposition
reads it; detection and end-to-end EM do not.)
