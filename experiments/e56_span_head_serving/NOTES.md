# e56_span_head_serving — the place-span head in the serving path (ladder step N1)

Campaign 2, the NER track. e55 built and measured the head offline and staged a
self-contained inference module plus two checkpoints; this entry **ports it into
`mordecai3/`** and produces the measurement nobody had: end-to-end exact match
for the head and the `outlet` block **together**, on one grid, on the D2
denominator. Full write-up: `experiments/campaign2/span_head_serving_report.md`.

**Verdict: the head is the largest single serving win the campaign has
measured, it composes additively with the outlet block, and it is FASTER than
the path it replaces. e2e EM on the 260 held-out documents goes 66.99 (e29
seed42, today's `serving` row) → 80.85 for e54 seed42 + the `gold` head + LGL/TR
outlets, against that configuration's oracle-span ceiling of 87.41. The
detection share of the loss halves, 21.79% of gold toponyms → 10.51%.
`span_detector` DEFAULTS TO None and nothing is flipped: the N1 gate — reproduce
on D1's untouched modern-news TEST corpus — is still open, and every number here
is on held-out documents of the corpora the head trained on.**

## What was integrated

| artifact | where it landed |
|---|---|
| `span_head.py` (e55 ship) | `mordecai3/span_head.py`, + `SPAN_HEAD_ASSETS` / `resolve_span_head` / `load_span_tagger` |
| `span_head_gold_42.pt` | `mordecai3/assets/span_head_2026-08-20_gold.pt` (md5 `0691b19b…`, unchanged) |
| `span_head_C_all_42.pt` | `mordecai3/assets/span_head_2026-08-20_all.pt` (md5 `5e8ec634…`, unchanged) |
| the flag | `Geoparser(span_detector=None|"gold"|"all"|<path>, span_threshold=None)` |
| package data | both checkpoints listed in `pyproject.toml` |
| tests | `tests/test_span_head.py`, 15 cases |
| harness | `tools/end_to_end_eval.py` variants `head_gold` / `head_all` |

Per e55's `INTEGRATION.md`, setting `span_detector` replaces **three** things
with one call: the `GEO_LABELS` filter over `doc.ents`, `trim_span_tokens`, and
the opt-in `nested_gazetteer_spans` pass (which is then ignored, with a
warning). Everything downstream — candidate retrieval, enrichment features, the
ranker, the reserved-row decode, `p_no_match`, the outlet plumbing — is
untouched.

## Gates

1. **`span_detector=None` is bit-identical.** The whole geoparse of the 260
   held-out documents, dumped canonically and md5'd, is the same under the
   current package and under a copy with the span-head hunks reverted
   (`identity_check.py`). The harness's `serving` row on e29 seed42 also
   reproduces `phase0_report.md` §1 cell for cell: **66.99** / 69.19 / 78.21 /
   75.50 / 77.08 / 83.45.
2. **The packaged heads reproduce e55's detection row exactly**, through the
   library module, on the D2 denominator (`parity_span_head.py`):
   gold 85.83 / 89.49 / **87.62**, nested 67.9, demonym FP 81, 2,173 preds;
   all 83.95 / 88.87 / **86.34**, nested 76.5, demonym FP 56, 2,206 preds.
   The identity hash is `f2656f881c0e85632ad9ec8235a0755d` over 2,159 mentions,
   and the current package produces it twice, so it is an identity and not a
   coincidence of a nondeterministic pipeline.
3. **Suite green modulo the known allowances**: `test_miss_oxford`,
   `test_prague`, 5 Phase-0 xfails. 94 passed, 1 skipped.
4. **e2e EM reproduces e55's own head seed** — 77.26 for gold on e29 seed42, one
   of e55's three-seed set {74.81, 77.02, 77.26} — from independent code.

## The grid (D2, 2,084 golds, 260 documents, `max_choices=100`)

e2e exact match; detection is ranker-invariant, so it is one row per detector.

| ranker | outlets | none | gold head | all head |
|---|---|---|---|---|
| e29 seed42 (Phase-0 reference) | — | 66.99 | **77.26** | 76.68 |
| e29 seed101 (packaged default) | — | 67.66 | **77.69** | 77.02 |
| e54 seed42 (staged) | none | 67.95 | **78.55** | 77.59 |
| e54 seed42 (staged) | LGL+TR | 70.39 | **80.85** | 80.13 |

| detector | det P | det R | det F1 | emitted-loc P (best cell) | demonym FP |
|---|---|---|---|---|---|
| none (spaCy filter + trim) | 75.50 | 78.21 | 76.83 | 82.32 | 59 |
| gold head | 85.83 | 89.49 | **87.62** | **83.71** | 81 |
| all head | 83.95 | 88.87 | 86.34 | 83.54 | **56** |

**Composition.** Head alone +10.27, outlet alone +3.40, both **+13.86** on a
+13.67 additive prediction — they compose, with a hair to spare, because they
fail on different toponyms.

**gold beats C_all end to end in all four ranker rows**, by 0.58 / 0.67 / 0.96 /
0.72 EM. C_all's +8.6 nested detection recall is real and it does convert into
nested e2e (e55: 61.48 vs 54.90), but nested toponyms are ~1/5 of the gold set
and resolve worse than flat ones, so 1.3 points of flat F1 given away costs more
than the nested recall buys. Its 25 fewer demonym false positives are the one
argument for it that the EM column does not see.

## Latency (LGL held-out, 50 documents, 3 reps, `geoparse_batch` wall clock)

| configuration | GPU ms/doc | CPU ms/doc |
|---|---|---|
| default (spaCy filter) | 44.78 | 195.25 |
| `nested_gazetteer_pass=True` | 60.22 | 209.36 |
| span head: gold | **37.57** | **184.70** |
| span head: all | 39.90 | 191.28 |

Not latency-neutral — **faster**, by 5–7 ms/doc on GPU. The head itself costs
4.9 ms/doc, and it *saves* more than that in Elasticsearch: the label-filter
path emits ~370 junk spans that retrieve nothing and trigger `add_es_data`'s
fuzzy-retry round trip. On the harness's LGL block, ES time is 5.3 s → 3.0 s.

## Abstention

`serving_probe.py calib`, all 260 documents. Not a recalibration — a
distribution check.

| ranker / detector | mentions | abstain rate | mean p_no_match | p > 0.5 |
|---|---|---|---|---|
| e29 seed101 / none | 2,159 | 14.17% | 0.1149 | 11.35% |
| e29 seed101 / gold | 2,173 | **5.34%** | 0.0421 | 3.18% |
| e54 seed42 / none | 2,159 | 17.46% | 0.1571 | 15.98% |
| e54 seed42 / gold | 2,173 | **7.36%** | 0.0650 | 5.48% |

The abstention machinery is doing less work because it is being asked fewer
"this is not a place" questions: same mention count, far higher detection
precision. A consumer thresholding `p_no_match` is at a different operating
point and should re-pick its threshold; the temperature itself is untouched.

## Where the remaining gap is (best cell: e54 + gold head + outlets)

2,084 gold toponyms: 80.85% correct, **10.51% lost to detection** (7.68 never
found, 2.84 found with the wrong boundary), **8.64% lost to resolution** (5.37
retrieval, 2.69 ranker, 0.58 abstain). Under the spaCy filter those numbers are
21.79% and 7.82%: the head halves the detection loss and pays 0.8 points of
resolution loss back, because the toponyms it newly finds — nested in
organisation names — are harder to retrieve (`retrieval_miss` 88 → 112).

Measured against the ceiling gold by gold, the 130-toponym gap to 87.41 is
**97.4% detection**: exactly **4** of 2,073 golds are ones the same ranker
resolves correctly from the gold span and wrongly from the head's span, and 22
go the other way. Changing the span source carries no hidden resolution debt,
and the next lever is retrieval, not ranking.

## Leave-one-corpus-out (the N1 gate, made cheap)

Train on two corpora, score detection on the third's held-out documents, 3
seeds, threshold picked on the training families' dev split. The bar is the
spaCy label-filter path on that corpus, not the in-family 87.6.

| arm | held out | out-of-family F1 | Δ vs spaCy | t(2) | spaCy | in-family |
|---|---|---|---|---|---|---|
| gold | TR-News | 83.21 ± 0.58 | **+5.86** | 17.65 \* | 77.35 | 83.87 |
| gold | LGL | 84.52 ± 1.20 | **+9.28** | 13.42 \* | 75.24 | 90.31 |
| gold | GeoWebNews | 80.81 ± 1.08 | +0.51 | 0.82 ns | 80.30 | 83.71 |
| dem10 | TR-News | 83.07 ± 0.71 | +5.72 | 14.00 \* | 77.35 | 83.87 |
| dem10 | LGL | 81.11 ± 1.46 | +5.87 | 6.98 \* | 75.24 | 90.31 |
| dem10 | GeoWebNews | 81.15 ± 0.94 | +0.85 | 1.58 ns | 80.30 | 83.71 |

**No fold falls below the spaCy path; the GWN fold ties it.** End to end on that
worst fold (e54 seed42, GWN, 523 golds): out-of-family **75.02 ± 1.81** (74.38 /
77.06 / 73.61) against the spaCy path's 74.19 and the in-family head's 79.35 —
+0.83, t = 0.79, n.s., with one seed below. So ~5 of the head's in-family +10.5
EM is corpus-specific span convention. Nested recall transfers everywhere
(43.9 vs 23.2 even on the tying fold); demonym FPs on GWN are the same
out-of-family (77.7) as in-family (77), i.e. not a generalisation failure but a
GWN annotation-policy one — TR and LGL contribute 0 and 4.

**Gate: amber.** Not closed; a real TEST corpus can still fail in ways three
benchmark corpora cannot show.

## Calibration on the serving frame

Not the campaign's oracle-span frame: here the detector picks the mentions and
19–29% of them have no right answer. e54 seed42 + LGL/TR outlets, both
detectors on the same frame, metric code imported from
`tools/calibration_eval.py`.

| | none | gold head |
|---|---|---|
| mentions / answerable | 2,159 / 1,541 | 2,173 / **1,752** |
| refit T | 1.034 | **1.106** |
| ECE at T = 0.874 → refit | 0.1037 → 0.0689 | 0.0941 → **0.0430** |
| AURC | 0.1209 | **0.1001** |
| selective EM @ 90% coverage | 75.50 | **85.63** |
| AUROC, unanswerable, `p_no_match` | 0.8945 | **0.7967** |

Three results: **the serving-frame temperature is above 1**, not the campaign's
0.874 (which sharpens a model that is under-confident only on gold spans);
**the recommended threshold under the head is `p ≥ 0.5` or the reserved-argmax
flag** (88.0% coverage, 86.5% selective EM — the old `p ≥ 0.7` now costs 81.4%
coverage); and **`p_no_match`'s ranking quality genuinely degraded**, −0.098
AUROC for unanswerability, because the easy junk-span negatives are gone and
what remains is harder. Absolute selective performance is better at every
coverage level. Any deployment thresholding `p_no_match` must re-pick its
threshold at the flip.

## Reproducing

```
uv run python experiments/e56_span_head_serving/parity_span_head.py gold   # and: all
uv run python experiments/e56_span_head_serving/detection_grid.py
bash experiments/e56_span_head_serving/run_grid.sh                          # ~20 min
uv run python experiments/e56_span_head_serving/aggregate.py e54_seed42_outlet:head_gold
uv run python experiments/e56_span_head_serving/serving_probe.py calib
uv run python experiments/e56_span_head_serving/serving_probe.py lat        # and: lat --cpu
uv run python experiments/e56_span_head_serving/identity_check.py
uv run python experiments/e56_span_head_serving/loco.py               # ~10 min
uv run python experiments/e56_span_head_serving/calibration_refit.py
```

`data/{tr,lgl,gwn}_docs.json` are e55's gold-span dumps, copied here so the
parity gate stays runnable after its session scratchpad is gone.
