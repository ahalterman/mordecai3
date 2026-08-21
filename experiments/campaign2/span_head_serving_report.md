# The place-span head in the serving path, and the first combined measurement (e56_span_head_serving)

Campaign 2, the NER track, ladder step **N1**. Written 2026-08-20. e55 built the
0.5 M-parameter place-span head, measured it offline against every label source
on disk, and staged a self-contained inference module with two checkpoints
(`ner_head_scaling_report.md`, `experiments/e55_ner_head/ship/INTEGRATION.md`).
This report lands it in `mordecai3/` behind a constructor flag and answers the
question no arm had the machinery to ask: **what does the pipeline score when
the span head and the `outlet` block are both on?**

**Verdict: the head is the largest serving win the campaign has measured, it is
additive with the outlet block, and it is *faster* than the path it replaces.
End-to-end exact match on the 260 held-out documents (D2, 2,084 gold toponyms):
66.99 today → 80.85 for `e54 seed42 + the gold head + LGL/TR outlets`, against
that configuration's oracle-span ceiling of 87.41. The head is worth +10.0 to
+10.6 EM in every ranker and outlet condition, the outlet +2.3 to +2.4 in every
span condition, and the two interact by less than ±0.15. Detection's share of
the loss halves (21.79% of golds → 10.51%) and latency falls 44.8 → 37.6 ms/doc
on GPU. `span_detector` DEFAULTS TO None and nothing is flipped here.**

**Follow-ups (§11, §12), added after the first pass.** Leave-one-corpus-out
generalisation: train on two corpora, score the third. **No fold falls below the
spaCy path; two of three beat it decisively (+5.9 and +9.3 F1); GeoWebNews ties
(+0.51, t = 0.82, n.s.), and ties end to end too (+0.83 EM, t = 0.79).** The
gate is amber. Calibration on the serving frame: the fitted temperature moves
**above** 1 (0.874 → 1.106), refitting halves ECE to 0.043, the recommended
threshold becomes **`p ≥ 0.5` or flag** (88.0% coverage, 86.5% selective EM),
and `p_no_match`'s AUROC for unanswerable detection **degrades 0.895 → 0.797**
even though selective performance improves at every coverage level.

Three caveats stated up front, all in §8:

* every number in §3–§7 is on **held-out documents of the corpora the head
  trained on**. The N1 gate — reproduce on D1's untouched modern-news TEST
  corpus — is **still open**; §11 narrows it but does not close it.
* the grid uses **one head seed**. e55's three gold-head seeds span 74.81 /
  77.02 / **77.26** e2e on e29 seed42, and the packaged seed 42 is the 77.26.
  Read the best cell as 80.85 with an inherited head-seed spread of about ±1.4,
  not as a point estimate.
* the head emits **overlapping spans**, which is a behaviour change for
  consumers. Quantified in §6: 24 of 2,013 emitted locations in the best cell.

---

## 1. What was integrated

| artifact | landed as |
|---|---|
| e55's `span_head.py` | `mordecai3/span_head.py` (+ `SPAN_HEAD_ASSETS`, `resolve_span_head`, `load_span_tagger`) |
| `span_head_gold_42.pt` | `mordecai3/assets/span_head_2026-08-20_gold.pt`, md5 `0691b19bc572ec4aaa90e19a69a5d5fd` — byte-identical to e55's |
| `span_head_C_all_42.pt` | `mordecai3/assets/span_head_2026-08-20_all.pt`, md5 `5e8ec634fe8163b97a408b55b7d7699c` — byte-identical |
| the flag | `Geoparser(span_detector=None \| "gold" \| "all" \| <path>, span_threshold=None)` |
| package data | both checkpoints added to `pyproject.toml` (1.6 MB each) |
| tests | `tests/test_span_head.py`, 15 cases |
| harness | `tools/end_to_end_eval.py` variants `head_gold` / `head_all`, `--variants` reuses one spaCy pass per corpus |

`span_detector` is not a fourth extraction option layered on the other three —
per e55's spec it **replaces** them. When it is set, `doc.ents` is no longer
filtered by `GEO_LABELS`, `trim_span_tokens` never runs, and
`nested_gazetteer_spans` is skipped with a warning if the caller also asked for
it. Everything downstream — candidate retrieval, the six enrichment blocks, the
outlet block, `ProductionData`, the ranker, the reserved-row decode,
`p_no_match`, temperature — is untouched, because `SpanTagger.doc_to_ex` returns
the same eight-key dicts `doc_to_ex_expanded` returns and computes seven of them
with the same code.

```python
# mordecai3/geoparse.py, Geoparser._geoparse_docs
if self.span_tagger is not None:
    doc_ex = self.span_tagger.doc_to_ex(doc, context_labels=CONTEXT_LABELS)
else:
    doc_ex = doc_to_ex_expanded(doc, geo_labels=self.geo_labels,
                                trim_spans=self.trim_spans)
    if self.nested_gazetteer_pass:
        doc_ex = doc_ex + nested_gazetteer_spans(doc, doc_ex, self.geonames)
        doc_ex.sort(key=lambda e: e["start_char"])
```

The head is loaded eagerly in `__init__`, so a bad name fails at construction
rather than on the first document, and the detection threshold travels inside
the checkpoint (0.5 for gold, 0.3 for all). `span_threshold=` exists for sweeps
and is the documented way to leave the measured operating point.

## 2. Gates

### 2a. `span_detector=None` is bit-identical to the pre-change path

Not asserted — measured. `experiments/e56_span_head_serving/identity_check.py`
geoparses all 260 held-out documents through `Geoparser.geoparse_batch`, dumps
every emitted field of every mention canonically, and md5s it. The same script
was then run against a copy of the package with the three span-head hunks
reverted and `span_head.py` deleted:

| package | mentions | md5 |
|---|---|---|
| current, run 1 | 2,159 | `f2656f881c0e85632ad9ec8235a0755d` |
| current, run 2 (determinism baseline) | 2,159 | `f2656f881c0e85632ad9ec8235a0755d` |
| **pre-change copy** | 2,159 | **`f2656f881c0e85632ad9ec8235a0755d`** |

The second row matters: without it a matching hash would not distinguish "the
change is inert" from "the pipeline is not deterministic anyway".

### 2b. The frozen reference row reproduces

`tools/end_to_end_eval.py evaluate --model-path experiments/e29_swa_ep15/seed42.pt
--variants serving,head_gold,head_all`, D2 denominator, `max_choices=100`:

| | e2e EM | acc@161 | det R | det P | emitted-loc P | oracle-span EM |
|---|---|---|---|---|---|---|
| `phase0_report.md` §1 | 66.99 | 69.19 | 78.22 | 75.50 | 77.08 | 83.45 |
| **here** | **66.99** | **69.19** | 78.21 | **75.50** | **77.08** | **83.45** |

Per corpus 64.95 / 65.04 / 72.85, matching Phase 0 to the decimal. The 78.21
against a published 78.22 is a pooling convention, not a difference: this report
pools from integer counts, the earlier one took a weighted mean of the rounded
per-corpus rates. It is worth at most 0.01 anywhere in this document; the e54
outlet row below reads 70.39 here against a published 70.40 for the same reason.

`e54 seed42, no outlet supplied → 67.95` and `LGL+TR outlets → 70.39` also
reproduce `outlet_integration_report.md` §7.

### 2c. The packaged heads reproduce e55's detection row exactly

`parity_span_head.py`, e55's scorer, D2 denominator, straight from the cached
DocBins, through the **library** module and the **packaged** assets:

| checkpoint | det P | det R | det F1 | nested det R | demonym FP | preds |
|---|---|---|---|---|---|---|
| gold, e55's row | 85.83 | 89.49 | 87.62 | 67.9 | 81 | 2,173 |
| **gold, here** | **85.83** | **89.49** | **87.62** | **67.9** | **81** | **2,173** |
| all, e55's row | 83.95 | 88.87 | 86.34 | 76.5 | 56 | 2,206 |
| **all, here** | **83.95** | **88.87** | **86.34** | **76.5** | **56** | **2,206** |

`detection_grid.py` adds the row the head is measured against and reproduces
e55's "serving path today" line as well: F1 **76.83**, nested recall **13.8**,
demonym FP **59**.

### 2d. Suite

`uv run pytest tests/` → **2 failed, 94 passed, 1 skipped, 5 xfailed**. The two
failures are the documented allowances (`test_miss_oxford`, `test_prague`) and
the five xfails are Phase 0's. The 15 new cases cover the checkpoints' operating
points, field-level parity with `doc_to_ex_expanded` on a shared span, nested
emission, demonym suppression, determiner-free spans, the empty document, the
default's inertness, and both `Geoparser(span_detector=...)` paths.

### 2e. An unplanned cross-check

e2e EM for the gold head on e29 seed42 comes out at **77.26**, which is exactly
one of e55's three head-seed values (74.81 / 77.02 / 77.26) — computed by
different code, through the library rather than e55's offline tagger. Two
harnesses, one number.

---

## 3. The grid

260 held-out documents of TR-News + LGL + GeoWebNews, D2 denominator (2,084
non-demonym linked gold toponyms), `max_choices=100`, exact-span exact-id match.
Raw JSON: `experiments/e56_span_head_serving/e2e/`.

### 3a. End-to-end exact match

| ranker | outlets | **none** | **gold head** | **all head** |
|---|---|---|---|---|
| e29 seed42 — the Phase-0 reference | — | 66.99 | **77.26** | 76.68 |
| e29 seed101 — the packaged default | — | 67.66 | **77.69** | 77.02 |
| e54 seed42 — staged | none | 67.95 | **78.55** | 77.59 |
| e54 seed42 — staged | LGL + TR | 70.39 | **80.85** | 80.13 |

acc@161 km tracks it: 69.19 / 79.94 / 79.46 in the first row, 72.36 / **82.87** /
82.20 in the last.

Per corpus, e2e EM:

| ranker / outlets | detector | TR-News | LGL | GeoWebNews |
|---|---|---|---|---|
| e29 seed101 | none | 67.07 | 65.69 | 72.66 |
| e29 seed101 | gold | 72.21 | 79.19 | 77.63 |
| e54 + outlets | none | 66.77 | 69.76 | 74.19 |
| e54 + outlets | gold | **72.81** | **83.66** | **79.35** |

LGL is where the head pays most (+13.5 under e29 seed101), which is the corpus
whose toponyms are most often buried in local-institution names.

### 3b. Detection, emitted-location precision, demonym false positives

Detection is ranker-invariant — identical in all four ranker rows — so it is one
row per detector. Emitted-location precision is "of the locations the pipeline
actually outputs, how many are a real gold toponym resolved to the right id",
and it *is* ranker-dependent, so the best cell is quoted.

| detector | det P | det R | det F1 | nested det R | preds | emitted-loc P (e54+outlets) | demonym FP |
|---|---|---|---|---|---|---|---|
| none | 75.50 | 78.21 | 76.83 | 13.8 | 2,159 | 82.32 | 59 |
| **gold** | **85.83** | **89.49** | **87.62** | 67.9 | 2,173 | **83.71** | 81 |
| **all** | 83.95 | 88.87 | 86.34 | **76.5** | 2,206 | 83.54 | **56** |

The head buys +10.8 detection F1 for **14 more predicted spans**. That is the
shape of the whole result: it is not trading precision for recall, it is finding
different spans.

### 3c. Composition — does the head's win survive the outlet's?

Yes, and essentially without interaction.

| effect | measured where | Δ e2e EM |
|---|---|---|
| gold head | on e29 seed42, no outlet | **+10.27** |
| gold head | on e29 seed101, no outlet | +10.03 |
| gold head | on e54 seed42, no outlet | +10.60 |
| gold head | on e54 seed42, LGL+TR outlets | +10.46 |
| outlets supplied | on e54 seed42, no head | +2.44 |
| outlets supplied | on e54 seed42, gold head | +2.30 |
| e29 seed42 → e54 seed42 (ranker only) | no head, no outlet | +0.96 |

Against the Phase-0 reference of 66.99, the full stack is **+13.86** where
strict additivity predicts +13.67. The interaction is +0.19, inside the noise of
a single head seed. They are independent because they fail on different things:
the head fixes spans, the outlet fixes which of several same-named places is
meant, and a toponym has to survive both.

---

## 4. gold vs C_all under the full pipeline

e55 could not call this: `gold` leads on flat detection F1 by 1.3, `C_all` leads
on nested detection recall by 8.6 and on demonym false positives by 25, and the
three point in different directions. End to end, on all four ranker rows:

| ranker / outlets | gold | all | Δ |
|---|---|---|---|
| e29 seed42 | 77.26 | 76.68 | **+0.58** |
| e29 seed101 | 77.69 | 77.02 | **+0.67** |
| e54, no outlet | 78.55 | 77.59 | **+0.96** |
| e54, LGL+TR | 80.85 | 80.13 | **+0.72** |

**gold wins, consistently, by about 0.7 EM.** The arithmetic is not subtle once
the decomposition is in front of you: 405 of the 2,084 golds are nested and
1,679 are flat. C_all's +8.6 nested recall is worth ~35 nested golds; its −2.9
flat recall costs ~49 flat golds, and flat golds resolve *better* than nested
ones once found, so the exchange is losing on both legs. Its `ner_miss` count is
higher than gold's (166 vs 160) and its boundary errors are higher too (32 vs
22).

The honest counter-argument, which the EM column cannot see:

* **demonym false positives: 56 vs 81.** Under D2 a demonym prediction is not a
  hallucination and not a recall failure, so it is invisible in every rate in
  §3 — but a user who sees "Turkish → Turkey" attached to "the Turkish
  president" experiences it as a wrong answer. C_all is the only detector here
  that beats the label filter's 59 on this metric; `gold` is meaningfully worse
  than what ships today.
* **nested end-to-end EM** (e55: 61.48 vs 54.90) is C_all's, by a lot, and a
  deployment whose value is in "which institution is where" rather than pooled
  EM should read that row, not this one.

Recommendation in §7 goes with `gold` on the pooled metric and names the flip to
`all` as a one-word change for a deployment that weights demonyms differently.

---

## 5. Latency

`serving_probe.py lat`, 50 held-out LGL documents, 3 repetitions, median, ES
candidate cache cleared before each. This is `geoparse_batch` wall clock — spaCy
+ extraction + Elasticsearch + ranker — i.e. what a caller pays, not a
microbenchmark of the head.

| configuration | GPU ms/doc | GPU docs/s | CPU ms/doc | emitted mentions |
|---|---|---|---|---|
| default (`span_detector=None`) | 44.78 | 22.33 | 195.25 | 309 |
| `nested_gazetteer_pass=True` | 60.22 | 16.61 | 209.36 | 442 |
| **span head: gold** | **37.57** | **26.61** | **184.70** | 294 |
| span head: all | 39.90 | 25.07 | 191.28 | 311 |

e55 predicted "roughly latency-neutral when it replaces the nested-gazetteer
pass". It is better than that: **−7.2 ms/doc against the default path**, which
does not run the gazetteer pass at all, and −22.7 against the pass it was
budgeted against. The head's own cost is 4.9 ms/doc (0.85 s over the harness's
175 LGL documents, consistent with e55's +5.2), so the arithmetic only works
because something else gets cheaper — and it does: **Elasticsearch time on the
same 175 documents falls 5.3 s → 3.0 s**. Of the label-filter path's 2,159
predicted spans, 431 sit on no D2 gold toponym at all (191 on an unlinked gold
row, 181 on nothing, 59 on a demonym); the ones the gazetteer cannot match each
trigger `add_es_data`'s fuzzy-retry round trip. Better spans mean fewer wasted
searches.

CPU-only shows the same sign with the margin compressed, as expected — there the
transformer forward pass is ~90% of the budget and everything else is noise
around it.

---

## 6. Two behaviour changes for callers

**Overlapping spans.** The head emits "Pittsburgh" and "University of
Pittsburgh" at overlapping offsets when it believes both. In the best cell, 24
of 2,013 emitted locations (1.2%) are a second prediction overlapping a gold
toponym another prediction already claimed; for `all` it is 38 of 1,999 (1.9%).
Under the label filter this category is structurally empty. Downstream code that
assumes a flat entity list needs to say which it wants; e55 measured containment
suppression at ~+0.1 F1, i.e. it is an output option, not an accuracy lever.

**Demonyms are suppressed by training, not guaranteed away.** §4. A caller who
must never see one needs a NORP-span veto as a post-filter — cheap, since
spaCy's NER is still running for the context tensor and `guess_in_rel`.

For completeness, what the pipeline emits with no gold toponym under it at all
("hallucinated locations", best cell): 44 of 1,782 emitted (2.5%) today, 51 of
2,013 (2.5%) with the gold head, 71 of 1,999 (3.6%) with `all`. The head emits
13% more locations, because it abstains far less (§7a), at an unchanged
hallucination *rate*.

---

## 7. Oracle attribution for the best cell

`e54 seed42 + gold head + LGL/TR outlets`, 2,084 golds. Where every gold
toponym goes, and — since the harness runs the *same ranker* over the gold spans
on the same documents — which of the ceiling's 1,815 correct answers the
pipeline loses and to what.

| bucket | none | gold head |
|---|---|---|
| correct | 1,467 (70.39%) | **1,685 (80.85%)** |
| ner_miss | 356 (17.08%) | 160 (7.68%) |
| boundary_wrong | 57 (2.74%) | 22 (1.06%) |
| boundary_ok (right place, wrong span) | 41 (1.97%) | 37 (1.78%) |
| retrieval_miss | 88 (4.22%) | 112 (5.37%) |
| ranker_error | 58 (2.78%) | 56 (2.69%) |
| null_answer (abstained) | 17 (0.82%) | 12 (0.58%) |
| **— detection loss** | **454 (21.79%)** | **219 (10.51%)** |
| **— resolution loss** | 163 (7.82%) | 180 (8.64%) |

Gold-by-gold against the ceiling (oracle-span EM 87.41, 1,815 of 2,073):

| | none | gold head | all head |
|---|---|---|---|
| gap to ceiling | 348 golds | **130** | 145 |
| lost to detection | 352 | 148 | 162 |
| lost to resolution *on a span it did find* | 9 | **4** | 4 |
| won where the oracle span failed | 13 | 22 | 21 |
| **detection share of the gap** | 97.5% | **97.4%** | 97.6% |

**The answer to the question the ladder asked: with the head on, the remaining
gap to its own ceiling is still 97% detection.** Of the 2,073 gold toponyms the
oracle pass covers, exactly **four** are ones the same ranker resolves correctly
from the gold span and mis-resolves from the span the head found. Handing the
ranker cleaner spans does not make it worse; there is no hidden resolution debt
in the head's output, and there is no ranker work waiting behind this change.

Two second-order effects worth naming because they run opposite ways:

* `retrieval_miss` rises 88 → 112. The toponyms the head newly finds are the
  ones nested inside organisation names, and those are exactly the ones the
  gazetteer query is least likely to bring back. This is the next lever, and it
  is a *retrieval* lever, not a ranker one.
* 22 golds are resolved correctly by the pipeline where the oracle *gold span*
  fails, up from 13. The head's span and its context tensor are occasionally
  better inputs than the corpus annotation's own offsets.

The absolute ceiling has not moved: 87.41 is the same oracle row for all three
detectors, and closing the last 6.56 points needs a better ranker or better
retrieval, not better spans.

### 7a. Abstention

`serving_probe.py calib`, all 260 documents, through `Geoparser`. A distribution
check, not a recalibration; the temperature (0.874) is untouched.

| ranker / detector | mentions | abstain rate | mean p_no_match | median | p > 0.5 | p < 0.01 |
|---|---|---|---|---|---|---|
| e29 seed101 / none | 2,159 | 14.17% | 0.1149 | 0.0025 | 11.35% | 70.2% |
| e29 seed101 / gold | 2,173 | **5.34%** | 0.0421 | 0.0018 | 3.18% | 81.6% |
| e29 seed101 / all | 2,206 | 6.80% | 0.0514 | 0.0018 | 4.22% | 79.6% |
| e54 seed42 / none | 2,159 | 17.46% | 0.1571 | 0.0038 | 15.98% | 65.4% |
| e54 seed42 / gold | 2,173 | **7.36%** | 0.0650 | 0.0029 | 5.48% | 75.7% |
| e54 seed42 / all | 2,206 | 9.38% | 0.0779 | 0.0031 | 6.98% | 73.8% |

The abstention rate falls by about 60% under the head at an almost identical
mention count. That is the mechanism working, not breaking: the label-filter
path hands the ranker ~370 spans that are not places, the ranker correctly
refuses most of them, and `p_no_match` is doing detection's job after the fact.
Give it real spans and the refusals go away. The ninth decile of `p_no_match`
moves 0.599 → 0.057 (e29) and 0.842 → 0.151 (e54).

**The consequence for a caller is real**: anyone thresholding `p_no_match` at a
fixed value inherits a much sparser tail and should re-pick the threshold. A
proper recalibration (temperature refit on the head's mention population) was
not run and is not blocking — the ranker's inputs are unchanged, only the
population of questions asked of it.

---

## 8. Limits

1. **The N1 gate is open and this report does not close it.** Every number in
   §3–§7 is on held-out *documents* of TR-News / LGL / GeoWebNews, which are the
   corpora the head trained on. It learns those corpora's span conventions —
   GWN's `Non_Literal_Modifier` policy, LGL's institution names — and 87.62
   detection F1 on their held-out documents is not a claim about a wire feed.
   §11 measures the cross-family drop with the three families on disk (87.6
   in-family → 80.8–84.5 out-of-family, and a tie against the spaCy path on the
   hardest fold), which is the closest thing available. The gate stands as e55
   and the scoping report wrote it: reproduce on D1's untouched modern-news TEST
   corpus first.
2. **One head seed.** e55's three gold-head seeds give 74.81 / 77.02 / 77.26
   e2e; seed 42 is packaged and is the 77.26. Every cell in §3 inherits that
   spread. The recommendation is robust to it — the worst head seed still beats
   no head by ~8 EM — but 80.85 is not a point estimate, and a 5-seed head
   ensemble was not measured.
3. **One ranker seed per ranker.** e29 seed101 vs seed42 differ by 0.43–0.67 EM
   on the same detector, which is the familiar ±0.01 exact-match seed noise.
4. **The corpora's own annotation is the ceiling.** 22 golds in the best cell
   are resolved correctly from the head's span and incorrectly from the gold
   span; that is the annotation, not the model.

---

## 9. Recommended default configuration

The measurement supports one recommendation and the flip is the owner's.
Spelled as the exact constructor state:

```python
# RECOMMENDED once the N1 TEST-corpus gate passes -- NOT applied here
Geoparser(
    model_path=None,          # -> DEFAULT_MODEL_ASSET
    span_detector="gold",     # today: None
    trim_spans=True,          # inert while span_detector is set
    nested_gazetteer_pass=False,
    include_fac=True,
    temperature=1.10,         # today: 0.874 -- see §12; not a one-line flip,
)                             # it is a per-configuration choice, and 0.874
                              # remains right for the oracle-span frame
# mordecai3/geoparse.py
DEFAULT_MODEL_ASSET = "assets/mordecai_2026-08-20_e54_seed42.pt"   # today: ..._seed101.pt
```

That is **two one-line flips**, and they are independent:

| # | flip | worth | staged or applied | blocked on |
|---|---|---|---|---|
| 1 | `DEFAULT_MODEL_ASSET` → `assets/mordecai_2026-08-20_e54_seed42.pt` | +0.29 EM with no outlet supplied, +2.7 with outlets, and it is what makes `outlet=` do anything | **staged** by e54, asset packaged, sidecar self-describing | nothing measured; owner's call (e54 §8) |
| 2 | `span_detector` default `None` → `"gold"` | **+10.0 to +10.6 EM** in-family, −0.6 to +2.9 EM out-of-family (§11b), −7.2 ms/doc, −60% abstentions | **staged** here, assets packaged, flag present and tested | **the N1 TEST-corpus gate** — narrowed to amber by §11, not closed |

and one documented default that is **not** a flip: under `span_detector="gold"`,
`temperature=1.10` and a `p ≥ 0.5` filter are the measured operating point
(§12). The temperature is a constructor argument, not a module constant, and
0.874 stays correct for the oracle-span frame the campaign's calibration report
lives in — so this is a documentation change plus a recommended argument, and it
must move at the same time as flip 2 or not at all.

Both applied: **80.85 e2e EM with outlets supplied, 78.55 without**, from 67.66
today. Flip 2 alone on today's default checkpoint: **77.69**.

Nothing in this report has been applied. `DEFAULT_MODEL_ASSET` is still
`assets/mordecai_2026-08-20_seed101.pt` and `span_detector` still defaults to
`None`; the only changes to shipped behaviour are additive (a new module, two
new assets, two new constructor arguments, one new variant pair in the harness),
and §2a shows the default path is bit-identical.

**If the owner wants `all` instead of `gold`** — one word, same flip — the trade
is −0.72 EM for 25 fewer demonym false positives (56 vs 81, against the label
filter's 59) and +8.6 nested detection recall. That is the right call for a
deployment where a demonym resolved to a country is a visible error and pooled
exact match is not the objective. It is the wrong call on this report's metric.

**Do not flip `nested_gazetteer_pass`.** It is what the head replaces: 60.2
ms/doc against 37.6, an Elasticsearch round trip per document, and detection
precision it does not recover. With `span_detector` set it is ignored anyway.

---

## 10. Surprises

1. **The head is faster than the path it replaces, not slower.** It was budgeted
   at +5.2 ms/doc against a pass costing 8.7. It came in at −7.2 ms/doc against
   the *default* path, because 431 spans per 260 documents that are not gold
   toponyms were each buying a failed Elasticsearch search and a fuzzy retry.
   Nobody had costed the label filter's precision in ES traffic.
2. **The head and the outlet block are additive to within 0.19 EM.** Two
   independent arms, one fixing spans and one fixing disambiguation, and there
   was no reason in advance to expect them not to overlap.
3. **C_all loses end to end** despite winning the two metrics e55 argued it
   should be judged on. Nested recall is a small denominator and nested
   toponyms resolve worse once found; 1.3 points of flat F1 is a bigger number
   than it looks.
4. **Abstentions fall by 60%.** `p_no_match` was silently doing detection's job:
   the reserved row was mostly answering "that is not a place" about spans the
   label filter should never have proposed.
5. **97% of the remaining gap is still detection, even with the head on.** The
   ladder's premise survives its own first step. And of the golds the head does
   find, exactly four are ones the ranker resolves correctly from the gold span
   and wrongly from the head's — there is no hidden cost to changing the span
   source.
6. **`retrieval_miss` went up, 88 → 112.** The head's new toponyms are nested in
   organisation names and the gazetteer query is worst at exactly those. The
   next lever on this ladder is retrieval, not ranking.
7. **Nested-toponym detection is the part that generalises.** On the fold where
   flat F1 ties the spaCy path, nested recall is still 43.9 against 23.2 (§11a),
   and on the other two folds it is 46.1/65.9 against 2.1/13.0. Span convention
   is what does not transfer; "there is a place name inside this organisation
   name" does.
8. **The demonym false-positive column is a GeoWebNews column.** TR 0, LGL 4,
   GWN 77 for the shipped head — and the GWN count is the same in-family (77)
   and out-of-family (77.7). The gold-vs-C_all demonym argument in §4 is
   entirely an argument about one corpus's annotation policy.
9. **The serving-frame temperature is above 1, not below.** T = 0.874 sharpens a
   model that is under-confident on gold spans; on real spans, with a fifth of
   mentions unanswerable, it is over-confident, and applying 0.874 roughly
   doubles ECE against the refit (§12).

---

## 11. Leave-one-corpus-out: the N1 gate, made cheap

§8's first limit is the one that blocks the flip: the head trained on TR-News +
LGL + GeoWebNews and every detection number above is on held-out *documents* of
those same three corpora. D1's modern-news TEST corpus does not exist yet, but
the three corpora on disk are three different families, so the cross-family read
can be taken now: **for each corpus, train the head on the other two and score
detection on the excluded corpus's held-out documents.**

`experiments/e56_span_head_serving/loco.py`, e55's training harness, 3 seeds per
fold, 20 epochs, lr 1e-3 — the packaged recipe. Two arms: the gold-only recipe
that produced the shipped head, and the demonym-negative w10 variant. **The
threshold is picked on the two *training* families' dev split and never on the
excluded corpus**, which is what makes this a gate rather than a tuning
exercise. Runs are deterministic in (config, seed) — the GWN fold was retrained
to save checkpoints for §11b and reproduced all three seeds' F1 exactly.

The bar is not the in-family 87.6. It is **the spaCy label-filter path on that
same corpus**: if an out-of-family head still beats what ships today on a corpus
it has never seen, the flip's premise survives.

### 11a. Detection

| arm | held-out corpus | out-of-family F1 (3 seeds) | Δ vs spaCy path | t(2) | spaCy path | in-family head |
|---|---|---|---|---|---|---|
| gold | TR-News | 83.21 ± 0.58 | **+5.86** | 17.65 \* | 77.35 | 83.87 |
| gold | LGL | 84.52 ± 1.20 | **+9.28** | 13.42 \* | 75.24 | 90.31 |
| gold | GeoWebNews | 80.81 ± 1.08 | +0.51 | 0.82 **ns** | 80.30 | 83.71 |
| dem10 | TR-News | 83.07 ± 0.71 | **+5.72** | 14.00 \* | 77.35 | 83.87 |
| dem10 | LGL | 81.11 ± 1.46 | **+5.87** | 6.98 \* | 75.24 | 90.31 |
| dem10 | GeoWebNews | 81.15 ± 0.94 | +0.85 | 1.58 **ns** | 80.30 | 83.71 |

`*` = |t| > 4.303 (3 seeds). Nested detection recall and demonym false
positives, gold arm, against the same two references:

| corpus | nested R out-of-family | nested R in-family | nested R spaCy | demonym FP out-of-family | in-family | spaCy |
|---|---|---|---|---|---|---|
| TR-News | 46.1 | 51.1 | 2.1 | 0.3 | 0 | 0 |
| LGL | 65.9 | 74.3 | 13.0 | 2.3 | 4 | 0 |
| GeoWebNews | 43.9 | 56.1 | 23.2 | 77.7 | 77 | 59 |

**No fold falls below the spaCy path.** Two of three beat it decisively, and
the third — GeoWebNews — is **statistically indistinguishable from it on flat
F1** (+0.51, t = 0.82). Read plainly: the gate is amber, not green and not red.

Three things are worth separating out of that GWN row, because they are not the
same failure:

* **The nested-toponym win survives everywhere.** Even on the fold that ties on
  F1, nested detection recall is 43.9 against the spaCy path's 23.2, and on the
  other two folds it is 46.1/65.9 against 2.1/13.0. Whatever else transfers,
  *finding toponyms inside organisation names* transfers.
* **The demonym problem on GWN is not a generalisation failure.** 77.7 demonym
  FPs out-of-family against **77** in-family: the head trained on GWN's own
  documents does no better. Note also that essentially every demonym FP in this
  report is a GWN row (TR 0, LGL 4, GWN 77): TR and LGL annotate demonyms as
  spaCy-NORP spans, which both the label filter and the head reject, while GWN's
  `Non_Literal_Modifier` rows are spans spaCy tags GPE. The demonym-FP column is
  in practice a *GeoWebNews annotation-convention* column.
* **GWN is the corpus the label filter is already good at** (80.30 F1, its best
  of the three) and the one with the most idiosyncratic policy (24.5% of its
  linked golds are D2 demonyms, against 5.7% TR and 8.8% LGL). It is the
  hardest fold by construction, and it is the one that should be hardest.

`dem10` does not rescue anything: it costs 3.4 F1 out-of-family on LGL and buys
0.34 points on GWN. It is not a better generalisation recipe, only a different
demonym operating point.

### 11b. What the worst fold costs end to end

The GWN-fold heads scored through the full pipeline (e54 seed42, GWN only, D2,
523 golds), against the two references on the same corpus and the same run:

| span source | e2e EM |
|---|---|
| spaCy label filter (`serving`) | 74.19 |
| **out-of-family head, 3 seeds** | **75.02 ± 1.81** (74.38 / 77.06 / 73.61) |
| in-family packaged head | **79.35** |

+0.83 EM, t = 0.79, **not significant**, and one of the three seeds (73.61) is
*below* the spaCy path. So on the hardest fold the head is a wash end to end
too, and the honest statement is: **on a corpus family the head has never seen,
it is worth somewhere between −0.6 and +2.9 EM, not the +10.5 it is worth
in-family.** The in-family number is ~5 points of that 10 being corpus-specific
span convention.

### 11c. What this does to the recommendation

It sharpens it rather than blocking it, for three reasons:

1. **The failure mode is a tie, not a regression.** Across six fold-arm cells
   and nine end-to-end seeds, nothing is significantly worse than what ships.
2. **The deployment case is in-family for two of three families.** TR-News and
   LGL are ordinary news wire and local news — the shape of what mordecai3 is
   pointed at — and both transfer at +6 to +9 F1. GWN is a deliberately
   heterogeneous benchmark with its own typed annotation policy.
3. **It is a labelling problem, not a modelling one**, and e55 said so: the head
   is limited by in-domain, complete, nested gold, of which 5,034 exist. The
   LOCO drop from 87.6 in-family to 80.8–84.5 out-of-family is the size of the
   prize for annotating a fourth family, and it is now measured rather than
   guessed.

**The N1 gate is not closed by this.** A real TEST corpus can still fail in ways
three benchmark corpora cannot show — a wire feed's ALL-CAPS datelines, bylines,
sports tables. What changed is that "does it generalise across families at all"
now has an answer, and the answer is "yes on two of three, tie on the third".

---

## 12. Calibration under the head

§7a showed the abstention *rate* moves a lot. This section re-derives the
operating points, because T = 0.874 and the recommended `p ≥ 0.7` policy were
fitted on a different population.

**The frame.** `tools/calibration_eval.py` calibrates on the *oracle-span*
frame: every mention it scores is a gold toponym out of the training pickles,
and the only question is whether the gold id is in the window. The serving frame
is different in kind — the detector chooses the mentions, and 19–29% of them
have no right answer at all. `calibration_refit.py` builds that frame from raw
text over the 260 held-out documents, on e54 seed42 with LGL/TR outlets, and
imports `ece`, `auroc`, `risk_coverage`, `masked_softmax` and the temperature
grid from `calibration_eval.py` so it stays on that lineage. A mention is
**answerable** if its span exactly matches a D2 gold toponym *and* that gold id
is a selectable candidate; an answer on an unanswerable mention is **wrong**,
which is what a user experiences.

Both detectors are run on this same frame, so every comparison below is paired.

| | `span_detector=None` | `span_detector="gold"` |
|---|---|---|
| mentions | 2,159 | 2,173 |
| answerable | 1,541 (71.4%) | **1,752 (80.6%)** |
| unanswerable | 618 | **421** |
| EM answering everything | 67.95 | **77.54** |
| **refit T** | **1.034** | **1.106** |
| ECE at T = 0.874 | 0.1037 | 0.0941 |
| **ECE at refit T** | 0.0689 | **0.0430** |
| AURC (lower better) | 0.1209 | **0.1001** |

**The temperature moves the other way from the campaign's.** T = 0.874 sharpens
the model because on the oracle-span frame it is under-confident. On the serving
frame it is *over*-confident — it answers unanswerable mentions with high
probability — so the fitted T is **above 1** in both columns, and applying 0.874
roughly doubles serving-frame ECE against the refit. Under the head, refitting
takes ECE from 0.094 to **0.043**, which is the best serving-frame calibration in
this report. This does not contradict `calibration_report.md` §5; it is a
different frame, and both are worth stating.

### 12a. Operating points under the head

`span_detector="gold"`, e54 seed42, T = 1.106. Coverage is the share of mentions
answered; selective EM counts an answer on an unanswerable mention as wrong.

| policy | coverage | selective EM | unanswerable caught (of 421) |
|---|---|---|---|
| answer everything | 100% | 77.54% | 0 |
| reserved-argmax flag only | 92.64% | 83.71% | 149 |
| **`p ≥ 0.5` or flag (recommended)** | **87.99%** | **86.51%** | 202 |
| `p ≥ 0.6` or flag | 85.09% | 87.40% | 218 |
| `p ≥ 0.7` or flag (the old default) | 81.41% | 88.36% | 237 |
| `p ≥ 0.8` or flag | 76.99% | 88.88% | 251 |
| `p ≥ 0.9` or flag (precision-first) | 62.95% | 89.99% | 292 |

The same table for the path that ships today, on the same frame, at its own
refit T = 1.034:

| policy | coverage | selective EM | unanswerable caught (of 618) |
|---|---|---|---|
| answer everything | 100% | 67.95% | 0 |
| reserved-argmax flag only | 82.54% | 82.32% | 361 |
| `p ≥ 0.5` or flag | 79.25% | 84.92% | 407 |
| `p ≥ 0.7` or flag | 73.46% | 87.14% | 439 |
| `p ≥ 0.9` or flag | 60.49% | 89.66% | 494 |

**The head dominates at every coverage level.** Risk-coverage, selective EM:

| coverage | none | gold head |
|---|---|---|
| 100% | 67.95 | **77.54** |
| 95% | 71.53 | **81.64** |
| 90% | 75.50 | **85.63** |
| 80% | 84.54 | **88.67** |
| 70% | 88.62 | **89.15** |

**Recommended threshold: `p ≥ 0.5` or the reserved-argmax flag**, at 88.0%
coverage and 86.5% selective EM. The old recipe's exchange rate was about two
points of coverage per point of selective EM; under the head, `flag only → 0.5`
buys +2.80 EM for 4.65 coverage and `0.5 → 0.6` buys only +0.89 for 2.90, so the
knee is at 0.5. The head's curve is much flatter above 0.6 than the old path's,
which is exactly what a higher-precision detector should do: there is less junk
left for the threshold to remove. `p ≥ 0.7` remains the right choice for a
precision-first deployment, now at 81.4% coverage rather than 90.6%.

### 12b. Abstention quality did degrade, on the ranking statistic

This was a first-class campaign goal, so it is reported plainly.

| AUROC | none | gold head | Δ |
|---|---|---|---|
| flagging a wrong answer, `p_pred_full` | 0.8810 | 0.8072 | **−0.0738** |
| flagging a wrong answer, `p_no_match` | 0.8786 | 0.7848 | **−0.0938** |
| detecting an unanswerable mention, `p_no_match` | 0.8945 | 0.7967 | **−0.0978** |

Both scores get materially worse at *ranking* errors under the head. The reading
is population, not model: the label filter's 618 unanswerable mentions include
~370 spans that are not toponyms at all, which the ranker flags trivially. The
head removes those, and the 421 that remain are real toponyms whose gold id was
not retrieved, or nested mentions with genuinely hard candidate sets — a harder
discrimination problem on which any score does worse.

**Absolute selective performance is nonetheless better everywhere** (the AURC
and risk-coverage rows above), so a user filtering by `p` gets more correct
answers per unit of coverage with the head than without. But two things follow
that a deployment has to act on:

* **`p_no_match` is no longer a good unanswerability detector on its own**
  (0.797 against 0.895). A pipeline that used it as a "there is no place here"
  signal is losing the part of that signal that detection now provides
  structurally — which is the right place for it, but it has moved.
* **A fixed threshold carried over from the old path is wrong in both
  directions**: the same nominal `p ≥ 0.7` now costs 9 points more coverage,
  and the reserved-argmax flag alone now covers 92.6% instead of 82.5%.

A full recalibration study on the serving frame — LOSO temperature across the
three corpora, the logistic combination score, and the 5-seed ensemble — was not
run and is not blocking. What is in front of the owner is one number (T ≈ 1.1
under the head) and one threshold (`p ≥ 0.5`), both measured on this frame.

---

## Appendix: files

| file | what |
|---|---|
| `mordecai3/span_head.py` | the inference module + asset resolution |
| `mordecai3/assets/span_head_2026-08-20_{gold,all}.pt` | the two heads, unchanged from e55 |
| `tests/test_span_head.py` | 15 cases |
| `experiments/e56_span_head_serving/parity_span_head.py` | gate 2c; e55's scorer vendored |
| `experiments/e56_span_head_serving/detection_grid.py` | §3b, all three detectors on one scorer |
| `experiments/e56_span_head_serving/identity_check.py` | gate 2a |
| `experiments/e56_span_head_serving/run_grid.sh` | the four harness runs, ~20 min |
| `experiments/e56_span_head_serving/aggregate.py` | pooling, §3a, §7 decomposition |
| `experiments/e56_span_head_serving/serving_probe.py` | §5 latency, §7a abstention |
| `experiments/e56_span_head_serving/loco.py` | §11, leave-one-corpus-out (18 runs, ~10 min) |
| `experiments/e56_span_head_serving/calibration_refit.py` | §12, serving-frame calibration |
| `experiments/e56_span_head_serving/{loco,calibration_refit}.json` | their raw output |
| `experiments/e56_span_head_serving/e2e/*.json` | raw harness output |
| `experiments/e56_span_head_serving/data/*_docs.json` | e55's gold-span dumps, copied so 2c stays runnable |
