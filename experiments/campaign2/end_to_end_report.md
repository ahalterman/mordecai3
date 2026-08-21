# End-to-end evaluation: what 92.6% exact match means once spaCy picks the spans

Second campaign, item 6 of `experiments/PLAN.md`. Written 2026-08-20.
Harness: `tools/end_to_end_eval.py`. Raw results:
`experiments/campaign2/e2e_heldout{,2,3,4}.json` (variant sweeps; `_heldout3`
also carries the output-precision counters).

## Headline

The accuracy campaign's 92.6% exact match is a number about a **subset of gold
toponyms, with the answer span handed to the model**. On the same held-out
documents, run end to end from raw text, the pipeline gets **58.6%** of gold
toponyms detected *and* resolved correctly.

| what is being measured | pooled EM | share of gold toponyms it covers |
|---|---|---|
| campaign metric (gold span given, spaCy tagged it GPE/LOC, gold retrievable) | **91.2%** | 67.9% (1,616 / 2,381) |
| gold span given, spaCy tagged it (drop the retrievability filter) | 86.3% | 71.7% |
| gold span given, *every* gold toponym (oracle NER) | 80.1% | 100% |
| **end to end: spaCy finds the span, model resolves it** | **58.6%** | 100% |

Pooled over the 2,381 held-out gold toponyms of TR-News + LGL + GWN
(260 documents). The reproduction check: this harness's campaign-parity number
is 90.5 / 91.0 / 91.8 on TR / LGL / GWN against the campaign's own
`e29_swa_ep15` seed-42 held-out scores of 89.7 / 90.3 / 92.8 — same measurement,
independently recomputed through the serving path at `max_choices=100`.

Two things cause the 32-point drop, in this order:

1. **The pickles silently dropped 28% of the gold toponyms.** `data_formatter`
   in `tools/train.py` keeps a gold toponym only if spaCy put a GPE or LOC on
   one of its tokens (`if not gpes: continue`), and `error_utils.evaluate_results`
   then skips any entity whose gold id is not in the candidate list. Neither
   filter is applied to the model's *predictions* — they are applied to the
   *test set*. Every toponym spaCy cannot see was removed from the exam.
2. **NER recall is genuinely the binding constraint, not the ranker.**
   Detection recall (exact span match) is 68.1% pooled; the ranker, given a
   correctly detected span, is right about 86% of the time.

The good news for the next campaign: **cheap, purely-configuration NER fixes
recover +14.3 points end to end** (58.6 → 72.9), at ~20% lower throughput and
no retraining of anything. If that arm's precision cost is unacceptable, a
conservative subset (trim spaCy's span boundaries, accept demonyms) is
**+6.3 points at unchanged precision and ~7% throughput**.

## 1. Method

`tools/end_to_end_eval.py` runs the real serving path — `nlp.pipe` →
`doc_to_ex_expanded` → `add_es_data_batch` → `ProductionData` → the ranker,
decoded exactly as `Geoparser._resolve_results` decodes it — over the raw
`<text>` of each corpus's held-out documents, then aligns predicted character
spans to the gold spans two ways (exact, and one-to-one greedy by largest
character overlap).

*Model*: `experiments/e29_swa_ep15/seed42.pt`, the actual ship recipe, loaded
as `Geoparser(feature_blocks="prom,name,cue,sib,geo,shape", oov_bucket_fix=True,
model_options={"return_logits": True, "mask_padding": True, "modern_mlp": True})`,
`max_choices=100`. **Note for the PI:** when this evaluation started, the
checkpoint at the repo root, `mordecai_2026-08-20.pt`, was byte-identical to
`experiments/e24_rstar/seed1848.pt` — the *rejected* R\*\* label-rewrite arm,
sidecar `"pickle_suffix": "_r2"`. It was replaced later the same day with a
file whose sidecar reads `"pickle_suffix": "", "weight_avg": "swa"`, i.e. the
real ship recipe. Nothing here depends on it (all runs use
`experiments/e29_swa_ep15/seed42.pt` explicitly), but the root artefact is
worth a checksum before anyone ships from it.

*Held-out documents*: `tools/train.py` splits each source's flat entity list
positionally at 70%, so the split is a document boundary up to the one article
that straddles it. `heldout_doc_indices` recovers documents from the pickle by
grouping consecutive entities with an identical `doc_tensor`, maps each group
back to its article by matching the group's `search_name` sequence against the
article's gold phrases, and keeps only articles entirely on the held-out side
(the straddling article is dropped: TR doc 88, LGL 402, GWN 138). Result:
TR 28 docs / 351 gold, LGL 175 / 1,348, GWN 57 / 693.

*Gold set*: every `<toponym>` with a non-empty geonames id. Rows annotated
without one (GeoWebNews annotates 4,211 of them — "church", "the building",
literal expressions) are excluded from recall, and a predicted span landing on
one is reported separately rather than as a hallucination.

*acc@161* uses the gold coordinates from the corpus XML, so it is defined even
when the gold id was never retrieved.

## 2. End-to-end results (ship configuration)

| corpus | det P | det R | det F1 | det F1 (overlap) | **E2E EM** | E2E acc@161 | oracle-span EM | campaign-parity EM |
|---|---|---|---|---|---|---|---|---|
| TR-News | 78.9 | 71.5 | 75.0 | 86.4 | **61.3** | 63.5 | 79.5 | 90.5 |
| LGL | 70.6 | 68.0 | 69.3 | 76.3 | **57.2** | 59.0 | 82.4 | 91.0 |
| GWN | 85.1 | 66.7 | 74.8 | 79.0 | **60.0** | 62.5 | 76.0 | 91.8 |
| pooled | 75.5 | 68.1 | 71.6 | — | **58.6** | 60.7 | 80.1 | 91.2 |

Allowing overlap instead of exact span match for the end-to-end credit moves
EM to 70.1 / 59.4 / 62.1 — i.e. on TR-News nearly nine points of the gap is
boundary disagreement where the resolver was right anyway.

### Spurious spans (what a user sees as a wrong location)

| corpus | predicted spans | overlapping no gold row at all | of those, resolved | overlapping an unlinked gold row |
|---|---|---|---|---|
| TR-News | 318 | 18 | 4 | 11 |
| LGL | 1,299 | 152 | 48 | 137 |
| GWN | 543 | 12 | 6 | 43 |

Most spurious spans are FAC entities the ranker then declines (predicts the
NULL candidate) — the abstention path is doing real work here. The ones that do
resolve are mostly facilities matched to a same-named facility row
(`Vandenberg Air Force Base`, `Springfield Mall Regional Shopping Center`) plus
a handful of genuine errors (`Aisles` → Isle of Man; `and` → United Kingdom;
`Complete Streets` → 26 July Street, Egypt).

## 3. Decomposition of every gold toponym

Percentages of all gold toponyms in the held-out documents; counts in
parentheses. `boundary_ok` = spaCy's span overlapped the gold span but was not
identical, and the resolver still returned the right geonameid.

| corpus | n gold | correct | boundary, resolved ok | boundary, resolved wrong | NER miss | retrieval miss | ranker error | model abstained |
|---|---|---|---|---|---|---|---|---|
| TR-News | 351 | 61.3 (215) | 8.8 (31) | 2.0 (7) | **17.7 (62)** | 4.3 (15) | 5.7 (20) | 0.3 (1) |
| LGL | 1,348 | 57.2 (771) | 2.2 (30) | 4.7 (63) | **25.1 (338)** | 4.2 (56) | 6.0 (81) | 0.7 (9) |
| GWN | 693 | 60.0 (416) | 2.0 (14) | 1.7 (12) | **29.6 (205)** | 2.5 (17) | 3.3 (23) | 0.9 (6) |

Read the row as: of everything that does not come out exactly right, NER
contributes **74% (TR), 75% (LGL), 83% (GWN)** and the ranker the rest. Even
after forgiving the boundary cases the resolver survived — i.e. counting only
outright misses and boundary errors that changed the answer — NER still owns
51% / 70% / 78% of the loss. Boundary errors are recoverable more often
than not on TR (31 of 38) and much less often on LGL (30 of 93), because LGL's
boundary errors are "Paris" vs "Paris City Hall" rather than "United States" vs
"the United States".

Retrieval at `max_choices=100` finds the gold id for 94.7% of the spaCy-tagged
gold spans (campaign figure: 98.5% at 500 candidates) and 88.7% of *all* gold
spans — the difference is again the toponyms spaCy never surfaced, which are
also the harder strings to retrieve.

## 4. What the NER misses actually are

Every missed gold toponym, classified by what spaCy did at that exact span:

| corpus | misses | ORG | NORP (demonym) | untagged | EVENT/LAW/WORK_OF_ART/… | offsets don't align |
|---|---|---|---|---|---|---|
| TR-News | 62 | 35 | 20 | 0 | 3 | 4 |
| LGL | 338 | 191 | 118 | 12 | 17 | 0 |
| GWN | 205 | 101 | 77 | 8 | 19 | 0 |

and by *kind* of failure, which is the actionable split:

| corpus | nested inside a bigger entity | same span, discarded label | no entity there at all |
|---|---|---|---|
| TR-News | 30 | 28 | 4 |
| LGL | 194 | 132 | 12 |
| GWN | 90 | 107 | 8 |
| total | 314 (52%) | 267 (44%) | 24 (4%) |

**Only 24 of the 605 misses (4%) are cases where spaCy found nothing.** The
en_core_web_trf NER is not failing to see these strings; the pipeline is
throwing them away. Two classes dominate:

* **Toponyms nested in an organisation name** (314 misses, 52%): the gold
  annotation marks `Paris` inside "Paris Police Department", `Pennsylvania`
  inside "University of Pennsylvania", `Montana` inside "Montana Department of
  Corrections", `Laurel County` inside "Laurel County Sheriff's Office". spaCy
  emits one ORG span, `doc_to_ex_expanded` skips ORG, and the toponym is gone.
  Accepting ORG as a geoparse label does **not** fix this — the span would be
  "Paris Police Department", still not the gold span — which is exactly what
  the measurement below shows.
* **Demonyms** (215 misses, 36%; 206 of them a whole spaCy NORP span):
  "Turkish", "Palestinian", "Russian",
  "Tunisian", "Chinese". `doc_to_ex_expanded` deliberately refuses to geoparse
  NORP ("NORPs are useful for context, but we don't want to geoparse them"),
  while all three corpora annotate them with the country's geonameid. That is a
  policy disagreement with the evaluation data, not a detection failure — but
  it is a policy the downstream user may well not want, and it costs 9% of gold.

The classic NER-robustness suspects are **not** the problem here: ALL-CAPS text
occurs in exactly 1 of 906 documents (case normalisation changes literally
nothing — see `truecase` below), lowercase mentions are absent, and non-Western
names are detected at the same rate as anything else. The residual "no entity
there" bucket is 24 items, mostly abbreviations in datelines ("LOUISVILLE, KY--")
and possessive/compound oddities.

### GeoWebNews by annotation type

GWN labels each toponym's usage, and it explains most of its gap:

| type | n | detection R | E2E EM |
|---|---|---|---|
| Literal | 220 | 93.6 | **80.9** |
| Metonymic ("Zimbabwe said…") | 118 | 88.1 | 88.1 |
| Mixed | 77 | 83.1 | 68.8 |
| Literal_Modifier | 46 | 60.9 | 54.3 |
| Non_Literal_Modifier (demonyms) | 153 | 34.6 | 32.7 |
| Embedded_Non_Lit (inside org names) | 65 | 4.6 | 4.6 |
| Coercion / Embedded_Literal | 14 | 28.6 | 21.4 |

On plain literal place mentions the shipped pipeline is at **80.9% end to end**.
Whether the other 68% of GWN's gold set is in scope is a product decision, and
the honest way to publish an end-to-end number is to state which usage types it
includes. TR-News and LGL do not type their toponyms, so their numbers mix all
of these together.

## 5. Cheap fixes, measured

All measured on the same held-out documents, same model, same decode. Pooled
over TR+LGL+GWN (n = 2,392 gold; binomial SE on EM ≈ 1.0 point, and these are
paired comparisons on identical documents so the deltas are much tighter than
that). `docs/s` is the median over the three corpora's own runs.

| variant | what changed | det P | det R | det F1 | **E2E EM** | Δ EM | resolved spurious spans |
|---|---|---|---|---|---|---|---|
| **ship** | as deployed | 75.5 | 68.1 | 71.6 | **58.6** | — | 58 |
| truecase | down-case ALL-CAPS lines | 75.5 | 68.1 | 71.6 | 58.6 | +0.0 | 58 |
| no_fac | drop FAC from the geoparse labels | 87.1 | 67.0 | 75.7 | 57.9 | −0.7 | 31 |
| **trim** | strip leading "the/a" and trailing "'s" from spans | 78.2 | 70.6 | 74.2 | **60.7** | **+2.0** | 63 |
| trim_nofac | trim + no FAC | 89.9 | 69.1 | 78.2 | 59.9 | +1.3 | 33 |
| labels_org | also geoparse ORG spans | 45.6 | 70.6 | 55.4 | 58.9 | +0.3 | 334 |
| **labels_norp** | also geoparse demonyms | 72.6 | 76.8 | 74.6 | **62.8** | **+4.2** | 116 |
| labels_all | ORG + NORP + EVENT + LAW + WORK_OF_ART | 44.6 | 79.2 | 57.1 | 63.2 | +4.6 | 427 |
| **trim_norp** | trim + demonyms (the conservative pick) | 75.0 | 79.2 | **77.1** | **64.9** | **+6.3** | 123 |
| gaz_out | gazetteer pass outside entities (multi-token only) | 72.4 | 70.4 | 71.4 | 60.3 | +1.7 | 110 |
| **gaz_nested** | gazetteer pass *inside* ORG/FAC/… entities | 67.0 | 77.9 | 72.0 | **66.5** | **+7.9** | 297 |
| gaz_both | both gazetteer rules | 65.1 | 76.9 | 70.5 | 65.8 | +7.1 | 315 |
| lg | en_core_web_lg for spans (trf still for tensors) | 76.3 | 63.0 | 69.0 | 54.3 | −4.3 | 63 |
| lg_gaz | lg + both gazetteer rules | 64.9 | 74.5 | 69.4 | 63.8 | +5.2 | 346 |
| **combo_best** | trim + NORP + nested gazetteer | 67.6 | **89.0** | **76.8** | **72.9** | **+14.3** | 365 |

Per-corpus, `combo_best` reads TR 66.4 (+5.1), LGL 75.6 (+18.4), GWN 70.9
(+10.8); its detection recall on LGL is 91.4%.

### (a) Accepting more entity labels

**ORG is a trap.** It raises detection recall by 2.5 points and end-to-end EM by
0.3, while detection precision collapses from 75.5 to 45.6 and the number of
resolved spurious spans goes from 58 to 334 — the pipeline starts confidently
geolocating "the Spina Bifida Association of Western Pennsylvania". It also
costs 69% of throughput, because every organisation name becomes an ES query.
Do not do this.

**NORP is the single best label change**: +4.2 EM pooled (LGL +5.4, GWN +3.8,
TR +0.3) for a 2.9-point precision cost and ~7% of throughput. It works because
geonames' `alternativenames` carry the adjectival forms, so "Palestinian" and
"Russian" retrieve the right country without any new machinery. The caveat is
that it *is* a semantic choice: it makes the geoparser report a location for
"the Turkish president", which some downstream users want and others do not.
Recommend making it a `Geoparser(..., geoparse_demonyms=False)` flag rather than
a silent default.

### (b) Case normalisation — nothing here

Exactly 1 of 906 documents in these corpora contains an ALL-CAPS line. The
variant is a bit-for-bit no-op on all three corpora. If headline/dateline text
matters for a deployment it has to be shown with a corpus that actually
contains it; these do not.

### (c) en_core_web_lg vs en_core_web_trf

Not a real option, for two reasons. First, the ranker consumes
en_core_web_trf's 768-dimensional token tensors, so `lg` can only ever be an
*extra* pass, never a replacement — there is no latency to save. Second, on
spans alone `lg` is worse: pooled detection recall 63.0 vs 68.1, end-to-end EM
54.3 vs 58.6 (LGL −5.6, GWN −4.2). Measured cost of the extra pass:
see the latency table.

### (d) Gazetteer second pass — the one that pays

Two bounded rules, both requiring a geonames entry whose own `name` equals the
candidate string exactly:

* *outside*: runs of 2–4 capitalised alphabetic tokens no entity covers. Worth
  +1.7 EM.
* *nested*: sub-spans of 1–3 tokens **inside** ORG/FAC/EVENT/WORK_OF_ART/LAW
  entities, leftmost-longest, restricted to feature class A/P (populated places
  and admin units). Worth **+7.9 EM** — it is aimed directly at the dominant
  miss class and it hits it: LGL detection recall 68.0 → 80.6.

The cost is precision: 75.5 → 67.0 pooled, and resolved spurious spans 58 → 297.
Many of those "spurious" spans are arguably not wrong — "Springfield" out
of "Springfield Mall" is a real place — but they are extra output the user did
not ask for. Combining the two rules is slightly *worse* than nested alone on
LGL (a long outside-match can block a nested one), so ship `nested` only.

### (e) Span trimming — free

Stripping a leading determiner and a trailing possessive from spaCy's spans
("the United States" → "United States", "New Mexico's" → "New Mexico") is +2.0
EM pooled and +2.6 detection F1 at zero measurable cost — it converts boundary
matches into exact ones and improves the ES query at the same time. This is a
five-line change to `doc_to_ex_expanded` and should land regardless of anything
else in this report.

## 6. Latency

Steady state on the RTX 4090, candidate cache cleared before every repetition,
median of 3, LGL held-out documents (short local-news articles, ~6 entities
each).

```
lgl: 50 documents
  variant        spaCy extra NER  gazetteer      ES   model   total   docs/s  entities
  ship            0.83      0.00       0.00    1.53    0.06    2.42    20.66       309
  trim            0.81      0.00       0.00    1.42    0.06    2.30    21.73       309
  no_fac          0.83      0.00       0.00    0.81    0.05    1.69    29.55       241
  labels_norp     0.82      0.00       0.00    1.71    0.06    2.59    19.28       341
  gaz_nested      0.81      0.00       0.27    1.80    0.09    2.97    16.83       442
  combo_best      0.83      0.00       0.34    1.82    0.10    3.09    16.18       473
  labels_org      0.82      0.00       0.00    6.79    0.10    7.70     6.49       560
  lg              0.83      0.43       0.00    1.66    0.06    2.98    16.80       274
```

Baseline throughput on these corpora is **12.7 docs/s (TR-News, long articles),
18.6 (LGL), 16.6 (GWN)**; the 20.7 above is the same LGL documents in a tighter
loop. Where the time goes in the shipped configuration: **spaCy 29–38%,
Elasticsearch 57–65%, the ranker 2–3%**. The transformer NER is not the
bottleneck — the candidate lookups are — which is why the fixes that add
*entities* cost far more than the fixes that add *model work* (none of them do).

Cost of each fix, as throughput: `trim` **free** (slightly faster, shorter
queries), `labels_norp` **−7%**, `gaz_nested` **−19%**, `combo_best` **−22%**,
`lg` **−19%** (a second NER pass, +0.43 s/50 docs, for worse recall),
`labels_org` **−69%** (560 entities instead of 309, and organisation names are
expensive queries). Dropping FAC makes the pipeline **43% faster**, because FAC
spans are numerous and rarely resolve.

### Output precision: what the user actually receives

Of the locations the pipeline emits (a span it resolved to a geonames id), how
many are a real gold toponym resolved correctly:

| variant | locations emitted | correct (exact span + id) | correct allowing boundary slack |
|---|---|---|---|
| ship | 1,781 | 78.7% | 82.9% |
| no_fac | 1,701 | 81.5% | 85.7% |
| trim | 1,817 | 79.9% | 82.3% |
| labels_norp | 1,978 | 75.9% | 79.8% |
| trim_norp | 2,018 | 76.9% | — |
| gaz_nested | 2,276 | 69.9% | 73.4% |
| combo_best | 2,518 | 69.2% | 71.2% |

In absolute terms, over these 260 documents `combo_best` returns **1,742
correct locations against ship's 1,402** (+24%), while wrong locations go from
379 to 776 (+105%). That is the trade to put to the product: markedly more
coverage, materially more noise. `trim` alone is strictly better on both axes
(1,452 correct / 365 wrong), and `trim_norp` returns 1,552 correct for 466
wrong -- +11% correct locations for +23% wrong ones.

## 7. What a dedicated NER campaign could plausibly buy

The ceiling is set by the oracle-span number: if detection were perfect at the
corpora's own span conventions, end-to-end EM would be **80.1%** pooled
(TR 79.5, LGL 82.4, GWN 76.0). That is 21.5 points above where the pipeline is
now, and it is the entire prize.

Against that ceiling:

* **Configuration alone gets ~2/3 of the way: +14.3 points (58.6 → 72.9), today,
  with no training.** Trimmed spans, demonyms as a flag, and a nested-gazetteer
  second pass. Cost: detection precision 75.5 → 67.6 and ~20% throughput.
  If precision matters, the conservative pick is `trim` + `labels_norp`:
  **+6.3 points (58.6 → 64.9) at detection precision 75.0 vs today's 75.5** --
  i.e. eleven points of recall for nothing -- and the best detection F1 of any
  arm tried (77.1). `trim` alone is +2.0 EM with precision *better* than today
  and no latency cost at all.
* **The remaining ~7 points are a real annotation problem, not a model
  problem.** They are toponyms inside organisation names that the gazetteer
  rule cannot disambiguate ("Sears" is a place; "Sears" in "Sears Holdings" is
  not), plus the 24 genuine NER blind spots. Buying those needs a *nested*
  toponym tagger — a token-level place-name model trained on TR/LGL/GWN spans
  themselves, which all three corpora provide for free — rather than
  off-the-shelf ontonotes NER. A small token-classification head over the trf
  tensors the pipeline already computes would cost nothing extra at inference,
  since the transformer forward pass is already paid for.
* **Do not spend the campaign on the ranker.** Given a correctly detected span,
  the ranker is right 86% of the time and the campaign already moved it 4.5
  points at considerable effort. The same 4.5 points are available from NER
  configuration in an afternoon, and three times that from a nested tagger.

### Evaluation debt this uncovers

1. The reported campaign numbers are conditioned on spaCy having found the
   mention. Any future report should state the conditioning explicitly, or
   report the oracle-span number over *all* gold toponyms (80.1%), which is the
   honest "how good is the resolver" figure.
2. `data_formatter`'s `if not gpes: continue` filters the *training and test*
   data with the model under test's own front end. It should at minimum be
   logged (it drops 361 / 1,217 / 821 gold toponyms from TR / LGL / GWN
   respectively before anything is trained).
3. `doc_to_ex_expanded` geoparses FAC but excludes it from the context tensor,
   and excludes NORP from geoparsing but includes it in context. Both are
   defensible; neither has ever been measured. FAC costs 11.6 points of
   detection precision for 1.1 points of recall.

## Reproducing

```
uv run python tools/end_to_end_eval.py evaluate \
  --model-path experiments/e29_swa_ep15/seed42.pt \
  --sources tr,lgl,gwn \
  --variants ship,no_fac,trim,trim_norp,labels_norp,gaz_nested,combo_best \
  --out experiments/campaign2/e2e_heldout3.json

uv run python tools/end_to_end_eval.py ner-errors experiments/campaign2/e2e_heldout3.json
uv run python tools/end_to_end_eval.py latency --n-docs 50 \
  --variants ship,trim,labels_norp,gaz_nested,combo_best,labels_org,lg
```
