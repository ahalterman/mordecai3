# R1 in the mainline: abbreviation normalisation, and what it is worth on top of the new stack (e57_r1_retrieval)

Campaign 2. e52 measured four gazetteer-hygiene rules against a frozen ranker
and recommended exactly one — **R1**, expanding a US-state / Canadian-province
abbreviation into the full ADM1 name *before* the Elasticsearch query. e56 then
put the place-span head into the serving path, took end-to-end exact match from
66.99 to 80.85, and ended with a surprise: `retrieval_miss` went **up**, 88 →
112, because the toponyms the head newly finds are nested in organisation names
and the gazetteer query is worst at exactly those. Its §10.6 named retrieval as
the next lever.

This report lands R1 in `mordecai3/` behind a flag and measures it against that
stack. Ledger: `experiments/e57_r1_retrieval/NOTES.md`.

**Verdict: ADOPT, default ON. The best serving cell moves 80.85 → 82.29 e2e EM
(+1.44, +30 golds, 0 lost); `retrieval_miss` under the head falls 112 → 84
(−25%); the oracle-span ceiling itself rises 87.41 → 89.05. R1 is worth MORE
under the head than under spaCy spans, not less. With the flag off the whole
pipeline is byte-identical to e56's published hash, and e52's frozen-ranker row
reproduces digit for digit through the canonicalised table. 61% of the residual
retrieval miss is one index-side decision — the `alt_name_length` sort — and the
next-largest class is adjectival country forms inside organisation names.**

---

## 1. What landed

| artifact | where |
|---|---|
| the alias tables and both guards | `mordecai3/place_aliases.py` — `STATES` (52) + `CA_PROV` (13) + `AP` (64 dotted forms), `alias_targets`, `alias_query` |
| the rule | 5 lines in `GeonamesService.build_name_search`, before `_clean_search_name` |
| the flag | `GeonamesService(normalize_place_abbrevs=True)`; `Geoparser(normalize_place_abbrevs=None\|True\|False)` |
| tests | `tests/test_place_aliases.py`, 52 cases |
| harness | `tools/end_to_end_eval.py --no-normalize-place-abbrevs`; per-gold `gold_outcomes` and `retrieval_examples` added to the JSON |

**The flag defaults to ON, and it is documented as fixing a bug rather than
expressing a preference.** Under the index's `sort: {alt_name_length: desc}` —
a pure fame prior with no relevance term — a mention that is an abbreviation
retrieves countries. Live, on the shipped index:

```
"Ky."  -> United Kingdom, United States, US Virgin Islands, Turkey ...   (Kentucky at rank 6)
"Ind." -> Indus River, Indianapolis, Indore, Indianapolis Airport ...    (no Indiana in 40 hits)
"N.M." -> Estación Aragón N.M, three Czech hills and a TV tower ...      (6 hits, no New Mexico)
"WA"   -> DR Congo, UAE, Central African Republic, UK, New Zealand ...   (Washington at rank 39)
```

e52's census found that **61% of every unretrievable gold in held-out
TR/LGL/GWN is this one defect**. With R1 on, all four put the state at rank 0
or 1.

`Geoparser(normalize_place_abbrevs=None)` — the default — means "use the
`GeonamesService`'s own setting", so a caller who passes `geonames=` keeps their
own configuration; True or False overrides it, including on a supplied service.
Setting it False reproduces the pre-e57 query byte for byte, which is what
anything reproducing a frozen candidate list (the `raw_data/` pickles included)
wants.

Three properties of the placement are load-bearing, all inherited from e52 and
all re-verified here:

* **Replace, do not prepend.** e52's first implementation prepended the expanded
  query's hits to the abbreviation's own list; at a serving window of 100 that
  evicts the tail and `Ky.` — which already had Kentucky at rank 6 — lost its
  own gold, for +27/−9. Replacing the query is both correct serving semantics
  and the version that scores +28/−0. It is also why the rule is free: one
  round trip in, one round trip out, no latency and no window spent twice.
* **Before `_clean_search_name`.** That function deletes the token "District"
  from any query, so expanding `D.C.` to "District of Columbia" would send
  "of Columbia". `QUERY_OVERRIDE` sends `"Washington, D.C."` instead, which is
  rank 0 for `4140963`. A test loops over every reachable expansion and asserts
  the cleaner leaves it intact.
* **Both guards.** A bare two-letter code expands only when the whole mention is
  that code, in capitals, with no dots (e51's rule); an AP form expands only
  when the mention actually ends in a period (e52's addition — without it
  `AP["la"]` fires on the word `La`, and `AP["miss"]`, `AP["man"]`, `AP["del"]`,
  `AP["ore"]`, `AP["ind"]` on ordinary English).

Nothing else in the pipeline moves. `res_formatter` measures every string
feature against the ORIGINAL mention (`ex["search_name"]`, not the query), so
each candidate feature keeps its training-time meaning and no retraining is
implied.

---

## 2. Gates

### 2a. Flag OFF is byte-identical

Not asserted — hashed. `experiments/e57_r1_retrieval/identity_check.py` is
e56's script with e56's key list: geoparse all 260 held-out documents through
`Geoparser.geoparse_batch`, dump every emitted field of every mention
canonically, md5 it.

| configuration | mentions | md5 |
|---|---|---|
| e56's published default-path hash | 2,159 | `f2656f881c0e85632ad9ec8235a0755d` |
| **`normalize_place_abbrevs=False`, here** | **2,159** | **`f2656f881c0e85632ad9ec8235a0755d`** |
| `normalize_place_abbrevs=True`, here | 2,159 | `658c603922190f5a46b8fb911d29f523` |

The third row is the other half of the gate: a matching hash in both columns
would mean the rule is inert.

### 2b. Every published end-to-end row reproduces with the flag off

`tools/end_to_end_eval.py evaluate --no-normalize-place-abbrevs`, D2
denominator, `max_choices=100`, 260 documents / 2,084 golds:

| row | published | here (OFF) |
|---|---|---|
| e29 seed42, spaCy spans (`phase0_report.md` §1) | 66.99 | **66.99** |
| e29 seed101, spaCy spans | 67.66 | **67.66** |
| e29 seed101 + gold head | 77.69 | **77.69** |
| e54 seed42 + gold head, no outlet | 78.55 | **78.55** |
| e54 seed42 + LGL/TR outlets, spaCy spans | 70.39 | **70.39** |
| **e54 seed42 + gold head + LGL/TR outlets** | **80.85** | **80.85** |

with the same `retrieval_miss` (88 spaCy / 112 head), the same detection
(76.83 / 87.62 F1), the same `n_pred` (2,159 / 2,173) and the same oracle rows
(83.45 / 84.37 / 87.41). The grid was run twice end to end and the pooled table
is identical, so these are identities rather than a nondeterministic pipeline
landing twice on the same place.

### 2c. e52's R1 result reproduces through the mainline table

`frozen_gate.py` imports e52's own harness read-only from its worktree, asserts
table parity against `mordecai3.place_aliases` over 1,159 probe strings
(every table key in every casing / punctuation form the guards care about, plus
the false-positive words), monkeypatches the mainline `alias_query` in, and runs
e52's `base` and `abbrev` arms — frozen `e29_swa_ep15/seed101.pt`, window 100,
all six held-out sources.

| | e52's report | e57, mainline table |
|---|---|---|
| TLG-hard `em_all` | 0.8244 → 0.8444 (+0.0200) | **0.8244 → 0.8444 (+0.0200)** |
| TLG-hard `em_cond` | 0.8551 → 0.8588 | **0.8551 → 0.8588** |
| macro-of-6 `em_cond` | 0.9214 → 0.9224 | **0.9214 → 0.9224** |
| macro-of-6 `em_all` | 0.9083 → 0.9151 | **0.9083 → 0.9151** |
| entity flips (n=8,977) | +28 / **−0** | **+28 / −0** (27 direct, 1 knock-on) |
| firings | 65 (0.72%), 0 false | **65 (0.72%)** |

Per source, `em_hard_cond`/`em_hard_all`: TR .8632/.8592, LGL .8686/.8591,
GWN .8447/.8150 — and **WikiDocs, Prodigy and Synth are digit-for-digit
identical to baseline**, which is e52's collateral-damage answer reproduced.
The gained strings are e52's list exactly, including the one knock-on gain
(`BELGRADE` in LGL, where an unretrievable `Mont.` had been injecting a garbage
anchor into its document's sibling geometry).

The only deviation from e52's prototype is one that cannot change a number: e52
kept the original query's non-duplicate hits behind the expanded query's, which
at window 100 are truncated away anyway; the serving patch sends one query and
keeps its hits. `QUERY_OVERRIDE` also drops e52's identity entry for
"Northwest Territories" — proven inert by the probe set rather than assumed.

### 2d. Suite

`uv run pytest tests/` → **146 passed, 2 failed, 1 skipped, 5 xfailed**. The two
failures are the documented allowances (`test_miss_oxford`, `test_prague`) and
the five xfails are Phase 0's. The 52 new cases cover the table, the period
guard, the case guard, the replace-not-prepend query shape, the
`D.C.`-before-`_clean_search_name` ordering, the flag's inertness on ordinary
mentions, the passthrough semantics of `Geoparser(normalize_place_abbrevs=…)`,
five live-index retrieval cases, and one full `geoparse_doc` case.

> A note on choosing the integration case: it must be `Ind.`, not `Ky.`. The
> baseline query for `Ky.` *does* return Kentucky, at rank 6, so the ranker
> finds it anyway and a `Ky.` test passes for the wrong reason. `Ind.` returns
> 40 hits with no Indiana in them, so it is unreachable until the query is
> rewritten. The first draft of this test used `Ky.` and was inert.

---

## 3. The composition measurement

Same code, same session, same documents; only the flag differs. D2 denominator,
2,084 gold toponyms over 260 held-out TR-News / LGL / GeoWebNews documents,
`max_choices=100`. Raw JSON in `experiments/e57_r1_retrieval/e2e/`.

### 3a. End-to-end exact match

| ranker / outlets | span det | EM off | **EM on** | Δ | acc@161 off → on |
|---|---|---|---|---|---|
| e29 seed42 — Phase-0 reference | spaCy | 66.99 | **68.19** | **+1.20** | 69.19 → 70.35 |
| e29 seed101 — packaged default | spaCy | 67.66 | **68.76** | **+1.10** | 69.91 → 70.87 |
| e29 seed101 | gold head | 77.69 | **79.08** | **+1.39** | 80.37 → 81.62 |
| e54 seed42, no outlet | gold head | 78.55 | **79.80** | **+1.25** | 80.71 → 81.86 |
| e54 seed42, LGL+TR outlets | spaCy | 70.39 | **71.64** | **+1.25** | 72.36 → 73.61 |
| **e54 seed42, LGL+TR outlets** | **gold head** | **80.85** | **82.29** | **+1.44** | 82.87 → **84.31** |

**Does the 80.85 best cell improve? Yes — to 82.29.**

Detection is bit-identical in every row (`n_pred` 2,159 / 2,173, det F1 76.83 /
87.62 unchanged), which it must be: the rule changes a query string, not a span.
So all of this is resolution. Emitted-location precision moves the same way,
83.71 → 83.86 in the best cell.

Per corpus, best cell: TR 72.81 → 73.41, **LGL 83.66 → 85.45**, GWN 79.35 →
80.50. LGL is where R1 pays most, which is the expected shape — it is US local
news, the corpus that writes `Ind.` and `Neb.`

### 3b. R1 is worth more under the head, not less

| span source | Δ EM from R1 |
|---|---|
| spaCy label filter | +1.10 / +1.20 / +1.25 |
| **gold span head** | **+1.25 / +1.39 / +1.44** |

The head does not absorb the retrieval fix; it *enlarges* it, because it finds
more of the abbreviation golds in the first place. Concretely, the head repairs
28 retrieval misses where the spaCy path repairs 26: it gains `S.C.` ×3 against
the spaCy path's ×1, and it gains `Vt.`, which the spaCy path cannot — there
`trim_span_tokens` strips the trailing period off the `Vt.` span and the AP
guard then (correctly) refuses to expand a bare `Vt`.

Against the head's +10.5 and the outlet's +2.4, R1's +1.4 composes with a hair
to spare: the full stack against the Phase-0 66.99 is now **+15.30**, against a
strictly additive prediction of +14.87.

### 3c. The oracle-span ceiling moves too

| ranker / outlets | oracle-span EM off → on | oracle retrieval recall |
|---|---|---|
| e29 seed42 | 83.45 → **84.80** | 92.91 → **94.36** |
| e29 seed101 | 84.37 → **85.82** | 92.91 → **94.36** |
| e54 seed42 + LGL/TR outlets | 87.41 → **89.05** | 92.91 → **94.36** |

R1 is not only closing the pipeline's gap to a fixed ceiling — it raises the
ceiling. **Anything quoting 87.41 as "the resolution ceiling on these documents"
needs updating to 89.05.**

### 3d. Gold by gold

The harness now records the decomposition bucket of every gold, so the two runs
diff gold by gold rather than in aggregate.

| cell | gained | lost | net | transitions |
|---|---|---|---|---|
| e29 s42, spaCy | 27 | 2 | +25 | 26 `retrieval_miss→correct`, 1 `null_answer→correct`, 2 `correct→ranker_error` |
| e29 s101, spaCy | 26 | 3 | +23 | 26 retrieval, 3 `correct→ranker_error` |
| e29 s101, head | 32 | 3 | +29 | 28 retrieval, 2 null, 2 ranker, 3 lost |
| e54 no outlet, head | 29 | 3 | +26 | 28 retrieval, 1 null, 3 lost |
| e54 + outlets, spaCy | 27 | 1 | +26 | 26 retrieval, 1 null, 1 lost |
| **e54 + outlets, head** | **30** | **0** | **+30** | 28 retrieval, 2 `null_answer→correct` |

**The +28/−0 no-regression property holds exactly in the best cell (+30/−0) and
holds approximately elsewhere: 1–3 losses against 26–32 gains.** The end-to-end
frame is not the frozen frame, and the difference is honest — the serving path
re-queries and re-anchors the whole document, so a repaired abbreviation becomes
a *stronger* anchor than the garbage one it replaces. Every single loss in the
grid is the same mechanism, and all three are identifiable:

| lost gold | mechanism |
|---|---|
| TR doc 96, `Paris` → 2988507 (France) | the document also contains `Ky.`; repaired, Georgetown/Kentucky anchors the geometry and Paris goes to a US Paris |
| TR doc 97, `Paris` → 2988507 | same, via `N.J.` and Newark |
| LGL doc 458, `Richmond` → 4305974 (Kentucky) | the document also contains `La.`; repaired, a Louisiana anchor appears |

This is e52's `BELGRADE` knock-on gain running the other way. Three golds in the
whole grid; the mechanism is symmetric and the sign is strongly positive.

---

## 4. The retrieval-miss residual, by cause

Best cell, R1 on: **112 → 84**. R1 addresses the abbreviation subset only, so
the question is what the other 84 are. `flips.py` classifies each one live
against the index.

| cause | head, on (84) | spaCy, on (62) | what it is |
|---|---|---|---|
| `past_window` | **51 (61%)** | 47 (76%) | the query *does* return the gold — past rank 100 |
| `nested_span` | 15 (18%) | 1 | a toponym inside an ORG/FAC name |
| `name_mismatch` | 14 (17%) | 10 | the gold row's names do not phrase-match the mention |
| `gold_row_gone` | 4 | 4 | stale label; the geonameid is not in GeoNames |
| `abbrev_other` | **0** | **0** | an abbreviation R1's table does not cover |

For reference, of the 112 before R1, **28 were the abbreviation class** — R1
takes all of it, and leaves nothing behind in that bucket.

**`past_window` — 51 of 84, the biggest single lever left in the system.**
Measured ranks: `Richmond` ×8 (rank 100–228), `Hanover` ×7, `Logan` ×5,
`Charles City` ×4, `McKee` at 182, `Paris`/`PARIS` at 228, `London` at 198,
`Detroit` at 160, `Hinton` at 135. These are small same-name `PPL` rows losing
to the `alt_name_length` fame prior — exactly e52 §5(ii) item 3, restated end to
end. It is an index-rebuild-and-retrain item (sort by `_score` with
`alt_name_length` as tie-break, or a `function_score`), not a serving flag,
because it changes every candidate list.

**`nested_span` — 15, and 14 of them are adjectival.** `British` ×6 (inside
"British Council", "British Waterways"), `European` ×2, `Canadian` ×2,
`European Union` ×2, `Turkish`, `Nigerian`, plus one real nested toponym
(`Branchville`, inside "the Branchville Correctional Facility"). These are the
head's *new* misses — the class that took 88 to 112 — and they survive D2
because spaCy labels the containing string ORG, so the adjective is not a NORP
span. **The fix is structurally identical to R1 one class over: a
demonym/adjectival→country alias table on the same query hook.** Sizing it is
the obvious e58.

`name_mismatch` (14) is mostly the data debt e52 already handed back:
`North Africa` → `Northern Africa RGN` ×4, `Mount Chaambi` → an Algerian AREA
row, `South Africa` → Durban, plus two `… Co.` county abbreviations
(`PLYMOUTH CO.`, `Platte Co.`) that a "Co." → "County" rule would reach.
`gold_row_gone` (4) is unrecoverable by any model.

---

## 5. Pricing the "LA" risk

e52 flagged it and could not price it: `LA` in capitals still expands to
Louisiana, and in US newswire `LA` usually means Los Angeles. No held-out
mention was `LA`, so that arm had no evidence either way.
`experiments/e57_r1_retrieval/la_risk.py` prices it three ways, strongest
evidence first.

### 5a. What the serving path actually queries

`build_name_search` was wrapped and every query string it received recorded
through a real end-to-end run, for both span sources, then classified.

| span source | distinct queries | AP firings | bare-code firings | blocked |
|---|---|---|---|---|
| spaCy label filter | 970 | 28 | **3** — `SC`, `WA`, `FL` | 1 (`Vt`, no period) |
| gold span head | 782 | 29 | **4** — `SC`, `WA`, `NC`, `FL` | 0 |

**No `LA` is queried at all, in either frame.** Nor `IN`, `OR`, `OK`, `ME`,
`DE`, `MD`, `ID` or `PA`.

### 5b. What the corpora annotate

Three of 3,992 held-out gold toponym rows have a bare code as their phrase:

| gold phrase | gold id | expansion | right? |
|---|---|---|---|
| `KY` (TR, "LOUISVILLE, KY--") | 6254925 Kentucky | Kentucky | ✓ |
| `SC` (GWN, "HORRY COUNTY, SC") | 4597040 South Carolina | South Carolina | ✓ |
| `WA` (GWN, "Wastewater in WA is diluted…") | **2058645 State of Western Australia** | Washington (USA) | **✗** |

That third row is the real thing e52 could not find: a capitalised bare code
whose gold is not the US state. **And it costs exactly zero.** With R1 off the
pipeline answers `Washington ADM1 USA`; with R1 on it answers `Washington ADM1
USA`. The baseline query already had Washington inside the 100-window (rank 39)
and the ranker already preferred it, so the mention is a `retrieval_miss` in
both arms, with the same wrong answer. R1 makes the failure no worse — and if
anything makes the case for adding `WA`→Western Australia to the table rather
than removing `WA` from it.

The two bare-code firings that land on no gold at all resolve identically in
both arms too: `NC` in "the NC Dinos" (a Korean baseball team) → North Carolina
**with and without R1**, and `FL` in "(R-FL)" → Florida likewise. Neither is
caused by R1; both are what the un-normalised query already did.

### 5c. What the text contains

The upper bound on exposure — standalone capitalised codes in the held-out
document text that are *not* in a `…, IL` dateline shape:

```
PA x4 (the Palestinian Authority)   OK x3 ("that's OK", "get an OK")
ID x3 (an ID badge, voter ID laws)  NC x2   KY x2 ("KY 52", a highway)
MD x1 (a person's initials)         WA x1
```

**None of these reaches the gazetteer.** The tagger does not label them as
places, so the table never sees them. That is the point of e52's first rider —
the alias table is for spans the tagger already called a place, and the tagger
is the outer guard. `KY 52` is also caught by the whole-mention guard: the span
is "KY 52", not "KY".

### 5d. Verdict

**Keep `LA`, and keep the rest of the bare-code table.** There is no held-out
evidence of harm; the single ambiguous gold is already answered wrong without
the rule; and the ambiguity that does exist in the raw text is filtered out one
layer earlier by the tagger. The residual risk is documented at the top of
`mordecai3/place_aliases.py` and the mitigation is a one-line deletion if a
production feed disagrees.

One interaction found while doing this, worth recording because it is not
obvious: **`trim_span_tokens` can eat the period the AP guard depends on.** On
the spaCy path, LGL 581's gold `Vt.` is emitted as the span `Vt`, the guard
refuses it, and the state is never retrieved. It costs nothing there (that gold
was already lost to a boundary error), but under the head — which emits `Vt.` —
the same gold becomes correct. Anyone tightening the guard should know the two
interact.

---

## 6. Limits

1. **One ranker seed per ranker, one head seed.** Inherited from e56: 80.85 is a
   point on a ±1.4 head-seed spread. R1's effect is measured *paired* on the
   same seeds, so the Δ is much more robust than the level — it is +1.10 to
   +1.44 across six independent (ranker, outlet, span-source) conditions with no
   sign changes.
2. **The e2e frame is the same 260 held-out documents of the same three
   corpora** as every other campaign-2 serving number. R1's mechanism is
   corpus-independent in a way the span head's is not — it is a gazetteer query
   fix, not a learned convention — but "62 of 8,977 mentions are AP datelines"
   is a property of US-heavy news, and a non-US feed will see less of it.
3. **The pickles under `raw_data/` were built with the flag off.** The frozen
   ranker is therefore scoring candidate lists it was not trained on for the 65
   mentions the rule touches; that train/serve skew is inside the measured
   number, not excluded from it. e52's rider stands: if the pickles are ever
   rebuilt, rebuild them **with** the flag on, and the win should grow, because
   27 of the entities the model can now see were previously unlearnable.
   `tools/train.py` constructs its own `GeonamesService` and will now default to
   the flag being on — that is the desired behaviour, and it means the next
   pickle rebuild changes those 65 candidate lists.
4. **`abbrev_other` is 0 on this held-out set, not in general.** Australian
   states, UK counties and non-US postal codes are not in the table.

---

## 7. Recommended default

```python
Geoparser(
    normalize_place_abbrevs=None,   # -> True; nothing to flip, it is already on
)
```

**Adopted and on.** Unlike e56's two staged flips, this one is applied: it has no
open gate. The reasons it is a different class of change from `span_detector`:

* it is **detection-invariant** — every detection number in the grid is
  bit-identical with the flag on and off, so there is no span-convention
  generalisation question to answer;
* it is **not learned** — there is no corpus-specific behaviour to transfer, and
  no equivalent of e56's leave-one-corpus-out gate to open;
* it is **strictly positive in every measured condition**: 6 of 6 e2e cells,
  2 checkpoints × 2 windows on the frozen-ranker frame, 0 collateral on the
  three sources it cannot reach, and 0 latency;
* the escape hatch exists and is tested, for anything that must reproduce a
  pre-e57 candidate list.

What this changes for other documents:

* `experiments/campaign2/span_head_serving_report.md` §7's best cell reads
  **82.29** with R1 on, its `retrieval_miss` row **84** rather than 112, and its
  oracle-span ceiling **89.05** rather than 87.41.
* e56's §10.6 surprise ("`retrieval_miss` went up 88 → 112; the next lever is
  retrieval") is now discharged by 25%, and the remaining 84 are named in §4.

### What to do next, in order of measured size

1. **The `alt_name_length` sort.** 51 of the 84 residual retrieval misses are
   golds the query already returns, past rank 100. Index rebuild + retrain;
   e52 §5(ii)3 has the spec. This is the largest retrieval lever left.
2. **A demonym / adjectival alias table.** 14 of 84, all created by the span
   head, all on the same query hook R1 uses (`British` → United Kingdom,
   `Canadian` → Canada). Structurally identical to R1 and probably cheaper,
   since e56 §4 already has the demonym inventory. Note the D2 subtlety: these
   are *in* the denominator precisely because spaCy calls the container an ORG,
   so they are not the demonyms D2 excludes.
3. **`… Co.` → `… County`**, 2 of 84, a one-line addition to the same table.
4. **Fold the aliases into the index as real alternate names** (e52 §5(ii)1:
   66 documents updated, ~4 KB) — which makes R1 unnecessary at query time and
   helps fuzzy matching too. Until then the client-side rule is the whole fix.
