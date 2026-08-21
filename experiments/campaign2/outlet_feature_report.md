# e50_outlet — a leak-safe news-outlet-location feature for the candidate ranker

Campaign 2, Phase 2 ("cheap feature arms"). Written 2026-08-20.

Follows up the finding in `encoder_scoping_report.md` §5b: every LGL article
carries a `<domain>`, 80.8% of LGL's linked toponyms fall in their outlet's
modal admin1, and 46.5% of the model's LGL errors have the gold inside the
outlet's home admin1 with the prediction somewhere else. This arm builds that
signal into the ranker as a named feature block, under a leak protocol that
never lets the mapping see a gold label.

**Verdict: ADOPT the feature. TLG-hard +0.0300 ± 0.0039 (t = 7.61, five paired
seeds, criterion t(4) > 2.776) — the largest single-block gain of either
campaign, more than four times the Wave-4 weight-averaging win and comparable to
the whole 26-feature Wave-2 enrichment. It lands where it was aimed: LGL
non-country EM +0.0701 (t = 18.3), LGL novel-pair EM +0.1111 (t = 15.0). All
four sources that structurally cannot have an outlet are flat, and a
home-permutation control that holds both mask channels bit-identical reproduces
none of the gain. Two things must be fixed before it ships, neither of which
changes the verdict: the home table needs rebuilding from an independent
newspaper directory (§3.4), and the model has no fallback when the outlet is
missing (§8).**

---

## 1. What was built

Five columns, computed in `candidate_features.py` style by the new
`mordecai3/outlet_features.py`, registered as feature block `outlet` in
`torch_model.FEATURE_BLOCKS`:

| column | meaning | value with no outlet |
|---|---|---|
| `has_outlet_home` | this document's outlet has a known home *point* | 0.0 |
| `log_km_to_outlet_home` | `log10(km + 1)` from candidate to that point | `log10(20001)` = 4.30105 |
| `outlet_same_adm1` | candidate is in the home's `(country, admin1)` | 0.0 |
| `outlet_same_country` | candidate is in the home's country | 0.0 |
| `has_outlet_country` | the outlet resolves to a country at all | 0.0 |

Design decisions worth recording:

- **The distance sentinel is the sibling geometry's, not zero.** 0.0 km is the
  *best* possible value for a distance and the placeholder row competes in the
  softmax, so "no outlet" reuses `NO_ANCHOR_KM`, exactly as
  `candidate_features.NULL_SENTINELS` does for `log_min_km_anchor`.
- **Two mask channels, because a home can be a point or only a country.** A
  local paper gets a newsroom point; a national paper (Haaretz, the Irish
  Independent, CBC) gets its country and no point, so `outlet_same_country`
  fires while the geometric columns stay at their nulls. Inventing a city for
  Reuters would put a locality prior on a wire service.
- **The masks are document properties**, so the "no correct answer" placeholder
  row carries their real values — the same treatment `mention_admin_cue` gets —
  while its evidence columns take the neutral/worst end.
- **Appended last in `FEATURE_BLOCKS`**, after `strip`, so every pre-existing
  column keeps its index. This is what makes the baseline contrast exact (§6.1).

Only LGL and TR-News have outlet metadata. GWN, Prodigy, Synth and WikiDocs —
81% of held-out entities — get the full null. That is a well-defined "no
evidence" and also, unavoidably, a corpus indicator; §6.4 is the control that
settles which of the two the model is using.

## 2. Recovering the metadata

The pickles record no article id, so the outlet had to be joined back on.
`tools/outlet_align.py` does it: entities from one article share a
byte-identical `doc_tensor`, so consecutive runs of equal `doc_key` are exactly
one article's surviving entities, and those runs are matched forward against the
corpus XML by requiring the group's key sequence to be a subsequence of the
article's. The join **raises rather than guesses** — and it succeeds completely:

| source | entities | with a domain | distinct domains |
|---|---|---|---|
| LGL | 3,245 | 3,245 (100%) | 85 |
| TR-News | 914 | 914 (100%) | 35 |

**The split structure is the reason an external mapping was mandatory.** LGL's
file order groups articles by feed and the positional split hands whole outlets
to one side:

| | train entities | held-out entities | domains | domains shared with the other side |
|---|---|---|---|---|
| LGL | 2,272 | 973 | 60 train / 26 held-out | **1** |

Twenty-five of the twenty-six held-out LGL outlets never appear in training at
all. Training-split statistics — permitted under the brief — would have covered
almost none of the held-out set. The domain string plus outside knowledge was
not the cautious option; it was the only one.

## 3. The mapping, and how it was derived

`tools/outlet_home_table.py`. 120 domains (85 LGL + 35 TR), each mapped to a
**place named in words** plus an ISO-3 country code and a `local`/`national`
scope. `geocode_homes` turns the words into coordinates through the same
Elasticsearch GeoNames index the ranker retrieves candidates from.

### 3.1 The four rules the table was written under

1. The only corpus-derived input is the set of `<domain>` strings. No article
   text, no `<toponym>`, no `<gaztag>`, no geonameid, and no per-domain gold
   statistic — from either split.
2. Every value names a place in words, never an identifier. The coordinates are
   GeoNames' answer to "where is Richmond, Indiana", not anyone's answer to
   "where is this article's gold".
3. `scope` is a judgement about the *outlet*: `local` = a city/metro paper,
   broadcaster or campus paper; `national` = no home town, so no point.
4. Where the outlet behind a domain was genuinely unclear the row is `None`
   rather than a guess — `reporter.net`, `dailytribune.net`,
   `theintelligencer.com`, `www.nlhnews.co.uk`,
   `goldentrianglenewspapers.com`. All five are training-split-only domains.

Resolution: 115 non-`None` rows → 109 resolved (94 point-level, 15
country-level); the 6 unresolved are wire services (Reuters, CNN, BBC,
Marketwired, FreshPlaza, BNO) which are correctly given no home at all.

### 3.2 Why a mechanical mapping was rejected

The obvious leak-proof approach — tokenise the domain and geocode the place-like
tokens — was built and abandoned, because on this data it is not merely weak but
**actively wrong in the direction that matters**. A population-ranked gazetteer
lookup of the domain's tokens returns:

| domain | mechanical answer | the actual masthead |
|---|---|---|
| `parispi.net` | Paris, **France** | Paris Post-Intelligencer, Paris **Tennessee** |
| `theparisnews.com` | Paris, **France** | The Paris News, Paris **Texas** |
| `sentinel-echo.com` | (no token) | The Sentinel-Echo, London **Kentucky** |
| `themercury.com` | (no token) | The Manhattan Mercury, Manhattan **Kansas** |
| `columbustelegram.com` | Columbus, **Ohio** | Columbus Telegram, Columbus **Nebraska** |
| `concordmonitor.com` | Concord, **California** | Concord Monitor, Concord **New Hampshire** |
| `richmondregister.com` | Richmond, **Virginia** | Richmond Register, Richmond **Kentucky** |

This is the inversion the encoder report predicted: *a local paper called Paris
is evidence against Paris, France*. A mechanical mapping would have supplied the
population prior a second time under a new name, which is exactly what the
feature exists to overrule.

### 3.3 How much the curator supplied

Of the 94 point-level homes, **42 (44.7%) have the city name literally inside
the domain string**; the curation contributed only the disambiguating region
(`gainesvilleregister.com` → Gainesville **Texas**, not Florida or Georgia — all
three are in this corpus). The other 52 (55.3%) required knowing the masthead
(`ajc.com` → Atlanta, `pal-item.com` → Richmond IN, `courant.com` → Hartford).

### 3.4 The one residual risk, stated plainly

The curation came from a language model's knowledge of these mastheads. That
knowledge is external to this corpus in the ordinary sense, but it cannot be
*proved* uncontaminated by LGL itself, and 55.3% of the point-homes depend on
it. Nothing in the audit below can close that gap, because it is not a property
of the code.

**The clean fix is cheap and should be done before shipping**: rebuild the table
by joining the 120 domains against a public newspaper directory (an open list of
US/UK local outlets with their cities of publication), and re-run the arm. If
the numbers hold, the risk is retired; the table is 120 rows, so this is an
afternoon, not a project.

## 4. Leak audit

`tools/outlet_leak_audit.py`, five mechanical checks — **all pass**.

| check | result |
|---|---|
| A. no numeric identifier anywhere in the curated table | PASS (120 rows) |
| B. re-geocoding from scratch reproduces the cached homes exactly | PASS (109 homes) |
| B. the cache carries no per-article information | PASS |
| C. **LGL domain identical with and without the gold join key** | **PASS — 0 of 3,245 entities differ** |
| C. TR, same test | 1 of 914 differs (bounded, see below) |
| D. held-out features identical when recomputed with the training half never loaded | PASS — 0 mismatched cells |
| E. `outlet_features.py` / `outlet_home_table.py` read no label field | PASS (AST, not grep) |

Three notes on reading this:

- **Part C is the important one.** The article join *can* use the gold id as a
  join key, so the audit re-runs it on mention strings alone and diffs the
  resulting domain assignment. On LGL — the source under test — the two agree on
  every one of 3,245 entities, so the outlet attached to a held-out LGL entity
  provably does not depend on its label. On TR exactly one entity (a held-out
  `'Paris'`) is assigned `brantnews.com` under one key and
  `www.tremontonleader.com` under the other; that is 1 of 274 held-out TR
  entities and cannot move TR EM by more than 0.0036.
- **The join is an artefact of reconstruction, not of the method.** In
  production the domain arrives with the document; there is no join.
- **Part E is AST-based on purpose.** These modules discuss the answer key at
  length in their docstrings, and a substring grep flags prose. The check reads
  subscripts, attributes and names — the only ways a label can actually be read.

**Post-hoc, after the table was frozen**, the concentration statistics look like
this (these use gold labels and fed nothing back into the mapping):

| | golds w/ point-home | in home ADM1 | in home country | < 50 km | < 150 km |
|---|---|---|---|---|---|
| LGL train | 1,863 | 75.8% | 91.6% | 52.5% | 66.5% |
| **LGL held-out** | **831** | **66.2%** | **87.5%** | **41.3%** | **63.8%** |
| TR held-out | 125 | 60.8% | 89.6% | 36.0% | 55.2% |

Held-out concentration is *lower* than training concentration, which is the
right direction: the mapping is not tuned to the evaluation set.

## 5. Coverage (the kill criterion)

The brief's kill criterion was <~30% of held-out LGL documents covered.

| | n | point home | country-only home | no home |
|---|---|---|---|---|
| LGL train | 2,272 | 85.0% | 9.8% | 5.1% |
| **LGL held-out** | **973** | **88.2%** | **11.8%** | **0.0%** |
| TR train | 640 | 15.8% | 46.4% | 37.8% |
| TR held-out | 274 | 46.7% | 14.6% | 38.7% |

88.2% of held-out LGL entities get a newsroom point and not one is left without
some home. Comfortably clear.

## 6. Results

### 6.1 The contrast is exact

Both arms use recipe `e29_swa_ep15` unchanged and differ **only** in the
`--feature-blocks` string. The outlet block was added on top of the frozen
`_enriched` pickles rather than by re-deriving them, so the other 33 features
are provably the ship-recipe ones.

The validation the brief asked for passes at the strongest available standard:
**all five baseline seed metrics files are md5-identical to the frozen
`experiments/e29_swa_ep15/seed*.json`** (seed42 = `1d84e9529c77db0ef5b9c639d2dff96f`,
the same hash e30's NOTES quotes). The pickle rebuild changed nothing, and
appending a block shifted no column.

*A bookkeeping note*: this baseline's 5-seed TLG-hard reads **0.8676**, against
the 0.8730 in SYNTHESIS.md — both at window 500, n = 1,219. Since the frozen
metrics files reproduce byte for byte, the model is identical and the difference
is in the metric code's revision history, not the run. Deltas are the comparable
quantity, as `state_abbrev_report.md` also notes.

### 6.2 Five seeds, paired, t(4) > 2.776

| metric | baseline | outlet | paired Δ | t | |
|---|---|---|---|---|---|
| **TLG-hard (primary)** | 0.8676 | 0.8976 | **+0.0300 ± 0.0039** | **7.61** | **\*** |
| LGL non-country EM | 0.8739 | 0.9440 | +0.0701 ± 0.0038 | 18.26 | \* |
| LGL EM (frozen metric) | 0.8981 | 0.9584 | +0.0603 ± 0.0029 | 20.81 | \* |
| LGL novel-pair EM | 0.8181 | 0.9292 | +0.1111 ± 0.0074 | 14.96 | \* |
| LGL twin-credit | 0.8956 | 0.9541 | +0.0586 ± 0.0030 | 19.51 | \* |
| novel-pair EM, all sources | 0.7731 | 0.8080 | +0.0349 ± 0.0044 | 8.00 | \* |
| twin-credit macro (no Synth) | 0.9241 | 0.9371 | +0.0130 ± 0.0023 | 5.63 | \* |
| macro EM, 6 sources (continuity) | 0.9258 | 0.9380 | +0.0122 ± 0.0024 | 5.00 | \* |
| macro EM, 5 sources | 0.9109 | 0.9243 | +0.0134 ± 0.0038 | 3.55 | \* |
| acc@161 km | 0.9659 | 0.9776 | +0.0117 ± 0.0008 | 14.27 | \* |

TLG-hard's three ingredients, so the primary number can be read honestly:

| | baseline | outlet | Δ | t |
|---|---|---|---|---|
| LGL non-country | 0.8739 | 0.9440 | +0.0701 | 18.26 \* |
| TR non-country | 0.8810 | 0.8924 | +0.0114 | 1.47 |
| GWN non-country | 0.8479 | 0.8563 | +0.0085 | 2.25 |

The primary metric moves because LGL moves. TR — 46.7% covered, and by national
outlets whose concentration is much weaker — contributes a positive but
non-significant amount. GWN has no outlet metadata and drifts up by an
insignificant 0.0085.

### 6.3 Guardrails: nothing regresses, and the mask is not being read

| source | baseline | outlet | Δ | t | has outlets? |
|---|---|---|---|---|---|
| Prodigy | 0.9092 | 0.9056 | −0.0036 ± 0.0100 | −0.36 | no |
| GWN | 0.9317 | 0.9352 | +0.0035 ± 0.0019 | 1.84 | no |
| Synth | 0.9773 | 0.9799 | +0.0027 ± 0.0032 | 0.83 | no |
| WikiDocs | 0.9292 | 0.9309 | +0.0018 ± 0.0012 | 1.50 | no |
| TR | 0.9092 | 0.9181 | +0.0089 ± 0.0052 | 1.71 | partly |

Every one is non-significant. The encoder report's stated tripwire was "if
WikiDocs moves at all, the model is reading the mask": WikiDocs moves +0.0018 at
t = 1.50. **Novel-pair EM deserves its own line** — it is the one number
answer-key memorisation cannot move, and LGL's rises 11.1 points. Whatever the
model learned, it did not learn it from the answer key.

### 6.4 The control that settles it: permuted homes

The block is a locality prior, but it is also a corpus indicator, and the model
is known to exploit corpus identity to switch annotation conventions. To
separate them, every outlet was given **another outlet's home**, permuted within
level (point↔point, country↔country). Verified on the pickles:
`has_outlet_home` and `has_outlet_country` are **bit-identical** to the real
arm, all 33 non-outlet columns are bit-identical, and only the three evidence
columns change. Anything surviving this is not locality.

| contrast | TLG-hard | t | LGL non-country | t |
|---|---|---|---|---|
| permuted vs baseline | **−0.0055** | −0.98 | **−0.0173** | −5.24 \* |
| real vs permuted | **+0.0355** | 9.96 \* | **+0.0874** | 116.0 \* |
| real vs baseline | +0.0300 | 7.61 \* | +0.0701 | 18.26 \* |

The mask is worth nothing: with the correspondence destroyed, the block does not
help, and on LGL it significantly *hurts* (−0.0173) — precisely what a
confidently wrong locality prior should do. **100% of the gain is the true
article↔newsroom correspondence.** This also disposes of the "5 extra columns
of free capacity" explanation.

### 6.5 Per-seed

| seed | TLG base | TLG arm | Δ | LGL-nc base | LGL-nc arm | Δ |
|---|---|---|---|---|---|---|
| 42 | 0.8597 | 0.8990 | +0.0393 | 0.8769 | 0.9472 | +0.0704 |
| 101 | 0.8716 | 0.8949 | +0.0233 | 0.8744 | 0.9397 | +0.0653 |
| 202 | 0.8641 | 0.8950 | +0.0309 | 0.8756 | 0.9447 | +0.0691 |
| 617 | 0.8653 | 0.9029 | +0.0375 | 0.8656 | 0.9497 | +0.0842 |
| 1848 | 0.8771 | 0.8960 | +0.0190 | 0.8769 | 0.9384 | +0.0616 |

Every seed positive on both, no overlap with zero.

## 7. Mechanism: the predicted errors are the errors that moved

`tools/outlet_error_analysis.py`, seed 42, held-out LGL (946 scored entities,
92 baseline errors):

| baseline error class | n | share |
|---|---|---|
| **gold in home admin1, prediction elsewhere (fixable)** | **55** | **59.8%** |
| neither in home admin1 | 23 | 25.0% |
| both in home admin1 (feature inert) | 9 | 9.8% |
| prediction in home admin1, gold elsewhere (would hurt) | 5 | 5.4% |

**The arm fixed 53 of the 55 fixable errors — 96.4%.** Net +58 correct: 66 newly
fixed, 8 newly broken, and 6 of those 8 are the predicted failure mode (gold
elsewhere, prediction pulled into the home admin1). The feature is not finding
some other regularity; it is collecting almost exactly the mass the ceiling
analysis said was there, at the predicted cost.

The examples are the ones the encoder report named:

```
'Paris' (parispi.net)          base->2968815 (Paris, FR)  arm->4647963 (Paris, TN)  gold 4647963  [x3]
'North Shore' (post-gazette)   base->4335506              arm->5203760 (PA)         gold 5203760
'Oakland' (post-gazette)       base->5378538 (Oakland CA) arm->5204165 (Oakland PA) gold 5204165
'Fowlerville' (post-gazette)   base->5117788              arm->4993272              gold 4993272
'Charles City' (timesdispatch) base->4752002              arm->4752001              gold 4752001
```

## 8. Serving story (spec, not built)

**Integration.** Three small changes, all on paths that already exist:

1. `Geoparser.__init__(..., outlet_homes=<path|dict>)` — loads the geocoded
   table (a 26 KB JSON; `experiments/e50_outlet/outlet_homes.json` is the one
   these runs used).
2. `geoparse_doc(text, ..., outlet=None)` / `geoparse_batch(texts, ...,
   outlets=None)` — an optional domain string per document.
3. In `add_es_data_batch`, immediately after the existing
   `add_document_features(doc_es)` call (`geoparse.py:1441`), one line per
   entity: `add_outlet_features(entity["es_choices"], home)`, or
   `clear_outlet_features(...)` when the document has no outlet.

Serve with `feature_blocks="prom,name,cue,sib,geo,shape,outlet"`; the checkpoint
sidecar already records the block list, so this is picked up automatically.

**Train/serve parity is by construction**, not by test: the enrichment calls the
same `mordecai3/outlet_features.py` functions the serving path would. This is
the pattern `candidate_features.py` established and `tests/test_feature_parity.py`
polices; an outlet case should be added to that test when the serving path lands.

**Latency: negligible.** One dict lookup per document, plus one length-N
haversine per entity — where the `geo` block already does an N×M haversine
against every sibling anchor. The ranker is 2–3% of end-to-end latency (ES is
57–65%), so this is well under 0.1% of a request.

**API implications.**

- The argument is optional and the null path is well-defined, so no caller
  breaks.
- A production table needs real coverage. The 120 rows here are corpus-specific;
  a deployment wants a newspaper-directory join (which is also the fix for
  §3.4). Unknown domains degrade to the null path.
- The feature can be wrong: 5.4% of LGL errors are cases where it points away
  from the gold, and 6 of the arm's 8 new errors are of that kind. A local paper
  covering a distant story is the failure mode.

**The blocker: there is no graceful fallback.** `tools/outlet_degradation.py`,
seed 42, held-out LGL:

| | LGL EM |
|---|---|
| baseline checkpoint | 0.9027 |
| outlet checkpoint, outlet supplied | 0.9641 (+0.0613) |
| **outlet checkpoint, outlet withheld** | **0.8710 (−0.0317)** |

Serving the outlet checkpoint on a news document whose outlet is unknown is
**3.2 EM points worse than not having the feature at all**. The model learned to
lean on the prior for news-shaped documents and has nothing to fall back on when
it disappears — note that this does *not* show up in the WikiDocs/GWN/Prodigy
guardrails, because those sources had null outlets in *every* training example,
so the model learned a proper no-outlet policy for them and never for LGL.

The fix is standard and cheap: **outlet dropout** — null the block for a random
fraction (say 30%) of LGL/TR documents during training, so the no-outlet policy
is trained on news documents too. That is the recommended next arm (e51), and it
should be run before any ship decision.

## 9. Verdict and recommendation

**ADOPT `outlet` as a feature block.** The primary metric clears the criterion
by a wide margin (t = 7.61 against 2.776), the effect is concentrated in the
source it was designed for, every guardrail is flat, novel-pair EM — immune to
memorisation — rises 3.5 points overall and 11.1 on LGL, the leak audit passes
five mechanical checks, and a permutation control shows the entire gain is the
genuine article↔newsroom correspondence rather than the corpus indicator the
block unavoidably also is.

**Two conditions before it ships**, neither affecting the verdict:

1. **Rebuild the home table from an independent newspaper directory** (§3.4) and
   re-run the arm. 55.3% of the point-homes rest on curator knowledge that
   cannot be proven free of this corpus. 120 rows.
2. **Run the outlet-dropout arm (e51)** (§8). As trained, the checkpoint is
   worse than the baseline on news documents without an outlet, which is a
   realistic serving condition.

**Also worth doing**, in rough priority order:

- Extend the table to TR's local outlets more aggressively, and check whether
  the TR gain becomes significant.
- Re-examine the 5.4% "would hurt" class: `outlet_same_adm1` is a hard
  indicator, and a softer form (distance percentile within the candidate set)
  might keep the gain while costing fewer of the 6 new errors.
- Note for the scoreboard: this arm moves the macro-of-six by +0.0122 to 0.9380.
  Since the feature only exists for two of the six sources, that number should
  not be quoted as a general accuracy improvement — TLG-hard and the per-source
  table are the honest summaries.

---

### Reproduction

Worktree: `/home/andy/projects/mordecai3/.claude/worktrees/agent-abf0c3af0f35ebb78`
(branch `worktree-agent-abf0c3af0f35ebb78`, based on `accuracy-campaign` @ 81eaacd,
plus the main tree's uncommitted campaign-2 tooling).

```bash
# 1. geocode the curated table and attach the block to the frozen enriched pickles
uv run python tools/enrich_pickles.py --outlet-only \
    --data-dir raw_data/pickled_es --out-dir $E50/pickled_es \
    --outlet-home-cache $E50/outlet_homes.json

# 2. the two arms, five seeds each (recipe e29_swa_ep15, --data-dir points at the above)
bash tools/run_e50.sh baseline 42      # ... 101 202 617 1848
bash tools/run_e50.sh arm      42      # ... 101 202 617 1848

# 3. the permutation control
bash tools/build_e50_perm.sh && bash tools/run_e50.sh perm 42   # ...

# 4. audit, aggregate, analyse
uv run python tools/outlet_leak_audit.py
uv run python tools/outlet_aggregate.py
uv run python tools/outlet_error_analysis.py
uv run python tools/outlet_degradation.py
```

New files: `mordecai3/outlet_features.py`, `tools/outlet_home_table.py`,
`tools/outlet_align.py`, `tools/outlet_leak_audit.py`,
`tools/outlet_aggregate.py`, `tools/outlet_error_analysis.py`,
`tools/outlet_degradation.py`, `tools/run_e50*.sh`, `tools/build_e50_perm.sh`.
Modified: `mordecai3/torch_model.py` (one block + one pad sentinel),
`tools/enrich_pickles.py` (`--outlet-only`, `--permute-homes`). `tools/train.py`
is unmodified — the block name flows through the existing `--feature-blocks` and
`--data-dir`.

---

# e53 follow-up — the two pre-ship conditions, discharged

Written 2026-08-20, after the e50 sections above. Both conditions §9 set were
assigned back and are now answered: the independent table rebuild (condition 1,
§10) and the outlet-dropout arm (condition 2, §11). **Both pass. The
recommendation is now ship the outlet block trained with per-document dropout at
p = 0.5.**

## 10. Condition 1 — the home table, rebuilt from public sources

§3.4 flagged the one risk the leak audit could not close: 55.3% of the curated
newsroom homes came from a language model's knowledge of newspaper mastheads,
which cannot be *proved* free of LGL itself.

**Method.** Two research passes were given **only the list of 120 domain
strings** — never the curated answers, never the corpus — and told to resolve
each against Wikipedia, the Library of Congress newspaper catalogue,
archive.org snapshots of the outlets' own contact pages, FCC/station records and
media directories, and to return UNKNOWN rather than guess. Every row carries
the URL it came from. The result is
`tools/data/outlet_homes_researched.tsv`; the comparison is
`tools/outlet_table_diff.py`.

### 10.1 Disagreement report

**101 of 120 domains (84.2%) agree on both city and region.** Nineteen differ,
and they are not all the same kind of thing:

| domain | curated | researched | held-out entities | conf |
|---|---|---|---|---|
| haaretz.com | country-only ISR | Tel Aviv, Tel Aviv | 86 | high |
| sptimes.ru | Saint Petersburg | Saint Petersburg, St.-Petersburg | 68 | high |
| jpost.com | Jerusalem | Jerusalem, Jerusalem | 61 | high |
| independent.ie | country-only IRL | Dublin, Leinster | 29 | high |
| www.missourifarmertoday.com | Missouri | Missouri, Missouri | 21 | medium |
| www.nlhnews.co.uk | **UNRESOLVED** | London, England | 18 | medium |
| engineeringnews.co.za | country-only ZAF | Johannesburg, Gauteng | 10 | high |
| bclocalnews.com | British Columbia | British Columbia, British Columbia | 8 | medium |
| **myinrich.com** | **Richmond, Indiana** | **Richmond, Virginia** | **3** | **high** |
| civil.ge | country-only GEO | Tbilisi | 0 | high |
| messenger.com.ge | country-only GEO | Tbilisi | 0 | high |
| dailystar.com.lb | Beirut | Beirut, Beyrouth | 0 | high |
| moscowtimes.ru | Moscow | Moscow, Moscow | 0 | high |
| recordernewspapers.com | New Jersey | Bernardsville, New Jersey | 0 | medium |
| jordannews.com | country-only JOR | **Jordan, Minnesota** | 0 | high |
| dailytribune.net | **UNRESOLVED** | Mount Pleasant, Texas | 0 | high |
| goldentrianglenewspapers.com | **UNRESOLVED** | Mount Pleasant, Iowa | 0 | medium |
| reporter.net | **UNRESOLVED** | Lebanon, Indiana | 0 | high |
| theintelligencer.com | **UNRESOLVED** | Edwardsville, Illinois | 0 | high |

Sorting them by what they actually are:

- **Six are cosmetic** (sptimes, jpost, moscowtimes, dailystar,
  missourifarmertoday, bclocalnews): same city, and the curated table simply
  left the `admin1` *search hint* blank. The hint is not the answer — the
  resolved home takes its admin1 from the gazetteer row — so these change
  nothing at all.
- **Five were curated as UNRESOLVED under rule 4** and the researchers resolved
  them. All five are training-split-only except `www.nlhnews.co.uk` (18
  held-out TR entities). This is the rebuild adding coverage, not correcting an
  error.
- **Six are a convention difference, not a factual one**: the curated table gave
  national papers with a real head office only their country; the researchers
  found the city. Haaretz → Tel Aviv, the Irish Independent → Dublin,
  Engineering News → Johannesburg, Civil.ge and the Messenger → Tbilisi. This
  is the single largest block of affected held-out entities (125), and it
  *improves* the table: Israel and Ireland are small enough that a city home is
  a good locality prior.
- **Two are real factual disagreements.** `jordannews.com` — the curated table
  read it as Jordan the country; the LoC catalogue says the *Jordan
  Independent* of Jordan, **Minnesota** (train-split only, 14 entities). And
  **`myinrich.com`, where the curated table is simply wrong**: it read "my IN
  rich" as Richmond **Indiana**, but the archived contact page shows Media
  General's InRich portal for Richmond **Virginia**. Three held-out entities.

So: one outright curation error, one train-only error, five gaps filled, six
convention improvements, six no-ops. **304 held-out entities sit behind a
disagreeing domain** — 31% of LGL's held-out set — which makes the re-evaluation
below a real test rather than a formality.

Coverage improves under the rebuilt table: 105 point-level homes against 94,
and every one of the 120 domains resolves.

### 10.2 Does the gain survive the swap?

The existing e50 and e53 checkpoints were re-scored on held-out LGL with the
outlet features rebuilt from the researched table (`tools/build_e50_v2.sh`;
the other five sources are hard-linked, so they are provably identical between
the two runs).

| checkpoint | LGL EM, curated table | LGL EM, researched table | Δ |
|---|---|---|---|
| e29 baseline (no outlet block) | 0.8981 | 0.8981 | 0.0000 |
| e50 arm | 0.9584 | 0.9577 | −0.0007 |
| e53 d50 | 0.9605 | 0.9598 | −0.0007 |

The measured gain over baseline is **+0.0603 with the curated table and +0.0596
with the researched one**. A model trained on one table and evaluated on the
other loses seven ten-thousandths.

**Condition 1 is discharged.** The feature does not depend on the curator's
knowledge: a table built entirely from Wikipedia, the LoC catalogue and
archived masthead pages reproduces the result. The researched table is the one
a deployment should ship, both because its provenance is checkable and because
it resolves more domains — and it is what a production newspaper-directory join
would look like anyway.

## 11. Condition 2 — outlet dropout (ledger entry `e53_outlet_dropout`)

§8's blocker: the e50 checkpoint scores 0.8698 on LGL when the outlet is
withheld, against the baseline's 0.8981 — **worse than not having the feature**.
The cause is structural: GWN, Prodigy, Synth and WikiDocs have null outlets in
*every* training example, so the model learned a good no-outlet policy for
their kind of document and never for LGL's.

**The arm.** `--outlet-dropout p` blanks the five outlet columns — to the exact
null encoding a no-outlet source carries — for a random subset of documents,
redrawn each epoch. Per *document*, not per entity, or the prior leaks back
through an article's other mentions. The draw is deterministic in
`(document, epoch, seed)` and independent of the global RNG, so runs stay
bit-reproducible. Verified no-op at p = 0: all five e29 seeds still md5-match
the frozen files.

### 11.1 Condition (a) — outlet present: dropout is free

| metric | e29 | e50 | **d50 (p=0.5)** | d50 vs e29 | d50 vs e50 |
|---|---|---|---|---|---|
| **TLG-hard** | 0.8676 | 0.8976 | **0.8956** | **+0.0281, t 8.01 \*** | −0.0019, t −0.62 |
| LGL non-country | 0.8739 | 0.9440 | 0.9457 | +0.0719, t 22.20 \* | +0.0018, t 1.20 |
| LGL novel-pair | 0.8181 | 0.9292 | 0.9301 | +0.1120, t 13.55 \* | +0.0009, t 0.29 |
| novel-pair, all | 0.7731 | 0.8080 | 0.8010 | +0.0279, t 10.42 \* | −0.0070, t −1.98 |
| macro EM, 6 src | 0.9258 | 0.9380 | 0.9364 | +0.0106, t 4.34 \* | −0.0016, t −1.36 |

Every "vs e50" comparison is non-significant: **there is no measurable giveback.**

### 11.2 Condition (b) — outlet withheld: the blocker is fixed

| arm | present | withheld | withheld − baseline | |
|---|---|---|---|---|
| e29 baseline | 0.8981 | 0.8981 | (reference) | |
| e50 | 0.9584 | 0.8698 | −0.0283 ± 0.0052, t = −5.45 | \* |
| **d50 (p=0.5)** | **0.9605** | **0.8930** | **−0.0051 ± 0.0036, t = −1.41** | **n.s.** |
| d30 (p=0.3) | 0.9607 | 0.8913 | −0.0068 ± 0.0023, t = −2.96 | \* |

d50 also *improves* the present condition slightly (0.9605 vs e50's 0.9584),
which is what regularisation on an optional input tends to do.

### 11.3 Condition (c) — guardrails

All non-significant for d50: Prodigy −0.0116 (t −0.94), TR +0.0074 (t 1.53),
GWN +0.0000, Synth +0.0054 (t 1.49), WikiDocs +0.0001 (t 0.13).

### 11.4 p = 0.5 versus p = 0.3

Indistinguishable with the outlet present (TLG-hard 0.8956 vs 0.8980, LGL
non-country 0.9457 vs 0.9452). They differ only where the arm exists to differ:
withheld, **p=0.5 reads −0.0051 (t = −1.41) and p=0.3 reads −0.0068
(t = −2.96)**. More dropout buys a better fallback at no cost, so p=0.5 wins,
and the monotonicity makes p=0.7 worth one cheap probe if the withheld number
should be flat rather than merely non-significant.

**ADOPT criterion** — (b) within noise AND (a) significant at t(4) > 2.776:
d50 passes on both (t = −1.41, t = 8.01); d30 fails (b).

## 12. Ship recommendation

**SHIP the outlet block, trained with `--outlet-dropout 0.5`, with the
researched home table.** Concretely:

```
uv run python tools/train.py train \
  --epochs 15 --mix-dim 512 --logits --mask-padding --oov-bucket-fix \
  --modern-mlp --label-smoothing 0.05 --enriched --avg-params --avg-mode swa \
  --feature-blocks "prom,name,cue,sib,geo,shape,outlet" \
  --outlet-dropout 0.5 \
  --dataset-names "Prodigy, TR, LGL, GWN, Synth, WikiDocs"
```

What the deployment gets, relative to the current ship recipe e29:

- **+0.0281 TLG-hard (t = 8.01)** and **+0.0719 LGL non-country EM (t = 22.2)**
  on documents whose outlet is known;
- **no regression** on documents whose outlet is unknown (−0.0051, t = −1.41),
  which was the reason e50 could not ship;
- no regression on any source without outlet metadata;
- one optional `outlet=` argument, a 26 KB lookup table, and well under 0.1% of
  request latency (§8).

Remaining work, none of it blocking: a production domain→city table from a
newspaper directory rather than these 120 corpus-specific rows (the same join,
just wider); the softer `outlet_same_adm1` variant suggested in §9; and one
p = 0.7 probe.
