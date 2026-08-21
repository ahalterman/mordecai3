# e57_r1_retrieval — e52's R1 abbreviation fix in the mainline, and how it composes

Campaign 2, Phase 2 / the NER track's retrieval follow-up. e52 measured four
candidate-set hygiene rules against a frozen ranker and recommended exactly one
(R1: expand a US-state / Canadian-province abbreviation before the Elasticsearch
query). This entry **ports R1 into `mordecai3/`** behind a constructor flag and
answers the question e56 left open: e56's §10.6 surprise was that
`retrieval_miss` went **up** 88 → 112 under the span head, and named retrieval
as the next lever. Full write-up:
`experiments/campaign2/r1_retrieval_report.md`.

**Verdict: ADOPTED, default ON. It composes with everything. The best serving
cell moves 80.85 → 82.29 e2e EM (+1.44, +30 golds), the head's
`retrieval_miss` falls 112 → 84 (−28, −25%), and the oracle-span ceiling itself
rises 87.41 → 89.05. Every OFF row reproduces byte for byte — the whole
geoparse of the 260 held-out documents md5s to e56's
`f2656f881c0e85632ad9ec8235a0755d` with the flag off. The e52 frozen-ranker row
reproduces digit for digit through the mainline table: TLG-hard `em_all`
0.8244 → 0.8444, +28/−0.**

## What was integrated

| artifact | where it landed |
|---|---|
| e52's alias tables + guards (`hygiene.py`, from `e51_state_abbrev/sizing.py`) | `mordecai3/place_aliases.py` (`STATES`, `CA_PROV`, `AP`, `BARE_CODES`, `QUERY_OVERRIDE`, `alias_targets`, `alias_query`) |
| the rule | 5 lines in `GeonamesService.build_name_search`, **before** `_clean_search_name` |
| the flag | `GeonamesService(normalize_place_abbrevs=True)` and `Geoparser(normalize_place_abbrevs=None\|True\|False)` |
| tests | `tests/test_place_aliases.py`, 52 cases |
| harness | `tools/end_to_end_eval.py --no-normalize-place-abbrevs`, plus per-gold `gold_outcomes` and `retrieval_examples` in the JSON |

The flag defaults to **True** because it fixes a retrieval bug rather than
expressing a preference. `Geoparser(normalize_place_abbrevs=None)` (the default)
means "use the service's own setting", so a caller passing `geonames=` keeps
whatever they configured; passing True/False overrides even a supplied service.
It is a property whose setter clears `_es_cache`, because that cache is keyed on
the mention, not on the query the mention produces — without it, flipping the
flag on a live service serves candidate lists built under the other setting.

Three implementation facts, all inherited from e52 and all verified here:

* **It replaces the query, it does not add one.** Prepending the expanded
  query's hits to the abbreviation's own list costs 9 entities at window 100
  (`Ky.` had Kentucky at rank 6 and lost it to the eviction). Replacement is
  also why the rule is free — one round trip in, one out.
* **It runs before `_clean_search_name`,** which deletes the token "District".
  `D.C.` therefore expands through `QUERY_OVERRIDE` to `"Washington, D.C."`,
  not to `"District of Columbia"` → `"of Columbia"`. Pinned by
  `test_dc_expansion_survives_clean_search_name` and by a loop over every
  reachable expansion.
* **The two guards.** A bare code expands only as the whole mention, in
  capitals, with no dots; an AP form expands only when the mention ends in a
  period. Without the second, `AP["la"]` fires on the word `La` and
  `AP["miss"] / ["man"] / ["del"] / ["ore"] / ["ind"]` on ordinary English.

`res_formatter` is untouched: it measures string features against the ORIGINAL
mention (`ex["search_name"]`), so no candidate feature changes meaning and no
retraining is implied.

## Gates

1. **Byte identity with the flag OFF.** `identity_check.py`, e56's script and
   key list: the whole geoparse of the 260 held-out documents, 2,159 mentions,
   md5 `f2656f881c0e85632ad9ec8235a0755d` — **e56's published hash**. With the
   flag on it is `658c603922190f5a46b8fb911d29f523`, so the rule is firing.
2. **Every published e2e row reproduces with the flag OFF**, cell for cell:
   66.99 / 67.66 / 77.69 / 78.55 / 70.39 / **80.85**, with the same
   `retrieval_miss` (88 / 112), the same detection (76.83 / 87.62 F1) and the
   same oracle rows (83.45 / 84.37 / 87.41). The grid was also run twice and
   the aggregate is identical, so these are identities, not coincidences of a
   nondeterministic pipeline.
3. **e52's R1 row reproduces digit for digit through the MAINLINE table.**
   `frozen_gate.py` swaps `mordecai3.place_aliases` into e52's own harness
   (imported read-only from its worktree) after asserting table parity over
   1,159 probe strings:

   | | e52 | e57 |
   |---|---|---|
   | TLG-hard `em_all` | 0.8244 → 0.8444 | **0.8244 → 0.8444** |
   | TLG-hard `em_cond` | 0.8551 → 0.8588 | **0.8551 → 0.8588** |
   | macro-of-6 `em_cond` | 0.9214 → 0.9224 | **0.9214 → 0.9224** |
   | entity flips | +28 / **−0** | **+28 / −0** (27 direct, 1 knock-on) |
   | firings | 65 in 8,977 (0.72%) | **65 in 8,977** |

   WikiDocs / Prodigy / Synth are digit-identical to baseline, as in e52.
4. **Suite**: `uv run pytest tests/` → 146 passed, 2 failed, 1 skipped,
   5 xfailed. The two failures are the documented allowances
   (`test_miss_oxford`, `test_prague`).

## The composition grid (D2, 2,084 golds, 260 documents, `max_choices=100`)

Both columns are the same code in the same session; only the flag differs.

| ranker / outlets | span det | EM off | **EM on** | Δ | retr_miss | oracle EM |
|---|---|---|---|---|---|---|
| e29 seed42 (Phase-0 ref) | spaCy | 66.99 | **68.19** | +1.20 | 88 → 62 | 83.45 → 84.80 |
| e29 seed101 (default) | spaCy | 67.66 | **68.76** | +1.10 | 88 → 62 | 84.37 → 85.82 |
| e29 seed101 | gold head | 77.69 | **79.08** | +1.39 | 112 → 84 | 84.37 → 85.82 |
| e54 seed42, no outlet | gold head | 78.55 | **79.80** | +1.25 | 112 → 84 | 84.42 → 85.96 |
| e54 seed42, LGL+TR | spaCy | 70.39 | **71.64** | +1.25 | 88 → 62 | 87.41 → 89.05 |
| **e54 seed42, LGL+TR** | **gold head** | **80.85** | **82.29** | **+1.44** | **112 → 84** | 87.41 → **89.05** |

Detection is bit-identical in every row (the rule only changes a query string),
so this is pure resolution. Oracle-span retrieval recall 92.91 → **94.36**.

**R1 is worth MORE under the head, not less** (+1.44 / +1.39 / +1.25 vs +1.10 /
+1.20 / +1.25 on spaCy spans), and the head's extra retrieval misses do not
absorb it: the head finds *more* abbreviation golds, so the same rule repairs
28 of them instead of 26.

Gold by gold in the best cell: **+30, −0**, all 28 retrieval repairs plus 2
`null_answer → correct`. Other cells are +26 to +32 gained against **1–3 lost**,
and every loss is the same mechanism — a repaired abbreviation changes its
document's anchor geometry and a neighbour moves (`Ky.` repaired in TR doc 96
pulls `Paris`/France to a US Paris; `La.` repaired in LGL doc 458 pulls
`Richmond`/Kentucky). That is the knock-on e52 saw as a *gain* (`BELGRADE`),
running the other way.

## The residual: where the remaining retrieval misses go

Best cell, 84 left of 112, classified live against Elasticsearch
(`flips.py`, causes in `flips_e54_seed42_outlet_head_gold.json`):

| cause | n | what it is |
|---|---|---|
| `past_window` | **51** | the gold IS returned by the query, past rank 100 — `Richmond` ×8 at rank 100–228, `Hanover` ×7, `Logan` ×5 |
| `nested_span` | 15 | a toponym nested in an ORG/FAC name — 14 of them **adjectival** (`British` ×6, `European`, `Canadian`, `Turkish`, `Nigerian`) |
| `name_mismatch` | 14 | the gold row's names do not phrase-match (`North Africa` → `Northern Africa` RGN, `Platte Co.`, `Mount Chaambi`) |
| `gold_row_gone` | 4 | stale label, the geonameid is not in GeoNames (`Red Sea`, `Black Sea`, `Hillsboro`) |
| `abbrev_other` | **0** | — R1 covers the whole abbreviation class in this frame |

Under spaCy spans the same residual is 62: 47 / 1 / 10 / 4 / 0.

**61% of what is left is the `alt_name_length` sort**, exactly e52 §5(ii)3: a
fame prior with no relevance term, so small same-name `PPL` rows are
unreachable at any serving window. That is the next retrieval lever and it is
an index-rebuild-and-retrain item, not a serving flag. Second is the head's own
new class — **adjectival country forms inside organisation names**, which is a
demonym→country alias table, i.e. structurally the same fix as R1 one class
over.

## The "LA" risk, priced

e52 flagged it and could not price it: `LA` in capitals expands to Louisiana,
and in US newswire it usually means Los Angeles. `la_risk.py` prices it three
ways on held-out.

* **What the pipeline actually queries** (recorded by wrapping
  `build_name_search` through a real run): 31 firings on the spaCy path, 33
  under the head. Bare-code firings are **`SC`, `WA`, `NC`, `FL`** and nothing
  else. **No `LA` anywhere.**
* **What the corpora annotate**: three gold phrases in 3,992 held-out rows are
  bare codes — `KY` → Kentucky ✓, `SC` → South Carolina ✓, and `WA` →
  **2058645 State of Western Australia ✗**.
* **What the text contains**: standalone capitalised codes not in a dateline —
  `PA` ×4 (the Palestinian Authority), `OK` ×3, `ID` ×3, `NC` ×2, `KY` ×2,
  `MD`, `WA`. **None of them reaches the gazetteer**: the tagger does not label
  them as places, which is the outer guard the table relies on.

**Cost of the one genuine misfire: zero.** GWN's `WA` (Western Australia) is
answered `Washington ADM1 USA` **with and without R1** — the baseline query
already had Washington inside the window and the ranker already picked it. R1
makes that failure no worse. `NC` in "the NC Dinos" (a Korean baseball team)
and `FL` in "(R-FL)" resolve identically in both arms too.

**Verdict: no held-out evidence of harm, and the one ambiguous gold is already
lost without the rule. Keep `LA`.** The mitigation is documented in
`place_aliases.py` and is a one-line deletion if a production feed disagrees.

One thing the scan did turn up: on the spaCy path the emitted span for LGL 581's
gold `Vt.` is **`Vt`** — `trim_span_tokens` ate the period — so the AP guard
correctly refused it. It cost nothing (that gold was already a `boundary_ok`
loss), but **span trimming and the period guard interact**, and under the head,
which emits `Vt.`, the same gold becomes correct.

## Traps and surprises

* **The frozen-ranker frame says +28/−0; the end-to-end frame says +30/−0 in the
  best cell but +26/−3 in others.** Both are honest. The frozen harness rebuilds
  document features on an edited candidate list; the serving path re-queries and
  re-anchors, and a repaired abbreviation is a *stronger* anchor than a garbage
  one. Three golds in the whole grid are worse for it.
* **R1 raises the oracle-span ceiling too** (87.41 → 89.05), so it is not just
  closing the pipeline's gap to a fixed ceiling — it moves the ceiling.
  Anything quoting 87.41 as "the resolution ceiling" needs updating.
* **`retrieval_miss` was the right diagnosis and the wrong size.** e56 called
  retrieval "the next lever"; R1 takes 25% of it, and 61% of what remains is one
  index-side sort decision.
* **The alias table is the smallest measured win per line of code in the
  campaign**: 65 firings in 8,977 held-out mentions (0.72%), 5 lines of serving
  code, +1.44 EM in the best cell, 0 latency.

## Reproducing

```
uv run python experiments/e57_r1_retrieval/identity_check.py            # gate 1, ~3 min
bash          experiments/e57_r1_retrieval/run_grid.sh                  # the grid, ~12 min
uv run python experiments/e57_r1_retrieval/aggregate.py
uv run python experiments/e57_r1_retrieval/frozen_gate.py               # gate 3, ~2 min
uv run python experiments/e57_r1_retrieval/flips.py e54_seed42_outlet head_gold
uv run python experiments/e57_r1_retrieval/la_risk.py
uv run pytest tests/test_place_aliases.py
```

`frozen_gate.py` imports e52's harness read-only from
`.claude/worktrees/agent-a8c4f7c56da33eaf9/experiments/e52_gaz_hygiene/`; if
that worktree is ever removed, the gate needs the harness vendored. The
`geonames` index was read-only throughout.
