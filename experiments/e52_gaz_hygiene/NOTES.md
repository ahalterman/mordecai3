# e52_gaz_hygiene — candidate-set hygiene as a serving transform, no retraining

Second campaign, Phase 2 (see `experiments/campaign2/SYNTHESIS.md` §"the
campaign-2 ladder"). Full write-up:
`experiments/campaign2/gazetteer_hygiene_report.md`. No training was run: the
frozen `e29_swa_ep15` checkpoints score candidate lists that have been
transformed before they reach the ranker.

**Verdict: ONE RULE ADOPTED, three rejected.** Expanding a US-state /
Canadian-province abbreviation before the Elasticsearch query
(`Ind.` → `Indiana`) is worth **+0.020 TLG-hard on a fixed denominator, on both
checkpoints and both windows, for 28 entities gained and 0 lost, at zero
latency and with digit-for-digit no change on WikiDocs, Prodigy or Synth.**
Duplicate-row collapse is an annotation-convention swap and costs 112 entities
(keep-P) or ~1,500 (keep-A). Defunct-row demotion nets +2 entities in 8,977.
The rules that `ACCURACY_CAMPAIGN.md` item 3 predicted were mostly the wrong
diagnosis; the one it did not mention is the whole win.

## Why the arm existed

`ACCURACY_CAMPAIGN.md` §"What's left" item 3: *"a quarter of the residual error
mass is ~20 repeated strings (D.C., Mauna Kea's duplicate rows, Kathmandu's
historical ADM3H entry)."* This arm censused those strings against the live
gazetteer, built the fixes the census supported, and measured them.

## Commands

```
cd <worktree>/experiments/e52_gaz_hygiene
python parity_check.py --source TR --n 300         # the gate; must read 0.0
python census.py    --sources TR,LGL,GWN           # + --sources <all six>
python gaz_audit.py --sources TR,LGL,GWN
python run_grid.py --arms base,abbrev,demote_h,junk,dedupe_p,dedupe_a,ship \
                   --seeds 101,42 --windows 100
python diff_arms.py res_base_s101_w100.json res_abbrev_s101_w100.json
```

Everything imports `mordecai3` / `tools` from the main tree read-only through
`sys.path`; the worktree branch predates the campaign and `git reset`/`git
merge` into it were refused by the permission classifier. **The shared
`geonames` index was read-only throughout** — `es_util.py` issues `_mget`,
`_search`, `_count` and nothing else.

## Reading the numbers

`error_utils.evaluate_results` **conditions on the gold being retrievable**, so a
retrieval fix adds the hardest entities back into the denominator (reported EM
can fall while the system strictly improves) and a rule that deletes golds
removes them (reported EM rises while the system strictly worsens). Both are
reported:

* `em_cond` — the campaign convention, moving denominator, for ledger continuity.
* `em_all` — every held-out mention, an unretrievable gold scored wrong. **Read
  this one.**

Parity gate before any of it: rebuilding a candidate list from live ES rows with
**no rule enabled** reproduces the frozen pickles to `0.0` on all 23 per-candidate
feature keys, and `add_document_features` reproduces the sibling/geometry block
to `0.0`. The only residual is `adm1_count`/`country_count` on exactly one
document per source — the one the 70/30 split cuts in half — which is a
pre-existing split artifact, not a hygiene effect.

### The arms (frozen e29 seed101, serving window 100, all six sources)

| arm | TLG-hard `em_cond` | TLG-hard `em_all` | macro-6 `em_cond` | net entities |
|---|---|---|---|---|
| base | 0.8551 | 0.8244 | 0.9214 | — |
| **R1 `abbrev`** | 0.8588 | **0.8444** | 0.9224 | **+28 / −0** |
| R2 `demote_h` | 0.8551 | 0.8244 | 0.9221 | +10 / −8 |
| R4 `demote_junk` | 0.8593 | 0.8273 | 0.9223 | +23 / −14 |
| R3 `dedupe` keep-P | 0.8567 | 0.8239 | *0.9233* | +22 / −134 |
| R3 `dedupe` keep-A | *0.8649* | **0.7670** | *0.9263* | large negative |
| **R1+R2+R4** | **0.8625** | **0.8469** | **0.9239** | **+59 / −16** |

*Italic = the conditioned metric rewarding a rule for deleting golds.*

### R1 and the bundle across checkpoints and windows (TR/LGL/GWN, non-country)

| arm | seed | window | `em_all` | Δ | `em_cond` | Δ |
|---|---|---|---|---|---|---|
| R1 | 101 | 100 | 0.8244 → 0.8444 | +0.0200 | 0.8551 → 0.8588 | +0.0037 |
| R1 | 101 | 500 | 0.8458 → 0.8675 | +0.0217 | 0.8771 → 0.8822 | +0.0051 |
| R1 | 42 | 100 | 0.8166 → 0.8347 | +0.0181 | 0.8473 → 0.8491 | +0.0018 |
| R1 | 42 | 500 | 0.8343 → 0.8527 | +0.0184 | 0.8654 → 0.8673 | +0.0019 |
| R1+R2+R4 | 101 | 100 | 0.8244 → 0.8469 | +0.0225 | 0.8551 → 0.8625 | +0.0074 |
| R1+R2+R4 | 42 | 100 | 0.8166 → 0.8376 | +0.0210 | 0.8473 → 0.8532 | +0.0059 |

## What the census actually said

Over the 219 non-country TR/LGL/GWN errors: 64.4% are the ranker choosing a
genuinely different place, 11.4% are the A/P convention, and the
gazetteer-addressable classes total 22.8% — of which the single largest is
**`retr_abbrev`, 12.3%**. Duplicate rows are 2.7% and defunct rows 0.5% on the
primary metric; they are 10.8% and 2.1% over the six-source macro, which is
where the memo's examples come from. Three corrections to the record:

* **`D.C.` is not a missing alt-name.** `4140963 Washington PPLC` is the rank-0
  hit for that query. The 20 errors are the A/P convention and twin credit
  already pays for them; the alias rule measurably changes nothing there.
* **`Mauna Kea` is a spelling trap, not a duplicate row.** The gold `5850911` is
  spelled `Maunakea`; `6326699 Mauna Kea` is a different mountain on Maui,
  150 km away, 1 alternate name, and it wins only because `exact_name_match` is
  whitespace-sensitive.
* **`Kathmandu` is exactly what the memo said** — `1283241 Kathmandu District
  ADM3H`, population 1,264,684, beating the live PPLC — and it is the only one.

The finding the memo did not have: **27 of the 44 unretrievable golds in
held-out TR/LGL/GWN (61%) are state abbreviations.** `Ky.` retrieves the United
Kingdom, `WA` retrieves DR Congo, `N.M.` retrieves a Santa Fe hotel, `Ind.`
retrieves the Indus River — the query sorts by `alt_name_length` descending,
which is a fame prior with no relevance term.

## Traps found (worth reusing)

* **Prepending is not normalising.** The first R1 implementation prepended the
  expanded query's hits to the abbreviation's own list. At window 100 that
  evicts the tail: `Ky.` already had Kentucky at rank 6 and lost it, and the arm
  read +27/−9. Replacing the query — expanded hits at the head, non-duplicate
  originals behind — is both correct serving semantics and +28/−0.
* **`alt_name_length` in a pickled candidate is `log(n_altnames + 1)`, not
  `n_altnames`.** R4's thresholds were written against raw counts, so the rule
  silently never fired and its first arm read exactly baseline.
* **Recomputing document features on a held-out slice is not a no-op** for the
  one document the 70/30 split cuts through — its `adm1_count` denominator
  changes. Worth folding into the Phase-0 S2 document-id re-split.
* **e51's alias table needs one more guard than e51 gave it.** Its AP entries
  are keyed on the de-dotted lowercase form, so `AP["la"]` fires on the bare
  word `La` and `AP["miss"] / ["man"] / ["del"] / ["ore"]` on ordinary English
  words. Requiring a trailing period on the AP path closes that class and is a
  digit-identical no-op on all six held-out sources (all 62 AP firings carry
  the period). Residual, unpriced: `LA` in caps still expands to Louisiana, and
  in US newswire it usually means Los Angeles.

## Kept for reuse

* `hygiene.py` — the alias tables (from `e51_state_abbrev`) plus the four rules,
  each with the guard that keeps it from deleting a gold.
* `rebuild.py` + `parity_check.py` — a 0.0-parity re-derivation of a candidate
  list's features from live ES rows. Any future arm that edits candidate sets
  needs this gate; without it the within-set normalisation silently changes
  every remaining candidate's features.
* `eval_hygiene.py` — the `em_cond` vs `em_all` split. Recommended for any arm
  that can move the retrievable set; §4a of the report shows an arm that gains
  0.010 conditioned TLG-hard while losing 0.057 on a fixed denominator.
* `gaz_audit.py` — model-free counts of unretrievable / defunct / duplicate /
  missing-row mass, and the "would this rule delete a gold" check.

## Handed back

* **Serving**: adopt R1 (`serving_patch.diff` in the worktree). Three lines in
  `GeonamesService.build_name_search` plus a `mordecai3/place_aliases.py`.
* **Index rebuild (specified, not built)**: state abbreviations as real alternate
  names on 66 ADM1 rows; a stored `is_historical` + `superseded_by` so the sort
  can penalise defunct rows instead of the client dropping them; **replace the
  `alt_name_length` sort with a relevance sort** (47 of 1,721 TLG-hard golds sit
  past rank 100); whitespace-stripped name variants as alternate names.
* **Data debt**: 32 held-out golds point at geonameids that no longer exist in
  GeoNames (stale dump, not an index defect); a short list of plainly wrong GWN
  golds (`South Africa` → Durban, `Mount Chaambi` → an Algerian AREA row); 9
  golds that are themselves defunct rows.
