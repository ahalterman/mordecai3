# Gazetteer hygiene (e52): the census, four candidate rules, and what survived

Campaign 2, Phase 2 arm `e52_gaz_hygiene`. Serving-side only — no retraining.
The frozen ship checkpoint `experiments/e29_swa_ep15/seed101.pt` (and `seed42.pt`
as a control) scores held-out candidate lists that have been transformed before
they reach the ranker. Ledger entry: `experiments/e52_gaz_hygiene/NOTES.md`.

Measured on the spaCy-span path with gold spans given — the same conditioning
every ledger number uses. **These are not comparable to the new end-to-end grid
(66.99 under the D2 denominator)**; they say what the ranker does once a span
exists, which is where a candidate-list transform lives.

---

## Headline

**One rule out of four is worth adopting, and it is not the one the campaign
memo predicted.**

| | verdict |
|---|---|
| **R1 abbreviation normalisation before the ES query** | **ADOPT.** +0.020 TLG-hard on a fixed denominator, 28 entities gained, **0 lost**, zero collateral on WikiDocs/Prodigy/Synth, reproduced on both seeds and both windows. Costs nothing: it replaces a query, it does not add one. |
| R2 demote defunct (`*H`, `PPLQ`) rows | Net **+2 entities in 8,977** under the frozen ranker. Fixes `Kathmandu` exactly as advertised and gives it back in feature-distribution knock-on. Index-rebuild item, not a serving item. |
| R3 collapse duplicate rows | **REJECT for serving.** It is a convention swap, not hygiene: keeping the P side costs 112 entities, keeping the A side costs ~1,500 — while the campaign's *conditioned* metric goes **up** in both cases, because the rows it deleted were golds. |
| R4 drop bare exact-name shadow rows | +0.003 TLG-hard alone; fixes `Mauna Kea` (19 entities). Real but small and threshold-sensitive. Fold into the index rebuild; ship only bundled, not alone. |

Bundled (R1+R2+R4) the arm is worth **+0.0225 / +0.0210 TLG-hard** (seed101 /
seed42, fixed denominator), +43 entities net.

**And three corrections to `ACCURACY_CAMPAIGN.md` §"What's left" item 3.** That
item reads: *"a quarter of the residual error mass is ~20 repeated strings
(D.C., Mauna Kea's duplicate rows, Kathmandu's historical ADM3H entry)."*
Measured:

1. **It is true of the six-source macro, not of the primary metric.** The top 25
   repeated strings are 32.3% of the non-country error mass over all six
   sources — but on TR/LGL/GWN alone (TLG-hard, the D1 primary) duplicate rows
   are **2.7%** of errors and defunct rows **0.5%**. All three named strings are
   WikiDocs.
2. **"D.C." is not a missing alt-name.** `4140963 Washington PPLC` is the
   **rank-0 hit** for the query `D.C.` The 20 errors are the A/P convention (the
   model answers `4138106 District of Columbia ADM1`, 4 km away) and twin credit
   already pays for them. Alias-normalising `D.C.` measurably changes nothing.
3. **"Mauna Kea's duplicate rows" is a spelling trap, not a duplicate.** The gold
   `5850911` is spelled **`Maunakea`** (one word, 43 alternate names);
   `6326699 Mauna Kea` is a different, unreferenced mountain on Maui, 150 km
   away, with 1 alternate name — and it wins because it is the only
   `exact_name_match`. Only `Kathmandu` is the class the memo said it was.

The thing the census *did* find is not in that list at all: **61% of every
unretrievable gold in held-out TR/LGL/GWN is a US-state abbreviation** whose
query returns countries.

---

## 1. The census

`census.py`, run against `experiments/campaign2/preds/e29_seed42_w100.parquet`,
with the gold and predicted rows fetched live from the `geonames` index.
Weighting is TLG-hard: TR/LGL/GWN, non-country golds (PCLI/PCL/PCLD/PCLS/PCLF/
PCLIX/TERR excluded). 1,263 non-country entities, **219 errors** (unretrievable
golds kept in the denominator).

### 1a. Root-cause mix

| cause | TLG-hard (n=219) | all six sources (n=846) |
|---|---|---|
| `wrong_place` — genuinely different place, >30 km | 141 (64.4%) | 405 (47.9%) |
| `retr_abbrev` — gold unretrievable, mention is a state abbreviation | **27 (12.3%)** | 27 (3.2%) |
| `granularity` — A/P or admin-level twin, same name, ≤30 km | 25 (11.4%) | 194 (22.9%) |
| `retr_other` — gold unretrievable, other reason | 13 (5.9%) | 76 (9.0%) |
| `dup_row` — co-located same-name rows, <1 km | 6 (2.7%) | 91 (10.8%) |
| `gold_row_gone` — gold geonameid not in the gazetteer at all | 4 (1.8%) | 32 (3.8%) |
| `historical` — a defunct `*H`/`PPLQ` row on either side | 1 (0.5%) | 18 (2.1%) |
| unknown | 2 | 3 |

The gazetteer-addressable classes (`retr_*`, `dup_row`, `gold_row_gone`,
`historical`) are **22.8% of TLG-hard errors** and 28.9% of the six-source
errors. Everything else is the ranker or the annotation convention.

### 1b. Top repeated strings, TLG-hard

25 strings, 105 of 219 errors (47.9%). `km` is the median gold↔prediction
distance; `−1` means the gold was never retrieved.

| mention | n | cause | gold | prediction | km |
|---|---|---|---|---|---|
| Richmond | 15 | wrong_place | Richmond PPL GBR | Richmond PPLA USA | 692 |
| Paris | 9 | wrong_place | Paris PPLA2 USA | Paris PPLC FRA | 6903 |
| Hanover | 6 | wrong_place | Hanover PPL USA | Hanover PPL USA | 583 |
| **Ind.** | **5** | **retr_abbrev** | Indiana ADM1 USA | Indianapolis PPLA USA | −1 |
| Lancaster | 5 | wrong_place | Lancaster PPLA2 USA | Lancaster PPL GBR | 5513 |
| Meadow Grove | 5 | wrong_place | Meadow Grove PPL USA | Meadow Grove PPL USA | 1254 |
| Springfield | 5 | wrong_place | Springfield PPLA2 USA | Springfield PPL USA | 1849 |
| Tipperary | 5 | wrong_place | Tipperary PPL IRL | County Tipperary ADM2 IRL | 31 |
| COLUMBUS | 4 | wrong_place | Columbus PPLA2 USA | Columbus PPLA USA | 1221 |
| Charles City | 4 | granularity | Charles City PPLA2 USA | Charles City County ADM2 USA | 1.7 |
| Columbus | 4 | wrong_place | Columbus PPLA2 USA | Columbus PPLA USA | 1037 |
| Edmonton | 4 | wrong_place | Edmonton PPL GBR | Edmonton PPLA CAN | 6788 |
| North Africa | 4 | wrong_place | Northern Africa RGN | Yakima PPLA2 USA | 9874 |
| Logan | 3 | wrong_place | Logan PPL USA | Logan PPLA2 USA | 1030 |
| London | 3 | wrong_place | London PPL CAN | London PPLC GBR | 5876 |
| New York | 3 | wrong_place | New York ADM1 USA | New York City PPL USA | 283 |
| Shanghai | 3 | granularity | Shanghai PPLA CHN | Shanghai Shi ADM1 CHN | 7.3 |
| **South Africa** | **3** | **retr_other** | **Durban PPLA2 ZAF** | South Africa PCLI ZAF | −1 |
| Twickenham | 3 | wrong_place | Twickenham PPLA3 GBR | Huntsville PPLA2 USA | 6825 |
| Berea | 2 | wrong_place | Berea PPL USA | Berea PPL USA | 471 |
| Hagerstown | 2 | wrong_place | Hagerstown PPL USA | Hagerstown PPLA2 USA | 637 |
| Hinton | 2 | wrong_place | Hinton PPL USA | Hinton PPLA2 USA | 1695 |
| Hopkinton | 2 | wrong_place | Hopkinton PPL USA | Hopkinton PPL USA | 193 |
| McKee | 2 | wrong_place | McKee PPLA2 USA | McKee School (historical) SCH | 614 |
| Miami Beach | 2 | wrong_place | Miami Beach PPL USA | Miami Beach BCH USA | 4436 |

**Reading it honestly: on the primary metric, this is a ranker problem, not a
gazetteer problem.** Sixteen of the top 25 are `Richmond`/`Springfield`/
`Columbus`-class same-name-different-place disambiguation. Two are the A/P
convention. Only `Ind.` (retrieval) and `South Africa` (a plainly wrong gold)
are squarely in e52's remit — plus `McKee`, where a `SCH (historical)` row beats
the county seat, and `Miami Beach`, where a `BCH` row beats the city; both of
those the R4 rule does reach.

### 1c. Top repeated strings, all six sources

Where the memo's examples live. 25 strings, 273 of 846 errors (32.3%).

| mention | n | cause | gold | prediction | km |
|---|---|---|---|---|---|
| Paris | 31 | dup_row | Paris ADM2 FRA | Paris PPLC FRA | 0.01 |
| **D.C.** | 20 | **granularity** (not alt-name) | Washington PPLC USA | District of Columbia ADM1 USA | 4.0 |
| **Mauna Kea** | 19 | **wrong_place** (spelling trap) | Maunakea MT USA | Mauna Kea MT USA (Maui) | 150 |
| Richmond | 15 | wrong_place | Richmond PPL GBR | Richmond PPLA USA | 692 |
| **Ireland** | 14 | **gold_row_gone** | *(2646052 — not in the index)* | Ireland PCLI IRL | — |
| Minsk | 14 | wrong_place | Horad Minsk ADM1 BLR | Mińsk Mazowiecki PPLA2 POL | 444 |
| Yangon | 12 | granularity | Yangon PPLA MMR | Yangon Region ADM1 MMR | 22 |
| Frankfurt | 11 | granularity | Frankfurt am Main PPLA3 | Frankfurt am Main ADM4 | 1.0 |
| Springfield | 10 | wrong_place | Springfield PPLA2 USA | Springfield PPL USA | 2227 |
| Aleppo | 9 | wrong_place | Aleppo Governorate ADM1 | Aleppo PPLA SYR | 41 |
| Homs | 9 | wrong_place | Homs Governorate ADM1 | Homs PPLA SYR | 147 |
| Daraa | 8 | granularity | Dar‘ā PPLA SYR | Daraa Governorate ADM1 | 28 |
| **Kathmandu** | **7** | **historical** | Kathmandu PPLC NPL | **Kathmandu District ADM3H** NPL | 1.3 |
| London | 7 | wrong_place | London PPLC GBR | London PPL CAN | 5876 |
| Bishkek | 6 | dup_row | Gorod Bishkek ADM1 KGZ | Bishkek PPLC KGZ | 0.9 |
| Damascus | 6 | wrong_place | Damascus Governorate ADM1 | Damascus PPLC SYR | 12 |
| Jacmel | 6 | granularity | Jacmel ADM3 HTI | Jacmel PPLA HTI | 7.1 |
| Leipzig | 6 | granularity | Kreisfreie Stadt Leipzig ADM3 | Leipzig PPLA3 DEU | 1.4 |
| Mannheim | 6 | granularity | Mannheim PPLA3 DEU | Stadtkreis Mannheim ADM3 | 2.5 |
| Salvador | 6 | wrong_place | Salvador PPLA BRA | Republic of El Salvador PCLI | 6304 |
| Al-Hasakah | 5 | granularity | Al Ḩasakah PPLA SYR | Al-Hasakah Governorate ADM1 | 14 |
| **Al-Zabadani** | 5 | **retr_other** | Az Zabadānī PPLA2 SYR | Al-Zabadani District ADM2 SYR | −1 |
| **Ind.** | 5 | **retr_abbrev** | Indiana ADM1 USA | Indianapolis PPLA USA | −1 |
| **Slavonia** | 5 | **gold_row_gone** | *(3205300 — not in the index)* | Požega-Slavonia County ADM1 | — |
| Solferino | 5 | dup_row | Solferino ADM3 ITA | Solferino PPLA3 ITA | 0.4 |

### 1d. Model-free gazetteer audit (`gaz_audit.py`)

Counted over every held-out candidate set; no model involved.

| | TR+LGL+GWN (n=1,721) | WikiDocs+Prodigy+Synth (n=7,256) |
|---|---|---|
| gold unretrievable | 44 | 95 |
| …of which the mention is a state/province abbreviation | **27 (61%)** | 0 |
| gold retrievable but past rank 100 | 47 | 41 |
| gold geonameid **not in the index at all** | 4 | 28 |
| entity has a defunct `*H`/`PPLQ` candidate | 1,045 (61%) | 3,855 (53%) |
| …a defunct row in the top 5 | 59 | 507 |
| **gold itself is a defunct row** | **1** | **8** |
| R2 fires / rows dropped / **golds destroyed** | 278 / 440 / **0** | 978 / 1,854 / **4** |
| R3 fires / rows dropped / **golds destroyed (keep P)** | 1,145 / 4,362 / **3** | 4,110 / 13,355 / **177** |
| R3 golds destroyed (keep A) | 105 | 449 |
| R4 fires / rows dropped / **golds destroyed** (all six sources) | 1,777 / 3,709 / **5** | — |

Two things fall out immediately. **R2's "a live twin must exist" guard works**:
zero golds destroyed on TLG, 4 of 9 protected on WikiDocs. And **R3 is not a
hygiene rule at all**: the 3-vs-105 and 177-vs-449 asymmetry *is* the A/P
annotation convention, measured without a model. TR/LGL/GWN annotate the P side;
WikiDocs annotates both.

---

## 2. The four rules

All in `hygiene.py`, each attributable to a census row. No speculative rules.

**R1 `abbrev` — mention normalisation before the ES query.** `build_name_search`
phrase-matches the mention against `alternativenames` and sorts by
`alt_name_length` descending, which is a fame prior. Verified live: `Ky.` →
United Kingdom, United States, Turkey; `WA` → DR Congo, UAE, Central African
Republic; `Ind.` → Indus River, Indianapolis, Indore (no Indiana anywhere in the
list); `N.M.` → a railway station in Mexico and a Santa Fe hotel. The rule
expands the mention through a 51 USPS + 13 Canadian + 62 AP-dotted alias table
(reused from `experiments/e51_state_abbrev/sizing.py`) and sends the expanded
query instead.

Two guards, both load-bearing. e51's: a bare two-letter code expands only when
the whole mention is that code, in caps, with no dots — `WA` and `NC` yes,
`Wa` and "in LA" no. And one this arm added: **an AP form expands only when the
mention actually ends in a period.** Without it, `AP["la"]` fires on the bare
word `La`, and `AP["miss"]`, `AP["man"]`, `AP["del"]`, `AP["ore"]`, `AP["ind"]`
on ordinary English words. All 62 AP firings in the six held-out sources are
written with the period, so the guard costs exactly nothing measured (the arm
re-runs digit-identical with it on) and closes the whole false-positive class.

One residual risk the held-out data cannot price: `LA` in caps still expands to
Louisiana, and in US newswire `LA` usually means Los Angeles. No held-out
mention is `LA`, so this arm has no evidence either way. If it bites in
production, drop `LA` (and arguably `IN`, `OR`, `ME`, `OK`) from the bare-code
table and keep only their dotted forms.

*Semantics matter here.* The first implementation **prepended** the expanded
query's hits to the original list. That is wrong and it cost 9 entities: at a
serving window of 100, prepending 100 rows evicts the tail — `Ky.` already had
Kentucky at rank 6 and lost it. Replacing the query (expanded hits at the head,
originals that are not duplicates behind them) is both the correct serving
semantics and the version that scores +28/−0.

**R2 `demote_h` — drop a defunct row when a live one for the same place is in
the list.** Guard: same country, same admin-word-stripped name key, within
25 km, and the live row must exist. Evidence: `Kathmandu` → `1283241 Kathmandu
District ADM3H`, carrying population 1,264,684, which wins the population
features over the live `1283240 Kathmandu PPLC`. The guard is what keeps
`Varzaqan` (TR #246, gold is a `PPLQ` row and nothing live shares the name)
alive.

**R3 `dedupe` — collapse co-located same-name rows.** Same country, same stripped
key, within 1 km, one representative kept. Evidence: `Solferino` ADM3/PPLA3 at
0.4 km, `Paris` ADM2/PPLC at 0.01 km, `Bishkek` ADM1/PPLC at 0.9 km. Included in
the grid precisely to test whether e24's "moving golds within a twin class is
worth exactly 0.0000" also holds when you move *candidates* instead of labels.
It does, and worse.

**R4 `demote_junk` — drop a bare unreferenced exact-name row shadowing a notable
twin.** A candidate with ≤2 alternate names, feature class not A or P, whose
whitespace-stripped name equals the mention, when another candidate in the same
country with the same stripped name has ≥10 alternate names. Evidence:
`Mauna Kea` (`6326699`, 1 alt name, pop 0, on the wrong island) beating
`5850911 Maunakea` (43 alt names) 19 times, purely because `exact_name_match` is
whitespace-sensitive. Also reaches `Miami Beach` BCH and
`McKee School (historical)` SCH.

> A trap worth recording: R4's thresholds were first written against raw
> alternate-name counts, but `res_formatter` stores
> `log(len(alternativenames) + 1)`. The rule silently never fired and the first
> arm read exactly baseline. `n_altnames()` now inverts the log. Anyone writing
> a rule against a pickled candidate field should check the units.

---

## 3. How it was measured, and what could have faked it

Three things make this arm easy to fake, so each has a gate.

**(a) The campaign metric conditions on a retrievable gold.** `evaluate_results`
skips an entity whose gold is not in the candidate list. A retrieval fix
therefore *adds* the hardest entities back into the denominator and can lower
the reported number while strictly helping; a rule that *deletes* golds raises
it while strictly hurting. Every table below carries both:

* `em_cond` — the campaign convention, moving denominator (ledger continuity).
* `em_all` — every held-out mention in the denominator, an unretrievable gold
  scored wrong. **This is the number to read.**

R3 is the demonstration. `dedupe --keep a` raises conditioned TLG-hard from
0.8551 to **0.8649** and the six-source macro from 0.9214 to **0.9263** — while
`em_all` TLG-hard collapses from 0.8244 to **0.7670**. It looks like the best arm
in the grid and it is by far the worst.

**(b) Editing a candidate list changes the features of the candidates that
remain.** `min_dist`/`max_dist`/`avg_dist`/`ascii_dist` are min–max normalised
within the set, and the whole `WITHIN_SET_KEYS` enrichment block
(`is_max_pop_exact_match`, `log_n_exact_matches`, `is_unique_exact_match`, …) is
computed over it. So every touched entity is **rebuilt from live ES rows through
the same arithmetic `geoparse.res_formatter(..., extra_features=True)` uses**,
and its document's features are recomputed with `_add_cross_entity_counts` +
`add_document_features`.

`parity_check.py` is the gate: rebuild with **no rule enabled** and diff against
the frozen pickles.

```
1. candidate rebuild: 274 entities checked, 0 skipped
   max |diff|: 0.0 on all 23 keys
2/3. document recompute over 29 docs
   max |diff|: {'adm1_count': 0.075, 'country_count': 0.075}
```

Per-candidate parity is exact, including every enrichment key. The `adm1_count`
residual is **not** a hygiene effect: it is confined to exactly **one document
per source** — the one the 70/30 split cuts in half, whose held-out slice is a
fraction of the document the pickle's counts were computed over. It is a
pre-existing split artifact (one more argument for Phase-0 S2's document-id
re-split) and it touches 5 of TR's 274 entities.

**(c) Train/serve skew is real and is reported, not excused.** The ranker was
trained on un-hygienic candidate sets. Shortening a list moves feature
distributions the model never saw. `diff_arms.py` splits every flip into
**direct** (this mention's own list was edited) and **knock-on** (a neighbour was
edited and the document geometry moved), which is exactly where that skew shows
up — and, symmetrically, where the PI's predicted garbage-anchor repair shows up.

---

## 4. Results

Frozen `e29_swa_ep15` checkpoints. The transform is deterministic and involves
no seeds; `seed42` is here to show the effect is not checkpoint-specific.

### 4a. Arm grid — seed101 (ship), serving window 100, all six sources

| arm | TLG-hard `em_cond` | TLG-hard `em_all` | macro-of-6 `em_cond` | macro-of-6 `em_all` |
|---|---|---|---|---|
| **base** | 0.8551 | 0.8244 | 0.9214 | 0.9083 |
| **R1 abbrev** | 0.8588 | **0.8444** | 0.9224 | 0.9151 |
| R2 demote_h | 0.8551 | 0.8244 | 0.9221 | 0.9089 |
| R4 demote_junk | 0.8593 | 0.8273 | 0.9223 | 0.9087 |
| R3 dedupe keep-P | 0.8567 | 0.8239 | *0.9233* | 0.9053 |
| R3 dedupe keep-A | *0.8649* | **0.7670** | *0.9263* | 0.8638 |
| **R1+R2+R4 ("ship")** | **0.8625** | **0.8469** | **0.9239** | **0.9160** |

*Italics mark numbers that look like wins only because golds left the
denominator.*

Per source, `em_cond`/`em_all`:

| arm | TR | LGL | GWN | WikiDocs | Prodigy | Synth |
|---|---|---|---|---|---|---|
| base | .8930/.8832 | .8848/.8602 | .9239/.8966 | .9233/.9099 | .9300/.9300 | .9732/.9700 |
| R1 abbrev | .8938/.8905 | .8890/.8808 | .9249/.9093 | .9233/.9099 | .9300/.9300 | .9732/.9700 |
| R2 demote_h | .8930/.8832 | .8848/.8602 | .9239/.8966 | .9240/.9100 | .9300/.9300 | .9766/.9733 |
| R4 junk | .8930/.8832 | .8876/.8602 | .9283/.9008 | .9248/.9111 | .9300/.9300 | .9699/.9667 |
| R3 keep-P | .8930/.8832 | .8856/.8592 | .9259/.8966 | .9318/**.8927** | .9300/.9300 | .9732/.9700 |
| R3 keep-A | .9055/**.8394** | .8922/**.7996** | .9305/**.8755** | .9272/**.8541** | .9274/.8940 | .9753/**.9200** |
| ship | .8938/.8905 | .8907/.8798 | .9292/.9135 | .9264/.9120 | .9300/.9300 | .9732/.9700 |

**R1 is exactly zero on the three sources it cannot reach.** WikiDocs, Prodigy
and Synth are digit-for-digit identical to baseline under R1, on both seeds.
That is the collateral-damage answer: there isn't any.

### 4b. Both seeds, both windows

TR/LGL/GWN, non-country golds.

| arm | seed | window | `em_all` base → arm | Δ | `em_cond` base → arm | Δ |
|---|---|---|---|---|---|---|
| R1 | 101 | 100 (serving) | 0.8244 → 0.8444 | **+0.0200** | 0.8551 → 0.8588 | +0.0037 |
| R1 | 101 | 500 (ledger) | 0.8458 → 0.8675 | **+0.0217** | 0.8771 → 0.8822 | +0.0051 |
| R1 | 42 | 100 | 0.8166 → 0.8347 | **+0.0181** | 0.8473 → 0.8491 | +0.0018 |
| R1 | 42 | 500 | 0.8343 → 0.8527 | **+0.0184** | 0.8654 → 0.8673 | +0.0019 |
| ship | 101 | 100 | 0.8244 → 0.8469 | **+0.0225** | 0.8551 → 0.8625 | +0.0074 |
| ship | 42 | 100 | 0.8166 → 0.8376 | **+0.0210** | 0.8473 → 0.8532 | +0.0059 |

Six for six, same sign, same magnitude. Six-source macro `em_cond` moves
0.9214 → 0.9239 (seed101) and 0.9125 → 0.9139 (seed42) under the bundle. For
reference the ledger's 5-seed e29 TLG-hard baseline is 0.8730 at window 500;
seed101 alone reads 0.8771 in this harness, which is the seed, not a harness
discrepancy (seed42 reads 0.8654).

### 4c. Entity-level, seed101 window 100 (all six sources, n=8,977)

| arm | gained | lost | net | direct | knock-on |
|---|---|---|---|---|---|
| R1 abbrev | 28 | **0** | **+28** | +27 / −0 | +1 / −0 |
| R2 demote_h | 10 | 8 | +2 | +0 / −1 | +10 / −7 |
| R4 demote_junk | 23 | 14 | +9 | (see note) | — |
| R3 dedupe keep-P | 22 | **134** | **−112** | +0 / −112 | +22 / −22 |
| ship (R1+R2+R4) | 59 | 16 | **+43** | +27 / −1 | +32 / −15 |

R1's gains, by string: `Ind.` ×5, `Mich.` ×2, `N.C.` ×2, `Minn.` ×2, `Neb.` ×2,
`N.M.` ×2, and one each of `Ark.`, `N.J.`, `Conn.`, `B.C.`, `Ky.`, `S.C.`,
`N.D.`, `Tenn.`, `Mont.`, `Wis.`, `N.B.`, `Ill.` — plus one **knock-on gain**,
`BELGRADE` in LGL, which is precisely the effect the PI flagged: an unretrievable
`Mont.` was injecting a garbage anchor into its document's sibling geometry, and
repairing it moved a neighbour. Twenty-one of the 28 were previously
`retrievable = False`.

R2's losses are the train/serve skew made visible: 7 of 8 are knock-on, and the
one direct loss is `Dunedin`, whose **gold is** `2191561 ADM1H`. R4's flips are
+19 on `Mauna Kea` and +2 on `Miami Beach` against 14 single-entity knock-on
losses (`Berlin`, `Davos`, `Melbourne`, `Dublin`, …), i.e. the same skew.

*(Note: the direct/knock-on split needs both runs to carry the per-entity
candidate count; the R4 row's baseline predates that field, so its flips fall
through to the knock-on bucket. Its direct effect is the 21 `Mauna Kea` /
`Miami Beach` gains.)*

R3's 134 losses are 112 direct: `Geneva` ×47 (gold `7285902 ADM3`), `Minsk` ×19,
`Solferino` ×11, `Lausanne` ×5 — WikiDocs' A-side convention, deleted out of the
candidate list by a rule that decided the P side is canonical.

### 4d. Alias-rule firing profile

65 firings in 8,977 held-out mentions (0.72%): 62 dotted AP forms, 3 bare
two-letter codes (`SC`, `WA`, `NC` — all correct expansions). **Zero false
expansions.** The largest single group is `D.C.` ×20 in WikiDocs, which is inert
(Washington PPLC is already rank 0 for `D.C.`; the rule neither helps nor hurts).
Latency: zero — one query is replaced by another, none is added.

---

## 5. Recommendation, three ways

### (i) Serving-side, adopt now

**Ship R1, and only R1.** Sketch in
`experiments/e52_gaz_hygiene/serving_patch.diff` (worktree copy): a
`mordecai3/place_aliases.py` holding the tables, and three lines in
`GeonamesService.build_name_search` that expand the mention *before*
`_clean_search_name`. Order matters — `_clean_search_name` strips the token
"District", so expanding `D.C.` to "District of Columbia" would send the query
"of Columbia"; the table sends "Washington, D.C." instead.

Nothing else in the pipeline moves: `res_formatter` measures edit distances
against the original mention string, so every candidate feature keeps its
training-time meaning. **+0.020 TLG-hard on a fixed denominator, 0 regressions,
0 latency, no retraining.**

Three riders:
* It should only fire on spans the tagger already labelled as places. Do not
  lift the table into a general text normaliser — `Miss.`, `Del.`, `Man.` and
  `Ore.` are ordinary English words outside a place span.
* Keep the trailing-period guard on the AP forms and e51's caps-and-no-dots
  guard on the bare codes (§2). Consider dropping `LA` from the bare-code
  table on prior grounds; held-out has no evidence either way.
* If the pickles are ever rebuilt, rebuild them **with** R1 enabled, so training
  sees the repaired lists too. The +0.020 here is what a frozen model gets from
  better retrieval; a retrained one should get more, because 27 of the entities
  it can now see were previously unlearnable.

R2 and R4 are defensible as a bundle (+0.0225/+0.0210 vs R1's +0.0200/+0.0181)
but the increment is 15 entities net against 15 knock-on losses, and both are
better expressed as index data than as client-side drops. If you want the
bundle, take it — it is not harmful — but the honest attribution is that R1 does
the work.

This also lands on the bottleneck the span head (e56) just exposed: with
`retrieval_miss` up 88→112 on held-out, the query string is now the constraint.

### (ii) ES index rebuild — spec, not built

The live index was treated as read-only throughout; nothing below was executed.
Index `geonames`, 12,571,784 docs, 1.9 GB, one shard.

1. **Add the abbreviation aliases as real alternate names** (makes R1
   unnecessary at query time and helps fuzzy/partial matching too). For each of
   the 51 US-state + 13 Canadian-province `ADM1` rows plus Puerto Rico and the
   District of Columbia: append the USPS code and the AP dotted form to
   `alternativenames`, recompute `alt_name_length`. **66 documents updated, 0
   added, 0 deleted**; size delta ~4 KB.
2. **Demote rather than delete defunct rows.** Add a stored `is_historical`
   boolean (already a model feature, currently derived client-side) and a
   `superseded_by` geonameid for any `*H`/`PPLQ` row that has a live same-name
   row within 25 km in the same country. On the held-out candidate sets alone
   that is **2,294 rows across 1,256 entities**; corpus-wide the scan population
   is the `ADM[1-5]H / ADMDH / PPLH / PCLH / RGNH / PPLQ / PPLW` families. **Do
   not drop them** — 9 held-out golds are defunct rows. With the flag stored, the
   retrieval sort can apply a fixed penalty instead of the client deleting
   candidates, which is the version of R2 that does not have to fight a frozen
   model's feature distribution.
3. **Fix the retrieval sort.** `sort: {alt_name_length: desc}` is a pure fame
   prior with no relevance term. It is why `Ky.` returns the United Kingdom, and
   why `Paris, Ontario` (`6942553`, 1 alternate name) is unreachable for the
   mention "Paris" at any window. **47 of 1,721 TLG-hard held-out golds sit past
   rank 100**, and a handful past 500 — `Charlotte`, `Cambridge City`, `Atlanta`,
   `Richland`, all tiny same-name PPL rows. Spec: sort by `_score` with
   `alt_name_length` as tie-break, or a `function_score` combining the two; then
   re-measure candidate recall at 100/500. This changes every candidate list, so
   it is a rebuild-and-retrain item rather than a serving flag — and it is
   probably the largest retrieval lever left in the system.
4. **Fold R4 in as data, not as a drop rule.** The `Mauna Kea` failure is
   `exact_name_match` being whitespace-sensitive. Cleanest build-time fix: where
   a row's `name`/`asciiname` contains a space and the whitespace-stripped form
   is not already an alternate name, add it (`Maunakea` ↔ `Mauna Kea`), so the
   notable row matches exactly too and the shadow row loses its only advantage.

Not recommended in any form: collapsing duplicate rows in the index (§4c).

### (iii) Data issues — for the PI, not for me

I did not touch any eval data.

1. **32 held-out golds point at geonameids that are not in GeoNames at all**
   (4 in TLG, 28 in WikiDocs): `Ireland` `2646052` ×14, `Slavonia` `3205300` ×5,
   `Darfur` `376731` ×4, `Jounieh` `6941107` ×4, `Red Sea` `235615` ×2,
   `Black Sea` `6640396`, `Hillsboro` `5087451`, `Yugoslavia` `8505035`. These
   are **stale labels from an older GeoNames dump, not an index-build defect** —
   the live index has `408666/408658/408660 North/West/South Darfur` (the 2011
   split of the old row), `273140 Jounieh`, `8505033 Serbia and Montenegro`.
   They are unrecoverable by any model and are pure noise in the denominator.
   GeoNames publishes deletion and merge histories; remap or drop them.
2. **Golds that are simply wrong.** Found while auditing the unretrievable set,
   so all of these are on the primary metric: GWN `South Africa` → `1007311`
   **Durban** ×3; GWN `Latin America` → `6255150` **South America**; GWN
   `Mount Chaambi` → `2496952` *Eulb ech Chaambi*, an **AREA in Algeria**
   (Chaambi is in Tunisia); GWN `South Florida` → `4155751` **Florida ADM1**;
   GWN `Paddock Hill Bend` → `6618982` **Brands Hatch** (a corner of a circuit
   annotated to the circuit — arguably out of task); LGL `Richland`/`RICHLAND` →
   `5076207`, a Nebraska village of 73 people. Consistent with the
   data-quality track's ~12% gold-wrong rate, and concentrated in GWN.
3. **The A/P convention, quantified as candidates rather than labels.** A dedupe
   rule keeping the P side destroys 3 golds in TR/LGL/GWN and 177 in WikiDocs;
   keeping the A side destroys 105 and 449. That asymmetry *is* the convention
   split, measured without a model, and it independently confirms
   `data_quality_report.md`: TR/LGL/GWN are P-side, WikiDocs is both. It also
   confirms the decision to adopt twin credit rather than rewrite labels — there
   is no candidate-level fix either.
4. **Nine held-out golds are defunct rows**: `Varzaqan` PPLQ (TR), and in
   WikiDocs `Dunedin` ADM1H ×2, `Hirara` ADM2H, `Naze` ADM2H, `East Germany`
   PCLH, `Warsaw` PPLH, `Okanagan` PPLQ, `Košutarica` PPLW. Worth
   adjudication under the same
   PPLQ/`*H` screen the data-quality report proposed — but note that whatever is
   decided, any `*H`-demoting rule has to keep working when the gold is one.

---

## 6. Reproducing

Everything is in the worktree
`/home/andy/projects/mordecai3/.claude/worktrees/agent-a8c4f7c56da33eaf9/experiments/e52_gaz_hygiene/`:

| file | what |
|---|---|
| `census.py` | the repeated-error census, root causes from live ES |
| `gaz_audit.py` | model-free audit: unretrievable / defunct / duplicate / missing-row counts |
| `hygiene.py` | the four rules and the alias tables |
| `rebuild.py` | faithful re-derivation of one entity's candidate features |
| `parity_check.py` | the 0.0-parity gate that has to pass first |
| `eval_hygiene.py` | the arm runner (`em_cond` / `em_all`, per source, TLG-hard) |
| `diff_arms.py` | per-entity flips, direct vs knock-on |
| `run_grid.py`, `summarise.py` | the grid and its table |
| `probe.py`, `es_util.py` | read-only ES helpers |
| `serving_patch.diff` | the R1 change against `mordecai3/geonames.py` |
| `res_*.json`, `census*.json`, `audit_*.txt` | every number above |

```
python parity_check.py --source TR --n 300        # must read 0.0 first
python census.py --sources TR,LGL,GWN
python gaz_audit.py --sources TR,LGL,GWN
python run_grid.py --arms base,abbrev,demote_h,junk,dedupe_p,dedupe_a,ship \
                   --seeds 101,42 --windows 100
python diff_arms.py res_base_s101_w100.json res_abbrev_s101_w100.json
```

The worktree branch was cut from a pre-campaign commit, and `git reset` / `git
merge` inside it were both refused by the permission classifier, so every script
imports `mordecai3` and `tools` from the main tree read-only via `sys.path` and
writes only inside the worktree. **The `geonames` index was never written to**:
`es_util.py` issues only `_mget`, `_search` and `_count`.
