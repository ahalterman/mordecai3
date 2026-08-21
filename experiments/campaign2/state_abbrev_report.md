# US state-abbreviation aliases for the sibling/cue features — sizing pass

Campaign-2 Phase 2 arm, ledger id `e51_state_abbrev`. Written 2026-08-20.

**Verdict: REJECTED AT THE SIZING GATE. No training runs were done.** The
mechanism the hypothesis names is real and the abbreviations are really there,
but the ceiling is **+0.0009 TLG-hard** (+0.0004 on the 5-seed ensemble),
roughly a third of the pre-declared kill threshold (~0.003) and a tenth of
seed noise. A strictly larger variant that scans the raw document text rather
than the annotated mentions tops out at **+0.0017**. Even a fantasy ceiling
that flips *every* wrong entity in any document containing an abbreviation
reaches only +0.0073.

**One finding worth keeping, from the same alias table: 61% of all
unretrievable golds in held-out TR/LGL/GWN (27 of 44) are state-abbreviation
mentions that Elasticsearch cannot resolve at all** — "Ind." retrieves *Ind,
India*; "Ky." retrieves *Ky Ky Pan, South Africa*; "N.M." retrieves a hotel in
Santa Fe. That is a retrieval/serving item, not a ranker feature, and it is
invisible to TLG-hard by construction. See §6.

## 1. The hypothesis, and what the feature actually reads

`sib_adm1` (mordecai3/candidate_features.py, `add_document_features`) fires
when a candidate's `admin1_name` is, verbatim and lowercased, one of the
*other mention strings in the same document*:

```python
choice["sib_adm1"] = 1.0 if (norm(adm1) in siblings and adm1 not in ("", "NULL")) else 0.0
```

`siblings` is built from the `search_name` of the document's other entities, so
"Springfield, Ill." can only fire `sib_adm1` for the Illinois Springfield if
some other mention is literally the string "illinois". "Ill." is not. The
proposed fix is a static USPS + AP-style alias table applied to sibling
mention strings before that comparison.

Two things follow that bound the arm before any code is written:

* the abbreviation has to be an **annotated toponym mention** for the aliasing
  to see it at all (the enrichment pickles carry no document text — only
  `search_name`, `tensor`, `locs_tensor`, `doc_tensor`), and
* the abbreviation mention, being a toponym, **already contributes an anchor**
  to the `geo` block, which is where most of the same evidence arrives.

## 2. Corpus presence — the cue is there

Across both halves of TR/LGL/GWN (5,739 entities), **137 mentions (2.4%) are
US-state or Canadian-province abbreviations**, in 48 distinct strings:
`W.Va.` 13, `Ky.` 11, `B.C.` 10, `D.C.` 8, `Ind.` 7, `Kan.` 7, `Ill.` 6,
`W. Va.` 4, `N.C.` 4, `N.J.` 3, `Mo.` 3, `Conn.` 3, `S.C.` 3, `Fla.` 3,
`Okla.` 3, `N.D.` 3, `Calif.` 3, … plus two bare USPS codes (`SC`, `WA`) and
dotless forms (`Ky`, `Calif`, `N.Y`).

The alias table is not the limiting factor. Scanning every short or dotted
mention string in the three corpora, the only ones the table does *not* match
are countries (`U.S.` 125, `US` 75, `UK` 28, `UAE` 5, `EU` 1), cities
(`L.A.` 4, `Phila.` 2, `St. Paul` 2) and Australian states (`ACT`, `SA`; 4
mentions). **No US state abbreviation is missed.** Guards used: bare USPS codes
must be two uppercase letters with no dots (`IL`, never `Il`/`il`), which is
what keeps `L.A.` from becoming Louisiana; AP forms match after stripping a
trailing period and lowercasing, so `Ky`, `Calif` and `N.Y` are caught too.

Held-out (the 30% halves; 1,721 entities in TR/LGL/GWN):

| source | held-out | entities in a doc with ≥1 abbrev mention | abbrevs whose expansion is not already a mention |
|---|---|---|---|
| TR | 274 | 60 | 25 |
| LGL | 973 | 160 | 106 |
| GWN | 474 | 37 | 23 |
| Prodigy | 500 | 0 | 0 |
| WikiDocs | 6,456 | 9 | 9 |

Prodigy has none (its "documents" are single sentences); WikiDocs has nine,
none of which change a feature value.

## 3. Feature deltas the aliasing would produce

Computed directly on the enriched pickles: for every held-out entity,
recompute `sib_adm1`/`sib_adm2` with the alias-expanded sibling set and count
candidates whose value flips 0 → 1.

| | TR | LGL | GWN | total |
|---|---|---|---|---|
| entities where ≥1 candidate gains `sib_adm1` | 25 | 69 | 19 | 113 |
| entities where the **gold** gains `sib_adm1` | 25 | 33 | 13 | 71 |
| … of which e29 seed42 is **already correct** | 23 | 30 | 12 | 65 |
| … of which e29 seed42 is **wrong** | 2 | 3 | 1 | 6 |

Restricted to the TLG-hard slice (non-country gold, gold in window), 67
entities have the gold gain the feature and **only 2 of them are currently
wrong** — both LGL. That is the entire upside.

Meanwhile the downside exposure is larger: **40 currently-correct TLG-hard
entities** (LGL 34, GWN 6) have the new firing land *only on a non-gold
candidate* — a 20:1 ratio of rows that could be pushed the wrong way to rows
that could be pushed the right way.

## 4. Ceilings

Frozen per-entity predictions from `experiments/campaign2/preds/`
(`e29_seed42_w100.parquet`, `e29_ens5_w100.parquet`). TLG-hard here is macro
over TR/LGL/GWN of EM on non-`PCL*` golds, restricted to `gold_in_window`,
n = 1,171. **Note the window**: these preds are scored at window 100, so the
absolute baseline reads 0.8832 (seed42) / 0.9029 (5-seed ensemble) rather than
the ledger's 0.8730 (5 seeds, window 500, n = 1,219). The *deltas* are the
comparable quantity, and the touched-entity counts are window-independent.

Ceiling = every touched entity that is currently wrong is credited as correct.

| variant | entities touched | of which wrong | ceiling Δ, seed42 | ceiling Δ, ens5 |
|---|---|---|---|---|
| **A. gold gains `sib_adm1`** (the proposed fix) | 67 | 2 | **+0.0009** | **+0.0004** |
| B. any candidate gains `sib_adm1` | 107 | 2 | +0.0009 | +0.0004 |
| C. gold's ADM1 named by an abbreviation anywhere in the raw **document text**, not already a sibling | 110 | 4 | +0.0017 | +0.0013 |
| D. fantasy cap: every wrong entity in a doc containing any abbreviation mention | 227 | 11 | +0.0073 | +0.0073 |

Per-source EM ceiling under A, over all answerable held-out entities:

| source | e29 seed42 EM | ceiling | Δ |
|---|---|---|---|
| LGL | 0.9170 | 0.9192 | +0.0022 |
| TR | 0.9147 | 0.9147 | +0.0000 |
| GWN | 0.9363 | 0.9363 | +0.0000 |
| Prodigy | 0.8960 | 0.8960 | +0.0000 |
| WikiDocs | 0.9293 | 0.9293 | +0.0000 |

LGL — the corpus the hypothesis singles out, and the largest slice — is the
only source that moves at all, and its best case is +0.0022 against a seed
noise of ~0.010.

Variant C was measured because the hypothesis as stated talks about a "context
window", which is wider than the mention set the feature currently reads.
Document texts were recovered from `raw_data/spacyed/source_{tr,lgl,gwn}.spacy`
and aligned to the pickles exactly, by recomputing `sha1(mean token tensor)` =
`doc_key` (0 documents unmatched, all 1,721 held-out entities placed). Both a
loose regex and a comma-guarded one (`, IL`) were tried; the comma guard
*lowers* the ceiling (+0.0013 vs +0.0017) because it costs recall without
removing any of the four flippable rows.

C would also be a far more invasive change than A: the pickles carry no
document text, so a text-window feature means threading raw text through
`enrich_pickles.py` *and* `geoparse.py` to preserve train/serve parity.
+0.0017 does not buy that.

## 5. Why the ceiling is this small

Three independent reasons, each measured:

1. **The full state name is usually there too.** Of the state names an
   abbreviation would alias to in held-out TR/LGL/GWN, **41% are already
   present in the same document as a full-name mention** — the article says
   "Indiana" on first reference and "Ind." after. `sib_adm1` already fires.
2. **The `geo` block already carries the signal.** The abbreviation mention is
   itself a resolved toponym and therefore an anchor. On the 71 entities whose
   gold would gain `sib_adm1`, the gold **already** has
   `anchor_same_adm1_frac > 0` in 56% of cases (mean 0.248) and an anchor
   within 150 km in 41%. This is the general lesson: the marginal value of the
   `sib` block is bounded by what the `geo` block already sees, and campaign
   1's headline "P(gold) 4.0% → 46.6% when `sib_adm1` fires" was measured
   *before* the geometry features existed.
3. **The residual errors are not where the abbreviations are.** 65 of the 127
   TLG-hard seed42 errors have US golds, but only **11 of 127** sit in a
   document that contains an abbreviation mention at all — and that 11 is the
   fantasy cap in row D, not an achievable number.

## 6. Salvage: the alias table belongs in the ES query, not the ranker

The same scan turned up a separate and much cleaner defect.

Of the **44 held-out TR/LGL/GWN entities whose own mention string is a state
abbreviation**, e29 gets 36.4% right (seed42 and 5-seed ensemble alike, against
an ~88% baseline). The cause is not ranking — it is retrieval:

* **27 of the 44 have no retrievable gold at all** (`gold_retrievable` False).
* On the 17 that are answerable, e29 already gets **16 right**.
* Those 27 are **61% of every unretrievable gold in held-out TR/LGL/GWN**
  (44 unretrievable in total; 92 unanswerable at window 100).

Confirmed against the live index — the phrase query never surfaces the state:

| query | top ES hits |
|---|---|
| `Ind.` | Ind (PPL, IND), Īnd (PPL, IND), Ind. Seteråsen (HLL, NOR) |
| `Ky.` | Ky Ky Pan (PAN, ZAF), Kaôh Ky (ISL, KHM) |
| `N.M.` | Estación Aragón N.M (RSTN, MEX), Hampton Inn Santa Fe, N.M. (HTL, USA) |
| `WA` | Wa (PPL, BFA), Wa (PPL, LBR), Wa (PPL, MLI) |
| `Indiana` | Indiana (PPL, BRA), Indiana (ADM2, BRA), … |

Unretrievable golds are excluded from the campaign denominator, so fixing all
27 is worth **+0.0004 TLG-hard** — nothing. But at serving they are 27
mentions the geoparser answers wrongly or blanks, and each one also injects a
**garbage anchor** into its document's `geo` features, a quiet tax on the other
mentions in the same article.

Recommendation: fold the alias table into **mention normalisation before
retrieval** (an extra `should` clause expanding `Ind.` → `Indiana`), alongside
the Phase-1 serving fixes and the gazetteer-hygiene item — not as a ranker
feature arm. It is a coverage / end-to-end win with a measurable denominator in
`tools/end_to_end_eval.py`, and it needs no retraining. The table itself (51
USPS codes + 13 Canadian provinces + 62 AP forms, with the case guards) is in
`experiments/e51_state_abbrev/sizing.py` in the arm's worktree.

## 7. What was and was not done

* **Not done**: no edit to `mordecai3/candidate_features.py`, no
  re-enrichment, no training runs, no checkpoints. The kill criterion in the
  protocol (ceiling < ~0.003 TLG-hard) was met by a factor of three on the
  arm's own mechanism and by a factor of two on the most generous honest
  variant.
* `tests/test_feature_parity.py` is untouched and unaffected — the arm's whole
  premise was that `candidate_features.py` is shared by training and serving,
  and it was not modified.
* Cost: ~0 GPU minutes.

## 8. Reproducing

Scripts live in the arm's worktree
(`/home/andy/projects/mordecai3/.claude/worktrees/agent-a24c16dbf871e98a0`,
directory `experiments/e51_state_abbrev/`); run from the repo root with the
project venv.

| script | what it produces |
|---|---|
| `sizing.py` | the alias table; §2 corpus counts; §3 feature-delta table; variant A/B ceilings |
| `sizing_text.py` | recovers document text from the `.spacy` DocBins via `doc_key`; variant C |
| `sizing_why.py` | §4 combined ceiling table (seed42 and ens5); §5 redundancy and geometry numbers |
| `sizing_selfabbrev.py` | §6 — resolution of the abbreviation mentions themselves |

Inputs: `raw_data/pickled_es/es_formatted_{tr,lgl,gwn,prodigy,wiki_docs}_500_all_loc_types_fuzzy_0_enriched.pkl`,
`raw_data/spacyed/source_{tr,lgl,gwn}.spacy`,
`experiments/campaign2/preds/e29_{seed42,ens5}_w100.parquet`.
