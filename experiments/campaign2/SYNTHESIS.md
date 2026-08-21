# Campaign 2 kickoff: five-track review synthesis

Written 2026-08-20, after five parallel research tracks reviewed the completed
accuracy campaign (0.881 → 0.926 macro EM) and probed where the next points
live. Track reports, all in this directory:

- `audit_experiments_report.md` — methodology audit of the 60-experiment ledger
- `data_quality_report.md` — training/eval data health, leakage, label audit
- `calibration_report.md` — confidence, abstention, and the serving recipe
- `end_to_end_report.md` — full-pipeline (NER + resolution) accuracy
- `encoder_scoping_report.md` — the frozen-encoder bet, piloted; new text signals
- `serving_fixes_report.md` — implementation of the measured serving-side wins

New ledger entries: `experiments/e30_absw100/`, `experiments/e31_abstain_w3/`
(both REJECTED on 5 seeds, see below). New tools: `tools/calibration_eval.py`,
`tools/end_to_end_eval.py`. `tools/train.py` gained `--window` and
`--abstain-weight` (defaults byte-identical to e29, verified).

## 1. The campaign's result survives audit — with corrections

Every headline number reproduces exactly from the seed jsons; the e29 seed42
checkpoint reproduces its metrics to 0.0; ensemble and twin-credit claims check
out. Selection-on-test bias (60 configs chosen on the reporting sets, no dev
set) is estimated at +0.002–0.004 of the +0.045 — real but small, and the
winning config is stable under leave-two-seeds-out. Corrections to the record:

- **The Wikipedia reversal is overstated.** The one-flag contrast (e11c→e11b)
  is +0.0027 macro, n.s. Uncapped WikiDocs buys WikiDocs (+0.042) and *costs
  TR* (−0.021, t=6.2). More Wikipedia helps Wikipedia.
- **Several recipe ingredients are tiebreaks, not findings**: ls=0.05
  specifically, epochs 15 (10 and 12 scored higher without SWA), dropping
  `cf`. Solid at t(4): enriched features (+0.0345), SWA (+0.0067), the
  Wave-1b bundle (+0.0052). The campaign's `*` rule (|mean|>2 SE) was too
  loose; the criterion is now t(4) > 2.776.
- **"Every source improved" is means-only** (Prodigy t=1.57); "seed spread
  halved" is only true vs e14, not e0.
- Pairing buys ~5% variance reduction, not "pure signal" (ρ≈0.12 between
  same-seed runs); the SEs were honest, the justification wasn't.
- The root `mordecai_2026-08-20.pt` had been clobbered by the *rejected*
  e24_rstar arm; a same-day rerun replaced it with a genuine ship-recipe
  model, but **no campaign checkpoint is installed where the library loads
  from** (`geoparse.py` still points at `assets/mordecai_2025-08-27.pt`).
  The campaign has shipped nothing yet. Ship-artifact promotion is decision
  D3 below.

## 2. The scoreboard measures less than we thought (data + audit tracks)

Classic leakage is a non-issue — document-level train/held-out overlap costs
−0.0012 EM (all but Synth negligible), cross-source duplication is two
articles. The real findings are structural:

- **Answer-key memorisation**: 81% of held-out entities have their exact
  (mention → gold id) pair somewhere in training; EM there is 0.960 vs
  **0.771 on the 1,702 novel pairs**. The six sources cross-pollinate
  (11–56% of each corpus's held-out answers appear in another corpus's train
  half), so they are not independent tests.
- **The macro overweights easy data**: 38% of held-out golds are countries
  (EM 0.9985); Synth's held-out is 100% `syn_caps` gazetteer lookups; the
  same checkpoint reads 0.920 macro → 0.909 TR+LGL+GWN → **0.866 TR+LGL+GWN
  excluding countries** ("TLG-hard").
- **The model identifies the corpus** (93.3% from `doc_tensor` alone) and
  adapts its A/P convention per corpus — measurable overfitting to
  annotation idiosyncrasies.
- **Label audit** (100 adjudicated): ~12% of "errors" have wrong golds, ~28%
  are convention/duplicate-row disputes, 60% genuine. Roughly 40% of the
  residual error budget is unrecoverable under the current key; addressable
  residual ≈ 4.5 pts, not 7.4. The A-side convention is NOT WikiDocs-only
  (GWN parishes; WikiDocs internally inconsistent; TR/LGL disagree on
  "New York").
- **Synth's split is entity-shuffled** (14% of its held-out docs appear in
  train) while carrying 1/6 of headline weight. Fix or drop (D4).
- The reported EM further excludes unretrievable golds and eval never scores
  the reserved row: **0.9204 reported → 0.9072 over all mentions → 0.9019
  through the serving path.**

**New metrics adopted for campaign 2** (both implemented, cross-validated
between two tracks to within one entity): primary **TLG-hard** = macro over
TR/LGL/GWN of EM on non-country golds (e29 5-seed baseline **0.8730**);
secondary twin-credit on the same slice; guardrail **novel-pair EM**
(0.7714) — the only number memorisation cannot move. Macro-of-six is kept for
ledger continuity only.

## 3. End-to-end is the headline finding

Pooled over 2,381 held-out gold toponyms of TR/LGL/GWN, from raw text:

| measurement | EM |
|---|---|
| campaign metric (gold span given, spaCy-tagged, retrievable) | 91.2% |
| oracle NER (gold span given, all gold) | 80.1% |
| **end to end from raw text (ship config)** | **58.6%** |
| end to end with measured config fixes (trim+NORP+nested) | **72.9%** |

NER-side losses own 74–83% of the end-to-end gap — but only 4% of misses are
spaCy finding nothing. 52% are toponyms nested inside larger entities
("Paris" in *Paris Police Department*), 44% are rejected labels, mostly NORP
demonyms that all three corpora annotate with country ids. Measured fixes:
span trim +2.0 (free, strictly dominant → now default), NORP +6.3 at
unchanged precision (flag), nested gazetteer pass +7.9 at −19% throughput
(flag). ORG is a trap; `en_core_web_lg` is strictly worse; FAC costs 11.6
detection precision and 43% throughput for 1.1 recall (now optional).
Latency is 57–65% Elasticsearch, 29–38% spaCy, **2–3% ranker** — so ranker
ensembling is nearly free end-to-end, and latency work means ES work.

The remaining ~7 pts to the oracle ceiling = a **nested place-name tagger**
(token-classification head over the trf tensors the pipeline already
computes, trained on TR/LGL/GWN spans). That is the biggest lever in the
whole review: NER config + a small tagger is worth ~3 ranker campaigns.

## 4. Calibration and abstention: solved at serving time, not train time

The PI's anecdote is half-true: on answerable mentions, confident errors are
*closer* than unconfident ones (confidently-wildly-wrong = 0.22% of
mentions). It is true on **unanswerable** mentions (2.5% of serving traffic:
gold unretrievable or beyond the 100 window), where the model answers with
p>0.9 on 30%, median miss 583 km, costing 2.35 EM vs the reported number.

Structural discovery: **the abstention class already exists** — training
targets the reserved row when nothing is correct, serving already acts on it
(86% precision) — but eval has been blind to it, and train/eval/serving used
three different reserved-row conventions. Unified convention (recommended,
§2a of the calibration report): reserve the row everywhere, argmax decides
*which*, p(reserved) decides *whether*, eval reports abstention as a third
outcome.

**Serving recipe (implemented)**: expose `p_pred_full` = softmax incl.
reserved row at **T=0.874** (AUROC 0.899 for correctness; model was
*under*-confident, ECE 0.019→0.0115), plus `p_no_match` = reserved-row prob
(AUROC 0.869 for unanswerable). Default filter p≥0.7 → **90.6% coverage at
95.2% selective EM**. Seed-ensemble disagreement is not a useful signal;
the ensemble's value is a sharper probability.

**Training-side arms both REJECTED on 5 seeds**: e30_absw100 (train at
window 100 so out-of-window golds become abstain labels) triples the OOD
gain (unanswerable AUROC 0.858→0.931\*) but costs answerable EM (−0.0035\*,
TLG-hard −0.0092\*) and is flat on the decisive metric (selective EM @90%
n.s., AURC n.s.) — it relocates error. e31 (3× abstain class weight) is
dominated. Ledgered with NOTES.md.

## 5. The encoder bet, reframed

A naive swap of the frozen spaCy tensors for a modern encoder **loses**
(−0.035\*, 5 seeds): modern encoders beat spaCy by 10–23 pts at encoding
country/admin1 but lose 3–12 pts on *feature class*, and the mention-vector
cosine against the code table is the only channel separating same-named
city from county — spaCy's OntoNotes fine-tuning supplies exactly that.
Context-slots-only swap is nearly free (−0.007, GWN +0.009\*). Also found:
the text pathway is a **4-scalar bottleneck**; the "doc tensor" never
contained document context (144-word-piece striding); `locs_tensor` is dead
weight (−0.0019 n.s. to ablate); serving fed NORP into context entities
while training didn't (now fixed). Re-embedding all sources is ~80 s GPU +
1.6 GB — no spaCy rerun, no ES — so 50 s/run economics survive any frozen
swap. Representation-harness controls (centering, z-score) currently cost
up to 0.015 EM and must read 0.000 before any swap verdict is trusted.

Revised ladder: e40 fix harness → e42 context-slot swap (free win) →
e41 widen the pathway, carry BOTH mention representations → e43 typed
encoder → e45 fix/reclaim locs_tensor.

**New text signals**: outlet metadata is the strong one — LGL has
`<domain>` on 100% of articles, and **46.5% of LGL held-out errors have
gold in the outlet's home admin1 with the prediction elsewhere** (5.9%
would move the wrong way), covering 79% of LGL's no-document-evidence
errors. Needs an external domain→location mapping (only 9/32 held-out
outlets appear in training) and a neutral null for the 81% of entities
without it. The Wikipedia link-frequency prior is **poisoned** — its argmax
matches gold on 89.8% of WikiDocs errors because it IS that corpus's
label-generating process; an honest version needs a fresh enwiki pass
excluding WikiDocs titles. `loc_rank_db.jsonl` is the 2017 Prodigy export,
not a rank db.

## The campaign-2 ladder (priority order)

**Phase 0 — scoreboard repair (before any new arm is scored)**
- S1. Unify the reserved-row convention across train/eval/serving; make eval
  report abstention as a third outcome; log `total_missing` and the
  three-number accuracy (conditioned / all-mention / serving) in every json.
- S2. Re-key splits on document id; fix Synth's shuffled split (or drop
  Synth from the headline — D4); re-cut Prodigy at document level.
- S3. Adopt TLG-hard (primary) + twin-credit (secondary) + novel-pair EM
  (guardrail); keep macro-of-six for continuity. Statistical rule: t(4).
- S4. Dev/test separation: freeze the current six held-out sets as DEV; a
  small modern-news corpus with a written granularity convention becomes the
  untouched TEST (D1).
- S5. Promote a ship artifact into `mordecai3/assets/` and point
  `geoparse.py` at it (D3); guard `--checkpoint-out` against clobbering.

**Phase 1 — serving wins, no retraining (measured; implementation in
`serving_fixes_report.md`)**: span trim default-on; NORP / nested-gazetteer
/ FAC-drop flags (D2); calibrated confidence + `p_no_match` + p≥0.7 filter.

**Phase 2 — cheap feature arms**: outlet-location feature with a leak
protocol; US state-abbreviation aliases for `sib_adm1` ("Fowlerville,
Mich." — evidence present, feature never fires); gazetteer hygiene in the
ES build (duplicate rows, *H demotion, D.C. cluster); untainted wiki prior
(later).

**Phase 3 — representation**: e40–e45 above.

**Phase 4 — the NER campaign (biggest lever)**: nested place-name tagger
head over existing trf tensors, trained on TR/LGL/GWN spans; target the
~7 e2e points between config-fixed (72.9) and oracle (80.1). Separately: ES
is the latency budget — candidate caching/batching, not model work.

**Data debt (parallel)**: targeted re-adjudication of auto-screened bad
golds (PPLQ/*H rows, gold >100 km from all anchors); the modern-news test
corpus; Prodigy document-level re-annotation only if Prodigy stays in the
headline.

## Decisions — RESOLVED by Andy, 2026-08-20

- **D1. ADOPTED**: TLG-hard primary, twin-credit secondary, novel-pair EM
  guardrail; the current six held-out sets are frozen as DEV; a new
  modern-news corpus with a written convention becomes the untouched TEST.
  Macro-of-six reported for ledger continuity only.
- **D2. DEMONYMS ARE OUT OF THE TASK.** Exclude NORP/demonym gold spans
  from all end-to-end eval denominators (detection recall and e2e EM over
  non-demonym toponyms only), and REMOVE the `accept_norp` option and code
  path from the library. E2e numbers must be restated on the new
  denominator (~44% of the old NER misses were demonyms).
- **D3. SHIP e29 seed101** (0.9300 macro, single checkpoint): promote into
  `mordecai3/assets/` with its config sidecar, point `geoparse.py` at it.
  No ensemble in the serving path for now.
- **D4. Synth DROPPED from the headline, kept in the training mix**
  unchanged. (TLG-hard already excludes it.)
- **D5. The NER tagger campaign is the flagship**: nested place-name
  tagger head over the trf tensors the pipeline already computes, trained
  on TR/LGL/GWN gold spans, targeting the gap between config-fixed e2e and
  the oracle-span ceiling (both to be restated on the D2 denominator).

---

## CAMPAIGN 2 EXECUTION — PI synthesis (written 2026-08-20, end of day)

All tracks below ran and closed the same day the decisions were resolved.
Every number is on the D2 denominator (2,084 of 2,392 e2e golds; demonyms
out) unless marked legacy. Full reports live beside this file; ledger
entries in experiments/e5x_*/ and e60_docsplit/.

### The scoreboard, morning → evening

| serving stack (raw text, 260 held-out docs) | e2e EM |
|---|---|
| start of day (e29 s101, spaCy filter), D2-restated | 67.66 |
| + e54 outlet ranker (dropout 0.5, researched table) | 70.39 (w/ outlets) |
| + gold span head (span_detector="gold") | 80.85 |
| + R1 abbreviation query normalisation (default ON) | **82.29** |
| oracle-span ceiling after R1 | 89.05 |

Gains compose almost exactly additively (+15.30 vs +14.87 predicted).
Detection owns ~97% of the remaining gap; of the 84 residual retrieval
misses, 61% are the `alt_name_length` index sort, 14 are adjectival
spans the head newly finds. Ranker-side (conditioned) scoreboard:
TLG-hard 0.8676 → 0.9043 (e54, t=11.8), novel-pair 0.7731 → 0.8040.

### Track verdicts

- **Phase 0 (main tree, landed)**: D1–D4 implemented; reserved-row
  convention unified (honest baselines TLG-hard 0.8676 / macro 0.9214;
  legacy 0.8730/0.9258 still reproduce); seed101 promoted; checkpoint
  clobber guard; new metrics printed by every run; doc-keyed splits
  DON'T reduce memorisation (novel-pair rose — leakage is lexical), e60.
- **e51 state-abbrev ranker feature: KILLED at sizing** (+0.0009
  ceiling). Salvage became R1. Lesson: sib-family gains are now bounded
  by what the geo block already sees.
- **e50/e53/e54 outlet: SHIPPED (staged)**. +0.0368 TLG-hard;
  permutation-controlled; leak-audited; dropout p=0.5 removes the
  no-outlet regression (residual −0.005 is a floor ≈5 mentions, not a
  dropout-tunable); training on the independently researched table adds
  +0.0087 (TR gets homes). p=0.7 probed and rejected.
- **e55 NER label scaling: the scaling hypothesis FAILED** (82 runs).
  Wiki anchors (27% complete), silver nested, self-training all n.s.;
  curve flat from ~2,500 docs. Demonym negatives are a freebie (FPs −14
  to −24, nested +5.9, no new data). Binding constraint: in-domain
  COMPLETE nested gold — an annotation project, not a collection one.
- **e56 span head serving: landed behind span_detector flag** (default
  None). Head is FASTER than the spaCy-filter path (37.6 vs 44.8
  ms/doc — junk spans cost more in failed ES queries than the head costs
  in compute). gold head beats C_all e2e in every cell. LOCO gate AMBER:
  out-of-family never below the spaCy path (TR +5.9*, LGL +9.3*, GWN
  tie) but honest out-of-family EM is −0.6..+2.9, not +10.5. Calibration
  refit under the head: T=1.106 serving frame, knee at p≥0.5, AURC
  improves at every coverage, but p_no_match standalone unanswerability
  AUROC 0.895→0.797 (population shift) — thresholds must be re-picked.
- **e52 gazetteer census**: the "quarter of errors is 20 strings" memo
  was a six-source artifact; D.C. and Mauna Kea were misdiagnosed
  (A/P convention; whitespace-spelling trap). Dedupe keep-A is a metric
  trap (juices conditioned EM by deleting golds) — retrieval changes
  must always report em_cond AND em_all. Index-rebuild spec written,
  not built.
- **e57 R1: landed, default ON.** +28/−0 frozen-ranker, +1.44 best
  cell, raises the oracle ceiling itself (87.41→89.05), takes the whole
  abbreviation miss class. LA-risk priced: one benign misfire (WA→
  W. Australia candidate, ranker still answers Washington).

### Staged decisions for Andy (none applied)

1. Flip `DEFAULT_MODEL_ASSET` → `assets/mordecai_2026-08-20_e54_seed42.pt`.
2. Flip `span_detector` default None → `"gold"` (carry temperature≈1.10
   and document the p≥0.5 filter). "all" instead costs −0.72 EM for 25
   fewer demonym FPs (a GWN-policy column, mostly).
3. Both flips → 82.29/79.80 e2e (with/without outlets) from 67.66.
   Caveats: single head seed (±1.4), out-of-family expectation
   −0.6..+2.9, overlapping spans now an output category (24/2,013),
   5 Phase-0 xfails need re-attribution.
4. ES index rebuild per e52 spec — the `alt_name_length`→relevance sort
   is now the largest single lever left (51 of 84 residual misses).
5. Eval-data debt: 32 stale-geonameid golds, wrong GWN golds
   (S. Africa→Durban ×3 etc.), 9 defunct-row golds — owner adjudication.
6. TEST corpus (D1) remains the publication blocker; LOCO narrowed but
   did not close it.

### Suggested commit slices (owner commits; house style)

1. phase 0: D2 removal, metrics suite, reserved-row unification,
   seed101 asset, checkpoint guard, doc-split option (+phase0 report,
   e60).
2. outlet feature: outlet_features.py, torch_model/train/enrich wiring,
   serving args, e54 assets+table, tests (+e50/e53/e54 ledgers,
   outlet reports).
3. span head: mordecai3/span_head.py, head assets, geoparse flag,
   e2e harness variants, tests (+e55/e56 ledgers, reports).
4. R1 retrieval: place_aliases.py, geonames.py query hook, tests
   (+e57 ledger, r1 report, e52 census report).
5. ledger: PLAN.md campaign-2 entries, SYNTHESIS.md, remaining
   campaign2/ reports (e51 sizing, scoping reports).

### Next campaign, in expected-value order

1. Index rebuild + relevance sort (then retrain: features shift).
2. Adjectival/demonym → place alias table on the R1 query hook (14
   residual misses; NOT a task change — these are non-demonym golds the
   head finds as adjectival spans).
3. In-domain nested-gold annotation sprint (the e55 constraint) +
   the TEST corpus with a written convention — one annotation effort,
   two payoffs.
4. N4 single shared encoder (spans + ranker tensors from one roberta
   pass) — cheaper than today's pipeline and the e43 "typed encoder" in
   one; gated on the annotation sprint.
