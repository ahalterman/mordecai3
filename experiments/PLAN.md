# Accuracy campaign: experiment plan

Updated 2026-08-19 after Phase 1 (audit + error analysis + baseline).

Goal: raise toponym resolution accuracy. Headline metrics: `_last5.exact_match_avg`
and `_last5.acc_at_161` (unweighted mean over the 6 held-out sources; exact match
is computed only over examples whose gold is retrievable, which makes it
comparable across NULL-handling designs). Methodology: 5 seeds
{42, 101, 202, 617, 1848}, paired per-seed deltas vs e0_baseline, score = mean of
last 5 epochs, `*` when |mean delta| > 2 SE. Training is bit-for-bit reproducible
at fixed seed, so same-seed differences are pure signal. One run ≈ 52 s.

Baseline (e0): `train.py train --epochs 30 --mix-dim 512
--dataset-names "Prodigy, TR, LGL, GWN, Synth, WikiDocs" --source-limits "WikiDocs=2000"`
→ exact match 0.8808 ± 0.0038, acc@161 0.9262 ± 0.0029.

## Phase 1 findings (full reports: audit + error analysis, 2026-08-19)

1. Retrieval is NOT the constraint: gold is in the 500-candidate list 98.5%
   of the time (97.4% TR+LGL+GWN) vs 86.4% model accuracy. ~12pt of ranker
   headroom. Retrieval work deferred (≤1.5pt ceiling, breaks comparability).
2. 55% of all errors are granularity: gold admin unit vs same-named co-located
   city (70% of WikiDocs errors, partly label artifacts). Geographically ~0km.
3. The hard real errors are same-name same-country ambiguity (75% of LGL errors
   are US-internal, e.g. Denver OH vs CO). Needs context, not priors.
4. Population prior ceiling is +2.7pt and a coin flip (53%) on same-name errors.
   Population is worth adding as a feature but will not fix the big error classes.
5. Confirmed bugs (audit): B1 double softmax (softmax in forward + CrossEntropyLoss);
   B5 unmasked padding (~46% of training slots, raw 99s into the MLP); B2/B3
   sentinel-slot clobbering + missing country override; B4 OOV feature codes
   collide with NULL embedding (18.9% of TR rows); B6/B7 NULL row and empty-ADM1
   always get max overlap features; B8 normalize() isn't [0,1]; B9 dropout on
   frozen embeddings and cosine inputs; B13 "Kansas City"→"Kansas" query cleaning;
   train/inference mismatches: 500 vs 100 candidate window (4x feature-scale
   shift via normalize()), per-token context entities in old formatters,
   eval/train disagree on the NULL slot (metrics note above).

## Wave 1 — training-signal fixes (DONE 2026-08-19)
Flags added to train.py/torch_model.py, defaults preserve current behavior;
regression-checked bit-for-bit vs e0 seed42. Ladder (5 seeds, paired vs e0):
- e1_logits (--logits): -0.0129 EM* — see finding below.
- e2_mask (+--mask-padding): neutral vs e1 (model had learned to ignore padding).
- e3_oov (+--oov-bucket-fix): +0.003 vs e2, right sign, within noise.
- e4_mlp (+--modern-mlp): best of the fixed family, -0.0093 EM* vs e0,
  acc@161 -0.0013 (parity). Peak EM 0.8859 @ epoch ~17 vs e0 0.8876 @ ~26.

KEY FINDING: the double softmax was an accidental extreme regularizer. With
the correct loss, train loss drops 5.35 -> 0.20, the model reaches e0's
epoch-25 accuracy in ~5 epochs, peaks ~epoch 17, then overfits downward.
Lower LR makes it worse (peak moves earlier); dropout 0.5 doesn't close it.
The 30-epoch lr=1e-3 recipe is tuned around the broken loss. Adopt the fixed
stack only together with a replacement regularizer (Wave 1b). Also: the
"more Wikipedia buys nothing" result was measured under the broken loss;
with a fittable loss and visible overfitting, data scaling is being retested.

## Wave 1b — re-regularize the fixed loss (DONE 2026-08-19)
WINNER e11b_combo_ls: fixed stack + --label-smoothing 0.05 --epochs 15 +
uncapped WikiDocs -> +0.0037 EM* / +0.0101 acc@161* vs e0 (abs EM 0.8845).
NEW STANDARD RECIPE AND COMPARISON BASELINE (paired vs e11b_combo_ls seeds).
- Label smoothing is the regularizer that replaces the double softmax
  (raises the peak to ~0.890 @ epoch ~12); weight decay does nothing.
- Three complementary ingredients: smoothing lifts peak, 15 epochs stops at
  it, uncapped data makes 15 epochs enough. Neither refinement arm
  (ls 0.1 / 22 epochs) beat it.
- WIKIPEDIA REVERSAL: uncapped WikiDocs (+15,065 train entities) now helps;
  WIKI_TRAINING_DATA.md's "more buys nothing" was a double-softmax artifact.
  (Memory + that memo's status updated.)
- Trap fixed: CrossEntropyLoss(label_smoothing) x --mask-padding puts
  smoothing mass on -1e9 padded classes -> masked_smoothed_ce in train.py.
- Prodigy fights the recipe (-0.04 under smoothing+data; sentence-level docs,
  different annotation style) -> needs its own investigation (e21_prodigy).
- Concurrency verified: 5-way parallel runs bit-identical to sequential.

## Wave 2 — features (enrichment script in progress)
Enriched pickles add per-candidate: log_population, exact_name_match,
exact_altname_match, mention_admin_cue (County/District/... in mention),
is_admin_class. Then extend gaz_info behind a flag and ablate:
- e5_features: all new features on top of best Wave-1 config
- e6_feature_ablation: drop-one if e5 wins
Targets finding 2 (admin cue x is_admin_class interaction) and finding 4.


## Wave 2b — cross-mention features (from the NGEC wiki_matcher review, 2026-08-19)
Document membership is recoverable from existing pickles by hashing doc_tensor
(verified). Measured on the pickles before building:
- e14_siblings: sib_adm1/sib_adm2/sib_country — candidate's parent-unit name
  appears as another mention in the same document. On name-matching candidates
  (the Denver problem): P(gold) 4.0% -> 46.6% given sib_adm1, -> 80.5% given
  sib_adm2; ~20x the coverage of admin1_parent_match. Targets finding 3. Cheap.
- e15_pooling: DONE (decode-time, 2026-08-19). Mean-raw-score pooling:
  +0.89 EM on TR+LGL+GWN, +0.28 overall, -1.0 on Prodigy (whose sentence-docs
  legitimately mix city/governorate golds); a confidence gate (skip pooling
  when own top-1 p>0.95) removes the regression. Verdict: worth shipping gated
  at inference, but the lever is nearly exhausted -- model is self-consistent
  on 95% of repeat groups; 10.8% of groups are unanimously wrong vs 4.6%
  inconsistent. Oracle ceiling +12 EM; pooling captures 2% of it. The value of
  document agreement must come through training (sib_* features, group loss),
  not decoding. Scripts: jobs tmp dir pooling/.
- e16_ap_twin: explicit "co-located same-name A/P twin in candidate set"
  feature (17.5% of entities have one; gold is in the pair 39.6% of those).
  Targets finding 2. Feature version first; candidate-merge version collides
  with e12_labels/metric comparability.
- e17_setnorm: generalize within-set normalization + is-argmax indicators to
  all 13 scalars + candidate-set size (train-time, create_gaz_features). Cheap
  stand-in for e8_listwise; also neutralizes B8 window dependence.
- e18_twopass: de-saturate adm1_count/country_count (weight sibling votes by
  1/|distinct adm1s|) and/or two-pass resolve-then-rescore. Medium.
Not adopted from NGEC: junk-candidate trimming (no recall problem), importance
boosts (their own notes: +0.4pt for 10x cost — the population-prior story).

- e19_geo (feature screen, 2026-08-19): sibling-geometry block — log km from
  candidate to sibling mentions' population-prior picks + normalized companions.
  log_min_km_anchor: residual accuracy 0.519 on hard-set entities where the
  population prior fails (floor 0.090; sib_adm1 0.251); max |r| 0.30 vs the
  existing 22 features; probe block delta +0.057, 5/5 seeds. Must ship with
  the set-shape gates (geo alone hurts the name-variant class). Arms:
  e19_geo (full block), e19b_geo_min (top 3), e20_shape (ambiguity-degree
  gates + is_seat_any alone).
- Screen negatives (do not build): wiki-index fame prior (doesn't beat
  population; circular on WikiDocs); es_rank (redundant with alt_name_length,
  window-dependent); discourse/surface blocks (corpus-identity recalibration,
  sign-flips across seeds); NO feature touches the A/P granularity class ->
  e12_labels promoted: granularity is a label-convention problem.
- B15 (new bug, feature screen): edit distances never case-fold; all-caps
  mentions (6.8% of entities) get min_dist==0 on only 22% of true name matches
  vs 99.9% normal-case. Fix in geoparse.py res_formatter (~:947) for serving;
  case-folded *_cf variants added via enrichment for training ablations.
- WARNING for all Wave 2 scoring: syn_cities is a geometry cheat sheet
  (template co-locates all mentions) and a population outlier -- never judge
  Wave 2 on synthetic held-outs; check per-source deltas.


## Wave 2 — feature blocks (DONE 2026-08-20)
WINNER e14_no_cf: --enriched --feature-blocks "prom,name,cue,sib,geo,shape"
(26 features) on the e11b recipe -> +0.0345 EM* / +0.0269 acc@161* vs e11b.
CAMPAIGN TOTAL vs e0: EM 0.8808 -> 0.9191 (+0.0382*), acc@161 0.9262 -> 0.9632
(+0.0370*). NEW COMPARISON BASELINE: experiments/e14_no_cf/seed*.json.
- Per-source: LGL +0.070*, TR +0.052*, Prodigy +0.054* (rescued -- its Wave-1b
  regression was missing evidence, not the recipe), all sources positive.
- sib,cue (6 feats) ~= geo,shape (13 feats) alone; both ~+0.022. prom,name
  nearly redundant in context (GWN -0.009*); cf dead weight (dropped).
- Synth flat while LGL +0.065 = gains are real disambiguation.
- NOTE: enriched runs peak ~40 GB RSS -> sequential only, until candidate
  dicts are pruned at load (queued).
- (Naming: Wave-2 dirs use e13_*/e14_* labels; the multitask idea formerly
  numbered e13 in this plan is renumbered e22_multitask.)

## Wave 3 — architecture (DONE 2026-08-20): DEAD END, model is
FEATURE-limited, not capacity-limited. Keep e14_no_cf; adopt --epochs 12
(+0.0013 EM n.s., +0.0006 acc@161*, 20% cheaper).
- e15arch_*: every capacity knob flat (country 24->128, code 8->32, mix
  256->1024, depth 3, residual: all inside noise; only shrinking hurts).
- e16arch_listwise: DESTRUCTIVE (-0.036 to -0.058 EM*, unstable across seeds,
  4-9x slower). Candidates-attending-to-candidates does not survive 500 slots
  of half-padding.
- e22_multitask (owner's idea): feature-class head does nothing; country head
  at 0.2 = best acc@161 of campaign (0.9655, +0.0024*) but trades Prodigy
  (-0.0195) for LGL (+0.0145*). Optional, only if acc@161 is the target.
- e17_nullrow: B2 fix is a proven bit-identical no-op at window 500 (gold
  never at the dropped index); --full-null-row flag exists for small windows.
- Infra: checkpoint config sidecar (mordecai_<date>.json); compact feature
  matrices -> runs 50 s, 10 GB steady, 4-wide concurrency.
- PARITY: DONE 2026-08-20. mordecai3/candidate_features.py is the single
  shared implementation; res_formatter/add_es_data_batch compute all 30
  features at inference; parity exact (0.0 diff, 43k rows, tests in
  tests/test_feature_parity.py); +14% end-to-end latency.
  Geoparser(feature_blocks=..., oov_bucket_fix=True, model_options={...}).
  e11_alignment CLOSED: window feature drift costs nothing (100-window ==
  500-window features at same candidates); keep max_choices=100, offer 500
  as accuracy flag (+0.33 EM, +51% latency, pure retrieval recall).
  Open: config sidecar for checkpoints; NULL-row truncation at full windows
  (B2) queued as a flag arm.
## Wave 4 — closing arms (RUNNING)
- e24_rstar: R** label rewrite (rewritten *_enriched_r2.pkl), decomposing
  learning effect vs metric effect; twin-credit standalone eval tool.
- Fresh error analysis of best model: DONE 2026-08-20. Error mass -27%;
  pooled EM 91.0-92.2 by seed. A/P bias eliminated (now seed variance);
  sib/geo features squeezed dry on the residual (5% "ignored signal");
  nothing underdetermined; 2 same-recipe seeds share only 70% of errors.
  Verdict: NEAR CEILING for this design. Oracle on current features +3pts,
  realistic +1-1.5. Final levers (both measured): (1) variance reduction --
  SWA/--avg-params/2-seed ensemble (ensemble measured 92.1 vs 91.7 expected
  single), +0.5-1.0; (2) stripped-name A/P twin features + *H historical-row
  penalty + duplicate collapse (13% of errors are name-variant twins), +0.5-1.0.
  Together ~0.93. Past that: replace frozen spaCy tensors (the last untouched
  component) or change annotation conventions -- not more same-family features,
  capacity, or data. Also: the unselectable reserved row wins the raw argmax on
  1.6% of entities and marks low-confidence answers (56-72% right vs 91-92%) --
  a free abstention signal for inference to expose.
- e25_final (QUEUED, after e24 releases train.py): SWA/ensemble arm +
  enrichment round 5 (ap_twin_stripped, is_historical) arm.

## Wave 3 (original outline) — architecture (after Wave 1 lands)
- e7_capacity: country_size/code_size bottlenecks (24/8 on 768-dim inputs are
  severe); bilinear or MLP scoring instead of cosine-after-projection.
- e8_listwise: cross-candidate attention over candidates (listwise ranking) so
  same-name candidates compete on evidence; targets finding 3.
- e9_hparams: lr/epochs/batch/mix_dim sweep on top of the winning config (lr
  matters more once logits are fixed).

- e13_multitask (owner's idea, 2026-08-19): auxiliary heads predicting the
  gold's country and feature class/code from the text pathway, on top of the
  Wave-1 fixes. The old `country_pred` head had the same double-softmax defect
  as the main head (B1), so its earlier null result is uninformative — retest.
  Variants: (a) country head only (fixed), (b) + feature-class head (directly
  supervises the A-vs-P distinction behind the dominant granularity errors and
  gives the 8-dim text_to_code pathway a real signal), (c) heads fed from text
  projections only (no gaz/ES features in the auxiliary path) to force the text
  encoder to carry country/type signal; sweep the loss mix (current 0.8/0.2).

## Wave 4 — bigger bets (only if Waves 1-3 plateau)
- e10_embeddings: replace frozen spaCy en_core_web_trf token tensors (requires
  pickle rebuild; decide encoder after seeing where errors remain).
- e11_alignment: retrain at inference window (100) or raise inference to 500;
  fix normalize() window dependence (B8) with a rebuild; fix B6/B7 counts.
- e12_labels: ANALYSIS DONE 2026-08-19 (scripts: jobs tmp e12_labels/).
  The A/P twin problem is ~all WikiDocs: twin-credit metric ceiling +3.8 EM
  there, +0.5 on TR+LGL+GWN (below seed noise). Model already picks P in 96.9%
  of twin cases; WikiDocs golds are consistently A-side (annotation convention,
  not noise). "55% granularity errors" decomposes: ~24% true A/P swaps + ~24%
  name-variant/hierarchy confusion (Moscow/Moskva, New Delhi/Delhi) that
  same-name rules don't reach. ADOPT: (a) twin-credit (nstrip_cc, cue-exempt)
  as secondary reported metric; (b) guarded label rewrite R** (A-side golds
  moved to P with 5 guards; 3.1% of WikiDocs train labels, 0.5% of human
  corpora) as a training arm AFTER Wave 2 reports, expecting movement on
  WikiDocs only. LGL's real budget remains wrong-admin1 disambiguation.
- e24_rstar: R** LABEL REWRITE **REJECTED** 2026-08-20 (experiments/e24_rstar/).
  The rewrite was built (tools/rewrite_labels.py -> *_enriched_r2.pkl, 482 train
  + 242 held-out labels moved, counts exactly as predicted) and trained on 5
  seeds via the new train.py --pickle-suffix (default "" is a verified no-op:
  the e14_no_cf rerun is bit-identical to the frozen jsons). It is a CONVENTION
  SWAP, not a correction:
  * On the 229 moved WikiDocs held-out labels, e14 is right 74.6% on the
    original key and e24 0.9%; e24 is right 93.3% on the rewritten key. The
    model learns whichever convention it is given.
  * Under twin credit the two arms are indistinguishable: learning effect
    +0.0021 +/- 0.0049 macro, -0.0001 +/- 0.0015 on WikiDocs. Every *metric
    effect* is exactly 0.0000 -- twin credit is provably invariant to moving a
    gold within its twin class, confirmed to the last digit.
  * On the frozen key: macro EM +0.0034 +/- 0.0083 (ns), acc@161
    -0.0012 +/- 0.0014 (ns), WikiDocs -0.0232*. The naive `_last5` comparison
    (+0.0069 EM*) is the answer key moving, not the model improving.
  WHY THE PREMISE EXPIRED: e12 measured the pre-Wave-2 checkpoint, which picked
  P in 96.9% of twin cases. e14_no_cf's cue/shape/prom blocks let it LEARN the
  WikiDocs A-side convention (74.6% right on exactly those labels), so there is
  no label tax left to refund. Wave 2 absorbed A/P granularity as a modelling
  problem.
  ADOPTED from e12: twin-credit as a secondary metric --
  tools/twin_credit_eval.py (standalone; error_utils.py untouched; reproduces
  the frozen per-source exact_match to 1e-12). On e14_no_cf seed101 it reads
  macro 91.90% strict -> 92.96% twin-credit (+1.06); TR+LGL+GWN pooled +0.60;
  LGL only +0.11, i.e. almost none of LGL's residual error is A/P granularity.
  KEPT: --pickle-suffix and the _r2 pickles/compact caches on disk, for cheap
  re-testing if a future recipe changes the premise again.

## Non-goals (evidence-based)
- More Wikipedia data (WIKI_TRAINING_DATA.md).
- ES recall work (finding 1).
- Population as a silver bullet (finding 4).

## Bookkeeping
experiments/eN_name/{seedS.json, seedS.log, NOTES.md}. NOTES.md: exact commands,
paired-delta table vs e0, verdict.

## FINAL (2026-08-20): campaign closed
SHIP RECIPE (e29_swa_ep15): --epochs 15 --mix-dim 512 --logits --mask-padding
--oov-bucket-fix --modern-mlp --label-smoothing 0.05 --enriched
--feature-blocks "prom,name,cue,sib,geo,shape" --avg-params --avg-mode swa
--checkpoint-out <path>, uncapped six-source mix.
RESULT vs e0: EM 0.8808 -> 0.9258 (+0.0450 +/- 0.0051*), acc@161 0.9262 ->
0.9661 (+0.0399 +/- 0.0024*); every source improved (LGL +0.098*, TR +0.062*,
WikiDocs +0.060*); seed spread halved. Twin-credit (secondary): 93.8%.
Attribution: ~3/4 enriched features (Wave 2), rest loss/recipe (Wave 1) +
SWA (Wave 4). Architecture: nothing.
Deployment option: 3-seed probability-averaged ensemble ~93.1% strict /
94.3% twin-credit (tools/ensemble_eval.py); SWA and ensembling are
substitutes -- ensemble plain-SWA seeds.
Final-wave rejections: strip block (neg. on acc@161* and twin-credit);
--train-after-eval (dropout was inert since epoch 2 all campaign -- fixing it
hurts); --lr-schedule (cosine was never applied -- flat 1e-3 stays).
Full report: ACCURACY_CAMPAIGN.md. 60 experiment dirs with NOTES.md.

## A second campaign: where the next points would come from

Everything below is informed by the closing error analysis (residual errors
have indistinguishable feature profiles; 2 seeds share only 70% of errors;
16% of errors have no document evidence at all). Ordered by expected size.

1. REPLACE THE FROZEN ENCODER (the big bet, +1-2pts?). The spaCy
   en_core_web_trf token tensors are the one untouched component, and the
   residual is exactly where better text understanding should bite: same-name
   candidates whose gazetteer features tie. Options, cheapest first:
   (a) swap in a modern compact encoder's mention-in-context embeddings
   (rebuild nlp_docs once, same pickles pipeline); (b) fine-tune a small
   encoder end-to-end with the ranker (the loss is finally fittable, so this
   is newly plausible); (c) distill (b) back into frozen features for cheap
   serving. Judge per source: the encoder should move LGL/TR residuals, not
   WikiDocs. Note training currently takes ~50s/run BECAUSE tensors are
   precomputed -- (b) changes the economics of every future experiment; build
   the frozen-feature comparison harness first.
2. VARIANCE, PROPERLY (+0.5 measured). 3-seed SWA ensemble is already
   measured at ~93.1/94.3 -- productionize tools/ensemble_eval.py into the
   serving path if the deployment can afford it. Beyond that: snapshot
   ensembles within one run (cheaper than 3 trainings), and check whether
   distillation of the ensemble into one model keeps the gain.
3. THE NO-EVIDENCE RESIDUAL (16% of errors). Same-name US county/city cases
   with no anchors and no siblings cannot be fixed by document features.
   Candidates: (a) corpus-level priors (which Denver do news articles
   usually mean -- a learned per-name prior from the training corpora or an
   external resource like Wikipedia link frequencies, used as a FEATURE not
   a rule); (b) publication metadata (LGL is local news -- outlet location is
   the missing context and exists in the raw XML; check what leaks into the
   other corpora before using).
4. GAZETTEER HYGIENE (+0.3-0.5, mechanical). Collapse duplicate rows
   (Mauna Kea twice), demote *H historical rows at serving unless the doc is
   historical, alias table for the D.C. cluster. ~20 strings = quarter of
   residual mass. Do it in the ES index build, not the model.
5. ABSTENTION AND CALIBRATION (product win, not EM). The reserved-row argmax
   signal (1.6% of entities, 56-72% accuracy) + temperature calibration of
   the softmax -> a usable confidence score; researchers filtering to
   high-confidence geolocations is the actual downstream use case.
6. END-TO-END EVALUATION. The campaign measured resolution GIVEN a detected
   mention. spaCy NER misses are invisible here. Measure end-to-end
   (detection + resolution) on TR/LGL/GWN; if NER recall is the bottleneck,
   a next campaign lives there, not in the ranker.
7. NEVER-RUN LEFTOVERS, cheap to try: two-pass resolve-then-rescore (e18);
   de-saturated adm1/country counts (weight sibling votes by ambiguity);
   fuzzy-retry retrieval for the 2-3% outright misses on TR/LGL/GWN (bounded
   by finding 1, so strictly last).
8. EVALUATION DEBT: decide weighted vs unweighted source aggregation before
   any publication; report twin-credit alongside strict; Prodigy's
   sentence-level "documents" fight document-evidence features by
   construction -- consider re-annotating it at document level or accepting
   its ceiling.

## CAMPAIGN 2 (opened 2026-08-20): five-track review + first arms

Full synthesis: experiments/campaign2/SYNTHESIS.md (reports for each track in
the same directory). Supersedes the "second campaign" sketch above where they
disagree. Headlines: campaign-1 result survives audit (selection bias only
+0.002-0.004) but the Wikipedia reversal is overstated (helps WikiDocs, hurts
TR, macro n.s.); 81% of held-out (mention,gold) pairs occur in training
(novel-pair EM 0.771); end-to-end from raw text is 58.6% vs 91.2% conditioned,
with NER-side pipeline discards (nested entities, NORP) owning 74-83% of the
gap and measured config fixes reaching 72.9%; naive encoder swap LOSES
(mention slot needs spaCy's OntoNotes feature-class signal); outlet metadata
covers 79% of LGL's no-evidence errors; abstention is a serving-time win
(p_pred_full T=0.874, p>=0.7 -> 90.6% coverage @ 95.2% selective EM).

New metrics: TLG-hard (TR/LGL/GWN macro EM, non-country golds; e29 baseline
0.8730), twin-credit secondary, novel-pair EM guardrail. Significance: t(4)
> 2.776, not |mean|>2SE.

New ledger entries (both 5-seed, paired vs e29_swa_ep15):
- e30_absw100 (--window 100, out-of-window golds labelled abstain): REJECTED.
  Unanswerable AUROC 0.858->0.931* but answerable EM -0.0035*, TLG-hard
  -0.0092*; selective EM @90% and AURC n.s. -- relocates error.
- e31_abstain_w3 (--abstain-weight 3): REJECTED, dominated by e30.
train.py gained --window and --abstain-weight (defaults byte-identical to
e29, verified vs frozen seed42.json).

Ladder: Phase 0 scoreboard repair (reserved-row unification, doc-id splits,
metric adoption, dev/test carve-out, ship-artifact promotion) -> Phase 1
serving wins (implemented; see campaign2/serving_fixes_report.md) -> Phase 2
cheap features (outlet-location, state-abbrev aliases for sib_adm1, gazetteer
hygiene) -> Phase 3 representation (e40 harness fix, e42 context-slot swap,
e41 dual-mention width, e43 typed encoder, e45 locs_tensor) -> Phase 4 NER
campaign (nested place-name tagger; biggest lever, ~3 ranker campaigns).
Decisions D1-D5 RESOLVED 2026-08-20 (see SYNTHESIS.md): TLG-hard primary +
dev/test freeze (D1); demonyms removed from task — NORP golds out of e2e
denominators, accept_norp deleted from library (D2); ship e29 seed101 single
checkpoint into assets/ (D3); Synth out of headline, stays in training (D4);
NER tagger campaign is the flagship (D5).

D5 EXTENSION (owner's idea, 2026-08-20, scoping in flight): beyond the
tagger-head-on-frozen-trf plan, a DEEPER retrain of the NER model itself on
OntoNotes(-derived) + other NER data — a place-specialized NER, possibly
replacing the trf encoder wholesale (which then requires re-embed + ranker
retrain; economics fine per encoder report, but the mention slot's
feature-class signal must survive or improve). Must handle nested spans
(52% of NER misses), bake in the D2 no-demonyms rule, and respect latency
(second encoder pass measured +60% on the NLP stage). Scoping + pilot
report expected at experiments/campaign2/ner_retrain_scoping_report.md,
comparing tagger-head vs full-retrain arms on identical data under the D2
denominator.

### Phase 0 CLOSED (2026-08-20): scoreboard repair, D1-D4 implemented

Report: experiments/campaign2/phase0_report.md. Ledger: experiments/e60_docsplit
(5 seeds). Code: mordecai3/geoparse.py, mordecai3/assets/, tools/train.py,
tools/error_utils.py, tools/calibration_eval.py, tools/end_to_end_eval.py,
tools/twin_credit_eval.py, tests/.

D2 (demonyms out of the task). accept_norp and its code path deleted from the
library; NORP can no longer be emitted by any configuration. e2e denominators
now exclude demonym gold rows = spaCy-NORP-only spans (215) UNION GWN
Non_Literal_Modifier (93 more) = 308 of 2,392, leaving 2,084; every summary
also carries legacy_incl_demonym. Restated on 260 held-out docs, e29 seed42,
max_choices=100 (pre-D2 -> D2):
  ship    58.61 -> 64.87 e2e EM, det R 68.15 -> 75.67, det P 75.47 -> 73.01
  serving 60.58 -> 66.99 e2e EM, det R 70.57 -> 78.22, det P 78.19 -> 75.50
  oracle spans 80.14 -> 83.45; campaign-parity 90.96 (n=1,560)
Gap to the oracle ceiling 19.6 -> 16.5 points. Matches
ner_retrain_scoping_report.md §2 cell for cell (independent implementations).
accept_norp under D2 is +0.14 EM for -10.9 det precision, i.e. strictly bad.
DEFERRED: data_formatter_wiki_docs still uses NORP in locs_tensor (<=0.1 EM, e45).

D3 (ship artifact). experiments/e29_swa_ep15/seed101.pt installed as
mordecai3/assets/mordecai_2026-08-20_seed101.pt + sidecar; geoparse.py defaults
to it and now reads feature_blocks/oov_bucket_fix/logits/mask_padding/
modern_mlp (and mix_depth/residual/listwise/aux_*) FROM THE SIDECAR. Verified:
calibration_eval reproduces seed101.json per source to 6 dp, macro 0.929968.
Cost: 5 curated tests now xfail (not rewritten) -- test_governorates (A/P
granularity), test_uk_oxford2 (Oxford UK -> Oxford MS), test_geneva_il
("Geneva, Illinois" -> Geneve CHE), and the two Hague tests (the ship model
abstains on the string "The Hague" at p_no_match 0.93-0.97 while resolving the
trimmed "Hague" correctly -- the Phase 1 trim guard is what exposes it).
Two of the three resolution failures are genuine US-bias errors: Phase 2 leads.
Root mordecai_2026-08-20.pt left untouched, unused.

Reserved-row unification (S1). One convention, documented in geoparse.py and
implemented via geoparse.candidate_row_count in geoparse.py, error_utils.py,
calibration_eval.py and end_to_end_eval.py: row W-1 is reserved and never a
candidate; candidates are 0..min(n,W)-1 minus W-1 on a full list; p(reserved)
decides whether to answer; abstention is reported as a third outcome with the
three accuracies (conditioned / all-mention / abstain rate). Serving output is
unchanged (proved and re-measured: harness still reads 58.61/60.58). Eval-time
delta on e29, macro: +0.0004 (w500) / +0.0013 (w100) from no longer crediting
the sentinel's score to the candidate it overwrote, -0.0033 / -0.0026 from
charging abstentions. evaluate_results and the frozen exact_match are untouched.
BYTE-IDENTITY: a rerun of the e29 recipe seed 42 reproduces seed42.pt
(md5 22785b60360b7c21edc1ca8b7261167e) AND seed42.json
(md5 1d84e9529c77db0ef5b9c639d2dff96f) -- verified before and after the changes.

Checkpoint guard. --checkpoint-out refuses to overwrite an existing file or its
sidecar unless --overwrite-checkpoint is passed; checked before data loads.

Metric adoption (D1/D4). Every training run now prints and writes
<metrics-out stem>.metrics2.json with TLG-hard (primary), novel-pair EM
(guardrail), twin-credit, macro-5 (headline, Synth dropped) and macro-6
(legacy), plus per-source. calibration_eval prints the same scoreboard with a
legacy/unified column pair; the two implementations agree exactly on seed 42.
Twin classes and training (mention, gold id) pairs are cached once by
`tools/twin_credit_eval.py --cache-out experiments/campaign2/twin_gold.json`.
5-seed e29 baselines, legacy convention (these reproduce the frozen numbers):
TLG-hard 0.8730 +- 0.0057, macro-6 0.9258 +- 0.0042, macro-5 0.9155,
twin-credit 0.9289, novel-pair 0.7810 (n=1,747; the quoted 0.7714/n=1,702 is a
pair-key definition difference). Unified convention: TLG-hard 0.8676 +- 0.0068,
macro-6 0.9214, novel-pair 0.7731.

e60_docsplit: e29 recipe under --split-mode doc (documents assigned by a hash
of doc_tensor), 5 seeds. UNPAIRED, different held-out sets, not comparable:
macro-6 0.9222 +- 0.0037 (vs 0.9258), TLG-hard 0.8525 (vs 0.8676), novel-pair
EM 0.8033 (vs 0.7731), unanswerable 2.20% (vs 1.55%). The recipe is not an
artifact of the split, and re-keying does NOT reduce answer-key memorisation --
it is lexical, not documentary. Default split unchanged; the six current
held-out sets are frozen as DEV. TEST protocol proposed in phase0_report.md §7.

### Phase 2 arm: outlet location (e50 -> e53 -> e54, ADOPTED and staged)

Reports: campaign2/outlet_feature_report.md (e50 build + leak audit; e53
follow-up sections 10-12), campaign2/outlet_integration_report.md (e54 merge).
Ledgers: experiments/e50_outlet, experiments/e53_outlet_dropout,
experiments/e54_outlet_ship. Code: mordecai3/outlet_features.py,
mordecai3/torch_model.py (block `outlet` appended last + TrainData.
set_outlet_dropout), mordecai3/geoparse.py (outlet= / outlets=),
tools/enrich_pickles.py --outlet-only, tools/train.py --outlet-dropout,
tools/end_to_end_eval.py --feature-blocks/--outlet-sources, tools/run_e54.sh,
tools/e54_aggregate.py, tools/outlet_*.py, tests/test_outlet_features.py.

The idea: a local paper writes about its own patch, and 80.8% of LGL's linked
toponyms sit in their outlet's modal admin1. Five columns per candidate
(has_outlet_home, log_km_to_outlet_home, outlet_same_adm1, outlet_same_country,
has_outlet_country), keyed on the article's <domain> through a 120-row
domain -> newsroom table geocoded against the same GeoNames index the ranker
retrieves from. Leak protocol: the table was written from domain strings plus
outside knowledge only -- no article text, no gaztag, no geonameid, no
per-domain gold statistic. tools/outlet_leak_audit.py, five mechanical checks,
all pass; decisively, the article<->entity join gives the identical domain for
all 3,245 LGL entities with and without the gold id as join key.

e50 (block, curated table): TLG-hard +0.0300 (t 7.61), LGL non-country +0.0701,
LGL novel-pair +0.1111. Home-permutation control (both mask channels
bit-identical, only the three evidence columns changed) reproduces NONE of the
gain and significantly HURTS LGL (-0.0173), so 100% of it is the true
article<->newsroom correspondence and not the corpus indicator the block also
unavoidably is. BLOCKER: with the outlet withheld the checkpoint scored -0.0283
on LGL, i.e. worse than not having the feature -- a realistic serving condition.

e53 (two pre-ship conditions, both discharged). (1) The home table was rebuilt
from public sources only (Wikipedia, LoC newspaper catalogue, archive.org
masthead pages, FCC records), URL per row, in tools/data/outlet_homes_
researched.tsv; 101 of 120 domains agree with the curated table and re-scoring
a trained checkpoint on the other table costs 0.0007 LGL EM. (2) --outlet-
dropout p blanks the block for a random subset of whole documents each epoch,
deterministic in (doc_hash, epoch, seed) via a private RNG. p=0.5 ADOPTED:
withheld regression back to -0.0051 (t -1.41, n.s.) at no measurable cost with
the outlet present. p=0.3 fails (b) at t -2.96.

e54 (mainline merge + ship candidate), 5 seeds paired vs e29:
  IDENTITY GATE, run 3x and across all 5 seeds: with the block off, the e29
  recipe reproduces experiments/e29_swa_ep15/seed*.pt AND seed*.json md5-
  identically on the re-enriched pickles (seed42 22785b60... / 1d84e952...).
  raw_data/pickled_es/*_enriched_compact.pkl now carry 47 columns, not 42.
  TLG-hard 0.8676 -> 0.9043, +0.0368 +- 0.0070 (t 11.80*)
  LGL non-country 0.8739 -> 0.9452 (+0.0714, t 19.79*); LGL novel-pair +0.1097
  novel-pair EM all sources 0.7731 -> 0.8040 (+0.0309, t 9.56*)
  twin-credit macro 0.9241 -> 0.9405; macro-5 0.9109 -> 0.9271 (t 3.83*)
  TR non-country +0.0333 (t 5.72*) -- SIGNIFICANT FOR THE FIRST TIME, because
    the researched table leaves 0 of TR's 914 entities homeless vs 340 curated.
    Paired against e53's d50, training on the researched table is worth
    +0.0087 TLG-hard (t 6.54*), all of it TR; LGL flat (-0.0005).
  Guardrails (Prodigy/GWN/Synth/WikiDocs, none of which can have an outlet):
    all n.s.; WikiDocs +0.0006 (t 0.69) against the "reading the mask" tripwire.
  Outlet WITHHELD on LGL: -0.0047 +- 0.0018, t -2.65, n.s. -- clears the
    criterion but with less margin than e53 (same point estimate, half the
    spread). One p=0.7 probe would settle flat-vs-merely-n.s.; not blocking.
  END TO END, seed42, serving variant, D2 denominator (2,084 golds, 260 docs):
    e29 reference 66.99 | e54 no outlet supplied 67.95 | e54 LGL outlets 70.01
    | e54 LGL+TR outlets 70.40. Detection identical in all four rows (R 78.22,
    P 75.50), so every point is resolution; emitted-location precision
    77.08 -> 82.32; oracle-span 83.45 -> 87.41. The NER gap does not close
    (16.5 -> 17.0), which is what a ranker win should look like.

SHIP CANDIDATE: seed 42 (leads on novel-pair, twin-credit, macro-5, macro-6 and
the withheld condition; 2nd on TLG-hard by 0.0029). STAGED, NOT DEFAULT:
mordecai3/assets/mordecai_2026-08-20_e54_seed42.pt(.json) and
assets/outlet_homes.json are packaged, and the sidecar turns the block on by
itself, so promotion is ONE line -- DEFAULT_MODEL_ASSET in mordecai3/geoparse.py.
Owner's call. A promotion should re-attribute the five Phase-0 xfails, which
were measured against e29 seed101.

p=0.7 PROBE (experiments/e54_outlet_ship/p07, 5 seeds, same setup): RUN, and
p=0.5 STANDS. Withheld-LGL -0.0057 +- 0.0071 (t -1.79) vs d50's -0.0047 --
marginally worse, statistically identical (paired d70-d50 -0.0011, t -0.53),
still 4/5 seeds negative. Present-side TLG-hard 0.9026 vs 0.9043, paired -0.0017
(t -0.61, n.s.), but every aggregate cell worse-or-equal and the only
significant movement anywhere is a Synth regression (-0.0067, t -3.16). The
decision rule needed withheld ~0 AND no present cost; the first clause fails.
e53's monotonicity reading (d30 -0.0068 -> d50 -0.0051 -> 0) was WRONG: the
residual is a FLOOR of ~5 entities of 946, not an under-trained fallback
policy. Lead if anyone wants it gone: error-analyse those five LGL mentions,
not another value of p. Not urgent -- end to end the no-outlet condition is
+0.96 EM OVER e29, not a regression.

Remaining, none blocking: a production domain->city table from a newspaper
directory instead of these 120 corpus-specific rows; the softer
outlet_same_adm1 variant (distance percentile) suggested in e50 section 9.

### Phase 2 arm: the place-span head in the serving path (e56, N1 integration)

Report: campaign2/span_head_serving_report.md. Ledger: experiments/e56_span_head_serving.
Code: mordecai3/span_head.py (+ two 1.6 MB assets and pyproject package data),
mordecai3/geoparse.py (span_detector= / span_threshold=), tests/test_span_head.py,
tools/end_to_end_eval.py (head_gold / head_all variants, per-corpus spaCy reuse,
gold-by-gold oracle gap counters).

e55 staged the head; this arm ports it and measures it WITH the outlet block --
the combined grid nobody had. span_detector=None|"gold"|"all"|<path>; when set
it REPLACES the GEO_LABELS filter, trim_span_tokens and nested_gazetteer_spans
in one call, and everything downstream (retrieval, features, ranker,
reserved-row decode, p_no_match, outlet plumbing) is untouched.

GATES. (1) BYTE IDENTITY with span_detector=None: the whole geoparse of the 260
held-out documents md5s to f2656f881c0e85632ad9ec8235a0755d under the current
package, under a rerun of it, and under a copy with the span-head hunks
reverted. (2) The harness `serving` row on e29 seed42 reproduces phase0 §1 cell
for cell (66.99 / 69.19 / 78.2 / 75.50 / 77.08 / 83.45); e54 rows reproduce
outlet_integration §7 (67.95 / 70.39). (3) Both packaged heads reproduce e55's
detection row EXACTLY through the library module: gold 85.83/89.49/87.62,
nested 67.9, demonym FP 81, 2,173 preds; all 83.95/88.87/86.34, nested 76.5,
demonym FP 56, 2,206 preds. (4) Suite 94 passed / 2 known failures / 5 xfails.
(5) Unplanned: gold-head e2e on e29 seed42 is 77.26, exactly one of e55's three
head-seed values, from independent code.

THE GRID (D2, 2,084 golds, 260 docs, max_choices=100), e2e EM:
                        none    gold head   all head
  e29 seed42 (ref)      66.99     77.26       76.68
  e29 seed101 (default) 67.66     77.69       77.02
  e54 seed42, no outlet 67.95     78.55       77.59
  e54 seed42, LGL+TR    70.39     80.85       80.13
Detection (ranker-invariant): none 75.50/78.21/76.83 F1, nested R 13.8, demFP 59
| gold 85.83/89.49/87.62, nested 67.9, demFP 81 | all 83.95/88.87/86.34,
nested 76.5, demFP 56. Emitted-loc precision in the best cell 82.32 -> 83.71.

COMPOSITION: the head is worth +10.0 to +10.6 EM in EVERY ranker/outlet
condition and the outlet +2.3 to +2.4 in every span condition; full stack
+13.86 against an additive prediction of +13.67. They do not overlap.

GOLD vs C_ALL: gold wins end to end in all four ranker rows (+0.58/+0.67/+0.96/
+0.72). C_all's +8.6 nested recall is ~35 nested golds; its -2.9 flat recall
costs ~49 flat golds, and flat golds resolve better once found. C_all's case is
the metric EM cannot see: 56 demonym FPs vs gold's 81 and the label filter's 59,
plus e55's nested e2e 61.48 vs 54.90.

LATENCY (50 LGL docs, geoparse_batch wall clock, median of 3): default 44.78
ms/doc GPU / 195.25 CPU; nested_gazetteer_pass 60.22 / 209.36; gold head 37.57 /
184.70; all head 39.90 / 191.28. NOT latency-neutral -- FASTER. The head costs
4.9 ms/doc and saves more in Elasticsearch (5.3 s -> 3.0 s over 175 LGL docs):
431 of the label filter's 2,159 spans sit on no gold toponym and buy a failed
search plus a fuzzy retry.

ORACLE ATTRIBUTION, best cell: detection loss 21.79% -> 10.51% of golds,
resolution loss 7.82% -> 8.64% (retrieval_miss 88 -> 112: nested toponyms
retrieve worst). Gold by gold, the 130-toponym gap to the 87.41 ceiling is
97.4% DETECTION -- exactly 4 of 2,073 golds are ones the ranker resolves
correctly from the gold span and wrongly from the head's, and 22 go the other
way. No hidden resolution debt; the next lever is retrieval, not ranking.

ABSTENTION (not a recalibration): abstain rate 14.17% -> 5.34% (e29 seed101) and
17.46% -> 7.36% (e54) at an unchanged mention count; mean p_no_match 0.115 ->
0.042 / 0.157 -> 0.065; ninth decile 0.599 -> 0.057. p_no_match was silently
doing detection's job. Anyone thresholding it must re-pick the threshold.

BEHAVIOUR CHANGES for callers: overlapping spans are emitted (24 of 2,013
emitted locations in the best cell, 38 of 1,999 for C_all -- structurally zero
before), and demonyms are suppressed by training rather than by the label
filter. Hallucinated locations (no gold under them) hold at 2.5% of output for
gold, 3.6% for C_all.

RECOMMENDED DEFAULT, STAGED NOT APPLIED -- two independent one-line flips:
  DEFAULT_MODEL_ASSET = "assets/mordecai_2026-08-20_e54_seed42.pt"  (e54, staged)
  span_detector default None -> "gold"                              (e56, staged)
Both: 80.85 e2e EM with outlets, 78.55 without, from 67.66 today. Flip 2 alone
on today's checkpoint: 77.69. Do NOT enable nested_gazetteer_pass; it is what
the head replaces and it is ignored when span_detector is set.

BLOCKER on flip 2, unchanged from e55: the N1 gate. Every number here is on
held-out DOCUMENTS of the corpora the head trained on; reproduce detection and
e2e on D1's untouched modern-news TEST corpus before the default moves. Also
inherited: one head seed (e55's three gold seeds span 74.81 / 77.02 / 77.26 and
seed 42 is the top one), so read 80.85 with a +-1.4 head-seed spread.

e56 follow-ups (same ledger, report sections 11-12).

LEAVE-ONE-CORPUS-OUT (the N1 gate, made cheap). experiments/e56_span_head_
serving/loco.py: train the head on two of {TR, LGL, GWN}, score detection on the
third's held-out docs, 3 seeds, threshold picked on the TRAINING families' dev
split. Bar = the spaCy label-filter path on that corpus, not the in-family 87.6.
  gold  TR  83.21 +- 0.58  vs spaCy 77.35  (+5.86, t 17.65*)  in-family 83.87
  gold  LGL 84.52 +- 1.20  vs spaCy 75.24  (+9.28, t 13.42*)  in-family 90.31
  gold  GWN 80.81 +- 1.08  vs spaCy 80.30  (+0.51, t 0.82 ns) in-family 83.71
  dem10 TR/LGL/GWN 83.07 / 81.11 / 81.15 -- dem10 is not a better generalisation
  recipe, it costs 3.4 F1 out-of-family on LGL and buys 0.34 on GWN.
NO FOLD FALLS BELOW THE SPACY PATH; GWN ties it. End to end on that worst fold
(e54 seed42, GWN, 523 golds): out-of-family 75.02 +- 1.81 (74.38/77.06/73.61)
vs spaCy 74.19 vs in-family head 79.35 -- +0.83, t 0.79 ns, one seed below. So
~5 of the head's in-family +10.5 EM is corpus-specific span convention. Nested
detection recall transfers everywhere (43.9 vs 23.2 even on the tying fold);
demonym FPs on GWN are the same out-of-family (77.7) as in-family (77) and are
essentially the ONLY demonym FPs anywhere (TR 0, LGL 4) -- that column is a
GeoWebNews annotation-policy column, not a generalisation one. GATE: AMBER,
not closed. tools/end_to_end_eval.py gained --span-head-paths for this.

CALIBRATION REFIT on the SERVING frame (experiments/e56_span_head_serving/
calibration_refit.py; metric code imported from tools/calibration_eval.py). Not
the campaign's oracle-span frame: the detector picks the mentions and 19-29% are
unanswerable. e54 seed42 + LGL/TR outlets, both detectors on the same frame.
                        none          gold head
  mentions/answerable   2159/1541     2173/1752
  refit T               1.034         1.106
  ECE @0.874 -> refit   0.1037->0.0689  0.0941->0.0430
  AURC                  0.1209        0.1001
  selective EM @90% cov 75.50         85.63
  AUROC unanswerable    0.8945        0.7967
THE SERVING-FRAME TEMPERATURE IS ABOVE 1, not 0.874: that value sharpens a model
that is under-confident only on gold spans, and on real spans it is over-
confident. Refitting halves ECE under the head. NEW RECOMMENDED THRESHOLD under
span_detector="gold": p >= 0.5 or the reserved-argmax flag -- 87.99% coverage at
86.51% selective EM (the old p >= 0.7 now costs 81.41% coverage at 88.36%).
ABSTENTION QUALITY DEGRADED on the ranking statistic: -0.074 AUROC for flagging
a wrong answer, -0.098 for detecting an unanswerable mention, because the easy
junk-span negatives are gone and the 421 that remain are hard. Absolute
selective performance is better at EVERY coverage level (AURC and the risk-
coverage rows), so a user filtering by p is better off -- but any deployment
thresholding p_no_match MUST re-pick its threshold at the flip, and p_no_match
is no longer a good standalone unanswerability detector.

RECOMMENDATION AFTER THE FOLLOW-UPS: unchanged in direction, sharper in caveat.
Flip 2 (span_detector -> "gold") should carry temperature=1.10 and a documented
p >= 0.5 filter; temperature stays a constructor argument and 0.874 remains
correct for the oracle-span frame. The blocker is still the TEST corpus, and
the honest out-of-family expectation is -0.6 to +2.9 EM, not +10.5.

### Phase 2 arm: R1 abbreviation normalisation in the mainline (e57)

Report: campaign2/r1_retrieval_report.md. Ledger: experiments/e57_r1_retrieval.
Code: mordecai3/place_aliases.py (new), mordecai3/geonames.py (5 lines in
build_name_search + the flag), mordecai3/geoparse.py (Geoparser passthrough),
tests/test_place_aliases.py (52 cases), tools/end_to_end_eval.py
(--no-normalize-place-abbrevs, per-gold gold_outcomes, retrieval_examples).

e52 recommended exactly one of its four hygiene rules; e56 ended by naming
retrieval as the next lever (retrieval_miss went UP 88 -> 112 under the span
head). This arm ports R1 and measures it on top of the whole new stack.

ADOPTED, DEFAULT ON. Unlike e56's two flips this one is applied: it has no open
gate. GeonamesService(normalize_place_abbrevs=True) by default;
Geoparser(normalize_place_abbrevs=None|True|False) where None = "use the
service's own setting". Off reproduces the pre-e57 query byte for byte.

GATES. (1) BYTE IDENTITY with the flag off: the whole geoparse of the 260
held-out documents md5s to f2656f881c0e85632ad9ec8235a0755d -- e56's published
hash -- and to 658c603922190f5a46b8fb911d29f523 with it on, so the rule is
firing. (2) Every published e2e row reproduces with the flag off, cell for cell:
66.99 / 67.66 / 77.69 / 78.55 / 70.39 / 80.85, same retrieval_miss (88/112),
same detection (76.83/87.62 F1), same oracle rows (83.45/84.37/87.41); grid run
twice, aggregate identical. (3) e52's frozen-ranker R1 row reproduces DIGIT FOR
DIGIT through the canonicalised table (frozen_gate.py monkeypatches
mordecai3.place_aliases into e52's own harness after 1,159-probe table parity):
TLG-hard em_all 0.8244 -> 0.8444, em_cond 0.8551 -> 0.8588, macro-6 em_cond
0.9214 -> 0.9224, +28/-0 entities, 65 firings in 8,977, WikiDocs/Prodigy/Synth
digit-identical. (4) Suite 146 passed / 2 known failures / 5 xfails.

THE COMPOSITION GRID (D2, 2,084 golds, 260 docs, max_choices=100), e2e EM:
                                  R1 off   R1 on    delta   retr_miss
  e29 seed42, spaCy spans          66.99   68.19    +1.20    88 -> 62
  e29 seed101, spaCy spans         67.66   68.76    +1.10    88 -> 62
  e29 seed101, gold head           77.69   79.08    +1.39   112 -> 84
  e54 seed42, gold head, no outlet 78.55   79.80    +1.25   112 -> 84
  e54 seed42, +LGL/TR, spaCy       70.39   71.64    +1.25    88 -> 62
  e54 seed42, +LGL/TR, gold head   80.85   82.29    +1.44   112 -> 84
Detection is BIT-IDENTICAL in every row (the rule changes a query string, not a
span), so all of this is resolution. Per corpus in the best cell: TR 72.81 ->
73.41, LGL 83.66 -> 85.45, GWN 79.35 -> 80.50.

R1 IS WORTH MORE UNDER THE HEAD, NOT LESS (+1.25/+1.39/+1.44 vs +1.10/+1.20/
+1.25 on spaCy spans): the head finds more of the abbreviation golds, so the
same rule repairs 28 instead of 26. Full stack vs Phase-0 66.99 is now +15.30
against an additive prediction of +14.87.

THE ORACLE-SPAN CEILING MOVES TOO: 87.41 -> 89.05 in the best cell (83.45 ->
84.80 and 84.37 -> 85.82 in the others), oracle retrieval recall 92.91 -> 94.36.
Anything quoting 87.41 as "the resolution ceiling" needs updating.

GOLD BY GOLD (the harness now records every gold's decomposition bucket):
best cell +30 / -0 (28 retrieval_miss->correct, 2 null_answer->correct); other
cells +26..+32 against 1-3 lost. Every loss in the whole grid is the SAME
mechanism and all three are named: a repaired abbreviation is a stronger
document anchor than the garbage it replaces, so a neighbour moves -- Ky. fixed
in TR doc 96 pulls Paris/France to a US Paris, N.J. in doc 97 the same, La. in
LGL doc 458 pulls Richmond/Kentucky. That is e52's BELGRADE knock-on gain
running the other way. e52's +28/-0 holds exactly in the best cell.

RESIDUAL RETRIEVAL MISS, best cell, 84 left of 112, classified live vs ES:
  past_window    51 (61%)  the gold IS returned, past rank 100 -- Richmond x8
                           (rank 100-228), Hanover x7, Logan x5, Paris at 228
  nested_span    15 (18%)  a toponym inside an ORG/FAC name; 14 of them
                           ADJECTIVAL (British x6, European, Canadian, Turkish)
  name_mismatch  14 (17%)  North Africa -> Northern Africa RGN x4, Platte Co.
  gold_row_gone   4        stale labels (Red Sea, Black Sea, Hillsboro)
  abbrev_other    0        R1 covers the whole abbreviation class here
Of the 112 before R1, 28 were the abbreviation class and R1 takes all of it.
THE BIGGEST LEVER LEFT IS THE alt_name_length SORT (e52 section 5(ii)3): 61% of
the residual is a gold the query already returns past the window. Index rebuild
+ retrain, not a serving flag. SECOND is a demonym/adjectival alias table on the
same query hook -- structurally identical to R1, one class over, and it is the
class the span head created.

"LA" RISK, PRICED (la_risk.py, three ways). (a) Every query the serving path
actually sends, recorded by wrapping build_name_search through a real run: 31
firings on the spaCy path, 33 under the head; bare-code firings are SC, WA, NC,
FL and nothing else. NO "LA" IS QUERIED AT ALL, nor IN/OR/OK/ME/DE/MD/ID/PA.
(b) Three of 3,992 held-out gold rows are bare codes: KY -> Kentucky OK, SC ->
South Carolina OK, and WA -> 2058645 STATE OF WESTERN AUSTRALIA, which is the
one genuine misfire -- and it COSTS ZERO: the pipeline answers Washington ADM1
USA with AND without R1 (the baseline already had it at rank 39 and the ranker
already preferred it). NC in "the NC Dinos" and FL in "(R-FL)" likewise resolve
identically in both arms. (c) Raw text has PA x4 (Palestinian Authority), OK x3,
ID x3, MD, KY x2 ("KY 52") standalone and capitalised -- NONE reaches the
gazetteer, because the tagger does not label them as places. VERDICT: keep LA;
no held-out evidence of harm, the one ambiguous gold is already lost without the
rule, and the outer guard is the tagger. Mitigation documented in
place_aliases.py as a one-line deletion.

TRAPS. (1) The integration test must use "Ind.", not "Ky.": the baseline query
for Ky. DOES return Kentucky at rank 6, so the ranker finds it anyway and the
test passes for the wrong reason. Ind. returns 40 hits with no Indiana. (2)
trim_span_tokens can eat the period the AP guard depends on -- LGL 581's gold
"Vt." is emitted as the span "Vt" on the spaCy path and correctly refused; under
the head, which emits "Vt.", the same gold becomes correct. (3) The raw_data
pickles were built with the flag OFF, so the frozen ranker is scoring 65
candidate lists it was not trained on; that skew is INSIDE the measured number.
tools/train.py now defaults to the flag being on, which is what e52's rider
asked for -- rebuild the pickles WITH it and the win should grow, because 27 of
those entities were previously unlearnable.
