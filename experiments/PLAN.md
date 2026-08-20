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
