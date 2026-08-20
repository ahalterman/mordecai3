# The accuracy campaign: 0.881 → 0.926 exact match in one day of experiments

Written 2026-08-20. Sequel to WIKI_TRAINING_DATA.md, same methodology: 5 seeds
{42, 101, 202, 617, 1848} per configuration, paired per-seed deltas, score =
mean of the last five epochs, `*` marks |mean| > 2 SE. Training is bit-for-bit
reproducible at a fixed seed, so same-seed differences between configs are pure
signal. The full ledger is `experiments/PLAN.md` plus a NOTES.md in each of the
60 `experiments/e*` directories.

## Headline

| | exact match | acc@161km |
|---|---|---|
| opening baseline (e0) | 0.8808 ± 0.0038 | 0.9262 ± 0.0029 |
| **final (e29_swa_ep15)** | **0.9258** | **0.9661** |
| delta (paired, 5 seeds) | **+0.0450 ± 0.0051 \*** | **+0.0399 ± 0.0024 \*** |

Every source improved: LGL +0.098\*, TR +0.062\*, WikiDocs +0.060\*, GWN,
Prodigy, Synth all positive. Seed-to-seed spread was cut roughly in half.
Under the new secondary twin-credit metric (which stops charging the model for
the city-vs-same-named-admin-unit convention), the final model reads 93.8%.

## The ship recipe

```
uv run python tools/train.py train \
  --epochs 15 --mix-dim 512 \
  --logits --mask-padding --oov-bucket-fix --modern-mlp \
  --label-smoothing 0.05 \
  --enriched --feature-blocks "prom,name,cue,sib,geo,shape" \
  --avg-params --avg-mode swa \
  --dataset-names "Prodigy, TR, LGL, GWN, Synth, WikiDocs" \
  --checkpoint-out <path>       # writes <path> + <path>.json config sidecar
```

Serving: `Geoparser(feature_blocks="prom,name,cue,sib,geo,shape",
oov_bucket_fix=True, model_options={"return_logits": True, "mask_padding":
True, "modern_mlp": True})`. All 26 features are computed at inference by
`mordecai3/candidate_features.py` — the identical functions the training
enrichment used, proven exact to 0.0 on 43k candidate rows
(`tests/test_feature_parity.py`) — at +14% end-to-end latency. Keep
`max_choices=100`: the 100-vs-500 window feature drift measurably costs
nothing; 500 is a flag-worthy option buying +0.33 EM of retrieval recall for
+51% latency. If a deployment can afford three trainings, a 3-seed
probability-averaged ensemble (`tools/ensemble_eval.py`) reaches 93.1%
strict / 94.3% twin-credit — but SWA and ensembling attack the same variance,
so ensemble plain-SWA seeds rather than stacking both.

## Where the 4.5 points came from

1. **Enriched candidate features: ~3.4 points** (Wave 2). The old scorer saw
   13 scalars per candidate. Twenty-six additions, screened offline before any
   training run: within-set population features (`is_max_pop_exact_match`
   separates gold from non-gold ~450:1 on LGL), exact-name/alt-name matches,
   admin-cue × feature-class interaction, sibling parent-name matches
   (P(gold) 4.0% → 46.6% given `sib_adm1` on same-name candidates), and
   sibling anchor geometry (`log_min_km_anchor`: residual accuracy 0.519 where
   the population prior fails at 0.090). Document membership was recovered
   from the existing pickles by hashing `doc_tensor` — no rebuild needed.
2. **Loss and recipe: ~0.4 points, and the enabler for everything else**
   (Wave 1). `forward()` applied softmax before `CrossEntropyLoss` — a double
   softmax that turned out to be the model's accidental (extreme) regularizer.
   Removing it alone *hurts*; removing it plus label smoothing 0.05 plus a
   short schedule plus uncapping WikiDocs wins. Also fixed: padding masking,
   OOV feature codes colliding with the NULL embedding, sentinel-row country
   leakage, GELU/dropout placement.
3. **Weight averaging: ~0.7 points** (Wave 4). The legacy `--avg-params` never
   evaluated or saved the averaged model; rewritten as shadow-weight SWA over
   epochs 8–15. The only arm that improved all six sources.

## Reversals and negative results (the part worth rereading)

- **"More Wikipedia buys nothing" was wrong** — an artifact of the double
  softmax, under which the loss could not fit even the small data. Uncapped
  WikiDocs is part of the winning recipe. WIKI_TRAINING_DATA.md's pipeline
  stands; its conclusion is repealed.
- **Architecture is a dead end.** Every capacity knob (country/code/mix
  width, depth, residuals) is flat; the model is feature-limited. Listwise
  attention over candidates is actively destructive (−0.036 to −0.058\*,
  unstable). The multitask country head buys a small acc@161 gain
  (+0.0024\*) at Prodigy's expense — optional.
- **The A/P "label problem" dissolved on inspection.** The R\*\* label rewrite
  is a convention swap: the Wave-2 model simply learns whichever convention
  it is trained on (74.6% vs 93.3% on the disputed labels depending on the
  key), and under twin-credit the learning effect is exactly 0.0000. Adopt
  the metric (`tools/twin_credit_eval.py`), not the rewrite.
- **Population is a weak prior here** (coin flip on same-name errors; its
  block is largely implied by geometry+siblings), **Wikipedia fame priors
  don't beat it**, **decode-time score pooling over repeat mentions is nearly
  exhausted** (the model is already self-consistent on 95% of repeat groups;
  errors are unanimous), and **stripped-name twin features** — despite 2.5×
  coverage at equal precision offline — are negative in training, including
  on the metric they were built to move.
- **Retrieval was never the constraint** (98.5% candidate recall) and still
  isn't.
- Two latent bugs are now deliberate defaults: dropout has been inert since
  epoch 2 in every run (fixing it hurts), and the cosine LR schedule was
  never applied (applying it makes SWA worse). Flat 1e-3, no live dropout
  after warmup: that is what all reported numbers mean.

## What's left, if anyone wants more

The fresh error analysis (jobs tmp `erroranalysis2/report.md`) puts the model
near this design's ceiling: gold and prediction have indistinguishable feature
profiles on the residual errors, only 4 of 8,838 held-out entities are
architecturally unscoreable, and two same-recipe seeds share only 70% of their
errors. Directions with real headroom:

1. **Replace the frozen spaCy en_core_web_trf token tensors** — the one
   untouched component. Requires re-running the NLP stage; decide the encoder
   from where the residual errors live.
2. **Expose the abstention signal**: the unselectable reserved row wins the
   raw argmax on 1.6% of entities, and those answers are right only 56–72% of
   the time vs 91–92% otherwise — a free confidence flag for serving.
3. **A gazetteer-hygiene pass**: a quarter of the residual error mass is ~20
   repeated strings (D.C., Mauna Kea's duplicate rows, Kathmandu's historical
   ADM3H entry).

## State of the tree

All changes are uncommitted on `wiki-training-data`, deliberately: the
campaign's edits (`mordecai3/torch_model.py`, `mordecai3/geoparse.py`, new
`mordecai3/candidate_features.py`, `tools/train.py`, `tools/enrich_pickles.py`,
`tools/rewrite_labels.py`, `tools/twin_credit_eval.py`,
`tools/ensemble_eval.py`, `tests/test_feature_parity.py`, `experiments/`)
should be reviewed and committed in whatever slices make sense. Enriched
pickles (3.7 GB, 33 features) rebuild in ~5 minutes via
`tools/enrich_pickles.py`; compact caches rebuild automatically. Two
pre-existing test failures (`test_miss_oxford`, `test_prague`) predate the
campaign.
