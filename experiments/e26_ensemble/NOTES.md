# e26_ensemble

Decode-time K-seed ensembling. Not a training change: `tools/ensemble_eval.py`
loads K checkpoints, softmaxes each model's scores over the live candidate rows,
averages the probabilities and argmaxes. Entity filtering and the twin-credit
definition are imported from `tools/twin_credit_eval.py`, so these numbers sit
on the frozen metric's denominator.

Averaging is done in probability space, not logit space: these checkpoints are
trained with `--logits`, so their raw score scales are not comparable across
seeds and one over-confident model would dominate a logit average.

Every K row averages over *all* subsets of that size (5 singletons, 10 pairs,
10 triples, 5 quadruples, 1 quintuple), so K=1 is an unbiased "expected single
model" and K=2 is not cherry-picked.

## Commands

```
uv run python tools/ensemble_eval.py --checkpoints "experiments/e14_no_cf/seed*.pt" \
  --json-out experiments/e26_ensemble/ens_e14.json
uv run python tools/ensemble_eval.py --checkpoints "experiments/e25_swa/seed*.pt" ...
uv run python tools/ensemble_eval.py --checkpoints "experiments/e29_swa_ep15/seed*.pt" ...
uv run python tools/ensemble_eval.py --checkpoints "experiments/e27_strip_ep15/seed*.pt" --k 1 3 5 ...
```

## Results (macro exact match over the six sources)

| config | K=1 | K=2 | K=3 | K=5 | twin K=1 | twin K=5 |
|---|---|---|---|---|---|---|
| e14_no_cf | 91.70% | 92.68% | 93.00% | 93.35% | 93.03% | 94.28% |
| e25_swa | 92.59% | 93.00% | 93.04% | 93.01% | 93.81% | 94.18% |
| e29_swa_ep15 | 92.58% | 93.01% | 93.13% | 93.17% | 93.79% | 94.28% |
| e27_strip_ep15 | 92.32% | — | 92.93% | 92.99% | 93.59% | 93.98% |

Per-source, e14_no_cf checkpoints:

| K | Prodigy | TR | LGL | GWN | Synth | WikiDocs | macro EM | macro twin |
|---|---|---|---|---|---|---|---|---|
| 1 | 91.36% | 90.18% | 87.65% | 91.83% | 96.72% | 92.43% | **91.70%** | 93.03% |
| 2 | 92.18% | 91.00% | 88.93% | 93.02% | 97.83% | 93.12% | **92.68%** | 93.77% |
| 3 | 92.58% | 91.22% | 89.25% | 93.57% | 98.06% | 93.31% | **93.00%** | 94.08% |
| 4 | 92.92% | 91.29% | 89.34% | 93.78% | 98.26% | 93.39% | **93.16%** | 94.17% |
| 5 | 93.40% | 91.51% | 89.22% | 94.13% | 98.33% | 93.49% | **93.35%** | 94.28% |

## Verdict

Ensembling is the largest single lever left: **+0.98 macro EM at K=2, +1.30 at
K=3, +1.65 at K=5** over the expected single model, and it lifts every source.
Twin credit moves with it (+1.25 at K=5).

The important interaction: **SWA and ensembling are substitutes, not
complements.** SWA alone buys +0.89 of the +1.65 a five-seed ensemble buys, for
one fifth of the training cost, and ensembling *SWA* models plateaus around
93.0-93.2 by K=3 -- no better than ensembling five plain models. Both are
attacking the same run-to-run variance. Ship SWA if you train once; ensemble
three seeds if you can afford three trainings; do not expect the two to add.
