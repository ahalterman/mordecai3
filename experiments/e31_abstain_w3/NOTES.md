# e31_abstain_w3 — upweight the "no correct answer" class in the loss (3x)

Second campaign, calibration/abstention track. Base recipe is the ship recipe
`e29_swa_ep15` unchanged, plus `--abstain-weight 3.0`. Window stays 500, so
this arm isolates *more emphasis on the abstention class* from e30's *shorter
window*.

**Verdict: REJECTED. It buys a third of e30's OOD gain and costs more
elsewhere — accuracy down (confirmed), wrong-answer AUROC down (confirmed),
selective EM at 90% coverage down (confirmed), flag precision down. Dominated
by e30 on the abstention side and by e29 on everything else.**

## Command

```
for S in 42 101 202 617 1848; do
  WANDB_MODE=offline uv run python tools/train.py train --mix-dim 512 \
    --logits --mask-padding --oov-bucket-fix --modern-mlp --label-smoothing 0.05 \
    --dataset-names "Prodigy, TR, LGL, GWN, Synth, WikiDocs" \
    --enriched --feature-blocks "prom,name,cue,sib,geo,shape" \
    --epochs 15 --avg-params --avg-mode swa --seed $S \
    --abstain-weight 3.0 \
    --checkpoint-out experiments/e31_abstain_w3/seed$S.pt \
    --metrics-out experiments/e31_abstain_w3/seed$S.json > experiments/e31_abstain_w3/seed$S.log 2>&1
done
```

`--abstain-weight w` multiplies the per-example loss by `w` for the examples
whose target is the reserved row (508 of 20,951 training entities, 2.42%; 125
of those are Synth). It reaches the loss only through `masked_smoothed_ce`, so
`train.py` rejects it without `--label-smoothing` and `--mask-padding`. At the
default 1.0 no weights are built and the loss takes its original code path —
verified byte-identical on `e29_swa_ep15/seed42.json`.

Unlike e30, this arm's `seedS.json` **is** on the ledger key (window 500), so
`_last5.exact_match_avg` is directly comparable to the campaign's numbers.
The tables below still use checkpoint re-scoring, so that e29/e30/e31 sit on
one denominator.

## Paired deltas vs e29_swa_ep15 (5 seeds, t(4), crit 2.776)

### Frozen key (window 500)

| metric | e29 | e31 | Δ | 2 SE | t(4) | verdict |
|---|---|---|---|---|---|---|
| macro EM, 6 sources | 0.9258 | 0.9225 | −0.0033 | 0.0021 | −3.20 | confirmed loss |
| pooled EM, group (a) | 0.9259 | 0.9226 | −0.0033 | 0.0017 | −3.88 | confirmed loss |
| TLG-hard (non-country golds) | 0.8730 | 0.8707 | −0.0023 | 0.0061 | −0.75 | ns |
| EM over all held-out mentions | 0.9115 | 0.9083 | −0.0033 | 0.0017 | −3.88 | confirmed loss |

Per source (Δ EM, window 500): WikiDocs −0.0032 (t −3.6), Prodigy −0.0104
(t −1.6), LGL −0.0025, TR −0.0022, GWN −0.0009, Synth −0.0007.

### Serving key (window 100)

| metric | e29 | e31 | Δ | 2 SE | t(4) | verdict |
|---|---|---|---|---|---|---|
| AUROC `p_reserved`, unanswerable | 0.8582 | 0.8844 | +0.0263 | 0.0087 | +6.01 | confirmed win |
| unanswerable caught by the flag (of 228) | 84.6 | 104.6 | +20.0 | 2.53 | +15.81 | confirmed win |
| flag precision | 0.8828 | 0.8405 | −0.0422 | 0.0316 | −2.67 | 2 SE only (loss) |
| flag recall over all wrong answers | 0.1419 | 0.1883 | +0.0465 | 0.0182 | +5.10 | confirmed |
| AUROC `p_pred_full`, wrong-answer | 0.8972 | 0.8872 | −0.0100 | 0.0057 | −3.49 | confirmed loss |
| AUROC `p_top1` | 0.8893 | 0.8774 | −0.0119 | 0.0054 | −4.42 | confirmed loss |
| AUROC combo | 0.9057 | 0.9036 | −0.0021 | 0.0068 | −0.62 | ns |
| pooled EM, group (a) answerable | 0.9296 | 0.9257 | −0.0039 | 0.0019 | −4.02 | confirmed loss |
| TLG-hard, answerable | 0.8915 | 0.8905 | −0.0011 | 0.0031 | −0.68 | ns |
| **selective EM @ 90% coverage** | 0.9556 | 0.9500 | −0.0056 | 0.0021 | −5.44 | **confirmed loss** |
| **AURC (lower better)** | 0.0197 | 0.0229 | +0.0033 | 0.0030 | +2.20 | 2 SE only (loss) |
| ECE, raw | 0.0230 | 0.0270 | +0.0040 | 0.0033 | +2.41 | 2 SE only |
| ECE, after LOSO temperature | 0.0238 | 0.0233 | −0.0005 | 0.0019 | −0.52 | ns |

## Why it fails where e30 half-succeeds

Weighting cannot teach what the labels do not contain. At window 500 the
reserved row's positives are only the 1.55% of mentions whose gold is missing
from the candidate list entirely; the 0.99% whose gold is merely *past row 100*
are still labelled with their true row, because at window 500 they are in
window. So e31's ceiling on the serving-time OOD problem is structurally lower
than e30's, and it pays for the emphasis with a blunter ranker: upweighting
2.4% of examples by 3x is a 7% reallocation of the gradient budget toward a
class the frozen metric cannot even see, and the confirmed drops in
`p_pred_full` AUROC and selective EM are that budget leaving the ranking task.

No follow-up weight sweep is recommended: the direction of every non-abstention
metric is already wrong at 3x, and the abstention gain it does buy is available
more cheaply and more precisely from e30's labels, or free from decode-time
thresholds on `e29`.
