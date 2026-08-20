# e28_cosine

Wave 4 (closing wave: variance levers). Base recipe is e14_no_cf:
`--mix-dim 512 --logits --mask-padding --oov-bucket-fix --modern-mlp
--label-smoothing 0.05 --enriched --feature-blocks "prom,name,cue,sib,geo,shape"`,
no WikiDocs cap. This arm: cosine LR schedule stepped (`--epochs 12 --lr-schedule`).
Seeds 42/101/202/617/1848, score = mean of last 5 epochs, paired deltas vs
`experiments/e14_no_cf/seed<S>.json`, `*` = |mean| > 2 SE. Arms that change the
epoch count are also compared to the matched-schedule control e23_ep12.

## Command

```
for S in 42 101 202 617 1848; do
  WANDB_MODE=offline uv run python tools/train.py train --mix-dim 512 \
    --logits --mask-padding --oov-bucket-fix --modern-mlp --label-smoothing 0.05 \
    --dataset-names "Prodigy, TR, LGL, GWN, Synth, WikiDocs" \
    --enriched --feature-blocks "prom,name,cue,sib,geo,shape" \
    --epochs 12 --lr-schedule --seed $S \
    --checkpoint-out experiments/e28_cosine/seed$S.pt \
    --metrics-out experiments/e28_cosine/seed$S.json > experiments/e28_cosine/seed$S.log 2>&1
done
```

## Wave 4 table

| config | what | exact_match_avg Δ | acc_at_161 Δ | abs EM | vs matched control |
|---|---|---|---|---|---|
| e14_no_cf (base) | e14 recipe, 15 epochs | — | — | 0.9191 | — |
| e23_ep12 | 12 epochs (schedule control) | +0.0013 ± 0.0029 | +0.0006 ± 0.0004 * | 0.9204 | — |
| e25_swa | SWA, average epochs 7-12 | +0.0055 ± 0.0030 * | +0.0020 ± 0.0006 * | 0.9246 | +0.0042 ± 0.0031 * vs e23_ep12 |
| e25_ema | EMA of weights, decay 0.9 | +0.0049 ± 0.0034 * | +0.0009 ± 0.0014 | 0.9240 | +0.0036 ± 0.0037 vs e23_ep12 |
| e29_swa_cosine | SWA + the cosine schedule actually stepped | +0.0047 ± 0.0030 * | +0.0014 ± 0.0004 * | 0.9237 | +0.0034 ± 0.0044 vs e23_ep12 |
| e29_swa_ep15 | SWA at 15 epochs, average epochs 8-15 | +0.0067 ± 0.0032 * | +0.0029 ± 0.0012 * | 0.9258 | — |
| e28_trainmode | restore train() after each eval (dropout bug) | -0.0008 ± 0.0044 | -0.0005 ± 0.0012 | 0.9182 | -0.0021 ± 0.0022 vs e23_ep12 |
| e28_cosine | cosine LR schedule stepped | +0.0028 ± 0.0027 * | +0.0004 ± 0.0005 | 0.9219 | +0.0015 ± 0.0038 vs e23_ep12 |
| e27_strip_nosw | strip block, no averaging | -0.0008 ± 0.0082 | -0.0017 ± 0.0024 | 0.9183 | -0.0021 ± 0.0089 vs e23_ep12 |
| e27_strip_ep12 | strip block + SWA | +0.0020 ± 0.0084 | -0.0006 ± 0.0017 | 0.9211 | -0.0035 ± 0.0072 vs e25_swa |
| e27_strip_ep15 | strip block + SWA at 15 epochs | +0.0037 ± 0.0080 | -0.0002 ± 0.0023 | 0.9227 | -0.0030 ± 0.0068 vs e29_swa_ep15 |

## Per-source `exact_match` delta vs e14_no_cf

| config | Prodigy | TR | LGL | GWN | Synth | WikiDocs |
|---|---|---|---|---|---|---|
| e23_ep12 | -0.0058 * | -0.0010 | +0.0070 | +0.0068 * | +0.0011 | -0.0002 |
| e25_swa | +0.0015 | -0.0010 | +0.0108 * | +0.0123 * | +0.0067 * | +0.0030 |
| e25_ema | +0.0050 | -0.0019 | +0.0065 * | +0.0134 * | +0.0066 | -0.0002 |
| e29_swa_cosine | -0.0010 | -0.0016 | +0.0101 * | +0.0118 * | +0.0064 | +0.0023 * |
| e29_swa_ep15 | +0.0031 * | +0.0052 | +0.0131 * | +0.0097 * | +0.0040 * | +0.0052 * |
| e28_trainmode | -0.0101 * | -0.0121 * | +0.0048 | +0.0085 * | +0.0055 * | -0.0017 |
| e28_cosine | -0.0026 | -0.0012 | +0.0089 * | +0.0071 * | +0.0047 | -0.0001 |
| e27_strip_nosw | -0.0025 | -0.0123 * | +0.0025 | +0.0086 | +0.0036 | -0.0045 * |
| e27_strip_ep12 | -0.0011 | -0.0124 * | +0.0052 | +0.0117 | +0.0076 * | +0.0012 |
| e27_strip_ep15 | +0.0021 | -0.0071 | +0.0061 | +0.0115 | +0.0068 * | +0.0027 |

## Decode-time ensembling (tools/ensemble_eval.py, final-epoch checkpoints, macro over sources)

| config | K=1 | K=2 | K=3 | K=5 | twin K=1 | twin K=5 |
|---|---|---|---|---|---|---|
| e14_no_cf | 91.70% | 92.68% | 93.00% | 93.35% | 93.03% | 94.28% |
| e25_swa | 92.59% | 93.00% | 93.04% | 93.01% | 93.81% | 94.18% |
| e29_swa_ep15 | 92.58% | 93.01% | 93.13% | 93.17% | 93.79% | 94.28% |
| e27_strip_ep15 | 92.32% | — | 92.93% | 92.99% | 93.59% | 93.98% |

## Verdict

The second latent bug: `scheduler.step()` sat inside the `avg_params` branch, so the cosine LR schedule that every run constructs has never actually been applied -- the campaign trained at a flat 1e-3 throughout. Stepping it gives +0.0028* against e14_no_cf but only +0.0015 (n.s.) against the matched 12-epoch control, i.e. most of the apparent gain is the shorter schedule. Not worth adopting on its own, and it makes SWA slightly worse.
