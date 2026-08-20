# e1_logits

Flags: `--logits`
Seeds: 42, 101, 202, 617, 1848. Score = mean of the last 5 epochs (`_last5`).
Paired per-seed deltas vs `experiments/e0_baseline/seed<S>.json`; `*` marks |mean| > 2 SE.

## Command

```
for S in 42 101 202 617 1848; do
WANDB_MODE=offline uv run python tools/train.py train --epochs 30 --mix-dim 512 \
  --dataset-names "Prodigy, TR, LGL, GWN, Synth, WikiDocs" --source-limits "WikiDocs=2000" \
  --seed $S --logits \
  --metrics-out experiments/e1_logits/seed$S.json > experiments/e1_logits/seed$S.log 2>&1
done
```

## Ladder (all configs, for context)

| config | flags | exact_match_avg Δ | acc_at_161 Δ | abs EM (last5) | peak EM | peak epoch |
|---|---|---|---|---|---|---|
| e0_baseline | (none) | — | — | 0.8808 | 0.8876 | 26.4 |
| e1_logits | `--logits` | -0.0129 ± 0.0038 * | -0.0055 ± 0.0019 * | 0.8679 | 0.8804 | 17.2 |
| e2_mask | `--logits --mask-padding` | -0.0135 ± 0.0040 * | -0.0065 ± 0.0026 * | 0.8673 | 0.8795 | 21.6 |
| e3_oov | `--logits --mask-padding --oov-bucket-fix` | -0.0109 ± 0.0046 * | -0.0027 ± 0.0035 | 0.8699 | 0.8803 | 15.6 |
| e4_mlp | `--logits --mask-padding --oov-bucket-fix --modern-mlp` | -0.0093 ± 0.0031 * | -0.0013 ± 0.0022 | 0.8716 | 0.8859 | 16.8 |

## Per-source `exact_match` delta, this config

| source | exact_match Δ |
|---|---|
| Prodigy | -0.0134 ± 0.0119 * |
| TR | +0.0041 ± 0.0099 |
| LGL | -0.0204 ± 0.0032 * |
| GWN | -0.0170 ± 0.0089 * |
| Synth | -0.0140 ± 0.0067 * |
| WikiDocs | -0.0168 ± 0.0038 * |

## exact_match_avg trajectory (mean over seeds)

| config | e1 | e2 | e3 | e5 | e7 | e10 | e15 | e20 | e25 | e30 |
|---|---|---|---|---|---|---|---|---|---|---|
| e0_baseline | 0.811 | 0.839 | 0.853 | 0.862 | 0.867 | 0.874 | 0.876 | 0.875 | 0.881 | 0.883 |
| e1_logits | 0.813 | 0.849 | 0.857 | 0.867 | 0.867 | 0.869 | 0.866 | 0.869 | 0.870 | 0.871 |
| e2_mask | 0.812 | 0.848 | 0.857 | 0.867 | 0.867 | 0.868 | 0.862 | 0.867 | 0.870 | 0.867 |
| e3_oov | 0.812 | 0.852 | 0.859 | 0.870 | 0.871 | 0.871 | 0.868 | 0.868 | 0.869 | 0.865 |
| e4_mlp | 0.849 | 0.863 | 0.864 | 0.876 | 0.870 | 0.875 | 0.870 | 0.874 | 0.868 | 0.872 |

## Verdict

Removing the double softmax makes optimization work as intended -- training loss
drops from ~5.35 (barely below the log(500)=6.21 uniform ceiling) to ~0.22, and
held-out accuracy in the first few epochs is 1-2 points ahead of baseline. But
the run then plateaus around epoch 17 and decays, while the baseline keeps
climbing to epoch ~26. Net: **-0.013 exact match, significant. Do not adopt on
its own** under this 30-epoch / lr=1e-3 recipe.
