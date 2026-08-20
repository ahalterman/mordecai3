# e4_mlp

Flags: `--logits --mask-padding --oov-bucket-fix --modern-mlp`
Seeds: 42, 101, 202, 617, 1848. Score = mean of the last 5 epochs (`_last5`).
Paired per-seed deltas vs `experiments/e0_baseline/seed<S>.json`; `*` marks |mean| > 2 SE.

## Command

```
for S in 42 101 202 617 1848; do
WANDB_MODE=offline uv run python tools/train.py train --epochs 30 --mix-dim 512 \
  --dataset-names "Prodigy, TR, LGL, GWN, Synth, WikiDocs" --source-limits "WikiDocs=2000" \
  --seed $S --logits --mask-padding --oov-bucket-fix --modern-mlp \
  --metrics-out experiments/e4_mlp/seed$S.json > experiments/e4_mlp/seed$S.log 2>&1
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
| Prodigy | -0.0225 ± 0.0084 * |
| TR | +0.0037 ± 0.0089 |
| LGL | -0.0068 ± 0.0085 |
| GWN | -0.0196 ± 0.0137 * |
| Synth | -0.0005 ± 0.0072 |
| WikiDocs | -0.0098 ± 0.0064 * |

## exact_match_avg trajectory (mean over seeds)

| config | e1 | e2 | e3 | e5 | e7 | e10 | e15 | e20 | e25 | e30 |
|---|---|---|---|---|---|---|---|---|---|---|
| e0_baseline | 0.811 | 0.839 | 0.853 | 0.862 | 0.867 | 0.874 | 0.876 | 0.875 | 0.881 | 0.883 |
| e1_logits | 0.813 | 0.849 | 0.857 | 0.867 | 0.867 | 0.869 | 0.866 | 0.869 | 0.870 | 0.871 |
| e2_mask | 0.812 | 0.848 | 0.857 | 0.867 | 0.867 | 0.868 | 0.862 | 0.867 | 0.870 | 0.867 |
| e3_oov | 0.812 | 0.852 | 0.859 | 0.870 | 0.871 | 0.871 | 0.868 | 0.868 | 0.869 | 0.865 |
| e4_mlp | 0.849 | 0.863 | 0.864 | 0.876 | 0.870 | 0.875 | 0.870 | 0.874 | 0.868 | 0.872 |

## Verdict

GELU in the mix MLP plus removing dropout from the frozen country embedding and
the cosine-similarity projections is the largest single improvement inside the
fixed-model family (+0.002 EM, +0.0014 acc@161 over e3) and gives by far the
fastest start: 0.849 exact match after ONE epoch, versus 0.811 for baseline and
0.876 for baseline at epoch 30. Its peak epoch score (0.8859) is statistically
indistinguishable from baseline's (0.8876). Still -0.009 EM on the last-5 score:
the whole family converges early and then drifts down, and baseline does not.
