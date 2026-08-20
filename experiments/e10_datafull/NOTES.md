# e10_datafull

Wave 1b. Full fixed stack plus this config's flags:
`--logits --mask-padding --oov-bucket-fix --modern-mlp `, epochs 30, WikiDocs cap none (all 15,065).
Seeds 42/101/202/617/1848, score = mean of last 5 epochs, paired deltas vs
`experiments/e0_baseline/seed<S>.json`, `*` = |mean| > 2 SE.

## Command

```
for S in 42 101 202 617 1848; do
  WANDB_MODE=offline uv run python tools/train.py train --epochs 30 --mix-dim 512 \
    --dataset-names "Prodigy, TR, LGL, GWN, Synth, WikiDocs"  \
    --seed $S --logits --mask-padding --oov-bucket-fix --modern-mlp  \
    --metrics-out experiments/e10_datafull/seed$S.json > experiments/e10_datafull/seed$S.log 2>&1
done
```

## Wave 1b table (all configs)

| config | epochs | WikiDocs cap | extra flags | exact_match_avg Δ | acc_at_161 Δ | abs EM | peak EM | peak epoch | wall |
|---|---|---|---|---|---|---|---|---|---|
| e0_baseline (no fixes) | 30 | 2000 | — | — | — | 0.8808 | 0.8876 | 26.4 | ~55s |
| e4_mlp | 30 | 2000 | `—` | -0.0093 ± 0.0031 * | -0.0013 ± 0.0022 | 0.8716 | 0.8859 | 16.8 | ~55s |
| e7_ls005 | 30 | 2000 | `--label-smoothing 0.05` | -0.0076 ± 0.0025 * | -0.0006 ± 0.0012 | 0.8732 | 0.8903 | 12.0 | ~54s |
| e7_ls01 | 30 | 2000 | `--label-smoothing 0.1` | -0.0083 ± 0.0062 * | -0.0030 ± 0.0049 | 0.8725 | 0.8912 | 11.0 | ~54s |
| e7_ls02 | 30 | 2000 | `--label-smoothing 0.2` | -0.0089 ± 0.0041 * | -0.0042 ± 0.0030 * | 0.8719 | 0.8901 | 8.8 | ~56s |
| e8_wd001 | 30 | 2000 | `--weight-decay 0.01` | -0.0091 ± 0.0038 * | -0.0003 ± 0.0011 | 0.8717 | 0.8863 | 14.8 | ~59s |
| e8_wd01 | 30 | 2000 | `--weight-decay 0.1` | -0.0090 ± 0.0077 * | -0.0010 ± 0.0048 | 0.8718 | 0.8860 | 19.8 | ~73s |
| e9_short | 15 | 2000 | `—` | -0.0046 ± 0.0033 * | +0.0028 ± 0.0030 | 0.8762 | 0.8855 | 11.4 | ~43s |
| e10_data8k | 30 | 8000 | `—` | -0.0053 ± 0.0040 * | +0.0037 ± 0.0042 | 0.8755 | 0.8870 | 18.2 | ~105s |
| e10_datafull | 30 | none (15,065) | `—` | -0.0033 ± 0.0067 | +0.0046 ± 0.0047 | 0.8775 | 0.8886 | 16.8 | ~124s |
| e11_combo | 15 | none (15,065) | `—` | -0.0014 ± 0.0034 | +0.0083 ± 0.0031 * | 0.8794 | 0.8876 | 12.6 | ~53s |
| e11b_combo_ls | 15 | none (15,065) | `--label-smoothing 0.05` | +0.0037 ± 0.0022 * | +0.0101 ± 0.0018 * | 0.8845 | 0.8903 | 12.6 | ~53s |
| e11c_ls_short | 15 | 2000 | `--label-smoothing 0.05` | +0.0011 ± 0.0040 | +0.0052 ± 0.0039 * | 0.8819 | 0.8902 | 11.2 | ~34s |
| e12_ls01_full15 | 15 | none (15,065) | `--label-smoothing 0.1` | +0.0013 ± 0.0037 | +0.0077 ± 0.0022 * | 0.8822 | 0.8882 | 13.0 | ~53s |
| e12_ls005_full22 | 22 | none (15,065) | `--label-smoothing 0.05` | +0.0010 ± 0.0035 | +0.0064 ± 0.0020 * | 0.8818 | 0.8910 | 16.4 | ~72s |

## Per-source `exact_match` delta, this config

| source | exact_match Δ |
|---|---|
| Prodigy | -0.0355 ± 0.0132 * |
| TR | -0.0090 ± 0.0239 |
| LGL | +0.0038 ± 0.0094 |
| GWN | -0.0277 ± 0.0027 * |
| Synth | +0.0051 ± 0.0086 |
| WikiDocs | +0.0435 ± 0.0057 * |

## Verdict

All 15,065 WikiDocs examples. First arm whose EM delta is not significant (-0.0033 ± 0.0067) and acc@161 is positive (+0.0046). WikiDocs +0.044. Now that the loss can actually fit, more Wikipedia data does help -- the opposite of the pre-fix finding in WIKI_TRAINING_DATA.md, which was measured under the double softmax.
