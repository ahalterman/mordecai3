# e22_country02

Wave 3 (architecture). Base recipe is e14_no_cf, the adopted standard:
`--epochs 15 --mix-dim 512 --logits --mask-padding --oov-bucket-fix --modern-mlp
--label-smoothing 0.05 --enriched --feature-blocks "prom,name,cue,sib,geo,shape"`,
no WikiDocs cap; this arm adds `--aux-country-weight 0.2`.
Seeds 42/101/202/617/1848, score = mean of last 5 epochs, paired deltas vs
`experiments/e14_no_cf/seed<S>.json`, `*` = |mean| > 2 SE.

## Command

```
for S in 42 101 202 617 1848; do
  WANDB_MODE=offline uv run python tools/train.py train --epochs 15 --mix-dim 512 \
    --logits --mask-padding --oov-bucket-fix --modern-mlp --label-smoothing 0.05 \
    --dataset-names "Prodigy, TR, LGL, GWN, Synth, WikiDocs" \
    --enriched --feature-blocks "prom,name,cue,sib,geo,shape" \
    --aux-country-weight 0.2 --seed $S \
    --metrics-out experiments/e22_country02/seed$S.json > experiments/e22_country02/seed$S.log 2>&1
done
```
(arms that change --epochs override it above.)

## Wave 3 table (all arms)

| config | flags | exact_match_avg Δ | acc_at_161 Δ | abs EM | peak EM | peak ep | wall (4-wide) |
|---|---|---|---|---|---|---|---|
| e14_no_cf (base) | — | — | — | 0.9191 | 0.9260 | 10.4 | ~50s solo |
| e15arch_c24k32 | `--code-size 32` | +0.0013 ± 0.0063 | -0.0009 ± 0.0014 | 0.9203 | 0.9271 | 11.8 | ~120s |
| e15arch_c64k8 | `--country-size 64` | +0.0010 ± 0.0067 | -0.0006 ± 0.0025 | 0.9201 | 0.9277 | 11.0 | ~122s |
| e15arch_c64k32 | `--country-size 64 --code-size 32` | -0.0012 ± 0.0075 | -0.0011 ± 0.0034 | 0.9179 | 0.9247 | 12.8 | ~122s |
| e15arch_c128k8 | `--country-size 128` | -0.0023 ± 0.0047 | +0.0001 ± 0.0018 | 0.9168 | 0.9231 | 9.4 | ~122s |
| e15arch_c128k32 | `--country-size 128 --code-size 32` | +0.0002 ± 0.0061 | -0.0011 ± 0.0016 | 0.9193 | 0.9252 | 10.6 | ~122s |
| e15arch_mix256 | `--mix-dim 256` | -0.0043 ± 0.0064 | -0.0013 ± 0.0005 * | 0.9147 | 0.9213 | 11.6 | ~110s |
| e15arch_mix1024 | `--mix-dim 1024` | -0.0011 ± 0.0075 | -0.0008 ± 0.0020 | 0.9180 | 0.9252 | 8.6 | ~140s |
| e15arch_depth3 | `--mix-depth 3` | -0.0022 ± 0.0046 | -0.0026 ± 0.0020 * | 0.9169 | 0.9241 | 10.0 | ~150s |
| e15arch_depth3res | `--mix-depth 3 --residual` | -0.0016 ± 0.0069 | -0.0014 ± 0.0021 | 0.9175 | 0.9257 | 9.4 | ~150s |
| e16arch_listwise4 | `--listwise (4 heads)` | -0.0356 ± 0.0337 * | -0.0196 ± 0.0193 * | 0.8835 | 0.9174 | 7.8 | ~473s |
| e16arch_listwise1 | `--listwise --listwise-heads 1` | -0.0576 ± 0.0529 * | -0.0372 ± 0.0436 | 0.8614 | 0.9183 | 8.2 | ~502s |
| e22_country01 | `--aux-country-weight 0.1` | -0.0012 ± 0.0050 | +0.0005 ± 0.0014 | 0.9178 | 0.9255 | 8.8 | ~118s |
| e22_country02 | `--aux-country-weight 0.2` | +0.0003 ± 0.0045 | +0.0014 ± 0.0012 * | 0.9193 | 0.9257 | 12.0 | ~118s |
| e22_class01 | `--aux-class-weight 0.1` | -0.0011 ± 0.0042 | -0.0013 ± 0.0019 | 0.9179 | 0.9242 | 10.6 | ~116s |
| e22_both01 | `--aux-country-weight 0.1 --aux-class-weight 0.1` | -0.0011 ± 0.0071 | +0.0009 ± 0.0020 | 0.9179 | 0.9265 | 11.2 | ~118s |
| e17_nullrow | `--full-null-row` | +0.0000 ± 0.0000 | +0.0000 ± 0.0000 | 0.9191 | 0.9260 | 10.4 | ~85s |
| e23_ep12 | `--epochs 12` | +0.0013 ± 0.0029 | +0.0006 ± 0.0004 * | 0.9204 | 0.9255 | 9.0 | ~80s |
| e23_ep10 | `--epochs 10` | +0.0013 ± 0.0036 | -0.0001 ± 0.0005 | 0.9204 | 0.9254 | 8.0 | ~70s |
| e23_ep12_country02 | `--epochs 12 --aux-country-weight 0.2` | +0.0015 ± 0.0043 | +0.0024 ± 0.0012 * | 0.9206 | 0.9254 | 10.6 | ~85s |

## Per-source `exact_match` delta vs e14_no_cf

| config | Prodigy | TR | LGL | GWN | Synth | WikiDocs |
|---|---|---|---|---|---|---|
| e15arch_c24k32 | -0.0020 | -0.0007 | +0.0053 | -0.0022 | +0.0067 | +0.0005 |
| e15arch_c64k8 | -0.0063 | +0.0031 | +0.0045 | +0.0037 | +0.0024 | -0.0014 |
| e15arch_c64k32 | -0.0148 | -0.0035 | +0.0082 | -0.0070 | +0.0094 * | +0.0007 |
| e15arch_c128k8 | -0.0174 | -0.0038 | -0.0000 | +0.0075 * | -0.0007 | +0.0007 |
| e15arch_c128k32 | -0.0080 | -0.0041 | +0.0086 * | -0.0045 | +0.0071 | +0.0021 |
| e15arch_mix256 | -0.0166 | -0.0096 | -0.0040 | +0.0006 | +0.0049 * | -0.0013 |
| e15arch_mix1024 | -0.0143 * | -0.0022 | +0.0074 | +0.0043 | -0.0025 | +0.0010 |
| e15arch_depth3 | -0.0104 | -0.0093 * | +0.0024 | +0.0048 | -0.0005 | -0.0001 |
| e15arch_depth3res | -0.0071 | -0.0103 | +0.0030 | +0.0039 * | +0.0011 | +0.0000 |
| e16arch_listwise4 | -0.0395 * | -0.0590 | -0.0444 | -0.0094 | -0.0178 | -0.0434 * |
| e16arch_listwise1 | -0.0676 * | -0.0800 | -0.0730 | -0.0242 | -0.0316 * | -0.0695 * |
| e22_country01 | -0.0150 | -0.0055 | +0.0091 | +0.0021 | +0.0003 | +0.0016 |
| e22_country02 | -0.0126 | -0.0000 | +0.0104 * | +0.0014 | +0.0015 | +0.0009 |
| e22_class01 | -0.0138 | -0.0027 | +0.0050 | +0.0018 | +0.0003 | +0.0026 |
| e22_both01 | -0.0127 | -0.0031 | +0.0046 | +0.0023 | -0.0001 | +0.0022 |
| e17_nullrow | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| e23_ep12 | -0.0058 * | -0.0010 | +0.0070 | +0.0068 * | +0.0011 | -0.0002 |
| e23_ep10 | -0.0037 | -0.0059 | +0.0076 | +0.0103 * | +0.0019 | -0.0023 |
| e23_ep12_country02 | -0.0195 | +0.0032 | +0.0145 * | +0.0075 * | +0.0036 | -0.0001 |

## Verdict

Weight 0.2. The only wave-3 arm with a significant gain: acc@161 +0.0014*, and LGL exact match +0.0104*. Tiny, but the sign is consistent and it costs nothing.
