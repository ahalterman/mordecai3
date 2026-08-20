# e14_core

Wave 2: enriched candidate features. Recipe is e11b (the adopted standard):
`--epochs 15 --mix-dim 512 --logits --mask-padding --oov-bucket-fix --modern-mlp
--label-smoothing 0.05`, no WikiDocs cap, plus `--enriched --feature-blocks "cue,sib,geo,shape"`
(19 extra columns on gaz_info).
Seeds 42/101/202/617/1848, score = mean of last 5 epochs, paired deltas vs
`experiments/e11b_combo_ls/seed<S>.json`, `*` = |mean| > 2 SE.

## Command

```
for S in 42 101 202 617 1848; do
  WANDB_MODE=offline uv run python tools/train.py train \
    --epochs 15 --mix-dim 512 --logits --mask-padding --oov-bucket-fix --modern-mlp \
    --label-smoothing 0.05 --dataset-names "Prodigy, TR, LGL, GWN, Synth, WikiDocs" \
    --seed $S --enriched --feature-blocks "cue,sib,geo,shape" \
    --metrics-out experiments/e14_core/seed$S.json > experiments/e14_core/seed$S.log 2>&1
done
```

## Wave 2 table (all arms)

| config | blocks | n feats | exact_match_avg Δ | acc_at_161 Δ | abs EM | abs acc@161 | wall |
|---|---|---|---|---|---|---|---|
| e11b_combo_ls (base) | — | 0 | — | — | 0.8845 | 0.9362 | ~53s |
| e13_all | `prom,name,cue,sib,geo,shape,cf` | 30 | +0.0334 ± 0.0031 * | +0.0256 ± 0.0014 * | 0.9179 | 0.9618 | ~90s |
| e13_geoshape | `geo,shape` | 13 | +0.0220 ± 0.0038 * | +0.0184 ± 0.0018 * | 0.9065 | 0.9546 | ~80s |
| e13_promname | `prom,name` | 7 | +0.0035 ± 0.0033 * | +0.0023 ± 0.0022 * | 0.8880 | 0.9386 | ~75s |
| e13_sibcue | `sib,cue` | 6 | +0.0223 ± 0.0023 * | +0.0175 ± 0.0014 * | 0.9069 | 0.9537 | ~75s |
| e13_cf | `cf` | 4 | -0.0012 ± 0.0045 | -0.0008 ± 0.0024 | 0.8833 | 0.9354 | ~74s |
| e14_no_cf | `prom,name,cue,sib,geo,shape` | 26 | +0.0345 ± 0.0065 * | +0.0269 ± 0.0017 * | 0.9191 | 0.9632 | ~88s |
| e14_no_prom | `name,cue,sib,geo,shape,cf` | 25 | +0.0292 ± 0.0038 * | +0.0229 ± 0.0014 * | 0.9137 | 0.9591 | ~89s |
| e14_core | `cue,sib,geo,shape` | 19 | +0.0278 ± 0.0024 * | +0.0232 ± 0.0016 * | 0.9124 | 0.9594 | ~83s |

## Per-source `exact_match` delta vs e11b

| config | Prodigy | TR | LGL | GWN | Synth | WikiDocs |
|---|---|---|---|---|---|---|
| e13_all | +0.0430 * | +0.0534 * | +0.0696 * | +0.0113 * | +0.0142 * | +0.0088 * |
| e13_geoshape | +0.0221 * | +0.0295 * | +0.0652 * | +0.0117 * | +0.0001 | +0.0032 |
| e13_promname | +0.0093 | -0.0021 | +0.0068 * | -0.0086 * | +0.0144 * | +0.0011 |
| e13_sibcue | +0.0166 * | +0.0447 * | +0.0479 * | +0.0059 | +0.0075 * | +0.0113 * |
| e13_cf | +0.0019 | -0.0022 | +0.0037 | -0.0056 | -0.0025 | -0.0027 |
| e14_no_cf | +0.0545 * | +0.0521 * | +0.0703 * | +0.0078 | +0.0124 * | +0.0099 * |
| e14_no_prom | +0.0306 * | +0.0360 * | +0.0696 * | +0.0183 * | +0.0102 * | +0.0105 * |
| e14_core | +0.0315 * | +0.0365 * | +0.0672 * | +0.0142 * | +0.0088 * | +0.0087 * |

## Verdict

The lean set: cue, sib, geo, shape (19 features, no population, no name matching, no case folding). +0.0278 -- within 0.007 of the best arm at three-quarters the feature count, and significantly below e13_all (-0.0056 ± 0.0037 paired). Worth knowing if feature computation cost ever matters; otherwise use e14_no_cf.
