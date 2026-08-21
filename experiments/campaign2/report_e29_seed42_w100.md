# Calibration and abstention report

checkpoints: experiments/e29_swa_ep15/seed42.pt
features:    ['prom', 'name', 'cue', 'sib', 'geo', 'shape'] (26 extra columns)
window:      100  (serving window)
temperature: 1.0  (raw model output)

## Entity groups

| group | what it is | N | share | EM | mean top-1 p | share p>0.9 |
|---|---|---|---|---|---|---|
| a_answerable | gold is in the scored window | 8749 | 97.46% | 92.75% | 0.910 | 78.55% |
| b_out_of_window | gold retrievable but past row 100 | 89 | 0.99% | 0.00% | 0.660 | 31.46% |
| c_unretrievable | gold not in the candidate list at all | 139 | 1.55% | 0.00% | 0.635 | 29.50% |

Exact match on group (a), pooled over entities, 92.75%; unweighted mean over the six sources -- the campaign's headline metric -- 92.77%; over every held-out entity, answerable or not, 90.40%.

## Calibration of the top-1 probability (group (a) only)

| source | N | accuracy | mean confidence | ECE | ECE (p_pred incl. reserved) |
|---|---|---|---|---|---|
| GWN | 455 | 93.63% | 0.921 | 0.0299 | 0.0346 |
| LGL | 916 | 91.70% | 0.877 | 0.0513 | 0.0502 |
| Prodigy | 500 | 89.60% | 0.899 | 0.0307 | 0.0409 |
| Synth | 298 | 97.32% | 0.942 | 0.0314 | 0.0432 |
| TR | 258 | 91.47% | 0.905 | 0.0393 | 0.0337 |
| WikiDocs | 6322 | 92.93% | 0.913 | 0.0186 | 0.0218 |
| **pooled** | 8749 | 92.75% | 0.910 | **0.0193** | 0.0249 |

Reliability (equal-mass bins, pooled):

| bin | N | p range | mean confidence | accuracy | gap |
|---|---|---|---|---|---|
| 1 | 584 | 0.115-0.611 | 0.470 | 0.500 | +0.030 |
| 2 | 584 | 0.612-0.787 | 0.706 | 0.733 | +0.027 |
| 3 | 584 | 0.787-0.887 | 0.843 | 0.832 | -0.011 |
| 4 | 584 | 0.887-0.932 | 0.912 | 0.933 | +0.021 |
| 5 | 583 | 0.932-0.951 | 0.943 | 0.983 | +0.040 |
| 6 | 583 | 0.951-0.960 | 0.956 | 0.973 | +0.017 |
| 7 | 583 | 0.960-0.966 | 0.963 | 0.993 | +0.030 |
| 8 | 583 | 0.966-0.970 | 0.968 | 0.993 | +0.025 |
| 9 | 583 | 0.970-0.975 | 0.973 | 0.997 | +0.024 |
| 10 | 583 | 0.975-0.978 | 0.977 | 0.991 | +0.015 |
| 11 | 583 | 0.978-0.982 | 0.980 | 0.998 | +0.018 |
| 12 | 583 | 0.982-0.985 | 0.983 | 1.000 | +0.017 |
| 13 | 583 | 0.985-0.988 | 0.987 | 0.993 | +0.007 |
| 14 | 583 | 0.988-0.992 | 0.990 | 0.998 | +0.008 |
| 15 | 583 | 0.992-1.000 | 0.996 | 0.997 | +0.001 |

## Temperature scaling

Global T (in-sample, all six sources): **0.874** on the selectable-rows softmax, 0.904 on the reserved-row-inclusive one.

| held-out source | T fit on the other five | ECE before | ECE after | accuracy | conf before | conf after |
|---|---|---|---|---|---|---|
| Prodigy | 0.874 | 0.0307 | 0.0387 | 89.60% | 0.899 | 0.924 |
| TR | 0.874 | 0.0393 | 0.0347 | 91.47% | 0.905 | 0.928 |
| LGL | 0.874 | 0.0513 | 0.0297 | 91.70% | 0.877 | 0.904 |
| GWN | 0.874 | 0.0299 | 0.0265 | 93.63% | 0.921 | 0.942 |
| Synth | 0.874 | 0.0314 | 0.0126 | 97.32% | 0.942 | 0.961 |
| WikiDocs | 0.874 | 0.0186 | 0.0144 | 92.93% | 0.913 | 0.938 |
| **mean** | | 0.0335 | 0.0261 | | | |
| **N-weighted** | | 0.0244 | 0.0185 | | | |

## Abstention scores

AUROC for flagging an answer that is wrong -- including the entities where nothing in the window could have been right.

| score | all entities | group (a) only | detects unanswerable |
|---|---|---|---|
| p_top1 | 0.8903 | 0.9151 | 0.7961 |
| margin | 0.8913 | 0.9168 | 0.7914 |
| logit_margin | 0.8893 | 0.9141 | 0.7881 |
| entropy | 0.8757 | 0.8979 | 0.7940 |
| p_pred_full | 0.8993 | 0.9164 | 0.8281 |
| p_reserved | 0.7460 | 0.6983 | 0.8689 |
| p_nullrow | 0.6212 | 0.5646 | 0.8551 |
| **combo (logistic, LOSO)** | 0.9023 | 0.9187 | 0.8356 |

Combination features: p_top1, margin, logit_margin, entropy, p_reserved.

### The reserved row as an explicit flag

- The reserved 'no correct answer' row wins the raw argmax on 1.51% of entities (136 of 8977).
- Of those, 60.29% have no answerable gold at all (base rate 2.54%).
- Answers where it fires are right 13.97% of the time, vs 91.57% where it does not.
- On the 4272 entities whose candidate list fills the window, the reserved row is *selectable*: it won 67 times, and the geoparser then returns the real candidate the sentinel overwrote -- right 0.00% of the time.

`p_reserved` read as 'this mention has no right answer here':

| p_reserved | N | share unanswerable | share wrong |
|---|---|---|---|
| 0.00-0.01 | 7049 | 0.74% | 5.28% |
| 0.01-0.05 | 1446 | 3.25% | 17.08% |
| 0.05-0.20 | 299 | 14.38% | 38.46% |
| 0.20-0.50 | 104 | 29.81% | 55.77% |
| 0.50-1.01 | 79 | 69.62% | 88.61% |

### How wrong the errors are, by confidence

Distance from the gold place to the predicted one, over the errors in group (a). A granularity slip (city vs the identically named county) is ~0 km; picking the wrong Denver is ~1,000 km.

| top-1 p | errors | median km | share <=161 km | share >1000 km |
|---|---|---|---|---|
| 0.00-0.50 | 179 | 50.6 | 54.75% | 30.17% |
| 0.50-0.80 | 278 | 29.2 | 67.27% | 23.02% |
| 0.80-0.90 | 99 | 35.6 | 71.72% | 25.25% |
| 0.90-0.99 | 75 | 23.5 | 70.67% | 17.33% |
| 0.99-1.01 | 3 | 4.2 | 100.00% | 0.00% |
| **all** | 634 | 30.6 | 64.98% | 24.61% |

## Risk-coverage (selective prediction)

Coverage = share of mentions answered, after abstaining on the least confident. Selective EM counts an answer on an unanswerable mention as wrong, which is what a user experiences.

| score | 100% | 95% | 90% | 80% | 70% | 60% | 50% | AURC |
|---|---|---|---|---|---|---|---|---|
| p_top1 | 90.40% | 93.25% | 95.27% | 97.51% | 98.47% | 98.74% | 98.91% | 0.0222 |
| margin | 90.40% | 93.11% | 95.07% | 97.42% | 98.49% | 98.96% | 99.02% | 0.0218 |
| p_pred_full | 90.40% | 93.32% | 95.35% | 97.61% | 98.66% | 98.98% | 99.11% | 0.0210 |
| combo | 90.40% | 93.35% | 95.28% | 97.54% | 98.52% | 98.96% | 99.24% | 0.0198 |

Thresholds on `p_pred_full`, the recommended score:

| coverage | p_pred_full cutoff | selective EM | unanswerable mentions caught |
|---|---|---|---|
| 100.00% | 0.009 | 90.40% | 0 of 228 |
| 95.00% | 0.505 | 93.32% | 95 of 228 |
| 90.00% | 0.659 | 95.35% | 131 of 228 |
| 80.00% | 0.861 | 97.61% | 161 of 228 |
| 70.00% | 0.932 | 98.66% | 185 of 228 |
| 60.00% | 0.953 | 98.98% | 198 of 228 |
| 50.00% | 0.963 | 99.11% | 204 of 228 |

## Serving policies

`flagged` = the mention the geoparser would mark low-confidence or refuse. Precision is the share of flagged mentions that really were wrong; recall is the share of all wrong answers that got flagged.

| policy | flagged | coverage | selective EM | flag precision | flag recall | unanswerable caught |
|---|---|---|---|---|---|---|
| geoparse.py today: reserved argmax or last-scored-row argmax | 136 | 98.49% | 91.57% | 86.03% | 13.57% | 82 of 228 |
| reserved row wins the raw argmax | 136 | 98.49% | 91.57% | 86.03% | 13.57% | 82 of 228 |
| p_pred_full < 0.5 | 438 | 95.12% | 93.25% | 65.30% | 33.18% | 94 of 228 |
| p_pred_full < 0.7 | 1038 | 88.44% | 95.72% | 50.29% | 60.56% | 138 of 228 |
| p_pred_full < 0.8 | 1430 | 84.07% | 96.83% | 43.57% | 72.27% | 148 of 228 |
| p_pred_full < 0.9 | 2174 | 75.78% | 98.19% | 33.99% | 85.73% | 170 of 228 |
| p_pred_full < 0.7 or reserved argmax | 1048 | 88.33% | 95.84% | 50.76% | 61.72% | 147 of 228 |

wrote experiments/campaign2/preds/e29_seed42_w100.parquet (8977 rows)
wrote experiments/campaign2/preds/e29_seed42_w100.json
