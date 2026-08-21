# Calibration and abstention report

checkpoints: experiments/e29_swa_ep15/seed101.pt, experiments/e29_swa_ep15/seed1848.pt, experiments/e29_swa_ep15/seed202.pt, experiments/e29_swa_ep15/seed42.pt, experiments/e29_swa_ep15/seed617.pt
features:    ['prom', 'name', 'cue', 'sib', 'geo', 'shape'] (26 extra columns)
window:      100  (serving window)
temperature: 1.0  (raw model output)

## Entity groups

| group | what it is | N | share | EM | mean top-1 p | share p>0.9 |
|---|---|---|---|---|---|---|
| a_answerable | gold is in the scored window | 8749 | 97.46% | 93.46% | 0.901 | 76.42% |
| b_out_of_window | gold retrievable but past row 100 | 89 | 0.99% | 0.00% | 0.631 | 24.72% |
| c_unretrievable | gold not in the candidate list at all | 139 | 1.55% | 0.00% | 0.610 | 21.58% |

Exact match on group (a), pooled over entities, 93.46%; unweighted mean over the six sources -- the campaign's headline metric -- 93.98%; over every held-out entity, answerable or not, 91.09%.

## Calibration of the top-1 probability (group (a) only)

| source | N | accuracy | mean confidence | ECE | ECE (p_pred incl. reserved) |
|---|---|---|---|---|---|
| GWN | 455 | 95.16% | 0.918 | 0.0377 | 0.0441 |
| LGL | 916 | 92.14% | 0.870 | 0.0515 | 0.0586 |
| Prodigy | 500 | 92.20% | 0.886 | 0.0398 | 0.0659 |
| Synth | 298 | 98.32% | 0.935 | 0.0483 | 0.0581 |
| TR | 258 | 92.64% | 0.895 | 0.0574 | 0.0595 |
| WikiDocs | 6322 | 93.44% | 0.905 | 0.0348 | 0.0393 |
| **pooled** | 8749 | 93.46% | 0.901 | **0.0332** | 0.0405 |

Reliability (equal-mass bins, pooled):

| bin | N | p range | mean confidence | accuracy | gap |
|---|---|---|---|---|---|
| 1 | 584 | 0.173-0.594 | 0.468 | 0.481 | +0.014 |
| 2 | 584 | 0.594-0.765 | 0.684 | 0.772 | +0.088 |
| 3 | 584 | 0.766-0.866 | 0.821 | 0.889 | +0.068 |
| 4 | 584 | 0.866-0.919 | 0.896 | 0.952 | +0.056 |
| 5 | 583 | 0.919-0.941 | 0.931 | 0.967 | +0.037 |
| 6 | 583 | 0.941-0.952 | 0.947 | 0.978 | +0.031 |
| 7 | 583 | 0.952-0.959 | 0.956 | 1.000 | +0.044 |
| 8 | 583 | 0.959-0.965 | 0.962 | 1.000 | +0.038 |
| 9 | 583 | 0.965-0.969 | 0.967 | 0.995 | +0.028 |
| 10 | 583 | 0.969-0.973 | 0.971 | 0.998 | +0.027 |
| 11 | 583 | 0.973-0.978 | 0.976 | 0.997 | +0.021 |
| 12 | 583 | 0.978-0.982 | 0.980 | 0.997 | +0.017 |
| 13 | 583 | 0.982-0.985 | 0.983 | 1.000 | +0.017 |
| 14 | 583 | 0.985-0.990 | 0.988 | 0.995 | +0.007 |
| 15 | 583 | 0.990-1.000 | 0.994 | 1.000 | +0.006 |

## Temperature scaling

Global T (in-sample, all six sources): **0.791** on the selectable-rows softmax, 0.818 on the reserved-row-inclusive one.

| held-out source | T fit on the other five | ECE before | ECE after | accuracy | conf before | conf after |
|---|---|---|---|---|---|---|
| Prodigy | 0.791 | 0.0398 | 0.0211 | 92.20% | 0.886 | 0.924 |
| TR | 0.791 | 0.0574 | 0.0287 | 92.64% | 0.895 | 0.932 |
| LGL | 0.791 | 0.0515 | 0.0222 | 92.14% | 0.870 | 0.911 |
| GWN | 0.791 | 0.0377 | 0.0249 | 95.16% | 0.918 | 0.948 |
| Synth | 0.791 | 0.0483 | 0.0227 | 98.32% | 0.935 | 0.961 |
| WikiDocs | 0.791 | 0.0348 | 0.0120 | 93.44% | 0.905 | 0.942 |
| **mean** | | 0.0449 | 0.0219 | | | |
| **N-weighted** | | 0.0381 | 0.0151 | | | |

## Abstention scores

AUROC for flagging an answer that is wrong -- including the entities where nothing in the window could have been right.

| score | all entities | group (a) only | detects unanswerable |
|---|---|---|---|
| p_top1 | 0.9045 | 0.9278 | 0.8220 |
| margin | 0.9029 | 0.9269 | 0.8154 |
| logit_margin | 0.8961 | 0.9172 | 0.8138 |
| entropy | 0.8901 | 0.9105 | 0.8199 |
| p_pred_full | 0.9103 | 0.9272 | 0.8466 |
| p_reserved | 0.7697 | 0.7231 | 0.8784 |
| p_nullrow | 0.6876 | 0.6310 | 0.8953 |
| vote_top_frac | 0.7168 | 0.7355 | 0.6546 |
| vote_n_distinct | 0.7161 | 0.7346 | 0.6546 |
| p_top1_std | 0.8695 | 0.8885 | 0.7981 |
| **combo (logistic, LOSO)** | 0.9135 | 0.9256 | 0.8661 |

Combination features: p_top1, margin, logit_margin, entropy, p_reserved, vote_top_frac, p_top1_std.

### The reserved row as an explicit flag

- The reserved 'no correct answer' row wins the raw argmax on 1.46% of entities (131 of 8977).
- Of those, 67.94% have no answerable gold at all (base rate 2.54%).
- Answers where it fires are right 10.69% of the time, vs 92.28% where it does not.
- On the 4272 entities whose candidate list fills the window, the reserved row is *selectable*: it won 71 times, and the geoparser then returns the real candidate the sentinel overwrote -- right 0.00% of the time.

`p_reserved` read as 'this mention has no right answer here':

| p_reserved | N | share unanswerable | share wrong |
|---|---|---|---|
| 0.00-0.01 | 6634 | 0.57% | 4.45% |
| 0.01-0.05 | 1794 | 2.90% | 13.88% |
| 0.05-0.20 | 369 | 10.84% | 33.06% |
| 0.20-0.50 | 103 | 34.95% | 61.17% |
| 0.50-1.01 | 77 | 80.52% | 92.21% |

### How wrong the errors are, by confidence

Distance from the gold place to the predicted one, over the errors in group (a). A granularity slip (city vs the identically named county) is ~0 km; picking the wrong Denver is ~1,000 km.

| top-1 p | errors | median km | share <=161 km | share >1000 km |
|---|---|---|---|---|
| 0.00-0.50 | 172 | 112.6 | 51.74% | 29.07% |
| 0.50-0.80 | 285 | 23.1 | 71.93% | 21.40% |
| 0.80-0.90 | 65 | 11.4 | 70.77% | 16.92% |
| 0.90-0.99 | 49 | 7.0 | 75.51% | 18.37% |
| 0.99-1.01 | 1 | 0.5 | 100.00% | 0.00% |
| **all** | 572 | 27.8 | 66.08% | 22.90% |

## Risk-coverage (selective prediction)

Coverage = share of mentions answered, after abstaining on the least confident. Selective EM counts an answer on an unanswerable mention as wrong, which is what a user experiences.

| score | 100% | 95% | 90% | 80% | 70% | 60% | 50% | AURC |
|---|---|---|---|---|---|---|---|---|
| p_top1 | 91.09% | 93.97% | 96.14% | 97.99% | 98.84% | 99.18% | 99.18% | 0.0174 |
| margin | 91.09% | 93.69% | 95.98% | 97.98% | 98.76% | 99.11% | 99.26% | 0.0171 |
| p_pred_full | 91.09% | 94.00% | 96.20% | 98.11% | 98.87% | 99.22% | 99.33% | 0.0163 |
| combo | 91.09% | 94.07% | 96.25% | 98.11% | 98.76% | 99.13% | 99.40% | 0.0148 |

Thresholds on `p_pred_full`, the recommended score:

| coverage | p_pred_full cutoff | selective EM | unanswerable mentions caught |
|---|---|---|---|
| 100.00% | 0.011 | 91.09% | 0 of 228 |
| 95.00% | 0.500 | 94.00% | 104 of 228 |
| 90.00% | 0.638 | 96.20% | 140 of 228 |
| 80.00% | 0.835 | 98.11% | 171 of 228 |
| 70.00% | 0.918 | 98.87% | 186 of 228 |
| 60.00% | 0.944 | 99.22% | 198 of 228 |
| 50.00% | 0.956 | 99.33% | 208 of 228 |

## Serving policies

`flagged` = the mention the geoparser would mark low-confidence or refuse. Precision is the share of flagged mentions that really were wrong; recall is the share of all wrong answers that got flagged.

| policy | flagged | coverage | selective EM | flag precision | flag recall | unanswerable caught |
|---|---|---|---|---|---|---|
| geoparse.py today: reserved argmax or last-scored-row argmax | 131 | 98.54% | 92.28% | 89.31% | 14.62% | 89 of 228 |
| reserved row wins the raw argmax | 131 | 98.54% | 92.28% | 89.31% | 14.62% | 89 of 228 |
| p_pred_full < 0.5 | 449 | 95.00% | 94.00% | 64.14% | 36.00% | 104 of 228 |
| p_pred_full < 0.7 | 1122 | 87.50% | 96.87% | 49.38% | 69.25% | 149 of 228 |
| p_pred_full < 0.8 | 1567 | 82.54% | 97.73% | 40.33% | 79.00% | 164 of 228 |
| p_pred_full < 0.9 | 2399 | 73.28% | 98.60% | 29.51% | 88.50% | 180 of 228 |
| p_pred_full < 0.7 or reserved argmax | 1137 | 87.33% | 97.05% | 50.04% | 71.12% | 161 of 228 |

wrote experiments/campaign2/preds/e29_ens5_w100.parquet (8977 rows)
wrote experiments/campaign2/preds/e29_ens5_w100.json
