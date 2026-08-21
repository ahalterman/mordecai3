# Calibration and abstention (second campaign, item 5)

Written 2026-08-20, updated the same day with the confirmed 5-seed results of
the two training arms this analysis proposed (§9). Model under test: the ship
recipe `e29_swa_ep15`, seeds {42, 101, 202, 617, 1848}. Tool:
`tools/calibration_eval.py` (reusable; every table below is one of its
sections). Per-entity dumps: `experiments/campaign2/preds/*.parquet`.

Everything is reported at the **serving window** (`max_choices=100`, what
`Geoparser` actually runs) unless a row says 500. Numbers quoted without a
seed are seed 42; the 5-seed spread is in §7.

Ledger entries produced by this track: `experiments/e30_absw100/` and
`experiments/e31_abstain_w3/` (5 seeds each, both rejected). `tools/train.py`
gained `--window` and `--abstain-weight`, both verified byte-identical no-ops
at their defaults.

---

## 1. Headline

1. **The abstention class already exists and is already trained.**
   `TrainData.create_labels` points at the reserved last row whenever nothing
   in the candidate list is correct — 508 of 20,951 training entities (2.42%).
   And `mordecai3/geoparse.py` already acts on it: `if pred[-1] == pred.max()`
   returns a blank result. It fires on **1.51% of mentions** (5-seed range
   1.25–1.80%) and is **86% precise**: those answers are right 14% of the time
   versus 91.6% elsewhere, and 60% of them are mentions whose gold is not
   reachable at all (base rate 2.5%). The campaign's exact-match metric never
   saw any of this, because `evaluate_results` argmaxes over the candidate
   rows only.
2. **The PI's anecdote is half right, and the half that is wrong is the
   reassuring half.** Among mentions the model *could* have gotten right,
   confident errors are *closer*, not wilder: errors with p>0.9 are 71% within
   161 km and 17% beyond 1,000 km, while errors with p<0.5 are 55% within
   161 km and 30% beyond 1,000 km. Confidently-and-wildly wrong is 20
   mentions in 8,977 (0.22%). The anecdote's real referent is the 2.54% of
   mentions where **no answer in the window is right by construction** — and
   there the model answers with p>0.9 on 30% of them. Confidence *is* useful,
   but it is not free of confident nonsense on unanswerable inputs.
3. **Calibration is good and gets better for free.** The model is
   systematically *under*-confident (label smoothing 0.05 + SWA): mean
   confidence 0.910 against 92.8% accuracy. Pooled ECE 0.019; leave-one-source-out
   temperature scaling with **T = 0.874** takes the mean per-source ECE from
   0.0335 to 0.0261 and the pooled ECE to **0.0115**. T is identical to grid
   resolution across all six LOSO folds.
4. **Best single abstention score: `p_pred_full`** — the probability of the
   chosen candidate under a softmax that *includes the reserved row*.
   AUROC 0.899 for "this answer is wrong" over all mentions (0.916 restricted
   to answerable ones), against 0.890 for the plain top-1 probability, and it
   is much better at spotting unanswerable mentions (0.828 vs 0.796). It costs
   one extra row in a softmax the model already computed.
5. **Selective accuracy**: at 90% coverage selective EM is **95.2%** (from
   90.4% at full coverage); at 80% coverage, 97.5%. With the 5-seed ensemble,
   96.2% and 98.1%.
6. **Two training arms were run on 5 seeds each and both are rejected** (§9).
   Training at the serving window (`e30_absw100`) does make a much better
   unanswerable detector — AUROC 0.858 → 0.931, catching 117 of 228 unanswerable
   mentions instead of 85, at higher precision — but it costs a confirmed
   −0.0057 macro EM, and the two cancel: selective EM at 90% coverage and AURC
   are statistically unchanged. Upweighting the abstention class at window 500
   (`e31_abstain_w3`) is worse on every axis. **The abstention win is a decode-
   time win, not a training win**: ship `e29_swa_ep15` with the recipe in §8.

---

## 2. What the model actually emits

Read this before any number below; the slot layout is the whole story.

* `es_choices` = the ES hits **plus a "none of the above" NULL row appended
  last** (`geoparse.py::_null_choice`). A mention with 100 hits has 101 entries.
* `ProductionData` lays the first `max_choices` of them into rows and then
  **overwrites the last row** with a reserved sentinel: feature code 53,
  country NULL, every gazetteer feature −1. The mask keeps that row live
  always, padding rows dead.
* `TrainData.create_labels` targets that reserved row when `sum(correct) == 0`.
  **So the reserved row is a trained "no correct answer" class**, and the
  gazetteer NULL row is not: it is never the gold, so training only ever
  pushes it down. Measured: the NULL row never once wins the visible argmax on
  8,977 held-out mentions, and its probability is the weakest score in the
  table (§5). The abstention signal lives entirely in the reserved row.
* Where the reserved row sits relative to the scorer:
  * candidate list **shorter** than the window → the reserved row is at an
    index past the end of `es_choices`, so `evaluate_results` and
    `geoparse.py`'s `results` list never rank it. Only the separate
    `pred[-1] == pred.max()` check in `geoparse.py` sees it.
  * candidate list **fills** the window (48% of held-out mentions at window
    100, 20% at window 500) → the sentinel occupies an index the scorer does
    read, and its score is attributed to *the real candidate the sentinel
    overwrote* (audit item B2). At window 100 that happens 67 times and is
    right 0 times out of 67. In serving this is harmless — `pred[-1] ==
    pred.max()` catches those mentions first and returns blank — but in
    `evaluate_results` they are scored as ordinary wrong answers.
* Two abstention branches in `geoparse.py` behave differently and should be
  reconciled: `pred[-1] == pred.max()` appends a result dict with no geo
  fields, while `argmax(scores) == len(scores)-1` drops the mention silently.
  The second branch never fires on this held-out set.

### 2a. Three components, three conventions for the same row

The experiment auditor is right that train, eval and serving each treat the
reserved row differently, and the disagreement is not cosmetic — it is why a
working abstention mechanism was invisible for the whole first campaign:

| component | convention | consequence |
|---|---|---|
| **training** (`TrainData.create_labels`) | the reserved row at index `max_choices-1` is the target class when nothing is correct | the model learns an abstention class on 2.42% of examples |
| **evaluation** (`error_utils.evaluate_results`) | scores only rows `< len(es_choices)`; the reserved row is unreachable on a short list, and on a full list its score is credited to the real candidate it overwrote | the frozen exact match has never measured abstention, and mis-scores 67 mentions per seed at window 100 |
| **serving** (`geoparse.py`) | `pred[-1] == pred.max()` → blank result; separately `argmax(scores) == len(scores)-1` → mention dropped | the shipped model abstains on 1.5% of mentions, silently, with no confidence attached |

**Which convention this report uses.** Every number here is computed on the
*serving* convention with one repair: the argmax is taken over the rows the
scorer can rank (`sel_mask`), the reserved row is kept in the softmax
denominator (`full_mask`), and a mention whose gold is unreachable counts as
wrong rather than being dropped from the denominator. Where a table says
"group (a)" it is restricted to answerable mentions and then matches
`evaluate_results` exactly — verified: at window 500 this tool reproduces
`e29_swa_ep15/seed42.json`'s per-source exact match to four decimals and its
headline to 92.04%.

**The unified convention to adopt** (recommended fix, all three components):

1. the reserved row is **always** the last row of the scored window, is
   **always** in the softmax denominator, and is **never** identified with a
   candidate — `choices_window` should reserve room for it (the existing
   `--full-null-row` flag does exactly this) so no real candidate is ever
   overwritten and no candidate id is ever attributed to a sentinel score;
2. `argmax` over candidates decides *which* place; `p(reserved)` decides
   *whether to answer*, exposed as a number rather than an implicit branch;
3. evaluation reports both: exact match on answerable mentions (ledger
   continuity) **and** exact match over all mentions with abstention counted
   as an explicit third outcome, so a change that trades answers for
   abstentions is visible instead of invisible.

---

## 3. The three groups at the serving window

| group | what it is | N | share | EM | mean top-1 p | share p>0.9 |
|---|---|---|---|---|---|---|
| (a) answerable | gold is in the scored window | 8749 | 97.46% | 92.75% | 0.910 | 78.6% |
| (b) out of window | gold retrievable at 500, past row 100 | 89 | 0.99% | 0.00% | 0.660 | 31.5% |
| (c) unretrievable | gold not in the candidate list at all | 139 | 1.55% | 0.00% | 0.635 | 29.5% |

The campaign's frozen exact match is group (a) only: 92.75% pooled over
entities, 92.77% as the unweighted source mean. Over every mention a user
actually submits it is **90.40%** — the 2.54% of unanswerable mentions are
2.35 points of silent error that no campaign table has ever shown. Per source,
unanswerable share: LGL 5.9%, TR 5.8%, GWN 4.0%, WikiDocs 2.1%, Synth 0.7%,
Prodigy 0.0%.

(Pipeline check: at window 500 this tool reproduces the frozen headline of
`experiments/e29_swa_ep15/seed42.json` exactly — unweighted source mean
92.04%, and every per-source exact match to four decimals.)

At window 500 group (b) is empty (the gold is never past row 499), which is
why this decomposition only becomes visible when you evaluate what is served.

---

## 4. Is the model confidently, wildly wrong?

Distance from gold to prediction, over the errors in group (a), seed 42:

| top-1 p | errors | median km | share ≤161 km | share >1000 km |
|---|---|---|---|---|
| 0.00–0.50 | 179 | 50.6 | 54.8% | 30.2% |
| 0.50–0.80 | 278 | 29.2 | 67.3% | 23.0% |
| 0.80–0.90 | 99 | 35.6 | 71.7% | 25.3% |
| 0.90–0.99 | 75 | 23.5 | 70.7% | 17.3% |
| 0.99–1.00 | 3 | 4.2 | 100.0% | 0.0% |
| **all** | 634 | 30.6 | 65.0% | 24.6% |

Confidence and wildness move in the *right* direction: the far errors
concentrate at low confidence. "Confidently and wildly wrong" (p>0.9 and
>1000 km) is 20 mentions of 8,977.

The anecdote's true case is the unanswerable groups. On the 228 mentions with no
usable gold, the model still returns an answer with p>0.9 on 69 of them (30%),
p>0.99 on 12. For group (b) specifically — the gold exists but sits past row
100 — the median miss is 583 km and 40% are beyond 1,000 km. So: *when the
right answer is absent, the model does produce confident far-away answers*;
when the right answer is present, high confidence is trustworthy.

---

## 5. Calibration

Pooled over group (a), equal-mass 15 bins (equal-width bins are useless here:
four fifths of the mass sits above p = 0.9).

| | accuracy | mean confidence | ECE |
|---|---|---|---|
| raw, window 100 | 92.75% | 0.910 | 0.0193 |
| raw, window 500 | 92.39% | 0.889 | 0.0352 |
| **T = 0.874, window 100** | 92.75% | 0.935 | **0.0115** |

The reliability table is monotone under-confidence in the mid range: bin
[0.932, 0.951] has mean confidence 0.943 and accuracy 0.983; the top bin is
already exact (0.996 vs 0.997). Under-confidence is what a 0.05 label-smoothed,
weight-averaged model should look like.

Leave-one-source-out temperature (no dev set exists — see §9):

| held-out source | T from the other five | ECE before | ECE after |
|---|---|---|---|
| Prodigy | 0.874 | 0.0307 | 0.0387 |
| TR | 0.874 | 0.0393 | 0.0347 |
| LGL | 0.874 | 0.0513 | 0.0297 |
| GWN | 0.874 | 0.0299 | 0.0265 |
| Synth | 0.874 | 0.0314 | 0.0126 |
| WikiDocs | 0.874 | 0.0186 | 0.0144 |
| **mean** | | 0.0335 | **0.0261** |
| **N-weighted** | | 0.0244 | **0.0185** |

Improves 5 of 6 sources; Prodigy (already the least under-confident, and the
source that fights every recipe) gets slightly worse. T is stable to grid
resolution across folds, but WikiDocs is 72% of the fitting mass, so this is
close to "WikiDocs' temperature applied everywhere". Temperature scaling is
monotone: it changes probabilities and thresholds, never AUROC, never
risk-coverage, never exact match.

---

## 6. Abstention scores

AUROC for flagging an answer that is wrong, seed 42, window 100. "all
entities" counts an answer on an unanswerable mention as wrong (what a user
experiences); "group (a)" is pure ranking error; "unanswerable" is the OOD
question, gold-missing vs gold-present.

| score | all entities | group (a) | detects unanswerable |
|---|---|---|---|
| `p_pred_full` (softmax incl. reserved row) | **0.8993** | 0.9164 | 0.8281 |
| `margin` (p1 − p2) | 0.8913 | **0.9168** | 0.7914 |
| `p_top1` | 0.8903 | 0.9151 | 0.7961 |
| `logit_margin` | 0.8893 | 0.9141 | 0.7881 |
| `entropy` | 0.8757 | 0.8979 | 0.7940 |
| `p_reserved` | 0.7460 | 0.6983 | **0.8689** |
| `p_nullrow` (gazetteer NULL row) | 0.6212 | 0.5646 | 0.8551 |
| logistic combo, LOSO-fit | 0.9023 | 0.9187 | 0.8356 |

Read: one number does the ranking job (`p_pred_full`), a *different* number
does the OOD job (`p_reserved`), and a fitted combination of all of them buys
+0.003 AUROC over `p_pred_full` alone — not worth the machinery. Ship the two
scalars, not the model.

`p_reserved` as a usable "this mention has no answer here" probability:

| p_reserved | N | share unanswerable | share wrong |
|---|---|---|---|
| 0.00–0.01 | 7049 | 0.7% | 5.3% |
| 0.01–0.05 | 1446 | 3.3% | 17.1% |
| 0.05–0.20 | 299 | 14.4% | 38.5% |
| 0.20–0.50 | 104 | 29.8% | 55.8% |
| 0.50–1.00 | 79 | 69.6% | 88.6% |

**Risk-coverage** (selective EM over all mentions, unanswerable counted wrong):

| score | 100% | 95% | 90% | 80% | 70% | AURC |
|---|---|---|---|---|---|---|
| `p_top1` | 90.40% | 93.25% | 95.27% | 97.51% | 98.47% | 0.0222 |
| `margin` | 90.40% | 93.11% | 95.07% | 97.42% | 98.49% | 0.0218 |
| `p_pred_full` | 90.40% | 93.32% | **95.35%** | **97.61%** | 98.66% | **0.0210** |
| combo | 90.40% | 93.35% | 95.28% | 97.54% | 98.52% | 0.0198 |
| 5-seed ensemble, `p_pred_full` | 91.09% | 94.00% | **96.20%** | **98.11%** | 98.87% | **0.0163** |

**Serving policies** at T = 0.874 (thresholds are on the calibrated
probability; `flag precision` = share of flagged mentions that really were
wrong, `recall` = share of all wrong answers caught):

| policy | flagged | coverage | selective EM | precision | recall | unanswerable caught |
|---|---|---|---|---|---|---|
| reserved argmax (what ships today) | 136 | 98.49% | 91.57% | 86.0% | 13.6% | 82 of 228 |
| `p* < 0.5` | 323 | 96.40% | 92.57% | 67.8% | 25.4% | 77 of 228 |
| `p* < 0.7` | 827 | 90.79% | 95.01% | 55.0% | 52.8% | 120 of 228 |
| `p* < 0.7` or reserved argmax | 847 | 90.56% | **95.24%** | 56.1% | 55.1% | 135 of 228 |
| `p* < 0.8` | 1172 | 86.94% | 96.11% | 47.6% | 64.7% | 138 of 228 |
| `p* < 0.9` | 1687 | 81.21% | 97.38% | 39.8% | 77.8% | 153 of 228 |

---

## 7. Does SWA hurt calibration? Does the ensemble?

The only non-SWA checkpoints on disk are `e27_strip_nosw` (the strip-feature
arm without weight averaging) and its SWA twin `e27_strip_ep15` — same recipe,
same seeds, averaging the only difference. 5 paired seeds, window 100:

| metric | no SWA | SWA | paired Δ | 2 SE |
|---|---|---|---|---|
| EM, group (a) | 0.9184 | 0.9271 | +0.0087 | 0.0017 \* |
| ECE (raw) | 0.0156 | 0.0225 | +0.0069 | 0.0068 \* |
| fitted T | 0.881 | 0.857 | −0.0235 | 0.0340 |
| ECE after LOSO temperature | 0.0272 | 0.0225 | **−0.0047** | 0.0019 \* |
| AUROC `p_pred_full` | 0.8939 | 0.8985 | +0.0046 | 0.0064 |

So yes, the folk result reproduces but weakly and only *before* scaling: SWA
adds ~0.007 ECE (right at 2 SE) by making the model more under-confident, and
after a temperature it is the *better* calibrated of the two. Keep SWA; scale.

The 5-seed probability-averaged ensemble is more under-confident still (raw
ECE 0.0332, T = 0.791), and after scaling it is the best of everything
(N-weighted ECE 0.0151) as well as the best ranker (AURC 0.0163). Seed
disagreement as a score is disappointing: `vote_top_frac` AUROC 0.717 and
`p_top1_std` 0.870, both below the ensemble's own `p_pred_full` (0.910).
The ensemble's value is its sharper probability, not its variance.

5-seed spread of the single-model numbers (window 100):

| | mean | sd | range |
|---|---|---|---|
| EM group (a) | 0.9296 | 0.0017 | 0.9275–0.9316 |
| EM all mentions | 0.9060 | 0.0017 | 0.9040–0.9080 |
| ECE (raw) | 0.0230 | 0.0025 | 0.0193–0.0262 |
| ECE after LOSO T | 0.0238 | 0.0024 | 0.0210–0.0263 |
| AUROC `p_pred_full` | 0.8972 | 0.0040 | 0.8919–0.9026 |
| AUROC `p_reserved` (unanswerable) | 0.8582 | 0.0149 | 0.8329–0.8689 |
| reserved-argmax fire rate | 0.0151 | 0.0021 | 0.0125–0.0180 |

---

## 8. Recommended serving recipe (final)

Nothing here needs retraining — and §9 shows that retraining does not help.

1. **Keep the reserved-row abstention** (`pred[-1] == pred.max()`), and make
   its two branches consistent: return an explicit `"no_match": True` result
   rather than a bare dict in one branch and a dropped mention in the other.
   Adopt the unified reserved-row convention in §2a while you are in there.
2. **Expose a calibrated probability.** Today `best['score']` is a raw logit,
   which is meaningless to a user. Replace it with
   `p = softmax(logits[live rows + reserved row] / T)[chosen]`, i.e.
   `p_pred_full` at **T = 0.874** (0.79 if serving the 5-seed ensemble).
   Expected calibration: pooled ECE 0.012, accuracy within 2 points of the
   stated probability in every equal-mass bin.
3. **Expose `p_reserved`** alongside it as `p_no_match`: the model's answer to
   "is this mention findable at all", AUROC 0.86 for exactly that (0.93 if you
   serve the e30 checkpoint — see §9). A mention scoring `p_reserved > 0.5` is
   unanswerable 70% of the time.
4. **Default filter for researchers**: keep answers with `p ≥ 0.7` and no
   reserved-argmax flag.
5. If three trainings are affordable, serve the **probability-averaged
   ensemble**; it is better on every axis here, at 3x training cost.
6. Document that the frozen exact match (group (a) only) is ~2.3 points above
   what a user sees end-to-end on resolution, because the metric silently drops
   mentions whose gold is unreachable.

Operating points, `e29_swa_ep15` seed 42 at T = 0.874, serving window
(coverage = share of mentions answered; selective EM counts an answer on an
unanswerable mention as wrong):

| policy | coverage | selective EM | unanswerable caught (of 228) |
|---|---|---|---|
| answer everything | 100% | 90.40% | 0 |
| reserved-argmax flag only (ships today) | 98.5% | 91.57% | 82 |
| **`p ≥ 0.7` or flag (recommended default)** | **90.6%** | **95.24%** | 135 |
| `p ≥ 0.8` | 86.9% | 96.11% | 138 |
| `p ≥ 0.9` (precision-first) | 81.2% | 97.38% | 153 |
| 5-seed ensemble, `p ≥ 0.7`-equivalent (90% coverage) | 90.0% | 96.20% | — |

5-seed mean of the recommended default on `e29_swa_ep15`: selective EM at 90%
coverage **95.56% ± 0.21** (sd over seeds), AURC 0.0197 ± 0.0009.

---

## 9. Training experiments (both run, both rejected)

Full ledger entries: `experiments/e30_absw100/NOTES.md`,
`experiments/e31_abstain_w3/NOTES.md`. 5 seeds each, paired per-seed deltas vs
`e29_swa_ep15`, `t(4)` against the auditor's 2.776 critical value (the old
|mean| > 2 SE rule is reported alongside; it would have called three of these
effects that the t-test does not).

`tools/train.py` gained two flags, both verified no-ops at their defaults: a
rerun of the e29 recipe with the edited file reproduces
`experiments/e29_swa_ep15/seed42.json` **byte for byte**
(md5 `1d84e9529c77db0ef5b9c639d2dff96f`).
* `--window N` separates the scoring window from `--max-choices`, which names
  the pickle files. Requires `--full-null-row` when smaller, because otherwise
  a gold outside the window has no label to point at (`IndexError`).
* `--abstain-weight w` multiplies the loss of examples whose target is the
  reserved row.

### e30_absw100 — train at the serving window (100), out-of-window golds = abstain

The mechanism worked; the trade did not.

| metric (serving window) | e29 | e30 | Δ | t(4) |
|---|---|---|---|---|
| AUROC `p_reserved`, detects unanswerable | 0.8582 | **0.9308** | +0.0727 | +7.52 \* |
| unanswerable caught by the flag (of 228) | 84.6 | **117.2** | +32.6 | +12.27 \* |
| flag precision | 0.8828 | **0.9021** | +0.0193 | +4.21 \* |
| pooled EM, answerable | 0.9296 | 0.9260 | −0.0035 | −4.50 \* |
| macro EM, 6 sources, frozen key (window 500) | 0.9258 | 0.9201 | −0.0057 | −2.78 \* |
| TLG-hard, answerable | 0.8915 | 0.8824 | −0.0092 | −7.05 \* |
| **selective EM @ 90% coverage** | 0.9556 | 0.9535 | −0.0021 | −1.95 ns |
| **AURC (lower better)** | 0.0197 | 0.0209 | +0.0013 | +0.94 ns |
| ECE raw / after temperature | 0.0230 / 0.0238 | 0.0373 / 0.0248 | +0.0142 / +0.0010 | +7.40 \* / +0.40 ns |

Giving the reserved row training signal for "the gold is past the window"
produces a decisively better unanswerable detector. But ranking accuracy falls
by a small, statistically confirmed amount everywhere (the likely mechanism:
a 100-row softmax carries one fifth as many negatives per example), and the two
effects land on opposite sides of the same product metric. On the metric that
contains both — selective EM at fixed coverage, and AURC — **e30 is
indistinguishable from e29**. It relocates error rather than removing it.

The pre-declared guard was "answerable EM must not fall by more than seed noise
(±0.010)". Numerically it passes (−0.0035); under the paired test the
regression is real and systematic. The stricter reading is the right one, and
it is the reason to reject: nothing is bought.

Keep the checkpoints. If a deployment's product requirement is specifically
"flag mentions that cannot be geolocated" rather than "answer more mentions
correctly", e30 is the better detector at a known cost of ~0.4–0.6 EM.

### e31_abstain_w3 — upweight the abstention class 3x at window 500

Dominated on every axis: it buys a third of e30's OOD gain (AUROC +0.0263) at
*lower* flag precision (−0.042), and costs confirmed EM (−0.0033 macro),
confirmed wrong-answer AUROC (`p_pred_full` −0.0100), and confirmed selective
EM at 90% coverage (−0.0056, t = −5.44). Weighting cannot teach what the labels
do not contain: at window 500 the "gold past row 100" mentions are still
labelled in-window, so the arm's ceiling on the serving-time OOD problem is
structurally lower than e30's, while the reallocation of gradient toward 2.4%
of examples blunts the ranker. No weight sweep is recommended.

### What these two arms establish

The abstention gain available from training is real but is paid for out of
ranking accuracy, at roughly one-for-one on the product metric. The decode-time
recipe in §8 is therefore the whole of the recommendation, and the remaining
headroom on abstention is architectural — a separate abstention head, scored
against the answer distribution rather than competing with it inside the same
softmax — not a matter of labels or loss weights on this design.

---

## 10. Caveats

* **No dev set.** The 30% held-out half is what every campaign number is
  measured on, so a temperature fit on it is in-sample. This report fits T
  leave-one-source-out and reports it on the untouched sixth source; the
  pooled "global T" is quoted only as a reference. Thresholds in §8 are
  chosen on the same data and should be re-checked on the first real corpus
  the geoparser runs on — they are ordinary quantiles of a well-behaved
  score, not tuned parameters, so the risk is small but not zero.
* `Synth` is 17.9% unanswerable in training and is a geometry cheat sheet;
  it contributes a quarter of the abstention class. Per-source numbers matter.
* Distances for group (c) cannot be computed — the gold is not in the
  candidate list, so it has no coordinates here. §4's distance table is
  group (a), and group (b) is quoted separately.
* End-to-end abstention (spaCy NER misses) is invisible here, as everywhere
  else in this campaign; see second-campaign item 6.

## Artifacts

* `experiments/campaign2/report_e29_seed42_w100.md` — full tool output, single
  checkpoint, serving window (the source of most tables above).
* `experiments/campaign2/report_e29_ens5_w100.md` — same for the 5-seed
  probability-averaged ensemble.
* `experiments/campaign2/preds/e29_{seed42,ens5}_w100.{parquet,json}` — one row
  per held-out entity (8,977): prediction, correctness, distance from gold,
  every confidence score, group label. Probabilities are raw (T = 1).
* `experiments/e30_absw100/`, `experiments/e31_abstain_w3/` — the two rejected
  training arms: 5 seeds each of `seedS.{json,log,pt,pt.json}` plus NOTES.md
  with the full paired-delta tables on both keys.
* `tools/train.py` — `--window N` (scoring window, decoupled from the pickle
  name `--max-choices`; needs `--full-null-row` when smaller) and
  `--abstain-weight w` (loss weight on the reserved-row class). Both default
  to current behavior and are verified no-ops.

## Reproduce

```
# per-entity table + full report, single checkpoint at the serving window
uv run python tools/calibration_eval.py \
  --checkpoints experiments/e29_swa_ep15/seed42.pt --window 100 \
  --preds-out experiments/campaign2/preds/e29_seed42_w100.parquet \
  --json-out experiments/campaign2/preds/e29_seed42_w100.json

# calibrated view (thresholds in the units serving would expose)
uv run python tools/calibration_eval.py \
  --checkpoints experiments/e29_swa_ep15/seed42.pt --window 100 --temperature 0.874

# 5-seed probability-averaged ensemble
uv run python tools/calibration_eval.py \
  --checkpoints "experiments/e29_swa_ep15/seed*.pt" --window 100

# SWA vs no SWA
uv run python tools/calibration_eval.py --checkpoints experiments/e27_strip_nosw/seed42.pt --window 100
uv run python tools/calibration_eval.py --checkpoints experiments/e27_strip_ep15/seed42.pt --window 100
```
